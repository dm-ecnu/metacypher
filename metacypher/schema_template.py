"""
Schema template management and path description building.

Extracted from subgraph_retrieval.py: SchemaTemplateManager loads
per-graph Cypher templates from JSON files; PathDescriptionBuilder
uses those templates to produce natural-language descriptions of
PathInstance objects.
"""

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from retrieval_config import OPERATOR_TEMPLATES, PROPERTY_NAMES

logger = logging.getLogger(__name__)


class SchemaTemplateManager:
    """加载并提供schema模板查询"""

    def __init__(self, base_dir: str):
        import json as _json
        self._json = _json
        self.base_dir = Path(base_dir)
        self.templates: Dict[Tuple[str, str, str], Dict[str, List[Dict[str, Any]]]] = {}
        self.loaded_graph: Optional[str] = None
        self.template_count: int = 0

    def load(self, graph_name: str) -> None:
        """根据图谱名加载模板"""
        if not graph_name:
            raise ValueError("graph_name is required for loading schema templates")

        template_file = self.base_dir / f"{graph_name}.json"
        if not template_file.exists():
            raise FileNotFoundError(f"Schema template file not found: {template_file}")

        with open(template_file, 'r', encoding='utf-8') as f:
            schema_data = self._json.load(f)

        templates: Dict[Tuple[str, str, str], Dict[str, List[Dict[str, Any]]]] = {}
        for rel in schema_data.get('relations', []):
            subj_label = rel.get('subj_label')
            obj_label = rel.get('obj_label')
            rel_type = rel.get('label')
            if not (subj_label and obj_label and rel_type):
                continue
            key = (subj_label, rel_type, obj_label)
            templates[key] = {
                'match': self._collect_template_variants(rel, 'match_template'),
                'optional': self._collect_template_variants(rel, 'optional_match_template')
            }

        self.templates = templates
        self.loaded_graph = graph_name
        self.template_count = len(templates)

    def get_match_template(self, subj_label: str, rel_type: str, obj_label: str) -> Optional[str]:
        """获取MATCH模板（兼容旧接口）"""
        return self.get_best_match_template(subj_label, rel_type, obj_label, available_rel_keys=set())

    def get_optional_template(self, subj_label: str, rel_type: str, obj_label: str) -> Optional[str]:
        """获取OPTIONAL MATCH模板（兼容旧接口）"""
        return self.get_best_optional_template(subj_label, rel_type, obj_label, available_rel_keys=set())

    def get_best_match_template(
        self,
        subj_label: str,
        rel_type: str,
        obj_label: str,
        available_rel_keys: Optional[Set[str]] = None
    ) -> Optional[str]:
        return self._get_best_template(
            (subj_label, rel_type, obj_label),
            'match',
            available_rel_keys or set()
        )

    def get_best_optional_template(
        self,
        subj_label: str,
        rel_type: str,
        obj_label: str,
        available_rel_keys: Optional[Set[str]] = None
    ) -> Optional[str]:
        return self._get_best_template(
            (subj_label, rel_type, obj_label),
            'optional',
            available_rel_keys or set()
        )

    def _collect_template_variants(self, rel: Dict[str, Any], base_field: str) -> List[Dict[str, Any]]:
        variant_fields: List[str] = []
        if base_field in rel:
            variant_fields.append(base_field)
        fallback_prefix = f"{base_field}_fallback"
        for field in sorted(rel.keys()):
            if field.startswith(fallback_prefix):
                variant_fields.append(field)

        variants: List[Dict[str, Any]] = []
        for field in variant_fields:
            template_text = rel.get(field)
            if template_text:
                variants.append({
                    'text': template_text,
                    'required_rel_keys': self._extract_rel_placeholders(template_text)
                })
        return variants

    def _extract_rel_placeholders(self, template: str) -> Set[str]:
        if not template:
            return set()
        placeholders = set(re.findall(r'\{([^}]+)\}', template))
        return {p for p in placeholders if p.startswith('rel_')}

    def _get_best_template(
        self,
        key: Tuple[str, str, str],
        template_type: str,
        available_rel_keys: Set[str]
    ) -> Optional[str]:
        entry = self.templates.get(key)
        if not entry:
            return None
        variants = entry.get(template_type) or []
        for variant in variants:
            required = variant.get('required_rel_keys', set())
            if required.issubset(available_rel_keys):
                return variant.get('text')
        return None


class PathDescriptionBuilder:
    """统一的路径描述生成器"""

    def __init__(
        self,
        template_manager: SchemaTemplateManager,
        constraint_text_fn: Optional[Callable[[Dict[str, Any], Any], str]] = None,
        operator_text_map: Optional[Dict[str, str]] = None,
        property_name_map: Optional[Dict[str, str]] = None
    ):
        self.template_manager = template_manager
        self.constraint_text_fn = constraint_text_fn
        self.operator_text_map = operator_text_map or OPERATOR_TEMPLATES
        self.property_name_map = property_name_map or PROPERTY_NAMES

    def build(
        self,
        path: Any,
        use_optional: bool = False,
        include_constraints: bool = True
    ) -> str:
        """使用模板生成路径描述"""
        sentences: List[str] = []
        for edge in path.edges:
            sentence = self._build_edge_sentence(path, edge, use_optional)
            if sentence:
                sentences.append(sentence)

        if not sentences and path.nodes:
            sentences.append(self._describe_single_node(path.nodes[0]))

        description = " ".join(sentences).strip()

        if include_constraints and getattr(path, 'constraints', None):
            constraint_text = self._build_constraint_text(path)
            if constraint_text:
                description = (
                    f"{description}. Constraints: {constraint_text}"
                    if description
                    else f"Constraints: {constraint_text}"
                )

        if description:
            description = re.sub(r'\s+', ' ', description).strip()
            description = re.sub(r'\.{2,}', '.', description)
            description = description.strip()

        return description

    def _build_edge_sentence(self, path: Any, edge: Dict[str, Any], use_optional: bool) -> str:
        rel_type = edge.get('rel_type', '')
        from_node = self._find_node_by_id(path, edge.get('from'))
        to_node = self._find_node_by_id(path, edge.get('to'))

        if from_node is None or to_node is None:
            logger.warning("[DescriptionBuilder] Missing nodes for edge %s", edge)
            return ""

        subj_label = from_node.get('label', '')
        obj_label = to_node.get('label', '')

        subj_name = self._get_node_display_name(from_node)
        obj_name = self._get_node_display_name(to_node)

        rel_vars, available_rel_keys = self._get_edge_rel_vars(edge)

        entry = self.template_manager.templates.get((subj_label, rel_type, obj_label))
        template = None
        if use_optional:
            template = self.template_manager.get_best_optional_template(
                subj_label,
                rel_type,
                obj_label,
                available_rel_keys=available_rel_keys
            )
        if not template:
            template = self.template_manager.get_best_match_template(
                subj_label,
                rel_type,
                obj_label,
                available_rel_keys=available_rel_keys
            )

        if template:
            values = {
                'subj_name': subj_name,
                'obj_name': obj_name,
                'subject': subj_name,
                'object': obj_name,
                'subj': subj_name,
                'obj': obj_name
            }
            values.update(rel_vars)
            try:
                sentence = template.format(**values)
                return self._clean_template_sentence(sentence)
            except KeyError as exc:
                missing_key = exc.args[0] if exc.args else "unknown"
                logger.warning(
                    "[DescriptionBuilder] Template missing key '%s' for relation (%s, %s, %s)",
                    missing_key, subj_label, rel_type, obj_label
                )
            except Exception as exc:
                logger.warning(
                    "[DescriptionBuilder] Template format failed for relation (%s, %s, %s): %s",
                    subj_label, rel_type, obj_label, exc
                )
        else:
            if not entry:
                logger.warning(
                    "[DescriptionBuilder] Missing template for relation (%s, %s, %s)",
                    subj_label, rel_type, obj_label
                )
        fallback_subj = subj_name or subj_label or "entity"
        fallback_obj = obj_name or obj_label or "entity"
        return self._clean_template_sentence(f"{fallback_subj} -[{rel_type}]-> {fallback_obj}")

    def _clean_template_sentence(self, text: str) -> str:
        cleaned = re.sub(r'\s+', ' ', text).strip()
        cleaned = re.sub(r'\s+([,.;])', r'\1', cleaned)
        cleaned = re.sub(
            r'(named|called)\s+(?=(?:is|are|was|were|flows|flowed|located|situated|in|on|at|,|\.))',
            '',
            cleaned,
            flags=re.IGNORECASE
        )
        cleaned = re.sub(r'(named|called)\s+$', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace(" .", ".").replace(" ,", ",")
        return cleaned

    def _get_edge_rel_vars(self, edge: Dict[str, Any]) -> Tuple[Dict[str, Any], Set[str]]:
        rel_props = edge.get('properties') or {}
        rel_vars: Dict[str, Any] = {}
        available_keys: Set[str] = set()
        for key, value in rel_props.items():
            rel_key = f"rel_{key}"
            rel_vars[rel_key] = value
            available_keys.add(rel_key)
        return rel_vars, available_keys

    def _build_constraint_text(self, path: Any) -> str:
        texts: List[str] = []
        for constraint in path.constraints:
            if self.constraint_text_fn:
                texts.append(self.constraint_text_fn(constraint, path))
            else:
                texts.append(self._fallback_constraint_to_text(constraint, path))
        return "; ".join(texts)

    def _fallback_constraint_to_text(self, constraint: Dict[str, Any], path: Any) -> str:
        prop_key = constraint.get('property', 'property')
        op = constraint.get('operator', 'eq')
        value = constraint.get('value')

        node_hint = self._resolve_constraint_node_hint(constraint, path)

        prop_text = self.property_name_map.get(prop_key, prop_key.replace('_', ' '))
        op_text = self.operator_text_map.get(op, op)

        return f"{node_hint} {prop_text} {op_text} {value}"

    def _resolve_constraint_node_hint(self, constraint: Dict[str, Any], path: Any) -> str:
        node_label = constraint.get('node_label')
        node_id = constraint.get('node_id')
        node_var = constraint.get('node_var')
        node_idx = constraint.get('node_idx')

        target_node = None
        if node_id:
            target_node = self._find_node_by_id(path, node_id)
        elif node_idx is not None and 0 <= node_idx < len(path.nodes):
            target_node = path.nodes[node_idx]
        elif node_var:
            for node in path.nodes:
                if node.get('var') == node_var:
                    target_node = node
                    break

        if target_node:
            label = target_node.get('label')
            name = target_node.get('properties', {}).get('name')
            if name:
                return f"{label} {name}" if label else name
            if label:
                return label

        if node_label:
            return node_label
        if node_var:
            return node_var
        if node_idx is not None:
            return f"node_{node_idx}"
        return "Node"

    def _find_node_by_id(self, path: Any, node_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if node_id is None:
            return None
        for node in path.nodes:
            if node.get('id') == node_id:
                return node
        return None

    def _get_node_display_name(self, node: Dict[str, Any]) -> str:
        props = node.get('properties') or {}
        if 'name' in props and props['name']:
            return props['name']
        display_name = node.get('display_name')
        if display_name:
            return display_name
        return ""

    def _describe_single_node(self, node: Dict[str, Any]) -> str:
        return self._get_node_display_name(node) or node.get('label', 'entity')
