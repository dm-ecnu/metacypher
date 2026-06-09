"""
子图检索系统 - 集成平衡扩展方案

完整功能:
1. ✓ Pattern-based快速检索
2. ✓ 平衡方案的Beam Search扩展（开放性 + 智能引导 + 重复关系）
3. ✓ BGE-M3 embedding
4. ✓ OPTIONAL MATCH扩展
5. ✓ 智能评分机制（related nodes + prior score）

日期: 2025-12-31
版本: 2.0 - Balanced Expansion

NOTE: This module has been refactored; the logic now lives in dedicated
sub-modules.  All previously public symbols are re-exported here for
backward compatibility.
"""

import itertools
import json
import logging
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import TEMPLATE_DIR, EMBED_MODEL_PATH

SCHEMA_TEMPLATE_DIR = str(TEMPLATE_DIR)

# ============================================================================
# Re-exports — keep every previously public symbol importable from here
# ============================================================================

from retrieval_config import (        # noqa: F401
    RetrievalConfig,
    RELATION_TEMPLATES,
    PROPERTY_NAMES,
    OPERATOR_TEMPLATES,
)

from cypher_utils import (            # noqa: F401
    _format_cypher_value,
    _format_cypher_date,
    _parse_numeric_value,
    build_match_clause_and_var_map,
    _extract_match_where,
)

from schema_template import (         # noqa: F401
    SchemaTemplateManager,
    PathDescriptionBuilder,
)

from path_model import (              # noqa: F401
    AttributeMatch,
    PathInstance,
    RepetitionInfo,
    ExpansionCandidate,
    compute_related_schema_coverage,
    build_pattern_structure,
    extract_anchor_nodes_from_path,
)

from beam_search import (             # noqa: F401
    BGEEmbedder,
    RepetitionDetector,
    BalancedCandidateGenerator,
    SemanticFilter,
    OptionalMatchExecutor,
    SemanticGuidedBeamSearch,
)

from path_scorer import PathScorer    # noqa: F401


# ============================================================================
# 完整的子图检索器（集成所有组件）
# ============================================================================

class SubgraphRetriever:
    """
    完整的子图检索器

    集成了平衡扩展方案的Beam Search
    """

    def __init__(self, config: RetrievalConfig, schema: Dict, full_schema_path: str = ''):
        self.config = config
        self.schema = schema
        self.full_schema_path = full_schema_path
        self.relation_templates = {}
        self.template_manager = SchemaTemplateManager(SCHEMA_TEMPLATE_DIR)
        self.description_builder = PathDescriptionBuilder(
            self.template_manager,
            constraint_text_fn=self._constraint_to_text,
            operator_text_map=OPERATOR_TEMPLATES
        )

        self.full_schema = self._load_full_schema() if full_schema_path else None

        if self.full_schema:
            self._load_relation_templates()

        self.relation_index = self._build_relation_index()

        if config.use_beam_search:
            self.embedder = BGEEmbedder(
                model_name=config.embedding_model,
                device=config.embedding_device
            )
        else:
            self.embedder = None

        self.scorer = PathScorer(config)

        self.label_property_index = self._build_label_property_index()

        logger.info("[SubgraphRetriever] 初始化完成（平衡扩展方案）")
        logger.info(f"  • Beam Search: {config.use_beam_search}")
        logger.info(f"  • 扩展策略: {'所有节点' if config.expand_from_all_nodes else '只端点'}")
        logger.info(f"  • 重复关系检测: {config.enable_repetition_detection}")
        if self.full_schema:
            logger.info(f"  • 完整Schema: {len(self.full_schema.get('relations', []))} 个关系")
            logger.info(f"  • 关系模板: {len(self.relation_templates)} 个")
        else:
            logger.info(f"  • 简化Schema: {len(self.schema.get('relationships', []))} 个关系")

    def _load_relation_templates(self):
        """从full_schema中加载关系模板"""
        if not self.full_schema:
            return

        relations = self.full_schema.get('relations', [])
        for rel in relations:
            subj_label = rel.get('subj_label')
            obj_label = rel.get('obj_label')
            rel_type = rel.get('label')
            match_template = rel.get('match_template', '')
            key = (subj_label, rel_type, obj_label)
            self.relation_templates[key] = match_template

        logger.info(f"[Templates] 加载了 {len(self.relation_templates)} 个关系模板")

    def _build_label_property_index(self) -> Dict[str, Set[str]]:
        """从schema/full_schema中收集各label支持的属性名"""
        property_index: Dict[str, Set[str]] = defaultdict(set)
        sources = []
        if self.full_schema:
            sources.append(self.full_schema)
        if self.schema:
            sources.append(self.schema)
        for source in sources:
            entities = source.get('entities', [])
            for entity in entities:
                label = entity.get('label')
                if not label:
                    continue
                props = entity.get('properties') or {}
                if isinstance(props, dict):
                    property_index[label].update(props.keys())
                elif isinstance(props, list):
                    for item in props:
                        if isinstance(item, dict):
                            prop_name = item.get('name')
                            if prop_name:
                                property_index[label].add(prop_name)
        return property_index

    def _is_valid_property(self, label: Optional[str], property_name: Optional[str]) -> bool:
        if not label or not property_name:
            return False
        return property_name in self.label_property_index.get(label, set())

    def _debug_print_path_cypher(self, path, prefix="", path_type="Path"):
        """Debug模式：打印路径的详细信息"""
        if not self.config.debug_print_cypher:
            return

        node_var_map = self._build_node_var_map(path)
        match_parts = []
        for i, node in enumerate(path.nodes):
            label = node.get('label')
            props = node.get('properties', {})
            var = node_var_map.get(node.get('id')) or f"n{i}"

            if props:
                prop_strs = []
                for key, value in props.items():
                    if isinstance(value, str):
                        prop_strs.append(f"{key}: '{value}'")
                    else:
                        prop_strs.append(f"{key}: {value}")
                props_clause = ", ".join(prop_strs)
                match_parts.append(f"({var}:{label} {{{props_clause}}})")
            else:
                match_parts.append(f"({var}:{label})")

            if i < len(path.edges):
                edge = path.edges[i]
                rel_type = edge.get('rel_type')
                from_id = edge.get('from')
                to_id = edge.get('to')
                current_node_id = node.get('id')
                next_node_id = path.nodes[i + 1].get('id') if i + 1 < len(path.nodes) else None

                if current_node_id == from_id and next_node_id == to_id:
                    match_parts.append(f"-[:{rel_type}]->")
                elif current_node_id == to_id and next_node_id == from_id:
                    match_parts.append(f"<-[:{rel_type}]-")
                else:
                    match_parts.append(f"-[:{rel_type}]-")

        match_clause = "MATCH " + "".join(match_parts)
        where_clause = self._build_where_clause_for_path(path)

        cypher_lines = [match_clause]
        if where_clause:
            cypher_lines.append(where_clause)
        cypher_lines.append("RETURN *")

        nl_description = self._generate_path_description(path)

        triples = []
        for i, edge in enumerate(path.edges):
            from_node = None
            to_node = None
            for node in path.nodes:
                if node.get('id') == edge.get('from'):
                    from_node = node
                if node.get('id') == edge.get('to'):
                    to_node = node

            if from_node and to_node:
                from_name = self._get_node_name(from_node)
                to_name = self._get_node_name(to_node)
                from_display = from_name if from_name else from_node['label']
                to_display = to_name if to_name else to_node['label']
                triples.append(f"({from_display}, {edge.get('rel_type')}, {to_display})")

        logger.info(f"\n{prefix}{'='*80}")
        logger.info(f"{prefix}[{path_type}] Path #{path.path_id}")
        logger.info(f"{prefix}{'='*80}")
        logger.info(f"{prefix}Pattern: {path.pattern_id} | Score: {path.final_score:.4f} | Anchors: {path.anchor_count}")
        logger.info(f"\n{prefix}Cypher Query:")
        logger.info(f"{prefix}{'─'*80}")
        for line in cypher_lines:
            logger.info(f"{prefix}  {line}")
        logger.info(f"{prefix}{'─'*80}")
        logger.info(f"\n{prefix}Natural Language Description:")
        logger.info(f"{prefix}{'─'*80}")
        logger.info(f"{prefix}  {nl_description}")
        logger.info(f"{prefix}{'─'*80}")
        logger.info(f"\n{prefix}Triple Form:")
        logger.info(f"{prefix}{'─'*80}")
        for i, triple in enumerate(triples, 1):
            logger.info(f"{prefix}  [{i}] {triple}")
        logger.info(f"{prefix}{'─'*80}")
        logger.info(f"\n{prefix}Nodes Detail:")
        for i, node in enumerate(path.nodes):
            props = node.get('properties', {})
            if props:
                props_str = ", ".join([f"{k}={v}" for k, v in props.items()])
                logger.info(f"{prefix}  [n{i}] {node['label']} {{ {props_str} }}")
            else:
                logger.info(f"{prefix}  [n{i}] {node['label']}")

        if path.constraints:
            logger.info(f"\n{prefix}Constraints:")
            for c in path.constraints:
                node_label = c.get('node_label')
                if not node_label:
                    node = self._find_node_by_id(path, c.get('node_id'))
                    node_label = node.get('label') if node else 'Node'
                prop = c.get('property')
                op = c.get('operator', 'eq')
                value = c.get('value')
                logger.info(f"{prefix}  • {node_label}.{prop} {op} {value}")

        logger.info(f"{prefix}{'='*80}\n")

    def _build_where_clause_for_path(self, path: 'PathInstance') -> str:
        """为debug打印构建WHERE子句"""
        if not path.constraints:
            return ""

        node_var_map = self._build_node_var_map(path)
        operator_map = {
            'lt': '<', 'le': '<=', 'gt': '>',
            'ge': '>=', 'eq': '=', 'ne': '!='
        }

        conditions = []
        for c in path.constraints:
            node_var = c.get('node_var')
            if not node_var:
                node_id = c.get('node_id')
                if node_id and node_var_map.get(node_id):
                    node_var = node_var_map[node_id]
            if not node_var:
                node_idx = c.get('node_idx', 0)
                node_var = f"n{node_idx}"
            prop = c.get('property')
            op = c.get('operator', 'eq')
            value = c.get('value')
            value_type = c.get('value_type', 'int')

            if value_type == 'date':
                cypher_value = _format_cypher_date(value)
                if cypher_value is None:
                    continue
                op_symbol = operator_map.get(op, '=')
                if c.get('allow_null'):
                    conditions.append(
                        f"({node_var}.{prop} {op_symbol} {cypher_value} "
                        f"OR {node_var}.{prop} IS NULL)"
                    )
                else:
                    conditions.append(f"{node_var}.{prop} {op_symbol} {cypher_value}")

            elif value_type == 'list(str)':
                conditions.append(f"{_format_cypher_value(value)} IN {node_var}.{prop}")

            elif value_type in ['int', 'float']:
                op_symbol = operator_map.get(op, '=')
                parsed_value = _parse_numeric_value(value, allow_float=(value_type == 'float'))
                if parsed_value is None:
                    continue
                conditions.append(f"{node_var}.{prop} {op_symbol} {parsed_value}")

            elif value_type == 'str':
                op_symbol = operator_map.get(op, '=')
                conditions.append(f"{node_var}.{prop} {op_symbol} {_format_cypher_value(value)}")

        return "WHERE " + " AND ".join(conditions) if conditions else ""

    def _build_node_var_map(self, path: 'PathInstance') -> Dict[str, str]:
        """构建node_id到Cypher变量名的映射"""
        mapping = {}
        for i, node in enumerate(path.nodes):
            node_id = node.get('id')
            if node_id is not None:
                mapping[node_id] = f"n{i}"
        return mapping

    def _find_node_by_id(self, path: 'PathInstance', node_id: Optional[str]) -> Optional[Dict]:
        if node_id is None:
            return None
        for node in path.nodes:
            if node.get('id') == node_id:
                return node
        return None

    def _relation_to_text(self, rel_type: str) -> str:
        if rel_type in RELATION_TEMPLATES:
            return RELATION_TEMPLATES[rel_type]
        words = re.findall('[A-Z][a-z]*|[a-z]+', rel_type)
        return ' '.join(words).lower()

    def _property_to_text(self, prop_name: str) -> str:
        if prop_name in PROPERTY_NAMES:
            return PROPERTY_NAMES[prop_name]
        return prop_name.replace('_', ' ')

    def _constraint_to_text(self, constraint: Dict, path: 'PathInstance') -> str:
        prop = constraint.get('property', 'name')
        op = constraint.get('operator', 'eq')
        value = constraint.get('value')
        value_type = constraint.get('value_type', 'int')

        prop_text = self._property_to_text(prop)
        op_text = OPERATOR_TEMPLATES.get(op, op)

        if value_type == 'date':
            value_text = f"{value}"
        elif value_type == 'list(str)':
            value_text = f"'{value}'"
        elif value_type == 'int':
            value_text = f"{value:,}" if isinstance(value, int) else str(value)
        else:
            value_text = str(value)

        node_hint = constraint.get('node_label')
        if not node_hint:
            node = self._find_node_by_id(path, constraint.get('node_id'))
            if node and node.get('label'):
                node_hint = node.get('label')
        if not node_hint and constraint.get('node_var'):
            node_hint = constraint.get('node_var')
        if not node_hint:
            idx = constraint.get('node_idx')
            node_hint = f"node_{idx}" if idx is not None else "Node"

        return f"{node_hint} {prop_text} {op_text} {value_text}"

    def _generate_path_description(self, path):
        if not hasattr(self, 'description_builder') or self.description_builder is None:
            return ""
        return self.description_builder.build(path, use_optional=False, include_constraints=True)

    def _ensure_path_description(self, path: PathInstance) -> str:
        if path.description:
            return path.description
        if hasattr(self, 'description_builder') and self.description_builder:
            description = self.description_builder.build(
                path, use_optional=False, include_constraints=True
            )
            if description:
                path.description = description
                return description
        description = self._generate_path_description_fallback(path)
        path.description = description
        return description

    def _ensure_paths_have_descriptions(
        self,
        paths: List[PathInstance],
        pool: Optional[Any] = None,
        db_name: str = ""
    ) -> List[PathInstance]:
        for path in paths:
            if pool is not None and db_name:
                self._hydrate_path_with_db_samples(path, pool, db_name)
            self._ensure_path_description(path)
            self._finalize_path_instance(path)
        return paths

    def _mark_node_as_anchor(self, node: Dict[str, Any], source_info: Optional[Dict[str, Any]]) -> None:
        node['is_anchor'] = True
        if source_info:
            node_sources = node.setdefault('anchor_sources', [])
            node_sources.append(source_info)

    def _build_cypher_query_for_path(self, path: PathInstance) -> str:
        match_clause, _ = build_match_clause_and_var_map(path)
        where_clause = self._build_where_clause_for_path(path)
        limit = (
            getattr(self.config, "max_results_per_query", None)
            or getattr(self.config, "max_cypher_limit", None)
            or 50
        )
        query_parts = [match_clause]
        if where_clause:
            query_parts.append(where_clause)
        query_parts.append(f"RETURN * LIMIT {limit}")
        return "\n".join(query_parts)

    def _finalize_path_instance(self, path: PathInstance) -> PathInstance:
        path.pattern_structure = build_pattern_structure(path)
        path.anchor_nodes = extract_anchor_nodes_from_path(path)
        path.anchor_count = len(path.anchor_nodes)
        query_text = (path.cypher_query or "").strip()
        if not query_text or "RETURN" not in query_text.upper():
            path.cypher_query = self._build_cypher_query_for_path(path)
            query_text = path.cypher_query
        match_where = _extract_match_where(query_text)
        path.pattern_signature = match_where or path.get_signature()
        if not path.sample_paths:
            path.sample_paths = [{'nodes': deepcopy(path.nodes), 'edges': deepcopy(path.edges)}]
        if not path.sample_paths_total_hits:
            path.sample_paths_total_hits = len(path.sample_paths)
        return path

    def _run_query(self, pool: Any, db_name: str, query: str) -> List[Dict[str, Any]]:
        try:
            if hasattr(pool, 'execute'):
                result = pool.execute(db_name, query)
                return [dict(record) for record in result]
            with pool.get_connection(db_name) as conn:
                with conn.session() as session:
                    result = session.run(query)
                    return [dict(record) for record in result]
        except Exception as exc:
            logger.warning(f"[Hydrate] Query execution failed: {exc}")
            return []

    def _graph_value_to_dict(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        try:
            return dict(value)
        except Exception:
            return {}

    def _graph_value_label(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            labels = list(value.labels)
            if labels:
                return labels[0]
        except Exception:
            pass
        return None

    def _hydrate_path_with_db_samples(
        self,
        path: PathInstance,
        pool: Any,
        db_name: str
    ) -> None:
        query = path.cypher_query or self._build_cypher_query_for_path(path)
        records = self._run_query(pool, db_name, query)
        logger.debug(
            "[Hydrate] Path %s returned %d rows for graph=%s",
            path.path_id, len(records), db_name
        )
        if not records:
            logger.debug("[Hydrate] Path %s has no DB rows; keep pattern-only info", path.path_id)
            return
        max_samples = getattr(self.config, "max_sample_paths_per_triple", 5)
        sample_entries: List[Dict[str, Any]] = []
        seen_signatures: Set[Tuple[Any, ...]] = set()
        for idx, record in enumerate(records):
            nodes_snapshot: List[Dict[str, Any]] = []
            edges_snapshot: List[Dict[str, Any]] = []
            for i, node in enumerate(path.nodes):
                alias = f"n{i}"
                node_value = record.get(alias)
                node_copy = deepcopy(node)
                if node_value is not None:
                    props = self._graph_value_to_dict(node_value)
                    if props:
                        node_copy['properties'] = props
                        if 'name' in props:
                            node_copy['display_name'] = props['name']
                    label = self._graph_value_label(node_value)
                    if label:
                        node_copy['label'] = label
                nodes_snapshot.append(node_copy)
            for i, edge in enumerate(path.edges):
                alias = f"r{i}"
                rel_value = record.get(alias)
                edge_copy = deepcopy(edge)
                if rel_value is not None:
                    rel_props = self._graph_value_to_dict(rel_value)
                    if rel_props:
                        edge_copy['properties'] = rel_props
                edges_snapshot.append(edge_copy)
            signature = tuple(
                (node.get('id'), node.get('properties', {}).get('name'))
                for node in nodes_snapshot
            )
            if signature not in seen_signatures and len(sample_entries) < max_samples:
                sample_entries.append({'nodes': nodes_snapshot, 'edges': edges_snapshot})
                seen_signatures.add(signature)
        if not sample_entries:
            logger.debug("[Hydrate] Path %s samples fallback to pattern structure only", path.path_id)
        path.sample_paths = sample_entries or [{'nodes': deepcopy(path.nodes), 'edges': deepcopy(path.edges)}]
        path.sample_paths_total_hits = len(records)

    def _get_node_name(self, node):
        if node.get('properties'):
            if 'name' in node['properties']:
                return node['properties']['name']
            values = list(node['properties'].values())
            if values:
                return values[0]
        if 'display_name' in node:
            return node['display_name']
        return None

    def _generate_path_description_fallback(self, path: 'PathInstance') -> str:
        parts = []
        for i, node in enumerate(path.nodes):
            if node.get('properties'):
                prop_strs = []
                for key, val in node['properties'].items():
                    prop_key = self._property_to_text(key)
                    prop_strs.append(f"{prop_key} '{val}'")
                node_desc = f"{node['label']} with {' and '.join(prop_strs)}"
            else:
                node_desc = node['label']

            parts.append(node_desc)

            if i < len(path.edges):
                rel = path.edges[i]
                rel_desc = self._relation_to_text(rel.get('rel_type', ''))
                parts.append(rel_desc)

        main_description = " ".join(parts)

        if path.constraints:
            constraint_parts = []
            for c in path.constraints:
                node_label = c.get('node_label')
                if not node_label:
                    node = self._find_node_by_id(path, c.get('node_id'))
                    node_label = node.get('label') if node else 'Node'
                constraint_desc = self._constraint_to_text(c, path)
                constraint_parts.append(f"{node_label} has {constraint_desc}")
            constraint_description = ", ".join(constraint_parts)
            return f"{main_description}, where {constraint_description}"
        else:
            return main_description

    def _compute_semantic_similarity(self, path_description: str, question: str) -> float:
        if not self.embedder:
            return 0.5

        try:
            desc_emb = self.embedder.encode_single(path_description)
            question_emb = self.embedder.encode_single(question)

            dot_product = np.dot(desc_emb, question_emb)
            norm_a = np.linalg.norm(desc_emb)
            norm_b = np.linalg.norm(question_emb)

            similarity = dot_product / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0.0
            similarity = (similarity + 1) / 2
            return float(similarity)

        except Exception as e:
            logger.warning(f"[Semantic] 相似度计算失败: {e}")
            return 0.5

    def retrieve(self, pool, db_name: str, case: Dict) -> Dict:
        """
        完整检索流程

        Args:
            pool: Neo4j连接池
            db_name: 数据库名
            case: {'question': str, 'question_analysis': Dict}

        Returns:
            {'pattern_paths', 'beam_paths', 'final_paths', 'stats'}
        """
        question = case.get('nl_question', '')
        question_analysis = case.get('analysis', {})

        logger.info(f"\n{'='*80}")
        logger.info(f"[SubgraphRetriever] 检索问题: {question}")
        logger.info(f"{'='*80}\n")

        graph_name = case.get('graph') or db_name
        self.template_manager.load(graph_name)
        logger.info(
            f"[Templates] Loaded {self.template_manager.template_count} templates for graph={graph_name}"
        )

        all_paths: List[PathInstance] = []
        stats: Dict[str, Any] = {}

        # ========== Stage 1: Pattern-Based检索 ==========
        pattern_paths: List[PathInstance] = []
        if hasattr(self.config, 'use_pattern_based') and self.config.use_pattern_based:
            logger.info("[Stage 1] Pattern-Based检索...")
            pattern_paths = self._pattern_based_retrieve(pool, db_name, case)
            all_paths.extend(pattern_paths)
            stats['pattern_paths_count'] = len(pattern_paths)
            logger.info(f"[Stage 1] 完成: {len(pattern_paths)} 条路径\n")
        else:
            logger.info("[Stage 1] 跳过Pattern-Based（使用mock初始路径）")
            pattern_paths = self._create_mock_initial_paths(question_analysis)
            all_paths.extend(pattern_paths)
            stats['pattern_paths_count'] = len(pattern_paths)

        # ========== Stage 2: Beam Search扩展 ==========
        beam_paths: List[PathInstance] = []
        if self.config.use_beam_search and pattern_paths and self.embedder:
            logger.info("[Stage 2] Beam Search扩展（平衡方案）...")

            beam_search = SemanticGuidedBeamSearch(
                config=self.config,
                schema=self.schema,
                question=question,
                question_analysis=question_analysis,
                embedder=self.embedder,
                path_scorer=self.scorer,
                relation_index=self.relation_index,
                description_builder=self.description_builder
            )

            beam_paths = beam_search.expand(pattern_paths, pool, db_name)
            all_paths.extend(beam_paths)
            stats['beam_paths_count'] = len(beam_paths)
            logger.info(f"[Stage 2] 完成: {len(beam_paths)} 条扩展路径\n")
        else:
            logger.info(
                "[Stage 2] 跳过Beam Search (use_beam_search=%s, pattern_paths=%d, embedder_ready=%s)\n",
                self.config.use_beam_search, len(pattern_paths), bool(self.embedder)
            )

        # ========== Final: 选择top-k路径 ==========
        logger.info("[Final] 选择最终路径...")
        final_paths = self.scorer.select_top_k_paths(all_paths, k=self.config.top_k_paths)
        final_paths = self._ensure_paths_have_descriptions(final_paths, pool=pool, db_name=db_name)
        stats['final_paths_count'] = len(final_paths)
        stats['total_paths_count'] = len(all_paths)

        logger.info(f"[Final] 完成: {len(final_paths)} 条最终路径")
        logger.info(f"\n{'='*80}")
        logger.info(f"[SubgraphRetriever] 检索完成")
        logger.info(f"  • Pattern路径: {stats['pattern_paths_count']}")
        logger.info(f"  • Beam扩展路径: {stats.get('beam_paths_count', 0)}")
        logger.info(f"  • 最终路径: {stats['final_paths_count']}")
        logger.info(f"{'='*80}\n")

        return {
            'pattern_paths': pattern_paths,
            'beam_paths': beam_paths,
            'final_paths': final_paths,
            'stats': stats
        }

    def _load_full_schema(self) -> Optional[Dict]:
        """加载完整的schema文件"""
        try:
            schema_file = Path(self.full_schema_path)
            if not schema_file.exists():
                logger.warning(f"[FullSchema] 文件不存在: {self.full_schema_path}")
                return None

            with open(schema_file, 'r', encoding='utf-8') as f:
                full_schema = json.load(f)

            logger.info(f"[FullSchema] 加载成功: {schema_file.name}")
            logger.info(f"  • 实体类型: {len(full_schema.get('entities', []))}")
            logger.info(f"  • 关系类型: {len(full_schema.get('relations', []))}")

            return full_schema

        except Exception as e:
            logger.error(f"[FullSchema] 加载失败: {e}")
            return None

    def _build_relation_index(self) -> Dict:
        """构建关系索引用于快速查找"""
        index: Dict = defaultdict(list)

        if self.full_schema:
            relations = self.full_schema.get('relations', [])
            for rel in relations:
                subj_label = rel.get('subj_label')
                obj_label = rel.get('obj_label')
                rel_type = rel.get('label')

                if subj_label and obj_label and rel_type:
                    index[(subj_label, 'out')].append((rel_type, obj_label))
                    index[(obj_label, 'in')].append((rel_type, subj_label))
        else:
            relationships = self.schema.get('relationships', [])
            for rel in relationships:
                from_label = rel.get('from')
                to_label = rel.get('to')
                rel_type = rel.get('type')

                if from_label and to_label and rel_type:
                    index[(from_label, 'out')].append((rel_type, to_label))
                    index[(to_label, 'in')].append((rel_type, from_label))

        logger.info(f"[RelationIndex] 构建完成，索引数: {len(index)}")
        return dict(index)

    def _pattern_based_retrieve(self, pool, db_name: str, case: Dict) -> List[PathInstance]:
        """Pattern-Based检索"""
        retrieved_patterns_data = case.get('retrieved_patterns', {})

        if isinstance(retrieved_patterns_data, dict):
            retrieved_patterns = retrieved_patterns_data.get('patterns', [])
        elif isinstance(retrieved_patterns_data, list):
            retrieved_patterns = retrieved_patterns_data
        else:
            retrieved_patterns = []

        if retrieved_patterns:
            logger.info(f"[Pattern-Based] 从retrieved_patterns创建初始路径")
            initial_paths = self._create_paths_from_patterns(retrieved_patterns, case)
            if initial_paths:
                logger.info(f"[Pattern-Based] 从patterns创建了 {len(initial_paths)} 条初始路径")

                if self.config.debug_print_cypher:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"[DEBUG] 初始路径的Cypher查询")
                    logger.info(f"{'='*80}")
                    for i, path in enumerate(initial_paths[:self.config.debug_max_results_display], 1):
                        self._debug_print_path_cypher(path, prefix=f"  [{i}] ", path_type="Initial")

                return initial_paths

        analysis = case.get('analysis', {})
        entities = analysis.get('entities', [])

        if entities:
            logger.info(f"[Pattern-Based] 从analysis.entities创建初始路径")
            initial_paths = self._create_paths_from_entities(entities, analysis)
            if initial_paths:
                logger.info(f"[Pattern-Based] 从entities创建了 {len(initial_paths)} 条初始路径")
                return initial_paths

        logger.info("[Pattern-Based] 使用schema创建基础初始路径")
        relationships = self.schema.get('relationships', [])

        if relationships:
            initial_paths = []
            for i, rel in enumerate(relationships[:5]):
                from_label = rel.get('from', 'Node')
                to_label = rel.get('to', 'Node')
                rel_type = rel.get('type', 'RELATES_TO')

                path = PathInstance(
                    path_id=f"schema_path_{i}",
                    nodes=[
                        {'label': from_label, 'id': f'{from_label.lower()}_{i}', 'properties': {}},
                        {'label': to_label, 'id': f'{to_label.lower()}_{i}', 'properties': {}}
                    ],
                    edges=[
                        {'rel_type': rel_type, 'from': f'{from_label.lower()}_{i}', 'to': f'{to_label.lower()}_{i}'}
                    ],
                    pattern_id='from_schema',
                    pattern_score=0.3,
                    anchor_count=0,
                    final_score=0.3
                )
                path.cypher_query = f"MATCH {path.get_signature()}"
                path = self._finalize_path_instance(path)
                initial_paths.append(path)

            logger.info(f"[Pattern-Based] 从schema创建了 {len(initial_paths)} 条初始路径")
            return initial_paths

        logger.warning("[Pattern-Based] 无法创建初始路径")
        return []

    def _create_constraint(
        self,
        entity: Dict,
        mapping: Dict,
        node_id: Optional[str],
        node_var: Optional[str],
        node_label: Optional[str]
    ) -> Optional[Dict]:
        value_type = entity.get('value_type')
        property_name = mapping.get('property')
        comparison = entity.get('comparison', {})
        operator = comparison.get('operator', 'eq')
        value = entity.get('value')

        if not property_name:
            logger.warning(
                "[Constraint] Missing property mapping for entity mention '%s'; skipping",
                entity.get('mention')
            )
            return None

        if node_label and not self._is_valid_property(node_label, property_name):
            logger.warning(
                "[Constraint] Property %s.%s not defined in schema; skipping constraint",
                node_label, property_name
            )
            return None

        constraint = {
            'node_id': node_id,
            'node_var': node_var or node_id,
            'node_label': node_label,
            'property': property_name,
            'value_type': value_type,
            'confidence': mapping.get('confidence', 0.5)
        }

        if value_type == 'date':
            formatted = _format_cypher_date(value)
            if formatted is None:
                logger.warning(
                    "[Constraint] Skipping date constraint for value '%s' (unparseable)", value
                )
                return None
            constraint['operator'] = operator
            constraint['value'] = value
            constraint['cypher_fn'] = 'date'

            assert property_name is not None, "Property name cannot be None for date constraints"
            if 'end' in property_name.lower() and operator in ['gt', 'ge']:
                constraint['allow_null'] = True

        elif value_type == 'list(str)':
            constraint['operator'] = 'in'
            constraint['value'] = value

        elif value_type in ['int', 'float']:
            constraint['operator'] = operator
            constraint['value'] = value

        elif value_type == 'str':
            constraint['operator'] = 'eq'
            constraint['value'] = value

        return constraint

    def _parse_cypher_pattern(self, cypher_pattern):
        """解析Cypher pattern，提取节点、边和方向"""
        nodes = []
        edges = []

        node_pattern = r'\((\w+):(\w+)(?:\s*\{[^}]*\})?\)'
        node_matches = re.findall(node_pattern, cypher_pattern)

        for var, label in node_matches:
            nodes.append({
                'label': label,
                'var': var,
                'id': var,
                'properties': {}
            })

        parts = re.split(r'(\([^)]+\))', cypher_pattern)

        for i, part in enumerate(parts):
            if not part or part.startswith('('):
                continue

            if '-[' in part or '<-[' in part:
                rel_match = re.search(r'\[\s*\w+\s*:(\w+)\]', part)
                if not rel_match:
                    continue

                rel_type = rel_match.group(1)

                direction = 'undirected'
                has_forward = '->' in part
                has_backward = '<-' in part
                if has_forward and not has_backward:
                    direction = 'forward'
                elif has_backward and not has_forward:
                    direction = 'backward'

                from_var = None
                to_var = None

                for j in range(i - 1, -1, -1):
                    if parts[j].startswith('('):
                        match = re.search(r'\((\w+):', parts[j])
                        if match:
                            from_var = match.group(1)
                            break

                for j in range(i + 1, len(parts)):
                    if parts[j].startswith('('):
                        match = re.search(r'\((\w+):', parts[j])
                        if match:
                            to_var = match.group(1)
                            break

                if from_var and to_var:
                    actual_from = from_var
                    actual_to = to_var

                    if direction == 'backward':
                        actual_from, actual_to = to_var, from_var

                    edges.append({
                        'rel_type': rel_type,
                        'from_var': actual_from,
                        'to_var': actual_to,
                        'from': actual_from,
                        'to': actual_to,
                        'direction': direction
                    })

        return nodes, edges

    def _create_paths_from_patterns(
        self,
        retrieved_patterns: List[Dict],
        case: Dict
    ) -> List[PathInstance]:
        """从retrieved_patterns创建PathInstance"""
        all_paths: List[PathInstance] = []
        entities = case.get('analysis', {}).get('entities', [])
        question = case.get('nl_question', '')

        top_n_string_candidates = 5
        min_confidence_threshold = 0.3

        for pattern_info in retrieved_patterns[:self.config.top_k_paths]:
            pattern_id = pattern_info.get('pattern_id', 'unknown')
            cypher_pattern = pattern_info.get('cypher_pattern', '')
            scores = pattern_info.get('scores', {})

            if not cypher_pattern:
                continue

            pattern_nodes, pattern_edges = self._parse_cypher_pattern(cypher_pattern)

            if not pattern_nodes or not pattern_edges:
                logger.warning(f"[Pattern {pattern_id}] 解析失败: {cypher_pattern}")
                continue

            string_options: Dict[int, List] = {}
            constraint_options: Dict[str, List] = {}

            for i, pnode in enumerate(pattern_nodes):
                label = pnode['label']
                for entity in entities:
                    if entity.get('value_type') == 'str':
                        mapped_to = entity.get('mapped_to', [])
                        for mapping in mapped_to:
                            if mapping.get('label') == label:
                                faiss_cands = entity.get('faiss_candidates', [])
                                if faiss_cands:
                                    top_cands = faiss_cands[:top_n_string_candidates]
                                    string_options[i] = [
                                        {
                                            'attribute_name': cand.get('attribute_name', 'name'),
                                            'value': cand.get('value'),
                                            'distance': cand.get('distance', 1.0),
                                            'entity': entity
                                        }
                                        for cand in top_cands
                                    ]
                                break

            for entity in entities:
                value_type = entity.get('value_type')

                if value_type in ['int', 'float', 'date', 'list(str)']:
                    mention = entity.get('mention', '')
                    constraint_options[mention] = []

                    mapped_to = entity.get('mapped_to', [])
                    for mapping in mapped_to:
                        if mapping.get('confidence', 0) < min_confidence_threshold:
                            continue

                        for i, pnode in enumerate(pattern_nodes):
                            if mapping.get('label') == pnode['label']:
                                constraint_options[mention].append({
                                    'node_id': pnode.get('id'),
                                    'node_var': pnode.get('var'),
                                    'label': pnode['label'],
                                    'property': mapping.get('property'),
                                    'confidence': mapping.get('confidence', 0.5),
                                    'entity': entity
                                })

            string_combinations: List[Dict] = []
            if string_options:
                indices = sorted(string_options.keys())
                options_lists = [string_options[idx] for idx in indices]
                for combo in itertools.product(*options_lists):
                    string_combinations.append({indices[i]: combo[i] for i in range(len(indices))})
            else:
                string_combinations = [{}]

            constraint_combinations: List[Dict] = []
            if constraint_options:
                mentions = sorted(constraint_options.keys())
                options_lists = [constraint_options[m] for m in mentions]
                for combo in itertools.product(*options_lists):
                    constraint_combinations.append({mentions[i]: combo[i] for i in range(len(mentions))})
            else:
                constraint_combinations = [{}]

            combo_count = 0
            for string_combo in string_combinations:
                for constraint_combo in constraint_combinations:

                    nodes = []
                    for i, pnode in enumerate(pattern_nodes):
                        node = {
                            'label': pnode['label'],
                            'id': pnode.get('id') or pnode.get('var') or f"{pnode['label'].lower()}_{i}",
                            'properties': {},
                            'var': pnode.get('var') or pnode.get('id')
                        }

                        if i in string_combo:
                            cand = string_combo[i]
                            node['properties'] = {cand['attribute_name']: cand['value']}
                            entity_info = cand.get('entity', {})
                            self._mark_node_as_anchor(node, {
                                'source': 'string_candidate',
                                'mention': entity_info.get('mention'),
                                'value': cand.get('value'),
                                'attribute': cand.get('attribute_name'),
                                'entity_label': entity_info.get('label')
                            })

                        nodes.append(node)

                    constraints = []
                    for mention, candidate in constraint_combo.items():
                        entity = candidate['entity']
                        mapping = {
                            'property': candidate['property'],
                            'confidence': candidate['confidence']
                        }

                        constraint = self._create_constraint(
                            entity=entity, mapping=mapping,
                            node_id=candidate.get('node_id'),
                            node_var=candidate.get('node_var'),
                            node_label=candidate.get('label')
                        )
                        if not constraint:
                            continue
                        constraints.append(constraint)

                        target_node_id = candidate.get('node_id')
                        if target_node_id:
                            for node in nodes:
                                if node.get('id') == target_node_id:
                                    self._mark_node_as_anchor(node, {
                                        'source': 'constraint_candidate',
                                        'mention': entity.get('mention'),
                                        'value': entity.get('value'),
                                        'property': candidate.get('property'),
                                        'entity_label': entity.get('label')
                                    })
                                    break

                    edges = []
                    for pedge in pattern_edges:
                        edges.append({
                            'rel_type': pedge['rel_type'],
                            'from': pedge.get('from'),
                            'to': pedge.get('to'),
                            'from_var': pedge.get('from_var'),
                            'to_var': pedge.get('to_var'),
                            'direction': pedge.get('direction')
                        })

                    combo_score = scores.get('hybrid', 0.5)

                    distance_penalty = 0.0
                    for node_idx, cand in string_combo.items():
                        distance = cand.get('distance', 0)
                        distance_penalty += distance * 0.1

                    avg_confidence = 1.0
                    if constraint_combo:
                        avg_conf = sum(c['confidence'] for c in constraint_combo.values()) / len(constraint_combo)
                        avg_confidence = avg_conf

                    combo_count += 1
                    path = PathInstance(
                        path_id=f"pattern_{pattern_id}_combo_{combo_count}",
                        nodes=nodes,
                        edges=edges,
                        pattern_id=pattern_id,
                        pattern_score=scores.get('hybrid', 0.5),
                        anchor_count=sum(1 for n in nodes if n.get('properties')),
                        constraints=constraints,
                        final_score=0.0
                    )

                    path_description = self._generate_path_description(path)
                    semantic_sim = self._compute_semantic_similarity(path_description, question)

                    final_score = (
                        combo_score * 0.25 +
                        (1 - min(distance_penalty, 1.0)) * 0.20 +
                        avg_confidence * 0.25 +
                        semantic_sim * 0.30
                    )

                    path.semantic_similarity = semantic_sim
                    path.cypher_query = cypher_pattern
                    path.final_score = final_score
                    path = self._finalize_path_instance(path)

                    all_paths.append(path)

            logger.info(f"[Pattern {pattern_id}] 生成 {combo_count} 个组合")

        logger.info(f"[Combinations] 总共生成 {len(all_paths)} 条初始路径")
        return all_paths

    def _create_paths_from_entities(
        self,
        entities: List[Dict],
        analysis: Dict
    ) -> List[PathInstance]:
        """从analysis.entities创建PathInstance"""
        paths: List[PathInstance] = []

        related_schema = analysis.get('related_schema', {})
        relation_candidates = related_schema.get('relation_candidates', [])

        if not relation_candidates:
            return []

        for i, rel_cand in enumerate(relation_candidates[:5]):
            subj_label = rel_cand.get('subj_label', 'Entity')
            obj_label = rel_cand.get('obj_label', 'Entity')
            rel_type = rel_cand.get('type', 'RELATES_TO')
            confidence = rel_cand.get('confidence', 0.5)

            subj_node: Dict[str, Any] = {'label': subj_label, 'id': f'{subj_label.lower()}_{i}', 'properties': {}}
            obj_node: Dict[str, Any] = {'label': obj_label, 'id': f'{obj_label.lower()}_{i}', 'properties': {}}

            for entity in entities:
                mapped_to = entity.get('mapped_to', [])
                faiss_candidates = entity.get('faiss_candidates', [])

                for mapping in mapped_to:
                    if mapping.get('label') == subj_label and faiss_candidates:
                        best = faiss_candidates[0]
                        subj_node['id'] = best.get('node_id', subj_node['id'])
                        subj_node['properties'] = {
                            best.get('attribute_name', 'name'): best.get('value', '')
                        }
                        self._mark_node_as_anchor(subj_node, {
                            'source': 'entity_candidate',
                            'mention': entity.get('mention'),
                            'value': best.get('value'),
                            'attribute': best.get('attribute_name', 'name'),
                            'entity_label': mapping.get('label')
                        })
                    elif mapping.get('label') == obj_label and faiss_candidates:
                        best = faiss_candidates[0]
                        obj_node['id'] = best.get('node_id', obj_node['id'])
                        obj_node['properties'] = {
                            best.get('attribute_name', 'name'): best.get('value', '')
                        }
                        self._mark_node_as_anchor(obj_node, {
                            'source': 'entity_candidate',
                            'mention': entity.get('mention'),
                            'value': best.get('value'),
                            'attribute': best.get('attribute_name', 'name'),
                            'entity_label': mapping.get('label')
                        })

            path = PathInstance(
                path_id=f"entity_rel_{i}",
                nodes=[subj_node, obj_node],
                edges=[{
                    'rel_type': rel_type,
                    'from': subj_node['id'],
                    'to': obj_node['id']
                }],
                pattern_id='from_entities',
                pattern_score=confidence,
                anchor_count=sum(1 for n in [subj_node, obj_node] if n.get('properties')),
                final_score=confidence
            )
            path.cypher_query = f"MATCH {path.get_signature()}"
            path = self._finalize_path_instance(path)
            paths.append(path)

        return paths

    def _create_mock_initial_paths(self, question_analysis: Dict) -> List[PathInstance]:
        """创建mock初始路径用于测试"""
        entities = question_analysis.get('entities', {})
        anchor_candidates = entities.get('anchor_candidates', [])
        faiss_candidates = entities.get('faiss_candidates', [])

        mock_paths: List[PathInstance] = []

        if anchor_candidates and faiss_candidates:
            anchor_label = anchor_candidates[0].get('label', 'Entity')
            target_label = faiss_candidates[0].get('label', 'Entity')

            path = PathInstance(
                path_id='mock_path_1',
                nodes=[
                    {'label': anchor_label, 'id': 'mock_1'},
                    {'label': target_label, 'id': 'mock_2'}
                ],
                edges=[{'rel_type': 'RELATES_TO'}],
                anchor_count=1
            )
            path.cypher_query = f"MATCH {path.get_signature()}"
            mock_paths.append(path)

        return mock_paths


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("完整子图检索系统 - 平衡扩展方案")
    print("="*80)

    config = RetrievalConfig(
        use_beam_search=True,
        beam_width=20,
        max_expansion_depth=2,
        expand_from_all_nodes=True,
        enable_repetition_detection=True,
        top_k_paths=10
    )

    schema = {
        'relationships': [
            {'from': 'River', 'to': 'Country', 'type': 'flowsThrough'},
            {'from': 'River', 'to': 'Lake', 'type': 'flowsInto'},
            {'from': 'River', 'to': 'DrainageBasin', 'type': 'partOf'},
            {'from': 'Country', 'to': 'Lake', 'type': 'hasLake'},
        ]
    }

    print("\n[1] 初始化SubgraphRetriever...")
    retriever = SubgraphRetriever(config, schema)

    case = {
        'question': "Rivers that flow through at least two countries",
        'question_analysis': {
            'entities': {
                'anchor_candidates': [{'label': 'River'}],
                'faiss_candidates': [
                    {'label': 'Country', 'confidence': 0.9},
                    {'label': 'Lake', 'confidence': 0.6}
                ]
            }
        }
    }

    print("\n[2] 执行检索...")
    results = retriever.retrieve(pool=None, db_name='geography', case=case)

    print("\n[3] 检索结果:")
    print(f"  • Pattern路径: {len(results['pattern_paths'])}")
    print(f"  • Beam扩展路径: {len(results['beam_paths'])}")
    print(f"  • 最终路径: {len(results['final_paths'])}")

    print("\n" + "="*80)
    print("完整系统测试完成！")
    print("="*80)
