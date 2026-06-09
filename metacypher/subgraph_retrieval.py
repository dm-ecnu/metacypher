"""
子图检索系统 - 集成平衡扩展方案

完整功能:
1. ✓ Pattern-based快速检索
2. ✓ 平衡方案的Beam Search扩展（开放性 + 智能引导 + 重复关系）
3. ✓ BGE-M3 embedding
4. ✓ OPTIONAL MATCH扩展
5. ✓ 智能评分机制（related nodes + prior score）

作者: Claude & Wonyoung
日期: 2025-12-31
版本: 2.0 - Balanced Expansion
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from pathlib import Path
import re
import math
import datetime
import numpy as np
from copy import deepcopy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import TEMPLATE_DIR, EMBED_MODEL_PATH

SCHEMA_TEMPLATE_DIR = str(TEMPLATE_DIR)

# ============================================================================
# 配置（扩展版 - 平衡方案）
# ============================================================================

@dataclass
class RetrievalConfig:
    """子图检索配置（包含平衡方案的Beam Search）"""
    
    # 锚点配置
    top_m_anchors: int = 3
    entity_distance_threshold: float = 0.4
    
    # Pattern配置
    top_p_patterns: int = 8
    pattern_score_threshold: float = 0.85
    
    # 路径配置
    top_k_paths: int = 20
    paths_per_signature: int = 2
    min_path_score: float = 0.3
    max_path_length: int = 4
    
    # 查询配置
    max_cypher_limit: int = 500
    
    # Debug配置
    debug_mode: bool = True
    debug_print_query: bool = True
    debug_print_results: bool = True
    debug_max_results_display: int = 3
    debug_print_cypher: bool = True  # ⭐ 新增：打印Cypher查询
    
    # 评分权重
    path_score_weights: Dict[str, float] = field(default_factory=lambda: {
        'pattern_relevance': 0.25,
        'anchor_coverage': 0.30,
        'attribute_match': 0.25,
        'structural_quality': 0.20,
    })
    
    # ========== Beam Search配置 ==========
    use_beam_search: bool = True
    
    # Beam参数
    beam_width: int = 20  # k: 每轮保留的路径数
    candidate_width: int = 40  # 2k: 语义筛选后执行的候选数
    max_expansion_depth: int = 2  # 最大扩展深度（跳数）
    
    # 扩展策略
    expansion_directions: List[str] = field(
        default_factory=lambda: ['forward', 'backward']
    )
    expand_from_all_nodes: bool = True  # 从路径所有节点扩展（不只端点）
    
    # 语义过滤
    use_semantic_filtering: bool = True
    min_semantic_similarity: float = 0.3
    
    # Schema过滤
    use_schema_filtering: bool = True
    max_relations_per_node: int = 10  # 每个节点最多扩展的关系数
    
    # ========== 平衡方案配置（新增）==========
    
    # 智能评分权重
    related_node_bonus: float = 0.3  # Related node额外分数
    in_path_bonus: float = 0.2  # 路径中已有类型的额外分
    repeated_relation_bonus: float = 0.15  # 重复关系的额外分
    
    # 重复关系检测
    enable_repetition_detection: bool = True
    min_repetition_confidence: float = 0.7
    
    # 执行配置
    max_results_per_query: int = 50
    beam_search_timeout: int = 30
    
    # Embedding配置
    embedding_model: str = EMBED_MODEL_PATH
    embedding_device: str = 'cuda'
    embedding_batch_size: int = 32
    use_pattern_based: bool = True
    
    beam_score_weights: Dict[str, float] = field(default_factory=lambda: {
        "semantic": 0.60,
        "coverage": 0.30,
        "prior": 0.10,
    })
    min_beam_final_score: float = 0.0
    coverage_beta: float = 1.0
    coverage_use_length_gain: bool = True
    coverage_length_tau: float = 8.0
    max_sample_paths_per_triple: int = 5


# ============================================================================
# 语义描述模板
# ============================================================================

# 关系类型到自然语言的映射
RELATION_TEMPLATES = {
    'flowsThrough': 'flows through',
    'flowsInto': 'flows into',
    'locatedIn': 'is located in',
    'partOf': 'is part of',
    'hasCapital': 'has capital',
    'bornIn': 'was born in',
    'writtenBy': 'was written by',
    'directedBy': 'was directed by',
    'actedIn': 'acted in',
    'releasedIn': 'was released in',
    'operatesIn': 'operates in',
    'manufactures': 'manufactures',
    'owns': 'owns',
    'subsidiaryOf': 'is a subsidiary of',
    'hasCEO': 'has CEO',
    'playsFor': 'plays for',
    'playsPosition': 'plays position',
    'receivesAward': 'receives award',
    'killedBy': 'was killed by',
    'hasMother': 'has mother',
    'hasFather': 'has father',
}

# 属性名到自然语言的映射
PROPERTY_NAMES = {
    'area_km2': 'area in square kilometers',
    'population': 'population count',
    'elevation_m': 'elevation in meters',
    'height_m': 'height in meters',
    'height_cm': 'height in centimeters',
    'depth_m': 'depth in meters',
    'length_km': 'length in kilometers',
    'founded_year': 'founding year',
    'launch_year': 'launch year',
    'start_year': 'start year',
    'end_year': 'end year',
    'birth_year': 'birth year',
    'death_year': 'death year',
    'creation_year': 'creation year',
    'first_flight': 'first flight date',
    'country_of_citizenship': 'country of citizenship',
    'discharge_m3_s': 'discharge rate in cubic meters per second',
    'global_box_office_usd': 'global box office in USD',
    'max_vertical_depth_m': 'maximum vertical depth in meters',
}

# 操作符到自然语言的映射
OPERATOR_TEMPLATES = {
    'lt': 'less than',
    'le': 'less than or equal to',
    'gt': 'greater than',
    'ge': 'greater than or equal to',
    'eq': 'equal to',
    'ne': 'not equal to',
    'in': 'contains'
}


def _format_cypher_value(value: Any) -> str:
    """格式化Cypher值，字符串使用双引号以避免单引号冲突"""
    if isinstance(value, str):
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f"\"{escaped}\""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "NULL"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_cypher_value(v) for v in value) + "]"
    return str(value)


def _format_cypher_date(value: Any) -> Optional[str]:
    """将输入转换为Cypher date(...) 表达式；若无法解析则返回None。"""
    if value is None:
        return None
    if isinstance(value, datetime.date):
        return f"date({_format_cypher_value(value.isoformat())})"
    if isinstance(value, (datetime.datetime, datetime.time)):
        try:
            return f"date({_format_cypher_value(value.date().isoformat())})"
        except Exception:
            return None
    if isinstance(value, int):
        return f"date({_format_cypher_value(f'{value:04d}-01-01')})"
    if isinstance(value, str):
        text = value.strip()
        # allow YYYY or YYYY-MM-DD or YYYY/MM/DD
        for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
            try:
                if pattern == "%Y":
                    parsed = datetime.date(int(text), 1, 1)
                else:
                    parsed = datetime.datetime.strptime(text, pattern).date()
                return f"date({_format_cypher_value(parsed.isoformat())})"
            except (ValueError, TypeError):
                continue
        logger.warning("[CypherDate] Unable to parse date literal '%s'; skipping constraint", text)
        return None
    return None


def _parse_numeric_value(value: Any, allow_float: bool = False) -> Optional[float]:
    """尝试解析数值类型，若失败返回None"""
    if value is None:
        return None
    try:
        if allow_float:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        logger.warning("[Numeric] 无法解析数值 '%s' (allow_float=%s)", value, allow_float)
        return None


# ============================================================================
# BGE-M3 Embedding接口
# ============================================================================

class BGEEmbedder:
    """BGE-M3 Embedding模型封装"""
    
    def __init__(self, model_name: str = 'BAAI/bge-m3', device: str = 'cuda'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            logger.info(f"[BGE] 加载模型 {self.model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # 移动到设备
            if self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.cuda()
            else:
                self.device = 'cpu'
                self.model = self.model.cpu()
            
            self.model.eval()
            
            logger.info(f"[BGE] 模型加载成功 (device={self.device})")
            
        except Exception as e:
            logger.warning(f"[BGE] 无法加载transformers模型: {e}")
            logger.warning(f"[BGE] 使用Mock模式（随机向量）")
            self.model = None
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本"""
        if self.model is None or self.tokenizer is None:
            # Mock模式
            return np.random.randn(len(texts), 768).astype(np.float32)
        
        try:
            import torch
            
            # 检查模型是否加载
            if self.model is None or self.tokenizer is None:
                return np.random.randn(len(texts), 768).astype(np.float32)
            
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # 移动到设备
                if self.device == 'cuda':
                    encoded = {k: v.cuda() for k, v in encoded.items()}
                
                # 编码
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
            
            return np.vstack(all_embeddings)
        
        except Exception as e:
            logger.error(f"[BGE] 编码失败: {e}")
            return np.random.randn(len(texts), 768).astype(np.float32)
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.encode([text])[0]


class SchemaTemplateManager:
    """加载并提供schema模板查询"""
    
    def __init__(self, base_dir: str):
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
            schema_data = json.load(f)
        
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
        constraint_text_fn: Optional[Callable[[Dict[str, Any], 'PathInstance'], str]] = None,
        operator_text_map: Optional[Dict[str, str]] = None,
        property_name_map: Optional[Dict[str, str]] = None
    ):
        self.template_manager = template_manager
        self.constraint_text_fn = constraint_text_fn
        self.operator_text_map = operator_text_map or OPERATOR_TEMPLATES
        self.property_name_map = property_name_map or PROPERTY_NAMES
    
    def build(
        self,
        path: 'PathInstance',
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
                description = f"{description}. Constraints: {constraint_text}" if description else f"Constraints: {constraint_text}"

        if description:
            description = re.sub(r'\s+', ' ', description).strip()
            description = re.sub(r'\.{2,}', '.', description)
            description = description.strip()
        
        return description
    
    def _build_edge_sentence(self, path: 'PathInstance', edge: Dict[str, Any], use_optional: bool) -> str:
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
                    missing_key,
                    subj_label,
                    rel_type,
                    obj_label
                )
            except Exception as exc:
                logger.warning(
                    "[DescriptionBuilder] Template format failed for relation (%s, %s, %s): %s",
                    subj_label,
                    rel_type,
                    obj_label,
                    exc
                )
        else:
            if not entry:
                logger.warning(
                    "[DescriptionBuilder] Missing template for relation (%s, %s, %s)",
                    subj_label,
                    rel_type,
                    obj_label
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
    
    def _build_constraint_text(self, path: 'PathInstance') -> str:
        texts: List[str] = []
        for constraint in path.constraints:
            if self.constraint_text_fn:
                texts.append(self.constraint_text_fn(constraint, path))
            else:
                texts.append(self._fallback_constraint_to_text(constraint, path))
        return "; ".join(texts)
    
    def _fallback_constraint_to_text(self, constraint: Dict[str, Any], path: 'PathInstance') -> str:
        prop_key = constraint.get('property', 'property')
        op = constraint.get('operator', 'eq')
        value = constraint.get('value')
        
        node_hint = self._resolve_constraint_node_hint(constraint, path)
        
        prop_text = self.property_name_map.get(prop_key, prop_key.replace('_', ' '))
        op_text = self.operator_text_map.get(op, op)
        
        return f"{node_hint} {prop_text} {op_text} {value}"

    def _resolve_constraint_node_hint(self, constraint: Dict[str, Any], path: 'PathInstance') -> str:
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
    
    def _find_node_by_id(self, path: 'PathInstance', node_id: Optional[str]) -> Optional[Dict[str, Any]]:
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


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class AttributeMatch:
    """属性匹配"""
    node_label: str
    property_name: str
    matched_value: Any
    confidence: float = 1.0


@dataclass
class PathInstance:
    """路径实例"""
    path_id: str
    nodes: List[Dict]
    edges: List[Dict]
    
    pattern_id: Optional[str] = None
    pattern_score: float = 0.0
    
    anchor_count: int = 0
    matched_attributes: List[AttributeMatch] = field(default_factory=list)
    
    # WHERE约束条件
    constraints: List[Dict] = field(default_factory=list)  # [{'node_id': 'n0', 'node_label': 'River', 'property': 'area_km2', ...}]
    
    scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    
    # Beam Search相关
    semantic_similarity: float = 0.0
    expansion_info: Optional[Dict] = None
    coverage_score: float = 0.0
    description: str = ""
    prior_score: float = 0.0
    pattern_structure: Dict[str, Any] = field(default_factory=dict)
    anchor_nodes: List[Dict[str, Any]] = field(default_factory=list)
    sample_paths: List[Dict[str, Any]] = field(default_factory=list)
    sample_paths_total_hits: int = 0
    cypher_query: str = ""
    pattern_signature: str = ""
    
    def get_length(self) -> int:
        return len(self.edges)
    
    def get_signature(self) -> str:
        """获取路径签名"""
        parts = []
        for i, node in enumerate(self.nodes):
            parts.append(node.get('label', 'Node'))
            if i < len(self.edges):
                parts.append(self.edges[i].get('rel_type', 'REL'))
        return '->'.join(parts)


def build_match_clause_and_var_map(path: PathInstance) -> Tuple[str, Dict[str, str]]:
    """根据PathInstance构建MATCH子句并返回node_id到变量名的映射"""
    parts: List[str] = []
    node_var_map: Dict[str, str] = {}
    for i, node in enumerate(path.nodes):
        node_label = node.get('label', 'Node')
        node_props = node.get('properties', {}) or {}
        var = f"n{i}"
        node_id = node.get('id')
        if node_id is not None:
            node_var_map[node_id] = var
        if node_props:
            prop_strs = [
                f"{key}: {_format_cypher_value(value)}"
                for key, value in node_props.items()
            ]
            parts.append(f"({var}:{node_label} {{{', '.join(prop_strs)}}})")
        else:
            parts.append(f"({var}:{node_label})")
        if i < len(path.edges):
            edge = path.edges[i]
            rel_type = edge.get('rel_type') or 'REL'
            rel_var = f"r{i}"
            from_id = edge.get('from')
            to_id = edge.get('to')
            curr_id = node.get('id')
            next_id = path.nodes[i + 1].get('id') if (i + 1) < len(path.nodes) else None
            if from_id and to_id and curr_id and next_id:
                if curr_id == from_id and next_id == to_id:
                    parts.append(f"-[{rel_var}:{rel_type}]->")
                elif curr_id == to_id and next_id == from_id:
                    parts.append(f"<-[{rel_var}:{rel_type}]-")
                else:
                    parts.append(f"-[{rel_var}:{rel_type}]-")
            else:
                parts.append(f"-[{rel_var}:{rel_type}]-")
    return "MATCH " + "".join(parts), node_var_map
def _extract_match_where(query: str) -> str:
    """提取Cypher查询的MATCH/WHERE部分"""
    if not query:
        return ""
    upper_q = query.upper()
    idx = upper_q.find("RETURN")
    if idx != -1:
        return query[:idx].strip()
    return query.strip()


@dataclass
class RepetitionInfo:
    """重复关系信息"""
    relation_type: str
    min_count: int  # 至少出现几次
    constraint_type: str = 'different_instances'  # 约束类型
    specific_values: List[str] = field(default_factory=list)


def compute_related_schema_coverage(
    path_nodes: Set[str],
    path_rels: Set[str],
    related_schema: Optional[Dict],
    beta: float = 1.0,
    use_length_gain: bool = False,
    length_tau: float = 8.0
) -> float:
    """计算路径对related schema的覆盖率（Precision-Recall Fβ）"""
    if not related_schema:
        return 0.0
    
    path_elements = {label for label in path_nodes if label}
    path_elements.update(rel for rel in path_rels if rel)
    
    related_elements = set(related_schema.get('node_labels') or [])
    related_elements.update(related_schema.get('relation_types') or [])
    
    if not path_elements:
        return 0.0
    
    intersection = path_elements & related_elements
    precision = len(intersection) / len(path_elements) if path_elements else 0.0
    recall = len(intersection) / len(related_elements) if related_elements else 0.0
    
    if precision == 0.0 and recall == 0.0:
        f_beta = 0.0
    else:
        beta_sq = beta * beta
        f_beta = (1 + beta_sq) * precision * recall
        denom = beta_sq * precision + recall
        f_beta = f_beta / denom if denom > 0 else 0.0
    
    if use_length_gain:
        tau = max(length_tau, 1e-6)
        length_gain = math.tanh(len(path_elements) / tau)
    else:
        length_gain = 1.0
    
    return max(0.0, min(1.0, f_beta * length_gain))


def build_pattern_structure(path: PathInstance) -> Dict[str, Any]:
    """根据路径的节点和边生成pattern结构信息"""
    structure = {
        'node_labels': [node.get('label') for node in path.nodes if node.get('label')],
        'rel_types': [edge.get('rel_type') for edge in path.edges if edge.get('rel_type')],
        'num_hops': len(path.edges),
        'num_nodes': len(path.nodes),
        'num_edges': len(path.edges),
        'has_properties': any(node.get('properties') for node in path.nodes),
        'is_optional': False
    }
    return structure


def extract_anchor_nodes_from_path(path: PathInstance) -> List[Dict[str, Any]]:
    """提取路径中被标记为锚点的节点信息"""
    anchors: List[Dict[str, Any]] = []
    for node in path.nodes:
        if node.get('is_anchor'):
            anchors.append({
                'node_id': node.get('id'),
                'label': node.get('label'),
                'name': node.get('properties', {}).get('name'),
                'sources': node.get('anchor_sources', [])
            })
    return anchors


@dataclass
class ExpansionCandidate:
    """扩展候选（支持重复关系和智能评分）"""
    base_path_id: str
    direction: str
    
    expand_from_node_idx: int
    expand_from_label: str
    expand_from_id: str
    
    relation_type: str
    target_label: Optional[str]
    
    # 重复关系相关
    is_repeated: bool = False
    instance_id: int = 0
    repetition_constraint: Optional[str] = None
    
    # 评分相关
    prior_score: float = 0.0  # 先验分数（related node bonus等）
    semantic_similarity: float = 0.0
    final_score: float = 0.0
    coverage_score: float = 0.0
    
    # 描述和Cypher
    cypher_pattern: str = ""
    optional_match_clause: str = ""
    pattern_description: str = ""


# ============================================================================
# 重复关系检测器
# ============================================================================

class RepetitionDetector:
    """检测问题中的重复关系需求"""
    
    # 触发词
    REPETITION_TRIGGERS = {
        'at least two': 2,
        'at least 2': 2,
        'at least three': 3,
        'at least 3': 3,
        'both': 2,
        'two': 2,
        'three': 3,
        'multiple': 2
    }
    
    def __init__(self, question: str, question_analysis: Optional[Dict] = None):
        self.question = question.lower()
        self.question_analysis = question_analysis or {}
    
    def detect(self) -> List[RepetitionInfo]:
        """检测重复关系需求"""
        repetitions = []
        
        # 方法1: 从问题文本检测
        text_based = self._detect_from_text()
        repetitions.extend(text_based)
        
        # 方法2: 从问题分析检测
        if self.question_analysis:
            analysis_based = self._detect_from_analysis()
            repetitions.extend(analysis_based)
        
        if repetitions:
            logger.info(f"[RepetitionDetector] 检测到 {len(repetitions)} 个重复关系需求")
            for rep in repetitions:
                logger.info(f"  • {rep.relation_type} (至少{rep.min_count}次)")
        
        return repetitions
    
    def _detect_from_text(self) -> List[RepetitionInfo]:
        """从问题文本检测"""
        repetitions = []
        
        for trigger, count in self.REPETITION_TRIGGERS.items():
            if trigger in self.question:
                # 推测关系类型
                if 'flow' in self.question:
                    relation_type = 'flowsThrough' if 'through' in self.question else 'flowsInto'
                elif 'located' in self.question or ' in ' in self.question:
                    relation_type = 'locatedIn'
                elif 'part of' in self.question:
                    relation_type = 'partOf'
                else:
                    continue
                
                repetitions.append(RepetitionInfo(
                    relation_type=relation_type,
                    min_count=count,
                    constraint_type='different_instances'
                ))
                
                logger.debug(f"  触发词 '{trigger}' → {relation_type} ×{count}")
        
        return repetitions
    
    def _detect_from_analysis(self) -> List[RepetitionInfo]:
        """从问题分析检测"""
        repetitions = []
        
        constraints = self.question_analysis.get('constraints', [])
        
        for constraint in constraints:
            if constraint.get('type') == 'cardinality':
                repetitions.append(RepetitionInfo(
                    relation_type=constraint.get('relation'),
                    min_count=constraint.get('min_count', 2),
                    constraint_type='different_instances'
                ))
        
        return repetitions


# ============================================================================
# 平衡候选生成器（核心）
# ============================================================================

class BalancedCandidateGenerator:
    
    def __init__(
        self,
        config: RetrievalConfig,
        schema: Dict,
        question: str,
        question_analysis: Optional[Dict] = None,
        relation_index: Optional[Dict] = None,
        description_builder: Optional[PathDescriptionBuilder] = None
    ):
        self.config = config
        self.schema = schema
        self.question = question
        self.question_analysis = question_analysis or {}
        self.relation_index = relation_index
        self.description_builder = description_builder
        
        # 构建schema索引（如果没有提供relation_index）
        if relation_index:
            self.relation_map = self._convert_index_to_map(relation_index)
            logger.info(f"[BalancedCandGen] 使用提供的relation_index")
        else:
            self._build_schema_index()
            logger.info(f"[BalancedCandGen] 使用简化schema构建索引")
        
        # 提取related nodes（从问题分析）
        self.related_nodes = self._extract_related_nodes()
        
        # 检测重复关系
        if self.config.enable_repetition_detection:
            self.repetition_detector = RepetitionDetector(question, question_analysis)
            self.repetitions = self.repetition_detector.detect()
        else:
            self.repetitions = []
        
        logger.info(f"[BalancedCandGen] 初始化完成")
        logger.info(f"  • Related nodes: {self.related_nodes}")
        logger.info(f"  • Repetitions: {len(self.repetitions)}")
    
    def _convert_index_to_map(self, relation_index: Dict) -> Dict:
        """
        将relation_index转换为relation_map格式
        
        relation_index: {
            ('River', 'out'): [('flowsThrough', 'Country'), ('flowsInto', 'Lake')],
            ...
        }
        
        relation_map: {
            'River': [{'type': 'flowsThrough', 'direction': 'forward', 'target': 'Country'}, ...]
        }
        """
        relation_map = defaultdict(list)
        
        for (node_label, direction_key), relations in relation_index.items():
            for rel_type, target_label in relations:
                if direction_key == 'out':
                    relation_map[node_label].append({
                        'type': rel_type,
                        'direction': 'forward',
                        'target': target_label
                    })
                elif direction_key == 'in':
                    relation_map[node_label].append({
                        'type': rel_type,
                        'direction': 'backward',
                        'target': target_label
                    })
        
        return dict(relation_map)
    
    def _build_schema_index(self):
        """构建schema索引"""
        self.relation_map = defaultdict(list)
        
        relationships = self.schema.get('relationships', [])
        for rel in relationships:
            from_label = rel.get('from')
            to_label = rel.get('to')
            rel_type = rel.get('type')
            
            # Forward
            self.relation_map[from_label].append({
                'type': rel_type,
                'direction': 'forward',
                'target': to_label
            })
            
            # Backward
            self.relation_map[to_label].append({
                'type': rel_type,
                'direction': 'backward',
                'target': from_label
            })
    
    def _extract_related_nodes(self) -> Set[str]:
        """从问题分析提取相关节点类别"""
        related = set()
        
        if not self.question_analysis:
            return related
        
        # 从entities提取（entities是list）
        entities = self.question_analysis.get('entities', [])
        for entity in entities:
            # 从mapped_to提取节点标签
            mapped_to = entity.get('mapped_to', [])
            for mapping in mapped_to:
                label = mapping.get('label', '')
                if label:
                    related.add(label)
            
            # 从faiss_candidates提取节点标签
            faiss_cands = entity.get('faiss_candidates', [])
            for cand in faiss_cands:
                label = cand.get('node_label', '')
                if label:
                    related.add(label)
        
        # 从related_schema提取
        related_schema = self.question_analysis.get('related_schema', {})
        node_labels = related_schema.get('node_labels', [])
        for label in node_labels:
            related.add(label)
        
        return related
    
    def generate_candidates(self, paths: List[PathInstance]) -> List[ExpansionCandidate]:
        """
        生成候选（平衡方案）
        
        策略:
        1. 从路径的所有节点扩展（可配置）
        2. 获取所有Schema关系（开放性）
        3. 计算prior_score（智能引导）
        4. 支持重复关系（复杂模式）
        """
        all_candidates = []
        
        for path in paths:
            # 决定从哪些节点扩展
            if self.config.expand_from_all_nodes:
                # 从所有节点
                expansion_nodes = list(range(len(path.nodes)))
            else:
                # 只从端点
                expansion_nodes = [0, len(path.nodes) - 1] if len(path.nodes) > 1 else [0]
            
            for node_idx in expansion_nodes:
                node = path.nodes[node_idx]
                node_label = node.get('label')
                node_id = node.get('id')
                
                # 跳过没有label或id的节点
                if not node_label or not node_id:
                    continue
                
                # 类型断言：确保node_label和node_id不是None
                assert node_label is not None
                assert node_id is not None
                
                # 决定扩展方向
                if node_idx == 0 and 'backward' in self.config.expansion_directions:
                    # 起点：backward
                    candidates = self._generate_from_node(
                        path, node_idx, node_label, node_id, 'backward'
                    )
                    all_candidates.extend(candidates)
                
                if node_idx == len(path.nodes) - 1 and 'forward' in self.config.expansion_directions:
                    # 终点：forward
                    candidates = self._generate_from_node(
                        path, node_idx, node_label, node_id, 'forward'
                    )
                    all_candidates.extend(candidates)
                
                # 中间节点：两个方向都可以
                if 0 < node_idx < len(path.nodes) - 1:
                    if 'forward' in self.config.expansion_directions:
                        candidates = self._generate_from_node(
                            path, node_idx, node_label, node_id, 'forward'
                        )
                        all_candidates.extend(candidates)
                    
                    if 'backward' in self.config.expansion_directions:
                        candidates = self._generate_from_node(
                            path, node_idx, node_label, node_id, 'backward'
                        )
                        all_candidates.extend(candidates)
        
        logger.info(f"[BalancedCandGen] 生成 {len(all_candidates)} 个候选（包含重复）")
        
        return all_candidates
    
    def _generate_from_node(
        self,
        path: PathInstance,
        node_idx: int,
        node_label: str,
        node_id: str,
        direction: str
    ) -> List[ExpansionCandidate]:
        """从单个节点生成候选"""
        candidates = []
        
        # 获取该节点的所有Schema关系（开放性）
        possible_relations = self.relation_map.get(node_label, [])
        
        # 过滤方向
        valid_relations = [r for r in possible_relations if r['direction'] == direction]
        
        # 限制数量（可配置）
        valid_relations = valid_relations[:self.config.max_relations_per_node]
        
        for rel_info in valid_relations:
            rel_type = rel_info['type']
            target_label = rel_info['target']
            
            # 检查是否需要重复
            repeat_count = self._get_repeat_count(rel_type)
            
            if repeat_count > 1:
                # 生成多个实例
                for instance_id in range(repeat_count):
                    candidate = self._create_candidate(
                        path, node_idx, node_label, node_id,
                        rel_type, target_label, direction,
                        is_repeated=True,
                        instance_id=instance_id
                    )
                    candidates.append(candidate)
            else:
                # 正常生成（不重复）
                candidate = self._create_candidate(
                    path, node_idx, node_label, node_id,
                    rel_type, target_label, direction,
                    is_repeated=False,
                    instance_id=0
                )
                candidates.append(candidate)
        
        return candidates
    
    def _get_repeat_count(self, rel_type: str) -> int:
        """获取关系需要重复的次数"""
        for rep in self.repetitions:
            if rep.relation_type == rel_type:
                return rep.min_count
        return 1  # 默认不重复
    
    def _create_candidate(
        self,
        path: PathInstance,
        node_idx: int,
        node_label: str,
        node_id: str,
        rel_type: str,
        target_label: str,
        direction: str,
        is_repeated: bool,
        instance_id: int
    ) -> ExpansionCandidate:
        """创建候选"""
        
        # 计算prior_score（核心：智能引导）
        prior_score = self._calculate_prior_score(
            target_label,
            is_repeated,
            path
        )
        
        # 生成描述
        description = self._generate_description(
            path=path,
            node_idx=node_idx,
            from_label=node_label,
            rel_type=rel_type,
            to_label=target_label,
            direction=direction,
            is_repeated=is_repeated,
            instance_id=instance_id
        )
        
        related_schema = (self.question_analysis or {}).get('related_schema', {})
        expected_nodes = {n.get('label') for n in path.nodes if n.get('label')}
        expected_rels = {edge.get('rel_type') for edge in path.edges if edge.get('rel_type')}
        if target_label:
            expected_nodes.add(target_label)
        if rel_type:
            expected_rels.add(rel_type)
        coverage_score = compute_related_schema_coverage(
            expected_nodes,
            expected_rels,
            related_schema,
            beta=self.config.coverage_beta,
            use_length_gain=self.config.coverage_use_length_gain,
            length_tau=self.config.coverage_length_tau
        )
        
        # 生成重复约束
        constraint = None
        if is_repeated and instance_id > 0:
            constraint = self._generate_repetition_constraint(
                rel_type, target_label, instance_id
            )
        
        candidate = ExpansionCandidate(
            base_path_id=path.path_id,
            direction=direction,
            expand_from_node_idx=node_idx,
            expand_from_label=node_label,
            expand_from_id=node_id,
            relation_type=rel_type,
            target_label=target_label,
            is_repeated=is_repeated,
            instance_id=instance_id,
            repetition_constraint=constraint,
            prior_score=prior_score,
            pattern_description=description,
            coverage_score=coverage_score
        )
        
        return candidate
    
    def _calculate_prior_score(
        self,
        target_label: str,
        is_repeated: bool,
        path: PathInstance
    ) -> float:
        score = 0.0
        
        # 1. Related node bonus
        if target_label in self.related_nodes:
            score += self.config.related_node_bonus
            logger.debug(f"  • {target_label}: related_node +{self.config.related_node_bonus}")
        
        # 2. 路径中已有类型
        path_labels = {n.get('label') for n in path.nodes}
        if target_label in path_labels:
            score += self.config.in_path_bonus
            logger.debug(f"  • {target_label}: in_path +{self.config.in_path_bonus}")
        
        # 3. 重复关系bonus
        if is_repeated:
            score += self.config.repeated_relation_bonus
            logger.debug(f"  • repeated_relation +{self.config.repeated_relation_bonus}")
        
        return score
    
    def _generate_description(
        self,
        path: PathInstance,
        node_idx: int,
        from_label: str,
        rel_type: str,
        to_label: str,
        direction: str,
        is_repeated: bool,
        instance_id: int
    ) -> str:
        """生成自然语言描述"""
        base_desc = ""
        if self.description_builder:
            base_desc = self.description_builder.build(
                path,
                use_optional=False,
                include_constraints=True
            )
        else:
            base_desc = self._path_to_text(path)
        
        expansion_sentence = self._build_expansion_sentence(
            path=path,
            node_idx=node_idx,
            rel_type=rel_type,
            target_label=to_label,
            direction=direction,
            is_repeated=is_repeated,
            instance_id=instance_id
        )
        
        if base_desc and expansion_sentence:
            return f"{base_desc} {expansion_sentence}".strip()
        return base_desc or expansion_sentence
    
    def _path_to_text(self, path: PathInstance) -> str:
        """路径转自然语言"""
        parts = []
        for i, node in enumerate(path.nodes):
            label = node.get('label', 'entity')
            parts.append(label)
            if i < len(path.edges):
                edge = path.edges[i]
                rel_type = edge.get('rel_type', 'relates to')
                # 兼容pattern解析阶段使用的from_var/to_var（尚未有真实id）
                from_id = edge.get('from')
                to_id = edge.get('to')
                # 兼容pattern解析阶段使用的from_var/to_var（尚未有真实id）
                if (not from_id or not to_id) and edge.get('from_var') and edge.get('to_var'):
                    # 用节点顺序近似绑定变量到label（此处仅用于描述，不影响执行）
                    # 若未来节点结构包含var字段，可在此改为var->label映射
                    pass
                rel_text = self._relation_to_text(rel_type)
                parts.append(rel_text)
        return ' '.join(parts)
    
    def _relation_to_text(self, rel_type: str) -> str:
        """关系类型转自然语言"""
        mapping = {
            'flowsThrough': 'flows through',
            'flowsInto': 'flows into',
            'locatedIn': 'is located in',
            'partOf': 'is part of',
            'hasCapital': 'has capital',
            'originatesFrom': 'originates from',
        }
        return mapping.get(rel_type, rel_type.lower().replace('_', ' '))
    
    def _get_node_display_name(self, node: Dict[str, Any]) -> str:
        props = node.get('properties') or {}
        if 'name' in props:
            return props['name']
        if node.get('display_name'):
            return node['display_name']
        label = node.get('label')
        if label:
            return f"unspecified {label.lower()}"
        return "unspecified entity"
    
    def _build_expansion_sentence(
        self,
        path: PathInstance,
        node_idx: int,
        rel_type: str,
        target_label: str,
        direction: str,
        is_repeated: bool,
        instance_id: int
    ) -> str:
        subj_node = path.nodes[node_idx]
        target_name = self._get_node_display_name({'label': target_label or 'Entity'})
        existing_name = self._get_node_display_name(subj_node)
        
        sentence = ""
        if self.description_builder:
            temp_nodes, temp_edge = self._build_temp_expansion_path(
                subj_node=subj_node,
                target_label=target_label,
                rel_type=rel_type,
                direction=direction
            )
            temp_path = PathInstance(
                path_id="exp_temp",
                nodes=temp_nodes,
                edges=[temp_edge]
            )
            sentence = self.description_builder.build(
                temp_path,
                use_optional=True,
                include_constraints=False
            )
        
        if not sentence:
            if direction == 'backward':
                sentence = f"{target_name} -[{rel_type}]-> {existing_name}"
            else:
                sentence = f"{existing_name} -[{rel_type}]-> {target_name}"

        if is_repeated:
            ordinals = ['first', 'second', 'third', 'fourth', 'fifth']
            ordinal = ordinals[instance_id] if instance_id < len(ordinals) else f"{instance_id+1}th"
            sentence = f"{sentence} (instance: {ordinal})"

        return sentence
    
    def _build_temp_expansion_path(
        self,
        subj_node: Dict[str, Any],
        target_label: str,
        rel_type: str,
        direction: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        subj_clone = self._clone_node_for_description(
            subj_node,
            fallback_id=f"temp_subj_{id(subj_node)}"
        )
        new_node_id = f"temp_{target_label or 'Node'}_{subj_clone['id']}"
        new_node = {
            'label': target_label,
            'id': new_node_id,
            'properties': {},
            'display_name': None
        }
        
        if direction == 'backward':
            nodes = [new_node, subj_clone]
            edge = {
                'rel_type': rel_type,
                'from': new_node_id,
                'to': subj_clone['id'],
                'properties': {}
            }
        else:
            nodes = [subj_clone, new_node]
            edge = {
                'rel_type': rel_type,
                'from': subj_clone['id'],
                'to': new_node_id,
                'properties': {}
            }
        return nodes, edge
    
    def _clone_node_for_description(self, node: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
        return {
            'label': node.get('label'),
            'id': node.get('id') or node.get('var') or fallback_id,
            'properties': dict(node.get('properties') or {}),
            'display_name': node.get('display_name')
        }
    
    def _generate_repetition_constraint(
        self,
        rel_type: str,
        target_label: str,
        instance_id: int
    ) -> str:
        """重复约束不在Cypher层实现（避免无效变量），返回空串"""
        return ""


# ============================================================================
# 语义过滤器（集成prior_score）
# ============================================================================

class SemanticFilter:
    """语义过滤器（集成prior_score）"""
    
    def __init__(
        self,
        config: RetrievalConfig,
        embedder: BGEEmbedder,
        question: str,
        related_schema: Optional[Dict] = None
    ):
        self.config = config
        self.embedder = embedder
        self.question = question
        self.related_schema = related_schema or {}
        
        # 预计算问题embedding
        self.question_embedding = embedder.encode_single(question)
        
        logger.info(f"[SemanticFilter] 初始化完成")
    
    def filter_and_rank(
        self,
        candidates: List[ExpansionCandidate],
        top_k: Optional[int] = None
    ) -> List[ExpansionCandidate]:
        """
        语义过滤和排序
        
        核心: final_score = semantic_similarity + prior_score
        """
        if not self.config.use_semantic_filtering:
            return candidates[:top_k] if top_k else candidates
        
        if not candidates:
            return []
        
        similarities = [0.0 for _ in candidates]
        valid_descriptions = []
        valid_indices = []
        for idx, candidate in enumerate(candidates):
            desc = (candidate.pattern_description or "").strip()
            if desc:
                valid_descriptions.append(desc)
                valid_indices.append(idx)
        
        if valid_descriptions:
            desc_embeddings = self.embedder.encode(valid_descriptions, self.config.embedding_batch_size)
            valid_similarities = self._cosine_similarity(self.question_embedding, desc_embeddings)
            for sim, idx in zip(valid_similarities, valid_indices):
                similarities[idx] = float(sim)
        
        weights = self.config.beam_score_weights
        # 3. 计算final_score = semantic + coverage + prior
        for i, candidate in enumerate(candidates):
            candidate.semantic_similarity = float(similarities[i])
            candidate.final_score = (
                weights.get("semantic", 0.0) * candidate.semantic_similarity +
                weights.get("coverage", 0.0) * candidate.coverage_score +
                weights.get("prior", 0.0) * candidate.prior_score
            )
        
        # 4. 过滤低分
        filtered = [
            c for c in candidates
            if c.final_score >= self.config.min_beam_final_score
        ]
        
        logger.info(f"[SemanticFilter] 过滤: {len(candidates)} → {len(filtered)}")
        
        # 5. 排序
        filtered.sort(key=lambda c: c.final_score, reverse=True)
        
        # 6. 取top-k
        if top_k:
            filtered = filtered[:top_k]
        
        # 记录top-3
        if filtered and logger.isEnabledFor(logging.INFO):
            logger.info(f"[SemanticFilter] Top-3候选:")
            for i, c in enumerate(filtered[:3], 1):
                logger.info(
                    f"  {i}. {c.expand_from_label} -[{c.relation_type}]-> {c.target_label} "
                    f"(sem={c.semantic_similarity:.3f}, prior={c.prior_score:.3f}, "
                    f"cov={c.coverage_score:.3f}, "
                    f"final={c.final_score:.3f})"
                )
        
        return filtered
    
    def _cosine_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """计算余弦相似度"""
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities


# ============================================================================
# OPTIONAL MATCH执行器
# ============================================================================

class OptionalMatchExecutor:
    """OPTIONAL MATCH执行器"""
    
    def __init__(
        self,
        config: RetrievalConfig,
        embedder: Optional[BGEEmbedder] = None,
        question_embedding: Optional[np.ndarray] = None,
        description_builder: Optional[PathDescriptionBuilder] = None,
        related_schema: Optional[Dict] = None
    ):
        self.config = config
        self.embedder = embedder
        self.question_embedding = question_embedding
        self.description_builder = description_builder
        self.related_schema = related_schema or {}
        if question_embedding is not None:
            norm = np.linalg.norm(question_embedding) + 1e-8
            self._normalized_question_embedding = question_embedding / norm
        else:
            self._normalized_question_embedding = None
    
    def execute_candidates(
        self,
        candidates: List[ExpansionCandidate],
        base_paths: Dict[str, PathInstance],
        pool,
        db_name: str,
        semantic_filter: Optional['SemanticFilter'] = None
    ) -> List[PathInstance]:
        """
        执行候选扩展
        
        使用OPTIONAL MATCH确保不会因扩展失败而丢失基础路径
        """
        expanded_paths = []
        
        logger.info(f"[OptionalMatch] 执行 {len(candidates)} 个候选")
        
        if semantic_filter:
            candidates = semantic_filter.filter_and_rank(
                candidates,
                top_k=self.config.candidate_width
            )
        else:
            candidates = candidates[:self.config.candidate_width]
        
        # ⭐ Debug: 打印前几个扩展查询
        if self.config.debug_print_cypher and candidates:
            logger.info(f"\n{'='*80}")
            logger.info(f"[DEBUG] 扩展路径的Cypher查询（前{min(3, len(candidates))}个）")
            logger.info(f"{'='*80}")
        
        for idx, candidate in enumerate(candidates):
            base_path = base_paths.get(candidate.base_path_id)
            if not base_path:
                continue
            
            try:
                # 构建OPTIONAL MATCH查询
                query = self._build_optional_match_query(candidate, base_path)
                
                # ⭐ Debug: 打印前几个查询
                if self.config.debug_print_cypher and idx < 3:
                    logger.info(f"\n  [{idx+1}] 扩展查询:")
                    logger.info(f"  {'─'*60}")
                    for line in query.split('\n'):
                        logger.info(f"  {line}")
                    logger.info(f"  {'─'*60}")
                    logger.info(f"  Base: {base_path.pattern_id}, 扩展: {candidate.relation_type}\n")
                
                # 执行
                results = self._execute_query(query, pool, db_name)
                
                # 解析结果
                new_paths = self._parse_results(results, candidate, base_path, query)
                
                expanded_paths.extend(new_paths)
                
            except Exception as e:
                logger.warning(f"[OptionalMatch] 执行失败: {e}")
                continue
        
        logger.info(f"[OptionalMatch] 生成 {len(expanded_paths)} 条新路径")
        
        return expanded_paths
    
    def _build_optional_match_query(
        self,
        candidate: ExpansionCandidate,
        base_path: PathInstance
    ) -> str:
        """构建OPTIONAL MATCH查询"""
        
        # 构建base MATCH（从现有路径）
        base_match, node_var_map = self._build_base_match(base_path)
        
        # 构建WHERE约束（来自constraints）
        where_clause = self._build_where_clause(base_path, node_var_map)
        
        # 构建OPTIONAL MATCH（扩展）
        optional_match = self._build_optional_match(candidate, base_path, node_var_map)
        
        # RETURN子句
        return_clause = self._build_return_clause(candidate, base_path, node_var_map)
        
        # 组合
        parts = [base_match]
        if where_clause:
            parts.append(where_clause)
        parts.append(optional_match)
        parts.append(return_clause)
        
        query = "\n".join(parts)
        
        if self.config.debug_print_query:
            logger.debug(f"[Query]\n{query}")
        
        return query
    
    def _build_base_match(self, path: PathInstance) -> Tuple[str, Dict[str, str]]:
        """构建基础MATCH"""
        return build_match_clause_and_var_map(path)
    
    def _build_where_clause(self, path: PathInstance, node_var_map: Dict[str, str]) -> str:
        """
        构建WHERE子句（来自constraints）
        
        支持所有属性类型:
        - str, int, float: 标准比较
        - date: 使用date()函数，end_*属性的gt要OR IS NULL
        - list(str): 使用IN操作符
        """
        if not path.constraints:
            return ""
        
        # 操作符映射
        operator_map = {
            'lt': '<',
            'le': '<=',
            'gt': '>',
            'ge': '>=',
            'eq': '=',
            'ne': '!='
        }
        
        conditions = []
        
        for c in path.constraints:
            node_var = self._resolve_node_var(
                constraint=c,
                node_var_map=node_var_map
            )
            prop = c.get('property')
            op = c.get('operator', 'eq')
            value = c.get('value')
            value_type = c.get('value_type', 'int')
            
            if value_type == 'date':
                # 日期类型：必须使用date()函数，若无法解析则跳过该约束
                cypher_value = _format_cypher_date(value)
                if cypher_value is None:
                    continue
                op_symbol = operator_map.get(op, '=')
                
                if c.get('allow_null'):
                    # 晚于查询（允许NULL）- 用于end_*属性
                    conditions.append(
                        f"({node_var}.{prop} {op_symbol} {cypher_value} "
                        f"OR {node_var}.{prop} IS NULL)"
                    )
                else:
                    conditions.append(f"{node_var}.{prop} {op_symbol} {cypher_value}")
            
            elif value_type == 'list(str)':
                # 列表类型：使用IN操作符
                cypher_val = _format_cypher_value(value)
                conditions.append(f"{cypher_val} IN {node_var}.{prop}")
            
            elif value_type in ['int', 'float']:
                # 数值类型：直接比较
                op_symbol = operator_map.get(op, '=')
                parsed_value = _parse_numeric_value(
                    value,
                    allow_float=(value_type == 'float')
                )
                if parsed_value is None:
                    continue
                conditions.append(f"{node_var}.{prop} {op_symbol} {parsed_value}")
            
            elif value_type == 'str':
                # 字符串类型：直接比较
                op_symbol = operator_map.get(op, '=')
                conditions.append(f"{node_var}.{prop} {op_symbol} {_format_cypher_value(value)}")
        
        if conditions:
            return "WHERE " + " AND ".join(conditions)
        return ""
    
    def _resolve_node_var(
        self,
        constraint: Dict,
        node_var_map: Dict[str, str]
    ) -> str:
        """根据约束解析节点变量名"""
        node_var = constraint.get('node_var')
        if node_var:
            return node_var
        
        node_id = constraint.get('node_id')
        if node_id and node_var_map.get(node_id):
            return node_var_map[node_id]
        
        node_idx = constraint.get('node_idx')
        if node_idx is not None:
            return f"n{node_idx}"
        
        return "n0"
    
    def _get_candidate_node_var(
        self,
        candidate: ExpansionCandidate,
        node_var_map: Dict[str, str]
    ) -> str:
        """为候选扩展解析节点变量名"""
        node_id = candidate.expand_from_id
        if node_id and node_var_map.get(node_id):
            return node_var_map[node_id]
        return f"n{candidate.expand_from_node_idx}"
    
    def _get_new_node_var_name(self, candidate: ExpansionCandidate) -> str:
        """生成新节点变量名"""
        return f"n_new{candidate.instance_id}" if candidate.is_repeated else "n_new"
    
    def _get_new_rel_var_name(self, candidate: ExpansionCandidate) -> str:
        """生成新关系变量名"""
        return f"r_new{candidate.instance_id}" if candidate.is_repeated else "r_new"
    
    def _build_optional_match(
        self,
        candidate: ExpansionCandidate,
        base_path: PathInstance,
        node_var_map: Dict[str, str]
    ) -> str:
        """构建OPTIONAL MATCH"""
        
        from_var = self._get_candidate_node_var(candidate, node_var_map)
        new_var = self._get_new_node_var_name(candidate)
        new_rel_var = self._get_new_rel_var_name(candidate)
        
        rel_type = candidate.relation_type
        target_label = candidate.target_label

        if candidate.direction == 'forward':
            optional = f"OPTIONAL MATCH ({from_var})-[{new_rel_var}:{rel_type}]->({new_var}:{target_label})"
        else:
            # backward表示: target_label -[rel_type]-> from_var  等价于 from_var <-[rel_type]- target
            optional = f"OPTIONAL MATCH ({from_var})<-[{new_rel_var}:{rel_type}]-({new_var}:{target_label})"
        
        # 添加重复约束
        if candidate.repetition_constraint:
            optional += f"\nWHERE {candidate.repetition_constraint}"
        
        return optional
    
    def _build_return_clause(
        self,
        candidate: ExpansionCandidate,
        base_path: PathInstance,
        node_var_map: Dict[str, str]
    ) -> str:
        """构建RETURN子句"""
        
        base_var = self._get_candidate_node_var(candidate, node_var_map)
        new_var = self._get_new_node_var_name(candidate)
        new_rel_var = self._get_new_rel_var_name(candidate)
        
        return (
            f"RETURN {base_var} AS base_node, "
            f"{new_var} AS new_node, "
            f"{new_rel_var} AS new_rel "
            f"LIMIT {self.config.max_results_per_query}"
        )
    
    def _build_sample_signature(self, nodes: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        signature = []
        for node in nodes:
            identifier = node.get('id') or node.get('properties', {}).get('name') or node.get('label')
            signature.append(identifier)
        return tuple(signature)
    
    def _execute_query(self, query: str, pool, db_name: str) -> List[Dict]:
        """执行查询"""
        try:
            # ⭐ 修复：使用pool.execute而不是pool.get_connection
            if hasattr(pool, 'execute'):
                # 新版API: pool.execute(db_name, query)
                result = pool.execute(db_name, query)
                records = []
                for record in result:
                    records.append(dict(record))
                return records
            else:
                # 旧版API: pool.get_connection
                with pool.get_connection(db_name) as conn:
                    with conn.session() as session:
                        result = session.run(query)
                        records = []
                        for record in result:
                            records.append(dict(record))
                        return records
        except Exception as e:
            logger.warning(f"[OptionalMatch] 查询执行失败: {e}")
            if self.config.debug_print_query:
                logger.debug(f"[FailedQuery]\n{query}")
            return []
    
    def _parse_results(
        self,
        results: List[Dict],
        candidate: ExpansionCandidate,
        base_path: PathInstance,
        executed_query: str
    ) -> List[PathInstance]:
        """解析结果"""
        new_paths = []
        sample_entries: List[Dict[str, Any]] = []
        seen_signatures: Set[Tuple[Any, ...]] = set()
        max_samples = getattr(self.config, "max_sample_paths_per_triple", 5)
        
        for record in results:
            new_node_data = record.get('new_node')
            if new_node_data is None:
                new_node_key = self._get_new_node_var_name(candidate)
                new_node_data = record.get(new_node_key)
            
            if new_node_data is None:
                # 扩展失败，保留原路径
                continue
            
            base_node_data = record.get('base_node')
            rel_data = record.get('new_rel')
            
            # 创建新路径
            new_path = deepcopy(base_path)
            
            # 添加新节点和边
            new_node_id = new_node_data.get('id')
            if new_node_id is None:
                new_node_id = new_node_data.get('eid')
            if new_node_id is None:
                assert candidate.target_label and candidate.instance_id is not None
                new_node_id = f"{candidate.target_label.lower()}_{len(new_path.nodes)}_{candidate.instance_id}"
            
            new_node = {
                'label': candidate.target_label,
                'id': new_node_id,
                'var': new_node_id,
                'properties': {},
                'display_name': new_node_data.get('name')
            }
            
            source_node = new_path.nodes[candidate.expand_from_node_idx]
            source_id = source_node.get('id')
            
            rel_properties = {}
            if rel_data:
                try:
                    rel_properties = dict(rel_data)
                except Exception:
                    rel_properties = {}

            new_edge = {
                'rel_type': candidate.relation_type,
                'direction': candidate.direction,
                'from': source_id if candidate.direction == 'forward' else new_node_id,
                'to': new_node_id if candidate.direction == 'forward' else source_id,
                'properties': rel_properties
            }
            
            if candidate.direction == 'forward':
                new_path.nodes.append(new_node)
                new_path.edges.append(new_edge)
            else:
                new_path.nodes.insert(0, new_node)
                new_path.edges.insert(0, new_edge)
            
            # 更新path_id
            new_path.path_id = f"{base_path.path_id}_exp{len(new_paths)}"
            
            new_path.expansion_info = {
                'latest_triplet': {
                    'from_node': self._extract_node_snapshot(
                        base_node_data,
                        fallback_label=source_node.get('label'),
                        fallback_id=source_node.get('id')
                    ),
                    'relationship': self._extract_relationship_snapshot(
                        rel_data,
                        candidate.relation_type
                    ),
                    'to_node': self._extract_node_snapshot(
                        new_node_data,
                        fallback_label=candidate.target_label,
                        fallback_id=new_node_id
                    )
                }
            }

            new_path.prior_score = candidate.prior_score
            new_path.cypher_query = executed_query
            
            self._populate_path_metadata(new_path)
            
            new_paths.append(new_path)
            
            sig = self._build_sample_signature(new_path.nodes)
            if sig not in seen_signatures and len(sample_entries) < max_samples:
                sample_entries.append({
                    'nodes': deepcopy(new_path.nodes),
                    'edges': deepcopy(new_path.edges)
                })
                seen_signatures.add(sig)
        
        total_hits = len(results)
        if not sample_entries and new_paths:
            sample_entries = [{'nodes': deepcopy(new_paths[0].nodes), 'edges': deepcopy(new_paths[0].edges)}]
        for path in new_paths:
            path.sample_paths_total_hits = total_hits
            path.sample_paths = [deepcopy(entry) for entry in sample_entries]
        
        return new_paths

    def _populate_path_metadata(self, path: PathInstance) -> None:
        """补齐路径描述、覆盖率和语义评分"""
        if self.description_builder:
            path.description = self.description_builder.build(
                path,
                use_optional=False,
                include_constraints=True
            )
        else:
            path.description = ""
        
        node_labels = {node.get('label') for node in path.nodes if node.get('label')}
        rel_types = {edge.get('rel_type') for edge in path.edges if edge.get('rel_type')}
        path.coverage_score = compute_related_schema_coverage(
            node_labels,
            rel_types,
            self.related_schema,
            beta=self.config.coverage_beta,
            use_length_gain=self.config.coverage_use_length_gain,
            length_tau=self.config.coverage_length_tau
        )
        
        path.semantic_similarity = self._compute_semantic_similarity(path.description)
        
        weights = self.config.beam_score_weights
        path.final_score = (
            weights.get("semantic", 0.0) * path.semantic_similarity +
            weights.get("coverage", 0.0) * path.coverage_score +
            weights.get("prior", 0.0) * path.prior_score
        )
        if not path.sample_paths:
            path.sample_paths = [{'nodes': deepcopy(path.nodes), 'edges': deepcopy(path.edges)}]
        if not path.sample_paths_total_hits:
            path.sample_paths_total_hits = len(path.sample_paths)
        if not path.pattern_structure:
            path.pattern_structure = build_pattern_structure(path)
        if not path.anchor_nodes:
            path.anchor_nodes = extract_anchor_nodes_from_path(path)
        path.anchor_count = len(path.anchor_nodes)
        path.pattern_signature = path.get_signature()

    def _compute_semantic_similarity(self, description: str) -> float:
        """计算路径描述与问题的语义相似度"""
        if (
            not description or
            self.embedder is None or
            self._normalized_question_embedding is None
        ):
            return 0.0
        
        try:
            desc_emb = self.embedder.encode_single(description)
            desc_norm = desc_emb / (np.linalg.norm(desc_emb) + 1e-8)
            similarity = float(np.dot(desc_norm, self._normalized_question_embedding))
            return similarity
        except Exception as exc:
            logger.warning("[OptionalMatch] Semantic similarity failed: %s", exc)
            return 0.0
    
    def _extract_node_snapshot(
        self,
        node_value: Any,
        fallback_label: Optional[str],
        fallback_id: Optional[str]
    ) -> Dict[str, Any]:
        """提取节点快照信息（用于三元组返回）"""
        snapshot = {
            'label': fallback_label,
            'id': fallback_id,
            'properties': {}
        }
        
        if node_value is None:
            return snapshot
        
        try:
            snapshot['properties'] = dict(node_value)
        except Exception:
            snapshot['properties'] = {}
        
        try:
            labels = list(node_value.labels)
            if labels:
                snapshot['label'] = labels[0]
        except Exception:
            pass
        
        try:
            node_id = node_value.get('id')
        except Exception:
            node_id = None
        if node_id is None:
            node_id = getattr(node_value, 'id', fallback_id)
        snapshot['id'] = node_id
        
        return snapshot
    
    def _extract_relationship_snapshot(
        self,
        rel_value: Any,
        rel_type: str
    ) -> Dict[str, Any]:
        """提取关系快照信息"""
        snapshot = {
            'type': rel_type,
            'properties': {}
        }
        
        if rel_value is None:
            return snapshot
        
        try:
            snapshot['properties'] = dict(rel_value)
        except Exception:
            snapshot['properties'] = {}
        
        return snapshot


# ============================================================================
# 语义引导的Beam Search（集成平衡方案）
# ============================================================================

class SemanticGuidedBeamSearch:
    """语义引导的Beam Search（集成平衡方案）"""
    
    def __init__(
        self,
        config: RetrievalConfig,
        schema: Dict,
        question: str,
        question_analysis: Optional[Dict],
        embedder: BGEEmbedder,
        path_scorer,
        relation_index: Optional[Dict] = None,
        description_builder: Optional[PathDescriptionBuilder] = None
    ):
        self.config = config
        self.schema = schema
        self.question = question
        self.question_analysis = question_analysis
        self.embedder = embedder
        self.path_scorer = path_scorer
        self.description_builder = description_builder
        
        # 初始化组件（使用平衡方案）
        self.candidate_generator = BalancedCandidateGenerator(
            config, schema, question, question_analysis, relation_index,
            description_builder=description_builder
        )
        
        related_schema = question_analysis.get('related_schema') if question_analysis else None
        self.semantic_filter = SemanticFilter(config, embedder, question, related_schema=related_schema)
        
        question_embedding = getattr(self.semantic_filter, "question_embedding", None)
        self.executor = OptionalMatchExecutor(
            config,
            embedder=embedder,
            question_embedding=question_embedding,
            description_builder=description_builder,
            related_schema=related_schema
        )
        
        logger.info(f"[BeamSearch] 初始化完成（平衡方案）")
    
    def expand(
        self,
        initial_paths: List[PathInstance],
        pool,
        db_name: str
    ) -> List[PathInstance]:
        """
        执行Beam Search扩展
        
        返回扩展后的路径
        """
        beam = initial_paths[:self.config.beam_width]
        all_expanded = []
        
        logger.info(f"[BeamSearch] 开始扩展")
        logger.info(f"  • 初始路径: {len(beam)}")
        logger.info(f"  • 最大深度: {self.config.max_expansion_depth}")
        
        for depth in range(self.config.max_expansion_depth):
            logger.info(f"\n[BeamSearch] ===== 深度 {depth+1}/{self.config.max_expansion_depth} =====")
            current_beam = beam[:]
            current_beam_ids = {p.path_id for p in current_beam}
            
            # 1. 生成候选（平衡方案：开放性 + 智能引导）
            all_candidates = self.candidate_generator.generate_candidates(beam)
            logger.info(
                "[BeamSearch] 深度%d: 生成候选=%d (beam=%d)",
                depth + 1,
                len(all_candidates),
                len(beam)
            )
            
            if not all_candidates:
                logger.info(f"[BeamSearch] 无候选，停止扩展")
                break
            
            # 2. 语义过滤 + prior_score排序
            top_candidates = self.semantic_filter.filter_and_rank(
                all_candidates,
                top_k=self.config.candidate_width
            )
            logger.info(
                "[BeamSearch] 深度%d: 语义筛选后候选=%d",
                depth + 1,
                len(top_candidates)
            )
            
            if not top_candidates:
                logger.info(f"[BeamSearch] 过滤后无候选，停止扩展")
                break
            
            # 3. 执行OPTIONAL MATCH
            base_paths_dict = {p.path_id: p for p in beam}
            expanded_paths = self.executor.execute_candidates(
                top_candidates,
                base_paths_dict,
                pool,
                db_name,
                semantic_filter=self.semantic_filter
            )
            logger.info(
                "[BeamSearch] 深度%d: 执行成功路径=%d",
                depth + 1,
                len(expanded_paths)
            )
            
            if not expanded_paths:
                logger.info(f"[BeamSearch] 无扩展成功，停止")
                break
            
            all_expanded.extend(expanded_paths)
            
            # 4. 选择top-k进入下一轮（父路径与扩展路径共同竞争）
            candidate_pool = current_beam + expanded_paths
            beam = self._select_top_k_for_next_round(
                candidate_pool,
                prev_beam_ids=current_beam_ids
            )
            
            logger.info(f"[BeamSearch] 本轮生成 {len(expanded_paths)} 条路径")
            logger.info(f"[BeamSearch] 下一轮beam: {len(beam)} 条路径")
        
        logger.info(f"\n[BeamSearch] 扩展完成")
        logger.info(f"  • 总扩展路径: {len(all_expanded)}")
        
        return all_expanded
    
    def _select_top_k_for_next_round(
        self,
        paths: List[PathInstance],
        prev_beam_ids: Optional[Set[str]] = None
    ) -> List[PathInstance]:
        """选择top-k路径进入下一轮（保留父路径与子路径的多样性）"""
        prev_ids = prev_beam_ids or set()
        
        # 按签名分组（保持多样性）
        by_signature = defaultdict(list)
        for path in paths:
            sig = path.get_signature()
            by_signature[sig].append(path)
        
        # 从每组选最好的
        selected = []
        def path_score(p: PathInstance) -> float:
            return p.final_score if p.final_score is not None else p.semantic_similarity
        
        for sig, group in by_signature.items():
            group.sort(key=path_score, reverse=True)
            parents = [p for p in group if p.path_id in prev_ids]
            children = [p for p in group if p.path_id not in prev_ids]
            
            signature_selected: List[PathInstance] = []
            if parents:
                signature_selected.append(parents[0])
            if children and len(signature_selected) < 2:
                signature_selected.append(children[0])
            
            for candidate in group:
                if len(signature_selected) >= 2:
                    break
                if candidate not in signature_selected:
                    signature_selected.append(candidate)
            
            selected.extend(signature_selected)
        
        # 总体排序并截断
        selected.sort(key=path_score, reverse=True)
        
        return selected[:self.config.beam_width]


# ============================================================================
# 路径评分器
# ============================================================================

class PathScorer:
    """路径评分器"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
    
    def score_paths(self, paths: List[PathInstance]) -> List[PathInstance]:
        """评分所有路径"""
        
        for path in paths:
            # 计算各维度分数
            semantic_score = path.semantic_similarity
            structural_score = self._calculate_structural_score(path)
            anchor_score = self._calculate_anchor_score(path)
            
            # 加权
            path.scores = {
                'semantic': semantic_score,
                'structural': structural_score,
                'anchor': anchor_score
            }
            
            # 计算final_score
            weights = self.config.path_score_weights
            path.final_score = (
                weights.get('pattern_relevance', 0.25) * semantic_score +
                weights.get('structural_quality', 0.20) * structural_score +
                weights.get('anchor_coverage', 0.30) * anchor_score
            )
        
        return paths
    
    def _calculate_structural_score(self, path: PathInstance) -> float:
        """计算结构分数"""
        # 基于路径长度
        length = path.get_length()
        if length == 0:
            return 0.5
        elif length == 1:
            return 1.0
        elif length == 2:
            return 0.9
        else:
            return 0.8
    
    def _calculate_anchor_score(self, path: PathInstance) -> float:
        """计算锚点分数"""
        if path.anchor_count == 0:
            return 0.0
        elif path.anchor_count == 1:
            return 0.7
        else:
            return 1.0
    
    def select_top_k_paths(self, paths: List[PathInstance], k: int = 20) -> List[PathInstance]:
        """选择top-k路径（保持多样性）"""
        
        if not paths:
            return []
        
        # 先评分
        scored_paths = self.score_paths(paths)
        
        # 按签名分组
        groups = defaultdict(list)
        for path in scored_paths:
            sig = path.get_signature()
            groups[sig].append(path)
        
        # 每组取最好的
        selected = []
        for sig, group in groups.items():
            group.sort(key=lambda p: p.final_score, reverse=True)
            selected.append(group[0])
        
        # 排序并取top-k
        selected.sort(key=lambda p: p.final_score, reverse=True)
        
        return selected[:k]


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
        self.relation_templates = {}  # ⭐ 关系模板映射
        self.template_manager = SchemaTemplateManager(SCHEMA_TEMPLATE_DIR)
        self.description_builder = PathDescriptionBuilder(
            self.template_manager,
            constraint_text_fn=self._constraint_to_text,
            operator_text_map=OPERATOR_TEMPLATES
        )
        
        # 加载完整schema
        self.full_schema = self._load_full_schema() if full_schema_path else None
        
        # ⭐ 加载关系模板
        if self.full_schema:
            self._load_relation_templates()
        
        # 构建关系索引（用于快速查找可能的扩展）
        self.relation_index = self._build_relation_index()
        
        # 初始化embedder
        if config.use_beam_search:
            self.embedder = BGEEmbedder(
                model_name=config.embedding_model,  # 直接使用路径
                device=config.embedding_device
            )
        else:
            self.embedder = None
        
        # 初始化scorer
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
            
            # 构建key: (subj_label, rel_type, obj_label)
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
    
    def _debug_print_path_cypher(
        self,
        path,
        prefix="",
        path_type="Path"
    ):
        """
        Debug模式：打印路径的详细信息
        包括：Cypher查询、自然语言描述、三元组形式
        
        Args:
            path: PathInstance对象
            prefix: 日志前缀
            path_type: 路径类型（如"Initial", "Expanded"）
        """
        if not self.config.debug_print_cypher:
            return
        
        # ========== 1. 构建Cypher查询 ==========
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
                next_node_id = path.nodes[i+1].get('id') if i+1 < len(path.nodes) else None
                
                # 判断方向
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
        
        # ========== 2. 生成自然语言描述 ==========
        nl_description = self._generate_path_description(path)
        
        # ========== 3. 生成三元组形式 ==========
        triples = []
        for i, edge in enumerate(path.edges):
            # 找到真正的from和to节点
            from_node = None
            to_node = None
            for node in path.nodes:
                if node.get('id') == edge.get('from'):
                    from_node = node
                if node.get('id') == edge.get('to'):
                    to_node = node
            
            if from_node and to_node:
                # 获取节点的显示名称
                from_name = self._get_node_name(from_node)
                to_name = self._get_node_name(to_node)
                
                from_display = from_name if from_name else from_node['label']
                to_display = to_name if to_name else to_node['label']
                
                triple = f"({from_display}, {edge.get('rel_type')}, {to_display})"
                triples.append(triple)
        
        # ========== 4. 打印所有信息 ==========
        logger.info(f"\n{prefix}{'='*80}")
        logger.info(f"{prefix}[{path_type}] Path #{path.path_id}")
        logger.info(f"{prefix}{'='*80}")
        
        # Pattern信息
        logger.info(f"{prefix}📊 Pattern: {path.pattern_id} | Score: {path.final_score:.4f} | Anchors: {path.anchor_count}")
        
        # Cypher查询
        logger.info(f"\n{prefix}🔍 Cypher Query:")
        logger.info(f"{prefix}{'─'*80}")
        for line in cypher_lines:
            logger.info(f"{prefix}  {line}")
        logger.info(f"{prefix}{'─'*80}")
        
        # 自然语言描述
        logger.info(f"\n{prefix}💬 Natural Language Description:")
        logger.info(f"{prefix}{'─'*80}")
        logger.info(f"{prefix}  {nl_description}")
        logger.info(f"{prefix}{'─'*80}")
        
        # 三元组形式
        logger.info(f"\n{prefix}🔗 Triple Form:")
        logger.info(f"{prefix}{'─'*80}")
        for i, triple in enumerate(triples, 1):
            logger.info(f"{prefix}  [{i}] {triple}")
        logger.info(f"{prefix}{'─'*80}")
        
        # 节点详情
        logger.info(f"\n{prefix}📦 Nodes Detail:")
        for i, node in enumerate(path.nodes):
            props = node.get('properties', {})
            if props:
                props_str = ", ".join([f"{k}={v}" for k, v in props.items()])
                logger.info(f"{prefix}  [n{i}] {node['label']} {{ {props_str} }}")
            else:
                logger.info(f"{prefix}  [n{i}] {node['label']}")
        
        # 约束详情
        if path.constraints:
            logger.info(f"\n{prefix}⚙️  Constraints:")
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
        """为debug打印构建WHERE子句（避免调用OptionalMatchExecutor）"""
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
                parsed_value = _parse_numeric_value(
                    value,
                    allow_float=(value_type == 'float')
                )
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
        """根据node_id查找节点"""
        if node_id is None:
            return None
        for node in path.nodes:
            if node.get('id') == node_id:
                return node
        return None
    
    def _relation_to_text(self, rel_type: str) -> str:
        """将关系类型转换为自然语言"""
        if rel_type in RELATION_TEMPLATES:
            return RELATION_TEMPLATES[rel_type]
        
        # Fallback: camelCase → space separated
        import re
        words = re.findall('[A-Z][a-z]*|[a-z]+', rel_type)
        return ' '.join(words).lower()
    
    def _property_to_text(self, prop_name: str) -> str:
        """将属性名转换为自然语言"""
        if prop_name in PROPERTY_NAMES:
            return PROPERTY_NAMES[prop_name]
        return prop_name.replace('_', ' ')
    
    def _constraint_to_text(self, constraint: Dict, path: 'PathInstance') -> str:
        """将约束转换为自然语言"""
        prop = constraint.get('property','name')
        op = constraint.get('operator', 'eq')
        value = constraint.get('value')
        value_type = constraint.get('value_type', 'int')
        
        # 属性名转换
        
        prop_text = self._property_to_text(prop)
        
        # 操作符转换
        op_text = OPERATOR_TEMPLATES.get(op, op)
        
        # 值格式化
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
        """
        生成路径的自然语言描述（统一调用PathDescriptionBuilder）
        """
        if not hasattr(self, 'description_builder') or self.description_builder is None:
            return ""
        return self.description_builder.build(
            path,
            use_optional=False,
            include_constraints=True
        )
    
    def _ensure_path_description(self, path: PathInstance) -> str:
        """确保路径具备描述文本（含约束）"""
        if path.description:
            return path.description
        if hasattr(self, 'description_builder') and self.description_builder:
            description = self.description_builder.build(
                path,
                use_optional=False,
                include_constraints=True
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
            path.path_id,
            len(records),
            db_name
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
        """提取节点的name属性"""
        if node.get('properties'):
            # 优先使用properties中的name
            if 'name' in node['properties']:
                return node['properties']['name']
            
            values = list(node['properties'].values())
            if values:
                return values[0]
        
        # 使用display_name作为非约束显示字段
        if 'display_name' in node:
            return node['display_name']
        
        return None  # ⭐ 修复：返回None
    
    def _generate_path_description_fallback(self, path: 'PathInstance') -> str:
        """Fallback: 不使用模板的描述生成"""
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
            full_description = f"{main_description}, where {constraint_description}"
        else:
            full_description = main_description
        
        return full_description
    
    def _compute_semantic_similarity(
        self,
        path_description: str,
        question: str
    ) -> float:
        """计算路径描述与问题的语义相似度"""
        if not self.embedder:
            return 0.5  # 默认值
        
        try:
            import numpy as np
            
            # 编码
            desc_emb = self.embedder.encode_single(path_description)
            question_emb = self.embedder.encode_single(question)
            
            # 余弦相似度
            dot_product = np.dot(desc_emb, question_emb)
            norm_a = np.linalg.norm(desc_emb)
            norm_b = np.linalg.norm(question_emb)
            
            similarity = dot_product / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0.0
            
            # 归一化到[0, 1]
            similarity = (similarity + 1) / 2
            
            return float(similarity)
        
        except Exception as e:
            logger.warning(f"[Semantic] 相似度计算失败: {e}")
            return 0.5
    
    def retrieve(
        self,
        pool,
        db_name: str,
        case: Dict
    ) -> Dict:
        """
        完整检索流程
        
        Args:
            pool: Neo4j连接池
            db_name: 数据库名
            case: {
                'question': str,
                'question_analysis': Dict  # 可选，但推荐提供
            }
        
        Returns:
            {
                'pattern_paths': List[PathInstance],  # Pattern-based路径
                'beam_paths': List[PathInstance],     # Beam Search扩展路径
                'final_paths': List[PathInstance],    # 最终选择的top-k
                'stats': Dict                         # 统计信息
            }
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
        
        all_paths = []
        stats = {}
        
        # ========== Stage 1: Pattern-Based检索 ==========
        pattern_paths = []
        if hasattr(self.config, 'use_pattern_based') and self.config.use_pattern_based:
            logger.info("[Stage 1] Pattern-Based检索...")
            pattern_paths = self._pattern_based_retrieve(pool, db_name, case)
            all_paths.extend(pattern_paths)
            stats['pattern_paths_count'] = len(pattern_paths)
            logger.info(f"[Stage 1] 完成: {len(pattern_paths)} 条路径\n")
        else:
            # 如果没有pattern-based，使用mock初始路径用于演示
            logger.info("[Stage 1] 跳过Pattern-Based（使用mock初始路径）")
            pattern_paths = self._create_mock_initial_paths(question_analysis)
            all_paths.extend(pattern_paths)
            stats['pattern_paths_count'] = len(pattern_paths)
        
        # ========== Stage 2: Beam Search扩展（平衡方案）==========
        beam_paths = []
        if self.config.use_beam_search and pattern_paths and self.embedder:
            logger.info("[Stage 2] Beam Search扩展（平衡方案）...")
            
            beam_search = SemanticGuidedBeamSearch(
                config=self.config,
                schema=self.schema,
                question=question,
                question_analysis=question_analysis,
                embedder=self.embedder,
                path_scorer=self.scorer,
                relation_index=self.relation_index,  # ⭐ 传递relation_index
                description_builder=self.description_builder
            )
            
            beam_paths = beam_search.expand(pattern_paths, pool, db_name)
            all_paths.extend(beam_paths)
            stats['beam_paths_count'] = len(beam_paths)
            logger.info(f"[Stage 2] 完成: {len(beam_paths)} 条扩展路径\n")
        else:
            logger.info(
                "[Stage 2] 跳过Beam Search (use_beam_search=%s, pattern_paths=%d, embedder_ready=%s)\n",
                self.config.use_beam_search,
                len(pattern_paths),
                bool(self.embedder)
            )
        
        # ========== Final: 选择top-k路径 ==========
        logger.info("[Final] 选择最终路径...")
        final_paths = self.scorer.select_top_k_paths(
            all_paths, 
            k=self.config.top_k_paths
        )
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
        import json
        from pathlib import Path
        
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
        """
        构建关系索引用于快速查找
        
        Returns:
            {
                ('River', 'out'): [('flowsThrough', 'Country'), ('flowsInto', 'Lake'), ...],
                ('Lake', 'in'): [('locatedIn', 'Country'), ...],
                ...
            }
        """
        index = defaultdict(list)
        
        # 使用完整schema（如果有）
        if self.full_schema:
            relations = self.full_schema.get('relations', [])
            for rel in relations:
                subj_label = rel.get('subj_label')
                obj_label = rel.get('obj_label')
                rel_type = rel.get('label')
                
                if subj_label and obj_label and rel_type:
                    # 出边: (起始节点标签, 'out') -> [(关系类型, 目标节点标签)]
                    index[(subj_label, 'out')].append((rel_type, obj_label))
                    # 入边: (目标节点标签, 'in') -> [(关系类型, 起始节点标签)]
                    index[(obj_label, 'in')].append((rel_type, subj_label))
        
        # Fallback: 使用简化schema
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
    
    def _pattern_based_retrieve(
        self,
        pool,
        db_name: str,
        case: Dict
    ) -> List[PathInstance]:
        """
        Pattern-Based检索
        
        从case中读取:
        1. retrieved_patterns: 检索到的路径模式
        2. analysis.entities: 实体信息和FAISS候选
        """
        
        # 方法1: 使用retrieved_patterns（优先）
        retrieved_patterns_data = case.get('retrieved_patterns', {})
        
        # 处理两种格式
        if isinstance(retrieved_patterns_data, dict):
            # 格式1: {'patterns': [...], 'top_k': 10, ...}
            retrieved_patterns = retrieved_patterns_data.get('patterns', [])
        elif isinstance(retrieved_patterns_data, list):
            # 格式2: 直接是list
            retrieved_patterns = retrieved_patterns_data
        else:
            retrieved_patterns = []
        
        if retrieved_patterns:
            logger.info(f"[Pattern-Based] 从retrieved_patterns创建初始路径")
            initial_paths = self._create_paths_from_patterns(retrieved_patterns, case)
            if initial_paths:
                logger.info(f"[Pattern-Based] 从patterns创建了 {len(initial_paths)} 条初始路径")
                
                # ⭐ Debug: 打印每条初始路径的Cypher
                if self.config.debug_print_cypher:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"[DEBUG] 初始路径的Cypher查询")
                    logger.info(f"{'='*80}")
                    for i, path in enumerate(initial_paths[:self.config.debug_max_results_display], 1):
                        self._debug_print_path_cypher(path, prefix=f"  [{i}] ", path_type="Initial")
                
                return initial_paths
        
        # 方法2: 使用analysis.entities中的faiss_candidates
        analysis = case.get('analysis', {})
        entities = analysis.get('entities', [])
        
        if entities:
            logger.info(f"[Pattern-Based] 从analysis.entities创建初始路径")
            initial_paths = self._create_paths_from_entities(entities, analysis)
            if initial_paths:
                logger.info(f"[Pattern-Based] 从entities创建了 {len(initial_paths)} 条初始路径")
                return initial_paths
        
        # 方法3: Fallback - 从schema创建
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
        """
        根据entity类型生成约束
        
        支持的类型:
        - str: 字符串（应该在faiss_candidates处理）
        - list(str): 字符串列表
        - date: 日期
        - float: 浮点数
        - int: 整数
        """
        value_type = entity.get('value_type')
        property_name = mapping.get('property')
        comparison = entity.get('comparison', {})
        operator = comparison.get('operator', 'eq')
        value = entity.get('value')
        
        if not property_name:
            logger.warning("[Constraint] Missing property mapping for entity mention '%s'; skipping", entity.get('mention'))
            return None

        if node_label and not self._is_valid_property(node_label, property_name):
            logger.warning(
                "[Constraint] Property %s.%s not defined in schema; skipping constraint",
                node_label,
                property_name
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
                    "[Constraint] Skipping date constraint for value '%s' (unparseable)",
                    value
                )
                return None
            constraint['operator'] = operator
            constraint['value'] = value
            constraint['cypher_fn'] = 'date'
            
            # 检查是否是end_*属性且操作符是gt（晚于）
            assert property_name is not None, "Property name cannot be None for date constraints"
            if 'end' in property_name.lower() and operator in ['gt', 'ge']:
                constraint['allow_null'] = True
                
        elif value_type == 'list(str)':
            # 列表类型
            constraint['operator'] = 'in'
            constraint['value'] = value
            
        elif value_type in ['int', 'float']:
            # 数值类型
            constraint['operator'] = operator
            constraint['value'] = value
            
        elif value_type == 'str':
            # 字符串类型（通常应该在faiss_candidates处理，这里作为fallback）
            constraint['operator'] = 'eq'
            constraint['value'] = value
        
        return constraint
    
    def _parse_cypher_pattern(self, cypher_pattern):
        """
        解析Cypher pattern，提取节点、边和方向
        
        示例: "(n0:River)-[r0:flowsThrough]->(n1:Country)<-[r1:locatedIn]-(n2:Lake)"
        返回: (nodes, edges)
        """
        import re
        
        nodes = []
        edges = []
        
        # 匹配节点: (n0:Label)
        node_pattern = r'\((\w+):(\w+)(?:\s*\{[^}]*\})?\)'
        node_matches = re.findall(node_pattern, cypher_pattern)
        
        for var, label in node_matches:
            nodes.append({
                'label': label,
                'var': var,
                'id': var,  # 使用var作为稳定key
                'properties': {}
            })
        
        # 分割pattern为parts
        parts = re.split(r'(\([^)]+\))', cypher_pattern)
        
        # 遍历parts查找边
        for i, part in enumerate(parts):
            if not part or part.startswith('('):
                continue
            
            if '-[' in part or '<-[' in part:
                rel_match = re.search(r'\[\s*\w+\s*:(\w+)\]', part)
                if not rel_match:
                    continue
                
                rel_type = rel_match.group(1)
                
                # 确定方向（默认无向，只有单侧箭头才决定方向）
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
        """
        从retrieved_patterns创建PathInstance
        
        支持排列组合:
        - String类型: 取top-N faiss_candidates (默认5)
        - 非String类型: 尝试所有mapped_to候选
        - 生成所有可能的组合
        """
        
        all_paths = []
        entities = case.get('analysis', {}).get('entities', [])
        question = case.get('nl_question', '')
        
        # 配置
        top_n_string_candidates = 5
        min_confidence_threshold = 0.3
        
        # 取top-k patterns
        for pattern_info in retrieved_patterns[:self.config.top_k_paths]:
            pattern_id = pattern_info.get('pattern_id', 'unknown')
            cypher_pattern = pattern_info.get('cypher_pattern', '')  # ⭐ 使用cypher_pattern
            scores = pattern_info.get('scores', {})
            
            if not cypher_pattern:
                continue
            
            # ⭐ 解析cypher_pattern获取带方向的nodes和edges
            pattern_nodes, pattern_edges = self._parse_cypher_pattern(cypher_pattern)
            
            if not pattern_nodes or not pattern_edges:
                logger.warning(f"[Pattern {pattern_id}] 解析失败: {cypher_pattern}")
                continue
            
            # ========== Step 1: 收集所有可能的约束组合 ==========
            
            string_options = {}
            constraint_options = {}
            
            # 1.1 收集String类型的选项
            for i, pnode in enumerate(pattern_nodes):  # ⭐ 使用pattern_nodes
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
            
            # 1.2 收集非String类型的选项
            for entity in entities:
                value_type = entity.get('value_type')
                
                if value_type in ['int', 'float', 'date', 'list(str)']:
                    mention = entity.get('mention', '')
                    constraint_options[mention] = []
                    
                    mapped_to = entity.get('mapped_to', [])
                    for mapping in mapped_to:
                        if mapping.get('confidence', 0) < min_confidence_threshold:
                            continue
                        
                        for i, pnode in enumerate(pattern_nodes):  # ⭐ 使用pattern_nodes
                            if mapping.get('label') == pnode['label']:
                                constraint_options[mention].append({
                                    'node_id': pnode.get('id'),
                                    'node_var': pnode.get('var'),
                                    'label': pnode['label'],
                                    'property': mapping.get('property'),
                                    'confidence': mapping.get('confidence', 0.5),
                                    'entity': entity
                                })
            
            # ========== Step 2: 生成所有组合 ==========
            
            # 准备组合的维度
            import itertools
            
            # String组合: 每个节点的可能值
            string_combinations = []
            if string_options:
                # 构建组合列表
                indices = sorted(string_options.keys())
                options_lists = [string_options[idx] for idx in indices]
                
                for combo in itertools.product(*options_lists):
                    string_combinations.append({
                        indices[i]: combo[i] for i in range(len(indices))
                    })
            else:
                string_combinations = [{}]  # 没有string约束
            
            # 非String组合: 每个entity的可能节点
            constraint_combinations = []
            if constraint_options:
                mentions = sorted(constraint_options.keys())
                options_lists = [constraint_options[m] for m in mentions]
                
                for combo in itertools.product(*options_lists):
                    constraint_combinations.append({
                        mentions[i]: combo[i] for i in range(len(mentions))
                    })
            else:
                constraint_combinations = [{}]  # 没有非string约束
            
            # ========== Step 3: 为每个组合创建PathInstance ==========
            
            combo_count = 0
            for string_combo in string_combinations:
                for constraint_combo in constraint_combinations:
                    
                    # ⭐ 构建nodes（使用pattern_nodes）
                    nodes = []
                    for i, pnode in enumerate(pattern_nodes):
                        node = {
                            'label': pnode['label'],
                            'id': pnode.get('id') or pnode.get('var') or f"{pnode['label'].lower()}_{i}",
                            'properties': {},
                            'var': pnode.get('var') or pnode.get('id')
                        }
                        
                        # 应用string约束
                        if i in string_combo:
                            cand = string_combo[i]
                            node['properties'] = {
                                cand['attribute_name']: cand['value']
                            }
                            entity_info = cand.get('entity', {})
                            self._mark_node_as_anchor(node, {
                                'source': 'string_candidate',
                                'mention': entity_info.get('mention'),
                                'value': cand.get('value'),
                                'attribute': cand.get('attribute_name'),
                                'entity_label': entity_info.get('label')
                            })
                        
                        nodes.append(node)
                    
                    # 构建constraints
                    constraints = []
                    for mention, candidate in constraint_combo.items():
                        entity = candidate['entity']
                        mapping = {
                            'property': candidate['property'],
                            'confidence': candidate['confidence']
                        }
                        
                        constraint = self._create_constraint(
                            entity=entity,
                            mapping=mapping,
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
                    
                    # ⭐ 构建edges（使用pattern_edges，保留方向）
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
                    
                    # 计算组合分数
                    combo_score = scores.get('hybrid', 0.5)
                    
                    # String候选的距离惩罚
                    distance_penalty = 0.0
                    for node_idx, cand in string_combo.items():
                        distance = cand.get('distance', 0)
                        distance_penalty += distance * 0.1
                    
                    # Constraint的confidence加权
                    avg_confidence = 1.0
                    if constraint_combo:
                        avg_conf = sum(c['confidence'] for c in constraint_combo.values()) / len(constraint_combo)
                        avg_confidence = avg_conf
                    
                    # 创建PathInstance（先不设置final_score）
                    combo_count += 1
                    path = PathInstance(
                        path_id=f"pattern_{pattern_id}_combo_{combo_count}",
                        nodes=nodes,
                        edges=edges,
                        pattern_id=pattern_id,
                        pattern_score=scores.get('hybrid', 0.5),
                        anchor_count=sum(1 for n in nodes if n.get('properties')),
                        constraints=constraints,
                        final_score=0.0  # 暂时设为0
                    )
                    
                    # ⭐ 生成语义描述并计算相似度
                    path_description = self._generate_path_description(path)
                    semantic_sim = self._compute_semantic_similarity(
                        path_description,
                        question
                    )
                    
                    # ⭐ 综合评分（多维度加权）
                    # 权重分配：pattern(0.25), string_sim(0.20), confidence(0.25), semantic(0.30)
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
        
        paths = []
        
        # 获取relation_candidates
        related_schema = analysis.get('related_schema', {})
        relation_candidates = related_schema.get('relation_candidates', [])
        
        if not relation_candidates:
            return []
        
        # 为每个relation创建路径
        for i, rel_cand in enumerate(relation_candidates[:5]):
            subj_label = rel_cand.get('subj_label', 'Entity')
            obj_label = rel_cand.get('obj_label', 'Entity')
            rel_type = rel_cand.get('type', 'RELATES_TO')
            confidence = rel_cand.get('confidence', 0.5)
            
            # 尝试找到对应的实体
            subj_node = {'label': subj_label, 'id': f'{subj_label.lower()}_{i}', 'properties': {}}
            obj_node = {'label': obj_label, 'id': f'{obj_label.lower()}_{i}', 'properties': {}}
            
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
            
            # 创建路径
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
        
        # 从question_analysis提取实体类型
        entities = question_analysis.get('entities', {})
        anchor_candidates = entities.get('anchor_candidates', [])
        faiss_candidates = entities.get('faiss_candidates', [])
        
        mock_paths = []
        
        # 创建一些mock路径
        if anchor_candidates and faiss_candidates:
            anchor_label = anchor_candidates[0].get('label', 'Entity')
            target_label = faiss_candidates[0].get('label', 'Entity')
            
            path = PathInstance(
                path_id='mock_path_1',
                nodes=[
                    {'label': anchor_label, 'id': 'mock_1'},
                    {'label': target_label, 'id': 'mock_2'}
                ],
                edges=[
                    {'rel_type': 'RELATES_TO'}
                ],
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
    
    # 配置
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
    
    # 初始化完整retriever
    print("\n[1] 初始化SubgraphRetriever...")
    retriever = SubgraphRetriever(config, schema)
    
    # 准备case
    case = {
        'question': "Rivers that flow through at least two countries",
        'question_analysis': {
            'entities': {
                'anchor_candidates': [
                    {'label': 'River'}
                ],
                'faiss_candidates': [
                    {'label': 'Country', 'confidence': 0.9},
                    {'label': 'Lake', 'confidence': 0.6}
                ]
            }
        }
    }
    
    # 执行检索
    print("\n[2] 执行检索...")
    results = retriever.retrieve(
        pool=None,  # Mock
        db_name='geography',
        case=case
    )
    
    print("\n[3] 检索结果:")
    print(f"  • Pattern路径: {len(results['pattern_paths'])}")
    print(f"  • Beam扩展路径: {len(results['beam_paths'])}")
    print(f"  • 最终路径: {len(results['final_paths'])}")
    
    print("\n" + "="*80)
    print("✅ 完整系统测试完成！")
    print("="*80)
