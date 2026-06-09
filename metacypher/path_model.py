"""
Core data structures and free functions for subgraph retrieval paths.

Extracted from subgraph_retrieval.py:
  - AttributeMatch
  - PathInstance
  - RepetitionInfo
  - ExpansionCandidate
  - compute_related_schema_coverage
  - build_pattern_structure
  - extract_anchor_nodes_from_path
"""

import math
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    constraints: List[Dict] = field(default_factory=list)

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


@dataclass
class RepetitionInfo:
    """重复关系信息"""
    relation_type: str
    min_count: int
    constraint_type: str = 'different_instances'
    specific_values: List[str] = field(default_factory=list)


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
    prior_score: float = 0.0
    semantic_similarity: float = 0.0
    final_score: float = 0.0
    coverage_score: float = 0.0

    # 描述和Cypher
    cypher_pattern: str = ""
    optional_match_clause: str = ""
    pattern_description: str = ""


# ============================================================================
# 路径相关工具函数
# ============================================================================

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
