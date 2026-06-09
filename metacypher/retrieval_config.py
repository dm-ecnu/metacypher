"""
RetrievalConfig dataclass and associated lookup tables.

Extracted from subgraph_retrieval.py to keep the configuration
self-contained and importable without pulling in the full pipeline.
"""

import math
from typing import Dict, List
from dataclasses import dataclass, field

from config import EMBED_MODEL_PATH


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
    debug_print_cypher: bool = True

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
    beam_width: int = 20
    candidate_width: int = 40
    max_expansion_depth: int = 2

    # 扩展策略
    expansion_directions: List[str] = field(
        default_factory=lambda: ['forward', 'backward']
    )
    expand_from_all_nodes: bool = True

    # 语义过滤
    use_semantic_filtering: bool = True
    min_semantic_similarity: float = 0.3

    # Schema过滤
    use_schema_filtering: bool = True
    max_relations_per_node: int = 10

    # ========== 平衡方案配置 ==========

    # 智能评分权重
    related_node_bonus: float = 0.3
    in_path_bonus: float = 0.2
    repeated_relation_bonus: float = 0.15

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
