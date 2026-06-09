"""
PathScorer — scores and selects top-k PathInstance objects.

Extracted from subgraph_retrieval.py.
"""

import logging
from collections import defaultdict
from typing import List

from retrieval_config import RetrievalConfig
from path_model import PathInstance

logger = logging.getLogger(__name__)


class PathScorer:
    """路径评分器"""

    def __init__(self, config: RetrievalConfig):
        self.config = config

    def score_paths(self, paths: List[PathInstance]) -> List[PathInstance]:
        """评分所有路径"""
        for path in paths:
            semantic_score = path.semantic_similarity
            structural_score = self._calculate_structural_score(path)
            anchor_score = self._calculate_anchor_score(path)

            path.scores = {
                'semantic': semantic_score,
                'structural': structural_score,
                'anchor': anchor_score
            }

            weights = self.config.path_score_weights
            path.final_score = (
                weights.get('pattern_relevance', 0.25) * semantic_score +
                weights.get('structural_quality', 0.20) * structural_score +
                weights.get('anchor_coverage', 0.30) * anchor_score
            )

        return paths

    def _calculate_structural_score(self, path: PathInstance) -> float:
        """计算结构分数"""
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

        scored_paths = self.score_paths(paths)

        groups: dict = defaultdict(list)
        for path in scored_paths:
            sig = path.get_signature()
            groups[sig].append(path)

        selected: List[PathInstance] = []
        for sig, group in groups.items():
            group.sort(key=lambda p: p.final_score, reverse=True)
            selected.append(group[0])

        selected.sort(key=lambda p: p.final_score, reverse=True)
        return selected[:k]
