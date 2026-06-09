"""
PathScorer — scores and selects top-k PathInstance objects.

Extracted from subgraph_retrieval.py.

Catalog integration (optional)
-------------------------------
Pass a CatalogResult via ``catalog=`` to enable two additional Eq.2 terms:

  phi_sparse   — exp(-n_hat / mu_P):  penalises data-absent paths
  sel_bonus    — catalog selectivity:  rewards high-selectivity paths

When no catalog is provided (default), these terms are skipped and all
existing scores are unchanged.  Nothing breaks without a catalog.
"""

import logging
import math
from collections import defaultdict
from typing import List, Optional, Any

from retrieval_config import RetrievalConfig
from path_model import PathInstance

logger = logging.getLogger(__name__)


def _phi_sparse_path(n_hat: float, mu_P: float) -> float:
    """Sparsity penalty exp(-n_hat/mu_P) — local thin wrapper for PathScorer.

    Avoids a circular import: PathScorer does not import validate_rank.
    The real phi_sparse() lives in validate_rank.py.
    """
    if mu_P <= 0.0:
        mu_P = 1.0
    if n_hat < 0.0:
        n_hat = 0.0
    return math.exp(-n_hat / mu_P)


class PathScorer:
    """路径评分器

    Optionally integrates catalog-derived sparsity and selectivity terms from
    the paper's Eq.2 into the final_score.  When ``catalog`` is None (the
    default) the scorer behaves exactly as before — existing callers are
    unaffected.
    """

    def __init__(self, config: RetrievalConfig, catalog: Optional[Any] = None):
        """
        Parameters
        ----------
        config : RetrievalConfig
        catalog : CatalogResult, optional
            If provided, phi_sparse and selectivity bonus are added to
            final_score using lambda_sparse from config (default 0.15) and
            sel_weight (default 0.10).
        """
        self.config = config
        self.catalog = catalog

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
            base_score = (
                weights.get('pattern_relevance', 0.25) * semantic_score +
                weights.get('structural_quality', 0.20) * structural_score +
                weights.get('anchor_coverage', 0.30) * anchor_score
            )

            # --- Optional catalog terms (phi_sparse + selectivity) ----------
            if self.catalog is not None:
                catalog_delta = self._catalog_score_delta(path)
                path.scores['catalog_delta'] = catalog_delta
                path.final_score = base_score + catalog_delta
            else:
                path.final_score = base_score

        return paths

    def _catalog_score_delta(self, path: PathInstance) -> float:
        """Compute the catalog-derived score adjustment for a path.

        Returns a *signed* delta to add to the base score:
          +sel_weight * sel  — selectivity bonus (higher sel → more discriminating)
          -lambda_sparse * phi_sparse(n_hat, mu_P)  — sparsity penalty

        When the path has no matching catalog entry the penalty is maximal
        (phi_sparse(0, mu_P) = 1.0) and sel bonus is 0.
        """
        # Look up by canonical signature derived from path node labels + edge types
        sig = self._path_to_catalog_sig(path)
        entry = self.catalog.get(sig) if sig is not None else None

        mu_P: float = float(self.catalog.median_card)

        if entry is not None:
            n_hat = float(entry.card)
            sel = entry.sel
        else:
            n_hat = 0.0
            sel = 0.0

        sp = _phi_sparse_path(n_hat, mu_P)

        # Coefficients: read from config if present, otherwise use paper defaults
        lambda_sparse: float = getattr(self.config, 'lambda_sparse', 0.15)
        sel_weight: float = getattr(self.config, 'sel_weight', 0.10)

        return sel_weight * sel - lambda_sparse * sp

    def _path_to_catalog_sig(self, path: PathInstance):
        """Derive a catalog MetaPathKey tuple from a PathInstance.

        Builds the alternating (label, rel_token, label, ...) key from the
        path's nodes and edges.  Returns None when the path is empty or
        cannot be mapped.
        """
        if not path.nodes or not path.edges:
            if path.nodes and len(path.nodes) == 1:
                # Single-node path — no relations; not in catalog
                return None
            return None

        parts = []
        for i, node in enumerate(path.nodes):
            lbl = node.get('label') or ''
            parts.append(lbl)
            if i < len(path.edges):
                edge = path.edges[i]
                rel_type = edge.get('rel_type') or ''
                direction = edge.get('direction', 'forward')
                if direction == 'forward':
                    parts.append(f"{rel_type}>")
                else:
                    parts.append(f"<{rel_type}")

        return tuple(parts) if len(parts) >= 3 else None

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
