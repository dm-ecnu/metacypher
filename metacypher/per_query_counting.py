"""Per-query-counting ablation: joint catalog versus per-query counting.

PerQueryCountingCatalog is a drop-in replacement for CatalogResult in
``PathScorer(catalog=)`` / ``SubgraphRetriever(catalog=)``.  It keeps the
scoring formula (phi_sparse sparsity penalty + selectivity bonus) and the
COUNT-probe primitive identical to the catalog path, and changes exactly one
factor: joint support is *measured at lookup time with live COUNT queries,
under a per-query probe budget, and discarded between queries* instead of
being materialized once per schema offline.

This isolates the paper's defining design choice (Section "Joint catalog
versus per-query counting"): with the offline catalog, every candidate's
joint cardinality is a free lookup and mu_P is exact; with per-query
counting, only the first ``probe_budget`` distinct candidates per question
get measured support (the rest score as unknown, i.e. maximal sparsity
penalty), and mu_P is a running median over the few counts observed so far.

Usage::

    pqc = PerQueryCountingCatalog(schema, count_fn, probe_budget=20)
    retriever = SubgraphRetriever(RetrievalConfig(), schema, catalog=pqc)
    for q in questions:
        pqc.start_query()          # reset budget + per-query memo
        ...run retrieval for q...
        stats = pqc.query_stats()  # probes spent on this question

The variant deliberately does NOT cache across queries.  Label populations
(selectivity denominators) are also re-counted per query and each such COUNT
is charged against the same budget.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable, Dict, List, Optional

from catalog import (
    CatalogEntry,
    MetaPathKey,
    _build_anchor_index,
    _compatible_anchor_fields,
    _compile_count_query,
    _compute_selectivity,
    _role_description,
    _safe_count,
)

SchemaDict = Dict[str, Any]


class PerQueryCountingCatalog:
    """Duck-typed stand-in for CatalogResult that counts on demand.

    Implements the two members PathScorer reads — ``get(key)`` and
    ``median_card`` — plus ``anchor_index`` (schema-derived, no counting).
    """

    def __init__(
        self,
        schema: SchemaDict,
        count_fn: Callable[[str], int],
        probe_budget: int = 20,
    ) -> None:
        self._count_fn = count_fn
        self.probe_budget = int(probe_budget)
        self.schema_name = schema.get("name", "")

        # Schema-derived (no instance counting) — same source as build_catalog.
        self._anchor_index_raw = _build_anchor_index(schema)
        self.anchor_index: Dict[str, List[str]] = {
            lbl: info["properties"] for lbl, info in self._anchor_index_raw.items()
        }

        # Per-query state.
        self._memo: Dict[MetaPathKey, Optional[CatalogEntry]] = {}
        self._label_pop: Dict[str, int] = {}
        self._cards_observed: List[int] = []
        self._probes_spent = 0

        # Lifetime instrumentation.
        self.total_probes = 0
        self.total_queries = 0

    # -- per-query lifecycle -------------------------------------------
    def start_query(self) -> None:
        """Discard all measured support; the next question starts cold."""
        self._memo.clear()
        self._label_pop.clear()
        self._cards_observed.clear()
        self._probes_spent = 0
        self.total_queries += 1

    def query_stats(self) -> Dict[str, int]:
        return {
            "probes_spent": self._probes_spent,
            "entries_measured": sum(1 for e in self._memo.values() if e is not None),
            "lookups_over_budget": sum(1 for e in self._memo.values() if e is None),
        }

    # -- CatalogResult interface ----------------------------------------
    @property
    def median_card(self) -> float:
        """mu_P estimated from the counts observed *in this query only*."""
        nonzero = [c for c in self._cards_observed if c > 0]
        if not nonzero:
            return 1.0
        return float(statistics.median(nonzero))

    def get(self, key: MetaPathKey) -> Optional[CatalogEntry]:
        if key in self._memo:
            return self._memo[key]

        # One joint-support COUNT + (up to) two label-population COUNTs.
        if self._probes_spent >= self.probe_budget:
            self._memo[key] = None  # unknown → scorer applies maximal penalty
            return None

        card = _safe_count(self._count_fn, _compile_count_query(key))
        self._charge(1)
        self._cards_observed.append(card)

        node_labels = [key[i] for i in range(0, len(key), 2)]
        rel_tokens = [key[i] for i in range(1, len(key), 2)]
        sel = _compute_selectivity(
            card,
            self._label_population(node_labels[0]),
            self._label_population(node_labels[-1]),
        )

        entry = CatalogEntry(
            key=key,
            node_labels=node_labels,
            rel_tokens=rel_tokens,
            card=card,
            src_distinct=0,
            dst_distinct=0,
            src_avg_degree=0.0,
            dst_avg_degree=0.0,
            sel=sel,
            role_description=_role_description(key),
            anchor_fields=_compatible_anchor_fields(key, self._anchor_index_raw),
        )
        self._memo[key] = entry
        return entry

    # -- internals -------------------------------------------------------
    def _label_population(self, label: str) -> int:
        if label not in self._label_pop:
            if self._probes_spent >= self.probe_budget:
                return 0  # unknown population → sel falls back to 0
            self._label_pop[label] = _safe_count(
                self._count_fn, f"MATCH (n:{label}) RETURN count(n) AS c"
            )
            self._charge(1)
        return self._label_pop[label]

    def _charge(self, n: int) -> None:
        self._probes_spent += n
        self.total_probes += n
