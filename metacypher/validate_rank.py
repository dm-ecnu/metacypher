"""
validate_rank.py — Online ranking algorithms for MetaCypher (Algorithm 2 & 3).

Implements the two-phase ValidateRank pipeline from §4 of the paper:

  PreRank(candidates, catalog, query_intents, probe_budget)
      → sorted list of (candidate, pre_score) keeping only top-P canonical sigs

  phi_sparse(n_hat, mu_P) → float
      Sparsity penalty: exp(-n_hat / mu_P).  Eq.2 term 5.

  validate_rank(candidates, count_fn, catalog, beam, probe_budget, ...)
      → ranked list of RankedCandidate
      Issues bounded, batched COUNT probes via injected count_fn(cypher)->int.
      Memoises by canonical path signature (no repeat probes across calls when
      cache is passed in).  Scores each candidate by the full Eq.2 objective.

  is_eligible(path_sig, query_intents, required_anchors, required_predicates)
      → (bool, list[str])
      Eligibility check: path covers Ω(Q) intents, all anchors, all predicates.
      Returns (True, []) when eligible; (False, missing) otherwise.

Paper reference
---------------
  Eq.2 (§4.3):
    J(s) = phi_desc(s) + phi_cov(s) + lambda * phi_struct(s)
           - lambda_miss * phi_miss(s) - lambda_sparse * phi_sparse(s)

  Algorithm 2 (SelectStructure):  outer loop that calls PreRank then ValidateRank
  Algorithm 3 (ValidateRank):     bounded COUNT probes + full scoring

  phi_sparse(s) = exp(-n_hat(s) / mu_P)
    where mu_P = catalog.median_card (median joint cardinality over non-zero entries)

Public API
----------
  RankedCandidate          — dataclass (sig, n_hat, scores, J, is_eligible, missing)
  phi_sparse(n_hat, mu_P)  — float
  pre_rank(candidates, catalog, query_intents, probe_budget) -> list[RankedCandidate]
  validate_rank(candidates, count_fn, catalog, ...) -> list[RankedCandidate]
  is_eligible(path_sig, query_intents, required_anchors, required_predicates) -> (bool, list)
  compile_count_probe(sig, anchors, predicates) -> str
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# Lazy import: catalog module lives in the same package
# We accept CatalogResult/CatalogEntry by duck typing to avoid a circular dep,
# but import the real classes for isinstance checks when available.
try:
    from catalog import CatalogResult, CatalogEntry
    _CATALOG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CATALOG_AVAILABLE = False
    CatalogResult = Any  # type: ignore[misc,assignment]
    CatalogEntry = Any   # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# A canonical path signature: tuple of alternating labels and direction-tagged
# relation tokens, identical to catalog.MetaPathKey.
PathSig = Tuple[str, ...]

# query_intents: set of node labels / role tokens that the query requires to
# be covered by the selected meta-graph.
IntentSet = Set[str]


# ---------------------------------------------------------------------------
# Default scoring coefficients (paper §4.3, fixed on a validation split)
# ---------------------------------------------------------------------------
LAMBDA_STRUCT: float = 0.20    # binding-continuity reward
LAMBDA_MISS: float = 0.30      # missing-intent penalty
LAMBDA_SPARSE: float = 0.15    # sparsity penalty


# ---------------------------------------------------------------------------
# RankedCandidate
# ---------------------------------------------------------------------------
@dataclass
class RankedCandidate:
    """Result record produced by pre_rank / validate_rank.

    Attributes
    ----------
    sig : PathSig
        Canonical path signature (MetaPathKey-compatible tuple).
    n_hat : float
        Probed (exact) or catalog-estimated support.  0 → data-absent.
    scores : dict
        Per-term breakdown: phi_desc, phi_cov, phi_struct, phi_miss, phi_sparse.
    J : float
        Final aggregate score J(s) from Eq.2.
    probed : bool
        True when n_hat comes from a live COUNT probe; False when from catalog.
    eligible : bool
        Whether this candidate satisfies the eligibility (stopping) condition.
    missing : list[str]
        Intent tokens not yet covered — populated by is_eligible() / validate_rank().
    witnesses : list[Any]
        Bounded witness bindings returned by the COUNT probe (optional).
    meta : dict
        Free-form extra info (catalog entry fields, etc.).
    """
    sig: PathSig
    n_hat: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    J: float = 0.0
    probed: bool = False
    eligible: bool = False
    missing: List[str] = field(default_factory=list)
    witnesses: List[Any] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Eq.2 building blocks
# ---------------------------------------------------------------------------

def phi_sparse(n_hat: float, mu_P: float) -> float:
    """Sparsity penalty term from Eq.2.

    phi_sparse(s) = exp(-n_hat(s) / mu_P)

    Properties
    ----------
    - n_hat = 0  → penalty = exp(0) = 1.0   (maximum penalty for absent structures)
    - n_hat = mu_P → penalty = exp(-1) ≈ 0.368
    - n_hat → ∞  → penalty → 0              (fully supported; no penalty)

    Parameters
    ----------
    n_hat : float
        Probed or estimated cardinality for this candidate.
    mu_P : float
        Median cardinality over non-zero catalog entries (CatalogResult.median_card).
        Must be > 0; caller should use the catalog's fallback value (1.0) when
        all catalog entries are data-absent.

    Returns
    -------
    float in [exp(-n_hat / mu_P), 1.0], strictly in (0, 1].
    """
    if mu_P <= 0.0:
        # Guard: mu_P should always be > 0 (catalog guarantees fallback=1.0)
        mu_P = 1.0
    if n_hat < 0.0:
        n_hat = 0.0
    return math.exp(-n_hat / mu_P)


def _phi_desc(sig: PathSig, query_role_tokens: Sequence[str],
              catalog: Optional[Any]) -> float:
    """Relation-role description fit: phi_desc.

    Measures lexical/token overlap between the path's role_description (from
    the catalog) and the query's role tokens.  Normalised to [0, 1].

    If no catalog is available or the key is missing, falls back to a
    heuristic overlap of the path's own relation tokens.
    """
    if catalog is not None:
        entry = catalog.get(sig)
        if entry is not None:
            role_desc = entry.role_description.lower()
        else:
            role_desc = " ".join(t.lower() for t in sig)
    else:
        role_desc = " ".join(t.lower() for t in sig)

    if not query_role_tokens:
        return 0.5  # neutral when no role tokens are given

    hits = sum(1 for tok in query_role_tokens
               if tok.lower() in role_desc)
    return hits / len(query_role_tokens)


def _phi_cov(sig: PathSig, query_intents: IntentSet,
             placed_anchors: Set[str], placed_predicates: Set[str]) -> float:
    """Anchor/predicate/target-role coverage fraction: phi_cov.

    Fraction of sigma(Q) elements already placed in this candidate path.
    phi_miss = 1 - phi_cov (required target roles get higher weight).

    Denominaor: len(query_intents) + len(placed_anchors) + len(placed_predicates)
    but we only count what's actually required (non-empty union).
    """
    node_labels_in_sig: Set[str] = set(sig[i] for i in range(0, len(sig), 2))
    rel_tokens_in_sig: Set[str] = set(sig[i] for i in range(1, len(sig), 2))
    all_in_sig = node_labels_in_sig | rel_tokens_in_sig

    covered_intents = sum(1 for i in query_intents if i in all_in_sig)
    covered_anchors = sum(1 for a in placed_anchors if a in node_labels_in_sig)
    covered_preds = sum(1 for p in placed_predicates if p in all_in_sig)

    total = len(query_intents) + len(placed_anchors) + len(placed_predicates)
    if total == 0:
        return 1.0
    return (covered_intents + covered_anchors + covered_preds) / total


def _phi_struct(n_hat: float, parent_n_hat: float) -> float:
    """Binding-continuity reward: phi_struct.

    Positive when the extension preserves at least one binding from the parent.
    Value is 1 when n_hat >= parent_n_hat (support is growing / stable),
    shrinks linearly toward 0 as support collapses.

    When parent_n_hat == 0 we cannot compare; return 0.5 (neutral).
    """
    if parent_n_hat <= 0:
        return 0.5
    ratio = min(n_hat / parent_n_hat, 1.0)
    return ratio


def _compute_J(
    phi_desc_val: float,
    phi_cov_val: float,
    phi_struct_val: float,
    phi_miss_val: float,
    phi_sparse_val: float,
    lambda_struct: float = LAMBDA_STRUCT,
    lambda_miss: float = LAMBDA_MISS,
    lambda_sparse: float = LAMBDA_SPARSE,
) -> float:
    """Aggregate Eq.2 score.

    J(s) = phi_desc + phi_cov + lambda * phi_struct
           - lambda_miss * phi_miss - lambda_sparse * phi_sparse
    """
    return (
        phi_desc_val
        + phi_cov_val
        + lambda_struct * phi_struct_val
        - lambda_miss * phi_miss_val
        - lambda_sparse * phi_sparse_val
    )


# ---------------------------------------------------------------------------
# is_eligible
# ---------------------------------------------------------------------------

def is_eligible(
    sig: PathSig,
    query_intents: IntentSet,
    required_anchors: Set[str],
    required_predicates: Set[str],
    n_hat: float = 0.0,
) -> Tuple[bool, List[str]]:
    """Eligibility (stopping) condition from §4.3 of the paper.

    A state is eligible only if its meta-graph:
      1. covers Ω(Q) — all required intent tokens (node labels / role tokens)
      2. contains all bound anchor labels
      3. places every attribute predicate on a compatible type (label)

    Parameters
    ----------
    sig : PathSig
        Canonical path signature.
    query_intents : set[str]
        Ω(Q): set of node labels / relation tokens that MUST be covered.
    required_anchors : set[str]
        Node labels that must appear in sig (bound entity mentions in sigma(Q)).
    required_predicates : set[str]
        Attribute-hosting labels or relation tokens that must be present.
    n_hat : float
        Current support estimate.  An eligible state must have n_hat > 0
        (per paper: "rare candidate may stop when it has non-zero anchored support").

    Returns
    -------
    (eligible, missing)
        eligible : bool
        missing  : list[str] of intent/anchor/predicate tokens not yet covered
    """
    node_labels_in_sig: Set[str] = {sig[i] for i in range(0, len(sig), 2)}
    rel_tokens_in_sig: Set[str] = {sig[i] for i in range(1, len(sig), 2)}
    all_tokens: Set[str] = node_labels_in_sig | rel_tokens_in_sig

    missing: List[str] = []

    # 1. Intent coverage
    for intent in sorted(query_intents):
        if intent not in all_tokens:
            missing.append(f"intent:{intent}")

    # 2. Anchor coverage
    for anchor_label in sorted(required_anchors):
        if anchor_label not in node_labels_in_sig:
            missing.append(f"anchor:{anchor_label}")

    # 3. Predicate hosting
    for pred in sorted(required_predicates):
        if pred not in all_tokens:
            missing.append(f"predicate:{pred}")

    # 4. Non-zero anchored support
    if n_hat <= 0:
        missing.append("support:zero")

    return (len(missing) == 0, missing)


# ---------------------------------------------------------------------------
# Probe compilation
# ---------------------------------------------------------------------------

def compile_count_probe(
    sig: PathSig,
    anchor_bindings: Optional[Dict[int, str]] = None,
    predicate_clauses: Optional[List[str]] = None,
) -> str:
    """Compile a parameterised COUNT probe Cypher for a canonical path signature.

    The probe contains only the typed skeleton + anchor bindings + placed
    predicates — no RETURN projection, aggregation, or ORDER BY.

    Parameters
    ----------
    sig : PathSig
        Canonical path signature, e.g.
        ("River", "FLOWS_THROUGH>", "Country", "<LOCATE_IN", "Lake")
    anchor_bindings : dict[int, str], optional
        Maps node-position index → literal value (name) for WHERE clause.
        E.g. {0: "Natori River"} adds WHERE n0.name = "Natori River".
    predicate_clauses : list[str], optional
        Extra WHERE conditions as raw Cypher fragments, e.g.
        ["n2.area < 390000"].

    Returns
    -------
    str : Cypher COUNT query string.
    """
    node_labels = [sig[i] for i in range(0, len(sig), 2)]
    rel_tokens = [sig[i] for i in range(1, len(sig), 2)]

    # Build MATCH pattern
    parts: List[str] = []
    for i, lbl in enumerate(node_labels):
        parts.append(f"(n{i}:{lbl})")
        if i < len(rel_tokens):
            tok = rel_tokens[i]
            if tok.endswith(">"):
                parts.append(f"-[r{i}:{tok[:-1]}]->")
            elif tok.startswith("<"):
                parts.append(f"<-[r{i}:{tok[1:]}]-")
            else:  # undirected ~
                parts.append(f"-[r{i}:{tok[:-1]}]-")

    match_clause = "MATCH " + "".join(parts)

    # Build WHERE clause
    conditions: List[str] = []
    if anchor_bindings:
        for node_idx, value in anchor_bindings.items():
            # Assume name is the anchor property (standard in MetaCypher)
            safe_val = value.replace('"', '\\"')
            conditions.append(f'n{node_idx}.name = "{safe_val}"')
    if predicate_clauses:
        conditions.extend(predicate_clauses)

    if conditions:
        return match_clause + "\nWHERE " + " AND ".join(conditions) + "\nRETURN count(*) AS c"
    return match_clause + "\nRETURN count(*) AS c"


# ---------------------------------------------------------------------------
# _canonical_probe_key
# ---------------------------------------------------------------------------

def _canonical_probe_key(
    sig: PathSig,
    anchor_bindings: Optional[Dict[int, str]],
    predicate_clauses: Optional[List[str]],
) -> str:
    """Stable string key for memoisation: sig + anchors + predicates."""
    anchor_part = ""
    if anchor_bindings:
        anchor_part = "|".join(f"{k}={v}" for k, v in sorted(anchor_bindings.items()))
    pred_part = ""
    if predicate_clauses:
        pred_part = "|".join(sorted(predicate_clauses))
    return str(sig) + "##" + anchor_part + "##" + pred_part


# ---------------------------------------------------------------------------
# pre_rank
# ---------------------------------------------------------------------------

def pre_rank(
    candidates: Sequence[PathSig],
    catalog: Any,                          # CatalogResult
    query_intents: Optional[IntentSet] = None,
    probe_budget: int = 20,
    query_role_tokens: Optional[Sequence[str]] = None,
    placed_anchors: Optional[Set[str]] = None,
    placed_predicates: Optional[Set[str]] = None,
    lambda_struct: float = LAMBDA_STRUCT,
    lambda_miss: float = LAMBDA_MISS,
    lambda_sparse: float = LAMBDA_SPARSE,
) -> List[RankedCandidate]:
    """PreRank phase (Algorithm 2, line 7): cheap catalog-only ranking.

    Reads catalog entries for each candidate, computes semantic and coverage
    features, and returns the best ``probe_budget`` canonical signatures for
    probing.  Zero-support candidates are NOT removed (paper §4.2: "zero and
    weak support enter as penalties, not hard filters").

    Parameters
    ----------
    candidates : sequence of PathSig
        Candidate canonical signatures to rank.  May contain duplicates;
        they are deduplicated (first occurrence wins).
    catalog : CatalogResult
        Built by build_catalog().  Required.
    query_intents : set[str], optional
        Ω(Q) — required node labels / role tokens.
    probe_budget : int
        Maximum number of distinct signatures to retain for the probe phase.
        Paper recommends P = min(2B², |P(G)|).
    query_role_tokens : sequence[str], optional
        Tokens from the query description used for phi_desc computation.
    placed_anchors : set[str], optional
        Anchor node labels already committed in sigma(Q).
    placed_predicates : set[str], optional
        Attribute-hosting labels / tokens already placed.
    lambda_struct, lambda_miss, lambda_sparse : float
        Eq.2 coefficients.

    Returns
    -------
    list[RankedCandidate], sorted descending by J, length ≤ probe_budget.
    """
    if query_intents is None:
        query_intents = set()
    if query_role_tokens is None:
        query_role_tokens = []
    if placed_anchors is None:
        placed_anchors = set()
    if placed_predicates is None:
        placed_predicates = set()

    mu_P: float = float(catalog.median_card) if catalog is not None else 1.0

    seen_sigs: Set[PathSig] = set()
    ranked: List[RankedCandidate] = []

    for sig in candidates:
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        # Get catalog cardinality estimate
        entry = catalog.get(sig) if catalog is not None else None
        n_hat: float = float(entry.card) if entry is not None else 0.0

        # Compute each Eq.2 term using catalog data only
        pd = _phi_desc(sig, query_role_tokens, catalog)
        pc = _phi_cov(sig, query_intents, placed_anchors, placed_predicates)
        ps = _phi_struct(n_hat, 0.0)   # parent is unknown at pre-rank; neutral
        pm = 1.0 - pc                  # phi_miss = 1 - phi_cov
        psp = phi_sparse(n_hat, mu_P)

        J = _compute_J(pd, pc, ps, pm, psp,
                       lambda_struct, lambda_miss, lambda_sparse)

        # Eligibility check (catalog-level: probed=False, no witnesses yet)
        elig, missing = is_eligible(sig, query_intents, placed_anchors,
                                    placed_predicates, n_hat=n_hat)

        rc = RankedCandidate(
            sig=sig,
            n_hat=n_hat,
            scores={
                "phi_desc": pd,
                "phi_cov": pc,
                "phi_struct": ps,
                "phi_miss": pm,
                "phi_sparse": psp,
            },
            J=J,
            probed=False,
            eligible=elig,
            missing=missing,
            meta={"catalog_card": n_hat, "sel": entry.sel if entry else 0.0},
        )
        ranked.append(rc)

    # Sort descending by J, then by catalog selectivity as tiebreaker
    ranked.sort(key=lambda r: (r.J, r.meta.get("sel", 0.0)), reverse=True)
    return ranked[:probe_budget]


# ---------------------------------------------------------------------------
# validate_rank
# ---------------------------------------------------------------------------

def validate_rank(
    candidates: Sequence[PathSig],
    count_fn: Callable[[str], int],
    catalog: Any,                          # CatalogResult
    beam_width: int = 10,
    probe_budget: int = 20,
    query_intents: Optional[IntentSet] = None,
    query_role_tokens: Optional[Sequence[str]] = None,
    placed_anchors: Optional[Set[str]] = None,
    placed_predicates: Optional[Set[str]] = None,
    anchor_bindings: Optional[Dict[int, str]] = None,
    predicate_clauses: Optional[List[str]] = None,
    probe_cache: Optional[Dict[str, int]] = None,
    parent_n_hat: float = 0.0,
    lambda_struct: float = LAMBDA_STRUCT,
    lambda_miss: float = LAMBDA_MISS,
    lambda_sparse: float = LAMBDA_SPARSE,
) -> List[RankedCandidate]:
    """ValidateRank phase (Algorithm 3): bounded COUNT probes + full scoring.

    For each uncached candidate in the pool, compile its typed skeleton
    (+ current anchors & predicates) into a COUNT query, issue the probe via
    count_fn, and memoize by canonical key.  Score every candidate using the
    full Eq.2 objective (probed n_hat for survivors, catalog synopsis otherwise).

    Parameters
    ----------
    candidates : sequence of PathSig
        Canonical signatures, typically the output of pre_rank().
    count_fn : callable (cypher: str) -> int
        Injected COUNT executor.  May raise — errors produce n_hat=0.
    catalog : CatalogResult
        Catalog for lookup (phi_desc, initial n_hat estimates).
    beam_width : int
        Top-B beam: the returned list is trimmed to beam_width.
    probe_budget : int
        Hard cap on the number of distinct COUNT probes issued this call.
        Memoized hits from probe_cache do NOT count against the budget.
    query_intents, query_role_tokens, placed_anchors, placed_predicates :
        As in pre_rank.
    anchor_bindings : dict[int, str], optional
        Node-position → literal name for WHERE anchoring in probes.
    predicate_clauses : list[str], optional
        Extra raw WHERE fragments added to every probe.
    probe_cache : dict[str, int], optional
        Mutable shared cache (key → count).  If supplied, hits from previous
        calls are reused and new results are written back.  Pass the same dict
        across beam layers to memoize across the full question.
    parent_n_hat : float
        Parent state's n_hat for phi_struct computation.
    lambda_struct, lambda_miss, lambda_sparse : float

    Returns
    -------
    list[RankedCandidate], sorted descending by J, length ≤ beam_width.
    """
    if query_intents is None:
        query_intents = set()
    if query_role_tokens is None:
        query_role_tokens = []
    if placed_anchors is None:
        placed_anchors = set()
    if placed_predicates is None:
        placed_predicates = set()
    if probe_cache is None:
        probe_cache = {}

    mu_P: float = float(catalog.median_card) if catalog is not None else 1.0

    # Deduplicate candidates; preserve order (pre_rank already sorted)
    seen_sigs: Set[PathSig] = set()
    unique_candidates: List[PathSig] = []
    for sig in candidates:
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            unique_candidates.append(sig)

    # ------------------------------------------------------------------
    # Phase 1: identify which signatures need fresh probes
    # ------------------------------------------------------------------
    probe_keys: Dict[PathSig, str] = {}
    for sig in unique_candidates:
        pkey = _canonical_probe_key(sig, anchor_bindings, predicate_clauses)
        probe_keys[sig] = pkey

    uncached_sigs: List[PathSig] = [
        sig for sig in unique_candidates
        if probe_keys[sig] not in probe_cache
    ]
    # Cap by probe_budget
    to_probe = uncached_sigs[:probe_budget]

    # ------------------------------------------------------------------
    # Phase 2: issue batched (sequential) COUNT probes
    # ------------------------------------------------------------------
    for sig in to_probe:
        pkey = probe_keys[sig]
        cypher = compile_count_probe(sig, anchor_bindings, predicate_clauses)
        try:
            count = int(count_fn(cypher))
            if count < 0:
                count = 0
        except Exception:
            count = 0
        probe_cache[pkey] = count

    # ------------------------------------------------------------------
    # Phase 3: score all candidates with full Eq.2 objective
    # ------------------------------------------------------------------
    ranked: List[RankedCandidate] = []

    for sig in unique_candidates:
        pkey = probe_keys[sig]
        entry = catalog.get(sig) if catalog is not None else None

        # Use probed count if available; fall back to catalog estimate
        if pkey in probe_cache:
            n_hat = float(probe_cache[pkey])
            probed = True
        else:
            n_hat = float(entry.card) if entry is not None else 0.0
            probed = False

        # Eq.2 terms
        pd = _phi_desc(sig, query_role_tokens, catalog)
        pc = _phi_cov(sig, query_intents, placed_anchors, placed_predicates)
        ps = _phi_struct(n_hat, parent_n_hat)
        pm = 1.0 - pc
        psp = phi_sparse(n_hat, mu_P)

        J = _compute_J(pd, pc, ps, pm, psp,
                       lambda_struct, lambda_miss, lambda_sparse)

        elig, missing = is_eligible(
            sig, query_intents, placed_anchors, placed_predicates, n_hat=n_hat
        )

        rc = RankedCandidate(
            sig=sig,
            n_hat=n_hat,
            scores={
                "phi_desc": pd,
                "phi_cov": pc,
                "phi_struct": ps,
                "phi_miss": pm,
                "phi_sparse": psp,
            },
            J=J,
            probed=probed,
            eligible=elig,
            missing=missing,
            meta={
                "catalog_card": entry.card if entry else 0,
                "sel": entry.sel if entry else 0.0,
                "probe_key": pkey,
            },
        )
        ranked.append(rc)

    # Sort descending by J; stable sort keeps pre_rank order for ties
    ranked.sort(key=lambda r: r.J, reverse=True)
    return ranked[:beam_width]


# ---------------------------------------------------------------------------
# select_structure  (Algorithm 2 loop — convenience wrapper)
# ---------------------------------------------------------------------------

def select_structure(
    candidate_pool_per_layer: Callable[[], List[PathSig]],
    count_fn: Callable[[str], int],
    catalog: Any,
    beam_width: int = 10,
    depth: int = 3,
    probe_budget: int = 20,
    query_intents: Optional[IntentSet] = None,
    query_role_tokens: Optional[Sequence[str]] = None,
    placed_anchors: Optional[Set[str]] = None,
    placed_predicates: Optional[Set[str]] = None,
    anchor_bindings: Optional[Dict[int, str]] = None,
    predicate_clauses: Optional[List[str]] = None,
    lambda_struct: float = LAMBDA_STRUCT,
    lambda_miss: float = LAMBDA_MISS,
    lambda_sparse: float = LAMBDA_SPARSE,
) -> Tuple[Optional[RankedCandidate], List[RankedCandidate], Dict[str, int]]:
    """Algorithm 2: SelectStructure with ValidateRank across D beam layers.

    This is a convenience wrapper for the outer Algorithm 2 loop.  It drives
    PreRank + ValidateRank per layer, checks the stopping condition, and
    returns the selected meta-graph (or the best partial with annotations).

    Parameters
    ----------
    candidate_pool_per_layer : callable () -> list[PathSig]
        Generates the raw candidate pool for one beam layer.  Called once per
        depth step.  In the full pipeline this would expand from the current
        beam states; here it's injected for testability.
    count_fn, catalog, beam_width, depth, probe_budget :
        As in validate_rank.
    ... remaining as in validate_rank.

    Returns
    -------
    (best, beam, probe_cache)
        best        : RankedCandidate or None.  The selected eligible meta-graph
                      (highest-J eligible), or the highest-J partial if none is
                      eligible.
        beam        : Final beam (top-B RankedCandidates after last layer).
        probe_cache : Shared probe memo accumulated across all layers.
    """
    probe_cache: Dict[str, int] = {}

    beam: List[RankedCandidate] = []
    best_eligible: Optional[RankedCandidate] = None

    for _t in range(depth):
        cands = candidate_pool_per_layer()
        if not cands:
            break

        # PreRank: cheap catalog-only filter to top-P
        pool_sigs = pre_rank(
            cands, catalog,
            query_intents=query_intents,
            probe_budget=probe_budget,
            query_role_tokens=query_role_tokens,
            placed_anchors=placed_anchors,
            placed_predicates=placed_predicates,
            lambda_struct=lambda_struct,
            lambda_miss=lambda_miss,
            lambda_sparse=lambda_sparse,
        )

        parent_n_hat = beam[0].n_hat if beam else 0.0

        # ValidateRank: probed + full scoring
        ranked = validate_rank(
            [rc.sig for rc in pool_sigs],
            count_fn=count_fn,
            catalog=catalog,
            beam_width=beam_width,
            probe_budget=probe_budget,
            query_intents=query_intents,
            query_role_tokens=query_role_tokens,
            placed_anchors=placed_anchors,
            placed_predicates=placed_predicates,
            anchor_bindings=anchor_bindings,
            predicate_clauses=predicate_clauses,
            probe_cache=probe_cache,
            parent_n_hat=parent_n_hat,
            lambda_struct=lambda_struct,
            lambda_miss=lambda_miss,
            lambda_sparse=lambda_sparse,
        )

        beam = ranked[:beam_width]

        # Stopping condition: any beam state covers Ω(Q) + anchors + predicates
        for rc in beam:
            if rc.eligible:
                if best_eligible is None or rc.J > best_eligible.J:
                    best_eligible = rc

        if best_eligible is not None:
            break  # early stop

    # Fallback: return best partial (highest J regardless of eligibility)
    if best_eligible is None and beam:
        best_eligible = beam[0]

    return best_eligible, beam, probe_cache
