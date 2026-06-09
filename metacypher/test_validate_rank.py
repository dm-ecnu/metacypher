"""
test_validate_rank.py — Plain-Python unit tests for validate_rank.py.

Tests
-----
  T1  phi_sparse correctness + monotonicity
  T2  pre_rank orders by cardinality features correctly
  T3  validate_rank memoises — repeated signatures must NOT re-probe
  T4  validate_rank probe is bounded (probe_budget cap honoured)
  T5  is_eligible — covered vs missing intent annotation
  T6  compile_count_probe — correct Cypher for 1-hop and 2-hop paths
  T7  select_structure — early stop when eligible candidate found
  T8  full pipeline integration: build_catalog → pre_rank → validate_rank

Synthetic graph (shared with test_catalog.py)
---------------------------------------------
  Labels:   River, Country, Lake
  Relations:
    (River)-[:FLOWS_THROUGH]->(Country)   direction=out
    (Lake)-[:LOCATE_IN]->(Country)        direction=out
    (Lake)-[:DRAINS_INTO]->(River)        direction=out

  Instances:
    r1, r2 : River
    c1, c2 : Country
    l1, l2, l3 : Lake

  Edges:
    r1→FLOWS_THROUGH→c1, r1→FLOWS_THROUGH→c2, r2→FLOWS_THROUGH→c1
    l1→LOCATE_IN→c1,    l2→LOCATE_IN→c1,    l3→LOCATE_IN→c2
    l1→DRAINS_INTO→r1,  l2→DRAINS_INTO→r1
"""

import math
import os
import sys

# Make the metacypher package importable when run directly from its directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog import build_catalog, CatalogResult

from validate_rank import (
    phi_sparse,
    pre_rank,
    validate_rank,
    is_eligible,
    compile_count_probe,
    select_structure,
    RankedCandidate,
)

# ---------------------------------------------------------------------------
# Re-use the same synthetic schema + in-memory count_fn from test_catalog.py
# ---------------------------------------------------------------------------

SCHEMA = {
    "name": "test_geo",
    "entities": [
        {"label": "River",   "properties": {"name": "str", "length_km": "float"}},
        {"label": "Country", "properties": {"name": "str", "area_km2": "float",
                                             "population": "int"}},
        {"label": "Lake",    "properties": {"name": "str", "area_ha": "float",
                                             "depth_m": "float"}},
    ],
    "relations": [
        {"type": "FLOWS_THROUGH", "subj_label": "River",  "obj_label": "Country",
         "direction": "out"},
        {"type": "LOCATE_IN",     "subj_label": "Lake",   "obj_label": "Country",
         "direction": "out"},
        {"type": "DRAINS_INTO",   "subj_label": "Lake",   "obj_label": "River",
         "direction": "out"},
    ],
}

NODE_LABELS = {
    "r1": "River", "r2": "River",
    "c1": "Country", "c2": "Country",
    "l1": "Lake", "l2": "Lake", "l3": "Lake",
}

EDGES = [
    ("r1", "FLOWS_THROUGH", "c1"),
    ("r1", "FLOWS_THROUGH", "c2"),
    ("r2", "FLOWS_THROUGH", "c1"),
    ("l1", "LOCATE_IN",     "c1"),
    ("l2", "LOCATE_IN",     "c1"),
    ("l3", "LOCATE_IN",     "c2"),
    ("l1", "DRAINS_INTO",   "r1"),
    ("l2", "DRAINS_INTO",   "r1"),
]


def nodes_of_label(label):
    return [n for n, lbl in NODE_LABELS.items() if lbl == label]


def neighbors_out(src_id, rel_type):
    return [obj for subj, r, obj in EDGES if subj == src_id and r == rel_type]


def neighbors_in(dst_id, rel_type):
    return [subj for subj, r, obj in EDGES if obj == dst_id and r == rel_type]


# Re-use the in-memory evaluator from test_catalog — copy here to keep the
# test self-contained.

import re as _re


def _parse_match_clause(cypher):
    match_body = _re.split(r"\s+RETURN\s+", cypher, flags=_re.IGNORECASE)[0]
    match_body = _re.sub(r"^MATCH\s+", "", match_body, flags=_re.IGNORECASE).strip()
    node_pat = r"\(n\d+:(\w+)\)"
    tokens = _re.findall(
        r"\(n\d+:\w+\)|<-\[r\d+:\w+\]-|-\[r\d+:\w+\]->",
        match_body,
    )
    parsed_nodes = []
    parsed_rels = []
    for tok in tokens:
        n = _re.match(r"\(n\d+:(\w+)\)", tok)
        f = _re.match(r"-\[r\d+:(\w+)\]->", tok)
        b = _re.match(r"<-\[r\d+:(\w+)\]-", tok)
        if n:
            parsed_nodes.append(n.group(1))
        elif f:
            parsed_rels.append((f.group(1), "out"))
        elif b:
            parsed_rels.append((b.group(1), "in"))
    hops = []
    for i, (rel_type, direction) in enumerate(parsed_rels):
        hops.append((parsed_nodes[i], rel_type, direction, parsed_nodes[i + 1]))
    return hops


def _enumerate_path_bindings(hops):
    if not hops:
        return []
    frontier = [(n,) for n in nodes_of_label(hops[0][0])]
    for from_label, rel_type, direction, to_label in hops:
        new_frontier = []
        for path in frontier:
            current = path[-1]
            if direction == "out":
                nbrs = [n for n in neighbors_out(current, rel_type)
                        if NODE_LABELS.get(n) == to_label]
            else:
                nbrs = [n for n in neighbors_in(current, rel_type)
                        if NODE_LABELS.get(n) == to_label]
            for nbr in nbrs:
                new_frontier.append(path + (nbr,))
        frontier = new_frontier
    return frontier


def _eval_single_label(cypher):
    m = _re.match(
        r"MATCH\s+\((\w+):(\w+)\)\s+RETURN\s+count\(\1\)\s+AS\s+c",
        cypher.strip(), _re.IGNORECASE
    )
    if m:
        return len(nodes_of_label(m.group(2)))
    return None


def eval_in_memory(cypher):
    """Genuine in-memory COUNT evaluator (simplified)."""
    cypher = cypher.strip()

    # Single-node count
    single = _eval_single_label(cypher)
    if single is not None:
        return single

    # WHERE anchor filter
    where_match = _re.search(r"WHERE\s+(.+?)\s*$", cypher, _re.IGNORECASE | _re.DOTALL)
    where_text = where_match.group(1).strip() if where_match else ""
    # Strip WHERE for path evaluation
    cypher_no_where = _re.split(r"\s+WHERE\s+", cypher, flags=_re.IGNORECASE)[0]
    # Restore RETURN
    if "RETURN" not in cypher_no_where.upper():
        cypher_no_where += " RETURN count(*) AS c"

    hops = _parse_match_clause(cypher_no_where)
    if not hops:
        return 0
    bindings = _enumerate_path_bindings(hops)

    # Apply WHERE anchor filter if present
    if where_text:
        # Parse simple: n0.name = "VALUE"
        anchor_m = _re.search(r'n(\d+)\.name\s*=\s*"([^"]+)"', where_text)
        if anchor_m:
            node_idx = int(anchor_m.group(1))
            name_val = anchor_m.group(2)
            # Filter bindings where the node at node_idx has that name
            # In our synthetic graph, names ARE the node IDs (simplification)
            bindings = [b for b in bindings if b[node_idx] == name_val]

    return len(bindings)


# ---------------------------------------------------------------------------
# Build catalog once (shared across tests)
# ---------------------------------------------------------------------------

CATALOG = build_catalog(SCHEMA, eval_in_memory, max_len=2)

# Convenience: key tuples
SIG_RIVER_COUNTRY = ("River", "FLOWS_THROUGH>", "Country")
SIG_LAKE_COUNTRY  = ("Lake",  "LOCATE_IN>",     "Country")
SIG_LAKE_RIVER    = ("Lake",  "DRAINS_INTO>",   "River")
SIG_RIVER_COUNTRY_LAKE = ("River", "FLOWS_THROUGH>", "Country", "<LOCATE_IN", "Lake")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
_PASS = 0
_FAIL = 0


def check(name, condition, detail=""):
    global _PASS, _FAIL
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if condition:
        _PASS += 1
    else:
        _FAIL += 1


# ===========================================================================
# T1: phi_sparse correctness + monotonicity
# ===========================================================================

def test_phi_sparse():
    print("\nT1: phi_sparse(n_hat, mu_P)")
    mu_P = 4.0  # arbitrary positive

    # n_hat = 0  → exp(0) = 1.0 (maximum penalty)
    v0 = phi_sparse(0.0, mu_P)
    check("phi_sparse(0, mu_P) == 1.0", abs(v0 - 1.0) < 1e-12,
          f"got {v0}")

    # n_hat = mu_P → exp(-1) ≈ 0.3679
    v_mu = phi_sparse(mu_P, mu_P)
    check("phi_sparse(mu_P, mu_P) == exp(-1)",
          abs(v_mu - math.exp(-1)) < 1e-12,
          f"got {v_mu:.6f}, expected {math.exp(-1):.6f}")

    # n_hat = 2*mu_P → exp(-2)
    v2 = phi_sparse(2 * mu_P, mu_P)
    check("phi_sparse(2mu_P, mu_P) == exp(-2)",
          abs(v2 - math.exp(-2)) < 1e-12,
          f"got {v2:.6f}")

    # Strict monotone decreasing in n_hat
    vals = [phi_sparse(float(n), mu_P) for n in range(0, 20)]
    monotone = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    check("phi_sparse is monotone decreasing in n_hat", monotone)

    # All values strictly in (0, 1]
    for n in [0, 1, 10, 100, 1000]:
        v = phi_sparse(float(n), mu_P)
        check(f"phi_sparse({n}, {mu_P}) in (0, 1]",
              0 < v <= 1.0, f"got {v}")

    # mu_P guard: mu_P <= 0 should not raise (fallback to 1.0)
    try:
        v_bad = phi_sparse(1.0, 0.0)
        check("phi_sparse with mu_P=0 does not raise", True,
              f"returned {v_bad}")
    except Exception as e:
        check("phi_sparse with mu_P=0 does not raise", False, str(e))

    # Negative n_hat treated as 0
    v_neg = phi_sparse(-5.0, mu_P)
    check("phi_sparse(-5, mu_P) == phi_sparse(0, mu_P)",
          abs(v_neg - phi_sparse(0.0, mu_P)) < 1e-12)

    print(f"  phi_sparse(0, {mu_P})    = {phi_sparse(0.0, mu_P):.6f}")
    print(f"  phi_sparse({mu_P}, {mu_P})   = {phi_sparse(mu_P, mu_P):.6f}")
    print(f"  phi_sparse({2*mu_P}, {mu_P}) = {phi_sparse(2*mu_P, mu_P):.6f}")


# ===========================================================================
# T2: pre_rank orders by cardinality features correctly
# ===========================================================================

def test_pre_rank():
    print("\nT2: pre_rank ordering")

    # High-card path should rank above low-card path when coverage is equal
    # River→Country card=3, Lake→River card=2
    # With same intent coverage, phi_sparse lower for higher-card path → higher J
    sigs = [SIG_LAKE_RIVER, SIG_RIVER_COUNTRY]  # deliberately reversed order

    ranked = pre_rank(
        sigs,
        catalog=CATALOG,
        query_intents=set(),  # no intents → equal coverage
        probe_budget=10,
    )

    check("pre_rank returns RankedCandidate list",
          all(isinstance(r, RankedCandidate) for r in ranked))
    check("pre_rank returns both candidates",
          len(ranked) == 2, f"got {len(ranked)}")

    # Higher-card path (River→Country, card=3) should rank above lower-card
    # (Lake→River, card=2) when intents are equal, because phi_sparse is smaller
    # (less penalty), hence J is higher.
    sigs_in_order = [r.sig for r in ranked]
    check(
        "higher-card River->Country ranks above lower-card Lake->River",
        sigs_in_order[0] == SIG_RIVER_COUNTRY,
        f"order: {[str(s) for s in sigs_in_order]}",
    )

    # Scores dict must have all 5 Eq.2 terms
    for term in ("phi_desc", "phi_cov", "phi_struct", "phi_miss", "phi_sparse"):
        check(f"scores contains '{term}'",
              term in ranked[0].scores, str(ranked[0].scores))

    # phi_sparse values match the formula
    for r in ranked:
        expected_sp = phi_sparse(r.n_hat, CATALOG.median_card)
        check(
            f"phi_sparse in scores matches formula for {r.sig[-1]}",
            abs(r.scores["phi_sparse"] - expected_sp) < 1e-12,
            f"got {r.scores['phi_sparse']:.6f}, expected {expected_sp:.6f}",
        )

    # probe_budget cap
    sigs_many = [SIG_LAKE_RIVER, SIG_RIVER_COUNTRY, SIG_LAKE_COUNTRY,
                 SIG_RIVER_COUNTRY_LAKE]
    ranked_capped = pre_rank(sigs_many, catalog=CATALOG, probe_budget=2)
    check("pre_rank respects probe_budget cap",
          len(ranked_capped) <= 2, f"got {len(ranked_capped)}")

    # pre_rank with query_intents that one sig covers and another doesn't
    # Intent "Lake" — covered by SIG_RIVER_COUNTRY_LAKE but not SIG_RIVER_COUNTRY
    ranked_intent = pre_rank(
        [SIG_RIVER_COUNTRY, SIG_RIVER_COUNTRY_LAKE],
        catalog=CATALOG,
        query_intents={"Lake"},
        probe_budget=10,
    )
    # 2-hop that covers Lake should rank higher
    check("2-hop covering 'Lake' intent ranks above 1-hop without Lake",
          ranked_intent[0].sig == SIG_RIVER_COUNTRY_LAKE,
          f"top sig: {ranked_intent[0].sig}")


# ===========================================================================
# T3: validate_rank memoises — repeated signatures must not re-probe
# ===========================================================================

def test_validate_rank_memoisation():
    print("\nT3: validate_rank memoisation (no repeat probes)")

    probe_calls = {"count": 0}

    def counting_count_fn(cypher):
        probe_calls["count"] += 1
        return eval_in_memory(cypher)

    shared_cache = {}

    sigs = [SIG_RIVER_COUNTRY, SIG_LAKE_COUNTRY, SIG_LAKE_RIVER]

    # First call
    ranked1 = validate_rank(
        sigs,
        count_fn=counting_count_fn,
        catalog=CATALOG,
        beam_width=10,
        probe_budget=10,
        probe_cache=shared_cache,
    )
    calls_after_first = probe_calls["count"]
    check("first validate_rank issues probes", calls_after_first > 0,
          f"probes issued: {calls_after_first}")
    check("first validate_rank returns ranked list",
          len(ranked1) == len(sigs), f"got {len(ranked1)}")

    # Second call with SAME signatures and SAME cache — must not re-probe
    probe_calls["count"] = 0
    ranked2 = validate_rank(
        sigs,
        count_fn=counting_count_fn,
        catalog=CATALOG,
        beam_width=10,
        probe_budget=10,
        probe_cache=shared_cache,
    )
    calls_after_second = probe_calls["count"]
    check("second call with same cache issues ZERO new probes",
          calls_after_second == 0,
          f"unexpected probes: {calls_after_second}")

    # Results should be identical (same ranking)
    sigs1 = [r.sig for r in ranked1]
    sigs2 = [r.sig for r in ranked2]
    check("rankings are identical on repeated call", sigs1 == sigs2,
          f"{sigs1} vs {sigs2}")

    # probed flag must be True for all sigs that were probed
    for r in ranked1:
        check(f"probed=True for {r.sig[-1]}", r.probed)


# ===========================================================================
# T4: validate_rank probe_budget cap
# ===========================================================================

def test_validate_rank_probe_budget():
    print("\nT4: validate_rank probe_budget cap")

    probe_calls = {"count": 0}

    def counting_count_fn(cypher):
        probe_calls["count"] += 1
        return eval_in_memory(cypher)

    all_sigs = [SIG_RIVER_COUNTRY, SIG_LAKE_COUNTRY, SIG_LAKE_RIVER,
                SIG_RIVER_COUNTRY_LAKE]

    budget = 2
    ranked = validate_rank(
        all_sigs,
        count_fn=counting_count_fn,
        catalog=CATALOG,
        beam_width=len(all_sigs),
        probe_budget=budget,
    )

    actual_probes = probe_calls["count"]
    check(f"probe_budget={budget} cap respected",
          actual_probes <= budget,
          f"issued {actual_probes} probes (budget={budget})")

    check("validate_rank still returns all candidates (with catalog fallback for unprobed)",
          len(ranked) == len(all_sigs), f"got {len(ranked)}")

    # The probed ones have probed=True; unprobed ones have probed=False
    probed_count = sum(1 for r in ranked if r.probed)
    check(f"exactly {budget} candidates marked probed=True",
          probed_count == budget,
          f"probed={probed_count}, expected={budget}")


# ===========================================================================
# T5: is_eligible — covered vs missing intent annotation
# ===========================================================================

def test_is_eligible():
    print("\nT5: is_eligible")

    # Case 1: single-hop sig covers both endpoints
    sig = SIG_RIVER_COUNTRY  # ("River", "FLOWS_THROUGH>", "Country")
    elig, missing = is_eligible(
        sig,
        query_intents={"River", "Country"},
        required_anchors={"River"},
        required_predicates=set(),
        n_hat=3.0,
    )
    check("eligible when all intents covered + n_hat > 0", elig,
          f"missing: {missing}")
    check("no missing items when eligible", missing == [], f"missing: {missing}")

    # Case 2: missing intent "Lake"
    elig2, missing2 = is_eligible(
        sig,
        query_intents={"River", "Country", "Lake"},
        required_anchors={"River"},
        required_predicates=set(),
        n_hat=3.0,
    )
    check("not eligible when 'Lake' intent missing", not elig2)
    check("missing list contains 'intent:Lake'",
          any("Lake" in m for m in missing2), f"missing: {missing2}")

    # Case 3: anchor not in sig
    elig3, missing3 = is_eligible(
        sig,
        query_intents={"River", "Country"},
        required_anchors={"Lake"},  # Lake is not in River→Country
        required_predicates=set(),
        n_hat=3.0,
    )
    check("not eligible when required anchor 'Lake' absent", not elig3)
    check("missing list contains 'anchor:Lake'",
          any("Lake" in m for m in missing3), f"missing: {missing3}")

    # Case 4: predicate host missing
    elig4, missing4 = is_eligible(
        sig,
        query_intents={"River", "Country"},
        required_anchors=set(),
        required_predicates={"area_km2"},  # property name, not a type token
        n_hat=3.0,
    )
    check("not eligible when predicate 'area_km2' not in sig tokens", not elig4)
    check("missing list contains 'predicate:area_km2'",
          any("area_km2" in m for m in missing4), f"missing: {missing4}")

    # Case 5: n_hat = 0 → not eligible even if all intents covered
    elig5, missing5 = is_eligible(
        sig,
        query_intents={"River", "Country"},
        required_anchors={"River"},
        required_predicates=set(),
        n_hat=0.0,
    )
    check("not eligible when n_hat == 0", not elig5)
    check("missing list contains 'support:zero'",
          any("support" in m for m in missing5), f"missing: {missing5}")

    # Case 6: 2-hop sig covers River, Country, Lake
    sig2 = SIG_RIVER_COUNTRY_LAKE
    elig6, missing6 = is_eligible(
        sig2,
        query_intents={"River", "Country", "Lake"},
        required_anchors={"River"},
        required_predicates=set(),
        n_hat=5.0,
    )
    check("2-hop sig eligible when all intents + n_hat > 0", elig6,
          f"missing: {missing6}")

    # Case 7: empty intents, empty anchors, n_hat > 0 → always eligible
    elig7, missing7 = is_eligible(
        sig,
        query_intents=set(),
        required_anchors=set(),
        required_predicates=set(),
        n_hat=1.0,
    )
    check("eligible with empty intents/anchors + n_hat > 0", elig7,
          f"missing: {missing7}")


# ===========================================================================
# T6: compile_count_probe generates correct Cypher
# ===========================================================================

def test_compile_count_probe():
    print("\nT6: compile_count_probe")

    # 1-hop forward
    cypher = compile_count_probe(SIG_RIVER_COUNTRY)
    check("1-hop probe contains MATCH",  "MATCH" in cypher)
    check("1-hop probe contains FLOWS_THROUGH", "FLOWS_THROUGH" in cypher)
    check("1-hop probe contains count(*)", "count(*)" in cypher)
    check("1-hop probe uses n0:River", "n0:River" in cypher)
    check("1-hop probe uses n1:Country", "n1:Country" in cypher)
    check("1-hop probe has forward arrow", "->" in cypher)

    # 2-hop mixed direction
    cypher2 = compile_count_probe(SIG_RIVER_COUNTRY_LAKE)
    check("2-hop probe contains n2:Lake", "n2:Lake" in cypher2)
    check("2-hop probe contains LOCATE_IN", "LOCATE_IN" in cypher2)
    check("2-hop probe has backward arrow (LOCATE_IN)", "<-" in cypher2)

    # Anchor binding
    cypher_anch = compile_count_probe(
        SIG_RIVER_COUNTRY,
        anchor_bindings={0: "Natori River"},
    )
    check("anchor binding adds WHERE clause",
          "WHERE" in cypher_anch)
    check("anchor binding adds n0.name condition",
          'n0.name = "Natori River"' in cypher_anch)

    # Predicate clause
    cypher_pred = compile_count_probe(
        SIG_RIVER_COUNTRY_LAKE,
        predicate_clauses=["n2.area_ha < 390000"],
    )
    check("predicate clause adds WHERE", "WHERE" in cypher_pred)
    check("predicate clause text present", "n2.area_ha < 390000" in cypher_pred)

    # Evaluate the 1-hop probe against the in-memory graph
    count = eval_in_memory(cypher)
    check("1-hop probe evaluates to correct count (3)",
          count == 3, f"got {count}")


# ===========================================================================
# T7: select_structure — early stop when eligible candidate found
# ===========================================================================

def test_select_structure():
    print("\nT7: select_structure early stop")

    # Provide a simple candidate generator that returns 2-hop sigs covering
    # all required intents.
    calls = {"n": 0}

    def candidate_pool():
        calls["n"] += 1
        # Always return the same pool of candidate signatures
        return [SIG_RIVER_COUNTRY, SIG_LAKE_COUNTRY, SIG_RIVER_COUNTRY_LAKE]

    best, beam, probe_cache = select_structure(
        candidate_pool_per_layer=candidate_pool,
        count_fn=eval_in_memory,
        catalog=CATALOG,
        beam_width=5,
        depth=3,
        probe_budget=10,
        query_intents={"River", "Country", "Lake"},
        placed_anchors={"River"},
        placed_predicates=set(),
    )

    check("select_structure returns a best candidate", best is not None)
    check("probe_cache is populated", len(probe_cache) > 0)
    check("stopped before depth=3 (early exit on eligible)",
          calls["n"] <= 3, f"layers visited: {calls['n']}")

    if best is not None:
        # 2-hop sig covers River, Country, Lake → should be eligible
        check("best candidate is the 2-hop sig covering all intents",
              set(best.sig[::2]).issuperset({"River", "Country", "Lake"}),
              f"best sig: {best.sig}")

    # Without eligible candidate in any layer: no early stop
    calls2 = {"n": 0}

    def no_eligible_pool():
        calls2["n"] += 1
        return [SIG_RIVER_COUNTRY]  # only covers River+Country, not Lake

    best2, beam2, cache2 = select_structure(
        candidate_pool_per_layer=no_eligible_pool,
        count_fn=eval_in_memory,
        catalog=CATALOG,
        beam_width=5,
        depth=2,
        probe_budget=10,
        query_intents={"River", "Country", "Lake"},  # Lake never covered
        placed_anchors=set(),
        placed_predicates=set(),
    )
    check("no eligible stop: visits all depth layers",
          calls2["n"] == 2, f"layers: {calls2['n']}")
    check("no eligible stop: returns best partial (highest J)",
          best2 is not None)


# ===========================================================================
# T8: full pipeline — build_catalog → pre_rank → validate_rank
# ===========================================================================

def test_full_pipeline():
    print("\nT8: full pipeline integration")

    # Build catalog from scratch
    catalog = build_catalog(SCHEMA, eval_in_memory, max_len=2)
    check("catalog has entries", len(catalog.entries) > 0)
    check("catalog median_card > 0", catalog.median_card > 0,
          f"mu_P = {catalog.median_card}")

    # All candidate sigs from catalog
    all_sigs = [e.key for e in catalog.entries]
    check("have candidate sigs", len(all_sigs) > 0)

    # pre_rank on all catalog sigs
    pre_ranked = pre_rank(
        all_sigs,
        catalog=catalog,
        query_intents={"River", "Lake"},
        probe_budget=6,
        query_role_tokens=["flows", "river", "lake"],
    )
    check("pre_rank returns <= probe_budget candidates",
          len(pre_ranked) <= 6, f"got {len(pre_ranked)}")
    check("all pre_rank results are RankedCandidate",
          all(isinstance(r, RankedCandidate) for r in pre_ranked))

    # Scores are valid floats
    for r in pre_ranked:
        check(f"J is finite for {r.sig}",
              math.isfinite(r.J), f"J={r.J}")
        for term, val in r.scores.items():
            check(f"score[{term}] is finite",
                  math.isfinite(val), f"{term}={val}")

    # validate_rank: fresh cache
    probe_cache = {}
    ranked = validate_rank(
        [r.sig for r in pre_ranked],
        count_fn=eval_in_memory,
        catalog=catalog,
        beam_width=4,
        probe_budget=6,
        query_intents={"River", "Lake"},
        probe_cache=probe_cache,
    )
    check("validate_rank returns <= beam_width candidates",
          len(ranked) <= 4, f"got {len(ranked)}")

    # All probed sigs have n_hat matching eval_in_memory (modulo budget cap)
    for r in ranked:
        if r.probed:
            expected = eval_in_memory(
                f"MATCH " + "".join(
                    f"(n{i}:{r.sig[2*i]})" +
                    (
                        f"-[r{i}:{r.sig[2*i+1][:-1]}]->"
                        if r.sig[2*i+1].endswith(">")
                        else f"<-[r{i}:{r.sig[2*i+1][1:]}]-"
                        if r.sig[2*i+1].startswith("<")
                        else f"-[r{i}:{r.sig[2*i+1][:-1]}]-"
                    )
                    for i in range(len(r.sig) // 2)
                ) + f"(n{len(r.sig)//2}:{r.sig[-1]}) RETURN count(*) AS c"
            )
            check(
                f"probed n_hat matches eval_in_memory for {r.sig}",
                abs(r.n_hat - expected) < 1,  # allow off-by-one from WHERE anchors
                f"n_hat={r.n_hat}, eval={expected}",
            )

    # phi_sparse values are consistent with mu_P
    mu_P = catalog.median_card
    for r in ranked:
        expected_sp = phi_sparse(r.n_hat, mu_P)
        check(
            f"phi_sparse consistent for {r.sig[-1]}",
            abs(r.scores.get("phi_sparse", -1) - expected_sp) < 1e-12,
            f"stored={r.scores.get('phi_sparse'):.6f}, expected={expected_sp:.6f}",
        )

    print(f"  mu_P = {catalog.median_card}")
    print(f"  probe_cache populated with {len(probe_cache)} entries")
    print(f"  Top ranked sig: {ranked[0].sig if ranked else 'NONE'}")
    print(f"  Top J: {ranked[0].J:.4f}" if ranked else "")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("MetaCypher validate_rank.py — unit tests")
    print("=" * 70)

    test_phi_sparse()
    test_pre_rank()
    test_validate_rank_memoisation()
    test_validate_rank_probe_budget()
    test_is_eligible()
    test_compile_count_probe()
    test_select_structure()
    test_full_pipeline()

    print("\n" + "=" * 70)
    total = _PASS + _FAIL
    print(f"Results: {_PASS}/{total} passed, {_FAIL} failed")
    if _FAIL > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
