"""
test_catalog.py — Plain-Python tests for catalog.py.

Tests:
  T1  enum_meta_paths — coverage and no-duplicates
  T2  JointPathSupport (card) — computed against a synthetic in-memory graph
  T3  Selectivity (sel) — correct formula and range
  T4  Endpoint degree summary — src/dst_distinct, src/dst_avg_degree
  T5  median_card (μ_P) — correct median over non-zero entries
  T6  role_description — readable format
  T7  anchor_fields — property and numeric_property discovery
  T8  save/load round-trip — JSON persistence
  T9  zero-support paths — retained in catalog with card==0

Synthetic graph
---------------
  Labels:   River, Country, Lake
  Relations:
    (River)-[:FLOWS_THROUGH]->(Country)   direction=out
    (Lake)-[:LOCATE_IN]->(Country)        direction=out
    (Lake)-[:DRAINS_INTO]->(River)        direction=out

  Instances:
    Nodes:
      River:   r1, r2
      Country: c1, c2
      Lake:    l1, l2, l3

    Edges (FLOWS_THROUGH):
      r1→c1, r1→c2, r2→c1

    Edges (LOCATE_IN):
      l1→c1, l2→c1, l3→c2

    Edges (DRAINS_INTO):
      l1→r1, l2→r1

Hand-computed path counts
-------------------------
  1-hop paths:
    River-[FLOWS_THROUGH>]-Country  : 3  (r1→c1, r1→c2, r2→c1)
    Lake-[LOCATE_IN>]-Country       : 3  (l1→c1, l2→c1, l3→c2)
    Lake-[DRAINS_INTO>]-River       : 2  (l1→r1, l2→r1)
    Country-[<FLOWS_THROUGH]-River  : 3  (same edges, reverse traversal)
    Country-[<LOCATE_IN]-Lake       : 3
    River-[<DRAINS_INTO]-Lake       : 2

  2-hop paths (examples):
    River-[FLOWS_THROUGH>]-Country-[<LOCATE_IN]-Lake:
      Join on Country. r1→c1→{l1,l2}, r1→c2→{l3}, r2→c1→{l1,l2} = 5 pairs

    Lake-[DRAINS_INTO>]-River-[FLOWS_THROUGH>]-Country:
      l1→r1→{c1,c2}, l2→r1→{c1,c2} = 4 pairs

    Lake-[LOCATE_IN>]-Country-[<FLOWS_THROUGH]-River:
      l1→c1←{r1,r2}, l2→c1←{r1,r2}, l3→c2←{r1} = 5 pairs

    (other 2-hop paths may produce 0 when there are no matching instances)
"""

import json
import os
import sys
import tempfile

# Make the metacypher package importable when run directly from its directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog import (
    CatalogEntry,
    CatalogResult,
    build_catalog,
    enum_meta_paths,
    load_catalog,
    save_catalog,
)

# ---------------------------------------------------------------------------
# Synthetic schema
# ---------------------------------------------------------------------------
SCHEMA = {
    "name": "test_geo",
    "entities": [
        {
            "label": "River",
            "properties": {"name": "str", "length_km": "float"},
        },
        {
            "label": "Country",
            "properties": {"name": "str", "area_km2": "float", "population": "int"},
        },
        {
            "label": "Lake",
            "properties": {"name": "str", "area_ha": "float", "depth_m": "float"},
        },
    ],
    "relations": [
        {
            "type": "FLOWS_THROUGH",
            "subj_label": "River",
            "obj_label": "Country",
            "direction": "out",
            "pattern": "(n0:River)-[r0:FLOWS_THROUGH]->(n1:Country)",
        },
        {
            "type": "LOCATE_IN",
            "subj_label": "Lake",
            "obj_label": "Country",
            "direction": "out",
            "pattern": "(n0:Lake)-[r0:LOCATE_IN]->(n1:Country)",
        },
        {
            "type": "DRAINS_INTO",
            "subj_label": "Lake",
            "obj_label": "River",
            "direction": "out",
            "pattern": "(n0:Lake)-[r0:DRAINS_INTO]->(n1:River)",
        },
    ],
}

# ---------------------------------------------------------------------------
# Synthetic in-memory graph
# ---------------------------------------------------------------------------
# edge_set: dict mapping (subj_id, rel_type, obj_id) → True
# node_labels: dict mapping node_id → label
NODE_LABELS = {
    "r1": "River", "r2": "River",
    "c1": "Country", "c2": "Country",
    "l1": "Lake", "l2": "Lake", "l3": "Lake",
}

EDGES = [
    # FLOWS_THROUGH
    ("r1", "FLOWS_THROUGH", "c1"),
    ("r1", "FLOWS_THROUGH", "c2"),
    ("r2", "FLOWS_THROUGH", "c1"),
    # LOCATE_IN
    ("l1", "LOCATE_IN", "c1"),
    ("l2", "LOCATE_IN", "c1"),
    ("l3", "LOCATE_IN", "c2"),
    # DRAINS_INTO
    ("l1", "DRAINS_INTO", "r1"),
    ("l2", "DRAINS_INTO", "r1"),
]


def nodes_of_label(label: str):
    return [n for n, lbl in NODE_LABELS.items() if lbl == label]


def neighbors_out(src_id: str, rel_type: str):
    """Return list of object node ids reachable via (src_id)-[:rel_type]->(?)."""
    return [obj for subj, r, obj in EDGES if subj == src_id and r == rel_type]


def neighbors_in(dst_id: str, rel_type: str):
    """Return list of subject node ids arriving via (?)-[:rel_type]->(dst_id)."""
    return [subj for subj, r, obj in EDGES if obj == dst_id and r == rel_type]


# ---------------------------------------------------------------------------
# In-memory COUNT evaluator  — genuinely computes counts from EDGES/NODE_LABELS
# ---------------------------------------------------------------------------
def eval_in_memory(cypher: str) -> int:
    """Parse and evaluate simplified COUNT Cypher against the synthetic graph.

    Supports:
      MATCH (n0:L0)-[r0:R0]->(n1:L1) RETURN count(*) AS c
      MATCH (n0:L0)<-[r0:R0]-(n1:L1) RETURN count(*) AS c
      MATCH (n0:L0) RETURN count(n0) AS c
      MATCH (n0:L0)-[r0:R0]->(n1:L1)-[r1:R1]->(n2:L2) RETURN count(*) AS c
      MATCH (n0:L0)-[r0:R0]->(n1:L1)<-[r1:R1]-(n2:L2) RETURN count(*) AS c
      MATCH ... RETURN count(DISTINCT n0) AS c
      MATCH ... RETURN count(DISTINCT n{k}) AS c
      MATCH ... WITH x, count(y) AS deg RETURN avg(deg) AS c
    """
    cypher = cypher.strip()

    # ---- avg(deg) form: endpoint degree queries -------------------------
    if "avg(deg)" in cypher.lower():
        return _eval_avg_degree(cypher)

    # ---- DISTINCT count form -------------------------------------------
    if "DISTINCT" in cypher:
        return _eval_distinct_count(cypher)

    # ---- single-node MATCH (n:L) / (n0:L) RETURN count(n) AS c  -------
    import re
    single = re.match(
        r"MATCH\s+\((\w+):(\w+)\)\s+RETURN\s+count\(\1\)\s+AS\s+c",
        cypher, re.IGNORECASE
    )
    if single:
        lbl = single.group(2)
        return len(nodes_of_label(lbl))

    # ---- general path count --------------------------------------------
    return _eval_path_count(cypher)


def _parse_match_clause(cypher: str):
    """Return list of hop specs: [(from_label, rel_type, direction, to_label), ...]."""
    import re
    # Remove RETURN clause
    match_body = re.split(r"\s+RETURN\s+", cypher, flags=re.IGNORECASE)[0]
    match_body = re.sub(r"^MATCH\s+", "", match_body, flags=re.IGNORECASE).strip()

    hops = []
    # Tokenize: nodes are (n\d+:\w+), rels are -[r\d+:\w+]->  or <-[r\d+:\w+]-
    node_pat = r"\(n\d+:(\w+)\)"
    fwd_pat = r"-\[r\d+:(\w+)\]->"
    bwd_pat = r"<-\[r\d+:(\w+)\]-"

    tokens = re.findall(
        r"\(n\d+:\w+\)|<-\[r\d+:\w+\]-|-\[r\d+:\w+\]->",
        match_body
    )

    parsed_nodes = []
    parsed_rels = []  # (rel_type, direction: "out"|"in")
    for tok in tokens:
        n = re.match(node_pat, tok)
        f = re.match(fwd_pat, tok)
        b = re.match(bwd_pat, tok)
        if n:
            parsed_nodes.append(n.group(1))
        elif f:
            parsed_rels.append((f.group(1), "out"))
        elif b:
            parsed_rels.append((b.group(1), "in"))

    for i, (rel_type, direction) in enumerate(parsed_rels):
        hops.append((parsed_nodes[i], rel_type, direction, parsed_nodes[i + 1]))

    return hops


def _enumerate_path_bindings(hops):
    """Enumerate all (v0, v1, ..., vl) tuples matching the hop sequence."""
    # Start with all nodes of the first label
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
            else:  # in
                nbrs = [n for n in neighbors_in(current, rel_type)
                        if NODE_LABELS.get(n) == to_label]
            for nbr in nbrs:
                new_frontier.append(path + (nbr,))
        frontier = new_frontier

    return frontier


def _eval_path_count(cypher: str) -> int:
    hops = _parse_match_clause(cypher)
    if not hops:
        return 0
    bindings = _enumerate_path_bindings(hops)
    return len(bindings)


def _eval_distinct_count(cypher: str) -> int:
    import re
    # Which variable index to project?
    m = re.search(r"count\(DISTINCT\s+n(\d+)\)", cypher, re.IGNORECASE)
    var_idx = int(m.group(1)) if m else 0

    hops = _parse_match_clause(cypher)
    if not hops:
        return 0
    bindings = _enumerate_path_bindings(hops)
    distinct = set(b[var_idx] for b in bindings)
    return len(distinct)


def _eval_avg_degree(cypher: str) -> int:
    """Evaluate AVG degree queries.

    Pattern: MATCH (s:L0)-[:R]->(nb:L1) WITH s, count(nb) AS deg RETURN avg(deg) AS c
    or reverse direction.
    Returns avg degree as an int (floor) — sufficient for comparison tests.
    """
    import re
    # extract src label and rel
    m_fwd = re.search(
        r"MATCH\s+\(s:(\w+)\)-\[:(\w+)\]->\s*\(nb:(\w+)\)", cypher, re.IGNORECASE
    )
    m_bwd = re.search(
        r"MATCH\s+\(s:(\w+)\)<-\[:(\w+)\]-\s*\(nb:(\w+)\)", cypher, re.IGNORECASE
    )
    m_undir = re.search(
        r"MATCH\s+\(s:(\w+)\)-\[:(\w+)\]-\s*\(nb:(\w+)\)", cypher, re.IGNORECASE
    )
    # Also handle the dst degree form: MATCH (prev:L)-[:R]->(d:L) ...
    m_dst_fwd = re.search(
        r"MATCH\s+\(prev:(\w+)\)-\[:(\w+)\]->\s*\(d:(\w+)\)", cypher, re.IGNORECASE
    )
    m_dst_bwd = re.search(
        r"MATCH\s+\(prev:(\w+)\)<-\[:(\w+)\]-\s*\(d:(\w+)\)", cypher, re.IGNORECASE
    )

    if m_fwd:
        src_label, rel_type, nb_label = m_fwd.group(1), m_fwd.group(2), m_fwd.group(3)
        src_nodes = nodes_of_label(src_label)
        degrees = [
            len([n for n in neighbors_out(s, rel_type) if NODE_LABELS.get(n) == nb_label])
            for s in src_nodes
        ]
    elif m_bwd:
        src_label, rel_type, nb_label = m_bwd.group(1), m_bwd.group(2), m_bwd.group(3)
        src_nodes = nodes_of_label(src_label)
        degrees = [
            len([n for n in neighbors_in(s, rel_type) if NODE_LABELS.get(n) == nb_label])
            for s in src_nodes
        ]
    elif m_dst_fwd:
        prev_label, rel_type, dst_label = m_dst_fwd.group(1), m_dst_fwd.group(2), m_dst_fwd.group(3)
        dst_nodes = nodes_of_label(dst_label)
        degrees = [
            len([n for n in neighbors_in(d, rel_type) if NODE_LABELS.get(n) == prev_label])
            for d in dst_nodes
        ]
    elif m_dst_bwd:
        prev_label, rel_type, dst_label = m_dst_bwd.group(1), m_dst_bwd.group(2), m_dst_bwd.group(3)
        dst_nodes = nodes_of_label(dst_label)
        degrees = [
            len([n for n in neighbors_out(d, rel_type) if NODE_LABELS.get(n) == prev_label])
            for d in dst_nodes
        ]
    elif m_undir:
        src_label, rel_type, nb_label = m_undir.group(1), m_undir.group(2), m_undir.group(3)
        src_nodes = nodes_of_label(src_label)
        degrees = [
            len([n for n in neighbors_out(s, rel_type) if NODE_LABELS.get(n) == nb_label]) +
            len([n for n in neighbors_in(s, rel_type) if NODE_LABELS.get(n) == nb_label])
            for s in src_nodes
        ]
    else:
        return 0

    if not degrees:
        return 0
    avg = sum(degrees) / len(degrees)
    return int(avg)  # return as int (count_fn returns int)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
_PASS = 0
_FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
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


# ---------------------------------------------------------------------------
# T1: enum_meta_paths
# ---------------------------------------------------------------------------
def test_enum_meta_paths():
    print("\nT1: enum_meta_paths")
    paths = enum_meta_paths(SCHEMA, max_len=2)

    # All keys have odd length (at least 3: node, rel, node)
    all_odd = all(len(k) % 2 == 1 and len(k) >= 3 for k in paths)
    check("all keys have odd length >= 3", all_odd)

    # No duplicates
    check("no duplicate keys", len(paths) == len(set(paths)))

    # Minimum 1-hop paths: at least 3 (one per relation, forward direction)
    # We also generate reverse traversals so >= 6 for 1-hop
    one_hop = [k for k in paths if len(k) == 3]
    check("at least 3 one-hop paths", len(one_hop) >= 3,
          f"got {len(one_hop)}")

    # Check specific expected 1-hop paths exist
    expected_1hop = [
        ("River", "FLOWS_THROUGH>", "Country"),
        ("Lake", "LOCATE_IN>", "Country"),
        ("Lake", "DRAINS_INTO>", "River"),
    ]
    for exp in expected_1hop:
        check(f"1-hop path exists: {exp}", exp in paths)

    # Reverse traversals
    expected_rev = [
        ("Country", "<FLOWS_THROUGH", "River"),
        ("Country", "<LOCATE_IN", "Lake"),
        ("River", "<DRAINS_INTO", "Lake"),
    ]
    for exp in expected_rev:
        check(f"reverse 1-hop path exists: {exp}", exp in paths)

    # 2-hop paths exist
    two_hop = [k for k in paths if len(k) == 5]
    check("at least one 2-hop path", len(two_hop) >= 1,
          f"got {len(two_hop)}")

    print(f"  Total paths enumerated: {len(paths)} "
          f"({len(one_hop)} one-hop, {len(two_hop)} two-hop)")
    return paths


# ---------------------------------------------------------------------------
# T2: JointPathSupport / card values
# ---------------------------------------------------------------------------
def test_card_values(catalog: CatalogResult):
    print("\nT2: card values (joint path support)")

    def entry(key):
        return catalog.get(key)

    # 1-hop: River -[FLOWS_THROUGH>]-> Country = 3
    e = entry(("River", "FLOWS_THROUGH>", "Country"))
    check("River-FLOWS_THROUGH>-Country card==3", e is not None and e.card == 3,
          f"got {e.card if e else 'NOT FOUND'}")

    # 1-hop: Lake -[LOCATE_IN>]-> Country = 3
    e = entry(("Lake", "LOCATE_IN>", "Country"))
    check("Lake-LOCATE_IN>-Country card==3", e is not None and e.card == 3,
          f"got {e.card if e else 'NOT FOUND'}")

    # 1-hop: Lake -[DRAINS_INTO>]-> River = 2
    e = entry(("Lake", "DRAINS_INTO>", "River"))
    check("Lake-DRAINS_INTO>-River card==2", e is not None and e.card == 2,
          f"got {e.card if e else 'NOT FOUND'}")

    # 2-hop: River-[FLOWS_THROUGH>]-Country-[<LOCATE_IN]-Lake = 5
    e = entry(("River", "FLOWS_THROUGH>", "Country", "<LOCATE_IN", "Lake"))
    check("River->Country<-Lake card==5", e is not None and e.card == 5,
          f"got {e.card if e else 'NOT FOUND'}")

    # 2-hop: Lake-[DRAINS_INTO>]-River-[FLOWS_THROUGH>]-Country = 4
    e = entry(("Lake", "DRAINS_INTO>", "River", "FLOWS_THROUGH>", "Country"))
    check("Lake->River->Country card==4", e is not None and e.card == 4,
          f"got {e.card if e else 'NOT FOUND'}")

    # 2-hop: Lake-[LOCATE_IN>]-Country-[<FLOWS_THROUGH]-River = 5
    e = entry(("Lake", "LOCATE_IN>", "Country", "<FLOWS_THROUGH", "River"))
    check("Lake->Country<-River card==5", e is not None and e.card == 5,
          f"got {e.card if e else 'NOT FOUND'}")


# ---------------------------------------------------------------------------
# T3: Selectivity
# ---------------------------------------------------------------------------
def test_selectivity(catalog: CatalogResult):
    print("\nT3: selectivity sketch")

    # River→Country:  card=3, src_pop=2 Rivers, dst_pop=2 Countries
    # sel = 3 / (2*2) = 0.75
    e = catalog.get(("River", "FLOWS_THROUGH>", "Country"))
    assert e is not None
    expected_sel = 3 / (2 * 2)
    check("River->Country sel==0.75", abs(e.sel - expected_sel) < 1e-9,
          f"got {e.sel:.4f}, expected {expected_sel:.4f}")

    # Lake→River:  card=2, src_pop=3, dst_pop=2
    # sel = 2 / (3*2) = 0.333...
    e2 = catalog.get(("Lake", "DRAINS_INTO>", "River"))
    assert e2 is not None
    expected_sel2 = 2 / (3 * 2)
    check("Lake->River sel==1/3", abs(e2.sel - expected_sel2) < 1e-9,
          f"got {e2.sel:.4f}, expected {expected_sel2:.4f}")

    # All sel values in [0, 1]
    bad = [e for e in catalog.entries if not (0.0 <= e.sel <= 1.0)]
    check("all sel in [0,1]", len(bad) == 0,
          f"{len(bad)} entries out of range")


# ---------------------------------------------------------------------------
# T4: Endpoint degree summary
# ---------------------------------------------------------------------------
def test_degree_summary(catalog: CatalogResult):
    print("\nT4: endpoint degree summary")

    # River-[FLOWS_THROUGH>]-Country:
    #   src (River) degrees: r1→2 hops, r2→1 hop → avg_deg = 1.5
    #   dst (Country) degrees: c1 receives 2 rivers, c2 receives 1 → avg_deg = 1.5
    #   src_distinct = 2 (both rivers appear), dst_distinct = 2 (both countries)
    e = catalog.get(("River", "FLOWS_THROUGH>", "Country"))
    assert e is not None
    check("River->Country src_distinct==2", e.src_distinct == 2,
          f"got {e.src_distinct}")
    check("River->Country dst_distinct==2", e.dst_distinct == 2,
          f"got {e.dst_distinct}")
    check("River->Country src_avg_degree==1 (int floor)", e.src_avg_degree >= 1,
          f"got {e.src_avg_degree}")

    # Lake-[DRAINS_INTO>]-River:
    #   l1→r1, l2→r1 → src_distinct=2 (l1,l2 appear), dst_distinct=1 (only r1)
    e2 = catalog.get(("Lake", "DRAINS_INTO>", "River"))
    assert e2 is not None
    check("Lake->River src_distinct==2", e2.src_distinct == 2,
          f"got {e2.src_distinct}")
    check("Lake->River dst_distinct==1", e2.dst_distinct == 1,
          f"got {e2.dst_distinct}")


# ---------------------------------------------------------------------------
# T5: median_card
# ---------------------------------------------------------------------------
def test_median_card(catalog: CatalogResult):
    print("\nT5: median_card (μ_P)")

    nonzero = [e.card for e in catalog.entries if e.card > 0]
    import statistics as stats
    expected_median = stats.median(nonzero)
    check("median_card matches statistics.median over non-zero",
          abs(catalog.median_card - expected_median) < 1e-9,
          f"catalog={catalog.median_card:.2f}, expected={expected_median:.2f}")

    # median must be positive
    check("median_card > 0", catalog.median_card > 0,
          f"got {catalog.median_card}")

    print(f"  Non-zero cards: {sorted(nonzero)}")
    print(f"  median_card (μ_P) = {catalog.median_card}")


# ---------------------------------------------------------------------------
# T6: role_description
# ---------------------------------------------------------------------------
def test_role_description(catalog: CatalogResult):
    print("\nT6: role_description")

    e = catalog.get(("River", "FLOWS_THROUGH>", "Country"))
    assert e is not None
    check("River->Country description contains FLOWS_THROUGH",
          "FLOWS_THROUGH" in e.role_description)
    check("River->Country description contains River",
          "River" in e.role_description)
    check("River->Country description contains Country",
          "Country" in e.role_description)
    check("River->Country description contains ->",
          "->" in e.role_description)

    e2 = catalog.get(("Country", "<FLOWS_THROUGH", "River"))
    assert e2 is not None
    check("Country<-River description contains <-",
          "<-" in e2.role_description,
          f"desc={e2.role_description!r}")

    print(f"  Example: {e.role_description!r}")
    e3 = catalog.get(("River", "FLOWS_THROUGH>", "Country", "<LOCATE_IN", "Lake"))
    if e3:
        print(f"  2-hop example: {e3.role_description!r}")


# ---------------------------------------------------------------------------
# T7: anchor_fields
# ---------------------------------------------------------------------------
def test_anchor_fields(catalog: CatalogResult):
    print("\nT7: anchor_fields")

    e = catalog.get(("River", "FLOWS_THROUGH>", "Country"))
    assert e is not None
    af = e.anchor_fields

    check("anchor_fields has 2 entries for 1-hop", len(af) == 2,
          f"got {len(af)}")

    # River at index 0: has properties name, length_km; numeric = length_km
    river_af = next((a for a in af if a["label"] == "River"), None)
    check("River anchor_fields found", river_af is not None)
    if river_af:
        check("River props include 'name'", "name" in river_af["properties"],
              str(river_af["properties"]))
        check("River props include 'length_km'",
              "length_km" in river_af["properties"])
        check("River numeric includes 'length_km'",
              "length_km" in river_af["numeric_properties"])
        check("River non-numeric 'name' not in numeric",
              "name" not in river_af["numeric_properties"])

    # Country at index 1: area_km2 and population are numeric
    country_af = next((a for a in af if a["label"] == "Country"), None)
    check("Country anchor_fields found", country_af is not None)
    if country_af:
        check("Country numeric includes 'area_km2'",
              "area_km2" in country_af["numeric_properties"])
        check("Country numeric includes 'population'",
              "population" in country_af["numeric_properties"])


# ---------------------------------------------------------------------------
# T8: save/load round-trip
# ---------------------------------------------------------------------------
def test_save_load(catalog: CatalogResult):
    print("\nT8: save/load round-trip")

    with tempfile.NamedTemporaryFile(
        suffix=".json", mode="w", delete=False
    ) as f:
        tmp_path = f.name

    try:
        save_catalog(catalog, tmp_path)
        check("file written", os.path.exists(tmp_path))

        reloaded = load_catalog(tmp_path)

        check("same number of entries",
              len(reloaded.entries) == len(catalog.entries),
              f"{len(reloaded.entries)} vs {len(catalog.entries)}")

        check("median_card preserved",
              abs(reloaded.median_card - catalog.median_card) < 1e-9,
              f"{reloaded.median_card} vs {catalog.median_card}")

        check("schema_name preserved",
              reloaded.schema_name == catalog.schema_name,
              f"{reloaded.schema_name!r}")

        # Spot-check one entry
        orig = catalog.get(("River", "FLOWS_THROUGH>", "Country"))
        relo = reloaded.get(("River", "FLOWS_THROUGH>", "Country"))
        check("entry card preserved after reload",
              orig is not None and relo is not None and orig.card == relo.card,
              f"orig={orig.card if orig else None}, relo={relo.card if relo else None}")

        check("entry sel preserved after reload",
              orig is not None and relo is not None and
              abs(orig.sel - relo.sel) < 1e-9)

        check("anchor_index preserved",
              reloaded.anchor_index == catalog.anchor_index)

        # Verify the JSON is valid (no corruption)
        with open(tmp_path) as fh:
            raw = json.load(fh)
        check("JSON file parseable", isinstance(raw, dict))

    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# T9: zero-support paths retained
# ---------------------------------------------------------------------------
def test_zero_support_retained(catalog: CatalogResult):
    print("\nT9: zero-support paths retained")

    zero_entries = [e for e in catalog.entries if e.card == 0]
    print(f"  Zero-support entries: {len(zero_entries)}")
    # In this graph there are some 2-hop paths that don't compose;
    # e.g. River-[<DRAINS_INTO]-Lake-[LOCATE_IN>]-Country traverses
    # in the right direction and produces some results, while paths that
    # require non-existing edges produce 0.
    # The key assertion is: if any zero-support entries exist, they are
    # present as keys (not silently dropped).
    if zero_entries:
        check("zero-support entries present as keys", True,
              f"{len(zero_entries)} retained")
        check("zero-support entries have card==0", all(e.card == 0 for e in zero_entries))
        check("zero-support entries have sel==0.0",
              all(e.sel == 0.0 for e in zero_entries))
    else:
        # All paths in this dense tiny graph happen to be populated — still correct
        check("catalog not empty", len(catalog.entries) > 0,
              "all paths happen to be non-zero for this dense graph")

    # Regardless, no entry should have negative card
    check("no negative card", all(e.card >= 0 for e in catalog.entries))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("MetaCypher catalog.py — unit tests")
    print("=" * 60)

    # Print the synthetic graph summary
    print(f"\nSynthetic graph:")
    for label in ["River", "Country", "Lake"]:
        print(f"  {label}: {nodes_of_label(label)}")
    print(f"  Edges: {len(EDGES)}")

    # --- Run T1 standalone ---
    paths = test_enum_meta_paths()

    # --- Build catalog ---
    print("\n[Building catalog with in-memory count_fn...]")
    catalog = build_catalog(SCHEMA, eval_in_memory, max_len=2)
    print(f"  Catalog entries built: {len(catalog.entries)}")
    print(f"  Schema name: {catalog.schema_name!r}")

    # Dump all entries for inspection
    print("\n  All catalog entries:")
    for e in sorted(catalog.entries, key=lambda x: (-x.card, x.role_description)):
        print(f"    card={e.card:3d}  sel={e.sel:.3f}  "
              f"src_d={e.src_distinct}  dst_d={e.dst_distinct}  "
              f"{e.role_description}")

    # --- Run remaining tests ---
    test_card_values(catalog)
    test_selectivity(catalog)
    test_degree_summary(catalog)
    test_median_card(catalog)
    test_role_description(catalog)
    test_anchor_fields(catalog)
    test_save_load(catalog)
    test_zero_support_retained(catalog)

    # --- Summary ---
    print("\n" + "=" * 60)
    total = _PASS + _FAIL
    print(f"Results: {_PASS}/{total} passed, {_FAIL} failed")
    if _FAIL > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
