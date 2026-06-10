"""
catalog.py — Offline cardinality catalog for MetaCypher.

Implements Algorithm 1 (Offline cardinality-catalog construction) from the paper:

  CatalogEntry(p) = <key(p), card(p), deg_hat(p), sel_hat(p), phi_desc(p), anchor(p)>

where
  - key(p)         canonical typed-path signature (tuple of label/rel strings)
  - card(p)        joint instance count: |{(v0,...,vl): vi:τi, (vi-1,vi):ri}|
  - deg_hat(p)     endpoint-degree summary dict (src_distinct, dst_distinct,
                   src_avg_degree, dst_avg_degree)
  - sel_hat(p)     selectivity sketch: card(p) / (src_pop * dst_pop), range [0,1]
  - phi_desc(p)    natural-language role description of the typed path
  - anchor(p)      compatible anchor fields: list of dicts
                   {node_index, label, properties, numeric_properties}

Public API
----------
  CatalogEntry                  dataclass
  enum_meta_paths(schema, max_len) -> list[tuple]
  build_catalog(schema, count_fn, max_len, label_count_fn) -> CatalogResult
  CatalogResult                 dataclass  (.entries, .median_card, .anchor_index)
  save_catalog(result, path)
  load_catalog(path) -> CatalogResult
  make_neo4j_count_fn(db_name, pool) -> Callable[[str], int]

count_fn signature:  (cypher: str) -> int
  Must return the integer COUNT from a Cypher query of the form
  "MATCH ... RETURN count(*) AS c"

label_count_fn signature (optional): (label: str) -> int
  Returns total node count for a label; used for selectivity denominator.
  Defaults to issuing "MATCH (n:Label) RETURN count(n) AS c".
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx  # required; networkx IS installed in this repo


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# A meta-path key is an immutable tuple whose elements alternate between
# node labels and relation labels with direction suffix, e.g.
#   ("River", "FLOWS_THROUGH>", "Country", "<LOCATE_IN", "Lake")
# Direction encoding:  "REL>"  = (subj)-[:REL]->(obj)  (out from left node)
#                     "<REL"   = (subj)<-[:REL]-(obj)   (in to left node)
MetaPathKey = Tuple[str, ...]

SchemaDict = Dict[str, Any]   # same shape as compact_schema_for_prompt output


# ---------------------------------------------------------------------------
# CatalogEntry
# ---------------------------------------------------------------------------
@dataclass
class CatalogEntry:
    """One row in the cardinality catalog.

    Mirrors the CatalogEntry definition in method.tex (subsec:prelim) and
    Fig. fig:offline.

    Attributes
    ----------
    key : MetaPathKey
        Canonical typed-path signature. Alternating (label, rel_dir, label, …).
        Odd-indexed elements are node labels; even-indexed are relation tokens
        of the form "REL>"  or "<REL".
    node_labels : list[str]
        Ordered list of node-label types along the path  (length = path_len+1).
    rel_tokens : list[str]
        Ordered list of relation tokens with direction  (length = path_len).
    card : int
        Joint path support: number of distinct (v0,…,vl) tuples conforming to p.
        Zero means the path is schema-valid but data-absent.
    src_distinct : int
        Distinct start-node bindings surviving in the joint path (|π_{τ0}(p)|).
    dst_distinct : int
        Distinct end-node bindings surviving in the joint path (|π_{τl}(p)|).
    src_avg_degree : float
        Average out-degree of τ0 nodes along the first relation step.
    dst_avg_degree : float
        Average in-degree  of τl nodes along the last  relation step.
    sel : float
        Selectivity sketch = card / (src_pop * dst_pop).  Range (0, 1].
        src_pop / dst_pop are total node counts for endpoint labels.
        Zero when card == 0.
    role_description : str
        Natural-language description of the path role, e.g.
        "River -[FLOWS_THROUGH]-> Country <-[LOCATE_IN]- Lake".
        Used for semantic matching in ValidateRank.
    anchor_fields : list[dict]
        Per-node-position anchor compatibility entries, each:
          {"node_index": int, "label": str, "properties": list[str],
           "numeric_properties": list[str]}
        Numeric properties can host comparison predicates offline.
    """
    key: MetaPathKey
    node_labels: List[str]
    rel_tokens: List[str]
    card: int
    src_distinct: int
    dst_distinct: int
    src_avg_degree: float
    dst_avg_degree: float
    sel: float
    role_description: str
    anchor_fields: List[Dict[str, Any]]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def path_len(self) -> int:
        """Number of relation hops."""
        return len(self.rel_tokens)

    @property
    def is_data_present(self) -> bool:
        return self.card > 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["key"] = list(d["key"])   # tuple → list for JSON
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CatalogEntry":
        d = dict(d)
        d["key"] = tuple(d["key"])
        d["node_labels"] = list(d["node_labels"])
        d["rel_tokens"] = list(d["rel_tokens"])
        d["anchor_fields"] = list(d["anchor_fields"])
        return cls(**d)


# ---------------------------------------------------------------------------
# CatalogResult
# ---------------------------------------------------------------------------
@dataclass
class CatalogResult:
    """Return type of build_catalog.

    Attributes
    ----------
    entries : list[CatalogEntry]
        All catalog entries, including zero-support ones (retained as keys).
    median_card : float
        μ_P — median joint cardinality over entries with card > 0.
        Used by the sparsity term phi_sparse(s) = exp(-n_hat(s) / μ_P).
        Set to 1.0 if all entries are data-absent (prevents division by zero).
    anchor_index : dict[str, list[str]]
        Label → list of property names that can anchor question mentions.
        Built once from the schema; used by CompatibleAnchorFields.
    schema_name : str
        Name of the schema this catalog was built from.
    """
    entries: List[CatalogEntry]
    median_card: float
    anchor_index: Dict[str, List[str]]
    schema_name: str = ""

    # fast lookup: key → entry
    _index: Dict[MetaPathKey, CatalogEntry] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self._index = {e.key: e for e in self.entries}

    def get(self, key: MetaPathKey) -> Optional[CatalogEntry]:
        return self._index.get(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "median_card": self.median_card,
            "anchor_index": self.anchor_index,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CatalogResult":
        entries = [CatalogEntry.from_dict(e) for e in d.get("entries", [])]
        return cls(
            entries=entries,
            median_card=float(d.get("median_card", 1.0)),
            anchor_index=d.get("anchor_index", {}),
            schema_name=d.get("schema_name", ""),
        )


# ---------------------------------------------------------------------------
# EnumMetaPaths  (Algorithm 1, line 1)
# ---------------------------------------------------------------------------
def _build_schema_graph(schema: SchemaDict) -> nx.MultiDiGraph:
    """Build a directed multigraph from the schema relation list.

    Each edge carries:
      rel_label   : str   relation type
      direction   : "out" | "in" | "undirected"
      subj_label  : str
      obj_label   : str
    """
    G = nx.MultiDiGraph()
    for e in schema.get("entities", []):
        G.add_node(e["label"], properties=e.get("properties", []))

    for r in schema.get("relations", []):
        subj = r.get("subj_label") or r.get("start_label")
        obj = r.get("obj_label") or r.get("end_label")
        rtype = r.get("type") or r.get("label")
        direction = r.get("direction", "out")
        if not (subj and obj and rtype):
            continue
        G.add_edge(
            subj, obj,
            rel_label=rtype,
            direction=direction,
            subj_label=subj,
            obj_label=obj,
        )
    return G


def _rel_token(rel_label: str, direction: str, from_node: str, to_node: str,
               subj_label: str, obj_label: str) -> str:
    """Encode a relation step as a directed token.

    Convention (from the perspective of the path traversal direction, i.e.
    from_node → to_node in the path):
      "REL>"   means we follow (from_node)-[:REL]->(to_node) where from=subj, to=obj
      "<REL"   means we follow (from_node)<-[:REL]-(to_node) where from=obj, to=subj
      "REL~"   undirected
    """
    if direction == "undirected":
        return f"{rel_label}~"
    # The edge is stored as subj→obj in the schema graph.
    # When we traverse from_node→to_node:
    #   if from_node==subj_label and to_node==obj_label → forward (out) → "REL>"
    #   if from_node==obj_label  and to_node==subj_label → reverse (in)  → "<REL"
    if from_node == subj_label and to_node == obj_label:
        return f"{rel_label}>"
    else:
        return f"<{rel_label}"


def enum_meta_paths(schema: SchemaDict, max_len: int = 2) -> List[MetaPathKey]:
    """Enumerate schema-valid meta-paths up to length max_len.

    Implements EnumMetaPaths(S, L) from Algorithm 1.

    Starting from each node label, follows outgoing AND incoming relation
    triples up to depth max_len, canonicalizes the signature, and deduplicates.

    Parameters
    ----------
    schema : dict
        Schema dict as produced by compact_schema_for_prompt:
          {entities: [{label, properties}], relations: [{type|label, subj_label,
           obj_label, direction?, pattern?}]}
    max_len : int
        Maximum number of hops (relations) per path.

    Returns
    -------
    list of MetaPathKey
        Each key is an immutable tuple alternating (node_label, rel_token, …, node_label).
    """
    G = _build_schema_graph(schema)
    seen: set[MetaPathKey] = set()
    results: List[MetaPathKey] = []

    def dfs(current_node: str, path_key: List[str], depth: int) -> None:
        if depth > 0:
            k = tuple(path_key)
            if k not in seen:
                seen.add(k)
                results.append(k)
        if depth >= max_len:
            return
        # Expand: iterate all edges incident to current_node in G
        # Forward edges: current_node is the tail (subj)
        for _, nbr, edata in G.out_edges(current_node, data=True):
            token = _rel_token(
                edata["rel_label"], edata["direction"],
                current_node, nbr,
                edata["subj_label"], edata["obj_label"],
            )
            dfs(nbr, path_key + [token, nbr], depth + 1)
        # Backward edges: current_node is the head (obj) — traverse reverse
        for src, _, edata in G.in_edges(current_node, data=True):
            if edata.get("direction", "out") == "undirected":
                continue  # already covered above as forward
            token = _rel_token(
                edata["rel_label"], edata["direction"],
                current_node, src,
                edata["subj_label"], edata["obj_label"],
            )
            dfs(src, path_key + [token, src], depth + 1)

    for start_label in list(G.nodes()):
        dfs(start_label, [start_label], 0)

    return results


# ---------------------------------------------------------------------------
# Cypher helpers
# ---------------------------------------------------------------------------
def _compile_count_query(key: MetaPathKey) -> str:
    """Compile a COUNT(*) Cypher query for a meta-path key.

    Uses variable names n0, n1, … and r0, r1, …

    Example for key=("River","FLOWS_THROUGH>","Country","<LOCATE_IN","Lake"):
      MATCH (n0:River)-[r0:FLOWS_THROUGH]->(n1:Country)<-[r1:LOCATE_IN]-(n2:Lake)
      RETURN count(*) AS c
    """
    node_labels = [key[i] for i in range(0, len(key), 2)]
    rel_tokens = [key[i] for i in range(1, len(key), 2)]

    parts: List[str] = []
    for i, node_label in enumerate(node_labels):
        parts.append(f"(n{i}:{node_label})")
        if i < len(rel_tokens):
            tok = rel_tokens[i]
            if tok.endswith(">"):
                rel_label = tok[:-1]
                parts.append(f"-[r{i}:{rel_label}]->")
            elif tok.startswith("<"):
                rel_label = tok[1:]
                parts.append(f"<-[r{i}:{rel_label}]-")
            else:  # undirected, ends with ~
                rel_label = tok[:-1]
                parts.append(f"-[r{i}:{rel_label}]-")
    match_clause = "".join(parts)
    return f"MATCH {match_clause} RETURN count(*) AS c"


def _compile_endpoint_queries(key: MetaPathKey) -> Tuple[str, str]:
    """Return (src_count_query, dst_count_query) for endpoint populations."""
    src_label = key[0]
    dst_label = key[-1]
    src_q = f"MATCH (n:{src_label}) RETURN count(n) AS c"
    dst_q = f"MATCH (n:{dst_label}) RETURN count(n) AS c"
    return src_q, dst_q


def _compile_src_distinct_query(key: MetaPathKey) -> str:
    """COUNT DISTINCT start-node bindings across the whole path."""
    # Build the MATCH clause exactly as _compile_count_query, then project n0.
    body = _compile_count_query(key)
    match_part = body.split(" RETURN ")[0]
    return f"{match_part} RETURN count(DISTINCT n0) AS c"


def _compile_dst_distinct_query(key: MetaPathKey) -> str:
    """COUNT DISTINCT end-node bindings across the whole path."""
    body = _compile_count_query(key)
    match_part = body.split(" RETURN ")[0]
    last_idx = len(key) // 2
    return f"{match_part} RETURN count(DISTINCT n{last_idx}) AS c"


def _compile_src_degree_query(key: MetaPathKey) -> str:
    """Average out-degree of src label along the first hop."""
    src_label = key[0]
    tok = key[1]
    hop_label = key[2]
    if tok.endswith(">"):
        rel = tok[:-1]
        return (
            f"MATCH (s:{src_label})-[:{rel}]->(nb:{hop_label}) "
            f"WITH s, count(nb) AS deg "
            f"RETURN avg(deg) AS c"
        )
    elif tok.startswith("<"):
        rel = tok[1:]
        return (
            f"MATCH (s:{src_label})<-[:{rel}]-(nb:{hop_label}) "
            f"WITH s, count(nb) AS deg "
            f"RETURN avg(deg) AS c"
        )
    else:
        rel = tok[:-1]
        return (
            f"MATCH (s:{src_label})-[:{rel}]-(nb:{hop_label}) "
            f"WITH s, count(nb) AS deg "
            f"RETURN avg(deg) AS c"
        )


def _compile_dst_degree_query(key: MetaPathKey) -> str:
    """Average in-degree of dst label along the last hop."""
    last_tok = key[-2]
    dst_label = key[-1]
    prev_label = key[-3]
    if last_tok.endswith(">"):
        rel = last_tok[:-1]
        return (
            f"MATCH (prev:{prev_label})-[:{rel}]->(d:{dst_label}) "
            f"WITH d, count(prev) AS deg "
            f"RETURN avg(deg) AS c"
        )
    elif last_tok.startswith("<"):
        rel = last_tok[1:]
        return (
            f"MATCH (prev:{prev_label})<-[:{rel}]-(d:{dst_label}) "
            f"WITH d, count(prev) AS deg "
            f"RETURN avg(deg) AS c"
        )
    else:
        rel = last_tok[:-1]
        return (
            f"MATCH (prev:{prev_label})-[:{rel}]-(d:{dst_label}) "
            f"WITH d, count(prev) AS deg "
            f"RETURN avg(deg) AS c"
        )


# ---------------------------------------------------------------------------
# RoleDescription  (Algorithm 1, line 8)
# ---------------------------------------------------------------------------
def _role_description(key: MetaPathKey) -> str:
    """Build a natural-language role description for the path.

    Uses relation token notation directly, e.g.
      "River -[FLOWS_THROUGH]-> Country <-[LOCATE_IN]- Lake"
    This is the phi_desc(p) field (paper "Descriptions, grounding, and update
    locality", subsubsec:descriptions).
    """
    parts: List[str] = []
    node_labels = [key[i] for i in range(0, len(key), 2)]
    rel_tokens = [key[i] for i in range(1, len(key), 2)]
    for i, lbl in enumerate(node_labels):
        parts.append(lbl)
        if i < len(rel_tokens):
            tok = rel_tokens[i]
            if tok.endswith(">"):
                parts.append(f"-[{tok[:-1]}]->")
            elif tok.startswith("<"):
                parts.append(f"<-[{tok[1:]}]-")
            else:
                parts.append(f"-[{tok[:-1]}]-")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# CompatibleAnchorFields  (Algorithm 1, line 9)
# ---------------------------------------------------------------------------
def _build_anchor_index(schema: SchemaDict) -> Dict[str, Dict[str, Any]]:
    """Build a per-label property index from the schema.

    Returns
    -------
    dict: label → {"properties": [str], "numeric_properties": [str]}
    """
    # Numeric type keywords from schema property type annotations
    _NUMERIC_TYPES = {"int", "integer", "float", "double", "long", "number",
                      "duration", "date", "datetime", "localdate", "localdatetime",
                      "time", "localtime"}

    index: Dict[str, Dict[str, Any]] = {}
    for e in schema.get("entities", []):
        label = e["label"]
        props_raw = e.get("properties", [])
        # properties may be a list of strings (compact form) or a list of dicts
        # or a dict {name: type}
        if isinstance(props_raw, dict):
            items = [(k, v) for k, v in props_raw.items()]
        elif isinstance(props_raw, list):
            items = []
            for p in props_raw:
                if isinstance(p, dict):
                    items.append((p.get("name", p.get("key", "")), p.get("type", "str")))
                else:
                    items.append((str(p), "str"))
        else:
            items = []

        all_props = [k for k, _ in items if k]
        numeric_props = [k for k, t in items if k and
                         any(nt in str(t).lower() for nt in _NUMERIC_TYPES)]
        index[label] = {"properties": all_props, "numeric_properties": numeric_props}
    return index


def _compatible_anchor_fields(
    key: MetaPathKey, anchor_index: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return per-node-position anchor compatibility dicts.

    Each entry:
      {"node_index": int, "label": str, "properties": list[str],
       "numeric_properties": list[str]}
    """
    node_labels = [key[i] for i in range(0, len(key), 2)]
    result: List[Dict[str, Any]] = []
    for idx, lbl in enumerate(node_labels):
        info = anchor_index.get(lbl, {"properties": [], "numeric_properties": []})
        result.append({
            "node_index": idx,
            "label": lbl,
            "properties": list(info.get("properties", [])),
            "numeric_properties": list(info.get("numeric_properties", [])),
        })
    return result


# ---------------------------------------------------------------------------
# JointPathSupport + SelectivitySketch  (Algorithm 1, lines 6–7)
# ---------------------------------------------------------------------------
def _safe_count(count_fn: Callable[[str], int], cypher: str) -> int:
    """Execute count_fn, returning 0 on any error."""
    try:
        result = count_fn(cypher)
        return max(0, int(result))
    except Exception:
        return 0


def _compute_selectivity(card: int, src_pop: int, dst_pop: int) -> float:
    """sel_hat(p) = card / (src_pop * dst_pop).

    Clamped to [0, 1].  Returns 0.0 when card == 0 or populations unknown.
    """
    if card <= 0 or src_pop <= 0 or dst_pop <= 0:
        return 0.0
    raw = card / (src_pop * dst_pop)
    return min(1.0, raw)


# ---------------------------------------------------------------------------
# build_catalog  (Algorithm 1)
# ---------------------------------------------------------------------------
def build_catalog(
    schema: SchemaDict,
    count_fn: Callable[[str], int],
    max_len: int = 2,
    label_count_fn: Optional[Callable[[str], int]] = None,
) -> CatalogResult:
    """Build the offline cardinality catalog.

    Implements Algorithm 1 (alg:profiling) from the offline-catalog section
    (sec:offline) of the paper.

    Parameters
    ----------
    schema : dict
        Schema dict as produced by compact_schema_for_prompt.
    count_fn : callable (cypher: str) -> int
        Executes a COUNT query and returns the integer result.
        Injected so the catalog is testable without a live Neo4j.
    max_len : int
        Maximum meta-path length L.
    label_count_fn : callable (label: str) -> int, optional
        Returns total node count for a label.  Used for selectivity denominator.
        If omitted, issues "MATCH (n:Label) RETURN count(n) AS c" via count_fn.

    Returns
    -------
    CatalogResult
    """
    if label_count_fn is None:
        def label_count_fn(label: str) -> int:  # type: ignore[misc]
            return _safe_count(count_fn, f"MATCH (n:{label}) RETURN count(n) AS c")

    # -- Step 1: EnumMetaPaths ------------------------------------------
    path_keys = enum_meta_paths(schema, max_len=max_len)

    # -- Step 1b: BuildAnchorIndex --------------------------------------
    anchor_index_raw = _build_anchor_index(schema)
    # Expose as label → list[str] for CatalogResult.anchor_index
    anchor_index: Dict[str, List[str]] = {
        lbl: info["properties"] for lbl, info in anchor_index_raw.items()
    }

    # Cache label population counts to avoid repeated queries
    _label_pop: Dict[str, int] = {}

    def label_pop(label: str) -> int:
        if label not in _label_pop:
            _label_pop[label] = label_count_fn(label)
        return _label_pop[label]

    # -- Steps 5–10: Per-path statistics --------------------------------
    entries: List[CatalogEntry] = []

    for key in path_keys:
        node_labels = [key[i] for i in range(0, len(key), 2)]
        rel_tokens = [key[i] for i in range(1, len(key), 2)]

        # JointPathSupport (line 6)
        card_query = _compile_count_query(key)
        card = _safe_count(count_fn, card_query)

        # EndpointDegreeSummary (line 5)
        if len(key) >= 3:
            src_distinct = _safe_count(count_fn, _compile_src_distinct_query(key))
            dst_distinct = _safe_count(count_fn, _compile_dst_distinct_query(key))
            src_deg = float(_safe_count(count_fn, _compile_src_degree_query(key)) or 0)
            dst_deg = float(_safe_count(count_fn, _compile_dst_degree_query(key)) or 0)
        else:
            # Shouldn't happen (min path len = 1 hop), but guard anyway
            src_distinct = 0
            dst_distinct = 0
            src_deg = 0.0
            dst_deg = 0.0

        # SelectivitySketch (line 7)
        src_pop = label_pop(node_labels[0])
        dst_pop = label_pop(node_labels[-1])
        sel = _compute_selectivity(card, src_pop, dst_pop)

        # RoleDescription (line 8)
        role_desc = _role_description(key)

        # CompatibleAnchorFields (line 9)
        anchor_fields = _compatible_anchor_fields(key, anchor_index_raw)

        entry = CatalogEntry(
            key=key,
            node_labels=node_labels,
            rel_tokens=rel_tokens,
            card=card,
            src_distinct=src_distinct,
            dst_distinct=dst_distinct,
            src_avg_degree=src_deg,
            dst_avg_degree=dst_deg,
            sel=sel,
            role_description=role_desc,
            anchor_fields=anchor_fields,
        )
        entries.append(entry)

    # line 12: compress zero-support entries (keep them; just mark them)
    # The paper says "compressed with zero support, so online search can demote
    # them without erasing a possible gold structure." We retain all entries.

    # Compute μ_P  (median card over non-zero entries; used by phi_sparse)
    nonzero_cards = [e.card for e in entries if e.card > 0]
    if nonzero_cards:
        median_card = statistics.median(nonzero_cards)
    else:
        median_card = 1.0  # fallback: prevents division-by-zero in phi_sparse

    return CatalogResult(
        entries=entries,
        median_card=float(median_card),
        anchor_index=anchor_index,
        schema_name=schema.get("name", ""),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_catalog(result: CatalogResult, path: str) -> None:
    """Serialize CatalogResult to a JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, ensure_ascii=False, indent=2)


def load_catalog(path: str) -> CatalogResult:
    """Deserialize CatalogResult from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return CatalogResult.from_dict(data)


# ---------------------------------------------------------------------------
# Production helper: build count_fn from graph_connector
# ---------------------------------------------------------------------------
def make_neo4j_count_fn(
    db_name: str,
    pool: Any,  # Neo4jConnectionPool — lazy import avoids hard dep at module load
) -> Callable[[str], int]:
    """Return a count_fn callable backed by a live Neo4j connection pool.

    Parameters
    ----------
    db_name : str
        Graph name, e.g. "geography", "company".
    pool : Neo4jConnectionPool
        An instance from graph_connector.create_pool().

    Returns
    -------
    callable (cypher: str) -> int
        Executes the cypher via pool.execute(db_name, cypher) and extracts
        the integer value of the first 'c' column from the first row.
    """
    def _count_fn(cypher: str) -> int:
        rows = pool.execute(db_name, cypher)
        if not rows:
            return 0
        first = rows[0]
        # Support both "c" and "count(*)" as column names
        for key in ("c", "count(*)", "count(n)", "count(DISTINCT n0)",
                    "count(DISTINCT n1)", "count(DISTINCT n2)"):
            if key in first:
                return int(first[key] or 0)
        # Fallback: take the first value
        val = next(iter(first.values()), 0)
        return int(val or 0)

    return _count_fn
