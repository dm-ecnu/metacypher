"""Mechanism-level diagnostics (paper Table `tab:mechanism` + Fig `fig:error_attr`).

Computes, from gold/predicted provenance subgraphs and the graph schema, the
quantities the paper reports as "script-derived from gold/predicted provenance
subgraphs with manual audit of representative groups":

- **gold meta-path recall@B** — fraction of gold typed paths retained in the beam
- **probe precision** — fraction of probed signatures returning positive support
- **empty-result query accuracy (EX)** — EX restricted to empty-gold-answer
  questions (a wrong structure would return spurious rows)
- **structural hallucination attribution** — predicted-Cypher structural errors
  grouped into phantom node / phantom relation / phantom attribute / invalid
  connectivity (and the no-hallucination share)

The structural checks parse a Cypher query with a small regex extractor (no
external parser) into the labels, relationship types, property accesses, and
node-label adjacencies it mentions, then validate each against the schema.
This is deterministic and the basis the paper audits manually on top of.

Schema shape (as the rest of the pipeline reads it)::

    {
      "nodes": [{"label": "River", "properties": ["name", "length"]}, ...],
      "relationships": [{"type": "FLOWS_THROUGH", "from": "River", "to": "Country"}, ...]
    }

Property lists may also be a dict (``{"name": "STRING"}``); both are accepted.
Run ``python3 test_diagnostics.py`` for the unit tests.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Schema view
# ---------------------------------------------------------------------------


@dataclass
class SchemaView:
    """Normalized, case-insensitive lookups over a graph schema."""

    labels: Set[str]
    rel_types: Set[str]
    props_by_label: Dict[str, Set[str]]
    # Undirected adjacency of label pairs that some relationship connects.
    label_pairs: Set[Tuple[str, str]]

    @classmethod
    def from_schema(cls, schema: Dict[str, Any]) -> "SchemaView":
        labels: Set[str] = set()
        props_by_label: Dict[str, Set[str]] = {}
        for node in schema.get("nodes", []) or []:
            label = str(node.get("label", "")).strip()
            if not label:
                continue
            labels.add(label.lower())
            raw_props = node.get("properties") or []
            names = raw_props.keys() if isinstance(raw_props, dict) else raw_props
            props_by_label[label.lower()] = {str(name).lower() for name in names}
        rel_types: Set[str] = set()
        label_pairs: Set[Tuple[str, str]] = set()
        for rel in schema.get("relationships", []) or []:
            rtype = str(rel.get("type", "")).strip()
            if rtype:
                rel_types.add(rtype.lower())
            src = str(rel.get("from", "")).strip().lower()
            dst = str(rel.get("to", "")).strip().lower()
            if src and dst:
                label_pairs.add(tuple(sorted((src, dst))))
                labels.add(src)
                labels.add(dst)
        return cls(labels=labels, rel_types=rel_types, props_by_label=props_by_label, label_pairs=label_pairs)

    def known_label(self, label: str) -> bool:
        return label.lower() in self.labels

    def known_rel(self, rtype: str) -> bool:
        return rtype.lower() in self.rel_types

    def known_prop(self, label: Optional[str], prop: str) -> bool:
        prop = prop.lower()
        if label is not None:
            props = self.props_by_label.get(label.lower())
            if props is not None:
                return prop in props
        # Variable not bound to a known label → accept iff any label has it.
        return any(prop in props for props in self.props_by_label.values())


# ---------------------------------------------------------------------------
# Cypher structural extractor (regex; no external parser)
# ---------------------------------------------------------------------------

_NODE_RE = re.compile(r"\(\s*(\w+)?\s*(?::\s*([A-Za-z_][\w]*))?[^()]*?\)")
_REL_RE = re.compile(r"\[\s*\w*\s*:\s*([A-Za-z_][\w]*)")
# var.prop  (property access)
_PROP_RE = re.compile(r"\b(\w+)\.(\w+)\b")
# A linear MATCH path segment: (a:La)-[:R]->(b:Lb)
_PATH_SEG_RE = re.compile(
    r"\(\s*\w*\s*:?\s*([A-Za-z_]\w*)?[^()]*?\)\s*"
    r"(<?-)\s*\[\s*\w*\s*:\s*([A-Za-z_]\w*)[^\]]*?\]\s*(->?)"
    r"\s*\(\s*\w*\s*:?\s*([A-Za-z_]\w*)?"
)


@dataclass
class CypherStructure:
    labels: List[str] = field(default_factory=list)
    rel_types: List[str] = field(default_factory=list)
    # (variable, property)
    prop_access: List[Tuple[str, str]] = field(default_factory=list)
    # variable -> label, from inline binding (:Label)
    var_labels: Dict[str, str] = field(default_factory=dict)
    # (label_a, rel_type, label_b) adjacency triples seen in path segments
    adjacencies: List[Tuple[Optional[str], str, Optional[str]]] = field(default_factory=list)


def parse_cypher(cypher: str) -> CypherStructure:
    """Extract the structural skeleton of a Cypher query.

    Intentionally lenient: it pulls every label, relationship type, property
    access, and path adjacency it can see. It does not validate Cypher syntax —
    malformed queries simply yield fewer extracted items.
    """
    struct = CypherStructure()
    if not cypher:
        return struct
    # Strip string literals so quoted text is not mistaken for identifiers.
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "''", cypher)

    for match in _NODE_RE.finditer(text):
        var, label = match.group(1), match.group(2)
        if label:
            struct.labels.append(label)
            if var:
                struct.var_labels[var] = label
    for match in _REL_RE.finditer(text):
        struct.rel_types.append(match.group(1))
    for var, prop in _PROP_RE.findall(text):
        # Skip numeric-looking matches and Cypher keywords used as vars.
        struct.prop_access.append((var, prop))
    for match in _PATH_SEG_RE.finditer(text):
        label_a, _, rtype, _, label_b = match.groups()
        struct.adjacencies.append((label_a, rtype, label_b))

    # de-duplicate while preserving order
    struct.labels = list(OrderedDict.fromkeys(struct.labels))
    struct.rel_types = list(OrderedDict.fromkeys(struct.rel_types))
    struct.prop_access = list(OrderedDict.fromkeys(struct.prop_access))
    return struct


# ---------------------------------------------------------------------------
# Structural hallucination attribution (fig:error_attr)
# ---------------------------------------------------------------------------

ERROR_CLASSES = ("phantom_node", "phantom_relation", "phantom_attribute", "invalid_connectivity")


def classify_hallucinations(cypher: str, schema: SchemaView) -> Dict[str, bool]:
    """Flag which structural-hallucination classes a predicted query exhibits.

    Returns a dict of class -> bool (present in this query). A query with all
    False is "no hallucination" (structurally schema-valid).
    """
    struct = parse_cypher(cypher)
    flags = {cls: False for cls in ERROR_CLASSES}

    for label in struct.labels:
        if not schema.known_label(label):
            flags["phantom_node"] = True
    for rtype in struct.rel_types:
        if not schema.known_rel(rtype):
            flags["phantom_relation"] = True
    for var, prop in struct.prop_access:
        label = struct.var_labels.get(var)
        if not schema.known_prop(label, prop):
            flags["phantom_attribute"] = True
    for label_a, rtype, label_b in struct.adjacencies:
        # Only judge connectivity when both endpoint labels and the rel type
        # are individually known; unknown pieces are already counted above.
        if (
            label_a
            and label_b
            and schema.known_label(label_a)
            and schema.known_label(label_b)
            and schema.known_rel(rtype)
        ):
            pair = tuple(sorted((label_a.lower(), label_b.lower())))
            if pair not in schema.label_pairs:
                flags["invalid_connectivity"] = True
    return flags


def hallucination_attribution(
    predicted_cyphers: Iterable[str], schema: Dict[str, Any]
) -> Dict[str, float]:
    """Aggregate per-class hallucination rates + no-hallucination share.

    Matches fig:error_attr: each rate is the fraction of queries exhibiting
    that error class; ``no_hallucination`` is the fraction with none.
    """
    view = SchemaView.from_schema(schema)
    counts = {cls: 0 for cls in ERROR_CLASSES}
    clean = 0
    total = 0
    for cypher in predicted_cyphers:
        total += 1
        flags = classify_hallucinations(cypher, view)
        if any(flags.values()):
            for cls, present in flags.items():
                counts[cls] += int(present)
        else:
            clean += 1
    if total == 0:
        return {cls: 0.0 for cls in ERROR_CLASSES} | {"no_hallucination": 0.0, "n": 0}
    result = {cls: counts[cls] / total for cls in ERROR_CLASSES}
    result["no_hallucination"] = clean / total
    result["n"] = total
    return result


# ---------------------------------------------------------------------------
# Beam / probe / empty-result diagnostics (tab:mechanism)
# ---------------------------------------------------------------------------


def _sig_key(sig: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(token).lower() for token in sig)


def gold_metapath_recall_at_b(
    beam_sigs: Sequence[Sequence[str]],
    gold_sigs: Sequence[Sequence[str]],
    beam_width: Optional[int] = None,
) -> Optional[float]:
    """Fraction of gold typed paths retained in the (top-B) beam.

    Returns None when there are no gold paths (so callers can skip the row).
    """
    if not gold_sigs:
        return None
    kept = beam_sigs if beam_width is None else list(beam_sigs)[:beam_width]
    kept_keys = {_sig_key(sig) for sig in kept}
    hit = sum(1 for gold in gold_sigs if _sig_key(gold) in kept_keys)
    return hit / len(gold_sigs)


def probe_precision(probe_results: Sequence[Any]) -> Optional[float]:
    """Fraction of probed signatures with positive support.

    Accepts a sequence of ints/floats (support counts) or objects with an
    ``n_hat`` attribute (e.g. RankedCandidate). Returns None if no probes.
    """
    supports: List[float] = []
    for item in probe_results:
        if hasattr(item, "n_hat"):
            supports.append(float(item.n_hat))
        elif isinstance(item, (int, float)):
            supports.append(float(item))
        elif isinstance(item, dict) and "n_hat" in item:
            supports.append(float(item["n_hat"]))
    if not supports:
        return None
    return sum(1 for value in supports if value > 0) / len(supports)


@dataclass
class EXRecord:
    """One evaluated question for empty-result accuracy."""

    gold_empty: bool   # the gold answer is the empty set
    correct: bool      # EX: predicted result set equals gold result set


def empty_result_accuracy(records: Iterable[EXRecord]) -> Optional[float]:
    """EX restricted to empty-gold-answer questions. None if none qualify."""
    relevant = [r for r in records if r.gold_empty]
    if not relevant:
        return None
    return sum(1 for r in relevant if r.correct) / len(relevant)


def mechanism_diagnostics(
    schema: Dict[str, Any],
    predicted_cyphers: Sequence[str],
    *,
    beams: Optional[Sequence[Sequence[Sequence[str]]]] = None,
    golds: Optional[Sequence[Sequence[Sequence[str]]]] = None,
    beam_width: Optional[int] = None,
    probe_results: Optional[Sequence[Any]] = None,
    ex_records: Optional[Sequence[EXRecord]] = None,
) -> Dict[str, Any]:
    """Compute the full tab:mechanism + fig:error_attr block for one dataset.

    All inputs except ``schema`` and ``predicted_cyphers`` are optional; a
    quantity is reported as None when its inputs are absent, so partial runs
    still produce a usable table.
    """
    recall = None
    if beams is not None and golds is not None:
        per_q = [
            gold_metapath_recall_at_b(beam, gold, beam_width)
            for beam, gold in zip(beams, golds)
        ]
        scored = [value for value in per_q if value is not None]
        recall = sum(scored) / len(scored) if scored else None

    return {
        "gold_metapath_recall_at_b": recall,
        "probe_precision": probe_precision(probe_results) if probe_results is not None else None,
        "empty_result_accuracy": empty_result_accuracy(ex_records) if ex_records is not None else None,
        "hallucination_attribution": hallucination_attribution(predicted_cyphers, schema),
    }
