#!/usr/bin/env python3
"""run_per_query_vs_catalog.py — ablation driver: offline joint catalog vs. per-query counting.

Compares the two cardinality-estimation regimes of MetaCypher (paper §"Joint
catalog versus per-query counting") on a sampled subset of CypherBench test
questions, end-to-end (question -> predicted Cypher), recording probe stats.

  config=catalog    : load the offline catalog_<graph>.json (build_catalog output,
                      saved via save_catalog) — every joint cardinality is a free
                      lookup, mu_P exact.
  config=per_query  : construct PerQueryCountingCatalog(schema, count_fn,
                      probe_budget) and call pqc.start_query() before each
                      question — joint support measured live under a budget and
                      discarded between queries.

Both regimes are injected via the SAME seam: SubgraphRetriever(catalog=...).
PerQueryCountingCatalog is a duck-typed stand-in (it implements .get(key),
.median_card, .anchor_index — the three members PathScorer reads).

USAGE (run on the remote box with Neo4j up on 15062-15068 and the LLM env set):

    export METACYPHER_DATA_DIR=/disk2/hyx/cypher/metacypher_data   # PROJECT_ROOT
    export METACYPHER_SCHEMA_DIR=$METACYPHER_DATA_DIR/schema       # <graph>.json live here
    # LLM env (ECNU/vLLM): METACYPHER_VLLM_BASE_URL / _MODEL / _API_KEY
    # Neo4j env: NEO4J_HOST / NEO4J_PASSWORD (default password "cypherbench")

    python run_per_query_vs_catalog.py \
        --config per_query \
        --n-per-graph 50 --seed 0 \
        --test-json /disk2/hyx/cypher/cypherbench/benchmark/test.json \
        --catalog-dir $METACYPHER_SCHEMA_DIR/catalogs \
        --out outputs/ablation_per_query.jsonl

Output: one JSON line per question with keys
    {config, graph, qid, question, gold, pred, error, probe_stats, instr}
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# The metacypher package modules use flat imports (e.g. `from config import ...`),
# so put the package dir on sys.path and import the modules directly.
_PKG_DIR = Path(__file__).resolve().parent / "metacypher"
sys.path.insert(0, str(_PKG_DIR))

import config  # noqa: E402  (metacypher/config.py — reads METACYPHER_* env vars)
from catalog import (  # noqa: E402
    load_catalog,
    make_neo4j_count_fn,
)
from per_query_counting import PerQueryCountingCatalog  # noqa: E402
from graph_connector import create_pool, CYPHERBENCH_PORTS  # noqa: E402
from subgraph_retrieval import RetrievalConfig  # noqa: E402
from triple_retrieval import EnhancedSubgraphRetriever  # noqa: E402
from generation import CypherGenerator  # noqa: E402
from all_subgraph_set import build_config_for_batch  # noqa: E402  (real pipeline beam config)
from query_analyze import (  # noqa: E402
    compact_schema_for_prompt,
    build_prompt,
    call_llm,
    safe_parse_json,
    supplement_entity_mappings,
)
import instrumentation as _instr  # noqa: E402

# The 7 graphs that actually exist in CypherBench test + graph_connector ports.
DEFAULT_GRAPHS = [
    "company",
    "fictional_character",
    "flight_accident",
    "geography",
    "movie",
    "nba",
    "politics",
]


# ---------------------------------------------------------------------------
# Data loading / sampling
# ---------------------------------------------------------------------------
def load_test(test_json: Path) -> List[Dict[str, Any]]:
    with test_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, list), "test.json is expected to be a top-level list"
    return data


def sample_per_graph(
    records: List[Dict[str, Any]],
    graphs: List[str],
    n_per_graph: int,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    by_graph: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        g = r.get("graph")
        if g in graphs:
            by_graph[g].append(r)
    rng = random.Random(seed)
    sampled: Dict[str, List[Dict[str, Any]]] = {}
    for g in graphs:
        pool_recs = by_graph.get(g, [])
        if not pool_recs:
            print(f"[WARN] no test records for graph={g}", file=sys.stderr)
            sampled[g] = []
            continue
        k = min(n_per_graph, len(pool_recs))
        sampled[g] = rng.sample(pool_recs, k)
    return sampled


# ---------------------------------------------------------------------------
# Schema loading (mirrors all_subgraph_set.load_schema_for_graph)
# ---------------------------------------------------------------------------
def load_schema_for_graph(graph: str) -> tuple[Dict[str, Any], str]:
    schema_path = config.SCHEMA_DIR / f"{graph}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"schema file not found for graph={graph}: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    return schema, str(schema_path)


# ---------------------------------------------------------------------------
# Stage 1 analysis (mirrors skill._stage1_analysis but with a known schema)
# ---------------------------------------------------------------------------
def run_analysis(question: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    schema_compact = compact_schema_for_prompt(schema)
    prompt = build_prompt(question, schema_compact)
    raw = call_llm(prompt)
    parsed, err = safe_parse_json(raw)
    if err is not None or parsed is None:
        raise ValueError(f"analysis parse error: {err}; raw={str(raw)[:200]!r}")
    return supplement_entity_mappings(question, schema_compact, parsed)


# ---------------------------------------------------------------------------
# Retriever construction — THE INJECTION SEAM
# ---------------------------------------------------------------------------
def build_retrieval_config(top_k: int) -> RetrievalConfig:
    # Pattern-only retrieval (no torch/faiss). Mirrors skill._stage2_retrieval.
    return RetrievalConfig(
        top_m_anchors=5,
        top_p_patterns=top_k * 2,
        top_k_paths=top_k * 4,
        min_path_score=0.0,
        max_path_length=5,
        pattern_score_threshold=0.0,
        use_beam_search=False,
        debug_mode=False,
        debug_print_query=False,
        debug_print_results=False,
        debug_print_cypher=False,
    )


def build_retriever(
    cfg: RetrievalConfig,
    schema: Dict[str, Any],
    schema_path: str,
    catalog_obj: Any,
) -> EnhancedSubgraphRetriever:
    """Construct the retriever with the catalog injected.

    NOTE ON THE SEAM
    ----------------
    `SubgraphRetriever.__init__(... , catalog=...)` (subgraph_retrieval.py:120)
    and `PathScorer(config, catalog=...)` (path_scorer.py:50) DO accept a
    catalog and PathScorer.score_paths reads `catalog.get(sig)` and
    `catalog.median_card` — exactly the duck-typed members that
    PerQueryCountingCatalog implements. So injection of *both* a CatalogResult
    (config=catalog) and a PerQueryCountingCatalog (config=per_query) works
    through this one constructor arg.

    HOWEVER: EnhancedSubgraphRetriever.__init__ (triple_retrieval.py:26) has the
    signature

        __init__(self, config, schema, full_schema_path="",
                 enable_semantics=False, template_base_path="")

    i.e. it calls super().__init__(config, schema, full_schema_path) and does
    NOT forward a `catalog=` argument to SubgraphRetriever. Therefore there is
    currently NO way to inject a catalog through EnhancedSubgraphRetriever.

    *** ONE-LINE CODE CHANGE REQUIRED in metacypher/triple_retrieval.py ***
    Add a `catalog=None` parameter to EnhancedSubgraphRetriever.__init__ and
    forward it to super():

        def __init__(self, config, schema, full_schema_path="",
                     enable_semantics=False, template_base_path="",
                     catalog=None):                      # <-- add
            super().__init__(config, schema, full_schema_path,
                             catalog=catalog)             # <-- add catalog=
            ...

    (BaseSubgraphRetriever == SubgraphRetriever already accepts catalog=.)

    Until that change lands, the line below will raise TypeError on the
    `catalog=` kwarg. It is written against the POST-FIX interface on purpose,
    per the task instruction to use the real seam rather than invent an API.
    """
    return EnhancedSubgraphRetriever(
        config=cfg,
        schema=schema,
        full_schema_path=schema_path,
        enable_semantics=False,
        # CypherBench schema files already carry relations[].{subj_label,label,
        # obj_label}; the template loader reads exactly those (match_template/
        # pattern are optional NL fields used only when enable_semantics=True).
        # So the schema dir itself serves as the template base.
        template_base_path=str(config.SCHEMA_DIR),
        catalog=catalog_obj,
    )


# ---------------------------------------------------------------------------
# Retrieval -> serialised top_triples (mirrors skill._stage2_retrieval tail)
# ---------------------------------------------------------------------------
def retrieve_top_triples(
    retriever: EnhancedSubgraphRetriever,
    pool: Any,
    graph: str,
    question: str,
    analysis: Dict[str, Any],
    top_k: int,
) -> List[Dict[str, Any]]:
    case = {
        "qid": "ablation_query",
        "nl_question": question,
        "graph": graph,
        "analysis": analysis,
    }
    result = retriever.retrieve_with_triples(pool=pool, db_name=graph, case=case)
    triples_objs = result.get("triples", []) or []

    out: List[Dict[str, Any]] = []
    for t in triples_objs:
        try:
            d = t.to_dict()
        except Exception:
            d = {
                "triple_id": getattr(t, "triple_id", None),
                "cypher_query": getattr(t, "cypher_query", None),
                "cypher_pattern": getattr(t, "cypher_pattern", None),
                "semantic_description": getattr(t, "semantic_description", ""),
                "score": getattr(t, "score", 0.0),
                "sample_paths": getattr(t, "sample_paths", []),
            }
        d.pop("subgraph", None)
        d.pop("embedding", None)
        d.setdefault("sample_paths", [])
        if not d.get("semantic_description"):
            d["semantic_description"] = d.get("cypher_pattern", "")
        if not d.get("cypher_query"):
            d["cypher_query"] = d.get("cypher_pattern", "")
        out.append(d)

    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out[:top_k]


# ---------------------------------------------------------------------------
# Generation (mirrors skill._stage3_generation)
# ---------------------------------------------------------------------------
def run_generation(
    question: str,
    analysis: Dict[str, Any],
    top_triples: List[Dict[str, Any]],
    top_k: int,
) -> str:
    generator = CypherGenerator(
        model=config.VLLM_MODEL,
        base_url=config.VLLM_BASE_URL,
        api_key=config.VLLM_API_KEY,
        temperature=0.1,
        max_tokens=2048,
        top_k_triples=top_k,
    )
    item = {
        "nl_question": question,
        "analysis": analysis,
        "subgraph_retrieval_result": {"top_triples": top_triples},
    }
    return generator.generate_cypher(item) or ""


# ---------------------------------------------------------------------------
# Correction (mirrors skill._stage4_correction / correction.py repair_one)
# ---------------------------------------------------------------------------
_REPAIRER = None


def run_correction(qid: str, graph: str, question: str, pred_cypher: str,
                   schema: Dict[str, Any]) -> str:
    """Repair the generated Cypher against the graph schema (full pipeline)."""
    global _REPAIRER
    from correction import CypherRepairer, RepairConfig, schema_to_prompt_text
    if _REPAIRER is None:
        _REPAIRER = CypherRepairer(RepairConfig(
            model=config.VLLM_MODEL, base_url=config.VLLM_BASE_URL,
            api_key=config.VLLM_API_KEY, temperature=0.0, max_tokens=2048,
            enable_memory=False,
        ))
    text = schema_to_prompt_text(schema)
    repaired = _REPAIRER.repair_one(qid, graph, question, pred_cypher, text)
    return (repaired or pred_cypher).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=["catalog", "per_query"], required=True)
    ap.add_argument("--n-per-graph", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graphs", nargs="+", default=DEFAULT_GRAPHS)
    ap.add_argument(
        "--test-json",
        type=Path,
        default=Path(os.environ.get("CYPHERBENCH_DIR", ".")) / "benchmark" / "test.json",
        help="CypherBench test.json (default: $CYPHERBENCH_DIR/benchmark/test.json)",
    )
    ap.add_argument(
        "--catalog-dir",
        type=Path,
        default=None,
        help="dir holding catalog_<graph>.json (config=catalog). "
        "Default: $METACYPHER_SCHEMA_DIR/catalogs",
    )
    ap.add_argument("--probe-budget", type=int, default=20,
                    help="per-query COUNT probe budget for config=per_query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--narrow", action="store_true",
                    help="use the narrow pattern-only retrieval config instead of "
                         "the real beam-search batch config (default: beam config)")
    ap.add_argument("--no-correction", action="store_true",
                    help="skip the correction/repair stage (default: run it)")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="stop after retrieval (record probe_stats only; no "
                         "generation/correction). For measuring true probe cost.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    catalog_dir = args.catalog_dir or (config.SCHEMA_DIR / "catalogs")

    records = load_test(args.test_json)
    sampled = sample_per_graph(records, args.graphs, args.n_per_graph, args.seed)

    # Use the REAL batch-pipeline retrieval config (beam search, beam_width=30,
    # top_p_patterns=30, top_k_paths=50, max_expansion_depth=2) so the funnel
    # explores many typed-path candidates per question. This is the regime where
    # a per-query 20-probe budget genuinely binds (covers only a fraction of
    # candidates), unlike the narrow pattern-only config which scored ~1.5/q.
    if args.narrow:
        cfg = build_retrieval_config(args.top_k)
    else:
        cfg = build_config_for_batch()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    # One pool for all CypherBench graphs (graph_connector dispatches by db_name).
    with create_pool("cypherbench") as pool, args.out.open("w", encoding="utf-8") as out_f:
        for graph in args.graphs:
            if graph not in CYPHERBENCH_PORTS:
                print(f"[WARN] graph={graph} not in CYPHERBENCH_PORTS; skipping", file=sys.stderr)
                continue
            graph_recs = sampled.get(graph, [])
            if not graph_recs:
                continue

            schema, schema_path = load_schema_for_graph(graph)

            # ---- build the catalog object for this graph (per-graph reuse) ----
            count_fn = make_neo4j_count_fn(graph, pool)
            if args.config == "catalog":
                cat_path = catalog_dir / f"catalog_{graph}.json"
                if not cat_path.exists():
                    raise FileNotFoundError(
                        f"offline catalog not found: {cat_path} "
                        f"(build it with catalog.build_catalog + save_catalog)"
                    )
                catalog_obj: Any = load_catalog(str(cat_path))
                pqc: Optional[PerQueryCountingCatalog] = None
            else:  # per_query
                pqc = PerQueryCountingCatalog(schema, count_fn, probe_budget=args.probe_budget)
                catalog_obj = pqc

            # ---- build retriever ONCE per graph with catalog injected ----------
            # (see build_retriever docstring: needs the triple_retrieval.py seam)
            retriever = build_retriever(cfg, schema, schema_path, catalog_obj)

            for rec in graph_recs:
                qid = rec.get("qid", "")
                question = rec.get("nl_question", "")
                gold = rec.get("gold_cypher", "")

                row: Dict[str, Any] = {
                    "config": args.config,
                    "graph": graph,
                    "qid": qid,
                    "question": question,
                    "gold": gold,
                    "pred": None,
                    "error": None,
                    "probe_stats": None,
                    "instr": None,
                }

                # Per-query lifecycle for per-query counting: reset budget+memo.
                if pqc is not None:
                    pqc.start_query()

                try:
                    with _instr.track_query() as stats:
                        analysis = run_analysis(question, schema)
                        top_triples = retrieve_top_triples(
                            retriever, pool, graph, question, analysis, args.top_k
                        )
                        if not args.retrieval_only:
                            pred = run_generation(question, analysis, top_triples, args.top_k)
                            if not args.no_correction and pred:
                                pred = run_correction(qid, graph, question, pred, schema)
                            row["pred"] = pred
                    row["instr"] = stats.as_dict()
                    row["n_candidates"] = len(top_triples)
                except Exception as exc:  # keep going; record the failure
                    row["error"] = f"{type(exc).__name__}: {exc}"

                if pqc is not None:
                    row["probe_stats"] = pqc.query_stats()

                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                n_written += 1

    print(f"[done] wrote {n_written} rows to {args.out}")


if __name__ == "__main__":
    main()
