"""
Batch prediction for EnhancedSubgraphRetriever.

Input:  JSONL where each line is a full case dict (same format as test_subgraph.py's case.json)
Output: JSONL where each line is the original case + new field `subgraph_retrieval_result`

Requirements implemented:
- No debug output; only print failing case qid + question + error message.
- BeamSearch hyperparam = 30 (best-effort via hasattr/setattr on config)
- Query limit = 20 (best-effort via hasattr/setattr on config)
- Use all related patterns from the JSON as initial patterns (achieved by setting top_p_patterns large and threshold low)
- Keep top-50 triples (pattern, semantic, subgraph triple content, score)
- Limit subgraph triple content size per triple to avoid output explosion

Note:
- In your codebase, EnhancedSubgraphRetriever converts ONLY base_result['top_k_paths'] to triples.
  So "all expanded + original candidates" can only be included if SubgraphRetriever already
  keeps them in top_k_paths. This script outputs the returned triples (top_k_paths-based) and
  takes Top-50 by score.
"""

import json
import sys
import logging
import os
import datetime as _dt
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
from graph_connector import create_pool
from subgraph_retrieval import RetrievalConfig
from triple_retrieval import EnhancedSubgraphRetriever
from tqdm import tqdm

try:
    from neo4j.time import Date as Neo4jDate, DateTime as Neo4jDateTime, Time as Neo4jTime
except Exception:
    Neo4jDate = Neo4jDateTime = Neo4jTime = None

try:
    from neo4j.spatial import Point as Neo4jPoint
except Exception:
    Neo4jPoint = None


# =========================
# Paths (from config; override via env vars — see config.py)
# =========================
from config import SUBGRAPH_DIR, SCHEMA_DIR as _SCHEMA_DIR

INPUT_JSONL_PATH = SUBGRAPH_DIR / "path_route_result" / "mtq_test_seed_path.jsonl"
OUTPUT_JSONL_PATH = SUBGRAPH_DIR / "final" / "mtq_cases_with_predictions.jsonl"

# Schema json files directory (one per graph, e.g., geography.json, politics.json)
SCHEMA_DIR = _SCHEMA_DIR
POOL_NAME = "mtq"

# =========================
# User-required knobs
# =========================
BEAM_K = 30                 # beam 保留宽度（若你的 RetrievalConfig 支持该字段）
EXEC_CAND_K = 30            # 每轮执行候选数（若支持）
QUERY_LIMIT = 20            # cypher LIMIT（若支持）
FINAL_TRIPLES_TOPK = 50     # 输出 top-50 三元组

# Pattern stage: try to include ALL related patterns from case
TOP_P_PATTERNS = 30
PATTERN_SCORE_THRESHOLD = 0.0

# Reduce output size
MAX_SUBGRAPH_TRIPLES_PER_RESULT = 50          # 每个 triple 最多保留 50 条 (u,rel,v)
MAX_SEMANTIC_CHARS = 500                      # 语义描述截断
MAX_CYPHER_CHARS = 800                        # cypher pattern 截断
DEBUG_TOP_ITEMS = 3
ENABLE_DEBUG_LOG = bool(os.getenv("ALL_SUBGRAPH_DEBUG"))


def _safe_get_case_id(case: Dict[str, Any]) -> str:
    return str(case.get("qid") or case.get("id") or "UNKNOWN_QID")


def _safe_get_question(case: Dict[str, Any]) -> str:
    return str(case.get("nl_question") or case.get("question") or "")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _json_default(value: Any):
    """Safely convert Neo4j/time/point objects (and other non-serializables) to JSON."""
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    for neo_type in (Neo4jDate, Neo4jDateTime, Neo4jTime):
        if neo_type and isinstance(value, neo_type):
            try:
                return value.isoformat()
            except Exception:
                return str(value)
    if Neo4jPoint and isinstance(value, Neo4jPoint):
        data = {"srid": getattr(value, "srid", None), "x": value.x, "y": value.y}
        if hasattr(value, "z"):
            try:
                data["z"] = value.z
            except Exception:
                pass
        return data
    if isinstance(value, set):
        return list(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")


def configure_logging_quiet() -> None:
    # Only show errors; suppress info/debug logs from your modules
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s:%(name)s:%(message)s")
    for name in [
        "subgraph_retrieval",
        "triple_retrieval",
        "graph_connector",
        "__main__",
    ]:
        logging.getLogger(name).setLevel(logging.ERROR)


def build_config_for_batch() -> RetrievalConfig:
    """
    IMPORTANT:
    - Only use the fields confirmed in test_subgraph.py to avoid init signature mismatch:
      top_m_anchors, top_p_patterns, top_k_paths, min_path_score, max_path_length, pattern_score_threshold
    - For other fields (beam/search/limit), set via hasattr+setattr (best effort).
    """
    cfg = RetrievalConfig(
        top_m_anchors=5,
        top_p_patterns=TOP_P_PATTERNS,
        top_k_paths=FINAL_TRIPLES_TOPK,     # ensure we can output top-50 triples
        min_path_score=0.3,                 # do not filter early
        max_path_length=5,
        pattern_score_threshold=PATTERN_SCORE_THRESHOLD
    )

    # Best-effort: if your RetrievalConfig actually has these fields, they will be applied.
    # If not, ignored without crashing.
    best_effort_fields = {
        # beam search
        "beam_width": BEAM_K,
        "candidate_width": EXEC_CAND_K,
        "beam_candidate_k": EXEC_CAND_K,
        "max_expansion_depth": 2,
        "use_beam_search": True,
        "enable_beam_search": True,
        "beam_search": True,

        # query limits
        "max_cypher_limit": QUERY_LIMIT,
        "query_limit": QUERY_LIMIT,
        "max_results_per_query": QUERY_LIMIT,
        "optional_match_limit": QUERY_LIMIT,

        # debug switches
        "debug_mode": False,
        "debug_print_query": False,
        "debug_print_results": False,
        "debug_print_cypher": False,
    }

    for k, v in best_effort_fields.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg


def graph_to_triples_list(G: nx.MultiDiGraph, max_triples: int) -> List[Tuple[str, str, str]]:
    """
    Convert a networkx subgraph to a compact list of (u, rel, v).
    """
    out: List[Tuple[str, str, str]] = []
    try:
        for u, v, data in G.edges(data=True):
            rel = data.get("rel_type") or data.get("label") or data.get("type") or "REL"
            out.append((str(u), str(rel), str(v)))
            if len(out) >= max_triples:
                break
    except Exception:
        return []
    return out


def triple_to_jsonable(triple_obj: Any) -> Dict[str, Any]:
    """
    Your SubgraphTriple has to_dict(); we keep it but remove huge/unserializable parts,
    and add a compact subgraph triple list for your later scoring.
    """
    d: Dict[str, Any] = {}
    try:
        d = triple_obj.to_dict()
    except Exception:
        # minimal fallback
        d = {
            "triple_id": getattr(triple_obj, "triple_id", None),
            "pattern_id": getattr(triple_obj, "pattern_id", None),
            "cypher_pattern": getattr(triple_obj, "cypher_pattern", None),
            "semantic_description": getattr(triple_obj, "semantic_description", None),
            "score": getattr(triple_obj, "score", 0.0),
        }

    # remove heavy / unserializable
    if "subgraph" in d:
        del d["subgraph"]
    if "embedding" in d:
        del d["embedding"]

    # truncate long strings to reduce file size
    if isinstance(d.get("semantic_description"), str) and len(d["semantic_description"]) > MAX_SEMANTIC_CHARS:
        d["semantic_description"] = d["semantic_description"][:MAX_SEMANTIC_CHARS]
    if isinstance(d.get("cypher_pattern"), str) and len(d["cypher_pattern"]) > MAX_CYPHER_CHARS:
        d["cypher_pattern"] = d["cypher_pattern"][:MAX_CYPHER_CHARS]
    if isinstance(d.get("cypher_query"), str) and len(d["cypher_query"]) > MAX_CYPHER_CHARS:
        d["cypher_query"] = d["cypher_query"][:MAX_CYPHER_CHARS]

    # add compact subgraph triples from object.subgraph if exists
    G = getattr(triple_obj, "subgraph", None)
    if G is not None:
        d["subgraph_triples"] = graph_to_triples_list(G, MAX_SUBGRAPH_TRIPLES_PER_RESULT)
    else:
        d["subgraph_triples"] = []

    return d


def sort_and_take_topk(triples: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    def score_of(x: Dict[str, Any]) -> float:
        try:
            return float(x.get("score", 0.0))
        except Exception:
            return 0.0

    triples.sort(key=score_of, reverse=True)
    return triples[:topk]


def debug_log_case_results(
    case_id: str,
    graph_name: str,
    base_result: Dict[str, Any],
    triples_objs: List[Any]
) -> None:
    """Print intermediate details for diagnosis."""
    print(f"\n[DEBUG] Case {case_id} graph={graph_name}")
    final_paths = base_result.get("final_paths", []) or []
    print(f"[DEBUG] Final paths count: {len(final_paths)}")
    for idx, path in enumerate(final_paths[:DEBUG_TOP_ITEMS], 1):
        signature = getattr(path, "get_signature", lambda: "")()
        description = getattr(path, "description", "")
        cypher_query = getattr(path, "cypher_query", "")
        pattern_structure = getattr(path, "pattern_structure", {})
        anchor_nodes = getattr(path, "anchor_nodes", [])
        sample_paths = getattr(path, "sample_paths", [])
        total_hits = getattr(path, "sample_paths_total_hits", len(sample_paths))
        print(f"  [Path #{idx}] id={getattr(path, 'path_id', '')}")
        print(f"    signature: {signature}")
        print(f"    cypher_query: {cypher_query}")
        print(f"    description: {description}")
        print(f"    pattern_structure: {pattern_structure}")
        print(f"    anchor_nodes: {anchor_nodes}")
        print(f"    sample_paths_total_hits: {total_hits}, sample_paths_kept={len(sample_paths)}")
    print(f"[DEBUG] Triples count: {len(triples_objs)}")
    for idx, triple in enumerate(triples_objs[:DEBUG_TOP_ITEMS], 1):
        triple_dict = triple_to_jsonable(triple)
        print(f"  [Triple #{idx}] triple_id={triple_dict.get('triple_id')}, score={triple_dict.get('score')}")
        print(f"    cypher_query: {triple_dict.get('cypher_query')}")
        print(f"    pattern_signature: {triple_dict.get('pattern_signature')}")
        print(f"    pattern_structure: {triple_dict.get('pattern_structure')}")
        print(f"    anchor_nodes: {triple_dict.get('anchor_nodes')}")
        print(f"    sample_paths_total_hits: {triple_dict.get('sample_paths_total_hits')}, sample_paths_kept={len(triple_dict.get('sample_paths', []))}")


# ==== Schema loader and retriever cache ====
_SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}

def load_schema_for_graph(graph_name: str) -> Tuple[Dict[str, Any], str]:
    """Load schema dict for a given graph name.

    Expected schema path: {SCHEMA_DIR}/{graph_name}.json (see config.py)
    Returns: (schema_dict, full_schema_path)
    """
    if graph_name in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[graph_name], str(SCHEMA_DIR / f"{graph_name}.json")

    schema_path = SCHEMA_DIR / f"{graph_name}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found for graph={graph_name}: {schema_path}")

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    _SCHEMA_CACHE[graph_name] = schema
    return schema, str(schema_path)


def main() -> None:
    configure_logging_quiet()

    cfg = build_config_for_batch()

    retrievers: Dict[str, EnhancedSubgraphRetriever] = {}
    total_rows = count_jsonl_rows(INPUT_JSONL_PATH)

    with create_pool(POOL_NAME) as pool, OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as out_f:
        for case in tqdm(read_jsonl(INPUT_JSONL_PATH), total=total_rows, desc="Cases"):
            qid = _safe_get_case_id(case)
            question = _safe_get_question(case)

            try:
                if "graph" not in case:
                    raise ValueError("case missing required field: graph")
                if "nl_question" not in case and "question" not in case:
                    raise ValueError("case missing required field: nl_question/question")

                graph_name = case["graph"]
                if graph_name not in retrievers:
                    schema, schema_path = load_schema_for_graph(graph_name)
                    retrievers[graph_name] = EnhancedSubgraphRetriever(
                        config=cfg,
                        schema=schema,
                        full_schema_path=schema_path,
                        enable_semantics=True,
                        template_base_path=str(SCHEMA_DIR / "template" / "schema_with_template")
                    )

                result = retrievers[graph_name].retrieve_with_triples(
                    pool=pool,
                    db_name=graph_name,
                    case=case
                )

                triples_objs = result.get("triples", []) or []
                triples_json = [triple_to_jsonable(t) for t in triples_objs]
                triples_json = sort_and_take_topk(triples_json, FINAL_TRIPLES_TOPK)
                case_id = qid
                if ENABLE_DEBUG_LOG:
                    debug_log_case_results(case_id, graph_name, result, triples_objs)

                out_case = deepcopy(case)
                out_case["subgraph_retrieval_result"] = {
                    "config_effective": {
                        "top_m_anchors": cfg.top_m_anchors,
                        "top_p_patterns": cfg.top_p_patterns,
                        "top_k_paths": cfg.top_k_paths,
                        "min_path_score": cfg.min_path_score,
                        "max_path_length": cfg.max_path_length,
                        "pattern_score_threshold": cfg.pattern_score_threshold,

                        # best-effort fields (only if exist)
                        **({ "beam_width": getattr(cfg, "beam_width") } if hasattr(cfg, "beam_width") else {}),
                        **({ "candidate_width": getattr(cfg, "candidate_width") } if hasattr(cfg, "candidate_width") else {}),
                        **({ "max_cypher_limit": getattr(cfg, "max_cypher_limit") } if hasattr(cfg, "max_cypher_limit") else {}),
                        **({ "max_results_per_query": getattr(cfg, "max_results_per_query") } if hasattr(cfg, "max_results_per_query") else {}),
                        # Add schema path for traceability
                        "schema_path": str(SCHEMA_DIR / f"{case['graph']}.json"),
                    },
                    "strategy_type": result.get("strategy_type"),
                    "num_patterns_used": result.get("num_patterns_used"),
                    "total_paths_found": result.get("total_paths_found"),
                    "num_triples": len(triples_json),
                    # 输出：模式、语义、子图内容三元组、分数等（Top-50）
                    "top_triples": triples_json,
                }

                out_f.write(json.dumps(out_case, ensure_ascii=False, default=_json_default) + "\n")
                out_f.flush()

            except Exception as e:
                # Only print failing case's question and qid as required
                print(f"[ERROR] qid={qid} graph={case.get('graph')} question={question} err={e}", file=sys.stderr)
                out_case = deepcopy(case)
                out_case["subgraph_retrieval_result"] = {
                    "error": True,
                    "message": str(e),
                }
                out_f.write(json.dumps(out_case, ensure_ascii=False, default=_json_default) + "\n")
                out_f.flush()


if __name__ == "__main__":
    main()
