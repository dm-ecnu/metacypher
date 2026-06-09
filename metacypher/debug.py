"""
快速调试单个case，便于观察Beam Search是否运行、候选数量、最终三元组内容。

使用方式:
    python debug_case.py --case /path/to/case.json --graph geography
"""

import argparse
import json
import logging
from pathlib import Path

from graph_connector import create_pool
from subgraph_retrieval import RetrievalConfig
from triple_retrieval import EnhancedSubgraphRetriever
from config import SCHEMA_DIR


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EnhancedSubgraphRetriever on single case")
    parser.add_argument("--case", required=True, type=Path, help="case JSON file")
    parser.add_argument("--graph", required=True, help="graph/db name")
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=SCHEMA_DIR,
        help="schema base directory (default: %(default)s)",
    )
    parser.add_argument(
        "--pool",
        default="cypherbench",
        help="graph connector pool name (default: %(default)s)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="top-k paths/triples to keep (default: %(default)s)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=20,
        help="beam width (default: %(default)s)",
    )
    parser.add_argument(
        "--candidate-width",
        type=int,
        default=40,
        help="candidate width (default: %(default)s)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="beam max expansion depth (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="set logging to INFO instead of WARNING",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RetrievalConfig:
    cfg = RetrievalConfig(
        top_m_anchors=5,
        top_p_patterns=30,
        top_k_paths=args.topk,
        min_path_score=0.3,
        max_path_length=5,
        pattern_score_threshold=0.0,
    )
    if hasattr(cfg, "beam_width"):
        cfg.beam_width = args.beam_width
    if hasattr(cfg, "candidate_width"):
        cfg.candidate_width = args.candidate_width
    if hasattr(cfg, "max_expansion_depth"):
        cfg.max_expansion_depth = args.max_depth
    if hasattr(cfg, "use_beam_search"):
        cfg.use_beam_search = True
    if hasattr(cfg, "max_cypher_limit"):
        cfg.max_cypher_limit = 20
    if hasattr(cfg, "max_results_per_query"):
        cfg.max_results_per_query = 20
    if hasattr(cfg, "debug_print_cypher"):
        cfg.debug_print_cypher = True
    return cfg


def main() -> None:
    args = build_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    case_data = json.loads(args.case.read_text(encoding="utf-8"))
    graph_name = args.graph
    schema_path = args.schema_dir / f"{graph_name}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"schema file missing: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    cfg = build_config(args)
    retriever = EnhancedSubgraphRetriever(
        config=cfg,
        schema=schema,
        full_schema_path=str(schema_path),
        enable_semantics=True,
        template_base_path=str(args.schema_dir / "template" / "schema_with_template"),
    )

    with create_pool(args.pool) as pool:
        result = retriever.retrieve_with_triples(pool=pool, db_name=graph_name, case=case_data)

    triples = result.get("triples", [])
    print(f"[RESULT] graph={graph_name}, total_paths={result.get('total_paths_found')}, triples={len(triples)}")
    for idx, triple in enumerate(triples[:5], 1):
        print(f"  [Triple #{idx}] {triple.triple_id} score={triple.score:.3f}")
        print(f"    signature: {triple.pattern_signature}")
        print(f"    cypher: {triple.cypher_pattern}")
        print(f"    semantic: {triple.semantic_description}")


if __name__ == "__main__":
    main()
