#!/usr/bin/env python3
"""build_all_catalogs.py — build the offline cardinality catalog per CypherBench
test graph by issuing real COUNT probes against live Neo4j (ports 15062-15068).

Writes catalog_<graph>.json under --out-dir (default $METACYPHER_SCHEMA_DIR/catalogs).
This is the offline-amortization side of the joint-catalog-vs-per-query comparison:
every joint cardinality is materialized once here so retrieval lookups are free.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent / "metacypher"
sys.path.insert(0, str(_PKG_DIR))

import config  # noqa: E402
from graph_connector import create_pool, CYPHERBENCH_PORTS  # noqa: E402
from catalog import build_catalog, make_neo4j_count_fn, save_catalog  # noqa: E402

DEFAULT_GRAPHS = [
    "company", "fictional_character", "flight_accident",
    "geography", "movie", "nba", "politics",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", nargs="+", default=DEFAULT_GRAPHS)
    ap.add_argument("--max-len", type=int, default=2,
                    help="max meta-path length probed offline (paper uses 2)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (config.SCHEMA_DIR / "catalogs")
    out_dir.mkdir(parents=True, exist_ok=True)

    with create_pool("cypherbench") as pool:
        for graph in args.graphs:
            if graph not in CYPHERBENCH_PORTS:
                print(f"[skip] {graph} not in CYPHERBENCH_PORTS", file=sys.stderr)
                continue
            schema_path = config.SCHEMA_DIR / f"{graph}.json"
            schema = json.load(schema_path.open("r", encoding="utf-8"))
            count_fn = make_neo4j_count_fn(graph, pool)
            cat = build_catalog(schema, count_fn, max_len=args.max_len)
            out_path = out_dir / f"catalog_{graph}.json"
            save_catalog(cat, str(out_path))
            print(f"[ok] {graph:20s} entries={len(cat.entries):5d} "
                  f"median_card={cat.median_card} -> {out_path}")

    print(f"[done] catalogs in {out_dir}")


if __name__ == "__main__":
    main()
