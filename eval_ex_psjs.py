#!/usr/bin/env python3
"""eval_ex_psjs.py — score driver outputs with the CypherBench evaluator.

Computes Execution Accuracy (EX) and Provenance-Subgraph Jaccard Similarity
(PSJS) for each {graph, pred, gold} row against the live per-graph Neo4j
(ports 15062-15068), then aggregates per config and per graph. This is the
measured EX/PSJS that the joint-catalog-vs-per-query comparison reports.

Usage (on the box, Neo4j up, venv python):
    python eval_ex_psjs.py \
        --inputs outputs/ablation_catalog.jsonl outputs/ablation_per_query.jsonl \
        --cypherbench /disk2/hyx/cypher/cypherbench \
        --out outputs/ex_psjs_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# CypherBench graph -> bolt port (matches graph_connector.CYPHERBENCH_PORTS).
PORTS = {
    "company": 15062, "fictional_character": 15063, "flight_accident": 15064,
    "geography": 15065, "movie": 15066, "nba": 15067, "politics": 15068,
}


def get_connectors(cb_root: Path, host: str, user: str, pw: str, graphs):
    sys.path.insert(0, str(cb_root))
    from cypherbench.neo4j_connector import Neo4jConnector  # noqa: E402
    conns = {}
    for g in graphs:
        conns[g] = Neo4jConnector(name=g, host=host, port=PORTS[g],
                                  username=user, password=pw)
    return conns


def score_rows(rows: List[Dict[str, Any]], conns) -> List[Dict[str, Any]]:
    from cypherbench.metrics import (  # noqa: E402
        execution_accuracy, provenance_subgraph_jaccard_similarity,
    )
    out = []
    for r in rows:
        g = r.get("graph")
        pred = (r.get("pred") or "").strip()
        gold = (r.get("gold") or "").strip()
        rec = {"config": r.get("config"), "graph": g, "qid": r.get("qid"),
               "ex": 0.0, "psjs": 0.0, "scored": False, "err": r.get("error")}
        if g not in conns or not pred or not gold:
            out.append(rec)
            continue
        conn = conns[g]
        try:
            rec["ex"] = float(execution_accuracy(pred, gold, conn))
        except Exception as e:
            rec["err"] = f"EX:{type(e).__name__}:{e}"
        try:
            rec["psjs"] = float(provenance_subgraph_jaccard_similarity(pred, gold, conn))
        except Exception as e:
            rec["err"] = (rec["err"] or "") + f" PSJS:{type(e).__name__}:{e}"
        rec["scored"] = True
        out.append(rec)
    return out


def aggregate(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cfg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in scored:
        by_cfg[s["config"]].append(s)
    summary = {}
    for cfg, rows in by_cfg.items():
        n = len(rows)
        ex = statistics.mean(r["ex"] for r in rows) if n else 0.0
        psjs = statistics.mean(r["psjs"] for r in rows) if n else 0.0
        per_graph = {}
        gset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            gset[r["graph"]].append(r)
        for g, gr in gset.items():
            per_graph[g] = {
                "n": len(gr),
                "ex": round(statistics.mean(x["ex"] for x in gr), 4),
                "psjs": round(statistics.mean(x["psjs"] for x in gr), 4),
            }
        summary[cfg] = {
            "n": n,
            "ex": round(ex, 4),
            "psjs": round(psjs, 4),
            "n_pred_empty": sum(1 for r in rows if not r.get("scored")),
            "per_graph": per_graph,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--cypherbench", type=Path,
                    default=Path(os.environ.get("CYPHERBENCH_DIR", ".")),
                    help="CypherBench repo root (provides cypherbench.metrics / "
                         ".neo4j_connector; default: $CYPHERBENCH_DIR)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default="cypherbench")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for p in args.inputs:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    graphs = sorted({r.get("graph") for r in rows if r.get("graph") in PORTS})
    conns = get_connectors(args.cypherbench, args.host, args.user, args.password, graphs)
    scored = score_rows(rows, conns)
    summary = aggregate(scored)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "per_row": scored}, fh, ensure_ascii=False, indent=2)

    # Human-readable comparison.
    print("\n=== EX / PSJS by config ===")
    print(f"{'config':14s} {'n':>5s} {'EX':>8s} {'PSJS':>8s} {'pred_empty':>11s}")
    for cfg, s in summary.items():
        print(f"{cfg:14s} {s['n']:>5d} {s['ex']:>8.4f} {s['psjs']:>8.4f} {s['n_pred_empty']:>11d}")
    if "catalog" in summary and "per_query" in summary:
        dex = summary["catalog"]["ex"] - summary["per_query"]["ex"]
        dpsjs = summary["catalog"]["psjs"] - summary["per_query"]["psjs"]
        print(f"\nGAP (catalog - per_query):  EX {dex:+.4f}   PSJS {dpsjs:+.4f}")
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
