"""
Sanity-check the ablation outputs before handing them to the evaluator.

Checks for each output file (bench × B):
  - file exists and has the expected line count
  - every record carries `subgraph_retrieval_result`
  - the recorded `config_effective.beam_width` matches the file name
  - error rate (records with `subgraph_retrieval_result.error == True`)
  - average number of triples returned (rough quality signal)

Usage:
    python verify_results.py
    PROJECT_ROOT=/your/path python verify_results.py
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or os.environ.get("METACYPHER_DATA_DIR") or (Path(__file__).resolve().parents[2] / "data"))
SUBGRAPH_DIR = PROJECT_ROOT / "subgraph"
OUT_DIR = SUBGRAPH_DIR / "final" / "ablation_beam"
SUBSET = {
    "cypherbench": SUBGRAPH_DIR / "subset" / "cypherbench_sampled_200.jsonl",
    "mtq":         SUBGRAPH_DIR / "subset" / "mtq_sampled_200.jsonl",
}
B_VALUES = [1, 5, 10, 20, 40, 60]


def expected_lines(bench):
    p = SUBSET[bench]
    if not p.exists():
        return None
    return sum(1 for _ in p.open(encoding="utf-8"))


def check_one(path: Path, bench: str, b: int, want_lines):
    if not path.exists():
        return ["MISSING"]

    errs = []
    n = 0
    err_records = 0
    bw_mismatch = 0
    missing_field = 0
    triples_counts = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            n += 1
            try:
                d = json.loads(line)
            except Exception as e:
                errs.append(f"bad JSON on line {n}: {e}")
                continue

            sgr = d.get("subgraph_retrieval_result")
            if sgr is None:
                missing_field += 1
                continue
            if sgr.get("error"):
                err_records += 1
                continue
            cfg = sgr.get("config_effective", {})
            bw = cfg.get("beam_width")
            if bw is not None and bw != b:
                bw_mismatch += 1
            nt = sgr.get("num_triples", 0)
            triples_counts.append(nt)

    if want_lines is not None and n != want_lines:
        errs.append(f"line count {n} != expected {want_lines}")
    if missing_field:
        errs.append(f"{missing_field} records missing subgraph_retrieval_result")
    if bw_mismatch:
        errs.append(f"{bw_mismatch} records have wrong beam_width (expected {b})")
    if err_records:
        errs.append(f"{err_records}/{n} records had retrieval errors")
    if triples_counts:
        avg_triples = sum(triples_counts) / len(triples_counts)
        errs.append(f"avg num_triples = {avg_triples:.1f}")
    return errs or ["OK"]


def main():
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"OUT_DIR      = {OUT_DIR}")
    if not OUT_DIR.exists():
        print(f"[FATAL] Output dir not found. Run run_ablation.py first.")
        return 1

    print()
    print(f"{'bench':<15} {'B':>4}   notes")
    any_problem = False
    for bench in ["cypherbench", "mtq"]:
        want = expected_lines(bench)
        for b in B_VALUES:
            path = OUT_DIR / f"{bench}_B{b}.jsonl"
            notes = check_one(path, bench, b, want)
            joined = "; ".join(notes)
            print(f"  {bench:<13} {b:>4}   {joined}")
            if any(s.startswith(("MISSING", "bad JSON", "line count", "records missing",
                                  "records have wrong", "records had"))
                   for s in notes):
                any_problem = True

    print()
    if any_problem:
        print("[RESULT] Some files have problems -- inspect above and re-run those B values.")
        return 1
    else:
        print("[RESULT] All ablation outputs look healthy.")
        print(f"Hand off the 12 files in {OUT_DIR} to the evaluator.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
