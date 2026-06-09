"""
Beam-size ablation sweep for MetaCypher.

Runs the subgraph retrieval on the 200-sample subset across:
    B ∈ {1, 5, 10, 20, 40, 60}
    benchmark ∈ {cypherbench, mtq}
= 12 runs total. Each run writes a JSONL with the same record schema as
all_subgraph_set.py's normal output, plus a `subgraph_retrieval_result.
config_effective.beam_width` field that records which B was used.

Idempotent: skips any run whose output file already exists and has the right
line count. Use --force to override, or delete the file.

Place this file in the SAME directory as all_subgraph_set.py and run:
    cd <subgraph project dir>
    python run_ablation.py                              # full sweep
    python run_ablation.py --bench cypherbench          # one benchmark only
    python run_ablation.py --b 20                       # one B value only
    python run_ablation.py --b 20 --bench mtq           # one cell only
    python run_ablation.py --dry-run                    # print plan, do nothing
    python run_ablation.py --force                      # re-run even if output exists
"""

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or os.environ.get("METACYPHER_DATA_DIR") or (Path(__file__).resolve().parents[2] / "data"))
SUBGRAPH_DIR = PROJECT_ROOT / "subgraph"

# Where ablation outputs go (one .jsonl per B per benchmark)
OUT_DIR = SUBGRAPH_DIR / "final" / "ablation_beam"
LOG_DIR = OUT_DIR / "logs"

# Input subsets (must exist before running this script)
SUBSET = {
    "cypherbench": SUBGRAPH_DIR / "subset" / "cypherbench_sampled_200.jsonl",
    "mtq":         SUBGRAPH_DIR / "subset" / "mtq_sampled_200.jsonl",
}

# The B values to sweep (per 学长: {1, 5, 10, 20, 40, 60})
B_VALUES = [1, 5, 10, 20, 40, 60]

# candidate_width policy. Historical run used beam=30, candidate=30 (1:1).
# We keep the same pairing — varying just B keeps the ablation interpretable.
# If you want to fix candidate_width at a constant, change this function.
def candidate_width_for(b: int) -> int:
    return b


def out_path(bench: str, b: int) -> Path:
    return OUT_DIR / f"{bench}_B{b}.jsonl"


def expected_line_count(bench: str) -> int:
    src = SUBSET[bench]
    if not src.exists():
        return -1
    return sum(1 for _ in src.open(encoding="utf-8"))


def already_done(bench: str, b: int) -> bool:
    p = out_path(bench, b)
    if not p.exists():
        return False
    have = sum(1 for _ in p.open(encoding="utf-8"))
    want = expected_line_count(bench)
    return want > 0 and have == want


def run_one(bench: str, b: int) -> tuple[bool, str]:
    """Patch all_subgraph_set globals and call its main(). Returns (ok, msg)."""
    import all_subgraph_set as M

    M.INPUT_JSONL_PATH = SUBSET[bench]
    M.OUTPUT_JSONL_PATH = out_path(bench, b)
    M.POOL_NAME = "cypherbench" if bench == "cypherbench" else "mtq"
    M.SCHEMA_DIR = PROJECT_ROOT / "schema"

    M.BEAM_K = b
    M.EXEC_CAND_K = candidate_width_for(b)
    # everything else (QUERY_LIMIT, FINAL_TRIPLES_TOPK, TOP_P_PATTERNS, ...) left at defaults

    # Ensure output dir exists
    M.OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"  >>> bench={bench}  B={b}  cand={M.EXEC_CAND_K}  "
          f"pool={M.POOL_NAME}  out={M.OUTPUT_JSONL_PATH}")
    t0 = time.time()
    try:
        M.main()
        dt = time.time() - t0
        # sanity check: line count
        n = sum(1 for _ in M.OUTPUT_JSONL_PATH.open(encoding="utf-8"))
        want = expected_line_count(bench)
        if n != want:
            return False, (f"output line count {n} != expected {want}  "
                           f"(took {dt:.1f}s)")
        return True, f"OK  ({dt:.1f}s, {n} lines)"
    except Exception as e:
        dt = time.time() - t0
        return False, f"EXCEPTION after {dt:.1f}s: {e!r}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["cypherbench", "mtq", "all"], default="all")
    p.add_argument("--b", type=int, default=None,
                   help="Run only this B value (default: full sweep)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if output file already complete")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan only, do not run")
    return p.parse_args()


def main():
    args = parse_args()
    benches = ["cypherbench", "mtq"] if args.bench == "all" else [args.bench]
    bs = B_VALUES if args.b is None else [args.b]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"OUT_DIR      = {OUT_DIR}")
    print(f"Plan: {len(benches)} benchmark(s) x {len(bs)} B value(s) = {len(benches)*len(bs)} runs")
    for bench in benches:
        n = expected_line_count(bench)
        if n < 0:
            print(f"  [{bench}] subset file MISSING: {SUBSET[bench]}")
        else:
            print(f"  [{bench}] subset has {n} cases")

    plan = []
    for bench in benches:
        for b in bs:
            if SUBSET[bench].exists() is False:
                plan.append((bench, b, "SKIP (subset missing)"))
            elif already_done(bench, b) and not args.force:
                plan.append((bench, b, "SKIP (already done)"))
            else:
                plan.append((bench, b, "RUN"))

    print()
    print(f"{'bench':<15} {'B':>4}   action")
    for bench, b, action in plan:
        print(f"  {bench:<13} {b:>4}   {action}")

    if args.dry_run:
        print("\n--dry-run: not executing")
        return 0

    todo = [(b_, B_) for (b_, B_, a) in plan if a == "RUN"]
    if not todo:
        print("\nNothing to do.")
        return 0

    print(f"\nExecuting {len(todo)} run(s).\n")
    results = []
    t_sweep = time.time()
    for i, (bench, b) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {bench} B={b}")
        ok, msg = run_one(bench, b)
        results.append((bench, b, ok, msg))
        status = "OK " if ok else "ERR"
        print(f"  [{status}] {msg}\n")

    dt_sweep = time.time() - t_sweep
    print("=" * 60)
    print(f"Sweep finished in {dt_sweep/60:.1f} min")
    n_ok = sum(1 for *_, ok, _ in results if ok)
    n_err = len(results) - n_ok
    print(f"  {n_ok} succeeded, {n_err} failed")
    for bench, b, ok, msg in results:
        if not ok:
            print(f"  [FAIL] {bench} B={b}: {msg}")
    print(f"Outputs in: {OUT_DIR}")
    print("Next: python verify_results.py")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
