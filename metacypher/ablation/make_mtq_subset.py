"""
Generate the MTQ 200-sample subset using the same seed=42 logic as the existing
CypherBench subset (sub_test.py). Run once before the ablation sweep.

Usage:
    python make_mtq_subset.py            # uses default paths below
    PROJECT_ROOT=/your/path python make_mtq_subset.py

Idempotent: if the output file already exists with 200 lines, it does nothing.
Set FORCE=1 in env to regenerate anyway.
"""

import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or os.environ.get("METACYPHER_DATA_DIR") or (Path(__file__).resolve().parents[2] / "data"))
SUBGRAPH_DIR = PROJECT_ROOT / "subgraph"

INPUT_JSONL = SUBGRAPH_DIR / "path_route_result" / "mtq_test_seed_path.jsonl"
OUTPUT_JSONL = SUBGRAPH_DIR / "subset" / "mtq_sampled_200.jsonl"

SAMPLE_SIZE = 200
RANDOM_SEED = 42   # MUST match sub_test.py for reproducibility


def main():
    force = bool(os.environ.get("FORCE"))

    if OUTPUT_JSONL.exists() and not force:
        n = sum(1 for _ in OUTPUT_JSONL.open(encoding="utf-8"))
        if n == SAMPLE_SIZE:
            print(f"[SKIP] {OUTPUT_JSONL} already has {n} lines. "
                  f"Set FORCE=1 to regenerate.")
            return 0
        else:
            print(f"[WARN] {OUTPUT_JSONL} exists with {n} lines (expected {SAMPLE_SIZE}). "
                  f"Regenerating...")

    if not INPUT_JSONL.exists():
        print(f"[ERROR] Input file not found: {INPUT_JSONL}", file=sys.stderr)
        return 1

    random.seed(RANDOM_SEED)

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    if total == 0:
        print(f"[ERROR] Empty input: {INPUT_JSONL}", file=sys.stderr)
        return 1

    if total <= SAMPLE_SIZE:
        sampled = lines
        print(f"[INFO] Input has only {total} lines, keeping all.")
    else:
        sampled = random.sample(lines, SAMPLE_SIZE)
        print(f"[INFO] Sampled {SAMPLE_SIZE} of {total} lines (seed={RANDOM_SEED}).")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        f.writelines(sampled)

    # Quick sanity report
    graphs = {}
    for line in sampled:
        d = json.loads(line)
        g = d.get("graph", "UNKNOWN")
        graphs[g] = graphs.get(g, 0) + 1
    print(f"[DONE] Wrote {len(sampled)} lines to {OUTPUT_JSONL}")
    print(f"[INFO] Graph distribution: {graphs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
