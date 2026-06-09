import json
import os
from pathlib import Path

# Env-overridable data dir (see ../config.py); defaults to <repo>/data/subgraph.
_SUBGRAPH = Path(os.environ.get("METACYPHER_SUBGRAPH_DIR") or
                 (Path(os.environ.get("PROJECT_ROOT") or os.environ.get("METACYPHER_DATA_DIR") or
                       (Path(__file__).resolve().parents[2] / "data")) / "subgraph"))
INPUT_PATH = str(_SUBGRAPH / "final" / "mtq_correction.jsonl")
OUTPUT_PATH = str(_SUBGRAPH / "final" / "mtq_correction.filtered.jsonl")

total = 0
kept = 0
dropped = 0

with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue

        total += 1
        obj = json.loads(line)

        sub = obj.get("subgraph_retrieval_result")
        if not isinstance(sub, dict) or "top_triples" not in sub:
            dropped += 1
            # 这里可以顺便打印一下方便你确认
            print(f"Dropping qid={obj.get('qid')} because no top_triples")
            continue

        # 正常样本写出去
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

print("Total:", total)
print("Kept:", kept)
print("Dropped:", dropped)
print("Output written to:", OUTPUT_PATH)