import json
import os
from pathlib import Path

# Env-overridable data dir (see ../config.py); defaults to <repo>/data/subgraph.
_SUBGRAPH = Path(os.environ.get("METACYPHER_SUBGRAPH_DIR") or
                 (Path(os.environ.get("PROJECT_ROOT") or os.environ.get("METACYPHER_DATA_DIR") or
                       (Path(__file__).resolve().parents[2] / "data")) / "subgraph"))
INPUT_PATH = str(_SUBGRAPH / "final" / "mtq_cases_with_predictions.jsonl")
OUTPUT_PATH = str(_SUBGRAPH / "final" / "mtq_correction.jsonl")

bad_count = 0          # 统计坏行数量
next_qid = 1           # 给 qid=None 的样本分配的起始编号

with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
    for idx, line in enumerate(f_in, start=1):
        line_stripped = line.strip()
        if not line_stripped:
            # 空行直接跳过，不写入输出
            continue

        try:
            obj = json.loads(line_stripped)
        except json.JSONDecodeError as e:
            # 1) 打印错误信息和附近片段
            print(f"Bad JSON at line {idx}")
            print(f"Error: {e}")

            pos = e.pos
            snippet = line_stripped[max(0, pos - 80):pos + 80]
            print("Around error:", snippet)

            # 2) 不再写入这一行，相当于“删除”
            bad_count += 1
            continue

        # 合法 JSON，这里处理 qid 为空的情况
        # 仅在 qid 为 None 的时候赋值；已有 qid 的保留
        if obj.get("qid") is None:
            obj["qid"] = next_qid
            next_qid += 1

        # 写回到新的 jsonl 文件
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Done.")
print("Bad lines skipped:", bad_count)
print("Last assigned qid:", next_qid - 1)
print("Output written to:", OUTPUT_PATH)