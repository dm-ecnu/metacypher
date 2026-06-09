#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

# ====== 路径来自 config（可用环境变量覆盖，见 ../config.py） ======
_SUBGRAPH = Path(os.environ.get("METACYPHER_SUBGRAPH_DIR") or
                 (Path(os.environ.get("PROJECT_ROOT") or os.environ.get("METACYPHER_DATA_DIR") or
                       (Path(__file__).resolve().parents[2] / "data")) / "subgraph"))
INPUT_JSONL = _SUBGRAPH / "final" / "related_schema" / "cb.jsonl"
OUTPUT_JSONL = _SUBGRAPH / "final" / "related_schema" / "clean_related" / "cb.jsonl"

# ====== 目标匹配条件 ======
TARGET_CONFIDENCE = 0.8
TARGET_REASON = "Lakes are located in countries."


def _is_target_pair(d: Dict[str, Any]) -> bool:
    """判断该 dict 是否包含需要删除的 confidence+reason 组合。"""
    if "confidence" not in d or "reason" not in d:
        return False

    # confidence 可能是 float/int/str，做一下鲁棒转换
    conf = d.get("confidence")
    try:
        conf_val = float(conf)
    except Exception:
        return False

    reason = d.get("reason")
    return abs(conf_val - float(TARGET_CONFIDENCE)) < 1e-12 and reason == TARGET_REASON


def remove_target_fields(obj: Any) -> Tuple[Any, int]:
    """
    递归遍历 JSON 对象：
    - 若遇到 dict 且同时满足 confidence==0.8 && reason==...，则删除这两个 key
    - 返回 (新对象, 删除次数)
    """
    removed = 0

    if isinstance(obj, dict):
        # 先处理当前层
        if _is_target_pair(obj):
            obj = dict(obj)  # 不原地修改，避免副作用
            obj.pop("confidence", None)
            obj.pop("reason", None)
            removed += 1

        # 再递归处理子元素
        new_d = {}
        for k, v in obj.items():
            new_v, r = remove_target_fields(v)
            new_d[k] = new_v
            removed += r
        return new_d, removed

    if isinstance(obj, list):
        new_l = []
        for item in obj:
            new_item, r = remove_target_fields(item)
            new_l.append(new_item)
            removed += r
        return new_l, removed

    # 其它类型原样返回
    return obj, removed


def process_jsonl(in_path: Path, out_path: Path, encoding: str = "utf-8") -> None:
    kept = 0
    invalid_json = 0
    total_removed = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding=encoding) as fin, out_path.open("w", encoding=encoding) as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            cleaned, removed = remove_target_fields(obj)
            total_removed += removed

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[DONE] input={in_path} output={out_path}")
    print(f"  lines_written={kept}")
    print(f"  invalid_json_lines={invalid_json}")
    print(f"  removed_pairs={total_removed}  (each pair = removed confidence+reason once)")


if __name__ == "__main__":
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")

    process_jsonl(INPUT_JSONL, OUTPUT_JSONL)