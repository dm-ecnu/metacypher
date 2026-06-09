import os
import sys
import json
import argparse
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


cypherbench = [
    "company",
    "fictional_character",
    "flight_accident",
    "geography",
    "movie",
    "nba",
    "politics",
    "soccer",
    "terrorist_attack",
]

mtq = [
    "bloom50",
    "covid",
    "er",
    "gdsc",
    "healthcare",
    "legis_graph",
    "osm",
    "pole",
    "twitter_trolls",
    "wwc",
]

# Paths come from config (env-overridable).
from config import EMBED_MODEL_PATH as MODEL_PATH, FAISS_INDEX_DIR as INDEX_DIR, SUBGRAPH_DIR

# Output directory for processed JSONL
DEFAULT_OUTPUT_DIR = SUBGRAPH_DIR / "new_root_seed_result"

# --------------------------------------------------------------------
# Global caches: model + per-schema FAISS index + metadata
# --------------------------------------------------------------------

_model = None
_index_cache = {}  # schema_name -> (index_format, faiss_index, metadata_list)


def get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading embedding model from {MODEL_PATH} (device={device})", file=sys.stderr)
        _model = SentenceTransformer(MODEL_PATH, device=device)
    return _model


def resolve_index_format(schema_name: str) -> str:
    """
    Determine index format for a given schema.
    - 'cypherbench': cosine similarity (query needs normalization)
    - 'mtq': L2 distance
    If schema_name is not found, default to 'mtq' but warn.
    """
    if schema_name in cypherbench:
        return "cypherbench"
    if schema_name in mtq:
        return "mtq"
    # Fallback: treat as mtq (L2) but log a warning
    print(f"[WARN] Schema '{schema_name}' not found in cypherbench/mtq list. "
          f"Defaulting to 'mtq' (L2 distance).", file=sys.stderr)
    return "mtq"


def get_index_and_metadata(schema_name: str):
    """Lazy-load FAISS index and its metadata for a schema."""
    global _index_cache

    if schema_name in _index_cache:
        return _index_cache[schema_name]

    index_format = resolve_index_format(schema_name)
    index_path = INDEX_DIR / schema_name / "attr.index"
    meta_path = INDEX_DIR / schema_name / "attr.meta.json"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Index or metadata file not found for schema '{schema_name}'. "
            f"Checked: {index_path} and {meta_path}"
        )

    print(f"[INFO] Loading FAISS index for schema '{schema_name}' from {index_path}", file=sys.stderr)
    index = faiss.read_index(str(index_path))

    print(f"[INFO] Loading metadata for schema '{schema_name}' from {meta_path}", file=sys.stderr)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    _index_cache[schema_name] = (index_format, index, metadata)
    return _index_cache[schema_name]


def search_single(schema_name: str, query_text: str, k: int = 5):
    """
    Perform FAISS search for one query_text under a given schema.
    Returns a list of up to k result dicts:
    [
      {
        "rank": int,
        "distance": float,
        "node_id": ...,
        "node_label": ...,
        "attribute_name": ...,
        "value": ...
      },
      ...
    ]
    """
    if not query_text or not query_text.strip():
        return []

    model = get_model()
    index_format, index, metadata = get_index_and_metadata(schema_name)

    # 1. Encode query
    query_text = query_text.strip()
    embedding = model.encode([query_text], convert_to_tensor=True)
    query_vec = embedding.cpu().numpy().astype("float32")  # shape (1, dim)

    # 2. Normalize if using cosine similarity (cypherbench style)
    if index_format == "cypherbench":
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

    # 3. FAISS search
    distances, indices = index.search(query_vec, k)

    results = []
    if indices.size == 0:
        return results

    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        if idx >= len(metadata):
            continue

        meta = metadata[idx]
        results.append(
            {
                "rank": int(rank + 1),
                "distance": float(distances[0][rank]),
                "node_id": meta.get("node_id"),
                "node_label": meta.get("node_label"),
                "attribute_name": meta.get("attribute_name"),
                "value": meta.get("value"),
            }
        )

    return results


# --------------------------------------------------------------------
# JSONL processing
# --------------------------------------------------------------------

def infer_schema_name(record: dict) -> str | None:
    """
    Get schema name for this record.
    Priority:
    1) record["meta"]["schema_name"]
    2) record["graph"]
    """
    meta = record.get("meta") or {}
    name = ''
    if isinstance(meta, dict) and meta.get("schema_name"):
        name = meta.get("schema_name")
    if record.get("graph"):
        name = record["graph"]
    if name == 'healthcare_analytics':
        name = 'healthcare'
    if name == 'wwc2019':
        name = 'wwc'
    if name == 'entity_resolution':
        name = 'er'
    if name == 'contact_tracing':
        name = 'covid'
    if name == 'graph_data_science':
        name = 'gdsc'
    if name == 'openstreetmap':
        name = 'osm'
    return name


def enrich_entities_with_candidates(record: dict, top_k: int = 5) -> dict:
    """
    For a single JSON record:
    - Locate analysis.entities
    - For each entity with value_type == 'str', run FAISS search and attach candidates.
    - Return the updated record (in-place modification).
    """
    schema_name = infer_schema_name(record)
    if not schema_name:
        # No schema info – we just return original record
        return record

    analysis = record.get("analysis")
    if not isinstance(analysis, dict):
        return record

    entities = analysis.get("entities")
    if not isinstance(entities, list):
        return record

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        if ent.get("value_type") != "str":
            continue

        # Choose query text: normalized > value > mention
        q = ent.get("normalized") or ent.get("value") or ent.get("mention")
        if not isinstance(q, str) or not q.strip():
            continue

        try:
            candidates = search_single(schema_name, q, k=top_k)
        except Exception as e:
            # 出错时不要中断整个文件处理，记个标记即可
            print(f"[ERROR] Search failed for schema='{schema_name}', query='{q}': {e}", file=sys.stderr)
            candidates = []

        # 新增字段：faiss_candidates
        ent["faiss_candidates"] = candidates

    # 写回 analysis.entities（虽然是原地修改，这一步可以不写）
    analysis["entities"] = entities
    record["analysis"] = analysis
    return record


def process_jsonl(input_path: Path, output_path: Path, top_k: int = 5):
    """
    Read input JSONL, enrich each line, and write to output JSONL.
    """
    print(f"[INFO] Input JSONL:  {input_path}", file=sys.stderr)
    print(f"[INFO] Output JSONL: {output_path}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_lines = 0
    num_entities_with_search = 0

    with input_path.open("r", encoding="utf-8") as fin, \
            output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            num_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse line {num_lines}: {e}", file=sys.stderr)
                continue

            # 统计一下有多少实体触发检索（仅用于日志）
            analysis = record.get("analysis") or {}
            ents = analysis.get("entities") or []
            before = sum(1 for e in ents
                         if isinstance(e, dict) and e.get("value_type") == "str")

            record = enrich_entities_with_candidates(record, top_k=top_k)

            analysis = record.get("analysis") or {}
            ents = analysis.get("entities") or []
            after = sum(1 for e in ents
                        if isinstance(e, dict) and "faiss_candidates" in e)
            num_entities_with_search += after

            # 单行写回，保持 JSONL 结构
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Processed {num_lines} lines; "
          f"{num_entities_with_search} string entities were enriched with candidates.",
          file=sys.stderr)


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="For each JSONL record, search FAISS index for string entities "
                    "and attach top-k candidate node IDs / metadata."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input JSONL file."
    )
    parser.add_argument(
        "-o", "--output",
        required=False,
        help=("Path to output JSONL file. "
              "If omitted, will be placed under the configured subgraph "
              "output dir (METACYPHER_SUBGRAPH_DIR) "
              "with suffix '_with_entity_candidates.jsonl'.")
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=30,
        help="Number of top FAISS candidates to attach per string entity (default: 5)."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            # 如果给的是相对路径，默认挂在 root_seed_result 目录下
            output_path = DEFAULT_OUTPUT_DIR / output_path
    else:
        # 默认输出文件名：<input_stem>_with_entity_candidates.jsonl
        output_name = f"{input_path.stem}_with_entity_candidates.jsonl"
        output_path = DEFAULT_OUTPUT_DIR / output_name

    process_jsonl(input_path, output_path, top_k=args.top_k)


if __name__ == "__main__":
    main()