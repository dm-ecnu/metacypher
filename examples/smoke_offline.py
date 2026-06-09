#!/usr/bin/env python3
"""Offline smoke test for MetaCypher — no Neo4j, no vLLM, no network.

Exercises the env-driven config and the pure query-analysis helpers (schema
compaction with edge-direction detection, and tolerant JSON extraction from
noisy LLM output). This is a fast sanity check that a fresh checkout is wired
correctly; the full pipeline additionally needs benchmark data, a Neo4j
instance, and an OpenAI-compatible LLM endpoint (see README).

    cd metacypher && python ../examples/smoke_offline.py
"""

import importlib
import os
import sys
from pathlib import Path

# Run from inside the package dir (flat imports), like the rest of the pipeline.
PKG = Path(__file__).resolve().parents[1] / "metacypher"
sys.path.insert(0, str(PKG))


def check_config() -> None:
    import config
    importlib.reload(config)
    # Default: data root derives from the repo, vLLM is local, no internal IPs.
    assert "localhost" in config.VLLM_BASE_URL, config.VLLM_BASE_URL
    assert config.NEO4J_HOST == "localhost", config.NEO4J_HOST

    # Env overrides take effect on reload.
    os.environ["METACYPHER_DATA_DIR"] = "/tmp/mc-smoke-data"
    os.environ["METACYPHER_VLLM_BASE_URL"] = "http://example.invalid:9/v1"
    importlib.reload(config)
    assert str(config.DATASET_DIR).startswith("/tmp/mc-smoke-data"), config.DATASET_DIR
    assert config.VLLM_BASE_URL == "http://example.invalid:9/v1", config.VLLM_BASE_URL
    for k in ("METACYPHER_DATA_DIR", "METACYPHER_VLLM_BASE_URL"):
        del os.environ[k]
    importlib.reload(config)
    print("ok: config defaults resolve and env overrides apply")


def check_schema_direction() -> None:
    import query_analyze as qa
    schema = {
        "name": "art",
        "entities": [
            {"label": "Painting", "properties": {"name": "str", "creation_year": "int"}},
            {"label": "Museum", "properties": {"name": "str"}},
        ],
        "relations": [
            {"label": "displayedAt", "subj_label": "Painting", "obj_label": "Museum",
             "pattern": "(n0:Painting)-[r0:displayedAt]->(n1:Museum)"},
            {"label": "ownedBy", "subj_label": "Museum", "obj_label": "Painting",
             "pattern": "(n0:Museum)<-[r0:ownedBy]-(n1:Painting)"},
        ],
    }
    compact = qa.compact_schema_for_prompt(schema)
    dirs = {r["type"]: r["direction"] for r in compact["relations"]}
    assert dirs["displayedAt"] == "out", dirs
    assert dirs["ownedBy"] == "in", dirs
    # Entity properties are reduced to key lists for the prompt.
    labels = {e["label"] for e in compact["entities"]}
    assert labels == {"Painting", "Museum"}, labels
    print("ok: compact_schema_for_prompt detects edge direction")


def check_json_extraction() -> None:
    import query_analyze as qa
    noisy = (
        "<think>reasoning we should ignore</think>\n"
        "```json\n{\"related_schema\": {\"node_labels\": [\"Painting\"]}, "
        "\"entities\": []}\n```\ntrailing text"
    )
    obj, err = qa.safe_parse_json(noisy)
    assert err is None, err
    assert obj["related_schema"]["node_labels"] == ["Painting"], obj
    print("ok: safe_parse_json recovers JSON from fenced/think-wrapped output")


def main() -> int:
    check_config()
    check_schema_direction()
    check_json_extraction()
    print("\nMetaCypher offline smoke: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
