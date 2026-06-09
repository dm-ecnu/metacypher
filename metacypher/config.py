"""Central runtime configuration for the MetaCypher artifact.

Every machine-specific path and service endpoint is read from an environment
variable with a sensible, reproducible default, so the code runs on any host
without editing source. Copy ``.env.example`` to ``.env`` (or export the
variables in your shell) to point the pipeline at your local data, embedding
model, vLLM endpoint, and Neo4j instance.

Layout assumed under ``PROJECT_ROOT`` (override individual dirs if yours
differs)::

    PROJECT_ROOT/
      dataset/MindtheQuery/      # benchmark questions (test.json, ...)
      schema/                    # one <graph>.json per graph
      schema/sandbox_schemas/    # MindtheQuery sandbox schemas
      schema/template/schema_with_template/  # template-annotated schemas
      subgraph/                  # intermediate retrieval artifacts
      faiss_search/index/        # per-schema FAISS attribute indexes
      outputs/                   # run outputs (gitignored)
"""

import os
from pathlib import Path


def _path_env(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val) if val else default


def _str_env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --------------------------------------------------------------------------
# Data root
# --------------------------------------------------------------------------
# METACYPHER_DATA_DIR (preferred) or the legacy PROJECT_ROOT env var; falls
# back to a ``data/`` directory next to this repository.
_REPO_DATA_DEFAULT = Path(__file__).resolve().parent.parent / "data"
PROJECT_ROOT = Path(
    os.environ.get("METACYPHER_DATA_DIR")
    or os.environ.get("PROJECT_ROOT")
    or _REPO_DATA_DEFAULT
)

# --------------------------------------------------------------------------
# Derived data directories (each independently overridable)
# --------------------------------------------------------------------------
DATASET_DIR = _path_env("METACYPHER_DATASET_DIR", PROJECT_ROOT / "dataset" / "MindtheQuery")
SCHEMA_DIR = _path_env("METACYPHER_SCHEMA_DIR", PROJECT_ROOT / "schema")
SANDBOX_SCHEMA_DIR = _path_env("METACYPHER_SANDBOX_SCHEMA_DIR", SCHEMA_DIR / "sandbox_schemas")
TEMPLATE_DIR = _path_env("METACYPHER_TEMPLATE_DIR", SCHEMA_DIR / "template" / "schema_with_template")
SUBGRAPH_DIR = _path_env("METACYPHER_SUBGRAPH_DIR", PROJECT_ROOT / "subgraph")
OUTPUT_DIR = _path_env("METACYPHER_OUTPUT_DIR", PROJECT_ROOT / "outputs")

# --------------------------------------------------------------------------
# Embedding model + FAISS attribute index
# --------------------------------------------------------------------------
# Default is the Hugging Face id, which downloads on first use; override with a
# local path for offline/air-gapped runs.
EMBED_MODEL_PATH = _str_env("METACYPHER_EMBED_MODEL", "BAAI/bge-m3")
FAISS_INDEX_DIR = _path_env("METACYPHER_FAISS_INDEX_DIR", PROJECT_ROOT / "faiss_search" / "index")

# --------------------------------------------------------------------------
# LLM endpoint (OpenAI-compatible, e.g. a local vLLM server)
# --------------------------------------------------------------------------
VLLM_BASE_URL = _str_env("METACYPHER_VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = _str_env("METACYPHER_VLLM_MODEL", "local_llm")
VLLM_API_KEY = _str_env("METACYPHER_VLLM_API_KEY", "EMPTY")

# --------------------------------------------------------------------------
# Neo4j (CypherBench / MindtheQuery sandbox graphs)
# --------------------------------------------------------------------------
# Host and credentials are shared across the per-graph port mapping defined in
# graph_connector.py / neo4j_client.py. "cypherbench" is the public default
# password shipped by the CypherBench benchmark; override via NEO4J_PASSWORD.
NEO4J_HOST = _str_env("NEO4J_HOST", "localhost")
NEO4J_USER = _str_env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _str_env("NEO4J_PASSWORD", "cypherbench")
