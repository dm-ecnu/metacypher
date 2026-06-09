# MetaCypher Code Artifact

This directory is the cleaned implementation snapshot for the MetaCypher
submission. It contains the method code only: external baselines, paper result
files, temporary JSONL outputs, debug fixtures, and cache directories are kept
outside this release tree.

## Framework

![MetaCypher framework](docs/framework.png)

`docs/framework.pdf` is the lightweight vector copy of the framework figure used
by the paper. The raw Draw.io/TikZ/ImageGen sources stay under the paper
`figures/` directory.

MetaCypher implements a catalog-grounded Text-to-Cypher pipeline:

```text
Question
  -> query analysis and entity/schema linking
  -> schema-aware subgraph and path retrieval
  -> ValidateRank-style evidence ranking
  -> evidence-to-Cypher generation
  -> execution-guided correction
  -> final Cypher query
```

## Layout

- `metacypher/`: implementation of the MetaCypher method.
- `metacypher/related_schema/`: utilities for related-schema enrichment,
  filtering, and cleanup.
- `metacypher/ablation/`: component and beam-ablation helpers for the paper.
- `docs/`: compact framework figure for artifact readers.

This repository contains the MetaCypher **method code** only. External baseline
implementations, raw run outputs / JSONL artifacts, and debug fixtures are not
part of this release. Benchmark data (graphs, schemas, FAISS indexes) is
supplied separately via `METACYPHER_DATA_DIR` (see Setup below).

## Implemented Functionality

The code currently covers the following paper components:

- Natural-language query analysis, entity extraction, schema mapping, and pattern
  hypothesis construction.
- Schema-template loading and graph/path candidate construction over Neo4j
  schema catalogs.
- Subgraph and triple retrieval for compact evidence records.
- Attribute-value search for hidden literal/entity matching when an embedding
  index is available.
- Cypher candidate generation with an OpenAI-compatible LLM endpoint.
- Post-generation correction using related-schema evidence and execution
  feedback.
- Ablation scripts for subset construction, preflight checks, run orchestration,
  and result verification.

## Main Pipeline

The method code is organized around this flow:

1. `metacypher/query_analyze.py`
   extracts entities, schema mappings, and pattern hypotheses from a natural
   language question.
2. `metacypher/subgraph_retrieval.py`
   builds and ranks schema-aware subgraph/path candidates.
3. `metacypher/triple_retrieval.py`
   converts selected paths into compact graph evidence.
4. `metacypher/all_subgraph_set.py`
   batch-produces MetaCypher evidence records.
5. `metacypher/generation.py`
   generates Cypher candidates from retrieved evidence.
6. `metacypher/correction.py`
   repairs generated Cypher using related-schema evidence and execution
   feedback.

## Runtime Dependencies

The scripts expect Python 3.10+. Install dependencies with:

```bash
pip install -r requirements.txt
```

The embedding stack (`sentence-transformers`, `torch`, `faiss-cpu`) is only
required for the FAISS attribute-search step; the rest of the pipeline runs
without it.

## Setup

All machine-specific paths and service endpoints are read from environment
variables by `metacypher/config.py` — **no source edits are needed**. Copy the
template and adjust it for your machine:

```bash
cp .env.example .env
# edit .env: data dir, vLLM endpoint, Neo4j host/credentials
set -a && source .env && set +a   # export the variables into your shell
```

Defaults (if you set nothing) assume:

- data under `./data/` (datasets, schemas, templates, FAISS index),
- a local vLLM server at `http://localhost:8000/v1`,
- Neo4j on `localhost` with the public CypherBench default password.

See `.env.example` for the full list of variables (`METACYPHER_DATA_DIR`,
`METACYPHER_VLLM_BASE_URL`, `NEO4J_HOST`/`NEO4J_USER`/`NEO4J_PASSWORD`, …).
No credentials are committed to this repository.

## Running the pipeline

Run the modules from inside the `metacypher/` package directory (they use flat
imports). End to end, the stages are:

```bash
cd metacypher

# 1. Query analysis: entities, schema mapping, pattern hypotheses
python query_analyze.py                 # batch over the dataset
QA_MODE=manual python query_analyze.py  # interactive single-question mode

# 2. Subgraph / path retrieval -> compact graph evidence
python all_subgraph_set.py

# 3. Cypher generation from retrieved evidence
python generation.py --input <retrieved.jsonl> --output <generated.jsonl>

# 4. Execution-guided correction
python correction.py --input_jsonl <generated.jsonl> \
                     --output_jsonl <corrected.jsonl> \
                     --schema_jsonl <schema.jsonl>
```

Optional attribute search (needs the embedding stack + a prebuilt FAISS index):

```bash
python search_entity.py -i <analysis.jsonl> -o <enriched.jsonl> -k 30
```

Beam-size ablation (see `metacypher/ablation/`): run `preflight.py` first to
verify data/Neo4j/LLM connectivity, then `run_ablation.py`.

## Artifact status

This is a cleaned research-code snapshot. All hard-coded local paths, the
internal service endpoint, and the Neo4j password have been moved into the
env-driven `config.py`, so the code runs on any host without source edits.
The benchmark data (CypherBench / MindtheQuery graphs, schemas, FAISS indexes)
is not bundled here and must be provided via `METACYPHER_DATA_DIR`.
