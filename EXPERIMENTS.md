# Running MetaCypher experiments

This is the runbook for producing **real** retrieval/EX numbers with the
cardinality-catalog + ValidateRank pipeline. The goal is to generate measured
results — do not hand-edit numbers.

## 0. Prerequisites

- **Neo4j** with the CypherBench graphs loaded (one instance per graph). Get the
  data from <https://huggingface.co/datasets/megagonlabs/cypherbench> and load
  per their instructions. Bolt ports follow `metacypher/neo4j_client.py`
  (CypherBench graphs on `15060`–`15070`).
- An **OpenAI-compatible LLM endpoint** (e.g. a local vLLM server).
- `pip install -r requirements.txt` (needs `networkx`, `neo4j`; the embedding
  stack is optional).

Set the environment (see `.env.example` for the full list):

```bash
export METACYPHER_DATA_DIR=/path/to/cypherbench         # schema/, dataset/, ...
export NEO4J_HOST=localhost NEO4J_USER=neo4j NEO4J_PASSWORD=cypherbench
export METACYPHER_VLLM_BASE_URL=http://localhost:8000/v1
export METACYPHER_VLLM_MODEL=<your-model-id>
```

## 1. Sanity check (no Neo4j/LLM needed)

Confirm the env first — these unit tests exercise the real logic offline:

```bash
cd metacypher
python3 test_catalog.py          # cardinality catalog: 50/50
python3 test_validate_rank.py    # PreRank/ValidateRank/phi_sparse: 115/115
python3 ../examples/smoke_offline.py
```

## 2. Build the cardinality catalog (offline, per graph)

The catalog issues real `COUNT` queries against Neo4j. Build once per graph and
cache to disk:

```python
# build_catalog_for_graph.py  (run from metacypher/)
import json
from graph_connector import create_pool
from catalog import build_catalog, make_neo4j_count_fn, save_catalog
from config import SCHEMA_DIR

GRAPH = "geography"                      # one of the CypherBench graphs
schema = json.load(open(f"{SCHEMA_DIR}/{GRAPH}.json"))
pool   = create_pool("cypherbench")      # live Neo4j
count_fn = make_neo4j_count_fn(GRAPH, pool)

catalog = build_catalog(schema, count_fn, max_len=2)   # real COUNT probes
save_catalog(catalog, f"catalog_{GRAPH}.json")
print(GRAPH, "entries:", len(catalog.entries), "median card:", catalog.median_card)
```

Loop over all graphs to materialize the full catalog set.

## 3. Run retrieval with ValidateRank

Wire the catalog into the retriever; the scorer then uses `phi_sparse` +
selectivity, and `validate_rank` issues bounded, memoized `COUNT` probes:

```python
from catalog import load_catalog, make_neo4j_count_fn
from subgraph_retrieval import SubgraphRetriever, RetrievalConfig
from graph_connector import create_pool

catalog  = load_catalog("catalog_geography.json")
pool     = create_pool("cypherbench")
count_fn = make_neo4j_count_fn("geography", pool)

retriever = SubgraphRetriever(RetrievalConfig(), schema, catalog=catalog)
ranked = retriever.validate_rank_candidates(candidate_sigs, count_fn)   # real probes
```

Or use the existing batch pipeline (`all_subgraph_set.py`) after constructing
the `SubgraphRetriever` with `catalog=`. Then run generation + correction:

```bash
python query_analyze.py                          # 1. analysis (LLM)
python all_subgraph_set.py                       # 2. retrieval (+catalog/ValidateRank)
python generation.py --input <r.jsonl> --output <g.jsonl>   # 3. generation
python correction.py --input_jsonl <g.jsonl> --output_jsonl <c.jsonl> --schema_jsonl <s.jsonl>
```

## 4. Evaluate

Compute EX / PSJS from the corrected Cypher against the CypherBench gold answers
(use the CypherBench evaluator). **These measured numbers are what should appear
in the paper** — they replace any previously hand-set values.

## 5. Ablations

`metacypher/ablation/` has the beam-width sweep machinery
(`run_ablation.py`, `verify_results.py`). Run `ablation/preflight.py` first to
check Neo4j/LLM/data, then the sweep. Report the raw measured curve (no offsets).
