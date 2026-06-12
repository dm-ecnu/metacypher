# Running MetaCypher experiments

This is the runbook for producing **real** retrieval/EX numbers with the
cardinality-catalog + ValidateRank pipeline. The goal is to generate measured
results — do not hand-edit numbers.

## 0. Prerequisites

- **Neo4j** with the CypherBench graphs loaded (one instance per graph). Get the
  data from <https://huggingface.co/datasets/megagonlabs/cypherbench> and load
  per their instructions. Bolt ports follow `metacypher/neo4j_client.py`
  (CypherBench graphs on `15060`–`15070`).
- An **OpenAI-compatible LLM endpoint** — a local vLLM server, or the hosted
  ECNU endpoint (verified working):
  `METACYPHER_VLLM_BASE_URL=https://chat.ecnu.edu.cn/open/api/v1`,
  `METACYPHER_VLLM_MODEL=ecnu-plus`, `METACYPHER_VLLM_API_KEY=<your key>`.
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

### Measure latency / LLM calls / probe counts (efficiency table)

The efficiency numbers must be measured, not estimated. Wrap each query with
`instrumentation.track_query()` — LLM calls are timed automatically inside
`SimpleLLMClient`, the four `skill.text_to_cypher` stages report their wall
time, and COUNT probes are counted by wrapping the `count_fn`:

```python
from instrumentation import track_query, instrumented_count_fn
from skill import text_to_cypher

with track_query() as stats:
    result = text_to_cypher(question, "geography", execute=True)
print(stats.as_dict())
# {"total_seconds": ..., "stage_seconds": {"analysis":..., "retrieval":...,
#  "generation":..., "correction":..., "execution":...},
#  "llm_calls": ..., "llm_seconds": ..., "probe_count": ..., "probe_seconds": ...}

# For ValidateRank probes:
ranked = retriever.validate_rank_candidates(candidate_sigs, instrumented_count_fn(count_fn))
```

Aggregate `as_dict()` rows per query into the per-dataset means/medians the
efficiency table reports (`python3 test_instrumentation.py` covers the module).

## 4. Evaluate

Compute EX / PSJS from the corrected Cypher against the CypherBench gold answers
(use the CypherBench evaluator). **These measured numbers are what should appear
in the evaluation** — they replace any previously hand-set values.

## 5. Ablations

### Beam-width sweep (fig:param_analysis)

`metacypher/ablation/` has the beam-width sweep machinery
(`run_ablation.py`, `verify_results.py`). Run `ablation/preflight.py` first to
check Neo4j/LLM/data, then the sweep. Report the raw measured curve (no offsets).

### Component ablation (tab:ablation_overall)

Three rows, each toggled without code edits:

```bash
# w/o adaptive expansion — question-blind, schema-valid candidate generation
METACYPHER_ABLATE_ADAPTIVE_EXPANSION=true python all_subgraph_set.py ...

# w/o structural context — drop the meta-graph from the generation prompt
METACYPHER_ABLATE_STRUCTURAL_CONTEXT=true python generation.py ...

# w/o execution pruning — no COUNT-probe filter: call validate_rank(..., probe_budget=0)
```

### Joint catalog versus per-query counting (the defining design choice)

`metacypher/per_query_counting.py` isolates offline amortization itself:
`PerQueryCountingCatalog` is a drop-in for `CatalogResult` that keeps the
scorer (phi_sparse + selectivity) and the COUNT primitive identical, but
measures joint support **at lookup time, under a per-query probe budget, with
nothing reused across queries**. Set the budget to MetaCypher's own online
probe count (20, per the efficiency table) for the equal-budget comparison:

```python
from per_query_counting import PerQueryCountingCatalog
from subgraph_retrieval import SubgraphRetriever, RetrievalConfig

pqc = PerQueryCountingCatalog(schema, count_fn, probe_budget=20)
retriever = SubgraphRetriever(RetrievalConfig(), schema, catalog=pqc)
for q in questions:
    pqc.start_query()                 # discard all measured support
    ...run retrieval/generation for q...
    print(pqc.query_stats())          # probes spent, lookups over budget
```

Run the full pipeline once with this retriever and once with the offline
catalog; the EX/PSJS gap is the per-query-counting row. `python3
test_per_query_counting.py` covers budget enforcement, per-query reset, and
card/sel parity with `build_catalog` (16 tests).

**Measured (2026-06-11, CypherBench 7 test graphs, Qwen2.5-Coder-14B, full
test set n=2,348 paired):** accuracy parity — EX 0.3624 (catalog) vs 0.3620
(perquery), PSJS 0.6366 vs 0.6385. The difference is cost: per-query counting reaches
183–316 s mean / 650–822 s p90 per question on the million-edge graphs with
the 20-probe budget exhausted (median 19/20), while the catalog arm stays at
a 12–27 s median on every graph; one-time builds ranged 8 s–1.9 h per schema.
Full provenance: the benchmark setup repo's `experiment-result/rq4-perquery-counting.md`.

Independent replication (2026-06-11, same 7 graphs, **ECNU `ecnu-plus`**,
n=100/graph = 700/config, full pipeline incl. correction): accuracy parity —
EX 0.517 (catalog) vs 0.521 (perquery), PSJS 0.729 vs 0.728 (gap within noise);
per-query online cost a mean of 16.7 joint-support `COUNT` probes/question
(max 56 on relation-dense graphs) vs the catalog's 0. Same conclusion: parity,
cost is the difference. (Absolute EX lower than the Qwen run above — model
differs.)

**End-to-end driver (this repo).** `run_per_query_vs_catalog.py` runs the whole
question→Cypher pipeline under either arm and records probe stats:
`--config {catalog,per_query}` (per-query takes `--probe-budget`),
`--n-per-graph`, `--narrow` (pattern-only retrieval vs the default beam config),
`--no-correction`, `--retrieval-only` (record probe cost without
generation/correction). `build_all_catalogs.py` materializes `catalog_<graph>.json`
per graph; `eval_ex_psjs.py` scores outputs with the CypherBench evaluator
(EX + PSJS, needs live Neo4j). Set `METACYPHER_CATALOG_PROBE_CAP` (e.g. `10000000`)
to bound joint-support `COUNT` probes so hub self-join patterns (path-instance
count quadratic in a hub's fan-out, e.g. `(:Lake)->(:Country)<-(:Lake)`)
terminate in ~1 s instead of >19 min; default `0` = unbounded (offline tests
stay byte-exact).

### Mechanism diagnostics (tab:mechanism, fig:error_attr)

`metacypher/diagnostics.py` computes the diagnostics table and the
hallucination-attribution figure from gold/predicted Cypher + the schema:

```python
from diagnostics import mechanism_diagnostics, hallucination_attribution

# Per-class structural-hallucination rates + no-hallucination share (fig:error_attr)
attr = hallucination_attribution(predicted_cyphers, schema)   # parses each query vs schema

# Full tab:mechanism block (any input may be omitted → that cell is None)
block = mechanism_diagnostics(
    schema, predicted_cyphers,
    beams=per_q_beam_sigs, golds=per_q_gold_sigs, beam_width=5,   # gold meta-path recall@5
    probe_results=ranked_candidates,                              # probe precision (uses .n_hat)
    ex_records=[EXRecord(gold_empty=..., correct=...) for ...],   # empty-result query EX
)
```

The structural checks (phantom node/relation/attribute, invalid connectivity)
use a stdlib regex parser — no external Cypher parser. `python3
test_diagnostics.py` covers all metrics. The benchmark setup audits these
script-derived labels manually on representative groups.
