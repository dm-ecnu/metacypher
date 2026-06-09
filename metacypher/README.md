# MetaCypher Implementation

This folder contains the MetaCypher method implementation. It is not a baseline
folder.

## Paper-to-Code Map

- Offline catalog preparation and schema templates:
  `pattern_generation.py`, `subgraph_retrieval.py`
- Question analysis and candidate intent extraction:
  `query_analyze.py`, `search_entity.py`
- Candidate subgraph/path retrieval:
  `path_search.py`, `subgraph_retrieval.py`, `subgraph_triple.py`
- Compact evidence construction:
  `triple_retrieval.py`, `all_subgraph_set.py`
- ValidateRank-style filtering and correction signals:
  `related_schema/`, `correction.py`
- Cypher generation:
  `generation.py`, `llm_client.py`
- Execution feedback:
  `neo4j_client.py`, `graph_connector.py`
- Optional literal/entity retrieval:
  `hidden_search_attr.py`

## Typical Call Chain

```text
query_analyze
  -> subgraph_retrieval / path_search
  -> triple_retrieval
  -> all_subgraph_set
  -> generation
  -> correction
```

`all_subgraph_set.py` is the closest current batch driver for producing method
evidence records. `generation.py` has the clearest command-line interface for
the generation stage. Several upstream modules still use local constants from
the original experiment machine; convert those constants to a config file before
making the artifact public.

## Core Modules

- `query_analyze.py`: question analysis, entity extraction, schema mapping, and
  pattern hypothesis construction.
- `subgraph_retrieval.py`: schema-aware subgraph/path candidate generation,
  scoring, beam expansion, and evidence retrieval.
- `triple_retrieval.py`: retrieval wrapper that converts paths into compact
  graph triples.
- `all_subgraph_set.py`: batch driver for applying MetaCypher retrieval over
  benchmark cases.
- `generation.py`: Cypher generation from retrieved evidence.
- `correction.py`: post-generation repair using related schema and execution
  feedback.
- `graph_connector.py`, `neo4j_client.py`: Neo4j execution layer.
- `llm_client.py`: OpenAI-compatible LLM client wrapper.
- `hidden_search_attr.py`: optional FAISS-based attribute-value search.

## Auxiliary Modules

- `related_schema/`: enrich, clean, and filter related-schema records.
- `ablation/`: MetaCypher ablation helpers, not external baseline code.

## Excluded Material

External and adapted baseline implementations were moved to
`../../archive/code-baselines/`. Run outputs and intermediate JSONL artifacts
were moved to `../../results/metacypher-runs/`.
