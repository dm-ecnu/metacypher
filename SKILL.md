# Skill: metacypher

**Translate a natural-language question to a Cypher query over a configured property-graph database.**

---

## When to use

Use this skill when a user asks a question that should be answered by querying a Neo4j
property-graph database and you want to generate (and optionally run) the corresponding
Cypher query automatically.

Examples:

- "Which paintings were created after 1880 and are displayed in a museum in France?"
- "Find all NBA players who have played for more than three teams."
- "Which senators voted for bill X and also sponsored bill Y?"

Do NOT use this skill when:

- The target graph does not yet have a prepared schema catalog (see Prerequisites).
- You need to write or modify the graph data — MetaCypher is read-only.
- The question maps to a relational database (SQL) rather than a property graph.

---

## Prerequisites

1. **Schema catalog.**  MetaCypher is catalog-grounded: it requires a prepared schema
   file for each graph you want to query.

   - CypherBench graphs (art, biology, company, fictional_character, flight_accident,
     geography, movie, nba, politics, soccer, terrorist_attack): one `<graph>.json`
     under `METACYPHER_DATA_DIR/schema/`.
   - MindtheQuery sandbox graphs (bloom50, covid, er, gdsc, healthcare, legis_graph,
     osm, pole, twitter_trolls, wwc): one `<graph>.json` under
     `METACYPHER_DATA_DIR/schema/sandbox_schemas/`.
   - Template-annotated schemas (used by the retrieval stage) go under
     `METACYPHER_DATA_DIR/schema/template/schema_with_template/`.

   **Pointing this skill at an arbitrary new Neo4j instance without building the catalog
   first will fail at Stage 2 (retrieval).**  Building the catalog is a one-time,
   dataset-specific step described in the benchmark setup artifact's README.

2. **Running Neo4j instances.**  Each graph runs as a separate Neo4j instance on a
   dedicated Bolt port (15060–15070 for CypherBench; 15091–15100 for sandbox graphs).
   Start them with `make neo4j-up` (Docker Compose) or run them manually.

3. **OpenAI-compatible LLM endpoint.**  Any model served via vLLM (or another
   OpenAI-compatible server) works.  Set `METACYPHER_VLLM_BASE_URL` and
   `METACYPHER_VLLM_MODEL` to point at it.

---

## Environment configuration

All variables are read from the environment (or from a `.env` file loaded before
invocation).  Copy `.env.example` as a starting point.

| Variable                        | Default                          | Description                                              |
|---------------------------------|----------------------------------|----------------------------------------------------------|
| `METACYPHER_DATA_DIR`           | `./data/`                        | Root for schemas, datasets, FAISS indexes, outputs       |
| `METACYPHER_VLLM_BASE_URL`      | `http://localhost:8000/v1`       | OpenAI-compatible LLM endpoint                           |
| `METACYPHER_VLLM_MODEL`         | `local_llm`                      | Model name forwarded to the endpoint                     |
| `METACYPHER_VLLM_API_KEY`       | `EMPTY`                          | API key (use `EMPTY` for local vLLM)                     |
| `NEO4J_HOST`                    | `localhost`                      | Neo4j host (shared across all graph instances)           |
| `NEO4J_USER`                    | `neo4j`                          | Neo4j username                                           |
| `NEO4J_PASSWORD`                | `cypherbench`                    | Neo4j password                                           |
| `METACYPHER_SANDBOX_SCHEMA_DIR` | `$METACYPHER_DATA_DIR/schema/sandbox_schemas/` | Schema dir for sandbox graphs   |
| `METACYPHER_SCHEMA_DIR`         | `$METACYPHER_DATA_DIR/schema/`   | Schema dir for CypherBench graphs                        |
| `METACYPHER_EMBED_MODEL`        | `BAAI/bge-m3`                    | Embedding model (only needed if beam search is enabled)  |

---

## Usage

Run from **inside** the `metacypher/` directory (the modules use flat imports):

```python
import sys, os
sys.path.insert(0, "/path/to/metacypher-artifact/metacypher")

from skill import text_to_cypher

# Generate Cypher without executing it
result = text_to_cypher(
    "Which paintings were created after 1880?",
    graph="art",
)
print(result["cypher"])
# MATCH (p:Painting) WHERE p.creation_year > 1880 RETURN p.name

# Generate and execute
result = text_to_cypher(
    "Which paintings were created after 1880?",
    graph="art",
    execute=True,
    top_k=5,
)
print(result["rows"])        # list of record dicts from Neo4j
print(result["error"])       # None if all stages succeeded
```

### Return value

`text_to_cypher` always returns a `dict` with the following keys:

| Key          | Type              | Description                                                  |
|--------------|-------------------|--------------------------------------------------------------|
| `question`   | `str`             | The original question                                        |
| `graph`      | `str`             | The graph name                                               |
| `analysis`   | `dict` or `None`  | Stage 1 output: entity/schema analysis                       |
| `cypher`     | `str` or `None`   | Final Cypher query (after generation + optional correction)  |
| `rows`       | `list` or `None`  | Neo4j result rows if `execute=True`, else `None`             |
| `error`      | `str` or `None`   | Human-readable error description if any stage failed         |

### Pipeline stages

| Stage | Module              | Description                                        |
|-------|---------------------|----------------------------------------------------|
| 1     | `query_analyze.py`  | LLM-based entity extraction and schema mapping     |
| 2     | `triple_retrieval.py` + `graph_connector.py` | Subgraph/triple retrieval from Neo4j |
| 3     | `generation.py`     | LLM-based Cypher generation from evidence          |
| 4     | `correction.py`     | Schema-guided LLM correction (best-effort)         |
| opt.  | `neo4j_client.py`   | Optional execution of the final query              |

Stages 2–4 (and optional execution) require live services.  If a stage fails,
the skill returns a partial result with an `error` field rather than raising.

### Honest limitations

- **Catalog dependency**: Stage 2 retrieval will fail if the schema catalog
  and template files are not present for the requested graph.  The skill cannot
  auto-discover or build the catalog.
- **Beam search disabled**: The skill disables the BGE-M3 embedding-based beam
  search by default to avoid requiring `sentence-transformers`/`faiss`/`torch`.
  Pattern-only retrieval is used.  For best accuracy (matching the benchmark setup's
  reported numbers) run the full batch pipeline with beam search enabled.
- **Single LLM endpoint**: All three LLM calls (analysis, generation, correction)
  share the same `METACYPHER_VLLM_BASE_URL` / `METACYPHER_VLLM_MODEL` endpoint.
- **Graph coverage**: Only graphs with Neo4j instances at the documented ports
  are supported out of the box.  New graphs require both a schema catalog and a
  running Neo4j instance at a matching port.
