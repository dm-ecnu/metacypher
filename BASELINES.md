# Baselines & compared systems

MetaCypher is compared against four categories of alternatives to its
pre-generation, catalog-grounded structure selection, plus prompt-based
references. Below, each compared system is listed with its one-line description,
**the specific configuration MetaCypher compares against** (as adapted in the
paper), and its citation key.

BibTeX for every key is in [CITATIONS.bib](CITATIONS.bib), copied verbatim from
the paper's `.bib` sources. All metrics in the paper: `Exec.` (runs without
error on Neo4j), `EX` (execution accuracy, primary), `PSJS` (provenance-subgraph
Jaccard similarity, `feng2025cypherbench`).

---

## Compared systems

| Category | System | What it is (one line) | Configuration MetaCypher compares against | Citation key |
|---|---|---|---|---|
| Prompt-based | Vanilla | Zero-shot Cypher from full schema + question. | Full schema serialized into the prompt, no retrieval. | _(no cite — prompt baseline)_ |
| Prompt-based | Few-shot | Vanilla plus top-_k_ retrieved NL-Cypher demonstrations. | Demonstrations retrieved by embedding similarity of the NL question. | _(no cite — prompt baseline)_ |
| Fine-tuned | SFT (Text2Cypher) | Supervised fine-tuning of open-weight models on paired NL-Cypher data. | Original Text2Cypher training configuration, applied to the train splits of both benchmarks (open-weight backbones only; "--" for frontier models). | `ozsoy2025text2cypher` |
| RAG-based | SchemaFilter | Prunes the schema by embedding similarity before generation. | **Schema pruning** via embedding-similarity filtering, then generation. | `ozsoy2025enhancing` |
| RAG-based | MAC (Mac-SQL) | Multi-agent collaborative Text-to-SQL framework, adapted to Cypher. | **3 agents** — a Selector (schema decomposition), a Decomposer (chain-of-thought generation), and a Refiner (corrects erroneous queries via execution feedback). Adapted from SQL to Cypher. | `wang2025mac` |
| RAG-based | MAGIC | Self-correction-guideline generation for in-context Text-to-SQL, adapted to Cypher. | **7 specialized agents** covering query generation, execution, named-entity extraction, and semantic evaluation; a Query Evaluator grades each candidate as Accept / Incorrect / Error-or-Empty and returns structured feedback, iterating until accepted or the retry budget is exhausted. | `askari2025magic` |
| Graph-native | R³-NL2GQL | Model-coordination + KG-alignment approach for NL2GQL. | Coordinates **small and large models** across ranking, rewriting, and generation stages. | `zhou-etal-2024-r3` |
| Graph-native | ChattyKG (Chatty-KG) | Multi-agent conversational QA system over knowledge graphs. | Maintains **session-level chat-history context** and applies a post-generation error-correction module that refines failed queries via execution feedback. | `chattykg2026` |
| Graph-native | UniQGen | Constraint-guided LLM-agent graph-query generation. | Applies a constraint-based **Chase & Backchase** algorithm extended with LLM agents to iteratively refine Cypher clauses without schema-matching fine-tuning (closest per-query-counting comparison). | `wang2026graph-query-generation-with-co` |
| Graph-native | FINER-SQL | Small LMs boosted for Text-to-SQL via fine-grained execution feedback, adapted to graph queries. | Trains small LMs via **group relative policy optimization (GRPO)** with dense, fine-grained execution feedback as the reward signal. | `others2026finer-sql-boosting-slms-for-te` |

**Adaptation note.** MAC and MAGIC were originally designed for SQL query repair;
the paper adapts the same post-generation correction paradigm to Cypher by
replacing the schema serialization and execution-feedback interfaces, **without**
exposing either method to MetaCypher's pre-generation catalog. All graph-native
methods decide query structure during or after generation; MetaCypher instead
commits to an instance-validated meta-graph **before** the generator is invoked.

---

## Models

### Embedding model

| Field | Value |
|---|---|
| **Name** | BAAI/bge-m3 (BGE M3-Embedding), with `BAAI/bge-reranker-v2-m3` cross-encoder reranking |
| **Role** | Embeds meta-path descriptions for the offline catalog; online dense retrieval uses the same encoder. |
| **Citation key** | `chen2024bge` |
| **Source** | <https://huggingface.co/BAAI/bge-m3> |

### LLM backbones

All methods are evaluated with four LLM backbones; system studies default to
Qwen2.5-Coder-14B-Instruct. Open-weight inference runs with vLLM on NVIDIA A100
GPUs; queries execute on Neo4j.

| Backbone | Table abbreviation | Type |
|---|---|---|
| Qwen2.5-Coder-7B-Instruct | Q7B | open-weight |
| Qwen2.5-Coder-14B-Instruct | Q14B | open-weight (default for system studies) |
| DeepSeek-v4-flash | DS4 | frontier (no fine-tuning, "--" in SFT row) |
| GPT-5.3-codex | GPT-5 | frontier (no fine-tuning, "--" in SFT row) |

> The backbones are named in `sections/experiment.tex` but carry **no `\cite{}`
> key** there, so no BibTeX entry exists in the paper sources for them; they are
> therefore not in [CITATIONS.bib](CITATIONS.bib).
