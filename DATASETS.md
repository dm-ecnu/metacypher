# Datasets

MetaCypher is evaluated on **two public Text-to-Cypher benchmarks** with complex
graph connectivity, heterogeneous schemas, and varied question types. The paper
uses the official splits, tunes only search hyperparameters on development data,
and reports test results by executing the generated queries on Neo4j.

BibTeX for the citation keys below is in [CITATIONS.bib](CITATIONS.bib), copied
verbatim from the paper's `.bib` sources.

---

## CypherBench

| Field | Value |
|---|---|
| **Official name** | CypherBench |
| **Citation key** | `feng2025cypherbench` |
| **What it is** | 11 Wikidata-derived property graphs with independently curated schemas, paired with NL questions whose Cypher queries combine structural patterns with `RETURN` templates. Released by Megagon Labs. |
| **Splits used** | All **11 graphs**. Schemas span 4–8 entity types, 4–12 relation types, and 8–27 properties per graph; instance scales range from 1.6K to 3.7M nodes and 1.5K to 7.5M edges. The paper reports test-split results and tunes search hyperparameters on development data. |
| **Dataset (HuggingFace)** | <https://huggingface.co/datasets/megagonlabs/cypherbench> |
| **Code / graph-loading** | <https://github.com/megagonlabs/cypherbench> |
| **License** | CC BY 4.0 (per the dataset card on the Megagon Labs HuggingFace repository above). Verify on the dataset card before redistribution. |

**How to obtain.** Clone the HuggingFace dataset and load each of the 11 graphs
into its own Neo4j instance per the CypherBench instructions, then point
`METACYPHER_DATA_DIR` at the directory holding `schema/`. The repo's
`docker-compose.yml` maps Neo4j bolt ports `15060`–`15070` (one per graph,
matching `metacypher/neo4j_client.py`) using the `NEO4J_AUTH=neo4j/cypherbench`
default. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full runbook.

The paper's PSJS (provenance-subgraph Jaccard similarity) metric is also defined
in the CypherBench work (`feng2025cypherbench`).

---

## Mind the Query (MTQ)

| Field | Value |
|---|---|
| **Official name** | Mind the Query (MTQ) |
| **Citation key** | `chauhan-etal-2025-mind` |
| **What it is** | A Text2Cypher benchmark of **11 real-world Neo4j production databases** across industrial domains, with independently defined schemas and questions organized into five semantic tiers (including a cross-entity tier that is especially sensitive to structure selection). |
| **Splits used** | All **11 production databases**. Schemas span 4–18 node labels, 3–20 relation types, and 12–52 properties per graph; instance scales range from 260 to 281K nodes and 611 to 523K edges. The paper evaluates on the official test split. |
| **Source / venue** | Published at EMNLP 2025 (Industry track), DOI `10.18653/v1/2025.emnlp-industry.133`. ACL Anthology landing page: <https://aclanthology.org/2025.emnlp-industry.133/> |
| **License** | Not stated in the MetaCypher paper. Confirm the license on the official benchmark release (ACL Anthology page / authors' repository) before redistribution. |

**How to obtain.** Use the benchmark release that accompanies the EMNLP 2025
paper (DOI above). In this artifact, the MTQ sandbox-graph evaluation schemas and
the test-split question file are expected under `dataset/MindtheQuery/` and
`schema/sandbox_schemas/` respectively (see the **Data** section of
[README.md](README.md)).

---

## Notes

- Both benchmarks are loaded into **Neo4j** for execution-based evaluation
  (`Exec.`, `EX`, and `PSJS` metrics).
- The offline cardinality catalog is built once per schema using the
  `BAAI/bge-m3` embedding model (`chen2024bge`); see [BASELINES.md](BASELINES.md)
  for the model entry.
- URLs marked as canonical (HuggingFace `megagonlabs/cypherbench`,
  `github.com/megagonlabs/cypherbench`, the ACL Anthology page) are the
  authoritative sources for these releases. License fields not stated in the
  paper are flagged accordingly — verify on the official source before relying
  on them.
