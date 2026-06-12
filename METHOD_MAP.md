# MetaCypher: Method-to-Code Map

This document maps MetaCypher method concepts and evaluation metrics to their implementing code in `metacypher/`. References use stable concept names rather than draft-specific section numbers.

Every `file:symbol` below was verified against the code with grep/read.

## Core data object

| Method concept | Code location | Note |
| --- | --- | --- |
| Property graph + schema `S=(T_V,T_E,A)`; query signature `σ(Q)` (def:signature) | `metacypher/query_analyze.py:build_prompt`, `compact_schema_for_prompt`, `call_llm`, `supplement_entity_mappings` | First LLM call extracts `σ(Q)` (anchors, attr predicates, target roles); fields are schema-checked downstream. |
| Cardinality catalog `C` / `CatalogEntry(p)` (subsec:prelim, fig:offline) | `metacypher/catalog.py:CatalogEntry`, `CatalogResult` | Dataclass with fields `key/card/deg/sel/role_description/anchor_fields`; `CatalogResult.median_card` = `μ_P`. |

## Offline: catalog construction (Algorithm 1, alg:profiling, sec:offline)

| Method concept | Code location | Note |
| --- | --- | --- |
| `EnumMetaPaths(S,L)` — schema-bounded meta-path enumeration up to length L (def:metapath) | `metacypher/catalog.py:enum_meta_paths` | DFS over schema multigraph following out/in relation triples; canonical typed-path key. |
| `BuildAnchorIndex` — anchor/attribute grounding fields | `metacypher/catalog.py:_build_anchor_index`, `_compatible_anchor_fields` | Per-label property + numeric-property index for hosting comparison predicates. |
| `JointPathSupport` — joint cardinality `card(p)` | `metacypher/catalog.py:build_catalog` (`_compile_count_query`) | Whole-path COUNT, not per-edge popularity; exact COUNT via injected `count_fn`. |
| Endpoint-retention `deg(p)` / `\|π(p)\|` | `metacypher/catalog.py:_compile_src_distinct_query`, `_compile_dst_distinct_query`, `_compile_src_degree_query`, `_compile_dst_degree_query` | Distinct endpoint bindings + avg endpoint degree. |
| `SelectivitySketch` — `sel(p)` (subsubsec:synopsis) | `metacypher/catalog.py:_compute_selectivity` | `card / (src_pop * dst_pop)`, clamped to (0,1]. |
| `RoleDescription` — `φ_desc(p)` (subsubsec:descriptions) | `metacypher/catalog.py:_role_description` | Cached natural-language role string; used for semantic matching online. |
| Zero-support paths "demoted, not discarded" (fig:offline) | `metacypher/catalog.py:build_catalog` | All entries retained including `card==0`; `CatalogEntry.is_data_present`. |
| `μ_P` = median support (used by `φ_sparse`) | `metacypher/catalog.py:build_catalog` → `CatalogResult.median_card` | Median over non-zero `card`; fallback 1.0 to avoid div-by-zero. |
| Catalog persistence / live Neo4j count_fn | `metacypher/catalog.py:save_catalog`, `load_catalog`, `make_neo4j_count_fn` | — |

## Online: ValidateRank structure selection (Algorithm 2/3, alg:search/alg:validaterank, sec:online)

| Method concept | Code location | Note |
| --- | --- | --- |
| `SelectStructure` outer beam loop (Algorithm 2) | `metacypher/validate_rank.py:select_structure` | Per-layer PreRank → ValidateRank → top-B select → stopping check. Convenience driver (candidate pool injected). |
| Catalog-driven candidate expansion / adaptive expansion (frontier ext., role connection, predicate hosting) | `metacypher/beam_search.py:BalancedCandidateGenerator`, `SemanticGuidedBeamSearch` | Production expansion path; question-guided. This is the ablation target "adaptive expansion". |
| `PreRank` — cheap catalog-only pre-ranking, keep top-P (alg:search line 7) | `metacypher/validate_rank.py:pre_rank` | Zero-support kept as penalty, not filtered. |
| `ValidateRank` — bounded, memoized batched COUNT probes (Algorithm 3) | `metacypher/validate_rank.py:validate_rank`, `compile_count_probe`, `_canonical_probe_key` | Probe budget `P`; cache memoizes by canonical signature; skeleton-only COUNT (no RETURN/ORDER BY). |
| Score `J(s)` (Eq.2, eq:score) | `metacypher/validate_rank.py:_compute_J` | `φ_desc + φ_cov + λ·φ_struct − λ_miss·φ_miss − λ_sparse·φ_sparse`. |
| `φ_sparse(s) = exp(−n̂/μ_P)` sparsity penalty (eq:score term 5) | `metacypher/validate_rank.py:phi_sparse` | Soft: down-weights, never deletes. Also re-implemented locally in `path_scorer.py:_phi_sparse_path`. |
| `φ_desc`, `φ_cov`, `φ_struct` terms | `metacypher/validate_rank.py:_phi_desc`, `_phi_cov`, `_phi_struct` | Role-description fit, anchor/predicate coverage, binding continuity. |
| Intent-completeness / stopping condition (subsec:termination) | `metacypher/validate_rank.py:is_eligible` (used in `select_structure`) | Eligible iff covers `Ω(Q)` + anchors + predicates AND `n̂>0`; returns missing-intent list. |
| Default coefficients λ (eq:score footnote) | `metacypher/validate_rank.py:LAMBDA_STRUCT/LAMBDA_MISS/LAMBDA_SPARSE` | Fixed on validation split; overridable per call. |

## Meta-graph → Cypher generation and repair

| Method concept | Code location | Note |
| --- | --- | --- |
| Surface realization `Y = F(Q, M*)` (subsec:generation) | `metacypher/generation.py:CypherGenerator._build_generation_prompt`, `generate_cypher` | Serializes selected meta-graph as structural context; LLM realizes projection/aggregation/syntax. |
| Post-generation repair (method's *baseline* paradigm, not a MetaCypher mechanism) | `metacypher/correction.py:CypherRepairer`, `enforce_distinct_by_node`, `enforce_no_abs_for_difference_questions` | LLM repair + deterministic enforcement. Pipeline plumbing only — MetaCypher's thesis is validate-before-generate, the opposite of post-hoc repair. |
| End-to-end pipeline `Q → M* → Y` (Eq.1, eq:task) | `metacypher/skill.py:text_to_cypher` | Chains analysis → retrieval → generation → repair → (optional) execution. |
| Online structure-retrieval stage aggregator | `metacypher/subgraph_retrieval.py` | Orchestrator + re-exports of `beam_search`, `path_scorer`, `validate_rank`. |

## Experiment-supporting code

| Evaluation artifact | Code location | Note |
| --- | --- | --- |
| `tab:mechanism` — gold meta-path recall@B, probe precision, empty-result EX | `metacypher/diagnostics.py:gold_metapath_recall_at_b`, `probe_precision`, `empty_result_accuracy`, `mechanism_diagnostics` | Deterministic, script-derived from gold/predicted provenance subgraphs. |
| `fig:error_attr` — structural-hallucination attribution (phantom node/relation/attribute, invalid connectivity) | `metacypher/diagnostics.py:classify_hallucinations`, `hallucination_attribution`, `parse_cypher`, `SchemaView` | Regex Cypher extractor validated against schema; no external parser. |
| Efficiency table (per-query latency, LLM-call count, COUNT-probe count; tab:efficiency, fig:context) | `metacypher/instrumentation.py:track_query`, `instrumented_count_fn`, `record_llm_call`, `record_prompt`, `estimate_tokens` | Opt-in thread-local counters; zero overhead outside `track_query()`. |
| Ablation "w/o adaptive expansion" | env `METACYPHER_ABLATE_ADAPTIVE_EXPANSION` → `config.py:ABLATE_ADAPTIVE_EXPANSION`, consumed in `beam_search.py:BalancedCandidateGenerator._create_candidate`, `_get_repeat_count` | Neutralizes question-derived prior/coverage scores → schema-valid but question-blind expansion. |
| Ablation "w/o execution pruning" | **No flag**: run `validate_rank(..., probe_budget=0)` | Disabling COUNT probes leaves only catalog-synopsis `n̂`; documented in `config.py` ablation block. |
| Ablation "w/o structural context" | env `METACYPHER_ABLATE_STRUCTURAL_CONTEXT` → `config.py:ABLATE_STRUCTURAL_CONTEXT`, consumed in `generation.py:_build_generation_prompt` | Drops the selected meta-graph serialization from the final prompt (schema + question kept). |

## Honest scope

- **Catalog / ValidateRank wiring is opt-in, not the default code path.** The
  production online stage (`subgraph_retrieval.py` → `beam_search.py` →
  `path_scorer.py`) runs semantic + prior-score beam search and works with **no
  catalog**. The catalog-driven `phi_sparse`/selectivity terms only activate when a
  `CatalogResult` is passed via `PathScorer(catalog=...)`; with the default
  `catalog=None`, `PathScorer.score_paths` skips those terms entirely
  (`path_scorer.py:84-89`). The bounded-COUNT-probe `PreRank`/`ValidateRank`
  pipeline (`validate_rank.py`) is a self-contained, fully-tested module
  (`test_validate_rank.py`, `test_catalog.py`) but is **not invoked** by the
  default `text_to_cypher` path in `skill.py`. So the catalog + ValidateRank
  algorithms are implemented and unit-tested, yet the end-to-end skill does not
  exercise them unless explicitly wired.

- **`select_structure` is a testability wrapper, not the production loop.** It
  takes the per-layer candidate pool as an injected callable rather than expanding
  from live beam states; the production expansion lives in
  `beam_search.SemanticGuidedBeamSearch`. The two are not yet joined into one path.

- **`correction.py` is the baseline paradigm, not a MetaCypher contribution.**
  MetaCypher contrasts MetaCypher (validate-before-generate) with
  post-hoc repair / execution-feedback baselines. `correction.py` exists so the
  pipeline emits runnable Cypher; it substantiates no mechanism claim.

- **Attribute-constraint blindness (C2) is intentionally unimplemented.** The method
  states (subsec:boundary) that attribute predicates are placed but not validated
  offline; numeric predicates are applied only by the online probe. The code matches
  this: `catalog.py:_build_anchor_index` records which attributes *can* host a
  predicate but never evaluates one.

- **Diagnostics/instrumentation compute metrics; they do not reproduce the method's
  reported numbers.** `diagnostics.py` and `instrumentation.py` are the measurement
  harness (with `test_diagnostics.py`, `test_instrumentation.py`); the actual
  benchmark tables are not regenerated from a committed run in this repo.
