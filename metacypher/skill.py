"""
metacypher.skill — high-level entry point for the MetaCypher pipeline.

Implements the end-to-end task pipeline of the paper (Eq.1, eq:task):
Q -> structure retrieval (M*) -> Cypher generation (Y), chaining query analysis,
online structure retrieval (sec:online), generation (subsec:generation), and
post-hoc repair.

Exposes a single callable:

    text_to_cypher(question, graph, *, execute=False, top_k=5) -> dict

This chains the real pipeline stages in memory:

    Stage 1  — query analysis   (query_analyze)
    Stage 2  — subgraph/triple retrieval  (triple_retrieval via graph_connector pool)
    Stage 3  — Cypher generation  (generation.CypherGenerator)
    Stage 4  — schema-based correction   (correction.CypherRepairer, best-effort)
    Optional — Neo4j execution   (neo4j_client.Neo4jConnector)

Each stage is wrapped in a try/except so that a missing service (no LLM, no
Neo4j) yields a clear error string in the returned dict rather than crashing.

Prerequisites (see SKILL.md):
- A prepared schema catalog under METACYPHER_DATA_DIR (config.SANDBOX_SCHEMA_DIR
  for sandbox graphs, config.SCHEMA_DIR / "<graph>.json" for CypherBench graphs).
- An OpenAI-compatible LLM endpoint reachable at METACYPHER_VLLM_BASE_URL.
- Neo4j instances (one per graph) running on the ports documented in
  graph_connector.py / neo4j_client.py, reachable at NEO4J_HOST.

Heavy or service-dependent imports (neo4j, sentence-transformers, torch, faiss)
are deferred to inside functions so that ``import skill`` always succeeds even
when those packages are absent.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

try:  # package-style import (pip install -e .)
    from . import instrumentation as _instr
except ImportError:  # flat import, run from inside the package dir
    import instrumentation as _instr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def text_to_cypher(
    question: str,
    graph: str,
    *,
    execute: bool = False,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Translate a natural-language question to a Cypher query for *graph*.

    Parameters
    ----------
    question:
        The natural-language question to answer (e.g. "Which paintings were
        created after 1880 and are displayed in a museum in France?").
    graph:
        Name of the target property graph, matching one of the schema files
        under ``METACYPHER_DATA_DIR``.  E.g. ``"art"``, ``"movie"``,
        ``"geography"``, ``"bloom50"``.
    execute:
        If ``True``, run the final Cypher against the corresponding Neo4j
        instance and include the result rows in the return dict.
        Requires that Neo4j is running and reachable.
    top_k:
        Number of top retrieved subgraph triples passed to the generator as
        evidence.  Lower values are faster; higher values give the LLM more
        context.  Default 5.

    Returns
    -------
    dict with keys:
        ``question``   — the original question (str)
        ``graph``      — the graph name (str)
        ``analysis``   — parsed query-analysis object from Stage 1 (dict | None)
        ``cypher``     — the generated (and optionally corrected) Cypher (str | None)
        ``rows``       — query result rows if *execute* is True, else None
        ``error``      — human-readable error string if any stage failed, else None

    Notes
    -----
    The pipeline requires a prepared schema catalog for *graph*: a
    ``<graph>.json`` schema file and (for CypherBench graphs) the full schema
    with template annotations.  Pointing this skill at an arbitrary new Neo4j
    instance without building the catalog first will fail at Stage 2.

    If the LLM endpoint is unavailable, Stages 1 and 3 will fail gracefully;
    the returned dict will carry error details but will not raise.

    If Neo4j is unavailable and ``execute=False``, the skill still attempts
    Stages 1–3 (analysis + retrieval + generation) but skips execution.

    Stage 4 (correction) is best-effort: it requires the schema dict, which
    is available from Stage 1's analysis output, and calls the LLM.  If it
    fails for any reason the Stage 3 Cypher is returned as-is.
    """
    result: Dict[str, Any] = {
        "question": question,
        "graph": graph,
        "analysis": None,
        "cypher": None,
        "rows": None,
        "error": None,
    }

    # ------------------------------------------------------------------
    # Stage 1 — Query analysis
    # ------------------------------------------------------------------
    schema_compact: Optional[Dict[str, Any]] = None
    full_schema: Optional[Dict[str, Any]] = None
    full_schema_path: str = ""
    analysis: Optional[Dict[str, Any]] = None

    try:
        with _instr.stage("analysis"):
            analysis, schema_compact, full_schema, full_schema_path = _stage1_analysis(
                question, graph
            )
        result["analysis"] = analysis
    except Exception as exc:
        result["error"] = f"Stage 1 (analysis) failed: {exc}"
        logger.exception("Stage 1 failed")
        return result

    # ------------------------------------------------------------------
    # Stage 2 — Subgraph / triple retrieval
    # ------------------------------------------------------------------
    top_triples: List[Dict[str, Any]] = []
    try:
        with _instr.stage("retrieval"):
            top_triples = _stage2_retrieval(
                question=question,
                graph=graph,
                analysis=analysis,
                full_schema=full_schema,
                full_schema_path=full_schema_path,
                top_k=top_k,
            )
    except Exception as exc:
        # Retrieval failure is non-fatal — we can still try generation with
        # an empty evidence list (quality will be lower).
        logger.warning("Stage 2 (retrieval) failed: %s", exc)
        result["error"] = f"Stage 2 (retrieval) failed: {exc}; proceeding without evidence"

    # ------------------------------------------------------------------
    # Stage 3 — Cypher generation
    # ------------------------------------------------------------------
    cypher: Optional[str] = None
    try:
        with _instr.stage("generation"):
            cypher = _stage3_generation(
                question=question,
                analysis=analysis,
                top_triples=top_triples,
                top_k=top_k,
            )
        result["cypher"] = cypher
    except Exception as exc:
        result["error"] = (result["error"] or "") + f"; Stage 3 (generation) failed: {exc}"
        logger.exception("Stage 3 failed")
        return result

    # ------------------------------------------------------------------
    # Stage 4 — Schema-based correction (best-effort)
    # ------------------------------------------------------------------
    if cypher and schema_compact is not None:
        try:
            with _instr.stage("correction"):
                corrected = _stage4_correction(
                    question=question,
                    graph=graph,
                    cypher=cypher,
                    schema=schema_compact,
                )
            if corrected:
                result["cypher"] = corrected
        except Exception as exc:
            logger.warning("Stage 4 (correction) failed (using Stage 3 output): %s", exc)

    # ------------------------------------------------------------------
    # Optional — Execute against Neo4j
    # ------------------------------------------------------------------
    if execute and result["cypher"]:
        try:
            with _instr.stage("execution"):
                result["rows"] = _execute_cypher(graph=graph, cypher=result["cypher"])
        except Exception as exc:
            result["error"] = (result["error"] or "") + f"; execution failed: {exc}"
            logger.warning("Neo4j execution failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Internal stage helpers
# ---------------------------------------------------------------------------

def _stage1_analysis(
    question: str,
    graph: str,
) -> tuple:
    """Run query analysis and return (analysis, schema_compact, full_schema, full_schema_path).

    Loads schema from SANDBOX_SCHEMA_DIR (sandbox graphs) or SCHEMA_DIR
    (CypherBench graphs), falling back between them.
    """
    # Lazy imports — openai is available; no heavy deps needed here
    from config import SANDBOX_SCHEMA_DIR, SCHEMA_DIR
    from query_analyze import (
        compact_schema_for_prompt,
        build_prompt,
        call_llm,
        safe_parse_json,
        supplement_entity_mappings,
    )

    full_schema: Optional[Dict[str, Any]] = None
    full_schema_path: str = ""

    # Try sandbox schemas first, then top-level schema dir
    schema: Optional[Dict[str, Any]] = None
    for candidate_dir in (SANDBOX_SCHEMA_DIR, SCHEMA_DIR):
        candidate_path = candidate_dir / f"{graph}.json"
        if candidate_path.exists():
            with candidate_path.open("r", encoding="utf-8") as fh:
                schema = json.load(fh)
            full_schema = schema
            full_schema_path = str(candidate_path)
            break

    if schema is None:
        raise FileNotFoundError(
            f"No schema file found for graph='{graph}' in {SANDBOX_SCHEMA_DIR} "
            f"or {SCHEMA_DIR}.  Build the schema catalog first."
        )

    schema_compact = compact_schema_for_prompt(schema)

    prompt = build_prompt(question, schema_compact)
    raw = call_llm(prompt)
    parsed, err = safe_parse_json(raw)

    if err is not None or parsed is None:
        raise ValueError(f"LLM analysis parse error: {err}; raw={raw[:200]!r}")

    analysis = supplement_entity_mappings(question, schema_compact, parsed)
    return analysis, schema_compact, full_schema, full_schema_path


def _stage2_retrieval(
    question: str,
    graph: str,
    analysis: Dict[str, Any],
    full_schema: Optional[Dict[str, Any]],
    full_schema_path: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Run subgraph/triple retrieval.  Returns a list of triple dicts.

    Mirrors the logic in all_subgraph_set.py:
    - Build a RetrievalConfig (beam search disabled to avoid torch/faiss dep)
    - Instantiate EnhancedSubgraphRetriever
    - Call retrieve_with_triples(pool, graph, case)
    - Serialise triples and return top_k by score

    NOTE: This stage requires a running Neo4j instance and that the schema
    catalog (full schema + templates) is in place.  The beam-search expansion
    is intentionally disabled here (use_beam_search=False) because it depends
    on sentence-transformers / FAISS, which may not be installed.  Pattern-only
    retrieval is used instead.
    """
    # Lazy heavy imports — neo4j and networkx must be installed; torch/faiss NOT
    import datetime as _dt

    from graph_connector import create_pool
    from subgraph_retrieval import RetrievalConfig
    from triple_retrieval import EnhancedSubgraphRetriever
    from config import SCHEMA_DIR

    # Build a minimal config that avoids the embedding stack
    cfg = RetrievalConfig(
        top_m_anchors=5,
        top_p_patterns=top_k * 2,
        top_k_paths=top_k * 4,   # over-fetch; trim after
        min_path_score=0.0,
        max_path_length=5,
        pattern_score_threshold=0.0,
        use_beam_search=False,    # no torch/faiss required
        debug_mode=False,
        debug_print_query=False,
        debug_print_results=False,
        debug_print_cypher=False,
    )

    # Determine pool name (sandbox vs cypherbench)
    from graph_connector import SANDBOX_PORTS, CYPHERBENCH_PORTS
    if graph in SANDBOX_PORTS:
        pool_name = "mtq"
    else:
        pool_name = "cypherbench"

    template_base = str(SCHEMA_DIR / "template" / "schema_with_template")

    case = {
        "qid": "skill_query",
        "nl_question": question,
        "graph": graph,
        "analysis": analysis,
    }

    retriever = EnhancedSubgraphRetriever(
        config=cfg,
        schema=full_schema or {},
        full_schema_path=full_schema_path,
        enable_semantics=False,
        template_base_path=template_base,
    )

    def _json_default(value: Any) -> Any:
        if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
            return value.isoformat()
        if isinstance(value, set):
            return list(value)
        return str(value)

    with create_pool(pool_name) as pool:
        retrieval_result = retriever.retrieve_with_triples(
            pool=pool, db_name=graph, case=case
        )

    triples_objs = retrieval_result.get("triples", []) or []

    # Serialise each triple to a plain dict (mirrors triple_to_jsonable in
    # all_subgraph_set.py but inline to avoid importing that batch script)
    triples_json: List[Dict[str, Any]] = []
    for t in triples_objs:
        d: Dict[str, Any] = {}
        try:
            d = t.to_dict()
        except Exception:
            d = {
                "triple_id": getattr(t, "triple_id", None),
                "cypher_query": getattr(t, "cypher_query", None),
                "cypher_pattern": getattr(t, "cypher_pattern", None),
                "semantic_description": getattr(t, "semantic_description", ""),
                "score": getattr(t, "score", 0.0),
                "sample_paths": getattr(t, "sample_paths", []),
            }

        # Remove unserializable fields
        d.pop("subgraph", None)
        d.pop("embedding", None)

        # Ensure required keys for generation stage
        if "sample_paths" not in d:
            d["sample_paths"] = []
        if "semantic_description" not in d or d["semantic_description"] is None:
            d["semantic_description"] = d.get("cypher_pattern", "")
        if "cypher_query" not in d or d["cypher_query"] is None:
            d["cypher_query"] = d.get("cypher_pattern", "")

        triples_json.append(d)

    # Sort by score and take top_k
    triples_json.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return triples_json[:top_k]


def _stage3_generation(
    question: str,
    analysis: Dict[str, Any],
    top_triples: List[Dict[str, Any]],
    top_k: int,
) -> str:
    """Generate Cypher from retrieved evidence.

    Assembles a synthetic item dict (matching generation.CypherGenerator's
    expected input format) and calls generate_cypher().
    """
    from config import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
    from generation import CypherGenerator

    generator = CypherGenerator(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=0.1,
        max_tokens=2048,
        top_k_triples=top_k,
    )

    # Build a synthetic item in the format CypherGenerator.generate_cypher() expects
    synthetic_item: Dict[str, Any] = {
        "nl_question": question,
        "analysis": analysis,
        "subgraph_retrieval_result": {
            "top_triples": top_triples,
        },
    }

    cypher = generator.generate_cypher(synthetic_item)
    return cypher or ""


def _stage4_correction(
    question: str,
    graph: str,
    cypher: str,
    schema: Dict[str, Any],
) -> str:
    """Apply schema-based LLM correction to the generated Cypher.

    Uses correction.CypherRepairer with the full schema dict (entities +
    relations).  Builds the schema text inline using correction.schema_to_prompt_text.

    This stage is best-effort: if the LLM call fails or returns garbage, the
    caller falls back to the Stage 3 output.
    """
    from config import VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
    from correction import CypherRepairer, RepairConfig, schema_to_prompt_text

    cfg = RepairConfig(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        temperature=0.0,
        max_tokens=1024,
        enable_memory=False,
    )
    repairer = CypherRepairer(cfg)

    related_schema_text = schema_to_prompt_text(schema)

    repaired = repairer.repair_one(
        qid="skill_query",
        graph=graph,
        nl_question=question,
        pred_cypher=cypher,
        related_schema_text=related_schema_text,
    )
    return repaired or cypher


def _execute_cypher(graph: str, cypher: str) -> List[Dict[str, Any]]:
    """Execute *cypher* against the Neo4j instance for *graph*.

    Uses neo4j_client.Neo4jConnector (CypherBench graphs) or
    graph_connector.Neo4jConnectionPool (sandbox graphs).  Returns the list
    of record dicts.

    Raises on connection or syntax errors so that the caller can add the
    error to the result dict.
    """
    # Lazy import — requires neo4j package
    from neo4j_client import Neo4jConnector, DATABASE_CONFIG as CB_CONFIGS
    from graph_connector import SANDBOX_PORTS

    if graph in CB_CONFIGS:
        # CypherBench graph — use Neo4jConnector
        with Neo4jConnector(graph) as conn:
            resp = conn.execute(cypher)
            if not resp.get("ok"):
                err_info = resp.get("error") or {}
                raise RuntimeError(
                    f"Cypher execution error ({err_info.get('type')}): "
                    f"{err_info.get('message')}"
                )
            return resp.get("records", [])

    elif graph in SANDBOX_PORTS:
        # Sandbox graph — use graph_connector pool
        from graph_connector import create_pool
        with create_pool("mtq") as pool:
            rows = pool.execute(graph, cypher)
            return rows

    else:
        raise ValueError(
            f"Unknown graph '{graph}'.  Known CypherBench graphs: "
            f"{sorted(CB_CONFIGS.keys())}; "
            f"sandbox graphs: {sorted(SANDBOX_PORTS.keys())}"
        )
