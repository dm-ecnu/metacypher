import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from llm_client import SimpleLLMClient
from config import VLLM_BASE_URL

_CODE_FENCE_RE = re.compile(r"```(?:cypher|cql|sql)?\s*([\s\S]*?)```", re.IGNORECASE)


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON on line {line_no} ({path}): {e}")


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_question(s: str) -> str:
    """Normalize NL question to improve exact matching stability."""
    s = (s or "").strip()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def extract_cypher(text: str) -> str:
    m = _CODE_FENCE_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return (text or "").strip()


def infer_return_spec(nl_question: str) -> Dict[str, Any]:
    q = (nl_question or "").strip().lower()
    if any(k in q for k in ["how many", "number of", "count of"]):
        return {"mode": "count"}
    if "names of" in q or "name of" in q:
        return {"mode": "property", "property": "name"}
    if any(k in q for k in ["list", "which", "identify", "show", "find"]):
        return {"mode": "node"}
    return {"mode": "node"}


def enforce_distinct_by_node(cypher: str) -> str:
    if not cypher:
        return cypher

    m = re.search(r"(?im)^\s*RETURN\s+DISTINCT\s+(.+?)\s*$", cypher)
    if not m:
        return cypher

    expr = m.group(1).strip()

    m2 = re.match(r"^([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)$", expr)
    if m2:
        var, prop = m2.group(1), m2.group(2)
        cypher_wo = re.sub(r"(?im)^\s*RETURN\s+DISTINCT\s+.+\s*$", "", cypher).rstrip()
        if re.search(rf"(?im)^\s*WITH\s+DISTINCT\s+{re.escape(var)}\b", cypher_wo):
            return cypher_wo + f"\nRETURN {var}.{prop}"
        return cypher_wo + f"\nWITH DISTINCT {var}\nRETURN {var}.{prop}"

    m3 = re.match(r"^([A-Za-z_]\w*)$", expr)
    if m3:
        var = m3.group(1)
        cypher_wo = re.sub(r"(?im)^\s*RETURN\s+DISTINCT\s+.+\s*$", "", cypher).rstrip()
        if re.search(rf"(?im)^\s*WITH\s+DISTINCT\s+{re.escape(var)}\b", cypher_wo):
            return cypher_wo + f"\nRETURN {var}"
        return cypher_wo + f"\nWITH DISTINCT {var}\nRETURN {var}"

    return cypher


def enforce_no_abs_for_difference_questions(nl_question: str, cypher: str) -> str:
    q = (nl_question or "").lower()
    if "difference" not in q:
        return cypher
    if not cypher or "abs(" not in cypher.lower():
        return cypher
    return re.sub(r"(?i)\babs\s*\(\s*([^)]+?)\s*\)", r"(\1)", cypher)


# ----------------------------
# related_schema injection
# ----------------------------

def schema_to_prompt_text(related_schema: Dict[str, Any]) -> str:
    """
    Turn your related_schema dict into a compact, LLM-friendly text.
    Emphasize:
      - node labels + properties
      - relation labels + (subj_label -> obj_label) + direction
    """
    if not related_schema:
        return "SCHEMA_MISSING"

    entities = related_schema.get("entities", []) or []
    relations = related_schema.get("relations", []) or []

    lines = []
    lines.append("NODE_LABELS_AND_PROPERTIES:")
    for e in entities:
        label = e.get("label")
        props = e.get("properties", {}) or {}
        # keep it compact: prop:type
        prop_str = ", ".join([f"{k}:{v}" for k, v in props.items()])
        lines.append(f"- {label} {{{prop_str}}}")

    lines.append("\nRELATIONSHIPS:")
    for r in relations:
        rlabel = r.get("label")
        subj = r.get("subj_label")
        obj = r.get("obj_label")
        direction = r.get("direction") or ""  # expected: "out" etc.
        pattern = r.get("pattern") or ""
        lines.append(f"- {rlabel}: {subj} -> {obj}, direction={direction}, pattern={pattern}")

    return "\n".join(lines).strip()


def build_schema_index(schema_jsonl_path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Index schema lines by (qid, normalized_nl_question) -> related_schema dict.
    """
    idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, row in read_jsonl(schema_jsonl_path):
        qid = str(row.get("qid", "")).strip()
        nq = normalize_question(row.get("nl_question", "") or row.get("question", ""))
        rs = row.get("related_schema")
        if not qid or not nq or not isinstance(rs, dict):
            continue
        idx[(qid, nq)] = rs
    return idx


# ----------------------------
# Prompting
# ----------------------------

SYSTEM_PROMPT = """You are a Cypher query repair assistant.
You will be given:
- A natural language question
- A predicted Cypher query (may be incorrect)
- The graph database schema (labels, relationship types, directions, and properties)

Your task:
Return ONLY a corrected Cypher query that is semantically consistent with the question and compliant with the schema.

Hard rules:
- Ensure property keys used in WHERE / MATCH exist on the correct node/relationship types per schema.
- Ensure relationship directions are consistent with schema patterns.
- If you use DISTINCT, deduplicate nodes first (WITH DISTINCT <nodeVar>) and only then project properties.
- If the question requests "names", confirm that returning <nodeVar>.name (or schema’s identifier property) is appropriate.
1.	Anchor-First Property Rule
If a question asks for a property of the anchored entity, return that property directly from the anchor node.
Do not introduce extra hops, neighbor nodes, or intermediate relations unless the question explicitly requires them.
Property location must follow the schema, not intuition.
	2.	Return-Type Strictness
Return exactly the type requested by the question.
If the question asks for a value, name, group, code, or year, return a scalar property, not a node.
Only return nodes when the question explicitly asks for entities.
Always add an explicit alias using AS.
	3.	Attribute Ownership Consistency
Do not move properties across entities.
A property belonging to one node must not be retrieved from a related node, even if the relationship exists.
Property ownership is fixed by schema.
	4.	Prefer Direct Property Filtering
If a condition can be expressed as a node property filter, use it directly.
Do not rewrite property filters as relationship traversals unless the question explicitly refers to the related entity.
	5.	Canonical Identifier Rule
Use schema-defined primary keys (such as id or primaryid) for anchoring entities.
Do not substitute name for id or id for name unless the question explicitly mentions that field.
Even identical values are not interchangeable.
	6.	Relationship Predicate Exactness
Relationship types must exactly match those defined in the schema.
Do not replace, paraphrase, or infer relationship names based on natural language similarity.
Relationship predicates are symbolic constraints.
	7.	Deterministic Value Matching
When the question specifies an exact value (such as a specific year), use equality on the target node.
Do not infer the value through time intervals or relationship properties unless the question explicitly asks for coverage or duration.
	8.	Minimal Query Structure
Avoid adding structural clauses unless required by the question.
Do not introduce WITH, DISTINCT, OPTIONAL MATCH, or LIMIT by default.
Use them only when the question explicitly requests deduplication, optional existence, sampling, or aggregation.
	9.	Output Column Name Alignment
Output column names must match the expected semantic target exactly.
Always use AS to specify the column name.
Do not default to common attributes like name when a more specific property is required.
    10. for date-based equality conditions, never directly compare a property to date('YYYY-MM-DD') unless the property is explicitly typed as Date; instead, normalize the property side using date(property) = date('YYYY-MM-DD');
    11.  when the question asks for a specific attribute value, always return the corresponding property with an explicit alias rather than returning the whole node;
    12. when a query is anchored on a specific entity (e.g., by id), first assume the requested attribute belongs to that anchor entity and do not traverse relationships unless the schema explicitly indicates that the attribute is owned by a neighboring node;


Output format:
- Output ONLY the final Cypher query.
- No explanations, no markdown, no extra text.
"""


def build_user_prompt(
    qid: str,
    graph: str,
    nl_question: str,
    pred_cypher: str,
    related_schema_text: str,
) -> str:
    ret_spec = infer_return_spec(nl_question)
    return f"""### Task
Repair the predicted Cypher to match the question and schema, following the hard rules.

### qid
{qid}

### graph
{graph}

### Natural language question
{nl_question}

### Predicted Cypher (to repair)
{pred_cypher}

### Schema (authoritative)
{related_schema_text}

### Return specification (must follow)
{json.dumps(ret_spec, ensure_ascii=False)}


"""


@dataclass
class RepairConfig:
    model: str
    base_url: str
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    enable_memory: bool = False


class CypherRepairer:
    def __init__(self, cfg: RepairConfig):
        self.client = SimpleLLMClient(
            provider="vllm",
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            enable_memory=cfg.enable_memory,
        )

    def repair_one(
        self,
        qid: str,
        graph: str,
        nl_question: str,
        pred_cypher: str,
        related_schema_text: str,
    ) -> str:
        user_prompt = build_user_prompt(
            qid=qid,
            graph=graph,
            nl_question=nl_question,
            pred_cypher=pred_cypher,
            related_schema_text=related_schema_text,
        )

        raw = self.client.generate(
            system_temp=SYSTEM_PROMPT,
            user_temp=user_prompt,
            temperature=0.0,
            max_tokens=1024,
            reset_memory=True,
        )
        repaired = extract_cypher(raw)

        repaired = enforce_distinct_by_node(repaired)
        repaired = enforce_no_abs_for_difference_questions(nl_question, repaired)
        return repaired.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)

    # schema jsonl that contains related_schema per example
    ap.add_argument("--schema_jsonl", required=True, help="Schema jsonl with (qid, nl_question, related_schema) per line")

    # LLM config
    ap.add_argument('--base_url', type=str, default=VLLM_BASE_URL, help='vLLM base URL (default from METACYPHER_VLLM_BASE_URL)')
    ap.add_argument("--model", default="local_llm")
    ap.add_argument("--api_key", default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=2048)

    # behavior when schema missing
    ap.add_argument("--drop_if_missing_schema", action="store_true", default=True,
                    help="If schema missing, drop the sample (default: True).")

    ap.add_argument("--keep_llm_raw", action="store_true")

    args = ap.parse_args()

    schema_index = build_schema_index(args.schema_jsonl)

    cfg = RepairConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_memory=False,
    )
    repairer = CypherRepairer(cfg)

    out_rows = []
    dropped = 0

    for line_no, row in read_jsonl(args.input_jsonl):
        qid = str(row.get("qid", f"line_{line_no}"))
        graph = str(row.get("graph", ""))
        nl_question = row.get("nl_question") or row.get("question") or ""
        pred_cypher = row.get("pred_cypher") or row.get("generated_cypher") or ""

        key = (str(qid).strip(), normalize_question(nl_question))
        related_schema = schema_index.get(key)

        if not related_schema:
            dropped += 1
            if not args.drop_if_missing_schema:
                # fallback: keep but without schema (not recommended)
                related_schema_text = "SCHEMA_NOT_FOUND_FOR_THIS_EXAMPLE"
            else:
                # drop row entirely
                continue
        else:
            related_schema_text = schema_to_prompt_text(related_schema)

        try:
            if args.keep_llm_raw:
                user_prompt = build_user_prompt(
                    qid=qid,
                    graph=graph,
                    nl_question=nl_question,
                    pred_cypher=pred_cypher,
                    related_schema_text=related_schema_text,
                )
                raw = repairer.client.generate(
                    system_temp=SYSTEM_PROMPT,
                    user_temp=user_prompt,
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                    reset_memory=True,
                )
                repaired = extract_cypher(raw)
                repaired = enforce_distinct_by_node(repaired)
                repaired = enforce_no_abs_for_difference_questions(nl_question, repaired)

                row["llm_raw"] = raw
                row["repaired_cypher"] = repaired.strip()
            else:
                row["repaired_cypher"] = repairer.repair_one(
                    qid=qid,
                    graph=graph,
                    nl_question=nl_question,
                    pred_cypher=pred_cypher,
                    related_schema_text=related_schema_text,
                )

            # keep optional meta for later debugging
            row["repair_meta"] = {"return_spec": infer_return_spec(nl_question)}

        except Exception as e:
            row["repaired_cypher"] = pred_cypher
            row["repair_error"] = str(e)

        out_rows.append(row)

    write_jsonl(args.output_jsonl, out_rows)

    # print a minimal summary for logs
    print(f"[DONE] input={args.input_jsonl} output={args.output_jsonl} kept={len(out_rows)} dropped_missing_schema={dropped}")


if __name__ == "__main__":
    main()