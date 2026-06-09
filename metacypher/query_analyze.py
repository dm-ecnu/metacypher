import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from llm_client import SimpleLLMClient
from config import (
    DATASET_DIR,
    SANDBOX_SCHEMA_DIR,
    SUBGRAPH_DIR,
    VLLM_BASE_URL,
    VLLM_MODEL,
)

_LLM_CLIENT: Optional[SimpleLLMClient] = None

DATA_DIR = str(DATASET_DIR)
SCHEMA_DIR = str(SANDBOX_SCHEMA_DIR)

# Input files (relative to base dirs above)
TEST_JSON_PATH = os.path.join(DATA_DIR, "test.json")
# SCHEMA_JSON_PATH is no longer used, we load all schemas from SCHEMA_DIR

# Output
OUT_JSONL_PATH = os.path.join(str(SUBGRAPH_DIR / "query_analys_result"), "mtq_test_question_analysis.jsonl")

# Prompt versioning
PROMPT_VERSION = "qa_schema_v2"

# vLLM (OpenAI-compatible) endpoint
# VLLM_BASE_URL / VLLM_MODEL come from config (env-overridable).
# IMPORTANT: VLLM_MODEL must match the model name exposed by your vLLM server (see /v1/models)
# Safer default output length for JSON analysis
VLLM_MAX_TOKENS = 2048
# Lower temperature to reduce formatting drift
VLLM_TEMPERATURE = 0.0


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_schemas(schema_dir: str) -> Dict[str, Any]:
    """Loads all schema.json files from a directory into a dictionary."""
    schemas = {}
    for filename in os.listdir(schema_dir):
        if filename.endswith(".json"):
            schema_name = os.path.splitext(filename)[0]
            file_path = os.path.join(schema_dir, filename)
            try:
                schema_data = load_json(file_path)
                schemas[schema_name] = schema_data
            except Exception as e:
                print(f"Warning: Could not load or parse schema file {file_path}: {e}")
    return schemas


def dump_jsonl_line(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def compact_schema_for_prompt(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Compact the full schema to a prompt-friendly view.

    Key requirement: relations must carry an explicit MATCH-style pattern with direction,
    so the model is less likely to generate Cypher with incorrect edge direction.
    """

    entities: List[Dict[str, Any]] = []
    for e in schema.get("entities", []):
        props = e.get("properties") or {}
        entities.append({
            "label": e.get("label"),
            # keep keys only (prompt-friendly); types are available in the original schema
            "properties": list(props.keys()),
        })

    relations: List[Dict[str, Any]] = []
    for r in schema.get("relations", []):
        rel_type = r.get("label")
        subj = r.get("subj_label")
        obj = r.get("obj_label")

        pattern = r.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            # fallback: default to subj -> obj
            pattern = f"(n0:{subj})-[r0:{rel_type}]->(n1:{obj})"
        else:
            pattern = pattern.strip()

        # Normalize direction detection
        # Examples:
        #  - (n0:Painting)-[r0:displayedAt]->(n1:Museum)  => out
        #  - (n0:A)<-[r0:REL]-(n1:B)                      => in (relative to n0)
        if "]->" in pattern or "-[]>" in pattern or "->" in pattern:
            direction = "out"
        elif "<-[" in pattern or "<-" in pattern:
            direction = "in"
        else:
            direction = "undirected"

        relations.append({
            "type": rel_type,
            "subj_label": subj,
            "obj_label": obj,
            "direction": direction,
            "pattern": pattern,
            "match": f"MATCH {pattern}",
        })

    return {
        "name": schema.get("name"),
        "entities": entities,
        "relations": relations,
    }


# =========================================================
# Prompt (ENGLISH ONLY)
# =========================================================
def build_prompt(nl_question: str, schema_compact: Dict[str, Any]) -> str:
    """
    The model must output STRICT JSON only.
    Encourage high recall: allow extra schema candidates.
    """
    example_question = (
        "Find paintings created after the year 1880. What are their names?"
    )
    example_schema = {
        "name": "art",
        "entities": [
            {"label": "Painting", "properties": ["eid", "creation_year", "name"]},
            {"label": "Museum", "properties": ["name"]}
        ],
        "relations": [
            {"type": "displayedAt", "subj_label": "Painting", "obj_label": "Museum", "pattern": "(n0:Painting)-[r0:displayedAt]->(n1:Museum)"}
        ]
    }

    # Example output (your intended answer), extended into our proposed JSON schema
    example_output = {
        "related_schema": {
            "node_labels": ["Painting"],
            "relation_types": [],
            "node_label_candidates": [
                {"label": "Painting", "confidence": 0.95, "reason": "The question is about paintings."}
            ],
            "relation_candidates": []
        },
        "entities": [
            {
                "mention": "1880",
                "normalized": "1880",
                "value": 1880,
                "value_type": "int",
                "comparison": {
                    "operator": "gt",
                    "reason": "The question asks for paintings created 'after' the year 1880, which implies a 'greater than' comparison."
                },
                "mapped_to": [
                    {
                        "label": "Painting",
                        "property": "creation_year",
                        "confidence": 0.9,
                        "reason": "'1880' is a year and maps to the 'creation_year' property of a Painting."
                    }
                ]
            }
        ],
        "notes": {
            "recall_bias": "high",
            "ambiguities": [],
            "assumptions": []
        }
    }
    prompt = """
You are an information extraction engine.

INPUT:
1) A natural language question about a Neo4j graph database.
2) A database schema in JSON.

TASK:
Analyze the question and output a SINGLE JSON OBJECT that strictly follows the schema below.
This task is NOT query generation.

--------------------------------
STEP 1: Relevant schema selection (HIGH RECALL)

Select ALL node labels and relation types that may be relevant to the question.
High recall is REQUIRED.
It is acceptable to include extra plausible labels or relations.

--------------------------------
STEP 2: Entity value extraction

Extract explicit entity values mentioned in the question.
Examples: names, titles, years, dates, numbers.

For each entity:
- Normalize the value if needed.
- Assign a value_type:
  str | int | float | date | boolean | point | unknown

--------------------------------
STEP 3: Comparison analysis (STRICT RULES)

ONLY do comparison analysis if:
- value_type is int, float, or date
AND
- the question explicitly or implicitly states a comparison

Comparison operator rules:
- "after", "greater than"      → gt
- "before", "less than"        → lt
- "on", "equal to"             → eq
- "at least", "no less than"   → gte
- "at most", "no more than"    → lte

DO NOT include a "comparison" field if no comparison is stated.

--------------------------------
STEP 4: Mapping

Map each extracted entity to (label, property) pairs from the schema.

**CRITICAL for non-string types (int, float, date):**
- You MUST provide AT LEAST 2-3 alternative mappings per non-string entity
- Consider ALL possible label contexts where this value could appear
- Include both direct node properties AND relationship properties
- Example: A date "after 1940" should map to:
  * Person.deathDate (if Person has this property)
  * Person.bornDate (if Person has this property)
  * Event.eventDate (if Event has this property)
  * Any other date-related properties in the schema
- Rank mappings by confidence, but include multiple hypotheses

**For string types:**
- 1-2 most likely mappings are sufficient

This multi-hypothesis approach is essential to prevent single-point failures in downstream query generation.

--------------------------------
OUTPUT RULES (VERY IMPORTANT):

- Output MUST be exactly ONE valid JSON object.
- Do NOT output any text before or after the JSON.
- Do NOT include explanations, reasoning, markdown, or code fences.
- If a field is not applicable, omit it.

--------------------------------
OUTPUT JSON SCHEMA (MUST MATCH EXACTLY):

{{
  "related_schema": {{
    "node_labels": ["..."],
    "relation_types": ["..."],
    "node_label_candidates": [
      {{"label": "...", "confidence": 0.0, "reason": "..."}}
    ],
    "relation_candidates": [
      {{
        "type": "...",
        "subj_label": "...",
        "obj_label": "...",
        "direction": "out|in|undirected",
        "pattern": "...",
        "confidence": 0.0,
        "reason": "..."
      }}
    ]
  }},
  "entities": [
    {{
      "mention": "...",
      "normalized": "...",
      "value": "...",
      "value_type": "str|int|float|date|boolean|point|unknown",
      "comparison": {{
        "operator": "gt|lt|eq|gte|lte",
        "reason": "Why this operator applies based on the question wording."
      }},
      "mapped_to": [
        {{
          "label": "...",
          "property": "...",
          "confidence": 0.0,
          "reason": "..."
        }}
      ]
    }}
  ],
  "notes": {{
    "recall_bias": "high",
    "ambiguities": ["..."],
    "assumptions": ["..."]
  }}
}}

--------------------------------
FINAL CHECK (MANDATORY):

Before producing the answer, verify:
- The output is valid JSON.
- The output contains exactly ONE top-level object.
- There is NO text outside the JSON.

--------------------------------
NOW PROCESS:

Question:
{question}

Schema:
{schema}
""".strip()

    prompt = prompt.format(
        question=nl_question,
        schema=json.dumps(schema_compact, ensure_ascii=False),
    )

    return prompt


def extract_first_json_object(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract the first top-level JSON object substring from a possibly noisy model output.

    Handles common cases:
    - leading reasoning text
    - <think>...</think>
    - markdown code fences ```json ... ```

    Returns (json_str, error).
    """
    if not isinstance(text, str):
        return None, "Model output is not a string."

    s = text.strip()
    if not s:
        return None, "Empty model output."

    # If code fences exist, try to focus on the fenced content that contains braces
    if "```" in s:
        parts = s.split("```")
        candidates = [p.strip() for p in parts if "{" in p and "}" in p]
        if candidates:
            s = candidates[0]
            if s.lower().startswith("json"):
                s = s[4:].lstrip()

    # Drop <think> blocks if present
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()

    start = s.find("{")
    if start == -1:
        return None, "No '{' found in model output."

    depth = 0
    in_str = False
    escape = False
    end = None

    for i in range(start, len(s)):
        ch = s[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        return None, "Unbalanced JSON braces in model output."

    return s[start:end + 1].strip(), None


def safe_parse_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    # First attempt: direct JSON
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return None, "Model output is not a JSON object."
        return obj, None
    except Exception:
        pass

    # Second attempt: extract JSON object from noisy output
    json_str, err = extract_first_json_object(text)
    if err is not None or json_str is None:
        return None, f"JSON parse error: {err}"

    try:
        obj = json.loads(json_str)
        if not isinstance(obj, dict):
            return None, "Extracted JSON is not a JSON object."
        return obj, None
    except Exception as e:
        return None, f"JSON parse error after extraction: {e}"


def call_llm(prompt: str) -> str:
    """Call remote vLLM (OpenAI-compatible) service and return raw text output.

    Reuses a single client instance across calls.
    """
    global _LLM_CLIENT

    if _LLM_CLIENT is None:
        _LLM_CLIENT = SimpleLLMClient(
            provider="vllm",
            model=VLLM_MODEL,
            base_url=VLLM_BASE_URL,
            api_key="",
            temperature=VLLM_TEMPERATURE,
            max_tokens=VLLM_MAX_TOKENS,
            enable_memory=False,
        )

    # Treat the whole constructed prompt as user content
    return _LLM_CLIENT.generate(
        system="You are a precise and reliable assistant for Neo4j graph question analysis. Your output must be a single, valid JSON object and nothing else. Do not include any text, headers, or markdown before or after the JSON object.",
        user_temp=prompt,
    )


def detect_insufficient_non_str_mappings(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect entities with non-string types that have insufficient mappings.
    
    Returns list of entities that need more mapping hypotheses.
    """
    insufficient = []
    entities = analysis.get("entities", [])
    
    for entity in entities:
        value_type = entity.get("value_type", "unknown")
        # Check if it's a non-string type
        if value_type in ["int", "float", "date"]:
            mapped_to = entity.get("mapped_to", [])
            # If fewer than 2 mappings, it's insufficient
            if len(mapped_to) < 2:
                insufficient.append(entity)
    
    return insufficient


def build_supplementary_mapping_prompt(
    nl_question: str,
    schema_compact: Dict[str, Any],
    entity: Dict[str, Any]
) -> str:
    """Build a focused prompt to generate additional mapping hypotheses for a specific entity."""
    
    prompt = f"""
You are a schema mapping expert for Neo4j graph databases.

TASK: Generate ADDITIONAL mapping hypotheses for a specific entity value.

QUESTION:
{nl_question}

ENTITY TO MAP:
- Text: "{entity.get('mention', '')}"
- Normalized Value: {entity.get('value', '')}
- Type: {entity.get('value_type', 'unknown')}

EXISTING MAPPINGS (already identified):
{json.dumps(entity.get('mapped_to', []), ensure_ascii=False, indent=2)}

FULL SCHEMA:
{json.dumps(schema_compact, ensure_ascii=False, indent=2)}

INSTRUCTIONS:
1. Find AT LEAST 2-3 MORE plausible (label, property) pairs for this entity
2. Look for:
   - Alternative node labels that might have this property
   - Similar property names (e.g., deathDate vs dateOfDeath vs diedOn)
   - Relationship properties (not just node properties)
   - Different semantic contexts (e.g., a date could be creation, modification, event time, etc.)

3. For each mapping, provide:
   - label: The node label
   - property: The property name
   - confidence: 0.0-1.0
   - reason: Why this mapping is plausible
   - access_path: "direct_property" or "relationship_property"

OUTPUT FORMAT (JSON only, no other text):
{{
  "additional_mappings": [
    {{
      "label": "...",
      "property": "...",
      "confidence": 0.0,
      "reason": "...",
      "access_path": "direct_property | relationship_property"
    }}
  ]
}}

IMPORTANT:
- Focus on finding DIFFERENT possibilities, not repeating existing mappings
- Even lower-confidence hypotheses (0.5-0.7) are valuable
- Return ONLY the JSON object
""".strip()
    
    return prompt


def build_pattern_hypotheses_prompt(
    nl_question: str,
    schema_compact: Dict[str, Any],
    analysis: Dict[str, Any]
) -> str:
    """Build prompt to generate multiple graph pattern hypotheses for queries without entities."""
    
    # Extract existing schema candidates
    node_labels = analysis.get("related_schema", {}).get("node_labels", [])
    relation_types = analysis.get("related_schema", {}).get("relation_types", [])
    
    prompt = f"""
You are a Neo4j graph pattern expert.

TASK: Generate MULTIPLE possible graph pattern hypotheses for a query that has NO specific entity values.

QUESTION:
{nl_question}

IDENTIFIED SCHEMA ELEMENTS:
Node Labels: {json.dumps(node_labels, ensure_ascii=False)}
Relation Types: {json.dumps(relation_types, ensure_ascii=False)}

FULL SCHEMA:
{json.dumps(schema_compact, ensure_ascii=False, indent=2)}

ANALYSIS:
This query has NO specific entity values (like names, dates, numbers).
It asks about GRAPH PATTERNS and RELATIONSHIPS.

YOUR TASK:
Generate 3-5 different possible Cypher pattern variations that could answer this query.
Consider:
1. Different relationship directions
2. Different ways to express the same semantic relationship
3. Alternative interpretations of the question
4. Self-referential patterns (e.g., Character kills Character)
5. Intermediate nodes that might be implied

PATTERN FORMAT:
Each pattern should be a valid Cypher MATCH clause fragment.
Examples:
- (n:Character)<-[:killedBy]-(m:Character), (n)-[:hasSpouse]-(m)
- (m:Character)-[:killed]->(n:Character), (m)-[:marriedTo]->(n)
- (c:Creator)-[:created]->(ch:Character)-[:killed]->(spouse:Character), (ch)-[:hasSpouse]-(spouse)

OUTPUT FORMAT (JSON only):
{{
  "pattern_hypotheses": [
    {{
      "pattern": "Cypher pattern string",
      "confidence": 0.0-1.0,
      "reason": "Why this pattern matches the question",
      "variables": {{
        "description": "What each variable represents",
        "return_variable": "Which variable(s) to return for the answer"
      }}
    }}
  ]
}}

CRITICAL RULES:
1. Generate AT LEAST 3 different pattern variations
2. Include both simple and complex interpretations
3. Consider different relationship directions (-> vs <-)
4. Consider both direct relationships and paths through intermediates
5. Each pattern should be syntactically valid
6. Rank by confidence (most likely first)
7. Return ONLY the JSON object, no other text

NOW GENERATE THE PATTERNS:
""".strip()
    
    return prompt


def supplement_pattern_hypotheses(
    nl_question: str,
    schema_compact: Dict[str, Any],
    analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate multiple pattern hypotheses for queries without entities.
    
    This is triggered when the query has no entity values but needs pattern exploration.
    """
    entities = analysis.get("entities", [])
    
    # Only generate pattern hypotheses if there are NO entities at all
    if len(entities) > 0:
        return analysis  # Has entities, skip pattern hypothesis generation
    
    print(f"  [Pattern Hypotheses] Query has no entities, generating multiple graph patterns")
    
    try:
        # Build pattern hypothesis prompt
        pattern_prompt = build_pattern_hypotheses_prompt(nl_question, schema_compact, analysis)
        
        # Call LLM
        raw_output = call_llm(pattern_prompt)
        
        # Parse response
        parsed, err = safe_parse_json(raw_output)
        
        if err is None and isinstance(parsed, dict):
            pattern_hypotheses = parsed.get("pattern_hypotheses", [])
            if pattern_hypotheses:
                # Add pattern hypotheses to analysis
                analysis["pattern_hypotheses"] = pattern_hypotheses
                print(f"    + Generated {len(pattern_hypotheses)} pattern hypothesis/hypotheses")
                
                # Print patterns for debugging
                for i, ph in enumerate(pattern_hypotheses, 1):
                    print(f"      #{i}: {ph.get('pattern', 'N/A')[:80]}... (conf: {ph.get('confidence', 0)})")
            else:
                print(f"    ! No pattern hypotheses generated")
        else:
            print(f"    ! Failed to parse pattern hypotheses: {err}")
            
    except Exception as e:
        print(f"    ! Pattern hypothesis generation failed: {e}")
    
    return analysis


def supplement_entity_mappings(
    nl_question: str,
    schema_compact: Dict[str, Any],
    analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Supplement mappings for entities with insufficient hypotheses.
    
    Modifies analysis in-place and returns it.
    """
    insufficient = detect_insufficient_non_str_mappings(analysis)
    
    if not insufficient:
        # No entity mapping supplementation needed
        # But check if we need pattern hypothesis generation (for queries without entities)
        entities = analysis.get("entities", [])
        if len(entities) == 0:
            # Generate pattern hypotheses for entity-less queries
            analysis = supplement_pattern_hypotheses(nl_question, schema_compact, analysis)
        return analysis
    
    print(f"  [Supplement] Found {len(insufficient)} non-string entities with insufficient mappings")
    
    for entity in insufficient:
        print(f"    - Supplementing mappings for: {entity.get('mention', 'unknown')}")
        
        try:
            # Build focused prompt for this entity
            supp_prompt = build_supplementary_mapping_prompt(nl_question, schema_compact, entity)
            
            # Call LLM
            raw_output = call_llm(supp_prompt)
            
            # Parse response
            parsed, err = safe_parse_json(raw_output)
            
            if err is None and isinstance(parsed, dict):
                additional = parsed.get("additional_mappings", [])
                if additional:
                    # Add to existing mappings
                    entity["mapped_to"] = entity.get("mapped_to", []) + additional
                    print(f"      + Added {len(additional)} additional mapping(s)")
                else:
                    print(f"      ! No additional mappings generated")
            else:
                print(f"      ! Failed to parse supplementary output: {err}")
                
        except Exception as e:
            print(f"      ! Supplementary call failed: {e}")
    
    return analysis



# =========================================================
# Manual interactive test mode (single question, no file IO)
# =========================================================
def manual_test() -> None:
    """Manual single-question test.

    This does NOT read the dataset JSON and does NOT write output files.
    It is intended for quickly inspecting the constructed prompt and the model response.
    """
    all_schemas = load_all_schemas(SCHEMA_DIR)
    if not all_schemas:
        print(f"[Error] No schemas found in {SCHEMA_DIR}. Manual test cannot proceed.")
        return

    compact_schemas = {name: compact_schema_for_prompt(schema) for name, schema in all_schemas.items()}
    
    print("\n[Manual Test] Available schemas:", list(compact_schemas.keys()))
    
    selected_schema_name = ""
    schema_compact = None

    while not schema_compact:
        try:
            choice = input(f"Enter schema to use (default: {list(compact_schemas.keys())[0]}): ").strip()
            if not choice:
                selected_schema_name = list(compact_schemas.keys())[0]
            else:
                selected_schema_name = choice
            
            if selected_schema_name in compact_schemas:
                schema_compact = compact_schemas[selected_schema_name]
            else:
                print(f"[Error] Schema '{selected_schema_name}' not found. Please choose from the available list.")
        except (EOFError, KeyboardInterrupt):
            return


    print(f"\nUsing schema: '{selected_schema_name}'")
    print("\nEnter a question (empty line to exit):")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break

        if not q:
            break

        prompt = build_prompt(q, schema_compact)

        print("\n===== PROMPT (EN) =====\n")
        print(prompt)
        print("\n===== MODEL OUTPUT (RAW) =====\n")

        try:
            raw = call_llm(prompt)
        except Exception as e:
            print(f"[Error] LLM call failed: {e}")
            continue

        print(raw)

        parsed, err = safe_parse_json(raw)
        print("\n===== PARSED JSON =====\n")
        if err is not None:
            print(f"[Parse Error] {err}")
        else:
            # Supplement mappings if needed
            assert parsed is not None  # for type checker
            parsed = supplement_entity_mappings(q, schema_compact, parsed)
            print(json.dumps(parsed, ensure_ascii=False, indent=2))

        print("\n[Manual Test] Enter another question (empty line to exit):")


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Main
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def main() -> None:
    # fresh output
    os.makedirs(os.path.dirname(OUT_JSONL_PATH), exist_ok=True)
    if os.path.exists(OUT_JSONL_PATH):
        os.remove(OUT_JSONL_PATH)

    dataset = load_json(TEST_JSON_PATH)
    all_schemas = load_all_schemas(SCHEMA_DIR)

    # Pre-compact all schemas
    compact_schemas = {name: compact_schema_for_prompt(schema) for name, schema in all_schemas.items()}

    total = 0
    processed = 0
    failed = 0

    for item in dataset:
        total += 1

        graph = item.get("graph")
        if not graph or graph not in compact_schemas:
            print(f"Skipping item with qid={item.get('qid')} due to missing or unknown graph: {graph}")
            continue

        processed += 1
        schema_compact = compact_schemas[graph]

        qid = item.get("qid")
        nl_question = item.get("nl_question", "")

        prompt = build_prompt(nl_question, schema_compact)

        record: Dict[str, Any] = {
            "qid": qid,
            "graph": graph,
            "nl_question": nl_question,
            "analysis": None,
            "raw_model_output": None,
            "meta": {
                "schema_name": schema_compact.get("name"),
                "prompt_version": PROMPT_VERSION,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        try:
            raw = call_llm(prompt)
            record["raw_model_output"] = raw

            parsed, err = safe_parse_json(raw)
            if err is not None:
                failed += 1
                record["analysis"] = {
                    "error": err,
                    "fallback": {
                        "related_schema": {"node_labels": [], "relation_types": [], "node_label_candidates": [], "relation_candidates": []},
                        "entities": [],
                        "notes": {"recall_bias": "high", "ambiguities": ["model_output_invalid_json"], "assumptions": []}
                    }
                }
            else:
                # Supplement mappings for non-string entities if needed
                assert parsed is not None  # for type checker
                parsed = supplement_entity_mappings(nl_question, schema_compact, parsed)
                record["analysis"] = parsed

        except Exception as e:
            failed += 1
            record["analysis"] = {
                "error": f"LLM call failed: {e}",
                "fallback": {
                    "related_schema": {"node_labels": [], "relation_types": [], "node_label_candidates": [], "relation_candidates": []},
                    "entities": [],
                    "notes": {"recall_bias": "high", "ambiguities": ["llm_call_failed"], "assumptions": []}
                }
            }

        dump_jsonl_line(OUT_JSONL_PATH, record)

    print(f"[Done] total_items_in_file={total}, processed={processed}, failed={failed}")
    print(f"[Output] {OUT_JSONL_PATH}")


if __name__ == "__main__":
    # Set MODE to "manual" for interactive testing, or keep "batch" for dataset processing.
    MODE = os.environ.get("QA_MODE", "batch").strip().lower()

    if MODE == "manual":
        manual_test()
    else:
        main()