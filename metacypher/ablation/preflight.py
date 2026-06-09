"""
Pre-flight checks for the beam-size ablation experiment.

Run this FIRST after logging into the lab server. It verifies every external
dependency before you commit to a multi-hour sweep.

Usage:
    cd <subgraph project dir>     # the dir that contains all_subgraph_set.py
    python /path/to/preflight.py

Edit the CONFIG block below (or set env var PROJECT_ROOT / METACYPHER_DATA_DIR)
if your data paths differ from the default ``<repo>/data`` layout.
"""

import json
import os
import socket
import sys
import time
from pathlib import Path

# ============================================================
# CONFIG  -- edit if your server paths differ
# ============================================================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or os.environ.get("METACYPHER_DATA_DIR") or (Path(__file__).resolve().parents[2] / "data"))
SUBGRAPH_DIR = PROJECT_ROOT / "subgraph"
SCHEMA_DIR = PROJECT_ROOT / "schema"
PATH_ROUTE_DIR = SUBGRAPH_DIR / "path_route_result"
SUBSET_DIR = SUBGRAPH_DIR / "subset"
ABLATION_OUT_DIR = SUBGRAPH_DIR / "final" / "ablation_beam"

# Whether to do a real BGE model load test (10-30s) -- set False to skip
BGE_LOAD_TEST = True

# ============================================================

CYPHERBENCH_PORTS = list(range(15062, 15069))   # 7 DBs
MTQ_PORTS = list(range(15091, 15101))           # 10 DBs

CYPHERBENCH_GRAPHS = [
    "company", "fictional_character", "flight_accident", "geography",
    "movie", "nba", "politics",
]
MTQ_GRAPHS = [
    "bloom50", "covid", "er", "gdsc", "healthcare",
    "legis_graph", "osm", "pole", "twitter_trolls", "wwc",
]


def color(s, c):
    codes = {"red": 31, "green": 32, "yellow": 33, "cyan": 36, "bold": 1}
    return f"\033[{codes[c]}m{s}\033[0m"


def ok(msg):    print(color("  [OK]   ", "green") + msg)
def warn(msg):  print(color("  [WARN] ", "yellow") + msg)
def fail(msg):  print(color("  [FAIL] ", "red") + msg)


def step(n, title):
    print()
    print(color(f"=== Step {n}: {title} ===", "cyan"))


def check_port(host, port, timeout=2.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def check_paths():
    step(1, "Project paths")
    issues = 0
    for label, p in [
        ("PROJECT_ROOT", PROJECT_ROOT),
        ("SUBGRAPH_DIR", SUBGRAPH_DIR),
        ("SCHEMA_DIR", SCHEMA_DIR),
        ("PATH_ROUTE_DIR", PATH_ROUTE_DIR),
        ("SUBSET_DIR", SUBSET_DIR),
    ]:
        if p.exists():
            ok(f"{label} = {p}")
        else:
            fail(f"{label} = {p}  (NOT FOUND)")
            issues += 1

    cb_subset = SUBSET_DIR / "cypherbench_sampled_200.jsonl"
    mtq_subset = SUBSET_DIR / "mtq_sampled_200.jsonl"
    path_route_mtq = PATH_ROUTE_DIR / "mtq_test_seed_path.jsonl"

    print()
    if cb_subset.exists():
        n = sum(1 for _ in cb_subset.open(encoding="utf-8"))
        ok(f"cypherbench_sampled_200.jsonl exists ({n} lines)")
    else:
        fail(f"cypherbench_sampled_200.jsonl NOT FOUND at {cb_subset}")
        issues += 1

    if mtq_subset.exists():
        n = sum(1 for _ in mtq_subset.open(encoding="utf-8"))
        ok(f"mtq_sampled_200.jsonl exists ({n} lines)")
    else:
        warn(f"mtq_sampled_200.jsonl NOT FOUND -- run make_mtq_subset.py to create it")
        if not path_route_mtq.exists():
            fail(f"  source file also missing: {path_route_mtq}")
            issues += 1

    try:
        ABLATION_OUT_DIR.mkdir(parents=True, exist_ok=True)
        ok(f"ABLATION_OUT_DIR writable: {ABLATION_OUT_DIR}")
    except Exception as e:
        fail(f"Cannot create ABLATION_OUT_DIR: {e}")
        issues += 1

    return issues


def check_schema_files():
    step(2, "Schema files (one .json per graph)")
    issues = 0
    template_dir = SCHEMA_DIR / "template" / "schema_with_template"
    if template_dir.exists():
        ok(f"template dir exists: {template_dir}")
    else:
        warn(f"template dir NOT FOUND: {template_dir}  (may or may not be required)")

    for label, graphs in [("CypherBench", CYPHERBENCH_GRAPHS), ("MTQ", MTQ_GRAPHS)]:
        missing = [g for g in graphs if not (SCHEMA_DIR / f"{g}.json").exists()]
        present = [g for g in graphs if (SCHEMA_DIR / f"{g}.json").exists()]
        ok(f"{label}: {len(present)}/{len(graphs)} schema files present")
        for g in missing:
            fail(f"  MISSING: schema/{g}.json")
            issues += 1
    return issues


def check_neo4j_ports():
    step(3, "Neo4j ports (bolt://localhost:PORT)")
    issues = 0
    for port in CYPHERBENCH_PORTS:
        if check_port("localhost", port):
            ok(f"localhost:{port}  (CypherBench)")
        else:
            fail(f"localhost:{port}  (CypherBench)  -- not reachable")
            issues += 1
    for port in MTQ_PORTS:
        if check_port("localhost", port):
            ok(f"localhost:{port}  (MTQ/Sandbox)")
        else:
            fail(f"localhost:{port}  (MTQ/Sandbox)  -- not reachable")
            issues += 1
    return issues


def check_neo4j_login():
    step(4, "Neo4j authentication (try one DB)")
    try:
        from neo4j import GraphDatabase
    except ImportError:
        fail("neo4j package not installed. Try: pip install neo4j")
        return 1

    neo4j_host = os.environ.get("NEO4J_HOST", "localhost")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "cypherbench")
    sample_uri = f"bolt://{neo4j_host}:15065"   # geography
    try:
        drv = GraphDatabase.driver(sample_uri, auth=(neo4j_user, neo4j_password))
        with drv.session() as sess:
            res = sess.run("RETURN 1 AS x").single()
            assert res["x"] == 1
        drv.close()
        ok(f"Connected and authenticated on {sample_uri}")
        return 0
    except Exception as e:
        fail(f"Could not authenticate to {sample_uri}: {e}")
        return 1


def check_python_packages():
    step(5, "Python packages")
    issues = 0
    for mod in ["neo4j", "networkx", "numpy", "tqdm", "torch", "transformers"]:
        try:
            __import__(mod)
            ok(f"import {mod}")
        except ImportError as e:
            fail(f"import {mod}  --  {e}")
            issues += 1
    return issues


def check_bge():
    step(6, "BGE-M3 embedding model (REAL load, not mock)")
    if not BGE_LOAD_TEST:
        warn("BGE_LOAD_TEST=False, skipping")
        return 0

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        fail(f"transformers/torch not importable: {e}")
        return 1

    model_name = "BAAI/bge-m3"
    print(f"  Loading {model_name} (this may take 10-30s)...")
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        m = AutoModel.from_pretrained(model_name)
        dt = time.time() - t0
        ok(f"BGE-M3 loaded in {dt:.1f}s")
        if torch.cuda.is_available():
            ok(f"CUDA available: {torch.cuda.device_count()} device(s), "
               f"current={torch.cuda.get_device_name(0)}")
        else:
            warn("CUDA NOT available -- BGE will run on CPU (much slower)")
        return 0
    except Exception as e:
        fail(f"BGE-M3 load failed: {e}")
        fail("  >>> CRITICAL: code falls back to mock random vectors,")
        fail("  >>> which will silently produce garbage ablation results.")
        fail("  >>> Fix HF cache / network / disk before proceeding.")
        return 1


def check_code_layout():
    step(7, "Subgraph code presence")
    issues = 0
    cwd = Path.cwd()
    expected = [
        "all_subgraph_set.py",
        "subgraph_retrieval.py",
        "triple_retrieval.py",
        "graph_connector.py",
        "subgraph_triple.py",
    ]
    print(f"  CWD = {cwd}")
    for f in expected:
        if (cwd / f).exists():
            ok(f"./{f}")
        else:
            fail(f"./{f}  (MISSING)")
            issues += 1
    if issues:
        fail(f"You must run preflight from the subgraph code dir "
             f"(probably {SUBGRAPH_DIR})")
    return issues


def main():
    print(color("=" * 60, "bold"))
    print(color("  Beam-Size Ablation -- Pre-flight Check", "bold"))
    print(color("=" * 60, "bold"))

    total = 0
    total += check_paths()
    total += check_schema_files()
    total += check_neo4j_ports()
    total += check_neo4j_login()
    total += check_python_packages()
    total += check_bge()
    total += check_code_layout()

    print()
    print(color("=" * 60, "bold"))
    if total == 0:
        print(color("  ALL CHECKS PASSED -- safe to proceed.", "green"))
        print(color("=" * 60, "bold"))
        return 0
    else:
        print(color(f"  {total} ISSUE(S) FOUND -- fix before running sweep.", "red"))
        print(color("=" * 60, "bold"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
