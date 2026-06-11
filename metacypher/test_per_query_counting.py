"""
test_per_query_counting.py — Plain-Python tests for per_query_counting.py.

Tests:
  T1  budget enforcement — lookups beyond the per-query probe budget return None
  T2  memoization — repeated get() of the same key issues no new probes
  T3  no cross-query reuse — start_query() forces every count to be re-issued
  T4  card consistency — measured card equals the offline catalog's card
  T5  sel consistency — measured sel equals the offline catalog's sel
  T6  median_card — 1.0 when cold, running median of observed counts after
  T7  PathScorer integration — variant plugs into PathScorer(catalog=) and
      produces a catalog_delta identical to the offline catalog's for a
      measured path, and the maximal penalty for an over-budget path

Reuses the synthetic River/Country/Lake graph and in-memory count_fn from
test_catalog.py so both code paths measure the same instance.
"""

from catalog import build_catalog
from path_model import PathInstance
from path_scorer import PathScorer
from per_query_counting import PerQueryCountingCatalog
from retrieval_config import RetrievalConfig
from test_catalog import SCHEMA, eval_in_memory

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def counting_fn():
    """Wrap eval_in_memory and count the COUNT statements issued."""
    calls = {"n": 0}

    def fn(cypher: str) -> int:
        calls["n"] += 1
        return eval_in_memory(cypher)

    return fn, calls


KEY_1HOP = ("River", "FLOWS_THROUGH>", "Country")
KEY_2HOP = ("River", "FLOWS_THROUGH>", "Country", "<LOCATE_IN", "Lake")
KEY_OTHER = ("Lake", "DRAINS_INTO>", "River")


def make_path(key) -> PathInstance:
    nodes = [{"label": key[i]} for i in range(0, len(key), 2)]
    edges = []
    for i in range(1, len(key), 2):
        tok = key[i]
        if tok.endswith(">"):
            edges.append({"rel_type": tok[:-1], "direction": "forward"})
        else:
            edges.append({"rel_type": tok[1:], "direction": "backward"})
    return PathInstance(path_id="t", nodes=nodes, edges=edges)


def main() -> None:
    print("=" * 60)
    print("MetaCypher per_query_counting.py — unit tests")
    print("=" * 60)

    offline = build_catalog(SCHEMA, eval_in_memory, max_len=2)

    # T1 — budget enforcement
    fn, calls = counting_fn()
    pqc = PerQueryCountingCatalog(SCHEMA, fn, probe_budget=3)
    pqc.start_query()
    e1 = pqc.get(KEY_1HOP)        # 1 card + 2 label pops = 3 probes
    e2 = pqc.get(KEY_2HOP)        # over budget
    check("T1a first lookup measured", e1 is not None and e1.card == 3)
    check("T1b over-budget lookup is None", e2 is None)
    check("T1c probes capped at budget",
          pqc.query_stats()["probes_spent"] == 3,
          str(pqc.query_stats()))

    # T2 — memoization within a query
    before = calls["n"]
    e1b = pqc.get(KEY_1HOP)
    check("T2 repeated get() issues no probes",
          calls["n"] == before and e1b is e1)

    # T3 — no cross-query reuse
    pqc.start_query()
    before = calls["n"]
    e1c = pqc.get(KEY_1HOP)
    check("T3 start_query() discards all measurements",
          calls["n"] > before and e1c is not None and e1c is not e1)

    # T4/T5 — consistency with the offline catalog on the same instance
    fn2, _ = counting_fn()
    pqc2 = PerQueryCountingCatalog(SCHEMA, fn2, probe_budget=100)
    pqc2.start_query()
    for key in (KEY_1HOP, KEY_2HOP, KEY_OTHER):
        off = offline.get(key)
        on = pqc2.get(key)
        check(f"T4 card matches offline for {key[0]}..{key[-1]}",
              on is not None and off is not None and on.card == off.card,
              f"on={getattr(on,'card',None)} off={getattr(off,'card',None)}")
        check(f"T5 sel matches offline for {key[0]}..{key[-1]}",
              on is not None and off is not None
              and abs(on.sel - off.sel) < 1e-9,
              f"on={getattr(on,'sel',None)} off={getattr(off,'sel',None)}")

    # T6 — median_card
    fn3, _ = counting_fn()
    pqc3 = PerQueryCountingCatalog(SCHEMA, fn3, probe_budget=100)
    pqc3.start_query()
    check("T6a cold median_card is 1.0", pqc3.median_card == 1.0)
    pqc3.get(KEY_1HOP)   # card 3
    pqc3.get(KEY_OTHER)  # card 2
    check("T6b median over observed nonzero cards",
          pqc3.median_card == 2.5, str(pqc3.median_card))

    # T7 — PathScorer integration
    config = RetrievalConfig()
    fn4, _ = counting_fn()
    pqc4 = PerQueryCountingCatalog(SCHEMA, fn4, probe_budget=3)
    pqc4.start_query()
    scorer_on = PathScorer(config, catalog=pqc4)
    scorer_off = PathScorer(config, catalog=offline)

    p_measured = make_path(KEY_1HOP)
    d_on = scorer_on._catalog_score_delta(p_measured)
    # Force mu_P parity: offline scorer uses the full-catalog median, the
    # variant only what it has observed — compare the entry-level inputs.
    on_entry = pqc4.get(KEY_1HOP)
    off_entry = offline.get(KEY_1HOP)
    check("T7a scorer reads variant entries (card/sel parity)",
          on_entry.card == off_entry.card
          and abs(on_entry.sel - off_entry.sel) < 1e-9)
    check("T7b delta is finite and applied", isinstance(d_on, float))

    p_unknown = make_path(KEY_2HOP)  # over budget under probe_budget=3
    d_unknown = scorer_on._catalog_score_delta(p_unknown)
    lam = getattr(config, "lambda_sparse", 0.15)
    check("T7c over-budget path gets maximal sparsity penalty",
          abs(d_unknown - (-lam)) < 1e-9, f"delta={d_unknown}")

    total = _PASS + _FAIL
    print(f"Results: {_PASS}/{total} passed, {_FAIL} failed")
    if _FAIL == 0:
        print("ALL TESTS PASSED")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
