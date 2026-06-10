"""Tests for per-query instrumentation. Run from inside the package dir:

    python3 -m unittest test_instrumentation -v
"""

import threading
import unittest

import instrumentation as instr


class TrackQueryTest(unittest.TestCase):
    def test_inactive_by_default(self):
        self.assertIsNone(instr.current())
        instr.record_llm_call(1.0)  # no collector — must be a no-op, not an error
        with instr.stage("analysis"):
            pass

    def test_collects_stages_llm_and_total(self):
        with instr.track_query() as stats:
            with instr.stage("analysis"):
                pass
            with instr.stage("generation"):
                pass
            with instr.stage("generation"):  # repeated stage accumulates
                pass
            instr.record_llm_call(0.5)
            instr.record_llm_call(0.25)
        self.assertEqual(stats.llm_calls, 2)
        self.assertAlmostEqual(stats.llm_seconds, 0.75)
        self.assertEqual(set(stats.stage_seconds), {"analysis", "generation"})
        self.assertGreater(stats.total_seconds, 0.0)
        self.assertIsNone(instr.current())

    def test_stage_records_time_on_exception(self):
        with instr.track_query() as stats:
            with self.assertRaises(ValueError):
                with instr.stage("retrieval"):
                    raise ValueError("boom")
        self.assertIn("retrieval", stats.stage_seconds)

    def test_nested_tracking_restores_outer(self):
        with instr.track_query() as outer:
            instr.record_llm_call(0.1)
            with instr.track_query() as inner:
                instr.record_llm_call(0.2)
            instr.record_llm_call(0.3)
        self.assertEqual(inner.llm_calls, 1)
        self.assertEqual(outer.llm_calls, 2)

    def test_as_dict_shape(self):
        with instr.track_query() as stats:
            pass
        d = stats.as_dict()
        self.assertEqual(
            set(d),
            {
                "total_seconds", "stage_seconds", "llm_calls", "llm_seconds",
                "probe_count", "probe_seconds", "prompt_count", "prompt_chars",
                "prompt_tokens_est",
            },
        )


class PromptRecordingTest(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertEqual(instr.estimate_tokens(""), 0)
        self.assertGreater(instr.estimate_tokens("MATCH (n) RETURN n"), 0)

    def test_record_prompt_accumulates(self):
        with instr.track_query() as stats:
            instr.record_prompt(100, 25)
            instr.record_prompt(40, 10)
        self.assertEqual(stats.prompt_count, 2)
        self.assertEqual(stats.prompt_chars, 140)
        self.assertEqual(stats.prompt_tokens_est, 35)

    def test_record_prompt_no_collector(self):
        instr.record_prompt(10, 3)  # must be a no-op


class CountFnTest(unittest.TestCase):
    def test_probes_counted_and_value_passthrough(self):
        calls = []

        def count_fn(cypher):
            calls.append(cypher)
            return 42

        wrapped = instr.instrumented_count_fn(count_fn)
        with instr.track_query() as stats:
            self.assertEqual(wrapped("MATCH (n) RETURN count(n)"), 42)
            self.assertEqual(wrapped("MATCH (m) RETURN count(m)"), 42)
        self.assertEqual(stats.probe_count, 2)
        self.assertGreaterEqual(stats.probe_seconds, 0.0)
        self.assertEqual(len(calls), 2)

    def test_probe_counted_even_when_count_fn_raises(self):
        def count_fn(cypher):
            raise RuntimeError("neo4j down")

        wrapped = instr.instrumented_count_fn(count_fn)
        with instr.track_query() as stats:
            with self.assertRaises(RuntimeError):
                wrapped("MATCH (n) RETURN count(n)")
        self.assertEqual(stats.probe_count, 1)

    def test_no_collector_passthrough(self):
        wrapped = instr.instrumented_count_fn(lambda c: 7)
        self.assertEqual(wrapped("x"), 7)


class ThreadIsolationTest(unittest.TestCase):
    def test_threads_do_not_share_collector(self):
        seen = {}

        def worker():
            seen["other_thread"] = instr.current()

        with instr.track_query():
            t = threading.Thread(target=worker)
            t.start()
            t.join()
        self.assertIsNone(seen["other_thread"])


if __name__ == "__main__":
    unittest.main()
