"""Tests for mechanism diagnostics. Run from inside the package dir:

    python3 -m unittest test_diagnostics -v
"""

import unittest

from diagnostics import (
    EXRecord,
    SchemaView,
    classify_hallucinations,
    empty_result_accuracy,
    gold_metapath_recall_at_b,
    hallucination_attribution,
    mechanism_diagnostics,
    parse_cypher,
    probe_precision,
)

SCHEMA = {
    "nodes": [
        {"label": "River", "properties": ["name", "length"]},
        {"label": "Country", "properties": ["name", "population"]},
        {"label": "City", "properties": {"name": "STRING", "founded": "INT"}},
    ],
    "relationships": [
        {"type": "FLOWS_THROUGH", "from": "River", "to": "Country"},
        {"type": "CAPITAL_OF", "from": "City", "to": "Country"},
    ],
}


class SchemaViewTest(unittest.TestCase):
    def setUp(self):
        self.view = SchemaView.from_schema(SCHEMA)

    def test_labels_and_rels_case_insensitive(self):
        self.assertTrue(self.view.known_label("river"))
        self.assertTrue(self.view.known_label("Country"))
        self.assertTrue(self.view.known_rel("flows_through"))
        self.assertFalse(self.view.known_label("Ocean"))
        self.assertFalse(self.view.known_rel("BORDERS"))

    def test_props_dict_and_list(self):
        self.assertTrue(self.view.known_prop("River", "length"))
        self.assertTrue(self.view.known_prop("City", "founded"))
        self.assertFalse(self.view.known_prop("River", "altitude"))

    def test_label_pairs_undirected(self):
        self.assertIn(("country", "river"), self.view.label_pairs)
        self.assertNotIn(("city", "river"), self.view.label_pairs)


class ParseCypherTest(unittest.TestCase):
    def test_extracts_labels_rels_props(self):
        s = parse_cypher("MATCH (r:River)-[:FLOWS_THROUGH]->(c:Country) WHERE r.length > 100 RETURN c.name")
        self.assertEqual(set(s.labels), {"River", "Country"})
        self.assertEqual(set(s.rel_types), {"FLOWS_THROUGH"})
        self.assertIn(("r", "length"), s.prop_access)
        self.assertIn(("c", "name"), s.prop_access)
        self.assertEqual(s.var_labels["r"], "River")

    def test_string_literals_ignored(self):
        s = parse_cypher("MATCH (c:Country) WHERE c.name = 'River.Of.Doom' RETURN c")
        # 'River.Of.Doom' is a literal; only c.name is a real property access
        self.assertEqual([p for p in s.prop_access if p[0] == "c"], [("c", "name")])

    def test_adjacency_captured(self):
        s = parse_cypher("MATCH (r:River)-[:FLOWS_THROUGH]->(c:Country) RETURN c")
        self.assertEqual(s.adjacencies[0][1], "FLOWS_THROUGH")


class HallucinationTest(unittest.TestCase):
    def setUp(self):
        self.view = SchemaView.from_schema(SCHEMA)

    def test_clean_query(self):
        flags = classify_hallucinations(
            "MATCH (r:River)-[:FLOWS_THROUGH]->(c:Country) RETURN c.name", self.view
        )
        self.assertFalse(any(flags.values()))

    def test_phantom_node(self):
        flags = classify_hallucinations("MATCH (o:Ocean) RETURN o", self.view)
        self.assertTrue(flags["phantom_node"])

    def test_phantom_relation(self):
        flags = classify_hallucinations(
            "MATCH (r:River)-[:BORDERS]->(c:Country) RETURN c", self.view
        )
        self.assertTrue(flags["phantom_relation"])

    def test_phantom_attribute(self):
        flags = classify_hallucinations("MATCH (r:River) RETURN r.altitude", self.view)
        self.assertTrue(flags["phantom_attribute"])

    def test_invalid_connectivity(self):
        # City and River both exist, FLOWS_THROUGH exists, but City-River is not a schema pair
        flags = classify_hallucinations(
            "MATCH (c:City)-[:FLOWS_THROUGH]->(r:River) RETURN r", self.view
        )
        self.assertTrue(flags["invalid_connectivity"])
        self.assertFalse(flags["phantom_node"])
        self.assertFalse(flags["phantom_relation"])

    def test_attribution_rates(self):
        preds = [
            "MATCH (r:River)-[:FLOWS_THROUGH]->(c:Country) RETURN c.name",  # clean
            "MATCH (o:Ocean) RETURN o",                                      # phantom node
            "MATCH (r:River) RETURN r.altitude",                            # phantom attr
        ]
        out = hallucination_attribution(preds, SCHEMA)
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["no_hallucination"], 1 / 3)
        self.assertAlmostEqual(out["phantom_node"], 1 / 3)
        self.assertAlmostEqual(out["phantom_attribute"], 1 / 3)
        self.assertEqual(out["phantom_relation"], 0.0)

    def test_empty_input(self):
        out = hallucination_attribution([], SCHEMA)
        self.assertEqual(out["n"], 0)


class BeamProbeEmptyTest(unittest.TestCase):
    def test_gold_recall(self):
        beam = [("River", "FLOWS_THROUGH>", "Country"), ("City", "CAPITAL_OF>", "Country")]
        gold = [("River", "FLOWS_THROUGH>", "Country")]
        self.assertEqual(gold_metapath_recall_at_b(beam, gold), 1.0)

    def test_gold_recall_case_insensitive_and_topb(self):
        beam = [("city", "capital_of>", "country"), ("river", "flows_through>", "country")]
        gold = [("River", "FLOWS_THROUGH>", "Country")]
        self.assertEqual(gold_metapath_recall_at_b(beam, gold, beam_width=1), 0.0)
        self.assertEqual(gold_metapath_recall_at_b(beam, gold, beam_width=2), 1.0)

    def test_gold_recall_none_without_gold(self):
        self.assertIsNone(gold_metapath_recall_at_b([("A",)], []))

    def test_probe_precision_from_floats_and_objects(self):
        class C:
            def __init__(self, n):
                self.n_hat = n

        self.assertEqual(probe_precision([3, 0, 5, 0]), 0.5)
        self.assertEqual(probe_precision([C(2), C(0), C(0), C(0)]), 0.25)
        self.assertIsNone(probe_precision([]))

    def test_empty_result_accuracy(self):
        records = [
            EXRecord(gold_empty=True, correct=True),
            EXRecord(gold_empty=True, correct=False),
            EXRecord(gold_empty=False, correct=True),  # ignored
        ]
        self.assertEqual(empty_result_accuracy(records), 0.5)
        self.assertIsNone(empty_result_accuracy([EXRecord(gold_empty=False, correct=True)]))


class IntegrationTest(unittest.TestCase):
    def test_full_block_partial_inputs(self):
        out = mechanism_diagnostics(
            SCHEMA,
            ["MATCH (r:River)-[:FLOWS_THROUGH]->(c:Country) RETURN c.name"],
        )
        self.assertIsNone(out["gold_metapath_recall_at_b"])
        self.assertIsNone(out["probe_precision"])
        self.assertIsNone(out["empty_result_accuracy"])
        self.assertEqual(out["hallucination_attribution"]["no_hallucination"], 1.0)

    def test_full_block_all_inputs(self):
        out = mechanism_diagnostics(
            SCHEMA,
            ["MATCH (r:River)-[:FLOWS_THROUGH]->(c:Country) RETURN c"],
            beams=[[("River", "FLOWS_THROUGH>", "Country")]],
            golds=[[("River", "FLOWS_THROUGH>", "Country")]],
            probe_results=[3, 0],
            ex_records=[EXRecord(gold_empty=True, correct=True)],
        )
        self.assertEqual(out["gold_metapath_recall_at_b"], 1.0)
        self.assertEqual(out["probe_precision"], 0.5)
        self.assertEqual(out["empty_result_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
