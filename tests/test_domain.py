"""核对领域原语的业务数值。"""
import unittest
from analysis_data.io import load
from analysis_data.domain import summarize
class DomainTests(unittest.TestCase):
    def test_weighted_observations(self):
        rows=summarize(load('data/normal.json')[0]);self.assertAlmostEqual(rows[0]['weighted_mean'],0.4)
        self.assertEqual(rows[0]['runs'],3);self.assertEqual(rows[0]['weight'],7500)
        flat=summarize(load('data/flat-platform.json')[0]);self.assertTrue(all(abs(r['weighted_mean']-0.8)<1e-12 for r in flat))
