"""Focused tests for the grand-canonical fluctuation-ratio postprocessor."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_gc_fluctuation_ratio as ratio  # noqa: E402


class GrandCanonicalFluctuationRatioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = HERE / "nvg_gc_fluctuation_ratio_results.json"
        cls.result = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_derived_artifact_validates(self):
        self.assertTrue(ratio.validate_result(self.result))
        self.assertEqual(self.result["coverage"], {"primary_rows": 8, "refined_rows": 8, "paired_W8_U16_contrasts": 4})
        self.assertTrue(self.result["all_controls_pass"])
        self.assertEqual(self.result["independent_evidence_weight"], 0.0)

    def test_ratio_is_recomputed_from_live_upstream_rows(self):
        for row in self.result["primary_rows"] + self.result["refined_rows"]:
            expected = float(row["T_MeV"]) * float(row["chi_T_MeV2"]) / float(row["n_MeV3"])
            self.assertAlmostEqual(float(row["kappa2_over_kappa1"]), expected, places=13)
            self.assertAlmostEqual(float(row["kappa2_over_kappa1_from_static_q0"]), expected, places=7)

    def test_mutated_observable_is_rejected(self):
        changed = copy.deepcopy(self.result)
        changed["primary_rows"][0]["kappa2_over_kappa1"] = "1.0"
        self.assertFalse(ratio.validate_result(changed))


if __name__ == "__main__":
    unittest.main()
