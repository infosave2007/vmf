"""Tests for the held-out TOV tidal audit."""

import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_forward_tov_heldout_audit as audit


class ForwardTOVTests(unittest.TestCase):
    def test_validates_live(self):
        data = json.loads((HERE / "nvg_forward_tov_heldout_results.json").read_text())
        self.assertTrue(audit.validate_result(data))
        self.assertEqual(data["evidence_weight"], 0.0)

    def test_Mmax_below_2(self):
        data = json.loads((HERE / "nvg_forward_tov_heldout_results.json").read_text())
        for b in data["science"]["branches"]:
            self.assertLess(b["M_max_Msun"], 2.0)
            self.assertGreater(b["M_max_Msun"], 0.9)

    def test_crust_identical(self):
        data = json.loads((HERE / "nvg_forward_tov_heldout_results.json").read_text())
        crust = data["science"]["crust"]
        self.assertEqual(crust["p_match_MeV_fm3"], 1.5)
        self.assertEqual(crust["Gamma"], 1.35)

    def test_Mmax_ordering(self):
        data = json.loads((HERE / "nvg_forward_tov_heldout_results.json").read_text())
        # softer isovector running -> lower Mmax; y=0.87 (softest gamma) should have highest Mmax among quartic branch
        ms = {b["y_target"]: b["M_max_Msun"] for b in data["science"]["branches"]}
        self.assertGreater(ms["0.87"], ms["0.90"])
        self.assertGreater(ms["0.88"], ms["0.93"])

    def test_convergence(self):
        data = json.loads((HERE / "nvg_forward_tov_heldout_results.json").read_text())
        conv = data["science"]["convergence_y087"]
        self.assertAlmostEqual(conv["terminal_Mmax"], conv["tight_Mmax"], places=6)
        self.assertAlmostEqual(conv["terminal_Mmax"], conv["loose_Mmax"], places=3)


if __name__ == "__main__":
    unittest.main()
