"""Tests for the forward beta EOS table."""

import json
import pathlib
import sys
import unittest

import mpmath as mp

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_forward_beta_eos_audit as audit


class ForwardBetaTests(unittest.TestCase):
    def test_validates_live(self):
        data = json.loads((HERE / "nvg_forward_beta_eos_results.json").read_text())
        self.assertTrue(audit.validate_result(data))
        self.assertEqual(data["evidence_weight"], 0.0)

    def test_monotonic_pressure(self):
        data = json.loads((HERE / "nvg_forward_beta_eos_results.json").read_text())
        for tab in data["science"]["forward_tables"]:
            ps = [float(mp.mpf(r["P_MeV_fm3"])) for r in tab["table"] if "P_MeV_fm3" in r]
            self.assertEqual(ps, sorted(ps))
            # pressure should increase with density
            for i in range(1, len(ps)):
                self.assertGreater(ps[i], ps[i-1])

    def test_y087_gamma_and_C_ratio(self):
        with mp.workdps(80):
            sci = audit.calculate(80)
            y87 = next(t for t in sci["forward_tables"] if t["y_target"] == "0.87")
            gamma = mp.mpf(y87["gamma_for_L50"])
            self.assertAlmostEqual(float(gamma), -0.6278, places=3)
            ratio = mp.mpf(y87["C_ratio_at_2n0"])
            self.assertAlmostEqual(float(ratio), float(mp.power(2, gamma)), places=6)

    def test_delta_in_range(self):
        data = json.loads((HERE / "nvg_forward_beta_eos_results.json").read_text())
        for tab in data["science"]["forward_tables"]:
            for r in tab["table"]:
                d = float(mp.mpf(r["delta_beta"]))
                self.assertGreater(d, 0.5)
                self.assertLess(d, 1.0)
                self.assertGreater(float(mp.mpf(r["mu_e_MeV"])), 0)

    def test_causality(self):
        data = json.loads((HERE / "nvg_forward_beta_eos_results.json").read_text())
        for tab in data["science"]["forward_tables"]:
            if tab["max_cs2_fd"] is not None:
                self.assertLess(tab["max_cs2_fd"], 1.0)


if __name__ == "__main__":
    unittest.main()
