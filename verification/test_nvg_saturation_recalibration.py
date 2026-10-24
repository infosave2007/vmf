"""Tests for the saturation recalibration counterfactuals."""

import json
import pathlib
import sys
import unittest

import mpmath as mp

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_saturation_recalibration as rec
import source_complete_scaling_saturation_audit as audit
from nvg_isospin_jet_audit import local_jet


class SaturationRecalibrationTests(unittest.TestCase):
    def test_result_validates_live(self):
        data = json.loads((HERE / "nvg_saturation_recalibration_results.json").read_text())
        self.assertTrue(rec.validate_result(data))
        self.assertEqual(data["evidence_weight"], 0.0)

    def test_baseline_is_source_complete(self):
        with mp.workdps(80):
            sci = rec.calculate(80)["baseline_at_n0"]
            self.assertAlmostEqual(float(mp.mpf(sci["P_MeV_fm3"])), 18.7236, places=2)
            self.assertAlmostEqual(float(mp.mpf(sci["binding_MeV"])), 124.3, places=1)
            self.assertAlmostEqual(float(mp.mpf(sci["Cv_over_Cs"])), 216.84, places=1)
            self.assertTrue(sci["scale_identity_pass"])
            self.assertTrue(sci["legendre_pass"])

    def test_quartic_calibration_only_P_and_binding(self):
        with mp.workdps(80):
            row = rec.calculate(80)["two_target_quartic_P_and_binding_only"]
            self.assertAlmostEqual(float(mp.mpf(row["binding_MeV"])), -16.0, places=6)
            self.assertTrue(row["K_fd_pass"])
            # K was not fitted, must be far above window
            self.assertGreater(float(mp.mpf(row["K_MeV"])), 350)
            self.assertTrue(row["P_MeV_fm3"].startswith("0"))

    def test_vector_ceiling_blocks_original_gomega(self):
        with mp.workdps(80):
            ceil = rec.calculate(80)["vector_ceiling_any_potential"]
            self.assertLess(float(mp.mpf(ceil["gomega_max"])), 10.12)

    def test_global_potential_threshold(self):
        with mp.workdps(80):
            sci = rec.calculate(80)
            # y<0.87 must be globally unbounded
            for row in sci["three_target_family_P_binding_K"]:
                if row.get("y_target") in ("0.60", "0.70", "0.75", "0.80", "0.85"):
                    self.assertFalse(row["globally_nonnegative_potential"])
                if row.get("y_target") in ("0.87", "0.88", "0.89", "0.90", "0.93"):
                    self.assertTrue(row["globally_nonnegative_potential"])

    def test_L_bottleneck_for_J32(self):
        with mp.workdps(80):
            sci = rec.calculate(80)
            for row in sci["three_target_family_P_binding_K"]:
                if row.get("y_target") == "0.90":
                    L = float(mp.mpf(row["isovector"]["L_MeV"]))
                    self.assertGreater(L, 65)  # outside L window
                    self.assertAlmostEqual(float(mp.mpf(row["isovector"]["J_MeV"])), 32.0, places=6)
                    # direct check against local_jet recomputation
                    model, state, _ = audit.inverse_potential_jet("0.90", "240", "0.16", "-16")
                    lj = local_jet(state, mp.mpf(row["isovector"]["C_rho_MeV_minus2"]))
                    self.assertAlmostEqual(float(lj["L"]), L, places=4)

    def test_feasibility_verdict(self):
        with mp.workdps(80):
            sci = rec.calculate(80)
            self.assertEqual(sci["feasibility"]["verdict"],
                             "NO_Y_SATISFIES_P_BINDING_K_J_L_MSTAR_WITH_GLOBAL_U8_AND_CONSTANT_CRHO")
            self.assertFalse(sci["feasibility"]["any_y_meets_all_windows_with_global_potential_and_constant_Crho"])
            for c in sci["empirical_compliance"]:
                self.assertFalse(c["all_five"])

    def test_no_y_with_constant_Crho_can_meet_L_window(self):
        """Independent scan: even varying J in 30-34, no physical y meets L<=65 with constant rho."""
        with mp.workdps(80):
            found = False
            for y_text in ("0.87", "0.88", "0.89", "0.90", "0.93"):
                model, state, jet = audit.inverse_potential_jet(y_text, "240", "0.16", "-16")
                if not jet["globally_nonnegative_potential"]:
                    continue
                for J in ("30", "32", "34"):
                    base = local_jet(state, 0)
                    T = base["T"]
                    if mp.mpf(J) <= T:
                        continue
                    c_rho = 8 * (mp.mpf(J) - T) / state["n"]
                    lj = local_jet(state, c_rho)
                    if lj["L"] <= 65 and state["y"] <= 0.7:
                        found = True
            self.assertFalse(found)

    def test_price_ratios(self):
        with mp.workdps(80):
            sci = rec.calculate(80)
            price = sci["price_of_breaking_source_complete"]
            self.assertLess(float(mp.mpf(price["Cv_ratio_y0.90_over_original"])), 0.2)
            self.assertLess(float(mp.mpf(price["lam_ratio_quartic_over_original"])), 0.01)

    def test_identity_K_second_derivative(self):
        with mp.workdps(80):
            model, state = audit.quartic_calibration()
            def binding_at(n):
                q = model.equilibrium(n, guess=("0.76", "0.80"))
                return q["energy_total"] / q["n"] - model.MN
            K2 = 9 * state["n"] ** 2 * mp.diff(binding_at, state["n"], 2)
            self.assertLess(abs(K2 - state["K"]) / max(abs(state["K"]), mp.mpf(1)), mp.mpf("1e-55"))


if __name__ == "__main__":
    unittest.main()
