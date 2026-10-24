"""Tests for the density-dependent isovector extension over the fixed isoscalar jet."""

import json
import pathlib
import sys
import unittest

import mpmath as mp

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_isovector_density_dependent_audit as dd
import source_complete_scaling_saturation_audit as audit
from nvg_isospin_jet_audit import local_jet


class DensityDependentTests(unittest.TestCase):
    def test_result_validates_live(self):
        data = json.loads((HERE / "nvg_isovector_density_dependent_results.json").read_text())
        self.assertTrue(dd.validate_result(data))
        self.assertEqual(data["evidence_weight"], 0.0)
        self.assertEqual(data["precision_check"]["pass"], True)

    def test_baseline_is_source_complete(self):
        with mp.workdps(80):
            b = dd.calculate(80)["baseline_at_n0_source_complete"]
            self.assertAlmostEqual(float(mp.mpf(b["P_MeV_fm3"])), 18.7236, places=2)
            self.assertAlmostEqual(float(mp.mpf(b["binding_MeV"])), 124.3, places=1)
            self.assertAlmostEqual(float(mp.mpf(b["K_MeV"])), 2063.6, places=0)

    def test_constant_Crho_still_fails(self):
        with mp.workdps(80):
            sci = dd.calculate(80)
            y90 = next(r for r in sci["constant_Crho_family"] if r["y_target"] == "0.90")
            self.assertGreater(float(mp.mpf(y90["L_const_MeV"])), 65)
            self.assertAlmostEqual(float(mp.mpf(y90["T_MeV"])), 13.02, places=1)
            # recompute via local_jet
            model, state, _ = audit.inverse_potential_jet("0.90", "240", "0.16", "-16")
            lj0 = local_jet(state, 0)
            T = lj0["T"]
            C0 = 8 * (mp.mpf(32) - T) / state["n"]
            lj = local_jet(state, C0)
            self.assertAlmostEqual(float(lj["L"]), float(mp.mpf(y90["L_const_MeV"])), places=4)

    def test_gamma_formula_for_L50(self):
        with mp.workdps(80):
            sci = dd.calculate(80)
            for y_target in ("0.87", "0.90", "0.93"):
                row = next(r for r in sci["constant_Crho_family"] if r["y_target"] == y_target)
                L_kin = mp.mpf(row["L_kin_MeV"])
                L_rho = mp.mpf(row["L_rho_const_MeV"])
                gamma = mp.mpf(row["gamma_for_L50"])
                L_gamma = L_kin + L_rho * (1 + gamma)
                self.assertAlmostEqual(float(L_gamma), 50.0, places=4)
                # alpha = -gamma for rational at n0 to same order
                alpha = mp.mpf(row["alpha_for_L50"])
                self.assertAlmostEqual(float(alpha), -float(gamma), places=6)
                L_alpha = L_kin + L_rho * (1 - alpha)
                self.assertAlmostEqual(float(L_alpha), 50.0, places=4)

    def test_gamma_window_ordering(self):
        with mp.workdps(80):
            sci = dd.calculate(80)
            for row in sci["constant_Crho_family"]:
                if "gamma_for_L35" not in row:
                    continue
                g35 = float(mp.mpf(row["gamma_for_L35"]))
                g50 = float(mp.mpf(row["gamma_for_L50"]))
                g65 = float(mp.mpf(row["gamma_for_L65"]))
                # 35<50<65 in L => gamma ordering depends on sign of L_rho>0 always
                # L = L_kin + L_rho*(1+gamma) strictly monotonic in gamma
                self.assertLess(g35, g50)
                self.assertLess(g50, g65)

    def test_C_ratio_at_2n0(self):
        with mp.workdps(80):
            sci = dd.calculate(80)
            row = next(r for r in sci["constant_Crho_family"] if r["y_target"] == "0.90")
            gamma = mp.mpf(row["gamma_for_L50"])
            alpha = mp.mpf(row["alpha_for_L50"])
            f_pow = float(mp.mpf(row["C_ratio_at_2n0_power"]))
            f_rat = float(mp.mpf(row["C_ratio_at_2n0_rational"]))
            self.assertAlmostEqual(f_pow, float(mp.power(2, gamma)), places=6)
            self.assertAlmostEqual(f_rat, float(1 / (1 + alpha)), places=6)
            # globally stable y needs only ~0.6 suppression
            self.assertGreater(f_pow, 0.5)
            self.assertLess(f_pow, 0.75)
            self.assertGreater(f_rat, 0.5)
            self.assertLess(f_rat, 0.75)

    def test_low_y_requires_steeper_running(self):
        with mp.workdps(80):
            sci = dd.calculate(80)
            y65 = next(r for r in sci["constant_Crho_family"] if r["y_target"] == "0.65")
            y90 = next(r for r in sci["constant_Crho_family"] if r["y_target"] == "0.90")
            self.assertLess(float(mp.mpf(y65["gamma_for_L50"])), float(mp.mpf(y90["gamma_for_L50"])))
            self.assertLess(float(mp.mpf(y65["gamma_for_L50"])), -2)

    def test_order5_can_stabilize_low_y(self):
        with mp.workdps(80):
            sci = dd.calculate(80)
            # y=0.65 with a5=1e9 should be globally nonneg
            hit = [r for r in sci["order5_global_stabilization_scan"] if r["y_target"] == "0.65" and r["a5_MeV4"] == "1000000000.0"]
            self.assertEqual(len(hit), 1)
            self.assertTrue(hit[0]["global_nonneg"])
            self.assertTrue(hit[0]["bounded"])
            # y=0.85 with a5=5e8 remains unstable, with 1.5e9 becomes stable (demonstrates threshold)
            r05 = next(r for r in sci["order5_global_stabilization_scan"] if r["y_target"] == "0.85" and r["a5_MeV4"] == "500000000.0")
            r15 = next(r for r in sci["order5_global_stabilization_scan"] if r["y_target"] == "0.85" and r["a5_MeV4"] == "1500000000.0")
            self.assertFalse(r05["global_nonneg"])
            self.assertTrue(r15["global_nonneg"])

    def test_feasibility_interpretation(self):
        with mp.workdps(80):
            sci = dd.calculate(80)
            feas = sci["feasibility"]
            self.assertFalse(feas["constant_Crho_any_y_meets_L35_65_with_global_quartic"])
            self.assertIn("0.90", feas["running_Crho_quartic_y_reaching_L50"])
            self.assertNotIn("0.70", feas["running_Crho_quartic_y_reaching_L50"])

    def test_diagnostic_S_at_2n0_positive(self):
        with mp.workdps(80):
            sci = dd.calculate(80)
            for row in sci["constant_Crho_family"]:
                if "S_at_2n0_diag_power_MeV" not in row:
                    continue
                self.assertGreater(float(mp.mpf(row["S_at_2n0_diag_power_MeV"])), 0)
                self.assertGreater(float(mp.mpf(row["S_at_2n0_diag_rational_MeV"])), 0)


if __name__ == "__main__":
    unittest.main()
