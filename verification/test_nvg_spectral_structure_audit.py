"""Tests for the bounded stationary spectral-structure audit."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import unittest
from unittest import mock

import mpmath as mp

import nvg_spectral_structure_audit as audit


class SpectralStructureTests(unittest.TestCase):
    def test_symbolic_recurrence_closed_form_and_inverse(self):
        rows = audit.symbolic_checks()
        self.assertGreaterEqual(len(rows), 9)
        self.assertTrue(all(row["passed"] is True and row["residual"] == "0"
                            for row in rows.values()))
        with mp.workdps(80):
            rec = audit.coefficient_recurrence(4, 1, max_order=5)["moments"]
            closed = audit.closed_moments(4, 1)
            for power in (1, 3, 5):
                self.assertLess(audit._relative_real(rec[power], closed[power]),
                                audit.MOMENT_TOLERANCE)

    def test_recurrence_and_independent_jacobi_powers_through_m9(self):
        with mp.workdps(80):
            result = audit.recurrence_jacobi_check(4, 1, dps=80)
            self.assertTrue(result["passed"])
            self.assertLess(result["max_relative_error"], mp.mpf("1e-35"))
            self.assertEqual(set(result["recurrence"]), {1, 3, 5, 7, 9})

    def test_direct_quadrature_plus_analytic_pole_and_tiny_weight(self):
        with mp.workdps(80):
            moments = audit.spectral_moments(4, 1, dps=80)
            self.assertTrue(moments["passed"])
            self.assertTrue(moments["pole"]["exists"])
            self.assertEqual(set(moments["target_relative_errors"]),
                             set(audit.MOMENT_ORDERS))
            self.assertTrue(all(value < audit.MOMENT_TOLERANCE
                                for value in moments["target_relative_errors"].values()))
            for power in (1, 3, 5):
                self.assertGreater(moments["pole_contribution"][power], 0)

            near = audit.spectral_moments("3.01", 1, dps=80)
            self.assertTrue(near["passed"])
            self.assertGreater(near["pole_contribution"][1], 0)
            self.assertLess(near["pole_contribution"][1], mp.mpf("1e-80"))
            self.assertTrue(all(value < audit.MOMENT_TOLERANCE
                                for value in near["pole_piece_relative_errors"].values()))

    def test_pole_consumption_rejects_tiny_gap_and_residue_tampering(self):
        with mp.workdps(110):
            genuine = audit.kinetic.pole_data("3.01", 1, dps=110)
        mutations = []
        wrong_gap = copy.deepcopy(genuine)
        wrong_gap["gap"] *= 2
        mutations.append(("gap", wrong_gap))
        wrong_log_gap = copy.deepcopy(genuine)
        wrong_log_gap["log_gap"] += mp.mpf("1e-8")
        mutations.append(("log_gap", wrong_log_gap))
        wrong_residue = copy.deepcopy(genuine)
        wrong_residue["residue"] *= -1
        mutations.append(("residue sign", wrong_residue))
        false_existence = copy.deepcopy(genuine)
        false_existence["exists"] = False
        mutations.append(("existence", false_existence))
        for label, payload in mutations:
            with self.subTest(mutation=label), mock.patch.object(
                    audit, "_pole_for", return_value=payload):
                with self.assertRaises(ArithmeticError):
                    audit.spectral_moments("3.01", 1, dps=80)

        # Both a normal isolated pole and a legitimate absent-pole state pass.
        self.assertTrue(audit.spectral_moments(4, 1, dps=80)["passed"])
        absent = audit.spectral_moments("2.9", 1, dps=80)
        self.assertTrue(absent["passed"])
        self.assertFalse(absent["pole"]["exists"])

    def test_centroid_is_not_pole_speed(self):
        with mp.workdps(110):
            moments = {"total": audit.closed_moments("3.01", 1)}
            raw = {
                "vF2": mp.mpf(".7"),
                "r": mp.mpf(1),
                "first_sound_squared": mp.mpf(".7")*(1+mp.mpf("3.01"))/6,
                "n": mp.mpf(1), "kF": mp.mpf(1), "mu": mp.mpf(1),
            }
            scale = audit.c_scale_and_shear(raw, moments)
            self.assertLess(scale["c_scale_squared"]/raw["vF2"], 1)
            pole = audit._pole_for(mp.mpf("3.01"), mp.mpf(1), dps=110)
            self.assertTrue(pole["exists"])
            self.assertGreater(pole["sigma_p"], 1)

    def test_resolvent_and_passive_function(self):
        result = audit.resolvent_checks(dps=100)
        self.assertTrue(result["passed"])
        self.assertLess(result["max_dimension256_absolute_error"], mp.mpf("1e-50"))
        self.assertGreater(result["min_dimension256_Im_minus_zg"], 0)
        self.assertIn("H=-z*g", result["passive_function"])

    def test_outside_domain_witness_and_admissible_rejection(self):
        with self.assertRaises(ValueError):
            audit.jacobi_matrix("-1.5", 0, dimension=8)
        with self.assertRaises(ValueError):
            audit.equilibrium_coefficients("nan")
        witness = audit.outside_domain_instability_witness(dps=110)
        self.assertTrue(witness["passed"])
        self.assertFalse(witness["admissible_state"])
        self.assertGreater(witness["y"], 0)
        self.assertLess(witness["denominator_residual"], mp.mpf("1e-70"))

    def test_exact_pole_partials_and_absence_semantics(self):
        ordinary = audit.pole_sensitivity(4, 1, dps=110)
        near = audit.pole_sensitivity("3.01", 1, dps=110)
        self.assertTrue(ordinary["passed"])
        self.assertTrue(near["passed"])
        self.assertLess(ordinary["relative_error_F0"], mp.mpf("1e-18"))
        self.assertLess(near["relative_error_r"], mp.mpf("1e-18"))
        absent = audit.pole_sensitivity("2.9", 1, dps=110)
        self.assertTrue(absent["passed"])
        self.assertIsNone(absent["partial_F0"])
        self.assertEqual(absent["absence_semantics"], "absent, not zero")

    def test_density_slopes_and_formal_threshold(self):
        with mp.workdps(80):
            for ratio in ("1", "10", "100", "200"):
                result = audit.density_slopes(ratio, dps=80)
                self.assertTrue(result["passed"], ratio)
                if ratio == "200":
                    self.assertFalse(result["pole_exists"])
                    self.assertIsNone(result["pole_slope_formula"])
                else:
                    self.assertTrue(result["pole_exists"])
                    self.assertLess(result["pole_slope_independent_errors"][
                        "h4_formula_vs_independent"], mp.mpf("1e-12"))
            threshold = audit.threshold_precision_check()
            self.assertTrue(threshold["passed"])
            self.assertFalse(threshold["dps80"]["globally_unique_threshold_proven"])
            self.assertLess(threshold["absolute_root_difference"], mp.mpf("1e-35"))

    def test_higher_harmonic_degeneracy_is_explicitly_nonphysical(self):
        result = audit.higher_harmonic_degeneracy()
        self.assertTrue(result["passed"])
        self.assertEqual(result["inferred_F0"], mp.mpf("1.5"))
        self.assertEqual(result["inferred_r"], mp.mpf("3"))
        self.assertTrue(result["is_mathematical_counterexample_only"])

    def test_negative_controls_include_wrong_sign_and_wrong_pole_weights(self):
        with mp.workdps(80):
            moments = audit.spectral_moments(4, 1, dps=80)
            controls = audit.negative_controls(4, 1, dps=80, moments=moments)
            self.assertTrue(controls["wrong_retarded_sign_rejected"])
            self.assertTrue(controls["missing_pole_rejected"])
            self.assertTrue(controls["doubled_pole_rejected"])
            self.assertTrue(controls["tiny_positive_pole_weight_checked_nonzero"])
            self.assertTrue(controls["hankel_identity_passed"])
            self.assertTrue(controls["intentionally_wrong_hankel_rejected"])
            self.assertTrue(controls["intentionally_wrong_quadrature_rejected"])
            self.assertTrue(controls["intentionally_wrong_slope_rejected"])

    def test_common_mode_moment_errors_fail_independent_recurrence_targets(self):
        original = audit._quadrature_form
        for bad_power in (9, -1):
            def wrong_form(F0, r, power, form, *, _bad=bad_power):
                value = original(F0, r, power, form)
                return 2*value if power == _bad else value

            with self.subTest(power=bad_power), mock.patch.object(
                    audit, "_quadrature_form", side_effect=wrong_form):
                result = audit.spectral_moments(4, 1, dps=80)
                self.assertFalse(result["passed"])
                self.assertGreaterEqual(result["target_relative_errors"][bad_power],
                                        audit.MOMENT_TOLERANCE)

    def test_symbolic_and_fine_precision_statuses_are_required_at_aggregation(self):
        good = audit.symbolic_checks()
        missing = dict(good)
        missing.pop(next(iter(missing)))
        failed = copy.deepcopy(good)
        failed[next(iter(failed))]["passed"] = False
        for invalid in ({}, missing, failed):
            self.assertFalse(audit._symbolic_checks_passed(invalid))
        with mock.patch.object(audit, "symbolic_checks", return_value={}):
            self.assertFalse(audit._state("1", dps=80, full=False)["passed"])

        def fake_state(n_ratio, *, dps, full):
            return {
                "passed": dps == 80,
                "coefficients": {key: mp.mpf("1") for key in (
                    "F0", "r", "NF", "B", "C", "V", "first_sound_squared")},
                "moments": {"total": {power: mp.mpf("1")
                                          for power in audit.MOMENT_ORDERS}},
            }

        with mock.patch.object(audit, "_state", side_effect=fake_state), \
                mock.patch.object(audit, "density_slopes", return_value={"passed": True}), \
                mock.patch.object(audit, "threshold_precision_check",
                                  return_value={"passed": True}):
            aggregate = audit.compute_all_controls(dps=80)
        self.assertFalse(aggregate["mathematical_checks_passed"])
        self.assertTrue(all(row["coarse_state_passed"] for row in
                            aggregate["precision_refinement"]))
        self.assertTrue(all(not row["fine_state_passed"] for row in
                            aggregate["precision_refinement"]))

    def test_failed_missing_and_nonfinite_equilibrium_residuals_fail_closed(self):
        original = audit.kinetic.coefficients("1", dps=80)
        for residuals in ({},
                          {**original["residuals"], "C_matches_source_energy": mp.nan},
                          {**original["residuals"], "C_matches_source_energy": True},
                          {**original["residuals"], "C_matches_source_energy": mp.mpf("1e-20")}):
            bad = {**original, "residuals": residuals}
            with mock.patch.object(audit.kinetic, "coefficients", return_value=bad):
                with self.assertRaises(ArithmeticError):
                    audit.equilibrium_coefficients("1", dps=80)

    def test_strict_json_cli_and_invalid_domain(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main(["--n-ratio", "200"])
        result = json.loads(output.getvalue(), parse_constant=lambda value: self.fail(value))
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], audit.STATUS)
        self.assertEqual(result["evidence_weight"], 0)
        self.assertTrue(result["mathematical_checks_passed"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main(["--n-ratio", "0"])
        self.assertEqual(code, 2)
        invalid = json.loads(output.getvalue(), parse_constant=lambda value: self.fail(value))
        self.assertFalse(invalid["mathematical_checks_passed"])
        self.assertEqual(invalid["evidence_weight"], 0)


if __name__ == "__main__":
    unittest.main()
