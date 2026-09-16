"""Focused tests for the regular-epsilon scalar perturbation audit."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import unittest

import mpmath as mp

try:
    import nvg_regular_scalar_perturbation_audit as audit
except ModuleNotFoundError:  # unittest invoked from the repository root
    from verification import nvg_regular_scalar_perturbation_audit as audit


class RegularScalarPerturbationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flat_rows = audit.flat_qy_symbolic_checks()
        cls.curved_rows = audit.curved_action_symbolic_checks()
        cls.reconstruction_rows = audit.curved_flat_reconstruction_symbolic_checks()

    def test_flat_qy_polynomials_and_epsilon_zero_frozen_control(self):
        self.assertEqual(set(self.flat_rows), set(audit.REQUIRED_FLAT_ROWS))
        self.assertTrue(audit.rows_pass(self.flat_rows,
                                        audit.REQUIRED_FLAT_ROWS))
        frozen = audit.flat_coefficients(0, mp.mpf("0.37"), mp.mpf("1.2"))
        self.assertEqual(frozen["N"],
                         mp.mpf("0.37") * (1 - mp.mpf("0.37")) / 3
                         * mp.mpf("1.2") ** 2
                         + 2 * mp.mpf("0.37") ** 2
                         * (2 * mp.mpf("0.37") + 1) * mp.mpf("1.2")
                         + mp.mpf("0.37") ** 3
                         * (3 + 7 * mp.mpf("0.37")
                            - 8 * mp.mpf("0.37") ** 2))

    def test_flat_hhddot_sign_is_a_real_negative_control(self):
        wrong = audit.flat_qy_symbolic_checks(h_hddot_sign=1)
        self.assertFalse(wrong["A2_from_general_coefficient"]["passed"])
        self.assertTrue(wrong["A0_from_general_coefficient"]["passed"])
        with self.assertRaises(ValueError):
            audit.flat_qy_symbolic_checks(h_hddot_sign=True)

    def test_analytic_epsilon_regimes_and_explicit_e_gt_one_counterexample(self):
        strict = audit.flat_bound_certificate("0.1")
        self.assertTrue(strict["certificate_passed"])
        self.assertTrue(strict["strict_all_domain"])
        marginal = audit.flat_bound_certificate(1)
        self.assertTrue(marginal["certificate_passed"])
        self.assertFalse(marginal["strict_all_domain"])
        self.assertEqual(marginal["marginal_point"]["cs2"], 0)
        counter = audit.flat_bound_certificate(2)["counterexample"]
        self.assertTrue(counter["negative_cs2"])
        self.assertLess(counter["point"]["cs2"], 0)
        self.assertGreater(counter["K"], 0)
        self.assertLess(counter["K"], counter["threshold_K"])

    def test_two_low_density_paths_do_not_force_finite_epsilon_gr(self):
        paths = audit.low_density_paths("0.1", "1")
        self.assertEqual(paths["fixed_K_positive_limit"], 1)
        self.assertTrue(paths["paths_are_distinct"])
        self.assertTrue(paths["finite_epsilon_proportional_differs_from_bare_GR"])
        self.assertNotEqual(paths["K_proportional_limit"], 1)

    def test_flat_transfer_refines_and_preserves_raw_determinant(self):
        for kappa in ("0.1", "1", "10"):
            with self.subTest(kappa=kappa):
                result = audit.flat_transfer_convergence(kappa)
                self.assertTrue(result["passed"], result)
                self.assertLess(result["fine"]["determinant_error"], 3e-7)
                self.assertLess(result["fine"]["coefficient_formula_max_abs"], 3e-12)
                self.assertNotIn("renormalized", result)

    def test_flat_and_curved_input_guards(self):
        for bad in (True, "nan", "inf", complex(1, 2), None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    audit.flat_coefficients("0.1", "0.5", bad)
        for x, k in ((0, 1), (-1, 1), ("1.01", 1), (1, -1)):
            with self.assertRaises(ValueError):
                audit.flat_coefficients("0.1", x, k)
        with self.assertRaises(ValueError):
            audit.curved_coefficient_point(p=3, C=1)
        with self.assertRaises(ValueError):
            audit.closed_w1_scalar_transfer(1)

    def test_curved_adm_density_derivation_and_controls(self):
        self.assertEqual(set(self.curved_rows), set(audit.REQUIRED_CURVED_ROWS))
        self.assertTrue(audit.rows_pass(self.curved_rows,
                                        audit.REQUIRED_CURVED_ROWS))
        for kwargs, expected_bad in (
                ({"shift_curvature_sign": 1}, "shift_square_curvature_term"),
                ({"cuscuton_gradient_sign": -1}, "cuscuton_spatial_gradient_term"),
                ({"cross_sign": -1}, "cuscuton_potential_gradient_cancellation"),
                ({"cuscuton_hessian_sign": -1}, "cuscuton_potential_curvature_term")):
            with self.subTest(kwargs=kwargs):
                wrong = audit.curved_action_symbolic_checks(**kwargs)
                self.assertFalse(wrong[expected_bad]["passed"])
                self.assertFalse(audit.rows_pass(wrong,
                                                 audit.REQUIRED_CURVED_ROWS))

    def test_curved_schur_reduction_including_h_zero(self):
        self.assertTrue(audit.rows_pass(
            self.reconstruction_rows, audit.REQUIRED_FLAT_RECONSTRUCTION_ROWS))
        for H in ("0.37", "0"):
            point = audit.curved_coefficient_point(H=H)
            self.assertTrue(point["passed"], point)
            self.assertLess(point["max_scaled_residual"], mp.mpf("1e-70"))
            self.assertLess(abs(point["residuals"]["auxiliary_solution"]),
                            mp.mpf("1e-70"))
            self.assertGreater(point["Delta"], 0)
            self.assertLess(point["determinant"], 0)
        certificate = audit.curved_sign_certificate(H=0)
        self.assertTrue(certificate["Delta_positive"])
        self.assertTrue(certificate["A_s_positive"])
        self.assertTrue(certificate["determinant_negative"])
        wrong = audit.curved_flat_reconstruction_symbolic_checks(include_Bdot=False)
        self.assertFalse(wrong["C0_G_includes_pdot_and_Ydot"]["passed"])
        wrong_sign = audit.curved_flat_reconstruction_symbolic_checks(
            h_hddot_sign=1)
        self.assertFalse(wrong_sign["C0_minus_HHddot_sign"]["passed"])

    def test_closed_w1_fixed_harmonics_refine_and_classify(self):
        results = audit.closed_w1_scalar_scan()
        self.assertEqual([row["ell"] for row in results], list(range(2, 13)))
        self.assertTrue(all(row["passed"] for row in results))
        self.assertTrue(all(row["fine"]["finite"] for row in results))
        self.assertTrue(all(row["fine"]["min_A_s"] > 0 for row in results))
        self.assertTrue(all(row["fine"]["min_U"] > 0 for row in results))
        self.assertTrue(all(row["fine"]["classification"]
                            in ("elliptic", "hyperbolic",
                                "parabolic_or_unresolved",
                                "numerically_unresolved")
                            for row in results))
        self.assertTrue(all(row["growth_verification"]["performed"] is False
                            for row in results))

    def test_acceptance_fails_closed_on_mutated_evidence(self):
        result = audit.compute_state(include_transfer=False, include_closed=False)
        self.assertTrue(audit.acceptance_from_result(result))
        bad = copy.deepcopy(result)
        bad["symbolic"]["curved"]["shift_square_curvature_term"] = {
            "residual": "0", "passed": False}
        self.assertFalse(audit.acceptance_from_result(bad))
        bad = copy.deepcopy(result)
        bad["flat_bounds"]["certificate_passed"] = "yes"
        self.assertFalse(audit.acceptance_from_result(bad))
        bad = copy.deepcopy(result)
        bad["curved_points"][0]["max_scaled_residual"] = "nan"
        self.assertFalse(audit.acceptance_from_result(bad))
        with_transfer = audit.compute_state(include_transfer=True, include_closed=False)
        self.assertTrue(audit.acceptance_from_result(with_transfer))
        bad = copy.deepcopy(with_transfer)
        bad["flat_transfers"][0]["fine"]["determinant"] = "2"
        self.assertFalse(audit.acceptance_from_result(bad))
        bad = copy.deepcopy(with_transfer)
        bad["flat_transfers"][0]["fine"]["determinant"] = "1.1"
        bad["flat_transfers"][0]["fine"]["determinant_error"] = "0.1"
        self.assertFalse(audit.acceptance_from_result(bad))

    @staticmethod
    def _refresh_raw_matrix_metadata(record):
        matrix = record["matrix"]
        a0, b0 = matrix[0]
        c0, d0 = matrix[1]
        determinant = a0 * d0 - b0 * c0
        record["determinant"] = determinant
        record["determinant_error"] = abs(determinant - 1.0)
        record["symplectic_error"] = abs(determinant - 1.0)
        if "classification" in record:
            transfer = audit._classify_transfer(matrix, determinant)
            record.update({
                "trace": transfer["trace"],
                "discriminant": transfer["discriminant"],
                "classification": transfer["classification"],
                "eigenvalues_real": transfer["eigenvalues_real"],
                "eigenvalues_imag": transfer["eigenvalues_imag"],
                "classification_tolerance":
                    transfer["classification_tolerance"],
            })

    @staticmethod
    def _matrix_relative_difference(coarse, fine):
        return max(abs(coarse["matrix"][i][j] - fine["matrix"][i][j])
                   / max(1.0, abs(fine["matrix"][i][j]))
                   for i in range(2) for j in range(2))

    def test_acceptance_recomputes_bounds_and_mode_complete_numeric_controls(self):
        full = audit.compute_state()
        self.assertTrue(audit.acceptance_from_result(full))

        # Preserve the determinant exactly with a shear, but make the matched
        # coarse/fine error far beyond the fixed refinement ceiling.
        excessive = copy.deepcopy(full)
        flat = excessive["flat_transfers"][0]
        coarse_matrix = flat["coarse"]["matrix"]
        shear = 0.02
        coarse_matrix[0][1] += shear * coarse_matrix[0][0]
        coarse_matrix[1][1] += shear * coarse_matrix[1][0]
        self._refresh_raw_matrix_metadata(flat["coarse"])
        flat["relative_matrix_difference"] = self._matrix_relative_difference(
            flat["coarse"], flat["fine"])
        flat["passed"] = True
        self.assertGreater(flat["relative_matrix_difference"],
                           audit.FLAT_REFINEMENT_TOL)
        self.assertFalse(audit.acceptance_from_result(excessive))

        # A flat formula mismatch must fail even if all producer flags remain
        # true and the value is finite/nonnegative.
        coefficient_bad = copy.deepcopy(full)
        coefficient_bad["flat_transfers"][0]["fine"][
            "coefficient_formula_max_abs"] = "1e-6"
        self.assertFalse(audit.acceptance_from_result(coefficient_bad))

        # Keep a closed raw determinant internally self-consistent but beyond
        # the fixed ceiling; this must not be rescued by ``passed``.
        determinant_bad = copy.deepcopy(full)
        closed_fine = determinant_bad["closed_w1_transfers"][0]["fine"]
        closed_fine["matrix"][0][0] *= 1.0001
        self._refresh_raw_matrix_metadata(closed_fine)
        self.assertGreater(closed_fine["determinant_error"],
                           audit.CLOSED_DETERMINANT_TOL)
        self.assertFalse(audit.acceptance_from_result(determinant_bad))

        # Every declared closed mode may not be relabeled consistently to a
        # classification that disagrees with its own raw matrix.
        class_bad = copy.deepcopy(full)
        for item in class_bad["closed_w1_transfers"]:
            item["coarse"]["classification"] = "hyperbolic"
            item["fine"]["classification"] = "hyperbolic"
        self.assertFalse(audit.acceptance_from_result(class_bad))

        # Empty and incomplete groups fail when the corresponding input flag
        # still declares the complete finite set.
        empty_bad = copy.deepcopy(full)
        empty_bad["flat_transfers"] = []
        self.assertFalse(audit.acceptance_from_result(empty_bad))
        missing_bad = copy.deepcopy(full)
        missing_bad["closed_w1_transfers"].pop()
        self.assertFalse(audit.acceptance_from_result(missing_bad))

        nan_bad = copy.deepcopy(full)
        nan_bad["closed_w1_transfers"][0]["fine"][
            "max_constraint_residual"] = "nan"
        self.assertFalse(audit.acceptance_from_result(nan_bad))
        negative_bad = copy.deepcopy(full)
        negative_bad["curved_points"][0]["max_scaled_residual"] = "-1"
        self.assertFalse(audit.acceptance_from_result(negative_bad))

        # Empty subsets are only truthful when explicitly requested and
        # explicitly described by scope.
        omitted = audit.compute_state(include_transfer=False,
                                      include_closed=False)
        self.assertTrue(audit.acceptance_from_result(omitted))
        omitted["scope"]["flat_transfer_scope"] = audit.FLAT_TRANSFER_SCOPE
        self.assertFalse(audit.acceptance_from_result(omitted))

    def test_closed_matched_raw_det_mutation_hits_explicit_ceiling(self):
        row = audit.closed_w1_transfer_convergence(2)
        row["growth_verification"] = {"performed": False,
                                      "reason": "elliptic"}
        for record in (row["coarse"], row["fine"]):
            record["matrix"] = [[1.1, 0.0], [0.0, 1.1]]
            self._refresh_raw_matrix_metadata(record)
        row["relative_matrix_difference"] = 0.0
        row["passed"] = True
        self.assertAlmostEqual(row["coarse"]["determinant"], 1.21)
        self.assertAlmostEqual(row["fine"]["determinant"], 1.21)
        self.assertGreater(row["fine"]["determinant_error"],
                           audit.CLOSED_DETERMINANT_TOL)
        self.assertFalse(audit._matrix_record_passes(row["coarse"],
                                                      kind="closed"))
        self.assertFalse(audit._matrix_record_passes(row["fine"],
                                                      kind="closed"))
        self.assertFalse(audit._convergence_record_passes(row,
                                                          kind="closed"))

    def test_cli_is_json_only_and_invalid_inputs_fail_closed(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main(["--skip-transfer", "--skip-closed"])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], audit.STATUS)
        self.assertTrue(result["mathematical_checks_passed"])
        with contextlib.redirect_stdout(io.StringIO()) as invalid:
            self.assertEqual(audit.main(["--epsilon", "nan", "--skip-transfer",
                                         "--skip-closed"]), 2)
        self.assertEqual(json.loads(invalid.getvalue())["status"],
                         "invalid_or_failed_audit")


if __name__ == "__main__":
    unittest.main()
