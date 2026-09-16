"""Action-to-spectrum checks with adverse coefficients and strict scope gates."""
import contextlib
import copy
import io
import json
import unittest
from unittest import mock

import mpmath as mp
import sympy as sp

import nvg_longitudinal_fluid_audit as audit


class LongitudinalFluidTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.point = audit.coefficients("1")
        cls.report = audit.compute_state()

    def test_complete_action_derived_symbolic_checks(self):
        checks = audit.symbolic_checks()
        self.assertEqual(set(checks), audit.EXPECTED_SYMBOLIC_CHECKS)
        self.assertEqual(len(checks), 22)
        self.assertTrue(audit._symbolic_passed(checks))

    def test_wrong_scalar_vector_mixing_fails_independent_identities(self):
        checks = audit.symbolic_checks(mixing_scale=0)
        self.assertFalse(checks["euler_matrix_from_quadratic_action"]["passed"])
        self.assertFalse(checks["zero_k_mixed_hessian"]["passed"])
        self.assertFalse(checks["zero_k_scalar_hessian"]["passed"])
        self.assertFalse(audit._symbolic_passed(checks))

    def test_wrong_fixed_A0_scalar_curvature_fails(self):
        checks = audit.symbolic_checks(scalar_curvature_sign=1)
        self.assertFalse(checks["zero_k_scalar_hessian"]["passed"])
        self.assertFalse(audit._symbolic_passed(checks))

    def test_eight_original_density_passports_match_independent_integrals(self):
        points = self.report["density_points"]
        self.assertEqual(len(points), 8)
        self.assertEqual(tuple(mp.mpf(p["n_ratio"]) for p in points),
                         tuple(mp.mpf(v) for v in audit.DENSITY_RATIOS))
        for point in points:
            self.assertIs(point["independent_checks_passed"], True)
            self.assertIs(point["positive_energy"]["strict_positive_all_z"], True)
            self.assertTrue(all(mp.mpf(e) < mp.mpf("1e-60") for e in point["independent_errors"].values()))

    def test_80_120_agree_without_changing_input_constants(self):
        self.assertLess(mp.mpf(self.report["coefficient_80_vs_120_max_relative_difference"]), mp.mpf("1e-60"))
        self.assertEqual(self.report["original_inputs"], audit.INPUTS)
        with mp.workdps(80):
            self.assertEqual(self.point["coefficients"]["g"], mp.mpf("10.12"))

    def test_acoustic_derivative_is_relaxed_sound_not_kinetic_mu(self):
        with mp.workdps(80):
            c = self.point["coefficients"]
            K = audit.static_hessian(c, 0)
            numerator = c["n"]*(K[0][0]-K[0][1]**2/K[1][1])
            correct = numerator/self.point["mu_total"]
            wrong = numerator/c["mu_F"]
            self.assertLess(abs(correct-self.point["acoustic_cs2"]), mp.mpf("1e-70"))
            self.assertGreater(abs(wrong-correct), mp.mpf("1e-3"))

    def test_static_hessian_polynomial_identity_at_several_z(self):
        with mp.workdps(80):
            c = self.point["coefficients"]
            certificate = audit.positivity_certificate(c)
            for ratio in (0, mp.mpf("0.01"), 1, 100):
                z = ratio*c["M2"]
                K = audit.static_hessian(c, z)
                actual = (z+c["M2"])*(K[0][0]*K[1][1]-K[0][1]**2)
                expected = c["a"]*z*z+certificate["P1"]*z+certificate["P0"]
                self.assertLess(abs(actual/expected-1), mp.mpf("1e-70"))

    def test_all_z_vertex_criterion_and_marginal_case(self):
        c = {"a": 1, "b": 1, "c": 2, "g": 1, "M2": 1, "n": 1, "mu_F": 1, "s": -3}
        certificate = audit.positivity_certificate(c)
        self.assertEqual(certificate["minimum_at_z"], 1)
        self.assertEqual(certificate["minimum_value"], 0)
        self.assertEqual(certificate["status"], "MARGINAL")
        self.assertIs(certificate["strict_positive_all_z"], False)
        for scalar in (-2, 1):
            self.assertIs(audit.positivity_certificate({**c, "s": scalar})["strict_positive_all_z"], True)

    def test_given_counterexample_has_positive_K0_but_negative_finite_wave_root(self):
        result = audit.finite_wave_counterexample()
        with mp.workdps(80):
            K = result["K0"]
            self.assertGreater(K[0][0], 0)
            self.assertGreater(K[0][0]*K[1][1]-K[0][1]**2, 0)
            self.assertIs(result["positive_energy"]["strict_positive_all_z"], False)
            self.assertAlmostEqual(result["positive_energy"]["P1"], -mp.mpf(49)/20, places=70)
            self.assertAlmostEqual(result["positive_energy"]["P0"], mp.mpf(1)/10, places=70)
            finite = result["finite_k"]
            self.assertEqual(finite["root_status"], "NEGATIVE_OMEGA_INSTABILITY")
            self.assertIs(finite["spectral_comparison_passed"], True)
            self.assertLess(mp.re(finite["Omega_roots_raw"][0]), -mp.mpf("0.24"))
            zero = result["zero_k"]["Omega_roots_raw"]
            self.assertEqual(zero[0], 0)
            self.assertAlmostEqual(mp.re(zero[1]), mp.mpf(11)/20, places=70)
            self.assertEqual(zero[2], 2)

    def test_zero_k_keeps_all_three_roots_including_zero(self):
        result = audit.dispersion_roots(self.point["coefficients"], 0)
        self.assertEqual(len(result["Omega_roots_raw"]), 3)
        self.assertEqual(result["Omega_roots_raw"][0], 0)
        self.assertEqual(result["root_status"], "MARGINAL_OR_UNRESOLVED_NEAR_ZERO")
        self.assertIs(result["spectral_comparison_passed"], True)

    def test_decoupled_scalar_vector_and_acoustic_modes(self):
        c = {"a": 1, "b": 0, "c": 0, "g": 0, "M2": 3, "n": 1, "mu_F": 1, "s": 2}
        result = audit.dispersion_roots(c, 1)
        with mp.workdps(80):
            for root, expected in zip(result["Omega_roots_raw"], (1, 3, 4)):
                self.assertLess(abs(root-expected), mp.mpf("1e-65"))
        self.assertIs(result["spectral_comparison_passed"], True)

    def test_exact_static_zero_mode_satisfies_original_euler_equations(self):
        model = audit._symbolic_model()
        replacements = dict(zip(model["symbols"], (1, 1, 2, 1, 1, 1, 1, -3)))
        replacements.update(dict(zip(model["amplitudes"], (0, 1, -1, 0))))
        replacements.update({model["omega"]: 0, model["k"]: 1})
        self.assertTrue(all(sp.simplify(eq.subs(replacements)) == 0 for eq in model["original_eom"]))
        c = dict(zip(audit.COEFFICIENT_NAMES, (1, 1, 2, 1, 1, 1, 1, -3)))
        result = audit.dispersion_roots(c, 1)
        self.assertEqual(result["root_status"], "MARGINAL_OR_UNRESOLVED_NEAR_ZERO")
        self.assertEqual(len(result["Omega_roots_raw"]), 3)

    def test_raw_complex_roots_are_preserved_and_rejected_as_positive(self):
        fake_roots = [mp.mpc(-1, "0.2"), mp.mpc(1, "-0.3"), mp.mpc(2, "0.1")]
        with mock.patch.object(audit.mp, "polyroots", return_value=fake_roots):
            result = audit.dispersion_roots(self.point["coefficients"], 1)
        self.assertTrue(any(mp.im(root) != 0 for root in result["Omega_roots_raw"]))
        self.assertEqual(result["root_status"], "COMPLEX_OMEGA")
        self.assertIs(result["spectral_comparison_passed"], False)

    def test_nine_spectra_match_independent_six_dimensional_generator(self):
        modes = self.report["representative_mode_checks"]
        self.assertEqual(len(modes), 9)
        for mode in modes:
            self.assertEqual(len(mode["generator_Omega_raw"]), 6)
            self.assertEqual(len(mode["Omega_roots_raw"]), 3)
            self.assertIs(mode["spectral_comparison_passed"], True)
            self.assertLess(mp.mpf(mode["generator_max_relative_difference"]), mp.mpf("1e-45"))
            self.assertEqual(mode["root_status"], "POSITIVE_REAL_WITHIN_NUMERIC_RESOLUTION")

    def test_global_certificate_uses_theorem_not_density_grid(self):
        with mock.patch.object(audit, "coefficients", side_effect=AssertionError("grid is not proof")):
            result = audit.global_energy_certificate()
        self.assertIs(result["all_positive_density_all_wave_number_certified"], True)
        self.assertGreater(result["Cv_over_Cs"], mp.mpf(5)/4)
        self.assertGreater(result["P1_strict_lower_bound"], 100)
        self.assertGreater(result["uniform_scalar_gap_squared_lower_bound"], 0)
        self.assertIn("NOT_a_finite_k_group_velocity_bound", result["principal"])

    def test_global_certificate_requires_prior_named_proof_and_premises(self):
        with mock.patch.object(audit.branch_certificate, "symbolic_certificates", return_value={"unrelated": True}):
            with self.assertRaises(ArithmeticError):
                audit.global_energy_certificate()
        original = audit.branch_certificate.sufficient_bounds
        def failed_premise(*args):
            result = original(*args)
            result["zero_to_one_cs2_certified"] = False
            return result
        with mock.patch.object(audit.branch_certificate, "sufficient_bounds", side_effect=failed_premise):
            self.assertIs(audit.global_energy_certificate()["all_positive_density_all_wave_number_certified"], False)

    def test_principal_characteristic_speeds_are_not_group_velocity_claim(self):
        for point in self.report["density_points"]:
            speeds = list(map(mp.mpf, point["principal_squared_speeds"]))
            self.assertTrue(0 < speeds[0] < mp.mpf(1)/3)
            self.assertEqual(speeds[1:], [1, 1])
        self.assertTrue(audit.symbolic_checks()["transverse_euler_roots"]["passed"])
        self.assertTrue(audit.symbolic_checks()["transverse_positive_hamiltonian"]["passed"])

    def test_invalid_density_precision_and_coefficient_domains(self):
        for bad in (True, False, None, "nan", "inf", "-inf", complex(1, 1), 0, -1):
            with self.assertRaises(ValueError):
                audit.coefficients(bad)
        for dps in (True, 15, 201, 80.0):
            with self.assertRaises(ValueError):
                audit.coefficients("1", dps=dps)
        for key in audit.COEFFICIENT_NAMES:
            for bad in (True, "nan", "inf", complex(1, 1)):
                with self.assertRaises(ValueError):
                    audit.positivity_certificate({**self.point["coefficients"], key: bad})
        for key in ("a", "n", "mu_F", "M2"):
            with self.assertRaises(ValueError):
                audit.static_hessian({**self.point["coefficients"], key: 0}, 1)
        with self.assertRaises(ValueError):
            audit.positivity_certificate({})

    def test_invalid_wave_numbers_and_negative_control_switches(self):
        for bad in (True, "nan", "inf", -1, complex(0, 1)):
            with self.assertRaises(ValueError):
                audit.dispersion_roots(self.point["coefficients"], bad)
        for kwargs in ({"mixing_scale": True}, {"mixing_scale": 2}, {"scalar_curvature_sign": True}):
            with self.assertRaises(ValueError):
                audit.symbolic_checks(**kwargs)

    def test_solver_failure_and_nonfinite_roots_fail_closed(self):
        with mock.patch.object(audit.mp, "polyroots", side_effect=mp.libmp.libhyper.NoConvergence("injected")):
            with self.assertRaises(ArithmeticError):
                audit.dispersion_roots(self.point["coefficients"], 1)
        with mock.patch.object(audit.mp, "polyroots", return_value=[mp.inf, 1, 2]):
            with self.assertRaises(ValueError):
                audit.dispersion_roots(self.point["coefficients"], 1)

    def test_symbolic_passport_requires_exact_names_true_and_zero(self):
        good = audit.symbolic_checks()
        missing = dict(good)
        missing.pop(next(iter(missing)))
        wrong_value = dict(good)
        wrong_value[next(iter(wrong_value))] = {"passed": "yes", "residual": "0"}
        nonzero = dict(good)
        nonzero[next(iter(nonzero))] = {"passed": True, "residual": "1"}
        for bad in ({}, missing, {"unrelated": {"passed": True, "residual": "0"}},
                    {**good, "extra": {"passed": True, "residual": "0"}}, wrong_value, nonzero):
            self.assertFalse(audit._symbolic_passed(bad))

    def test_numeric_passport_rejects_missing_duplicate_and_truthy_rows(self):
        points, modes = self.report["density_points"], self.report["representative_mode_checks"]
        self.assertTrue(audit._numeric_passport_passed(points, modes))
        for ps, ms in (([], modes), (points, []), (points[:-1], modes),
                       (points, modes[:-1]), ([points[0]]*8, modes), (points, [modes[0]]*9)):
            self.assertFalse(audit._numeric_passport_passed(ps, ms))
        corrupted = copy.deepcopy(points)
        corrupted[0]["independent_checks_passed"] = "yes"
        self.assertFalse(audit._numeric_passport_passed(corrupted, modes))
        corrupted = copy.deepcopy(points)
        corrupted[0]["independent_errors"] = {}
        self.assertFalse(audit._numeric_passport_passed(corrupted, modes))
        with mock.patch.object(audit, "DENSITY_RATIOS", ()):
            self.assertIs(audit.compute_state()["mathematical_checks_passed"], False)

    def test_cli_json_strict_status_and_failure_code(self):
        with mock.patch.object(audit, "compute_state", return_value=self.report):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(audit.main(), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], audit.STATUS)
        self.assertEqual(result["evidentiary_weight"], 0)
        self.assertIn("NOT_collisionless", result["scope"]["closure"])
        with mock.patch.object(audit, "symbolic_checks", return_value={}):
            invalid = audit.compute_state()
        self.assertIs(invalid["mathematical_checks_passed"], False)
        with mock.patch.object(audit, "compute_state", return_value=invalid):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(audit.main(), 1)

    def test_cli_rejects_truthy_success_and_returns_json_on_exception(self):
        with mock.patch.object(audit, "compute_state", return_value={"mathematical_checks_passed": "yes"}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(audit.main(), 1)
        self.assertIs(json.loads(output.getvalue())["mathematical_checks_passed"], False)
        with mock.patch.object(audit, "compute_state", side_effect=ArithmeticError("injected")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(audit.main(), 2)
        self.assertIs(json.loads(output.getvalue())["mathematical_checks_passed"], False)


if __name__ == "__main__":
    unittest.main()
