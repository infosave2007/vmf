"""Independent checks, adverse inputs and deliberately wrong-model controls."""
import contextlib
import io
import json
import unittest
from unittest import mock

import mpmath as mp
import sympy as sp

import covariant_bounce_completion_audit as audit


class CovariantBounceCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.background = audit.symbolic_checks()
        cls.perturbations = audit.perturbation_symbolic_checks()

    def test_variation_and_background_identities(self):
        self.assertGreaterEqual(len(self.background), 25)
        self.assertTrue(all(item["passed"] for item in self.background.values()))

    def test_exact_canonical_bridge(self):
        keys = [key for key in self.background if key.startswith("canonical_bridge_")]
        self.assertEqual(len(keys), 5)
        self.assertTrue(all(self.background[key]["passed"] for key in keys))

    def test_wrong_kinetic_sign_rejected(self):
        wrong = audit.symbolic_checks(kinetic_sign=1)
        self.assertFalse(wrong["field_equation_on_reconstructed_solution"]["passed"])
        self.assertFalse(wrong["raychaudhuri_without_H_division"]["passed"])
        self.assertFalse(wrong["added_field_continuity"]["passed"])

    def test_wrong_potential_rejected(self):
        for scale in (-1, 2):
            with self.subTest(scale=scale):
                wrong = audit.symbolic_checks(potential_scale=scale)
                self.assertFalse(wrong["friedmann_from_lapse_and_parametrization"]["passed"])
                self.assertFalse(wrong["potential_gradient_from_parametrization"]["passed"])

    def test_mu_normalization_is_not_new_physics(self):
        self.assertTrue(self.background["mu_normalization_redundant_without_other_couplings"]["passed"])
        with mp.workdps(60):
            X = mp.mpf("0.37")
            for mu2 in (mp.mpf("1e-20"), mp.mpf("2.3"), mp.mpf("1e20")):
                self.assertAlmostEqual(-mu2*mp.sqrt(2*X/mu2**2), -mp.sqrt(2*X), places=55)

    def test_numeric_independent_derivatives(self):
        with mp.workdps(80):
            for q in (-mp.mpf("1.4"), -mp.pi/4, 0, mp.pi/4, mp.mpf("1.4")):
                result = audit.numerical_derivative_check(q, "2.3", "1.7", "0.8")
                self.assertTrue(result["passed"], result)

    def test_friedmann_and_continuity_at_asymmetric_points(self):
        with mp.workdps(80):
            for q in ("-1.2", "-0.1", "0", "0.7", "1.3"):
                point = audit.parametric_state(q, "7", "3", "2")
                self.assertLess(abs(3*point["Mpl"]**2*point["H"]**2-
                                    point["rho"]-point["V"]), mp.mpf("1e-75"))
                self.assertLess(abs(2*point["Mpl"]**2*point["Hdot"]+
                                    point["enthalpy"]-point["Psidot"]), mp.mpf("1e-75"))

    def test_bounce_has_positive_acceleration_without_H_division(self):
        point = audit.parametric_state(0, 7, 3, 2)
        self.assertEqual(point["H"], 0)
        self.assertEqual(point["rho"]+point["V"], 0)
        self.assertGreater(point["Hdot"], 0)
        self.assertGreater(point["Psidot"], 0)

    def test_both_inflection_folds_are_regular(self):
        with mp.workdps(80):
            for q in (-mp.pi/4, mp.pi/4):
                point = audit.parametric_state(q, "5", "2", "3")
                self.assertLess(abs(point["V_PsiPsi"]), mp.mpf("1e-78"))
                self.assertAlmostEqual(point["constraint_determinant"], 3, places=75)
                self.assertAlmostEqual(point["Psidot"], 3, places=75)
                self.assertGreater(point["Psi_q"], 0)

    def test_monotonic_field_and_safe_inverse(self):
        with mp.workdps(80):
            previous = -mp.inf
            for q in map(mp.mpf, ("-1.5", "-1", "-0.2", "0", "0.3", "1", "1.5")):
                point = audit.parametric_state(q, 3, 2)
                self.assertGreater(point["Psi"], previous)
                previous = point["Psi"]
                inverse = audit.inverse_field(point["Psi"], 3, 2)
                self.assertLess(abs(inverse-q), mp.mpf("1e-74"))

    def test_endpoint_is_C1_not_C2(self):
        with mp.workdps(80):
            last_curvature = 0
            for eps in (mp.mpf("1e-4"), mp.mpf("1e-6"), mp.mpf("1e-8")):
                point = audit.parametric_state(mp.pi/2-eps, 3, 2)
                gap = mp.pi/(2*point["alpha"])-point["Psi"]
                asymptotic = -3*(3*point["alpha"]*gap/2)**(mp.mpf(4)/3)
                self.assertLess(abs(point["V"]/asymptotic-1), 2*eps**2)
                self.assertLess(point["V_PsiPsi"], last_curvature)
                last_curvature = point["V_PsiPsi"]
                self.assertLess(abs(point["V_Psi"]), 2*eps)

    def test_open_domain_endpoints_rejected(self):
        with mp.workdps(60):
            for q in (-mp.pi/2, mp.pi/2, -2, 2):
                with self.assertRaises(ValueError):
                    audit.parametric_state(q)
            alpha = mp.sqrt(mp.mpf(3)/4)
            for Psi in (-mp.pi/(2*alpha), mp.pi/(2*alpha), 10):
                with self.assertRaises(ValueError):
                    audit.inverse_field(Psi)

    def test_nonfinite_and_boolean_guards(self):
        for value in (True, False, "nan", "inf", "-inf", complex(1, 1), None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    audit.parametric_state(value)
                with self.assertRaises(ValueError):
                    audit.parametric_state(0, enthalpy=value)
                with self.assertRaises(ValueError):
                    audit.scalar_massless_coefficients("0.5", value)
                with self.assertRaises(ValueError):
                    audit.inverse_field(value)

    def test_positive_inputs_and_perturbation_domain(self):
        for name in ("rho_c", "Mpl", "enthalpy"):
            for bad in (0, -1):
                with self.assertRaises(ValueError):
                    audit.parametric_state(0, **{name: bad})
        for x, K in ((0, 1), (-1, 1), ("1.01", 1), (1, -1)):
            with self.assertRaises(ValueError):
                audit.scalar_massless_coefficients(x, K)
        with self.assertRaises(ValueError):
            audit.symbolic_checks(kinetic_sign=True)
        with self.assertRaises(ValueError):
            audit.perturbation_symbolic_checks(h_hddot_sign=True)

    def test_perturbation_coefficients_derived_from_primary_source(self):
        self.assertTrue(all(item["passed"] for item in self.perturbations.values()))

    def test_wrong_HHddot_sign_fails_physical_GR_limit(self):
        wrong = audit.perturbation_symbolic_checks(h_hddot_sign=1)
        self.assertFalse(wrong["A2_from_source_eq43"]["passed"])
        self.assertFalse(wrong["physical_GR_limit_K_proportional_x"]["passed"])
        expr = sp.sympify(wrong["physical_GR_limit_K_proportional_x"]["residual"])
        self.assertNotEqual(sp.simplify(expr.subs(next(iter(expr.free_symbols)), 1)), 0)

    def test_sound_bounds_and_kinetic_positive_grid(self):
        with mp.workdps(80):
            for x in map(mp.mpf, ("1e-20", "0.01", "0.2", "0.5", "0.9", "1")):
                for K in map(mp.mpf, ("0", "0.01", "1", "10", "1e20")):
                    point = audit.scalar_massless_coefficients(x, K)
                    self.assertGreater(point["kinetic"], 0)
                    self.assertGreaterEqual(point["cs2"], mp.mpf(1)/3)
                    self.assertLess(point["cs2"], 3)

    def test_bounce_superluminality_is_reported_not_clamped(self):
        for K in (0, 1, 2, 100):
            point = audit.scalar_massless_coefficients(1, K)
            self.assertAlmostEqual(float(point["cs2"]), (3*K+1)/(K+3))
            self.assertEqual(point["superluminal_phase_speed"], K > 1)

    def test_numeric_GR_limit_preserves_physical_k_squared_over_density(self):
        with mp.workdps(60):
            for ratio in map(mp.mpf, ("0.1", "1", "10")):
                x = mp.mpf("1e-15")
                value = audit.scalar_massless_coefficients(x, ratio*x)["cs2"]
                self.assertLess(abs(value-1), 10*x)

    def test_zero_mode_exact_transfer(self):
        result = audit.scalar_massless_transfer(0)
        T = result["matrix"]
        self.assertEqual(T[0][0], 1)
        self.assertEqual(T[1][0], 0)
        self.assertEqual(T[1][1], 1)
        self.assertEqual(result["determinant_error"], 0)
        # Independent integral: 1/b=(tau^2+2)/(3*(1+tau^2)^(3/2)).
        expected = 2*(mp.asinh(5)+5/mp.sqrt(26))/3
        self.assertAlmostEqual(T[0][1], float(expected), places=10)

    def test_finite_transfer_through_bounce_and_folds_converges(self):
        for kappa in ("0.1", 1, 10):
            result = audit.transfer_convergence(kappa)
            self.assertTrue(result["passed"], result)
            self.assertLess(result["fine"]["determinant_error"], 1e-7)

    def test_transfer_extremes_fail_closed(self):
        for kwargs in ({"kappa": True}, {"kappa": "1e1000"}, {"kappa": -1},
                       {"kappa": 1, "rtol": 0}, {"kappa": 1, "rtol": "1e-1000"},
                       {"kappa": 1, "tau_start": 2, "tau_end": 1}):
            with self.assertRaises(ValueError):
                audit.scalar_massless_transfer(**kwargs)

    def test_pure_cli_json_and_honest_scope(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = audit.main(["--skip-transfer"])
        result = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(result["status"], audit.STATUS)
        self.assertEqual(result["evidentiary_weight"], 0)
        self.assertTrue(result["mathematical_checks_passed"])
        self.assertIn("not_derived", result["scope"]["extension"])
        self.assertIn("not_C2", result["scope"]["vacuum_endpoints"])
        self.assertIn("no_UV_causality", result["scope"]["causality"])
        with contextlib.redirect_stdout(io.StringIO()) as invalid:
            self.assertEqual(audit.main(["--q", "nan", "--skip-transfer"]), 2)
        self.assertEqual(json.loads(invalid.getvalue())["status"], "invalid_or_failed_audit")

    def test_cli_fails_closed_for_empty_or_contradictory_checks(self):
        for checks in ({}, {"bad": {"passed": True, "residual": "1"}},
                       {"bad": {"passed": "yes", "residual": "0"}},
                       {"bad": {"passed": True, "residual": 0}}):
            with self.subTest(checks=checks):
                with mock.patch.object(audit, "symbolic_checks", return_value=checks):
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(audit.main(["--skip-transfer"]), 1)
                self.assertIs(json.loads(output.getvalue())["mathematical_checks_passed"], False)
        with mock.patch.object(audit, "numerical_derivative_check", return_value={"passed": "yes"}):
            result = audit.compute_state(include_transfer=False)
        self.assertIs(result["mathematical_checks_passed"], False)
        with mock.patch.object(audit, "transfer_convergence", return_value={"passed": 1}):
            result = audit.compute_state(include_transfer=True)
        self.assertIs(result["mathematical_checks_passed"], False)


if __name__ == "__main__":
    unittest.main()
