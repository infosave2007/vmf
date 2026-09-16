"""Independent limits, negative controls and conservation tests for the audit."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import mpmath as mp
import sympy as sp

import bounce_action_derivation_audit as audit


class ActionDerivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checks = audit.symbolic_checks()

    def test_derivative_generated_identities_vanish(self):
        self.assertGreaterEqual(len(self.checks), 25)
        for name, check in self.checks.items():
            self.assertEqual(sp.sympify(check["residual"]), 0, name)
            self.assertTrue(check["passed"], name)

    def test_wrong_phase_kinetic_sign_fails_positive_vector_identity(self):
        wrong = audit.symbolic_checks(phase_sign=-1)
        self.assertFalse(wrong["positive_vector_energy_from_routh"]["passed"])
        self.assertTrue(wrong["gauss_neutrality"]["passed"])
        with patch.object(audit, "symbolic_checks", return_value=wrong):
            with self.assertRaisesRegex(ArithmeticError, "symbolic"):
                audit.build_result()

    def test_altered_gravity_function_is_not_the_requested_bounce_law(self):
        wrong = audit.symbolic_checks(map_deformation=1)
        self.assertFalse(wrong["sine_friedmann_from_hamiltonian"]["passed"])
        self.assertFalse(wrong["sine_raychaudhuri_from_hamiltonian"]["passed"])
        self.assertFalse(wrong["bounce_density_equals_rho_c"]["passed"])
        # A matching small-momentum limit alone cannot certify the bounce.
        self.assertTrue(wrong["GR_low_momentum_coefficient"]["passed"])
        with patch.object(audit, "symbolic_checks", return_value=wrong):
            with self.assertRaises(ArithmeticError):
                audit.build_result()

    def test_fixed_scalar_momentum_pressure_with_independent_powerlaw(self):
        with mp.workdps(70):
            volume, scalar_momentum, baryons, W = map(mp.mpf, ("2.3", "0.7", "1.4", "1.1"))
            energy = lambda n, w: n**(mp.mpf(4)/3)+n*w*w+w**4
            matter_H = lambda v: scalar_momentum**2/(2*v)+v*energy(baryons/v, W)
            kinetic = scalar_momentum**2/(2*volume**2)
            n = baryons/volume
            expected = kinetic+n**(mp.mpf(4)/3)/3-W**4
            derived = -mp.diff(matter_H, volume)
            self.assertLess(abs(derived-expected), mp.mpf("1e-60"))
            # Holding Wdot fixed instead of Pi reverses this kinetic pressure.
            wrong = -kinetic+n**(mp.mpf(4)/3)/3-W**4
            self.assertGreater(abs(derived-wrong), mp.mpf("0.01"))

    def test_generic_maximum_criterion_needs_enthalpy_and_negative_curvature(self):
        with mp.workdps(60):
            for sign, expected_sign in ((-1, 1), (1, -1)):
                function = lambda p: 7+sign*(p-2)**2+(p-2)**4
                curvature = mp.diff(function, 2, 2)
                self.assertEqual(mp.diff(function, 2), 0)
                Hdot = -curvature*3/3
                self.assertEqual(mp.sign(Hdot), expected_sign)
            self.assertEqual(-mp.diff(function, 2, 2)*0/3, 0)

    def test_closed_witness_and_near_vacuum_counterexample(self):
        bounced = audit.source_witness()
        maximum = audit.source_witness(W_ratio="1")
        self.assertLess(mp.mpf(bounced["rho_plus_3P_MeV4"]), 0)
        self.assertEqual(bounced["closed_turning_point"], "bounce")
        self.assertGreater(mp.mpf(maximum["rho_plus_3P_MeV4"]), 0)
        self.assertEqual(maximum["closed_turning_point"], "maximum")
        for row in (bounced, maximum):
            self.assertGreater(mp.mpf(row["rho_plus_P_MeV4"]), 0)
            self.assertFalse(row["flat_H0_allowed"])
            self.assertFalse(row["open_H0_allowed"])
            self.assertTrue(row["closed_H0_allowed"])
            self.assertFalse(row["stationarity_imposed"])
            self.assertLess(mp.mpf(row["independent_integral_max_relative_residual"]), mp.mpf("1e-60"))

    def test_scalar_motion_can_remove_potential_dominated_closed_bounce(self):
        row = audit.source_witness(Wdot="1e7")
        self.assertGreater(mp.mpf(row["rho_plus_3P_MeV4"]), 0)
        self.assertEqual(row["closed_turning_point"], "maximum")

    def test_source_functional_mutation_is_caught_by_independent_integrals(self):
        original = audit.BulkModel.state

        def changed(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            result["pressure_total"] += 1
            return result

        with patch.object(audit.BulkModel, "state", changed):
            with self.assertRaisesRegex(ArithmeticError, "functional"):
                audit.source_witness()

    def test_nonfinite_source_cannot_become_a_nan_string_success(self):
        original = audit.BulkModel.state

        def corrupted(model, *args, **kwargs):
            result = original(model, *args, **kwargs)
            result["energy_total"] = mp.nan
            return result

        with patch.object(audit.BulkModel, "state", corrupted):
            with self.assertRaisesRegex(ValueError, "finite"):
                audit.source_witness()

    def test_reconstructed_sine_equations_and_exact_density_endpoint(self):
        with mp.workdps(90):
            rc, gn, enthalpy = mp.mpf(7), mp.mpf("0.2"), mp.mpf(3)
            alpha = mp.sqrt(6*mp.pi*gn/rc)
            for angle in (mp.mpf("0.1"), mp.pi/3, mp.pi/2, 2*mp.pi/3):
                row = audit.reconstructed_point(mp.nstr(angle/alpha, 90),
                    rho_c="7", G_Newton="0.2", enthalpy="3")
                self.assertLess(abs(mp.mpf(row["friedmann_residual"])), mp.mpf("1e-65"))
                self.assertLess(abs(mp.mpf(row["raychaudhuri_residual"])), mp.mpf("1e-65"))
                if angle == mp.pi/2:
                    self.assertLess(abs(mp.mpf(row["rho"])-rc), mp.mpf("1e-65"))
                    self.assertLess(abs(mp.mpf(row["H"])), mp.mpf("1e-65"))
                    self.assertLess(mp.mpf(row["f_second"]), 0)
                    self.assertLess(abs(mp.mpf(row["Hdot"])/(4*mp.pi*gn*enthalpy)-1), mp.mpf("1e-33"))

    def test_GR_limit_and_both_hubble_signs(self):
        with mp.workdps(80):
            row = audit.reconstructed_point("1e-12")
            self.assertLess(abs(mp.mpf(row["rho"])/(6*mp.pi*mp.mpf("1e-24"))-1), mp.mpf("1e-22"))
            self.assertLess(abs(mp.mpf(row["Hdot"])/(-4*mp.pi)-1), mp.mpf("1e-21"))
            alpha = mp.sqrt(6*mp.pi)
            contraction = audit.reconstructed_point(mp.nstr(mp.pi/(3*alpha), 80))
            expansion = audit.reconstructed_point(mp.nstr(2*mp.pi/(3*alpha), 80))
            self.assertLess(mp.mpf(contraction["H"]), 0)
            self.assertGreater(mp.mpf(expansion["H"]), 0)

    def test_finite_physical_inputs_are_required(self):
        for bad in (mp.nan, mp.inf, -mp.inf, True, "not a number"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    audit.finite_number(bad, "input")
                with self.assertRaises(ValueError):
                    audit.source_witness(n_ratio=bad)
                with self.assertRaises(ValueError):
                    audit.reconstructed_point(bad)
        for kwargs in ({"rho_c": 0}, {"rho_c": -1}, {"G_Newton": 0}, {"enthalpy": -1}):
            with self.assertRaises(ValueError):
                audit.reconstructed_point(1, **kwargs)
        for kwargs in ({"W_ratio": 0}, {"n_ratio": -1}, {"Wdot": mp.nan}):
            with self.assertRaises(ValueError):
                audit.source_witness(**kwargs)
        with self.assertRaises(ValueError):
            audit.decimal(mp.nan)

    def test_short_time_evolves_scalar_and_preserves_scaled_constraints(self):
        coarse = audit.short_time_leading_order(rtol=2e-8, atol=2e-10, max_step=0.15)
        fine = audit.short_time_leading_order()
        self.assertLess(abs(coarse["return_tau"]-fine["return_tau"]), 1e-6)
        self.assertGreater(fine["accepted_steps"], coarse["accepted_steps"])
        self.assertLess(fine["energy_max_relative_drift"], 1e-8)
        self.assertLess(fine["scaled_constraint_max_absolute_residual"], 1e-8)
        self.assertGreater(fine["return_tau"], 1.0)
        self.assertLess(fine["return_tau"], 1.1)
        self.assertNotAlmostEqual(fine["W_over_W0_at_return"], 1.1, places=3)
        self.assertAlmostEqual(fine["R_at_return"], -2*fine["L_at_return"], places=8)
        self.assertLess(mp.mpf(fine["log_a_growth_at_return"]), mp.mpf("1e-39"))
        self.assertGreater(fine["mass_adiabatic_ratio_max"], 0.1)
        self.assertIn("NOT_FULL_EVOLUTION", fine["status"])
        with self.assertRaises(ValueError):
            audit.short_time_leading_order(rtol=mp.nan)

    def test_result_and_cli_have_no_artifact_write_or_physics_promotion(self):
        with patch.object(Path, "write_text", side_effect=AssertionError("unexpected artifact write")):
            result = audit.build_result()
            stream = io.StringIO()
            with patch.object(audit, "build_result", return_value=result), contextlib.redirect_stdout(stream):
                returned = audit.main()
        self.assertEqual(returned, result)
        self.assertEqual(json.loads(stream.getvalue()), result)
        self.assertEqual(result["evidence_weight"], 0)
        self.assertIsNone(result["observed_likelihood"])
        self.assertIn("MATHEMATICAL_DERIVATION", result["status"])
        self.assertEqual(result["reconstructed_gravity"]["status"], "RECONSTRUCTED_GRAVITY_NOT_DERIVED_FROM_NVG")
        self.assertIn("no validation", result["short_time_formal_closure"]["fine"]["limitation"])

    def test_false_symbolic_status_is_not_accepted_without_zero_residual(self):
        wrong = copy.deepcopy(self.checks)
        wrong["dynamic_NEC_sum"] = {"residual": "1", "passed": True}
        with patch.object(audit, "symbolic_checks", return_value=wrong):
            with self.assertRaises(ArithmeticError):
                audit.build_result()


if __name__ == "__main__":
    unittest.main()
