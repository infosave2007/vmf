"""Fixed-input tracking-jet checks: identities are not trajectory validation."""
import contextlib
import io
import json
import unittest
from unittest import mock

import mpmath as mp

import nvg_bounce_tracking_audit as audit
from source_complete_scaling_saturation_audit import BulkModel, INPUTS


class NvgBounceTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bounce = audit.state("1", "1")
        cls.fold = audit.state("1", "0.5")
        cls.general = audit.state("10", "0.3")

    def test_symbolic_chain_rule_stress_and_kinetic_completion(self):
        checks = audit.symbolic_checks()
        self.assertEqual(len(checks), 10)
        self.assertEqual(set(checks), audit.EXPECTED_SYMBOLIC_CHECKS)
        self.assertTrue(audit._symbolic_passed(checks))

    def test_missing_extra_or_unrelated_symbolic_checks_cannot_succeed(self):
        complete = audit.symbolic_checks()
        missing = dict(complete)
        missing.pop("stationary_trial_stress_completion")
        unrelated = {"unrelated": {"passed": True, "residual": "0"}}
        extra = {**complete, **unrelated}
        for wrong in (missing, unrelated, extra):
            self.assertFalse(audit._symbolic_passed(wrong))
            with mock.patch.object(audit, "symbolic_checks", return_value=wrong):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(audit.main([]), 1)
            self.assertIs(json.loads(output.getvalue())["mathematical_checks_passed"], False)

    def test_wrong_residual_sign_is_detected(self):
        checks = audit.symbolic_checks(residual_hdot_sign=1)
        self.assertFalse(checks["stationary_trial_residual_from_chain_rule"]["passed"])
        self.assertFalse(checks["stationary_trial_stress_completion"]["passed"])

    def test_wrong_correction_sign_is_detected(self):
        checks = audit.symbolic_checks(correction_sign=1)
        self.assertFalse(checks["leading_algebraic_correction_cancels_residual"]["passed"])
        self.assertFalse(checks["bounce_leading_correction"]["passed"])

    def test_wrong_kinetic_sign_is_detected(self):
        checks = audit.symbolic_checks(kinetic_sign=-1)
        self.assertFalse(checks["kinetic_completed_friedmann_constraint"]["passed"])
        self.assertFalse(checks["kinetic_completed_fraction"]["passed"])

    def test_fixed_original_inputs_and_physical_planck_scale(self):
        with mp.workdps(80):
            s = self.bounce["numeric"]
            self.assertEqual(s["G_Newton"], mp.mpf(audit.G_NEWTON_MEV_MINUS2))
            self.assertTrue(mp.mpf("2.8e18") < s["Mpl_over_W0"] < mp.mpf("2.9e18"))
        original = dict(INPUTS)
        audit.state("0.1", "0.2")
        self.assertEqual(INPUTS, original)

    def test_independent_equilibrium_derivatives_and_quadrature(self):
        for point in (self.bounce, self.fold, self.general, audit.state("1000", "0.2")):
            result = point["independent_derivatives"]
            self.assertIs(result["passed"], True)
            for key, value in result.items():
                if key != "passed":
                    self.assertLess(value, mp.mpf("1e-60"))

    def test_implicit_third_derivatives_against_energy_differentiation(self):
        with mp.workdps(80):
            model = BulkModel()
            s = self.general["numeric"]
            n, W = s["n"], s["W"]
            energy = lambda density, field: model.state(density, field/model.W0)["energy_total"]
            numerical = {
                "B": mp.diff(energy, (n, W), (1, 1)),
                "C": mp.diff(energy, (n, W), (0, 2)),
                "E_Wnn": mp.diff(energy, (n, W), (2, 1)),
                "E_WWn": mp.diff(energy, (n, W), (1, 2)),
                "E_WWW": mp.diff(energy, (n, W), (0, 3)),
            }
            for key, value in numerical.items():
                self.assertLess(abs(value-s[key])/max(abs(value), mp.mpf("1e-30")), mp.mpf("1e-65"))

    def test_exact_pointwise_gravity_and_trial_kinetic_energy(self):
        with mp.workdps(80):
            for point in (self.bounce, self.fold, self.general):
                s = point["numeric"]
                M2, H2 = s["Mpl"]**2, s["H_squared"]
                self.assertLess(abs(3*M2*H2-s["rho_trial"]*(1-s["x"]))/s["E"], mp.mpf("1e-70"))
                self.assertLess(abs(2*M2*s["Hdot_assigned_Raychaudhuri"]+
                                    s["enthalpy_trial"]*(1-2*s["x"]))/s["E"], mp.mpf("1e-70"))
                self.assertLess(abs(s["Ktrack"]-s["Wdot_trial"]**2/2)/s["E"], mp.mpf("1e-70"))
                expected = s["eta"]*(1-s["x"])/(1-s["eta"]*(1-s["x"]))
                self.assertLess(abs(s["Ktrack_over_E"]-expected), mp.mpf("1e-70"))

    def test_bounce_zero_velocity_does_not_make_scalar_exact(self):
        s = self.bounce["numeric"]
        self.assertEqual(s["H"], 0)
        self.assertEqual(s["Wdot_trial"], 0)
        self.assertEqual(s["stress_continuity_defect"], 0)
        self.assertNotEqual(s["R_exact_trial_defect"], 0)
        self.assertGreater(s["Hdot_assigned_Raychaudhuri"], 0)
        self.assertTrue(mp.mpf("1e-44") < s["abs_R_over_CW"] < mp.mpf("1e-42"))

    def test_fold_is_regular_but_scalar_and_stress_defects_remain(self):
        s = self.fold["numeric"]
        self.assertEqual(s["Hdot_assigned_Raychaudhuri"], 0)
        self.assertNotEqual(s["R_exact_trial_defect"], 0)
        self.assertNotEqual(s["stress_continuity_defect"], 0)
        self.assertEqual(s["friedmann_constraint_time_defect"], 0)

    def test_completion_residual_by_independent_time_jet(self):
        with mp.workdps(80):
            s = self.general["numeric"]
            n, H, hd = s["n"], s["H"], s["Hdot_assigned_Raychaudhuri"]
            ndot, nddot = -3*H*n, -3*hd*n+9*H*H*n
            acceleration = s["Wnn"]*ndot**2+s["Wn"]*nddot
            residual = acceleration+3*H*s["Wdot_trial"]
            self.assertLess(abs(residual/s["R_exact_trial_defect"]-1), mp.mpf("1e-70"))
            rho_dot = s["Wdot_trial"]*acceleration+s["mu"]*ndot
            stress = rho_dot+3*H*s["enthalpy_trial"]
            # Individual continuity terms are 40 orders larger than the
            # defect: high precision is essential, not a float-zero test.
            self.assertLess(abs(stress-s["stress_continuity_defect"])/
                            abs(s["stress_continuity_defect"]), mp.mpf("1e-30"))

    def test_uniform_bounds_cover_x_endpoints_and_interior(self):
        with mp.workdps(80):
            for x in ("0", "0.01", "0.25", "0.5", "0.75", "0.99", "1"):
                point = audit.state("10", x)
                s, b = point["numeric"], point["uniform_x_bounds"]
                for value, upper in ((s["H_squared"], b["H_squared_max"]),
                                     (s["Ktrack"], b["Ktrack_max"]),
                                     (abs(s["Hdot_assigned_Raychaudhuri"]), b["abs_Hdot_max"]),
                                     (abs(s["R_exact_trial_defect"]), b["abs_R_max"])):
                    self.assertLessEqual(value, upper*(1+mp.mpf("1e-70")))

    def test_x_zero_is_labelled_GR_limit_not_finite_rho_c(self):
        point = audit.state("1", "0")
        self.assertIsNone(point["rho_c_conditional_input"])
        self.assertEqual(point["rho_c_status"], "GR_limit_not_finite_rho_c")
        self.assertIn("NOT_NVG_prediction", self.bounce["rho_c_status"])

    def test_80_120_precision_agreement_and_output_digits(self):
        with mp.workdps(120):
            result = audit.compute_state("10", "0.3")
            self.assertIs(result["precision_check"]["passed"], True)
            self.assertLess(mp.mpf(result["precision_check"]["max_relative_difference"]), mp.mpf("1e-55"))
        # Serialization occurs in ambient precision but must retain the
        # original high-precision decimal passport, not round through float.
        result = audit.compute_state("1", "1")
        with mp.workdps(80):
            shown = mp.mpf(result["point"]["numeric"]["G_Newton"])
            self.assertEqual(shown, mp.mpf(audit.G_NEWTON_MEV_MINUS2))

    def test_invalid_density_x_and_precision_fail_closed(self):
        for bad in (True, False, None, "nan", "inf", "-inf", complex(1, 1), 0, -1):
            with self.assertRaises(ValueError):
                audit.state(bad, "0.5")
        for bad in (True, None, "nan", "inf", -0.1, 1.1):
            with self.assertRaises(ValueError):
                audit.state("1", bad)
        for bad in (True, 20, 201, 80.0):
            with self.assertRaises(ValueError):
                audit.state(dps=bad)
        with self.assertRaises(ValueError):
            audit.symbolic_checks(kinetic_sign=True)

    def test_nonpositive_gap_and_bad_kinetic_denominator_fail_closed(self):
        with mock.patch.object(audit.BulkModel, "equilibrium", return_value={"y": 1, "C_y": -1}):
            with self.assertRaises(ValueError):
                audit.state()
        original = audit._stationary_jet
        def large_slope(model, n):
            jet = original(model, n)
            jet["Wn"] = mp.mpf("1e50")
            return jet
        with mock.patch.object(audit, "_stationary_jet", side_effect=large_slope), \
             mock.patch.object(audit, "_independent_jet_check", return_value={"passed": True}):
            with self.assertRaises(ValueError):
                audit.state("1", "0.5")

    def test_cli_json_scope_and_no_static_success(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(audit.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], audit.STATUS)
        self.assertEqual(result["evidentiary_weight"], 0)
        self.assertIn("NOT_validated_error_bound", result["scope"]["correction"])
        self.assertIn("initial_free_oscillations", result["scope"]["tracking_conditions"])
        for wrong in ({}, {"bad": {"passed": "yes", "residual": "0"}},
                      {"bad": {"passed": True, "residual": "1"}}):
            with mock.patch.object(audit, "symbolic_checks", return_value=wrong):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(audit.main([]), 1)
            self.assertIs(json.loads(output.getvalue())["mathematical_checks_passed"], False)

    def test_failed_independent_derivative_check_is_refused(self):
        with mock.patch.object(audit, "_independent_jet_check", return_value={"passed": "yes"}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(audit.main([]), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "INVALID_OR_FAILED_TRACKING_AUDIT")


if __name__ == "__main__":
    unittest.main()
