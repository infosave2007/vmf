"""Independent checks for the potential/curved-action audit."""
import contextlib
import io
import json
import unittest

import mpmath as mp

import nvg_bounce_potential_audit as audit


class BouncePotentialAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.potential = audit.potential_symbolic_checks()
        cls.flat = audit.flat_symbolic_checks()
        cls.curved = audit.curved_action_symbolic_checks()
        cls.endpoint = audit.endpoint_regularization_checks()
        cls.gr = audit.flat_gr_identity_checks()
        cls.actual = audit.actual_binding_design_certificate()

    def test_manufactured_polynomial_fixture_is_nonnegative_and_coercive(self):
        self.assertTrue(all(row["passed"] for row in self.potential.values()))
        with mp.workdps(80):
            for y in map(mp.mpf, ("0.01", "0.5", "1", "1.4", "10")):
                values = audit.coercive_potential_derivatives(y, "2.3")
                self.assertGreaterEqual(values["U"], 0)
                self.assertTrue(all(mp.isfinite(values[key])
                                    for key in ("U", "Uy", "Uyy")))
            self.assertAlmostEqual(float(audit.coercive_potential(mp.sqrt(2), "2")), 2.0)

    def test_old_polynomial_is_explicitly_manufactured_and_not_actual(self):
        fixture = audit.manufactured_polynomial_fixture(beta="192501197.440766144072560254781",
                                                        y=".75")
        self.assertTrue(fixture["manufactured_polynomial_fixture"])
        self.assertFalse(fixture["used_for_actual_acceptance"])
        actual = self.actual["off_equilibrium_force_samples"][1]["potential"]
        self.assertNotAlmostEqual(float(actual), float(fixture["U"]), places=3)
        self.assertTrue(self.actual["scope"]["actual_potential_used"])

    def test_flat_constraint_uses_full_EW(self):
        self.assertGreaterEqual(len(self.flat), 11)
        self.assertTrue(all(row["passed"] for row in self.flat.values()))
        self.assertNotIn("dynamic_enthalpy", self.flat)
        self.assertNotIn("fermi_legendre_enthalpy", self.flat)

    def test_wrong_scalar_force_and_q_rate_are_rejected(self):
        wrong_force = audit.flat_symbolic_checks(scalar_force_sign=-1)
        self.assertFalse(wrong_force["flat_constraint_propagation_full_EW"]["passed"])
        self.assertFalse(wrong_force["full_force_cancellation"]["passed"])
        wrong_rate = audit.flat_symbolic_checks(q_rate_scale=2)
        self.assertFalse(wrong_rate["flat_constraint_propagation_full_EW"]["passed"])
        self.assertFalse(wrong_rate["flat_raychaudhuri_from_regular_q"]["passed"])

    def test_fixed_binding_design_bounds_force_and_positive_fermi_integrals(self):
        result = self.actual
        self.assertTrue(result["passed"], result)
        bounds = result["barrier_and_bounds"]
        self.assertTrue(bounds["barrier_ratio_at_z2_ge_one_sixteenth"])
        self.assertTrue(bounds["tail_U_bound_checked"])
        self.assertTrue(bounds["fixed_n_W_lower_positive"])
        self.assertTrue(bounds["vector_coefficient_identity_passed"])
        self.assertTrue(result["envelope_regularity"]["passed"])
        self.assertTrue(all(row["Q_positive"] and row["Q_upper_bound"]
                            for row in result["off_equilibrium_force_samples"]))
        self.assertTrue(all(abs(float(row["force_residual_relative"])) < 1e-30
                            for row in result["off_equilibrium_force_samples"]))
        self.assertTrue(all(row["positive_pressure_gap"]
                            for row in result["fermi_integral_controls"]["rows"]))

    def test_actual_acceptance_fails_closed_on_missing_nan_and_mutated_force(self):
        state = audit.compute_state(include_cycle=False)
        self.assertTrue(audit.acceptance_from_result(state), state)
        missing = json.loads(json.dumps(state))
        del missing["actual_binding_design"]["off_equilibrium_force_samples"][0]
        self.assertFalse(audit.acceptance_from_result(missing))
        nan_value = json.loads(json.dumps(state))
        nan_value["actual_binding_design"]["off_equilibrium_force_samples"][0]["E_W"] = "nan"
        self.assertFalse(audit.acceptance_from_result(nan_value))
        mutated = json.loads(json.dumps(state))
        mutated["actual_binding_design"]["off_equilibrium_force_samples"][0]["force_residual_relative"] = "1"
        self.assertFalse(audit.acceptance_from_result(mutated))
        missing_symbolic = json.loads(json.dumps(state))
        del missing_symbolic["flat_symbolic_checks"]["matter_continuity_full_EW"]
        self.assertFalse(audit.acceptance_from_result(missing_symbolic))
        bad_curved = json.loads(json.dumps(state))
        bad_curved["curved_numeric_check"]["residuals"]["raychaudhuri"] = "nan"
        self.assertFalse(audit.acceptance_from_result(bad_curved))

    def test_flat_positive_energy_bounds(self):
        result = audit.flat_bound_certificate(rho_c="16", beta="2", n=".5",
                                              Gv="3", C_F="1.2", W0="7")
        self.assertTrue(result["tail_constant_exact"])
        self.assertTrue(result["n_bound_positive"])
        self.assertTrue(result["W_lower_positive_for_charge"])
        self.assertGreater(mp.mpf(result["n_max"]), 0)
        with self.assertRaises(ValueError):
            audit.flat_bound_certificate(rho_c=0)

    def test_curved_action_variation_and_candidate(self):
        self.assertGreaterEqual(len(self.curved), 14)
        self.assertTrue(all(row["passed"] for row in self.curved.values()))
        # The explicit kappa=0 reductions are checked independently from the
        # curved identities and must remain exact.
        self.assertTrue(self.curved["candidate_q0_H_zero"]["passed"])
        self.assertTrue(self.curved["candidate_q0_chi_q"]["passed"])
        self.assertTrue(self.curved["candidate_endpoint_chi_q_zero"]["passed"])
        self.assertTrue(self.curved["candidate_endpoint_chi_dot_zero"]["passed"])
        self.assertNotIn("candidate_full_constraint_definition", self.curved)

    def test_wrong_branch_factor_and_potential_fail(self):
        wrong_branch = audit.curved_action_symbolic_checks(branch_sign=1)
        self.assertFalse(wrong_branch["candidate_cuscuton_EL_branch"]["passed"])
        self.assertFalse(wrong_branch["candidate_raychaudhuri_curved"]["passed"])
        wrong_factor = audit.curved_action_symbolic_checks(qdot_curvature_scale=0)
        self.assertFalse(wrong_factor["candidate_constraint_derivative"]["passed"])
        wrong_potential = audit.curved_action_symbolic_checks(potential_scale=2)
        self.assertFalse(wrong_potential["candidate_friedmann_curved"]["passed"])

    def test_endpoint_is_C1_not_C2_and_regularization_is_not_silent(self):
        self.assertTrue(all(row["passed"] for row in self.endpoint.values()))
        witness = audit.regularization_witness(mu2="2", epsilon=".5", X="1e-10")
        self.assertTrue(witness["L_X_negative"])
        self.assertGreater(mp.mpf(witness["negative_for_X_below"]), 0)
        self.assertEqual(witness["declared_X_convention"],
                         "X=(partial chi)^2/2; L=-mu2*sqrt(2X)+epsilon*X/2")
        branches = audit.endpoint_branch_controls()
        self.assertTrue(branches["passed"])
        self.assertTrue(branches["q0"]["H_zero"])
        self.assertTrue(branches["q0"]["X_positive"])
        self.assertTrue(branches["q_pi_over_2"]["X_zero"])
        # At the endpoint, q is regular only as a parameter; chi_q vanishes.
        self.assertIn("potential_second_derivative_diverges", self.endpoint)

    def test_flat_GR_NEC_obstruction_is_retained(self):
        self.assertTrue(all(row["passed"] for row in self.gr.values()))
        Q = mp.mpf("3")
        M2 = mp.mpf("5")
        hdot = -Q/(2*M2)
        self.assertLessEqual(hdot, 0)

    @classmethod
    def _cycle(cls):
        if not hasattr(cls, "cycle"):
            cls.cycle = audit.formal_closed_example()
        return cls.cycle

    def test_formal_closed_original_eos_cycle_refines_and_conserves(self):
        result = self._cycle()
        self.assertTrue(result["formal_background_checks_passed"], result)
        self.assertTrue(result["flags"]["sample_pressure_nonnegative"])
        self.assertTrue(result["flags"]["conservation_refined"])
        self.assertTrue(result["flags"]["constraint_refined"])
        self.assertTrue(result["flags"]["period_refined"])
        self.assertGreater(mp.mpf(result["turnarounds"]["A_lower_q0"]), 0)
        self.assertAlmostEqual(float(result["turnarounds"]["A_upper_q_pi_over_2"]),
                               1.0, places=5)
        self.assertLess(mp.mpf(result["period"]["coarse_fine_relative_difference"]),
                        mp.mpf("2e-6"))
        self.assertIn("formal", result["scope"]["trajectory"])
        self.assertIn("not a regular", result["scope"]["health"])

    def test_input_guards_and_pure_cli(self):
        for kwargs in ({"rho_c_ratio": True}, {"n_ref_ratio": "nan"},
                       {"dps": 20}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                audit.formal_closed_example(**kwargs)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main(["--skip-cycle"])
            result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], audit.STATUS)
        self.assertEqual(result["evidentiary_weight"], 0)
        self.assertTrue(result["mathematical_checks_passed"])
        self.assertIsNone(result["formal_closed_example"])


if __name__ == "__main__":
    unittest.main()
