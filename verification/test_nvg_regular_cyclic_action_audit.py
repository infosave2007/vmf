"""Independent tests for the regular epsilon cuscuton counterfactual."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import unittest

import mpmath as mp

import nvg_regular_cyclic_action_audit as audit


class RegularCyclicActionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.action = audit.action_symbolic_checks()
        cls.flat = audit.flat_reduction_symbolic_checks()
        cls.curved = audit.curved_law_symbolic_checks()
        cls.cost = audit.cost_theorem_symbolic_checks()
        cls.regularity = audit.regularity_symbolic_checks()
        cls.barotropic = audit.closed_barotropic_symbolic_checks()

    def test_actual_lapse_action_and_candidate_equations(self):
        self.assertGreaterEqual(len(self.action), 12)
        self.assertEqual(
            set(self.action),
            set(audit.REQUIRED_SYMBOLIC_ROWS["action_variation_and_candidate"]))
        self.assertTrue(audit.rows_pass(
            self.action,
            audit.REQUIRED_SYMBOLIC_ROWS["action_variation_and_candidate"]))
        self.assertEqual(
            self.action["candidate_key_identity"]["residual"], "0")
        self.assertEqual(
            self.action["candidate_constraint_derivative"]["residual"], "0")

    def test_branch_and_rate_negative_controls_fail(self):
        wrong_branch = audit.action_symbolic_checks(branch_sign=1)
        self.assertFalse(wrong_branch["candidate_cuscuton_EL_branch"]["passed"])
        self.assertFalse(wrong_branch["candidate_raychaudhuri_curved"]["passed"])
        wrong_rate = audit.action_symbolic_checks(q_rate_scale=2)
        self.assertFalse(wrong_rate["candidate_raychaudhuri_curved"]["passed"])
        self.assertFalse(wrong_rate["candidate_constraint_derivative"]["passed"])

    def test_flat_reduction_and_bare_curvature_derivatives(self):
        self.assertTrue(audit.rows_pass(self.flat))
        self.assertTrue(audit.rows_pass(self.curved))
        self.assertIn("low_density_curvature_coefficient", self.curved)
        self.assertEqual(
            self.curved["constraint_F_uses_bare_M"]["residual"], "0")

    def test_cost_theorem_and_regular_map_symbolics(self):
        self.assertTrue(audit.rows_pass(self.cost))
        self.assertTrue(audit.rows_pass(self.regularity))
        self.assertTrue(audit.rows_pass(self.barotropic))
        self.assertEqual(self.cost["family_gravity_ratio"]["residual"], "0")
        self.assertEqual(
            self.regularity["epsilon_zero_gradient"]["residual"], "0")

    def test_high_precision_endpoint_inverse_at_80_and_120_digits(self):
        for dps in (80, 120):
            result = audit.endpoint_inverse_derivative(dps=dps)
            self.assertTrue(result["valid"])
            self.assertTrue(result["passed"])
            self.assertEqual(result["product_check"], "1.0")
            self.assertLess(
                mp.mpf(result["finite_difference_relative_error"]),
                mp.power(10, -(dps // 2)))

    def test_high_precision_vacuum_limit_at_80_and_120_digits(self):
        for dps in (80, 120):
            result = audit.vacuum_limit_check(dps=dps)
            self.assertTrue(result["valid"])
            self.assertTrue(result["passed"])
            self.assertLess(
                mp.mpf(result["direct_relative_error"]),
                mp.power(10, -(dps // 2)))
            self.assertEqual(result["theorem_relative_error"], "0.0")

    def test_independent_curved_point_residuals(self):
        point = audit.point_identity_check()
        self.assertTrue(point["passed"])
        self.assertLess(mp.mpf(point["max_scaled_residual"]), mp.mpf("1e-70"))
        self.assertLess(mp.mpf(point["residuals"]["key_identity"]), mp.mpf("1e-70"))

    def test_both_manufactured_constant_w_cycles(self):
        for w in ("0", "0.3333333333333333333333333333333333"):
            cycle = audit.constant_w_cycle(w=w, epsilon="0.1",
                                           rho_c_ratio="1")
            self.assertTrue(cycle["inputs_manufactured"])
            self.assertTrue(cycle["passed"])
            self.assertGreater(cycle["A_low"], 0)
            self.assertLess(cycle["A_low"], cycle["A_high"])
            self.assertGreater(cycle["min_dimensionless_Psi_dot"], 0)
            self.assertGreaterEqual(cycle["min_dimensionless_Psi_q"], 0.1)
            self.assertGreater(cycle["turnaround_dimensionless_Psi_dot"], 0)
            self.assertLess(cycle["max_F_derivative"], 0)
            self.assertLess(cycle["quadrature_refinement_relative"], 3e-7)
            self.assertLess(cycle["ode_vs_quadrature_relative"], 3e-6)
            self.assertLess(cycle["trajectory_constraint_curve_max_abs"], 3e-7)
            self.assertEqual(
                cycle["algebraic_curve_constraint_max_abs"],
                cycle["max_constraint_residual"])
            self.assertEqual(
                cycle["direct_ode_trajectory_curve_max_abs"],
                cycle["trajectory_constraint_curve_max_abs"])
            self.assertEqual(cycle["normalization"]["A"], "a/a_ref")
            self.assertEqual(cycle["normalization"]["tau"], "t/a_ref")
            self.assertIn("algebraic curve and direct ODE trajectory are separate",
                          cycle["scope"])

    def test_closed_barotropic_theorem_is_scoped(self):
        theorem = audit.closed_barotropic_theorem(w="0.3333333333333333")
        self.assertTrue(theorem["P_nonnegative"])
        self.assertTrue(theorem["constant_w_qrate_positive_on_A_le_1"])
        self.assertIn("no S2 EOS reconstruction", theorem["scope"])

    def test_endpoint_and_model_negative_controls(self):
        positive = audit.epsilon_domain_control(epsilon="0.1")
        zero = audit.epsilon_domain_control(epsilon="0")
        negative = audit.epsilon_domain_control(epsilon="-0.1")
        self.assertTrue(positive["valid_regular_family"])
        self.assertFalse(zero["valid_regular_family"])
        self.assertTrue(zero["zero_gradient_obstruction"])
        self.assertFalse(negative["valid_regular_family"])
        self.assertTrue(negative["negative_epsilon_obstruction"])

        old = audit.old_constraint_control()
        omitted = audit.omitted_rate_control()
        kinetic = audit.kinetic_substitution_control()
        self.assertFalse(old["accepted"])
        self.assertTrue(old["nonzero_for_positive_epsilon"])
        self.assertFalse(omitted["accepted"])
        self.assertTrue(omitted["nonzero"])
        self.assertFalse(kinetic["accepted"])
        self.assertTrue(kinetic["same_action"] is False)
        self.assertTrue(kinetic["L_X_negative_near_zero"])
        self.assertTrue(kinetic["combination_equals_epsilon_over_2"])

    def test_rows_pass_fails_closed_when_evidence_is_mutated(self):
        rows = {"identity": {"residual": "0", "passed": True}}
        self.assertTrue(audit.rows_pass(rows))
        rows["identity"]["passed"] = False
        self.assertFalse(audit.rows_pass(rows))
        rows["identity"]["passed"] = True
        rows["identity"]["residual"] = "nonzero"
        self.assertFalse(audit.rows_pass(rows))
        self.assertFalse(audit.rows_pass({}))

    def test_aggregate_gate_rejects_mutated_numerical_evidence(self):
        result = audit.run_audit(epsilon="0.1", rho_c_ratio="1")
        self.assertTrue(audit.acceptance_from_result(result))
        bad_numeric = copy.deepcopy(result)
        bad_numeric["numerical"]["cycles"][0]["max_raychaudhuri_residual"] = "1"
        self.assertFalse(audit.acceptance_from_result(bad_numeric))
        bad_symbolic = copy.deepcopy(result)
        bad_symbolic["symbolic"]["curved_law"]["candidate"] = {
            "residual": "0", "passed": False}
        self.assertFalse(audit.acceptance_from_result(bad_symbolic))

    def test_aggregate_gate_requires_every_symbolic_row(self):
        result = audit.run_audit(epsilon="0.1", rho_c_ratio="1")
        self.assertTrue(audit.acceptance_from_result(result))
        for group, required_rows in audit.REQUIRED_SYMBOLIC_ROWS.items():
            for row_name in required_rows:
                mutated = copy.deepcopy(result)
                del mutated["symbolic"][group][row_name]
                with self.subTest(group=group, row=row_name):
                    self.assertFalse(audit.acceptance_from_result(mutated))

    def test_aggregate_gate_requires_cycle_schema_and_finite_numbers(self):
        result = audit.run_audit(epsilon="0.1", rho_c_ratio="1")
        self.assertTrue(audit.acceptance_from_result(result))
        for field in audit.REQUIRED_CYCLE_KEYS:
            missing = copy.deepcopy(result)
            del missing["numerical"]["cycles"][0][field]
            with self.subTest(missing_required_cycle_field=field):
                self.assertFalse(audit.acceptance_from_result(missing))
        for field in audit.CYCLE_NUMERIC_FIELDS:
            missing = copy.deepcopy(result)
            del missing["numerical"]["cycles"][0][field]
            with self.subTest(missing=field):
                self.assertFalse(audit.acceptance_from_result(missing))
            for replacement in ("", "NaN", "Infinity", True):
                malformed = copy.deepcopy(result)
                malformed["numerical"]["cycles"][0][field] = replacement
                with self.subTest(field=field, replacement=repr(replacement)):
                    self.assertFalse(audit.acceptance_from_result(malformed))

    def test_aggregate_gate_requires_every_numeric_record_field(self):
        result = audit.run_audit(epsilon="0.1", rho_c_ratio="1")
        self.assertTrue(audit.acceptance_from_result(result))
        records = (
            ("point", ("numerical", "curved_point"), audit.REQUIRED_POINT_KEYS),
            ("inverse80", ("numerical", "endpoint_inverse_80"),
             audit.REQUIRED_ENDPOINT_INVERSE_KEYS),
            ("vacuum120", ("numerical", "vacuum_limit_120"),
             audit.REQUIRED_VACUUM_LIMIT_KEYS),
            ("old", ("negative_controls", "old_constraint"),
             audit.REQUIRED_OLD_CONSTRAINT_KEYS),
            ("omitted", ("negative_controls", "omitted_rate_factor"),
             audit.REQUIRED_OMITTED_RATE_KEYS),
            ("kinetic", ("negative_controls", "epsilon_X_kinetic_substitution"),
             audit.REQUIRED_KINETIC_NEGATIVE_KEYS),
            ("domain", ("negative_controls", "epsilon_domain"),
             audit.REQUIRED_DOMAIN_CONTROL_KEYS),
        )
        for label, (parent, record), required_keys in records:
            for field in required_keys:
                mutated = copy.deepcopy(result)
                del mutated[parent][record][field]
                with self.subTest(record=label, field=field):
                    self.assertFalse(audit.acceptance_from_result(mutated))
        for field in audit.POINT_RESIDUAL_NAMES:
            mutated = copy.deepcopy(result)
            del mutated["numerical"]["curved_point"]["residuals"][field]
            with self.subTest(record="point residuals", field=field):
                self.assertFalse(audit.acceptance_from_result(mutated))

    def test_aggregate_gate_requires_unique_cases_precision_and_normalization(self):
        result = audit.run_audit(epsilon="0.1", rho_c_ratio="1")
        self.assertTrue(audit.acceptance_from_result(result))

        duplicate_case = copy.deepcopy(result)
        duplicate_case["numerical"]["cycles"][1]["w"] = \
            duplicate_case["numerical"]["cycles"][0]["w"]
        self.assertFalse(audit.acceptance_from_result(duplicate_case))

        for name in ("endpoint_inverse_80", "endpoint_inverse_120",
                     "vacuum_limit_80", "vacuum_limit_120"):
            mutated = copy.deepcopy(result)
            mutated["numerical"][name]["dps"] = 90
            with self.subTest(precision=name):
                self.assertFalse(audit.acceptance_from_result(mutated))

        float_precision = copy.deepcopy(result)
        float_precision["numerical"]["endpoint_inverse_80"]["dps"] = 80.0
        self.assertFalse(audit.acceptance_from_result(float_precision))

        swapped = copy.deepcopy(result)
        swapped["numerical"]["endpoint_inverse_80"]["dps"] = 120
        self.assertFalse(audit.acceptance_from_result(swapped))

        bad_normalization = copy.deepcopy(result)
        bad_normalization["numerical"]["cycles"][0]["normalization"]["A"] = "a"
        self.assertFalse(audit.acceptance_from_result(bad_normalization))

        for field in ("algebraic_curve_constraint_max_abs",
                      "direct_ode_trajectory_curve_max_abs"):
            missing = copy.deepcopy(result)
            del missing["numerical"]["cycles"][0][field]
            with self.subTest(separate_residual=field):
                self.assertFalse(audit.acceptance_from_result(missing))

    def test_period_accuracy_gate_rejects_consistent_large_errors(self):
        result = audit.run_audit(epsilon="0.1", rho_c_ratio="1")
        self.assertTrue(audit.acceptance_from_result(result))
        for period_field, error_field in (
                ("period_quadrature_coarse", "quadrature_refinement_relative"),
                ("period_direct_ode", "ode_vs_quadrature_relative")):
            mutated = copy.deepcopy(result)
            cycle = mutated["numerical"]["cycles"][0]
            cycle[period_field] = 2 * cycle["period_quadrature_fine"]
            cycle[error_field] = abs(
                cycle[period_field] - cycle["period_quadrature_fine"]
            ) / max(1, abs(cycle["period_quadrature_fine"]))
            self.assertTrue(cycle["passed"])
            with self.subTest(period=period_field, error=error_field):
                self.assertFalse(audit.acceptance_from_result(mutated))

    def test_aggregate_gate_recomputes_negative_controls_and_domain(self):
        result = audit.run_audit(epsilon="0.1", rho_c_ratio="1")
        self.assertTrue(audit.acceptance_from_result(result))
        mutations = (
            ("old_constraint", "actual_minus_old_constraint", "0"),
            ("omitted_rate_factor", "raychaudhuri_residual", "0"),
            ("omitted_rate_factor", "wrong_qdot", "0"),
            ("epsilon_X_kinetic_substitution", "L_X", "0"),
            ("epsilon_domain", "epsilon", "0.2"),
        )
        for control, field, replacement in mutations:
            mutated = copy.deepcopy(result)
            mutated["negative_controls"][control][field] = replacement
            with self.subTest(control=control, field=field):
                self.assertFalse(audit.acceptance_from_result(mutated))

        bad_point = copy.deepcopy(result)
        bad_point["numerical"]["curved_point"]["residuals"]["friedmann"] = True
        self.assertFalse(audit.acceptance_from_result(bad_point))
        bad_point = copy.deepcopy(result)
        bad_point["numerical"]["curved_point"]["max_scaled_residual"] = "NaN"
        self.assertFalse(audit.acceptance_from_result(bad_point))

    def test_module_has_no_s2_dependency_and_has_zero_empirical_weight(self):
        with open(audit.SOURCE_PATH, encoding="utf-8") as source_file:
            source = source_file.read()
        self.assertNotIn("nvg_bounce_potential_audit", source)
        self.assertEqual(audit.EVIDENCE_WEIGHT, 0)
        self.assertTrue(audit.SOURCES["planck_mass_shift_context"].endswith(
            "astro-ph/0702002"))

    def test_cli_is_json_only_and_default_passes(self):
        command = [sys.executable, audit.SOURCE_PATH]
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(command, check=False, capture_output=True,
                                   text=True, env=environment,
                                   cwd=os.path.dirname(os.path.dirname(
                                       audit.SOURCE_PATH)))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertTrue(result["mathematical_checks_passed"])
        self.assertEqual(result["evidence_weight"], 0)
        self.assertEqual(completed.stderr, "")

    def test_cli_rejects_zero_epsilon_contract(self):
        command = [sys.executable, audit.SOURCE_PATH, "--epsilon", "0"]
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(command, check=False, capture_output=True,
                                   text=True, env=environment,
                                   cwd=os.path.dirname(os.path.dirname(
                                       audit.SOURCE_PATH)))
        self.assertNotEqual(completed.returncode, 0)
        result = json.loads(completed.stdout)
        self.assertFalse(result["mathematical_checks_passed"])
        self.assertFalse(result["regularity"]["valid_regular_family"])
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
