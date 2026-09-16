"""Focused regression tests for the finite-N W8 droplet audit."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_finite_droplet_audit as audit


class FiniteDropletAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The four live cases are computed once; mutation tests below exercise
        # copies and never turn a saved answer table into an input.
        cls.result = audit.build_result()

    def test_four_frozen_cases_and_protocol_coverage(self):
        self.assertEqual(self.result["schema_version"], audit.SCHEMA)
        self.assertEqual(set(self.result["cases"]), set(audit.TARGETS))
        for target in audit.TARGETS:
            for case_name in ("N40", "N208"):
                case = self.result["cases"][target][case_name]
                self.assertEqual(case["target_y"], target)
                self.assertEqual(case["target_N"], int(case_name[1:]))
                self.assertTrue(case["all_seed_outcomes_reported"])
                self.assertEqual(len(case["seed_attempts"]), 3)

    def test_bound_n208_candidates_are_live_and_not_static_claims(self):
        for target in audit.TARGETS:
            branches = self.result["cases"][target]["N208"]["distinct_branches"]
            accepted = [branch for branch in branches if branch["terminal_stationarity_candidate"]]
            self.assertTrue(accepted)
            terminal = accepted[0]["larger_domain_check"]
            self.assertTrue(terminal["localized_vacuum_exterior"])
            self.assertTrue(terminal["binding_condition_E_per_N_lt_M"])
            self.assertLessEqual(terminal["conserved_N_relative_error"], audit.N_RELATIVE_LIMIT)
            self.assertLessEqual(terminal["field_residual_relative"], audit.FIELD_RESIDUAL_LIMIT)
            self.assertLessEqual(terminal["virial_relative"], audit.VIRIAL_RELATIVE_LIMIT)

    def test_gauss_boundary_identity_and_scale_oracle_are_explicit(self):
        for target in audit.TARGETS:
            branch = next(
                branch for branch in self.result["cases"][target]["N208"]["distinct_branches"]
                if branch["terminal_stationarity_candidate"]
            )
            row = branch["larger_domain_check"]
            self.assertLess(row["gauss_energy_identity_relative"], 1.0e-8)
            self.assertLess(abs(row["energy_minus_muN_identity_residual_MeV"]), 1.0e-3)
            oracle = branch["fixed_N_scale_check"]
            self.assertTrue(oracle["gauss_resolved_for_each_lambda"])
            self.assertEqual(len(oracle["steps"]), 4)
            self.assertTrue(oracle["finite_difference_converged"])
            self.assertTrue(oracle["analytic_and_re_solved_agree"])
            self.assertIn("Gauss_boundary_flux", row["energy_components_MeV"])

    def test_n40_is_not_promoted_to_a_bound_claim(self):
        for target in audit.TARGETS:
            branches = self.result["cases"][target]["N40"]["distinct_branches"]
            self.assertTrue(branches)
            for branch in branches:
                terminal = branch["larger_domain_check"]
                if terminal.get("converged"):
                    self.assertFalse(branch["terminal_stationarity_candidate"])
                    self.assertFalse(terminal["binding_condition_E_per_N_lt_M"])

    def test_potential_second_derivative_matches_independent_polynomial(self):
        # Keep this helper regression independent of its closed-form algebra:
        # differentiate the U(y) polynomial with NumPy's Polynomial object.
        import numpy as np

        samples = np.asarray((0.71, 0.83, 0.93, 1.07, 1.19), dtype=float)
        for target in audit.TARGETS:
            design = audit.W8Design.from_target(target)
            a2, a3, a4 = design.coefficients_dim
            z = np.polynomial.Polynomial((-1.0, 0.0, 1.0))
            polynomial = a2 * z**2 + a3 * z**3 + a4 * z**4
            expected = polynomial.deriv(2)(samples)
            actual = design.potential_yy(samples)
            np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)

    def test_mutations_fail_closed(self):
        missing = copy.deepcopy(self.result)
        missing["cases"]["0.90"].pop("N208")
        self.assertFalse(audit.validate_result(missing))
        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["N208"]["distinct_branches"][0]["fixed_N_scale_check"]["gauss_resolved_for_each_lambda"] = False
        self.assertFalse(audit.validate_result(changed))
        changed = copy.deepcopy(self.result)
        changed["evidence_weight"] = 1.0
        self.assertFalse(audit.validate_result(changed))

    def test_fresh_build_validates_end_to_end(self):
        self.assertTrue(audit.validate_result(audit.build_result()))

    def test_json_roundtrip_preserves_validation(self):
        encoded = json.dumps(self.result, allow_nan=False)
        roundtripped = json.loads(encoded)
        self.assertTrue(audit.validate_result(roundtripped))

    def test_terminal_aggregation_guards_use_actual_protocol_evidence(self):
        target = "0.90"
        base = next(
            branch for branch in self.result["cases"][target]["N208"]["distinct_branches"]
            if branch["terminal_stationarity_candidate"]
        )

        changed = copy.deepcopy(self.result)
        branch = next(
            branch for branch in changed["cases"][target]["N208"]["distinct_branches"]
            if branch["branch_id"] == base["branch_id"]
        )
        branch["larger_domain_check"]["converged"] = False
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        branch = next(
            branch for branch in changed["cases"][target]["N208"]["distinct_branches"]
            if branch["branch_id"] == base["branch_id"]
        )
        branch["refinements"].pop()
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        branch = next(
            branch for branch in changed["cases"][target]["N208"]["distinct_branches"]
            if branch["branch_id"] == base["branch_id"]
        )
        branch["fixed_N_scale_check"]["steps"][0]["N_plus_controlled"] += 1.0
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        branch = next(
            branch for branch in changed["cases"][target]["N208"]["distinct_branches"]
            if branch["branch_id"] == base["branch_id"]
        )
        branch["larger_domain_check"]["energy_components_MeV"]["Gauss_boundary_flux"] += 1.0
        self.assertFalse(audit.validate_result(changed))

    def test_cli_is_strict_json_and_no_write(self):
        script = HERE / "nvg_finite_droplet_audit.py"
        before = script.read_bytes()
        completed = subprocess.run(
            [sys.executable, "-B", str(script)],
            cwd=HERE.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema_version"], audit.SCHEMA)
        self.assertEqual(payload["evidence_weight"], 0.0)
        self.assertEqual(script.read_bytes(), before)
        self.assertNotIn("output_path", payload)


if __name__ == "__main__":
    unittest.main()
