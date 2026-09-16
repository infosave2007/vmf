"""Focused regression tests for the bounded nuclear-closure audit.

The two genuine calculations are made once per test class.  The mutation
checks then exercise the validator against copies, so an empty atlas or a
missing branch cannot accidentally pass while every test reruns the full
interval subdivision.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_nuclear_closure_audit as audit


class NuclearClosureAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The required 80/110 pair is computed once.  All individual tests
        # below use this validated result or mutated copies of it.
        cls.built = audit.build_result()
        cls.high = cls.built["science"]
        audit.validate_result(cls.high)

    def test_global_support_has_closed_union_and_obligations(self):
        for target in audit.TARGETS:
            row = self.high["global_support"][target]
            self.assertEqual(row["global_status"], "CERTIFIED_OUTWARD_INTERVAL_GLOBAL_SUPPORT")
            self.assertTrue(row["proof_integrity_passed"])
            middle = row["middle_interval_proof"]
            self.assertTrue(middle["union_has_no_gaps"])
            self.assertTrue(middle["compact_union_coverage"]["passed"])
            self.assertEqual(middle["unresolved_count"], 0)
            for side in ("left", "right"):
                atlas = middle[side]
                self.assertTrue(atlas["intervals"])
                self.assertTrue(atlas["coverage"]["passed"])
                self.assertFalse(atlas["unresolved"])
                self.assertTrue(all(leaf["accepted"] for leaf in atlas["intervals"]))
            onset = row["onset_tail_proof"]
            self.assertTrue(onset["passed"])
            self.assertTrue(onset["intervals"])
            self.assertTrue(onset["coverage"]["passed"])
            self.assertFalse(onset["unresolved"])
            self.assertTrue(row["positive_grid_is_not_proof"])

    def test_equality_tiny_y_onset_and_directed_calibration_are_real(self):
        for target in audit.TARGETS:
            row = self.high["global_support"][target]
            equality = row["equality_neighborhood"]
            self.assertTrue(equality["equality_H_contains_zero"])
            self.assertTrue(equality["equality_Hprime_contains_zero"])
            self.assertLess(mp.mpf(equality["equality_H_exact_residual"]), mp.mpf("1e-55"))
            self.assertLess(mp.mpf(equality["equality_Hprime_exact_residual"]), mp.mpf("1e-55"))
            self.assertGreater(mp.mpf(equality["Hsecond_lower_MeV4"]), 0)
            tiny = row["tiny_y_bound"]
            self.assertGreater(mp.mpf(tiny["U_lower_MeV4"]), mp.mpf(tiny["Pi_upper_MeV4"]))
            self.assertTrue(row["onset_to_infinity"]["pi_zero_for_y_ge_onset"])
            self.assertTrue(row["onset_to_infinity"]["a4_positive"])
            self.assertTrue(row["calibration_interval"]["outward_from_exact_declared_inputs"])
            self.assertTrue(all(len(interval) == 2 for interval in row["coefficients_interval_MeV4"]))
            self.assertTrue(equality["exact_equality_from_inverse_jet_system"])
            self.assertTrue(equality["calibration_matrix_determinant_formula_passed"])
            self.assertTrue(equality["Fyy_nonnegative_integrand_guard"])
            self.assertGreater(mp.mpf(equality["optimizer_D_n_lower"]), 0)

    def test_static_finite_q_uses_independent_saddle_and_controls(self):
        for target in audit.TARGETS:
            row = self.high["finite_q"][target]
            self.assertEqual(row["finite_q_status"], "STRICT_STATIC_TF_ALL_Q_CERTIFICATE")
            self.assertTrue(row["guards"]["Dq_positive_all_x"])
            self.assertTrue(row["full_uneliminated_hessian"]["passed"])
            for error in row["full_uneliminated_hessian"]["relative_errors"].values():
                self.assertLess(mp.mpf(error), mp.mpf("1e-55"))
            schur = row["schur_reduction"]
            self.assertEqual(len(schur["P_coefficients"]), 3)
            self.assertEqual(len(schur["Qc_coefficients"]), 3)
            self.assertTrue(schur["P_classification"]["strict_positive"])
            self.assertTrue(schur["Qc_classification"]["strict_positive"])
            self.assertTrue(schur["P0_identity_passed"])
            self.assertTrue(row["negative_controls"]["missing_h"]["rejected"])
            self.assertTrue(row["negative_controls"]["wrong_h_sign"]["rejected"])
            directed = row["directed_interval_certificate"]
            self.assertTrue(directed["passed"])
            self.assertTrue(directed["P0_K_identity_contains_zero"])
            self.assertTrue(directed["threshold_strict_actual"])
            self.assertTrue(all(mp.mpf(value) > 0 for value in directed["P_coefficient_lower_bounds"]))
        self.assertEqual(
            self.high["finite_q"]["0.90"]["gradient_threshold"]["status"],
            "STRICT_FOR_G_GREATER_THAN_GMIN",
        )
        self.assertEqual(
            self.high["finite_q"]["0.93"]["gradient_threshold"]["status"],
            "STRICT_FOR_EVERY_G_POSITIVE",
        )

    def test_negative_control_recomputes_a4_and_direct_witness(self):
        row = self.high["negative_control"]
        self.assertEqual(row["global_status"], "COUNTEREXAMPLE_UNBOUNDED_NEGATIVE_TAIL")
        self.assertTrue(row["proof_integrity_passed"])
        self.assertTrue(row["unbounded_negative_tail"])
        self.assertLess(mp.mpf(row["a4_upper"]), 0)
        self.assertTrue(row["direct_witness"]["negative_interval"])
        self.assertLess(mp.mpf(row["direct_witness"]["outward_upper_MeV4"]), 0)

    def test_quadratic_classifier_covers_strict_marginal_unstable_and_invalid(self):
        self.assertEqual(audit.classify_quadratic_halfline(1, 0, 1)["status"], "STRICT_POSITIVE")
        self.assertEqual(audit.classify_quadratic_halfline(0, 0, 0)["status"], "MARGINAL_FLAT")
        band = audit.classify_quadratic_halfline(1, -3, 2)
        self.assertEqual(band["status"], "UNSTABLE_BAND")
        self.assertEqual([str(x) for x in band["roots"]], ["1.0", "2.0"])
        self.assertEqual(audit.classify_quadratic_halfline(-1, 0, 1)["status"], "UNSTABLE_TAIL")
        with self.assertRaises(audit.NuclearClosureError):
            audit.classify_quadratic_halfline("nan", 0, 1)
        with self.assertRaises(audit.NuclearClosureError):
            audit.classify_quadratic_halfline("inf", 0, 1)

    def test_mutated_copies_fail_closed(self):
        empty_atlas = copy.deepcopy(self.high)
        empty_atlas["global_support"]["0.90"]["middle_interval_proof"]["left"]["intervals"] = []
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(empty_atlas)

        missing_branch = copy.deepcopy(self.high)
        missing_branch["finite_q"].pop("0.93")
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(missing_branch)

        equality_mutation = copy.deepcopy(self.high)
        equality_mutation["global_support"]["0.90"]["equality_neighborhood"]["equality_Hprime_contains_zero"] = False
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(equality_mutation)

        gap_mutation = copy.deepcopy(self.high)
        gap_mutation["global_support"]["0.90"]["middle_interval_proof"]["compact_union_coverage"]["passed"] = False
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(gap_mutation)

        microscopic_gap = copy.deepcopy(self.high)
        microscopic_gap["global_support"]["0.90"]["middle_interval_proof"]["left"]["intervals"][0]["lo"] = "0.50000000000000000000000000000000005"
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(microscopic_gap)

        proof_box_mismatch = copy.deepcopy(self.high)
        proof_box_mismatch["global_support"]["0.90"]["middle_interval_proof"]["left"]["intervals"][0]["proof_box_endpoints"][0] = "0.49999999999999999999999999999999995"
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(proof_box_mismatch)

        missing_symbol = copy.deepcopy(self.high)
        missing_symbol["symbolic_checks"].pop("gradient_threshold_identity")
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(missing_symbol)

        exact_equality = copy.deepcopy(self.high)
        exact_equality["global_support"]["0.90"]["equality_neighborhood"]["exact_equality_from_inverse_jet_system"] = False
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(exact_equality)

        onset_leaf = copy.deepcopy(self.high)
        onset_leaf["global_support"]["0.90"]["onset_tail_proof"]["intervals"][0]["accepted"] = False
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(onset_leaf)

        onset_overlap = copy.deepcopy(self.high)
        onset_overlap["global_support"]["0.90"]["onset_cover_overlap"]["compact_end_strictly_above_exact_onset"] = False
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(onset_overlap)

        optimizer_guard = copy.deepcopy(self.high)
        optimizer_guard["global_support"]["0.90"]["optimizer"]["Cv_interval_positive"] = False
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(optimizer_guard)

        tiny_argmin = copy.deepcopy(self.high)
        tiny_argmin["global_support"]["0.90"]["tiny_y_bound"]["q_tail_argmin_used"] = "-1"
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(tiny_argmin)

        finite_guard = copy.deepcopy(self.high)
        finite_guard["finite_q"]["0.90"]["directed_interval_certificate"]["P0_K_identity_contains_zero"] = False
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(finite_guard)

        threshold = copy.deepcopy(self.high)
        threshold["finite_q"]["0.90"]["gradient_threshold"]["strict_actual"] = False
        with self.assertRaises(audit.NuclearClosureError):
            audit.validate_result(threshold)

    def test_cli_is_strict_json_and_does_not_write(self):
        script = HERE / "nvg_nuclear_closure_audit.py"
        command = [sys.executable, "-B", str(script), "--dps", "79"]
        completed = subprocess.run(command, cwd=HERE.parent, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "INVALID_OR_FAILED_NUCLEAR_CLOSURE_AUDIT")
        self.assertFalse(payload["audit_integrity"]["passed"])

        # A valid smoke result is JSON round-trippable and includes no output
        # path or write marker beyond the explicit no-write contract.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(audit.main(["--dps", "80"]), 0)
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["audit_integrity"]["strict_json_no_write"])
        self.assertNotIn("output_path", payload)


if __name__ == "__main__":
    unittest.main()
