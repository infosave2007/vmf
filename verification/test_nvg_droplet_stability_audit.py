"""Focused tests for the live finite-droplet static stability audit."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_droplet_stability_audit as audit


class _SummaryBackground:
    """Small dependency fixture exposing only the live summary boundary."""

    def __init__(self, summary):
        self._summary = copy.deepcopy(summary)

    def summary(self):
        return copy.deepcopy(self._summary)


class DropletStabilityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The live result is computed once.  Mutation tests below operate on
        # copies and never turn a saved answer table into a producer input.
        cls.result = audit.build_result()

    def test_two_live_backgrounds_and_full_protocol_matrix(self):
        self.assertEqual(self.result["schema_version"], audit.SCHEMA)
        self.assertEqual(self.result["status"], audit.STATUS)
        self.assertEqual(self.result["evidence_weight"], 0.0)
        self.assertEqual(set(self.result["cases"]), set(audit.TARGETS))
        for target in audit.TARGETS:
            case = self.result["cases"][target]
            self.assertEqual(case["target_N"], audit.TARGET_N)
            self.assertTrue(case["all_planned_ell_sectors_reported"])
            self.assertTrue(case["all_spectrum_records_available"])
            self.assertEqual(set(case["sectors"]), {str(ell) for ell in audit.ELL_VALUES})
            for ell in audit.ELL_VALUES:
                sector = case["sectors"][str(ell)]
                self.assertEqual(
                    [row["basis_size_each_block"] for row in sector["basis_sweep"]],
                    list(audit.BASIS_SIZES),
                )
                self.assertEqual(
                    [row["intervals"] for row in sector["grid_sweep_largest_basis"]],
                    list(audit.GRID_INTERVALS),
                )
                self.assertEqual(
                    [row["box_fm"] for row in sector["domain_sweep_largest_basis"]],
                    list(audit.DOMAIN_BOXES_FM),
                )
                terminal = sector["terminal_largest_basis"]
                self.assertTrue(terminal["positive_operator"])
                self.assertEqual(terminal["basis_dimension"], 2 * audit.BASIS_SIZES[-1])
                self.assertEqual(len(terminal["eigenvalues_Q_over_W0"]), terminal["gram_rank"])

    def test_fd_translation_and_scope_guards_are_live(self):
        for target in audit.TARGETS:
            case = self.result["cases"][target]
            translation = case["translation_metrics_24fm"]
            self.assertTrue(translation["translation_basis_in_trial_space"])
            self.assertTrue(translation["operator_positive"])
            self.assertEqual(
                translation["translation_consistency"],
                "MEASURED_FINITE_DOMAIN_IDENTITY",
            )
            self.assertLess(abs(translation["translation_Q_over_W0"]), 1.0e-5)
            checks = case["independent_energy_difference_checks_24fm"]
            self.assertGreaterEqual(len(checks), 2)
            for check in checks:
                self.assertTrue(check["available"])
                self.assertTrue(check["agreement"])
                self.assertTrue(check["hessian_operator_positive"])
                self.assertEqual(
                    [value["amplitude"] for value in check["values"]],
                    list(audit.FD_AMPLITUDES),
                )
                self.assertLessEqual(
                    check["fixed_N_relative_error_max"], audit.FIXED_N_RELATIVE_LIMIT
                )
            consistency = case["terminal_consistency_checks"]
            self.assertTrue(consistency["fixed_N_additive_paths"]["all_checks_pass"])
            self.assertTrue(consistency["global_Gauss_re_solves"]["all_checks_pass"])
            self.assertTrue(consistency["scalar_only_energy_difference"]["agreement"])
            self.assertTrue(self.result["model_scope"]["curvatures_are_not_frequencies"])
            self.assertTrue(self.result["model_scope"]["positive_finite_basis_is_not_full_stability"])

    def test_i2_projection_constraints_rank_and_error_guards_are_live(self):
        for target in audit.TARGETS:
            case = self.result["cases"][target]
            self.assertTrue(case["terminal_controls_pass"])
            self.assertEqual(case["terminal_control_failures"], [])
            self.assertIn("24fm_tighter", case["backgrounds"])
            for ell in audit.ELL_VALUES:
                sector = case["sectors"][str(ell)]
                rows = (
                    sector["basis_sweep"]
                    + sector["grid_sweep_largest_basis"]
                    + sector["domain_sweep_largest_basis"]
                    + [sector["tighter_background_same_domain"]]
                )
                for row in rows:
                    self.assertEqual(row["basis_dimension"], 2 * row["basis_size_each_block"])
                    expected_internal = row["basis_dimension"] - (1 if ell == 1 else 0)
                    self.assertEqual(row["projection_removed_dimension"], 1 if ell == 1 else 0)
                    self.assertEqual(row["projection_basis_dimension"], expected_internal)
                    self.assertEqual(row["internal_gram_size"], expected_internal)
                    self.assertLessEqual(
                        row["projection_orthogonality_relative"],
                        audit.PROJECTION_ORTHOGONALITY_LIMIT,
                    )
                    self.assertEqual(
                        len(row["eigen_residual_absolute_Q_over_W0"]),
                        row["gram_rank"],
                    )
                    self.assertEqual(
                        len(row["internal_eigen_residual_absolute_Q_over_W0"]),
                        row["internal_gram_rank"],
                    )
                self.assertTrue(sector["rayleigh_ritz_monotonicity"]["monotone_with_allowance"])
                self.assertTrue(isinstance(sector["background_delta_max"], float))
                if ell == 0:
                    self.assertLessEqual(
                        sector["terminal_largest_basis"]["number_constraint_residual_max"],
                        audit.NUMBER_CONSTRAINT_RELATIVE_LIMIT,
                    )
                    self.assertEqual(
                        sector["terminal_largest_basis"]["number_constraint_zero_vector_count"],
                        audit.BASIS_SIZES[-1],
                    )

    def test_i2_validator_rejects_missing_controls_and_stale_pass_flags(self):
        changed = copy.deepcopy(self.result)
        del changed["cases"]["0.90"]["sectors"]["6"]
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["backgrounds"]["24fm_tighter"]["converged"] = False
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["sectors"]["0"]["terminal_largest_basis"]["number_constraint_residual_max"] = 1.0
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["sectors"]["1"]["terminal_largest_basis"]["projection_removed_dimension"] = 0
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["sectors"]["1"]["terminal_largest_basis"]["projection_orthogonality_relative"] = 1.0
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["sectors"]["0"]["terminal_largest_basis"]["eigen_residual_absolute_max_Q_over_W0"] = 1.0
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["sectors"]["6"]["terminal_largest_basis"]["gram_rank_loss"] = 0
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["terminal_controls_pass"] = False
        self.assertFalse(audit.validate_result(changed))

    def test_manufactured_positive_gauss_control(self):
        design = audit.finite.W8Design.from_target("0.90")
        x = np.linspace(0.0, 6.0, 241)
        y = np.full_like(x, 0.9)
        envelope = np.exp(-0.4 * x**2)
        source = np.column_stack((0.4 * envelope, -0.2 * x * envelope))
        solved, pivot, positive = audit._operator_solve(design, x, y, 0, source)
        self.assertTrue(positive)
        self.assertGreater(pivot, 0.0)
        quadratic = audit._integral_matrix(
            x,
            x[:, None, None] ** 2 * source[:, :, None] * solved[:, None, :],
        )
        self.assertTrue(np.all(np.linalg.eigvalsh(0.5 * (quadratic + quadratic.T)) > 0.0))

    def test_synthetic_destabilized_hessian_control_is_separate(self):
        synthetic = audit._generalized_spectrum(
            np.diag((-1.0, 2.0)),
            np.eye(2),
            ["synthetic_negative", "synthetic_positive"],
        )
        self.assertLess(float(synthetic["eigenvalues_Q_over_W0"][0]), 0.0)
        self.assertNotEqual(
            self.result["status"], "SYNTHETIC_DESTABILIZED_HESSIAN"
        )

    def test_mutations_fail_closed(self):
        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["sectors"]["0"]["terminal_largest_basis"]["positive_operator"] = False
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["sectors"]["0"]["terminal_largest_basis"]["eigenvalues_Q_over_W0"][0] += 1.0
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["sectors"]["0"]["grid_sweep_largest_basis"].pop()
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["independent_energy_difference_checks_24fm"][0]["agreement"] = False
        self.assertFalse(audit.validate_result(changed))

        changed = copy.deepcopy(self.result)
        changed["evidence_weight"] = 1.0
        self.assertFalse(audit.validate_result(changed))

    def _run_case_with_fault(
        self,
        *,
        sector_fault=None,
        translation_fault=None,
        fd_fault=None,
    ):
        """Run the live aggregation path with explicit dependency fixtures."""

        baseline = self.result["cases"]["0.90"]
        backgrounds = tuple(
            _SummaryBackground(baseline["backgrounds"][box])
            for box in ("24fm", "32fm", "24fm_tighter")
        )
        sectors = {
            key: copy.deepcopy(value)
            for key, value in baseline["sectors"].items()
        }
        if sector_fault is not None:
            sector_fault(sectors)
        translation = copy.deepcopy(baseline["translation_metrics_24fm"])
        if translation_fault is not None:
            translation_fault(translation)
        fd_checks = copy.deepcopy(baseline["independent_energy_difference_checks_24fm"])
        if fd_fault is not None:
            fd_fault(fd_checks)

        def sector_provider(_background24, _background32, _background_tight24, ell):
            return copy.deepcopy(sectors[str(ell)])

        with patch.object(audit, "_backgrounds", return_value=backgrounds), \
             patch.object(audit, "_spectrum_sector", side_effect=sector_provider), \
             patch.object(audit, "_translation_metrics", return_value=translation), \
             patch.object(audit, "_fd_test_directions", return_value=[{}, {}]), \
             patch.object(audit, "_fd_energy_check", side_effect=fd_checks):
            return audit._case_result("0.90")

    def test_i4_translation_controls_fail_at_case_aggregation(self):
        mutations = (
            lambda item: item["translation_identity_residuals_relative_weighted_L2"].__setitem__(
                "scalar_equation_weighted_L2_relative", 1.0
            ),
            lambda item: item.__setitem__("translation_consistency", False),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                result = self._run_case_with_fault(translation_fault=mutate)
                self.assertFalse(result["terminal_controls_pass"])
                self.assertIn("translation_identity", result["terminal_control_failures"])
                self.assertFalse(any(
                    sector["conclusion"] in {
                        "FINITE_BASIS_POSITIVE_ONLY",
                        "FINITE_BASIS_ROBUST_NEGATIVE_DIRECTION",
                    }
                    for sector in result["sectors"].values()
                    if isinstance(sector, dict)
                ))

    def test_i4_coupled_fd_control_fails_at_case_aggregation(self):
        def fail_coupled(checks):
            next(
                check for check in checks
                if check["name"] == "coupled_zero_integral_density_scalar"
            )["agreement"] = False

        result = self._run_case_with_fault(fd_fault=fail_coupled)
        self.assertFalse(result["terminal_controls_pass"])
        self.assertIn("energy_difference_paths", result["terminal_control_failures"])
        self.assertFalse(any(
            sector["conclusion"] == "FINITE_BASIS_POSITIVE_ONLY"
            for sector in result["sectors"].values()
            if isinstance(sector, dict)
        ))

    def test_i4_missing_controls_and_invalid_negative_witness_fail_closed(self):
        missing = self._run_case_with_fault(
            sector_fault=lambda sectors: sectors.__setitem__("6", None)
        )
        self.assertFalse(missing["terminal_controls_pass"])
        self.assertIn("spectrum_l6", missing["terminal_control_failures"])
        self.assertFalse(missing["all_spectrum_records_available"])
        self.assertFalse(any(
            sector["conclusion"] in {
                "FINITE_BASIS_POSITIVE_ONLY",
                "FINITE_BASIS_ROBUST_NEGATIVE_DIRECTION",
            }
            for sector in missing["sectors"].values()
            if isinstance(sector, dict)
        ))

        def fake_negative(sectors):
            sector = sectors["0"]
            sector["conclusion"] = "FINITE_BASIS_ROBUST_NEGATIVE_DIRECTION"
            sector["basis_sweep"][0]["positive_operator"] = False

        negative = self._run_case_with_fault(sector_fault=fake_negative)
        self.assertFalse(negative["terminal_controls_pass"])
        self.assertIn("spectrum_l0", negative["terminal_control_failures"])
        self.assertFalse(any(
            sector["conclusion"] in {
                "FINITE_BASIS_POSITIVE_ONLY",
                "FINITE_BASIS_ROBUST_NEGATIVE_DIRECTION",
            }
            for sector in negative["sectors"].values()
            if isinstance(sector, dict)
        ))

    def test_i4_cli_shape_guard_rejects_injected_failed_output(self):
        damaged = copy.deepcopy(self.result)
        damaged["cases"]["0.90"]["translation_metrics_24fm"]["translation_consistency"] = False
        output = io.StringIO()
        with patch.object(audit, "build_result", return_value=damaged), \
             patch.object(audit, "validate_result", side_effect=AssertionError("recursive validation")), \
             redirect_stdout(output):
            status = audit.main([])
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 2)
        self.assertEqual(payload["status"], "INVALID_OR_FAILED_DROPLET_STABILITY_AUDIT")

    def test_i4_build_result_cache_isolation(self):
        first = audit.build_result()
        first["cases"]["0.90"]["sectors"]["0"]["basis_sweep"].clear()
        first["cases"]["0.93"]["translation_metrics_24fm"]["translation_consistency"] = False
        second = audit.build_result()
        self.assertEqual(second, self.result)
        self.assertTrue(audit.validate_result(second))

    def test_fresh_build_and_json_roundtrip_validate(self):
        self.assertTrue(audit.validate_result(audit.build_result()))
        roundtripped = json.loads(json.dumps(self.result, allow_nan=False))
        self.assertTrue(audit.validate_result(roundtripped))

    def test_cli_is_strict_json_and_no_write(self):
        script = HERE / "nvg_droplet_stability_audit.py"
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
