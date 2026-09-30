"""Focused source-only tests for the NVG field-energy transfer calculation."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_field_energy_transfer as transfer  # noqa: E402


class FieldEnergyTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = transfer.build_result()

    @staticmethod
    def mp(value: object) -> mp.mpf:
        with mp.workdps(90):
            return mp.mpf(str(value))

    def test_live_schema_coverage_lineage_and_strict_finite_json(self) -> None:
        result = self.result
        self.assertEqual(result["schema_version"], transfer.SCHEMA_VERSION)
        self.assertEqual(result["status"], transfer.STATUS)
        self.assertEqual(result["evidence_weight"], 0.0)
        self.assertTrue(transfer.validate_result(result))
        self.assertEqual(result["coverage"]["row_count"], transfer.EXPECTED_ROW_COUNT)
        self.assertTrue(result["coverage"]["all_rows_present"])
        self.assertTrue(result["coverage"]["all_statuses_retained"])
        self.assertEqual(
            {(row["family"], row["target_y"], row["q_MeV"]) for row in result["rows"]},
            {
                ("Q4", None, q + ".0")
                for q in transfer.Q_GRID_MEV
            }
            | {
                ("W8", target, q + ".0")
                for target in transfer.W8_TARGETS
                for q in transfer.Q_GRID_MEV
            },
        )
        self.assertIn("inverse_potential_jet", result["branch_lineage"]["W8.90"])
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertEqual(json.loads(payload)["coverage"], result["coverage"])
        for forbidden in (".codex", ".work", "private", "hidden"):
            self.assertNotIn(forbidden, payload)

    def test_source_rows_match_maintained_response_and_keep_transfer_identity(self) -> None:
        rows = self.result["rows"]
        self.assertTrue(all(row["response"]["direct_status"] == "DIRECT_SOLVE_OK" for row in rows))
        self.assertTrue(all(row["response"]["stable"] for row in rows))
        for row in rows:
            response = row["response"]
            transfer_data = row["energy_transfer"]
            self.assertLessEqual(self.mp(response["direct_vs_schur_relative"]), mp.mpf("1e-35"))
            self.assertLessEqual(self.mp(transfer_data["r2_identity_relative"]), mp.mpf("1e-35"))
            # The relaxation coefficient is the difference between frozen and
            # relaxed quadratic coefficients; it is not called an efficiency.
            frozen = self.mp(transfer_data["frozen_coefficient_per_abs_dn2"])
            relaxed = self.mp(transfer_data["relaxed_coefficient_per_abs_dn2"])
            released = self.mp(transfer_data["relaxation_coefficient_per_abs_dn2"])
            with mp.workdps(90):
                self.assertLessEqual(abs((frozen - relaxed) - released), mp.mpf("1e-30"))
            self.assertTrue(transfer_data["not_a_power_or_efficiency"])

    def test_completion_square_finite_difference_and_coordinate_controls_pass(self) -> None:
        controls = self.result["controls"]
        for key in (
            "direct_inverse_vs_schur",
            "completion_of_square",
            "finite_difference_scalar_minimum",
            "coordinate_rescaling_invariance",
        ):
            self.assertTrue(controls[key]["pass"], key)
        self.assertTrue(controls["zero_coupling"]["pass"])
        self.assertTrue(controls["near_soft_stable_limit"]["pass"])
        self.assertTrue(controls["invalid_nonfinite_guards"]["pass"])

    def test_unstable_rows_are_visible_and_r2_is_not_clipped(self) -> None:
        row = transfer.reduced_response_from_hessian(
            {"a": "1", "b": "2", "d0": "-5", "g": "1", "h": "0", "t0": "1"},
            "4",
            "0",
        )
        self.assertEqual(row["status"], "UNSTABLE_SCALAR_CURVATURE")
        self.assertFalse(row["stable"])
        self.assertLess(row["r2"], 0)  # no clipping into a cosmetic [0,1] interval
        self.assertIsNotNone(row["direct_chi"])
        self.assertLess(abs(row["r2"] - row["r2_from_direct_chi"]), mp.mpf("1e-45"))

    def test_zero_coupling_and_near_soft_stable_limit(self) -> None:
        zero = transfer.reduced_response_from_hessian(
            {"a": "2", "b": "0", "d0": "1", "g": "0", "h": "0", "t0": "1"},
            "4",
            "0",
        )
        self.assertEqual(zero["r2"], 0)
        self.assertEqual(zero["S"], zero["D"])
        soft = transfer.reduced_response_from_hessian(
            {"a": "1", "b": "0.9", "d0": "1", "g": "0", "h": "0", "t0": "1"},
            "4",
            "0",
        )
        self.assertTrue(soft["stable"])
        self.assertGreater(soft["r2"], 0)
        self.assertLess(soft["r2"], 1)
        self.assertGreater(soft["S"], 0)

    def test_invalid_nonfinite_and_indefinite_guards(self) -> None:
        with self.assertRaises(transfer.FieldEnergyTransferError):
            transfer.reduced_response_from_hessian(
                {"a": "NaN", "b": "0", "d0": "1", "g": "0", "h": "0", "t0": "1"},
                "4",
                "0",
            )
        with self.assertRaises(transfer.FieldEnergyTransferError):
            transfer.reduced_response_from_hessian(
                {"a": "1", "b": "0", "d0": "1", "g": "0", "h": "0", "t0": "1"},
                "0",
                "0",
            )
        with self.assertRaises(transfer.FieldEnergyTransferError):
            transfer.rescale_reduced_coefficients("1", "0", "1", "0", "1")

    def test_source_sensitivity_proves_no_hardcoded_response_table(self) -> None:
        baseline = transfer.calculate(50)
        original = transfer.maintained.finite_q_coefficients

        def perturbed(model, state, q, scale="1"):
            coefficients = dict(original(model, state, q, scale))
            coefficients["Pi_vv"] = coefficients["Pi_vv"] * mp.mpf("1.01")
            coefficients["a"] = 1 / coefficients["Pi_vv"]
            coefficients["b"] = coefficients["b"] / mp.mpf("1.01")
            coefficients["dferm"] = coefficients["dferm"] / mp.mpf("1.01")
            coefficients["d0"] = coefficients["dferm"] + state["Uyy"] - model.momega**2 * coefficients["A0"]**2
            return coefficients

        with patch.object(transfer.maintained, "finite_q_coefficients", side_effect=perturbed):
            changed = transfer.calculate(50)
        self.assertNotEqual(
            baseline["rows"][0]["response"]["D_MeVminus2"],
            changed["rows"][0]["response"]["D_MeVminus2"],
        )

    def test_cli_is_strict_json_no_implicit_result_write_and_explicit_output(self) -> None:
        result_path = HERE / "nvg_field_energy_transfer_results.json"
        existed = result_path.exists()
        before = result_path.read_bytes() if existed else None
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(HERE)}
        completed = subprocess.run(
            [sys.executable, str(transfer.SOURCE_PATH), "--dps", "40"],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["coverage"]["row_count"], transfer.EXPECTED_ROW_COUNT)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(result_path.exists(), existed)
        if existed:
            self.assertEqual(result_path.read_bytes(), before)

        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "transfer.json"
            subprocess.run(
                [sys.executable, str(transfer.SOURCE_PATH), "--dps", "40", "--output", str(explicit)],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertTrue(explicit.exists())
            self.assertEqual(json.loads(explicit.read_text(encoding="utf-8"))["coverage"]["row_count"], transfer.EXPECTED_ROW_COUNT)


if __name__ == "__main__":
    unittest.main()
