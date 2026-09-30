"""Focused live tests for the finite-temperature static spatial response."""
from __future__ import annotations

import json
import copy
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_thermal_spatial_response as response  # noqa: E402


class ThermalSpatialResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is deliberately a live producer run; the checked-in result
        # artifact is never read as a physics input.
        cls.result = response.build_result()

    def test_live_64_row_contract_and_no_covariance_claim(self):
        result = self.result
        self.assertEqual(result["status"], response.STATUS_PASS)
        self.assertTrue(response.validate_result(result))
        self.assertEqual(result["coverage"]["row_count"], 64)
        self.assertEqual(len(result["states"]), 8)
        self.assertEqual(len(result["rows"]), response.EXPECTED_ROW_COUNT)
        self.assertEqual(set(row["status"] for row in result["rows"]), {"STABLE_STATIC_TF_RESPONSE"})
        self.assertTrue(all(row["raw_vector_saddle_negative"] for row in result["rows"]))
        self.assertTrue(all(not row["curvature_controls"]["raw_3x3_positive_definite_required"] for row in result["rows"]))
        text = json.dumps(result, ensure_ascii=False)
        self.assertIn("equal-time quantum density covariance", text)
        self.assertNotIn("fit_score", text)
        self.assertNotIn("refolded_count_variance", text)

    def test_independent_controls_are_live_and_cover_antiparticle_scalar_and_cold_limits(self):
        controls = self.result["controls"]
        self.assertTrue(controls["q0_thermostatic_match"]["pass"])
        self.assertEqual(len(controls["q0_thermostatic_match"]["rows"]), 8)
        self.assertTrue(controls["quadrature_refinement"]["pass"])
        self.assertTrue(all("response_relative_differences" in row for row in controls["quadrature_refinement"]["rows"]))
        self.assertTrue(controls["qsmall_continuity"]["pass"])
        self.assertTrue(controls["charge_conjugation_parity"]["pass"])
        self.assertTrue(controls["low_temperature_old_cold_kernel"]["pass"])
        self.assertTrue(controls["antiparticle_material_control"]["pass"])
        self.assertTrue(all(row["omitted_antiparticle_vv_relative_difference"] > 0.5 for row in controls["antiparticle_material_control"]["rows"]))
        scalar = controls["scalar_negative_energy_control"]
        self.assertTrue(scalar["detected_and_rejected"])
        self.assertGreater(float(scalar["omitted_I_MeV2"]), 0.0)
        self.assertTrue(controls["synthetic_branch_controls"]["pass"])
        self.assertTrue(all(row["chi_is_null"] for row in controls["synthetic_branch_controls"]["rows"]))

    def test_same_q_tf_and_q0_ratios_are_explicit(self):
        for row in self.result["rows"]:
            for key in ("chi_nonlocal_MeV2", "chi_TF_same_q_MeV2", "chi_q0_MeV2"):
                self.assertIsNotNone(row[key])
            if row["q_MeV"] == "0":
                self.assertAlmostEqual(float(row["ratio_nonlocal_to_TF"]), 1.0, places=12)
                self.assertAlmostEqual(float(row["ratio_nonlocal_to_q0"]), 1.0, places=12)
            else:
                self.assertLess(float(row["ratio_nonlocal_to_q0"]), 1.0001)

    def test_independent_direct_3x3_solve_and_residual(self):
        # Re-solve a bounded sample with NumPy rather than trusting the product
        # direct-solve field or the Schur algebra alone.
        for row in self.result["rows"][::11]:
            h = np.asarray(row["raw_hessian"], dtype=float)
            # H_ny and the Gauss-reduced mixed curvature carry units of MeV.
            self.assertIn("B_MeV", row["curvature_controls"])
            self.assertNotIn("B_dimensionless", row["curvature_controls"])
            direct = np.asarray([float(x) for x in row["direct_solution"]], dtype=float)
            expected = np.linalg.solve(h, np.array([1.0, 0.0, 0.0]))
            np.testing.assert_allclose(direct, expected, rtol=3e-12, atol=3e-12)
            residual = h @ direct - np.array([1.0, 0.0, 0.0])
            self.assertLess(float(np.max(np.abs(residual))), 2e-8)
            self.assertEqual(row["direct_status"], "DIRECT_SOLVE_OK")

    def test_scales_are_retained_once_and_q0_invariant(self):
        for state in self.result["states"]:
            self.assertTrue(all(item["pass"] for item in state["scale_controls"]))
        for state_key in ((a, t, mu, model) for a in response.ANCHORS for t in response.TEMPERATURES for mu in response.CHEMICAL_POTENTIALS for model in response.MODEL_ORDER):
            a, t, mu, model = state_key
            rows = [row for row in self.result["rows"] if (row["anchor"], row["T_MeV"], row["mu_MeV"], row["model"]) == state_key and row["q_MeV"] == "0"]
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[0]["chi_nonlocal_MeV2"]), float(rows[1]["chi_nonlocal_MeV2"]), places=10)
        scale_control = self.result["controls"]["nonzero_q_scale_increment"]
        self.assertTrue(scale_control["pass"])
        self.assertEqual(len(scale_control["rows"]), 24)
        self.assertTrue(all(float(item["relative_difference"]) < 2e-9 for item in scale_control["rows"]))
        self.assertTrue(self.result["controls"]["independent_direct_solve_both_scales"]["pass"])
        self.assertEqual(len(self.result["controls"]["independent_direct_solve_both_scales"]["rows"]), 64)

    def test_input_guards_and_public_source_boundary(self):
        for args in (("0", "1", "0", "70"), ("1", "1", "-1", "70"), ("1", "1", "1", "0"), ("1", "NaN", "1", "70")):
            with self.assertRaises(response.ThermalSpatialResponseError):
                response.finite_temperature_kernel(*args)
        source = (HERE / "nvg_thermal_spatial_response.py").read_text(encoding="utf-8")
        self.assertNotIn("Lunacy/runs", source)
        self.assertNotIn("evidence_parent", source)
        self.assertNotIn("thermal-spatial-correlations-2026-09-19", source)
        self.assertEqual(self.result["source_sha256"]["producer"], hashlib.sha256((HERE / "nvg_thermal_spatial_response.py").read_bytes()).hexdigest())
        self.assertEqual(self.result["source_sha256"]["thermal_bridge"], hashlib.sha256(Path(response.thermal.__file__).resolve().read_bytes()).hexdigest())
        self.assertEqual(self.result["source_sha256"]["nonlocal_response"], hashlib.sha256(response.cold_response.SOURCE_PATH.read_bytes()).hexdigest())

    def test_validator_rejects_bounded_mutations_and_digest_drift(self):
        mutated = copy.deepcopy(self.result)
        mutated["rows"][1]["row_id"] = mutated["rows"][0]["row_id"]
        self.assertFalse(response.validate_result(mutated))
        mutated = copy.deepcopy(self.result)
        mutated["controls"]["q0_thermostatic_match"]["pass"] = False
        self.assertFalse(response.validate_result(mutated))
        mutated = copy.deepcopy(self.result)
        mutated["rows"][0]["Pi_vv_MeV2"] = "NaN"
        self.assertFalse(response.validate_result(mutated))
        mutated = copy.deepcopy(self.result)
        mutated["integrity_sha256"] = "0" * 64
        self.assertFalse(response.validate_result(mutated))

    def test_cli_returns_nonzero_for_failed_gate(self):
        failed = {"schema_version": response.SCHEMA_VERSION, "status": response.STATUS_FAIL}
        output = io.StringIO()
        with mock.patch.object(response, "build_result", return_value=failed), redirect_stdout(output):
            self.assertEqual(response.main([]), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], response.STATUS_FAIL)

    def test_cli_result_is_strict_json_and_does_not_read_fixed_artifact(self):
        completed = subprocess.run(
            [sys.executable, str(HERE / "nvg_thermal_spatial_response.py")],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(HERE)},
        )
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["coverage"]["row_count"], response.EXPECTED_ROW_COUNT)
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
