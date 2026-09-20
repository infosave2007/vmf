"""Focused live tests for the finite-T six-current master response."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
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

import nvg_thermal_master_response as response  # noqa: E402


class ThermalMasterResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = response.build_result()

    def test_live_coverage_and_complete_projector_rows(self):
        result = self.result
        self.assertEqual(result["status"], response.STATUS_PASS)
        self.assertTrue(response.validate_result(result))
        self.assertEqual(result["coverage"]["state_count"], 8)
        self.assertEqual(result["coverage"]["response_row_count"], 160)
        self.assertEqual(result["coverage"]["thermal_cut_row_count"], 32)
        self.assertEqual(len(result["response_rows"]), 160)
        self.assertEqual(len(result["thermal_cut_rows"]), 32)
        self.assertEqual({row["z_label"] for row in result["response_rows"]}, set(response.Z_LABELS))
        self.assertEqual({row["status"] for row in result["response_rows"]}, {"MASTER_RESPONSE_OFF_POLE"})
        for row in result["response_rows"]:
            self.assertEqual(set(row["Pi"]), {"vv_MeV2", "vs_MeV2", "ss_MeV2", "vL_MeV2", "sL_MeV2", "LL_MeV2"})
            self.assertEqual(len(row["response"]["H"]), 4)
            self.assertTrue(row["negative_vector_saddle_retained"])

    def test_controls_attack_actual_physics(self):
        controls = self.result["controls"]
        for name in (
            "static_limit_bridge", "q0_order_of_limits", "raw_six_current_Ward",
            "particle_hole_only_negative_control", "charge_conjugation_antimaterial",
            "frequency_conjugation", "direct_4x4_vs_schur", "scale_applied_once",
            "quadrature_response_refinement", "low_temperature_old_cold_UHP",
            "thermal_cut_cold_limit_and_warm_tail", "thermal_cut_independent_smearing",
            "thermal_cut_low_temperature_edge",
            "input_and_singular_failure_controls",
        ):
            self.assertTrue(controls[name]["pass"], name)
        self.assertEqual(controls["raw_six_current_Ward"]["row_count"], 160)
        self.assertEqual(len(controls["static_limit_bridge"]["rows"]), 32)
        self.assertGreater(controls["particle_hole_only_negative_control"]["Ward_max_relative"], 1e-4)
        self.assertTrue(controls["thermal_cut_cold_limit_and_warm_tail"]["independent_smearing_not_final_chi"])
        self.assertEqual(controls["quadrature_response_refinement"]["unique_Pi_count"], 80)
        self.assertEqual(controls["quadrature_response_refinement"]["response_check_count"], 160)

    def test_low_temperature_same_band_log_ratio_and_cut_edge(self):
        # This API probe sits on the particle Fermi edge at T=.1 MeV.  The
        # exact divided difference is 1/(900-700), not an inf/inf zero.
        particle = response._same_band_fermi_divided_difference(
            np.asarray([700.0]), np.asarray([900.0]), s=1, nu=800.0, temperature=0.1,
        )
        antiparticle = response._same_band_fermi_divided_difference(
            np.asarray([700.0]), np.asarray([900.0]), s=-1, nu=-800.0, temperature=0.1,
        )
        self.assertAlmostEqual(float(particle[0]), 5.0e-3, places=14)
        self.assertAlmostEqual(float(antiparticle[0]), 5.0e-3, places=14)

        direct = response.thermal_cut_imag(700.0, 1200.0, 100.0, 25.0, 0.1)
        cold = response.hessian.analytic_ph_imag(
            700.0, math.sqrt(1200.0 ** 2 - 700.0 ** 2), 100.0, 25.0, d=response.DEGENERACY,
        )
        self.assertLess(max(abs(direct[idx] - cold[idx]) / max(abs(cold[idx]), 1e-8) for idx in range(3)), 3e-6)
        reflected = response.thermal_cut_imag(700.0, -1200.0, 100.0, 25.0, 0.1)
        self.assertLess(abs(direct[0] - reflected[0]) / max(abs(direct[0]), 1e-8), 3e-10)
        self.assertLess(abs(direct[1] + reflected[1]) / max(abs(direct[1]), 1e-8), 3e-10)
        self.assertLess(abs(direct[2] - reflected[2]) / max(abs(direct[2]), 1e-8), 3e-10)
        # Here Emin lies above the Fermi edge, so the all-tail underflow is a
        # genuine explicit failure rather than a false endpoint rejection.
        with self.assertRaises(response.ThermalMasterResponseError):
            response.thermal_cut_imag(700.0, 760.0, 100.0, 75.0, 0.1)

    def test_q0_limits_and_matsubara_are_not_overclaimed(self):
        q0 = self.result["controls"]["q0_order_of_limits"]
        self.assertTrue(q0["pass"])
        self.assertTrue(all(row["dynamic_scalar_pair_abs"] > 0 for row in q0["rows"]))
        diagnostic = self.result["controls"]["Matsubara_positive_sample_diagnostic"]
        self.assertTrue(diagnostic["not_a_global_positivity_or_covariance_proof"])
        text = json.dumps(self.result, ensure_ascii=False)
        self.assertIn("not a summed covariance", text)
        self.assertNotIn("physical_width", text)
        self.assertNotIn("HADES agreement", text)

    def test_q0_ll_and_same_band_qsmall_regressions(self):
        rows = self.result["controls"]["q0_order_of_limits"]["rows"]
        self.assertTrue(all(row["LL_vs_1d_relative_error"] < 3e-11 for row in rows))
        self.assertTrue(all(row["angular_vs_1d_relative_error"] < 3e-11 for row in rows))
        self.assertTrue(all(max(row["qsmall_dynamic_density_abs"]) < 1e-5 for row in rows))
        for nu in (0, 70, -70):
            q0 = response.finite_temperature_master_kernel(500, nu, 0, 70, 0j)
            qsmall = response.finite_temperature_master_kernel(500, nu, 1e-13, 70, 0j, p_order=220, u_order=180, cutoff=6000)
            np_rel = [abs(qsmall[idx] - q0[idx]) / max(abs(q0[idx]), 1e-20) for idx in range(3)]
            self.assertLess(max(np_rel), 1e-8)

    def test_cut_is_bare_and_finite_eta_is_forbidden(self):
        for row in self.result["thermal_cut_rows"]:
            self.assertTrue(row["bare_polarization_only"])
            self.assertFalse(row["finite_eta"])
            self.assertEqual(row["status"], "FINITE_T_SPACELIKE_ABSORPTION")
            self.assertTrue(all(float(v) >= 0 for v in row["Pi_absorptive_Im_MeV2"].values()))
            self.assertTrue(all(float(v) >= 0 for v in row["integration_error_estimate_MeV2"].values()))
        with self.assertRaises(response.ThermalMasterResponseError):
            response.thermal_cut_imag(700, 760, 100, 100, 70)
        with self.assertRaises(response.ThermalMasterResponseError):
            response.finite_temperature_master_kernel(700, 760, 100, 70, 75.0)
        with self.assertRaises(response.ThermalMasterResponseError):
            response.thermal_cut_imag(700, 760, 100, 75, 0.1)
        self.assertTrue(self.result["controls"]["thermal_cut_independent_smearing"]["pass"])

    def test_api_guards_and_mutation_rejection(self):
        for args in ((0, 760, 100, 70, 1j), (700, 760, -1, 70, 1j), (700, 760, 100, 0, 1j), (700, 760, 100, 70, 75.0)):
            with self.assertRaises(response.ThermalMasterResponseError):
                response.finite_temperature_master_kernel(*args)
        candidate = copy.deepcopy(self.result)
        candidate["response_rows"].pop()
        self.assertFalse(response.validate_result(candidate))

    def test_cli_returns_nonzero_for_failed_gate(self):
        failed = {"schema_version": response.SCHEMA_VERSION, "status": response.STATUS_FAIL}
        output = io.StringIO()
        with mock.patch.object(response, "build_result", return_value=failed), redirect_stdout(output):
            self.assertEqual(response.main([]), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], response.STATUS_FAIL)
        output = io.StringIO()
        with mock.patch.object(response, "build_result", side_effect=response.ThermalMasterResponseError("forced refinement failure")), redirect_stdout(output):
            self.assertEqual(response.main([]), 1)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], response.STATUS_FAIL)
        self.assertIn("forced refinement failure", payload["error"])
        candidate = copy.deepcopy(self.result)
        candidate["integrity_sha256"] = "0" * 64
        self.assertFalse(response.validate_result(candidate))

    def test_source_boundary_and_live_cli(self):
        source = (HERE / "nvg_thermal_master_response.py").read_text(encoding="utf-8")
        self.assertNotIn("Lunacy/runs", source)
        self.assertNotIn("evidence_parent", source)
        self.assertEqual(self.result["source_sha256"]["producer"], hashlib.sha256((HERE / "nvg_thermal_master_response.py").read_bytes()).hexdigest())
        completed = subprocess.run(
            [sys.executable, str(HERE / "nvg_thermal_master_response.py")],
            check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(HERE)},
        )
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["coverage"]["response_row_count"], 160)
        self.assertEqual(parsed["coverage"]["thermal_cut_row_count"], 32)
        self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
