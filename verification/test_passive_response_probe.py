"""Focused contract tests for the passive-RLC JSON probe.

The full deterministic operating-characteristic benchmark is run separately by
the P1-A terminal command.  These tests keep malformed-input, sign,
fair-baseline, and finite-band interpretation gates fast and explicit.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_passive_response_probe as probe


FIXTURE = ROOT / "verification/data/passive_response_example_2026.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class ConventionTests(unittest.TestCase):
    def test_series_model_uses_exp_minus_iomega_sign(self) -> None:
        y = probe.series_rlc_admittance([20.0, 2.0e6], 10.0, 1.0e-2, 1.0e-6)
        self.assertLess(y[0].imag, 0.0)
        self.assertGreater(y[1].imag, 0.0)
        self.assertGreaterEqual(y.real.min(), 0.0)

    def test_absorbed_power_identity_and_conjugate_representation(self) -> None:
        v = 2.4 - 1.7j
        y = 0.13 + 0.42j
        p = probe.absorbed_power_watts(v, y)
        self.assertAlmostEqual(p, 0.5 * abs(v) ** 2 * y.real, places=14)
        self.assertAlmostEqual(
            p,
            probe.absorbed_power_watts(np.conjugate(v), np.conjugate(y)),
            places=14,
        )

    def test_absorbed_power_api_is_explicitly_peak_normalized(self) -> None:
        self.assertIn("voltage_peak_v", str(__import__("inspect").signature(probe.absorbed_power_watts)))
        v_rms = 2.0
        y = 0.5 + 0.1j
        self.assertAlmostEqual(
            probe.absorbed_power_watts(math.sqrt(2.0) * v_rms, y),
            v_rms**2 * y.real,
            places=14,
        )

    def test_parallel_endpoint_identity_is_declared(self) -> None:
        y, c, l, meta = probe._synthetic_truth("extra_positive_branch", np.asarray([100.0, 1000.0]))
        self.assertEqual(meta["family"], "parallel_two_positive_series_rlc_branches")
        self.assertGreater(c, probe.REFERENCE_C)
        self.assertLess(l, probe.REFERENCE_L)
        self.assertTrue(np.all(np.isfinite(y)))


class InputValidationTests(unittest.TestCase):
    def test_fixture_is_valid_and_constraints_are_independent(self) -> None:
        measurement = probe.parse_input(_fixture())
        self.assertEqual(measurement.n, 12)
        self.assertEqual(measurement.units["imaginary_sign"], probe.IMAGINARY_SIGN)
        self.assertEqual(measurement.capacitance.unit, "F")
        self.assertEqual(measurement.inductance.unit, "H")

    def test_duplicate_frequency_fails_closed(self) -> None:
        payload = _fixture()
        payload["measurement"]["frequencies_hz"][2] = payload["measurement"]["frequencies_hz"][1]
        with self.assertRaises(probe.PassiveResponseError):
            probe.parse_input(payload)

    def test_nonfinite_frequency_fails_closed(self) -> None:
        payload = _fixture()
        payload["measurement"]["frequencies_hz"][0] = "NaN"
        with self.assertRaises(probe.PassiveResponseError):
            probe.parse_input(payload)

    def test_unit_and_polarity_controls_fail_closed(self) -> None:
        for field, value in (("frequency", "rad/s"), ("admittance", "mS")):
            payload = _fixture()
            payload["units"][field] = value
            with self.subTest(field=field):
                with self.assertRaises(probe.PassiveResponseError):
                    probe.parse_input(payload)
        payload = _fixture()
        payload["units"]["imaginary_sign"] = "positive_for_capacitive"
        with self.assertRaises(probe.PassiveResponseError):
            probe.parse_input(payload)

    def test_endpoint_inference_is_not_accepted_as_independent(self) -> None:
        payload = _fixture()
        payload["constraint_metadata"]["independent"] = False
        payload.pop("constraints_independently_measured", None)
        with self.assertRaises(probe.PassiveResponseError):
            probe.parse_input(payload)

    def test_noise_scaling_and_negative_noise_controls(self) -> None:
        payload = _fixture()
        payload["noise"] = {"model": "independent_gaussian_cartesian", "relative_sigma": 0.02}
        measurement = probe.parse_input(payload)
        self.assertTrue(np.all(measurement.sigma_real_s > 0.0))
        bad = _fixture()
        bad["noise"]["sigma_real_S"] = -1.0
        with self.assertRaises(probe.PassiveResponseError):
            probe.parse_input(bad)


class AnalysisTests(unittest.TestCase):
    def test_reference_analysis_is_finite_and_reports_both_methods(self) -> None:
        result = probe.analyze(_fixture())
        self.assertEqual(result["schema"], probe.RESULT_SCHEMA)
        self.assertEqual(result["status"], probe.STATUS)
        self.assertIn("sweep_only_baseline", result)
        self.assertIn("constrained_fit", result)
        self.assertTrue(result["constrained_fit"]["success"])
        self.assertTrue(result["diagnostics"]["finite_band_unidentifiability"])
        self.assertTrue(result["diagnostics"]["unknown_positive_modes_outside_band_not_ruled_out"])
        json.dumps(result, allow_nan=False)

    def test_known_sigma_covariance_is_not_residual_scaled_and_intervals_are_nonzero(self) -> None:
        measurement = probe.parse_input(_fixture())
        fit = probe._fit(measurement, include_endpoints=True, method="test_joint")
        self.assertEqual(fit.covariance_status, "valid_known_noise_log10_local")
        expected = np.linalg.solve(fit.jacobian.T @ fit.jacobian, np.eye(3))
        np.testing.assert_allclose(fit.covariance_log, expected, rtol=2e-7, atol=1e-22)
        result = probe.analyze(_fixture())
        for entry in result["constrained_fit"]["conditional_parameter_intervals_95"].values():
            self.assertLess(entry["low_95"], entry["estimate"])
            self.assertGreater(entry["high_95"], entry["estimate"])
            self.assertEqual(entry["interval_status"], "valid_known_noise_log10_local")

    def test_rank_deficient_jacobian_flags_intervals_without_pseudoinverse(self) -> None:
        original = probe.least_squares

        def rank_deficient(*args, **kwargs):
            candidate = original(*args, **kwargs)
            candidate.jac = np.zeros_like(candidate.jac)
            return candidate

        with patch.object(probe, "least_squares", rank_deficient):
            result = probe.analyze(_fixture())
        for method in ("sweep_only_baseline", "constrained_fit"):
            fit = result[method]
            self.assertEqual(fit["covariance_status"], "invalid_rank")
            self.assertIsNone(fit["conditional_parameter_intervals_95"]["R_ohm"]["low_95"])

    def test_nonconverged_candidates_are_indeterminate_not_compatible(self) -> None:
        original = probe.least_squares

        def nonconverged(*args, **kwargs):
            candidate = original(*args, **kwargs)
            candidate.success = False
            return candidate

        with patch.object(probe, "least_squares", nonconverged):
            result = probe.analyze(_fixture())
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["decision"], "INDETERMINATE_NUMERICAL_FIT")
        self.assertNotIn("diagnostics", result)

    def test_actual_objective_dof_and_alpha_are_exposed_per_method(self) -> None:
        result = probe.analyze(_fixture(), alpha=0.05)
        self.assertEqual(result["sweep_only_baseline"]["degrees_of_freedom"], 21)
        self.assertEqual(result["constrained_fit"]["degrees_of_freedom"], 23)
        self.assertEqual(result["sweep_only_baseline"]["decision_alpha"], 0.05)
        self.assertEqual(result["constrained_fit"]["decision_alpha"], 0.05)
        self.assertNotIn("decision_threshold_chi2_99", result["sweep_only_baseline"])

    def test_valid_overflow_input_is_strict_json_not_traceback(self) -> None:
        path = ROOT / "Lunacy/runs/practical-prototypes-2026-09-19/evidence_R/a_valid_overflow_input.json"
        if not path.is_file():
            self.skipTest("named adversarial input is not present in this checkout")
        completed = subprocess.run(
            [sys.executable, str(HERE / "nvg_passive_response_probe.py"), "--input", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertIn(completed.returncode, (0, 2))
        self.assertEqual(completed.stderr, "")
        document = json.loads(completed.stdout)
        self.assertIn(document["status"], (probe.STATUS, "FAIL"))
        json.dumps(document, allow_nan=False)

    def test_truth_labels_do_not_change_classification(self) -> None:
        first = probe.analyze(_fixture())
        payload = _fixture()
        payload["scenario"] = "active_nonpassive_control"
        payload["truth"] = {"R_ohm": -10.0}
        second = probe.analyze(payload)
        self.assertEqual(first["diagnostics"]["classification"], second["diagnostics"]["classification"])
        self.assertEqual(first["constrained_fit"]["score"], second["constrained_fit"]["score"])

    def test_phase_fault_and_active_control_do_not_get_passed_as_reference(self) -> None:
        phase = probe.analyze(probe._make_payload("phase_fault", 2026093004, noisy=True))
        active = probe.analyze(probe._make_payload("active_nonpassive_control", 2026093007, noisy=True))
        self.assertFalse(phase["constrained_fit"]["pass_single_series_rlc"])
        self.assertTrue(active["diagnostics"]["passivity"]["material_negative_real_admittance"])

    def test_finite_band_control_is_explicitly_not_a_nonexistence_claim(self) -> None:
        result = probe.analyze(probe._make_payload("finite_band_indistinguishable", 2026093006, noisy=True))
        self.assertTrue(result["diagnostics"]["finite_band_unidentifiability"])
        self.assertTrue(result["diagnostics"]["unknown_positive_modes_outside_band_not_ruled_out"])
        self.assertIn("finite_band", result["diagnostics"]["interpretation_limit"])

    def test_invalid_json_cli_document_is_fail_not_pass(self) -> None:
        document = probe._error_document(probe.PassiveResponseError("bad input"))
        self.assertEqual(document["status"], "FAIL")
        self.assertNotEqual(document["status"], probe.STATUS)
        json.dumps(document, allow_nan=False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
