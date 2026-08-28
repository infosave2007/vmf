"""Semantic regression checks for the direct-report repair (P2-S1)."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_black_hole_entropy as entropy
import nvg_hyperon_puzzle_tov as hyperon
import nvg_joint_ns_inference as joint
import run_nvg_suite as ledger
import nvg_verification_suite as suite


class DirectReportIntegrityTests(unittest.TestCase):
    def test_entropy_summary_uses_computed_s8_sign_and_downgrades_qs(self):
        state = entropy.compute_tests()
        s8 = state["P"]
        self.assertGreater(s8["sigma8_ratio"], 1.0)
        self.assertGreater(s8["remaining_tension_sigma"], s8["initial_tension_sigma"])
        self.assertEqual(s8["direction"], "away from weak-lensing value")
        self.assertIn("WORSENS", s8["status"])
        self.assertIn("ILLUSTRATIVE ONLY", state["Q"]["status"])
        self.assertIn("UNSUPPORTED", state["S"]["status"])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            entropy._print_report(state)
        rendered = output.getvalue()
        self.assertIn(f"Computed S8 = {s8['s8_nvg']:.3f}", rendered)
        self.assertIn("Direction from computed distances: away from weak-lensing value", rendered)
        self.assertNotIn("correct direction", rendered.lower())
        self.assertNotIn("natural solution", rendered.lower())
        self.assertNotIn("Kerr-Hayward OK", rendered)

    def test_verification_suite_consumes_canonical_eos_and_echo_outputs(self):
        state = suite.compute_suite_state()
        eos = state["eos"]
        echo = state["echo"]
        self.assertEqual(eos["solver"], "nvg_tidal_deformability.EOS + solve_tov_tidal")
        self.assertGreater(eos["M_max"], 0.0)
        self.assertGreater(eos["R_1.4"], 0.0)
        self.assertGreater(eos["Lambda_1.4"], 0.0)
        self.assertEqual(echo["solver"], "nvg_gw_echo_prediction.calculate_kerr_echo_delay")
        self.assertGreater(echo["delay_s"], 0.0)
        details = "\n".join(check["details"] for check in state["checks"])
        self.assertIn(f"{eos['M_max']:.3f}", details)
        self.assertIn(f"{echo['delay_s']:.5f}", details)

    def test_joint_calibration_row_cannot_change_independent_chi_squared(self):
        predictions = {"M_max": 2.0, "R_1.4": 12.0, "Lambda_1.4": 300.0, "Cooling_Dichotomy": None}
        result = joint.run_joint_inference(predictions, joint.ROW_METADATA)
        cooling = next(row for row in result["rows"] if row["key"] == "Cooling_Dichotomy")
        self.assertFalse(cooling["included"])
        self.assertIsNone(cooling["pull"])
        self.assertEqual(result["dof"], 3)
        expected = ((2.0 - 2.14) / 0.10) ** 2 + ((12.0 - 12.2) / 0.50) ** 2 + ((300.0 - 190.0) / 390.0) ** 2
        self.assertAlmostEqual(result["chi_squared_total"], expected)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            joint._print_report(result)
        rendered = output.getvalue()
        self.assertIn("excluded (no independent solver)", rendered)
        self.assertIn(f"Reduced chi-squared            : {result['reduced_chi']:.3f}", rendered)

    def test_ledger_rows_are_runtime_values_and_inverse_is_closed(self):
        result = ledger.run_forward_model()
        rows = ledger.generate_evidence_ledger(result)
        self.assertTrue(any(f"{result['m_max']:.3f}" in row["value"] for row in rows))
        self.assertTrue(any(f"{result['r_14']:.3f}" in row["value"] for row in rows))
        self.assertTrue(any(f"{result['lambda_14']:.1f}" in row["value"] for row in rows))
        self.assertEqual(ledger.solve_inverse_qcd(500.0, 2.15)["status"].split(":", 1)[0], "UNSUPPORTED")
        self.assertIn("Cooling_Dichotomy", result["calibrated_rows"])

    def test_hyperon_summary_tracks_raw_curve_maxima(self):
        curves = {
            "baseline": ([10.0, 11.0, 12.0], [1.0, 2.0, 1.8]),
            "modified": ([9.0, 10.0, 11.0], [0.8, 1.4, 1.2]),
        }
        summary = hyperon.summarize_curves(curves)
        self.assertEqual(summary["baseline"][0], 2.0)
        self.assertEqual(summary["baseline"][1], 11.0)
        self.assertEqual(summary["modified"][0], 1.4)
        self.assertEqual(summary["modified"][1], 10.0)


if __name__ == "__main__":
    unittest.main()
