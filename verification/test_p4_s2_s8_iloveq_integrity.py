"""Semantic and artifact checks for the Phase 4 S8/I--Love--Q repair."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_iloveq_plot as iloveq
import nvg_s8_tension_check as s8


class S8ILoveQIntegrityTests(unittest.TestCase):
    def test_s8_wrapper_consumes_maintained_worsening_result(self) -> None:
        result = s8.compute_s8_result()
        self.assertEqual(result["source"], "nvg_black_hole_entropy.compute_s8_test")
        self.assertEqual(result["wrapper_status"], "CONNECTED_MAINTAINED_ROUTE_NO_INDEPENDENT_PREDICTION")
        self.assertIn("WORSENS", result["status"])
        self.assertGreater(result["remaining_tension_sigma"], result["initial_tension_sigma"])
        self.assertGreater(result["s8_nvg"], result["s8_planck"])

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            s8.main()
        output = rendered.getvalue()
        self.assertIn("Computed S8 = 0.851", output)
        self.assertIn("WORSENS", output)
        self.assertNotIn("RESOLVED", output)
        self.assertNotIn("A_drag", output)

    def test_iloveq_is_transform_only_and_uses_maximum_deviation(self) -> None:
        sequence = iloveq.compute_vmf_sequence()
        summary = iloveq.summarize_sequence(sequence)
        self.assertGreaterEqual(summary["sequence_count"], 3)
        self.assertEqual(summary["status"], "TRANSFORM_ONLY_NO_INDEPENDENT_IQ_SOLVE")
        self.assertFalse(summary["independent_i_solve"])
        self.assertFalse(summary["independent_q_solve"])
        expected_max = max(row["I_relative_deviation"] for row in sequence)
        self.assertAlmostEqual(summary["max_i_relative_deviation"], expected_max)
        self.assertAlmostEqual(summary["max_relative_deviation"], expected_max)
        self.assertIsNone(summary["tolerance_gate"])
        self.assertIn("maximum", summary["max_deviation_semantics"])
        self.assertGreater(summary["max_relative_deviation"], 0.01)

    def test_generated_iloveq_report_matches_runtime_and_readmes_are_reconciled(self) -> None:
        report_path = HERE / "fig_iloveq_universal_report.json"
        figure_path = HERE / "fig_iloveq_universal.png"
        self.assertTrue(report_path.is_file())
        self.assertGreater(figure_path.stat().st_size, 1000)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "TRANSFORM_ONLY_NO_INDEPENDENT_IQ_SOLVE")
        self.assertEqual(report["summary"]["sequence_count"], len(report["sequence"]))
        self.assertAlmostEqual(
            report["summary"]["max_relative_deviation"],
            max(row["I_relative_deviation"] for row in report["sequence"]),
        )
        self.assertIsNone(report["summary"]["tolerance_gate"])

        root = HERE.parent
        readme_en = (root / "README.md").read_text(encoding="utf-8")
        readme_ru = (root / "README_RU.md").read_text(encoding="utf-8")
        for readme in (readme_en, readme_ru):
            self.assertNotRegex(readme, r"(?i)S8[^\n]{0,100}(?:resolved|решено)")
            self.assertNotIn("all points within", readme.lower())
        self.assertIn("$S_8 \\approx 0.851$", readme_en)
        self.assertIn("$S_8 \\approx 0.851$", readme_ru)
        self.assertIn("transform-only", readme_en.lower())
        self.assertIn("только как преобразование", readme_ru)


if __name__ == "__main__":
    unittest.main()
