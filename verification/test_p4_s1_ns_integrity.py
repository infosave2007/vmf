"""Semantic tests for the Phase 4 NS selection/statistics repair."""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_global_significance as global_stats
import nvg_joint_ns_inference as joint
import nvg_ns_parameter_scan as scan
import nvg_ns_nicer_joint_audit as audit
import nvg_tidal_deformability as tidal
import nvg_tidal_deformability_gw170817 as tidal_sibling
import run_nvg_suite as suite


class NSSelectionStatisticsIntegrityTests(unittest.TestCase):
    def test_selection_record_names_in_sample_constraints(self):
        record = tidal.CANONICAL_SELECTION
        self.assertEqual(record["provenance"]["status"], "CONDITIONAL_IN_SAMPLE")
        self.assertFalse(record["provenance"]["independent"])
        sources = {item["source"] for item in record["provenance"]["constraints"]}
        self.assertEqual(sources, {"J0740", "GW170817", "NICER"})
        self.assertEqual(record["parameters"], tidal_sibling.CANONICAL_SELECTION["parameters"])

    def test_selection_function_uses_runtime_margin(self):
        results = {
            (1.8, 0.0): (2.02, 12.0, 450.0, 500.0, 500.0, True, 0.45),
            (2.0, 0.0): (2.04, 12.2, 430.0, 510.0, 510.0, True, 0.15),
        }
        best, record = scan.select_canonical_point(results)
        self.assertEqual(best, (1.8, 0.0))
        self.assertEqual(record["selected_point"]["n_trans_ratio"], 1.8)
        self.assertEqual(record["selected_point"]["cs2_q"], 1.0 / 3.0)
        self.assertEqual(record["parameters"]["n_trans_ratio"], 1.8)
        self.assertEqual(record["provenance"]["status"], "CONDITIONAL_IN_SAMPLE")

    def test_both_canonical_interpolators_fail_closed_at_endpoint(self):
        for module in (tidal, tidal_sibling):
            eos = module.EOS()
            self.assertGreater(eos.pressure_max, eos.pressure_min)
            self.assertGreater(eos.get_eps(eos.pressure_max), 0.0)
            self.assertEqual(eos.get_eps(0.0), 0.0)
            self.assertTrue(math.isfinite(eos.get_dedp(eos.pressure_max)))
            with self.assertRaises(ValueError):
                eos.get_eps(eos.pressure_max + 1.0)
            with self.assertRaises(ValueError):
                eos.get_eps(-1.0)
            with self.assertRaises(ValueError):
                module.solve_tov_tidal(eos, eos.pressure_max + 1.0)

    def test_joint_score_is_runtime_conditional_and_calibration_excluded(self):
        predictions = {
            "M_max": 2.0,
            "R_1.4": 12.0,
            "Lambda_1.4": 300.0,
            "Cooling_Dichotomy": None,
        }
        result = joint.run_joint_inference(predictions, joint.ROW_METADATA)
        self.assertEqual(result["comparison_status"], "CONDITIONAL_IN_SAMPLE")
        self.assertEqual(result["in_sample_count"], 3)
        cooling = next(row for row in result["rows"] if row["key"] == "Cooling_Dichotomy")
        self.assertFalse(cooling["included"])
        self.assertIsNone(cooling["pull"])
        expected = ((2.0 - 2.14) / 0.10) ** 2 + ((12.0 - 12.2) / 0.50) ** 2 + ((300.0 - 190.0) / 390.0) ** 2
        self.assertAlmostEqual(result["chi_squared_total"], expected)
        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            joint._print_report(result)
        self.assertIn("CONDITIONAL_IN_SAMPLE", rendered.getvalue())
        self.assertNotIn("independent reduced", rendered.getvalue().lower())

    def test_downstream_surfaces_consume_runtime_canonical_values(self):
        result = suite.run_forward_model()
        self.assertEqual(result["comparison_status"], "CONDITIONAL_IN_SAMPLE")
        self.assertEqual(result["conditional_rows"], 3)
        rows = suite.generate_evidence_ledger(result)
        self.assertTrue(any(f"{result['m_max']:.3f}" in row["value"] for row in rows))
        self.assertTrue(any("CONDITIONAL_IN_SAMPLE" in row["status"] for row in rows))
        audit_result = audit.run_audit()
        self.assertEqual(audit_result["status"], "CONDITIONAL_IN_SAMPLE")
        self.assertAlmostEqual(audit_result["canonical"]["M_max"], result["m_max"], places=12)

    def test_global_significance_withholds_p_value(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = global_stats.main()
        self.assertIsNone(result["p_value"])
        self.assertEqual(result["status"], "WITHHELD_NO_HELD_OUT_PRODUCER")
        self.assertIn("p-value: WITHHELD", output.getvalue())
        self.assertNotIn("p = 0.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
