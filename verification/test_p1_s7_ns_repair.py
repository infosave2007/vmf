"""Focused controls for the P1-S7 zero-jump/provenance repair."""

from __future__ import annotations

import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_eos_beta_css_softening as soft
import nvg_ns_predictive_audit as audit
import nvg_tidal_deformability as tidal


class NSP1S7RepairTests(unittest.TestCase):
    def test_eq15_provenance_distinguishes_eq14_derivative(self):
        self.assertEqual(audit.DENSITY_JUMP_SOURCE["equation"], "Eq. 15")
        self.assertEqual(audit.DENSITY_JUMP_SOURCE["derivative_equation"], "Eq. 14")
        self.assertIn("distributional", audit.DENSITY_JUMP_SOURCE["derivative_role"])

    def test_zero_jump_uses_the_same_interface_hook_once(self):
        baseline = soft.build_baseline_arrays(nn=220)
        eos = audit._make_hybrid_eos(baseline, 2.0, 0.0, 1.0 / 3.0)
        self.assertIsNotNone(eos)
        calls = []
        original = tidal.hinderer_density_jump_y_match

        def wrapped(*args, **kwargs):
            calls.append(float(args[3]))
            return original(*args, **kwargs)

        tidal.hinderer_density_jump_y_match = wrapped
        try:
            values = tidal.solve_tov_tidal(
                eos,
                53.8,
                dr=0.05,
                rtol=1.0e-6,
                atol=1.0e-9,
                density_jump_matching=True,
            )
        finally:
            tidal.hinderer_density_jump_y_match = original
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertEqual(calls, [0.0])

    def test_quick_continuity_and_interior_resolution_gate(self):
        result = audit.run_audit(quick=True)
        continuity = result["zero_limit_continuity"]
        self.assertEqual(continuity["status"], "PASS")
        self.assertIn(96, continuity["pressure_table_levels"])
        self.assertLessEqual(
            continuity["max_relative_tidal_delta"],
            audit.ROW_CONVERGENCE_THRESHOLD_RELATIVE,
        )
        self.assertEqual(
            result["row_convergence"]["axes"]["pressure_table_points"],
            [48, 96, 192],
        )
        self.assertIn("unresolved_candidate_rows", result["sensitivity_summary"])
        self.assertIn("screening_only_rows", result["sensitivity_summary"])

    def test_terminal_v3_artifact_semantics(self):
        path = os.path.join(HERE, "nvg_ns_predictive_audit_p1s7_results.json")
        if not os.path.exists(path):
            self.skipTest("terminal P1-S7 artifact not materialized in this focused test run")
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        self.assertEqual(result["schema_version"], "P1-S7-ns-predictive-audit-v3")
        self.assertEqual(result["audit_status"], "P1-S7_REPAIRED_ZERO_LIMIT_AUDIT")
        self.assertEqual(result["density_jump_treatment"]["source"]["equation"], "Eq. 15")
        self.assertEqual(result["density_jump_treatment"]["source"]["derivative_equation"], "Eq. 14")
        self.assertEqual(result["zero_limit_continuity"]["status"], "PASS")
        unresolved = result["grid_summary"]["unresolved_candidate_rows"]
        screening_only = result["grid_summary"]["screening_only_rows"]
        non_evidence = result["grid_summary"]["excluded_non_evidence_rows"]
        self.assertGreater(unresolved, 0)
        self.assertGreater(screening_only, 0)
        self.assertEqual(unresolved + screening_only, non_evidence)
        self.assertIn(160, result["row_convergence"]["axes"]["pressure_table_points"])


if __name__ == "__main__":
    unittest.main()
