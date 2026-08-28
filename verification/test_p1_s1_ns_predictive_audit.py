"""Focused semantics for the Phase 1 NS predictive/uncertainty audit."""

from __future__ import annotations

import math
import os
import sys
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_ns_predictive_audit as audit
import nvg_tidal_deformability as tidal


class NSPredictiveAuditTests(unittest.TestCase):
    def test_declared_grid_is_dense_and_boundary_refined(self):
        self.assertGreaterEqual(len(audit.DECLARED_GRID["n_trans_ratio"]), 8)
        self.assertGreaterEqual(len(audit.DECLARED_GRID["delta_eps_ratio"]), 8)
        self.assertEqual(audit.DECLARED_GRID["delta_eps_ratio"][0], 0.0)
        self.assertLess(audit.DECLARED_GRID["delta_eps_ratio"][1], 0.02)
        self.assertIn(1.0, audit.DECLARED_GRID["cs2_q"])

    def test_configured_tolerances_reuse_canonical_solver(self):
        eos = tidal.EOS()
        baseline = tidal.solve_tov_tidal(eos, 100.0)
        adaptive = tidal.solve_tov_tidal(eos, 100.0, dr=0.05, rtol=1.0e-5, atol=1.0e-8)
        self.assertTrue(all(math.isfinite(value) for value in baseline))
        self.assertTrue(all(math.isfinite(value) for value in adaptive))
        self.assertGreater(adaptive[0], 0.0)
        self.assertGreater(adaptive[1], 5.0)
        with self.assertRaises(ValueError):
            tidal.solve_tov_tidal(eos, 100.0, rtol=1.0e-5)

    def test_quick_audit_has_conditional_loo_and_blocked_anchor(self):
        result = audit.run_audit(quick=True)
        self.assertEqual(result["status"], "CONDITIONAL_IN_SAMPLE")
        self.assertAlmostEqual(result["canonical"]["M_max"], result["canonical"]["observables"]["M_max"])
        self.assertEqual(result["canonical"]["selection"]["provenance"]["independent"], False)
        self.assertEqual(set(result["loo"]), set(audit.OBSERVABLE_CONSTRAINTS))
        for row in result["loo"].values():
            self.assertFalse(row["independent"])
            self.assertEqual(row["evidence_weight"], 0.0)
            self.assertIn("conditional", row["semantics"].lower())
        dependency = result["m_omega_dependency"]
        self.assertEqual(dependency["status"], "BLOCKED_NO_PHYSICAL_DEPENDENCY")
        self.assertFalse(dependency["off_anchor_evaluated"])
        self.assertIn("no M_Omega_0 argument", dependency["blocked_link"])

    def test_grid_rows_report_domain_and_interval_semantics(self):
        result = audit.run_audit(quick=True)
        self.assertGreater(result["grid_summary"]["valid_rows"], 0)
        self.assertIn("sensitivity_envelopes", result["semantics"]["grid_intervals"])
        self.assertIn("conditional", result["semantics"]["likelihood_intervals"])
        for row in result["grid_rows"]:
            self.assertIn("domain", row)
            self.assertIn("causality_ok", row["domain"])
            self.assertIn("constraints", row)


if __name__ == "__main__":
    unittest.main()
