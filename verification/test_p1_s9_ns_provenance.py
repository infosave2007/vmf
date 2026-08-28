"""Focused controls for the P1-S9 exact ODE provenance repair."""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_ns_predictive_audit as audit
import nvg_tidal_deformability as tidal


class NSP1S9ProvenanceTests(unittest.TestCase):
    def test_docstring_and_live_source_provenance_are_exact(self):
        doc = tidal.solve_tov_tidal.__doc__ or ""
        for token in ("solve_ivp", "DOP853", "rtol", "atol", "max_step"):
            self.assertIn(token, doc)
        provenance = tidal.adaptive_solver_provenance()
        self.assertEqual(provenance["backend"], "scipy.integrate.solve_ivp")
        self.assertEqual(provenance["method"], "DOP853")
        self.assertEqual(provenance["max_step_cap_km"], {"rtol_le_1e-4": 0.05, "rtol_gt_1e-4": 0.10})
        self.assertTrue(all(provenance["source_checks"].values()))
        self.assertEqual(tidal.adaptive_solver_max_step(0.05, 1.0e-4), 0.05)
        self.assertEqual(tidal.adaptive_solver_max_step(0.05, 1.0e-6), 0.05)
        self.assertEqual(tidal.adaptive_solver_max_step(0.05, 1.0e-8), 0.05)
        self.assertEqual(tidal.adaptive_solver_max_step(0.01, 1.0e-3), 0.10)

    def test_artifact_provenance_assertion_rejects_drift(self):
        result = audit.run_audit(quick=True)
        audit.assert_solver_provenance_artifact(result)
        drifted = copy.deepcopy(result)
        drifted["solver_provenance"]["method"] = "adaptive RK4"
        with self.assertRaises(AssertionError):
            audit.assert_solver_provenance_artifact(drifted)
        drifted = copy.deepcopy(result)
        drifted["convergence"]["ode_tolerances"]["method"] = "adaptive RK4"
        with self.assertRaises(AssertionError):
            audit.assert_solver_provenance_artifact(drifted)

    def test_terminal_v3_artifact_has_d853_metadata(self):
        path = os.path.join(HERE, "nvg_ns_predictive_audit_p1s7_results.json")
        if not os.path.exists(path):
            self.skipTest("terminal P1-S7 v3 artifact not materialized")
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        audit.assert_solver_provenance_artifact(result)
        ode = result["convergence"]["ode_tolerances"]
        self.assertEqual(ode["backend"], "scipy.integrate.solve_ivp")
        self.assertEqual(ode["method"], "DOP853")
        self.assertIn("max_step", ode["rows"][0])
        self.assertEqual(result["source_to_artifact_provenance"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
