"""Focused controls for the Phase-3 S1 frozen NS/Hartle forecast."""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_ns_frozen_forecasts as forecast


class P3S1FrozenForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Quick mode exercises both coarse and terminal pressure ladders while
        # keeping focused tests independent of the terminal figure/report.
        cls.payload = forecast.build_payload(quick=True)

    def test_same_background_equations_and_no_lookup_outputs(self):
        self.assertEqual(self.payload["status"], "FROZEN_CONDITIONAL_FORECAST")
        self.assertTrue(self.payload["equation_provenance"]["same_background_contract"])
        self.assertEqual(self.payload["configuration"]["tidal_on_same_background"]["lookup_table_used"], False)
        self.assertGreaterEqual(len(self.payload["forecast_mass_grid"]), 6)
        for row in self.payload["forecast_mass_grid"]:
            self.assertEqual(row["branch_id"], forecast.TARGET_BRANCH_ID)
            self.assertEqual(row["classification"], "derived_conditional")
            self.assertTrue(row["same_background"])
            self.assertFalse(row["lookup_table_used"])
            self.assertGreater(row["lambda"], 0.0)
            self.assertGreater(row["inertia_cgs_g_cm2"], 0.0)
            self.assertAlmostEqual(row["Lambda"], row["lambda"], places=12)
            self.assertAlmostEqual(row["Ibar"], row["inertia_bar"], places=12)
            self.assertEqual(row["legacy_i_love_overlay"]["status"], "retired_overlay_descriptive_only")
            self.assertFalse(row["legacy_i_love_overlay"]["used_for_forecast"])

    def test_pressure_order_branch_and_grid_ladder_are_fail_closed(self):
        rows = self.payload["grid_ladder"]["rows"]
        self.assertEqual([row["label"] for row in rows], ["coarse_28", "terminal_81"])
        for row in rows:
            self.assertFalse(row["mass_sorted"])
            self.assertFalse(row["disconnected_bridging"])
            self.assertEqual(row["selected_branch_id"], 1)
            self.assertGreater(row["selected_branch_points"], 3)
        self.assertFalse(self.payload["grid_ladder"]["gap_crossed"])
        self.assertEqual(
            self.payload["phase1_candidate_sensitivity_envelope"]["status"],
            "blocked",
        )
        self.assertFalse(self.payload["phase1_candidate_sensitivity_envelope"]["used_in_forecast"])
        self.assertEqual(
            self.payload["frozen_sequence_identity"]["status"],
            "PASS_FROZEN_SEQUENCE_IDENTITY",
        )

    def test_j0737_mass_solver_and_prospective_falsification_semantics(self):
        j0737 = self.payload["j0737a"]
        self.assertEqual(j0737["status"], "derived_conditional")
        self.assertEqual(j0737["branch_id"], 1)
        self.assertFalse(j0737["gap_crossed"])
        self.assertEqual(j0737["inverse_mass"]["status"], "blocked")
        self.assertEqual(len(j0737["input_mass_envelope"]["rows"]), 3)
        self.assertEqual(j0737["input_mass_envelope"]["status"], "sensitivity_only")
        ladder = j0737["numerical_solver_envelope"]
        self.assertEqual(ladder["status"], "sensitivity_only")
        self.assertEqual(len(ladder["rows"]), 3)
        self.assertEqual(ladder["method"], "tidal_ode_ladder_on_exact_hartle_background")
        self.assertEqual(
            [row["label"] for row in j0737["input_mass_envelope"]["rows"]],
            ["minus_1sigma", "central", "plus_1sigma"],
        )
        for row in j0737["input_mass_envelope"]["rows"]:
            self.assertEqual(row["method"], "pressure_ordered_exact_mass_root")
            self.assertFalse(row["gap_crossed"])
            self.assertLessEqual(
                abs(row["target_solve"]["mass_residual_msun"]),
                max(row["target_solve"]["mass_abs_tol_msun"], 1.0e-7),
            )
        for row in self.payload["forecast_mass_grid"]:
            self.assertEqual(row["method"], "pressure_ordered_exact_mass_root")
            self.assertEqual(row["target_solve"]["branch_id"], 1)
            self.assertEqual(row["target_solve"]["mass_sorted"], False)
            self.assertEqual(row["target_solve"]["disconnected_bridging"], False)
        bands = self.payload["falsification_bands"]
        self.assertEqual(bands["status"], "sensitivity_only")
        self.assertEqual(bands["semantic_label"], "sensitivity_only_prospective_model_conditional")
        self.assertFalse(bands["joint_I_Lambda_R"]["theory_wide_confidence_interval"])
        self.assertTrue(bands["joint_I_Lambda_R"]["prospective_only"])
        self.assertLess(bands["I_A"]["lower"], j0737["forecast"]["inertia_cgs_g_cm2"])
        self.assertGreater(bands["I_A"]["upper"], j0737["forecast"]["inertia_cgs_g_cm2"])
        for key in ("I_A", "Lambda_A", "R_A"):
            self.assertNotIn("incompatibility_rule", bands[key])
            self.assertIn("prospective_falsifier", bands[key])
            self.assertLessEqual(bands[key]["display_lower"], bands[key]["lower"])
            self.assertGreaterEqual(bands[key]["display_upper"], bands[key]["upper"])

    def test_independent_unit_identities_and_canonical_regression(self):
        residuals = self.payload["controls"]["identity_max_relative_residual"]
        self.assertLess(residuals["compactness"], 1.0e-12)
        self.assertLess(residuals["inertia_bar"], 1.0e-12)
        self.assertLess(residuals["inertia_unit_round_trip"], 1.0e-12)
        self.assertLess(residuals["lambda_k2_compactness"], 1.0e-12)
        self.assertEqual(self.payload["canonical_regression"]["status"], "PASS")
        self.assertEqual(self.payload["canonical_regression"]["computed"], self.payload["canonical_regression"]["frozen"])

    def test_mutations_fail_closed(self):
        forecast.assert_artifact_provenance(self.payload)
        mutated = copy.deepcopy(self.payload)
        mutated["source_to_artifact_provenance"]["hartle_source_sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            forecast.assert_artifact_provenance(mutated)
        mutated = copy.deepcopy(self.payload)
        mutated["forecast_mass_grid"][0]["branch_id"] = 0
        with self.assertRaises(AssertionError):
            forecast.assert_artifact_provenance(mutated)
        mutated = copy.deepcopy(self.payload)
        mutated["configuration"]["tidal_on_same_background"]["lookup_table_used"] = True
        with self.assertRaises(AssertionError):
            forecast.assert_artifact_provenance(mutated)
        mutated = copy.deepcopy(self.payload)
        mutated["falsification_bands"]["joint_I_Lambda_R"]["theory_wide_confidence_interval"] = True
        with self.assertRaises(AssertionError):
            forecast.assert_artifact_provenance(mutated)
        mutated = copy.deepcopy(self.payload)
        mutated["j0737a"]["forecast"]["lambda"] = 1.0
        with self.assertRaises(AssertionError):
            forecast.assert_artifact_provenance(mutated)
        mutated = copy.deepcopy(self.payload)
        mutated["falsification_bands"]["Lambda_A"]["lower"] = 1.0
        with self.assertRaises(AssertionError):
            forecast.assert_artifact_provenance(mutated)
        mutated = copy.deepcopy(self.payload)
        mutated["equation_provenance"]["tidal"]["y_equation"] = "WRONG"
        with self.assertRaises(AssertionError):
            forecast.assert_artifact_provenance(mutated)
        mutated = copy.deepcopy(self.payload)
        mutated["j0737a"]["inverse_mass"]["status"] = "derived_conditional"
        with self.assertRaises(AssertionError):
            forecast.assert_artifact_provenance(mutated)

    def test_terminal_artifact_if_materialized(self):
        if not forecast.RESULT_PATH.exists():
            self.skipTest("terminal P3-S1 artifact not materialized in focused test run")
        payload = json.loads(forecast.RESULT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], forecast.SCHEMA_VERSION)
        self.assertEqual(payload["audit"], forecast.AUDIT_ID)
        self.assertEqual(payload["artifacts"]["csv"], "verification/nvg_ns_frozen_forecasts_p3s1.csv")
        self.assertEqual(payload["artifacts"]["figure_png"], "verification/fig_ns_frozen_forecasts_p3s1.png")
        self.assertEqual(payload["configuration"]["target_branch_id"], 1)
        self.assertGreaterEqual(len(payload["forecast_mass_grid"]), 6)
        forecast.assert_artifact_provenance(payload)


if __name__ == "__main__":
    unittest.main()
