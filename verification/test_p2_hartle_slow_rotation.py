"""Focused controls for the independent P2-S1/P2-S5 Hartle producer."""

from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_hartle_slow_rotation as hartle
import nvg_tidal_deformability as tidal


class HartleSlowRotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eos = tidal.EOS()

    def test_primary_provenance_and_canonical_boundary(self):
        self.assertEqual(hartle.EQUATION_PROVENANCE["status"], "PRIMARY_SOURCE_EQUATIONS")
        self.assertEqual(hartle.MODEL_IDENTITY["status"], "CONDITIONAL_IN_SAMPLE_BACKGROUND")
        self.assertEqual(hartle.MODEL_IDENTITY["evidence_weight"], 0.0)
        self.assertIn("url", hartle.EQUATION_PROVENANCE["frame_dragging"])
        selection = self.eos.canonical_selection
        self.assertEqual(selection["provenance"]["status"], "CONDITIONAL_IN_SAMPLE")
        self.assertFalse(selection["provenance"]["independent"])

    def test_first_order_star_is_regular_positive_and_matches_vacuum(self):
        star = hartle.integrate_star(100.0, eos=self.eos)
        self.assertGreater(star.mass_msun, 0.0)
        self.assertGreater(star.radius_km, 5.0)
        self.assertGreater(star.compactness, 0.0)
        self.assertLess(star.compactness, 0.5)
        self.assertGreater(star.inertia_geom_km3, 0.0)
        self.assertGreater(star.inertia_cgs, 0.0)
        self.assertGreater(star.inertia_bar, 0.0)
        diagnostics = star.diagnostics
        self.assertTrue(diagnostics["centre_boundary"]["regularity_ok"])
        self.assertTrue(diagnostics["surface_boundary"]["surface_ok"])
        self.assertTrue(diagnostics["vacuum_boundary"]["vacuum_ok"])
        self.assertTrue(diagnostics["independent_extraction"]["identity_ok"])
        self.assertEqual(diagnostics["positivity"]["status"], "PASS")

    def test_independent_inertia_conversions_and_integral_identity(self):
        star = hartle.integrate_star(100.0, eos=self.eos)
        round_trip = hartle.inertia_geom_to_cgs(star.inertia_geom_km3)
        self.assertAlmostEqual(round_trip, star.inertia_cgs, delta=1.0e-9 * star.inertia_cgs)
        self.assertAlmostEqual(
            hartle.inertia_cgs_to_geom(star.inertia_cgs),
            star.inertia_geom_km3,
            delta=1.0e-9 * star.inertia_geom_km3,
        )
        self.assertLess(star.diagnostics["independent_extraction"]["integral_identity_relative_residual"], 5.0e-3)

    def test_constant_density_benchmark_is_validated(self):
        benchmark = hartle.constant_density_benchmark()
        self.assertEqual(benchmark["status"], "validated")
        self.assertAlmostEqual(benchmark["inertia_over_MR2"], 0.4182, delta=0.002)
        self.assertLess(benchmark["relative_difference"], benchmark["threshold"])

    def test_sequence_j0737_and_quadrupole_boundary(self):
        sequence = hartle.generate_sequence(
            eos=self.eos,
            central_pressures=np.logspace(-0.30, 3.20, 28),
        )
        j0737 = hartle.predict_j0737a(sequence, hartle.load_j0737a_input())
        self.assertEqual(j0737["status"], "derived_conditional")
        self.assertAlmostEqual(j0737["mass_msun"], 1.3381, places=4)
        self.assertGreater(j0737["predicted_inertia_cgs_g_cm2"], 1.0e45)
        self.assertEqual(j0737["mass_prediction"]["status"], "blocked")
        self.assertEqual(
            hartle.MODEL_IDENTITY["quadrupole_status"], "BLOCKED"
        )

    def test_pressure_ordered_branch_split_never_mass_sorts(self):
        masses = (1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 1.0, 1.2, 1.4)
        rows = [
            {"central_pressure": float(index + 1), "mass": mass,
             "radius_km": 12.0, "inertia_geom_km3": 100.0}
            for index, mass in enumerate(masses)
        ]
        topology = hartle._branch_topology(rows)
        self.assertEqual(topology["status"], "PASS_STABLE_BRANCH_TOPOLOGY")
        self.assertFalse(topology["mass_sorted"])
        self.assertFalse(topology["disconnected_bridging"])
        self.assertEqual(topology["rows"][3]["branch_id"], -1)
        self.assertEqual(topology["rows"][3]["status"], "EXCLUDED_UNRESOLVED")
        self.assertTrue(topology["rows"][3]["exclusion_reason"])
        for branch in topology["branches"]:
            indices = branch["row_indices"]
            self.assertEqual(indices, sorted(indices))
            self.assertTrue(all(topology["rows"][i]["branch_id"] == branch["id"] for i in indices))

    def test_previous_p2s4_probe_excludes_low_mass_gap(self):
        sequence = hartle.generate_sequence(
            eos=self.eos,
            central_pressures=np.linspace(1.5, 6.0, 33),
        )
        topology = hartle._branch_topology(sequence)
        self.assertEqual(len(topology["branches"]), 2)
        excluded = [row for row in topology["rows"] if row["branch_id"] < 0]
        self.assertGreaterEqual(len(excluded), 5)
        self.assertTrue(any(row["exclusion_reason"] == "DESCENDING_MASS_GAP" for row in excluded))
        self.assertGreater(len(topology["unresolved_intervals"]), 0)
        # The overlapping low-mass ranges are intentionally ambiguous; no
        # interpolation may bridge the unresolved pressure gap.
        self.assertIsNone(hartle._interpolate_star(sequence, 0.2193))
        self.assertEqual(topology["branches"][0]["id"], 0)
        self.assertEqual(topology["branches"][1]["id"], 1)
        self.assertLess(topology["branches"][0]["end_central_pressure"], topology["branches"][1]["start_central_pressure"])

    def test_j0737_interpolation_is_single_branch_and_resolution_scoped(self):
        sequence = hartle.generate_sequence(
            eos=self.eos,
            central_pressures=np.logspace(-0.30, 3.20, 28),
        )
        input_record = hartle.load_j0737a_input()
        prediction = hartle.predict_j0737a(sequence, input_record)
        self.assertEqual(prediction["status"], "derived_conditional")
        self.assertEqual(prediction["branch_id"], 1)
        self.assertFalse(prediction["gap_crossed"])
        self.assertEqual(prediction["interpolation"]["status"], "PASS_SINGLE_STABLE_BRANCH")
        sensitivity = hartle.interpolation_resolution_sensitivity(
            self.eos, input_record, sequence, quick=True,
        )
        self.assertEqual(sensitivity["status"], "PASS_BRANCH_LOCAL_RESOLUTION")
        self.assertEqual(sensitivity["branch_id_consistency"], [1])
        self.assertGreaterEqual(sensitivity["j0737a_inertia_relative_spread"], 0.0)

    def test_convergence_controls_and_frozen_regression(self):
        payload = hartle.build_payload(quick=False)
        self.assertEqual(payload["canonical_regression"]["status"], "PASS")
        self.assertEqual(payload["convergence"]["status"], "PASS")
        self.assertEqual(payload["boundary_checks"]["vacuum"], "PASS")
        self.assertEqual(payload["boundary_checks"]["independent_identities"], "PASS")
        self.assertEqual(payload["classification"]["quadrupole_Q"], "blocked")
        self.assertEqual(payload["classification"]["empirical_confirmation"], "not_claimed")
        self.assertEqual(
            payload["canonical_regression"]["computed"],
            hartle.FROZEN_CANONICAL,
        )

    def test_artifact_provenance_fails_closed_on_mutation(self):
        payload = hartle.build_payload(quick=True)
        hartle.assert_artifact_provenance(payload)
        mutated = json.loads(json.dumps(payload))
        mutated["source_to_artifact_provenance"]["source_sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            hartle.assert_artifact_provenance(mutated)

        mutated = json.loads(json.dumps(payload))
        mutated["source_to_artifact_provenance"]["test_sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            hartle.assert_artifact_provenance(mutated)

        mutated = json.loads(json.dumps(payload))
        mutated["source_to_artifact_provenance"]["input_sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            hartle.assert_artifact_provenance(mutated)

    def test_terminal_p2s5_artifact_branch_and_hash_semantics(self):
        if not hartle.RESULT_PATH.exists():
            self.skipTest("terminal P2-S5 artifact not materialized in this focused test run")
        payload = json.loads(hartle.RESULT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["audit"], "P2-S5")
        self.assertEqual(payload["schema_version"], hartle.SCHEMA_VERSION)
        self.assertEqual(payload["artifacts"]["result_json"], "verification/nvg_hartle_slow_rotation_p2s5_results.json")
        self.assertEqual(payload["observable_status"], "derived_conditional_branch_scoped")
        self.assertEqual(payload["public_status"], "DERIVED_CONDITIONAL_ALLOWED_BRANCHES_LOW_MASS_UNRESOLVED")
        self.assertEqual(payload["branch_topology"]["status"], "PASS_STABLE_BRANCH_TOPOLOGY")
        self.assertFalse(payload["branch_topology"]["mass_sorted"])
        self.assertFalse(payload["branch_topology"]["disconnected_bridging"])
        self.assertEqual(payload["j0737a"]["gap_crossed"], False)
        self.assertEqual(payload["interpolation_resolution_sensitivity"]["status"], "PASS_BRANCH_LOCAL_RESOLUTION")
        hartle.assert_artifact_provenance(payload)


if __name__ == "__main__":
    unittest.main()
