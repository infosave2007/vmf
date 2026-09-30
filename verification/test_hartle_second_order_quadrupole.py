"""Semantic tests for the Hartle second-order spin-quadrupole probe.

Equation-based convergence, matching, the Maclaurin limit and authenticated
first-order reproduction gate the quadrupole. The band and I-Q relation are
only diagnostic overlays. The l=0 mass change remains separately blocked.
The frozen first-order producer and forecast artifact stay untouched.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import unittest
from decimal import Decimal, localcontext
from unittest.mock import patch


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_hartle_second_order_quadrupole_probe as probe
import nvg_hartle_slow_rotation as hartle
import nvg_tidal_deformability as tidal


class HartleSecondOrderQuadrupoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eos = tidal.EOS()
        # build_payload performs the full (non-quick) computation once; the
        # quick flag only governs whether main() asserts/writes artifacts.
        cls.payload = probe.build_payload(quick=True)

    def test_maclaurin_newtonian_limit_is_validated(self):
        mac = self.payload["maclaurin_validation"]
        self.assertEqual(mac["status"], "PASS_MACLAURIN_NEWTONIAN_LIMIT")
        self.assertTrue(all(mac["checks"].values()), mac["checks"])
        limits = mac["richardson_limits_C0"]
        tol = mac["tolerances"]
        self.assertAlmostEqual(limits["q_tilde_ratio_C0"], 1.0, delta=tol["q_tilde_ratio"])
        self.assertAlmostEqual(limits["Q_over_Om2R5_C0"], -0.5, delta=tol["Q_over_Om2R5"])
        self.assertAlmostEqual(
            limits["deltaM_over_MR2Om2_C0"], 0.4, delta=tol["deltaM_over_MR2Om2"]
        )
        self.assertAlmostEqual(limits["I_over_MR2_C0"], 0.4, delta=tol["I_over_MR2"])

    def test_j0737a_second_order_forecast_is_in_band_and_stable(self):
        j0737 = self.payload["j0737a"]
        self.assertEqual(j0737["status"], "derived_conditional_second_order")
        self.assertTrue(j0737["q_tilde_in_band"])
        lo, hi = j0737["q_tilde_band"]
        self.assertGreaterEqual(j0737["q_tilde"], lo)
        self.assertLessEqual(j0737["q_tilde"], hi)
        self.assertLess(j0737["q_tilde_seed_radius_drift"], 1.0e-4)

    def test_first_order_ledger_state_is_reproduced_fail_closed(self):
        repro = self.payload["j0737a"]["first_order_reproduced"]
        tolerances = repro["tolerances"]
        for key, tol in tolerances.items():
            got = repro[key]
            want = probe.LEDGER_J0737A[key]
            self.assertLessEqual(
                abs(got - want), tol * max(abs(want), 1e-300),
                f"{key}: {got} vs {want} (tol {tol})",
            )

    def test_iloveq_universal_crosscheck_corroborates(self):
        iq = self.payload["j0737a"]["iloveq_crosscheck"]
        self.assertEqual(iq["status"], "PASS_IQ_UNIVERSAL_1PCT")
        self.assertLess(iq["relative_deviation"], 0.02)
        # The relation is a non-authoritative overlay, never evidence.
        self.assertIn("non-authoritative", iq["authority"])

    def test_first_order_module_is_hash_locked_and_still_blocked(self):
        prov = self.payload["source_provenance"]
        self.assertEqual(
            prov["first_order_dependency"]["source_sha256"],
            hartle._sha256(hartle.SOURCE_PATH),
        )
        # The probe resolves the blocker in its own artifact; the frozen
        # first-order module identity is deliberately left BLOCKED.
        self.assertEqual(hartle.MODEL_IDENTITY["quadrupole_status"], "BLOCKED")
        self.assertEqual(self.payload["blocker_resolution"]["previous_status"], "BLOCKED")
        self.assertFalse(self.payload["blocker_resolution"]["substitute_used"])

    def test_artifact_provenance_fails_closed_on_mutation(self):
        probe.assert_artifact_provenance(self.payload)

        mutated = json.loads(json.dumps(self.payload))
        mutated["source_provenance"]["first_order_dependency"]["source_sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            probe.assert_artifact_provenance(mutated)

        mutated = json.loads(json.dumps(self.payload))
        mutated["maclaurin_validation"]["status"] = "BLOCKED_MACLAURIN_LIMIT"
        with self.assertRaises(AssertionError):
            probe.assert_artifact_provenance(mutated)

        mutated = json.loads(json.dumps(self.payload))
        mutated["j0737a"]["q_tilde_in_band"] = False
        with self.assertRaises(AssertionError):
            probe.assert_artifact_provenance(mutated)

        mutated = json.loads(json.dumps(self.payload))
        mutated["j0737a"]["iloveq_crosscheck"]["relative_deviation"] = 0.5
        with self.assertRaises(AssertionError):
            probe.assert_artifact_provenance(mutated)

    def test_refinement_axes_and_numerical_range(self):
        c = self.payload["j0737a"]["convergence"]
        self.assertEqual(c["status"], "PASS_NUMERICAL_CONVERGENCE")
        expected = {"max_step_km": probe.STEP_LADDER, "rtol": probe.RTOL_LADDER,
                    "seed_radius_km": probe.SEED_LADDER,
                    "background_max_step_km": probe.STEP_LADDER,
                    "background_rtol": probe.RTOL_LADDER}
        self.assertEqual(set(c["ladders"]), set(expected))
        for axis, controls in expected.items():
            rows = c["ladders"][axis]
            self.assertEqual([r["control"] for r in rows], list(controls))
            drift = abs(rows[-1]["q_tilde"] / rows[-2]["q_tilde"] - 1.0)
            self.assertLess(drift, 1e-3)
            self.assertTrue(c["checks"][axis])
        values = [r["q_tilde"] for rows in c["ladders"].values() for r in rows]
        self.assertEqual(c["q_tilde_numerical_range"], [min(values), max(values)])

    def test_spin_normalization_and_matching_residuals(self):
        c = self.payload["j0737a"]["convergence"]
        self.assertEqual([r["control"] for r in c["spin_rescaling"]], [0.5, 2.0])
        for row in c["spin_rescaling"]:
            self.assertLess(row["q_relative_drift"], 1e-4)
            self.assertLess(row["Q_quadratic_relative_drift"], 1e-4)
        rows = [r for rows in c["ladders"].values() for r in rows] + c["spin_rescaling"]
        for row in rows:
            self.assertTrue(all(math.isfinite(v) for v in row.values()))
            self.assertLess(row["h2_matching_residual"], 1e-8)
            self.assertLess(row["K2_matching_residual"], 1e-8)
            self.assertGreaterEqual(row["matching_condition_number"], 1.0)

    def test_high_precision_legendre_weak_field_limit(self):
        with localcontext() as ctx:
            ctx.prec = 60
            for x0 in (5.3, 10.0, 100.0, 1000.0):
                x = Decimal(str(x0))
                d = x*x - 1
                L = ((x+1)/(x-1)).ln()
                q20 = (3*x*x-1)*L/4 - 3*x/2
                dq = 3*x*L/2 - (3*x*x-1)/(2*d) - Decimal("1.5")
                ddq = 3*L/2 - 3*x/d + 2*x/(d*d)
                expected = {"Q20": q20, "Q21_cs": -d.sqrt()*dq, "Q22": d*ddq}
                got = probe._legendre_l2(x0)
                for key, value in expected.items():
                    self.assertAlmostEqual(got[key]/float(value), 1.0, delta=1e-9)

    def test_successive_maclaurin_extrapolations(self):
        mac = self.payload["maclaurin_validation"]
        self.assertEqual([r["compactness"] for r in mac["rows"]], list(probe.MACLAURIN_COMPACTNESSES))
        self.assertEqual(mac["rows"][-1]["compactness"], 0.0015)
        self.assertLess(abs(mac["previous_q_tilde_ratio_C0"] - 1.0), 0.002)
        self.assertLess(mac["extrapolation_difference"], 0.002)

    def test_mass_correction_is_separate_and_not_promoted(self):
        self.assertEqual(self.payload["classification"]["mass_correction"], "blocked")
        mass = self.payload["j0737a"]["mass_correction"]
        self.assertEqual(mass["status"], "BLOCKED_SURFACE_AND_ENSEMBLE_VALIDATION")
        self.assertIn("fixed central pressure", mass["sequence"])
        mac = self.payload["maclaurin_validation"]
        self.assertNotIn("deltaM_ratio_to_two_fifths", mac["quadrupole_checks"])
        for row in mac["rows"]:
            self.assertEqual(row["mass_correction_scope"], "bulk_only_not_total_mass")
            self.assertGreater(row["deltaM_surface_over_MR2Om2"], 0.0)
            self.assertAlmostEqual(row["deltaM_total_over_MR2Om2"],
                                   row["deltaM_over_MR2Om2"] + row["deltaM_surface_over_MR2Om2"], delta=1e-10)

    def test_empirical_overlay_is_not_an_acceptance_gate(self):
        # A deliberately incompatible diagnostic fit must not change the
        # equation-based numerical result or its acceptance status.
        with patch.object(probe, "QTILDE_BAND", (100.0, 200.0)), \
             patch.object(probe, "yy_iq_ibar", return_value=1.0), \
             patch.object(probe, "quadrupole_convergence", return_value=self.payload["j0737a"]["convergence"]):
            result = probe.j0737a_second_order(eos=self.eos)
        self.assertFalse(result["q_tilde_in_band"])
        self.assertEqual(result["iloveq_crosscheck"]["status"], "REVIEW_IQ_UNIVERSALITY")
        self.assertEqual(result["status"], "derived_conditional_second_order")
        self.assertFalse(result["iloveq_crosscheck"]["acceptance_gate"])
        self.assertFalse(result["q_tilde_band_acceptance_gate"])

    def test_resigned_numerical_and_semantic_mutations_rejected(self):
        mutations = [
            ("q_tilde", lambda p: p["j0737a"].update(q_tilde=6.7)),
            ("first_order", lambda p: p["j0737a"]["first_order_reproduced"].update(R_A_km=13.0)),
            ("ladder", lambda p: p["j0737a"]["convergence"]["ladders"]["rtol"][0].update(q_tilde=6.7)),
            ("flag", lambda p: p["j0737a"]["convergence"]["checks"].update(rtol=False)),
            ("missing_checks", lambda p: p["j0737a"]["convergence"].update(checks={})),
            ("mass_promotion", lambda p: p["classification"].update(mass_correction="derived")),
            ("stale_source", lambda p: p["source_provenance"].update(producer_sha256="0"*64)),
            ("stale_test", lambda p: p["source_provenance"].update(test_sha256="0"*64)),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                value = copy.deepcopy(self.payload)
                mutate(value)
                value["payload_integrity"]["sha256"] = probe.payload_digest(value)
                with self.assertRaises(AssertionError):
                    probe.assert_artifact_provenance(value)

    def test_nonfinite_and_unsigned_numeric_changes_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf"), 6.7):
            value = copy.deepcopy(self.payload)
            value["j0737a"]["q_tilde"] = bad
            with self.assertRaises(AssertionError):
                probe.assert_artifact_provenance(value)

    def test_frozen_target_has_no_mutable_ledger_dependency(self):
        self.assertEqual(probe.frozen_target(), probe.LEDGER_J0737A)
        provenance = self.payload["source_provenance"]
        self.assertNotIn("predictive_research_ledger.json", json.dumps(provenance))
        self.assertEqual(provenance["frozen_target"]["source_sha256"], probe.FROZEN_FORECAST_SHA256)
        with patch.object(probe, "FROZEN_FORECAST_SHA256", "0"*64):
            with self.assertRaises(AssertionError):
                probe.frozen_target()

    def test_failed_maclaurin_blocks_both_quadrupole_classifications(self):
        mac = copy.deepcopy(self.payload["maclaurin_validation"])
        mac["status"] = "BLOCKED_MACLAURIN_LIMIT"
        with patch.object(probe, "maclaurin_validation", return_value=mac), \
             patch.object(probe, "j0737a_second_order", return_value=self.payload["j0737a"]):
            result = probe.build_payload()
        self.assertEqual(result["public_status"], "BLOCKED_SECOND_ORDER")
        self.assertEqual(result["classification"]["quadrupole_Q"], "blocked")
        self.assertEqual(result["classification"]["q_tilde"], "blocked")

    def test_terminal_artifact_semantics(self):
        if not probe.RESULT_PATH.exists():
            self.skipTest("second-order quadrupole artifact not materialized in this focused run")
        payload = json.loads(probe.RESULT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["audit"], "hartle-second-order-quadrupole")
        self.assertEqual(payload["schema_version"], probe.SCHEMA_VERSION)
        self.assertEqual(
            payload["artifacts"]["result_json"],
            "verification/nvg_hartle_second_order_quadrupole_probe_results.json",
        )
        self.assertEqual(
            payload["artifacts"]["figure_png"],
            "verification/nvg_hartle_second_order_quadrupole_probe.png",
        )
        self.assertEqual(payload["public_status"], "DERIVED_CONDITIONAL_SECOND_ORDER")
        self.assertEqual(
            payload["classification"]["quadrupole_Q"], "derived_conditional_second_order"
        )
        self.assertEqual(payload["classification"]["empirical_confirmation"], "not_claimed")
        probe.assert_artifact_provenance(payload)


if __name__ == "__main__":
    unittest.main()
