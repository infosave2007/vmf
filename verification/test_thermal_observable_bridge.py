"""Focused tests for the live finite-temperature observable bridge."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_thermal_observable_bridge as bridge  # noqa: E402


class ThermalObservableBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Rebuilds the accepted calibration and thermal integrals live; the
        # serialized result artifact is intentionally not used as an input.
        cls.result = bridge.build_result()

    def test_live_pass_fixed_grid_and_public_contract(self):
        result = self.result
        self.assertEqual(result["status"], bridge.STATUS_PASS)
        self.assertEqual(result["evidence_weight"], "0.0")
        self.assertTrue(result["physical_inputs_fixed"])
        self.assertFalse(result["protocol_frozen_before_rows"])
        self.assertIn("AFTER_P0_PILOT", result["protocol_provenance"])
        self.assertEqual(result["inputs"]["anchors"], list(bridge.ANCHORS))
        self.assertEqual(result["inputs"]["temperature_MeV"], list(bridge.TEMPERATURES))
        self.assertEqual(result["inputs"]["chemical_potential_MeV"], list(bridge.CHEMICAL_POTENTIALS))
        self.assertEqual(set(result["anchors"]), set(bridge.ANCHORS))
        self.assertTrue(bridge.validate_result(result))

    def test_eight_live_states_are_locally_stable_and_thermodynamic(self):
        for target in bridge.ANCHORS:
            for temperature in bridge.TEMPERATURES:
                for mu in bridge.CHEMICAL_POTENTIALS:
                    combo = self.result["anchors"][target][temperature][mu]
                    self.assertEqual(set(combo["models"]), set(bridge.MODEL_ORDER))
                    for name in bridge.MODEL_ORDER:
                        model = combo["models"][name]
                        state = model["state"]
                        self.assertEqual(model["amplitude"], bridge.MODEL_AMPLITUDES[name])
                        self.assertTrue(state["local_stable_C_positive"])
                        self.assertTrue(state["local_stable_fnn_positive"])
                        self.assertLess(float(state["gap_residual_relative"]), 1e-25)
                        self.assertLess(float(state["gibbs_identity_relative"]), 2e-20)
                        self.assertLess(float(state["pressure_kinetic_log_relative"]), 2e-20)
                        self.assertLess(float(state["chi1_vs_schur_relative"]), 1e-25)
                        self.assertTrue(model["root_discovery"]["distinct_root_count"] >= 1)
                        self.assertEqual(len(model["root_discovery"]["attempted_starts"]), 2)
                        self.assertFalse(model["root_discovery"]["global_phase_claim"])
                        self.assertTrue(model["root_discovery"]["bounded_scan_window"]["not_exhaustive"])

    def test_implicit_susceptibilities_and_independent_controls(self):
        for target in bridge.ANCHORS:
            for temperature in bridge.TEMPERATURES:
                for mu in bridge.CHEMICAL_POTENTIALS:
                    for name in bridge.MODEL_ORDER:
                        model = self.result["anchors"][target][temperature][mu]["models"][name]
                        state = model["state"]
                        self.assertTrue(all(mp.isfinite(mp.mpf(str(state[key]))) for key in ("R21", "R32", "R42")))
                        fd = model["finite_difference_controls"]
                        self.assertTrue(fd["all_pass"])
                        self.assertTrue(fd["steps"][-1]["trend_pass"])
                        self.assertLess(float(fd["steps"][-1]["max_relative_error"]), 5e-3)
                        station = model["stationary_pressure_controls"]
                        self.assertTrue(station["pass"])
                        quad = model["quadrature_controls"]
                        self.assertTrue(quad["pass"])

    def test_controls_and_low_temperature_limit(self):
        self.assertTrue(self.result["controls"]["ideal_fixed_field"]["all_pass"])
        self.assertTrue(self.result["controls"]["classical_particle_and_antiparticle"]["pass"])
        for target in bridge.ANCHORS:
            low = self.result["low_temperature_fixed_density"][target]
            self.assertTrue(low["monotone_y_error"])
            self.assertTrue(all(row["pass"] for row in low["rows"]))
            self.assertLess(float(abs(mp.mpf(low["rows"][-1]["y_difference"]))), 3e-4)

    def test_deformation_effects_are_live_and_not_experimental_scores(self):
        effects = self.result["deformation_effects"]
        self.assertEqual(len(effects), 4)
        self.assertTrue(all(row["classification"] == "resolved_live_difference" for row in effects.values()))
        self.assertTrue(all("combined_resolution_absolute" in row and "u16_resolution_absolute" in row for row in effects.values()))
        self.assertTrue(any(abs(float(row["u16_minus_w8"]["R42"])) > 1e-8 for row in effects.values()))
        self.assertTrue(any("experimental" in text for text in self.result["interpretation_limits"]))

    def test_mutation_fails_closed_without_result_json_as_input(self):
        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["70"]["875"]["models"]["u16"]["state"]["R42"] = "999"
        self.assertFalse(bridge.validate_result(mutated))

    def test_control_gate_cannot_be_forged_from_failed_subtrees(self):
        mutated = copy.deepcopy(self.result)
        mutated["low_temperature_fixed_density"]["0.90"]["rows"][1]["pass"] = False
        self.assertFalse(bridge._all_controls_pass(mutated))
        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["70"]["775"]["models"]["w8"]["state"]["chi1_vs_schur_relative"] = "1"
        self.assertFalse(bridge._all_controls_pass(mutated))
        mutated = copy.deepcopy(self.result)
        mutated["controls"]["ideal_fixed_field"]["all_pass"] = False
        self.assertFalse(bridge._all_controls_pass(mutated))
        mutated = copy.deepcopy(self.result)
        mutated["status"] = bridge.STATUS_FAIL
        self.assertFalse(bridge.validate_result(mutated))

    def test_invalid_inputs_singular_branch_and_precision_controls(self):
        with self.assertRaises(bridge.ThermalBridgeError):
            bridge._integrator_for(0)
        with self.assertRaises(bridge.ThermalBridgeError):
            bridge._mp("NaN")
        with self.assertRaises(bridge.ThermalBridgeError):
            bridge._select_root([{"stable": False, "y": mp.mpf(".9"), "nu": mp.mpf("700")}], mp.mpf(".9"))
        with self.assertRaises((ZeroDivisionError, ValueError, ArithmeticError)):
            bridge._solve_linear(((1, 1), (2, 2)), (1, 1))
        with mp.workdps(70):
            before = mp.mp.dps
            integrator = bridge._integrator_for("8", focused=True, focus_p="250")
            self.assertEqual(mp.mp.dps, before)
            self.assertGreater(len(integrator._grid), 0)

    def test_focused_quadrature_cache_keys_include_width(self):
        bridge._gauss_grid.cache_clear()
        grid_a = bridge._gauss_grid(40, 48, "1000", "250", "100")
        grid_b = bridge._gauss_grid(40, 48, "1000", "250", "200")
        grid_a_again = bridge._gauss_grid(40, 48, "1000", "250", "100")
        self.assertNotEqual(grid_a[0][0], grid_b[0][0])
        self.assertEqual(grid_a, grid_a_again)

    def test_deformation_resolution_uses_both_model_errors(self):
        fields = ("n_fm3", "pressure_MeV_fm3", "entropy_fm3", "R21", "R32", "R42")
        w = {field: mp.mpf("1") for field in fields}
        u = {field: mp.mpf("1") for field in fields}
        u["R42"] = mp.mpf("1.001")
        exact = {"relative_differences": {field: mp.mpf("0") for field in fields}}
        uncertain_u16 = {"relative_differences": {field: mp.mpf("0") for field in fields}}
        uncertain_u16["relative_differences"]["R42"] = mp.mpf("1e-2")
        row = bridge._classify_deformation_effect(w, u, exact, uncertain_u16)
        self.assertEqual(row["classification"], "numerically_unresolved")
        self.assertEqual(row["resolved_fields"], [])
        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.93"]["70"]["775"]["models"]["w8"]["root_discovery"]["global_phase_claim"] = True
        self.assertFalse(bridge.validate_result(mutated))

    def test_public_source_does_not_reference_run_workspace(self):
        source = (HERE / "nvg_thermal_observable_bridge.py").read_text(encoding="utf-8")
        self.assertNotIn("Lunacy/runs", source)
        self.assertNotIn("evidence_parent", source)
        self.assertNotIn("thermal-observable-data-2026-09-19", source)


if __name__ == "__main__":
    unittest.main()
