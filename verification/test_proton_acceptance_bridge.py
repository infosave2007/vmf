"""Focused controls for the live conditional proton bridge."""
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

import nvg_proton_acceptance_bridge as bridge  # noqa: E402


class ProtonAcceptanceBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One live build is intentionally shared by the focused matrix.  The
        # serialized output file is never used to construct this result.
        cls.result = bridge.build_result()

    def test_live_status_and_full_matrix(self):
        result = self.result
        self.assertEqual(result["status"], bridge.STATUS_PASS)
        self.assertTrue(bridge.validate_result(result))
        self.assertEqual(len(result["state_inputs"]), 8)
        self.assertEqual(len(result["predictions"]), 16)
        self.assertTrue(result["fit_or_score_guardrails"]["parameter_fit"] is False)
        self.assertTrue(result["fit_or_score_guardrails"]["data_used_as_model_input"] is False)
        for row in result["predictions"]:
            self.assertIn("gce", row)
            self.assertIn("constrained", row)
            self.assertTrue(row["data_comparison"]["no_significance_or_chi2"])
            for closure in (row["gce"], row["constrained"]):
                self.assertTrue(all(value is not None for value in closure["ratios"].values()))
                self.assertTrue(mp.mpf(closure["cumulants"]["K1"]) > 0)
                self.assertTrue(mp.mpf(closure["cumulants"]["K2"]) > 0)

    def test_geometry_and_model_controls(self):
        result = self.result
        self.assertTrue(result["controls"]["all_pass"])
        for cut in bridge.RAPIDITY_CUTS:
            check = result["geometry"][cut]
            self.assertTrue(check["pass"])
            self.assertTrue(check["denominator_control"]["pass"])
            self.assertTrue(check["i1_angular_average_control"]["pass"])
            self.assertTrue(check["h0_control"]["pass"])
            self.assertLessEqual(mp.mpf(check["q_sample_max_primary"]), mp.mpf(bridge.Q_FREE))
        self.assertEqual(len(result["model_difference_controls"]), 8)

    def test_live_index_conversion_and_antiparticle_boundary(self):
        for state in self.result["state_inputs"]:
            self.assertTrue(max(mp.mpf(value) for value in state["bulk"]["index_conversion_relative_errors"]) < mp.mpf("1e-24"))
            indicator = state["antiparticle_indicator"]
            self.assertTrue(indicator["pass"])
            self.assertLess(mp.mpf(indicator["antiparticle_to_particle_ratio"]), mp.mpf(bridge.ANTIPARTICLE_THRESHOLD))

    def test_standard_formula_controls(self):
        self.assertTrue(bridge._all_controls_pass(self.result))
        extremes = self.result["controls"]["extreme_acceptance"]
        self.assertEqual(extremes["full_charge_q1"]["ratios"]["K3_over_K2"], None)
        self.assertEqual(extremes["empty_charge_q0"]["ratios"]["K2_over_K1"], None)

    def test_mutation_fails_closed(self):
        mutated = copy.deepcopy(self.result)
        mutated["predictions"][0]["constrained"]["ratios"]["K2_over_K1"] = "999"
        self.assertFalse(bridge.validate_result(mutated))

    def test_source_and_data_boundaries(self):
        source = (HERE / "nvg_proton_acceptance_bridge.py").read_text(encoding="utf-8")
        self.assertNotIn("Lunacy/runs", source)
        self.assertNotIn("evidence_parent", source)
        self.assertNotIn("nvg_thermal_observable_bridge_results.json", source)
        self.assertEqual(self.result["data"]["acceptance"]["centralities"], ["0-5%"])
        self.assertIsNone(self.result["data"]["uncertainty_boundary"]["systematics"])
        self.assertIsNone(self.result["data"]["uncertainty_boundary"]["covariance"])

    def test_input_validation_and_undefined_extremes(self):
        with self.assertRaises(bridge.ProtonBridgeError):
            bridge.validate_moments(["1", "0.3", "-0.1", "0", "0"])
        with self.assertRaises(bridge.ProtonBridgeError):
            bridge.validate_moments(["1", "0.3", "0.08", "0.1", "0.01"], qmax="0.2")
        with self.assertRaises(bridge.ProtonBridgeError):
            bridge._validate_probability("1.1")
        extremes = bridge._control_extreme_acceptance()
        self.assertTrue(extremes["full_charge_q1"]["pass"])
        self.assertTrue(extremes["empty_charge_q0"]["pass"])

    def test_support_moment_realizability_is_scale_aware(self):
        for volume in (mp.mpf("1e-120"), mp.mpf("1"), mp.mpf("1e120")):
            q = mp.mpf("0.2")
            bridge.validate_moments([volume * q**j for j in range(5)], qmax="0.25")
            alpha = mp.mpf("0.35")
            bridge.validate_moments([volume] + [volume * alpha * q**j for j in range(1, 5)], qmax="0.25")
        bridge.validate_moments(["1", "0", "0", "0", "0"], qmax="0")
        with self.assertRaises(bridge.ProtonBridgeError):
            bridge.validate_moments(["1", "1e-10", "1e-25", "0", "0"], qmax="0")
        with self.assertRaises(bridge.ProtonBridgeError):
            bridge.validate_moments(["1", "0.1", "0.005", "0.0001", "0.000005"], qmax="1")
        with self.assertRaises(bridge.ProtonBridgeError):
            bridge.validate_moments(["1", "0.2", "0.04", "0.008", "0.5"], qmax="1")

    def test_nonfinite_and_out_of_support_acceptance_inputs_fail_closed(self):
        for xi in (float("nan"), float("inf"), -float("inf"), 2.0):
            with self.assertRaises(bridge.ProtonBridgeError):
                bridge._local_acceptance(1.0, xi, 0.2, pt_count=16, rapidity_count=16)
        for radius, cut, H in ((float("nan"), 0.2, None), (1.0, float("nan"), None), (1.0, 0.2, -0.1), (-1.0, 0.2, None)):
            with self.assertRaises(bridge.ProtonBridgeError):
                bridge._local_acceptance(radius, 0.0, cut, pt_count=16, rapidity_count=16, h_fm_inv=H)

    def test_all_production_gates_are_fail_closed(self):
        for path in (
            ("predictions", 0, "closure_refinement", "pass"),
            ("predictions", 0, "variance_diagnostics", "pass"),
            ("predictions", 0, "variance_diagnostics", "bound_pass_gce"),
            ("state_inputs", 0, "antiparticle_indicator", "pass"),
        ):
            mutated = copy.deepcopy(self.result)
            target = mutated
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = False
            self.assertFalse(bridge._all_controls_pass(mutated), path)

    def test_model_resolution_uses_actual_refinement_bounds(self):
        rows = [copy.deepcopy(row) for row in self.result["predictions"] if row["anchor"] == "0.90" and row["mu_MeV"] == "775" and row["rapidity_halfwidth"] == "0.2"]
        self.assertEqual({row["model"] for row in rows}, {"w8", "u16"})
        first, second = rows
        field = "gce.K2_over_K1"
        section, name = field.split(".")
        second[section]["ratios"][name] = str(mp.mpf(first[section]["ratios"][name]) + mp.mpf("1e-6"))
        second["closure_refinement"]["fields"][field]["absolute_error"] = "1e-3"
        checked = bridge._model_difference([first, second])
        self.assertFalse(checked["field_status"][field]["resolved"])
        self.assertIn(field, checked["unresolved_fields"])

    def test_data_contract_links_panels_and_preserves_observation_only_role(self):
        payload = json.loads(bridge.DATA_PATH.read_text(encoding="utf-8"))
        bad = copy.deepcopy(payload)
        bad["points"][0]["source_series"] = "K3/K2 (0-5%)"
        with self.assertRaises(bridge.ProtonBridgeError):
            bridge._fixed_data(bad)
        bad = copy.deepcopy(payload)
        bad["points"][0]["centrality"] = "0-10%"
        with self.assertRaises(bridge.ProtonBridgeError):
            bridge._fixed_data(bad)
        bad = copy.deepcopy(payload)
        bad["points"][0]["errors"]["statistical"]["symmetric"] = "NaN"
        with self.assertRaises(bridge.ProtonBridgeError):
            bridge._fixed_data(bad)
        bad = copy.deepcopy(payload)
        bad["points"][0]["errors"]["statistical"]["symmetric"] = "-0.1"
        with self.assertRaises(bridge.ProtonBridgeError):
            bridge._fixed_data(bad)
        bad = copy.deepcopy(payload)
        bad["points"] = bad["points"][:-1]
        with self.assertRaises(bridge.ProtonBridgeError):
            bridge._fixed_data(bad)
        observed = bridge._fixed_data()["points"]["0.2"]
        closure = self.result["predictions"][0]
        original = bridge._data_comparison(closure, observed)
        changed = copy.deepcopy(observed)
        changed["K2/K1"]["value"] = "9.0"
        altered = bridge._data_comparison(closure, changed)
        self.assertEqual(closure["gce"]["ratios"], self.result["predictions"][0]["gce"]["ratios"])
        self.assertNotEqual(original["residuals_model_minus_observation"], altered["residuals_model_minus_observation"])

    def test_relative_inversion_guard_is_volume_invariant(self):
        c = [mp.mpf("1"), mp.mpf("1"), mp.mpf("1"), mp.mpf("1")]
        q = mp.mpf("0.2")
        row = bridge._fixed_data()["points"]["0.2"]
        closure = bridge._closure(c, [mp.mpf("1"), q, q**2, q**3, q**4], qmax=1)
        first = bridge._variance_diagnostics(c, [mp.mpf("1"), q, q**2, q**3, q**4], row, closure)
        scaled = bridge._variance_diagnostics(c, [mp.mpf("1e120"), mp.mpf("1e120") * q, mp.mpf("1e120") * q**2, mp.mpf("1e120") * q**3, mp.mpf("1e120") * q**4], row, bridge._closure(c, [mp.mpf("1e120"), mp.mpf("1e120") * q, mp.mpf("1e120") * q**2, mp.mpf("1e120") * q**3, mp.mpf("1e120") * q**4], qmax=1))
        self.assertEqual(first["inverse_status"], "UNIDENTIFIABLE_VARQ_ZERO_OR_BELOW_GUARD")
        self.assertEqual(first["inverse_status"], scaled["inverse_status"])
        self.assertIsNone(first["required_r2_constrained_for_observed_central"])


if __name__ == "__main__":
    unittest.main()
