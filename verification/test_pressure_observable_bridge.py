"""Focused tests for the live reduced-pressure observable bridge."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_pressure_observable_bridge as bridge  # noqa: E402


class PressureObservableBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = bridge.build_result()

    def test_exact_quartic_units_and_factorials(self):
        with mp.workdps(90):
            n0 = mp.mpf("0.16")
            h = mp.mpf("0.10")
            K, Q, Z = mp.mpf("240"), mp.mpf("37"), mp.mpf("2.5")

            def e_x(x):
                return K * x / 9 + Q * x**2 / 54 + Z * x**3 / 486

            def pressure(x):
                return n0 * (1 + x) ** 2 * e_x(x)

            points = [pressure(multiplier * h) for multiplier in (-2, -1, 1, 2)]
            contrast = bridge.pressure_contrast(points, n0_fm3=n0, h=h)
            self.assertLess(abs(contrast["Z_h_MeV"] - Z), mp.mpf("1e-70"))
            # The same pressure in natural units must reduce to the same physical
            # contrast after P_nat/(hbar*c)^3 conversion.
            hc = mp.mpf(bridge.HBARC_MEV_FM)
            # Independent natural-unit check: P_nat=n_nat^2*d e/d n_nat.
            natural_points = []
            for multiplier in (-2, -1, 1, 2):
                x = multiplier * h
                n_nat = n0 * hc**3 * (1 + x)
                de_dn_nat = e_x(x) / (n0 * hc**3)
                natural_points.append(n_nat**2 * de_dn_nat)
            converted = bridge.pressure_contrast(
                [value / hc**3 for value in natural_points], n0_fm3=n0, h=h
            )
            self.assertLess(abs(converted["Z_h_MeV"] - Z), mp.mpf("1e-70"))

    def test_independent_polynomial_nuisances_cancel_in_reduced_pressure(self):
        with mp.workdps(90):
            n0, h = mp.mpf("0.16"), mp.mpf("0.10")
            baseline = lambda x: mp.mpf("2.5") * x**3 / 486
            nuisance = lambda x: mp.mpf("7.0") + mp.mpf("3.0") * x - mp.mpf("11.0") * x**2
            points = [n0 * (1 + x) ** 2 * (baseline(x) + nuisance(x)) for x in (-2*h, -h, h, 2*h)]
            contrast = bridge.pressure_contrast(points, n0_fm3=n0, h=h)
            expected = mp.mpf("2.5")
            self.assertLess(abs(contrast["Z_h_MeV"] - expected), mp.mpf("1e-70"))

    def test_gaussian_quantiles_and_no_sqrt2_budget(self):
        q95 = bridge.standard_normal_quantile("0.95")
        q80 = bridge.standard_normal_quantile("0.80")
        self.assertAlmostEqual(float(bridge.standard_normal_cdf(q95)), 0.95, places=14)
        self.assertAlmostEqual(float(bridge.standard_normal_cdf(q80)), 0.80, places=14)
        budget = bridge.gaussian_discrimination_budget("2", "4", alpha="0.05", power="0.80")
        expected_factor = q95 + q80
        self.assertAlmostEqual(float(budget["mean_separation_factor"]), float(expected_factor), places=25)
        self.assertAlmostEqual(float(budget["sigmaP_max_MeV_fm3"]), float(2 / (4 * expected_factor)), places=25)
        self.assertFalse(budget["sqrt2_for_two_data_sets"])
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.gaussian_discrimination_budget("2", "4", alpha="0")
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.standard_normal_quantile("1")
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.gaussian_discrimination_budget("0", "100", alpha="0.05", power="0.80")

    def test_covariance_offdiagonal_psd_and_uncertainty(self):
        pressures = ["-0.2", "-0.1", "0.1", "0.2"]
        diagonal = [["0.01" if i == j else "0" for j in range(4)] for i in range(4)]
        result = bridge.pressure_contrast(pressures, covariance=diagonal)
        weights = result["weights_on_pressure_fm3"]
        expected_variance = mp.mpf("0.01") * sum(weight * weight for weight in weights)
        self.assertAlmostEqual(float(result["variance_Z_MeV2"]), float(expected_variance), places=20)
        correlated = [[mp.mpf("0.01") if i == j else mp.mpf("0.002") for j in range(4)] for i in range(4)]
        correlated_result = bridge.pressure_contrast(pressures, covariance=correlated)
        self.assertGreater(float(correlated_result["sigma_Z_MeV"]), 0.0)
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast(pressures, covariance=[[1, 0], [0, 1]])
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast(pressures, covariance=[[1, 2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast(pressures, covariance=[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast(pressures, covariance=diagonal, covariance_units=None)

    def test_ambient_precision_accepts_rank_one_psd_and_preserves_null_variance(self):
        pressures = ["-0.7", "-0.3", "0.4", "1"]
        with mp.workdps(15):
            for vector in ([1, 2, 3, 4], [1, -2, 3, -4]):
                covariance = [
                    [str(mp.mpf("1e-12") * left * right) for right in vector]
                    for left in vector
                ]
                result = bridge.pressure_contrast(pressures, covariance=covariance)
                with mp.workdps(90):
                    expected = (
                        sum(
                            weight * mp.mpf(str(component))
                            for weight, component in zip(result["weights_on_pressure_fm3"], vector)
                        )
                        ** 2
                        * mp.mpf("1e-12")
                    )
                    self.assertLess(abs(result["variance_Z_MeV2"] - expected) / expected, mp.mpf("1e-50"))

            # Choose a rank-one direction orthogonal to the contrast weights;
            # valid PSD with intentionally tiny propagated variance.
            with mp.workdps(90):
                baseline = bridge.pressure_contrast(pressures)
                weights = baseline["weights_on_pressure_fm3"]
                vector = (weights[1], -weights[0], mp.mpf(0), mp.mpf(0))
                covariance = [
                    [mp.nstr(mp.mpf("1e-12") * left * right, 80) for right in vector]
                    for left in vector
                ]
            null_result = bridge.pressure_contrast(pressures, covariance=covariance)
            self.assertLess(float(null_result["variance_Z_MeV2"]), 1e-120)

    def test_scaled_materially_indefinite_covariance_is_rejected(self):
        vector = ["1", "2", "3", "4"]
        covariance = [[mp.mpf("1e-12") * mp.mpf(left) * mp.mpf(right) for right in vector] for left in vector]
        covariance[0][0] -= mp.mpf("1e-10")
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast(["-0.7", "-0.3", "0.4", "1"], covariance=covariance)

    def test_covariance_psd_is_invariant_under_positive_rescaling(self):
        pressures = ["-0.7", "-0.3", "0.4", "1"]
        vector = (1, 2, 3, 4)
        with mp.workdps(100):
            for scale_text in ("1e-120", "1", "1e120"):
                scale = mp.mpf(scale_text)
                covariance = [
                    [mp.nstr(scale * left * right, 90) for right in vector]
                    for left in vector
                ]
                result = bridge.pressure_contrast(pressures, covariance=covariance)
                expected = (
                    sum(
                        weight * component
                        for weight, component in zip(result["weights_on_pressure_fm3"], vector)
                    )
                    ** 2
                    * scale
                )
                self.assertLess(abs(result["variance_Z_MeV2"] - expected) / expected, mp.mpf("1e-70"))

                negative = [
                    [mp.nstr(-scale if i == j == 0 else scale if i == j else 0, 90) for j in range(4)]
                    for i in range(4)
                ]
                with self.assertRaises(bridge.PressureBridgeError):
                    bridge.pressure_contrast(pressures, covariance=negative)

                asymmetric = [["0"] * 4 for _ in range(4)]
                asymmetric[0][1] = mp.nstr(scale, 90)
                with self.assertRaises(bridge.PressureBridgeError):
                    bridge.pressure_contrast(pressures, covariance=asymmetric)

            zero = bridge.pressure_contrast(pressures, covariance=[["0"] * 4 for _ in range(4)])
            self.assertEqual(zero["variance_Z_MeV2"], 0)

    def test_invalid_units_order_shape_and_nonfinite_inputs_fail_closed(self):
        pressures = [1, 2, 3, 4]
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast(pressures, pressure_units="MeV")
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast(pressures, density_units="MeV^-3")
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast(pressures, n0_fm3=0)
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast(pressures, h=0)
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast(pressures, x_nodes=[0.2, 0.1, -0.1, -0.2])
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast([1, 2, 3])
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast([1, 2, 3, "NaN"])
        with self.assertRaises(bridge.PressureBridgeError):
            bridge.pressure_contrast([1, 2, 3, 4], x_nodes=[-0.2, -0.1, 0.1, 0.3])

    def test_live_pass_schema_rows_and_branch_controls(self):
        result = self.result
        self.assertEqual(result["status"], bridge.STATUS_PASS)
        self.assertEqual(result["evidence_weight"], "0.0")
        self.assertEqual(result["inputs"]["anchors"], ["0.90", "0.93"])
        self.assertEqual(result["inputs"]["x_nodes"], list(bridge.X_NODES))
        self.assertFalse(result["protocol_frozen_before_rows"])
        self.assertTrue(result["physical_inputs_fixed"])
        self.assertTrue(bridge.validate_result(result))
        for target in bridge.ANCHORS:
            anchor = result["anchors"][target]
            self.assertEqual(set(anchor["models"]), set(bridge.MODEL_ORDER))
            self.assertEqual(set(anchor["contrasts"]), {"0.1", "0.05", "0.025"})
            for name in bridge.MODEL_ORDER:
                model = anchor["models"][name]
                self.assertEqual(model["amplitude"], bridge.MODEL_AMPLITUDES[name])
                self.assertEqual(len(model["rows"]), len(bridge.X_NODES))
                self.assertTrue(all(row["positive_local_branch"] for row in model["rows"]))
                self.assertTrue(all(float(row["C_y"]) > 0 and float(row["mu_prime"]) > 0 for row in model["rows"]))
                self.assertTrue(all(float(row["stationarity_relative"]) < 1e-60 for row in model["rows"]))
            for contrast in anchor["contrasts"].values():
                self.assertGreater(float(contrast["delta_Z_h_MeV"]), 0.0)
                self.assertGreater(float(contrast["delta_Z_live_MeV"]), 0.0)
                self.assertGreater(float(contrast["gaussian_budget"]["sigmaP_max_MeV_fm3"]), 0.0)

    def test_live_precision_bias_and_halving_controls(self):
        self.assertTrue(self.result["precision_control_summary"]["all_pass"])
        self.assertEqual(len(self.result["precision_controls"]), 4)
        self.assertTrue(all(item["pass"] for item in self.result["precision_controls"]))
        for anchor in self.result["anchors"].values():
            for contrast in anchor["contrasts"].values():
                for model in contrast["models"].values():
                    self.assertNotEqual(model["taylor_bias_MeV"], "0.0")
            for diagnostic in anchor["halving_diagnostics"]:
                self.assertGreater(float(diagnostic["noise_norm_growth"]), 6.0)
                self.assertLess(float(diagnostic["differential_bias_ratio"]), 0.5)

    def test_unfavourable_declared_row_is_retained(self):
        original = bridge._pressure_row

        def fail_one(model, target_y, x):
            if x == mp.mpf("-0.20"):
                raise bridge.PressureBridgeError("manufactured row failure")
            return original(model, target_y, x)

        with mock.patch.object(bridge, "_pressure_row", side_effect=fail_one):
            snapshot = bridge._snapshot(80)
        rows = snapshot["anchors"]["0.90"]["models"]["w8"]["rows"]
        self.assertEqual(len(rows), len(bridge.X_NODES))
        self.assertEqual(rows[0]["row_status"], "FAIL")
        self.assertTrue(bridge._snapshot_has_failures(snapshot))

    def test_result_validation_rejects_scientific_mutation_and_flags(self):
        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["contrasts"]["0.1"]["delta_Z_h_MeV"] = "999"
        self.assertFalse(bridge.validate_result(mutated))
        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.93"]["models"]["w8"]["rows"][0]["pressure_MeV_fm3"] = "999"
        self.assertFalse(bridge.validate_result(mutated))
        mutated = copy.deepcopy(self.result)
        mutated["precision_controls"][0]["pass"] = False
        self.assertFalse(bridge.validate_result(mutated))
        mutated = copy.deepcopy(self.result)
        mutated["status"] = "PASS_FORGED"
        self.assertFalse(bridge.validate_result(mutated))

    def test_cli_default_is_strict_json_and_does_not_write_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = str(HERE)
            completed = subprocess.run(
                [sys.executable, "-B", str(HERE / "nvg_pressure_observable_bridge.py")],
                cwd=temporary,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(completed.stderr, "")
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], bridge.STATUS_PASS)
            self.assertNotIn("NaN", completed.stdout)
            self.assertNotIn("Infinity", completed.stdout)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
