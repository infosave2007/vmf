#!/usr/bin/env python3
"""Focused controls for the source -> retarded NVG -> added RLC producer."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nvg_driven_density_load as producer  # noqa: E402
import nvg_retarded_response as retarded  # noqa: E402


class DrivenDensityLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = producer.build_result()
        cls.rows = cls.result["rows"]
        cls.backgrounds = producer._backgrounds()

    def test_result_schema_and_declared_coverage(self) -> None:
        self.assertTrue(producer.validate_result(self.result))
        self.assertEqual(len(self.rows), 54)
        self.assertEqual(self.result["coverage"]["base_row_count"], 18)
        self.assertEqual(self.result["coverage"]["loaded_row_count"], 54)
        self.assertEqual(self.result["full_coordinate_order"], ["n", "y", "A0", "A_L", "Q"])
        self.assertEqual({row["load"]["R_hat"] for row in self.rows}, {"0.25", "1", "4"})

    def test_full_five_by_five_and_schur_match(self) -> None:
        max_error = max(float(row["response"]["direct_vs_schur_relative_error"]) for row in self.rows)
        scalar_error = max(float(row["response"]["scalar_identity_relative_error"]) for row in self.rows)
        self.assertLess(max_error, 3e-12)
        self.assertLess(scalar_error, 3e-12)
        row = self.rows[0]
        # The serialized matrix is the solution vector; use a fresh exact
        # solve to inspect the declared off-diagonal signs and Q block.
        bg = self.backgrounds[0]
        q = float(row["q_MeV"])
        omega = float(row["omega_MeV"])
        pi = retarded.polarization_kernel(bg["mass"], bg["kF"], q, omega, d=bg["model"].d)
        ans = retarded.retarded_response(bg["model"], bg["state"], q, omega, pi, 1.0)
        h4 = np.asarray(ans["H_normalized"], complex)
        loaded = producer.solve_loaded(h4, omega / q, R_hat=float(row["load"]["R_hat"]))
        om = omega / q
        self.assertAlmostEqual(loaded["full_matrix"][0, 4].real, 0.0, places=14)
        self.assertAlmostEqual(loaded["full_matrix"][0, 4].imag, -om * producer.G_HAT, places=14)
        self.assertAlmostEqual(loaded["full_matrix"][4, 0].imag, om * producer.G_HAT, places=14)
        self.assertAlmostEqual(loaded["full_matrix"][4, 4].real, 1 - om * om, places=12)
        self.assertAlmostEqual(loaded["full_matrix"][4, 4].imag, -om * float(row["load"]["R_hat"]), places=12)

    def test_current_identity_and_power_balance(self) -> None:
        current_max = max(float(row["response"]["current_identity_relative_error"]) for row in self.rows)
        balance_max = max(abs(float(row["response"]["power_balance_residual_normalized"])) for row in self.rows)
        self.assertLess(current_max, 3e-12)
        self.assertLess(balance_max, 3e-12)
        self.assertTrue(all(row["physical_watts"] is None and row["physical_efficiency"] is None for row in self.rows))

    def test_outside_cut_formal_eta_is_visible_not_claimed(self) -> None:
        outside = [row for row in self.rows if row["omega_fraction"] == "1.25"]
        self.assertEqual(len(outside), 18)
        for row in outside:
            response = row["response"]
            self.assertAlmostEqual(float(response["P_intrinsic_normalized"]), 0.0, places=12)
            eta = float(response["eta_normalized"])
            self.assertAlmostEqual(eta, 1.0, places=10)
            self.assertEqual(response["eta_interpretation"], "per_unit_algebra_only; not hardware efficiency")

    def test_source_off_g0_r0_and_invalid_active_load(self) -> None:
        h = np.asarray([[2 - .1j, .2, .1, 0], [.2, 3, 0, 0], [.1, 0, 1.5, .1], [0, 0, .1, 2]], complex)
        off = producer.solve_loaded(h, .7, source_hat=0)
        self.assertLess(float(np.max(np.abs(off["direct_solution"]))), 1e-14)
        g0 = producer.solve_loaded(h, .7, G_hat_value=0)
        k, *_ = producer._effective_density_inverse(h)
        self.assertLess(producer._rel(g0["n_hat"], 1 / k), 1e-12)
        r0 = producer.solve_loaded(h, .7, R_hat=0)
        self.assertAlmostEqual(r0["P_out_normalized"], 0.0, places=14)
        with self.assertRaises(producer.DrivenLoadError):
            producer.validate_load_parameters(R_hat=-.1)
        with self.assertRaises(producer.DrivenLoadError):
            producer.validate_load_parameters(R_hat=1 + .2j)
        with self.assertRaises(producer.DrivenLoadError):
            producer.rlc_impedance(0)
        with self.assertRaises(producer.DrivenLoadError):
            producer.solve_loaded(np.zeros((4, 4), dtype=complex), .7)

    def test_loaded_dc_and_small_frequency_limit(self) -> None:
        h = np.asarray([[2 - .1j, .2, .1, 0], [.2, 3, 0, 0], [.1, 0, 1.5, .1], [0, 0, .1, 2]], complex)
        dc = producer.solve_loaded(h, 0.0, R_hat=1.0)
        near = producer.solve_loaded(h, 1e-5, R_hat=1.0)
        self.assertIsNone(dc["Z_e"])
        self.assertEqual(dc["P_in_normalized"], 0.0)
        self.assertEqual(dc["P_out_normalized"], 0.0)
        self.assertEqual(dc["I_hat"], 0j)
        self.assertIsNone(dc["eta_normalized"])
        self.assertLess(producer._rel(dc["n_hat"], near["n_hat"], floor=abs(dc["n_hat"])), 1e-8)

    def test_dressed_denominator_and_active_controls(self) -> None:
        controls = self.result["controls"]
        dressed = controls["dressed_denominator_controls"]
        self.assertTrue(dressed["bare_K0_regularized_finite"])
        self.assertTrue(dressed["bare_Ze0_regularized_finite"])
        self.assertEqual(dressed["decoupled_singular_rejection"], "DrivenLoadError")
        self.assertEqual(dressed["tuned_loaded_singular_rejection"], "DrivenLoadError")
        active = controls["active_intrinsic_negative_control"]
        self.assertGreater(float(active["Im_K_NVG"]), 0.0)
        self.assertTrue(active["eta_is_null"])
        tiny = controls["tiny_positive_intrinsic_sign_control"]
        self.assertGreater(float(tiny["Im_K_NVG"]), 0.0)
        self.assertTrue(tiny["sign_indeterminate"])
        self.assertEqual(tiny["intrinsic_passivity_status"], "INDETERMINATE_SIGNED_RESPONSE")
        self.assertTrue(tiny["eta_is_null"])
        self.assertLess(float(tiny["signed_P_intrinsic_normalized"]), 0.0)

    def test_matching_control_and_lossless_guard(self) -> None:
        matching = self.result["controls"]["matching_control"]
        self.assertLess(float(matching["max_eta_error"]), 1e-12)
        self.assertLess(float(matching["max_available_ratio_error"]), 1e-12)
        self.assertEqual(matching["conjugate_match_eta"], "0.5")
        self.assertEqual(matching["lossless_guard"]["status"], "LOSSLESS_MATCHING_UNDEFINED_NO_DIVISION")

    def test_static_limit_and_coordinate_rescaling_controls(self) -> None:
        controls = self.result["controls"]
        self.assertTrue(controls["static_limit_control"]["finite_density_response"])
        self.assertLess(float(controls["static_limit_control"]["direct_vs_schur_relative_error"]), 1e-12)
        self.assertLess(float(controls["congruence_rescaling_relative_error"]), 1e-12)

    def test_synthetic_positive_energy_work_control(self) -> None:
        synthetic = self.result["controls"]["synthetic_time_verifier"]
        self.assertEqual(synthetic["status"], "PASS_DIMENSIONLESS_SYNTHETIC_WORK_IDENTITY")
        self.assertTrue(synthetic["zero_initial_energy"])
        self.assertTrue(synthetic["finite_discharge_bounded_by_stored_energy"])
        self.assertLess(abs(float(synthetic["work_balance_residual"])), 3e-5)

    def test_provenance_and_normalization_passport(self) -> None:
        receiver = self.result["receiver"]
        self.assertTrue(receiver["not_derived_from_closed_NVG_action"])
        self.assertIsNone(receiver["physical_mapping"])
        passport = receiver["normalization_passport"]
        self.assertFalse(passport["physical_units"])
        self.assertFalse(passport["device_mapping"])
        self.assertIn("Veff*n0*W0", passport["E_star"])
        self.assertIn("q/hbar", passport["omega_star"])
        self.assertIn("P_hat", passport["definitions"])
        self.assertIn("pressure_port", self.result)
        self.assertIsNone(self.result["pressure_port"]["Veff_and_overlap"])
        self.assertTrue(Path(self.result["provenance"]["retarded_source"]).is_file())
        self.assertEqual(len(self.result["provenance"]["retarded_source_sha256"]), 64)

    def test_cli_strict_json_default_is_no_write(self) -> None:
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run([sys.executable, "-B", str(HERE / "nvg_driven_density_load.py")],
                                   cwd=HERE.parent, env=env, check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        parsed = json.loads(completed.stdout)
        self.assertTrue(producer.validate_result(parsed))
        self.assertEqual(completed.stderr, "")

    def test_result_mutation_fails_closed(self) -> None:
        mutated = json.loads(json.dumps(self.result))
        mutated["rows"][0]["load"]["R_hat"] = "-1"
        self.assertFalse(producer.validate_result(mutated))
        mutated = json.loads(json.dumps(self.result))
        mutated["scenario_flag"] = "HARDWARE"
        self.assertFalse(producer.validate_result(mutated))


if __name__ == "__main__":
    unittest.main()
