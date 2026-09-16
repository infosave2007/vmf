"""Focused live controls for the bounded static-response audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_spatial_identifiability_audit as audit  # noqa: E402


class SpatialIdentifiabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit.build_result()

    @staticmethod
    def close(a, b, tolerance="1e-45"):
        with mp.workdps(120):
            a, b = mp.mpf(str(a)), mp.mpf(str(b))
            if not mp.isfinite(a) or not mp.isfinite(b):
                raise AssertionError("nonfinite comparison")
            error = abs(a - b) / max(abs(a), abs(b), mp.mpf(1))
            if error > mp.mpf(tolerance):
                raise AssertionError(f"relative error {error} > {tolerance}")

    def test_schema_grid_hashes_and_public_scope(self):
        result = self.result
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["status"], audit.STATUS)
        self.assertEqual(result["evidence_weight"], 0.0)
        self.assertEqual(result["coverage"]["row_count"], 72)
        self.assertTrue(result["coverage"]["all_rows_present"])
        self.assertEqual(result["inputs_and_scope"]["density_ratios_Q4"], list(audit.DENSITY_RATIOS))
        self.assertEqual(result["inputs_and_scope"]["W8_target_y"], list(audit.W8_TARGETS))
        self.assertEqual(result["inputs_and_scope"]["scales"], list(audit.SCALES))
        self.assertEqual(result["inputs_and_scope"]["wave_numbers_MeV"], list(audit.WAVE_NUMBERS_MEV))
        self.assertEqual(result["source_sha256"], hashlib.sha256(audit.SOURCE_PATH.read_bytes()).hexdigest())
        self.assertEqual(result["upstream_source_sha256"], hashlib.sha256(audit.UPSTREAM_PATH.read_bytes()).hexdigest())
        self.assertEqual(result["contract_sha256"], hashlib.sha256(audit.CONTRACT_PATH.read_bytes()).hexdigest())
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
        parsed = json.loads(payload)
        self.assertEqual(parsed["coverage"], result["coverage"])
        public_text = payload + audit.CONTRACT_PATH.read_text(encoding="utf-8")
        for forbidden in (".codex", "Lunacy/", ".work", "private", "hidden"):
            self.assertNotIn(forbidden, public_text)

    def test_all_72_rows_are_retained_and_finite(self):
        rows = self.result["rows"]
        expected = {
            (family, density, target, scale, k)
            for family, density, target in (
                [("Q4", ratio, None) for ratio in audit.DENSITY_RATIOS]
                + [("W8", "1", target) for target in audit.W8_TARGETS]
            )
            for scale in audit.SCALES
            for k in audit.WAVE_NUMBERS_MEV
        }
        actual = {(r["family"], r["density_ratio"], r["target_y"], r["scale"], r["k_MeV"].removesuffix(".0")) for r in rows}
        self.assertEqual(actual, expected)
        self.assertTrue(all(r["stable"] and r["status"] == "STABLE_STATIC_TF_RESPONSE" for r in rows))
        self.assertTrue(all(r["direct_status"] == "DIRECT_SOLVE_OK" for r in rows))
        for row in rows:
            self.assertLessEqual(float(row["direct_vs_schur_relative_error"]), 1e-45)
            self.assertIsNotNone(row["chi_MeV2"])
            for key in ("D_MeVminus2", "B_MeV", "C_MeV4", "S_MeVminus2", "chi_MeV2"):
                self.assertNotIn(row[key], ("nan", "+inf", "-inf"))

    def test_full_hessian_derivatives_direct_solve_and_mixed_sign_control(self):
        for background in self.result["backgrounds"]:
            check = background["full_hessian_derivative_check"]
            self.assertTrue(check["passed"], background["background_id"])
            for error in check["relative_errors"].values():
                self.assertLessEqual(float(error), 1e-45)
        for control in self.result["negative_controls"]:
            self.assertTrue(control["detected_and_rejected"])
            self.assertGreater(float(control["wrong_sign_derivative_mismatch"]), 1.0)
            self.assertGreater(float(control["wrong_sign_response_relative_error"]), 1e-45)
            self.assertGreater(float(control["omitted_response_relative_error"]), 1e-45)

    def test_k0_is_homogeneous_thermodynamic_match(self):
        for row in self.result["k0_thermodynamic_matches"]:
            self.assertTrue(row["pass"])
            self.assertLessEqual(float(row["S_relative_error"]), 1e-45)
            self.assertLessEqual(float(row["chi_relative_error"]), 1e-45)
            self.assertIn("grand-canonical", row["ensemble_note"])
        self.assertTrue(self.result["scope"]["fixed_total_N_uniform_mode_excluded"])

    def test_nonbinary_scale_map_and_exact_monotonic_difference(self):
        scale_map = self.result["scale_maps"]
        for background_id, records in scale_map.items():
            self.assertEqual([r["scale"] for r in records], list(audit.SCALES))
            for record in records:
                self.assertTrue(record["homogeneous_state_invariant"], background_id)
                self.assertTrue(record["homogeneous_hessian_invariant"], background_id)
                params = record["parameter"]
                self.assertTrue(params["polynomial_coefficients_in_y_unchanged"])
                self.close(params["ratios"]["Cv"], 1)
                if params["potential_family"] == "Q4":
                    self.close(params["ratios"]["A"], 1)
                    self.assertTrue(params["potential_metadata"]["lambda_used_by_potential"])
                    self.assertTrue(params["potential_metadata"]["A_used_by_potential"])
                    self.assertIsNone(params["potential_metadata"]["unused_quartic_auxiliary"])
                else:
                    self.assertIsNone(params["lambda"])
                    self.assertIsNone(params["homogeneous_A_from_actual_inputs"])
                    self.assertIsNone(params["ratios"]["lambda"])
                    self.assertIsNone(params["ratios"]["A"])
                    self.assertFalse(params["potential_metadata"]["lambda_used_by_potential"])
                    self.assertFalse(params["potential_metadata"]["A_used_by_potential"])
                    self.assertEqual(params["potential_metadata"]["representation"], "polynomial_z2_z3_z4")
                    self.assertIsNotNone(params["potential_metadata"]["polynomial_coefficients_in_y"])
                    self.assertIsNotNone(params["potential_metadata"]["unused_quartic_auxiliary"])
            nonbinary = next(r for r in records if r["scale"] == "1.3")
            with mp.workdps(120):
                self.close(nonbinary["parameter"]["ratios"]["gradient_coefficient"], mp.mpf("1.3") ** 2)
            self.assertNotEqual(nonbinary["parameter"]["ratios"]["gradient_coefficient"], "1")
        for law in self.result["scale_laws"]:
            self.assertTrue(law["all_pairs_pass"], law["background_id"])
            for pair in law["pairs"]:
                self.assertLessEqual(float(pair["relative_error"]), 1e-45)
                self.assertTrue(pair["dS_dscale2_nonnegative"])
                self.assertTrue(pair["dchi_dscale2_nonpositive"])
            for derivative in law["derivatives"]:
                self.assertTrue(derivative["nonnegative_S_derivative"])
                self.assertTrue(derivative["nonpositive_chi_derivative_on_stable_domain"])
        # The finite-wave rows actually carry the scale discriminator; k=0 is blind.
        for background_id in scale_map:
            zero = [r for r in self.result["rows"] if r["background_id"] == background_id and r["k_MeV"] == "0.0"]
            finite = [r for r in self.result["rows"] if r["background_id"] == background_id and r["k_MeV"] == "100.0"]
            self.assertEqual(len(zero), 4)
            self.assertEqual(len(finite), 4)
            self.assertTrue(all(r["gradient_inversion"]["status"] == "REFUSED_K0_SCALE_BLIND" for r in zero))
            self.assertTrue(all(r["gradient_inversion"]["status"] == "RECONSTRUCTED_NONDEGENERATE" for r in finite))

    def test_physical_vacuum_mass_uses_actual_potential_curvature(self):
        """Absolute masses come from d^2 U(W/W0)/dW^2, not only a scale ratio."""
        expected_at_scale_one = {
            "Q4:n_over_n0=0.37": "1244.8092624976727734824332511008833910177374139417957883",
            "Q4:n_over_n0=1": "1244.8092624976727734824332511008833910177374139417957883",
            "Q4:n_over_n0=2.75": "1244.8092624976727734824332511008833910177374139417957883",
            "Q4:n_over_n0=7.25": "1244.8092624976727734824332511008833910177374139417957883",
            "W8:y_star=0.90": "115.0067672342335018061967397423044818275625526586611177",
            "W8:y_star=0.93": "321.3203094199959628879872887687566966375137510450908219",
        }
        with mp.workdps(130):
            base = audit.upstream.BulkModel()
            backgrounds = audit._make_backgrounds(base, 110)
            for background in backgrounds:
                background_id = background["background_id"]
                reference = background["model"]
                records = self.result["scale_maps"][background_id]
                for scale_text in ("1", "1.3"):
                    candidate = audit.ScaledResponseModel(scale_text, reference)
                    def potential_of_W(W):
                        return candidate.potential(W / candidate.W0)[0]
                    independent_curvature = mp.diff(potential_of_W, candidate.W0, 2)
                    independent_mass = mp.sqrt(independent_curvature)
                    record = next(item for item in records if item["scale"] == scale_text)
                    parameter = record["parameter"]
                    self.close(parameter["m_sigma_MeV"], independent_mass)
                    self.close(parameter["potential_metadata"]["vacuum_curvature_Uyy_MeV4"],
                               candidate.potential(mp.mpf(1))[2])
                    if scale_text == "1":
                        self.close(parameter["m_sigma_MeV"], expected_at_scale_one[background_id])
                    else:
                        self.close(parameter["m_sigma_MeV"], independent_mass)
                        self.close(parameter["m_sigma_MeV"],
                                   mp.mpf(expected_at_scale_one[background_id]) / mp.mpf("1.3"))

    def test_inverse_reconstruction_and_degenerate_refusal(self):
        rows = [r for r in self.result["rows"] if r["scale"] == "1" and r["k_MeV"] in ("100.0", "200.0")]
        for row in rows:
            coeff = row["coefficients"]
            reconstructed = audit.invert_gradient_coefficient(coeff, row["k_MeV"], row["chi_MeV2"])
            self.close(reconstructed, row["Z_MeV2"])
            self.assertGreater(float(row["gradient_inversion"]["subtraction_conditioning_ratio"]), 1e-45)
        row = next(r for r in self.result["rows"] if r["scale"] == "1" and r["k_MeV"] == "0.0")
        with self.assertRaises(audit.SpatialIdentifiabilityError):
            audit.invert_gradient_coefficient(row["coefficients"], 0, row["chi_MeV2"])
        # B=0 is an independent degeneracy, even at positive k.
        blind_coeff = {"a": "1", "b": "1", "d0": "2", "g": "1", "h": "-2", "t0": "1"}
        blind_k = mp.sqrt(1)  # -g*h/b-t0 = 1
        response = audit.response_from_hessian(blind_coeff, 4, blind_k)
        self.assertEqual(response["B"], 0)
        with self.assertRaises(audit.SpatialIdentifiabilityError):
            audit.invert_gradient_coefficient(blind_coeff, blind_k, response["chi"])
        with self.assertRaises(ValueError):
            audit.upstream.inverse_potential_jet("1", audit.CALIBRATION_K_MEV)

    def test_near_k0_subtraction_cancellation_refuses(self):
        """A tiny chi perturbation cannot resolve the k^2 scale signal."""
        row = next(
            r for r in self.result["rows"]
            if r["family"] == "Q4" and r["density_ratio"] == "1"
            and r["scale"] == "1" and r["k_MeV"] == "0.0"
        )
        with mp.workdps(130):
            perturbed_chi = mp.mpf(row["chi_MeV2"]) * (mp.mpf("1") - mp.mpf("1e-60"))
            with self.assertRaisesRegex(audit.SpatialIdentifiabilityError, "C subtraction.*ill-conditioned"):
                audit.invert_gradient_coefficient(row["coefficients"], "1e-40", perturbed_chi)

    def test_blind_wave_classification_and_high_k_scope(self):
        by_id = {b["background_id"]: b for b in self.result["backgrounds"]}
        self.assertEqual(by_id["Q4:n_over_n0=2.75"]["blind_wave"]["status"], "POSITIVE_FINITE_BLIND_WAVE")
        self.assertEqual(by_id["Q4:n_over_n0=7.25"]["blind_wave"]["status"], "POSITIVE_FINITE_BLIND_WAVE")
        self.assertGreater(float(by_id["Q4:n_over_n0=7.25"]["blind_wave"]["k_blind"]), 1000)
        self.assertEqual(by_id["Q4:n_over_n0=0.37"]["blind_wave"]["status"], "NO_POSITIVE_BLIND_WAVE")
        self.assertEqual(by_id["W8:y_star=0.90"]["blind_wave"]["status"], "NO_POSITIVE_BLIND_WAVE")
        for background in self.result["backgrounds"]:
            if background["blind_wave"]["k_blind"] is not None:
                self.close(background["blind_wave"]["B_at_k_blind"], 0)
                self.assertIn(background["blind_wave"]["response_status"], {
                    "STABLE_STATIC_TF_RESPONSE", "UNSTABLE_SCALAR_CURVATURE",
                    "UNSTABLE_DENSITY_RESPONSE", "SINGULAR_DENSITY_RESPONSE",
                })
        self.assertTrue(self.result["scope"]["high_k_blind_points_are_formal_local_TF"])

    def test_unstable_and_singular_statuses_are_explicit(self):
        positive = {"a": "1", "b": "0", "d0": "-1", "g": "1", "h": "0", "t0": "1"}
        unstable = audit.response_from_hessian(positive, "4", "0")
        self.assertEqual(unstable["status"], "UNSTABLE_SCALAR_CURVATURE")
        self.assertFalse(unstable["stable"])
        singular = audit.response_from_hessian(
            {"a": "1", "b": "0", "d0": "0", "g": "1", "h": "0", "t0": "1"}, "4", "0"
        )
        self.assertEqual(singular["status"], "SINGULAR_SCALAR_CURVATURE")
        self.assertFalse(singular["stable"])
        with self.assertRaises(audit.SpatialIdentifiabilityError):
            audit.response_from_hessian(positive, "0", "100")
        with self.assertRaises(audit.SpatialIdentifiabilityError):
            audit.response_from_hessian(positive, "4", "-1")
        with self.assertRaises(audit.SpatialIdentifiabilityError):
            audit.response_from_hessian({**positive, "a": "NaN"}, "4", "100")

    def test_baseline_guard_is_reused_and_fails_closed_on_drift(self):
        with patch.dict(audit.upstream.INPUTS, {"W0": "860"}):
            with self.assertRaises(ArithmeticError):
                audit.build_result()
            self.assertFalse(audit.validate_result(self.result))

    def test_precision_control_and_explicit_decimal_parse(self):
        precision = self.result["precision_controls"]
        self.assertEqual(precision["dps"], [80, 110])
        self.assertEqual(precision["pass"], True)
        self.assertLessEqual(float(precision["max_relative_error"]), 1e-45)
        self.assertLessEqual(float(precision["parsed_decimal_relative_error"]), 1e-45)
        parsed = audit.parse_high_precision("0.123456789012345678901234567890123456789", 110)
        self.assertGreater(parsed._mpf_[3], 100)

    def test_cli_is_strict_json_and_no_write_result_cache(self):
        command = [sys.executable, str(audit.SOURCE_PATH)]
        completed = subprocess.run(command, check=True, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(HERE)})
        result = json.loads(completed.stdout)
        self.assertEqual(result["coverage"]["row_count"], 72)
        self.assertEqual(completed.stderr, "")
        self.assertFalse((HERE / "nvg_spatial_identifiability_audit_results.json").exists())


if __name__ == "__main__":
    unittest.main()
