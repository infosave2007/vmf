"""Focused live controls for the foundation identifiability audit."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_foundation_identifiability_audit as audit  # noqa: E402


class FoundationIdentifiabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit.build_result()

    @staticmethod
    def close(a, b, tolerance="1e-45"):
        with mp.workdps(110):
            a, b = mp.mpf(str(a)), mp.mpf(str(b))
            if not mp.isfinite(a) or not mp.isfinite(b):
                raise AssertionError("nonfinite comparison")
            if abs(a - b) / max(abs(a), abs(b), mp.mpf(1)) > mp.mpf(tolerance):
                raise AssertionError(f"{a} and {b} differ")

    def test_status_scope_and_live_digests(self):
        result = self.result
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["status"], "LIVE_FOUNDATION_IDENTIFIABILITY_AUDIT_CALIBRATION_NULL_NOT_EMPIRICAL")
        self.assertEqual(result["evidence_weight"], 0.0)
        self.assertEqual(result["source_sha256"], hashlib.sha256(audit.SOURCE_PATH.read_bytes()).hexdigest())
        self.assertEqual(result["upstream_source_sha256"], hashlib.sha256(audit.UPSTREAM_PATH.read_bytes()).hexdigest())
        self.assertEqual(result["passport_source_sha256"], hashlib.sha256(audit.PASSPORT_PATH.read_bytes()).hexdigest())
        self.assertEqual(result["contract_sha256"], hashlib.sha256(audit.CONTRACT_PATH.read_bytes()).hexdigest())
        self.assertTrue(result["scope"]["q_wave_number_distinct_from_q_phi"])
        self.assertTrue(result["scope"]["new_field_interaction_adopted"] is False)
        self.assertEqual(set(result["original"]["state"]), set(audit.STATE_FIELDS))

    def test_original_saturation_is_live_no_binding(self):
        original = self.result["original"]
        self.assertEqual(original["classification"], "ORIGINAL_NOT_SATURATED_POSITIVE_BINDING")
        self.assertGreater(float(original["binding_MeV"]), 0.0)
        self.assertGreater(float(original["state"]["pressure_MeV_fm3"]), 0.0)
        self.assertFalse(original["binding_target_match"])
        self.assertFalse(original["pressure_target_match"])
        self.assertTrue(original["global_no_binding_sufficient_condition"])
        self.assertGreater(float(original["Cv_over_Cs"]), 1.0)
        self.close(original["independent_K_MeV"], original["state"]["K_MeV"])

    def test_vector_ceiling_is_necessary_bound_and_original_exceeds(self):
        ceiling = self.result["vector_ceiling"]
        self.assertTrue(ceiling["original_exceeds_ceiling"])
        self.assertGreater(float(ceiling["original_gomega_dimensionless"]), float(ceiling["gomega_max"]))
        self.assertGreater(float(ceiling["original_mu_min_MeV"]), float(ceiling["mu_target_MeV"]))
        self.assertLess(float(ceiling["ceiling_state_B_y_relative"]), 1e-45)
        self.assertLess(float(ceiling["ceiling_state_mu_target_relative"]), 1e-45)
        self.close(ceiling["ceiling_state_K_MeV"], 3 * __import__("mpmath").mpf(ceiling["ceiling_state_mu_MeV"]))

    def test_quartic_calibration_k_is_independent_negative_control(self):
        calibration = self.result["two_target_quartic_calibration"]
        self.assertFalse(calibration["K_was_fitted"])
        self.assertTrue(calibration["calibration_is_not_held_out_evidence"])
        self.assertTrue(calibration["K_negative_control"])
        self.close(calibration["state"]["binding_MeV"], -16)
        self.close(calibration["state"]["pressure_MeV_fm3"], 0)
        self.close(calibration["independent_K_MeV"], calibration["K_from_state_MeV"])
        self.assertGreater(float(calibration["independent_K_MeV"]), 350.0)

    def test_independent_quadratures_are_for_new_live_states(self):
        rows = self.result["independent_quadrature"]
        self.assertEqual([row["density_ratio"] for row in rows], ["0.37", "2.75"])
        for row in rows:
            self.assertTrue(row["pass"])
            for error in row["errors"].values():
                self.assertLessEqual(float(error), 1e-45)
            self.assertTrue(all(key in row["state"] for key in audit.STATE_FIELDS))

    def test_scales_use_declared_map_and_invariant_homogeneous_rows(self):
        block = self.result["scale_identifiability"]
        self.assertEqual(block["density_ratios"], list(audit.OFF_GRID_DENSITY_RATIOS))
        self.assertEqual(block["fixed_y_values"], list(audit.FIXED_Y_VALUES))
        scales = {row["scale"]: row for row in block["scales"]}
        self.assertEqual(set(scales), {"0.5", "1.0", "1.3", "2.0"})
        for scale, row in scales.items():
            self.assertTrue(row["homogeneous_eos_invariant_at_fixed_y"])
            self.assertTrue(row["all_fixed_y_rows_pass"])
            self.assertTrue(row["all_equilibrium_rows_pass"])
            self.assertEqual(len(row["fixed_y_rows"]), len(audit.OFF_GRID_DENSITY_RATIOS) * len(audit.FIXED_Y_VALUES))
            self.assertEqual(len(row["equilibrium_rows"]), len(audit.OFF_GRID_DENSITY_RATIOS))
            for fixed in row["fixed_y_rows"]:
                self.assertTrue(fixed["pass"])
                self.assertTrue(all(float(value) <= 1e-45 for value in fixed["relative_differences"].values()))
            for equilibrium in row["equilibrium_rows"]:
                self.assertTrue(equilibrium["pass"])
                self.assertTrue(all(float(value) <= 1e-45 for value in equilibrium["relative_differences"].values()))

    def test_scale_parameter_ratios_and_full_dynamics_discriminator(self):
        rows = {row["scale"]: row for row in self.result["scale_identifiability"]["scales"]}
        for scale_text, row in rows.items():
            s = float(scale_text)
            ratios = row["parameter"]["ratios"]
            expected = row["parameter"]["expected_ratios"]
            for key, value in expected.items():
                self.close(ratios[key], value)
            self.close(ratios["A"], 1)
            self.close(ratios["Cv"], 1)
            if s == 1:
                self.assertFalse(row["full_dynamics_not_invariant"])
            else:
                self.assertTrue(row["full_dynamics_not_invariant"])
                self.assertGreater(float(ratios["gradient_coefficient"]), 0.0)
                self.assertNotAlmostEqual(float(ratios["gradient_coefficient"]), 1.0)
                self.assertNotAlmostEqual(float(ratios["m_sigma"]), 1.0)

    def test_scaled_constructor_recomputes_invariants_and_rejects_invalid_scale(self):
        for value in ("NaN", "+Inf", "-Inf", "0", "-1"):
            base = audit.upstream.BulkModel()
            with self.assertRaises(ValueError):
                audit.ScaledBulkModel(value, base)
        with mp.workdps(110):
            base = audit.upstream.BulkModel()
            candidate = audit.ScaledBulkModel("1.3", base)
            self.close(candidate.A, candidate.lam * candidate.W0**4)
            self.close(candidate.Cv, candidate.gomega**2 / candidate.momega**2)
            self.close(candidate.Cs, candidate.MN**2 / (2 * candidate.A))

            # Perturbing only one dimensional input changes the invariant A;
            # this guards against a reference-value pin masking a wrong map.
            lambda_only = audit.ScaledBulkModel("1.3", base)
            lambda_only.lam *= mp.mpf("1.01")
            lambda_only.A = lambda_only.lam * lambda_only.W0**4
            lambda_only.Cs = lambda_only.MN**2 / (2 * lambda_only.A)
            w0_only = audit.ScaledBulkModel("1.3", base)
            w0_only.W0 *= mp.mpf("1.01")
            w0_only.A = w0_only.lam * w0_only.W0**4
            w0_only.Cs = w0_only.MN**2 / (2 * w0_only.A)
            self.assertGreater(abs(lambda_only.A - candidate.A), mp.mpf("1e-20"))
            self.assertGreater(abs(w0_only.A - candidate.A), mp.mpf("1e-20"))

    def test_upstream_input_drift_fails_closed_before_nominal_success(self):
        drift_values = {
            "W0": "860",
            "lam": "1.06",
            "MN": "940",
            "momega": "783",
            "gomega": "8.0",
            "hbarc": "197.3",
            "n0_fm3": "0.161",
            "d": 2,
        }
        for key, value in drift_values.items():
            with self.subTest(key=key), patch.dict(audit.upstream.INPUTS, {key: value}):
                with self.assertRaises(ArithmeticError):
                    audit.build_result()
                self.assertFalse(audit.validate_result(self.result))

    def test_passport_numeric_drift_fails_closed_before_nominal_success(self):
        original_passport = audit.passport_source.passport

        def drifted_passport():
            payload = original_passport()
            for record in payload["records"]:
                if record["input_id"] == "g_omega":
                    record["value"] = 8.0
            return payload

        with patch.object(audit.passport_source, "passport", side_effect=drifted_passport):
            with self.assertRaises(ArithmeticError):
                audit.build_result()
            self.assertFalse(audit.validate_result(self.result))

    def test_passport_unit_drift_fails_closed(self):
        original_passport = audit.passport_source.passport

        def drifted_passport():
            payload = original_passport()
            for record in payload["records"]:
                if record["input_id"] == "g_omega":
                    record["unit"] = "MeV"
            return payload

        with patch.object(audit.passport_source, "passport", side_effect=drifted_passport):
            with self.assertRaises(ArithmeticError):
                audit.build_result()
            self.assertFalse(audit.validate_result(self.result))

    def test_decimal_difference_is_rejected_at_declared_precision(self):
        with self.assertRaises(AssertionError):
            self.close("1", "1.000000000000000000000000000001", tolerance="1e-45")

    def test_finite_wave_number_proof_is_nonzero_and_charge_is_distinct(self):
        rows = {row["scale"]: row for row in self.result["scale_identifiability"]["scales"]}
        for scale_text, row in rows.items():
            finite_q = row["finite_q"]
            self.assertEqual(finite_q["wave_number_symbol"], "k (MeV), not q_phi")
            self.assertEqual(finite_q["charge_symbol"], "q_phi (dimensionless Higgs charge)")
            self.assertTrue(finite_q["finite_wave_number_proof_reached"])
            self.assertTrue(finite_q["zero_wave_number_equal"])
            self.assertEqual(len(finite_q["rows"]), len(audit.WAVE_NUMBERS_MEV))
            self.assertTrue(all(float(item["kernel_identity_relative_error"]) <= 1e-45 for item in finite_q["rows"]))
            if float(scale_text) == 1:
                self.assertFalse(finite_q["finite_wave_number_nonzero_for_s_not_1"])
                self.assertTrue(all(float(item["absolute_difference_MeVminus2"]) == 0 for item in finite_q["rows"]))
            else:
                self.assertTrue(finite_q["finite_wave_number_nonzero_for_s_not_1"])
                self.assertTrue(all(item["nonzero_finite_q_difference"] for item in finite_q["rows"][1:]))
                self.assertGreater(float(finite_q["rows"][2]["absolute_difference_MeVminus2"]), 0.0)

    def test_two_precision_replay(self):
        controls = self.result["precision_controls"]
        self.assertGreaterEqual(self.result["inputs_and_scope"]["primary_working_precision_digits"], 50)
        self.assertGreaterEqual(self.result["inputs_and_scope"]["independent_control_precision_digits"], 80)
        self.assertEqual(len(controls), len(set(item["check_id"] for item in controls)))
        self.assertTrue(all(item["pass"] for item in controls))
        self.assertLessEqual(max(float(item["max_relative_difference"]) for item in controls), 1e-55)

    def test_passport_is_reused_and_strictly_validated(self):
        self.assertTrue(self.result["passport_validation"]["pass"])
        self.assertEqual(self.result["passport_validation"]["record_count"], 9)
        passport = self.result["passport"]
        self.assertEqual(len(passport["records"]), 7)
        self.assertEqual(len(passport["derived_records"]), 2)
        self.assertEqual(passport["renormalization_block"]["record_count"], 37)

    def test_mutations_missing_nan_and_success_flags_fail_closed(self):
        mutations = []
        missing = copy.deepcopy(self.result)
        missing["scale_identifiability"]["scales"].pop()
        mutations.append(missing)
        missing_q = copy.deepcopy(self.result)
        missing_q["scale_identifiability"]["scales"][0]["finite_q"]["rows"].pop()
        mutations.append(missing_q)
        nan_state = copy.deepcopy(self.result)
        nan_state["original"]["state"]["mu_MeV"] = "NaN"
        mutations.append(nan_state)
        forged = copy.deepcopy(self.result)
        forged["scale_identifiability"]["scales"][0]["full_dynamics_not_invariant"] = False
        mutations.append(forged)
        wrong_weight = copy.deepcopy(self.result)
        wrong_weight["evidence_weight"] = 1.0
        mutations.append(wrong_weight)
        for candidate in mutations:
            self.assertFalse(audit.validate_result(candidate))

    def test_roundtrip_json_and_source_contract_mutations_fail_closed(self):
        roundtripped = json.loads(json.dumps(self.result, allow_nan=False))
        self.assertTrue(audit.validate_result(roundtripped))
        altered = copy.deepcopy(self.result)
        altered["scale_identifiability"]["scales"][0]["parameter"]["ratios"]["gradient_coefficient"] = "1"
        self.assertFalse(audit.validate_result(altered))
        altered = copy.deepcopy(self.result)
        altered["contract_sha256"] = "0" * 64
        self.assertFalse(audit.validate_result(altered))

    def test_fresh_cli_is_strict_json_and_does_not_write_old_artifacts(self):
        paths = (
            audit.SOURCE_PATH,
            audit.CONTRACT_PATH,
            audit.UPSTREAM_PATH,
            HERE / "source_complete_scaling_saturation_results.json",
            HERE / "source_complete_solution_audit_results.json",
        )
        before = {path: path.read_bytes() if path.exists() else None for path in paths}
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str(HERE)
        completed = subprocess.run(
            [sys.executable, "-B", str(audit.SOURCE_PATH)],
            cwd=HERE.parent,
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(payload["evidence_weight"], 0.0)
        self.assertNotIn("output_path", payload)
        for path, old in before.items():
            self.assertEqual(path.read_bytes() if path.exists() else None, old)


if __name__ == "__main__":
    unittest.main()
