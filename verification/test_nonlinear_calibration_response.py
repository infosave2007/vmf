"""Focused tests for the live W8/U16 nonlinear inverse response producer."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_nonlinear_calibration_response as producer  # noqa: E402


class NonlinearCalibrationResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = producer.build_result()

    @staticmethod
    def _resign(payload):
        payload["integrity_sha256"] = hashlib.sha256(producer._canonical_payload(payload)).hexdigest()
        return payload

    def test_live_pass_schema_and_protocol_provenance(self):
        result = self.result
        self.assertEqual(result["status"], producer.STATUS_PASS)
        self.assertEqual(result["evidence_weight"], 0.0)
        self.assertEqual(result["protocol_version"], producer.PROTOCOL_VERSION)
        self.assertFalse(result["protocol_frozen_before_rows"])
        self.assertTrue(result["physical_inputs_fixed"])
        self.assertEqual(result["protocol_provenance"], producer.PROTOCOL_PROVENANCE)
        self.assertEqual(tuple(result["inputs"]["anchors"]), producer.ANCHORS)
        self.assertEqual(result["inputs"]["deformation_amplitudes"], [0, 1])
        self.assertEqual(result["inputs"]["response_u"], list(producer.RESPONSE_U))
        self.assertTrue(producer.validate_result(result))

    def test_public_protocol_matches_steps_and_survives_private_checkout_absence(self):
        protocol_text = producer.PROTOCOL_PATH.read_text(encoding="utf-8")
        for marker in producer.PROTOCOL_MARKERS:
            self.assertIn(marker, protocol_text)
        self.assertEqual(producer.PROTOCOL_PATH, HERE / "data" / "nonlinear_calibration_protocol_2026.md")
        self.assertNotIn("Lunacy", str(producer.PROTOCOL_PATH))

        # Rebuild from a temporary public-only verification tree.  No run
        # workspace is copied, so portability is tested rather than inferred
        # from the current checkout.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verification = root / "verification"
            data = verification / "data"
            data.mkdir(parents=True)
            for name in (
                "nvg_nonlinear_calibration_response.py",
                "nvg_density_universality_audit.py",
                "source_complete_scaling_saturation_audit.py",
            ):
                shutil.copy2(HERE / name, verification / name)
            shutil.copy2(producer.PROTOCOL_PATH, data / producer.PROTOCOL_PATH.name)
            output = root / "portable_result.json"
            completed = subprocess.run(
                [sys.executable, "-B", str(verification / "nvg_nonlinear_calibration_response.py"), "--output", str(output)],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.stderr, "")
            portable = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(portable["status"], producer.STATUS_PASS)
            self.assertEqual(portable["protocol_version"], producer.PROTOCOL_VERSION)
            self.assertFalse(portable["protocol_frozen_before_rows"])
            self.assertEqual(portable["inputs"]["inverse_fd_u_steps"], ["0.005", "0.002"])

    def test_public_protocol_missing_or_tampered_fails_closed(self):
        original = producer.PROTOCOL_PATH
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            missing = temporary_path / "missing.md"
            producer.PROTOCOL_PATH = missing
            try:
                self.assertFalse(producer.validate_result(self.result))
            finally:
                producer.PROTOCOL_PATH = original

            tampered = temporary_path / "tampered.md"
            tampered.write_bytes(original.read_bytes() + b"\n# tampered provenance\n")
            producer.PROTOCOL_PATH = tampered
            try:
                self.assertFalse(producer.validate_result(self.result))
            finally:
                producer.PROTOCOL_PATH = original

    def test_full_freeze_claim_cannot_be_resigned_to_true(self):
        forged = copy.deepcopy(self.result)
        forged["protocol_frozen_before_rows"] = True
        self._resign(forged)
        self.assertFalse(producer.validate_result(forged))

    def test_two_anchors_two_amplitudes_and_inverse_signs(self):
        for target in producer.ANCHORS:
            anchor = self.result["anchors"][target]
            self.assertEqual(set(anchor["models"]), set(producer.MODEL_ORDER))
            self.assertEqual(anchor["models"]["w8"]["amplitude"], 0)
            self.assertEqual(anchor["models"]["u16"]["amplitude"], 1)
            for model in anchor["models"].values():
                rows = model["responses"]
                self.assertEqual([float(row["u"]) for row in rows], [float(value) for value in producer.RESPONSE_U])
                self.assertTrue(any(float(row["u"]) < 0 for row in rows))
                self.assertTrue(any(float(row["u"]) > 0 for row in rows))
                for row in rows:
                    self.assertTrue(row["window_accept"])
                    self.assertEqual(row["branch_lineage"], "continued_from_anchor_positive_y_positive_C_positive_mu_prime")
                    self.assertLessEqual(float(row["mu_relative"]), 1e-60)
                    self.assertLessEqual(float(row["stationarity_relative"]), 1e-70)
                    self.assertGreater(float(row["C_y"]), 0)
                    self.assertGreater(float(row["mu_prime"]), 0)

    def test_inverse_calculus_identity_and_dimensionless_conversion(self):
        for target in producer.ANCHORS:
            comparison = self.result["anchors"][target]["comparison"]
            self.assertLessEqual(float(comparison["delta_chi3_identity_relative"]), 1e-45)
            self.assertLessEqual(float(comparison["dimensionless_identity_relative"]), 1e-45)
            self.assertTrue(comparison["nonzero_separation_above_floor"])
            self.assertGreater(float(comparison["max_abs_response_delta_x"]), 1e-30)
            lower = comparison["common_lower_jet_relative_differences"]
            self.assertLessEqual(float(lower["epsilon_1"]), 1e-45)
            self.assertLessEqual(float(lower["epsilon_2"]), 1e-30)
            self.assertLessEqual(float(lower["epsilon_3"]), 1e-30)
            self.assertGreater(float(comparison["delta_Z_MeV"]), 0)
            self.assertLess(float(comparison["delta_chi3"]), 0)

    def test_live_derivative_and_inverse_controls(self):
        for anchor in self.result["anchors"].values():
            for model in anchor["models"].values():
                derivatives = model["derivatives"]
                checks = derivatives["checks"]
                self.assertLessEqual(float(checks["epsilon1_equals_mu"]), 1e-45)
                self.assertLessEqual(float(checks["epsilon2_equals_mu_prime"]), 1e-30)
                self.assertTrue(checks["positive_curvature"])
                self.assertTrue(checks["positive_compressibility"])
                self.assertTrue(checks["finite"])
                controls = model["inverse_fd_controls"]
                self.assertEqual([row["u_step"] for row in controls], list(producer.FD_U_STEPS))
                self.assertTrue(all(row["pass"] for row in controls))
                self.assertLess(float(controls[-1]["chi3_relative_error"]), float(controls[0]["chi3_relative_error"]))

    def test_positive_calibration_null_deformation_and_toy(self):
        self.assertTrue(self.result["analytic_toy_control"]["pass"])
        for anchor in self.result["anchors"].values():
            deformation = anchor["models"]["u16"]["deformation"]
            self.assertEqual(deformation["amplitude"], 1)
            self.assertTrue(deformation["delta_U_nonnegative_by_even_power_proof"])
            self.assertTrue(deformation["exact_zero_through_order_3_at_vacuum_and_anchor"])
            self.assertGreater(float(deformation["delta_U_fourth_derivative_at_anchor_MeV4"]), 0)
            self.assertTrue(all(float(value) > 0 for value in deformation["positive_sample_delta_U_MeV4"]))

    def test_precision_controls_are_independent_and_pass(self):
        self.assertEqual(len(self.result["precision_controls"]), 4)
        self.assertTrue(all(item["primary_dps"] == 90 and item["control_dps"] == 120 for item in self.result["precision_controls"]))
        self.assertTrue(all(item["pass"] for item in self.result["precision_controls"]))
        self.assertTrue(all(float(item["max_relative_difference"]) <= 1e-45 for item in self.result["precision_controls"]))

    def test_fail_closed_mutations(self):
        missing = copy.deepcopy(self.result)
        missing["anchors"].pop("0.90")
        self.assertFalse(producer.validate_result(missing))

        corrupted = copy.deepcopy(self.result)
        corrupted["anchors"]["0.90"]["models"]["u16"]["responses"][0]["x"] = "999"
        self.assertFalse(producer.validate_result(corrupted))

        nonfinite = copy.deepcopy(self.result)
        nonfinite["anchors"]["0.93"]["models"]["w8"]["derivatives"]["chi"]["chi3"] = "NaN"
        self.assertFalse(producer.validate_result(nonfinite))

        wrong_branch = copy.deepcopy(self.result)
        wrong_branch["anchors"]["0.90"]["models"]["u16"]["responses"][0]["window_accept"] = False
        self.assertFalse(producer.validate_result(wrong_branch))

        # A forged checksum must still fail the live scientific rebuild.
        forged = copy.deepcopy(self.result)
        forged["anchors"]["0.93"]["comparison"]["delta_chi3"] = "123"
        self._resign(forged)
        self.assertFalse(producer.validate_result(forged))

    def test_resigned_reviewer_attacks_fail_semantic_validation(self):
        """Each named B1 attack is re-signed; checksum alone must not pass it."""
        attacks = []

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["models"]["u16"]["responses"][0]["u"] = "999"
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["inputs"]["response_u"] = ["999"]
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["models"]["u16"]["amplitude"] = 0
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["anchor_inputs"]["target_y"] = "0.91"
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["branch"]["density_window_ratio"] = ["0.01", "9"]
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["models"]["u16"]["derivatives"]["epsilon_derivatives"]["epsilon_4"] = "0"
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        original = float(mutated["anchors"]["0.90"]["models"]["u16"]["derivatives"]["epsilon_derivatives"]["epsilon_4"])
        mutated["anchors"]["0.90"]["models"]["u16"]["derivatives"]["epsilon_derivatives"]["epsilon_4"] = str(original * 100)
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["comparison"]["delta_epsilon_4"] = "0"
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["models"]["w8"]["inverse_fd_controls"] = []
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["models"]["w8"]["inverse_fd_controls"][0]["u_step"] = "0.01"
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["models"]["w8"]["responses"][0]["branch_lineage"] = "wrong_branch"
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["models"]["w8"]["responses"][0]["n_MeV3"] = "1"
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["models"]["w8"]["responses"][0]["mu_target_MeV"] = "1"
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["anchors"]["0.90"]["models"]["w8"]["responses"][0]["mu_residual_MeV"] = "1"
        attacks.append(mutated)

        mutated = copy.deepcopy(self.result)
        mutated["units_and_conventions"]["mu"] = "wrong units"
        attacks.append(mutated)

        for attack in attacks:
            self._resign(attack)
            self.assertFalse(producer.validate_result(attack))

    def test_json_roundtrip_and_strict_cli(self):
        roundtripped = json.loads(json.dumps(self.result, allow_nan=False))
        self.assertTrue(producer.validate_result(roundtripped))
        completed = subprocess.run(
            [sys.executable, "-B", str(HERE / "nvg_nonlinear_calibration_response.py")],
            cwd=HERE.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], producer.STATUS_PASS)
        self.assertEqual(completed.stderr, "")
        self.assertNotIn('"model": {', completed.stdout)
        self.assertNotIn("NaN", completed.stdout)


if __name__ == "__main__":
    unittest.main()
