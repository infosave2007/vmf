"""Focused live tests for the sealed density-universality calculation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_density_universality_audit as audit


class DensityUniversalityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One fresh calculation serves the focused mutation tests.  The CLI
        # test below is the separate-process freshness check.
        cls.result = audit.build_result()

    def test_five_cases_and_full_density_coverage(self):
        self.assertEqual(tuple(self.result["cases"]), audit.CASE_ORDER)
        for case_id in audit.CASE_ORDER:
            case = self.result["cases"][case_id]
            self.assertEqual(len(case["rows"]), len(audit.DENSITY_RATIOS))
            self.assertEqual([row["density_ratio"] for row in case["rows"]], list(audit.DENSITY_RATIOS))
            self.assertTrue(case["final_asymptotic_check"]["pass"])
            for row in case["rows"]:
                self.assertEqual(set(row["fd_checks"]), set(audit.FD_STEPS))

    def test_units_and_precision_declarations_are_sealed(self):
        units = self.result["units_and_inputs"]
        self.assertEqual(units["state_variables"], "y=W/W0 dimensionless; n in natural MeV^3; U and epsilon in MeV^4")
        self.assertEqual(units["comparison_variables"], "nbar=n/W0^3 and ebar=epsilon/W0^4")
        self.assertEqual(units["density_ratios"], list(audit.DENSITY_RATIOS))
        self.assertEqual(units["working_precision_digits"], 70)
        self.assertEqual(units["independent_control_precision_digits"], 110)

    def test_tail_orders_and_formula_limits(self):
        self.assertEqual([item["p"] for item in self.result["formula_controls"]], [2, 4, 6, 8, 16])
        formula = {item["p"]: item for item in self.result["formula_controls"]}
        self.assertEqual(formula[2]["cold_fermi_vector_w"], "0.333333333333333333333333333333333333333333333333333333333333")
        self.assertEqual(formula[2]["naive_power_tail_ratio"], "0.0")
        self.assertTrue(formula[2]["p2_naive_ratio_rejected"])
        self.assertTrue(formula[4]["fermion_retained_in_leading_balance"])
        self.assertEqual(self.result["cases"]["original_quartic"]["tail_degree_in_y"], 4)
        self.assertEqual(self.result["cases"]["w8_90"]["tail_degree_in_y"], 8)
        self.assertEqual(self.result["cases"]["u16_90"]["tail_degree_in_y"], 16)
        self.assertEqual(self.result["cases"]["w8_90"]["limit"]["expected_w"], "0.6")
        self.assertEqual(self.result["cases"]["u16_90"]["limit"]["expected_w"], "0.777777777777777777777777777777777777777777777777777777777778")
        self.assertEqual(self.result["cases"]["u16_90"]["limit"]["expected_potential_fraction"], "0.111111111111111111111111111111111111111111111111111111111111")
        self.assertEqual(self.result["cases"]["u16_90"]["limit"]["expected_vector_fraction"], "0.888888888888888888888888888888888888888888888888888888888889")
        self.assertTrue(self.result["cases"]["u16_90"]["limit"]["fermion_subleading"])

    def test_analytic_fourth_order_jet_zeros_and_positive_deformation(self):
        import mpmath as mp

        with mp.workdps(80):
            for target in ("0.90", "0.93"):
                for y in (mp.mpf(1), mp.mpf(target)):
                    values = audit._tail_factor_derivatives(y, mp.mpf(target))
                    self.assertEqual(values[0], 0)
                    self.assertEqual(values[1], 0)
                    self.assertEqual(values[2], 0)
                    self.assertEqual(values[3], 0)
        for item in self.result["deformation_certificate"].values():
            self.assertTrue(item["eta_positive"])
            self.assertTrue(item["delta_U_nonnegative_by_even_power_proof"])
            self.assertTrue(item["exact_zero_through_order_3"])
            self.assertTrue(all(mp.mpf(value) > 0 for value in item["positive_sample_delta_U_MeV4"]))

    def test_calibration_null_comparison_preserves_stationarity_and_curvature(self):
        for target, item in self.result["calibration_point_comparisons"].items():
            self.assertTrue(item["jets_unchanged_through_order_3"])
            self.assertTrue(item["positive_deformation_eta"])
            self.assertTrue(all(value == "0.0" for value in item["w8_u16_relative_differences"].values()))
            for case_id, entry in item["entries"].items():
                self.assertLessEqual(float(entry["stationarity_relative"]), 1e-25)
                self.assertAlmostEqual(float(entry["state"]["pressure_MeV4"]), 0.0, places=10)
                self.assertAlmostEqual(float(entry["state"]["binding_MeV"]), -16.0, places=10)
                self.assertAlmostEqual(float(entry["state"]["K_MeV"]), 240.0, places=8)

    def test_full_fermi_is_not_kinetic_free_limit(self):
        comparison = self.result["p4_full_fermi_vs_kinetic_free"]
        self.assertGreater(float(comparison["relative_z_difference"]), 1e-3)
        self.assertNotEqual(comparison["full_fermi_z"], comparison["kinetic_free_z"])

    def test_first_local_discriminator_is_off_calibration_and_positive(self):
        for item in self.result["local_fourth_density_discriminator"].values():
            self.assertTrue(item["positive_when_nonzero_slope"])
            self.assertGreater(float(item["delta_Z"]), 0.0)
            self.assertGreater(float(item["symmetric_fourth_coefficient"]), 0.0)

    def test_stationarity_identity_fd_and_fractions_are_live(self):
        for case in self.result["cases"].values():
            for row in case["rows"]:
                self.assertLessEqual(float(row["checks"]["stationarity_relative"]), 1e-25)
                self.assertLessEqual(float(row["checks"]["legendre_relative"]), 1e-25)
                self.assertLessEqual(float(row["checks"]["trace_identity_relative"]), 1e-25)
                fractions = row["derived"]["fractions"]
                self.assertAlmostEqual(sum(float(fractions[key]) for key in ("fermi", "potential", "vector")), 1.0, places=20)
                for fd in row["fd_checks"].values():
                    self.assertTrue(fd["pass"])
                    self.assertLessEqual(float(fd["mu_relative_error"]), 1e-7)
                    self.assertLessEqual(float(fd["field_relative_error"]), 1e-7)

    def test_mutations_fail_closed(self):
        missing = copy.deepcopy(self.result)
        missing["cases"].pop("w8_90")
        self.assertFalse(audit.validate_result(missing))

        missing_grid = copy.deepcopy(self.result)
        missing_grid["cases"]["w8_90"]["rows"].pop()
        self.assertFalse(audit.validate_result(missing_grid))

        wrong_tail = copy.deepcopy(self.result)
        wrong_tail["cases"]["u16_90"]["tail_degree_in_y"] = 8
        self.assertFalse(audit.validate_result(wrong_tail))

        corrupted_residual = copy.deepcopy(self.result)
        corrupted_residual["cases"]["w8_90"]["rows"][0]["raw_evidence"]["residual"] = "1"
        self.assertFalse(audit.validate_result(corrupted_residual))

        corrupted_norm = copy.deepcopy(self.result)
        corrupted_norm["cases"]["w8_90"]["rows"][0]["raw_evidence"]["stationarity_scale"] = "1"
        self.assertFalse(audit.validate_result(corrupted_norm))

        corrupted_jet = copy.deepcopy(self.result)
        corrupted_jet["deformation_certificate"]["0.90"]["calibration_delta_derivatives_0_to_3"][3] = "1"
        self.assertFalse(audit.validate_result(corrupted_jet))

        nonfinite = copy.deepcopy(self.result)
        nonfinite["cases"]["w8_90"]["rows"][0]["state"]["cs2"] = "NaN"
        self.assertFalse(audit.validate_result(nonfinite))

        wrong_weight = copy.deepcopy(self.result)
        wrong_weight["evidence_weight"] = 1.0
        self.assertFalse(audit.validate_result(wrong_weight))

        forged_success = copy.deepcopy(self.result)
        forged_success["cases"]["w8_90"]["rows"][0]["checks"]["positive_C"] = False
        self.assertFalse(audit.validate_result(forged_success))

        corrupted_precision = copy.deepcopy(self.result)
        corrupted_precision["precision_controls"][0]["max_relative_difference"] = "1"
        self.assertFalse(audit.validate_result(corrupted_precision))

        missing_fd = copy.deepcopy(self.result)
        missing_fd["cases"]["w8_90"]["rows"][0]["fd_checks"].pop("1e-5")
        self.assertFalse(audit.validate_result(missing_fd))

        corrupted_calibration = copy.deepcopy(self.result)
        corrupted_calibration["calibration_point_comparisons"]["0.90"]["w8_u16_relative_differences"]["mu"] = "1"
        self.assertFalse(audit.validate_result(corrupted_calibration))

        corrupted_calibration_state = copy.deepcopy(self.result)
        corrupted_calibration_state["calibration_point_comparisons"]["0.93"]["entries"]["u16_93"]["state"]["K_MeV"] = "0"
        self.assertFalse(audit.validate_result(corrupted_calibration_state))

        corrupted_units = copy.deepcopy(self.result)
        corrupted_units["units_and_inputs"]["working_precision_digits"] = 69
        self.assertFalse(audit.validate_result(corrupted_units))

        corrupted_root = copy.deepcopy(self.result)
        corrupted_root["cases"]["w8_90"]["rows"][0]["root"]["bracket_low"] = "NaN"
        self.assertFalse(audit.validate_result(corrupted_root))

    def test_parent_gate_five_false_accepts_are_rejected(self):
        corrupted_cs2 = copy.deepcopy(self.result)
        corrupted_cs2["cases"]["u16_90"]["rows"][-1]["state"]["cs2"] = "999"
        self.assertFalse(audit.validate_result(corrupted_cs2))

        corrupted_discriminator = copy.deepcopy(self.result)
        corrupted_discriminator["local_fourth_density_discriminator"]["0.90"]["delta_Z"] = "999"
        self.assertFalse(audit.validate_result(corrupted_discriminator))

        duplicated_precision = copy.deepcopy(self.result)
        duplicated_precision["precision_controls"] = [
            copy.deepcopy(self.result["precision_controls"][0])
            for _ in self.result["precision_controls"]
        ]
        self.assertFalse(audit.validate_result(duplicated_precision))

        corrupted_fd = copy.deepcopy(self.result)
        corrupted_fd["cases"]["u16_90"]["rows"][0]["fd_checks"]["1e-5"]["mu_prime_fd"] = "999"
        self.assertFalse(audit.validate_result(corrupted_fd))

        corrupted_gravity = copy.deepcopy(self.result)
        corrupted_gravity["gravity_boundary"]["cases"][0]["normalized_flat_flrw_kretschmann_factor"] = "-1"
        self.assertFalse(audit.validate_result(corrupted_gravity))

    def test_numeric_evidence_ties_reject_negative_and_malformed_controls(self):
        negative_precision = copy.deepcopy(self.result)
        negative_precision["precision_controls"][0]["max_relative_difference"] = "-1"
        self.assertFalse(audit.validate_result(negative_precision))

        negative_fd_error = copy.deepcopy(self.result)
        negative_fd_error["cases"]["w8_90"]["rows"][0]["fd_checks"]["1e-5"]["mu_relative_error"] = "-1"
        self.assertFalse(audit.validate_result(negative_fd_error))

        malformed_case = copy.deepcopy(self.result)
        malformed_case["cases"]["w8_90"] = None
        self.assertFalse(audit.validate_result(malformed_case))

        malformed_fd = copy.deepcopy(self.result)
        malformed_fd["cases"]["w8_90"]["rows"][0]["fd_checks"]["1e-5"] = None
        self.assertFalse(audit.validate_result(malformed_fd))

    def test_roundtrip_and_cache_isolation(self):
        roundtripped = json.loads(json.dumps(self.result, allow_nan=False))
        self.assertTrue(audit.validate_result(roundtripped))
        changed = copy.deepcopy(self.result)
        changed["cases"]["u16_93"]["rows"][0]["derived"]["field_ratio"] = "2"
        self.assertFalse(audit.validate_result(changed))

    def test_cli_is_strict_json_fresh_and_no_write(self):
        script = HERE / "nvg_density_universality_audit.py"
        before = script.read_bytes()
        completed = subprocess.run(
            [sys.executable, "-B", str(script)],
            cwd=HERE.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "COMPUTED_FORMAL_TAIL_CLASSES_CALIBRATION_NULL_COUNTEREXAMPLE")
        self.assertEqual(payload["evidence_weight"], 0.0)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(script.read_bytes(), before)
        self.assertNotIn("output_path", payload)


if __name__ == "__main__":
    unittest.main()
