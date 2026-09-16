"""Adversarial and precision checks for the causal kinetic spectrum audit."""
import contextlib
import copy
import io
import json
import unittest
from unittest import mock

import mpmath as mp

import nvg_kinetic_spectral_audit as audit


class KineticSpectralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = audit.audit("1", dps=80)
        cls.full = audit.compute_state(dps=80)

    def test_exact_symbolic_contract_and_full_source_vector(self):
        checks = audit.symbolic_checks()
        self.assertEqual(set(checks), audit.REQUIRED_SYMBOLIC_CHECKS)
        self.assertTrue(audit._symbolic_passed(checks))
        source = self.state["source_vector"]
        self.assertIs(source["full_source_vector_included"], True)
        self.assertIs(source["no_density_contact_term"], True)
        self.assertIs(source["passed"], True)
        self.assertLess(mp.mpf(source["determinant_relative_error"]), mp.mpf("1e-50"))
        self.assertLess(mp.mpf(source["density_response_relative_error"]), mp.mpf("1e-50"))
        self.assertTrue(source["missing_second_source_rejected"])

    def test_retarded_continuum_and_finite_frequency_absorption(self):
        absorption = self.state["continuum_absorption"]
        self.assertTrue(absorption["finite_frequency_absorption"])
        self.assertGreater(mp.mpf(absorption["retarded_S_over_NF"]), 0)
        self.assertLess(mp.mpf(absorption["advanced_S_over_NF"]), 0)
        self.assertTrue(absorption["wrong_sign_rejected"])
        self.assertTrue(self.state["angular_response"]["passed"])

    def test_independent_endpoint_quadratures_and_two_moments(self):
        moments = self.state["spectral_moments"]
        self.assertTrue(moments["integrals_converged"])
        self.assertTrue(moments["passed"])
        for key in ("m1_relative_error", "mminus1_relative_error",
                    "quadrature_m1_relative_difference",
                    "quadrature_mminus1_relative_difference"):
            self.assertLess(mp.mpf(moments[key]), audit.MOMENT_TOLERANCE)
        self.assertGreater(mp.mpf(moments["pole_contribution_m1"]), 0)
        self.assertGreater(mp.mpf(moments["pole_contribution_mminus1"]), 0)

    def test_moment_bridge_is_explicit_and_reaches_live_effective_response(self):
        bridge = self.state["moment_bridge"]
        self.assertIs(bridge["passed"], True)
        self.assertLess(mp.mpf(bridge["moment_to_collisionless_relative_error"]),
                        audit.MOMENT_TOLERANCE)
        self.assertLess(mp.mpf(bridge["moment_to_effective_response_relative_error"]),
                        audit.MOMENT_TOLERANCE)
        with mp.workdps(80):
            expected = (mp.mpf(self.state["coefficients"]["vF2"])
                        *mp.mpf(self.state["spectral_moments"]["m1_total"])
                        /mp.mpf(self.state["spectral_moments"]["mminus1_total"]))
            self.assertLess(abs(mp.mpf(bridge["vF2_m1_over_mminus1"])/expected-1),
                            mp.mpf("1e-40"))
        self.assertEqual(bridge["effective_response_acoustic_squared_speed"],
                         self.state["effective_response_bridge"]["source_acoustic_squared_speed"])

    def test_static_dynamic_limits_are_noncommuting_and_high_coefficient_matches(self):
        limits = self.state["limits"]
        self.assertTrue(limits["passed"])
        self.assertTrue(limits["noncommuting_static_dynamic_limits"])
        self.assertLess(mp.mpf(limits["static_relative_error"]), mp.mpf("1e-50"))
        self.assertLess(mp.mpf(limits["high_sigma_relative_error"]), audit.MOMENT_TOLERANCE)
        self.assertAlmostEqual(float(mp.mpf(limits["dynamic_limit_exact"])), 0.0)

    def test_live_effective_response_acoustic_and_inertia_bridge(self):
        bridge = self.state["effective_response_bridge"]
        self.assertEqual(bridge["source_producer"],
                         "verification/nvg_longitudinal_fluid_audit.py")
        self.assertIs(bridge["source_independent_checks_passed"], True)
        self.assertIs(bridge["mathematical_checks_passed"], True)
        self.assertTrue(bridge["source_comparison_checked"])
        self.assertTrue(bridge["passed"])
        self.assertLess(mp.mpf(bridge["inertial_relative_error"]), mp.mpf("1e-50"))

    def test_fixed_controls_include_no_pole_high_density_case_and_refinement(self):
        result = self.full
        self.assertIs(result["mathematical_checks_passed"], True)
        self.assertIs(result["numeric_passport_passed"], True)
        self.assertEqual(tuple(result["representative_density_ratios"]),
                         audit.REPRESENTATIVE_DENSITIES)
        self.assertEqual(len(result["states"]), 4)
        self.assertEqual(len(result["precision_refinement"]), 4)
        by_ratio = {row["inputs"]["n_over_n0"]: row for row in result["states"]}
        self.assertFalse(by_ratio["200.0"]["pole"]["exists"])
        self.assertTrue(by_ratio["100.0"]["pole"]["exists"])
        for row in result["precision_refinement"]:
            self.assertIs(row["passed"], True)
            self.assertLess(mp.mpf(row["max_relative_difference"]), mp.mpf("1e-50"))

    def test_mathematical_controls_keep_noninteracting_current_wrong_sign_and_omitted_pole(self):
        controls = self.full["controls"]
        self.assertIs(controls["passed"], True)
        self.assertTrue(controls["noninteracting"]["pole_absent"])
        self.assertTrue(controls["current_feedback"]["current_feedback_visible"])
        self.assertTrue(controls["wrong_retarded_sign"]["wrong_sign_rejected"])
        self.assertTrue(controls["omitted_pole"]["omitted_pole_rejected"])
        near = controls["near_threshold_log_gap"]
        self.assertEqual(len(near), 3)
        self.assertTrue(all(row["positive_gap_and_residue"] for row in near))

    def test_pole_residue_is_positive_and_gap_is_retained_near_threshold(self):
        with mp.workdps(110):
            pole = audit.pole_data(mp.mpf("3.01"), 1, dps=110)
            self.assertTrue(pole["exists"])
            self.assertGreater(pole["gap"], 0)
            self.assertGreater(pole["residue"], 0)
            self.assertLess(pole["log_gap"], -100)
            self.assertLess(abs(pole["dispersion_residual"]), mp.mpf("1e-60"))

    def test_tiny_pole_gap_and_residue_are_scale_aware_and_duplicate_safe(self):
        with mp.workdps(110):
            pole = audit._shown(audit.pole_data(mp.mpf("3.01"), 1, dps=110))
            self.assertTrue(audit._validate_pole_payload(pole, mp.mpf("3.01"), 1))
            for key in ("gap", "residue"):
                for replacement in ("0", "-1", audit.number(mp.mpf(pole[key])*2)):
                    mutated = copy.deepcopy(pole)
                    mutated[key] = replacement
                    with self.subTest(field=key, replacement=replacement):
                        self.assertFalse(
                            audit._validate_pole_payload(mutated, mp.mpf("3.01"), 1))

        duplicate = copy.deepcopy(self.state)
        duplicate["spectral_moments"]["pole"]["gap"] = "0"
        self.assertFalse(audit._numeric_state_passed(duplicate))
        duplicate = copy.deepcopy(self.state)
        duplicate["spectral_moments"]["pole"]["residue"] = "0"
        self.assertFalse(audit._numeric_state_passed(duplicate))

    def test_tiny_pole_moment_contributions_are_scale_aware(self):
        with mp.workdps(110):
            moments = audit._shown(
                audit.integrate_spectral_moments(mp.mpf("3.01"), 1, dps=110))
            pole = moments["pole"]
            self.assertTrue(pole["exists"])
            self.assertTrue(audit._validate_moments_payload(moments, pole,
                                                            mp.mpf("3.01"), mp.mpf("1")))
            for field in ("pole_contribution_m1", "pole_contribution_mminus1"):
                self.assertGreater(mp.mpf(moments[field]), 0)
                for replacement in ("0", "-1", audit.number(mp.mpf(moments[field])*2)):
                    mutated = copy.deepcopy(moments)
                    mutated[field] = replacement
                    with self.subTest(field=field, replacement=replacement):
                        self.assertFalse(audit._validate_moments_payload(
                            mutated, mutated["pole"], mp.mpf("3.01"), mp.mpf("1")))
            both_zero = copy.deepcopy(moments)
            both_zero["pole_contribution_m1"] = "0"
            both_zero["pole_contribution_mminus1"] = "0"
            self.assertFalse(audit._validate_moments_payload(
                both_zero, both_zero["pole"], mp.mpf("3.01"), mp.mpf("1")))

            absent = audit._shown(audit.integrate_spectral_moments(0, 0, dps=110))
            self.assertFalse(absent["pole"]["exists"])
            self.assertEqual(mp.mpf(absent["pole_contribution_m1"]), 0)
            self.assertEqual(mp.mpf(absent["pole_contribution_mminus1"]), 0)
            self.assertTrue(audit._validate_moments_payload(
                absent, absent["pole"], mp.mpf("0"), mp.mpf("0")))
            for field in ("pole_contribution_m1", "pole_contribution_mminus1"):
                mutated = copy.deepcopy(absent)
                mutated[field] = "1e-50"
                self.assertFalse(audit._validate_moments_payload(
                    mutated, mutated["pole"], mp.mpf("0"), mp.mpf("0")))

    def test_invalid_parameters_and_branch_values_fail_closed(self):
        for bad in (True, None, "nan", "inf", "-1"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                audit._landau_parameters(bad, 0)
        for bad in (True, -1, "nan", "inf"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                audit._landau_parameters(0, bad)
        with self.assertRaises(ValueError):
            audit.continuum_spectral_density("0.5", 0, 0, branch="principal")
        with self.assertRaises(ValueError):
            audit.integrate_spectral_moments(0, 0, dps=79)

    def test_failed_or_malformed_upstream_residuals_and_status_are_rejected(self):
        raw = audit.coefficients("1", dps=80)
        malformed = copy.deepcopy(raw)
        malformed["residuals"].pop(next(iter(malformed["residuals"])))
        with mock.patch.object(audit, "coefficients", return_value=malformed):
            with self.assertRaises(ArithmeticError):
                audit.audit("1", dps=80)
        failed = {"status": audit.collisionless.STATUS,
                  "mathematical_checks_passed": False}
        with mock.patch.object(audit.collisionless, "audit", return_value=failed):
            with self.assertRaises(ArithmeticError):
                audit.audit("1", dps=80)

    def test_mutated_derived_fields_and_nonconverged_measurements_cannot_pass(self):
        mutated = copy.deepcopy(self.state)
        mutated["spectral_moments"]["m1_total"] = "0"
        self.assertFalse(audit._numeric_state_passed(mutated))
        mutated = copy.deepcopy(self.state)
        mutated["spectral_moments"]["quadrature_m1_relative_difference"] = "1"
        self.assertFalse(audit._numeric_state_passed(mutated))
        mutated = copy.deepcopy(self.state)
        mutated["pole"]["residue"] = "0"
        self.assertFalse(audit._numeric_state_passed(mutated))
        mutated = copy.deepcopy(self.state)
        mutated["pole"]["delta_piece_required"] = False
        self.assertFalse(audit._numeric_state_passed(mutated))
        mutated = copy.deepcopy(self.full)
        mutated["states"][0]["mathematical_checks_passed"] = "yes"
        self.assertFalse(audit._numeric_passport_passed(mutated["states"],
                                                        mutated["precision_refinement"]))

    def test_parent_witnessed_serialized_false_positives_are_all_rejected(self):
        mutations = []

        mutated = copy.deepcopy(self.state)
        mutated["upstream"]["status"] = "FAILED"
        mutations.append(mutated)

        mutated = copy.deepcopy(self.state)
        mutated["source_vector"]["density_response_relative_error"] = "-1"
        mutations.append(mutated)

        mutated = copy.deepcopy(self.state)
        mutated["effective_response_bridge"]["source_independent_checks_passed"] = False
        mutations.append(mutated)

        mutated = copy.deepcopy(self.state)
        mutated["effective_response_bridge"]["source_acoustic_squared_speed"] = "999"
        mutations.append(mutated)

        mutated = copy.deepcopy(self.state)
        mutated["pole"]["residue"] = "999"
        mutations.append(mutated)

        mutated = copy.deepcopy(self.state)
        mutated["coefficients"]["NF"] = "nan"
        mutations.append(mutated)

        mutated = copy.deepcopy(self.state)
        mutated["spectral_moments"]["quadrature"]["exponential_endpoint"]["m1_continuum"] = "nan"
        mutations.append(mutated)

        mutated = copy.deepcopy(self.state)
        for key in ("m1_total", "m1_total_without_pole", "pole_contribution_m1"):
            mutated["spectral_moments"][key] = audit.number(
                mp.mpf(mutated["spectral_moments"][key])*2)
        mutations.append(mutated)

        for index, state in enumerate(mutations):
            with self.subTest(index=index):
                self.assertFalse(audit._numeric_state_passed(state))

    def test_serialized_closure_rejects_missing_leaves_negative_errors_and_bad_duplicates(self):
        mutated = copy.deepcopy(self.state)
        mutated["spectral_moments"].pop("m1_total")
        self.assertFalse(audit._numeric_state_passed(mutated))

        mutated = copy.deepcopy(self.state)
        mutated["angular_response"]["max_response_relative_error"] = "-1"
        self.assertFalse(audit._numeric_state_passed(mutated))

        mutated = copy.deepcopy(self.state)
        mutated["spectral_moments"]["pole"]["residue"] = "0"
        self.assertFalse(audit._numeric_state_passed(mutated))

        mutated = copy.deepcopy(self.state)
        mutated["moment_bridge"]["vF2_m1_over_mminus1"] = "999"
        self.assertFalse(audit._numeric_state_passed(mutated))

    def test_passport_requires_exact_refinements_and_recomputable_components(self):
        self.assertTrue(audit._numeric_passport_passed(
            self.full["states"], self.full["precision_refinement"]))
        self.assertFalse(audit._numeric_passport_passed(self.full["states"]))

        mutated = copy.deepcopy(self.full)
        mutated["precision_refinement"].pop()
        self.assertFalse(audit._numeric_passport_passed(
            mutated["states"], mutated["precision_refinement"]))

        mutated = copy.deepcopy(self.full)
        mutated["precision_refinement"][0]["max_relative_difference"] = "-1"
        self.assertFalse(audit._numeric_passport_passed(
            mutated["states"], mutated["precision_refinement"]))

        mutated = copy.deepcopy(self.full)
        mutated["precision_refinement"][0]["difference_components"]["F0"] = "1"
        self.assertFalse(audit._numeric_passport_passed(
            mutated["states"], mutated["precision_refinement"]))

    def test_cli_is_strict_json_and_has_no_artifact_mode(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main(["--n-ratio", "1", "--dps", "80"])
        self.assertEqual(code, 0)
        parsed = json.loads(output.getvalue(),
                            parse_constant=lambda value: self.fail(value))
        self.assertEqual(parsed["status"], audit.STATUS)
        self.assertIs(parsed["mathematical_checks_passed"], True)
        self.assertEqual(parsed["evidence_weight"], 0)
        with contextlib.redirect_stdout(output := io.StringIO()):
            code = audit.main(["--dps", "20"])
        self.assertEqual(code, 2)
        failed = json.loads(output.getvalue())
        self.assertIs(failed["mathematical_checks_passed"], False)


if __name__ == "__main__":
    unittest.main()
