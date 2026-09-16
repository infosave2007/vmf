"""Independent spatial closure, saddle-sign and interface controls."""

import contextlib
import io
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

import nvg_spatial_binding_audit as audit


class SpatialBindingTests(unittest.TestCase):
    def setUp(self):
        self.context = mp.workdps(80)
        self.context.__enter__()
        self.design = audit.BindingDesign(dps=80)

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def close(self, a, b, tol="1e-55"):
        self.assertLess(abs(a - b) / max(abs(a), abs(b), mp.mpf(1)), mp.mpf(tol))

    def test_symbolic_certificate_has_all_required_signs_and_identities(self):
        rows = audit.symbolic_checks()
        self.assertEqual(set(rows), audit.REQUIRED_SYMBOLIC)
        self.assertEqual(len(rows), len(audit.REQUIRED_SYMBOLIC))
        self.assertNotIn("sandwich_lower", rows)
        self.assertNotIn("sandwich_upper", rows)
        for row in rows.values():
            self.assertIs(row["passed"], True)
            self.assertEqual(row["residual"], "0")
        self.assertTrue(audit._symbolic_rows_passed(rows))

    def test_symbolic_action_and_force_mutations_are_rejected(self):
        scalar_mutation = audit.symbolic_checks(scalar_force_sign=-1)
        self.assertFalse(scalar_mutation["scalar_el_equation"]["passed"])
        self.assertFalse(audit._symbolic_rows_passed(scalar_mutation))

        vector_mutation = audit.symbolic_checks(vector_source_sign=-1)
        self.assertFalse(vector_mutation["vector_el_equation"]["passed"])
        self.assertFalse(audit._symbolic_rows_passed(vector_mutation))

        action_mutation = audit.symbolic_checks(vector_action_sign=1)
        self.assertFalse(action_mutation["vector_hessian_negative"]["passed"])
        self.assertFalse(audit._symbolic_rows_passed(action_mutation))

    def test_physical_mixed_hessian_chain_and_actual_mp_control(self):
        rows = audit.symbolic_checks()
        for key in (
            "pressure_mixed_closed_form",
            "mixed_hessian_w_a_physical",
            "mixed_hessian_a_w_physical",
            "mixed_hessian_closed_expression",
            "hessian_cross_symmetry",
        ):
            self.assertIs(rows[key]["passed"], True)
            self.assertEqual(rows[key]["residual"], "0")

        control = audit.mixed_hessian_controls(self.design)
        self.assertTrue(audit._mixed_hessian_evidence_passed(control))
        self.assertTrue(control["active_branch"])
        self.assertLess(float(control["fermion_chain_term"]), 0)
        self.assertGreater(float(control["omitted_term_residual"]), 1e-6)
        self.assertGreater(float(control["wrong_sign_term_residual"]), 1e-6)

        omitted = dict(control)
        omitted["closed_expression"] = control["omitted_fermion_expression"]
        self.assertFalse(audit._mixed_hessian_evidence_passed(omitted))

        wrong_sign = dict(control)
        wrong_sign["closed_expression"] = control["wrong_sign_fermion_expression"]
        self.assertFalse(audit._mixed_hessian_evidence_passed(wrong_sign))

        missing_chain = dict(control)
        missing_chain["fermion_chain_term"] = "0"
        self.assertFalse(audit._mixed_hessian_evidence_passed(missing_chain))

        # A high-precision corruption must not disappear through float
        # downcast, even when its reported residual is changed consistently.
        paired_corruption = dict(control)
        corrupted_wa = mp.mpf(control["actual_hessian_WA"]) * (
            1 + mp.mpf("1e-20")
        )
        paired_corruption["actual_hessian_WA"] = mp.nstr(corrupted_wa, 75)
        closed = mp.mpf(control["closed_expression"])
        scale = max(abs(corrupted_wa), abs(closed), mp.mpf(1))
        paired_corruption["relative_residual_WA"] = mp.nstr(
            abs(corrupted_wa - closed) / scale, 75
        )
        self.assertFalse(audit._mixed_hessian_evidence_passed(paired_corruption))

        missing_flag = dict(control)
        missing_flag.pop("closed_expression_matches_actual")
        self.assertFalse(audit._mixed_hessian_evidence_passed(missing_flag))

        bool_number = dict(control)
        bool_number["actual_hessian_WA"] = True
        self.assertFalse(audit._mixed_hessian_evidence_passed(bool_number))

        negative_residual = dict(control)
        negative_residual["relative_residual_WA"] = "-1e-30"
        self.assertFalse(audit._mixed_hessian_evidence_passed(negative_residual))

        nan_residual = dict(control)
        nan_residual["wrong_sign_term_residual"] = "nan"
        self.assertFalse(audit._mixed_hessian_evidence_passed(nan_residual))

        altered_negative_control = dict(control)
        altered_negative_control["omitted_term_residual"] = "0"
        self.assertFalse(
            audit._mixed_hessian_evidence_passed(altered_negative_control)
        )

        for dps in (80, 120):
            roundtrip = json.loads(json.dumps(
                audit.audit(dps=dps, include_numerics=False),
                allow_nan=False,
            ))
            self.assertTrue(roundtrip["mathematical_checks_passed"])
            self.assertTrue(
                audit._mixed_hessian_evidence_passed(
                    roundtrip["mixed_hessian_controls"]
                )
            )

    def test_analytic_inequalities_are_separate_from_symbolic_identities(self):
        certificates = audit.analytic_certificates(self.design)
        self.assertTrue(audit._analytic_certificates_passed(certificates))
        self.assertEqual(len(certificates), 8)
        for row in certificates.values():
            self.assertNotIn("residual", row)
            self.assertIsInstance(row["kind"], str)
            self.assertIsInstance(row["statement"], str)

    def test_tf_pressure_and_scalar_density_match_positive_quadratures(self):
        d = self.design
        for ratio in (".1", ".5", "2"):
            mass = mp.mpf("700")
            k = mass * mp.mpf(ratio)
            p = audit._fermi_pressure_mp(k, mass, d.base.d)
            ns = audit._fermi_scalar_density_mp(k, mass, d.base.d)
            p_ref = d.base.d / (6 * mp.pi**2) * mp.quad(
                lambda pp: pp**4 / mp.sqrt(pp**2 + mass**2), [0, k]
            )
            ns_ref = d.base.d / (2 * mp.pi**2) * mp.quad(
                lambda pp: pp**2 * mass / mp.sqrt(pp**2 + mass**2), [0, k]
            )
            self.close(p, p_ref, "1e-60")
            self.close(ns, ns_ref, "1e-60")

    def test_local_envelope_stationarity_and_weighted_derivative_bound(self):
        d = self.design
        checks = audit.derivative_bound(d)
        self.assertTrue(checks["first_term_dominates"])
        self.assertTrue(checks["eta_less_than_one"])
        eta = mp.mpf(checks["eta"])
        self.assertLess(eta, 1)
        self.assertGreater(eta, mp.mpf(".15"))
        for row in checks["samples"]:
            self.assertLess(abs(mp.mpf(row["residual"])), mp.mpf("1e-60"))
            self.assertLess(abs(mp.mpf(row["stationarity_residual"])), mp.mpf("1e-60"))
            self.assertLessEqual(mp.mpf(row["abs_slope"]), eta)
        state = audit.local_envelope_state(d, d.y)
        self.close(state["A_L"], d.gomega * d.n / state["M2"])
        self.assertGreater(state["n"], 0)
        self.assertGreater(state["a_F"], 0)
        self.assertLess(state["dA_dW"], 0)
        self.assertTrue(audit._envelope_evidence_passed(checks, audit.onset_regularity(d)))

    def test_onset_is_regular_but_fractional_and_above_onset_is_vacuum(self):
        d = self.design
        result = audit.onset_regularity(d)
        self.assertEqual(result["density_power"], "3/2")
        self.assertEqual(result["floor_power"], "5/2")
        self.assertEqual(result["A_power"], "3/2")
        rows = result["rows"]
        self.assertGreater(
            abs(mp.mpf(rows[0]["floor_over_x_5_2"]) - 1),
            abs(mp.mpf(rows[-1]["floor_over_x_5_2"]) - 1),
        )
        # The reused BindingDesign onset solver intentionally retains its
        # 80-digit-safe branch; at x=1e-32 its closed-form series joins a
        # double-ratio coefficient, so 1e-14 is the appropriate independent
        # regularity tolerance here.
        self.assertLess(abs(mp.mpf(rows[-1]["floor_over_x_5_2"]) - 1), mp.mpf("1e-14"))
        self.assertEqual(mp.mpf(result["A_L_above_onset"]), 0)
        self.assertIn("C2 but not C3", result["regularity_statement"])

    def test_saddle_and_opposite_sign_control(self):
        result = audit.saddle_controls()
        self.assertTrue(result["correct_is_maximum"])
        self.assertTrue(result["opposite_is_not_maximum"])
        self.assertLess(float(result["correct_vector_second_variation"]), 0)
        self.assertGreater(float(result["opposite_sign_second_variation"]), 0)

    def test_actual_endpoint_saddle_hessians_are_used_by_acceptance(self):
        result = audit.saddle_controls(self.design)
        self.assertTrue(audit._saddle_evidence_passed(result))
        self.assertTrue(result["actual_vector_hessian_negative"])
        self.assertTrue(result["actual_scalar_endpoint_criteria_passed"])
        self.assertEqual(len(result["actual_hessian_samples"]), 2)
        for row in result["actual_hessian_samples"]:
            self.assertLess(float(row["grand_hessian_AA"]), 0)
            self.assertGreater(float(row["reduced_scalar_schur"]), 0)
        mutated = dict(result)
        mutated["actual_hessian_samples"] = [dict(row) for row in result["actual_hessian_samples"]]
        mutated["actual_hessian_samples"][0]["grand_hessian_AA"] = "nan"
        self.assertFalse(audit._saddle_evidence_passed(mutated))

    def test_exact_tension_integral_and_certified_band(self):
        result = audit.tension_bounds(self.design)
        self.assertLess(abs(mp.mpf(result["sigma0_integral_residual"])), mp.mpf("1e-55"))
        lower = mp.mpf(result["lower_bound_MeV3"])
        upper = mp.mpf(result["upper_bound_MeV3"])
        self.assertGreater(lower, 0)
        self.assertLess(lower, upper)
        self.assertLess(
            abs(mp.mpf(result["sigma0_MeV_fm_minus2"])
                - mp.mpf(result["sigma0_MeV3_formula"]) / self.design.base.hbarc**2),
            mp.mpf("1e-30"),
        )
        self.assertTrue(result["lower_direction_certified"])
        self.assertTrue(result["upper_direction_certified"])
        self.assertTrue(result["inequality_certificates_are_analytic"])
        self.assertTrue(result["bound_is_for_profiles_in_Wstar_to_W0_band"])
        self.assertTrue(result["not_global_unrestricted_minimum"])

    def test_coupled_interface_refinement_reports_residuals_stress_and_truncation(self):
        result = audit.interface_refinement(self.design)
        self.assertTrue(result["available"])
        self.assertEqual(len(result["attempts"]), 3)
        self.assertTrue(result["converged_coupled_solution"])
        self.assertTrue(result["finite_domain_refinement_valid"])
        self.assertTrue(result["tension_refinement_passed"])
        self.assertFalse(result["infinite_wall_certified"])
        self.assertTrue(result["tension_band_is_diagnostic_only"])
        self.assertTrue(result["final_tension_in_declared_band"])
        self.assertTrue(result["final"]["finite_domain_accuracy_passed"])
        self.assertTrue(any(not row["finite_domain_accuracy_passed"]
                            for row in result["attempts"]))
        for row in result["attempts"]:
            self.assertTrue(row["finite_domain_approximation"])
            self.assertTrue(row["boundary_truncation_is_reported"])
            self.assertTrue(row["not_a_scalar_profile_claim"])
            self.assertFalse(row["vector_positivity_certified"])
            self.assertIn("no maximum-principle enclosure", row["vector_positivity_scope"])
            for key in ("independent_residual_passed", "stress_criterion_passed",
                        "boundary_derivative_passed", "band_excursion_passed",
                        "vector_undershoot_within_finite_domain_limit"):
                self.assertIsInstance(row[key], bool)
            self.assertLessEqual(float(row["max_EL_residual_dimensionless"]),
                                 float(row["independent_residual_limit"]) * 20)
        final = result["final"]
        self.assertTrue(final["successive_tension_difference_passed"])
        self.assertLessEqual(float(final["successive_tension_relative_difference"]),
                             float(result["tension_refinement_limit"]))

    def test_periodic_gauss_operator_and_refinement_are_independent_controls(self):
        result = audit.periodic_controls(self.design)
        self.assertTrue(result["available"])
        self.assertTrue(result["includes_W_below_Wstar"])
        self.assertTrue(result["positive_samples_do_not_prove_global_minimum"])
        self.assertFalse(result["negative_E_minus_muN_witness"])
        self.assertTrue(result["calculation_valid"])
        self.assertEqual(result["interpretation"], "no_negative_witness_in_finite_trials")
        self.assertEqual(result["unrestricted_extension_status"], "undecided_positive_samples_only")
        self.assertGreaterEqual(len(result["trials"]), 3)
        for row in result["trials"]:
            self.assertTrue(row["calculation_valid_both_grids"])
            self.assertTrue(row["energy_refinement_passed"])
            self.assertTrue(row["positive_both_grids"])
            for grid in ("coarse", "fine"):
                sample = row[grid]
                self.assertTrue(sample["calculation_valid"])
                self.assertTrue(sample["energy_sign_is_not_a_validity_gate"])
                self.assertTrue(sample["uses_actual_positive_gauss_operator"])
                self.assertGreater(float(sample["positive_gauss_operator_min_eigenvalue"]), 0)
                self.assertGreater(float(sample["fixed_density_vector_energy_MeV3"]), 0)
                self.assertLess(float(sample["gauss_residual_max_dimensionless"]), 1e-10)
                self.assertLess(abs(float(sample["gauss_energy_identity_residual"])), 1e-8)
            self.assertLess(abs(float(row["omega_relative_refinement"])), 1e-2)

    def test_periodic_negative_witness_is_distinct_from_numerical_failure(self):
        sample = dict(audit.periodic_trial(self.design, 0.50, 0.10, grid_size=128))
        sample["omega_E_minus_muN_MeV3"] = "-1"
        negative = audit.classify_periodic_sample(sample)
        self.assertTrue(negative["calculation_valid"])
        self.assertTrue(negative["negative_witness"])
        self.assertEqual(negative["interpretation"], "valid_negative_periodic_witness")

        invalid = dict(sample)
        invalid["gauss_residual_max_dimensionless"] = "nan"
        failed = audit.classify_periodic_sample(invalid)
        self.assertFalse(failed["calculation_valid"])
        self.assertFalse(failed["negative_witness"])
        self.assertEqual(failed["interpretation"], "numerical_failure_or_insufficient_refinement")

    def test_acceptance_helpers_fail_closed_on_missing_nan_and_mutated_evidence(self):
        symbols = audit.symbolic_checks()
        missing = dict(symbols)
        missing.pop(next(iter(missing)))
        self.assertFalse(audit._symbolic_rows_passed(missing))
        mutated = dict(symbols)
        mutated["scalar_el_equation"] = dict(mutated["scalar_el_equation"])
        mutated["scalar_el_equation"]["residual"] = "1"
        self.assertFalse(audit._symbolic_rows_passed(mutated))

        bounds = dict(audit.tension_bounds(self.design))
        bounds["sigma0_integral_residual"] = "nan"
        self.assertFalse(audit._tension_evidence_passed(bounds))

        mixed = dict(audit.mixed_hessian_controls(self.design))
        mixed["actual_hessian_WA"] = "nan"
        self.assertFalse(audit._mixed_hessian_evidence_passed(mixed))

        self.assertFalse(audit._interface_attempt_passes({}))
        interface = audit.interface_refinement(self.design)
        final = dict(interface["final"])
        final["max_EL_residual_dimensionless"] = "nan"
        self.assertFalse(audit._interface_attempt_passes(final))

        with patch.object(audit, "symbolic_checks", return_value={}):
            rejected = audit.audit(dps=80, include_numerics=False)
        self.assertFalse(rejected["mathematical_checks_passed"])

    def test_80_and_120_digit_constants_agree_without_result_tables(self):
        low = audit.BindingDesign(dps=80)
        high = audit.BindingDesign(dps=120)
        for key in ("Cv", "gomega", "By", "Cy", "delta_y", "beta", "U", "Uy", "Uyy"):
            self.close(getattr(low, key), getattr(high, key), "1e-65")
            self.assertEqual(audit.number(getattr(low, key)), audit.number(getattr(high, key)))
        low_tension = audit.tension_bounds(low)
        high_tension = audit.tension_bounds(high)
        self.close(
            mp.mpf(low_tension["sigma0_MeV3_formula"]),
            mp.mpf(high_tension["sigma0_MeV3_formula"]),
            "1e-65",
        )

    def test_scope_flags_do_not_promote_counterfactual_or_finite_samples(self):
        result = audit.audit(dps=80, include_numerics=False)
        self.assertEqual(result["evidence_weight"], 0)
        self.assertTrue(result["mathematical_checks_passed"])
        self.assertTrue(result["design_inputs"]["all_are_fixed_calibration_inputs_not_predictions"])
        self.assertTrue(result["interface_bvp"]["skipped"])
        self.assertTrue(result["periodic_global_attack"]["skipped"])
        self.assertTrue(result["tension_bounds"]["not_global_unrestricted_minimum"])
        self.assertTrue(audit._analytic_certificates_passed(result["analytic_certificates"]))
        self.assertIn("counterfactual", result["scope"][0])

    def test_strict_json_cli_and_no_artifact_write(self):
        before = audit.SOURCE_PATH.read_bytes()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main(["--skip-numerics"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue(), parse_constant=lambda value: self.fail(value))
        self.assertIs(result["mathematical_checks_passed"], True)
        self.assertEqual(result["evidence_weight"], 0)
        self.assertEqual(audit.SOURCE_PATH.read_bytes(), before)
        with contextlib.redirect_stdout(output := io.StringIO()):
            code = audit.main(["--dps", "79"])
        self.assertEqual(code, 2)
        failed = json.loads(output.getvalue(), parse_constant=lambda value: self.fail(value))
        self.assertFalse(failed["mathematical_checks_passed"])


if __name__ == "__main__":
    unittest.main()
