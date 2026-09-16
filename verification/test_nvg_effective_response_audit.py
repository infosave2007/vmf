"""Adversarial checks for the real-amplitude gyroscopic response audit."""
import contextlib
import io
import json
import unittest
from unittest import mock

import mpmath as mp

import nvg_effective_response_audit as audit
import nvg_longitudinal_fluid_audit as fluid


class EffectiveResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = fluid.coefficients("1", dps=80)
        cls.coefficients = cls.source["coefficients"]

    def test_symbolic_passport_derives_real_gauss_and_generic_schur(self):
        checks = audit.symbolic_checks()
        self.assertEqual(set(checks), audit.REQUIRED_SYMBOLIC_CHECKS)
        self.assertEqual(len(checks), 14)
        self.assertTrue(audit._symbolic_passed(checks))
        self.assertEqual(checks["gauss_solution_from_real_action"]["residual"], "0")
        self.assertEqual(checks["longitudinal_determinant_bridge"]["residual"], "0")

    def test_wrong_sign_mutation_is_not_accepted(self):
        wrong = audit.symbolic_checks(correction_sign=-1)
        self.assertFalse(audit._symbolic_passed(wrong))
        self.assertFalse(wrong["generic_Meff"]["passed"])
        self.assertNotEqual(wrong["generic_Meff"]["residual"], "0")

    def test_generic_one_retained_two_excluded_composition(self):
        H = mp.matrix([[4, 1, 2], [1, 3, 1], [2, 1, 2]])
        M = mp.matrix([[2, mp.mpf(".2"), mp.mpf("-.1")],
                       [mp.mpf(".2"), 3, mp.mpf(".4")],
                       [mp.mpf("-.1"), mp.mpf(".4"), 2]])
        B = mp.matrix([[0, 2, -1], [-2, 0, mp.mpf(".5")],
                       [1, mp.mpf("-.5"), 0]])
        result = audit.low_frequency_effective(H, M, B, 1)
        self.assertEqual(result["R"].rows, 3)
        self.assertEqual(result["R"].cols, 1)
        self.assertLess(audit.matrix_norm(result["R"].T*H*result["Ez"]), mp.mpf("1e-70"))
        self.assertLess(audit.matrix_norm(result["C"]-result["R"].T*B*result["Ez"]), mp.mpf("1e-70"))
        for omega in (mp.mpf("1e-3"), mp.mpf("5e-4")):
            exact = audit.exact_schur(H, M, B, omega, 1)
            approx = (result["Heff"]+mp.j*omega*result["Beff"]
                      -omega**2*result["Meff"])
            error = audit._relative_matrix_error(exact, approx)
            if omega == mp.mpf("1e-3"):
                first = error
            else:
                self.assertLess(error, first)
        self.assertGreater(first, 0)

    def test_real_gauss_reduction_keeps_A_amplitude_at_zero_frequency(self):
        with mp.workdps(80):
            matrices = audit.gauss_eliminated_matrices(self.coefficients, 0)
            self.assertEqual(matrices["phi_solution"],
                             "(k*dot(A)-c*w-g*n*k*xi)/(k^2+M2)")
            self.assertEqual(matrices["B"][0, 2], -self.coefficients["g"]*self.coefficients["n"])
            self.assertEqual(matrices["B"][1, 2], 0)
            self.assertLess(audit.matrix_norm(matrices["H"]-matrices["H"].T), mp.mpf("1e-70"))
            self.assertLess(audit.matrix_norm(matrices["M"]-matrices["M"].T), mp.mpf("1e-70"))
            self.assertLess(audit.matrix_norm(matrices["B"]+matrices["B"].T), mp.mpf("1e-70"))

    def test_static_and_live_determinant_bridges_at_finite_k(self):
        with mp.workdps(80):
            for k in (0, mp.mpf("50"), mp.mpf("120")):
                static = audit.static_bridge(self.coefficients, k)
                self.assertLess(static["static_hessian_relative_error"], mp.mpf("1e-65"))
                for ratio in (mp.mpf(1)/32, mp.mpf(1)/64):
                    omega = ratio*mp.sqrt(k*k+self.coefficients["M2"])
                    bridge = audit.determinant_bridge(self.coefficients, k, omega)
                    self.assertLess(bridge["determinant_reduction_relative_error"], mp.mpf("1e-55"))

    def test_exact_and_expanded_nvg_response_are_compared_away_from_pole(self):
        result = audit.finite_frequency_check(self.coefficients, 50, dps=80)
        self.assertTrue(result["away_from_excluded_poles"])
        self.assertTrue(result["expanded_response_passed"])
        self.assertLess(result["relative_errors_exact_vs_expanded"][1],
                        result["relative_errors_exact_vs_expanded"][0])
        wrong = audit.finite_frequency_check(self.coefficients, 50, dps=80,
                                             correction_sign=-1)
        self.assertFalse(wrong["expanded_response_passed"])
        self.assertGreater(wrong["relative_errors_exact_vs_expanded"][1],
                           result["relative_errors_exact_vs_expanded"][1]*100)

    def test_acoustic_limit_matches_current_determinant_and_inertia(self):
        result = audit.acoustic_limit(self.coefficients, dps=80, n_ratio="1")
        self.assertLess(result["determinant_relative_error"], mp.mpf("1e-65"))
        self.assertLess(result["source_relative_error"], mp.mpf("1e-65"))
        self.assertIs(result["source_comparison_checked"], True)
        self.assertLess(result["inertial_relative_error"], mp.mpf("1e-70"))
        static = fluid.static_hessian(self.coefficients, 0)
        relaxed_force = self.coefficients["n"]*(static[0][0]-static[0][1]**2/static[1][1])
        wrong_inertia = self.coefficients["n"]*self.coefficients["mu_F"]
        wrong_speed = relaxed_force/wrong_inertia
        self.assertGreater(abs(wrong_speed-result["acoustic_squared_speed"]), mp.mpf("1e-3"))
        self.assertGreater(abs(result["low_frequency_inertial_coefficient"]-wrong_inertia), mp.mpf("1"))
        self.assertTrue(result["finite_k_group_velocity_not_claimed"])

    def test_acoustic_limit_without_density_ratio_marks_source_comparison_unavailable(self):
        result = audit.acoustic_limit(self.coefficients, dps=80)
        self.assertIs(result["source_acoustic_squared_speed"], None)
        self.assertIs(result["source_relative_error"], None)
        self.assertIs(result["source_comparison_checked"], False)
        self.assertLess(result["determinant_relative_error"], mp.mpf("1e-65"))

    def test_positive_and_indefinite_counterexamples_remain_visible(self):
        controls = audit.counterexamples(dps=80)
        self.assertTrue(controls["positive_Hzz_and_M"]["positivity_implication_passed"])
        self.assertTrue(controls["indefinite_Hzz_counterexample"]["negative_effective_inertia_visible"])
        self.assertFalse(controls["indefinite_Hzz_counterexample"]["positivity_certificate_applicable"])
        self.assertTrue(controls["indefinite_Hzz_counterexample"]["general_schur_algebra_still_defined"])
        self.assertTrue(controls["wrong_sign_control"]["wrong_sign_rejected"])

    def test_matrix_validation_and_singular_gap_fail_closed(self):
        H = [[2, 0], [0, 1]]
        M = [[1, 0], [0, 1]]
        B = [[0, 1], [-1, 0]]
        with self.assertRaises(ValueError):
            audit.low_frequency_effective(H, [[1, 1], [0, 1]], B, 1)
        with self.assertRaises(ValueError):
            audit.low_frequency_effective(H, M, [[0, 1], [1, 0]], 1)
        with self.assertRaises(ValueError):
            audit.low_frequency_effective([[2, 0], [0, 0]], M, B, 1)
        with self.assertRaises(ValueError):
            audit.low_frequency_effective(H, M, B, 0, correction_sign=0)
        for bad in (True, -1, "nan", "inf", complex(1, 2)):
            with self.assertRaises(ValueError):
                audit.gauss_eliminated_matrices(self.coefficients, bad)

    def test_nvg_connection_is_live_and_has_zero_evidence_weight(self):
        result = audit.nvg_connection("1", "50", dps=80)
        self.assertIs(result["mathematical_checks_passed"], True)
        self.assertEqual(result["source_producer"], "verification/nvg_longitudinal_fluid_audit.py")
        self.assertIs(result["source_independent_checks_passed"], True)
        self.assertIs(result["acoustic"]["source_comparison_checked"], True)
        self.assertEqual(result["units"], "natural_hbar_c_1; k and omega in MeV")
        self.assertEqual(result["acoustic"]["finite_k_group_velocity_not_claimed"], True)

    def test_nvg_connection_rejects_failed_and_nonboolean_source_status(self):
        for marker in (False, None, "yes", 1):
            with self.subTest(marker=marker):
                mutated = {**self.source, "coefficients": dict(self.source["coefficients"])}
                mutated["independent_checks_passed"] = marker
                with mock.patch.object(audit.fluid, "coefficients", return_value=mutated):
                    result = audit.nvg_connection("1", "50", dps=80)
                self.assertIs(result["source_independent_checks_passed"], marker)
                self.assertIs(result["mathematical_checks_passed"], False)

    def test_nvg_connection_propagates_missing_source_status(self):
        mutated = {**self.source, "coefficients": dict(self.source["coefficients"])}
        mutated.pop("independent_checks_passed")
        with mock.patch.object(audit.fluid, "coefficients", return_value=mutated):
            result = audit.nvg_connection("1", "50", dps=80)
        self.assertIs(result["source_independent_checks_passed"], None)
        self.assertIs(result["mathematical_checks_passed"], False)

    def test_full_audit_preserves_scope_and_next_closure(self):
        result = audit.audit("1", "50", dps=80)
        self.assertIs(result["mathematical_checks_passed"], True)
        self.assertEqual(result["evidence_weight"], 0)
        self.assertIn("cuscuton/gravity", result["next_missing_closure"])
        self.assertFalse(result["wrong_sign_symbolic_checks_passed"])

    def test_fixed_representative_density_state_is_live(self):
        result = audit.compute_state(k_MeV="50", dps=80)
        self.assertIs(result["mathematical_checks_passed"], True)
        self.assertEqual(tuple(result["representative_density_ratios"]), ("1", "10"))
        self.assertEqual(len(result["connections"]), 2)
        self.assertTrue(all(row["mathematical_checks_passed"] for row in result["connections"]))
        self.assertTrue(all(row["source_independent_checks_passed"] is True
                            for row in result["connections"]))
        self.assertTrue(all(row["acoustic"]["source_comparison_checked"] is True
                            and row["acoustic"]["source_relative_error"] is not None
                            for row in result["connections"]))

    def test_cli_outputs_json_only_and_fails_closed_on_bad_dps(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = audit.main(["--n-ratio", "1", "--k-MeV", "50", "--dps", "80"])
        self.assertEqual(code, 0)
        parsed = json.loads(output.getvalue())
        self.assertEqual(parsed["status"], audit.STATUS)
        self.assertIs(parsed["mathematical_checks_passed"], True)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code = audit.main(["--dps", "20"])
        self.assertEqual(code, 2)
        parsed = json.loads(output.getvalue())
        self.assertIs(parsed["mathematical_checks_passed"], False)
        self.assertEqual(parsed["evidence_weight"], 0)

    def test_cli_rejects_truthy_success_marker(self):
        fake = {"status": audit.STATUS, "evidence_weight": 0,
                "mathematical_checks_passed": "yes"}
        with mock.patch.object(audit, "audit", return_value=fake):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = audit.main([])
        self.assertEqual(code, 1)
        self.assertIs(json.loads(output.getvalue())["mathematical_checks_passed"], False)


if __name__ == "__main__":
    unittest.main()
