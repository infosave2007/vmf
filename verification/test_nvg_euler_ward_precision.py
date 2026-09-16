"""Independent high-precision Ward controls; no output tables or artifact writes."""

import contextlib
from dataclasses import replace
import io
import json
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np

import nvg_euler_ward_precision as audit


class EulerWardPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jet = audit.manufactured_jet()
        cls.report = audit.compute_state()

    def exact_jet(self):
        return audit.BinaryJet(**{name: audit._mp_tree(getattr(self.jet, name))
                                 for name in self.jet.__dataclass_fields__})

    def test_all_six_independent_euler_and_connection_checks(self):
        self.assertTrue(audit.require_checks(self.report["independent_checks"]))
        self.assertEqual(set(self.report["independent_checks"]), audit.REQUIRED_CHECKS)
        self.assertIs(self.report["all_pass"], True)

    def test_80_120_precision_convergence(self):
        self.assertEqual(self.report["precision_dps"], [80, 120])
        self.assertTrue(audit.require_checks(self.report["precision_checks"]))
        for row in self.report["precision_checks"].values():
            self.assertLess(mp.mpf(row["relative_error"]), mp.mpf("1e-70"))

    def test_full_ward_residual_and_wrong_connection_sign(self):
        self.assertLess(mp.mpf(self.report["identity_relative"]), mp.mpf("1e-70"))
        self.assertGreater(mp.mpf(self.report["wrong_connection_sign_relative"]), mp.mpf("1e-10"))
        wrong = audit.evaluate(self.jet, connection_sign=-1)
        self.assertGreater(wrong["independent_ward"], mp.mpf("1e-10"))

    def test_same_original_binary_fields_and_gamma_matrices(self):
        import source_complete_solution_audit as original
        phi, dp, N, dN, A, _, _ = original.manufactured_fields()
        for actual, expected in ((self.jet.state[2], N), (self.jet.state[6], dN),
                                 (self.jet.state[8], A)):
            self.assertTrue(np.array_equal(np.asarray(actual), expected))
        for actual, expected in zip(audit.GAMMA, original.GAMMA):
            self.assertTrue(np.array_equal(np.asarray(actual), expected))
        self.assertEqual(self.jet.state[0], original.PARAMS.W0/np.sqrt(2.0)*(1+0.013j))

    def test_rng_seed_and_draw_order_match_declared_control(self):
        rng = np.random.default_rng(1701)
        sp = (rng.normal(size=(4, 4))+0.3j*rng.normal(size=(4, 4)))*0.01
        sN = (rng.normal(size=(4, 4, 4))+0.3j*rng.normal(size=(4, 4, 4)))*0.01
        dA = rng.normal(size=(4, 4))*0.01
        for actual, expected in ((self.jet.second_phi, sp), (self.jet.second_N, sN),
                                 (self.jet.first_A, dA)):
            self.assertTrue(np.array_equal(np.asarray(actual), expected))

    def test_binary_passport_fingerprint_is_input_sensitive(self):
        self.assertEqual(audit.input_fingerprint(self.jet),
                         audit.input_fingerprint(audit.manufactured_jet()))
        state = list(self.jet.state)
        state[0] += 0.01
        self.assertNotEqual(audit.input_fingerprint(self.jet),
                            audit.input_fingerprint(replace(self.jet, state=tuple(state))))

    def test_gamma_clifford_identity(self):
        for mu in range(4):
            for nu in range(4):
                left, right = np.array(audit.GAMMA[mu]), np.array(audit.GAMMA[nu])
                expected = 2*audit.ETA[mu]*np.eye(4) if mu == nu else np.zeros((4, 4))
                self.assertTrue(np.array_equal(left@right+right@left, expected))

    def test_conjugate_scalar_and_spinor_norm_checks(self):
        with mp.workdps(80):
            values = audit.analytic_eulers(self.exact_jet())
            self.assertEqual(values["euler_phi"], mp.conj(values["euler_phibar"]))
            self.assertEqual(audit._norm(values["euler_N"]), audit._norm(values["euler_bar_N"]))

    def test_independent_density_agrees_with_original_at_binary_precision(self):
        import source_complete_solution_audit as original
        state = tuple(np.array(value) if isinstance(value, tuple) else value for value in self.jet.state)
        legacy = original._manufactured_offshell_density(state)
        with mp.workdps(80):
            exact = self.exact_jet()
            value = audit.density(exact.state, exact.parameters)
            self.assertLess(abs(value-legacy)/max(1, abs(value)), mp.mpf("1e-11"))

    def test_wrong_analytic_spinor_sign_is_detected(self):
        original = audit.analytic_eulers
        def wrong(jet):
            result = original(jet)
            result["euler_N"] = tuple(-item for item in result["euler_N"])
            return result
        with patch.object(audit, "analytic_eulers", side_effect=wrong):
            result = audit.evaluate(self.jet)
        self.assertGreater(result["errors"]["euler_N"], mp.mpf("0.1"))
        self.assertGreater(result["analytic_ward"], mp.mpf("1e-10"))

    def test_gauge_invariant_wrong_yukawa_sign_is_still_detected(self):
        original = audit.analytic_eulers
        def wrong(jet):
            result = original(jet)
            phi, bar, N, barN = jet.state[:4]
            correction = 2*jet.parameters[2]*audit._dot(barN, N)/mp.sqrt(2*phi*bar)
            result["euler_phi"] += correction*bar
            result["euler_phibar"] += correction*phi
            return result
        with patch.object(audit, "analytic_eulers", side_effect=wrong):
            result = audit.evaluate(self.jet)
        self.assertGreater(result["errors"]["euler_phi"], mp.mpf("1e-10"))
        # Ward alone cannot reject every gauge-invariant wrong action term.
        self.assertLess(result["analytic_ward"], mp.mpf("1e-60"))

    def test_require_checks_rejects_empty_missing_and_unrelated(self):
        good = self.report["independent_checks"]
        for rows in ({}, {"unrelated": {"passed": True, "relative_error": "0"}},
                     {key: value for key, value in good.items() if key != "connection"}):
            self.assertFalse(audit.require_checks(rows))

    def test_require_checks_rejects_truthy_nonfinite_or_inaccurate_rows(self):
        good = self.report["independent_checks"]
        for row in ({"passed": "yes", "relative_error": "0"},
                    {"passed": True, "relative_error": "1e-30"},
                    {"passed": True, "relative_error": "nan"},
                    {"passed": True, "relative_error": "inf"},
                    {"passed": True, "relative_error": "-1"},
                    {"passed": True, "relative_error": False}):
            rows = dict(good, connection=row)
            self.assertFalse(audit.require_checks(rows))

    def test_invalid_precision_and_sign_fail_closed(self):
        for value in (True, 0, 69, 301, 80.5, "80", float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.evaluate(self.jet, dps=value)
        for value in (True, 0, 2, float("nan"), float("inf")):
            with self.subTest(sign=value), self.assertRaises(ValueError):
                audit.evaluate(self.jet, connection_sign=value)
        with self.assertRaises(ValueError):
            audit.compute_state(dps=80, verify_dps=80)

    def test_invalid_parameter_values_fail_closed(self):
        import source_complete_solution_audit as original
        for value in (True, 0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.manufactured_jet(replace(original.PARAMS, W0=value))

    def test_invalid_jet_values_and_dimensions_fail_closed(self):
        for value in (True, float("nan"), complex(1, float("inf")), 0):
            state = list(self.jet.state)
            state[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.evaluate(replace(self.jet, state=tuple(state)))
        with self.assertRaises(ValueError):
            audit.evaluate(replace(self.jet, first_A=()))

    def test_precision_context_restored_on_success_and_failure(self):
        before = mp.mp.prec
        audit.evaluate(self.jet, dps=70)
        self.assertEqual(mp.mp.prec, before)
        with patch.object(audit, "differentiated_eulers", side_effect=ValueError("injected")):
            with self.assertRaises(ValueError):
                audit.evaluate(self.jet, dps=70)
        self.assertEqual(mp.mp.prec, before)

    def test_legacy_interface_uses_calculated_fields_and_truthful_method(self):
        with patch.object(audit, "compute_state", return_value=self.report):
            result = audit.manufactured_euler_ward_controls()
        self.assertIs(result["all_pass"], True)
        self.assertIsNone(result["finite_difference_step"])
        self.assertEqual(result["precision_dps"], [80, 120])
        for name in ("euler_phi", "euler_phibar", "euler_N", "euler_bar_N"):
            self.assertEqual(result[name+"_abs"], round(float(self.report["norms"][name]), 6))
        json.dumps(result, allow_nan=False)

    def test_cli_json_success_failure_and_truthy_rejection(self):
        for report, expected in ((self.report, 0), ({"all_pass": "yes"}, 1),
                                 ({"all_pass": True}, 1)):
            stream = io.StringIO()
            with patch.object(audit, "compute_state", return_value=dict(report)), contextlib.redirect_stdout(stream):
                code = audit.main([])
            self.assertEqual(code, expected)
            self.assertIs(json.loads(stream.getvalue())["all_pass"], expected == 0)
        stream = io.StringIO()
        with patch.object(audit, "compute_state", side_effect=ValueError("injected")), contextlib.redirect_stdout(stream):
            self.assertEqual(audit.main([]), 2)
        self.assertIs(json.loads(stream.getvalue())["all_pass"], False)

    def test_full_result_rejects_forged_success_with_bad_checks_or_norms(self):
        for changes in ({"independent_checks": {}}, {"precision_checks": {}},
                        {"norms": {}}, {"identity_relative": "nan"},
                        {"identity_relative": "1e-20"},
                        {"wrong_connection_sign_relative": "0"}):
            self.assertFalse(audit.require_result(dict(self.report, **changes)))
        result = dict(self.report, independent_checks={})
        with patch.object(audit, "compute_state", return_value=result):
            self.assertIs(audit.manufactured_euler_ward_controls()["all_pass"], False)

    def test_malformed_cli_arguments_return_json_without_stderr(self):
        for argv in (["--dps", "69"], ["--dps", "not-a-number"],
                     ["--verify-dps"], ["--unknown"]):
            stdout, stderr = io.StringIO(), io.StringIO()
            with self.subTest(argv=argv), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(audit.main(argv), 2)
            result = json.loads(stdout.getvalue())
            self.assertIs(result["all_pass"], False)
            self.assertEqual(result["evidence_weight"], 0)
            self.assertTrue(result["error"])
            self.assertEqual(stderr.getvalue(), "")

    def test_cli_help_is_json_without_a_success_claim(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(audit.main(["--help"]), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "help")
        self.assertIs(result["all_pass"], False)
        self.assertEqual(result["evidence_weight"], 0)
        self.assertEqual(stderr.getvalue(), "")

    def test_no_nonfinite_legacy_float_conversion(self):
        norms = dict(self.report["norms"], euler_phi="1e1000")
        result = dict(self.report, norms=norms)
        with patch.object(audit, "compute_state", return_value=result), self.assertRaises(ValueError):
            audit.manufactured_euler_ward_controls()

    def test_report_scope_and_json_safety(self):
        self.assertEqual(self.report["evidence_weight"], 0)
        self.assertIn("not_physical_evidence", self.report["status"])
        self.assertIs(self.report["input_passport"]["binary_inputs_are_exact"], True)
        self.assertEqual(self.report["runtime"]["mpmath_backend"], mp.libmp.BACKEND)
        json.dumps(self.report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
