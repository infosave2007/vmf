"""Independent momentum integrals, limits, precision and failure controls."""
import contextlib
import io
import json
import unittest
from unittest import mock

import mpmath as mp

import nvg_collisionless_response_audit as audit
from source_complete_scaling_saturation_audit import BulkModel, INPUTS


class CollisionlessResponseTests(unittest.TestCase):
    def test_exact_identities_and_required_names(self):
        rows = audit.symbolic_checks()
        self.assertEqual(set(rows), audit.REQUIRED_IDENTITIES)
        for row in rows.values():
            self.assertIs(row["passed"], True, row)
            self.assertEqual(row["residual"], "0")

    def test_eight_original_states_and_static_energy_derivatives(self):
        with mp.workdps(80):
            model = BulkModel()
            for nr in (".001", ".01", ".1", "1", "2", "4", "10", "100"):
                with self.subTest(n_ratio=nr):
                    c = audit.coefficients(nr)
                    self.assertTrue(all(abs(v) < mp.mpf("1e-60") for v in c["residuals"].values()))
                    E = lambda nn, ww: model.state(nn, ww/model.W0)["energy_total"]
                    Enn = mp.diff(lambda nn: E(nn, c["W"]), c["n"], 2)
                    EnW = mp.diff(lambda ww: mp.diff(lambda nn: E(nn, ww), c["n"]), c["W"])
                    EWW = mp.diff(lambda ww: E(c["n"], ww), c["W"], 2)
                    self.assertLess(abs(EnW-c["B"]), mp.mpf("1e-55"))
                    self.assertLess(abs(EWW-c["C"])/c["C"], mp.mpf("1e-55"))
                    thermodynamic_chi = -1/(Enn-EnW*EnW/EWW)
                    self.assertLess(abs((-c["NF"]/(1+c["F0"]))/thermodynamic_chi-1), mp.mpf("1e-55"))
                    self.assertGreater(1+c["F0"], 0)
                    self.assertGreater(1+c["F1"]/3, 0)

    def test_independent_current_tadpole_integral(self):
        with mp.workdps(80):
            model = BulkModel()
            for nr in (".1", "1", "100"):
                c = audit.coefficients(nr)
                mass = model.MN*c["y"]
                integral = model.d/(2*mp.pi**2)*mp.quad(
                    lambda p: p*p*(1/mp.sqrt(p*p+mass*mass)
                                    -p*p/(3*(p*p+mass*mass)**mp.mpf("1.5"))), [0, c["kF"]])
                self.assertLess(abs(integral/(c["n"]/c["EF"])-1), mp.mpf("1e-65"))
                f1 = -c["V"]*c["vF2"]/(1+c["V"]*integral)
                self.assertLess(abs(c["NF"]*f1-c["F1"]), mp.mpf("1e-65"))

    def test_independent_angular_moments_at_computed_pole(self):
        with mp.workdps(80):
            for nr in (".001", "1", "10", "100"):
                c = audit.coefficients(nr)
                root = audit.zero_sound_root(c["F0"], c["r"])
                self.assertIsNotNone(root)
                sig = 1+root["gap"]
                L = mp.quad(lambda u: u/(sig-u)/2, [-1, 0, 1])
                moment2 = mp.quad(lambda u: u*u/(sig-u)/2, [-1, 0, 1])
                moment3 = mp.quad(lambda u: u**3/(sig-u)/2, [-1, 0, 1])
                self.assertLess(abs(L-audit.lindhard_from_log_gap(root["log_gap"])), mp.mpf("1e-65"))
                self.assertLess(abs(moment2-sig*L), mp.mpf("1e-65"))
                self.assertLess(abs(moment3-(sig*sig*L-mp.mpf(1)/3)), mp.mpf("1e-65"))
                determinant = ((1-c["F0"]*L)*(1-c["F1"]*moment3)
                               -c["F0"]*c["F1"]*moment2*moment2)
                self.assertLess(abs(determinant), mp.mpf("1e-60"))

    def test_sub_luminal_bracket_and_monotonicity(self):
        with mp.workdps(80):
            for nr in (".001", "1", "10", "100"):
                c = audit.coefficients(nr)
                root = audit.zero_sound_root(c["F0"], c["r"])
                sig = 1+root["gap"]
                bound = mp.sqrt(c["F0"]/(3*c["r"]))
                self.assertTrue(1 < sig < bound)
                self.assertTrue(c["vF2"] < c["vF2"]*sig*sig < 1)
                for value in ((1+sig)/2, sig, (sig+bound)/2):
                    derivative = mp.diff(lambda ss: (c["F0"]-3*c["r"]*ss*ss)
                                         *(ss*mp.log((ss+1)/(ss-1))/2-1), value)
                    self.assertLess(derivative, 0)

    def test_80_120_digit_coefficients_and_root(self):
        with mp.workdps(120):
            for nr in ("1", "100", "150", "200", "1e-12"):
                low = audit.coefficients(nr, dps=80)
                high = audit.coefficients(nr, dps=120)
                for key in ("B", "C", "F0", "F1", "first_sound_squared"):
                    self.assertEqual(audit.number(low[key]), audit.number(high[key]))
                    self.assertLess(abs(low[key]/high[key]-1), mp.mpf("1e-60"))
                root80 = audit.zero_sound_root(low["F0"], low["r"], dps=80)
                root120 = audit.zero_sound_root(high["F0"], high["r"], dps=120)
                if root80 is None:
                    self.assertIsNone(root120)
                else:
                    self.assertIsNotNone(root120)
                    self.assertEqual(audit.number(root80["log_gap"]), audit.number(root120["log_gap"]))
                    self.assertLess(abs(root80["log_gap"]/root120["log_gap"]-1), mp.mpf("1e-60"))

    def test_exponentially_small_gap_is_not_a_root_at_one(self):
        with mp.workdps(80):
            c = audit.coefficients("1e-12")
            root = audit.zero_sound_root(c["F0"], c["r"])
            self.assertTrue(0 < root["gap"] < mp.eps)
            self.assertLess(root["log_gap"], -100)
            self.assertLess(abs(root["residual"]), mp.mpf("1e-60"))
            result = audit.audit("1e-12")
            self.assertTrue(result["zero_sound"]["positive_gap_retained_separately_from_rounded_speed"])
            self.assertGreater(mp.mpf(result["zero_sound"]["sigma_minus_one"]), 0)
            self.assertGreater(mp.mpf(result["zero_sound"]["speed_squared_above_continuum"]), 0)

    def test_no_isolated_pole_is_not_a_fluid_instability(self):
        with mp.workdps(80):
            c = audit.coefficients("200")
            self.assertLess(c["F0"]-3*c["r"], 0)
            self.assertIsNone(audit.zero_sound_root(c["F0"], c["r"]))
            self.assertGreater(c["first_sound_squared"], 0)
            self.assertGreater(1+c["F0"], 0)
            self.assertGreater(1+c["F1"]/3, 0)
            self.assertIsNone(audit.zero_sound_root(".5", mp.mpf(1)/3))
            self.assertIsNone(audit.zero_sound_root(1, mp.mpf(1)/3))

    def test_numerical_threshold_witness_not_global_uniqueness(self):
        low, high = audit.threshold_witness(dps=80), audit.threshold_witness(dps=120)
        self.assertEqual(low["n_over_n0"], high["n_over_n0"])
        self.assertFalse(low["globally_unique_density_threshold_proven"])
        with mp.workdps(80):
            n = mp.mpf(low["n_over_n0"])
            for factor, sign in ((mp.mpf(".999"), 1), (mp.mpf("1.001"), -1)):
                c = audit.coefficients(n*factor)
                self.assertGreater(sign*(c["F0"]-3*c["r"]), 0)

    def test_omitting_current_feedback_creates_false_superluminal_pole(self):
        with mp.workdps(80):
            c = audit.coefficients("10")
            correct = audit.zero_sound_root(c["F0"], c["r"])
            wrong = audit.zero_sound_root(c["F0"], 0)
            self.assertLess(c["vF2"]*(1+correct["gap"])**2, 1)
            self.assertGreater(c["vF2"]*(1+wrong["gap"])**2, 1)

    def test_first_sound_substitution_fails_collisionless_equation(self):
        with mp.workdps(80):
            c = audit.coefficients("1")
            sigma = mp.sqrt(c["first_sound_squared"]/c["vF2"])
            self.assertGreater(sigma, 1)
            residual = audit.dispersion_from_log_gap(mp.log(sigma-1), c["F0"], 3*c["r"])
            self.assertGreater(abs(residual), mp.mpf(".01"))

    def test_dropping_scalar_vector_mixing_breaks_static_response(self):
        with mp.workdps(80):
            c = audit.coefficients("100")
            correct = c["a"]+c["V"]-c["B"]**2/c["C"]
            wrong = c["a"]+c["V"]-c["b"]**2/c["C"]
            self.assertGreater(abs(wrong/correct-1), mp.mpf(".01"))

    def test_invalid_inputs_fail_closed(self):
        for invalid in (True, None, "nan", "inf", "0", "-1", "1e-19", "1e7"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                audit.coefficients(invalid)
        for invalid in (True, 20, 301, 80.0):
            with self.assertRaises(ValueError):
                audit.coefficients(dps=invalid)
        for f0, r in ((True, 1), (1, True), ("nan", 1), (1, "inf"), (1, -1)):
            with self.assertRaises(ValueError):
                audit.zero_sound_root(f0, r)

    def test_incomplete_symbolic_result_cannot_claim_success(self):
        complete = audit.symbolic_checks()
        short = dict(complete)
        short.pop(next(iter(short)))
        for invalid in ({}, {"unrelated": {"passed": True, "residual": "0"}}, short):
            with mock.patch.object(audit, "symbolic_checks", return_value=invalid):
                self.assertFalse(audit.audit()["mathematical_checks_passed"])

    def test_incomplete_nonfinite_residuals_cannot_claim_success(self):
        original = audit.coefficients()
        short = dict(original["residuals"])
        short.pop(next(iter(short)))
        for bad in ({}, short, {**original["residuals"], "C_matches_source_energy": mp.nan},
                    {**original["residuals"], "C_matches_source_energy": False}):
            with mock.patch.object(audit, "coefficients", return_value={**original, "residuals": bad}):
                with self.assertRaises(ArithmeticError):
                    audit.audit()

    def test_cli_strict_json_no_artifacts_and_honest_scope(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main([])
        result = json.loads(output.getvalue(), parse_constant=lambda s: self.fail(s))
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], audit.STATUS)
        self.assertEqual(result["evidence_weight"], 0)
        self.assertEqual(result["inputs"]["original_parameters"], INPUTS)
        self.assertIn("No finite-k", " ".join(result["scope"]))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main(["--n-ratio", "0"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "invalid_or_failed_audit")
        for args in (["--dps", "wrong"], ["--not-an-option"]):
            output, error = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                code = audit.main(args)
            result = json.loads(output.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(error.getvalue(), "")
            self.assertEqual(result["evidence_weight"], 0)
            self.assertIs(result["mathematical_checks_passed"], False)


if __name__ == "__main__":
    unittest.main()
