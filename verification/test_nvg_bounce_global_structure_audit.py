"""Independent algebra, integral, inequality and failure tests; no saved tables."""
import contextlib
import io
import json
import unittest
from unittest import mock

import mpmath as mp

import nvg_bounce_global_structure_audit as audit


class GlobalStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbols = audit.symbolic_checks()

    def test_symbolic_identities(self):
        self.assertGreaterEqual(len(self.symbols), 17)
        for row in self.symbols.values():
            self.assertIs(row["passed"], True)
            self.assertEqual(row["residual"], "0")

    def test_wrong_scalar_force_breaks_energy_and_constraint(self):
        wrong = audit.symbolic_checks(scalar_force_sign=-1)
        self.assertFalse(wrong["matter_continuity_off_equilibrium"]["passed"])
        self.assertFalse(wrong["sine_constraint_propagation"]["passed"])

    def test_wrong_q_rate_breaks_gravity(self):
        wrong = audit.symbolic_checks(sine_rate_scale="2")
        self.assertFalse(wrong["sine_constraint_propagation"]["passed"])
        self.assertFalse(wrong["raychaudhuri_from_regular_q"]["passed"])

    def test_off_equilibrium_moving_field_witnesses(self):
        for n, W, v, rc in (("1", "1", "0", "1"), ("10", "1.01", ".01", "1"),
                            ("2", ".5", "-.2", "1"), ("100", "1.4", ".3", "10")):
            for direction in (-1, 1):
                with self.subTest(n=n, W=W, v=v, direction=direction):
                    result = audit.point_state(n, W, v, rc, direction=direction)
                    self.assertIs(result["mathematical_checks_passed"], True, result)
                    self.assertTrue(all(result["inequalities"].values()))
                    self.assertIs(result["stationarity_imposed"], False)

    def test_independent_precision(self):
        a = audit.point_state("2", ".5", "-.2", "1", dps=80)
        b = audit.point_state("2", ".5", "-.2", "1", dps=120)
        self.assertEqual(a["state"], b["state"])
        self.assertEqual(a["bounds"], b["bounds"])

    def test_curvature_bounds_grid_and_saturation(self):
        # Independent numerical evaluation supports, but does not replace,
        # the exact endpoint/convexity certificates in the implementation.
        with mp.workdps(80):
            for i in range(41):
                x = mp.mpf(i)/40
                for j in range(41):
                    u = mp.mpf(j)/20
                    H2 = x*(1-x)/3
                    Hd = -u*x*(1-2*x)/2
                    R = 6*(Hd+2*H2)
                    K = 12*((Hd+H2)**2+H2**2)
                    self.assertGreaterEqual(R+mp.mpf('1e-75'), -mp.mpf(1)/8)
                    self.assertLessEqual(R, 6)
                    self.assertLessEqual(K, 12)
            x, u = mp.mpf(1)/8, mp.mpf(2)
            self.assertEqual(-3*u*x*(1-2*x)+4*x*(1-x), -mp.mpf(1)/8)

    def test_nonvacuum_time_and_size_bounds(self):
        result = audit.point_state(direction=-1)
        with mp.workdps(80):
            self.assertGreater(mp.mpf(result["bounds"]["time_to_bounce_upper_MeV_minus1"]), 0)
            self.assertGreater(mp.mpf(result["bounds"]["a_min_over_a_initial"]), 0)
            self.assertLess(mp.mpf(result["bounds"]["a_min_over_a_initial"]), 1)
        self.assertEqual(result["bounds"]["time_direction_to_bounce"], "future")

    def test_invalid_inputs_fail_closed(self):
        for key in ("n_ratio", "W_ratio", "rho_c_ratio"):
            for bad in (True, "nan", "inf", "0", "-1", None):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    audit.point_state(**{key: bad})
        for bad in (True, "nan", "inf", None):
            with self.assertRaises(ValueError):
                audit.point_state(speed_ratio=bad)
        for direction in (True, 0, 2):
            with self.assertRaises(ValueError):
                audit.point_state(direction=direction)
        for dps in (True, 20, 301, 80.5):
            with self.assertRaises(ValueError):
                audit.point_state(dps=dps)
        with self.assertRaises(ValueError):
            audit.point_state(speed_ratio="2")
        with self.assertRaises(ValueError):
            audit.symbolic_checks(scalar_force_sign=True)

    def test_no_false_success_from_symbolic_status(self):
        incomplete = dict(self.symbols)
        incomplete.pop("sine_constraint_propagation")
        for invalid in ({}, incomplete, {"unrelated": {"passed": True, "residual": "0"}},
                        {"bad": {"passed": "yes", "residual": "0"}},
                        {"bad": {"passed": True, "residual": "1"}},
                        {"bad": {"passed": True, "residual": 0}}):
            with mock.patch.object(audit, "symbolic_checks", return_value=invalid):
                self.assertIs(audit.audit()["mathematical_checks_passed"], False)

    def test_cli_honest_status_and_invalid_cap(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main([])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], audit.STATUS)
        self.assertIs(result["full_history_integrated"], False)
        self.assertIs(result["rho_c_is_external_input"], True)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main(["--rho-c-ratio", "0.00001"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "invalid_or_failed_audit")


if __name__ == "__main__":
    unittest.main()
