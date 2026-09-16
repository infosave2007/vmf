"""Independent differentiation, integral, and boundary checks for Hayward geometry."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mpmath as mp
import numpy as np
from scipy.integrate import quad
import sympy as sp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_hayward_evaporation as canonical
import regular_core_geometry_audit as audit


class RegularCoreGeometryTests(unittest.TestCase):
    def assertRelative(self, actual, expected, tolerance=3e-12):
        self.assertTrue(math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=1e-14),
                        (actual, expected))

    def test_stresses_from_independent_mass_differentiation(self):
        with mp.workdps(60):
            for M, ell in ((mp.mpf("0.4"), mp.mpf("1.3")), (mp.mpf(7), mp.mpf("0.2"))):
                mass = lambda r: M * r**3 / (r**3 + 2 * M * ell**2)
                for r in (mp.mpf("0.03"), mp.mpf("0.4"), mp.mpf(3), mp.mpf(20)):
                    state = audit.stress_tensor(r, M, ell)
                    rho = mp.diff(mass, r) / (4 * mp.pi * r**2)
                    pt = -mp.diff(mass, r, 2) / (8 * mp.pi * r)
                    self.assertRelative(state["rho"], rho)
                    self.assertRelative(state["p_t"], pt)
                    self.assertRelative(state["anisotropy"], rho + pt)

    def test_density_integrates_to_enclosed_and_total_mass(self):
        for M, ell in ((0.4, 1.3), (7.0, 0.2)):
            scale = (2 * M * ell**2)**(1 / 3)
            integrand = lambda x: 4 * math.pi * (scale * x)**2 * audit.stress_tensor(scale * x, M, ell)["rho"] * scale
            for xmax in (0.05, 1.0, 20.0, math.inf):
                recovered, error = quad(integrand, 0, xmax, epsabs=1e-12, epsrel=1e-12)
                target = M if math.isinf(xmax) else audit.mass_function(scale * xmax, M, ell)
                self.assertLess(abs(recovered - target), max(5 * error, 2e-12 * M))

    def test_conservation_from_independent_density_derivative(self):
        with mp.workdps(60):
            M, ell = mp.mpf(2), mp.mpf("0.7")
            mass = lambda r: M * r**3 / (r**3 + 2 * M * ell**2)
            rho = lambda r: mp.diff(mass, r) / (4 * mp.pi * r**2)
            for r in (mp.mpf("0.01"), mp.mpf(1), mp.mpf(10)):
                pr_prime = -mp.diff(rho, r)
                pt = -mp.diff(mass, r, 2) / (8 * mp.pi * r)
                residual = pr_prime + 2 * (-rho(r) - pt) / r
                self.assertLess(abs(residual), mp.mpf("1e-50"))
                self.assertLess(abs(audit.conservation_residual(r, M, ell)), 1e-12)
        scales = audit.critical_scales(0.7)
        self.assertLess(abs(audit.conservation_residual(scales["r_crit"], scales["M_crit"], 0.7)), 1e-12)
        self.assertEqual(audit.conservation_residual(0, 2, 0.7), 0)

    def test_curvature_from_independent_metric_derivatives(self):
        with mp.workdps(60):
            M, ell = mp.mpf("1.2"), mp.mpf("0.8")
            f = lambda r: 1 - 2 * M * r**2 / (r**3 + 2 * M * ell**2)
            for r in (mp.mpf("0.001"), mp.mpf("0.4"), mp.mpf(2), mp.mpf(30)):
                f0, f1, f2 = f(r), mp.diff(f, r), mp.diff(f, r, 2)
                scalar = -f2 - 4 * f1 / r + 2 * (1 - f0) / r**2
                t_component = -f2 / 2 - f1 / r
                angular_component = (1 - f0 - r * f1) / r**2
                ricci2 = 2 * t_component**2 + 2 * angular_component**2
                kretschmann = f2**2 + 4 * (f1 / r)**2 + 4 * ((1 - f0) / r**2)**2
                values = audit.curvature_invariants(r, M, ell)
                self.assertRelative(values["ricci_scalar"], scalar)
                self.assertRelative(values["ricci_squared"], ricci2)
                self.assertRelative(values["kretschmann"], kretschmann)
                self.assertRelative(values["weyl_squared"], kretschmann - 2 * ricci2 + scalar**2 / 3)

    def test_central_limits_match_de_sitter_curvature(self):
        r, ell = sp.symbols("r ell", positive=True)
        f = 1 - r**2 / ell**2
        scalar = sp.simplify(-sp.diff(f, r, 2) - 4 * sp.diff(f, r) / r + 2 * (1 - f) / r**2)
        K = sp.simplify(sp.diff(f, r, 2)**2 + 4 * (sp.diff(f, r) / r)**2 + 4 * ((1 - f) / r**2)**2)
        for length in (0.2, 1.0, 3.0):
            central = audit.curvature_invariants(0, 2, length)
            self.assertRelative(central["ricci_scalar"], scalar.subs(ell, length))
            self.assertRelative(central["kretschmann"], K.subs(ell, length))
            self.assertRelative(central["ricci_squared"], scalar.subs(ell, length)**2 / 4)
            self.assertEqual(central["weyl_squared"], 0)

    def test_energy_conditions_follow_directional_stress(self):
        M, ell = 2.0, 0.7
        for q in (0, 0.1, 0.49, 0.51, 1, 1.99, 2.01, 8, 100):
            r = (2 * M * ell**2 * q)**(1 / 3)
            stress = audit.stress_tensor(r, M, ell)
            values = audit.energy_conditions(r, M, ell)
            self.assertEqual(values["radial_nec"], 0)
            self.assertRelative(values["tangential_nec"], stress["rho"] + stress["p_t"])
            self.assertGreaterEqual(values["tangential_nec"], 0)
            self.assertEqual(values["sec_sum"] > 0, q > 0.5)
            self.assertEqual(values["tangential_dec_margin"] >= 0, q <= 2)
        for r in (math.sqrt(3) * ell * 1.01, 3 * ell, 10 * ell):
            M = audit.horizon_mass(r, ell)
            self.assertLess(audit.energy_conditions(r, M, ell)["tangential_dec_margin"], 0)

    def test_schwarzschild_deviation_and_asymptotic_coefficient(self):
        with mp.workdps(70):
            M, ell = mp.mpf("1.2"), mp.mpf("0.7")
            for r in (mp.mpf(2), mp.mpf(20), mp.mpf("1e5")):
                hayward = 1 - 2 * M * r**2 / (r**3 + 2 * M * ell**2)
                schwarzschild = 1 - 2 * M / r
                deviation = audit.schwarzschild_deviation(r, M, ell)
                self.assertRelative(deviation, hayward - schwarzschild)
                self.assertGreater(deviation, 0)
            r = mp.mpf("1e5")
            self.assertRelative(r**4 * audit.schwarzschild_deviation(r, M, ell), 4 * M**2 * ell**2)

    def test_horizon_and_temperature_extrema_by_differentiation(self):
        r, ell = sp.symbols("r ell", positive=True)
        M = r**3 / (2 * (r**2 - ell**2))
        T = (r**2 - 3 * ell**2) / (4 * sp.pi * r**3)
        critical_radii = sp.solve(sp.diff(M, r), r)
        peak_radii = sp.solve(sp.diff(T, r), r)
        self.assertEqual(len(critical_radii), 1)
        self.assertEqual(len(peak_radii), 1)
        for length in (0.3, 1.0, 7.0):
            scales = audit.critical_scales(length)
            rcrit = float(critical_radii[0].subs(ell, length))
            rpeak = float(peak_radii[0].subs(ell, length))
            self.assertRelative(scales["r_crit"], rcrit)
            self.assertRelative(scales["r_temperature_peak"], rpeak)
            self.assertGreater(float(sp.diff(M, r, 2).subs({r: rcrit, ell: length})), 0)
            self.assertLess(float(sp.diff(T, r, 2).subs({r: rpeak, ell: length})), 0)
            self.assertRelative(scales["M_crit"], audit.horizon_mass(rcrit, length))
            self.assertRelative(scales["M_temperature_peak"], audit.horizon_mass(rpeak, length))
            self.assertRelative(scales["T_max"], audit.horizon_temperature(rpeak, length))
            self.assertEqual(audit.horizon_temperature(scales["r_crit"], length), 0)

    def test_symbolic_result_is_computed_and_pure(self):
        with patch.object(Path, "write_text", side_effect=AssertionError("unexpected write")):
            result = audit.build_result()
        for name, check in result["checks"]["symbolic"].items():
            self.assertEqual(sp.sympify(check["residual"]), 0, name)
            self.assertTrue(check["passed"], name)
        self.assertTrue(result["checks"]["mass_quadrature"]["passed"])
        self.assertEqual(result["evidence_status"], audit.EVIDENCE_STATUS)
        self.assertIsNone(result["observed_likelihood"])
        self.assertTrue(audit.validate_result(result))
        changed = copy.deepcopy(result)
        changed["central_invariants_at_unit_length"]["kretschmann"] *= 1.1
        self.assertFalse(audit.validate_result(changed))

    def test_validator_rejects_boolean_numeric_substitutions_and_nonfinite_json(self):
        result = audit.build_result()
        for key, value in (("schema_version", True), ("evidence_weight", False)):
            changed = copy.deepcopy(result)
            changed[key] = value
            self.assertEqual(changed, result)  # This was accepted by plain dict equality.
            self.assertFalse(audit.validate_result(changed))
        changed = copy.deepcopy(result)
        check = next(iter(changed["checks"]["symbolic"].values()))
        check["passed"] = 1
        self.assertEqual(changed, result)
        self.assertFalse(audit.validate_result(changed))
        changed = copy.deepcopy(result)
        changed["unit_length_horizon_scales"]["T_max"] = math.nan
        self.assertFalse(audit.validate_result(changed))

    def test_cli_writes_only_with_explicit_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            with patch.object(audit, "RESULT_PATH", output), contextlib.redirect_stdout(io.StringIO()) as captured:
                result = audit.main([])
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(captured.getvalue()), result)
            with patch.object(audit, "RESULT_PATH", output), contextlib.redirect_stdout(io.StringIO()):
                written = audit.main(["--write"])
            self.assertEqual(json.loads(output.read_text()), written)

    def test_invalid_geometry_parameters_fail_explicitly(self):
        functions = (audit.mass_function, audit.metric_function, audit.stress_tensor,
                     audit.curvature_invariants, audit.energy_conditions, audit.conservation_residual)
        for function in functions:
            for invalid in (-1, math.nan, math.inf, -math.inf):
                with self.assertRaises(ValueError):
                    function(invalid, 1, 1)
                with self.assertRaises(ValueError):
                    function(1, invalid, 1)
                with self.assertRaises(ValueError):
                    function(1, 1, invalid)
            with self.assertRaises(ValueError):
                function(1, 0, 1)
            with self.assertRaises(ValueError):
                function(1, 1, 0)


class CanonicalHorizonBoundaryTests(unittest.TestCase):
    def test_cli_falsifiers_name_the_computed_threshold_and_specific_hypothesis(self):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            result = canonical.main()
        output = captured.getvalue()
        self.assertNotIn("sub-solar BLACK HOLE", output)
        self.assertNotIn("NVG dead", output)
        self.assertIn(f"below computed M_crit = {canonical.M_CRIT/canonical.M_sun:.6f} M_sun", output)
        self.assertEqual(output.count("falsifies this fixed-anchor Hayward hypothesis"), 3)
        self.assertIn("Alternative core-density scales and other NVG realizations", output)
        self.assertEqual(result["evidence_status"], "MODEL_DERIVED_NO_EVAPORATION_LIKELIHOOD")
        self.assertIsNone(result["observed_likelihood"])

    def test_exact_extremal_double_horizon_and_horizonless_side(self):
        radius = canonical.horizon(canonical.M_CRIT)
        self.assertEqual(radius, math.sqrt(3) * canonical.l_h)
        self.assertEqual(canonical.hawking_T(canonical.M_CRIT), 0)
        for mass in (0, canonical.M_CRIT / 2, np.nextafter(canonical.M_CRIT, 0)):
            self.assertIsNone(canonical.horizon(mass))
            self.assertEqual(canonical.hawking_T(mass), 0)

    def test_just_above_extremality_against_high_precision_root(self):
        with mp.workdps(80):
            critical = mp.mpf(canonical.M_CRIT)
            rcrit = mp.mpf(math.sqrt(3) * canonical.l_h)
            masses = (np.nextafter(canonical.M_CRIT, math.inf),
                      canonical.M_CRIT * (1 + 1e-12), canonical.M_CRIT * (1 + 1e-8),
                      canonical.M_CRIT * 1.001, canonical.M_CRIT * 1.5)
            radii = []
            for mass in masses:
                delta = (mp.mpf(float(mass)) - critical) / critical
                equation = lambda t: t**2 * (3 + 2 * t) / (2 + 6 * t + 3 * t**2) - delta
                t = mp.findroot(equation, mp.sqrt(2 * delta / 3))
                radius = canonical.horizon(mass)
                radii.append(radius)
                expected_radius = rcrit * (1 + t)
                expected_temperature = (mp.mpf(canonical.hbar * canonical.c / canonical.k_B)
                                        * t * (2 + t) / (4 * mp.pi * rcrit * (1 + t)**3))
                self.assertTrue(math.isclose(radius, float(expected_radius), rel_tol=3e-15))
                self.assertTrue(math.isclose(canonical.hawking_T(mass), float(expected_temperature), rel_tol=3e-8))
                self.assertGreater(radius, float(rcrit))
                self.assertGreater(canonical.hawking_T(mass), 0)
            self.assertEqual(radii, sorted(radii))

    def test_canonical_peak_and_large_mass_limits(self):
        peak_mass = 27 * canonical.l_h * canonical.c**2 / (16 * canonical.G)
        peak_temperature = canonical.hbar * canonical.c / (18 * math.pi * canonical.k_B * canonical.l_h)
        self.assertTrue(math.isclose(canonical.horizon(peak_mass), 3 * canonical.l_h, rel_tol=2e-14))
        self.assertTrue(math.isclose(canonical.hawking_T(peak_mass), peak_temperature, rel_tol=2e-14))
        for mass in (canonical.M_CRIT * 1e6, 1e100, np.finfo(float).max):
            radius = canonical.horizon(mass)
            self.assertTrue(math.isfinite(radius))
            self.assertTrue(math.isfinite(canonical.hawking_T(mass)))
            self.assertTrue(math.isclose(radius, 2 * canonical.G * mass / canonical.c**2, rel_tol=1e-10))
            self.assertTrue(math.isclose(canonical.hawking_T(mass), canonical.schwarzschild_T(mass), rel_tol=1e-10))

    def test_nonfinite_negative_masses_are_rejected(self):
        for function in (canonical.horizon, canonical.hawking_T, canonical.schwarzschild_T):
            for mass in (-1, math.nan, math.inf, -math.inf):
                with self.assertRaises(ValueError):
                    function(mass)
        with self.assertRaises(ValueError):
            canonical.schwarzschild_T(0)


if __name__ == "__main__":
    unittest.main()
