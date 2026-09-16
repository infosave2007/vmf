#!/usr/bin/env python3
"""Focused tests for the isovector-contact and form-factor probe.

These tests use synthetic densities and the retained R3 result file only;
they never launch a BVP solve.  The heavy live-solver path is exercised by
the producer script itself, which fails closed on any protocol failure.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

try:  # Both direct-script and package-style imports are supported.
    import nvg_isovector_formfactor_probe as probe
except ImportError:  # pragma: no cover - package import support.
    from . import nvg_isovector_formfactor_probe as probe

HERE = Path(__file__).resolve().parent
R3_PATH = HERE.parent / probe.ROOT_SCALE_SOURCE


class FormFactorAnalyticTests(unittest.TestCase):
    def test_gaussian_form_factor_and_moments(self) -> None:
        sigma = 1.7
        r = np.linspace(0.0, 12.0, 4801)
        density = np.exp(-(r**2) / (2.0 * sigma**2))
        count = probe._spherical_integral(r, density)
        ff = probe.species_form_factor(r, density, count, (0.1, 0.3, 0.6, 0.9))
        for q in (0.1, 0.3, 0.6, 0.9):
            expected = math.exp(-(q**2) * sigma**2 / 2.0)
            self.assertAlmostEqual(ff["F_of_q"][probe._number(q, 10)], expected, places=9)
        self.assertAlmostEqual(ff["moment_r2_fm2"], 3.0 * sigma**2, places=6)
        self.assertAlmostEqual(ff["rms_from_moments_fm"], sigma * math.sqrt(3.0), places=6)
        # Gaussian has no diffraction zero on the declared scan.
        self.assertIsNone(ff["first_zero_q_fm"])
        self.assertLess(ff["count_relative_difference"], 1.0e-12)

    def test_uniform_sphere_first_zero(self) -> None:
        radius = 4.2
        r = np.linspace(0.0, 10.0, 4001)
        density = np.where(r <= radius, 1.0, 0.0)
        count = probe._spherical_integral(r, density)
        ff = probe.species_form_factor(r, density, count, (0.1, 0.5, 1.0))
        first_zero = ff["first_zero_q_fm"]
        self.assertIsNotNone(first_zero)
        expected = probe.UNIFORM_SPHERE_FIRST_ZERO_X / radius
        self.assertLess(abs(first_zero - expected), 2.0 * probe.FF_ZERO_SCAN_STEP_FM)
        # Moment accuracy is limited by the discontinuous edge on the grid.
        self.assertLess(abs(ff["moment_r2_fm2"] - 0.6 * radius**2), 1.0e-2)

    def test_spherical_bessel_j0_origin(self) -> None:
        values = probe._spherical_bessel_j0(np.array([0.0, 1.0e-12, 1.0]))
        self.assertAlmostEqual(values[0], 1.0, places=12)
        self.assertAlmostEqual(values[1], 1.0, places=12)
        self.assertAlmostEqual(values[2], math.sin(1.0), places=12)


class IsovectorMetricTests(unittest.TestCase):
    def test_gaussian_isovector_integral_analytic(self) -> None:
        sigma = 1.9
        amplitude = 0.07
        r = np.linspace(0.0, 14.0, 5601)
        gaussian = np.exp(-(r**2) / (2.0 * sigma**2))
        profile = {"r_fm": r, "nn_fm3": amplitude * gaussian, "np_fm3": 0.0 * gaussian}
        metrics = probe.isovector_metrics(profile, n0_fm3=0.16)
        # integral of exp(-r^2/sigma^2) over 3D is (pi sigma^2)^(3/2)
        expected = amplitude**2 * (math.pi * sigma**2) ** 1.5 / 0.16
        self.assertAlmostEqual(metrics["I_A_dimensionless"], expected, places=9)
        # Half-density radius of the Gaussian matter profile.
        self.assertAlmostEqual(
            metrics["half_density_radius_fm"], sigma * math.sqrt(2.0 * math.log(2.0)), places=3
        )
        self.assertAlmostEqual(
            metrics["I_bulk_dimensionless"] + metrics["I_surface_dimensionless"],
            metrics["I_A_dimensionless"],
            places=12,
        )

    def test_negative_density_rejected(self) -> None:
        r = np.linspace(0.0, 5.0, 101)
        profile = {"r_fm": r, "nn_fm3": np.full_like(r, 0.1), "np_fm3": -np.full_like(r, 0.05)}
        with self.assertRaises(probe.IsovectorProbeError):
            probe.isovector_metrics(profile)


class AnchoredSlopeTests(unittest.TestCase):
    def test_anchor_nucleus_has_zero_anchored_slope(self) -> None:
        slope = probe.anchored_binding_slope(0.05, 120.0, 0.05, 120.0, 40)
        self.assertEqual(slope, 0.0)

    def test_toy_model_identity_control(self) -> None:
        check = probe.anchored_identity_toy_check()
        self.assertTrue(check["pass"], check)
        self.assertLess(check["relative_difference"], 1.0e-6)

    def test_nonpositive_reference_gradient_rejected(self) -> None:
        with self.assertRaises(probe.IsovectorProbeError):
            probe.anchored_binding_slope(0.1, 10.0, 0.1, 0.0, 40)


class NucleonOperatorTests(unittest.TestCase):
    def test_nucleon_electric_form_factor_normalization(self) -> None:
        g_ep, g_en = probe._nucleon_electric_form_factors(0.0)
        self.assertAlmostEqual(g_ep, 1.0, places=12)
        self.assertAlmostEqual(g_en, 0.0, places=12)
        g_ep2, g_en2 = probe._nucleon_electric_form_factors(1.0)
        self.assertGreater(g_ep2, 0.0)
        self.assertLess(g_ep2, 1.0)
        self.assertGreater(g_en2, 0.0)  # Galster term is positive here (mu_n<0)

    def test_nucleon_weak_electric_form_factor_normalization(self) -> None:
        g_ewp, g_ewn = probe._nucleon_weak_electric_form_factors(0.0)
        self.assertAlmostEqual(g_ewp, probe.Q_WEAK_P, places=12)
        self.assertAlmostEqual(g_ewn, probe.Q_WEAK_N, places=12)
        # At finite q the neutron weak form factor stays finite and negative:
        # it must not inherit the q->0 zero of G_E^n.
        g_ewp2, g_ewn2 = probe._nucleon_weak_electric_form_factors(1.0)
        self.assertLess(g_ewn2, 0.0)
        self.assertGreater(g_ewn2, -1.5)

    def test_charge_weak_normalization_and_apv_consistency(self) -> None:
        # Two narrow Gaussians make the point form factors near unity, so the
        # analytic q->0 limits of the convolutions are transparent.
        sigma = 0.05
        r = np.linspace(0.0, 3.0, 6001)
        shape = np.exp(-(r**2) / (2.0 * sigma**2))
        z, n_n = 8, 10
        norm_z = probe._spherical_integral(r, shape)
        norm_n = probe._spherical_integral(r, shape)
        result = probe.charge_weak_form_factors(
            r, shape / norm_n * n_n, shape / norm_z * z, n_n, z, (0.05, 0.10, 0.15)
        )
        # F_W(0)=1 with the CVC weak-charge normalization: the smallest-q
        # value must sit near one, not near the broken G_E^n-only limit.
        smallest_q = min(float(q) for q in result["F_W_of_q"])
        f_w_smallest = result["F_W_of_q"][probe._number(smallest_q, 10)]
        self.assertGreater(f_w_smallest, 0.98)
        self.assertIsNotNone(result["F_W_normalization_check"])
        # Recompute A_PV from the returned form factors: the full weak-charge
        # normalization Q_W^A/Z must be present and the value must be in ppm.
        for q in probe.APV_Q_LIST_FM:
            q_gev = q * probe.HBARC_GEV_FM
            prefactor = probe.GF_GEV_MINUS2 * q_gev**2 / (
                4.0 * math.pi * 0.0072973525643 * math.sqrt(2.0)
            )
            f_ch = result["F_ch_of_q"][probe._number(q, 10)]
            f_w = result["F_W_of_q"][probe._number(q, 10)]
            weak_total = probe.Q_WEAK_P * z + probe.Q_WEAK_N * n_n
            expected_ppm = -prefactor * weak_total * f_w / (z * f_ch) * 1.0e6
            self.assertAlmostEqual(
                result["A_PV_pw_per_million"][probe._number(q, 10)],
                expected_ppm,
                places=6,
            )
        # Weak-minus-charge radius is finite and positive-ish for N>Z.
        self.assertIsNotNone(result["weak_minus_charge_radius_fm"])

    def test_count_mismatch_is_visible_not_masked(self) -> None:
        r = np.linspace(0.0, 6.0, 1201)
        density = np.exp(-(r**2) / (2.0 * 1.5**2))
        wrong_count = 7.0  # not the integral of the density
        ff = probe.species_form_factor(r, density, wrong_count, (0.1,))
        self.assertGreater(ff["count_relative_difference"], 0.1)
        self.assertNotAlmostEqual(ff["F_of_q"][probe._number(0.1, 10)], 1.0, places=2)


class ModuleContractTests(unittest.TestCase):
    def test_evidence_boundary_constants(self) -> None:
        self.assertEqual(probe.EVIDENCE_WEIGHT, 0.0)
        self.assertEqual(probe.FAMILY, "W8.93")
        self.assertEqual(probe.BASELINE_RHO, "no_rho")
        self.assertEqual(probe.CONTACT_RHO, "covariant_contact_J32")
        self.assertEqual(tuple(probe.NUCLEI), ("Ca40", "Zr90", "Pb208"))

    def test_baseline_expected_matches_retained_r3(self) -> None:
        if not R3_PATH.exists():
            self.skipTest("retained R3 result is unavailable")
        result = json.loads(R3_PATH.read_text(encoding="utf-8"))
        family = result["calibrations"]["W8.93"]
        root_scale = family["root"]["scale"]
        self.assertEqual(root_scale, probe.ROOT_SCALE)
        for item in family["frozen_predictions"]:
            nucleus = item["nucleus"]
            expected = probe.BASELINE_EXPECTED[nucleus]
            self.assertEqual(item["model_binding_per_A_MeV"], expected["binding_per_A_MeV"])
            domain = item["terminal_control_rows"]["domain40_check"]
            self.assertEqual(
                domain["rms_neutron_radius_fm"], expected["rms_neutron_radius_fm"]
            )
            self.assertEqual(
                domain["rms_point_proton_radius_fm"],
                expected["rms_point_proton_radius_fm"],
            )
            components = domain["energy_components_MeV"]
            self.assertEqual(components["T_W"], expected["T_W_MeV"])
            self.assertEqual(components["total_E"], expected["total_E_MeV"])

    def test_analytic_validation_block_passes(self) -> None:
        block = probe.analytic_formfactor_validation()
        self.assertTrue(block["pass"], block)


if __name__ == "__main__":
    unittest.main()
