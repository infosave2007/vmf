#!/usr/bin/env python3
"""Focused tests for the Ca48 held-out and quantum-surface probe.

These tests use synthetic densities and cited-input arithmetic only; they
never launch a BVP solve.  The heavy live-solver path is exercised by the
producer script itself, which fails closed on any protocol failure.
"""

from __future__ import annotations

import json
import math
import unittest

import numpy as np

try:  # Both direct-script and package-style imports are supported.
    import nvg_ca48_quantum_surface_probe as probe
    import nvg_finite_monopole as static
    import nvg_isovector_formfactor_probe as ivp
except ImportError:  # pragma: no cover - package import support.
    from . import nvg_ca48_quantum_surface_probe as probe
    from . import nvg_finite_monopole as static
    from . import nvg_isovector_formfactor_probe as ivp


H_MASS_U = 1.00782503223
N_MASS_U = 1.00866491595
U_MEV = 931.49410242
CA48_ATOMIC_MASS_U = 47.952522654


class AnalyticGradientTests(unittest.TestCase):
    def test_gaussian_grad_squared_over_n(self) -> None:
        validation = probe.analytic_validation()
        self.assertLess(validation["grad_squared_over_n_relative_error"], 1.0e-5)

    def test_gaussian_etf2_energy(self) -> None:
        validation = probe.analytic_validation()
        self.assertLess(validation["etf2_energy_relative_error"], 1.0e-5)

    def test_gaussian_i_grad(self) -> None:
        validation = probe.analytic_validation()
        self.assertLess(validation["i_grad_relative_error"], 1.0e-5)

    def test_masked_leq_full(self) -> None:
        validation = probe.analytic_validation()
        self.assertTrue(validation["masked_leq_full"])

    def test_uniform_density_has_no_gradient_energy(self) -> None:
        r = np.linspace(0.0, 10.0, 4001)
        n = np.full_like(r, 0.08)
        y = np.ones_like(r)
        block = probe.quantum_surface_energy(r, n, y, 939.0, 197.3269804, probe.ETF2_COEFF)
        self.assertAlmostEqual(block["energy_full_MeV"], 0.0, places=12)
        for entry in block["masked_by_eps"].values():
            self.assertAlmostEqual(entry["energy_MeV"], 0.0, places=12)

    def test_vw_is_exactly_linear_multiple_of_etf2(self) -> None:
        r = np.linspace(0.0, 10.0, 4001)
        n = 0.08 * np.exp(-(r**2) / (2.0 * 1.7**2))
        y = np.ones_like(r)
        etf2 = probe.quantum_surface_energy(r, n, y, 939.0, 197.3269804, probe.ETF2_COEFF)
        vw = probe.quantum_surface_energy(r, n, y, 939.0, 197.3269804, probe.VW_COEFF)
        self.assertAlmostEqual(probe.VW_OVER_ETF2, 4.5, places=12)
        self.assertAlmostEqual(
            vw["energy_full_MeV"] / etf2["energy_full_MeV"], 4.5, places=12
        )
        for threshold in probe.EPS_VALIDITY_THRESHOLDS:
            key = str(probe._number(threshold, 10))
            self.assertAlmostEqual(
                vw["masked_by_eps"][key]["energy_MeV"] / etf2["masked_by_eps"][key]["energy_MeV"],
                4.5,
                places=12,
            )

    def test_local_fermi_momentum(self) -> None:
        n = np.array([0.08, 0.16])
        expected = (3.0 * math.pi**2 * n) ** (1.0 / 3.0)
        np.testing.assert_allclose(probe.local_fermi_momentum(n), expected, rtol=1.0e-14)

    def test_wkb_eps_small_in_flat_region_large_at_edge(self) -> None:
        r = np.linspace(0.0, 8.0, 3201)
        edge = 5.0
        # Smooth the step a little so the gradient is finite on the grid.
        n = 0.08 / (1.0 + np.exp((r - edge) / 0.05))
        eps = probe.wkb_validity_parameter(r, n)
        interior = r <= 4.5
        self.assertLess(float(np.max(eps[interior])), 1.0)  # flat interior
        self.assertGreater(float(np.max(eps)), 1.0e3)  # far tail

    def test_masked_integral_with_full_mask_equals_full(self) -> None:
        r = np.linspace(0.0, 8.0, 1601)
        f = np.exp(-(r**2) / (2.0 * 1.5**2))
        full = ivp._spherical_integral(r, f)
        masked = probe._masked_spherical_integral(r, f, np.ones_like(f, dtype=bool))
        self.assertAlmostEqual(full, masked, places=9)


class Ca48InputTests(unittest.TestCase):
    def test_ca48_target_arithmetic(self) -> None:
        target = probe.ca48_binding_target()
        # Electron correction comes from the bridge convention.
        import nvg_finite_static_bridge as bridge

        expected = probe.CA48_B_ATOM_PER_A_MEV - bridge.electron_correction_MeV(20) / 48.0
        self.assertAlmostEqual(target["B_nuc_target_per_A_MeV"], expected, places=12)
        self.assertEqual(target["A"], 48)
        self.assertEqual(target["Z"], 20)
        self.assertEqual(target["N"], 28)
        self.assertEqual(target["A"], target["N"] + target["Z"])

    def test_ca48_binding_consistent_with_atomic_mass(self) -> None:
        # AME2020 convention: B_atom = [Z M(1H) + N M(n) - M_atom] * u.
        b_atom_total = (
            20.0 * H_MASS_U + 28.0 * N_MASS_U - CA48_ATOMIC_MASS_U
        ) * U_MEV
        per_a = b_atom_total / 48.0
        self.assertLess(abs(per_a - probe.CA48_B_ATOM_PER_A_MEV), 2.0e-4)

    def test_ca48_cited_sources_present(self) -> None:
        for source in (
            probe.CA48_B_ATOM_SOURCE,
            probe.CA48_R_CH_SOURCE,
            probe.CA48_SKIN_SOURCE,
        ):
            self.assertIsInstance(source, str)
            self.assertGreater(len(source), 20)
        self.assertEqual(probe.CA48_ROLE, "pre_registered_heldout_descriptive_comparison_only")


class PatchContextTests(unittest.TestCase):
    def test_patch_adds_and_restores(self) -> None:
        self.assertNotIn(probe.CA48, static.BRIDGE_NUCLEI)
        with probe._ca48_nuclei_context():
            self.assertIn(probe.CA48, static.BRIDGE_NUCLEI)
            self.assertEqual(static.BRIDGE_NUCLEI[probe.CA48], {"A": 48, "Z": 20, "N": 28})
        self.assertNotIn(probe.CA48, static.BRIDGE_NUCLEI)

    def test_patch_restores_on_exception(self) -> None:
        with self.assertRaises(RuntimeError):
            with probe._ca48_nuclei_context():
                raise RuntimeError("boom")
        self.assertNotIn(probe.CA48, static.BRIDGE_NUCLEI)

    def test_patch_refuses_double_entry(self) -> None:
        with probe._ca48_nuclei_context():
            with self.assertRaises(probe.Ca48ProbeError):
                with probe._ca48_nuclei_context():
                    pass
        self.assertNotIn(probe.CA48, static.BRIDGE_NUCLEI)


class AnchoredOperatorTests(unittest.TestCase):
    def test_reference_nucleus_gets_exactly_zero(self) -> None:
        shift = probe.anchored_operator_slope(12.0, 300.0, 12.0, 300.0, 208)
        self.assertEqual(shift, 0.0)

    def test_operator_proportional_to_t_w_is_invisible(self) -> None:
        # An operator whose energy scales exactly like T_W produces no anchored
        # shift: the anchor subtraction removes the T-proportional part.
        for t_ref, t_a in ((117.7, 356.0), (117.7, 202.6)):
            shift = probe.anchored_operator_slope(0.5 * t_a, t_a, 0.5 * t_ref, t_ref, 208)
            self.assertAlmostEqual(shift, 0.0, places=12)

    def test_anchored_shift_sign_and_magnitude(self) -> None:
        # Larger-than-T-scaled operator energy lowers the anchored binding.
        shift = probe.anchored_operator_slope(100.0, 356.0, 20.0, 117.7, 208)
        expected = -(100.0 - (356.0 / 117.7) * 20.0) / 208.0
        self.assertAlmostEqual(shift, expected, places=12)
        self.assertLess(shift, 0.0)


class ModuleContractTests(unittest.TestCase):
    def test_constants_and_schema(self) -> None:
        self.assertEqual(probe.SCHEMA, "nvg_ca48_quantum_surface_probe.v1")
        self.assertEqual(probe.EVIDENCE_WEIGHT, 0.0)
        self.assertEqual(probe.ETF2_COEFF, 1.0 / 36.0)
        self.assertEqual(probe.VW_COEFF, 1.0 / 8.0)
        self.assertEqual(probe.CA48_NUCLEUS_ROW, {"A": 48, "Z": 20, "N": 28})
        self.assertEqual(probe.ROOT_SCALE, ivp.ROOT_SCALE)
        self.assertEqual(probe.FAMILY, "W8.93")

    def test_gradient_isovector_half_density_split_adds_up(self) -> None:
        r = np.linspace(0.0, 10.0, 4001)
        nn = 0.09 * np.exp(-(r**2) / (2.0 * 2.4**2))
        np_ = 0.07 * np.exp(-(r**2) / (2.0 * 2.3**2))
        grad = probe.gradient_isovector_integrals(r, nn, np_)
        total = float(grad["I_grad_full_fm_minus2"])
        bulk = float(grad["I_grad_bulk_inside_half_fm_minus2"])
        surface = float(grad["I_grad_surface_outside_half_fm_minus2"])
        self.assertGreater(total, 0.0)
        self.assertAlmostEqual(bulk + surface, total, places=9)
        masked = grad["masked_by_eps"]["1.0"]["I_grad_fm_minus2"]
        self.assertLessEqual(float(masked), total + 1.0e-12)


if __name__ == "__main__":
    unittest.main()
