#!/usr/bin/env python3
"""Focused tests for the Sn132 discriminating-set probe.

These tests use cited-input arithmetic and synthetic verdict data only; they
never launch a BVP solve.  The heavy live-solver path is exercised by the
producer script itself, which fails closed on any protocol failure.
"""

from __future__ import annotations

import math
import unittest

try:  # Both direct-script and package-style imports are supported.
    import nvg_sn132_discriminating_set_probe as probe
    import nvg_finite_monopole as static
except ImportError:  # pragma: no cover - package import support.
    from . import nvg_sn132_discriminating_set_probe as probe
    from . import nvg_finite_monopole as static


H_MASS_U = 1.00782503223
N_MASS_U = 1.00866491595
U_MEV = 931.49410242
SN132_ATOMIC_MASS_U = 131.917823898


class ContactLinearityTests(unittest.TestCase):
    def test_t_contact_matches_declared_probes(self) -> None:
        # j is exactly linear in J: T = J - j is design-independent.
        self.assertAlmostEqual(probe.T_CONTACT_MEV, 32.0 - 19.35588155703005, places=12)
        self.assertAlmostEqual(probe.T_CONTACT_MEV, 20.0 - 7.355881557030049, places=12)

    def test_universal_design_j_roundtrip(self) -> None:
        for j in (7.355881557030049, 11.0, 12.374939437720375, 19.35588155703005):
            self.assertAlmostEqual(probe.universal_design_j(j) - probe.T_CONTACT_MEV, j, places=12)


class VerdictTests(unittest.TestCase):
    def test_supported_when_all_equal(self) -> None:
        verdict = probe.verdict_bands({"Zr90": 11.2, "Pb208": 11.2, "Ca48": 11.2, "Sn132": 11.2})
        self.assertEqual(verdict["status"], "UNIVERSAL_CONSTANT_SUPPORTED")
        self.assertEqual(verdict["max_relative_deviation"], 0.0)

    def test_supported_within_band(self) -> None:
        verdict = probe.verdict_bands({"Zr90": 11.0, "Pb208": 10.0, "Ca48": 12.0, "Sn132": 11.5})
        # mean 11.125; max |dev| = 1.125/11.125 = 0.101 < 0.15
        self.assertEqual(verdict["status"], "UNIVERSAL_CONSTANT_SUPPORTED")

    def test_partial_between_bands(self) -> None:
        verdict = probe.verdict_bands({"Zr90": 10.0, "Pb208": 10.0, "Ca48": 10.0, "Sn132": 12.4})
        # mean 10.6; max dev = 1.8/10.6 = 0.17 (0.15 < dev <= 0.35)
        self.assertEqual(verdict["status"], "PARTIAL_UNIVERSALITY")

    def test_refuted_beyond_band(self) -> None:
        verdict = probe.verdict_bands({"Zr90": 10.0, "Pb208": 10.0, "Ca48": 10.0, "Sn132": 20.0})
        self.assertEqual(verdict["status"], "UNIVERSAL_CONSTANT_REFUTED")

    def test_insufficient_tangents(self) -> None:
        verdict = probe.verdict_bands({"Zr90": 10.0})
        self.assertEqual(verdict["status"], "INSUFFICIENT_TANGENTS")

    def test_nonfinite_entries_are_ignored(self) -> None:
        verdict = probe.verdict_bands({"Zr90": 11.0, "Pb208": float("nan"), "Ca48": 11.2})
        self.assertEqual(verdict["status"], "UNIVERSAL_CONSTANT_SUPPORTED")
        self.assertNotIn("Pb208", verdict["j_values_MeV"])

    def test_registered_bands_are_recorded(self) -> None:
        verdict = probe.verdict_bands({"Zr90": 10.0, "Pb208": 10.5})
        self.assertEqual(verdict["registered_bands"]["supported_max_dev"], probe.VERDICT_BAND_SUPPORTED)
        self.assertEqual(verdict["registered_bands"]["refuted_min_dev"], probe.VERDICT_BAND_REFUTED)


class Sn132InputTests(unittest.TestCase):
    def test_sn132_target_arithmetic(self) -> None:
        import nvg_finite_static_bridge as bridge

        target = probe.sn132_binding_target()
        expected = probe.SN132_B_ATOM_PER_A_MEV - bridge.electron_correction_MeV(50) / 132.0
        self.assertAlmostEqual(target["B_nuc_target_per_A_MeV"], expected, places=12)
        self.assertEqual(target["A"], 132)
        self.assertEqual(target["Z"], 50)
        self.assertEqual(target["N"], 82)

    def test_sn132_binding_consistent_with_atomic_mass(self) -> None:
        # AME2020 convention: B_atom = [Z M(1H) + N M(n) - M_atom] * u.
        b_atom_total = (
            50.0 * H_MASS_U + 82.0 * N_MASS_U - SN132_ATOMIC_MASS_U
        ) * U_MEV
        per_a = b_atom_total / 132.0
        self.assertLess(abs(per_a - probe.SN132_B_ATOM_PER_A_MEV), 2.0e-4)

    def test_delta_values(self) -> None:
        self.assertAlmostEqual(probe.delta2_of(probe.SN132_NUCLEUS_ROW), (32.0 / 132.0) ** 2, places=15)
        self.assertAlmostEqual(probe.delta2_of({"A": 48, "Z": 20, "N": 28}), (8.0 / 48.0) ** 2, places=15)

    def test_invalid_nucleus_row_rejected(self) -> None:
        with self.assertRaises(probe.Sn132ProbeError):
            probe.delta2_of({"A": 10, "Z": 5, "N": 4})  # A != N + Z

    def test_sn132_cited_sources_present(self) -> None:
        self.assertIn("8354.8726", probe.SN132_B_ATOM_SOURCE)
        self.assertIn("39.7 s", probe.SN132_B_ATOM_SOURCE)
        self.assertIn("Angeli", probe.SN132_R_CH_SOURCE)
        self.assertEqual(probe.SN132_ROLE, "pre_registered_heldout_descriptive_comparison_only")

    def test_sn132_patched_inputs_add_rows(self) -> None:
        inputs = probe.sn132_patched_inputs()
        self.assertIn("Sn132", inputs["binding"]["values"])
        self.assertIn("Sn132", inputs["charge_radii"]["values"])
        self.assertEqual(inputs["binding"]["values"]["Sn132"]["N"], 82)
        self.assertEqual(inputs["charge_radii"]["values"]["Sn132"]["R_ch_fm"], 4.7093)
        # The production file on disk is untouched.
        production = probe.bridge._load_inputs()
        self.assertNotIn("Sn132", production["binding"]["values"])


class HeldoutContextTests(unittest.TestCase):
    def test_patch_adds_and_restores_both_rows(self) -> None:
        self.assertNotIn("Sn132", static.BRIDGE_NUCLEI)
        with probe._heldout_context():
            self.assertIn("Ca48", static.BRIDGE_NUCLEI)
            self.assertIn("Sn132", static.BRIDGE_NUCLEI)
            self.assertEqual(static.BRIDGE_NUCLEI["Sn132"], {"A": 132, "Z": 50, "N": 82})
            self.assertEqual(static.BRIDGE_NUCLEI["Ca48"], {"A": 48, "Z": 20, "N": 28})
        self.assertNotIn("Ca48", static.BRIDGE_NUCLEI)
        self.assertNotIn("Sn132", static.BRIDGE_NUCLEI)

    def test_patch_restores_on_exception(self) -> None:
        with self.assertRaises(RuntimeError):
            with probe._heldout_context():
                raise RuntimeError("boom")
        self.assertNotIn("Ca48", static.BRIDGE_NUCLEI)
        self.assertNotIn("Sn132", static.BRIDGE_NUCLEI)

    def test_patch_refuses_double_entry(self) -> None:
        with probe._heldout_context():
            with self.assertRaises(probe.Sn132ProbeError):
                with probe._heldout_context():
                    pass
        self.assertNotIn("Sn132", static.BRIDGE_NUCLEI)


class ModuleContractTests(unittest.TestCase):
    def test_constants_and_schema(self) -> None:
        self.assertEqual(probe.SCHEMA, "nvg_sn132_discriminating_set_probe.v1")
        self.assertEqual(probe.EVIDENCE_WEIGHT, 0.0)
        self.assertEqual(probe.VERDICT_BAND_SUPPORTED, 0.15)
        self.assertEqual(probe.VERDICT_BAND_REFUTED, 0.35)
        self.assertEqual(probe.HELDOUT, ("Zr90", "Pb208", "Ca48", "Sn132"))

    def test_pre_registration_documented(self) -> None:
        # The verdict bands must be registered in the producer file itself
        # (pre-registration), not supplied at runtime.
        source = probe.__doc__ or ""
        self.assertIn("SUPPORTED", source)
        self.assertIn("REFUTED", source)
        self.assertIn("pre-registered", source.lower())


if __name__ == "__main__":
    unittest.main()
