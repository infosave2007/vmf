#!/usr/bin/env python3
"""Focused tests for the j identification audit.

Cited-anchor arithmetic, contact-convention identities, band logic and the
fail-closed artifact gates are tested with pure algebra and synthetic data.
The full ``calculate()`` path additionally requires the retained
discriminating-set artifact (Lunacy is not tracked), so those tests skip
automatically where the artifact is absent; the producer itself fails closed.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

try:  # Both direct-script and package-style imports are supported.
    import nvg_j_identification_audit as probe
    import nvg_sn132_discriminating_set_probe as sn
except ImportError:  # pragma: no cover - package import support.
    from . import nvg_j_identification_audit as probe
    from . import nvg_sn132_discriminating_set_probe as sn

ARTIFACT = probe.ROOT / probe.SN132_RESULT_REL
J_BAR = 11.13601607117819
HBARC = 197.3269804


class CitedAnchorTests(unittest.TestCase):
    def test_cited_anchor_constants(self) -> None:
        # Textbook SEMF, Wang et al. PRC 91, 044308 (2015), upper LDM fits.
        self.assertEqual(probe.A_SYM_TEXTBOOK_MEV, 23.2)
        self.assertEqual(probe.A_SYM_WANG_2015_MEV, 22.90)
        self.assertEqual(probe.A_SYM_WANG_2015_SIGMA_MEV, 0.15)
        self.assertEqual(probe.A_SYM_FIT_UPPER_MEV, 23.7)
        # PDG rho(770) and the RMF meson mass used in FSUGold/IU-FSU tables.
        self.assertEqual(probe.M_RHO_MEV, 775.26)
        self.assertEqual(probe.M_RHO_RMF_MEV, 763.0)
        self.assertEqual(probe.RMF_G_RHO_RANGE, (4.1, 6.8))

    def test_registered_bands(self) -> None:
        self.assertEqual(probe.T_KINETIC_FERMI_GAS_BAND_MEV, (11.0, 13.0))
        self.assertEqual(probe.J_LIQUID_DROP_REL_TOL, 0.05)
        self.assertEqual(probe.J_INTERACTION_REL_TOL, 0.10)
        self.assertEqual(probe.J_BAR_SANITY_BAND_MEV, (10.0, 13.0))
        self.assertEqual(len(probe.VERDICT_KEYS), 4)


class ContactConventionTests(unittest.TestCase):
    def test_model_saturation_state_reproduces_t_contact(self) -> None:
        state = probe.model_saturation_state()
        # Exact identity: the decoded kinetic part IS the contact's T.
        self.assertAlmostEqual(state["T_kinetic_MeV"], sn.T_CONTACT_MEV, places=12)
        self.assertAlmostEqual(state["y_effective_mass_ratio"], 0.93, places=12)
        self.assertAlmostEqual(state["k_F_MeV"], 263.041, places=2)
        self.assertGreater(state["e_F_MeV"], 900.0)
        self.assertAlmostEqual(state["hbarc_MeV_fm"], HBARC, places=9)

    def test_free_fermi_gas_reference(self) -> None:
        ref = probe.free_fermi_gas_reference(HBARC)
        self.assertAlmostEqual(ref["k_F_MeV"], 263.041, places=2)
        self.assertAlmostEqual(ref["T_nonrel_MeV"], 12.281, places=2)
        self.assertAlmostEqual(ref["T_relativistic_MeV"], 11.826, places=2)
        # Relativistic kinetic symmetry energy is below the nonrelativistic one.
        self.assertLess(ref["T_relativistic_MeV"], ref["T_nonrel_MeV"])

    def test_liquid_drop_decomposition(self) -> None:
        ld = probe.liquid_drop_comparison(J_BAR, sn.T_CONTACT_MEV, 12.281)
        self.assertAlmostEqual(ld["design_J_MeV"], 23.780135, places=5)
        self.assertAlmostEqual(ld["kinetic_share"] + ld["interaction_share"], 1.0, places=12)
        self.assertAlmostEqual(
            ld["kinetic_share"], sn.T_CONTACT_MEV / (sn.T_CONTACT_MEV + J_BAR), places=12
        )
        by_anchor = {row["anchor"]: row for row in ld["anchors"]}
        self.assertEqual(len(by_anchor), 3)
        textbook = by_anchor["textbook_semf_23.2"]
        self.assertAlmostEqual(
            textbook["design_J_over_a_sym_minus_1"],
            (sn.T_CONTACT_MEV + J_BAR) / 23.2 - 1.0,
            places=12,
        )
        self.assertAlmostEqual(
            textbook["interaction_piece_MeV"], 23.2 - 12.281, places=12
        )

    def test_rho_channel_algebra(self) -> None:
        rho = probe.rho_channel_identification(J_BAR, HBARC)
        # C_rho = 8 j / n0 in MeV fm^3.
        self.assertAlmostEqual(rho["C_rho_MeV_fm3"], 8.0 * J_BAR / 0.16, places=9)
        # g_rho(tau/2) = sqrt(C_rho_nat * m_rho^2); tau convention halves it.
        expected = math.sqrt(rho["C_rho_natural"] * probe.M_RHO_MEV**2)
        self.assertAlmostEqual(rho["g_rho_tau_half_convention"], expected, places=9)
        self.assertAlmostEqual(
            rho["g_rho_tau_convention"], rho["g_rho_tau_half_convention"] / 2.0, places=12
        )
        # A lighter rho mass lowers the implied coupling at fixed j.
        self.assertLess(
            rho["g_rho_tau_half_m_rho_763"], rho["g_rho_tau_half_convention"]
        )


class BandLogicTests(unittest.TestCase):
    def _real_inputs(self):
        state = probe.model_saturation_state()
        free = probe.free_fermi_gas_reference(HBARC)
        ld = probe.liquid_drop_comparison(J_BAR, sn.T_CONTACT_MEV, free["T_nonrel_MeV"])
        rho = probe.rho_channel_identification(J_BAR, HBARC)
        return state, free, ld, rho

    def test_all_bands_true_for_the_measured_numbers(self) -> None:
        bands = probe.identification_bands(*self._real_inputs())
        for key in probe.VERDICT_KEYS:
            self.assertTrue(bands[key], key)
        self.assertAlmostEqual(
            bands["band_values"]["g_rho_tau_half"], 6.60, places=2
        )

    def test_bands_fail_when_j_is_too_large(self) -> None:
        state, free, _ld, _rho = self._real_inputs()
        ld = probe.liquid_drop_comparison(30.0, sn.T_CONTACT_MEV, free["T_nonrel_MeV"])
        rho = probe.rho_channel_identification(30.0, HBARC)
        bands = probe.identification_bands(state, free, ld, rho)
        self.assertFalse(bands["j_matches_interaction_piece"])
        self.assertFalse(bands["g_rho_in_published_rmf_range"])
        self.assertTrue(bands["T_kinetic_in_fermi_gas_band"])  # unchanged

    def test_kinetic_band_fails_outside_range(self) -> None:
        state, free, ld, rho = self._real_inputs()
        state = dict(state, T_kinetic_MeV=20.0)
        bands = probe.identification_bands(state, free, ld, rho)
        self.assertFalse(bands["T_kinetic_in_fermi_gas_band"])

    def test_verdict_complete_and_incomplete(self) -> None:
        complete = {key: True for key in probe.VERDICT_KEYS}
        verdict = probe.identification_verdict(complete)
        self.assertEqual(verdict["status"], "IDENTIFIED_AS_SYMMETRY_ENERGY_INTERACTION_PART")
        self.assertEqual(verdict["failed_bands"], [])
        broken = dict(complete, g_rho_in_published_rmf_range=False)
        verdict = probe.identification_verdict(broken)
        self.assertEqual(verdict["status"], "IDENTIFICATION_INCOMPLETE")
        self.assertEqual(verdict["failed_bands"], ["g_rho_in_published_rmf_range"])
        self.assertIn("post-hoc", verdict["post_hoc_notice"])


@unittest.skipUnless(ARTIFACT.is_file(), "retained discriminating-set artifact not present")
class CalculateTests(unittest.TestCase):
    def test_calculate_identifies_the_constant(self) -> None:
        result = probe.calculate()
        self.assertEqual(result["schema"], "nvg_j_identification_audit.v1")
        self.assertEqual(result["evidence_weight"], 0.0)
        conv = result["contact_convention"]
        self.assertAlmostEqual(conv["j_bar_MeV"], J_BAR, places=9)
        self.assertAlmostEqual(conv["J_design_MeV"], 23.780135, places=5)
        self.assertTrue(conv["j_exactly_linear_in_J"])
        self.assertEqual(
            result["identification_verdict"]["status"],
            "IDENTIFIED_AS_SYMMETRY_ENERGY_INTERACTION_PART",
        )
        self.assertFalse(result["provenance"]["bvp_solved"])
        self.assertFalse(result["provenance"]["production_inputs_patched"])
        self.assertFalse(result["provenance"]["coefficient_added_to_theory"])
        self.assertTrue(result["limits"])

    def test_calculate_fails_closed_on_wrong_verdict(self) -> None:
        saved = probe.SN132_RESULT_REL
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"verdict": {"status": "PARTIAL_UNIVERSALITY"}}), "utf-8")
            probe.SN132_RESULT_REL = bad
            try:
                with self.assertRaises(probe.JIdentificationError):
                    probe.calculate()
            finally:
                probe.SN132_RESULT_REL = saved

    def test_calculate_fails_closed_on_t_contact_drift(self) -> None:
        saved = probe.SN132_RESULT_REL
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "drift.json"
            payload = {
                "verdict": {"status": "UNIVERSAL_CONSTANT_SUPPORTED"},
                "universal_j_demonstration": {
                    "j_bar_MeV": J_BAR,
                    "t_contact_MeV": sn.T_CONTACT_MEV + 1.0e-6,
                },
            }
            bad.write_text(json.dumps(payload), "utf-8")
            probe.SN132_RESULT_REL = bad
            try:
                with self.assertRaises(probe.JIdentificationError):
                    probe.calculate()
            finally:
                probe.SN132_RESULT_REL = saved

    def test_cli_writes_strict_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            buffer = StringIO()
            with redirect_stdout(buffer):
                code = probe.main(["--output", str(out)])
            self.assertEqual(code, 0)
            written = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(written["schema"], "nvg_j_identification_audit.v1")
            stdout_payload = json.loads(buffer.getvalue())
            self.assertEqual(
                stdout_payload["identification_verdict"]["status"],
                "IDENTIFIED_AS_SYMMETRY_ENERGY_INTERACTION_PART",
            )


if __name__ == "__main__":
    unittest.main()
