"""Tests for the PREX-II/CREX external-consistency probe.

Fast tests verify the pre-registered published bands, the kinematics
derivation, the verdict logic, the A_PV reconstruction identity against the
form-factor operator stack (synthetic profiles, no BVP solve) and the
fail-closed retained-artifact controls.  Artifact tests verify the retained
payload and skip wherever the untracked Lunacy evidence tree is absent.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_isovector_formfactor_probe as ivp
import nvg_isovector_transferability_probe as tp
import nvg_prex_crex_external_consistency_probe as pc

TRANSFER_RESULT = ROOT / pc.TRANSFER_RESULT_REL
FORMFACTOR_RESULT = ROOT / pc.FORMFACTOR_RESULT_REL
CA48_RESULT = ROOT / pc.CA48_RESULT_REL
RESULT = ROOT / Path(
    "Lunacy/runs/prex-crex-consistency-2026-09-17/prex_crex_external_consistency_result.json"
)


def _load_result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


class BandConstantsTests(unittest.TestCase):
    def test_published_skin_constants(self) -> None:
        self.assertEqual(pc.PREX2["skin_fm"], 0.283)
        self.assertEqual(pc.PREX2["skin_sigma_fm"], 0.071)
        self.assertEqual(pc.CREX["skin_fm"], 0.121)
        self.assertEqual(pc.CREX["skin_sigma_exp_fm"], 0.026)
        self.assertEqual(pc.CREX["skin_sigma_model_fm"], 0.024)

    def test_published_citations_are_declared(self) -> None:
        for published in (pc.PREX2, pc.CREX):
            self.assertIn("Phys. Rev. Lett.", published["citation"])
            self.assertIn("arXiv:", published["citation"])
        self.assertIn("126, 172502", pc.PREX2["citation"])
        self.assertIn("129, 042501", pc.CREX["citation"])

    def test_published_weak_and_apv_values(self) -> None:
        self.assertEqual(pc.PREX2["F_W"], 0.368)
        self.assertEqual(pc.PREX2["F_W_sigma"], 0.013)
        self.assertEqual(pc.PREX2["A_PV_ppb"], 550.0)
        self.assertEqual(pc.CREX["F_W"], 0.1304)
        self.assertEqual(pc.CREX["A_PV_ppb"], 2668.0)

    def test_kinematics_derivation(self) -> None:
        self.assertAlmostEqual(
            pc.Q_PREX2_FM, (0.00616 ** 0.5) / ivp.HBARC_GEV_FM, places=15
        )
        self.assertEqual(pc.Q_CREX_FM, 0.8733)

    def test_pre_registered_bands(self) -> None:
        self.assertEqual(pc.SKIN_SIGMA_LEVEL, 2.0)
        # The empty-pairs edge case is pre-registered as the excluded verdict.
        self.assertEqual(pc.verdict({}), "SKINS_EXCLUDED_BOTH_BRANCHES")

    def test_probe_constants_tie_to_prior_probes(self) -> None:
        self.assertEqual(pc.J_BAR_MEV, tp.J_BAR_MEV)
        self.assertEqual(pc.J_DESIGN_IDENTIFIED, 23.78013451414814)
        self.assertEqual(pc.S_RHO_REANCHORED, 0.2267063450812736)
        self.assertEqual(pc.BRANCHES, ("no_rho", "identified_j"))
        self.assertEqual(pc.SKIN_NUCLEI, ("Pb208", "Ca48"))
        self.assertEqual(pc.PREX2["nucleus"], "Pb208")
        self.assertEqual(pc.CREX["nucleus"], "Ca48")


class VerdictLogicTests(unittest.TestCase):
    def test_skin_consistency_central_value(self) -> None:
        block = pc.skin_consistency(0.283, 0.283, 0.071)
        self.assertTrue(block["consistent_at_2sigma"])
        self.assertEqual(block["band"], "CONSISTENT_2SIGMA")
        self.assertAlmostEqual(block["n_sigma"], 0.0, places=12)

    def test_skin_consistency_boundary_is_inclusive(self) -> None:
        block = pc.skin_consistency(0.283 + 2.0 * 0.071, 0.283, 0.071)
        self.assertTrue(block["consistent_at_2sigma"])
        block = pc.skin_consistency(0.283 - 2.0 * 0.071, 0.283, 0.071)
        self.assertTrue(block["consistent_at_2sigma"])

    def test_skin_consistency_exclusion(self) -> None:
        # Identified-j branch Pb208 expectation: skin = -0.0256 fm.
        block = pc.skin_consistency(-0.0256, 0.283, 0.071)
        self.assertFalse(block["consistent_at_2sigma"])
        self.assertEqual(block["band"], "EXCLUDED_2SIGMA")
        self.assertAlmostEqual(block["delta_fm"], -0.3086, places=4)
        self.assertAlmostEqual(block["n_sigma"], -0.3086 / 0.071, places=6)

    def test_verdict_any_consistent(self) -> None:
        pairs = {
            "PREX2_Pb208:no_rho": pc.skin_consistency(-0.08, 0.283, 0.071),
            "PREX2_Pb208:identified_j": pc.skin_consistency(-0.03, 0.283, 0.071),
            "CREX_Ca48:no_rho": pc.skin_consistency(0.16, 0.121, 0.0354),
            "CREX_Ca48:identified_j": pc.skin_consistency(0.12, 0.121, 0.0354),
        }
        self.assertEqual(pc.verdict(pairs), "SKINS_CONSISTENT_SOME_BRANCH")

    def test_verdict_all_excluded(self) -> None:
        pairs = {
            "PREX2_Pb208:no_rho": pc.skin_consistency(-0.08, 0.283, 0.071),
            "CREX_Ca48:identified_j": pc.skin_consistency(0.02, 0.121, 0.0354),
        }
        self.assertEqual(pc.verdict(pairs), "SKINS_EXCLUDED_BOTH_BRANCHES")


class ApvReconstructionTests(unittest.TestCase):
    """The A_PV reconstruction must be identical to the ivp operator stack."""

    @staticmethod
    def _synthetic_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.linspace(0.0, 12.0, 481)
        nn = np.exp(-((r / 5.4) ** 2))
        pp = 0.9 * np.exp(-((r / 5.2) ** 2))
        return r, nn, pp

    def test_apv_reconstruction_matches_operator_stack(self) -> None:
        r, nn, pp = self._synthetic_profile()
        z, n_n = 82, 126
        ff = ivp.charge_weak_form_factors(r, nn, pp, n_n, z, (pc.Q_CREX_FM,))
        q_probe = 0.397  # a member of ivp's own APV kinematics list
        ff_probe = ivp.charge_weak_form_factors(r, nn, pp, n_n, z, (q_probe,))

        def lookup(table: dict, q: float) -> float:
            for key, value in table.items():
                if abs(float(key) - q) <= 1.0e-9:
                    return float(value)
            raise AssertionError(f"missing q={q}")

        f_ch = lookup(ff_probe["F_ch_of_q"], q_probe)
        f_w = lookup(ff_probe["F_W_of_q"], q_probe)
        reconstructed_ppb = pc._apv_ppb(q_probe, f_ch, f_w, z, n_n)
        published_ppm = lookup(ff_probe["A_PV_pw_per_million"], q_probe)
        self.assertAlmostEqual(
            reconstructed_ppb, published_ppm * 1000.0, delta=1.0e-9 * max(1.0, abs(published_ppm * 1000.0))
        )
        # The CREX-kinematics reconstruction must also be positive and finite.
        f_ch_crex = lookup(ff["F_ch_of_q"], pc.Q_CREX_FM)
        f_w_crex = lookup(ff["F_W_of_q"], pc.Q_CREX_FM)
        apv_crex = pc._apv_ppb(pc.Q_CREX_FM, f_ch_crex, f_w_crex, z, n_n)
        self.assertTrue(np.isfinite(apv_crex))
        self.assertGreater(apv_crex, 0.0)

    def test_apv_prefactor_scale(self) -> None:
        # PREX-II-like kinematics: the plane-wave asymmetry for Pb208-like
        # charges is of the measured order (hundreds of ppb), not wild.
        r, nn, pp = self._synthetic_profile()
        ff = ivp.charge_weak_form_factors(r, nn, pp, 126, 82, (pc.Q_PREX2_FM,))
        f_ch = next(iter(ff["F_ch_of_q"].values()))
        f_w = next(iter(ff["F_W_of_q"].values()))
        apv = pc._apv_ppb(pc.Q_PREX2_FM, f_ch, f_w, 82, 126)
        self.assertGreater(apv, 100.0)
        self.assertLess(apv, 2000.0)

    def test_published_apv_sigma_quadrature(self) -> None:
        self.assertAlmostEqual(
            pc._apv_published_sigma(16.0, 8.0), (16.0 ** 2 + 8.0 ** 2) ** 0.5, places=12
        )
        self.assertAlmostEqual(
            pc._apv_published_sigma(106.0, 40.0), 113.29607230614837, places=9
        )


class RetainedControlTests(unittest.TestCase):
    @unittest.skipUnless(TRANSFER_RESULT.is_file(), "retained transferability artifact not present (untracked)")
    def test_transferability_control_loads(self) -> None:
        control = pc._load_transferability_control()
        self.assertAlmostEqual(control["j_bar_MeV"], pc.J_BAR_MEV, places=12)
        self.assertAlmostEqual(control["s_rho_reanchored"], pc.S_RHO_REANCHORED, places=12)

    @unittest.skipUnless(TRANSFER_RESULT.is_file(), "retained transferability artifact not present (untracked)")
    def test_transferability_control_fails_closed_on_tampering(self) -> None:
        payload = json.loads(TRANSFER_RESULT.read_text(encoding="utf-8"))
        payload["part_a_reanchored_finite_branch"]["j_bar_MeV"] = 20.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            original = pc.TRANSFER_RESULT_REL
            try:
                pc.TRANSFER_RESULT_REL = Path(tmp) / "tampered.json"
                with self.assertRaises(pc.PrexCrexError):
                    pc._load_transferability_control()
            finally:
                pc.TRANSFER_RESULT_REL = original

    @unittest.skipUnless(FORMFACTOR_RESULT.is_file(), "retained form-factor artifact not present (untracked)")
    def test_formfactor_control_loads(self) -> None:
        control = pc._load_formfactor_control()
        self.assertAlmostEqual(
            control["Pb208_point_skin_fm"],
            pc.NO_RHO_CONTROL_RADII["Pb208"]["rms_neutron_radius_fm"]
            - pc.NO_RHO_CONTROL_RADII["Pb208"]["rms_point_proton_radius_fm"],
            places=12,
        )

    @unittest.skipUnless(CA48_RESULT.is_file(), "retained Ca48 artifact not present (untracked)")
    def test_ca48_control_loads(self) -> None:
        control = pc._load_ca48_control()
        self.assertAlmostEqual(
            control["Ca48_point_skin_fm"],
            pc.NO_RHO_CONTROL_RADII["Ca48"]["rms_neutron_radius_fm"]
            - pc.NO_RHO_CONTROL_RADII["Ca48"]["rms_point_proton_radius_fm"],
            places=12,
        )


@unittest.skipUnless(RESULT.is_file(), "retained PREX/CREX artifact not present (untracked)")
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = _load_result()

    def test_schema_status_and_evidence_weight(self) -> None:
        self.assertEqual(self.payload["schema"], pc.SCHEMA)
        self.assertEqual(self.payload["status"], pc.STATUS)
        self.assertEqual(self.payload["evidence_weight"], 0.0)

    def test_verdict_matches_pairs(self) -> None:
        payload = self.payload
        pairs = {}
        for experiment, block in payload["skin_comparison"].items():
            for branch in pc.BRANCHES:
                pairs[f"{experiment}:{branch}"] = block[branch]
        self.assertEqual(payload["verdict"], pc.verdict(pairs))
        self.assertIn(
            payload["verdict"],
            ("SKINS_CONSISTENT_SOME_BRANCH", "SKINS_EXCLUDED_BOTH_BRANCHES"),
        )
        for name, pair in pairs.items():
            expected = abs(pair["delta_fm"]) <= pc.SKIN_SIGMA_LEVEL * pair["sigma_fm"]
            self.assertEqual(pair["consistent_at_2sigma"], expected, name)

    def test_skin_bands_use_published_constants(self) -> None:
        prex = self.payload["skin_comparison"]["PREX2_Pb208"]
        self.assertAlmostEqual(prex["center_fm"], 0.283, places=12)
        self.assertAlmostEqual(prex["sigma_fm"], 0.071, places=12)
        crex = self.payload["skin_comparison"]["CREX_Ca48"]
        self.assertAlmostEqual(crex["center_fm"], 0.121, places=12)
        self.assertAlmostEqual(crex["sigma_fm"], (0.026 ** 2 + 0.024 ** 2) ** 0.5, places=12)
        for block in (prex, crex):
            self.assertAlmostEqual(
                block["two_sigma_band_fm"][1] - block["two_sigma_band_fm"][0],
                4.0 * block["sigma_fm"], places=10,
            )

    def test_branch_rows_reproduce_retained_controls(self) -> None:
        rows = self.payload["branch_rows"]
        for branch, nucleus, target in (
            ("no_rho", "Pb208", pc.NO_RHO_CONTROL_RADII["Pb208"]),
            ("no_rho", "Ca48", pc.NO_RHO_CONTROL_RADII["Ca48"]),
            ("identified_j", "Ca40", pc.IDENTIFIED_BRANCH_RADII["Ca40"]),
            ("identified_j", "Pb208", pc.IDENTIFIED_BRANCH_RADII["Pb208"]),
            ("identified_j", "Ca48", pc.IDENTIFIED_BRANCH_RADII["Ca48"]),
        ):
            row = rows[f"{branch}:{nucleus}"]
            for key, want in target.items():
                self.assertAlmostEqual(row[key], want, delta=1.0e-9 * max(1.0, abs(want)))
            self.assertAlmostEqual(
                row["point_neutron_skin_fm"],
                row["rms_neutron_radius_fm"] - row["rms_point_proton_radius_fm"],
                places=9,
            )

    def test_weak_comparisons_are_descriptive(self) -> None:
        blocks = self.payload["weak_form_factor_and_apv_descriptive"]
        self.assertEqual(len(blocks), 4)
        for block in blocks:
            self.assertIn("descriptive only", block["weight"])
        labels = {block["label"] for block in blocks}
        self.assertEqual(
            labels,
            {"PREX2_Pb208", "PREX2_Pb208_no_rho", "CREX_Ca48", "CREX_Ca48_no_rho"},
        )

    def test_naive_j_entries_present(self) -> None:
        naive = self.payload["naive_j_for_skins_descriptive"]
        for experiment in ("PREX2_Pb208", "CREX_Ca48"):
            entry = naive[experiment]
            self.assertGreater(entry["response_fm_per_MeV"], 0.0)
            self.assertIsNotNone(entry["j_naive_MeV"])
            self.assertGreater(entry["j_naive_MeV"], pc.J_BAR_MEV)
            self.assertIn("extrapolation", entry["note"])

    def test_artifact_is_strict_json(self) -> None:
        def reject_constant(value: str) -> None:
            raise AssertionError(f"non-finite constant in artifact: {value}")

        json.loads(RESULT.read_text(encoding="utf-8"), parse_constant=reject_constant)

    def test_cli_help_smoke(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            pc.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
