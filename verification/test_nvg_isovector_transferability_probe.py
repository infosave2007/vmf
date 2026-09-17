"""Tests for the isovector transferability probe.

Fast tests cover cited anchors, transfer algebra, pre-registered verdict-band
logic and the live symmetry-energy arithmetic of the saturated-vector model
(no BVP or TOV solve).  Artifact tests verify the retained result payload and
skip wherever the untracked Lunacy evidence tree is absent (CI checkouts).
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_isovector_transferability_probe as tp
import nvg_sn132_discriminating_set_probe as sn

ARTIFACT = ROOT / tp.SN132_RESULT_REL
RESULT_REL = Path("Lunacy/runs/isovector-transferability-2026-09-17/isovector_transferability_result.json")
RESULT = ROOT / RESULT_REL


def _load_result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


class CitedAnchorTests(unittest.TestCase):
    def test_universal_design_j_identity(self) -> None:
        self.assertAlmostEqual(float(sn.universal_design_j(tp.J_BAR_MEV)), 23.78013451414814, places=12)

    def test_t_contact_identity(self) -> None:
        self.assertAlmostEqual(float(sn.T_CONTACT_MEV), 12.644118442969951, places=12)

    def test_canonical_chain_constants_live_verified(self) -> None:
        canonical = tp._load_nsa_canonical()
        self.assertAlmostEqual(canonical["canonical"]["M_max"], tp.CANONICAL_CHAIN["M_max_msun"], places=12)
        self.assertAlmostEqual(canonical["canonical"]["R_1.4"], tp.CANONICAL_CHAIN["R_1.4_km"], places=12)
        self.assertAlmostEqual(canonical["canonical"]["Lambda_1.4"], tp.CANONICAL_CHAIN["Lambda_1.4"], places=12)

    @unittest.skipUnless(ARTIFACT.is_file(), "retained Sn132 artifact not present (untracked)")
    def test_sn132_artifact_matches_cited_constants(self) -> None:
        retained = tp._load_sn132_artifact()
        self.assertAlmostEqual(retained["j_bar_MeV"], tp.J_BAR_MEV, places=12)
        self.assertAlmostEqual(
            retained["J_design_universal_MeV"], float(sn.universal_design_j(tp.J_BAR_MEV)), places=12
        )
        self.assertEqual(set(retained["cases"]), set(tp.ALL_NUCLEI))

    @unittest.skipUnless(ARTIFACT.is_file(), "retained Sn132 artifact not present (untracked)")
    def test_sn132_artifact_verifier_fails_closed_on_tampering(self) -> None:
        payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        payload["universal_j_demonstration"]["j_bar_MeV"] = 12.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            original = tp.SN132_RESULT_REL
            try:
                tp.SN132_RESULT_REL = Path(tmp) / "tampered.json"
                with self.assertRaises(tp.TransferabilityError):
                    tp._load_sn132_artifact()
            finally:
                tp.SN132_RESULT_REL = original


class TransferAlgebraTests(unittest.TestCase):
    def test_transfer_coupling_algebra(self) -> None:
        c_transfer = 2.0 * tp.J_BAR_MEV / tp.N0_FM3
        c_finite_convention = 8.0 * tp.J_BAR_MEV / tp.N0_FM3
        self.assertAlmostEqual(c_transfer, c_finite_convention / 4.0, places=12)

    def test_ns_baseline_contact_piece(self) -> None:
        self.assertAlmostEqual(0.5 * tp.NS_RHO_BASELINE * tp.N0_FM3, 48.0, places=12)

    def test_cross_branch_ratio(self) -> None:
        ratio = (0.5 * tp.NS_RHO_BASELINE * tp.N0_FM3) / tp.J_BAR_MEV
        self.assertAlmostEqual(ratio, 4.3108, places=3)

    def test_verdict_part_b_thresholds(self) -> None:
        self.assertEqual(tp.verdict_part_b(2.05), "TRANSFERABLE_TO_NS_SECTOR")
        self.assertEqual(tp.verdict_part_b(2.01), "TRANSFERABLE_TO_NS_SECTOR")
        self.assertEqual(tp.verdict_part_b(2.009), "PARTIAL_TENSION_WITH_J0740")
        self.assertEqual(tp.verdict_part_b(1.90), "PARTIAL_TENSION_WITH_J0740")
        self.assertEqual(tp.verdict_part_b(1.85), "NOT_TRANSFERABLE_MAX_MASS_LOST")

    def test_verdict_part_a_requires_all_bands(self) -> None:
        self.assertEqual(
            tp.verdict_part_a(anchor_ok=True, ds_ok=True, total_ok=True, per_nucleus_ok=True),
            "REANCHORED_UNIVERSAL_J_SURVIVES",
        )
        for kwargs in (
            {"anchor_ok": False},
            {"ds_ok": False},
            {"total_ok": False},
            {"per_nucleus_ok": False},
        ):
            base = {"anchor_ok": True, "ds_ok": True, "total_ok": True, "per_nucleus_ok": True}
            base.update(kwargs)
            self.assertEqual(tp.verdict_part_a(**base), "REANCHORED_UNIVERSAL_J_BREAKS_BANDS")


class SymmetryEnergyTests(unittest.TestCase):
    """Live arithmetic of the saturated-vector functional (fast, no TOV)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = tp.symmetry_energy_summary(tp.NS_RHO_BASELINE)
        cls.retuned = tp.symmetry_energy_summary(2.0 * tp.J_BAR_MEV / tp.N0_FM3)

    def test_baseline_symmetry_energy_in_sanity_band(self) -> None:
        low, high = tp.J_BASELINE_SANITY_BAND_MEV
        self.assertGreaterEqual(self.baseline["J_diff_MeV"], low)
        self.assertLessEqual(self.baseline["J_diff_MeV"], high)
        l_low, l_high = tp.L_BASELINE_SANITY_BAND_MEV
        self.assertGreaterEqual(self.baseline["L_MeV"], l_low)
        self.assertLessEqual(self.baseline["L_MeV"], l_high)

    def test_baseline_contact_piece(self) -> None:
        self.assertAlmostEqual(self.baseline["j_contact_MeV"], 48.0, places=12)

    def test_retuned_contact_piece_is_the_identified_j(self) -> None:
        self.assertAlmostEqual(self.retuned["j_contact_MeV"], tp.J_BAR_MEV, places=12)

    def test_retuned_isospin_is_softer_than_baseline(self) -> None:
        self.assertLess(self.retuned["J_diff_MeV"], self.baseline["J_diff_MeV"])
        self.assertLess(self.retuned["L_MeV"], self.baseline["L_MeV"])

    def test_nonparabolicity_is_small(self) -> None:
        for summary in (self.baseline, self.retuned):
            self.assertLess(abs(summary["nonparabolicity_MeV"]), 5.0)

    def test_symmetry_curve_grid_is_declared(self) -> None:
        steps = int(round((tp.SYMMETRY_GRID_MAX_FM3 - tp.SYMMETRY_GRID_MIN_FM3) / tp.SYMMETRY_GRID_STEP_FM3))
        self.assertEqual(len(self.baseline["S_diff_curve"]), steps + 1)
        for point in self.baseline["S_diff_curve"]:
            self.assertTrue(math.isfinite(point["S_diff_MeV"]))


@unittest.skipUnless(RESULT.is_file(), "retained transferability artifact not present (untracked)")
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = _load_result()

    def test_schema_status_and_evidence_weight(self) -> None:
        self.assertEqual(self.payload["schema"], tp.SCHEMA)
        self.assertEqual(self.payload["status"], tp.STATUS)
        self.assertEqual(self.payload["evidence_weight"], 0.0)

    def test_retained_inputs_verified(self) -> None:
        retained = self.payload["retained_inputs"]["sn132_discriminating_set"]
        self.assertAlmostEqual(retained["j_bar_MeV"], tp.J_BAR_MEV, places=12)

    def test_part_a_anchor_is_reanchored(self) -> None:
        part_a = self.payload["part_a_reanchored_finite_branch"]
        recal = part_a["recalibration"]
        self.assertLessEqual(abs(recal["anchor_residual_per_A_MeV"]), tp.ANCHOR_RECAL_ABS_TOL_BA_MEV)
        ds = abs(recal["ds_over_s"])
        self.assertGreaterEqual(ds, tp.DS_OVER_S_ABS_BAND[0])
        self.assertLessEqual(ds, tp.DS_OVER_S_ABS_BAND[1])

    def test_part_a_residual_bands(self) -> None:
        part_a = self.payload["part_a_reanchored_finite_branch"]
        for name in tp.HELDOUT_NUCLEI:
            residual = part_a["cases"][name]["residual_total_MeV"]
            self.assertLessEqual(
                abs(residual), tp.PART_A_PER_NUCLEUS_ABS_BAND_MEV[1], f"{name}: {residual} MeV"
            )
        self.assertLessEqual(
            abs(part_a["heldout_total_residual_MeV"]), tp.PART_A_TOTAL_ABS_RESIDUAL_BAND_MEV[1]
        )

    def test_part_a_verdict_matches_bands(self) -> None:
        part_a = self.payload["part_a_reanchored_finite_branch"]
        bands = part_a["bands"]
        expected = tp.verdict_part_a(
            anchor_ok=bands["anchor_within_tol"],
            ds_ok=bands["ds_over_s_within_band"],
            total_ok=bands["total_residual_within_band"],
            per_nucleus_ok=bands["per_nucleus_residuals_within_band"],
        )
        self.assertEqual(part_a["verdict"], expected)
        self.assertIn(part_a["verdict"], ("REANCHORED_UNIVERSAL_J_SURVIVES", "REANCHORED_UNIVERSAL_J_BREAKS_BANDS"))

    def test_part_b_canonical_control_reproduces_frozen_chain(self) -> None:
        control = self.payload["part_b_ns_sector_transfer"]["canonical_chain_control"]
        self.assertAlmostEqual(control["M_max_msun"], tp.CANONICAL_CHAIN["M_max_msun"], places=9)
        self.assertAlmostEqual(control["R_1.4_km"], tp.CANONICAL_CHAIN["R_1.4_km"], places=9)
        self.assertAlmostEqual(control["Lambda_1.4"], tp.CANONICAL_CHAIN["Lambda_1.4"], places=9)
        self.assertTrue(control["reproduces_frozen_canonical"])

    def test_part_b_transfer_coupling_and_verdict(self) -> None:
        part_b = self.payload["part_b_ns_sector_transfer"]
        transfer = part_b["cross_branch_transfer"]
        self.assertAlmostEqual(transfer["c_rho_transfer_MeV_fm3"], 2.0 * tp.J_BAR_MEV / tp.N0_FM3, places=12)
        retuned = part_b["retuned_chain"]
        self.assertEqual(part_b["verdict"], tp.verdict_part_b(float(retuned["M_max_msun"])))
        self.assertAlmostEqual(part_b["retuned_symmetry_energy"]["j_contact_MeV"], tp.J_BAR_MEV, places=12)

    def test_artifact_is_strict_json(self) -> None:
        def reject_constant(value: str) -> None:
            raise AssertionError(f"non-finite constant in artifact: {value}")

        text = RESULT.read_text(encoding="utf-8")
        json.loads(text, parse_constant=reject_constant)

    def test_cli_help_smoke(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            tp.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
