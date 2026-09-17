"""Tests for the soft-isospin NS-family re-selection probe.

Fast tests verify the pre-registered grid, verdict bands, the canonical
constraint mirror, baseline-patch restoration and the two retained-control
reproductions (about one second each).  Artifact tests verify the retained
scan payload and skip wherever the untracked Lunacy evidence tree is absent.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_eos_beta_css_softening as soft
import nvg_isovector_transferability_probe as tp
import nvg_ns_predictive_audit as nsa
import nvg_soft_isospin_ns_reselection_probe as sp

TRANSFER_RESULT = ROOT / sp.TRANSFER_RESULT_REL
RESULT = ROOT / Path("Lunacy/runs/soft-isospin-reselection-2026-09-17/soft_isospin_reselection_result.json")


def _load_result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


class GridAndBandsTests(unittest.TestCase):
    def test_transferred_coupling_algebra(self) -> None:
        self.assertAlmostEqual(sp.C_RHO_TRANSFER, 2.0 * tp.J_BAR_MEV / tp.N0_FM3, places=15)
        self.assertAlmostEqual(sp.C_RHO_TRANSFER, (8.0 * tp.J_BAR_MEV / tp.N0_FM3) / 4.0, places=15)

    def test_pre_registered_grid_is_declared(self) -> None:
        self.assertEqual(sp.K1_GRID, (0.15, 0.20, 0.25, 0.30, 0.35))
        self.assertEqual(sp.K2_GRID, (0.60, 0.80, 1.00))
        self.assertEqual(sp.CS_GRID, (700.0, 800.0, 900.0, 1000.0, 1100.0))
        self.assertEqual(sp.N_TRANS_GRID, (1.4, 1.6, 1.8, 2.0, 2.2, 2.4))
        self.assertEqual(sp.DE_GRID, (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40))
        total = len(sp.K1_GRID) * len(sp.K2_GRID) * len(sp.CS_GRID) * len(sp.N_TRANS_GRID) * len(sp.DE_GRID)
        self.assertEqual(total, 3150)
        # Stage 1 (drop-in transition re-selection) is the baseline-isoscalar subset.
        self.assertEqual(len(sp.N_TRANS_GRID) * len(sp.DE_GRID), 42)

    def test_constraints_mirror_canonical_audit(self) -> None:
        self.assertEqual(sp.CONSTRAINT_NAMES, ("J0740_M_max", "GW170817_Lambda_tilde", "NICER_R14"))
        for name in sp.CONSTRAINT_NAMES:
            spec = nsa.OBSERVABLE_CONSTRAINTS[name]
            self.assertIsNotNone(spec)
        self.assertEqual(nsa.OBSERVABLE_CONSTRAINTS["J0740_M_max"]["lower"], 2.01)
        self.assertEqual(nsa.OBSERVABLE_CONSTRAINTS["GW170817_Lambda_tilde"]["upper"], 720.0)
        self.assertEqual(nsa.OBSERVABLE_CONSTRAINTS["NICER_R14"]["lower"], 11.2)
        self.assertEqual(nsa.OBSERVABLE_CONSTRAINTS["NICER_R14"]["upper"], 13.2)

    def test_verdict_logic(self) -> None:
        self.assertEqual(sp.verdict(0, 1), "SOFT_ISOSPIN_SURVIVOR_FOUND")
        self.assertEqual(sp.verdict(3, 5), "SOFT_ISOSPIN_SURVIVOR_FOUND")
        self.assertEqual(sp.verdict(0, 0), "NO_SURVIVOR_STIFFNESS_FLOOR")
        self.assertEqual(sp.verdict(4, 0), "NO_SURVIVOR_STIFFNESS_FLOOR")

    def test_fixed_inputs_at_canonical_values(self) -> None:
        self.assertEqual(sp.FIXED_VECTOR, {"alpha_v": 4.0, "nu_v": 2.0})
        self.assertAlmostEqual(sp.CS2_Q, 1.0 / 3.0, places=15)
        self.assertEqual(sp.SEQUENCE_POINTS, 72)
        self.assertEqual(sp.ROBUSTNESS_POINTS, 120)
        self.assertEqual(sp.BASELINE_ISOSCALAR, {"k1": 0.25, "k2": 0.80, "Cs": 900.0})


class ControlTests(unittest.TestCase):
    def test_control_baseline_reproduces_retained_chain(self) -> None:
        row = sp._control_chain(sp.CONTROL_BASELINE)
        self.assertEqual(row["status"], "EVALUATED")
        self.assertAlmostEqual(row["M_max_msun"], sp.CONTROL_BASELINE["M_max_msun"], places=9)
        self.assertAlmostEqual(row["R_1.4_km"], sp.CONTROL_BASELINE["R_1.4_km"], places=9)
        self.assertAlmostEqual(row["Lambda_1.4"], sp.CONTROL_BASELINE["Lambda_1.4"], places=9)

    def test_control_transferred_reproduces_retained_chain(self) -> None:
        row = sp._control_chain(sp.CONTROL_TRANSFERRED)
        self.assertEqual(row["status"], "EVALUATED")
        self.assertAlmostEqual(row["M_max_msun"], sp.CONTROL_TRANSFERRED["M_max_msun"], places=9)
        self.assertAlmostEqual(row["R_1.4_km"], sp.CONTROL_TRANSFERRED["R_1.4_km"], places=9)

    def test_control_fails_closed_on_tampering(self) -> None:
        original = copy.deepcopy(sp.CONTROL_TRANSFERRED)
        try:
            sp.CONTROL_TRANSFERRED["M_max_msun"] = 2.5
            with self.assertRaises(sp.SoftIsospinError):
                sp._control_chain(sp.CONTROL_TRANSFERRED)
        finally:
            sp.CONTROL_TRANSFERRED.clear()
            sp.CONTROL_TRANSFERRED.update(original)

    def test_baseline_patch_restores_all_keys(self) -> None:
        before = dict(soft.BEST_BASELINE)
        with sp._patched_baseline(k1=0.30, Cs=1100.0, Crho=139.2):
            self.assertEqual(soft.BEST_BASELINE["k1"], 0.30)
            self.assertEqual(soft.BEST_BASELINE["Crho"], 139.2)
        self.assertEqual(soft.BEST_BASELINE, before)
        with self.assertRaises(sp.SoftIsospinError):
            with sp._patched_baseline(bogus=1.0):
                pass
        self.assertEqual(soft.BEST_BASELINE, before)

    @unittest.skipUnless(TRANSFER_RESULT.is_file(), "retained transferability artifact not present (untracked)")
    def test_transfer_control_verifies_retained_coupling(self) -> None:
        control = sp._load_transfer_control()
        self.assertAlmostEqual(control["c_rho_transfer_MeV_fm3"], sp.C_RHO_TRANSFER, places=12)

    @unittest.skipUnless(TRANSFER_RESULT.is_file(), "retained transferability artifact not present (untracked)")
    def test_transfer_control_fails_closed_on_tampering(self) -> None:
        payload = json.loads(TRANSFER_RESULT.read_text(encoding="utf-8"))
        payload["part_b_ns_sector_transfer"]["cross_branch_transfer"]["c_rho_transfer_MeV_fm3"] = 500.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            original = sp.TRANSFER_RESULT_REL
            try:
                sp.TRANSFER_RESULT_REL = Path(tmp) / "tampered.json"
                with self.assertRaises(sp.SoftIsospinError):
                    sp._load_transfer_control()
            finally:
                sp.TRANSFER_RESULT_REL = original


class ChainRowTests(unittest.TestCase):
    """Live chain semantics on the two already-retained grid points (~1 s)."""

    def test_canon_transfer_point_flags(self) -> None:
        row = sp._chain_row(0.25, 0.80, 900.0, 2.0, 0.00)
        self.assertEqual(row["status"], "EVALUATED")
        self.assertAlmostEqual(row["M_max_msun"], sp.CONTROL_TRANSFERRED["M_max_msun"], places=9)
        flags = row["flags"]
        self.assertTrue(flags["J0740_M_max"])
        self.assertFalse(flags["GW170817_Lambda_tilde"])
        self.assertFalse(flags["NICER_R14"])
        self.assertFalse(row["all_constraints_pass"])
        self.assertEqual(row["all_constraints_pass"], all(flags.values()))
        # Margin-normalized binding constraint: Lambda_tilde is more violated
        # (−1.10) than R_1.4 (−0.50) at this point.
        self.assertEqual(sp._binding_constraint(row), "GW170817_Lambda_tilde")

    def test_max_softening_point_binds_j0740(self) -> None:
        row = sp._chain_row(0.25, 0.80, 900.0, 1.4, 0.40)
        self.assertEqual(row["status"], "EVALUATED")
        self.assertLess(row["M_max_msun"], 2.01)
        self.assertFalse(row["all_constraints_pass"])
        self.assertEqual(sp._binding_constraint(row), "J0740_M_max")


@unittest.skipUnless(RESULT.is_file(), "retained re-selection artifact not present (untracked)")
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = _load_result()

    def test_schema_status_and_evidence_weight(self) -> None:
        self.assertEqual(self.payload["schema"], sp.SCHEMA)
        self.assertEqual(self.payload["status"], sp.STATUS)
        self.assertEqual(self.payload["evidence_weight"], 0.0)

    def test_controls_reproduced_in_artifact(self) -> None:
        controls = self.payload["controls"]
        self.assertAlmostEqual(
            controls["baseline_chain_C_rho_600"]["M_max_msun"],
            sp.CONTROL_BASELINE["M_max_msun"], places=9,
        )
        self.assertAlmostEqual(
            controls["transferred_chain_C_rho_139_2"]["M_max_msun"],
            sp.CONTROL_TRANSFERRED["M_max_msun"], places=9,
        )

    def test_scan_counts_consistent(self) -> None:
        scan = self.payload["scan"]
        rows = self.payload["rows"]
        self.assertEqual(scan["grid_total"], len(rows))
        self.assertEqual(scan["evaluated"] + scan["build_failures"], scan["grid_total"])
        self.assertEqual(scan["grid_total"], 3150)
        stage1_rows = [
            row for row in rows
            if (row["k1"], row["k2"], row["Cs"]) == (0.25, 0.80, 900.0)
        ]
        self.assertEqual(len(stage1_rows), 42)
        self.assertLessEqual(scan["stage1_evaluated"], 42)
        self.assertLessEqual(scan["stage1_survivors"], scan["stage1_evaluated"])
        self.assertLessEqual(scan["stage2_survivors"], scan["evaluated"])

    def test_verdict_matches_survivor_counts(self) -> None:
        scan = self.payload["scan"]
        expected = sp.verdict(scan["stage1_survivors"], scan["stage2_survivors"])
        self.assertEqual(self.payload["verdict"], expected)
        self.assertIn(self.payload["verdict"], ("SOFT_ISOSPIN_SURVIVOR_FOUND", "NO_SURVIVOR_STIFFNESS_FLOOR"))

    def test_best_point_consistent_with_verdict(self) -> None:
        best = self.payload["best_point"]
        self.assertIsNotNone(best)
        self.assertEqual(best["status"], "EVALUATED")
        if self.payload["verdict"] == "SOFT_ISOSPIN_SURVIVOR_FOUND":
            self.assertTrue(best["all_constraints_pass"])
            self.assertGreater(best["margin"], 0.0)
        else:
            self.assertIn(best["binding_constraint"], sp.CONSTRAINT_NAMES)
        self.assertEqual(best["robustness_120_points"]["sequence_points"], 120)
        self.assertIn("J_diff_MeV", best["symmetry_energy_descriptive"])
        # The recorded binding constraint must be reproducible from the row.
        self.assertEqual(sp._binding_constraint(best), best["binding_constraint"])

    def test_survivor_rows_actually_pass(self) -> None:
        for row in self.payload["rows"]:
            if row["status"] == "EVALUATED" and row["all_constraints_pass"]:
                self.assertTrue(all(row["flags"].values()))
                self.assertGreaterEqual(row["M_max_msun"], 2.01)
                self.assertLessEqual(max(row["Lambda_tilde_symmetric_1.36"],
                                         row["Lambda_tilde_asymmetric_1.46_1.27"]), 720.0)
                self.assertGreaterEqual(min(row["Lambda_tilde_symmetric_1.36"],
                                            row["Lambda_tilde_asymmetric_1.46_1.27"]), 70.0)
                self.assertGreaterEqual(row["R_1.4_km"], 11.2)
                self.assertLessEqual(row["R_1.4_km"], 13.2)

    def test_artifact_is_strict_json(self) -> None:
        def reject_constant(value: str) -> None:
            raise AssertionError(f"non-finite constant in artifact: {value}")

        json.loads(RESULT.read_text(encoding="utf-8"), parse_constant=reject_constant)

    def test_cli_help_smoke(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            sp.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
