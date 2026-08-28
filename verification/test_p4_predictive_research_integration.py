"""Integration contracts for the Phase-4 predictive-research ledger."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_predictive_research_ledger as ledger
from registry import validate_registry


class PredictiveResearchIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = ledger.build_ledger()

    def test_registry_and_manifest_cover_predictive_inventory(self) -> None:
        counts = validate_registry(ROOT)
        self.assertGreaterEqual(counts["entries"], 300)
        registry = json.loads((HERE / "registry.json").read_text(encoding="utf-8"))["entries"]
        inventory = self.payload["inventory"]
        for field in ("predictive_producers", "predictive_tests", "durable_artifacts", "source_inputs", "manifests"):
            for path in inventory[field]:
                self.assertIn(path, registry, path)
        self.assertTrue(inventory["immutable_controls_excluded"])
        self.assertEqual(self.payload["evidence_policy"]["independent_evidence_weight"], 0.0)

    def test_tracked_pbh_promotion_is_live_and_has_no_ignored_dependency(self) -> None:
        tracked = ledger.PBH_RESULT_PATH
        figure = ledger.PBH_FIGURE_PATH
        self.assertEqual(tracked, HERE / "P1-S3-pbh-nanograv-audit.json")
        self.assertTrue(tracked.is_file())
        self.assertTrue(figure.is_file())
        self.assertEqual(hashlib.sha256(tracked.read_bytes()).hexdigest(), ledger.PBH_RESULT_SHA256)
        self.assertEqual(hashlib.sha256(figure.read_bytes()).hexdigest(), ledger.PBH_FIGURE_SHA256)
        serialized = json.dumps(self.payload, sort_keys=True)
        self.assertNotIn("Lunacy/", serialized)
        self.assertIn("verification/P1-S3-pbh-nanograv-audit.json", serialized)
        self.assertIn("verification/P1-S3-pbh-nanograv-boundary.png", serialized)
        artifacts = json.loads((HERE / "artifact_manifest.json").read_text(encoding="utf-8"))["entries"]
        for path, digest in (
            ("verification/P1-S3-pbh-nanograv-audit.json", ledger.PBH_RESULT_SHA256),
            ("verification/P1-S3-pbh-nanograv-boundary.png", ledger.PBH_FIGURE_SHA256),
        ):
            self.assertIn(path, artifacts)
            self.assertEqual(artifacts[path]["sha256"], digest)

    def test_gw_template_regeneration_is_deterministic_and_non_evidence(self) -> None:
        path = HERE / "data" / "nvg_gw_template.txt"
        before = path.read_bytes()
        completed = subprocess.run(
            [sys.executable, str(HERE / "nvg_gw_spectrum_template.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(path.read_bytes(), before)
        source = path.read_text(encoding="utf-8")
        self.assertIn("# Provenance: verification/nvg_gw_spectrum_template.py", source)
        self.assertIn("# Status: NON_EVIDENCE (generated_unverified; synthetic template)", source)
        digest = hashlib.sha256(before).hexdigest()
        provenance = json.loads((HERE / "data" / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["entries"]["nvg_gw_template.txt"]["sha256"], digest)

    def test_final_claims_preserve_p3_s8_scope(self) -> None:
        claims = {row["id"]: row for row in self.payload["claims"]}
        self.assertEqual(self.payload["status"], "FINAL_P4_S1_MACHINE_LEDGER")
        self.assertEqual(claims["ns_hartle_j0737a"]["branch_id"], 1)
        self.assertFalse(claims["ns_hartle_j0737a"]["gap_crossed"])
        self.assertEqual(claims["ns_hartle_j0737a"]["supported_display"]["Lambda_A"], {"lower": 680.0, "upper": 686.0, "digits": 0})
        self.assertEqual(claims["barotropic_null"]["status"], "DERIVED_NULL_CONTROL_BAROTROPIC_NEUTRAL_BUOYANCY")
        self.assertEqual(claims["echo"]["q_payload_integrity"]["shape"], [259, 15])
        self.assertEqual(claims["echo"]["observational_status"], "BLOCKED_MISSING_STRAIN_POSTERIOR_NOISE_PRODUCTS")
        self.assertEqual(claims["s8"]["critical_boundary_status"], "NOT_IDENTIFIABLE_GRID_AND_NORMALIZATION_DEPENDENT")
        self.assertEqual(claims["s8"]["resolution_crossings"], [0.3, 0.3, 0.305])
        self.assertEqual(
            claims["s8"]["normalization_one_sigma_counts_by_omega"]["0.3"]["omega_m_0.315"]["one_sigma_overlap"],
            45,
        )
        self.assertEqual(claims["s8"]["payload_sha256"], "419777c3ebff39e38036d20fbef3a5b9dd00e9a064409b9e2d45b85ebdf26267")
        self.assertEqual(claims["pbh"]["seed_band"]["minimum_required_rate_product"], 299558.79267302185)
        self.assertEqual(claims["pbh"]["expanded_scan"]["first_reachable_rung"], 17)
        self.assertEqual(claims["M_Omega"]["status"], "BLOCKED_NO_PHYSICAL_DEPENDENCY")
        self.assertEqual(claims["ns_quadrupole_Q"]["status"], "blocked")
        self.assertTrue(all(row["evidence_weight"] == 0.0 for row in self.payload["claims"]))

    def test_ledger_digest_and_semantic_mutations_fail_closed(self) -> None:
        ledger.assert_ledger(self.payload)
        mutated = copy.deepcopy(self.payload)
        mutated["claims"][1]["raw_bands"]["Lambda_A"][0] = 1.0
        with self.assertRaises(AssertionError):
            ledger.assert_ledger(mutated)
        mutated = copy.deepcopy(self.payload)
        mutated["claims"][10]["critical_boundary_status"] = "CRITICAL_THRESHOLD"
        with self.assertRaises(AssertionError):
            ledger.assert_ledger(mutated)
        mutated = copy.deepcopy(self.payload)
        mutated["claims"][11]["max_row_integrity"]["expanded_max_row"]["cycle"] = 1
        with self.assertRaises(AssertionError):
            ledger.assert_ledger(mutated)

    def test_cli_regenerates_deterministically(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(HERE / "nvg_predictive_research_ledger.py"), "--no-write"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("FINAL_P4_S1_MACHINE_LEDGER", completed.stdout)
        self.assertIn("claims=12", completed.stdout)


if __name__ == "__main__":
    unittest.main()
