"""Focused semantic checks for the Phase 3 prospective reach ledger."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_prospective_falsification_reach as audit


class P3S3ProspectiveReachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit.build_result()

    def test_scope_statuses_and_pending_ns_merge(self) -> None:
        result = self.result
        self.assertEqual(result["schema_version"], audit.SCHEMA_VERSION)
        self.assertEqual(result["status"], "COMPLETE_WITH_PROSPECTIVE_SENSITIVITY_AND_PENDING_NS_MERGE")
        self.assertEqual(result["echo"]["status"], "sensitivity")
        self.assertEqual(result["s8"]["status"], "sensitivity")
        self.assertEqual(result["pbh"]["status"], "sensitivity")
        self.assertEqual(result["ns_hartle_pending_merge"]["status"], "blocked")
        self.assertIn("no import", result["ns_hartle_pending_merge"]["merge_rule"])
        self.assertEqual(result["target_leakage_controls"]["status"], "PASS_NO_TARGET_FIT_OR_OBSERVATION_RELABELLING")
        self.assertTrue(all(result["target_leakage_controls"]["checks"].values()))

    def test_echo_exposure_union_and_asymptotic_control(self) -> None:
        echo_result = self.result["echo"]
        self.assertEqual(echo_result["selected_event_count"], 259)
        curve = echo_result["exposure_curve"]
        self.assertEqual(curve[0]["selected_event_count"], 1)
        self.assertEqual(curve[-1]["selected_event_count"], 259)
        self.assertTrue(all(curve[i]["exposure_network_snr2"] < curve[i + 1]["exposure_network_snr2"] for i in range(len(curve) - 1)))
        self.assertTrue(all(curve[i]["a90_raw"] > curve[i + 1]["a90_raw"] for i in range(len(curve) - 1)))
        self.assertEqual(echo_result["numerical_union"]["raw_interval"], [0.03319408380348728, 0.03356685238591077])
        self.assertEqual(echo_result["asymptotic_scaling"]["expected_small_amplitude_exponent"], -0.5)
        self.assertTrue(echo_result["asymptotic_scaling"]["finite_prefix_is_not_asymptotic_claim"])
        self.assertIn("not observational", echo_result["scope"])

    def test_s8_boundary_expansion_is_resolution_and_overlay_scoped(self) -> None:
        s8_result = self.result["s8"]
        self.assertEqual(s8_result["maintained_slice"]["one_sigma_overlap_count"], 0)
        nearest = s8_result["boundary_expansion"]["nearest_grid_boundary"]
        self.assertIsNotNone(nearest)
        self.assertAlmostEqual(nearest["omega_m"], 0.30, places=12)
        self.assertTrue(s8_result["resolution_probe"]["boundary_resolution_dependent"])
        self.assertEqual(s8_result["boundary_expansion"]["critical_boundary_status"], "NOT_IDENTIFIABLE_GRID_AND_NORMALIZATION_DEPENDENT")
        self.assertEqual(s8_result["normalization_probe"]["status"], "NOT_IDENTIFIABLE_GRID_AND_NORMALIZATION_DEPENDENT")
        self.assertTrue(s8_result["normalization_probe"]["normalization_changes_one_sigma_overlap"])
        current = [row["counts"]["omega_m_0.315"]["one_sigma_overlap"] for row in s8_result["normalization_probe"]["rows"]]
        alternate = [row["counts"]["omega_m_0.300"]["one_sigma_overlap"] for row in s8_result["normalization_probe"]["rows"]]
        self.assertEqual(current, [45, 2, 0, 0, 0, 0])
        self.assertEqual(alternate, [0, 0, 0, 0, 0, 0])
        self.assertFalse(s8_result["legacy_overlay_scope"]["observed_likelihood"])
        self.assertEqual(s8_result["legacy_overlay_scope"]["provenance_status"], "unknown_provenance_sensitivity_overlay")

    def test_pbh_seed_band_and_expanded_rung_requirements(self) -> None:
        pbh_result = self.result["pbh"]
        seed = pbh_result["seed_band_requirements"]
        self.assertTrue(seed["all_rate_requirements_outside_box"])
        self.assertGreater(seed["minimum_required_rate_product"], 1.0)
        self.assertAlmostEqual(seed["minimum_row"]["cycle"], 11)
        self.assertEqual(pbh_result["expanded_jwst_upper_density_scan"]["first_reachable_rung"], 17)
        row17 = next(row for row in pbh_result["unit_rate_abundance_requirements"]["rows"] if row["cycle"] == 17)
        self.assertTrue(row17["inside_jwst_seed_density_band"])
        self.assertTrue(row17["inside_dm_budget"])
        row11 = next(row for row in pbh_result["unit_rate_abundance_requirements"]["rows"] if row["cycle"] == 11)
        self.assertFalse(row11["inside_jwst_seed_density_band"])
        self.assertTrue(row11["inside_dm_budget"])
        envelope = pbh_result["frozen_expanded_envelope_boundary"]
        self.assertTrue(envelope["at_upper_cycle_boundary"])
        self.assertTrue(envelope["at_upper_abundance_boundary"])
        self.assertTrue(envelope["at_upper_rate_boundaries"])

    def test_mutation_guards_fail_closed(self) -> None:
        echo_path = audit.ECHO_RESULT_PATH
        echo_payload = json.loads(echo_path.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(echo_payload)
        mutated["provenance"]["source_to_artifact_provenance"]["status"] = "MUTATED"
        with self.assertRaises(audit.ProvenanceError):
            audit._validate_echo_input(mutated)
        pbh_payload = json.loads(audit.PBH_RESULT_PATH.read_text(encoding="utf-8"))
        pbh_mutated = copy.deepcopy(pbh_payload)
        pbh_mutated["scan_contract"]["external_exclusions_used"] = True
        with self.assertRaises(audit.ProvenanceError):
            audit._validate_pbh_input(pbh_mutated)

        q_mutated = copy.deepcopy(echo_payload)
        q_mutated["sensitivity"]["calibration"]["injections"]["efficiency_rows"][0]["efficiency"][0] += 1.0e-4
        with self.assertRaises(audit.ProvenanceError):
            audit._validate_echo_input(q_mutated)

        max_mutated = copy.deepcopy(pbh_payload)
        max_mutated["internal_budget_expanded_boundary"]["max_row"]["fraction_dm"] = 0.123456789
        with self.assertRaises(audit.ProvenanceError):
            audit._validate_pbh_input(max_mutated)

    def test_complete_payload_integrity_and_former_probes(self) -> None:
        audit.assert_artifact_provenance(self.result)
        self.assertEqual(self.result["echo"]["q_payload_integrity"]["shape"], [259, 15])
        self.assertEqual(self.result["pbh"]["target_leakage"]["scan_maxima_copied_from_frozen_artifact"], False)
        self.assertEqual(self.result["adversarial_mutations"]["status"], "PASS_FAIL_CLOSED_FORMER_ADVERSARIAL_PROBES")
        mutated = copy.deepcopy(self.result)
        mutated["s8"]["normalization_probe"]["status"] = "MUTATED"
        with self.assertRaises(AssertionError):
            audit.assert_artifact_provenance(mutated)

    def test_artifacts_and_cli_are_runtime_derived(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = audit.write_artifacts(self.result, json_path=Path(temp_dir) / "result.json", csv_path=Path(temp_dir) / "reach.csv", figure_path=Path(temp_dir) / "reach.png")
            for path in paths:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 1000)
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], audit.SCHEMA_VERSION)
            with paths[1].open(newline="", encoding="utf-8") as handle:
                rows = list(__import__("csv").DictReader(handle))
            self.assertGreater(len(rows), 20)

        completed = subprocess.run(
            [sys.executable, str(HERE / "nvg_prospective_falsification_reach.py"), "--no-write"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("COMPLETE_WITH_PROSPECTIVE_SENSITIVITY_AND_PENDING_NS_MERGE", completed.stdout)
        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            self.assertEqual(audit.main(["--no-write"]), 0)
        self.assertIn("raw union", rendered.getvalue())


if __name__ == "__main__":
    unittest.main()
