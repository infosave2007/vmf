"""P11-S2 path, provenance, and generated-artifact contract checks."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_gwtc3_plot_results as gwtc3_plot
import nvg_gwtc3_statistics as gwtc3_stats
import nvg_gwtc3_mass_scan as gwtc3_scan
import nvg_echo_waveform_generator as echo_waveform
import nvg_iloveq_plot
import nvg_pycbc_echo_search as pycbc_echo


class P11S2ArtifactProvenanceTests(unittest.TestCase):
    def test_provenance_schema_hashes_and_paths(self):
        path = HERE / "data" / "provenance.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 2)
        for name, entry in payload["entries"].items():
            self.assertIn(entry["kind"], {"input", "generated_artifact"})
            for field in ("status", "producer", "config"):
                self.assertIn(field, entry)
            data_path = path.parent / name
            self.assertTrue(data_path.is_file(), data_path)
            if entry.get("sha256"):
                digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
                self.assertEqual(digest, entry["sha256"], name)

    def test_artifact_manifest_entries_are_canonical_and_present(self):
        path = HERE / "artifact_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertGreaterEqual(len(payload["entries"]), 10)
        for artifact_path, entry in payload["entries"].items():
            self.assertEqual(entry["kind"], "generated_artifact")
            for field in ("producer", "command", "status", "canonical_path"):
                self.assertIn(field, entry)
            self.assertEqual(entry["canonical_path"], artifact_path)
            if not entry["status"].startswith("ignored_not_tracked"):
                self.assertTrue((ROOT / artifact_path).is_file(), artifact_path)

    def test_owned_default_paths_ignore_temporary_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_cwd = Path.cwd()
            os.chdir(temporary)
            try:
                self.assertEqual(gwtc3_scan.resolve_output_path(), HERE / "gwtc3_nvg_results.csv")
                self.assertEqual(gwtc3_stats.resolve_input_path(), HERE / "gwtc3_nvg_results.csv")
                self.assertEqual(
                    gwtc3_plot.resolve_path(None, gwtc3_plot.DEFAULT_OUTPUT),
                    HERE / "nvg_gwtc3_snr_distribution.png",
                )
                self.assertEqual(echo_waveform.OUTPUT_PATH, HERE / "nvg_echo_template.png")
                self.assertEqual(pycbc_echo.OUTPUT_PATH, HERE / "nvg_gw150914_echo_snr.png")
                gwtc3_stats.calculate_stats()
                self.assertFalse((Path(temporary) / "gwtc3_nvg_results.csv").exists())
            finally:
                os.chdir(old_cwd)

    def test_plot_and_waveform_producers_target_canonical_survivors(self):
        with tempfile.TemporaryDirectory() as temporary:
            old_cwd = Path.cwd()
            os.chdir(temporary)
            try:
                with patch.object(gwtc3_plot.plt, "savefig") as savefig:
                    gwtc3_plot.plot_results()
                    savefig.assert_called_once()
                    self.assertEqual(Path(savefig.call_args.args[0]), HERE / "nvg_gwtc3_snr_distribution.png")

                with patch.object(echo_waveform.plt, "savefig") as savefig:
                    runpy.run_path(str(HERE / "nvg_echo_waveform_generator.py"), run_name="__main__")
                    savefig.assert_called_once()
                    self.assertEqual(Path(savefig.call_args.args[0]), HERE / "nvg_echo_template.png")
            finally:
                os.chdir(old_cwd)

    def test_ilove_report_matches_runtime_producer(self):
        report = json.loads((HERE / "fig_iloveq_universal_report.json").read_text(encoding="utf-8"))
        sequence = nvg_iloveq_plot.compute_vmf_sequence()
        summary = nvg_iloveq_plot.summarize_sequence(sequence)
        self.assertEqual(report["source_script"], "verification/nvg_iloveq_plot.py")
        self.assertEqual(report["figure"], "verification/fig_iloveq_universal.png")
        self.assertEqual(report["status"], summary["status"])
        self.assertEqual(report["summary"]["sequence_count"], len(sequence))
        self.assertAlmostEqual(report["summary"]["lambda_14"], summary["lambda_14"], places=12)
        self.assertAlmostEqual(report["summary"]["max_deviation"], summary["max_deviation"], places=12)
        self.assertEqual(report["sequence"], sequence)

    def test_authorized_root_duplicates_are_absent_and_survivors_remain(self):
        duplicates = (
            "gwtc3_nvg_results.csv",
            "nvg_echo_template.png",
            "nvg_gw150914_echo_snr.png",
            "nvg_gwtc3_snr_distribution.png",
        )
        survivors = (
            HERE / "gwtc3_nvg_results.csv",
            HERE / "nvg_echo_template.png",
            HERE / "nvg_gw150914_echo_snr.png",
            HERE / "nvg_gwtc3_snr_distribution.png",
        )
        for duplicate in duplicates:
            self.assertFalse((ROOT / duplicate).exists(), duplicate)
        for survivor in survivors:
            self.assertTrue(survivor.is_file(), survivor)


if __name__ == "__main__":
    unittest.main()
