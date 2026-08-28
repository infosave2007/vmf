"""Semantic checks for the Phase 6 public/static-report repair (P6-S1)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ROOT_PATH = Path(ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_fork_b_full_chain as fork_b
import nvg_joint_ns_inference as joint
import run_all_checks


class P6S1PublicReportTests(unittest.TestCase):
    def test_public_reports_match_runtime_canonical_values(self):
        predictions, metadata = joint.compute_nvg_predictions()
        result = joint.run_joint_inference(predictions, metadata)
        self.assertAlmostEqual(predictions["M_max"], 2.0479507406, places=8)
        self.assertAlmostEqual(predictions["R_1.4"], 12.5500010, places=6)
        self.assertAlmostEqual(predictions["Lambda_1.4"], 519.4223808, places=6)
        self.assertAlmostEqual(result["reduced_chi"], 0.6835938355, places=8)
        self.assertEqual(result["comparison_status"], "CONDITIONAL_IN_SAMPLE")
        self.assertEqual(result["in_sample_count"], 3)
        for name in ("README.md", "README_RU.md"):
            text = (ROOT_PATH / name).read_text(encoding="utf-8")
            self.assertIn("2.048", text)
            self.assertIn("12.550", text)
            self.assertIn("519.4", text)
            self.assertIn("0.684", text)
            self.assertIn("conditional" if name == "README.md" else "условн", text.lower())

    def test_global_and_sn1987a_public_claims_are_withheld(self):
        forbidden = ("2.76/3", "p = 0.43", "7 quantitative pulls", "7 количественных пулов")
        for name in ("README.md", "README_RU.md"):
            text = (ROOT_PATH / name).read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker.lower(), text.lower())
            row = next(line for line in text.splitlines() if "SN1987A" in line)
            self.assertRegex(row.lower(), r"conditional|условн")
            self.assertRegex(row.lower(), r"no evidence|без доказательств|не устанавливает")
            self.assertNotRegex(row, r"✅.*(?:20%|20\\s*%|Limit|предел|огранич)")

    def test_fork_b_is_quarantined_and_fails_closed(self):
        self.assertEqual(fork_b.FORK_STATUS, "QUARANTINED_ALTERNATE_CALIBRATION")
        self.assertEqual(fork_b.EVIDENCE_WEIGHT, 0)
        source = (Path(HERE) / "nvg_fork_b_full_chain.py").read_text(encoding="utf-8")
        self.assertIn("QUARANTINED_ALTERNATE_CALIBRATION", source)
        self.assertIn("evidence_weight=", source)
        self.assertNotIn("FULL NEUTRON-STAR CHAIN", source)
        self.assertNotIn("FORK-B CANONICAL CANDIDATE", source)

    def test_runner_has_process_metadata_only(self):
        source = (Path(HERE) / "run_all_checks.py").read_text(encoding="utf-8")
        self.assertNotIn("claim", source.lower())
        allowed = {"name", "script", "critical", "timeout"}
        self.assertGreater(len(run_all_checks.CHECKS), 0)
        for check in run_all_checks.CHECKS + run_all_checks.OPTIONAL_CHECKS:
            self.assertTrue(set(check).issubset(allowed))
            self.assertIn("name", check)
            self.assertIn("script", check)

    def test_readme_numbers_do_not_reintroduce_superseded_ns_snapshot(self):
        # These were the stale fitted snapshot values identified by P5-S1.
        stale = ("1.89", "13.11", "393", "489", "313", "12.49", "12.27", "12.85")
        for name in ("README.md", "README_RU.md"):
            text = (ROOT_PATH / name).read_text(encoding="utf-8")
            for value in stale:
                self.assertNotIn(value, text)


if __name__ == "__main__":
    unittest.main()
