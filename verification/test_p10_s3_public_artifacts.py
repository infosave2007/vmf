"""Semantic checks for the Phase 10 public/artifact synchronization boundary."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


class P10S3PublicArtifactTests(unittest.TestCase):
    def test_bilingual_readmes_use_current_runtime_boundary(self):
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        russian = (ROOT / "README_RU.md").read_text(encoding="utf-8")
        for text in (english, russian):
            self.assertNotIn("Falsifiable_Predictions-56", text)
            self.assertNotIn("Confirmed_Against_Data-52", text)
            self.assertNotIn("Awaiting_Future_Experiments-4", text)
            self.assertIn("PUBLICATION_STATUS.md", text)
            self.assertIn("2.048", text)
            self.assertIn("12.550", text)
            self.assertIn("519.4", text)
            self.assertIn("0.684", text)
        self.assertNotIn("0.63", russian)
        self.assertNotRegex(russian, r"(?i)NANOGrav.*(?:✅|подтвержд|охватывает амплитуду)")
        self.assertNotIn("p\\text{-value} < 10^{-14}", english)
        self.assertNotIn("p\\text{-value} < 10^{-14}", russian)
        self.assertNotIn("strictly derived from QCD and match observations", english)
        self.assertNotIn("теория успешно мимикрирует под ОТО", russian)
        self.assertNotIn("verification/rhic_bell_test.py", english)
        self.assertNotIn("verification/rhic_bell_test.py", russian)
        self.assertIn("retired", english.lower())
        self.assertIn("Отозвано", russian)

    def test_english_dashboards_are_machine_marked_non_evidence(self):
        for name in ("nvg_3d_viz.html", "nvg_3d_viz_v2.html"):
            source = (ROOT / "visualization" / name).read_text(encoding="utf-8")
            match = re.search(
                r'<script\s+id="nvg-provenance"\s+type="application/json">(.*?)</script>',
                source,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(match, name)
            payload = json.loads(match.group(1))
            self.assertEqual(payload["status"], "ILLUSTRATIVE_NON_EVIDENCE")
            self.assertFalse(payload["independent"])
            self.assertIsNone(payload["source_script"])
            self.assertIn('data-nvg-status="ILLUSTRATIVE_NON_EVIDENCE"', source)
            self.assertIn("Illustrative animation only", source)
            self.assertNotIn("Ironclad Discoveries", source)

    def test_tracked_txt_snapshots_have_provenance_and_no_favorable_verdict(self):
        expected = {
            "nvg_dark_photon_kinetic_mixing_output.txt": "nvg_dark_photon_kinetic_mixing.py",
            "nvg_em_response_derivation_output.txt": "nvg_em_response_derivation.py",
            "nvg_em_response_higher_order_output.txt": "nvg_em_response_higher_order.py",
            "nvg_vacuum_w_field_derivation_output.txt": "nvg_vacuum_w_field_derivation.py",
        }
        for filename, producer in expected.items():
            text = (HERE / filename).read_text(encoding="utf-8")
            self.assertIn("# Provenance:", text, filename)
            self.assertIn(f"verification/{producer}", text, filename)
            self.assertIn("NON_EVIDENCE", text, filename)
            self.assertNotRegex(text, r"(?i)consistent with ALL standard")
            self.assertNotRegex(text, r"(?i)PREDICTION VERIFIED|verified successfully")

    def test_publication_banner_covers_every_tracked_article_and_private_doc(self):
        import subprocess

        tracked = [
            Path(line)
            for line in subprocess.check_output(
                ["git", "ls-files", "article/*.md", ".docs/*.md"], cwd=ROOT, text=True
            ).splitlines()
        ]
        self.assertEqual(len(tracked), 11)
        for relative in tracked:
            path = ROOT / relative
            first = path.read_text(encoding="utf-8").splitlines()[:3]
            banner = " ".join(first)
            self.assertRegex(banner, r"(?i)publication status|статус публикации")
            self.assertIn("PUBLICATION_STATUS.md", banner)

    def test_publication_status_names_current_and_historical_surfaces(self):
        text = (ROOT / "PUBLICATION_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("historical/publication material", text)
        self.assertIn("not as current computational evidence", text)
        self.assertIn("run_nvg_suite.py", text)
        self.assertIn("article TeX/PDF", text)


if __name__ == "__main__":
    unittest.main()
