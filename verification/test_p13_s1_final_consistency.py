"""Phase 13 contracts for the final public evidence boundaries (B1--B6)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _payload(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script\s+id="nvg-provenance"\s+type="application/json">(.*?)</script>',
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"{path} has no nvg-provenance payload")
    return json.loads(match.group(1))


class P13S1FinalConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = json.loads((HERE / "registry.json").read_text(encoding="utf-8"))
        self.artifacts = json.loads((HERE / "artifact_manifest.json").read_text(encoding="utf-8"))[
            "entries"
        ]

    def test_all_tracked_html_has_registered_machine_and_visible_status(self):
        html_paths = sorted((ROOT / "visualization").glob("*.html"))
        self.assertEqual(len(html_paths), 5)
        for path in html_paths:
            relative = path.relative_to(ROOT).as_posix()
            self.assertIn(relative, self.registry["entries"])
            self.assertIn(relative, self.artifacts)
            entry = self.registry["entries"][relative]
            self.assertEqual(entry["kind"], "artifact")
            self.assertEqual(entry["evidence_weight"], 0.0)
            source = path.read_text(encoding="utf-8")
            payload = _payload(path)
            self.assertIn(payload["status"], {"ILLUSTRATIVE_NON_EVIDENCE", "CONDITIONAL_IN_SAMPLE"})
            self.assertEqual(self.artifacts[relative]["status"], payload["status"])
            self.assertRegex(source, r'<meta\s+name="nvg-status"\s+content="(?:ILLUSTRATIVE_NON_EVIDENCE|CONDITIONAL_IN_SAMPLE)">')
            self.assertRegex(source, r'data-nvg-status="(?:ILLUSTRATIVE_NON_EVIDENCE|CONDITIONAL_IN_SAMPLE)"')
            if payload["status"] == "ILLUSTRATIVE_NON_EVIDENCE":
                self.assertFalse(payload.get("independent", True))
                self.assertIsNone(payload.get("source_script"))

    def test_russian_dashboard_has_no_untraced_favorable_panel(self):
        source = (ROOT / "visualization" / "nvg_3d_viz_v2_ru.html").read_text(encoding="utf-8")
        for marker in ("p=0.77", "p = 0.77", "данные уже сняты", "4.5σ", "контраст 65%", "m<sub>θ</sub> = 53"):
            self.assertNotIn(marker.lower(), source.lower())
        self.assertIn("runtime-результат и наблюдательное p не публикуются", source)
        self.assertIn("data-nvg-global-status=\"WITHHELD_NO_HELD_OUT_PRODUCER\"", source)

    def test_bilingual_public_boundaries_match_maintained_runtime(self):
        english = (ROOT / "README.md").read_text(encoding="utf-8")
        russian = (ROOT / "README_RU.md").read_text(encoding="utf-8")
        for text, historical_marker, catalogue_marker, lite_marker in (
            (english, "not current runtime evidence", "no machine-readable", "not the Planck likelihood"),
            (russian, "не текущее runtime-свидетельство", "нет машиночитаемого", "не likelihood Planck"),
        ):
            self.assertIn(historical_marker, text)
            self.assertIn(catalogue_marker, text)
            self.assertIn(lite_marker, text)
            self.assertNotRegex(text, r"(?i)(resolves? the (hyperon|early supermassive)|robust parameter-free resolution|решает парадокс сверхранних|разрешение гиперонной загадки.*безпараметр)")
        self.assertIn("Object-level JWST seeding is retired", english)
        self.assertIn("Объектное заявление JWST отозвано", russian)
        cmb_entry = self.registry["entries"]["verification/nvg_cmb_lowl_refit.py"]
        self.assertIn("lite-likelihood", cmb_entry["description"])
        self.assertIn("not the Planck likelihood", cmb_entry["description"])

    def test_g_mode_is_forecast_zero_weight_in_registry(self):
        entry = self.registry["entries"]["verification/nvg_ns_g_modes.py"]
        self.assertEqual(entry["role"], "forecast")
        self.assertEqual(entry["status"], "forward_only")
        self.assertEqual(entry["evidence_weight"], 0.0)
        self.assertIn("verification/test_p13_s1_final_consistency.py", entry["tests"])
        snapshot = self.registry["entries"]["verification/nvg_dark_photon_kinetic_mixing_output.txt"]
        self.assertEqual(snapshot["producer"], "verification/nvg_dark_photon_kinetic_mixing.py")
        self.assertIn("verification/test_p13_s1_final_consistency.py", snapshot["tests"])
        source = (HERE / "nvg_ns_g_modes.py").read_text(encoding="utf-8")
        self.assertIn("forecast rather than a confirmed", source)

    def test_gw_template_header_and_hash_match_publication_boundary(self):
        path = HERE / "data" / "nvg_gw_template.txt"
        source = path.read_text(encoding="utf-8")
        self.assertIn("# Provenance:", source)
        self.assertIn("NON_EVIDENCE", source)
        self.assertIn("nvg_gw_spectrum_template.py", source)
        publication = (ROOT / "PUBLICATION_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("audited deterministic TXT snapshots", publication)
        self.assertIn("data/nvg_gw_template.txt", publication)
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            json.loads((HERE / "data" / "provenance.json").read_text(encoding="utf-8"))["entries"]["nvg_gw_template.txt"]["sha256"],
        )

    def test_dark_photon_cli_and_registry_are_conditional(self):
        entry = self.registry["entries"]["verification/nvg_dark_photon_kinetic_mixing.py"]
        self.assertEqual(entry["role"], "forecast")
        self.assertEqual(entry["status"], "forward_only")
        self.assertEqual(entry["evidence_weight"], 0.0)
        self.assertIn("verification/test_p13_s1_final_consistency.py", entry["tests"])
        completed = subprocess.run(
            [sys.executable, str(HERE / "nvg_dark_photon_kinetic_mixing.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("CONDITIONAL_BENCHMARK_NO_COMPLETE_LIKELIHOOD", completed.stdout)
        self.assertIn("no complete", completed.stdout.lower())
        self.assertNotIn("ALL standard", completed.stdout)
        self.assertNotIn("consistent with ALL", completed.stdout)


if __name__ == "__main__":
    unittest.main()
