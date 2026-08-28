"""Semantic and artifact checks for the Phase 8 tracked visualizations."""

from __future__ import annotations

import json
import os
import re
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_nvg_suite


def _provenance_payload(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script\s+id="nvg-provenance"\s+type="application/json">(.*?)</script>',
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"{path} has no machine-readable provenance payload")
    return json.loads(match.group(1))


class P8S1VisualizationIntegrityTests(unittest.TestCase):
    def test_russian_dashboard_uses_current_runtime_values_and_status(self):
        path = ROOT / "visualization" / "nvg_3d_viz_v2_ru.html"
        source = path.read_text(encoding="utf-8")
        payload = _provenance_payload(path)
        runtime = run_nvg_suite.run_forward_model()

        self.assertEqual(payload["status"], "CONDITIONAL_IN_SAMPLE")
        self.assertFalse(payload["independent"])
        self.assertEqual(payload["held_out_observations"], [])
        self.assertEqual(payload["global_significance"], "WITHHELD_NO_HELD_OUT_PRODUCER")
        self.assertAlmostEqual(payload["values"]["M_max"], runtime["m_max"], places=10)
        self.assertAlmostEqual(payload["values"]["R_1.4"], runtime["r_14"], places=10)
        self.assertAlmostEqual(payload["values"]["Lambda_1.4"], runtime["lambda_14"], places=10)

        # Values are rendered from the payload at page load, not maintained as
        # a second static result table in the visible markup.
        for element_id in (
            "v-canonical-mmax",
            "v-canonical-r14",
            "v-canonical-lambda14",
            "v-canonical-status",
            "v-global-status",
            "v-canonical-source",
        ):
            self.assertIn(f'id="{element_id}"', source)
        self.assertIn("Number(NVG_RUNTIME_VALUES.M_max).toFixed(3)", source)
        self.assertIn("Number(NVG_RUNTIME_VALUES['R_1.4']).toFixed(3)", source)
        self.assertIn("Number(NVG_RUNTIME_VALUES['Lambda_1.4']).toFixed(1)", source)
        self.assertIn('data-nvg-status="CONDITIONAL_IN_SAMPLE"', source)
        self.assertIn('data-nvg-global-status="WITHHELD_NO_HELD_OUT_PRODUCER"', source)
        self.assertIn("verification/run_nvg_suite.py", source)
        self.assertIn("verification/fig_iloveq_universal_report.json", source)

        stale_markers = ("2.76/3", "p = 0.43", "8/8", "2.07", "12.49", "313")
        for marker in stale_markers:
            self.assertNotIn(marker.lower(), source.lower())

    def test_merger_lambda_is_explicitly_illustrative_non_evidence(self):
        path = ROOT / "visualization" / "nvg_ns_merger_3d.html"
        source = path.read_text(encoding="utf-8")
        payload = _provenance_payload(path)

        self.assertEqual(payload["status"], "ILLUSTRATIVE_NON_EVIDENCE")
        self.assertIsNone(payload["value"])
        self.assertIsNone(payload["source_script"])
        self.assertIn("animation-only surrogate", payload["semantics"])
        self.assertIn('data-nvg-status="ILLUSTRATIVE_NON_EVIDENCE"', source)
        self.assertIn("illustrative only", source.lower())
        self.assertIn("no runtime value", source.lower())
        self.assertNotIn("470", source)
        self.assertIsNone(_provenance_payload(path)["value"])
        self.assertNotRegex(source, r"(?i)(?:lambda|Λ)\s*=\s*[-+]?\d")


if __name__ == "__main__":
    unittest.main()
