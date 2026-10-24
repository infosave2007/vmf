"""Focused integrity tests for the static-to-dynamic identifiability audit."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_dynamic_response_identifiability as audit  # noqa: E402


class DynamicIdentifiabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = HERE / "nvg_dynamic_response_identifiability_results.json"
        cls.result = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_saved_live_result_validates(self):
        self.assertTrue(audit.validate_result(self.result))
        self.assertEqual(self.result["coverage"], {"state_count": 8, "point_count": 32})
        self.assertFalse(self.result["minimal_completion"]["real_frequency_prediction"])
        self.assertFalse(self.result["minimal_completion"]["device_or_power_prediction"])

    def test_relative_shape_is_recomputed(self):
        for state in self.result["states"]:
            points = state["points"]
            self.assertEqual([point["q_MeV"] for point in points], ["0", "50", "100", "200"])
            self.assertAlmostEqual(float(points[0]["relative_frequency_if_constant_inertia"]), 1.0, places=13)
            for point in points:
                self.assertAlmostEqual(
                    float(point["relative_frequency_if_constant_inertia"]),
                    float(point["relative_frequency_from_chi"]),
                    places=12,
                )

    def test_rehashed_dynamic_claim_mutation_is_rejected(self):
        changed = copy.deepcopy(self.result)
        changed["states"][0]["points"][2]["relative_frequency_if_constant_inertia"] = "1.0"
        changed.pop("integrity_sha256")
        changed["integrity_sha256"] = hashlib.sha256(audit._payload_bytes(changed)).hexdigest()
        self.assertFalse(audit.validate_result(changed))

    def test_source_does_not_read_a_result_file(self):
        source = (HERE / "nvg_dynamic_response_identifiability.py").read_text(encoding="utf-8")
        build_slice = source.split("def build_result", 1)[1].split("def validate_result", 1)[0]
        self.assertNotIn("results.json", build_slice)
        self.assertIn("no_saved_results_as_inputs", source)


if __name__ == "__main__":
    unittest.main()
