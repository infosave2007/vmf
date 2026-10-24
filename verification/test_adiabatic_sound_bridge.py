"""Focused integrity tests for the no-fit adiabatic sound-speed bridge."""
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

import nvg_adiabatic_sound_bridge as bridge  # noqa: E402


class AdiabaticSoundBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = HERE / "nvg_adiabatic_sound_bridge_results.json"
        cls.result = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_saved_live_result_validates(self):
        self.assertTrue(bridge.validate_result(self.result))
        self.assertEqual(self.result["coverage"], {"expected_rows": 8, "primary_rows": 8, "refined_rows": 8})
        self.assertFalse(self.result["fit_used"])
        self.assertTrue(self.result["all_controls_pass"])

    def test_all_speeds_are_causal_and_direct_cross_check_agrees(self):
        for row in self.result["primary_rows"]:
            adiabatic = row["adiabatic"]
            cs2 = float(adiabatic["cs2_implicit"])
            direct = float(adiabatic["cs2_direct_isentrope"])
            self.assertGreater(cs2, 0.0)
            self.assertLess(cs2, 1.0)
            self.assertAlmostEqual(cs2, direct, delta=4e-4 * max(cs2, direct))
            self.assertTrue(row["gates"]["pass"])

    def test_rehashed_acausal_mutation_is_rejected(self):
        changed = copy.deepcopy(self.result)
        changed["primary_rows"][0]["adiabatic"]["cs2_implicit"] = "1.01"
        changed.pop("integrity_sha256")
        changed["integrity_sha256"] = hashlib.sha256(bridge._payload_bytes(changed)).hexdigest()
        self.assertFalse(bridge.validate_result(changed))

    def test_producer_does_not_read_a_saved_result(self):
        source = (HERE / "nvg_adiabatic_sound_bridge.py").read_text(encoding="utf-8")
        build_slice = source.split("def build_result", 1)[1].split("def validate_result", 1)[0]
        self.assertNotIn("results.json", build_slice)
        self.assertIn("no_saved_results_as_inputs", source)


if __name__ == "__main__":
    unittest.main()
