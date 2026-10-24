"""Focused acceptance tests for the bounded grand-potential landscape scan."""
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

import nvg_grand_potential_landscape as landscape  # noqa: E402


class GrandPotentialLandscapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = HERE / "nvg_grand_potential_landscape_results.json"
        cls.result = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_saved_live_landscape_validates(self):
        self.assertTrue(landscape.validate_result(self.result))
        self.assertEqual(self.result["coverage"], {"expected_rows": 8, "primary_rows": 8, "refined_rows": 8})
        self.assertFalse(self.result["fit_used"])
        self.assertTrue(self.result["all_controls_pass"])
        self.assertIn("finite grid", " ".join(self.result["caveats"]))

    def test_each_fixed_model_state_has_one_scanned_pressure_maximum(self):
        for row in self.result["primary_rows"]:
            scan = row["scan"]
            self.assertEqual(scan["grid_points"], 58)
            self.assertEqual(scan["gap_sign_change_count"], 1)
            self.assertEqual(scan["gap_zero_bracket_count"], 1)
            self.assertEqual(len(scan["stationary_points"]), 1)
            root = scan["stationary_points"][0]
            self.assertTrue(root["pressure_local_maximum"])
            self.assertEqual((root["gap_sign_left"], root["gap_sign_right"]), ("negative", "positive"))
            self.assertLess(float(root["gap_residual_relative"]), 2e-20)
            self.assertLess(float(root["pressure_gradient_identity_relative"]), 1e-5)
            self.assertTrue(scan["single_interior_maximum_beats_boundaries"])

    def test_refinement_comparison_is_recomputed(self):
        self.assertTrue(all(item["pass"] for item in self.result["refinement"]))
        self.assertLess(
            max(float(item["max_relative_y_or_pressure_difference"]) for item in self.result["refinement"]),
            1e-15,
        )

    def test_rehashed_extra_phase_is_rejected(self):
        changed = copy.deepcopy(self.result)
        changed["primary_rows"][0]["scan"]["stationary_points"].append(
            copy.deepcopy(changed["primary_rows"][0]["scan"]["stationary_points"][0])
        )
        changed.pop("integrity_sha256")
        changed["integrity_sha256"] = hashlib.sha256(landscape._payload_bytes(changed)).hexdigest()
        self.assertFalse(landscape.validate_result(changed))

    def test_producer_never_loads_its_saved_result_as_input(self):
        source = (HERE / "nvg_grand_potential_landscape.py").read_text(encoding="utf-8")
        build_slice = source.split("def build_result", 1)[1].split("def validate_result", 1)[0]
        self.assertNotIn("results.json", build_slice)
        self.assertIn("no_saved_results_as_inputs", source)


if __name__ == "__main__":
    unittest.main()
