"""Tests for the live EOS-to-static-q=0 closure audit."""
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

import nvg_thermo_static_zero_mode_closure as audit  # noqa: E402


class ThermoStaticZeroModeClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = HERE / "nvg_thermo_static_zero_mode_results.json"
        cls.result = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_live_result_validates_and_covers_all_fixed_states(self):
        self.assertTrue(audit.validate_result(self.result))
        self.assertEqual(self.result["coverage"], {"expected_rows": 8, "primary_rows": 8, "refined_rows": 8})
        self.assertTrue(self.result["all_controls_pass"])
        self.assertLess(float(self.result["controls"]["primary_cross_sector_identity"]["max_relative_error"]), 2e-7)
        self.assertLess(float(self.result["controls"]["refined_cross_sector_identity"]["max_relative_error"]), 2e-7)
        self.assertLess(float(self.result["controls"]["quadrature_and_root_refinement"]["max_relative_change"]), 3e-5)
        for row in self.result["primary_rows"] + self.result["refined_rows"]:
            self.assertTrue(row["checks"]["pass"])
            self.assertTrue(row["state"]["locally_stable"])

    def test_rehashed_susceptibility_mutation_is_rejected(self):
        changed = copy.deepcopy(self.result)
        changed["primary_rows"][0]["static_q0_side"]["chi_static_MeV2"] = "1.0"
        changed.pop("integrity_sha256")
        changed["integrity_sha256"] = hashlib.sha256(audit._payload_bytes(changed)).hexdigest()
        self.assertFalse(audit.validate_result(changed))

    def test_producer_does_not_use_saved_results_as_inputs(self):
        source = (HERE / "nvg_thermo_static_zero_mode_closure.py").read_text(encoding="utf-8")
        build_slice = source.split("def _build_result", 1)[1].split("def build_result", 1)[0]
        self.assertNotIn("results.json", build_slice)
        self.assertIn("no_saved_results_as_inputs", source)


if __name__ == "__main__":
    unittest.main()
