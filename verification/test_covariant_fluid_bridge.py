"""Focused integrity tests for the local covariant-fluid reconstruction."""
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

import nvg_covariant_fluid_bridge as bridge  # noqa: E402


class CovariantFluidBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = HERE / "nvg_covariant_fluid_bridge_results.json"
        cls.result = json.loads(cls.path.read_text(encoding="utf-8"))

    def test_saved_live_result_validates(self):
        self.assertTrue(bridge.validate_result(self.result))
        self.assertEqual(self.result["coverage"], {"expected_rows": 8, "primary_rows": 8, "refined_rows": 8})
        self.assertFalse(self.result["fit_used"])
        self.assertTrue(self.result["all_controls_pass"])

    def test_eight_states_satisfy_local_fluid_identities(self):
        for row in self.result["primary_rows"]:
            action = row["fluid_action_reconstruction"]
            gates = row["gates"]
            self.assertTrue(gates["local_stable"])
            self.assertTrue(gates["positive_causal_sound"])
            self.assertLess(float(gates["pressure_identity_relative"]), 4e-4)
            self.assertLess(float(gates["enthalpy_gradient_relative"]), 4e-4)
            self.assertLess(float(gates["grand_canonical_euler_relative"]), 4e-4)
            self.assertAlmostEqual(float(action["cs2_from_action"]), float(action["cs2_from_direct_isentrope"]), places=13)

    def test_rehashed_pressure_identity_mutation_is_rejected(self):
        changed = copy.deepcopy(self.result)
        changed["primary_rows"][0]["fluid_action_reconstruction"]["pressure_from_n_epsilon_n_minus_epsilon_MeV4"] = "0"
        changed.pop("integrity_sha256")
        changed["integrity_sha256"] = hashlib.sha256(bridge._payload_bytes(changed)).hexdigest()
        self.assertFalse(bridge.validate_result(changed))

    def test_producer_does_not_read_a_saved_result(self):
        source = (HERE / "nvg_covariant_fluid_bridge.py").read_text(encoding="utf-8")
        build_slice = source.split("def build_result", 1)[1].split("def validate_result", 1)[0]
        self.assertNotIn("results.json", build_slice)
        self.assertIn("no_saved_results_as_inputs", source)


if __name__ == "__main__":
    unittest.main()
