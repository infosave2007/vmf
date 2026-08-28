"""Semantic regression checks for the P2-S2 model-chain repairs."""

from __future__ import annotations

import importlib
import unittest


class ModelChainIntegrityTests(unittest.TestCase):
    def test_bbn_radius_consumers_share_computed_eos_state(self):
        mod = importlib.import_module("verification.nvg_bbn_reionization")
        self.assertGreater(mod.RESULTS["eos"]["m_max"], 1.4)
        self.assertAlmostEqual(
            mod.RESULTS["postmerger"]["r_16_km"], mod.RESULTS["eos"]["r_16"], places=12
        )
        self.assertEqual(
            mod.RESULTS["direct_urca"]["status"],
            "ASSUMPTION_SENSITIVITY_NOT_INDEPENDENT_EVIDENCE",
        )

    def test_cpl_route_is_not_claimed_as_a_fit(self):
        mod = importlib.import_module("verification.nvg_gravitational_waves_tests")
        cyclic = mod.RESULTS["cyclic_dark_energy"]
        self.assertIsNone(cyclic["cpl_parameters"])
        self.assertIn("NO_INDEPENDENT_CPL", cyclic["status"])
        self.assertLess(mod.RESULTS["tensor_ratio"]["r_scale_estimate"], mod.RESULTS["tensor_ratio"]["bound_input"])

    def test_i_love_and_echo_use_runtime_sources(self):
        mod = importlib.import_module("verification.nvg_iloveq_gw_echoes")
        self.assertGreater(mod.RESULTS["i_love"]["lambda"], 0.0)
        self.assertIn("NO_INDEPENDENT_I", mod.RESULTS["i_love"]["status"])
        self.assertTrue(all(row["source"].endswith("calculate_echo_delay") for row in mod.RESULTS["echoes"]["rows"]))

    def test_cooling_and_pbh_outputs_are_explicitly_limited(self):
        cooling = importlib.import_module("verification.nvg_cooling_dark_matter")
        self.assertEqual(cooling.RESULTS["cooling"]["fraction_grid"], [0.0, 0.5, 1.0])
        self.assertIn("SENSITIVITY_ONLY", cooling.RESULTS["cooling"]["status"])
        self.assertIn("NO_ABUNDANCE", cooling.RESULTS["pbh"]["status"])

    def test_cmb_and_continuity_scripts_do_not_claim_continuity(self):
        cmb = importlib.import_module("verification.nvg_cmb_smbh_cyclic")
        continuity = importlib.import_module("verification.nvg_pbh_continuity_test")
        self.assertIn("NO_ABUNDANCE", cmb.RESULTS["pbh"]["status"])
        self.assertIn("NO_SMBH_CONTINUITY", continuity.RESULTS["status"])
        masses = [row["mass_msun"] for row in continuity.RESULTS["rows"]]
        self.assertTrue(all(a < b for a, b in zip(masses, masses[1:])))


if __name__ == "__main__":
    unittest.main()
