"""Semantic/data-flow checks for the P10-S1 legacy repair boundary."""

from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_advanced_calculations as advanced
import nvg_advanced_observables_I as obs_i
import nvg_advanced_observables_II as obs_ii
import nvg_advanced_observables_III as obs_iii
import nvg_chime_frb_check as chime
import nvg_eos_fork_a as fork_a
import nvg_eos_fork_b as fork_b
import nvg_eos_fork_b_nl as fork_nl
import nvg_hadrons_magnetic_fields as hadrons
import nvg_hyperon_puzzle as hyperon
import nvg_hyperon_puzzle_solution as hyperon_solution
import nvg_nanograv_background as nanograv
import nvg_new_directions_verification as directions
import nvg_ns_g_modes as g_modes
import nvg_ns_mass_bound as mass_bound
import nvg_pulsar_population_test as pulsars


class P10S1LegacySemanticTests(unittest.TestCase):
    def test_advanced_observables_i_uses_canonical_ns_and_retires_missing_rows(self):
        state = obs_i.compute_observables()
        from nvg_joint_ns_inference import compute_nvg_predictions

        predictions, _ = compute_nvg_predictions()
        self.assertAlmostEqual(state["ns_redshift"]["mass_msun"], predictions["M_max"], places=12)
        self.assertAlmostEqual(state["ns_redshift"]["radius_km"], predictions["R_1.4"], places=12)
        self.assertEqual(state["postmerger_f_peak"]["value_khz"], None)
        self.assertIn("RETIRED", state["postmerger_f_peak"]["status"])
        self.assertEqual(state["hades"]["independent_data"], False)

    def test_forward_ledgers_have_no_observational_confirmation(self):
        self.assertIn("NO_CMB_LIKELIHOOD", obs_ii.compute_observables()["cmb_spectrum"]["status"])
        self.assertIn("NO_EHT_LIKELIHOOD", obs_ii.compute_observables()["eht_shadow"]["status"])
        from nvg_pbh_mass_spectrum import get_pbh_mass

        self.assertEqual(obs_ii.pbh_mass(10), get_pbh_mass(10))
        state = obs_iii.compute_observables()
        self.assertIn("NO_GRB_LIKELIHOOD", state["lorentz"]["status"])
        self.assertIn("RETIRED", state["cooling"]["status"])
        self.assertGreater(len(state["meson"]["rows"]), 0)

    def test_mock_population_claims_are_retired_without_p_values(self):
        self.assertIsNone(chime.run_chime_frb_check()["p_value"])
        self.assertIsNone(pulsars.run_pulsar_population_test()["falsifiers"])
        self.assertIn("RETIRED", directions.run_pulsar_verification()["status"])
        self.assertIsNone(advanced.run_frb_dm_check()["p_value"])
        self.assertIn("RETIRED", advanced.run_jwst_smbh_check()["status"])

    def test_new_directions_uses_real_gwtc_mass_only_for_forward_delay(self):
        result = directions.run_echo_verification()
        self.assertGreater(result["mass_msun"], 0.0)
        self.assertGreater(result["delay_s"], 0.0)
        self.assertIn("NO_STRAIN_LIKELIHOOD", result["status"])

    def test_hyperon_and_gmode_results_are_runtime_or_explicitly_limited(self):
        threshold = hyperon.compute_thresholds(points=41)
        self.assertIsNotNone(threshold["lambda_onset_n0"])
        self.assertIn("NO_COMPLETE_HYPERON_LIKELIHOOD", threshold["status"])
        solved = hyperon_solution.compute_thresholds(points=41)
        self.assertIsNotNone(solved["thresholds_n0"]["Lambda"])
        gstate = g_modes.compute_gmode_state()
        self.assertEqual(len(gstate["period_ms"]), 3)
        self.assertIn("FORECAST", gstate["status"])

    def test_fork_hades_templates_use_raw_poles_without_anchor(self):
        for module in (fork_a, fork_b, fork_nl):
            with self.subTest(module=module.__name__):
                result = module.hades_shape_summary(700.0)
                self.assertAlmostEqual(result["pole_mev"], 700.0)
                self.assertGreater(result["centroid_mev"], 0.0)
                self.assertIn("NO_HADES_LIKELIHOOD", result["status"])

    def test_mass_bound_and_nanograv_are_not_retuned_matches(self):
        bound = mass_bound.compute_report()
        self.assertIn("NO_EOS_RETUNE", bound["status"])
        self.assertIn("CONDITIONAL_IN_SAMPLE", bound["canonical"]["status"])
        background = nanograv.run_nanograv_verification()
        self.assertIn("RETIRED", background["status"])
        self.assertGreater(background["rows"][0]["deficit_factor"], 100.0)

    def test_extended_summary_uses_canonical_ns_and_limited_statuses(self):
        state = hadrons.compute_summary()
        from nvg_joint_ns_inference import compute_nvg_predictions

        canonical, _ = compute_nvg_predictions()
        self.assertAlmostEqual(state["canonical_ns"]["M_max"], canonical["M_max"], places=12)
        self.assertIn("NO_MHD_SOLVER", state["magnetic_seed"]["status"])
        self.assertIn("NO_BARYON_NUMBER", state["baryon_asymmetry"]["status"])
        self.assertIn("NO_ENTROPY_TRANSFER_DYNAMICS", state["entropy"]["status"])
        suite = advanced.compute_suite()
        self.assertIsNone(suite["neutrino"]["mass_ev"])
        self.assertIn("RETIRED", suite["pta"]["status"])

    def test_owned_source_does_not_reintroduce_seeded_catalog_or_hades_anchor(self):
        owned = (
            "nvg_advanced_calculations.py",
            "nvg_chime_frb_check.py",
            "nvg_pulsar_population_test.py",
            "nvg_new_directions_verification.py",
            "nvg_eos_fork_a.py",
            "nvg_eos_fork_b.py",
            "nvg_eos_fork_b_nl.py",
        )
        for name in owned:
            source = (HERE / name).read_text(encoding="utf-8")
            self.assertNotRegex(source, r"np\.random\.(seed|normal|choice)")
        for name in ("nvg_eos_fork_a.py", "nvg_eos_fork_b.py", "nvg_eos_fork_b_nl.py"):
            source = (HERE / name).read_text(encoding="utf-8")
            self.assertNotIn("6" + "3.0", source)
            self.assertNotIn("2" + "0.1", source)

    def test_cli_status_output_is_semantic(self):
        for name, marker in (("nvg_chime_frb_check.py", "RETIRED_NO_CHIME"),
                             ("nvg_pulsar_population_test.py", "RETIRED_NO_ATNF"),
                             ("nvg_nanograv_background.py", "RETIRED_PBH_NANOGRAV")):
            completed = subprocess.run([sys.executable, str(HERE / name)], capture_output=True,
                                       text=True, cwd=str(HERE), check=False)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn(marker, completed.stdout)


if __name__ == "__main__":
    unittest.main()
