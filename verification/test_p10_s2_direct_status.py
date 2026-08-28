"""Semantic/data-flow tests for the P10-S2 direct-status repair boundary."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_baryogenesis_bsm_closure as baryogenesis
import nvg_baryon_asymmetry as baryon
import nvg_cas_a_cooling_curve as cas_a
import nvg_cyclic_cosmology as cyclic
import nvg_detector_forward_model as detector
import nvg_dm_direct_detection as direct_detection
import nvg_dna_chirality as dna
import nvg_hades_dielectron_sim as hades
import nvg_hayward_evaporation as hayward
import nvg_ds_core_oscillations as ds_core
import nvg_em_maxwell_decoherence as em_response
import nvg_higgs_mass_shift as higgs
import nvg_magnetar_eos as magnetar
import nvg_neutrinoless_dbeta as neutrinoless
import nvg_perihelion_shift as perihelion
import nvg_pbh_dark_matter as pbh_dm
import nvg_pbh_jwst_seeds as jwst
import nvg_pbh_mass_spectrum as pbh_ladder
import nvg_sgr_frb_rate as sgr_frb
import nvg_sgr_temperature as sgr_temperature
import nvg_speed_of_sound_bayesian as speed_bayes
import nvg_speed_of_sound_curve as speed_curve
import nvg_unified_field_equations as unified
import nvg_verification_suite as suite
import nvg_wd_cooling as wd


class P10S2DirectStatusTests(unittest.TestCase):
    def test_detector_and_hades_reuse_canonical_mass_producer(self):
        from nvg_fair_hades_link import in_medium_mass, n_0

        expected = in_medium_mass(detector.MASS_VAC, detector.MASS_CURRENT, 2.0 * n_0)
        state = detector.compute_forward_model(mass_grid=np.linspace(300.0, 1000.0, 120))
        self.assertAlmostEqual(state["mass_medium_mev"], expected, places=12)
        self.assertEqual(state["evidence_status"], "FORWARD_MODEL_ONLY")
        self.assertIsNone(state["observed_likelihood"])

        hstate = hades.compute_simulation(np.linspace(300.0, 1000.0, 120))
        self.assertAlmostEqual(hades.get_in_medium_mass(2.0 * n_0), expected, places=12)
        self.assertEqual(hstate["evidence_status"], "FORWARD_MODEL_ONLY")
        self.assertIsNone(hstate["observed_likelihood"])
        self.assertTrue(np.all(np.isfinite(hstate["spectra"]["vmf"])))

    def test_baryogenesis_is_not_closed_by_a_target_anchor(self):
        state = baryon.calculate_baryon_asymmetry()
        self.assertEqual(state["evidence_status"], "RETIRED_MISSING_BARYOGENESIS_SOURCE")
        self.assertIsNone(state["observed_likelihood"])
        self.assertNotAlmostEqual(state["eta_proxy"], state["eta_observed"], places=12)
        self.assertTrue(state["missing_components"])
        closure = baryogenesis.compute_channels()
        self.assertIn("RETIRED", closure["evidence_status"])
        self.assertNotIn("0.158", (HERE / "nvg_baryon_asymmetry.py").read_text())

    def test_model_statuses_and_canonical_ladders(self):
        channels = direct_detection.compute_all_channels()
        self.assertEqual(channels["evidence_status"], "MODEL_SENSITIVITY_ONLY")
        self.assertIsNone(channels["observed_likelihood"])
        self.assertIn("LZ", direct_detection.experimental_limits())

        self.assertEqual(pbh_ladder.get_pbh_mass(10), 0.38 * 4.0**10)
        pstate = pbh_dm.compute_spectrum(cycles=[-21, 0, 10])
        self.assertEqual(pstate["evidence_status"], "CALIBRATED_GRID_NO_LIKELIHOOD")
        self.assertEqual(pstate["rows"][0]["evidence_status"], "CALIBRATED_GRID_NO_LIKELIHOOD")
        self.assertAlmostEqual(jwst.M_seed_nvg, pbh_ladder.get_pbh_mass(10), places=12)

    def test_canonical_eos_and_formal_statuses(self):
        sound = speed_curve.compute_speed_sound()
        self.assertEqual(sound["evidence_status"], "CANONICAL_MODEL_CONSISTENCY")
        self.assertLess(sound["max_cs2"], 1.0)
        bayes = speed_bayes.compute_comparison(save_plot=False)
        self.assertEqual(bayes["evidence_status"], "CONDITIONAL_LITERATURE_COMPARISON")
        self.assertIsNone(bayes["observed_likelihood"])
        eos_state = suite.compute_suite_state()
        self.assertEqual(eos_state["evidence_status"], "PROCESS_CHECKS_ONLY")
        self.assertIsNone(eos_state["observed_likelihood"])

    def test_all_remaining_runtime_surfaces_expose_their_boundary(self):
        with contextlib.redirect_stdout(io.StringIO()):
            ladder_state = pbh_ladder.main()
            hayward_state = hayward.main()
        states = (
            (sgr_temperature.compute_sgr_thermal(), "CALIBRATED_CONSISTENCY_ONLY"),
            (sgr_frb.compute_rate_scaling(), "MODEL_SENSITIVITY_ONLY"),
            (wd.compute_cooling_grid(), "MODEL_SENSITIVITY_ONLY"),
            (cas_a.compute_cas_a_state(), "CALIBRATED_MODEL_COMPARISON"),
            (higgs.compute_shift(), "MODEL_SENSITIVITY_ONLY"),
            (dna.compute_scales(), "RETIRED_MISSING_PVED_SOURCE"),
            (neutrinoless.compute_neutrino_observables(), "CONDITIONAL_MODEL_COMPARISON"),
            (perihelion.compute_perihelion(), "MODEL_SENSITIVITY_ONLY"),
            (magnetar.compute_magnetar_state(), "CANONICAL_MODEL_SENSITIVITY"),
            (cyclic.compute_cyclic_state(), "CALIBRATED_HORIZON_CHAIN"),
            (ladder_state, "THEORY_LADDER_ONLY"),
            (ds_core.compute_core_grid(), "FORWARD_MODEL_ONLY"),
            (unified.compute_unified_state(), "FORMAL_MODEL_SENSITIVITY"),
            (jwst.compute_seed_growth(), "CONDITIONAL_FORWARD_MODEL"),
            (em_response.compute_dielectric_table(), "FORMAL_EFT_SENSITIVITY"),
            (hayward_state, "MODEL_DERIVED_NO_EVAPORATION_LIKELIHOOD"),
        )
        for state, expected in states:
            self.assertEqual(state["evidence_status"], expected)
            self.assertIsNone(state["observed_likelihood"])
        self.assertNotEqual(cas_a.compute_cas_a_state()["slope_observed_K_per_yr"], -3650.0)

    def test_hayward_model_routes_horizonless_masses(self):
        self.assertIsNone(hayward.horizon(hayward.M_CRIT))
        pstate = pbh_dm.compute_spectrum(cycles=[-30, 0])
        self.assertTrue(all("horizonless" in row["horizon_class"] for row in pstate["rows"]))

    def test_scratch_endpoint_paths_fail_closed(self):
        spec = importlib.util.spec_from_file_location("scratch_urca", ROOT / "scratch" / "test_urca_tov.py")
        urca = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(urca)
        with self.assertRaises(ValueError):
            urca.interpolate_in_domain(-1.0, [0.0, 1.0], [0.0, 1.0])

        spec2 = importlib.util.spec_from_file_location("scratch_tidal", ROOT / "scratch" / "test_tidal_pure.py")
        tidal = importlib.util.module_from_spec(spec2)
        assert spec2.loader is not None
        with contextlib.redirect_stdout(io.StringIO()):
            spec2.loader.exec_module(tidal)
        eos = tidal.PureEOS()
        with self.assertRaises(ValueError):
            eos.get_eps(eos.p_arr[0] - 1.0)

    def test_cli_statuses_are_runtime_semantic(self):
        for name, marker in (
            ("nvg_detector_forward_model.py", "FORWARD_MODEL_ONLY"),
            ("nvg_hades_dielectron_sim.py", "FORWARD_MODEL_ONLY"),
            ("nvg_baryogenesis_bsm_closure.py", "RETIRED_MISSING_BARYOGENESIS_SOURCE"),
            ("nvg_verification_suite.py", "PROCESS_CHECKS_ONLY"),
        ):
            completed = subprocess.run(
                [sys.executable, str(HERE / name)],
                cwd=str(HERE),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn(marker, completed.stdout)
            self.assertNotIn("ALL DECLARED CRITERIA PASSED", completed.stdout)
            self.assertNotIn("PREDICTION VERIFIED", completed.stdout)


if __name__ == "__main__":
    unittest.main()
