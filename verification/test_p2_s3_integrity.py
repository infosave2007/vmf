"""Semantic regression tests for the P2-S3 integrity boundary."""

import json
from pathlib import Path

import numpy as np
import pytest

import nvg_dark_photon_observables as dark_photon
import nvg_magnetar_population_scan as population
import nvg_observational_data_fit as observational
import nvg_eos_beta_checked as beta_checked
import nvg_eos_beta_saturated_vector as beta_saturated
import nvg_eos_proof_checked as proof_checked
import nvg_echo_timeslide_background as timeslide


def test_missing_event_mass_fails_closed():
    with pytest.raises(ValueError, match="missing valid catalog remnant mass"):
        timeslide.require_mass(None, "missing-event")
    with pytest.raises(ValueError):
        timeslide.require_mass(float("nan"), "nan-event")


def test_catalog_provenance_manifest_is_machine_readable():
    manifest_path = Path(timeslide.PROVENANCE_PATH)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["entries"]["gwtc_events.csv"]["trust_status"] == "verified_public"
    assert manifest["entries"]["planck2018_tt_full.txt"]["trust_status"] == "unknown_provenance"
    assert manifest["entries"]["nvg_gw_template.txt"]["trust_status"] == "generated_unverified"


def test_fallback_sample_is_explicit_non_evidence(monkeypatch):
    def fail_fetch():
        raise OSError("offline")

    monkeypatch.setattr(population, "parse_mcgill_magnetars", fail_fetch)
    sample, note = population.build_sample()
    assert sample
    assert "NON-EVIDENCE" in note


def test_observational_surfaces_report_calibration_status():
    cmb = observational.run_cmb_verification()
    desi = observational.run_desi_verification()
    cooling = observational.run_cooling_verification()
    assert cmb["evidence_status"] == "not_independent_prediction"
    assert desi["evidence_status"] == "unverified_input_comparison"
    assert cooling["evidence_status"] == "unverified_input_comparison"


def test_dark_photon_uses_real_dependency_and_conditional_math():
    assert dark_photon.solve_gap.__module__ == "nvg_em_response_derivation"
    assert dark_photon.decay_lifetime(0.0) == float("inf")


@pytest.mark.parametrize("module", [beta_checked, beta_saturated, proof_checked])
def test_eos_tov_interpolation_rejects_out_of_bounds(module, monkeypatch):
    def probe(eps_of_p, _p_c):
        # An endpoint-clamping interpolator would return silently here.
        eps_of_p(3.0)
        return 1.0, 10.0

    monkeypatch.setattr(module, "solve_tov", probe)
    masses, radii = module.tov_scan(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    assert masses is None and radii is None
