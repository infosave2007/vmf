#!/usr/bin/env python3
"""Pytest suite for Bridge 6: Pre-Registered Envelope & Blind 6-Nucleus Dirac-Hartree."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

VERIF_DIR = Path(__file__).resolve().parent
if str(VERIF_DIR) not in sys.path:
    sys.path.insert(0, str(VERIF_DIR))

import nvg_unified_calibration_envelope_probe as b6  # noqa: E402


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(b6.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))


def test_validate_live_artifact() -> None:
    ok, errs = b6.validate_artifact(b6.DEFAULT_OUTPUT_PATH)
    assert ok, errs


def test_evidence_weight_and_zero_refit(payload: dict) -> None:
    assert payload["evidence_weight"] == 0.0
    assert payload["provenance"]["zero_finite_nucleus_fitting"] is True
    assert payload["status"] == b6.STATUS_PASS
    assert all(payload["gates"].values())


def test_stage_a_grid_complete_and_bounded(payload: dict) -> None:
    sa = payload["stage_a_saturation_envelope"]
    assert len(sa["all_grid_rows"]) == 32
    assert all(r["global_stability_pass"] and r["causality_pass"] for r in sa["all_grid_rows"])
    assert len(sa["pareto_frontier"]) >= 1


def test_honest_primary_gate_reporting(payload: dict) -> None:
    sa = payload["stage_a_saturation_envelope"]
    prim = [r for r in sa["all_grid_rows"] if r["passes_primary_gate"]]
    assert sa["primary_gate_satisfied"] == (len(prim) > 0)
    assert sa["primary_survivor_count"] == len(prim)


def test_selected_candidate_deterministic_and_bounded(payload: dict) -> None:
    cand = payload["stage_a_saturation_envelope"]["selected_candidate"]
    assert cand["global_stability_pass"] is True
    assert cand["causality_pass"] is True
    assert 0.59 <= cand["M_eff_over_M_N"] <= 0.62


def test_both_models_evaluate_all_six_nuclei(payload: dict) -> None:
    sb = payload["stage_b_blind_six_nuclei_dirac_hartree"]
    assert sb["nuclei_evaluated"] == ["O16", "Ca40", "Ca48", "Zr90", "Sn132", "Pb208"]
    for branch_key in ("cycle1_baseline", "stage_a_candidate"):
        names = [r["nucleus"] for r in sb[branch_key]["nuclei"]]
        assert names == ["O16", "Ca40", "Ca48", "Zr90", "Sn132", "Pb208"]


def test_zr90_and_sn132_exact_particle_counts_and_energy_identity(payload: dict) -> None:
    sb = payload["stage_b_blind_six_nuclei_dirac_hartree"]
    for branch_key in ("cycle1_baseline", "stage_a_candidate"):
        for r in sb[branch_key]["nuclei"]:
            assert r["converged"] is True
            assert r["particle_rel_err"] < 1.0e-6
            assert r["two_energy_formula_diff_MeV"] < 1.0e-3
            assert r["max_field_residual_rel"] < 1.0e-6


def test_positive_neutron_skins_heavy_nuclei(payload: dict) -> None:
    sb = payload["stage_b_blind_six_nuclei_dirac_hartree"]
    for branch_key in ("cycle1_baseline", "stage_a_candidate"):
        by_name = {r["nucleus"]: r for r in sb[branch_key]["nuclei"]}
        assert by_name["Ca48"]["Delta_R_np_fm"] > 0.10
        assert by_name["Zr90"]["Delta_R_np_fm"] > 0.04
        assert by_name["Sn132"]["Delta_R_np_fm"] > 0.18
        assert by_name["Pb208"]["Delta_R_np_fm"] > 0.15


def test_spin_orbit_deepens_for_lighter_mstar_candidate(payload: dict) -> None:
    sb = payload["stage_b_blind_six_nuclei_dirac_hartree"]
    base_ca48 = next(r for r in sb["cycle1_baseline"]["nuclei"] if r["nucleus"] == "Ca48")
    cand_ca48 = next(r for r in sb["stage_a_candidate"]["nuclei"] if r["nucleus"] == "Ca48")
    assert cand_ca48["spin_orbit_MeV"]["1f"]["neutron_MeV"] > base_ca48["spin_orbit_MeV"]["1f"]["neutron_MeV"]


def test_ca48_ca40_charge_shift_improved_over_w8(payload: dict) -> None:
    sb = payload["stage_b_blind_six_nuclei_dirac_hartree"]
    for branch_key in ("cycle1_baseline", "stage_a_candidate"):
        shift = sb[branch_key]["summary_metrics"]["Delta_R_ch_Ca48_minus_Ca40_fm"]
        assert abs(shift) < 0.05  # far smaller than W8 TF's +0.166 fm


def test_stage_a_reproducibility_live() -> None:
    r1 = b6.evaluate_stage_a_point(0.160, 0.61, 56.0)
    r2 = b6.evaluate_stage_a_point(0.160, 0.61, 56.0)
    assert r1["M_max_Msun"] == pytest.approx(r2["M_max_Msun"], rel=1e-13)
    assert r1["Lambda_1_4"] == pytest.approx(r2["Lambda_1_4"], rel=1e-13)


def test_fail_closed_rejects_tampered_json(tmp_path: Path, payload: dict) -> None:
    bad = json.loads(json.dumps(payload))
    bad["evidence_weight"] = 0.5
    p1 = tmp_path / "bad_ew.json"
    p1.write_text(json.dumps(bad), encoding="utf-8")
    ok, _ = b6.validate_artifact(p1)
    assert not ok

    bad2 = json.loads(json.dumps(payload))
    bad2["stage_a_saturation_envelope"]["selected_candidate"]["b6_MeV_minus2"] *= 2.0
    p2 = tmp_path / "bad_b6.json"
    p2.write_text(json.dumps(bad2), encoding="utf-8")
    ok2, _ = b6.validate_artifact(p2)
    assert not ok2
