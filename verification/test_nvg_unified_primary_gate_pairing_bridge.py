#!/usr/bin/env python3
"""Pytest suite for Bridge 8: Widened zeta_RMF Primary Gate, RH-BCS Pairing & Cross-Bridge Check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

VERIF_DIR = Path(__file__).resolve().parent
if str(VERIF_DIR) not in sys.path:
    sys.path.insert(0, str(VERIF_DIR))

import nvg_unified_primary_gate_pairing_bridge as b8  # noqa: E402


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(b8.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))


def test_validate_live_artifact() -> None:
    ok, errs = b8.validate_artifact(b8.DEFAULT_OUTPUT_PATH)
    assert ok, errs


def test_status_evidence_weight_and_gates(payload: dict) -> None:
    assert payload["schema_version"] == b8.SCHEMA_VERSION
    assert payload["bridge_id"] == b8.BRIDGE_ID
    assert payload["status"] == b8.STATUS_PASS
    assert payload["evidence_weight"] == 0.0
    assert payload["provenance"]["zero_finite_nucleus_fitting"] is True
    assert all(payload["gates"].values())


def test_stage_a_widened_grid_clears_primary_gate(payload: dict) -> None:
    sa = payload["stage_a_widened_primary_gate"]
    assert len(sa["all_grid_rows"]) == 54
    assert sa["primary_gate_satisfied"] is True
    assert sa["primary_survivor_count"] >= 2
    assert all(r["global_stability_pass"] and r["causality_pass"] for r in sa["all_grid_rows"])

    win = sa["winner"]
    assert win["passes_primary_gate"] is True
    assert win["M_max_Msun"] >= 2.08
    assert win["Lambda_1_4"] <= 580.0
    assert win["R_1_4_km"] <= 13.0
    assert 0.59 <= win["M_eff_over_M_N"] <= 0.62


def test_stage_a_winner_matches_minimum_jeff_criterion(payload: dict) -> None:
    sa = payload["stage_a_widened_primary_gate"]
    win = sa["winner"]
    best = min(sa["primary_survivors"], key=lambda r: (r["abs_j_eff_diff_MeV"], -r["M_max_Msun"]))
    assert win["zeta_RMF"] == best["zeta_RMF"]
    assert win["n0_fm3"] == best["n0_fm3"]
    assert win["M_eff_over_M_N"] == best["M_eff_over_M_N"]
    assert win["L_MeV"] == best["L_MeV"]


def test_stage_b1_all_six_nuclei_zero_refit(payload: dict) -> None:
    sb1 = payload["stage_b1_blind_six_nuclei_winner"]
    assert [r["nucleus"] for r in sb1["nuclei"]] == ["O16", "Ca40", "Ca48", "Zr90", "Sn132", "Pb208"]
    for r in sb1["nuclei"]:
        assert r["converged"] is True
        assert r["particle_rel_err"] < 1.0e-6
        assert r["two_energy_formula_diff_MeV"] < 1.0e-3
        assert r["max_field_residual_rel"] < 1.0e-7
    assert sb1["summary_metrics"]["all_skins_heavy_positive"] is True


def test_stage_b2_sn132_exact_normal_collapse(payload: dict) -> None:
    sn_rows = {r["nucleus"]: r for r in payload["stage_b2_tin_chain_rh_bcs"]["rows"]}
    sn132 = sn_rows["Sn132"]
    assert sn132["bcs"]["collapsed_to_normal"] is True
    assert sn132["bcs"]["threshold_ratio"] < 1.0
    assert sn132["bcs"]["Delta_n_MeV"] == 0.0
    assert sn132["bcs"]["E_pair_MeV"] == 0.0
    assert abs(sn132["pairing_energy_gain_MeV"]) < 1.0e-6


def test_stage_b2_open_tin_bcs_gaps_and_binding_gain(payload: dict) -> None:
    sn_rows = {r["nucleus"]: r for r in payload["stage_b2_tin_chain_rh_bcs"]["rows"]}
    for name in ("Sn116", "Sn120", "Sn124"):
        row = sn_rows[name]
        bcs = row["bcs"]
        assert bcs["collapsed_to_normal"] is False
        assert 0.4 <= bcs["Delta_n_MeV"] <= 2.2
        assert bcs["E_pair_MeV"] < 0.0
        assert bcs["bcs_eq_residual"] < 1.0e-10
        assert bcs["particle_rel_err"] < 1.0e-6
        assert bcs["two_energy_formula_diff_MeV"] < 1.0e-3
        assert bcs["max_field_residual_rel"] < 1.0e-7
        assert row["pairing_energy_gain_MeV"] < -0.05


def test_stage_c_thermal_schur_and_ward(payload: dict) -> None:
    th = payload["stage_c_cross_bridge_verification"]["thermal_response_T70_mu875"]
    assert th["schur_vs_fd_rel_err"] < 1.0e-9
    assert th["ward_max_rel_err"] < 1.0e-8
    assert th["ph_only_negative_control_ward_rel_err"] > 1.0e-2
    assert th["direct_5x5_vs_schur_rel_err"] < 1.0e-12


def test_stage_c_hartle_iloveq_winner(payload: dict) -> None:
    stars = {s["label"]: s for s in payload["stage_c_cross_bridge_verification"]["hartle_iloveq_stars"]}
    assert set(stars) == {"PSR_J0737_3039A", "canonical_1_4"}
    for label, target_m in (("PSR_J0737_3039A", 1.3381), ("canonical_1_4", 1.4000)):
        s = stars[label]
        assert abs(s["M_Msun"] - target_m) < 1.0e-4
        assert s["YY_IQ_rel_dev"] < 1.0e-2
        assert s["YY_ILove_rel_dev"] < 1.0e-2
        assert s["YY_QLove_rel_dev"] < 1.0e-2


def test_live_reproducibility_stage_a_and_bcs(payload: dict) -> None:
    win = payload["stage_a_widened_primary_gate"]["winner"]
    r_check = b8.evaluate_widened_stage_a_point(
        win["n0_fm3"], win["M_eff_over_M_N"], win["L_MeV"], win["zeta_RMF"]
    )
    assert r_check["M_max_Msun"] == pytest.approx(win["M_max_Msun"], rel=1.0e-12)
    assert r_check["Lambda_1_4"] == pytest.approx(win["Lambda_1_4"], rel=1.0e-12)

    sn120_store = next(r for r in payload["stage_b2_tin_chain_rh_bcs"]["rows"] if r["nucleus"] == "Sn120")["bcs"]
    sn120_live = b8.solve_nucleus_rh_bcs(50, 70, payload["stage_a_widened_primary_gate"]["winner_frozen_params"])
    assert sn120_live["Delta_n_MeV"] == pytest.approx(sn120_store["Delta_n_MeV"], rel=1.0e-10)
    assert sn120_live["B_over_A_MeV"] == pytest.approx(sn120_store["B_over_A_MeV"], rel=1.0e-12)


def test_fail_closed_rejects_tampered_json(tmp_path: Path, payload: dict) -> None:
    bad1 = json.loads(json.dumps(payload))
    bad1["evidence_weight"] = 0.25
    p1 = tmp_path / "bad_ew.json"
    p1.write_text(json.dumps(bad1), encoding="utf-8")
    assert not b8.validate_artifact(p1)[0]

    bad2 = json.loads(json.dumps(payload))
    bad2["stage_a_widened_primary_gate"]["winner"]["b6_MeV_minus2"] *= 2.0
    p2 = tmp_path / "bad_b6.json"
    p2.write_text(json.dumps(bad2), encoding="utf-8")
    assert not b8.validate_artifact(p2)[0]
