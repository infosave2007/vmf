#!/usr/bin/env python3
"""Pytest verification suite for Law 2 (Globally Bounded Two-Condensate Unified Matter Action).

Tests SPEC-v2 fail-closed scientific integrity, exact calibration-vs-prediction
separation, global vacuum stability, Hugenholtz-Van Hove thermodynamic closure,
static NN exchange escape from Q4 no-go, cold beta-equilibrium TOV+Hinderer
convergence, causality & conformal c_s^2 -> 1/3 asymptote, finite-T susceptibility
& liquid-gas spinodal, CLI --validate, and explicit negative controls.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_two_condensate_unified_audit as audit  # noqa: E402

RESULT_PATH = HERE / "nvg_two_condensate_unified_results.json"
PRODUCER_PATH = HERE / "nvg_two_condensate_unified_audit.py"
BASELINE_PATH = HERE / "source_complete_solution_audit.py"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert RESULT_PATH.is_file(), f"Missing generated result JSON: {RESULT_PATH}"
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live_params() -> audit.ModelParams:
    return audit.calibrate_model()


def test_01_schema_spec_v2_zero_evidence_and_canonical_json(artifact: dict) -> None:
    assert artifact["schema_version"] == audit.SCHEMA_VERSION
    assert artifact["bridge_id"] == audit.BRIDGE_ID
    assert artifact["status"] == audit.STATUS_PASS
    assert artifact["evidence_weight"] == 0.0
    assert not isinstance(artifact["evidence_weight"], bool)
    assert artifact["source_sha256"] == hashlib.sha256(PRODUCER_PATH.read_bytes()).hexdigest()
    assert artifact["baseline_anchor_verification"]["baseline_anchor_sha256"] == hashlib.sha256(
        BASELINE_PATH.read_bytes()
    ).hexdigest()
    # Deterministic canonical JSON formatting check (sort_keys=True, indent=2, allow_nan=False)
    raw = RESULT_PATH.read_text(encoding="utf-8")
    assert raw == audit.serialize_canonical(artifact)
    assert all(v is True for v in artifact["gates"].values())


def test_02_baseline_nvg_anchor_preserved_exact(artifact: dict) -> None:
    fp = artifact["frozen_parameters"]
    assert fp["W0_MeV"] == 859.0
    assert fp["lam_W"] == 1.05
    assert fp["M_N_MeV"] == 939.0
    assert fp["m_omega_MeV"] == 782.6
    assert fp["m_rho_MeV"] == 775.3
    assert fp["n0_fm3"] == 0.16
    assert fp["hbar_c_MeV_fm"] == 197.3269804
    assert math.isclose(fp["g_W"], 939.0 / 859.0, rel_tol=1e-14)
    assert math.isclose(fp["q_phi"], 782.6 / 859.0, rel_tol=1e-14)
    assert math.isclose(fp["m_W_MeV"], math.sqrt(2.0 * 1.05) * 859.0, rel_tol=1e-14)
    # Exact RMF-to-covariant vector quartic dictionary
    expected_zeta_rmf = 6.0 * fp["zeta_g"] * (fp["q_phi"] / fp["g_omega"]) ** 4
    assert math.isclose(fp["zeta_RMF"], expected_zeta_rmf, rel_tol=1e-13)


def test_03_calibration_targets_exact_and_separated(artifact: dict) -> None:
    cal = artifact["calibration_targets"]
    assert cal["zero_ns_or_finite_nucleus_fitting"] is True
    assert cal["all_targets_matched"] is True
    rec = cal["recomputed_at_n0"]
    assert abs(rec["binding_energy_per_nucleon_MeV"] - (-16.0)) < 1e-9
    assert abs(rec["pressure_MeV_fm3"]) < 1e-9
    assert abs(rec["K_MeV"] - 240.0) < 1e-3
    assert abs(rec["M_eff_over_M_N"] - 0.65) < 1e-10
    assert abs(rec["J_MeV"] - 31.0) < 1e-9
    assert abs(rec["L_MeV"] - 56.0) < 1e-3


def test_04_global_vacuum_stability_certificate_and_live_scan(artifact: dict, live_params: audit.ModelParams) -> None:
    stab = artifact["global_stability_certificate"]
    assert stab["U_W_globally_nonnegative"] is True
    assert stab["U_ch_highest_coeff_positive"] is True
    assert stab["U_ch_globally_strictly_positive_away_from_origin"] is True
    assert stab["U_ch_strictly_monotone_on_physical_positive_s"] is True
    assert stab["unique_global_vacuum_W0_zero"] is True
    assert stab["no_false_vacuum_or_runaway"] is True
    assert stab["U_ch_q_min_MeV2"] > 10.0

    # Independent live dense sweep across [-4000, +4000] MeV
    assert audit.U_W(live_params.W0, live_params) == 0.0
    assert audit.U_ch_from_phi(0.0, live_params) == 0.0
    phis = np.linspace(-4000.0, 4000.0, 8001)
    for phi in phis:
        if phi != 0.0:
            assert audit.U_ch_from_phi(float(phi), live_params) > 0.0
        if phi > 0.0:
            assert audit.U_ch_prime_from_phi(float(phi), live_params) > 0.0


def test_05_hvh_thermodynamic_consistency_and_live_spot_check(artifact: dict, live_params: audit.ModelParams) -> None:
    tc = artifact["thermodynamic_consistency"]
    assert tc["pass"] is True
    assert tc["max_symmetric_hvh_pressure_rel"] < 1e-12
    assert tc["max_symmetric_hvh_mu_fd_rel"] < 1e-8
    assert tc["max_beta_hvh_pressure_rel"] < 1e-12
    assert tc["max_beta_hvh_mu_fd_rel"] < 1e-6
    assert tc["max_eom_rel"] < 1e-10

    # Live spot check at arbitrary asymmetric nuclear matter (n_B = 2.5 n0, x_p = 0.22)
    n_B = 2.5 * live_params.n0_nat
    xp = 0.22
    st = audit.solve_zero_t_state(live_params, xp * n_B, (1.0 - xp) * n_B)
    p_legendre = xp * n_B * st["mu_p"] + (1.0 - xp) * n_B * st["mu_n"] - st["eps"]
    assert abs(st["prs"] - p_legendre) / abs(st["eps"]) < 1e-13
    # Independent finite difference d(eps)/dn_p at fixed n_n == mu_p
    dn = 1e-5 * n_B
    stp = audit.solve_zero_t_state(live_params, xp * n_B + dn, (1.0 - xp) * n_B, (st["W"], st["Phi"], st["A0"]))
    stm = audit.solve_zero_t_state(live_params, xp * n_B - dn, (1.0 - xp) * n_B, (st["W"], st["Phi"], st["A0"]))
    mu_p_fd = (stp["eps"] - stm["eps"]) / (2.0 * dn)
    assert abs(st["mu_p"] - mu_p_fd) / abs(st["mu_p"]) < 1e-8


def test_06_pure_quartic_no_go_diagnostic_verified(artifact: dict) -> None:
    qd = artifact["pure_quartic_no_go_diagnostic"]
    assert qd["b4"] > 0.0
    assert qd["U_ch_globally_bounded"] is True
    assert qd["M_max_Msun"] < 2.02
    assert qd["reaches_2_02_Msun"] is False


def test_07_forward_spin_orbit_depth_and_saturation_fields(artifact: dict) -> None:
    so = artifact["forward_predictions"]["saturation_and_spin_orbit"]
    assert 0.58 <= so["M_eff_over_M_N"] <= 0.65
    assert 0.99 < so["W_over_W0_at_n0"] < 1.01
    assert so["V0_MeV"] > 200.0
    assert so["S_MeV"] < -250.0
    assert so["V0_minus_S_spin_orbit_depth_MeV"] > 500.0
    assert so["spin_orbit_enhancement_vs_W8_93"] > 6.5


def test_08_forward_nn_static_exchange_escapes_q4_nogo(artifact: dict) -> None:
    nn = artifact["forward_predictions"]["nn_static_exchange"]
    assert nn["Q4_baseline_V_Q0_MeV_minus2"] > 0.0
    assert nn["V_Q0_MeV_minus2"] < 0.0
    assert nn["hard_repulsive_core_at_short_distance"] is True
    assert nn["net_long_range_attraction_V_Q0_negative"] is True
    assert nn["escapes_Q4_static_repulsion_nogo"] is True
    assert 0.15 < nn["r_cross_fm"] < 0.80
    assert nn["V_at_0_15fm_MeV"] > 0.0
    assert nn["V_at_1_00fm_MeV"] < 0.0
    assert nn["V_min_MeV"] < -20.0


def test_09_forward_blind_isovector_jeff_prediction(artifact: dict, live_params: audit.ModelParams) -> None:
    jf = artifact["forward_predictions"]["blind_isovector_jeff_011"]
    live_s11 = audit.symmetry_energy_at_density(live_params, 0.11)
    assert math.isclose(jf["j_eff_011_MeV"], live_s11["S_int_MeV"], rel_tol=1e-12)
    assert 10.0 < jf["j_eff_011_MeV"] < 13.0
    assert math.isclose(jf["repo_blind_extracted_j_MeV"], 11.13601607117819, rel_tol=1e-12)


def test_10_forward_ns_tov_tidal_and_direct_urca(artifact: dict) -> None:
    ns = artifact["forward_predictions"]["cold_beta_ns_tov_tidal"]
    tov = ns["tov_primary"]
    assert tov["M_max_Msun"] >= 2.02
    assert tov["R_1_4_km"] <= 12.70
    assert 11.0 <= tov["R_1_4_km"] <= 12.70
    assert 70.0 <= tov["Lambda_1_4"] <= 580.0
    assert 10.5 <= tov["R_2_0_km"] <= 13.0
    assert tov["n_c_Mmax_fm3"] > 0.7

    du = ns["direct_urca"]
    assert du["active_in_1_4_Msun_star"] is False
    assert du["active_in_Mmax_star"] is True
    assert 1.5 < du["exact_triangle_onset_M_Msun"] < tov["M_max_Msun"]
    assert du["exact_triangle_onset_n_fm3"] > du["muon_free_xp_one_ninth_n_fm3"]


def test_11_causality_and_conformal_one_third_asymptote(artifact: dict) -> None:
    causal = artifact["forward_predictions"]["cold_beta_ns_tov_tidal"]["causality_and_asymptote"]
    assert causal["all_causal_0_lt_cs2_lt_1_beta_up_to_10n0"] is True
    assert causal["all_positive_M_and_W_up_to_10n0"] is True
    assert causal["no_hand_glued_css_phase"] is True
    assert 0.0 < causal["min_cs2_beta_up_to_10n0"] < causal["max_cs2_beta_up_to_10n0"] < 1.0
    assert causal["approaches_one_third_monotonically_from_10_to_200n0"] is True
    samples = causal["symmetric_asymptote_samples"]
    assert samples[0]["cs2"] > samples[1]["cs2"] > samples[2]["cs2"] > 1.0 / 3.0
    assert abs(samples[2]["cs2"] - 1.0 / 3.0) < 0.02


def test_12_tov_ode_convergence_ladder(artifact: dict) -> None:
    conv = artifact["forward_predictions"]["cold_beta_ns_tov_tidal"]["tov_convergence_control"]
    assert conv["pass"] is True
    assert conv["M_max_primary_vs_tight_abs"] < 1e-4
    assert conv["R_1_4_primary_vs_tight_abs_km"] < 1e-3
    assert conv["Lambda_1_4_primary_vs_tight_rel"] < 1e-3


def test_13_finite_temperature_r2_and_liquid_gas_spinodal(artifact: dict) -> None:
    ft = artifact["forward_predictions"]["finite_temperature"]
    gc = ft["T_70_MeV_grand_canonical"]
    assert gc["all_quadrature_converged"] is True
    rows = {r["mu_B_MeV"]: r for r in gc["rows"]}
    assert set(rows.keys()) == {775.0, 875.0}
    assert rows[775.0]["r2_T_chiB_over_nB"] > rows[875.0]["r2_T_chiB_over_nB"] > 0.0
    assert rows[775.0]["within_strict_W8_band_0_5316_to_1_1296"] is True
    assert rows[875.0]["near_W8_band_within_0_50_to_1_15"] is True
    for r in gc["rows"]:
        assert r["r2_quadrature_control_rel"] < 1e-6

    spin = ft["liquid_gas_spinodal"]
    assert spin["spinodal_present_at_10MeV"] is True
    assert spin["spinodal_absent_at_20MeV"] is True
    assert 12.0 < spin["critical_temperature_T_c_MeV"] < 18.0


def test_14_cli_validate_passes_on_real_artifact() -> None:
    proc = subprocess.run(
        [sys.executable, str(PRODUCER_PATH), "--validate", str(RESULT_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads(proc.stdout)
    assert out["valid"] is True
    assert out["evidence_weight"] == 0.0


def test_15_negative_control_corrupted_json_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "corrupted.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(audit.AuditError, match="Corrupted JSON"):
        audit.validate_result_file(bad)
    proc = subprocess.run(
        [sys.executable, str(PRODUCER_PATH), "--validate", str(bad)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert json.loads(proc.stdout)["valid"] is False


def test_16_negative_control_spec_v2_and_hash_tampering_rejected(artifact: dict, tmp_path: Path) -> None:
    # 1) Non-zero evidence_weight rejected
    d1 = copy.deepcopy(artifact)
    d1["evidence_weight"] = 1.0
    p1 = tmp_path / "bad_weight.json"
    p1.write_text(json.dumps(d1), encoding="utf-8")
    with pytest.raises(audit.AuditError, match="evidence_weight"):
        audit.validate_result_file(p1)

    # 2) Tampered source_sha256 rejected
    d2 = copy.deepcopy(artifact)
    d2["source_sha256"] = "0" * 64
    p2 = tmp_path / "bad_sha.json"
    p2.write_text(json.dumps(d2), encoding="utf-8")
    with pytest.raises(audit.AuditError, match="source_sha256"):
        audit.validate_result_file(p2)

    # 3) Non-finite NaN rejected
    p3 = tmp_path / "bad_nan.json"
    p3.write_text(RESULT_PATH.read_text(encoding="utf-8").replace("500.0", "NaN", 1), encoding="utf-8")
    with pytest.raises(audit.AuditError, match="Non-finite"):
        audit.validate_result_file(p3)


def test_17_negative_control_wrong_sign_and_threshold_tampering_rejected(artifact: dict, tmp_path: Path) -> None:
    # 1) Wrong sign on V(Q=0) (Q4 repulsion regression) rejected
    d1 = copy.deepcopy(artifact)
    d1["forward_predictions"]["nn_static_exchange"]["V_Q0_MeV_minus2"] = 1.0e-4
    p1 = tmp_path / "bad_vq0.json"
    p1.write_text(json.dumps(d1), encoding="utf-8")
    with pytest.raises(audit.AuditError, match="V\\(Q=0\\)"):
        audit.validate_result_file(p1)

    # 2) Unbounded chiral potential (q_min <= 0) rejected
    d2 = copy.deepcopy(artifact)
    d2["global_stability_certificate"]["U_ch_q_min_MeV2"] = -1.0
    p2 = tmp_path / "bad_qmin.json"
    p2.write_text(json.dumps(d2), encoding="utf-8")
    with pytest.raises(audit.AuditError, match="Global vacuum stability"):
        audit.validate_result_file(p2)

    # 3) Sub-threshold M_max < 2.02 rejected
    d3 = copy.deepcopy(artifact)
    d3["forward_predictions"]["cold_beta_ns_tov_tidal"]["tov_primary"]["M_max_Msun"] = 1.95
    p3 = tmp_path / "bad_mmax.json"
    p3.write_text(json.dumps(d3), encoding="utf-8")
    with pytest.raises(audit.AuditError, match="M_max"):
        audit.validate_result_file(p3)

    # 4) Tampered frozen parameter rejected against live calibration
    d4 = copy.deepcopy(artifact)
    d4["frozen_parameters"]["g_omega"] = 9.99
    p4 = tmp_path / "bad_param.json"
    p4.write_text(json.dumps(d4), encoding="utf-8")
    with pytest.raises(audit.AuditError, match="Frozen parameter mismatch"):
        audit.validate_result_file(p4)


def test_18_negative_control_unbounded_potential_caught_live(live_params: audit.ModelParams) -> None:
    """Setting b6=0 while keeping b4<0 must fail the global vacuum stability certificate."""
    unbounded = dataclasses.replace(live_params, b6=0.0)
    assert unbounded.b4 < 0.0
    stab = audit.check_global_vacuum_stability(unbounded)
    assert stab["U_ch_highest_coeff_positive"] is False
    assert stab["U_ch_globally_strictly_positive_away_from_origin"] is False
    assert stab["no_false_vacuum_or_runaway"] is False
