#!/usr/bin/env python3
"""Pytest suite for Law 3B & Law 4 Unified Covariant Gravity Bridge Audit."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import mpmath as mp
import pytest

import nvg_unified_gravity_bridge_audit as audit


HERE = Path(__file__).resolve().parent
SCRIPT_PATH = HERE / "nvg_unified_gravity_bridge_audit.py"
ARTIFACT_PATH = HERE / "nvg_unified_gravity_bridge_results.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT_PATH.exists(), f"Missing generated artifact: {ARTIFACT_PATH}"
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


# 1. Artifact schema, canonical JSON, zero evidence weight & source hash
def test_artifact_exists_schema_and_zero_evidence_weight(artifact: dict) -> None:
    assert artifact["schema_version"] == audit.SCHEMA_VERSION
    assert artifact["bridge_id"] == audit.BRIDGE_ID
    assert artifact["status"] == audit.STATUS
    assert artifact["evidence_weight"] == 0.0
    assert not isinstance(artifact["evidence_weight"], bool)
    assert artifact["observed_likelihood"] is None

    sha = hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
    assert artifact["source_sha256"] == sha

    raw = ARTIFACT_PATH.read_text(encoding="utf-8")
    assert raw == json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + "\n"

    contract = artifact["calibration_vs_prediction_contract"]
    assert len(contract["calibration_targets"]) >= 2
    assert contract["forward_observational_predictions_claimed"] == []


# 2. CLI --validate succeeds on valid artifact and rejects corrupted artifacts (negative controls)
def test_cli_validate_passes_and_rejects_corrupted_json(tmp_path: Path, artifact: dict) -> None:
    proc_ok = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--validate", str(ARTIFACT_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_ok.returncode == 0, proc_ok.stderr
    assert "VALIDATION_OK" in proc_ok.stdout

    # Negative control A: malformed JSON syntax
    bad_syntax = tmp_path / "bad_syntax.json"
    bad_syntax.write_text("{not valid json", encoding="utf-8")
    ok, errs = audit.validate_artifact(bad_syntax)
    assert not ok and errs

    # Negative control B: tampered evidence_weight != 0.0
    bad_weight = tmp_path / "bad_weight.json"
    tampered_w = json.loads(json.dumps(artifact))
    tampered_w["evidence_weight"] = 1.0
    bad_weight.write_text(audit.canonical_json_dumps(tampered_w), encoding="utf-8")
    ok, errs = audit.validate_artifact(bad_weight)
    assert not ok
    assert any("evidence_weight" in e for e in errs)

    # Negative control C: tampered numerical value
    bad_num = tmp_path / "bad_num.json"
    tampered_n = json.loads(json.dumps(artifact))
    tampered_n["pillar_2_vacuum_bag_scale_bridge"]["bridge"]["U0_MeV_fm3"] = "99999.0"
    bad_num.write_text(audit.canonical_json_dumps(tampered_n), encoding="utf-8")
    ok, errs = audit.validate_artifact(bad_num)
    assert not ok and errs

    # Negative control D: missing required key via CLI returns non-zero exit code
    bad_missing = tmp_path / "bad_missing.json"
    tampered_m = json.loads(json.dumps(artifact))
    del tampered_m["pillar_4_ir_cosmology_theorem"]
    bad_missing.write_text(audit.canonical_json_dumps(tampered_m), encoding="utf-8")
    proc_fail = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--validate", str(bad_missing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_fail.returncode != 0


# 3. Symbolic symmetry separation: U(1)_B unitary gauge absorption vs global foliation Psi
def test_symbolic_symmetry_separation_u1b_vs_foliation(artifact: dict) -> None:
    ids = artifact["pillar_1_symbolic_symmetry_and_geometry"]["identities"]
    for key in (
        "u1b_unitary_gauge_absorption",
        "u1b_proca_combination_gauge_invariant",
        "foliation_unit_vector_norm",
        "cuscuton_kinetic_degeneracy_zero_ghosts",
        "cuscuton_energy_equals_potential",
        "cuscuton_euler_lagrange_nabla_u_eq_Vprime",
        "cuscuton_pressure_from_scale_variation",
    ):
        assert ids[key]["passed"] is True, f"Failed {key}: {ids[key]}"
        assert ids[key]["residual"] == "0"


# 4. Symbolic cuscuton bounce identities + negative control (wrong kinetic sign)
def test_symbolic_cuscuton_bounce_and_wrong_sign_negative_control(artifact: dict) -> None:
    ids = artifact["pillar_1_symbolic_symmetry_and_geometry"]["identities"]
    for key in (
        "bounce_potential_gradient_eq_3H",
        "bounce_friedmann_from_cuscuton",
        "bounce_target_friedmann_law",
        "bounce_raychaudhuri_from_cuscuton",
        "bounce_target_raychaudhuri",
        "bounce_matter_conservation",
        "bounce_point_H_zero",
        "bounce_point_Hdot_positive",
        "bounce_hubble_upper_bound_certificate",
    ):
        assert ids[key]["passed"] is True, f"Failed {key}"

    # Negative control: flipping cuscuton kinetic sign to +1 MUST fail field/Raychaudhuri equations
    bad = audit.symbolic_symmetry_and_geometry_proofs(cuscuton_kinetic_sign=+1)
    assert bad["all_passed"] is False
    assert bad["identities"]["cuscuton_euler_lagrange_nabla_u_eq_Vprime"]["passed"] is False
    assert bad["identities"]["bounce_raychaudhuri_from_cuscuton"]["passed"] is False


# 5. Symbolic BH interior foliation, shear anisotropy, Hayward core + isotropic no-go control
def test_symbolic_hayward_anisotropic_interior_and_isotropic_nogo_control(artifact: dict) -> None:
    ids = artifact["pillar_1_symbolic_symmetry_and_geometry"]["identities"]
    for key in (
        "bh_interior_foliation_unit_norm",
        "bh_interior_chi_equals_extrinsic_trace",
        "bh_interior_shear_traceless",
        "bh_interior_shear_ratio_2_to_minus1",
        "bh_interior_shear_ang_ratio",
        "flrw_shear_identically_zero",
        "hayward_density_einstein_eq",
        "hayward_radial_pressure_eq_minus_eps",
        "hayward_tangential_pressure_einstein_eq",
        "hayward_covariant_conservation",
        "hayward_anisotropy_traceless_matches_shear",
        "hayward_anisotropy_exact_formula",
        "hayward_pg_foliation_chi_exact",
        "hayward_pg_foliation_core_limit",
        "hayward_M_crit_exact",
        "hayward_T_crit_zero",
        "hayward_T_max_exact",
    ):
        assert ids[key]["passed"] is True, f"Failed {key}: {ids[key]}"

    # Negative control: forcing isotropic pressure (anisotropy_scale=0 => P_t = P_r = -eps)
    # MUST fail Hayward tangential Einstein equation and covariant conservation (isotropic no-go!)
    iso_bad = audit.symbolic_symmetry_and_geometry_proofs(hayward_anisotropy_scale=0)
    assert iso_bad["all_passed"] is False
    assert iso_bad["identities"]["hayward_tangential_pressure_einstein_eq"]["passed"] is False
    assert iso_bad["identities"]["hayward_covariant_conservation"]["passed"] is False
    assert iso_bad["identities"]["hayward_anisotropy_exact_formula"]["passed"] is False


# 6. Kretschmann finiteness global polynomial certificates (BH core & FLRW bounce)
def test_kretschmann_global_finiteness_certificates(artifact: dict) -> None:
    ids = artifact["pillar_1_symbolic_symmetry_and_geometry"]["identities"]
    for key in (
        "bounce_kretschmann_at_stiff_bounce",
        "bounce_kretschmann_convex_in_u",
        "bounce_kretschmann_u0_upper_bound",
        "bounce_kretschmann_u2_upper_bound",
        "hayward_kretschmann_exact",
        "hayward_kretschmann_core_limit",
        "hayward_kretschmann_global_bound_certificate",
        "hayward_core_to_bounce_Kmax_ratio",
    ):
        assert ids[key]["passed"] is True, f"Failed {key}"

    # Direct numerical sweep across 6 decades of q = r^3/(2 M ell^2)
    ell = 2.20167287855
    K_core = 24.0 / (ell**4)
    for q in (0.0, 1e-6, 1e-3, 0.1, 0.5, 1.0, 2.0, 10.0, 1e3, 1e6):
        K_val = 12.0 * (q**4 - 4.0 * q**3 + 18.0 * q**2 - 2.0 * q + 2.0) / ((ell**4) * ((1.0 + q) ** 6))
        assert 0.0 < K_val <= K_core + 1e-12


# 7. Vacuum bag scale bridge (60/90-dps mpmath) & exact scaling laws
def test_vacuum_bag_scale_bridge_60dps_precision(artifact: dict) -> None:
    sec = artifact["pillar_2_vacuum_bag_scale_bridge"]
    assert sec["precision_check"]["primary_dps"] == 60
    assert sec["precision_check"]["control_dps"] == 90
    assert sec["precision_check"]["passed"] is True

    br = sec["bridge"]
    bag = br["regimes"]["bag_kappa_1"]
    leg = br["regimes"]["legacy_W0_4"]

    with mp.workdps(60):
        lam = mp.mpf("1.05")
        W0 = mp.mpf("859")
        hbarc = mp.mpf("197.3269804")
        U0_expected = (lam / 4) * W0**4 / hbarc**3

        assert abs(mp.mpf(bag["rho_c_MeV_fm3"]) - U0_expected) < mp.mpf("1e-38")
        assert abs(mp.mpf(leg["rho_c_MeV_fm3"]) / mp.mpf(bag["rho_c_MeV_fm3"]) - 4 / lam) < mp.mpf("1e-40")

        # Cross-check legacy values against BLACK_HOLES_AND_CYCLES_RU.md (ell ~ 1.128 km, T_max ~ 3.59e-8 K)
        assert abs(float(leg["ell_km"]) - 1.12802168) < 1e-5
        assert abs(float(leg["T_max_K"]) - 3.589829e-8) < 1e-11

        # Verify bag scale kappa_g = 1 scaling: ell_bag = sqrt(4/lam) * ell_leg ~ 2.2017 km
        ratio_ell = mp.mpf(bag["ell_km"]) / mp.mpf(leg["ell_km"])
        assert abs(ratio_ell - mp.sqrt(4 / lam)) < mp.mpf("1e-40")
        assert abs(mp.mpf(bag["K_hayward_over_K_bounce"]) - mp.mpf(2) / 9) < mp.mpf("1e-40")
        assert abs(mp.mpf(leg["K_max_bounce_MeV4"]) / mp.mpf(bag["K_max_bounce_MeV4"]) - (4 / lam) ** 2) < mp.mpf("1e-40")


# 8. Quartic-vector symbolic identities & zeta=0 blowup negative control
def test_quartic_vector_identities_and_zeta_zero_blowup_control(artifact: dict) -> None:
    ids = artifact["pillar_1_symbolic_symmetry_and_geometry"]["identities"]
    for key in (
        "quartic_vector_chemical_potential_identity",
        "quartic_vector_legendre_pressure_identity",
        "quartic_vector_scalar_force_smooth_at_W0",
        "dynamic_matter_cuscuton_constraint_propagation",
    ):
        assert ids[key]["passed"] is True

    # Negative control: flipping scalar force sign breaks constraint propagation
    bad_force = audit.symbolic_symmetry_and_geometry_proofs(quartic_force_sign=-1)
    assert bad_force["identities"]["dynamic_matter_cuscuton_constraint_propagation"]["passed"] is False

    # Negative control: zeta=0 raises ZeroDivisionError at y=0 and blows up by > 1e6 at y=1e-4
    ctrl = artifact["pillar_3_dynamic_matter_bounce"]["zeta_zero_blowup_control"]
    assert ctrl["blowup_ratio_zeta0_over_zeta025"] > 1e6
    assert math.isfinite(ctrl["eV_zeta_025_at_exact_y_0"])
    with pytest.raises(ZeroDivisionError):
        audit.QuarticVectorMatter(zeta=0.0).solve_omega_hat(0.1, 0.0)


# 9. Dynamic matter bounce ODE (kappa_g = 1 bag scale): constraint < 1e-12
def test_dynamic_bounce_ode_bag_scale_constraint_conservation(artifact: dict) -> None:
    tr = artifact["pillar_3_dynamic_matter_bounce"]["trajectories"]["bag_kappa_1_mu5"]
    diag = tr["diagnostics"]
    mid = tr["bounce_state_at_tau_0"]
    assert diag["constraint_preserved_lt_1e12"] is True
    assert diag["baryon_preserved_lt_1e12"] is True
    assert diag["max_constraint_error"] < 1e-12
    assert diag["max_baryon_conservation_error"] < 1e-12
    assert diag["bounce_signature_verified"] is True
    assert diag["H_tc_start"] < 0.0 < diag["H_tc_end"]
    assert abs(mid["H_tc"]) < 1e-11
    assert mid["Hdot_tc2"] > 0.0


# 10. Dynamic matter bounce ODE (melted W=0 crossing & higher stiffness mu=20)
def test_dynamic_bounce_ode_melted_zero_crossing_and_stiffness(artifact: dict) -> None:
    for key in ("legacy_melted_crossing_mu5", "legacy_melted_crossing_mu20"):
        tr = artifact["pillar_3_dynamic_matter_bounce"]["trajectories"][key]
        diag = tr["diagnostics"]
        assert diag["constraint_preserved_lt_1e12"] is True
        assert diag["baryon_preserved_lt_1e12"] is True
        assert diag["max_constraint_error"] < 1e-12
        assert diag["bounce_signature_verified"] is True
        assert diag["crossed_W_zero_smoothly"] is True
        assert diag["y_min"] < 0.0 < diag["y_max"]
        assert math.isfinite(diag["max_omega_hat"])


# 11. Negative control: corrupted qdot rate breaks ODE constraint preservation
def test_negative_control_corrupted_bounce_rate_fails_constraint() -> None:
    bad_run = audit.integrate_dynamic_bounce(
        kappa_g=1.0,
        zeta=0.25,
        mu_stiff=5.0,
        y_bounce=0.5,
        v_bounce=0.15,
        qdot_rate_scale=1.02,
    )
    assert bad_run["diagnostics"]["constraint_preserved_lt_1e12"] is False
    assert bad_run["diagnostics"]["max_constraint_error"] > 1e-3


# 12. Law 4: Constant Lambda > 0 no-cycle theorem (strictly 1 bounce, 0 turnarounds, w = -1)
def test_constant_lambda_no_cycle_theorem(artifact: dict) -> None:
    ids = artifact["pillar_1_symbolic_symmetry_and_geometry"]["identities"]
    assert ids["law4_const_lambda_w_eq_minus1"]["passed"] is True
    assert ids["law4_const_lambda_asymptotic_H2"]["passed"] is True

    cl = artifact["pillar_4_ir_cosmology_theorem"]["constant_lambda_no_cycle"]
    assert cl["no_turnaround_verified"] is True
    assert cl["turnaround_events_count"] == 0
    assert cl["H_strictly_positive_post_bounce"] is True
    assert cl["w_DE_at_omega_L_ref"] == -1.0
    assert cl["H_tc_convergence_error"] < 1e-10
    assert cl["q_end_over_pi"] < 0.5
    assert cl["max_constraint_error"] < 1e-12


# 13. Law 4: External IR scalar (linear V_DE crossing zero) -> thawing w != -1 + turnaround + bounce 2
def test_ir_scalar_linear_thawing_turnaround_and_next_bounce(artifact: dict) -> None:
    ids = artifact["pillar_1_symbolic_symmetry_and_geometry"]["identities"]
    for key in (
        "law4_ir_turnaround_H_zero_at_pi_over_2",
        "law4_ir_turnaround_Hdot_negative_at_pi_over_2",
        "law4_ir_next_bounce_H_zero_at_pi",
        "law4_ir_next_bounce_Hdot_positive_at_pi",
        "law4_ir_scalar_thawing_eos_identity",
    ):
        assert ids[key]["passed"] is True

    lin = artifact["pillar_4_ir_cosmology_theorem"]["external_ir_linear_potential_cycle"]
    assert lin["cycle_and_thawing_verified"] is True
    assert lin["constraint_preserved_lt_1e12"] is True
    ev = lin["events"]

    # Thawing quintessence at Omega_DE = 0.69: w > -1 and dw/d(ln a) > 0 (w_a < 0)
    assert ev["omega_de_ref_epoch"]["w_DE"] > -1.0
    assert ev["dw_DE_d_lna_at_ref"] > 0.0

    # Zero crossing followed by turnaround (H=0, Hdot<0, V_DE<0) and next bounce (H=0, Hdot>0)
    assert ev["V_DE_zero_crossing"]["tau"] < ev["turnaround_1"]["tau"] < ev["bounce_2"]["tau"]
    assert abs(ev["turnaround_1"]["H_tc"]) < 1e-11
    assert ev["turnaround_1"]["Hdot_tc2"] < 0.0
    assert ev["turnaround_1"]["V_DE_hat"] < 0.0
    assert abs(ev["bounce_2"]["H_tc"]) < 1e-11
    assert ev["bounce_2"]["Hdot_tc2"] > 0.0
    assert ev["bounce_2"]["a_over_a_bounce1"] < ev["turnaround_1"]["a_over_a_bounce1"]


# 14. Law 4: Periodic cosine IR scalar completes multiple bounce-turnaround cycles
def test_ir_scalar_cosine_multicycle_recurrence(artifact: dict) -> None:
    cos_sec = artifact["pillar_4_ir_cosmology_theorem"]["external_ir_cosine_multicycle"]
    assert cos_sec["multicycle_verified"] is True
    assert cos_sec["constraint_preserved_lt_1e12"] is True
    ev = cos_sec["events"]
    assert ev["turnaround_1"]["tau"] < ev["bounce_2"]["tau"] < ev["turnaround_2"]["tau"] < ev["bounce_3"]["tau"]
    assert ev["turnaround_1"]["Hdot_tc2"] < 0.0
    assert ev["bounce_2"]["Hdot_tc2"] > 0.0
    assert ev["turnaround_2"]["Hdot_tc2"] < 0.0
    assert ev["bounce_3"]["Hdot_tc2"] > 0.0


# 15. Law 4 negative control: strictly positive scalar potential prevents turnaround
def test_negative_control_positive_potential_forbids_turnaround(artifact: dict) -> None:
    ctrl = artifact["pillar_4_ir_cosmology_theorem"]["negative_control_positive_potential_no_turnaround"]
    assert ctrl["no_turnaround_verified"] is True
    assert ctrl["constraint_preserved_lt_1e12"] is True
    assert ctrl["events"]["turnaround_1"] is None
    assert ctrl["events"]["V_DE_zero_crossing"] is None
    assert ctrl["q_end_over_pi"] < 0.5


# 16. Explicit external IR module segregation & honest caveats check
def test_ir_module_segregated_and_honest_caveats_present(artifact: dict) -> None:
    ir = artifact["pillar_4_ir_cosmology_theorem"]
    assert ir["external_ir_module"] is True
    assert ir["derived_from_W0_859_MeV"] is False
    caveats = artifact["honest_caveats"]
    assert len(caveats) >= 5
    joined = " ".join(caveats).lower()
    assert "evidence_weight" in joined
    assert "not derived from w0" in joined
    assert "stiffness" in joined
