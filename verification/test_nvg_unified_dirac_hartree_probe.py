#!/usr/bin/env python3
"""Pytest suite for the Law 3A spherical relativistic Dirac-Hartree bridge."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

import nvg_unified_dirac_hartree_probe as dh

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "nvg_unified_dirac_hartree_probe.py"
RESULT_JSON = HERE / "nvg_unified_dirac_hartree_results.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert RESULT_JSON.exists(), f"missing generated artifact {RESULT_JSON}"
    return json.loads(RESULT_JSON.read_text(encoding="utf-8"))


# 1. Analytical relativistic Dirac benchmarks (zero spurious states!)
def test_analytical_dirac_benchmarks_live_and_spurious_free() -> None:
    bench = dh.run_analytical_dirac_benchmarks()
    assert bench["pass"] is True
    assert bench["spurious_states_in_mass_gap"] == 0
    assert bench["free_spherical_box"]["max_abs_error_MeV"] < 1.0e-10
    assert bench["spin_symmetric_dirac_ho"]["max_abs_error_MeV"] < 1.0e-10
    assert bench["dirac_coulomb_sommerfeld"]["max_rel_error"] < 1.0e-4
    assert bench["dirac_coulomb_sommerfeld"]["states"]["1s1/2"]["rel_error"] < 1.0e-8


# 2. SNM calibration exactness and bounded-below vacuum stability
def test_snm_calibration_exact_and_bounded_below() -> None:
    cal = dh.calibrate_two_condensate_from_snm()
    assert cal["bounded_below_scalar"] is True
    assert cal["globally_nonnegative_U_s"] is True
    assert cal["c_s"] > 0.0
    assert cal["lam_W"] > 0.0
    assert cal["Lambda_v"] > 0.0
    ver = cal["snm_verification"]
    assert abs(ver["P_MeV_fm3"]) < 1.0e-9
    assert abs(ver["B_over_A_MeV"] - (-16.0)) < 1.0e-9
    assert abs(ver["K_MeV"] - 240.0) < 1.0e-6
    assert abs(ver["J_MeV"] - 31.0) < 1.0e-9
    assert abs(ver["L_MeV"] - 68.0) < 1.0e-6
    assert 0.55 < ver["M_eff_over_MN"] < 0.68
    assert 0.995 < ver["W_over_W0"] < 1.0


# 3. SPEC-v2 integrity: zero evidence weight & calibration vs forward separation
def test_spec_v2_schema_zero_evidence_and_canonical_json(artifact: dict) -> None:
    assert artifact["schema_version"] == dh.SCHEMA_VERSION
    assert artifact["status"] == dh.STATUS
    assert artifact["evidence_weight"] == 0.0
    assert artifact["calibration_targets"]["finite_nuclei_used_in_calibration"] is False
    assert "forward_predictions" in artifact
    assert "reference_comparisons" in artifact
    # Verify exact canonical JSON formatting on disk
    disk_bytes = RESULT_JSON.read_bytes()
    canonical_bytes = (
        json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert disk_bytes == canonical_bytes


# 4. Closed-shell particle-number normalization & orbital norms across all 4 nuclei
def test_shell_normalization_and_particle_numbers(artifact: dict) -> None:
    nuclei = artifact["forward_predictions"]["nuclei"]
    for name, z_exp, n_exp in dh.DOUBLY_MAGIC_NUCLEI:
        row = nuclei[name]
        pnum = row["particle_numbers"]
        assert pnum["Z_rel_err"] < 1.0e-7, f"{name} Z_rel_err={pnum['Z_rel_err']}"
        assert pnum["N_rel_err"] < 1.0e-7, f"{name} N_rel_err={pnum['N_rel_err']}"
        assert row["convergence"]["max_orbital_norm_error"] < 1.0e-12
        assert row["convergence"]["max_orbital_eig_residual"] < 1.0e-12
        # Sum of shell occupancies equals exact Z and N
        assert sum(s["occupancy"] for s in row["occupied_shells"]["proton"]) == pytest.approx(z_exp, abs=1e-12)
        assert sum(s["occupancy"] for s in row["occupied_shells"]["neutron"]) == pytest.approx(n_exp, abs=1e-12)


# 5. Field equation residuals < 1e-6 for all 5 coupled fields
def test_coupled_field_residuals_below_threshold(artifact: dict) -> None:
    nuclei = artifact["forward_predictions"]["nuclei"]
    for name, _, _ in dh.DOUBLY_MAGIC_NUCLEI:
        fres = nuclei[name]["convergence"]["field_residuals"]
        assert nuclei[name]["convergence"]["converged"] is True
        for key in ("W_rel", "s_rel", "A0_rel", "rho0_rel", "V_C_rel", "max_rel"):
            assert fres[key] < 1.0e-6, f"{name} {key}={fres[key]}"


# 6. Two independent total-energy formulas identity
def test_two_energy_formula_identity(artifact: dict) -> None:
    nuclei = artifact["forward_predictions"]["nuclei"]
    for name, _, _ in dh.DOUBLY_MAGIC_NUCLEI:
        en = nuclei[name]["energies"]
        assert en["B_over_A_eig_MeV"] < -6.0
        assert en["B_over_A_eig_MeV"] > -10.0
        assert en["energy_formula_diff_per_A_MeV"] < 1.0e-5, (
            f"{name} energy formula mismatch: {en['energy_formula_diff_per_A_MeV']}"
        )
        assert abs(en["B_over_A_eig_MeV"] - en["B_over_A_direct_MeV"]) < 1.0e-5


# 7. Positive neutron skins for Ca48 & Pb208 (fixing W8 TF inverted -0.026 fm skin)
def test_positive_neutron_skins_and_coulomb_symmetric_shift(artifact: dict) -> None:
    nuclei = artifact["forward_predictions"]["nuclei"]
    skin_o16 = nuclei["O16"]["radii"]["Delta_R_np_fm"]
    skin_ca40 = nuclei["Ca40"]["radii"]["Delta_R_np_fm"]
    skin_ca48 = nuclei["Ca48"]["radii"]["Delta_R_np_fm"]
    skin_pb208 = nuclei["Pb208"]["radii"]["Delta_R_np_fm"]
    # N=Z nuclei have slightly negative skin due to Coulomb repulsion on protons
    assert -0.10 < skin_o16 < 0.0
    assert -0.10 < skin_ca40 < 0.0
    # Neutron-rich nuclei have strongly POSITIVE neutron skins (unlike W8 TF -0.026 fm)
    assert 0.10 < skin_ca48 < 0.30
    assert 0.10 < skin_pb208 < 0.35
    # Verify R_ch = sqrt(R_p^2 + 0.8414^2) consistency
    for name, _, _ in dh.DOUBLY_MAGIC_NUCLEI:
        rad = nuclei[name]["radii"]
        assert rad["R_ch_fm"] == pytest.approx(math.hypot(rad["R_p_fm"], 0.8414), abs=1e-12)


# 8. Isotopic charge-radius shift Delta R_ch(Ca48 - Ca40) vs W8 TF failure (+0.166 fm)
def test_ca48_ca40_isotopic_shift_beats_w8_tf(artifact: dict) -> None:
    summary = artifact["forward_predictions"]["summary"]
    d_rch = summary["Delta_R_ch_Ca48_minus_Ca40_fm"]
    w8_tf = artifact["reference_comparisons"]["prior_w8_thomas_fermi_failures"]["Delta_R_ch_Ca48_minus_Ca40_fm"]
    assert abs(d_rch) < 0.04, f"isotopic shift {d_rch} fm should be small"
    assert abs(d_rch) < 0.25 * abs(w8_tf)
    assert summary["improves_over_W8_TF_isotopic_shift"] is True


# 9. Physical spin-orbit splittings (1p in O16, 1d in Ca40, 1f in Ca48)
def test_spin_orbit_splittings_physical_sign_and_magnitude(artifact: dict) -> None:
    nuclei = artifact["forward_predictions"]["nuclei"]
    so_1p = nuclei["O16"]["spin_orbit_splittings_MeV"]["1p"]["neutron_MeV"]
    so_1d = nuclei["Ca40"]["spin_orbit_splittings_MeV"]["1d"]["neutron_MeV"]
    so_1f = nuclei["Ca48"]["spin_orbit_splittings_MeV"]["1f"]["neutron_MeV"]
    assert 4.0 < so_1p < 9.0
    assert 4.0 < so_1d < 9.0
    assert 5.0 < so_1f < 10.0


# 10. Heavy gluon condensate W(r) finite-nucleus behavior
def test_heavy_gluon_condensate_dip_and_vacuum_restoration(artifact: dict) -> None:
    w0 = artifact["frozen_params"]["W0"]
    for name, _, _ in dh.DOUBLY_MAGIC_NUCLEI:
        cf = artifact["forward_predictions"]["nuclei"][name]["central_fields"]
        assert 0.99 < cf["W_center_over_W0"] < 1.0
        assert cf["W_boundary_MeV"] == pytest.approx(w0, abs=1e-12)
        assert 0.50 < cf["M_eff_center_over_MN"] < 0.75


# 11. Grid-refinement convergence check on Ca48
def test_grid_refinement_convergence(artifact: dict) -> None:
    gchk = artifact["grid_refinement_check"]
    assert gchk is not None
    assert gchk["pass"] is True
    assert gchk["delta_B_over_A_MeV"] < 1.0e-5
    assert gchk["delta_R_ch_fm"] < 1.0e-5
    assert gchk["delta_skin_fm"] < 1.0e-5


# 12. Reusable API & --params-json frozen parameter roundtrip (zero refitting)
def test_params_json_roundtrip_zero_refit(tmp_path: Path, artifact: dict) -> None:
    pfile = tmp_path / "frozen_law2_params.json"
    pfile.write_text(json.dumps({"frozen_params": artifact["frozen_params"]}, indent=2), encoding="utf-8")
    loaded = dh.load_params_from_json(pfile)
    o16_live = dh.solve_nucleus_dirac_hartree(8, 8, loaded)
    o16_saved = artifact["forward_predictions"]["nuclei"]["O16"]
    assert o16_live["energies"]["B_over_A_eig_MeV"] == pytest.approx(
        o16_saved["energies"]["B_over_A_eig_MeV"], abs=1e-12
    )
    assert o16_live["radii"]["R_ch_fm"] == pytest.approx(o16_saved["radii"]["R_ch_fm"], abs=1e-12)


# 13. CLI --validate passes on the real artifact
def test_cli_validate_passes_on_artifact() -> None:
    proc = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--validate", str(RESULT_JSON)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "VALIDATION_PASS" in proc.stdout


# 14. Negative controls: fail-closed rejection of corrupted JSON and tampered invariants
def test_negative_controls_fail_closed(tmp_path: Path, artifact: dict) -> None:
    # (a) Corrupted JSON syntax rejected by --validate CLI
    bad_json = tmp_path / "corrupted.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--validate", str(bad_json)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "VALIDATION_FAILED" in proc.stderr

    # (b) Non-zero evidence_weight rejected
    p_ev = copy.deepcopy(artifact)
    p_ev["evidence_weight"] = 1.0
    with pytest.raises(dh.DiracHartreeBridgeError, match="evidence_weight"):
        dh.validate_payload(p_ev, check_source_sha=False)

    # (c) Inverted (negative) Pb208 neutron skin rejected
    p_skin = copy.deepcopy(artifact)
    p_skin["forward_predictions"]["nuclei"]["Pb208"]["radii"]["Delta_R_np_fm"] = -0.026
    p_skin["forward_predictions"]["nuclei"]["Pb208"]["radii"]["R_n_fm"] = (
        p_skin["forward_predictions"]["nuclei"]["Pb208"]["radii"]["R_p_fm"] - 0.026
    )
    with pytest.raises(dh.DiracHartreeBridgeError, match="neutron skins must be positive"):
        dh.validate_payload(p_skin, check_source_sha=False)

    # (d) Field residual above threshold 1e-6 rejected
    p_res = copy.deepcopy(artifact)
    p_res["forward_predictions"]["nuclei"]["Ca48"]["convergence"]["field_residuals"]["max_rel"] = 5.0e-5
    with pytest.raises(dh.DiracHartreeBridgeError, match="field residual"):
        dh.validate_payload(p_res, check_source_sha=False)

    # (e) Two-energy-formula mismatch rejected
    p_en = copy.deepcopy(artifact)
    p_en["forward_predictions"]["nuclei"]["Ca40"]["energies"]["B_over_A_direct_MeV"] += 0.05
    p_en["forward_predictions"]["nuclei"]["Ca40"]["energies"]["energy_formula_diff_per_A_MeV"] = 0.05
    with pytest.raises(dh.DiracHartreeBridgeError, match="two-energy-formula"):
        dh.validate_payload(p_en, check_source_sha=False)

    # (f) Wrong-sign spin-orbit splitting rejected
    p_so = copy.deepcopy(artifact)
    p_so["forward_predictions"]["nuclei"]["O16"]["spin_orbit_splittings_MeV"]["1p"]["neutron_MeV"] = -6.0
    with pytest.raises(dh.DiracHartreeBridgeError, match="spin-orbit"):
        dh.validate_payload(p_so, check_source_sha=False)

    # (g) Unbounded / false-vacuum scalar potentials in --params-json rejected fail-closed
    bad_params = tmp_path / "unbounded_params.json"
    bp = copy.deepcopy(artifact["frozen_params"])
    bp["d_s"] = 0.0
    bp["c_s"] = -10.0
    bad_params.write_text(json.dumps(bp), encoding="utf-8")
    with pytest.raises(dh.DiracHartreeBridgeError, match="bounded-below"):
        dh.load_params_from_json(bad_params)

    # (h) Negative sextic d_s < 0 rejected fail-closed
    bp_neg6 = copy.deepcopy(artifact["frozen_params"])
    bp_neg6["d_s"] = -1.0e-6
    bad_params.write_text(json.dumps(bp_neg6), encoding="utf-8")
    with pytest.raises(dh.DiracHartreeBridgeError, match="bounded-below"):
        dh.load_params_from_json(bad_params)

    # (i) False vacuum deeper than s = 0 (q_min <= 0 with d_s > 0) rejected fail-closed
    bp_fv = copy.deepcopy(artifact["frozen_params"])
    bp_fv["c_s"] = -1000.0
    bad_params.write_text(json.dumps(bp_fv), encoding="utf-8")
    with pytest.raises(dh.DiracHartreeBridgeError, match="bounded-below"):
        dh.load_params_from_json(bad_params)


# 15. Cross-bridge synchronization: exact parameter identity with Law 2 artifact
def test_law2_shared_parameter_synchronization(artifact: dict) -> None:
    law2_path = HERE / "nvg_two_condensate_unified_results.json"
    assert law2_path.exists(), f"missing Law 2 artifact {law2_path}"
    law2 = json.loads(law2_path.read_text(encoding="utf-8"))
    l2p = law2["frozen_parameters"]
    fp = artifact["frozen_params"]

    assert artifact["calibration_targets"]["mode"] == "external_frozen_params_zero_refit"
    pairs = [
        ("W0", "W0_MeV"),
        ("lam_W", "lam_W"),
        ("M_N", "M_N_MeV"),
        ("m_omega", "m_omega_MeV"),
        ("m_rho", "m_rho_MeV"),
        ("m_s", "m_s_MeV"),
        ("q_phi", "q_phi"),
        ("m_W", "m_W_MeV"),
        ("g_s", "g_s_ch"),
        ("b3", "b3"),
        ("b4", "b4"),
        ("b6_MeV_minus2", "b6_MeV_minus2"),
        ("g_omega", "g_omega"),
        ("zeta", "zeta_RMF"),
        ("g_rho", "g_rho"),
        ("Lambda_v", "Lambda_v"),
    ]
    for k3a, k2 in pairs:
        assert fp[k3a] == pytest.approx(l2p[k2], rel=1e-13, abs=1e-15), f"mismatch on {k3a} vs {k2}"

    ver = artifact["calibration_targets"]["snm_verification"]
    assert abs(ver["P_MeV_fm3"]) < 1.0e-10
    assert abs(ver["B_over_A_MeV"] - (-16.0)) < 1.0e-10
    assert abs(ver["K_MeV"] - 240.0) < 1.0e-5
    assert abs(ver["J_MeV"] - 31.0) < 1.0e-10
    assert abs(ver["L_MeV"] - 56.0) < 1.0e-5
    assert abs(ver["M_eff_over_MN"] - 0.65) < 1.0e-12
