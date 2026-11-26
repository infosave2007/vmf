#!/usr/bin/env python3
"""Pytest suite for Bridge 5: Unified Thermal Master Response & Schur-Hessian."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

VERIF_DIR = Path(__file__).resolve().parent
if str(VERIF_DIR) not in sys.path:
    sys.path.insert(0, str(VERIF_DIR))

import nvg_unified_thermal_response_bridge as b5  # noqa: E402


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(b5.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))


def test_validate_live_artifact() -> None:
    ok, errs = b5.validate_artifact(b5.DEFAULT_OUTPUT_PATH)
    assert ok, errs


def test_evidence_weight_and_provenance(payload: dict) -> None:
    assert payload["evidence_weight"] == 0.0
    assert payload["provenance"]["zero_parameter_refit"] is True
    assert payload["status"] == b5.STATUS_PASS
    assert all(payload["gates"].values())


def test_schur_matches_thermodynamic_fd(payload: dict) -> None:
    assert payload["thermodynamic_and_schur_hessian"]["max_schur_vs_fd_rel_err"] < 1.0e-9
    for row in payload["thermodynamic_and_schur_hessian"]["rows"]:
        H = np.array(row["H_T_MeV_fm3"])
        chi = np.array(row["chi_T_fm3_per_MeV"])
        assert np.allclose(H @ chi, np.eye(2), atol=1.0e-10)


def test_symmetric_off_diagonal_vanishes(payload: dict) -> None:
    sym_rows = [r for r in payload["thermodynamic_and_schur_hessian"]["rows"] if abs(r["delta"]) < 1e-12]
    assert len(sym_rows) == 2
    for row in sym_rows:
        assert abs(row["H_T_MeV_fm3"][0][1]) < 1.0e-12
        assert abs(row["chi_T_fm3_per_MeV"][0][1]) < 1.0e-12


def test_asymmetric_coupling_positive_and_adiabatic_stiffer(payload: dict) -> None:
    asym_rows = [r for r in payload["thermodynamic_and_schur_hessian"]["rows"] if abs(r["delta"] - 0.2) < 1e-12]
    assert len(asym_rows) == 2
    for row in asym_rows:
        assert abs(row["H_T_MeV_fm3"][0][1]) > 1.0
        assert row["K_sigma_MeV_fm3"] > row["K_T_MeV_fm3"]
        assert 0.0 < row["cs2_adiabatic"] < 1.0
        assert row["adiabatic_path_rel_err"] < 1.0e-7


def test_thermo_identity_and_eom_residuals(payload: dict) -> None:
    assert payload["thermodynamic_and_schur_hessian"]["max_thermo_id_rel_err"] < 1.0e-12
    assert payload["thermodynamic_and_schur_hessian"]["max_eom_rel_err"] < 1.0e-11


def test_all_ward_identities_hold(payload: dict) -> None:
    assert payload["finite_qz_master_response"]["max_ward_rel_err"] < 1.0e-8
    assert len(payload["finite_qz_master_response"]["rows"]) == 12


def test_ph_only_negative_control_violates_ward(payload: dict) -> None:
    assert payload["finite_qz_master_response"]["min_ph_only_negative_control_ward_rel_err"] > 1.0e-2


def test_direct_5x5_matches_block_schur(payload: dict) -> None:
    assert payload["finite_qz_master_response"]["max_direct_5x5_vs_schur_rel_err"] < 1.0e-12


def test_low_q_static_recovers_thermodynamic_chi(payload: dict) -> None:
    assert payload["finite_qz_master_response"]["max_low_q_thermo_recovery_rel_err"] < 1.0e-4


def test_frequency_conjugation_symmetry() -> None:
    params, _ = b5.load_law2_frozen_params()
    st = b5.solve_gc_for_asymmetry(params, 70.0, 875.0, 0.0)
    r_pos = b5._finite_q_master_5x5(params, st, 150.0, complex(80.0, 40.0))
    r_neg = b5._finite_q_master_5x5(params, st, 150.0, complex(-80.0, 40.0))
    c_pos = complex(r_pos["chi_direct_MeV2"]["real"], r_pos["chi_direct_MeV2"]["imag"])
    c_neg = complex(r_neg["chi_direct_MeV2"]["real"], r_neg["chi_direct_MeV2"]["imag"])
    assert abs(c_pos - c_neg.conjugate()) / abs(c_pos) < 1.0e-12


def test_isovector_zero_sound_causal(payload: dict) -> None:
    iv = payload["isovector_zero_sound_cold"]
    assert iv["v_F_over_c"] < iv["u_IV_over_c"] < 1.0
    assert 0.25 < iv["u_IV_over_c"] < 0.45


def test_liquid_gas_critical_fluctuations_vs_T70(payload: dict) -> None:
    hd = payload["hades_fluctuation_diagnostic"]
    assert max(hd["T70_symmetric_r2_values"]) < b5.HADES_NECESSARY_R2_MIN
    assert hd["max_supercritical_r2"] > b5.HADES_NECESSARY_R2_MIN * 2.0


def test_fail_closed_rejects_tampered_json(tmp_path: Path, payload: dict) -> None:
    bad = dict(payload)
    bad["evidence_weight"] = 1.0
    p1 = tmp_path / "bad_ew.json"
    p1.write_text(json.dumps(bad), encoding="utf-8")
    ok, _ = b5.validate_artifact(p1)
    assert not ok

    bad2 = json.loads(json.dumps(payload))
    bad2["thermodynamic_and_schur_hessian"]["rows"][0]["n_B_fm3"] += 0.05
    p2 = tmp_path / "bad_val.json"
    p2.write_text(json.dumps(bad2), encoding="utf-8")
    ok2, _ = b5.validate_artifact(p2)
    assert not ok2
