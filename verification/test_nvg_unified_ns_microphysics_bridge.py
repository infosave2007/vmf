"""Pytest suite for Bridge 9A: NS Core-Crust Transition, Direct Urca & Cowling Asteroseismology."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_unified_ns_microphysics_bridge as b9a

ARTIFACT_PATH = HERE / "nvg_unified_ns_microphysics_results.json"
SCRIPT_PATH = HERE / "nvg_unified_ns_microphysics_bridge.py"


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT_PATH.exists(), f"Missing generated artifact: {ARTIFACT_PATH}"
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_zero_evidence_schema_and_live_validate(tmp_path: Path, artifact: dict) -> None:
    assert artifact["schema_version"] == b9a.SCHEMA_VERSION
    assert artifact["bridge_id"] == b9a.BRIDGE_ID
    assert artifact["status"] == b9a.STATUS_PASS
    assert artifact["evidence_weight"] == 0.0
    assert artifact["observed_likelihood"] is None
    assert all(bool(v) for v in artifact["gates"].values())

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--validate", str(ARTIFACT_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Negative controls: tampered evidence_weight and tampered gate
    bad_w = tmp_path / "bad_weight.json"
    mod_w = json.loads(json.dumps(artifact))
    mod_w["evidence_weight"] = 1.0
    bad_w.write_text(json.dumps(mod_w), encoding="utf-8")
    ok, errs = b9a.validate_artifact(bad_w)
    assert not ok and any("evidence_weight" in e for e in errs)

    bad_gate = tmp_path / "bad_gate.json"
    mod_gate = json.loads(json.dumps(artifact))
    mod_gate["gates"]["winner_le_chatelier_and_causal_up_to_Mmax"] = False
    bad_gate.write_text(json.dumps(mod_gate), encoding="utf-8")
    ok, errs = b9a.validate_artifact(bad_gate)
    assert not ok and errs


def test_all_8_survivors_spinodal_crust_and_durca(artifact: dict) -> None:
    survs = artifact["survivor_microphysics"]
    assert len(survs) == 8
    winners = [s for s in survs if s["is_winner"]]
    assert len(winners) == 1

    for s in survs:
        cc = s["core_crust_transition"]
        nt_dyn = cc["dynamical"]["n_t_fm3"]
        nt_th = cc["thermodynamical"]["n_t_fm3"]
        assert 0.03 < nt_dyn < nt_th < s["n0_fm3"]
        assert cc["dynamical"]["schur_vs_fd_rel"] < 1.0e-8
        ic = s["star_1_4_crustal_inertia"]
        assert ic["I_total_int_rel_err"] < 1.0e-4
        assert ic["I_crust_over_I"] >= b9a.VELA_ICRUST_OVER_I_MIN
        th = s["thresholds"]
        assert th["M_max_Msun"] >= 2.08
        assert th["R_1_4_km"] <= 13.0
        assert th["Lambda_1_4"] <= 580.0


def test_le_chatelier_sound_speeds_and_cowling_controls(artifact: dict) -> None:
    wm = artifact["winner_microphysics"]
    mmax_nc = next(s["thresholds"]["n_c_max_fm3"] for s in artifact["survivor_microphysics"] if s["is_winner"])
    for row in wm["beta_equilibrium_profile"]:
        if row["n_fm3"] <= mmax_nc + 0.02:
            assert 0.0 < row["ce2"] < 1.0
            assert 0.0 < row["cs2"] < 1.0
            assert row["cs2_minus_ce2"] >= -1.0e-9

    ctrl = wm["controls"]
    kelvin = ctrl["uniform_kelvin_cowling"]
    assert kelvin["passed"] is True
    assert kelvin["rel_err"] < 1.0e-3

    null = ctrl["barotropic_null_at_1_4_Msun"]
    assert null["passed_no_gmodes"] is True
    assert null["sign_changes_found"] == 0

    modes = wm["cowling_asteroseismology_l2"]
    assert len(modes) == 6
    for row in modes:
        assert 1500.0 <= row["f_mode_Hz"] <= 2800.0
        assert 50.0 <= row["g1_mode_Hz"] <= 500.0
        assert max(row["g1_radial_nodes_W"], row["g1_radial_nodes_V"]) >= 1
        assert row["g2_mode_Hz"] is not None
        assert row["g2_mode_Hz"] < row["g1_mode_Hz"] < row["f_mode_Hz"]
