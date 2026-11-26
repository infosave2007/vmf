"""Semantic and scaling tests for the Zenodo 21812161 theta-DM haloscope protocol."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_theta_haloscope_design as design


@pytest.fixture(scope="module")
def result() -> dict:
    return design.build_result()


def test_zero_evidence_contract_and_benchmark_anchors(result: dict) -> None:
    assert result["evidence_weight"] == 0.0
    assert result["observed_likelihood"] is None
    assert result["zenodo_doi"] == "10.5281/zenodo.21812161"

    b = result["benchmark"]
    assert 1.06e11 <= b["f_a_GeV"] <= 1.08e11
    assert 53.0 <= b["m_theta_ueV"] <= 53.4
    assert 12.82 <= b["f_theta_GHz"] <= 12.90
    assert 2.05e-14 <= b["g_ksvz_GeV_inv"] <= 2.11e-14
    assert 7.9e-15 <= b["g_dfsz_GeV_inv"] <= 8.3e-15


def test_cnu_scaling_and_dielectric_stack_power_invariance() -> None:
    fa1, m1, f1 = design.seesaw_chain(1.0)
    fa2, m2, f2 = design.seesaw_chain(2.0)
    assert math.isclose(fa2 / fa1, 2.0, rel_tol=1e-12)
    assert math.isclose(m2 / m1, 0.5, rel_tol=1e-12)
    assert math.isclose(f2 / f1, 0.5, rel_tol=1e-12)

    g1, _ = design.photon_couplings(fa1)
    g2, _ = design.photon_couplings(fa2)
    assert math.isclose(g2 / g1, 0.5, rel_tol=1e-12)

    # Fixed dielectric stack power scales as (g / m)^2, which is c_nu-invariant
    a_disk = math.pi * (0.6 ** 2)
    p1 = design.P_stack_nat(g1, 10.0, a_disk, 1.0e4, 0.24, m1)
    p2 = design.P_stack_nat(g2, 10.0, a_disk, 1.0e4, 0.24, m2)
    assert math.isclose(p1, p2, rel_tol=1e-12)


def test_organ_calibration_and_axion_band_crosscheck(result: dict) -> None:
    cal = result["organ_calibration"]
    assert 0.95 <= cal["ratio_check"] <= 1.05

    band = result["benchmark"]["axion_band_crosscheck"]
    assert band["relative_offset"] < 0.03


def test_dielectric_stack_protocol_and_monte_carlo_window(result: dict) -> None:
    stk = result["protocols"]["fixed_dielectric_stack"]
    assert 15.0 <= stk["t_beta2_1e4_hours"] <= 22.0
    assert 1.0 <= stk["t_beta2_1e5_minutes"] <= 15.0

    mcc = result["monte_carlo"]["conditional_cnu1"]
    assert 12.80 <= mcc["percentiles_GHz"]["50"] <= 12.92
    assert 0.01 <= mcc["rel_1sigma"] <= 0.02
    assert mcc["z_local_5sigma_global"] > 6.5


def test_multichannel_bounds_and_neutrino_kinematics(result: dict) -> None:
    mch = result["multichannel"]
    assert mch["decay_to_nu_nu_forbidden"] is True
    assert mch["g_thee_dfsz"] < 3.0e-13
    assert mch["g_thee_ksvz"] < 3.0e-13
    assert 58.0 <= mch["sum_m_nu_meV"] <= 60.0
    assert 8.0 <= mch["m_beta_meV"] <= 10.0
    assert 0.005 < mch["y_nu_cnu1"] < 0.05
    assert 1.5 < mch["y_nu_over_y_tau"] < 2.2


def test_cli_json_smoke() -> None:
    proc = subprocess.run(
        [sys.executable, str(HERE / "nvg_theta_haloscope_design.py"), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == design.SCHEMA_VERSION
    assert payload["evidence_weight"] == 0.0
