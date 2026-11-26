#!/usr/bin/env python3
"""Pytest suite for Bridge 7: Hartle I-Love-Q & Synchronized Two-Condensate Bounce."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

VERIF_DIR = Path(__file__).resolve().parent
if str(VERIF_DIR) not in sys.path:
    sys.path.insert(0, str(VERIF_DIR))

import nvg_unified_astrophysics_stability_bridge as b7  # noqa: E402


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(b7.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))


def test_validate_live_artifact() -> None:
    ok, errs = b7.validate_artifact(b7.DEFAULT_OUTPUT_PATH)
    assert ok, errs


def test_evidence_weight_and_gates(payload: dict) -> None:
    assert payload["evidence_weight"] == 0.0
    assert payload["provenance"]["zero_parameter_refit"] is True
    assert payload["status"] == b7.STATUS_PASS
    assert all(payload["gates"].values())


def test_maclaurin_newtonian_limits(payload: dict) -> None:
    ctrl = payload["part_a_hartle_iloveq"]["controls"]
    assert ctrl["maclaurin_I_abs_err"] < 1.0e-4
    assert ctrl["maclaurin_Q_abs_err"] < 1.0e-4
    assert ctrl["maclaurin_qtilde_abs_err"] < 2.0e-3


def test_four_benchmark_masses_and_j0737a(payload: dict) -> None:
    stars = payload["part_a_hartle_iloveq"]["benchmark_stars"]
    assert [s["label"] for s in stars] == ["PSR_J0737_3039A", "canonical_1_4", "heavy_1_8", "massive_2_0"]
    j0737 = payload["part_a_hartle_iloveq"]["psr_j0737_3039a_forecast"]
    assert abs(j0737["M_Msun"] - 1.3381) < 1.0e-4
    assert 1.1e45 < j0737["I_g_cm2"] < 2.0e45
    assert 10.0 < j0737["Ibar"] < 20.0
    assert 2.0 < j0737["Qbar"] < 9.0


def test_independent_iloveq_universality(payload: dict) -> None:
    assert payload["part_a_hartle_iloveq"]["max_yagi_yunes_rel_dev"] < 0.015
    for row in payload["part_a_hartle_iloveq"]["benchmark_stars"]:
        assert row["YY_IQ_rel_dev"] < 0.015
        assert row["YY_ILove_rel_dev"] < 0.015
        assert row["YY_QLove_rel_dev"] < 0.015


def test_hartle_surface_matching_and_integral_identity(payload: dict) -> None:
    assert payload["part_a_hartle_iloveq"]["max_surface_matching_rel_res"] < 1.0e-8
    for row in payload["part_a_hartle_iloveq"]["benchmark_stars"]:
        # nvg_hartle_slow_rotation computes inertia_integral via np.trapz on max_step=0.05 (~2.3e-5)
        assert row["first_order_integral_rel"] < 1.0e-4


def test_spin_rescaling_invariance_and_radial_convergence(payload: dict) -> None:
    ctrl = payload["part_a_hartle_iloveq"]["controls"]
    assert ctrl["spin_rescaling_invariance_rel_err"] < 1.0e-8
    assert ctrl["j0737_Ibar_radial_conv_rel_err"] < 1.0e-4
    assert ctrl["j0737_Qbar_radial_conv_rel_err"] < 1.0e-4


def test_coupled_bounce_constraint_and_baryon_conservation(payload: dict) -> None:
    pb = payload["part_b_coupled_two_condensate_bounce"]
    for key in ("vacuum_bag_kappa1_tracking", "w0_scale_mu20_tracking", "w0_scale_W_zero_crossing"):
        d = pb[key]["diagnostics"]
        assert d["bounce_signature_verified"] is True
        assert d["max_constraint_error"] < 1.0e-11
        assert d["max_baryon_conservation_error"] < 1.0e-12
        assert d["kretschmann_bounded_le_12"] is True


def test_coupled_bounce_crosses_w_zero_smoothly(payload: dict) -> None:
    d = payload["part_b_coupled_two_condensate_bounce"]["w0_scale_W_zero_crossing"]["diagnostics"]
    assert d["crossed_W_zero_smoothly"] is True
    assert d["yW_min"] < 0.0 < d["yW_max"]


def test_coupled_bounce_tracking_and_fluid_causality(payload: dict) -> None:
    pb = payload["part_b_coupled_two_condensate_bounce"]
    for key in ("vacuum_bag_kappa1_tracking", "w0_scale_mu20_tracking"):
        samples = pb[key]["diagnostics"]["fluid_cs2_samples"]
        for s in samples:
            assert 0.0 < s["fluid_cs2_eq"] < 1.0


def test_quintin_yoshida_benchmark_positive_along_bounce(payload: dict) -> None:
    pb = payload["part_b_coupled_two_condensate_bounce"]
    for key in ("vacuum_bag_kappa1_tracking", "w0_scale_mu20_tracking", "w0_scale_W_zero_crossing"):
        d = pb[key]["diagnostics"]
        assert d["qy_benchmark_Q_min"] > 0.0
        assert d["qy_benchmark_cs2_min"] >= 1.0 / 3.0 - 1.0e-12
        assert d["qy_benchmark_cs2_max"] < 3.0


def test_negative_control_bad_qdot_breaks_constraint(payload: dict) -> None:
    assert payload["part_b_coupled_two_condensate_bounce"]["negative_control_bad_qdot_constraint_err"] > 0.05


def test_negative_control_zeta_zero_raises_at_w_zero() -> None:
    p, _, _ = b7.load_law2_model_params()
    m = b7.TwoCondensateBounceMatter(p)
    m.zeta_A = 0.0
    with pytest.raises(ZeroDivisionError):
        m.solve_A0_hat(0.1, 0.0)


def test_fail_closed_rejects_tampered_json(tmp_path: Path, payload: dict) -> None:
    bad = json.loads(json.dumps(payload))
    bad["evidence_weight"] = 1.0
    p1 = tmp_path / "bad_ew.json"
    p1.write_text(json.dumps(bad), encoding="utf-8")
    ok, _ = b7.validate_artifact(p1)
    assert not ok

    bad2 = json.loads(json.dumps(payload))
    bad2["part_a_hartle_iloveq"]["psr_j0737_3039a_forecast"]["Ibar"] *= 1.5
    p2 = tmp_path / "bad_ibar.json"
    p2.write_text(json.dumps(bad2), encoding="utf-8")
    ok2, _ = b7.validate_artifact(p2)
    assert not ok2
