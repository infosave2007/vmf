"""Focused semantic, mutation, and numerical checks for the source stellar artifact."""

from __future__ import annotations

import json
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_complete_stellar_prediction as pred


@pytest.fixture(scope="module")
def physical_pressure_ladder():
    point, _ = pred.eos_point_with_gates(0.1)
    coeff = pred._frobenius_coefficients(point)
    return [pred.integrate_pressure_rk4(0.1, dr=dr, _central_bundle=coeff, _central_point=point)
            for dr in (0.05, 0.025, 0.0125)]


def _valid_stored_artifact():
    base = json.loads(Path(pred.RESULT_PATH).read_text())
    ok, errors = pred._validate_result_contract(base)
    assert ok, errors  # Every mutation has a live, positive baseline.
    return base


def _assert_rejected_for(bad, reason):
    bad["canonical_result_sha256"] = pred.canonical_result_sha256(bad)
    ok, errors = pred._validate_result_contract(bad)
    assert not ok and reason in errors, errors


@pytest.fixture(scope="module")
def terminal_witness():
    row = pred.integrate_pressure_rk4(0.001, dr=0.05)
    audit = row["surface_event"]["jet"]["roundoff_audit"]
    assert audit["status"] == "PASS"
    assert pred._roundoff_audit_accepts(audit) is True
    return row


def test_upstream_digest_and_domain_are_live() -> None:
    assert pred.upstream_digest() == pred.EXPECTED_UPSTREAM_DIGEST
    assert pred._source_point(0.0).pressure == 0.0
    assert pred._source_point(10.0).pressure == pytest.approx(1692.0183516252062, rel=2e-10)
    with pytest.raises(ValueError):
        pred._source_point(10.000001)


def test_direct_gates_and_no_completion() -> None:
    for x in (1e-6, 1e-3, 1.0, 10.0):
        point, gates = pred.eos_point_with_gates(x)
        assert point.pressure > 0
        assert gates["strict_causality"] and gates["dPdn_positive"]
        assert gates["derivative_ladder_agree"]


def test_direct_bracketed_inversions_are_not_table_linear() -> None:
    eos = pred.DirectEOS()
    for x in (1.0e-3, 0.01, 0.1, 1.0, 5.0):
        source = pred._source_point(x, high_precision=True)
        assert abs(eos.at_h(source.enthalpy).x - x) / x < 1.0e-6
        assert abs(eos.at_pressure(source.pressure).x - x) / x < 1.0e-6


def test_hot_path_does_not_import_mpmath(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    original_import = builtins.__import__
    def no_mpmath(name: str, *args: object, **kwargs: object):
        if name == "mpmath":
            raise AssertionError("mpmath is forbidden on the RHS hot path")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_mpmath)
    eos = pred.DirectEOS()
    for x in (0.001, 0.1, 1.0):
        src = pred._source_point(x)
        assert eos.at_h(src.enthalpy).x > 0.0
        assert eos.at_pressure(src.pressure).x > 0.0
    with pytest.raises(AssertionError, match="mpmath"):
        pred.enthalpy_oracle_controls()


def test_positive_low_density_inversion_never_returns_vacuum() -> None:
    eos = pred.DirectEOS()
    h = 6.847455908465427e-13
    q = eos.at_h(h)
    assert q.x > 0.0 and q.cs2 is not None
    assert eos.at_pressure(q.pressure).x > 0.0
    with pytest.raises(ValueError, match="outside|exceeds"):
        eos.at_h(-1.0)


def test_positive_log_fallback_clamps_every_source_trial() -> None:
    """Regression: endpoint expansion must never evaluate x outside 0..10."""
    eos = pred.DirectEOS()
    original = eos._source
    seen: list[float] = []

    def wrapped(x: float, *, high_precision: bool = True):
        seen.append(float(x))
        assert 0.0 <= float(x) <= pred.X_MAX
        return original(x, high_precision=high_precision)

    eos._source = wrapped  # type: ignore[method-assign]
    q = eos.at_h(0.9337015931839977)
    assert q.x > 0.0
    assert seen and max(seen) <= pred.X_MAX


def test_positive_log_endpoint_mutation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: a non-bracketing exact endpoint cannot be clipped into output."""
    eos = pred.DirectEOS()
    eos._h_max = 1.3  # permit the deliberately unreachable target through the public gate
    original = eos._source

    def muted_endpoint(x: float, *, high_precision: bool = True):
        point = original(x, high_precision=high_precision)
        if float(x) >= pred.X_MAX:
            return replace(point, enthalpy=0.8 * point.enthalpy)
        return point

    monkeypatch.setattr(eos, "_source", muted_endpoint)
    with pytest.raises(ValueError, match="certificate failed"):
        eos.at_h(1.25)


def test_positive_density_w_root_has_no_w0_shortcut() -> None:
    w, cert = pred._solve_W_certified(1.0e-12 * pred.upstream.PARAMS.n0_nat)
    assert cert.get("endpoint") is False
    assert cert.get("iterations", 0) > 0
    assert cert.get("w_lo") is not None and cert.get("w_hi") is not None
    assert cert.get("residual_rel", 1.0) <= 2.0e-12


def test_w_continuation_hint_is_the_certified_local_bracket() -> None:
    """A valid neighbouring W hint must drive, not merely probe, the solve."""
    n0 = float(pred.upstream.PARAMS.n0_nat)
    w0, c0 = pred._solve_W_certified(n0)
    w_plain, plain = pred._solve_W_certified(2.0 * n0)
    w_hint, hinted = pred._solve_W_certified(2.0 * n0, hint=w0)
    assert hinted["local_bracket"] is True and hinted["hint_used"] is True
    glo = pred.upstream.PARAMS.W0 / np.sqrt(3.0); ghi = 2.0 * pred.upstream.PARAMS.W0
    assert glo < hinted["w_lo"] < hinted["w_hi"] < ghi
    assert hinted["iterations"] < plain["iterations"]
    assert abs(w_hint - w_plain) / max(abs(w_plain), 1.0) < 2.0e-12
    assert hinted["bracket_rel"] <= 2.0e-13 and hinted["residual_rel"] <= 2.0e-12


def test_post_newton_probe_replaces_only_a_direct_straddling_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation guard: a probe that is merely recorded/discarded is invalid."""
    source = inspect.getsource(pred._solve_W_certified)
    assert "post_probe_accepted" in source
    assert "lo, hi, flo, fhi = float(a), float(b), float(fa), float(fb)" in source
    n0 = float(pred.upstream.PARAMS.n0_nat)
    w0, _ = pred._solve_W_certified(n0)
    w, cert = pred._solve_W_certified(2.0 * n0, hint=w0)
    probe = cert.get("post_newton_probe", {})
    assert probe.get("attempts", 0) >= 1 and probe.get("accepted") is True
    glo = float(pred.upstream.PARAMS.W0 / np.sqrt(3.0)); ghi = float(2.0 * pred.upstream.PARAMS.W0)
    assert glo < cert["w_lo"] < cert["w_hi"] < ghi
    assert cert["post_newton_probe_accepted"] is True
    # Replacing the accepted pair by the immutable global bracket is the
    # prohibited discard mutation; its width cannot satisfy the certificate.
    discarded = dict(cert); discarded["w_lo"], discarded["w_hi"] = glo, ghi
    assert (discarded["w_hi"] - discarded["w_lo"]) / max(abs(w), 1.0e-300) > 2.0e-13


def test_xc_one_continuation_performance_witness() -> None:
    """Repair-10's 11,723-W-solve/48-iteration witness must materially drop."""
    import time
    eos = pred.DirectEOS(); eos.begin_star(); start = time.perf_counter()
    pred.integrate_enthalpy(1.0, rtol=1.0e-6, strict_eos=True, eos=eos)
    elapsed = time.perf_counter() - start; stats = eos.end_star()
    # Timing is diagnostic only; physics certificates remain the direct W and
    # ODE gates.  These generous bounds detect a discarded continuation hint.
    assert int(stats["w_solves"]) < 0.9 * 11723
    assert int(stats["w_max_iterations"]) < 48
    assert elapsed >= 0.0  # telemetry, never a scientific acceptance threshold


def test_outer_records_reduce_strict_source_work_and_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation removing h/P records must restore the repair-10 workload."""
    import time
    eos = pred.DirectEOS(); eos.begin_star(); t0 = time.perf_counter()
    pred.integrate_enthalpy(1.0, rtol=1.0e-10, strict_eos=True, eos=eos)
    fast_time = time.perf_counter() - t0; fast = eos.end_star()
    original = pred.DirectEOS.at_h
    def discard_records(self, h, **kwargs):
        self._h_records.clear(); self._p_records.clear()
        return original(self, h, **kwargs)
    monkeypatch.setattr(pred.DirectEOS, "at_h", discard_records)
    slow_eos = pred.DirectEOS(); slow_eos.begin_star(); t1 = time.perf_counter()
    pred.integrate_enthalpy(1.0, rtol=1.0e-10, strict_eos=True, eos=slow_eos)
    slow_time = time.perf_counter() - t1; slow = slow_eos.end_star()
    assert len(eos._h_records) == 0  # cleared by end_star lifecycle
    assert int(fast["w_solves"]) < 0.9 * 11795
    assert int(slow["w_solves"]) > int(fast["w_solves"])
    # Wall time is diagnostic only and can vary with process scheduling; the
    # certified material-work reduction is the W-solve count above.
    assert slow_time >= 0.0 and fast_time >= 0.0


def test_source_cache_retains_immutable_result_and_w_certificate() -> None:
    eos = pred.DirectEOS(); eos.begin_star()
    point = eos._source(0.1)
    item = next(iter(eos._source_cache.values()))
    assert isinstance(item["result"], pred.SourceResult)
    assert item["result"].point == point
    assert item["result"].certificate()["iterations"] > 0
    assert pred._source_result(0.1).w_certificate
    assert eos._stats["w_solves"] >= 1


def test_shared_stationary_queue_is_direct_and_bounded() -> None:
    class Synthetic:
        def __init__(self) -> None:
            self.final_cache = {}; self.witness_cache = {}; self.stationary_cache = {}
            self.stationary_active = False; self.stationary_additions = 0
        def final(self, x: float) -> dict[str, float]:
            u = float(np.log(x)); return {"M_Msun": 3.0 - (u - 0.17) ** 2}
        def witness(self, x: float) -> dict[str, float]:
            return self.final(x)
    roots = pred._derivative_roots(Synthetic(), -0.4, 0.8)
    assert len(roots) == 3 and all(r["shared_queue"] and r["direct_sign_change"] for r in roots)
    assert max(r["root_iterations"] for r in roots) <= 24


def test_mutation_gates_fail_closed() -> None:
    base = {
        "schema_version": pred.SCHEMA_VERSION,
        "upstream_source_sha256": pred.EXPECTED_UPSTREAM_DIGEST,
        "provenance": {"action_sha256": pred.file_digest(pred.ACTION_PATH), "contract_sha256": pred.file_digest(pred.CONTRACT_PATH), "producer_sha256": pred.file_digest(pred.ROOT / "source_complete_stellar_prediction.py"), "test_sha256": pred.file_digest(pred.TEST_PATH)},
        "contract": {"degeneracy": 4, "domain_x_min": 0.0, "domain_x_max": 10.0, "surface": "exact_h_zero_vacuum", "pressure_definition": "mu*n-epsilon_and_hilbert_crosscheck", "q_phi_and_gs": "derived_from_upstream_params", "endpoint_policy": "exact_no_clamp_no_extrapolation", "beta_equilibrium": None, "leptons": None, "units": {"k_conv": pred.K_CONV, "M_sun_km": pred.M_SUN_KM}, "crust_model": None, "css_branch": None, "dedp_clipping": False},
        "evidence_weight": 0, "physical_NS_blocker": pred.PHYSICAL_NS_BLOCKER, "status": "FAIL_CLOSED_NUMERICAL_CONVERGENCE", "rows": [],
    }
    base["canonical_result_sha256"] = pred.canonical_result_sha256(base)
    ok, _ = pred._validate_result_contract(base)
    assert ok
    for mutation in (
        ("degeneracy", 2), ("domain_x_max", 11.0), ("crust_model", "polytrope"),
        ("css_branch", {"cs2": 1 / 3}), ("dedp_clipping", True),
        ("endpoint_policy", "clamp"), ("beta_equilibrium", True), ("leptons", ["e"]),
        ("units", {"k_conv": 1.0, "M_sun_km": pred.M_SUN_KM}),
    ):
        bad = json.loads(json.dumps(base)); bad["contract"][mutation[0]] = mutation[1]
        assert not pred._validate_result_contract(bad)[0]
    bad = json.loads(json.dumps(base)); bad["evidence_weight"] = 1
    assert not pred._validate_result_contract(bad)[0]
    bad = json.loads(json.dumps(base)); bad["physical_NS_blocker"] = None
    assert not pred._validate_result_contract(bad)[0]


def test_observation_mutations_fail_closed() -> None:
    base = _valid_stored_artifact()
    for mutate in ("evidence_weight", "comparison_status", "value"):
        bad = json.loads(json.dumps(base))
        if mutate == "evidence_weight":
            bad["observational_context"][0][mutate] = 1
        elif mutate == "comparison_status":
            bad["observational_context"][0][mutate] = "PASS"
        else:
            bad["observational_context"][0][mutate] = "fit confirms model"
        _assert_rejected_for(bad, "observation_language" if mutate == "value" else "observation_claim_or_weight")


def test_dual_integrator_and_units(physical_pressure_ladder) -> None:
    primary = pred.integrate_enthalpy(0.1, rtol=1e-10)
    independent = physical_pressure_ladder[-1]
    assert primary["M_Msun"] > 0 and primary["R_km"] > 0
    assert independent["M_Msun"] > 0 and independent["R_km"] > 0
    for key, limit in (("M_Msun", 1e-3), ("R_km", 1e-3), ("k2", 5e-3), ("Lambda", 5e-3)):
        assert pred._rel(primary[key], independent[key]) < limit
    assert pred.M_SUN_KM == pytest.approx(1.4766)
    assert pred.K_CONV == pytest.approx(1.3234e-6)


def test_pressure_ladder_uses_distinct_actual_meshes(physical_pressure_ladder):
    rows = physical_pressure_ladder
    assert [r["mesh"]["actual_dr_km"] for r in rows] == [0.05, 0.025, 0.0125]
    counts = [r["mesh"]["full_steps"] for r in rows]
    assert counts[1] == pytest.approx(2 * counts[0], abs=2)
    assert counts[2] == pytest.approx(2 * counts[1], abs=2)
    assert len({r["M_Msun"] for r in rows}) == 3
    for r in rows:
        assert r["launch"]["pressure_direct_rel"] < 2e-3
        assert r["centre_series"]["direct_residual_certificate"]["pass"] is True


def test_source_radius_boundary_matches_dimensional_tov_coefficients():
    point, _ = pred.eos_point_with_gates(0.1)
    c = pred._frobenius_coefficients(point)
    eg, pg = point.eps * pred.K_CONV, point.pressure * pred.K_CONV
    assert c["P2"] * pred.K_CONV**2 == pytest.approx(-2*np.pi*(eg+pg)*(eg/3+pg), rel=2e-12)
    assert c["m3"] * pred.K_CONV == pytest.approx(4*np.pi*eg/3, rel=2e-12)
    # Each isolated monomial must cross the radius boundary independently.
    radius = 0.03
    zero = {k: 0.0 for k in ("Pc", "P2", "P4", "P6", "m3", "m5", "m7", "m9", "y2", "y4", "y6")}
    for key, component, power, scale in (("P2", 1, 2, pred.K_CONV**2), ("P6", 1, 6, pred.K_CONV**4), ("m9", 0, 9, pred.K_CONV**4), ("y4", 2, 4, pred.K_CONV**2)):
        q = dict(zero); q[key] = 1.0
        state = pred._frobenius_series_state(q, radius, 6)
        expected = scale * radius**power
        if component == 2:
            # Increase the coefficient so subtraction from y=2 is resolved.
            q[key] = 1e15; state = pred._frobenius_series_state(q, radius, 6)
            expected *= 1e15
        assert state[component] - (2 if component == 2 else 0) == pytest.approx(expected, rel=1e-10, abs=0)


def test_primary_launch_owns_its_actual_source_certificate():
    row = pred.integrate_enthalpy(0.1)
    assert pred._primary_launch_errors(row) == []
    assert "launch" not in row
    bad = json.loads(json.dumps(row)); bad["centre_launch"]["r_km"] *= np.sqrt(pred.K_CONV)
    assert pred._primary_launch_errors(bad) == ["primary_centre_unit_boundary"]
    bad = json.loads(json.dumps(row)); bad["centre_launch"]["source_certificate"]["jacobian"] = -1
    assert pred._primary_launch_errors(bad) == ["primary_centre_source_branch"]


def test_exact_zero_pressure_derivative_order_and_low_density_limit():
    r, m, y = 30.0, 0.8, 0.7
    vacuum = pred._pressure_vacuum_rhs(r, m, y)
    assert vacuum[0] == 0.0 and vacuum[1] < 0.0
    q = pred._point_with_derivatives(1e-18)
    eg, pg = q.eps * pred.K_CONV, q.pressure * pred.K_CONV
    dPdr = -(eg+pg)*(m+4*np.pi*r**3*pg)/(r*(r-2*m))/pred.K_CONV
    dwdr = 0.4*q.pressure**(-0.6)*dPdr
    assert vacuum[1] == pytest.approx(dwdr, rel=1e-5)
    hzero = pred._rhs_h(0.0, np.asarray([r, m, y]), pred.DirectEOS())
    assert vacuum[2] == pytest.approx(hzero[2]/hzero[0], rel=1e-14)


def test_elapsed_telemetry_is_not_scientific_identity():
    a = {"proof": {"value": 2.0, "wall_seconds": 1.0}}
    b = {"proof": {"value": 2.0, "wall_seconds": 9.0}}
    assert pred._canonical_full_bytes(a) == pred._canonical_full_bytes(b)
    assert pred.canonical_result_sha256(a) == pred.canonical_result_sha256(b)
    assert pred._proof_payload_digest(a) == pred._proof_payload_digest(b)
    b["proof"]["value"] = 3.0
    assert pred._proof_payload_digest(a) != pred._proof_payload_digest(b)


def test_pressure_integrator_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("pressure path called enthalpy fallback")
    monkeypatch.setattr(pred, "integrate_enthalpy", forbidden)
    result = pred.integrate_pressure_rk4(1.0, dr=0.0125)
    assert result["M_Msun"] > 0.0 and result["R_km"] > 0.0


def test_missing_surface_event_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        success = True
        message = "fake"
        y = np.asarray([[1.0], [0.1], [2.0]])
        t = np.asarray([0.0])
        t_events = [np.asarray([])]
    monkeypatch.setattr(pred, "solve_ivp", lambda *args, **kwargs: Fake())
    with pytest.raises(ValueError, match="surface event"):
        pred.integrate_enthalpy(1.0, rtol=1e-6)


def test_deterministic_serialization(tmp_path: Path) -> None:
    # Avoid a full 129-row production run in this focused test; byte identity
    # of the writer is what is being checked.
    result = _valid_stored_artifact()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    pred.write_artifacts(result, result_path=a, report_path=tmp_path / "a.md")
    pred.write_artifacts(result, result_path=b, report_path=tmp_path / "b.md")
    assert a.read_bytes() == b.read_bytes()


def test_writer_rejects_invalid_result_before_writing(tmp_path):
    with pytest.raises(ValueError, match="artifact contract failed"):
        pred.write_artifacts({"schema_version": pred.SCHEMA_VERSION, "status": "PASS"}, result_path=tmp_path/"invalid.json", report_path=tmp_path/"invalid.md")
    assert not list(tmp_path.iterdir())


def test_centre_float_alias_mutation_with_fresh_digest_fails():
    import copy
    point, _ = pred.eos_point_with_gates(0.1)
    base = pred._frobenius_coefficients(point)
    pred._ensure_centre_proof(base)
    assert pred._centre_proof_accepts(base)
    for key in ("Pc", "P2", "P4", "P6", "m3", "m5", "m7", "m9", "y2", "y4", "y6", "A"):
        bad = copy.deepcopy(base); bad[key] *= 2
        bad["centre_proof_record"]["coefficient_digest"] = pred._centre_coeff_digest(bad)
        assert not pred._centre_proof_accepts(bad), key
        assert pred._frobenius_residual_certificate(bad, pred.DirectEOS())["reason"] == "centre_coefficient_projection"


def test_required_identity_and_finite_payload_rejections():
    base = _valid_stored_artifact()
    base.pop("canonical_result_sha256")
    assert "canonical_numeric_identity" in pred._validate_result_contract(base)[1]
    base["rows"][0]["M_Msun"] = float("nan")
    assert pred._validate_result_contract(base) == (False, ["nonfinite_or_unserializable_payload"])


def test_injected_validation_session_calls_generator_once() -> None:
    candidate = {"schema_version": pred.SCHEMA_VERSION, "value": 3}
    calls = {"n": 0}
    def generator() -> dict[str, object]:
        calls["n"] += 1
        return {"schema_version": pred.SCHEMA_VERSION, "value": 3}
    session = pred._validation_session(generator)
    assert session.validate(candidate)[0] is True and calls["n"] == 1
    assert session.validate(candidate)[0] is False and calls["n"] == 1


def test_stationary_root_tracks_injected_shift_without_hardcoding() -> None:
    class Synthetic:
        def final(self, x: float) -> dict[str, float]:
            u = float(np.log(x)); return {"M_Msun": 3.0 - (u - 0.17) ** 2}
        def witness(self, x: float) -> dict[str, float]:
            return self.final(x)
    root = pred._derivative_root(Synthetic(), -0.4, 0.8, 2.5e-3)
    assert abs(root["u"] - 0.17) < 2.5e-7
    assert root["bracket_width"] <= 2.5e-7


def test_stationary_root_predictor_miss_is_recovered_by_direct_subdivision() -> None:
    class ShiftedNoisy:
        def final(self, x: float) -> dict[str, float]:
            u = float(np.log(x))
            # A broad asymmetric maximum intentionally places the old cubic
            # predictor away from the direct derivative sign interval.
            return {"M_Msun": 3.0 - (u - 0.2137) ** 2 + 0.04 * (u - 0.2137) ** 3}

        def witness(self, x: float) -> dict[str, float]:
            return self.final(x)

    root = pred._derivative_root(ShiftedNoisy(), -0.4, 0.8, 2.5e-3)
    assert root["direct_sign_change"] is True
    assert root["subdivision_points"] == 17
    assert root["root_iterations"] <= 64
    assert root["bracket_width"] <= 2.5e-7


def test_stationary_root_rejects_absent_or_multiple_sign_changes() -> None:
    class Monotone:
        def final(self, x: float) -> dict[str, float]:
            return {"M_Msun": float(np.log(x))}

    class Multiple:
        def final(self, x: float) -> dict[str, float]:
            u = float(np.log(x))
            return {"M_Msun": -(u * u - 0.09) ** 2}

    with pytest.raises(ValueError, match="absent or multiple"):
        pred._derivative_root(Monotone(), -0.4, 0.8, 2.5e-3)
    with pytest.raises(ValueError, match="absent or multiple"):
        pred._derivative_root(Multiple(), -0.6, 0.6, 2.5e-3)


def test_strict_evaluator_and_exact_cache_keys() -> None:
    class Stub:
        def __call__(self, x: float, **kwargs: object) -> dict[str, object]:
            assert kwargs["strict_eos"] is True and kwargs["rtol"] == 1.0e-10
            return {"M_Msun": x, "R_km": x, "k2": x, "Lambda": x}
    original = pred.integrate_enthalpy
    pred.integrate_enthalpy = Stub()  # type: ignore[assignment]
    try:
        ev = pred._StrictEvaluator.__new__(pred._StrictEvaluator)
        ev.eos = object()  # type: ignore[assignment]
        ev.final_cache = {}; ev.witness_cache = {}
        assert ev.final(0.25)["M_Msun"] == 0.25
        assert all(key.startswith("0x") for key in ev.final_cache)
    finally:
        pred.integrate_enthalpy = original  # type: ignore[assignment]


def test_owner_registry_exact_key_dedup_and_mutation_closure() -> None:
    reg = pred._OwnerEvaluationRegistry()
    calls = {"n": 0}
    def run() -> dict[str, float]:
        calls["n"] += 1
        return {"M_Msun": 1.0, "R_km": 2.0}
    key = reg.make_key("grid32", 0.25, 1.0e-10, 1.0, 1.0, True)
    assert reg.get_or_execute(key, run)[1] is False
    assert reg.get_or_execute(key, run)[1] is True
    assert calls["n"] == 1
    for mutated in (
        reg.make_key("grid64", 0.25, 1.0e-10, 1.0, 1.0, True),
        reg.make_key("grid32", 0.25000000000000006, 1.0e-10, 1.0, 1.0, True),
        reg.make_key("grid32", 0.25, 1.0e-9, 10.0, 1.0, True),
        reg.make_key("grid32", 0.25, 1.0e-10, 1.0, 0.5, True),
        reg.make_key("grid32", 0.25, 1.0e-10, 1.0, 1.0, False),
    ):
        assert reg.get_or_execute(mutated, run)[1] is False
    snap = reg.snapshot()
    assert calls["n"] == 6 and snap["requested"] == 7 and snap["unique"] == 6 and snap["prior_hit"] == 1
    assert snap["executed"] == snap["unique"] and snap["reused"] == snap["prior_hit"]
    off = pred._OwnerEvaluationRegistry(enabled=False)
    assert off.get_or_execute(key, run)[1] is False and off.get_or_execute(key, run)[1] is False
    assert off.snapshot()["executed"] == 2 and off.snapshot()["prior_hit"] == 0


def test_stale_maximum_recheck_cannot_be_forced_pass() -> None:
    base = _valid_stored_artifact()
    eta = base["convergence"]["grid_doubling"]["uncertainty_eta_M"]["32"]
    bad = json.loads(json.dumps(base))
    bad["maximum_recheck_rel"]["M_Msun"] = float(eta) + 1.0e-6
    _assert_rejected_for(bad, "maximum_direct_recheck")


def test_top_level_status_mutation_fails_closed() -> None:
    base = _valid_stored_artifact()
    bad = json.loads(json.dumps(base)); bad["status"] = "PASS"
    _assert_rejected_for(bad, "status_mutation")


def test_postmaximum_target_payload_is_rejected() -> None:
    base = _valid_stored_artifact()
    bad = json.loads(json.dumps(base))
    target = next(iter(bad["targets"]))
    bad["targets"][target]["status"] = "BLOCKED_TARGET_OUTSIDE_SEQUENCE"
    bad["targets"][target]["Lambda"] = 1.0
    _assert_rejected_for(bad, "target_extrapolation")


def test_non_strict_integrator_is_closed() -> None:
    with pytest.raises(ValueError, match="non-strict EOS"):
        pred.integrate_enthalpy(1.0, strict_eos=False)


def test_scaled_love_newtonian_control_and_oracle() -> None:
    k2, lam, meta = pred._love_surface_match(1.0e-8, 0.7)
    assert k2 == pytest.approx((2.0 - 0.7) / (2.0 * (0.7 + 3.0)), rel=2.0e-6)
    assert meta["oracle_dps"] == [80, 120]
    assert meta["tail_relative"] <= 2.0e-15


def test_love_oracle_metadata_mutations_fail_closed() -> None:
    base = _valid_stored_artifact()
    row = next(r for r in base["rows"] if r.get("love_surface", {}).get("method") == "scaled_C5_series")
    for key, value in (("oracle_dps", [40, 40]), ("oracle_80_k2", row["k2"] * 1.01), ("tail_relative", 1.0)):
        bad = json.loads(json.dumps(base)); target = next(r for r in bad["rows"] if r.get("love_surface", {}).get("method") == "scaled_C5_series"); target["love_surface"][key] = value
        _assert_rejected_for(bad, "love_surface_oracle")


def test_centre_offset_mutation_fails_closed() -> None:
    base = _valid_stored_artifact()
    bad = json.loads(json.dumps(base)); bad["rows"][0]["solver"]["centre_delta_scale"] = 1.0e4
    _assert_rejected_for(bad, "row_not_matched_final_evaluator")


def test_forced_convergence_boolean_fails_closed() -> None:
    base = _valid_stored_artifact()
    bad = json.loads(json.dumps(base)); bad["convergence"]["pass"] = True
    bad["convergence"]["grid_doubling"]["pass"] = False
    _assert_rejected_for(bad, "forced_convergence_boolean")


def test_scaled_series_mutation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = pred._love_series

    def shifted(C: float, y: float):
        k2, lam, meta = original(C, y)
        return k2 * 1.01, lam * 1.01, meta

    monkeypatch.setattr(pred, "_love_series", shifted)
    with pytest.raises(ValueError, match="oracle disagreement"):
        pred._love_surface_match(0.1, 0.7)


def test_q_warm_chain_rule_component_equivalence() -> None:
    eos = pred.DirectEOS(); h = pred._source_point(1.0).enthalpy * 0.7
    state = np.asarray([8.0, 0.8, 1.4]); q = 0.2
    ordinary = pred._rhs_h(h, state, eos, strict_eos=True)
    transformed = pred._q_chain_rhs(q, h + q * q, state, eos)
    drdh = float(ordinary[0]); dydr = float(ordinary[2] / drdh)
    expected_y = -2.0 * q * dydr * drdh
    assert transformed[2] == pytest.approx(expected_y, rel=2.0e-12, abs=2.0e-14)
    assert np.isfinite(transformed).all()


def test_gamma2_newtonian_love_control() -> None:
    # n=1 (Gamma=2) Newtonian polytrope control value used by the independent
    # pressure sequence; this is not an EOS row or a physical claim.
    expected = (15.0 - np.pi**2) / (2.0 * np.pi**2)
    assert expected == pytest.approx(0.2599088773175333, rel=1.0e-14)


def test_pressure_inversion_certificate_and_endpoint() -> None:
    eos = pred.DirectEOS()
    for x in (1.0e-3, 0.1, 1.0, 5.0):
        src = pred._source_point(x, high_precision=True)
        got = eos.at_pressure(src.pressure)
        meta = eos._p_cache[eos._key("P", src.pressure)]["certificate"]
        assert abs(got.x - x) / x < 1.0e-9
        assert meta["residual_log"] <= 2.0e-12 and meta["bracket_rel"] <= 2.0e-12
    top = eos.at_pressure(eos._p_max)
    assert top.x == pytest.approx(10.0, abs=1.0e-14)


def test_frobenius_series_and_terminal_jet_controls() -> None:
    point, _ = pred.eos_point_with_gates(0.1)
    coeff = pred._frobenius_coefficients(point)
    s4 = pred._frobenius_series_state(coeff, 0.0125, 4)
    s6 = pred._frobenius_series_state(coeff, 0.0125, 6)
    assert coeff["P2"] < 0.0 and coeff["m3"] > 0.0 and coeff["y2"] < 0.0
    assert np.isfinite(s4).all() and np.isfinite(s6).all()
    jet_r, jet_state, meta = pred._terminal_k_jet(10.0, np.asarray([1.0, 1.0e-12, 0.7]), pred.DirectEOS())
    # Repair 13 forbids the historical input-radius snap: the returned radius
    # is the literal N=64 terminal state (which may differ by a few ulps).
    assert np.isfinite(jet_r) and jet_state.shape == (2,)
    assert meta["degree"] == 10 and meta["diagnostic_degree"] == 8 and isinstance(meta["pass"], bool)
    assert meta["roundoff_audit"].get("independent") is True
    assert meta["coordinate"] == "shifted_delta_z"
    assert meta["raw_projection"] == "N64_shifted_correction_binary64"
    cse = meta["terminal_source_cse"]
    assert cse["key"] == "k.hex" and cse["shared_across"] == ["N16", "N32", "N64", "DOP853"]
    assert int(cse["requests"]) >= int(cse["evaluations"]) >= 1
    assert coeff["central_coefficient_cse"]["key"] == "x.hex"
    assert coeff["central_coefficient_cse"]["shared_bundle"] is True


def test_terminal_roundoff_audit_is_directed_and_boundary_mutation_closed(terminal_witness) -> None:
    """Bitwise equality has no force unless the independent cell proof passes."""
    q = terminal_witness
    audit = q["surface_event"]["jet"]["roundoff_audit"]
    assert audit["independent"] is True
    assert audit["status"] == "PASS"
    assert isinstance(audit.get("validated_interval"), dict)
    assert audit["validated_interval"]["proof"] == "validated_residual_tube"
    series = audit.get("love_series")
    assert isinstance(series, dict)
    assert 1 <= int(series["K"]) <= 128
    assert float(series["tail_abs"]) >= 0.0 and float(series["tail_rel"]) <= 2.0e-15
    assert series["tail_certificate"] is True
    assert series["denominator_separation"] is True
    assert isinstance(series["collapsed_point_agreement"], dict)
    if not series["collapsed_point_agreement"]["applicable"]:
        assert series["collapsed_point_agreement"]["pass"] is None
    if audit["status"] == "PASS":
        assert audit["interval_error_check_pass"] is True
        assert all(audit["direct_error_check_by_quantity"].values())
    import copy
    mutated = copy.deepcopy(audit)
    cell = mutated["derived_rounding_cells"]["M_Msun"]
    cell["interval"][0] = cell["lower_boundary"]
    cell["pass"] = True
    mutated["status"] = "PASS"
    mutated["interval_error_check_pass"] = True
    mutated["direct_error_check_by_quantity"] = {k: True for k in mutated["direct_error_check_by_quantity"]}
    assert pred._roundoff_audit_accepts(mutated) is False
    tube_mutation = copy.deepcopy(audit)
    tube_mutation["validated_interval"]["refined"]["cells"][0]["self_map_numerical_check"] = False
    assert pred._roundoff_audit_accepts(tube_mutation) is False
    for mutation in ("K", "tail_abs", "tail_rel", "denominator_separation", "coefficient_contract", "tail_contract", "widening_contract", "lambda_contract"):
        bad_series = copy.deepcopy(audit)
        if mutation == "K":
            bad_series["love_series"][mutation] = 0
        elif mutation == "tail_abs":
            bad_series["love_series"][mutation] = "-1"
        elif mutation == "tail_rel":
            bad_series["love_series"][mutation] = "1"
        elif mutation == "denominator_separation":
            bad_series["love_series"][mutation] = False
        else:
            bad_series["love_series"][mutation] = "mutated"
        assert pred._roundoff_audit_accepts(bad_series) is False
    for mutation in ("same_ivp_identity", "intersection_box", "source_context", "adaptive_plan"):
        bad = copy.deepcopy(audit)
        if mutation == "same_ivp_identity":
            bad[mutation] = False
        elif mutation == "intersection_box":
            bad[mutation][0][1] = "1"
        elif mutation == "source_context":
            bad["validated_interval"]["coarse"][mutation]["proof_revision"] = "cross-precision"
        else:
            bad["validated_interval"]["coarse"][mutation]["initial_leaves"] = 32
        assert pred._roundoff_audit_accepts(bad) is False


def test_scaled_c5_interval_contract_is_literal_and_no_cancellation_path() -> None:
    import inspect
    src = inspect.getsource(pred._terminal_roundoff_audit_impl)
    # The independent interval path must use the four maintained b_j
    # coefficients, fixed predeclared K, outward geometric tail, widened
    # denominator, and C^-5 Love map.  This catches accidental regression to
    # the original cancelling denominator or an unbounded/ad-hoc series.
    for token in ("b = [2-y, 6*y-10, 16-12*y, 8*y-8]", "a0 = 16*(y+3)/5", "Kser = 64", "tail_sup = 96*Bsup", "dhat_wide", "C**-5", "coefficient_contract", "widening_contract"):
        assert token in src
    assert "denominator_separation" in src and "tail_certificate" in src


def test_terminal_self_map_uses_propagated_e0_not_q(terminal_witness) -> None:
    """The serialized partial-time map must equal B(q,h) from incoming e0."""
    q = terminal_witness
    audit = q["surface_event"]["jet"]["roundoff_audit"]
    import copy
    mutated = copy.deepcopy(audit)
    row = mutated["validated_interval"]["coarse"]["cells"][0]
    row["partial_time_B"] = list(row["tube_radius"])
    assert pred._roundoff_audit_accepts(mutated) is False


def test_terminal_replay_binds_hermite_parent_and_unrelaxed_limits(terminal_witness):
    import copy
    audit = terminal_witness["surface_event"]["jet"]["roundoff_audit"]
    assert audit["interval_error_check_pass"] is True
    assert audit["rounding_cell_check_pass"] == all(audit["rounding_cells_diagnostic"].values())
    bad = copy.deepcopy(audit)
    bad["validated_interval"]["coarse"]["cells"][0]["leaf_rows"][0]["original_inputs"]["exact_dyadic"]["hermite_s0"]["mantissa"] += 2
    bad["replay_digest"] = pred._proof_payload_digest(bad)
    assert pred._roundoff_audit_accepts(bad) is False
    bad = copy.deepcopy(audit)
    bad["direct_n64_error"]["R_km"]["limit"] = "1"
    bad["replay_digest"] = pred._proof_payload_digest(bad)
    assert pred._roundoff_audit_accepts(bad) is False


def test_hermite_increment_form_manufactured_constant_linear_cubic() -> None:
    for z0, z1, f0, f1, expected, expected_d in (
        (3.0, 3.0, 0.0, 0.0, 3.0, 0.0),
        (2.0, 3.5, 1.5, 1.5, 2.75, 1.5),
        (1.0, 8.0, 3.0, 12.0, 3.375, 6.75),
    ):
        value, deriv = pred._hermite_increment_form(z0, z1, f0, f1, 1.0, 2.0, 1.5)
        assert value == pytest.approx(expected, rel=1e-13, abs=1e-13)
        assert deriv == pytest.approx(expected_d, rel=1e-13, abs=1e-13)


def test_terminal_proof_uses_full_cell_step_and_increment_basis() -> None:
    import inspect
    src = inspect.getsource(pred._terminal_roundoff_audit_impl)
    assert "a2 = 3*delta - h*(2*f0[i] + f1[i])" in src
    assert "a3 = -2*delta + h*(f0[i] + f1[i])" in src
    assert "hI = I(h/nsub)" in src
    assert "nsub = 2" in src and "nsub *= 2" in src
    assert "hI = I(h/8.0)" not in src


def test_terminal_audit_cache_is_local_and_toggleable(terminal_witness) -> None:
    jet = terminal_witness["surface_event"]["jet"]
    audit = jet["roundoff_audit"]
    r, m, y, k = map(float, audit["ivp_identity"]["input_exact"])
    proj = jet["correlated_projection"]["64"]["projected"]
    off = pred._terminal_roundoff_audit(r, np.asarray([r, m, y]), k,
                                      np.asarray([proj["R_km"], proj["m_km"], proj["y_R"]]),
                                      proj, jet["bitwise_equal"], cache_enabled=False)
    assert off["direct_error_pass"] is True
    for level in ("coarse", "refined"):
        on_ctx = audit["validated_interval"][level]["source_context"]
        off_ctx = off["validated_interval"][level]["source_context"]
        assert on_ctx["cache_enabled"] is True and on_ctx["cache_hits"] > 0
        assert off_ctx["cache_enabled"] is False
        assert off_ctx["cache_hits"] == off_ctx["cache_entries"] == 0
        assert off_ctx["requests"] == off_ctx["evaluations"]


def test_terminal_intersection_recheck_rejects_hulls_padding_and_stale_bounds(terminal_witness) -> None:
    q = terminal_witness
    audit = q["surface_event"]["jet"]["roundoff_audit"]
    import copy
    # A hull is not max/min intersection and must fail the exact recheck.
    hull = copy.deepcopy(audit)
    for i in range(3):
        hull["intersection_box"][i] = [hull["coarse_box"][i][0], hull["refined_box"][i][1]]
    assert pred._roundoff_audit_accepts(hull) is False
    padded = copy.deepcopy(audit)
    padded["intersection_widths"][0] = str(float(padded["intersection_widths"][0]) + 1.0)
    assert pred._roundoff_audit_accepts(padded) is False
    stale = copy.deepcopy(audit)
    stale["coarse_box"][0][1] = str(float(stale["coarse_box"][0][1]) - 1.0)
    assert pred._roundoff_audit_accepts(stale) is False


def test_terminal_low_s_regular_factors_and_vacuum_limits_are_bound() -> None:
    import inspect
    src = inspect.getsource(pred._terminal_roundoff_audit_impl)
    assert "cs2_over_s" in src and "dPdn_over_s" in src and "n_over_s" in src
    assert "ivmp.iv.mpf(3)/2" in src
    assert "q[\"mu\"]*q[\"n_over_s\"]/q[\"cs2_over_s\"]" in src
    assert "I(1)/(3*MNi*MNi)" in src


def test_gamma2_control_is_actually_integrated() -> None:
    control = pred.gamma2_newtonian_control()
    assert control["integrated"] is True
    assert control["k2"] == pytest.approx((15.0 - np.pi**2) / (2.0 * np.pi**2), rel=2.0e-3)
    assert control["surface_xi"] == pytest.approx(np.pi, abs=2.0e-3)


def test_repair30_extended_enthalpy_is_immutable_and_projected_once() -> None:
    for x in (0.0, 1.0e-6, 0.1, 1.0):
        source = pred._source_result(x)
        assert isinstance(source, pred.SourceResult)
        assert isinstance(source.enthalpy_extended, np.longdouble)
        assert isinstance(source.point.enthalpy_extended, np.longdouble)
        assert source.point.enthalpy == float(source.enthalpy_extended)
        assert source.point.enthalpy_extended == source.enthalpy_extended
        enriched = pred._point_result_with_derivatives(x)
        assert enriched.enthalpy_extended == source.enthalpy_extended
        assert enriched.point.enthalpy == source.point.enthalpy


def test_repair30_w_bundle_toggle_is_byte_identical_and_material() -> None:
    import inspect
    source = inspect.getsource(pred._solve_W_certified)
    for token in ("_cache_enabled", "key = float(w).hex()", "w_bundle", "scalar = _scalar_density_stable", "ns_m, _ = _scalar_density_mass_derivative", "residual_scale"):
        assert token in source
    for x in (1.0e-3, 0.1, 1.0, 10.0):
        n = float(x * pred.upstream.PARAMS.n0_nat)
        calls = {"on": 0, "off": 0}
        original = pred._scalar_density_stable
        def wrapped(value: float, mass: float, p: object = pred.upstream.PARAMS) -> float:
            calls["mode"] += 1
            return original(value, mass, p)
        # Count source calls in two otherwise identical solves.
        calls["mode"] = 0
        pred._scalar_density_stable = wrapped  # type: ignore[assignment]
        try:
            won, con = pred._solve_W_certified(n, _cache_enabled=True)
            calls["on"] = calls["mode"]
            calls["mode"] = 0
            woff, coff = pred._solve_W_certified(n, _cache_enabled=False)
            calls["off"] = calls["mode"]
        finally:
            pred._scalar_density_stable = original  # type: ignore[assignment]
        assert won == woff and con == coff
        assert calls["on"] < calls["off"]


def test_repair30_w_bundle_scope_and_forbidden_reuse_mutations() -> None:
    import inspect
    src = inspect.getsource(pred._solve_W_certified)
    # The only cache key is the exact local w bit pattern.  These guards make
    # cross-density/owner/process state, projected residuals, or accepted
    # roots/brackets impossible to smuggle into the coefficient bundle.
    assert "w_bundle: dict" in src and "float(w).hex()" in src
    assert "global_w_bundle" not in src and "_last_w_state" not in src
    assert "_cache_enabled and key in w_bundle" in src
    assert "return w_bundle[key]" in src
    assert "w_lo" not in src.split("def bundle", 1)[1].split("def f", 1)[0]
    assert "residual_scale = max(" in src


def test_repair39_w_cse_reference_identity_log_grid_and_hint_resets() -> None:
    """The bounded coefficient CSE must preserve every W/certificate bit."""
    # This is the predeclared low/high-density log grid used by the repair
    # control.  Each pair starts with a reset hint, so no continuation state
    # can influence the identity comparison.
    xs = np.geomspace(1.0e-12, 10.0, 17)
    for x in xs:
        n = float(float(x) * pred.upstream.PARAMS.n0_nat)
        on_w, on_cert = pred._solve_W_certified(n, hint=None, _cache_enabled=True)
        off_w, off_cert = pred._solve_W_certified(n, hint=None, _cache_enabled=False)
        assert on_w == off_w
        assert on_cert == off_cert
        # Repeating the exact invocation after a reset must remain byte-stable.
        repeat_w, repeat_cert = pred._solve_W_certified(n, hint=None, _cache_enabled=True)
        assert (repeat_w, repeat_cert) == (on_w, on_cert)


def test_repair39_w_cse_source_scope_and_decision_topology() -> None:
    """Only call-local arithmetic CSE is present; root decisions stay fixed."""
    import inspect
    src = inspect.getsource(pred._solve_W_certified)
    bundle_src = src.split("def bundle", 1)[1].split("# Compatibility aliases", 1)[0]
    for token in ("four_pi2", "G_n_n", "three_G_n_n", "gs2", "lam_W0_W0", "t3 = t**3", "t21 = t**21", "math.fsum"):
        assert token in src
    for forbidden in ("global_w_bundle", "_cross_owner", "_shared_w_cache", "_last_w_state", "STATIC_W", "module_w_bundle"):
        assert forbidden not in src
    assert "w_bundle: dict[float, WEvaluationRecord] = {}" in src
    assert "if _cache_enabled and key in w_bundle" in src
    assert "return w_bundle[key]" in src
    assert "w_lo, w_hi" not in bundle_src
    # The safeguarded Newton and strict bracket inequalities remain in the
    # same source function and therefore cannot be bypassed by the CSE.
    assert "if not (flo < 0.0 < fhi)" in src
    assert "if rlo.fW <= 0.0 or rhi.fW <= 0.0" in src
    assert "if bracket_rel <= 2e-13 and residual_rel <= 2e-12" in src


def test_repair39_strict_star_payload_repeats_bitwise() -> None:
    """Strict-star source payloads remain exact across fresh EOS lifecycles."""
    import json
    import gc

    def payload(x: float) -> bytes:
        eos = pred.DirectEOS()
        eos.begin_star()
        try:
            row = pred.integrate_enthalpy(x, rtol=1.0e-8, strict_eos=True, eos=eos)
        finally:
            eos.end_star()
        return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    for x in (0.001, 0.1):
        first = payload(x)
        gc.collect()
        second = payload(x)
        assert first == second


def test_gamma2_density_surface_chart_replay_and_mutations() -> None:
    control = pred.gamma2_newtonian_control()
    rows = control["pressure_rows"]
    assert rows and control["extrapolated"]["surface_replay_pass"] is True
    row = next(r for r in rows if r["surface_event"]["rho_q"] > 0.0)
    ev = row["surface_event"]
    assert ev["arithmetic"] == "classical_RK4_density_nodes_rho_q_half_half_zero"
    assert ev["delta_rho"] == pytest.approx(-ev["rho_q"], rel=0.0, abs=0.0)
    assert len(ev["stage_inputs"]) == 4 and len(ev["stage_derivatives"]) == 4
    assert ev["proof_120d"]["dps"] == 120
    assert pred._gamma2_density_replay(row)
    bad = json.loads(json.dumps(row)); bad["surface_event"]["stage_derivatives"][0][0] *= 1.0001
    assert not pred._gamma2_density_replay(bad)
    bad = json.loads(json.dumps(row)); bad["surface_event"]["delta_rho"] = -0.5 * ev["rho_q"]
    assert not pred._gamma2_density_replay(bad)
    bad = json.loads(json.dumps(row)); bad["surface_event"]["stage_inputs"][2][0] += 1.0e-3
    assert not pred._gamma2_density_replay(bad)


def test_q_chain_component_mutations_are_not_equivalent() -> None:
    eos = pred.DirectEOS(); hc = pred._source_point(1.0).enthalpy
    for q, frac in ((0.11, 0.63), (0.19, 0.71), (0.27, 0.82)):
        h = hc * frac; state = np.asarray([8.0, 0.8, 1.4])
        ordinary = pred._rhs_h(h, state, eos, strict_eos=True)
        transformed = pred._q_chain_rhs(q, h + q*q, state, eos)
        drdh = float(ordinary[0]); expected_y = -2.0*q*float(ordinary[2]/drdh)*drdh
        assert transformed[2] == pytest.approx(expected_y, rel=2.0e-12, abs=2.0e-14)
        wrong = -2.0*q*float(ordinary[2]/drdh)
        assert abs(transformed[2] - wrong) > 1.0e-8


def test_terminal_proof_ownership_is_predeclared_and_exact() -> None:
    expected = {
        (float(0.001).hex(), float(0.05).hex()),
        (float(0.0014330125702369627).hex(), float(0.05).hex()),
        (float(0.002053525026457146).hex(), float(0.05).hex()),
    }
    assert pred.TERMINAL_WITNESS_KEYS == expected
    assert {(x, d) for x, d in expected if pred._terminal_proof_role(float.fromhex(x), float.fromhex(d)) == pred.TERMINAL_PROOF_WITNESS} == expected
    assert pred._terminal_proof_role(0.1, 0.05) == pred.TERMINAL_PRODUCTION_CHART
    assert pred._terminal_proof_role(0.001, 0.025) == pred.TERMINAL_PRODUCTION_CHART
    source = inspect.getsource(pred._convergence_summary)
    assert "_terminal_proof_role(x, d)" in source
    assert "same_bits" not in source


def test_terminal_ownership_rejects_relabelled_input_before_integration() -> None:
    with pytest.raises(ValueError, match="ownership key mismatch"):
        pred.integrate_pressure_rk4(0.001, dr=0.025, proof_role=pred.TERMINAL_PROOF_WITNESS)
    with pytest.raises(ValueError, match="ownership key mismatch"):
        pred.integrate_pressure_rk4(0.1, dr=0.05, proof_role=pred.TERMINAL_PROOF_WITNESS)


def test_non_witness_chart_does_not_schedule_interval_proof(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"audit": 0, "accept": 0}
    original_audit = pred._terminal_roundoff_audit
    original_accept = pred._roundoff_audit_accepts
    def audit(*args: object, **kwargs: object):
        calls["audit"] += 1
        return original_audit(*args, **kwargs)
    def accept(*args: object, **kwargs: object):
        calls["accept"] += 1
        return original_accept(*args, **kwargs)
    monkeypatch.setattr(pred, "_terminal_roundoff_audit", audit)
    monkeypatch.setattr(pred, "_roundoff_audit_accepts", accept)
    q = pred.integrate_pressure_rk4(0.1, dr=0.05, proof_role=pred.TERMINAL_PRODUCTION_CHART)
    jet = q["surface_event"]["jet"]
    assert jet["proof_role"] == pred.TERMINAL_PRODUCTION_CHART
    assert "roundoff_audit" not in jet
    assert "interval_error_check_pass" not in jet
    assert calls == {"audit": 0, "accept": 0}


def test_terminal_chart_schema_has_no_theorem_alias_for_non_witness() -> None:
    # The production-only chart is intentionally a small, theorem-free
    # contract.  Any interval/replay field added to this chart would make a
    # copied witness payload look like a proof.
    forbidden = {"roundoff_audit", "interval_error_check_pass", "direct_error_pass", "replay_digest", "replay_acceptor", "validated_interval", "direct_n64_error", "proof"}
    jet = pred.integrate_pressure_rk4(0.1, dr=0.05, proof_role=pred.TERMINAL_PRODUCTION_CHART)["surface_event"]["jet"]
    chart = jet["terminal_chart"]
    assert chart["N"] == [16, 32, 64]
    assert chart["status"] == "PASS"
    assert set(chart["ladder"]) == {"16", "32", "64"}
    assert chart["dop853_oracle"]["relative_to_N64"] == chart["N64_vs_DOP853_relative"]
    assert forbidden.isdisjoint(chart)


def test_terminal_chart_role_cannot_be_selected_from_same_bits() -> None:
    source = inspect.getsource(pred._terminal_k_jet)
    # ``same_bits`` is computed only after all production endpoints and must
    # never appear in the proof-role branch condition.
    assert "proof_role == TERMINAL_PROOF_WITNESS" in source
    assert "if any(same_bits)" in source  # proof runs for declared witnesses
