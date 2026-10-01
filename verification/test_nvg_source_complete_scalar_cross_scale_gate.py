"""Focused tests for the source-action scalar cross-scale entry gate."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_source_complete_scalar_cross_scale_gate as audit  # noqa: E402
from source_complete_solution_audit import PARAMS  # noqa: E402


def _fraction(record: dict) -> Fraction:
    return Fraction(int(record["exact_numerator"]), int(record["exact_denominator"]))


@pytest.fixture(scope="module")
def result() -> dict:
    return audit.build_result()


def test_live_action_relations_are_exact_and_no_other_parameter_is_deformed():
    exact = audit.exact_parameters_from_live()
    assert exact.W0 == Fraction("859")
    assert exact.lambda_baseline == Fraction("1.05")
    assert exact.M_N == Fraction("939")
    assert exact.g_s == Fraction("939") / Fraction("859")
    assert exact.q_phi == Fraction("782.6") / Fraction("859")

    payload = audit.build_result()
    contract = payload["one_parameter_deformation_contract"]
    assert contract["symbol"] == "lambda_d"
    assert contract["domain"] == "lambda_d>0"
    assert contract["parameter_fit_performed"] is False
    assert contract["no_new_operator_introduced"] is True
    assert contract["hypothetical_deformed_action_member"] is True
    assert contract["physical_action_replacement_claimed"] is False
    assert "g_omega" in contract["held_fixed"]


def test_q4_sharp_threshold_is_bracketed_with_rational_pi_bounds():
    exact = audit.exact_parameters_from_live()
    lower, upper = audit.lambda_q4_bounds(exact)
    display = audit.lambda_q4_display(exact)
    assert lower < upper
    assert mp.mpf(lower.numerator) / lower.denominator < display < mp.mpf(upper.numerator) / upper.denominator
    assert exact.lambda_baseline > upper
    assert str(audit.Q4_C_CRITICAL) == "1/4"


def test_q0_zero_kernel_threshold_is_exact_and_signs_are_correct():
    exact = audit.exact_parameters_from_live()
    threshold = audit.lambda_q0_attraction(exact)
    vector = exact.g_omega**2 / exact.m_omega**2

    def kernel(lam: Fraction) -> Fraction:
        return vector - exact.M_N**2 / (2 * lam * exact.W0**4)

    assert kernel(threshold / 2) < 0
    assert kernel(threshold) == 0
    assert kernel(2 * threshold) > 0
    assert kernel(exact.lambda_baseline) > 0


def test_coordinate_range_threshold_and_single_tail_crossover_are_exact():
    exact = audit.exact_parameters_from_live()
    q0 = audit.lambda_q0_attraction(exact)
    q4_lower, q4_upper = audit.lambda_q4_bounds(exact)
    coordinate_range = audit.lambda_coordinate_tail(exact)
    assert coordinate_range == exact.m_omega**2 / (2 * exact.W0**2)
    assert q0 < q4_lower < q4_upper < coordinate_range < exact.lambda_baseline
    assert exact.g_omega**2 > exact.g_s**2

    q4_crossing = audit.tail_crossover_radius_fm(audit.lambda_q4_display(exact), exact)
    q0_crossing = audit.tail_crossover_radius_fm(q0, exact)
    assert q4_crossing > 0
    assert q0_crossing > 0
    with mp.workdps(80):
        gs2 = mp.mpf(exact.g_s.numerator) ** 2 / mp.mpf(exact.g_s.denominator) ** 2
        gw2 = mp.mpf(exact.g_omega.numerator) ** 2 / mp.mpf(exact.g_omega.denominator) ** 2
        hc = mp.mpf(exact.hbar_c.numerator) / exact.hbar_c.denominator
        momega = mp.mpf(exact.m_omega.numerator) / exact.m_omega.denominator
        lam = mp.mpf(q0.numerator) / q0.denominator
        msigma = mp.sqrt(2 * lam * (mp.mpf(exact.W0.numerator) / exact.W0.denominator) ** 2)

        def coordinate_numerator(r_fm: mp.mpf) -> mp.mpf:
            return gw2 * mp.exp(-momega * r_fm / hc) - gs2 * mp.exp(-msigma * r_fm / hc)

        assert coordinate_numerator(q0_crossing / 2) > 0
        assert abs(coordinate_numerator(q0_crossing)) < mp.mpf("1e-60")
        assert coordinate_numerator(2 * q0_crossing) < 0

    with pytest.raises(audit.CrossScaleGateError):
        audit.tail_crossover_radius_fm(coordinate_range, exact)


def test_cross_scale_window_is_strict_subset_of_q4_evasion_interval(result: dict):
    exact = audit.exact_parameters_from_live()
    q0 = audit.lambda_q0_attraction(exact)
    q4_lower, _ = audit.lambda_q4_bounds(exact)
    assert q0 < q4_lower

    gate = result["cross_scale_entry_gate"]
    assert gate["logical_intersection"] == "0 < lambda_d < lambda_q0"
    assert gate["exact_ordering_certificate"]["lambda_q0_strictly_less_than_lambda_Q4"] is True
    assert gate["exact_ordering_certificate"]["lambda_Q4_strictly_less_than_lambda_range"] is True
    assert gate["exact_ordering_certificate"]["lambda_range_strictly_less_than_baseline"] is True
    assert gate["exact_ordering_certificate"]["certified_ordering"] == (
        "lambda_q0 < lambda_Q4 < lambda_range < lambda_baseline"
    )
    assert gate["baseline_is_outside_intersection"] is True
    assert mp.mpf(gate["display_ratio_lambda_Q4_over_lambda_q0"]) > 1


def test_baseline_reports_q4_no_go_and_q0_repulsion_without_claiming_binding(result: dict):
    assert result["status"] == audit.STATUS
    assert result["evidence_weight"] == 0.0
    assert result["q4_tf_gate"]["baseline"]["status"] == "STRICT_Q4_NO_SELF_BINDING_CERTIFICATE_ACTIVE"
    assert result["static_q0_yukawa_gate"]["baseline"]["status"] == "STATIC_Q0_REPULSIVE"
    assert result["static_coordinate_yukawa_gate"]["baseline"]["status"] == "STATIC_COORDINATE_ALL_R_REPULSIVE"
    assert _fraction(result["static_q0_yukawa_gate"]["baseline"]["static_net_kernel_baseline_MeV_minus2"]) > 0
    assert any("existence of a self-bound" in item for item in result["conclusion"]["not_claimed"])
    assert any("No empirical nuclear/NN data" in item for item in result["caveats"])


def test_semantic_or_static_result_mutations_fail_closed(result: dict):
    assert audit.validate_result(result) is None
    parsed = json.loads(json.dumps(result, allow_nan=False))
    assert audit.validate_result(parsed) is None
    mutations = (
        lambda payload: payload.__setitem__("status", "BINDING_FOUND"),
        lambda payload: payload["one_parameter_deformation_contract"].__setitem__("parameter_fit_performed", True),
        lambda payload: payload["q4_tf_gate"].__setitem__("condition_merely_to_evade_strict_certificate", "lambda_d=1.05"),
        lambda payload: payload["static_q0_yukawa_gate"]["baseline"].__setitem__("status", "ATTRACTIVE"),
        lambda payload: payload["static_coordinate_yukawa_gate"].__setitem__(
            "long_range_attractive_tail_condition", "lambda_d=1.05"
        ),
        lambda payload: payload["provenance"].__setitem__("q4_audit_sha256", "0" * 64),
    )
    for mutate in mutations:
        changed = copy.deepcopy(result)
        mutate(changed)
        with pytest.raises(audit.CrossScaleGateError):
            audit.validate_result(changed)


def test_baseline_below_or_nonpositive_surface_fails_closed():
    # A different lambda would be a different source action, not an alternate
    # answer.  The live-baseline certificate refuses to silently relabel it.
    with pytest.raises(audit.CrossScaleGateError):
        audit.build_result(replace(PARAMS, lam=0.09))
    with pytest.raises(audit.CrossScaleGateError):
        audit.exact_parameters_from_live(replace(PARAMS, g_omega=0.0))
    with pytest.raises(audit.CrossScaleGateError):
        audit.exact_parameters_from_live(replace(PARAMS, degeneracy=2))


def test_ambient_precision_cannot_change_canonical_result():
    before = mp.mp.dps
    try:
        mp.mp.dps = 10
        low = audit._canonical_json(audit.build_result())
        mp.mp.dps = 120
        high = audit._canonical_json(audit.build_result())
    finally:
        mp.mp.dps = before
    assert low == high


def test_cli_regeneration_and_validation_are_canonical(tmp_path: Path):
    output = tmp_path / "cross-scale.json"
    script = HERE / "nvg_source_complete_scalar_cross_scale_gate.py"
    generated = subprocess.run(
        [sys.executable, "-B", str(script), "--output", str(output)],
        cwd=HERE.parent,
        text=True,
        capture_output=True,
        check=True,
    )
    stdout = json.loads(generated.stdout)
    disk = json.loads(output.read_text(encoding="utf-8"))
    assert stdout == disk
    before = output.read_bytes()
    checked = subprocess.run(
        [sys.executable, "-B", str(script), "--validate", str(output)],
        cwd=HERE.parent,
        text=True,
        capture_output=True,
        check=True,
    )
    assert checked.stdout.strip() == "VALIDATION_PASS"
    assert output.read_bytes() == before


def test_source_does_not_import_fitted_or_legacy_nuclear_producers():
    source = (HERE / "nvg_source_complete_scalar_cross_scale_gate.py").read_text(encoding="utf-8")
    assert "from source_complete_solution_audit import PARAMS, Params" in source
    for forbidden in (
        "source_complete_scaling_saturation_audit",
        "inverse_potential_jet",
        "nvg_finite_droplet_audit",
        "nvg_nuclear_closure_audit",
        "requests",
        "urllib",
    ):
        assert forbidden not in source


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
