"""Focused tests for the formal source-action S-wave threshold audit."""

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

import nvg_source_complete_formal_swave_threshold_audit as audit  # noqa: E402
from source_complete_solution_audit import PARAMS  # noqa: E402


@pytest.fixture(scope="module")
def result() -> dict:
    return audit.build_result()


def _fraction(record: dict) -> Fraction:
    return Fraction(int(record["exact_numerator"]), int(record["exact_denominator"]))


def test_live_action_surface_and_formal_contract_are_frozen(result: dict):
    exact = audit.exact_parameters_from_live()
    assert exact.W0 == Fraction("859")
    assert exact.lambda_baseline == Fraction("1.05")
    assert exact.M_N == Fraction("939")
    assert exact.g_s == Fraction("939") / Fraction("859")
    assert exact.q_phi == Fraction("782.6") / Fraction("859")
    assert exact.reduced_mass == Fraction("939") / 2

    contract = result["formal_model_contract"]
    assert contract["parameter_fit_performed"] is False
    assert contract["no_new_operator_introduced"] is True
    assert contract["hypothetical_deformed_action_member"] is True
    assert contract["physical_action_replacement_claimed"] is False
    assert contract["empirical_target_or_data_read"] is False
    assert contract["physical_claim_permitted"] is False
    assert "mu=M_N/2" in contract["held_fixed"]
    assert "g_omega" in contract["held_fixed"]


def test_monotonic_static_potential_and_analytic_comparator_ordering(result: dict):
    exact = audit.exact_parameters_from_live()
    q0 = audit.lambda_q0_attraction(exact)
    coordinate_range = audit.lambda_coordinate_range(exact)
    bargmann_lower, bargmann_upper = audit.lambda_bargmann_bounds(exact)
    bargmann_display = audit.lambda_bargmann_display(exact)
    assert q0 < bargmann_lower < bargmann_upper < coordinate_range < exact.lambda_baseline
    assert _fraction(result["analytic_formal_certificates"]["cross_scale_thresholds"]["lambda_q0"]) == q0
    assert mp.mpf(bargmann_lower.numerator) / bargmann_lower.denominator < bargmann_display < (
        mp.mpf(bargmann_upper.numerator) / bargmann_upper.denominator
    )

    # At fixed physical coordinate z=m_omega*r_nat, increasing lambda makes
    # the formal potential pointwise less attractive.
    low_lambda, high_lambda = 0.001, 0.002
    for z_omega in (0.1, 1.0, 10.0):
        assert audit._omega_coordinate_potential(z_omega, low_lambda, exact) < audit._omega_coordinate_potential(
            z_omega, high_lambda, exact
        )

    certificate = result["analytic_formal_certificates"]["bargmann_comparator"]
    assert certificate["strict_no_binding_for_lambda_d_at_least_rational_upper"] is True
    witness = result["analytic_formal_certificates"]["declared_negative_variational_witness"]
    assert mp.mpf(witness["rayleigh_energy_MeV_display"]) < -mp.mpf("1e-6")


def test_first_nodeless_zero_energy_threshold_is_controlled(result: dict):
    surface = result["runtime_control_surface"]
    assert [row["max_step"] for row in surface["shooting_controls"]] == ["0.1", "0.075", "0.05"]
    assert surface["search"]["pivot_relative_floor"] == repr(audit.PIVOT_RELATIVE_FLOOR)
    shooting = result["zero_energy_shooting"]
    assert shooting["box_tolerance_convergence_pass"] is True
    assert shooting["first_threshold_nodeless"] is True
    assert mp.mpf(shooting["relative_spread_across_declared_box_tolerance_controls"]) < mp.mpf(
        shooting["relative_spread_limit"]
    )
    assert len(shooting["controls"]) == 3
    roots = []
    for row in shooting["controls"]:
        assert row["control"]["outer_control_has_at_least_30_scalar_ranges"] is True
        assert 0 < mp.mpf(row["control"]["max_step"]) <= mp.mpf(row["control"]["x_sigma_max"])
        assert row["first_threshold_nodeless"] is True
        lower = mp.mpf(row["lambda_bracket"]["lower_negative_log_derivative"])
        upper = mp.mpf(row["lambda_bracket"]["upper_positive_log_derivative"])
        assert lower < upper
        assert mp.mpf(row["lower_state"]["endpoint_log_derivative"]) < 0
        assert mp.mpf(row["upper_state"]["endpoint_log_derivative"]) > 0
        assert row["lower_state"]["node_count"] == 0
        assert row["upper_state"]["node_count"] == 0
        roots.append(mp.mpf(row["midpoint_lambda_display"]))
    assert max(roots) - min(roots) < mp.mpf("1e-10")

    diagnostics = result["formal_threshold_diagnostics"]
    enclosure = diagnostics["rounded_numerical_enclosure"]
    fine_root = roots[-1]
    assert mp.mpf(enclosure["strict_lower"]) < fine_root < mp.mpf(enclosure["strict_upper"])
    assert enclosure["statement"] == "0.0011028 < lambda_c_formal < 0.0011029"
    assert mp.mpf(diagnostics["lambda_c_over_lambda_q0_display"]) < 1
    assert mp.mpf(diagnostics["lambda_baseline_over_lambda_c_display"]) > 900


def test_origin_series_replays_preserve_nodeless_sign_bracket(result: dict):
    replays = result["zero_energy_shooting"]["origin_series_replays"]
    assert [row["x_start"] for row in replays] == ["1.0e-6", "1.0e-10"]
    for row in replays:
        assert row["sign_bracket_preserved"] is True
        assert row["nodes_lower_and_upper"] == [0, 0]
        assert mp.mpf(row["lower_log_derivative"]) < 0
        assert mp.mpf(row["upper_log_derivative"]) > 0


def test_independent_finite_difference_inertia_signs_are_stable(result: dict):
    fd = result["finite_difference_sign_controls"]
    assert fd["controls_are_operator_not_droplet_boundaries"] is True
    assert mp.mpf(fd["lambda_below_shooting_midpoint"]) < mp.mpf(fd["lambda_above_shooting_midpoint"])
    assert len(fd["rows"]) == 3
    dimensions = []
    for row in fd["rows"]:
        below = row["below_shooting_midpoint"]
        above = row["above_shooting_midpoint"]
        assert row["sign_control_pass"] is True
        assert below["negative_eigenvalue_count"] == 1
        assert above["negative_eigenvalue_count"] == 0
        assert mp.mpf(below["relative_minimum_abs_ldl_pivot"]) > audit.PIVOT_RELATIVE_FLOOR
        assert mp.mpf(above["relative_minimum_abs_ldl_pivot"]) > audit.PIVOT_RELATIVE_FLOOR
        dimensions.append(below["matrix_dimension"])
    assert dimensions == sorted(dimensions)


def test_diagnostics_recompute_the_formal_scalar_and_tail_scales(result: dict):
    exact = audit.exact_parameters_from_live()
    fine = result["zero_energy_shooting"]["controls"][-1]
    midpoint = float(fine["midpoint_lambda_display"])
    m_sigma = audit.scalar_mass_mev(midpoint, exact)
    diagnostics = result["formal_threshold_diagnostics"]
    assert mp.almosteq(mp.mpf(diagnostics["m_sigma_at_lambda_c_MeV_display"]), mp.mpf(str(m_sigma)), rel_eps=mp.mpf("1e-12"))
    assert 4.0 < mp.mpf(diagnostics["scalar_range_hbarc_over_msigma_fm_display"]) < 6.0
    assert 1.0 < mp.mpf(diagnostics["tree_static_coordinate_tail_crossover_fm_display_not_prediction"]) < 1.5


def test_cache_returns_isolated_payload_and_mutations_fail_closed(result: dict, monkeypatch: pytest.MonkeyPatch):
    cached = audit.build_result()
    assert cached == result
    cached["status"] = "MUTATED_LOCAL_COPY"
    assert audit.build_result()["status"] == audit.STATUS
    assert audit.validate_result(result) is None
    parsed = json.loads(json.dumps(result, allow_nan=False))
    assert audit.validate_result(parsed) is None

    # The cache key includes all declared numerical controls.  Changing one
    # must force a re-derivation rather than accepting the previous payload.
    monkeypatch.setattr(audit, "WITNESS_LAMBDA_DIVISOR", 9)
    with pytest.raises(audit.FormalSWaveAuditError):
        audit.validate_result(result)
    monkeypatch.undo()
    assert audit.validate_result(result) is None

    # Controls that may not change a particular live root still have to be
    # recorded: a re-run under altered criteria is not the same certificate.
    monkeypatch.setattr(audit, "PIVOT_RELATIVE_FLOOR", audit.PIVOT_RELATIVE_FLOOR / 10.0)
    with pytest.raises(audit.FormalSWaveAuditError):
        audit.validate_result(result)
    monkeypatch.undo()
    assert audit.validate_result(result) is None

    mutations = (
        lambda payload: payload.__setitem__("status", "PHYSICAL_DEUTERON_FOUND"),
        lambda payload: payload["formal_model_contract"].__setitem__("parameter_fit_performed", True),
        lambda payload: payload["zero_energy_shooting"].__setitem__("first_threshold_nodeless", False),
        lambda payload: payload["finite_difference_sign_controls"]["rows"][0]["below_shooting_midpoint"].__setitem__(
            "negative_eigenvalue_count", 0
        ),
        lambda payload: payload["provenance"].__setitem__("producer_sha256", "0" * 64),
    )
    for mutate in mutations:
        changed = copy.deepcopy(result)
        mutate(changed)
        with pytest.raises(audit.FormalSWaveAuditError):
            audit.validate_result(changed)


def test_nonlive_or_contract_broken_parameter_surfaces_fail_closed():
    with pytest.raises(audit.FormalSWaveAuditError):
        audit.build_result(replace(PARAMS, lam=0.1))
    with pytest.raises(audit.FormalSWaveAuditError):
        audit.exact_parameters_from_live(replace(PARAMS, g_omega=0.0))
    with pytest.raises(audit.FormalSWaveAuditError):
        audit.exact_parameters_from_live(replace(PARAMS, M_N=0.0))
    with pytest.raises(audit.FormalSWaveAuditError):
        audit._finite_difference_inertia(0.0, audit.FINITE_DIFFERENCE_CONTROLS[0], audit.exact_parameters_from_live())


def test_cli_regeneration_and_validation_are_canonical(tmp_path: Path):
    output = tmp_path / "formal-swave.json"
    script = HERE / "nvg_source_complete_formal_swave_threshold_audit.py"
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


def test_source_is_self_contained_and_does_not_import_fitted_or_network_producers():
    source = (HERE / "nvg_source_complete_formal_swave_threshold_audit.py").read_text(encoding="utf-8")
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
