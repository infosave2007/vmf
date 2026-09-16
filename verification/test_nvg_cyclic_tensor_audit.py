"""Focused derivation, transfer, and fail-closed controls for the TT audit."""

from __future__ import annotations

import copy
import json

import pytest

import nvg_cyclic_tensor_audit as audit


@pytest.fixture(scope="module")
def full_result():
    return audit.compute_audit()


def test_lorentzian_tt_variation_and_conformal_conversion_are_derived() -> None:
    rows = audit.tensor_action_symbolic_checks()
    assert audit.rows_pass(rows)
    assert rows["lorentzian_time_sign"]["residual"] == "0"
    assert rows["unit_tensor_characteristic_speed"]["residual"] == "0"
    assert rows["conformal_frequency_equation"]["residual"] == "0"
    bad_conformal = audit.tensor_action_symbolic_checks(conformal_extra_A_minus2=1)
    assert not audit.rows_pass(bad_conformal)
    assert bad_conformal["conformal_nonunit_A_extra_control"]["passed"] is False
    duplicated_bad = audit.tensor_action_symbolic_checks(
        conformal_extra_A_minus2=1,
        conformal_target_extra_A_minus2=1,
    )
    # The wrong A^-2 terms cancel in the final u equation, but the independently
    # derived h_sigma/Pi_sigma row still rejects the pair.
    assert duplicated_bad["conformal_frequency_equation"]["passed"] is True
    assert duplicated_bad["conformal_h_second_derivative"]["passed"] is False
    assert not audit.rows_pass(duplicated_bad)


@pytest.mark.parametrize(
    "mutation",
    [
        {"lorentzian_time_sign": -1},
        {"lorentzian_spatial_sign": -1},
        {"laplacian_offset": 2},
        {"curvature_shift": 1},
        {"conformal_extra_A_minus2": 1},
    ],
)
def test_action_sign_and_true_curvature_mutations_fail_closed(mutation) -> None:
    assert not audit.rows_pass(audit.tensor_action_symbolic_checks(**mutation))


def test_s3_tt_eigenvalue_and_frequency_are_not_flat_lookup_values() -> None:
    data = audit.harmonic_data(3)
    assert data["laplacian_eigenvalue"] == -6.0
    assert data["frequency_squared_at_a1"] == 8.0
    assert audit.harmonic_data(20)["frequency_squared_at_a1"] == 399.0
    assert audit.harmonic_data(3, curvature_shift=1)["frequency_squared_at_a1"] == 7.0


def test_static_scale_rotation_and_generator_are_exactly_symplectic() -> None:
    static = audit.static_scale_check(n=7, A=1.75, phase=0.61)
    assert static["passed"]
    assert static["raw_determinant"] == 1.0
    assert static["symplectic_residual"] == 0.0
    assert audit.generator_symplectic_check()["passed"]


@pytest.mark.parametrize("w", [0.0, 1.0 / 3.0, 1.0])
def test_background_period_is_refined_and_returns_to_same_turning_point(w) -> None:
    background = audit.cycle_background(w=w)
    assert 0.0 < background["A_low"] < 1.0
    assert background["period_quadrature_fine"] > 0.0
    assert background["quadrature_refinement_relative"] < 3e-7
    assert background["ode_vs_quadrature_relative"] < 3e-6
    assert background["trajectory_curve_max_abs"] < 3e-7
    assert background["algebraic_curve_constraint_max_abs"] < 3e-7
    assert abs(background["A_end_direct_ode"] - background["A_low"]) < 3e-7
    assert abs(background["q_end_direct_ode"] - 3.141592653589793) < 3e-8


def test_actual_transfer_uses_raw_determinant_and_independent_formulations() -> None:
    row = audit.tensor_mode_transfer(w=0.0, n=3)
    assert row["passed"] and audit.mode_acceptance(row)
    assert row["determinant_fine"] == audit.matrix_determinant(row["matrix_fine"])
    assert row["determinant_error_fine"] < 3e-6
    assert row["matrix_refinement_relative"] < 3e-6
    assert row["q_vs_time_relative"] < 3e-6
    assert row["u_vs_time_relative"] < 3e-6
    assert row["classification"] in {"elliptic", "hyperbolic", "parabolic"}


def test_full_declared_set_is_finite_sampled_and_exposes_instability(full_result) -> None:
    assert full_result["mathematical_checks_passed"]
    assert audit.acceptance_from_result(full_result)
    assert full_result["summary"]["mode_count"] == 54
    assert full_result["summary"]["finite_mode_sample_only"]
    assert full_result["summary"]["classification_counts"]["numerically_unresolved"] == 0
    # The resolved hyperbolic controls remain visible rather than being
    # relabelled as numerical failures.
    assert full_result["summary"]["robust_hyperbolic_modes"]
    assert all(audit.growth_check_acceptance(row) for row in full_result["growth_checks"])


def test_growing_eigenmode_is_evolved_for_multiple_cycles(full_result) -> None:
    checks = full_result["growth_checks"]
    assert checks
    for check in checks:
        assert check["cycles"] >= 3
        assert check["eigenvalue_modulus"] > 1.0
        assert check["per_cycle_growth"][-1] > 1.0
        assert max(check["relative_errors"]) < 3e-4


def test_mode_acceptance_rejects_missing_nan_and_matched_but_large_error(full_result) -> None:
    original = full_result["cycles"][0]
    missing = copy.deepcopy(original)
    del missing["matrix_u"]
    assert not audit.mode_acceptance(missing)

    nan_value = copy.deepcopy(original)
    nan_value["determinant_fine"] = "nan"
    assert not audit.mode_acceptance(nan_value)

    matched_large = copy.deepcopy(original)
    matched_large["matrix_refinement_relative"] = 0.25
    # Keeping the stale/pass flag cannot turn an unacceptably large error into
    # evidence; the validator recomputes and bounds the residual.
    assert not audit.mode_acceptance(matched_large)


def test_result_acceptance_rejects_missing_and_mutated_summary(full_result) -> None:
    missing = copy.deepcopy(full_result)
    del missing["growth_checks"]
    assert not audit.acceptance_from_result(missing)

    mutated = copy.deepcopy(full_result)
    mutated["summary"]["classification_counts"]["elliptic"] += 1
    assert not audit.acceptance_from_result(mutated)


@pytest.mark.parametrize("bad", [True, False, None, "nan", "inf", -1.0])
def test_invalid_controls_are_rejected(bad) -> None:
    with pytest.raises(ValueError):
        audit.tensor_mode_transfer(w=bad, n=3)
    with pytest.raises(ValueError):
        audit.harmonic_data(bad)


def test_single_cli_is_strict_json_and_does_not_embed_result_tables(capsys) -> None:
    assert audit.main(["--single-w", "0", "--single-n", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == audit.SCHEMA
    assert payload["cycles"][0]["inputs_manufactured"] is True
    assert payload["mathematical_checks_passed"]
    source = audit.SOURCE_PATH.read_text(encoding="utf-8")
    assert '"matrix_fine": [[' not in source
