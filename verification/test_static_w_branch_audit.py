"""Semantic tests for the independent W-branch audit."""

from __future__ import annotations

import math

from static_w_branch_audit import (
    Inputs,
    action_branch,
    build_audit,
    calibration,
    energy_density,
    fixed_charge_branch,
    fixed_frequency_branch,
    omega_gauged,
    potential,
    render_report,
    write_outputs,
)


def test_calibration_recomputes_declared_parameters() -> None:
    result = build_audit()
    c = result["calibration"]
    assert math.isclose(c["I1"], 0.5277, rel_tol=0.0, abs_tol=2e-4)
    assert math.isclose(c["C_scaling"], 1.1367, rel_tol=0.0, abs_tol=2e-4)
    assert math.isclose(c["q"], 47.0405, rel_tol=0.0, abs_tol=2e-3)
    assert math.isclose(c["lambda_sigma"], 1.0575, rel_tol=0.0, abs_tol=2e-4)
    assert math.isclose(c["gamma_per_n0_MeV"], 955.547, rel_tol=0.0, abs_tol=1e-3)


def test_fixed_frequency_vacuum_is_zero_energy_anchor() -> None:
    inp = Inputs()
    cal = calibration(inp)
    branch = fixed_frequency_branch(cal["mu_theta"], inp, cal)
    assert math.isclose(branch["W_MeV"], inp.W_vac_MeV, rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(branch["rho_MeV4"], 0.0, rel_tol=0.0, abs_tol=1e-3)
    # A non-zero timelike phase gradient breaks vacuum Lorentz invariance:
    # canonical P=L retains the positive phase-kinetic contribution.
    assert math.isclose(
        branch["pressure_MeV4"], inp.W_vac_MeV**2 * cal["mu_theta"]**2,
        rel_tol=0.0, abs_tol=1e-3,
    )
    assert branch["stable"]


def test_action_stationarity_has_plus_frequency_sign_and_does_not_melt() -> None:
    inp = Inputs()
    cal = calibration(inp)
    omega = cal["mu_theta"]
    branch = action_branch(omega, inp, cal)
    expected = math.sqrt((cal["mu_squared"] + omega**2) / inp.lambda_model)
    assert math.isclose(branch["W_MeV"], expected, rel_tol=0.0, abs_tol=1e-10)
    assert branch["W_MeV"] > inp.W_vac_MeV
    assert branch["effective_potential_hessian_MeV2"] > 0.0
    assert abs(branch["eom_residual_MeV3"]) < 1e-5
    assert all(
        row["action"]["W_MeV"] > 0.0
        for row in build_audit()["branches_at_declared_grid"]
    )


def test_hessians_match_centered_second_derivatives() -> None:
    inp = Inputs()
    cal = calibration(inp)
    omega = omega_gauged(1.0, inp, cal)
    action = action_branch(omega, inp, cal)
    energy = fixed_frequency_branch(omega, inp, cal)

    def action_u(w: float) -> float:
        return potential(w, inp, cal) - 0.5 * omega**2 * w**2

    def energy_u(w: float) -> float:
        return energy_density(w, omega, inp, cal)

    h = 1.0e-2
    action_fd = (
        action_u(action["W_MeV"] + h)
        - 2.0 * action_u(action["W_MeV"])
        + action_u(action["W_MeV"] - h)
    ) / h**2
    energy_fd = (
        energy_u(energy["W_MeV"] + h)
        - 2.0 * energy_u(energy["W_MeV"])
        + energy_u(energy["W_MeV"] - h)
    ) / h**2
    assert math.isclose(
        action_fd, action["effective_potential_hessian_MeV2"], rel_tol=1e-6
    )
    assert math.isclose(
        energy_fd, energy["energy_hessian_MeV2"], rel_tol=5e-6
    )


def test_fixed_frequency_melting_root_is_derived_from_gauged_current() -> None:
    result = build_audit()
    cal = result["calibration"]
    t = result["thresholds"]
    expected = (cal["mu_theta"] + cal["mu"]) / cal["gamma_per_n0_MeV"]
    assert math.isclose(t["Omega_equals_minus_mu_positive_x_n0"], expected, abs_tol=1e-12)
    assert not math.isclose(expected, 2.5, rel_tol=1e-6)
    omega = -cal["mu"]
    branch = fixed_frequency_branch(omega, Inputs(), cal)
    assert abs(omega + cal["mu"]) < 1e-9
    assert branch["branch"] == "critical_flat_quadratic"
    assert branch["W_MeV"] == 0.0


def test_canonical_pressure_sign_and_melted_sec() -> None:
    inp = Inputs()
    cal = calibration(inp)
    melted = fixed_frequency_branch(cal["mu"] * 1.1, inp, cal)
    assert melted["W_MeV"] == 0.0
    assert math.isclose(melted["pressure_MeV4"], -cal["V0_MeV4"], abs_tol=1e-6)
    assert math.isclose(melted["SEC_MeV4"], -2.0 * cal["V0_MeV4"], abs_tol=1e-6)
    vacuum_pressure = fixed_frequency_branch(cal["mu_theta"], inp, cal)["pressure_MeV4"]
    assert vacuum_pressure > 0.0


def test_fixed_charge_routhian_is_stable_and_excludes_zero_for_nonzero_charge() -> None:
    result = build_audit()
    rows = result["fixed_charge_branches"]
    nonzero = rows[1]
    assert nonzero["canonical_phase_charge_MeV3"] > 0.0
    assert nonzero["W_MeV"] > rows[0]["W_MeV"]
    assert math.isclose(nonzero["W_MeV"], 1211.7993443, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(nonzero["Omega_MeV"], 393.2463903, rel_tol=0.0, abs_tol=1e-6)
    assert nonzero["stable"]
    assert not nonzero["zero_branch_allowed"]
    assert nonzero["stationarity_equation"] == "lambda*y^3-mu^2*y^2-N^2=0, y=W^2"
    assert abs(nonzero["stationarity_residual_MeV3"]) < 1e-6
    assert nonzero["stationarity_residual_relative"] < 1e-12


def test_legacy_script_comparison_exposes_vacuum_and_pressure_mismatches() -> None:
    result = build_audit()
    legacy = result["contract_comparison"]["maintained_script"]
    assert legacy["legacy_vacuum_rho_MeV_fm3"] > 1.0e4
    assert legacy["legacy_vacuum_pressure_MeV_fm3"] < 0.0
    assert legacy["vacuum_energy_expected_MeV_fm3"] == 0.0
    assert legacy["legacy_first_melted_grid_x_n0"] == 2.5
    assert result["status"] == "FAIL_CLOSED_INCOMPATIBLE_CONTRACT"
    assert result["incompatible_contracts"]


def test_result_and_report_are_derived_and_structurally_complete() -> None:
    result = build_audit()
    report = render_report(result)
    assert result["schema"] == "static_w_branch_audit.v1"
    assert len(result["branches_at_declared_grid"]) == 11
    assert "FAIL_CLOSED_INCOMPATIBLE_CONTRACT" in report
    assert "article/NVG_VACUUM_W_FIELD_DERIVATION_RU.md" in report
    assert "2.05161" in report
    assert "static_w_branch_audit.py" in report


def test_repair_writer_generates_json_and_two_reports(tmp_path) -> None:
    result_path = tmp_path / "results.json"
    report_path = tmp_path / "verification-report.md"
    control_path = tmp_path / "control-report.md"
    result = write_outputs(result_path, report_path, control_path)
    assert result_path.exists() and report_path.exists() and control_path.exists()
    assert report_path.read_text(encoding="utf-8") == control_path.read_text(encoding="utf-8")
    assert "w-fixed-charge-repair-1" in result["repair"]["attempt"]
