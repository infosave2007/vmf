"""Focused regression and contamination guards for the local isospin jet."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import mpmath as mp
import pytest

import nvg_isospin_jet_audit as audit


def _m(value):
    return mp.mpf(value)


@pytest.fixture(scope="module")
def result():
    return audit.build_result()


def test_result_is_live_and_design_matrix_visible(result):
    assert result["schema"] == audit.SCHEMA
    assert result["evidence_weight"] == 0.0
    assert result["precision_check"]["pass"] is True
    predictions = result["predictions"]
    assert [row["label"] for row in predictions["controls"]] == [
        "baseline_no_rho_original",
        "two_target_quartic_no_rho_control",
    ]
    assert len(predictions["inverse_design_grid"]) == 5
    assert len(predictions["conditional_J_controls"]) == 15
    assert len(predictions["design_sensitivity"]) == 2
    assert [row["chosen_y"] for row in predictions["inverse_design_grid"]] == [
        "0.6", "0.75", "0.85", "0.9", "0.93"
    ]


def test_baseline_and_global_failure_controls_remain_visible(result):
    controls = result["predictions"]["controls"]
    baseline, quartic = controls
    assert baseline["global_candidate_status"].startswith("BASELINE_")
    assert quartic["design_role"] == "TWO_TARGET_QUARTIC_NO_RHO_CONTROL"
    rows = result["predictions"]["inverse_design_grid"]
    rejected = [row for row in rows if row["global_candidate_status"].startswith("REJECTED_")]
    accepted_local = [
        row
        for row in rows
        if row["global_candidate_status"].startswith("LOCAL_RESPONSE_RETAINED")
    ]
    assert {row["chosen_y"] for row in rejected} == {"0.6", "0.75", "0.85"}
    assert {row["chosen_y"] for row in accepted_local} == {"0.9", "0.93"}
    # y=.60 has negative S4 but positive S2 and a stable scalar curvature;
    # the statuses must not collapse this into an isospin-instability claim.
    low_y = next(row for row in rows if row["chosen_y"] == "0.6")
    assert low_y["local_jet_status"] == "COMPUTED_LOCAL_STABLE"
    assert low_y["isospin_local_status"] == "S2_POSITIVE_LOCAL_ISOVECTOR_CURVATURE"
    assert "NOT_BY_ITSELF_ISOSPIN_INSTABILITY" in low_y["S4_sign_status"]


def test_direct_quadrature_and_restationarization_checks(result):
    rows = result["predictions"]["inverse_design_grid"]
    for y in ("0.75", "0.9"):
        row = next(item for item in rows if item["chosen_y"] == y)
        direct = row["direct_restationarization_check"]
        assert _m(direct["parity_energy_max"]) < _m("1e-60")
        assert abs(_m(direct["S2_identity_residual"])) < _m("1e-8")
        assert abs(_m(direct["S4_hessian_residual"])) < _m("1e-8")
        assert abs(_m(direct["L_identity_residual"])) < _m("1e-4")
        assert abs(_m(direct["y_delta2_residual"])) < _m("1e-8")
        for species in row["direct_species_quadrature_check"]["species"]:
            assert _m(species["energy_relative_error"]) < _m("1e-50")
            assert _m(species["pressure_relative_error"]) < _m("1e-50")
            assert _m(species["scalar_density_relative_error"]) < _m("1e-50")


def test_local_identities_and_rho_shift_are_recomputed():
    with mp.workdps(80):
        model, state, _ = audit.inverse_potential_jet("0.90")
        T = state["k"] ** 2 / (6 * state["ef"])
        c_rho = audit.constant_rho_coupling(model, state, "32")
        jet = audit.local_jet(state, c_rho)
        assert abs(jet["J"] - _m(32)) < _m("1e-60")
        assert abs(jet["Kfr_minus_K"] - 9 * state["n"] * state["B_y"] ** 2 / state["C_y"]) < _m("1e-60")
        assert abs(jet["S4_hessian"] - jet["S4_eliminated"]) < _m("1e-50")
        assert abs(jet["L"] - (jet["L_kin"] + 3 * c_rho * state["n"] / 8)) < _m("1e-60")
        assert abs(jet["J"] - (T + c_rho * state["n"] / 8)) < _m("1e-60")
        thermo = audit.rho_thermodynamics(c_rho, state["n"], _m("0.2"))
        assert thermo["pressure"] == thermo["energy_density"]
        assert thermo["mu_n"] == -thermo["mu_p"]
        assert thermo["mu_fixed_delta"] == c_rho * state["n"] * _m("0.2") ** 2 / 4


def test_delta_parity_and_no_rho_control():
    with mp.workdps(80):
        model, state, _ = audit.inverse_potential_jet("0.90")
        no_rho = audit.IsospinJet(model, 0)
        plus = no_rho.equilibrium(state["n"], _m("0.15"), guess=(state["y"], state["y"]))
        minus = no_rho.equilibrium(state["n"], _m("-0.15"), guess=(state["y"], state["y"]))
        assert abs(plus["energy_total"] - minus["energy_total"]) < _m("1e-60")
        assert abs(plus["y"] - minus["y"]) < _m("1e-60")
        zero = no_rho.equilibrium(state["n"], 0, guess=(state["y"], state["y"]))
        jet = audit.local_jet(state, 0)
        assert abs(zero["energy_total"] - state["energy_total"]) < _m("1e-60")
        assert abs(jet["J"] - state["k"] ** 2 / (6 * state["ef"])) < _m("1e-60")


def test_j_shift_changes_L_by_three_delta_j_and_not_S4():
    with mp.workdps(80):
        model, state, _ = audit.inverse_potential_jet("0.90")
        c32 = audit.constant_rho_coupling(model, state, "32")
        c31 = audit.constant_rho_coupling(model, state, "31")
        j32, j31 = audit.local_jet(state, c32), audit.local_jet(state, c31)
        assert abs((j32["L"] - j31["L"]) - 3) < _m("1e-60")
        assert abs(j32["S4_hessian"] - j31["S4_hessian"]) < _m("1e-60")


def test_exact_and_near_degeneracy_use_direct_hessian_without_division():
    with mp.workdps(80):
        common = dict(n="1", y="0.8", k="1", ef="1.3", c_rho="0", C="0.9", D="2")
        exact = audit.local_jet_from_derivatives(B="0", **common)
        assert exact["Kfr_minus_K"] == 0
        assert exact["S4_eliminated"] is None
        assert exact["degeneracy_status"] == "NEAR_DEGENERATE_DIRECT_HESSIAN_ONLY"
        assert mp.isfinite(exact["S4_hessian"])
        near = audit.local_jet_from_derivatives(B="1e-40", **common)
        assert near["S4_eliminated"] is None
        assert mp.isfinite(near["S4_hessian"])
        # A finite but resolved mixing branch still exposes both formulas.
        resolved = audit.local_jet_from_derivatives(B="0.2", **common)
        assert resolved["S4_eliminated"] is not None
        assert abs(resolved["S4_eliminated"] - resolved["S4_hessian"]) < _m("1e-60")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n="1", y="0.8", k="1", ef="1.3", c_rho="-1", C="1", B="0", D="1"),
        dict(n="1", y="0.8", k="1", ef="1.3", c_rho="nan", C="1", B="0", D="1"),
        dict(n="1", y="0.8", k="1", ef="1.3", c_rho="0", C="0", B="0", D="1"),
        dict(n="1", y="0.8", k="1", ef="1.3", c_rho="0", C="-1", B="0", D="1"),
    ],
)
def test_invalid_domains_fail_closed(kwargs):
    with pytest.raises(audit.IsospinJetError):
        audit.local_jet_from_derivatives(**kwargs)
    with pytest.raises(audit.IsospinJetError):
        audit.rho_thermodynamics(-1, 1, 0)


def test_hidden_density_dependent_rho_contamination_is_not_silent():
    with mp.workdps(80):
        model, state, _ = audit.inverse_potential_jet("0.90")
        c_rho = audit.constant_rho_coupling(model, state, "32")
        jet = audit.IsospinJet(model, c_rho)
        # The declared extension is constant.  A caller that tries to make
        # C_rho depend on density must produce a different explicit derivative;
        # the module's constant-rho result remains unchanged and exposes the
        # fixed normalization in the returned rho contribution.
        constant = jet.state(state["n"], _m("0.2"), state["y"])
        contaminated = audit.IsospinJet(model, c_rho * _m("1.1")).state(
            state["n"], _m("0.2"), state["y"]
        )
        assert constant["rho_energy"] != contaminated["rho_energy"]
        assert jet.C_rho == c_rho
        assert "C_rho*n^2*delta^2/8" == audit.local_jet(state, c_rho)["rho_action_normalization"]


def test_rho_term_has_no_hidden_scalar_field_dependence():
    with mp.workdps(80):
        model, state, _ = audit.inverse_potential_jet("0.90")
        base = audit.IsospinJet(model, 0)
        extended = audit.IsospinJet(model, audit.constant_rho_coupling(model, state, "32"))
        delta = _m("0.2")
        y_a, y_b = state["y"], state["y"] * _m("1.01")
        base_a = base.state(state["n"], delta, y_a)
        base_b = base.state(state["n"], delta, y_b)
        ext_a = extended.state(state["n"], delta, y_a)
        ext_b = extended.state(state["n"], delta, y_b)
        rho_a = ext_a["energy_total"] - base_a["energy_total"]
        rho_b = ext_b["energy_total"] - base_b["energy_total"]
        assert abs(rho_a - rho_b) < _m("1e-60")
        assert abs(ext_a["residual"] - base_a["residual"]) < _m("1e-60")
        assert abs(ext_b["residual"] - base_b["residual"]) < _m("1e-60")


def test_benchmark_mutation_cannot_change_prediction_rows(tmp_path):
    payload = audit.load_declared_inputs()
    original = audit.build_result()
    mutated = copy.deepcopy(payload)
    mutated["descriptive_benchmarks"]["roca_maza_eq14"]["slope"] = "123456"
    mutated["descriptive_benchmarks"]["asy_eos"]["L_MeV"] = "-999"
    candidate = tmp_path / "mutated.json"
    candidate.write_text(json.dumps(mutated), encoding="utf-8")
    changed = audit.build_result(candidate)
    assert changed["predictions"] == original["predictions"]
    assert changed["descriptive_comparisons"] != original["descriptive_comparisons"]


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("residual_y", ["0.90"], "residual_y"),
        ("J_reference_controls_MeV", ["31"], "J_reference_controls_MeV"),
        ("sensitivity_K_MeV", ["240"], "sensitivity_K_MeV"),
        ("direct_check_y", ["0.99"], "direct_check_y"),
        ("residual_y", ["0.75", "0.750", "0.85", "0.90", "0.93"], "residual_y"),
    ],
)
def test_sealed_protocol_rejects_missing_foreign_and_duplicate_inputs(
    tmp_path, field, value, needle
):
    payload = audit.load_declared_inputs()
    payload["calculation_inputs"][field] = value
    candidate = tmp_path / f"invalid-{field}.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(audit.IsospinJetError, match=needle):
        audit.build_result(candidate)


def test_cli_rejects_actual_a_missing_rows_and_unmatched_direct_target(tmp_path):
    payload = audit.load_declared_inputs()
    payload["calculation_inputs"]["residual_y"] = ["0.90"]
    reduced = tmp_path / "reduced.json"
    reduced.write_text(json.dumps(payload), encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(audit.SOURCE_PATH), "--input", str(reduced)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "residual_y" in completed.stderr

    payload = audit.load_declared_inputs()
    payload["calculation_inputs"]["direct_check_y"] = ["0.99"]
    unmatched = tmp_path / "unmatched.json"
    unmatched.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(audit.SOURCE_PATH), "--input", str(unmatched)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "direct_check_y" in completed.stderr


def test_corrupted_direct_restationarization_residual_fails_closed(monkeypatch):
    original = audit.direct_restationarization_check

    def corrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        result["L_identity_residual"] = _m("1")
        return result

    monkeypatch.setattr(audit, "direct_restationarization_check", corrupted)
    with pytest.raises(audit.IsospinJetError, match="L_identity_residual"):
        audit.build_result()


def test_corrupted_species_quadrature_residual_fails_closed(monkeypatch):
    original = audit.direct_species_quadrature_check

    def corrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        result["species"][0]["energy_relative_error"] = _m("1")
        return result

    monkeypatch.setattr(audit, "direct_species_quadrature_check", corrupted)
    with pytest.raises(audit.IsospinJetError, match="energy_relative_error"):
        audit.build_result()


def test_fine_precision_missing_check_does_not_disappear(monkeypatch):
    original = audit.calculate

    def corrupted(dps=80, input_path=audit.INPUT_PATH):
        result = original(dps, input_path)
        if int(dps) == 110:
            row = next(
                item
                for item in result["predictions"]["inverse_design_grid"]
                if item["chosen_y"] == "0.75"
            )
            row.pop("direct_restationarization_check", None)
        return result

    monkeypatch.setattr(audit, "calculate", corrupted)
    with pytest.raises(audit.IsospinJetError, match="110.*direct_restationarization_check"):
        audit.build_result()


def test_duplicate_output_control_row_fails_closed(monkeypatch):
    original = audit.calculate

    def corrupted(dps=80, input_path=audit.INPUT_PATH):
        result = original(dps, input_path)
        if int(dps) == 80:
            result["predictions"]["controls"].append(
                copy.deepcopy(result["predictions"]["controls"][0])
            )
        return result

    monkeypatch.setattr(audit, "calculate", corrupted)
    with pytest.raises(audit.IsospinJetError, match="duplicate output"):
        audit.build_result()


def test_cli_is_strict_json_and_no_write(tmp_path):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(audit.SOURCE_PATH)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    parsed = json.loads(completed.stdout)
    assert parsed["schema"] == audit.SCHEMA
    assert parsed["precision_check"]["pass"] is True
    assert parsed["descriptive_comparisons"]["likelihood_used"] is False
    assert not list(tmp_path.iterdir())
