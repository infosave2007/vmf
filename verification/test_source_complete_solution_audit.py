"""Semantic tests for the bound source-complete solution audit."""

from __future__ import annotations

import copy
import difflib
import inspect
import json
import shutil
from fractions import Fraction
from pathlib import Path

import pytest

import source_complete_solution_audit as audit


def test_status_and_contract_are_derived_fail_closed():
    result = audit.build_result()
    assert result["status"] == "PASS_CONTRACT_FAIL_CLOSED_MELTING_NO_GO"
    assert result["contract"]["ensemble"].startswith("zero-temperature fixed baryon")
    assert result["gates"]["W_zero_at_finite_density"] is False
    assert result["gates"]["legacy_threshold_supported"] is False
    assert result["gates"]["renorm_matching_available"] is False


def test_vacuum_mass_and_higgs_charge_limits():
    result = audit.build_result()
    v = result["analytic_controls"]["vacuum"]
    p = result["parameters"]
    assert abs(v["W_MeV"] - p["W0_MeV"]) < 1.0e-12
    assert abs(v["Mstar_MeV"] - p["M_N_MeV"]) < 1.0e-12
    assert abs(v["A0_MeV"]) < 1.0e-12
    assert abs(p["q_phi"] - p["m_omega_MeV"] / p["W0_MeV"]) < 1.0e-12
    assert result["analytic_controls"]["W_to_zero_diverges"]


def test_coupled_branch_is_stationary_and_nonmelting():
    result = audit.build_result()
    rows = result["rows"]
    assert all(row["W_MeV"] > 0.0 for row in rows)
    assert rows[0]["W_MeV"] == result["parameters"]["W0_MeV"]
    high = next(row for row in rows if row["n_over_n0"] == 2.5)
    assert high["W_MeV"] > 0.99 * result["parameters"]["W0_MeV"]
    assert result["closure_summary"]["max_W_stationarity_relative"] < 1.0e-8
    assert result["closure_summary"]["max_current_relative"] < 1.0e-14


def test_hvh_legendre_and_generator_closure():
    result = audit.build_result()
    closure = result["closure_summary"]
    assert closure["max_HVH_mu_relative"] < 1.0e-7
    assert closure["max_HVH_pressure_relative"] < 1.0e-12
    assert closure["max_generator_residual_MeV4"] < 1.0e-5
    assert closure["max_stress_pressure_relative"] < 1.0e-11
    assert result["gates"]["hvh_legendre_closure"]


def test_convergence_and_legacy_comparison():
    result = audit.build_result()
    assert result["convergence"]["pass"]
    assert result["convergence"]["max_delta_MeV"] < 1.0e-7
    legacy = next(row for row in result["legacy_comparison"] if row["n_over_n0"] == 2.5)
    assert legacy["legacy_melting_branch"]
    assert result["scientific_boundary"]["legacy_threshold"].startswith("NOT_SUPPORTED")


def test_passport_and_correlated_jacobian_gate():
    result = audit.build_result()
    passport = result["passport"]
    required = audit.REQUIRED_PASSPORT_FIELDS
    assert all(required <= set(row) for row in passport["records"] + passport["derived_records"])
    assert passport["renormalization_block"]["record_count"] == 37
    assert passport["renormalization_block"]["parameter_refit"] == "NOT_PERFORMED"
    assert result["covariance_control"]["invariant_pass"]
    assert result["gates"]["correlated_derived_input_covariance"]


def test_massless_fermi_exact_and_continuous_limits():
    controls = audit.massless_controls()
    assert controls["exact_pass"]
    assert controls["continuity_pass"]
    for row in controls["rows"]:
        assert row["energy_formula_relative"] < 1.0e-14
        assert row["pressure_formula_relative"] < 1.0e-14
        assert row["energy_continuity_relative"] < 1.0e-9
        assert row["pressure_continuity_relative"] < 1.0e-9
        assert row["massless_trace_relative"] < 1.0e-14


def test_local_u1_manufactured_action_current_and_ward_controls():
    controls = audit.manufactured_gauge_controls()
    assert controls["all_pass"]
    assert controls["single_covariant_nucleon_coupling"]
    assert controls["no_duplicate_explicit_current"]
    assert controls["A_variation_current_relative"] < 1.0e-8
    assert max(controls[key] for key in ("ward_global_relative", "ward_connection_relative", "ward_full_relative")) < 1.0e-10


def test_explicit_euler_derivative_noether_ward_identity():
    controls = audit.manufactured_euler_ward_controls()
    assert controls["all_pass"]
    assert controls["identity_relative"] < 1.0e-10
    assert controls["connection_divergence_abs"] > 1.0
    result = audit.build_result()
    assert result["gates"]["euler_noether_ward_identity"]
    assert result["gates"]["local_u1_covariance_current_ward"]
    # A connection-divergence sign mutation is caught by the normalized
    # residual rather than being hidden by a transformed-action smoke test.
    assert not audit.manufactured_euler_ward_controls(connection_sign=-1.0)["all_pass"]


def test_euler_backend_is_pinned_and_does_not_call_legacy_stencil(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy cancellation stencil must not own the canonical control")
    monkeypatch.setattr(audit, "_legacy_manufactured_euler_ward_controls", forbidden)
    result = audit.manufactured_euler_ward_controls()
    assert result["all_pass"] is True
    assert result["finite_difference_step"] is None
    assert result["independent_variation_checks_pass"] is True
    assert result["precision_convergence_pass"] is True
    assert result["backend_source_sha256"] == audit.EULER_WARD_BACKEND_SHA256
    monkeypatch.setattr(audit, "EULER_WARD_BACKEND_SHA256", "0"*64)
    with pytest.raises(RuntimeError, match="backend identity changed"):
        audit.manufactured_euler_ward_controls()


def test_independent_fixed_coordinate_b_metric_variation_closes():
    controls = audit.metric_variation_controls()
    assert controls["pass"]
    assert tuple(controls["metric_steps"]) == (1.0e-4, 5.0e-5, 2.5e-5)
    source = inspect.getsource(audit.metric_variation_observables)
    assert "hilbert_stress" not in source
    assert "generator_G" not in source
    assert all(row["W_spread_MeV"] < 1.0e-7 for row in controls["rows"])
    assert all(row["pressure_metric_action_relative"] < 1.0e-6 for row in controls["rows"])


@pytest.mark.parametrize("caller_dps", [15, 37, 80])
def test_metric_action_is_independent_of_and_restores_caller_precision(caller_dps):
    cases = [(0.0, 1.0, 1.0), (0.1, 1.000025, 1.0),
             (0.1, 1.0, 1.000025), (2.5, 0.999975, 1.000025)]
    for x, lapse, scale in cases:
        B = x * audit.PARAMS.n0_nat
        W, A0 = audit.solve_metric_fields(lapse, scale, B)
        with audit.mp.workdps(50):
            reference = audit.reduced_lapse_scale_action(lapse, scale, B, W, A0)
        with audit.mp.workdps(caller_dps):
            actual = audit.reduced_lapse_scale_action(lapse, scale, B, W, A0)
            assert actual == reference
            assert audit.mp.mp.dps == caller_dps
        if x == 0.0:
            assert actual == 0.0
        else:
            assert actual < 0.0


def test_metric_action_restores_caller_precision_after_exception(monkeypatch):
    B = 0.1 * audit.PARAMS.n0_nat
    W, A0 = audit.solve_metric_fields(1.0, 1.0, B)
    # Establish that the same finite-density input succeeds before injection.
    assert audit.reduced_lapse_scale_action(1.0, 1.0, B, W, A0) < 0.0

    def interrupted_sqrt(_value):
        assert audit.mp.mp.dps == 50
        raise RuntimeError("injected metric-action failure")

    monkeypatch.setattr(audit.mp, "sqrt", interrupted_sqrt)
    with audit.mp.workdps(23):
        with pytest.raises(RuntimeError, match="injected metric-action failure"):
            audit.reduced_lapse_scale_action(1.0, 1.0, B, W, A0)
        assert audit.mp.mp.dps == 23


def test_passport_rejects_contract_mutations():
    baseline = audit.passport()
    assert audit.validate_passport(baseline)["pass"]

    mutations = []
    mutated = copy.deepcopy(baseline)
    mutated["unexpected"] = True
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["records"][0]["producer"] = ""
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["records"][1]["citation"] = "missing/source.md"
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["records"][2]["unit"] = "GeV"
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["derived_records"][0]["formula"] = "m_omega*W0"
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["derived_records"][0]["formula"] = "M_N/W0"
    mutated["derived_records"][1]["formula"] = "m_omega/W0"
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["derived_records"][1]["value"] *= 1.0 + 1.0e-12
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["derived_records"][0]["value"] = float("nan")
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["records"][0]["citation"] = "article/NVG_VACUUM_W_FIELD_DERIVATION_RU.md#missing-anchor"
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["records"][0]["transformation"] = "unknown"
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["records"][0]["input_id"] = "lambda"
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["renormalization_block"]["record_count"] = 36
    mutations.append(mutated)

    mutated = copy.deepcopy(baseline)
    mutated["renormalization_block"]["parameter_refit"] = "PERFORMED"
    mutations.append(mutated)

    for ident in ("W0", "m_omega", "M_N"):
        mutated = copy.deepcopy(baseline)
        next(row for row in mutated["records"] if row["input_id"] == ident)["value"] *= 1.1
        mutations.append(mutated)

    for combo in (("W0", "m_omega"), ("W0", "M_N"), ("m_omega", "M_N"), ("W0", "m_omega", "M_N")):
        mutated = copy.deepcopy(baseline)
        for index, ident in enumerate(combo):
            factor = 1.1 + 0.1 * index
            next(row for row in mutated["records"] if row["input_id"] == ident)["value"] *= factor
        mutations.append(mutated)

    assert all(not audit.validate_passport(payload)["pass"] for payload in mutations)


def test_passport_accepts_payload_relative_recomputation():
    payload = audit.passport()
    base = {row["input_id"]: row for row in payload["records"]}
    base["W0"]["value"] *= 1.1
    base["m_omega"]["value"] *= 1.1
    base["M_N"]["value"] *= 0.9
    derived = {row["input_id"]: row for row in payload["derived_records"]}
    derived["q_phi"]["value"] = base["m_omega"]["value"] / base["W0"]["value"]
    derived["g_s"]["value"] = base["M_N"]["value"] / base["W0"]["value"]
    assert audit.validate_passport(payload)["pass"]


def test_passport_resolves_without_private_run_directory(tmp_path, monkeypatch):
    payload = audit.passport()
    for record in payload["records"] + payload["derived_records"]:
        for field in ("producer", "source", "citation"):
            relative = record[field].split("#", 1)[0]
            assert not relative.startswith("Lunacy/")
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audit.REPO_ROOT / relative, target)
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    assert audit.validate_passport(payload)["pass"]


def test_reference_resolution_requires_exact_anchor(tmp_path: Path):
    original_root = audit.REPO_ROOT
    audit.REPO_ROOT = tmp_path
    try:
        target = tmp_path / "anchors.md"
        target.write_text("# Exact Heading\n\n<a id=\"explicit-anchor\"></a>\n", encoding="utf-8")
        assert audit._resolved_reference("anchors.md")
        assert audit._resolved_reference("anchors.md#exact-heading")
        assert audit._resolved_reference("anchors.md#explicit-anchor")
        assert not audit._resolved_reference("anchors.md#")
        assert not audit._resolved_reference("anchors.md#missing-anchor")
        assert not audit._resolved_reference("anchors.md#exact")
    finally:
        audit.REPO_ROOT = original_root


def test_artifacts_are_byte_regenerated(tmp_path: Path):
    result = audit.build_result()
    result_a = tmp_path / "a.json"
    report_a = tmp_path / "a.md"
    result_b = tmp_path / "b.json"
    report_b = tmp_path / "b.md"
    audit.write_artifacts(result, result_a, report_a)
    audit.write_artifacts(audit.build_result(), result_b, report_b)
    assert result_a.read_bytes() == result_b.read_bytes()
    assert report_a.read_bytes() == report_b.read_bytes()
    parsed = json.loads(result_a.read_text())
    assert parsed["status"] == result["status"]


def test_committed_generated_artifacts_match_reconstruction():
    result = audit.build_result()
    assert audit.RESULT_PATH.exists()
    assert audit.REPORT_PATH.exists()
    expected = audit.RESULT_PATH.read_bytes()
    reconstructed = audit.serialize_result(result)
    # Keep exact byte equality. If an old finite-difference diagnostic drifts,
    # retain the actual differing numbers instead of only a byte offset.
    assert expected == reconstructed, "\n".join(difflib.unified_diff(
        expected.decode("utf-8").splitlines(), reconstructed.decode("utf-8").splitlines(),
        fromfile="committed JSON", tofile="fresh reconstruction", lineterm=""))
    assert audit.REPORT_PATH.read_text(encoding="utf-8") == audit.render_report(result)


def test_scalar_covariance_matches_exact_sum_of_binary_products():
    control = audit.covariance_control()
    J, C = control["jacobian"], control["covariance"]
    for i in range(2):
        for j in range(2):
            products = [J[i][a]*C[a][b]*J[j][b] for a in range(3) for b in range(3)]
            exact_sum = sum((Fraction.from_float(term) for term in products), Fraction(0))
            assert control["derived_covariance"][i][j] == float(exact_sum)


def test_generated_report_exposes_repaired_gate_evidence():
    result = audit.build_result()
    report = audit.render_report(result)
    for phrase in (
        "Massless Fermi limits",
        "Manufactured local-U(1)",
        "Euler-derivative Noether/Ward identity",
        "Independent fixed-coordinate-`B` lapse/scale metric variation",
        "Strict passport validation",
        "W -> 0",
        "legacy_threshold_supported",
    ):
        assert phrase in report
