"""Semantic, mutation and deterministic checks for the resonator producer."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import source_complete_resonator_prediction as audit


def _rec(value, unit="1", normalization="unit-test normalization"):
    return {
        "value": value,
        "uncertainty": None,
        "unit": unit,
        "normalization": normalization,
        "provider": "verification/test_source_complete_resonator_prediction.py",
        "reference": "verification/contracts/observable_predictions.md",
    }


def complete_inputs():
    omega = audit.spectrum()["modes"][0]["omega_rad_s"]
    channels = [dict(_rec(0.25, "s^-1"), name="internal", role="internal"), dict(_rec(0.25, "s^-1"), name="output", role="output")]
    return {
        "source_interaction": {
            "kind": "scalar_source",
            "term": "sigma_s_ext",
            "mode": "scalar_radial",
            "amplitude_record": "drive_amplitude",
            "phase_record": "drive_phase",
            "overlap_record": "overlap",
            "uncertainty": None,
            "unit": "natural",
            "provider": "verification/test_source_complete_resonator_prediction.py",
            "reference": "verification/contracts/observable_predictions.md",
            "normalization": "declared typed scalar interaction",
        },
        "volume": _rec(1.0, "m^3"),
        "mode_inertia": _rec(1.0, "kg"),
        "overlap": _rec(1.0, "N"),
        "drive_amplitude": _rec(1.0, "1"),
        "drive_phase": _rec(0.0, "rad"),
        "drive_frequency": _rec(omega, "rad/s"),
        "damping_channels": channels,
        "output_port": {
            "kind": "damping_output",
            "channel_name": "output",
            "uncertainty": None,
            "unit": "1",
            "provider": "verification/test_source_complete_resonator_prediction.py",
            "reference": "verification/contracts/observable_predictions.md",
            "normalization": "declared output port",
        },
    }


def test_default_result_is_fixed_spectrum_with_blocked_power():
    result = audit.build_result()
    assert result["status"] == "PASS_FIXED_SPECTRUM_CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    assert result["gates"]["upstream_passport_and_digest"]
    assert result["gates"]["fixed_spectrum"]
    assert result["gates"]["normalized_benchmark"]
    assert not result["gates"]["physical_power_gate"]
    response = result["response"]
    assert response["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    assert response["stored_energy_J"] is None
    assert response["pump_power_W"] is None
    assert response["output_power_W"] is None
    assert response["Q_total"] is None


def test_spectrum_masses_conversions_and_polarizations():
    spec = audit.spectrum()
    assert spec["status"] == "PASS_FIXED_VACUUM_SPECTRUM"
    scalar, vector = spec["modes"]
    assert scalar["physical_polarizations"] == 1
    assert vector["physical_polarizations"] == 3
    assert spec["physical_mode_count"] == 4
    assert abs(scalar["mass_MeV"] - 1244.8092624976728) < 1.0e-9
    assert abs(vector["mass_MeV"] - 782.6) < 1.0e-12
    assert spec["conversion_checks"]["pass"]
    assert spec["dispersion_checks"]["pass"]
    assert len(spec["polarization_modes"]) == 4
    assert spec["uncertainty_status"] == "NOT_SUPPLIED"


def test_normalized_benchmark_and_transient_identity():
    benchmark = audit.normalized_identity_benchmark()
    assert benchmark["all_pass"]
    resonance = next(row for row in benchmark["rows"] if row["Omega"] == 1.0)
    assert resonance["X_abs"] == pytest.approx(1.0, abs=1.0e-14)
    assert resonance["Ebar_hat"] == pytest.approx(0.5, abs=1.0e-14)
    assert resonance["Ppump_hat"] == pytest.approx(0.5, abs=1.0e-14)
    assert resonance["Pint_hat"] == pytest.approx(0.25, abs=1.0e-14)
    assert resonance["Pout_hat"] == pytest.approx(0.25, abs=1.0e-14)
    assert benchmark["zero_drive"]["Ppump_hat"] == 0.0
    assert benchmark["transient"]["residual_relative"] < 1.0e-7


def test_complete_response_is_passive_and_traces_channels():
    result = audit.build_result(response_inputs=complete_inputs())
    response = result["response"]
    assert response["status"] == "PASS_CONDITIONAL_PASSIVE_RESPONSE"
    assert response["physical_units"] and response["device_mapping"]
    assert response["passivity"]["pass"]
    assert response["pump_power_W"] == pytest.approx(response["dissipated_power_W"])
    assert response["output_power_W"] <= response["pump_power_W"]
    assert response["Q_total"] == pytest.approx(
        result["spectrum"]["modes"][0]["omega_rad_s"] / (2.0 * sum(ch["value"] for ch in complete_inputs()["damping_channels"])),
        rel=1.0e-12,
    )
    assert sum(response["channel_power_W"].values()) == pytest.approx(response["pump_power_W"], rel=1.0e-12)
    assert result["gates"]["physical_power_gate"]
    assert response["transient_work_integral"]["pass"]
    assert response["transient_work_integral"]["parameters"]["M_kg"] == 1.0


def test_incomplete_metadata_and_negative_loss_fail_closed():
    inputs = complete_inputs()
    del inputs["volume"]["provider"]
    response = audit.conditional_response(audit.spectrum(), inputs)
    assert response["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    assert response["stored_energy_J"] is None
    inputs = complete_inputs()
    inputs["damping_channels"][0]["value"] = -1.0
    response = audit.conditional_response(audit.spectrum(), inputs)
    assert response["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    assert response["output_power_W"] is None
    inputs = complete_inputs()
    inputs["output_port"]["provider"] = "unknown"
    response = audit.conditional_response(audit.spectrum(), inputs)
    assert response["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"


def test_zero_drive_has_zero_response():
    inputs = complete_inputs()
    inputs["drive_amplitude"]["value"] = 0.0
    response = audit.conditional_response(audit.spectrum(), inputs)
    assert response["status"] == "PASS_CONDITIONAL_PASSIVE_RESPONSE"
    assert response["amplitude"]["magnitude"] == 0.0
    assert response["pump_power_W"] == 0.0
    assert response["output_power_W"] == 0.0
    assert response["transient_work_integral"]["zero_drive"]
    assert response["transient_work_integral"]["work_J"] == 0.0


def test_missing_output_port_and_hidden_defaults_rejected():
    inputs = complete_inputs()
    inputs.pop("output_port")
    response = audit.conditional_response(audit.spectrum(), inputs)
    assert response["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    assert response["output_power_W"] is None
    assert "missing output_port" in response["missing_inputs"]
    response = audit.conditional_response(audit.spectrum(), {"Q": 1.0})
    assert response["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"


def test_redundant_q_must_match_channel_sum():
    inputs = complete_inputs()
    omega = audit.spectrum()["modes"][0]["omega_rad_s"]
    inputs["Q"] = _rec(omega / (2.0 * 0.5), "1")
    assert audit.conditional_response(audit.spectrum(), inputs)["status"] == "PASS_CONDITIONAL_PASSIVE_RESPONSE"
    inputs["Q"]["value"] *= 2.0
    blocked = audit.conditional_response(audit.spectrum(), inputs)
    assert blocked["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    assert "disagrees" in blocked["missing_inputs"][0]


def test_upstream_payload_mutations_and_digest_fail_closed(monkeypatch):
    payload = copy.deepcopy(audit.upstream.passport())
    payload["derived_records"][0]["value"] *= 1.01
    assert audit.validate_upstream(payload)["pass"] is False
    monkeypatch.setattr(audit, "UPSTREAM_SOURCE_DIGEST", "bad-digest")
    assert audit.build_result()["status"] == "FAIL_CLOSED_RESONATOR_AUDIT"


def test_hbar_c_mutation_fails_closed_instead_of_rescaling_prediction():
    payload = copy.deepcopy(audit.upstream.passport())
    base = next(row for row in payload["records"] if row["input_id"] == "hbar_c")
    base["value"] *= 1.01
    altered = audit.spectrum(payload)
    assert altered["status"] == "BLOCKED_UPSTREAM_VALIDATION"
    assert any("conversion" in err for err in altered["errors"])


def test_immutable_c_mutation_fails_closed(monkeypatch):
    monkeypatch.setattr(audit, "C_M_S", audit.C_M_S * 1.01)
    altered = audit.spectrum()
    assert altered["status"] == "BLOCKED_UPSTREAM_VALIDATION"
    assert any("conversion" in err for err in altered["errors"])


@pytest.mark.parametrize("name", ["HBAR_MEV_S", "H_MEV_S"])
def test_immutable_hbar_h_mutations_fail_closed(monkeypatch, name):
    monkeypatch.setattr(audit, name, getattr(audit, name) * 1.01)
    altered = audit.spectrum()
    assert altered["status"] == "BLOCKED_UPSTREAM_VALIDATION"
    assert any("conversion" in err for err in altered["errors"])


def test_producer_and_test_provenance_are_embedded_and_validated(monkeypatch):
    result = audit.build_result()
    assert result["upstream"]["producer_sha256"] == result["upstream"]["producer_expected_sha256"]
    assert result["upstream"]["test_sha256"] == result["upstream"]["test_expected_sha256"]
    monkeypatch.setattr(audit, "_load_expected_fingerprints", lambda: {"producer_sha256": result["upstream"]["producer_expected_sha256"], "test_sha256": "bad-digest"})
    assert audit.build_result()["status"] == "FAIL_CLOSED_RESONATOR_AUDIT"


def test_runtime_specifications_are_portable_and_live():
    for path in (audit.ACTION_PATH, audit.CONTRACT_PATH, audit.FINGERPRINT_PATH):
        assert path.is_file()
        assert "Lunacy" not in path.relative_to(audit.REPO_ROOT).parts
    assert audit.validate_upstream()["pass"] is True


def test_failed_audit_cannot_overwrite_result_files(tmp_path, monkeypatch):
    result_path = tmp_path / "result.json"
    report_path = tmp_path / "report.md"
    result_path.write_text("previous result", encoding="utf-8")
    report_path.write_text("previous report", encoding="utf-8")
    monkeypatch.setattr(audit, "UPSTREAM_SOURCE_DIGEST", "invalid-source")
    failed = audit.build_result()
    assert failed["status"] == "FAIL_CLOSED_RESONATOR_AUDIT"
    with pytest.raises(ValueError, match="cannot publish a failed"):
        audit.write_artifacts(failed, result_path, report_path)
    assert result_path.read_text(encoding="utf-8") == "previous result"
    assert report_path.read_text(encoding="utf-8") == "previous report"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scientific_values_cannot_be_serialized(value):
    with pytest.raises(ValueError):
        audit.serialize_result({"computed_value": value})


def test_drive_phase_is_bound_and_pump_is_not_loss_sum():
    inputs = complete_inputs()
    response = audit.conditional_response(audit.spectrum(), inputs)
    assert response["passivity"]["pump_definition"].startswith("0.5*Re")
    assert response["pump_power_W"] == pytest.approx(response["passivity"]["pump_power_direct_W"])
    assert response["dissipated_power_W"] == pytest.approx(response["passivity"]["loss_sum_comparison_W"])
    inputs["source_interaction"]["phase_record"] = "wrong_phase"
    blocked = audit.conditional_response(audit.spectrum(), inputs)
    assert blocked["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    assert blocked["pump_power_W"] is None
    assert not blocked["passivity"]["pass"]


def test_output_port_requires_exactly_one_named_channel():
    inputs = complete_inputs()
    inputs["output_port"]["channel_name"] = "missing"
    blocked = audit.conditional_response(audit.spectrum(), inputs)
    assert blocked["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    inputs = complete_inputs()
    inputs["damping_channels"].append(dict(_rec(0.1, "s^-1"), name="output", role="output"))
    blocked = audit.conditional_response(audit.spectrum(), inputs)
    assert blocked["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"


def test_source_interaction_must_be_nonzero_and_consistent():
    inputs = complete_inputs()
    inputs["source_interaction"]["term"] = "A_mu_j_ext_mu"
    blocked = audit.conditional_response(audit.spectrum(), inputs)
    assert blocked["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    inputs = complete_inputs()
    inputs["source_interaction"] = {
        "kind": "vector_current",
        "term": "A_mu_j_ext_mu",
        "mode": "massive_vector",
        "amplitude_record": "drive_amplitude",
        "phase_record": "drive_phase",
        "overlap_record": "overlap",
        "current_conserved": True,
        "conservation_reference": "declared conserved current",
        "uncertainty": None,
        "unit": "natural",
        "provider": "verification/test_source_complete_resonator_prediction.py",
        "reference": "verification/contracts/observable_predictions.md",
        "normalization": "declared typed vector interaction",
    }
    inputs["drive_frequency"]["value"] = audit.spectrum()["modes"][1]["omega_rad_s"]
    # A vector source is semantically supported and binds to the vector mode.
    assert audit.conditional_response(audit.spectrum(), inputs)["status"] == "PASS_CONDITIONAL_PASSIVE_RESPONSE"


def test_typed_source_rejects_lexical_or_cross_variant_records():
    inputs = complete_inputs()
    inputs["source_interaction"] = dict(inputs["source_interaction"], kind="vector_current", term="sigma_s_ext")
    assert audit.conditional_response(audit.spectrum(), inputs)["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    inputs = complete_inputs()
    inputs["source_interaction"] = dict(inputs["source_interaction"], value="L_ext=A_mu*j_ext^mu; sigma appears only as a label")
    assert audit.conditional_response(audit.spectrum(), inputs)["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    inputs = complete_inputs()
    inputs["source_interaction"]["amplitude_record"] = "other_amplitude"
    assert audit.conditional_response(audit.spectrum(), inputs)["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"


def test_output_role_is_semantic_not_just_a_name():
    inputs = complete_inputs()
    inputs["damping_channels"][0]["role"] = "output"
    inputs["damping_channels"][1]["role"] = "internal"
    assert audit.conditional_response(audit.spectrum(), inputs)["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
    inputs = complete_inputs()
    inputs["damping_channels"][1]["role"] = "mystery"
    assert audit.conditional_response(audit.spectrum(), inputs)["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"


def _physical_fixture(mass, kappa, drive, gamma_internal, gamma_output, *, frequency_scale=1.0):
    inputs = complete_inputs()
    omega = audit.spectrum()["modes"][0]["omega_rad_s"]
    inputs["mode_inertia"]["value"] = mass
    inputs["overlap"]["value"] = kappa
    inputs["drive_amplitude"]["value"] = drive
    inputs["drive_frequency"]["value"] = omega * frequency_scale
    inputs["damping_channels"][0]["value"] = gamma_internal
    inputs["damping_channels"][1]["value"] = gamma_output
    return inputs


def test_physical_transient_uses_submitted_parameters_and_refines():
    fixtures = [
        _physical_fixture(1.0, 1.0, 1.0, 0.25, 0.25),
        _physical_fixture(10.0, 2.0, 3.0, 0.5, 0.1),
        _physical_fixture(0.1, 0.2, 4.0, 2.0, 0.3),
        _physical_fixture(1.0, 1.0, 1.0, 0.25, 0.25, frequency_scale=0.5),
    ]
    transients = []
    for fixture in fixtures:
        response = audit.conditional_response(audit.spectrum(), fixture)
        assert response["status"] == "PASS_CONDITIONAL_PASSIVE_RESPONSE"
        transient = response["transient_work_integral"]
        assert transient["pass"]
        assert len(transient["resolutions"]) == 2
        assert max(transient["final_two_relative"].values()) < 1.0e-7
        transients.append(json.dumps(transient, sort_keys=True))
    assert len(set(transients)) == len(transients)


def test_normalized_transient_substitution_is_rejected(monkeypatch):
    monkeypatch.setattr(audit, "physical_transient_balance", lambda **kwargs: audit.normalized_transient_identity())
    response = audit.conditional_response(audit.spectrum(), complete_inputs())
    assert response["status"] == "FAIL_CLOSED_PASSIVITY"
    assert not response["passivity"]["transient_parameter_binding"]


def test_invalid_spectrum_blocks_response():
    payload = copy.deepcopy(audit.upstream.passport())
    next(row for row in payload["records"] if row["input_id"] == "lambda")["value"] = -1.0
    result = audit.build_result(payload=payload)
    assert result["status"] == "FAIL_CLOSED_RESONATOR_AUDIT"
    assert result["spectrum"]["status"] == "BLOCKED_UPSTREAM_VALIDATION"


def test_artifacts_byte_regenerate_and_report_exposes_controls(tmp_path: Path):
    result = audit.build_result()
    first_json, first_report = tmp_path / "a.json", tmp_path / "a.md"
    second_json, second_report = tmp_path / "b.json", tmp_path / "b.md"
    audit.write_artifacts(result, first_json, first_report)
    audit.write_artifacts(audit.build_result(), second_json, second_report)
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_report.read_bytes() == second_report.read_bytes()
    assert json.loads(first_json.read_text())["schema"] == "source-complete-resonator-prediction.v1"
    committed = audit.RESULT_PATH.read_bytes()
    assert committed == audit.serialize_result(result)
    assert audit.REPORT_PATH.read_text(encoding="utf-8") == audit.render_report(result)
    report = audit.REPORT_PATH.read_text(encoding="utf-8")
    for phrase in ("Spectrum", "Open-system response", "NORMALIZED_IDENTITY_BENCHMARK", "blocked", "passivity"):
        assert phrase.lower() in report.lower()


def test_record_alias_coupling_is_not_a_hidden_default():
    inputs = complete_inputs()
    inputs["coupling"] = inputs.pop("overlap")
    assert audit.conditional_response(audit.spectrum(), inputs)["status"] == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER"
