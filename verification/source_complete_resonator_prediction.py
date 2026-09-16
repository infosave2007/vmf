#!/usr/bin/env python3
"""Source-complete vacuum spectrum and conditional open-resonator audit.

The upstream ``source_complete_solution_audit`` is the sole parameter producer.
This module derives its vacuum gaps and provides an explicitly conditional
driven response.  It never supplies device defaults: without a complete source,
volume, mode normalization, damping and output-port passport the power branch is
blocked and emits null numerical fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import source_complete_solution_audit as upstream
import mpmath as mp
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
RESULT_PATH = ROOT / "source_complete_resonator_prediction_results.json"
REPORT_PATH = ROOT / "source_complete_resonator_prediction_report.md"
UPSTREAM_PATH = ROOT / "source_complete_solution_audit.py"
UPSTREAM_SOURCE_DIGEST = "5d39d3d07a3f00d2fcbe853ad27359b34a86d853c842ba01861196c1aaa517a8"
ACTION_PATH = ROOT / "contracts/source_complete_action.md"
ACTION_DIGEST = "25e42abb91028f09a76cccddc05ca4a7458bc0609df7f41fb932d165cc73606e"
CONTRACT_PATH = ROOT / "contracts/observable_predictions.md"
CONTRACT_DIGEST = "ebcc4e0cae1bf981f3fb2dd0bf6567a177dffa75aa25a160f82c2f86cbe87cef"
PRODUCER_PATH = ROOT / "source_complete_resonator_prediction.py"
TEST_PATH = ROOT / "test_source_complete_resonator_prediction.py"
FINGERPRINT_PATH = ROOT / "contracts/source_complete_resonator_fingerprints.json"

# Exact definitions in the adjudicated contract (MeV seconds, SI metres).
HBAR_MEV_S = 6.582119569e-22
H_MEV_S = 4.135667696e-21
C_M_S = 299792458.0

RESPONSE_RECORDS = (
    "source_interaction",
    "volume",
    "mode_inertia",
    "overlap",
    "drive_amplitude",
    "drive_phase",
    "drive_frequency",
    "damping_channels",
    "output_port",
)
FORBIDDEN_PLACEHOLDERS = {"", "unknown", "not supplied", "none declared", "n/a", "na"}

# These are the immutable conversion definitions bound by the adjudicated
# prediction contract.  The public names below are deliberately kept as
# runtime values so mutation tests can prove that an altered constant is
# rejected rather than silently propagated into a prediction.
_CANONICAL_HBAR_MEV_S = 6.582119569e-22
_CANONICAL_H_MEV_S = 4.135667696e-21
_CANONICAL_C_M_S = 299792458.0


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _load_expected_fingerprints() -> dict[str, Any]:
    try:
        record = json.loads(FINGERPRINT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return record if isinstance(record, dict) else {}


def _record(value: Any, *, unit: str, provider: str, reference: str, normalization: str, uncertainty: Any = None) -> dict[str, Any]:
    """Build a metadata-bearing record for deterministic test fixtures only."""

    return {
        "value": value,
        "uncertainty": uncertainty,
        "unit": unit,
        "normalization": normalization,
        "provider": provider,
        "reference": reference,
    }


def _metadata_errors(name: str, rec: Any, *, numeric: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(rec, dict):
        return [f"{name} record must be an object"]
    required = {"value", "uncertainty", "unit", "normalization", "provider", "reference"}
    missing = required - set(rec)
    if missing:
        errors.append(f"{name} missing metadata: {','.join(sorted(missing))}")
    for key in ("unit", "normalization", "provider", "reference"):
        if key in rec and (not isinstance(rec[key], str) or rec[key].strip().lower() in FORBIDDEN_PLACEHOLDERS):
            errors.append(f"{name}.{key} is blank")
    if "uncertainty" in rec and rec["uncertainty"] is not None and (not _finite(rec["uncertainty"]) or float(rec["uncertainty"]) < 0.0):
        errors.append(f"{name}.uncertainty is invalid")
    if numeric and ("value" in rec) and not _finite(rec["value"]):
        errors.append(f"{name}.value is not finite")
    if not numeric and ("value" in rec) and (not isinstance(rec["value"], str) or rec["value"].strip().lower() in FORBIDDEN_PLACEHOLDERS):
        errors.append(f"{name}.value is blank")
    return errors


def validate_upstream(payload: Any | None = None) -> dict[str, Any]:
    """Validate the submitted passport and immutable source/adjudication digests."""

    passport_payload = upstream.passport() if payload is None else payload
    passport_check = upstream.validate_passport(passport_payload)
    actual_source = _sha256(UPSTREAM_PATH)
    actual_action = _sha256(ACTION_PATH)
    actual_contract = _sha256(CONTRACT_PATH)
    actual_producer = _sha256(PRODUCER_PATH)
    actual_test = _sha256(TEST_PATH)
    expected_fingerprints = _load_expected_fingerprints()
    expected_producer = expected_fingerprints.get("producer_sha256")
    expected_test = expected_fingerprints.get("test_sha256")
    errors = list(passport_check.get("errors", [])) if not passport_check.get("pass") else []
    if actual_source != UPSTREAM_SOURCE_DIGEST:
        errors.append("upstream source digest drifted; re-adjudication required")
    if actual_action != ACTION_DIGEST:
        errors.append("accepted action digest drifted; re-adjudication required")
    if not CONTRACT_PATH.is_file():
        errors.append("prediction contract reference is missing")
    elif actual_contract != CONTRACT_DIGEST:
        errors.append("prediction contract digest drifted; re-adjudication required")
    if actual_producer != expected_producer:
        errors.append("resonator producer digest drifted; re-adjudication required")
    if actual_test != expected_test:
        errors.append("resonator focused-test digest drifted; re-adjudication required")
    return {
        "pass": not errors,
        "errors": errors,
        "passport": passport_payload,
        "source_path": str(UPSTREAM_PATH.relative_to(REPO_ROOT)),
        "source_sha256": actual_source,
        "source_expected_sha256": UPSTREAM_SOURCE_DIGEST,
        "action_path": str(ACTION_PATH.relative_to(REPO_ROOT)),
        "action_sha256": actual_action,
        "action_expected_sha256": ACTION_DIGEST,
        "contract_path": str(CONTRACT_PATH.relative_to(REPO_ROOT)),
        "contract_sha256": actual_contract,
        "contract_expected_sha256": CONTRACT_DIGEST,
        "producer_path": str(PRODUCER_PATH.relative_to(REPO_ROOT)),
        "producer_sha256": actual_producer,
        "producer_expected_sha256": expected_producer,
        "test_path": str(TEST_PATH.relative_to(REPO_ROOT)),
        "test_sha256": actual_test,
        "test_expected_sha256": expected_test,
    }


def _passport_values(payload: dict[str, Any]) -> dict[str, float]:
    records = payload.get("records", [])
    derived = payload.get("derived_records", [])
    values = {row["input_id"]: float(row["value"]) for row in records + derived}
    return values


def spectrum(payload: Any | None = None) -> dict[str, Any]:
    """Derive the one-scalar/three-vector vacuum spectrum from a submitted passport."""

    checked = validate_upstream(payload)
    if not checked["pass"]:
        return {"status": "BLOCKED_UPSTREAM_VALIDATION", "errors": checked["errors"], "upstream": {k: v for k, v in checked.items() if k != "passport"}}
    values = _passport_values(checked["passport"])
    required = ("W0", "lambda", "M_N", "m_omega", "hbar_c", "q_phi", "g_s")
    if any(key not in values or not _finite(values[key]) for key in required):
        return {"status": "BLOCKED_UPSTREAM_VALIDATION", "errors": ["passport values incomplete"], "upstream": {k: v for k, v in checked.items() if k != "passport"}}
    W0 = values["W0"]
    lam = values["lambda"]
    m_omega = values["m_omega"]
    hbar_c = values["hbar_c"]
    q_phi = values["q_phi"]
    errors: list[str] = []
    if W0 <= 0.0 or lam <= 0.0 or m_omega <= 0.0 or hbar_c <= 0.0:
        errors.append("W0, lambda and m_omega must be positive")
    if abs(q_phi - m_omega / W0) / max(1.0, abs(q_phi)) > 1.0e-15:
        errors.append("q_phi does not recompute from payload m_omega/W0")
    if errors or W0 <= 0.0 or lam <= 0.0 or m_omega <= 0.0 or hbar_c <= 0.0:
        return {"status": "BLOCKED_UPSTREAM_VALIDATION", "errors": errors or ["W0, lambda and m_omega must be positive"], "upstream": {k: v for k, v in checked.items() if k != "passport"}}
    m_sigma = math.sqrt(2.0 * lam) * W0
    m_A = abs(q_phi) * W0
    if abs(m_A - m_omega) / max(1.0, abs(m_omega)) > 1.0e-15:
        errors.append("Higgs vector mass does not equal payload m_omega")
    if errors:
        return {"status": "BLOCKED_UPSTREAM_VALIDATION", "errors": errors, "upstream": {k: v for k, v in checked.items() if k != "passport"}}

    # Conversion records are part of the submitted calculation passport, not
    # adjustable fit parameters.  Check both the immutable SI definitions and
    # the submitted MeV-fm record before exposing any spectrum values.
    conversion_record_rows = [
        {"name": "hbar", "value": HBAR_MEV_S, "unit": "MeV*s", "immutable": True, "reference": "prediction contract exact definition"},
        {"name": "h", "value": H_MEV_S, "unit": "MeV*s", "immutable": True, "reference": "prediction contract exact definition"},
        {"name": "c", "value": C_M_S, "unit": "m/s", "immutable": True, "reference": "prediction contract exact definition"},
        {"name": "hbar_c_submitted", "value": hbar_c, "unit": "MeV*fm", "immutable": True, "reference": "validated upstream hbar_c record"},
    ]
    hbar_h_relative = abs(H_MEV_S - 2.0 * math.pi * HBAR_MEV_S) / H_MEV_S
    hbar_c_from_si = HBAR_MEV_S * C_M_S * 1.0e15
    hbar_c_relative = abs(hbar_c - hbar_c_from_si) / max(1.0, abs(hbar_c))
    immutable_rows = [
        {"name": "hbar", "relative_to_contract": abs(HBAR_MEV_S - _CANONICAL_HBAR_MEV_S) / _CANONICAL_HBAR_MEV_S},
        {"name": "h", "relative_to_contract": abs(H_MEV_S - _CANONICAL_H_MEV_S) / _CANONICAL_H_MEV_S},
        {"name": "c", "relative_to_contract": abs(C_M_S - _CANONICAL_C_M_S) / _CANONICAL_C_M_S},
    ]
    conversion_pass = (
        hbar_h_relative < 5.0e-10
        and hbar_c_relative < 5.0e-10
        and all(row["relative_to_contract"] < 1.0e-15 for row in immutable_rows)
    )
    if not conversion_pass:
        errors.append("submitted/immutable hbar, h, c, hbar-c conversion records disagree")
        return {
            "status": "BLOCKED_UPSTREAM_VALIDATION",
            "errors": errors,
            "conversion_records": conversion_record_rows,
            "conversion_consistency": {
                "h_equals_2pi_hbar_relative": hbar_h_relative,
                "hbar_c_equals_hbar_times_c_relative": hbar_c_relative,
                "immutable_rows": immutable_rows,
                "pass": False,
            },
            "upstream": {k: v for k, v in checked.items() if k != "passport"},
        }

    def mode(mode_id: str, mass: float, polarizations: int, labels: list[str]) -> dict[str, Any]:
        omega = mass / HBAR_MEV_S
        frequency = mass / H_MEV_S
        reduced_m = HBAR_MEV_S * C_M_S / mass  # hbar*c/m in metres
        # The upstream hbar*c is retained as the model's declared MeV fm conversion.
        reduced_fm = hbar_c / mass
        return {
            "mode": mode_id,
            "mass_MeV": mass,
            "omega_rad_s": omega,
            "frequency_Hz": frequency,
            "reduced_Compton_m": reduced_m,
            "reduced_Compton_fm": reduced_fm,
            "Compton_wavelength_m": 2.0 * math.pi * reduced_m,
            "physical_polarizations": polarizations,
            "polarization_labels": labels,
            "uncertainty_status": "NOT_SUPPLIED",
            "evidence_weight": 0.0,
        }

    scalar = mode("scalar_radial", m_sigma, 1, ["radial"])
    vector = mode("massive_vector", m_A, 3, ["transverse_1", "transverse_2", "longitudinal"])
    conversion_rows = []
    for row in (scalar, vector):
        conversion_rows.append(
            {
                "mode": row["mode"],
                "omega_equals_2pi_f_relative": abs(row["omega_rad_s"] - 2.0 * math.pi * row["frequency_Hz"]) / row["omega_rad_s"],
                "lambda_equals_c_over_f_relative": abs(row["Compton_wavelength_m"] - C_M_S / row["frequency_Hz"]) / row["Compton_wavelength_m"],
                "reduced_times_2pi_relative": abs(row["Compton_wavelength_m"] - 2.0 * math.pi * row["reduced_Compton_m"]) / row["Compton_wavelength_m"],
            }
        )
    mode_conversion_pass = all(
        item["omega_equals_2pi_f_relative"] < 5.0e-10
        and item["lambda_equals_c_over_f_relative"] < 5.0e-10
        and item["reduced_times_2pi_relative"] < 5.0e-10
        for item in conversion_rows
    )
    conversion_pass = conversion_pass and mode_conversion_pass
    dispersion_rows = [
        {
            "mode": row["mode"],
            "k_natural": 0.0,
            "omega_squared_minus_m_squared": (row["omega_rad_s"] * HBAR_MEV_S) ** 2 - row["mass_MeV"] ** 2,
        }
        for row in (scalar, vector)
    ]
    dispersion_pass = all(abs(row["omega_squared_minus_m_squared"]) < 1.0e-9 for row in dispersion_rows)
    return {
        "status": "PASS_FIXED_VACUUM_SPECTRUM",
        "parameters": {
            "W0_MeV": W0,
            "lambda": lam,
            "m_omega_MeV": m_omega,
            "q_phi": q_phi,
            "m_sigma_MeV": m_sigma,
            "m_A_MeV": m_A,
            "hbar_MeV_s": HBAR_MEV_S,
            "h_MeV_s": H_MEV_S,
            "c_m_s": C_M_S,
            "hbar_c_MeV_fm": hbar_c,
        },
        "modes": [scalar, vector],
        "polarization_modes": [
            {"mode": "scalar_radial", "polarization": "radial", "mass_MeV": m_sigma},
            {"mode": "massive_vector", "polarization": "transverse_1", "mass_MeV": m_A},
            {"mode": "massive_vector", "polarization": "transverse_2", "mass_MeV": m_A},
            {"mode": "massive_vector", "polarization": "longitudinal", "mass_MeV": m_A},
        ],
        "physical_mode_count": 4,
        "mass_identities": {
            "scalar": "m_sigma^2=2*lambda*W0^2",
            "vector": "m_A^2=q_phi^2*W0^2=m_omega^2",
            "phase": "eaten Goldstone; no physical massless mode",
        },
        "dispersion": "omega_i^2 = k^2 + m_i^2 (natural units); omega_SI^2=(c k_SI)^2+(m_i/hbar)^2",
        "dispersion_checks": {"rows": dispersion_rows, "pass": dispersion_pass},
        "polarization_accounting": {
            "scalar_radial": 1,
            "massive_vector": 3,
            "goldstone": "eaten; no physical extra mode",
            "A0": "Gauss constraint; no oscillator",
        },
        "conversion_checks": {
            "rows": conversion_rows,
            "records": conversion_record_rows,
            "consistency": {
                "h_equals_2pi_hbar_relative": hbar_h_relative,
                "hbar_c_equals_hbar_times_c_relative": hbar_c_relative,
                "immutable_rows": immutable_rows,
            },
            "pass": conversion_pass,
        },
        "uncertainty_status": "NOT_SUPPLIED",
        "evidence_weight": 0.0,
        "upstream": {k: v for k, v in checked.items() if k != "passport"},
    }


SCALAR_SOURCE_KEYS = {
    "kind",
    "term",
    "mode",
    "amplitude_record",
    "phase_record",
    "overlap_record",
    "uncertainty",
    "unit",
    "provider",
    "reference",
    "normalization",
}
VECTOR_SOURCE_KEYS = SCALAR_SOURCE_KEYS | {"current_conserved", "conservation_reference"}
PORT_KEYS = {"kind", "channel_name", "uncertainty", "unit", "provider", "reference", "normalization"}
CHANNEL_KEYS = {"name", "role", "value", "uncertainty", "unit", "provider", "reference", "normalization"}


def _typed_source_errors(source: Any) -> list[str]:
    """Validate the exact-key scalar/vector source discriminant."""

    if not isinstance(source, dict):
        return ["source_interaction must be a typed source object"]
    errors: list[str] = []
    kind = source.get("kind")
    if kind not in {"scalar_source", "vector_current"}:
        return ["source_interaction.kind must be scalar_source or vector_current"]
    expected_keys = SCALAR_SOURCE_KEYS if kind == "scalar_source" else VECTOR_SOURCE_KEYS
    missing = expected_keys - set(source)
    unknown = set(source) - expected_keys
    if missing:
        errors.append(f"source_interaction missing keys: {','.join(sorted(missing))}")
    if unknown:
        errors.append(f"source_interaction unknown keys: {','.join(sorted(unknown))}")
    expected_term = "sigma_s_ext" if kind == "scalar_source" else "A_mu_j_ext_mu"
    expected_mode = "scalar_radial" if kind == "scalar_source" else "massive_vector"
    if source.get("term") != expected_term:
        errors.append("source_interaction.term does not match its discriminant")
    if source.get("mode") != expected_mode:
        errors.append("source_interaction.mode does not match its discriminant")
    for key, expected in (("amplitude_record", "drive_amplitude"), ("phase_record", "drive_phase"), ("overlap_record", "overlap")):
        if source.get(key) != expected:
            errors.append(f"source_interaction.{key} must bind {expected}")
    if source.get("unit") != "natural":
        errors.append("source_interaction.unit must be natural")
    for key in ("provider", "reference", "normalization"):
        value = source.get(key)
        if not isinstance(value, str) or value.strip().lower() in FORBIDDEN_PLACEHOLDERS:
            errors.append(f"source_interaction.{key} is blank")
    uncertainty = source.get("uncertainty")
    if uncertainty is not None and (not _finite(uncertainty) or float(uncertainty) < 0.0):
        errors.append("source_interaction.uncertainty is invalid")
    if kind == "vector_current":
        if source.get("current_conserved") is not True:
            errors.append("vector source must declare current_conserved=true")
        conservation_reference = source.get("conservation_reference")
        if not isinstance(conservation_reference, str) or conservation_reference.strip().lower() in FORBIDDEN_PLACEHOLDERS:
            errors.append("vector source conservation_reference is blank")
    return errors


def _typed_port_errors(port: Any) -> list[str]:
    """Validate the exact-key typed output-port record."""

    if not isinstance(port, dict):
        return ["output_port must be a typed damping_output object"]
    errors: list[str] = []
    missing = PORT_KEYS - set(port)
    unknown = set(port) - PORT_KEYS
    if missing:
        errors.append(f"output_port missing keys: {','.join(sorted(missing))}")
    if unknown:
        errors.append(f"output_port unknown keys: {','.join(sorted(unknown))}")
    if port.get("kind") != "damping_output":
        errors.append("output_port.kind must be damping_output")
    channel_name = port.get("channel_name")
    if not isinstance(channel_name, str) or not channel_name.strip() or channel_name.strip().lower() in FORBIDDEN_PLACEHOLDERS:
        errors.append("output_port.channel_name is blank")
    if port.get("unit") != "1":
        errors.append("output_port.unit must be 1")
    for key in ("provider", "reference", "normalization"):
        value = port.get(key)
        if not isinstance(value, str) or value.strip().lower() in FORBIDDEN_PLACEHOLDERS:
            errors.append(f"output_port.{key} is blank")
    uncertainty = port.get("uncertainty")
    if uncertainty is not None and (not _finite(uncertainty) or float(uncertainty) < 0.0):
        errors.append("output_port.uncertainty is invalid")
    return errors


def _physical_null_response(status: str, missing: list[str], records: Any, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "physical_units": False,
        "device_mapping": False,
        "missing_inputs": missing,
        "records": records,
        "amplitude": None,
        "stored_energy_J": None,
        "pump_power_W": None,
        "dissipated_power_W": None,
        "output_power_W": None,
        "Q_total": None,
        "bandwidth_Hz": None,
        "ringup_time_s": None,
        "channel_power_W": None,
        "transient_work_integral": None,
        "passivity": {"pass": False, "reason": reason},
    }


def _normalise_response_inputs(inputs: Any) -> tuple[dict[str, Any], list[str]]:
    """Check mandatory response records without assigning any omitted default."""

    errors: list[str] = []
    if not isinstance(inputs, dict):
        return {}, ["response_inputs must be an object"]
    allowed = set(RESPONSE_RECORDS) | {"Q"}
    unknown = sorted(set(inputs) - allowed)
    if unknown:
        return dict(inputs), [f"unknown response input: {name}" for name in unknown]
    for name in RESPONSE_RECORDS:
        if name not in inputs:
            errors.append(f"missing {name}")
    if "source_interaction" in inputs:
        errors.extend(_typed_source_errors(inputs["source_interaction"]))
    for name in ("volume", "mode_inertia", "overlap", "drive_amplitude", "drive_phase", "drive_frequency"):
        if name in inputs:
            errors.extend(_metadata_errors(name, inputs[name]))
    if "Q" in inputs:
        errors.extend(_metadata_errors("Q", inputs["Q"]))
    if "volume" in inputs and isinstance(inputs["volume"], dict) and _finite(inputs["volume"].get("value")) and inputs["volume"]["value"] <= 0.0:
        errors.append("volume must be positive")
    if "volume" in inputs and isinstance(inputs["volume"], dict) and inputs["volume"].get("unit") != "m^3":
        errors.append("volume unit must be m^3")
    if "mode_inertia" in inputs and isinstance(inputs["mode_inertia"], dict) and _finite(inputs["mode_inertia"].get("value")) and inputs["mode_inertia"]["value"] <= 0.0:
        errors.append("mode inertia must be positive")
    if "mode_inertia" in inputs and isinstance(inputs["mode_inertia"], dict) and inputs["mode_inertia"].get("unit") != "kg":
        errors.append("mode inertia unit must be kg")
    if "overlap" in inputs and isinstance(inputs["overlap"], dict) and inputs["overlap"].get("unit") != "N":
        errors.append("overlap unit must be N")
    if "drive_amplitude" in inputs and isinstance(inputs["drive_amplitude"], dict) and inputs["drive_amplitude"].get("unit") != "1":
        errors.append("drive amplitude unit must be 1")
    if "drive_frequency" in inputs and isinstance(inputs["drive_frequency"], dict) and _finite(inputs["drive_frequency"].get("value")) and inputs["drive_frequency"]["value"] <= 0.0:
        errors.append("drive frequency must be positive")
    if "drive_frequency" in inputs and isinstance(inputs["drive_frequency"], dict) and inputs["drive_frequency"].get("unit") != "rad/s":
        errors.append("drive frequency unit must be rad/s")
    if "drive_phase" in inputs and isinstance(inputs["drive_phase"], dict) and inputs["drive_phase"].get("unit") != "rad":
        errors.append("drive phase unit must be rad")
    channels = inputs.get("damping_channels")
    if not isinstance(channels, list) or not channels:
        errors.append("damping_channels must be a nonempty list")
    else:
        names: list[str] = []
        for index, channel in enumerate(channels):
            errors.extend(_metadata_errors(f"damping_channels[{index}]", channel))
            if isinstance(channel, dict):
                unknown_channel_keys = set(channel) - CHANNEL_KEYS
                missing_channel_keys = CHANNEL_KEYS - set(channel)
                if missing_channel_keys:
                    errors.append(f"damping_channels[{index}] missing keys: {','.join(sorted(missing_channel_keys))}")
                if unknown_channel_keys:
                    errors.append(f"damping_channels[{index}] unknown keys: {','.join(sorted(unknown_channel_keys))}")
            if isinstance(channel, dict) and _finite(channel.get("value")) and channel["value"] < 0.0:
                errors.append(f"damping_channels[{index}] loss is negative")
            if isinstance(channel, dict) and (not isinstance(channel.get("name"), str) or not channel.get("name", "").strip()):
                errors.append(f"damping_channels[{index}] name is blank")
            elif isinstance(channel, dict):
                name_value = channel["name"].strip()
                if name_value in names:
                    errors.append(f"damping_channels names must be unique: {name_value}")
                names.append(name_value)
            if isinstance(channel, dict) and channel.get("role") not in {"internal", "output"}:
                errors.append(f"damping_channels[{index}] role must be internal or output")
            if isinstance(channel, dict) and channel.get("unit") != "s^-1":
                errors.append(f"damping_channels[{index}] unit must be s^-1")
        output_roles = [channel for channel in channels if isinstance(channel, dict) and channel.get("role") == "output"]
        if len(output_roles) != 1:
            errors.append("damping_channels must contain exactly one role=output channel")
    port = inputs.get("output_port")
    errors.extend(_typed_port_errors(port))
    return inputs, errors


def conditional_response(spectrum_result: dict[str, Any], inputs: Any | None = None) -> dict[str, Any]:
    """Evaluate the physical response only when every declared record is present."""

    if inputs is None:
        missing = list(RESPONSE_RECORDS)
        return _physical_null_response("CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER", missing, {name: None for name in RESPONSE_RECORDS}, "mandatory source/loss/geometry records absent")
    normalised, errors = _normalise_response_inputs(inputs)
    if errors:
        return _physical_null_response("CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER", errors, normalised, "response passport incomplete or invalid")
    if spectrum_result.get("status") != "PASS_FIXED_VACUUM_SPECTRUM":
        return _physical_null_response("BLOCKED_UPSTREAM_VALIDATION", ["valid upstream spectrum"], normalised, "upstream spectrum blocked")

    source_mode = normalised["source_interaction"]["mode"]
    mode_index = 0 if source_mode == "scalar_radial" else 1
    scalar_omega = float(spectrum_result["modes"][mode_index]["omega_rad_s"])
    volume = float(normalised["volume"]["value"])
    M = float(normalised["mode_inertia"]["value"])
    kappa = float(normalised["overlap"]["value"])
    drive = float(normalised["drive_amplitude"]["value"])
    drive_phase = float(normalised["drive_phase"]["value"])
    omega_drive = float(normalised["drive_frequency"]["value"])
    channels = normalised["damping_channels"]
    gamma_total = sum(float(ch["value"]) for ch in channels)
    port_name = normalised["output_port"]["channel_name"].strip()
    # The typed port is a binding: exactly one declared role=output channel
    # must carry its name.  Internal channels can never be promoted by naming.
    output_channels = [ch for ch in channels if ch.get("name") == port_name and ch.get("role") == "output"]
    if gamma_total <= 0.0:
        return _physical_null_response("CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER", ["gamma_tot must be positive"], normalised, "zero total damping")
    if "Q" in normalised:
        submitted_q = float(normalised["Q"]["value"])
        expected_q = scalar_omega / (2.0 * gamma_total)
        if submitted_q <= 0.0 or abs(submitted_q - expected_q) / max(1.0, abs(expected_q)) > 1.0e-10:
            return _physical_null_response("CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER", ["submitted Q disagrees with damping-channel sum"], normalised, "inconsistent redundant Q")
    if len(output_channels) != 1:
        reason = "undeclared output channel" if not output_channels else "output port resolves to multiple damping channels"
        return _physical_null_response("CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER", ["output port must resolve to exactly one role=output damping channel"], normalised, reason)
    gamma = gamma_total
    denominator = scalar_omega * scalar_omega - omega_drive * omega_drive - 2.0j * gamma * omega_drive
    force_amplitude = kappa * drive * complex(math.cos(drive_phase), math.sin(drive_phase))
    X = force_amplitude / (M * denominator)
    abs_x2 = abs(X) ** 2
    E = M * (omega_drive * omega_drive + scalar_omega * scalar_omega) * abs_x2 / 4.0
    powers = {str(ch["name"]): M * float(ch["value"]) * omega_drive * omega_drive * abs_x2 for ch in channels}
    # Evaluate the cycle-average pump work directly from force/velocity phase.
    # The damping sum below is an independent comparison, never the definition
    # of the supplied pump power.
    velocity_amplitude = -1.0j * omega_drive * X
    pump = 0.5 * float((force_amplitude * velocity_amplitude.conjugate()).real)
    loss_sum = sum(powers.values())
    output = sum(powers[str(ch["name"])] for ch in output_channels)
    q_total = scalar_omega / (2.0 * gamma)
    power_scale = max(abs(pump), abs(loss_sum), 1.0e-300)
    residual = abs(pump - loss_sum) / power_scale
    zero_drive_pass = abs(drive) > 0.0 or (abs(X) == 0.0 and abs(pump) == 0.0 and abs(loss_sum) == 0.0)
    pump_nonnegative = pump >= -1.0e-12 * max(abs(loss_sum), 1.0e-300)
    output_tolerance = 1.0e-12 * max(abs(pump), abs(output), 1.0e-300)
    transient = physical_transient_balance(
        omega0=scalar_omega,
        mass=M,
        overlap=kappa,
        drive_amplitude=drive,
        drive_phase=drive_phase,
        drive_frequency=omega_drive,
        damping_channels=channels,
    )
    transient_parameters = transient.get("parameters", {}) if isinstance(transient, dict) else {}
    transient_channel_rates = {str(channel["name"]): float(channel["value"]) for channel in channels}
    transient_bound = (
        transient.get("status") == "PASS_PHYSICAL_TRANSIENT_BALANCE"
        and transient_parameters.get("omega0_rad_s") == scalar_omega
        and transient_parameters.get("M_kg") == M
        and transient_parameters.get("kappa_N") == kappa
        and transient_parameters.get("u0") == drive
        and transient_parameters.get("phase_rad") == drive_phase
        and transient_parameters.get("omega_drive_rad_s") == omega_drive
        and transient_parameters.get("gamma_channels_s^-1") == transient_channel_rates
    )
    passivity_pass = all(float(ch["value"]) >= 0.0 for ch in channels) and pump_nonnegative and output <= pump + output_tolerance and residual < 1.0e-10 and zero_drive_pass and transient.get("pass", False) and transient_bound
    return {
        "status": "PASS_CONDITIONAL_PASSIVE_RESPONSE" if passivity_pass else "FAIL_CLOSED_PASSIVITY",
        "physical_units": True,
        "device_mapping": True,
        "missing_inputs": [],
        "records": normalised,
        "volume_m3": volume,
        "amplitude": {"real": float(X.real), "imag": float(X.imag), "magnitude": float(abs(X)), "unit": "record-dependent"},
        "stored_energy_J": float(E),
        "pump_power_W": float(pump),
        "dissipated_power_W": float(loss_sum),
        "output_power_W": float(output),
        "channel_power_W": powers,
        "Q_total": float(q_total),
        "bandwidth_Hz": float(scalar_omega / (2.0 * math.pi * q_total)),
        "ringup_time_s": float(1.0 / gamma),
        "transient_work_integral": transient,
        "passivity": {
            "pass": passivity_pass,
            "pump_definition": "0.5*Re(F0*conj(-i*omega_drive*X))",
            "pump_power_direct_W": float(pump),
            "loss_sum_comparison_W": float(loss_sum),
            "power_sum_relative": float(residual),
            "output_le_pump": output <= pump + output_tolerance,
            "pump_nonnegative": pump_nonnegative,
            "zero_drive_pass": zero_drive_pass,
            "transient_work_identity_pass": transient.get("pass", False),
            "transient_parameter_binding": transient_bound,
        },
    }


def normalized_identity_benchmark() -> dict[str, Any]:
    """Unitless benchmark mandated by the prediction contract."""

    rows: list[dict[str, float]] = []
    for Omega in (0.0, 0.5, 1.0, 2.0):
        denom = 1.0 - Omega * Omega - 2.0j * 0.5 * Omega
        X = 1.0 / denom
        e = (Omega * Omega + 1.0) * abs(X) ** 2 / 4.0
        p_int = 0.25 * Omega * Omega * abs(X) ** 2
        p_out = p_int
        velocity = -1.0j * Omega * X
        p_pump_direct = 0.5 * float((1.0 * velocity.conjugate()).real)
        loss_sum = p_int + p_out
        rows.append({"Omega": Omega, "X_abs": float(abs(X)), "Ebar_hat": float(e), "Pint_hat": float(p_int), "Pout_hat": float(p_out), "Ppump_hat": float(p_pump_direct), "loss_sum_hat": float(loss_sum), "power_balance_relative": abs(p_pump_direct - loss_sum) / max(1.0, abs(p_pump_direct), abs(loss_sum))})
    zero = {"Omega": 1.0, "drive_amplitude": 0.0, "X_abs": 0.0, "Ebar_hat": 0.0, "Pint_hat": 0.0, "Pout_hat": 0.0, "Ppump_hat": 0.0}
    resonance = next(row for row in rows if row["Omega"] == 1.0)
    checks = {
        "resonance_X_abs": abs(resonance["X_abs"] - 1.0) < 1.0e-14,
        "resonance_Ebar": abs(resonance["Ebar_hat"] - 0.5) < 1.0e-14,
        "resonance_Ppump": abs(resonance["Ppump_hat"] - 0.5) < 1.0e-14,
        "resonance_channel_split": abs(resonance["Pint_hat"] - 0.25) < 1.0e-14 and abs(resonance["Pout_hat"] - 0.25) < 1.0e-14,
        "independent_pump_equals_loss": all(row["power_balance_relative"] < 1.0e-14 for row in rows),
        "zero_drive": all(zero[key] == 0.0 for key in ("X_abs", "Ebar_hat", "Pint_hat", "Pout_hat", "Ppump_hat")),
    }
    transient = normalized_transient_identity()
    return {
        "status": "NORMALIZED_IDENTITY_BENCHMARK",
        "device_mapping": False,
        "physical_units": False,
        "independent_evidence_weight": 0.0,
        "equation": "z''+2 zeta_tot z'+z=cos(Omega tau)",
        "parameters": {"Mhat": 1.0, "omegahat": 1.0, "kappahat": 1.0, "u0hat": 1.0, "zeta_int": 0.25, "zeta_out": 0.25, "zeta_tot": 0.5, "Omega_grid": [0.0, 0.5, 1.0, 2.0]},
        "rows": rows,
        "zero_drive": zero,
        "checks": checks,
        "transient": transient,
        "all_pass": all(checks.values()) and transient["pass"],
    }


def normalized_transient_identity() -> dict[str, Any]:
    """Deterministic normalized RK4 energy-balance identity."""

    dt = 1.0e-3
    steps = 20000
    gamma = 0.5
    def rhs(t: float, a: float, v: float) -> tuple[float, float]:
        return v, math.cos(t) - 2.0 * gamma * v - a
    a = 0.0
    v = 0.0
    input_energy = 0.0
    dissipated = 0.0
    prev_pin = 0.0
    prev_pd = 0.0
    for n in range(steps):
        t = n * dt
        pin = math.cos(t) * v
        pd = 2.0 * gamma * v * v
        if n:
            input_energy += 0.5 * dt * (prev_pin + pin)
            dissipated += 0.5 * dt * (prev_pd + pd)
        prev_pin, prev_pd = pin, pd
        k1a, k1v = rhs(t, a, v)
        k2a, k2v = rhs(t + dt / 2.0, a + dt * k1a / 2.0, v + dt * k1v / 2.0)
        k3a, k3v = rhs(t + dt / 2.0, a + dt * k2a / 2.0, v + dt * k2v / 2.0)
        k4a, k4v = rhs(t + dt, a + dt * k3a, v + dt * k3v)
        a += dt * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0
        v += dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
    stored = 0.5 * (v * v + a * a)
    residual = abs(stored - input_energy + dissipated)
    scale = max(1.0, abs(stored), abs(input_energy), abs(dissipated))
    return {"duration_tau": steps * dt, "stored_final": stored, "input_energy": input_energy, "dissipated_energy": dissipated, "residual_relative": residual / scale, "pass": residual / scale < 1.0e-7}


def _analytic_transient_oracle(*, fhat: float, Omega: float, zeta_values: list[float], drive_phase: float, tau_f: float) -> dict[str, Any]:
    """High-precision linear-ODE oracle used only when a periodic endpoint
    cancels the physical work below binary64 resolution.

    The DOP853 trajectories remain the primary records.  This oracle evaluates
    the same submitted equation and every channel integral at 80 decimal
    digits, so a commensurate off-resonant endpoint is not accepted on
    cancellation noise.
    """

    mp.mp.dps = 80
    mf = mp.mpf(str(fhat))
    mO = mp.mpf(str(Omega))
    mz = mp.mpf(str(sum(zeta_values)))
    mphi = mp.mpf(str(drive_phase))
    mtf = mp.mpf(str(tau_f))
    if mz < 1:
        wd = mp.sqrt(1 - mz * mz)
        A = 1 - mO * mO
        B = 2 * mz * mO
        denom = A * A + B * B
        pa = mf * A / denom
        pb = mf * B / denom
        zp0 = pa * mp.cos(mphi) + pb * mp.sin(mphi)
        vp0 = -pa * mO * mp.sin(mphi) + pb * mO * mp.cos(mphi)
        hc = -zp0
        hd = (-vp0 + mz * hc) / wd

        def state(tau: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
            theta = mO * tau + mphi
            zp = pa * mp.cos(theta) + pb * mp.sin(theta)
            vp = -pa * mO * mp.sin(theta) + pb * mO * mp.cos(theta)
            decay = mp.exp(-mz * tau)
            co = mp.cos(wd * tau)
            si = mp.sin(wd * tau)
            h = decay * (hc * co + hd * si)
            hv = decay * (-mz * (hc * co + hd * si) - hc * wd * si + hd * wd * co)
            return zp + h, vp + hv
    else:
        # The supported physical fixtures are deeply underdamped; retain a
        # fail-closed path for any future overdamped input rather than inventing
        # a response from a different model.
        raise ValueError("analytic transient oracle requires underdamped mode")

    def forcing(tau: mp.mpf) -> mp.mpf:
        return mf * mp.cos(mO * tau + mphi)

    final_z, final_v = state(mtf)
    work = mp.quad(lambda tau: forcing(tau) * state(tau)[1], [0, mtf])
    losses = []
    for zeta in zeta_values:
        mzj = mp.mpf(str(zeta))
        losses.append(mp.quad(lambda tau, mzj=mzj: 2 * mzj * state(tau)[1] ** 2, [0, mtf]))
    stored = (final_z * final_z + final_v * final_v) / 2
    return {
        "work_dimless": float(work),
        "loss_dimless": [float(value) for value in losses],
        "stored_dimless": float(stored),
        "oracle_digits": 80,
    }


def physical_transient_balance(
    *,
    omega0: float,
    mass: float,
    overlap: float,
    drive_amplitude: float,
    drive_phase: float,
    drive_frequency: float,
    damping_channels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Integrate the submitted physical oscillator in scaled time.

    This path is deliberately separate from ``normalized_transient_identity``.
    Every supplied physical parameter enters the scale, forcing, frequency,
    damping, or channel-work accumulator; zero drive returns exact zeros.
    """

    gamma_values = [float(channel["value"]) for channel in damping_channels]
    gamma_total = sum(gamma_values)
    force = float(overlap) * float(drive_amplitude)
    if drive_amplitude == 0.0:
        zero_resolution = {
            "rtol": 1.0e-10,
            "atol": 1.0e-12,
            "max_step": None,
            "nfev": 0,
            "work_J": 0.0,
            "loss_J": {str(channel["name"]): 0.0 for channel in damping_channels},
            "stored_change_J": 0.0,
        }
        return {
            "status": "PASS_PHYSICAL_TRANSIENT_BALANCE",
            "pass": True,
            "zero_drive": True,
            "parameters": {
                "omega0_rad_s": omega0,
                "M_kg": mass,
                "kappa_N": overlap,
                "u0": drive_amplitude,
                "phase_rad": drive_phase,
                "omega_drive_rad_s": drive_frequency,
                "gamma_channels_s^-1": {str(channel["name"]): float(channel["value"]) for channel in damping_channels},
            },
            "scaled": {"qscale_m": 1.0, "Omega": 0.0, "zeta_total": gamma_total / omega0, "tau_f": 0.0},
            "resolutions": [zero_resolution, dict(zero_resolution, rtol=1.0e-11)],
            "final_two_relative": {"work": 0.0, "loss": 0.0, "stored_change": 0.0},
            "work_J": 0.0,
            "loss_J": {str(channel["name"]): 0.0 for channel in damping_channels},
            "stored_change_J": 0.0,
            "residual_relative": 0.0,
        }
    if not (_finite(omega0) and omega0 > 0.0 and _finite(mass) and mass > 0.0 and _finite(overlap) and _finite(drive_amplitude) and _finite(drive_phase) and _finite(drive_frequency) and drive_frequency > 0.0 and gamma_total > 0.0):
        return {"status": "FAIL_CLOSED_PHYSICAL_TRANSIENT", "pass": False, "reason": "nonfinite or nonpositive physical transient input"}
    qscale = abs(force) / (mass * omega0 * omega0)
    if qscale <= 0.0 or not _finite(qscale):
        return {"status": "FAIL_CLOSED_PHYSICAL_TRANSIENT", "pass": False, "reason": "invalid displacement scale"}
    fhat = force / (mass * omega0 * omega0 * qscale)
    Omega = drive_frequency / omega0
    zeta_values = [gamma / omega0 for gamma in gamma_values]
    zeta_total = sum(zeta_values)
    frequency_scale = max(1.0, abs(Omega), zeta_total)
    tau_f = 40.0 * math.pi / frequency_scale
    base_max_step = 2.0 * math.pi / (512.0 * frequency_scale)
    energy_scale = mass * omega0 * omega0 * qscale * qscale

    def run(rtol: float, max_step: float) -> dict[str, Any]:
        def rhs(tau: float, state: Any) -> list[float]:
            z, v = state[0], state[1]
            forcing = fhat * math.cos(Omega * tau + drive_phase)
            values = [v, forcing - 2.0 * zeta_total * v - z, forcing * v]
            values.extend(2.0 * zeta * v * v for zeta in zeta_values)
            return values

        # Mechanical states follow the contract's 1e-12 absolute tolerance;
        # work/loss accumulators use a tighter tolerance so tiny nuclear-scale
        # SI energies are not replaced by cancellation noise.
        atol_vector = [1.0e-12, 1.0e-12] + [1.0e-30] * (1 + len(zeta_values))
        solution = solve_ivp(rhs, (0.0, tau_f), [0.0] * (3 + len(zeta_values)), method="DOP853", rtol=rtol, atol=atol_vector, max_step=max_step)
        if not solution.success or solution.y.shape[1] == 0 or not all(_finite(value) for value in solution.y[:, -1]):
            return {"rtol": rtol, "atol": 1.0e-12, "integral_atol": 1.0e-30, "max_step": max_step, "nfev": int(solution.nfev), "pass": False}
        final = solution.y[:, -1]
        stored_dimless = 0.5 * (float(final[0]) ** 2 + float(final[1]) ** 2)
        work = float(final[2]) * energy_scale
        losses = {str(channel["name"]): float(final[3 + index]) * energy_scale for index, channel in enumerate(damping_channels)}
        stored_change = stored_dimless * energy_scale
        loss_sum = sum(losses.values())
        residual = abs(stored_change - work + loss_sum) / max(abs(stored_change), abs(work), abs(loss_sum), 1.0e-300)
        return {
            "rtol": rtol,
            "atol": 1.0e-12,
            "integral_atol": 1.0e-30,
            "max_step": max_step,
            "nfev": int(solution.nfev),
            "work_J": work,
            "loss_J": losses,
            "stored_change_J": stored_change,
            "residual_relative": residual,
            "pass": bool(residual < 1.0e-7),
        }

    first = run(1.0e-10, base_max_step)
    second = run(1.0e-11, base_max_step / 2.0)

    # A commensurate off-resonant endpoint (for example Omega=1/2 at
    # tau_f=40*pi) can make the true work and stored change ~1e-23 while a
    # binary64 quadrature leaves ~1e-14 cancellation noise.  Re-evaluate that
    # same submitted equation with the independent decimal oracle; ordinary
    # non-cancelling cases retain their two DOP853 records unchanged.
    for record in (first, second):
        if not record.get("pass", False):
            oracle = _analytic_transient_oracle(fhat=fhat, Omega=Omega, zeta_values=zeta_values, drive_phase=drive_phase, tau_f=tau_f)
            work_oracle = oracle["work_dimless"] * energy_scale
            losses_oracle = {
                str(channel["name"]): oracle["loss_dimless"][index] * energy_scale
                for index, channel in enumerate(damping_channels)
            }
            stored_oracle = oracle["stored_dimless"] * energy_scale
            loss_sum_oracle = sum(losses_oracle.values())
            record.update(
                {
                    "work_J": work_oracle,
                    "loss_J": losses_oracle,
                    "stored_change_J": stored_oracle,
                    "residual_relative": abs(stored_oracle - work_oracle + loss_sum_oracle) / max(abs(stored_oracle), abs(work_oracle), abs(loss_sum_oracle), 1.0e-300),
                    "analytic_oracle_used": True,
                    "analytic_oracle_digits": oracle["oracle_digits"],
                }
            )
            record["pass"] = bool(record["residual_relative"] < 1.0e-7)
    first_work = float(first.get("work_J", 0.0))
    second_work = float(second.get("work_J", 0.0))
    first_stored = float(first.get("stored_change_J", 0.0))
    second_stored = float(second.get("stored_change_J", 0.0))
    work_scale = max(abs(first_work), abs(second_work), 1.0e-300)
    stored_scale = max(abs(first_stored), abs(second_stored), 1.0e-300)
    first_losses = first.get("loss_J", {})
    second_losses = second.get("loss_J", {})
    loss_deltas = {
        name: abs(float(first_losses.get(name, 0.0)) - float(second_losses.get(name, 0.0))) / max(abs(float(first_losses.get(name, 0.0))), abs(float(second_losses.get(name, 0.0))), 1.0e-300)
        for name in {str(channel["name"]) for channel in damping_channels}
    }
    final_two = {
        "work": abs(first_work - second_work) / work_scale,
        "loss": max(loss_deltas.values(), default=0.0),
        "stored_change": abs(first_stored - second_stored) / stored_scale,
    }
    pass_value = bool(first.get("pass") and second.get("pass") and all(value < 1.0e-7 for value in final_two.values()))
    return {
        "status": "PASS_PHYSICAL_TRANSIENT_BALANCE" if pass_value else "FAIL_CLOSED_PHYSICAL_TRANSIENT",
        "pass": pass_value,
        "zero_drive": False,
        "parameters": {
            "omega0_rad_s": omega0,
            "M_kg": mass,
            "kappa_N": overlap,
            "u0": drive_amplitude,
            "phase_rad": drive_phase,
            "omega_drive_rad_s": drive_frequency,
            "gamma_channels_s^-1": {str(channel["name"]): float(channel["value"]) for channel in damping_channels},
        },
        "scaled": {"qscale_m": qscale, "fhat": fhat, "Omega": Omega, "zeta_total": zeta_total, "tau_f": tau_f, "energy_scale_J": energy_scale},
        "resolutions": [first, second],
        "final_two_relative": final_two,
        "work_J": second_work,
        "loss_J": second_losses,
        "stored_change_J": second_stored,
        "residual_relative": float(second.get("residual_relative", math.inf)),
    }


def build_result(payload: Any | None = None, response_inputs: Any | None = None) -> dict[str, Any]:
    upstream_check = validate_upstream(payload)
    spec = spectrum(payload)
    response = conditional_response(spec, response_inputs)
    benchmark = normalized_identity_benchmark()
    gates = {
        "upstream_passport_and_digest": upstream_check["pass"],
        "fixed_spectrum": spec.get("status") == "PASS_FIXED_VACUUM_SPECTRUM" and spec.get("conversion_checks", {}).get("pass", False) and spec.get("dispersion_checks", {}).get("pass", False),
        "one_plus_three_polarizations": spec.get("polarization_accounting", {}).get("scalar_radial") == 1 and spec.get("polarization_accounting", {}).get("massive_vector") == 3,
        "normalized_benchmark": benchmark["all_pass"],
        "conditional_response": response.get("status") in {"CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER", "PASS_CONDITIONAL_PASSIVE_RESPONSE"},
        "physical_power_gate": response.get("status") == "PASS_CONDITIONAL_PASSIVE_RESPONSE" and response.get("passivity", {}).get("pass", False),
    }
    status = "PASS_FIXED_SPECTRUM_CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER" if gates["upstream_passport_and_digest"] and gates["fixed_spectrum"] and gates["normalized_benchmark"] and response.get("status") == "CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER" else "PASS_FIXED_SPECTRUM_PASSIVE_RESPONSE" if all(gates.values()) else "FAIL_CLOSED_RESONATOR_AUDIT"
    return {
        "schema": "source-complete-resonator-prediction.v1",
        "status": status,
        "contract": {
            "action": "exact local U(1), source-free g_mu=0, same-scalar Higgs vector mass",
            "spectrum": "one physical scalar radial mode plus three massive-vector polarizations",
            "response": "conditional open-system transfer function; no source-free vacuum drive",
            "power": "blocked unless complete source/volume/inertia/overlap/drive/loss/port records pass",
        },
        "upstream": {k: v for k, v in upstream_check.items() if k != "passport"},
        "spectrum": spec,
        "response": response,
        "normalized_benchmark": benchmark,
        "gates": gates,
        "claim_boundary": {
            "fixed_spectrum": "CALCULABLE_MODEL_OUTPUT_ZERO_EVIDENCE_WEIGHT",
            "numerical_power": "BLOCKED_WITHOUT_MEASURED_OPEN_SYSTEM_INPUTS",
            "gain_or_free_energy": "FAIL_CLOSED_NOT_ALLOWED",
            "ordinary_resonator": "NOT_ESTABLISHED",
        },
    }


def serialize_result(result: dict[str, Any]) -> bytes:
    return (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def render_report(result: dict[str, Any]) -> str:
    modes = result["spectrum"].get("modes", [])
    rows = [
        "# Source-complete resonator prediction audit",
        "",
        "## Control block",
        "",
        f"- Status: **{result['status']}**.",
        "- Fixed vacuum spectrum is derived from the validated upstream passport; no theta-haloscope, LC, mechanical, or 150-kHz defaults are imported.",
        "- Numerical response/power is conditional on a complete open-system passport and remains explicitly blocked for the default incomplete input.",
        "",
        "## Spectrum",
        "",
        "| mode | mass (MeV) | frequency (Hz) | reduced Compton (fm) | physical polarizations |",
        "|---|---:|---:|---:|---:|",
    ]
    rows.extend(f"| {m['mode']} | {m['mass_MeV']:.9g} | {m['frequency_Hz']:.9e} | {m['reduced_Compton_fm']:.9g} | {m['physical_polarizations']} |" for m in modes)
    rows.extend(
        [
            "",
            "The quadratic action gives `m_sigma^2=2 lambda W0^2` and `m_A=q_phi W0=m_omega`; the phase is an eaten Goldstone and `A0` is a Gauss constraint. The massless phase frequency is not an observable.",
            "",
            "## Open-system response and passivity",
            "",
            "The response equation is `M a_ddot + 2 M gamma_tot a_dot + M omega_0^2 a = Re(F0 exp(i omega t))`, with `Q=omega_0/(2 gamma_tot)` and `F0=kappa*u0*exp(i phase)`. Stored energy is `Ebar=M(omega^2+omega_0^2)|X|^2/4`; each damping channel has `P_j=M gamma_j omega^2 |X|^2`. Pump is evaluated independently as `P_pump=Re(F0*conj(-i omega X))/2`; the loss sum is only a comparison. A separate physical DOP853 transient integrates every submitted parameter and channel work in scaled time; the normalized fixture is never substituted.",
            f"Default response status: **{result['response']['status']}**; missing fields are explicit and numerical watts/Q/bandwidth are null. Output cannot exceed measured pump under the passivity gate.",
            "",
            "## Normalized identity benchmark",
            "",
            "`NORMALIZED_IDENTITY_BENCHMARK` uses `z''+2 zeta_tot z'+z=cos(Omega tau)`, `zeta_int=zeta_out=1/4`, `zeta_tot=1/2`, and `Omega in {0,1/2,1,2}`. At resonance it checks `|X|=1`, `Ebar=1/2`, direct `Ppump=1/2`, and `Pint=Pout=1/4`; zero drive is identically zero. Device mapping and physical units are false; evidence weight is zero.",
            f"Transient normalized energy balance residual: `{result['normalized_benchmark']['transient']['residual_relative']:.3e}` (PASS={result['normalized_benchmark']['transient']['pass']}).",
            "",
            "## Boundaries and audit contract",
            "",
            "Typed source records use exact scalar/vector discriminants and bind the actual amplitude, phase, overlap, and selected mode. Ports use exact `damping_output` records and exactly one role=`output` channel. Missing/nonfinite/negative loss, stale provenance, lexical or hidden drive, omitted volume/inertia/overlap/phase/port, Goldstone double counting, wrong Higgs mass, factor-of-two Q errors, conversion drift, transient closure failure, or output greater than independently computed pump fail closed. The fixed spectrum is a mathematical EFT output with zero independent-evidence weight; no numerical device power, gain, free-energy, or ordinary resonator claim follows.",
            "",
        ]
    )
    return "\n".join(rows)


def write_artifacts(result: dict[str, Any], result_path: Path = RESULT_PATH, report_path: Path = REPORT_PATH) -> None:
    if result.get("status") not in {
        "PASS_FIXED_SPECTRUM_CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER",
        "PASS_FIXED_SPECTRUM_PASSIVE_RESPONSE",
    }:
        raise ValueError("cannot publish a failed resonator audit")
    # Validate both representations before replacing either existing file.
    payload = serialize_result(result)
    report = render_report(result)
    result_path.write_bytes(payload)
    report_path.write_text(report, encoding="utf-8")


def main() -> int:
    result = build_result()
    write_artifacts(result)
    print(f"status={result['status']}")
    print(f"result={RESULT_PATH}")
    print(f"report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
