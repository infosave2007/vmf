#!/usr/bin/env python3
"""Passive series-RLC consistency probe with a frozen synthetic benchmark.

The probe is deliberately an input/validator/fit tool, not a physical NVG
model.  It accepts a JSON measurement (frequency, complex admittance,
declared Cartesian noise, and *independently measured* low/high frequency
endpoint constraints), fits a positive series-RLC family, and reports a
conditional consistency diagnostic.  A sweep-only fit is retained as a fair
baseline: it sees exactly the same observations, but does not let the endpoint
constraints move its parameters.  The proposed constrained fit and the
baseline are never described as a universal passive certificate.

The phasor convention is exp(-i omega t), so the Laplace substitution is
``s = -i*omega`` and a capacitor has ``Z_C = +i/(omega*C)``.  Thus the
series model is exactly ``Y(s) = 1/(R + sL + 1/(sC))``.  Under the same
convention, an ideal capacitor has negative-imaginary admittance and an ideal
inductor has positive-imaginary admittance.

Importing this module is side-effect free.  The CLI prints one JSON document
to stdout and writes a file only when ``--output`` is explicitly supplied.
Benchmark generation is deterministic and uses only declared synthetic
truths; truth/case labels are kept outside the payload passed to ``analyze``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # scipy is part of the repository's research environment.
    from scipy.optimize import least_squares
    from scipy.stats import chi2
except Exception as exc:  # pragma: no cover - import failure is surfaced by CLI.
    least_squares = None  # type: ignore[assignment]
    chi2 = None  # type: ignore[assignment]
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


HERE = Path(__file__).resolve().parent
SCHEMA = "nvg_passive_response_input.v1"
RESULT_SCHEMA = "nvg_passive_response_probe.v1"
BENCHMARK_SCHEMA = "nvg_passive_response_benchmark.v1"
STATUS = "COMPUTED_PASSIVE_RLC_CONSISTENCY_DIAGNOSTIC"
EVIDENCE_WEIGHT = 0.0

# Frozen model/bounds.  These are numerical design choices, not calibrated
# scientific coefficients.  The nominal benchmark reference is explicitly
# synthetic and deliberately not an NVG physical coefficient.
REFERENCE_R = 10.0
REFERENCE_L = 1.0e-2
REFERENCE_C = 1.0e-6
PARAMETER_BOUNDS = {
    "R_ohm": (1.0e-6, 1.0e6),
    "L_H": (1.0e-12, 1.0e6),
    "C_F": (1.0e-15, 1.0e2),
}
FIT_MAX_NFEV = 2500
# Three deterministic starts are enough for this three-parameter, bounded
# problem and keep the full 64+128 held-out benchmark inexpensive.  The
# endpoint-centered start is always first; two deliberately distinct starts
# protect the sweep-only baseline when an endpoint is badly misspecified.
FIT_START_FACTORS = (0.1, 1.0, 10.0)
COVARIANCE_MAX_CONDITION = 1.0e12
DEFAULT_ALPHA = 0.01
DEFAULT_MIN_FREQ_HZ = 1.0e-12
DEFAULT_SIGMA_FLOOR_S = 1.0e-18

IMAGINARY_SIGN = "negative_for_capacitive_e_minus_iomega_t"
PHASOR_CONVENTION = "e^-iomega_t"
SUPPORTED_FREQUENCY_UNIT = "Hz"
SUPPORTED_ADMITTANCE_UNIT = "S"


class PassiveResponseError(ValueError):
    """Fail-closed input, numerical, or protocol error."""


class NumericalFitError(PassiveResponseError):
    """No finite, converged optimizer result satisfied the fit contract."""


@dataclass(frozen=True)
class EndpointConstraint:
    """One independent endpoint observation."""

    value: float
    sigma: float
    unit: str


@dataclass(frozen=True)
class Measurement:
    """Validated, immutable numerical input used by both fit methods."""

    frequencies_hz: np.ndarray
    admittance_s: np.ndarray
    sigma_real_s: np.ndarray
    sigma_imag_s: np.ndarray
    capacitance: EndpointConstraint
    inductance: EndpointConstraint
    units: Mapping[str, str]
    constraint_source: str

    @property
    def n(self) -> int:
        return int(self.frequencies_hz.size)


@dataclass
class FitResult:
    """Internal fit result; serialised through ``_fit_to_json``."""

    method: str
    success: bool
    params: dict[str, float]
    predicted_s: np.ndarray
    chi2_sweep: float
    chi2_endpoint: float
    chi2_total: float
    dof: int
    score: float
    message: str
    nfev: int
    jacobian: np.ndarray | None
    covariance_log: np.ndarray | None
    jacobian_rank: int | None
    jacobian_condition_number: float | None
    covariance_status: str


def _finite_number(value: Any, label: str) -> float:
    """Return a finite real number, rejecting booleans and complex values."""

    if isinstance(value, bool) or isinstance(value, (list, tuple, dict)):
        raise PassiveResponseError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PassiveResponseError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise PassiveResponseError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        raise PassiveResponseError(f"{label} must be > 0")
    return result


def _json_number(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise PassiveResponseError("nonfinite numerical result")
    return value


def _jsonable(value: Any) -> Any:
    """Convert numpy/scalar values while failing closed on non-finite data."""

    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise PassiveResponseError("nonfinite numerical result")
        return value
    if isinstance(value, complex):
        raise PassiveResponseError("complex values must be represented by real/imag fields")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except FileNotFoundError as exc:
        raise PassiveResponseError(f"input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PassiveResponseError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise PassiveResponseError("top-level JSON value must be an object")
    return payload


def _canonical_unit(units: Mapping[str, Any], key: str, expected: str) -> str:
    value = units.get(key)
    if not isinstance(value, str) or value != expected:
        raise PassiveResponseError(f"units.{key} must be exactly {expected!r}")
    return value


def _as_vector(value: Any, n: int, label: str, *, positive: bool = False) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise PassiveResponseError(f"{label} must have length {n}")
        values = [_finite_number(item, f"{label}[{i}]") for i, item in enumerate(value)]
    else:
        item = _finite_number(value, label)
        values = [item] * n
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise PassiveResponseError(f"{label} contains nonfinite values")
    if positive and np.any(array <= 0.0):
        raise PassiveResponseError(f"{label} must be > 0")
    return array


def _parse_admittance_point(value: Any, index: int) -> complex:
    if isinstance(value, Mapping):
        real = value.get("real_S", value.get("real"))
        imag = value.get("imag_S", value.get("imag"))
        if real is None or imag is None:
            raise PassiveResponseError(f"admittance[{index}] needs real_S and imag_S")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        real, imag = value
    else:
        raise PassiveResponseError(
            f"admittance[{index}] must be {{real_S, imag_S}} or [real, imag]"
        )
    return complex(
        _finite_number(real, f"admittance[{index}].real"),
        _finite_number(imag, f"admittance[{index}].imag"),
    )


def _constraint_entry(
    constraints: Mapping[str, Any],
    names: Sequence[str],
    value_unit: str,
    sigma_names: Sequence[str],
    label: str,
) -> EndpointConstraint:
    entry: Any = None
    for name in names:
        if name in constraints:
            entry = constraints[name]
            break
    if entry is None:
        raise PassiveResponseError(f"missing endpoint constraint: {label}")
    if not isinstance(entry, Mapping):
        raise PassiveResponseError(f"endpoint_constraints.{label} must be an object")
    if "value" not in entry:
        raise PassiveResponseError(f"endpoint_constraints.{label}.value is required")
    sigma: Any = None
    for name in sigma_names:
        if name in entry:
            sigma = entry[name]
            break
    if sigma is None:
        raise PassiveResponseError(f"endpoint_constraints.{label} sigma is required")
    value_num = _positive(entry["value"], f"endpoint_constraints.{label}.value")
    sigma_num = _positive(sigma, f"endpoint_constraints.{label}.sigma")
    unit = entry.get("unit", value_unit)
    if unit != value_unit:
        raise PassiveResponseError(f"endpoint_constraints.{label}.unit must be {value_unit!r}")
    return EndpointConstraint(value=value_num, sigma=sigma_num, unit=value_unit)


def parse_input(payload: Mapping[str, Any]) -> Measurement:
    """Validate canonical measurement JSON and return an immutable input.

    Extra keys (including benchmark truth/case labels) are ignored rather than
    consulted.  This is intentional: classification can depend only on the
    measured data, declared noise, units, and endpoint constraints.
    """

    if not isinstance(payload, Mapping):
        raise PassiveResponseError("top-level JSON value must be an object")
    schema = payload.get("schema")
    if schema != SCHEMA:
        raise PassiveResponseError(f"schema must be {SCHEMA!r}")

    units = payload.get("units")
    if not isinstance(units, Mapping):
        raise PassiveResponseError("units object is required")
    _canonical_unit(units, "frequency", SUPPORTED_FREQUENCY_UNIT)
    _canonical_unit(units, "admittance", SUPPORTED_ADMITTANCE_UNIT)
    phasor = units.get("phasor_convention")
    if phasor != PHASOR_CONVENTION:
        raise PassiveResponseError(f"units.phasor_convention must be {PHASOR_CONVENTION!r}")
    imaginary_sign = units.get("imaginary_sign")
    if imaginary_sign != IMAGINARY_SIGN:
        raise PassiveResponseError(f"units.imaginary_sign must be {IMAGINARY_SIGN!r}")

    measurement = payload.get("measurement")
    if not isinstance(measurement, Mapping):
        raise PassiveResponseError("measurement object is required")
    frequency_key = "frequencies_hz" if "frequencies_hz" in measurement else "frequencies"
    if frequency_key not in measurement:
        raise PassiveResponseError("measurement.frequencies_hz is required")
    frequencies = measurement[frequency_key]
    if not isinstance(frequencies, (list, tuple)) or len(frequencies) < 5:
        raise PassiveResponseError("measurement.frequencies_hz must contain at least 5 values")
    freq = np.asarray(
        [_positive(item, f"measurement.frequencies_hz[{i}]") for i, item in enumerate(frequencies)],
        dtype=float,
    )
    if np.any(freq <= DEFAULT_MIN_FREQ_HZ):
        raise PassiveResponseError("measurement frequencies must be strictly positive")
    if np.any(np.diff(freq) <= 0.0):
        raise PassiveResponseError("measurement frequencies must be strictly increasing; duplicates are invalid")
    if "admittance" not in measurement:
        raise PassiveResponseError("measurement.admittance is required")
    values = measurement["admittance"]
    if not isinstance(values, (list, tuple)) or len(values) != len(freq):
        raise PassiveResponseError("measurement.admittance must match frequency length")
    admittance = np.asarray([_parse_admittance_point(item, i) for i, item in enumerate(values)], dtype=complex)
    if not np.all(np.isfinite(admittance.real)) or not np.all(np.isfinite(admittance.imag)):
        raise PassiveResponseError("measurement.admittance contains nonfinite values")

    noise = payload.get("noise")
    if not isinstance(noise, Mapping):
        raise PassiveResponseError("noise object with declared positive sigma is required")
    model = noise.get("model", "independent_gaussian_cartesian")
    if model not in ("independent_gaussian_cartesian", "independent_cartesian_gaussian"):
        raise PassiveResponseError("noise.model must declare independent Cartesian Gaussian noise")
    # Canonical keys are sigma_real_S/sigma_imag_S; common explicit aliases are
    # accepted to keep hand-authored measurement files readable.
    sigma_real_value = noise.get("sigma_real_S", noise.get("real_sigma_S"))
    sigma_imag_value = noise.get("sigma_imag_S", noise.get("imag_sigma_S"))
    sigma_common = noise.get("sigma_S")
    if sigma_real_value is None and sigma_common is not None:
        sigma_real_value = sigma_common
    if sigma_imag_value is None and sigma_common is not None:
        sigma_imag_value = sigma_common
    relative = noise.get("relative_sigma")
    if sigma_real_value is None and relative is not None:
        rel = _positive(relative, "noise.relative_sigma")
        sigma_floor = _positive(noise.get("sigma_floor_S", DEFAULT_SIGMA_FLOOR_S), "noise.sigma_floor_S")
        sigma_real = np.maximum(rel * np.maximum(np.abs(admittance.real), sigma_floor), sigma_floor)
    else:
        if sigma_real_value is None:
            raise PassiveResponseError("noise.sigma_real_S is required")
        sigma_real = _as_vector(sigma_real_value, len(freq), "noise.sigma_real_S", positive=True)
    if sigma_imag_value is None and relative is not None:
        rel = _positive(relative, "noise.relative_sigma")
        sigma_floor = _positive(noise.get("sigma_floor_S", DEFAULT_SIGMA_FLOOR_S), "noise.sigma_floor_S")
        sigma_imag = np.maximum(rel * np.maximum(np.abs(admittance.imag), sigma_floor), sigma_floor)
    else:
        if sigma_imag_value is None:
            raise PassiveResponseError("noise.sigma_imag_S is required")
        sigma_imag = _as_vector(sigma_imag_value, len(freq), "noise.sigma_imag_S", positive=True)

    constraints = payload.get("endpoint_constraints")
    if not isinstance(constraints, Mapping):
        raise PassiveResponseError("endpoint_constraints object is required")
    capacitance = _constraint_entry(
        constraints,
        ("low_frequency_capacitance_F", "capacitance_F", "C_F"),
        "F",
        ("sigma_F", "sigma", "uncertainty_F"),
        "capacitance",
    )
    inductance = _constraint_entry(
        constraints,
        ("high_frequency_inductance_H", "inductance_H", "L_H"),
        "H",
        ("sigma_H", "sigma", "uncertainty_H"),
        "inductance",
    )
    metadata = payload.get("constraint_metadata", {})
    independent = metadata.get("independent") if isinstance(metadata, Mapping) else None
    if independent is not True and payload.get("constraints_independently_measured") is not True:
        raise PassiveResponseError(
            "endpoint constraints must explicitly declare independent=true; do not infer them from this sweep"
        )
    source = "separate_measurement"
    if isinstance(metadata, Mapping) and isinstance(metadata.get("source"), str):
        source = metadata["source"]
    if not source.strip():
        raise PassiveResponseError("constraint_metadata.source must not be empty")

    # Make arrays read-only so a fit cannot accidentally mutate the validated
    # measurement shared by the two methods.
    for array in (freq, admittance, sigma_real, sigma_imag):
        array.setflags(write=False)
    return Measurement(
        frequencies_hz=freq,
        admittance_s=admittance,
        sigma_real_s=sigma_real,
        sigma_imag_s=sigma_imag,
        capacitance=capacitance,
        inductance=inductance,
        units={"frequency": "Hz", "admittance": "S", "phasor_convention": PHASOR_CONVENTION, "imaginary_sign": IMAGINARY_SIGN},
        constraint_source=source,
    )


def series_rlc_admittance(frequencies_hz: Iterable[float], R_ohm: float, L_H: float, C_F: float) -> np.ndarray:
    """Return ``Y`` for the declared exp(-i omega t) series-RLC convention."""

    r = _positive(R_ohm, "R_ohm")
    l = _positive(L_H, "L_H")
    c = _positive(C_F, "C_F")
    frequencies = np.asarray(list(frequencies_hz), dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise PassiveResponseError("frequencies_hz must be finite and > 0")
    s = -1j * 2.0 * math.pi * frequencies
    z = r + s * l + 1.0 / (s * c)
    y = 1.0 / z
    if not np.all(np.isfinite(y.real)) or not np.all(np.isfinite(y.imag)):
        raise PassiveResponseError("series-RLC evaluation produced nonfinite admittance")
    return y


def absorbed_power_watts(voltage_peak_v: complex, admittance_s: complex) -> float:
    """Compute peak-phasor ``0.5 Re(V I*) = 0.5 |V|² Re(Y)`` in watts.

    ``voltage_peak_v`` is explicitly a peak (not RMS) phasor.  Callers with an
    RMS voltage must pass ``sqrt(2)*V_rms`` before using this function.

    The equality is independent of whether a caller uses the conjugate
    exp(+i omega t) representation: changing both V and Y to their complex
    conjugates leaves the real absorbed power unchanged.
    """

    v = complex(voltage_peak_v)
    y = complex(admittance_s)
    i = y * v
    direct = 0.5 * (v * np.conjugate(i)).real
    equivalent = 0.5 * (abs(v) ** 2) * y.real
    if not math.isclose(float(direct), float(equivalent), rel_tol=1e-12, abs_tol=1e-15):
        raise PassiveResponseError("absorbed-power identity failed")
    return float(equivalent)


def _endpoint_residuals(measurement: Measurement, params: Mapping[str, float]) -> np.ndarray:
    c = (float(params["C_F"]) - measurement.capacitance.value) / measurement.capacitance.sigma
    l = (float(params["L_H"]) - measurement.inductance.value) / measurement.inductance.sigma
    return np.asarray([c, l], dtype=float)


def _measurement_residuals(measurement: Measurement, predicted: np.ndarray) -> np.ndarray:
    real = (predicted.real - measurement.admittance_s.real) / measurement.sigma_real_s
    imag = (predicted.imag - measurement.admittance_s.imag) / measurement.sigma_imag_s
    return np.concatenate((real, imag)).astype(float, copy=False)


def _params_from_log(log_params: np.ndarray) -> dict[str, float]:
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        values = np.power(10.0, np.asarray(log_params, dtype=float))
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise PassiveResponseError("invalid logarithmic fit parameters")
    return {"R_ohm": float(values[0]), "L_H": float(values[1]), "C_F": float(values[2])}


def _log_bounds() -> tuple[np.ndarray, np.ndarray]:
    low = np.log10([PARAMETER_BOUNDS[key][0] for key in ("R_ohm", "L_H", "C_F")])
    high = np.log10([PARAMETER_BOUNDS[key][1] for key in ("R_ohm", "L_H", "C_F")])
    return low.astype(float), high.astype(float)


def _start_points(measurement: Measurement) -> list[np.ndarray]:
    y = measurement.admittance_s
    z = np.divide(1.0, y, out=np.full_like(y, np.nan + 1j * np.nan), where=np.abs(y) > 0)
    r0 = float(np.nanmedian(np.maximum(np.real(z), PARAMETER_BOUNDS["R_ohm"][0])))
    if not math.isfinite(r0):
        r0 = REFERENCE_R
    # Endpoint measurements are independent but are valid numerical starts for
    # the constrained fit and a deliberately explicit baseline prior; starts do
    # not alter the objective or scientific decision.
    c0 = measurement.capacitance.value
    l0 = measurement.inductance.value
    lows, highs = _log_bounds()
    starts: list[np.ndarray] = []
    for rf, lf, cf in (
        (1.0, 1.0, 1.0),
        (FIT_START_FACTORS[2], 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ):
        values = np.log10([r0 * rf, l0 * lf, c0 * cf])
        starts.append(np.clip(values, lows + 1e-9, highs - 1e-9))
    # Add a simple sweep-only impedance-scale start and the frozen reference
    # start for robustness on inputs whose endpoints are intentionally wrong.
    starts.append(np.clip(np.log10([r0, REFERENCE_L, REFERENCE_C]), lows + 1e-9, highs - 1e-9))
    # Deduplicate deterministic starts.
    unique: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for start in starts:
        key = tuple(np.round(start, 12))
        if key not in seen:
            unique.append(start)
            seen.add(key)
    return unique


def _fit(measurement: Measurement, *, include_endpoints: bool, method: str) -> FitResult:
    if least_squares is None:  # pragma: no cover - environment failure.
        raise PassiveResponseError(f"scipy.optimize.least_squares unavailable: {_SCIPY_IMPORT_ERROR}")
    lows, highs = _log_bounds()

    def residual(log_params: np.ndarray) -> np.ndarray:
        params = _params_from_log(log_params)
        predicted = series_rlc_admittance(measurement.frequencies_hz, **params)
        pieces = [_measurement_residuals(measurement, predicted)]
        if include_endpoints:
            pieces.append(_endpoint_residuals(measurement, params))
        result = np.concatenate(pieces)
        if not np.all(np.isfinite(result)):
            raise PassiveResponseError("nonfinite fit residual")
        return result

    best: Any = None
    best_cost = math.inf
    for start in _start_points(measurement):
        try:
            candidate = least_squares(
                residual,
                start,
                bounds=(lows, highs),
                method="trf",
                x_scale="jac",
                max_nfev=FIT_MAX_NFEV,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
        except (ValueError, FloatingPointError, OverflowError, PassiveResponseError):
            continue
        # A low-cost result is not usable evidence unless scipy explicitly
        # reports convergence and all serialized numerical fields are finite.
        # In particular, a test double or a max-evaluation result must never
        # become a compatible/pass classification.
        try:
            if not bool(getattr(candidate, "success", False)):
                continue
            if not np.all(np.isfinite(candidate.x)) or not np.isfinite(candidate.cost):
                continue
            candidate_residual = np.asarray(residual(candidate.x), dtype=float)
            candidate_jac = np.asarray(candidate.jac, dtype=float)
            if not np.all(np.isfinite(candidate_residual)) or not np.all(np.isfinite(candidate_jac)):
                continue
            if candidate_jac.ndim != 2 or candidate_jac.shape[1] != 3:
                continue
        except (AttributeError, TypeError, ValueError, FloatingPointError, OverflowError, PassiveResponseError):
            continue
        if float(candidate.cost) < best_cost:
            best = candidate
            best_cost = float(candidate.cost)
    if best is None:
        raise NumericalFitError(f"{method} has no finite converged candidate")
    try:
        params = _params_from_log(best.x)
        predicted = series_rlc_admittance(measurement.frequencies_hz, **params)
        sweep_residual = _measurement_residuals(measurement, predicted)
        endpoint_residual = _endpoint_residuals(measurement, params)
    except (ValueError, FloatingPointError, OverflowError, PassiveResponseError) as exc:
        raise NumericalFitError(f"{method} converged candidate could not be serialized") from exc
    chi2_sweep = float(np.dot(sweep_residual, sweep_residual))
    chi2_endpoint = float(np.dot(endpoint_residual, endpoint_residual))
    chi2_total = chi2_sweep + (chi2_endpoint if include_endpoints else 0.0)
    observation_count = 2 * measurement.n + (2 if include_endpoints else 0)
    dof = max(1, observation_count - 3)
    score = chi2_total
    covariance: np.ndarray | None = None
    jacobian_rank: int | None = None
    jacobian_condition: float | None = None
    covariance_status = "missing_jacobian"
    j = np.asarray(best.jac, dtype=float)
    if j.ndim == 2 and j.shape[1] == 3 and np.all(np.isfinite(j)):
        try:
            singular_values = np.linalg.svd(j, compute_uv=False)
            if singular_values.size:
                tolerance = np.finfo(float).eps * max(j.shape) * singular_values[0]
                jacobian_rank = int(np.count_nonzero(singular_values > tolerance))
                smallest = float(singular_values[-1])
                jacobian_condition = float(singular_values[0] / smallest) if smallest > 0.0 else math.inf
                if jacobian_rank == 3 and jacobian_condition <= COVARIANCE_MAX_CONDITION:
                    normal = j.T @ j
                    covariance = np.linalg.solve(normal, np.eye(3, dtype=float))
                    if not np.all(np.isfinite(covariance)):
                        covariance = None
                        covariance_status = "invalid_nonfinite_covariance"
                    else:
                        covariance_status = "valid_known_noise_log10_local"
                elif jacobian_rank != 3:
                    covariance_status = "invalid_rank"
                else:
                    covariance_status = "invalid_conditioning"
        except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
            covariance = None
            covariance_status = "invalid_covariance"
    return FitResult(
        method=method,
        success=True,
        params=params,
        predicted_s=predicted,
        chi2_sweep=chi2_sweep,
        chi2_endpoint=chi2_endpoint,
        chi2_total=chi2_total,
        dof=dof,
        score=score,
        message=str(getattr(best, "message", "converged")),
        nfev=int(getattr(best, "nfev", 0)),
        jacobian=j,
        covariance_log=covariance,
        jacobian_rank=jacobian_rank,
        jacobian_condition_number=jacobian_condition,
        covariance_status=covariance_status,
    )


def _safe_pow10(exponent: float) -> tuple[float | None, str]:
    """Convert a log10 endpoint without clipping away overflow evidence."""

    if not math.isfinite(exponent):
        return None, "nonfinite"
    min_log10 = math.log10(np.nextafter(0.0, 1.0))
    max_log10 = math.log10(np.finfo(float).max)
    if exponent < min_log10:
        return None, "underflow"
    if exponent > max_log10:
        return None, "overflow"
    try:
        value = float(10.0**exponent)
    except (OverflowError, ValueError, FloatingPointError):
        return None, "overflow"
    if not math.isfinite(value):
        return None, "overflow"
    return value, "ok"


def _intervals(fit: FitResult) -> dict[str, Any]:
    names = ("R_ohm", "L_H", "C_F")
    output: dict[str, dict[str, Any]] = {}
    if fit.covariance_log is None or fit.covariance_log.shape != (3, 3):
        for name in names:
            value = fit.params[name]
            output[name] = {
                "estimate": value,
                "low_95": None,
                "high_95": None,
                "interval_status": fit.covariance_status,
            }
        return output
    z = 1.959963984540054
    for i, name in enumerate(names):
        variance = float(fit.covariance_log[i, i])
        value = fit.params[name]
        if not math.isfinite(variance) or variance < 0.0:
            low = high = None
            interval_status = "invalid_variance"
        else:
            sigma_log10 = math.sqrt(variance)
            center_log10 = math.log10(value)
            low, low_status = _safe_pow10(center_log10 - z * sigma_log10)
            high, high_status = _safe_pow10(center_log10 + z * sigma_log10)
            statuses = [status for status in (low_status, high_status) if status != "ok"]
            if statuses:
                interval_status = "invalid_" + "_and_".join(statuses)
            else:
                interval_status = "valid_known_noise_log10_local"
                lo_bound, hi_bound = PARAMETER_BOUNDS[name]
                if low < lo_bound or high > hi_bound:
                    interval_status = "valid_but_outside_declared_fit_bounds"
        output[name] = {
            "estimate": value,
            "low_95": low,
            "high_95": high,
            "interval_status": interval_status,
        }
    return output


def _chi2_threshold(dof: int, alpha: float = DEFAULT_ALPHA) -> float:
    if chi2 is None:  # pragma: no cover - environment failure.
        raise PassiveResponseError(f"scipy.stats.chi2 unavailable: {_SCIPY_IMPORT_ERROR}")
    if not (0.0 < alpha < 1.0):
        raise PassiveResponseError("alpha must be in (0,1)")
    threshold = float(chi2.ppf(1.0 - alpha, dof))
    if not math.isfinite(threshold):
        raise PassiveResponseError("nonfinite chi-square threshold")
    return threshold


def _fit_to_json(measurement: Measurement, fit: FitResult, threshold: float, alpha: float) -> dict[str, Any]:
    endpoint_residual = _endpoint_residuals(measurement, fit.params)
    return {
        "method": fit.method,
        "success": fit.success,
        "fit_status": "CONVERGED_FINITE_CANDIDATE" if fit.success else "INDETERMINATE_NUMERICAL_FIT",
        "parameters": _jsonable(fit.params),
        "conditional_parameter_intervals_95": _jsonable(_intervals(fit)),
        "uncertainty_assumptions": "known declared Gaussian sigmas; whitened-Jacobian Fisher covariance in log10 coordinates; local/asymptotic only; no residual-variance rescaling",
        "jacobian_rank": fit.jacobian_rank,
        "jacobian_condition_number": _json_number(fit.jacobian_condition_number) if fit.jacobian_condition_number is not None and math.isfinite(fit.jacobian_condition_number) else None,
        "covariance_status": fit.covariance_status,
        "covariance_coordinate": "log10(R_ohm,L_H,C_F)",
        "covariance_log10_known_noise": _jsonable(fit.covariance_log) if fit.covariance_log is not None else None,
        "chi2_sweep": _json_number(fit.chi2_sweep),
        "chi2_endpoint": _json_number(fit.chi2_endpoint),
        "chi2_total": _json_number(fit.chi2_total),
        "degrees_of_freedom": int(fit.dof),
        "decision_alpha": _json_number(alpha),
        "decision_threshold_chi2": _json_number(threshold),
        "score": _json_number(fit.score),
        "pass_single_series_rlc": bool(fit.success and fit.score <= threshold),
        "nfev": int(fit.nfev),
        "message": fit.message,
        "endpoint_standardized_residuals": _jsonable(endpoint_residual),
    }


def _passivity_diagnostic(measurement: Measurement) -> dict[str, Any]:
    real = measurement.admittance_s.real
    # This is a finite-sample warning only.  It is not a global positive-real
    # certificate and cannot rule out hidden positive modes outside the band.
    scale = np.maximum(measurement.sigma_real_s, DEFAULT_SIGMA_FLOOR_S)
    materially_negative = real < -5.0 * scale
    return {
        "minimum_measured_real_admittance_S": _json_number(float(np.min(real))),
        "minimum_real_admittance_sigma_units": _json_number(float(np.min(real / scale))),
        "material_negative_real_admittance": bool(np.any(materially_negative)),
        "interpretation": (
            "finite-band nonpassive signal; not a universal passivity certificate"
            if bool(np.any(materially_negative))
            else "no material negative-real signal in measured band; passivity is not certified"
        ),
    }


def analyze(payload: Mapping[str, Any], *, alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """Validate and analyse a measurement JSON object.

    The function never reads optional truth/case/scenario fields.  Input
    validation raises ``PassiveResponseError``; numerical fit failure returns
    a structured indeterminate/FAIL document rather than a false PASS.
    """

    measurement = parse_input(payload)
    try:
        sweep_fit = _fit(measurement, include_endpoints=False, method="sweep_only_baseline")
        constrained_fit = _fit(measurement, include_endpoints=True, method="constrained_positive_real_series_rlc")
    except NumericalFitError as exc:
        # Keep the direct API fail-closed as well as the CLI: a numerical
        # failure is an indeterminate result, never a family/physics reject.
        return _error_document(exc)
    if not sweep_fit.success or not constrained_fit.success:
        return _error_document(NumericalFitError("fit result was not marked as a converged finite candidate"))
    baseline_threshold = _chi2_threshold(sweep_fit.dof, alpha=alpha)
    constrained_threshold = _chi2_threshold(constrained_fit.dof, alpha=alpha)
    baseline_json = _fit_to_json(measurement, sweep_fit, baseline_threshold, alpha)
    constrained_json = _fit_to_json(measurement, constrained_fit, constrained_threshold, alpha)
    nonpassive = _passivity_diagnostic(measurement)
    baseline_pass = bool(baseline_json["pass_single_series_rlc"])
    constrained_pass = bool(constrained_json["pass_single_series_rlc"])
    if nonpassive["material_negative_real_admittance"]:
        classification = "nonpassive_signal_and_single_rlc_family_inconsistent"
    elif constrained_pass:
        classification = "single_rlc_consistent_with_declared_constraints"
    else:
        classification = "single_rlc_family_inconsistent_in_finite_band"
    return {
        "schema": RESULT_SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "input_contract": {
            "phasor_convention": PHASOR_CONVENTION,
            "imaginary_sign": IMAGINARY_SIGN,
            "frequency_unit": SUPPORTED_FREQUENCY_UNIT,
            "admittance_unit": SUPPORTED_ADMITTANCE_UNIT,
            "n_frequencies": measurement.n,
            "frequency_min_hz": _json_number(float(measurement.frequencies_hz[0])),
            "frequency_max_hz": _json_number(float(measurement.frequencies_hz[-1])),
            "independent_endpoint_constraints": True,
            "endpoint_constraint_source": measurement.constraint_source,
        },
        "model": {
            "family": "positive_real_series_RLC",
            "admittance": "Y(s)=1/(R+sL+1/(sC))",
            "laplace_axis": "s=-i*omega for exp(-i*omega*t)",
            "parameter_bounds": _jsonable(PARAMETER_BOUNDS),
            "reference_parameters_are_synthetic": True,
        },
        "sweep_only_baseline": baseline_json,
        "constrained_fit": constrained_json,
        "diagnostics": {
            "classification": classification,
            "baseline_pass": baseline_pass,
            "constrained_pass": constrained_pass,
            "passivity": nonpassive,
            "finite_band_unidentifiability": True,
            "unknown_positive_modes_outside_band_not_ruled_out": True,
            "absorbed_power_identity": "0.5*Re(V*conj(I))=0.5*|V|^2*Re(Y), phasor-choice invariant",
            "interpretation_limit": "failure rejects only this single-RLC family over this finite band (finite_band_unidentifiability remains); it does not prove active physics or nonexistence of other passive modes",
        },
    }


# ---------------------------------------------------------------------------
# Frozen deterministic synthetic benchmark.  Truth labels remain outside the
# JSON payload passed to ``analyze``; they are used only to aggregate results.
# ---------------------------------------------------------------------------

BENCHMARK_FREQUENCIES_HZ = tuple(float(x) for x in np.geomspace(20.0, 2.0e6, 64))
BENCHMARK_SIGMA_REAL_S = 2.0e-7
BENCHMARK_SIGMA_IMAG_S = 2.0e-7
BENCHMARK_SIGMA_C_F = 2.0e-9
BENCHMARK_SIGMA_L_H = 2.0e-5
BENCHMARK_DEVELOPMENT_REPLAY_SEEDS = tuple(range(2026092001, 2026092129))  # old holdout; no new claim
BENCHMARK_A1_TRAIN_SEEDS = tuple(range(2026093901, 2026093965))  # fresh 64-row calibration block
BENCHMARK_A1_HELDOUT_SEEDS = tuple(range(2026094001, 2026094129))  # fresh 128-row claim block
BENCHMARK_A1_FIXED_CASE_SEEDS = {
    "exact_reference": 2026095001,
    "nominal_noise": 2026095002,
    "false_polarity": 2026095003,
    "phase_fault": 2026095004,
    "extra_positive_branch": 2026095005,
    "finite_band_indistinguishable": 2026095006,
    "active_nonpassive_control": 2026095007,
}


def _complex_points(values: np.ndarray) -> list[dict[str, float]]:
    return [{"real_S": float(x.real), "imag_S": float(x.imag)} for x in values]


def _branch_admittance(frequencies_hz: np.ndarray, R: float, L: float, C: float) -> np.ndarray:
    if R > 0.0:
        return series_rlc_admittance(frequencies_hz, R, L, C)
    # Only the synthetic negative-R control uses this raw expression.  The
    # public model evaluator remains positive-real and rejects nonpositive R.
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    s = -1j * 2.0 * math.pi * frequencies_hz
    return 1.0 / (R + s * L + 1.0 / (s * C))


def _synthetic_truth(case: str, frequencies_hz: np.ndarray) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    """Generate declared synthetic truth and endpoint totals for one case."""

    base = _branch_admittance(frequencies_hz, REFERENCE_R, REFERENCE_L, REFERENCE_C)
    if case == "exact_reference" or case == "nominal_noise":
        return base, REFERENCE_C, REFERENCE_L, {"family": "single_series_rlc"}
    if case == "false_polarity":
        return base, REFERENCE_C, REFERENCE_L, {"family": "single_series_rlc", "fault": "imaginary_polarity_reversal"}
    if case == "phase_fault":
        phase = math.radians(8.0)
        return base * np.exp(1j * phase), REFERENCE_C, REFERENCE_L, {"family": "single_series_rlc", "fault": "8_degree_phase_rotation"}
    if case == "extra_positive_branch":
        # Parallel positive branch: endpoints obey C_total=sum(C_i) and
        # 1/L_eff=sum(1/L_i), while its finite-band shape is not single-RLC.
        r2, l2, c2 = 120.0, 3.0e-3, 1.5e-7
        second = _branch_admittance(frequencies_hz, r2, l2, c2)
        return base + second, REFERENCE_C + c2, 1.0 / (1.0 / REFERENCE_L + 1.0 / l2), {
            "family": "parallel_two_positive_series_rlc_branches",
            "branch2": {"R_ohm": r2, "L_H": l2, "C_F": c2},
        }
    if case == "finite_band_indistinguishable":
        # A positive branch whose corner is deliberately outside most of the
        # declared window.  This is a control for finite-band identifiability,
        # not evidence that the unseen branch does not exist.
        # f0 ~= 8 MHz, beyond the 2 MHz upper edge; C2 is intentionally below
        # the public fit bound because this is a hidden-mode control, not a
        # candidate single-RLC parameter.
        r2, l2, c2 = 3.0e3, 2.0e2, 2.0e-18
        second = _branch_admittance(frequencies_hz, r2, l2, c2)
        return base + second, REFERENCE_C + c2, 1.0 / (1.0 / REFERENCE_L + 1.0 / l2), {
            "family": "parallel_two_positive_series_rlc_branches",
            "branch2": {"R_ohm": r2, "L_H": l2, "C_F": c2},
        }
    if case == "active_nonpassive_control":
        active = _branch_admittance(frequencies_hz, -3.0, REFERENCE_L, REFERENCE_C)
        return active, REFERENCE_C, REFERENCE_L, {"family": "negative_R_control", "fault": "active_nonpassive"}
    raise PassiveResponseError(f"unknown synthetic benchmark case: {case}")


def _make_payload(case: str, seed: int, *, noisy: bool, exact: bool = False) -> dict[str, Any]:
    frequencies = np.asarray(BENCHMARK_FREQUENCIES_HZ, dtype=float)
    truth, endpoint_c, endpoint_l, _meta = _synthetic_truth(case, frequencies)
    rng = np.random.default_rng(seed)
    if noisy and not exact:
        measured = truth + rng.normal(0.0, BENCHMARK_SIGMA_REAL_S, size=truth.size) + 1j * rng.normal(
            0.0, BENCHMARK_SIGMA_IMAG_S, size=truth.size
        )
        c_obs = endpoint_c + rng.normal(0.0, BENCHMARK_SIGMA_C_F)
        l_obs = endpoint_l + rng.normal(0.0, BENCHMARK_SIGMA_L_H)
    else:
        measured = truth.copy()
        c_obs = endpoint_c
        l_obs = endpoint_l
    if case == "false_polarity":
        measured = measured.real - 1j * measured.imag
    # For exact-reference output, nonzero declared sigmas still express the
    # instrument model; no noise is injected into the data.
    return {
        "schema": SCHEMA,
        "units": {
            "frequency": "Hz",
            "admittance": "S",
            "phasor_convention": PHASOR_CONVENTION,
            "imaginary_sign": IMAGINARY_SIGN,
        },
        "measurement": {
            "frequencies_hz": [float(x) for x in frequencies],
            "admittance": _complex_points(measured),
        },
        "noise": {
            "model": "independent_gaussian_cartesian",
            "sigma_real_S": BENCHMARK_SIGMA_REAL_S,
            "sigma_imag_S": BENCHMARK_SIGMA_IMAG_S,
        },
        "endpoint_constraints": {
            "low_frequency_capacitance_F": {"value": float(c_obs), "sigma_F": BENCHMARK_SIGMA_C_F, "unit": "F"},
            "high_frequency_inductance_H": {"value": float(l_obs), "sigma_H": BENCHMARK_SIGMA_L_H, "unit": "H"},
        },
        "constraint_metadata": {
            "independent": True,
            "source": "synthetic_separate_low_frequency_C_and_high_frequency_L_measurements",
        },
    }


def _wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        raise PassiveResponseError("Wilson interval requires n>0")
    p = float(k) / float(n)
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _summarize_nominal(rows: Sequence[dict[str, Any]], method_key: str) -> dict[str, Any]:
    valid_rows = [row for row in rows if method_key in row]
    flags = [not bool(row[method_key]["pass_single_series_rlc"]) for row in valid_rows]
    false_alarms = int(sum(flags))
    n = len(rows)
    valid_n = len(valid_rows)
    low, high = _wilson_interval(false_alarms, n) if n else (0.0, 0.0)
    return {
        "n": n,
        "valid_rows": valid_n,
        "numerical_failure_count": n - valid_n,
        "false_alarm_count": false_alarms,
        "false_alarm_rate": _json_number(false_alarms / n) if n else 0.0,
        "false_alarm_rate_95_wilson": [_json_number(low), _json_number(high)],
    }


def _summarize_parameter_coverage(rows: Sequence[dict[str, Any]], method_key: str) -> dict[str, Any]:
    """Describe nominal interval containment without claiming universal coverage."""

    parameters = {"R_ohm": REFERENCE_R, "L_H": REFERENCE_L, "C_F": REFERENCE_C}
    summary: dict[str, Any] = {}
    for name, truth in parameters.items():
        valid = 0
        covered = 0
        invalid = 0
        for row in rows:
            fit = row.get(method_key)
            if not isinstance(fit, Mapping):
                invalid += 1
                continue
            estimate = fit.get("parameters", {}).get(name)
            interval = fit.get("conditional_parameter_intervals_95", {}).get(name, {})
            low = interval.get("low_95")
            high = interval.get("high_95")
            if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (estimate, low, high)):
                invalid += 1
                continue
            valid += 1
            covered += int(float(low) <= truth <= float(high))
        summary[name] = {
            "truth": truth,
            "n": len(rows),
            "valid_intervals": valid,
            "covered": covered,
            "coverage_rate": _json_number(covered / valid) if valid else None,
            "invalid_interval_count": invalid,
            "interpretation": "synthetic nominal containment diagnostic only; not a universal coverage claim",
        }
    return summary


def _summarize_scores(rows: Sequence[dict[str, Any]], method_key: str) -> dict[str, Any]:
    valid = [float(row[method_key]["score"]) for row in rows if method_key in row]
    if not valid:
        return {"valid_rows": 0, "numerical_failure_count": len(rows), "min": None, "max": None, "median": None}
    return {
        "valid_rows": len(valid),
        "numerical_failure_count": len(rows) - len(valid),
        "min": _json_number(min(valid)),
        "max": _json_number(max(valid)),
        "median": _json_number(float(np.median(valid))),
    }


def _benchmark_analyze(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Retain numerical failures as structured rows instead of aborting runs."""

    try:
        return analyze(payload)
    except NumericalFitError as exc:
        return _error_document(exc)


def run_benchmark() -> dict[str, Any]:
    """Execute the pre-registered deterministic benchmark.

    The protocol is intentionally represented in the returned artifact as a
    compact audit trail.  ``evidence_A1/PROTOCOL.md`` is the authoritative
    human-readable freeze and is written before this corrected benchmark run.
    """

    protocol = {
        "frequency_window_hz": [20.0, 2.0e6],
        "sampling": "64 logarithmically spaced frequencies",
        "noise_model": "independent Gaussian Cartesian real/imag with sigma=2e-7 S; independent endpoint Gaussian sigma_C=2e-9 F, sigma_L=2e-5 H",
        "threshold_algorithm": "fixed chi-square 0.99 upper quantile with actual objective dof: sweep=2N-3, joint=2N+2-3; no evaluation-label tuning",
        "parameter_bounds": copy.deepcopy(PARAMETER_BOUNDS),
        "train_test_split": {"a1_train_nominal": 64, "a1_heldout_nominal": 128, "heldout_required_minimum": 64},
        "fixed_cases": list(BENCHMARK_A1_FIXED_CASE_SEEDS),
        "seed_ranges": {
            "development_replay_old_holdout": [BENCHMARK_DEVELOPMENT_REPLAY_SEEDS[0], BENCHMARK_DEVELOPMENT_REPLAY_SEEDS[-1]],
            "a1_train": [BENCHMARK_A1_TRAIN_SEEDS[0], BENCHMARK_A1_TRAIN_SEEDS[-1]],
            "a1_heldout": [BENCHMARK_A1_HELDOUT_SEEDS[0], BENCHMARK_A1_HELDOUT_SEEDS[-1]],
            "a1_fixed_cases": [min(BENCHMARK_A1_FIXED_CASE_SEEDS.values()), max(BENCHMARK_A1_FIXED_CASE_SEEDS.values())],
        },
        "reference": {"R_ohm": REFERENCE_R, "L_H": REFERENCE_L, "C_F": REFERENCE_C, "synthetic_only": True},
        "comparison": "sweep-only positive-real series-RLC fit plus endpoint score versus joint constrained positive-real series-RLC fit; no general-rational oracle",
    }
    development_replay_rows: list[dict[str, Any]] = []
    for seed in BENCHMARK_DEVELOPMENT_REPLAY_SEEDS:
        payload = _make_payload("nominal_noise", seed, noisy=True)
        result = _benchmark_analyze(payload)
        development_replay_rows.append(result)
    train_rows: list[dict[str, Any]] = []
    for seed in BENCHMARK_A1_TRAIN_SEEDS:
        payload = _make_payload("nominal_noise", seed, noisy=True)
        result = _benchmark_analyze(payload)
        train_rows.append(result)
    heldout_rows: list[dict[str, Any]] = []
    for seed in BENCHMARK_A1_HELDOUT_SEEDS:
        payload = _make_payload("nominal_noise", seed, noisy=True)
        result = _benchmark_analyze(payload)
        heldout_rows.append(result)
    fixed_rows: list[dict[str, Any]] = []
    for case, seed in BENCHMARK_A1_FIXED_CASE_SEEDS.items():
        result = _benchmark_analyze(_make_payload(case, seed, noisy=(case != "exact_reference"), exact=(case == "exact_reference")))
        # The case label is aggregation metadata, never an algorithm input.
        row = {"case": case, "seed": seed, "status": result["status"]}
        if "decision" in result:
            row.update({"decision": result["decision"], "error_type": result.get("error_type"), "error": result.get("error")})
        else:
            row.update(
                {
                    "classification": result["diagnostics"]["classification"],
                    "sweep_only_baseline": result["sweep_only_baseline"],
                    "constrained_fit": result["constrained_fit"],
                    "diagnostics": result["diagnostics"],
                }
            )
        fixed_rows.append(row)
    baseline_summary = _summarize_nominal(heldout_rows, "sweep_only_baseline")
    constrained_summary = _summarize_nominal(heldout_rows, "constrained_fit")
    baseline_false = baseline_summary["false_alarm_count"]
    constrained_false = constrained_summary["false_alarm_count"]
    if constrained_false < baseline_false:
        comparison = "constrained_fit_has_lower_heldout_false_alarm_count"
    elif constrained_false > baseline_false:
        comparison = "sweep_only_baseline_has_lower_heldout_false_alarm_count"
    else:
        comparison = "no_heldout_false_alarm_advantage_for_constrained_fit"
    return {
        "schema": BENCHMARK_SCHEMA,
        "status": "COMPLETED_FROZEN_PASSIVE_RLC_BENCHMARK",
        "evidence_weight": EVIDENCE_WEIGHT,
        "protocol": protocol,
        "train_nominal": {
            "n": len(train_rows),
            "baseline_score_summary": _summarize_scores(train_rows, "sweep_only_baseline"),
            "constrained_score_summary": _summarize_scores(train_rows, "constrained_fit"),
        },
        "development_replay_old_holdout": {
            "seed_range": [BENCHMARK_DEVELOPMENT_REPLAY_SEEDS[0], BENCHMARK_DEVELOPMENT_REPLAY_SEEDS[-1]],
            "no_new_operating_claim": True,
            "baseline": _summarize_nominal(development_replay_rows, "sweep_only_baseline"),
            "constrained_fit": _summarize_nominal(development_replay_rows, "constrained_fit"),
            "baseline_parameter_interval_coverage": _summarize_parameter_coverage(development_replay_rows, "sweep_only_baseline"),
            "constrained_parameter_interval_coverage": _summarize_parameter_coverage(development_replay_rows, "constrained_fit"),
            "rows": development_replay_rows,
        },
        "heldout_nominal": {
            "baseline": baseline_summary,
            "constrained_fit": constrained_summary,
            "baseline_parameter_interval_coverage": _summarize_parameter_coverage(heldout_rows, "sweep_only_baseline"),
            "constrained_parameter_interval_coverage": _summarize_parameter_coverage(heldout_rows, "constrained_fit"),
            # Keep every held-out row, not only the aggregate false-alarm
            # count, so a later read-only acceptance can audit all failures
            # without regenerating the benchmark.
            "rows": heldout_rows,
            "all_rows_recorded": True,
        },
        "fixed_cases": fixed_rows,
        "comparison": {
            "verdict": comparison,
            "scientific_advantage_claimed": False,
            "reason": "The benchmark is an engineering operating-characteristic comparison; no universal passive certification or physical inference is claimed.",
        },
        "failure_inventory": {
            "negative_invalid_json_unit_polarity_scaling_controls": True,
            "genuine_heldout_nominal_coverage": len(heldout_rows),
            "new_seed_block_disjoint_from_old": True,
            "old_holdout_replayed_as_development_only": len(development_replay_rows),
            "fixed_case_numerical_failure_count": sum(1 for row in fixed_rows if row["status"] == "FAIL"),
            "finite_band_unidentifiability_reported": True,
            "active_nonpassive_control_included": True,
        },
    }


def _error_document(exc: Exception) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "status": "FAIL",
        "decision": "INDETERMINATE_NUMERICAL_FIT" if isinstance(exc, NumericalFitError) else "INVALID_OR_NUMERIC_INPUT",
        "evidence_weight": EVIDENCE_WEIGHT,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(document), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="measurement JSON input")
    parser.add_argument("--benchmark", action="store_true", help="run the frozen deterministic synthetic benchmark")
    parser.add_argument("--output", type=Path, help="explicit JSON output path")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="fixed upper-tail alpha (default: 0.01)")
    args = parser.parse_args(argv)
    if bool(args.input) == bool(args.benchmark):
        parser.error("choose exactly one of --input or --benchmark")
    try:
        if args.benchmark:
            document = run_benchmark()
        else:
            assert args.input is not None
            document = analyze(_read_json(args.input), alpha=args.alpha)
        if args.output is not None:
            _write_json(args.output, document)
        sys.stdout.write(json.dumps(_jsonable(document), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        return 2 if document.get("status") == "FAIL" else 0
    except (PassiveResponseError, OSError, ValueError, TypeError, AttributeError, OverflowError, FloatingPointError) as exc:
        document = _error_document(exc)
        if args.output is not None:
            try:
                _write_json(args.output, document)
            except (OSError, PassiveResponseError, ValueError, TypeError, AttributeError, OverflowError, FloatingPointError):
                pass
        sys.stdout.write(json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
