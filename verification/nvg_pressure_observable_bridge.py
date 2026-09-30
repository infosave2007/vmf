#!/usr/bin/env python3
"""Live reduced-pressure observable bridge for the accepted W8/U16 pair.

The module reuses the live cold homogeneous BulkModel and the existing W8/U16
calibration constructor.  It computes a reduced-pressure four-point contrast
that removes constant, slope, and curvature nuisance terms in the reduced
pressure F=P/[n0(1+x)^2].  This is a prospective model-discrimination budget,
not an experimental result or a finite-temperature/finite-nucleus observable.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_nonlinear_calibration_response as nonlinear  # noqa: E402

SCHEMA_VERSION = "nvg_pressure_observable_bridge.v1"
STATUS_PASS = "PASS_LIVE_REDUCED_PRESSURE_BRIDGE"
STATUS_FAIL = "FAIL_LIVE_REDUCED_PRESSURE_BRIDGE"
EVIDENCE_WEIGHT = mp.mpf("0")
ANCHORS = ("0.90", "0.93")
MODEL_ORDER = ("w8", "u16")
MODEL_AMPLITUDES = {"w8": 0, "u16": 1}
X_NODES = ("-0.20", "-0.10", "-0.05", "-0.025", "0", "0.025", "0.05", "0.10", "0.20")
H_VALUES = ("0.10", "0.05", "0.025")
CONTRAST_X_MULTIPLIERS = (-2, -1, 1, 2)
CONTRAST_COEFFICIENTS = (-1, 2, -2, 1)
N0_FM3 = "0.16"
BINDING_MEV = "-16"
K_TARGET_MEV = "240"
HBARC_MEV_FM = "197.3269804"
PRIMARY_DPS = 90
CONTROL_DPS = 120
PRECISION_REL_LIMIT = mp.mpf("1e-45")
ZERO_SCALE = mp.mpf("1e-100")
STATE_REL_LIMIT = mp.mpf("1e-60")
BRANCH_LOW_RATIO = "0.80"
BRANCH_HIGH_RATIO = "1.20"
GAUSSIAN_ALPHA = "0.05"
GAUSSIAN_POWER = "0.80"
PRESSURE_UNITS = "MeV fm^-3"
DENSITY_UNITS = "fm^-3"
COVARIANCE_UNITS = "(MeV fm^-3)^2"
PROTOCOL_VERSION = "reduced-pressure-bridge-public-v1"
PROTOCOL_PROVENANCE = "ADOPTED_AFTER_PILOT_NOT_INDEPENDENTLY_TIMESTAMPED_PREREGISTRATION"

# This compact dictionary is the public protocol.  It intentionally contains
# declared inputs and controls, never saved scientific rows.
PROTOCOL = {
    "protocol_version": PROTOCOL_VERSION,
    "anchors": ANCHORS,
    "deformation_amplitudes": (0, 1),
    "x_nodes": X_NODES,
    "contrast_h": H_VALUES,
    "n0_fm3": N0_FM3,
    "binding_MeV": BINDING_MEV,
    "K_target_MeV": K_TARGET_MEV,
    "primary_dps": PRIMARY_DPS,
    "control_dps": CONTROL_DPS,
    "branch_ratio": ("0.80", "1.20"),
    "physical_inputs_fixed": True,
    "protocol_frozen_before_rows": False,
}


class PressureBridgeError(ValueError):
    """Invalid observable input or failed live scientific control."""


def _mp(value: Any) -> mp.mpf:
    if isinstance(value, mp.mpf):
        return value
    if isinstance(value, bool):
        raise PressureBridgeError("boolean is not a finite scalar")
    try:
        return mp.mpf(str(value))
    except (TypeError, ValueError) as exc:
        raise PressureBridgeError("value must be a finite real scalar") from exc


def _finite(value: Any) -> bool:
    try:
        return bool(mp.isfinite(_mp(value)))
    except (TypeError, ValueError, PressureBridgeError):
        return False


def _number(value: Any, digits: int = 90) -> str:
    value = _mp(value)
    if not mp.isfinite(value):
        raise PressureBridgeError("scientific output is nonfinite")
    return mp.nstr(value, digits)


def _relative(a: Any, b: Any, *, zero_scale: mp.mpf = ZERO_SCALE) -> mp.mpf:
    """Relative difference on the actual physical scale, never max(1, ...)."""
    aa, bb = _mp(a), _mp(b)
    return abs(aa - bb) / max(abs(aa), abs(bb), zero_scale)


def _relative_to_expected(given: Any, expected: Any, *, zero_tol: mp.mpf = ZERO_SCALE) -> bool:
    g, e = _mp(given), _mp(expected)
    if e == 0:
        return abs(g) <= zero_tol
    return abs(g - e) / abs(e) <= mp.mpf("1e-40")


def _jsonable(value: Any) -> Any:
    if isinstance(value, mp.mpf):
        return _number(value)
    if isinstance(value, mp.mpc):
        raise PressureBridgeError("complex scientific output")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _validate_units(pressure_units: Any, density_units: Any, covariance_units: Any | None) -> None:
    if pressure_units != PRESSURE_UNITS:
        raise PressureBridgeError(f"pressure_units must be exactly {PRESSURE_UNITS!r}")
    if density_units != DENSITY_UNITS:
        raise PressureBridgeError(f"density_units must be exactly {DENSITY_UNITS!r}")
    if covariance_units is not None and covariance_units != COVARIANCE_UNITS:
        raise PressureBridgeError(f"covariance_units must be exactly {COVARIANCE_UNITS!r}")


def _validate_contrast_geometry(n0_fm3: Any, h: Any, x_nodes: Sequence[Any] | None) -> tuple[mp.mpf, mp.mpf, tuple[mp.mpf, ...]]:
    n0 = _mp(n0_fm3)
    step = _mp(h)
    if not mp.isfinite(n0) or n0 <= 0:
        raise PressureBridgeError("n0_fm3 must be positive and finite")
    if not mp.isfinite(step) or step <= 0:
        raise PressureBridgeError("h must be positive and finite")
    if x_nodes is None:
        nodes = tuple(-step * 2 if multiplier == -2 else
                      -step if multiplier == -1 else
                      step if multiplier == 1 else 2 * step
                      for multiplier in CONTRAST_X_MULTIPLIERS)
    else:
        if isinstance(x_nodes, (str, bytes)) or len(x_nodes) != 4:
            raise PressureBridgeError("x_nodes must contain exactly four ordered nodes")
        nodes = tuple(_mp(value) for value in x_nodes)
    expected = tuple(step * multiplier for multiplier in CONTRAST_X_MULTIPLIERS)
    if any(not mp.isfinite(value) for value in nodes):
        raise PressureBridgeError("x_nodes must be finite")
    if any(nodes[index] != expected[index] for index in range(4)):
        raise PressureBridgeError("x_nodes must be ordered as (-2h,-h,+h,+2h)")
    if any(1 + value <= 0 for value in nodes):
        raise PressureBridgeError("contrast density ratios 1+x must be positive")
    return n0, step, nodes


def _validate_pressures(pressures: Sequence[Any]) -> tuple[mp.mpf, ...]:
    if isinstance(pressures, (str, bytes)) or len(pressures) != 4:
        raise PressureBridgeError("pressures must contain exactly four ordered values")
    values = tuple(_mp(value) for value in pressures)
    if any(not mp.isfinite(value) for value in values):
        raise PressureBridgeError("pressures must be finite")
    return values


def _validate_covariance(covariance: Sequence[Sequence[Any]] | None) -> mp.matrix | None:
    if covariance is None:
        return None
    if isinstance(covariance, (str, bytes)) or len(covariance) != 4:
        raise PressureBridgeError("covariance must be a 4x4 symmetric PSD matrix")
    rows = []
    for row in covariance:
        if isinstance(row, (str, bytes)) or len(row) != 4:
            raise PressureBridgeError("covariance must be a 4x4 symmetric PSD matrix")
        rows.append([_mp(value) for value in row])
    matrix = mp.matrix(rows)
    if any(not mp.isfinite(matrix[i, j]) for i in range(4) for j in range(4)):
        raise PressureBridgeError("covariance entries must be finite")
    scale = max(max(abs(matrix[i, j]) for j in range(4)) for i in range(4))
    # Covariance positivity is invariant under multiplication by a positive
    # scalar.  Normalize before symmetry/eigenvalue checks so tiny physical
    # covariances (or very large unit-scaled ones) do not inherit an absolute
    # tolerance or variance floor.  Keep the original matrix for the returned
    # uncertainty in the caller's units.
    if scale == 0:
        return matrix
    normalized = matrix / scale
    symmetry_tol = mp.mpf("1e-40")
    for i in range(4):
        for j in range(i + 1, 4):
            if abs(normalized[i, j] - normalized[j, i]) > symmetry_tol:
                raise PressureBridgeError("covariance must be symmetric")
    # Eigenvalues are the direct PSD predicate; a tiny scale-relative negative
    # is accepted only as high-precision roundoff, never a materially negative mode.
    eigenvalues, _ = mp.eigsy(normalized)
    psd_tol = mp.mpf("1e-40")
    if min(eigenvalues) < -psd_tol:
        raise PressureBridgeError("covariance must be positive semidefinite")
    return matrix


def _pressure_contrast_impl(
    pressures: Sequence[Any],
    *,
    n0_fm3: Any = N0_FM3,
    h: Any = "0.10",
    x_nodes: Sequence[Any] | None = None,
    covariance: Sequence[Sequence[Any]] | None = None,
    pressure_units: str = PRESSURE_UNITS,
    density_units: str = DENSITY_UNITS,
    covariance_units: str | None = COVARIANCE_UNITS,
) -> dict[str, Any]:
    """Return the reduced-pressure third-jet contrast and optional uncertainty.

    Four physical pressures in MeV fm^-3 are required in the exact order
    ``(-2h,-h,+h,+2h)``.  The returned ``Z_h`` is in MeV, and covariance is in
    ``(MeV fm^-3)^2``.  The function intentionally has no model/truth input.
    """
    _validate_units(pressure_units, density_units, covariance_units)
    if covariance is not None and covariance_units != COVARIANCE_UNITS:
        raise PressureBridgeError("supplied covariance requires covariance_units='(MeV fm^-3)^2'")
    values = _validate_pressures(pressures)
    n0, step, nodes = _validate_contrast_geometry(n0_fm3, h, x_nodes)
    covariance_matrix = _validate_covariance(covariance)
    coefficients = tuple(_mp(value) for value in CONTRAST_COEFFICIENTS)
    weights = tuple(
        mp.mpf(81) * coefficient / (2 * step**3 * n0 * (1 + node) ** 2)
        for coefficient, node in zip(coefficients, nodes)
    )
    reduced = tuple(value / (n0 * (1 + node) ** 2) for value, node in zip(values, nodes))
    z_h = mp.mpf(81) * sum(coefficient * value for coefficient, value in zip(coefficients, reduced)) / (2 * step**3)
    result: dict[str, Any] = {
        "n0_fm3": n0,
        "h": step,
        "x_nodes": nodes,
        "contrast_coefficients": coefficients,
        "pressures_MeV_fm3": values,
        "reduced_pressures_MeV": reduced,
        "weights_on_pressure_fm3": weights,
        "weight_norm_fm3": mp.sqrt(sum(weight * weight for weight in weights)),
        "Z_h_MeV": z_h,
        "pressure_units": pressure_units,
        "density_units": density_units,
        "covariance_units": covariance_units,
        "covariance_supplied": covariance_matrix is not None,
    }
    if covariance_matrix is None:
        result["variance_Z_MeV2"] = None
        result["sigma_Z_MeV"] = None
    else:
        weight_vector = mp.matrix(weights)
        variance = (weight_vector.T * covariance_matrix * weight_vector)[0]
        covariance_scale = max(
            abs(covariance_matrix[i, j]) for i in range(4) for j in range(4)
        )
        weight_norm_squared = sum(weight * weight for weight in weights)
        contraction_scale = covariance_scale * weight_norm_squared
        variance_tol = mp.mpf("1e-40") * contraction_scale
        if variance < 0 and abs(variance) <= variance_tol:
            variance = mp.mpf(0)
        if variance < 0:
            raise PressureBridgeError("covariance contraction is negative")
        result["variance_Z_MeV2"] = variance
        result["sigma_Z_MeV"] = mp.sqrt(variance)
    return result


def pressure_contrast(
    pressures: Sequence[Any],
    *,
    n0_fm3: Any = N0_FM3,
    h: Any = "0.10",
    x_nodes: Sequence[Any] | None = None,
    covariance: Sequence[Sequence[Any]] | None = None,
    pressure_units: str = PRESSURE_UNITS,
    density_units: str = DENSITY_UNITS,
    covariance_units: str | None = COVARIANCE_UNITS,
) -> dict[str, Any]:
    """Self-contained reduced-pressure contrast with a precision floor.

    Decimal-string inputs, especially rank-one covariance matrices, are parsed
    inside a local context of at least 80 digits rather than at ambient
    double-like mpmath precision.  A caller using higher precision keeps it.
    """
    local_dps = max(int(mp.mp.dps), 80)
    with mp.workdps(local_dps):
        return _pressure_contrast_impl(
            pressures,
            n0_fm3=n0_fm3,
            h=h,
            x_nodes=x_nodes,
            covariance=covariance,
            pressure_units=pressure_units,
            density_units=density_units,
            covariance_units=covariance_units,
        )


def standard_normal_cdf(value: Any) -> mp.mpf:
    z = _mp(value)
    if not mp.isfinite(z):
        raise PressureBridgeError("normal CDF input must be finite")
    return (1 + mp.erf(z / mp.sqrt(2))) / 2


def standard_normal_quantile(probability: Any) -> mp.mpf:
    probability = _mp(probability)
    if not mp.isfinite(probability) or not 0 < probability < 1:
        raise PressureBridgeError("normal quantile probability must lie strictly in (0,1)")
    return mp.sqrt(2) * mp.erfinv(2 * probability - 1)


def gaussian_discrimination_budget(
    delta_z_h_MeV: Any,
    weight_norm_fm3: Any,
    *,
    alpha: Any = GAUSSIAN_ALPHA,
    power: Any = GAUSSIAN_POWER,
    orientation: str = "predeclared_one_sided",
) -> dict[str, Any]:
    """Compute prospective one-sided Gaussian mean-separation sensitivity.

    ``sigmaP_max`` is the equal-independent pressure error per point in one
    four-point scan for false-alarm alpha and power.  It compares one data set
    to fixed theoretical means, so there is deliberately no sqrt(2) factor.
    """
    delta = _mp(delta_z_h_MeV)
    norm = _mp(weight_norm_fm3)
    alpha = _mp(alpha)
    power = _mp(power)
    if not mp.isfinite(delta):
        raise PressureBridgeError("delta_z_h_MeV must be finite")
    if not mp.isfinite(norm) or norm <= 0:
        raise PressureBridgeError("weight_norm_fm3 must be positive and finite")
    if not mp.isfinite(alpha) or not 0 < alpha < 1:
        raise PressureBridgeError("alpha must lie strictly in (0,1)")
    if not mp.isfinite(power) or not 0 < power < 1:
        raise PressureBridgeError("power must lie strictly in (0,1)")
    if orientation != "predeclared_one_sided":
        raise PressureBridgeError("only the predeclared one-sided orientation is supported")
    if delta == 0:
        raise PressureBridgeError(
            "zero model separation is unidentifiable; no finite pressure error attains requested power"
        )
    q_alpha = standard_normal_quantile(1 - alpha)
    q_power = standard_normal_quantile(power)
    factor = q_alpha + q_power
    if factor <= 0:
        raise PressureBridgeError("Gaussian threshold factor must be positive")
    sigma = abs(delta) / (norm * factor)
    return {
        "false_alarm_alpha": alpha,
        "power": power,
        "orientation": orientation,
        "quantile_false_alarm": q_alpha,
        "quantile_power": q_power,
        "mean_separation_factor": factor,
        "delta_Z_h_MeV": delta,
        "weight_norm_fm3": norm,
        "sigmaP_max_MeV_fm3": sigma,
        "sqrt2_for_two_data_sets": False,
        "interpretation": "prospective equal-independent Gaussian pressure error; not observational significance",
    }


def _n0_from_model(model: Any) -> mp.mpf:
    n0 = model.n0 / model.hbarc**3
    if _relative(n0, _mp(N0_FM3)) > mp.mpf("1e-70"):
        raise PressureBridgeError("live model n0 conversion disagrees with declared n0_fm3")
    return n0


def _pressure_row(model: Any, target_y: mp.mpf, x: mp.mpf) -> dict[str, Any]:
    q = 1 + x
    branch_low = _mp(BRANCH_LOW_RATIO)
    branch_high = _mp(BRANCH_HIGH_RATIO)
    if not branch_low <= q <= branch_high:
        raise PressureBridgeError("requested density leaves the safe positive branch")
    n = model.n0 * q
    y, state, controls = nonlinear._stationary_root(model, n, target_y)
    pressure_nat = state["n"] * state["mu"] - state["energy_total"]
    pressure_from_state = state["pressure_total"]
    # The identity is checked against the natural energy scale.  Comparing
    # two nearly-zero cancellation residues at x=0 would manufacture a large
    # relative error and is not a physical separation scale.
    pressure_identity = abs(pressure_nat - pressure_from_state) / max(
        abs(state["n"] * state["mu"]), abs(state["energy_total"]), mp.mpf(1)
    )
    if pressure_identity > STATE_REL_LIMIT:
        raise PressureBridgeError("P=n*mu-epsilon identity failed")
    if not (y > 0 and state["C_y"] > 0 and state["mu_prime"] > 0):
        raise PressureBridgeError("stationary row is outside positive local branch")
    if controls["stationarity_relative"] > STATE_REL_LIMIT:
        raise PressureBridgeError("stationarity residual exceeds bridge limit")
    n0_fm3 = _n0_from_model(model)
    pressure_fm3 = pressure_nat / model.hbarc**3
    reduced = pressure_fm3 / (n0_fm3 * q**2)
    return {
        "x": x,
        "density_ratio": q,
        "density_fm3": n0_fm3 * q,
        "n_natural_MeV3": n,
        "y": y,
        "pressure_natural_MeV4": pressure_nat,
        "pressure_MeV_fm3": pressure_fm3,
        "reduced_pressure_MeV": reduced,
        "pressure_identity_relative": pressure_identity,
        "stationarity_relative": controls["stationarity_relative"],
        "C_y": state["C_y"],
        "mu_prime": state["mu_prime"],
        "positive_local_branch": True,
        "branch_ratio_window": (branch_low, branch_high),
    }


def _snapshot(dps: int) -> dict[str, Any]:
    if dps < 80:
        raise PressureBridgeError("bridge precision must be at least 80 digits")
    with mp.workdps(dps):
        anchors: dict[str, Any] = {}
        x_values = tuple(_mp(value) for value in X_NODES)
        h_values = tuple(_mp(value) for value in H_VALUES)
        for target in ANCHORS:
            target_y = _mp(target)
            infos = nonlinear._build_models(target)
            model_rows: dict[str, Any] = {}
            derivatives: dict[str, Any] = {}
            n0_fm3 = None
            for name in MODEL_ORDER:
                model = infos[name]["model"]
                n0_fm3 = _n0_from_model(model)
                rows = []
                for x in x_values:
                    try:
                        rows.append(_pressure_row(model, target_y, x))
                    except Exception as exc:
                        # Keep every declared node, including an unfavorable
                        # or unresolved row, rather than silently dropping it.
                        rows.append({
                            "x": x,
                            "density_ratio": 1 + x,
                            "row_status": "FAIL",
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                derivative_error = None
                try:
                    deriv = nonlinear._derivatives(model, target_y)
                except Exception as exc:
                    deriv = None
                    derivative_error = f"{type(exc).__name__}: {exc}"
                derivatives[name] = deriv
                model_payload = {
                    "id": infos[name]["id"],
                    "amplitude": infos[name]["amplitude"],
                    "target_y": target_y,
                    "eta_MeV4": infos[name]["eta_MeV4"],
                    "rows": rows,
                    "Z_live_MeV": None if deriv is None else deriv["Z_MeV"],
                    "epsilon_derivatives": {} if deriv is None else deriv["epsilon_derivatives"],
                    "derivative_checks": {} if deriv is None else deriv["checks"],
                }
                if derivative_error is not None:
                    model_payload["derivative_error"] = derivative_error
                model_rows[name] = {
                    **model_payload,
                }
            contrasts: dict[str, Any] = {}
            for h in h_values:
                per_model: dict[str, Any] = {}
                for name in MODEL_ORDER:
                    model_payload = model_rows[name]
                    failed_rows = [row for row in model_payload["rows"] if row.get("row_status") == "FAIL"]
                    if model_payload.get("derivative_error") is not None or failed_rows:
                        per_model[name] = {
                            "status": "FAIL",
                            "h": h,
                            "failed_rows": failed_rows,
                            "error": model_payload.get("derivative_error") or "one or more declared pressure rows failed",
                        }
                        continue
                    rows_by_x = {row["x"]: row for row in model_rows[name]["rows"]}
                    contrast_x = tuple(-h * 2 if multiplier == -2 else
                                       -h if multiplier == -1 else
                                       h if multiplier == 1 else 2 * h
                                       for multiplier in CONTRAST_X_MULTIPLIERS)
                    pressures = tuple(rows_by_x[x]["pressure_MeV_fm3"] for x in contrast_x)
                    contrast = pressure_contrast(pressures, n0_fm3=n0_fm3, h=h)
                    live_z = derivatives[name]["Z_MeV"]
                    bias = contrast["Z_h_MeV"] - live_z
                    contrast.update({
                        "Z_live_MeV": live_z,
                        "taylor_bias_MeV": bias,
                        "taylor_bias_relative": abs(bias) / max(abs(live_z), ZERO_SCALE),
                        "contrast_node_rows": [rows_by_x[x] for x in contrast_x],
                    })
                    per_model[name] = contrast
                if any(model.get("status") == "FAIL" for model in per_model.values()):
                    contrasts[_number(h, 12)] = {
                        "h": h,
                        "status": "FAIL",
                        "models": per_model,
                        "error": "one or more declared model rows were retained as failures",
                    }
                    continue
                delta_z_h = per_model["u16"]["Z_h_MeV"] - per_model["w8"]["Z_h_MeV"]
                delta_z_live = per_model["u16"]["Z_live_MeV"] - per_model["w8"]["Z_live_MeV"]
                differential_bias = per_model["u16"]["taylor_bias_MeV"] - per_model["w8"]["taylor_bias_MeV"]
                budget = gaussian_discrimination_budget(
                    delta_z_h,
                    per_model["w8"]["weight_norm_fm3"],
                    alpha=GAUSSIAN_ALPHA,
                    power=GAUSSIAN_POWER,
                )
                contrasts[_number(h, 12)] = {
                    "h": h,
                    "models": per_model,
                    "delta_Z_h_MeV": delta_z_h,
                    "delta_Z_live_MeV": delta_z_live,
                    "differential_taylor_bias_MeV": differential_bias,
                    "delta_Z_h_minus_delta_Z_live_MeV": delta_z_h - delta_z_live,
                    "delta_Z_h_relative_to_live": abs(delta_z_h - delta_z_live) / max(abs(delta_z_live), ZERO_SCALE),
                    "gaussian_budget": budget,
                }
            # Exact finite-grid dependence of halving.  This is a diagnostic,
            # not a gate claiming asymptotic behavior at every finite h.
            ordered_h = tuple(_number(h, 12) for h in h_values)
            halving = []
            if all(contrasts[key].get("status", "PASS") == "PASS" for key in ordered_h):
                for previous, current in zip(ordered_h, ordered_h[1:]):
                    old_norm = contrasts[previous]["models"]["w8"]["weight_norm_fm3"]
                    new_norm = contrasts[current]["models"]["w8"]["weight_norm_fm3"]
                    old_bias = abs(contrasts[previous]["differential_taylor_bias_MeV"])
                    new_bias = abs(contrasts[current]["differential_taylor_bias_MeV"])
                    halving.append({
                        "from_h": contrasts[previous]["h"],
                        "to_h": contrasts[current]["h"],
                        "noise_norm_growth": new_norm / old_norm,
                        "differential_bias_ratio": new_bias / max(old_bias, ZERO_SCALE),
                        "asymptotic_note": "noise tends to grow ~8 and O(h^2) bias tends to fall ~4; exact (1+x)^-2 weights apply",
                    })
            anchors[target] = {
                "anchor_inputs": {
                    "target_y": target_y,
                    "n0_fm3": n0_fm3,
                    "binding_MeV": _mp(BINDING_MEV),
                    "K_target_MeV": _mp(K_TARGET_MEV),
                    "composition": "cold homogeneous fixed composition",
                },
                "models": model_rows,
                "contrasts": contrasts,
                "halving_diagnostics": halving,
                "branch": {
                    "density_ratio_window": (_mp(BRANCH_LOW_RATIO), _mp(BRANCH_HIGH_RATIO)),
                    "uniform_stability_claim_over_0.5_to_2": False,
                    "interpretation": "local smooth positive-curvature homogeneous branch only",
                },
            }
        return {"anchors": anchors}


def _precision_controls(primary: dict[str, Any], control: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for target in ANCHORS:
        for name in MODEL_ORDER:
            p = primary["anchors"][target]["models"][name]
            c = control["anchors"][target]["models"][name]
            fields: dict[str, tuple[Any, Any]] = {
                "Z_live_MeV": (p["Z_live_MeV"], c["Z_live_MeV"]),
                "epsilon_4": (p["epsilon_derivatives"]["epsilon_4"], c["epsilon_derivatives"]["epsilon_4"]),
            }
            for index, (pr, cr) in enumerate(zip(p["rows"], c["rows"])):
                # x=0 is an exact calibration pressure.  Its two high-
                # precision cancellation residues are checked by the explicit
                # zero bound below, not by a meaningless ratio of residues.
                if index == 4:
                    continue
                fields[f"pressure_{index}"] = (pr["pressure_MeV_fm3"], cr["pressure_MeV_fm3"])
                fields[f"reduced_pressure_{index}"] = (pr["reduced_pressure_MeV"], cr["reduced_pressure_MeV"])
            relative = {field: _relative(a, b, zero_scale=ZERO_SCALE) for field, (a, b) in fields.items()}
            maximum = max(relative.values())
            # Exact-zero pressure at x=0 uses an explicit zero bound, while
            # all nonzero fields use their actual magnitude as the scale.
            zero_row = primary["anchors"][target]["models"][name]["rows"][4]
            zero_control = control["anchors"][target]["models"][name]["rows"][4]
            zero_ok = (
                abs(_mp(zero_row["pressure_MeV_fm3"])) <= mp.mpf("1e-60")
                and abs(_mp(zero_control["pressure_MeV_fm3"])) <= mp.mpf("1e-60")
            )
            checks.append({
                "anchor": target,
                "model": name,
                "primary_dps": PRIMARY_DPS,
                "control_dps": CONTROL_DPS,
                "max_relative_difference": maximum,
                "fields": relative,
                "x0_pressure_zero_bound": mp.mpf("1e-60"),
                "x0_pressure_zero_pass": zero_ok,
                "pass": bool(maximum <= PRECISION_REL_LIMIT and zero_ok),
            })
    return checks


def _snapshot_has_failures(snapshot: dict[str, Any]) -> bool:
    for anchor in snapshot["anchors"].values():
        for model in anchor["models"].values():
            if model.get("derivative_error") is not None:
                return True
            if any(row.get("row_status") == "FAIL" for row in model["rows"]):
                return True
        if any(contrast.get("status") == "FAIL" for contrast in anchor["contrasts"].values()):
            return True
    return False


def _public_result(
    primary: dict[str, Any],
    control: dict[str, Any],
    precision: list[dict[str, Any]],
    *,
    status: str = STATUS_PASS,
) -> dict[str, Any]:
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    upstream_hash = hashlib.sha256(nonlinear.UPSTREAM_PATH.read_bytes()).hexdigest()
    protocol_hash = nonlinear._protocol_sha256()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "evidence_weight": EVIDENCE_WEIGHT,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_provenance": PROTOCOL_PROVENANCE,
        "protocol_sha256": protocol_hash,
        "source_sha256": source_hash,
        "upstream_source_sha256": upstream_hash,
        "physical_inputs_fixed": True,
        "protocol_frozen_before_rows": False,
        "units_and_conventions": {
            "pressure": PRESSURE_UNITS,
            "density": DENSITY_UNITS,
            "reduced_pressure": "MeV",
            "Z": "MeV",
            "natural_density": "n_natural=fm^-3*(hbar*c)^3 in MeV^3",
            "natural_pressure": "P_natural=MeV^4; P_fm3=P_natural/(hbar*c)^3",
            "hbarc_MeV_fm": HBARC_MEV_FM,
            "derivatives": "no factorial absorbed; Z=81*n0^4*d^4(epsilon/n)/dn^4",
        },
        "inputs": _jsonable(PROTOCOL),
        "gaussian_budget_defaults": {
            "false_alarm_alpha": GAUSSIAN_ALPHA,
            "power": GAUSSIAN_POWER,
            "orientation": "predeclared_one_sided",
            "sqrt2_for_two_data_sets": False,
            "interpretation": "prospective ideal equal-independent Gaussian error; not actual apparatus uncertainty",
        },
        "contrast_definition": {
            "x_nodes": tuple(_mp(value) for value in X_NODES),
            "h_values": tuple(_mp(value) for value in H_VALUES),
            "x_multipliers": CONTRAST_X_MULTIPLIERS,
            "coefficients": CONTRAST_COEFFICIENTS,
            "formula": "Z_h=81*sum(c_i*P_i/[n0_fm3*(1+x_i)^2])/(2*h^3)",
            "nuisance_removed": "constant, slope, curvature in reduced pressure F; not arbitrary EOS refit",
        },
        "anchors": primary["anchors"],
        "precision_controls": precision,
        "precision_control_summary": {
            "all_pass": all(item["pass"] for item in precision),
            "primary_dps": PRIMARY_DPS,
            "control_dps": CONTROL_DPS,
            "scale_rule": "actual-value/separation scaled; no max(1,value)",
        },
        "interpretation_limits": [
            "cold homogeneous fixed-composition local branch only",
            "prospective fixed-template pressure sensitivity, not an experimental comparison",
            "not finite-temperature baryon cumulants or a finite-nucleus monopole spectrum",
            "finite-h contrast is exact for fixed templates but Z_h has declared Taylor bias",
            "temperature, composition, density-calibration, covariance, and model-systematic errors are not supplied",
        ],
    }


def build_result() -> dict[str, Any]:
    """Rebuild primary/control rows live and return JSON-compatible data."""
    primary = _snapshot(PRIMARY_DPS)
    control = _snapshot(CONTROL_DPS)
    if _snapshot_has_failures(primary) or _snapshot_has_failures(control):
        precision = [{
            "status": "FAIL",
            "pass": False,
            "error": "one or more declared pressure rows or live derivatives failed and were retained",
        }]
        return _jsonable(_public_result(primary, control, precision, status=STATUS_FAIL))
    precision = _precision_controls(primary, control)
    status = STATUS_PASS if all(item["pass"] for item in precision) else STATUS_FAIL
    return _jsonable(_public_result(primary, control, precision, status=status))


@lru_cache(maxsize=1)
def _expected_result_json() -> str:
    return json.dumps(build_result(), sort_keys=True, ensure_ascii=False, allow_nan=False)


def validate_result(result: Any) -> bool:
    """Fail closed by comparing against a fresh live rebuilt scientific tree."""
    try:
        if not isinstance(result, dict):
            return False
        expected = json.loads(_expected_result_json())
        actual = json.loads(json.dumps(result, allow_nan=False))
        return actual == expected
    except (ArithmeticError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError, json.JSONDecodeError):
        return False


def _write_json(payload: Mapping[str, Any], path: Path | None) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path is not None:
        path.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional explicit JSON output path")
    args = parser.parse_args(argv)
    try:
        result = build_result()
        if result.get("status") != STATUS_PASS:
            _write_json(result, args.output)
            return 1
        if not validate_result(result):
            raise PressureBridgeError("fresh result failed live validation")
        _write_json(result, args.output)
        return 0
    except Exception as exc:  # strict finite JSON on failure
        failure = {"schema_version": SCHEMA_VERSION, "status": STATUS_FAIL, "error": str(exc)}
        _write_json(failure, args.output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
