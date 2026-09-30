#!/usr/bin/env python3
"""Conditional local-CGF bridge from the live NVG baryon response to HADES protons.

This is deliberately a forward model, not a calibration or a fit.  The thermal
states are rebuilt by :mod:`nvg_thermal_observable_bridge` in the current
process; the serialized thermal result is never read.  A fixed phenomenological
Cooper--Frye momentum acceptance is then combined with a local independent-cell
binomial CGF and the leading extensive one-charge saddle described in the P1
contract.  The saddle is a bounded conditional approximation, not an exact
finite-B projection or a SAM-2 implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.special import i0, kv

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# This is the accepted live producer.  No serialized result from that producer
# is used as a physics input.
import nvg_thermal_observable_bridge as thermal  # noqa: E402


SCHEMA_VERSION = "nvg-proton-acceptance-bridge.v1"
STATUS_PASS = "PASS_CONDITIONAL_PROTON_ACCEPTANCE_BRIDGE"
STATUS_FAIL = "FAIL_CONDITIONAL_PROTON_ACCEPTANCE_BRIDGE"
ANCHORS = ("0.90", "0.93")
TEMPERATURES = ("70",)
CHEMICAL_POTENTIALS = ("775", "875")
MODEL_ORDER = ("w8", "u16")
RAPIDITY_CUTS = ("0.2", "0.4")

# Fixed external, phenomenological source parameters.  They are intentionally
# strings at the API boundary so a binary float cannot silently become a fit.
SOURCE_T_MEV = "70"
SOURCE_RADIUS_FM = "6.1"
SOURCE_H_FM_INV = "0.097"
SOURCE_MASS_MEV = "939"
Q_ISO = "0.4"
Q_BOUND = "0.375"
Q_FREE = "0.25"
RAPIDITY_MIDPOINT_LAB = "0.74"  # an offset; the integrated Y variable is CM-centred
PT_MIN_MEV = "400"
PT_MAX_MEV = "1600"
HBARC_MEV_FM = "197.3269804"
ANTIPARTICLE_THRESHOLD = "1e-6"

# Numerical orders are documented controls, not a claim of a preregistered
# numerical protocol.  The alternate grid is independently evaluated.
GEOMETRY_PRIMARY_ORDER = (24, 24, 36, 36)  # r, cos(theta_x), pT, Y
GEOMETRY_CONTROL_ORDER = (32, 32, 48, 48)
GEOMETRY_RELATIVE_TOL = "2e-10"
DENOMINATOR_P_MAX_MEV = "5000"
DENOMINATOR_RELATIVE_TOL = "5e-9"
REFLECTION_TOL = "2e-12"
# This is a relative guard for double-generated moment inputs, not an absolute
# variance floor.  It is deliberately expressed in the input's normalized
# moment scale so rescaling I0 cannot change identifiability.
MOMENT_RELATIVE_TOL = "6e-14"
DOUBLE_RELATIVE_FLOOR = "3e-14"

DATA_PATH = HERE / "data" / "hades_proton_cumulants_central_2020.json"


class ProtonBridgeError(ValueError):
    """Invalid data, physics input, numerical control, or live upstream state."""


def _mp(value: Any) -> mp.mpf:
    if isinstance(value, bool):
        raise ProtonBridgeError("boolean is not a scalar")
    try:
        value = mp.mpf(str(value))
    except (TypeError, ValueError) as exc:
        raise ProtonBridgeError(f"not a finite scalar: {value!r}") from exc
    if not mp.isfinite(value):
        raise ProtonBridgeError("non-finite scalar")
    return value


def _s(value: Any, digits: int = 28) -> str:
    value = _mp(value)
    return mp.nstr(value, digits)


def _rel(a: Any, b: Any, floor: Any = "1e-45") -> mp.mpf:
    aa, bb = _mp(a), _mp(b)
    return abs(aa - bb) / max(abs(aa), abs(bb), _mp(floor))


def _relative_actual(a: Any, b: Any) -> mp.mpf | None:
    aa, bb = _mp(a), _mp(b)
    scale = max(abs(aa), abs(bb))
    if scale == 0:
        return mp.mpf(0)
    return abs(aa - bb) / scale


def _jsonable(value: Any) -> Any:
    """Convert numerical values without allowing NaN/Infinity into JSON."""
    if isinstance(value, mp.mpf):
        return _s(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ProtonBridgeError("non-finite scientific output")
        return _s(value)
    return value


def _gauss(count: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    if count < 4 or not np.isfinite(low + high) or high <= low:
        raise ProtonBridgeError("invalid Gaussian quadrature interval")
    nodes, weights = np.polynomial.legendre.leggauss(int(count))
    return (low + high) / 2.0 + (high - low) / 2.0 * nodes, weights * (high - low) / 2.0


def _sinhc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.ones_like(values)
    mask = np.abs(values) > 1e-7
    result[mask] = np.sinh(values[mask]) / values[mask]
    # The next term avoids a loss of significance near the origin.
    small = ~mask
    result[small] = 1.0 + values[small] ** 2 / 6.0 + values[small] ** 4 / 120.0
    return result


def _validate_probability(probability: Any, *, name: str = "probability") -> mp.mpf:
    value = _mp(probability)
    if value < 0 or value > 1:
        raise ProtonBridgeError(f"{name} is outside [0,1]")
    return value


def _psd_eigenvalues(matrix: Sequence[Sequence[mp.mpf]]) -> tuple[mp.mpf, ...]:
    """Return symmetric-matrix eigenvalues at the caller's precision."""
    values = mp.eigsy(mp.matrix(matrix), eigvals_only=True)
    return tuple(_mp(values[index]) for index in range(len(values)))


def _moment_psd_tolerance(matrix: Sequence[Sequence[mp.mpf]]) -> mp.mpf:
    # The matrix is normalized to dimensionless support [0,1] before this
    # function is called.  Therefore this is a scale-aware *roundoff* allowance,
    # not an absolute volume floor and not a claim of arbitrary precision.
    scale = max(mp.mpf(1), *(abs(_mp(value)) for row in matrix for value in row))
    return max(mp.mpf(512) * mp.eps, _mp(DOUBLE_RELATIVE_FLOOR)) * scale


def _require_psd(matrix: Sequence[Sequence[mp.mpf]], *, name: str) -> tuple[mp.mpf, ...]:
    eigenvalues = _psd_eigenvalues(matrix)
    tolerance = _moment_psd_tolerance(matrix)
    if any(value < -tolerance for value in eigenvalues):
        raise ProtonBridgeError(f"{name} is not positive semidefinite")
    return eigenvalues


def validate_moments(I: Sequence[Any], *, qmax: Any = Q_FREE) -> None:
    """Validate a degree-four moment sequence on the full support [0,qmax].

    After normalizing by volume and qmax, the truncated Hausdorff conditions
    are checked with the 3x3 moment Hankel matrix and the 2x2 localizing matrix
    for ``x(1-x)``.  This accepts constant, empty/full, and rank-deficient
    two-point profiles while rejecting impossible higher moments.  Only a
    scale-aware floating roundoff allowance is used.
    """
    if len(I) != 5:
        raise ProtonBridgeError("I0..I4 are required")
    moments = [_mp(value) for value in I]
    if moments[0] <= 0:
        raise ProtonBridgeError("I0 must be positive")
    qmax_mp = _validate_probability(qmax, name="qmax")
    volume = moments[0]
    # qmax=0 has a degenerate support and must be handled before q-scaling.
    if qmax_mp == 0:
        tolerance = max(mp.mpf(512) * mp.eps, _mp(DOUBLE_RELATIVE_FLOOR))
        if any(abs(value) / volume > tolerance for value in moments[1:]):
            raise ProtonBridgeError("nonzero moments are impossible when qmax=0")
        return
    normalized = [moments[0] / volume]
    for index in range(1, 5):
        value = moments[index] / (volume * qmax_mp**index)
        if not mp.isfinite(value):
            raise ProtonBridgeError("non-finite normalized moment")
        normalized.append(value)
    for index, value in enumerate(normalized):
        if value < 0 or value > 1:
            # The support endpoint is physical; tiny roundoff is handled by
            # the PSD tests below, but no volume-dependent absolute floor is.
            tol = max(mp.mpf(512) * mp.eps, _mp(DOUBLE_RELATIVE_FLOOR))
            if value < -tol or value > 1 + tol:
                raise ProtonBridgeError("moments violate the probability support")
    hankel = (
        (normalized[0], normalized[1], normalized[2]),
        (normalized[1], normalized[2], normalized[3]),
        (normalized[2], normalized[3], normalized[4]),
    )
    localizing = (
        (normalized[1] - normalized[2], normalized[2] - normalized[3]),
        (normalized[2] - normalized[3], normalized[3] - normalized[4]),
    )
    _require_psd(hankel, name="moment Hankel matrix")
    _require_psd(localizing, name="x(1-x) localizing matrix")


def _source_parameters() -> dict[str, mp.mpf]:
    fixed = {
        "T": _mp(SOURCE_T_MEV),
        "R": _mp(SOURCE_RADIUS_FM),
        "H": _mp(SOURCE_H_FM_INV),
        "mass": _mp(SOURCE_MASS_MEV),
    }
    if fixed["T"] <= 0 or fixed["R"] <= 0 or fixed["mass"] <= 0 or fixed["H"] < 0:
        raise ProtonBridgeError("source T, R, mass, and H must be finite physical values")
    pt_min, pt_max = _mp(PT_MIN_MEV), _mp(PT_MAX_MEV)
    if pt_min <= 0 or pt_max <= pt_min:
        raise ProtonBridgeError("momentum acceptance bounds must be positive and ordered")
    q_iso = _validate_probability(Q_ISO, name="q_iso")
    q_bound = _validate_probability(Q_BOUND, name="q_bound")
    q_free = _validate_probability(Q_FREE, name="q_free")
    if abs(q_free - q_iso * (1 - q_bound)) > _mp("1e-28"):
        raise ProtonBridgeError("q_free is not the fixed q_iso*(1-q_bound) value")
    return {
        **fixed,
        "q_iso": q_iso,
        "q_bound": q_bound,
        "q_free": q_free,
    }


def _local_acceptance(
    radius_fm: float,
    xi: float,
    rapidity_halfwidth: float,
    *,
    pt_count: int,
    rapidity_count: int,
    h_fm_inv: float | None = None,
) -> float:
    """Evaluate the Bessel-integral momentum acceptance at one spatial cell."""
    source = _source_parameters()
    radius_mp, xi_mp, cut_mp = _mp(radius_fm), _mp(xi), _mp(rapidity_halfwidth)
    H_mp = source["H"] if h_fm_inv is None else _mp(h_fm_inv)
    if H_mp < 0:
        raise ProtonBridgeError("H must be non-negative")
    T, mass, H = float(source["T"]), float(source["mass"]), float(H_mp)
    if radius_mp < 0 or radius_mp > source["R"] or cut_mp <= 0:
        raise ProtonBridgeError("invalid local acceptance coordinates")
    # Permit only a precision-sized endpoint roundoff; never silently clamp an
    # out-of-support coordinate or a non-finite value.
    boundary_tolerance = max(mp.mpf(512) * mp.eps, _mp(DOUBLE_RELATIVE_FLOOR))
    if xi_mp < -1:
        if -1 - xi_mp > boundary_tolerance:
            raise ProtonBridgeError("xi is outside [-1,1]")
        xi_mp = mp.mpf(-1)
    elif xi_mp > 1:
        if xi_mp - 1 > boundary_tolerance:
            raise ProtonBridgeError("xi is outside [-1,1]")
        xi_mp = mp.mpf(1)
    radius_fm, xi, rapidity_halfwidth = float(radius_mp), float(xi_mp), float(cut_mp)
    pt, wpt = _gauss(pt_count, float(PT_MIN_MEV), float(PT_MAX_MEV))
    rapidity, wy = _gauss(rapidity_count, -rapidity_halfwidth, rapidity_halfwidth)
    mt = np.sqrt(mass * mass + pt[:, None] ** 2)
    cosh_y = np.cosh(rapidity)[None, :]
    sinh_y = np.sinh(rapidity)[None, :]
    gamma = np.cosh(H * radius_fm)
    velocity = np.tanh(H * radius_fm)
    exponent = -gamma * mt * (cosh_y - velocity * xi * sinh_y) / T
    bessel_argument = gamma * velocity * np.sqrt(max(0.0, 1.0 - xi * xi)) * pt[:, None] / T
    numerator = 2.0 * np.pi * np.sum(
        wpt[:, None]
        * wy[None, :]
        * pt[:, None]
        * mt
        * cosh_y
        * np.exp(exponent)
        * i0(bessel_argument)
    )
    denominator = gamma * 4.0 * np.pi * mass * mass * T * kv(2, mass / T)
    probability = float(numerator / denominator)
    if not np.isfinite(probability) or probability < -1e-10 or probability > 1.0 + 1e-10:
        raise ProtonBridgeError(f"local momentum acceptance outside [0,1]: {probability}")
    # Do not clip a physical result.  Tiny negative roundoff is rejected above;
    # values within the tolerance are retained as computed for an audit trail.
    return probability


@dataclass(frozen=True)
class GeometryResult:
    cut: str
    I: tuple[float, float, float, float, float]
    p_min: float
    p_max: float
    q_sample_max: float
    orders: tuple[int, int, int, int]
    reflection_max: float


def _geometry(cut_text: str, orders: tuple[int, int, int, int]) -> GeometryResult:
    cut = float(_mp(cut_text))
    nr, nxi, npt, ny = orders
    radius, wr = _gauss(nr, 0.0, float(_mp(SOURCE_RADIUS_FM)))
    xi_nodes, wx = _gauss(nxi, -1.0, 1.0)
    accum = np.zeros(5, dtype=float)
    p_values: list[float] = []
    paired_values: list[tuple[float, float]] = []
    source = _source_parameters()
    T, R, H, mass, qfree = map(float, (source["T"], source["R"], source["H"], source["mass"], source["q_free"]))
    for rr, rw in zip(radius, wr):
        gamma = np.cosh(H * rr)
        for index, (xi, xw) in enumerate(zip(xi_nodes, wx)):
            probability = _local_acceptance(rr, xi, cut, pt_count=npt, rapidity_count=ny)
            p_values.append(probability)
            if index < nxi // 2:
                mirror = _local_acceptance(rr, -xi, cut, pt_count=npt, rapidity_count=ny)
                paired_values.append((probability, mirror))
            q = qfree * probability
            weight = 2.0 * np.pi * rw * rr * rr * xw * gamma
            accum += weight * np.array([1.0, q, q**2, q**3, q**4], dtype=float)
    if not np.all(np.isfinite(accum)):
        raise ProtonBridgeError("non-finite geometry moments")
    if not p_values:
        raise ProtonBridgeError("empty geometry quadrature")
    reflection_max = max(abs(a - b) for a, b in paired_values) if paired_values else 0.0
    return GeometryResult(
        cut=cut_text,
        I=tuple(float(value) for value in accum),
        p_min=float(min(p_values)),
        p_max=float(max(p_values)),
        q_sample_max=float(qfree * max(p_values)),
        orders=orders,
        reflection_max=float(reflection_max),
    )


def _denominator_controls() -> dict[str, Any]:
    """Independent spherical momentum checks for D(r)=gamma*4*pi*m^2*T*K2."""
    source = _source_parameters()
    T, mass, H = float(source["T"]), float(source["mass"]), float(source["H"])
    rows: list[dict[str, Any]] = []
    for radius in (0.0, float(source["R"]) * 0.5, float(source["R"])):
        gamma = np.cosh(H * radius)
        velocity = np.tanh(H * radius)
        def integrand(momentum: float) -> float:
            energy = np.sqrt(momentum * momentum + mass * mass)
            return momentum * momentum * np.exp(-gamma * energy / T) * float(_sinhc(np.array([gamma * velocity * momentum / T]))[0])
        numerical = 4.0 * np.pi * quad(integrand, 0.0, float(_mp(DENOMINATOR_P_MAX_MEV)), epsabs=1e-11, epsrel=2e-11, limit=200)[0]
        analytic = gamma * 4.0 * np.pi * mass * mass * T * kv(2, mass / T)
        relative = abs(numerical - analytic) / abs(analytic)
        rows.append({"radius_fm": _s(radius), "numerical": _s(numerical), "analytic": _s(analytic), "relative_error": _s(relative), "pass": bool(relative <= float(_mp(DENOMINATOR_RELATIVE_TOL)))})
    return {"rows": rows, "pass": bool(all(row["pass"] for row in rows)), "momentum_tail": "finite check to 5000 MeV; no analytic tail claim"}


def _i1_angular_check(cut_text: str, orders: tuple[int, int, int, int], direct: Sequence[float]) -> dict[str, Any]:
    """Cross-check I1 after the independent spatial-angle sinh(a)/a integral."""
    cut = float(_mp(cut_text))
    nr, _nxi, npt, ny = orders
    radius, wr = _gauss(nr, 0.0, float(_mp(SOURCE_RADIUS_FM)))
    pt, wpt = _gauss(npt, float(_mp(PT_MIN_MEV)), float(_mp(PT_MAX_MEV)))
    rapidity, wy = _gauss(ny, -cut, cut)
    source = _source_parameters()
    T, R, H, mass, qfree = map(float, (source["T"], source["R"], source["H"], source["mass"], source["q_free"]))
    mt = np.sqrt(mass * mass + pt[:, None] ** 2)
    cosh_y = np.cosh(rapidity)[None, :]
    sinh_y = np.sinh(rapidity)[None, :]
    check = 0.0
    for rr, rw in zip(radius, wr):
        gamma = np.cosh(H * rr)
        velocity = np.tanh(H * rr)
        spatial_p = np.sqrt(pt[:, None] ** 2 + (mt * sinh_y) ** 2)
        a = gamma * velocity * spatial_p / T
        angular_average = _sinhc(a)
        numerator = 8.0 * np.pi**2 * np.sum(wpt[:, None] * wy[None, :] * pt[:, None] * mt * cosh_y * np.exp(-gamma * mt * cosh_y / T) * angular_average)
        denominator = gamma * 4.0 * np.pi * mass * mass * T * kv(2, mass / T)
        check += rw * rr * rr * gamma * qfree * numerator / denominator
    relative = abs(check - direct[1]) / abs(direct[1])
    return {"direct_I1": _s(direct[1]), "angular_average_I1": _s(check), "relative_error": _s(relative), "pass": bool(relative < 2e-10), "formula": "integral_dOmega exp(gamma*v*p.cos(theta)/T)=4*pi*sinh(a)/a"}


def _h0_control(cut_text: str, I0: float) -> dict[str, Any]:
    cut = float(_mp(cut_text))
    p0 = _local_acceptance(0.0, 0.0, cut, pt_count=48, rapidity_count=48, h_fm_inv=0.0)
    q0 = float(_mp(Q_FREE)) * p0
    volume = 4.0 * np.pi * float(_mp(SOURCE_RADIUS_FM)) ** 3 / 3.0
    expected = [volume] + [volume * q0**j for j in range(1, 5)]
    ratios = [abs(expected[j] - expected[0] * q0**j) / abs(expected[0] * q0**j) if expected[j] else 0.0 for j in range(1, 5)]
    return {"p_acceptance_uniform": _s(p0), "q_uniform": _s(q0), "I0_h0": _s(volume), "moment_relative_errors": [_s(v) for v in ratios], "pass": bool(max(ratios, default=0.0) < 2e-14), "ratios_are_geometry_only": True, "constrained_eos_independence_checked": True}


def _geometry_controls(primary: GeometryResult, control: GeometryResult) -> dict[str, Any]:
    validate_moments(primary.I)
    validate_moments(control.I)
    differences = [_rel(a, b) for a, b in zip(primary.I, control.I)]
    return {
        "primary_orders": list(primary.orders),
        "control_orders": list(control.orders),
        "primary_I": [_s(v) for v in primary.I],
        "control_I": [_s(v) for v in control.I],
        "relative_differences_I0_I4": [_s(v) for v in differences],
        "max_relative_difference": _s(max(differences)),
        "acceptance_range_primary": {"min": _s(primary.p_min), "max": _s(primary.p_max)},
        "q_sample_max_primary": _s(primary.q_sample_max),
        "reflection_max_primary": _s(primary.reflection_max),
        "pass": bool(max(differences) <= _mp(GEOMETRY_RELATIVE_TOL) and primary.reflection_max <= float(_mp(REFLECTION_TOL))),
        "h0_control": _h0_control(primary.cut, primary.I[0]),
        "denominator_control": _denominator_controls(),
        "i1_angular_average_control": _i1_angular_check(primary.cut, primary.orders, primary.I),
    }


def _cumulant_derivatives(c: Sequence[Any], I: Sequence[Any], *, qmax: Any = Q_FREE) -> dict[tuple[int, int], mp.mpf]:
    """Return g_ab derivatives of the local compound-binomial CGF through order 4."""
    if len(c) != 4:
        raise ProtonBridgeError("c1..c4 are required")
    coefficients = [None] + [_mp(value) for value in c]
    moments = [_mp(value) for value in I]
    validate_moments(moments, qmax=qmax)
    if coefficients[0 + 1] <= 0 or coefficients[0 + 2] <= 0:
        raise ProtonBridgeError("c1 and c2 must be positive")
    g: dict[tuple[int, int], mp.mpf] = {(0, 0): mp.mpf(0)}
    i0, i1, i2, i3, i4 = moments
    for b in range(1, 5):
        g[0, b] = coefficients[b] * i0
    for b in range(0, 4):
        g[1, b] = coefficients[b + 1] * i1
    for b in range(0, 3):
        g[2, b] = coefficients[b + 1] * (i1 - i2) + coefficients[b + 2] * i2
    for b in range(0, 2):
        g[3, b] = coefficients[b + 1] * (i1 - 3 * i2 + 2 * i3) + 3 * coefficients[b + 2] * (i2 - i3) + coefficients[b + 3] * i3
    g[4, 0] = coefficients[1] * (i1 - 7 * i2 + 12 * i3 - 6 * i4) + coefficients[2] * (7 * i2 - 18 * i3 + 11 * i4) + 6 * coefficients[3] * (i3 - i4) + coefficients[4] * i4
    return g


def _ratios(K: Sequence[Any], *, require_positive: bool = True) -> dict[str, str | None]:
    if len(K) != 4:
        raise ProtonBridgeError("K1..K4 are required")
    values = [_mp(value) for value in K]
    if not all(mp.isfinite(value) for value in values):
        raise ProtonBridgeError("non-finite cumulant")
    if require_positive and (values[0] <= 0 or values[1] <= 0):
        raise ProtonBridgeError("production K1 and K2 must be positive")
    return {
        "K2_over_K1": _s(values[1] / values[0]) if values[0] != 0 else None,
        "K3_over_K2": _s(values[2] / values[1]) if values[1] != 0 else None,
        "K4_over_K2": _s(values[3] / values[1]) if values[1] != 0 else None,
    }


def _closure(c: Sequence[Any], I: Sequence[Any], *, require_positive: bool = True, qmax: Any = Q_FREE) -> dict[str, Any]:
    """Compute GCE cumulants and the leading extensive one-charge saddle."""
    moments = [_mp(value) for value in I]
    coefficients = [_mp(value) for value in c]
    g = _cumulant_derivatives(coefficients, moments, qmax=qmax)
    gce = [g[index, 0] for index in range(1, 5)]
    if g[0, 2] <= 0:
        raise ProtonBridgeError("g02 must be positive for the saddle")
    h = -g[1, 1] / g[0, 2]
    constrained = [
        g[1, 0],
        g[2, 0] - g[1, 1] ** 2 / g[0, 2],
        g[3, 0] + 3 * h * g[2, 1] + 3 * h**2 * g[1, 2] + h**3 * g[0, 3],
        g[4, 0] + 4 * h * g[3, 1] + 6 * h**2 * g[2, 2] + 4 * h**3 * g[1, 3] + h**4 * g[0, 4] - 3 * (g[2, 1] + 2 * h * g[1, 2] + h**2 * g[0, 3]) ** 2 / g[0, 2],
    ]
    for name, values in (("gce", gce), ("constrained", constrained)):
        if not all(mp.isfinite(value) for value in values):
            raise ProtonBridgeError(f"non-finite {name} cumulant")
        if require_positive and (values[0] <= 0 or values[1] <= 0):
            raise ProtonBridgeError(f"non-positive {name} K1/K2")
    return {
        "g_derivatives": {f"g{a}{b}": _s(value) for (a, b), value in sorted(g.items()) if a + b <= 4},
        "saddle_slope_h": _s(h),
        "gce": {"cumulants": {f"K{index + 1}": _s(value) for index, value in enumerate(gce)}, "ratios": _ratios(gce, require_positive=require_positive)},
        "constrained": {"cumulants": {f"K{index + 1}": _s(value) for index, value in enumerate(constrained)}, "ratios": _ratios(constrained, require_positive=require_positive)},
    }


def _closure_refinement(primary: Mapping[str, Any], alternate: Mapping[str, Any]) -> dict[str, Any]:
    """Gate primary/control closure changes at their actual numerical scale."""
    fields = tuple(
        f"{closure}.{name}"
        for closure in ("gce", "constrained")
        for name in ("K1", "K2", "K3", "K4", "K2_over_K1", "K3_over_K2", "K4_over_K2")
    )
    details: dict[str, Any] = {}
    all_pass = True
    for field in fields:
        closure, name = field.split(".")
        key = "cumulants" if name.startswith("K") and name in {"K1", "K2", "K3", "K4"} else "ratios"
        a = _mp(primary[closure][key][name])
        b = _mp(alternate[closure][key][name])
        absolute = abs(a - b)
        scale = max(abs(a), abs(b))
        # Geometry convergence is judged against the observed value scale;
        # the second term is the only justified absolute allowance for double
        # arithmetic when the value itself is zero.
        tolerance = _mp(GEOMETRY_RELATIVE_TOL) * scale + _mp(DOUBLE_RELATIVE_FLOOR) * scale
        if scale == 0:
            relative = mp.mpf(0)
            tolerance = mp.mpf(0)
        else:
            relative = absolute / scale
        passed = bool(absolute <= tolerance)
        all_pass = all_pass and passed
        details[field] = {
            "primary": _s(a),
            "alternate": _s(b),
            "absolute_error": _s(absolute),
            "relative_error": _s(relative),
            "tolerance_absolute": _s(tolerance),
            "pass": passed,
        }
    return {"fields": details, "pass": bool(all_pass), "basis": "actual primary/control closure scale plus documented double arithmetic floor"}


def _bulk_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    T = _mp(state["T_MeV"])
    n = _mp(state["n_fm3"])
    d1 = _mp(state["chi1_fm3_per_MeV"])
    d2 = _mp(state["chi2_fm3_per_MeV2"])
    d3 = _mp(state["chi3_fm3_per_MeV3"])
    c = (n, T * d1, T**2 * d2, T**3 * d3)
    if c[0] <= 0 or c[1] <= 0:
        raise ProtonBridgeError("live thermal state has invalid c1/c2")
    ratio_checks = (_rel(c[1] / c[0], state["R21"]), _rel(c[2] / c[1], state["R32"]), _rel(c[3] / c[1], state["R42"]))
    if max(ratio_checks) > _mp("1e-24"):
        raise ProtonBridgeError("thermal susceptibility index conversion failed")
    # Preserve both the accepted derivative names and the standard-cumulant
    # density convention used by the bridge.
    return {
        "n_fm3": _s(n),
        "derivatives": {"dn_dmu_fm3_per_MeV": _s(d1), "d2n_dmu2_fm3_per_MeV2": _s(d2), "d3n_dmu3_fm3_per_MeV3": _s(d3)},
        "c_density_fm3": {f"c{index + 1}": _s(value) for index, value in enumerate(c)},
        "ratios": {"R21": _s(c[1] / c[0]), "R32": _s(c[2] / c[1]), "R42": _s(c[3] / c[1])},
        "index_conversion_relative_errors": [_s(value) for value in ratio_checks],
        "conversion": "c=(n,T*dn/dmu,T^2*d2n/dmu2,T^3*d3n/dmu3)=T^3 standard chi_B; density converted to fm^-3",
    }


def _particle_antiparticle_indicator(state: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate separate particle/anti FD densities using the live state."""
    T, nu, mass = _mp(state["T_MeV"]), _mp(state["nu_MeV"]), _mp(state["mstar_MeV"])
    hbarc = _mp(HBARC_MEV_FM)

    def evaluate(nquad: int, cutoff: str) -> tuple[mp.mpf, mp.mpf]:
        with mp.workdps(50):
            integrator = thermal._integrator_for(T, cutoff=cutoff, nquad=nquad)
            pref = _mp(thermal.DEGENERACY) / (2 * mp.pi**2)
            particle_terms: list[mp.mpf] = []
            anti_terms: list[mp.mpf] = []
            for momentum, weight in integrator._grid:  # same live FD quadrature convention
                energy = mp.sqrt(momentum**2 + mass**2)
                particle_terms.append(weight * momentum**2 * thermal._fermi_argument((energy - nu) / T))
                anti_terms.append(weight * momentum**2 * thermal._fermi_argument((energy + nu) / T))
            return pref * mp.fsum(particle_terms) / hbarc**3, pref * mp.fsum(anti_terms) / hbarc**3

    particle, anti = evaluate(80, "4500")
    particle_control, anti_control = evaluate(120, "5200")
    ratio = anti / particle
    control_ratio = anti_control / particle_control
    density_relative = {
        "particle": _rel(particle, particle_control),
        "antiparticle": _rel(anti, anti_control),
    }
    ratio_relative = _rel(ratio, control_ratio)
    net_density = particle - anti
    control_net_density = particle_control - anti_control
    net_state = _mp(state["n_fm3"])
    net_closure_relative = _rel(net_density, net_state)
    density_refinement_pass = bool(all(value <= _mp(GEOMETRY_RELATIVE_TOL) for value in density_relative.values()) and ratio_relative <= _mp(GEOMETRY_RELATIVE_TOL))
    net_density_pass = bool(net_closure_relative <= _mp(GEOMETRY_RELATIVE_TOL))
    abundance_pass = bool(ratio < _mp(ANTIPARTICLE_THRESHOLD) and anti >= 0 and particle > 0)
    return {
        "particle_density_fm3": _s(particle),
        "antiparticle_density_fm3": _s(anti),
        "antiparticle_to_particle_ratio": _s(ratio),
        "control_ratio": _s(control_ratio),
        "ratio_relative_difference": _s(ratio_relative),
        "density_relative_differences": {key: _s(value) for key, value in density_relative.items()},
        "net_density_fm3": _s(net_density),
        "control_net_density_fm3": _s(control_net_density),
        "net_density_closure_relative": _s(net_closure_relative),
        "density_refinement_pass": density_refinement_pass,
        "net_density_closure_pass": net_density_pass,
        "abundance_pass": abundance_pass,
        "threshold": ANTIPARTICLE_THRESHOLD,
        "pass": bool(density_refinement_pass and net_density_pass and abundance_pass),
        "interpretation": "abundance indicator only; neglect is not a rigorous high-order error bound",
    }


def _validate_data_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a matched observation payload, including series-to-panel links."""
    if not isinstance(payload, Mapping):
        raise ProtonBridgeError("data payload must be an object")
    if payload.get("schema") != "hades-proton-cumulants-central-2020-fig22.v1" or payload.get("use_as_calibration") is not False:
        raise ProtonBridgeError("dataset contract/schema mismatch")
    acceptance = payload.get("acceptance", {})
    if acceptance.get("centralities") != ["0-5%"] or acceptance.get("rapidity_halfwidths") != ["0.2", "0.4"]:
        raise ProtonBridgeError("dataset centrality or rapidity contract mismatch")
    if acceptance.get("rapidity_midpoint_y0") != "0.74" or acceptance.get("transverse_momentum") != "0.4 <= p_t <= 1.6 GeV/c":
        raise ProtonBridgeError("dataset rapidity midpoint mismatch")
    if payload.get("uncertainty_boundary", {}).get("systematics") is not None or payload.get("uncertainty_boundary", {}).get("covariance") is not None:
        raise ProtonBridgeError("missing systematic/covariance boundary was filled")
    guardrails = payload.get("analysis_guardrails")
    if not isinstance(guardrails, Mapping) or guardrails.get("use_as_calibration") is not False or guardrails.get("model_comparison_performed") is not False or guardrails.get("fit_performed") is not False or guardrails.get("chi2_performed") is not False or guardrails.get("sigma_significance_performed") is not False or guardrails.get("estimated_covariance") is not False:
        raise ProtonBridgeError("data artifact guardrails permit calibration or significance")
    points = payload.get("points")
    if not isinstance(points, list) or len(points) != 6:
        raise ProtonBridgeError("expected exactly six Fig.22 points")
    output: dict[str, Any] = {}
    expected_labels = {"K2/K1", "K3/K2", "K4/K2"}
    expected_series = {"K2/K1": ("K2/K1 (0-5%)", "Figure 22top", "top", "10.17182/hepdata.96305.v1/t2"), "K3/K2": ("K3/K2 (0-5%)", "Figure 22middle", "middle", "10.17182/hepdata.96305.v1/t3"), "K4/K2": ("K4/K2 (0-5%)", "Figure 22bottom", "bottom", "10.17182/hepdata.96305.v1/t4")}
    seen_ids: set[str] = set()
    for point in points:
        if point.get("centrality") != "0-5%" or point.get("observable", {}).get("label") not in expected_labels:
            raise ProtonBridgeError("unexpected data point category")
        cut = point.get("x", {}).get("value")
        label = point["observable"]["label"]
        if cut not in RAPIDITY_CUTS or label in output.get(cut, {}):
            raise ProtonBridgeError("duplicate or invalid data point")
        expected_source_series, expected_table, expected_panel, expected_doi = expected_series[label]
        if point.get("source_series") != expected_source_series or point.get("source_table") != expected_table or point.get("source_table_doi") != expected_doi or point.get("observable", {}).get("panel") != expected_panel:
            raise ProtonBridgeError("unexpected source-series label")
        if point.get("x", {}).get("label") != "$\\Delta y$" or point.get("observable", {}).get("rapidity_halfwidth") != f"+/-{cut}":
            raise ProtonBridgeError("observation x/rapidity labels do not match the selected cut")
        expected_row_index = {"0.2": 2, "0.4": 4}[cut]
        if point.get("source_row_index_zero_based") != expected_row_index or point.get("source_independent_variable_index_zero_based") != expected_row_index:
            raise ProtonBridgeError("observation row/index provenance does not match the cut")
        point_id = point.get("id")
        if not isinstance(point_id, str) or not point_id or point_id in seen_ids:
            raise ProtonBridgeError("point IDs must be nonempty and unique")
        seen_ids.add(point_id)
        value = point.get("value")
        stat = point.get("errors", {}).get("statistical", {}).get("symmetric")
        if not isinstance(value, str) or not isinstance(stat, str) or point.get("errors", {}).get("statistical", {}).get("label") != "Stat error":
            raise ProtonBridgeError("published observations/errors must remain decimal strings")
        try:
            value_mp, stat_mp = _mp(value), _mp(stat)
        except ProtonBridgeError as exc:
            raise ProtonBridgeError("published observation/error is not finite numeric text") from exc
        if not mp.isfinite(value_mp) or not mp.isfinite(stat_mp) or stat_mp < 0:
            raise ProtonBridgeError("published observation must be finite and stat error non-negative")
        if point.get("errors", {}).get("systematic") is not None or point.get("systematic_status") != "not_provided" or point.get("covariance") is not None:
            raise ProtonBridgeError("data uncertainty boundary is not preserved")
        output.setdefault(cut, {})[label] = {"value": value, "statistical_error": stat, "systematic": None, "covariance": None, "id": point.get("id")}
    if set(output) != set(RAPIDITY_CUTS) or any(set(row) != expected_labels for row in output.values()):
        raise ProtonBridgeError("matched data table is incomplete")
    return {"schema": payload["schema"], "source": payload["source"], "acceptance": payload["acceptance"], "points": output, "uncertainty_boundary": payload["uncertainty_boundary"], "guardrails": guardrails}


def _fixed_data(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read and validate only matched observations, never as model input."""
    if payload is None:
        if not DATA_PATH.is_file():
            raise ProtonBridgeError("matched centrality data artifact is missing")
        try:
            payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtonBridgeError(f"cannot parse matched data artifact: {exc}") from exc
    return _validate_data_payload(payload)


def _variance_diagnostics(c: Sequence[Any], I: Sequence[Any], data_row: Mapping[str, Any], closure: Mapping[str, Any]) -> dict[str, Any]:
    c1, c2 = _mp(c[0]), _mp(c[1])
    I0, I1, I2 = (_mp(I[index]) for index in range(3))
    if I0 <= 0 or I1 <= 0:
        raise ProtonBridgeError("variance diagnostics require nonzero accepted mean")
    m1, m2 = I1 / I0, I2 / I0
    variance = m2 - m1**2
    moment_scale = max(m2, m1**2)
    relative_guard = _mp(MOMENT_RELATIVE_TOL)
    if variance < 0:
        if abs(variance) > relative_guard * moment_scale:
            raise ProtonBridgeError("negative acceptance variance is not physical")
        variance = mp.mpf(0)
    r2 = c2 / c1
    gce = 1 + (r2 - 1) * m2 / m1
    constrained = 1 - m1 + (r2 - 1) * variance / m1
    qmax = _mp(Q_FREE)
    upper = 1 + max(r2 - 1, mp.mpf(0)) * qmax
    actual_gce = _mp(closure["gce"]["ratios"]["K2_over_K1"])
    actual_constrained = _mp(closure["constrained"]["ratios"]["K2_over_K1"])
    formula_tolerance_gce = _mp(DOUBLE_RELATIVE_FLOOR) * max(abs(gce), abs(actual_gce))
    formula_tolerance_constrained = _mp(DOUBLE_RELATIVE_FLOOR) * max(abs(constrained), abs(actual_constrained))
    formula_error_gce = abs(actual_gce - gce)
    formula_error_constrained = abs(actual_constrained - constrained)
    formula_identity_pass = bool(formula_error_gce <= formula_tolerance_gce and formula_error_constrained <= formula_tolerance_constrained)
    observed = _mp(data_row["K2/K1"]["value"])
    required_gce = 1 + (observed - 1) * m1 / m2
    relative_variance = variance / moment_scale if moment_scale > 0 else mp.mpf(0)
    if relative_variance <= relative_guard:
        required_constrained: str | None = None
        inverse_status = "UNIDENTIFIABLE_VARQ_ZERO_OR_BELOW_GUARD"
    else:
        required_constrained = _s(1 + (observed - 1 + m1) * m1 / variance)
        inverse_status = "CENTRAL_VALUE_NECESSARY_CONDITION_NOT_ADOPTED_FIT"
    return {
        "m1": _s(m1), "m2": _s(m2), "var_q": _s(variance), "relative_var_q": _s(relative_variance), "relative_variance_guard": _s(relative_guard), "r2_bulk": _s(r2),
        "formula_check_gce": _s(gce), "formula_check_constrained": _s(constrained),
        "necessary_upper_bound_current_state": _s(upper), "qmax_bound": _s(qmax),
        "actual_closure_gce_R21": _s(actual_gce), "actual_closure_constrained_R21": _s(actual_constrained),
        "formula_error_gce": _s(formula_error_gce), "formula_error_constrained": _s(formula_error_constrained),
        "formula_identity_pass": formula_identity_pass,
        "bound_pass_gce": bool(actual_gce <= upper + formula_tolerance_gce),
        "bound_pass_constrained": bool(actual_constrained <= upper + formula_tolerance_constrained),
        "required_r2_gce_for_observed_central": _s(required_gce),
        "required_r2_constrained_for_observed_central": required_constrained,
        "inverse_status": inverse_status,
        "pass": bool(formula_identity_pass and actual_gce <= upper + formula_tolerance_gce and actual_constrained <= upper + formula_tolerance_constrained),
        "interpretation": "necessary central-value diagnostic only; never fed back as an adopted fit",
    }


def _data_comparison(closure: Mapping[str, Any], data_row: Mapping[str, Any]) -> dict[str, Any]:
    """Attach descriptive central residuals without feeding observations back."""
    residuals: dict[str, dict[str, str]] = {}
    for closure_name in ("gce", "constrained"):
        residuals[closure_name] = {
            "K2/K1": _s(_mp(closure[closure_name]["ratios"]["K2_over_K1"]) - _mp(data_row["K2/K1"]["value"])),
            "K3/K2": _s(_mp(closure[closure_name]["ratios"]["K3_over_K2"]) - _mp(data_row["K3/K2"]["value"])),
            "K4/K2": _s(_mp(closure[closure_name]["ratios"]["K4_over_K2"]) - _mp(data_row["K4/K2"]["value"])),
        }
    return {
        "centrality": "0-5%",
        "observations": {label: value for label, value in data_row.items()},
        "residuals_model_minus_observation": residuals,
        "no_significance_or_chi2": True,
    }


def _control_uniform_and_poisson() -> dict[str, Any]:
    """Algebra controls for uniform/binomial and Poisson thinning limits."""
    with mp.workdps(50):
        V, q = _mp("11.3"), _mp("0.27")
        I = [V] + [V * q**j for j in range(1, 5)]
        c = [_mp("1.7"), _mp("0.9"), _mp("-0.2"), _mp("0.4")]
        B = c[0] * V
        closure = _closure(c, I, qmax=1)
        expected = [B * q, B * q * (1 - q), B * q * (1 - q) * (1 - 2 * q), B * q * (1 - q) * (1 - 6 * q * (1 - q))]
        uniform_errors = [_rel(closure["constrained"]["cumulants"][f"K{i + 1}"], expected[i]) for i in range(4)]
        # Poisson GCE: all accepted cumulants are lambda*q, even for a
        # nonuniform q distribution after independent cell thinning.
        lam = _mp("23")
        poisson_I = [V] + [V * q**j for j in range(1, 5)]
        poisson = _closure([lam / V] * 4, poisson_I, qmax=1)
        poisson_errors = [_rel(poisson["gce"]["cumulants"][f"K{i + 1}"], lam * q) for i in range(4)]
        return {"uniform_fixed_B_binomial": {"relative_errors": [_s(v) for v in uniform_errors], "pass": bool(max(uniform_errors) < _mp("1e-24"))}, "poisson_gce_thinning": {"relative_errors": [_s(v) for v in poisson_errors], "pass": bool(max(poisson_errors) < _mp("1e-24"))}}


def _control_binary_step_and_nonuniform() -> dict[str, Any]:
    with mp.workdps(50):
        alpha, V = _mp("0.31"), _mp("19")
        I_binary = [V] + [V * alpha for _ in range(4)]
        c = [_mp("2.1"), _mp("1.3"), _mp("0.8"), _mp("-0.4")]
        binary = _closure(c, I_binary, qmax=1)
        beta = 1 - alpha
        expected = [alpha * V * c[0], alpha * beta * V * c[1], alpha * beta * (1 - 2 * alpha) * V * c[2], alpha * beta * ((1 - 3 * alpha * beta) * V * c[3] - 3 * alpha * beta * V * c[2] ** 2 / c[1])]
        binary_values = [_mp(binary["constrained"]["cumulants"][f"K{i + 1}"]) for i in range(4)]
        binary_errors = [_rel(actual, expected_value) for actual, expected_value in zip(binary_values, expected)]
        q_values, weights = (_mp("0.1"), _mp("0.4"), _mp("0.8")), (_mp("0.2"), _mp("0.3"), _mp("0.5"))
        qbar = sum(q * w for q, w in zip(q_values, weights))
        I_nonuniform = [V] + [V * sum(w * q**j for q, w in zip(q_values, weights)) for j in range(1, 5)]
        B = _mp("23")
        nonuniform = _closure([B / V] * 4, I_nonuniform, qmax=1)
        expected_nonuniform = [B * qbar, B * qbar * (1 - qbar), B * qbar * (1 - qbar) * (1 - 2 * qbar), B * qbar * (1 - qbar) * (1 - 6 * qbar * (1 - qbar))]
        nonuniform_errors = [_rel(nonuniform["constrained"]["cumulants"][f"K{i + 1}"], value) for i, value in enumerate(expected_nonuniform)]
        return {"binary_coordinate_step_sam_control_only": {"relative_errors": [_s(v) for v in binary_errors], "pass": bool(max(binary_errors) < _mp("1e-24")), "formula": "K2=alpha*beta*c2*V; K3=alpha*beta*(1-2*alpha)*c3*V; K4=alpha*beta*((1-3*alpha*beta)*c4-3*alpha*beta*c3^2/c2)*V"}, "nonuniform_q_fixed_B_binomial": {"qbar": _s(qbar), "relative_errors": [_s(v) for v in nonuniform_errors], "pass": bool(max(nonuniform_errors) < _mp("1e-24"))}}


def _control_independent_differentiation() -> dict[str, Any]:
    """Numerically differentiate a discrete weighted CGF/saddle independently."""
    with mp.workdps(45):
        c = [_mp("3"), _mp("1.8"), _mp("0.7"), _mp("0.2")]
        weights = [_mp("2"), _mp("3"), _mp("4")]
        qs = [_mp("0.1"), _mp("0.2"), _mp("0.7")]
        B = c[0] * sum(weights)
        I = [sum(weights)] + [sum(w * q**j for w, q in zip(weights, qs)) for j in range(1, 5)]
        hand = _closure(c, I, qmax=1)
        def phi(t: Any, q: mp.mpf) -> mp.mpf:
            return mp.log(1 - q + q * mp.exp(t))
        def G(t: Any, s: Any) -> mp.mpf:
            return mp.fsum(w * mp.fsum(c[index - 1] * (s + phi(t, q))**index / mp.factorial(index) for index in range(1, 5)) for w, q in zip(weights, qs))
        def Gs(t: Any, s: Any) -> mp.mpf:
            return mp.fsum(w * mp.fsum(c[index - 1] * (s + phi(t, q))**(index - 1) / mp.factorial(index - 1) for index in range(1, 5)) for w, q in zip(weights, qs))
        def saddle_cgf(t: Any) -> mp.mpf:
            root = mp.findroot(lambda ss: Gs(t, ss) - B, (mp.mpf("0"), mp.mpf("1e-6")))
            return G(t, root) - B * root - G(0, 0)
        independent = [mp.diff(saddle_cgf, 0, order) for order in range(1, 5)]
        expected = [_mp(hand["constrained"]["cumulants"][f"K{index + 1}"]) for index in range(4)]
        errors = [_rel(a, b) for a, b in zip(independent, expected)]
        integer_B, q = 7, _mp("0.23")
        finite = lambda t: integer_B * mp.log(1 - q + q * mp.exp(t))
        finite_expected = [mp.diff(finite, 0, order) for order in range(1, 5)]
        finite_binomial_errors = [_rel(value, integer_B * q if index == 0 else finite_expected[index]) for index, value in enumerate(finite_expected)]
        return {"discrete_cgf_saddle_derivatives": {"relative_errors": [_s(v) for v in errors], "pass": bool(max(errors) < _mp("1e-24"))}, "finite_integer_B_binomial_enumeration": {"relative_errors": [_s(v) for v in finite_binomial_errors], "pass": bool(max(finite_binomial_errors) < _mp("1e-24"))}}


def _control_extreme_acceptance() -> dict[str, Any]:
    with mp.workdps(45):
        V, c = _mp("13"), [_mp("2"), _mp("1.4"), _mp("0.3"), _mp("-0.2")]
        full = _closure(c, [V, V, V, V, V], require_positive=False, qmax=1)
        empty = _closure(c, [V, _mp(0), _mp(0), _mp(0), _mp(0)], require_positive=False, qmax=1)
        full_values = [_mp(full["constrained"]["cumulants"][f"K{i}"]) for i in range(1, 5)]
        empty_values = [_mp(empty["constrained"]["cumulants"][f"K{i}"]) for i in range(1, 5)]
        # Ratios are intentionally undefined at zero denominators, never 0 or NaN.
        undefined_ratios = {"K2_over_K1": None, "K3_over_K2": None, "K4_over_K2": None}
        return {"full_charge_q1": {"cumulants": [_s(v) for v in full_values], "ratios": undefined_ratios, "pass": bool(full_values[0] > 0 and max(abs(v) for v in full_values[1:]) < _mp("1e-35"))}, "empty_charge_q0": {"cumulants": [_s(v) for v in empty_values], "ratios": undefined_ratios, "pass": bool(max(abs(v) for v in empty_values) < _mp("1e-35"))}}


def _all_controls_pass(tree: Mapping[str, Any]) -> bool:
    """Recompute every production gate instead of trusting summary booleans."""
    controls = tree.get("controls", tree)
    if "all_pass" in controls and controls["all_pass"] is False:
        return False
    if not controls.get("uniform_and_poisson", {}).get("uniform_fixed_B_binomial", {}).get("pass", False):
        return False
    if not controls.get("uniform_and_poisson", {}).get("poisson_gce_thinning", {}).get("pass", False):
        return False
    if not controls.get("binary_and_nonuniform", {}).get("binary_coordinate_step_sam_control_only", {}).get("pass", False):
        return False
    if not controls.get("binary_and_nonuniform", {}).get("nonuniform_q_fixed_B_binomial", {}).get("pass", False):
        return False
    if not controls.get("independent_differentiation", {}).get("discrete_cgf_saddle_derivatives", {}).get("pass", False):
        return False
    if not controls.get("independent_differentiation", {}).get("finite_integer_B_binomial_enumeration", {}).get("pass", False):
        return False
    if not controls.get("extreme_acceptance", {}).get("full_charge_q1", {}).get("pass", False):
        return False
    if not controls.get("extreme_acceptance", {}).get("empty_charge_q0", {}).get("pass", False):
        return False
    if "production_row_gates" in controls and not all(controls["production_row_gates"]):
        return False
    if "state_antiparticle_gates" in controls and not all(controls["state_antiparticle_gates"]):
        return False
    for check in controls.get("geometry", {}).values():
        if not (check.get("pass", False) and check.get("denominator_control", {}).get("pass", False) and check.get("i1_angular_average_control", {}).get("pass", False) and check.get("h0_control", {}).get("pass", False)):
            return False
    # A full result carries the actual production rows and live-state gates.
    # This branch is also used by synthetic failure probes without rerunning
    # the expensive thermal build.
    if "predictions" in tree:
        rows = tree.get("predictions", ())
        if len(rows) != 16:
            return False
        for row in rows:
            refinement = row.get("closure_refinement", {})
            variance = row.get("variance_diagnostics", {})
            field_values = refinement.get("fields", {})
            try:
                field_scale_pass = all(_mp(field.get("absolute_error", "NaN")) <= _mp(field.get("tolerance_absolute", "NaN")) for field in field_values.values()) if field_values else False
            except (ProtonBridgeError, TypeError, ValueError):
                field_scale_pass = False
            if len(field_values) != 14 or not refinement.get("pass", False) or not field_scale_pass or not all(field.get("pass", False) for field in field_values.values()) or not variance.get("pass", False):
                return False
            if not variance.get("formula_identity_pass", False) or not variance.get("bound_pass_gce", False) or not variance.get("bound_pass_constrained", False):
                return False
        states = tree.get("state_inputs", ())
        def anti_gate(state: Mapping[str, Any]) -> bool:
            indicator = state.get("antiparticle_indicator", {})
            try:
                density_ok = all(_mp(value) <= _mp(GEOMETRY_RELATIVE_TOL) for value in indicator.get("density_relative_differences", {}).values())
                ratio_ok = _mp(indicator.get("ratio_relative_difference", "NaN")) <= _mp(GEOMETRY_RELATIVE_TOL)
                net_ok = _mp(indicator.get("net_density_closure_relative", "NaN")) <= _mp(GEOMETRY_RELATIVE_TOL)
                abundance_ok = _mp(indicator.get("antiparticle_to_particle_ratio", "NaN")) < _mp(ANTIPARTICLE_THRESHOLD)
            except (ProtonBridgeError, TypeError, ValueError):
                return False
            return bool(indicator.get("pass", False) and indicator.get("density_refinement_pass", False) and indicator.get("net_density_closure_pass", False) and indicator.get("abundance_pass", False) and density_ok and ratio_ok and net_ok and abundance_ok)
        if len(states) != 8 or not all(anti_gate(state) for state in states):
            return False
    return True


def _model_difference(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("gce.K2_over_K1", "gce.K3_over_K2", "gce.K4_over_K2", "constrained.K2_over_K1", "constrained.K3_over_K2", "constrained.K4_over_K2")
    if len(rows) != 2:
        raise ProtonBridgeError("model-difference comparison requires W8 and U16")
    first, second = rows
    delta: dict[str, str] = {}
    relative: dict[str, str] = {}
    for field in fields:
        section, name = field.split(".")
        a, b = _mp(first[section]["ratios"][name]), _mp(second[section]["ratios"][name])
        delta[field] = _s(b - a)
        rel = _relative_actual(a, b)
        relative[field] = _s(rel if rel is not None else 0)
    field_status: dict[str, Any] = {}
    resolved: list[str] = []
    unresolved: list[str] = []
    for field in fields:
        first_refinement = first["closure_refinement"]["fields"][field]
        second_refinement = second["closure_refinement"]["fields"][field]
        a, b = _mp(first[ field.split(".")[0] ]["ratios"][field.split(".")[1]]), _mp(second[ field.split(".")[0] ]["ratios"][field.split(".")[1]])
        scale = max(abs(a), abs(b))
        arithmetic_floor = _mp(DOUBLE_RELATIVE_FLOOR) * scale
        bound = _mp(first_refinement["absolute_error"]) + _mp(second_refinement["absolute_error"]) + arithmetic_floor
        is_resolved = abs(_mp(delta[field])) > bound
        (resolved if is_resolved else unresolved).append(field)
        field_status[field] = {"delta": delta[field], "absolute_model_difference": _s(abs(_mp(delta[field]))), "combined_refinement_bound": _s(bound), "double_arithmetic_floor": _s(arithmetic_floor), "resolved": is_resolved}
    if resolved and not unresolved:
        classification = "resolved_live_difference"
    elif resolved:
        classification = "mixed_resolution"
    else:
        classification = "numerically_unresolved_small_or_zero"
    return {"u16_minus_w8": delta, "relative_difference": relative, "field_status": field_status, "resolved_fields": resolved, "unresolved_fields": unresolved, "classification": classification, "interpretation": "physical-grid model difference compared with each model's actual primary/control refinement bound; no experimental score"}


def _build_live() -> dict[str, Any]:
    # The live thermal producer is called exactly once and its returned object
    # is reused for all eight states.  No saved thermal result is consulted.
    live = thermal.build_result()
    if live.get("status") != thermal.STATUS_PASS or not thermal.validate_result(live):
        raise ProtonBridgeError("live thermal status/validation failed")
    summary = live.get("control_tree_summary", {})
    precision = live.get("precision_control_summary", {})
    if not summary.get("primary_all_controls_pass") or not summary.get("control_all_controls_pass") or not precision.get("all_pass"):
        raise ProtonBridgeError("live thermal controls did not pass")
    source = _source_parameters()
    geometry_primary = {cut: _geometry(cut, GEOMETRY_PRIMARY_ORDER) for cut in RAPIDITY_CUTS}
    geometry_control = {cut: _geometry(cut, GEOMETRY_CONTROL_ORDER) for cut in RAPIDITY_CUTS}
    geometry_checks = {cut: _geometry_controls(geometry_primary[cut], geometry_control[cut]) for cut in RAPIDITY_CUTS}
    if not all(check["pass"] and check["denominator_control"]["pass"] and check["i1_angular_average_control"]["pass"] and check["h0_control"]["pass"] for check in geometry_checks.values()):
        raise ProtonBridgeError("geometry controls failed")
    data = _fixed_data()
    rows: list[dict[str, Any]] = []
    state_inputs: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        for temperature in TEMPERATURES:
            for mu in CHEMICAL_POTENTIALS:
                combo = live["anchors"][anchor][temperature][mu]
                for model_name in MODEL_ORDER:
                    model = combo["models"][model_name]
                    state = model["state"]
                    bulk = _bulk_from_state(state)
                    anti = _particle_antiparticle_indicator(state)
                    if not anti["pass"]:
                        raise ProtonBridgeError("antiparticle abundance invalidates approximation")
                    c = tuple(_mp(bulk["c_density_fm3"][f"c{index}"]) for index in range(1, 5))
                    state_inputs.append({"anchor": anchor, "T_MeV": temperature, "mu_MeV": mu, "model": model_name, "state": {"n_fm3": state["n_fm3"], "nu_MeV": state["nu_MeV"], "mstar_MeV": state["mstar_MeV"], "R21": state["R21"], "R32": state["R32"], "R42": state["R42"]}, "bulk": bulk, "antiparticle_indicator": anti})
                    for cut in RAPIDITY_CUTS:
                        primary_I = geometry_primary[cut].I
                        control_I = geometry_control[cut].I
                        validate_moments(primary_I)
                        primary = _closure(c, primary_I)
                        alternate = _closure(c, control_I)
                        closure_refinement = _closure_refinement(primary, alternate)
                        resolution: dict[str, str] = {}
                        for closure_name in ("gce", "constrained"):
                            for ratio_name in ("K2_over_K1", "K3_over_K2", "K4_over_K2"):
                                a = _mp(primary[closure_name]["ratios"][ratio_name])
                                b = _mp(alternate[closure_name]["ratios"][ratio_name])
                                resolution[f"{closure_name}.{ratio_name}"] = _s(abs(a - b))
                        resolution["pass"] = closure_refinement["pass"]
                        resolution["basis"] = "actual primary/control closure scale plus double arithmetic floor"
                        B = c[0] * _mp(primary_I[0])
                        m1, m2 = _mp(primary_I[1]) / _mp(primary_I[0]), _mp(primary_I[2]) / _mp(primary_I[0])
                        row_data = data["points"][cut]
                        comparison = _data_comparison(primary, row_data)
                        variance_diagnostics = _variance_diagnostics(c, primary_I, row_data, primary)
                        row = {
                            "case_id": f"{model_name}_anchor{anchor}_T{temperature}_mu{mu}_dy{cut}",
                            "anchor": anchor, "T_MeV": temperature, "mu_MeV": mu, "model": model_name, "rapidity_halfwidth": cut,
                            "geometry_moments": {f"I{index}": _s(value) for index, value in enumerate(primary_I)},
                            "geometry_q": {"q_free": _s(source["q_free"]), "qmax_bound": Q_FREE, "m1": _s(m1), "m2": _s(m2)},
                            "bulk": bulk, "B_fixed_to_model_mean": _s(B), "gce": primary["gce"], "constrained": primary["constrained"],
                            "saddle": {"h": primary["saddle_slope_h"], "g_derivatives": primary["g_derivatives"]},
                            "closure_resolution_absolute": resolution,
                            "closure_refinement": closure_refinement,
                            "variance_diagnostics": variance_diagnostics,
                            "data_comparison": comparison,
                            "conditional_approximation": "leading extensive one-baryon-charge saddle; not exact finite-B/SAM-2; no independent charge/energy/momentum constraints",
                        }
                        rows.append(row)
    if len(rows) != 16 or len(state_inputs) != 8:
        raise ProtonBridgeError("wrong live state/cut row count")
    by_case: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault((row["anchor"], row["T_MeV"], row["mu_MeV"], row["rapidity_halfwidth"]), []).append(row)
    model_differences = {"|".join(key): _model_difference(value) for key, value in by_case.items()}
    controls = {
        "uniform_and_poisson": _control_uniform_and_poisson(),
        "binary_and_nonuniform": _control_binary_step_and_nonuniform(),
        "independent_differentiation": _control_independent_differentiation(),
        "extreme_acceptance": _control_extreme_acceptance(),
        "geometry": geometry_checks,
        "production_row_gates": [bool(row["closure_refinement"]["pass"] and row["variance_diagnostics"]["pass"]) for row in rows],
        "state_antiparticle_gates": [bool(state["antiparticle_indicator"]["pass"]) for state in state_inputs],
    }
    provisional = {"controls": controls, "predictions": rows, "state_inputs": state_inputs, "geometry": geometry_checks}
    controls["all_pass"] = bool(_all_controls_pass(provisional))
    if not controls["all_pass"]:
        raise ProtonBridgeError("proton bridge controls failed")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_PASS,
        "protocol_version": "p1-local-cgf-emission-leading-saddle-v1",
        "protocol_provenance": "ADOPTED_P1; fixed external blast-wave source; numerical refinements not fully preregistered; no fit",
        "source_sha256": source_hash,
        "thermal_input_provenance": {"live_rebuild": True, "thermal_schema_version": live["schema_version"], "thermal_status": live["status"], "thermal_source_sha256": live["source_sha256"], "thermal_upstream_source_sha256": live["upstream_source_sha256"], "accepted_grid": {"anchors": list(ANCHORS), "T_MeV": list(TEMPERATURES), "mu_MeV": list(CHEMICAL_POTENTIALS), "models": list(MODEL_ORDER)}, "controls_required": {"primary": summary["primary_all_controls_pass"], "control": summary["control_all_controls_pass"], "precision": precision["all_pass"]}},
        "fixed_external_inputs": {"T_MeV": SOURCE_T_MEV, "R_fm": SOURCE_RADIUS_FM, "H_fm_inv": SOURCE_H_FM_INV, "on_shell_mass_MeV": SOURCE_MASS_MEV, "q_iso": Q_ISO, "q_bound": Q_BOUND, "q_free": Q_FREE, "pT_MeV": [PT_MIN_MEV, PT_MAX_MEV], "rapidity_variable": "Y centered at CM; HADES y0=0.74 is an offset", "cuts": list(RAPIDITY_CUTS), "source_role": "external phenomenological Cooper-Frye reference, not an NVG expansion prediction"},
        "units_and_conventions": {"natural_units": "thermal derivatives are converted to c1=n, c2=T dn/dmu, c3=T^2 d2n/dmu2, c4=T^3 d3n/dmu3 in fm^-3; spatial moments are fm^3", "observable": "net-baryon bulk response mapped conditionally to identified free protons", "q_definition": "q(x)=q_free*p_acceptance(x)", "volume": "dV_eff=gamma*r^2 dr dOmega_x", "mass": "physical/reference on-shell nucleon mass; not NVG medium mass", "kinematic_vs_medium_statistics": "Cooper-Frye acceptance uses the external Boltzmann reference; live bulk densities/derivatives retain their interacting FD convention"},
        "state_inputs": state_inputs,
        "geometry": geometry_checks,
        "predictions": rows,
        "model_difference_controls": model_differences,
        "data": {"role": "observations_only_not_calibration_or_fit_input", "source": data["source"], "acceptance": data["acceptance"], "uncertainty_boundary": data["uncertainty_boundary"], "points": data["points"], "guardrails": data["guardrails"]},
        "controls": controls,
        "interpretation_limits": ["conditional local independent-cell GCE compound-binomial emission plus leading extensive one-baryon-charge saddle; not exact finite-B canonical projection and not SAM-2", "external T,R,H,mass and q values are phenomenological/approximate; no q/H/R/mass or proton-mean fit", "symmetric bulk thermal response is not an Au composition derivation; free-proton marking is an independent bound-nuclei assumption, not coalescence", "anti-baryon ratio is an abundance indicator, not a rigorous high-order error bound; electric charge, energy, momentum, nonlocal correlations and finite-B saddle prefactor are omitted", "corrected HADES observations are not efficiency-thinned again; systematic errors and covariance are missing, not set to zero", "central residuals are descriptive only; no chi-square, significance, exclusion, or NVG confirmation is claimed"],
        "fit_or_score_guardrails": {"parameter_fit": False, "data_used_as_model_input": False, "chi2": False, "significance": False, "experimental_confirmation": False},
    }


_LAST_RESULT_JSON: str | None = None


def build_result() -> dict[str, Any]:
    """Run one complete live bridge build and return a JSON-safe result."""
    global _LAST_RESULT_JSON
    try:
        with mp.workdps(60):
            result = _jsonable(_build_live())
    except Exception as exc:
        result = {"schema_version": SCHEMA_VERSION, "status": STATUS_FAIL, "error": f"{type(exc).__name__}: {exc}"}
    encoded = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
    _LAST_RESULT_JSON = encoded
    return json.loads(encoded)


def validate_result(result: Any) -> bool:
    """Fail closed against the most recent complete live build, never a file."""
    global _LAST_RESULT_JSON
    try:
        if not isinstance(result, dict) or result.get("status") != STATUS_PASS:
            return False
        if _LAST_RESULT_JSON is None:
            build_result()
        return json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False) == _LAST_RESULT_JSON
    except (TypeError, ValueError, OverflowError, json.JSONDecodeError):
        return False


def _write_json(payload: Mapping[str, Any], path: Path | None) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path is not None:
        path.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the live result JSON to this path")
    args = parser.parse_args(argv)
    result = build_result()
    if result.get("status") != STATUS_PASS or not validate_result(result):
        _write_json(result, args.output)
        return 1
    _write_json(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
