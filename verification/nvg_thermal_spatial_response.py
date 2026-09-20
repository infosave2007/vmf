#!/usr/bin/env python3
"""Finite-temperature static spatial baryon response.

This is the deliberately narrow P1 product following the thermal observable
bridge.  The producer rebuilds the eight thermal states (two anchors, two
chemical potentials, and the W8/U16 pair) from the maintained calibration
code, evaluates the vacuum-subtracted finite-temperature Dirac kernel, and
eliminates the full scalar/vector saddle.  The result is a static response,
not an equal-time covariance or a proton-count prediction.

The finite-q radial formula is the accepted angular-integrated Dirac medium
kernel with Fermi--Dirac particle and antiparticle weights.  The logarithmic
principal-value coefficient is subtracted at ``p=q/2`` and its elementary
integral is restored analytically.  At q=0 the exact thermostatics limit is
used.  A small deterministic NumPy Gauss grid is only a numerical quadrature;
it is never a physics input and is checked by refinement in ``build_result``.

The module has no result-cache input and the CLI writes only when an explicit
output path is requested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_nonlinear_calibration_response as nonlinear  # noqa: E402
import nvg_nonlocal_response as cold_response  # noqa: E402
import nvg_spatial_identifiability_audit as spatial  # noqa: E402
import nvg_thermal_observable_bridge as thermal  # noqa: E402


SCHEMA_VERSION = "nvg_thermal_spatial_response.v1"
STATUS_PASS = "PASS_LIVE_THERMAL_SPATIAL_RESPONSE"
STATUS_FAIL = "FAIL_LIVE_THERMAL_SPATIAL_RESPONSE"
EVIDENCE_WEIGHT = "0.0"

ANCHORS = ("0.90", "0.93")
TEMPERATURES = ("70",)
CHEMICAL_POTENTIALS = ("775", "875")
MODEL_ORDER = ("w8", "u16")
SCALES = ("1", "1.3")
Q_GRID = ("0", "50", "100", "200")
DEGENERACY = 4
HBARC_MEV_FM = "197.3269804"
EXPECTED_ROW_COUNT = (
    len(ANCHORS) * len(TEMPERATURES) * len(CHEMICAL_POTENTIALS)
    * len(MODEL_ORDER) * len(SCALES) * len(Q_GRID)
)

# Numerical controls.  The values are fixed protocol inputs, not selected by
# the output.  The primary grid is intentionally modest; the result carries a
# second deterministic refinement and an independent q=0 one-dimensional
# control.  ``numpy.leggauss`` is used only for the finite quadrature.
PRIMARY_ORDER = 120
CONTROL_ORDER = 180
PRIMARY_CUTOFF_MEV = 5000.0
CONTROL_CUTOFF_MEV = 6000.0
THERMAL_DPS = 38
RELATIVE_NUMERIC_LIMIT = 3.0e-5
Q0_THERMO_LIMIT = 3.0e-8
Q_CONTINUITY_LIMIT = 8.0e-3
DIRECT_RESIDUAL_LIMIT = 2.0e-28


class ThermalSpatialResponseError(ValueError):
    """Malformed input or unresolved numerical result."""


def _finite(value: Any) -> bool:
    try:
        return bool(mp.isfinite(value if isinstance(value, mp.mpf) else mp.mpf(str(value))))
    except (TypeError, ValueError, OverflowError):
        return False


def _mp(value: Any, name: str = "value") -> mp.mpf:
    if isinstance(value, bool):
        raise ThermalSpatialResponseError(f"{name} must be a finite real, not bool")
    try:
        out = value if isinstance(value, mp.mpf) else mp.mpf(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ThermalSpatialResponseError(f"{name} must be a finite real") from exc
    if not mp.isfinite(out):
        raise ThermalSpatialResponseError(f"{name} must be finite")
    return out


def _positive(value: Any, name: str) -> mp.mpf:
    out = _mp(value, name)
    if out <= 0:
        raise ThermalSpatialResponseError(f"{name} must be positive")
    return out


def _nonnegative(value: Any, name: str) -> mp.mpf:
    out = _mp(value, name)
    if out < 0:
        raise ThermalSpatialResponseError(f"{name} must be nonnegative")
    return out


def _number(value: Any, digits: int = 20) -> str:
    out = _mp(value, "derived number")
    return mp.nstr(out, digits)


def _relative(left: Any, right: Any, floor: float = 1.0) -> mp.mpf:
    a, b = _mp(left, "left"), _mp(right, "right")
    return abs(a - b) / max(abs(a), abs(b), _mp(floor, "floor"))


def _jsonable(value: Any) -> Any:
    """Convert high precision/internal values into finite JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, mp.matrix):
        return [[_jsonable(value[i, j]) for j in range(value.cols)] for i in range(value.rows)]
    if isinstance(value, mp.mpf):
        return _number(value, 26)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ThermalSpatialResponseError("nonfinite floating output")
        return value
    return value


def _log_integral(q: float, upper: float) -> float:
    """Integral of log| (q+2p)/(q-2p) | from zero to ``upper``."""
    if q <= 0 or upper <= 0:
        return 0.0
    r = 2.0 * upper / q
    if abs(r) < 1.0e-5:
        # The expansion is stable near r=0.
        total = 0.0
        for n in range(100):
            term = 2.0 * r ** (2 * n + 2) / ((2 * n + 1) * (2 * n + 2))
            total += term
            if n and abs(term) < np.finfo(float).eps * max(abs(total), 1.0):
                break
        return q * total / 2.0
    second = 0.0 if r == 1.0 else (1.0 - r) * math.log(abs(1.0 - r))
    return q * ((1.0 + r) * math.log1p(r) + second) / 2.0


@lru_cache(maxsize=32)
def _gauss_nodes(order: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    if not (isinstance(order, int) and order >= 24 and high > low):
        raise ThermalSpatialResponseError("invalid Gauss interval")
    x, w = np.polynomial.legendre.leggauss(order)
    return (np.asarray((low + high) / 2.0 + (high - low) / 2.0 * x),
            np.asarray((high - low) / 2.0 * w))


def _fermi_array(argument: np.ndarray) -> np.ndarray:
    out = np.empty_like(argument, dtype=float)
    high = argument > 40.0
    low = argument < -40.0
    middle = ~(high | low)
    out[high] = np.exp(-argument[high])
    out[low] = 1.0 / (1.0 + np.exp(argument[low]))
    out[middle] = 1.0 / (1.0 + np.exp(argument[middle]))
    return out


def _intervals(cutoff: float, *, q: float = 0.0, m: float = 0.0, nu: float = 0.0, temperature: float = 70.0) -> tuple[tuple[float, float], ...]:
    """Fixed intervals resolve the thermal edge and logarithmic point."""
    points = [0.0, 600.0, 1200.0, 2400.0, float(cutoff)]
    # A declared numerical edge focus; it is not a data-dependent physical
    # split.  At low T this protects the cold-limit control.
    if nu > m:
        pf = math.sqrt(max(nu * nu - m * m, 0.0))
        width = max(30.0, 24.0 * temperature)
        points.extend((pf, max(0.0, pf - width), min(cutoff, pf + width)))
    if q > 0.0:
        points.append(q / 2.0)
    values = sorted(set(max(0.0, min(float(cutoff), p)) for p in points))
    return tuple((a, b) for a, b in zip(values[:-1], values[1:]) if b > a)


def _integral_nodes(fn, cutoff: float, order: int, *, q: float, m: float, nu: float, temperature: float) -> float:
    total = 0.0
    for low, high in _intervals(cutoff, q=q, m=m, nu=nu, temperature=temperature):
        p, w = _gauss_nodes(order, low, high)
        total += float(np.sum(w * fn(p)))
    return total


def _log_weight(weight_fn, q: float, cutoff: float, order: int, *, m: float, nu: float, temperature: float) -> float:
    """PV logarithm with its singular coefficient subtracted/restored."""
    if q <= 0.0:
        raise ThermalSpatialResponseError("log weight requires q>0")
    p0 = q / 2.0
    w0 = float(weight_fn(np.asarray([p0]))[0])
    residual = _integral_nodes(
        lambda p: (weight_fn(p) - w0) * np.log(np.abs((q + 2.0 * p) / (q - 2.0 * p))),
        cutoff, order, q=q, m=m, nu=nu, temperature=temperature,
    )
    return residual + w0 * _log_integral(q, cutoff)


def _kernel_float(mass: Any, nu: Any, q: Any, temperature: Any, *, order: int = PRIMARY_ORDER, cutoff: float = PRIMARY_CUTOFF_MEV) -> tuple[float, float, float]:
    """Finite-T vacuum-subtracted Dirac kernel in MeV².

    ``Pi_vs`` is odd under ``nu -> -nu`` while ``Pi_vv`` and ``Pi_ss`` are
    even.  The scalar negative-energy/seagull term is the ``-p²/E³`` term in
    the q=0 limit and is retained for all finite q by the ``ss0`` radial term.
    """
    m = float(_positive(mass, "mass"))
    n = float(_mp(nu, "nu"))
    qv = float(_nonnegative(q, "q"))
    t = float(_positive(temperature, "temperature"))
    cutoff = float(_positive(cutoff, "cutoff"))
    pref = DEGENERACY / (2.0 * math.pi ** 2)

    def occupations(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        e = np.sqrt(p * p + m * m)
        fp = _fermi_array((e - n) / t)
        fa = _fermi_array((e + n) / t)
        return e, fp + fa, fp - fa

    if qv == 0.0:
        def vv(p: np.ndarray) -> np.ndarray:
            e, we, _ = occupations(p)
            fp = _fermi_array((e - n) / t)
            fa = _fermi_array((e + n) / t)
            return p * p * (fp * (1.0 - fp) + fa * (1.0 - fa)) / t

        def vs(p: np.ndarray) -> np.ndarray:
            e, _, _ = occupations(p)
            fp = _fermi_array((e - n) / t)
            fa = _fermi_array((e + n) / t)
            return p * p * m / e * (fp * (1.0 - fp) - fa * (1.0 - fa)) / t

        def ss(p: np.ndarray) -> np.ndarray:
            e, we, _ = occupations(p)
            fp = _fermi_array((e - n) / t)
            fa = _fermi_array((e + n) / t)
            return p * p * ((m / e) ** 2 * (fp * (1.0 - fp) + fa * (1.0 - fa)) / t - p * p / e ** 3 * we)

        return tuple(pref * _integral_nodes(fn, cutoff, order, q=0.0, m=m, nu=n, temperature=t) for fn in (vv, vs, ss))

    def e_of(p: np.ndarray) -> np.ndarray:
        return np.sqrt(p * p + m * m)

    def even(p: np.ndarray) -> np.ndarray:
        return occupations(p)[1]

    def odd(p: np.ndarray) -> np.ndarray:
        return occupations(p)[2]

    def e0(p: np.ndarray) -> np.ndarray:
        return e_of(p)

    def vv0(p: np.ndarray) -> np.ndarray:
        return even(p) * p * p / e0(p)

    def ss0(p: np.ndarray) -> np.ndarray:
        return -even(p) * p * p / e0(p)

    def vvL(p: np.ndarray) -> np.ndarray:
        return even(p) * p * (e0(p) ** 2 - qv * qv / 4.0) / (e0(p) * qv)

    def vsL(p: np.ndarray) -> np.ndarray:
        return odd(p) * m * p / qv

    def ssL(p: np.ndarray) -> np.ndarray:
        return even(p) * p * (m * m + qv * qv / 4.0) / (e0(p) * qv)
    vv = _integral_nodes(vv0, cutoff, order, q=qv, m=m, nu=n, temperature=t) + _log_weight(vvL, qv, cutoff, order, m=m, nu=n, temperature=t)
    vs = _log_weight(vsL, qv, cutoff, order, m=m, nu=n, temperature=t)
    ss = _integral_nodes(ss0, cutoff, order, q=qv, m=m, nu=n, temperature=t) + _log_weight(ssL, qv, cutoff, order, m=m, nu=n, temperature=t)
    out = tuple(pref * value for value in (vv, vs, ss))
    if not all(math.isfinite(x) for x in out):
        raise ThermalSpatialResponseError("finite-q kernel is nonfinite")
    return out


def finite_temperature_kernel(mass: Any, nu: Any, q: Any, temperature: Any, *, order: int = PRIMARY_ORDER, cutoff: float = PRIMARY_CUTOFF_MEV) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Public high-precision-shaped tuple wrapper around the fixed grid."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 24 or order > 600:
        raise ThermalSpatialResponseError("order must be an integer in [24,600]")
    values = _kernel_float(mass, nu, q, temperature, order=order, cutoff=float(cutoff))
    return tuple(mp.mpf(str(value)) for value in values)


thermal_dirac_kernel = finite_temperature_kernel
_finite_temperature_kernel = finite_temperature_kernel
_finite_t_kernel = finite_temperature_kernel


def _independent_q0_thermo(mass: Any, nu: Any, temperature: Any, *, order: int = CONTROL_ORDER, cutoff: float = CONTROL_CUTOFF_MEV) -> tuple[float, float, float]:
    """Independent q=0 integral written separately from the kernel dispatch."""
    m, n, t = float(_positive(mass, "mass")), float(_mp(nu, "nu")), float(_positive(temperature, "temperature"))
    pref = DEGENERACY / (2.0 * math.pi ** 2)

    def integrand(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        e = np.sqrt(p * p + m * m)
        fp, fa = _fermi_array((e - n) / t), _fermi_array((e + n) / t)
        dp, da = fp * (1.0 - fp) / t, fa * (1.0 - fa) / t
        return (p * p * (dp + da), p * p * m / e * (dp - da), p * p * ((m / e) ** 2 * (dp + da) - p * p / e ** 3 * (fp + fa)))

    vals = []
    for idx in range(3):
        vals.append(pref * _integral_nodes(lambda p, i=idx: integrand(p)[i], cutoff, order, q=0.0, m=m, nu=n, temperature=t))
    return tuple(vals)


def _single_particle_branch_kernel(mass: Any, temperature: Any, q: Any, *, order: int = CONTROL_ORDER, cutoff: float = CONTROL_CUTOFF_MEV) -> tuple[float, float, float]:
    """One-branch (particle only) ν=0 control, independent of dispatch.

    At ν=0 the full finite-T kernel must be exactly twice this result in its
    even VV/SS channels, while the mixed VS channel must vanish.  This control
    is intentionally not obtained by calling ``finite_temperature_kernel``.
    """
    m = float(_positive(mass, "mass"))
    t = float(_positive(temperature, "temperature"))
    qv = float(_nonnegative(q, "q"))
    pref = DEGENERACY / (2.0 * math.pi ** 2)

    def fermi(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = np.sqrt(p * p + m * m)
        f = _fermi_array(e / t)
        return e, f

    if qv == 0.0:
        def vv(p: np.ndarray) -> np.ndarray:
            e, f = fermi(p)
            return p * p * f * (1.0 - f) / t

        def ss(p: np.ndarray) -> np.ndarray:
            e, f = fermi(p)
            return p * p * ((m / e) ** 2 * f * (1.0 - f) / t - p * p / e ** 3 * f)

        return (pref * _integral_nodes(vv, cutoff, order, q=0.0, m=m, nu=0.0, temperature=t),
                0.0,
                pref * _integral_nodes(ss, cutoff, order, q=0.0, m=m, nu=0.0, temperature=t))

    def vv0(p: np.ndarray) -> np.ndarray:
        e, f = fermi(p)
        return f * p * p / e

    def ss0(p: np.ndarray) -> np.ndarray:
        e, f = fermi(p)
        return -f * p * p / e

    def vvL(p: np.ndarray) -> np.ndarray:
        e, f = fermi(p)
        return f * p * (e * e - qv * qv / 4.0) / (e * qv)

    def ssL(p: np.ndarray) -> np.ndarray:
        e, f = fermi(p)
        return f * p * (m * m + qv * qv / 4.0) / (e * qv)

    vv = _integral_nodes(vv0, cutoff, order, q=qv, m=m, nu=0.0, temperature=t) + _log_weight(vvL, qv, cutoff, order, m=m, nu=0.0, temperature=t)
    ss = _integral_nodes(ss0, cutoff, order, q=qv, m=m, nu=0.0, temperature=t) + _log_weight(ssL, qv, cutoff, order, m=m, nu=0.0, temperature=t)
    return pref * vv, 0.0, pref * ss


class CorrelatedScaleModel:
    """Scale only the canonical W0 gradient coefficient, once.

    The homogeneous potential is already expressed in y=W/W0.  Keeping that
    potential and all thermal state inputs unchanged while replacing W0 by
    ``scale*W0`` is the accepted correlated-scale convention.
    """

    def __init__(self, reference: Any, scale: Any):
        self._reference = reference
        self.scale = _positive(scale, "scale")
        self.W0 = reference.W0 * self.scale
        self.lam = reference.lam / self.scale ** 4
        self.A = reference.A
        self.Cs = reference.Cs
        self.Cv = reference.Cv
        self.MN = reference.MN
        self.momega = reference.momega
        self.gomega = reference.gomega
        self.hbarc = reference.hbarc
        self.n0_fm3 = reference.n0_fm3
        self.n0 = reference.n0
        self.d = reference.d
        self.polynomial = getattr(reference, "polynomial", None)

    def potential(self, y: Any):
        return self._reference.potential(y)

    def fermi(self, n: Any, y: Any):
        return self._reference.fermi(n, y)


def _response(model: Any, state: Mapping[str, Any], pi: Sequence[Any], q: Any, scale: Any) -> dict[str, Any]:
    vv, vs, ss = tuple(_mp(x, "polarization") for x in pi)
    if vv <= 0:
        raise ThermalSpatialResponseError("Pi_vv must be positive on a physical branch")
    qv = _nonnegative(q, "q")
    s = _positive(scale, "scale")
    n, y = _positive(state["n_MeV3"], "n"), _positive(state["y"], "y")
    g, mw, mn = _mp(model.gomega), _mp(model.momega), _mp(model.MN)
    A0 = g * n / (mw * mw * y * y)
    coefficients = {
        "a": 1 / vv,
        "b": mn * vs / vv,
        "dferm": mn * mn * (vs * vs / vv - ss),
        "g": g,
        "h": -2 * mw * mw * y * A0,
        "t0": mw * mw * y * y,
        "d0": mn * mn * (vs * vs / vv - ss) + _mp(state["Uyy_MeV4"], "Uyy") - mw * mw * A0 * A0,
        "Pi_vv": vv,
        "Pi_vs": vs,
        "Pi_ss": ss,
        "A0": A0,
    }
    # ``model`` is already the correlated-scale model.  Applying ``scale`` a
    # second time here would turn the adopted Z=s² W0² into s⁴ W0².
    Z = _mp(model.W0) ** 2
    raw = spatial.response_from_hessian(coefficients, Z, qv)
    h = raw["H"]
    direct = raw["direct_solution"]
    residual = None
    condition = None
    if direct is not None:
        rhs = mp.matrix([1, 0, 0])
        vec = mp.matrix(direct)
        residual = max(abs(sum(h[i, j] * vec[j] for j in range(3)) - rhs[i]) for i in range(3))
        try:
            hinv = h ** -1
            norm_h = max(sum(abs(h[i, j]) for j in range(3)) for i in range(3))
            norm_inv = max(sum(abs(hinv[i, j]) for j in range(3)) for i in range(3))
            condition = norm_h * norm_inv
        except (ArithmeticError, ValueError, ZeroDivisionError):
            condition = None
    return {
        "coefficients": coefficients,
        "Z_MeV2": Z,
        "response": raw,
        "C_MeV4": raw["C"],
        "S_MeVminus2": raw["S"],
        "chi_MeV2": raw["chi"],
        "direct_solution": direct,
        "direct_status": raw["direct_status"],
        "direct_residual": residual,
        "direct_condition_inf": condition,
        "raw_vector_curvature_MeV2": -(raw["coefficients"]["t0"] + qv * qv),
        "raw_vector_saddle_negative": bool(raw["coefficients"]["t0"] + qv * qv > 0),
        "status": raw["status"],
        "stable": raw["stable"],
        "q_MeV": qv,
        "scale": s,
        "response_inputs": {
            "MN_MeV": mn,
            "momega_MeV": mw,
            "gomega": g,
            "A0_MeV": A0,
            "Uyy_MeV4": _mp(state["Uyy_MeV4"], "Uyy"),
            "W0_MeV": _mp(model.W0),
        },
    }


def _state_live(model: Any, temperature: mp.mpf, mu: mp.mpf, anchor: mp.mpf) -> dict[str, Any]:
    integrator = thermal._integrator_for(temperature, cutoff="5000", nquad=80)
    roots, attempts = thermal._discover_roots(integrator, model, mu, anchor)
    try:
        selected = thermal._select_root(roots, anchor)
    except thermal.ThermalBridgeError:
        # A discovered unstable branch is retained for response classification;
        # it is never silently replaced by a clipped or absolute curvature.
        selected = min(roots, key=lambda root: (abs(root["y"] - anchor), abs(root["nu"])))
    state = thermal._state(integrator, model, mu, selected["nu"], selected["y"], with_derivatives=False)
    return {"integrator": integrator, "roots": roots, "attempts": attempts, "selected": selected, "state": state}


def _state_public(payload: Mapping[str, Any]) -> dict[str, Any]:
    state = payload["state"]
    selected = payload["selected"]
    return {
        "T_MeV": state["T_MeV"],
        "mu_MeV": state["mu_MeV"],
        "nu_MeV": state["nu_MeV"],
        "y": state["y"],
        "mstar_MeV": state["mstar_MeV"],
        "n_MeV3": state["n_MeV3"],
        "n_fm3": state["n_fm3"],
        "f_nn_MeVminus2": state["f_nn"],
        "N_nu_MeV2": state["N_nu"],
        "S_nu_MeV": state["S_nu"],
        "S_m_MeV2": state["S_m"],
        "Uyy_MeV4": state["Uyy_MeV4"],
        "root_residual_relative": selected["residual_relative"],
        "local_stable_C_positive": state["local_stable_C_positive"],
        "local_stable_fnn_positive": state["local_stable_fnn_positive"],
        "discovered_root_count": len(payload["roots"]),
        "attempted_root_count": len(payload["attempts"]),
        "bounded_root_scan_not_exhaustive": True,
    }


def _state_scale_controls(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("nu_MeV", "y", "n_MeV3", "n_fm3", "Uyy_MeV4")
    errors = {key: _relative(reference["state"][key], candidate["state"][key], floor=1e-30) for key in keys}
    return {"relative_errors": errors, "pass": bool(max(errors.values()) < 3e-20)}


def _row(anchor: str, temperature: str, mu: str, model_name: str, scale: str, q_text: str, base: Mapping[str, Any], candidate_state: Mapping[str, Any], model: Any, scale_model: Any, kernel_cache: dict[tuple[str, ...], tuple[mp.mpf, mp.mpf, mp.mpf]], *, order: int, cutoff: float) -> dict[str, Any]:
    q = float(q_text)
    t = float(temperature)
    state = base["state"]
    mass, nu = float(state["mstar_MeV"]), float(state["nu_MeV"])
    cache_key = (anchor, temperature, mu, model_name, q_text)
    if cache_key not in kernel_cache:
        kernel_cache[cache_key] = finite_temperature_kernel(mass, nu, q, t, order=order, cutoff=cutoff)
    pi = kernel_cache[cache_key]
    q0_key = (anchor, temperature, mu, model_name, "0")
    if q0_key not in kernel_cache:
        kernel_cache[q0_key] = finite_temperature_kernel(mass, nu, 0.0, t, order=order, cutoff=cutoff)
    q0_pi = kernel_cache[q0_key]
    direct = _response(scale_model, candidate_state, pi, q, scale)
    tf = _response(scale_model, candidate_state, q0_pi, q, scale)
    q0 = _response(scale_model, candidate_state, q0_pi, 0.0, scale)
    chi, chi_tf, chi_q0 = direct["chi_MeV2"], tf["chi_MeV2"], q0["chi_MeV2"]
    def ratio(a: Any, b: Any) -> Any:
        return None if a is None or b in (None, 0) else a / b
    return {
        "row_id": f"anchor={anchor}:T={temperature}:mu={mu}:model={model_name}:scale={scale}:q={q_text}",
        "anchor": anchor,
        "T_MeV": temperature,
        "mu_MeV": mu,
        "model": model_name,
        "scale": scale,
        "q_MeV": q_text,
        "state": _state_public(base),
        "scale_state": _state_public({"state": candidate_state, "selected": base["selected"], "roots": base["roots"], "attempts": base["attempts"]}),
        "Pi_vv_MeV2": pi[0],
        "Pi_vs_MeV2": pi[1],
        "Pi_ss_MeV2": pi[2],
        "response_inputs": direct["response_inputs"],
        "Z_MeV2": direct["Z_MeV2"],
        "q0_thermostatic_control": {"Pi_vv_MeV2": q0_pi[0], "Pi_vs_MeV2": q0_pi[1], "Pi_ss_MeV2": q0_pi[2]},
        "C_MeV4": direct["C_MeV4"],
        "S_MeVminus2": direct["S_MeVminus2"],
        "chi_nonlocal_MeV2": chi,
        "chi_TF_same_q_MeV2": chi_tf,
        "chi_q0_MeV2": chi_q0,
        "ratio_nonlocal_to_TF": ratio(chi, chi_tf),
        "ratio_nonlocal_to_q0": ratio(chi, chi_q0),
        "TF_status": tf["status"],
        "q0_status": q0["status"],
        "status": direct["status"],
        "stable": direct["stable"],
        "raw_vector_curvature_MeV2": direct["raw_vector_curvature_MeV2"],
        "raw_vector_saddle_negative": direct["raw_vector_saddle_negative"],
        "direct_status": direct["direct_status"],
        "direct_solution": direct["direct_solution"],
        "raw_hessian": direct["response"]["H"],
        "direct_residual": direct["direct_residual"],
        "direct_condition_inf": direct["direct_condition_inf"],
        "determinant": direct["response"]["determinant"],
        "curvature_controls": {
            "D_MeVminus2": direct["response"]["D"],
            "B_MeV": direct["response"]["B"],
            "C_MeV4": direct["response"]["C"],
            "S_MeVminus2": direct["response"]["S"],
            "physical_static_stability": direct["stable"],
            "raw_3x3_positive_definite_required": False,
        },
        "scale_state_control": base["scale_controls_by_scale"][scale],
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _response_from_row_pi(row: Mapping[str, Any], pi: Sequence[Any], q: Any) -> dict[str, Any]:
    """Rebuild the saddle from a refined kernel without reusing its Pi rows."""
    vv, vs, ss = tuple(_mp(value, "refined polarization") for value in pi)
    if vv <= 0:
        raise ThermalSpatialResponseError("refined Pi_vv must be positive")
    params = row["response_inputs"]
    mn = _mp(params["MN_MeV"], "MN")
    mw = _mp(params["momega_MeV"], "momega")
    g = _mp(params["gomega"], "gomega")
    A0 = _mp(params["A0_MeV"], "A0")
    uyy = _mp(params["Uyy_MeV4"], "Uyy")
    dferm = mn * mn * (vs * vs / vv - ss)
    coefficients = {
        "a": 1 / vv,
        "b": mn * vs / vv,
        "dferm": dferm,
        "g": g,
        "h": -2 * mw * mw * _mp(row["state"]["y"], "y") * A0,
        "t0": mw * mw * _mp(row["state"]["y"], "y") ** 2,
        "d0": dferm + uyy - mw * mw * A0 * A0,
    }
    return spatial.response_from_hessian(coefficients, _mp(row["Z_MeV2"], "Z"), _mp(q, "q"))


def _compute(*, order: int = PRIMARY_ORDER, cutoff: float = PRIMARY_CUTOFF_MEV) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    kernel_cache: dict[Any, tuple[mp.mpf, mp.mpf, mp.mpf]] = {}
    scale_controls: list[dict[str, Any]] = []
    with mp.workdps(THERMAL_DPS):
        for anchor in ANCHORS:
            infos = nonlinear._build_models(anchor)
            for temperature in TEMPERATURES:
                for mu in CHEMICAL_POTENTIALS:
                    for model_name in MODEL_ORDER:
                        model = infos[model_name]["model"]
                        base = _state_live(model, _mp(temperature), _mp(mu), _mp(anchor))
                        scaled_payloads: dict[str, Mapping[str, Any]] = {}
                        for scale in SCALES:
                            scaled_model = CorrelatedScaleModel(model, _mp(scale))
                            candidate_state = thermal._state(
                                base["integrator"], scaled_model, _mp(mu), base["selected"]["nu"], base["selected"]["y"], with_derivatives=False,
                            )
                            scaled_payloads[scale] = candidate_state
                            control = _state_scale_controls(base, {"state": candidate_state})
                            scale_controls.append({"anchor": anchor, "T_MeV": temperature, "mu_MeV": mu, "model": model_name, "scale": scale, **control})
                        base["scale_controls_by_scale"] = {
                            item["scale"]: {"relative_errors": item["relative_errors"], "pass": item["pass"]}
                            for item in scale_controls
                            if item["anchor"] == anchor and item["T_MeV"] == temperature and item["mu_MeV"] == mu and item["model"] == model_name
                        }
                        states.append({"anchor": anchor, "T_MeV": temperature, "mu_MeV": mu, "model": model_name, "state": _state_public(base), "scale_controls": scale_controls[-2:]})
                        for scale in SCALES:
                            scaled_model = CorrelatedScaleModel(model, _mp(scale))
                            for q_text in Q_GRID:
                                # The base state and the candidate state are
                                # mathematically identical; retain the latter
                                # in the row to make the scale boundary visible.
                                row = _row(anchor, temperature, mu, model_name, scale, q_text, base, scaled_payloads[scale], model, scaled_model, kernel_cache, order=order, cutoff=cutoff)
                                rows.append(row)

    if len(rows) != EXPECTED_ROW_COUNT:
        raise ThermalSpatialResponseError(f"expected {EXPECTED_ROW_COUNT} rows, produced {len(rows)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_PASS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "physical_inputs_fixed": True,
        "protocol_provenance": "P1_ADOPTED_AFTER_P0_STATIC_VS_EQUAL_TIME_JUDGMENT",
        "no_saved_results_as_inputs": True,
        "inputs": {
            "anchors": list(ANCHORS),
            "temperature_MeV": list(TEMPERATURES),
            "chemical_potential_MeV": list(CHEMICAL_POTENTIALS),
            "models": list(MODEL_ORDER),
            "correlated_scales": list(SCALES),
            "q_MeV": list(Q_GRID),
            "degeneracy": DEGENERACY,
            "kernel": "vacuum_subtracted_finite_T_Dirac_radial_FD_particle_plus_antiparticle_with_scalar_negative_energy_term",
            "quadrature": {"primary_order": order, "primary_cutoff_MeV": cutoff, "fixed_intervals": [0, 600, 1200, 2400, "cutoff"], "log_pv": "coefficient_subtraction_plus_exact_integral"},
        },
        "coverage": {"row_count": len(rows), "expected_row_count": EXPECTED_ROW_COUNT, "all_rows_present": True, "states": len(states)},
        "states": _jsonable(states),
        "scale_controls": _jsonable(scale_controls),
        "rows": _jsonable(rows),
        "interpretation_limits": [
            "static susceptibility only; T*chi is not asserted to be the equal-time quantum density covariance",
            "medium-subtracted response is not a full renormalized vacuum RPA or all-frequency spectral completion",
            "finite-grid stability is local and is not a global phase-equilibrium or finite-volume proof",
            "the isoscalar symmetric baryon sector does not predict proton acceptance, HADES variance, charge projection, flow, or clusters",
            "the negative raw A0/vector saddle entry is eliminated by Gauss/Schur reduction and is not labelled a physical instability",
        ],
        "source_sha256": {
            "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "thermal_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(),
            "nonlocal_response": hashlib.sha256(cold_response.SOURCE_PATH.read_bytes()).hexdigest(),
        },
    }


def _relative_or_none(left: Any, right: Any) -> float | None:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return float("inf")
    return float(_relative(left, right, floor=1e-30))


def _ratio_value(numerator: Any, denominator: Any) -> Any:
    if numerator is None or denominator in (None, 0, "0"):
        return None
    return _mp(numerator, "ratio numerator") / _mp(denominator, "ratio denominator")


def _response_observable_errors(primary: Mapping[str, Any], refined: Mapping[str, Any], *, q0_primary: Mapping[str, Any], q0_refined: Mapping[str, Any], q: float) -> dict[str, float]:
    p_ratio_tf = _ratio_value(primary["chi"], _response_from_row_pi(primary["row"], q0_primary["pi"], q)["chi"])
    c_ratio_tf = _ratio_value(refined["chi"], _response_from_row_pi(refined["row"], q0_refined["pi"], q)["chi"])
    p_ratio_q0 = _ratio_value(primary["chi"], _response_from_row_pi(primary["row"], q0_primary["pi"], 0.0)["chi"])
    c_ratio_q0 = _ratio_value(refined["chi"], _response_from_row_pi(refined["row"], q0_refined["pi"], 0.0)["chi"])
    errors = {
        "C_MeV4": _relative_or_none(primary["C"], refined["C"]),
        "S_MeVminus2": _relative_or_none(primary["S"], refined["S"]),
        "chi_MeV2": _relative_or_none(primary["chi"], refined["chi"]),
        "ratio_nonlocal_to_TF": _relative_or_none(p_ratio_tf, c_ratio_tf),
        "ratio_nonlocal_to_q0": _relative_or_none(p_ratio_q0, c_ratio_q0),
    }
    return errors


def _add_controls(result: dict[str, Any]) -> dict[str, Any]:
    """Attach independent q0, scale, refinement, branch, and cold controls."""
    controls: dict[str, Any] = {}
    refine_rows, q0_rows, parity_rows, cold_rows, qsmall_rows = [], [], [], [], []
    scale_increment_rows, direct_control_rows = [], []
    row_map = {(r["anchor"], r["T_MeV"], r["mu_MeV"], r["model"], r["scale"], r["q_MeV"]): r for r in result["rows"]}

    for state in result["states"]:
        anchor, t_text, mu_text, model_name = state["anchor"], state["T_MeV"], state["mu_MeV"], state["model"]
        t = float(t_text)
        m, nu = float(state["state"]["mstar_MeV"]), float(state["state"]["nu_MeV"])
        q0_product = row_map[(anchor, t_text, mu_text, model_name, "1", "0")]
        p0 = finite_temperature_kernel(m, nu, 0.0, t, order=PRIMARY_ORDER, cutoff=PRIMARY_CUTOFF_MEV)
        independent = _independent_q0_thermo(m, nu, t)
        pi_err = max(abs(float(p0[i] / independent[i] - 1.0)) for i in range(3))
        f_nn = _mp(state["state"]["f_nn_MeVminus2"], "f_nn")
        s_err = _relative(q0_product["S_MeVminus2"], f_nn, floor=1e-30)
        chi_expected = 1 / f_nn
        chi_err = _relative(q0_product["chi_nonlocal_MeV2"], chi_expected, floor=1e-30)
        q0_rows.append({"anchor": anchor, "T_MeV": t_text, "mu_MeV": mu_text, "model": model_name, "kernel": p0, "independent_thermostatics": independent, "Pi_max_relative_difference": pi_err, "f_nn_MeVminus2": f_nn, "response_S_relative_difference": s_err, "response_chi_relative_difference": chi_err, "pass": bool(pi_err <= Q0_THERMO_LIMIT and s_err <= 2e-7 and chi_err <= 2e-7)})

        for q_text in Q_GRID:
            q = float(q_text)
            p = finite_temperature_kernel(m, nu, q, t, order=PRIMARY_ORDER, cutoff=PRIMARY_CUTOFF_MEV)
            c = finite_temperature_kernel(m, nu, q, t, order=CONTROL_ORDER, cutoff=CONTROL_CUTOFF_MEV)
            row = row_map[(anchor, t_text, mu_text, model_name, "1", q_text)]
            primary = _response_from_row_pi(row, p, q)
            refined = _response_from_row_pi(row, c, q)
            q0p = finite_temperature_kernel(m, nu, 0.0, t, order=PRIMARY_ORDER, cutoff=PRIMARY_CUTOFF_MEV)
            q0c = finite_temperature_kernel(m, nu, 0.0, t, order=CONTROL_ORDER, cutoff=CONTROL_CUTOFF_MEV)
            obs_errors = _response_observable_errors(primary={"row": row, **primary}, refined={"row": row, **refined}, q0_primary={"pi": q0p}, q0_refined={"pi": q0c}, q=q)
            kernel_error = max(abs(float(p[i] / c[i] - 1)) for i in range(3))
            max_error = max([kernel_error, *obs_errors.values()])
            refine_rows.append({"anchor": anchor, "T_MeV": t_text, "mu_MeV": mu_text, "model": model_name, "q_MeV": q_text, "kernel_max_relative_difference": kernel_error, "response_relative_differences": obs_errors, "max_relative_difference": max_error, "pass": bool(max_error <= RELATIVE_NUMERIC_LIMIT)})
        qsmall = finite_temperature_kernel(m, nu, 1.0, t, order=CONTROL_ORDER, cutoff=CONTROL_CUTOFF_MEV)
        qsmall_err = max(abs(float(qsmall[i] / independent[i] - 1.0)) for i in range(3))
        qsmall_rows.append({"anchor": anchor, "T_MeV": t_text, "mu_MeV": mu_text, "model": model_name, "qsmall_MeV": "1", "q0": p0, "qsmall": qsmall, "max_relative_difference": qsmall_err, "pass": bool(qsmall_err <= Q_CONTINUITY_LIMIT)})
        parity = []
        for q in (0.0, 100.0):
            plus = finite_temperature_kernel(m, nu, q, t, order=CONTROL_ORDER, cutoff=CONTROL_CUTOFF_MEV)
            minus = finite_temperature_kernel(m, -nu, q, t, order=CONTROL_ORDER, cutoff=CONTROL_CUTOFF_MEV)
            vv_even = abs(float(plus[0] / minus[0] - 1))
            ss_even = abs(float(plus[2] / minus[2] - 1))
            vs_odd = None if minus[1] == 0 else abs(float(plus[1] / (-minus[1]) - 1))
            parity.append({"q_MeV": _number(q), "vv_even": vv_even, "vs_odd": vs_odd, "ss_even": ss_even, "pass": bool(vv_even < 2e-8 and ss_even < 2e-8 and (vs_odd is None or vs_odd < 2e-8))})
        parity_rows.append({"anchor": anchor, "T_MeV": t_text, "mu_MeV": mu_text, "model": model_name, "rows": parity})

        # The scale test is nonzero-q by construction and uses both scale
        # rows, so q=0 invariance cannot hide a double application of s.
        for q_text in ("50", "100", "200"):
            one = row_map[(anchor, t_text, mu_text, model_name, "1", q_text)]
            scaled = row_map[(anchor, t_text, mu_text, model_name, "1.3", q_text)]
            q = float(q_text)
            w0 = _mp(one["response_inputs"]["W0_MeV"], "W0")
            expected_increment = (_mp("1.3") ** 2 - 1) * w0 ** 2 * q ** 2
            actual_increment = _mp(scaled["C_MeV4"], "scaled C") - _mp(one["C_MeV4"], "base C")
            increment_error = _relative(actual_increment, expected_increment, floor=1e-20)
            scale_increment_rows.append({"anchor": anchor, "T_MeV": t_text, "mu_MeV": mu_text, "model": model_name, "q_MeV": q_text, "expected_C_increment_MeV4": expected_increment, "actual_C_increment_MeV4": actual_increment, "relative_difference": increment_error, "pass": bool(increment_error <= 2e-9)})

        if not cold_rows:
            cold_m, cold_nu, cold_q = 700.0, 760.0, 100.0
            low = finite_temperature_kernel(cold_m, cold_nu, cold_q, 2.0, order=CONTROL_ORDER, cutoff=CONTROL_CUTOFF_MEV)
            kf = math.sqrt(cold_nu * cold_nu - cold_m * cold_m)
            old = cold_response.polarization_kernel(str(cold_m), str(kf), str(cold_q), d=4, dps=50)
            cold_err = max(abs(float(low[i] / old[i] - 1)) for i in range(3))
            cold_rows.append({"mass_MeV": cold_m, "nu_MeV": cold_nu, "T_MeV": 2.0, "q_MeV": cold_q, "thermal_kernel": low, "old_cold_kernel": old, "max_relative_difference": cold_err, "pass": bool(cold_err <= 2.0e-2)})

    # Independent direct solves are evaluated for every row and both scales.
    for row in result["rows"]:
        h = np.asarray(row["raw_hessian"], dtype=float)
        try:
            expected = np.linalg.solve(h, np.array([1.0, 0.0, 0.0]))
            given = np.asarray([float(value) for value in row["direct_solution"]], dtype=float) if row["direct_solution"] is not None else None
            error = None if given is None else float(np.max(np.abs(expected - given)) / max(np.max(np.abs(expected)), 1.0))
            valid = bool(given is not None and np.all(np.isfinite(expected)) and error <= 3e-11 and row["direct_status"] == "DIRECT_SOLVE_OK")
        except (np.linalg.LinAlgError, TypeError, ValueError):
            error, valid = None, False
        direct_control_rows.append({"row_id": row["row_id"], "scale": row["scale"], "relative_difference": error, "pass": valid})

    controls["quadrature_refinement"] = {"rows": refine_rows, "pass": bool(all(row["pass"] for row in refine_rows))}
    controls["q0_thermostatic_match"] = {"rows": q0_rows, "pass": bool(all(row["pass"] for row in q0_rows))}
    controls["qsmall_continuity"] = {"rows": qsmall_rows, "pass": bool(all(row["pass"] for row in qsmall_rows))}
    controls["charge_conjugation_parity"] = {"rows": parity_rows, "pass": bool(all(item["pass"] for row in parity_rows for item in row["rows"]))}
    controls["low_temperature_old_cold_kernel"] = {"rows": cold_rows, "pass": bool(all(row["pass"] for row in cold_rows))}
    direct_nulls = sum(row["direct_solution"] is None for row in result["rows"])
    controls["direct_solve"] = {"row_count": len(direct_control_rows), "null_count": direct_nulls, "failed_count": sum(not row["pass"] for row in direct_control_rows), "max_residual": max(float(row["direct_residual"]) for row in result["rows"] if row["direct_residual"] is not None), "all_ok": bool(direct_control_rows and direct_nulls == 0 and all(row["pass"] for row in direct_control_rows))}
    controls["independent_direct_solve_both_scales"] = {"rows": direct_control_rows, "pass": bool(direct_control_rows and all(row["pass"] for row in direct_control_rows))}
    controls["scale_invariance"] = {"all_pass": bool(all(row["pass"] for row in result["scale_controls"]))}
    controls["nonzero_q_scale_increment"] = {"rows": scale_increment_rows, "pass": bool(all(row["pass"] for row in scale_increment_rows))}

    # ν=0 branch decomposition is independent of the full dispatcher: both
    # even channels must be two single-branch integrals and VS must vanish.
    anti_rows = []
    for q in (0.0, 100.0):
        full = finite_temperature_kernel(500.0, 0.0, q, 70.0, order=CONTROL_ORDER, cutoff=CONTROL_CUTOFF_MEV)
        single = _single_particle_branch_kernel(500.0, 70.0, q)
        vv_error = abs(float(full[0] / (2 * single[0]) - 1))
        ss_error = abs(float(full[2] / (2 * single[2]) - 1))
        vs_abs = abs(float(full[1]))
        anti_rows.append({"q_MeV": _number(q), "full_particle_plus_antiparticle": full, "single_particle_branch": single, "vv_two_branch_relative_difference": vv_error, "ss_two_branch_relative_difference": ss_error, "vs_at_nu0": vs_abs, "omitted_antiparticle_vv_relative_difference": abs(float(full[0] / single[0] - 1)), "pass": bool(vv_error < 2e-8 and ss_error < 2e-8 and vs_abs < 2e-9 and abs(float(full[0] / single[0] - 1)) > 0.5)})
    controls["antiparticle_material_control"] = {"mass_MeV": 500.0, "nu_MeV": 0.0, "T_MeV": 70.0, "rows": anti_rows, "pass": bool(all(row["pass"] for row in anti_rows))}

    sample_state = result["states"][0]["state"]
    sm, sn, st = float(sample_state["mstar_MeV"]), float(sample_state["nu_MeV"]), float(sample_state["T_MeV"])
    pref = DEGENERACY / (2.0 * math.pi ** 2)
    def scalar_I(p: np.ndarray) -> np.ndarray:
        e = np.sqrt(p * p + sm * sm)
        fp, fa = _fermi_array((e - sn) / st), _fermi_array((e + sn) / st)
        return p ** 4 / e ** 3 * (fp + fa)
    omitted_I = pref * _integral_nodes(scalar_I, PRIMARY_CUTOFF_MEV, CONTROL_ORDER, q=0.0, m=sm, nu=sn, temperature=st)
    full_ss = float(result["rows"][0]["Pi_ss_MeV2"])
    scalar_pass = bool(omitted_I > 0 and abs(omitted_I) > 1e-6)
    controls["scalar_negative_energy_control"] = {"full_Pi_ss_MeV2": full_ss, "omitted_I_MeV2": omitted_I, "difference_if_omitted_MeV2": omitted_I, "detected_and_rejected": scalar_pass, "pass": scalar_pass}

    synthetic = []
    for name, coeffs, expected_status in (
        ("unstable_scalar", {"a": "1", "b": "0", "d0": "-1", "g": "1", "h": "0", "t0": "1"}, "UNSTABLE_SCALAR_CURVATURE"),
        ("singular_scalar", {"a": "1", "b": "0", "d0": "0", "g": "1", "h": "0", "t0": "1"}, "SINGULAR_SCALAR_CURVATURE"),
    ):
        probe = spatial.response_from_hessian(coeffs, "4", "0")
        synthetic.append({"name": name, "status": probe["status"], "expected_status": expected_status, "chi_is_null": probe["chi"] is None, "direct_status": probe["direct_status"], "pass": bool(probe["status"] == expected_status and probe["chi"] is None)})
    controls["synthetic_branch_controls"] = {"rows": synthetic, "pass": bool(all(row["pass"] for row in synthetic))}
    controls["status_counts"] = {status: sum(1 for row in result["rows"] if row["status"] == status) for status in sorted({row["status"] for row in result["rows"]})}
    result["controls"] = _jsonable(controls)
    result["all_controls_pass"] = bool(all(value.get("pass", value.get("all_pass", value.get("all_ok", False))) for value in controls.values() if isinstance(value, dict) and any(key in value for key in ("pass", "all_pass", "all_ok"))))
    return result


def build_result() -> dict[str, Any]:
    result = _compute()
    result = _add_controls(result)
    result["status"] = STATUS_PASS if result["all_controls_pass"] else STATUS_FAIL
    result["integrity_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str) and value.lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        return False
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    return True


def _gate_values(controls: Mapping[str, Any]) -> list[bool]:
    values = []
    for control in controls.values():
        if not isinstance(control, Mapping):
            continue
        for key in ("pass", "all_pass", "all_ok"):
            if key in control:
                values.append(bool(control[key]))
                break
    return values


def validate_result(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    if not _finite_tree(result):
        return False
    if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != STATUS_PASS:
        return False
    if result.get("evidence_weight") != EVIDENCE_WEIGHT or not result.get("physical_inputs_fixed") or not result.get("no_saved_results_as_inputs"):
        return False
    coverage = result.get("coverage", {})
    rows = result.get("rows", [])
    if coverage.get("row_count") != EXPECTED_ROW_COUNT or coverage.get("expected_row_count") != EXPECTED_ROW_COUNT or coverage.get("states") != len(ANCHORS) * len(TEMPERATURES) * len(CHEMICAL_POTENTIALS) * len(MODEL_ORDER) or len(rows) != EXPECTED_ROW_COUNT or not coverage.get("all_rows_present"):
        return False
    expected_keys = {(a, t, mu, model, scale, q) for a in ANCHORS for t in TEMPERATURES for mu in CHEMICAL_POTENTIALS for model in MODEL_ORDER for scale in SCALES for q in Q_GRID}
    actual_keys = {(row.get("anchor"), row.get("T_MeV"), row.get("mu_MeV"), row.get("model"), row.get("scale"), row.get("q_MeV")) for row in rows if isinstance(row, Mapping)}
    if actual_keys != expected_keys or len({row.get("row_id") for row in rows if isinstance(row, Mapping)}) != EXPECTED_ROW_COUNT:
        return False
    allowed_statuses = {"STABLE_STATIC_TF_RESPONSE", "STABLE_STATIC_NONLOCAL_RESPONSE", "UNSTABLE_SCALAR_CURVATURE", "UNSTABLE_DENSITY_RESPONSE", "SINGULAR_SCALAR_CURVATURE", "SINGULAR_DENSITY_RESPONSE", "SINGULAR_FULL_HESSIAN"}
    observable_fields = ("Pi_vv_MeV2", "Pi_vs_MeV2", "Pi_ss_MeV2", "C_MeV4", "S_MeVminus2", "chi_nonlocal_MeV2", "chi_TF_same_q_MeV2", "chi_q0_MeV2", "direct_residual", "direct_condition_inf", "determinant")
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") not in allowed_statuses:
            return False
        if any(value is not None and not _finite(value) for key in observable_fields if (value := row.get(key)) is not None):
            return False
        matrix = row.get("raw_hessian")
        if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(line, list) or len(line) != 3 or any(not _finite(x) for x in line) for line in matrix):
            return False
        if row.get("stable") and (row.get("chi_nonlocal_MeV2") is None or row.get("S_MeVminus2") is None or row.get("direct_solution") is None):
            return False
    controls = result.get("controls")
    required_controls = ("quadrature_refinement", "q0_thermostatic_match", "qsmall_continuity", "charge_conjugation_parity", "low_temperature_old_cold_kernel", "direct_solve", "independent_direct_solve_both_scales", "scale_invariance", "nonzero_q_scale_increment", "antiparticle_material_control", "scalar_negative_energy_control", "synthetic_branch_controls")
    if not isinstance(controls, Mapping) or any(name not in controls for name in required_controls) or any(not isinstance(controls[name], Mapping) for name in required_controls):
        return False
    gates = _gate_values(controls)
    if not gates or not all(gates) or bool(result.get("all_controls_pass")) != all(gates):
        return False
    if len(controls["q0_thermostatic_match"].get("rows", [])) != 8 or len(controls["nonzero_q_scale_increment"].get("rows", [])) != 24:
        return False
    source_hashes = result.get("source_sha256", {})
    expected_hashes = {
        "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thermal_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(),
        "nonlocal_response": hashlib.sha256(cold_response.SOURCE_PATH.read_bytes()).hexdigest(),
    }
    if source_hashes != expected_hashes:
        return False
    digest = result.get("integrity_sha256")
    payload = dict(result)
    payload.pop("integrity_sha256", None)
    if not isinstance(digest, str) or hashlib.sha256(_canonical_json(payload)).hexdigest() != digest:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = _jsonable(build_result())
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0 if payload.get("status") == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
