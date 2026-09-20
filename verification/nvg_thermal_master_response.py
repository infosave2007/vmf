#!/usr/bin/env python3
"""Finite-temperature six-current Dirac master response and thermal cut.

The producer is intentionally bounded.  It rebuilds the live W8/U16 thermal
states and evaluates the complete positive/negative-energy projector sum for
z=0 and upper-half-plane probes.  The same four-variable Hartree saddle is
then solved directly and by its longitudinal Schur reduction.  The analytic
spacelike cut is a separate bare-polarization API; it is not an imaginary
part obtained by adding a regulator to the coupled response.

No saved result is read as an input.  Complex z values are analytic probes;
only the dedicated cut API evaluates a real positive frequency.  This module
does not infer a positive all-frequency spectral measure, a collision width,
or an equal-time covariance from a finite Matsubara sample.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import mpmath as mp
from numpy.polynomial.legendre import leggauss
from scipy.integrate import IntegrationWarning, quad

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Maintained public live constructors.  No run-workspace or saved-result input
# is imported here.
import nvg_nonlinear_calibration_response as nonlinear  # noqa: E402
import nvg_retarded_response as hessian  # noqa: E402
import nvg_thermal_observable_bridge as thermal  # noqa: E402
import nvg_thermal_spatial_response as spatial  # noqa: E402


SCHEMA_VERSION = "nvg_thermal_master_response.v1"
STATUS_PASS = "PASS_LIVE_THERMAL_MASTER_RESPONSE"
STATUS_FAIL = "FAIL_LIVE_THERMAL_MASTER_RESPONSE"
EVIDENCE_WEIGHT = 0.0

ANCHORS = ("0.90", "0.93")
TEMPERATURES = ("70",)
CHEMICAL_POTENTIALS = ("775", "875")
MODEL_ORDER = ("w8", "u16")
SCALES = ("1", "1.3")
Q_GRID = ("100", "200")
Z_LABELS = ("z0", "iT", "i2piT", "i4piT", "spacelike")
FREQUENCY_FRACTIONS = ("0.25", "0.75")
DEGENERACY = 4
EXPECTED_STATE_COUNT = len(ANCHORS) * len(TEMPERATURES) * len(CHEMICAL_POTENTIALS) * len(MODEL_ORDER)
EXPECTED_RESPONSE_ROW_COUNT = EXPECTED_STATE_COUNT * len(SCALES) * len(Q_GRID) * len(Z_LABELS)
EXPECTED_CUT_ROW_COUNT = EXPECTED_STATE_COUNT * len(Q_GRID) * len(FREQUENCY_FRACTIONS)
# Compatibility aliases useful to focused callers.
EXPECTED_ROW_COUNT = EXPECTED_RESPONSE_ROW_COUNT
EXPECTED_CUT_COUNT = EXPECTED_CUT_ROW_COUNT

PRIMARY_P_ORDER = 160
PRIMARY_U_ORDER = 120
CONTROL_P_ORDER = 220
CONTROL_U_ORDER = 180
PRIMARY_CUTOFF_MEV = 5000.0
CONTROL_CUTOFF_MEV = 6000.0
THERMAL_DPS = 38
KERNEL_RELATIVE_LIMIT = 3.0e-5
WARD_RELATIVE_LIMIT = 2.0e-8
DIRECT_RELATIVE_LIMIT = 3.0e-8


class ThermalMasterResponseError(ValueError):
    """Malformed input or an unresolved numerical/physics gate."""


# ---------------------------------------------------------------------------
# Small numeric and serialization helpers


def _real(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ThermalMasterResponseError(f"{name} must be a finite real scalar")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ThermalMasterResponseError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(out):
        raise ThermalMasterResponseError(f"{name} must be finite")
    if positive and out <= 0:
        raise ThermalMasterResponseError(f"{name} must be positive")
    if nonnegative and out < 0:
        raise ThermalMasterResponseError(f"{name} must be nonnegative")
    return out


def _complex_frequency(value: Any, name: str = "z") -> complex:
    if isinstance(value, bool):
        raise ThermalMasterResponseError(f"{name} must be complex or real")
    try:
        z = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ThermalMasterResponseError(f"{name} must be finite complex") from exc
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        raise ThermalMasterResponseError(f"{name} must be finite complex")
    # Real nonzero frequencies are intentionally reserved for thermal_cut_imag.
    if abs(z.imag) <= 1.0e-14 and abs(z.real) > 1.0e-14:
        raise ThermalMasterResponseError("real nonzero frequency belongs to the analytic cut API")
    if z.imag < -1.0e-14:
        raise ThermalMasterResponseError("z must be zero or in the upper half-plane")
    return 0j if abs(z) <= 1.0e-14 else z


def _relative(left: Any, right: Any, floor: float = 1.0) -> float:
    a, b = complex(left), complex(right)
    return float(abs(a - b) / max(abs(a), abs(b), floor))


def _cjson(value: Any) -> dict[str, float]:
    z = complex(value)
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        raise ThermalMasterResponseError("nonfinite complex output")
    # Numeric real/imag fields keep the artifact strict JSON and easy to audit.
    return {"real": float(z.real), "imag": float(z.imag)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, mp.mpf):
        x = float(value)
        if not math.isfinite(x):
            raise ThermalMasterResponseError("nonfinite mpmath output")
        return x
    if isinstance(value, mp.mpc):
        return _cjson(complex(value))
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, complex):
        return _cjson(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ThermalMasterResponseError("nonfinite floating output")
        return value
    return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str) and value.lower() in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity"}:
        return False
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    return True


# ---------------------------------------------------------------------------
# Complete six-current projector kernel


@lru_cache(maxsize=128)
def _gauss_nodes(order: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(order, int) or isinstance(order, bool) or order < 24 or high <= low:
        raise ThermalMasterResponseError("invalid Gauss interval")
    x, w = leggauss(order)
    return ((low + high) / 2.0 + (high - low) / 2.0 * x,
            (high - low) / 2.0 * w)


def _fermi_array(argument: np.ndarray) -> np.ndarray:
    out = np.empty_like(argument, dtype=float)
    high = argument > 40.0
    low = argument < -40.0
    middle = ~(high | low)
    out[high] = np.exp(-argument[high])
    out[low] = 1.0 / (1.0 + np.exp(argument[low]))
    out[middle] = 1.0 / (1.0 + np.exp(argument[middle]))
    return out


def _p_intervals(cutoff: float, *, mass: float, nu: float, temperature: float, q: float) -> tuple[tuple[float, float], ...]:
    # Fixed, declared splits resolve the thermal edge and q/2 angular geometry.
    points = [0.0, 600.0, 1200.0, 2400.0, cutoff, q / 2.0]
    if nu > mass:
        pf = math.sqrt(max(nu * nu - mass * mass, 0.0))
        width = max(30.0, 24.0 * temperature)
        points.extend((pf, max(0.0, pf - width), min(cutoff, pf + width)))
    values = sorted(set(max(0.0, min(cutoff, float(point))) for point in points))
    return tuple((lo, hi) for lo, hi in zip(values[:-1], values[1:]) if hi > lo)


def _same_band_fermi_divided_difference(
    energy_p: np.ndarray,
    energy_k: np.ndarray,
    *,
    s: int,
    nu: float,
    temperature: float,
) -> np.ndarray:
    """Stable ``[δN_s(p)-δN_s(k)]/[s(Ek-Ep)]``.

    The explicit band sign cancels between ``δN_s=s f`` and the denominator.
    The midpoint identity

    ``[f(a-h)-f(a+h)]/(2 T h) = sinh(h)/(2 T h [cosh(a)+cosh(h)])``

    is used instead of subtracting two occupations.  It is well behaved at
    q→0 and has the same positive ``f(1-f)/T`` limit for both bands.
    """
    h = (energy_k - energy_p) / (2.0 * temperature)
    midpoint = (energy_k + energy_p) / (2.0 * temperature) - s * nu / temperature
    abs_h = np.abs(h)
    abs_midpoint = np.abs(midpoint)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        # Work entirely in logarithms.  Computing sinh(h)/h and the cosh sum
        # separately turns a finite low-temperature response into inf/inf.
        # The large-|h| branch is log(sinhc(h)) with the leading exponential
        # removed analytically; the small branch retains the even Taylor
        # series through h^6.
        log_sinhc = np.empty_like(abs_h, dtype=float)
        small = abs_h < 1.0e-4
        h2 = abs_h[small] * abs_h[small]
        log_sinhc[small] = np.log1p(h2 / 6.0 + h2 * h2 / 120.0 + h2 * h2 * h2 / 5040.0)
        large_h = abs_h[~small]
        log_sinhc[~small] = (
            large_h - np.log(2.0 * large_h) + np.log1p(-np.exp(-2.0 * large_h))
        )

        # log(cosh(a)+cosh(h)) with a scaled two-term log-sum-exp.  No
        # exponentiation of the common scale is performed.
        scale = np.maximum(abs_midpoint, abs_h)
        scaled_cosh_sum = 0.5 * (
            np.exp(abs_midpoint - scale) * (1.0 + np.exp(-2.0 * abs_midpoint))
            + np.exp(abs_h - scale) * (1.0 + np.exp(-2.0 * abs_h))
        )
        log_cosh_sum = scale + np.log(scaled_cosh_sum)
        log_ratio = log_sinhc - math.log(2.0 * temperature) - log_cosh_sum
        ratio = np.exp(log_ratio)
    if not np.all(np.isfinite(log_ratio)) or not np.all(np.isfinite(ratio)):
        raise ThermalMasterResponseError("same-band Fermi divided difference is nonfinite")
    # A genuinely exponentially small result may underflow to numerical zero;
    # unlike the old implementation, no nonfinite intermediate is silently
    # converted into that physical-looking value.
    return ratio


def _divided_difference(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    s: int,
    t: int,
    energy_p: np.ndarray,
    energy_k: np.ndarray,
    nu: float,
    temperature: float,
    zero_frequency: bool,
) -> np.ndarray:
    """Stable [deltaN_s(p)-deltaN_t(k)]/[tEk-sEp-z]."""
    if zero_frequency and s == t:
        return _same_band_fermi_divided_difference(
            energy_p, energy_k, s=s, nu=nu, temperature=temperature,
        ).astype(complex)
    return numerator / denominator


def _master_kernel_float(
    mass: Any,
    nu: Any,
    q: Any,
    temperature: Any,
    z: Any,
    *,
    p_order: int = PRIMARY_P_ORDER,
    u_order: int = PRIMARY_U_ORDER,
    cutoff: float = PRIMARY_CUTOFF_MEV,
    branches: Iterable[tuple[int, int]] = ((1, 1), (1, -1), (-1, 1), (-1, -1)),
) -> np.ndarray:
    """Return (vv,vs,ss,vL,sL,LL) in MeV² from the complete projector sum."""
    m = _real(mass, "mass", positive=True)
    n = _real(nu, "nu")
    qv = _real(q, "q", nonnegative=True)
    t = _real(temperature, "temperature", positive=True)
    cutoff = _real(cutoff, "cutoff", positive=True)
    zv = _complex_frequency(z)
    branches = tuple((int(s), int(tt)) for s, tt in branches)
    if not branches:
        raise ThermalMasterResponseError("at least one projector branch is required")
    if qv == 0.0:
        if zv == 0j:
            # Grand-canonical static limit.  This is the exact q=0 limit of
            # the thermodynamic action, not a dynamic conserved-mode limit.
            x, w = _gauss_nodes(p_order, 0.0, cutoff)
            p = x
            e = np.sqrt(p * p + m * m)
            fp, fa = _fermi_array((e - n) / t), _fermi_array((e + n) / t)
            dp, da = fp * (1.0 - fp) / t, fa * (1.0 - fa) / t
            pref = DEGENERACY / (2.0 * math.pi ** 2)
            values = np.asarray((
                np.sum(w * p * p * (dp + da)),
                np.sum(w * p * p * m / e * (dp - da)),
                np.sum(w * p * p * ((m / e) ** 2 * (dp + da) - p * p / e ** 3 * (fp + fa))),
                0.0,
                0.0,
                0.0,
            ), dtype=complex)
            return pref * values
        # Number conservation kills vv/vs at q=0 and nonzero z.  The remote
        # scalar pair channel remains.  The longitudinal-current square does
        # not vanish: angular averaging of V_LL gives (1-p²/(3E²)).
        x, w = _gauss_nodes(p_order, 0.0, cutoff)
        p = x
        e = np.sqrt(p * p + m * m)
        fp, fa = _fermi_array((e - n) / t), _fermi_array((e + n) / t)
        occupation = fp + fa
        pair = -2.0 * DEGENERACY / (math.pi ** 2) * np.sum(w * p ** 4 * occupation / (e * (4.0 * e * e - zv * zv)))
        longitudinal = -2.0 * DEGENERACY / (math.pi ** 2) * np.sum(
            w * p * p * e * occupation * (1.0 - p * p / (3.0 * e * e)) / (4.0 * e * e - zv * zv)
        )
        return np.asarray((0j, 0j, complex(pair), 0j, 0j, complex(longitudinal)), dtype=complex)

    x_u, w_u = _gauss_nodes(u_order, -1.0, 1.0)
    total = np.zeros(6, dtype=complex)
    for low, high in _p_intervals(cutoff, mass=m, nu=n, temperature=t, q=qv):
        x_p, w_p = _gauss_nodes(p_order, low, high)
        p = x_p[:, None]
        u = x_u[None, :]
        weights = w_p[:, None] * w_u[None, :] * p * p
        pz = p * u
        kz = pz + qv
        dot = p * p + p * qv * u
        ek2 = p * p + qv * qv + 2.0 * p * qv * u
        ep = np.sqrt(m * m + p * p)
        ek = np.sqrt(m * m + ek2)
        for s, tt in branches:
            dn_p = s * _fermi_array((ep - s * n) / t)
            for branch_t in (tt,):
                dn_k = branch_t * _fermi_array((ek - branch_t * n) / t)
                denominator = branch_t * ek - s * ep - zv
                numerator = dn_p - dn_k
                ratio = _divided_difference(
                    numerator, denominator, s=s, t=branch_t, energy_p=ep, energy_k=ek,
                    nu=n, temperature=t,
                    zero_frequency=(zv == 0j and s == branch_t),
                )
                st = s * branch_t
                vv = (1.0 + st * (dot + m * m) / (ep * ek)) / 2.0
                vs = (s * m / ep + branch_t * m / ek) / 2.0
                ss = (1.0 + st * (m * m - dot) / (ep * ek)) / 2.0
                vl = (s * pz / ep + branch_t * kz / ek) / 2.0
                sl = st * m * (pz + kz) / (2.0 * ep * ek)
                ll = (1.0 + st * (2.0 * pz * kz - dot - m * m) / (ep * ek)) / 2.0
                for idx, vertex in enumerate((vv, vs, ss, vl, sl, ll)):
                    total[idx] += np.sum(weights * ratio * vertex)
    out = DEGENERACY / (4.0 * math.pi ** 2) * total
    if not np.all(np.isfinite(out)):
        raise ThermalMasterResponseError("six-current kernel is nonfinite")
    return out


def finite_temperature_master_kernel(
    mass: Any,
    nu: Any,
    q: Any,
    temperature: Any,
    z: Any = 0j,
    *,
    p_order: int = PRIMARY_P_ORDER,
    u_order: int = PRIMARY_U_ORDER,
    order: int | None = None,
    cutoff: float = PRIMARY_CUTOFF_MEV,
) -> tuple[complex, complex, complex, complex, complex, complex]:
    """Evaluate the complete finite-T six-current analytic kernel.

    ``z=0`` is the static grand-canonical limit.  Every nonzero accepted
    probe has Im(z)>0; a real positive-frequency response is deliberately not
    accepted here and must use :func:`thermal_cut_imag`.
    """
    if order is not None:
        if isinstance(order, bool) or not isinstance(order, int):
            raise ThermalMasterResponseError("order must be an integer")
        p_order = u_order = order
    if p_order < 24 or u_order < 24 or p_order > 600 or u_order > 600:
        raise ThermalMasterResponseError("quadrature order must be in [24,600]")
    values = _master_kernel_float(mass, nu, q, temperature, z, p_order=p_order, u_order=u_order, cutoff=cutoff)
    return tuple(complex(value) for value in values)  # type: ignore[return-value]


# Names used by external focused controls.
master_kernel = finite_temperature_master_kernel
thermal_dirac_master_kernel = finite_temperature_master_kernel


# ---------------------------------------------------------------------------
# Analytic spacelike thermal cut


def _fermi_scalar(x: float) -> float:
    if x > 40.0:
        e = math.exp(-x)
        return e / (1.0 + e)
    if x < -40.0:
        e = math.exp(x)
        return 1.0 / (1.0 + e)
    return 1.0 / (1.0 + math.exp(x))


def _fd_difference(
    E: float,
    nu: float,
    temperature: float,
    omega: float,
    sign: int,
    *,
    allow_underflow: bool = False,
) -> float:
    """Stable ``f(x)-f(x+d)`` without subtracting nearby occupations.

    ``f(x) f(-x-d) [1-exp(-d)]`` is positive for every finite E and omega>0.
    Tail samples beyond floating-point range are allowed only inside the
    convergent infinity-tail quadrature; the thermal-edge probe below refuses
    an all-underflow API call rather than reporting physical zero absorption.
    """
    x = (E - sign * nu) / temperature
    d = omega / temperature
    if d <= 0.0 or not math.isfinite(d):
        raise ThermalMasterResponseError("omega/temperature must be finite positive")
    factor = -math.expm1(-d)
    fx = _fermi_scalar(x)
    complement = _fermi_scalar(-x - d)
    value = fx * complement * factor
    if value == 0.0 and not allow_underflow:
        raise ThermalMasterResponseError(
            "finite-T Fermi difference underflow; use a larger T or inspect a log-tail"
        )
    return value


def thermal_cut_imag(
    mass: Any,
    nu: Any,
    q: Any,
    omega: Any,
    temperature: Any,
    *,
    epsrel: float = 2.0e-8,
    epsabs: float = 1.0e-80,
    _return_details: bool = False,
) -> tuple[float, float, float] | tuple[tuple[float, float, float], tuple[float, float, float], float, bool]:
    """Bare finite-T spacelike Im(Pi_vv,Pi_vs,Pi_ss), no physical eta.

    The integral is over the exact on-shell threshold ``Emin`` to infinity.
    Thermal-edge breakpoints at ``|nu|-omega`` and ``|nu|`` are retained when
    they lie in the domain; the infinity tail is a numerical integration
    limit, not a new physical UV cutoff.
    """
    m = _real(mass, "mass", positive=True)
    n = _real(nu, "nu")
    qv = _real(q, "q", positive=True)
    w = _real(omega, "omega", positive=True)
    t = _real(temperature, "temperature", positive=True)
    if not (0.0 < w < qv):
        raise ThermalMasterResponseError("analytic thermal cut requires 0<omega<q")
    if epsrel <= 0 or epsabs <= 0 or not math.isfinite(epsrel) or not math.isfinite(epsabs):
        raise ThermalMasterResponseError("integration tolerances must be positive finite")
    threshold = 0.5 * (qv * math.sqrt(1.0 + 4.0 * m * m / (qv * qv - w * w)) - w)

    # The lower endpoint can lie deep inside a filled Fermi sea, where both
    # occupations round to one and the local difference is below floating
    # point even though the absorption peak is well resolved later.  Probe the
    # largest occupation difference near the thermal edge instead of treating
    # Emin alone as the whole API domain.  For a negative chemical potential
    # the same point probes the antiparticle edge through sign=-1.
    probe_energy = max(threshold, abs(n) - w / 2.0)
    probe_values = tuple(
        _fd_difference(probe_energy, n, t, w, sign, allow_underflow=True)
        for sign in (1, -1)
    )
    if not any(value > 0.0 for value in probe_values):
        raise ThermalMasterResponseError(
            "finite-T cut is below floating-point range across its threshold/edge probe; no physical zero claimed"
        )

    # Split at the exact particle/antiparticle thermal edges and at their
    # midpoint.  The finite intervals resolve the transition; only the final
    # [last, infinity) interval is a numerical tail where zero occupation
    # values are harmlessly expected.
    edge_points = [threshold]
    for edge in (abs(n) - w, abs(n) - w / 2.0, abs(n)):
        if threshold < edge and math.isfinite(edge):
            edge_points.append(edge)
    edge_points = sorted(set(edge_points))
    intervals = tuple(
        (lo, hi) for lo, hi in zip(edge_points, edge_points[1:]) if hi > lo
    ) + ((edge_points[-1], math.inf),)
    underflow_in_finite_domain = [False]
    underflow_in_tail = [False]

    def integrand(E: float, channel: int, *, numerical_tail: bool) -> float:
        dp = _fd_difference(E, n, t, w, 1, allow_underflow=True)
        da = _fd_difference(E, n, t, w, -1, allow_underflow=True)
        if dp == 0.0 or da == 0.0:
            if numerical_tail:
                underflow_in_tail[0] = True
            else:
                underflow_in_finite_domain[0] = True
        summed, difference = dp + da, dp - da
        if channel == 0:
            return summed * (E * (E + w) + (w * w - qv * qv) / 4.0)
        if channel == 1:
            return difference * m * (E + w / 2.0)
        return summed * (m * m + (qv * qv - w * w) / 4.0)

    pref = DEGENERACY / (4.0 * math.pi * qv)
    values: list[float] = []
    errors: list[float] = []
    for channel in range(3):
        channel_value = 0.0
        channel_error = 0.0
        for low, high in intervals:
            numerical_tail = math.isinf(high)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IntegrationWarning)
                value, error = quad(
                    lambda energy, c=channel, tail=numerical_tail: integrand(energy, c, numerical_tail=tail),
                    low, high, epsabs=epsabs, epsrel=epsrel, limit=400,
                )
            if any(issubclass(item.category, IntegrationWarning) for item in caught):
                raise ThermalMasterResponseError("analytic cut integration did not converge")
            if not (math.isfinite(value) and math.isfinite(error)):
                raise ThermalMasterResponseError("analytic cut integration failed")
            channel_value += value
            channel_error += error
        values.append(pref * channel_value)
        errors.append(pref * channel_error)
    if not any(value > 0.0 for value in values):
        if underflow_in_finite_domain[0] or underflow_in_tail[0]:
            raise ThermalMasterResponseError(
                "analytic cut underflowed throughout its declared domain; no zero absorption claimed"
            )
        raise ThermalMasterResponseError("analytic cut returned no finite positive channel")
    result = tuple(values)  # type: ignore[assignment]
    if _return_details:
        return result, tuple(errors), threshold, bool(underflow_in_tail[0] and not underflow_in_finite_domain[0])  # type: ignore[return-value]
    return result  # type: ignore[return-value]


analytic_thermal_cut = thermal_cut_imag
finite_temperature_cut = thermal_cut_imag


def _thermal_smearing_old_cold_cut(
    mass: float,
    nu: float,
    q: float,
    omega: float,
    temperature: float,
    *,
    epsrel: float = 2.0e-8,
    epsabs: float = 1.0e-80,
) -> tuple[float, float, float]:
    """Independent finite-T smearing of the maintained cold irreducible Pi.

    The convolution is over a fictitious zero-temperature Fermi energy and
    never touches the dressed Hartree susceptibility.  It is a control only;
    the production cut above integrates the thermal on-shell formula directly.
    """
    m = _real(mass, "mass", positive=True)
    n = _real(nu, "nu")
    qv = _real(q, "q", positive=True)
    w = _real(omega, "omega", positive=True)
    t = _real(temperature, "temperature", positive=True)
    if not (0.0 < w < qv):
        raise ThermalMasterResponseError("cold-cut smearing requires 0<omega<q")

    def fermi_derivative(fermi_energy: float) -> float:
        x = (fermi_energy - n) / t
        if abs(x) > 40.0:
            e = math.exp(-abs(x))
            return e / (t * (1.0 + e) ** 2)
        f = 1.0 / (1.0 + math.exp(x))
        return f * (1.0 - f) / t

    values: list[float] = []
    for channel in range(3):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            value, error = quad(
                lambda fermi_energy: fermi_derivative(fermi_energy)
                * hessian.analytic_ph_imag(
                    m, math.sqrt(max(fermi_energy * fermi_energy - m * m, 0.0)), qv, w, d=DEGENERACY
                )[channel],
                m, math.inf, epsabs=epsabs, epsrel=epsrel, limit=400,
            )
        if any(issubclass(item.category, IntegrationWarning) for item in caught) or not math.isfinite(error):
            raise ThermalMasterResponseError("cold-cut thermal-smearing control did not converge")
        if not math.isfinite(value):
            raise ThermalMasterResponseError("cold-cut thermal-smearing control is nonfinite")
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Live states and Hartree saddle


def _build_live_backgrounds() -> list[dict[str, Any]]:
    backgrounds: list[dict[str, Any]] = []
    with mp.workdps(THERMAL_DPS):
        for anchor in ANCHORS:
            infos = nonlinear._build_models(anchor)
            for temperature in TEMPERATURES:
                for mu in CHEMICAL_POTENTIALS:
                    for model_name in MODEL_ORDER:
                        model = infos[model_name]["model"]
                        integrator = thermal._integrator_for(
                            temperature, cutoff=str(int(PRIMARY_CUTOFF_MEV)), nquad=80,
                        )
                        roots, attempts = thermal._discover_roots(
                            integrator, model, mp.mpf(mu), mp.mpf(anchor),
                        )
                        if not roots or not any(root["stable"] for root in roots):
                            raise ThermalMasterResponseError("bounded thermal root scan has no stable branch")
                        selected = thermal._select_root(roots, mp.mpf(anchor))
                        state = thermal._state(
                            integrator, model, mp.mpf(mu), selected["nu"], selected["y"], with_derivatives=False,
                        )
                        if not bool(state["local_stable_C_positive"] and state["local_stable_fnn_positive"]):
                            raise ThermalMasterResponseError("selected thermal root is not locally stable")
                        backgrounds.append({
                            "state_id": f"anchor={anchor}:T={temperature}:mu={mu}:model={model_name}",
                            "anchor": anchor, "T_MeV": temperature, "mu_MeV": mu,
                            "model_name": model_name, "model": model, "state": state,
                            "selected": selected, "roots": roots, "attempts": attempts,
                        })
    if len(backgrounds) != EXPECTED_STATE_COUNT:
        raise ThermalMasterResponseError("live state coverage is incomplete")
    return backgrounds


def _state_number(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ThermalMasterResponseError("nonfinite live state")
    return out


def _public_state(background: Mapping[str, Any]) -> dict[str, Any]:
    state = background["state"]
    selected = background["selected"]
    return {
        "state_id": background["state_id"],
        "anchor": background["anchor"], "T_MeV": background["T_MeV"],
        "mu_MeV": background["mu_MeV"], "model": background["model_name"],
        "nu_MeV": _state_number(state["nu_MeV"]), "y": _state_number(state["y"]),
        "mstar_MeV": _state_number(state["mstar_MeV"]), "n_MeV3": _state_number(state["n_MeV3"]),
        "n_fm3": _state_number(state["n_fm3"]), "Uyy_MeV4": _state_number(state["Uyy_MeV4"]),
        "f_nn_MeVminus2": _state_number(state["f_nn"]),
        "root_residual_relative": _state_number(selected["residual_relative"]),
        "local_stable_C_positive": bool(state["local_stable_C_positive"]),
        "local_stable_fnn_positive": bool(state["local_stable_fnn_positive"]),
        "discovered_root_count": len(background["roots"]),
        "attempted_root_count": len(background["attempts"]),
        "bounded_root_scan_not_exhaustive": True,
    }


def _coefficients(background: Mapping[str, Any], scale: float, pi: Sequence[complex]) -> tuple[dict[str, Any], float]:
    model = background["model"]
    state = background["state"]
    vv, vs, ss = (complex(pi[index]) for index in range(3))
    if abs(vv) == 0 or not all(np.isfinite(value) for value in (vv, vs, ss)):
        raise ThermalMasterResponseError("density/scalar kernel is singular or nonfinite")
    n = _state_number(state["n_MeV3"])
    y = _state_number(state["y"])
    mw, mn, g = float(model.momega), float(model.MN), float(model.gomega)
    A0 = g * n / (mw * mw * y * y)
    h = -2.0 * mw * mw * y * A0
    t0 = mw * mw * y * y
    dferm = mn * mn * (vs * vs / vv - ss)
    d0 = dferm + _state_number(state["Uyy_MeV4"]) - mw * mw * A0 * A0
    W0 = float(model.W0) * scale
    coeff = {
        "a": 1.0 / vv, "b": mn * vs / vv, "d0": d0,
        "dferm": dferm, "g": g, "h": h, "t0": t0,
        "A0": A0, "Pi_vv": vv, "Pi_vs": vs, "Pi_ss": ss,
        "_n0": n, "_W0": W0,
    }
    return coeff, W0 * W0


def _response(background: Mapping[str, Any], scale: float, q: float, z: complex, pi: Sequence[complex]) -> dict[str, Any]:
    coeff, Z = _coefficients(background, scale, pi)
    try:
        answer = hessian.response_from_hessian(coeff, Z, q, z)
    except Exception as exc:
        raise ThermalMasterResponseError(f"Hartree Hessian evaluation failed: {exc}") from exc
    return answer


def _q0_longitudinal_1d(
    mass: Any,
    nu: Any,
    temperature: Any,
    z: Any,
    *,
    order: int = CONTROL_P_ORDER,
    cutoff: float = CONTROL_CUTOFF_MEV,
) -> complex:
    """Independent q=0 angular average for the LL pair channel."""
    m = _real(mass, "mass", positive=True)
    n = _real(nu, "nu")
    t = _real(temperature, "temperature", positive=True)
    zv = _complex_frequency(z)
    if zv == 0j:
        raise ThermalMasterResponseError("q=0 dynamic LL control requires nonzero z")
    p, w = _gauss_nodes(order, 0.0, _real(cutoff, "cutoff", positive=True))
    e = np.sqrt(p * p + m * m)
    fsum = _fermi_array((e - n) / t) + _fermi_array((e + n) / t)
    return complex(-2.0 * DEGENERACY / (math.pi ** 2) * np.sum(
        w * p * p * e * fsum * (1.0 - p * p / (3.0 * e * e)) / (4.0 * e * e - zv * zv)
    ))


def _q0_longitudinal_angular(
    mass: Any,
    nu: Any,
    temperature: Any,
    z: Any,
    *,
    p_order: int = CONTROL_P_ORDER,
    u_order: int = CONTROL_U_ORDER,
    cutoff: float = CONTROL_CUTOFF_MEV,
) -> complex:
    """Raw q=0 projector integral, retained as an independent control."""
    m = _real(mass, "mass", positive=True)
    n = _real(nu, "nu")
    t = _real(temperature, "temperature", positive=True)
    zv = _complex_frequency(z)
    if zv == 0j:
        raise ThermalMasterResponseError("q=0 dynamic angular control requires nonzero z")
    p, wp = _gauss_nodes(p_order, 0.0, _real(cutoff, "cutoff", positive=True))
    u, wu = _gauss_nodes(u_order, -1.0, 1.0)
    pp = p[:, None]
    uu = u[None, :]
    ee = np.sqrt(pp * pp + m * m)
    weight = wp[:, None] * wu[None, :] * pp * pp
    total = np.zeros_like(weight, dtype=complex)
    for s, tt in ((1, -1), (-1, 1)):
        dn_s = s * _fermi_array((ee - s * n) / t)
        dn_t = tt * _fermi_array((ee - tt * n) / t)
        denominator = tt * ee - s * ee - zv
        pz = pp * uu
        ll = (1.0 + s * tt * (2.0 * pz * pz - pp * pp - m * m) / (ee * ee)) / 2.0
        total += weight * (dn_s - dn_t) / denominator * ll
    return complex(DEGENERACY / (4.0 * math.pi ** 2) * np.sum(total))


def _z_for(label: str, q: float, temperature: float) -> complex:
    if label == "z0":
        return 0j
    if label == "iT":
        return 1j * temperature
    if label == "i2piT":
        return 1j * 2.0 * math.pi * temperature
    if label == "i4piT":
        return 1j * 4.0 * math.pi * temperature
    if label == "spacelike":
        return complex(0.6 * q, 0.2 * q)
    raise ThermalMasterResponseError(f"unknown z label {label}")


def _matrix_json(matrix: Any) -> list[list[dict[str, float]]]:
    return [[_cjson(complex(matrix[i, j])) for j in range(matrix.shape[1])] for i in range(matrix.shape[0])]


def _answer_public(answer: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("chi", "direct_chi", "D", "B", "C", "S", "F", "L", "determinant")
    out: dict[str, Any] = {
        "status": answer.get("status"), "regular": bool(answer.get("regular")),
        "direct_status": answer.get("direct_status"),
        "physical_stability_assessed": bool(answer.get("physical_stability_assessed", False)),
        "direct_relative_error": None if answer.get("direct_relative_error") is None else float(answer["direct_relative_error"]),
        "normalization_coordinate_scales": [float(item) for item in answer.get("coordinate_scales", ())],
        "determinant_identity_relative_error": float(answer.get("determinant_identity_relative_error", 0.0)),
        "determinant_residual": float(answer.get("determinant_residual", 0.0)),
    }
    for field in fields:
        out[field] = None if answer.get(field) is None else _cjson(answer[field])
    if answer.get("H") is not None:
        out["H"] = _matrix_json(np.asarray(answer["H"], dtype=complex))
    else:
        out["H"] = None
    direct = answer.get("direct_solution")
    out["direct_solution"] = None if direct is None else [_cjson(item) for item in direct]
    return out


def _channel_public(pi: Sequence[complex]) -> dict[str, dict[str, float]]:
    names = ("vv_MeV2", "vs_MeV2", "ss_MeV2", "vL_MeV2", "sL_MeV2", "LL_MeV2")
    return {name: _cjson(value) for name, value in zip(names, pi)}


def _ward_residual(pi: Sequence[complex], q: float, z: complex) -> tuple[dict[str, dict[str, float]], float]:
    vv, vs, _ss, vl, sl, ll = map(complex, pi)
    residuals = (q * vl - z * vv, q * sl - z * vs, q * ll - z * vl)
    scale = max(1.0, abs(q * vl), abs(z * vv), abs(q * sl), abs(z * vs), abs(q * ll), abs(z * vl))
    relative = max(abs(value) for value in residuals) / scale
    return ({"q_vL_minus_z_vv": _cjson(residuals[0]), "q_sL_minus_z_vs": _cjson(residuals[1]), "q_LL_minus_z_vL": _cjson(residuals[2])}, float(relative))


def _make_response_row(background: Mapping[str, Any], scale_text: str, q_text: str, z_label: str, pi: Sequence[complex]) -> dict[str, Any]:
    scale = float(scale_text)
    q = float(q_text)
    temperature = float(background["T_MeV"])
    z = _z_for(z_label, q, temperature)
    answer = _response(background, scale, q, z, pi)
    ward, ward_relative = _ward_residual(pi, q, z)
    return {
        "row_id": f"{background['state_id']}:scale={scale_text}:q={q_text}:z={z_label}",
        "state_id": background["state_id"], "anchor": background["anchor"],
        "T_MeV": background["T_MeV"], "mu_MeV": background["mu_MeV"],
        "model": background["model_name"], "scale": scale_text, "q_MeV": q_text,
        "z_label": z_label, "z_MeV": _cjson(z),
        "frequency_interpretation": "static grand-canonical limit" if z_label == "z0" else "upper-half-plane analytic probe; not a real oscillation frequency",
        "Pi": _channel_public(pi), "Ward": ward, "Ward_max_relative": ward_relative,
        "response": _answer_public(answer),
        "status": "MASTER_RESPONSE_OFF_POLE" if answer.get("regular") else str(answer.get("status")),
        "negative_vector_saddle_retained": True,
        "gradient_scale_applied_once": True,
    }


def _cut_row(background: Mapping[str, Any], q_text: str, fraction_text: str) -> dict[str, Any]:
    q = float(q_text)
    fraction = float(fraction_text)
    omega = fraction * q
    state = background["state"]
    pi, errors, threshold_from_integrator, underflow = thermal_cut_imag(
        state["mstar_MeV"], state["nu_MeV"], q, omega, state["T_MeV"],
        _return_details=True,
    )
    threshold = threshold_from_integrator
    return {
        "row_id": f"{background['state_id']}:q={q_text}:omega_over_q={fraction_text}",
        "state_id": background["state_id"], "anchor": background["anchor"],
        "T_MeV": background["T_MeV"], "mu_MeV": background["mu_MeV"],
        "model": background["model_name"], "q_MeV": q_text,
        "omega_over_q": fraction_text, "omega_MeV": omega,
        "Emin_MeV": threshold, "Pi_absorptive_Im_MeV2": {
            "vv": pi[0], "vs": pi[1], "ss": pi[2],
        },
        "integration_error_estimate_MeV2": {"vv": errors[0], "vs": errors[1], "ss": errors[2]},
        "underflow_encountered_only_in_numerical_tail": underflow,
        "integration_domain": "E from exact on-shell Emin to infinity; infinity is numerical tail integration, not a physical cutoff",
        "finite_eta": False, "bare_polarization_only": True,
        "status": "FINITE_T_SPACELIKE_ABSORPTION",
    }


# ---------------------------------------------------------------------------
# Controls and artifact assembly


def _background_lookup(backgrounds: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(item["state_id"]): item for item in backgrounds}


def _pi_from_row(row: Mapping[str, Any]) -> tuple[complex, ...]:
    names = ("vv_MeV2", "vs_MeV2", "ss_MeV2", "vL_MeV2", "sL_MeV2", "LL_MeV2")
    return tuple(complex(row["Pi"][name]["real"], row["Pi"][name]["imag"]) for name in names)


def _state_static_control(background: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    state = background["state"]
    m, n, t, q = state["mstar_MeV"], state["nu_MeV"], state["T_MeV"], float(row["q_MeV"])
    live = tuple(complex(value) for value in spatial.finite_temperature_kernel(m, n, q, t))
    pi = _pi_from_row(row)
    kernel_errors = [_relative(pi[index], live[index]) for index in range(3)]
    static_answer = _response(background, float(row["scale"]), q, 0j, live)
    candidate = complex(row["response"]["chi"]["real"], row["response"]["chi"]["imag"])
    reference = complex(static_answer["chi"])
    return {
        "state_id": background["state_id"], "q_MeV": row["q_MeV"],
        "Pi_relative_errors": kernel_errors,
        "chi_relative_error": _relative(candidate, reference),
        "pass": bool(max(kernel_errors + [_relative(candidate, reference)]) < 3.0e-5),
    }


def _add_controls(result: dict[str, Any], backgrounds: Sequence[Mapping[str, Any]], kernel_cache: Mapping[tuple[str, str, str], tuple[complex, ...]]) -> dict[str, Any]:
    rows = result["response_rows"]
    row_map = {(row["state_id"], row["scale"], row["q_MeV"], row["z_label"]): row for row in rows}
    controls: dict[str, Any] = {}
    lookup = _background_lookup(backgrounds)

    # Static bridge: all eight states and both q values, against the maintained
    # radial thermal producer and a newly rebuilt Hessian response.
    static_rows = []
    for background in backgrounds:
        for scale_text in SCALES:
            for q_text in Q_GRID:
                static_rows.append(_state_static_control(background, row_map[(background["state_id"], scale_text, q_text, "z0")]))
    controls["static_limit_bridge"] = {"rows": static_rows, "pass": bool(static_rows and all(item["pass"] for item in static_rows))}

    # q=0 order of limits: static grand-canonical thermodynamics versus
    # nonzero-frequency number conservation with scalar pair channel retained.
    q0_rows = []
    for background in backgrounds:
        state = background["state"]
        static_pi = finite_temperature_master_kernel(state["mstar_MeV"], state["nu_MeV"], 0.0, state["T_MeV"], 0j)
        static_answer = _response(background, 1.0, 0.0, 0j, static_pi)
        dynamic_z = 1j * float(state["T_MeV"])
        dynamic_pi = finite_temperature_master_kernel(state["mstar_MeV"], state["nu_MeV"], 0.0, state["T_MeV"], dynamic_z)
        independent_ll = _q0_longitudinal_1d(state["mstar_MeV"], state["nu_MeV"], state["T_MeV"], dynamic_z)
        angular_ll = _q0_longitudinal_angular(state["mstar_MeV"], state["nu_MeV"], state["T_MeV"], dynamic_z)
        qsmall_pi = finite_temperature_master_kernel(
            state["mstar_MeV"], state["nu_MeV"], 1.0e-3, state["T_MeV"], dynamic_z,
            p_order=CONTROL_P_ORDER, u_order=CONTROL_U_ORDER, cutoff=CONTROL_CUTOFF_MEV,
        )
        qsmall_density_abs = [abs(qsmall_pi[index]) for index in (0, 1)]
        qsmall_errors = [_relative(qsmall_pi[index], dynamic_pi[index], floor=1.0e-20) for index in (2, 5)]
        s_err = _relative(static_answer["S"], state["f_nn"], floor=1e-24)
        q0_rows.append({
            "state_id": background["state_id"],
            "static_S_vs_fnn_relative_error": s_err,
            "dynamic_Pi_vv_abs": abs(dynamic_pi[0]), "dynamic_Pi_vs_abs": abs(dynamic_pi[1]),
            "dynamic_scalar_pair_abs": abs(dynamic_pi[2]),
            "dynamic_LL": _cjson(dynamic_pi[5]), "independent_1d_LL": _cjson(independent_ll),
            "independent_angular_LL": _cjson(angular_ll),
            "LL_vs_1d_relative_error": _relative(dynamic_pi[5], independent_ll, floor=1e-20),
            "angular_vs_1d_relative_error": _relative(angular_ll, independent_ll, floor=1e-20),
            "qsmall_dynamic_density_abs": qsmall_density_abs,
            "qsmall_dynamic_scalar_LL_relative_errors": qsmall_errors,
            "pass": bool(
                s_err < 3e-6 and abs(dynamic_pi[0]) < 1e-10 and abs(dynamic_pi[1]) < 1e-10
                and abs(dynamic_pi[2]) > 1e-14 and _relative(dynamic_pi[5], independent_ll, floor=1e-20) < 3e-11
                and _relative(angular_ll, independent_ll, floor=1e-20) < 3e-11
                and max(qsmall_density_abs) < 1.0e-5 and max(qsmall_errors) < 3e-3
            ),
        })
    controls["q0_order_of_limits"] = {"rows": q0_rows, "pass": bool(q0_rows and all(item["pass"] for item in q0_rows))}

    ward_rows = [{"row_id": row["row_id"], "Ward_max_relative": row["Ward_max_relative"], "pass": row["Ward_max_relative"] < WARD_RELATIVE_LIMIT} for row in rows]
    controls["raw_six_current_Ward"] = {"row_count": len(ward_rows), "max_relative": max(item["Ward_max_relative"] for item in ward_rows), "rows": ward_rows, "pass": bool(ward_rows and all(item["pass"] for item in ward_rows))}

    # Particle-hole-only (+,+) control: it is deliberately not a production
    # path and must fail the raw Ward gate.
    probe = backgrounds[0]
    probe_state = probe["state"]
    ph_pi = tuple(_master_kernel_float(
        probe_state["mstar_MeV"], probe_state["nu_MeV"], 100.0, probe_state["T_MeV"], 1j * probe_state["T_MeV"],
        p_order=CONTROL_P_ORDER, u_order=CONTROL_U_ORDER, cutoff=CONTROL_CUTOFF_MEV, branches=((1, 1),),
    ))
    _ph_residual, ph_relative = _ward_residual(ph_pi, 100.0, 1j * float(probe_state["T_MeV"]))
    controls["particle_hole_only_negative_control"] = {
        "state_id": probe["state_id"], "Ward_max_relative": ph_relative,
        "detected_and_rejected": bool(ph_relative > 1.0e-4),
        "pass": bool(ph_relative > 1.0e-4),
    }

    # Charge conjugation and material/antimaterial controls at fixed declared
    # numerical inputs, including nu=0 and +/-70 MeV.
    parity_rows = []
    for n in (0.0, 70.0):
        plus = finite_temperature_master_kernel(700.0, n, 100.0, 70.0, 1j * 70.0, p_order=CONTROL_P_ORDER, u_order=CONTROL_U_ORDER, cutoff=CONTROL_CUTOFF_MEV)
        minus = finite_temperature_master_kernel(700.0, -n, 100.0, 70.0, 1j * 70.0, p_order=CONTROL_P_ORDER, u_order=CONTROL_U_ORDER, cutoff=CONTROL_CUTOFF_MEV)
        parity_rows.append({
            "nu_pair_MeV": [n, -n],
            "vv_even": _relative(plus[0], minus[0]), "ss_even": _relative(plus[2], minus[2]),
            "vs_odd": _relative(plus[1], -minus[1]), "sL_odd": _relative(plus[4], -minus[4]),
            "pass": bool(_relative(plus[0], minus[0]) < 3e-7 and _relative(plus[2], minus[2]) < 3e-7 and _relative(plus[1], -minus[1]) < 3e-7 and _relative(plus[4], -minus[4]) < 3e-7),
        })
    controls["charge_conjugation_antimaterial"] = {"rows": parity_rows, "pass": bool(all(item["pass"] for item in parity_rows))}

    z_probe = 60.0 + 20.0j
    z_conj = -z_probe.conjugate()
    freq_plus = finite_temperature_master_kernel(700.0, 70.0, 100.0, 70.0, z_probe, p_order=CONTROL_P_ORDER, u_order=CONTROL_U_ORDER, cutoff=CONTROL_CUTOFF_MEV)
    freq_minus = finite_temperature_master_kernel(700.0, 70.0, 100.0, 70.0, z_conj, p_order=CONTROL_P_ORDER, u_order=CONTROL_U_ORDER, cutoff=CONTROL_CUTOFF_MEV)
    expected_frequency = (np.conjugate(freq_plus[0]), np.conjugate(freq_plus[1]), np.conjugate(freq_plus[2]), -np.conjugate(freq_plus[3]), -np.conjugate(freq_plus[4]), np.conjugate(freq_plus[5]))
    freq_errors = [_relative(freq_minus[i], expected_frequency[i]) for i in range(6)]
    controls["frequency_conjugation"] = {"z": _cjson(z_probe), "minus_conjugate_z": _cjson(z_conj), "current_channel_sign": "vL,sL are odd under z->-z*; vv,vs,ss,LL are even", "relative_errors": freq_errors, "pass": bool(max(freq_errors) < 3e-7)}

    # Positive Matsubara samples are reported only as a diagnostic.  They do
    # not promote this finite grid to a covariance or a positivity theorem.
    matsubara_rows = []
    for row in rows:
        if row["scale"] == "1" and row["z_label"] in {"iT", "i2piT", "i4piT"}:
            chi = row["response"]["chi"]
            matsubara_rows.append({"row_id": row["row_id"], "chi": chi, "real_positive": chi["real"] > 0.0, "imag_abs": abs(chi["imag"])})
    controls["Matsubara_positive_sample_diagnostic"] = {
        "rows": matsubara_rows, "all_real_positive": bool(matsubara_rows and all(item["real_positive"] for item in matsubara_rows)),
        "not_a_global_positivity_or_covariance_proof": True,
    }

    # Direct 4x4 solve versus Schur response is checked for every production
    # row; no singular row is silently dropped.
    direct_rows = [{"row_id": row["row_id"], "relative_error": row["response"]["direct_relative_error"], "direct_status": row["response"]["direct_status"], "pass": row["response"]["direct_relative_error"] is not None and row["response"]["direct_relative_error"] < DIRECT_RELATIVE_LIMIT} for row in rows]
    controls["direct_4x4_vs_schur"] = {"rows": direct_rows, "max_relative_error": max(item["relative_error"] for item in direct_rows if item["relative_error"] is not None), "pass": bool(direct_rows and all(item["pass"] for item in direct_rows))}

    scale_rows = []
    for background in backgrounds:
        for q_text in Q_GRID:
            for z_label in Z_LABELS:
                one = row_map[(background["state_id"], "1", q_text, z_label)]
                scaled = row_map[(background["state_id"], "1.3", q_text, z_label)]
                # Z itself is carried by the response's normalized coordinate
                # scale; inspect H through the coefficient-independent ratio.
                W0 = float(background["model"].W0)
                expected = 1.3 ** 2
                # Rebuild from the raw Pi to avoid trusting serialized Z.
                pi = _pi_from_row(one)
                _, z1 = _coefficients(background, 1.0, pi)
                _, z13 = _coefficients(background, 1.3, pi)
                err = abs(z13 / z1 - expected) / expected
                scale_rows.append({"state_id": background["state_id"], "q_MeV": q_text, "z_label": z_label, "Z_ratio": z13 / z1, "expected_ratio": expected, "relative_difference": err, "pass": err < 1e-12})
    controls["scale_applied_once"] = {"rows": scale_rows, "pass": bool(scale_rows and all(item["pass"] for item in scale_rows))}

    # Fine quadrature response-level convergence for every unique Pi (all five
    # z samples and both q values) and both scale responses: 80 unique kernels,
    # 160 response checks.  Near-zero current channels use a declared absolute
    # floor rather than the unit floor used for ordinary susceptibilities.
    refinement_rows = []
    for background in backgrounds:
        state = background["state"]
        for q_text in Q_GRID:
            q = float(q_text)
            for z_label in Z_LABELS:
                z = _z_for(z_label, q, float(state["T_MeV"]))
                primary_pi = kernel_cache[(background["state_id"], q_text, z_label)]
                refined_pi = tuple(_master_kernel_float(state["mstar_MeV"], state["nu_MeV"], q, state["T_MeV"], z, p_order=CONTROL_P_ORDER, u_order=CONTROL_U_ORDER, cutoff=CONTROL_CUTOFF_MEV))
                near_zero_floor = 1.0e-8 * max(abs(primary_pi[0]), abs(primary_pi[2]), 1.0)
                pi_errors = [_relative(primary_pi[idx], refined_pi[idx], floor=near_zero_floor) for idx in range(6)]
                for scale_text in SCALES:
                    primary_answer = _response(background, float(scale_text), q, z, primary_pi)
                    refined_answer = _response(background, float(scale_text), q, z, refined_pi)
                    chi_error = _relative(primary_answer["chi"], refined_answer["chi"], floor=1e-20)
                    max_error = max(max(pi_errors), chi_error)
                    refinement_rows.append({
                        "state_id": background["state_id"], "scale": scale_text,
                        "q_MeV": q_text, "z_label": z_label,
                        "Pi_channel_relative_errors": pi_errors,
                        "Pi_near_zero_absolute_floor": near_zero_floor,
                        "Pi_max_relative_difference": max(pi_errors),
                        "chi_relative_difference": chi_error,
                        "max_relative_difference": max_error,
                        "pass": max_error < KERNEL_RELATIVE_LIMIT,
                    })
    controls["quadrature_response_refinement"] = {
        "unique_Pi_count": EXPECTED_STATE_COUNT * len(Q_GRID) * len(Z_LABELS),
        "response_check_count": len(refinement_rows), "rows": refinement_rows,
        "pass": bool(len(refinement_rows) == EXPECTED_RESPONSE_ROW_COUNT and all(item["pass"] for item in refinement_rows)),
    }

    # UHP cold-limit control against the maintained old cold retarded kernel.
    cold_m, cold_nu, cold_q, cold_z, cold_T = 700.0, 760.0, 100.0, 70.0j, 2.0
    cold_kf = math.sqrt(cold_nu * cold_nu - cold_m * cold_m)
    warm_low = finite_temperature_master_kernel(cold_m, cold_nu, cold_q, cold_T, cold_z, p_order=CONTROL_P_ORDER, u_order=CONTROL_U_ORDER, cutoff=CONTROL_CUTOFF_MEV)
    old_low = tuple(complex(value) for value in hessian.polarization_kernel(cold_m, cold_kf, cold_q, cold_z, d=DEGENERACY))
    cold_errors = [_relative(warm_low[idx], old_low[idx], floor=1e-5) for idx in range(3)]
    controls["low_temperature_old_cold_UHP"] = {"mass_MeV": cold_m, "nu_MeV": cold_nu, "T_MeV": cold_T, "q_MeV": cold_q, "z": _cjson(cold_z), "relative_errors": cold_errors, "pass": bool(max(cold_errors) < 3e-3)}

    # Fixed cut control: strict T=0 cold closed form is zero at omega=75,
    # while finite-T tails are nonzero.  T=2 is retained to expose underflow
    # handling, and T=70 is the declared warm material point.
    cut_control_rows = []
    for T in (2.0, 70.0):
        cut = thermal_cut_imag(700.0, 760.0, 100.0, 75.0, T, epsrel=2e-8, epsabs=1e-80)
        cut_control_rows.append({"T_MeV": T, "ImPi": cut, "all_finite": all(math.isfinite(v) for v in cut), "all_positive": all(v > 0.0 for v in cut)})
    cold_cut = hessian.analytic_ph_imag(700.0, cold_kf, 100.0, 75.0, d=DEGENERACY)
    underflow_rejected = False
    underflow_message = ""
    try:
        thermal_cut_imag(700.0, 760.0, 100.0, 75.0, 0.1, epsrel=2e-8, epsabs=1e-100)
    except ThermalMasterResponseError as exc:
        underflow_rejected = True
        underflow_message = str(exc)
    # Independent irreducible-Pi smearing of the maintained cold closed form,
    # evaluated at a bounded nonzero-T->0 point inside the old cold cut.
    smear_mass, smear_nu, smear_q, smear_w, smear_T = 700.0, 760.0, 100.0, 25.0, 2.0
    cold_inside = hessian.analytic_ph_imag(smear_mass, math.sqrt(smear_nu ** 2 - smear_mass ** 2), smear_q, smear_w, d=DEGENERACY)
    exact_inside = thermal_cut_imag(smear_mass, smear_nu, smear_q, smear_w, smear_T, epsrel=2e-8, epsabs=1e-80)
    smeared_inside = _thermal_smearing_old_cold_cut(smear_mass, smear_nu, smear_q, smear_w, smear_T)
    smear_errors = [_relative(exact_inside[idx], smeared_inside[idx], floor=1e-8) for idx in range(3)]
    controls["thermal_cut_cold_limit_and_warm_tail"] = {
        "fixed_control": {"mass_MeV": 700.0, "nu_MeV": 760.0, "q_MeV": 100.0, "omega_MeV": 75.0},
        "T0_cold_closed_form_ImPi": cold_cut, "rows": cut_control_rows,
        "underflow_probe_T_MeV": 0.1, "underflow_rejected": underflow_rejected, "underflow_message": underflow_message,
        "pass": bool(all(row["all_finite"] and row["all_positive"] for row in cut_control_rows) and max(abs(v) for v in cold_cut) == 0.0 and cut_control_rows[1]["all_positive"] and underflow_rejected),
        "independent_smearing_not_final_chi": True,
    }
    controls["thermal_cut_independent_smearing"] = {
        "sample": {"mass_MeV": smear_mass, "nu_MeV": smear_nu, "q_MeV": smear_q, "omega_MeV": smear_w, "T_MeV": smear_T},
        "cold_inside_cut_ImPi": cold_inside, "direct_thermal_ImPi": exact_inside,
        "smeared_cold_ImPi": smeared_inside, "relative_errors": smear_errors,
        "irreducible_Pi_only": True, "dressed_chi_not_smeared": True,
        "pass": bool(max(abs(v) for v in cold_inside) > 0.0 and all(v > 0.0 for v in exact_inside) and max(smear_errors) < 3e-6),
    }

    # Low-temperature edge control: when Emin is below the Fermi edge, the
    # exact thermal cut must approach the maintained cold particle-hole cut
    # even if the endpoint occupation difference itself rounds to zero.  The
    # reflected chemical potential checks particle/antiparticle parity in the
    # same public API rather than relying on an assumed sign convention.
    edge_m, edge_n, edge_q, edge_w, edge_T = 700.0, 1200.0, 100.0, 25.0, 0.1
    edge_direct = thermal_cut_imag(edge_m, edge_n, edge_q, edge_w, edge_T, epsrel=2e-8, epsabs=1e-80)
    edge_kf = math.sqrt(edge_n * edge_n - edge_m * edge_m)
    edge_cold = tuple(float(complex(value).real) for value in hessian.analytic_ph_imag(edge_m, edge_kf, edge_q, edge_w, d=DEGENERACY))
    edge_errors = [_relative(edge_direct[idx], edge_cold[idx], floor=1e-8) for idx in range(3)]
    edge_negative = thermal_cut_imag(edge_m, -edge_n, edge_q, edge_w, edge_T, epsrel=2e-8, epsabs=1e-80)
    edge_parity_errors = (
        _relative(edge_direct[0], edge_negative[0], floor=1e-8),
        _relative(edge_direct[1], -edge_negative[1], floor=1e-8),
        _relative(edge_direct[2], edge_negative[2], floor=1e-8),
    )
    controls["thermal_cut_low_temperature_edge"] = {
        "sample": {"mass_MeV": edge_m, "nu_MeV": edge_n, "q_MeV": edge_q, "omega_MeV": edge_w, "T_MeV": edge_T},
        "direct_thermal_ImPi": edge_direct, "old_cold_analytic_ImPi": edge_cold,
        "relative_errors_vs_old_cold": edge_errors,
        "reflected_nu_direct_thermal_ImPi": edge_negative,
        "particle_antiparticle_parity_errors": edge_parity_errors,
        "pass": bool(max(edge_errors) < 3e-6 and max(edge_parity_errors) < 3e-10),
    }

    # Guards and explicit failure controls must exercise real rejection paths.
    invalid_cases = ((0, 1, 100, 70, 1j), (700, 1, -1, 70, 1j), (700, 1, 100, 0, 1j), (700, 1, 100, 70, 75.0))
    invalid_rejections = []
    for case in invalid_cases:
        try:
            finite_temperature_master_kernel(*case)
            invalid_rejections.append(False)
        except ThermalMasterResponseError:
            invalid_rejections.append(True)
    singular = hessian.response_from_hessian({"a": 1, "b": 0, "d0": 0, "g": 1, "h": 0, "t0": 1}, 4, 100, 100)
    controls["input_and_singular_failure_controls"] = {"invalid_case_rejections": invalid_rejections, "singular_status": singular["status"], "pass": bool(all(invalid_rejections) and singular["chi"] is None)}

    # Cut rows are assembled before controls, and every expected row is kept.
    cut_rows = result["thermal_cut_rows"]
    controls["thermal_cut_coverage"] = {"row_count": len(cut_rows), "expected_row_count": EXPECTED_CUT_ROW_COUNT, "all_finite": all(_finite_tree(row) for row in cut_rows), "pass": bool(len(cut_rows) == EXPECTED_CUT_ROW_COUNT and all(_finite_tree(row) for row in cut_rows))}

    # A compact status summary does not suppress any non-OK row.
    controls["status_counts"] = {status: sum(1 for row in rows if row["status"] == status) for status in sorted({row["status"] for row in rows})}
    controls["production_row_coverage"] = {"row_count": len(rows), "expected_row_count": EXPECTED_RESPONSE_ROW_COUNT, "all_rows_present": len(rows) == EXPECTED_RESPONSE_ROW_COUNT, "pass": len(rows) == EXPECTED_RESPONSE_ROW_COUNT}
    return controls


def build_result() -> dict[str, Any]:
    backgrounds = _build_live_backgrounds()
    kernel_cache: dict[tuple[str, str, str], tuple[complex, ...]] = {}
    rows: list[dict[str, Any]] = []
    with mp.workdps(THERMAL_DPS):
        for background in backgrounds:
            state = background["state"]
            for q_text in Q_GRID:
                q = float(q_text)
                for z_label in Z_LABELS:
                    z = _z_for(z_label, q, float(state["T_MeV"]))
                    pi = finite_temperature_master_kernel(
                        state["mstar_MeV"], state["nu_MeV"], q, state["T_MeV"], z,
                        p_order=PRIMARY_P_ORDER, u_order=PRIMARY_U_ORDER, cutoff=PRIMARY_CUTOFF_MEV,
                    )
                    kernel_cache[(background["state_id"], q_text, z_label)] = pi
                    for scale_text in SCALES:
                        rows.append(_make_response_row(background, scale_text, q_text, z_label, pi))
    cut_rows = [_cut_row(background, q_text, fraction_text) for background in backgrounds for q_text in Q_GRID for fraction_text in FREQUENCY_FRACTIONS]
    if len(rows) != EXPECTED_RESPONSE_ROW_COUNT or len(cut_rows) != EXPECTED_CUT_ROW_COUNT:
        raise ThermalMasterResponseError("production row coverage is incomplete")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "status": STATUS_PASS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "protocol_provenance": "P1_ADOPTED_A1_COMPLETE_FINITE_T_DIRAC_PROJECTOR_MASTER_RESPONSE",
        "physical_inputs_fixed": True, "no_saved_results_as_inputs": True,
        "states": [_public_state(background) for background in backgrounds],
        "inputs": {
            "anchors": list(ANCHORS), "temperature_MeV": list(TEMPERATURES), "chemical_potential_MeV": list(CHEMICAL_POTENTIALS),
            "models": list(MODEL_ORDER), "correlated_scales": list(SCALES), "q_MeV": list(Q_GRID),
            "z_samples": {label: "declared per-row map" for label in Z_LABELS},
            "degeneracy": DEGENERACY, "projector_branches": [[1, 1], [1, -1], [-1, 1], [-1, -1]],
            "kernel_formula": "d/(4*pi^2) integral p^2 dp du sum_st (deltaN_s(p)-deltaN_t(k))/(t Ek-s Ep-z) V_AB^st",
            "deltaN": "s*f((E-s*nu)/T)",
            "quadrature": {"primary_p_order": PRIMARY_P_ORDER, "primary_u_order": PRIMARY_U_ORDER, "primary_cutoff_MeV": PRIMARY_CUTOFF_MEV, "control_p_order": CONTROL_P_ORDER, "control_u_order": CONTROL_U_ORDER, "control_cutoff_MeV": CONTROL_CUTOFF_MEV, "tail_interpretation": "numerical infinity approximation, not a physical UV cutoff"},
        },
        "coverage": {"state_count": len(backgrounds), "response_row_count": len(rows), "expected_response_row_count": EXPECTED_RESPONSE_ROW_COUNT, "thermal_cut_row_count": len(cut_rows), "expected_thermal_cut_row_count": EXPECTED_CUT_ROW_COUNT, "all_response_rows_present": True, "all_cut_rows_present": True},
        # ``rows``/``cut_rows`` are compatibility names; both aliases are
        # checked against the descriptive names by validate_result.
        "response_rows": _jsonable(rows), "rows": _jsonable(rows),
        "thermal_cut_rows": _jsonable(cut_rows), "cut_rows": _jsonable(cut_rows),
        "interpretation_limits": [
            "The z=0 row is a grand-canonical static response; q=0 and nonzero-z number conservation are noncommuting limits.",
            "Imaginary-axis rows are analytic/Matsubara probes, not measured oscillation widths and not a summed covariance.",
            "The raw six-current Ward checks test the complete projector quadrature; the particle-hole-only control is rejected.",
            "The analytic thermal cut is bare spacelike absorption; it does not provide an interacting pole width, collision rate, or detector prediction.",
            "Finite grids and a medium subtraction do not establish a positive completed spectral measure, all-plane stability, vacuum RPA, or empirical agreement.",
            "The thermal root is a bounded local branch; global phase equilibrium, composition, finite nuclei, clusters, gravity, and detector mapping remain open.",
        ],
        "source_sha256": {
            "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "thermal_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(),
            "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest(),
            "retarded_hessian": hashlib.sha256(Path(hessian.__file__).resolve().read_bytes()).hexdigest(),
        },
    }
    result["controls"] = _add_controls(result, backgrounds, kernel_cache)
    controls = result["controls"]
    # Diagnostic-only Matsubara positivity is intentionally excluded from this
    # acceptance conjunction; it is recorded but cannot certify a theorem.
    gate_names = [name for name, item in controls.items() if isinstance(item, Mapping) and "pass" in item and name != "Matsubara_positive_sample_diagnostic"]
    result["all_controls_pass"] = bool(gate_names and all(bool(controls[name]["pass"]) for name in gate_names))
    result["status"] = STATUS_PASS if result["all_controls_pass"] else STATUS_FAIL
    result["integrity_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def validate_result(result: Any) -> bool:
    if not isinstance(result, Mapping) or not _finite_tree(result):
        return False
    if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != STATUS_PASS:
        return False
    if result.get("evidence_weight") != EVIDENCE_WEIGHT or not result.get("physical_inputs_fixed") or not result.get("no_saved_results_as_inputs"):
        return False
    coverage = result.get("coverage", {})
    rows = result.get("response_rows", [])
    cuts = result.get("thermal_cut_rows", [])
    if result.get("rows") != rows or result.get("cut_rows") != cuts:
        return False
    if coverage.get("state_count") != EXPECTED_STATE_COUNT or coverage.get("response_row_count") != EXPECTED_RESPONSE_ROW_COUNT or coverage.get("expected_response_row_count") != EXPECTED_RESPONSE_ROW_COUNT or len(rows) != EXPECTED_RESPONSE_ROW_COUNT or not coverage.get("all_response_rows_present"):
        return False
    if coverage.get("thermal_cut_row_count") != EXPECTED_CUT_ROW_COUNT or coverage.get("expected_thermal_cut_row_count") != EXPECTED_CUT_ROW_COUNT or len(cuts) != EXPECTED_CUT_ROW_COUNT or not coverage.get("all_cut_rows_present"):
        return False
    keys = {(a, t, mu, model, scale, q, z) for a in ANCHORS for t in TEMPERATURES for mu in CHEMICAL_POTENTIALS for model in MODEL_ORDER for scale in SCALES for q in Q_GRID for z in Z_LABELS}
    actual = {(row.get("anchor"), row.get("T_MeV"), row.get("mu_MeV"), row.get("model"), row.get("scale"), row.get("q_MeV"), row.get("z_label")) for row in rows if isinstance(row, Mapping)}
    if actual != keys or len({row.get("row_id") for row in rows}) != EXPECTED_RESPONSE_ROW_COUNT:
        return False
    cut_keys = {(a, t, mu, model, q, f) for a in ANCHORS for t in TEMPERATURES for mu in CHEMICAL_POTENTIALS for model in MODEL_ORDER for q in Q_GRID for f in FREQUENCY_FRACTIONS}
    cut_actual = {(row.get("anchor"), row.get("T_MeV"), row.get("mu_MeV"), row.get("model"), row.get("q_MeV"), row.get("omega_over_q")) for row in cuts if isinstance(row, Mapping)}
    if cut_actual != cut_keys or len({row.get("row_id") for row in cuts}) != EXPECTED_CUT_ROW_COUNT:
        return False
    for row in rows:
        if row.get("status") != "MASTER_RESPONSE_OFF_POLE":
            return False
        pi = row.get("Pi", {})
        if set(pi) != {"vv_MeV2", "vs_MeV2", "ss_MeV2", "vL_MeV2", "sL_MeV2", "LL_MeV2"}:
            return False
        for value in pi.values():
            if not isinstance(value, Mapping) or not math.isfinite(float(value.get("real"))) or not math.isfinite(float(value.get("imag"))):
                return False
        response = row.get("response", {})
        if response.get("direct_status") != "DIRECT_SOLVE_OK" or response.get("chi") is None or response.get("direct_solution") is None:
            return False
        if response.get("direct_relative_error") is None or float(response["direct_relative_error"]) >= DIRECT_RELATIVE_LIMIT:
            return False
        if not isinstance(response.get("H"), list) or len(response["H"]) != 4 or any(len(line) != 4 for line in response["H"]):
            return False
    for row in cuts:
        if row.get("status") != "FINITE_T_SPACELIKE_ABSORPTION" or not row.get("bare_polarization_only") or row.get("finite_eta"):
            return False
        if not all(math.isfinite(float(v)) for v in row.get("Pi_absorptive_Im_MeV2", {}).values()):
            return False
        if not all(math.isfinite(float(v)) for v in row.get("integration_error_estimate_MeV2", {}).values()):
            return False
    controls = result.get("controls", {})
    required = ("static_limit_bridge", "q0_order_of_limits", "raw_six_current_Ward", "particle_hole_only_negative_control", "charge_conjugation_antimaterial", "frequency_conjugation", "direct_4x4_vs_schur", "scale_applied_once", "quadrature_response_refinement", "low_temperature_old_cold_UHP", "thermal_cut_cold_limit_and_warm_tail", "thermal_cut_independent_smearing", "thermal_cut_low_temperature_edge", "input_and_singular_failure_controls", "thermal_cut_coverage", "production_row_coverage")
    if not isinstance(controls, Mapping) or any(name not in controls for name in required):
        return False
    for name in required:
        if not controls[name].get("pass"):
            return False
    source_hashes = result.get("source_sha256", {})
    expected_hashes = {
        "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thermal_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(),
        "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest(),
        "retarded_hessian": hashlib.sha256(Path(hessian.__file__).resolve().read_bytes()).hexdigest(),
    }
    if source_hashes != expected_hashes:
        return False
    digest = result.get("integrity_sha256")
    payload = dict(result)
    payload.pop("integrity_sha256", None)
    return isinstance(digest, str) and hashlib.sha256(_canonical_json(payload)).hexdigest() == digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        payload = _jsonable(build_result())
    except Exception as exc:
        payload = {"schema_version": SCHEMA_VERSION, "status": STATUS_FAIL, "error": f"{type(exc).__name__}: {exc}"}
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0 if payload.get("status") == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
