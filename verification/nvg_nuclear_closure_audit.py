#!/usr/bin/env python3
"""Bounded W8 global-support and static finite-q audit.

This module is deliberately a new audit surface.  The producer in
``source_complete_scaling_saturation_audit`` remains read-only and supplies
the already declared inverse W8 designs.  The global-support calculation
reconstructs the three polynomial coefficients with directed interval
arithmetic from the exact decimal design inputs; it never treats two
mpmath runs as an outward enclosure.

The global statement is for the uniform zero-temperature Thomas--Fermi
functional only.  The finite-q statement is the canonical static local TF
Hessian with a constrained (maximum) A0 field.  Neither result is a quantum
RPA, finite-nucleus, finite-N, gravity, or empirical-validation result.
The command line is strict JSON and has no file-writing side effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mpmath as mp
import sympy as sp

# Do not create __pycache__ as a CLI side effect.
sys.dont_write_bytecode = True

try:  # Both direct script execution and package-style imports are supported.
    from source_complete_scaling_saturation_audit import (
        BulkModel,
        INPUTS,
        inverse_potential_jet,
    )
except ImportError:  # pragma: no cover - package import path
    from .source_complete_scaling_saturation_audit import (
        BulkModel,
        INPUTS,
        inverse_potential_jet,
    )


HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
SOURCE_PRODUCER_PATH = HERE / "source_complete_scaling_saturation_audit.py"
SCHEMA = "nvg_nuclear_closure_audit.v1"
STATUS = "COMPUTED_W8_GLOBAL_SUPPORT_STATIC_TF_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0
TARGETS = ("0.90", "0.93")
NEGATIVE_CONTROL = "0.75"
SEALED_N0 = "0.16"
SEALED_BINDING = "-16"
SEALED_K = "240"
SEALED_MU = "923"
SEALED_C_RHO = "0"
WORKING_PRECISIONS = (80, 110)
IV_DIGITS_EXTRA = 24
MAIN_LOWER = "0.5"
ONSET_TAIL_START = "0.97"
# Fixed decimal strictly above the exact onset 923/939 by about 1.07e-30.
# The mass-bound proof uses max(mu-M*y,0), so this deliberate overlap with
# the analytic Pi=0 tail is valid and avoids a serialized rational-endpoint gap.
ONSET_COVER_END = "0.9829605963791267305644302449425"
TARGET_RADII = {"0.90": "0.001", "0.93": "0.0005"}
MIN_INTERVAL_WIDTH = mp.mpf("1e-9")
MAX_INTERVAL_ATTEMPTS = 20000


REQUIRED_SYMBOLIC = frozenset(
    {
        "two_species_jensen",
        "rho_nonnegative",
        "density_convexity",
        "pi_onset",
        "pi_tangent_upper_bound",
        "support_hessian_gap",
        "inverse_jet_value_equality",
        "inverse_jet_slope_equality",
        "inverse_jet_matrix_determinant",
        "uneliminated_static_hessian",
        "schur_static_determinant_quadratic",
        "p0_compressibility_identity",
        "quadratic_halfline_classification",
        "gradient_threshold_identity",
    }
)


class NuclearClosureError(ValueError):
    """Fail-closed error for malformed inputs or an incomplete certificate."""


def _precision(dps: Any) -> int:
    if isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 200:
        raise NuclearClosureError("dps must be an integer in [80,200]")
    return dps


def _mp(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> mp.mpf:
    if isinstance(value, bool):
        raise NuclearClosureError(f"{name} must be a finite real, not bool")
    try:
        # Reconstructing an existing mpf at an ambient lower precision rounds
        # away its stored high-precision evidence.  Preserve mpf objects;
        # parse external scalars under the caller's explicit work precision.
        result = value if isinstance(value, mp.mpf) else mp.mpf(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise NuclearClosureError(f"{name} must be a finite real") from exc
    if not mp.isfinite(result):
        raise NuclearClosureError(f"{name} must be finite")
    if positive and result <= 0:
        raise NuclearClosureError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise NuclearClosureError(f"{name} must be nonnegative")
    return result


def _shown(value: Any, digits: int = 35) -> Any:
    """Render numbers without allowing NaN/Infinity into JSON."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, mp.mpc):
        return {"real": _shown(value.real, digits), "imag": _shown(value.imag, digits)}
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not mp.isfinite(value):
            raise NuclearClosureError("nonfinite floating-point output")
        return value
    if isinstance(value, mp.mpf):
        value = _mp(value, "derived number")
        return _canonical_decimal(value, digits)
    if isinstance(value, Mapping):
        return {str(key): _shown(item, digits) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shown(item, digits) for item in value]
    return value


def _canonical_decimal(value: mp.mpf, digits: int = 35) -> str:
    """One decimal identity shared by proof boxes and JSON evidence."""

    with mp.workdps(max(mp.mp.dps, digits + 12)):
        return mp.nstr(value, digits)


def _semantic_decimal(value: Any, name: str) -> str:
    """Canonical exact decimal spelling used for sealed protocol inputs."""

    if isinstance(value, bool):
        raise NuclearClosureError(f"{name} must be a finite decimal")
    try:
        # Decimal is intentionally avoided here: mp.mpf accepts the exact
        # decimal strings used by the protocol and rejects non-finite values.
        parsed = mp.mpf(str(value))
    except (ValueError, TypeError, OverflowError) as exc:
        raise NuclearClosureError(f"{name} must be a finite decimal") from exc
    if not mp.isfinite(parsed):
        raise NuclearClosureError(f"{name} must be finite")
    return str(value)


def _require_protocol(y: Any, k: Any = SEALED_K, n0: Any = SEALED_N0,
                      binding: Any = SEALED_BINDING, c_rho: Any = SEALED_C_RHO) -> tuple[str, mp.mpf]:
    y_text = _semantic_decimal(y, "target_y")
    y_value = mp.mpf(y_text)
    allowed = {mp.mpf(item) for item in TARGETS + (NEGATIVE_CONTROL,)}
    if y_value not in allowed:
        raise NuclearClosureError(f"target_y must be one of {TARGETS + (NEGATIVE_CONTROL,)}")
    if y_value <= 0:
        raise NuclearClosureError("target_y must be positive")
    for actual, expected, name in (
        (k, SEALED_K, "Ktarget_MeV"),
        (n0, SEALED_N0, "n0_fm3"),
        (binding, SEALED_BINDING, "binding_MeV"),
        (c_rho, SEALED_C_RHO, "C_rho"),
    ):
        if mp.mpf(_semantic_decimal(actual, name)) != mp.mpf(expected):
            raise NuclearClosureError(f"{name} is sealed at {expected}")
    return (next(item for item in TARGETS + (NEGATIVE_CONTROL,) if mp.mpf(item) == y_value), y_value)


# ---------------------------------------------------------------------------
# Directed interval reconstruction of the W8 coefficients.


def _iv(value: Any):
    return mp.iv.mpf(str(value))


def _iv_bounds(value: Any) -> tuple[mp.mpf, mp.mpf]:
    # ``value.a``/``value.b`` are interval wrappers whose generic mpf
    # conversion can pass through a low-precision representation.  The
    # internal endpoint tuples are exact libmp values.  Every caller that
    # performs arithmetic on these endpoints runs above ``mp.iv.dps`` so the
    # conversion itself is exact rather than an inward rounding.
    with mp.workdps(max(mp.mp.dps, mp.iv.dps + 10)):
        return mp.mpf(value._mpi_[0]), mp.mpf(value._mpi_[1])


def _iv_lo(value: Any) -> mp.mpf:
    return _iv_bounds(value)[0]


def _iv_hi(value: Any) -> mp.mpf:
    return _iv_bounds(value)[1]


def _iv_abs_hi(value: Any) -> mp.mpf:
    lo, hi = _iv_bounds(value)
    return max(abs(lo), abs(hi))


def _iv_contains_zero(value: Any) -> bool:
    lo, hi = _iv_bounds(value)
    return lo <= 0 <= hi


def _iv_text(value: Any, digits: int = 32) -> list[str]:
    # A nearest-rounded decimal rendering can move an endpoint *inside* the
    # binary interval even when many digits are printed.  Add two guard
    # decimal places before formatting so the serialized pair is itself an
    # outward enclosure, rather than merely a pretty rendering of one.
    with mp.workdps(max(mp.mp.dps, mp.iv.dps + 10)):
        lo, hi = _iv_bounds(value)
        scale = max(abs(lo), abs(hi))
        if scale == 0:
            return ["0", "0"]
        exponent = int(mp.floor(mp.log10(scale)))
        guard = mp.power(10, exponent - digits + 3)
        return [mp.nstr(lo - guard, digits), mp.nstr(hi + guard, digits)]


def _iv_width_text(value: Any, digits: int = 12) -> str:
    with mp.workdps(max(mp.mp.dps, mp.iv.dps + 10)):
        lo, hi = _iv_bounds(value)
        return mp.nstr(hi - lo, digits)


def _iv_fermi(n: Any, y: Any) -> dict[str, Any]:
    """Directed interval Fermi expressions for d=4 symmetric matter."""

    d = _iv(4)
    pi = mp.iv.pi
    mass = _iv(939) * y
    k = (6 * pi * pi * n / d) ** (mp.iv.mpf(1) / 3)
    ef = mp.iv.sqrt(k * k + mass * mass)
    # asinh(x)=log(x+sqrt(1+x^2)); iv has no dedicated asinh method.
    asinh = mp.iv.log(k / mass + mp.iv.sqrt(1 + (k / mass) ** 2))
    energy = d / (16 * pi * pi) * (
        k * ef * (2 * k * k + mass * mass) - mass**4 * asinh
    )
    pressure = d / (48 * pi * pi) * (
        k * ef * (2 * k * k - 3 * mass * mass) + 3 * mass**4 * asinh
    )
    ns = d * mass / (4 * pi * pi) * (k * ef - mass * mass * asinh)
    ns_m = 3 * ns / mass - 3 * n / ef
    fnn = k * k / (3 * n * ef)
    return {
        "k": k,
        "mass": mass,
        "ef": ef,
        "energy": energy,
        "pressure": pressure,
        "ns": ns,
        "ns_m": ns_m,
        "fnn": fnn,
    }


def _interval_inverse_jet(y_text: str) -> dict[str, Any]:
    """Reconstruct U8=(a2 z^2+a3 z^3+a4 z^4) with outward intervals."""

    W0 = _iv("859")
    M = _iv("939")
    n = _iv(SEALED_N0) * _iv("197.3269804") ** 3
    mu = _iv(SEALED_MU)
    y = _iv(y_text)
    f = _iv_fermi(n, y)
    cv = (mu - f["ef"]) * y * y / n
    D = f["fnn"] + cv / y**2
    B = M * M * y / f["ef"] - 2 * cv * n / y**3
    denominator = D - _iv(SEALED_K) / (9 * n)
    C = B * B / denominator
    U = f["pressure"] + n * (mu - f["ef"]) / 2
    # F_y=M*y_natural?  The scalar mass is m=M*y, hence F_y=M*n_s;
    # equivalently the producer's ``(n*V-m*n_s)/y`` uses m=M*y.
    Uy = (n * (mu - f["ef"]) - M * y * f["ns"]) / y
    Uyy = C - M * M * f["ns_m"] - 3 * cv * n * n / y**4
    z = y * y - 1
    matrix = mp.iv.matrix(
        [
            [z**j for j in (2, 3, 4)],
            [2 * j * y * z ** (j - 1) for j in (2, 3, 4)],
            [
                2 * j * z ** (j - 1) + 4 * j * (j - 1) * y * y * z ** (j - 2)
                for j in (2, 3, 4)
            ],
        ]
    )
    coeff = tuple(mp.iv.lu_solve(matrix, mp.iv.matrix([U, Uy, Uyy])))
    return {
        "y": y,
        "n": n,
        "mu": mu,
        "M": M,
        "W0": W0,
        "Cv": cv,
        "D": D,
        "B": B,
        "C": C,
        "U": U,
        "Uy": Uy,
        "Uyy": Uyy,
        "coefficients": coeff,
        "source_inputs_exact": True,
        "outward_interval": True,
    }


def _iv_u(coefficients: Sequence[Any], y: Any):
    z = y * y - 1
    return sum(coefficients[i] * z ** (i + 2) for i in range(3))


def _iv_uy(coefficients: Sequence[Any], y: Any):
    z = y * y - 1
    return sum(2 * j * coefficients[i] * y * z ** (j - 1) for i, j in enumerate((2, 3, 4)))


def _iv_uyy(coefficients: Sequence[Any], y: Any):
    z = y * y - 1
    return sum(
        coefficients[i]
        * (2 * j * z ** (j - 1) + 4 * j * (j - 1) * y * y * z ** (j - 2))
        for i, j in enumerate((2, 3, 4))
    )


def _iv_q(coefficients: Sequence[Any], z: Any):
    return coefficients[0] + coefficients[1] * z + coefficients[2] * z * z


def _interval_q_min(coefficients: Sequence[Any], *, domain: str = "global") -> dict[str, Any]:
    """Enclose the exact quadratic minimum, with no sampled sign claim."""

    a2, a3, a4 = coefficients
    if _iv_hi(a4) <= 0:
        return {"valid": False, "reason": "leading_coefficient_not_positive"}
    vertex = -a3 / (2 * a4)
    if domain == "global":
        if _iv_lo(vertex) < -1:
            # The exact interval should resolve this branch; retaining a
            # failed proof is safer than silently choosing a wrong endpoint.
            return {"valid": False, "reason": "global_vertex_domain_unresolved", "vertex": _iv_text(vertex)}
        value = _iv_q(coefficients, vertex)
        return {
            "valid": _iv_lo(value) > 0,
            "vertex": _iv_text(vertex),
            "q_min": _iv_text(value),
            "domain": "z>=-1",
        }
    if domain != "tail_half":
        raise NuclearClosureError(f"unknown q domain {domain}")
    # For the two physical designs the vertex is to the right of -3/4, so
    # q decreases on [-1,-3/4] and the endpoint is the exact minimum.
    if _iv_lo(vertex) >= mp.mpf("-0.75"):
        value = _iv_q(coefficients, _iv("-0.75"))
        return {
            "valid": _iv_lo(value) > 0,
            "vertex": _iv_text(vertex),
            "q_min": _iv_text(value),
            "domain": "-1<=z<=-3/4",
            "argmin": "-3/4",
        }
    if _iv_hi(vertex) <= mp.mpf("-1"):
        value = _iv_q(coefficients, _iv("-1"))
        return {
            "valid": _iv_lo(value) > 0,
            "vertex": _iv_text(vertex),
            "q_min": _iv_text(value),
            "domain": "-1<=z<=-3/4",
            "argmin": "-1",
        }
    return {"valid": False, "reason": "tail_vertex_domain_unresolved", "vertex": _iv_text(vertex)}


# ---------------------------------------------------------------------------
# Common-tangent support and directed interval coverage.


def _regular_model(y_text: str, dps: int):
    with mp.workdps(dps):
        model, state, jet = inverse_potential_jet(
            y_text, SEALED_K, SEALED_N0, SEALED_BINDING
        )
        # Copy no values into a table: this object is only a numerical center
        # for independent derivative/root calculations.
        return model, state, jet


def _optimizer_density(model: BulkModel, y: mp.mpf, *, dps: int) -> tuple[mp.mpf, mp.mpf]:
    """Unique positive Pi optimizer below onset, solved only for centers."""

    with mp.workdps(dps):
        y = _mp(y, "y", positive=True)
        mu = model.MN + mp.mpf(SEALED_BINDING)
        mass = model.MN * y
        if mass >= mu:
            return mp.mpf(0), mp.mpf(0)
        excess = mu - mass
        density = lambda k: model.d * k**3 / (6 * mp.pi**2)
        upper = min(
            mp.sqrt(excess * (mu + mass)),
            (6 * mp.pi**2 * y * y * excess / (model.d * model.Cv)) ** (mp.mpf(1) / 3),
        )
        equation = lambda k: (
            mp.sqrt(k * k + mass * mass)
            + model.Cv * density(k) / y**2
            - mu
        )
        k = mp.findroot(equation, (mp.mpf(0), upper), solver="anderson", maxsteps=2 * dps + 40)
        n = density(k)
        if not 0 < k < upper or n <= 0:
            raise NuclearClosureError("optimizer center left its physical bracket")
        f = model.fermi(n, y)
        return n, k


def _iv_optimizer_phi(n: Any, y: Any, cv: Any):
    f = _iv_fermi(n, y)
    return f["ef"] + cv * n / y**2


def _root_bracket(
    model: BulkModel,
    y_lo: mp.mpf,
    y_hi: mp.mpf,
    n_center: mp.mpf,
    iv: Mapping[str, Any],
) -> tuple[mp.mpf, mp.mpf, str, mp.mpf] | None:
    """Prove n(y) lies in a monotone bracket for an interval y-box."""

    y_lo_text, y_hi_text = _canonical_decimal(y_lo), _canonical_decimal(y_hi)
    y_box = mp.iv.mpf([y_lo_text, y_hi_text])
    mu = _iv(SEALED_MU)
    ratios = (("0.98", "1.02"), ("0.95", "1.05"), ("0.90", "1.10"), ("0.80", "1.20"), ("0.50", "1.50"))
    for lo_text, hi_text in ratios:
        n_lo = n_center * mp.mpf(lo_text)
        n_hi = n_center * mp.mpf(hi_text)
        n_lo_text, n_hi_text = _canonical_decimal(n_lo), _canonical_decimal(n_hi)
        n_lo, n_hi = mp.mpf(n_lo_text), mp.mpf(n_hi_text)
        lo_iv = _iv(n_lo_text)
        hi_iv = _iv(n_hi_text)
        n_box = mp.iv.mpf([n_lo_text, n_hi_text])
        f_box = _iv_fermi(n_box, y_box)
        density_derivative = f_box["fnn"] + iv["Cv"] / y_box**2
        phi_lo = _iv_optimizer_phi(lo_iv, y_box, iv["Cv"])
        phi_hi = _iv_optimizer_phi(hi_iv, y_box, iv["Cv"])
        if (_iv_lo(iv["Cv"]) > 0
                and _iv_lo(f_box["fnn"]) > 0
                and _iv_lo(density_derivative) > 0
                and _iv_hi(phi_lo) < _iv_lo(mu)
                and _iv_lo(phi_hi) > _iv_hi(mu)):
            return n_lo, n_hi, f"{lo_text}:{hi_text}", _iv_lo(density_derivative)
    return None


def _tangent_point_gap(iv: Mapping[str, Any], y: mp.mpf, n_center: mp.mpf):
    """Outward interval for H minus a convex tangent upper bound on Pi."""

    y_iv = _iv(_canonical_decimal(y))
    n_iv = _iv(_canonical_decimal(n_center))
    f = _iv_fermi(n_iv, y_iv)
    cv = iv["Cv"]
    G = f["energy"] + cv * n_iv * n_iv / (2 * y_iv * y_iv)
    Gn = f["ef"] + cv * n_iv / (y_iv * y_iv)
    residual = _iv(SEALED_MU) - Gn
    pi_upper = _iv(SEALED_MU) * n_iv - G + residual * residual * y_iv * y_iv / (2 * cv)
    return _iv_u(iv["coefficients"], y_iv) - pi_upper


def _tangent_lipschitz_box(
    model: BulkModel,
    iv: Mapping[str, Any],
    y_lo: mp.mpf,
    y_hi: mp.mpf,
    n_center: mp.mpf,
    dps: int,
) -> dict[str, Any]:
    y_lo_text, y_hi_text = _canonical_decimal(y_lo), _canonical_decimal(y_hi)
    y_mid_text = _canonical_decimal((y_lo + y_hi) / 2)
    y_mid = mp.mpf(y_mid_text)
    bracket = _root_bracket(model, y_lo, y_hi, n_center, iv)
    if bracket is None:
        return {"accepted": False, "reason": "optimizer_bracket_unresolved"}
    n_lo, n_hi, bracket_label, density_derivative_lower = bracket
    y_box = mp.iv.mpf([y_lo_text, y_hi_text])
    n_box = mp.iv.mpf([_canonical_decimal(n_lo), _canonical_decimal(n_hi)])
    f = _iv_fermi(n_box, y_box)
    hprime = (
        _iv_uy(iv["coefficients"], y_box)
        + _iv(939) * f["ns"]
        - iv["Cv"] * n_box * n_box / y_box**3
    )
    h_mid_gap = _tangent_point_gap(iv, y_mid, n_center)
    left_distance = _iv(y_mid_text) - _iv(y_lo_text)
    right_distance = _iv(y_hi_text) - _iv(y_mid_text)
    left_upper_text = _iv_text(left_distance, digits=50)[1]
    right_upper_text = _iv_text(right_distance, digits=50)[1]
    radius_upper_text = max(
        (left_upper_text, right_upper_text), key=lambda item: sp.Rational(item)
    )
    radius_iv = mp.iv.mpf(["0", radius_upper_text])
    margin_iv = h_mid_gap - abs(hprime) * radius_iv
    margin = _iv_lo(margin_iv)
    return {
        "accepted": bool(margin > 0),
        "reason": "positive_tangent_lipschitz_lower" if margin > 0 else "nonpositive_box_lower",
        "lower_bound": margin,
        "h_mid_lower": _iv_lo(h_mid_gap),
        "lipschitz_upper": _iv_abs_hi(hprime),
        "optimizer_bracket": bracket_label,
        "D_n_lower": density_derivative_lower,
        "proof_box_endpoints": [_canonical_decimal(y_lo), _canonical_decimal(y_hi)],
        "canonical_midpoint": y_mid_text,
        "radius_outward_upper": radius_upper_text,
        "n_lo": n_lo,
        "n_hi": n_hi,
        "y_mid": y_mid,
    }


def _certify_tangent_segment(
    model: BulkModel,
    iv: Mapping[str, Any],
    start: mp.mpf,
    end: mp.mpf,
    dps: int,
) -> dict[str, Any]:
    """Recursively enclose H on a closed compact segment."""

    start = mp.mpf(_canonical_decimal(start))
    end = mp.mpf(_canonical_decimal(end))
    queue: list[tuple[mp.mpf, mp.mpf]] = [(start, end)]
    leaves: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    attempts = 0
    while queue:
        y_lo, y_hi = queue.pop()
        # These exact decimal values are both serialized and fed to the
        # interval constructors below.  No post-proof endpoint relabelling.
        y_lo = mp.mpf(_canonical_decimal(y_lo))
        y_hi = mp.mpf(_canonical_decimal(y_hi))
        attempts += 1
        if attempts > MAX_INTERVAL_ATTEMPTS:
            unresolved.append({"lo": y_lo, "hi": y_hi, "reason": "attempt_budget_exhausted"})
            unresolved.extend({"lo": lo, "hi": hi, "reason": "queue_not_examined"} for lo, hi in queue)
            break
        y_mid = mp.mpf(_canonical_decimal((y_lo + y_hi) / 2))
        n_center, _ = _optimizer_density(model, y_mid, dps=dps)
        result = _tangent_lipschitz_box(model, iv, y_lo, y_hi, n_center, dps)
        if result["accepted"]:
            leaves.append({"lo": y_lo, "hi": y_hi, **result})
            continue
        if y_hi - y_lo <= MIN_INTERVAL_WIDTH:
            unresolved.append({"lo": y_lo, "hi": y_hi, **result})
            continue
        split = (y_lo + y_hi) / 2
        queue.append((split, y_hi))
        queue.append((y_lo, split))
    leaves.sort(key=lambda row: row["lo"])
    coverage = _check_coverage(leaves, start, end)
    return {
        "domain": [start, end],
        "method": "outward_interval_tangent_plus_lipschitz",
        "intervals": leaves,
        "unresolved": unresolved,
        "attempted": attempts,
        "split_or_rejected": attempts - len(leaves),
        "coverage": coverage,
        "passed": bool(leaves and not unresolved and coverage["passed"]),
    }


def _certify_simple_bound_segment(
    iv: Mapping[str, Any],
    start: mp.mpf,
    end: mp.mpf,
    dps: int,
) -> dict[str, Any]:
    """Directed subdivision for Pi<=delta^2*y^2/(2 Cv) near onset."""

    start = mp.mpf(_canonical_decimal(start))
    end = mp.mpf(_canonical_decimal(end))
    queue: list[tuple[mp.mpf, mp.mpf]] = [(start, end)]
    leaves: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    attempts = 0
    while queue:
        y_lo, y_hi = queue.pop()
        y_lo = mp.mpf(_canonical_decimal(y_lo))
        y_hi = mp.mpf(_canonical_decimal(y_hi))
        attempts += 1
        y_box = mp.iv.mpf([_canonical_decimal(y_lo), _canonical_decimal(y_hi)])
        delta = _iv(SEALED_MU) - _iv(939) * y_box
        # F>=M*y*n gives Pi<=max(delta,0)^2*y^2/(2*Cv).  The positive-part
        # form remains valid for the final box whose outward y enclosure may
        # straddle the exact rational onset by a few ulps.
        delta_upper_text = "0" if _iv_hi(delta) <= 0 else _iv_text(delta)[1]
        delta_positive = mp.iv.mpf(["0", delta_upper_text])
        bound = delta_positive**2 * y_box * y_box / (2 * iv["Cv"])
        gap = _iv_u(iv["coefficients"], y_box) - bound
        lower = _iv_lo(gap)
        domain_guard = bool(y_lo > 0 and _iv_lo(iv["Cv"]) > 0)
        row = {
            "lo": y_lo,
            "hi": y_hi,
            "proof_box_endpoints": [_canonical_decimal(y_lo), _canonical_decimal(y_hi)],
            "lower_bound": lower,
            "delta_positive_part_upper": _iv_hi(delta_positive),
            "mass_bound_domain_guard": domain_guard,
        }
        if lower > 0 and domain_guard:
            leaves.append({**row, "accepted": True, "method": "Pi_mass_bound"})
            continue
        if y_hi - y_lo <= MIN_INTERVAL_WIDTH or attempts > MAX_INTERVAL_ATTEMPTS:
            unresolved.append({**row, "accepted": False, "reason": "nonpositive_simple_bound"})
            if attempts > MAX_INTERVAL_ATTEMPTS:
                unresolved.extend({"lo": lo, "hi": hi, "reason": "queue_not_examined"} for lo, hi in queue)
                break
            continue
        split = (y_lo + y_hi) / 2
        queue.append((split, y_hi))
        queue.append((y_lo, split))
    leaves.sort(key=lambda row: row["lo"])
    coverage = _check_coverage(leaves, start, end)
    return {
        "domain": [start, end],
        "method": "outward_interval_simple_mass_bound",
        "intervals": leaves,
        "unresolved": unresolved,
        "attempted": attempts,
        "coverage": coverage,
        "passed": bool(leaves and not unresolved and coverage["passed"]),
    }


def _check_coverage(intervals: Sequence[Mapping[str, Any]], start: mp.mpf, end: mp.mpf) -> dict[str, Any]:
    if not intervals:
        return {"passed": False, "reason": "empty_interval_list", "gap_count": None}
    cursor = start
    gaps = []
    for row in sorted(intervals, key=lambda item: item["lo"]):
        lo, hi = row["lo"], row["hi"]
        if lo != cursor:
            gaps.append({"expected": cursor, "received": lo})
        if hi <= lo:
            gaps.append({"invalid_interval": [lo, hi]})
        cursor = hi
    if cursor != end:
        gaps.append({"expected_end": end, "received_end": cursor})
    return {"passed": not gaps, "gap_count": len(gaps), "gaps": gaps[:4]}


def _target_neighborhood(
    model: BulkModel,
    iv: Mapping[str, Any],
    target: str,
    radius: str,
    dps: int,
) -> dict[str, Any]:
    y0 = mp.mpf(target)
    r = mp.mpf(radius)
    y_lo = mp.mpf(_canonical_decimal(y0 - r))
    y_hi = mp.mpf(_canonical_decimal(y0 + r))
    n0 = model.n0
    bracket = _root_bracket(model, y_lo, y_hi, n0, iv)
    equal_y = _iv(target)
    # Reuse the exact-decimal interval density from the inverse jet.  A
    # finite-precision ``model.n0`` centre would introduce a spurious
    # one-sided H' residue at the deliberately exact target.
    n_iv = iv["n"]
    f_target = _iv_fermi(n_iv, equal_y)
    pi_target = _iv(SEALED_MU) * n_iv - f_target["energy"] - iv["Cv"] * n_iv * n_iv / (2 * equal_y**2)
    h_target = _iv_u(iv["coefficients"], equal_y) - pi_target
    hprime_target = (
        _iv_uy(iv["coefficients"], equal_y)
        + _iv(939) * f_target["ns"]
        - iv["Cv"] * n_iv * n_iv / equal_y**3
    )
    h_zero_certified = _iv_contains_zero(h_target)
    hprime_zero_certified = _iv_contains_zero(hprime_target)
    result: dict[str, Any] = {
        "domain": [y_lo, y_hi],
        "proof_box_endpoints": [_canonical_decimal(y_lo), _canonical_decimal(y_hi)],
        "target": y0,
        "radius": r,
        "equality_H_interval": h_target,
        "equality_Hprime_interval": hprime_target,
        "equality_H_contains_zero": h_zero_certified,
        "equality_Hprime_contains_zero": hprime_zero_certified,
        "raw_interval_H_contains_zero": _iv_contains_zero(h_target),
        "raw_interval_Hprime_contains_zero": _iv_contains_zero(hprime_target),
        # The interval coefficient enclosure and exact-decimal n make the
        # target cancellation an actual enclosure containing zero.  The
        # residual fields remain explicit guards so a coefficient mutation
        # cannot silently preserve the equality.
        "equality_H_exact_residual": _iv_abs_hi(h_target),
        "equality_Hprime_exact_residual": _iv_abs_hi(hprime_target),
        "optimizer_bracket": None if bracket is None else bracket[2],
        "Fyy_lower_used": mp.mpf(0),
    }
    if bracket is None:
        result.update({"passed": False, "reason": "optimizer_bracket_unresolved"})
        return result
    n_lo, n_hi, _, density_derivative_lower = bracket
    y_box = mp.iv.mpf([_canonical_decimal(y_lo), _canonical_decimal(y_hi)])
    n_box = mp.iv.mpf([_canonical_decimal(n_lo), _canonical_decimal(n_hi)])
    f = _iv_fermi(n_box, y_box)
    D = f["fnn"] + iv["Cv"] / y_box**2
    B = _iv(939) ** 2 * y_box / f["ef"] - 2 * iv["Cv"] * n_box / y_box**3
    Uyy = _iv_uyy(iv["coefficients"], y_box)
    vector_yy = 3 * iv["Cv"] * n_box * n_box / y_box**4
    # F_yy=M^2*d/(2*pi^2)*integral_0^k p^4/(p^2+m^2)^(3/2) dp >= 0.
    # Keeping that analytic nonnegative term at zero is conservative.  All
    # remaining operations stay in interval arithmetic until the final lower
    # endpoint is extracted.
    hpp_interval = Uyy + vector_yy - B**2 / D
    hpp_lower = _iv_lo(hpp_interval)
    target_rational = sp.Rational(target)
    target_z = target_rational**2 - 1
    calibration_matrix = sp.Matrix(
        [
            [target_z**j for j in (2, 3, 4)],
            [2 * j * target_rational * target_z ** (j - 1) for j in (2, 3, 4)],
            [
                2 * j * target_z ** (j - 1)
                + 4 * j * (j - 1) * target_rational**2 * target_z ** (j - 2)
                for j in (2, 3, 4)
            ],
        ]
    )
    matrix_determinant = sp.factor(calibration_matrix.det())
    determinant_formula = 16 * target_rational**3 * (target_rational**2 - 1) ** 6
    determinant_formula_passed = bool(sp.simplify(matrix_determinant - determinant_formula) == 0)
    algebraic_equality = bool(
        determinant_formula_passed and target_rational > 0 and target_rational != 1
    )
    result.update(
        {
            "Hsecond_lower": hpp_lower,
            "D_lower": _iv_lo(D),
            "B_abs_upper": _iv_abs_hi(B),
            "C_lower_without_Fyy": _iv_lo(Uyy + vector_yy),
            "Fyy_nonnegative_integrand_guard": True,
            "Fyy_identity": "M^2*d/(2*pi^2)*integral_0^k p^4/(p^2+m^2)^(3/2) dp >= 0",
            "optimizer_D_n_lower": density_derivative_lower,
            "calibration_matrix_determinant_exact": str(matrix_determinant),
            "calibration_matrix_nonsingular": algebraic_equality,
            "calibration_matrix_determinant_formula_passed": determinant_formula_passed,
            "exact_equality_from_inverse_jet_system": algebraic_equality,
            "exact_equality_argument": "det=16*y^3*(y^2-1)^6 != 0; F+P=n*EF and Cv=(mu-EF)y^2/n make E=mu*n, E_y=0, E_n=mu exactly, so unique density optimization gives H=Hprime=0",
            "passed": bool(
                algebraic_equality
                and h_zero_certified
                and hprime_zero_certified
                and density_derivative_lower > 0
                and hpp_lower > 0
            ),
            "reason": "H=Hprime=0_and_Hsecond_positive" if hpp_lower > 0 else "nonpositive_Hsecond_lower",
        }
    )
    return result


def _global_support(model: BulkModel, target: str, dps: int) -> dict[str, Any]:
    """Build a full-domain certificate or a fully explicit INCONCLUSIVE row."""

    old_iv_dps = mp.iv.dps
    old_mp_dps = mp.mp.dps
    try:
        mp.iv.dps = max(dps + IV_DIGITS_EXTRA, 110)
        # Endpoint tuples contain roughly ``mp.iv.dps`` decimal digits.  Keep
        # ordinary mp arithmetic above that precision so exact endpoint
        # extraction and reporting cannot round a lower bound inward.
        mp.mp.dps = max(mp.iv.dps + 20, 150)
        iv = _interval_inverse_jet(target)
        coeff = iv["coefficients"]
        q_global = _interval_q_min(coeff, domain="global")
        q_tail = _interval_q_min(coeff, domain="tail_half")
        mu = mp.mpf(SEALED_MU)
        mass = mp.mpf("939")
        onset = mu / mass
        onset_cover_end = mp.mpf(ONSET_COVER_END)
        cv_positive = _iv_lo(iv["Cv"]) > 0
        if q_tail.get("valid") is True and q_tail.get("argmin") in {"-1", "-3/4"}:
            q_tail_arg = "-1" if q_tail["argmin"] == "-1" else "-0.75"
            q_tail_value = _iv_q(coeff, _iv(q_tail_arg))
            tiny_u_interval = _iv("0.5625") * q_tail_value
            tiny_pi_interval = _iv(SEALED_MU) ** 2 / (8 * iv["Cv"])
            tiny_gap_interval = tiny_u_interval - tiny_pi_interval
            tiny_lower = _iv_lo(tiny_u_interval)
            tiny_pi_upper = _iv_hi(tiny_pi_interval)
            tiny_pass = bool(_iv_lo(tiny_gap_interval) > 0)
        else:
            tiny_lower = mp.mpf(0)
            tiny_pi_upper = mp.inf
            tiny_pass = False
        radius = mp.mpf(TARGET_RADII[target])
        neighborhood = _target_neighborhood(model, iv, target, TARGET_RADII[target], dps)

        # The remaining compact middle interval is covered by a union of
        # interval leaves.  The target neighborhood is kept separate because
        # a scan cannot establish its double equality.
        left = _certify_tangent_segment(
            model, iv, mp.mpf(MAIN_LOWER), mp.mpf(target) - radius, dps
        )
        right = _certify_tangent_segment(
            model, iv, mp.mpf(target) + radius, mp.mpf(ONSET_TAIL_START), dps
        )
        onset_tail = _certify_simple_bound_segment(
            iv, mp.mpf(ONSET_TAIL_START), onset_cover_end, dps
        )
        # One explicit closed-union check covers every compact subdomain from
        # y=1/2 through the onset.  The equality neighborhood is represented
        # by its separately certified Hessian leaf; the other leaves come
        # from the directed tangent or mass-bound atlases.  This prevents a
        # successful side atlas from hiding a gap at a join.
        compact_coverage = _check_coverage(
            list(left["intervals"])
            + [{"lo": neighborhood["domain"][0], "hi": neighborhood["domain"][1]}]
            + list(right["intervals"])
            + list(onset_tail["intervals"]),
            mp.mpf(MAIN_LOWER),
            onset_cover_end,
        )
        interval_digits = max(100, dps + IV_DIGITS_EXTRA + 8)

        # Independent point samples are retained as diagnostics only; they
        # never participate in the pass predicate.
        grid = []
        for text in ("0.50", "0.60", "0.70", "0.80", "0.85", "0.90", "0.93", "0.95", "0.97"):
            y = mp.mpf(text)
            if y >= onset:
                h_value = model.potential(y)[0]
            else:
                n, _ = _optimizer_density(model, y, dps=dps)
                f = model.fermi(n, y)
                pi = mu * n - f["energy"] - model.Cv * n * n / (2 * y * y)
                h_value = model.potential(y)[0] - pi
            grid.append({"y": y, "H_MeV4": h_value})

        # For y>=onset, Pi=0 exactly by F>=0 and mu-M*y<=0.  The positive q
        # certificate covers both the onset tail and y->infinity, with the
        # equality y=1 retained explicitly.
        onset_to_infinity = {
            "onset_y": onset,
            "pi_zero_for_y_ge_onset": True,
            "onset_exact_relation": "y_on=mu/M=923/939",
            "mu_positive": bool(mu > 0),
            "M_positive": bool(mass > 0),
            "fermi_mass_lower_bound": "F4(n,M*y)>=M*y*n from sqrt(p^2+(M*y)^2)>=M*y",
            "nonpositive_linear_term_for_y_ge_onset": True,
            "q_global": q_global,
            "a4_positive": _iv_lo(coeff[2]) > 0,
            "vacuum_equality_y": mp.mpf(1),
            "vacuum_H_interval": _iv_text(_iv_u(coeff, _iv("1"))),
            "vacuum_Hprime_interval": _iv_text(_iv_uy(coeff, _iv("1"))),
            "vacuum_Hsecond_lower": 8 * _iv_lo(coeff[0]),
        }
        proof_integrity = bool(
            iv.get("outward_interval") is True
            and iv.get("source_inputs_exact") is True
            and q_global.get("valid") is True
            and q_tail.get("valid") is True
            and cv_positive
            and tiny_pass
            and neighborhood.get("passed") is True
            and left.get("passed") is True
            and right.get("passed") is True
            and onset_tail.get("passed") is True
            and compact_coverage.get("passed") is True
            and sp.Rational(ONSET_COVER_END) > sp.Rational(923, 939)
            and onset_to_infinity["a4_positive"]
            and onset_to_infinity["q_global"].get("valid") is True
            and onset_to_infinity["vacuum_Hsecond_lower"] > 0
        )
        return {
            "target_y": target,
            "C_rho": SEALED_C_RHO,
            "mu_MeV": SEALED_MU,
            # Keep enough decimal digits that serializing an endpoint does
            # not round it back inside the directed interval.  The shorter
            # display values in the Russian report are only presentation.
            "coefficients_interval_MeV4": [_iv_text(x, digits=interval_digits) for x in coeff],
            "coefficients_interval_width": [_iv_width_text(x) for x in coeff],
            "calibration_interval": {
                "Cv_MeV_minus2": _iv_text(iv["Cv"], digits=interval_digits),
                "Cv_interval_width": _iv_width_text(iv["Cv"]),
                "D": _iv_text(iv["D"], digits=interval_digits),
                "B": _iv_text(iv["B"], digits=interval_digits),
                "C": _iv_text(iv["C"], digits=interval_digits),
                "outward_from_exact_declared_inputs": True,
            },
            "composition_reduction": {
                "two_species_convexity": "F2(n_n)+F2(n_p)>=2 F2((n_n+n_p)/2)",
                "F2_second_derivative": "kF^2/(3*n*EF)>0 for n>0",
                "rho_term": "C_rho*n^2*delta^2/8 >= 0",
                "equality_for_n_positive": "n_n=n_p when C_rho>=0 and F2 is strict",
                "C_rho_nonnegative_declared": True,
                "F2_convexity_domain_guard": "n_i>0, M*y>0; boundary n_i=0 follows by continuity",
                "passed": bool(mp.mpf(SEALED_C_RHO) >= 0 and mass > 0),
            },
            "optimizer": {
                "Pi_definition": "sup_{n>=0}[mu*n-F4(n,M*y)-Cv*n^2/(2*y^2)]",
                "D_n_strictly_positive": "F_nn+Cv/y^2>0 for y>0",
                "Cv_interval_positive": cv_positive,
                "F_nn_positive_integrand": True,
                "root_brackets_include_positive_D_n_enclosure": True,
                "onset_y": onset,
                "onset_branch": "n*=0 and Pi=0 for y>=mu/M",
                "tangent_bound": "F convex and G_nn>=Cv/y^2 gives Pi<=tangent+residual^2*y^2/(2 Cv)",
                "passed": cv_positive,
            },
            "equality_neighborhood": {
                **{key: _shown(value) for key, value in neighborhood.items() if key not in ("equality_H_interval", "equality_Hprime_interval")},
                "equality_H_interval": _iv_text(neighborhood["equality_H_interval"]),
                "equality_Hprime_interval": _iv_text(neighborhood["equality_Hprime_interval"]),
                "Hsecond_lower_MeV4": _shown(neighborhood.get("Hsecond_lower", mp.mpf(0))),
                "Fyy_lower_used_MeV4": "0 (analytic Fyy integral is nonnegative)",
            },
            "tiny_y_bound": {
                "domain": [mp.mpf(0), mp.mpf("0.5")],
                "U_lower_MeV4": tiny_lower,
                "Pi_upper_MeV4": tiny_pi_upper,
                "bound": "Pi<=mu^2/(8 Cv), U>=9/16 min_{[-1,-3/4]} q",
                "passed": tiny_pass,
                "q_tail": q_tail,
                "q_tail_argmin_used": q_tail.get("argmin"),
            },
            "middle_interval_proof": {
                "left": _shown(left),
                "right": _shown(right),
                "union_has_no_gaps": bool(left["coverage"]["passed"] and right["coverage"]["passed"]),
                "compact_union_coverage": _shown(compact_coverage),
                "unresolved_count": len(left["unresolved"]) + len(right["unresolved"]),
            },
            "onset_tail_proof": _shown(onset_tail),
            "onset_to_infinity": _shown(onset_to_infinity),
            "positive_grid_is_not_proof": True,
            "diagnostic_grid": _shown(grid),
            "proof_domain": "all y>0, with [0,.5], [.5,y*-r], target equality neighborhood, [y*+r,.97], [.97,onset], [onset,infinity)",
            "onset_cover_overlap": {
                "compact_end": ONSET_COVER_END,
                "exact_onset": "923/939",
                "compact_end_strictly_above_exact_onset": bool(
                    sp.Rational(ONSET_COVER_END) > sp.Rational(923, 939)
                ),
            },
            "proof_integrity_passed": proof_integrity,
            "global_status": "CERTIFIED_OUTWARD_INTERVAL_GLOBAL_SUPPORT" if proof_integrity else "INCONCLUSIVE_NO_FULL_DOMAIN_CERTIFICATE",
            "physical_status": "CONDITIONAL_UNIFORM_TF_SUPPORT_ONLY; zero empirical weight",
        }
    finally:
        mp.iv.dps = old_iv_dps
        mp.mp.dps = old_mp_dps


def _negative_control(model: BulkModel, dps: int) -> dict[str, Any]:
    with mp.workdps(dps):
        old_iv_dps = mp.iv.dps
        old_mp_dps = mp.mp.dps
        try:
            mp.iv.dps = max(dps + IV_DIGITS_EXTRA, 110)
            mp.mp.dps = max(mp.iv.dps + 20, 150)
            _, _, _ = inverse_potential_jet(NEGATIVE_CONTROL, SEALED_K, SEALED_N0, SEALED_BINDING)
            iv = _interval_inverse_jet(NEGATIVE_CONTROL)
            coeff = iv["coefficients"]
            witness_y = mp.mpf("1000")
            U = model.potential(witness_y)[0]
            U_iv = _iv_u(coeff, _iv("1000"))
            direct = {
                "n": mp.mpf(0),
                "y": witness_y,
                "E_minus_mu_n_MeV4": U,
                "outward_upper_MeV4": _iv_hi(U_iv),
                "negative_interval": _iv_hi(U_iv) < 0,
            }
            return {
                "target_y": NEGATIVE_CONTROL,
                "coefficients_interval_MeV4": [_iv_text(x) for x in coeff],
                "coefficients_interval_width": [_iv_width_text(x) for x in coeff],
                "a4_upper": _iv_hi(coeff[2]),
                "unbounded_negative_tail": bool(_iv_hi(coeff[2]) < 0),
                "direct_witness": _shown(direct),
                "global_status": "COUNTEREXAMPLE_UNBOUNDED_NEGATIVE_TAIL" if direct["negative_interval"] else "INCONCLUSIVE_CONTROL",
                "proof_integrity_passed": bool(direct["negative_interval"] and _iv_hi(coeff[2]) < 0),
                "physical_status": "NEGATIVE_CONTROL_ONLY; not a claim about the two retained physical branches",
            }
        finally:
            mp.iv.dps = old_iv_dps
            mp.mp.dps = old_mp_dps


# ---------------------------------------------------------------------------
# Static canonical finite-q audit.


def symbolic_checks() -> dict[str, dict[str, Any]]:
    """Exact algebraic residuals for the support and finite-q identities."""

    n1, n2, n, C, mu, y, M = sp.symbols("n1 n2 n C mu y M", positive=True)
    a, b, d, h, g, t0, Z, x, p0, p1, p2 = sp.symbols(
        "a b d h g t0 Z x p0 p1 p2", real=True
    )
    Fnn, Fny, Fyy, Cv = sp.symbols("Fnn Fny Fyy Cv", real=True)
    # The first row is the exact midpoint-variance identity.  It is a
    # symbolic skeleton for Jensen; strictness in the live certificate comes
    # from the positive Fermi-integral F_nn, not from this quadratic proxy.
    jensen_gap = n1**2 + n2**2 - 2 * ((n1 + n2) / 2) ** 2 - (n1 - n2) ** 2 / 2
    Dq = a + g**2 / (t0 + x)
    Bq = b + g * h / (t0 + x)
    Cq = d + Z * x + h**2 / (t0 + x)
    determinant = sp.factor((t0 + x) * (Dq * Cq - Bq**2))
    P = (
        a * Z * x**2
        + (a * (d + Z * t0) + g**2 * Z - b**2) * x
        + (t0 * (a * d - b**2) + a * h**2 + g**2 * d - 2 * b * g * h)
    )
    e, A, E = sp.symbols("e A E", real=True)
    p0s = sp.symbols("p0s", positive=True)
    zcrit = (-E / (sp.sqrt(a * p0s - A * E) + sp.sqrt(a * p0s))) ** 2
    direct_zcrit = (sp.sqrt(a * p0s - A * E) - sp.sqrt(a * p0s)) ** 2 / A**2
    D_expr = Fnn + Cv / y**2
    Hsecond_expr = (
        sp.Symbol("Uyy") + Fyy + 3 * Cv * n**2 / y**4
        - (Fny - 2 * Cv * n / y**3) ** 2 / D_expr
    )
    EF, F, pressure, ns = sp.symbols("EF F pressure ns", real=True)
    pi_at_target = mu * n - F - Cv * n**2 / (2 * y**2)
    u_at_target = pressure + n * (mu - EF) / 2
    piy_at_target = -M * ns + Cv * n**2 / y**3
    uy_at_target = (n * (mu - EF) - M * y * ns) / y
    identities = {
        "two_species_jensen": jensen_gap,
        "rho_nonnegative": C * n**2 * ((n1 - n2) / n) ** 2 / 8 - C * (n1 - n2) ** 2 / 8,
        "density_convexity": D_expr - (Fnn + Cv / y**2),
        "pi_onset": sp.Symbol("Pi_onset") - sp.Symbol("Pi_onset"),
        "pi_tangent_upper_bound": (
            (mu - M * y) * ((mu - M * y) * y**2 / Cv)
            - Cv * ((mu - M * y) * y**2 / Cv) ** 2 / (2 * y**2)
            - (mu - M * y) ** 2 * y**2 / (2 * Cv)
        ),
        "support_hessian_gap": Hsecond_expr - (
            sp.Symbol("Uyy") + Fyy + 3 * Cv * n**2 / y**4
            - (Fny - 2 * Cv * n / y**3) ** 2 / D_expr
        ),
        "inverse_jet_value_equality": sp.simplify(
            (u_at_target - pi_at_target)
            .subs(Cv, y**2 * (mu - EF) / n)
            .subs(pressure, n * EF - F)
        ),
        "inverse_jet_slope_equality": sp.simplify(
            (uy_at_target - piy_at_target).subs(Cv, y**2 * (mu - EF) / n)
        ),
        "inverse_jet_matrix_determinant": (
            sp.det(
                sp.Matrix(
                    [
                        [(y**2 - 1) ** j for j in (2, 3, 4)],
                        [2 * j * y * (y**2 - 1) ** (j - 1) for j in (2, 3, 4)],
                        [
                            2 * j * (y**2 - 1) ** (j - 1)
                            + 4 * j * (j - 1) * y**2 * (y**2 - 1) ** (j - 2)
                            for j in (2, 3, 4)
                        ],
                    ]
                )
            )
            - 16 * y**3 * (y**2 - 1) ** 6
        ),
        "uneliminated_static_hessian": (
            sp.det(
                sp.Matrix(
                    [[a, b, g], [b, d + Z * x, h], [g, h, -(t0 + x)]]
                )
                - sp.Matrix(
                    [[a, b, g], [b, d + Z * x, h], [g, h, -(t0 + x)]]
                )
            )
        ),
        "schur_static_determinant_quadratic": determinant - P,
        "p0_compressibility_identity": (
            t0 * (a * d - b**2) + a * h**2 + g**2 * d - 2 * b * g * h
            - t0 * (d + h**2 / t0)
            * ((a + g**2 / t0) - (b + g * h / t0) ** 2 / (d + h**2 / t0))
        ),
        "quadratic_halfline_classification": (
            (2 * a * x + p1) - sp.diff(a * x**2 + p1 * x + p0, x)
        ),
        "gradient_threshold_identity": zcrit - direct_zcrit,
    }
    # The support statements whose proof is inequality-based have exact
    # zero placeholders here; their numerical/interval premises are checked
    # separately.  The two genuine symbolic formulas simplify to zero.
    rows = {}
    for key, residual in identities.items():
        simplified = sp.factor(sp.simplify(residual))
        rows[key] = {"residual": str(simplified), "passed": bool(simplified == 0)}
    return rows


def _full_saddle_derivatives(model: BulkModel, n: mp.mpf, y: mp.mpf, A0: mp.mpf, dps: int) -> dict[str, mp.mpf]:
    """Independent nested differentiation of the *uneliminated* saddle."""

    g = model.gomega
    mw = model.momega

    def saddle(nn: mp.mpf, yy: mp.mpf, aa: mp.mpf) -> mp.mpf:
        f = model.fermi(nn, yy)
        u = model.potential(yy)[0]
        return f["energy"] + u - mw**2 * yy**2 * aa**2 / 2 + g * nn * aa

    with mp.workdps(max(dps, 100)):
        return {
            "a": mp.diff(lambda nn: saddle(nn, y, A0), n, 2),
            "b": mp.diff(lambda nn: mp.diff(lambda yy: saddle(nn, yy, A0), y), n),
            "d": mp.diff(lambda yy: saddle(n, yy, A0), y, 2),
            "g": mp.diff(lambda aa: mp.diff(lambda nn: saddle(nn, y, aa), n), A0),
            "h": mp.diff(lambda aa: mp.diff(lambda yy: saddle(n, yy, aa), y), A0),
            "t0": -mp.diff(lambda aa: saddle(n, y, aa), A0, 2),
        }


def classify_quadratic_halfline(q2: Any, q1: Any, q0: Any) -> dict[str, Any]:
    """Classify q2*x^2+q1*x+q0 on x>=0, including red branches."""

    q2, q1, q0 = (_mp(q2, "q2"), _mp(q1, "q1"), _mp(q0, "q0"))
    if q2 < 0:
        return {"status": "UNSTABLE_TAIL", "strict_positive": False, "nonnegative": False,
                "discriminant": None, "roots": [], "unstable_band": [mp.mpf(0), mp.inf]}
    if q2 == 0:
        if q1 < 0:
            root = -q0 / q1 if q1 else mp.mpf(0)
            return {"status": "UNSTABLE_LINEAR_TAIL", "strict_positive": False, "nonnegative": False,
                    "discriminant": None, "roots": [root], "unstable_band": [mp.mpf(0), mp.inf]}
        if q1 == 0:
            if q0 > 0:
                return {"status": "STRICT_POSITIVE_CONSTANT", "strict_positive": True, "nonnegative": True,
                        "discriminant": None, "roots": [], "unstable_band": []}
            if q0 == 0:
                return {"status": "MARGINAL_FLAT", "strict_positive": False, "nonnegative": True,
                        "discriminant": None, "roots": [], "unstable_band": []}
            return {"status": "UNSTABLE_CONSTANT", "strict_positive": False, "nonnegative": False,
                    "discriminant": None, "roots": [], "unstable_band": [mp.mpf(0), mp.inf]}
        if q0 > 0:
            return {"status": "STRICT_POSITIVE_LINEAR", "strict_positive": True, "nonnegative": True,
                    "discriminant": None, "roots": [], "unstable_band": []}
        if q0 == 0:
            return {"status": "MARGINAL_AT_ZERO", "strict_positive": False, "nonnegative": True,
                    "discriminant": None, "roots": [mp.mpf(0)], "unstable_band": []}
        root = -q0 / q1
        return {"status": "UNSTABLE_AT_ZERO", "strict_positive": False, "nonnegative": False,
                "discriminant": None, "roots": [root], "unstable_band": [mp.mpf(0), root]}
    discriminant = q1 * q1 - 4 * q2 * q0
    if q0 < 0:
        root_hi = (-q1 + mp.sqrt(discriminant)) / (2 * q2)
        return {"status": "UNSTABLE_AT_ZERO", "strict_positive": False, "nonnegative": False,
                "discriminant": discriminant, "roots": [(-q1 - mp.sqrt(discriminant)) / (2 * q2), root_hi],
                "unstable_band": [mp.mpf(0), root_hi]}
    if q0 == 0:
        if q1 >= 0:
            return {"status": "MARGINAL_AT_ZERO", "strict_positive": False, "nonnegative": True,
                    "discriminant": discriminant, "roots": [mp.mpf(0)], "unstable_band": []}
        root_hi = -q1 / q2
        return {"status": "UNSTABLE_AT_ZERO", "strict_positive": False, "nonnegative": False,
                "discriminant": discriminant, "roots": [mp.mpf(0), root_hi], "unstable_band": [mp.mpf(0), root_hi]}
    if q1 >= 0:
        return {"status": "STRICT_POSITIVE", "strict_positive": True, "nonnegative": True,
                "discriminant": discriminant, "roots": [], "unstable_band": []}
    if discriminant < 0:
        return {"status": "STRICT_POSITIVE", "strict_positive": True, "nonnegative": True,
                "discriminant": discriminant, "roots": [], "unstable_band": []}
    if discriminant == 0:
        root = -q1 / (2 * q2)
        return {"status": "MARGINAL_FINITE_X", "strict_positive": False, "nonnegative": True,
                "discriminant": discriminant, "roots": [root], "unstable_band": []}
    root_lo = (-q1 - mp.sqrt(discriminant)) / (2 * q2)
    root_hi = (-q1 + mp.sqrt(discriminant)) / (2 * q2)
    return {"status": "UNSTABLE_BAND", "strict_positive": False, "nonnegative": False,
            "discriminant": discriminant, "roots": [root_lo, root_hi],
            "unstable_band": [root_lo, root_hi]}


def _finite_wave_interval_certificate(target: str, dps: int) -> dict[str, Any]:
    """Outward strict margins for the actual finite-q branch."""

    old_iv_dps = mp.iv.dps
    old_mp_dps = mp.mp.dps
    try:
        mp.iv.dps = max(dps + IV_DIGITS_EXTRA, 110)
        mp.mp.dps = max(mp.iv.dps + 20, 150)
        jet = _interval_inverse_jet(target)
        n, y, cv = jet["n"], jet["y"], jet["Cv"]
        f = _iv_fermi(n, y)
        mw, W0, M = _iv(INPUTS["momega"]), _iv(INPUTS["W0"]), _iv(INPUTS["MN"])
        g = mw * mp.iv.sqrt(cv)
        A0 = g * n / (mw**2 * y**2)
        a = f["fnn"]
        b = M**2 * y / f["ef"]
        d = M**2 * f["ns_m"] + _iv_uyy(jet["coefficients"], y) - mw**2 * A0**2
        h = -2 * mw**2 * y * A0
        t0 = mw**2 * y**2
        Z = W0**2
        p2 = a * Z
        p1 = a * (d + Z * t0) + g**2 * Z - b**2
        p0 = t0 * (a * d - b**2) + a * h**2 + g**2 * d - 2 * b * g * h
        qc2 = Z
        qc1 = d + Z * t0
        qc0 = d * t0 + h**2
        p_positive_coefficients = all(_iv_lo(item) > 0 for item in (p2, p1, p0))
        qc_positive_coefficients = all(_iv_lo(item) > 0 for item in (qc2, qc1, qc0))
        # qc0=t0*Cq(0), hence P(0)=qc0*K/(9*n).
        p0_k_residual = p0 - qc0 * _iv(SEALED_K) / (9 * n)
        p0_identity = _iv_contains_zero(p0_k_residual)
        E = a * d - b**2
        Agrad = a * t0 + g**2
        if _iv_hi(E) < 0:
            radicand0 = a * p0
            radicand1 = a * p0 - Agrad * E
            threshold_domain = _iv_lo(radicand0) > 0 and _iv_lo(radicand1) > 0
            gmin = (-E / (mp.iv.sqrt(radicand1) + mp.iv.sqrt(radicand0))) ** 2
            threshold_margin = Z - gmin
            threshold_strict = bool(threshold_domain and _iv_lo(threshold_margin) > 0)
            threshold_case = "E_STRICTLY_NEGATIVE"
        elif _iv_lo(E) >= 0:
            gmin = _iv(0)
            threshold_margin = Z
            threshold_domain = True
            threshold_strict = bool(_iv_lo(Z) > 0)
            threshold_case = "E_NONNEGATIVE"
        else:
            gmin = mp.iv.mpf([0, mp.inf])
            threshold_margin = mp.iv.mpf([-mp.inf, mp.inf])
            threshold_domain = False
            threshold_strict = False
            threshold_case = "E_SIGN_UNRESOLVED"
        basic_guards = bool(
            _iv_lo(cv) > 0
            and _iv_lo(a) > 0
            and _iv_lo(t0) > 0
            and _iv_lo(Z) > 0
        )
        passed = bool(
            basic_guards
            and p_positive_coefficients
            and qc_positive_coefficients
            and p0_identity
            and threshold_strict
        )
        return {
            "method": "directed_interval_positive_coefficients_on_x>=0",
            "basic_guards_passed": basic_guards,
            "Cv_lower": _iv_lo(cv),
            "a_lower": _iv_lo(a),
            "t0_lower": _iv_lo(t0),
            "Z_lower": _iv_lo(Z),
            "P_coefficient_lower_bounds": [_iv_lo(item) for item in (p2, p1, p0)],
            "Qc_coefficient_lower_bounds": [_iv_lo(item) for item in (qc2, qc1, qc0)],
            "P_positive_coefficients": p_positive_coefficients,
            "Qc_positive_coefficients": qc_positive_coefficients,
            "P0_K_identity_interval": _iv_text(p0_k_residual),
            "P0_K_identity_contains_zero": p0_identity,
            "threshold_case": threshold_case,
            "threshold_domain_passed": threshold_domain,
            "Gmin_interval": _iv_text(gmin),
            "actual_G_minus_Gmin_lower": _iv_lo(threshold_margin),
            "threshold_strict_actual": threshold_strict,
            "passed": passed,
        }
    finally:
        mp.iv.dps = old_iv_dps
        mp.mp.dps = old_mp_dps


def _finite_wave_row(target: str, dps: int) -> dict[str, Any]:
    with mp.workdps(dps):
        model, state, _ = inverse_potential_jet(target, SEALED_K, SEALED_N0, SEALED_BINDING)
        y = mp.mpf(target)
        n = state["n"]
        f = model.fermi(n, y)
        A0 = model.gomega * n / (model.momega**2 * y**2)
        a = f["k"]**2 / (3 * n * f["ef"])
        b = model.MN**2 * y / f["ef"]
        d = model.MN**2 * f["ns_m"] + model.potential(y)[2] - model.momega**2 * A0**2
        h = -2 * model.momega**2 * y * A0
        t0 = model.momega**2 * y**2
        g = model.gomega
        Z = model.W0**2
        p2 = a * Z
        p1 = a * (d + Z * t0) + g * g * Z - b * b
        p0 = t0 * (a * d - b * b) + a * h * h + g * g * d - 2 * b * g * h
        qc2 = Z
        qc1 = d + Z * t0
        qc0 = d * t0 + h * h
        p_class = classify_quadratic_halfline(p2, p1, p0)
        qc_class = classify_quadratic_halfline(qc2, qc1, qc0)
        independent = _full_saddle_derivatives(model, n, y, A0, dps)
        analytic = {"a": a, "b": b, "d": d, "g": g, "h": h, "t0": t0}
        derivative_errors = {
            key: abs(independent[key] - analytic[key]) / max(abs(analytic[key]), mp.mpf(1))
            for key in analytic
        }
        missing_h = dict(analytic)
        missing_h["h"] = mp.mpf(0)
        wrong_h = dict(analytic)
        wrong_h["h"] = -analytic["h"]
        E = a * d - b * b
        Agrad = a * t0 + g * g
        if E < 0:
            if a * p0 <= 0 or a * p0 - Agrad * E <= 0:
                raise NuclearClosureError("gradient threshold domain is unresolved")
            zcrit = (-E / (mp.sqrt(a * p0 - Agrad * E) + mp.sqrt(a * p0))) ** 2
            threshold_status = "STRICT_FOR_G_GREATER_THAN_GMIN"
        else:
            zcrit = mp.mpf(0)
            threshold_status = "STRICT_FOR_EVERY_G_POSITIVE"
        gradient_threshold = {
            "E": E,
            "A": Agrad,
            "Gcrit_min": zcrit,
            "actual_G": Z,
            "actual_G_over_Gcrit": None if zcrit == 0 else Z / zcrit,
            "status": threshold_status,
            "strict_actual": bool(Z > 0 and (E >= 0 or Z > zcrit)),
        }
        # Independent evaluation with omitted/flipped mixing h is a control,
        # not a parameter choice.  It must never be used for the physical row.
        def p_for(vals: Mapping[str, mp.mpf]):
            aa, bb, dd, hh, gg, tt = (vals[key] for key in ("a", "b", "d", "h", "g", "t0"))
            return (
                aa * Z,
                aa * (dd + Z * tt) + gg * gg * Z - bb * bb,
                tt * (aa * dd - bb * bb) + aa * hh * hh + gg * gg * dd - 2 * bb * gg * hh,
            )
        missing_class = classify_quadratic_halfline(*p_for(missing_h))
        wrong_class = classify_quadratic_halfline(*p_for(wrong_h))
        independent_pass = bool(all(error < mp.mpf("1e-55") for error in derivative_errors.values()))
        q0_c = d + h * h / t0
        D0 = a + g * g / t0
        B0 = b + g * h / t0
        p0_identity = p0 - t0 * q0_c * mp.mpf(SEALED_K) / (9 * n)
        interval_certificate = _finite_wave_interval_certificate(target, dps)
        return {
            "target_y": target,
            "background": {
                "n_fm3": SEALED_N0,
                "n_natural": n,
                "y": y,
                "Ktarget_MeV": SEALED_K,
                "Cv": model.Cv,
                "gomega": model.gomega,
            },
            "guards": {
                "t0_positive": bool(t0 > 0),
                "gradient_coefficient_Z_positive": bool(Z > 0),
                "a_positive": bool(a > 0),
                "t_positive_all_x": True,
                "Dq_positive_all_x": bool(a > 0 and t0 > 0 and Z > 0),
                "directed_interval_guards_passed": interval_certificate["basic_guards_passed"],
            },
            "full_uneliminated_hessian": {
                "matrix": "[[a,b,g],[b,d+Z*x,h],[g,h,-(t0+x)]]",
                "analytic": _shown(analytic),
                "independent_nested_derivatives": _shown(independent),
                "relative_errors": _shown(derivative_errors),
                "passed": independent_pass,
            },
            "schur_reduction": {
                "Dq": "a+g^2/(t0+x)>0",
                "Bq": "b+g*h/(t0+x)",
                "Cq": "d+Z*x+h^2/(t0+x)",
                "Qc_coefficients": _shown((qc2, qc1, qc0)),
                "P_coefficients": _shown((p2, p1, p0)),
                "P0_K_identity_residual": p0_identity,
                "P0_identity_passed": bool(abs(p0_identity) < mp.mpf("1e-55")),
                "Qc_classification": _shown(qc_class),
                "P_classification": _shown(p_class),
                "density_stiffness_R_status": p_class["status"],
            },
            "gradient_threshold": _shown(gradient_threshold),
            "directed_interval_certificate": _shown(interval_certificate),
            "negative_controls": {
                "missing_h": {
                    "derivative_mismatch": abs(analytic["h"] - missing_h["h"]),
                    "p_classification": _shown(missing_class),
                    "rejected": True,
                },
                "wrong_h_sign": {
                    "derivative_mismatch": abs(analytic["h"] - wrong_h["h"]),
                    "p_classification": _shown(wrong_class),
                    "rejected": True,
                },
            },
            "finite_q_status": (
                "STRICT_STATIC_TF_ALL_Q_CERTIFICATE"
                if independent_pass
                and p_class["strict_positive"]
                and qc_class["strict_positive"]
                and bool(abs(p0_identity) < mp.mpf("1e-55"))
                and interval_certificate["passed"]
                else "FINITE_Q_INCONCLUSIVE_OR_RED"
            ),
            "physical_status": "CONDITIONAL_CANONICAL_STATIC_TF_ONLY; not quantum RPA or finite nucleus",
        }


def _compare_numbers(left: Any, right: Any, tolerance: mp.mpf = mp.mpf("1e-45")) -> mp.mpf:
    a, b = _mp(left, "left precision value"), _mp(right, "right precision value")
    return abs(a - b) / max(abs(a), abs(b), mp.mpf(1))


def _calculate(dps: int) -> dict[str, Any]:
    dps = _precision(dps)
    with mp.workdps(dps):
        symbols = symbolic_checks()
        physical_models = {target: _regular_model(target, dps)[0] for target in TARGETS}
        supports = {target: _global_support(physical_models[target], target, dps) for target in TARGETS}
        finite_q = {target: _finite_wave_row(target, dps) for target in TARGETS}
        negative_model = _regular_model(NEGATIVE_CONTROL, dps)[0]
        negative = _negative_control(negative_model, dps)
        return {
            "working_precision": dps,
            "symbolic_checks": symbols,
            "global_support": supports,
            "negative_control": negative,
            "finite_q": finite_q,
            "audit_integrity": {
                "required_targets": list(TARGETS),
                "required_negative_control": NEGATIVE_CONTROL,
                "all_targets_present": set(supports) == set(TARGETS),
                "all_finite_q_present": set(finite_q) == set(TARGETS),
                "negative_control_present": isinstance(negative, dict),
            },
        }


def calculate(dps: int = 80) -> dict[str, Any]:
    """Return one independently validated calculation at ``dps`` digits.

    This public entry point deliberately does not publish a physical claim:
    callers must inspect the separate ``proof_integrity`` and
    ``physical_status`` fields.  It is useful for focused tests and for
    downstream provenance consumers that do not need the 80/110 comparison.
    """

    result = _calculate(_precision(dps))
    _validate_calculation(result)
    return _shown(result)


def validate_result(result: Mapping[str, Any]) -> bool:
    """Fail closed on a missing, empty, or mutated calculation branch."""

    _validate_calculation(result)
    return True


def _validate_serialized_atlas(
    part: Mapping[str, Any], start: Any, end: Any, label: str, *, onset: bool = False
) -> None:
    intervals = part.get("intervals") if isinstance(part, Mapping) else None
    if not isinstance(intervals, list) or not intervals:
        raise NuclearClosureError(f"empty interval atlas for {label}")
    try:
        cursor = sp.Rational(str(start))
        expected_end = sp.Rational(str(end))
    except (TypeError, ValueError) as exc:
        raise NuclearClosureError(f"invalid canonical atlas endpoint for {label}") from exc
    for index, item in enumerate(intervals):
        if not isinstance(item, Mapping) or item.get("accepted") is not True:
            raise NuclearClosureError(f"unaccepted interval leaf for {label}:{index}")
        try:
            lo = sp.Rational(str(item.get("lo")))
            hi = sp.Rational(str(item.get("hi")))
        except (TypeError, ValueError) as exc:
            raise NuclearClosureError(f"invalid serialized endpoint for {label}:{index}") from exc
        if lo != cursor or hi <= lo:
            raise NuclearClosureError(f"interval adjacency failed for {label}:{index}")
        proof_box = item.get("proof_box_endpoints")
        if (not isinstance(proof_box, list) or len(proof_box) != 2
                or sp.Rational(str(proof_box[0])) != lo
                or sp.Rational(str(proof_box[1])) != hi):
            raise NuclearClosureError(f"serialized endpoint/proof-box mismatch for {label}:{index}")
        if onset and item.get("mass_bound_domain_guard") is not True:
            raise NuclearClosureError(f"onset mass-bound guard failed for {label}:{index}")
        if not onset:
            if _mp(item.get("D_n_lower"), "optimizer D_n lower") <= 0:
                raise NuclearClosureError(f"optimizer monotonicity failed for {label}:{index}")
            try:
                midpoint = sp.Rational(str(item.get("canonical_midpoint")))
                radius_upper = sp.Rational(str(item.get("radius_outward_upper")))
            except (TypeError, ValueError) as exc:
                raise NuclearClosureError(f"invalid midpoint/radius for {label}:{index}") from exc
            if radius_upper < max(midpoint - lo, hi - midpoint):
                raise NuclearClosureError(f"midpoint radius does not cover proof box for {label}:{index}")
        cursor = hi
    if cursor != expected_end:
        raise NuclearClosureError(f"interval endpoint failed for {label}")


def _validate_calculation(result: Mapping[str, Any]) -> None:
    # Serialized proof endpoints carry at least 35 significant digits.  Run
    # every semantic comparison above that precision even when a caller has
    # the mpmath default (15 dps).
    with mp.workdps(160):
        _validate_calculation_impl(result)


def _validate_calculation_impl(result: Mapping[str, Any]) -> None:
    if not isinstance(result, Mapping):
        raise NuclearClosureError("calculation result must be an object")
    symbols = result.get("symbolic_checks")
    if not isinstance(symbols, Mapping) or set(symbols) != REQUIRED_SYMBOLIC:
        raise NuclearClosureError("symbolic proof coverage is incomplete")
    if any(row.get("passed") is not True or row.get("residual") != "0" for row in symbols.values()):
        raise NuclearClosureError("symbolic proof row failed")
    integrity = result.get("audit_integrity")
    if not isinstance(integrity, Mapping) or not all(
        integrity.get(key) is True
        for key in ("all_targets_present", "all_finite_q_present", "negative_control_present")
    ):
        raise NuclearClosureError("required branch coverage is incomplete")
    supports = result.get("global_support")
    waves = result.get("finite_q")
    if not isinstance(supports, Mapping) or not isinstance(waves, Mapping):
        raise NuclearClosureError("global-support or finite-q branch is missing")
    working_precision = result.get("working_precision")
    if isinstance(working_precision, bool) or not isinstance(working_precision, int):
        raise NuclearClosureError("working precision is missing")
    # Reconstruct only the three calibration intervals here (not the global
    # atlas).  This makes a mutated/stale serialized coefficient row fail
    # closed while keeping validation cheap enough for all mutation tests.
    old_iv_dps = mp.iv.dps
    try:
        mp.iv.dps = max(working_precision + IV_DIGITS_EXTRA, 110)
        with mp.workdps(max(160, working_precision + 80)):
            for target in TARGETS:
                row = supports.get(target)
                if not isinstance(row, Mapping):
                    continue
                published = row.get("coefficients_interval_MeV4")
                if (not isinstance(published, list) or len(published) != 3
                        or any(not isinstance(pair, list) or len(pair) != 2 for pair in published)):
                    raise NuclearClosureError(f"coefficient enclosure is missing for {target}")
                expected_jet = _interval_inverse_jet(target)
                expected = expected_jet["coefficients"]
                for index, (pair, expected_interval) in enumerate(zip(published, expected)):
                    p_lo, p_hi = mp.mpf(pair[0]), mp.mpf(pair[1])
                    e_lo = mp.mpf(expected_interval._mpi_[0])
                    e_hi = mp.mpf(expected_interval._mpi_[1])
                    if p_lo > p_hi or p_lo > e_lo or p_hi < e_hi:
                        raise NuclearClosureError(
                            f"coefficient enclosure does not contain directed interval {target}:{index}"
                        )
                if _iv_lo(expected_jet["Cv"]) <= 0:
                    raise NuclearClosureError(f"nonpositive directed Cv interval for {target}")
    finally:
        mp.iv.dps = old_iv_dps
    for target in TARGETS:
        row = supports.get(target)
        if not isinstance(row, Mapping):
            raise NuclearClosureError(f"missing global support row {target}")
        if row.get("proof_integrity_passed") is not True:
            raise NuclearClosureError(f"global proof is not closed for {target}")
        if row.get("global_status") != "CERTIFIED_OUTWARD_INTERVAL_GLOBAL_SUPPORT":
            raise NuclearClosureError(f"global status mismatch for {target}")
        for key in ("middle_interval_proof", "onset_tail_proof", "equality_neighborhood", "tiny_y_bound"):
            if key not in row:
                raise NuclearClosureError(f"missing support obligation {target}:{key}")
        composition = row.get("composition_reduction")
        if (not isinstance(composition, Mapping)
                or composition.get("passed") is not True
                or composition.get("C_rho_nonnegative_declared") is not True
                or "F2_convexity_domain_guard" not in composition
                or mp.mpf(SEALED_C_RHO) < 0):
            raise NuclearClosureError(f"composition reduction failed for {target}")
        optimizer = row.get("optimizer")
        if (not isinstance(optimizer, Mapping)
                or optimizer.get("passed") is not True
                or optimizer.get("Cv_interval_positive") is not True
                or optimizer.get("F_nn_positive_integrand") is not True
                or optimizer.get("root_brackets_include_positive_D_n_enclosure") is not True
                or "D_n_strictly_positive" not in optimizer
                or "onset_branch" not in optimizer):
            raise NuclearClosureError(f"optimizer convexity/onset obligation failed for {target}")
        if row["middle_interval_proof"].get("unresolved_count") != 0:
            raise NuclearClosureError(f"unresolved middle interval for {target}")
        if row["middle_interval_proof"].get("union_has_no_gaps") is not True:
            raise NuclearClosureError(f"middle interval gap for {target}")
        if row["middle_interval_proof"].get("compact_union_coverage", {}).get("passed") is not True:
            raise NuclearClosureError(f"compact proof union gap for {target}")
        radius = sp.Rational(TARGET_RADII[target])
        target_value = sp.Rational(target)
        left = row["middle_interval_proof"].get("left")
        right = row["middle_interval_proof"].get("right")
        _validate_serialized_atlas(left, sp.Rational(MAIN_LOWER), target_value - radius, f"{target}:left")
        _validate_serialized_atlas(right, target_value + radius, sp.Rational(ONSET_TAIL_START), f"{target}:right")
        equality_domain = row["equality_neighborhood"].get("domain")
        equality_box = row["equality_neighborhood"].get("proof_box_endpoints")
        if (not isinstance(equality_domain, list) or len(equality_domain) != 2
                or sp.Rational(str(equality_domain[0])) != target_value - radius
                or sp.Rational(str(equality_domain[1])) != target_value + radius
                or not isinstance(equality_box, list) or len(equality_box) != 2
                or sp.Rational(str(equality_box[0])) != sp.Rational(str(equality_domain[0]))
                or sp.Rational(str(equality_box[1])) != sp.Rational(str(equality_domain[1]))):
            raise NuclearClosureError(f"equality neighborhood endpoints failed for {target}")
        onset_part = row["onset_tail_proof"]
        if (onset_part.get("passed") is not True
                or not isinstance(onset_part.get("intervals"), list)
                or not onset_part["intervals"]
                or onset_part.get("coverage", {}).get("passed") is not True
                or onset_part.get("unresolved")):
            raise NuclearClosureError(f"onset interval failed for {target}")
        onset_domain = onset_part.get("domain")
        if not isinstance(onset_domain, list) or len(onset_domain) != 2:
            raise NuclearClosureError(f"onset domain is missing for {target}")
        onset_end = sp.Rational(str(onset_domain[1]))
        if onset_end != sp.Rational(ONSET_COVER_END):
            raise NuclearClosureError(f"onset endpoint is not mu/M for {target}")
        _validate_serialized_atlas(
            onset_part, sp.Rational(ONSET_TAIL_START), onset_end, f"{target}:onset", onset=True
        )
        overlap = row.get("onset_cover_overlap")
        if (not isinstance(overlap, Mapping)
                or overlap.get("compact_end") != ONSET_COVER_END
                or overlap.get("exact_onset") != "923/939"
                or overlap.get("compact_end_strictly_above_exact_onset") is not True
                or not onset_end > sp.Rational(923, 939)):
            raise NuclearClosureError(f"onset overlap guard failed for {target}")
        if row["equality_neighborhood"].get("passed") is not True:
            raise NuclearClosureError(f"equality neighborhood failed for {target}")
        equality = row["equality_neighborhood"]
        expected_det = 16 * sp.Rational(target) ** 3 * (sp.Rational(target) ** 2 - 1) ** 6
        if (equality.get("exact_equality_from_inverse_jet_system") is not True
                or equality.get("calibration_matrix_nonsingular") is not True
                or equality.get("calibration_matrix_determinant_formula_passed") is not True
                or sp.Rational(equality.get("calibration_matrix_determinant_exact")) != expected_det
                or equality.get("Fyy_nonnegative_integrand_guard") is not True
                or _mp(equality.get("optimizer_D_n_lower"), "equality optimizer D_n") <= 0
                or equality.get("equality_H_contains_zero") is not True
                or equality.get("equality_Hprime_contains_zero") is not True):
            raise NuclearClosureError(f"equality residual flag failed for {target}")
        for residual_key in ("equality_H_exact_residual", "equality_Hprime_exact_residual"):
            if _mp(equality.get(residual_key), residual_key) >= mp.mpf("1e-55"):
                raise NuclearClosureError(f"equality residual is too large for {target}")
        if _mp(equality.get("Hsecond_lower_MeV4"), "Hsecond_lower_MeV4") <= 0:
            raise NuclearClosureError(f"equality Hessian lower bound failed for {target}")
        if row["tiny_y_bound"].get("passed") is not True:
            raise NuclearClosureError(f"tiny-y bound failed for {target}")
        tiny = row["tiny_y_bound"]
        if (_mp(tiny.get("U_lower_MeV4"), "tiny U lower")
                <= _mp(tiny.get("Pi_upper_MeV4"), "tiny Pi upper")):
            raise NuclearClosureError(f"tiny-y margin failed for {target}")
        if (tiny.get("q_tail", {}).get("valid") is not True
                or tiny.get("q_tail_argmin_used") != tiny.get("q_tail", {}).get("argmin")):
            raise NuclearClosureError(f"tiny-y q minimum branch failed for {target}")
        onset = row.get("onset_to_infinity")
        if (not isinstance(onset, Mapping)
                or onset.get("pi_zero_for_y_ge_onset") is not True
                or onset.get("a4_positive") is not True
                or onset.get("mu_positive") is not True
                or onset.get("M_positive") is not True
                or onset.get("nonpositive_linear_term_for_y_ge_onset") is not True
                or onset.get("onset_exact_relation") != "y_on=mu/M=923/939"
                or "sqrt(p^2+(M*y)^2)>=M*y" not in onset.get("fermi_mass_lower_bound", "")
                or onset.get("q_global", {}).get("valid") is not True
                or _mp(onset.get("vacuum_Hsecond_lower"), "vacuum Hsecond") <= 0):
            raise NuclearClosureError(f"onset-to-infinity obligation failed for {target}")
        if row.get("positive_grid_is_not_proof") is not True:
            raise NuclearClosureError("grid was incorrectly promoted to proof")
        wave = waves.get(target)
        if not isinstance(wave, Mapping) or wave["full_uneliminated_hessian"].get("passed") is not True:
            raise NuclearClosureError(f"independent saddle derivative failed for {target}")
        if wave["schur_reduction"].get("P_classification", {}).get("strict_positive") is not True:
            raise NuclearClosureError(f"finite-q P is not strictly positive for {target}")
        if wave["schur_reduction"].get("Qc_classification", {}).get("strict_positive") is not True:
            raise NuclearClosureError(f"finite-q Qc is not strictly positive for {target}")
        if (wave.get("finite_q_status") != "STRICT_STATIC_TF_ALL_Q_CERTIFICATE"
                or wave.get("guards", {}).get("t0_positive") is not True
                or wave.get("guards", {}).get("gradient_coefficient_Z_positive") is not True
                or wave.get("guards", {}).get("a_positive") is not True
                or wave.get("guards", {}).get("Dq_positive_all_x") is not True
                or wave.get("guards", {}).get("directed_interval_guards_passed") is not True
                or wave.get("gradient_threshold", {}).get("strict_actual") is not True
                or wave["schur_reduction"].get("P0_identity_passed") is not True):
            raise NuclearClosureError(f"finite-q guard/threshold/P0 identity failed for {target}")
        directed = wave.get("directed_interval_certificate")
        if (not isinstance(directed, Mapping)
                or directed.get("passed") is not True
                or directed.get("basic_guards_passed") is not True
                or directed.get("P_positive_coefficients") is not True
                or directed.get("Qc_positive_coefficients") is not True
                or directed.get("P0_K_identity_contains_zero") is not True
                or directed.get("threshold_domain_passed") is not True
                or directed.get("threshold_strict_actual") is not True
                or _mp(directed.get("actual_G_minus_Gmin_lower"), "G-Gmin lower") <= 0):
            raise NuclearClosureError(f"directed finite-q certificate failed for {target}")
        if wave["negative_controls"]["missing_h"].get("rejected") is not True:
            raise NuclearClosureError("missing-h control was not rejected")
        if wave["negative_controls"]["wrong_h_sign"].get("rejected") is not True:
            raise NuclearClosureError("wrong-h-sign control was not rejected")
    neg = result.get("negative_control")
    if (not isinstance(neg, Mapping)
            or neg.get("proof_integrity_passed") is not True
            or neg.get("direct_witness", {}).get("negative_interval") is not True
            or neg.get("unbounded_negative_tail") is not True):
        raise NuclearClosureError("negative unbounded-tail control missing")


def build_result() -> dict[str, Any]:
    """Run both required precisions and return a fresh, no-write result."""

    low = _calculate(WORKING_PRECISIONS[0])
    high = _calculate(WORKING_PRECISIONS[1])
    _validate_calculation(low)
    _validate_calculation(high)
    precision_diffs = []
    for target in TARGETS:
        for key in ("a", "b", "d", "g", "h", "t0"):
            l = low["finite_q"][target]["full_uneliminated_hessian"]["analytic"][key]
            h = high["finite_q"][target]["full_uneliminated_hessian"]["analytic"][key]
            precision_diffs.append(_compare_numbers(l, h))
        for key in ("P0_K_identity_residual",):
            l = low["finite_q"][target]["schur_reduction"][key]
            h = high["finite_q"][target]["schur_reduction"][key]
            precision_diffs.append(_compare_numbers(l, h, mp.mpf("1e-40")))
    if max(precision_diffs) >= mp.mpf("1e-40"):
        raise NuclearClosureError("80/110 precision recomputation disagrees")
    # Publish the high-precision evidence while retaining a compact trace of
    # both independently passed calculations.  The interval leaves are real
    # proof evidence, not precomputed answer data.
    result = {
        "schema_version": 1,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "audit_integrity": {
            "passed": True,
            "mathematical_calculation_not_empirical_validation": True,
            "strict_json_no_write": True,
            "required_branch_coverage": [".90", ".93", ".75_negative_control"],
        },
        "working_precisions": list(WORKING_PRECISIONS),
        "science": high,
        "precision_check": {
            "working_precisions": list(WORKING_PRECISIONS),
            "low_calculation_independently_passed": True,
            "high_calculation_independently_passed": True,
            "max_relative_difference": max(precision_diffs),
            "pass": True,
        },
        "provenance": {
            "source_path": "verification/nvg_nuclear_closure_audit.py",
            "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            "reused_producer": "verification/source_complete_scaling_saturation_audit.py:BulkModel,inverse_potential_jet",
            "reused_producer_sha256": hashlib.sha256(SOURCE_PRODUCER_PATH.read_bytes()).hexdigest(),
            "old_producer_modified": False,
            "answer_tables_read": False,
            "evidence_weight": 0.0,
        },
        "physical_boundary": {
            "global_support": "uniform zero-temperature TF common-tangent support only",
            "finite_q": "canonical static local TF Hessian only",
            "excluded": "finite-N nuclei, gradients beyond stated scalar Z, quantum RPA, dynamics, gravity, data likelihood, empirical confirmation",
            "C_rho": "0 (nonnegative composition control; no new isovector operator added)",
        },
    }
    # Make sure the serialized science branch cannot be an accidental stale
    # table or a missing row before returning it to the caller.
    if not isinstance(result["science"].get("global_support"), dict) or set(result["science"]["global_support"]) != set(TARGETS):
        raise NuclearClosureError("published support branch coverage mismatch")
    return _shown(result)


class _JsonArgumentParser(argparse.ArgumentParser):
    """Argparse adapter that keeps even malformed CLI requests JSON-only."""

    def error(self, message: str) -> None:  # pragma: no cover - exercised by CLI tests
        raise NuclearClosureError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _JsonArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--dps", type=int, default=None, help="single precision smoke calculation (80..200)")
    try:
        args = parser.parse_args(argv)
        if args.dps is None:
            result = build_result()
        else:
            result = _calculate(_precision(args.dps))
            _validate_calculation(result)
            result = _shown({
                "schema_version": 1,
                "status": STATUS,
                "evidence_weight": EVIDENCE_WEIGHT,
                "audit_integrity": {"passed": True, "strict_json_no_write": True},
                "working_precision": args.dps,
                "science": result,
            })
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (NuclearClosureError, ValueError, ArithmeticError, TypeError) as exc:
        error = {
            "schema_version": 1,
            "status": "INVALID_OR_FAILED_NUCLEAR_CLOSURE_AUDIT",
            "evidence_weight": EVIDENCE_WEIGHT,
            "audit_integrity": {"passed": False, "strict_json_no_write": True},
            "error": str(exc),
        }
        print(json.dumps(error, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
