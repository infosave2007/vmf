#!/usr/bin/env python3
"""Algebraic gradient matching for the accepted retarded Hartree matrix.

No kernel is calculated here.  For a fixed-background coefficient map and a
full complex response, remove the known longitudinal-vector Schur block and
reconstruct ``Z`` from ``chi=C/(D*C-B**2)``, ``C=C0+Z*K``, ``K=q**2-z**2``:
``Z=(B**2*chi/(D*chi-1)-C0)/K`` and
``dZ/dchi=-B**2/(K*(D*chi-1)**2)``.  The algebraic result stays complex or
non-positive when the input says so; it is never clipped into a physical
positive-real gradient.  Units follow ``nvg_retarded_response``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Any, Mapping


COEFFICIENT_KEYS = ("a", "b", "d0", "g", "h", "t0")
# A conditioning guard is a numerical precision boundary, not a physical
# uncertainty.  It is deliberately close to double precision roundoff.
DEFAULT_CONDITIONING_LIMIT = 128.0 * sys.float_info.epsilon


class RetardedGradientMatchingError(ValueError):
    """Malformed, singular, or non-retarded input to the matching helper."""


def _finite_real(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise RetardedGradientMatchingError(f"{name} must be a finite real, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetardedGradientMatchingError(f"{name} must be a finite real") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise RetardedGradientMatchingError(f"{name} must be {qualifier}")
    return result


def _finite_complex(value: Any, name: str, *, retarded: bool = False) -> complex:
    if isinstance(value, bool):
        raise RetardedGradientMatchingError(f"{name} must be finite complex, not bool")
    if isinstance(value, (tuple, list)):
        if len(value) != 2 or any(isinstance(item, bool) for item in value):
            raise RetardedGradientMatchingError(f"{name} must be a finite complex number")
        try:
            value = complex(_finite_real(value[0], f"Re {name}"),
                            _finite_real(value[1], f"Im {name}"))
        except RetardedGradientMatchingError:
            raise
    else:
        try:
            value = complex(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RetardedGradientMatchingError(f"{name} must be a finite complex number") from exc
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise RetardedGradientMatchingError(f"{name} must be finite")
    if retarded and value.imag < 0.0:
        raise RetardedGradientMatchingError("retarded z must lie on or above the real axis")
    return value


def _coefficients(coefficients: Mapping[str, Any]) -> dict[str, complex]:
    if not isinstance(coefficients, Mapping):
        raise RetardedGradientMatchingError("coefficients must be a mapping")
    missing = [key for key in COEFFICIENT_KEYS if key not in coefficients]
    if missing:
        raise RetardedGradientMatchingError(f"coefficients missing {', '.join(missing)}")
    result = {}
    for key in COEFFICIENT_KEYS:
        result[key] = _finite_complex(coefficients[key], key)
    if result["t0"] == 0.0:
        raise RetardedGradientMatchingError("t0 must be nonzero")
    return result


def _conditioning(value: complex, scale: float) -> float:
    return abs(value) / max(abs(scale), 1.0, sys.float_info.min)


def _conditioning_limit(value: Any) -> float:
    limit = _finite_real(value, "conditioning_limit")
    if limit <= 0.0:
        raise RetardedGradientMatchingError("conditioning_limit must be positive")
    return limit


@dataclass(frozen=True)
class RetardedGradientMatch:
    """Reconstruction and explicit conditioning/physical-family diagnostics."""

    reconstructed_Z: complex
    derivative_dZ_dchi: complex
    sensitivity_abs: float
    relative_sensitivity: float | None
    q: float
    z: complex
    chi: complex
    K: complex
    D: complex
    B: complex
    C0: complex
    C_recovered: complex
    denominator: complex
    lightcone_conditioning: float
    denominator_conditioning: float
    subtraction_conditioning: float
    accepted: bool
    positive_real_candidate: bool
    status: str

    @property
    def Z(self) -> complex:
        """Short alias for callers that use the coefficient symbol."""
        return self.reconstructed_Z

    @property
    def dZ_dchi(self) -> complex:
        return self.derivative_dZ_dchi


def _match_blocks(D: complex, B: complex, C0: complex, q: float, z: complex, chi: complex,
                  limit: float) -> RetardedGradientMatch:
    K = q * q - z * z
    lightcone = _conditioning(K, max(q * q, abs(z * z)))
    if K == 0.0:
        raise RetardedGradientMatchingError("light-cone K=q^2-z^2 is singular")
    denominator = D * chi - 1.0
    denominator_cond = _conditioning(denominator, max(abs(D * chi), 1.0))
    if denominator == 0.0:
        raise RetardedGradientMatchingError("D*chi-1 is singular")
    C_recovered = B * B * chi / denominator
    separation = C_recovered - C0
    subtraction_cond = _conditioning(separation, max(abs(C_recovered), abs(C0)))
    reconstructed = separation / K
    derivative = -B * B / (K * denominator * denominator)
    if not all(math.isfinite(value.real) and math.isfinite(value.imag)
               for value in (K, denominator, C_recovered, reconstructed, derivative)):
        raise RetardedGradientMatchingError("matching arithmetic produced a nonfinite value")
    sensitivity = abs(derivative)
    relative = None
    if chi != 0.0 and reconstructed != 0.0:
        relative = abs(derivative * chi / reconstructed)
    positive_real = reconstructed.real > 0.0 and reconstructed.imag == 0.0
    if lightcone <= limit:
        status, accepted = "REFUSED_LIGHT_CONE_CONDITIONING", False
    elif denominator_cond <= limit:
        status, accepted = "REFUSED_DENOMINATOR_CONDITIONING", False
    elif subtraction_cond <= limit:
        status, accepted = "REFUSED_C_SUBTRACTION_CONDITIONING", False
    elif not positive_real:
        status, accepted = "DIAGNOSTIC_COMPLEX_OR_NONPOSITIVE_Z", True
    else:
        status, accepted = "RETARDED_GRADIENT_MATCH_OK", True
    return RetardedGradientMatch(
        reconstructed_Z=reconstructed, derivative_dZ_dchi=derivative,
        sensitivity_abs=sensitivity, relative_sensitivity=relative,
        q=q, z=z, chi=chi, K=K, D=D, B=B, C0=C0,
        C_recovered=C_recovered, denominator=denominator,
        lightcone_conditioning=lightcone, denominator_conditioning=denominator_cond,
        subtraction_conditioning=subtraction_cond, accepted=accepted,
        positive_real_candidate=positive_real, status=status)


def match_retarded_blocks(D: Any, B: Any, C0: Any, q: Any, z: Any, chi: Any,
                          *, conditioning_limit: Any = DEFAULT_CONDITIONING_LIMIT) -> RetardedGradientMatch:
    """Match from already-eliminated ``D,B,C0`` blocks.

    ``B=0`` is a decoupled density/scalar channel and is refused even though
    the rational expression could be evaluated.  Near-zero B is retained as
    a sensitivity diagnostic, rather than being silently regularized.
    """
    limit = _conditioning_limit(conditioning_limit)
    Dv, Bv, C0v = (_finite_complex(value, name) for value, name in
                   ((D, "D"), (B, "B"), (C0, "C0")))
    qv = _finite_real(q, "q", positive=True)
    zv = _finite_complex(z, "z", retarded=True)
    chiv = _finite_complex(chi, "chi")
    K = qv * qv - zv * zv
    if K == 0.0:
        raise RetardedGradientMatchingError("light-cone K=q^2-z^2 is singular")
    if Bv == 0.0:
        raise RetardedGradientMatchingError("B=0 is a decoupled channel")
    return _match_blocks(Dv, Bv, C0v, qv, zv, chiv, limit)


def infer_gradient_coefficient(coefficients: Mapping[str, Any], q: Any, z: Any, chi: Any,
                               *, conditioning_limit: Any = DEFAULT_CONDITIONING_LIMIT) -> RetardedGradientMatch:
    """Infer ``Z`` from a full retarded response and fixed Hessian coefficients."""
    limit = _conditioning_limit(conditioning_limit)
    c = _coefficients(coefficients)
    qv = _finite_real(q, "q", positive=True)
    zv = _finite_complex(z, "z", retarded=True)
    K = qv * qv - zv * zv
    if K == 0.0:
        raise RetardedGradientMatchingError("light-cone K=q^2-z^2 is singular")
    L = c["t0"] + K
    if L == 0.0:
        raise RetardedGradientMatchingError("vector-elimination factor t0+K is singular")
    L_conditioning = abs(L) / max(abs(c["t0"]), abs(K), sys.float_info.min)
    if L_conditioning <= limit:
        raise RetardedGradientMatchingError(
            "vector-elimination conditioning is too poor for double precision"
        )
    D = c["a"] + c["g"] * c["g"] * (1.0 - zv * zv / (qv * qv)) / L
    B = c["b"] + c["g"] * c["h"] / L
    C0 = c["d0"] + c["h"] * c["h"] * (1.0 - zv * zv / c["t0"]) / L
    return _match_blocks(D, B, C0, qv, zv, _finite_complex(chi, "chi"), limit)


invert_retarded_gradient_coefficient = infer_gradient_coefficient
invert_gradient_coefficient = infer_gradient_coefficient


__all__ = [
    "COEFFICIENT_KEYS", "DEFAULT_CONDITIONING_LIMIT", "RetardedGradientMatch",
    "RetardedGradientMatchingError", "match_retarded_blocks", "infer_gradient_coefficient",
    "invert_retarded_gradient_coefficient", "invert_gradient_coefficient",
]
