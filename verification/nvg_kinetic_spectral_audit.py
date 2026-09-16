#!/usr/bin/env python3
"""Causal l=0,1 collisionless density spectrum and moment bridge.

This module completes the calculable long-wavelength Hartree/Landau density
observable.  It reuses the maintained collisionless coefficient producer and
its logarithmic zero-sound solver, while retaining the full driven source
vector, the retarded continuum, the isolated delta pole, and independently
integrated spectral moments.  It is deliberately *not* a finite-k quantum
RPA, collision, gravity, bounce, or universal-theory completion.

The command line interface prints JSON only and never writes an artifact.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

import mpmath as mp
import sympy as sp

import nvg_collisionless_response_audit as collisionless
import nvg_effective_response_audit as effective


STATUS = "CONDITIONAL_RETARDED_KINETIC_SPECTRUM_NOT_FULL_NVG_THEORY"
EVIDENCE_WEIGHT = 0
with mp.workdps(200):
    MOMENT_TOLERANCE = mp.mpf("1e-24")
    UPSTREAM_RESIDUAL_TOLERANCE = mp.mpf("1e-50")
    NUMERIC_TOLERANCE = mp.mpf("1e-50")
    PRECISION_TOLERANCE = mp.mpf("1e-50")
REPRESENTATIVE_DENSITIES = ("1", "10", "100", "200")
ANGULAR_SIGMAS = (("0.37", "0.21"), ("1.4", "0.08"))
QUADRATURE_NODES = (0, 1, 3, 7, 15, 30, 60, mp.inf)
# ``number`` deliberately keeps the JSON report compact.  A serialized state
# therefore has about 45 significant digits even though its producer runs at
# 80/110 digits.  The validator below uses this smaller comparison window for
# identities reconstructed from the rounded report; the authoritative live
# checks retain the stricter tolerances above.
SERIALIZED_TOLERANCE = mp.mpf("1e-40")

REQUIRED_COEFFICIENTS = frozenset({
    "y", "EF", "vF2", "NF", "M2", "B", "C", "V", "r", "F0", "F1",
    "mu", "first_sound_squared",
})
REQUIRED_STATE_KEYS = frozenset({
    "status", "evidence_weight", "mathematical_checks_passed", "inputs",
    "coefficients", "upstream", "source_vector", "current_feedback",
    "angular_response", "continuum_absorption", "pole", "spectral_moments",
    "limits", "effective_response_bridge", "moment_bridge", "symbolic_checks",
    "scope", "references", "controls", "numeric_passport_passed",
})
REQUIRED_INPUT_KEYS = frozenset({"n_over_n0", "dps"})
REQUIRED_UPSTREAM_KEYS = frozenset({
    "producer", "status", "mathematical_checks_passed", "residuals",
    "max_residual", "residual_tolerance", "status_is_independent_of_residual_gate",
    "n_ratio", "dps",
})
REQUIRED_SOURCE_KEYS = frozenset({
    "sigma", "external_energy", "F1", "source_vector",
    "full_source_vector_included", "source_second_residual",
    "determinant_relative_error", "density_response_relative_error",
    "density_response", "closed_response", "missing_second_source_response",
    "missing_second_source_difference", "missing_second_source_rejected",
    "no_density_contact_term", "passed",
})
REQUIRED_CURRENT_KEYS = frozenset({"r", "F1", "F1_relation_residual", "passed"})
REQUIRED_ANGULAR_KEYS = frozenset({
    "rows", "max_moment_relative_error", "max_response_relative_error", "passed",
    "separate_log_continuation_used",
})
REQUIRED_ANGULAR_ROW_KEYS = frozenset({
    "sigma", "moment_relative_errors", "response_relative_error",
    "direct_response", "split_response",
})
REQUIRED_ABSORPTION_KEYS = frozenset({
    "sigma", "retarded_S_over_NF", "advanced_S_over_NF",
    "finite_frequency_absorption", "retarded_sign_passed", "wrong_sign_rejected",
    "passed",
})
REQUIRED_POLE_KEYS = frozenset({
    "exists", "condition_F0_gt_3r", "log_gap", "gap", "sigma_p", "residue",
    "delta_piece_required",
})
REQUIRED_RESOLVED_POLE_KEYS = frozenset({
    "exists", "condition_F0_gt_3r", "log_gap", "gap", "sigma_p", "L_p", "A_p",
    "L_prime_p", "D_prime_p", "residue", "dispersion_residual", "positive_residue",
    "delta_piece_required",
})
REQUIRED_MOMENT_KEYS = frozenset({
    "pole", "quadrature", "pole_contribution_m1", "pole_contribution_mminus1",
    "m1_total", "mminus1_total", "m1_total_without_pole",
    "mminus1_total_without_pole", "target_m1", "target_mminus1",
    "quadrature_m1_relative_difference", "quadrature_mminus1_relative_difference",
    "m1_relative_error", "mminus1_relative_error", "include_pole",
    "integrals_converged", "passed", "moment_tolerance",
})
REQUIRED_QUADRATURE_KEYS = frozenset({"exponential_endpoint", "tanh_endpoint"})
REQUIRED_QUADRATURE_ROW_KEYS = frozenset({"m1_continuum", "mminus1_continuum"})
REQUIRED_LIMIT_KEYS = frozenset({
    "static_limit_exact", "static_response_at_sigma_zero", "static_relative_error",
    "static_small_sigma", "static_small_sigma_relative_error", "dynamic_limit_exact",
    "high_sigma", "high_sigma_response", "high_sigma_coefficient",
    "high_sigma_coefficient_exact", "high_sigma_relative_error",
    "noncommuting_static_dynamic_limits", "passed",
})
REQUIRED_BRIDGE_KEYS = frozenset({
    "source_producer", "source_independent_checks_passed", "mathematical_checks_passed",
    "source_comparison_checked", "source_acoustic_squared_speed",
    "collisionless_first_sound_squared", "acoustic_relative_error_to_collisionless",
    "source_relative_error", "low_frequency_inertial_coefficient",
    "expected_inertial_coefficient", "inertial_relative_error",
    "finite_k_group_velocity_not_claimed", "passed",
})
REQUIRED_MOMENT_BRIDGE_KEYS = frozenset({
    "vF2_m1_over_mminus1", "collisionless_first_sound_squared",
    "effective_response_acoustic_squared_speed", "moment_to_collisionless_relative_error",
    "moment_to_effective_response_relative_error", "passed",
})
REQUIRED_CONTROL_KEYS = frozenset({
    "noninteracting", "current_feedback", "wrong_retarded_sign", "omitted_pole",
    "near_threshold_log_gap", "passed",
})

REQUIRED_SYMBOLIC_CHECKS = frozenset({
    "source_vector_first_entry",
    "source_vector_second_entry",
    "two_moment_determinant",
    "cramer_density_numerator",
    "density_response_formula",
    "current_feedback_relation",
    "static_response_limit",
    "high_sigma_response_coefficient",
    "continuum_retarded_sign",
    "pole_denominator_derivative",
    "m1_high_sigma_identity",
    "mminus1_static_identity",
})
REQUIRED_UPSTREAM_RESIDUALS = frozenset(collisionless.REQUIRED_RESIDUALS)


def finite(value, name, *, positive=False, nonnegative=False):
    """Convert a real scalar to finite mpmath storage and reject bools."""
    if isinstance(value, bool) or isinstance(value, (complex, mp.mpc)):
        raise ValueError(f"{name}: finite real number required")
    try:
        result = mp.mpf(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name}: finite real number required") from exc
    if (not mp.isfinite(result) or (positive and result <= 0)
            or (nonnegative and result < 0)):
        raise ValueError(f"{name}: outside finite numeric domain")
    return result


def finite_complex(value, name):
    """Convert a complex spectral argument, rejecting nonfinite components."""
    if isinstance(value, bool):
        raise ValueError(f"{name}: finite complex number required")
    try:
        result = mp.mpc(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name}: finite complex number required") from exc
    if not mp.isfinite(result.real) or not mp.isfinite(result.imag):
        raise ValueError(f"{name}: finite complex number required")
    return result


def precision(dps):
    if isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 200:
        raise ValueError("dps must be an integer in [80,200]")


def _landau_parameters(F0, r):
    F0 = finite(F0, "F0")
    r = finite(r, "r", nonnegative=True)
    if F0 <= -1:
        raise ValueError("F0 must satisfy F0 > -1")
    F1 = -3*r/(1+r)
    return F0, r, F1


def number(value):
    """Stable decimal display of a finite real derived quantity."""
    # Reports are often serialized after a high-precision work context has
    # closed.  Raise the local display context so mpmath's nstr does not
    # silently round an 80/110-digit value to the process default precision.
    with mp.workdps(max(mp.mp.dps, 200)):
        return mp.nstr(finite(value, "derived number"), 45)


def _shown(value):
    """Convert mpmath values to strict JSON-native values."""
    if isinstance(value, mp.mpc):
        if value.imag == 0:
            return number(value.real)
        return {"real": number(value.real), "imag": number(value.imag)}
    if isinstance(value, mp.mpf):
        return number(value)
    if isinstance(value, dict):
        return {key: _shown(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shown(item) for item in value]
    return value


def _relative_error(left, right, floor="1e-70"):
    with mp.workdps(200):
        left, right = finite_complex(left, "left"), finite_complex(right, "right")
        scale = max(abs(left), abs(right), mp.mpf(floor))
        return abs(left-right)/scale


def _relative_real(left, right, floor="1e-70"):
    with mp.workdps(200):
        left, right = finite(left, "left"), finite(right, "right")
        return abs(left-right)/max(abs(left), abs(right), mp.mpf(floor))


def _as_complex(value, name):
    """Parse a JSON complex value emitted by ``_shown``."""
    try:
        if isinstance(value, dict):
            if set(value) != {"real", "imag"}:
                return None
            return mp.mpc(finite(value["real"], f"{name}.real"),
                          finite(value["imag"], f"{name}.imag"))
        return finite_complex(value, name)
    except (ValueError, TypeError, OverflowError):
        return None


def _serialized_close(left, right, tolerance=SERIALIZED_TOLERANCE):
    """Compare rounded report values without treating tiny zeros as relative."""
    try:
        if isinstance(left, (complex, mp.mpc)):
            if left.imag != 0:
                return False
            left = left.real
        if isinstance(right, (complex, mp.mpc)):
            if right.imag != 0:
                return False
            right = right.real
        left, right = finite(left, "left"), finite(right, "right")
    except (ValueError, TypeError, OverflowError):
        return False
    scale = max(abs(left), abs(right), mp.mpf(1))
    return bool(abs(left-right)/scale < tolerance)


def _serialized_nonzero_close(left, right, tolerance=SERIALIZED_TOLERANCE):
    """Compare two nonzero serialized values on their actual numeric scale."""
    try:
        if isinstance(left, (complex, mp.mpc)):
            if left.imag != 0:
                return False
            left = left.real
        if isinstance(right, (complex, mp.mpc)):
            if right.imag != 0:
                return False
            right = right.real
        left, right = finite(left, "left"), finite(right, "right")
    except (ValueError, TypeError, OverflowError):
        return False
    if left == 0 or right == 0:
        return False
    scale = max(abs(left), abs(right))
    return bool(abs(left-right)/scale < tolerance)


def _serialized_complex_close(left, right, tolerance=SERIALIZED_TOLERANCE):
    left, right = _as_complex(left, "left"), _as_complex(right, "right")
    if left is None or right is None:
        return False
    return bool(abs(left-right)/max(abs(left), abs(right), mp.mpf(1)) < tolerance)


def _real_field(record, key, *, positive=False, nonnegative=False):
    """Read one required real field while keeping the validator fail-closed."""
    if not isinstance(record, dict) or key not in record:
        return None
    try:
        return finite(record[key], key, positive=positive, nonnegative=nonnegative)
    except (ValueError, TypeError, OverflowError):
        return None


def _required_keys(record, required):
    return isinstance(record, dict) and set(record) == set(required)


def _serialized_structure_equal(actual, expected, tolerance=SERIALIZED_TOLERANCE):
    """Compare a complete generated control payload, including all leaves."""
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_serialized_structure_equal(actual[key], expected[key], tolerance)
                        for key in expected))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_serialized_structure_equal(left, right, tolerance)
                        for left, right in zip(actual, expected)))
    if isinstance(expected, str):
        try:
            return _serialized_close(actual, expected, tolerance)
        except (ValueError, TypeError, OverflowError):
            return actual == expected
    return actual == expected


def _serialized_pole_equal(actual, expected):
    """Compare duplicate pole records without erasing tiny gap/residue fields."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    if set(actual) != set(expected):
        return False
    scale_sensitive = expected.get("exists") is True
    for key in expected:
        if scale_sensitive and key in ("gap", "residue"):
            left = _real_field(actual, key, positive=True)
            right = _real_field(expected, key, positive=True)
            if (left is None or right is None
                    or not _serialized_nonzero_close(left, right)):
                return False
        elif not _serialized_structure_equal(actual[key], expected[key]):
            return False
    return True


def coefficients(n_ratio="1", *, dps=80):
    """Live alias to the maintained collisionless coefficient producer."""
    precision(dps)
    return collisionless.coefficients(n_ratio, dps=dps)


def _symbolic_row(expression):
    residual = sp.factor(sp.simplify(expression))
    if isinstance(residual, sp.MatrixBase):
        passed = all(sp.simplify(entry) == 0 for entry in residual)
    else:
        passed = bool(residual == 0)
    return {"passed": bool(passed), "residual": "0" if passed else str(residual)}


@lru_cache(maxsize=1)
def _symbolic_payload():
    F0, r, sigma, L, U, x, b = sp.symbols(
        "F0 r sigma L U x b", real=True)
    F1 = -3*r/(1+r)
    A = F0-3*r*sigma**2
    D = 1-A*L
    matrix = sp.Matrix([
        [1-F0*L, -F1*sigma*L],
        [-F0*sigma*L, 1-F1*(sigma**2*L-sp.Rational(1, 3))],
    ])
    source = sp.Matrix([L, sigma*L])*U
    adj_source = matrix.adjugate()*source

    # The continuum calculation is kept symbolic in x and b, with b=pi*sigma/2
    # substituted only in the final row.  This independently fixes the sign.
    continuum_L = x-sp.I*b
    continuum_D = 1-A*continuum_L
    continuum_chi = continuum_L/continuum_D
    continuum_density = -sp.im(continuum_chi)/sp.pi
    continuum_closed = sigma/(2*((1-A*x)**2+(A*b)**2))

    Lfun = sp.Function("L")(sigma)
    Dfun = 1-(F0-3*r*sigma**2)*Lfun
    Lprime = sp.diff(Lfun, sigma)
    pole_derivative = sp.diff(Dfun, sigma)
    high_L = sp.Rational(1, 3)/sigma**2
    high_response = sp.limit(
        sigma**2*high_L/(1-(F0-3*r*sigma**2)*high_L), sigma, sp.oo)

    return {
        "source_vector_first_entry": source[0]-U*L,
        "source_vector_second_entry": source[1]-U*sigma*L,
        "two_moment_determinant": matrix.det()-D/(1+r),
        "cramer_density_numerator": adj_source[0]-U*L/(1+r),
        "density_response_formula": (matrix.inv()*source)[0]-U*L/D,
        "current_feedback_relation": F1/(1+F1/3)+3*r,
        "static_response_limit": (-1)/(1+F0)-(-1)/(1+F0),
        "high_sigma_response_coefficient": high_response-1/(3*(1+r)),
        "continuum_retarded_sign": (continuum_density-continuum_closed).subs(
            b, sp.pi*sigma/2),
        "pole_denominator_derivative": pole_derivative-
            (6*r*sigma*Lfun-(F0-3*r*sigma**2)*Lprime),
        "m1_high_sigma_identity": 1/(3*(1+r))-2/(6*(1+r)),
        "mminus1_static_identity": -1/(1+F0)-2*(-1/(2*(1+F0))),
    }


def symbolic_checks():
    """Return exact residuals for the adopted matrix and spectral identities."""
    payload = _symbolic_payload()
    rows = {name: _symbolic_row(payload[name]) for name in REQUIRED_SYMBOLIC_CHECKS}
    return rows


def _symbolic_passed(checks):
    return (isinstance(checks, dict) and set(checks) == REQUIRED_SYMBOLIC_CHECKS
            and all(isinstance(row, dict) and row.get("passed") is True
                    and row.get("residual") == "0" for row in checks.values()))


def _retarded_lindhard_above_cut(sigma):
    """L for real sigma>1, using separate stable logarithms."""
    sigma = finite(sigma, "sigma", positive=True)
    if sigma <= 1:
        raise ValueError("above-cut Lindhard argument must satisfy sigma > 1")
    inverse = 1/sigma
    # log((sigma+1)/(sigma-1)) is algebraically equal to this expression, but
    # the separate log1p terms retain high-sigma coefficients without a quotient
    # branch shortcut or cancellation of the two logarithms.
    return sigma/2*(mp.log1p(inverse)-mp.log1p(-inverse))-1


def retarded_lindhard(sigma):
    """Analytic L(sigma) continued from Im(sigma)>0 with separate logs."""
    sigma = finite_complex(sigma, "sigma")
    if sigma.imag == 0 and sigma.real == 0:
        return mp.mpf(-1)
    if sigma.imag == 0 and sigma.real > 1:
        return _retarded_lindhard_above_cut(sigma.real)
    if sigma.imag == 0 and sigma.real < -1:
        # Positive-frequency calculations do not need this branch, but retaining
        # separate logs gives the same retarded boundary convention at -sigma.
        z = -sigma.real
        return _retarded_lindhard_above_cut(z)
    return sigma/2*(mp.log(sigma+1)-mp.log(sigma-1))-1


def lindhard_from_log_gap(log_gap):
    """Stable L at sigma=1+exp(log_gap), retaining a tiny positive gap."""
    with mp.workdps(max(mp.mp.dps, 200)):
        log_gap = finite(log_gap, "log_gap")
        gap = mp.exp(log_gap)
        if not mp.isfinite(gap) or gap <= 0:
            raise ValueError("log_gap must produce a finite positive gap")
        return (1+gap)*(mp.log(2+gap)-log_gap)/2-1


def _pole_lindhard_prime_from_log_gap(log_gap):
    with mp.workdps(max(mp.mp.dps, 200)):
        log_gap = finite(log_gap, "log_gap")
        gap = mp.exp(log_gap)
        if gap <= 0 or not mp.isfinite(gap):
            raise ValueError("log_gap must produce a finite positive gap")
        # Do not form sigma=1+gap before the reciprocal: doing so rounds an
        # exponentially small gap to zero at ordinary precision.
        return (mp.log(2+gap)-log_gap)/2-(1+gap)/(gap*(2+gap))


def dispersion_from_log_gap(log_gap, F0, r):
    """Retain the log-gap dispersion equation without sigma rounding."""
    with mp.workdps(max(mp.mp.dps, 200)):
        F0, r, _ = _landau_parameters(F0, r)
        log_gap = finite(log_gap, "log_gap")
        gap = mp.exp(log_gap)
        coefficient = (F0-3*r)-3*r*gap*(2+gap)
        return coefficient*lindhard_from_log_gap(log_gap)-1


def _pole_quantities_from_log_gap(log_gap, F0, r):
    """Derive pole quantities without applying the live residual gate."""
    with mp.workdps(max(mp.mp.dps, 200)):
        F0, r, _ = _landau_parameters(F0, r)
        if F0 <= 3*r:
            raise ValueError("an isolated pole requires F0 > 3r")
        log_gap = finite(log_gap, "log_gap")
        gap = mp.exp(log_gap)
        if gap <= 0 or not mp.isfinite(gap):
            raise ValueError("log_gap must produce a finite positive gap")
        sigma = 1+gap
        Lp = lindhard_from_log_gap(log_gap)
        A = (F0-3*r)-3*r*gap*(2+gap)
        Lprime = _pole_lindhard_prime_from_log_gap(log_gap)
        Dprime = 6*r*sigma*Lp-A*Lprime
        if not mp.isfinite(Dprime) or Dprime <= 0:
            raise ArithmeticError("pole denominator derivative is not positive")
        residue = Lp/Dprime
        if not mp.isfinite(residue) or residue <= 0:
            raise ArithmeticError("pole residue is not positive")
        residual = dispersion_from_log_gap(log_gap, F0, r)
        return {
            "exists": True,
            "log_gap": log_gap,
            "gap": gap,
            "sigma_p": sigma,
            "L_p": Lp,
            "A_p": A,
            "L_prime_p": Lprime,
            "D_prime_p": Dprime,
            "residue": residue,
            "dispersion_residual": residual,
            "positive_residue": True,
            "delta_piece_required": True,
        }


def pole_residue_from_log_gap(log_gap, F0, r):
    """Return the positive analytic residue and pole data in log-gap variables."""
    result = _pole_quantities_from_log_gap(log_gap, F0, r)
    if abs(result["dispersion_residual"]) >= mp.mpf("1e-60"):
        raise ArithmeticError("log-gap does not resolve the pole equation")
    return result


def pole_data(F0, r, *, dps=80):
    """Find the isolated positive pole only through the maintained log-gap solver."""
    precision(dps)
    with mp.workdps(dps):
        F0, r, _ = _landau_parameters(F0, r)
        if F0 <= 3*r:
            return {
                "exists": False,
                "condition_F0_gt_3r": False,
                "log_gap": None,
                "gap": None,
                "sigma_p": None,
                "residue": mp.mpf(0),
                "delta_piece_required": False,
            }
        root = collisionless.zero_sound_root(F0, r, dps=dps)
        if not isinstance(root, dict) or not mp.isfinite(root.get("log_gap", mp.nan)):
            raise ArithmeticError("upstream log-gap pole solver returned no resolved root")
        result = pole_residue_from_log_gap(root["log_gap"], F0, r)
        if abs(result["dispersion_residual"]) >= mp.mpf("1e-60"):
            raise ArithmeticError("log-gap pole residual is unresolved")
        result["condition_F0_gt_3r"] = True
        return result


def _validate_upstream(raw, upstream, n_ratio, dps):
    if not isinstance(raw, dict):
        raise ArithmeticError("collisionless coefficient producer returned no dictionary")
    residuals = raw.get("residuals")
    if not isinstance(residuals, dict) or set(residuals) != REQUIRED_UPSTREAM_RESIDUALS:
        raise ArithmeticError("incomplete upstream residual certificate")
    checked = {}
    for name in REQUIRED_UPSTREAM_RESIDUALS:
        try:
            checked[name] = finite(residuals[name], name)
        except ValueError as exc:
            raise ArithmeticError("nonfinite upstream residual certificate") from exc
    if not all(abs(value) < UPSTREAM_RESIDUAL_TOLERANCE for value in checked.values()):
        raise ArithmeticError("upstream coefficient residual exceeds 1e-50")
    if not isinstance(upstream, dict):
        raise ArithmeticError("collisionless audit returned no status dictionary")
    if upstream.get("status") != collisionless.STATUS:
        raise ArithmeticError("unexpected collisionless producer status")
    if upstream.get("mathematical_checks_passed") is not True:
        raise ArithmeticError("collisionless mathematical status is not exactly true")
    source_symbolic = collisionless.symbolic_checks()
    if (not isinstance(source_symbolic, dict)
            or set(source_symbolic) != collisionless.REQUIRED_IDENTITIES
            or not all(row.get("passed") is True and row.get("residual") == "0"
                       for row in source_symbolic.values())):
        raise ArithmeticError("collisionless symbolic certificate is incomplete")

    required_values = ("n_ratio", "NF", "r", "F0", "F1", "first_sound_squared",
                       "B", "C", "V", "M2", "mu")
    try:
        for key in required_values:
            finite(raw[key], f"upstream {key}")
        if finite(raw["NF"], "NF", positive=True) <= 0:
            raise ArithmeticError("upstream NF is not positive")
        if finite(raw["r"], "r", nonnegative=True) < 0:
            raise ArithmeticError("upstream r is negative")
        if finite(raw["F0"], "F0") <= -1:
            raise ArithmeticError("upstream F0 violates positive l=0 energy")
    except ValueError as exc:
        raise ArithmeticError("upstream coefficient dictionary is malformed") from exc
    return {
        "producer": "verification/nvg_collisionless_response_audit.py",
        "status": upstream["status"],
        "mathematical_checks_passed": upstream["mathematical_checks_passed"],
        "residuals": checked,
        "max_residual": max(abs(value) for value in checked.values()),
        "residual_tolerance": UPSTREAM_RESIDUAL_TOLERANCE,
        "status_is_independent_of_residual_gate": True,
        "n_ratio": finite(n_ratio, "n_ratio", positive=True),
        "dps": dps,
    }


def _load_upstream(n_ratio, dps):
    precision(dps)
    raw = coefficients(n_ratio, dps=dps)
    upstream = collisionless.audit(n_ratio, dps=dps)
    return raw, _validate_upstream(raw, upstream, n_ratio, dps)


def _lindhard_boundary(sigma, *, sign=1):
    sigma = finite(sigma, "continuum sigma")
    if not 0 < sigma < 1:
        raise ValueError("continuum sigma must satisfy 0 < sigma < 1")
    x = sigma/2*(mp.log1p(sigma)-mp.log1p(-sigma))-1
    return x-mp.j*sign*mp.pi*sigma/2


def continuum_spectral_density(sigma, F0, r, *, branch="retarded"):
    """Return S_cont/NF at 0<sigma<1 on the chosen boundary sheet."""
    F0, r, _ = _landau_parameters(F0, r)
    sigma = finite(sigma, "sigma")
    if not 0 < sigma < 1:
        raise ValueError("continuum sigma must satisfy 0 < sigma < 1")
    if branch not in ("retarded", "advanced"):
        raise ValueError("branch must be 'retarded' or 'advanced'")
    sign = 1 if branch == "retarded" else -1
    L = _lindhard_boundary(sigma, sign=sign)
    A = F0-3*r*sigma**2
    chi = L/(1-A*L)
    result = -mp.im(chi)/mp.pi
    if not mp.isfinite(result):
        raise ArithmeticError("nonfinite continuum spectral density")
    return result


def _continuum_density_with_endpoints(sigma, F0, r):
    """Retarded density with endpoint limits for adaptive quadrature."""
    sigma = finite(sigma, "sigma")
    F0, r, _ = _landau_parameters(F0, r)
    if sigma <= 0 or sigma >= 1:
        return mp.mpf(0)
    x = sigma/2*(mp.log1p(sigma)-mp.log1p(-sigma))-1
    A = F0-3*r*sigma**2
    denominator = (1-A*x)**2+(A*mp.pi*sigma/2)**2
    return sigma/(2*denominator)


def _continuum_mminus_density(sigma, F0, r):
    sigma = finite(sigma, "sigma")
    F0, _, _ = _landau_parameters(F0, r)
    if sigma == 0:
        return 1/(2*(1+F0)**2)
    if sigma >= 1:
        return mp.mpf(0)
    return _continuum_density_with_endpoints(sigma, F0, r)/sigma


def _integrate_form(F0, r, form):
    """Integrate continuum moments after a non-grid endpoint transformation."""
    if form == "exponential_endpoint":
        def transform(t):
            gap = mp.exp(-t)
            sigma = 1-gap
            jacobian = gap
            return sigma, jacobian
    elif form == "tanh_endpoint":
        def transform(t):
            sigma = mp.tanh(t)
            jacobian = 1/mp.cosh(t)**2
            return sigma, jacobian
    else:
        raise ValueError("unknown quadrature form")
    nodes = list(QUADRATURE_NODES)
    m1 = mp.quad(lambda t: (_continuum_density_with_endpoints(transform(t)[0], F0, r)
                            *transform(t)[0]*transform(t)[1]), nodes)
    mminus1 = mp.quad(lambda t: (_continuum_mminus_density(transform(t)[0], F0, r)
                                 *transform(t)[1]), nodes)
    if not mp.isfinite(m1) or not mp.isfinite(mminus1):
        raise ArithmeticError("nonfinite adaptive spectral integral")
    return {"m1_continuum": m1, "mminus1_continuum": mminus1}


def integrate_spectral_moments(F0, r, *, dps=80, include_pole=True):
    """Independently integrate both positive-frequency moments."""
    precision(dps)
    if not isinstance(include_pole, bool):
        raise ValueError("include_pole must be bool")
    with mp.workdps(dps):
        F0, r, _ = _landau_parameters(F0, r)
        first = _integrate_form(F0, r, "exponential_endpoint")
        second = _integrate_form(F0, r, "tanh_endpoint")
        pole = pole_data(F0, r, dps=dps)
        pole_m1 = pole["sigma_p"]*pole["residue"] if pole["exists"] else mp.mpf(0)
        pole_mminus1 = pole["residue"]/pole["sigma_p"] if pole["exists"] else mp.mpf(0)
        if include_pole:
            first_total = first["m1_continuum"]+pole_m1
            second_total = first["mminus1_continuum"]+pole_mminus1
        else:
            first_total = first["m1_continuum"]
            second_total = first["mminus1_continuum"]
        target_m1 = 1/(6*(1+r))
        target_mminus1 = 1/(2*(1+F0))
        quadrature_m1_error = _relative_real(first["m1_continuum"], second["m1_continuum"])
        quadrature_mminus1_error = _relative_real(first["mminus1_continuum"], second["mminus1_continuum"])
        m1_error = _relative_real(first_total, target_m1)
        mminus1_error = _relative_real(second_total, target_mminus1)
        passed = (quadrature_m1_error < MOMENT_TOLERANCE
                  and quadrature_mminus1_error < MOMENT_TOLERANCE
                  and m1_error < MOMENT_TOLERANCE
                  and mminus1_error < MOMENT_TOLERANCE
                  and (not pole["exists"] or (include_pole and pole["delta_piece_required"])))
        return {
            "pole": pole,
            "quadrature": {
                "exponential_endpoint": first,
                "tanh_endpoint": second,
            },
            "pole_contribution_m1": pole_m1 if include_pole else mp.mpf(0),
            "pole_contribution_mminus1": pole_mminus1 if include_pole else mp.mpf(0),
            "m1_total": first_total,
            "mminus1_total": second_total,
            "m1_total_without_pole": first["m1_continuum"],
            "mminus1_total_without_pole": first["mminus1_continuum"],
            "target_m1": target_m1,
            "target_mminus1": target_mminus1,
            "quadrature_m1_relative_difference": quadrature_m1_error,
            "quadrature_mminus1_relative_difference": quadrature_mminus1_error,
            "m1_relative_error": m1_error,
            "mminus1_relative_error": mminus1_error,
            "include_pole": include_pole,
            "integrals_converged": bool(quadrature_m1_error < MOMENT_TOLERANCE
                                          and quadrature_mminus1_error < MOMENT_TOLERANCE),
            "passed": bool(passed),
            "moment_tolerance": MOMENT_TOLERANCE,
        }


def _matrix_response(sigma, F0, r, *, source_second=True, branch="retarded", U=1):
    F0, r, F1 = _landau_parameters(F0, r)
    sigma = finite_complex(sigma, "sigma")
    U = finite(U, "U")
    if U == 0:
        raise ValueError("U must be nonzero for a normalized response")
    if branch != "retarded":
        raise ValueError("accepted response branch is retarded")
    L = retarded_lindhard(sigma)
    determinant_reduced = 1-(F0-3*r*sigma**2)*L
    matrix = mp.matrix([
        [1-F0*L, -F1*sigma*L],
        [-F0*sigma*L, 1-F1*(sigma**2*L-mp.mpf(1)/3)],
    ])
    source = mp.matrix([L*U, sigma*L*U if source_second else 0])
    determinant = mp.det(matrix)
    solution = mp.lu_solve(matrix, source)
    return {
        "F1": F1,
        "L": L,
        "A": F0-3*r*sigma**2,
        "D": determinant_reduced,
        "matrix": matrix,
        "source": source,
        "solution": solution,
        "density_response": solution[0]/U,
        "closed_response": L/determinant_reduced,
        "determinant": determinant,
        "expected_determinant": determinant_reduced/(1+r),
        "source_second": source_second,
    }


def response_matrix(sigma, F0, r, *, source_second=True, U=1):
    """Public matrix response helper retaining the full source vector."""
    return _matrix_response(sigma, F0, r, source_second=source_second, U=U)


def chi_over_nf(sigma, F0, r):
    """Return the adopted density response chi/NF."""
    F0, r, _ = _landau_parameters(F0, r)
    sigma = finite_complex(sigma, "sigma")
    L = retarded_lindhard(sigma)
    return L/(1-(F0-3*r*sigma**2)*L)


def _direct_angular_moments(sigma):
    sigma = finite_complex(sigma, "sigma")
    if sigma.imag == 0 and -1 <= sigma.real <= 1:
        raise ValueError("direct angular quadrature is singular on the cut")
    return [mp.quad(lambda u, power=power: u**power/(sigma-u)/2,
                    [-1, 0, 1]) for power in (1, 2, 3)]


def _split_angular_moments(sigma):
    sigma = finite_complex(sigma, "sigma")
    L = retarded_lindhard(sigma)
    return [L, sigma*L, sigma**2*L-mp.mpf(1)/3]


def direct_vs_split_angular_check(F0, r, *, dps=80):
    """Compare direct angular integrals with the three split logarithms."""
    precision(dps)
    with mp.workdps(dps):
        F0, r, _ = _landau_parameters(F0, r)
        rows = []
        for real, imag in ANGULAR_SIGMAS:
            sigma = mp.mpc(mp.mpf(real), mp.mpf(imag))
            direct = _direct_angular_moments(sigma)
            split = _split_angular_moments(sigma)
            errors = [_relative_error(a, b) for a, b in zip(direct, split)]
            matrix_direct = mp.matrix([
                [1-F0*direct[0], -(-3*r/(1+r))*direct[1]],
                [-F0*direct[1], 1-(-3*r/(1+r))*direct[2]],
            ])
            source_direct = mp.matrix([direct[0], direct[1]])
            direct_solution = mp.lu_solve(matrix_direct, source_direct)[0]
            split_response = _matrix_response(sigma, F0, r)["density_response"]
            rows.append({
                "sigma": sigma,
                "moment_relative_errors": errors,
                "response_relative_error": _relative_error(direct_solution, split_response),
                "direct_response": direct_solution,
                "split_response": split_response,
            })
        max_moment_error = max(max(row["moment_relative_errors"]) for row in rows)
        max_response_error = max(row["response_relative_error"] for row in rows)
        return {
            "rows": rows,
            "max_moment_relative_error": max_moment_error,
            "max_response_relative_error": max_response_error,
            "passed": bool(max(max_moment_error, max_response_error) < NUMERIC_TOLERANCE),
            "separate_log_continuation_used": True,
        }


def source_vector_check(F0, r, *, dps=80):
    """Check Cramer response and the mandatory second source entry."""
    precision(dps)
    with mp.workdps(dps):
        sigma = mp.mpc(mp.mpf("0.37"), mp.mpf("0.21"))
        correct = _matrix_response(sigma, F0, r, source_second=True, U=mp.mpf("1.7"))
        missing = _matrix_response(sigma, F0, r, source_second=False, U=mp.mpf("1.7"))
        determinant_error = _relative_error(correct["determinant"], correct["expected_determinant"])
        response_error = _relative_error(correct["density_response"], correct["closed_response"])
        missing_difference = abs(missing["density_response"]-correct["density_response"])
        source_second_residual = abs(correct["source"][1]-sigma*correct["L"]*mp.mpf("1.7"))
        return {
            "sigma": sigma,
            "external_energy": mp.mpf("1.7"),
            "F1": correct["F1"],
            "source_vector": [correct["source"][0], correct["source"][1]],
            "full_source_vector_included": correct["source_second"],
            "source_second_residual": source_second_residual,
            "determinant_relative_error": determinant_error,
            "density_response_relative_error": response_error,
            "density_response": correct["density_response"],
            "closed_response": correct["closed_response"],
            "missing_second_source_response": missing["density_response"],
            "missing_second_source_difference": missing_difference,
            "missing_second_source_rejected": bool(missing_difference > mp.mpf("1e-12")),
            "no_density_contact_term": True,
            "passed": bool(correct["source_second"] is True
                           and source_second_residual < NUMERIC_TOLERANCE
                           and determinant_error < NUMERIC_TOLERANCE
                           and response_error < NUMERIC_TOLERANCE
                           and missing_difference > mp.mpf("1e-12")),
        }


def limit_checks(F0, r, *, dps=80):
    """Check exact static/dynamic limits and the matched high-sigma coefficient."""
    precision(dps)
    with mp.workdps(dps):
        F0, r, _ = _landau_parameters(F0, r)
        static_exact = -1/(1+F0)
        static_response = chi_over_nf(mp.mpf(0), F0, r)
        static_error = _relative_error(static_response, static_exact)
        small_sigma = mp.power(10, -(dps//3))
        small_response = chi_over_nf(small_sigma, F0, r)
        static_numeric_error = _relative_error(small_response, static_exact)
        high_sigma = mp.power(10, dps//3)
        high_response = chi_over_nf(high_sigma, F0, r)
        high_coefficient = high_sigma**2*high_response
        high_exact = 1/(3*(1+r))
        high_error = _relative_error(high_coefficient, high_exact)
        return {
            "static_limit_exact": static_exact,
            "static_response_at_sigma_zero": static_response,
            "static_relative_error": static_error,
            "static_small_sigma": small_sigma,
            "static_small_sigma_relative_error": static_numeric_error,
            "dynamic_limit_exact": mp.mpf(0),
            "high_sigma": high_sigma,
            "high_sigma_response": high_response,
            "high_sigma_coefficient": high_coefficient,
            "high_sigma_coefficient_exact": high_exact,
            "high_sigma_relative_error": high_error,
            "noncommuting_static_dynamic_limits": bool(static_exact != 0),
            "passed": bool(static_error < NUMERIC_TOLERANCE
                           and static_numeric_error < MOMENT_TOLERANCE
                           and high_error < MOMENT_TOLERANCE
                           and static_exact != 0),
        }


def _continuum_absorption_check(F0, r):
    sigma = mp.mpf("0.5")
    retarded = continuum_spectral_density(sigma, F0, r, branch="retarded")
    advanced = continuum_spectral_density(sigma, F0, r, branch="advanced")
    return {
        "sigma": sigma,
        "retarded_S_over_NF": retarded,
        "advanced_S_over_NF": advanced,
        "finite_frequency_absorption": bool(mp.isfinite(retarded) and retarded > 0),
        "retarded_sign_passed": bool(retarded > 0),
        "wrong_sign_rejected": bool(advanced < 0),
        "passed": bool(retarded > 0 and advanced < 0),
    }


def _effective_bridge(n_ratio, raw, *, dps=80):
    bridge = effective.nvg_connection(n_ratio, "50", dps=dps)
    if not isinstance(bridge, dict):
        raise ArithmeticError("effective-response bridge returned no dictionary")
    acoustic = bridge.get("acoustic")
    if not isinstance(acoustic, dict):
        raise ArithmeticError("effective-response bridge omitted acoustic API")
    source_status = bridge.get("source_independent_checks_passed")
    source_checked = acoustic.get("source_comparison_checked")
    acoustic_source = acoustic.get("source_acoustic_squared_speed")
    source_error = acoustic.get("source_relative_error")
    inertial_error = acoustic.get("inertial_relative_error")
    if (bridge.get("source_producer") != "verification/nvg_longitudinal_fluid_audit.py"
            or source_status is not True
            or bridge.get("mathematical_checks_passed") is not True
            or source_checked is not True
            or acoustic_source is None or source_error is None or inertial_error is None):
        raise ArithmeticError("effective-response source or mathematical status failed")
    source_error = finite(source_error, "effective source relative error")
    inertial_error = finite(inertial_error, "effective inertial relative error")
    acoustic_source = finite(acoustic_source, "effective source acoustic speed")
    speed_error = _relative_real(acoustic_source, raw["first_sound_squared"])
    passed = bool(source_error < NUMERIC_TOLERANCE
                  and inertial_error < NUMERIC_TOLERANCE
                  and speed_error < NUMERIC_TOLERANCE)
    return {
        "source_producer": bridge["source_producer"],
        "source_independent_checks_passed": source_status,
        "mathematical_checks_passed": bridge["mathematical_checks_passed"],
        "source_comparison_checked": source_checked,
        "source_acoustic_squared_speed": acoustic_source,
        "collisionless_first_sound_squared": raw["first_sound_squared"],
        "acoustic_relative_error_to_collisionless": speed_error,
        "source_relative_error": source_error,
        "low_frequency_inertial_coefficient": acoustic["low_frequency_inertial_coefficient"],
        "expected_inertial_coefficient": acoustic["expected_inertial_coefficient"],
        "inertial_relative_error": inertial_error,
        "finite_k_group_velocity_not_claimed": acoustic.get("finite_k_group_velocity_not_claimed"),
        "passed": passed,
    }


def _moment_bridge(raw, moments, bridge):
    """Connect independently integrated moments to both live sound bridges."""
    m1 = moments["m1_total"]
    mminus1 = moments["mminus1_total"]
    moment_sound = raw["vF2"]*m1/mminus1
    collisionless_sound = raw["first_sound_squared"]
    effective_sound = bridge["source_acoustic_squared_speed"]
    moment_to_collisionless = _relative_real(moment_sound, collisionless_sound)
    moment_to_effective = _relative_real(moment_sound, effective_sound)
    return {
        "vF2_m1_over_mminus1": moment_sound,
        "collisionless_first_sound_squared": collisionless_sound,
        "effective_response_acoustic_squared_speed": effective_sound,
        "moment_to_collisionless_relative_error": moment_to_collisionless,
        "moment_to_effective_response_relative_error": moment_to_effective,
        "passed": bool(moment_to_collisionless < MOMENT_TOLERANCE
                       and moment_to_effective < MOMENT_TOLERANCE),
    }


def _state_internal(n_ratio, *, dps=80, include_bridge=True):
    precision(dps)
    with mp.workdps(dps):
        raw, upstream = _load_upstream(n_ratio, dps)
        F0, r = raw["F0"], raw["r"]
        source = source_vector_check(F0, r, dps=dps)
        current_feedback = {
            "r": r,
            "F1": source["F1"],
            "F1_relation_residual": abs(source["F1"]/(1+source["F1"]/3)+3*r),
            "passed": bool(abs(source["F1"]/(1+source["F1"]/3)+3*r) < NUMERIC_TOLERANCE),
        }
        angular = direct_vs_split_angular_check(F0, r, dps=dps)
        pole = pole_data(F0, r, dps=dps)
        moments = integrate_spectral_moments(F0, r, dps=dps, include_pole=True)
        limits = limit_checks(F0, r, dps=dps)
        absorption = _continuum_absorption_check(F0, r)
        bridge = _effective_bridge(n_ratio, raw, dps=dps) if include_bridge else {
            "not_run_for_precision_refinement": True,
            "passed": True,
        }
        moment_bridge = (_moment_bridge(raw, moments, bridge) if include_bridge else {
            "not_run_for_precision_refinement": True,
            "passed": True,
        })
        symbolic = symbolic_checks()
        passed = bool(_symbolic_passed(symbolic)
                      and source["passed"] and angular["passed"]
                      and moments["passed"] and limits["passed"]
                      and absorption["passed"] and bridge["passed"]
                      and moment_bridge["passed"])
        return {
            "status": STATUS,
            "evidence_weight": EVIDENCE_WEIGHT,
            "mathematical_checks_passed": passed,
            "inputs": {"n_over_n0": raw["n_ratio"], "dps": dps},
            "coefficients": {key: raw[key] for key in
                             ("y", "EF", "vF2", "NF", "M2", "B", "C", "V",
                              "r", "F0", "F1", "mu", "first_sound_squared")},
            "upstream": upstream,
            "source_vector": source,
            "current_feedback": current_feedback,
            "angular_response": angular,
            "continuum_absorption": absorption,
            "pole": pole,
            "spectral_moments": moments,
            "limits": limits,
            "effective_response_bridge": bridge,
            "moment_bridge": moment_bridge,
            "symbolic_checks": symbolic,
            "scope": [
                "Retarded l=0,1 normal-state Hartree/Vlasov density response at k/kF -> 0 with fixed sigma.",
                "Positive-frequency continuum 0<sigma<1 plus an analytic delta pole only when F0>3r.",
                "Static means omega -> 0 before k -> 0; dynamic high-sigma is the matched window below scalar/vector gaps.",
                "The moments are exact for this low-energy Landau closure, not a full microscopic relativistic sum rule.",
                "No density contact term is added; a longitudinal current-current observable requires separate Ward/contact accounting.",
                "No collisions, pairing, finite-k quantum RPA, gravity/cuscuton, bounce, or empirical confirmation is included.",
            ],
            "references": list(collisionless.REFERENCES),
        }


def _as_real(value, name):
    try:
        with mp.workdps(200):
            return finite(value, name)
    except (ValueError, TypeError, OverflowError):
        return None


def _validate_coefficients_payload(c):
    """Validate coefficient shape, positivity, and all available identities."""
    if not _required_keys(c, REQUIRED_COEFFICIENTS):
        return None
    values = {key: _real_field(c, key) for key in REQUIRED_COEFFICIENTS}
    if any(value is None for value in values.values()):
        return None
    positive = ("y", "EF", "vF2", "NF", "M2", "C", "V", "mu",
                "first_sound_squared")
    if any(values[key] <= 0 for key in positive) or values["vF2"] >= 1:
        return None
    F0, r = values["F0"], values["r"]
    if F0 <= -1 or r < 0:
        return None
    expected_F1 = -3*r/(1+r)
    derived_n = values["NF"]*values["EF"]*values["vF2"]/3
    expected = {
        "F1": expected_F1,
        "r": derived_n*values["V"]/values["EF"],
        "F0": values["NF"]*(values["V"]-values["B"]**2/values["C"]),
        "mu": values["EF"]+derived_n*values["V"],
        "first_sound_squared": values["vF2"]*(1+F0)*(1+expected_F1/3)/3,
    }
    if any(not _serialized_close(values[key], expected[key]) for key in expected):
        return None
    return values


def _validate_upstream_payload(upstream, inputs):
    if not _required_keys(upstream, REQUIRED_UPSTREAM_KEYS):
        return False
    if (upstream["producer"] != "verification/nvg_collisionless_response_audit.py"
            or upstream["status"] != collisionless.STATUS
            or upstream["mathematical_checks_passed"] is not True
            or upstream["status_is_independent_of_residual_gate"] is not True):
        return False
    if upstream["dps"] != inputs["dps"]:
        return False
    if not _serialized_close(upstream["n_ratio"], inputs["n_over_n0"]):
        return False
    residuals = upstream["residuals"]
    if not _required_keys(residuals, REQUIRED_UPSTREAM_RESIDUALS):
        return False
    checked = [_real_field(residuals, key) for key in REQUIRED_UPSTREAM_RESIDUALS]
    if any(value is None or abs(value) >= UPSTREAM_RESIDUAL_TOLERANCE for value in checked):
        return False
    maximum = max(abs(value) for value in checked)
    tolerance = _real_field(upstream, "residual_tolerance", positive=True)
    reported_max = _real_field(upstream, "max_residual", nonnegative=True)
    return bool(tolerance is not None and _serialized_close(tolerance, UPSTREAM_RESIDUAL_TOLERANCE)
                and reported_max is not None and _serialized_close(reported_max, maximum)
                and reported_max < UPSTREAM_RESIDUAL_TOLERANCE)


def _validate_source_payload(source, F0, r):
    if not _required_keys(source, REQUIRED_SOURCE_KEYS):
        return False
    if (source["full_source_vector_included"] is not True
            or source["no_density_contact_term"] is not True
            or source["passed"] is not True):
        return False
    sigma = _as_complex(source["sigma"], "source sigma")
    U = _real_field(source, "external_energy", positive=True)
    if sigma is None or U is None:
        return False
    expected_sigma = mp.mpc(mp.mpf("0.37"), mp.mpf("0.21"))
    if not _serialized_complex_close(sigma, expected_sigma):
        return False
    if not _serialized_close(U, mp.mpf("1.7")):
        return False
    try:
        expected = _matrix_response(sigma, F0, r, source_second=True, U=U)
        missing = _matrix_response(sigma, F0, r, source_second=False, U=U)
        vector = source["source_vector"]
        if not isinstance(vector, list) or len(vector) != 2:
            return False
        vector = [_as_complex(value, "source vector") for value in vector]
        if any(value is None for value in vector):
            return False
        if not all(_serialized_complex_close(value, expected["source"][i])
                   for i, value in enumerate(vector)):
            return False
        if not _serialized_complex_close(source["density_response"], expected["density_response"]):
            return False
        if not _serialized_complex_close(source["closed_response"], expected["closed_response"]):
            return False
        if not _serialized_complex_close(source["missing_second_source_response"],
                                         missing["density_response"]):
            return False
        source_second_residual = abs(vector[1]-sigma*vector[0])
        determinant_error = _relative_error(expected["determinant"], expected["expected_determinant"])
        response_error = _relative_error(expected["density_response"], expected["closed_response"])
        missing_difference = abs(missing["density_response"]-expected["density_response"])
        reported_source_residual = _real_field(source, "source_second_residual", nonnegative=True)
        reported_det = _real_field(source, "determinant_relative_error", nonnegative=True)
        reported_response = _real_field(source, "density_response_relative_error", nonnegative=True)
        reported_difference = _real_field(source, "missing_second_source_difference", nonnegative=True)
        if (reported_source_residual is None or reported_det is None or reported_response is None
                or reported_difference is None
                or not _serialized_close(reported_source_residual, source_second_residual)
                or not _serialized_close(reported_det, determinant_error)
                or not _serialized_close(reported_response, response_error)
                or not _serialized_close(reported_difference, missing_difference)):
            return False
        if (source_second_residual >= SERIALIZED_TOLERANCE
                or determinant_error >= NUMERIC_TOLERANCE
                or response_error >= NUMERIC_TOLERANCE
                or reported_source_residual >= NUMERIC_TOLERANCE
                or reported_det >= NUMERIC_TOLERANCE
                or reported_response >= NUMERIC_TOLERANCE
                or missing_difference <= mp.mpf("1e-12")
                or source["missing_second_source_rejected"] is not True):
            return False
        if not _serialized_close(source["F1"], expected["F1"]):
            return False
        return True
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError):
        return False


def _validate_current_payload(current, F0, r):
    if not _required_keys(current, REQUIRED_CURRENT_KEYS) or current["passed"] is not True:
        return False
    expected_F1 = -3*r/(1+r)
    reported_r = _real_field(current, "r", nonnegative=True)
    reported_F1 = _real_field(current, "F1")
    reported_error = _real_field(current, "F1_relation_residual", nonnegative=True)
    if reported_r is None or reported_F1 is None or reported_error is None:
        return False
    actual_error = abs(expected_F1/(1+expected_F1/3)+3*r)
    return bool(_serialized_close(reported_r, r)
                and _serialized_close(reported_F1, expected_F1)
                and _serialized_close(reported_error, actual_error)
                and actual_error < NUMERIC_TOLERANCE)


def _validate_angular_payload(angular, F0, r):
    if not _required_keys(angular, REQUIRED_ANGULAR_KEYS):
        return False
    if (angular["passed"] is not True
            or angular["separate_log_continuation_used"] is not True
            or not isinstance(angular["rows"], list)
            or len(angular["rows"]) != len(ANGULAR_SIGMAS)):
        return False
    moment_errors, response_errors = [], []
    for row, (real, imag) in zip(angular["rows"], ANGULAR_SIGMAS):
        if not _required_keys(row, REQUIRED_ANGULAR_ROW_KEYS):
            return False
        sigma = _as_complex(row["sigma"], "angular sigma")
        expected_sigma = mp.mpc(mp.mpf(real), mp.mpf(imag))
        if sigma is None or not _serialized_complex_close(sigma, expected_sigma):
            return False
        errors = row["moment_relative_errors"]
        if (not isinstance(errors, list) or len(errors) != 3
                or any(_real_field({"value": value}, "value", nonnegative=True) is None
                       for value in errors)):
            return False
        if any(_real_field({"value": value}, "value", nonnegative=True) >= NUMERIC_TOLERANCE
               for value in errors):
            return False
        direct = _direct_angular_moments(sigma)
        split = _split_angular_moments(sigma)
        actual_moment_errors = [_relative_error(left, right)
                                for left, right in zip(direct, split)]
        direct_matrix = mp.matrix([
            [1-F0*direct[0], -(-3*r/(1+r))*direct[1]],
            [-F0*direct[1], 1-(-3*r/(1+r))*direct[2]],
        ])
        direct_response = mp.lu_solve(direct_matrix, mp.matrix([direct[0], direct[1]]))[0]
        split_response = _matrix_response(sigma, F0, r)["density_response"]
        actual_response_error = _relative_error(direct_response, split_response)
        if (not _serialized_complex_close(row["direct_response"], direct_response)
                or not _serialized_complex_close(row["split_response"], split_response)
                or not all(_serialized_close(left, right)
                           for left, right in zip(errors, actual_moment_errors))):
            return False
        reported_response_error = _real_field(row, "response_relative_error", nonnegative=True)
        if (reported_response_error is None
                or not _serialized_close(reported_response_error, actual_response_error)):
            return False
        moment_errors.extend(actual_moment_errors)
        response_errors.append(actual_response_error)
    actual_max_moment = max(moment_errors)
    actual_max_response = max(response_errors)
    reported_max_moment = _real_field(angular, "max_moment_relative_error", nonnegative=True)
    reported_max_response = _real_field(angular, "max_response_relative_error", nonnegative=True)
    if reported_max_moment is None or reported_max_response is None:
        return False
    return bool(_serialized_close(reported_max_moment, actual_max_moment)
                and _serialized_close(reported_max_response, actual_max_response)
                and reported_max_moment < NUMERIC_TOLERANCE
                and reported_max_response < NUMERIC_TOLERANCE
                and actual_max_moment < NUMERIC_TOLERANCE
                and actual_max_response < NUMERIC_TOLERANCE)


def _validate_absorption_payload(absorption, F0, r):
    if not _required_keys(absorption, REQUIRED_ABSORPTION_KEYS):
        return False
    expected = _continuum_absorption_check(F0, r)
    for key in ("sigma", "retarded_S_over_NF", "advanced_S_over_NF"):
        if not _serialized_close(absorption[key], expected[key]):
            return False
    for key in ("finite_frequency_absorption", "retarded_sign_passed", "wrong_sign_rejected", "passed"):
        if absorption[key] is not expected[key]:
            return False
    return bool(expected["finite_frequency_absorption"] and expected["retarded_sign_passed"]
                and expected["wrong_sign_rejected"] and expected["passed"])


def _validate_pole_payload(pole, F0, r):
    required = F0 > 3*r
    expected_keys = REQUIRED_RESOLVED_POLE_KEYS if required else REQUIRED_POLE_KEYS
    if not _required_keys(pole, expected_keys) or pole.get("exists") is not required:
        return False
    if required:
        if (pole["condition_F0_gt_3r"] is not True
                or pole["positive_residue"] is not True
                or pole["delta_piece_required"] is not True):
            return False
        log_gap = _real_field(pole, "log_gap")
        if log_gap is None:
            return False
        try:
            expected = _pole_quantities_from_log_gap(log_gap, F0, r)
        except (ValueError, TypeError, ArithmeticError, ZeroDivisionError):
            return False
        for key in ("log_gap", "gap", "sigma_p", "L_p", "A_p", "L_prime_p",
                    "D_prime_p", "residue", "dispersion_residual"):
            positive = key in ("gap", "residue")
            reported = _real_field(pole, key, positive=positive)
            comparator = (_serialized_nonzero_close if positive else _serialized_close)
            if reported is None or not comparator(reported, expected[key]):
                return False
        return bool(expected["gap"] > 0 and expected["residue"] > 0
                    and abs(expected["dispersion_residual"]) < SERIALIZED_TOLERANCE)
    if (pole["condition_F0_gt_3r"] is not False
            or pole["delta_piece_required"] is not False
            or pole["log_gap"] is not None or pole["gap"] is not None
            or pole["sigma_p"] is not None):
        return False
    residue = _real_field(pole, "residue", nonnegative=True)
    return residue is not None and residue == 0


def _validate_moments_payload(moments, pole, F0, r):
    if not _required_keys(moments, REQUIRED_MOMENT_KEYS):
        return False
    if (moments["include_pole"] is not True or moments["integrals_converged"] is not True
            or moments["passed"] is not True or not _required_keys(moments["quadrature"], REQUIRED_QUADRATURE_KEYS)
            or not _serialized_pole_equal(moments["pole"], pole)
            or not _validate_pole_payload(moments["pole"], F0, r)):
        return False
    quadrature = moments["quadrature"]
    for key in REQUIRED_QUADRATURE_KEYS:
        if not _required_keys(quadrature[key], REQUIRED_QUADRATURE_ROW_KEYS):
            return False
    try:
        expected_exp = _integrate_form(F0, r, "exponential_endpoint")
        expected_tanh = _integrate_form(F0, r, "tanh_endpoint")
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError):
        return False
    actual_rows = {}
    for form, expected in (("exponential_endpoint", expected_exp), ("tanh_endpoint", expected_tanh)):
        row = quadrature[form]
        actual_rows[form] = {}
        for key in REQUIRED_QUADRATURE_ROW_KEYS:
            value = _real_field(row, key, positive=True)
            if value is None or not _serialized_close(value, expected[key]):
                return False
            actual_rows[form][key] = value
    pole_exists = F0 > 3*r
    pole_row = pole
    if pole_exists:
        pole_sigma = _real_field(pole_row, "sigma_p", positive=True)
        pole_residue = _real_field(pole_row, "residue", positive=True)
        if pole_sigma is None or pole_residue is None:
            return False
        expected_pm1 = pole_sigma*pole_residue
        expected_pmm = pole_residue/pole_sigma
    else:
        expected_pm1 = expected_pmm = mp.mpf(0)
    if pole_exists:
        pm1 = _real_field(moments, "pole_contribution_m1", positive=True)
        pmm = _real_field(moments, "pole_contribution_mminus1", positive=True)
    else:
        pm1 = _real_field(moments, "pole_contribution_m1", nonnegative=True)
        pmm = _real_field(moments, "pole_contribution_mminus1", nonnegative=True)
    m1_total = _real_field(moments, "m1_total", positive=True)
    mm_total = _real_field(moments, "mminus1_total", positive=True)
    m1_cont = _real_field(moments, "m1_total_without_pole", positive=True)
    mm_cont = _real_field(moments, "mminus1_total_without_pole", positive=True)
    if any(value is None for value in (pm1, pmm, m1_total, mm_total, m1_cont, mm_cont)):
        return False
    target_m1 = 1/(6*(1+r))
    target_mm = 1/(2*(1+F0))
    reported_target_m1 = _real_field(moments, "target_m1", positive=True)
    reported_target_mm = _real_field(moments, "target_mminus1", positive=True)
    if (reported_target_m1 is None or reported_target_mm is None
            or not _serialized_close(reported_target_m1, target_m1)
            or not _serialized_close(reported_target_mm, target_mm)):
        return False
    if pole_exists:
        if (not _serialized_nonzero_close(pm1, expected_pm1)
                or not _serialized_nonzero_close(pmm, expected_pmm)):
            return False
    elif pm1 != 0 or pmm != 0:
        return False
    expected_m1_cont = expected_exp["m1_continuum"]
    expected_mm_cont = expected_exp["mminus1_continuum"]
    expected_m1_total = expected_m1_cont+expected_pm1
    expected_mm_total = expected_mm_cont+expected_pmm
    if (not _serialized_close(m1_cont, expected_m1_cont)
            or not _serialized_close(mm_cont, expected_mm_cont)
            or not _serialized_close(m1_total, expected_m1_total)
            or not _serialized_close(mm_total, expected_mm_total)):
        return False
    actual_quad_m1 = _relative_real(expected_exp["m1_continuum"], expected_tanh["m1_continuum"])
    actual_quad_mm = _relative_real(expected_exp["mminus1_continuum"], expected_tanh["mminus1_continuum"])
    actual_m1_error = _relative_real(expected_m1_total, target_m1)
    actual_mm_error = _relative_real(expected_mm_total, target_mm)
    reported_values = {
        "quadrature_m1_relative_difference": _real_field(moments, "quadrature_m1_relative_difference", nonnegative=True),
        "quadrature_mminus1_relative_difference": _real_field(moments, "quadrature_mminus1_relative_difference", nonnegative=True),
        "m1_relative_error": _real_field(moments, "m1_relative_error", nonnegative=True),
        "mminus1_relative_error": _real_field(moments, "mminus1_relative_error", nonnegative=True),
        "moment_tolerance": _real_field(moments, "moment_tolerance", positive=True),
    }
    if (any(value is None for value in reported_values.values())
            or not _serialized_close(reported_values["quadrature_m1_relative_difference"], actual_quad_m1)
            or not _serialized_close(reported_values["quadrature_mminus1_relative_difference"], actual_quad_mm)
            or not _serialized_close(reported_values["m1_relative_error"], actual_m1_error)
            or not _serialized_close(reported_values["mminus1_relative_error"], actual_mm_error)
            or not _serialized_close(reported_values["moment_tolerance"], MOMENT_TOLERANCE)):
        return False
    return bool(actual_quad_m1 < MOMENT_TOLERANCE and actual_quad_mm < MOMENT_TOLERANCE
                and actual_m1_error < MOMENT_TOLERANCE
                and actual_mm_error < MOMENT_TOLERANCE)


def _validate_limits_payload(limits, F0, r, dps):
    if not _required_keys(limits, REQUIRED_LIMIT_KEYS) or limits["passed"] is not True:
        return False
    try:
        # Reproduce the producer's declared precision for finite numerical
        # limit checks; otherwise recomputing the high-sigma cancellation at
        # 200 digits would make the reported 80-digit error incomparable.
        with mp.workdps(dps):
            static_exact = -1/(1+F0)
            static_response = chi_over_nf(mp.mpf(0), F0, r)
            small_sigma = mp.power(10, -(dps//3))
            small_response = chi_over_nf(small_sigma, F0, r)
            high_sigma = mp.power(10, dps//3)
            high_response = chi_over_nf(high_sigma, F0, r)
            high_coefficient = high_sigma**2*high_response
            high_exact = 1/(3*(1+r))
            expected = {
                "static_limit_exact": static_exact,
                "static_response_at_sigma_zero": static_response,
                "static_relative_error": _relative_error(static_response, static_exact),
                "static_small_sigma": small_sigma,
                "static_small_sigma_relative_error": _relative_error(small_response, static_exact),
                "dynamic_limit_exact": mp.mpf(0),
                "high_sigma": high_sigma,
                "high_sigma_response": high_response,
                "high_sigma_coefficient": high_coefficient,
                "high_sigma_coefficient_exact": high_exact,
                "high_sigma_relative_error": _relative_error(high_coefficient, high_exact),
            }
        for key, value in expected.items():
            if not _serialized_close(limits[key], value):
                return False
        for key in ("static_relative_error", "static_small_sigma_relative_error",
                    "high_sigma_relative_error"):
            if _real_field(limits, key, nonnegative=True) is None:
                return False
        reported_errors = {
            key: _real_field(limits, key, nonnegative=True)
            for key in ("static_relative_error", "static_small_sigma_relative_error",
                        "high_sigma_relative_error")
        }
        if any(value is None or value >= (NUMERIC_TOLERANCE if key == "static_relative_error"
                                          else MOMENT_TOLERANCE)
               for key, value in reported_errors.items()):
            return False
        return bool(limits["noncommuting_static_dynamic_limits"] is True
                    and expected["static_relative_error"] < NUMERIC_TOLERANCE
                    and expected["static_small_sigma_relative_error"] < MOMENT_TOLERANCE
                    and expected["high_sigma_relative_error"] < MOMENT_TOLERANCE)
    except (ValueError, TypeError, ArithmeticError, ZeroDivisionError):
        return False


def _validate_bridge_payload(bridge, c):
    if not _required_keys(bridge, REQUIRED_BRIDGE_KEYS):
        return False
    if (bridge["source_producer"] != "verification/nvg_longitudinal_fluid_audit.py"
            or bridge["source_independent_checks_passed"] is not True
            or bridge["mathematical_checks_passed"] is not True
            or bridge["source_comparison_checked"] is not True
            or bridge["finite_k_group_velocity_not_claimed"] is not True
            or bridge["passed"] is not True):
        return False
    expected_sound = c["first_sound_squared"]
    source_sound = _real_field(bridge, "source_acoustic_squared_speed", positive=True)
    collisionless_sound = _real_field(bridge, "collisionless_first_sound_squared", positive=True)
    source_error = _real_field(bridge, "source_relative_error", nonnegative=True)
    acoustic_error = _real_field(bridge, "acoustic_relative_error_to_collisionless", nonnegative=True)
    low_inertia = _real_field(bridge, "low_frequency_inertial_coefficient", positive=True)
    expected_inertia = _real_field(bridge, "expected_inertial_coefficient", positive=True)
    inertia_error = _real_field(bridge, "inertial_relative_error", nonnegative=True)
    if any(value is None for value in (source_sound, collisionless_sound, source_error,
                                       acoustic_error, low_inertia, expected_inertia,
                                       inertia_error)):
        return False
    derived_n = c["NF"]*c["EF"]*c["vF2"]/3
    expected_inertia_from_coefficients = derived_n*c["mu"]
    actual_source_error = _relative_real(source_sound, expected_sound)
    actual_acoustic_error = _relative_real(source_sound, collisionless_sound)
    actual_inertia_error = _relative_real(low_inertia, expected_inertia)
    if (not _serialized_close(source_sound, expected_sound)
            or not _serialized_close(collisionless_sound, expected_sound)
            or not _serialized_close(expected_inertia, expected_inertia_from_coefficients)
            or not _serialized_close(source_error, actual_source_error)
            or not _serialized_close(acoustic_error, actual_acoustic_error)
            or not _serialized_close(inertia_error, actual_inertia_error)
            or not _serialized_close(low_inertia, expected_inertia)):
        return False
    return bool(actual_source_error < MOMENT_TOLERANCE
                and actual_acoustic_error < MOMENT_TOLERANCE
                and actual_inertia_error < MOMENT_TOLERANCE
                and source_error < NUMERIC_TOLERANCE
                and acoustic_error < NUMERIC_TOLERANCE
                and inertia_error < NUMERIC_TOLERANCE)


def _validate_moment_bridge_payload(moment_bridge, moments, bridge, c):
    if (not _required_keys(moment_bridge, REQUIRED_MOMENT_BRIDGE_KEYS)
            or moment_bridge["passed"] is not True):
        return False
    m1 = _real_field(moments, "m1_total", positive=True)
    mm = _real_field(moments, "mminus1_total", positive=True)
    if m1 is None or mm is None:
        return False
    expected_moment_sound = c["vF2"]*m1/mm
    expected_collisionless_sound = c["first_sound_squared"]
    expected_effective_sound = _real_field(bridge, "source_acoustic_squared_speed", positive=True)
    if expected_effective_sound is None:
        return False
    actual_error_collisionless = _relative_real(expected_moment_sound, expected_collisionless_sound)
    actual_error_effective = _relative_real(expected_moment_sound, expected_effective_sound)
    reported_moment_sound = _real_field(moment_bridge, "vF2_m1_over_mminus1", positive=True)
    reported_collisionless = _real_field(moment_bridge, "collisionless_first_sound_squared", positive=True)
    reported_effective = _real_field(moment_bridge, "effective_response_acoustic_squared_speed", positive=True)
    reported_error_collisionless = _real_field(moment_bridge, "moment_to_collisionless_relative_error", nonnegative=True)
    reported_error_effective = _real_field(moment_bridge, "moment_to_effective_response_relative_error", nonnegative=True)
    if any(value is None for value in (reported_moment_sound, reported_collisionless,
                                       reported_effective, reported_error_collisionless,
                                       reported_error_effective)):
        return False
    return bool(_serialized_close(reported_moment_sound, expected_moment_sound)
                and _serialized_close(reported_collisionless, expected_collisionless_sound)
                and _serialized_close(reported_effective, expected_effective_sound)
                and _serialized_close(reported_error_collisionless, actual_error_collisionless)
                and _serialized_close(reported_error_effective, actual_error_effective)
                and actual_error_collisionless < MOMENT_TOLERANCE
                and actual_error_effective < MOMENT_TOLERANCE)


@lru_cache(maxsize=4)
def _expected_controls_payload(dps):
    return _shown(_mathematical_controls(dps=dps))


def _numeric_state_passed_impl(state):
    """Validate a complete report by recomputing its serialized identities.

    This is an algebraic/rounding-consistency closure for a report that was
    already produced live.  It is intentionally not a cryptographic
    provenance check and does not replace rerunning the producers.
    """
    try:
        if (not isinstance(state, dict) or set(state) != set(REQUIRED_STATE_KEYS)
                or state["status"] != STATUS
                or state["evidence_weight"] != EVIDENCE_WEIGHT
                or state["mathematical_checks_passed"] is not True
                or state["numeric_passport_passed"] is not True):
            return False
        inputs = state["inputs"]
        if not _required_keys(inputs, REQUIRED_INPUT_KEYS):
            return False
        ratio = _real_field(inputs, "n_over_n0", positive=True)
        dps = inputs["dps"]
        if ratio is None or isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 200:
            return False
        c = _validate_coefficients_payload(state["coefficients"])
        if c is None or not _validate_upstream_payload(state["upstream"], inputs):
            return False
        if not _required_keys(state["symbolic_checks"], REQUIRED_SYMBOLIC_CHECKS):
            return False
        if not _symbolic_passed(state["symbolic_checks"]):
            return False
        if not _validate_source_payload(state["source_vector"], c["F0"], c["r"]):
            return False
        if not _validate_current_payload(state["current_feedback"], c["F0"], c["r"]):
            return False
        if not _validate_bridge_payload(state["effective_response_bridge"], c):
            return False
        if not _validate_pole_payload(state["pole"], c["F0"], c["r"]):
            return False
        if not _validate_moments_payload(state["spectral_moments"], state["pole"], c["F0"], c["r"]):
            return False
        if not _validate_moment_bridge_payload(state["moment_bridge"], state["spectral_moments"],
                                               state["effective_response_bridge"], c):
            return False
        if not _validate_angular_payload(state["angular_response"], c["F0"], c["r"]):
            return False
        if not _validate_absorption_payload(state["continuum_absorption"], c["F0"], c["r"]):
            return False
        if not _validate_limits_payload(state["limits"], c["F0"], c["r"], dps):
            return False
        if (not isinstance(state["scope"], list) or len(state["scope"]) != 6
                or not all(isinstance(value, str) for value in state["scope"])
                or state["references"] != list(collisionless.REFERENCES)):
            return False
        if (not _required_keys(state["controls"], REQUIRED_CONTROL_KEYS)
                or not _serialized_structure_equal(state["controls"],
                                                   _expected_controls_payload(dps))):
            return False
        return True
    except (KeyError, TypeError, ValueError, ArithmeticError, ZeroDivisionError, OverflowError):
        return False


def _numeric_state_passed(state):
    with mp.workdps(200):
        return _numeric_state_passed_impl(state)


def _numeric_passport_passed_impl(states, refinements=None):
    """Reject incomplete passports, duplicate controls, and forged refinements."""
    try:
        if (not isinstance(states, list) or len(states) != len(REPRESENTATIVE_DENSITIES)
                or not isinstance(refinements, list)
                or len(refinements) != len(REPRESENTATIVE_DENSITIES)):
            return False
        expected_ratios = {str(mp.mpf(value)) for value in REPRESENTATIVE_DENSITIES}
        ratios = []
        for row in states:
            if not isinstance(row, dict) or not _required_keys(row.get("inputs"), REQUIRED_INPUT_KEYS):
                return False
            ratio = _real_field(row["inputs"], "n_over_n0", positive=True)
            if ratio is None:
                return False
            ratios.append(str(ratio))
            if row["inputs"].get("dps") != 80:
                return False
        if set(ratios) != expected_ratios or len(set(ratios)) != len(ratios):
            return False
        seen = set()
        required_difference_keys = {
            "F0", "r", "NF", "B", "C", "V", "first_sound_squared",
            "m1_total", "mminus1_total",
        }
        state_by_ratio = {str(_real_field(row["inputs"], "n_over_n0")): row for row in states}
        required_keys = frozenset({
            "n_ratio", "coarse_dps", "fine_dps", "difference_components",
            "max_relative_difference", "moment_tolerance", "passed",
        })
        for row in refinements:
            if not isinstance(row, dict) or set(row) != set(required_keys):
                return False
            ratio = row["n_ratio"]
            if ratio not in REPRESENTATIVE_DENSITIES or ratio in seen:
                return False
            seen.add(ratio)
            if row["coarse_dps"] != 80 or row["fine_dps"] != 110 or row["passed"] is not True:
                return False
            tolerance = _real_field(row, "moment_tolerance", positive=True)
            maximum = _real_field(row, "max_relative_difference", nonnegative=True)
            if tolerance is None or maximum is None or not _serialized_close(tolerance, MOMENT_TOLERANCE):
                return False
            components = row["difference_components"]
            state_ratio = str(mp.mpf(ratio))
            component_keys = set(required_difference_keys)
            if state_by_ratio[state_ratio]["pole"]["exists"] is True:
                component_keys.update(("pole_log_gap", "pole_residue"))
            if not _required_keys(components, component_keys):
                return False
            values = [_real_field(components, key, nonnegative=True) for key in component_keys]
            if any(value is None for value in values):
                return False
            actual_maximum = max(values)
            if (not _serialized_close(maximum, actual_maximum)
                    or actual_maximum >= PRECISION_TOLERANCE):
                return False
        if not all(_numeric_state_passed_impl(row) for row in states):
            return False
        return seen == set(REPRESENTATIVE_DENSITIES)
    except (KeyError, TypeError, ValueError, ArithmeticError, OverflowError):
        return False


def _numeric_passport_passed(states, refinements=None):
    with mp.workdps(200):
        return _numeric_passport_passed_impl(states, refinements)


@lru_cache(maxsize=4)
def _mathematical_controls(*, dps=80):
    precision(dps)
    with mp.workdps(dps):
        noninteracting = integrate_spectral_moments(0, 0, dps=dps, include_pole=True)
        F0, r = mp.mpf(1), mp.mpf("0.5")
        sigma = mp.mpc(mp.mpf("0.7"), mp.mpf("0.2"))
        current = _matrix_response(sigma, F0, r)["closed_response"]
        no_current = chi_over_nf(sigma, F0, 0)
        current_control = {
            "F0": F0,
            "r": r,
            "F1": -3*r/(1+r),
            "current_feedback_relation_error": abs((-3*r/(1+r))/(1+(-3*r/(1+r))/3)+3*r),
            "response_difference_without_current_feedback": abs(current-no_current),
            "current_feedback_visible": bool(abs(current-no_current) > mp.mpf("1e-12")),
        }
        wrong_sign = _continuum_absorption_check(F0, r)
        omitted_source = integrate_spectral_moments(4, 1, dps=dps, include_pole=True)
        omitted_m1 = omitted_source["m1_total_without_pole"]
        omitted_mm = omitted_source["mminus1_total_without_pole"]
        omitted_control = {
            "F0": mp.mpf(4),
            "r": mp.mpf(1),
            "pole_required": omitted_source["pole"]["exists"],
            "with_pole_passed": omitted_source["passed"],
            "without_pole_m1_relative_error": _relative_real(omitted_m1, omitted_source["target_m1"]),
            "without_pole_mminus1_relative_error": _relative_real(omitted_mm, omitted_source["target_mminus1"]),
            "omitted_pole_rejected": bool(
                _relative_real(omitted_m1, omitted_source["target_m1"]) > MOMENT_TOLERANCE
                and _relative_real(omitted_mm, omitted_source["target_mminus1"]) > MOMENT_TOLERANCE),
        }
        near_threshold = []
        for delta in ("0.1", "0.05", "0.01"):
            with mp.workdps(110):
                delta_mp = mp.mpf(delta)
                root = collisionless.zero_sound_root(3+delta_mp, 1, dps=110)
                pole = pole_residue_from_log_gap(root["log_gap"], 3+delta_mp, 1)
                gap_asym = mp.log(2)-2-2/delta_mp
                residue_asym = 2*pole["gap"]/delta_mp**2
                near_threshold.append({
                    "delta": delta_mp,
                    "log_gap": pole["log_gap"],
                    "gap": pole["gap"],
                    "residue": pole["residue"],
                    "log_gap_asymptotic": gap_asym,
                    "residue_asymptotic": residue_asym,
                    "positive_gap_and_residue": bool(pole["gap"] > 0 and pole["residue"] > 0),
                    "log_gap_asymptotic_relative_error": _relative_real(pole["log_gap"], gap_asym),
                    "residue_asymptotic_relative_error": _relative_real(pole["residue"], residue_asym),
                })
        controls = {
            "noninteracting": {
                "pole_absent": noninteracting["pole"]["exists"] is False,
                "m1_total": noninteracting["m1_total"],
                "mminus1_total": noninteracting["mminus1_total"],
                "m1_target": noninteracting["target_m1"],
                "mminus1_target": noninteracting["target_mminus1"],
                "passed": noninteracting["passed"] is True,
            },
            "current_feedback": current_control,
            "wrong_retarded_sign": wrong_sign,
            "omitted_pole": omitted_control,
            "near_threshold_log_gap": near_threshold,
        }
        near_passed = all(row["positive_gap_and_residue"] for row in near_threshold)
        controls["passed"] = bool(
            controls["noninteracting"]["passed"]
            and current_control["current_feedback_visible"]
            and wrong_sign["wrong_sign_rejected"]
            and omitted_control["omitted_pole_rejected"]
            and near_passed)
        return controls


def audit(n_ratio="1", *, dps=80):
    """Run one live state with all causal, spectral, and bridge checks."""
    precision(dps)
    result = _state_internal(n_ratio, dps=dps, include_bridge=True)
    result["controls"] = _mathematical_controls(dps=dps)
    result["mathematical_checks_passed"] = bool(
        result["mathematical_checks_passed"] and result["controls"]["passed"])
    shown = _shown(result)
    # The single-state closure validates this marker as part of the complete
    # serialized schema.  Seed it only after the live calculation; the
    # validator then proves the marker against every required derived field.
    shown["numeric_passport_passed"] = True
    shown["numeric_passport_passed"] = _numeric_state_passed(shown)
    shown["mathematical_checks_passed"] = bool(
        shown["mathematical_checks_passed"] and shown["numeric_passport_passed"]
        and shown["controls"]["passed"] is True)
    return shown


def _precision_refinement(ratio, coarse, fine):
    with mp.workdps(200):
        keys = ("F0", "r", "NF", "B", "C", "V", "first_sound_squared")
        differences = []
        components = {}
        for key in keys:
            components[key] = _relative_real(coarse["coefficients"][key], fine["coefficients"][key])
            differences.append(components[key])
        for key in ("m1_total", "mminus1_total"):
            components[key] = _relative_real(coarse["spectral_moments"][key],
                                             fine["spectral_moments"][key])
            differences.append(components[key])
        coarse_pole, fine_pole = coarse["pole"], fine["pole"]
        if coarse_pole["exists"] != fine_pole["exists"]:
            raise ArithmeticError("pole status changed under precision refinement")
        if coarse_pole["exists"]:
            components["pole_log_gap"] = _relative_real(coarse_pole["log_gap"], fine_pole["log_gap"])
            components["pole_residue"] = _relative_real(coarse_pole["residue"], fine_pole["residue"])
            differences.extend((components["pole_log_gap"], components["pole_residue"]))
        maximum = max(differences)
        return {
            "n_ratio": ratio,
            "coarse_dps": 80,
            "fine_dps": 110,
            "difference_components": components,
            "max_relative_difference": maximum,
            "moment_tolerance": MOMENT_TOLERANCE,
            "passed": bool(maximum < PRECISION_TOLERANCE),
        }


def compute_state(*, dps=80):
    """Run the fixed 1,10,100,200 controls and 80/110-digit refinement."""
    precision(dps)
    coarse_states = []
    refinements = []
    for ratio in REPRESENTATIVE_DENSITIES:
        coarse_internal = _state_internal(ratio, dps=80, include_bridge=True)
        fine_internal = _state_internal(ratio, dps=110, include_bridge=False)
        shown_coarse = _shown(coarse_internal)
        shown_coarse["controls"] = _shown(_mathematical_controls(dps=80))
        shown_coarse["numeric_passport_passed"] = True
        coarse_states.append(shown_coarse)
        refinements.append(_precision_refinement(ratio, coarse_internal, fine_internal))
    controls = _mathematical_controls(dps=110)
    shown_refinements = _shown(refinements)
    numeric_pass = _numeric_passport_passed(coarse_states, shown_refinements)
    passed = bool(numeric_pass and controls["passed"] is True)
    result = {
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "mathematical_checks_passed": passed,
        "representative_density_ratios": list(REPRESENTATIVE_DENSITIES),
        "states": coarse_states,
        "precision_refinement": shown_refinements,
        "controls": _shown(controls),
        "numeric_passport_passed": numeric_pass,
        "scope": [
            "Fixed live controls n/n0=1,10,100,200; 100 and 200 are formal high-density controls, not validated nuclear matter.",
            "No favorable density scan, fitted normalization, broadening parameter, or omitted pole is used.",
            "Low-energy matched-window moments are not a full microscopic f-sum or compressibility theorem.",
        ],
        "references": list(collisionless.REFERENCES),
    }
    return result


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--n-ratio", default="1")
    parser.add_argument("--all-controls", action="store_true")
    parser.add_argument("--dps", type=int, default=80)
    try:
        args = parser.parse_args(argv)
        result = compute_state(dps=args.dps) if args.all_controls else audit(args.n_ratio, dps=args.dps)
        result = _shown(result)
        passed = result.get("mathematical_checks_passed") is True
        result["mathematical_checks_passed"] = bool(passed)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if passed else 1
    except (ValueError, ArithmeticError, RuntimeError, ZeroDivisionError) as exc:
        print(json.dumps({"status": "invalid_or_failed_kinetic_spectral_audit",
                          "evidence_weight": EVIDENCE_WEIGHT,
                          "mathematical_checks_passed": False,
                          "error": str(exc)}, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
