#!/usr/bin/env python3
"""Bounded spectral-structure audit for the stationary collisionless closure.

The calculation is deliberately additive.  It consumes the maintained
equilibrium coefficient producer and the maintained logarithmic pole/retarded
response helpers, then independently derives higher moments, a Jacobi
resolvent, spectral quadratures, pole sensitivities, and density drift.  The
scope is the stationary, collisionless, long-wave Landau truncation with
``F_l>=2 = 0``.  It is not a finite-k, collisional, gravitational, cyclic, or
empirical completion of NVG.

The command line interface prints strict JSON and has no artifact-writing
option.  In particular, no target physical result table is loaded or used to
rescale a numerical result.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

import mpmath as mp
import sympy as sp

import nvg_collisionless_response_audit as collisionless
import nvg_kinetic_spectral_audit as kinetic


STATUS = "CONDITIONAL_STATIONARY_COLLISIONLESS_SPECTRAL_STRUCTURE"
EVIDENCE_WEIGHT = 0

# These are numerical acceptance gates for the declared calculation, not
# physical agreement tolerances.  Moment and resolvent tests use their actual
# scales; tiny positive pole weights are checked with a separate nonzero
# comparison so that they cannot disappear into a unit absolute floor.
MOMENT_TOLERANCE = mp.mpf("1e-35")
JACOBI_TOLERANCE = mp.mpf("1e-35")
RESOLVENT_TOLERANCE = mp.mpf("1e-50")
POLE_FD_TOLERANCE = mp.mpf("1e-18")
DENSITY_TOLERANCE = mp.mpf("1e-12")
THRESHOLD_RESIDUAL_TOLERANCE = mp.mpf("1e-40")
THRESHOLD_PRECISION_TOLERANCE = mp.mpf("1e-35")

REPRESENTATIVE_DENSITIES = ("1", "10", "100", "200")
MOMENT_ORDERS = (-1, 1, 3, 5, 7, 9)
QUADRATURE_NODES = (0, 1, 3, 7, 15, 30, 60, mp.inf)
RESOLVENT_CASES = (("4", "1"), ("-0.5", "0"))
RESOLVENT_POINTS = (("0.3", "0.4"), ("1.8", "0.2"), ("-0.3", "0.4"))

REQUIRED_RESIDUALS = frozenset(collisionless.REQUIRED_RESIDUALS)
REQUIRED_COEFFICIENT_KEYS = frozenset({
    "n", "kF", "EF", "vF2", "NF", "B", "C", "V", "r", "F0", "F1",
    "mu", "first_sound_squared",
})
REQUIRED_SYMBOLIC_CHECKS = frozenset({
    "recurrence_through_m9", "m1_closed", "m3_closed", "m5_closed",
    "hankel_closed", "qminus_closed", "qplus_closed", "inverse_F0",
    "inverse_r",
})


def finite(value, name, *, positive=False, nonnegative=False):
    """Convert a real scalar to an mpmath value and fail closed."""
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
    """Convert a complex spectral argument while rejecting nonfinite parts."""
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


def number(value):
    """Return a strict-JSON decimal with enough digits for audit replay."""
    if isinstance(value, mp.mpc):
        if value.imag != 0:
            raise ValueError("complex values must be shown with _shown")
        value = value.real
    with mp.workdps(max(mp.mp.dps, 200)):
        return mp.nstr(finite(value, "derived number"), 50)


def _shown(value):
    """Convert mpmath values recursively to strict JSON-native values."""
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


def _relative(left, right, *, floor=None):
    """Relative error on the actual scale, optionally with an explicit floor."""
    left, right = finite_complex(left, "left"), finite_complex(right, "right")
    scale = max(abs(left), abs(right))
    if floor is not None:
        scale = max(scale, finite(floor, "relative floor", positive=True))
    if scale == 0:
        return mp.mpf(0)
    return abs(left - right) / scale


def _relative_real(left, right, *, floor=None):
    left, right = finite(left, "left"), finite(right, "right")
    scale = max(abs(left), abs(right))
    if floor is not None:
        scale = max(scale, finite(floor, "relative floor", positive=True))
    if scale == 0:
        return mp.mpf(0)
    return abs(left - right) / scale


def _nonzero_relative(left, right):
    """Relative error that rejects a vanished nonzero expected contribution."""
    left, right = finite(left, "left"), finite(right, "right")
    if left == 0 or right == 0:
        return mp.inf
    return abs(left - right) / max(abs(left), abs(right))


def _landau_parameters(F0, r):
    F0 = finite(F0, "F0")
    r = finite(r, "r", nonnegative=True)
    if F0 <= -1:
        raise ValueError("F0 must satisfy F0 > -1")
    return F0, r


def _required_coefficients(raw):
    """Validate an upstream coefficient dictionary and its source residuals."""
    if not isinstance(raw, dict) or not REQUIRED_COEFFICIENT_KEYS.issubset(raw):
        raise ArithmeticError("equilibrium coefficient producer returned an incomplete state")
    residuals = raw.get("residuals")
    if not isinstance(residuals, dict) or set(residuals) != REQUIRED_RESIDUALS:
        raise ArithmeticError("equilibrium residual certificate is missing or incomplete")
    checked_residuals = {}
    for key in REQUIRED_RESIDUALS:
        try:
            # Residuals are signed differences; only their absolute values
            # enter the source-consistency gate.
            value = finite(residuals[key], key)
        except ValueError as exc:
            raise ArithmeticError("equilibrium residual certificate is nonfinite") from exc
        checked_residuals[key] = value
    if any(abs(value) >= mp.mpf("1e-55") for value in checked_residuals.values()):
        raise ArithmeticError("equilibrium residual certificate exceeds 1e-55")
    values = {}
    for key in REQUIRED_COEFFICIENT_KEYS:
        values[key] = finite(raw[key], key)
    if any(values[key] <= 0 for key in ("n", "kF", "EF", "vF2", "NF", "C", "V", "mu",
                                        "first_sound_squared")):
        raise ArithmeticError("equilibrium state has a nonpositive required coefficient")
    if values["vF2"] >= 1 or values["r"] < 0 or values["F0"] <= -1:
        raise ArithmeticError("equilibrium state is outside its inherited representable corridor")
    # Independent algebraic relations catch a wrong source/parameter payload
    # even when the residual dictionary itself has the right shape.
    expected = {
        "F1": -3*values["r"]/(1+values["r"]),
        "F0": values["NF"]*(values["V"]-values["B"]**2/values["C"]),
        "mu": values["EF"] + values["n"]*values["V"],
        "first_sound_squared": values["vF2"]*(1+values["F0"])
        *(1+values["F1"]/3)/3,
    }
    # The producer's n=NF*EF*vF2/3 identity and r=nV/EF relation are checked
    # independently below, rather than inferred from the first-sound ratio.
    expected["r"] = values["n"]*values["V"]/values["EF"]
    expected_density = values["NF"]*values["EF"]*values["vF2"]/3
    if _relative_real(values["n"], expected_density) >= mp.mpf("1e-55"):
        raise ArithmeticError("equilibrium density/coefficient relation failed")
    for key, expected_value in expected.items():
        if _relative_real(values[key], expected_value) >= mp.mpf("1e-55"):
            raise ArithmeticError(f"equilibrium {key} relation failed")
    return values


def equilibrium_coefficients(n_ratio="1", *, dps=80):
    """Load one original equilibrium state through the maintained producer."""
    precision(dps)
    with mp.workdps(dps):
        ratio = finite(n_ratio, "n_ratio", positive=True)
        if not mp.mpf("1e-18") <= ratio <= mp.mpf("1e6"):
            raise ValueError("numerical audit covers 1e-18 <= n/n0 <= 1e6")
        raw = kinetic.coefficients(ratio, dps=dps)
        return _required_coefficients(raw)


def _alpha(index):
    return mp.mpf(1)/(2*index+1)


def coefficient_recurrence(F0, r, *, max_order=5):
    """Return c_n=2m_(2n-1) from the exact adopted recurrence."""
    F0, r = _landau_parameters(F0, r)
    if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 5:
        raise ValueError("max_order must be an integer >= 5 (through m9)")
    with mp.workdps(max(mp.mp.dps, 80)):
        coefficients = []
        moments = {}
        A = 1+r
        for n in range(1, max_order+1):
            rhs = _alpha(n)
            for j in range(1, n):
                d_j = F0*_alpha(j)-3*r*_alpha(j+1)
                rhs += d_j*coefficients[n-j-1]
            c_n = rhs/A
            coefficients.append(c_n)
            moments[2*n-1] = c_n/2
        return {"c": coefficients, "moments": moments}


def closed_moments(F0, r):
    """Closed m1,m3,m5 identities adopted by J-01."""
    F0, r = _landau_parameters(F0, r)
    A = 1+r
    return {
        -1: 1/(2*(1+F0)),
        1: 1/(6*A),
        3: (9+5*F0)/(90*A**2),
        5: (675+630*F0+175*F0**2+108*r)/(9450*A**3),
    }


@lru_cache(maxsize=1)
def symbolic_checks():
    """Exact SymPy recurrence, closed-form, Hankel, and inversion checks."""
    F0, r = sp.symbols("F0 r", real=True)
    A = 1+r
    alpha = lambda n: sp.Rational(1, 2*n+1)
    c = []
    for n in range(1, 6):
        rhs = alpha(n)
        for j in range(1, n):
            rhs += (F0*alpha(j)-3*r*alpha(j+1))*c[n-j-1]
        c.append(sp.factor(rhs/A))
    m1, m3, m5 = c[0]/2, c[1]/2, c[2]/2
    mminus = 1/(2*(1+F0))
    qminus = sp.factor(mminus*m3/m1**2-1)
    qplus = sp.factor(m1*m5/m3**2-1)
    # Independently expand the closed response in t=1/z.  The coefficient of
    # t^10 is 2*m9 and is not obtained from the recurrence loop above.
    t = sp.symbols("t")
    L_series = sum(t**(2*j)*sp.Rational(1, 2*j+1) for j in range(1, 8))
    g_series = sp.series(L_series/(1-(F0-3*r/t**2)*L_series), t, 0, 12).removeO()
    m9_series = sp.expand(g_series).coeff(t, 10)/2
    rows = {
        "recurrence_through_m9": c[4]/2 - m9_series,
        "m1_closed": m1 - 1/(6*A),
        "m3_closed": m3 - (9+5*F0)/(90*A**2),
        "m5_closed": m5 - (675+630*F0+175*F0**2+108*r)/(9450*A**3),
        "hankel_closed": m1*m5-m3**2 - 1/(525*A**3),
        "qminus_closed": qminus - 4/(5*(1+F0)),
    }
    # qplus identity is written in the inverse form: Q+ = 108(1+r)
    # /[7(5F0+9)^2].
    rows["qplus_closed"] = sp.factor(qplus - 108*(1+r)/(7*(5*F0+9)**2))
    rows["inverse_F0"] = (4/(5*qminus)-1) - F0
    rows["inverse_r"] = (7*(5*F0+9)**2*qplus/108-1) - r
    return {
        key: {"passed": bool(sp.simplify(value) == 0),
              "residual": "0" if sp.simplify(value) == 0 else str(sp.factor(value))}
        for key, value in rows.items()
    }


def _symbolic_checks_passed(checks):
    """Require the complete named symbolic proof set at every gate."""
    return bool(
        isinstance(checks, dict)
        and set(checks) == REQUIRED_SYMBOLIC_CHECKS
        and all(
            isinstance(row, dict)
            and set(row) == {"passed", "residual"}
            and row["passed"] is True
            and row["residual"] == "0"
            for row in checks.values()
        )
    )


def jacobi_matrix(F0, r, *, dimension):
    """Build the finite Jacobi truncation of B=A^(1/2)uA^(1/2)."""
    F0, r = _landau_parameters(F0, r)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2:
        raise ValueError("dimension must be an integer >= 2")
    A = 1+r
    factors = [1+F0, 1/A] + [mp.mpf(1)]*(dimension-2)
    matrix = mp.matrix(dimension)
    for ell in range(dimension-1):
        b = (ell+1)*mp.sqrt(factors[ell]*factors[ell+1])
        b /= mp.sqrt((2*ell+1)*(2*ell+3))
        matrix[ell, ell+1] = b
        matrix[ell+1, ell] = b
    return matrix


def jacobi_moments(F0, r, *, max_order=5, dimension=None):
    """Compute moments independently as finite Jacobi powers."""
    F0, r = _landau_parameters(F0, r)
    if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 5:
        raise ValueError("max_order must be an integer >= 5")
    dimension = max(8, max_order+2) if dimension is None else dimension
    matrix = jacobi_matrix(F0, r, dimension=dimension)
    vector = mp.matrix(dimension, 1)
    vector[0] = 1
    moments = {}
    for order in range(1, max_order+1):
        for _ in range(2):
            vector = matrix*vector
        moments[2*order-1] = vector[0]/(2*(1+F0))
    return moments


def recurrence_jacobi_check(F0, r, *, dps=80):
    """Compare recurrence and finite Jacobi powers through m9."""
    precision(dps)
    with mp.workdps(dps):
        F0, r = _landau_parameters(F0, r)
        recurrence = coefficient_recurrence(F0, r, max_order=5)["moments"]
        jacobi = jacobi_moments(F0, r, max_order=5, dimension=8)
        errors = {str(power): _relative_real(recurrence[power], jacobi[power])
                  for power in (1, 3, 5, 7, 9)}
        closed = closed_moments(F0, r)
        closed_errors = {str(power): _relative_real(recurrence[power], closed[power])
                         for power in (1, 3, 5)}
        hankel = recurrence[1]*recurrence[5]-recurrence[3]**2
        hankel_target = 1/(525*(1+r)**3)
        hankel_error = _relative_real(hankel, hankel_target)
        return {
            "recurrence": recurrence,
            "jacobi": jacobi,
            "relative_errors": errors,
            "max_relative_error": max(errors.values()),
            "closed_relative_errors": closed_errors,
            "max_closed_relative_error": max(closed_errors.values()),
            "hankel": hankel,
            "hankel_target": hankel_target,
            "hankel_relative_error": hankel_error,
            "passed": bool(max(errors.values()) < JACOBI_TOLERANCE
                           and max(closed_errors.values()) < MOMENT_TOLERANCE
                           and hankel_error < MOMENT_TOLERANCE),
        }


def _continuum_density(sigma, F0, r):
    """Direct retarded positive-frequency S/NF on 0<sigma<1."""
    sigma = finite(sigma, "continuum sigma")
    F0, r = _landau_parameters(F0, r)
    if not 0 < sigma < 1:
        return mp.mpf(0)
    x = sigma/2*(mp.log1p(sigma)-mp.log1p(-sigma))-1
    coefficient = F0-3*r*sigma**2
    denominator = (1-coefficient*x)**2 + (coefficient*mp.pi*sigma/2)**2
    density = sigma/(2*denominator)
    if not mp.isfinite(density) or density < 0:
        raise ArithmeticError("nonfinite or negative retarded continuum density")
    return density


def _quadrature_form(F0, r, power, form):
    if form == "exponential_endpoint":
        def transform(t):
            gap = mp.exp(-t)
            return 1-gap, gap
    elif form == "tanh_endpoint":
        def transform(t):
            return mp.tanh(t), 1/mp.cosh(t)**2
    else:
        raise ValueError("unknown endpoint transformation")

    def integrand(t):
        sigma, jacobian = transform(t)
        if sigma <= 0 or sigma >= 1:
            # The p=-1 endpoint has a finite limiting integrand, while the
            # transformed Jacobian vanishes at the upper endpoint.  Returning
            # zero at both exact endpoints leaves the quadrature unchanged.
            return mp.mpf(0)
        return sigma**power*_continuum_density(sigma, F0, r)*jacobian

    value = mp.quad(integrand, list(QUADRATURE_NODES))
    if not mp.isfinite(value) or value < 0:
        raise ArithmeticError("nonfinite or negative continuum quadrature")
    return value


def _pole_for(F0, r, *, dps):
    # Use the maintained stable log-gap representation and preserve the tiny
    # positive gap/residue fields without converting them through float.
    return kinetic.pole_data(F0, r, dps=dps)


def spectral_moments(F0, r, *, dps=80, include_pole=True, pole_multiplier=1):
    """Direct continuum-plus-delta moments for -1,1,3,5,7,9."""
    precision(dps)
    if not isinstance(include_pole, bool):
        raise ValueError("include_pole must be bool")
    pole_multiplier = finite(pole_multiplier, "pole_multiplier")
    with mp.workdps(dps):
        F0, r = _landau_parameters(F0, r)
        pole = _pole_for(F0, r, dps=dps)
        # Consume the maintained narrow pole identity checker at the actual
        # boundary.  In particular, do not let a caller-supplied deep copy,
        # false ``exists`` flag, wrong tiny gap, or wrong residue become a
        # mathematically passing spectrum merely because its global moment
        # contribution is small.
        if not kinetic._validate_pole_payload(pole, F0, r):
            raise ArithmeticError("kinetic pole payload failed its identity validation")
        continuum = {}
        for form in ("exponential_endpoint", "tanh_endpoint"):
            continuum[form] = {power: _quadrature_form(F0, r, power, form)
                               for power in MOMENT_ORDERS}
        pole_values = {}
        for power in MOMENT_ORDERS:
            if pole["exists"]:
                pole_values[power] = pole["sigma_p"]**power*pole["residue"]
            else:
                pole_values[power] = mp.mpf(0)
        pole_contributions_positive = bool(
            not pole["exists"] or all(value > 0 for value in pole_values.values()))
        totals = {}
        closed = closed_moments(F0, r)
        recurrence = coefficient_recurrence(F0, r, max_order=5)["moments"]
        # The static m_{-1} identity and the independently derived recurrence
        # cover every reported moment.  Keep the closed m1/m3/m5 forms as a
        # second algebraic check instead of using a measured value as a
        # normalization target.
        expected = {-1: closed[-1], **recurrence}
        for power in MOMENT_ORDERS:
            delta = pole_values[power] if include_pole else mp.mpf(0)
            totals[power] = continuum["exponential_endpoint"][power] + delta
        quadrature_errors = {
            power: _relative_real(continuum["exponential_endpoint"][power],
                                  continuum["tanh_endpoint"][power])
            for power in MOMENT_ORDERS
        }
        target_errors = {power: _relative_real(totals[power], expected[power])
                         for power in MOMENT_ORDERS}
        closed_errors = {power: _relative_real(recurrence[power], closed[power])
                         for power in (1, 3, 5)}
        alternate_target_errors = {
            power: _relative_real(
                continuum["tanh_endpoint"][power]
                + (pole_values[power] if include_pole else mp.mpf(0)),
                expected[power],
            )
            for power in MOMENT_ORDERS
        }
        pole_piece_errors = {}
        if pole["exists"]:
            # Re-derive the expected delta weights from the validated
            # log-gap, preserving actual-scale comparisons for tiny residues.
            reference = kinetic.pole_residue_from_log_gap(pole["log_gap"], F0, r)
            expected_pole_values = {
                power: reference["sigma_p"]**power*reference["residue"]
                for power in MOMENT_ORDERS
            }
        else:
            expected_pole_values = {power: mp.mpf(0) for power in MOMENT_ORDERS}
        for power in MOMENT_ORDERS:
            actual = pole_values[power] if include_pole else mp.mpf(0)
            if pole["exists"]:
                pole_piece_errors[power] = _nonzero_relative(actual, expected_pole_values[power])
            else:
                pole_piece_errors[power] = mp.mpf(0) if actual == 0 else mp.inf
        passed = bool(
            all(error < MOMENT_TOLERANCE for error in quadrature_errors.values())
            and all(error < MOMENT_TOLERANCE for error in target_errors.values())
            and all(error < MOMENT_TOLERANCE for error in alternate_target_errors.values())
            and all(error < MOMENT_TOLERANCE for error in closed_errors.values())
            and (not pole["exists"] or (include_pole and pole_multiplier == 1
                                         and all(error < MOMENT_TOLERANCE
                                                for error in pole_piece_errors.values())
                                         and pole_contributions_positive))
            and (not pole["exists"] or pole["delta_piece_required"] is True)
        )
        # ``pole_multiplier`` is only a negative-control hook; the physical
        # total must include exactly one analytic delta weight.
        if pole["exists"] and include_pole and pole_multiplier != 1:
            for power in MOMENT_ORDERS:
                totals[power] = continuum["exponential_endpoint"][power] + pole_multiplier*pole_values[power]
            target_errors = {power: _relative_real(totals[power], expected[power])
                             for power in MOMENT_ORDERS}
            passed = False
        return {
            "pole": pole,
            "continuum": continuum,
            "pole_contribution": pole_values,
            "total": totals,
            "expected": expected,
            "recurrence_expected": recurrence,
            "closed_identity_relative_errors": closed_errors,
            "quadrature_relative_errors": quadrature_errors,
            "target_relative_errors": target_errors,
            "alternate_target_relative_errors": alternate_target_errors,
            "pole_piece_relative_errors": pole_piece_errors,
            "pole_contributions_positive": pole_contributions_positive,
            "include_pole": include_pole,
            "pole_multiplier": pole_multiplier,
            "integrals_converged": bool(all(error < MOMENT_TOLERANCE
                                             for error in quadrature_errors.values())),
            "passed": passed,
        }


def c_scale_and_shear(raw, moments):
    """Return the spectral centroid, offset, and collisionless shear identity."""
    m1 = moments["total"][1]
    m3 = moments["total"][3]
    vF2 = raw["vF2"]
    A = 1+raw["r"]
    c_scale_squared = vF2*m3/m1
    c_first_squared = raw["first_sound_squared"]
    offset = 4*vF2/(15*A)
    n, pF, vF, mu = raw["n"], raw["kF"], mp.sqrt(vF2), raw["mu"]
    G_kin = n*pF*vF/5
    shear_ratio = 4*G_kin/(3*n*mu)
    return {
        "c_scale_squared": c_scale_squared,
        "c_first_squared": c_first_squared,
        "offset": offset,
        "offset_identity_residual": c_scale_squared-c_first_squared-offset,
        "G_kin": G_kin,
        "G_kin_definition": "n*pF*vF/5 collisionless kinetic shear specialization",
        "shear_ratio": shear_ratio,
        "shear_identity_residual": shear_ratio-offset,
        "c_scale_is_not_pole_or_front_speed": True,
        "passed": bool(abs(c_scale_squared-c_first_squared-offset)
                       /max(abs(offset), mp.mpf("1e-80")) < MOMENT_TOLERANCE
                       and abs(shear_ratio-offset)
                       /max(abs(offset), mp.mpf("1e-80")) < MOMENT_TOLERANCE),
    }


def ratio_inversion(F0, r, moments):
    """Compute Q-, Q+ and the truncated-model synthetic inverse."""
    F0, r = _landau_parameters(F0, r)
    values = moments["total"] if isinstance(moments, dict) and "total" in moments else moments
    mminus, m1, m3, m5 = (values[-1], values[1], values[3], values[5])
    qminus = mminus*m3/m1**2-1
    qplus = m1*m5/m3**2-1
    if qminus <= 0 or qplus <= 0:
        raise ArithmeticError("spectral ratios must be positive for the inverse")
    inferred_F0 = 4/(5*qminus)-1
    inferred_r = 7*(5*inferred_F0+9)**2*qplus/108-1
    sensitivities = {
        "dF0_dlnQminus": -(1+inferred_F0),
        "dF0_dlnQplus": mp.mpf(0),
        "dr_dlnQminus": -(mp.mpf(35)/54)*(5*inferred_F0+9)*qplus*(1+inferred_F0),
        "dr_dlnQplus": 1+inferred_r,
    }
    return {
        "Qminus": qminus,
        "Qplus": qplus,
        "inferred_F0": inferred_F0,
        "inferred_r": inferred_r,
        "input_log_sensitivity": sensitivities,
        "round_trip_relative_errors": {
            # A zero interaction parameter is an exact boundary; use a unit
            # scale only for this parameter-recovery diagnostic so numerical
            # last bits are not misreported as 100% relative error.
            "F0": _relative_real(inferred_F0, F0, floor=1),
            "r": _relative_real(inferred_r, r, floor=1),
        },
        "truncated_model_only": True,
        "passed": bool(_relative_real(inferred_F0, F0, floor=1) < MOMENT_TOLERANCE
                       and _relative_real(inferred_r, r, floor=1) < MOMENT_TOLERANCE),
    }


def higher_harmonic_degeneracy(F0="4", r="1", F2="5", F3="7"):
    """Return the adopted mathematical non-identifiability counterexample."""
    F0, r = _landau_parameters(F0, r)
    F2 = finite(F2, "F2")
    F3 = finite(F3, "F3")
    a0, a1 = 1+F0, 1/(1+r)
    a2, a3 = 1+F2/5, 1+F3/7
    if min(a0, a1, a2, a3) <= 0:
        raise ValueError("degeneracy control must have positive Jacobi energies")
    qminus = 4*a2/(5*a0)
    hankel = a1**3*a2**2*a3/525
    inferred_F0 = a0/a2-1
    inferred_r = (1+r)*a3-1
    return {
        "inputs": {"F0": F0, "r": r, "F2": F2, "F3": F3},
        "positive_jacobi_factors": {"a0": a0, "a1": a1, "a2": a2, "a3": a3},
        "Qminus": qminus,
        "hankel_determinant": hankel,
        "inferred_F0": inferred_F0,
        "inferred_r": inferred_r,
        "expected_concrete_inferred_F0": mp.mpf("1.5"),
        "expected_concrete_inferred_r": mp.mpf("3"),
        "is_mathematical_counterexample_only": True,
        "passed": bool(inferred_F0 == mp.mpf("1.5") and inferred_r == mp.mpf("3")),
    }


def jacobi_resolvent(z, F0, r, *, dimension=256):
    """Evaluate the finite Jacobi continued fraction for M and g."""
    precision(max(80, min(mp.mp.dps, 200)))
    z = finite_complex(z, "z")
    F0, r = _landau_parameters(F0, r)
    matrix_dimension = int(dimension)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or matrix_dimension < 2:
        raise ValueError("dimension must be an integer >= 2")
    A = 1+r
    factors = [1+F0, 1/A] + [mp.mpf(1)]*(matrix_dimension-2)
    tail = z
    for ell in range(matrix_dimension-2, -1, -1):
        b = (ell+1)*mp.sqrt(factors[ell]*factors[ell+1])
        b /= mp.sqrt((2*ell+1)*(2*ell+3))
        if tail == 0:
            raise ArithmeticError("Jacobi continued fraction encountered a zero tail")
        tail = z-b*b/tail
    M = 1/tail
    g = (z*M-1)/(1+F0)
    return {"M": M, "g": g, "H": -z*g}


def resolvent_checks(*, dps=100):
    """Compare dimensions 32..256 to the closed response at six points."""
    precision(dps)
    with mp.workdps(dps):
        rows = []
        for F0_text, r_text in RESOLVENT_CASES:
            F0, r = mp.mpf(F0_text), mp.mpf(r_text)
            for real_text, imag_text in RESOLVENT_POINTS:
                z = mp.mpc(mp.mpf(real_text), mp.mpf(imag_text))
                closed = kinetic.chi_over_nf(z, F0, r)
                dimensions = {}
                for dimension in (32, 64, 128, 256):
                    row = jacobi_resolvent(z, F0, r, dimension=dimension)
                    dimensions[dimension] = {
                        "g": row["g"],
                        "H": row["H"],
                        "absolute_error": abs(row["g"]-closed),
                        "passivity_im_H": mp.im(row["H"]),
                    }
                rows.append({
                    "F0": F0,
                    "r": r,
                    "z": z,
                    "closed_g": closed,
                    "dimensions": dimensions,
                    "dimension256_absolute_error": dimensions[256]["absolute_error"],
                    "dimension256_passivity": dimensions[256]["passivity_im_H"] > 0,
                })
        max_error = max(row["dimension256_absolute_error"] for row in rows)
        min_passivity = min(row["dimensions"][256]["passivity_im_H"] for row in rows)
        return {
            "rows": rows,
            "max_dimension256_absolute_error": max_error,
            "min_dimension256_Im_minus_zg": min_passivity,
            "passive_function": "H=-z*g; g itself is not the Herglotz function",
            "passed": bool(max_error < RESOLVENT_TOLERANCE and min_passivity > 0),
        }


def outside_domain_instability_witness(*, dps=110):
    """Solve the separately labelled F0=-1.5,r=0 upper-half-plane witness."""
    precision(dps)
    with mp.workdps(dps):
        F0, r = mp.mpf("-1.5"), mp.mpf(0)

        def denominator(y):
            sigma = mp.j*y
            return 1-(F0-3*r*sigma**2)*kinetic.retarded_lindhard(sigma)

        root = mp.findroot(denominator, (mp.mpf(".2"), mp.mpf(".3")))
        y = mp.re(root)
        residual = abs(denominator(y))
        return {
            "F0": F0,
            "r": r,
            "upper_half_plane_sigma": mp.j*y,
            "y": y,
            "denominator_residual": residual,
            "admissible_state": False,
            "reason": "F0<=-1 violates positive l=0 energy; this is a mathematical exception control",
            "passed": bool(y > 0 and residual < mp.mpf("1e-70")),
        }


def _root_log_gap(F0, r, *, dps):
    result = collisionless.zero_sound_root(F0, r, dps=dps)
    if result is None:
        return None
    return finite(result["log_gap"], "log_gap")


def pole_sensitivity(F0, r, *, dps=110, step="1e-12"):
    """Verify exact pole partials using independent log-gap finite differences."""
    precision(dps)
    with mp.workdps(dps):
        F0, r = _landau_parameters(F0, r)
        step = finite(step, "step", positive=True)
        base_log_gap = _root_log_gap(F0, r, dps=dps)
        if base_log_gap is None:
            return {
                "exists": False,
                "F0": F0,
                "r": r,
                "step": step,
                "partial_F0": None,
                "partial_r": None,
                "finite_difference_F0": None,
                "finite_difference_r": None,
                "relative_error_F0": None,
                "relative_error_r": None,
                "passed": True,
                "absence_semantics": "absent, not zero",
            }
        pole = kinetic.pole_residue_from_log_gap(base_log_gap, F0, r)
        sigma = pole["sigma_p"]
        residue = pole["residue"]
        partial_F0 = residue
        partial_r = -3*sigma**2*residue
        log_plus = _root_log_gap(F0+step, r, dps=dps)
        log_minus = _root_log_gap(F0-step, r, dps=dps)
        log_r_plus = _root_log_gap(F0, r+step, dps=dps)
        log_r_minus = _root_log_gap(F0, r-step, dps=dps)
        if None in (log_plus, log_minus, log_r_plus, log_r_minus):
            raise ArithmeticError("finite-difference pole neighborhood lost its isolated root")
        gap = mp.exp(base_log_gap)
        fd_F0 = gap*(log_plus-log_minus)/(2*step)
        fd_r = gap*(log_r_plus-log_r_minus)/(2*step)
        error_F0 = _nonzero_relative(fd_F0, partial_F0)
        error_r = _nonzero_relative(fd_r, partial_r)
        return {
            "exists": True,
            "F0": F0,
            "r": r,
            "step": step,
            "log_gap": base_log_gap,
            "sigma_p": sigma,
            "residue": residue,
            "partial_F0": partial_F0,
            "partial_r": partial_r,
            "finite_difference_F0": fd_F0,
            "finite_difference_r": fd_r,
            "relative_error_F0": error_F0,
            "relative_error_r": error_r,
            "passed": bool(error_F0 < POLE_FD_TOLERANCE and error_r < POLE_FD_TOLERANCE),
            "absence_semantics": "not applicable; isolated root exists",
        }


def _coefficient_slope_at(n_ratio, h, *, dps):
    ratio = finite(n_ratio, "n_ratio", positive=True)
    h = finite(h, "log-density step", positive=True)
    plus = equilibrium_coefficients(ratio*mp.exp(h), dps=dps)
    minus = equilibrium_coefficients(ratio*mp.exp(-h), dps=dps)
    return {
        "h": h,
        "dF0_dlnn": (plus["F0"]-minus["F0"])/(2*h),
        "dr_dlnn": (plus["r"]-minus["r"])/(2*h),
    }


def _richardson(coarse, fine):
    return {
        "dF0_dlnn": fine["dF0_dlnn"]+(fine["dF0_dlnn"]-coarse["dF0_dlnn"])/3,
        "dr_dlnn": fine["dr_dlnn"]+(fine["dr_dlnn"]-coarse["dr_dlnn"])/3,
    }


def density_slopes(n_ratio, *, dps=80):
    """Compute h,h/2,h/4 equilibrium slopes and independent pole slopes."""
    precision(dps)
    with mp.workdps(dps):
        ratio = finite(n_ratio, "n_ratio", positive=True)
        base = equilibrium_coefficients(ratio, dps=dps)
        h_values = (mp.mpf("1e-4"), mp.mpf("5e-5"), mp.mpf("2.5e-5"))
        raw_slopes = [_coefficient_slope_at(ratio, h, dps=dps) for h in h_values]
        richardson_h2 = _richardson(raw_slopes[0], raw_slopes[1])
        richardson_h4 = _richardson(raw_slopes[1], raw_slopes[2])
        adjacent = max(abs(richardson_h4[key]-richardson_h2[key])
                       for key in ("dF0_dlnn", "dr_dlnn"))
        base_log_gap = _root_log_gap(base["F0"], base["r"], dps=dps)
        if base_log_gap is None:
            pole = None
            pole_slopes = [None, None, None]
            formula_slopes = None
            independent_errors = None
        else:
            pole = kinetic.pole_residue_from_log_gap(base_log_gap, base["F0"], base["r"])
            sigma = pole["sigma_p"]
            residue = pole["residue"]
            formula_slopes = {
                "d_sigma_p_dlnn": residue*(richardson_h4["dF0_dlnn"]
                                            -3*sigma**2*richardson_h4["dr_dlnn"]),
                "from_richardson": True,
            }
            pole_slopes = []
            for h in h_values:
                # Solve both neighboring log gaps independently; the base
                # log-gap is used only to retain the positive multiplicative
                # gap in the derivative.
                cp = equilibrium_coefficients(ratio*mp.exp(h), dps=dps)
                cm = equilibrium_coefficients(ratio*mp.exp(-h), dps=dps)
                lp = _root_log_gap(cp["F0"], cp["r"], dps=dps)
                lm = _root_log_gap(cm["F0"], cm["r"], dps=dps)
                if lp is None or lm is None:
                    raise ArithmeticError("density finite-difference neighborhood lost pole")
                pole_slopes.append(mp.exp(base_log_gap)*(lp-lm)/(2*h))
            independent_richardson_h2 = pole_slopes[1]+(pole_slopes[1]-pole_slopes[0])/3
            independent_richardson_h4 = pole_slopes[2]+(pole_slopes[2]-pole_slopes[1])/3
            independent_errors = {
                "h4_formula_vs_independent": _relative_real(
                    formula_slopes["d_sigma_p_dlnn"], independent_richardson_h4),
                "h4_formula_vs_independent_absolute": abs(
                    formula_slopes["d_sigma_p_dlnn"]-independent_richardson_h4),
                "adjacent_independent_richardson": abs(independent_richardson_h4
                                                         -independent_richardson_h2),
            }
            formula_slopes["independent_richardson_h4"] = independent_richardson_h4
        return {
            "n_ratio": ratio,
            "base_F0": base["F0"],
            "base_r": base["r"],
            "raw_coefficient_slopes": raw_slopes,
            "richardson_h2": richardson_h2,
            "richardson_h4": richardson_h4,
            "adjacent_richardson_max_absolute_difference": adjacent,
            "pole_exists": base_log_gap is not None,
            "pole": pole,
            "pole_slopes_independent_log_gap": pole_slopes,
            "pole_slope_formula": formula_slopes,
            "pole_slope_independent_errors": independent_errors,
            "passed": bool(adjacent < DENSITY_TOLERANCE
                           and (base_log_gap is None or (
                               independent_errors["h4_formula_vs_independent"] < DENSITY_TOLERANCE
                               and independent_errors["h4_formula_vs_independent_absolute"]
                               < DENSITY_TOLERANCE
                               and independent_errors["adjacent_independent_richardson"]
                               < DENSITY_TOLERANCE))),
            "absence_semantics": "absent, not zero" if base_log_gap is None
            else "isolated pole branch present",
        }


def density_threshold(*, dps=80):
    """Find the declared coefficient crossing on [100,200], not a phase transition."""
    precision(dps)
    with mp.workdps(dps):
        def difference(ratio):
            state = equilibrium_coefficients(ratio, dps=dps)
            return state["F0"]-3*state["r"]

        low, high = difference(mp.mpf(100)), difference(mp.mpf(200))
        if not low > 0 > high:
            raise ArithmeticError("declared density threshold is not bracketed")
        root = mp.findroot(difference, (mp.mpf(100), mp.mpf(200)))
        root = mp.re(root)
        residual = difference(root)
        if not 100 < root < 200 or abs(residual) >= THRESHOLD_RESIDUAL_TOLERANCE:
            raise ArithmeticError("density threshold residual or bracket failed")
        return {
            "n_over_n0": root,
            "difference_at_100": low,
            "difference_at_200": high,
            "coefficient_residual": residual,
            "globally_unique_threshold_proven": False,
            "scope": "formal isolated-mode loss into the continuum, not a thermodynamic phase transition",
            "passed": True,
        }


def threshold_precision_check():
    low = density_threshold(dps=80)
    high = density_threshold(dps=110)
    difference = abs(low["n_over_n0"]-high["n_over_n0"])
    return {
        "dps80": low,
        "dps110": high,
        "absolute_root_difference": difference,
        "passed": bool(difference < THRESHOLD_PRECISION_TOLERANCE
                       and abs(low["coefficient_residual"]) < THRESHOLD_RESIDUAL_TOLERANCE
                       and abs(high["coefficient_residual"]) < THRESHOLD_RESIDUAL_TOLERANCE),
    }


def negative_controls(F0="4", r="1", *, dps=80, moments=None):
    """Run substantive wrong-sign/weighting/Hankel/slope controls."""
    precision(dps)
    with mp.workdps(dps):
        F0, r = _landau_parameters(F0, r)
        if moments is None:
            moments = spectral_moments(F0, r, dps=dps, include_pole=True)
        wrong_sign = kinetic.continuum_spectral_density(mp.mpf(".5"), F0, r,
                                                         branch="advanced")
        if moments["pole"]["exists"]:
            doubled_totals = {
                power: moments["continuum"]["exponential_endpoint"][power]
                + 2*moments["pole_contribution"][power]
                for power in MOMENT_ORDERS
            }
            doubled_errors = {
                power: _relative_real(doubled_totals[power], moments["expected"][power])
                for power in MOMENT_ORDERS
            }
            doubled_rejected = any(error >= MOMENT_TOLERANCE
                                   for error in doubled_errors.values())
            omitted_total = {
                power: moments["continuum"]["exponential_endpoint"][power]
                for power in MOMENT_ORDERS
            }
            omitted_errors = {
                power: _relative_real(omitted_total[power], moments["expected"][power])
                for power in MOMENT_ORDERS
            }
            omitted_rejected = any(error >= MOMENT_TOLERANCE
                                   for error in omitted_errors.values())
        else:
            doubled_rejected = False
            omitted_rejected = False
        expected_hankel = 1/(525*(1+r)**3)
        wrong_hankel = moments["total"][1]*moments["total"][5]-moments["total"][3]**2
        wrong_quadrature = (moments["continuum"]["exponential_endpoint"][3]
                            *mp.mpf("1.001"))
        slopes = density_slopes("1", dps=dps)
        wrong_slope = slopes["pole_slope_formula"]["d_sigma_p_dlnn"] + mp.mpf(".01")
        return {
            "wrong_retarded_sign_rejected": bool(wrong_sign < 0),
            # No-pole controls are not failures: there is no delta weight to
            # omit or double.  The positive-pole branch above is the required
            # substantive rejection test.
            "missing_pole_rejected": bool(not moments["pole"]["exists"] or omitted_rejected),
            "doubled_pole_rejected": bool(not moments["pole"]["exists"] or doubled_rejected),
            "tiny_positive_pole_weight_checked_nonzero": bool(
                not moments["pole"]["exists"] or all(value < MOMENT_TOLERANCE
                                                       for value in moments["pole_piece_relative_errors"].values())),
            "hankel_identity_passed": bool(_relative_real(wrong_hankel, expected_hankel)
                                            < MOMENT_TOLERANCE),
            "intentionally_wrong_hankel_rejected": bool(
                _relative_real(wrong_hankel+mp.mpf(".01"), expected_hankel)
                >= MOMENT_TOLERANCE),
            "intentionally_wrong_quadrature_rejected": bool(
                _relative_real(wrong_quadrature,
                               moments["continuum"]["tanh_endpoint"][3])
                >= MOMENT_TOLERANCE),
            "intentionally_wrong_slope_rejected": bool(
                _relative_real(wrong_slope, slopes["pole_slope_formula"]["d_sigma_p_dlnn"])
                >= DENSITY_TOLERANCE),
        }


def _state(n_ratio, *, dps=80, full=True):
    precision(dps)
    with mp.workdps(dps):
        raw = equilibrium_coefficients(n_ratio, dps=dps)
        moments = spectral_moments(raw["F0"], raw["r"], dps=dps, include_pole=True)
        recurrence_jacobi = recurrence_jacobi_check(raw["F0"], raw["r"], dps=dps)
        scale = c_scale_and_shear(raw, moments)
        inversion = ratio_inversion(raw["F0"], raw["r"], moments)
        symbolic = symbolic_checks()
        symbolic_passed = _symbolic_checks_passed(symbolic)
        passed = bool(symbolic_passed and moments["passed"] and recurrence_jacobi["passed"]
                      and scale["passed"] and inversion["passed"])
        result = {
            "n_ratio": finite(n_ratio, "n_ratio", positive=True),
            "dps": dps,
            "coefficients": raw,
            "moments": moments,
            "recurrence_jacobi": recurrence_jacobi,
            "scale_and_shear": scale,
            "ratio_inversion": inversion,
            "symbolic_checks": symbolic,
            "passed": passed,
        }
        if full:
            result["resolvent"] = resolvent_checks(dps=max(80, min(100, dps)))
            result["outside_domain_witness"] = outside_domain_instability_witness(dps=max(80, min(110, dps)))
            result["pole_sensitivity_ordinary"] = pole_sensitivity(4, 1, dps=110)
            result["pole_sensitivity_near_threshold"] = pole_sensitivity("3.01", 1, dps=110)
            result["higher_harmonic_degeneracy"] = higher_harmonic_degeneracy()
            result["negative_controls"] = negative_controls(raw["F0"], raw["r"], dps=dps,
                                                             moments=moments)
            result["passed"] = bool(result["passed"] and result["resolvent"]["passed"]
                                      and result["outside_domain_witness"]["passed"]
                                      and result["pole_sensitivity_ordinary"]["passed"]
                                      and result["pole_sensitivity_near_threshold"]["passed"]
                                      and result["higher_harmonic_degeneracy"]["passed"]
                                      and all(result["negative_controls"].values()))
        return result


def audit(n_ratio="1", *, dps=80):
    """Run one complete bounded spectral-structure audit."""
    state = _state(n_ratio, dps=dps, full=True)
    shown = _shown(state)
    return {
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "mathematical_checks_passed": bool(state["passed"]),
        "state": shown,
        "scope": [
            "Stationary isotropic normal T=0 Hartree/Vlasov response in the accepted l=0,1 long-wave closure.",
            "All moments and Jacobi identities assume F_l>=2=0; higher-harmonic inversion is only a counterexample.",
            "Positive-frequency continuum plus an analytic isolated delta pole when F0>3r.",
            "c_scale is a spectral centroid/scaling stiffness, not a pole speed or front velocity.",
            "No collisions, pairing, finite-k quantum RPA, gravity/cuscuton, cyclic/Floquet or empirical claim.",
            "Densities 100 and 200, and the threshold crossing, are formal model controls, not validated nuclear matter.",
        ],
        "original_parameters": dict(collisionless.INPUTS),
        "references": list(collisionless.REFERENCES),
    }


def compute_all_controls(*, dps=80):
    """Compute all four original controls at 80/110 digits and density drift."""
    precision(dps)
    states = []
    fine_states = []
    refinements = []
    for ratio in REPRESENTATIVE_DENSITIES:
        coarse = _state(ratio, dps=80, full=False)
        fine = _state(ratio, dps=110, full=False)
        fine_states.append(fine)
        components = {}
        for key in ("F0", "r", "NF", "B", "C", "V", "first_sound_squared"):
            components[key] = _relative_real(coarse["coefficients"][key], fine["coefficients"][key])
        for power in MOMENT_ORDERS:
            components[f"m{power}"] = _relative_real(coarse["moments"]["total"][power],
                                                      fine["moments"]["total"][power])
        components["max"] = max(components.values())
        coarse_passed = coarse.get("passed") is True
        fine_passed = fine.get("passed") is True
        refinements.append({"n_ratio": ratio, "coarse_dps": 80, "fine_dps": 110,
                            "difference_components": components,
                            "max_relative_difference": components["max"],
                            "coarse_state_passed": coarse_passed,
                            "fine_state_passed": fine_passed,
                            "passed": bool(coarse_passed and fine_passed
                                           and components["max"] < RESOLVENT_TOLERANCE)})
        states.append(coarse)
    slopes = {ratio: density_slopes(ratio, dps=80) for ratio in REPRESENTATIVE_DENSITIES}
    threshold = threshold_precision_check()
    all_passed = bool(all(row["passed"] for row in states)
                      and all(row.get("passed") is True for row in fine_states)
                      and all(row["passed"] for row in refinements)
                      and all(row["passed"] for row in slopes.values())
                      and threshold["passed"])
    return {
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "mathematical_checks_passed": all_passed,
        "representative_density_ratios": list(REPRESENTATIVE_DENSITIES),
        "states": _shown(states),
        "precision_refinement": _shown(refinements),
        "density_slopes": _shown(slopes),
        "threshold": _shown(threshold),
        "symbolic_checks": symbolic_checks(),
        "scope": [
            "Four original controls 1,10,100,200 with actual equilibrium coefficients at 80/110 digits.",
            "Central log-density steps h=1e-4,h/2,h/4 plus Richardson; absent poles are null, not zero.",
            "The coefficient crossing on [100,200] is an isolated-mode continuum loss, not a thermodynamic phase transition.",
            "No target physical output table, fitted normalization, broadening, or empirical evidence is used.",
        ],
        "references": list(collisionless.REFERENCES),
        "original_parameters": dict(collisionless.INPUTS),
    }


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
        result = compute_all_controls(dps=args.dps) if args.all_controls else audit(args.n_ratio, dps=args.dps)
        result = _shown(result)
        result["mathematical_checks_passed"] = bool(result.get("mathematical_checks_passed") is True)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["mathematical_checks_passed"] else 1
    except (ValueError, ArithmeticError, RuntimeError, ZeroDivisionError, OverflowError) as exc:
        print(json.dumps({"status": "invalid_or_failed_spectral_structure_audit",
                          "evidence_weight": EVIDENCE_WEIGHT,
                          "mathematical_checks_passed": False,
                          "error": str(exc)}, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
