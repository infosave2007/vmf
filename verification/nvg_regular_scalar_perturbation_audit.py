#!/usr/bin/env python3
"""Derived scalar perturbation audit for the regular epsilon construction.

This module is an executable mathematical audit of the explicitly declared
negative-cuscuton epsilon extension.  It is not a claim about the preserved
NVG action, NVG matter, a quantum spectrum, or a nonlinear completion.  The
flat canonical-massless benchmark is reconstructed from the universal
Quintin--Yoshida coefficient expressions, including the ``-H Hddot`` sign.
The closed calculation uses the post-background-reduced Lorentzian ADM density
and parent-documented EH curvature inputs to derive the auxiliary Hessian, then
integrates a finite canonical pair.  It does not claim an independent EH
curvature derivation.

All scales in the cycle controls are manufactured dimensionless inputs.  The
command-line interface emits JSON only and does not save numerical output.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mpmath as mp
import sympy as sp


STATUS = "regular_epsilon_scalar_perturbation_counterfactual"
EVIDENCE_WEIGHT = 0
SOURCES = {
    "derivation_and_scope": "NVG_CYCLIC_PERTURBATION_CLOSURE_RU.md",
    "background_map": "verification/nvg_regular_cyclic_action_audit.py",
    "prior_flat_control": "verification/covariant_bounce_completion_audit.py",
    "flat_coefficients": "https://arxiv.org/html/1911.06040v2",
}

# These are the complete finite mode sets claimed by the default aggregate.
# A skipped group is accepted only when its scope explicitly says that the
# corresponding subset was omitted; an empty list is never silently complete.
DECLARED_FLAT_KAPPAS = ("0.1", "1", "10")
DECLARED_CLOSED_ELL = tuple(range(2, 13))

# Acceptance uses these fixed ceilings rather than trusting a producer's
# boolean ``passed`` field.  The values are deliberately the same bounds used
# by the corresponding convergence producers (with a determinant ceiling also
# applied to the coarse record).
FLAT_DETERMINANT_TOL = mp.mpf("3e-7")
FLAT_COEFFICIENT_TOL = mp.mpf("3e-12")
FLAT_REFINEMENT_TOL = mp.mpf("3e-6")
CLOSED_DETERMINANT_TOL = mp.mpf("3e-6")
CLOSED_CONSTRAINT_TOL = mp.mpf("3e-8")
CLOSED_BACKGROUND_TOL = mp.mpf("3e-8")
CLOSED_REFINEMENT_TOL = mp.mpf("3e-5")
GROWTH_RESIDUAL_TOL = mp.mpf("3e-4")
MATRIX_ENTRY_ABS_MAX = mp.mpf("1e12")
POINT_RESIDUAL_TOL = mp.mpf("1e-60")

FLAT_TRANSFER_SCOPE = (
    "complete declared finite transfer set kappa={0.1,1,10}; "
    "raw canonical pair; fixed-vs-fine refinement")
FLAT_TRANSFER_OMITTED_SCOPE = (
    "omitted by explicit include_transfer=false; no finite transfer claim")
CLOSED_TRANSFER_SCOPE = (
    "complete declared finite S3 transfer set ell=2..12; "
    "raw canonical one-cycle pair; finite Floquet sample only")
CLOSED_TRANSFER_OMITTED_SCOPE = (
    "omitted by explicit include_closed=false; no finite Floquet claim")
CURVED_EH_PROVENANCE = (
    "EH extrinsic/spatial curvature and integrated S3 shift identity are "
    "analytical inputs from NVG_CYCLIC_PERTURBATION_CLOSURE_RU.md section 6; this module checks their "
    "post-background-reduced ADM regrouping and Schur reduction, not an "
    "independent EH derivation")


REQUIRED_FLAT_ROWS = frozenset({
    "HHddot_from_continuity",
    "A2_from_general_coefficient",
    "A0_from_general_coefficient",
    "B2_from_general_coefficient",
    "B0_from_general_coefficient",
    "flat_N_from_general_coefficients",
    "flat_D_from_general_coefficients",
    "kinetic_from_general_coefficient",
    "epsilon_zero_frozen_N",
    "epsilon_zero_frozen_D",
    "epsilon_zero_frozen_kinetic",
    "lower_bound_decomposition",
    "upper_bound_decomposition",
    "F_lower_identity",
    "low_density_fixed_K_path",
    "low_density_proportional_path",
})
REQUIRED_CURVED_ROWS = frozenset({
    "eh_extrinsic_terms",
    "matter_canonical_lapse_term",
    "matter_lapse_series",
    "matter_lapse_linear_source",
    "matter_lapse_quadratic_source",
    "shift_square_curvature_term",
    "shift_square_curvature_identity",
    "cuscuton_square_root_series",
    "cuscuton_has_no_sdot2",
    "cuscuton_lapse_series_cancellation",
    "cuscuton_spatial_gradient_term",
    "cuscuton_potential_curvature_term",
    "cuscuton_potential_gradient_cancellation",
    "cuscuton_cross_after_parts",
    "assembled_quadratic_density",
    "actual_hessian_J",
    "actual_velocity_vector_u",
    "actual_coordinate_vector_v",
    "auxiliary_solution_residual",
    "determinant_true_scale",
    "A_s_candidate",
    "B_s_candidate",
    "C_s_candidate",
})
REQUIRED_FLAT_RECONSTRUCTION_ROWS = frozenset({
    "C0_A_s_is_QY_kinetic",
    "C0_G_includes_pdot_and_Ydot",
    "C0_G_over_pA_is_QY_sound_speed",
    "C0_minus_HHddot_sign",
})


def finite(value: Any, name: str, *, positive: bool = False,
           nonnegative: bool = False) -> mp.mpf:
    """Parse a finite real number without accepting booleans or NaNs."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        result = mp.mpf(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not mp.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_float(value: Any, name: str, *, positive: bool = False,
                  nonnegative: bool = False) -> float:
    parsed = finite(value, name, positive=positive, nonnegative=nonnegative)
    result = float(parsed)
    if not math.isfinite(result):
        raise ValueError(f"{name} is outside finite float range")
    return result


def decimal(value: Any) -> str:
    return mp.nstr(finite(value, "derived value"), 35)


def _json_values(value: Any) -> Any:
    if isinstance(value, mp.mpf):
        return decimal(value)
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ArithmeticError("nonfinite JSON output")
    return value


def _safe_finite(value: Any) -> Optional[mp.mpf]:
    if isinstance(value, bool):
        return None
    try:
        number = mp.mpf(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if mp.isfinite(number) else None


def _reported_matches(actual: Any, expected: Any,
                      *, relative: Any = "1e-18") -> bool:
    a = _safe_finite(actual)
    b = _safe_finite(expected)
    if a is None or b is None:
        return False
    scale = max(mp.mpf("1e-120"), abs(a), abs(b))
    return bool(abs(a - b) <= mp.mpf(relative) * scale)


def _rows(expressions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for name, expression in expressions.items():
        reduced = sp.factor(sp.trigsimp(sp.simplify(expression)))
        result[name] = {"residual": str(reduced),
                        "passed": bool(reduced == 0)}
    return result


def rows_pass(rows: Any, required_names: Optional[Iterable[str]] = None) -> bool:
    """Fail closed on missing rows, schema drift, or nonzero residuals."""
    if not isinstance(rows, dict) or not rows:
        return False
    if required_names is not None and set(rows) != set(required_names):
        return False
    return all(
        isinstance(row, dict)
        and set(row) == {"residual", "passed"}
        and isinstance(row["residual"], str)
        and isinstance(row["passed"], bool)
        and row["passed"] is True
        and row["residual"] == "0"
        for row in rows.values()
    )


# ---------------------------------------------------------------------------
# Flat canonical massless benchmark


def _flat_source_expressions(epsilon: Any = None,
                             *, h_hddot_sign: int = -1):
    """Return the source coefficient algebra in dimensionless variables.

    ``epsilon=None`` leaves epsilon symbolic.  The source expressions are
    written once here; the simplified polynomial is constructed separately in
    :func:`flat_coefficients` so the two paths can be compared.
    """
    if isinstance(h_hddot_sign, bool) or h_hddot_sign not in (-1, 1):
        raise ValueError("h_hddot_sign must be -1 or +1")
    x, e, K = sp.symbols("x epsilon K", positive=True)
    A = x * (1 - x) / 3
    hd = x * (2 * x - 1)
    # xdot=-6 H x and H^2/L=A imply H Hddot/L^2=-6 A x(4x-1).
    hhddot = -6 * A * x * (4 * x - 1)
    Y = (1 + e) * x
    A2 = Y * (12 * A + 3 * hd + Y) + 2 * hd ** 2 \
        + h_hddot_sign * hhddot
    A0 = Y ** 2 * (15 * A + hd - Y) \
        - Y * (12 * A * hd - 2 * hd ** 2 + 3 * hhddot)
    B2 = Y * (6 * A + hd + Y)
    B0 = 3 * Y ** 2 * (3 * A + hd + Y)
    N = sp.expand(A * K ** 2 + A2 * K + A0)
    D = sp.expand(A * K ** 2 + B2 * K + B0)
    kinetic = Y * (K + 3 * Y) / (K * A + Y * (3 * A + hd + Y))
    return {"x": x, "e": e, "K": K, "A": A, "hd": hd,
            "hhddot": hhddot, "Y": Y, "A2": A2, "A0": A0,
            "B2": B2, "B0": B0, "N": N, "D": D,
            "kinetic": kinetic}


def flat_qy_symbolic_checks(*, h_hddot_sign: int = -1) \
        -> Dict[str, Dict[str, Any]]:
    """Independently reconstruct the flat QY polynomials and sign controls."""
    expr = _flat_source_expressions(h_hddot_sign=h_hddot_sign)
    x, e, K = expr["x"], expr["e"], expr["K"]
    A, hd, Y, hhddot = expr["A"], expr["hd"], expr["Y"], expr["hhddot"]
    N, D, kinetic = expr["N"], expr["D"], expr["kinetic"]
    c = 1 + e
    F = 3 + 2 * e - e ** 2 + (7 - 3 * e) * x - 8 * x ** 2
    expected_N = A * K ** 2 + x ** 2 * (e + 2) * (e + 2 * x + 1) * K \
        + c * x ** 3 * F
    expected_D = A * K ** 2 + x ** 2 * c * (e + 2) * K \
        + 3 * c ** 2 * x ** 3 * (e + x + 1)
    expected_kinetic = c * x * (K + 3 * c * x) \
        / (K * A + c * x * (3 * A + hd + c * x))

    lower = sp.expand(3 * c * N - (1 - e) * D)
    lower_decomposition = (
        -2 * x * (2 * e + 1) * (x - 1) * K ** 2 / 3
        + 2 * x ** 2 * c * (e + 2) * (2 * e + 3 * x + 1) * K
        - 6 * x ** 3 * c ** 2 * (x - 1) * (e + 4 * x + 1))
    upper = sp.expand((e + 3) * D - c * N)
    upper_decomposition = (
        -2 * x * (x - 1) * K ** 2 / 3
        - 2 * x ** 2 * c * (e + 2) * (x - 1) * K
        + 2 * x ** 3 * c ** 2
        * (2 * e ** 2 + 3 * e * x + 5 * e + 4 * x ** 2 + x + 3))
    r = sp.symbols("ratio", positive=True)
    # The proportional path is the physically distinct K/x=constant path;
    # fixed K>0 is the gradient-dominated path.  A zero-K limit is reported
    # numerically below as an additional control.
    fixed_limit = sp.limit(N / D, x, 0, dir="+")
    proportional_limit = sp.limit((N / D).subs(K, r * x), x, 0, dir="+")
    rows = {
        "HHddot_from_continuity": hhddot + 6 * A * x * (4 * x - 1),
        "A2_from_general_coefficient": expr["A2"]
        - x ** 2 * (e + 2) * (e + 2 * x + 1),
        "A0_from_general_coefficient": expr["A0"] - c * x ** 3 * F,
        "B2_from_general_coefficient": expr["B2"]
        - x ** 2 * c * (e + 2),
        "B0_from_general_coefficient": expr["B0"]
        - 3 * c ** 2 * x ** 3 * (e + x + 1),
        "flat_N_from_general_coefficients": N - expected_N,
        "flat_D_from_general_coefficients": D - expected_D,
        "kinetic_from_general_coefficient": kinetic - expected_kinetic,
        "epsilon_zero_frozen_N": expected_N.subs(e, 0)
        - (A * K ** 2 + 2 * x ** 2 * (2 * x + 1) * K
           + x ** 3 * (3 + 7 * x - 8 * x ** 2)),
        "epsilon_zero_frozen_D": expected_D.subs(e, 0)
        - (A * K ** 2 + 2 * x ** 2 * K + 3 * x ** 3 * (x + 1)),
        "epsilon_zero_frozen_kinetic": expected_kinetic.subs(e, 0)
        - 3 * (K + 3 * x) / (K * (1 - x) + 3 * x * (x + 1)),
        "lower_bound_decomposition": lower - lower_decomposition,
        "upper_bound_decomposition": upper - upper_decomposition,
        "F_lower_identity": F - (1 - e) * (e + x + 1)
        - 2 * (1 - x) * (4 * x + e + 1),
        "low_density_fixed_K_path": fixed_limit - 1,
        "low_density_proportional_path": proportional_limit
        - (r ** 2 / 3 + r * c * (e + 2) + c ** 2 * (3 - e))
        / (r ** 2 / 3 + r * c * (e + 2) + 3 * c ** 3),
    }
    return _rows(rows)


def flat_general_coefficients(epsilon: Any, x: Any, K: Any) \
        -> Dict[str, mp.mpf]:
    """Evaluate the unsimplified QY coefficient expressions independently."""
    e = finite(epsilon, "epsilon", nonnegative=True)
    xv = finite(x, "x", positive=True)
    kv = finite(K, "K", nonnegative=True)
    if xv > 1:
        raise ValueError("x must lie in (0,1]")
    A = xv * (1 - xv) / 3
    hd = xv * (2 * xv - 1)
    hhddot = -6 * A * xv * (4 * xv - 1)
    Y = (1 + e) * xv
    A2 = Y * (12 * A + 3 * hd + Y) + 2 * hd ** 2 - hhddot
    A0 = Y ** 2 * (15 * A + hd - Y) \
        - Y * (12 * A * hd - 2 * hd ** 2 + 3 * hhddot)
    B2 = Y * (6 * A + hd + Y)
    B0 = 3 * Y ** 2 * (3 * A + hd + Y)
    N = A * kv ** 2 + A2 * kv + A0
    D = A * kv ** 2 + B2 * kv + B0
    if D <= 0:
        raise ArithmeticError("flat coefficient denominator is not positive")
    qkin = Y * (kv + 3 * Y) / (kv * A + Y * (3 * A + hd + Y))
    return {"A": A, "hd": hd, "hhddot": hhddot, "Y": Y,
            "A2": A2, "A0": A0, "B2": B2, "B0": B0,
            "N": N, "D": D, "kinetic": qkin, "cs2": N / D}


def flat_coefficients(epsilon: Any, x: Any, K: Any) -> Dict[str, mp.mpf]:
    """Evaluate the simplified epsilon polynomials, preserving negative cs2."""
    e = finite(epsilon, "epsilon", nonnegative=True)
    xv = finite(x, "x", positive=True)
    kv = finite(K, "K", nonnegative=True)
    if xv > 1:
        raise ValueError("x must lie in (0,1]")
    A = xv * (1 - xv) / 3
    c = 1 + e
    F = 3 + 2 * e - e ** 2 + (7 - 3 * e) * xv - 8 * xv ** 2
    N = A * kv ** 2 + xv ** 2 * (e + 2) * (e + 2 * xv + 1) * kv \
        + c * xv ** 3 * F
    D = A * kv ** 2 + xv ** 2 * c * (e + 2) * kv \
        + 3 * c ** 2 * xv ** 3 * (e + xv + 1)
    if D <= 0:
        raise ArithmeticError("flat coefficient denominator is not positive")
    kinetic = c * xv * (kv + 3 * c * xv) \
        / (kv * A + c * xv * (3 * A + xv * (2 * xv - 1) + c * xv))
    return {"epsilon": e, "x": xv, "K": kv, "A": A, "F": F,
            "N": N, "D": D, "kinetic": kinetic, "cs2": N / D,
            "positive_kinetic": bool(kinetic > 0),
            "superluminal_phase_speed": bool(N / D > 1)}


def scalar_massless_coefficients(x: Any, K: Any, *, epsilon: Any = "0") \
        -> Dict[str, mp.mpf]:
    """Frozen-benchmark spelling; epsilon is opt-in for compatibility."""
    return flat_coefficients(epsilon, x, K)


def flat_bound_certificate(epsilon: Any = "0.1") -> Dict[str, Any]:
    """Return the analytic sign regime and an explicit epsilon>1 counterexample."""
    e = finite(epsilon, "epsilon", nonnegative=True)
    c = 1 + e
    out: Dict[str, Any] = {
        "epsilon": e,
        "kinetic_denominator_form": "K*A+(1+epsilon)*x^2*(x+1+epsilon)",
        "kinetic_strictly_positive_for_x_gt_0": True,
        "lower_bound": (1 - e) / (3 * c),
        "upper_bound": (e + 3) / c,
        "analytic_domain": "0<epsilon<1, 0<x<=1, K>=0",
        "proof_is_polynomial_not_scan": True,
    }
    if e == 0:
        out.update({"regime": "frozen_epsilon_zero_control",
                    "certificate_passed": True,
                    "strict_all_domain": True})
    elif e < 1:
        out.update({"regime": "strict_positive_sign_bounds",
                    "certificate_passed": True,
                    "strict_all_domain": True})
    elif e == 1:
        marginal = flat_coefficients(e, 1, 0)
        out.update({"regime": "marginal_epsilon_one",
                    "certificate_passed": True,
                    "strict_all_domain": False,
                    "marginal_point": marginal,
                    "marginal_reason": "cs2=0 at x=1,K=0"})
    else:
        threshold = (e ** 2 - 1) / (e + 3)
        Kcounter = threshold / 2
        counter = flat_coefficients(e, 1, Kcounter)
        out.update({"regime": "negative_long_wavelength_coefficient",
                    "certificate_passed": True,
                    "strict_all_domain": False,
                    "counterexample": {
                        "x": mp.mpf(1), "K": Kcounter,
                        "threshold_K": threshold, "point": counter,
                        "negative_cs2": bool(counter["cs2"] < 0),
                        "interpretation": "long-wavelength squared coefficient; not a UV or late-time theorem",
                    }})
    return out


def low_density_paths(epsilon: Any = "0.1", ratio: Any = "1") -> Dict[str, Any]:
    """Compare distinct x->0 limits without imposing a bare-GR value."""
    e = finite(epsilon, "epsilon", nonnegative=True)
    r = finite(ratio, "ratio", nonnegative=True)
    c = 1 + e
    proportional = (r ** 2 / 3 + r * c * (e + 2) + c ** 2 * (3 - e)) \
        / (r ** 2 / 3 + r * c * (e + 2) + 3 * c ** 3)
    zero = (3 - e) / (3 * c)
    return {
        "epsilon": e, "K_over_x": r,
        "fixed_K_positive_limit": mp.mpf(1),
        "K_zero_limit": zero,
        "K_proportional_limit": proportional,
        "bare_GR_reference": mp.mpf(1),
        "finite_epsilon_proportional_differs_from_bare_GR": bool(
            e > 0 and abs(proportional - 1) > mp.mpf("1e-40")),
        "paths_are_distinct": bool(abs(proportional - 1) > mp.mpf("1e-40")
                                    or abs(zero - 1) > mp.mpf("1e-40")),
        "scope": "two low-density scalings of the extension; no forced finite-epsilon GR limit",
    }


# Explicit descriptive alias for callers that want the path audit by name.
low_density_path_checks = low_density_paths


def flat_scalar_transfer(kappa: Any, *, epsilon: Any = "0.1",
                         tau_start: Any = -5, tau_end: Any = 5,
                         rtol: Any = "1e-10", atol: Any = "1e-12",
                         max_step: Any = "0.04") -> Dict[str, Any]:
    """Integrate the actual flat canonical pair through a finite bounce."""
    import numpy as np
    from scipy.integrate import solve_ivp

    kap = _finite_float(kappa, "kappa", nonnegative=True)
    e = _finite_float(epsilon, "epsilon", nonnegative=True)
    t0 = _finite_float(tau_start, "tau_start")
    t1 = _finite_float(tau_end, "tau_end")
    rr = _finite_float(rtol, "rtol", positive=True)
    aa = _finite_float(atol, "atol", positive=True)
    step = _finite_float(max_step, "max_step", positive=True)
    if t0 >= t1:
        raise ValueError("tau_start must precede tau_end")
    if rr < 3e-14 or aa < 1e-300:
        raise ValueError("solver tolerance is outside the supported float range")
    calls = 0
    max_formula_residual = 0.0

    def rhs(tau: float, state: np.ndarray) -> List[float]:
        nonlocal calls, max_formula_residual
        calls += 1
        if calls > 300000:
            raise ArithmeticError("flat transfer exceeded bounded evaluation budget")
        x = 1.0 / (1.0 + tau * tau)
        a = (1.0 + tau * tau) ** (1.0 / 6.0)
        K = (kap / a) ** 2
        poly = flat_coefficients(e, x, K)
        general = flat_general_coefficients(e, x, K)
        local = max(abs(float(poly[name] - general[name]))
                    for name in ("N", "D", "kinetic", "cs2"))
        max_formula_residual = max(max_formula_residual, local)
        b = a ** 3 * float(poly["kinetic"])
        omega2 = float(poly["cs2"]) * K / 3.0
        result = [float(state[1] / b), float(-b * omega2 * state[0]),
                  float(state[3] / b), float(-b * omega2 * state[2])]
        if not all(math.isfinite(v) for v in result):
            raise ArithmeticError("nonfinite flat perturbation evolution")
        return result

    try:
        sol = solve_ivp(rhs, (t0, t1), (1.0, 0.0, 0.0, 1.0),
                        method="DOP853", rtol=rr, atol=aa, max_step=step)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("transfer inputs exceed finite float arithmetic") from exc
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise ArithmeticError("flat finite-interval integration failed")
    z1, p1, z2, p2 = (float(v) for v in sol.y[:, -1])
    matrix = [[z1, z2], [p1, p2]]
    determinant = z1 * p2 - z2 * p1
    if not math.isfinite(determinant):
        raise ArithmeticError("nonfinite flat transfer determinant")
    return {
        "kappa": kap, "epsilon": e, "interval": [t0, t1],
        "matrix": matrix, "determinant": determinant,
        "determinant_error": abs(determinant - 1.0),
        "symplectic_error": abs(determinant - 1.0),
        "coefficient_formula_max_abs": max_formula_residual,
        "rhs_evaluations": int(sol.nfev), "finite": True,
        "scope": "flat canonical massless scalar pair across finite bounce only",
    }


def flat_transfer_convergence(kappa: Any, *, epsilon: Any = "0.1") \
        -> Dict[str, Any]:
    coarse = flat_scalar_transfer(kappa, epsilon=epsilon, rtol="2e-8",
                                  atol="2e-10", max_step="0.15")
    fine = flat_scalar_transfer(kappa, epsilon=epsilon, rtol="2e-10",
                                atol="2e-12", max_step="0.04")
    diff = max(abs(coarse["matrix"][i][j] - fine["matrix"][i][j])
               / max(1.0, abs(fine["matrix"][i][j]))
               for i in range(2) for j in range(2))
    return {"kappa": coarse["kappa"], "epsilon": coarse["epsilon"],
            "coarse": coarse, "fine": fine,
            "relative_matrix_difference": diff,
            "passed": bool(diff < float(FLAT_REFINEMENT_TOL)
                            and fine["determinant_error"]
                            < float(FLAT_DETERMINANT_TOL)
                            and fine["coefficient_formula_max_abs"]
                            < float(FLAT_COEFFICIENT_TOL))}


# ---------------------------------------------------------------------------
# Curved ADM scalar density and Schur reduction


def curved_quadratic_density(*, shift_curvature_sign: int = -1,
                             cuscuton_gradient_sign: int = 1,
                             cross_sign: int = 1,
                             cuscuton_hessian_sign: int = 1):
    """Assemble the post-background-reduced Lorentzian ADM density.

    The EH extrinsic/spatial pieces and the integrated S3 shift identity are
    analytical inputs documented in ``NVG_CYCLIC_PERTURBATION_CLOSURE_RU.md``
    section 6; this function
    checks their declared regrouping and Schur reduction, not an independent
    EH curvature derivation.  The negative-cuscuton square-root and canonical
    matter lapse/source pieces are expanded independently below in
    :func:`curved_action_symbolic_checks`.
    """
    for name, value in (("shift_curvature_sign", shift_curvature_sign),
                        ("cuscuton_gradient_sign", cuscuton_gradient_sign),
                        ("cross_sign", cross_sign),
                        ("cuscuton_hessian_sign", cuscuton_hessian_sign)):
        if isinstance(value, bool) or value not in (-1, 1):
            raise ValueError(f"{name} must be -1 or +1")
    H, p, C, Y, D, U = sp.symbols("H p C Y D_H U", real=True)
    z, zd, ell, beta, s, sdot = sp.symbols(
        "zeta zdot ell beta s sdot", real=True)
    d = p - 3 * C
    # The trace part is -3(zdot-H*ell)^2; the scalar shift contraction is
    # -2 p beta(zdot-H*ell).  Expanding these ADM invariants gives the
    # Lorentzian time signs below without importing a Euclidean action.
    eh_extrinsic = (-3 * (zd - H * ell) ** 2
                    - 2 * p * beta * (zd - H * ell))
    # On a unit S3 harmonic, the integrated Hessian identity is
    # int[(D_iD_j beta)^2-(Delta beta)^2] = -2 C p int[beta^2].
    # The ADM shift-square carries one half of this combination.
    hess_shift_integral = -2 * C * p * beta ** 2
    eh_shift_curvature = -shift_curvature_sign * hess_shift_integral / 2
    eh_spatial_curvature = p * z ** 2 - 3 * C * z ** 2 \
        + 2 * p * ell * z - 6 * C * ell * z
    matter = Y * ell ** 2
    cuscuton_cross = cross_sign * s * (3 * zd - 3 * H * ell + p * beta)
    cuscuton_spatial = cuscuton_gradient_sign * p * s ** 2 / (4 * U)
    cuscuton_potential = -3 * cuscuton_hessian_sign * D * s ** 2 / (4 * U)
    cuscuton = cuscuton_cross + cuscuton_spatial + cuscuton_potential
    density = sp.expand(eh_extrinsic + eh_shift_curvature
                        + eh_spatial_curvature + matter + cuscuton)
    expected = (-3 * zd ** 2 + (Y - 3 * H ** 2) * ell ** 2
                + 6 * H * ell * zd - 2 * p * beta * (zd - H * ell)
                - C * p * beta ** 2 + d * z ** 2 + 2 * d * ell * z
                + s * (3 * zd - 3 * H * ell + p * beta)
                + (p - 3 * D) * s ** 2 / (4 * U))
    return {"H": H, "p": p, "C": C, "Y": Y, "D": D, "U": U,
            "z": z, "zd": zd, "ell": ell, "beta": beta, "s": s,
            "sdot": sdot, "d": d, "eh_extrinsic": eh_extrinsic,
            "hess_shift_integral": hess_shift_integral,
            "eh_shift_curvature": eh_shift_curvature,
            "eh_spatial_curvature": eh_spatial_curvature,
            "matter": matter, "cuscuton_cross": cuscuton_cross,
            "cuscuton_spatial": cuscuton_spatial,
            "cuscuton_potential": cuscuton_potential,
            "cuscuton": cuscuton, "provenance": CURVED_EH_PROVENANCE,
            "density": density, "expected": expected}


def curved_action_symbolic_checks(*, shift_curvature_sign: int = -1,
                                  cuscuton_gradient_sign: int = 1,
                                  cross_sign: int = 1,
                                  cuscuton_hessian_sign: int = 1) \
        -> Dict[str, Dict[str, Any]]:
    """Derive and Schur-reduce the curved scalar action symbolically."""
    q = curved_quadratic_density(
        shift_curvature_sign=shift_curvature_sign,
        cuscuton_gradient_sign=cuscuton_gradient_sign,
        cross_sign=cross_sign,
        cuscuton_hessian_sign=cuscuton_hessian_sign)
    H, p, C, Y, D, U = (q[key] for key in ("H", "p", "C", "Y", "D", "U"))
    z, zd, ell, beta, s, sdot = (q[key] for key in
                                  ("z", "zd", "ell", "beta", "s", "sdot"))
    d = q["d"]
    density = q["density"]
    expected = q["expected"]
    Vpsi = 3 * cross_sign * H
    # Independent short series controls.  For a positive background
    # cuscuton velocity 2U, the negative square root is expanded on the
    # physical branch.  Keeping the lapse in the expression exposes its
    # exact cancellation and tests that no sdot**2 term is manufactured.
    t = sp.symbols("series_t", real=True)
    Upos = sp.symbols("Upos", positive=True)
    ell_series, sdot_series, grad_series = sp.symbols(
        "ell_series sdot_series grad_series", real=True)
    cusp_speed = 2 * Upos + t * sdot_series
    cusp_root = cusp_speed * sp.sqrt(
        1 - t ** 2 * grad_series / cusp_speed ** 2)
    cusp_lapse_density = -(1 + t * ell_series) * cusp_root \
        / (1 + t * ell_series)
    cusp_series = sp.expand(sp.series(cusp_lapse_density, t, 0, 3)
                            .removeO())
    cusp_expected_series = (-2 * Upos - t * sdot_series
                            + t ** 2 * grad_series / (4 * Upos))
    Ypos = sp.symbols("Ypos", positive=True)
    matter_series_ell = sp.symbols("ell_matter", real=True)
    matter_lapse_series = sp.expand(
        sp.series(Ypos / (1 + t * matter_series_ell), t, 0, 3)
        .removeO())
    matter_expected_series = (Ypos - t * Ypos * matter_series_ell
                              + t ** 2 * Ypos * matter_series_ell ** 2)
    # The term -3 zeta delta(Psi_dot) integrates to +3 zdot*s+9H*zeta*s;
    # the potential cross term is -3 V_Psi zeta*s and cancels when V_Psi=3H.
    parts_rows = {
        "eh_extrinsic_terms": q["eh_extrinsic"]
        - (-3 * zd ** 2 - 3 * H ** 2 * ell ** 2 + 6 * H * ell * zd
           - 2 * p * beta * (zd - H * ell)),
        "matter_canonical_lapse_term": q["matter"] - Y * ell ** 2,
        "matter_lapse_series": matter_lapse_series - matter_expected_series,
        "matter_lapse_linear_source":
            sp.expand(matter_lapse_series).coeff(t, 1)
            + Ypos * matter_series_ell,
        "matter_lapse_quadratic_source":
            sp.expand(matter_lapse_series).coeff(t, 2)
            - Ypos * matter_series_ell ** 2,
        "shift_square_curvature_term": sp.expand(density).coeff(beta, 2)
        + C * p,
        "shift_square_curvature_identity": (
            q["hess_shift_integral"] + 2 * C * p * beta ** 2),
        "cuscuton_square_root_series": cusp_series - cusp_expected_series,
        "cuscuton_has_no_sdot2": sp.expand(cusp_series).coeff(sdot_series, 2),
        "cuscuton_lapse_series_cancellation":
            sp.diff(cusp_series, ell_series),
        "cuscuton_spatial_gradient_term":
            q["cuscuton_spatial"].coeff(s, 2) - p / (4 * U),
        "cuscuton_potential_curvature_term": (
            q["cuscuton_potential"].coeff(s, 2)
            + 3 * D / (4 * U)),
        "cuscuton_potential_gradient_cancellation": 9 * H - 3 * Vpsi,
        "cuscuton_cross_after_parts": (
            cross_sign * 3 * zd * s + 9 * H * z * s
            - 3 * Vpsi * z * s
            - 3 * zd * s),
        "assembled_quadratic_density": density - expected,
    }

    auxiliaries = sp.Matrix([ell, beta, s])
    J = sp.hessian(density, auxiliaries)
    u = sp.Matrix([sp.diff(sp.diff(density, item), zd)
                   for item in auxiliaries])
    v = sp.Matrix([sp.diff(sp.diff(density, item), z)
                   for item in auxiliaries])
    J_expected = sp.Matrix([[2 * (Y - 3 * H ** 2), 2 * H * p, -3 * H],
                            [2 * H * p, -2 * C * p, p],
                            [-3 * H, p, (p - 3 * D) / (2 * U)]])
    u_expected = sp.Matrix([6 * H, -2 * p, 3])
    v_expected = sp.Matrix([2 * d, 0, 0])

    def matrix_residual(matrix):
        """Turn an exact-zero matrix residual into one scalar row."""
        return sum(item ** 2 for item in matrix)

    parts_rows.update({
        "actual_hessian_J": matrix_residual(J - J_expected),
        "actual_velocity_vector_u": matrix_residual(u - u_expected),
        "actual_coordinate_vector_v": matrix_residual(v - v_expected),
    })

    # The candidate formulas require the background constraint
    # U=Y+D_H-C.  Keep U independent during variation, impose it only in the
    # comparison, and never divide by H.
    relation = {U: Y + D - C}
    source = -3 * zd ** 2 + d * z ** 2 \
        + zd * (u.T * auxiliaries)[0] + z * (v.T * auxiliaries)[0] \
        + (auxiliaries.T * J * auxiliaries)[0] / 2
    solved = -J.inv() * (u * zd + v * z)
    solved_residual = J * solved + u * zd + v * z
    Delta = (d + 3 * Y) * (d * H ** 2 + C * Y) + d * Y * U
    A_candidate = Y * d * (d + 3 * Y) / Delta
    B_candidate = 2 * H * d ** 2 * (d + 3 * Y) / Delta
    C_candidate = d - d ** 2 * (Y * p + D * d) / Delta
    A_actual = -3 - (u.T * J.inv() * u)[0] / 2
    B_actual = -(u.T * J.inv() * v)[0]
    C_actual = d - (v.T * J.inv() * v)[0] / 2
    det_row = J.det() + 2 * p * Delta / U
    parts_rows.update({
        "auxiliary_solution_residual": matrix_residual(solved_residual),
        "determinant_true_scale": det_row.subs(relation),
        "A_s_candidate": (A_actual - A_candidate).subs(relation),
        "B_s_candidate": (B_actual - B_candidate).subs(relation),
        "C_s_candidate": (C_actual - C_candidate).subs(relation),
    })
    return _rows({name: value for name, value in parts_rows.items()})


def curved_coefficient_point(*, H: Any = "0.37", p: Any = "8", C: Any = "1",
                             Y: Any = "3", D_H: Any = "0.5",
                             U: Optional[Any] = None) -> Dict[str, Any]:
    """Compare direct numerical auxiliary elimination with the candidate."""
    with mp.workdps(90):
        hv = finite(H, "H")
        pv = finite(p, "p", positive=True)
        cv = finite(C, "C", nonnegative=True)
        yv = finite(Y, "Y", positive=True)
        dv = finite(D_H, "D_H")
        uv = finite(yv + dv - cv if U is None else U, "U", positive=True)
        if pv <= 3 * cv:
            raise ValueError("physical inhomogeneous mode requires p>3C")
        d = pv - 3 * cv
        J = mp.matrix([[2 * (yv - 3 * hv ** 2), 2 * hv * pv, -3 * hv],
                       [2 * hv * pv, -2 * cv * pv, pv],
                       [-3 * hv, pv, (pv - 3 * dv) / (2 * uv)]])
        u = mp.matrix([6 * hv, -2 * pv, 3])
        v = mp.matrix([2 * d, 0, 0])
        inverse = J ** -1
        direct_A = -3 - (u.T * inverse * u)[0] / 2
        direct_B = -(u.T * inverse * v)[0]
        direct_C = d - (v.T * inverse * v)[0] / 2
        delta = (d + 3 * yv) * (d * hv ** 2 + cv * yv) + d * yv * uv
        cand_A = yv * d * (d + 3 * yv) / delta
        cand_B = 2 * hv * d ** 2 * (d + 3 * yv) / delta
        cand_C = d - d ** 2 * (yv * pv + dv * d) / delta
        det_direct = mp.det(J)
        det_formula = -2 * pv * delta / uv
        # A nonzero test displacement checks the actual lapse/shift/cuscuton
        # stationarity equations, independently of the closed-form Schur
        # coefficients.
        test_z, test_zdot = mp.mpf("0.41"), mp.mpf("-0.27")
        solved_auxiliary = -inverse * (u * test_zdot + v * test_z)
        auxiliary_residual = J * solved_auxiliary + u * test_zdot + v * test_z
        residuals = {
            "A_s": direct_A - cand_A, "B_s": direct_B - cand_B,
            "C_s": direct_C - cand_C,
            "determinant": det_direct - det_formula,
            "auxiliary_solution": max(abs(value)
                                       for value in auxiliary_residual),
        }
        scale = max(mp.mpf(1), *(abs(item) for item in residuals.values()))
        scaled = {name: abs(value) / scale
                  for name, value in residuals.items()}
        max_scaled = max(scaled.values())
        return {
            "H": hv, "p": pv, "C": cv, "Y": yv, "D_H": dv, "U": uv,
            "d": d, "Delta": delta, "A_s": direct_A, "B_s": direct_B,
            "C_s": direct_C, "determinant": det_direct,
            "determinant_formula": det_formula,
            "residuals": residuals, "max_scaled_residual": max_scaled,
            "passed": bool(max_scaled < mp.mpf("1e-70")),
            "finite": True,
            "scope": "physical scalar harmonic p>3C; l=0,1 excluded",
        }


def curved_sign_certificate(*, p: Any = "8", C: Any = "1", Y: Any = "3",
                            H: Any = "0", D_H: Any = "0.5",
                            U: Optional[Any] = None) -> Dict[str, Any]:
    """Expose the analytic positivity factors of Delta and A_s."""
    pv = finite(p, "p", positive=True)
    cv = finite(C, "C", nonnegative=True)
    yv = finite(Y, "Y", positive=True)
    hv = finite(H, "H")
    dv = finite(D_H, "D_H")
    uv = finite(yv + dv - cv if U is None else U, "U", positive=True)
    if pv <= 3 * cv:
        raise ValueError("certificate requires p>3C")
    d = pv - 3 * cv
    first = (d + 3 * yv) * (d * hv ** 2 + cv * yv)
    second = d * yv * uv
    delta = first + second
    numerator = yv * d * (d + 3 * yv)
    return {
        "p": pv, "C": cv, "Y": yv, "H": hv, "D_H": dv, "U": uv,
        "d": d, "Delta_first_nonnegative": first,
        "Delta_second_strictly_positive": second, "Delta": delta,
        "A_s_numerator": numerator, "Delta_positive": bool(delta > 0),
        "A_s_positive": bool(numerator / delta > 0),
        "determinant_formula": -2 * pv * delta / uv,
        "determinant_negative": bool(-2 * pv * delta / uv < 0),
        "scope": "analytic sign certificate for C>=0,p>3C,Y>0,U>0, including H=0",
    }


def curved_flat_reconstruction_symbolic_checks(*, include_Bdot: bool = True,
                                               h_hddot_sign: int = -1) \
        -> Dict[str, Dict[str, Any]]:
    """Use the curved formulas at C=0 and recover the flat QY coefficient.

    This calculation explicitly differentiates p and Y along the background;
    it therefore tests the ``Bdot`` integration-by-parts term rather than
    comparing only the instantaneous Schur coefficient.
    """
    if not isinstance(include_Bdot, bool):
        raise ValueError("include_Bdot must be bool")
    if isinstance(h_hddot_sign, bool) or h_hddot_sign not in (-1, 1):
        raise ValueError("h_hddot_sign must be -1 or +1")
    x, e, K = sp.symbols("x epsilon K", positive=True)
    H = sp.symbols("H", positive=True)
    A = x * (1 - x) / 3
    hd = x * (2 * x - 1)
    # Keep coefficient variables independent while differentiating B.  If
    # Y/p/D/U are replaced by background expressions before ``diff``, SymPy
    # quite correctly refuses to differentiate with respect to a non-symbol.
    ys, ps, ds, us = sp.symbols("Y p D_H U", positive=True)
    d = ps
    delta = (d + 3 * ys) * (d * H ** 2) + d * ys * us
    As = ys * d * (d + 3 * ys) / delta
    B = 2 * H * d ** 2 * (d + 3 * ys) / delta
    Cs = d - d ** 2 * (ys * ps + ds * d) / delta
    Hdot = hd
    p_dot = -2 * H * ps
    Y_dot = -6 * H * ys
    # Ddot is the actual background derivative.  The argument
    # ``h_hddot_sign`` is reserved for the independent source-sign control
    # below; it must not silently mutate the background trajectory.
    D_dot = (-6 * A * x * (4 * x - 1)) / H
    U_dot = Y_dot + D_dot
    Bdot = (sp.diff(B, H) * Hdot + sp.diff(B, ps) * p_dot
            + sp.diff(B, ys) * Y_dot + sp.diff(B, us) * U_dot)
    if not include_Bdot:
        Bdot = 0
    G = (Bdot + 3 * H * B) / 2 - Cs
    N = A * K ** 2 + x ** 2 * (e + 2) * (e + 2 * x + 1) * K \
        + (1 + e) * x ** 3 * (3 + 2 * e - e ** 2
                               + (7 - 3 * e) * x - 8 * x ** 2)
    Dden = A * K ** 2 + x ** 2 * (1 + e) * (e + 2) * K \
        + 3 * (1 + e) ** 2 * x ** 3 * (e + x + 1)
    Y = (1 + e) * x
    p = K
    D = hd
    U = Y + D
    qkin = Y * (K + 3 * Y) / (K * A + Y * (3 * A + hd + Y))
    substitution = {H ** 2: A, ps: p, ys: Y, ds: D, us: U}
    rows = {
        "C0_A_s_is_QY_kinetic": (As - qkin).subs(substitution),
        "C0_G_includes_pdot_and_Ydot":
            (G - K * As * N / Dden).subs(substitution),
        "C0_G_over_pA_is_QY_sound_speed":
            (G / (K * As) - N / Dden).subs(substitution),
        "C0_minus_HHddot_sign":
            h_hddot_sign * (-6 * A * x * (4 * x - 1))
            - (-1) * (-6 * A * x * (4 * x - 1)),
    }
    return _rows(rows)


# ---------------------------------------------------------------------------
# Closed S3 w=1 control


def _closed_w1_background(A: float, q: float, epsilon: float,
                          rho_c_ratio: float) -> Dict[str, float]:
    if not math.isfinite(A) or A <= 0 or not math.isfinite(q):
        raise ArithmeticError("invalid closed background state")
    e = epsilon
    R = rho_c_ratio
    sqrtR = math.sqrt(R)
    pref = 3.0 / (2.0 * sqrtR * (1.0 + e))
    H = sqrtR * math.sin(2.0 * q) / 2.0
    q_tau = pref * (2.0 * A ** -6 - 2.0 / (3.0 * A ** 2))
    A_tau = H * A
    q_A = pref * (-12.0 * A ** -7 + 4.0 / (3.0 * A ** 3))
    q_tau_tau = q_A * A_tau
    H_tau = sqrtR * math.cos(2.0 * q) * q_tau
    H_tau_tau = sqrtR * (-2.0 * math.sin(2.0 * q) * q_tau ** 2
                         + math.cos(2.0 * q) * q_tau_tau)
    Y = 3.0 * A ** -6
    C = A ** -2
    U = Y + H_tau - C
    return {"A": A, "q": q, "H": H, "A_tau": A_tau,
            "q_tau": q_tau, "H_tau": H_tau,
            "H_tau_tau": H_tau_tau, "Y": Y, "C": C, "U": U}


def _closed_w1_A_low(epsilon: float, rho_c_ratio: float) -> float:
    from scipy.optimize import brentq
    level = (1.0 + epsilon) * rho_c_ratio
    value = brentq(lambda A: A ** -6 - A ** -2 - level,
                   1e-6, 1.0 - 1e-12, xtol=2e-14, rtol=1e-14)
    if not math.isfinite(value) or value <= 0 or value >= 1:
        raise ArithmeticError("closed w=1 lower turning point is invalid")
    return float(value)


_B_DERIVATIVE_LAMBDAS = None


def _b_derivative_lambdas():
    global _B_DERIVATIVE_LAMBDAS
    if _B_DERIVATIVE_LAMBDAS is None:
        H, p, C, Y, D = sp.symbols("H p C Y D", real=True)
        U = Y + D - C
        d = p - 3 * C
        delta = (d + 3 * Y) * (d * H ** 2 + C * Y) + d * Y * U
        B = 2 * H * d ** 2 * (d + 3 * Y) / delta
        functions = [B] + [sp.diff(B, var) for var in (H, p, C, Y, D)]
        _B_DERIVATIVE_LAMBDAS = sp.lambdify((H, p, C, Y, D), functions,
                                             "math")
    return _B_DERIVATIVE_LAMBDAS


def _curved_coefficients_float(H: float, p: float, C: float, Y: float,
                               D: float, U: float) -> Tuple[float, float,
                                                             float, float]:
    d = p - 3.0 * C
    delta = (d + 3.0 * Y) * (d * H * H + C * Y) + d * Y * U
    if not all(math.isfinite(v) for v in (d, delta)) or delta <= 0:
        raise ArithmeticError("curved auxiliary determinant denominator invalid")
    A = Y * d * (d + 3.0 * Y) / delta
    B = 2.0 * H * d * d * (d + 3.0 * Y) / delta
    C_s = d - d * d * (Y * p + D * d) / delta
    det = -2.0 * p * delta / U
    if not all(math.isfinite(v) for v in (A, B, C_s, det)):
        raise ArithmeticError("nonfinite curved scalar coefficients")
    return A, B, C_s, delta


def _closed_w1_coefficients(A: float, q: float, ell: int, epsilon: float,
                            rho_c_ratio: float) -> Dict[str, float]:
    bg = _closed_w1_background(A, q, epsilon, rho_c_ratio)
    lam = float(ell * (ell + 2))
    p = lam / (A * A)
    D = bg["H_tau"]
    As, Bs, Cs, delta = _curved_coefficients_float(
        bg["H"], p, bg["C"], bg["Y"], D, bg["U"])
    values = _b_derivative_lambdas()(bg["H"], p, bg["C"], bg["Y"], D)
    B = float(values[0])
    dB_dH, dB_dp, dB_dC, dB_dY, dB_dD = (float(v) for v in values[1:])
    p_tau = -2.0 * bg["H"] * p
    C_tau = -2.0 * bg["H"] * bg["C"]
    Y_tau = -6.0 * bg["H"] * bg["Y"]
    B_tau = (dB_dH * D + dB_dp * p_tau + dB_dC * C_tau
             + dB_dY * Y_tau + dB_dD * bg["H_tau_tau"])
    G = (B_tau + 3.0 * bg["H"] * B) / 2.0 - Cs
    if not all(math.isfinite(v) for v in (B, B_tau, G)):
        raise ArithmeticError("nonfinite closed scalar gradient coefficient")
    return {"lambda": lam, "p": p, "A_s": As, "B_s": Bs,
            "C_s": Cs, "B_tau": B_tau, "G_s": G, "Delta": delta,
            "U": bg["U"], "H": bg["H"], "H_tau": D,
            "Y": bg["Y"], "C": bg["C"]}


def _classify_transfer(matrix: Sequence[Sequence[float]], determinant: float) \
        -> Dict[str, Any]:
    trace = float(matrix[0][0] + matrix[1][1])
    discriminant = float(trace * trace - 4.0 * determinant)
    tol = 1e-8 * max(1.0, abs(trace * trace), 4.0 * abs(determinant))
    if abs(determinant - 1.0) > 2e-5:
        classification = "numerically_unresolved"
    elif discriminant > tol:
        classification = "hyperbolic"
    elif discriminant < -tol:
        classification = "elliptic"
    else:
        classification = "parabolic_or_unresolved"
    if discriminant >= 0:
        root = math.sqrt(max(0.0, discriminant))
        eigen_real = [(trace + root) / 2.0, (trace - root) / 2.0]
        eigen_imag = [0.0, 0.0]
    else:
        root = math.sqrt(-discriminant)
        eigen_real = [trace / 2.0, trace / 2.0]
        eigen_imag = [root / 2.0, -root / 2.0]
    return {"trace": trace, "discriminant": discriminant,
            "classification": classification,
            "eigenvalues_real": eigen_real,
            "eigenvalues_imag": eigen_imag,
            "classification_tolerance": tol}


def _integrate_closed_w1(ell: int, epsilon: float, rho_c_ratio: float,
                         *, rtol: float, atol: float, max_step: float,
                         cycles: int = 1,
                         initial_pair: Optional[Tuple[float, float]] = None):
    import numpy as np
    from scipy.integrate import solve_ivp

    if isinstance(ell, bool) or not isinstance(ell, int) or ell < 2:
        raise ValueError("ell must be an integer >=2")
    if cycles < 1:
        raise ValueError("cycles must be positive")
    A_low = _closed_w1_A_low(epsilon, rho_c_ratio)
    pair = (1.0, 0.0, 0.0, 1.0) if initial_pair is None else initial_pair
    if initial_pair is None:
        initial = (A_low, 0.0, 1.0, 0.0, 0.0, 1.0)
        fundamental = True
    else:
        initial = (A_low, 0.0, float(pair[0]), float(pair[1]))
        fundamental = False
    calls = 0
    diagnostics = {"max_constraint_residual": 0.0,
                   "min_A_s": float("inf"), "min_U": float("inf"),
                   "min_Delta": float("inf"),
                   "max_background_equation_residual": 0.0}

    def rhs(_, state):
        nonlocal calls
        calls += 1
        if calls > 500000:
            raise ArithmeticError("closed scalar transfer exceeded bounded evaluation budget")
        A, q = float(state[0]), float(state[1])
        bg = _closed_w1_background(A, q, epsilon, rho_c_ratio)
        lam = ell * (ell + 2)
        coeff = _closed_w1_coefficients(A, q, ell, epsilon, rho_c_ratio)
        diagnostics["min_A_s"] = min(diagnostics["min_A_s"], coeff["A_s"])
        diagnostics["min_U"] = min(diagnostics["min_U"], coeff["U"])
        diagnostics["min_Delta"] = min(diagnostics["min_Delta"], coeff["Delta"])
        z1, pi1 = float(state[2]), float(state[3])
        out = [bg["A_tau"], bg["q_tau"],
               pi1 / (A ** 3 * coeff["A_s"]),
               -A ** 3 * coeff["G_s"] * z1]
        if fundamental:
            z2, pi2 = float(state[4]), float(state[5])
            out.extend([pi2 / (A ** 3 * coeff["A_s"]),
                        -A ** 3 * coeff["G_s"] * z2])
        if not all(math.isfinite(v) for v in out):
            raise ArithmeticError("nonfinite closed scalar RHS")
        return out

    def event(_, state):
        return float(state[1]) - cycles * math.pi

    event.terminal = True
    event.direction = 1
    try:
        sol = solve_ivp(rhs, (0.0, max(4.0 * cycles + 2.0, 8.0)),
                        initial, method="DOP853", rtol=rtol, atol=atol,
                        max_step=max_step, events=event)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("closed scalar inputs exceed finite float arithmetic") from exc
    if not sol.success or len(sol.t_events[0]) != 1 \
            or not np.all(np.isfinite(sol.y)):
        raise ArithmeticError("closed w=1 integration did not reach endpoint")
    # Error control may evaluate trial stages away from the accepted path.
    # Constraint residuals are therefore audited on the accepted trajectory
    # mesh (including the event endpoint), not on rejected RK stages.
    for A_value, q_value in zip(sol.y[0], sol.y[1]):
        A_value, q_value = float(A_value), float(q_value)
        bg_value = _closed_w1_background(A_value, q_value, epsilon,
                                         rho_c_ratio)
        constraint = A_value ** -6 - A_value ** -2 \
            - (1.0 + epsilon) * rho_c_ratio * math.cos(q_value) ** 2
        friedmann = bg_value["H"] ** 2 + bg_value["C"] - A_value ** -6 \
            + rho_c_ratio * math.cos(q_value) ** 2 \
            * (math.cos(q_value) ** 2 + epsilon)
        ray = (2.0 / 3.0) * (bg_value["H_tau"] - bg_value["C"]) \
            + 2.0 * A_value ** -6 - 2.0 * bg_value["U"] / 3.0
        diagnostics["max_constraint_residual"] = max(
            diagnostics["max_constraint_residual"], abs(constraint))
        diagnostics["max_background_equation_residual"] = max(
            diagnostics["max_background_equation_residual"],
            abs(friedmann), abs(ray))
    endpoint = sol.y_events[0][0]
    if fundamental:
        z1, pi1, z2, pi2 = (float(v) for v in endpoint[2:])
        matrix = [[z1, z2], [pi1, pi2]]
        determinant = z1 * pi2 - z2 * pi1
        return matrix, determinant, float(endpoint[0]), float(endpoint[1]), \
            diagnostics, int(sol.nfev)
    return (float(endpoint[2]), float(endpoint[3]), float(endpoint[0]),
            float(endpoint[1]), diagnostics, int(sol.nfev))


def closed_w1_scalar_transfer(ell: Any, *, epsilon: Any = "0.1",
                              rho_c_ratio: Any = "1", rtol: Any = "2e-10",
                              atol: Any = "2e-12", max_step: Any = "0.025") \
        -> Dict[str, Any]:
    """Compute one closed w=1 S3 scalar transfer for ell>=2."""
    if isinstance(ell, bool) or not isinstance(ell, int):
        raise ValueError("ell must be an integer >=2")
    e = _finite_float(epsilon, "epsilon", nonnegative=True)
    R = _finite_float(rho_c_ratio, "rho_c_ratio", positive=True)
    rr = _finite_float(rtol, "rtol", positive=True)
    aa = _finite_float(atol, "atol", positive=True)
    step = _finite_float(max_step, "max_step", positive=True)
    if rr < 3e-14 or aa < 1e-300:
        raise ValueError("solver tolerance is outside supported float range")
    matrix, determinant, A_end, q_end, diag, calls = _integrate_closed_w1(
        ell, e, R, rtol=rr, atol=aa, max_step=step)
    transfer = _classify_transfer(matrix, determinant)
    return {"ell": ell, "lambda": ell * (ell + 2),
            "epsilon": e, "rho_c_over_rho_ref": R,
            "A_low": _closed_w1_A_low(e, R), "A_end": A_end,
            "q_end": q_end, "matrix": matrix,
            "determinant": determinant,
            "determinant_error": abs(determinant - 1.0),
            "symplectic_error": abs(determinant - 1.0),
            "trace": transfer["trace"],
            "discriminant": transfer["discriminant"],
            "classification": transfer["classification"],
            "eigenvalues_real": transfer["eigenvalues_real"],
            "eigenvalues_imag": transfer["eigenvalues_imag"],
            "classification_tolerance": transfer["classification_tolerance"],
            "max_constraint_residual": diag["max_constraint_residual"],
            "max_background_equation_residual": diag["max_background_equation_residual"],
            "min_A_s": diag["min_A_s"], "min_U": diag["min_U"],
            "min_Delta": diag["min_Delta"], "rhs_evaluations": calls,
            "finite": True,
            "scope": "closed massless canonical w=1; physical S3 harmonics ell=2..; finite cycle only"}


def closed_w1_transfer_convergence(ell: Any, *, epsilon: Any = "0.1",
                                   rho_c_ratio: Any = "1") -> Dict[str, Any]:
    coarse = closed_w1_scalar_transfer(ell, epsilon=epsilon,
                                       rho_c_ratio=rho_c_ratio,
                                       rtol="2e-8", atol="2e-10",
                                       max_step="0.06")
    fine = closed_w1_scalar_transfer(ell, epsilon=epsilon,
                                     rho_c_ratio=rho_c_ratio,
                                     rtol="2e-10", atol="2e-12",
                                     max_step="0.025")
    diff = max(abs(coarse["matrix"][i][j] - fine["matrix"][i][j])
               / max(1.0, abs(fine["matrix"][i][j]))
               for i in range(2) for j in range(2))
    passed = bool(diff < float(CLOSED_REFINEMENT_TOL)
                  and fine["determinant_error"]
                  < float(CLOSED_DETERMINANT_TOL)
                  and fine["max_constraint_residual"]
                  < float(CLOSED_CONSTRAINT_TOL)
                  and fine["max_background_equation_residual"]
                  < float(CLOSED_BACKGROUND_TOL)
                  and fine["min_A_s"] > 0 and fine["min_U"] > 0
                  and fine["min_Delta"] > 0)
    return {"ell": ell, "epsilon": coarse["epsilon"],
            "rho_c_over_rho_ref": coarse["rho_c_over_rho_ref"],
            "coarse": coarse, "fine": fine,
            "relative_matrix_difference": diff, "passed": passed}


def _growth_verification(ell: int, epsilon: float, rho_c_ratio: float,
                         fine: Dict[str, Any]) -> Dict[str, Any]:
    """Evolve a real growing eigenvector over three cycles when resolved."""
    if fine["classification"] != "hyperbolic":
        return {"performed": False,
                "reason": "no resolved hyperbolic eigenvalue in sampled cycle"}
    matrix = fine["matrix"]
    a, b = matrix[0]
    c, d = matrix[1]
    disc = fine["discriminant"]
    root = math.sqrt(max(0.0, disc))
    candidates = [(fine["trace"] + root) / 2.0,
                  (fine["trace"] - root) / 2.0]
    eigenvalue = max(candidates, key=abs)
    if abs(b) > 1e-12:
        vector = (b, eigenvalue - a)
    elif abs(c) > 1e-12:
        vector = (eigenvalue - d, c)
    else:
        vector = (1.0, 0.0)
    initial_norm = math.hypot(*vector)
    final_z, final_pi, _, _, diag, calls = _integrate_closed_w1(
        ell, epsilon, rho_c_ratio, rtol=2e-10, atol=2e-12,
        max_step=0.025, cycles=3, initial_pair=vector)
    final_norm = math.hypot(final_z, final_pi)
    observed = final_norm / initial_norm
    expected = abs(eigenvalue) ** 3
    return {"performed": True, "cycles": 3, "eigenvalue": eigenvalue,
            "initial_norm": initial_norm, "final_norm": final_norm,
            "observed_norm_ratio": observed,
            "expected_eigenvalue_power": expected,
            "relative_growth_residual": abs(observed - expected)
            / max(1.0, abs(expected)),
            "max_constraint_residual": diag["max_constraint_residual"],
            "rhs_evaluations": calls,
            "passed": bool(math.isfinite(observed)
                            and abs(observed - expected)
                            / max(1.0, abs(expected))
                            < float(GROWTH_RESIDUAL_TOL))}


def closed_w1_scalar_scan(*, epsilon: Any = "0.1",
                          rho_c_ratio: Any = "1",
                          harmonics: Iterable[int] = range(2, 13)) \
        -> List[Dict[str, Any]]:
    e = _finite_float(epsilon, "epsilon", nonnegative=True)
    R = _finite_float(rho_c_ratio, "rho_c_ratio", positive=True)
    values: List[Dict[str, Any]] = []
    for ell in harmonics:
        if isinstance(ell, bool) or not isinstance(ell, int) or ell < 2:
            raise ValueError("all harmonics must be integers >=2")
        row = closed_w1_transfer_convergence(ell, epsilon=e, rho_c_ratio=R)
        if row["fine"]["classification"] == "hyperbolic":
            row["growth_verification"] = _growth_verification(
                ell, e, R, row["fine"])
            row["passed"] = bool(row["passed"]
                                  and row["growth_verification"].get("passed") is True)
        else:
            row["growth_verification"] = {"performed": False,
                                            "reason": "finite sampled transfer is not hyperbolic"}
        values.append(row)
    return values


# Small compatibility aliases make the new module easy to interrogate beside
# the two frozen scalar/background producers without conflating their scopes.
perturbation_symbolic_checks = flat_qy_symbolic_checks
transfer_convergence = flat_transfer_convergence
curved_perturbation_symbolic_checks = curved_action_symbolic_checks
flat_perturbation_symbolic_checks = flat_qy_symbolic_checks
scalar_perturbation_symbolic_checks = flat_qy_symbolic_checks
curved_scalar_action_symbolic_checks = curved_action_symbolic_checks
closed_scalar_transfer = closed_w1_scalar_transfer


# ---------------------------------------------------------------------------
# Aggregate JSON surface and fail-closed acceptance


def _matrix_values(record: Any) -> Optional[List[mp.mpf]]:
    """Parse a raw 2x2 matrix and bound entries before any arithmetic."""
    if not isinstance(record, dict):
        return None
    matrix = record.get("matrix")
    if (not isinstance(matrix, list) or len(matrix) != 2
            or any(not isinstance(row, list) or len(row) != 2
                   for row in matrix)):
        return None
    values: List[mp.mpf] = []
    for row in matrix:
        for value in row:
            parsed = _safe_finite(value)
            if parsed is None or abs(parsed) > MATRIX_ENTRY_ABS_MAX:
                return None
            values.append(parsed)
    return values


def _matrix_float_values(record: Any) -> Optional[List[float]]:
    """Use the producer's IEEE-754 arithmetic for reported raw metadata."""
    values = _matrix_values(record)
    if values is None:
        return None
    try:
        converted = [float(value) for value in values]
    except (OverflowError, ValueError, TypeError):
        return None
    return converted if all(math.isfinite(value) for value in converted) \
        else None


def _positive_integer(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and value > 0)


def _mode_set_matches(rows: Any, field: str,
                      expected: Sequence[Any]) -> bool:
    """Require a nonempty, duplicate-free, complete declared mode set."""
    if not isinstance(rows, list) or len(rows) != len(expected) or not rows:
        return False
    actual: List[mp.mpf] = []
    for row in rows:
        if not isinstance(row, dict):
            return False
        value = _safe_finite(row.get(field))
        if value is None:
            return False
        if any(abs(value - prior) <= mp.mpf("1e-12") for prior in actual):
            return False
        actual.append(value)
    return all(any(abs(value - _safe_finite(target)) <= mp.mpf("1e-12")
                   for value in actual)
               for target in expected)


def _record_input_matches(record: Any, *, epsilon: mp.mpf,
                          rho_c_ratio: mp.mpf, kind: str,
                          mode: Any) -> bool:
    """Ensure each raw transfer record preserves the declared inputs."""
    if not isinstance(record, dict):
        return False
    record_epsilon = _safe_finite(record.get("epsilon"))
    if (record_epsilon is None
            or not _reported_matches(record_epsilon, epsilon,
                                     relative="1e-12")):
        return False
    if kind == "flat":
        kappa = _safe_finite(record.get("kappa"))
        expected_mode = _safe_finite(mode)
        return (kappa is not None and expected_mode is not None
                and _reported_matches(kappa, expected_mode,
                                      relative="1e-12"))
    if kind == "closed":
        if (isinstance(record.get("ell"), bool)
                or not isinstance(record.get("ell"), int)
                or record.get("ell") != mode):
            return False
        ratio = _safe_finite(record.get("rho_c_over_rho_ref"))
        return (ratio is not None
                and _reported_matches(ratio, rho_c_ratio,
                                       relative="1e-12"))
    return False


def _matrix_record_passes(record: Any, *, kind: str) -> bool:
    values = _matrix_values(record)
    float_values = _matrix_float_values(record)
    if values is None or float_values is None or not isinstance(record, dict):
        return False
    determinant_float = (float_values[0] * float_values[3]
                         - float_values[1] * float_values[2])
    if not math.isfinite(determinant_float):
        return False
    determinant = mp.mpf(determinant_float)
    determinant_error = mp.mpf(abs(determinant_float - 1.0))
    reported = _safe_finite(record.get("determinant"))
    error = _safe_finite(record.get("determinant_error"))
    if (reported is None or error is None
            or not _reported_matches(reported, determinant, relative="1e-12")
            or not _reported_matches(error, determinant_error,
                                     relative="1e-12")):
        return False
    symplectic_error = _safe_finite(record.get("symplectic_error"))
    if (symplectic_error is None
            or not _reported_matches(symplectic_error, determinant_error,
                                     relative="1e-12")
            or record.get("finite") is not True
            or not _positive_integer(record.get("rhs_evaluations"))):
        return False
    if error < 0 or symplectic_error < 0:
        return False
    if kind == "flat":
        coefficient_error = _safe_finite(record.get("coefficient_formula_max_abs"))
        return (coefficient_error is not None and coefficient_error >= 0
                and coefficient_error < FLAT_COEFFICIENT_TOL
                and error < FLAT_DETERMINANT_TOL)
    if kind != "closed":
        return False
    if error >= CLOSED_DETERMINANT_TOL \
            or symplectic_error >= CLOSED_DETERMINANT_TOL:
        return False
    for name in ("max_constraint_residual",
                 "max_background_equation_residual"):
        number = _safe_finite(record.get(name))
        limit = (CLOSED_CONSTRAINT_TOL if name == "max_constraint_residual"
                 else CLOSED_BACKGROUND_TOL)
        if number is None or number < 0 or number >= limit:
            return False
    for name in ("min_A_s", "min_U", "min_Delta"):
        number = _safe_finite(record.get(name))
        if number is None or number <= 0:
            return False
    for name, positive in (("A_low", True), ("A_end", True),
                           ("q_end", False)):
        number = _safe_finite(record.get(name))
        if number is None or (positive and number <= 0):
            return False
    trace = _safe_finite(record.get("trace"))
    discriminant = _safe_finite(record.get("discriminant"))
    classification_tolerance = _safe_finite(
        record.get("classification_tolerance"))
    if (trace is None or discriminant is None
            or classification_tolerance is None):
        return False
    expected_trace = mp.mpf(float_values[0] + float_values[3])
    expected_discriminant = mp.mpf(
        (float_values[0] + float_values[3]) ** 2
        - 4.0 * determinant_float)
    if (not _reported_matches(trace, expected_trace, relative="1e-10")
            or not _reported_matches(discriminant, expected_discriminant,
                                     relative="1e-10")):
        return False
    try:
        recomputed = _classify_transfer(
            [[float_values[0], float_values[1]],
             [float_values[2], float_values[3]]], determinant_float)
    except (OverflowError, ValueError, TypeError):
        return False
    if (record.get("classification") != recomputed["classification"]
            or not _reported_matches(classification_tolerance,
                                     recomputed["classification_tolerance"],
                                     relative="1e-10")):
        return False
    for name, expected_values in (
            ("eigenvalues_real", recomputed["eigenvalues_real"]),
            ("eigenvalues_imag", recomputed["eigenvalues_imag"])):
        values_reported = record.get(name)
        if (not isinstance(values_reported, list)
                or len(values_reported) != 2):
            return False
        for reported_value, expected_value in zip(values_reported,
                                                  expected_values):
            if not _reported_matches(reported_value, expected_value,
                                     relative="1e-10"):
                return False
    return True


def _convergence_record_passes(record: Any, *, kind: str,
                               epsilon: Optional[mp.mpf] = None,
                               rho_c_ratio: Optional[mp.mpf] = None,
                               mode: Any = None) -> bool:
    if not isinstance(record, dict) or record.get("passed") is not True:
        return False
    coarse = record.get("coarse")
    fine = record.get("fine")
    if (not _matrix_record_passes(coarse, kind=kind)
            or not _matrix_record_passes(fine, kind=kind)):
        return False
    if kind == "flat":
        top_mode = _safe_finite(record.get("kappa"))
        top_epsilon = _safe_finite(record.get("epsilon"))
        if top_mode is None or top_epsilon is None or top_mode < 0 \
                or top_epsilon < 0:
            return False
        if mode is not None and not _reported_matches(
                top_mode, mode, relative="1e-12"):
            return False
        for nested in (coarse, fine):
            if (not _record_input_matches(
                    nested, epsilon=top_epsilon,
                    rho_c_ratio=(rho_c_ratio if rho_c_ratio is not None
                                 else mp.mpf(0)),
                    kind=kind, mode=top_mode)
                    or not _reported_matches(nested.get("kappa"), top_mode,
                                             relative="1e-12")):
                return False
        if epsilon is not None and not _reported_matches(
                top_epsilon, epsilon, relative="1e-12"):
            return False
    elif kind == "closed":
        top_epsilon = _safe_finite(record.get("epsilon"))
        top_ratio = _safe_finite(record.get("rho_c_over_rho_ref"))
        top_mode = record.get("ell")
        if (top_epsilon is None or top_epsilon < 0 or top_ratio is None
                or top_ratio <= 0 or isinstance(top_mode, bool)
                or not isinstance(top_mode, int) or top_mode < 2):
            return False
        if mode is not None and top_mode != mode:
            return False
        for nested in (coarse, fine):
            if (not _record_input_matches(
                    nested, epsilon=top_epsilon,
                    rho_c_ratio=top_ratio, kind=kind, mode=top_mode)
                    or nested.get("ell") != top_mode):
                return False
        if (epsilon is not None and not _reported_matches(
                top_epsilon, epsilon, relative="1e-12")):
            return False
        if (rho_c_ratio is not None and not _reported_matches(
                top_ratio, rho_c_ratio, relative="1e-12")):
            return False
    else:
        return False
    try:
        coarse_values = _matrix_float_values(coarse)
        fine_values = _matrix_float_values(fine)
        if coarse_values is None or fine_values is None:
            return False
        expected = mp.mpf(max(
            abs(coarse_values[index] - fine_values[index])
            / max(1.0, abs(fine_values[index]))
            for index in range(4)))
    except (TypeError, ValueError, OverflowError):
        return False
    reported = _safe_finite(record.get("relative_matrix_difference"))
    if reported is None or not _reported_matches(reported, expected,
                                                 relative="1e-10"):
        return False
    if kind == "flat" and expected >= FLAT_REFINEMENT_TOL:
        return False
    if kind == "closed" and expected >= CLOSED_REFINEMENT_TOL:
        return False
    if kind == "closed":
        growth = record.get("growth_verification")
        if not isinstance(growth, dict):
            return False
        if fine.get("classification") == "hyperbolic":
            if growth.get("performed") is not True \
                    or growth.get("passed") is not True \
                    or growth.get("cycles") != 3:
                return False
            eigenvalue = _safe_finite(growth.get("eigenvalue"))
            initial_norm = _safe_finite(growth.get("initial_norm"))
            final_norm = _safe_finite(growth.get("final_norm"))
            observed = _safe_finite(growth.get("observed_norm_ratio"))
            expected_power = _safe_finite(
                growth.get("expected_eigenvalue_power"))
            residual = _safe_finite(growth.get("relative_growth_residual"))
            growth_constraint = _safe_finite(
                growth.get("max_constraint_residual"))
            if (eigenvalue is None or initial_norm is None
                    or final_norm is None or observed is None
                    or expected_power is None or residual is None
                    or growth_constraint is None
                    or initial_norm <= 0 or final_norm <= 0
                    or observed <= 0 or expected_power <= 0
                    or residual < 0 or residual >= GROWTH_RESIDUAL_TOL
                    or growth_constraint < 0
                    or growth_constraint >= CLOSED_CONSTRAINT_TOL
                    or not _positive_integer(growth.get("rhs_evaluations"))):
                return False
            if not _reported_matches(observed, final_norm / initial_norm,
                                     relative="1e-10"):
                return False
            if not _reported_matches(expected_power, abs(eigenvalue) ** 3,
                                     relative="1e-10"):
                return False
            expected_residual = abs(observed - expected_power) \
                / max(mp.mpf(1), abs(expected_power))
            if not _reported_matches(residual, expected_residual,
                                     relative="1e-10"):
                return False
            eigenvalues = fine.get("eigenvalues_real")
            if (not isinstance(eigenvalues, list) or len(eigenvalues) != 2
                    or not any(_reported_matches(eigenvalue, candidate,
                                                 relative="1e-10")
                               for candidate in eigenvalues)
                    or abs(eigenvalue) <= 1):
                return False
        elif growth.get("performed") is not False:
            return False
        elif set(growth) != {"performed", "reason"} \
                or not isinstance(growth.get("reason"), str) \
                or not growth.get("reason"):
            return False
    return True


def _flat_bounds_passes(bounds: Any, epsilon: mp.mpf) -> bool:
    if not isinstance(bounds, dict):
        return False
    if (bounds.get("certificate_passed") is not True
            or bounds.get("kinetic_strictly_positive_for_x_gt_0") is not True
            or bounds.get("proof_is_polynomial_not_scan") is not True
            or bounds.get("analytic_domain")
            != "0<epsilon<1, 0<x<=1, K>=0"
            or not _reported_matches(bounds.get("epsilon"), epsilon,
                                     relative="1e-12")):
        return False
    expected_lower = (1 - epsilon) / (3 * (1 + epsilon))
    expected_upper = (epsilon + 3) / (1 + epsilon)
    if (not _reported_matches(bounds.get("lower_bound"), expected_lower,
                              relative="1e-12")
            or not _reported_matches(bounds.get("upper_bound"), expected_upper,
                                     relative="1e-12")):
        return False
    if epsilon < 1:
        expected_regime = ("frozen_epsilon_zero_control" if epsilon == 0
                           else "strict_positive_sign_bounds")
        return (bounds.get("regime") == expected_regime
                and bounds.get("strict_all_domain") is True
                and "marginal_point" not in bounds
                and "counterexample" not in bounds)
    if epsilon == 1:
        marginal = bounds.get("marginal_point")
        return (bounds.get("regime") == "marginal_epsilon_one"
                and bounds.get("strict_all_domain") is False
                and isinstance(marginal, dict)
                and _reported_matches(marginal.get("epsilon"), epsilon,
                                      relative="1e-12")
                and _reported_matches(marginal.get("x"), 1,
                                      relative="1e-12")
                and _reported_matches(marginal.get("K"), 0,
                                      relative="1e-12")
                and _reported_matches(marginal.get("cs2"), 0,
                                      relative="1e-12"))
    threshold = (epsilon ** 2 - 1) / (epsilon + 3)
    counter = bounds.get("counterexample")
    if (bounds.get("regime") != "negative_long_wavelength_coefficient"
            or bounds.get("strict_all_domain") is not False
            or not isinstance(counter, dict)
            or counter.get("negative_cs2") is not True):
        return False
    point = counter.get("point")
    expected_k = threshold / 2
    if (not isinstance(point, dict)
            or not _reported_matches(counter.get("x"), 1,
                                     relative="1e-12")
            or not _reported_matches(counter.get("K"), expected_k,
                                     relative="1e-12")
            or not _reported_matches(counter.get("threshold_K"), threshold,
                                     relative="1e-12")
            or not _reported_matches(point.get("epsilon"), epsilon,
                                     relative="1e-12")
            or not _reported_matches(point.get("x"), 1,
                                     relative="1e-12")
            or not _reported_matches(point.get("K"), expected_k,
                                     relative="1e-12")
            or _safe_finite(point.get("cs2")) is None
            or _safe_finite(point.get("cs2")) >= 0):
        return False
    expected_point = flat_coefficients(epsilon, 1, expected_k)
    return _reported_matches(point.get("cs2"), expected_point["cs2"],
                             relative="1e-10")


def _paths_passes(paths: Any, epsilon: mp.mpf) -> bool:
    if not isinstance(paths, dict):
        return False
    ratio = mp.mpf(1)
    c = 1 + epsilon
    proportional = (ratio ** 2 / 3 + ratio * c * (epsilon + 2)
                    + c ** 2 * (3 - epsilon)) \
        / (ratio ** 2 / 3 + ratio * c * (epsilon + 2) + 3 * c ** 3)
    zero = (3 - epsilon) / (3 * c)
    if (paths.get("scope")
            != "two low-density scalings of the extension; no forced finite-epsilon GR limit"
            or not _reported_matches(paths.get("epsilon"), epsilon,
                                      relative="1e-12")
            or not _reported_matches(paths.get("K_over_x"), ratio,
                                     relative="1e-12")
            or not _reported_matches(paths.get("fixed_K_positive_limit"), 1,
                                     relative="1e-12")
            or not _reported_matches(paths.get("K_zero_limit"), zero,
                                     relative="1e-12")
            or not _reported_matches(paths.get("K_proportional_limit"),
                                     proportional, relative="1e-12")
            or not _reported_matches(paths.get("bare_GR_reference"), 1,
                                     relative="1e-12")
            or paths.get("finite_epsilon_proportional_differs_from_bare_GR")
            is not (epsilon > 0 and abs(proportional - 1)
                    > mp.mpf("1e-40"))
            or paths.get("paths_are_distinct")
            is not (abs(proportional - 1) > mp.mpf("1e-40")
                    or abs(zero - 1) > mp.mpf("1e-40"))):
        return False
    return True


def _sign_certificate_passes(certificate: Any) -> bool:
    if not isinstance(certificate, dict):
        return False
    p, C, Y, H, D_H, U = (mp.mpf(value) for value in
                           (8, 1, 3, 0, mp.mpf("0.5"), mp.mpf("2.5")))
    d = p - 3 * C
    first = (d + 3 * Y) * (d * H ** 2 + C * Y)
    second = d * Y * U
    delta = first + second
    numerator = Y * d * (d + 3 * Y)
    expected = {
        "p": p, "C": C, "Y": Y, "H": H, "D_H": D_H, "U": U,
        "d": d, "Delta_first_nonnegative": first,
        "Delta_second_strictly_positive": second, "Delta": delta,
        "A_s_numerator": numerator,
        "determinant_formula": -2 * p * delta / U,
    }
    for name, value in expected.items():
        if not _reported_matches(certificate.get(name), value,
                                 relative="1e-12"):
            return False
    return (certificate.get("Delta_positive") is True
            and certificate.get("A_s_positive") is True
            and certificate.get("determinant_negative") is True
            and certificate.get("scope")
            == "analytic sign certificate for C>=0,p>3C,Y>0,U>0, including H=0")


def _points_pass(result_points: Any) -> bool:
    if not isinstance(result_points, list) or len(result_points) != 2:
        return False
    for row in result_points:
        if (not isinstance(row, dict) or row.get("passed") is not True
                or row.get("finite") is not True):
            return False
        residual = _safe_finite(row.get("max_scaled_residual"))
        if (residual is None or residual < 0
                or residual >= POINT_RESIDUAL_TOL):
            return False
        residuals = row.get("residuals")
        if not isinstance(residuals, dict) or set(residuals) != {
                "A_s", "B_s", "C_s", "determinant", "auxiliary_solution"}:
            return False
        for value in residuals.values():
            parsed = _safe_finite(value)
            if parsed is None or abs(parsed) >= POINT_RESIDUAL_TOL:
                return False
        for name in ("p", "Y", "U", "Delta", "A_s"):
            value = _safe_finite(row.get(name))
            if value is None or value <= 0:
                return False
        det = _safe_finite(row.get("determinant"))
        if det is None or det >= 0:
            return False
    return True


def acceptance_from_result(result: Any) -> bool:
    try:
        if not isinstance(result, dict):
            return False
        if (result.get("status") != STATUS
                or result.get("evidentiary_weight") != 0
                or result.get("mathematical_checks_passed") is not True
                or result.get("sources") != SOURCES):
            return False
        symbolic = result.get("symbolic")
        if (not isinstance(symbolic, dict)
                or set(symbolic) != {"flat", "curved", "flat_reconstruction"}
                or not rows_pass(symbolic.get("flat"), REQUIRED_FLAT_ROWS)
                or not rows_pass(symbolic.get("curved"), REQUIRED_CURVED_ROWS)
                or not rows_pass(symbolic.get("flat_reconstruction"),
                                 REQUIRED_FLAT_RECONSTRUCTION_ROWS)):
            return False
        inputs = result.get("inputs")
        if (not isinstance(inputs, dict)
                or set(inputs) != {"epsilon", "rho_c_over_rho_ref",
                                    "include_transfer", "include_closed"}
                or not isinstance(inputs.get("include_transfer"), bool)
                or not isinstance(inputs.get("include_closed"), bool)):
            return False
        epsilon = _safe_finite(inputs.get("epsilon"))
        rho_c_ratio = _safe_finite(inputs.get("rho_c_over_rho_ref"))
        if epsilon is None or epsilon < 0 or rho_c_ratio is None \
                or rho_c_ratio <= 0:
            return False
        scope = result.get("scope")
        if (not isinstance(scope, dict)
                or scope.get("declared_flat_kappas") != list(DECLARED_FLAT_KAPPAS)
                or scope.get("declared_closed_ell") != list(DECLARED_CLOSED_ELL)
                or scope.get("curved_expansion_provenance")
                != CURVED_EH_PROVENANCE):
            return False
        expected_flat_scope = (FLAT_TRANSFER_SCOPE
                               if inputs["include_transfer"]
                               else FLAT_TRANSFER_OMITTED_SCOPE)
        expected_closed_scope = (CLOSED_TRANSFER_SCOPE
                                 if inputs["include_closed"]
                                 else CLOSED_TRANSFER_OMITTED_SCOPE)
        if (scope.get("flat_transfer_scope") != expected_flat_scope
                or scope.get("closed_transfer_scope") != expected_closed_scope):
            return False
        if not _flat_bounds_passes(result.get("flat_bounds"), epsilon):
            return False
        if not _paths_passes(result.get("low_density_paths"), epsilon):
            return False
        if not _sign_certificate_passes(result.get("curved_sign_certificate")):
            return False
        if not _points_pass(result.get("curved_points")):
            return False

        flat_group = result.get("flat_transfers")
        closed_group = result.get("closed_w1_transfers")
        if inputs["include_transfer"]:
            if not _mode_set_matches(flat_group, "kappa",
                                     DECLARED_FLAT_KAPPAS):
                return False
            for row in flat_group:
                if not _convergence_record_passes(
                        row, kind="flat", epsilon=epsilon,
                        rho_c_ratio=rho_c_ratio, mode=row.get("kappa")):
                    return False
        elif flat_group != []:
            return False
        if inputs["include_closed"]:
            if not _mode_set_matches(closed_group, "ell",
                                     DECLARED_CLOSED_ELL):
                return False
            for row in closed_group:
                if not _convergence_record_passes(
                        row, kind="closed", epsilon=epsilon,
                        rho_c_ratio=rho_c_ratio, mode=row.get("ell")):
                    return False
        elif closed_group != []:
            return False
        return True
    except Exception:
        # Evidence is untrusted input.  Any malformed arithmetic fails closed.
        return False


def compute_state(*, epsilon: Any = "0.1", rho_c_ratio: Any = "1",
                  include_transfer: bool = True,
                  include_closed: bool = True) -> Dict[str, Any]:
    if not isinstance(include_transfer, bool) or not isinstance(include_closed, bool):
        raise ValueError("include_transfer and include_closed must be bool")
    e = finite(epsilon, "epsilon", nonnegative=True)
    R = finite(rho_c_ratio, "rho_c_ratio", positive=True)
    with mp.workdps(90):
        flat_rows = flat_qy_symbolic_checks()
        curved_rows = curved_action_symbolic_checks()
        flat_reconstruction = curved_flat_reconstruction_symbolic_checks()
        bounds = flat_bound_certificate(e)
        paths = low_density_paths(e)
        sign_certificate = curved_sign_certificate()
        points = [
            curved_coefficient_point(H="0.37", p="8", C="1", Y="3", D_H="0.5"),
            curved_coefficient_point(H="0", p="8", C="1", Y="3", D_H="0.5"),
        ]
        flat_transfers = []
        if include_transfer:
            flat_transfers = [flat_transfer_convergence(k, epsilon=e)
                              for k in ("0.1", "1", "10")]
        closed_transfers = []
        if include_closed:
            closed_transfers = closed_w1_scalar_scan(epsilon=e,
                                                     rho_c_ratio=R)
        symbolic_ok = (rows_pass(flat_rows, REQUIRED_FLAT_ROWS)
                       and rows_pass(curved_rows, REQUIRED_CURVED_ROWS)
                       and rows_pass(flat_reconstruction,
                                     REQUIRED_FLAT_RECONSTRUCTION_ROWS)
                       and bounds.get("certificate_passed") is True
                       and sign_certificate.get("Delta_positive") is True
                       and sign_certificate.get("A_s_positive") is True
                       and all(item.get("passed") is True for item in points))
        numeric_ok = ((not include_transfer or
                       all(item.get("passed") is True for item in flat_transfers))
                      and (not include_closed or
                           all(item.get("passed") is True
                               for item in closed_transfers)))
        result = {
            "status": STATUS, "evidentiary_weight": EVIDENCE_WEIGHT,
            "mathematical_checks_passed": bool(symbolic_ok and numeric_ok),
            "inputs": {"epsilon": e, "rho_c_over_rho_ref": R,
                       "include_transfer": include_transfer,
                       "include_closed": include_closed},
            "symbolic": {"flat": flat_rows, "curved": curved_rows,
                         "flat_reconstruction": flat_reconstruction},
            "flat_bounds": bounds, "low_density_paths": paths,
            "curved_sign_certificate": sign_certificate,
            "curved_points": points, "flat_transfers": flat_transfers,
            "closed_w1_transfers": closed_transfers,
            "scope": {
                "model": "declared negative-cuscuton epsilon extension; mathematical counterfactual",
                "flat": "canonical massless P=rho benchmark; QY coefficient identities and finite classical pair",
                "curved": "matter-comoving gauge on unit S3; physical ell>=2 harmonics only",
                "instability": "instantaneous signs and finite Floquet samples are distinct; no all-mode theorem",
                "causality": "superluminal phase coefficient is reported separately from front velocity and UV completion",
                "flat_transfer_scope": (FLAT_TRANSFER_SCOPE if include_transfer
                                        else FLAT_TRANSFER_OMITTED_SCOPE),
                "closed_transfer_scope": (CLOSED_TRANSFER_SCOPE if include_closed
                                          else CLOSED_TRANSFER_OMITTED_SCOPE),
                "declared_flat_kappas": list(DECLARED_FLAT_KAPPAS),
                "declared_closed_ell": list(DECLARED_CLOSED_ELL),
                "curved_expansion_provenance": CURVED_EH_PROVENANCE,
                "excluded": "l=0,1 constraints; quantum spectrum; nonlinear completion; full NVG matter and observations",
            },
            "sources": SOURCES,
        }
        # The aggregate flag is itself fail-closed: acceptance recomputes all
        # fixed bounds and mode/input controls instead of trusting producer
        # booleans.  This call is local and does not rerun any integration.
        result["mathematical_checks_passed"] = bool(
            result["mathematical_checks_passed"]
            and acceptance_from_result(result))
        return _json_values(result)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", default="0.1")
    parser.add_argument("--rho-c-ratio", default="1")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-closed", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = compute_state(epsilon=args.epsilon,
                               rho_c_ratio=args.rho_c_ratio,
                               include_transfer=not args.skip_transfer,
                               include_closed=not args.skip_closed)
    except (ValueError, ArithmeticError) as exc:
        print(json.dumps({"status": "invalid_or_failed_audit",
                          "evidentiary_weight": 0, "error": str(exc)},
                         allow_nan=False))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if acceptance_from_result(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
