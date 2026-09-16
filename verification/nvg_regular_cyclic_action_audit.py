#!/usr/bin/env python3
"""Independent audit of the regular epsilon deformation of the cuscuton action.

This is a separately labelled mathematical counterfactual.  It does not import
the preserved NVG action or either preceding phase product.  The kinetic term
is the unchanged negative cuscuton term ``-sqrt((d Psi)^2)``; epsilon changes
the field map and the potential together.  All numerical scales and constant-w
cycles are manufactured controls, and the command-line interface prints JSON
only.

Conventions are +---, ``M2`` is the bare Planck-mass squared, ``kappa`` is the
FLRW curvature sign, and the positive-gradient branch has ``Psi_dot > 0``.
The action variation is first done with a lapse and with an explicit matter
energy functional, before the candidate reconstruction is tested.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict

import mpmath as mp
import sympy as sp


STATUS = "regular_epsilon_cyclic_counterfactual_not_NVG_completion"
EVIDENCE_WEIGHT = 0
SOURCE_PATH = __file__
SOURCES = {
    "run_authority": "Lunacy/runs/spatial-action-closure-2026-09-09/PLAN.md",
    "action_variation_context": "verification/bounce_action_derivation_audit.py",
    "cuscuton_branch_check": "https://arxiv.org/abs/1911.06040v2",
    "planck_mass_shift_context": "https://arxiv.org/abs/astro-ph/0702002",
}


# Exact group coverage is part of the evidence contract.  These are row names,
# not expected-result tables: each row is still produced by the symbolic
# derivation at runtime and must carry its own residual.
REQUIRED_SYMBOLIC_ROWS = {
    "action_variation_and_candidate": frozenset({
        "lapse_variation_from_actual_action",
        "cuscuton_EL_from_actual_action",
        "scale_variation_from_actual_action",
        "raychaudhuri_after_lapse_elimination",
        "candidate_cuscuton_EL_branch",
        "candidate_key_identity",
        "candidate_friedmann_curved",
        "candidate_raychaudhuri_curved",
        "candidate_constraint_derivative",
        "candidate_flat_reduction",
        "candidate_bounce_H_zero",
        "candidate_bounce_Hdot_positive",
    }),
    "flat_reduction": frozenset({
        "flat_H2_from_q_vs_rho_law",
        "bounce_density_from_q",
        "bounce_H_from_q",
        "flat_bounce_zero_of_H2",
        "flat_low_density_slope_from_q_limit",
    }),
    "curved_law": frozenset({
        "constraint_F_uses_bare_M",
        "low_density_F_slope",
        "low_density_curvature_coefficient",
        "curvature_does_not_use_M_cosmological",
    }),
    "cost_theorem": frozenset({
        "smooth_maximum_vacuum_limit",
        "family_endpoint_curvature",
        "family_gravity_ratio",
        "family_flat_coefficient",
    }),
    "regularity": frozenset({
        "positive_map_derivative",
        "positive_map_endpoint_derivative",
        "potential_gradient",
        "endpoint_V_Psi_zero",
        "endpoint_curvature_K",
        "epsilon_zero_gradient",
        "epsilon_zero_cubic_gap",
        "epsilon_zero_cubic_coefficient",
        "epsilon_zero_quartic_potential",
    }),
    "barotropic_F": frozenset({"F_monotone_derivative_identity"}),
}
SYMBOLIC_GROUP_NAMES = frozenset(REQUIRED_SYMBOLIC_ROWS)
SYMBOLIC_ROW_SCHEMA = frozenset({"residual", "passed"})

POINT_RESIDUAL_NAMES = frozenset({
    "friedmann", "raychaudhuri", "cuscuton_gradient",
    "constraint_derivative", "key_identity",
})

REQUIRED_CYCLE_NORMALIZATION = {
    "A": "a/a_ref",
    "a_ref": "sqrt(3*M2/rho_ref)",
    "tau": "t/a_ref",
    "Psi_dot_dimensionless": "physical Psi_dot/rho_ref",
    "rho_c": "R*rho_ref",
}

REQUIRED_CYCLE_W = ("0", "0.3333333333333333333333333333333333")
CYCLE_NUMERIC_FIELDS = (
    "w", "epsilon", "rho_c_over_rho_ref", "m", "C=(1+epsilon)*R",
    "A_low", "A_high", "period_quadrature_coarse",
    "period_quadrature_fine", "quadrature_error_estimate",
    "quadrature_refinement_relative", "period_direct_ode",
    "ode_vs_quadrature_relative", "A_end_direct_ode", "q_end_direct_ode",
    "algebraic_curve_constraint_max_abs",
    "direct_ode_trajectory_curve_max_abs",
    "trajectory_constraint_curve_max_abs", "max_constraint_residual",
    "max_friedmann_residual", "max_raychaudhuri_residual",
    "max_continuity_residual", "min_dimensionless_Psi_q",
    "min_dimensionless_Psi_dot", "turnaround_dimensionless_Psi_dot",
    "turnaround_dimensionless_Psi_q", "min_q_rate", "min_pressure",
    "max_F_derivative",
)
CYCLE_BOOL_FIELDS = ("passed", "inputs_manufactured")
REQUIRED_CYCLE_KEYS = frozenset(
    CYCLE_NUMERIC_FIELDS + CYCLE_BOOL_FIELDS + ("samples", "normalization", "scope"))

POINT_NUMERIC_FIELDS = (
    "epsilon", "q", "Q", "M2", "rho_c", "kappa", "scale_factor",
    "max_scaled_residual",
)
REQUIRED_POINT_KEYS = frozenset(
    POINT_NUMERIC_FIELDS + ("dps", "residuals", "passed"))

ENDPOINT_INVERSE_NUMERIC_FIELDS = (
    "epsilon", "M2", "rho_c", "h", "endpoint_Psi_q",
    "endpoint_dq_dPsi", "finite_difference_dq_dPsi",
    "finite_difference_relative_error", "product_check",
)
REQUIRED_ENDPOINT_INVERSE_KEYS = frozenset(
    ENDPOINT_INVERSE_NUMERIC_FIELDS + ("dps", "passed", "valid"))

VACUUM_LIMIT_NUMERIC_FIELDS = (
    "epsilon", "M2", "rho_c", "delta", "direct_ratio_at_q",
    "exact_flat_vacuum_ratio", "direct_relative_error", "K_endpoint",
    "theorem_ratio", "theorem_relative_error", "gravity_ratio_to_bare",
)
REQUIRED_VACUUM_LIMIT_KEYS = frozenset(
    VACUUM_LIMIT_NUMERIC_FIELDS + ("dps", "passed", "valid"))

OLD_CONSTRAINT_NUMERIC_FIELDS = (
    "epsilon", "q", "M2", "rho_c", "kappa", "scale_factor",
    "actual_minus_old_constraint",
)
REQUIRED_OLD_CONSTRAINT_KEYS = frozenset(
    OLD_CONSTRAINT_NUMERIC_FIELDS
    + ("nonzero_for_positive_epsilon", "accepted", "scope"))

OMITTED_RATE_NUMERIC_FIELDS = (
    "epsilon", "wrong_qdot", "q_sample", "Q", "M2", "rho_c", "kappa",
    "scale_factor", "raychaudhuri_residual",
)
REQUIRED_OMITTED_RATE_KEYS = frozenset(
    OMITTED_RATE_NUMERIC_FIELDS + ("nonzero", "accepted", "scope"))

KINETIC_NEGATIVE_NUMERIC_FIELDS = ("epsilon", "X", "mu", "L_X")
REQUIRED_KINETIC_NEGATIVE_KEYS = frozenset(
    KINETIC_NEGATIVE_NUMERIC_FIELDS
    + ("L_X_negative_near_zero", "L_X_plus_2X_L_XX",
       "combination_residual", "combination_equals_epsilon_over_2",
       "same_action", "accepted", "scope"))

DOMAIN_CONTROL_NUMERIC_FIELDS = ("epsilon", "endpoint_Psi_q")
REQUIRED_DOMAIN_CONTROL_KEYS = frozenset(
    DOMAIN_CONTROL_NUMERIC_FIELDS
    + ("valid_regular_family", "global_Psi_q_positive",
       "inverse_analytic_for_fixed_epsilon", "scope"))


def finite(value: Any, name: str, *, positive: bool = False,
           nonnegative: bool = False) -> mp.mpf:
    """Parse a finite real number without accepting bools or NaNs."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        number = mp.mpf(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not mp.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def decimal(value: Any) -> str:
    return mp.nstr(finite(value, "derived value"), 35)


def input_decimal(value: Any) -> str:
    """Preserve exact decimal-string controls in JSON metadata."""
    if isinstance(value, str):
        return value
    return decimal(value)


def _safe_finite(value: Any) -> Any:
    """Return an mp number or ``None`` for malformed numerical evidence."""
    if isinstance(value, bool):
        return None
    try:
        number = mp.mpf(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if mp.isfinite(number) else None


def _reported_matches(value: Any, expected: Any,
                      *, relative: Any = "1e-20") -> bool:
    """Compare independently reported numbers without accepting non-numbers."""
    actual_number = _safe_finite(value)
    expected_number = _safe_finite(expected)
    if actual_number is None or expected_number is None:
        return False
    if expected_number == 0:
        return bool(actual_number == 0)
    scale = max(mp.mpf("1e-120"), abs(actual_number), abs(expected_number))
    return bool(abs(actual_number - expected_number) <=
                mp.mpf(relative) * scale)


def _json_values(value: Any) -> Any:
    if isinstance(value, mp.mpf):
        return decimal(value)
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ArithmeticError("nonfinite JSON output")
    return value


def _rows(expressions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Simplify independently constructed symbolic residuals."""
    result = {}
    for name, expression in expressions.items():
        reduced = sp.trigsimp(sp.factor(sp.simplify(expression)))
        result[name] = {"residual": str(reduced), "passed": bool(reduced == 0)}
    return result


def rows_pass(rows: Dict[str, Dict[str, Any]],
              required_names: Any = None) -> bool:
    """Fail closed on missing rows, schema drift, non-booleans or nonzeros."""
    if not isinstance(rows, dict) or not rows:
        return False
    if required_names is not None and set(rows) != set(required_names):
        return False
    return all(
        isinstance(row, dict)
        and set(row) == SYMBOLIC_ROW_SCHEMA
        and isinstance(row["residual"], str)
        and isinstance(row["passed"], bool)
        and row["passed"] is True
        and row["residual"] == "0" for row in rows.values()
    )


def _construction(epsilon: Any = "0.1", M2: Any = "1", rho_c: Any = "1"):
    eps = finite(epsilon, "epsilon")
    m2 = finite(M2, "M2", positive=True)
    rc = finite(rho_c, "rho_c", positive=True)
    alpha = mp.sqrt(3 / (4 * m2 * rc))
    return eps, m2, rc, alpha


def psi_map(q: Any, *, epsilon: Any = "0.1", M2: Any = "1",
            rho_c: Any = "1") -> mp.mpf:
    qv = finite(q, "q")
    eps, _, _, alpha = _construction(epsilon, M2, rho_c)
    return ((1 + eps) * qv + mp.sin(2 * qv) / 2) / alpha


def potential_map(q: Any, *, epsilon: Any = "0.1", M2: Any = "1",
                  rho_c: Any = "1") -> mp.mpf:
    qv = finite(q, "q")
    eps, _, rc, _ = _construction(epsilon, M2, rho_c)
    x = mp.cos(qv) ** 2
    return -rc * x * (x + eps)


def hubble_map(q: Any, *, epsilon: Any = "0.1", M2: Any = "1",
               rho_c: Any = "1") -> mp.mpf:
    qv = finite(q, "q")
    _, _, rc, alpha = _construction(epsilon, M2, rho_c)
    return alpha * rc * mp.sin(2 * qv) / 3


def psi_q_map(q: Any, *, epsilon: Any = "0.1", M2: Any = "1",
              rho_c: Any = "1") -> mp.mpf:
    qv = finite(q, "q")
    eps, _, _, alpha = _construction(epsilon, M2, rho_c)
    return (eps + 2 * mp.cos(qv) ** 2) / alpha


def hubble_q_map(q: Any, *, epsilon: Any = "0.1", M2: Any = "1",
                 rho_c: Any = "1") -> mp.mpf:
    qv = finite(q, "q")
    _, _, rc, alpha = _construction(epsilon, M2, rho_c)
    return 2 * alpha * rc * mp.cos(2 * qv) / 3


def density_constraint(q: Any, *, epsilon: Any = "0.1", M2: Any = "1",
                       rho_c: Any = "1", kappa: Any = "0",
                       scale_factor: Any = "1") -> mp.mpf:
    qv = finite(q, "q")
    eps, m2, rc, _ = _construction(epsilon, M2, rho_c)
    kap = finite(kappa, "kappa")
    a = finite(scale_factor, "scale_factor", positive=True)
    return (1 + eps) * rc * mp.cos(qv) ** 2 + 3 * m2 * kap / a ** 2


def q_rate(Q: Any, *, epsilon: Any = "0.1", M2: Any = "1",
           rho_c: Any = "1", kappa: Any = "0",
           scale_factor: Any = "1") -> mp.mpf:
    qent = finite(Q, "Q", positive=True)
    eps, m2, _, alpha = _construction(epsilon, M2, rho_c)
    kap = finite(kappa, "kappa")
    a = finite(scale_factor, "scale_factor", positive=True)
    return alpha / (1 + eps) * (qent - 2 * m2 * kap / a ** 2)


def psi_dot(Q: Any, q: Any, *, epsilon: Any = "0.1", M2: Any = "1",
            rho_c: Any = "1", kappa: Any = "0",
            scale_factor: Any = "1") -> mp.mpf:
    return psi_q_map(q, epsilon=epsilon, M2=M2, rho_c=rho_c) * q_rate(
        Q, epsilon=epsilon, M2=M2, rho_c=rho_c, kappa=kappa,
        scale_factor=scale_factor)


def action_symbolic_checks(*, branch_sign: int = -1,
                           q_rate_scale: Any = "1") -> Dict[str, Dict[str, Any]]:
    """Vary the actual lapse action, then test the epsilon reconstruction.

    ``branch_sign=-1`` means the declared ``-sqrt((d Psi)^2)-V`` branch.
    ``q_rate_scale`` is a negative-control multiplier; setting it to ``2``
    intentionally omits the exact candidate rate normalization.
    """
    if isinstance(branch_sign, bool) or branch_sign not in (-1, 1):
        raise ValueError("branch_sign must be -1 or +1")
    if isinstance(q_rate_scale, bool):
        raise ValueError("q_rate_scale must be finite")
    rate_scale = sp.Rational(str(finite(q_rate_scale, "q_rate_scale")))

    a, N, adot, addot, Ndot = sp.symbols(
        "a N adot addot Ndot", positive=True)
    B, wdot, W, Wdot, M2, kappa = sp.symbols(
        "B wdot W Wdot M2 kappa", positive=True)
    Psi, Psidot = sp.symbols("Psi Psidot", real=True)
    Vfun = sp.Function("V")
    Efun = sp.Function("E")
    n = B / a ** 3
    nvar = sp.symbols("nvar", positive=True)
    energy = Efun(nvar, W).subs(nvar, n)
    En = sp.diff(Efun(nvar, W), nvar).subs(nvar, n)
    rho_m = wdot ** 2 / (2 * N ** 2) + energy
    pressure_m = wdot ** 2 / (2 * N ** 2) + n * En - energy

    # The homogeneous cuscuton term is s*a^3*Psi_dot/N times N, and its
    # boundary term is dropped exactly: s*a^3*Psi_dot -> -3*s*a^2*adot*Psi.
    L = (-3 * M2 * a * adot ** 2 / N + 3 * M2 * kappa * N * a
         - 3 * branch_sign * a ** 2 * adot * Psi - N * a ** 3 * Vfun(Psi)
         + a ** 3 * wdot ** 2 / (2 * N) - N * a ** 3 * energy)

    def d_dt(expr):
        return (sp.diff(expr, a) * adot + sp.diff(expr, adot) * addot
                + sp.diff(expr, N) * Ndot + sp.diff(expr, Psi) * Psidot
                + sp.diff(expr, W) * Wdot)

    H = adot / (N * a)
    Hdot = addot / (N ** 2 * a) - adot * Ndot / (N ** 3 * a) - H ** 2
    lapse_el = sp.diff(L, N) / a ** 3
    psi_el = (sp.diff(L, Psi) - d_dt(sp.diff(L, Psidot))) / (N * a ** 3)
    scale_el = (sp.diff(L, a) - d_dt(sp.diff(L, adot))) / (3 * N * a ** 2)

    # These right sides are independently obtained from lapse/scale
    # variations and the Legendre pressure at fixed baryon number.
    friedmann = 3 * M2 * (H ** 2 + kappa / a ** 2) - rho_m - Vfun(Psi)
    psi_equation = -3 * branch_sign * H - sp.diff(Vfun(Psi), Psi)
    scale_equation = (M2 * (2 * Hdot + 3 * H ** 2 + kappa / a ** 2)
                      + pressure_m - Vfun(Psi)
                      + branch_sign * Psidot / N)
    raychaudhuri = (2 * M2 * (Hdot - kappa / a ** 2) + rho_m
                    + pressure_m + branch_sign * Psidot / N)

    q, eps, rc, alpha, Q, aa = sp.symbols(
        "q epsilon rho_c alpha Q a0", positive=True, real=True)
    x = sp.cos(q) ** 2
    Psi_q = ((1 + eps) * q + sp.sin(2 * q) / 2) / alpha
    V_q = -rc * x * (x + eps)
    H_q = alpha * rc * sp.sin(2 * q) / 3
    qdot = rate_scale * alpha / (1 + eps) * (
        Q - 2 * M2 * kappa / aa ** 2)
    M2_rule = 3 / (4 * rc * alpha ** 2)
    V_Psi = sp.diff(V_q, q) / sp.diff(Psi_q, q)
    candidate_rho = (1 + eps) * rc * x + 3 * M2 * kappa / aa ** 2

    # Candidate rows use derivatives of the independently declared map and
    # potential; no row is formed by subtracting an expression from itself.
    candidate = {
        "candidate_cuscuton_EL_branch": V_Psi + 3 * branch_sign * H_q,
        "candidate_key_identity": (sp.diff(Psi_q, q)
            - 2 * M2 * sp.diff(H_q, q) - (1 + eps) / alpha),
        "candidate_friedmann_curved": (
            3 * M2 * (H_q ** 2 + kappa / aa ** 2)
            - candidate_rho - V_q),
        "candidate_raychaudhuri_curved": (
            2 * M2 * (sp.diff(H_q, q) * qdot - kappa / aa ** 2)
            + Q + branch_sign * sp.diff(Psi_q, q) * qdot),
        "candidate_constraint_derivative": (
            -3 * H_q * Q + (1 + eps) * rc * sp.sin(2 * q) * qdot
            + 6 * M2 * kappa * H_q / aa ** 2),
        "candidate_flat_reduction": (
            H_q ** 2 - (candidate_rho.subs(kappa, 0)
            / (3 * M2 * (1 + eps)))
            * (1 - candidate_rho.subs(kappa, 0) / ((1 + eps) * rc))),
        "candidate_bounce_H_zero": H_q.subs(q, 0),
        "candidate_bounce_Hdot_positive": (
            sp.diff(H_q, q).subs(q, 0)
            * qdot.subs({q: 0, kappa: 0})
            - Q / (2 * M2 * (1 + eps))),
    }
    # Apply the alpha relation only after differentiation, retaining the bare
    # M2 curvature terms until the final simplification.
    candidate = {name: value.subs(M2, M2_rule) for name, value in candidate.items()}

    action_rows = {
        "lapse_variation_from_actual_action": lapse_el - friedmann,
        "cuscuton_EL_from_actual_action": psi_el - psi_equation,
        "scale_variation_from_actual_action": scale_el - scale_equation,
        "raychaudhuri_after_lapse_elimination": scale_el - raychaudhuri
        - friedmann,
    }
    return _rows({**action_rows, **candidate})


def flat_reduction_symbolic_checks() -> Dict[str, Dict[str, Any]]:
    """Derive the flat density law and its cosmological Planck-mass shift."""
    eps, rc, M2, rho = sp.symbols("epsilon rho_c M2 rho", positive=True)
    C = 1 + eps
    q = sp.symbols("q", real=True)
    alpha = sp.sqrt(3 / (4 * M2 * rc))
    Hq = alpha * rc * sp.sin(2 * q) / 3
    rhoq = C * rc * sp.cos(q) ** 2
    H2_from_rho = rho / (3 * M2 * C) * (1 - rho / (C * rc))
    rows = {
        "flat_H2_from_q_vs_rho_law": Hq ** 2 - (
            rhoq / (3 * M2 * C) * (1 - rhoq / (C * rc))),
        "bounce_density_from_q": rhoq.subs(q, 0) - C * rc,
        "bounce_H_from_q": Hq.subs(q, 0),
        "flat_bounce_zero_of_H2": H2_from_rho.subs(rho, C * rc),
        "flat_low_density_slope_from_q_limit": sp.limit(
            Hq ** 2 / rhoq, q, sp.pi / 2, dir="-")
        - 1 / (3 * M2 * C),
    }
    return _rows(rows)


def curved_law_symbolic_checks() -> Dict[str, Dict[str, Any]]:
    """Check the curved law in terms of F with bare M retained."""
    eps, rc, M2 = sp.symbols("epsilon rho_c M2", positive=True)
    rho, curvature = sp.symbols("rho curvature", real=True)
    C = 1 + eps
    F = rho - 3 * M2 * curvature
    H2 = F / (3 * M2 * C) * (1 - F / (C * rc))
    rows = {
        "constraint_F_uses_bare_M": sp.diff(F, curvature) + 3 * M2,
        "low_density_F_slope": sp.diff(H2, rho).subs({rho: 0, curvature: 0})
        - 1 / (3 * M2 * C),
        "low_density_curvature_coefficient": sp.diff(H2, curvature).subs(
            {rho: 0, curvature: 0}) + 1 / C,
        "curvature_does_not_use_M_cosmological": sp.diff(F, M2)
        + 3 * curvature,
    }
    return _rows(rows)


def cost_theorem_symbolic_checks() -> Dict[str, Dict[str, Any]]:
    """Derive the smooth-maximum vacuum limit and the epsilon cost."""
    M2, K, rho, eps, rc = sp.symbols(
        "M2 K rho epsilon rho_c", positive=True)
    ratio = 1 / (3 * (M2 + sp.Rational(3, 2) / K))
    friedmann_ratio = 1 / (3 * M2 + sp.Rational(9, 2) / K)
    # The family value is obtained from the actual q-map/potential endpoint
    # curvature, using alpha^2=3/(4*M2*rho_c), rather than asserted as an
    # independent target.
    K_from_map = 2 * rc * (sp.Rational(3, 4) / (M2 * rc)) / eps
    K_family = sp.Rational(3, 2) / (M2 * eps)
    rows = {
        "smooth_maximum_vacuum_limit": ratio - friedmann_ratio,
        "family_endpoint_curvature": (
            K_from_map - K_family),
        "family_gravity_ratio": (
            M2 / (M2 + sp.Rational(3, 2) / K_family) - 1 / (1 + eps)),
        "family_flat_coefficient": (
            ratio.subs(K, K_family) - 1 / (3 * M2 * (1 + eps))),
    }
    return _rows(rows)


def regularity_symbolic_checks() -> Dict[str, Dict[str, Any]]:
    """Check the positive-epsilon inverse map and the epsilon-zero boundary."""
    q, eps, alpha, rc = sp.symbols("q epsilon alpha rho_c", positive=True)
    Psi = ((1 + eps) * q + sp.sin(2 * q) / 2) / alpha
    V = -rc * sp.cos(q) ** 2 * (sp.cos(q) ** 2 + eps)
    Psi_q = sp.diff(Psi, q)
    V_Psi = sp.diff(V, q) / Psi_q
    V_PsiPsi = sp.diff(V_Psi, q) / Psi_q
    endpoint = sp.pi / 2
    delta = sp.symbols("delta", positive=True)
    Psi0 = Psi.subs(eps, 0)
    gap0 = sp.limit(Psi0.subs(q, endpoint) - Psi0.subs(q, endpoint - delta),
                    delta, 0, dir="+")
    gap0_coefficient = sp.limit(
        (Psi0.subs(q, endpoint) - Psi0.subs(q, endpoint - delta)) / delta ** 3,
        delta, 0, dir="+")
    V0_coefficient = sp.limit(V.subs(eps, 0).subs(q, endpoint - delta)
                              / delta ** 4, delta, 0, dir="+")
    rows = {
        "positive_map_derivative": Psi_q - (eps + 2 * sp.cos(q) ** 2) / alpha,
        "positive_map_endpoint_derivative": Psi_q.subs(q, endpoint) - eps / alpha,
        "potential_gradient": V_Psi - alpha * rc * sp.sin(2 * q),
        "endpoint_V_Psi_zero": sp.limit(V_Psi, q, endpoint, dir="-"),
        "endpoint_curvature_K": (sp.limit(V_PsiPsi, q, endpoint, dir="-")
                                  + 2 * alpha ** 2 * rc / eps),
        "epsilon_zero_gradient": Psi_q.subs({q: endpoint, eps: 0}),
        "epsilon_zero_cubic_gap": gap0,
        "epsilon_zero_cubic_coefficient": gap0_coefficient
        - 2 / (3 * alpha),
        "epsilon_zero_quartic_potential": V0_coefficient + rc,
    }
    return _rows(rows)


def endpoint_inverse_derivative(*, epsilon: Any = "0.1", M2: Any = "1",
                                rho_c: Any = "1", dps: int = 80) -> Dict[str, Any]:
    """Use a high-precision finite-difference inverse derivative at q=pi/2."""
    if isinstance(dps, bool) or not isinstance(dps, int) or not 60 <= dps <= 160:
        raise ValueError("dps must be an integer in [60,160]")
    eps, m2, rc, alpha = _construction(epsilon, M2, rho_c)
    if eps <= 0:
        return {"epsilon": decimal(eps), "valid": False,
                "endpoint_Psi_q": decimal(eps / alpha),
                "reason": "nonpositive endpoint derivative"}
    with mp.workdps(dps):
        # Reparse inside the precision context so alpha is not inherited from
        # the caller's default precision.
        eps, m2, rc, alpha = _construction(epsilon, M2, rho_c)
        qend = mp.pi / 2
        h = mp.power(10, -(dps // 3))
        psi = lambda z: ((1 + eps) * z + mp.sin(2 * z) / 2) / alpha
        dpsi = mp.diff(psi, qend)
        inverse = 1 / dpsi
        finite_difference = h / (psi(qend) - psi(qend - h))
        rel = abs(finite_difference - inverse) / abs(inverse)
        return {
            "epsilon": decimal(eps), "M2": decimal(m2), "rho_c": decimal(rc),
            "dps": dps, "h": decimal(h), "endpoint_Psi_q": decimal(dpsi),
            "endpoint_dq_dPsi": decimal(inverse),
            "finite_difference_dq_dPsi": decimal(finite_difference),
            "finite_difference_relative_error": decimal(rel),
            "product_check": decimal(dpsi * inverse),
            "passed": bool(rel < mp.power(10, -(dps // 2))),
            "valid": True,
        }


def vacuum_limit_check(*, epsilon: Any = "0.1", M2: Any = "1",
                       rho_c: Any = "1", dps: int = 80) -> Dict[str, Any]:
    """Independent high-precision q->pi/2 limit and smooth-maximum theorem."""
    if isinstance(dps, bool) or not isinstance(dps, int) or not 60 <= dps <= 160:
        raise ValueError("dps must be an integer in [60,160]")
    eps, m2, rc, _ = _construction(epsilon, M2, rho_c)
    if eps <= 0:
        return {"epsilon": decimal(eps), "valid": False,
                "reason": "nonpositive epsilon is outside regular family"}
    with mp.workdps(dps):
        eps, m2, rc, alpha = _construction(epsilon, M2, rho_c)
        qend = mp.pi / 2
        delta = mp.power(10, -(dps // 3))
        q = qend - delta
        # These are independently written maps, not calls to the module's
        # point-evaluation helpers.
        H = alpha * rc * mp.sin(2 * q) / 3
        rho = (1 + eps) * rc * mp.cos(q) ** 2
        direct_ratio = H ** 2 / rho
        exact_ratio = 1 / (3 * m2 * (1 + eps))
        direct_error = abs(direct_ratio - exact_ratio) / exact_ratio
        K = 2 * rc * alpha ** 2 / eps
        theorem_ratio = 1 / (3 * (m2 + 3 / (2 * K)))
        theorem_error = abs(theorem_ratio - exact_ratio) / exact_ratio
        return {
            "epsilon": decimal(eps), "M2": decimal(m2), "rho_c": decimal(rc),
            "dps": dps, "delta": decimal(delta),
            "direct_ratio_at_q": decimal(direct_ratio),
            "exact_flat_vacuum_ratio": decimal(exact_ratio),
            "direct_relative_error": decimal(direct_error),
            "K_endpoint": decimal(K),
            "theorem_ratio": decimal(theorem_ratio),
            "theorem_relative_error": decimal(theorem_error),
            "gravity_ratio_to_bare": decimal(m2 / (m2 + 3 / (2 * K))),
            "passed": bool(direct_error < mp.power(10, -(dps // 2))
                           and theorem_error < mp.power(10, -(dps // 2))),
            "valid": True,
        }


def epsilon_domain_control(*, epsilon: Any = "0.1", M2: Any = "1",
                           rho_c: Any = "1") -> Dict[str, Any]:
    """Classify epsilon, including the explicit singular and negative controls."""
    eps, m2, rc, alpha = _construction(epsilon, M2, rho_c)
    endpoint = eps / alpha
    if eps > 0:
        return {
            "epsilon": input_decimal(epsilon), "valid_regular_family": True,
            "endpoint_Psi_q": decimal(endpoint), "global_Psi_q_positive": True,
            "inverse_analytic_for_fixed_epsilon": True,
            "scope": "explicit mathematical input; not measured, fitted or NVG-derived",
        }
    if eps == 0:
        return {
            "epsilon": input_decimal(epsilon), "valid_regular_family": False,
            "endpoint_Psi_q": decimal(endpoint), "global_Psi_q_positive": False,
            "inverse_analytic_for_fixed_epsilon": False,
            "zero_gradient_obstruction": True,
            "scope": "original cusp boundary; excluded from regular family",
        }
    return {
        "epsilon": input_decimal(epsilon), "valid_regular_family": False,
        "endpoint_Psi_q": decimal(endpoint), "global_Psi_q_positive": False,
        "inverse_analytic_for_fixed_epsilon": False,
        "negative_epsilon_obstruction": True,
        "scope": "negative epsilon loses globally positive map derivative",
    }


def old_constraint_control(*, epsilon: Any = "0.1", rho_c: Any = "1",
                           q: Any = "0", M2: Any = "1", kappa: Any = "1",
                           scale_factor: Any = "1") -> Dict[str, Any]:
    """Reject the S2/original constraint when applied to the deformed map."""
    eps, m2, rc, _ = _construction(epsilon, M2, rho_c)
    qv = finite(q, "q")
    kap = finite(kappa, "kappa")
    a = finite(scale_factor, "scale_factor", positive=True)
    actual = density_constraint(qv, epsilon=eps, M2=m2, rho_c=rc,
                               kappa=kap, scale_factor=a)
    old = rc * mp.cos(qv) ** 2 + 3 * m2 * kap / a ** 2
    residual = actual - old
    return {
        "epsilon": input_decimal(epsilon), "q": decimal(qv),
        "M2": decimal(m2), "rho_c": decimal(rc),
        "kappa": decimal(kap), "scale_factor": decimal(a),
        "actual_minus_old_constraint": decimal(residual),
        "nonzero_for_positive_epsilon": bool(abs(residual) > 0),
        "accepted": False,
        "scope": "negative control: original epsilon=0 constraint is not reused",
    }


def omitted_rate_control(*, epsilon: Any = "0.1", Q: Any = "3",
                         M2: Any = "1", rho_c: Any = "1", kappa: Any = "1",
                         scale_factor: Any = "1") -> Dict[str, Any]:
    """Show the Raychaudhuri residual caused by omitting 1+epsilon."""
    eps, m2, rc, alpha = _construction(epsilon, M2, rho_c)
    qent = finite(Q, "Q", positive=True)
    kap = finite(kappa, "kappa")
    a = finite(scale_factor, "scale_factor", positive=True)
    force = qent - 2 * m2 * kap / a ** 2
    # Wrong rate = alpha*force, whereas the map requires alpha*force/(1+eps).
    wrong_qdot = alpha * force
    qsample = mp.mpf("0.37")
    residual = (2 * m2 * hubble_q_map(qsample, epsilon=eps, M2=m2,
                                      rho_c=rc) * wrong_qdot
                + force - psi_q_map(qsample, epsilon=eps, M2=m2,
                                    rho_c=rc) * wrong_qdot)
    return {
        "epsilon": input_decimal(epsilon), "wrong_qdot": decimal(wrong_qdot),
        "q_sample": decimal(qsample), "Q": decimal(qent),
        "M2": decimal(m2), "rho_c": decimal(rc),
        "kappa": decimal(kap), "scale_factor": decimal(a),
        "raychaudhuri_residual": decimal(residual),
        "nonzero": bool(abs(residual) > mp.mpf("1e-30")),
        "accepted": False,
        "scope": "negative control: omitted (1+epsilon) rate factor",
    }


def kinetic_substitution_control(*, epsilon: Any = "0.1", X: Any = "1e-12",
                                 mu: Any = "1") -> Dict[str, Any]:
    """Reject adding epsilon*X/2 as a supposedly regularized kinetic term."""
    eps = finite(epsilon, "epsilon", positive=True)
    xx = finite(X, "X", positive=True)
    muc = finite(mu, "mu", positive=True)
    Xs, es, mus = sp.symbols("X epsilon mu", positive=True)
    Lx = -mus / (2 * sp.sqrt(Xs)) + es / 2
    Lxx = sp.diff(Lx, Xs)
    combination = sp.simplify(Lx + 2 * Xs * Lxx)
    lx_numeric = -muc / (2 * mp.sqrt(xx)) + eps / 2
    return {
        "epsilon": input_decimal(epsilon), "X": decimal(xx),
        "mu": decimal(muc),
        "L_X": decimal(lx_numeric),
        "L_X_negative_near_zero": bool(lx_numeric < 0),
        "L_X_plus_2X_L_XX": str(combination),
        "combination_residual": str(sp.simplify(combination - es / 2)),
        "combination_equals_epsilon_over_2": bool(
            sp.simplify(combination - es / 2) == 0),
        "same_action": False, "accepted": False,
        "scope": "negative control; epsilon changes map/potential, not cuscuton kinetic term",
    }


def closed_barotropic_theorem(*, w: Any = "0") -> Dict[str, Any]:
    """State and check the monotone F theorem for P=w*rho, w>=0."""
    wv = finite(w, "w", nonnegative=True)
    m = 3 * (1 + wv)
    return {
        "w": decimal(wv), "m": decimal(m), "P_nonnegative": bool(wv >= 0),
        "F_derivative": "-3*(Q-2*M2*kappa/a^2)/a",
        "F_decreases_when": "Q>2*M2*kappa/a^2",
        "constant_w_qrate_positive_on_A_le_1": bool(3 * (1 + wv) / 2 > 1),
        "scope": "barotropic mathematical theorem; no S2 EOS reconstruction",
    }


def closed_barotropic_symbolic_checks() -> Dict[str, Dict[str, Any]]:
    """Differentiate F(a) using continuity, without importing the S2 EOS."""
    a, M2, kappa, Q = sp.symbols("a M2 kappa Q", positive=True)
    rho = sp.Function("rho")(a)
    F = rho - 3 * M2 * kappa / a ** 2
    derivative_from_conservation = sp.diff(F, a).subs(
        sp.diff(rho, a), -3 * Q / a)
    target = -3 / a * (Q - 2 * M2 * kappa / a ** 2)
    return _rows({
        "F_monotone_derivative_identity": derivative_from_conservation - target,
    })


def _cycle_A_low(w: float, R: float, epsilon: float) -> float:
    from scipy.optimize import brentq
    m = 3 * (1 + w)
    C = (1 + epsilon) * R

    def f(A):
        return A ** (-m) - A ** (-2) - C

    return float(brentq(f, 1e-6, 1 - 1e-12, xtol=2e-14, rtol=1e-14))


def constant_w_cycle(*, w: Any = "0", epsilon: Any = "0.1",
                     rho_c_ratio: Any = "1", quadrature_rtol: Any = "3e-11",
                     ode_rtol: Any = "2e-10", ode_atol: Any = "2e-12",
                     samples: int = 33) -> Dict[str, Any]:
    """Compute one manufactured closed constant-w cycle two independent ways."""
    import numpy as np
    from scipy.integrate import quad, solve_ivp
    from scipy.optimize import brentq

    wv = float(finite(w, "w", nonnegative=True))
    eps = float(finite(epsilon, "epsilon", positive=True))
    R = float(finite(rho_c_ratio, "rho_c_ratio", positive=True))
    qr = float(finite(quadrature_rtol, "quadrature_rtol", positive=True))
    orr = float(finite(ode_rtol, "ode_rtol", positive=True))
    oa = float(finite(ode_atol, "ode_atol", positive=True))
    if not 0 <= wv <= 1 / 3:
        raise ValueError("constant-w control requires 0<=w<=1/3")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 17:
        raise ValueError("samples must be an integer >=17")
    m = 3 * (1 + wv)
    C = (1 + eps) * R
    A_low = _cycle_A_low(wv, R, eps)

    def F(A):
        return A ** (-m) - A ** (-2)

    def A_of_q(q):
        target = C * math.cos(q) ** 2
        if target >= C * (1 - 2e-14):
            return A_low
        if target <= C * 2e-15:
            return 1.0
        return brentq(lambda A: F(A) - target, A_low, 1.0,
                      xtol=2e-14, rtol=1e-14)

    def qprime_at(q):
        A = A_of_q(q)
        rho = A ** (-m)
        Q = (1 + wv) * rho
        return 3 / (2 * math.sqrt(R) * (1 + eps)) * (
            Q - 2 / (3 * A ** 2))

    def integrand(q):
        value = qprime_at(q)
        if not math.isfinite(value) or value <= 0:
            raise ArithmeticError("nonpositive q rate in quadrature")
        return 1 / value

    coarse, coarse_err = quad(integrand, 0.0, math.pi,
                              epsabs=max(qr * 30, 2e-8),
                              epsrel=max(qr * 30, 2e-8),
                              points=[math.pi / 2], limit=160)
    fine, fine_err = quad(integrand, 0.0, math.pi,
                          epsabs=qr, epsrel=qr,
                          points=[math.pi / 2], limit=240)

    sqrtR = math.sqrt(R)

    def rhs(_, state):
        A, q = state
        if not math.isfinite(A) or A <= 0:
            raise ArithmeticError("invalid scale factor in direct ODE")
        rho = A ** (-m)
        Q = (1 + wv) * rho
        H = sqrtR * math.sin(2 * q) / 2
        qdot = 3 / (2 * sqrtR * (1 + eps)) * (Q - 2 / (3 * A ** 2))
        return (A * H, qdot)

    def event(_, state):
        return state[1] - math.pi

    event.terminal = True
    event.direction = 1
    direct = solve_ivp(rhs, (0.0, max(2 * fine + 1, 4.0)),
                       (A_low, 0.0), method="DOP853", rtol=orr, atol=oa,
                       max_step=0.025, events=event, dense_output=True)
    if not direct.success or len(direct.t_events[0]) != 1:
        raise ArithmeticError("independent direct constant-w ODE failed")
    ode_period = float(direct.t_events[0][0])
    A_end, q_end = (float(v) for v in direct.y_events[0][0])

    max_constraint = 0.0
    max_friedmann = 0.0
    max_ray = 0.0
    max_continuity = 0.0
    max_traj = 0.0
    min_psi_q_alpha = float("inf")
    min_psi_dot = float("inf")
    min_qrate = float("inf")
    min_pressure = float("inf")
    turnaround_psidot = eps / (1 + eps) * ((1 + wv) - 2 / 3)
    max_F_A = -float("inf")
    grid = np.linspace(0.0, math.pi, max(samples, 33))
    for q in grid:
        A = A_of_q(float(q))
        rho = A ** (-m)
        Q = (1 + wv) * rho
        pressure = wv * rho
        H = sqrtR * math.sin(2 * q) / 2
        qp = 3 / (2 * sqrtR * (1 + eps)) * (Q - 2 / (3 * A ** 2))
        psiq_alpha = eps + 2 * math.cos(q) ** 2
        psidot = psiq_alpha / (1 + eps) * (Q - 2 / (3 * A ** 2))
        Hdot = sqrtR * math.cos(2 * q) * qp
        F_A = -m * A ** (-m - 1) + 2 * A ** (-3)
        constraint = (rho - A ** (-2)) - C * math.cos(q) ** 2
        friedmann = H ** 2 + A ** (-2) - rho + R * math.cos(q) ** 2 * (
            math.cos(q) ** 2 + eps)
        ray = 2 / 3 * (Hdot - A ** (-2)) + Q - psidot
        continuity = -m * A ** (-m - 1) * A * H + 3 * H * Q
        max_constraint = max(max_constraint, abs(constraint))
        max_friedmann = max(max_friedmann, abs(friedmann))
        max_ray = max(max_ray, abs(ray))
        max_continuity = max(max_continuity, abs(continuity))
        min_psi_q_alpha = min(min_psi_q_alpha, psiq_alpha)
        min_psi_dot = min(min_psi_dot, psidot)
        min_qrate = min(min_qrate, qp)
        min_pressure = min(min_pressure, pressure)
        max_F_A = max(max_F_A, F_A)

    # Compare the direct trajectory against the independently inverted
    # constraint curve at times sampled from the ODE's own dense solution.
    for tau in np.linspace(0.0, ode_period, max(samples, 33)):
        A_ode, q_ode = (float(v) for v in direct.sol(float(tau)))
        max_traj = max(max_traj, abs(A_ode - A_of_q(q_ode)))

    period_ref = max(1.0, abs(fine))
    quadrature_refinement = abs(coarse - fine) / period_ref
    ode_quadrature = abs(ode_period - fine) / period_ref
    end_scale = max(1.0, abs(A_low))
    passed = bool(
        A_low > 0 and A_low < 1 and min_qrate > 0 and min_psi_dot > 0
        and min_psi_q_alpha >= eps and min_pressure >= 0 and max_F_A < 0
        and max_constraint < 3e-10 and max_friedmann < 3e-10
        and max_ray < 3e-10 and max_continuity < 3e-10
        and quadrature_refinement < 3e-7 and ode_quadrature < 3e-6
        and abs(A_end - A_low) / end_scale < 3e-7
        and abs(q_end - math.pi) < 3e-8 and max_traj < 3e-7
    )
    return {
        "w": wv, "epsilon": eps, "rho_c_over_rho_ref": R,
        "m": m, "C=(1+epsilon)*R": C, "A_low": A_low, "A_high": 1.0,
        "period_quadrature_coarse": coarse,
        "period_quadrature_fine": fine,
        "quadrature_error_estimate": fine_err,
        "quadrature_refinement_relative": quadrature_refinement,
        "period_direct_ode": ode_period,
        "ode_vs_quadrature_relative": ode_quadrature,
        "A_end_direct_ode": A_end, "q_end_direct_ode": q_end,
        # Keep the algebraic constraint residual distinct from the direct ODE
        # trajectory's distance to that curve; they test different objects.
        "algebraic_curve_constraint_max_abs": max_constraint,
        "direct_ode_trajectory_curve_max_abs": max_traj,
        "trajectory_constraint_curve_max_abs": max_traj,
        "max_constraint_residual": max_constraint,
        "max_friedmann_residual": max_friedmann,
        "max_raychaudhuri_residual": max_ray,
        "max_continuity_residual": max_continuity,
        "min_dimensionless_Psi_q": min_psi_q_alpha,
        "min_dimensionless_Psi_dot": min_psi_dot,
        "turnaround_dimensionless_Psi_dot": turnaround_psidot,
        "turnaround_dimensionless_Psi_q": eps,
        "min_q_rate": min_qrate, "min_pressure": min_pressure,
        "max_F_derivative": max_F_A,
        "samples": max(samples, 33), "passed": passed,
        "inputs_manufactured": True,
        "normalization": {
            "A": "a/a_ref",
            "a_ref": "sqrt(3*M2/rho_ref)",
            "tau": "t/a_ref",
            "Psi_dot_dimensionless": "physical Psi_dot/rho_ref",
            "rho_c": "R*rho_ref",
        },
        "scope": "formal constant-w background control; algebraic curve and direct ODE trajectory are separate; not NVG matter or cosmic-age prediction",
    }


def point_identity_check(*, epsilon: Any = "0.1", q: Any = "0.37",
                         Q: Any = "2.4", M2: Any = "3", rho_c: Any = "5",
                         kappa: Any = "1", scale_factor: Any = "2.1") -> Dict[str, Any]:
    """Independent high-precision residuals at one curved point."""
    with mp.workdps(90):
        eps, m2, rc, alpha = _construction(epsilon, M2, rho_c)
        qv = finite(q, "q")
        Qv = finite(Q, "Q", positive=True)
        kap = finite(kappa, "kappa")
        a = finite(scale_factor, "scale_factor", positive=True)
        H = alpha * rc * mp.sin(2 * qv) / 3
        Hq = 2 * alpha * rc * mp.cos(2 * qv) / 3
        psiq = (eps + 2 * mp.cos(qv) ** 2) / alpha
        qd = alpha / (1 + eps) * (Qv - 2 * m2 * kap / a ** 2)
        rho = (1 + eps) * rc * mp.cos(qv) ** 2 + 3 * m2 * kap / a ** 2
        V = -rc * mp.cos(qv) ** 2 * (mp.cos(qv) ** 2 + eps)
        residuals = {
            "friedmann": 3 * m2 * (H ** 2 + kap / a ** 2) - rho - V,
            "raychaudhuri": 2 * m2 * (Hq * qd - kap / a ** 2)
            + Qv - psiq * qd,
            "cuscuton_gradient": (
                mp.diff(lambda z: potential_map(z, epsilon=eps, M2=m2,
                                                 rho_c=rc), qv) /
                mp.diff(lambda z: psi_map(z, epsilon=eps, M2=m2,
                                           rho_c=rc), qv)
                - 3 * H),
            "constraint_derivative": -3 * H * Qv
            + (1 + eps) * rc * mp.sin(2 * qv) * qd
            + 6 * m2 * kap * H / a ** 2,
            "key_identity": psiq - 2 * m2 * Hq - (1 + eps) / alpha,
        }
        scale = max(mp.mpf(1), *(abs(v) for v in residuals.values()))
        max_scaled = max(abs(v) / scale for v in residuals.values())
        return {
            "epsilon": decimal(eps), "q": decimal(qv), "Q": decimal(Qv),
            "M2": decimal(m2), "rho_c": decimal(rc), "kappa": decimal(kap),
            "scale_factor": decimal(a),
            "dps": 90,
            "max_scaled_residual": decimal(max_scaled),
            "residuals": {name: decimal(value)
                          for name, value in residuals.items()},
            "passed": bool(max_scaled < mp.mpf("1e-70")),
        }


def _number_is_finite_and_below(value: Any, bound: Any) -> bool:
    if isinstance(value, bool) or isinstance(bound, bool):
        return False
    try:
        number = mp.mpf(value)
        limit = mp.mpf(bound)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(mp.isfinite(number) and mp.isfinite(limit)
                and limit > 0 and number >= 0 and number < limit)


def _required_numeric_fields(record: Any, names: Any) -> Any:
    """Parse a required numerical schema, returning ``None`` on any defect."""
    if not isinstance(record, dict):
        return None
    parsed = {}
    for name in names:
        if name not in record:
            return None
        number = _safe_finite(record[name])
        if number is None:
            return None
        parsed[name] = number
    return parsed


def _point_evidence_passes(point: Any) -> bool:
    if not isinstance(point, dict):
        return False
    if set(point) != set(REQUIRED_POINT_KEYS):
        return False
    dps = point.get("dps")
    if (not isinstance(dps, int) or isinstance(dps, bool) or dps != 90
            or point.get("passed") is not True):
        return False
    values = _required_numeric_fields(point, POINT_NUMERIC_FIELDS)
    residuals = point.get("residuals")
    if values is None or not isinstance(residuals, dict):
        return False
    if set(residuals) != set(POINT_RESIDUAL_NAMES):
        return False
    residual_values = _required_numeric_fields(residuals, POINT_RESIDUAL_NAMES)
    if residual_values is None:
        return False
    scale = max(mp.mpf(1), *(abs(value)
                             for value in residual_values.values()))
    expected_max = max(abs(value) / scale
                       for value in residual_values.values())
    return bool(
        values["epsilon"] > 0 and values["Q"] > 0
        and values["M2"] > 0 and values["rho_c"] > 0
        and values["scale_factor"] > 0
        and _reported_matches(values["max_scaled_residual"], expected_max)
        and _number_is_finite_and_below(values["max_scaled_residual"],
                                        "1e-70"))


def _endpoint_inverse_evidence_passes(item: Any, expected_dps: int) -> bool:
    if not isinstance(item, dict):
        return False
    if set(item) != set(REQUIRED_ENDPOINT_INVERSE_KEYS):
        return False
    dps = item.get("dps")
    if (not isinstance(dps, int) or isinstance(dps, bool)
            or dps != expected_dps):
        return False
    if item.get("valid") is not True or item.get("passed") is not True:
        return False
    values = _required_numeric_fields(item, ENDPOINT_INVERSE_NUMERIC_FIELDS)
    if values is None:
        return False
    error_bound = "1e-40" if expected_dps == 80 else "1e-60"
    return bool(
        values["epsilon"] > 0 and values["M2"] > 0 and values["rho_c"] > 0
        and values["h"] > 0 and values["endpoint_Psi_q"] > 0
        and values["endpoint_dq_dPsi"] > 0
        and values["finite_difference_dq_dPsi"] > 0
        and _number_is_finite_and_below(
            values["finite_difference_relative_error"], error_bound)
        and _reported_matches(values["product_check"], 1)
    )


def _vacuum_limit_evidence_passes(item: Any, expected_dps: int) -> bool:
    if not isinstance(item, dict):
        return False
    if set(item) != set(REQUIRED_VACUUM_LIMIT_KEYS):
        return False
    dps = item.get("dps")
    if (not isinstance(dps, int) or isinstance(dps, bool)
            or dps != expected_dps):
        return False
    if item.get("valid") is not True or item.get("passed") is not True:
        return False
    values = _required_numeric_fields(item, VACUUM_LIMIT_NUMERIC_FIELDS)
    if values is None:
        return False
    error_bound = "1e-40" if expected_dps == 80 else "1e-60"
    return bool(
        values["epsilon"] > 0 and values["M2"] > 0 and values["rho_c"] > 0
        and values["delta"] > 0 and values["direct_ratio_at_q"] > 0
        and values["exact_flat_vacuum_ratio"] > 0
        and values["K_endpoint"] > 0 and values["theorem_ratio"] > 0
        and values["gravity_ratio_to_bare"] > 0
        and _number_is_finite_and_below(values["direct_relative_error"],
                                        error_bound)
        and _number_is_finite_and_below(values["theorem_relative_error"],
                                        "1e-40")
    )


def _cycle_evidence_passes(cycle: Any) -> bool:
    """Recompute acceptance from numerical evidence, not only ``passed``."""
    if not isinstance(cycle, dict) or cycle.get("passed") is not True:
        return False
    try:
        if set(cycle) != set(REQUIRED_CYCLE_KEYS):
            return False
        values = _required_numeric_fields(cycle, CYCLE_NUMERIC_FIELDS)
        if values is None:
            return False
        if any(not isinstance(cycle.get(name), bool)
               for name in CYCLE_BOOL_FIELDS):
            return False
        if cycle.get("inputs_manufactured") is not True:
            return False
        normalization = cycle.get("normalization")
        if not isinstance(normalization, dict):
            return False
        if set(normalization) != set(REQUIRED_CYCLE_NORMALIZATION):
            return False
        if any(normalization.get(name) != value
               for name, value in REQUIRED_CYCLE_NORMALIZATION.items()):
            return False
        scope = cycle.get("scope")
        if (not isinstance(scope, str)
                or "algebraic curve" not in scope
                or "direct ODE trajectory" not in scope
                or "separate" not in scope):
            return False
        samples = cycle.get("samples")
        if isinstance(samples, bool) or not isinstance(samples, int):
            return False
        if samples < 33:
            return False
        A_low = values["A_low"]
        A_high = values["A_high"]
        min_q = values["min_dimensionless_Psi_q"]
        min_dot = values["min_dimensionless_Psi_dot"]
        min_rate = values["min_q_rate"]
        max_fa = values["max_F_derivative"]
        pressure = values["min_pressure"]
        q_end = values["q_end_direct_ode"]
        A_end = values["A_end_direct_ode"]
        traj = values["direct_ode_trajectory_curve_max_abs"]
        eps = values["epsilon"]
        w = values["w"]
        R = values["rho_c_over_rho_ref"]
        fine = values["period_quadrature_fine"]
        coarse = values["period_quadrature_coarse"]
        direct = values["period_direct_ode"]
        period_ref = max(mp.mpf(1), abs(fine))
        expected_refinement = abs(coarse - fine) / period_ref
        expected_ode_difference = abs(direct - fine) / period_ref
        expected_turnaround_dot = eps / (1 + eps) * (
            1 + w - mp.mpf(2) / 3)
    except Exception:
        return False
    return bool(
        0 <= w <= mp.mpf(1) / 3 and eps > 0 and R > 0
        and _reported_matches(values["m"], 3 * (1 + w), relative="1e-12")
        and _reported_matches(values["C=(1+epsilon)*R"], (1 + eps) * R,
                              relative="1e-12")
        and 0 < A_low < A_high
        and _reported_matches(A_high, 1, relative="1e-12")
        and min_q >= eps and min_dot > 0 and min_rate > 0
        and max_fa < 0 and pressure >= 0
        and values["turnaround_dimensionless_Psi_q"] > 0
        and _reported_matches(values["turnaround_dimensionless_Psi_q"], eps,
                              relative="1e-12")
        and _reported_matches(values["turnaround_dimensionless_Psi_dot"],
                              expected_turnaround_dot, relative="1e-12")
        and values["period_quadrature_coarse"] > 0
        and values["period_quadrature_fine"] > 0
        and values["period_direct_ode"] > 0
        and _number_is_finite_and_below(values["quadrature_error_estimate"],
                                        "3e-7")
        and _number_is_finite_and_below(
            values["quadrature_refinement_relative"], "3e-7")
        and _reported_matches(values["quadrature_refinement_relative"],
                              expected_refinement, relative="1e-12")
        and _number_is_finite_and_below(
            values["ode_vs_quadrature_relative"], "3e-6")
        and _reported_matches(values["ode_vs_quadrature_relative"],
                              expected_ode_difference, relative="1e-12")
        and _number_is_finite_and_below(values["max_constraint_residual"],
                                        "3e-10")
        and _number_is_finite_and_below(
            values["algebraic_curve_constraint_max_abs"], "3e-10")
        and _reported_matches(values["max_constraint_residual"],
                              values["algebraic_curve_constraint_max_abs"],
                              relative="1e-20")
        and _number_is_finite_and_below(values["max_friedmann_residual"],
                                        "3e-10")
        and _number_is_finite_and_below(values["max_raychaudhuri_residual"],
                                        "3e-10")
        and _number_is_finite_and_below(values["max_continuity_residual"],
                                        "3e-10")
        and _number_is_finite_and_below(traj, "3e-7")
        and _reported_matches(traj,
                              values["trajectory_constraint_curve_max_abs"],
                              relative="1e-20")
        and _reported_matches(traj,
                              values["direct_ode_trajectory_curve_max_abs"],
                              relative="1e-20")
        and _number_is_finite_and_below(
            abs(A_end - A_low) / max(mp.mpf(1), abs(A_low)), "3e-7")
        and _number_is_finite_and_below(abs(q_end - mp.pi), "3e-8")
    )


def _construction_evidence(result: Any) -> Any:
    """Validate the fixed construction metadata used by all controls."""
    if not isinstance(result, dict) or result.get("status") != STATUS:
        return None
    weight = result.get("evidence_weight")
    if (not isinstance(weight, int) or isinstance(weight, bool)
            or weight != EVIDENCE_WEIGHT):
        return None
    construction = result.get("construction")
    if not isinstance(construction, dict):
        return None
    names = ("epsilon", "rho_c_over_rho_ref", "M2", "rho_c",
             "kappa", "scale_factor")
    values = _required_numeric_fields(construction, names)
    if values is None:
        return None
    if (values["epsilon"] <= 0 or values["rho_c_over_rho_ref"] <= 0
            or values["M2"] <= 0 or values["rho_c"] <= 0
            or values["scale_factor"] <= 0):
        return None
    if not _reported_matches(values["rho_c"],
                             values["rho_c_over_rho_ref"]):
        return None
    if not (_reported_matches(values["M2"], 1)
            and _reported_matches(values["kappa"], 1)
            and _reported_matches(values["scale_factor"], 1)):
        return None
    if construction.get("not_NVG_derived") is not True:
        return None
    scope = result.get("scope")
    if not isinstance(scope, dict):
        return None
    if (not isinstance(scope.get("evidence_weight"), int)
            or isinstance(scope.get("evidence_weight"), bool)
            or scope.get("evidence_weight") != EVIDENCE_WEIGHT):
        return None
    for name in ("manufactured_controls", "counterfactual",
                 "no_perturbative_or_nonlinear_stability_claim",
                 "no_empirical_success_or_universal_theory_claim",
                 "bare_M_retained_in_curvature",
                 "gravity_and_rho_c_extra_assumptions"):
        if scope.get(name) is not True:
            return None
    return values


def _old_constraint_negative_passes(old: Any, eps: mp.mpf,
                                    construction: Dict[str, mp.mpf]) -> bool:
    if not isinstance(old, dict) or set(old) != set(REQUIRED_OLD_CONSTRAINT_KEYS):
        return False
    values = _required_numeric_fields(old, OLD_CONSTRAINT_NUMERIC_FIELDS)
    if values is None:
        return False
    if (old.get("accepted") is not False
            or old.get("nonzero_for_positive_epsilon") is not True
            or not isinstance(old.get("scope"), str)):
        return False
    if not _reported_matches(values["epsilon"], eps, relative="1e-12"):
        return False
    for name in ("M2", "rho_c", "kappa", "scale_factor"):
        if not _reported_matches(values[name], construction[name],
                                 relative="1e-12"):
            return False
    expected = values["epsilon"] * values["rho_c"] * mp.cos(values["q"]) ** 2
    residual = values["actual_minus_old_constraint"]
    return bool(abs(residual) > mp.mpf("1e-30")
                and _reported_matches(residual, expected, relative="1e-12"))


def _omitted_rate_negative_passes(omitted: Any, eps: mp.mpf,
                                  construction: Dict[str, mp.mpf]) -> bool:
    if (not isinstance(omitted, dict)
            or set(omitted) != set(REQUIRED_OMITTED_RATE_KEYS)):
        return False
    values = _required_numeric_fields(omitted, OMITTED_RATE_NUMERIC_FIELDS)
    if values is None:
        return False
    if (omitted.get("accepted") is not False
            or omitted.get("nonzero") is not True
            or not isinstance(omitted.get("scope"), str)):
        return False
    if not _reported_matches(values["epsilon"], eps, relative="1e-12"):
        return False
    for name in ("M2", "rho_c", "kappa", "scale_factor"):
        if not _reported_matches(values[name], construction[name],
                                 relative="1e-12"):
            return False
    alpha = mp.sqrt(3 / (4 * values["M2"] * values["rho_c"]))
    force = (values["Q"]
             - 2 * values["M2"] * values["kappa"]
             / values["scale_factor"] ** 2)
    expected_qdot = alpha * force
    qsample = values["q_sample"]
    expected_residual = (
        2 * values["M2"] * hubble_q_map(
            qsample, epsilon=values["epsilon"], M2=values["M2"],
            rho_c=values["rho_c"]) * expected_qdot + force
        - psi_q_map(qsample, epsilon=values["epsilon"], M2=values["M2"],
                    rho_c=values["rho_c"]) * expected_qdot)
    residual = values["raychaudhuri_residual"]
    return bool(
        _reported_matches(values["wrong_qdot"], expected_qdot,
                          relative="1e-12")
        and abs(residual) > mp.mpf("1e-30")
        and _reported_matches(residual, expected_residual, relative="1e-12")
    )


def _kinetic_negative_passes(kinetic: Any, eps: mp.mpf) -> bool:
    if (not isinstance(kinetic, dict)
            or set(kinetic) != set(REQUIRED_KINETIC_NEGATIVE_KEYS)):
        return False
    values = _required_numeric_fields(kinetic, KINETIC_NEGATIVE_NUMERIC_FIELDS)
    if values is None:
        return False
    if (kinetic.get("accepted") is not False
            or kinetic.get("same_action") is not False
            or kinetic.get("L_X_negative_near_zero") is not True
            or kinetic.get("combination_equals_epsilon_over_2") is not True
            or kinetic.get("combination_residual") != "0"
            or kinetic.get("L_X_plus_2X_L_XX") != "epsilon/2"
            or not isinstance(kinetic.get("scope"), str)):
        return False
    expected = (-values["mu"] / (2 * mp.sqrt(values["X"]))
                + values["epsilon"] / 2)
    return bool(
        values["epsilon"] > 0 and values["X"] > 0 and values["mu"] > 0
        and _reported_matches(values["epsilon"], eps, relative="1e-12")
        and values["L_X"] < 0
        and _reported_matches(values["L_X"], expected, relative="1e-12")
    )


def _domain_control_passes(domain: Any, eps: mp.mpf,
                           construction: Dict[str, mp.mpf]) -> bool:
    if (not isinstance(domain, dict)
            or set(domain) != set(REQUIRED_DOMAIN_CONTROL_KEYS)):
        return False
    values = _required_numeric_fields(domain, DOMAIN_CONTROL_NUMERIC_FIELDS)
    if values is None:
        return False
    if (domain.get("valid_regular_family") is not True
            or domain.get("global_Psi_q_positive") is not True
            or domain.get("inverse_analytic_for_fixed_epsilon") is not True
            or not isinstance(domain.get("scope"), str)):
        return False
    alpha = mp.sqrt(3 / (4 * construction["M2"] * construction["rho_c"]))
    return bool(
        _reported_matches(values["epsilon"], eps, relative="1e-12")
        and values["endpoint_Psi_q"] > 0
        and _reported_matches(values["endpoint_Psi_q"], eps / alpha,
                              relative="1e-12")
    )


def acceptance_from_result(result: Any) -> bool:
    """Fail-closed aggregate gate used by the CLI and mutation tests."""
    try:
        construction = _construction_evidence(result)
        if construction is None:
            return False
        symbolic = result.get("symbolic")
        numerical = result.get("numerical")
        negative = result.get("negative_controls")
        if not isinstance(symbolic, dict) or not isinstance(numerical, dict):
            return False
        if set(symbolic) != set(SYMBOLIC_GROUP_NAMES):
            return False
        for name, required_rows in REQUIRED_SYMBOLIC_ROWS.items():
            if not rows_pass(symbolic.get(name), required_rows):
                return False
        required_numerical = {
            "curved_point", "endpoint_inverse_80", "endpoint_inverse_120",
            "vacuum_limit_80", "vacuum_limit_120", "cycles",
        }
        if set(numerical) != required_numerical:
            return False
        if not _point_evidence_passes(numerical.get("curved_point")):
            return False
        inverse_items = [numerical.get("endpoint_inverse_80"),
                         numerical.get("endpoint_inverse_120")]
        vacuum_items = [numerical.get("vacuum_limit_80"),
                        numerical.get("vacuum_limit_120")]
        if (not _endpoint_inverse_evidence_passes(inverse_items[0], 80)
                or not _endpoint_inverse_evidence_passes(inverse_items[1], 120)
                or {item.get("dps") for item in inverse_items} != {80, 120}):
            return False
        if (not _vacuum_limit_evidence_passes(vacuum_items[0], 80)
                or not _vacuum_limit_evidence_passes(vacuum_items[1], 120)
                or {item.get("dps") for item in vacuum_items} != {80, 120}):
            return False
        cycles = numerical.get("cycles")
        if not isinstance(cycles, list) or len(cycles) != 2:
            return False
        if not all(_cycle_evidence_passes(cycle) for cycle in cycles):
            return False
        cycle_ws = [_safe_finite(cycle["w"]) for cycle in cycles]
        required_ws = [_safe_finite(value) for value in REQUIRED_CYCLE_W]
        if (any(value is None for value in cycle_ws)
                or any(value is None for value in required_ws)
                or not all(any(abs(actual - expected) <= mp.mpf("1e-14")
                               for actual in cycle_ws)
                           for expected in required_ws)
                or abs(cycle_ws[0] - cycle_ws[1]) <= mp.mpf("1e-14")):
            return False
        for cycle in cycles:
            if (not _reported_matches(cycle["epsilon"],
                                      construction["epsilon"],
                                      relative="1e-12")
                    or not _reported_matches(
                        cycle["rho_c_over_rho_ref"],
                        construction["rho_c_over_rho_ref"],
                        relative="1e-12")):
                return False
        if not isinstance(negative, dict):
            return False
        required_negative = {"old_constraint", "omitted_rate_factor",
                             "epsilon_X_kinetic_substitution", "epsilon_domain"}
        if set(negative) != required_negative:
            return False
        eps = construction["epsilon"]
        if (not _old_constraint_negative_passes(
                negative.get("old_constraint"), eps, construction)
                or not _omitted_rate_negative_passes(
                    negative.get("omitted_rate_factor"), eps, construction)
                or not _kinetic_negative_passes(
                    negative.get("epsilon_X_kinetic_substitution"), eps)
                or not _domain_control_passes(
                    negative.get("epsilon_domain"), eps, construction)):
            return False
        return True
    except Exception:
        # Evidence is untrusted input to this helper.  A malformed row/field
        # must fail the aggregate gate, never become an acceptance exception.
        return False


def run_audit(*, epsilon: Any = "0.1", rho_c_ratio: Any = "1") -> Dict[str, Any]:
    """Run the bounded S3 evidence set and return a JSON-safe result."""
    eps = finite(epsilon, "epsilon")
    ratio = finite(rho_c_ratio, "rho_c_ratio", positive=True)
    eps_arg = epsilon if isinstance(epsilon, str) else input_decimal(eps)
    ratio_arg = rho_c_ratio if isinstance(rho_c_ratio, str) else input_decimal(ratio)
    if eps <= 0:
        return {
            "status": "invalid_or_failed_audit", "evidence_weight": EVIDENCE_WEIGHT,
            "mathematical_checks_passed": False,
            "regularity": epsilon_domain_control(epsilon=epsilon),
        }
    action = action_symbolic_checks()
    flat = flat_reduction_symbolic_checks()
    curved_law = curved_law_symbolic_checks()
    cost = cost_theorem_symbolic_checks()
    regularity = regularity_symbolic_checks()
    barotropic = closed_barotropic_symbolic_checks()
    point = point_identity_check(epsilon=eps_arg)
    inverse80 = endpoint_inverse_derivative(epsilon=eps_arg, dps=80)
    inverse120 = endpoint_inverse_derivative(epsilon=eps_arg, dps=120)
    vacuum80 = vacuum_limit_check(epsilon=eps_arg, dps=80)
    vacuum120 = vacuum_limit_check(epsilon=eps_arg, dps=120)
    cycles = [constant_w_cycle(w=w, epsilon=eps_arg, rho_c_ratio=ratio_arg)
              for w in ("0", "0.3333333333333333333333333333333333")]
    old = old_constraint_control(epsilon=eps_arg, rho_c=ratio_arg)
    omitted = omitted_rate_control(epsilon=eps_arg, rho_c=ratio_arg)
    kinetic = kinetic_substitution_control(epsilon=eps_arg)
    domain = epsilon_domain_control(epsilon=eps_arg, M2="1",
                                    rho_c=ratio_arg)
    result = {
        "status": STATUS, "evidence_weight": EVIDENCE_WEIGHT,
        "construction": {
            "epsilon": input_decimal(eps_arg),
            "rho_c_over_rho_ref": input_decimal(ratio_arg),
            "M2": "1", "rho_c": input_decimal(ratio_arg),
            "kappa": "1", "scale_factor": "1",
            "kinetic_term": "-sqrt((partial Psi)^2), unchanged",
            "potential_and_map": "epsilon deformation; new gravitational potential",
            "not_NVG_derived": True,
            "flat_bounce_shape": "M_cosmological^2=M^2*(1+epsilon); rho_bounce=(1+epsilon)*rho_c",
        },
        "symbolic": {"action_variation_and_candidate": action,
                     "flat_reduction": flat, "curved_law": curved_law,
                     "cost_theorem": cost,
                     "regularity": regularity,
                     "barotropic_F": barotropic},
        "numerical": {"curved_point": point, "endpoint_inverse_80": inverse80,
                      "endpoint_inverse_120": inverse120,
                      "vacuum_limit_80": vacuum80,
                      "vacuum_limit_120": vacuum120, "cycles": cycles},
        "negative_controls": {"old_constraint": old,
                              "omitted_rate_factor": omitted,
                              "epsilon_X_kinetic_substitution": kinetic,
                              "epsilon_domain": domain},
        "scope": {
            "manufactured_controls": True, "evidence_weight": 0,
            "counterfactual": True,
            "no_perturbative_or_nonlinear_stability_claim": True,
            "no_empirical_success_or_universal_theory_claim": True,
            "bare_M_retained_in_curvature": True,
            "gravity_and_rho_c_extra_assumptions": True,
        },
        "mathematical_checks_passed": False,
    }
    result["mathematical_checks_passed"] = acceptance_from_result(result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", default="0.1")
    parser.add_argument("--rho-c-ratio", default="1")
    args = parser.parse_args(argv)
    try:
        result = run_audit(epsilon=args.epsilon,
                           rho_c_ratio=args.rho_c_ratio)
    except (ArithmeticError, ValueError, ImportError) as exc:
        print(json.dumps({"status": "invalid_or_failed_audit",
                          "evidence_weight": EVIDENCE_WEIGHT,
                          "mathematical_checks_passed": False,
                          "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(_json_values(result), ensure_ascii=False,
                     indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("mathematical_checks_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
