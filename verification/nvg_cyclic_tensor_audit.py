#!/usr/bin/env python3
"""Lorentzian TT action and finite closed-cycle tensor-transfer audit.

This module is an executable calculation for the fixed positive-epsilon
extension described by the active Lunacy plan.  It uses the standard
Lorentzian Einstein-Hilbert TT quadratic form (with its analytic curvature
and sign justification), keeps the bare Planck mass in the action, and then
integrates the actual Hamiltonian transfer matrix on a small declared set of
manufactured constant-w cycles.  It does not claim to reproduce every
component of a four-dimensional EH expansion.  The cycles are controls only;
they are not NVG matter or observational predictions.

The harmonic convention is the unit-S3 convention
``Delta_T Q_ij = -(n**2-3) Q_ij`` for ``n >= 3``.  The Lorentzian action has
positive time kinetic term and negative spatial term.  The Euclidean signs
sometimes used when tabulating the S3 harmonics are never used for the time
evolution here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import sympy as sp


STATUS = "nvg_cyclic_tensor_action_transfer_audit"
SCHEMA = "nvg_cyclic_tensor_audit.v1"
EVIDENCE_WEIGHT = 0
SOURCE_PATH = Path(__file__).resolve()
REPORT_PATH = (SOURCE_PATH.parent.parent /
               "NVG_CYCLIC_PERTURBATION_CLOSURE_RU.md")

EPSILON = 0.1
RHO_C_RATIO = 1.0
DEFAULT_W = (0.0, 1.0 / 3.0, 1.0)
DEFAULT_N = tuple(range(3, 21))
DECLARED_W_LABELS = ("0", "0.3333333333333333333333333333333333", "1")

SOURCES = {
    "derivation_and_scope": "NVG_CYCLIC_PERTURBATION_CLOSURE_RU.md",
    "accepted_background_map": "verification/nvg_regular_cyclic_action_audit.py",
    "harmonic_convention_cross_check":
        "https://arxiv.org/html/1706.07415",
    "TT_action_status":
        "standard Lorentzian Einstein-Hilbert TT quadratic form, analytically justified; full component EH expansion is not claimed here",
    "derivation_note":
        "Lorentzian variation and Hamilton/conformal conversion are explicit; Euclidean time signs not copied",
}

NORMALIZATION = {
    "A": "a/a_ref",
    "a_ref": "sqrt(3*M2/rho_ref)",
    "tau": "t/a_ref",
    "sigma": "eta/a_ref with d sigma=d tau/A",
    "rho_c": "R*rho_ref",
    "M2": "bare Planck-mass squared in the action",
    "Pi": "p_h/a_ref^2, so h_tau=Pi/A^3",
    "TT_amplitude": "dimensionless h_n",
}

REQUIRED_SYMBOLIC_ROWS = frozenset({
    "lorentzian_time_sign",
    "lorentzian_spatial_sign",
    "bare_M2_kinetic_coefficient",
    "unit_tensor_characteristic_speed",
    "TT_harmonic_laplacian",
    "curvature_frequency_shift",
    "physical_frequency_on_unit_S3",
    "mode_euler_lagrange",
    "hamiltonian_h_equation",
    "hamiltonian_Pi_equation",
    "conformal_h_sigma_relation",
    "conformal_Pi_sigma_relation",
    "conformal_h_second_derivative_from_chain",
    "conformal_h_second_derivative",
    "conformal_nonunit_A_extra_control",
    "conformal_scale_second_derivative",
    "conformal_canonical_conversion",
    "conformal_frequency_equation",
})

MODE_NUMERIC_FIELDS = (
    "w", "n", "epsilon", "rho_c_over_rho_ref", "A_low",
    "period_quadrature_coarse", "period_quadrature_fine",
    "period_direct_coarse", "period_direct_fine", "period_direct_ode",
    "quadrature_refinement_relative", "ode_vs_quadrature_relative",
    "trajectory_curve_max_abs", "algebraic_curve_constraint_max_abs",
    "A_end_direct_ode", "q_end_direct_ode",
    "matrix_refinement_relative", "q_vs_time_relative", "u_vs_time_relative",
    "determinant_coarse", "determinant_fine", "determinant_q", "determinant_u",
    "determinant_error_coarse", "determinant_error_fine",
    "determinant_error_q", "determinant_error_u",
    "symplectic_residual_coarse", "symplectic_residual_fine",
    "symplectic_residual_q", "symplectic_residual_u",
    "trace_fine", "discriminant_fine", "eigen_modulus_min",
    "eigen_modulus_max",
)

MODE_REQUIRED_KEYS = frozenset(
    MODE_NUMERIC_FIELDS + (
        "matrix_coarse", "matrix_fine", "matrix_q", "matrix_u",
        "classification", "passed", "inputs_manufactured", "normalization",
        "scope", "solver_controls",
    )
)

RESULT_REQUIRED_KEYS = frozenset({
    "status", "schema", "evidence_weight", "mathematical_checks_passed",
    "symbolic_checks", "static_scale_check", "generator_check", "cycles",
    "growth_checks", "summary", "normalization", "scope", "sources",
})

_MATRIX_KEYS = ("matrix_coarse", "matrix_fine", "matrix_q", "matrix_u")
_DET_KEYS = ("determinant_coarse", "determinant_fine", "determinant_q", "determinant_u")


def _finite(value: Any, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    """Parse a finite real without accepting bools, NaNs, or infinities."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArithmeticError("nonfinite JSON output")
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _rows(expressions: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for name, expression in expressions.items():
        reduced = sp.trigsimp(sp.factor(sp.simplify(expression)))
        rows[name] = {"residual": str(reduced), "passed": bool(reduced == 0)}
    return rows


def rows_pass(rows: Any, required_names: Iterable[str] = REQUIRED_SYMBOLIC_ROWS) -> bool:
    """Strict symbolic-row validator used by both the CLI and tests."""
    if not isinstance(rows, dict) or set(rows) != set(required_names):
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


def harmonic_data(n: Any, *, kappa: Any = 1, laplacian_offset: Any = 3,
                  curvature_shift: Any = 2) -> Dict[str, float]:
    """Return the S3 TT eigenvalue and physical frequency without fitting."""
    nv = _integer(n, "n", minimum=3)
    kv = _finite(kappa, "kappa", nonnegative=True)
    lo = _finite(laplacian_offset, "laplacian_offset")
    cs = _finite(curvature_shift, "curvature_shift")
    lam = float(nv * nv - lo)
    # a=1 in this algebraic helper; the cycle code inserts A explicitly.
    frequency = float(lam + cs * kv)
    return {
        "n": float(nv),
        "laplacian_eigenvalue": -lam,
        "laplacian_positive_eigenvalue": lam,
        "curvature_kappa": kv,
        "curvature_shift": cs,
        "frequency_squared_at_a1": frequency,
    }


def tensor_action_symbolic_checks(*, lorentzian_time_sign: Any = 1,
                                  lorentzian_spatial_sign: Any = 1,
                                  laplacian_offset: Any = 3,
                                  curvature_shift: Any = 2,
                                  conformal_extra_A_minus2: Any = 0,
                                  conformal_target_extra_A_minus2: Any = 0) -> Dict[str, Dict[str, Any]]:
    """Check the Lorentzian variation of the standard TT mode action.

    The normalized mode density is

    ``L_T/(M2*a^3) = 1/2 [h_dot^2 - (lambda+2*kappa) h^2/a^2]``.

    This is the standard Lorentzian Einstein-Hilbert TT quadratic form,
    analytically justified for an isotropic minimally coupled source.  The
    function explicitly varies that mode form and derives the Hamilton and
    conformal equations; it does not claim to reproduce every component of a
    four-dimensional EH expansion.  Isotropic matter/cuscuton anisotropic
    stress is therefore an input assumption of this tensor-only sector, not a
    hard-coded EH contraction certificate.

    ``lorentzian_*_sign`` and the harmonic offsets are explicit mutation
    controls.  They are not physical options and a mutated value must leave
    at least one required residual nonzero.
    """
    ts = _finite(lorentzian_time_sign, "lorentzian_time_sign")
    ss = _finite(lorentzian_spatial_sign, "lorentzian_spatial_sign")
    lo = _finite(laplacian_offset, "laplacian_offset")
    cs = _finite(curvature_shift, "curvature_shift")
    extra = _finite(conformal_extra_A_minus2,
                    "conformal_extra_A_minus2", nonnegative=True)
    target_extra = _finite(conformal_target_extra_A_minus2,
                           "conformal_target_extra_A_minus2", nonnegative=True)

    n, kappa, a, M2 = sp.symbols("n kappa a M2", real=True)
    h, hdot, hddot, p = sp.symbols("h hdot hddot p", real=True)
    H, Hdot = sp.symbols("H Hdot", real=True)
    lam = n**2 - sp.Rational(str(lo))
    omega_action = (lam + sp.Rational(str(cs)) * kappa) / a**2
    omega_expected_general = (n**2 - 3 + 2 * kappa) / a**2
    omega_expected = (n**2 - 1) / a**2
    # The overall M2/8 convention is immaterial; the positive M2 is retained.
    L = sp.Rational(1, 2) * M2 * a**3 * (
        sp.Rational(str(ts)) * hdot**2
        - sp.Rational(str(ss)) * omega_action * h**2)
    kinetic_coeff = sp.diff(L, hdot, 2) / a**3
    spatial_coeff = sp.diff(L, h, 2) / a**3
    # Euler-Lagrange expression after using a_dot/a=H and
    # d(a^3 hdot)/dt=a^3(hddot+3H hdot).
    euler = (sp.Rational(str(ts)) * M2 * (hddot + 3 * H * hdot)
             + sp.Rational(str(ss)) * M2 * omega_action * h)
    p_action = sp.Rational(str(ts)) * a**3 * hdot
    force_action = (-sp.Rational(str(ss)) * a
                    * (n**2 - sp.Rational(str(lo))
                       + sp.Rational(str(cs)) * kappa) * h)
    # Canonical conversion in conformal time sigma, d sigma=d tau/A.  Start
    # only from h_sigma=Pi/A**2, Pi_sigma=-(n**2-1)A**2 h and
    # A_sigma=A**2 H; the factors of A are deliberately independent of the
    # cosmic-time omega_expected expression above.
    A, hs, Ps = sp.symbols("A hs Ps", positive=True, real=True)
    As = sp.symbols("As", real=True)
    A_tau = A * H
    As_from_chain = sp.simplify(A * A_tau)
    Ass_from_chain = sp.simplify(
        A * (2 * A * A_tau * H + A**2 * Hdot))
    omega_n = n**2 - 1
    hsig_from_chain = Ps / A**2
    pisig_from_hamilton = -omega_n * A**2 * hs
    hss_from_chain = sp.simplify(
        pisig_from_hamilton / A**2 - 2 * Ps * As_from_chain / A**3)
    hss_expected = -omega_n * hs - 2 * H * Ps / A
    # ``extra`` is a deliberate negative control: it inserts an erroneous
    # A^-2 into h_ss while the independently derived expected expression
    # remains fixed.  The direct chain row below catches a duplicated wrong
    # factor even if a second equation were mutated in the same way.
    hss_used = hss_from_chain - extra * omega_n * hs / A**2
    usig_from_definition = As_from_chain * hs + A * hsig_from_chain
    uss = sp.expand(Ass_from_chain * hs + 2 * As_from_chain * hsig_from_chain
                    + A * hss_used)
    conformal_target = ((omega_n - Ass_from_chain / A)
                        + target_extra * omega_n / A**2) * (A * hs)
    conformal_residual = sp.simplify(uss + conformal_target)

    return _rows({
        "lorentzian_time_sign": kinetic_coeff - M2,
        "lorentzian_spatial_sign": spatial_coeff + M2 * omega_expected_general,
        "bare_M2_kinetic_coefficient": kinetic_coeff - M2,
        "unit_tensor_characteristic_speed":
            (-spatial_coeff / (M2 * omega_action)) - 1,
        "TT_harmonic_laplacian": lam - (n**2 - 3),
        "curvature_frequency_shift": omega_action - omega_expected_general,
        "physical_frequency_on_unit_S3":
            omega_expected_general.subs(kappa, 1) - (n**2 - 1) / a**2,
        "mode_euler_lagrange": euler - M2 * (hddot + 3 * H * hdot
                                               + omega_expected_general * h),
        "hamiltonian_h_equation": p_action / a**3 - hdot,
        "hamiltonian_Pi_equation": force_action
        + (n**2 - 3 + 2 * kappa) * a * h,
        "conformal_h_sigma_relation": hsig_from_chain - Ps / A**2,
        "conformal_Pi_sigma_relation":
            pisig_from_hamilton + omega_n * A**2 * hs,
        "conformal_h_second_derivative_from_chain":
            hss_from_chain - hss_expected,
        "conformal_h_second_derivative": hss_used - hss_expected,
        "conformal_nonunit_A_extra_control":
            (hss_used - hss_expected).subs(A, sp.Integer(2)),
        "conformal_scale_second_derivative":
            Ass_from_chain - A**3 * (2 * H**2 + Hdot),
        "conformal_canonical_conversion":
            usig_from_definition - (As_from_chain * hs + Ps / A),
        "conformal_frequency_equation": conformal_residual,
    })


# Friendly aliases used by callers and focused tests.
action_symbolic_checks = tensor_action_symbolic_checks


def hamiltonian_generator(A: Any, n: Any) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Hamiltonian generator for the normalized pair (h, Pi)."""
    av = _finite(A, "A", positive=True)
    nv = _integer(n, "n", minimum=3)
    omega_n = float(nv * nv - 1)
    return ((0.0, av ** -3), (-omega_n * av, 0.0))


def _matrix(matrix: Any, name: str = "matrix") -> List[List[float]]:
    if (not isinstance(matrix, (list, tuple)) or len(matrix) != 2
            or any(not isinstance(row, (list, tuple)) or len(row) != 2
                   for row in matrix)):
        raise ValueError(f"{name} must be a 2x2 matrix")
    result = []
    for i, row in enumerate(matrix):
        result.append([_finite(value, f"{name}[{i}]") for value in row])
    return result


def matrix_determinant(matrix: Any) -> float:
    m = _matrix(matrix)
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def matrix_max_difference(left: Any, right: Any) -> float:
    a = _matrix(left, "left")
    b = _matrix(right, "right")
    return max(abs(a[i][j] - b[i][j])
               for i in range(2) for j in range(2))


def symplectic_residual(matrix: Any) -> float:
    m = _matrix(matrix)
    # For 2x2 matrices M^T J M=(det M)J, but evaluate the identity directly
    # so a future extension does not silently normalize the determinant.
    j = ((0.0, 1.0), (-1.0, 0.0))
    mtj = [[m[0][i] * j[0][k] + m[1][i] * j[1][k]
            for k in range(2)] for i in range(2)]
    mtjm = [[mtj[i][0] * m[0][j2] + mtj[i][1] * m[1][j2]
             for j2 in range(2)] for i in range(2)]
    return max(abs(mtjm[i][j2] - j[i][j2])
               for i in range(2) for j2 in range(2))


def classify_transfer(matrix: Any, *, determinant_tolerance: float = 3e-7,
                      discriminant_tolerance: float = 3e-7) -> str:
    """Classify a raw monodromy matrix without determinant normalization."""
    m = _matrix(matrix)
    det = matrix_determinant(m)
    if not math.isfinite(det) or abs(det - 1.0) > determinant_tolerance:
        return "numerically_unresolved"
    trace = m[0][0] + m[1][1]
    discriminant = trace * trace - 4.0 * det
    if not math.isfinite(discriminant):
        return "numerically_unresolved"
    if discriminant > discriminant_tolerance:
        return "hyperbolic"
    if discriminant < -discriminant_tolerance:
        return "elliptic"
    return "parabolic"


def static_scale_harmonic_rotation(n: Any = 3, A: Any = 2,
                                   phase: Any = 0.73) -> List[List[float]]:
    """Exact static-scale harmonic rotation in the (h, Pi) basis.

    For constant ``A``, ``theta=sqrt(n**2-1)*Delta_tau/A`` and the matrix is
    an exact symplectic rotation.  ``phase`` is theta itself.
    """
    nv = _integer(n, "n", minimum=3)
    av = _finite(A, "A", positive=True)
    theta = _finite(phase, "phase")
    kn = math.sqrt(nv * nv - 1.0)
    c, s = math.cos(theta), math.sin(theta)
    return [[c, s / (av * av * kn)], [-av * av * kn * s, c]]


def static_scale_check(*, n: Any = 3, A: Any = 2,
                       phase: Any = 0.73) -> Dict[str, Any]:
    exact = static_scale_harmonic_rotation(n, A, phase)
    det = matrix_determinant(exact)
    return {
        "n": _integer(n, "n", minimum=3),
        "A": _finite(A, "A", positive=True),
        "phase": _finite(phase, "phase"),
        "matrix": exact,
        "raw_determinant": det,
        "determinant_error": abs(det - 1.0),
        "symplectic_residual": symplectic_residual(exact),
        "passed": bool(abs(det - 1.0) < 2e-15
                       and symplectic_residual(exact) < 2e-15),
        "scope": "exact static-scale TT harmonic rotation; no cycle input",
    }


def generator_symplectic_check(*, n_values: Sequence[int] = (3, 7, 20),
                               A_values: Sequence[float] = (0.67, 1.0, 2.0)) -> Dict[str, Any]:
    rows = []
    j = ((0.0, 1.0), (-1.0, 0.0))
    maximum = 0.0
    for n in n_values:
        nv = _integer(n, "n", minimum=3)
        for A in A_values:
            av = _finite(A, "A", positive=True)
            g = hamiltonian_generator(av, nv)
            # G^T J + J G, evaluated directly from the declared generator.
            gtj = [[g[0][i] * j[0][k] + g[1][i] * j[1][k]
                    for k in range(2)] for i in range(2)]
            jg = [[j[i][0] * g[0][k] + j[i][1] * g[1][k]
                   for k in range(2)] for i in range(2)]
            residual = max(abs(gtj[i][k] + jg[i][k])
                           for i in range(2) for k in range(2))
            maximum = max(maximum, residual)
            rows.append({"n": nv, "A": av, "residual": residual})
    return {"rows": rows, "max_residual": maximum, "passed": bool(maximum == 0.0),
            "scope": "Hamiltonian generator identity JG+G^T J=0"}


def _control(w: Any, epsilon: Any, rho_c_over_rho_ref: Any) -> Tuple[float, float, float]:
    wv = _finite(w, "w", nonnegative=True)
    eps = _finite(epsilon, "epsilon", positive=True)
    ratio = _finite(rho_c_over_rho_ref, "rho_c_over_rho_ref", positive=True)
    if wv not in (0.0, 1.0 / 3.0, 1.0):
        raise ValueError("only declared manufactured controls w=0,1/3,1 are allowed")
    if eps != EPSILON or ratio != RHO_C_RATIO:
        raise ValueError("tensor cycle controls are fixed at epsilon=.1 and R=1")
    return wv, eps, ratio


def _check_n_declared(n: Any) -> int:
    nv = _integer(n, "n", minimum=3)
    if nv > 20:
        raise ValueError("declared tensor controls stop at n=20")
    return nv


def _imports():
    # Keep scientific imports local so symbolic-only mutation tests remain
    # usable in a minimal interpreter.
    import numpy as np
    from scipy.integrate import quad, solve_ivp
    from scipy.optimize import brentq
    return np, quad, solve_ivp, brentq


def _cycle_A_low(w: float, ratio: float, epsilon: float) -> float:
    _, _, _, brentq = _imports()
    m = 3.0 * (1.0 + w)
    C = (1.0 + epsilon) * ratio

    def f(A: float) -> float:
        return A ** (-m) - A ** (-2.0) - C

    return float(brentq(f, 1e-8, 1.0, xtol=2e-14, rtol=1e-14))


def _density(A: float, w: float) -> float:
    return A ** (-3.0 * (1.0 + w))


def _A_of_q(q: float, w: float, ratio: float, epsilon: float,
            A_low: float) -> float:
    _, _, _, brentq = _imports()
    qv = _finite(q, "q")
    # Dense-output samples can differ from the terminal event by a few ulps;
    # this is endpoint representation error, not an extension of the physical
    # q domain.  Larger excursions remain hard failures below.
    if -1e-10 < qv < 0.0:
        qv = 0.0
    if math.pi < qv < math.pi + 1e-10:
        qv = math.pi
    if not 0.0 <= qv <= math.pi:
        raise ValueError("q must lie in [0, pi] for one declared cycle")
    if qv == 0.0 or qv == math.pi:
        return A_low
    if abs(qv - math.pi / 2.0) < 5e-15:
        return 1.0
    target = (1.0 + epsilon) * ratio * math.cos(qv) ** 2
    m = 3.0 * (1.0 + w)

    def f(A: float) -> float:
        return A ** (-m) - A ** (-2.0) - target

    # At a turning endpoint the exact root is A_low.  Floating evaluation of
    # F(A_low)-target can lose its sign when q differs from pi by one ulp;
    # recognize only that tiny endpoint representation band.
    f_low, f_high = f(A_low), f(1.0)
    if abs(f_low) <= 1e-12:
        return A_low
    if abs(f_high) <= 1e-12:
        return 1.0
    return float(brentq(f, A_low, 1.0, xtol=2e-14, rtol=1e-14))


def _background_rates(A: float, q: float, w: float, ratio: float,
                      epsilon: float) -> Tuple[float, float]:
    av = _finite(A, "A", positive=True)
    qv = _finite(q, "q")
    sqrt_ratio = math.sqrt(ratio)
    Hbar = sqrt_ratio * math.sin(2.0 * qv) / 2.0
    qbar = 3.0 / (2.0 * sqrt_ratio * (1.0 + epsilon)) * (
        (1.0 + w) * _density(av, w) - 2.0 / (3.0 * av * av))
    if not math.isfinite(Hbar) or not math.isfinite(qbar) or qbar <= 0.0:
        raise ArithmeticError("nonpositive or nonfinite q rate on declared cycle")
    return av * Hbar, qbar


def cycle_background(*, w: Any = 0.0, epsilon: Any = EPSILON,
                     rho_c_over_rho_ref: Any = RHO_C_RATIO,
                     quadrature_rtol: Any = 3e-11,
                     ode_rtol: Any = 2e-11, ode_atol: Any = 2e-13,
                     samples: Any = 41) -> Dict[str, Any]:
    """Build the declared background two ways and expose trajectory residuals."""
    np, quad, solve_ivp, _ = _imports()
    wv, eps, ratio = _control(w, epsilon, rho_c_over_rho_ref)
    qr = _finite(quadrature_rtol, "quadrature_rtol", positive=True)
    orr = _finite(ode_rtol, "ode_rtol", positive=True)
    oa = _finite(ode_atol, "ode_atol", positive=True)
    nsamp = _integer(samples, "samples", minimum=17)
    A_low = _cycle_A_low(wv, ratio, eps)
    m = 3.0 * (1.0 + wv)

    def qrate(q: float) -> float:
        A = _A_of_q(float(q), wv, ratio, eps, A_low)
        return _background_rates(A, float(q), wv, ratio, eps)[1]

    coarse, coarse_error = quad(
        lambda q: 1.0 / qrate(q), 0.0, math.pi,
        epsabs=max(30.0 * qr, 2e-8), epsrel=max(30.0 * qr, 2e-8),
        points=[math.pi / 2.0], limit=220)
    fine, fine_error = quad(
        lambda q: 1.0 / qrate(q), 0.0, math.pi,
        epsabs=qr, epsrel=qr, points=[math.pi / 2.0], limit=320)

    def rhs(tau: float, state: Sequence[float]) -> Tuple[float, float]:
        A, q = (float(state[0]), float(state[1]))
        if A <= 0.0 or not math.isfinite(A):
            raise ArithmeticError("invalid A in direct background ODE")
        return _background_rates(A, q, wv, ratio, eps)

    def event(_tau: float, state: Sequence[float]) -> float:
        return float(state[1] - math.pi)

    event.terminal = True
    event.direction = 1
    max_time = max(2.0 * float(fine) + 1.0, 4.0)
    coarse_sol = solve_ivp(
        rhs, (0.0, max_time), (A_low, 0.0), method="DOP853",
        rtol=max(orr * 30.0, 1e-8), atol=max(oa * 30.0, 1e-11),
        max_step=0.1, events=event, dense_output=True)
    fine_sol = solve_ivp(
        rhs, (0.0, max_time), (A_low, 0.0), method="DOP853",
        rtol=orr, atol=oa, max_step=0.025, events=event, dense_output=True)
    for label, sol in (("coarse", coarse_sol), ("fine", fine_sol)):
        if (not sol.success or len(sol.t_events[0]) != 1
                or not np.all(np.isfinite(sol.y))):
            raise ArithmeticError(f"{label} direct background ODE failed")
    period_coarse = float(coarse_sol.t_events[0][0])
    period_direct = float(fine_sol.t_events[0][0])
    A_end, q_end = (float(v) for v in fine_sol.y_events[0][0])

    trajectory_curve = 0.0
    for tau in np.linspace(0.0, period_direct, max(33, nsamp)):
        A_t, q_t = (float(v) for v in fine_sol.sol(float(tau)))
        trajectory_curve = max(
            trajectory_curve,
            abs(A_t - _A_of_q(q_t, wv, ratio, eps, A_low)))

    # Check the algebraic constraint itself independently over q.
    algebraic_constraint = 0.0
    for q in np.linspace(0.0, math.pi, max(33, nsamp)):
        A = _A_of_q(float(q), wv, ratio, eps, A_low)
        algebraic_constraint = max(
            algebraic_constraint,
            abs(_density(A, wv) - A ** -2.0
                - (1.0 + eps) * ratio * math.cos(float(q)) ** 2))

    period_ref = max(1.0, abs(float(fine)))
    quadrature_refinement = abs(float(coarse) - float(fine)) / period_ref
    ode_vs_quad = abs(period_direct - float(fine)) / period_ref
    return {
        "w": wv,
        "epsilon": eps,
        "rho_c_over_rho_ref": ratio,
        "m": m,
        "C": (1.0 + eps) * ratio,
        "A_low": A_low,
        "A_high": 1.0,
        "period_quadrature_coarse": float(coarse),
        "period_quadrature_fine": float(fine),
        "quadrature_error_estimate": float(max(coarse_error, fine_error)),
        "period_direct_coarse": period_coarse,
        "period_direct_fine": period_direct,
        "period_direct_ode": period_direct,
        "quadrature_refinement_relative": quadrature_refinement,
        "ode_vs_quadrature_relative": ode_vs_quad,
        "A_end_direct_ode": A_end,
        "q_end_direct_ode": q_end,
        "trajectory_curve_max_abs": trajectory_curve,
        "algebraic_curve_constraint_max_abs": algebraic_constraint,
        "samples": max(33, nsamp),
        "normalization": dict(NORMALIZATION),
        "scope": "manufactured constant-w background control; no NVG matter prediction",
    }


def _matrix_from_solution(solution: Any) -> List[List[float]]:
    # [h1,Pi1,h2,Pi2] is integrated as four simultaneous columns.
    y = solution.y[:, -1]
    return [[float(y[2]), float(y[4])], [float(y[3]), float(y[5])]]


def _time_transfer(background: Mapping[str, Any], n: int, *, rtol: float,
                   atol: float, max_step: float) -> Dict[str, Any]:
    np, _, solve_ivp, _ = _imports()
    wv = float(background["w"])
    eps = float(background["epsilon"])
    ratio = float(background["rho_c_over_rho_ref"])
    A_low = float(background["A_low"])
    period_fine = float(background["period_quadrature_fine"])
    omega_n = float(n * n - 1)
    calls = 0

    def rhs(_tau: float, state: Sequence[float]) -> Tuple[float, ...]:
        nonlocal calls
        calls += 1
        if calls > 500000:
            raise ArithmeticError("bounded tensor ODE evaluation budget exceeded")
        A, q, h1, P1, h2, P2 = (float(x) for x in state)
        A_tau, q_tau = _background_rates(A, q, wv, ratio, eps)
        values = (A_tau, q_tau, P1 / A**3, -omega_n * A * h1,
                  P2 / A**3, -omega_n * A * h2)
        if not all(math.isfinite(v) for v in values):
            raise ArithmeticError("nonfinite time tensor evolution")
        return values

    def event(_tau: float, state: Sequence[float]) -> float:
        return float(state[1] - math.pi)

    event.terminal = True
    event.direction = 1
    max_time = max(2.0 * period_fine + 1.0, 4.0)
    sol = solve_ivp(
        rhs, (0.0, max_time), (A_low, 0.0, 1.0, 0.0, 0.0, 1.0),
        method="DOP853", rtol=rtol, atol=atol, max_step=max_step,
        events=event, dense_output=True)
    if (not sol.success or len(sol.t_events[0]) != 1
            or not np.all(np.isfinite(sol.y))):
        raise ArithmeticError("direct time tensor transfer failed")
    matrix = _matrix_from_solution(sol)
    A_end, q_end = (float(v) for v in sol.y_events[0][0][:2])
    return {
        "matrix": matrix,
        "period": float(sol.t_events[0][0]),
        "A_end": A_end,
        "q_end": q_end,
        "rhs_evaluations": int(sol.nfev),
        "finite": True,
    }


def _q_transfer(background: Mapping[str, Any], n: int, *, rtol: float,
                atol: float, max_step: float) -> Dict[str, Any]:
    """Integrate the same Hamilton pair with q as the independent variable."""
    np, _, solve_ivp, _ = _imports()
    wv = float(background["w"])
    eps = float(background["epsilon"])
    ratio = float(background["rho_c_over_rho_ref"])
    A_low = float(background["A_low"])
    omega_n = float(n * n - 1)
    calls = 0

    def rhs(q: float, state: Sequence[float]) -> Tuple[float, ...]:
        nonlocal calls
        calls += 1
        if calls > 500000:
            raise ArithmeticError("bounded q tensor ODE evaluation budget exceeded")
        A, h1, P1, h2, P2 = (float(x) for x in state)
        A_tau, q_tau = _background_rates(A, q, wv, ratio, eps)
        values = (A_tau / q_tau, P1 / (A**3 * q_tau),
                  -omega_n * A * h1 / q_tau,
                  P2 / (A**3 * q_tau), -omega_n * A * h2 / q_tau)
        if not all(math.isfinite(v) for v in values):
            raise ArithmeticError("nonfinite q tensor evolution")
        return values

    sol = solve_ivp(
        rhs, (0.0, math.pi), (A_low, 1.0, 0.0, 0.0, 1.0),
        method="DOP853", rtol=rtol, atol=atol, max_step=max_step)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise ArithmeticError("q-independent tensor transfer failed")
    y = sol.y[:, -1]
    matrix = [[float(y[1]), float(y[3])], [float(y[2]), float(y[4])]]
    return {
        "matrix": matrix,
        "A_end": float(y[0]),
        "q_end": math.pi,
        "rhs_evaluations": int(sol.nfev),
        "finite": True,
    }


def _conformal_transfer(background: Mapping[str, Any], n: int, *, rtol: float,
                        atol: float, max_step: float) -> Dict[str, Any]:
    """Integrate u=A*h, pi=u_sigma and transform back at periodic endpoints."""
    np, _, solve_ivp, _ = _imports()
    wv = float(background["w"])
    eps = float(background["epsilon"])
    ratio = float(background["rho_c_over_rho_ref"])
    A_low = float(background["A_low"])
    omega_n = float(n * n - 1)
    calls = 0

    def rhs(q: float, state: Sequence[float]) -> Tuple[float, ...]:
        nonlocal calls
        calls += 1
        if calls > 500000:
            raise ArithmeticError("bounded conformal ODE evaluation budget exceeded")
        # A(q) is solved from the algebraic background independently of the
        # q-trajectory used by _q_transfer.
        A = _A_of_q(q, wv, ratio, eps, A_low)
        _, q_tau = _background_rates(A, q, wv, ratio, eps)
        H = math.sqrt(ratio) * math.sin(2.0 * q) / 2.0
        H_tau = math.sqrt(ratio) * math.cos(2.0 * q) * q_tau
        sigma_q = 1.0 / (A * q_tau)
        omega_sigma = omega_n - A * A * (2.0 * H * H + H_tau)
        u1, pi1, u2, pi2 = (float(x) for x in state)
        values = (pi1 * sigma_q, -omega_sigma * u1 * sigma_q,
                  pi2 * sigma_q, -omega_sigma * u2 * sigma_q)
        if not all(math.isfinite(v) for v in values):
            raise ArithmeticError("nonfinite conformal tensor evolution")
        return values

    # At each turning point H=0, so A_sigma=0.  Keep the full conversion
    # matrix explicit nonetheless; this prevents an accidental periodic-basis
    # assumption from hiding a wrong endpoint transformation.
    H0 = 0.0
    As0 = A_low * A_low * H0
    u0 = ((A_low, 0.0), (As0, 1.0 / A_low))
    sol = solve_ivp(
        rhs, (0.0, math.pi), (u0[0][0], u0[1][0], u0[0][1], u0[1][1]),
        method="DOP853", rtol=rtol, atol=atol, max_step=max_step)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise ArithmeticError("conformal tensor transfer failed")
    y = sol.y[:, -1]
    U = [[float(y[0]), float(y[2])], [float(y[1]), float(y[3])]]
    H1 = 0.0
    As1 = A_low * A_low * H1
    C0 = [[A_low, 0.0], [As0, 1.0 / A_low]]
    C1 = [[A_low, 0.0], [As1, 1.0 / A_low]]
    # The integrated columns were initialized as ``C0`` times the h/P basis,
    # so the returned h/P transfer is C1^{-1} times this integrated matrix.
    # Do not multiply by C0 a second time; doing so is a subtle endpoint-basis
    # error that the independent transfer comparison is meant to expose.
    c1det = matrix_determinant(C1)
    C1inv = [[C1[1][1] / c1det, -C1[0][1] / c1det],
             [-C1[1][0] / c1det, C1[0][0] / c1det]]
    M = [[C1inv[i][0] * U[0][j] + C1inv[i][1] * U[1][j]
          for j in range(2)] for i in range(2)]
    return {
        "matrix": M,
        "conformal_matrix": U,
        "endpoint_basis_initial": C0,
        "endpoint_basis_final": C1,
        "sigma_span": None,
        "rhs_evaluations": int(sol.nfev),
        "finite": True,
    }


def _relative_matrix_error(left: Any, right: Any) -> float:
    a = _matrix(left, "left")
    b = _matrix(right, "right")
    return max(abs(a[i][j] - b[i][j]) / max(1.0, abs(b[i][j]))
               for i in range(2) for j in range(2))


def tensor_mode_transfer(*, w: Any = 0.0, n: Any = 3,
                         epsilon: Any = EPSILON,
                         rho_c_over_rho_ref: Any = RHO_C_RATIO,
                         coarse_rtol: Any = 2e-8,
                         coarse_atol: Any = 2e-10,
                         fine_rtol: Any = 2e-11,
                         fine_atol: Any = 2e-13,
                         coarse_max_step: Any = 0.1,
                         fine_max_step: Any = 0.025) -> Dict[str, Any]:
    """Compute one actual closed-cycle transfer and all independent controls."""
    wv, eps, ratio = _control(w, epsilon, rho_c_over_rho_ref)
    nv = _check_n_declared(n)
    cr = _finite(coarse_rtol, "coarse_rtol", positive=True)
    ca = _finite(coarse_atol, "coarse_atol", positive=True)
    fr = _finite(fine_rtol, "fine_rtol", positive=True)
    fa = _finite(fine_atol, "fine_atol", positive=True)
    cs = _finite(coarse_max_step, "coarse_max_step", positive=True)
    fs = _finite(fine_max_step, "fine_max_step", positive=True)
    if cr <= fr or ca <= fa or cs <= fs:
        raise ValueError("coarse solver controls must be looser than fine controls")

    background = cycle_background(
        w=wv, epsilon=eps, rho_c_over_rho_ref=ratio,
        ode_rtol=fr, ode_atol=fa)
    coarse = _time_transfer(background, nv, rtol=cr, atol=ca, max_step=cs)
    fine = _time_transfer(background, nv, rtol=fr, atol=fa, max_step=fs)
    q_result = _q_transfer(background, nv, rtol=fr, atol=fa,
                           max_step=math.pi / 100.0)
    u_result = _conformal_transfer(background, nv, rtol=fr, atol=fa,
                                   max_step=math.pi / 100.0)
    matrices = {
        "matrix_coarse": coarse["matrix"],
        "matrix_fine": fine["matrix"],
        "matrix_q": q_result["matrix"],
        "matrix_u": u_result["matrix"],
    }
    determinants = {name.replace("matrix", "determinant"): matrix_determinant(value)
                   for name, value in matrices.items()}
    determinant_errors = {
        name.replace("determinant", "determinant_error"): abs(value - 1.0)
        for name, value in determinants.items()
    }
    symplectic = {
        name.replace("matrix", "symplectic_residual"): symplectic_residual(value)
        for name, value in matrices.items()
    }
    matrix_refinement = _relative_matrix_error(matrices["matrix_coarse"],
                                               matrices["matrix_fine"])
    q_error = _relative_matrix_error(matrices["matrix_q"], matrices["matrix_fine"])
    u_error = _relative_matrix_error(matrices["matrix_u"], matrices["matrix_fine"])
    trace = matrices["matrix_fine"][0][0] + matrices["matrix_fine"][1][1]
    discriminant = trace * trace - 4.0 * determinants["determinant_fine"]
    classification = classify_transfer(matrices["matrix_fine"])
    # Eigenvalue moduli are real for hyperbolic cases and unit-modulus for
    # elliptic cases; use the raw determinant in the quadratic formula.
    if discriminant >= 0.0:
        root = math.sqrt(discriminant)
        eigenvalues = ((trace + root) / 2.0, (trace - root) / 2.0)
        moduli = tuple(abs(v) for v in eigenvalues)
    else:
        realpart = trace / 2.0
        imagpart = math.sqrt(-discriminant) / 2.0
        mod = math.sqrt(realpart * realpart + imagpart * imagpart)
        moduli = (mod, mod)
    passed = bool(
        background["A_low"] > 0.0 and background["A_low"] < 1.0
        and background["quadrature_refinement_relative"] < 3e-7
        and background["ode_vs_quadrature_relative"] < 3e-6
        and background["trajectory_curve_max_abs"] < 3e-7
        and abs(background["A_end_direct_ode"] - background["A_low"]) < 3e-7
        and abs(background["q_end_direct_ode"] - math.pi) < 3e-8
        and matrix_refinement < 3e-6
        and q_error < 3e-6 and u_error < 3e-6
        and all(value < 3e-6 for value in determinant_errors.values())
        and all(value < 3e-6 for value in symplectic.values())
        and classification != "numerically_unresolved"
        and all(math.isfinite(value) for value in moduli)
    )
    return {
        "w": wv,
        "n": nv,
        "epsilon": eps,
        "rho_c_over_rho_ref": ratio,
        "A_low": background["A_low"],
        "period_quadrature_coarse": background["period_quadrature_coarse"],
        "period_quadrature_fine": background["period_quadrature_fine"],
        "period_direct_coarse": coarse["period"],
        "period_direct_fine": fine["period"],
        "period_direct_ode": fine["period"],
        "quadrature_refinement_relative": background["quadrature_refinement_relative"],
        "ode_vs_quadrature_relative": background["ode_vs_quadrature_relative"],
        "trajectory_curve_max_abs": background["trajectory_curve_max_abs"],
        "algebraic_curve_constraint_max_abs": background["algebraic_curve_constraint_max_abs"],
        "A_end_direct_ode": fine["A_end"],
        "q_end_direct_ode": fine["q_end"],
        "matrix_refinement_relative": matrix_refinement,
        "q_vs_time_relative": q_error,
        "u_vs_time_relative": u_error,
        **determinants,
        **determinant_errors,
        **symplectic,
        "trace_fine": trace,
        "discriminant_fine": discriminant,
        "eigen_modulus_min": min(moduli),
        "eigen_modulus_max": max(moduli),
        **matrices,
        "classification": classification,
        "passed": passed,
        "inputs_manufactured": True,
        "normalization": dict(NORMALIZATION),
        "scope": (
            "actual finite closed-cycle TT transfer for one declared constant-w control; "
            "not an all-mode theorem, nonlinear/quantum result, or NVG matter prediction"),
        "solver_controls": {
            "method": "scipy.integrate.solve_ivp:DOP853",
            "coarse": {"rtol": cr, "atol": ca, "max_step": cs},
            "fine": {"rtol": fr, "atol": fa, "max_step": fs},
            "q_independent": {"rtol": fr, "atol": fa, "max_step": math.pi / 100.0},
            "conformal_u": {"rtol": fr, "atol": fa, "max_step": math.pi / 100.0},
            "determinant_normalization": "none; raw Hamiltonian transfer determinant reported",
        },
    }


def _growing_eigenpair(matrix: Any) -> Tuple[float, Tuple[float, float]]:
    """Return the larger-modulus real eigenvalue/eigenvector of a hyperbolic M."""
    m = _matrix(matrix)
    trace = m[0][0] + m[1][1]
    det = matrix_determinant(m)
    disc = trace * trace - 4.0 * det
    if disc <= 0.0:
        raise ValueError("growing eigenpair requires a hyperbolic transfer")
    root = math.sqrt(disc)
    candidates = ((trace + root) / 2.0, (trace - root) / 2.0)
    lam = max(candidates, key=lambda value: abs(value))
    if abs(m[0][1]) > 1e-14:
        vector = (m[0][1], lam - m[0][0])
    elif abs(m[1][0]) > 1e-14:
        vector = (lam - m[1][1], m[1][0])
    else:
        vector = (1.0, 0.0)
    norm = math.hypot(vector[0], vector[1])
    if norm == 0.0 or not math.isfinite(norm):
        raise ArithmeticError("invalid growing eigenvector")
    return lam, (vector[0] / norm, vector[1] / norm)


def multi_cycle_growing_mode(*, w: Any, n: Any, matrix: Any,
                             cycles: Any = 3, epsilon: Any = EPSILON,
                             rho_c_over_rho_ref: Any = RHO_C_RATIO,
                             rtol: Any = 2e-11, atol: Any = 2e-13,
                             max_step: Any = 0.025) -> Dict[str, Any]:
    """Evolve a resolved growing eigenmode over several fresh cycles.

    The monodromy supplies only the initial eigenvector.  A separate scalar
    ODE solve evolves one state vector through each cycle, retaining the
    continuously evolved periodic background rather than multiplying a stored
    matrix.  This is an independent growth check for the finite controls.
    """
    np, _, solve_ivp, _ = _imports()
    wv, eps, ratio = _control(w, epsilon, rho_c_over_rho_ref)
    nv = _check_n_declared(n)
    count = _integer(cycles, "cycles", minimum=2)
    rr = _finite(rtol, "rtol", positive=True)
    aa = _finite(atol, "atol", positive=True)
    step = _finite(max_step, "max_step", positive=True)
    eigenvalue, vector = _growing_eigenpair(matrix)
    A = _cycle_A_low(wv, ratio, eps)
    q = 0.0
    state = [A, q, vector[0], vector[1]]
    norms = [math.hypot(state[2], state[3])]
    periods = []
    for cycle_index in range(count):
        calls = 0
        target = (cycle_index + 1) * math.pi

        def rhs(_tau: float, values: Sequence[float]) -> Tuple[float, ...]:
            nonlocal calls
            calls += 1
            if calls > 500000:
                raise ArithmeticError("bounded growing-mode ODE budget exceeded")
            A_now, q_now, h_now, P_now = (float(x) for x in values)
            A_tau, q_tau = _background_rates(A_now, q_now, wv, ratio, eps)
            out = (A_tau, q_tau, P_now / A_now**3,
                   -(nv * nv - 1.0) * A_now * h_now)
            if not all(math.isfinite(value) for value in out):
                raise ArithmeticError("nonfinite growing-mode evolution")
            return out

        def event(_tau: float, values: Sequence[float]) -> float:
            return float(values[1] - target)

        event.terminal = True
        event.direction = 1
        # The first cycle starts at tau=0; later calls use a local clock, so a
        # generous bound based on the declared period remains sufficient.
        local_bound = max(2.0 * 4.0, 4.0)
        sol = solve_ivp(
            rhs, (0.0, local_bound), tuple(state), method="DOP853",
            rtol=rr, atol=aa, max_step=step, events=event)
        if (not sol.success or len(sol.t_events[0]) != 1
                or not np.all(np.isfinite(sol.y))):
            raise ArithmeticError("multi-cycle growing-mode integration failed")
        state = [float(v) for v in sol.y_events[0][0]]
        periods.append(float(sol.t_events[0][0]))
        norms.append(math.hypot(state[2], state[3]))

    expected = [abs(eigenvalue) ** index * norms[0]
                for index in range(count + 1)]
    relative_errors = [abs(actual - target) / max(1.0, abs(target))
                       for actual, target in zip(norms, expected)]
    per_cycle_growth = [norms[index + 1] / norms[index]
                        for index in range(count)]
    # A strong hyperbolic mode should grow by the eigenvalue modulus.  Keep the
    # threshold explicit and generous enough for the independent ODE path.
    passed = bool(
        abs(eigenvalue) > 1.0 + 1e-4
        and max(relative_errors) < 3e-4
        and all(value > 1.0 + 1e-4 for value in per_cycle_growth))
    return {
        "w": wv,
        "n": nv,
        "cycles": count,
        "growing_eigenvalue": eigenvalue,
        "eigenvalue_modulus": abs(eigenvalue),
        "initial_vector": [vector[0], vector[1]],
        "cycle_periods": periods,
        "norms": norms,
        "expected_norms": expected,
        "relative_errors": relative_errors,
        "per_cycle_growth": per_cycle_growth,
        "passed": passed,
        "scope": "independent multi-cycle evolution of a resolved finite-sample growing TT eigenmode",
    }


def growth_check_acceptance(check: Any) -> bool:
    if not isinstance(check, dict):
        return False
    required = {
        "w", "n", "cycles", "growing_eigenvalue", "eigenvalue_modulus",
        "initial_vector", "cycle_periods", "norms", "expected_norms",
        "relative_errors", "per_cycle_growth", "passed", "scope",
    }
    if set(check) != required or check.get("passed") is not True:
        return False
    try:
        wv = _finite(check["w"], "w")
        nv = _check_n_declared(check["n"])
        count = _integer(check["cycles"], "cycles", minimum=2)
        eig = _finite(check["growing_eigenvalue"], "growing_eigenvalue")
        modulus = _finite(check["eigenvalue_modulus"], "eigenvalue_modulus", positive=True)
    except (KeyError, ValueError):
        return False
    if wv not in DEFAULT_W or nv < 3 or count != len(check["norms"]) - 1:
        return False
    if not all(isinstance(check[name], list) for name in (
            "initial_vector", "cycle_periods", "norms", "expected_norms",
            "relative_errors", "per_cycle_growth")):
        return False
    if len(check["initial_vector"]) != 2 or len(check["cycle_periods"]) != count:
        return False
    if len(check["expected_norms"]) != count + 1 or len(check["relative_errors"]) != count + 1:
        return False
    if len(check["per_cycle_growth"]) != count:
        return False
    try:
        vector = [_finite(value, "initial_vector") for value in check["initial_vector"]]
        periods = [_finite(value, "cycle_period") for value in check["cycle_periods"]]
        norms = [_finite(value, "norm") for value in check["norms"]]
        expected = [_finite(value, "expected_norm") for value in check["expected_norms"]]
        errors = [_finite(value, "relative_error", nonnegative=True)
                  for value in check["relative_errors"]]
        growth = [_finite(value, "per_cycle_growth", positive=True)
                  for value in check["per_cycle_growth"]]
    except (KeyError, ValueError):
        return False
    if (abs(modulus - abs(eig)) > 1e-10 * max(1.0, modulus)
            or abs(math.hypot(*vector) - 1.0) > 1e-8
            or not all(value > 0.0 for value in periods + norms + expected)
            or not _number_matches(expected[0], norms[0], rel=1e-10)):
        return False
    expected_growth = [abs(eig) ** index * norms[0] for index in range(count + 1)]
    if any(abs(a - b) > 1e-8 * max(1.0, abs(a), abs(b))
           for a, b in zip(expected, expected_growth)):
        return False
    recomputed_errors = [abs(a - b) / max(1.0, abs(b))
                         for a, b in zip(norms, expected)]
    if any(abs(a - b) > 1e-8 * max(1.0, abs(a), abs(b))
           for a, b in zip(errors, recomputed_errors)):
        return False
    recomputed_growth = [norms[i + 1] / norms[i] for i in range(count)]
    if any(abs(a - b) > 1e-8 * max(1.0, abs(a), abs(b))
           for a, b in zip(growth, recomputed_growth)):
        return False
    return bool(modulus > 1.0 + 1e-4 and max(errors) < 3e-4
                and all(value > 1.0 + 1e-4 for value in growth)
                and isinstance(check["scope"], str)
                and "independent multi-cycle evolution" in check["scope"])


def _number_matches(actual: Any, expected: Any, *, rel: float = 1e-10) -> bool:
    try:
        a = _finite(actual, "actual")
        e = _finite(expected, "expected")
    except ValueError:
        return False
    return abs(a - e) <= rel * max(1.0, abs(a), abs(e))


def _required_numeric(mapping: Mapping[str, Any], names: Iterable[str]) -> Dict[str, float] | None:
    values: Dict[str, float] = {}
    try:
        for name in names:
            values[name] = _finite(mapping[name], name)
    except (KeyError, ValueError):
        return None
    return values


def _matrix_matches_reported(row: Mapping[str, Any], matrix_name: str,
                             determinant_name: str, residual_name: str) -> bool:
    try:
        matrix = _matrix(row[matrix_name], matrix_name)
        determinant = matrix_determinant(matrix)
        residual = symplectic_residual(matrix)
        return (_number_matches(row[determinant_name], determinant, rel=1e-10)
                and _number_matches(row[residual_name], residual, rel=1e-10))
    except (KeyError, ValueError):
        return False


def _solver_controls_acceptance(controls: Any) -> bool:
    if not isinstance(controls, dict):
        return False
    if set(controls) != {"method", "coarse", "fine", "q_independent",
                         "conformal_u", "determinant_normalization"}:
        return False
    if (controls.get("method") != "scipy.integrate.solve_ivp:DOP853"
            or controls.get("determinant_normalization")
            != "none; raw Hamiltonian transfer determinant reported"):
        return False
    for name in ("coarse", "fine", "q_independent", "conformal_u"):
        item = controls.get(name)
        if not isinstance(item, dict) or set(item) != {"rtol", "atol", "max_step"}:
            return False
        try:
            _finite(item["rtol"], f"{name}.rtol", positive=True)
            _finite(item["atol"], f"{name}.atol", positive=True)
            _finite(item["max_step"], f"{name}.max_step", positive=True)
        except ValueError:
            return False
    try:
        coarse = controls["coarse"]
        fine = controls["fine"]
        q_control = controls["q_independent"]
        u_control = controls["conformal_u"]
        if not (coarse["rtol"] > fine["rtol"]
                and coarse["atol"] > fine["atol"]
                and coarse["max_step"] > fine["max_step"]):
            return False
        if q_control != u_control:
            return False
    except (KeyError, TypeError):
        return False
    return True


def _static_check_acceptance(static: Any) -> bool:
    required = {"n", "A", "phase", "matrix", "raw_determinant",
                "determinant_error", "symplectic_residual", "passed", "scope"}
    if not isinstance(static, dict) or set(static) != required or static.get("passed") is not True:
        return False
    try:
        expected = static_scale_harmonic_rotation(static["n"], static["A"], static["phase"])
        return (
            static.get("scope") == "exact static-scale TT harmonic rotation; no cycle input"
            and _relative_matrix_error(static["matrix"], expected) == 0.0
            and _number_matches(static["raw_determinant"], matrix_determinant(expected), rel=1e-12)
            and _number_matches(static["determinant_error"], abs(matrix_determinant(expected) - 1.0), rel=1e-12)
            and _number_matches(static["symplectic_residual"], symplectic_residual(expected), rel=1e-12)
            and abs(matrix_determinant(expected) - 1.0) < 2e-15
            and symplectic_residual(expected) < 2e-15
        )
    except (KeyError, ValueError):
        return False


def _generator_check_acceptance(generator: Any) -> bool:
    if not isinstance(generator, dict) or set(generator) != {"rows", "max_residual", "passed", "scope"}:
        return False
    if generator.get("passed") is not True or generator.get("scope") != "Hamiltonian generator identity JG+G^T J=0":
        return False
    rows = generator.get("rows")
    if not isinstance(rows, list) or len(rows) != 9:
        return False
    expected = {(n, A) for n in (3, 7, 20) for A in (0.67, 1.0, 2.0)}
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"n", "A", "residual"}:
            return False
        try:
            n = _integer(row["n"], "n", minimum=3)
            A = _finite(row["A"], "A", positive=True)
            residual = _finite(row["residual"], "residual", nonnegative=True)
        except (KeyError, ValueError):
            return False
        key = (n, A)
        if key not in expected or key in seen or residual != 0.0:
            return False
        seen.add(key)
    return (seen == expected and _number_matches(generator["max_residual"], 0.0)
            and _finite(generator["max_residual"], "max_residual", nonnegative=True) == 0.0)


def mode_acceptance(row: Any) -> bool:
    """Fail closed on missing, nonfinite, stale, or tolerance-breaching rows."""
    if not isinstance(row, dict) or set(row) != set(MODE_REQUIRED_KEYS):
        return False
    if row.get("passed") is not True or row.get("inputs_manufactured") is not True:
        return False
    values = _required_numeric(row, MODE_NUMERIC_FIELDS)
    if values is None:
        return False
    try:
        wv = values["w"]
        nv = int(values["n"])
    except (KeyError, ValueError):
        return False
    if values["n"] != nv or nv not in DEFAULT_N:
        return False
    if wv not in DEFAULT_W or values["epsilon"] <= 0 or values["rho_c_over_rho_ref"] <= 0:
        return False
    if values["epsilon"] != EPSILON or values["rho_c_over_rho_ref"] != RHO_C_RATIO:
        return False
    if any(values[name] < 0.0 for name in (
            "quadrature_refinement_relative", "ode_vs_quadrature_relative",
            "trajectory_curve_max_abs", "algebraic_curve_constraint_max_abs",
            "matrix_refinement_relative", "q_vs_time_relative", "u_vs_time_relative",
            "determinant_error_coarse", "determinant_error_fine",
            "determinant_error_q", "determinant_error_u",
            "symplectic_residual_coarse", "symplectic_residual_fine",
            "symplectic_residual_q", "symplectic_residual_u",
            "eigen_modulus_min", "eigen_modulus_max")):
        return False
    if any(values[name] <= 0.0 for name in (
            "period_quadrature_coarse", "period_quadrature_fine",
            "period_direct_coarse", "period_direct_fine", "period_direct_ode")):
        return False
    if row.get("normalization") != NORMALIZATION:
        return False
    if (not isinstance(row.get("scope"), str)
            or "actual finite closed-cycle TT transfer" not in row["scope"]
            or "not an all-mode theorem" not in row["scope"]):
        return False
    if not isinstance(row.get("solver_controls"), dict):
        return False
    if not _solver_controls_acceptance(row["solver_controls"]):
        return False
    for matrix_name, det_name in zip(_MATRIX_KEYS, _DET_KEYS):
        if not _matrix_matches_reported(
                row, matrix_name, det_name,
                matrix_name.replace("matrix", "symplectic_residual")):
            return False
    expected_refinement = _relative_matrix_error(row["matrix_coarse"], row["matrix_fine"])
    expected_q = _relative_matrix_error(row["matrix_q"], row["matrix_fine"])
    expected_u = _relative_matrix_error(row["matrix_u"], row["matrix_fine"])
    expected_qr = abs(values["period_quadrature_coarse"]
                      - values["period_quadrature_fine"]) / max(
                          1.0, abs(values["period_quadrature_fine"]))
    expected_oq = abs(values["period_direct_ode"]
                      - values["period_quadrature_fine"]) / max(
                          1.0, abs(values["period_quadrature_fine"]))
    if not (_number_matches(values["matrix_refinement_relative"], expected_refinement)
            and _number_matches(values["q_vs_time_relative"], expected_q)
            and _number_matches(values["u_vs_time_relative"], expected_u)
            and _number_matches(values["quadrature_refinement_relative"], expected_qr)
            and _number_matches(values["ode_vs_quadrature_relative"], expected_oq)):
        return False
    if not (_number_matches(values["determinant_error_coarse"],
                           abs(values["determinant_coarse"] - 1.0))
            and _number_matches(values["determinant_error_fine"],
                                abs(values["determinant_fine"] - 1.0))
            and _number_matches(values["determinant_error_q"],
                                abs(values["determinant_q"] - 1.0))
            and _number_matches(values["determinant_error_u"],
                                abs(values["determinant_u"] - 1.0))):
        return False
    expected_trace = row["matrix_fine"][0][0] + row["matrix_fine"][1][1]
    expected_disc = expected_trace * expected_trace - 4.0 * values["determinant_fine"]
    if not (_number_matches(values["trace_fine"], expected_trace)
            and _number_matches(values["discriminant_fine"], expected_disc)):
        return False
    if expected_disc >= 0.0:
        root = math.sqrt(expected_disc)
        eigs = ((expected_trace + root) / 2.0,
                (expected_trace - root) / 2.0)
        moduli = (abs(eigs[0]), abs(eigs[1]))
    else:
        mod = math.sqrt((expected_trace / 2.0) ** 2
                        + (-expected_disc) / 4.0)
        moduli = (mod, mod)
    if not (_number_matches(values["eigen_modulus_min"], min(moduli))
            and _number_matches(values["eigen_modulus_max"], max(moduli))):
        return False
    if not (0 < values["A_low"] < 1.0
            and abs(values["A_end_direct_ode"] - values["A_low"]) < 3e-7
            and abs(values["q_end_direct_ode"] - math.pi) < 3e-8
            and values["quadrature_refinement_relative"] < 3e-7
            and values["ode_vs_quadrature_relative"] < 3e-6
            and values["trajectory_curve_max_abs"] < 3e-7
            and values["algebraic_curve_constraint_max_abs"] < 3e-7
            and values["matrix_refinement_relative"] < 3e-6
            and values["q_vs_time_relative"] < 3e-6
            and values["u_vs_time_relative"] < 3e-6
            and all(values[name] < 3e-6 for name in (
                "determinant_error_coarse", "determinant_error_fine",
                "determinant_error_q", "determinant_error_u",
                "symplectic_residual_coarse", "symplectic_residual_fine",
                "symplectic_residual_q", "symplectic_residual_u"))):
        return False
    if (row["classification"] not in {"elliptic", "hyperbolic", "parabolic"}
            or row["classification"] != classify_transfer(row["matrix_fine"])):
        return False
    return True


def compute_audit(*, w_values: Sequence[Any] = DEFAULT_W,
                  n_values: Sequence[Any] = DEFAULT_N) -> Dict[str, Any]:
    """Compute exactly the declared finite control set; no parameter hunt."""
    if isinstance(w_values, (str, bytes)) or not isinstance(w_values, Sequence):
        raise ValueError("w_values must be a finite declared sequence")
    if isinstance(n_values, (str, bytes)) or not isinstance(n_values, Sequence):
        raise ValueError("n_values must be a finite declared sequence")
    wv_list = []
    for value in w_values:
        checked, _, _ = _control(value, EPSILON, RHO_C_RATIO)
        if checked not in wv_list:
            wv_list.append(checked)
    nv_list = []
    for value in n_values:
        checked = _check_n_declared(value)
        if checked not in nv_list:
            nv_list.append(checked)
    if not wv_list or not nv_list:
        raise ValueError("at least one declared w and n are required")
    if len(wv_list) > len(DEFAULT_W) or len(nv_list) > len(DEFAULT_N):
        raise ValueError("control set exceeds the declared finite audit")

    symbolic = tensor_action_symbolic_checks()
    static = static_scale_check()
    generator = generator_symplectic_check()
    cycles = [tensor_mode_transfer(w=w, n=n) for w in wv_list for n in nv_list]
    passed_count = sum(1 for row in cycles if mode_acceptance(row))
    classifications = {
        name: sum(1 for row in cycles if row["classification"] == name)
        for name in ("elliptic", "hyperbolic", "parabolic", "numerically_unresolved")
    }
    robust_rows = [
        row for row in cycles
        if row["classification"] == "hyperbolic"
        and row["matrix_refinement_relative"] < 1e-7
        and row["q_vs_time_relative"] < 1e-7
        and row["u_vs_time_relative"] < 1e-7
    ]
    growth_checks = [
        multi_cycle_growing_mode(w=row["w"], n=row["n"],
                                 matrix=row["matrix_fine"])
        for row in robust_rows
    ]
    summary = {
        "mode_count": len(cycles),
        "accepted_mode_count": passed_count,
        "classification_counts": classifications,
        "robust_hyperbolic_modes": [
            {"w": row["w"], "n": row["n"]} for row in cycles
            if row in robust_rows
        ],
        "growth_check_count": len(growth_checks),
        "finite_mode_sample_only": True,
    }
    all_passed = bool(
        rows_pass(symbolic)
        and static["passed"]
        and generator["passed"]
        and passed_count == len(cycles)
        and classifications["numerically_unresolved"] == 0
        and all(growth_check_acceptance(check) for check in growth_checks))
    return _json_safe({
        "status": STATUS,
        "schema": SCHEMA,
        "evidence_weight": EVIDENCE_WEIGHT,
        "mathematical_checks_passed": all_passed,
        "symbolic_checks": symbolic,
        "static_scale_check": static,
        "generator_check": generator,
        "cycles": cycles,
        "growth_checks": growth_checks,
        "summary": summary,
        "normalization": dict(NORMALIZATION),
        "scope": {
            "action": "Einstein-Hilbert TT sector with isotropic matter and cuscuton; no tensor anisotropic stress",
            "background": "fixed epsilon=.1, R=1, declared constant-w manufactured controls",
            "transfer": "raw finite one-cycle Hamiltonian monodromy; no determinant normalization",
            "interpretation": "elliptic/hyperbolic classification is finite sampled evidence, not an all-mode theorem",
            "excluded": "scalar sector, nonlinear/quantum stability, observational/NVG matter claim",
        },
        "sources": SOURCES,
    })


def compute_state() -> Dict[str, Any]:
    """Compatibility-shaped entry point for the complete declared audit."""
    return compute_audit()


def acceptance_from_result(result: Any) -> bool:
    """Recompute acceptance and reject missing/NaN/mutated/large-error data."""
    if not isinstance(result, dict) or set(result) != set(RESULT_REQUIRED_KEYS):
        return False
    if (result.get("status") != STATUS or result.get("schema") != SCHEMA
            or result.get("evidence_weight") != EVIDENCE_WEIGHT
            or result.get("mathematical_checks_passed") is not True):
        return False
    if result.get("normalization") != NORMALIZATION:
        return False
    if result.get("sources") != SOURCES:
        return False
    scope = result.get("scope")
    if (not isinstance(scope, dict)
            or scope.get("transfer") != "raw finite one-cycle Hamiltonian monodromy; no determinant normalization"
            or scope.get("interpretation") != "elliptic/hyperbolic classification is finite sampled evidence, not an all-mode theorem"):
        return False
    if not rows_pass(result.get("symbolic_checks")):
        return False
    if not _static_check_acceptance(result.get("static_scale_check")):
        return False
    if not _generator_check_acceptance(result.get("generator_check")):
        return False
    cycles = result.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        return False
    if any(not mode_acceptance(row) for row in cycles):
        return False
    growth_checks = result.get("growth_checks")
    if not isinstance(growth_checks, list):
        return False
    if any(not growth_check_acceptance(check) for check in growth_checks):
        return False
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return False
    if summary.get("mode_count") != len(cycles):
        return False
    if summary.get("accepted_mode_count") != len(cycles):
        return False
    counts = summary.get("classification_counts")
    if not isinstance(counts, dict):
        return False
    expected_counts = {name: sum(1 for row in cycles if row["classification"] == name)
                       for name in ("elliptic", "hyperbolic", "parabolic", "numerically_unresolved")}
    if counts != expected_counts or counts["numerically_unresolved"] != 0:
        return False
    if summary.get("growth_check_count") != len(growth_checks):
        return False
    if summary.get("finite_mode_sample_only") is not True:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-w", default=None,
                        help="optional one declared control: 0, 1/3, or 1")
    parser.add_argument("--single-n", type=int, default=None,
                        help="optional one declared harmonic n in 3..20")
    args = parser.parse_args(argv)
    try:
        if args.single_w is None and args.single_n is None:
            result = compute_audit()
        else:
            if args.single_w is None or args.single_n is None:
                raise ValueError("--single-w and --single-n must be supplied together")
            parsed_w = float(args.single_w)
            result = compute_audit(w_values=(parsed_w,), n_values=(args.single_n,))
    except (ValueError, ArithmeticError) as exc:
        print(json.dumps({"status": "invalid_or_failed_audit",
                          "evidence_weight": EVIDENCE_WEIGHT,
                          "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["mathematical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
