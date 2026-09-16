#!/usr/bin/env python3
"""Spatial Thomas--Fermi binding/interface audit.

This module is a counterfactual calculation attached to the fixed inverse
design in :mod:`nvg_binding_feasibility_audit`.  It does not alter the
source-complete action or any accepted result.  The static planar closure is
local T=0 Thomas--Fermi matter with ``nu=mu-g*A`` and ``m=g_s*W``.  The
grand functional is a saddle: a minimum in ``W`` and a maximum in ``A``.

The CLI prints a fresh JSON derivation and numerical audit and writes no
artifact.  SciPy is used only for the finite-domain BVP and periodic-control
diagnostics; all fixed design constants and analytic bounds are recomputed
from ``BindingDesign``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Optional

import mpmath as mp
import sympy as sp

try:  # The high-precision identities remain usable without SciPy.
    import numpy as np
    from scipy.integrate import simpson, solve_bvp
    from scipy.optimize import brentq
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh, spsolve
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on minimal installs.
    np = None
    _SCIPY_AVAILABLE = False

HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
STATUS = "counterfactual_spatial_binding_interface_not_NVG_prediction"

# These are explicit finite-domain diagnostics in dimensionless x=W0*z units,
# not analytic enclosures or physical-parameter adjustments.  The final
# (300, 1e-3) solve is required to resolve independent equations and endpoint
# tails at this scale; earlier refinement levels remain reported as lower-
# accuracy approximations when they miss one of the limits.
BVP_EL_RESIDUAL_LIMIT = 5e-5
BVP_STRESS_SPREAD_LIMIT = 5e-6
BVP_ENDPOINT_DERIVATIVE_LIMIT = 5e-5
BVP_BAND_EXCURSION_LIMIT = 2e-4
BVP_VECTOR_UNDERSHOOT_LIMIT = 2e-4
BVP_TENSION_REFINEMENT_LIMIT = 2e-2
BVP_COLLOCATION_FACTOR = 5.0
PERIODIC_GAUSS_RESIDUAL_LIMIT = 5e-10
PERIODIC_ENERGY_IDENTITY_LIMIT = 1e-8
PERIODIC_REFINEMENT_LIMIT = 1e-2

try:
    from nvg_binding_feasibility_audit import BindingDesign
    from source_complete_scaling_saturation_audit import INPUTS
except ImportError:  # Allow direct execution from another working directory.
    sys.path.insert(0, str(HERE))
    from nvg_binding_feasibility_audit import BindingDesign
    from source_complete_scaling_saturation_audit import INPUTS


REQUIRED_SYMBOLIC = frozenset({
    "density_closure",
    "pressure_density_identity",
    "pressure_mass_identity",
    "fermi_susceptibility_identity",
    "pressure_mixed_identity",
    "pressure_mixed_closed_form",
    "scalar_el_equation",
    "scalar_hessian_chain_rule",
    "vector_el_equation",
    "vector_hessian_chain_rule",
    "translation_first_integral",
    "translation_noether_identity",
    "vector_hessian_negative",
    "opposite_vector_action_hessian_formula",
    "mixed_hessian_w_a_physical",
    "mixed_hessian_a_w_physical",
    "mixed_hessian_closed_expression",
    "hessian_cross_symmetry",
    "fixed_density_gauss_stationarity",
    "fixed_density_gauss_energy",
    "envelope_stationarity",
    "envelope_second_derivative",
    "envelope_derivative_formula",
    "tension_integral_formula",
})


def precision(dps: int) -> None:
    if isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 300:
        raise ValueError("dps must be an integer from 80 to 300")


def finite(value, name: str, *, positive: bool = False) -> mp.mpf:
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool is not a physical number")
    try:
        result = mp.mpf(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: finite real number required") from exc
    if not mp.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name}: finite {'positive ' if positive else ''}number required")
    return result


def number(value, digits: int = 35) -> str:
    value = finite(value, "derived number")
    return mp.nstr(value, digits)


def float_number(value, digits: int = 16) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite numerical diagnostic")
    return format(value, f".{digits}g")


def _fermi_pressure_mp(k: mp.mpf, mass: mp.mpf, degeneracy: mp.mpf) -> mp.mpf:
    """Positive pressure integral, with a series near the nonrelativistic edge."""
    k, mass, degeneracy = map(mp.mpf, (k, mass, degeneracy))
    if k <= 0:
        return mp.mpf(0)
    ratio = k / mass
    if ratio <= mp.mpf(".5"):
        term = ratio**5 / 5
        total = term
        coeff = mp.mpf(1)
        for j in range(1, 1000):
            coeff *= -mp.mpf(2 * j - 1) / (2 * j)
            term = coeff * ratio ** (5 + 2 * j) / (5 + 2 * j)
            total += term
            if abs(term) <= mp.eps * max(abs(total), mp.mpf(1)) / 8:
                break
        return degeneracy * mass**4 * total / (6 * mp.pi**2)
    return degeneracy / (48 * mp.pi**2) * (
        k * mp.sqrt(k * k + mass * mass) * (2 * k * k - 3 * mass * mass)
        + 3 * mass**4 * mp.asinh(k / mass)
    )


def _fermi_scalar_density_mp(k: mp.mpf, mass: mp.mpf, degeneracy: mp.mpf) -> mp.mpf:
    if k <= 0:
        return mp.mpf(0)
    ef = mp.sqrt(k * k + mass * mass)
    return degeneracy * mass / (4 * mp.pi**2) * (
        k * ef - mass * mass * mp.asinh(k / mass)
    )


def _mp_al(design: BindingDesign, y: mp.mpf) -> mp.mpf:
    """Pointwise local vector maximizer in MeV, including the onset branch."""
    y = finite(y, "y", positive=True)
    onset = design.mu / design.base.MN
    if y >= onset:
        return mp.mpf(0)
    state = design.envelope_state(y)
    W = design.base.W0 * y
    q = design.base.momega / design.base.W0
    return design.gomega * state["n"] / (q * q * W * W)


def local_envelope_state(design: BindingDesign, y) -> dict:
    """Return the fixed-design ``n_bar(W), A_L(W)`` state and derivative data."""
    with design._precision():
        y = finite(y, "y", positive=True)
        W0 = design.base.W0
        gs = design.base.MN / W0
        q = design.base.momega / W0
        W = W0 * y
        onset = design.mu / design.base.MN
        if y >= onset:
            return {
                "y": y, "W": W, "n": mp.mpf(0), "k": mp.mpf(0),
                "EF": design.mu, "A_L": mp.mpf(0), "dA_dW": mp.mpf(0),
                "b": mp.mpf(0), "a_F": mp.mpf(0), "M2": q * q * W * W,
                "stationarity_residual": mp.mpf(0),
                "active": False,
            }
        state = design.envelope_state(y)
        n, k = state["n"], state["k"]
        mass = gs * W
        ef = mp.sqrt(k * k + mass * mass)
        A = design.gomega * n / (q * q * W * W)
        M2 = q * q * W * W
        a_f = k * k / (3 * n * ef)
        b = gs * mass / ef
        derivative = -(
            design.gomega * b + 2 * a_f * M2 * A / W
        ) / (design.gomega**2 + a_f * M2)
        stationarity_residual = ef + design.Cv * n / (y * y) - design.mu
        return {
            "y": y, "W": W, "n": n, "k": k, "EF": ef, "A_L": A,
            "dA_dW": derivative, "b": b, "a_F": a_f, "M2": M2,
            "stationarity_residual": stationarity_residual,
            "active": True,
        }


def symbolic_checks(*, scalar_force_sign: int = 1,
                    vector_source_sign: int = 1,
                    vector_action_sign: int = -1) -> dict:
    """Differentiate the actual TF functional and its Legendre envelope.

    The optional sign arguments are deliberate mutations.  The default action
    has a negative vector quadratic form and the force equations below are
    compared with that fixed physical convention.  No row is a subtraction of
    an expression from itself: pressure derivatives, Euler variations,
    Hessians, the Gauss completion of the square and the envelope derivatives
    are formed independently.
    """
    for name, value in (("scalar_force_sign", scalar_force_sign),
                        ("vector_source_sign", vector_source_sign),
                        ("vector_action_sign", vector_action_sign)):
        if isinstance(value, bool) or value not in (-1, 1):
            raise ValueError(f"{name} must be -1 or +1")

    # Active Thomas--Fermi pressure as a function of independent chemical
    # potential nu and mass m.  The positive branch nu>m is part of this
    # derivation; the inactive n=P=0 branch is handled by the numerical model.
    W, A, z, gs, q, g = sp.symbols("W A z g_s q g", positive=True)
    mu = sp.symbols("mu", positive=True)
    Wp, Ap, Wpp, App = sp.symbols("W_p A_p W_pp A_pp", real=True)
    U0, U1, U2 = sp.symbols("U U_W U_WW", real=True)
    nu, m, d = sp.symbols("nu m d", positive=True)
    k_active = sp.sqrt(nu**2 - m**2)
    n_expr = d * k_active**3 / (6 * sp.pi**2)
    pressure = d / (48 * sp.pi**2) * (
        k_active * nu * (2 * k_active**2 - 3 * m**2)
        + 3 * m**4 * sp.asinh(k_active / m)
    )
    scalar_density = d * m / (4 * sp.pi**2) * (
        k_active * nu - m**2 * sp.asinh(k_active / m)
    )
    susceptibility = d * nu * k_active / (2 * sp.pi**2)
    pressure_nu = sp.diff(pressure, nu)
    pressure_m = sp.diff(pressure, m)
    pressure_nunu = sp.diff(pressure_nu, nu)
    pressure_mm = sp.diff(pressure, m, 2)
    pressure_num = sp.diff(pressure_nu, m)
    pressure_num_closed = -d * m * k_active / (2 * sp.pi**2)
    pressure_WW_chain = sp.diff(pressure.subs(m, gs * W), W, 2)
    pressure_AA_chain = sp.diff(pressure.subs(nu, mu - g * A), A, 2)

    M2 = q**2 * W**2
    # Chain rule: m=g_s W and nu=mu-g A.  Keep nu,m independent so the
    # positivity assumptions remain visible to SymPy.
    dL_dW = U1 + vector_action_sign * q**2 * W * A**2 - gs * pressure_m
    dL_dA = vector_action_sign * M2 * A + g * pressure_nu
    scalar_euler = Wpp - dL_dW
    vector_euler = vector_action_sign * App - dL_dA
    scalar_expected = Wpp - (U1 + scalar_force_sign * gs * scalar_density - q**2 * W * A**2)
    vector_expected = -App + M2 * A - vector_source_sign * g * n_expr

    L = (sp.Rational(1, 2) * Wp**2
         + vector_action_sign * (sp.Rational(1, 2) * Ap**2
                                 + sp.Rational(1, 2) * M2 * A**2)
         + U0 - pressure)
    hamiltonian = Wp * sp.diff(L, Wp) + Ap * sp.diff(L, Ap) - L
    expected_hamiltonian = (sp.Rational(1, 2) * Wp**2
                            - sp.Rational(1, 2) * Ap**2 - U0
                            + sp.Rational(1, 2) * M2 * A**2 + pressure)
    # The total derivative includes U_W and the chain-rule rates
    # nu_z=-g A_z, m_z=g_s W_z.
    hamiltonian_prime = (
        sp.diff(hamiltonian, W) * Wp
        + sp.diff(hamiltonian, A) * Ap
        + sp.diff(hamiltonian, Wp) * Wpp
        + sp.diff(hamiltonian, Ap) * App
        + sp.diff(hamiltonian, U0) * U1 * Wp
        + sp.diff(hamiltonian, nu) * (-g * Ap)
        + sp.diff(hamiltonian, m) * gs * Wp
    )
    scalar_hessian = U2 + vector_action_sign * q**2 * A**2 - gs**2 * pressure_mm
    vector_hessian = vector_action_sign * (z + M2) - g**2 * pressure_nunu
    scalar_hessian_chain = (
        U2 + vector_action_sign * q**2 * A**2 - pressure_WW_chain
    )
    vector_hessian_chain = (
        vector_action_sign * (z + M2) - pressure_AA_chain
    )
    # The mixed W/A derivative must be taken after imposing the physical
    # chain nu=mu-g*A, m=g_s*W.  Keeping nu,m independent above is useful for
    # the one-field pressure identities, but it intentionally cannot capture
    # the fermionic cross term.  Differentiate the actual substituted action
    # in both orders, then compare it to the independently differentiated
    # P_(nu,m) closed form.  The default vector action sign is -1, hence the
    # vector contribution is -2*q^2*W*A.
    physical_substitutions = {
        nu: mu - g * A,
        m: gs * W,
    }
    physical_pressure = pressure.subs(physical_substitutions, simultaneous=True)
    physical_L = (
        sp.Rational(1, 2) * Wp**2
        + vector_action_sign * (
            sp.Rational(1, 2) * Ap**2 + sp.Rational(1, 2) * M2 * A**2
        )
        + U0 - physical_pressure
    )
    physical_dL_dW = sp.diff(physical_L, W)
    physical_dL_dA = sp.diff(physical_L, A)
    mixed_w_a_physical = sp.diff(physical_dL_dW, A)
    mixed_a_w_physical = sp.diff(physical_dL_dA, W)
    pressure_num_physical = pressure_num.subs(
        physical_substitutions, simultaneous=True
    )
    pressure_num_closed_physical = pressure_num_closed.subs(
        physical_substitutions, simultaneous=True
    )
    mixed_hessian_closed = -2 * q**2 * W * A + g * gs * pressure_num_physical

    # Fixed-density Legendre envelope.  F_n=E_F and F_nn=a_F are themselves
    # obtained by differentiating the positive Fermi energy integral.
    ne, me, de, mu_e, Cw, We = sp.symbols(
        "n_e m_e d_e mu_e C_w W_e", positive=True
    )
    ke = (6 * sp.pi**2 * ne / de)**sp.Rational(1, 3)
    efe = sp.sqrt(ke**2 + me**2)
    fermi_energy = de / (16 * sp.pi**2) * (
        ke * efe * (2 * ke**2 + me**2) - me**4 * sp.asinh(ke / me)
    )
    envelope = mu_e * ne - fermi_energy - Cw * ne**2 / (2 * We**2)
    a_f = ke**2 / (3 * ne * efe)
    envelope_n = sp.diff(envelope, ne)
    envelope_nn = sp.diff(envelope, ne, 2)
    b_e = sp.symbols("b_e", positive=True)
    n_w = (2 * Cw * ne / We**3 - b_e) / (a_f + Cw / We**2)
    A_env = g * ne / (q**2 * We**2)
    dA_from_stationarity = sp.diff(A_env, ne) * n_w + sp.diff(A_env, We)
    Cw_relation = g**2 / q**2
    weighted_dA = -(
        g * b_e + 2 * a_f * (q**2 * We**2) * A_env / We
    ) / (g**2 + a_f * q**2 * We**2)

    # The fixed-density vector sector is a concave quadratic in A.  Its
    # stationary value is the positive energy 1/2 J K^{-1}J.
    J = g * sp.symbols("n_source", positive=True)
    K = z + M2
    A_star = J / K
    omega_A = -sp.Rational(1, 2) * K * A**2 + J * A

    yy = sp.symbols("yy", positive=True)
    ys = sp.symbols("y_star", positive=True)
    p_momentum = sp.symbols("p_momentum", nonnegative=True)
    sigma_y_integral = sp.integrate(
        (1 - yy**2) * (yy**2 - ys**2), (yy, ys, 1)
    )
    sigma_formula_y = sp.Rational(2, 15) * (1 - ys)**3 * (
        ys**2 + 3 * ys + 1
    )

    identities = {
        "density_closure": (
            sp.integrate(
                d / (2 * sp.pi**2) * p_momentum**2,
                (p_momentum, 0, k_active),
            )
            - n_expr
        ),
        "pressure_density_identity": pressure_nu - n_expr,
        "pressure_mass_identity": pressure_m + scalar_density,
        "fermi_susceptibility_identity": pressure_nunu - susceptibility,
        "pressure_mixed_identity": pressure_num + sp.diff(scalar_density, nu),
        "pressure_mixed_closed_form": pressure_num - pressure_num_closed,
        "scalar_el_equation": scalar_euler - scalar_expected,
        "scalar_hessian_chain_rule": scalar_hessian_chain - scalar_hessian.subs(m, gs * W),
        "vector_el_equation": vector_euler - vector_expected,
        "vector_hessian_chain_rule": (
            vector_hessian_chain
            - (vector_action_sign * (z + M2)
               - g**2 * pressure_nunu.subs(nu, mu - g * A))
        ),
        "translation_first_integral": hamiltonian - expected_hamiltonian,
        "translation_noether_identity": hamiltonian_prime - Wp * scalar_euler - Ap * vector_euler,
        "vector_hessian_negative": vector_hessian - (
            -(z + M2 + g**2 * susceptibility)
        ),
        "opposite_vector_action_hessian_formula": (
            (z + M2 - g**2 * susceptibility)
            - (z + M2 - g**2 * pressure_nunu)
        ),
        "mixed_hessian_w_a_physical": mixed_w_a_physical - mixed_hessian_closed,
        "mixed_hessian_a_w_physical": mixed_a_w_physical - mixed_hessian_closed,
        "mixed_hessian_closed_expression": (
            sp.refine(
                sp.simplify(
                    mixed_hessian_closed
                    - (-2 * q**2 * W * A + g * gs * pressure_num_closed_physical)
                ),
                sp.Q.positive(mu - g * A),
            )
        ),
        "hessian_cross_symmetry": mixed_w_a_physical - mixed_a_w_physical,
        "fixed_density_gauss_stationarity": sp.diff(omega_A, A).subs(A, A_star),
        "fixed_density_gauss_energy": omega_A.subs(A, A_star) - J**2 / (2 * K),
        "envelope_stationarity": envelope_n - (
            mu_e - efe - Cw * ne / We**2
        ),
        "envelope_second_derivative": envelope_nn - (
            -a_f - Cw / We**2
        ),
        "envelope_derivative_formula": (
            dA_from_stationarity.subs(Cw, Cw_relation) - weighted_dA
        ),
        "tension_integral_formula": sigma_y_integral - sigma_formula_y,
    }
    result = {}
    for key, value in identities.items():
        simplified = sp.factor(sp.simplify(value))
        result[key] = {
            "residual": str(simplified),
            "passed": bool(simplified == 0),
        }
    return result


def _symbolic_rows_passed(rows: dict) -> bool:
    """Fail closed on missing rows, nonzero residuals or truthy non-booleans."""
    return bool(
        isinstance(rows, dict)
        and set(rows) == REQUIRED_SYMBOLIC
        and all(
            isinstance(row, dict)
            and row.get("passed") is True
            and row.get("residual") == "0"
            for row in rows.values()
        )
    )


def analytic_certificates(design: BindingDesign) -> dict:
    """Return inequality premises separately from exact symbolic identities.

    These entries intentionally have no fake zero residual.  Their statements
    record which positive-integrand or square argument carries the inequality;
    the booleans only certify the fixed design's declared parameter domain.
    Finite samples are diagnostics and never replace these premises.
    """
    with design._precision():
        W0 = design.base.W0
        gs = design.base.MN / W0
        q = design.base.momega / W0
        Wstar = W0 * design.y
        g = design.gomega
        eta_scalar = gs / g
        eta_vector = 2 * (design.mu - gs * Wstar) / (g * Wstar)
        eta = max(eta_scalar, eta_vector)
        z2 = mp.mpf(2)
        barrier_ratio_at_z2 = (
            (z2 - 1) ** 2 * (z2 - design.y**2) ** 2 / z2**4
        )
        gv_actual = design.gomega**2 / q**2
        return {
            "envelope_floor_nonnegative": {
                "passed": bool(design.Cv > 0 and design.base.d > 0),
                "kind": "analytic_positive_integrand",
                "statement": "U_floor=P_F+Cv*n_bar^2/(2*y^2), with P_F and the vector term nonnegative",
            },
            "barrier_nonnegative": {
                "passed": bool(design.beta > 0),
                "kind": "analytic_square",
                "statement": "B=beta*(y^2-1)^2*(y^2-y_star^2)^2",
            },
            "barrier_tail_bound": {
                "passed": bool(design.beta > 0 and 0 < design.y <= 1
                                  and barrier_ratio_at_z2 >= mp.mpf(1) / 16),
                "kind": "analytic_monotone_tail",
                "statement": "for z=y^2>=2 and y_star^2<=1, B/(beta*y^8)>=1/16",
                "ratio_at_z2": number(barrier_ratio_at_z2),
            },
            "fermi_enthalpy_bound": {
                "passed": bool(design.base.d > 0),
                "kind": "analytic_positive_integrand",
                "statement": "n*E_F=F_F+P_F and F_F-P_F=m*n_s>=0, so Q>0 and Q<=2*rho on n>0",
                "strict_scope": "n>0 or scalar speed nonzero; U>=0 and vector energy>=0",
            },
            "positive_gauss_operator": {
                "passed": bool(q > 0 and Wstar > 0),
                "kind": "elliptic_operator_premise",
                "statement": "-d_z^2+q^2*W^2 has positive minimum mass on W>=W_star>0",
                "actual_Gv": number(gv_actual),
            },
            "uniform_envelope_slope_bound": {
                "passed": bool(eta < 1 and eta_scalar >= eta_vector),
                "kind": "analytic_weighted_mean_bound",
                "statement": "|dA_L/dW| is bounded by eta=max(g_s/g,2*(mu-g_s*W_star)/(g*W_star))",
                "eta_scalar": number(eta_scalar),
                "eta_vector": number(eta_vector),
                "eta": number(eta),
            },
            "sandwich_lower_direction": {
                "passed": bool(eta < 1),
                "kind": "analytic_inequality",
                "statement": "evaluate Omega at A_L and use |A_L'|<=eta<1",
            },
            "sandwich_upper_direction": {
                "passed": bool(q > 0 and design.beta > 0),
                "kind": "analytic_saddle_inequality",
                "statement": "sup_A Omega is bounded above by the pointwise A maximum and negative A-gradient term",
            },
        }


def _analytic_certificates_passed(certificates: dict) -> bool:
    """Validate inequality premises without treating a string as evidence."""
    required = {
        "envelope_floor_nonnegative", "barrier_nonnegative", "barrier_tail_bound",
        "fermi_enthalpy_bound", "positive_gauss_operator",
        "uniform_envelope_slope_bound", "sandwich_lower_direction",
        "sandwich_upper_direction",
    }
    if not isinstance(certificates, dict) or set(certificates) != required:
        return False
    for row in certificates.values():
        if not isinstance(row, dict) or row.get("passed") is not True:
            return False
        if not isinstance(row.get("kind"), str) or not isinstance(row.get("statement"), str):
            return False
        if "residual" in row:
            return False
        for key, value in row.items():
            if key in {"passed", "kind", "statement", "strict_scope"}:
                continue
            if isinstance(value, bool):
                return False
            try:
                if not mp.isfinite(mp.mpf(value)):
                    return False
            except (TypeError, ValueError, OverflowError):
                return False
    return True


def _envelope_evidence_passed(envelope: dict, regularity: dict) -> bool:
    """Require computed derivative/onset evidence, not only an eta label."""
    if not isinstance(envelope, dict) or not isinstance(regularity, dict):
        return False
    if not all(envelope.get(key) is True for key in (
        "first_term_dominates", "eta_less_than_one"
    )):
        return False
    if not all(_finite_float(envelope.get(key)) for key in (
        "eta_scalar_gs_over_g", "eta_vector_on_band", "eta"
    )):
        return False
    eta_scalar = float(envelope["eta_scalar_gs_over_g"])
    eta_vector = float(envelope["eta_vector_on_band"])
    eta = float(envelope["eta"])
    if not (eta < 1.0 and eta >= 0.0 and eta_scalar >= eta_vector
            and abs(eta - max(eta_scalar, eta_vector)) < 1e-12):
        return False
    samples = envelope.get("samples")
    if not isinstance(samples, list) or not samples:
        return False
    for row in samples:
        if not isinstance(row, dict) or row.get("below_eta") is not True:
            return False
        if not all(_finite_float(row.get(key)) for key in (
            "y", "formula_dA_dW", "direct_dA_dW", "residual",
            "abs_slope", "stationarity_residual"
        )):
            return False
        direct_minus_formula = float(row["direct_dA_dW"]) - float(row["formula_dA_dW"])
        if (abs(direct_minus_formula - float(row["residual"])) > 1e-50
                or abs(abs(float(row["formula_dA_dW"])) - float(row["abs_slope"])) > 1e-50
                or abs(float(row["residual"])) > 1e-50
                or abs(float(row["stationarity_residual"])) > 1e-50
                or abs(float(row["abs_slope"])) > eta
                or row.get("below_eta") is not True):
            return False
    rows = regularity.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        for key in ("x_MeV", "n_over_x_3_2", "floor_over_x_5_2", "A_L_over_x_3_2"):
            if not _finite_float(row.get(key)):
                return False
    return bool(
        _finite_float(regularity.get("A_L_above_onset"))
        and float(regularity["A_L_above_onset"]) == 0.0
        and abs(float(rows[-1]["floor_over_x_5_2"]) - 1.0) < 1e-10
    )


def _tension_evidence_passed(bounds: dict) -> bool:
    """Check exact quadrature/formula agreement and finite bound ordering."""
    if not isinstance(bounds, dict):
        return False
    required = (
        "sigma0_MeV3_quadrature", "sigma0_MeV3_formula",
        "sigma0_MeV_fm_minus2", "sigma0_integral_residual",
        "lower_bound_MeV3", "upper_bound_MeV3",
        "lower_bound_le_upper_bound", "lower_direction_certified",
        "upper_direction_certified", "inequality_certificates_are_analytic",
        "bound_is_for_profiles_in_Wstar_to_W0_band", "not_global_unrestricted_minimum",
    )
    if any(key not in bounds for key in required):
        return False
    for key in ("sigma0_MeV3_quadrature", "sigma0_MeV3_formula",
                "sigma0_MeV_fm_minus2", "sigma0_integral_residual",
                "lower_bound_MeV3", "upper_bound_MeV3"):
        if not _finite_float(bounds[key]):
            return False
    return bool(
        abs(float(bounds["sigma0_integral_residual"])) < 1e-40
        and abs(float(bounds["sigma0_MeV3_quadrature"])
                - float(bounds["sigma0_MeV3_formula"])
                - float(bounds["sigma0_integral_residual"])) < 1e-30
        and float(bounds["sigma0_MeV_fm_minus2"]) > 0
        and float(bounds["lower_bound_MeV3"]) > 0
        and float(bounds["upper_bound_MeV3"]) > float(bounds["lower_bound_MeV3"])
        and all(bounds[key] is True for key in required[6:])
    )


def _interface_refinement_passed(result: dict) -> bool:
    """Validate the final finite-domain approximation and its refinement."""
    if not isinstance(result, dict) or result.get("available") is not True:
        return False
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or len(attempts) < 2:
        return False
    final = result.get("final")
    if not isinstance(final, dict):
        return False
    # Earlier points may be lower-accuracy approximations, but they must still
    # carry finite diagnostics so the last-step comparison is meaningful.
    diagnostic_keys = (
        "solver_status", "collocation_max_rms", "max_EL_residual_dimensionless",
        "max_W_EL_residual_dimensionless", "max_Gauss_residual_dimensionless",
        "independent_residual_limit", "stress_spread",
        "boundary_derivative_max_dimensionless", "min_y", "max_y",
        "vector_field_negative_undershoot", "tension_MeV3", "tolerance",
        "band_excursion_below", "band_excursion_above", "stress_min", "stress_max",
        "boundary_derivatives", "finite_domain_accuracy_passed",
        "finite_domain_approximation", "independent_residual_passed",
        "stress_criterion_passed", "boundary_derivative_passed",
        "band_excursion_passed", "vector_undershoot_within_finite_domain_limit",
    )
    numeric_keys = tuple(key for key in diagnostic_keys if key not in {
        "solver_status", "boundary_derivatives", "finite_domain_accuracy_passed",
        "finite_domain_approximation", "independent_residual_passed",
        "stress_criterion_passed", "boundary_derivative_passed",
        "band_excursion_passed", "vector_undershoot_within_finite_domain_limit",
    })
    for row in attempts:
        if (not isinstance(row, dict)
                or any(key not in row for key in diagnostic_keys)):
            return False
        if (not all(_finite_float(row[key]) for key in numeric_keys)
                or not isinstance(row["solver_status"], int)
                or isinstance(row["solver_status"], bool)):
            return False
        if (not isinstance(row["boundary_derivatives"], list)
                or len(row["boundary_derivatives"]) != 4
                or not all(_finite_float(value) for value in row["boundary_derivatives"])):
            return False
        if not all(isinstance(row[key], bool) for key in (
            "finite_domain_accuracy_passed", "finite_domain_approximation",
            "independent_residual_passed", "stress_criterion_passed",
            "boundary_derivative_passed", "band_excursion_passed",
            "vector_undershoot_within_finite_domain_limit",
        )):
            return False
        if row.get("finite_domain_approximation") is not True:
            return False
    previous = attempts[-2]
    if not (_finite_float(previous.get("tension_MeV3"))
            and _finite_float(final.get("tension_MeV3"))
            and _finite_float(final.get("successive_tension_relative_difference"))):
        return False
    previous_tension = float(previous["tension_MeV3"])
    final_tension = float(final["tension_MeV3"])
    relative = abs(final_tension - previous_tension) / max(abs(final_tension), 1e-30)
    expected_refinement = bool(relative <= BVP_TENSION_REFINEMENT_LIMIT)
    return bool(
        result.get("finite_domain_refinement_valid") is True
        and result.get("converged_coupled_solution") is True
        and result.get("tension_refinement_passed") is expected_refinement
        and final.get("successive_tension_difference_passed") is True
        and result.get("infinite_wall_certified") is False
        and _interface_attempt_passes(final)
        and expected_refinement
        and abs(relative - float(final["successive_tension_relative_difference"])) < 1e-7
        and _finite_float(result.get("tension_refinement_limit"))
        and result.get("tension_band_is_diagnostic_only") is True
    )


def _periodic_controls_passed(result: dict) -> bool:
    """Validate Gauss/refinement evidence while ignoring energy sign."""
    if not isinstance(result, dict) or result.get("available") is not True:
        return False
    trials = result.get("trials")
    if not isinstance(trials, list) or not trials:
        return False
    if result.get("calculation_valid") is not True:
        return False
    witness = False
    expected_rows_valid = []
    for row in trials:
        if not isinstance(row, dict):
            return False
        if not _finite_float(row.get("omega_relative_refinement")):
            return False
        grid_valid = {}
        for grid in ("coarse", "fine"):
            if not isinstance(row.get(grid), dict):
                return False
            sample = row[grid]
            classified = classify_periodic_sample(sample)
            grid_valid[grid] = bool(
                sample.get("calculation_valid") is True
                    and classified.get("calculation_valid") is True
            )
            if not grid_valid[grid]:
                return False
            if not all(sample.get(key) is True for key in (
                "uses_actual_positive_gauss_operator",
                "coefficients_unchanged",
                "energy_sign_is_not_a_validity_gate",
            )):
                return False
            witness = bool(witness or classified.get("negative_witness") is True)
        coarse_omega = float(row["coarse"]["omega_E_minus_muN_MeV3"])
        fine_omega = float(row["fine"]["omega_E_minus_muN_MeV3"])
        relative = (fine_omega - coarse_omega) / max(abs(fine_omega), 1e-30)
        refinement_ok = bool(abs(relative) <= PERIODIC_REFINEMENT_LIMIT)
        if (row.get("calculation_valid_both_grids") is not all(grid_valid.values())
                or row.get("energy_refinement_passed") is not refinement_ok
                or not refinement_ok
                or abs(relative - float(row["omega_relative_refinement"])) > 1e-7):
            return False
        positive_both = bool(coarse_omega > 0 and fine_omega > 0)
        if row.get("positive_both_grids") is not positive_both:
            return False
        expected_rows_valid.append(bool(all(grid_valid.values()) and refinement_ok))
    expected_calculation_valid = bool(all(expected_rows_valid))
    if (result.get("calculation_valid") is not expected_calculation_valid
            or result.get("negative_E_minus_muN_witness") is not witness
            or result.get("valid_negative_witness_present") is not witness
            or result.get("numerical_validity_failed") is not (not expected_calculation_valid)):
        return False
    expected_interpretation = (
        "valid_negative_periodic_witness" if witness and expected_calculation_valid
        else "valid_negative_witness_but_other_trials_failed" if witness
        else "numerical_failure_or_insufficient_refinement" if not expected_calculation_valid
        else "no_negative_witness_in_finite_trials"
    )
    expected_unrestricted_status = (
        "refuted_by_valid_negative_periodic_witness" if witness
        else "undecided_numerical_validity_failed" if not expected_calculation_valid
        else "undecided_positive_samples_only"
    )
    return bool(
        result.get("interpretation") == expected_interpretation
        and result.get("unrestricted_extension_status") == expected_unrestricted_status
        and result.get("positive_samples_do_not_prove_global_minimum") is True
        and result.get("no_unbounded_optimizer_or_parameter_fit") is True
    )


def derivative_bound(design: BindingDesign) -> dict:
    """Check the weighted-mean formula and a uniform bound on ``dA_L/dW``."""
    with design._precision():
        W0 = design.base.W0
        gs = design.base.MN / W0
        g = design.gomega
        q = design.base.momega / W0
        Wstar = W0 * design.y
        eta_scalar = gs / g
        eta_vector = 2 * (design.mu - gs * Wstar) / (g * Wstar)
        eta = max(eta_scalar, eta_vector)
        samples = []
        # Stay away from the nonanalytic onset for derivative comparisons;
        # regularity is checked separately by the fractional-power ratios.
        for fraction in ("0", ".2", ".4", ".6", ".8", ".95"):
            y = design.y + (design.mu / design.base.MN - design.y) * mp.mpf(fraction)
            state = local_envelope_state(design, y)
            # Evaluate a direct W derivative at the same point.  This is
            # deliberately independent of the closed-form weighted mean.
            direct = mp.diff(lambda WW: _mp_al(design, WW / W0), state["W"])
            samples.append({
                "y": number(y),
                "formula_dA_dW": number(state["dA_dW"]),
                "direct_dA_dW": number(direct),
                "residual": number(direct - state["dA_dW"]),
                "stationarity_residual": number(state["stationarity_residual"]),
                "abs_slope": number(abs(state["dA_dW"])),
                "below_eta": bool(abs(state["dA_dW"]) <= eta),
            })
        onset = design.mu / design.base.MN
        return {
            "eta_scalar_gs_over_g": number(eta_scalar),
            "eta_vector_on_band": number(eta_vector),
            "eta": number(eta),
            "first_term_dominates": bool(eta_scalar >= eta_vector),
            "eta_less_than_one": bool(eta < 1),
            "onset_y": number(onset),
            "samples": samples,
            "above_onset_A_L": number(local_envelope_state(design, onset * mp.mpf("1.001"))["A_L"]),
        }


def onset_regularity(design: BindingDesign) -> dict:
    """Verify ``n_bar,A_L~x^(3/2)`` and ``floor~x^(5/2)`` at the onset."""
    with design._precision():
        mu, MN, d = design.mu, design.base.MN, design.base.d
        onset = mu / MN
        ncoef = d * (2 * mu) ** mp.mpf("1.5") / (6 * mp.pi**2)
        ucoef = 2 * ncoef / 5
        rows = []
        for power in (8, 16, 24, 32):
            x = mp.mpf(10) ** (-power)
            y = (mu - x) / MN
            state = design.envelope_state(y)
            rows.append({
                "x_MeV": number(x),
                "n_over_x_3_2": number(state["n"] / (ncoef * x**mp.mpf("1.5"))),
                "floor_over_x_5_2": number(state["value"] / (ucoef * x**mp.mpf("2.5"))),
                "A_L_over_x_3_2": number(local_envelope_state(design, y)["A_L"] / x**mp.mpf("1.5")),
            })
        return {
            "onset_y": number(onset),
            "density_power": "3/2",
            "floor_power": "5/2",
            "A_power": "3/2",
            "density_coefficient": number(ncoef),
            "floor_coefficient": number(ucoef),
            "rows": rows,
            "A_L_above_onset": number(local_envelope_state(design, onset)["A_L"]),
            "regularity_statement": "n_bar and A_L are C1 at onset; floor is C2 but not C3",
        }


def tension_bounds(design: BindingDesign) -> dict:
    """Independent exact and quadrature evaluation of the bare scalar wall."""
    with design._precision():
        ys = design.y
        W0 = design.base.W0
        beta = design.beta
        integrand = lambda y: W0 * mp.sqrt(2 * design.barrier(y))
        quadrature = mp.quad(integrand, [ys, 1])
        formula = W0 * mp.sqrt(2 * beta) * mp.mpf(2) / 15 * (1 - ys) ** 3 * (
            ys**2 + 3 * ys + 1
        )
        hbarc = design.base.hbarc
        eta_data = derivative_bound(design)
        eta = mp.mpf(eta_data["eta"])
        certificates = analytic_certificates(design)
        lower = mp.sqrt(1 - eta * eta) * formula
        return {
            "sigma0_MeV3_quadrature": number(quadrature),
            "sigma0_MeV3_formula": number(formula),
            "sigma0_integral_residual": number(quadrature - formula),
            "sigma0_MeV_fm_minus2": number(formula / hbarc**2),
            "eta": number(eta),
            "lower_bound_MeV3": number(lower),
            "upper_bound_MeV3": number(formula),
            "lower_bound_MeV_fm_minus2": number(lower / hbarc**2),
            "upper_bound_MeV_fm_minus2": number(formula / hbarc**2),
            "sandwich": "integral[.5(1-eta^2) W'^2+B] <= Omega_eff <= integral[.5 W'^2+B]",
            "lower_bound_le_upper_bound": bool(lower <= formula),
            "lower_direction_certified": bool(certificates["sandwich_lower_direction"]["passed"]),
            "upper_direction_certified": bool(certificates["sandwich_upper_direction"]["passed"]),
            "inequality_certificates_are_analytic": True,
            "bound_is_for_profiles_in_Wstar_to_W0_band": True,
            "not_global_unrestricted_minimum": True,
        }


class _DoubleSpatialModel:
    """Double-precision adapter used only by finite-domain numerical controls."""

    def __init__(self, design: BindingDesign):
        if not _SCIPY_AVAILABLE:
            raise RuntimeError("SciPy is unavailable for spatial numerical controls")
        self.W0 = float(design.base.W0)
        self.MN = float(design.base.MN)
        self.mu = float(design.mu)
        self.g = float(design.gomega)
        self.q = float(design.base.momega / design.base.W0)
        self.gs = self.MN / self.W0
        self.d = float(design.base.d)
        self.hbarc = float(design.base.hbarc)
        self.Cv = float(design.Cv)
        self.beta = float(design.beta)
        self.ys = float(design.y)
        self.zs = self.ys * self.ys - 1.0
        self.mvac = math.sqrt(8.0 * self.beta * self.zs * self.zs) / self.W0

    def nbar(self, y):
        y = np.asarray(y, dtype=float)
        yy = np.maximum(y, 1e-7)
        mass = self.MN * yy
        active = mass < self.mu
        k = np.sqrt(np.maximum(self.mu * self.mu - mass * mass, 0.0))
        coeff = self.Cv * self.d / (6.0 * math.pi**2 * yy * yy)
        for _ in range(14):
            ef = np.sqrt(k * k + mass * mass)
            f = ef + coeff * k**3 - self.mu
            df = np.divide(k, ef, out=np.ones_like(k), where=ef > 0) + 3.0 * coeff * k * k
            k = np.where(active, np.maximum(k - np.divide(f, df, out=np.zeros_like(f), where=df > 0), 0.0), 0.0)
        return self.d * k**3 / (6.0 * math.pi**2)

    def _k_from_n(self, n, y):
        return np.cbrt(np.maximum(6.0 * math.pi**2 * n / self.d, 0.0))

    def al(self, y):
        yy = np.maximum(np.asarray(y, dtype=float), 1e-7)
        return self.g * self.nbar(yy) / (self.q * self.q * self.W0**3 * yy * yy)

    def matter(self, y, a):
        yy = np.maximum(np.asarray(y, dtype=float), 1e-7)
        aa = np.asarray(a, dtype=float)
        nu = self.mu - self.g * self.W0 * aa
        mass = self.MN * yy
        k2 = np.maximum((nu - mass) * (nu + mass), 0.0)
        active = nu > mass
        k = np.sqrt(k2)
        n = np.where(active, self.d * k**3 / (6.0 * math.pi**2), 0.0)
        ns = np.where(
            active,
            self.d * mass / (4.0 * math.pi**2) * (
                k * nu - mass * mass * np.arcsinh(np.divide(k, mass, out=np.zeros_like(k), where=mass > 0))
            ),
            0.0,
        )
        chi = np.where(active, self.d * nu * k / (2.0 * math.pi**2), 0.0)
        return nu, mass, k, n, ns, chi

    def pressure(self, nu, mass, k):
        """Vectorized positive pressure with a convergent small-k series."""
        nu = np.asarray(nu, dtype=float)
        mass = np.maximum(np.asarray(mass, dtype=float), 1e-12)
        k = np.asarray(k, dtype=float)
        ratio = np.divide(k, mass, out=np.zeros_like(k), where=mass > 0)
        small = ratio <= 0.35
        # Integral m^4 sum_j binom(-1/2,j) r^(5+2j)/(5+2j).
        term = ratio**5 / 5.0
        series = term.copy()
        coeff = np.ones_like(ratio)
        for j in range(1, 24):
            coeff *= -(2.0 * j - 1.0) / (2.0 * j)
            term = coeff * ratio ** (5 + 2 * j) / (5.0 + 2.0 * j)
            series += term
        p_small = self.d * mass**4 * series / (6.0 * math.pi**2)
        p_full = self.d / (48.0 * math.pi**2) * (
            k * nu * (2.0 * k * k - 3.0 * mass * mass)
            + 3.0 * mass**4 * np.arcsinh(ratio)
        )
        return np.where(nu > mass, np.where(small, p_small, p_full), 0.0)

    def barrier(self, y):
        yy = np.asarray(y, dtype=float)
        return self.beta * (yy * yy - 1.0) ** 2 * (yy * yy - self.ys * self.ys) ** 2

    def barrier_y(self, y):
        yy = np.asarray(y, dtype=float)
        z = yy * yy - 1.0
        t = yy * yy - self.ys * self.ys
        return 4.0 * self.beta * yy * (z * t * t + t * z * z)

    def floor(self, y):
        yy = np.maximum(np.asarray(y, dtype=float), 1e-7)
        n = self.nbar(yy)
        k = self._k_from_n(n, yy)
        mass = self.MN * yy
        ef = np.sqrt(k * k + mass * mass)
        p = self.pressure(ef, mass, k)
        return p + self.Cv * n * n / (2.0 * yy * yy)

    def floor_y(self, y):
        yy = np.maximum(np.asarray(y, dtype=float), 1e-7)
        n = self.nbar(yy)
        k = self._k_from_n(n, yy)
        mass = self.MN * yy
        ef = np.sqrt(k * k + mass * mass)
        ns = np.where(
            n > 0,
            self.d * mass / (4.0 * math.pi**2) * (
                k * ef - mass * mass * np.arcsinh(np.divide(k, mass, out=np.zeros_like(k), where=mass > 0))
            ),
            0.0,
        )
        V = self.mu - ef
        # This is (n V - m n_s)/y = n V/y - M_N n_s; the latter form is
        # important for exact cancellation in the scalar EL equation.
        return np.where(n > 0, n * V / yy - self.MN * ns, 0.0)

    def potential(self, y):
        return self.floor(y) + self.barrier(y)

    def potential_y(self, y):
        return self.floor_y(y) + self.barrier_y(y)

    def ode(self, x, Y):
        y, yp, a, ap = Y
        nu, mass, k, n, ns, _ = self.matter(y, a)
        del nu, mass, k
        out = np.empty_like(Y)
        out[0] = yp
        out[1] = self.potential_y(y) / self.W0**4 + self.gs * ns / self.W0**3 - self.q**2 * y * a * a
        out[2] = ap
        out[3] = self.q**2 * y * y * a - self.g * n / self.W0**3
        return out

    def boundary(self, left, right):
        return np.array([left[0] - self.ys, left[2] - float(self.al(self.ys)), right[0] - 1.0, right[2]])

    def initial(self, x):
        y = self.ys + (1.0 - self.ys) * (np.tanh(x / 45.0) + 1.0) / 2.0
        a = self.al(y)
        return np.vstack([y, np.gradient(y, x), a, np.gradient(a, x)])

    def local_integrand(self, y, a, yp, ap):
        nu, mass, k, _, _, _ = self.matter(y, a)
        p = self.pressure(nu, mass, k)
        v = self.potential(y) / self.W0**4 - 0.5 * self.q**2 * y * y * a * a - p / self.W0**4
        return 0.5 * yp * yp - 0.5 * ap * ap + v


def _finite_float(value) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _interface_attempt_passes(row: dict) -> bool:
    """Validate one finite-domain solve without inferring an infinite wall."""
    required = (
        "solver_status", "collocation_max_rms", "max_EL_residual_dimensionless",
        "max_W_EL_residual_dimensionless", "max_Gauss_residual_dimensionless",
        "independent_residual_limit",
        "stress_spread", "boundary_derivative_max_dimensionless", "min_y",
        "max_y", "vector_field_negative_undershoot", "tension_MeV3", "tolerance",
        "band_excursion_below", "band_excursion_above",
        "stress_min", "stress_max", "boundary_derivatives",
        "finite_domain_accuracy_passed", "finite_domain_approximation",
        "independent_residual_passed", "stress_criterion_passed",
        "boundary_derivative_passed", "band_excursion_passed",
        "vector_undershoot_within_finite_domain_limit",
    )
    if not isinstance(row, dict) or any(key not in row for key in required):
        return False
    numeric = (
        "collocation_max_rms", "max_EL_residual_dimensionless", "stress_spread",
        "max_W_EL_residual_dimensionless", "max_Gauss_residual_dimensionless",
        "independent_residual_limit",
        "boundary_derivative_max_dimensionless", "min_y", "max_y",
        "vector_field_negative_undershoot", "tension_MeV3", "tolerance",
        "band_excursion_below", "band_excursion_above",
        "stress_min", "stress_max",
    )
    if not all(_finite_float(row[key]) for key in numeric):
        return False
    derivatives = row.get("boundary_derivatives")
    if (not isinstance(derivatives, list) or len(derivatives) != 4
            or not all(_finite_float(value) for value in derivatives)):
        return False
    status = row.get("solver_status")
    tolerance = float(row["tolerance"])
    max_w_el = float(row["max_W_EL_residual_dimensionless"])
    max_gauss = float(row["max_Gauss_residual_dimensionless"])
    max_el = float(row["max_EL_residual_dimensionless"])
    residual_limit = float(row["independent_residual_limit"])
    stress_min = float(row["stress_min"])
    stress_max = float(row["stress_max"])
    stress_spread = float(row["stress_spread"])
    endpoint = float(row["boundary_derivative_max_dimensionless"])
    derivative_max = max(abs(float(value)) for value in derivatives)
    return bool(
        isinstance(status, int) and not isinstance(status, bool) and status == 0
        and tolerance > 0
        and float(row["collocation_max_rms"]) <= max(
            BVP_COLLOCATION_FACTOR * tolerance, 1e-4
        )
        and residual_limit == BVP_EL_RESIDUAL_LIMIT
        and max_w_el <= BVP_EL_RESIDUAL_LIMIT
        and max_gauss <= BVP_EL_RESIDUAL_LIMIT
        and max_el <= BVP_EL_RESIDUAL_LIMIT
        and max_el + 1e-10 >= max(max_w_el, max_gauss)
        and float(row["stress_spread"]) <= BVP_STRESS_SPREAD_LIMIT
        and stress_max >= stress_min
        and abs(stress_spread - (stress_max - stress_min)) <= 1e-8
        and endpoint <= BVP_ENDPOINT_DERIVATIVE_LIMIT
        and derivative_max <= BVP_ENDPOINT_DERIVATIVE_LIMIT
        and endpoint + 1e-10 >= derivative_max
        and float(row["band_excursion_below"]) <= BVP_BAND_EXCURSION_LIMIT
        and float(row["band_excursion_above"]) <= BVP_BAND_EXCURSION_LIMIT
        and float(row["vector_field_negative_undershoot"]) <= BVP_VECTOR_UNDERSHOOT_LIMIT
        and float(row["min_y"]) > 0
        and float(row["max_y"]) > float(row["min_y"])
        and float(row["tension_MeV3"]) > 0
        and row.get("finite_domain_accuracy_passed") is True
        and row.get("finite_domain_approximation") is True
        and all(row.get(key) is True for key in (
            "independent_residual_passed", "stress_criterion_passed",
            "boundary_derivative_passed", "band_excursion_passed",
            "vector_undershoot_within_finite_domain_limit",
        ))
    )


def _run_interface_bvp(model: _DoubleSpatialModel, domain: float, nodes: int, tol: float) -> dict:
    if not _SCIPY_AVAILABLE:
        return {"available": False, "converged": False, "failure": "SciPy unavailable"}
    x = np.linspace(-float(domain), float(domain), int(nodes))
    guess = model.initial(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            solution = solve_bvp(
                model.ode, model.boundary, x, guess,
                tol=float(tol), max_nodes=max(4000, int(nodes) * 20), verbose=0,
            )
        except (ArithmeticError, FloatingPointError, ValueError, RuntimeError) as exc:
            return {
                "available": True, "converged": False,
                "domain_x": float_number(domain), "initial_nodes": nodes,
                "tolerance": float_number(tol), "failure": str(exc),
            }
    grid = np.linspace(-float(domain), float(domain), max(1201, int(nodes) * 8 + 1))
    vals = solution.sol(grid)
    first = solution.sol(grid, 1)
    second = solution.sol(grid, 2)
    y, a, yp, ap = vals[0], vals[2], first[0], first[2]
    collocation = float(np.max(solution.rms_residuals)) if solution.rms_residuals.size else math.inf
    if not (
        np.all(np.isfinite(vals))
        and np.all(np.isfinite(first))
        and np.all(np.isfinite(second))
        and _finite_float(collocation)
    ):
        return {
            "available": True,
            "converged": False,
            "solver_status": int(solution.status),
            "solver_message": str(solution.message),
            "domain_x": float_number(domain),
            "initial_nodes": nodes,
            "tolerance": float_number(tol),
            "failure": "nonfinite finite-domain BVP output",
        }
    rhs = model.ode(grid, vals)
    w_residual = second[0] - rhs[1]
    # The equation is -a_xx + q^2 y^2 a = g n/W0^3.
    gauss_residual = -second[2] + q_square(model.q) * y * y * a
    gauss_residual -= model.g * model.matter(y, a)[3] / model.W0**3
    integrand = model.local_integrand(y, a, yp, ap)
    tension = float(simpson(integrand, x=grid) * model.W0**3)
    local_v = model.potential(y) / model.W0**4
    _, _, _, n, _, _ = model.matter(y, a)
    stress = 0.5 * yp * yp - 0.5 * ap * ap - (
        local_v - 0.5 * model.q**2 * y * y * a * a
        - model.pressure(model.matter(y, a)[0], model.matter(y, a)[1], model.matter(y, a)[2]) / model.W0**4
    )
    boundary_derivative = max(abs(yp[0]), abs(yp[-1]), abs(ap[0]), abs(ap[-1]))
    field_positive = bool(np.min(y) > 0.0)
    max_w_el = float(np.max(np.abs(w_residual)))
    max_gauss = float(np.max(np.abs(gauss_residual)))
    max_el = max(max_w_el, max_gauss)
    stress_spread = float(np.max(stress) - np.min(stress))
    endpoint_ok = bool(boundary_derivative <= BVP_ENDPOINT_DERIVATIVE_LIMIT)
    band_below = max(0.0, float(model.ys - np.min(y)))
    band_above = max(0.0, float(np.max(y) - 1.0))
    band_ok = bool(
        band_below <= BVP_BAND_EXCURSION_LIMIT
        and band_above <= BVP_BAND_EXCURSION_LIMIT
    )
    vector_undershoot = max(0.0, float(-np.min(a)))
    vector_undershoot_ok = bool(vector_undershoot <= BVP_VECTOR_UNDERSHOOT_LIMIT)
    residual_ok = bool(max_el <= BVP_EL_RESIDUAL_LIMIT)
    stress_ok = bool(stress_spread <= BVP_STRESS_SPREAD_LIMIT)
    collocation_ok = bool(collocation <= max(BVP_COLLOCATION_FACTOR * tol, 1e-4))
    finite_domain_accuracy_passed = bool(
        solution.status == 0 and field_positive and math.isfinite(tension)
        and residual_ok and stress_ok and endpoint_ok and band_ok
        and vector_undershoot_ok and collocation_ok
    )
    return {
        "available": True,
        "converged": finite_domain_accuracy_passed,
        "finite_domain_approximation": True,
        "infinite_wall_certified": False,
        "finite_domain_accuracy_passed": finite_domain_accuracy_passed,
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "domain_x": float_number(domain),
        "domain_z_MeV_minus1": float_number(domain / model.W0),
        "initial_nodes": int(nodes),
        "final_nodes": int(solution.x.size),
        "tolerance": float_number(tol),
        "collocation_max_rms": float_number(collocation, 8),
        "collocation_limit": float_number(max(BVP_COLLOCATION_FACTOR * tol, 1e-4), 8),
        "max_EL_residual_dimensionless": float_number(max_el, 8),
        "max_W_EL_residual_dimensionless": float_number(max_w_el, 8),
        "max_Gauss_residual_dimensionless": float_number(max_gauss, 8),
        "independent_residual_limit": float_number(BVP_EL_RESIDUAL_LIMIT, 8),
        "independent_residual_passed": residual_ok,
        "stress_min": float_number(np.min(stress), 8),
        "stress_max": float_number(np.max(stress), 8),
        "stress_spread": float_number(stress_spread, 8),
        "stress_spread_limit": float_number(BVP_STRESS_SPREAD_LIMIT, 8),
        "stress_criterion_passed": stress_ok,
        "tension_MeV3": float_number(tension, 8),
        "tension_MeV_fm_minus2": float_number(tension / model.hbarc**2, 8),
        "boundary_derivative_max_dimensionless": float_number(boundary_derivative, 8),
        "boundary_derivative_limit": float_number(BVP_ENDPOINT_DERIVATIVE_LIMIT, 8),
        "boundary_derivative_passed": endpoint_ok,
        "boundary_derivatives": [float_number(yp[0], 8), float_number(yp[-1], 8),
                                 float_number(ap[0], 8), float_number(ap[-1], 8)],
        "band_excursion_below": float_number(band_below, 8),
        "band_excursion_above": float_number(band_above, 8),
        "band_excursion_limit": float_number(BVP_BAND_EXCURSION_LIMIT, 8),
        "band_excursion_passed": band_ok,
        "min_y": float_number(np.min(y), 8),
        "max_y": float_number(np.max(y), 8),
        "min_a": float_number(np.min(a), 8),
        "max_a": float_number(np.max(a), 8),
        "vector_field_sample_nonnegative": bool(np.min(a) >= 0.0),
        "vector_field_negative_undershoot": float_number(vector_undershoot, 8),
        "vector_undershoot_limit": float_number(BVP_VECTOR_UNDERSHOOT_LIMIT, 8),
        "vector_undershoot_within_finite_domain_limit": vector_undershoot_ok,
        "vector_positivity_certified": False,
        "vector_positivity_scope": "sampled finite-grid values only; no maximum-principle enclosure",
        "density_integral_dimensionless": float_number(simpson(n / model.W0**3, x=grid), 8),
        "boundary_truncation_is_reported": True,
        "not_a_scalar_profile_claim": True,
    }


def q_square(q: float) -> float:
    return q * q


def interface_refinement(design: BindingDesign) -> dict:
    """Run a bounded coupled BVP refinement, preserving failures honestly."""
    if not _SCIPY_AVAILABLE:
        return {
            "available": False,
            "attempts": [],
            "converged_coupled_solution": False,
            "failure": "SciPy unavailable",
        }
    model = _DoubleSpatialModel(design)
    attempts = []
    for domain, nodes, tol in ((180.0, 121, 4e-3), (240.0, 161, 2e-3), (300.0, 201, 1e-3)):
        attempts.append(_run_interface_bvp(model, domain, nodes, tol))
    for previous, current in zip(attempts, attempts[1:]):
        if _finite_float(previous.get("tension_MeV3")) and _finite_float(current.get("tension_MeV3")):
            previous_tension = float(previous["tension_MeV3"])
            current_tension = float(current["tension_MeV3"])
            relative = abs(current_tension - previous_tension) / max(abs(current_tension), 1e-30)
            current["successive_tension_relative_difference"] = float_number(relative, 8)
            current["successive_tension_difference_passed"] = bool(
                relative <= BVP_TENSION_REFINEMENT_LIMIT
            )
        else:
            current["successive_tension_relative_difference"] = None
            current["successive_tension_difference_passed"] = False
    final = attempts[-1]
    last_tension_refinement = bool(
        len(attempts) >= 2
        and attempts[-1].get("successive_tension_difference_passed") is True
    )
    finite_domain_refinement_valid = bool(
        _interface_attempt_passes(final) and last_tension_refinement
    )
    with design._precision():
        bare_sigma = design.base.W0 * mp.sqrt(2 * design.beta) * mp.mpf(2) / 15 * (
            1 - design.y
        ) ** 3 * (design.y**2 + 3 * design.y + 1)
        eta = mp.mpf(derivative_bound(design)["eta"])
        lower_sigma = mp.sqrt(1 - eta * eta) * bare_sigma
    final_tension = (float(final["tension_MeV3"])
                     if _finite_float(final.get("tension_MeV3")) else math.nan)
    tension_in_band = bool(
        math.isfinite(final_tension)
        and float(lower_sigma) <= final_tension <= float(bare_sigma)
    )
    return {
        "available": True,
        "model": {
            "coordinate": "x=W0*z",
            "fields": "y=W/W0, a=A/W0",
            "domain_refinement": [180, 240, 300],
            "tolerance_refinement": ["4e-3", "2e-3", "1e-3"],
            "positive_W_transform": "direct y with physical-domain rejection",
        },
        "attempts": attempts,
        "converged_coupled_solution": finite_domain_refinement_valid,
        "finite_domain_refinement_valid": finite_domain_refinement_valid,
        "tension_refinement_passed": last_tension_refinement,
        "tension_refinement_limit": float_number(BVP_TENSION_REFINEMENT_LIMIT, 8),
        "final_tension_in_declared_band": tension_in_band,
        "tension_band_is_diagnostic_only": True,
        "infinite_wall_certified": False,
        "final": final,
        "method_scope": "finite-domain coupled EL/Gauss BVP; no quantum surface sector",
        "remaining_boundary_issue": (
            "finite endpoint derivatives are truncation diagnostics, not zero-asymptote proof"
        ),
    }


def _periodic_matrix(y: np.ndarray, model: _DoubleSpatialModel, length: float):
    n = y.size
    h = float(length) / n
    main = 2.0 / h**2 + model.q**2 * y * y
    off = np.full(n - 1, -1.0 / h**2)
    matrix = diags([off, main, off], [-1, 0, 1], shape=(n, n), format="lil")
    matrix[0, -1] = -1.0 / h**2
    matrix[-1, 0] = -1.0 / h**2
    return matrix.tocsr(), h


def classify_periodic_sample(sample: dict, *, max_newton: int = 30) -> dict:
    """Separate numerical validity from the sign of a periodic energy witness."""
    required = (
        "gauss_residual_max_dimensionless", "gauss_energy_identity_residual",
        "positive_gauss_operator_min_eigenvalue", "positive_newton_jacobian_lower_bound",
        "omega_E_minus_muN_MeV3", "newton_iterations", "grid_size", "min_y", "max_y",
    )
    if not isinstance(sample, dict) or any(key not in sample for key in required):
        return {
            "calculation_valid": False,
            "negative_witness": False,
            "interpretation": "numerical_failure_or_missing_evidence",
        }
    numeric_keys = required[:5] + required[7:]
    finite_ok = all(_finite_float(sample[key]) for key in numeric_keys)
    if any(isinstance(sample[key], bool) for key in ("newton_iterations", "grid_size")):
        return {
            "calculation_valid": False,
            "negative_witness": False,
            "interpretation": "numerical_failure_or_invalid_numeric_type",
        }
    try:
        iterations = int(sample["newton_iterations"])
        grid_size = int(sample["grid_size"])
    except (TypeError, ValueError, OverflowError):
        iterations, grid_size = -1, -1
    gauss_residual = (abs(float(sample["gauss_residual_max_dimensionless"]))
                      if _finite_float(sample["gauss_residual_max_dimensionless"]) else math.inf)
    energy_residual = (abs(float(sample["gauss_energy_identity_residual"]))
                       if _finite_float(sample["gauss_energy_identity_residual"]) else math.inf)
    operator_min = (float(sample["positive_gauss_operator_min_eigenvalue"])
                    if _finite_float(sample["positive_gauss_operator_min_eigenvalue"]) else -math.inf)
    jacobian_min = (float(sample["positive_newton_jacobian_lower_bound"])
                    if _finite_float(sample["positive_newton_jacobian_lower_bound"]) else -math.inf)
    min_y = float(sample["min_y"]) if _finite_float(sample["min_y"]) else math.nan
    max_y = float(sample["max_y"]) if _finite_float(sample["max_y"]) else math.nan
    valid = bool(
        finite_ok
        and iterations >= 1 and iterations <= int(max_newton)
        and grid_size >= 32
        and min_y > 0.0 and max_y > min_y
        and operator_min > 0.0 and jacobian_min > 0.0
        and gauss_residual <= PERIODIC_GAUSS_RESIDUAL_LIMIT
        and energy_residual <= PERIODIC_ENERGY_IDENTITY_LIMIT
    )
    omega = float(sample["omega_E_minus_muN_MeV3"]) if _finite_float(
        sample["omega_E_minus_muN_MeV3"]
    ) else math.nan
    negative = bool(valid and omega < -1e-12)
    if negative:
        interpretation = "valid_negative_periodic_witness"
    elif valid:
        interpretation = "valid_sample_no_negative_witness"
    else:
        interpretation = "numerical_failure_or_insufficient_refinement"
    return {
        "calculation_valid": valid,
        "negative_witness": negative,
        "interpretation": interpretation,
        "gauss_residual_limit": float_number(PERIODIC_GAUSS_RESIDUAL_LIMIT, 8),
        "energy_identity_limit": float_number(PERIODIC_ENERGY_IDENTITY_LIMIT, 8),
        "energy_sign_is_not_a_validity_gate": True,
    }


def periodic_trial(
    design: BindingDesign,
    mean: float,
    amplitude: float,
    *,
    length: float = 100.0,
    grid_size: int = 128,
    mode: int = 1,
    max_newton: int = 30,
) -> dict:
    """Solve the nonlinear periodic Gauss equation for a fixed smooth W trial."""
    if not _SCIPY_AVAILABLE:
        raise RuntimeError("SciPy unavailable for periodic control")
    model = _DoubleSpatialModel(design)
    x = np.linspace(0.0, float(length), int(grid_size), endpoint=False)
    h = float(length) / int(grid_size)
    y = float(mean) + float(amplitude) * np.cos(2.0 * math.pi * int(mode) * x / float(length))
    if np.min(y) <= 0:
        raise ValueError("periodic trial has W<=0")
    operator, _ = _periodic_matrix(y, model, float(length))
    a = model.al(y)
    iterations = 0
    residual_norm = math.inf
    for iterations in range(1, int(max_newton) + 1):
        nu, mass, k, n, _, chi = model.matter(y, a)
        residual = operator @ a - model.g * n / model.W0**3
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm < 2e-13:
            break
        jacobian = operator + diags(model.g**2 * chi / model.W0**2, 0, format="csr")
        delta = spsolve(jacobian, -residual)
        old_norm = float(np.linalg.norm(residual))
        step = 1.0
        for _ in range(20):
            candidate = a + step * delta
            n_candidate = model.matter(y, candidate)[3]
            trial_residual = operator @ candidate - model.g * n_candidate / model.W0**3
            if float(np.linalg.norm(trial_residual)) < old_norm:
                a = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("periodic Gauss Newton line search failed")
    nu, mass, k, n, _, chi = model.matter(y, a)
    residual = operator @ a - model.g * n / model.W0**3
    residual_norm = float(np.max(np.abs(residual)))
    yx = (np.roll(y, -1) - np.roll(y, 1)) / (2.0 * h)
    ax = (np.roll(a, -1) - np.roll(a, 1)) / (2.0 * h)
    integrand = model.local_integrand(y, a, yx, ax)
    omega = float(np.mean(integrand) * float(length) * model.W0**3)
    n_per_area = float(np.mean(n) * float(length) / model.W0)
    # Independent Gauss energy identity on the same discrete positive matrix.
    source = model.g * n / model.W0**3
    lhs = float(np.dot(a, operator @ a) * h)
    rhs = float(np.dot(a, source) * h)
    try:
        eig_min = float(eigsh(operator, k=1, which="SA", return_eigenvectors=False)[0])
    except Exception:
        eig_min = float(model.q**2 * np.min(y * y))
    # The susceptibility term makes the Newton Jacobian even more positive.
    jac_min_lower = eig_min + float(np.min(model.g**2 * chi / model.W0**2))
    result = {
        "mean_y": float_number(mean),
        "amplitude_y": float_number(amplitude),
        "length_x": float_number(length),
        "mode": int(mode),
        "grid_size": int(grid_size),
        "min_y": float_number(np.min(y)),
        "max_y": float_number(np.max(y)),
        "newton_iterations": int(iterations),
        "gauss_residual_max_dimensionless": float_number(residual_norm),
        "omega_E_minus_muN_MeV3": float_number(omega),
        "N_per_area_MeV2": float_number(n_per_area),
        "gauss_energy_lhs": float_number(lhs),
        "gauss_energy_rhs": float_number(rhs),
        "gauss_energy_identity_residual": float_number(lhs - rhs),
        "fixed_density_vector_energy_MeV3": float_number(0.5 * rhs * model.W0**3),
        "fixed_density_vector_energy_formula": ".5 <g*n,(-d_x^2+q^2*y^2)^(-1) g*n>",
        "positive_gauss_operator_min_eigenvalue": float_number(eig_min),
        "positive_newton_jacobian_lower_bound": float_number(jac_min_lower),
        "nonuniform_trial": bool(abs(amplitude) > 0),
        "uses_actual_positive_gauss_operator": True,
        "coefficients_unchanged": True,
    }
    result.update(classify_periodic_sample(result, max_newton=max_newton))
    return result


def periodic_controls(design: BindingDesign) -> dict:
    """Bounded smooth trials, including W below W_star, with N-grid refinement."""
    if not _SCIPY_AVAILABLE:
        return {
            "available": False,
            "trials": [],
            "negative_E_minus_muN_witness": False,
            "calculation_valid": False,
            "valid_negative_witness_present": False,
            "numerical_validity_failed": True,
            "interpretation": "numerical_failure_or_missing_scipy",
            "unrestricted_extension_status": "undecided_numerical_validity_failed",
            "positive_samples_do_not_prove_global_minimum": True,
            "no_unbounded_optimizer_or_parameter_fit": True,
        }
    specs = (
        ("below_band", 0.50, 0.10),
        ("low_wide", 0.50, 0.24),
        ("cross_band", 0.65, 0.20),
        ("near_vacuum", 0.90, 0.08),
    )
    rows = []
    for label, mean, amplitude in specs:
        coarse = periodic_trial(design, mean, amplitude, grid_size=128)
        fine = periodic_trial(design, mean, amplitude, grid_size=256)
        omega0 = float(coarse["omega_E_minus_muN_MeV3"])
        omega1 = float(fine["omega_E_minus_muN_MeV3"])
        refinement = (omega1 - omega0) / max(abs(omega1), 1e-30)
        rows.append({
            "label": label,
            "coarse": coarse,
            "fine": fine,
            "omega_relative_refinement": float_number(refinement, 8),
            "calculation_valid_both_grids": bool(
                coarse.get("calculation_valid") is True
                and fine.get("calculation_valid") is True
            ),
            "energy_refinement_passed": bool(abs(refinement) <= PERIODIC_REFINEMENT_LIMIT),
            "positive_both_grids": bool(omega0 > 0 and omega1 > 0),
        })
    witness = any(
        row[grid].get("negative_witness") is True
        for row in rows for grid in ("coarse", "fine")
    )
    calculation_valid = bool(
        all(row["calculation_valid_both_grids"] for row in rows)
        and all(row["energy_refinement_passed"] for row in rows)
    )
    if witness and calculation_valid:
        interpretation = "valid_negative_periodic_witness"
        unrestricted_status = "refuted_by_valid_negative_periodic_witness"
    elif witness:
        interpretation = "valid_negative_witness_but_other_trials_failed"
        unrestricted_status = "refuted_by_valid_negative_periodic_witness"
    elif not calculation_valid:
        interpretation = "numerical_failure_or_insufficient_refinement"
        unrestricted_status = "undecided_numerical_validity_failed"
    else:
        interpretation = "no_negative_witness_in_finite_trials"
        unrestricted_status = "undecided_positive_samples_only"
    return {
        "available": True,
        "trial_domain": "smooth periodic y=mean+amplitude*cos(2*pi*x/L), fixed design",
        "trials": rows,
        "negative_E_minus_muN_witness": bool(witness),
        "calculation_valid": calculation_valid,
        "valid_negative_witness_present": bool(witness),
        "numerical_validity_failed": bool(not calculation_valid),
        "refinement_limit": float_number(PERIODIC_REFINEMENT_LIMIT, 8),
        "interpretation": interpretation,
        "unrestricted_extension_status": unrestricted_status,
        "positive_samples_do_not_prove_global_minimum": True,
        "includes_W_below_Wstar": True,
        "no_unbounded_optimizer_or_parameter_fit": True,
    }


def _grand_density_mp(design: BindingDesign, W, A) -> mp.mpf:
    """Evaluate the actual fixed-design grand density at one active point."""
    with design._precision():
        W = finite(W, "W", positive=True)
        A = finite(A, "A")
        y = W / design.base.W0
        nu = design.mu - design.gomega * A
        mass = design.base.MN * y
        if nu <= mass:
            pressure = mp.mpf(0)
        else:
            k = mp.sqrt(nu * nu - mass * mass)
            pressure = _fermi_pressure_mp(k, mass, design.base.d)
        q = design.base.momega / design.base.W0
        return (design.potential(y) - mp.mpf(".5") * q * q * W * W * A * A
                - pressure)


def mixed_hessian_controls(design: BindingDesign) -> dict:
    """Compare the physical mixed Hessian with the actual grand density.

    The point is deliberately on the active local envelope, where the
    piecewise pressure is smooth.  The two mixed derivatives are taken in
    opposite orders directly from ``_grand_density_mp``.  Their reference is
    the independently derived chain expression
    ``-2*q^2*W*A + g*g_s*P_(nu,m)``; the vector-only and wrong-sign fermion
    variants are retained as negative controls.
    """
    with design._precision():
        y = finite(design.y, "mixed-control y", positive=True)
        state = local_envelope_state(design, y)
        W = state["W"]
        A = state["A_L"]
        g = design.gomega
        gs = design.base.MN / design.base.W0
        q = design.base.momega / design.base.W0
        nu = design.mu - g * A
        mass = gs * W
        active = bool(nu > mass)
        if not active:
            return {
                "available": True,
                "active_branch": False,
                "control_passed": False,
                "failure": "mixed control point is not on the active pressure branch",
            }
        k = mp.sqrt(nu * nu - mass * mass)
        vector_term = -2 * q * q * W * A
        fermion_term = -g * gs * design.base.d * mass * k / (2 * mp.pi**2)
        closed = vector_term + fermion_term
        omitted = vector_term
        wrong_sign = vector_term - fermion_term

        actual_w_a = mp.diff(
            lambda WW: mp.diff(
                lambda AA: _grand_density_mp(design, WW, AA), A
            ),
            W,
        )
        actual_a_w = mp.diff(
            lambda AA: mp.diff(
                lambda WW: _grand_density_mp(design, WW, AA), W
            ),
            A,
        )
        scale = lambda value: max(abs(value), mp.mpf(1))
        relative = lambda lhs, rhs: abs(lhs - rhs) / max(
            scale(lhs), scale(rhs)
        )
        residual_w_a = relative(actual_w_a, closed)
        residual_a_w = relative(actual_a_w, closed)
        residual_symmetry = relative(actual_w_a, actual_a_w)
        residual_decomposition = relative(closed, vector_term + fermion_term)
        omitted_residual = relative(actual_w_a, omitted)
        wrong_sign_residual = relative(actual_w_a, wrong_sign)
        return {
            "available": True,
            "active_branch": active,
            "sample_y": number(y),
            "sample_W_MeV": number(W),
            "sample_A_MeV": number(A),
            "nu_minus_mass_MeV": number(nu - mass),
            "actual_hessian_WA": number(actual_w_a),
            "actual_hessian_AW": number(actual_a_w),
            "closed_expression": number(closed),
            "vector_cross_term": number(vector_term),
            "fermion_chain_term": number(fermion_term),
            "omitted_fermion_expression": number(omitted),
            "wrong_sign_fermion_expression": number(wrong_sign),
            "relative_residual_WA": number(residual_w_a),
            "relative_residual_AW": number(residual_a_w),
            "relative_residual_symmetry": number(residual_symmetry),
            "relative_residual_closed_decomposition": number(residual_decomposition),
            "omitted_term_residual": number(omitted_residual),
            "wrong_sign_term_residual": number(wrong_sign_residual),
            "actual_derivatives_finite": bool(
                mp.isfinite(actual_w_a) and mp.isfinite(actual_a_w)
            ),
            "closed_expression_matches_actual": bool(
                residual_w_a < mp.mpf("1e-25")
                and residual_a_w < mp.mpf("1e-25")
            ),
            "omitted_term_rejected": bool(omitted_residual > mp.mpf("1e-6")),
            "wrong_sign_term_rejected": bool(wrong_sign_residual > mp.mpf("1e-6")),
            "control_passed": bool(
                active
                and residual_w_a < mp.mpf("1e-25")
                and residual_a_w < mp.mpf("1e-25")
                and residual_symmetry < mp.mpf("1e-25")
                and residual_decomposition < mp.mpf("1e-25")
                and omitted_residual > mp.mpf("1e-6")
                and wrong_sign_residual > mp.mpf("1e-6")
            ),
            "derivative_scope": "one active fixed-design point; no global Hessian theorem",
        }


def _mixed_hessian_evidence_passed(result: dict) -> bool:
    """Validate actual mixed-derivative evidence and its negative controls."""
    if not isinstance(result, dict) or result.get("available") is not True:
        return False
    required_flags = (
        "available", "active_branch", "actual_derivatives_finite",
        "closed_expression_matches_actual", "omitted_term_rejected",
        "wrong_sign_term_rejected", "control_passed",
    )
    required_numeric = (
        "sample_y", "sample_W_MeV", "sample_A_MeV", "nu_minus_mass_MeV",
        "actual_hessian_WA", "actual_hessian_AW", "closed_expression",
        "vector_cross_term", "fermion_chain_term",
        "omitted_fermion_expression", "wrong_sign_fermion_expression",
        "relative_residual_WA", "relative_residual_AW",
        "relative_residual_symmetry", "relative_residual_closed_decomposition",
        "omitted_term_residual", "wrong_sign_term_residual",
    )
    if (any(key not in result for key in required_flags + required_numeric)
            or result.get("derivative_scope") != (
                "one active fixed-design point; no global Hessian theorem"
            )
            or any(not isinstance(result[key], bool) for key in required_flags)
            or not all(result[key] is True for key in required_flags)):
        return False
    # ``number()`` serializes ordinary evidence to 35 significant digits.
    # Parse at much higher precision, and allow only the resulting ~1e-35
    # operand-rounding discrepancy when checking a reported tiny residual.
    # The actual derivative/decomposition gates remain the existing 1e-25;
    # this floor is not a relaxation of those gates.
    with mp.workdps(160):
        def parse_number(value):
            if isinstance(value, bool):
                return None
            try:
                parsed = mp.mpf(value)
            except (TypeError, ValueError, OverflowError):
                return None
            return parsed if mp.isfinite(parsed) else None

        values = {}
        for key in required_numeric:
            parsed = parse_number(result[key])
            if parsed is None:
                return False
            values[key] = parsed

        residual_keys = (
            "relative_residual_WA", "relative_residual_AW",
            "relative_residual_symmetry", "relative_residual_closed_decomposition",
            "omitted_term_residual", "wrong_sign_term_residual",
        )
        if any(values[key] < 0 for key in residual_keys):
            return False
        if not (values["sample_y"] > 0
                and values["sample_W_MeV"] > 0
                and values["sample_A_MeV"] > 0
                and values["nu_minus_mass_MeV"] > 0):
            return False

        actual_wa = values["actual_hessian_WA"]
        actual_aw = values["actual_hessian_AW"]
        closed = values["closed_expression"]
        vector = values["vector_cross_term"]
        fermion = values["fermion_chain_term"]
        omitted = values["omitted_fermion_expression"]
        wrong_sign = values["wrong_sign_fermion_expression"]

        def scale(lhs, rhs):
            return max(abs(lhs), abs(rhs), mp.mpf(1))

        def relative(lhs, rhs):
            return abs(lhs - rhs) / scale(lhs, rhs)

        recomputed = {
            "relative_residual_WA": relative(actual_wa, closed),
            "relative_residual_AW": relative(actual_aw, closed),
            "relative_residual_symmetry": relative(actual_wa, actual_aw),
            "relative_residual_closed_decomposition": relative(
                closed, vector + fermion
            ),
            "omitted_term_residual": relative(actual_wa, omitted),
            "wrong_sign_term_residual": relative(actual_wa, wrong_sign),
        }
        serialization_floor = mp.mpf("1e-32")
        matching_threshold = mp.mpf("1e-25")
        # Reported residuals are cross-checked against recomputation.  For a
        # nominal tiny residual, serialized operands can make recomputation
        # larger by ~1e-35, so consistency is absolute at this floor rather
        # than an impossible relative comparison to a value near zero.
        if any(
            abs(values[key] - recomputed[key]) > serialization_floor
            for key in residual_keys
        ):
            return False
        if any(
            recomputed[key] >= matching_threshold
            for key in (
                "relative_residual_WA", "relative_residual_AW",
                "relative_residual_symmetry",
                "relative_residual_closed_decomposition",
            )
        ):
            return False
        if not (
            relative(omitted, vector) <= serialization_floor
            and relative(wrong_sign, vector - fermion) <= serialization_floor
            and recomputed["omitted_term_residual"] > mp.mpf("1e-6")
            and recomputed["wrong_sign_term_residual"] > mp.mpf("1e-6")
        ):
            return False
        return bool(
            vector < 0
            and fermion < 0
            and closed < 0
            and values["fermion_chain_term"] != 0
            and result["closed_expression_matches_actual"] is True
            and result["omitted_term_rejected"] is True
            and result["wrong_sign_term_rejected"] is True
        )


def _local_saddle_hessian(design: BindingDesign, y) -> dict:
    """Differentiate the actual grand density at the envelope maximizer."""
    with design._precision():
        y = finite(y, "y", positive=True)
        W0 = design.base.W0
        W = W0 * y
        q = design.base.momega / W0
        state = local_envelope_state(design, y)
        A = state["A_L"]
        h_ww = mp.diff(lambda WW: _grand_density_mp(design, WW, A), W, 2)
        h_aa = mp.diff(lambda AA: _grand_density_mp(design, W, AA), A, 2)
        h_wa = mp.diff(
            lambda WW: mp.diff(
                lambda AA: _grand_density_mp(design, WW, AA), A
            ),
            W,
        )
        schur = h_ww - h_wa * h_wa / h_aa if h_aa < 0 else mp.nan
        return {
            "y": number(y),
            "A_L": number(A),
            "grand_hessian_WW": number(h_ww),
            "grand_hessian_WA": number(h_wa),
            "grand_hessian_AA": number(h_aa),
            "reduced_scalar_schur": number(schur),
            "vector_direction_is_negative": bool(h_aa < 0),
            "scalar_minimum_direction_at_endpoint": bool(schur > 0),
        }


def saddle_controls(design: Optional[BindingDesign] = None) -> dict:
    """Check the vector saddle sign and actual fixed-design Hessian criteria.

    The one-mode sign calculation is retained as a cheap mutation control.  If
    ``design`` is supplied, the local Hessian is differentiated numerically
    from the actual grand density at the liquid and vacuum endpoints.  The
    scalar direction is not asserted positive through the barrier interior;
    only endpoint minima are checked.
    """
    # A periodic perturbation with one wavelength has a strictly negative
    # correct quadratic form.  Flipping the whole vector quadratic sector
    # makes the same mode strictly positive, so treating A as a minimum is a
    # mathematically distinct (and wrong) problem.
    length = 20.0
    m2 = 0.7**2
    mode = 3
    k = 2.0 * math.pi * mode / length
    correct = -0.5 * length * (k * k + m2)
    opposite = -correct
    result = {
        "correct_vector_second_variation": float_number(correct),
        "opposite_sign_second_variation": float_number(opposite),
        "correct_is_maximum": bool(correct < 0),
        "opposite_is_not_maximum": bool(opposite > 0),
        "saddle_failure_is_control_only": True,
    }
    if design is None:
        result.update({
            "actual_hessian_checks_available": False,
            "actual_vector_hessian_negative": False,
            "actual_scalar_endpoint_criteria_passed": False,
        })
        return result
    samples = [_local_saddle_hessian(design, design.y),
               _local_saddle_hessian(design, mp.mpf(1))]
    result.update({
        "actual_hessian_checks_available": True,
        "actual_hessian_samples": samples,
        "actual_vector_hessian_negative": bool(all(
            row["vector_direction_is_negative"] for row in samples
        )),
        "actual_scalar_endpoint_criteria_passed": bool(all(
            row["scalar_minimum_direction_at_endpoint"] for row in samples
        )),
        "hessian_scope": "actual fixed-design grand density at liquid/vacuum endpoints; barrier interior may have negative scalar curvature",
    })
    return result


def _saddle_evidence_passed(result: dict) -> bool:
    """Require finite endpoint Hessians, not only caller-supplied sign flags."""
    if not isinstance(result, dict):
        return False
    if not all(result.get(key) is True for key in (
        "actual_hessian_checks_available", "actual_vector_hessian_negative",
        "actual_scalar_endpoint_criteria_passed", "correct_is_maximum",
        "opposite_is_not_maximum",
    )):
        return False
    samples = result.get("actual_hessian_samples")
    if not isinstance(samples, list) or len(samples) != 2:
        return False
    required = (
        "y", "grand_hessian_WW", "grand_hessian_WA", "grand_hessian_AA",
        "reduced_scalar_schur", "vector_direction_is_negative",
        "scalar_minimum_direction_at_endpoint",
    )
    for row in samples:
        if not isinstance(row, dict) or any(key not in row for key in required):
            return False
        if not all(_finite_float(row[key]) for key in required[:5]):
            return False
        if row["vector_direction_is_negative"] is not True:
            return False
        if row["scalar_minimum_direction_at_endpoint"] is not True:
            return False
        if not (float(row["grand_hessian_AA"]) < 0
                and float(row["reduced_scalar_schur"]) > 0):
            return False
    return True


def audit(*, dps: int = 80, include_numerics: bool = True) -> dict:
    precision(dps)
    with mp.workdps(dps):
        design = BindingDesign(dps=dps)
        symbols = symbolic_checks()
        symbolic_pass = _symbolic_rows_passed(symbols)
        certificates = analytic_certificates(design)
        envelope = derivative_bound(design)
        regularity = onset_regularity(design)
        bounds = tension_bounds(design)
        mixed_hessian = mixed_hessian_controls(design)
        saddle = saddle_controls(design)
        interface = interface_refinement(design) if include_numerics else {"available": False, "skipped": True}
        periodic = periodic_controls(design) if include_numerics else {"available": False, "skipped": True}
        # All numerical entries are independently generated; these are scope
        # checks, not claims that a finite sample is a global theorem.
        numerics_pass = bool(
            (not include_numerics)
            or (
                _interface_refinement_passed(interface)
                and _periodic_controls_passed(periodic)
            )
        )
        evidence_pass = bool(
            symbolic_pass
            and _analytic_certificates_passed(certificates)
            and _envelope_evidence_passed(envelope, regularity)
            and _tension_evidence_passed(bounds)
            and _mixed_hessian_evidence_passed(mixed_hessian)
            and _saddle_evidence_passed(saddle)
        )
        return {
            "status": STATUS,
            "evidence_weight": 0,
            "mathematical_checks_passed": bool(evidence_pass and numerics_pass),
            "design_inputs": {
                "mu0_MeV": number(design.mu),
                "target_y_star": number(design.y),
                "target_n_fm_minus3": number(design.n_fm3),
                "target_binding_MeV": number(design.binding),
                "target_K_MeV": number(design.K),
                "gomega_new": number(design.gomega),
                "q_momega_over_W0": number(design.base.momega / design.base.W0),
                "dps": int(dps),
                "all_are_fixed_calibration_inputs_not_predictions": True,
                "original_parameters_unchanged": dict(INPUTS),
            },
            "grand_functional": {
                "formula": "integral[.5 W'^2-.5 A'^2+U(W)-.5 q^2 W^2 A^2-P_F(mu0-g A,g_s W)] dz",
                "signature": "+---",
                "W_role": "minimum",
                "A_role": "maximum",
                "density_closure": "n=d[(nu^2-g_s^2 W^2)_+]^(3/2)/(6*pi^2), active only nu>g_s W",
                "scalar_EL": "W''=U_W+g_s*n_s-q^2*W*A^2",
                "vector_EL": "-A''+q^2*W^2*A=g*n",
                "translation_first_integral": ".5 W'^2-.5 A'^2-[U-.5 q^2 W^2 A^2-P_F]=constant",
                "fixed_density_vector_energy": ".5 <g n,(-d_z^2+q^2 W^2)^(-1) g n>",
                "homogeneous_local_vector_replacement_forbidden": True,
            },
            "symbolic_checks": symbols,
            "analytic_certificates": certificates,
            "vector_saddle_controls": saddle,
            "envelope": {
                "pointwise_maximizer": "A_L(W)=g*n_bar(W)/(q^2 W^2)",
                "derivative_formula": "dA_L/dW=-(g*b+2*a_F*M2*A_L/W)/(g^2+a_F*M2)",
                "weighted_mean_statement": "negative weighted mean of b/g and 2*A_L/W",
                "checks": envelope,
            },
            "onset_regularity": regularity,
            "tension_bounds": bounds,
            "mixed_hessian_controls": mixed_hessian,
            "interface_bvp": interface,
            "periodic_global_attack": periodic,
            "scope": [
                "Fixed inverse-designed U(W) and changed gomega are a counterfactual binding design, not empirical validation.",
                "Only static planar local T=0 normal Thomas-Fermi matter is included; no sea, pairing or Coulomb sector.",
                "The coupled BVP is finite-domain and reports EL/Gauss residuals, stress and endpoint truncation.",
                "The sandwich certifies a constrained W_star-to-W0 variational tension interval, not an unrestricted global minimum.",
                "Periodic positive samples are controls only; they do not prove the unrestricted inhomogeneous ground-state theorem.",
            ],
        }


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None) -> int:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--skip-numerics", action="store_true")
    try:
        args = parser.parse_args(argv)
        result = audit(dps=args.dps, include_numerics=not args.skip_numerics)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        print(json.dumps({
            "status": "invalid_or_failed_spatial_audit",
            "evidence_weight": 0,
            "mathematical_checks_passed": False,
            "error": str(exc),
        }, ensure_ascii=False, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["mathematical_checks_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
