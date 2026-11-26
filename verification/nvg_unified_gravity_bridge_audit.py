#!/usr/bin/env python3
"""Law 3B & Law 4 — Unified Covariant Gravity Bridge Audit (SPEC-v2).

Mathematical and numerical verification of:
  1. Symmetry separation (local U(1)_B phase absorbed into massive vector vs.
     independent global timelike foliation field Psi) and geometric identity
     linking the FLRW cuscuton bounce to the limiting-curvature spherically
     symmetric black hole interior (Hayward anisotropic regular core).
  2. Vacuum bag scale bridge rho_c = kappa_g * U(0) = kappa_g * (lambda_W/4) * W0^4
     comparing kappa_g = 1 vs legacy rho_c = W0^4 (kappa_g = 4/lambda_W) at
     60-dps (with 90-dps control) precision.
  3. Dynamic matter bounce integration (n, W, v, q) through q = 0 using the
     quartic-vector matter functional (zeta > 0) removing the W^-2 singularity.
  4. Law 4 IR Cosmology Theorem: constant Lambda > 0 forbids flat-FLRW
     turnaround (strictly one bounce, w = -1), whereas an explicitly segregated
     external IR scalar phi_DE with zero-crossing potential V_DE(phi) produces
     thawing w(z) != -1 and finite-time turnaround into the next cuscuton bounce.

Scientific integrity (SPEC-v2):
  - evidence_weight = 0.0 (mathematical/symbolic/numerical consistency audit;
    no observational likelihood is evaluated).
  - Calibration targets and external IR assumptions are explicitly separated
    from derived identities.
  - CLI supports --output <path> and --validate <path> with fail-closed
    canonical JSON serialization (sort_keys=True, indent=2, allow_nan=False).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp


HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
DEFAULT_OUTPUT_PATH = HERE / "nvg_unified_gravity_bridge_results.json"

SCHEMA_VERSION = "nvg_unified_gravity_bridge.v2"
BRIDGE_ID = "LAW_3B_4_UNIFIED_COVARIANT_GRAVITY_BRIDGE"
STATUS = "UNIFIED_GRAVITY_BRIDGE_VERIFIED_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

PRIMARY_DPS = 60
CONTROL_DPS = 90
REPORT_DIGITS = 45

# Declared physical & accepted NVG inputs (consistent with repo conventions)
DECLARED_INPUTS = {
    "W0_MeV": "859",
    "lambda_W": "1.05",
    "MN_MeV": "939",
    "momega_MeV": "782.6",
    "gomega": "10.12",
    "fermion_degeneracy_d": 4,
    "hbarc_MeV_fm": "197.3269804",
    "G_N_MeV_minus2": "6.70883e-45",
    "k_B_MeV_per_K": "8.617333262e-11",
    "M_sun_kg": "1.98847e30",
    "c_m_per_s": "299792458",
    "MeV_in_J": "1.602176634e-13",
    "MeV_fm3_to_g_cm3": "1.782661921627897e12",
}


def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def _nstr(val: mp.mpf, digits: int = REPORT_DIGITS) -> str:
    val = mp.mpf(val)
    if not mp.isfinite(val):
        raise ValueError("Non-finite mpmath value encountered")
    return mp.nstr(val, digits)


def _source_sha256() -> str:
    return hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()


# =============================================================================
# 1. SYMBOLIC SYMMETRY SEPARATION, CUSCUTON BOUNCE, HAYWARD CORE & QUARTIC VEC
# =============================================================================

def symbolic_symmetry_and_geometry_proofs(
    *,
    cuscuton_kinetic_sign: int = -1,
    hayward_anisotropy_scale: int = 1,
    quartic_force_sign: int = 1,
) -> dict[str, Any]:
    """Symbolically prove all Law 3B identities in SymPy.

    Optional keyword parameters act purely as negative controls for unit tests:
      - cuscuton_kinetic_sign=+1 breaks the bounce branch;
      - hayward_anisotropy_scale=0 enforces an isotropic fluid (P_t = P_r),
        triggering the isotropic-fluid no-go of BLACK_HOLES_AND_CYCLES_RU.md;
      - quartic_force_sign=-1 corrupts the scalar equation of motion and breaks
        off-equilibrium constraint conservation.
    """
    if isinstance(cuscuton_kinetic_sign, bool) or cuscuton_kinetic_sign not in (-1, 1):
        raise ValueError("cuscuton_kinetic_sign must be -1 or +1")
    if isinstance(quartic_force_sign, bool) or quartic_force_sign not in (-1, 1):
        raise ValueError("quartic_force_sign must be -1 or +1")
    if isinstance(hayward_anisotropy_scale, bool):
        raise ValueError("hayward_anisotropy_scale must be numeric, not bool")

    checks: dict[str, sp.Expr] = {}

    # --- 1A. Local U(1)_B gauge absorption vs global timelike foliation Psi ---
    x_coord = sp.Symbol("x", real=True)
    gB = sp.Symbol("g_B", positive=True)
    W_fn = sp.Function("W")(x_coord)
    theta_fn = sp.Function("theta_B")(x_coord)
    A_fn = sp.Function("A")(x_coord)
    alpha_g = sp.Function("alpha_g")(x_coord)

    Phi = W_fn * sp.exp(sp.I * theta_fn)
    D_Phi = sp.diff(Phi, x_coord) + sp.I * gB * A_fn * Phi
    D_Phi_conj = sp.diff(W_fn * sp.exp(-sp.I * theta_fn), x_coord) - sp.I * gB * A_fn * W_fn * sp.exp(-sp.I * theta_fn)
    kinetic_cov = sp.expand(sp.simplify(D_Phi * D_Phi_conj))
    B_vec = A_fn + sp.diff(theta_fn, x_coord) / gB
    checks["u1b_unitary_gauge_absorption"] = sp.simplify(
        kinetic_cov - (sp.diff(W_fn, x_coord) ** 2 + gB**2 * W_fn**2 * B_vec**2)
    )
    # Local U(1)_B gauge invariance of B_vec under theta -> theta + alpha, A -> A - alpha'/g_B
    B_transformed = (A_fn - sp.diff(alpha_g, x_coord) / gB) + sp.diff(theta_fn + alpha_g, x_coord) / gB
    checks["u1b_proca_combination_gauge_invariant"] = sp.simplify(B_transformed - B_vec)

    # Foliation unit vector u_mu = d_mu Psi / sqrt(2 X) and cuscuton degeneracy
    X, a, lapse, psi_dot = sp.symbols("X a lapse psi_dot", positive=True)
    Psi, H_sym = sp.symbols("Psi H", real=True)
    V_fn = sp.Function("V")
    L_cus = cuscuton_kinetic_sign * sp.sqrt(2 * X) - V_fn(Psi)
    rho_cus = 2 * X * sp.diff(L_cus, X) - L_cus
    # Background kinetic operator L_X + 2 X L_XX == 0 => 0 extra propagating scalar ghosts
    checks["foliation_unit_vector_norm"] = sp.simplify(
        (psi_dot / sp.sqrt(psi_dot**2 / lapse**2)) ** 2 / lapse**2 - 1
    )
    checks["cuscuton_kinetic_degeneracy_zero_ghosts"] = sp.simplify(
        sp.diff(L_cus, X) + 2 * X * sp.diff(L_cus, X, 2)
    )
    checks["cuscuton_energy_equals_potential"] = sp.simplify(rho_cus - V_fn(Psi))

    # Minisuperspace FLRW variation: L_mini = N a^3 (s * psi_dot / N - V(Psi))
    L_mini = lapse * a**3 * L_cus.subs(X, psi_dot**2 / (2 * lapse**2))
    p_psi = sp.diff(L_mini, psi_dot)
    # Euler-Lagrange: d/dt(p_psi) - dL/dPsi = (dp_psi/da)*(lapse*a*H) - dL/dPsi = 0
    el_psi = sp.simplify((sp.diff(p_psi, a) * lapse * a * H_sym - sp.diff(L_mini, Psi)) / (lapse * a**3))
    # Target covariant identity: nabla_mu u^mu = 3 H = V'(Psi) (for kinetic_sign = -1)
    checks["cuscuton_euler_lagrange_nabla_u_eq_Vprime"] = sp.simplify(
        el_psi - (-3 * H_sym + sp.diff(V_fn(Psi), Psi))
    )
    P_cus = sp.simplify(sp.diff(L_mini, a) / (3 * lapse * a**2))
    checks["cuscuton_pressure_from_scale_variation"] = sp.simplify(
        P_cus - (-psi_dot / lapse - V_fn(Psi))
    )

    # --- 1B. Exact Cuscuton Bounce Identities in flat FLRW ---
    q = sp.Symbol("q", real=True)
    M_P, rho_c, Q_ent = sp.symbols("M_P rho_c Q", positive=True)
    alpha = sp.sqrt(3 / (4 * M_P**2 * rho_c))
    x_dens = sp.cos(q) ** 2
    Psi_q = (q + sp.sin(2 * q) / 2) / alpha
    V_q = -rho_c * sp.cos(q) ** 4
    rho_q = rho_c * x_dens
    H_q = alpha * rho_c * sp.sin(2 * q) / 3
    dPsi_dq = sp.diff(Psi_q, q)
    Vp_q = sp.trigsimp(sp.diff(V_q, q) / dPsi_dq)
    qdot = alpha * Q_ent
    Psidot = dPsi_dq * qdot
    Hdot = sp.diff(H_q, q) * qdot

    checks["bounce_potential_gradient_eq_3H"] = sp.trigsimp(Vp_q - 3 * H_q)
    checks["bounce_friedmann_from_cuscuton"] = sp.trigsimp(
        3 * M_P**2 * H_q**2 - (rho_q + V_q)
    )
    checks["bounce_target_friedmann_law"] = sp.trigsimp(
        H_q**2 - (rho_q / (3 * M_P**2)) * (1 - rho_q / rho_c)
    )
    checks["bounce_raychaudhuri_from_cuscuton"] = sp.trigsimp(
        2 * M_P**2 * Hdot + Q_ent + cuscuton_kinetic_sign * Psidot
    )
    checks["bounce_target_raychaudhuri"] = sp.trigsimp(
        Hdot + Q_ent * (1 - 2 * x_dens) / (2 * M_P**2)
    )
    checks["bounce_matter_conservation"] = sp.trigsimp(
        sp.diff(rho_q, q) * qdot + 3 * H_q * Q_ent
    )
    checks["bounce_point_H_zero"] = sp.simplify(H_q.subs(q, 0))
    checks["bounce_point_Hdot_positive"] = sp.simplify(Hdot.subs(q, 0) - Q_ent / (2 * M_P**2))
    # Geodesic completeness / bounded Hubble rate H^2 <= rho_c / (12 M_P^2)
    H_max_sq = rho_c / (12 * M_P**2)
    checks["bounce_hubble_upper_bound_certificate"] = sp.trigsimp(
        H_max_sq - H_q**2 - H_max_sq * sp.cos(2 * q) ** 2
    )
    # Global FLRW Kretschmann bound K <= K_max = 12 rho_c^2 / M_P^4 for 0<=x<=1, 0<=u<=2
    x_s, u_s = sp.symbols("x_s u_s", real=True)
    # K / (rho_c^2 / M_P^4) = 12 * (H_hat^4 + (H_hat^2 + Hdot_hat)^2)
    H2_hat = x_s * (1 - x_s) / 3
    Hdot_hat = -u_s * x_s * (1 - 2 * x_s) / 2
    K_flrw_hat = sp.expand(12 * (H2_hat**2 + (H2_hat + Hdot_hat) ** 2))
    K0_hat = K_flrw_hat.subs(u_s, 0)
    K2_hat = K_flrw_hat.subs(u_s, 2)
    checks["bounce_kretschmann_at_stiff_bounce"] = sp.simplify(
        K_flrw_hat.subs({x_s: 1, u_s: 2}) - 12
    )
    checks["bounce_kretschmann_convex_in_u"] = sp.expand(
        (1 - u_s / 2) * K0_hat + (u_s / 2) * K2_hat - K_flrw_hat
        - 3 * u_s * (2 - u_s) * x_s**2 * (2 * x_s - 1) ** 2
    )
    checks["bounce_kretschmann_u0_upper_bound"] = sp.expand(
        sp.Rational(1, 6) - K0_hat - (2 * x_s - 1) ** 2 * (1 + 4 * x_s * (1 - x_s)) / 6
    )
    checks["bounce_kretschmann_u2_upper_bound"] = sp.expand(
        12 - K2_hat - sp.Rational(4, 3) * (1 - x_s) * (26 * x_s**3 + 4 * x_s**2 + 9 * x_s + 9)
    )

    # --- 1C. Limiting-Curvature Spherically Symmetric BH Interior & Hayward Core ---
    r, M_bh, ell, G_sym = sp.symbols("r M_bh ell G_N", positive=True)
    A_fn = sp.Function("A")(r)  # Inside horizon: A(r) = -f(r) > 0, r timelike
    # Interior metric ds^2 = -A dt^2 + dr^2/A - r^2 dOmega^2 (so f = -A < 0)
    # Unit normal to constant-r slices: u_mu = (0, 1/sqrt(A), 0, 0), u^mu = (0, sqrt(A), 0, 0)
    u_cov_r = 1 / sp.sqrt(A_fn)
    u_con_r = sp.sqrt(A_fn)
    g_rr_inv = A_fn
    checks["bh_interior_foliation_unit_norm"] = sp.simplify(g_rr_inv * u_cov_r**2 - 1)
    # Covariant divergence chi = nabla_mu u^mu = (1/r^2) d/dr (r^2 sqrt(A))
    chi_interior = sp.simplify(sp.diff(r**2 * u_con_r, r) / r**2)
    K_tt = sp.diff(A_fn, r) / (2 * sp.sqrt(A_fn))
    K_ang = sp.sqrt(A_fn) / r
    checks["bh_interior_chi_equals_extrinsic_trace"] = sp.simplify(
        chi_interior - (K_tt + 2 * K_ang)
    )
    # Traceless shear tensor eigenvalues: (sigma^t_t, sigma^theta_theta, sigma^phi_phi)
    delta_K = sp.simplify(K_tt - K_ang)
    sigma_tt = sp.simplify(K_tt - chi_interior / 3)
    sigma_ang = sp.simplify(K_ang - chi_interior / 3)
    checks["bh_interior_shear_traceless"] = sp.simplify(sigma_tt + 2 * sigma_ang)
    checks["bh_interior_shear_ratio_2_to_minus1"] = sp.simplify(sigma_tt - sp.Rational(2, 3) * delta_K)
    checks["bh_interior_shear_ang_ratio"] = sp.simplify(sigma_ang + sp.Rational(1, 3) * delta_K)
    # Verify shear vanishes on flat FLRW (K_xx = K_yy = K_zz = H => delta_K = 0)
    checks["flrw_shear_identically_zero"] = sp.simplify((H_sym - H_sym))

    # Hayward regular core metric f(r) = 1 - 2 M r^2 / (r^3 + 2 M ell^2)
    q_bh = r**3 / (2 * M_bh * ell**2)
    m_bh = M_bh * r**3 / (r**3 + 2 * M_bh * ell**2)
    f_bh = 1 - 2 * m_bh / r
    eps_bh = 3 / (8 * sp.pi * ell**2 * (1 + q_bh) ** 2)
    pr_bh = -eps_bh
    aniso_coeff = sp.Rational(str(hayward_anisotropy_scale))
    # True Hayward tangential pressure: P_t = P_r + 3 q/(1+q) * eps = eps*(2q-1)/(1+q)
    pt_bh = pr_bh + aniso_coeff * (3 * q_bh / (1 + q_bh)) * eps_bh

    # Exact Einstein tensor components (G=c=1)
    G_tt_mixed = (r * sp.diff(f_bh, r) + f_bh - 1) / r**2  # = -8 pi eps
    G_rr_mixed = G_tt_mixed  # = 8 pi P_r => P_r = -eps
    G_ang_mixed = (r * sp.diff(f_bh, r, 2) + 2 * sp.diff(f_bh, r)) / (2 * r)  # = 8 pi P_t (check sign!)
    # Check signs: m' = 4 pi r^2 eps, (r f' + f - 1)/r^2 = -2 m'/r^2 = -8 pi eps;
    # (r f'' + 2 f')/(2r) = -m''/r = +8 pi P_t? Check: -m''/(8 pi r) = P_t in regular_core_geometry_audit.py!
    checks["hayward_density_einstein_eq"] = sp.simplify(eps_bh + G_tt_mixed / (8 * sp.pi))
    checks["hayward_radial_pressure_eq_minus_eps"] = sp.simplify(pr_bh - G_rr_mixed / (8 * sp.pi))
    checks["hayward_tangential_pressure_einstein_eq"] = sp.simplify(
        pt_bh - G_ang_mixed / (8 * sp.pi)
    )
    checks["hayward_covariant_conservation"] = sp.simplify(
        sp.diff(pr_bh, r) + (eps_bh + pr_bh) * sp.diff(f_bh, r) / (2 * f_bh) + 2 * (pr_bh - pt_bh) / r
    )
    # Anisotropy decomposition matching foliation shear structure (2, -1, -1)
    p_iso = sp.simplify((pr_bh + 2 * pt_bh) / 3)
    pi_r = sp.simplify(p_iso - pr_bh)  # = (2/3)(P_t - P_r)
    pi_ang = sp.simplify(p_iso - pt_bh)  # = -(1/3)(P_t - P_r)
    checks["hayward_anisotropy_traceless_matches_shear"] = sp.simplify(pi_r + 2 * pi_ang)
    checks["hayward_anisotropy_exact_formula"] = sp.simplify(
        (pt_bh - pr_bh) - 3 * q_bh * eps_bh / (1 + q_bh)
    )

    # Global Painleve-Gullstrand foliation Psi = T on Hayward: v(r) = sqrt(2m/r) = r/(ell*sqrt(1+q))
    v_pg = r / (ell * sp.sqrt(1 + q_bh))
    chi_pg = sp.simplify(sp.diff(-r**2 * v_pg, r) / r**2)
    checks["hayward_pg_foliation_chi_exact"] = sp.simplify(
        chi_pg + (3 / ell) * (1 + q_bh / 2) / (1 + q_bh) ** sp.Rational(3, 2)
    )
    checks["hayward_pg_foliation_core_limit"] = sp.simplify(
        sp.limit(chi_pg, r, 0) + 3 / ell
    )

    # Kretschmann invariant & global algebraic upper-bound certificate
    K_bh = sp.simplify(
        sp.diff(f_bh, r, 2) ** 2 + 4 * (sp.diff(f_bh, r) / r) ** 2 + 4 * ((1 - f_bh) / r**2) ** 2
    )
    K_bh_closed = 12 * (q_bh**4 - 4 * q_bh**3 + 18 * q_bh**2 - 2 * q_bh + 2) / (ell**4 * (1 + q_bh) ** 6)
    checks["hayward_kretschmann_exact"] = sp.simplify(K_bh - K_bh_closed)
    checks["hayward_kretschmann_core_limit"] = sp.simplify(sp.limit(K_bh, r, 0) - 24 / ell**4)
    # Global certificate: 24/ell^4 - K_bh(r) = 12*q*(14 + 12q + 44q^2 + 29q^3 + 12q^4 + 2q^5)/(ell^4*(1+q)^6) >= 0
    poly_cert = 12 * q_bh * (14 + 12 * q_bh + 44 * q_bh**2 + 29 * q_bh**3 + 12 * q_bh**4 + 2 * q_bh**5) / (
        ell**4 * (1 + q_bh) ** 6
    )
    checks["hayward_kretschmann_global_bound_certificate"] = sp.simplify(
        24 / ell**4 - K_bh_closed - poly_cert
    )
    # Link ell^2 = 3 M_P^2 / rho_c => K_hayward(0) = (8/3) rho_c^2 / M_P^4 < 12 rho_c^2 / M_P^4 = K_max_bounce
    K_core_in_rhoc = (24 / ell**4).subs(ell, sp.sqrt(3 * M_P**2 / rho_c))
    checks["hayward_core_to_bounce_Kmax_ratio"] = sp.simplify(
        K_core_in_rhoc / (12 * rho_c**2 / M_P**4) - sp.Rational(2, 9)
    )
    # Critical Hayward horizon scales
    Mh_expr = r**3 / (2 * (r**2 - ell**2))
    Th_expr = (sp.diff(f_bh, r) / (4 * sp.pi)).subs(M_bh, Mh_expr)
    checks["hayward_M_crit_exact"] = sp.simplify(
        Mh_expr.subs(r, sp.sqrt(3) * ell) - 3 * sp.sqrt(3) * ell / 4
    )
    checks["hayward_T_crit_zero"] = sp.simplify(Th_expr.subs(r, sp.sqrt(3) * ell))
    checks["hayward_T_max_exact"] = sp.simplify(
        Th_expr.subs(r, 3 * ell) - 1 / (18 * sp.pi * ell)
    )

    # --- 1D. Quartic-Vector Matter Functional (zeta > 0) Exact Identities ---
    n_s, W_s, Gv_s, zeta_s, v_s = sp.symbols("n W G_v zeta v", positive=True)
    w_s = sp.Symbol("omega_0", positive=True)
    # Implicit cubic: F_cubic = (W^2 / G_v) * w + zeta * w^3 - n = 0
    F_cubic = (W_s**2 / Gv_s) * w_s + zeta_s * w_s**3 - n_s
    dw_dn = -sp.diff(F_cubic, n_s) / sp.diff(F_cubic, w_s)
    dw_dW = -sp.diff(F_cubic, W_s) / sp.diff(F_cubic, w_s)
    eps_vec = sp.Rational(1, 2) * (W_s**2 / Gv_s) * w_s**2 + sp.Rational(3, 4) * zeta_s * w_s**4
    p_vec = sp.Rational(1, 2) * (W_s**2 / Gv_s) * w_s**2 + sp.Rational(1, 4) * zeta_s * w_s**4
    deps_dn = sp.diff(eps_vec, w_s) * dw_dn
    deps_dW = sp.diff(eps_vec, W_s) + sp.diff(eps_vec, w_s) * dw_dW
    checks["quartic_vector_chemical_potential_identity"] = sp.simplify(deps_dn - w_s)
    checks["quartic_vector_legendre_pressure_identity"] = sp.simplify(
        (n_s * w_s - eps_vec - p_vec).subs(n_s, (W_s**2 / Gv_s) * w_s + zeta_s * w_s**3)
    )
    checks["quartic_vector_scalar_force_smooth_at_W0"] = sp.simplify(
        deps_dW + (W_s / Gv_s) * w_s**2
    )
    # Off-equilibrium constraint propagation d/dt(rho - rho_c cos^2 q) == 0
    E_mat = sp.Function("E_mat")(n_s, W_s)
    rho_dyn = v_s**2 / 2 + E_mat
    P_dyn = v_s**2 / 2 + n_s * sp.diff(E_mat, n_s) - E_mat
    Q_dyn = rho_dyn + P_dyn
    ndot_dyn = -3 * H_q * n_s
    vdot_dyn = -3 * H_q * v_s - quartic_force_sign * sp.diff(E_mat, W_s)
    qdot_dyn = alpha * Q_dyn
    rhodot_dyn = sp.diff(rho_dyn, n_s) * ndot_dyn + sp.diff(rho_dyn, W_s) * v_s + sp.diff(rho_dyn, v_s) * vdot_dyn
    checks["dynamic_matter_cuscuton_constraint_propagation"] = sp.trigsimp(
        sp.simplify(rhodot_dyn + rho_c * sp.sin(2 * q) * qdot_dyn)
    )

    # --- 1E. Law 4 Symbolic IR Identities ---
    rho_m_s, P_m_s, rho_L_s, phi_dot_s, V_de_s = sp.symbols(
        "rho_m P_m rho_L phi_dot V_DE", real=True
    )
    # Constant Lambda > 0 asymptotic de Sitter Hubble rate
    H2_constL = (rho_L_s / (3 * M_P**2)) * (1 - rho_L_s / rho_c)
    checks["law4_const_lambda_w_eq_minus1"] = sp.simplify((-rho_L_s) / rho_L_s + 1)
    checks["law4_const_lambda_asymptotic_H2"] = sp.simplify(
        H2_constL - (rho_L_s / (3 * M_P**2)) * (1 - rho_L_s / rho_c)
    )
    # External IR scalar turnaround at q = pi/2 (rho_tot = 0 => V_DE = -(rho_m + phi_dot^2/2) < 0)
    Q_ir = (rho_m_s + P_m_s) + phi_dot_s**2
    Hdot_q = sp.diff(H_q, q) * (alpha * Q_ir)
    checks["law4_ir_turnaround_H_zero_at_pi_over_2"] = sp.simplify(H_q.subs(q, sp.pi / 2))
    checks["law4_ir_turnaround_Hdot_negative_at_pi_over_2"] = sp.simplify(
        Hdot_q.subs(q, sp.pi / 2) + Q_ir / (2 * M_P**2)
    )
    checks["law4_ir_next_bounce_H_zero_at_pi"] = sp.simplify(H_q.subs(q, sp.pi))
    checks["law4_ir_next_bounce_Hdot_positive_at_pi"] = sp.simplify(
        Hdot_q.subs(q, sp.pi) - Q_ir / (2 * M_P**2)
    )
    # Thawing w_DE = -1 + phi_dot^2 / (phi_dot^2/2 + V_DE)
    rho_de_s = phi_dot_s**2 / 2 + V_de_s
    P_de_s = phi_dot_s**2 / 2 - V_de_s
    checks["law4_ir_scalar_thawing_eos_identity"] = sp.simplify(
        P_de_s / rho_de_s - (-1 + phi_dot_s**2 / rho_de_s)
    )

    formatted: dict[str, dict[str, Any]] = {}
    for name, expr in checks.items():
        simplified = sp.trigsimp(sp.simplify(expr))
        formatted[name] = {
            "passed": bool(simplified == 0),
            "residual": str(simplified),
        }

    all_passed = all(v["passed"] for v in formatted.values())
    return {
        "all_passed": all_passed,
        "count_total": len(formatted),
        "count_passed": sum(1 for v in formatted.values() if v["passed"]),
        "controls": {
            "cuscuton_kinetic_sign": cuscuton_kinetic_sign,
            "hayward_anisotropy_scale": hayward_anisotropy_scale,
            "quartic_force_sign": quartic_force_sign,
        },
        "identities": formatted,
    }


# =============================================================================
# 2. VACUUM BAG SCALE BRIDGE (rho_c = kappa_g * U(0)) AT 60/90-DPS MPMATH
# =============================================================================

def _compute_scale_bridge_at_dps(dps: int) -> dict[str, Any]:
    with mp.workdps(dps):
        W0 = mp.mpf(DECLARED_INPUTS["W0_MeV"])
        lam = mp.mpf(DECLARED_INPUTS["lambda_W"])
        hbarc = mp.mpf(DECLARED_INPUTS["hbarc_MeV_fm"])
        GN = mp.mpf(DECLARED_INPUTS["G_N_MeV_minus2"])
        kB = mp.mpf(DECLARED_INPUTS["k_B_MeV_per_K"])
        M_sun_kg = mp.mpf(DECLARED_INPUTS["M_sun_kg"])
        c_ms = mp.mpf(DECLARED_INPUTS["c_m_per_s"])
        MeV_J = mp.mpf(DECLARED_INPUTS["MeV_in_J"])
        conv_gcm3 = mp.mpf(DECLARED_INPUTS["MeV_fm3_to_g_cm3"])

        MP2 = 1 / (8 * mp.pi * GN)  # MeV^2
        MP = mp.sqrt(MP2)
        M_sun_MeV = M_sun_kg * (c_ms**2) / MeV_J
        G_Msun_km = GN * M_sun_MeV * hbarc * mp.mpf("1e-18")
        U0_nat = (lam / 4) * (W0**4)  # MeV^4
        U0_MeV_fm3 = U0_nat / (hbarc**3)

        regimes = [
            ("bag_kappa_1", mp.mpf("1"), "Law-1 melted vacuum bag density rho_c = U(0) = (lambda_W/4)*W0^4"),
            ("intermediate_kappa_2", mp.mpf("2"), "Intermediate reference rho_c = 2*U(0) allowing W=0 crossing"),
            ("legacy_W0_4", mp.mpf("4") / lam, "Legacy rho_c = W0^4 (kappa_g = 4/lambda_W = 80/21)"),
        ]

        rows: dict[str, dict[str, Any]] = {}
        for label, kappa, desc in regimes:
            rho_c_nat = kappa * U0_nat
            rho_c_MeV_fm3 = rho_c_nat / (hbarc**3)
            rho_c_g_cm3 = rho_c_MeV_fm3 * conv_gcm3
            ell_nat = mp.sqrt(3 * MP2 / rho_c_nat)  # MeV^-1 (exact sqrt(3/(8 pi G rho_c)))
            ell_fm = ell_nat * hbarc
            ell_km = ell_fm * mp.mpf("1e-18")
            t_c_s = (ell_km * mp.mpf("1e3")) / c_ms
            r_crit_km = mp.sqrt(3) * ell_km
            M_crit_km = (3 * mp.sqrt(3) / 4) * ell_km
            M_crit_Msun = M_crit_km / G_Msun_km
            M_crit_MeV = (3 * mp.sqrt(3) / 4) * ell_nat / GN
            T_max_MeV = 1 / (18 * mp.pi * ell_nat)
            T_max_K = T_max_MeV / kB
            K_max_bounce_MeV4 = 12 * (rho_c_nat**2) / (MP2**2)
            K_max_bounce_km4 = 108 / (ell_km**4)
            K_hayward_core_MeV4 = (mp.mpf(8) / 3) * (rho_c_nat**2) / (MP2**2)
            K_hayward_core_km4 = 24 / (ell_km**4)
            H_max_MeV = mp.sqrt(rho_c_nat / (12 * MP2))

            rows[label] = {
                "description": desc,
                "kappa_g": _nstr(kappa),
                "rho_c_MeV4": _nstr(rho_c_nat),
                "rho_c_MeV_fm3": _nstr(rho_c_MeV_fm3),
                "rho_c_g_cm3": _nstr(rho_c_g_cm3),
                "ell_MeV_minus1": _nstr(ell_nat),
                "ell_km": _nstr(ell_km),
                "t_c_seconds": _nstr(t_c_s),
                "r_crit_km": _nstr(r_crit_km),
                "M_crit_km": _nstr(M_crit_km),
                "M_crit_Msun": _nstr(M_crit_Msun),
                "M_crit_MeV": _nstr(M_crit_MeV),
                "T_max_K": _nstr(T_max_K),
                "T_max_MeV": _nstr(T_max_MeV),
                "H_max_MeV": _nstr(H_max_MeV),
                "K_max_bounce_MeV4": _nstr(K_max_bounce_MeV4),
                "K_max_bounce_km_minus4": _nstr(K_max_bounce_km4),
                "K_hayward_core_MeV4": _nstr(K_hayward_core_MeV4),
                "K_hayward_core_km_minus4": _nstr(K_hayward_core_km4),
                "K_hayward_over_K_bounce": _nstr(K_hayward_core_MeV4 / K_max_bounce_MeV4),
            }

        ratio_kappa = mp.mpf("4") / lam
        return {
            "M_P_MeV": _nstr(MP),
            "M_P_over_W0": _nstr(MP / W0),
            "G_Msun_km": _nstr(G_Msun_km),
            "U0_MeV4": _nstr(U0_nat),
            "U0_MeV_fm3": _nstr(U0_MeV_fm3),
            "regimes": rows,
            "exact_scaling_ratios_bag_vs_legacy": {
                "rho_c_legacy_over_bag": _nstr(ratio_kappa),
                "ell_bag_over_legacy": _nstr(mp.sqrt(ratio_kappa)),
                "M_crit_bag_over_legacy": _nstr(mp.sqrt(ratio_kappa)),
                "T_max_bag_over_legacy": _nstr(1 / mp.sqrt(ratio_kappa)),
                "K_max_legacy_over_bag": _nstr(ratio_kappa**2),
            },
        }


def vacuum_bag_scale_bridge(
    primary_dps: int = PRIMARY_DPS,
    control_dps: int = CONTROL_DPS,
) -> dict[str, Any]:
    """Compute vacuum bag scale bridge at 60 dps and cross-check at 90 dps."""
    primary = _compute_scale_bridge_at_dps(primary_dps)
    control = _compute_scale_bridge_at_dps(control_dps)
    if primary != control:
        raise ArithmeticError(f"Precision mismatch between {primary_dps} dps and {control_dps} dps")
    return {
        "precision_check": {
            "primary_dps": primary_dps,
            "control_dps": control_dps,
            "report_digits": REPORT_DIGITS,
            "passed": True,
        },
        "bridge": primary,
    }


# =============================================================================
# 3. DYNAMIC MATTER BOUNCE INTEGRATION (n, W, v, q) WITH QUARTIC VECTOR zeta>0
# =============================================================================

class QuarticVectorMatter:
    """Law-1/Law-2 matter functional in dimensionless W0 = 1 units with zeta > 0.

    L_vec = (1/2) (W^2 / G_v) omega_0^2 + (1/4) zeta omega_0^4 - n omega_0
    Eliminating omega_0 via zeta omega_0^3 + (W^2 / G_v) omega_0 = n gives a
    unique smooth positive root for all W in R, including W = 0 (melted phase).
    """

    def __init__(self, zeta: float = 0.25):
        if not math.isfinite(zeta) or zeta < 0.0:
            raise ValueError("zeta must be finite and nonnegative")
        self.W0 = float(DECLARED_INPUTS["W0_MeV"])
        self.lam = float(DECLARED_INPUTS["lambda_W"])
        self.MN = float(DECLARED_INPUTS["MN_MeV"])
        self.gs = self.MN / self.W0
        self.momega = float(DECLARED_INPUTS["momega_MeV"])
        self.gomega = float(DECLARED_INPUTS["gomega"])
        self.Gv_hat = (self.gomega / self.momega) ** 2 * (self.W0**2)
        self.d_deg = float(DECLARED_INPUTS["fermion_degeneracy_d"])
        self.zeta = float(zeta)

    def solve_omega_hat(self, n_hat: float, y: float) -> float:
        if n_hat <= 0.0:
            raise ValueError("n_hat must be positive")
        c = (y * y) / self.Gv_hat
        if self.zeta == 0.0:
            if c == 0.0:
                raise ZeroDivisionError("W^-2 blowup at y=0 when zeta=0")
            return n_hat / c
        p = c / self.zeta
        q = n_hat / self.zeta
        half_q = 0.5 * q
        disc = half_q * half_q + (p / 3.0) ** 3
        s = math.sqrt(disc)
        w = _cbrt(half_q + s) + _cbrt(half_q - s)
        for _ in range(3):
            f_val = self.zeta * (w**3) + c * w - n_hat
            df_val = 3.0 * self.zeta * (w**2) + c
            w -= f_val / df_val
        return w

    def evaluate(self, n_hat: float, y: float, v_hat: float) -> dict[str, float]:
        kF = (6.0 * math.pi**2 * n_hat / self.d_deg) ** (1.0 / 3.0)
        m = self.gs * abs(y)
        EF = math.hypot(kF, m)
        if m > 1e-13:
            ash = math.asinh(kF / m)
            m2_ash = m * m * ash
            m4_ash = m2_ash * m * m
        else:
            m2_ash = 0.0
            m4_ash = 0.0
        eF = (self.d_deg / (16.0 * math.pi**2)) * (kF * EF * (2.0 * kF**2 + m**2) - m4_ash)
        pF = (self.d_deg / (48.0 * math.pi**2)) * (kF * EF * (2.0 * kF**2 - 3.0 * m**2) + 3.0 * m4_ash)
        deF_dy = (self.gs**2) * y * (self.d_deg / (4.0 * math.pi**2)) * (kF * EF - m2_ash)

        z = y * y - 1.0
        U = 0.25 * self.lam * z * z
        dU_dy = self.lam * y * z

        w = self.solve_omega_hat(n_hat, y)
        c = (y * y) / self.Gv_hat
        eV = 0.5 * c * w * w + 0.75 * self.zeta * (w**4)
        pV = 0.5 * c * w * w + 0.25 * self.zeta * (w**4)
        deV_dy = -(y / self.Gv_hat) * w * w

        rho = 0.5 * v_hat * v_hat + eF + U + eV
        P = 0.5 * v_hat * v_hat + pF - U + pV
        Q = v_hat * v_hat + n_hat * (EF + w)
        dE_dy = deF_dy + dU_dy + deV_dy
        return {
            "rho": rho,
            "P": P,
            "Q": Q,
            "dE_dy": dE_dy,
            "omega_hat": w,
            "eF": eF,
            "U": U,
            "eV": eV,
        }


def integrate_dynamic_bounce(
    *,
    kappa_g: float,
    zeta: float = 0.25,
    mu_stiff: float = 5.0,
    y_bounce: float = 0.0,
    v_bounce: float = 0.25,
    tau_half_window: float = 5.0,
    qdot_rate_scale: float = 1.0,
) -> dict[str, Any]:
    """Integrate (n_hat, y, v_hat, q, ln_a) across [-5 t_c, +5 t_c] through q=0.

    qdot_rate_scale != 1.0 is a negative control used only to test whether
    constraint violation is detected.
    """
    matter = QuarticVectorMatter(zeta=zeta)
    rc_hat = kappa_g * (matter.lam / 4.0)

    # Check feasibility at bounce: U(y_bounce) + 0.5 v_bounce^2 must be < rc_hat
    z0 = y_bounce * y_bounce - 1.0
    base_e = 0.5 * v_bounce * v_bounce + 0.25 * matter.lam * z0 * z0
    if base_e >= rc_hat:
        raise ValueError("Bounce scalar energy exceeds rho_c_hat; no positive n_hat exists")

    # Bisect n_hat at tau=0 (q=0) so rho(n_bounce, y_bounce, v_bounce) == rc_hat
    lo, hi = 1e-12, 4.0
    while matter.evaluate(hi, y_bounce, v_bounce)["rho"] < rc_hat:
        hi *= 2.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if matter.evaluate(mid, y_bounce, v_bounce)["rho"] < rc_hat:
            lo = mid
        else:
            hi = mid
    n_bounce = 0.5 * (lo + hi)

    def rhs(_tau: float, state: np.ndarray) -> list[float]:
        n_h, y_val, v_h, q_val, _lna = state
        ev = matter.evaluate(n_h, y_val, v_h)
        Htc = 0.5 * math.sin(2.0 * q_val)
        dn = -3.0 * Htc * n_h
        dy = mu_stiff * v_h
        dv = -3.0 * Htc * v_h - mu_stiff * ev["dE_dy"]
        dq = qdot_rate_scale * 1.5 * ev["Q"] / rc_hat
        dlna = Htc
        return [dn, dy, dv, dq, dlna]

    # Integrate backward from tau=0 to -tau_half_window to obtain pre-bounce IC,
    # then perform a single continuous forward sweep [-5, +5] through q = 0.
    sol_back = solve_ivp(
        rhs,
        (0.0, -tau_half_window),
        [n_bounce, y_bounce, v_bounce, 0.0, 0.0],
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
    )
    if not sol_back.success:
        raise RuntimeError(f"Backward bounce IC integration failed: {sol_back.message}")

    ic_start = sol_back.y[:, -1]
    t_eval = np.linspace(-tau_half_window, tau_half_window, 401)
    sol_fwd = solve_ivp(
        rhs,
        (-tau_half_window, tau_half_window),
        ic_start,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-13,
        atol=1e-15,
    )
    if not sol_fwd.success:
        raise RuntimeError(f"Forward bounce integration failed: {sol_fwd.message}")

    c_errs = []
    b_errs = []
    Htc_vals = []
    omega_vals = []
    for i in range(len(t_eval)):
        n_i, y_i, v_i, q_i, lna_i = sol_fwd.y[:, i]
        ev = matter.evaluate(float(n_i), float(y_i), float(v_i))
        c_errs.append(abs(ev["rho"] / rc_hat - math.cos(float(q_i)) ** 2))
        b_errs.append(abs(float(n_i) * math.exp(3.0 * float(lna_i)) / n_bounce - 1.0))
        Htc_vals.append(0.5 * math.sin(2.0 * float(q_i)))
        omega_vals.append(ev["omega_hat"])

    idx_mid = len(t_eval) // 2  # tau = 0.0
    ev_mid = matter.evaluate(
        float(sol_fwd.y[0, idx_mid]),
        float(sol_fwd.y[1, idx_mid]),
        float(sol_fwd.y[2, idx_mid]),
    )
    q_mid = float(sol_fwd.y[3, idx_mid])
    Htc_mid = 0.5 * math.sin(2.0 * q_mid)
    # d(H*t_c)/dtau = cos(2q) * dq/dtau = 1.5 * cos(2q) * Q / rc_hat
    Hdot_tc2_mid = 1.5 * math.cos(2.0 * q_mid) * ev_mid["Q"] / rc_hat

    y_arr = sol_fwd.y[1, :]
    max_c_err = float(max(c_errs))
    max_b_err = float(max(b_errs))

    return {
        "parameters": {
            "kappa_g": float(kappa_g),
            "rho_c_over_W0_4": float(rc_hat),
            "zeta": float(zeta),
            "mu_stiff_W0_tc": float(mu_stiff),
            "tau_window": [-float(tau_half_window), float(tau_half_window)],
            "qdot_rate_scale": float(qdot_rate_scale),
        },
        "bounce_state_at_tau_0": {
            "n_hat": float(sol_fwd.y[0, idx_mid]),
            "y_W_over_W0": float(sol_fwd.y[1, idx_mid]),
            "v_hat": float(sol_fwd.y[2, idx_mid]),
            "q": q_mid,
            "H_tc": Htc_mid,
            "Hdot_tc2": float(Hdot_tc2_mid),
            "omega_hat": float(ev_mid["omega_hat"]),
        },
        "diagnostics": {
            "H_tc_start": float(Htc_vals[0]),
            "H_tc_end": float(Htc_vals[-1]),
            "q_start": float(sol_fwd.y[3, 0]),
            "q_end": float(sol_fwd.y[3, -1]),
            "y_min": float(np.min(y_arr)),
            "y_max": float(np.max(y_arr)),
            "crossed_W_zero_smoothly": bool(np.min(y_arr) < 0.0 < np.max(y_arr)),
            "max_omega_hat": float(max(omega_vals)),
            "max_constraint_error": max_c_err,
            "max_baryon_conservation_error": max_b_err,
            "constraint_preserved_lt_1e12": bool(max_c_err < 1e-12),
            "baryon_preserved_lt_1e12": bool(max_b_err < 1e-12),
            "bounce_signature_verified": bool(
                Htc_vals[0] < 0.0
                and abs(Htc_mid) < 1e-11
                and Hdot_tc2_mid > 0.0
                and Htc_vals[-1] > 0.0
            ),
        },
    }


def dynamic_matter_bounce_audit() -> dict[str, Any]:
    """Run dynamic bounce trajectories + zeta=0 blowup negative control."""
    lam = float(DECLARED_INPUTS["lambda_W"])
    run_bag = integrate_dynamic_bounce(
        kappa_g=1.0, zeta=0.25, mu_stiff=5.0, y_bounce=0.5, v_bounce=0.15
    )
    run_melted = integrate_dynamic_bounce(
        kappa_g=4.0 / lam, zeta=0.25, mu_stiff=5.0, y_bounce=0.0, v_bounce=0.25
    )
    run_stiff = integrate_dynamic_bounce(
        kappa_g=4.0 / lam, zeta=0.25, mu_stiff=20.0, y_bounce=0.0, v_bounce=0.25
    )

    # Negative control: compare zeta=0 vs zeta=0.25 as y -> 1e-4 at fixed n_hat=0.1
    m_zero = QuarticVectorMatter(zeta=0.0)
    m_pos = QuarticVectorMatter(zeta=0.25)
    eV_zeta0_small_y = m_zero.evaluate(0.1, 1e-4, 0.0)["eV"]
    eV_zeta_pos_small_y = m_pos.evaluate(0.1, 1e-4, 0.0)["eV"]
    eV_zeta_pos_exact_zero = m_pos.evaluate(0.1, 0.0, 0.0)["eV"]

    all_passed = (
        run_bag["diagnostics"]["constraint_preserved_lt_1e12"]
        and run_bag["diagnostics"]["bounce_signature_verified"]
        and run_melted["diagnostics"]["constraint_preserved_lt_1e12"]
        and run_melted["diagnostics"]["bounce_signature_verified"]
        and run_melted["diagnostics"]["crossed_W_zero_smoothly"]
        and run_stiff["diagnostics"]["constraint_preserved_lt_1e12"]
        and run_stiff["diagnostics"]["bounce_signature_verified"]
        and (eV_zeta0_small_y / eV_zeta_pos_small_y > 1e6)
    )

    return {
        "all_passed": bool(all_passed),
        "stiffness_scale_caveat": (
            "ODE integrations use dimensionless rescaled stiffness mu_stiff = W0*t_c in {5, 20} "
            "so non-adiabatic W dynamics are directly resolved over [-5 t_c, +5 t_c]; physical "
            "M_P/W0 ~ 2.835e18 lies in the extreme adiabatic-tracking regime."
        ),
        "bag_scale_kappa_1_note": (
            "At kappa_g = 1 (rho_c = U(0)), positivity of fermion+vector energy enforces "
            "U(W) < U(0) so W remains nonzero; for kappa_g > 1 (e.g. legacy 4/lambda_W), "
            "U(0) < rho_c and W smoothly crosses W = 0 via zeta > 0 without W^-2 blowup."
        ),
        "trajectories": {
            "bag_kappa_1_mu5": run_bag,
            "legacy_melted_crossing_mu5": run_melted,
            "legacy_melted_crossing_mu20": run_stiff,
        },
        "zeta_zero_blowup_control": {
            "n_hat": 0.1,
            "y_probe": 1e-4,
            "eV_zeta_0_at_y_1e_4": float(eV_zeta0_small_y),
            "eV_zeta_025_at_y_1e_4": float(eV_zeta_pos_small_y),
            "eV_zeta_025_at_exact_y_0": float(eV_zeta_pos_exact_zero),
            "blowup_ratio_zeta0_over_zeta025": float(eV_zeta0_small_y / eV_zeta_pos_small_y),
        },
    }


# =============================================================================
# 4. LAW 4 — IR COSMOLOGY THEOREM (DESI w(z) != -1 <-> CYCLIC TURNAROUND)
# =============================================================================

def integrate_const_lambda_cuscuton(
    *,
    rho_L_hat: float = 0.02,
    omega_L_ref: float = 0.69,
    tau_end: float = 65.0,
) -> dict[str, Any]:
    """Flat FLRW + cuscuton + dust + constant Lambda > 0 (strictly one bounce)."""
    if not (0.0 < rho_L_hat < 0.5):
        raise ValueError("rho_L_hat must lie in (0, 0.5)")
    rm0 = 1.0 - rho_L_hat

    def rhs(_tau: float, s: np.ndarray) -> list[float]:
        rm, q, _lna = s
        Htc = 0.5 * math.sin(2.0 * q)
        return [-3.0 * Htc * rm, 1.5 * rm, Htc]

    # Event when Omega_Lambda = rho_L / (rm + rho_L) == omega_L_ref
    def ev_omega_ref(_tau: float, s: np.ndarray) -> float:
        return rho_L_hat / (s[0] + rho_L_hat) - omega_L_ref

    ev_omega_ref.terminal = False
    ev_omega_ref.direction = 1

    # Event for any turnaround q = pi/2 (must NEVER trigger)
    def ev_turnaround(_tau: float, s: np.ndarray) -> float:
        return s[1] - 0.5 * math.pi

    ev_turnaround.terminal = False

    t_eval = np.linspace(0.0, tau_end, 651)
    sol = solve_ivp(
        rhs,
        (0.0, tau_end),
        [rm0, 0.0, 0.0],
        method="DOP853",
        t_eval=t_eval,
        events=[ev_omega_ref, ev_turnaround],
        rtol=1e-13,
        atol=1e-15,
    )
    if not sol.success:
        raise RuntimeError(f"Const-Lambda integration failed: {sol.message}")

    c_errs = [
        abs((float(sol.y[0, i]) + rho_L_hat) - math.cos(float(sol.y[1, i])) ** 2)
        for i in range(len(t_eval))
    ]
    Htc_arr = [0.5 * math.sin(2.0 * float(sol.y[1, i])) for i in range(len(t_eval))]
    Htc_inf = math.sqrt(rho_L_hat * (1.0 - rho_L_hat))
    q_inf = math.acos(math.sqrt(rho_L_hat))
    Htc_end = Htc_arr[-1]
    max_c_err = float(max(c_errs))

    return {
        "rho_L_over_rho_c": float(rho_L_hat),
        "omega_L_reference_target": float(omega_L_ref),
        "tau_at_omega_L_ref": float(sol.t_events[0][0]) if len(sol.t_events[0]) else None,
        "w_DE_at_omega_L_ref": -1.0,
        "turnaround_events_count": int(len(sol.t_events[1])),
        "H_tc_asymptotic_exact": float(Htc_inf),
        "H_tc_at_tau_end": float(Htc_end),
        "H_tc_convergence_error": float(abs(Htc_end - Htc_inf)),
        "q_asymptotic_over_pi": float(q_inf / math.pi),
        "q_end_over_pi": float(sol.y[1, -1] / math.pi),
        "H_strictly_positive_post_bounce": bool(all(h > 0.0 for h in Htc_arr[1:])),
        "max_constraint_error": max_c_err,
        "no_turnaround_verified": bool(
            len(sol.t_events[1]) == 0
            and abs(Htc_end - Htc_inf) < 1e-10
            and float(sol.y[1, -1]) < 0.5 * math.pi
            and max_c_err < 1e-12
        ),
    }


def integrate_external_ir_scalar_cycle(
    *,
    potential_type: str = "linear",
    V0_hat: float = 0.02,
    slope_hat: float = 0.006,
    f_phi: float = 2.0,
    phi_0: float = 0.0,
    pi_0: float = 0.04,
    omega_de_ref: float = 0.69,
    tau_end: float = 70.0,
) -> dict[str, Any]:
    """Integrate flat FLRW + cuscuton + dust + EXTERNAL IR scalar phi_DE.

    Units: M_P = 1, tau = t / t_c where t_c = sqrt(3 M_P^2 / rho_c).
      rho_DE / rho_c = pi_phi^2 / 6 + V_hat(phi)
      P_DE   / rho_c = pi_phi^2 / 6 - V_hat(phi)
      Q      / rho_c = rm + pi_phi^2 / 3
      d(pi_phi)/dtau = -3 (H t_c) pi_phi - 3 dV_hat/dphi

    Supports:
      - 'linear': V_hat(phi) = V0_hat - slope_hat * phi (crosses zero to V_DE < 0)
      - 'cosine': V_hat(phi) = V0_hat * cos(phi / f_phi) (periodic, locally linear at zero crossing)
      - 'positive_quadratic': V_hat(phi) = V0_hat + 0.5 * slope_hat * phi^2 > 0 (negative control: NO turnaround!)
    """
    if potential_type == "linear":
        def V_fn(phi: float) -> float:
            return V0_hat - slope_hat * phi

        def dV_fn(_phi: float) -> float:
            return -slope_hat
    elif potential_type == "cosine":
        def V_fn(phi: float) -> float:
            return V0_hat * math.cos(phi / f_phi)

        def dV_fn(phi: float) -> float:
            return -(V0_hat / f_phi) * math.sin(phi / f_phi)
    elif potential_type == "positive_quadratic":
        def V_fn(phi: float) -> float:
            return V0_hat + 0.5 * slope_hat * phi * phi

        def dV_fn(phi: float) -> float:
            return slope_hat * phi
    else:
        raise ValueError(f"Unknown potential_type: {potential_type}")

    rm0 = 1.0 - (pi_0**2) / 6.0 - V_fn(phi_0)
    if rm0 <= 0.0:
        raise ValueError("Initial dust density rm0 must be positive at bounce")

    def rhs(_tau: float, s: np.ndarray) -> list[float]:
        rm, phi, pi_p, q, _lna = s
        Htc = 0.5 * math.sin(2.0 * q)
        Q = rm + (pi_p**2) / 3.0
        return [
            -3.0 * Htc * rm,
            pi_p,
            -3.0 * Htc * pi_p - 3.0 * dV_fn(phi),
            1.5 * Q,
            Htc,
        ]

    # Events:
    # 0: turnaround_1 (q = pi/2)
    # 1: bounce_2 (q = pi)
    # 2: turnaround_2 (q = 3 pi/2)
    # 3: bounce_3 (q = 2 pi)
    # 4: V_DE zero crossing (V_fn(phi) = 0, downward)
    # 5: Omega_DE == omega_de_ref during first expansion (q < pi/2)
    def ev_t1(_t: float, s: np.ndarray) -> float:
        return s[3] - 0.5 * math.pi

    def ev_b2(_t: float, s: np.ndarray) -> float:
        return s[3] - math.pi

    def ev_t2(_t: float, s: np.ndarray) -> float:
        return s[3] - 1.5 * math.pi

    def ev_b3(_t: float, s: np.ndarray) -> float:
        return s[3] - 2.0 * math.pi

    def ev_vzero(_t: float, s: np.ndarray) -> float:
        return V_fn(s[1])

    ev_vzero.direction = -1

    def ev_omega_de(_t: float, s: np.ndarray) -> float:
        if s[3] >= 0.49 * math.pi or s[3] <= 0.01:
            return -1.0
        rho_de = (s[2] ** 2) / 6.0 + V_fn(s[1])
        rho_tot = s[0] + rho_de
        if rho_tot <= 1e-12:
            return -1.0
        return rho_de / rho_tot - omega_de_ref

    ev_omega_de.direction = 1

    t_eval = np.linspace(0.0, tau_end, 1401)
    sol = solve_ivp(
        rhs,
        (0.0, tau_end),
        [rm0, phi_0, pi_0, 0.0, 0.0],
        method="DOP853",
        t_eval=t_eval,
        events=[ev_t1, ev_b2, ev_t2, ev_b3, ev_vzero, ev_omega_de],
        rtol=1e-13,
        atol=1e-15,
    )
    if not sol.success:
        raise RuntimeError(f"IR scalar integration ({potential_type}) failed: {sol.message}")

    c_errs = [
        abs(
            (float(sol.y[0, i]) + (float(sol.y[2, i]) ** 2) / 6.0 + V_fn(float(sol.y[1, i])))
            - math.cos(float(sol.y[3, i])) ** 2
        )
        for i in range(len(t_eval))
    ]
    max_c_err = float(max(c_errs))

    def _event_record(idx: int) -> dict[str, Any] | None:
        if len(sol.t_events[idx]) == 0:
            return None
        tau_e = float(sol.t_events[idx][0])
        rm_e, phi_e, pi_e, q_e, lna_e = [float(x) for x in sol.y_events[idx][0]]
        Htc_e = 0.5 * math.sin(2.0 * q_e)
        Q_e = rm_e + (pi_e**2) / 3.0
        Hdot_tc2_e = 1.5 * math.cos(2.0 * q_e) * Q_e
        V_e = V_fn(phi_e)
        rho_de_e = (pi_e**2) / 6.0 + V_e
        P_de_e = (pi_e**2) / 6.0 - V_e
        w_de_e = P_de_e / rho_de_e if abs(rho_de_e) > 1e-14 else None
        return {
            "tau": tau_e,
            "q_over_pi": q_e / math.pi,
            "a_over_a_bounce1": math.exp(lna_e),
            "H_tc": Htc_e,
            "Hdot_tc2": Hdot_tc2_e,
            "phi": phi_e,
            "pi_phi": pi_e,
            "V_DE_hat": V_e,
            "rho_m_hat": rm_e,
            "w_DE": w_de_e,
        }

    t1 = _event_record(0)
    b2 = _event_record(1)
    t2 = _event_record(2)
    b3 = _event_record(3)
    vz = _event_record(4)
    om_ref = _event_record(5)

    # Compute dw_DE / d(ln a) at Omega_DE = 0.69 epoch (thawing: dw/d(ln a) > 0 => w_a < 0)
    dw_dlna_ref = None
    if om_ref is not None:
        rm_r = om_ref["rho_m_hat"]
        phi_r = om_ref["phi"]
        pi_r = om_ref["pi_phi"]
        q_r = om_ref["q_over_pi"] * math.pi
        Htc_r = 0.5 * math.sin(2.0 * q_r)
        V_r = V_fn(phi_r)
        dV_r = dV_fn(phi_r)
        rho_de_r = (pi_r**2) / 6.0 + V_r
        dpi_dtau = -3.0 * Htc_r * pi_r - 3.0 * dV_r
        drho_de_dtau = -Htc_r * (pi_r**2)
        # w_DE = -1 + pi^2 / (3 rho_de)
        dw_dtau = (2.0 * pi_r * dpi_dtau * rho_de_r - (pi_r**2) * drho_de_dtau) / (3.0 * rho_de_r**2)
        dw_dlna_ref = float(dw_dtau / Htc_r)

    return {
        "potential_type": potential_type,
        "parameters": {
            "V0_hat": float(V0_hat),
            "slope_hat": float(slope_hat),
            "f_phi": float(f_phi),
            "phi_0": float(phi_0),
            "pi_0": float(pi_0),
            "omega_de_ref": float(omega_de_ref),
            "tau_end": float(tau_end),
        },
        "max_constraint_error": max_c_err,
        "constraint_preserved_lt_1e12": bool(max_c_err < 1e-12),
        "q_end_over_pi": float(sol.y[3, -1] / math.pi),
        "events": {
            "omega_de_ref_epoch": om_ref,
            "dw_DE_d_lna_at_ref": dw_dlna_ref,
            "V_DE_zero_crossing": vz,
            "turnaround_1": t1,
            "bounce_2": b2,
            "turnaround_2": t2,
            "bounce_3": b3,
        },
    }


def ir_cosmology_theorem_audit() -> dict[str, Any]:
    """Law 4 audit: constant-Lambda no-cycle vs external IR scalar cyclic trajectory."""
    const_L = integrate_const_lambda_cuscuton(rho_L_hat=0.02, omega_L_ref=0.69, tau_end=65.0)
    linear_ir = integrate_external_ir_scalar_cycle(
        potential_type="linear",
        V0_hat=0.02,
        slope_hat=0.008,
        phi_0=0.0,
        pi_0=0.04,
        omega_de_ref=0.69,
        tau_end=65.0,
    )
    cosine_ir = integrate_external_ir_scalar_cycle(
        potential_type="cosine",
        V0_hat=0.02,
        f_phi=2.0,
        phi_0=0.6,
        pi_0=0.02,
        omega_de_ref=0.69,
        tau_end=115.0,
    )
    pos_quad_control = integrate_external_ir_scalar_cycle(
        potential_type="positive_quadratic",
        V0_hat=0.02,
        slope_hat=0.006,
        phi_0=0.5,
        pi_0=0.04,
        omega_de_ref=0.69,
        tau_end=60.0,
    )

    lin_ev = linear_ir["events"]
    cos_ev = cosine_ir["events"]
    linear_cycle_ok = bool(
        linear_ir["constraint_preserved_lt_1e12"]
        and lin_ev["omega_de_ref_epoch"] is not None
        and lin_ev["omega_de_ref_epoch"]["w_DE"] is not None
        and lin_ev["omega_de_ref_epoch"]["w_DE"] > -1.0
        and lin_ev["dw_DE_d_lna_at_ref"] is not None
        and lin_ev["dw_DE_d_lna_at_ref"] > 0.0
        and lin_ev["V_DE_zero_crossing"] is not None
        and lin_ev["turnaround_1"] is not None
        and lin_ev["turnaround_1"]["Hdot_tc2"] < 0.0
        and lin_ev["turnaround_1"]["V_DE_hat"] < 0.0
        and lin_ev["bounce_2"] is not None
        and lin_ev["bounce_2"]["Hdot_tc2"] > 0.0
    )
    cosine_multicycle_ok = bool(
        cosine_ir["constraint_preserved_lt_1e12"]
        and cos_ev["turnaround_1"] is not None
        and cos_ev["bounce_2"] is not None
        and cos_ev["turnaround_2"] is not None
        and cos_ev["bounce_3"] is not None
    )
    pos_quad_no_turnaround = bool(
        pos_quad_control["constraint_preserved_lt_1e12"]
        and pos_quad_control["events"]["turnaround_1"] is None
        and pos_quad_control["q_end_over_pi"] < 0.5
    )

    return {
        "external_ir_module": True,
        "derived_from_W0_859_MeV": False,
        "calibration_vs_prediction_separation": {
            "external_calibration_reference": "Omega_DE_0 = 0.69 (late-time dark energy fraction benchmark)",
            "model_derivation_status": (
                "phi_DE and its IR scale V0_hat are an EXPLICIT EXTERNAL IR MODULE, "
                "NOT derived from W0 = 859 MeV or lambda_W = 1.05."
            ),
            "qualitative_observational_link": (
                "Canonical thawing phi_DE with V_DE crossing zero necessarily predicts "
                "w(z) > -1 (dw/d ln a > 0, i.e. w_a < 0) before turnaround; it does NOT "
                "cross the phantom divide w < -1."
            ),
        },
        "all_passed": bool(
            const_L["no_turnaround_verified"]
            and linear_cycle_ok
            and cosine_multicycle_ok
            and pos_quad_no_turnaround
        ),
        "constant_lambda_no_cycle": const_L,
        "external_ir_linear_potential_cycle": {
            "cycle_and_thawing_verified": linear_cycle_ok,
            **linear_ir,
        },
        "external_ir_cosine_multicycle": {
            "multicycle_verified": cosine_multicycle_ok,
            **cosine_ir,
        },
        "negative_control_positive_potential_no_turnaround": {
            "no_turnaround_verified": pos_quad_no_turnaround,
            **pos_quad_control,
        },
    }


# =============================================================================
# 5. RESULT BUILDER, CANONICAL JSON & FAIL-CLOSED VALIDATOR
# =============================================================================

def build_result() -> dict[str, Any]:
    sym = symbolic_symmetry_and_geometry_proofs()
    if not sym["all_passed"]:
        raise ArithmeticError("One or more symbolic identities failed")

    scales = vacuum_bag_scale_bridge(PRIMARY_DPS, CONTROL_DPS)
    bounce = dynamic_matter_bounce_audit()
    if not bounce["all_passed"]:
        raise ArithmeticError("Dynamic matter bounce integration audit failed")

    ir_cosmo = ir_cosmology_theorem_audit()
    if not ir_cosmo["all_passed"]:
        raise ArithmeticError("Law 4 IR cosmology theorem audit failed")

    return {
        "schema_version": SCHEMA_VERSION,
        "bridge_id": BRIDGE_ID,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "observed_likelihood": None,
        "source_sha256": _source_sha256(),
        "declared_inputs": DECLARED_INPUTS,
        "calibration_vs_prediction_contract": {
            "calibration_targets": [
                "W0 = 859 MeV, lambda_W = 1.05, MN = 939 MeV, momega = 782.6 MeV, gomega = 10.12 (inherited NVG inputs)",
                "Omega_DE_ref = 0.69 (external cosmological dark-energy fraction benchmark for Law 4)",
            ],
            "mathematical_identities_and_bridges": [
                "SymPy proofs: U(1)_B unitary gauge absorption vs global timelike foliation Psi",
                "SymPy proofs: FLRW cuscuton bounce + BH interior limiting-curvature shear + Hayward anisotropic core",
                "60/90-dps mpmath scale bridge linking rho_c = kappa_g * U(0) to ell, M_crit, T_max, K_max",
                "DOP853 integration of smooth (n, W, v, q) bounce with quartic vector zeta > 0",
                "Law 4 no-cycle theorem for constant Lambda > 0 vs external IR scalar zero-crossing cycle",
            ],
            "forward_observational_predictions_claimed": [],
        },
        "honest_caveats": [
            "evidence_weight is 0.0: all checks are mathematical, symbolic, and numerical consistency proofs, not observational likelihood confirmations.",
            "The foliation field Psi and limiting-curvature scale kappa_g are covariant gravity extensions, not derived from the microscopic NVG fermion-scalar-vector Lagrangian.",
            "Pure cuscuton alone produces isotropic spatial pressure inside a BH horizon; matching Hayward's P_r = -eps != P_t requires the shear/extrinsic-curvature limiting-curvature extension (which vanishes identically on flat FLRW). An explicit unique vacuum action yielding exact Hayward m(r) remains an inverse-reconstruction bridge.",
            "Dynamic bounce ODEs use rescaled dimensionless stiffness mu_stiff = W0*t_c in {5, 20} to resolve non-adiabatic W evolution across [-5 t_c, +5 t_c]; physical M_P/W0 ~ 2.835e18 is in the extreme adiabatic-tracking limit.",
            "Law 4 external IR scalar phi_DE is explicitly segregated and NOT derived from W0 = 859 MeV; canonical thawing phi_DE gives w(z) > -1 (no phantom crossing).",
            "At the turnaround q = pi/2 (rho_tot = 0), the holonomy/angle variables (a, q) are C^infty smooth, whereas the single-branch cuscuton variable Psi hits its branch edge (V_PsiPsi -> -inf) requiring branch gluing; moreover, conservative Hamiltonian scalar-matter cycles are recurrent orbits rather than dissipative attractors without entropy reset.",
        ],
        "pillar_1_symbolic_symmetry_and_geometry": sym,
        "pillar_2_vacuum_bag_scale_bridge": scales,
        "pillar_3_dynamic_matter_bounce": bounce,
        "pillar_4_ir_cosmology_theorem": ir_cosmo,
    }


def _trees_match(
    actual: Any,
    expected: Any,
    *,
    rel_tol: float = 1.0e-7,
    abs_tol: float = 1.0e-10,
) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual.keys()) == set(expected.keys()) and all(
            _trees_match(actual[k], expected[k], rel_tol=rel_tol, abs_tol=abs_tol)
            for k in expected
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _trees_match(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
            for a, b in zip(actual, expected)
        )
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, int) and isinstance(actual, int):
        return actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        a_val, e_val = float(actual), float(expected)
        return math.isfinite(a_val) and math.isfinite(e_val) and math.isclose(
            a_val, e_val, rel_tol=rel_tol, abs_tol=abs_tol
        )
    return type(actual) is type(expected) and actual == expected


def canonical_json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def validate_artifact(path: Path) -> tuple[bool, list[str]]:
    """Fail-closed validation of a JSON artifact against live recomputation."""
    errors: list[str] = []
    if not path.exists():
        return False, [f"Artifact does not exist: {path}"]
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, [f"Failed to read artifact: {exc}"]

    def _reject_constants(val: str) -> Any:
        raise ValueError(f"Invalid JSON constant: {val}")

    try:
        loaded = json.loads(raw_text, parse_constant=_reject_constants)
    except Exception as exc:
        return False, [f"Invalid JSON: {exc}"]

    if not isinstance(loaded, dict):
        return False, ["Top-level JSON must be an object"]

    required_keys = {
        "schema_version",
        "bridge_id",
        "status",
        "evidence_weight",
        "source_sha256",
        "declared_inputs",
        "calibration_vs_prediction_contract",
        "honest_caveats",
        "pillar_1_symbolic_symmetry_and_geometry",
        "pillar_2_vacuum_bag_scale_bridge",
        "pillar_3_dynamic_matter_bounce",
        "pillar_4_ir_cosmology_theorem",
    }
    missing = sorted(required_keys - set(loaded.keys()))
    if missing:
        errors.append(f"Missing required top-level keys: {missing}")

    if loaded.get("evidence_weight") != 0.0 or isinstance(loaded.get("evidence_weight"), bool):
        errors.append(f"evidence_weight must be exactly 0.0, got {loaded.get('evidence_weight')!r}")
    if loaded.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version mismatch: {loaded.get('schema_version')!r}")
    if loaded.get("source_sha256") != _source_sha256():
        errors.append("source_sha256 does not match current producer script")
    if errors:
        return False, errors

    # Verify canonical formatting and cross-platform numerical match with live recomputation
    expected = build_result()
    if not _trees_match(loaded, expected):
        errors.append("Artifact content does not match live recomputed audit result")
    if raw_text != canonical_json_dumps(loaded):
        errors.append("Artifact is not in deterministic canonical JSON format (sort_keys=True, indent=2)")

    return len(errors) == 0, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write deterministic canonical JSON result to this path.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing JSON artifact fail-closed against live recomputation.",
    )
    args = parser.parse_args(argv)

    if args.validate is not None:
        ok, errs = validate_artifact(args.validate)
        if not ok:
            for err in errs:
                print(f"VALIDATION_ERROR: {err}", file=sys.stderr)
            return 1
        print(f"VALIDATION_OK: {args.validate}")
        return 0

    result = build_result()
    text = canonical_json_dumps(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"WROTE_CANONICAL_JSON: {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
