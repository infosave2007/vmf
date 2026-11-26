#!/usr/bin/env python3
"""Self-consistent spherical relativistic Dirac-Hartree finite-nuclei bridge (Law 3A).

Implements a fast, self-contained, spurious-free spherical relativistic
Dirac-Hartree solver for doubly-magic closed-shell nuclei (O16, Ca40, Ca48,
Pb208) using the two-condensate Lagrangian structure of Law 2:

  * Heavy gluon condensate W(r) with quartic vacuum potential
        U_W(W) = (lam_W / 4) (W^2 - W0^2)^2,   m_W = sqrt(2 lam_W) W0
  * Light chiral scalar field s(r) with bounded-below nonlinear self-interaction
        U_s(s) = (1/2) m_s^2 s^2 + (1/3) b_s s^3 + (1/4) c_s s^4 + (1/6) d_s s^6
    where b_s = b3 M_N g_s^3, c_s = b4 g_s^4, d_s = b6 g_s^6 (globally bounded
    below: either d_s > 0 with q_min > 0, or d_s = 0 and c_s > 0)
  * Effective nucleon mass
        M_eff(r) = M_N * (W(r) / W0) - g_s * s(r)
  * Vector omega field A0(r) with covariant Higgs mass m_A(W) = q_phi W
    (q_phi = m_omega / W0) and quartic self-interaction zeta:
        (1/2) q_phi^2 W^2 A0^2 + (zeta / 24) g_omega^4 A0^4
  * Finite-range isovector rho0(r) with omega-rho mixing Lambda_v:
        (1/2) m_rho^2 rho0^2 + Lambda_v g_omega^2 g_rho^2 A0^2 rho0^2
  * Electromagnetic Coulomb potential V_C(r) = e A_C(r)

Numerical method:
  * Radial Dirac equations are discretized via a QR-orthonormalized
    kinetic-balanced polynomial spectral Galerkin basis on [0, R_max] with
    exact r^(l+1) origin regularity, reducing the generalized eigenproblem to
    a well-conditioned standard symmetric eigenproblem with zero spurious
    states inside the mass gap (-M, +M).
  * The 5 coupled radial Helmholtz/Poisson equations are solved via exact
    Newton-Raphson on an even-Chebyshev collocation grid (automatically
    enforcing f'(0) = 0) and driven to self-consistency with Anderson
    acceleration (under-relaxation fallback) to residual < 1e-6 (typically
    < 1e-10).

Scientific integrity (SPEC-v2):
  * Calibration targets are strictly bulk nuclear matter saturation properties
    (n0, B/A, K, J, L) — NEVER fitted to finite nuclei — or frozen parameters
    supplied via ``--params-json`` with zero refitting.
  * All finite-nucleus observables are forward predictions; comparisons with
    experiment and earlier W8 Thomas-Fermi failures are descriptive only.
  * ``evidence_weight`` is 0.0 (mathematical/in-sample/forward-model audit).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Avoid BLAS oversubscription on small dense eigensolves and linear systems.
for _blas_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import numpy as np
from scipy.linalg import eigh

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
DEFAULT_OUTPUT_PATH = HERE / "nvg_unified_dirac_hartree_results.json"

SCHEMA_VERSION = "nvg_unified_dirac_hartree_probe.v1"
STATUS = "COMPUTED_TWO_CONDENSATE_DIRAC_HARTREE_BRIDGE_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

HBARC_MEV_FM = 197.3269804
ALPHA_EM = 1.0 / 137.035999084
PROTON_CHARGE_RADIUS_FM = 0.8414

DEFAULT_FIXED_INPUTS: dict[str, float] = {
    "M_N": 939.0,
    "W0": 859.0,
    "lam_W": 1.05,
    "m_s": 500.0,
    "m_omega": 782.6,
    "m_rho": 775.3,
    "g_omega": 13.0,
    "zeta": 0.04,
}

DEFAULT_SNM_CALIBRATION_TARGETS: dict[str, float] = {
    "n0_fm3": 0.16,
    "B_over_A_MeV": -16.0,
    "K_MeV": 240.0,
    "J_MeV": 31.0,
    "L_MeV": 68.0,
}

DOUBLY_MAGIC_NUCLEI: tuple[tuple[str, int, int], ...] = (
    ("O16", 8, 8),
    ("Ca40", 20, 20),
    ("Ca48", 20, 28),
    ("Pb208", 82, 126),
)

# Cited experimental values and prior W8 Thomas-Fermi benchmarks for descriptive
# forward comparison ONLY (never used for fitting).
REFERENCE_COMPARISONS: dict[str, Any] = {
    "role": "descriptive_heldout_comparison_only_never_fitted",
    "experimental": {
        "O16": {
            "B_over_A_MeV": -7.9762,
            "R_ch_fm": 2.6991,
            "R_ch_sigma_fm": 0.0052,
            "SO_1p_MeV_approx": 6.18,
            "source": "AME2020; Angeli & Marinova ADNDT 99 (2013) 69",
        },
        "Ca40": {
            "B_over_A_MeV": -8.5513,
            "R_ch_fm": 3.4776,
            "R_ch_sigma_fm": 0.0019,
            "SO_1d_MeV_approx": 6.0,
            "source": "AME2020; Angeli & Marinova ADNDT 99 (2013) 69",
        },
        "Ca48": {
            "B_over_A_MeV": -8.6667,
            "R_ch_fm": 3.4771,
            "R_ch_sigma_fm": 0.0020,
            "Delta_R_np_fm": 0.121,
            "Delta_R_np_exp_sigma_fm": 0.026,
            "Delta_R_np_model_sigma_fm": 0.024,
            "Delta_R_ch_Ca48_minus_Ca40_fm": -0.0005,
            "SO_1f_MeV_approx": 8.4,
            "source": "AME2020; Angeli & Marinova (2013); CREX PRL 129, 042501 (2022)",
        },
        "Pb208": {
            "B_over_A_MeV": -7.8675,
            "R_ch_fm": 5.5012,
            "R_ch_sigma_fm": 0.0013,
            "Delta_R_np_PREX2_fm": 0.283,
            "Delta_R_np_PREX2_sigma_fm": 0.071,
            "source": "AME2020; Angeli & Marinova (2013); PREX-II PRL 126, 172502 (2021)",
        },
    },
    "prior_w8_thomas_fermi_failures": {
        "Delta_R_ch_Ca48_minus_Ca40_fm": 0.166,
        "Delta_R_np_Pb208_fm": -0.026,
        "note": "Prior single-condensate W8 Thomas-Fermi closure suffered inverted Pb208 skin (-0.026 fm) and 160x inflated Ca48-Ca40 charge-radius shift (+0.166 fm).",
    },
}

FIELD_RESIDUAL_LIMIT = 1.0e-6
PARTICLE_NUMBER_REL_LIMIT = 1.0e-6
ORBITAL_NORM_LIMIT = 1.0e-8
ENERGY_IDENTITY_PER_A_LIMIT_MEV = 1.0e-3
GRID_REFINEMENT_BA_LIMIT_MEV = 1.0e-3
GRID_REFINEMENT_RADIUS_LIMIT_FM = 1.0e-3

SPECTROSCOPIC_L = ("s", "p", "d", "f", "g", "h", "i", "j", "k")


class DiracHartreeBridgeError(ValueError):
    """Fail-closed error for physical, numerical, or schema violations."""


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise DiracHartreeBridgeError(f"{name} must be a finite number, not bool")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DiracHartreeBridgeError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(out):
        raise DiracHartreeBridgeError(f"{name} is not finite: {value!r}")
    return out


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _orbital_l_from_kappa(kappa: int) -> int:
    if kappa == 0:
        raise DiracHartreeBridgeError("Dirac quantum number kappa cannot be 0")
    return kappa if kappa > 0 else -kappa - 1


def _shell_label(kappa: int, n_radial: int) -> str:
    l = _orbital_l_from_kappa(kappa)
    l_char = SPECTROSCOPIC_L[l] if l < len(SPECTROSCOPIC_L) else f"l{l}"
    two_j = 2 * abs(kappa) - 1
    return f"{n_radial + 1}{l_char}{two_j}/2"


# ---------------------------------------------------------------------------
# Uniform Nuclear Matter Calibration & Independent Verification (Law 2 EOS)
# ---------------------------------------------------------------------------


def _fermi_snm(n_nat: float, m_star: float) -> dict[str, float]:
    """Relativistic symmetric nuclear matter Fermi integrals (natural units MeV)."""
    if n_nat <= 0.0 or m_star <= 0.0:
        raise DiracHartreeBridgeError("n_nat and m_star must be positive")
    k_f = (1.5 * math.pi * math.pi * n_nat) ** (1.0 / 3.0)
    e_f = math.hypot(k_f, m_star)
    asinh_km = math.asinh(k_f / m_star)
    eps_f = (k_f * e_f * (2.0 * k_f * k_f + m_star * m_star) - (m_star**4) * asinh_km) / (
        4.0 * math.pi * math.pi
    )
    p_f = (k_f * e_f * (2.0 * k_f * k_f - 3.0 * m_star * m_star) + 3.0 * (m_star**4) * asinh_km) / (
        12.0 * math.pi * math.pi
    )
    rho_s = m_star * (k_f * e_f - m_star * m_star * asinh_km) / (math.pi * math.pi)
    drho_s_dm = 3.0 * rho_s / m_star - 3.0 * n_nat / e_f
    return {
        "k_F": k_f,
        "E_F": e_f,
        "eps_F": eps_f,
        "P_F": p_f,
        "rho_s": rho_s,
        "drho_s_dM": drho_s_dm,
    }


def _solve_positive_cubic(c3: float, c1: float, c0: float) -> float:
    """Solve c3 * x^3 + c1 * x + c0 = 0 for the unique positive real root (c3>=0, c1>0, c0<0)."""
    if c3 == 0.0:
        return -c0 / c1
    x = -c0 / c1
    for _ in range(40):
        f = c3 * x**3 + c1 * x + c0
        df = 3.0 * c3 * x**2 + c1
        step = f / df
        x -= step
        if abs(step) <= 1.0e-14 * max(1.0, abs(x)):
            break
    if not (math.isfinite(x) and x > 0.0):
        raise DiracHartreeBridgeError("failed to find positive root of vector cubic equation")
    return x


def _check_chiral_potential_bounded_below(
    m_s: float, b_s: float, c_s: float, d_s: float
) -> dict[str, Any]:
    """Verify exact global boundedness U_s(s) >= 0 and unique vacuum at s = 0."""
    m_s2 = m_s * m_s
    if d_s < 0.0:
        raise DiracHartreeBridgeError(
            f"chiral scalar potential is not bounded-below (sextic d_s={d_s:.4e} < 0)"
        )
    if d_s == 0.0:
        if c_s <= 0.0:
            raise DiracHartreeBridgeError(
                f"chiral scalar potential is not bounded-below (d_s=0 and c_s={c_s:.4e} <= 0)"
            )
        crit_real = [float(-(b_s / 3.0) / (0.5 * c_s))]
    else:
        roots = np.roots([2.0 * d_s / 3.0, 0.0, 0.5 * c_s, b_s / 3.0])
        crit_real = [float(r.real) for r in roots if abs(r.imag) < 1.0e-8]

    def q_val(x: float) -> float:
        return 0.5 * m_s2 + (b_s / 3.0) * x + 0.25 * c_s * (x**2) + (d_s / 6.0) * (x**4)

    q_candidates = [(0.0, q_val(0.0))] + [(x, q_val(x)) for x in crit_real]
    q_min_s, q_min = min(q_candidates, key=lambda pair: pair[1])
    if q_min <= 0.0:
        raise DiracHartreeBridgeError(
            f"chiral scalar potential is not bounded-below / violates unique vacuum (q_min={q_min:.4e} <= 0 at s={q_min_s:.4f})"
        )
    return {
        "bounded_below": True,
        "q_min_MeV2": float(q_min),
        "q_min_at_s_MeV": float(q_min_s),
    }


def _solve_uniform_snm_fields(n_nat: float, params: Mapping[str, float]) -> dict[str, float]:
    """Solve uniform SNM field equations (W, s, A0) for arbitrary density n_nat."""
    m_n = float(params["M_N"])
    w0 = float(params["W0"])
    lam_w = float(params["lam_W"])
    m_s = float(params["m_s"])
    m_om = float(params["m_omega"])
    m_rho = float(params["m_rho"])
    g_s = float(params["g_s"])
    b_s = float(params["b_s"])
    c_s = float(params["c_s"])
    d_s = float(params.get("d_s", 0.0))
    g_om = float(params["g_omega"])
    zeta = float(params["zeta"])
    g_rho = float(params["g_rho"])
    lam_v = float(params["Lambda_v"])
    q_phi = float(params.get("q_phi", m_om / w0))
    g_w = m_n / w0
    c_z = (zeta / 6.0) * (g_om**4)

    # 3D Newton for coupled (W, s, A0) given n_nat
    w = w0 * 0.999
    s = 0.35 * m_n / g_s
    a0 = _solve_positive_cubic(c_z, (q_phi * w) ** 2, -g_om * n_nat)
    for _ in range(60):
        m_eff = g_w * w - g_s * s
        if m_eff <= 1.0:
            s *= 0.8
            m_eff = max(g_w * w - g_s * s, 50.0)
        fm = _fermi_snm(n_nat, m_eff)
        rho_s = fm["rho_s"]
        dr_dm = fm["drho_s_dM"]
        f1 = lam_w * w * (w * w - w0 * w0) + g_w * rho_s - (q_phi**2) * w * (a0**2)
        f2 = m_s * m_s * s + b_s * s * s + c_s * (s**3) + d_s * (s**5) - g_s * rho_s
        f3 = (q_phi**2) * (w**2) * a0 + c_z * (a0**3) - g_om * n_nat
        if max(abs(f1), abs(f2), abs(f3)) < 1.0e-10:
            break
        j_mat = np.array(
            [
                [
                    lam_w * (3.0 * w * w - w0 * w0) - (q_phi**2) * (a0**2) + (g_w**2) * dr_dm,
                    -g_w * g_s * dr_dm,
                    -2.0 * (q_phi**2) * w * a0,
                ],
                [
                    -g_s * g_w * dr_dm,
                    m_s * m_s + 2.0 * b_s * s + 3.0 * c_s * (s**2) + 5.0 * d_s * (s**4) + (g_s**2) * dr_dm,
                    0.0,
                ],
                [
                    2.0 * (q_phi**2) * w * a0,
                    0.0,
                    (q_phi**2) * (w**2) + 3.0 * c_z * (a0**2),
                ],
            ],
            dtype=float,
        )
        delta = np.linalg.solve(j_mat, np.array([f1, f2, f3], dtype=float))
        w -= float(delta[0])
        s -= float(delta[1])
        a0 -= float(delta[2])
    m_eff = g_w * w - g_s * s
    fm = _fermi_snm(n_nat, m_eff)
    u_w = 0.25 * lam_w * (w * w - w0 * w0) ** 2
    u_s = (
        0.5 * m_s * m_s * s * s
        + (b_s / 3.0) * (s**3)
        + 0.25 * c_s * (s**4)
        + (d_s / 6.0) * (s**6)
    )
    t_a2 = 0.5 * (q_phi**2) * (w**2) * (a0**2)
    eps_tot = fm["eps_F"] + u_w + u_s + t_a2 + (zeta / 8.0) * (g_om**4) * (a0**4)
    p_tot = fm["P_F"] - u_w - u_s + t_a2 + (zeta / 24.0) * (g_om**4) * (a0**4)
    mu = fm["E_F"] + g_om * a0
    m_rho_eff2 = m_rho * m_rho + 2.0 * lam_v * (g_om**2) * (g_rho**2) * (a0**2)
    sym_e = (fm["k_F"] ** 2) / (6.0 * fm["E_F"]) + (g_rho * g_rho * n_nat) / (8.0 * m_rho_eff2)
    return {
        "n_nat": n_nat,
        "W": w,
        "s": s,
        "A0": a0,
        "M_eff": m_eff,
        "eps_tot": eps_tot,
        "P_tot": p_tot,
        "mu": mu,
        "B_over_A": eps_tot / n_nat - m_n,
        "S_sym": sym_e,
        "k_F": fm["k_F"],
        "E_F": fm["E_F"],
        "rho_s": fm["rho_s"],
    }


def verify_uniform_snm(params: Mapping[str, float], n0_fm3: float = 0.16) -> dict[str, float]:
    """Independently evaluate saturation properties (P, B/A, K, J, L) via centered finite differences."""
    hc3 = HBARC_MEV_FM**3
    n0 = float(n0_fm3) * hc3
    h = 1.0e-4 * n0
    st0 = _solve_uniform_snm_fields(n0, params)
    stp = _solve_uniform_snm_fields(n0 + h, params)
    stm = _solve_uniform_snm_fields(n0 - h, params)
    dmu_dn = (stp["mu"] - stm["mu"]) / (2.0 * h)
    k_mev = 9.0 * n0 * dmu_dn
    ds_dn = (stp["S_sym"] - stm["S_sym"]) / (2.0 * h)
    l_mev = 3.0 * n0 * ds_dn
    return {
        "n0_fm3": float(n0_fm3),
        "P_MeV_fm3": st0["P_tot"] / hc3,
        "B_over_A_MeV": st0["B_over_A"],
        "K_MeV": k_mev,
        "J_MeV": st0["S_sym"],
        "L_MeV": l_mev,
        "M_eff_over_MN": st0["M_eff"] / float(params["M_N"]),
        "W_over_W0": st0["W"] / float(params["W0"]),
        "s_MeV": st0["s"],
        "A0_MeV": st0["A0"],
    }


def calibrate_two_condensate_from_snm(
    fixed: Mapping[str, float] | None = None,
    targets: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Analytically calibrate (g_s, b_s, c_s, g_rho, Lambda_v) strictly to bulk SNM targets.

    Zero finite-nucleus data enters this calibration.
    """
    fx = dict(DEFAULT_FIXED_INPUTS)
    if fixed is not None:
        fx.update({str(k): _require_finite_float(v, str(k)) for k, v in fixed.items()})
    tg = dict(DEFAULT_SNM_CALIBRATION_TARGETS)
    if targets is not None:
        tg.update({str(k): _require_finite_float(v, str(k)) for k, v in targets.items()})

    m_n = fx["M_N"]
    w0 = fx["W0"]
    lam_w = fx["lam_W"]
    m_w = math.sqrt(2.0 * lam_w) * w0
    m_s = fx["m_s"]
    m_om = fx["m_omega"]
    m_rho = fx["m_rho"]
    g_om = fx["g_omega"]
    zeta = fx["zeta"]

    n0_fm3 = tg["n0_fm3"]
    ba_t = tg["B_over_A_MeV"]
    k_t = tg["K_MeV"]
    j_t = tg["J_MeV"]
    l_t = tg["L_MeV"]

    n0 = n0_fm3 * (HBARC_MEV_FM**3)
    mu0 = m_n + ba_t
    k_f = (1.5 * math.pi * math.pi * n0) ** (1.0 / 3.0)
    q_phi = m_om / w0
    g_w = m_n / w0
    c_z = (zeta / 6.0) * (g_om**4)

    # Solve coupled (W_n0, A0) with exact covariant Higgs vector mass m_A(W) = q_phi * W
    w_n0 = w0 * 0.999
    a0 = _solve_positive_cubic(c_z, (q_phi * w_n0) ** 2, -g_om * n0)
    for _ in range(40):
        e_f = mu0 - g_om * a0
        if e_f <= k_f:
            raise DiracHartreeBridgeError("vector potential too large for real effective mass at n0")
        m_star = math.sqrt(e_f * e_f - k_f * k_f)
        fm = _fermi_snm(n0, m_star)
        rho_s = fm["rho_s"]
        dr_dm = fm["drho_s_dM"]
        f_w = lam_w * w_n0 * (w_n0 * w_n0 - w0 * w0) + g_w * rho_s - (q_phi**2) * w_n0 * (a0**2)
        f_a = (q_phi**2) * (w_n0**2) * a0 + c_z * (a0**3) - g_om * n0
        if max(abs(f_w), abs(f_a)) < 1.0e-11:
            break
        dm_da0 = -g_om * e_f / m_star
        jac_wa = np.array(
            [
                [
                    lam_w * (3.0 * w_n0 * w_n0 - w0 * w0) - (q_phi**2) * (a0**2),
                    g_w * dr_dm * dm_da0 - 2.0 * (q_phi**2) * w_n0 * a0,
                ],
                [
                    2.0 * (q_phi**2) * w_n0 * a0,
                    (q_phi**2) * (w_n0**2) + 3.0 * c_z * (a0**2),
                ],
            ],
            dtype=float,
        )
        dwa = np.linalg.solve(jac_wa, np.array([f_w, f_a], dtype=float))
        w_n0 -= float(dwa[0])
        a0 -= float(dwa[1])

    e_f = mu0 - g_om * a0
    m_star = math.sqrt(e_f * e_f - k_f * k_f)
    fm = _fermi_snm(n0, m_star)
    rho_s = fm["rho_s"]
    dr_dm = fm["drho_s_dM"]
    u_w = 0.25 * lam_w * (w_n0 * w_n0 - w0 * w0) ** 2

    phi0 = g_w * w_n0 - m_star
    if phi0 <= 0.0:
        raise DiracHartreeBridgeError("chiral scalar shift Phi0 must be positive")

    # Pressure = 0 at n0 => U_s(Phi0), scalar EOM => U_s'(Phi0), K => U_s''(Phi0)
    u_s0 = fm["P_F"] + 0.5 * (q_phi**2) * (w_n0**2) * (a0**2) + (zeta / 24.0) * (g_om**4) * (a0**4) - u_w
    u_s_p = rho_s
    rhs_k = k_t / (9.0 * n0) - (k_f * k_f) / (3.0 * n0 * e_f)
    c_aa = (q_phi**2) * (w_n0**2) + 3.0 * c_z * (a0**2)
    c_aw = 2.0 * (q_phi**2) * w_n0 * a0
    c_ww = lam_w * (3.0 * w_n0 * w_n0 - w0 * w0) - (q_phi**2) * (a0**2)
    resp_mat = np.array(
        [
            [c_aw, c_aa, 0.0],
            [c_ww + (g_w**2) * dr_dm, -c_aw, -g_w * dr_dm],
            [(m_star / e_f) * g_w, g_om, -(m_star / e_f)],
        ],
        dtype=float,
    )
    resp_rhs = np.array([g_om, -g_w * (m_star / e_f), rhs_k], dtype=float)
    dw_dn, da0_dn, dphi_dn = [float(v) for v in np.linalg.solve(resp_mat, resp_rhs)]
    dm_dn = g_w * dw_dn - dphi_dn
    if dphi_dn <= 0.0:
        raise DiracHartreeBridgeError("chiral scalar response dPhi/dn is non-positive")
    u_s_pp = (m_star / e_f + dr_dm * dm_dn) / dphi_dn

    mat = np.array(
        [
            [0.5 * phi0**2, (phi0**3) / 3.0, 0.25 * phi0**4],
            [phi0, phi0**2, phi0**3],
            [1.0, 2.0 * phi0, 3.0 * phi0**2],
        ],
        dtype=float,
    )
    inv_c_s, b_tilde, c_tilde = np.linalg.solve(mat, np.array([u_s0, u_s_p, u_s_pp], dtype=float))
    if inv_c_s <= 0.0:
        raise DiracHartreeBridgeError("calibrated scalar coupling g_s^2 <= 0")
    g_s = m_s / math.sqrt(float(inv_c_s))
    b_s = float(b_tilde) * (g_s**3)
    c_s = float(c_tilde) * (g_s**4)
    if c_s <= 0.0:
        raise DiracHartreeBridgeError(
            f"scalar potential is unbounded below (c_s={c_s:.4f} <= 0); increase zeta or adjust g_omega"
        )

    # Check global non-negativity of U_s(s) = s^2 (0.5 m_s^2 + (b_s/3) s + (c_s/4) s^2)
    disc_us = (b_s / 3.0) ** 2 - 0.5 * m_s * m_s * c_s
    globally_nonnegative_us = bool(c_s > 0.0 and disc_us <= 0.0)

    # Isovector calibration to (J, L)
    s_kin = (k_f * k_f) / (6.0 * e_f)
    l_kin = ((k_f * k_f) / (3.0 * e_f)) * (
        1.0 - (k_f * k_f) / (2.0 * e_f * e_f) - 1.5 * n0 * m_star * dm_dn / (e_f * e_f)
    )
    s_pot = j_t - s_kin
    if s_pot <= 0.0:
        raise DiracHartreeBridgeError("target J <= S_kin; no positive g_rho")
    c_rho_eff = 8.0 * s_pot / n0
    lam_v = (1.0 - (l_t - l_kin) / (3.0 * s_pot)) / (
        4.0 * (g_om**2) * c_rho_eff * a0 * n0 * da0_dn
    )
    g_rho_denom = 1.0 - 2.0 * lam_v * (g_om**2) * (a0**2) * c_rho_eff
    if g_rho_denom <= 0.0:
        raise DiracHartreeBridgeError("isovector mixing Lambda_v exceeds physical pole")
    g_rho = math.sqrt((m_rho * m_rho * c_rho_eff) / g_rho_denom)

    params: dict[str, Any] = {
        "M_N": float(m_n),
        "W0": float(w0),
        "lam_W": float(lam_w),
        "m_W": float(m_w),
        "m_s": float(m_s),
        "m_omega": float(m_om),
        "m_rho": float(m_rho),
        "q_phi": float(q_phi),
        "g_s": float(g_s),
        "b_s": float(b_s),
        "c_s": float(c_s),
        "d_s": 0.0,
        "g_omega": float(g_om),
        "zeta": float(zeta),
        "g_rho": float(g_rho),
        "Lambda_v": float(lam_v),
        "bounded_below_scalar": bool(c_s > 0.0),
        "globally_nonnegative_U_s": globally_nonnegative_us,
        "U_s_discriminant_MeV2": float(disc_us),
    }
    check = verify_uniform_snm(params, n0_fm3=n0_fm3)
    if (
        abs(check["P_MeV_fm3"]) > 1.0e-8
        or abs(check["B_over_A_MeV"] - ba_t) > 1.0e-8
        or abs(check["K_MeV"] - k_t) > 1.0e-5
        or abs(check["J_MeV"] - j_t) > 1.0e-8
        or abs(check["L_MeV"] - l_t) > 1.0e-5
    ):
        raise DiracHartreeBridgeError(f"SNM self-calibration verification failed: {check}")
    params["snm_verification"] = check
    return params


def normalize_params(raw: Mapping[str, Any]) -> dict[str, float]:
    """Validate and normalize a Law 2 parameter dictionary (supporting standard aliases)."""
    # Unwrap nested containers if passed a full Law 2 JSON artifact.
    src: Mapping[str, Any] = raw
    for key in ("frozen_parameters", "frozen_params", "parameters", "params"):
        if key in src and isinstance(src[key], Mapping):
            src = src[key]
            break
    if "science" in src and isinstance(src["science"], Mapping):
        for key in ("frozen_parameters", "frozen_params", "parameters", "params"):
            if key in src["science"] and isinstance(src["science"][key], Mapping):
                src = src["science"][key]
                break

    def _pick(canonical: str, aliases: Sequence[str], default: float | None = None) -> float:
        for k in (canonical, *aliases):
            if k in src and src[k] is not None:
                return _require_finite_float(src[k], canonical)
        if default is not None:
            return float(default)
        raise DiracHartreeBridgeError(f"missing required parameter {canonical!r} (aliases {aliases})")

    def _pick_opt(keys: Sequence[str]) -> float | None:
        for k in keys:
            if k in src and src[k] is not None:
                return _require_finite_float(src[k], k)
        return None

    m_n = _pick("M_N", ("MN", "M_N_MeV", "m_N"), 939.0)
    w0 = _pick("W0", ("W0_MeV",), 859.0)
    lam_w = _pick("lam_W", ("lambda_W", "lamW"), 1.05)
    m_w_default = math.sqrt(2.0 * lam_w) * w0 if lam_w > 0.0 else None
    m_w = _pick("m_W", ("m_W_MeV", "mW"), m_w_default)
    m_s = _pick("m_s", ("m_s_MeV", "ms", "m_sigma"), 500.0)
    m_om = _pick("m_omega", ("m_om", "m_omega_MeV", "momega"), 782.6)
    m_rho = _pick("m_rho", ("m_rho_MeV", "mrho"), 775.3)
    q_phi = _pick("q_phi", (), m_om / w0 if w0 > 0.0 else None)
    g_s = _pick("g_s", ("g_s_ch", "gs", "g_sigma"))

    # Support both direct s-field coefficients (b_s, c_s, d_s) and Law 2 dimensionless (b3, b4, b6)
    b_s_raw = _pick_opt(("b_s", "bs", "b_s_MeV", "g2", "kappa_s"))
    if b_s_raw is not None:
        b_s = b_s_raw
    else:
        b3 = _pick("b3", ())
        b_s = b3 * m_n * (g_s**3)

    c_s_raw = _pick_opt(("c_s", "cs", "g3", "lambda_s"))
    if c_s_raw is not None:
        c_s = c_s_raw
    else:
        b4 = _pick("b4", ())
        c_s = b4 * (g_s**4)

    d_s_raw = _pick_opt(("d_s", "ds", "d_s_MeV_minus2"))
    if d_s_raw is not None:
        d_s = d_s_raw
    else:
        b6 = _pick_opt(("b6_MeV_minus2", "b6"))
        d_s = 0.0 if b6 is None else b6 * (g_s**6)

    g_om = _pick("g_omega", ("g_om", "gomega"))
    zeta = _pick("zeta", ("zeta_RMF", "zeta_rmf", "zeta_omega"), 0.0)
    g_rho = _pick("g_rho", ("grho",))
    lam_v = _pick("Lambda_v", ("Lam_v", "lambda_v", "Lambda_V"), 0.0)

    if any(x <= 0.0 for x in (m_n, w0, lam_w, m_w, m_s, m_om, m_rho, q_phi, g_s, g_om, g_rho)):
        raise DiracHartreeBridgeError("masses, W0, lam_W, q_phi, and base couplings must be strictly positive")
    _check_chiral_potential_bounded_below(m_s, b_s, c_s, d_s)
    if zeta < 0.0 or lam_v < 0.0:
        raise DiracHartreeBridgeError(f"zeta ({zeta}) and Lambda_v ({lam_v}) must be non-negative")

    return {
        "M_N": m_n,
        "W0": w0,
        "lam_W": lam_w,
        "m_W": m_w,
        "m_s": m_s,
        "m_omega": m_om,
        "m_rho": m_rho,
        "q_phi": q_phi,
        "g_s": g_s,
        "b_s": b_s,
        "c_s": c_s,
        "d_s": d_s,
        "b3": b_s / (m_n * (g_s**3)),
        "b4": c_s / (g_s**4),
        "b6_MeV_minus2": d_s / (g_s**6),
        "g_omega": g_om,
        "zeta": zeta,
        "g_rho": g_rho,
        "Lambda_v": lam_v,
    }


def load_params_from_json(path: Path) -> dict[str, float]:
    """Load and validate frozen two-condensate parameters from a JSON file (zero refitting)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiracHartreeBridgeError(f"failed to load params JSON {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise DiracHartreeBridgeError("params JSON root must be an object")
    return normalize_params(raw)


# ---------------------------------------------------------------------------
# Spectral Discretization: Even-Chebyshev Fields + Kinetic-Balanced Dirac
# ---------------------------------------------------------------------------


class EvenChebyshevGrid:
    """Half of a symmetric Chebyshev-Lobatto grid on [-R_max, R_max].

    Represents smooth even radial functions f(-r) = f(r) on [0, R_max],
    automatically enforcing f'(0) = 0 and regularizing the radial Laplacian
    at r = 0 via L'Hopital: (nabla^2 f)(0) = 3 f''(0).
    """

    def __init__(self, r_max: float, m_intervals: int = 84) -> None:
        if r_max <= 0.0 or m_intervals < 16:
            raise DiracHartreeBridgeError("invalid EvenChebyshevGrid parameters")
        self.R_max = float(r_max)
        self.M = int(m_intervals)
        n2 = 2 * self.M
        k = np.arange(n2 + 1)
        x_full = np.cos(math.pi * k / n2)

        c = np.ones(n2 + 1)
        c[0] = 2.0
        c[-1] = 2.0
        c = c * ((-1.0) ** k)
        x_col = np.tile(x_full, (n2 + 1, 1)).T
        d_full = np.outer(c, 1.0 / c) / ((x_col - x_col.T) + np.eye(n2 + 1))
        d_full -= np.diag(np.sum(d_full, axis=1))
        d2_full = d_full @ d_full

        n_pts = self.M + 1
        d1_even = np.zeros((n_pts, n_pts), dtype=float)
        d2_even = np.zeros((n_pts, n_pts), dtype=float)
        for j in range(n_pts):
            if j == self.M:
                d1_even[:, j] = d_full[:n_pts, j]
                d2_even[:, j] = d2_full[:n_pts, j]
            else:
                d1_even[:, j] = d_full[:n_pts, j] + d_full[:n_pts, n2 - j]
                d2_even[:, j] = d2_full[:n_pts, j] + d2_full[:n_pts, n2 - j]

        self.t = x_full[:n_pts][::-1].copy()
        self.t[0] = 0.0
        self.t[-1] = 1.0
        self.r = self.R_max * self.t
        self.D1 = d1_even[::-1, ::-1] / self.R_max
        self.D2 = d2_even[::-1, ::-1] / (self.R_max * self.R_max)
        self.L = self.D2.copy()
        self.L[0, :] = 3.0 * self.D2[0, :]
        self.L[1:, :] += (2.0 / self.r[1:, None]) * self.D1[1:, :]

        # Spectral Clenshaw-Curtis weights for integrating even functions on [0, R_max].
        theta = math.pi * np.arange(n2 + 1) / n2
        w_full = np.zeros(n2 + 1, dtype=float)
        ii = np.arange(1, n2)
        v = np.ones(n2 - 1, dtype=float)
        for kk in range(1, n2 // 2):
            v -= 2.0 * np.cos(2.0 * kk * theta[ii]) / (4.0 * kk * kk - 1.0)
        v -= np.cos(n2 * theta[ii]) / (n2 * n2 - 1.0)
        w_full[ii] = 2.0 * v / n2
        w_full[0] = 1.0 / (n2 * n2 - 1.0)
        w_full[n2] = 1.0 / (n2 * n2 - 1.0)

        w_even = np.zeros(n_pts, dtype=float)
        w_even[: self.M] = w_full[: self.M]
        w_even[self.M] = 0.5 * w_full[self.M]
        self.w = w_even[::-1] * self.R_max

        t_nodes = np.cos(np.outer(np.arange(n_pts), 2.0 * np.arccos(self.t)))
        self._T_inv = np.linalg.inv(t_nodes.T)

    def interp_matrix(self, t_target: np.ndarray) -> np.ndarray:
        """Exact even-Chebyshev polynomial interpolation matrix to target t in [0, 1]."""
        n_pts = self.M + 1
        th = np.arccos(np.clip(np.asarray(t_target, dtype=float), 0.0, 1.0))
        return np.cos(np.outer(th, 2.0 * np.arange(n_pts))) @ self._T_inv


class SpectralDiracSolver:
    """QR-orthonormalized kinetic-balanced polynomial spectral Galerkin Dirac solver.

    Large-component basis:  phi_k(r) = t^(l+1) (1 - t) P_k(2t - 1), t = r / R_max.
    Small-component basis:  chi_k(r) = (hbarc / 2 M_ref) (d/dr + kappa/r) phi_k(r).

    QR orthonormalization on the Gauss-Legendre quadrature grid converts the
    generalized Galerkin pencil H c = E S c into a standard symmetric eigenproblem
    with S = I and exact off-diagonal block H_SL = 2 M_ref R_S R_L^{-1}.
    By the Stanton-Havriliak kinetic-balance theorem, no spurious states appear
    in the mass gap (-M, +M).
    """

    def __init__(
        self,
        r_max: float,
        n_basis: int = 44,
        n_quad: int = 120,
        mass_ref: float = 939.0,
    ) -> None:
        if r_max <= 0.0 or n_basis < 12 or n_quad < 2 * n_basis:
            raise DiracHartreeBridgeError("invalid SpectralDiracSolver parameters")
        self.R_max = float(r_max)
        self.n_basis = int(n_basis)
        self.n_quad = int(n_quad)
        self.mass_ref = float(mass_ref)
        x_gl, w_gl = np.polynomial.legendre.leggauss(self.n_quad)
        self.t_q = 0.5 * (x_gl + 1.0)
        self.w_q = 0.5 * w_gl * self.R_max
        self.r_q = self.R_max * self.t_q
        self._cache: dict[tuple[int, int], tuple[Any, ...]] = {}

    def _basis_at(self, t: np.ndarray, kappa: int) -> tuple[np.ndarray, np.ndarray]:
        l = _orbital_l_from_kappa(kappa)
        p = l + 1
        n = self.n_basis
        x = 2.0 * t - 1.0
        leg = np.empty((n, t.size), dtype=float)
        dleg = np.empty((n, t.size), dtype=float)
        leg[0] = 1.0
        dleg[0] = 0.0
        if n > 1:
            leg[1] = x
            dleg[1] = 1.0
        for k in range(1, n - 1):
            leg[k + 1] = ((2 * k + 1) * x * leg[k] - k * leg[k - 1]) / (k + 1)
            dleg[k + 1] = (2 * k + 1) * leg[k] + dleg[k - 1]
        q = (1.0 - t)[None, :] * leg
        dq_dt = -leg + 2.0 * (1.0 - t)[None, :] * dleg
        tp = t**p
        phi = tp[None, :] * q
        scale = HBARC_MEV_FM / (2.0 * self.mass_ref * self.R_max)
        if kappa < 0:
            # p + kappa = (l + 1) - (l + 1) = 0 identically -> smooth at r = 0.
            chi = scale * (tp[None, :] * dq_dt)
        else:
            chi = scale * ((p + kappa) * (t ** (p - 1))[None, :] * q + tp[None, :] * dq_dt)
            # For kappa > 0, every derivative basis function satisfies
            # int_0^R r^kappa chi_k dr = [r^kappa phi_k]_0^R = 0 identically.
            # Appending one smooth function t^l (1 - t) removes this unphysical
            # integral constraint in spatially varying potentials while keeping
            # the kinetic-balance mass gap (-M, +M) strictly spurious-free.
            chi_extra = ((t**l) * (1.0 - t))[None, :]
            chi = np.vstack([chi, chi_extra])
        return phi, chi

    def _basis_over_r_at_zero(self, kappa: int) -> tuple[np.ndarray, np.ndarray]:
        """Exact analytical limit lim_{r->0} (phi_k(r)/r, chi_k(r)/r)."""
        n = self.n_basis
        phi0 = np.zeros(n, dtype=float)
        chi0 = np.zeros(n, dtype=float)
        k = np.arange(n, dtype=float)
        p_m1 = (-1.0) ** k
        if kappa == -1:
            phi0 = p_m1 / self.R_max
            dp_m1 = ((-1.0) ** (k + 1.0)) * 0.5 * k * (k + 1.0)
            chi0 = (HBARC_MEV_FM / (2.0 * self.mass_ref * self.R_max * self.R_max)) * (
                -p_m1 + 2.0 * dp_m1
            )
        elif kappa == 1:
            chi0 = 3.0 * (HBARC_MEV_FM / (2.0 * self.mass_ref * self.R_max * self.R_max)) * p_m1
        if kappa > 0:
            extra0 = (1.0 / self.R_max) if kappa == 1 else 0.0
            chi0 = np.append(chi0, extra0)
        return phi0, chi0

    def get_sector(self, kappa: int, t_eval: np.ndarray | None = None) -> tuple[Any, ...]:
        eval_size = 0 if t_eval is None else int(t_eval.size)
        key = (int(kappa), eval_size)
        if key not in self._cache:
            phi_q, chi_q = self._basis_at(self.t_q, kappa)
            n_l = phi_q.shape[0]
            sw = np.sqrt(self.w_q)
            q_l, r_l = np.linalg.qr((phi_q * sw[None, :]).T, mode="reduced")
            q_s, r_s = np.linalg.qr((chi_q * sw[None, :]).T, mode="reduced")
            h_sl = (2.0 * self.mass_ref) * np.linalg.solve(r_l.T, r_s[:, :n_l].T).T
            if t_eval is not None:
                phi_e, chi_e = self._basis_at(t_eval, kappa)
                phi_orth_e = np.linalg.solve(r_l.T, phi_e)
                chi_orth_e = np.linalg.solve(r_s.T, chi_e)
                phi0, chi0 = self._basis_over_r_at_zero(kappa)
                phi0_orth = np.linalg.solve(r_l.T, phi0)
                chi0_orth = np.linalg.solve(r_s.T, chi0)
            else:
                phi_orth_e = chi_orth_e = phi0_orth = chi0_orth = None
            self._cache[key] = (q_l, q_s, h_sl, phi_orth_e, chi_orth_e, phi0_orth, chi0_orth)
        return self._cache[key]

    def solve_sector(
        self,
        kappa: int,
        m_eff_q: np.ndarray,
        v_q: np.ndarray,
        t_eval: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        q_l, q_s, h_sl, phi_e, chi_e, phi0, chi0 = self.get_sector(kappa, t_eval)
        n_l = q_l.shape[1]
        n_s = q_s.shape[1]
        h_mat = np.empty((n_l + n_s, n_l + n_s), dtype=float)
        h_mat[:n_l, :n_l] = (q_l.T * (m_eff_q + v_q)[None, :]) @ q_l
        h_mat[n_l:, :n_l] = h_sl
        h_mat[:n_l, n_l:] = h_sl.T
        h_mat[n_l:, n_l:] = (q_s.T * (-m_eff_q + v_q)[None, :]) @ q_s
        vals, vecs = eigh(h_mat, check_finite=False)
        return vals, vecs, h_mat, phi_e, chi_e, phi0, chi0


# ---------------------------------------------------------------------------
# Coupled Field Solver (Newton-Raphson) + Anderson SCF Acceleration
# ---------------------------------------------------------------------------


def _solve_fields_newton(
    grid: EvenChebyshevGrid,
    params: Mapping[str, float],
    rho_s: np.ndarray,
    rho_p: np.ndarray,
    rho_n: np.ndarray,
    w_init: np.ndarray,
    s_init: np.ndarray,
    a0_init: np.ndarray,
    rho0_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve all 5 coupled radial Helmholtz/Poisson equations to machine precision."""
    n_pts = grid.M + 1
    hc2 = HBARC_MEV_FM * HBARC_MEV_FM
    hc3 = HBARC_MEV_FM**3
    lap = grid.L
    rho_b = rho_p + rho_n
    rho_3 = rho_p - rho_n

    m_n = params["M_N"]
    w0 = params["W0"]
    lam_w = params["lam_W"]
    m_s2 = params["m_s"] ** 2
    b_s = params["b_s"]
    c_s = params["c_s"]
    d_s = params.get("d_s", 0.0)
    g_s = params["g_s"]
    q_phi2 = params.get("q_phi", params["m_omega"] / w0) ** 2
    g_om = params["g_omega"]
    zeta = params["zeta"]
    m_rho2 = params["m_rho"] ** 2
    g_rho = params["g_rho"]
    lam_v = params["Lambda_v"]

    # 1. Light chiral scalar s(r):
    #    -hc2 L s + m_s^2 s + b_s s^2 + c_s s^3 + d_s s^5 - g_s hc3 rho_s = 0
    s = s_init.copy()
    for _ in range(25):
        res = (
            -hc2 * (lap @ s)
            + m_s2 * s
            + b_s * (s**2)
            + c_s * (s**3)
            + d_s * (s**5)
            - g_s * hc3 * rho_s
        )
        res[-1] = s[-1]
        if float(np.max(np.abs(res))) < 1.0e-11:
            break
        jac = -hc2 * lap + np.diag(m_s2 + 2.0 * b_s * s + 3.0 * c_s * (s**2) + 5.0 * d_s * (s**4))
        jac[-1, :] = 0.0
        jac[-1, -1] = 1.0
        s -= np.linalg.solve(jac, res)

    # 2. Coupled heavy gluon W(r), vector A0(r), and isovector rho0(r) via 3x3 block Newton-Raphson
    w = w_init.copy()
    a0 = a0_init.copy()
    rho0 = rho0_init.copy()
    c_lv = 2.0 * lam_v * (g_om**2) * (g_rho**2)
    c_z = (zeta / 6.0) * (g_om**4)
    zeros_blk = np.zeros((n_pts, n_pts), dtype=float)
    for _ in range(25):
        f_w = (
            -hc2 * (lap @ w)
            + lam_w * w * (w * w - w0 * w0)
            - q_phi2 * w * (a0**2)
            + (m_n / w0) * hc3 * rho_s
        )
        f_a = (
            -hc2 * (lap @ a0)
            + q_phi2 * (w**2) * a0
            + c_z * (a0**3)
            + c_lv * (rho0**2) * a0
            - g_om * hc3 * rho_b
        )
        f_r = (
            -hc2 * (lap @ rho0)
            + m_rho2 * rho0
            + c_lv * (a0**2) * rho0
            - 0.5 * g_rho * hc3 * rho_3
        )
        f_w[-1] = w[-1] - w0
        f_a[-1] = a0[-1]
        f_r[-1] = rho0[-1]
        rhs = np.concatenate([f_w, f_a, f_r])
        if float(np.max(np.abs(rhs))) < 1.0e-11:
            break
        j_ww = -hc2 * lap + np.diag(lam_w * (3.0 * w * w - w0 * w0) - q_phi2 * (a0**2))
        j_wa = np.diag(-2.0 * q_phi2 * w * a0)
        j_aw = np.diag(2.0 * q_phi2 * w * a0)
        j_aa = -hc2 * lap + np.diag(q_phi2 * (w**2) + 3.0 * c_z * (a0**2) + c_lv * (rho0**2))
        j_ar = np.diag(2.0 * c_lv * a0 * rho0)
        j_ra = j_ar.copy()
        j_rr = -hc2 * lap + np.diag(m_rho2 + c_lv * (a0**2))
        for blk in (j_ww, j_wa, j_aw, j_aa, j_ar, j_ra, j_rr):
            blk[-1, :] = 0.0
        j_ww[-1, -1] = 1.0
        j_aa[-1, -1] = 1.0
        j_rr[-1, -1] = 1.0
        j_full = np.block(
            [
                [j_ww, j_wa, zeros_blk],
                [j_aw, j_aa, j_ar],
                [zeros_blk, j_ra, j_rr],
            ]
        )
        delta = np.linalg.solve(j_full, rhs)
        w -= delta[:n_pts]
        a0 -= delta[n_pts : 2 * n_pts]
        rho0 -= delta[2 * n_pts :]

    # 4. Coulomb V_C(r): -L V_C = 4 pi alpha hbarc rho_p, exact exterior Robin BC R V_C'(R) + V_C(R) = 0
    mat_c = -lap.copy()
    rhs_c = 4.0 * math.pi * ALPHA_EM * HBARC_MEV_FM * rho_p
    mat_c[-1, :] = grid.R_max * grid.D1[-1, :]
    mat_c[-1, -1] += 1.0
    rhs_c[-1] = 0.0
    v_c = np.linalg.solve(mat_c, rhs_c)

    if not all(np.all(np.isfinite(arr)) for arr in (w, s, a0, rho0, v_c)):
        raise DiracHartreeBridgeError("non-finite field encountered in Newton solver")
    return w, s, a0, rho0, v_c


def _anderson_mix(
    x_cur: np.ndarray,
    x_new: np.ndarray,
    hist_x: list[np.ndarray],
    hist_f: list[np.ndarray],
    beta: float = 0.40,
    m_max: int = 6,
    reg: float = 1.0e-12,
) -> np.ndarray:
    """Anderson acceleration with Tikhonov regularization and safe damping fallback."""
    f = x_new - x_cur
    hist_x.append(x_cur.copy())
    hist_f.append(f.copy())
    if len(hist_x) > m_max + 1:
        hist_x.pop(0)
        hist_f.pop(0)
    k = len(hist_x) - 1
    if k == 0:
        return x_cur + beta * f
    df = np.column_stack([hist_f[i + 1] - hist_f[i] for i in range(k)])
    dx = np.column_stack([hist_x[i + 1] - hist_x[i] for i in range(k)])
    gram = df.T @ df
    scale = max(float(np.trace(gram)) / k, 1.0)
    try:
        gamma = np.linalg.solve(gram + (reg * scale) * np.eye(k), df.T @ f)
    except np.linalg.LinAlgError:
        return x_cur + beta * f
    x_bar = x_cur - dx @ gamma
    f_bar = f - df @ gamma
    cand = x_bar + beta * f_bar
    if not np.all(np.isfinite(cand)):
        return x_cur + beta * f
    return cand


def _candidate_kappas_for_A(a_mass: int) -> tuple[int, ...]:
    if a_mass <= 20:
        return (-1, -2, 1, -3, 2)
    if a_mass <= 60:
        # Includes kappa=+3 (1f5/2) so Ca48 1f5/2 - 1f7/2 spin-orbit splitting is computed.
        return (-1, -2, 1, -3, 2, -4, 3)
    return (-1, -2, 1, -3, 2, -4, 3, -5, 4, -6, 5, -7, 6)


def _field_residuals(
    grid: EvenChebyshevGrid,
    params: Mapping[str, float],
    w: np.ndarray,
    s: np.ndarray,
    a0: np.ndarray,
    rho0: np.ndarray,
    v_c: np.ndarray,
    rho_s: np.ndarray,
    rho_p: np.ndarray,
    rho_n: np.ndarray,
) -> dict[str, float]:
    """Compute normalized relative residuals for all 5 coupled radial PDEs."""
    hc2 = HBARC_MEV_FM * HBARC_MEV_FM
    hc3 = HBARC_MEV_FM**3
    lap = grid.L
    rho_b = rho_p + rho_n
    rho_3 = rho_p - rho_n

    c_lv = 2.0 * params["Lambda_v"] * (params["g_omega"] ** 2) * (params["g_rho"] ** 2)
    c_z = (params["zeta"] / 6.0) * (params["g_omega"] ** 4)
    q_phi2 = params.get("q_phi", params["m_omega"] / params["W0"]) ** 2
    d_s = params.get("d_s", 0.0)

    res_w = (
        -hc2 * (lap @ w)
        + params["lam_W"] * w * (w * w - params["W0"] ** 2)
        - q_phi2 * w * (a0**2)
        + (params["M_N"] / params["W0"]) * hc3 * rho_s
    )
    res_s = (
        -hc2 * (lap @ s)
        + (params["m_s"] ** 2) * s
        + params["b_s"] * (s**2)
        + params["c_s"] * (s**3)
        + d_s * (s**5)
        - params["g_s"] * hc3 * rho_s
    )
    res_a = (
        -hc2 * (lap @ a0)
        + q_phi2 * (w**2) * a0
        + c_z * (a0**3)
        + c_lv * (rho0**2) * a0
        - params["g_omega"] * hc3 * rho_b
    )
    res_rho = (
        -hc2 * (lap @ rho0)
        + (params["m_rho"] ** 2) * rho0
        + c_lv * (a0**2) * rho0
        - 0.5 * params["g_rho"] * hc3 * rho_3
    )
    res_c = -(lap @ v_c) - 4.0 * math.pi * ALPHA_EM * HBARC_MEV_FM * rho_p

    scale_w = max(1.0, float(np.max(np.abs((params["M_N"] / params["W0"]) * hc3 * rho_s))))
    scale_s = max(1.0, float(np.max(np.abs(params["g_s"] * hc3 * rho_s))))
    scale_a = max(1.0, float(np.max(np.abs(params["g_omega"] * hc3 * rho_b))))
    scale_rho = max(1.0e3, float(np.max(np.abs(0.5 * params["g_rho"] * hc3 * rho_3))))
    scale_c = max(1.0e-6, float(np.max(np.abs(4.0 * math.pi * ALPHA_EM * HBARC_MEV_FM * rho_p))))

    rw = float(np.max(np.abs(res_w[:-1]))) / scale_w
    rs = float(np.max(np.abs(res_s[:-1]))) / scale_s
    ra = float(np.max(np.abs(res_a[:-1]))) / scale_a
    rrho = float(np.max(np.abs(res_rho[:-1]))) / scale_rho
    rc = float(np.max(np.abs(res_c[:-1]))) / scale_c
    return {
        "W_rel": rw,
        "s_rel": rs,
        "A0_rel": ra,
        "rho0_rel": rrho,
        "V_C_rel": rc,
        "max_rel": max(rw, rs, ra, rrho, rc),
    }


def solve_nucleus_dirac_hartree(
    Z: int,
    N: int,
    params: Mapping[str, Any],
    *,
    R_max: float | None = None,
    M_field: int = 84,
    n_basis: int = 44,
    n_quad: int = 120,
    tol: float = 1.0e-10,
    max_iter: int = 100,
    include_profiles: bool = False,
) -> dict[str, Any]:
    """Solve a closed-shell spherical nucleus (Z, N) self-consistently in Dirac-Hartree.

    This is the canonical reusable solver entry point for Phase 2 (frozen Law 2
    parameter evaluation with zero refitting).
    """
    z_int = int(Z)
    n_int = int(N)
    if z_int <= 0 or n_int <= 0:
        raise DiracHartreeBridgeError("Z and N must be positive integers")
    a_int = z_int + n_int
    p = normalize_params(params)

    r_box = float(R_max) if R_max is not None else (16.0 if a_int <= 50 else 20.0)
    grid = EvenChebyshevGrid(r_box, m_intervals=M_field)
    dirac = SpectralDiracSolver(r_box, n_basis=n_basis, n_quad=n_quad, mass_ref=p["M_N"])
    interp_q = grid.interp_matrix(dirac.t_q)
    r = grid.r

    # Smooth Woods-Saxon initial guess for fields
    r0 = 1.2 * (a_int ** (1.0 / 3.0))
    ws = 1.0 / (1.0 + np.exp((r - r0) / 0.55))
    w = p["W0"] * (1.0 - 0.001 * ws)
    s = (350.0 / p["g_s"]) * ws
    a0 = (280.0 / p["g_omega"]) * ws
    rho0 = -5.0 * ((n_int - z_int) / a_int) * ws
    v_c = np.where(
        r <= r0,
        z_int * ALPHA_EM * HBARC_MEV_FM / (2.0 * r0) * (3.0 - (r / r0) ** 2),
        z_int * ALPHA_EM * HBARC_MEV_FM / np.maximum(r, 1.0e-12),
    )

    kappas = _candidate_kappas_for_A(a_int)
    hist_x: list[np.ndarray] = []
    hist_f: list[np.ndarray] = []
    converged = False
    scf_err = math.inf
    iterations = 0

    def _solve_orbitals_and_densities(
        w_f: np.ndarray, s_f: np.ndarray, a0_f: np.ndarray, rho0_f: np.ndarray, vc_f: np.ndarray
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, list[dict[str, Any]]], dict[str, dict[tuple[int, int], float]], np.ndarray]:
        m_eff = p["M_N"] * (w_f / p["W0"]) - p["g_s"] * s_f
        v_p = p["g_omega"] * a0_f + 0.5 * p["g_rho"] * rho0_f + vc_f
        v_n = p["g_omega"] * a0_f - 0.5 * p["g_rho"] * rho0_f
        m_eff_q = interp_q @ m_eff
        vp_q = interp_q @ v_p
        vn_q = interp_q @ v_n

        dens_out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        orbs_out: dict[str, list[dict[str, Any]]] = {}
        spec_out: dict[str, dict[tuple[int, int], float]] = {}

        for sp, count, vq in (("p", z_int, vp_q), ("n", n_int, vn_q)):
            cands: list[dict[str, Any]] = []
            sp_map: dict[tuple[int, int], float] = {}
            for kap in kappas:
                vals, vecs, h_mat, phi_e, chi_e, phi0, chi0 = dirac.solve_sector(
                    kap, m_eff_q, vq, grid.t
                )
                assert phi_e is not None and chi_e is not None and phi0 is not None and chi0 is not None
                pos_idx = np.where(vals > 0.0)[0]
                for nr, ii in enumerate(pos_idx[:6]):
                    e_val = float(vals[ii])
                    vec = vecs[:, ii]
                    c_l = vec[:n_basis]
                    c_s = vec[n_basis:]
                    g_r = c_l @ phi_e
                    f_r = c_s @ chi_e
                    g0 = float(c_l @ phi0)
                    f0 = float(c_s @ chi0)
                    eig_res = float(np.linalg.norm(h_mat @ vec - e_val * vec) / max(abs(e_val), 1.0))
                    cands.append(
                        {
                            "E": e_val,
                            "kappa": kap,
                            "n_radial": nr,
                            "degeneracy": 2 * abs(kap),
                            "G": g_r,
                            "F": f_r,
                            "g0": g0,
                            "f0": f0,
                            "vec": vec,
                            "eig_res": eig_res,
                        }
                    )
                    sp_map[(kap, nr)] = e_val
            cands.sort(key=lambda item: item["E"])
            rem = float(count)
            rho_v = np.zeros_like(r)
            rho_sc = np.zeros_like(r)
            occ_list: list[dict[str, Any]] = []
            for cand in cands:
                if rem <= 1.0e-12:
                    break
                occ = min(float(cand["degeneracy"]), rem)
                rem -= occ
                g_r = cand["G"]
                f_r = cand["F"]
                rho_v[1:] += occ * (g_r[1:] ** 2 + f_r[1:] ** 2) / (4.0 * math.pi * r[1:] ** 2)
                rho_sc[1:] += occ * (g_r[1:] ** 2 - f_r[1:] ** 2) / (4.0 * math.pi * r[1:] ** 2)
                rho_v[0] += occ * (cand["g0"] ** 2 + cand["f0"] ** 2) / (4.0 * math.pi)
                rho_sc[0] += occ * (cand["g0"] ** 2 - cand["f0"] ** 2) / (4.0 * math.pi)
                cand["occupancy"] = occ
                occ_list.append(cand)
            if rem > 1.0e-8:
                raise DiracHartreeBridgeError(
                    f"insufficient bound orbitals for {sp} (Z={z_int}, N={n_int}); {rem} particles unplaced"
                )
            dens_out[sp] = (rho_v, rho_sc)
            orbs_out[sp] = occ_list
            spec_out[sp] = sp_map
        return dens_out, orbs_out, spec_out, m_eff_q

    dens: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    orbs: dict[str, list[dict[str, Any]]] = {}
    all_spec: dict[str, dict[tuple[int, int], float]] = {}
    m_eff_q = np.zeros_like(dirac.r_q)

    for it in range(1, max_iter + 1):
        iterations = it
        dens, orbs, all_spec, m_eff_q = _solve_orbitals_and_densities(w, s, a0, rho0, v_c)
        rho_p, rhos_p = dens["p"]
        rho_n, rhos_n = dens["n"]
        rho_s = rhos_p + rhos_n
        w_new, s_new, a0_new, rho0_new, vc_new = _solve_fields_newton(
            grid, p, rho_s, rho_p, rho_n, w, s, a0, rho0
        )
        x_cur = np.concatenate([w, s, a0, rho0, v_c])
        x_new = np.concatenate([w_new, s_new, a0_new, rho0_new, vc_new])
        scf_err = float(np.max(np.abs(x_new - x_cur)))
        if scf_err < tol:
            w, s, a0, rho0, v_c = w_new, s_new, a0_new, rho0_new, vc_new
            converged = True
            break
        x_next = _anderson_mix(x_cur, x_new, hist_x, hist_f, beta=0.40, m_max=6)
        # Guard against unphysical scalar/gluon excursions during early mixing
        w_t, s_t, a0_t, rho0_t, vc_t = np.split(x_next, 5)
        if np.min(w_t) <= 0.2 * p["W0"] or np.min(p["M_N"] * (w_t / p["W0"]) - p["g_s"] * s_t) <= 50.0:
            x_next = 0.75 * x_cur + 0.25 * x_new
            hist_x.clear()
            hist_f.clear()
            w_t, s_t, a0_t, rho0_t, vc_t = np.split(x_next, 5)
        w, s, a0, rho0, v_c = w_t, s_t, a0_t, rho0_t, vc_t

    if not converged:
        raise DiracHartreeBridgeError(
            f"SCF failed to converge for (Z={z_int}, N={n_int}) in {max_iter} iterations (err={scf_err:.3e})"
        )

    # One final exact synchronization pass so orbitals & fields match to < tol
    dens, orbs, all_spec, m_eff_q = _solve_orbitals_and_densities(w, s, a0, rho0, v_c)
    rho_p, rhos_p = dens["p"]
    rho_n, rhos_n = dens["n"]
    rho_s = rhos_p + rhos_n
    rho_b = rho_p + rho_n
    rho_3 = rho_p - rho_n

    fres = _field_residuals(grid, p, w, s, a0, rho0, v_c, rho_s, rho_p, rho_n)

    # Spherical integration weights 4 pi r^2 w_j (fm^3)
    w3 = 4.0 * math.pi * (r * r) * grid.w
    hc3 = HBARC_MEV_FM**3

    # Independent Energy Formula 1: single-particle eigenvalue sum + field rearrangement (NO gradients)
    q_phi2 = p["q_phi"] ** 2
    d_s = p["d_s"]
    eig_sum = sum(orb["occupancy"] * orb["E"] for sp in ("p", "n") for orb in orbs[sp])
    e_field_eig = float(
        np.dot(
            w3,
            (
                0.5 * p["g_s"] * s * rho_s
                - (p["b_s"] / 6.0) * (s**3) / hc3
                - (p["c_s"] / 4.0) * (s**4) / hc3
                - (d_s / 3.0) * (s**6) / hc3
                - 0.5 * (p["M_N"] / p["W0"]) * (w - p["W0"]) * rho_s
                - 0.25 * p["lam_W"] * ((w - p["W0"]) ** 3) * (w + p["W0"]) / hc3
                + 0.5 * q_phi2 * w * (w - p["W0"]) * (a0**2) / hc3
                - 0.5 * p["g_omega"] * a0 * rho_b
                + (p["zeta"] / 24.0) * (p["g_omega"] ** 4) * (a0**4) / hc3
                - 0.25 * p["g_rho"] * rho0 * rho_3
                + p["Lambda_v"] * (p["g_omega"] ** 2) * (p["g_rho"] ** 2) * (a0**2) * (rho0**2) / hc3
                - 0.5 * v_c * rho_p
            ),
        )
    )
    e_tot_eig = eig_sum + e_field_eig

    # Independent Energy Formula 2: direct Dirac kinetic+mass expectation + explicit T^{00} field functional
    e_kin_mass = 0.0
    max_orb_eig_res = 0.0
    max_orb_norm_err = 0.0
    for sp in ("p", "n"):
        for orb in orbs[sp]:
            q_l, q_s, h_sl, _, _, _, _ = dirac.get_sector(orb["kappa"], grid.t)
            c_l = orb["vec"][:n_basis]
            c_s = orb["vec"][n_basis:]
            # Exact L2 norm on the Gauss-Legendre grid (S = I)
            norm_q = float(np.dot(c_l, c_l) + np.dot(c_s, c_s))
            max_orb_norm_err = max(max_orb_norm_err, abs(norm_q - 1.0))
            max_orb_eig_res = max(max_orb_eig_res, orb["eig_res"])
            kin_mass_i = float(
                c_l @ ((q_l.T * m_eff_q[None, :]) @ q_l) @ c_l
                - c_s @ ((q_s.T * m_eff_q[None, :]) @ q_s) @ c_s
                + 2.0 * (c_s @ (h_sl @ c_l))
            )
            e_kin_mass += orb["occupancy"] * kin_mass_i

    dw = grid.D1 @ w
    ds = grid.D1 @ s
    da0 = grid.D1 @ a0
    drho0 = grid.D1 @ rho0
    dvc = grid.D1 @ v_c

    e_field_direct = float(
        np.dot(
            w3,
            (
                0.5 * (dw**2) / HBARC_MEV_FM
                + 0.25 * p["lam_W"] * ((w * w - p["W0"] ** 2) ** 2) / hc3
                + 0.5 * (ds**2) / HBARC_MEV_FM
                + (
                    0.5 * (p["m_s"] ** 2) * (s**2)
                    + (p["b_s"] / 3.0) * (s**3)
                    + 0.25 * p["c_s"] * (s**4)
                    + (d_s / 6.0) * (s**6)
                )
                / hc3
                + 0.5 * (da0**2) / HBARC_MEV_FM
                + (0.5 * q_phi2 * (w**2) * (a0**2) + (p["zeta"] / 8.0) * (p["g_omega"] ** 4) * (a0**4))
                / hc3
                + 0.5 * (drho0**2) / HBARC_MEV_FM
                + (
                    0.5 * (p["m_rho"] ** 2) * (rho0**2)
                    + 3.0 * p["Lambda_v"] * (p["g_omega"] ** 2) * (p["g_rho"] ** 2) * (a0**2) * (rho0**2)
                )
                / hc3
                + (dvc**2) / (8.0 * math.pi * ALPHA_EM * HBARC_MEV_FM)
            ),
        )
    ) + r_box * (v_c[-1] ** 2) / (2.0 * ALPHA_EM * HBARC_MEV_FM)
    e_tot_direct = e_kin_mass + e_field_direct

    e_cm = -0.75 * 41.0 * (a_int ** (-1.0 / 3.0))
    ba_eig = (e_tot_eig + e_cm - a_int * p["M_N"]) / a_int
    ba_direct = (e_tot_direct + e_cm - a_int * p["M_N"]) / a_int

    z_integrated = float(np.dot(w3, rho_p))
    n_integrated = float(np.dot(w3, rho_n))
    r_p = math.sqrt(float(np.dot(w3, (r * r) * rho_p)) / z_integrated)
    r_n = math.sqrt(float(np.dot(w3, (r * r) * rho_n)) / n_integrated)
    r_m = math.sqrt(float(np.dot(w3, (r * r) * rho_b)) / (z_integrated + n_integrated))
    r_ch = math.sqrt(r_p * r_p + PROTON_CHARGE_RADIUS_FM * PROTON_CHARGE_RADIUS_FM)

    # Extract spin-orbit splittings for key pairs present in spectrum
    so_pairs = (
        ("1p", -2, 1, 0),
        ("1d", -3, 2, 0),
        ("1f", -4, 3, 0),
        ("1g", -5, 4, 0),
        ("2p", -2, 1, 1),
    )
    spin_orbit: dict[str, dict[str, float]] = {}
    for label, k_align, k_anti, nr in so_pairs:
        if (k_align, nr) in all_spec["n"] and (k_anti, nr) in all_spec["n"]:
            e_n_al = all_spec["n"][(k_align, nr)]
            e_n_an = all_spec["n"][(k_anti, nr)]
            e_p_al = all_spec["p"][(k_align, nr)]
            e_p_an = all_spec["p"][(k_anti, nr)]
            if e_n_al < p["M_N"] and e_n_an < p["M_N"]:
                spin_orbit[label] = {
                    "neutron_MeV": float(e_n_an - e_n_al),
                    "proton_MeV": float(e_p_an - e_p_al),
                }

    def _serialize_shells(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "label": _shell_label(item["kappa"], item["n_radial"]),
                "kappa": int(item["kappa"]),
                "n_radial": int(item["n_radial"]),
                "degeneracy": int(item["degeneracy"]),
                "occupancy": float(item["occupancy"]),
                "E_MeV": float(item["E"]),
                "eps_MeV": float(item["E"] - p["M_N"]),
                "eig_residual": float(item["eig_res"]),
            }
            for item in items
        ]

    m_eff_0 = float(p["M_N"] * (w[0] / p["W0"]) - p["g_s"] * s[0])
    out: dict[str, Any] = {
        "Z": z_int,
        "N": n_int,
        "A": a_int,
        "numerical_settings": {
            "R_max_fm": r_box,
            "M_field": int(M_field),
            "n_basis": int(n_basis),
            "n_quad": int(n_quad),
            "tol": float(tol),
        },
        "convergence": {
            "converged": converged,
            "iterations": int(iterations),
            "scf_field_update_MeV": scf_err,
            "field_residuals": fres,
            "max_orbital_eig_residual": max_orb_eig_res,
            "max_orbital_norm_error": max_orb_norm_err,
        },
        "particle_numbers": {
            "Z_integrated": z_integrated,
            "N_integrated": n_integrated,
            "Z_rel_err": abs(z_integrated - z_int) / z_int,
            "N_rel_err": abs(n_integrated - n_int) / n_int,
        },
        "energies": {
            "E_tot_eig_MeV": float(e_tot_eig),
            "E_tot_direct_MeV": float(e_tot_direct),
            "energy_formula_diff_MeV": float(abs(e_tot_eig - e_tot_direct)),
            "energy_formula_diff_per_A_MeV": float(abs(ba_eig - ba_direct)),
            "E_cm_MeV": float(e_cm),
            "B_over_A_eig_MeV": float(ba_eig),
            "B_over_A_direct_MeV": float(ba_direct),
            "binding_energy_per_A_MeV": float(-ba_eig),
            "single_particle_sum_MeV": float(eig_sum),
            "dirac_kin_mass_MeV": float(e_kin_mass),
        },
        "radii": {
            "R_p_fm": float(r_p),
            "R_n_fm": float(r_n),
            "R_matter_fm": float(r_m),
            "R_ch_fm": float(r_ch),
            "proton_charge_radius_input_fm": PROTON_CHARGE_RADIUS_FM,
            "Delta_R_np_fm": float(r_n - r_p),
        },
        "central_fields": {
            "W_center_MeV": float(w[0]),
            "W_center_over_W0": float(w[0] / p["W0"]),
            "W_boundary_MeV": float(w[-1]),
            "s_center_MeV": float(s[0]),
            "M_eff_center_MeV": m_eff_0,
            "M_eff_center_over_MN": float(m_eff_0 / p["M_N"]),
            "V_omega_center_MeV": float(p["g_omega"] * a0[0]),
            "V_rho_neutron_center_MeV": float(-0.5 * p["g_rho"] * rho0[0]),
            "V_coulomb_center_MeV": float(v_c[0]),
            "rho_B_center_fm3": float(rho_b[0]),
        },
        "spin_orbit_splittings_MeV": spin_orbit,
        "occupied_shells": {
            "proton": _serialize_shells(orbs["p"]),
            "neutron": _serialize_shells(orbs["n"]),
        },
    }
    if include_profiles:
        out["profiles"] = {
            "r_fm": r.tolist(),
            "rho_p_fm3": rho_p.tolist(),
            "rho_n_fm3": rho_n.tolist(),
            "rho_s_fm3": rho_s.tolist(),
            "W_MeV": w.tolist(),
            "s_MeV": s.tolist(),
            "A0_MeV": a0.tolist(),
            "rho0_MeV": rho0.tolist(),
            "V_C_MeV": v_c.tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# Analytical Relativistic Dirac Benchmarks (Zero Spurious States Verification)
# ---------------------------------------------------------------------------


def run_analytical_dirac_benchmarks() -> dict[str, Any]:
    """Verify the radial Dirac solver against 3 exact analytical relativistic solutions."""
    m_n = 939.0
    r_box = 15.0
    solver = SpectralDiracSolver(r_box, n_basis=44, n_quad=120, mass_ref=m_n)

    # 1. Free relativistic spherical box (V=0, M_eff=M_N, G(R)=0)
    box_checks: dict[str, Any] = {}
    spurious_in_gap = 0
    max_box_err = 0.0
    for kap in (-1, -2, 1, -3, 2, -6, 5):
        vals, _, _, _, _, _, _ = solver.solve_sector(
            kap, np.full_like(solver.r_q, m_n), np.zeros_like(solver.r_q)
        )
        n_gap = int(np.sum(np.abs(vals) < m_n - 1.0e-9))
        spurious_in_gap += n_gap
        pos = vals[vals > 0.0]
        row: dict[str, Any] = {"min_positive_MeV": float(pos[0]), "states_in_mass_gap": n_gap}
        if kap == -1:
            exact = [math.sqrt(m_n * m_n + (HBARC_MEV_FM * n * math.pi / r_box) ** 2) for n in (1, 2, 3)]
            errs = [abs(float(pos[i]) - exact[i]) for i in range(3)]
            max_box_err = max(max_box_err, max(errs))
            row["exact_s_wave_MeV"] = exact
            row["abs_errors_MeV"] = errs
        box_checks[f"kappa={kap}"] = row

    # 2. Exact relativistic spin-symmetric Dirac harmonic oscillator
    #    M_eff(r) - V(r) = M_N, M_eff(r) + V(r) = M_N + (1/2) k r^2
    #    Exact eigenvalues satisfy: (E - M_N) sqrt(E + M_N) = hbarc sqrt(2 k) (2 n_r + l + 3/2)
    k_ho = 5.0  # MeV / fm^2
    u_ho = 0.5 * k_ho * (solver.r_q**2)
    m_eff_ho = m_n + 0.5 * u_ho
    v_ho = 0.5 * u_ho
    ho_checks: dict[str, Any] = {}
    max_ho_err = 0.0
    for kap in (-1, -2, 1, -3, 2):
        vals, _, _, _, _, _, _ = solver.solve_sector(kap, m_eff_ho, v_ho)
        pos = vals[vals > 0.0]
        l = _orbital_l_from_kappa(kap)
        errs = []
        exacts = []
        for nr in range(3):
            rhs = HBARC_MEV_FM * math.sqrt(2.0 * k_ho) * (2 * nr + l + 1.5)
            e_ex = m_n + 20.0
            for _ in range(40):
                f_val = (e_ex - m_n) * math.sqrt(e_ex + m_n) - rhs
                df_val = math.sqrt(e_ex + m_n) + (e_ex - m_n) / (2.0 * math.sqrt(e_ex + m_n))
                e_ex -= f_val / df_val
            exacts.append(e_ex)
            errs.append(abs(float(pos[nr]) - e_ex))
        max_ho_err = max(max_ho_err, max(errs))
        ho_checks[f"kappa={kap}"] = {
            "exact_first3_MeV": exacts,
            "numeric_first3_MeV": [float(pos[i]) for i in range(3)],
            "max_abs_error_MeV": max(errs),
        }

    # 3. Exact relativistic Dirac-Coulomb (Sommerfeld formula) for Z*alpha = 0.2, m = 50 MeV
    m_c = 50.0
    z_alpha = 0.2
    coul_solver = SpectralDiracSolver(250.0, n_basis=55, n_quad=160, mass_ref=m_c)
    v_coul = -z_alpha * HBARC_MEV_FM / coul_solver.r_q
    coul_checks: dict[str, Any] = {}
    max_coul_rel_err = 0.0
    for label, kap, nr_eff in (("1s1/2", -1, 0), ("2p3/2", -2, 0), ("2p1/2", 1, 1)):
        vals, _, _, _, _, _, _ = coul_solver.solve_sector(kap, np.full_like(coul_solver.r_q, m_c), v_coul)
        pos = vals[vals > 0.0]
        gam = math.sqrt(kap * kap - z_alpha * z_alpha)
        e_exact = m_c / math.sqrt(1.0 + (z_alpha / (nr_eff + gam)) ** 2)
        e_num = float(pos[0])
        rel_err = abs(e_num - e_exact) / m_c
        max_coul_rel_err = max(max_coul_rel_err, rel_err)
        coul_checks[label] = {
            "kappa": kap,
            "E_exact_MeV": e_exact,
            "E_numeric_MeV": e_num,
            "rel_error": rel_err,
        }

    passed = (
        spurious_in_gap == 0
        and max_box_err < 1.0e-10
        and max_ho_err < 1.0e-10
        and max_coul_rel_err < 1.0e-4
    )
    return {
        "pass": bool(passed),
        "spurious_states_in_mass_gap": spurious_in_gap,
        "free_spherical_box": {"max_abs_error_MeV": max_box_err, "sectors": box_checks},
        "spin_symmetric_dirac_ho": {"max_abs_error_MeV": max_ho_err, "sectors": ho_checks},
        "dirac_coulomb_sommerfeld": {"max_rel_error": max_coul_rel_err, "states": coul_checks},
    }


def run_grid_refinement_check(
    params: Mapping[str, Any],
    base_ca48: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare Ca48 on standard vs refined spectral grid and enlarged box."""
    std = (
        dict(base_ca48)
        if base_ca48 is not None
        else solve_nucleus_dirac_hartree(20, 28, params, R_max=16.0, M_field=84, n_basis=44, n_quad=120)
    )
    ref = solve_nucleus_dirac_hartree(20, 28, params, R_max=18.0, M_field=104, n_basis=54, n_quad=150)
    dba = abs(std["energies"]["B_over_A_eig_MeV"] - ref["energies"]["B_over_A_eig_MeV"])
    drch = abs(std["radii"]["R_ch_fm"] - ref["radii"]["R_ch_fm"])
    dskin = abs(std["radii"]["Delta_R_np_fm"] - ref["radii"]["Delta_R_np_fm"])
    dso = abs(
        std["spin_orbit_splittings_MeV"]["1f"]["neutron_MeV"]
        - ref["spin_orbit_splittings_MeV"]["1f"]["neutron_MeV"]
    )
    passed = (
        dba < GRID_REFINEMENT_BA_LIMIT_MEV
        and drch < GRID_REFINEMENT_RADIUS_LIMIT_FM
        and dskin < GRID_REFINEMENT_RADIUS_LIMIT_FM
    )
    return {
        "nucleus": "Ca48",
        "standard_grid": std["numerical_settings"],
        "refined_grid": ref["numerical_settings"],
        "delta_B_over_A_MeV": float(dba),
        "delta_R_ch_fm": float(drch),
        "delta_skin_fm": float(dskin),
        "delta_SO_1f_neutron_MeV": float(dso),
        "pass": bool(passed),
    }


# ---------------------------------------------------------------------------
# Full Payload Builder & Fail-Closed Validator (SPEC-v2)
# ---------------------------------------------------------------------------


def build_result(
    params_override: Mapping[str, Any] | None = None,
    params_source: str = "self_calibrated_from_snm_default",
    run_grid_check: bool = True,
) -> dict[str, Any]:
    """Run the complete Law 3A Dirac-Hartree finite-nuclei audit."""
    if params_override is None:
        cal = calibrate_two_condensate_from_snm()
        frozen_params = normalize_params(cal)
        snm_verif = cal["snm_verification"]
        calibration_mode = "snm_self_calibrated"
    else:
        frozen_params = normalize_params(params_override)
        snm_verif = verify_uniform_snm(frozen_params, n0_fm3=DEFAULT_SNM_CALIBRATION_TARGETS["n0_fm3"])
        calibration_mode = "external_frozen_params_zero_refit"

    benchmarks = run_analytical_dirac_benchmarks()
    nuclei_results: dict[str, Any] = {}
    for name, z_val, n_val in DOUBLY_MAGIC_NUCLEI:
        nuclei_results[name] = solve_nucleus_dirac_hartree(z_val, n_val, frozen_params)

    grid_check = (
        run_grid_refinement_check(frozen_params, base_ca48=nuclei_results["Ca48"])
        if run_grid_check
        else None
    )

    ca40_rch = nuclei_results["Ca40"]["radii"]["R_ch_fm"]
    ca48_rch = nuclei_results["Ca48"]["radii"]["R_ch_fm"]
    delta_rch_ca = ca48_rch - ca40_rch
    skin_ca48 = nuclei_results["Ca48"]["radii"]["Delta_R_np_fm"]
    skin_pb208 = nuclei_results["Pb208"]["radii"]["Delta_R_np_fm"]

    summary_predictions = {
        "Delta_R_ch_Ca48_minus_Ca40_fm": float(delta_rch_ca),
        "Delta_R_np_Ca48_fm": float(skin_ca48),
        "Delta_R_np_Pb208_fm": float(skin_pb208),
        "Ca48_skin_positive": bool(skin_ca48 > 0.0),
        "Pb208_skin_positive": bool(skin_pb208 > 0.0),
        "improves_over_W8_TF_isotopic_shift": bool(
            abs(delta_rch_ca) < abs(REFERENCE_COMPARISONS["prior_w8_thomas_fermi_failures"]["Delta_R_ch_Ca48_minus_Ca40_fm"])
        ),
        "SO_O16_1p_neutron_MeV": nuclei_results["O16"]["spin_orbit_splittings_MeV"]["1p"]["neutron_MeV"],
        "SO_Ca40_1d_neutron_MeV": nuclei_results["Ca40"]["spin_orbit_splittings_MeV"]["1d"]["neutron_MeV"],
        "SO_Ca48_1f_neutron_MeV": nuclei_results["Ca48"]["spin_orbit_splittings_MeV"]["1f"]["neutron_MeV"],
        "max_field_residual_all_nuclei": max(
            nuclei_results[k]["convergence"]["field_residuals"]["max_rel"] for k, _, _ in DOUBLY_MAGIC_NUCLEI
        ),
        "max_energy_formula_diff_per_A_MeV": max(
            nuclei_results[k]["energies"]["energy_formula_diff_per_A_MeV"] for k, _, _ in DOUBLY_MAGIC_NUCLEI
        ),
        "max_particle_number_rel_err": max(
            max(
                nuclei_results[k]["particle_numbers"]["Z_rel_err"],
                nuclei_results[k]["particle_numbers"]["N_rel_err"],
            )
            for k, _, _ in DOUBLY_MAGIC_NUCLEI
        ),
    }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "conventions": {
            "hbarc_MeV_fm": HBARC_MEV_FM,
            "alpha_em": ALPHA_EM,
            "proton_charge_radius_fm": PROTON_CHARGE_RADIUS_FM,
            "R_ch_formula": "sqrt(R_p^2 + 0.8414^2) fm (raw point radii R_p, R_n reported separately)",
            "E_cm_formula": "-(3/4) * 41 * A^(-1/3) MeV",
            "B_over_A_sign": "negative means bound: B/A = (E_tot + E_cm - A * M_N) / A; positive binding_energy_per_A_MeV = -(B/A)",
            "dirac_kappa_convention": "kappa = -(l+1) for j=l+1/2, kappa = +l for j=l-1/2",
        },
        "calibration_targets": {
            "mode": calibration_mode,
            "params_source": params_source,
            "finite_nuclei_used_in_calibration": False,
            "snm_targets": dict(DEFAULT_SNM_CALIBRATION_TARGETS) if params_override is None else None,
            "snm_verification": snm_verif,
        },
        "frozen_params": frozen_params,
        "analytical_dirac_benchmarks": benchmarks,
        "forward_predictions": {
            "role": "zero_refit_forward_predictions_for_doubly_magic_nuclei",
            "nuclei": nuclei_results,
            "summary": summary_predictions,
        },
        "grid_refinement_check": grid_check,
        "reference_comparisons": REFERENCE_COMPARISONS,
        "honest_caveats": [
            (
                f"Zero finite-nucleus fitting was performed: all couplings were loaded verbatim from external frozen Law 2 parameters ({params_source}) with bulk SNM verification (n0={snm_verif['n0_fm3']:.2f} fm^-3, B/A={snm_verif['B_over_A_MeV']:.2f} MeV, K={snm_verif['K_MeV']:.1f} MeV, J={snm_verif['J_MeV']:.1f} MeV, L={snm_verif['L_MeV']:.1f} MeV, M*/M_N={snm_verif['M_eff_over_MN']:.3f})."
                if params_override is not None
                else "Zero finite-nucleus fitting was performed: all couplings come exclusively from bulk SNM saturation (n0=0.16, B/A=-16, K=240, J=31, L=68) or an external frozen parameter file."
            ),
            "Saturating at n0=0.16 fm^-3 with m_s=500 MeV yields slightly compact charge radii (~0.8-2.3% below experiment) and modest underbinding (~0.01-0.25 MeV/A) relative to standard RMF parameterizations fitted to finite nuclei (which prefer n0~0.148 fm^-3).",
            "Center-of-mass energy uses the standard harmonic-oscillator estimate E_cm = -(3/4)*41*A^(-1/3) MeV rather than full microscopic Peierls-Yoccoz projection; pairing correlations vanish identically for doubly-magic closed shells.",
            "No-sea Hartree approximation (Dirac sea polarization neglected; Fock exchange terms neglected).",
            "evidence_weight is 0.0: this module verifies mathematical self-consistency and forward predictions of the Law 2 two-condensate Lagrangian, not empirical proof.",
        ],
    }
    validate_payload(payload, check_source_sha=False)
    return payload


def validate_payload(payload: Any, *, check_source_sha: bool = True) -> None:
    """Fail-closed validation of a Law 3A Dirac-Hartree JSON artifact."""
    if not isinstance(payload, Mapping):
        raise DiracHartreeBridgeError("payload must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise DiracHartreeBridgeError(f"unexpected schema_version: {payload.get('schema_version')!r}")
    if payload.get("status") != STATUS:
        raise DiracHartreeBridgeError(f"unexpected status: {payload.get('status')!r}")
    ev = _require_finite_float(payload.get("evidence_weight"), "evidence_weight")
    if ev != 0.0:
        raise DiracHartreeBridgeError(f"evidence_weight must be 0.0, got {ev}")
    if check_source_sha:
        expected_sha = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
        if payload.get("source_sha256") != expected_sha:
            raise DiracHartreeBridgeError("source_sha256 does not match current producer script")

    cal = payload.get("calibration_targets")
    if not isinstance(cal, Mapping) or cal.get("finite_nuclei_used_in_calibration") is not False:
        raise DiracHartreeBridgeError("calibration_targets must explicitly declare finite_nuclei_used_in_calibration=False")

    fp = normalize_params(payload.get("frozen_params", {}))
    if fp["lam_W"] <= 0.0:
        raise DiracHartreeBridgeError("heavy gluon potential must be bounded below (lam_W <= 0)")
    _check_chiral_potential_bounded_below(fp["m_s"], fp["b_s"], fp["c_s"], fp["d_s"])

    bench = payload.get("analytical_dirac_benchmarks")
    if not isinstance(bench, Mapping) or not bench.get("pass") or bench.get("spurious_states_in_mass_gap") != 0:
        raise DiracHartreeBridgeError("analytical Dirac benchmarks failed or reported spurious gap states")

    fwd = payload.get("forward_predictions")
    if not isinstance(fwd, Mapping) or not isinstance(fwd.get("nuclei"), Mapping):
        raise DiracHartreeBridgeError("missing forward_predictions.nuclei")
    nuclei = fwd["nuclei"]
    for name, z_exp, n_exp in DOUBLY_MAGIC_NUCLEI:
        if name not in nuclei:
            raise DiracHartreeBridgeError(f"missing required nucleus {name}")
        row = nuclei[name]
        if row.get("Z") != z_exp or row.get("N") != n_exp or row.get("A") != z_exp + n_exp:
            raise DiracHartreeBridgeError(f"wrong Z/N/A bookkeeping for {name}")
        conv = row.get("convergence", {})
        if not conv.get("converged"):
            raise DiracHartreeBridgeError(f"{name} did not converge")
        fres_max = _require_finite_float(conv.get("field_residuals", {}).get("max_rel"), f"{name}.field_residual")
        if fres_max >= FIELD_RESIDUAL_LIMIT:
            raise DiracHartreeBridgeError(f"{name} field residual {fres_max:.3e} >= {FIELD_RESIDUAL_LIMIT:.1e}")
        norm_err = _require_finite_float(conv.get("max_orbital_norm_error"), f"{name}.norm_err")
        if norm_err > ORBITAL_NORM_LIMIT:
            raise DiracHartreeBridgeError(f"{name} orbital norm error {norm_err:.3e} > {ORBITAL_NORM_LIMIT:.1e}")

        pnum = row.get("particle_numbers", {})
        z_err = _require_finite_float(pnum.get("Z_rel_err"), f"{name}.Z_rel_err")
        n_err = _require_finite_float(pnum.get("N_rel_err"), f"{name}.N_rel_err")
        if z_err > PARTICLE_NUMBER_REL_LIMIT or n_err > PARTICLE_NUMBER_REL_LIMIT:
            raise DiracHartreeBridgeError(f"{name} particle number error exceeded limit: Z={z_err:.3e}, N={n_err:.3e}")

        en = row.get("energies", {})
        ba_eig = _require_finite_float(en.get("B_over_A_eig_MeV"), f"{name}.B_over_A_eig")
        ba_dir = _require_finite_float(en.get("B_over_A_direct_MeV"), f"{name}.B_over_A_direct")
        diff_a = _require_finite_float(en.get("energy_formula_diff_per_A_MeV"), f"{name}.energy_diff")
        if ba_eig >= 0.0 or ba_dir >= 0.0:
            raise DiracHartreeBridgeError(f"{name} is unbound (B/A={ba_eig})")
        if abs(ba_eig - ba_dir) > ENERGY_IDENTITY_PER_A_LIMIT_MEV or diff_a > ENERGY_IDENTITY_PER_A_LIMIT_MEV:
            raise DiracHartreeBridgeError(f"{name} two-energy-formula identity violated: diff={diff_a:.3e} MeV/A")

        rad = row.get("radii", {})
        r_p = _require_finite_float(rad.get("R_p_fm"), f"{name}.R_p")
        r_n = _require_finite_float(rad.get("R_n_fm"), f"{name}.R_n")
        r_ch = _require_finite_float(rad.get("R_ch_fm"), f"{name}.R_ch")
        skin = _require_finite_float(rad.get("Delta_R_np_fm"), f"{name}.Delta_R_np")
        if abs(r_ch - math.hypot(r_p, PROTON_CHARGE_RADIUS_FM)) > 1.0e-10:
            raise DiracHartreeBridgeError(f"{name} R_ch inconsistent with R_p")
        if abs(skin - (r_n - r_p)) > 1.0e-10:
            raise DiracHartreeBridgeError(f"{name} Delta_R_np inconsistent with R_n - R_p")

    # Physical sign checks for neutron skins and spin-orbit splittings
    skin_ca48 = _require_finite_float(nuclei["Ca48"]["radii"]["Delta_R_np_fm"], "Ca48.skin")
    skin_pb208 = _require_finite_float(nuclei["Pb208"]["radii"]["Delta_R_np_fm"], "Pb208.skin")
    if skin_ca48 <= 0.0 or skin_pb208 <= 0.0:
        raise DiracHartreeBridgeError(f"neutron skins must be positive: Ca48={skin_ca48}, Pb208={skin_pb208}")

    so_o16 = _require_finite_float(nuclei["O16"]["spin_orbit_splittings_MeV"]["1p"]["neutron_MeV"], "O16.SO_1p")
    so_ca40 = _require_finite_float(nuclei["Ca40"]["spin_orbit_splittings_MeV"]["1d"]["neutron_MeV"], "Ca40.SO_1d")
    so_ca48 = _require_finite_float(nuclei["Ca48"]["spin_orbit_splittings_MeV"]["1f"]["neutron_MeV"], "Ca48.SO_1f")
    if so_o16 <= 0.0 or so_ca40 <= 0.0 or so_ca48 <= 0.0:
        raise DiracHartreeBridgeError(f"spin-orbit splittings must be positive: {so_o16}, {so_ca40}, {so_ca48}")

    summary = fwd.get("summary", {})
    d_rch = _require_finite_float(summary.get("Delta_R_ch_Ca48_minus_Ca40_fm"), "summary.Delta_R_ch")
    expected_d_rch = nuclei["Ca48"]["radii"]["R_ch_fm"] - nuclei["Ca40"]["radii"]["R_ch_fm"]
    if abs(d_rch - expected_d_rch) > 1.0e-10:
        raise DiracHartreeBridgeError("summary Delta_R_ch mismatch")

    gchk = payload.get("grid_refinement_check")
    if gchk is not None:
        if not isinstance(gchk, Mapping) or not gchk.get("pass"):
            raise DiracHartreeBridgeError("grid_refinement_check failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write canonical JSON result to this path",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Validate an existing JSON result fail-closed (writes no files)",
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help="Load frozen Law 2 parameters from JSON with zero refitting",
    )
    parser.add_argument(
        "--no-grid-check",
        action="store_true",
        help="Skip grid-refinement check on Ca48",
    )
    args = parser.parse_args(argv)

    if args.validate is not None:
        try:
            raw_text = args.validate.read_text(encoding="utf-8")
            loaded = json.loads(raw_text)
            validate_payload(loaded, check_source_sha=True)
        except (OSError, json.JSONDecodeError, DiracHartreeBridgeError) as exc:
            print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
            return 1
        print("VALIDATION_PASS")
        return 0

    try:
        if args.params_json is not None:
            ext_params = load_params_from_json(args.params_json)
            result = build_result(
                params_override=ext_params,
                params_source=str(args.params_json),
                run_grid_check=not args.no_grid_check,
            )
        else:
            result = build_result(run_grid_check=not args.no_grid_check)
        canonical = _canonical_json_bytes(result)
        if args.output is not None:
            args.output.write_bytes(canonical)
        else:
            sys.stdout.buffer.write(canonical)
    except DiracHartreeBridgeError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
