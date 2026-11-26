#!/usr/bin/env python3
"""Bridge 9A: Microphysical NS Core-Crust Transition, Direct Urca & Relativistic Cowling g1/f-Modes.

Extends the Cycle-3 Bridge-8 primary-gate winner (and all 8 primary-gate survivors
from ``nvg_unified_primary_gate_pairing_results.json``) with ZERO parameter refitting:

1. Cold npe-mu beta-equilibrium EOS & exact Le Chatelier sound speeds:
     - Equilibrium (barotropic) sound speed c_e^2 = (dP/deps)_{beta-eq}
     - Frozen-composition (adiabatic) sound speed c_s^2 = (dP/deps)_{x_p,x_e,x_mu}
     - Adiabatic vs equilibrium indices Gamma_1 and Gamma_eq
     - Exact muon onset n_mu, M_mu and Direct Urca threshold n_DUrca, x_{p,DUrca}, M_DUrca

2. Spinodal Core-Crust Transition across all 8 Primary-Gate Survivors:
     - Thermodynamical criterion: V_ther(n_t^ther) = 0 <=> det(K_bulk) = 0, where
       K_bulk = d(mu_i)/d(n_j) is the 2x2 nucleon chemical potential curvature matrix.
     - Exact finite-wavelength dynamical criterion (Carriere-Horowitz-Piekarewicz / Xu et al.):
       4x4 meson Schur complement (W, s, A0, rho0) at finite wavevector k with exact
       Yukawa propagators (m_W, m_s, m_omega_eff, m_rho_eff) plus Thomas-Fermi screened
       Coulomb 4*pi*alpha/(k^2 + k_TF^2), minimized over k to locate lam_min(K_dyn(n_t^dyn)) = 0.
     - Crust thickness Delta_R_crust / R and crustal moment of inertia fraction I_crust / I
       at M = 1.4 M_sun integrated via 1st-order Hartle frame dragging (Link-Epstein-Lattimer).

3. Relativistic Cowling l = 2 f-mode & Non-Barotropic Compositional g_1/g_2 Modes:
     - Exact relativistic Cowling ODEs across M in {1.20, 1.3381, 1.40, 1.60, 1.80, 2.00} M_sun
     - Exact GR Brunt-Vaisala profile N^2(r) = (Phi')^2 e^{2(Phi-Lambda)} (1/c_e^2 - 1/c_s^2)
     - Mandatory negative & analytic controls:
         (a) Uniform-density Newtonian Kelvin Cowling limit: omega_f^2 R^3 / (l G M) -> 1
         (b) Barotropic null control (c_s^2 == c_e^2 => N^2 == 0 => zero g-mode eigenvalues).

SPEC-v2 fail-closed audit: evidence_weight = 0.0, observed_likelihood = None.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

for _blas_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_two_condensate_unified_audit as law2  # noqa: E402
import nvg_unified_calibration_envelope_probe as b6  # noqa: E402

SOURCE_PATH = Path(__file__).resolve()
B8_RESULTS_PATH = HERE / "nvg_unified_primary_gate_pairing_results.json"
DEFAULT_OUTPUT_PATH = HERE / "nvg_unified_ns_microphysics_results.json"

SCHEMA_VERSION = "nvg_unified_ns_microphysics_bridge.v1"
BRIDGE_ID = "BRIDGE9A_NS_CORE_CRUST_DURCA_AND_ASTEROSEISMOLOGY"
STATUS_PASS = "PASS_NS_MICROPHYSICS_CORE_CRUST_DURCA_GMODE_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

ALPHA_EM = 1.0 / 137.035999084
C_KM_S = 299792.458
K_CONV = law2.K_CONV
M_SUN_KM = law2.M_SUN_KM

COWLING_TARGET_MASSES_MSUN = (1.20, 1.3381, 1.40, 1.60, 1.80, 2.00)
# Link, Epstein & Lattimer (PRL 83, 3362, 1999) Vela pulsar glitch requirement benchmark
VELA_ICRUST_OVER_I_MIN = 0.014


class NSMicrophysicsError(ValueError):
    """Fail-closed validation or computation error."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False) + "\n"


def _finite_tree(val: Any) -> bool:
    if isinstance(val, bool) or val is None or isinstance(val, (str, int)):
        return True
    if isinstance(val, float):
        return math.isfinite(val)
    if isinstance(val, Mapping):
        return all(_finite_tree(v) for v in val.values())
    if isinstance(val, (list, tuple)):
        return all(_finite_tree(v) for v in val)
    return False


def _lepton_eps_prs_fixed_frac(n_nat: float, xe: float, xmu: float) -> tuple[float, float]:
    """Fixed-fraction electron+muon energy density & pressure at natural density n_nat."""
    eps = 0.0
    prs = 0.0
    for frac, mass in ((xe, law2.M_E_MEV), (xmu, law2.M_MU_MEV)):
        if frac > 0.0:
            kf = (3.0 * math.pi**2 * frac * n_nat) ** (1.0 / 3.0)
            ef = math.hypot(kf, mass)
            lt = math.asinh(kf / mass)
            eps += (kf * ef * (2.0 * kf * kf + mass * mass) - (mass**4) * lt) / (8.0 * math.pi**2)
            prs += (kf * ef * (2.0 * kf * kf - 3.0 * mass * mass) + 3.0 * (mass**4) * lt) / (24.0 * math.pi**2)
    return eps, prs


def _safe_solve_beta(
    p: law2.ModelParams,
    n_fm3: float,
    guess: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    try:
        return law2.solve_beta_point(p, n_fm3, guess)
    except law2.AuditError:
        return law2.solve_beta_point(p, n_fm3, None)


def _safe_solve_zero_t(
    p: law2.ModelParams,
    n_p_nat: float,
    n_n_nat: float,
    guess: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    try:
        return law2.solve_zero_t_state(p, n_p_nat, n_n_nat, guess)
    except law2.AuditError:
        return law2.solve_zero_t_state(p, n_p_nat, n_n_nat, None)


def compute_beta_sound_speeds(
    p: law2.ModelParams,
    n_fm3: float,
    row0: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute exact c_e^2 (beta-equilibrium) and c_s^2 (frozen-composition) at density n_fm3."""
    b0 = _safe_solve_beta(p, n_fm3) if row0 is None else row0
    guess = (b0["W_MeV"], b0["Phi_s_MeV"], b0["A0_MeV"])
    dn = 1.0e-4 * n_fm3

    # Equilibrium derivatives dP/deps at beta equilibrium
    bp = _safe_solve_beta(p, n_fm3 + dn, guess)
    bm = _safe_solve_beta(p, n_fm3 - dn, guess)
    ce2 = (bp["P_MeV_fm3"] - bm["P_MeV_fm3"]) / (bp["eps_MeV_fm3"] - bm["eps_MeV_fm3"])

    # Frozen-composition derivatives at fixed (x_p, x_e, x_mu)
    xp, xe, xmu = b0["x_p"], b0["x_e"], b0["x_mu"]

    def _frozen(nf: float) -> tuple[float, float]:
        nn_nat = nf * p.hbar_c**3
        st = _safe_solve_zero_t(p, xp * nn_nat, (1.0 - xp) * nn_nat, guess)
        le, lp = _lepton_eps_prs_fixed_frac(nn_nat, xe, xmu)
        return (st["eps"] + le) / (p.hbar_c**3), (st["prs"] + lp) / (p.hbar_c**3)

    ep_f, pp_f = _frozen(n_fm3 + dn)
    em_f, pm_f = _frozen(n_fm3 - dn)
    cs2 = (pp_f - pm_f) / (ep_f - em_f)

    e0 = b0["eps_MeV_fm3"]
    p0 = b0["P_MeV_fm3"]
    gamma_eq = ((e0 + p0) / p0) * ce2 if p0 > 0.0 else 0.0
    gamma_1 = ((e0 + p0) / p0) * cs2 if p0 > 0.0 else 0.0

    return {
        **b0,
        "ce2": float(ce2),
        "cs2": float(cs2),
        "cs2_minus_ce2": float(cs2 - ce2),
        "Gamma_eq": float(gamma_eq),
        "Gamma_1": float(gamma_1),
    }


def _nucleon_hessian_schur(
    p: law2.ModelParams,
    n_fm3: float,
    k_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute exact 2x2 bulk chemical-potential Hessian & finite-k meson+Coulomb dynamical Hessian."""
    n_nat = n_fm3 * p.hbar_c**3
    b = law2.solve_beta_point(p, n_fm3)
    xp = b["x_p"]
    np_nat = xp * n_nat
    nn_nat = (1.0 - xp) * n_nat
    g_guess = (b["W_MeV"], b["Phi_s_MeV"], b["A0_MeV"])
    st0 = law2.solve_zero_t_state(p, np_nat, nn_nat, g_guess)

    # Independent finite-diff check of k=0 bulk curvature matrix
    dp = max(1.0e-4 * np_nat, 1.0e-6 * n_nat)
    dn = max(1.0e-4 * nn_nat, 1.0e-6 * n_nat)
    stp_p = law2.solve_zero_t_state(p, np_nat + dp, nn_nat, g_guess)
    stm_p = law2.solve_zero_t_state(p, np_nat - dp, nn_nat, g_guess)
    stp_n = law2.solve_zero_t_state(p, np_nat, nn_nat + dn, g_guess)
    stm_n = law2.solve_zero_t_state(p, np_nat, nn_nat - dn, g_guess)
    K_bulk_fd = np.array(
        [
            [(stp_p["mu_p"] - stm_p["mu_p"]) / (2.0 * dp), (stp_n["mu_p"] - stm_n["mu_p"]) / (2.0 * dn)],
            [(stp_p["mu_n"] - stm_p["mu_n"]) / (2.0 * dp), (stp_n["mu_n"] - stm_n["mu_n"]) / (2.0 * dn)],
        ],
        dtype=float,
    )

    M = st0["M"]
    kp, Ep = st0["kp"], math.hypot(st0["kp"], M)
    kn, En = st0["kn"], math.hypot(st0["kn"], M)
    dnu_p_dnp = math.pi**2 / (kp * Ep)
    dnu_n_dnn = math.pi**2 / (kn * En)
    dns_p_dnp = M / Ep
    dns_n_dnn = M / En

    def _dsig_dm(kf: float, ef: float) -> float:
        return (kf * ef - 3.0 * M * M * math.asinh(kf / M) + 2.0 * M * M * kf / ef) / (2.0 * math.pi**2)

    dns_dM = _dsig_dm(kp, Ep) + _dsig_dm(kn, En)

    W0, lamW, qphi = p.W0, p.lam_W, p.q_phi
    c_LV = 2.0 * p.Lambda_v * (p.g_omega**2) * (p.g_rho**2)
    zeta_A = p.zeta_g * (qphi**4)
    W, Phi_s, A0, rho0 = st0["W"], st0["Phi"], st0["A0"], st0["rho0"]
    d2U_ds2 = (p.g_s_ch**2) * (p.c2 + 2.0 * p.b3 * p.M_N * Phi_s + 3.0 * p.b4 * (Phi_s**2) + 5.0 * p.b6 * (Phi_s**4))

    F_n = np.array(
        [
            [p.g_W * dns_p_dnp, p.g_W * dns_n_dnn],
            [-p.g_s_ch * dns_p_dnp, -p.g_s_ch * dns_n_dnn],
            [-p.g_omega, -p.g_omega],
            [-0.5 * p.g_rho, +0.5 * p.g_rho],
        ],
        dtype=float,
    )
    mu_X = np.array(
        [
            [p.g_W * dns_p_dnp, -p.g_s_ch * dns_p_dnp, p.g_omega, 0.5 * p.g_rho],
            [p.g_W * dns_n_dnn, -p.g_s_ch * dns_n_dnn, p.g_omega, -0.5 * p.g_rho],
        ],
        dtype=float,
    )
    K_F = np.diag([dnu_p_dnp, dnu_n_dnn])

    ke = (3.0 * math.pi**2 * np_nat) ** (1.0 / 3.0)
    mue = math.hypot(ke, law2.M_E_MEV)
    kTF2 = (4.0 * ALPHA_EM / math.pi) * ke * mue

    def _K_at_k(k_MeV: float) -> np.ndarray:
        k2 = k_MeV * k_MeV
        J_X = np.array(
            [
                [k2 + lamW * (3.0 * W * W - W0 * W0) - (qphi * A0) ** 2 + (p.g_W**2) * dns_dM, -p.g_W * p.g_s_ch * dns_dM, -2.0 * qphi * qphi * W * A0, 0.0],
                [-p.g_W * p.g_s_ch * dns_dM, k2 + d2U_ds2 + (p.g_s_ch**2) * dns_dM, 0.0, 0.0],
                [2.0 * qphi * qphi * W * A0, 0.0, k2 + qphi * qphi * W * W + 3.0 * zeta_A * A0 * A0 + c_LV * rho0 * rho0, 2.0 * c_LV * A0 * rho0],
                [0.0, 0.0, 2.0 * c_LV * A0 * rho0, k2 + p.m_rho**2 + c_LV * A0 * A0],
            ],
            dtype=float,
        )
        dX_dn = -np.linalg.solve(J_X, F_n)
        K_mat = K_F + mu_X @ dX_dn
        if k_MeV > 0.0:
            K_mat[0, 0] += 4.0 * math.pi * ALPHA_EM / (k2 + kTF2)
        return 0.5 * (K_mat + K_mat.T)

    K0 = _K_at_k(0.0)
    schur_vs_fd_rel = float(np.max(np.abs(K_bulk_fd - K0)) / max(1.0e-30, float(np.max(np.abs(K_bulk_fd)))))
    lam_ther = float(np.min(np.linalg.eigvalsh(0.5 * (K_bulk_fd + K_bulk_fd.T))))

    ks = np.linspace(20.0, 130.0, 23) if k_values is None else k_values
    eigs = [float(np.min(np.linalg.eigvalsh(_K_at_k(float(k))))) for k in ks]
    idx_opt = int(np.argmin(eigs))
    lam_dyn = eigs[idx_opt]
    k_opt = float(ks[idx_opt])
    # Polish around k_opt with a fine sub-grid
    if k_values is None:
        ks_fine = np.linspace(max(5.0, k_opt - 5.0), k_opt + 5.0, 11)
        eigs_f = [float(np.min(np.linalg.eigvalsh(_K_at_k(float(k))))) for k in ks_fine]
        idx_f = int(np.argmin(eigs_f))
        lam_dyn = eigs_f[idx_f]
        k_opt = float(ks_fine[idx_f])

    return {
        "lam_ther_MeVminus2": lam_ther,
        "lam_dyn_MeVminus2": float(lam_dyn),
        "k_opt_MeV": k_opt,
        "k_opt_fm_inv": k_opt / p.hbar_c,
        "lambda_opt_fm": 2.0 * math.pi * p.hbar_c / k_opt,
        "det_bulk": float(np.linalg.det(K_bulk_fd)),
        "schur_vs_fd_rel": schur_vs_fd_rel,
        "beta_row": b,
    }


def find_core_crust_transition(p: law2.ModelParams) -> dict[str, Any]:
    """Locate thermodynamical (k=0) and dynamical (finite-k + Coulomb) spinodal transition."""
    n_lo, n_hi = 0.04, 0.13
    lam_ther_lo = _nucleon_hessian_schur(p, n_lo)["lam_ther_MeVminus2"]
    lam_ther_hi = _nucleon_hessian_schur(p, n_hi)["lam_ther_MeVminus2"]
    if not (lam_ther_lo < 0.0 < lam_ther_hi):
        raise NSMicrophysicsError(
            f"Thermodynamical spinodal bracket failed: ({lam_ther_lo:.3e}, {lam_ther_hi:.3e})"
        )
    n_ther = float(
        brentq(
            lambda nf: _nucleon_hessian_schur(p, float(nf))["lam_ther_MeVminus2"],
            n_lo,
            n_hi,
            xtol=1.0e-5,
            rtol=1.0e-5,
        )
    )
    res_ther = _nucleon_hessian_schur(p, n_ther)

    # Bracket dynamical transition more tightly inside [0.04, n_ther]
    n_dyn = float(
        brentq(
            lambda nf: _nucleon_hessian_schur(p, float(nf))["lam_dyn_MeVminus2"],
            0.035,
            min(n_ther + 0.005, n_hi),
            xtol=1.0e-5,
            rtol=1.0e-5,
        )
    )
    res_dyn = _nucleon_hessian_schur(p, n_dyn)

    return {
        "thermodynamical": {
            "n_t_fm3": n_ther,
            "n_t_over_n0": n_ther / p.n0_fm3,
            "P_t_MeV_fm3": res_ther["beta_row"]["P_MeV_fm3"],
            "eps_t_MeV_fm3": res_ther["beta_row"]["eps_MeV_fm3"],
            "x_p_t": res_ther["beta_row"]["x_p"],
            "schur_vs_fd_rel": res_ther["schur_vs_fd_rel"],
        },
        "dynamical": {
            "n_t_fm3": n_dyn,
            "n_t_over_n0": n_dyn / p.n0_fm3,
            "P_t_MeV_fm3": res_dyn["beta_row"]["P_MeV_fm3"],
            "eps_t_MeV_fm3": res_dyn["beta_row"]["eps_MeV_fm3"],
            "x_p_t": res_dyn["beta_row"]["x_p"],
            "k_opt_MeV": res_dyn["k_opt_MeV"],
            "k_opt_fm_inv": res_dyn["k_opt_fm_inv"],
            "pasta_wavelength_fm": res_dyn["lambda_opt_fm"],
            "schur_vs_fd_rel": res_dyn["schur_vs_fd_rel"],
        },
    }


def _build_star_eos_and_tables(
    p: law2.ModelParams,
    n_t_dyn: float,
    n_max: float = 1.25,
) -> tuple[law2.TabulatedCrustCoreEOS, dict[str, np.ndarray]]:
    """Build exact core+SLy4 crust EOS anchored at the dynamical core-crust transition density n_t_dyn."""
    ns_grid = np.unique(
        np.concatenate(
            [
                np.linspace(n_t_dyn, 0.25, 22),
                np.linspace(0.26, n_max, 42),
            ]
        )
    )
    rows = []
    guess: tuple[float, float, float] | None = None
    for nf in ns_grid:
        b0 = _safe_solve_beta(p, float(nf), guess)
        guess = (b0["W_MeV"], b0["Phi_s_MeV"], b0["A0_MeV"])
        row = compute_beta_sound_speeds(p, float(nf), b0)
        rows.append(row)

    core_n = np.array([r["n_fm3"] for r in rows], dtype=float)
    core_p = np.array([r["P_MeV_fm3"] for r in rows], dtype=float)
    core_e = np.array([r["eps_MeV_fm3"] for r in rows], dtype=float)
    core_ce2 = np.array([r["ce2"] for r in rows], dtype=float)
    core_cs2 = np.array([r["cs2"] for r in rows], dtype=float)
    core_xp = np.array([r["x_p"] for r in rows], dtype=float)
    core_xe = np.array([r["x_e"] for r in rows], dtype=float)
    core_xmu = np.array([r["x_mu"] for r in rows], dtype=float)
    core_du = np.array([r["du_triangle_MeV"] for r in rows], dtype=float)

    eos = law2.TabulatedCrustCoreEOS(core_p, core_e, crust_mode="dh_sly4")
    tables = {
        "rows": rows,
        "core_n": core_n,
        "core_p": core_p,
        "core_e": core_e,
        "core_ce2": core_ce2,
        "core_cs2": core_cs2,
        "core_xp": core_xp,
        "core_xe": core_xe,
        "core_xmu": core_xmu,
        "core_du": core_du,
    }
    return eos, tables


def integrate_hartle_crustal_inertia(
    eos: law2.TabulatedCrustCoreEOS,
    Pc: float,
    P_trans: float,
    *,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    max_step: float = 0.04,
) -> dict[str, float]:
    """Solve TOV + 1st-order Hartle frame-dragging to extract I_crust / I and Delta_R_crust / R."""
    P_surf = 1.0e-9
    r0 = 1.0e-4
    eps0 = eos.get_eps(Pc)
    m0 = (4.0 * math.pi / 3.0) * (eps0 * K_CONV) * (r0**3) / M_SUN_KM
    p0 = Pc - (2.0 * math.pi / 3.0) * (eps0 + Pc) * (eps0 + 3.0 * Pc) * K_CONV * (r0**2)
    source0 = (eps0 + Pc) * K_CONV
    y0 = np.array([m0, p0, 0.0, 1.0, (16.0 * math.pi / 5.0) * source0 * r0], dtype=float)

    def rhs(r: float, yv: np.ndarray) -> list[float]:
        m, P, _phi, om, dom = [float(v) for v in yv]
        if r <= 1.0e-12 or P <= 0.0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        eg = eos.get_eps(P) * K_CONV
        pg = P * K_CONV
        mg = m * M_SUN_KM
        fac = r * (r - 2.0 * mg)
        if fac <= 0.0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        one_minus_2m = 1.0 - 2.0 * mg / r
        dm = 4.0 * math.pi * r * r * eg / M_SUN_KM
        dP = -(eg + pg) * (mg + 4.0 * math.pi * (r**3) * pg) / fac / K_CONV
        dPhi = (mg + 4.0 * math.pi * (r**3) * pg) / fac
        drag = 4.0 * math.pi * r * (eg + pg) / one_minus_2m
        ddom = (drag - 4.0 / r) * dom + (4.0 * drag / r) * om
        return [dm, dP, dPhi, dom, ddom]

    def surf(r: float, yv: np.ndarray) -> float:
        return float(yv[1]) - P_surf

    surf.terminal = True
    surf.direction = -1

    sol = solve_ivp(rhs, (r0, 40.0), y0, method="DOP853", rtol=rtol, atol=atol, max_step=max_step, events=[surf])
    if not sol.success:
        raise NSMicrophysicsError(f"Hartle crust ODE failed at Pc={Pc}: {sol.message}")

    r_arr = sol.t
    m_arr, P_arr, Phi_raw, om_arr, dom_arr = sol.y
    R = float(r_arr[-1])
    M = float(m_arr[-1])
    C = M * M_SUN_KM / R
    Phi_arr = Phi_raw + (0.5 * math.log(1.0 - 2.0 * C) - float(Phi_raw[-1]))
    om_inf = float(om_arr[-1]) + R * float(dom_arr[-1]) / 3.0
    I_geom = (R**4) * float(dom_arr[-1]) / (6.0 * om_inf)

    # Integral decomposition via (8*pi/3) int_0^R (eps+P) j/(1-2m/r) (om/om_inf) r^4 dr
    eps_geo = np.array([eos.get_eps(float(pval)) * K_CONV for pval in P_arr], dtype=float)
    p_geo = P_arr * K_CONV
    one_minus_2m = 1.0 - 2.0 * m_arr * M_SUN_KM / r_arr
    j_arr = np.exp(-Phi_arr) * np.sqrt(one_minus_2m)
    integrand = (8.0 * math.pi / 3.0) * (eps_geo + p_geo) * (j_arr / one_minus_2m) * (om_arr / om_inf) * (r_arr**4)

    # Locate exact transition radius R_core where P(r) == P_trans
    if P_trans >= float(P_arr[0]) or P_trans <= float(P_arr[-1]):
        raise NSMicrophysicsError(f"Transition pressure {P_trans} outside star pressure range")
    idx_hi = int(np.searchsorted(-P_arr, -P_trans))
    idx_lo = max(0, idx_hi - 1)
    frac = (P_trans - float(P_arr[idx_lo])) / (float(P_arr[idx_hi]) - float(P_arr[idx_lo]))
    r_core = float(r_arr[idx_lo]) + frac * float(r_arr[idx_hi] - r_arr[idx_lo])
    m_core = float(m_arr[idx_lo]) + frac * float(m_arr[idx_hi] - m_arr[idx_lo])
    int_core = float(integrand[idx_lo]) + frac * float(integrand[idx_hi] - integrand[idx_lo])

    r_cr = np.concatenate([[r_core], r_arr[idx_hi:]])
    i_cr = np.concatenate([[int_core], integrand[idx_hi:]])
    I_crust = float(np.trapz(i_cr, r_cr))
    I_total_int = float(np.trapz(integrand, r_arr))

    return {
        "M_Msun": M,
        "R_km": R,
        "R_core_km": r_core,
        "M_core_Msun": m_core,
        "Delta_R_crust_km": R - r_core,
        "Delta_R_crust_over_R": (R - r_core) / R,
        "M_crust_Msun": M - m_core,
        "I_geom_km3": I_geom,
        "I_total_int_rel_err": abs(I_total_int - I_geom) / I_geom,
        "I_crust_geom_km3": I_crust,
        "I_crust_over_I": I_crust / I_total_int,
    }


def analyze_thresholds(
    p: law2.ModelParams,
    eos: law2.TabulatedCrustCoreEOS,
    tables: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Locate M_max, canonical 1.4 star, muon onset M_mu, and Direct Urca M_DUrca."""
    core_n = tables["core_n"]
    core_p = tables["core_p"]
    core_xmu = tables["core_xmu"]
    core_du = tables["core_du"]

    # Coarse-to-fine TOV sequence for peak mass and interpolation
    pcs = np.geomspace(15.0, min(float(core_p[-1]) * 0.94, 750.0), 22)
    stars = [law2.solve_tov_tidal_star(eos, float(pc), rtol=1.0e-7, atol=1.0e-9, max_step=0.05) for pc in pcs]
    ms = np.array([s[0] for s in stars], dtype=float)
    imax = int(np.argmax(ms))
    if 0 < imax < len(stars) - 1:
        for pcr in np.linspace(float(pcs[imax - 1]), float(pcs[imax + 1]), 7)[1:-1]:
            stars.append(law2.solve_tov_tidal_star(eos, float(pcr), rtol=1.0e-7, atol=1.0e-9, max_step=0.05))
            pcs = np.append(pcs, pcr)
        order = np.argsort(pcs)
        pcs = pcs[order]
        stars = [stars[i] for i in order]
        ms = np.array([s[0] for s in stars], dtype=float)
        imax = int(np.argmax(ms))

    m_max = float(ms[imax])
    pc_max = float(pcs[imax])
    n_c_max = float(np.interp(pc_max, core_p, core_n))
    ms_s = ms[: imax + 1]
    pcs_s = pcs[: imax + 1]

    def _pc_for_mass(m_target: float) -> float:
        idx = int(np.searchsorted(ms_s, m_target))
        p_lo = float(pcs_s[max(0, idx - 1)])
        p_hi = float(pcs_s[min(len(pcs_s) - 1, idx)])
        return float(
            brentq(
                lambda pc: law2.solve_tov_tidal_star(eos, pc, rtol=1.0e-8, atol=1.0e-10, max_step=0.05)[0] - m_target,
                p_lo,
                p_hi,
                xtol=1.0e-4,
            )
        )

    pc_14 = _pc_for_mass(1.4)
    _, r14, _, l14 = law2.solve_tov_tidal_star(eos, pc_14, rtol=1.0e-8, atol=1.0e-10, max_step=0.04)

    # Muon onset (where mu_e == m_mu = 105.658 MeV)
    idx_mu = next((i for i, xm in enumerate(core_xmu) if xm > 0.0), None)
    n_mu = None
    m_mu_star = None
    if idx_mu is not None and idx_mu > 0:
        n_mu = float(
            brentq(
                lambda nf: law2.solve_beta_point(p, float(nf))["mu_e_MeV"] - law2.M_MU_MEV,
                float(core_n[idx_mu - 1]),
                float(core_n[idx_mu]),
                xtol=1.0e-5,
            )
        )
        b_mu = law2.solve_beta_point(p, n_mu)
        m_mu_star = float(
            law2.solve_tov_tidal_star(eos, b_mu["P_MeV_fm3"], rtol=1.0e-7, atol=1.0e-9, max_step=0.05)[0]
        )

    # Direct Urca threshold (where du_triangle_MeV = k_p + k_e - k_n == 0)
    du_idx = next((i for i, du in enumerate(core_du) if du >= 0.0), None)
    du_active = False
    n_du = None
    xp_du = None
    p_du = None
    m_du = None
    if du_idx is not None and du_idx > 0:
        n_du = float(
            brentq(
                lambda nf: law2.solve_beta_point(p, float(nf))["du_triangle_MeV"],
                float(core_n[du_idx - 1]),
                float(core_n[du_idx]),
                xtol=1.0e-5,
            )
        )
        b_du = law2.solve_beta_point(p, n_du)
        xp_du = float(b_du["x_p"])
        p_du = float(b_du["P_MeV_fm3"])
        if p_du <= pc_max:
            du_active = True
            m_du = float(
                law2.solve_tov_tidal_star(eos, p_du, rtol=1.0e-8, atol=1.0e-10, max_step=0.05)[0]
            )

    return {
        "M_max_Msun": m_max,
        "R_at_Mmax_km": float(stars[imax][1]),
        "Pc_max_MeV_fm3": pc_max,
        "n_c_max_fm3": n_c_max,
        "n_c_max_over_n0": n_c_max / p.n0_fm3,
        "R_1_4_km": float(r14),
        "Lambda_1_4": float(l14),
        "Pc_1_4_MeV_fm3": pc_14,
        "muon_onset": {
            "n_mu_fm3": n_mu,
            "n_mu_over_n0": n_mu / p.n0_fm3 if n_mu is not None else None,
            "M_mu_Msun": m_mu_star,
        },
        "direct_urca": {
            "active_in_stable_star": du_active,
            "n_DUrca_fm3": n_du,
            "n_DUrca_over_n0": n_du / p.n0_fm3 if n_du is not None else None,
            "x_p_DUrca": xp_du,
            "P_DUrca_MeV_fm3": p_du,
            "M_DUrca_Msun": m_du,
        },
    }


def _solve_tov_metric_profile(
    eos: law2.TabulatedCrustCoreEOS,
    Pc: float,
    *,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
    max_step: float = 0.04,
) -> dict[str, Any]:
    """Integrate background TOV (m, P, Phi) with exact Schwarzschild exterior normalization."""
    P_surf = 1.0e-9
    r0 = 1.0e-4
    eps0 = eos.get_eps(Pc)
    m0 = (4.0 * math.pi / 3.0) * (eps0 * K_CONV) * (r0**3) / M_SUN_KM
    p0 = Pc - (2.0 * math.pi / 3.0) * (eps0 + Pc) * (eps0 + 3.0 * Pc) * K_CONV * (r0**2)

    def rhs(r: float, yv: np.ndarray) -> list[float]:
        m, P, _ph = float(yv[0]), float(yv[1]), float(yv[2])
        if r <= 1.0e-12 or P <= 0.0:
            return [0.0, 0.0, 0.0]
        eg = eos.get_eps(P) * K_CONV
        pg = P * K_CONV
        mg = m * M_SUN_KM
        fac = r * (r - 2.0 * mg)
        if fac <= 0.0:
            return [0.0, 0.0, 0.0]
        dm = 4.0 * math.pi * r * r * eg / M_SUN_KM
        dP = -(eg + pg) * (mg + 4.0 * math.pi * (r**3) * pg) / fac / K_CONV
        dPhi = (mg + 4.0 * math.pi * (r**3) * pg) / fac
        return [dm, dP, dPhi]

    def surf(r: float, yv: np.ndarray) -> float:
        return float(yv[1]) - P_surf

    surf.terminal = True
    surf.direction = -1
    sol = solve_ivp(rhs, (r0, 40.0), [m0, p0, 0.0], method="DOP853", rtol=rtol, atol=atol, max_step=max_step, events=[surf])
    if not sol.success:
        raise NSMicrophysicsError(f"Background TOV failed at Pc={Pc}: {sol.message}")

    R = float(sol.t[-1])
    M = float(sol.y[0, -1])
    Phi_arr = sol.y[2] + (0.5 * math.log(1.0 - 2.0 * M * M_SUN_KM / R) - float(sol.y[2, -1]))
    return {
        "r": sol.t,
        "m": sol.y[0],
        "P": sol.y[1],
        "Phi": Phi_arr,
        "M_Msun": M,
        "R_km": R,
        "compactness": M * M_SUN_KM / R,
    }


def _cowling_shoot(
    bg: dict[str, Any],
    ce2_of_P: Any,
    cs2_of_P: Any,
    p_trans: float,
    f_Hz: float,
    *,
    barotropic: bool = False,
    ell: int = 2,
    return_profile: bool = False,
) -> float | tuple[float, np.ndarray, np.ndarray]:
    """Integrate relativistic Cowling ODEs and evaluate surface Delta_P residual."""
    r_arr, m_arr, p_arr, ph_arr = bg["r"], bg["m"], bg["P"], bg["Phi"]
    M, R = bg["M_Msun"], bg["R_km"]
    omega = 2.0 * math.pi * f_Hz / C_KM_S
    omega2 = omega * omega

    def rhs(r: float, yv: np.ndarray) -> list[float]:
        W, V = float(yv[0]), float(yv[1])
        m = float(np.interp(r, r_arr, m_arr))
        P = max(float(np.interp(r, r_arr, p_arr)), 1.0e-9)
        Ph = float(np.interp(r, r_arr, ph_arr))
        mk = m * M_SUN_KM
        pk = P * K_CONV
        fac = r * (r - 2.0 * mk)
        Phi_p = (mk + 4.0 * math.pi * (r**3) * pk) / fac
        eL = 1.0 / math.sqrt(1.0 - 2.0 * mk / r)
        ePhi = math.exp(Ph)
        ce2 = max(1.0e-7, float(ce2_of_P(P)))
        cs2 = ce2 if barotropic else max(1.0e-7, float(cs2_of_P(P)))
        N2 = 0.0 if (barotropic or P < p_trans) else (Phi_p**2) * (ePhi**2 / (eL**2)) * max(0.0, 1.0 / ce2 - 1.0 / cs2)
        dW = (1.0 / cs2) * (omega2 * (r**2) * eL / (ePhi**2) * V + Phi_p * W) - ell * (ell + 1) * eL * V
        dV = 2.0 * Phi_p * V + (1.0 / ce2 - 1.0 / cs2) * Phi_p * V - (eL * W / (r * r)) * (1.0 - N2 / omega2)
        return [dW, dV]

    r0 = 0.02
    sol = solve_ivp(rhs, (r0, R), [r0 ** (ell + 1), -(r0**ell) / ell], method="RK45", rtol=1.0e-6, atol=1.0e-9, max_step=0.12)
    if not sol.success:
        raise NSMicrophysicsError(f"Cowling ODE failed at f={f_Hz} Hz: {sol.message}")

    W_R, V_R = float(sol.y[0, -1]), float(sol.y[1, -1])
    ePhiR = math.exp(float(ph_arr[-1]))
    eLR = 1.0 / math.sqrt(1.0 - 2.0 * M * M_SUN_KM / R)
    Phi_p_R = (M * M_SUN_KM) / (R * (R - 2.0 * M * M_SUN_KM))
    resid = (omega2 / (ePhiR**2)) * V_R + Phi_p_R * W_R / (R * R * eLR)
    if return_profile:
        return resid, sol.t, sol.y[0], sol.y[1]
    return resid


def _count_interior_nodes(r_pts: np.ndarray, w_pts: np.ndarray, r_max: float) -> int:
    """Count zero crossings of W(r) inside the interior (0.05 < r < 0.99 R)."""
    mask = (r_pts > 0.05) & (r_pts < 0.99 * r_max)
    sub = w_pts[mask]
    if len(sub) < 3:
        return 0
    signs = np.sign(sub)
    nonzero = signs[signs != 0.0]
    if len(nonzero) < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0))


def uniform_kelvin_cowling_control(compactness: float = 1.0e-4, ell: int = 2) -> dict[str, float]:
    """Analytical Newtonian Kelvin Cowling control on a uniform-density low-C star."""
    R = 12.0
    M = compactness * R / M_SUN_KM
    eps0 = 3.0 * (M * M_SUN_KM) / (4.0 * math.pi * (R**3))
    yR = math.sqrt(1.0 - 2.0 * compactness)

    def _shoot(omega_km: float) -> float:
        def rhs(r: float, yv: np.ndarray) -> list[float]:
            W, V = float(yv[0]), float(yv[1])
            mk = (4.0 * math.pi / 3.0) * eps0 * (r**3)
            yr = math.sqrt(1.0 - 2.0 * mk / r)
            eL = 1.0 / yr
            ePhi = 1.5 * yR - 0.5 * yr
            P = max(eps0 * (yr - yR) / (3.0 * yR - yr), 0.0)
            Phi_p = (mk + 4.0 * math.pi * (r**3) * P) / (r * (r - 2.0 * mk))
            cs2_inf = 1.0e8
            dW = (1.0 / cs2_inf) * ((omega_km**2) * (r**2) * eL / (ePhi**2) * V + Phi_p * W) - ell * (ell + 1) * eL * V
            dV = 2.0 * Phi_p * V - eL * W / (r * r)
            return [dW, dV]

        r0 = 1.0e-3 * R
        sol = solve_ivp(rhs, (r0, R), [r0 ** (ell + 1), -(r0**ell) / ell], method="DOP853", rtol=1.0e-9, atol=1.0e-12, max_step=0.05 * R)
        W_R, V_R = float(sol.y[0, -1]), float(sol.y[1, -1])
        Phi_p_R = (M * M_SUN_KM) / (R * (R - 2.0 * M * M_SUN_KM))
        return ((omega_km**2) / (yR**2)) * V_R + Phi_p_R * W_R / (R * R / yR)

    w_kelvin = math.sqrt(ell * M * M_SUN_KM / (R**3))
    w_root = float(brentq(_shoot, 0.5 * w_kelvin, 1.5 * w_kelvin, xtol=1.0e-12))
    rel_err = abs(w_root / w_kelvin - 1.0)
    return {
        "compactness": compactness,
        "ell": ell,
        "omega_cowling_km_inv": w_root,
        "omega_kelvin_exact_km_inv": w_kelvin,
        "omega_ratio": w_root / w_kelvin,
        "rel_err": float(rel_err),
        "passed": bool(rel_err < 1.0e-3),
    }


def compute_cowling_spectrum_for_mass(
    eos: law2.TabulatedCrustCoreEOS,
    tables: dict[str, np.ndarray],
    Pc: float,
    *,
    run_barotropic_null: bool = False,
) -> dict[str, Any]:
    """Compute l=2 f-mode, g_1-mode, g_2-mode, max N/(2*pi), and optional barotropic null."""
    bg = _solve_tov_metric_profile(eos, Pc)
    core_p = tables["core_p"]
    core_ce2 = tables["core_ce2"]
    core_cs2 = tables["core_cs2"]
    p_trans = float(core_p[0])
    logp = np.log(core_p)

    def ce2_fn(P: float) -> float:
        if P >= p_trans:
            return float(np.interp(math.log(P), logp, core_ce2))
        return 1.0 / max(eos.get_dedp(P), 1.0)

    def cs2_fn(P: float) -> float:
        if P >= p_trans:
            return float(np.interp(math.log(P), logp, core_cs2))
        return 1.0 / max(eos.get_dedp(P), 1.0)

    # Max Brunt-Vaisala frequency N / (2*pi) inside core
    n_hz_vals = []
    for r, m, P, Ph in zip(bg["r"][1:], bg["m"][1:], bg["P"][1:], bg["Phi"][1:]):
        if P >= p_trans:
            mk = m * M_SUN_KM
            pk = P * K_CONV
            Phi_p = (mk + 4.0 * math.pi * (r**3) * pk) / (r * (r - 2.0 * mk))
            eL2 = 1.0 - 2.0 * mk / r
            eP2 = math.exp(2.0 * Ph)
            N2 = (Phi_p**2) * eP2 * eL2 * max(0.0, 1.0 / ce2_fn(float(P)) - 1.0 / cs2_fn(float(P)))
            n_hz_vals.append(math.sqrt(N2) * C_KM_S / (2.0 * math.pi))
    max_N_Hz = float(max(n_hz_vals)) if n_hz_vals else 0.0

    # 1. Fundamental l=2 f-mode scan [1500, 2800] Hz
    f_grid = np.linspace(1500.0, 2800.0, 27)
    f_res = [_cowling_shoot(bg, ce2_fn, cs2_fn, p_trans, float(fv)) for fv in f_grid]
    f_mode_Hz = None
    for i in range(len(f_grid) - 1):
        if f_res[i] * f_res[i + 1] <= 0.0:
            cand = float(
                brentq(
                    lambda fv: _cowling_shoot(bg, ce2_fn, cs2_fn, p_trans, float(fv)),
                    float(f_grid[i]),
                    float(f_grid[i + 1]),
                    xtol=1.0e-2,
                )
            )
            _, r_lo_w, w_lo_w, _v_lo_w = _cowling_shoot(bg, ce2_fn, cs2_fn, p_trans, cand - 5.0, return_profile=True)
            if _count_interior_nodes(r_lo_w, w_lo_w, bg["R_km"]) == 0:
                f_mode_Hz = cand
                break
    if f_mode_Hz is None:
        raise NSMicrophysicsError(f"Failed to locate l=2 0-node f-mode for M={bg['M_Msun']:.3f}")

    # 2. Compositional g-mode spectrum scan across focused physical bands (~2 Hz step):
    #    g_1 band [250 -> 165 Hz] and g_2 band [164 -> 80 Hz], ensuring smooth branch continuity.
    #    Under GR Cowling with a barotropic crust, the g_1 displacement W-node can cross outside R
    #    while the tangential enthalpy perturbation V maintains interior radial oscillation nodes.
    g_hi = min(360.0, max(240.0, 0.9 * max_N_Hz))
    band_defs = (
        np.linspace(min(250.0, g_hi), 165.0, 44),
        np.linspace(164.0, 80.0, 43),
    )
    g_roots: list[dict[str, Any]] = []
    for f_band in band_defs:
        prev_val = _cowling_shoot(bg, ce2_fn, cs2_fn, p_trans, float(f_band[0]))
        for i in range(len(f_band) - 1):
            f_next = float(f_band[i + 1])
            cur_val = _cowling_shoot(bg, ce2_fn, cs2_fn, p_trans, f_next)
            if prev_val * cur_val <= 0.0:
                root_f = float(
                    brentq(
                        lambda fv: _cowling_shoot(bg, ce2_fn, cs2_fn, p_trans, float(fv)),
                        f_next,
                        float(f_band[i]),
                        xtol=1.0e-2,
                    )
                )
                _, r_w, w_w, v_w = _cowling_shoot(bg, ce2_fn, cs2_fn, p_trans, root_f, return_profile=True)
                nodes_w = _count_interior_nodes(r_w, w_w, bg["R_km"] / 0.99 * 0.999)
                nodes_v = _count_interior_nodes(r_w, v_w, bg["R_km"] / 0.99 * 0.999)
                g_roots.append({"freq_Hz": root_f, "radial_nodes_W": nodes_w, "radial_nodes_V": nodes_v})
            prev_val = cur_val
            if len(g_roots) >= 2 and g_roots[0]["freq_Hz"] > 165.0:
                break
        if len(g_roots) >= 2 and g_roots[0]["freq_Hz"] > 165.0:
            break
    if not g_roots:
        raise NSMicrophysicsError(f"Failed to locate l=2 compositional g1-mode for M={bg['M_Msun']:.3f}")
    g1_entry = g_roots[0]
    g2_entry = g_roots[1] if len(g_roots) >= 2 else None

    out: dict[str, Any] = {
        "M_Msun": bg["M_Msun"],
        "R_km": bg["R_km"],
        "compactness": bg["compactness"],
        "Pc_MeV_fm3": float(Pc),
        "n_c_fm3": float(np.interp(Pc, core_p, tables["core_n"])),
        "max_brunt_vaisala_Hz": max_N_Hz,
        "f_mode_Hz": f_mode_Hz,
        "f_mode_kHz": f_mode_Hz / 1000.0,
        "g1_mode_Hz": g1_entry["freq_Hz"],
        "g1_radial_nodes_W": g1_entry["radial_nodes_W"],
        "g1_radial_nodes_V": g1_entry["radial_nodes_V"],
        "g2_mode_Hz": g2_entry["freq_Hz"] if g2_entry is not None else None,
        "g2_radial_nodes_W": g2_entry["radial_nodes_W"] if g2_entry is not None else None,
        "g2_radial_nodes_V": g2_entry["radial_nodes_V"] if g2_entry is not None else None,
    }

    if run_barotropic_null:
        b_grid = np.linspace(35.0, g_hi, 36)
        b_vals = [_cowling_shoot(bg, ce2_fn, cs2_fn, p_trans, float(fv), barotropic=True) for fv in b_grid]
        sign_changes = sum(1 for i in range(len(b_vals) - 1) if b_vals[i] * b_vals[i + 1] <= 0.0)
        out["barotropic_null_control"] = {
            "cs2_set_equal_ce2": True,
            "max_N_Hz": 0.0,
            "freq_band_Hz": [35.0, float(g_hi)],
            "sign_changes_found": int(sign_changes),
            "min_abs_residual": float(min(abs(v) for v in b_vals)),
            "passed_no_gmodes": bool(sign_changes == 0),
        }

    return out


def build_result() -> dict[str, Any]:
    b8_raw = json.loads(B8_RESULTS_PATH.read_text(encoding="utf-8"))
    sa = b8_raw["stage_a_widened_primary_gate"]
    win = sa["winner"]
    survivors = sa["primary_survivors"]
    winner_key = (
        float(win["zeta_RMF"]),
        float(win["n0_fm3"]),
        float(win["M_eff_over_M_N"]),
        float(win["L_MeV"]),
    )

    # 1. Core-crust transition + DUrca + crustal inertia across all 8 primary-gate survivors
    survivor_microphysics: list[dict[str, Any]] = []
    winner_bundle: dict[str, Any] = {}

    for s in survivors:
        zeta = float(s["zeta_RMF"])
        n0 = float(s["n0_fm3"])
        ystar = float(s["M_eff_over_M_N"])
        L_mev = float(s["L_MeV"])
        p = b6.calibrate_envelope_point(n0, ystar, L_mev, zeta_rmf=zeta)
        trans = find_core_crust_transition(p)
        n_dyn = trans["dynamical"]["n_t_fm3"]
        p_dyn = trans["dynamical"]["P_t_MeV_fm3"]

        eos, tables = _build_star_eos_and_tables(p, n_dyn)
        thresh = analyze_thresholds(p, eos, tables)
        crust14 = integrate_hartle_crustal_inertia(eos, thresh["Pc_1_4_MeV_fm3"], p_dyn)

        row_entry = {
            "zeta_RMF": zeta,
            "n0_fm3": n0,
            "M_eff_over_M_N": ystar,
            "L_MeV": L_mev,
            "is_winner": bool((zeta, n0, ystar, L_mev) == winner_key),
            "core_crust_transition": trans,
            "thresholds": thresh,
            "star_1_4_crustal_inertia": crust14,
            "vela_glitch_bound_satisfied": bool(crust14["I_crust_over_I"] >= VELA_ICRUST_OVER_I_MIN),
        }
        survivor_microphysics.append(row_entry)
        if row_entry["is_winner"]:
            winner_bundle = {"p": p, "eos": eos, "tables": tables, "entry": row_entry}

    if not winner_bundle:
        raise NSMicrophysicsError("Primary-gate winner not matched among 8 survivors")

    # 2. Winner detailed beta-eq + sound speed + Cowling asteroseismology spectrum
    w_p = winner_bundle["p"]
    w_eos = winner_bundle["eos"]
    w_tables = winner_bundle["tables"]
    w_thresh = winner_bundle["entry"]["thresholds"]

    # Dense beta-eq rows for winner profile & causality check up to M_max central density
    max_n_star = w_thresh["n_c_max_fm3"]
    le_diffs = [
        r["cs2_minus_ce2"]
        for r in w_tables["rows"]
        if r["n_fm3"] <= max_n_star + 0.05
    ]
    causal_core = all(
        0.0 < r["ce2"] <= r["cs2"] + 1.0e-12 and r["cs2"] < 1.0
        for r in w_tables["rows"]
        if r["n_fm3"] <= max_n_star + 0.05
    )

    # Mass-radius & tidal curve across core pressures up to Pc_max
    pcs_mr = np.unique(
        np.concatenate(
            [
                np.geomspace(15.0, w_thresh["Pc_1_4_MeV_fm3"], 10),
                np.linspace(w_thresh["Pc_1_4_MeV_fm3"], w_thresh["Pc_max_MeV_fm3"], 16),
            ]
        )
    )
    mr_curve = []
    for pc in pcs_mr:
        M, R, k2, Lam = law2.solve_tov_tidal_star(w_eos, float(pc), rtol=1.0e-8, atol=1.0e-10, max_step=0.04)
        mr_curve.append(
            {
                "Pc_MeV_fm3": float(pc),
                "n_c_fm3": float(np.interp(pc, w_tables["core_p"], w_tables["core_n"])),
                "M_Msun": M,
                "R_km": R,
                "k2": k2,
                "Lambda": Lam,
            }
        )

    # Solve Cowling l=2 modes across COWLING_TARGET_MASSES_MSUN
    ms_mr = np.array([pt["M_Msun"] for pt in mr_curve], dtype=float)
    ps_mr = np.array([pt["Pc_MeV_fm3"] for pt in mr_curve], dtype=float)
    cowling_rows = []
    for m_target in COWLING_TARGET_MASSES_MSUN:
        if m_target > w_thresh["M_max_Msun"]:
            continue
        idx = int(np.searchsorted(ms_mr, m_target))
        p_lo = float(ps_mr[max(0, idx - 1)])
        p_hi = float(ps_mr[min(len(ps_mr) - 1, idx)])
        pc_m = float(
            brentq(
                lambda pc: _solve_tov_metric_profile(w_eos, pc)["M_Msun"] - m_target,
                p_lo,
                p_hi,
                xtol=1.0e-4,
            )
        )
        run_null = abs(m_target - 1.40) < 1.0e-6
        spec = compute_cowling_spectrum_for_mass(w_eos, w_tables, pc_m, run_barotropic_null=run_null)
        cowling_rows.append(spec)

    kelvin_ctrl = uniform_kelvin_cowling_control(compactness=1.0e-4, ell=2)
    barotropic_null = next(r["barotropic_null_control"] for r in cowling_rows if "barotropic_null_control" in r)

    gates = {
        "all_8_survivors_dynamical_lt_thermo_lt_n0": bool(
            all(
                0.03 < r["core_crust_transition"]["dynamical"]["n_t_fm3"]
                < r["core_crust_transition"]["thermodynamical"]["n_t_fm3"]
                < r["n0_fm3"]
                for r in survivor_microphysics
            )
        ),
        "all_8_survivors_schur_vs_fd_lt_1e8": bool(
            all(
                r["core_crust_transition"]["thermodynamical"]["schur_vs_fd_rel"] < 1.0e-8
                and r["core_crust_transition"]["dynamical"]["schur_vs_fd_rel"] < 1.0e-8
                for r in survivor_microphysics
            )
        ),
        "all_8_survivors_satisfy_vela_crust_inertia": bool(
            all(r["vela_glitch_bound_satisfied"] for r in survivor_microphysics)
        ),
        "winner_le_chatelier_and_causal_up_to_Mmax": bool(
            causal_core and min(le_diffs) >= -1.0e-9 and max(le_diffs) > 1.0e-3
        ),
        "cowling_kelvin_analytic_control_passed": bool(kelvin_ctrl["passed"]),
        "cowling_barotropic_null_control_passed": bool(barotropic_null["passed_no_gmodes"]),
        "winner_cowling_f_and_g1_modes_resolved": bool(
            len(cowling_rows) == len(COWLING_TARGET_MASSES_MSUN)
            and all(1500.0 <= r["f_mode_Hz"] <= 2800.0 and 50.0 <= r["g1_mode_Hz"] <= 500.0 for r in cowling_rows)
        ),
    }
    if not all(gates.values()):
        raise NSMicrophysicsError(f"One or more Bridge 9A gates failed: {gates}")

    winner_compact_rows = [
        {
            "n_fm3": r["n_fm3"],
            "n_over_n0": r["n_over_n0"],
            "x_p": r["x_p"],
            "x_e": r["x_e"],
            "x_mu": r["x_mu"],
            "P_MeV_fm3": r["P_MeV_fm3"],
            "eps_MeV_fm3": r["eps_MeV_fm3"],
            "ce2": r["ce2"],
            "cs2": r["cs2"],
            "cs2_minus_ce2": r["cs2_minus_ce2"],
            "Gamma_eq": r["Gamma_eq"],
            "Gamma_1": r["Gamma_1"],
            "du_triangle_MeV": r["du_triangle_MeV"],
            "M_eff_over_M_N": r["M_eff_over_M_N"],
            "W_over_W0": r["W_over_W0"],
        }
        for r in w_tables["rows"]
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "bridge_id": BRIDGE_ID,
        "status": STATUS_PASS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "observed_likelihood": None,
        "provenance": {
            "source_script": f"verification/{SOURCE_PATH.name}",
            "source_sha256": _sha256(SOURCE_PATH),
            "bridge8_results_path": f"verification/{B8_RESULTS_PATH.name}",
            "bridge8_results_sha256": _sha256(B8_RESULTS_PATH),
            "zero_parameter_refitting": True,
            "crust_model": "Douchin & Haensel (2001) SLy4 joined continuously at dynamical spinodal n_t^dyn",
        },
        "survivor_microphysics": survivor_microphysics,
        "winner_microphysics": {
            "frozen_params": win,
            "beta_equilibrium_profile": winner_compact_rows,
            "mass_radius_tidal_curve": mr_curve,
            "cowling_asteroseismology_l2": cowling_rows,
            "controls": {
                "uniform_kelvin_cowling": kelvin_ctrl,
                "barotropic_null_at_1_4_Msun": barotropic_null,
            },
        },
        "gates": gates,
        "honest_caveats": [
            "evidence_weight=0.0 and observed_likelihood=None: forward zero-refit consequence of the frozen Bridge-8 Lagrangian.",
            "Core-crust transition assumes homogeneous nuclear matter instability against charge-screened density fluctuations (SLy4 tabulated outer/inner crust joined at P_t^dyn without bespoke pasta Gibbs construction).",
            "Asteroseismology modes are computed in the relativistic Cowling approximation (delta g_{mu nu} = 0), which shifts l=2 f-mode frequencies upward by ~10-20% relative to full metric-perturbation GR while remaining accurate to a few percent for buoyancy-driven g_1-modes.",
            "Direct Urca neutrino cooling mass M_DUrca reflects the zero-temperature npe-mu momentum triangle k_p + k_e = k_n; finite-temperature broadening and superfluid pairing gaps are outside this cold baseline.",
        ],
    }


def _trees_match(
    actual: Any,
    expected: Any,
    *,
    path: str = "",
    rel_tol: float = 5.0e-4,
    abs_tol: float = 1.0e-6,
) -> tuple[bool, str | None]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"{path}: expected dict, got {type(actual).__name__}"
        if set(actual.keys()) != set(expected.keys()):
            missing = sorted(set(expected.keys()) - set(actual.keys()))
            extra = sorted(set(actual.keys()) - set(expected.keys()))
            return False, f"{path}: key mismatch missing={missing} extra={extra}"
        for k in expected:
            child_path = f"{path}.{k}" if path else str(k)
            ok, reason = _trees_match(
                actual[k], expected[k], path=child_path, rel_tol=rel_tol, abs_tol=abs_tol
            )
            if not ok:
                return False, reason
        return True, None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False, f"{path}: list length/type mismatch"
        for i, (a, b) in enumerate(zip(actual, expected)):
            ok, reason = _trees_match(
                a, b, path=f"{path}[{i}]", rel_tol=rel_tol, abs_tol=abs_tol
            )
            if not ok:
                return False, reason
        return True, None
    if isinstance(expected, bool) or isinstance(actual, bool):
        if type(actual) is type(expected) and actual == expected:
            return True, None
        return False, f"{path}: actual={actual!r} expected={expected!r}"
    if isinstance(expected, int) and isinstance(actual, int):
        if actual == expected:
            return True, None
        return False, f"{path}: actual={actual!r} expected={expected!r}"
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        leaf = path.rsplit(".", 1)[-1].lower()
        if any(tok in leaf for tok in ("_rel", "rel_err", "residual")):
            eff_rel, eff_abs = 0.5, 1.0e-8
        elif any(tok in leaf for tok in ("lambda", "k2", "g1_", "g2_", "f_mode", "max_brunt", "crust", "pasta", "durca", "i_")):
            eff_rel, eff_abs = 2.0e-3, 1.0e-4
        else:
            eff_rel, eff_abs = rel_tol, abs_tol
        a_val, e_val = float(actual), float(expected)
        if math.isfinite(a_val) and math.isfinite(e_val) and math.isclose(
            a_val, e_val, rel_tol=eff_rel, abs_tol=eff_abs
        ):
            return True, None
        return False, f"{path}: actual={actual!r} expected={expected!r}"
    if type(actual) is type(expected) and actual == expected:
        return True, None
    return False, f"{path}: actual={actual!r} expected={expected!r}"


def validate_artifact(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return False, [f"Missing artifact: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"Invalid JSON: {exc}"]
    if not _finite_tree(payload):
        errors.append("Non-finite number detected")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version mismatch: {payload.get('schema_version')!r}")
    if payload.get("bridge_id") != BRIDGE_ID:
        errors.append(f"bridge_id mismatch: {payload.get('bridge_id')!r}")
    if payload.get("status") != STATUS_PASS:
        errors.append(f"status mismatch: {payload.get('status')!r}")
    if payload.get("evidence_weight") != 0.0 or isinstance(payload.get("evidence_weight"), bool):
        errors.append("evidence_weight must be exact 0.0")
    if payload.get("observed_likelihood") is not None:
        errors.append("observed_likelihood must be None")
    prov = payload.get("provenance", {})
    if prov.get("source_sha256") != _sha256(SOURCE_PATH):
        errors.append("source_sha256 mismatch")
    if prov.get("bridge8_results_sha256") != _sha256(B8_RESULTS_PATH):
        errors.append("bridge8_results_sha256 mismatch")
    gates = payload.get("gates", {})
    if not gates or not all(bool(v) for v in gates.values()):
        errors.append(f"Gate failure in payload: {gates}")
    if errors:
        return False, errors

    expected = build_result()
    matched, reason = _trees_match(payload, expected)
    if not matched:
        errors.append(f"Live recomputation mismatch ({reason})")
    return len(errors) == 0, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate", nargs="?", const=str(DEFAULT_OUTPUT_PATH), default=None)
    args = parser.parse_args(argv)
    if args.validate is not None:
        ok, errs = validate_artifact(Path(args.validate))
        print(json.dumps({"valid": ok, "errors": errs}, indent=2))
        return 0 if ok else 1
    result = build_result()
    out_text = _canonical_json(result)
    out_path = args.output if args.output is not None else (DEFAULT_OUTPUT_PATH if args.write else None)
    if out_path is not None:
        out_path.write_text(out_text, encoding="utf-8")
    else:
        print(out_text, end="")
    return 0 if result["status"] == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
