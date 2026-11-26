#!/usr/bin/env python3
"""NVG theta-DM resonator & haloscope design protocol (Zenodo 10.5281/zenodo.21812161).

Conditional phenomenological target from the theta-seesaw hypothesis:
  m_3 = c_nu * (alpha_s / (4*pi))^2 * v_EW^2 / f_a,
  m_theta = sqrt(chi_top) / f_a.

At benchmark c_nu = 1.0, alpha_s(M_Z) = 0.1184, v_EW = 246.22 GeV,
m_3 = 50.3 meV (normal ordering, near-minimal lightest mass), and
chi_top^(1/4) = 75.5 MeV:
  - f_a ~ 1.07e11 GeV
  - m_theta ~ 53.2 ueV -> f_theta ~ 12.86 GHz
  - g_KSVZ ~ 2.08e-14 GeV^-1, g_DFSZ ~ 8.1e-15 GeV^-1

Features reproduced cleanly from the Zenodo 21812161 companion code:
  [0] Calibration of the Sikivie formula against published ORGAN Phase 1a powers
  [1] Single-cavity and cavity-array benchmarks (TM010 at 12.86 GHz)
  [2] Fixed MADMAX-style dielectric stack (beta^2 power-boost convention)
  [3] Published search gap summary & QCD axion-band normalization cross-check (2.6%)
  [4] Monte Carlo propagation of (m_3, chi_top) + sensitivity bands for c_nu + trials factor
  [5] Multi-channel nucleon/electron/neutrino couplings, minimal NO benchmark, UV matching

Scientific boundary: evidence_weight = 0.0. All predictions are conditional on the
stated theta-seesaw hypothesis and carry no independent empirical likelihood.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np

SCHEMA_VERSION = "nvg_theta_haloscope_design.v1"
ZENODO_DOI = "10.5281/zenodo.21812161"
EVIDENCE_WEIGHT = 0.0

# ---------------------------------------------------------------------------
# Physical constants & repo anchors
# ---------------------------------------------------------------------------
C_LIGHT = 2.99792458e8
KB = 1.380649e-23
EV = 1.602176634e-19
H = 6.62607015e-34
GEV_TO_J = 1.602176634e-10
GEV2_TO_W = GEV_TO_J / 6.582119569e-25       # ~2.42e14 W per GeV^2
GEV3_TO_M3 = 1.3012e47                       # 1 m^3 in GeV^-3
GEV2_TO_M2 = (5.0677e15) ** 2                # 1 m^2 in GeV^-2
ALPHA = 1.0 / 137.035999
T_EV2_PER_TESLA = 195.0e-18                  # eB in GeV^2 per Tesla

# Benchmark theta-seesaw inputs
ALPHA_S = 0.1184
V_EW = 246.22                                # GeV
M3_MEV = 50.3e-3                             # 50.3 meV in eV
CHI4_MEV = 75.5                              # MeV, lattice topological susceptibility
C_NU = 1.0                                   # benchmark; framework hypothesis
LOOP2 = (ALPHA_S / (4.0 * math.pi)) ** 2     # ~8.88e-5

RHO_DM_GEV_CM3 = 0.40
F_THETA_DM = 0.24                            # misalignment coherent fraction
Q_AXION = 1.0e6                              # virial linewidth Q
X01 = 2.40483


def seesaw_chain(
    c_nu: Any,
    m3_meV: Any = M3_MEV * 1.0e3,
    chi4_MeV: Any = CHI4_MEV,
) -> tuple[Any, Any, Any]:
    """Return (f_a [GeV], m_theta [ueV], f_theta [GHz]) for given inputs."""
    fa = c_nu * LOOP2 * (V_EW ** 2) / (m3_meV * 1.0e-12)
    mth = 5.691 * (chi4_MeV / 75.5) ** 2 * (1.0e12 / fa)
    fth = mth * 0.2417989
    return fa, mth, fth


def photon_couplings(f_a_GeV: float) -> tuple[float, float]:
    """Return (g_ksvz, g_dfsz) in GeV^-1."""
    g_ksvz = ALPHA / (2.0 * math.pi) * 1.92 / f_a_GeV
    g_dfsz = ALPHA / (2.0 * math.pi) * 0.75 / f_a_GeV
    return g_ksvz, g_dfsz


def axion_band_crosscheck(f_a_GeV: float, m_theta_ueV: float) -> dict[str, float]:
    """Check m_theta * f_a consistency against standard QCD axion band."""
    m_pi_f_pi = 134.977e6 * 92.2e6           # eV^2
    z_ud = 0.477
    axion_band_eV2 = m_pi_f_pi * math.sqrt(z_ud) / (1.0 + z_ud)
    implied_fa_GeV = axion_band_eV2 / (m_theta_ueV * 1.0e-6) / 1.0e9
    ratio = implied_fa_GeV / f_a_GeV
    return {
        "qcd_axion_band_eV2": axion_band_eV2,
        "repo_product_eV2": (m_theta_ueV * 1.0e-6) * (f_a_GeV * 1.0e9),
        "implied_fa_GeV": implied_fa_GeV,
        "ratio_to_seesaw_fa": ratio,
        "relative_offset": abs(1.0 - ratio),
    }


def rho_nat(frac: float = 1.0) -> float:
    return RHO_DM_GEV_CM3 * frac * 1.0e6 / GEV3_TO_M3


def m_nat(m_theta_ueV: float) -> float:
    return m_theta_ueV * 1.0e-15


def P_cavity_nat(g: float, B_T: float, V_m3: float, C: float, Q: float, frac: float, m_theta_ueV: float) -> float:
    b2 = (B_T * T_EV2_PER_TESLA) ** 2
    return (g ** 2) * (rho_nat(frac) / m_nat(m_theta_ueV)) * b2 * (V_m3 * GEV3_TO_M3) * C * Q


def P_stack_nat(g: float, B_T: float, A_m2: float, beta2: float, frac: float, m_theta_ueV: float) -> float:
    b2 = (B_T * T_EV2_PER_TESLA) ** 2
    return (g ** 2) * rho_nat(frac) * b2 * (A_m2 * GEV2_TO_M2) * beta2 / (m_nat(m_theta_ueV) ** 2)


def t_snr_cavity(P_s: float, f0: float, Q: float, T_sys: float = 1.0, snr: float = 5.0) -> float:
    df = f0 / Q
    return ((snr * KB * T_sys / P_s) ** 2) * df


def t_snr_stack(P_s: float, df_axion: float, T_sys: float = 1.0, snr: float = 5.0) -> float:
    return ((snr * KB * T_sys / P_s) ** 2) * df_axion


def build_result(n_mc: int = 200000, seed: int = 20260805) -> dict[str, Any]:
    """Compute all conditional haloscope design, MC, and multi-channel metrics."""
    f_a_GeV, m_theta_ueV, f_theta_GHz = seesaw_chain(C_NU)
    g_ksvz, g_dfsz = photon_couplings(f_a_GeV)
    band_check = axion_band_crosscheck(f_a_GeV, m_theta_ueV)

    f0 = m_theta_ueV * 1.0e-6 * EV / H
    df_axion = f0 / Q_AXION

    # [0] Calibration vs ORGAN Phase 1a (Sci. Adv. 8, eabq3765, 2022)
    ratio_model = (0.97 / 0.36) ** 2
    ratio_pub = 2.3e-25 / 3.1e-26
    ratio_check = ratio_model / ratio_pub

    m64_fa = 5.819e15 / 64.0e-6 / 1.0e9
    g64_ksvz = 0.97 * ALPHA / (2.0 * math.pi) / m64_fa
    B_organ = 7.0
    per_vcq = (g64_ksvz ** 2) * (rho_nat(1.0) / m_nat(m_theta_ueV)) * ((B_organ * T_EV2_PER_TESLA) ** 2)
    vcq_implied = (2.3e-25 / GEV2_TO_W) / per_vcq / GEV3_TO_M3
    r_15 = X01 * C_LIGHT / (2.0 * math.pi * 15.6e9)
    l_implied = vcq_implied / (0.5 * 1.5e4) / (math.pi * (r_15 ** 2))

    # [1] Cavity architecture (TM010 @ f0)
    r_cav = X01 * C_LIGHT / (2.0 * math.pi * f0)
    l_cav = 5.0e-2
    v_cav = math.pi * (r_cav ** 2) * l_cav
    c_form = 0.69
    b0 = 10.0
    q_base, q_opt = 1.0e5, 1.0e6
    t_sys = 1.0

    p_1cav = P_cavity_nat(g_ksvz, b0, v_cav, c_form, q_base, F_THETA_DM, m_theta_ueV) * GEV2_TO_W
    t_1cav_s = t_snr_cavity(p_1cav, f0, q_base, T_sys=t_sys)

    def _array_days(n_cav: int, q_val: float) -> float:
        p = P_cavity_nat(g_ksvz, b0, v_cav, c_form, q_val, F_THETA_DM, m_theta_ueV) * GEV2_TO_W
        return t_snr_cavity(p, f0, q_val, T_sys=t_sys) / n_cav / 86400.0

    # [2] MADMAX-style FIXED dielectric booster stack
    d_disk = 1.2
    a_disk = math.pi * ((d_disk / 2.0) ** 2)
    p_stack_4 = P_stack_nat(g_ksvz, b0, a_disk, 1.0e4, F_THETA_DM, m_theta_ueV) * GEV2_TO_W
    p_stack_5 = P_stack_nat(g_ksvz, b0, a_disk, 1.0e5, F_THETA_DM, m_theta_ueV) * GEV2_TO_W
    t_stack_4_s = t_snr_stack(p_stack_4, df_axion, T_sys=t_sys)
    t_stack_5_s = t_snr_stack(p_stack_5, df_axion, T_sys=t_sys)

    # [4] Monte Carlo uncertainty propagation
    rng = np.random.default_rng(seed)
    m3_s = np.abs(rng.normal(M3_MEV * 1.0e3, 0.25, n_mc))
    chi4_s = np.abs(rng.normal(CHI4_MEV, 0.007 * CHI4_MEV, n_mc))
    cnu_s = np.exp(rng.uniform(math.log(0.5), math.log(2.0), n_mc))

    _, _, f_mc = seesaw_chain(cnu_s, m3_s, chi4_s)
    pct = np.percentile(f_mc, [2.5, 16.0, 50.0, 84.0, 97.5])

    _, _, f_mc_fix = seesaw_chain(1.0, m3_s, chi4_s)
    pct_fix = np.percentile(f_mc_fix, [2.5, 16.0, 50.0, 84.0, 97.5])
    sig_fix = float(0.5 * (pct_fix[3] - pct_fix[1]) / pct_fix[2])

    _, _, f_lo_all = seesaw_chain(3.0)
    _, _, f_hi_all = seesaw_chain(1.0 / 3.0)

    f_win_lo = float(pct_fix[2] * (1.0 - 3.0 * sig_fix))
    f_win_hi = float(pct_fix[2] * (1.0 + 3.0 * sig_fix))
    n_eff = (f_win_hi - f_win_lo) * 1.0e9 / df_axion
    p_glob = math.erfc(5.0 / math.sqrt(2.0))
    z_lo, z_hi = 5.0, 9.0
    for _ in range(200):
        z_mid = 0.5 * (z_lo + z_hi)
        if n_eff * math.erfc(z_mid / math.sqrt(2.0)) > p_glob:
            z_lo = z_mid
        else:
            z_hi = z_mid
    z_local_thr = 0.5 * (z_lo + z_hi)

    # [5] Multi-channel nucleon/electron/neutrino couplings & UV matching
    mp_gev, mn_gev, me_gev = 0.93827, 0.93957, 0.510999e-3
    cp_ksvz, cn_ksvz = -0.47, -0.02
    g_thpp = cp_ksvz * mp_gev / f_a_GeV
    g_thnn = cn_ksvz * mn_gev / f_a_GeV

    tb3 = 3.0
    cb3, sb3 = 1.0 / (1.0 + tb3 ** 2), (tb3 ** 2) / (1.0 + tb3 ** 2)
    ce_dfsz = cb3 / 3.0
    g_thee_dfsz = ce_dfsz * me_gev / f_a_GeV
    g_thee_ksvz = 1.0e-5 * me_gev / f_a_GeV
    cu0_d, cd0_d = cb3 / 3.0, sb3 / 3.0
    cp_d = -0.47 + 0.88 * cu0_d - 0.39 * cd0_d - 0.49 * cd0_d
    cn_d = -0.02 - 0.02 * cu0_d + 0.88 * cd0_d - 0.49 * cd0_d
    g_thpp_d = cp_d * mp_gev / f_a_GeV
    g_thnn_d = cn_d * mn_gev / f_a_GeV

    c_nu3 = 1.0
    m3_mev_val = M3_MEV * 1.0e3
    g_thnnu = c_nu3 * m3_mev_val * 1.0e-12 / f_a_GeV

    s12, s13, d21 = 0.307, 0.0220, 7.42e-5
    m2_nu = (d21 ** 0.5) * 1.0e3
    sum_nu = m2_nu + m3_mev_val
    mbeta_nu = (s12 * (1.0 - s13) * (m2_nu ** 2) + s13 * (m3_mev_val ** 2)) ** 0.5
    a_bb, b_bb = s12 * (1.0 - s13) * m2_nu, s13 * m3_mev_val

    y_tau = 1.77686 / V_EW
    uv_match = 1.0 / (2.0 * LOOP2)
    y_nu_bench = (1.0 / uv_match) ** 0.5

    return {
        "schema_version": SCHEMA_VERSION,
        "zenodo_doi": ZENODO_DOI,
        "status": "CONDITIONAL_TARGET_REPRODUCED_ZERO_EVIDENCE",
        "evidence_weight": EVIDENCE_WEIGHT,
        "observed_likelihood": None,
        "benchmark": {
            "c_nu": C_NU,
            "f_a_GeV": f_a_GeV,
            "m_theta_ueV": m_theta_ueV,
            "f_theta_GHz": f_theta_GHz,
            "f0_Hz": f0,
            "df_axion_Hz": df_axion,
            "g_ksvz_GeV_inv": g_ksvz,
            "g_dfsz_GeV_inv": g_dfsz,
            "f_theta_dm_fraction": F_THETA_DM,
            "axion_band_crosscheck": band_check,
        },
        "organ_calibration": {
            "ratio_model": ratio_model,
            "ratio_published": ratio_pub,
            "ratio_check": ratio_check,
            "vcq_implied_m3": vcq_implied,
            "l_implied_m": l_implied,
        },
        "protocols": {
            "cavity": {
                "R_mm": r_cav * 1.0e3,
                "L_mm": l_cav * 1.0e3,
                "V_cm3": v_cav * 1.0e6,
                "P_1cav_W": p_1cav,
                "t_1cav_days": t_1cav_s / 86400.0,
                "arrays_days": {
                    "N64_Q1e5": _array_days(64, q_base),
                    "N256_Q1e5": _array_days(256, q_base),
                    "N64_Q1e6": _array_days(64, q_opt),
                    "N256_Q1e6": _array_days(256, q_opt),
                },
            },
            "fixed_dielectric_stack": {
                "D_disk_m": d_disk,
                "A_disk_m2": a_disk,
                "P_beta2_1e4_W": p_stack_4,
                "t_beta2_1e4_hours": t_stack_4_s / 3600.0,
                "P_beta2_1e5_W": p_stack_5,
                "t_beta2_1e5_minutes": t_stack_5_s / 60.0,
            },
        },
        "monte_carlo": {
            "n_mc": n_mc,
            "seed": seed,
            "f_mc_percentiles_GHz": {
                "2.5": float(pct[0]),
                "16": float(pct[1]),
                "50": float(pct[2]),
                "84": float(pct[3]),
                "97.5": float(pct[4]),
            },
            "conditional_cnu1": {
                "percentiles_GHz": {
                    "2.5": float(pct_fix[0]),
                    "16": float(pct_fix[1]),
                    "50": float(pct_fix[2]),
                    "84": float(pct_fix[3]),
                    "97.5": float(pct_fix[4]),
                },
                "rel_1sigma": sig_fix,
                "preregistered_window_GHz": [f_win_lo, f_win_hi],
                "n_eff_bins": n_eff,
                "z_local_5sigma_global": z_local_thr,
            },
            "extreme_cnu_envelope_GHz": [f_lo_all, f_hi_all],
        },
        "multichannel": {
            "g_thpp_ksvz": g_thpp,
            "g_thnn_ksvz": g_thnn,
            "g_thee_dfsz": g_thee_dfsz,
            "g_thee_ksvz": g_thee_ksvz,
            "g_thpp_dfsz": g_thpp_d,
            "g_thnn_dfsz": g_thnn_d,
            "g_thnnu": g_thnnu,
            "decay_to_nu_nu_forbidden": bool(2.0 * m3_mev_val > m_theta_ueV * 1.0e-3),
            "sum_m_nu_meV": sum_nu,
            "m_beta_meV": mbeta_nu,
            "m_betabeta_range_meV": [abs(a_bb - b_bb), a_bb + b_bb],
            "uv_match_factor": uv_match,
            "y_nu_cnu1": y_nu_bench,
            "y_nu_over_y_tau": y_nu_bench / y_tau,
        },
    }


def format_report(res: dict[str, Any]) -> str:
    b = res["benchmark"]
    cal = res["organ_calibration"]
    cav = res["protocols"]["cavity"]
    stk = res["protocols"]["fixed_dielectric_stack"]
    mc = res["monte_carlo"]
    mcc = mc["conditional_cnu1"]
    mch = res["multichannel"]
    pct = mc["f_mc_percentiles_GHz"]
    band = b["axion_band_crosscheck"]
    arr = cav["arrays_days"]
    lines = [
        "=" * 78,
        f"  NVG theta-DM RESONATOR & HALOSCOPE PROTOCOL (DOI {res['zenodo_doi']})",
        "=" * 78,
        f"[S] THETA-SEESAW CHAIN (c_nu = {b['c_nu']:.1f} benchmark, evidence_weight = {res['evidence_weight']:.1f}):",
        f"    f_a = {b['f_a_GeV']:.3e} GeV | m_theta = {b['m_theta_ueV']:.1f} ueV -> f_theta = {b['f_theta_GHz']:.2f} GHz",
        f"    g_KSVZ = {b['g_ksvz_GeV_inv']:.2e} GeV^-1 | g_DFSZ = {b['g_dfsz_GeV_inv']:.2e} GeV^-1",
        f"    QCD axion band ratio = {band['ratio_to_seesaw_fa']:.3f} ({band['relative_offset']*100:.1f}% off)",
        "[0] FORMULA CALIBRATION vs ORGAN Phase 1a:",
        f"    ratio check = {cal['ratio_check']:.3f} | implied V*C*Q = {cal['vcq_implied_m3']:.2f} m^3",
        "[1] CAVITY PROTOCOL (TM010):",
        f"    R = {cav['R_mm']:.2f} mm, V = {cav['V_cm3']:.1f} cm^3, P_1cav = {cav['P_1cav_W']:.2e} W -> SNR=5 in {cav['t_1cav_days']:.0f} d",
        f"    arrays (SNR=5): N64/Q1e5={arr['N64_Q1e5']:.0f}d, N256/Q1e5={arr['N256_Q1e5']:.0f}d, N64/Q1e6={arr['N64_Q1e6']:.1f}d, N256/Q1e6={arr['N256_Q1e6']:.1f}d",
        "[2] FIXED MADMAX-STYLE DIELECTRIC STACK:",
        f"    D = {stk['D_disk_m']:.1f} m (A = {stk['A_disk_m2']:.2f} m^2), B = 10 T",
        f"    beta^2 = 1e4: P = {stk['P_beta2_1e4_W']:.2e} W -> SNR=5 in {stk['t_beta2_1e4_hours']:.1f} h",
        f"    beta^2 = 1e5: P = {stk['P_beta2_1e5_W']:.2e} W -> SNR=5 in {stk['t_beta2_1e5_minutes']:.1f} min",
        f"[4] MONTE CARLO (N = {mc['n_mc']}, seed = {mc['seed']}):",
        f"    c_nu in [0.5, 2]: 2.5%={pct['2.5']:.2f}, 50%={pct['50']:.2f}, 97.5%={pct['97.5']:.2f} GHz",
        f"    c_nu = 1 window: [{mcc['preregistered_window_GHz'][0]:.2f}, {mcc['preregistered_window_GHz'][1]:.2f}] GHz (N_eff={mcc['n_eff_bins']:.2e}, z_loc={mcc['z_local_5sigma_global']:.2f})",
        "[5] MULTI-CHANNEL & UV CONSISTENCY:",
        f"    sum m_nu = {mch['sum_m_nu_meV']:.1f} meV, m_beta = {mch['m_beta_meV']:.1f} meV, m_bb in [{mch['m_betabeta_range_meV'][0]:.2f}, {mch['m_betabeta_range_meV'][1]:.2f}] meV",
        f"    y_nu(c_nu=1, kappa=1) = {mch['y_nu_cnu1']:.2e} ({mch['y_nu_over_y_tau']:.1f} y_tau)",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON summary.")
    args = parser.parse_args(argv)
    res = build_result()
    if args.json:
        print(json.dumps(res, indent=2, sort_keys=True))
    else:
        print(format_report(res), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
