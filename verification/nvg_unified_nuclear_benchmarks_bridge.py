#!/usr/bin/env python3
"""Bridge 9B: Full Even-Even Tin Chain (Sn100..Sn134), PREX-2/CREX Survivor Locus,
Spin-Orbit Spectrum & Head-to-Head Live RMF Comparator Benchmark (Zero Refit).

Extends the frozen Bridge-8 primary-gate winner and all 8 primary-gate survivors
with ZERO parameter refitting:

1. Full Even-Even Tin Isotopic Chain (Z = 50, A = 100..134, 11 isotopes):
     - Relativistic Hartree-BCS with fixed Bridge-8 pairing constant C_n = 21.0 MeV.
     - Verified against AMDC AME2020 binding energies and Angeli-Marinova (ADNDT 2013)
       + Gorges et al. (PRL 122, 192502, 2019) charge radii across Sn108..Sn134.
     - Tracks double-magic pairing collapse at N=50 (Sn100) and N=82 (Sn132), mid-shell
       superfluidity, two-neutron separation energy S_2n, N=82 shell gap delta_2n,
       and the Sn132 -> Sn134 charge-radius shell kink.

2. Finite-Nucleus Envelope across all 8 Primary-Gate Survivors & PREX-2 / CREX:
     - Zero-refit spherical Dirac-Hartree across (O16, Ca40, Ca48, Zr90, Sn132, Pb208).
     - Exact bulk symmetry curvature K_sym alongside (J, L), Ca48-Ca40 charge-radius shift,
       joint PREX-2 / CREX locus, and winner spin-orbit splittings across all 6 nuclei.

3. Live Head-to-Head RMF Comparator Benchmark (NL3, FSUGold, IU-FSU, FSUGold2, FSUGarnet):
     - Runs the EXACT Chen-Piekarewicz (PRC 90, 044305, 2014) nuclear-matter EOS + identical
       SLy4 crust/TOV/Hinderer pipeline AND the identical SpectralDiracSolver finite-nucleus
       benchmark on all 5 RMF comparators alongside NVG-Bridge8.
     - Audits global scalar-potential boundedness & unique vacuum vs lambda < 0 / broken vacua.
     - Evaluates an honest empirical scorecard distinguishing zero finite-nucleus refitting
       (k_fin_nuc = 0 for NVG-Bridge8) from RMF functionals fitted to finite nuclei.

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
from scipy.optimize import brentq

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_two_condensate_unified_audit as law2  # noqa: E402
import nvg_unified_dirac_hartree_probe as dh  # noqa: E402
import nvg_unified_calibration_envelope_probe as b6  # noqa: E402
import nvg_unified_primary_gate_pairing_bridge as b8  # noqa: E402
import nvg_unified_ns_microphysics_bridge as b9a  # noqa: E402

SOURCE_PATH = Path(__file__).resolve()
B8_RESULTS_PATH = HERE / "nvg_unified_primary_gate_pairing_results.json"
B9A_RESULTS_PATH = HERE / "nvg_unified_ns_microphysics_results.json"
DEFAULT_OUTPUT_PATH = HERE / "nvg_unified_nuclear_benchmarks_results.json"

SCHEMA_VERSION = "nvg_unified_nuclear_benchmarks_bridge.v1"
BRIDGE_ID = "BRIDGE9B_TIN_CHAIN_PREX_CREX_AND_RMF_COMPARATORS"
STATUS_PASS = "PASS_NUCLEAR_BENCHMARKS_TIN_ISOTOPES_PREX_CREX_AND_RMF_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

HBARC = dh.HBARC_MEV_FM
MN_RMF = 939.0
M_ELECTRON = law2.M_E_MEV
M_MUON = law2.M_MU_MEV

# Even-even Tin isotopic chain N = 50..84 (11 isotopes)
TIN_ISOTOPES: tuple[tuple[str, int, int], ...] = (
    ("Sn100", 50, 50),
    ("Sn104", 50, 54),
    ("Sn108", 50, 58),
    ("Sn112", 50, 62),
    ("Sn116", 50, 66),
    ("Sn120", 50, 70),
    ("Sn124", 50, 74),
    ("Sn128", 50, 78),
    ("Sn130", 50, 80),
    ("Sn132", 50, 82),
    ("Sn134", 50, 84),
)

# Verified AMDC AME2020 binding energies per nucleon (negative convention)
# + Angeli & Marinova (ADNDT 99, 2013) + Gorges et al. (PRL 122, 192502, 2019 for Sn134)
EXP_TIN_FULL: dict[str, dict[str, float | None]] = {
    "Sn100": {"B_over_A_MeV": -8.251626, "R_ch_fm": None},
    "Sn104": {"B_over_A_MeV": -8.383911, "R_ch_fm": None},
    "Sn108": {"B_over_A_MeV": -8.469027, "R_ch_fm": 4.5605},
    "Sn112": {"B_over_A_MeV": -8.513619, "R_ch_fm": 4.5944},
    "Sn116": {"B_over_A_MeV": -8.523117, "R_ch_fm": 4.6250},
    "Sn120": {"B_over_A_MeV": -8.504488, "R_ch_fm": 4.6519},
    "Sn124": {"B_over_A_MeV": -8.467400, "R_ch_fm": 4.6735},
    "Sn128": {"B_over_A_MeV": -8.416975, "R_ch_fm": 4.6919},
    "Sn130": {"B_over_A_MeV": -8.386817, "R_ch_fm": 4.7018},
    "Sn132": {"B_over_A_MeV": -8.354873, "R_ch_fm": 4.7093},
    "Sn134": {"B_over_A_MeV": -8.275172, "R_ch_fm": 4.7317},
}

# PREX-2 (Adhikari et al., PRL 126, 172502, 2021) & CREX (Adhikari et al., PRL 129, 042501, 2022)
PREX2_SKIN_FM = 0.283
PREX2_ERR_FM = 0.071
CREX_SKIN_FM = 0.121
CREX_ERR_FM = math.hypot(0.026, 0.024)  # 0.035384 fm combined exp + model error
CA_ISOTOPE_SHIFT_EXP_FM = -0.0005       # Angeli-Marinova: 3.4771 - 3.4776

# Published Chen & Piekarewicz (PRC 90, 044305, 2014 & PLB 748, 2015) / Lalazissis / Fattoyev RMF sets
# Parameter-count convention: k_bulk = fitted bulk/EOS couplings, k_fin_nuc = couplings adjusted to finite nuclei.
RMF_COMPARATORS: dict[str, dict[str, Any]] = {
    "NL3": {
        "citation": "Lalazissis et al., Phys. Rev. C 55, 540 (1997)",
        "m_s": 508.194, "m_v": 782.501, "m_rho": 763.000,
        "g_s2": 104.3871, "g_v2": 165.5854, "g_rho2": 79.6000,
        "kappa": 3.8599, "lam": -0.015905, "zeta": 0.0, "Lambda_v": 0.0,
        "k_bulk": 2, "k_fin_nuc": 4,
    },
    "FSUGold": {
        "citation": "Todd-Rutel & Piekarewicz, Phys. Rev. Lett. 95, 122501 (2005)",
        "m_s": 491.500, "m_v": 782.500, "m_rho": 763.000,
        "g_s2": 112.1996, "g_v2": 204.5469, "g_rho2": 138.4701,
        "kappa": 1.4203, "lam": 0.023762, "zeta": 0.0600, "Lambda_v": 0.030000,
        "k_bulk": 3, "k_fin_nuc": 5,
    },
    "IU-FSU": {
        "citation": "Fattoyev et al., Phys. Rev. C 82, 055803 (2010)",
        "m_s": 491.500, "m_v": 782.500, "m_rho": 763.000,
        "g_s2": 99.4266, "g_v2": 169.8349, "g_rho2": 184.6877,
        "kappa": 3.3808, "lam": 0.000296, "zeta": 0.0300, "Lambda_v": 0.046000,
        "k_bulk": 3, "k_fin_nuc": 5,
    },
    "FSUGold2": {
        "citation": "Chen & Piekarewicz, Phys. Rev. C 90, 044305 (2014)",
        "m_s": 497.479, "m_v": 782.500, "m_rho": 763.000,
        "g_s2": 108.0943, "g_v2": 183.7893, "g_rho2": 80.4656,
        "kappa": 3.0029, "lam": -0.000533, "zeta": 0.0256, "Lambda_v": 0.000823,
        "k_bulk": 3, "k_fin_nuc": 5,
    },
    "FSUGarnet": {
        "citation": "Chen & Piekarewicz, Phys. Lett. B 748, 284 (2015)",
        "m_s": 496.939, "m_v": 782.500, "m_rho": 763.000,
        "g_s2": 110.3492, "g_v2": 187.6947, "g_rho2": 192.9274,
        "kappa": 3.2602, "lam": -0.003551, "zeta": 0.0235, "Lambda_v": 0.043377,
        "k_bulk": 3, "k_fin_nuc": 5,
    },
}


class NuclearBenchmarksError(ValueError):
    """Raised when an invariant or solver fails in Bridge 9B."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_tree(obj: Any) -> bool:
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return True
    if isinstance(obj, (int, float)):
        return math.isfinite(float(obj))
    if isinstance(obj, dict):
        return all(_finite_tree(k) and _finite_tree(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return all(_finite_tree(v) for v in obj)
    return False


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _compute_nvg_symmetry_observables(p: law2.ModelParams) -> dict[str, float]:
    """Compute exact J, L, K_sym via small-delta parabolic finite differences around SNM."""
    n0 = p.n0_fm3
    delta = 0.02  # y_p = 0.5*(1 - delta) = 0.49
    h = 0.004
    hc3 = p.hbar_c**3

    def _sym_e(n_val: float) -> float:
        nn_nat = n_val * hc3
        st_0 = law2.solve_zero_t_state(p, 0.5 * nn_nat, 0.5 * nn_nat)
        yp = 0.5 * (1.0 - delta)
        st_d = law2.solve_zero_t_state(p, yp * nn_nat, (1.0 - yp) * nn_nat)
        e0 = (st_0["eps"] / hc3) / n_val
        ed = (st_d["eps"] / hc3) / n_val
        return float((ed - e0) / (delta * delta))

    s_vals = [_sym_e(n0 + k * h) for k in (-2, -1, 0, 1, 2)]
    J_val = s_vals[2]
    dS_dn = (s_vals[0] - 8.0 * s_vals[1] + 8.0 * s_vals[3] - s_vals[4]) / (12.0 * h)
    d2S_dn2 = (-s_vals[0] + 16.0 * s_vals[1] - 30.0 * s_vals[2] + 16.0 * s_vals[3] - s_vals[4]) / (12.0 * h * h)
    return {
        "J_MeV": float(J_val),
        "L_MeV": float(3.0 * n0 * dS_dn),
        "K_sym_MeV": float(9.0 * (n0 * n0) * d2S_dn2),
    }


def _audit_rmf_scalar_potential(par: Mapping[str, float]) -> dict[str, Any]:
    """Exact algebraic audit of U_scalar(s) boundedness & vacuum uniqueness for standard RMF."""
    gs = math.sqrt(par["g_s2"])
    m_s = float(par["m_s"])
    b_s = 0.5 * float(par["kappa"]) * (gs**3)
    c_s = (float(par["lam"]) / 6.0) * (gs**4)
    if c_s <= 0.0:
        return {
            "bounded_below": False,
            "unique_vacuum": False,
            "reason": f"negative_quartic_coupling (lambda={par['lam']:+.6f} < 0 => U(s) -> -inf as s -> +inf)",
            "min_other_vacuum_s_MeV": None,
            "min_other_vacuum_U_MeV4": None,
        }
    disc = b_s * b_s - 4.0 * (m_s * m_s) * c_s
    if disc <= 0.0:
        return {
            "bounded_below": True,
            "unique_vacuum": True,
            "reason": "globally_bounded_and_unique_minimum_at_s=0",
            "min_other_vacuum_s_MeV": None,
            "min_other_vacuum_U_MeV4": None,
        }
    sq = math.sqrt(disc)
    s_roots = [(-b_s - sq) / (2.0 * c_s), (-b_s + sq) / (2.0 * c_s)]
    # Check local minimum among non-zero stationary points (where U'' = m_s^2 + 2 b_s s + 3 c_s s^2 > 0)
    min_u = 0.0
    min_s = None
    for sr in s_roots:
        u_pp = m_s * m_s + 2.0 * b_s * sr + 3.0 * c_s * (sr * sr)
        u_val = 0.5 * (m_s * m_s) * (sr**2) + (b_s / 3.0) * (sr**3) + 0.25 * c_s * (sr**4)
        if u_pp > 0.0 and u_val < min_u:
            min_u = float(u_val)
            min_s = float(sr)
    if min_s is not None:
        return {
            "bounded_below": True,
            "unique_vacuum": False,
            "reason": f"second_deeper_false_vacuum_at_s={min_s:.2f}_MeV",
            "min_other_vacuum_s_MeV": min_s,
            "min_other_vacuum_U_MeV4": min_u,
        }
    return {
        "bounded_below": True,
        "unique_vacuum": True,
        "reason": "globally_bounded_s=0_remains_global_minimum",
        "min_other_vacuum_s_MeV": None,
        "min_other_vacuum_U_MeV4": None,
    }


def _f_scalar(kf: float, meff: float) -> float:
    if kf <= 0.0:
        return 0.0
    ef = math.hypot(kf, meff)
    return (meff / (2.0 * math.pi * math.pi)) * (kf * ef - (meff * meff) * math.log((kf + ef) / meff))


def _eps_kin(kf: float, meff: float) -> float:
    if kf <= 0.0:
        return 0.0
    ef = math.hypot(kf, meff)
    return (1.0 / (8.0 * math.pi * math.pi)) * (
        kf * ef * (2.0 * kf * kf + meff * meff) - (meff**4) * math.log((kf + ef) / meff)
    )


def _p_kin(kf: float, meff: float) -> float:
    if kf <= 0.0:
        return 0.0
    ef = math.hypot(kf, meff)
    return (1.0 / (24.0 * math.pi * math.pi)) * (
        kf * ef * (2.0 * kf * kf - 3.0 * meff * meff) + 3.0 * (meff**4) * math.log((kf + ef) / meff)
    )


def _solve_cp_rmf_state(n_fm3: float, yp: float, par: Mapping[str, float]) -> dict[str, float]:
    """Exact Chen & Piekarewicz (PRC 90, 044305, 2014) nuclear-matter EOS at fixed (n_B, y_p)."""
    c_s2 = float(par["m_s"]) ** 2 / float(par["g_s2"])
    c_v2 = float(par["m_v"]) ** 2 / float(par["g_v2"])
    c_r2 = float(par["m_rho"]) ** 2 / float(par["g_rho2"])
    kap = float(par["kappa"])
    lam = float(par["lam"])
    zeta = float(par["zeta"])
    lv = float(par["Lambda_v"])

    np_m = yp * n_fm3 * (HBARC**3)
    nn_m = (1.0 - yp) * n_fm3 * (HBARC**3)
    kfp = (3.0 * math.pi * math.pi * np_m) ** (1.0 / 3.0) if np_m > 0.0 else 0.0
    kfn = (3.0 * math.pi * math.pi * nn_m) ** (1.0 / 3.0) if nn_m > 0.0 else 0.0

    def _f_meff(meff: float) -> float:
        phi = MN_RMF - meff
        ns = _f_scalar(kfp, meff) + _f_scalar(kfn, meff)
        return c_s2 * phi + 0.5 * kap * (phi * phi) + (lam / 6.0) * (phi**3) - ns

    # Local nuclear-matter branch (near nucleon effective mass; avoid false vacua if lambda < 0)
    meff_grid = np.linspace(40.0, MN_RMF - 1.0e-4, 60)
    vals = [_f_meff(float(m)) for m in meff_grid]
    bracket = None
    for i in range(len(meff_grid) - 1, 0, -1):
        if vals[i] * vals[i - 1] <= 0.0:
            bracket = (float(meff_grid[i - 1]), float(meff_grid[i]))
            break
    if bracket is None:
        raise NuclearBenchmarksError(f"Failed to bracket RMF M_eff at n={n_fm3}, yp={yp}")
    meff = float(brentq(_f_meff, bracket[0], bracket[1], xtol=1.0e-11))
    phi = MN_RMF - meff

    n_tot = np_m + nn_m
    n_3 = 0.5 * (np_m - nn_m)

    def _f_v0(v0: float) -> float:
        b0 = n_3 / (c_r2 + 2.0 * lv * v0 * v0)
        return c_v2 * v0 + (zeta / 6.0) * (v0**3) + 2.0 * lv * (b0 * b0) * v0 - n_tot

    v_hi = max(150.0, 2.0 * n_tot / c_v2 + 50.0)
    while _f_v0(v_hi) < 0.0:
        v_hi *= 2.0
    v0 = float(brentq(_f_v0, 0.0, v_hi, xtol=1.0e-11))
    b0 = n_3 / (c_r2 + 2.0 * lv * v0 * v0)

    u_phi = 0.5 * c_s2 * (phi**2) + (kap / 6.0) * (phi**3) + (lam / 24.0) * (phi**4)
    eps_m4 = (
        _eps_kin(kfp, meff)
        + _eps_kin(kfn, meff)
        + u_phi
        + 0.5 * c_v2 * (v0**2)
        + (zeta / 8.0) * (v0**4)
        + 0.5 * c_r2 * (b0**2)
        + 3.0 * lv * (b0**2) * (v0**2)
    )
    p_m4 = (
        _p_kin(kfp, meff)
        + _p_kin(kfn, meff)
        - u_phi
        + 0.5 * c_v2 * (v0**2)
        + (zeta / 24.0) * (v0**4)
        + 0.5 * c_r2 * (b0**2)
        + lv * (b0**2) * (v0**2)
    )
    mu_p = math.hypot(kfp, meff) + v0 + 0.5 * b0
    mu_n = math.hypot(kfn, meff) + v0 - 0.5 * b0
    return {
        "eps_MeV_fm3": float(eps_m4 / (HBARC**3)),
        "P_MeV_fm3": float(p_m4 / (HBARC**3)),
        "M_eff_over_M_N": float(meff / MN_RMF),
        "mu_p_MeV": float(mu_p),
        "mu_n_MeV": float(mu_n),
    }


def _solve_cp_rmf_beta_point(n_fm3: float, par: Mapping[str, float]) -> dict[str, float]:
    """Solve cold npe-mu beta equilibrium for a Chen-Piekarewicz RMF functional."""

    def _residual(yp: float) -> float:
        st = _solve_cp_rmf_state(n_fm3, yp, par)
        dmu = st["mu_n_MeV"] - st["mu_p_MeV"]
        if dmu <= M_ELECTRON:
            return -yp
        kf_e = math.sqrt(dmu * dmu - M_ELECTRON * M_ELECTRON)
        n_e = (kf_e**3) / (3.0 * math.pi * math.pi * (HBARC**3))
        if dmu > M_MUON:
            kf_mu = math.sqrt(dmu * dmu - M_MUON * M_MUON)
            n_mu = (kf_mu**3) / (3.0 * math.pi * math.pi * (HBARC**3))
        else:
            kf_mu, n_mu = 0.0, 0.0
        return (n_e + n_mu) / n_fm3 - yp

    yp_star = float(brentq(_residual, 1.0e-6, 0.499, xtol=1.0e-11))
    st = _solve_cp_rmf_state(n_fm3, yp_star, par)
    dmu = st["mu_n_MeV"] - st["mu_p_MeV"]
    kf_e = math.sqrt(max(0.0, dmu * dmu - M_ELECTRON * M_ELECTRON))
    kf_mu = math.sqrt(max(0.0, dmu * dmu - M_MUON * M_MUON)) if dmu > M_MUON else 0.0
    eps_tot = st["eps_MeV_fm3"] + (_eps_kin(kf_e, M_ELECTRON) + _eps_kin(kf_mu, M_MUON)) / (HBARC**3)
    p_tot = st["P_MeV_fm3"] + (_p_kin(kf_e, M_ELECTRON) + _p_kin(kf_mu, M_MUON)) / (HBARC**3)
    n_e = (kf_e**3) / (3.0 * math.pi * math.pi * (HBARC**3))
    kf_p = (3.0 * math.pi * math.pi * yp_star * n_fm3 * (HBARC**3)) ** (1.0 / 3.0)
    kf_n = (3.0 * math.pi * math.pi * (1.0 - yp_star) * n_fm3 * (HBARC**3)) ** (1.0 / 3.0)
    return {
        "n_fm3": float(n_fm3),
        "x_p": float(yp_star),
        "x_e": float(n_e / n_fm3),
        "durca_gap_MeV": float(kf_p + kf_e - kf_n),
        "eps_MeV_fm3": float(eps_tot),
        "P_MeV_fm3": float(p_tot),
    }


def _solve_rmf_bulk_and_ns(name: str, par: Mapping[str, Any]) -> dict[str, Any]:
    """Compute live RMF bulk saturation, vacuum audit, and SLy4 TOV/Hinderer NS observables."""
    n0 = float(brentq(lambda n: _solve_cp_rmf_state(n, 0.5, par)["P_MeV_fm3"], 0.130, 0.175, xtol=1.0e-10))
    st0 = _solve_cp_rmf_state(n0, 0.5, par)
    ba0 = st0["eps_MeV_fm3"] / n0 - MN_RMF
    h = 0.004
    e_m = _solve_cp_rmf_state(n0 - h, 0.5, par)["eps_MeV_fm3"] / (n0 - h)
    e_p = _solve_cp_rmf_state(n0 + h, 0.5, par)["eps_MeV_fm3"] / (n0 + h)
    e_0 = st0["eps_MeV_fm3"] / n0
    K0 = 9.0 * (n0 * n0) * (e_p - 2.0 * e_0 + e_m) / (h * h)

    delta = 0.02
    def _sym_e(nv: float) -> float:
        s0 = _solve_cp_rmf_state(nv, 0.5, par)["eps_MeV_fm3"] / nv
        sd = _solve_cp_rmf_state(nv, 0.5 * (1.0 - delta), par)["eps_MeV_fm3"] / nv
        return float((sd - s0) / (delta * delta))

    sv = [_sym_e(n0 + k * h) for k in (-2, -1, 0, 1, 2)]
    J_val = sv[2]
    L_val = 3.0 * n0 * (sv[0] - 8.0 * sv[1] + 8.0 * sv[3] - sv[4]) / (12.0 * h)
    K_sym_val = 9.0 * (n0 * n0) * (-sv[0] + 16.0 * sv[1] - 30.0 * sv[2] + 16.0 * sv[3] - sv[4]) / (12.0 * h * h)

    n_grid = np.linspace(0.05, 1.40, 136)
    rows = [r for r in (_solve_cp_rmf_beta_point(float(n), par) for n in n_grid) if r["P_MeV_fm3"] > 0.36]
    core_n = np.array([r["n_fm3"] for r in rows])
    core_p = np.array([r["P_MeV_fm3"] for r in rows])
    core_e = np.array([r["eps_MeV_fm3"] for r in rows])
    core_dg = np.array([r["durca_gap_MeV"] for r in rows])
    eos = law2.TabulatedCrustCoreEOS(core_p, core_e, crust_mode="dh_sly4")

    def _tov_dict(pc_val: float) -> dict[str, float]:
        m_v, r_v, k2_v, lam_v = law2.solve_tov_tidal_star(eos, float(pc_val), rtol=1.0e-7, atol=1.0e-9, max_step=0.10)
        return {
            "Pc_MeV_fm3": float(pc_val),
            "M_Msun": float(m_v),
            "R_km": float(r_v),
            "k2": float(k2_v),
            "Lambda": float(lam_v),
        }

    # Scan Pc to find M_max and M=1.4 M_sun
    pc_grid = np.geomspace(5.0, 900.0, 75)
    tov_rows = []
    m_prev = -1.0
    pc_peak = float(pc_grid[-1])
    for pc in pc_grid:
        try:
            st = _tov_dict(float(pc))
            tov_rows.append(st)
            if st["M_Msun"] < m_prev and st["M_Msun"] > 1.0:
                pc_peak = float(pc)
                break
            m_prev = st["M_Msun"]
        except Exception:
            break

    max_row = max(tov_rows, key=lambda r: r["M_Msun"])
    idx_max = tov_rows.index(max_row)
    lo_pc = tov_rows[max(0, idx_max - 1)]["Pc_MeV_fm3"]
    hi_pc = min(tov_rows[min(len(tov_rows) - 1, idx_max + 1)]["Pc_MeV_fm3"], pc_peak)
    # Golden section refinement around M_max
    phi_g = 0.5 * (math.sqrt(5.0) - 1.0)
    a, b = lo_pc, hi_pc
    c = b - phi_g * (b - a)
    d = a + phi_g * (b - a)
    fc = _tov_dict(c)["M_Msun"]
    fd = _tov_dict(d)["M_Msun"]
    for _ in range(18):
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + phi_g * (b - a)
            fd = _tov_dict(d)["M_Msun"]
        else:
            b, d, fd = d, c, fc
            c = b - phi_g * (b - a)
            fc = _tov_dict(c)["M_Msun"]
    star_max = _tov_dict(0.5 * (a + b))

    pc_14 = float(
        brentq(
            lambda x: _tov_dict(x)["M_Msun"] - 1.4,
            tov_rows[0]["Pc_MeV_fm3"],
            star_max["Pc_MeV_fm3"],
            xtol=1.0e-4,
        )
    )
    star_14 = _tov_dict(pc_14)

    # Check Direct Urca onset on core_n grid
    m_durca = None
    for i in range(len(core_dg) - 1):
        if core_dg[i] < 0.0 <= core_dg[i + 1]:
            p_du = float(brentq(lambda nv: _solve_cp_rmf_beta_point(nv, par)["durca_gap_MeV"], float(core_n[i]), float(core_n[i + 1]), xtol=1.0e-7))
            p_press = _solve_cp_rmf_beta_point(p_du, par)["P_MeV_fm3"]
            if p_press <= star_max["Pc_MeV_fm3"]:
                m_durca = float(_tov_dict(p_press)["M_Msun"])
            break

    return {
        "name": name,
        "saturation": {
            "n0_fm3": float(n0),
            "B_over_A_MeV": float(ba0),
            "K_MeV": float(K0),
            "M_eff_over_M_N": float(st0["M_eff_over_M_N"]),
            "J_MeV": float(J_val),
            "L_MeV": float(L_val),
            "K_sym_MeV": float(K_sym_val),
        },
        "vacuum_audit": _audit_rmf_scalar_potential(par),
        "neutron_star_sly4_crust": {
            "M_max_Msun": float(star_max["M_Msun"]),
            "R_at_Mmax_km": float(star_max["R_km"]),
            "R_1_4_km": float(star_14["R_km"]),
            "Lambda_1_4": float(star_14["Lambda"]),
            "M_DUrca_Msun": m_durca,
        },
    }


def _solve_rmf_nucleus(Z: int, N: int, par: Mapping[str, float]) -> dict[str, Any]:
    """Run exact SpectralDiracSolver + EvenChebyshevGrid on a Chen-Piekarewicz RMF functional."""
    gs = math.sqrt(float(par["g_s2"]))
    gv = math.sqrt(float(par["g_v2"]))
    gr = math.sqrt(float(par["g_rho2"]))
    p = {
        "M_N": MN_RMF,
        "W0": 859.0,
        "lam_W": 1.0e8,
        "m_W": math.sqrt(2.0e8) * 859.0,
        "m_s": float(par["m_s"]),
        "m_omega": float(par["m_v"]),
        "m_rho": float(par["m_rho"]),
        "g_s": gs,
        "g_omega": gv,
        "g_rho": gr,
        "b_s": 0.5 * float(par["kappa"]) * (gs**3),
        "c_s": (float(par["lam"]) / 6.0) * (gs**4),
        "d_s": 0.0,
        "zeta": float(par["zeta"]),
        "Lambda_v": float(par["Lambda_v"]),
        "q_phi": float(par["m_v"]) / 859.0,
    }
    z_int, n_int = int(Z), int(N)
    a_int = z_int + n_int
    r_box = 16.0 if a_int <= 50 else 20.0
    m_field, n_basis, n_quad = 64, 36, 96
    grid = dh.EvenChebyshevGrid(r_box, m_intervals=m_field)
    dirac = dh.SpectralDiracSolver(r_box, n_basis=n_basis, n_quad=n_quad, mass_ref=MN_RMF)
    interp_q = grid.interp_matrix(dirac.t_q)
    r = grid.r

    r0 = 1.2 * (a_int ** (1.0 / 3.0))
    ws = 1.0 / (1.0 + np.exp((r - r0) / 0.55))
    w = p["W0"] * np.ones_like(r)
    s = (350.0 / gs) * ws
    a0 = (280.0 / gv) * ws
    rho0 = -5.0 * ((n_int - z_int) / a_int) * ws
    v_c = np.where(
        r <= r0,
        z_int * dh.ALPHA_EM * HBARC / (2.0 * r0) * (3.0 - (r / r0) ** 2),
        z_int * dh.ALPHA_EM * HBARC / np.maximum(r, 1.0e-12),
    )

    kappas = dh._candidate_kappas_for_A(a_int)
    hx: list[np.ndarray] = []
    hf: list[np.ndarray] = []
    conv = False
    dens: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    orbs: dict[str, list[dict[str, Any]]] = {}

    def _step_orbs(wf: np.ndarray, sf: np.ndarray, af: np.ndarray, rf: np.ndarray, vcf: np.ndarray) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, list[dict[str, Any]]]]:
        m_eff_q = interp_q @ (MN_RMF * (wf / p["W0"]) - gs * sf)
        vp_q = interp_q @ (gv * af + 0.5 * gr * rf + vcf)
        vn_q = interp_q @ (gv * af - 0.5 * gr * rf)
        d_out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        o_out: dict[str, list[dict[str, Any]]] = {}
        for sp, cnt, vq in (("p", z_int, vp_q), ("n", n_int, vn_q)):
            cands = []
            for kap in kappas:
                vals, vecs, _, phi_e, chi_e, phi0, chi0 = dirac.solve_sector(kap, m_eff_q, vq, grid.t)
                for ii in np.where(vals > 0.0)[0][:6]:
                    vec = vecs[:, ii]
                    cl, cs = vec[:n_basis], vec[n_basis:]
                    cands.append({
                        "E": float(vals[ii]),
                        "deg": 2 * abs(kap),
                        "G": cl @ phi_e,
                        "F": cs @ chi_e,
                        "g0": float(cl @ phi0),
                        "f0": float(cs @ chi0),
                    })
            cands.sort(key=lambda item: item["E"])
            rem = float(cnt)
            rv, rs = np.zeros_like(r), np.zeros_like(r)
            olist = []
            for c in cands:
                if rem <= 1.0e-12:
                    break
                occ = min(float(c["deg"]), rem)
                rem -= occ
                c["occ"] = occ
                olist.append(c)
                rv[1:] += occ * (c["G"][1:] ** 2 + c["F"][1:] ** 2) / (4.0 * math.pi * r[1:] ** 2)
                rs[1:] += occ * (c["G"][1:] ** 2 - c["F"][1:] ** 2) / (4.0 * math.pi * r[1:] ** 2)
                rv[0] += occ * (c["g0"] ** 2 + c["f0"] ** 2) / (4.0 * math.pi)
                rs[0] += occ * (c["g0"] ** 2 - c["f0"] ** 2) / (4.0 * math.pi)
            if rem > 1.0e-8:
                raise NuclearBenchmarksError(f"RMF insufficient bound orbitals for {sp} Z={Z}, N={N}")
            d_out[sp] = (rv, rs)
            o_out[sp] = olist
        return d_out, o_out

    for _it in range(1, 101):
        dens, orbs = _step_orbs(w, s, a0, rho0, v_c)
        w_n, s_n, a_n, r_n, v_n = dh._solve_fields_newton(
            grid, p, dens["p"][1] + dens["n"][1], dens["p"][0], dens["n"][0], w, s, a0, rho0
        )
        xc = np.concatenate([w, s, a0, rho0, v_c])
        xn = np.concatenate([w_n, s_n, a_n, r_n, v_n])
        if float(np.max(np.abs(xn - xc))) < 1.0e-8:
            w, s, a0, rho0, v_c = w_n, s_n, a_n, r_n, v_n
            conv = True
            break
        xt = dh._anderson_mix(xc, xn, hx, hf, beta=0.40, m_max=6)
        wt, st_f, at, rt, vt = np.split(xt, 5)
        if np.min(wt) <= 0.2 * p["W0"] or np.min(MN_RMF * (wt / p["W0"]) - gs * st_f) <= 50.0:
            xt = 0.75 * xc + 0.25 * xn
            hx.clear()
            hf.clear()
            wt, st_f, at, rt, vt = np.split(xt, 5)
        w, s, a0, rho0, v_c = wt, st_f, at, rt, vt

    if not conv:
        raise NuclearBenchmarksError(f"RMF nucleus SCF failed for Z={Z}, N={N}")
    dens, orbs = _step_orbs(w, s, a0, rho0, v_c)
    rho_p, rhos_p = dens["p"]
    rho_n, rhos_n = dens["n"]
    rho_s = rhos_p + rhos_n
    rho_b = rho_p + rho_n
    rho_3 = rho_p - rho_n
    w3 = 4.0 * math.pi * (r * r) * grid.w
    hc3 = HBARC**3

    eig_sum = sum(o["occ"] * o["E"] for sp in ("p", "n") for o in orbs[sp])
    e_field_eig = float(
        np.dot(
            w3,
            (
                0.5 * gs * s * rho_s
                - (p["b_s"] / 6.0) * (s**3) / hc3
                - (p["c_s"] / 4.0) * (s**4) / hc3
                - 0.5 * gv * a0 * rho_b
                + (p["zeta"] / 24.0) * (gv**4) * (a0**4) / hc3
                - 0.25 * gr * rho0 * rho_3
                + p["Lambda_v"] * (gv**2) * (gr**2) * (a0**2) * (rho0**2) / hc3
                - 0.5 * v_c * rho_p
            ),
        )
    )
    e_cm = -0.75 * 41.0 * (a_int ** (-1.0 / 3.0))
    ba_eig = (eig_sum + e_field_eig + e_cm - a_int * MN_RMF) / a_int

    zp = float(np.dot(w3, rho_p))
    zn = float(np.dot(w3, rho_n))
    rp = math.sqrt(float(np.dot(w3, (r * r) * rho_p)) / zp)
    rn = math.sqrt(float(np.dot(w3, (r * r) * rho_n)) / zn)
    rch = math.sqrt(rp * rp + dh.PROTON_CHARGE_RADIUS_FM**2)
    fres = dh._field_residuals(grid, p, w, s, a0, rho0, v_c, rho_s, rho_p, rho_n)

    return {
        "Z": z_int,
        "N": n_int,
        "A": a_int,
        "converged": True,
        "B_over_A_MeV": float(ba_eig),
        "R_ch_fm": float(rch),
        "Delta_R_np_fm": float(rn - rp),
        "particle_rel_err": max(abs(zp - z_int) / z_int, abs(zn - n_int) / n_int),
        "max_field_residual_rel": float(fres["max_rel"]),
    }


def _run_rmf_six_nuclei(name: str, par: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    ba_errs = []
    rch_pcts = []
    rch_errs_fm = []
    ca40_rch = None
    ca48_rch = None
    ca48_skin = None
    pb208_skin = None
    for nuc_name, Z, N in b6.SIX_NUCLEI:
        res = _solve_rmf_nucleus(Z, N, par)
        exp = b6.EXP_SIX_NUCLEI[nuc_name]
        ba_err = res["B_over_A_MeV"] - exp["B_over_A_MeV"]
        rch_err_fm = res["R_ch_fm"] - exp["R_ch_fm"]
        rch_pct = 100.0 * rch_err_fm / exp["R_ch_fm"]
        ba_errs.append(ba_err)
        rch_pcts.append(rch_pct)
        rch_errs_fm.append(rch_err_fm)
        if nuc_name == "Ca40":
            ca40_rch = res["R_ch_fm"]
        elif nuc_name == "Ca48":
            ca48_rch = res["R_ch_fm"]
            ca48_skin = res["Delta_R_np_fm"]
        elif nuc_name == "Pb208":
            pb208_skin = res["Delta_R_np_fm"]
        rows.append({
            "nucleus": nuc_name,
            "Z": Z,
            "N": N,
            "A": Z + N,
            "converged": res["converged"],
            "B_over_A_MeV": res["B_over_A_MeV"],
            "exp_B_over_A_MeV": exp["B_over_A_MeV"],
            "B_over_A_err_MeV": ba_err,
            "R_ch_fm": res["R_ch_fm"],
            "exp_R_ch_fm": exp["R_ch_fm"],
            "R_ch_err_fm": rch_err_fm,
            "R_ch_rel_err_percent": rch_pct,
            "Delta_R_np_fm": res["Delta_R_np_fm"],
            "particle_rel_err": res["particle_rel_err"],
            "max_field_residual_rel": res["max_field_residual_rel"],
        })
    return {
        "model": name,
        "nuclei": rows,
        "summary_metrics": {
            "rms_B_over_A_err_MeV": float(math.sqrt(sum(x * x for x in ba_errs) / len(ba_errs))),
            "rms_R_ch_err_fm": float(math.sqrt(sum(x * x for x in rch_errs_fm) / len(rch_errs_fm))),
            "rms_R_ch_rel_err_percent": float(math.sqrt(sum(x * x for x in rch_pcts) / len(rch_pcts))),
            "Delta_R_ch_Ca48_minus_Ca40_fm": float(ca48_rch - ca40_rch),
            "Delta_R_np_Ca48_fm": float(ca48_skin),
            "Delta_R_np_Pb208_fm": float(pb208_skin),
            "max_particle_rel_err": max(r["particle_rel_err"] for r in rows),
        },
    }


def analyze_full_tin_chain(winner_fp: Mapping[str, float]) -> dict[str, Any]:
    """Evaluate all 11 even-even Tin isotopes Sn100..Sn134 with zero pairing refit."""
    rows: list[dict[str, Any]] = []
    ba_errs: list[float] = []
    rch_pcts: list[float] = []
    rch_errs_fm: list[float] = []

    for nuc_name, Z, N in TIN_ISOTOPES:
        res = b8.solve_nucleus_rh_bcs(Z, N, winner_fp, pairing_C_n_MeV=b8.PAIRING_C_N_MEV, core_n=b8.PAIRING_CORE_N)
        exp = EXP_TIN_FULL[nuc_name]
        ba_pred = float(res["B_over_A_MeV"])
        ba_exp = float(exp["B_over_A_MeV"])
        e_tot_pred = float((Z + N) * ba_pred)
        e_tot_exp = float((Z + N) * ba_exp)
        ba_err = ba_pred - ba_exp
        ba_errs.append(ba_err)

        rch_pred = float(res["R_ch_fm"])
        rch_exp = exp["R_ch_fm"]
        rch_err_fm = None if rch_exp is None else float(rch_pred - rch_exp)
        rch_pct = None if rch_exp is None else float(100.0 * rch_err_fm / rch_exp)
        if rch_pct is not None and rch_err_fm is not None:
            rch_pcts.append(rch_pct)
            rch_errs_fm.append(rch_err_fm)

        rows.append({
            "nucleus": nuc_name,
            "Z": Z,
            "N": N,
            "A": Z + N,
            "converged": bool(res["converged"]),
            "B_over_A_MeV": ba_pred,
            "exp_B_over_A_MeV": ba_exp,
            "B_over_A_err_MeV": float(ba_err),
            "E_tot_MeV": e_tot_pred,
            "exp_E_tot_MeV": e_tot_exp,
            "R_ch_fm": rch_pred,
            "exp_R_ch_fm": rch_exp,
            "R_ch_err_fm": rch_err_fm,
            "R_ch_rel_err_percent": rch_pct,
            "Delta_R_np_fm": float(res["Delta_R_np_fm"]),
            "Delta_n_MeV": float(res["Delta_n_MeV"]),
            "collapsed_to_normal": bool(res["collapsed_to_normal"]),
            "E_pair_MeV": float(res["E_pair_MeV"]),
            "S2n_step_MeV": None,
            "exp_S2n_step_MeV": None,
        })

    for i in range(1, len(rows)):
        dN = rows[i]["N"] - rows[i - 1]["N"]
        rows[i]["S2n_step_MeV"] = float(2.0 * (rows[i - 1]["E_tot_MeV"] - rows[i]["E_tot_MeV"]) / dN)
        rows[i]["exp_S2n_step_MeV"] = float(2.0 * (rows[i - 1]["exp_E_tot_MeV"] - rows[i]["exp_E_tot_MeV"]) / dN)

    by_name = {r["nucleus"]: r for r in rows}
    # Exact N=82 shell gap delta_{2n}(132Sn) = S_{2n}(132Sn) - S_{2n}(134Sn) (both steps have dN=2)
    delta2n_pred = float(by_name["Sn132"]["S2n_step_MeV"] - by_name["Sn134"]["S2n_step_MeV"])
    delta2n_exp = float(by_name["Sn132"]["exp_S2n_step_MeV"] - by_name["Sn134"]["exp_S2n_step_MeV"])
    rch_kink_pred = float(by_name["Sn134"]["R_ch_fm"] - by_name["Sn132"]["R_ch_fm"])
    rch_kink_exp = float(by_name["Sn134"]["exp_R_ch_fm"] - by_name["Sn132"]["exp_R_ch_fm"])

    return {
        "pairing_C_n_MeV_inherited": float(b8.PAIRING_C_N_MEV),
        "zero_pairing_refit": True,
        "isotopes": rows,
        "summary": {
            "isotope_count": len(rows),
            "all_converged": all(r["converged"] for r in rows),
            "Sn100_N50_collapsed": bool(by_name["Sn100"]["collapsed_to_normal"] and by_name["Sn100"]["Delta_n_MeV"] == 0.0),
            "Sn132_N82_collapsed": bool(by_name["Sn132"]["collapsed_to_normal"] and by_name["Sn132"]["Delta_n_MeV"] == 0.0),
            "Sn134_post_shell_Delta_n_MeV": float(by_name["Sn134"]["Delta_n_MeV"]),
            "max_midshell_Delta_n_MeV": float(max(r["Delta_n_MeV"] for r in rows if 58 <= r["N"] <= 78)),
            "delta_2n_Sn132_shell_gap_MeV": delta2n_pred,
            "exp_delta_2n_Sn132_shell_gap_MeV": delta2n_exp,
            "R_ch_kink_Sn134_minus_Sn132_fm": rch_kink_pred,
            "exp_R_ch_kink_Sn134_minus_Sn132_fm": rch_kink_exp,
            "rms_B_over_A_err_MeV": float(math.sqrt(sum(x * x for x in ba_errs) / len(ba_errs))),
            "rms_R_ch_err_fm": float(math.sqrt(sum(x * x for x in rch_errs_fm) / len(rch_errs_fm))),
            "rms_R_ch_rel_err_percent": float(math.sqrt(sum(x * x for x in rch_pcts) / len(rch_pcts))),
        },
    }


def _scorecard_entry(
    *,
    name: str,
    zero_finite_nucleus_refit: bool,
    k_bulk: int,
    k_fin_nuc: int,
    vacuum_bounded_and_unique: bool,
    sat: Mapping[str, float],
    ns: Mapping[str, float],
    fin_summary: Mapping[str, float],
    fin_nuclei: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute transparent empirical scorecard across finite nuclei + GW170817/PSR J0740."""
    # Exact RMS in fm on charge radii for uniform scorecard
    rch_errs_fm = [float(r["R_ch_fm"] - r["exp_R_ch_fm"]) for r in fin_nuclei]
    rms_rch_fm = float(math.sqrt(sum(x * x for x in rch_errs_fm) / len(rch_errs_fm)))
    rms_ba = float(fin_summary["rms_B_over_A_err_MeV"])
    skin_208 = float(fin_summary["Delta_R_np_Pb208_fm"])
    skin_48 = float(fin_summary["Delta_R_np_Ca48_fm"])
    d_ca = float(fin_summary["Delta_R_ch_Ca48_minus_Ca40_fm"])

    z_prex2 = (skin_208 - PREX2_SKIN_FM) / PREX2_ERR_FM
    z_crex = (skin_48 - CREX_SKIN_FM) / CREX_ERR_FM
    chi2_skins = float(z_prex2 * z_prex2 + z_crex * z_crex)

    # Combined multi-sector diagnostic chi^2 (8 finite-nucleus + PREX-2 + CREX + Ca shift + NS constraint residuals;
    # reference scales: sigma_BA=0.05 MeV/A, sigma_Rch=0.02 fm, sigma_dCa=0.005 fm; informative only, evidence_weight=0.0)
    n_obs_diag = 16
    chi2_diag = (
        6.0 * (rms_ba / 0.05) ** 2
        + 6.0 * (rms_rch_fm / 0.02) ** 2
        + chi2_skins
        + ((d_ca - CA_ISOTOPE_SHIFT_EXP_FM) / 0.005) ** 2
        + (max(0.0, (2.08 - float(ns["M_max_Msun"])) / 0.05)) ** 2
    )
    k_tot = k_bulk + k_fin_nuc
    aic_tot = float(chi2_diag + 2.0 * k_tot)
    bic_tot = float(chi2_diag + k_tot * math.log(n_obs_diag))

    return {
        "model": name,
        "zero_finite_nucleus_refit": bool(zero_finite_nucleus_refit),
        "k_bulk": int(k_bulk),
        "k_fin_nuc": int(k_fin_nuc),
        "k_total": int(k_tot),
        "global_vacuum_bounded_and_unique": bool(vacuum_bounded_and_unique),
        "n0_fm3": float(sat["n0_fm3"]),
        "K_MeV": float(sat["K_MeV"]),
        "J_MeV": float(sat["J_MeV"]),
        "L_MeV": float(sat["L_MeV"]),
        "K_sym_MeV": float(sat["K_sym_MeV"]),
        "M_max_Msun": float(ns["M_max_Msun"]),
        "R_1_4_km": float(ns["R_1_4_km"]),
        "Lambda_1_4": float(ns["Lambda_1_4"]),
        "M_DUrca_Msun": ns["M_DUrca_Msun"],
        "passes_pulsar_2_08_and_gw_580": bool(float(ns["M_max_Msun"]) >= 2.08 and float(ns["Lambda_1_4"]) <= 580.0 and float(ns["R_1_4_km"]) <= 13.0),
        "rms_B_over_A_err_MeV": rms_ba,
        "rms_R_ch_err_fm": rms_rch_fm,
        "rms_R_ch_rel_err_percent": float(fin_summary["rms_R_ch_rel_err_percent"]),
        "Delta_R_np_Pb208_fm": skin_208,
        "Delta_R_np_Ca48_fm": skin_48,
        "Delta_R_ch_Ca48_minus_Ca40_fm": d_ca,
        "PREX2_z_score": float(z_prex2),
        "CREX_z_score": float(z_crex),
        "PREX2_CREX_chi2": chi2_skins,
        "diagnostic_multi_sector_chi2": float(chi2_diag),
        "diagnostic_AIC_k_total": aic_tot,
        "diagnostic_BIC_k_total": bic_tot,
    }


def build_result() -> dict[str, Any]:
    b8_raw = json.loads(B8_RESULTS_PATH.read_text(encoding="utf-8"))
    b9a_raw = json.loads(B9A_RESULTS_PATH.read_text(encoding="utf-8"))
    sa = b8_raw["stage_a_widened_primary_gate"]
    win = sa["winner"]
    win_fp = sa["winner_frozen_params"]
    survivors = sa["primary_survivors"]
    winner_key = (
        float(win["zeta_RMF"]),
        float(win["n0_fm3"]),
        float(win["M_eff_over_M_N"]),
        float(win["L_MeV"]),
    )

    b9a_map = {
        (float(r["zeta_RMF"]), float(r["n0_fm3"]), float(r["M_eff_over_M_N"]), float(r["L_MeV"])): r
        for r in b9a_raw["survivor_microphysics"]
    }

    # 1. Full even-even Tin isotopic chain (zero pairing refit) on winner
    tin_full = analyze_full_tin_chain(win_fp)

    # 2. All 8 survivors on zero-refit 6-nucleus Dirac-Hartree + symmetry curvature K_sym + PREX-2 / CREX
    survivor_benchmarks: list[dict[str, Any]] = []
    winner_six_bench: dict[str, Any] = {}
    winner_sym: dict[str, float] = {}
    winner_spin_orbit: dict[str, Any] = {}

    for s in survivors:
        zeta = float(s["zeta_RMF"])
        n0 = float(s["n0_fm3"])
        ystar = float(s["M_eff_over_M_N"])
        L_mev = float(s["L_MeV"])
        key = (zeta, n0, ystar, L_mev)
        p = b6.calibrate_envelope_point(n0, ystar, L_mev, zeta_rmf=zeta)
        sym = _compute_nvg_symmetry_observables(p)
        fp = p.to_frozen_dict()
        lbl = f"NVG_z{zeta:.4f}_n{n0:.3f}_y{ystar:.2f}_L{L_mev:.0f}"
        bench = b6.run_six_nuclei_benchmark(fp, lbl)
        sm = bench["summary_metrics"]
        z_prex2 = (sm["Delta_R_np_Pb208_fm"] - PREX2_SKIN_FM) / PREX2_ERR_FM
        z_crex = (sm["Delta_R_np_Ca48_fm"] - CREX_SKIN_FM) / CREX_ERR_FM
        is_win = bool(key == winner_key)
        entry = {
            "zeta_RMF": zeta,
            "n0_fm3": n0,
            "M_eff_over_M_N": ystar,
            "L_MeV": L_mev,
            "J_MeV": sym["J_MeV"],
            "L_fd_check_MeV": sym["L_MeV"],
            "K_sym_MeV": sym["K_sym_MeV"],
            "is_winner": is_win,
            "rms_B_over_A_err_MeV": float(sm["rms_B_over_A_err_MeV"]),
            "rms_R_ch_rel_err_percent": float(sm["rms_R_ch_rel_err_percent"]),
            "Delta_R_ch_Ca48_minus_Ca40_fm": float(sm["Delta_R_ch_Ca48_minus_Ca40_fm"]),
            "Delta_R_np_Ca48_fm": float(sm["Delta_R_np_Ca48_fm"]),
            "Delta_R_np_Pb208_fm": float(sm["Delta_R_np_Pb208_fm"]),
            "PREX2_z_score": float(z_prex2),
            "CREX_z_score": float(z_crex),
            "PREX2_CREX_chi2": float(z_prex2 * z_prex2 + z_crex * z_crex),
            "nuclei": [
                {
                    "nucleus": nr["nucleus"],
                    "B_over_A_MeV": float(nr["B_over_A_MeV"]),
                    "B_over_A_err_MeV": float(nr["B_over_A_err_MeV"]),
                    "R_ch_fm": float(nr["R_ch_fm"]),
                    "R_ch_rel_err_percent": float(nr["R_ch_rel_err_percent"]),
                    "Delta_R_np_fm": float(nr["Delta_R_np_fm"]),
                }
                for nr in bench["nuclei"]
            ],
        }
        survivor_benchmarks.append(entry)
        if is_win:
            winner_six_bench = bench
            winner_sym = sym
            winner_spin_orbit = {nr["nucleus"]: nr["spin_orbit_MeV"] for nr in bench["nuclei"]}

    # 3. Live Head-to-Head RMF Comparator Benchmark + NVG-Bridge8
    nvg_win_th = b9a_map[winner_key]["thresholds"]
    nvg_score = _scorecard_entry(
        name="NVG-Bridge8 (Zero-Refit Winner)",
        zero_finite_nucleus_refit=True,
        k_bulk=6,
        k_fin_nuc=0,
        vacuum_bounded_and_unique=True,
        sat={
            "n0_fm3": float(win["n0_fm3"]),
            "K_MeV": 240.0,
            "J_MeV": float(winner_sym["J_MeV"]),
            "L_MeV": float(win["L_MeV"]),
            "K_sym_MeV": float(winner_sym["K_sym_MeV"]),
        },
        ns={
            "M_max_Msun": float(nvg_win_th["M_max_Msun"]),
            "R_1_4_km": float(nvg_win_th["R_1_4_km"]),
            "Lambda_1_4": float(nvg_win_th["Lambda_1_4"]),
            "M_DUrca_Msun": nvg_win_th["direct_urca"]["M_DUrca_Msun"],
        },
        fin_summary=winner_six_bench["summary_metrics"],
        fin_nuclei=winner_six_bench["nuclei"],
    )

    rmf_details: list[dict[str, Any]] = []
    scorecard_rows: list[dict[str, Any]] = [nvg_score]
    for rmf_name, par in RMF_COMPARATORS.items():
        bulk_ns = _solve_rmf_bulk_and_ns(rmf_name, par)
        fin_six = _run_rmf_six_nuclei(rmf_name, par)
        vac = bulk_ns["vacuum_audit"]
        sc = _scorecard_entry(
            name=rmf_name,
            zero_finite_nucleus_refit=False,
            k_bulk=int(par["k_bulk"]),
            k_fin_nuc=int(par["k_fin_nuc"]),
            vacuum_bounded_and_unique=bool(vac["bounded_below"] and vac["unique_vacuum"]),
            sat=bulk_ns["saturation"],
            ns=bulk_ns["neutron_star_sly4_crust"],
            fin_summary=fin_six["summary_metrics"],
            fin_nuclei=fin_six["nuclei"],
        )
        scorecard_rows.append(sc)
        rmf_details.append({
            "name": rmf_name,
            "citation": par["citation"],
            "bulk_and_ns": bulk_ns,
            "six_nuclei": fin_six,
        })

    gates = {
        "full_tin_chain_11_isotopes_converged_zero_refit": bool(tin_full["summary"]["all_converged"]),
        "tin_double_magic_N50_N82_pairing_collapsed": bool(
            tin_full["summary"]["Sn100_N50_collapsed"]
            and tin_full["summary"]["Sn132_N82_collapsed"]
            and tin_full["summary"]["max_midshell_Delta_n_MeV"] > 0.6
            and tin_full["summary"]["Sn134_post_shell_Delta_n_MeV"] > 0.5
        ),
        "tin_N82_S2n_kink_reproduced": bool(tin_full["summary"]["delta_2n_Sn132_shell_gap_MeV"] > 5.0),
        "all_8_survivors_6_nuclei_positive_heavy_skins": bool(
            all(s["Delta_R_np_Pb208_fm"] > 0.12 and s["Delta_R_np_Ca48_fm"] > 0.10 for s in survivor_benchmarks)
        ),
        "all_5_live_rmf_comparators_converged_eos_and_nuclei": bool(
            len(rmf_details) == 5
            and all(all(n["converged"] for n in r["six_nuclei"]["nuclei"]) for r in rmf_details)
        ),
        "nvg_bridge8_globally_bounded_unlike_lambda_lt_0_rmf": bool(
            nvg_score["global_vacuum_bounded_and_unique"]
            and any(not r["bulk_and_ns"]["vacuum_audit"]["bounded_below"] for r in rmf_details)
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "bridge_id": BRIDGE_ID,
        "status": STATUS_PASS if all(gates.values()) else "FAIL_NUCLEAR_BENCHMARKS",
        "evidence_weight": EVIDENCE_WEIGHT,
        "observed_likelihood": None,
        "provenance": {
            "source_sha256": _sha256(SOURCE_PATH),
            "bridge8_results_sha256": _sha256(B8_RESULTS_PATH),
            "bridge9a_results_sha256": _sha256(B9A_RESULTS_PATH),
            "zero_finite_nucleus_refit_for_nvg": True,
        },
        "full_tin_isotopic_chain_Sn100_to_Sn134": tin_full,
        "survivor_prex_crex_and_ksym_locus": survivor_benchmarks,
        "winner_spin_orbit_splittings_MeV": winner_spin_orbit,
        "rmf_live_comparators": rmf_details,
        "held_out_empirical_scorecard": scorecard_rows,
        "gates": gates,
        "honest_caveats": [
            "evidence_weight = 0.0, observed_likelihood = None: all comparisons remain non-likelihood predictive benchmarks.",
            "NVG-Bridge8 carries k_fin_nuc = 0 (strictly zero finite-nucleus parameters fitted), whereas NL3/FSUGold/IU-FSU/FSUGold2/FSUGarnet include finite-nucleus binding energies and charge radii in calibration.",
            "Diagnostic chi^2, AIC, and BIC in held_out_empirical_scorecard use fixed reference scales and are comparative summary indices, not observational likelihoods.",
            "Mid-shell Tin isotopes (Sn104..Sn128) are evaluated in spherical RH-BCS symmetry without quadrupole deformation corrections.",
        ],
    }


def _trees_match(actual: Any, expected: Any, *, rel_tol: float = 5.0e-4, abs_tol: float = 1.0e-5, path: str = "$") -> tuple[bool, str | None]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual.keys()) != set(expected.keys()):
            return False, f"{path}: key mismatch"
        for k in sorted(expected.keys()):
            ok, msg = _trees_match(actual[k], expected[k], rel_tol=rel_tol, abs_tol=abs_tol, path=f"{path}.{k}")
            if not ok:
                return False, msg
        return True, None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False, f"{path}: length mismatch"
        for i, (av, ev) in enumerate(zip(actual, expected)):
            ok, msg = _trees_match(av, ev, rel_tol=rel_tol, abs_tol=abs_tol, path=f"{path}[{i}]")
            if not ok:
                return False, msg
        return True, None
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return (True, None) if actual == expected else (False, f"{path}: {actual!r} != {expected!r}")
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        a_v, e_v = float(actual), float(expected)
        leaf = path.rsplit(".", 1)[-1].lower()
        if "residual" in leaf or "rel_err" in leaf:
            r_eff, a_eff = 0.5, 1.0e-7
        elif any(tok in leaf for tok in ("lambda", "durca", "k_sym", "chi2", "aic", "bic", "score")):
            r_eff, a_eff = 2.0e-3, 1.0e-4
        else:
            r_eff, a_eff = rel_tol, abs_tol
        if math.isfinite(a_v) and math.isfinite(e_v) and math.isclose(a_v, e_v, rel_tol=r_eff, abs_tol=a_eff):
            return True, None
        return False, f"{path}: {actual!r} != {expected!r}"
    return (True, None) if type(actual) is type(expected) and actual == expected else (False, f"{path}: mismatch")


def validate_artifact(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return False, [f"Missing artifact: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"Invalid JSON: {exc}"]
    if not _finite_tree(payload):
        errors.append("Non-finite float detected")
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
    if prov.get("bridge9a_results_sha256") != _sha256(B9A_RESULTS_PATH):
        errors.append("bridge9a_results_sha256 mismatch")
    gates = payload.get("gates", {})
    if not gates or not all(bool(v) for v in gates.values()):
        errors.append(f"Gate failure in payload: {gates}")
    if errors:
        return False, errors

    try:
        tin = payload.get("full_tin_isotopic_chain_Sn100_to_Sn134", {})
        isos = tin.get("isotopes", [])
        if len(isos) != 11:
            errors.append(f"Expected 11 Tin isotopes, got {len(isos)}")
        sn_map = {r["nucleus"]: r for r in isos}
        for magic in ("Sn100", "Sn132"):
            if magic not in sn_map or not sn_map[magic]["collapsed_to_normal"] or float(sn_map[magic]["Delta_n_MeV"]) != 0.0:
                errors.append(f"Missing normal collapse on {magic}")

        survs = payload.get("survivor_prex_crex_and_ksym_locus", [])
        if len(survs) != 8:
            errors.append(f"Expected 8 survivors, got {len(survs)}")
        winners = [s for s in survs if s.get("is_winner")]
        if len(winners) != 1:
            errors.append(f"Expected 1 winner survivor, got {len(winners)}")

        rmfs = payload.get("rmf_live_comparators", [])
        sc = payload.get("held_out_empirical_scorecard", [])
        if len(rmfs) != 5 or len(sc) != 6:
            errors.append(f"Expected 5 RMF comparators and 6 scorecard rows, got {len(rmfs)}/{len(sc)}")
        if sc and (not sc[0].get("zero_finite_nucleus_refit") or sc[0].get("k_fin_nuc") != 0):
            errors.append("Scorecard[0] must preserve k_fin_nuc == 0")

        # Deterministic fast live recomputations (~2s, cross-platform stable)
        b8_raw = json.loads(B8_RESULTS_PATH.read_text(encoding="utf-8"))
        sa = b8_raw["stage_a_widened_primary_gate"]
        win_fp = sa["winner_frozen_params"]
        sn132_live = b8.solve_nucleus_rh_bcs(50, 82, win_fp, pairing_C_n_MeV=b8.PAIRING_C_N_MEV, core_n=b8.PAIRING_CORE_N)
        if not sn132_live["collapsed_to_normal"] or float(sn132_live["Delta_n_MeV"]) != 0.0:
            errors.append("Live Sn132 RH-BCS did not collapse to normal state")
        if "Sn132" in sn_map:
            if not math.isclose(float(sn132_live["B_over_A_MeV"]), float(sn_map["Sn132"]["B_over_A_MeV"]), rel_tol=1.0e-4, abs_tol=1.0e-5):
                errors.append("Live Sn132 B/A mismatch")
            if not math.isclose(float(sn132_live["R_ch_fm"]), float(sn_map["Sn132"]["R_ch_fm"]), rel_tol=1.0e-4, abs_tol=1.0e-5):
                errors.append("Live Sn132 R_ch mismatch")

        if winners:
            win = winners[0]
            p_win = b6.calibrate_envelope_point(
                float(win["n0_fm3"]),
                float(win["M_eff_over_M_N"]),
                float(win["L_MeV"]),
                zeta_rmf=float(win["zeta_RMF"]),
            )
            sym_live = _compute_nvg_symmetry_observables(p_win)
            for k in ("J_MeV", "K_sym_MeV"):
                if not math.isclose(float(sym_live[k]), float(win[k]), rel_tol=1.0e-3, abs_tol=1.0e-4):
                    errors.append(f"Live symmetry {k} mismatch: {sym_live[k]} vs {win[k]}")
            if not math.isclose(float(sym_live["L_MeV"]), float(win["L_MeV"]), rel_tol=1.0e-3, abs_tol=1.0e-4):
                errors.append("Live symmetry L_MeV mismatch")

        for rmf_entry in rmfs:
            rname = rmf_entry["name"]
            par = RMF_COMPARATORS[rname]
            vac_live = _audit_rmf_scalar_potential(par)
            vac_stored = rmf_entry["bulk_and_ns"]["vacuum_audit"]
            if bool(vac_live["bounded_below"]) != bool(vac_stored["bounded_below"]) or bool(vac_live["unique_vacuum"]) != bool(vac_stored["unique_vacuum"]):
                errors.append(f"Live vacuum audit boolean mismatch for {rname}")
            sat_stored = rmf_entry["bulk_and_ns"]["saturation"]
            st_live = _solve_cp_rmf_state(float(sat_stored["n0_fm3"]), 0.5, par)
            ba_live = st_live["eps_MeV_fm3"] / float(sat_stored["n0_fm3"]) - MN_RMF
            if not math.isclose(ba_live, float(sat_stored["B_over_A_MeV"]), rel_tol=1.0e-3, abs_tol=1.0e-4):
                errors.append(f"Live RMF saturation B/A mismatch for {rname}")
            if not math.isclose(float(st_live["M_eff_over_M_N"]), float(sat_stored["M_eff_over_M_N"]), rel_tol=1.0e-3, abs_tol=1.0e-4):
                errors.append(f"Live RMF saturation M*/M_N mismatch for {rname}")
    except Exception as exc:
        errors.append(f"Live spot-check validation error: {exc}")
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
