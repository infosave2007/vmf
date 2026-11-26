#!/usr/bin/env python3
"""Generate 5 publication-grade vector PDF + 300-DPI PNG figures for the Cycle-4 paper.

Reads verified Cycle-2/3/4 JSON outputs and evaluates lightweight plotting curves directly
from the verified Bridge-9A/9B solvers (zero parameter refitting).
Outputs go to ``preprints/figures_unified_paper/``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

for _blas_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = ROOT / "preprints" / "figures_unified_paper"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_two_condensate_unified_audit as law2  # noqa: E402
import nvg_unified_calibration_envelope_probe as b6  # noqa: E402
import nvg_unified_ns_microphysics_bridge as b9a  # noqa: E402
import nvg_unified_nuclear_benchmarks_bridge as b9b  # noqa: E402

B8_PATH = HERE / "nvg_unified_primary_gate_pairing_results.json"
B9A_PATH = HERE / "nvg_unified_ns_microphysics_results.json"
B9B_PATH = HERE / "nvg_unified_nuclear_benchmarks_results.json"
TH_PATH = HERE / "nvg_unified_thermal_response_results.json"
AST_PATH = HERE / "nvg_unified_astrophysics_stability_results.json"


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9.2,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.6,
        "axes.linewidth": 0.85,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def _save_fig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def _tov_curve_for_nvg(p: law2.ModelParams, n_dyn: float) -> list[dict[str, float]]:
    eos, tables = b9a._build_star_eos_and_tables(p, n_dyn)
    pcs = np.geomspace(4.0, 520.0, 38)
    pts: list[dict[str, float]] = []
    m_prev = -1.0
    for pc in pcs:
        try:
            m_v, r_v, k2_v, lam_v = law2.solve_tov_tidal_star(eos, float(pc), rtol=1.0e-7, atol=1.0e-9, max_step=0.12)
            pts.append({"Pc_MeV_fm3": float(pc), "M_Msun": float(m_v), "R_km": float(r_v), "k2": float(k2_v), "Lambda": float(lam_v)})
            if m_v < m_prev and m_v > 1.5:
                break
            m_prev = m_v
        except Exception:
            break
    return pts


def _tov_curve_for_rmf(par: dict[str, Any]) -> list[dict[str, float]]:
    n_grid = np.linspace(0.05, 1.35, 95)
    rows = [r for r in (b9b._solve_cp_rmf_beta_point(float(n), par) for n in n_grid) if r["P_MeV_fm3"] > 0.36]
    eos = law2.TabulatedCrustCoreEOS(
        np.array([r["P_MeV_fm3"] for r in rows]),
        np.array([r["eps_MeV_fm3"] for r in rows]),
        crust_mode="dh_sly4",
    )
    pcs = np.geomspace(5.0, 650.0, 36)
    pts: list[dict[str, float]] = []
    m_prev = -1.0
    for pc in pcs:
        try:
            m_v, r_v, k2_v, lam_v = law2.solve_tov_tidal_star(eos, float(pc), rtol=1.0e-7, atol=1.0e-9, max_step=0.12)
            pts.append({"Pc_MeV_fm3": float(pc), "M_Msun": float(m_v), "R_km": float(r_v), "k2": float(k2_v), "Lambda": float(lam_v)})
            if m_v < m_prev and m_v > 1.5:
                break
            m_prev = m_v
        except Exception:
            break
    return pts


def make_fig1(b8: dict, b9a_d: dict) -> None:
    sa = b8["stage_a_widened_primary_gate"]
    win = sa["winner"]
    fp = sa["winner_frozen_params"]
    p_win = b6.calibrate_envelope_point(win["n0_fm3"], win["M_eff_over_M_N"], win["L_MeV"], zeta_rmf=win["zeta_RMF"])
    win_9a = next(r for r in b9a_d["survivor_microphysics"] if r["is_winner"])
    n_dyn = win_9a["core_crust_transition"]["dynamical"]["n_t_fm3"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)

    # (a) Global scalar vacuum stability U(s)
    ax = axes[0, 0]
    s_grid = np.linspace(-2600.0, 420.0, 400)
    # NVG-Bridge8
    b_nvg = p_win.b3 * p_win.M_N * (p_win.g_s_ch ** 3)
    c_nvg = p_win.b4 * (p_win.g_s_ch ** 4)
    d_nvg = p_win.b6 * (p_win.g_s_ch ** 6)
    u_nvg = (0.5 * (p_win.m_s ** 2) * s_grid**2 + (b_nvg / 3.0) * s_grid**3 + 0.25 * c_nvg * s_grid**4 + (d_nvg / 6.0) * s_grid**6) / 1.0e11
    ax.plot(s_grid, np.arcsinh(u_nvg), color="#0072B2", lw=2.2, label="NVG-Bridge8 ($b_6 > 0$)")
    rmf_styles = [
        ("NL3", "#D55E00", "--"),
        ("IU-FSU", "#009E73", "-."),
        ("FSUGarnet", "#CC79A7", ":"),
    ]
    for nm, col, ls in rmf_styles:
        par = b9b.RMF_COMPARATORS[nm]
        gs = math.sqrt(par["g_s2"])
        bs = 0.5 * par["kappa"] * (gs**3)
        cs = (par["lam"] / 6.0) * (gs**4)
        u_r = (0.5 * (par["m_s"] ** 2) * s_grid**2 + (bs / 3.0) * s_grid**3 + 0.25 * cs * s_grid**4) / 1.0e11
        ax.plot(s_grid, np.arcsinh(u_r), color=col, lw=1.7, ls=ls, label=nm)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axvline(0.0, color="0.5", lw=0.7, ls=":")
    ax.set_xlabel(r"Scalar condensate $s$ (MeV)")
    ax.set_ylabel(r"$\mathrm{arcsinh}\!\left[U(s) / 10^{11}\,\mathrm{MeV}^4\right]$")
    ax.set_title(r"(a) Global Scalar Vacuum Stability ($U_{\rm ch}(s) > 0$)")
    ax.grid(True)
    ax.legend(loc="lower left", framealpha=0.92)

    # (b) M(R) and (c) Lambda(M)
    ax_mr = axes[0, 1]
    ax_lm = axes[1, 0]
    nvg_pts = _tov_curve_for_nvg(p_win, n_dyn)
    rmf_curves = {nm: _tov_curve_for_rmf(b9b.RMF_COMPARATORS[nm]) for nm in ("NL3", "FSUGold", "IU-FSU", "FSUGold2", "FSUGarnet")}

    # Plot observational bands first
    ax_mr.axhspan(2.01, 2.15, color="0.82", alpha=0.55, label=r"PSR J0740+6620 ($2.08\pm0.07\,M_\odot$)")
    ax_mr.axvline(13.0, color="#D55E00", lw=1.0, ls=":", alpha=0.8)
    ax_mr.text(13.06, 1.02, r"$R_{1.4}\leq 13.0$ km", color="#D55E00", fontsize=7.6, rotation=90, va="bottom")
    ax_mr.plot([c["R_km"] for c in nvg_pts], [c["M_Msun"] for c in nvg_pts], color="#0072B2", lw=2.4, label="NVG-Bridge8 (SLy4)")

    m_nvg = [c["M_Msun"] for c in nvg_pts]
    l_nvg = [max(c["Lambda"], 1.0) for c in nvg_pts]
    ax_lm.plot(m_nvg, l_nvg, color="#0072B2", lw=2.4, label="NVG-Bridge8")

    # Mark DUrca onset on NVG M(R)
    m_du = win_9a["thresholds"]["direct_urca"]["M_DUrca_Msun"]
    if m_du is not None:
        r_du = float(np.interp(m_du, [c["M_Msun"] for c in nvg_pts], [c["R_km"] for c in nvg_pts]))
        ax_mr.scatter([r_du], [m_du], color="#0072B2", edgecolor="k", s=52, zorder=6)
        ax_mr.annotate(f"DUrca ({m_du:.2f}$M_\\odot$)", xy=(r_du, m_du), xytext=(r_du - 2.05, m_du - 0.18),
                       fontsize=7.6, arrowprops=dict(arrowstyle="->", lw=0.8, color="#0072B2"))

    palette = {
        "NL3": ("#D55E00", "--"),
        "FSUGold": ("#E69F00", "-."),
        "IU-FSU": ("#009E73", ":"),
        "FSUGold2": ("#CC79A7", (0, (4, 1.5, 1, 1.5))),
        "FSUGarnet": ("#56B4E9", (0, (3, 1, 1, 1, 1, 1))),
    }
    for nm, pts in rmf_curves.items():
        col, ls = palette[nm]
        ax_mr.plot([c["R_km"] for c in pts], [c["M_Msun"] for c in pts], color=col, lw=1.45, ls=ls, label=nm)
        ax_lm.plot([c["M_Msun"] for c in pts], [max(c["Lambda"], 1.0) for c in pts], color=col, lw=1.45, ls=ls, label=nm)

    ax_mr.set_xlim(10.0, 15.8)
    ax_mr.set_ylim(0.5, 2.85)
    ax_mr.set_xlabel(r"Radius $R$ (km)")
    ax_mr.set_ylabel(r"Gravitational mass $M\ (M_\odot)$")
    ax_mr.set_title(r"(b) Cold $npe\mu$ Neutron-Star Mass–Radius Relation")
    ax_mr.grid(True)
    ax_mr.legend(loc="upper right", fontsize=7.2, framealpha=0.92, ncol=2)

    ax_lm.axhline(580.0, color="#D55E00", lw=1.1, ls="--", label=r"GW170817 $\Lambda_{1.4}\leq 580$")
    ax_lm.axvline(1.4, color="0.4", lw=0.8, ls=":")
    ax_lm.set_yscale("log")
    ax_lm.set_xlim(1.0, 2.35)
    ax_lm.set_ylim(10.0, 1.0e4)
    ax_lm.set_xlabel(r"Gravitational mass $M\ (M_\odot)$")
    ax_lm.set_ylabel(r"Tidal deformability $\Lambda(M)$")
    ax_lm.set_title(r"(c) Tidal Deformability vs. GW170817 Bound")
    ax_lm.grid(True, which="both")
    ax_lm.legend(loc="upper right", fontsize=7.3, framealpha=0.92, ncol=2)

    # (d) Bridge-8 widened primary gate landscape
    ax_g = axes[1, 1]
    all_rows = sa["all_grid_rows"]
    oth = [r for r in all_rows if not r["passes_primary_gate"]]
    surv = [r for r in all_rows if r["passes_primary_gate"]]
    ax_g.scatter([r["Lambda_1_4"] for r in oth], [r["M_max_Msun"] for r in oth], s=22, color="0.68", alpha=0.65, label="Grid points failing primary gate")
    ax_g.scatter([r["Lambda_1_4"] for r in surv], [r["M_max_Msun"] for r in surv], s=48, color="#009E73", edgecolor="k", zorder=5, label=f"Primary-gate survivors ($N={len(surv)}$)")
    ax_g.scatter([win["Lambda_1_4"]], [win["M_max_Msun"]], s=120, marker="*", color="#0072B2", edgecolor="k", zorder=6, label=r"Winner ($\zeta=0.0165,\,L=60$)")
    ax_g.axvline(580.0, color="#D55E00", lw=1.0, ls="--")
    ax_g.axhline(2.08, color="#D55E00", lw=1.0, ls="--")
    ax_g.fill_between([400.0, 580.0], 2.08, 2.22, color="#009E73", alpha=0.09)
    ax_g.set_xlim(420.0, 920.0)
    ax_g.set_ylim(1.94, 2.20)
    ax_g.set_xlabel(r"Canonical tidal deformability $\Lambda_{1.4}$")
    ax_g.set_ylabel(r"Maximum mass $M_{\max}\ (M_\odot)$")
    ax_g.set_title(r"(d) Bridge-8 Pre-Registered Primary Gate ($\zeta_{\rm RMF}\times n_0\times y^*\times L$)")
    ax_g.grid(True)
    ax_g.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

    _save_fig(fig, "fig1_vacuum_stability_and_ns_eos")


def make_fig2(b9b_d: dict) -> None:
    tin = b9b_d["full_tin_isotopic_chain_Sn100_to_Sn134"]
    isos = tin["isotopes"]
    summ = tin["summary"]
    ns = np.array([r["N"] for r in isos], dtype=float)
    ba_th = np.array([r["B_over_A_MeV"] for r in isos])
    ba_ex = np.array([r["exp_B_over_A_MeV"] for r in isos])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)

    # (a) B/A
    ax = axes[0, 0]
    ax.plot(ns, ba_ex, "o-", color="0.2", ms=4.8, lw=1.5, label="AME2020 (Exp.)")
    ax.plot(ns, ba_th, "s--", color="#0072B2", ms=4.8, lw=1.8, label=r"NVG RH-BCS (zero refit, $C_n=21$ MeV)")
    ax.axvline(50, color="0.5", ls=":", lw=0.9)
    ax.axvline(82, color="0.5", ls=":", lw=0.9)
    ax.text(50.6, -8.25, "$N=50$", fontsize=8.0, color="0.35")
    ax.text(80.8, -8.25, "$N=82$", fontsize=8.0, color="0.35", ha="right")
    ax.text(0.97, 0.08, f"RMS $|\\Delta(B/A)| = {summ['rms_B_over_A_err_MeV']:.3f}$ MeV", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0, bbox=dict(boxstyle="round,pad=0.25", fc="w", alpha=0.88))
    ax.set_ylim(-8.58, -7.81)
    ax.set_xlabel(r"Neutron number $N$")
    ax.set_ylabel(r"Binding energy per nucleon $B/A$ (MeV)")
    ax.set_title(r"(a) Even–Even Sn Chain Binding Energy ($A=100\text{–}134$)")
    ax.grid(True)
    ax.legend(loc="upper center", framealpha=0.92)

    # (b) Charge radius R_ch
    ax = axes[0, 1]
    r_rows = [r for r in isos if r["exp_R_ch_fm"] is not None]
    nr = np.array([r["N"] for r in r_rows])
    ax.plot(nr, [r["exp_R_ch_fm"] for r in r_rows], "o-", color="0.2", ms=4.8, lw=1.5, label="Angeli–Marinova (2013) + Gorges (2019)")
    ax.plot(ns, [r["R_ch_fm"] for r in isos], "s--", color="#0072B2", ms=4.5, lw=1.8, label="NVG RH-BCS (zero refit)")
    ax.axvline(82, color="#D55E00", ls=":", lw=1.1)
    ax.annotate(r"Shell kink $\delta R_{\rm ch}(134-132)$", xy=(83.0, 4.721), xytext=(66.0, 4.53),
                fontsize=7.6, arrowprops=dict(arrowstyle="->", lw=0.8, color="#D55E00"))
    ax.text(0.97, 0.08, f"RMS rel err = {summ['rms_R_ch_rel_err_percent']:.2f}%", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0, bbox=dict(boxstyle="round,pad=0.25", fc="w", alpha=0.88))
    ax.set_xlabel(r"Neutron number $N$")
    ax.set_ylabel(r"Charge radius $R_{\rm ch}$ (fm)")
    ax.set_title(r"(b) Sn Charge Radii & $N=82$ Shell Kink")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.3, framealpha=0.92)

    # (c) Neutron pairing gap Delta_n
    ax = axes[1, 0]
    deltas = [r["Delta_n_MeV"] for r in isos]
    ax.plot(ns, deltas, "D-", color="#009E73", ms=5.2, lw=1.9, label=r"Neutron BCS gap $\Delta_n$")
    ax.scatter([50, 82], [0.0, 0.0], color="#D55E00", edgecolor="k", s=65, zorder=5, label=r"Double-magic collapse ($\Delta_n = 0$)")
    ax.set_xlabel(r"Neutron number $N$")
    ax.set_ylabel(r"Pairing gap $\Delta_n$ (MeV)")
    ax.set_title(r"(c) Natural Shell Collapse at $N=50, 82$ ($C_n=21.0$ MeV)")
    ax.set_ylim(-0.08, max(deltas) + 0.25)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.92)

    # (d) Step-normalized S_2n
    ax = axes[1, 1]
    s_rows = [r for r in isos if r["S2n_step_MeV"] is not None]
    ns_s = np.array([r["N"] for r in s_rows])
    ax.plot(ns_s, [r["exp_S2n_step_MeV"] for r in s_rows], "o-", color="0.2", ms=4.8, lw=1.5, label="AME2020 (Exp.)")
    ax.plot(ns_s, [r["S2n_step_MeV"] for r in s_rows], "s--", color="#0072B2", ms=4.8, lw=1.8, label="NVG RH-BCS")
    ax.axvline(82, color="#D55E00", ls=":", lw=1.1)
    ax.text(0.05, 0.10, f"$\\delta_{{2n}}(^{{132}}\\mathrm{{Sn}})$ = {summ['delta_2n_Sn132_shell_gap_MeV']:.2f} MeV\nAME2020 = {summ['exp_delta_2n_Sn132_shell_gap_MeV']:.2f} MeV",
            transform=ax.transAxes, fontsize=8.0, bbox=dict(boxstyle="round,pad=0.28", fc="w", alpha=0.88))
    ax.set_xlabel(r"Neutron number $N$")
    ax.set_ylabel(r"Two-neutron separation proxy $S_{2n}$ (MeV)")
    ax.set_title(r"(d) $S_{2n}$ Trend & $N=82$ Shell Gap $\delta_{2n}$")
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.92)

    _save_fig(fig, "fig2_tin_isotopic_chain")


def make_fig3(b9b_d: dict) -> None:
    survs = b9b_d["survivor_prex_crex_and_ksym_locus"]
    sc = b9b_d["held_out_empirical_scorecard"]
    win_s = next(s for s in survs if s["is_winner"])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), constrained_layout=True)

    # (a) PREX-2 vs CREX plane
    ax = axes[0]
    ell1 = Ellipse((b9b.PREX2_SKIN_FM, b9b.CREX_SKIN_FM), 2 * b9b.PREX2_ERR_FM, 2 * b9b.CREX_ERR_FM,
                   facecolor="#56B4E9", alpha=0.22, edgecolor="#0072B2", lw=1.1, label=r"PREX-2 $\times$ CREX ($1\sigma$)")
    ell2 = Ellipse((b9b.PREX2_SKIN_FM, b9b.CREX_SKIN_FM), 4 * b9b.PREX2_ERR_FM, 4 * b9b.CREX_ERR_FM,
                   facecolor="none", edgecolor="#0072B2", lw=0.9, ls=":", label=r"PREX-2 $\times$ CREX ($2\sigma$)")
    ax.add_patch(ell1)
    ax.add_patch(ell2)

    ax.scatter([s["Delta_R_np_Pb208_fm"] for s in survs], [s["Delta_R_np_Ca48_fm"] for s in survs],
               s=45, color="#009E73", edgecolor="k", zorder=5, label=r"NVG 8 Survivors ($k_{\rm fin,nuc}=0$)")
    ax.scatter([win_s["Delta_R_np_Pb208_fm"]], [win_s["Delta_R_np_Ca48_fm"]],
               s=125, marker="*", color="#0072B2", edgecolor="k", zorder=6, label="NVG-Bridge8 Winner")

    rmf_cols = {"NL3": "#D55E00", "FSUGold": "#E69F00", "IU-FSU": "#CC79A7", "FSUGold2": "#8C564B", "FSUGarnet": "#7F7F7F"}
    offsets = {"NL3": (0.005, -0.014), "FSUGold": (0.006, 0.004), "IU-FSU": (0.006, -0.010), "FSUGold2": (-0.015, 0.008), "FSUGarnet": (-0.032, -0.014)}
    for r in sc[1:]:
        nm = r["model"]
        ax.scatter([r["Delta_R_np_Pb208_fm"]], [r["Delta_R_np_Ca48_fm"]], s=55, marker="D", color=rmf_cols[nm], edgecolor="k", zorder=5)
        dx, dy = offsets[nm]
        ax.text(r["Delta_R_np_Pb208_fm"] + dx, r["Delta_R_np_Ca48_fm"] + dy, nm, fontsize=7.8, fontweight="bold", color=rmf_cols[nm])

    ax.set_xlim(0.11, 0.36)
    ax.set_ylim(0.09, 0.25)
    ax.set_xlabel(r"$\Delta R_{np}(^{208}\mathrm{Pb})$ (fm) [PREX-2]")
    ax.set_ylabel(r"$\Delta R_{np}(^{48}\mathrm{Ca})$ (fm) [CREX]")
    ax.set_title(r"(a) PREX-2 vs. CREX Neutron-Skin Plane")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.3, framealpha=0.92)

    # (b) Held-out Finite-Nucleus Scorecard (RMS B/A vs RMS R_ch%)
    ax = axes[1]
    labels = [r["model"].replace(" (Zero-Refit Winner)", "\n($k_{\\rm fin}=0$)") for r in sc]
    x = np.arange(len(sc))
    w = 0.36
    ba_vals = [r["rms_B_over_A_err_MeV"] for r in sc]
    rc_vals = [r["rms_R_ch_rel_err_percent"] for r in sc]
    b1 = ax.bar(x - w / 2, ba_vals, width=w, color="#0072B2", alpha=0.88, label=r"RMS $|\Delta(B/A)|$ (MeV)")
    b2 = ax.bar(x + w / 2, rc_vals, width=w, color="#E69F00", alpha=0.88, label=r"RMS $|\Delta R_{\rm ch}/R_{\rm ch}|$ (%)")

    for idx, r in enumerate(sc):
        badge = "NS+Vac PASS" if (r["passes_pulsar_2_08_and_gw_580"] and r["global_vacuum_bounded_and_unique"]) else ("Vac FAIL" if not r["global_vacuum_bounded_and_unique"] else "NS FAIL")
        col_b = "#009E73" if badge == "NS+Vac PASS" else "#D55E00"
        top_y = max(ba_vals[idx], rc_vals[idx]) + 0.04
        ax.text(x[idx], top_y, badge, ha="center", va="bottom", fontsize=6.8, color=col_b, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylim(0.0, max(max(ba_vals), max(rc_vals)) * 1.26)
    ax.set_ylabel(r"6-Nucleus RMS Error (MeV or %)")
    ax.set_title(r"(b) 6-Nucleus Error ($k_{\rm fin}=0$ NVG vs. Fitted RMF)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92)

    _save_fig(fig, "fig3_prex_crex_and_rmf_scorecard")


def make_fig4(th: dict, b9a_d: dict) -> None:
    fqz = th["finite_qz_master_response"]["rows"]
    wm = b9a_d["winner_microphysics"]
    beta_prof = wm["beta_equilibrium_profile"]
    mmax_nc = next(s["thresholds"]["n_c_max_fm3"] for s in b9a_d["survivor_microphysics"] if s["is_winner"])
    modes = wm["cowling_asteroseismology_l2"]
    n0 = next(s["n0_fm3"] for s in b9a_d["survivor_microphysics"] if s["is_winner"])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)

    # (a) Thermal Ward identity across 12 (mu_B, z) master points
    ax = axes[0, 0]
    pts_x = np.arange(len(fqz))
    full_w = [max(r["ward_max_rel_err"], 1.0e-16) for r in fqz]
    ph_w = [max(r["ph_only_negative_control_ward_rel_err"], 1.0e-16) for r in fqz]
    ax.semilogy(pts_x, full_w, "o-", color="#0072B2", ms=5.0, lw=1.6, label="Full Schur RPA + meson feedback")
    ax.semilogy(pts_x, ph_w, "x--", color="#D55E00", ms=5.5, lw=1.5, label="Particle–hole-only control (no meson feedback)")
    ax.set_xticks(pts_x)
    ax.set_xticklabels([f"{int(r['mu_B_MeV'])}\n{r['z_label']}" for r in fqz], fontsize=6.7)
    ax.set_ylim(1.0e-17, 1.0e3)
    ax.set_xlabel(r"Master point $(\mu_B\ [\mathrm{MeV}],\,z_j)$ at $T=70$ MeV, $q=100$ MeV")
    ax.set_ylabel(r"Max relative Ward residual")
    ax.set_title(r"(a) Exact Finite-$T$ Ward Identity vs. $ph$-Only Control")
    ax.grid(True, which="both")
    ax.legend(loc=(0.03, 0.34), fontsize=7.3, framealpha=0.94)

    # (b) Sound speeds cs^2 vs ce^2
    ax = axes[0, 1]
    bp = [r for r in beta_prof if r["n_fm3"] <= mmax_nc + 0.08]
    un = np.array([r["n_fm3"] / n0 for r in bp])
    ax.plot(un, [r["cs2"] for r in bp], color="#0072B2", lw=2.0, label=r"Frozen sound speed $c_s^2$")
    ax.plot(un, [r["ce2"] for r in bp], color="#009E73", lw=1.8, ls="--", label=r"Equilibrium sound speed $c_e^2$")
    ax.plot(un, [10.0 * r["cs2_minus_ce2"] for r in bp], color="#CC79A7", lw=1.6, ls="-.", label=r"$10\times(c_s^2 - c_e^2)$")
    ax.axvline(mmax_nc / n0, color="0.4", lw=0.9, ls=":")
    ax.text(mmax_nc / n0 - 0.12, 0.72, r"$n_c(M_{\max})$", fontsize=7.6, color="0.3", ha="right")
    ax.axhline(1.0, color="#D55E00", lw=1.0, ls=":")
    ax.set_xlabel(r"Normalized baryon density $n_B / n_0$")
    ax.set_ylabel(r"Dimensionless squared speed")
    ax.set_title(r"(b) Cold $npe\mu$ Le Chatelier Ordering ($c_s^2 > c_e^2$)")
    ax.set_ylim(0.0, 0.92)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.4, framealpha=0.92)

    # (c) Cowling l=2 f-mode vs M
    ax = axes[1, 0]
    ms = np.array([m["M_Msun"] for m in modes])
    f_khz = np.array([m["f_mode_kHz"] for m in modes])
    ax.plot(ms, f_khz, "o-", color="#0072B2", lw=2.0, ms=5.5, label=r"NVG-Bridge8 $\ell=2$ $f$-mode")
    kel = wm["controls"]["uniform_kelvin_cowling"]
    ax.text(0.05, 0.90, f"Uniform Kelvin control:\n$\\omega_{{\\rm Cowling}}/\\omega_{{\\rm Kelvin}} = {kel['omega_ratio']:.5f}$",
            transform=ax.transAxes, va="top", fontsize=7.8, bbox=dict(boxstyle="round,pad=0.25", fc="w", alpha=0.88))
    ax.set_xlabel(r"Gravitational mass $M\ (M_\odot)$")
    ax.set_ylabel(r"Frequency $\nu_f$ (kHz)")
    ax.set_title(r"(c) Relativistic Cowling $\ell=2$ Fundamental $f$-Mode")
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.92)

    # (d) Compositional g_1 & g_2 modes + barotropic null
    ax = axes[1, 1]
    g1 = np.array([m["g1_mode_Hz"] for m in modes])
    g2 = np.array([m["g2_mode_Hz"] for m in modes])
    ax.plot(ms, g1, "s-", color="#009E73", lw=2.0, ms=5.2, label=r"Compositional $g_1$-mode")
    ax.plot(ms, g2, "D--", color="#CC79A7", lw=1.8, ms=5.0, label=r"Compositional $g_2$-mode")
    ax.axhline(0.0, color="#D55E00", lw=1.2, ls=":", label=r"Barotropic null ($N^2 \equiv 0 \Rightarrow 0$ modes)")
    ax.set_xlabel(r"Gravitational mass $M\ (M_\odot)$")
    ax.set_ylabel(r"Frequency $\nu_g$ (Hz)")
    ax.set_ylim(-14.0, 240.0)
    ax.set_title(r"(d) Compositional $g_1 / g_2$ Asteroseismology Spectrum")
    ax.grid(True)
    ax.legend(loc=(0.30, 0.14), fontsize=7.4, framealpha=0.94)

    _save_fig(fig, "fig4_thermal_ward_and_asteroseismology")


def make_fig5(b8: dict, b9a_d: dict, ast: dict) -> None:
    survs = b9a_d["survivor_microphysics"]
    win_9a = next(r for r in survs if r["is_winner"])
    p_win = b6.calibrate_envelope_point(win_9a["n0_fm3"], win_9a["M_eff_over_M_N"], win_9a["L_MeV"], zeta_rmf=win_9a["zeta_RMF"])
    nt_dyn = win_9a["core_crust_transition"]["dynamical"]["n_t_fm3"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)

    # (a) Finite-k dynamical spinodal eigenvalues
    ax = axes[0, 0]
    k_grid = np.linspace(1.0, 145.0, 45)
    for n_val, lbl, col, ls in [
        (nt_dyn + 0.008, f"$n_B = {nt_dyn + 0.008:.4f}\\ \\mathrm{{fm}}^{{-3}}$ (stable)", "#009E73", "--"),
        (nt_dyn, f"$n_t^{{\\rm dyn}} = {nt_dyn:.4f}\\ \\mathrm{{fm}}^{{-3}}$ (critical)", "#0072B2", "-"),
        (nt_dyn - 0.006, f"$n_B = {nt_dyn - 0.006:.4f}\\ \\mathrm{{fm}}^{{-3}}$ (unstable)", "#D55E00", "-."),
    ]:
        y_eig = [b9a._nucleon_hessian_schur(p_win, n_val, k_values=np.array([float(kv)]))["lam_dyn_MeVminus2"] for kv in k_grid]
        ax.plot(k_grid, y_eig, color=col, lw=1.85, ls=ls, label=lbl)
    ax.axhline(0.0, color="0.3", lw=0.9)
    ax.scatter([win_9a["core_crust_transition"]["dynamical"]["k_opt_MeV"]], [0.0], color="#0072B2", edgecolor="k", s=55, zorder=5)
    ax.set_xlabel(r"Wavevector $k$ (MeV)")
    ax.set_ylabel(r"$\lambda_{\min}[K_{\rm dyn}(k)]$ ($\mathrm{MeV}^{-2}$)")
    ax.set_title(r"(a) Dynamical Core–Crust Spinodal Dispersion ($k_{\rm opt} \approx 62$ MeV)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7.4, framealpha=0.92)

    # (b) Dynamical vs thermodynamical n_t across all 8 survivors
    ax = axes[0, 1]
    l_vals = np.array([s["L_MeV"] for s in survs])
    nt_th = np.array([s["core_crust_transition"]["thermodynamical"]["n_t_fm3"] for s in survs])
    nt_dy = np.array([s["core_crust_transition"]["dynamical"]["n_t_fm3"] for s in survs])
    ax.scatter(l_vals, nt_th, s=52, marker="s", color="#D55E00", edgecolor="k", alpha=0.88, label=r"Thermodynamical $n_t^{\rm ther}$ ($k=0$)")
    ax.scatter(l_vals, nt_dy, s=55, marker="o", color="#0072B2", edgecolor="k", alpha=0.90, label=r"Dynamical $n_t^{\rm dyn}$ (mesons + Coulomb)")
    ax.scatter([win_9a["L_MeV"]], [nt_dyn], s=135, marker="*", color="#009E73", edgecolor="k", zorder=6, label="Bridge-8 Winner")
    ax.set_xlabel(r"Symmetry slope $L$ (MeV)")
    ax.set_ylabel(r"Transition density $n_t$ ($\mathrm{fm}^{-3}$)")
    ax.set_title(r"(b) Core–Crust Transition across 8 Primary-Gate Survivors")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92)

    # (c) 1.4 M_sun I_crust/I vs Vela glitch bound
    ax = axes[1, 0]
    ic_pct = [100.0 * s["star_1_4_crustal_inertia"]["I_crust_over_I"] for s in survs]
    s_labels = [f"z{s['zeta_RMF']:.4f}\ny{s['M_eff_over_M_N']:.2f}\nL{int(s['L_MeV'])}" + ("*" if s["is_winner"] else "") for s in survs]
    cols = ["#0072B2" if s["is_winner"] else "#56B4E9" for s in survs]
    xp = np.arange(len(survs))
    ax.bar(xp, ic_pct, color=cols, edgecolor="k", lw=0.6)
    ax.axhline(1.4, color="#D55E00", lw=1.3, ls="--", label=r"Vela glitch minimum ($I_{\rm crust}/I = 1.4\%$)")
    ax.set_xticks(xp)
    ax.set_xticklabels(s_labels, fontsize=6.7)
    ax.set_ylim(0.0, max(ic_pct) * 1.18)
    ax.set_ylabel(r"Crustal inertia fraction $I_{\rm crust}/I$ (%)")
    ax.set_title(r"(c) $1.4\,M_\odot$ Crustal Inertia vs. Vela Pulsar Glitch Bound")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    # (d) Hartle I-Love-Q Yagi-Yunes Universality
    ax = axes[1, 1]
    l_curve = np.geomspace(18.0, 2000.0, 200)
    ln_l = np.log(l_curve)
    # Yagi & Yunes (2013, 2017) fit coefficients already stored in b8 / ast bridge
    ci = (1.496, 0.05951, 0.02238, -6.953e-4, 8.345e-6)
    cq = (0.194, 0.0936, 0.0474, -4.211e-3, 1.231e-4)
    ibar_yy = np.exp(ci[0] + ci[1]*ln_l + ci[2]*ln_l**2 + ci[3]*ln_l**3 + ci[4]*ln_l**4)
    qbar_yy = np.exp(cq[0] + cq[1]*ln_l + cq[2]*ln_l**2 + cq[3]*ln_l**3 + cq[4]*ln_l**4)
    ax.loglog(l_curve, ibar_yy, color="#0072B2", lw=1.7, label=r"Yagi–Yunes universal $\bar{I}(\bar{\Lambda})$")
    ax.loglog(l_curve, qbar_yy, color="#009E73", lw=1.7, ls="--", label=r"Yagi–Yunes universal $\bar{Q}(\bar{\Lambda})$")

    b8_stars = b8["stage_c_cross_bridge_verification"]["hartle_iloveq_stars"]
    ast_stars = ast["part_a_hartle_iloveq"]["benchmark_stars"]
    ax.scatter([s["Lambda"] for s in b8_stars], [s["Ibar"] for s in b8_stars], s=65, marker="*", color="#D55E00", edgecolor="k", zorder=6, label=r"NVG-Bridge8 ($\Delta \leq 0.21\%$)")
    ax.scatter([s["Lambda"] for s in b8_stars], [s["Qbar"] for s in b8_stars], s=55, marker="s", color="#CC79A7", edgecolor="k", zorder=6)
    ax.scatter([s["Lambda"] for s in ast_stars], [s["Ibar"] for s in ast_stars], s=32, marker="o", color="0.3", alpha=0.7, label="Bridge-3 Law-2 stars")
    ax.scatter([s["Lambda"] for s in ast_stars], [s["Qbar"] for s in ast_stars], s=28, marker="o", color="0.3", alpha=0.7)
    ax.set_xlabel(r"Dimensionless tidal deformability $\bar{\Lambda}$")
    ax.set_ylabel(r"Dimensionless $\bar{I}$ and $\bar{Q}$")
    ax.set_title(r"(d) Hartle Slow-Rotation $I$-Love-$Q$ Universality")
    ax.grid(True, which="both")
    ax.legend(loc="upper left", fontsize=7.2, framealpha=0.92)

    _save_fig(fig, "fig5_core_crust_spinodal_and_iloveq")


def main() -> int:
    _configure_style()
    b8 = json.loads(B8_PATH.read_text(encoding="utf-8"))
    b9a_d = json.loads(B9A_PATH.read_text(encoding="utf-8"))
    b9b_d = json.loads(B9B_PATH.read_text(encoding="utf-8"))
    th = json.loads(TH_PATH.read_text(encoding="utf-8"))
    ast = json.loads(AST_PATH.read_text(encoding="utf-8"))

    make_fig1(b8, b9a_d)
    make_fig2(b9b_d)
    make_fig3(b9b_d)
    make_fig4(th, b9a_d)
    make_fig5(b8, b9a_d, ast)
    print(f"Wrote 5 PDF+PNG figure pairs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
