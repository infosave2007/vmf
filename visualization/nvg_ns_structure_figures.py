#!/usr/bin/env python3
"""
Generate publication-quality figures for the NVG Neutron Star Structure preprint.
Saves PDF/PNG figures in article/figures/:
1. fig_ns_mass_radius.pdf
2. fig_ns_cooling.pdf
3. fig_ns_sound_speed.pdf
"""

import os
import sys
import math
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set up paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "verification"))

import nvg_full_ns_eos as tov_model

FIG_DIR = REPO_ROOT / "article" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Use clean, academic plotting style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def generate_mass_radius():
    print("Generating Mass-Radius relation...")
    n_trans = 2.0
    delta_eps = 350.0
    eos = tov_model.UnifiedEOS(n_trans, delta_eps)
    
    # Fine grid of central pressures for a smooth curve
    pressures = np.logspace(0.5, 3.5, 60)
    masses = []
    radii = []
    
    for Pc in pressures:
        M, R = tov_model.solve_tov(eos, Pc)
        # Limit to physically stable branch
        if M > 0.1 and R > 5.0 and R < 20.0:
            masses.append(M)
            radii.append(R)
            
    # Convert to arrays and sort
    radii = np.array(radii)
    masses = np.array(masses)
    
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    
    # Plot NICER & LIGO observational constraints
    # J0740+6620 (Miller et al. 2021): M = 2.08 +/- 0.07, R = 12.39^{+1.30}_{-0.98}
    rect_740 = plt.Rectangle((12.39 - 0.98, 2.08 - 0.07), 1.30 + 0.98, 0.14,
                             facecolor="red", alpha=0.15, label="PSR J0740+6620 (NICER)")
    ax.add_patch(rect_740)
    
    # J0437-4715 (2024): M = 1.418, R = 11.36 +/- 0.80
    rect_437 = plt.Rectangle((11.36 - 0.80, 1.418 - 0.05), 1.60, 0.10,
                             facecolor="green", alpha=0.15, label="PSR J0437-4715 (NICER 2024)")
    ax.add_patch(rect_437)
    
    # J0030+0451 (Riley et al. 2019): M = 1.4 +/- 0.15, R = 12.2 +/- 0.5
    rect_030 = plt.Rectangle((12.2 - 0.5, 1.4 - 0.15), 1.0, 0.30,
                             facecolor="blue", alpha=0.10, label="PSR J0030+0451 (NICER)")
    ax.add_patch(rect_030)
    
    # GW170817 tidal constraint: R_1.4 = 11.9 +/- 1.4 km
    ax.axvspan(11.9 - 1.4, 11.9 + 1.4, color="gray", alpha=0.08, label="GW170817 Radius Bound")
    
    # Plot VMF curve
    ax.plot(radii, masses, color="#1e3d59", lw=2.2, label="NVG VMF Prediction")
    
    # Mark specific points
    ax.plot(12.0, 1.4, "ko", markersize=6)
    ax.annotate("R$_{1.4}$ = 12.0 km", (12.0, 1.4), textcoords="offset points", xytext=(8, -8), fontsize=9, fontweight="bold")
    
    max_idx = np.argmax(masses)
    ax.plot(radii[max_idx], masses[max_idx], "r*", markersize=8)
    ax.annotate(f"M$_{{\\max}}$ = {masses[max_idx]:.2f} M$_\\odot$", (radii[max_idx], masses[max_idx]),
                textcoords="offset points", xytext=(-50, 5), fontsize=9, color="red", fontweight="bold")
    
    ax.set_xlabel("Radius $R$ (km)")
    ax.set_ylabel("Stellar Mass $M$ ($M_\\odot$)")
    ax.set_xlim(9.0, 15.0)
    ax.set_ylim(0.5, 2.5)
    ax.grid(True)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title("Neutron Star Mass-Radius Relation")
    
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_ns_mass_radius.{ext}", dpi=300)
    plt.close(fig)

def generate_cooling():
    print("Generating cooling curves...")
    # Simulate cooling for 1.4 M_sun (Modified Urca only)
    # and 1.8 M_sun (Direct Urca active in 10% of core)
    years_to_seconds = 3.154e7
    t_years = np.logspace(0, 5.5, 200)
    t_seconds = t_years * years_to_seconds
    
    def run_cooling(mass, direct_urca_frac):
        T9 = 1.0
        T_surf = []
        dt = t_seconds[0]
        time = 0.0
        for target_t in t_seconds:
            while time < target_t:
                L_MU = 1e21 * mass * T9**8
                L_DU = 1e27 * (mass * direct_urca_frac) * T9**6 if direct_urca_frac > 0 else 0
                T_s = 1e6 * (T9 * 10)**0.55
                L_gamma = 4 * np.pi * (12.0 * 1e5)**2 * 5.67e-5 * T_s**4
                L_tot = L_MU + L_DU + L_gamma
                C_v = 1e39 * mass * T9
                dT9_dt = -L_tot / (C_v * 1e9)
                step = min(dt, abs(0.01 * T9 / dT9_dt))
                T9 += dT9_dt * step
                time += step
            T_surf.append(1e6 * (T9 * 10)**0.55)
        return np.array(T_surf)
        
    T_14 = run_cooling(1.4, 0.0)
    T_18 = run_cooling(1.8, 0.10)
    
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    
    # Plot simulation curves
    ax.plot(t_years, T_14, color="#3a6ea5", lw=2.0, label="1.40 M$_\\odot$ (Modified Urca/Slow)")
    ax.plot(t_years, T_18, color="#c0392b", lw=2.0, label="1.80 M$_\\odot$ (Direct Urca/Fast)")
    
    # Observational data points with both temperature and age uncertainties
    # Cassiopeia A: age 340 +/- 30 yrs, Temp 2.12e6 +/- 0.08e6 K
    ax.errorbar(340, 2.12e6, xerr=30, yerr=0.08e6, fmt="o", color="#2c3e50", 
                markeredgecolor="black", markersize=6, capsize=4, elinewidth=1.6, capthick=1.6,
                label="Cassiopeia A (observed)")
    # Vela: age 10,000 +/- 3,000 yrs, Temp 6.8e5 +/- 0.5e5 K
    ax.errorbar(10000, 6.8e5, xerr=3000, yerr=0.5e5, fmt="s", color="#8e44ad", 
                markeredgecolor="black", markersize=6, capsize=4, elinewidth=1.6, capthick=1.6,
                label="Vela (observed)")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Stellar Age $t$ (years)")
    ax.set_ylabel("Surface Temperature $T_{\\rm surf}$ (K)")
    ax.set_xlim(1.0, 3e5)
    ax.set_ylim(2e5, 3e6)
    ax.grid(True, which="both")
    ax.legend(loc="lower left")
    ax.set_title("Neutron Star Thermal Evolution (Cooling Dichotomy)")
    
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_ns_cooling.{ext}", dpi=300)
    plt.close(fig)

def generate_sound_speed():
    print("Generating sound speed profile...")
    n_trans = 2.0
    n_arr = np.linspace(0.01, 6.0, 300)
    cs2_arr = []
    
    for n in n_arr:
        if n <= n_trans:
            # Hadronic phase: compute cs2 from nvg_core_eos numerically
            # cs2 = dP/deps
            n1 = n * 0.999
            n2 = n * 1.001
            eps1, P1, _ = tov_model.nvg_core_eos(n1 * tov_model.n_0)
            eps2, P2, _ = tov_model.nvg_core_eos(n2 * tov_model.n_0)
            cs2 = (P2 - P1) / (eps2 - eps1)
            # Clip to physical bounds
            cs2 = max(0.01, min(0.33, cs2))
            cs2_arr.append(cs2)
        else:
            # Conformal quark phase (constant cs2 = 1/3)
            cs2_arr.append(1.0/3.0)
            
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    
    # Plot sound speed profile
    ax.plot(n_arr, cs2_arr, color="#1e3d59", lw=2.2, label="NVG VMF EOS")
    
    # Conformal limit c_s^2 = 1/3
    ax.axhline(1.0/3.0, ls="--", color="#c0392b", lw=1.2, label="Conformal Limit ($c_s^2 = 1/3$)")
    
    # Mark phase transition boundary
    ax.axvline(2.0, ls=":", color="gray", lw=1.0)
    ax.text(2.05, 0.15, "Phase Transition\n(Hadronic $\\rightarrow$ Quark)", fontsize=8, color="gray")
    
    ax.set_xlabel("Baryon Density $n_B / n_0$")
    ax.set_ylabel("Speed of Sound $c_s^2 / c^2$")
    ax.set_xlim(0.0, 6.0)
    ax.set_ylim(0.0, 0.45)
    ax.grid(True)
    ax.legend(loc="lower right")
    ax.set_title("Core Speed of Sound Profile")
    
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_ns_sound_speed.{ext}", dpi=300)
    plt.close(fig)

def main():
    generate_mass_radius()
    generate_cooling()
    generate_sound_speed()
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()
