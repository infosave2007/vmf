#!/usr/bin/env python3
"""
 NVG Hyperon EOS comparison: TOV Solver and Visualization.

This script solves the Tolman-Oppenheimer-Volkoff (TOV) equations for two
independent EOS baselines (stiff NL3 and soft SLy) in three scenarios:
1. Nucleon only (base hadronic matter)
2. Hadronic + Lambda hyperons (Hyperon Puzzle softening)
3. NVG + Lambda hyperons (VMF/NVG vacuum restoration)

It generates and saves three publication-quality figures:
- fig_hyperon_mr_baselines.png / .pdf (the two-panel baseline comparison)
- fig_hyperon_eos_compare.png / .pdf (the pressure vs. energy density EOS comparison)
- fig_hyperon_mr_nl3.png / .pdf (detailed standalone NL3 mass-radius plot)
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# --- CONSTANTS ---
G_c2 = 1.323e-6        # km/M_sun (G/c^2)
M_sun_km = 1.4766      # 1 M_sun in km (G M_sun / c^2)
conv_Pa_to_MeV_fm3 = 1.0e-33 / 1.6022e-13  # conversion factor
conv_MeV_fm3_to_geo = 1.3234e-6           # km^-2

## --- EQUATIONS OF STATE ---
# We parameterize the baseline equations of state using the standard piecewise polytrope
# representation of Read et al. 2009 (Phys. Rev. D 79, 124032), Table II.
# This represents the true physical NL3 (stiff) and SLy (soft) baselines.

def get_pp_eos(log_p1, gamma1, gamma2, gamma3):
    c = 2.99792458e10           # speed of light (cm/s)
    conv_cgs = 1.6021766e33     # 1 MeV/fm^3 in cgs (dyn/cm^2)
    gamma_cr = 1.3569239        # crust polytropic index
    K_cr = 3.5966e13            # crust polytropic constant (cgs)
    
    rho_1 = 10**14.7            # transition density 1 (g/cm^3)
    rho_2 = 10**15.0            # transition density 2 (g/cm^3)
    p1 = 10**log_p1             # pressure at rho_1 (dyn/cm^2)
    
    K_1 = p1 / (rho_1 ** gamma1)
    # Transition density from crust to core (pressure crossing point)
    rho_tr = (K_1 / K_cr) ** (1.0 / (gamma_cr - gamma1))
    
    K_2 = K_1 * (rho_1 ** (gamma1 - gamma2))
    K_3 = K_2 * (rho_2 ** (gamma2 - gamma3))
    
    # Integration constants for energy density continuity
    a_1 = (K_cr / (gamma_cr - 1.0)) * (rho_tr ** (gamma_cr - 1.0)) / (c**2) - (K_1 / (gamma1 - 1.0)) * (rho_tr ** (gamma1 - 1.0)) / (c**2)
    a_2 = a_1 + (K_1 / (gamma1 - 1.0)) * (rho_1 ** (gamma1 - 1.0)) / (c**2) - (K_2 / (gamma2 - 1.0)) * (rho_1 ** (gamma2 - 1.0)) / (c**2)
    a_3 = a_2 + (K_2 / (gamma2 - 1.0)) * (rho_2 ** (gamma2 - 1.0)) / (c**2) - (K_3 / (gamma3 - 1.0)) * (rho_2 ** (gamma3 - 1.0)) / (c**2)
    
    # Grid of rest-mass densities
    rho_grid = np.geomspace(1e3, 5e15, 1000)
    eps_grid = np.zeros_like(rho_grid)
    p_grid = np.zeros_like(rho_grid)
    
    for idx, rho in enumerate(rho_grid):
        if rho <= rho_tr:
            p = K_cr * (rho ** gamma_cr)
            eps = rho * c**2 + p / (gamma_cr - 1.0)
        elif rho <= rho_1:
            p = K_1 * (rho ** gamma1)
            eps = (1.0 + a_1) * rho * c**2 + p / (gamma1 - 1.0)
        elif rho <= rho_2:
            p = K_2 * (rho ** gamma2)
            eps = (1.0 + a_2) * rho * c**2 + p / (gamma2 - 1.0)
        else:
            p = K_3 * (rho ** gamma3)
            eps = (1.0 + a_3) * rho * c**2 + p / (gamma3 - 1.0)
            
        p_grid[idx] = p / conv_cgs
        eps_grid[idx] = eps / conv_cgs
        
    return eps_grid, p_grid

def get_nl3_eos(scenario="nucleon"):
    # NL3 parameters from Read et al. 2009
    eps, P_nuc = get_pp_eos(34.909, 2.77, 3.29, 2.47)
    
    if scenario == "nucleon":
        return eps, P_nuc
        
    elif scenario == "hyperon":
        P_hyp = np.copy(P_nuc)
        onset = 235.0
        mask = eps > onset
        P_hyp[mask] = P_nuc[mask] - 0.12 * (eps[mask] - onset)**1.15
        for i in range(1, len(P_hyp)):
            if P_hyp[i] < P_hyp[i-1] + 1e-15:
                P_hyp[i] = P_hyp[i-1] + 1e-15
        return eps, P_hyp
        
    elif scenario == "nvg":
        P_nvg = np.copy(P_nuc)
        onset = 235.0
        mask = eps > onset
        P_nvg[mask] = P_nuc[mask] - 0.12 * (eps[mask] - onset)**1.15
        
        # Add quadratic NVG repulsion
        P_add = 0.375 * (eps / 150.0)**2
        P_nvg += P_add
        
        nc_eps = 330.0
        mask_c = eps >= nc_eps
        P_nvg[mask_c] = P_nvg[mask_c] + 15.0  # de Sitter boost
        
        for i in range(1, len(P_nvg)):
            if P_nvg[i] < P_nvg[i-1] + 1e-15:
                P_nvg[i] = P_nvg[i-1] + 1e-15
        return eps, P_nvg

def get_sly_eos(scenario="nucleon"):
    # SLy parameters from Read et al. 2009
    eps, P_nuc = get_pp_eos(34.384, 3.005, 2.988, 2.851)
    
    if scenario == "nucleon":
        return eps, P_nuc
        
    elif scenario == "hyperon":
        P_hyp = np.copy(P_nuc)
        onset = 230.0
        mask = eps > onset
        P_hyp[mask] = P_nuc[mask] - 0.08 * (eps[mask] - onset)**1.12
        for i in range(1, len(P_hyp)):
            if P_hyp[i] < P_hyp[i-1] + 1e-15:
                P_hyp[i] = P_hyp[i-1] + 1e-15
        return eps, P_hyp
        
    elif scenario == "nvg":
        P_nvg = np.copy(P_nuc)
        onset = 230.0
        mask = eps > onset
        P_nvg[mask] = P_nuc[mask] - 0.08 * (eps[mask] - onset)**1.12
        
        # NVG quadratic restoration
        P_add = 0.375 * (eps / 130.0)**2.1
        P_nvg += P_add
        
        nc_eps = 328.0
        mask_c = eps >= nc_eps
        P_nvg[mask_c] = P_nvg[mask_c] + 25.0  # de Sitter boost
        
        for i in range(1, len(P_nvg)):
            if P_nvg[i] < P_nvg[i-1] + 1e-15:
                P_nvg[i] = P_nvg[i-1] + 1e-15
        return eps, P_nvg

# --- TOV INTEGRATION ---
def solve_tov(eps_arr, p_arr, p_c):
    if len(p_arr) < 2 or not np.all(np.isfinite(p_arr)) or not np.all(np.isfinite(eps_arr)):
        raise ValueError("EOS table is empty or non-finite")
    if np.any(np.diff(p_arr) <= 0.0):
        raise ValueError("EOS pressure table must be strictly increasing")
    p_surface = float(p_arr[0])
    p_max = float(p_arr[-1])
    if not np.isfinite(p_c) or not (p_surface < p_c <= p_max):
        raise ValueError(f"central pressure {p_c!r} outside EOS domain ({p_surface}, {p_max}]")
    eps_of_p = interp1d(p_arr, eps_arr, bounds_error=True)
    
    # Initial conditions at r = r0 (small radius to avoid singularity)
    r0 = 1.0e-3  # km
    e_c = float(eps_of_p(p_c))
    m0 = 4.0 * np.pi * r0**3 * e_c * conv_MeV_fm3_to_geo / 3.0
    
    def rhs(radius, state):
        mass, pressure = state
        if pressure <= p_surface:
            return [0.0, 0.0]
        energy = float(eps_of_p(pressure))
        denom = radius * (radius - 2.0 * mass)
        if denom <= 0.0:
            return [0.0, 0.0]
            
        dmdr = 4.0 * np.pi * radius**2 * energy * conv_MeV_fm3_to_geo
        dpdr = -(energy + pressure) * (mass + 4.0 * np.pi * radius**3 * pressure * conv_MeV_fm3_to_geo) / denom
        return [dmdr, dpdr]
        
    def stop_condition(radius, state):
        # The table starts above vacuum; stop at its lower validated boundary
        # instead of silently substituting an endpoint energy density.
        return state[1] - p_surface
    stop_condition.terminal = True
    stop_condition.direction = -1
    
    sol = solve_ivp(rhs, [r0, 80.0], [m0, p_c], events=stop_condition, max_step=0.25, rtol=1.0e-5, atol=1.0e-8)
    
    if sol.t_events[0].size > 0:
        radius = float(sol.t_events[0][0])
        mass = float(sol.y_events[0][0][0]) / M_sun_km
    else:
        radius = float(sol.t[-1])
        mass = float(sol.y[0, -1]) / M_sun_km
        
    return mass, radius

def generate_mr_curve(eps_arr, p_arr, p_min=0.005, p_max=600.0, n_points=100):
    p_grid = np.logspace(np.log10(p_min), np.log10(p_max), n_points)
    masses = []
    radii = []
    
    for pc in p_grid:
        m, r = solve_tov(eps_arr, p_arr, pc)
        if 0.1 < m < 4.5 and 5.0 < r < 60.0:
            masses.append(m)
            radii.append(r)
            
    return np.array(radii), np.array(masses)


def summarize_curves(curves):
    """Return maxima derived directly from raw ``(radius, mass)`` arrays."""

    summary = {}
    for name, (radii, masses) in curves.items():
        if len(masses) == 0:
            raise ValueError(f"empty TOV curve: {name}")
        index = int(np.argmax(masses))
        summary[name] = (float(masses[index]), float(radii[index]), index)
    return summary

# --- MAIN GENERATION ---
def main():
    print("Starting descriptive hyperon EOS calculations and figure generation...")
    
    # 1. Generate equations of state
    eps_nl3_n, P_nl3_n = get_nl3_eos("nucleon")
    eps_nl3_h, P_nl3_h = get_nl3_eos("hyperon")
    eps_nl3_nvg, P_nl3_nvg = get_nl3_eos("nvg")
    
    eps_sly_n, P_sly_n = get_sly_eos("nucleon")
    eps_sly_h, P_sly_h = get_sly_eos("hyperon")
    eps_sly_nvg, P_sly_nvg = get_sly_eos("nvg")
    
    # 2. Run TOV scans
    print("Running TOV solver for NL3 baseline...")
    r_nl3_n, m_nl3_n = generate_mr_curve(eps_nl3_n, P_nl3_n, p_max=350)
    r_nl3_h, m_nl3_h = generate_mr_curve(eps_nl3_h, P_nl3_h, p_max=320)
    r_nl3_nvg, m_nl3_nvg = generate_mr_curve(eps_nl3_nvg, P_nl3_nvg, p_max=500)
    
    print("Running TOV solver for SLy baseline...")
    r_sly_n, m_sly_n = generate_mr_curve(eps_sly_n, P_sly_n, p_max=300)
    r_sly_h, m_sly_h = generate_mr_curve(eps_sly_h, P_sly_h, p_max=280)
    r_sly_nvg, m_sly_nvg = generate_mr_curve(eps_sly_nvg, P_sly_nvg, p_max=450)
    
    # Keep the raw TOV curves.  Previous versions multiplied each mass curve
    # by an abstract target and shifted radii to make the figure match a
    # preferred narrative.  Such presentation transforms are not physical
    # outputs and would also hide that the two baselines have different raw
    # maxima.  Every label and annotation below is generated from these arrays.
    curves = {
        "NL3 nucleon": (r_nl3_n, m_nl3_n),
        "NL3 hyperon": (r_nl3_h, m_nl3_h),
        "NL3 NVG": (r_nl3_nvg, m_nl3_nvg),
        "SLy nucleon": (r_sly_n, m_sly_n),
        "SLy hyperon": (r_sly_h, m_sly_h),
        "SLy NVG": (r_sly_nvg, m_sly_nvg),
    }

    maxima = summarize_curves(curves)
    
    def get_r14(r, m):
        idx = np.argsort(m)
        return float(np.interp(1.4, m[idx], r[idx]))
        
    print(f"NL3 Nucleon R1.4: {get_r14(r_nl3_n, m_nl3_n):.3f} km; Mmax = {maxima['NL3 nucleon'][0]:.3f} M_sun")
    print(f"NL3 Hyperon R1.4: {get_r14(r_nl3_h, m_nl3_h):.3f} km; Mmax = {maxima['NL3 hyperon'][0]:.3f} M_sun")
    print(f"NL3 NVG R1.4:     {get_r14(r_nl3_nvg, m_nl3_nvg):.3f} km; Mmax = {maxima['NL3 NVG'][0]:.3f} M_sun")
    print(f"SLy Nucleon R1.4: {get_r14(r_sly_n, m_sly_n):.3f} km; Mmax = {maxima['SLy nucleon'][0]:.3f} M_sun")
    print(f"SLy Hyperon R1.4: {get_r14(r_sly_h, m_sly_h):.3f} km; Mmax = {maxima['SLy hyperon'][0]:.3f} M_sun")
    print(f"SLy NVG R1.4:     {get_r14(r_sly_nvg, m_sly_nvg):.3f} km; Mmax = {maxima['SLy NVG'][0]:.3f} M_sun")
    print("STATUS: descriptive raw TOV curves; no target rescaling or fitted maxima")
    
    # Ensure directories exist
    os.makedirs("article/figures", exist_ok=True)
    os.makedirs("verification", exist_ok=True)
    
    # Define colors
    c_nuc = "#3b6fe2"    # blue
    c_hyp = "#d9381e"    # red/orange
    c_nvg = "#087030"    # deep green
    
    # =========================================================================
    # FIGURE 1: Two-panel MR comparison (NL3 and SLy)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    
    # (a) NL3 Panel
    ax1.plot(r_nl3_n, m_nl3_n, label=f"Nucleon only ({maxima['NL3 nucleon'][0]:.2f} $M_\\odot$)", color=c_nuc, linestyle="--", linewidth=2.5)
    ax1.plot(r_nl3_h, m_nl3_h, label=f"N+$\\Lambda$ ({maxima['NL3 hyperon'][0]:.2f} $M_\\odot$)", color=c_hyp, linewidth=2.5)
    ax1.plot(r_nl3_nvg, m_nl3_nvg, label=f"NVG+$\\Lambda$ ({maxima['NL3 NVG'][0]:.2f} $M_\\odot$)", color=c_nvg, linewidth=3.0)
    
    # Annotations for NL3
    ax1.set_title("(a) NL3 — stiff EOS baseline", fontsize=12)
    ax1.set_xlabel("Radius (km)", fontsize=11)
    ax1.set_ylabel("Mass ($M_\\odot$)", fontsize=11)
    ax1.set_xlim(9.0, max(17.0, float(np.max(r_nl3_n)) + 0.5))
    ax1.set_ylim(0.1, max(3.3, 1.1 * maxima['NL3 NVG'][0]))
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # Show the raw hyperon-branch change
    ax1.annotate("hyperon", xy=(maxima['NL3 hyperon'][1], maxima['NL3 hyperon'][0]),
                 xytext=(maxima['NL3 hyperon'][1], maxima['NL3 hyperon'][0] - 0.12),
                 arrowprops=dict(arrowstyle="->", color=c_hyp, lw=1.5),
                 color=c_hyp, fontsize=10, ha='center')
    
    # Show NVG mass gain
    nvg_gain_nl3 = maxima['NL3 NVG'][0] - maxima['NL3 hyperon'][0]
    ax1.annotate(f"{nvg_gain_nl3:+.2f} $M_\\odot$\nNVG", xy=(maxima['NL3 NVG'][1], maxima['NL3 NVG'][0]),
                 xytext=(maxima['NL3 NVG'][1] + 0.2, maxima['NL3 NVG'][0] - 0.15),
                 arrowprops=dict(arrowstyle="<-", color=c_nvg, lw=1.5),
                 color=c_nvg, fontsize=10, ha='center')
    
    # Shaded bands for observables
    ax1.axhspan(1.97, 2.05, color='#f2dcdc', alpha=0.5) # PSR J0348+0432 / J1614
    ax1.text(10.6, 2.01, "J0348: 2.01 $M_\\odot$ (observed)", color="#d9381e", fontsize=9, va='center')
    
    # NICER J0740 ellipse
    circ_740 = plt.Circle((12.39, 2.08), 1.2, color='#cde2d3', fill=False, linestyle='--', linewidth=1.5)
    ax1.add_patch(circ_740)
    ax1.text(13.5, 2.2, "J0740\n(NICER)", color=c_nvg, fontsize=9)
    
    # NICER J0030 ellipse
    circ_0030 = plt.Circle((13.02, 1.44), 1.1, color='#aac8f2', fill=False, linestyle='--', linewidth=1.5)
    ax1.add_patch(circ_0030)
    ax1.text(13.5, 1.3, "J0030\n(NICER)", color=c_nuc, fontsize=9)
    
    # GW170817 constraint
    ax1.axvspan(10.5, 13.6, color='#f9f6ea', alpha=0.4, zorder=0)
    ax1.text(10.6, 1.7, "GW170817\n$R_{1.4}$", color="#a38b43", fontsize=9)
    
    ax1.legend(loc="lower right", fontsize=9.5)
    
    # (b) SLy Panel
    ax2.plot(r_sly_n, m_sly_n, label=f"Nucleon only ({maxima['SLy nucleon'][0]:.2f} $M_\\odot$)", color=c_nuc, linestyle="--", linewidth=2.5)
    ax2.plot(r_sly_h, m_sly_h, label=f"N+$\\Lambda$ ({maxima['SLy hyperon'][0]:.2f} $M_\\odot$)", color=c_hyp, linewidth=2.5)
    ax2.plot(r_sly_nvg, m_sly_nvg, label=f"NVG+$\\Lambda$ ({maxima['SLy NVG'][0]:.2f} $M_\\odot$)", color=c_nvg, linewidth=3.0)
    
    # Annotations for SLy
    ax2.set_title("(b) SLy — soft EOS baseline", fontsize=12)
    ax2.set_xlabel("Radius (km)", fontsize=11)
    ax2.set_xlim(9.0, 17.0)
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    # Show the raw hyperon-branch change
    sly_softening = maxima['SLy hyperon'][0] - maxima['SLy nucleon'][0]
    ax2.text(maxima['SLy hyperon'][1], maxima['SLy hyperon'][0] - 0.08,
             f"{sly_softening:+.2f} $M_\\odot$\nhyperon", color=c_hyp, fontsize=9, ha='center')
    
    # Show NVG mass gain
    nvg_gain_sly = maxima['SLy NVG'][0] - maxima['SLy hyperon'][0]
    ax2.annotate(f"{nvg_gain_sly:+.2f} $M_\\odot$\nNVG", xy=(maxima['SLy NVG'][1], maxima['SLy NVG'][0]),
                 xytext=(maxima['SLy NVG'][1] + 0.3, maxima['SLy NVG'][0] - 0.15),
                 arrowprops=dict(arrowstyle="<-", color=c_nvg, lw=1.5),
                 color=c_nvg, fontsize=10, ha='center')
    
    ax2.axhspan(1.97, 2.05, color='#f2dcdc', alpha=0.5)
    
    # NICER ellipses
    circ_740_2 = plt.Circle((12.39, 2.08), 1.2, color='#cde2d3', fill=False, linestyle='--', linewidth=1.5)
    ax2.add_patch(circ_740_2)
    circ_0030_2 = plt.Circle((13.02, 1.44), 1.1, color='#aac8f2', fill=False, linestyle='--', linewidth=1.5)
    ax2.add_patch(circ_0030_2)
    
    ax2.axvspan(10.5, 13.6, color='#f9f6ea', alpha=0.4, zorder=0)
    
    ax2.legend(loc="lower right", fontsize=9.5)
    
    plt.suptitle("Raw TOV mass-radius curves for two EOS baselines\n$\\beta$-equilibrium | NVG $W$-field", fontsize=13, y=0.98)
    plt.tight_layout()
    
    # Save Figure 1
    plt.savefig("article/figures/fig_hyperon_mr_baselines.png", dpi=300)
    plt.savefig("article/figures/fig_hyperon_mr_baselines.pdf")
    plt.savefig("verification/fig_hyperon_mr_baselines.png", dpi=300)
    print("Saved fig_hyperon_mr_baselines.png and .pdf")
    plt.close()

    # =========================================================================
    # FIGURE 2: EOS comparison (P vs eps) for NL3
    # =========================================================================
    plt.figure(figsize=(8.5, 6))
    plt.plot(eps_nl3_n, P_nl3_n, label="NL3 nucleon only", color=c_nuc, linestyle="--", linewidth=3.0)
    plt.plot(eps_nl3_h, P_nl3_h, label="NL3 + $\\Lambda$ hyperon branch", color=c_hyp, linewidth=3.0)
    plt.plot(eps_nl3_nvg, P_nl3_nvg, label="NVG + $\\Lambda$ branch", color=c_nvg, linewidth=3.0)
    
    # Causal limit P = eps
    plt.plot(eps_nl3_n, eps_nl3_n, label="$P = \\varepsilon$ (causal limit)", color="#a0a0a0", linestyle=":", linewidth=1.5)
    
    plt.xlim(0, 500)
    plt.ylim(-5, 200)
    
    # Annotations
    plt.axvline(235.0, color=c_hyp, linestyle=":", alpha=0.7)
    plt.text(238.0, 10, "$\\Lambda$ onset", color=c_hyp, rotation=90, va='bottom', fontsize=10)
    
    plt.axvline(330.0, color=c_nvg, linestyle=":", alpha=0.7)
    plt.text(333.0, 10, "$n_c = 2.05\\,n_0$", color=c_nvg, rotation=90, va='bottom', fontsize=10)
    
    plt.title("Equation of State: raw NL3 branch comparison\n$\\beta$-equilibrium | NVG $W$-field correction", fontsize=12)
    plt.xlabel("Energy density  $\\varepsilon$  (MeV/fm$^3$)", fontsize=11)
    plt.ylabel("Pressure  $P$  (MeV/fm$^3$)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10.5)
    plt.tight_layout()
    
    # Save Figure 2
    plt.savefig("article/figures/fig_hyperon_eos_compare.png", dpi=300)
    plt.savefig("article/figures/fig_hyperon_eos_compare.pdf")
    plt.savefig("verification/fig_hyperon_eos_compare.png", dpi=300)
    print("Saved fig_hyperon_eos_compare.png and .pdf")
    plt.close()

    # =========================================================================
    # FIGURE 3: Standalone NL3 MR relation
    # =========================================================================
    plt.figure(figsize=(9, 7.2))
    
    # Plot curves
    plt.plot(r_nl3_n, m_nl3_n, label=f"NL3 nucleon only ($M_{{\\max}} = {maxima['NL3 nucleon'][0]:.2f}\\,M_\\odot$)", color=c_nuc, linestyle="--", linewidth=3.0)
    plt.plot(r_nl3_h, m_nl3_h, label=f"NL3 + $\\Lambda$ hyperon ($M_{{\\max}} = {maxima['NL3 hyperon'][0]:.2f}\\,M_\\odot$)", color=c_hyp, linewidth=3.0)
    plt.plot(r_nl3_nvg, m_nl3_nvg, label=f"NVG + $\\Lambda$ ($M_{{\\max}} = {maxima['NL3 NVG'][0]:.2f}\\,M_\\odot$)", color=c_nvg, linewidth=3.5)
    
    # Shaded band for J0348
    plt.axhspan(1.97, 2.05, color='#f2dcdc', alpha=0.5, zorder=0)
    plt.text(10.6, 2.01, "PSR J0348+0432\n$2.01 \\pm 0.04\\,M_\\odot$", color="#d9381e", fontsize=10, va='center')
    
    # Shaded band for GW170817
    plt.axvspan(10.5, 13.6, color='#f9f6ea', alpha=0.4, zorder=0)
    plt.text(10.6, 1.5, "GW170817\n$R_{1.4} \\in [10.5, 13.6]\\,km$", color="#a38b43", fontsize=10)
    
    # NICER ellipses
    circ_740_3 = plt.Circle((12.39, 2.08), 1.2, color='#cde2d3', fill=False, linestyle='--', linewidth=1.8)
    plt.gca().add_patch(circ_740_3)
    plt.text(13.9, 2.08, "PSR J0740+6620 (NICER 2021)", color=c_nvg, fontsize=10, va='center')
    
    circ_0030_3 = plt.Circle((13.02, 1.44), 1.1, color='#aac8f2', fill=False, linestyle='--', linewidth=1.8)
    plt.gca().add_patch(circ_0030_3)
    plt.text(14.8, 1.33, "PSR J0030+0451 (NICER 2019)", color=c_nuc, fontsize=10, va='center')
    
    # Points and arrows for maximum masses
    # NL3 Nucleon Max
    idx_n_max = np.argmax(m_nl3_n)
    plt.scatter(r_nl3_n[idx_n_max], m_nl3_n[idx_n_max], color=c_nuc, s=60, edgecolors='white', zorder=5)
    plt.text(r_nl3_n[idx_n_max] + 0.1, m_nl3_n[idx_n_max] - 0.08,
             f"{maxima['NL3 nucleon'][0]:.2f} $M_\\odot$", color=c_nuc, fontsize=11, ha='right')
    
    # NL3 Hyperon Max
    idx_h_max = np.argmax(m_nl3_h)
    plt.scatter(r_nl3_h[idx_h_max], m_nl3_h[idx_h_max], color=c_hyp, s=60, edgecolors='white', zorder=5)
    plt.text(r_nl3_h[idx_h_max] - 0.1, m_nl3_h[idx_h_max] - 0.12,
             f"{maxima['NL3 hyperon'][0]:.2f} $M_\\odot$", color=c_hyp, fontsize=11, ha='right')
    
    # NVG Max
    idx_nvg_max = np.argmax(m_nl3_nvg)
    plt.scatter(r_nl3_nvg[idx_nvg_max], m_nl3_nvg[idx_nvg_max], color=c_nvg, s=60, edgecolors='white', zorder=5)
    plt.text(r_nl3_nvg[idx_nvg_max] - 0.1, m_nl3_nvg[idx_nvg_max] + 0.08,
             f"{maxima['NL3 NVG'][0]:.2f} $M_\\odot$", color=c_nvg, fontsize=11, ha='right')
    
    # Draw arrow representing the softening and the restoration
    plt.annotate("", xy=(r_nl3_h[idx_h_max], m_nl3_h[idx_h_max]), xytext=(r_nl3_n[idx_n_max], m_nl3_n[idx_n_max]),
                 arrowprops=dict(arrowstyle="->", color=c_hyp, lw=2))
    nl3_softening = maxima['NL3 hyperon'][0] - maxima['NL3 nucleon'][0]
    plt.text(maxima['NL3 hyperon'][1], maxima['NL3 hyperon'][0] - 0.16,
             f"$\\Delta M = {nl3_softening:+.2f}\\,M_\\odot$\n(hyperon)", color=c_hyp, fontsize=10, ha='center')
    
    plt.annotate("", xy=(r_nl3_nvg[idx_nvg_max], m_nl3_nvg[idx_nvg_max]), xytext=(r_nl3_h[idx_h_max], m_nl3_h[idx_h_max]),
                 arrowprops=dict(arrowstyle="->", color=c_nvg, lw=2))
    nl3_gain = maxima['NL3 NVG'][0] - maxima['NL3 hyperon'][0]
    plt.text(maxima['NL3 NVG'][1], maxima['NL3 NVG'][0] - 0.08,
             f"$\\Delta M = {nl3_gain:+.2f}\\,M_\\odot$\n(NVG)", color=c_nvg, fontsize=10, ha='center')
    
    plt.title("Neutron Star Mass-Radius: raw TOV outputs\n$\\beta$-equilibrium NL3 RMF + NVG $W$-field", fontsize=12)
    plt.xlabel("Radius  (km)", fontsize=12)
    plt.ylabel("Mass  ($M_\\odot$)", fontsize=12)
    plt.xlim(9.0, max(17.5, float(np.max(r_nl3_n)) + 0.5))
    plt.ylim(0.1, max(3.3, 1.1 * maxima['NL3 NVG'][0]))
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    
    # Save Figure 3
    plt.savefig("article/figures/fig_hyperon_mr_nl3.png", dpi=300)
    plt.savefig("article/figures/fig_hyperon_mr_nl3.pdf")
    plt.savefig("verification/fig_hyperon_mr_nl3.png", dpi=300)
    print("Saved fig_hyperon_mr_nl3.png and .pdf")
    plt.close()
    
    print("All figures successfully calculated and generated.")

if __name__ == "__main__":
    main()
