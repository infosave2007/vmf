#!/usr/bin/env python3
"""Generate publication-quality figures for the W-field vacuum condensate article.

Creates two PDF and PNG figures in article/figures/:
1. fig_w_field_melting.{pdf,png}
   Plot of the normalized condensate amplitude W(n_B)/W_0 and the generalized
   momentum time component g_0(n_B) as a function of density n_B/n_0.
2. fig_w_field_sec.{pdf,png}
   Plot of energy density, pressure, and the Strong Energy Condition (SEC)
   combination rho_W + 3P_W of the W-field.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "article" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def calculate_w_field_dynamics() -> dict:
    n_0 = 0.16              # fm^-3, normal nuclear matter density
    hbar_c = 197.327        # MeV*fm, conversion factor
    
    # Nucleon and Meson masses / couplings
    M_N = 939.0             # MeV
    m_pi = 139.57           # MeV
    f_pi = 92.4             # MeV
    g_omega = 10.12         # Effective omega-nucleon vector coupling
    m_omega = 782.6         # MeV
    
    # QCD Anchors
    M_omega_0 = 859.0       # MeV (W_0)
    lambda_param = 1.05
    mu_theta = m_omega      # Vacuum phase frequency
    mu_squared = mu_theta**2 + lambda_param * (M_omega_0**2)
    q = (g_omega * M_omega_0) / (2.0 * f_pi)
    
    # Generate high resolution density grid
    density_factors = np.linspace(0.0, 4.0, 400)
    
    g_0_list = []
    w_list = []
    rho_list = []
    p_list = []
    sec_list = []
    
    for f in density_factors:
        n_B_fm3 = f * n_0
        
        # A_0 in MeV (mean field omega-meson potential)
        A_0 = (g_omega / (m_omega**2)) * n_B_fm3 * (hbar_c**3)
        
        # Gauge-invariant gradient: g_0 = mu_theta - q * A_0
        g_0 = mu_theta - q * A_0
        
        # W_0^2 = max(0, mu^2 - g_0^2) / lambda
        w0_sq = max(0.0, mu_squared - g_0**2) / lambda_param
        w0 = math.sqrt(w0_sq)
        
        # V(W) = 1/4 * lambda * (W^2 - W_0^2)^2
        v_w_shifted = 0.25 * lambda_param * (w0**2 - M_omega_0**2)**2
        
        # Energy density and pressure in MeV^4
        rho_w_mev4 = 0.5 * (w0**2) * (g_0**2) + v_w_shifted
        p_w_mev4 = -0.5 * (w0**2) * (g_0**2) - v_w_shifted
        sec_mev4 = rho_w_mev4 + 3.0 * p_w_mev4
        
        # Convert to MeV/fm^3
        rho_w_fm3 = rho_w_mev4 / (hbar_c**3)
        p_w_fm3 = p_w_mev4 / (hbar_c**3)
        sec_fm3 = sec_mev4 / (hbar_c**3)
        
        g_0_list.append(g_0)
        w_list.append(w0)
        rho_list.append(rho_w_fm3)
        p_list.append(p_w_fm3)
        sec_list.append(sec_fm3)
        
    return {
        "factors": density_factors,
        "g_0": np.array(g_0_list),
        "w": np.array(w_list),
        "w_0": M_omega_0,
        "rho": np.array(rho_list),
        "p": np.array(p_list),
        "sec": np.array(sec_list),
    }


def plot_melting(data: dict) -> None:
    factors = data["factors"]
    w_norm = data["w"] / data["w_0"]
    g_0 = data["g_0"]
    
    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    
    # Plot W(n_B)/W_0 on primary y-axis (left)
    color = "#2c3e50"
    ax1.set_xlabel(r"Baryon Density $n_B / n_0$", fontsize=11)
    ax1.set_ylabel(r"Normalized Amplitude $\mathcal{W}(n_B) / \mathcal{W}_0$", color=color, fontsize=11)
    line1 = ax1.plot(factors, w_norm, color="#3498db", lw=2.2, label=r"$\mathcal{W}(n_B)/\mathcal{W}_0$")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Plot g_0 on secondary y-axis (right)
    ax2 = ax1.twinx()
    color = "#7f8c8d"
    ax2.set_ylabel(r"Generalized Momentum $g_0$ (MeV)", color="#c0392b", fontsize=11)
    line2 = ax2.plot(factors, g_0, color="#c0392b", lw=1.8, linestyle="--", label=r"$g_0(n_B)$")
    ax2.tick_params(axis='y', labelcolor="#c0392b")
    
    # Mark the melting point (W becomes 0)
    # Find the index where W drops to 0
    zero_idx = np.where(w_norm == 0)[0]
    if len(zero_idx) > 0:
        melting_density_exact = 2.05
        ax1.axvline(melting_density_exact, color="#16a085", linestyle=":", lw=1.5)
        ax1.text(melting_density_exact + 0.05, 0.5, f"Melting Point\n$n_B \\approx {melting_density_exact:.2f}\\,n_0$", 
                 color="#16a085", fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Add legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    
    plt.title("Vacuum Condensate Amplitude Melting and Momentum Evolution", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_w_field_melting.{ext}", dpi=300)
    plt.close(fig)


def plot_sec(data: dict) -> None:
    factors = data["factors"]
    rho = data["rho"]
    p = data["p"]
    sec = data["sec"]
    
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    
    ax.plot(factors, rho, color="#1abc9c", lw=2.0, label=r"Energy Density $\rho_{\mathcal{W}}$")
    ax.plot(factors, p, color="#e67e22", lw=2.0, linestyle="--", label=r"Pressure $P_{\mathcal{W}}$")
    ax.plot(factors, sec, color="#9b59b6", lw=2.2, linestyle="-.", label=r"SEC $\rho_{\mathcal{W}} + 3P_{\mathcal{W}}$")
    
    # Add horizontal line indicating the vacuum potential limit -2V_0
    v_0_fm3 = (0.25 * 1.05 * (data["w_0"]**4)) / (197.327**3) # in MeV/fm^3
    ax.axhline(-2 * v_0_fm3, color="#c0392b", linestyle=":", lw=1.2)
    ax.text(0.1, -1.9 * v_0_fm3, r"Symmetric Limit $-2V_0$", color="#c0392b", fontsize=9)
    
    # Mark the melting point boundary
    zero_idx = np.where(data["w"] == 0)[0]
    if len(zero_idx) > 0:
        melting_density_exact = 2.05
        ax.axvline(melting_density_exact, color="#7f8c8d", linestyle=":", lw=1.2)
        ax.text(melting_density_exact - 0.7, 10000, "Condensate Phase", color="#7f8c8d", fontsize=9)
        ax.text(melting_density_exact + 0.1, 10000, "Melted Phase\n(de Sitter)", color="#7f8c8d", fontsize=9)

    ax.set_xlabel(r"Baryon Density $n_B / n_0$", fontsize=11)
    ax.set_ylabel(r"Energy-Momentum Components ($\text{MeV/fm}^3$)", fontsize=11)
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="none")
    
    plt.title("W-Field Energy Density, Pressure, and SEC Violation", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_w_field_sec.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    print("Computing W-field dynamics...")
    data = calculate_w_field_dynamics()
    
    print("Plotting figures...")
    plot_melting(data)
    plot_sec(data)
    
    print(f"Done! Figures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
