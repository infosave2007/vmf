#!/usr/bin/env python3
"""
NVG Publication Figure: Speed of Sound c_s^2(n_B) with Bayesian Contours
------------------------------------------------------------------------
Generates a publication-quality plot of the squared speed of sound c_s^2
as a function of baryon density n_B/n_0, comparing the VMF EOS prediction
against 90% CI Bayesian contours from joint NICER+LIGO inference.

Bayesian bands are digitized from:
  - Legred et al. (2021) Phys. Rev. D 104, 063003 (Fig. 5)
  - Miller et al. (2021) ApJ Lett. 918, L28

Output: fig_speed_of_sound_bayesian.png
"""

from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Add local path to import EOS solving classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nvg_full_ns_eos import UnifiedEOS, n_0


def get_bayesian_contours():
    """
    Digitized 90% CI for c_s^2(n_B) from Legred et al. (2021), Fig. 5.
    Joint NICER (PSR J0030+0451 + PSR J0740+6620) + LIGO (GW170817) posterior.
    Values: (n_B/n_0, median, lower_5%, upper_95%)
    """
    # Density points (n_B / n_0)
    n_points = np.array([
        0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8,
        3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0
    ])

    # Median c_s^2 from joint posterior
    median = np.array([
        0.04, 0.09, 0.14, 0.19, 0.28, 0.36, 0.40, 0.42, 0.42, 0.40,
        0.38, 0.36, 0.35, 0.34, 0.34, 0.34, 0.33, 0.33, 0.33
    ])

    # Lower 5% quantile
    lower = np.array([
        0.01, 0.03, 0.06, 0.09, 0.13, 0.16, 0.18, 0.18, 0.17, 0.15,
        0.13, 0.11, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10
    ])

    # Upper 95% quantile
    upper = np.array([
        0.10, 0.18, 0.27, 0.38, 0.55, 0.68, 0.73, 0.74, 0.72, 0.68,
        0.64, 0.58, 0.54, 0.50, 0.48, 0.46, 0.44, 0.42, 0.41
    ])

    return n_points, median, lower, upper


def main():
    print("=" * 80)
    print("     NVG PUBLICATION FIGURE: c_s^2(n_B) WITH BAYESIAN CONTOURS")
    print("=" * 80)

    # Phase transition parameters (same as nvg_full_ns_eos.py)
    n_trans = 2.0
    delta_eps = 350.0

    print(f"Loading Unified VMF EOS (Onset: {n_trans} n_0, Latent Heat: {delta_eps} MeV/fm³)...")
    eos = UnifiedEOS(n_trans, delta_eps)

    # Calculate density grid
    n_grid = np.logspace(-4, 1.5, 2000) * n_0
    eps_arr = eos.eps_arr
    p_arr = eos.p_arr

    # Compute cs^2 = dP/deps numerically
    deps = np.diff(eps_arr)
    dp = np.diff(p_arr)

    cs2 = np.zeros_like(deps)
    valid = deps > 0.0
    cs2[valid] = dp[valid] / deps[valid]

    # Baryon density at midpoints (in units of n_0)
    n_mid_n0 = ((n_grid[:-1] + n_grid[1:]) / 2.0) / n_0

    # Clip negative cs2 values
    cs2 = np.clip(cs2, 0.0, 1.2)

    # Get Bayesian contours
    n_bayes, med_bayes, lo_bayes, hi_bayes = get_bayesian_contours()

    # Identify key features of the VMF curve
    max_cs2 = np.max(cs2)
    max_idx = np.argmax(cs2)
    max_density = n_mid_n0[max_idx]

    high_density_mask = n_mid_n0 > 5.0
    asymptotic_cs2 = np.mean(cs2[high_density_mask]) if np.any(high_density_mask) else 0.0

    print(f"VMF Maximum c_s^2 = {max_cs2:.4f} at {max_density:.2f} n_0")
    print(f"VMF Asymptotic c_s^2 = {asymptotic_cs2:.4f}")
    print(f"Conformal limit 1/3 = {1.0/3.0:.4f}")

    # ── Publication-Quality Figure ──────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Background gradient for visual depth
    ax.set_facecolor('#fafafa')

    # 1. Bayesian 90% CI band (NICER+LIGO)
    ax.fill_between(n_bayes, lo_bayes, hi_bayes,
                    color='#4ECDC4', alpha=0.22,
                    label=r'NICER+LIGO 90% CI (Legred+ 2021)')
    # Bayesian median
    ax.plot(n_bayes, med_bayes,
            color='#4ECDC4', linewidth=1.5, linestyle='--', alpha=0.8,
            label=r'NICER+LIGO median')

    # 2. VMF EOS curve (main result)
    mask_plot = n_mid_n0 < 8.0
    ax.plot(n_mid_n0[mask_plot], cs2[mask_plot],
            color='#FF6B6B', linewidth=2.8, zorder=5,
            label=r'VMF EOS ($M_\Omega = 859$ MeV)')

    # 3. Reference lines
    ax.axhline(1.0 / 3.0, color='#555555', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.text(7.5, 0.345, r'$c_s^2 = 1/3$', fontsize=10, color='#555555',
            ha='right', va='bottom', style='italic')

    ax.axhline(1.0, color='#999999', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(7.5, 1.02, r'Causality', fontsize=9, color='#999999',
            ha='right', va='bottom')

    # 4. Phase transition marker
    ax.axvline(n_trans, color='#FF9F43', linestyle='-.', linewidth=1.2, alpha=0.7)
    ax.text(n_trans + 0.1, 0.95, r'QGP transition',
            fontsize=9, color='#FF9F43', rotation=90, va='top')

    # 5. pQCD asymptotic arrow
    ax.annotate(r'pQCD: $c_s^2 \to 1/3$',
                xy=(6.8, 0.333), xytext=(5.5, 0.50),
                fontsize=9, color='#666666',
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.0),
                ha='center')

    # Axes
    ax.set_xlabel(r'Baryon density $n_B / n_0$', fontsize=14)
    ax.set_ylabel(r'Speed of sound $c_s^2 / c^2$', fontsize=14)
    ax.set_title(r'Speed of Sound in Dense Matter: VMF vs NICER+LIGO',
                 fontsize=14, fontweight='bold', pad=12)

    ax.set_xlim(0, 8.0)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#cccccc')
    ax.grid(True, linestyle='--', alpha=0.25)

    # Add text box with key results
    textstr = (r'$c_{s,\max}^2 = %.2f$ at $%.1f\,n_0$' % (max_cs2, max_density) + '\n'
               + r'Asymptotic: $c_s^2 \to %.3f$' % asymptotic_cs2)
    props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9)
    ax.text(0.97, 0.70, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    plot_path = os.path.join(os.path.dirname(__file__), "fig_speed_of_sound_bayesian.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {plot_path}")

    # Assertions
    assert max_cs2 < 1.0, "EOS violates causality!"
    assert abs(asymptotic_cs2 - 1.0 / 3.0) < 0.05, "Quark phase doesn't approach conformal limit!"

    print("Speed of sound Bayesian comparison PASSED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
