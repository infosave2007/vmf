#!/usr/bin/env python3
"""
NVG W-field direct-detection forward calculation.
======================================================================
Evaluates three declared EFT interaction scenarios for a QCD-vacuum
condensate W-field.  The benchmark ratios are sensitivity diagnostics only:
without event data, detector response, and nuisance likelihood they cannot
establish a WIMP exclusion or a measured null result.

Physics:
  The W-field (vacuum condensate order parameter) couples to ALL quarks
  simultaneously through the QCD trace anomaly — it IS the medium that
  generates 91% of the nucleon mass. Asking "can you detect W in a
  scattering experiment?" is like asking "can you detect the ocean by
  throwing a fish into it?"

  Three coupling channels are computed as a sensitivity study:

  1. DIRECT QCD (trace anomaly): g_eff ~ λ_v · f_N · m_N / m_W²
     → σ_SI ~ 10⁻²⁴ cm² (declared EFT scenario)
     This is the "natural" coupling: W generates the nucleon mass.

  2. HIGGS PORTAL (t-channel h exchange): λ_WH · v · f_N · m_N / (m_H² · m_W)
     → σ_SI ~ 10⁻³⁸ cm² (declared EFT scenario)

  3. EFFECTIVE (screened): a loop-suppressed comparison scenario.

  The cross sections are model-dependent EFT estimates.  They are not a
  detector likelihood and do not establish a measured exclusion.

  NVG dark matter = PBH hierarchy 4^N + frozen θ-topological defects.
  Any null-signal statement remains conditional on this interaction model.

Output: fig_dm_no_wimp.png
"""

from __future__ import annotations
import os
import math
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nvg_magnetar_closure import M_OMEGA_0


# ── Physical Constants ──────────────────────────────────────────────
M_Omega_0 = float(M_OMEGA_0)  # MeV — canonical QCD vacuum anchor
m_N = 939.0              # MeV — nucleon mass
hbar_c = 197.327         # MeV·fm
MeV_to_GeV = 1e-3
fm_to_cm = 1e-13


# ── Three coupling channels ────────────────────────────────────────
def compute_all_channels():
    """
    Compute σ_SI for all three W-nucleon coupling channels.
    The channels are explicit EFT scenarios.  Their values are useful for
    sensitivity plotting but require a detector response and likelihood before
    any exclusion can be inferred.
    """
    lambda_v = 1.02
    m_W = math.sqrt(2.0 * lambda_v) * M_Omega_0  # MeV
    m_W_GeV = m_W * MeV_to_GeV

    v_EW = 246e3        # MeV — Higgs VEV
    m_H = 125.1e3       # MeV — Higgs mass
    f_N = 0.30          # nuclear matrix element (total)
    sigma_piN = 44.0    # MeV
    mu_r = m_W * m_N / (m_W + m_N)  # MeV

    results = {
        'lambda_v': lambda_v,
        'm_W_MeV': m_W,
        'm_W_GeV': m_W_GeV,
        'mu_r_MeV': mu_r,
        'f_N': f_N,
    }

    # ── Channel 1: DIRECT QCD (trace anomaly) ──────────────────────
    # W generates nucleon mass → coupling is O(1) at QCD scale
    # g_eff = λ_v · f_N · m_N / m_W²
    g_direct = lambda_v * f_N * m_N / m_W**2  # MeV⁻¹
    sigma_direct = g_direct**2 * mu_r**2 / math.pi  # MeV⁻²
    sigma_direct_cm2 = sigma_direct * (hbar_c * fm_to_cm)**2

    results['channel_1'] = {
        'name': 'Direct QCD (trace anomaly)',
        'g_eff': g_direct,
        'sigma_cm2': sigma_direct_cm2,
    }

    # ── Channel 2: HIGGS PORTAL (t-channel h exchange) ─────────────
    # λ_WH ~ λ_v · (M_Ω/v_EW)² → suppressed by (QCD/EW)²
    lambda_WH = lambda_v * (M_Omega_0 / v_EW)**2
    amplitude_h = lambda_WH * v_EW * f_N * m_N / (m_H**2 * m_W)  # MeV⁻¹
    sigma_higgs = amplitude_h**2 * mu_r**2 / math.pi
    sigma_higgs_cm2 = sigma_higgs * (hbar_c * fm_to_cm)**2

    results['channel_2'] = {
        'name': 'Higgs portal (t-channel)',
        'lambda_WH': lambda_WH,
        'g_eff': amplitude_h,
        'sigma_cm2': sigma_higgs_cm2,
    }

    # ── Channel 3: MAXIMALLY SCREENED ──────────────────────────────
    # Even assuming loop suppression (α_s/4π)² and form factor
    alpha_s = 0.3
    screening = (alpha_s / (4 * math.pi))**2
    sigma_screened_cm2 = sigma_direct_cm2 * screening

    results['channel_3'] = {
        'name': 'Maximally screened (loop)',
        'screening': screening,
        'sigma_cm2': sigma_screened_cm2,
    }

    results["evidence_status"] = "MODEL_SENSITIVITY_ONLY"
    results["observed_likelihood"] = None
    results["limitation"] = (
        "The repository has no event-level detector data or likelihood for the "
        "declared direct-detection benchmark curves."
    )
    results["canonical_producer"] = "verification/nvg_magnetar_closure.py:M_OMEGA_0"
    return results


def experimental_limits():
    """Declared benchmark curves digitized from named experiments.

    These arrays are display inputs, not event-level likelihood data.  The
    provenance is therefore kept explicit in the returned record.
    """
    lz_mass = [5, 6, 7, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000]
    lz_sigma = [3e-44, 8e-45, 3e-45, 1.5e-45, 8e-46, 4e-46, 2.5e-46,
                1.5e-46, 1.2e-46, 1.5e-46, 2.5e-46, 5e-46, 1e-45]

    xenon_mass = [6, 8, 10, 15, 20, 30, 50, 100, 200, 500, 1000]
    xenon_sigma = [2e-43, 2e-44, 5e-45, 2e-45, 1.5e-45, 1e-45, 9e-46,
                   1e-45, 2e-45, 4e-45, 8e-45]

    ds_mass = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
    ds_sigma = [1e-39, 3e-40, 5e-41, 1e-41, 5e-42, 1e-42, 5e-43, 2e-43]

    pandax_mass = [5, 7, 10, 15, 20, 30, 50, 100, 200, 500, 1000]
    pandax_sigma = [5e-44, 5e-45, 2e-45, 8e-46, 5e-46, 3e-46, 2.5e-46,
                    3e-46, 5e-46, 9e-46, 2e-45]

    nu_mass = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    nu_sigma = [1e-42, 3e-44, 5e-45, 5e-46, 2e-47, 3e-48, 1e-48,
                5e-49, 3e-49, 2e-49, 1e-49]

    return {
        'LZ': (lz_mass, lz_sigma),
        'XENON1T': (xenon_mass, xenon_sigma),
        'DarkSide-50': (ds_mass, ds_sigma),
        'PandaX-4T': (pandax_mass, pandax_sigma),
        'Neutrino Floor': (nu_mass, nu_sigma),
    }


def experimental_limit_provenance() -> str:
    return "declared benchmark curves; event likelihood unavailable"


def main():
    print("=" * 80)
    print("  NVG: W-FIELD DIRECT-DETECTION INTERACTION SENSITIVITY")
    print("=" * 80)

    results = compute_all_channels()
    limits = experimental_limits()

    # ── 1. W-field parameters ─────────────────────────────────────────
    print(f"\nW-field condensate parameters:")
    print(f"  Self-coupling λ_v = {results['lambda_v']:.2f}")
    print(f"  Condensate excitation mass: m_W = {results['m_W_MeV']:.0f} MeV"
          f" = {results['m_W_GeV']:.3f} GeV")
    print(f"  Nuclear matrix element: f_N = {results['f_N']:.2f}")
    print(f"  Reduced mass: μ_r = {results['mu_r_MeV']:.1f} MeV")

    # ── 2. Three coupling channels ────────────────────────────────────
    # Get DarkSide-50 limit at m_W
    ds_limit = np.interp(np.log10(results['m_W_GeV']),
                         np.log10(limits['DarkSide-50'][0]),
                         np.log10(limits['DarkSide-50'][1]))
    ds_limit = 10**ds_limit

    print(f"\n{'─'*80}")
    print(f"Three coupling channels (model sensitivity scenarios):")
    print(f"  DarkSide-50 limit at m = {results['m_W_GeV']:.2f} GeV: "
          f"σ_limit = {ds_limit:.1e} cm²\n")

    for i in range(1, 4):
        ch = results[f'channel_{i}']
        ratio = ch['sigma_cm2'] / ds_limit
        print(f"  Channel {i}: {ch['name']}")
        print(f"    σ_SI = {ch['sigma_cm2']:.2e} cm²")
        print(f"    Ratio to benchmark curve: {ratio:.1e}× (not a likelihood result)")
        print()

    # ── 3. Physical interpretation ────────────────────────────────────
    print(f"{'─'*80}")
    print(f"PHYSICAL INTERPRETATION:")
    print(f"  The W-field is modelled here as a QCD-vacuum condensate.")
    print(f"  This script evaluates interaction scenarios, not an experimental fit:")
    print(f"    1. In this model it is parameterized as a vacuum medium, not a WIMP")
    print(f"    2. The displayed couplings are broad EFT scenarios, not measured rates")
    print(f"    3. the displayed σ values use explicit declared inputs")
    print(f"    4. detector acceptance/backgrounds are not represented")
    print(f"")
    print(f"  NVG dark matter consists of:")
    print(f"    • Primordial Black Holes (4^N mass hierarchy, #13-14)")
    print(f"    • Frozen topological defects in θ-phase at QCD transition")
    print(f"  Their WIMP-signature compatibility is not tested without an event likelihood.")

    # ── 4. Conditional test boundary ─────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"CONDITIONAL TEST:")
    print(f"  A detector comparison would require event data, response and nuisance model.")
    print(f"  Current status: MODEL_SENSITIVITY_ONLY (no observational likelihood).")

    # ══════════════════════════════════════════════════════════════════
    # PUBLICATION FIGURE
    # ══════════════════════════════════════════════════════════════════
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 12,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
    })

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_facecolor('#fafafa')

    # Plot experimental limits
    colors_exp = {
        'LZ': '#2196F3', 'PandaX-4T': '#FF9800',
        'XENON1T': '#9E9E9E', 'DarkSide-50': '#4CAF50',
        'Neutrino Floor': '#E0E0E0',
    }
    linestyles = {
        'LZ': '-', 'PandaX-4T': '-', 'XENON1T': '--',
        'DarkSide-50': '-', 'Neutrino Floor': ':',
    }

    for name, (masses, sigmas) in limits.items():
        if name == 'Neutrino Floor':
            ax.fill_between(masses, sigmas, 1e-55, alpha=0.12,
                           color=colors_exp[name], zorder=1)
            ax.plot(masses, sigmas, color='#BDBDBD', linewidth=1.5,
                   linestyle=':', label=r'Neutrino floor ($\nu$-coherent)',
                   zorder=1)
        else:
            ax.plot(masses, sigmas, color=colors_exp[name], linewidth=2.0,
                   linestyle=linestyles[name], label=name, zorder=3)
            ax.fill_between(masses, sigmas, 1e-20, alpha=0.04,
                           color=colors_exp[name])

    # Plot the three W-field coupling channels
    m_W = results['m_W_GeV']

    # Channel 1: Direct QCD
    sigma_1 = results['channel_1']['sigma_cm2']
    ax.plot(m_W, sigma_1, 'X', color='#D32F2F', markersize=16,
            markeredgecolor='black', markeredgewidth=1.5, zorder=10,
            label=r'W-field: direct QCD (EFT scenario)')

    # Channel 2: Higgs portal
    sigma_2 = results['channel_2']['sigma_cm2']
    ax.plot(m_W, sigma_2, 'D', color='#FF6F00', markersize=13,
            markeredgecolor='black', markeredgewidth=1.5, zorder=10,
            label=r'W-field: Higgs portal (EFT scenario)')

    # Channel 3: Maximally screened
    sigma_3 = results['channel_3']['sigma_cm2']
    ax.plot(m_W, sigma_3, 's', color='#FFC107', markersize=11,
            markeredgecolor='black', markeredgewidth=1.2, zorder=10,
            label=r'W-field: max screened (EFT scenario)')

    # Draw arrow connecting the three channels
    ax.annotate('', xy=(m_W, sigma_2), xytext=(m_W, sigma_1),
                arrowprops=dict(arrowstyle='->', color='#D32F2F',
                               lw=2, ls='--'))
    ax.text(m_W * 1.15, math.sqrt(sigma_1 * sigma_2),
            r'$10^{14}\times$' + '\nsuppression',
            fontsize=9, color='#D32F2F', ha='left', va='center')

    # Draw the region above the declared benchmark curve; this is not an
    # event-level exclusion because no detector likelihood is available.
    ax.axhspan(ds_limit, 1e-20, xmin=0, xmax=0.15,
               alpha=0.15, color='#D32F2F', zorder=0)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.5, 1500)
    ax.set_ylim(1e-50, 1e-22)
    ax.set_xlabel(r'Dark Matter Mass $m_{\rm DM}$ [GeV]', fontsize=14)
    ax.set_ylabel(r'SI Cross Section $\sigma_{\rm SI}$ [cm$^2$]', fontsize=14)
    ax.set_title(r'NVG: W-field direct-detection model sensitivity',
                 fontsize=14, fontweight='bold', pad=12)

    ax.legend(fontsize=9.5, loc='upper right', framealpha=0.9, edgecolor='#ccc')
    ax.grid(True, which='major', linestyle='--', alpha=0.2)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1)

    # Conclusion box
    textstr = ('Model sensitivity scenarios\n'
               r'$\mathcal{W}$-field = vacuum-condensate EFT\n'
               'No event likelihood in repository')
    props = dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                 edgecolor='#2E7D32', alpha=0.95)
    ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props, fontweight='bold')

    # Arrow from scenario markers to the benchmark-comparison note
    ax.annotate('EFT scenarios\nabove benchmark\ncurve',
                xy=(m_W, sigma_2), xytext=(5, 1e-30),
                fontsize=10, fontweight='bold', color='#D32F2F',
                ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                         edgecolor='#D32F2F', alpha=0.9))

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "fig_dm_no_wimp.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {plot_path}")

    print("\n" + "=" * 80)
    print("RUNTIME SUMMARY: direct-detection interaction scenarios")
    print(f"  Channel 1 (QCD): σ = {results['channel_1']['sigma_cm2']:.1e} cm²"
          f" — benchmark ratio {results['channel_1']['sigma_cm2']/ds_limit:.0e}×")
    print(f"  Channel 2 (Higgs): σ = {results['channel_2']['sigma_cm2']:.1e} cm²"
          f" — benchmark ratio {results['channel_2']['sigma_cm2']/ds_limit:.0e}×")
    print(f"  Channel 3 (screened): σ = {results['channel_3']['sigma_cm2']:.1e} cm²"
          f" — benchmark ratio {results['channel_3']['sigma_cm2']/ds_limit:.0e}×")
    print()
    print("Evidence status: MODEL_SENSITIVITY_ONLY")
    print("No measured exclusion or null result is claimed by this forward calculation.")
    print("=" * 80)
    return {
        "results": results,
        "limits": limits,
        "limits_provenance": experimental_limit_provenance(),
        "darkside_limit_cm2": ds_limit,
    }


if __name__ == "__main__":
    main()
