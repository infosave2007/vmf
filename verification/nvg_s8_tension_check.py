#!/usr/bin/env python3
"""
NVG Cosmology: S8 Tension Relief Verification
---------------------------------------------
Calculates the predicted S8 parameter under the NVG cyclic cosmology model,
combining the dynamic dark energy growth rate and the VMF structure growth
suppression on small scales due to the de Sitter regular core in PBH dark matter.

Compares the result against:
  1. Planck standard Lambda-CDM: S8 = 0.832 ± 0.013
  2. Weak Lensing (DESI DR2 + DES Y6): S8 = 0.776 ± 0.017
Honest accounting: the NVG dynamical dark energy slightly INCREASES structure
growth (+1.4%), moving S8 away from the lensing value (~4 sigma), and a
capacity check shows the proposed de Sitter core mechanism cannot contribute
at Mpc scales (DM core volume fraction ~1e-47). S8 is an open problem for
NVG and a potential falsifier of this sector.
"""

import os
import sys
import numpy as np

# Add local path to import derive_w0_wa
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nvg_dark_energy_w0wa import derive_w0_wa

def run_s8_tension_check():
    print("==========================================================================")
    print("  NVG COSMOLOGY: S8 TENSION GROWTH SUPPRESSION AUDIT")
    print("==========================================================================")
    
    # 1. QCD Anchor & VMF parameters
    M_Omega_0 = 859.0
    
    # 2. Observational parameters
    S8_planck = 0.832
    S8_planck_err = 0.013
    sigma8_planck = 0.811   # Planck 2018 LCDM
    S8_lensing = 0.776      # DESI DR2 + DES Y6 / DES Y3 consensus
    S8_lensing_err = 0.017
    
    print("Observational benchmarks:")
    print(f"  Planck Lambda-CDM: S8 = {S8_planck} ± {S8_planck_err}")
    print(f"  Weak Lensing consensus (DESI DR2 + DES Y6): {S8_lensing} ± {S8_lensing_err}")
    print(f"  Initial standard tension: {(S8_planck - S8_lensing) / S8_lensing_err:.2f}σ")
    
    # Cosmology parameters
    Omega_m = 0.315
    Omega_DE = 0.685
    
    # Dynamically derive w0 and wa from the VMF cyclic cosmology equations:
    w_0, w_a = derive_w0_wa(M_Omega_0)
    
    # 3. Integrate growth factor f(a) = d ln D / d ln a ≈ Omega_m(a)^0.55
    N_steps = 2000
    a_arr = np.linspace(0.01, 1.0, N_steps)
    da = a_arr[1] - a_arr[0]
    
    growth_ratio_integral = 0.0
    for a in a_arr:
        # standard Lambda-CDM
        Omega_m_a_lcdm = Omega_m * a**(-3.0) / (Omega_m * a**(-3.0) + Omega_DE)
        f_lcdm = Omega_m_a_lcdm ** 0.55
        
        # NVG: DE density ratio
        rho_DE_ratio = a**(-3.0 * (1.0 + w_0 + w_a)) * np.exp(-3.0 * w_a * (1.0 - a))
        Omega_DE_a = Omega_DE * rho_DE_ratio
        Omega_m_a_nvg = Omega_m * a**(-3.0) / (Omega_m * a**(-3.0) + Omega_DE_a)
        f_nvg = Omega_m_a_nvg ** 0.55
        
        growth_ratio_integral += (f_nvg - f_lcdm) * da / a
        
    # Growth ratio from DE evolution only
    sigma8_ratio_de = np.exp(growth_ratio_integral)
    
    # 4. Capacity check of the proposed core mechanism (computed, not asserted).
    # The de Sitter cores of PBH dark matter occupy a volume fraction
    # rho_DM / rho_c of space; any core-induced modification of clustering is
    # bounded by scales ~ r_0 (centimeters), while sigma8 is defined at 8 Mpc.
    rho_c_gcm3 = 1.264e17            # bounce density, g/cm^3
    rho_dm_gcm3 = 2.24e-30           # mean DM density today, g/cm^3
    filling = rho_dm_gcm3 / rho_c_gcm3
    M_pbh_g = 8.64e-14 * 1.989e33    # peak PBH mass
    r_core_cm = (3.0 * M_pbh_g / (4.0 * np.pi * rho_c_gcm3)) ** (1.0 / 3.0)
    gap_needed = 0.078               # suppression required to reach the lensing S8

    # 4b. MASS-MELTING DARK MATTER AND THE ANCHORING FRAME.
    # The w0-wa sector (derive_w0_wa) transfers energy INTO matter:
    # Q = +beta H rho_m, i.e. DM masses GROW with time and the comoving
    # matter density scales as rho_m a^3 ~ a^beta after the melting is active.
    # The observable consequence depends on where the evolution is anchored:
    #
    #   (A) anchored TODAY (Omega_m0 fixed, deficit pushed into the past):
    #       the growth source is weaker at early times and sigma8 drops.
    #       This frame puts a large matter deficit into the CMB era and is
    #       NOT observable — kept below only as a diagnostic.
    #   (B) anchored at RECOMBINATION (the physically correct frame: same
    #       omega_m and same primordial amplitude as Planck, melting acts
    #       only afterwards): late-time matter is HIGHER than the a^-3
    #       continuation, growth is ENHANCED and Omega_m0 rises — S8 goes UP.
    #
    # A frame-A computation in an earlier revision suggested S8 ~ 0.75
    # ("relief"); that result was an artifact of the anchoring and is
    # RETRACTED here. Frame B is what the data see.
    beta_melt = (3.0 * 1.0**2) / (2.0 * 1.0**2 * 0.16**2) * 0.002048  # = 0.12, as in derive_w0_wa

    def growth_D_anchored(beta, a_on, w0, wa, om_early=Omega_m, a_i=0.005, n=8000):
        """Growth factor with CMB-side anchoring: comoving matter density equals
        the LCDM value before a_on, then grows as (a/a_on)^beta. Returns
        (D_today, Omega_m_today). beta=0 reproduces LCDM/CPL."""
        import numpy as _np
        a = _np.linspace(a_i, 1.0, n)
        boost = _np.where(a > a_on, (a / a_on) ** beta, 1.0)
        om_a = om_early * boost                    # comoving density rho_m a^3
        om_today = float(om_a[-1])
        ode_today = 1.0 - om_today                 # flat: DE drained into matter
        rho_m = om_a * a ** (-3.0)
        rho_de = ode_today * a ** (-3.0 * (1.0 + w0 + wa)) * _np.exp(-3.0 * wa * (1.0 - a))
        E2 = rho_m + rho_de
        dlnE2 = _np.gradient(_np.log(E2), a)
        D = _np.zeros(n); Dp = _np.zeros(n)
        D[0], Dp[0] = a[0], 1.0
        for i in range(n - 1):
            h = a[i + 1] - a[i]
            Dpp = -(3.0 / a[i] + 0.5 * dlnE2[i]) * Dp[i] + 1.5 * rho_m[i] / (E2[i] * a[i] ** 2) * D[i]
            Dp[i + 1] = Dp[i] + h * Dpp
            D[i + 1] = D[i] + h * Dp[i]
        return float(D[-1]), om_today

    D_lcdm, _ = growth_D_anchored(0.0, 1.0, -1.0, 0.0)
    print(f"\nMass-melting DM in the CMB-anchored frame (the observable one):")
    print(f"  Same early universe as Planck; melting active for a > a_on with")
    print(f"  beta = {beta_melt:.3f} (fixed by the DESI w0-wa fit — no new parameter).")
    print(f"  {'a_on':>6} {'z_on':>6} {'Om_m0':>7} {'D/D_L':>7} {'sigma8':>7} {'S8':>6} {'tension':>8}")
    S8_w0wa_fit = None
    for a_on in (0.40, 0.50, 0.70):
        D_nvg, om0 = growth_D_anchored(beta_melt, a_on, w_0, w_a)
        sig8 = sigma8_planck * D_nvg / D_lcdm
        S8_b = sig8 * (om0 / 0.3) ** 0.5
        t = abs(S8_b - S8_lensing) / S8_lensing_err
        flag = "  <- w0-wa fit range" if a_on == 0.40 else ""
        print(f"  {a_on:6.2f} {1/a_on-1:6.2f} {om0:7.3f} {D_nvg/D_lcdm:7.4f} {sig8:7.4f} {S8_b:6.3f} {t:7.1f}σ{flag}")
        if a_on == 0.40:
            S8_w0wa_fit = (S8_b, t, om0)

    S8_nvg = S8_planck * sigma8_ratio_de   # DE-background effect only (no melting rescue)
    tension_nvg = abs(S8_nvg - S8_lensing) / S8_lensing_err

    print(f"\n  VERDICT (frame-honest):")
    print(f"  1. The earlier 'relief to 1.6 sigma' was a today-anchored artifact — retracted.")
    print(f"  2. In the CMB-anchored frame the melting mechanism RAISES S8: with the")
    print(f"     w0-wa-fit configuration (a_on ~ 0.4) S8 = {S8_w0wa_fit[0]:.3f} ({S8_w0wa_fit[1]:.1f} sigma from")
    print(f"     lensing) and Omega_m0 = {S8_w0wa_fit[2]:.3f} (vs BAO/SNe ~ 0.30-0.33).")
    print(f"  3. Weak lensing therefore ACTIVELY CONSTRAINS the melting sector:")
    print(f"     beta * ln(1/a_on) must be << the w0-wa-fit value — the NVG dark-energy")
    print(f"     mechanism and S8 are in direct conflict through the model's own sector.")
    print(f"  4. S8 status: OPEN PROBLEM, sharpened into a falsifier — the honest NVG")
    print(f"     value without melting is S8 = {S8_nvg:.3f} ({tension_nvg:.1f} sigma, worse than LCDM's 3.3).")
    is_ok = True  # honest accounting; no observational claim of resolution
    
    print("==========================================================================")

if __name__ == "__main__":
    run_s8_tension_check()
