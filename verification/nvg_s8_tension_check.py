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

    # Honest S8: dynamical-DE growth only (PBH dark matter is CDM at Mpc scales)
    S8_nvg = S8_planck * sigma8_ratio_de
    tension_nvg = abs(S8_nvg - S8_lensing) / S8_lensing_err

    print(f"\nNVG Structure Growth Results (honest accounting):")
    print(f"  Growth shift from DE w(z):  {sigma8_ratio_de:.4f} ({(sigma8_ratio_de - 1.0)*100:+.1f}%)")
    print(f"  NVG S8 (DE growth only):    {S8_nvg:.4f}")
    print(f"  Tension with weak lensing:  {tension_nvg:.2f}σ "
          f"(vs {abs(S8_planck - S8_lensing)/S8_lensing_err:.2f}σ in LCDM)")
    print(f"\n  Core-mechanism capacity check:")
    print(f"    DM volume fraction inside de Sitter cores: {filling:.1e}")
    print(f"    Core radius: {r_core_cm:.1f} cm vs sigma8 scale 8 Mpc = 2.5e25 cm")
    print(f"    Suppression needed to reach lensing S8: {gap_needed:.3f} (7.8%)")
    print(f"    → the core mechanism is ~45 orders of magnitude short of 7.8%;")
    print(f"      PBH dark matter behaves as CDM on all cosmological scales.")
    print(f"\n  Status: S8 is an OPEN PROBLEM for NVG — its dynamical dark energy")
    print(f"  slightly increases structure growth ({(sigma8_ratio_de-1.0)*100:+.1f}%), moving S8 away from")
    print(f"  the lensing value; the proposed core suppression cannot contribute.")
    print(f"  A future lensing consensus at S8 ≈ 0.78 with LCDM growth excluded")
    print(f"  would falsify this sector as currently formulated.")
    is_ok = True  # honest accounting; no observational claim of resolution
    
    print("==========================================================================")

if __name__ == "__main__":
    run_s8_tension_check()
