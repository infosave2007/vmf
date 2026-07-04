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
Maps the S8 tension onto a VMF core growth suppression. NOTE: the 7.8%
suppression coefficient is FITTED to the lensing S8, not derived (see the
honesty note in the code) — this is a calibration, not a resolution.
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
    
    # 4. Small-scale growth suppression attributed to the VMF regular core in PBH DM.
    # HONESTY NOTE: the coefficient 0.078 below is NOT derived anywhere in this repo —
    # it is exactly the fractional gap between the Planck input S8 = 0.832 and the
    # weak-lensing target 0.776, i.e. a FITTED parameter dressed as a mechanism.
    # Until the suppression is computed from the core scale (r_c, halo profile),
    # this row is a calibration, not a resolution of the S8 tension.
    suppression_vmf_core = 1.0 - 0.078 * (859.0 / M_Omega_0)**2
    
    # Net growth suppression ratio: D_NVG(a=1) / D_LCDM(a=1)
    sigma8_ratio_net = sigma8_ratio_de * suppression_vmf_core
    
    # Calculate NVG S8 = S8_planck * growth_suppression_net
    S8_nvg = S8_planck * sigma8_ratio_net
    
    # Remaining tension
    tension_nvg = abs(S8_nvg - S8_lensing) / S8_lensing_err
    
    print(f"\nNVG Structure Growth Results:")
    print(f"  Growth shift from DE w(z):  {sigma8_ratio_de:.4f} (+{(sigma8_ratio_de - 1.0)*100:.1f}%)")
    print(f"  VMF core small-scale suppression: {suppression_vmf_core:.4f} (-{(1.0 - suppression_vmf_core)*100:.1f}%)")
    print(f"  Net growth suppression ratio σ8(NVG)/σ8(ΛCDM): {sigma8_ratio_net:.4f} (-{(1.0 - sigma8_ratio_net)*100:.1f}%)")
    print(f"  NVG predicted S8: {S8_nvg:.4f}")
    print(f"  Remaining tension with Weak Lensing: {tension_nvg:.2f}σ")
    
    is_ok = tension_nvg <= 1.0
    print(f"  Status: ⚠️ CALIBRATED, not resolved — the 7.8% suppression is fitted to the")
    print(f"          lensing S8 (see honesty note above); deriving it from the core scale")
    print(f"          is an open task.")
    
    print("\nPhysics Context:")
    print("Under NVG cyclic cosmology, the small-scale structure growth is suppressed by the")
    print("de Sitter regular core of primordial black hole (PBH) dark matter. This core")
    print("acts as a physical regularization scale, preventing infinite density singular clustering.")
    print("The net S8 ≈ 0.776 matches weak lensing BY CONSTRUCTION: the 7.8% core")
    print("suppression was fitted to that value. Deriving it from the core scale (r_c,")
    print("halo profile) would turn this calibration into a genuine prediction.")
    print("==========================================================================")

if __name__ == "__main__":
    run_s8_tension_check()
