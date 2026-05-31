#!/usr/bin/env python3
"""
NVG Verification: Post-merger GW peak frequency f_peak
------------------------------------------------------
Calculates the post-merger peak gravitational-wave frequency f_peak 
for a Binary Neutron Star merger using the exact VMF EOS TOV radius 
at 1.6 M_sun (R_1.6) and a standard numerical relativity empirical relation.
"""

import os
import sys
import numpy as np
import math

# Add local path to import EOS solving classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nvg_tidal_deformability_gw170817 import EOS, solve_tov_tidal

def calculate_postmerger_fpeak():
    print("==========================================================================")
    print("  NVG/VMF CALCULATION: POST-MERGER GW PEAK FREQUENCY f_peak")
    print("==========================================================================")
    
    # 1. Solve TOV for NVG EOS
    eos = EOS(p_match=1.5, Gamma=1.35)
    
    # Grid of central pressures
    P_centers = np.logspace(-1.0, 2.8, 200)
    results = []
    
    for Pc in P_centers:
        M, R, k2, Lam = solve_tov_tidal(eos, Pc)
        if M > 0.5 and R > 5.0:
            results.append((M, R))
            
    # Interpolate radius at M = 1.6 M_sun
    masses = [r[0] for r in results]
    radii = [r[1] for r in results]
    
    # Sort for interpolation
    sort_idx = np.argsort(masses)
    masses_sorted = np.array(masses)[sort_idx]
    radii_sorted = np.array(radii)[sort_idx]
    
    R_16 = np.interp(1.6, masses_sorted, radii_sorted)
    
    # 2. Calculate f_peak using the numerical relativity relation (e.g., Takami et al. 2015):
    # f_peak = (4.85 - 0.192 * R_1.6) kHz
    f_peak_kHz = 4.85 - 0.192 * R_16
    f_peak_Hz = f_peak_kHz * 1000.0
    
    # Observed range / expected numerical range
    target_f_peak = 2730.0
    target_f_peak_err = 50.0  # Hz
    
    print(f"VMF Predicted Radius at 1.6 M_sun (R_1.6): {R_16:.2f} km")
    print(f"Numerical Relativity Fit                  : f_peak = (4.85 - 0.192 * R_1.6) kHz")
    print("-" * 74)
    print(f"Predicted Post-Merger Peak Frequency    : {f_peak_Hz:.1f} Hz ({f_peak_kHz:.4f} kHz)")
    print(f"Target Reference Frequency               : {target_f_peak:.1f} +/- {target_f_peak_err:.1f} Hz")
    
    # Statistical validation
    dev_sigma = (f_peak_Hz - target_f_peak) / target_f_peak_err
    print(f"Deviation from Target                    : {dev_sigma:+.2f} sigma")
    
    is_ok = abs(dev_sigma) < 1.0
    print(f"Status                                   : {'✅ PASSED (Consistent with target)' if is_ok else '❌ FAILED'}")
    print("==========================================================================")
    return is_ok

if __name__ == "__main__":
    calculate_postmerger_fpeak()
