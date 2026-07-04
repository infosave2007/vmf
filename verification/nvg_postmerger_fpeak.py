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
    
    # HONESTY NOTE: no post-merger signal has been observed yet (GW170817's
    # post-merger phase was below detector sensitivity), so there is NO target
    # to compare against. An earlier version validated against a "target" of
    # 2730 +/- 50 Hz that was simply this script's own output under the old
    # (falsified) EOS parameterization — a self-referential check.
    print(f"VMF Predicted Radius at 1.6 M_sun (R_1.6): {R_16:.2f} km")
    print(f"Numerical Relativity Fit                  : f_peak = (4.85 - 0.192 * R_1.6) kHz")
    print("-" * 74)
    print(f"FORWARD PREDICTION (no observation yet)  : f_peak = {f_peak_Hz:.0f} Hz ({f_peak_kHz:.3f} kHz)")
    print(f"Testable by                              : Einstein Telescope / Cosmic Explorer")
    print(f"                                           (post-merger spectra of BNS mergers)")

    is_ok = 1.5 < f_peak_kHz < 4.0  # sanity range of the NR fit's validity, not a data test
    print(f"Sanity check (1.5 < f_peak < 4.0 kHz)    : {'✅ OK' if is_ok else '❌ out of NR-fit range'}")
    print("==========================================================================")
    return is_ok

if __name__ == "__main__":
    calculate_postmerger_fpeak()
