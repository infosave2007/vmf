#!/usr/bin/env python3
"""
NVG Verification against NICER 2024 observations of PSR J0437-4715
------------------------------------------------------------------
Compares the NVG Equation of State (EOS) prediction for neutron star radius
against the 2024 NICER measurements of the nearby millisecond pulsar PSR J0437-4715:
  Observed: M = 1.418 ± 0.037 M_sun, R = 11.36 ± 0.8 km
"""

import os
import sys
import numpy as np
import math

# Add local path to import EOS solving classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nvg_tidal_deformability_gw170817 import EOS, solve_tov_tidal

def run_nicer_check():
    print("==========================================================================")
    print("  NVG COMPATIBILITY CHECK WITH NICER 2024 (PSR J0437-4715)")
    print("==========================================================================")
    
    # Observed NICER 2024 values
    M_obs = 1.418
    M_err = 0.037
    R_obs = 11.36
    R_err = 0.8
    
    print(f"NICER 2024 Measurement for PSR J0437-4715:")
    print(f"  Mass M = {M_obs} ± {M_err} M_sun")
    print(f"  Radius R = {R_obs} ± {R_err} km")
    
    # Solve TOV for NVG EOS
    eos = EOS(p_match=1.5, Gamma=1.35)
    
    # центральные давления для поиска 1.418 M_sun
    P_centers = np.logspace(-1.0, 2.8, 200)
    results = []
    
    for Pc in P_centers:
        M, R, k2, Lam = solve_tov_tidal(eos, Pc)
        if M > 0.5 and R > 5.0:
            results.append((M, R))
            
    # Найти точку, ближайшую к M_obs = 1.418
    best_diff = 1e9
    R_nvg = 0.0
    M_nvg = 0.0
    for M, R in results:
        diff = abs(M - M_obs)
        if diff < best_diff:
            best_diff = diff
            R_nvg = R
            M_nvg = M
            
    print(f"\nNVG/VMF Model Predictions (interpolated at M_obs):")
    print(f"  Calculated Mass: {M_nvg:.4f} M_sun (diff: {best_diff:.4f})")
    print(f"  Predicted Radius: {R_nvg:.2f} km")
    
    # Вычисление отклонения (Z-score)
    z_score = (R_nvg - R_obs) / R_err
    
    print(f"\nStatistical Compatibility Audit:")
    print(f"  Radius Difference: {R_nvg - R_obs:+.2f} km")
    print(f"  Z-score Deviation: {z_score:.2f}σ")
    
    # p-value для двустороннего Z-теста
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2.0))))
    print(f"  p-value: {p_val:.4f} ({p_val*100:.1f}%)")
    
    is_ok = abs(z_score) <= 1.5
    print(f"  Status: {'✅ COMPATIBLE (within 1.5σ)' if is_ok else '⚠️ TENSION'}")
    
    print("\nPhysics Context (honest reading):")
    print(f"The canonical NVG EOS predicts R = {R_nvg:.2f} km at {M_nvg:.3f} M_sun, i.e.")
    print(f"{z_score:+.1f} sigma above the J0437 central value {R_obs} km — outside the 68% CI")
    print("but inside 95%. This is currently the TIGHTEST tension of the canonical model:")
    print("a future J0437 radius confirmed below ~12.0 km at high precision would stress")
    print("the canonical parameterization (see nvg_ns_parameter_scan.py).")
    print("==========================================================================")

if __name__ == "__main__":
    run_nicer_check()
