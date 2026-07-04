#!/usr/bin/env python3
"""
NVG Verification: Gravitational Redshift z_surf from NS surface
---------------------------------------------------------------
Calculates the surface gravitational redshift z_surf for a 1.4 M_sun neutron
star from the canonical VMF EOS TOV radius (computed, not hardcoded).

HONESTY NOTE: an earlier version compared z(R = 12.0 km hardcoded) against a
"target" 0.235 that was itself derived from the same 12.0 km — a circular
self-check. Now z_surf is derived from the canonical EOS radius and presented
as a forward prediction for STROBE-X/eXTP, with no target to "pass".
"""

import os
import sys
import numpy as np
import math

# Add local path to import EOS solving classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nvg_tidal_deformability_gw170817 import EOS, solve_tov_tidal

def calculate_ns_redshift():
    print("==========================================================================")
    print("  NVG/VMF CALCULATION: NEUTRON STAR SURFACE GRAVITATIONAL REDSHIFT")
    print("==========================================================================")
    
    # Constants
    G_over_c2 = 1.4766  # km / M_sun
    M_ref = 1.4         # M_sun
    
    # 1. Solve TOV for NVG EOS (crust-softened case)
    eos = EOS(p_match=1.5, Gamma=1.35)
    P_centers = np.logspace(-1.0, 2.8, 200)
    results = []
    
    for Pc in P_centers:
        M, R, k2, Lam = solve_tov_tidal(eos, Pc)
        if M > 0.5 and R > 5.0:
            results.append((M, R))
            
    # Interpolate radius at M = 1.4 M_sun
    masses = [r[0] for r in results]
    radii = [r[1] for r in results]
    
    sort_idx = np.argsort(masses)
    masses_sorted = np.array(masses)[sort_idx]
    radii_sorted = np.array(radii)[sort_idx]
    
    R_14 = np.interp(1.4, masses_sorted, radii_sorted)

    # z_surf = (1 - 2GM/Rc^2)^(-0.5) - 1, from the COMPUTED canonical radius
    C_14 = G_over_c2 * M_ref / R_14
    z_14 = (1.0 - 2.0 * C_14)**(-0.5) - 1.0

    print(f"Canonical VMF EOS radius at 1.4 M_sun    : R_1.4 = {R_14:.2f} km (computed)")
    print(f"Compactness GM/(Rc^2)                    : {C_14:.4f}")
    print(f"Predicted surface redshift               : z_surf = {z_14:.4f}")
    print("-" * 74)
    print("\nPhysics Context:")
    print("Direct measurements of z_surf are currently absent (claims like z ≈ 0.35")
    print("from Cottam et al. 2002 for EXO 0748-676 were not confirmed). This is a")
    print("FORWARD prediction with no observed target yet — directly testable by")
    print("STROBE-X/eXTP to <1% precision.")

    is_ok = 0.1 < z_14 < 0.4  # sanity range for a 1.4 M_sun NS, not an observational test
    print(f"\nSanity check (0.1 < z < 0.4)             : {'✅ OK' if is_ok else '❌ out of range'}")
    print("==========================================================================")
    return is_ok

if __name__ == "__main__":
    calculate_ns_redshift()
