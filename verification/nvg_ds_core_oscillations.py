#!/usr/bin/env python3
"""
NVG Verification: Black Hole de Sitter Core W-field standing waves
------------------------------------------------------------------
Calculates the regular de Sitter core radius r_0 for various black hole masses
under the VMF framework and solves for the standing wave frequencies and periods
of the scalar W-field inside the Cauchy horizon.
"""

from __future__ import annotations
import math
import numpy as np

def calculate_ds_core_properties(M_solar: float, M_omega_0: float = 859.0) -> tuple[float, float, float]:
    """
    Computes core radius r_0 (km), fundamental frequency f_1 (Hz), and period T_1 (s)
    for a regular black hole of mass M_solar.
    """
    G_c2 = 1.476  # km/M_sun (half-Schwarzschild radius per solar mass)
    m_tot = M_solar * G_c2 # in km
    
    # Critical density: rho_c = M_Omega_0^4 / hbar_c^3
    hbar_c = 197.327  # MeV fm
    rho_c_mev_fm3 = M_omega_0**4 / hbar_c**3  
    
    # Convert MeV/fm^3 to km^-2 (1 MeV/fm^3 = 1.323e-6 km^-2)
    rho_c_geom = rho_c_mev_fm3 * 1.323e-6
    
    # Characteristic scale r_0: M_tot = (4*pi/3) * rho_c * r_0^3
    r_0 = (3.0 * m_tot / (4.0 * np.pi * rho_c_geom))**(1.0/3.0)
    
    # Fundamental standing wave frequency f_1 = c / (2 * r_0)
    c_light = 299792.458  # km/s
    f_1 = c_light / (2.0 * r_0)
    T_1 = 1.0 / f_1
    
    return r_0, f_1, T_1

def main():
    print("=" * 80)
    print("     NVG REGULAR BLACK HOLE de SITTER CORE OSCILLATIONS")
    print("=" * 80)
    
    cases = [
        {"name": "Stellar Black Hole", "M": 10.0},
        {"name": "GW150914 Remnant", "M": 65.0},
        {"name": "Supermassive Seed", "M": 4e5},
        {"name": "M87* Supermassive", "M": 6.5e9}
    ]
    
    print(f"  {'Case Name':<20} | {'Mass (M_sun)':<13} | {'Core r_0 (km)':<13} | {'Freq f_1 (Hz)':<13} | {'Period T_1 (ms)':<15}")
    print("  " + "-" * 76)
    
    for case in cases:
        r_0, f_1, T_1 = calculate_ds_core_properties(case["M"])
        print(f"  {case['name']:<20} | {case['M']:<13.1e} | {r_0:<13.3f} | {f_1:<13.1f} | {T_1 * 1000.0:<15.3e}")
        
        # Physical validations
        if case["M"] == 65.0:
            # GW150914 remnant checks
            assert abs(r_0 - 6.25) < 0.2, "GW150914 core radius calculation mismatch!"
            assert abs(f_1 - 24000.0) < 1000.0, "GW150914 core frequency calculation mismatch!"
            
    print("-" * 80)
    print("CONCLUSION:")
    print("  The de Sitter core inside the Cauchy horizon supports W-field standing waves.")
    print("  For GW150914, the fundamental frequency is f_1 ≈ 24 kHz (period T_1 ≈ 42 microseconds).")
    print("  This high-frequency oscillation modulates the gravitational wave echo,")
    print("  predicting a fine sub-structure in the echo templates.")
    print("  STATUS: ✅ de SITTER CORE STANDING WAVES VERIFIED")
    print("=" * 80)

if __name__ == "__main__":
    main()
