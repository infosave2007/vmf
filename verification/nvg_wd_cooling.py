#!/usr/bin/env python3
"""
NVG Verification: White Dwarf Cooling Rate under VMF
---------------------------------------------------
Calculates the VMF correction to the white dwarf cooling rate due to minor
W-field melting in the degenerate electron core, and compares it to SDSS + Gaia limits.
"""

from __future__ import annotations
import math

def main():
    print("=" * 80)
    print("     NVG WHITE DWARF COOLING RATE & GAIA + SDSS CONSTRAINTS")
    print("=" * 80)

    # 1. Physical Parameters and Scaling
    # Standard white dwarf parameter values
    M_solar_ref = 0.6       # M_sun, typical average WD mass in catalogs
    delta_W_0 = 1.2e-6      # VMF melting fraction at typical 0.6 M_sun core density (10^6 g/cm3)
    alpha_VMF = 1.5         # Coupling coefficient of W-field to core thermal transport
    
    # Representative WD masses to check
    masses = [0.4, 0.6, 0.8, 1.0, 1.2]
    
    # SDSS + Gaia typical age determination uncertainty (approx 5%)
    observational_uncertainty = 0.05
    
    print(f"Base VMF W-field core melting (0.6 M_sun) : {delta_W_0:.2e}")
    print(f"Transport coupling coefficient (alpha)    : {alpha_VMF}")
    print(f"Gaia + SDSS age observational uncertainty : {observational_uncertainty:.1%}")
    print("-" * 80)
    print(f"  {'Mass (M_sun)':<13} | {'Core Density (g/cm³)':<20} | {'W-field melting':<17} | {'Age Shift Δt/t':<15} | {'Status':<10}")
    print("  " + "-" * 76)
    
    all_passed = True
    for M in masses:
        # Core density scales as M^2 due to mass-radius relation R ~ M^(-1/3)
        rho_c = 1e6 * (M / M_solar_ref) ** 2
        
        # W-field melting scales with core density
        delta_W = delta_W_0 * (M / M_solar_ref) ** 2
        
        # Effective mass reduction of electron
        m_e_eff_ratio = 1.0 - delta_W
        
        # Relative shift in cooling age (negative since increased conductivity speeds up cooling)
        age_shift = - alpha_VMF * delta_W
        
        # Check against observational error
        passed = abs(age_shift) < observational_uncertainty
        if not passed:
            all_passed = False
            
        status = "PASSED" if passed else "FAILED"
        print(f"  {M:<13.1f} | {rho_c:<20.2e} | {delta_W:<17.2e} | {age_shift:<15.2e} | {status:<10}")
        
    print("-" * 80)
    print(f"Constraints Verification Status: {'✅ ALL PASSED' if all_passed else '❌ FAILED'}")
    assert all_passed, "White dwarf cooling age shift exceeds Gaia + SDSS observational bounds!"
    print("=" * 80)

if __name__ == "__main__":
    main()
