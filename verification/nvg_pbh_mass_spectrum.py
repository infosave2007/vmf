#!/usr/bin/env python3
"""
NVG Verification: Primordial Black Hole Mass Spectrum
------------------------------------------------------
Calculates the discrete PBH mass spectrum M_N = 0.38 * 4^N M_sun derived from
the VMF cyclic cosmology, verifying the mass hierarchy at different cycles N.
"""

import numpy as np

def get_pbh_mass(n_cycle: int) -> float:
    # PBH mass spectrum: M_N = 0.38 * 4^N M_sun
    return 0.38 * (4.0 ** n_cycle)

def main():
    print("==========================================================================")
    print("  NVG COSMOLOGY: PRIMORDIAL BLACK HOLE DISCRETE MASS SPECTRUM")
    print("==========================================================================")
    
    # Selected representative cycles
    # N = -21: Peak asteroid mass dark matter
    # N = 10: Early SMBH seeds at z > 6 (JWST)
    # N = 0: Stellar mass PBH
    cycles = [-28, -25, -21, -15, 0, 10]
    
    print(f"{'Cycle N':<10} | {'PBH Mass (M_sun)':<20} | {'Interpretation':<35}")
    print("-" * 74)
    
    for N in cycles:
        mass = get_pbh_mass(N)
        if N == -21:
            desc = "Asteroid-mass DM Peak (Unconstrained)"
        elif N == 10:
            desc = "Early SMBH seeds at z > 6 (JWST)"
        elif N == 0:
            desc = "Stellar-mass PBH (LIGO band)"
        elif N < -21:
            desc = "Sub-asteroid PBH (Hawking radiation)"
        else:
            desc = "Intermediate PBH seed"
            
        print(f"N = {N:<7d} | {mass:<20.4e} | {desc:<35}")
        
    print("-" * 74)
    # Assertions
    m_21 = get_pbh_mass(-21)
    m_10 = get_pbh_mass(10)
    
    assert abs(m_21 - 8.64e-14) < 1e-15, "Asteroid peak mass mismatch!"
    assert abs(m_10 - 3.98e5) < 1e3, "JWST seed mass mismatch!"
    
    print("Status: ✅ PBH mass spectrum hierarchy verified successfully.")
    print("==========================================================================")

if __name__ == "__main__":
    main()
