#!/usr/bin/env python3
"""
NVG Verification: Gravitational Wave Echo Delay Prediction
------------------------------------------------------------
Calculates the post-merger GW echo delay time (Delta t) for a 65 M_sun remnant,
showing that the regular de Sitter core predicted by NVG yields Delta t = 0.00512 s.
"""

import math
import numpy as np

# Constants
G_cgs = 6.674e-8
c_cgs = 2.998e10
M_sun = 1.989e33

# NVG QCD Anchor
M_Omega_0 = 859.0 # MeV
hbar_c = 197.327 # MeV fm
eps_max = M_Omega_0**4 / hbar_c**3  # MeV/fm^3
MeV_fm3_to_gcm3 = 1.7827e12
rho_c = eps_max * MeV_fm3_to_gcm3   # ~1.26e17 g/cm^3

def get_bh_parameters(M_bh_solar):
    """Calculates the physical parameters of the black hole core."""
    M_cgs = M_bh_solar * M_sun
    r_0_cgs = (3.0 * M_cgs / (4.0 * math.pi * rho_c))**(1/3.0)
    R_g_cgs = 2.0 * G_cgs * M_cgs / c_cgs**2
    return r_0_cgs, R_g_cgs

def calculate_kerr_echo_delay(M_bh_solar, a_spin):
    """
    Calculates the round-trip echo travel time from the regular core boundary
    to the photon sphere analytically, using coordinate regularization near horizons.
    """
    r_0, R_g = get_bh_parameters(M_bh_solar)
    M_geom = R_g / 2.0
    a = a_spin * M_geom
    
    # Kerr horizons in geometric units
    r_plus = M_geom + math.sqrt(M_geom**2 - a**2)
    r_minus = M_geom - math.sqrt(M_geom**2 - a**2)
    
    R_ph = 1.5 * R_g
    
    # In VMF, the regular de Sitter core boundary r_0 acts as the physical cutoff scale
    delta = r_0  # cm
    
    # Evaluate tortoise coordinate travel time analytically:
    # dt_echo = 2 * (r_star(R_ph) - r_star(r_plus + delta)) / c
    # The analytical difference below prevents double-precision loss (since delta << r_plus)
    term1 = R_ph - r_plus
    term2 = (2.0 * M_geom * r_plus / (r_plus - r_minus)) * math.log((R_ph - r_plus) / delta)
    term3 = (2.0 * M_geom * r_minus / (r_plus - r_minus)) * math.log((R_ph - r_minus) / (r_plus - r_minus))
    
    dt = 2.0 * (term1 + term2 - term3) / c_cgs
    return dt

def main():
    print("==========================================================================")
    print("  NVG COSMOLOGY: POST-MERGER GRAVITATIONAL WAVE ECHO PREDICTION")
    print("==========================================================================")
    M_remnant = 65.0 # M_sun
    
    # Note: Remnant spin a_spin = 0.67 is an observational input from the LIGO
    # GW150914 event, not a free parameter of the NVG model.
    a_spin = 0.67
    
    print(f"QCD Anchor M_Omega_0             : {M_Omega_0} MeV")
    print(f"Critical Density (rho_c)         : {rho_c:.4e} g/cm^3")
    print(f"Remnant Mass                     : {M_remnant} M_sun")
    print(f"Remnant Spin (LIGO Observational): a_spin = {a_spin}")
    
    dt_spin = calculate_kerr_echo_delay(M_remnant, a_spin)
    
    print("-" * 74)
    print(f"Derived Core scale r_0           : {get_bh_parameters(M_remnant)[0]/1e5:.4f} km")
    print(f"Schwarzschild Radius R_g         : {get_bh_parameters(M_remnant)[1]/1e5:.4f} km")
    print(f"Predicted echo delay: {dt_spin:.5f} s")
    print("Note: testable with LIGO O5 for remnants ~65 M_sun")
    print("==========================================================================")

if __name__ == "__main__":
    main()
