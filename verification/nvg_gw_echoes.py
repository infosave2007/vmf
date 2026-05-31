#!/usr/bin/env python3
"""
NVG Gravitational Wave Echoes: Inner Core Simulation
------------------------------------------------------
This script calculates the characteristic scales of the regular de Sitter
core for stellar-mass and supermassive black holes within the NVG model.
It estimates the post-merger gravitational wave echo delay time (Delta t)
analytically using coordinate regularization near horizons.
"""

import numpy as np
import math

# Constants
G_cgs = 6.674e-8
c_cgs = 2.998e10
M_sun = 1.989e33

# NVG QCD Anchor
M_Omega_0 = 859.0       # MeV
hbar_c = 197.327        # MeV fm
eps_max = M_Omega_0**4 / hbar_c**3  # MeV/fm^3
MeV_fm3_to_gcm3 = 1.7827e12
rho_c = eps_max * MeV_fm3_to_gcm3   # ~1.26e17 g/cm^3

def get_bh_parameters(M_bh_solar):
    """Calculate the Hayward metric parameters for a given BH mass."""
    M_cgs = M_bh_solar * M_sun
    r_0_cgs = (3.0 * M_cgs / (4.0 * math.pi * rho_c))**(1/3.0)
    R_g_cgs = 2.0 * G_cgs * M_cgs / c_cgs**2
    return M_cgs, r_0_cgs, R_g_cgs

def solve_roots(R_g, r_0):
    """Find the roots of r^3 - R_g * r^2 + r_0^3 = 0."""
    coeffs = [1.0, -R_g, 0.0, r_0**3]
    roots = np.roots(coeffs)
    roots = sorted(roots, key=lambda x: x.real)
    return roots[0].real, roots[1].real, roots[2].real

def calculate_echo_delay(M_bh_solar):
    """
    Calculates the Schwarzschild-Hayward post-merger echo delay analytically
    using the physical proper distance cutoff l_cutoff = 1.0 cm (the bounce scale).
    """
    M_cgs, r_0, R_g = get_bh_parameters(M_bh_solar)
    r1, r2, r3 = solve_roots(R_g, r_0) # r1 negative, r2 = r_minus, r3 = r_plus
    
    # Partial fraction coefficients for 1/f(r) = (r^3 + r_0^3) / P(r)
    A1 = (r1**3 + r_0**3) / ((r1 - r2) * (r1 - r3))
    A2 = (r2**3 + r_0**3) / ((r2 - r1) * (r2 - r3))
    A3 = (r3**3 + r_0**3) / ((r3 - r1) * (r3 - r2))
    
    # Physical proper distance cutoff near the event horizon is equal to the
    # cosmological bounce scale (l_cutoff = 1.0 cm).
    # This corresponds to a coordinate cutoff of dr3 = l_cutoff^2 / (4 * A3)
    l_cutoff = 1.0  # cm
    dr3 = l_cutoff**2 / (4.0 * abs(A3))
    
    # Exterior round-trip travel time: Delta t = 2 * (F_ext - F_r3_plus) / c
    # evaluated analytically to prevent double-precision loss.
    R_ph = 1.5 * R_g
    term_const = (1.5*R_g - r3) + A1 * math.log((1.5*R_g - r1)/(r3 - r1)) + A2 * math.log((1.5*R_g - r2)/(r3 - r2)) + A3 * math.log(1.5*R_g - r3)
    diff = term_const - A3 * math.log(dr3)
    
    delta_t_echo = 2.0 * diff / c_cgs
    
    return {
        'mass': M_bh_solar,
        'r_0_km': r_0 / 1e5,
        'r_minus_km': r2 / 1e5,
        'r_plus_km': r3 / 1e5,
        'R_g_km': R_g / 1e5,
        'delta_t_echo_s': delta_t_echo,
        'echo_freq_Hz': 1.0 / delta_t_echo if delta_t_echo > 0 else 0
    }

def main():
    print("=====================================================================")
    print(" NVG GRAVITATIONAL WAVE ECHO PREDICTIONS")
    print("=====================================================================")
    print(f"Critical Density rho_c: {rho_c:.2e} g/cm^3")
    print("-" * 75)
    print(f"{'Mass (M_sun)':>12} | {'r_0 (km)':>10} | {'r_- (km)':>12} | {'r_+ (km)':>10} | {'Echo Delay (s)':>15}")
    print("-" * 75)
    
    masses = [3.0, 10.0, 65.0, 4e6] # NS collapse, Stellar BH, GW150914, Sgr A*
    
    for m in masses:
        res = calculate_echo_delay(m)
        if res:
            print(f"{res['mass']:12.1f} | {res['r_0_km']:10.2e} | {res['r_minus_km']:12.2e} | {res['r_plus_km']:10.2f} | {res['delta_t_echo_s']:15.4f}")
            
    print("-" * 75)
    print("ANALYSIS:")
    print("- r_0 is the characteristic scale of the regular de Sitter core.")
    print("- r_- is the inner Cauchy horizon (boundary of the core).")
    print("- The 'Echo Delay' is calculated analytically using the physical proper")
    print("  distance cutoff l_cutoff = 1.0 cm (the VMF bounce scale) at the horizon.")
    print("  For GW150914 (65 M_sun), Schwarzschild echoes are predicted to arrive")
    print("  every 0.0445 seconds.")
    print("\nOBSERVATIONAL GOAL:")
    print("Detecting repeating echoes with this specific frequency spacing")
    print("in LIGO/Virgo post-merger data would confirm the existence of")
    print("a regular NVG core instead of a classical singularity.")
    print("=====================================================================")

if __name__ == "__main__":
    main()
