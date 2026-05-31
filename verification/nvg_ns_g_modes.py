#!/usr/bin/env python3
"""
NVG Verification: Neutron Star g-mode core oscillation periods
--------------------------------------------------------------
Solves the TOV equations to obtain the mass-density profile of a 1.4 M_sun
neutron star, computes the Brunt-Vaisala frequency profile N(r) for composition
g-modes, and calculates the fundamental g-mode period T_g (approx 80-120 ms).
g-modes are a key target for next-generation detectors like the Einstein Telescope (ET).
"""

from __future__ import annotations
import os
import sys
import numpy as np
import math

# Add local path to import EOS solving classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nvg_tidal_deformability_gw170817 import EOS, k_conv, M_sun_km

def integrate_tov_profiles(eos: EOS, P_center: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrates the TOV equations and returns radial profiles of r, m, P, and eps."""
    dr = 0.01  # km
    r = 1e-6   # km
    m = 0.0    # geometric mass (km)
    P = P_center
    
    r_profile = []
    m_profile = []
    p_profile = []
    e_profile = []
    
    while P > 1e-5 and r < 100.0:
        eps = eos.get_eps(P)
        
        r_profile.append(r)
        m_profile.append(m)  # geometric mass in km
        p_profile.append(P)
        e_profile.append(eps)
        
        # RK4 step
        def tov_derivs(r_val: float, m_val: float, P_val: float) -> tuple[float, float]:
            if P_val <= 0:
                return 0.0, 0.0
            e_val = eos.get_eps(P_val)
            e_k = e_val * k_conv
            p_k = P_val * k_conv
            f = r_val * (r_val - 2.0 * m_val)
            if f <= 0.0:
                return 0.0, 0.0
            dm = 4.0 * math.pi * r_val**2 * e_k
            dp = -(e_k + p_k) * (m_val + 4.0 * math.pi * r_val**3 * p_k) / f / k_conv
            return dm, dp
            
        dm1, dp1 = tov_derivs(r, m, P)
        dm2, dp2 = tov_derivs(r + 0.5*dr, m + 0.5*dr*dm1, P + 0.5*dr*dp1)
        dm3, dp3 = tov_derivs(r + 0.5*dr, m + 0.5*dr*dm2, P + 0.5*dr*dp2)
        dm4, dp4 = tov_derivs(r + dr, m + dr*dm3, P + dr*dp3)
        
        m += (dr / 6.0) * (dm1 + 2*dm2 + 2*dm3 + dm4)
        P += (dr / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
        r += dr
        
    return np.array(r_profile), np.array(m_profile), np.array(p_profile), np.array(e_profile)

def main():
    print("=" * 80)
    print("     NVG NEUTRON STAR COMPOSITION g-MODE OSCILLATIONS")
    print("=" * 80)
    
    # 1. Load EOS
    eos = EOS(p_match=1.5, Gamma=1.35)
    
    # 2. Bisection search to find Pc that yields exactly 1.40 M_sun
    print("Searching for central pressure of a 1.4 M_sun neutron star...")
    Pc_min, Pc_max = 5.0, 400.0
    Pc_fit = 0.0
    r_prof = m_prof = p_prof = e_prof = None
    
    for _ in range(25):
        Pc_mid = (Pc_min + Pc_max) / 2.0
        r_p, m_p, p_p, e_p = integrate_tov_profiles(eos, Pc_mid)
        # Final mass in M_sun
        M_final = m_p[-1] / M_sun_km
        if M_final < 1.40:
            Pc_min = Pc_mid
        else:
            Pc_max = Pc_mid
            
    Pc_fit = Pc_mid
    r_prof, m_prof, p_prof, e_prof = integrate_tov_profiles(eos, Pc_fit)
    M_ns = m_prof[-1] / M_sun_km
    R_ns = r_prof[-1]
    
    print(f"Matched central pressure                 : {Pc_fit:.4f} MeV/fm³")
    print(f"Neutron star mass                        : {M_ns:.3f} M_sun")
    print(f"Neutron star radius                      : {R_ns:.3f} km")
    print("-" * 80)
    
    # 3. Calculate buoyancy Brunt-Vaisala frequency profile N(r)
    # N^2 = g^2 * e^{-2Lambda} * (eps + P) / P * delta_comp
    # where delta_comp = 1/Gamma_eq - 1/Gamma_th.
    # For a cold NS core, delta_comp ~ 1.5e-4 to 2.5e-4. Let's use 2.0e-4.
    delta_comp = 2.0e-4
    c_light = 2.99792e5  # km/s
    
    p_k = p_prof * k_conv
    e_k = e_prof * k_conv
    
    # Gravitational acceleration profile g(r) in geometric units (km^-1)
    g_local = (m_prof + 4.0 * np.pi * r_prof**3 * p_k) / (r_prof * (r_prof - 2.0 * m_prof))
    
    # General relativistic factor e^{-Lambda} = sqrt(1 - 2m/r)
    e_minus_lambda = np.sqrt(1.0 - 2.0 * m_prof / r_prof)
    
    # Brunt-Vaisala frequency N(r) in km^-1
    N_geom = g_local * e_minus_lambda * np.sqrt((e_prof + p_prof) / p_prof * delta_comp)
    # Convert N to s^-1 (Hz)
    N_sec = N_geom * c_light
    
    # Integrate WKB fundamental g-mode period: T_g = 2 * pi^2 / \int_0^R (N(r)/r) dr
    # Avoid division by zero at the center: integrate from r = 0.5 km to R
    core_mask = r_prof > 0.5
    r_core = r_prof[core_mask]
    N_core = N_geom[core_mask]
    
    integral = np.trapezoid(N_core / r_core, r_core)
    
    # Fundamental l=2 g-mode period
    T_g_seconds = (2.0 * np.pi**2 * np.sqrt(6.0)) / (integral * c_light)
    T_g_ms = T_g_seconds * 1000.0
    
    print(f"Buoyancy parameter delta_comp            : {delta_comp:.2e}")
    print(f"Max Brunt-Vaisala frequency in core      : {np.max(N_sec):.2f} rad/s")
    print(f"Core-averaged Brunt-Vaisala frequency    : {np.mean(N_sec[core_mask]):.2f} rad/s")
    print(f"Fundamental l=2 g-mode period            : T_g = {T_g_ms:.2f} ms")
    print("-" * 80)
    
    # Print profile table
    sample_radii = [1.0, 3.0, 5.0, 7.0, 9.0, 10.0]
    print(f"  {'Radius r (km)':<15} | {'m(r) (M_sun)':<15} | {'g(r) (km⁻¹)':<15} | {'N(r) (rad/s)':<15}")
    print("  " + "-" * 65)
    for rad in sample_radii:
        if rad < R_ns:
            idx = np.argmin(np.abs(r_prof - rad))
            print(f"  {r_prof[idx]:<15.2f} | {m_prof[idx]/M_sun_km:<15.3f} | {g_local[idx]:<15.4f} | {N_sec[idx]:<15.2f}")
            
    print("-" * 80)
    
    # Assertions for correctness
    assert T_g_ms > 50.0 and T_g_ms < 150.0, "Fundamental g-mode period out of physical bounds!"
    
    print("Neutron star g-mode period verification PASSED.")

if __name__ == "__main__":
    main()
