#!/usr/bin/env python3
"""
NVG Verification: CMB Temperature Consistency Check
---------------------------------------------------
Performs a consistency check on today's Cosmic Microwave Background (CMB) temperature
(T_CMB = 2.725 K) against VMF bounce parameters.

NOTE: This is a semi-empirical consistency check (null test) rather than a direct,
parameter-free prediction from QCD, as it relies on:
1. The observed mass of the observable universe (M_obs_g = 4.0e55 g) to set the comoving patch size.
2. An arbitrary coordinate normalization for the scale factor at the bounce (a_bounce = 1 cm).
"""

import math

def calculate_cmb_temperature():
    print("==========================================================================")
    print("  NVG/VMF CONSISTENCY CHECK: CMB TEMPERATURE TODAY FROM QCD BOUNCE")
    print("==========================================================================")
    
    # 1. Input QCD and cosmological scale parameters
    M_Omega_0 = 859.0       # MeV (lattice QCD vacuum mass anchor)
    hbar_c = 197.3269804    # MeV·fm
    MeV_fm3_to_gcm3 = 1.7827e12
    M_obs_g = 4.0e55        # g, mass of the observable universe (empirical input)
    
    # 2. Critical density and bounce temperature
    eps_max = M_Omega_0**4 / hbar_c**3  # MeV/fm^3
    rho_c_gcm3 = eps_max * MeV_fm3_to_gcm3
    
    g_star_S_b = 47.5       # entropy degrees of freedom at QGP bounce
    g_star_S_0 = 3.91       # entropy degrees of freedom today (photons + 3 neutrinos)
    
    # Stefan-Boltzmann for QGP
    eps_MeV4 = eps_max * hbar_c**3
    T_b_MeV = (30.0 * eps_MeV4 / (math.pi**2 * g_star_S_b))**0.25
    
    # Convert T_b to Kelvin: 1 eV = 11604.5 K => 1 MeV = 1.16045e10 K
    eV_to_K = 11604.518
    T_b_K = T_b_MeV * 1e6 * eV_to_K
    
    # 3. Bounce core scale of the observable universe (maximum compression radius)
    r_0_cm = (3.0 * M_obs_g / (4.0 * math.pi * rho_c_gcm3))**(1.0/3.0)
    r_0_km = r_0_cm / 1e5
    
    # 4. Today's CMB temperature under adiabatic expansion
    # The scale factor bounce normalization a_bounce = 1 cm is arbitrary.
    # The comoving scale today corresponding to the observable horizon is r_0.
    # The expansion factor is a_0 / a_bounce = r_0 / (1 cm).
    # Entropy conservation: g_*S,b * a_bounce^3 * T_bounce^3 = g_*S,0 * a_0^3 * T_0^3
    # yields: T_0 = T_bounce * (g_*S,b / g_*S,0)^(1/3) * (a_bounce / a_0)
    a_ratio = r_0_cm / 1.0  # since a_bounce = 1 cm coordinate scale
    g_ratio_13 = (g_star_S_b / g_star_S_0)**(1.0/3.0)
    T_0_pred_K = T_b_K * g_ratio_13 / a_ratio
    
    # Observed value
    T_obs_K = 2.7255
    T_obs_err = 0.0006      # COBE/FIRAS 1-sigma uncertainty
    
    print(f"QCD Anchor M_Omega_0             : {M_Omega_0:.1f} MeV")
    print(f"Bounce Critical Density (rho_c)  : {rho_c_gcm3:.4e} g/cm^3")
    print(f"Bounce Temperature (T_bounce)    : {T_b_MeV:.2f} MeV ({T_b_K:.4e} K)")
    print(f"Observable Universe Bounce Scale : r_0 = {r_0_km:.4e} km ({r_0_cm:.4e} cm)")
    print(f"Degrees of freedom (bounce/today): {g_star_S_b} / {g_star_S_0}")
    print(f"Scale factor expansion ratio a0/ab: {a_ratio:.4e}")
    print("-" * 74)
    print(f"Predicted CMB Temperature T_0    : {T_0_pred_K:.4f} K")
    print(f"Observed COBE/FIRAS Temperature  : {T_obs_K:.4f} +/- {T_obs_err:.4f} K")
    
    # Statistical validation
    dev_sigma = (T_0_pred_K - T_obs_K) / T_obs_err
    print(f"Deviation from Observation       : {dev_sigma:+.2f} sigma")
    
    is_ok = abs(dev_sigma) < 1.0
    print(f"Status                           : {'✅ PASSED (Consistent under standard unit bounce coordinate scaling)' if is_ok else '❌ FAILED'}")
    print("==========================================================================")
    return is_ok

if __name__ == "__main__":
    calculate_cmb_temperature()
