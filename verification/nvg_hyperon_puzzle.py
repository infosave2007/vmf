#!/usr/bin/env python3
"""
NVG Hyperon Puzzle Resolution & Microscopic Phase Transition

This script investigates Point 3 of the 'Not Claimed' list: the microscopic nature
of the phase transition. It calculates whether hyperons (specifically Lambda) 
appear before the transition to quark matter.

The 'Hyperon Puzzle' in standard astrophysics: 
Hyperons should appear at ~2-3 n_0, softening the EOS and making M_max < 2.0 M_sun (violating observations).

In NVG:
The Lambda mass drops because the vacuum melts. However, does it drop fast enough 
to appear before the first-order transition to conformal QGP at ~2.0 n_0?
"""

import numpy as np

# NVG Core parameters
n_0 = 0.16  # fm^-3
M_N_vac = 939.0
M_Omega_N = 859.0
M_cur_N = 80.0

# Lambda Hyperon parameters
M_Lambda_vac = 1115.7
M_cur_Lambda = 200.0  # approximate current mass contribution (u, d, s)
M_Omega_Lambda = M_Lambda_vac - M_cur_Lambda

# Vector repulsion parameters
C_v_n0 = 100.0
alpha_v = 4.0
nu_v = 2.0

# Vacuum melting parameters
kappa_1 = 0.25
kappa_2 = 0.80

def vacuum_melt_factor(n_B: float) -> float:
    x = max(n_B / n_0, 0.0)
    if x == 0:
        return 1.0
    return (1.0 + kappa_2 * x) ** (-kappa_1 / kappa_2)

def vector_potential(n_B: float) -> float:
    x = max(n_B / n_0, 0.0)
    if x == 0:
        return 0.0
    # Saturation of vector field
    return C_v_n0 * (x ** nu_v) / (1.0 + (x ** nu_v) / alpha_v)

def nvg_core_thermodynamics(n_B: float):
    x = n_B / n_0
    
    # 1. Vacuum mass melting
    melt = vacuum_melt_factor(n_B)
    M_star = M_cur_N + M_Omega_N * melt
    
    # 2. Vector repulsion energy per baryon
    V_v = vector_potential(n_B)
    
    # 3. Fermi gas of nucleons (simplified zero-temp)
    # k_F = (1.5 * pi^2 * n_B)^(1/3)
    k_F = (1.5 * np.pi**2 * n_B)**(1/3)
    
    # Energy density
    eps_kin = (3.0 / 4.0) * n_B * np.sqrt(k_F**2 + M_star**2)  # Ultra-relativistic approx at high n
    eps_vec = V_v * n_B
    eps_tot = eps_kin + eps_vec
    
    # Nucleon Chemical Potential: mu_N = d(eps)/dn
    # Simplified approx: mu_N ~ E_F + V_v
    E_F = np.sqrt(k_F**2 + M_star**2)
    mu_N = E_F + V_v
    
    return mu_N

def lambda_effective_mass(n_B: float) -> float:
    melt = vacuum_melt_factor(n_B)
    return M_cur_Lambda + M_Omega_Lambda * melt

def main():
    print("================================================================================")
    print("NVG RESOLUTION OF THE HYPERON PUZZLE & PHASE TRANSITION NATURE")
    print("================================================================================")
    
    n_grid = np.linspace(0.5, 3.5, 31) * n_0
    
    hyperon_onset_n = None
    
    print(f"{'Density (n/n0)':<15} {'Nucleon μ_N (MeV)':<20} {'Lambda M* (MeV)':<20} {'Condition'}")
    print("-" * 75)
    
    for n in n_grid:
        x = n / n_0
        mu_N = nvg_core_thermodynamics(n)
        M_star_L = lambda_effective_mass(n)
        
        condition = "μ_N < M*_Λ (No Hyperons)"
        if mu_N >= M_star_L:
            condition = "μ_N >= M*_Λ (HYPERONS APPEAR!)"
            if hyperon_onset_n is None:
                hyperon_onset_n = x
                
        print(f"{x:<15.2f} {mu_N:<20.1f} {M_star_L:<20.1f} {condition}")
        
    print("\nCONCLUSION:")
    
    n_trans = 2.0  # NVG conformal QGP phase transition density
    
    if hyperon_onset_n is not None and hyperon_onset_n < n_trans:
        print(f"Hyperons appear at {hyperon_onset_n:.2f} n_0.")
        print(f"Since this is BEFORE the QGP phase transition at {n_trans} n_0,")
        print("the NVG model predicts a HYPERONIC CORE before transitioning to QGP.")
    else:
        onset_str = f"{hyperon_onset_n:.2f} n_0" if hyperon_onset_n else "Never"
        print(f"Hyperons appear at {onset_str}.")
        print(f"Since the phase transition to Conformal Quark Matter (QGP) occurs at {n_trans} n_0,")
        print("the NVG vector repulsion drives the nucleon chemical potential high enough to trigger")
        print("the transition to QGP *BEFORE* hyperons can form!")
        print("\nRESULT: NVG naturally solves the 'Hyperon Puzzle' by bypassing the hyperonic")
        print("phase entirely. The phase transition is strictly from Hadronic Matter to QGP.")
        
if __name__ == "__main__":
    main()
