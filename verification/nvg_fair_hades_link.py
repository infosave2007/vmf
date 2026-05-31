#!/usr/bin/env python3
"""
NVG Hadron Universality & FAIR/HADES Observables

This script executes points 3, 4, & 5 of the NVG Research Program:
3. Calculates f_Omega for the hadron spectrum. (Also in nvg_hadron_mass_fractions.py)
4. Checks the universality of f_Omega.
5. Links the NVG vacuum melting parameter kappa_Omega(n_B) to the expected 
   in-medium mass shifts of vector mesons (rho, omega), which are directly 
   measurable via dielectron spectra at the FAIR (CBM) and GSI (HADES) facilities.
"""

from __future__ import annotations
import math

# ── Constants & Parameters ──────────────────────────────────────────
n_0 = 0.16  # Saturation density (fm^-3)

# Analytical derivation of the melting parameter kappa_1 from W-field coupling:
# kappa_1 = (mu_theta * g_omega * A_0(n_0)) / (lambda * W_vac^2)
# With mu_theta = m_omega = 782.6 MeV, g_omega = 10.12, A_0(n_0) ≈ 20.31 MeV,
# lambda ≈ 1.05, and W_vac = M_Omega_0 = 859.0 MeV, we get:
# kappa_1 = (782.6 * 10.12 * 20.31) / (1.05 * 859.0^2) ≈ 0.207 ≈ 0.21.
# The parameter kappa_2 = 0.80 governs the non-linear saturation.
kappa_1 = 0.21
kappa_2 = 0.80

# ── Hadron Spectrum (Points 3 & 4) ──────────────────────────────────
# Hadron: (Mass_MeV, Current_Mass_MeV)
HADRONS = {
    "Nucleon (N)": (939.0, 80.0),    # mostly light quarks
    "Pion (pi)": (139.6, 67.0),      # Goldstone boson, highly current-mass dependent
    "Kaon (K)": (493.7, 362.0),      # Contains strange quark
    "Rho (rho)": (775.3, 80.0),      # Vector meson, light quarks
    "Omega (omega)": (782.6, 80.0),  # Vector meson, light quarks
    "Phi (phi)": (1019.5, 300.0),    # Hidden strange (s-sbar)
    "Lambda (Lambda)": (1115.7, 200.0) # Hyperon (u d s)
}

def analyze_universality() -> None:
    print("=" * 80)
    print("POINTS 3 & 4: HADRON SPECTRUM & f_Omega UNIVERSALITY")
    print("=" * 80)
    print(f"  {'Hadron':<18} {'Total Mass':<15} {'M_Omega':<15} {'f_Omega (%)':<15}")
    print("  " + "-" * 62)
    
    for name, (M_H, M_cur) in HADRONS.items():
        M_omega = M_H - M_cur
        f_omega = M_omega / M_H * 100.0
        print(f"  {name:<18} {M_H:<15.1f} {M_omega:<15.1f} {f_omega:<15.1f}")
        
    print()
    print("CONCLUSION (Universality Check):")
    print("f_Omega is NOT strictly universal across the entire hadron spectrum.")
    print("Goldstone bosons (pions, kaons) have a huge current-mass fraction because")
    print("chiral symmetry breaking protects their mass. However, for non-Goldstone")
    print("hadrons composed of light quarks (Nucleon, Rho, Omega), f_Omega is highly")
    print("universal, clustering around ~90%. Heavy/strange hadrons have intermediate")
    print("f_Omega values.")
    print()


# ── In-Medium Mass Shifts for FAIR/HADES (Point 5) ──────────────────

def in_medium_mass(M_H: float, M_cur: float, n_B: float) -> float:
    """Calculates the in-medium mass of a hadron according to NVG."""
    M_omega_vac = M_H - M_cur
    
    x = n_B / n_0
    if x <= 0:
        return M_H
        
    # NVG vacuum melting function
    M_omega_med = M_omega_vac * (1.0 + kappa_2 * x) ** (-kappa_1 / kappa_2)
    return M_cur + M_omega_med

def link_to_hades() -> None:
    print("=" * 80)
    print("POINT 5: LINKING kappa_Omega TO HADES/CBM DATA (FAIR)")
    print("=" * 80)
    print("HADES (GSI) and CBM (FAIR) measure the invariant mass of dielectron (e+e-)")
    print("pairs emitted from dense nuclear matter in heavy-ion collisions.")
    print("The dominant source of these pairs is the decay of vector mesons (rho, omega).")
    print("If the QCD vacuum melts (as NVG predicts), the mass of these mesons should")
    print("shift downward in the dense medium.")
    print()
    
    densities = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    print("Predicted In-Medium Mass Shifts:")
    print(f"  {'Density (n/n_0)':<18} {'Rho Mass (MeV)':<18} {'Omega Mass (MeV)':<18} {'Shift ΔM_rho (%)':<18}")
    print("  " + "-" * 72)
    
    for n in densities:
        n_B = n * n_0
        
        M_rho_vac = HADRONS["Rho (rho)"][0]
        M_rho_cur = HADRONS["Rho (rho)"][1]
        M_rho_med = in_medium_mass(M_rho_vac, M_rho_cur, n_B)
        
        M_omega_vac = HADRONS["Omega (omega)"][0]
        M_omega_cur = HADRONS["Omega (omega)"][1]
        M_omega_med = in_medium_mass(M_omega_vac, M_omega_cur, n_B)
        
        shift_pct = (1.0 - M_rho_med / M_rho_vac) * 100.0
        
        print(f"  {n:<18.1f} {M_rho_med:<18.1f} {M_omega_med:<18.1f} {shift_pct:<18.1f}")
        
    print()
    print("CONCLUSION (FAIR/HADES Observable):")
    print("The NVG model predicts a ~13% mass drop for the rho meson at saturation")
    print("density (n_0), and a massive ~20% drop at 2 n_0 (typical collision density).")
    print("This provides a direct, highly falsifiable experimental signature that")
    print("ties the NVG parameter kappa_Omega(n_B) to the dielectron invariant mass")
    print("spectra currently being measured by HADES and planned for FAIR CBM.")
    print()


def main() -> None:
    analyze_universality()
    link_to_hades()


if __name__ == "__main__":
    main()
