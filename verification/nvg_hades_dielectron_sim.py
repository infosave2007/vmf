#!/usr/bin/env python3
"""
NVG Hadron Physics: HADES/CBM Dielectron Spectrum Simulator
------------------------------------------------------------
This script simulates the in-medium spectral function of the rho-meson
and the resulting e+e- dielectron invariant mass distribution in heavy-ion
collisions, comparing:
1. Vacuum decay (Breit-Wigner peak at 775 MeV).
2. Standard collision broadening (no mass shift, width increases).
3. NVG/VMF model (broadening + dynamic vacuum melting mass shift).
"""

from __future__ import annotations
import math
import numpy as np

# Physical constants and parameters
n_0 = 0.16                     # Saturation density (fm^-3)
alpha_EM = 1.0 / 137.036       # Fine structure constant
hbar_c = 197.327               # MeV fm

# Rho meson vacuum parameters
M_rho_vac = 775.3              # MeV
Gamma_rho_vac = 149.1          # MeV
M_rho_cur = 80.0               # MeV, current quark mass contribution
M_rho_omega_vac = M_rho_vac - M_rho_cur

# NVG vacuum melting parameters
kappa_1 = 0.25
kappa_2 = 0.80

# Fireball evolution parameters (Au+Au collision at SIS18)
n_max = 2.5 * n_0              # Peak baryon density
t_peak = 8.0                   # fm/c, time of maximum compression
tau = 3.0                      # fm/c, duration scale of dense phase
t_max = 20.0                   # fm/c, maximum integration time
dt = 0.1                       # fm/c, time step

# Meson width broadening parameters
delta_Gamma_coll = 50.0        # MeV, additional collision width at n_0

def get_density(t: float) -> float:
    """Calculates baryon density at time t (fm/c) using a Gaussian fireball profile."""
    return n_max * math.exp(-((t - t_peak) ** 2) / (2.0 * (tau ** 2)))

def get_in_medium_mass(n_B: float) -> float:
    """Calculates in-medium mass of the rho-meson under VMF vacuum melting."""
    x = n_B / n_0
    if x <= 0:
        return M_rho_vac
    # VMF scaling of the non-perturbative mass component
    M_omega_med = M_rho_omega_vac * (1.0 + kappa_2 * x) ** (-kappa_1 / kappa_2)
    return M_rho_cur + M_omega_med

def get_in_medium_width(n_B: float) -> float:
    """Calculates in-medium decay width of the rho-meson including collision broadening."""
    return Gamma_rho_vac + delta_Gamma_coll * (n_B / n_0)

def spectral_function(M: float, M_r: float, Gamma_r: float) -> float:
    """Calculates the Breit-Wigner spectral function shape of the rho meson."""
    numerator = (M_r ** 3) * Gamma_r
    denominator = (M**2 - M_r**2)**2 + (M_r**2) * (Gamma_r**2)
    return numerator / denominator

def simulate_spectrum(M_array: np.ndarray, mode: str) -> np.ndarray:
    """
    Integrates the differential dielectron emission rate dR_ee/dM over the fireball history.
    Rate: dR_ee/dM ~ 1/M^2 * SpectralFunction(M, M_r, Gamma_r)
    """
    spectrum = np.zeros_like(M_array)
    t_points = np.arange(0.0, t_max, dt)
    
    for t in t_points:
        n_B = get_density(t)
        
        # Select mass and width parameters based on the mode
        if mode == "vacuum":
            M_r = M_rho_vac
            Gamma_r = Gamma_rho_vac
        elif mode == "broadening":
            M_r = M_rho_vac
            Gamma_r = get_in_width = get_in_medium_width(n_B)
        elif mode == "vmf":
            M_r = get_in_medium_mass(n_B)
            Gamma_r = get_in_medium_width(n_B)
        else:
            raise ValueError(f"Unknown simulation mode: {mode}")
            
        for idx, M in enumerate(M_array):
            # In-medium decay rate weighting factor 1/M^2
            rate = (1.0 / M**2) * spectral_function(M, M_r, Gamma_r)
            # Integrate over time step (volume is assumed constant/normalized)
            spectrum[idx] += rate * dt
            
    # Normalize the spectrum so that the integral of dN/dM is 1 for comparison
    dx = np.diff(M_array)
    total_area = float(np.sum((spectrum[:-1] + spectrum[1:]) / 2.0 * dx))
    if total_area > 0:
        spectrum = spectrum / total_area
        
    return spectrum

def main():
    print("=" * 80)
    print("      NVG HADES/CBM DIELECTRON SPECTRA SIMULATION")
    print("=" * 80)
    
    # Invariant mass range (MeV)
    M_min, M_max_val = 300.0, 1000.0
    M_array = np.linspace(M_min, M_max_val, 700)
    
    # Run simulations
    spec_vac = simulate_spectrum(M_array, "vacuum")
    spec_broad = simulate_spectrum(M_array, "broadening")
    spec_vmf = simulate_spectrum(M_array, "vmf")
    
    # Find peak positions
    peak_vac_idx = np.argmax(spec_vac)
    peak_broad_idx = np.argmax(spec_broad)
    peak_vmf_idx = np.argmax(spec_vmf)
    
    M_peak_vac = M_array[peak_vac_idx]
    M_peak_broad = M_array[peak_broad_idx]
    M_peak_vmf = M_array[peak_vmf_idx]
    
    # Calculate peak relative shifts
    shift_broad_pct = (M_peak_broad - M_peak_vac) / M_peak_vac * 100.0
    shift_vmf_pct = (M_peak_vmf - M_peak_vac) / M_peak_vac * 100.0
    
    print(f"Fireball peak density                   : {n_max / n_0:.2f} n_0")
    print(f"Rho meson vacuum peak position          : {M_peak_vac:.1f} MeV")
    print(f"Collision broadening-only peak position : {M_peak_broad:.1f} MeV (shift: {shift_broad_pct:+.2f}%)")
    print(f"NVG/VMF model peak position             : {M_peak_vmf:.1f} MeV (shift: {shift_vmf_pct:+.2f}%)")
    print("-" * 80)
    
    # Print a table of spectral shape to stdout
    print(f"  {'Mass (MeV)':<15} | {'Vacuum (norm)':<18} | {'Broadening (norm)':<20} | {'VMF (norm)':<15}")
    print("  " + "-" * 72)
    sample_masses = [400.0, 500.0, 600.0, 700.0, 775.0, 850.0, 950.0]
    for m in sample_masses:
        # Find closest index
        idx = np.argmin(np.abs(M_array - m))
        print(f"  {M_array[idx]:<15.1f} | {spec_vac[idx]:<18.6e} | {spec_broad[idx]:<20.6e} | {spec_vmf[idx]:<15.6e}")
        
    print("-" * 80)
    print("ANALYSIS & INTERPRETATION:")
    print("- Under standard collision broadening, the peak remains at ~775 MeV but gets")
    print("  widely smeared out (width is doubled).")
    print("- Under NVG/VMF, the vacuum melting of W shifts the entire strength of the")
    print("  spectral density to lower invariant masses, resulting in a prominent peak shift")
    print(f"  down to ~{M_peak_vmf:.1f} MeV.")
    print("- This shift is highly distinguishable from mere collision broadening and serves")
    print("  as a direct smoking gun for VMF at GSI/FAIR energies.")
    print("=" * 80)
    
    # Assertions for correctness
    assert M_peak_vac > 760.0 and M_peak_vac < 775.0, "Vacuum peak position out of bounds!"
    assert abs(M_peak_broad - M_peak_vac) < 20.0, "Broadening peak shift is unexpectedly large!"
    assert M_peak_vmf < 720.0, "VMF mass shift is too small!"
    assert M_peak_vmf > 680.0, "VMF mass shift is too large!"
    
    print("HADES dielectron spectrum simulation verified successfully.")

if __name__ == "__main__":
    main()
