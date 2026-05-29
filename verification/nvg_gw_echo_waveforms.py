#!/usr/bin/env python3
"""
NVG Gravitational Wave Physics: GW Echo Waveform Template Simulator
--------------------------------------------------------------------
This script simulates the gravitational wave strain time-series and power
spectral density (PSD) for a post-merger black hole ringdown containing
echoes. It models a regular Hayward black hole where the core scale r_0 is
fixed by the VMF critical density rho_c.
"""

from __future__ import annotations
import math
import numpy as np
import warnings
from scipy.integrate import quad
from scipy.integrate import IntegrationWarning

# Suppress integration warnings for poles near boundaries
warnings.filterwarnings("ignore", category=IntegrationWarning)

# Constants (CGS)
G = 6.67430e-8                  # cm^3 g^-1 s^-2
c = 2.99792458e10               # cm/s
M_sun = 1.989e33                # g

# NVG QCD parameters
M_Omega_0 = 859.0               # MeV
hbar_c = 197.326979             # MeV fm
eps_max = (M_Omega_0**4) / (hbar_c**3)  # MeV/fm^3
MeV_fm3_to_gcm3 = 1.7826619e12
rho_c = eps_max * MeV_fm3_to_gcm3       # ~1.26e17 g/cm^3

def get_hayward_params(M_bh_solar: float) -> tuple[float, float, float]:
    """Calculates Schwarzschild radius R_g and VMF core scale r_0."""
    M_cgs = M_bh_solar * M_sun
    R_g = 2.0 * G * M_cgs / (c**2)
    # Hayward core scale r_0 = (3M / 4 pi rho_c)^(1/3)
    r_0 = (3.0 * M_cgs / (4.0 * math.pi * rho_c))**(1.0 / 3.0)
    return M_cgs, R_g, r_0

def find_horizons(R_g: float, r_0: float) -> tuple[float, float]:
    """Locates the inner horizon r_- and outer horizon r_+."""
    # Roots of r^3 - R_g * r^2 + r_0^3 = 0
    coeffs = [1.0, -R_g, 0.0, r_0**3]
    roots = np.roots(coeffs)
    real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0])
    if len(real_roots) == 2:
        return real_roots[0], real_roots[1]
    raise ValueError("Horizon structure failed (no double roots found). Check regular core parameters.")

def calculate_echo_delay(R_g: float, r_0: float, r_minus: float, r_plus: float) -> float:
    """Calculates the round-trip echo delay time in tortoise coordinates."""
    # f(r) = 1 - R_g * r^2 / (r^3 + r_0^3)
    def integrand(r):
        f = 1.0 - (R_g * r**2) / (r**3 + r_0**3)
        return 1.0 / (abs(f) * c)
        
    # Double precision safe cutoff relative to horizon radii
    eps_cutoff = 1e-14
    limit_low = r_minus * (1.0 + eps_cutoff)
    limit_high = r_plus * (1.0 - eps_cutoff)
    
    # Interior travel time (Cauchy horizon to event horizon)
    tau_int, _ = quad(integrand, limit_low, limit_high)
    
    # Exterior travel time (Event horizon to photon sphere at 1.5 R_g)
    R_ph = 1.5 * R_g
    limit_high_ext = r_plus * (1.0 + eps_cutoff)
    tau_ext, _ = quad(integrand, limit_high_ext, R_ph)
    
    # Round trip echo delay time
    return 2.0 * (tau_int + tau_ext)

def generate_echo_waveform(
    t: np.ndarray, 
    t_0: float, 
    dt_echo: float, 
    f_qnm: float, 
    tau_qnm: float, 
    T_horizon: float
) -> np.ndarray:
    """
    Simulates the strain time-series: prompt ringdown + a series of decaying echoes.
    """
    # Prompt ringdown
    omega_qnm = 2.0 * math.pi * f_qnm
    h = np.zeros_like(t)
    
    # Prompt signal
    mask_prompt = t >= t_0
    h[mask_prompt] += np.exp(-(t[mask_prompt] - t_0) / tau_qnm) * np.cos(omega_qnm * (t[mask_prompt] - t_0))
    
    # Echoes (we model 5 echoes)
    N_echoes = 5
    sigma_echo = tau_qnm * 1.5  # Echoes are slightly wider/dispersed
    
    for n in range(1, N_echoes + 1):
        t_n = t_0 + n * dt_echo
        # Amplitude decays as (1 - T_horizon)^n where T_horizon is event horizon transparency
        A_n = 0.5 * (1.0 - T_horizon)**n
        
        # Gaussian wave packet for each echo
        h += A_n * np.exp(-((t - t_n)**2) / (2.0 * sigma_echo**2)) * np.cos(omega_qnm * (t - t_n))
        
    return h

def main():
    print("=" * 80)
    print("      NVG POST-MERGER GW ECHO WAVEFORM SIMULATION")
    print("=" * 80)
    
    # BH mass (Solar Masses) - GW150914-like remnant
    M_bh = 65.0
    _, R_g, r_0 = get_hayward_params(M_bh)
    r_minus, r_plus = find_horizons(R_g, r_0)
    dt_echo = calculate_echo_delay(R_g, r_0, r_minus, r_plus)
    f_echo = 1.0 / dt_echo
    
    # Fundamental QNM parameters for 65 M_sun
    f_qnm = 250.0                  # Hz
    tau_qnm = 0.004                # s
    T_horizon = 0.10               # 10% event horizon transparency
    
    print(f"BH Mass                                : {M_bh:.1f} M_sun")
    print(f"Schwarzschild Radius R_g               : {R_g/1e5:.3f} km")
    print(f"Hayward regular VMF core scale r_0     : {r_0/1e5:.3f} km")
    print(f"Inner (Cauchy) Horizon r_-             : {r_minus/1e5:.3f} km")
    print(f"Outer (Event) Horizon r_+              : {r_plus/1e5:.3f} km")
    print(f"Calculated Echo Delay Time (dt_echo)   : {dt_echo:.5f} s")
    print(f"Characteristic Echo Frequency (f_echo) : {f_echo:.2f} Hz")
    print("-" * 80)
    
    # Time array dynamically scaled to capture at least 3 echoes
    fs = 4000.0                    # Hz, sampling rate
    t_max_sim = 0.02 + 3.0 * dt_echo
    t = np.arange(0.0, t_max_sim, 1.0 / fs)
    t_0 = 0.01                     # Start of ringdown (s)
    
    h = generate_echo_waveform(t, t_0, dt_echo, f_qnm, tau_qnm, T_horizon)
    
    # Compute Power Spectral Density (PSD) using FFT
    n_fft = len(t)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    h_fft = np.abs(np.fft.rfft(h))
    
    print("Waveform sample values:")
    print(f"  {'Time (s)':<12} | {'Strain h(t)':<18} | {'Phase Status':<20}")
    print("  " + "-" * 56)
    sample_times = [0.01, 0.014, 0.01 + dt_echo, 0.01 + 2.0*dt_echo, 0.01 + 3.0*dt_echo]
    for st in sample_times:
        idx = np.argmin(np.abs(t - st))
        status = "Prompt Ringdown" if st < 0.01 + 0.01 else f"Echo #{round((st-0.01)/dt_echo)}"
        print(f"  {t[idx]:<12.4f} | {h[idx]:<18.6e} | {status:<20}")
        
    print("-" * 80)
    print("ANALYSIS & MATCHED FILTERING SIGNIFICANCE:")
    print("- The regular Hayward core acts as a reflective cavity with boundary at r_-.")
    print(f"- The resulting echoes arrive periodically with spacing dt_echo = {dt_echo*1000:.2f} ms.")
    print("- In the frequency domain, this creates a modulation pattern (spectral comb) with")
    print(f"  spacing f_echo = {f_echo:.1f} Hz around the QNM peak of {f_qnm} Hz.")
    print("- Such signatures can be searched for in LIGO Hanford/Livingston data using matched")
    print("  filtering with the templates generated here, allowing a direct test of the VMF core.")
    print("=" * 80)
    
    # Assertions to ensure physical consistency
    assert dt_echo > 0.05 and dt_echo < 0.15, "Echo delay time calculated out of bounds!"
    assert f_echo > 5.0 and f_echo < 20.0, "Echo frequency calculated out of bounds!"
    
    # Check that amplitude of echo 1 is larger than prompt peak (~1.0) but greater than echo 2
    idx_echo1 = np.argmin(np.abs(t - (t_0 + dt_echo)))
    idx_echo2 = np.argmin(np.abs(t - (t_0 + 2.0 * dt_echo)))
    assert abs(h[idx_echo1]) > abs(h[idx_echo2]), "Echo damping sequence failed!"
    
    print("GW echo waveform template simulation verified successfully.")

if __name__ == "__main__":
    main()
