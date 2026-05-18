#!/usr/bin/env python3
"""
NVG Verification: Detector-Level Forward Model (HADES/CBM/NICA)
---------------------------------------------------------------
Simulates the expected dilepton invariant mass spectrum (e+ e-) for 
the rho meson, incorporating the NVG in-medium mass shift (-23.2% at 2n_0),
collisional broadening, and detector resolution smearing.
"""
import numpy as np

print("=" * 70)
print(" NVG DETECTOR FORWARD MODEL (HADES / CBM / NICA)")
print("=" * 70)

# 1. Physics Parameters
mass_vac = 775.0        # MeV
width_vac = 149.0       # MeV
mass_medium = 595.0     # MeV (NVG predicted -23.2% shift at 2n_0)
width_medium = 250.0    # MeV (Collisional broadening)

# 2. Detector Resolution
resolution_sigma = 15.0 # MeV (Typical invariant mass resolution)

def breit_wigner(m, m0, gamma):
    """Relativistic Breit-Wigner distribution"""
    num = m * m0 * gamma
    den = (m**2 - m0**2)**2 + (m0 * gamma)**2
    return num / den

def gaussian(m, mu, sigma):
    """Gaussian for detector smearing"""
    return np.exp(-0.5 * ((m - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

def convolve_spectrum(mass_array, spectrum, sigma):
    """Smear spectrum with detector resolution"""
    dm = mass_array[1] - mass_array[0]
    smeared = np.zeros_like(spectrum)
    for i, m in enumerate(mass_array):
        weights = gaussian(mass_array, m, sigma)
        smeared[i] = np.sum(spectrum * weights) * dm
    return smeared

# 3. Simulate Spectrum
m_range = np.linspace(200, 1200, 500)
spec_vac = breit_wigner(m_range, mass_vac, width_vac)
spec_med = breit_wigner(m_range, mass_medium, width_medium)

# Mix 30% vacuum decays (halo) and 70% in-medium decays (fireball)
spec_mixed = 0.3 * spec_vac + 0.7 * spec_med

# Apply Detector Resolution
spec_observed = convolve_spectrum(m_range, spec_mixed, resolution_sigma)

peak_observed = m_range[np.argmax(spec_observed)]

print(f"  Vacuum peak     : {mass_vac:.1f} MeV (Gamma = {width_vac} MeV)")
print(f"  In-Medium peak  : {mass_medium:.1f} MeV (Gamma = {width_medium} MeV)")
print(f"  Detector Sigma  : {resolution_sigma:.1f} MeV")
print(f"  => OBSERVED PEAK: {peak_observed:.1f} MeV")
print("")
print("  OBSERVATIONAL FORECAST:")
print("  Due to fireball dynamics and detector smearing, the pure -23% shift")
print("  will appear as an asymmetric broadening with the primary peak shifted")
print(f"  down to ~{peak_observed:.1f} MeV in the dilepton channel.")
print("  If CBM/HADES observes this exact structure, NVG mass-melting is confirmed.")
print("======================================================================")
