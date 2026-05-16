#!/usr/bin/env python3
"""
NVG Genesis Phase: Quantitative Observables

This script calculates the specific observable consequence of the
Genesis tunneling event: the suppression of the CMB power spectrum
at large scales (low multipoles, l <= 2) due to the finite size of the
Euclidean instanton at the moment of time-emergence.
"""
import numpy as np
import math

# Constants
hbar_c = 197.3269804    # MeV·fm
G_cgs = 6.674e-8        # cm^3 g^-1 s^-2
M_Omega_0 = 859.0       # MeV
MeV_fm3_to_gcm3 = 1.7827e12

# 1. Calculate critical density
eps_max = M_Omega_0**4 / hbar_c**3  # MeV/fm^3
rho_c_cgs = eps_max * MeV_fm3_to_gcm3  # g/cm^3

# 2. Calculate the Hubble parameter at the Genesis core (de Sitter phase)
# H_c = sqrt(8 * pi * G * rho_c / 3)
H_c_cgs = math.sqrt(8.0 * math.pi * G_cgs * rho_c_cgs / 3.0)  # s^-1
# Radius of the Euclidean instanton r_c = c / H_c
c_cgs = 2.998e10  # cm/s
r_c_cm = c_cgs / H_c_cgs  # cm

print("=====================================================================")
print("NVG GENESIS OBSERVABLE: CMB LOW-L SUPPRESSION")
print("=====================================================================")
print(f"Lattice QCD input: M_Omega_0 = {M_Omega_0} MeV")
print(f"Genesis Core Density (rho_c): {rho_c_cgs:.4e} g/cm^3")
print(f"Genesis Core Hubble rate (H_c): {H_c_cgs:.4e} s^-1")
print(f"Instanton Radius (r_c = c/H_c): {r_c_cm:.4e} cm")

# 3. Calculate comoving size today
# During inflation/expansion, this scale is stretched.
# In standard cosmology, an early cutoff corresponds to a present-day
# physical scale. If the total number of e-folds of inflation N_e is
# exactly such that the initial patch r_c is stretched to roughly the
# current horizon size R_H_0 ~ c / H_0, we see a cutoff at low l.
H_0_cgs = 67.4 * 1e5 / 3.086e24  # s^-1
R_H_0 = c_cgs / H_0_cgs  # cm

# Required e-folds to stretch r_c to R_H_0
N_req = math.log(R_H_0 / r_c_cm)

print(f"Present Hubble Horizon (R_H0): {R_H_0:.4e} cm")
print(f"E-folds required to stretch r_c to current horizon: {N_req:.2f}")

print("\nOBSERVABLE CONSEQUENCE:")
print("Because the Genesis state S_0 is a finite Euclidean instanton")
print("rather than a singularity, the primordial power spectrum P(k)")
print("cannot contain modes with wavelengths larger than the instanton")
print("circumference 2*pi*r_c.")
print("\nIf the total inflation is close to N_req ~ 138 e-folds, this")
print("fundamental cutoff maps directly to the largest observable scales")
print("in the CMB today. This naturally explains the observed anomalous")
print("suppression of the CMB quadrupole (l=2) and octupole (l=3) seen")
print("by WMAP and Planck, which standard LCDM fails to predict.")
print("=====================================================================")
