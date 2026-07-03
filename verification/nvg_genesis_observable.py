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
# The cycle index bounds the e-folds: since the turnaround horizon scales as R_n = r_c * 2^(n-1),
# any moment inside cycle n = 77 satisfies N_e ∈ [76*ln(2), 77*ln(2)] = [52.68, 53.38].
# NOTE (calibration, not derivation): H_0 = 72.8 km/s/Mpc is an INPUT here, anchored to the
# local distance-ladder measurement (SH0ES). N_req = ln(R_H0/r_c) ≈ 53.08 is computed FROM it
# and then reused as N_e in nvg_hubble_tension.py. The model does not independently predict
# the position within the cycle; only the interval above is a prediction.
H_0_cgs = 72.8 * 1e5 / 3.086e24  # s^-1
R_H_0 = c_cgs / H_0_cgs  # cm

# Required e-folds to stretch r_c to R_H_0
N_req = math.log(R_H_0 / r_c_cm)

print(f"Present Hubble Horizon (R_H0) for H0=72.8: {R_H_0:.4e} cm")
print(f"E-folds calibrated from local H_0 (bounded to [52.68, 53.38] by cycle 77): {N_req:.2f}")

print("\nOBSERVABLE CONSEQUENCE:")
print("Because the Genesis state S_0 is a finite Euclidean instanton")
print("rather than a singularity, the primordial power spectrum P(k)")
print("cannot contain modes with wavelengths larger than the instanton")
print("circumference 2*pi*r_c.")
print("\nSince the calibrated expansion yields N_req ≈ 53.08 e-folds, this")
print("fundamental cutoff maps directly to the largest observable scales")
print("in the CMB today. Tested against Planck 2018 TT (nvg_cmb_lowl_refit.py):")
print("the cutoff improves the low-l fit by Delta chi2 ~ +0.9 with the best-fit")
print("scale within ~1.2x of the predicted k_c = 1/R_H0 — a real but mild effect;")
print("the observed deficit is also consistent with cosmic variance.")
print("=====================================================================")
