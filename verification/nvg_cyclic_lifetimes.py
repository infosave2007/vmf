#!/usr/bin/env python3
"""
NVG Cyclic Lifetimes: Tolman's Entropy Growth

This script calculates the lifetime and maximum scale of the 1st Universe
(Genesis) and subsequent cycles, based on Richard Tolman's thermodynamic
accumulation of entropy in cyclic models, adapted for the NVG framework.
"""
import math

# ── Constants ──
G = 6.674e-8         # cm^3 / (g s^2)
c = 2.998e10         # cm/s
M_sun = 1.989e33     # g
yr_to_s = 3.154e7    # seconds in a year

# NVG Anchor
rho_c = 1.263e17     # g/cm^3

# To have global turnaround, we assume a closed universe (k=+1) or an
# effective cyclic turnaround mechanism where the turnaround radius R_max
# and lifetime T_life depend on the total mass/entropy of the cycle.
# According to Tolman's cyclic model, entropy S grows each cycle.
# Holographic entropy of the NVG core: S_core ~ R_core^2 ~ M^(2/3)
# T_life ~ G * M / c^3

# 1. First Universe (Genesis)
# Born directly from the Euclidean instanton.
# Instanton radius:
H_c = math.sqrt(8.0 * math.pi * G * rho_c / 3.0)
r_instanton = c / H_c  # ~ 1.13e5 cm (1.13 km)

# Mass of the first universe (if the instanton volume is filled with rho_c)
V_instanton = (4.0/3.0) * math.pi * r_instanton**3
M_1 = V_instanton * rho_c

# Lifetime of the first universe (Tolman turnaround T ~ G M / c^3)
# More precisely, T = pi * G * M / c^3
T_1_s = math.pi * G * M_1 / c**3
T_1_yr = T_1_s / yr_to_s

# Current Universe (Cycle N)
# Mass of observable universe ~ 10^53 kg = 10^56 g
M_current = 1.5e56  # g
T_current_s = math.pi * G * M_current / c**3
T_current_yr = T_current_s / yr_to_s

print("=====================================================================")
print(" NVG CYCLIC LIFETIMES: TOLMAN ENTROPY GROWTH")
print("=====================================================================")

print("\n--- CYCLE 1 (GENESIS UNIVERSE) ---")
print(f"Radius at birth: {r_instanton/1e5:.3f} km")
print(f"Total Mass: {M_1:.2e} g (~ {M_1/M_sun:.2e} Solar Masses)")
print(f"Lifetime of 1st Cycle: {T_1_s:.2e} seconds (~ {T_1_yr:.2e} years)")

print("\n--- CURRENT CYCLE (CYCLE 'N') ---")
print(f"Total Mass: {M_current:.2e} g")
print(f"Expected Turnaround Lifetime: {T_current_yr:.2e} years")

# Estimate cycle number N
# If entropy (and mass) grows by a factor 'f' per cycle due to BH evaporation
# and starlight. Suppose mass grows by factor of 2 every cycle:
# M_N = M_1 * 2^N
N_cycles = math.log(M_current / M_1) / math.log(2)
print(f"Estimated Cycle Number (if mass doubles each cycle): N ~ {int(N_cycles)}")

print("\nANALYSIS:")
print("Because the 1st Universe was born from a quantum instanton of")
print("finite size (~1.1 km), its total energy/mass was minuscule")
print("compared to today (only ~0.38 Solar Masses!).")
print("\nConsequently, the very FIRST cycle lived and died in a fraction")
print("of a second! Each subsequent cycle produced irreversible entropy,")
print("increasing the turnaround radius and lifetime (Tolman's growth).")
print("Our current universe is merely the N-th iteration of this process,")
print("grown massive and long-lived over hundreds of previous fast cycles.")
print("=====================================================================")
