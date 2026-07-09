import numpy as np

import os

# Physical constants
G_c2 = 1.476  # km/M_sun
rho_c_mev_fm3 = 7.09e4
rho_c_geom = rho_c_mev_fm3 * 1.323e-6  # km^-2

def calc_mcrit_l(rho):
    # Parameterization 1: constant l = r_c
    l = np.sqrt(3 / (8 * np.pi * rho))
    M_crit_km = (3 * np.sqrt(3) / 4) * l
    r_crit_km = np.sqrt(3) * l
    return M_crit_km / G_c2, r_crit_km

def calc_mcrit_r0(rho):
    # Parameterization 2: mass-dependent r_0 = (3M / 4 pi rho)^1/3
    alpha = 3 / (4 * np.pi * rho)
    M_crit_km = (3 * np.sqrt(3) / (4 * np.sqrt(2))) * np.sqrt(alpha)
    r_crit_km = np.sqrt(1.5 * alpha)
    return M_crit_km / G_c2, r_crit_km

print("=== Critical Mass Analysis ===")
m1, r1 = calc_mcrit_l(rho_c_geom)
m2, r2 = calc_mcrit_r0(rho_c_geom)

print(f"Param 1 (l = r_c): M_crit = {m1:.4f} M_sun, r_crit = {r1:.4f} km")
print(f"Param 2 (r_0):     M_crit = {m2:.4f} M_sun, r_crit = {r2:.4f} km")
print(f"Difference:        {abs(m1-m2):.2e} M_sun")

# Uncertainty +/- 10%
print("\n=== Uncertainty (+/- 10% on rho_c) ===")
rho_low = rho_c_geom * 0.9
rho_high = rho_c_geom * 1.1

m_low, _ = calc_mcrit_l(rho_low)
m_high, _ = calc_mcrit_l(rho_high)
print(f"rho_c -10%: M_crit = {m_low:.4f} M_sun")
print(f"rho_c +10%: M_crit = {m_high:.4f} M_sun")
print(f"Range: {m_high:.4f} to {m_low:.4f} M_sun")

import csv

# Check GWTC events
gwtc_path = "data/gwtc_events.csv"
if os.path.exists(gwtc_path):
    print("\n=== GWTC Catalog Check ===")
    with open(gwtc_path, 'r') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        m1_col = next((c for c in cols if 'mass_1_source' in c and 'lower' not in c and 'upper' not in c), None)
        m2_col = next((c for c in cols if 'mass_2_source' in c and 'lower' not in c and 'upper' not in c), None)
        
        if m1_col and m2_col:
            count = 0
            for row in reader:
                try:
                    m1_val = float(row[m1_col])
                    m2_val = float(row[m2_col])
                    if m1_val < 1.4 or m2_val < 1.4:
                        print(f"  - {row['id']}: M1 = {m1_val:.2f}, M2 = {m2_val:.2f}")
                        count += 1
                except ValueError:
                    pass
            print(f"Found {count} events with component mass < 1.4 M_sun.")
        else:
            print("Could not find mass columns in GWTC csv. Columns:", cols)
else:
    print(f"\nGWTC file not found at {gwtc_path}")
