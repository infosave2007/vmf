import math

# Constants
c = 2.99792458e10      # cm/s
G = 6.67430e-8         # cm^3/g/s^2
M_sun = 1.98847e33     # g
Mpc_to_cm = 3.08567758e24

# 1. Start with H_0 = 72.8 km/s/Mpc
H_0_km_s_Mpc = 72.8
H_0_s = H_0_km_s_Mpc * 1e5 / Mpc_to_cm
print(f"H_0 = {H_0_km_s_Mpc} km/s/Mpc = {H_0_s:.3e} s^-1")
R_H0 = c / H_0_s

# NVG calibrated N_e
N_e = 53.08
r_c_cm = R_H0 / math.exp(N_e)
r_c_km = r_c_cm / 1e5
print(f"r_c = {r_c_km:.4f} km")

M_crit_g = (3.0 * math.sqrt(3.0) / 4.0) * (c**2 * r_c_cm / G)
M_crit_msun = M_crit_g / M_sun
print(f"M_crit = {M_crit_msun:.4f} M_sun")
