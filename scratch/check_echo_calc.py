import math
import numpy as np

G_cgs = 6.674e-8
c_cgs = 2.998e10
M_sun = 1.989e33
M_Omega_0 = 859.0
hbar_c = 197.327
eps_max = M_Omega_0**4 / hbar_c**3
MeV_fm3_to_gcm3 = 1.7827e12
rho_c = eps_max * MeV_fm3_to_gcm3

def get_bh_parameters(M_bh_solar):
    M_cgs = M_bh_solar * M_sun
    r_0_cgs = (3.0 * M_cgs / (4.0 * math.pi * rho_c))**(1/3.0)
    R_g_cgs = 2.0 * G_cgs * M_cgs / c_cgs**2
    return r_0_cgs, R_g_cgs

def solve_roots(R_g, r_0):
    coeffs = [1.0, -R_g, 0.0, r_0**3]
    roots = np.roots(coeffs)
    roots = sorted(roots, key=lambda x: x.real)
    return roots[0].real, roots[1].real, roots[2].real

def calculate_echo_delay_r0(M_bh_solar):
    r_0, R_g = get_bh_parameters(M_bh_solar)
    r1, r2, r3 = solve_roots(R_g, r_0)
    
    A1 = (r1**3 + r_0**3) / ((r1 - r2) * (r1 - r3))
    A2 = (r2**3 + r_0**3) / ((r2 - r1) * (r2 - r3))
    A3 = (r3**3 + r_0**3) / ((r3 - r1) * (r3 - r2))
    
    # We use r_0 as the cutoff
    delta = r_0
    
    # Exterior round-trip time F_ext - F_r3_plus
    term_const = (1.5*R_g - r3) + A1 * math.log((1.5*R_g - r1)/(r3 - r1)) + A2 * math.log((1.5*R_g - r2)/(r3 - r2)) + A3 * math.log(1.5*R_g - r3)
    diff = term_const - A3 * math.log(delta)
    dt_echo = 2.0 * diff / c_cgs
    return dt_echo

print("Schwarzschild delay with delta = r_0:")
print(f"M = 65.0 M_sun: {calculate_echo_delay_r0(65.0):.6f} s")
print(f"M = 3.0 M_sun: {calculate_echo_delay_r0(3.0):.6f} s")
