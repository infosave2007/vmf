from decimal import Decimal, getcontext
import numpy as np

# Set arbitrary precision for catastrophic cancellation
getcontext().prec = 100

G_val = Decimal('6.67430e-11')
c_val = Decimal('2.99792458e8')
M_sun_val = Decimal('1.98847e30')
hbar_val = Decimal('1.054571817e-34')
l_val = Decimal('1.128e3')

def compute_precise(M_solar_float):
    M_solar = Decimal(str(M_solar_float))
    M = M_solar * M_sun_val
    rg = Decimal('2') * G_val * M / (c_val**2)
    epsilon = l_val / rg
    
    # 5th order expansion for roots to ensure extreme precision
    # x_in = e + 1/2 e^2 + 5/8 e^3 + 7/8 e^4 + 21/16 e^5
    x_in = epsilon + Decimal('0.5')*epsilon**2 + Decimal('0.625')*epsilon**3 + Decimal('0.875')*epsilon**4 + Decimal('1.3125')*epsilon**5
    r_in = rg * x_in
    
    # x_out = 1 - e^2 - 2 e^4
    x_out = Decimal('1') - epsilon**2 - Decimal('2')*epsilon**4 - Decimal('7')*epsilon**6
    r_out = rg * x_out
    
    pi = Decimal(np.pi)
    coeff = pi * c_val**3 / (G_val * hbar_val)
    
    S_schw = coeff * rg**2
    S_out = coeff * r_out**2
    S_in_local = coeff * r_in**2
    
    # Deficit
    delta_S = S_schw - (S_out + S_in_local)
    
    # Relative
    rel_dev_exact = (delta_S / S_in_local) - Decimal('1')
    
    # Analytical prediction: -2 * epsilon
    rel_dev_analytical = Decimal('-2') * epsilon
    
    return {
        'M': M_solar_float,
        'r_in': float(r_in) / 1e3,
        'rel_dev_exact': float(rel_dev_exact) * 100,
        'rel_dev_analytical': float(rel_dev_analytical) * 100
    }

masses = [10, 65, 1000, 1e6, 1e9]
print(f"{'Mass (M_sun)':<15} | {'r_in (km)':<15} | {'Delta S / S_in - 1 (%)':<25} | {'Analytical Limit (%)':<25}")
print("-" * 85)
for m in masses:
    res = compute_precise(m)
    print(f"{res['M']:<15.0e} | {res['r_in']:<15.6f} | {res['rel_dev_exact']:<25.6f} | {res['rel_dev_analytical']:<25.6f}")

