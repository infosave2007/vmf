import numpy as np
from scipy.integrate import odeint

Omega_m0 = 0.315
Omega_W0 = 1.0 - Omega_m0
beta = 0.12

def sys_a(y, a, beta, U0):
    rho_m, K = y
    
    rho_W = K + U0
    H = np.sqrt(rho_m + rho_W)
    
    Q = beta * H * rho_m
    
    drho_m_da = Q / (a * H) - 3.0 * rho_m / a
    dK_da = - Q / (a * H) - 6.0 * K / a
    
    return [drho_m_da, dK_da]

# Find U0 and K0. We know K0 + U0 = Omega_W0
# If w_W0 = -0.9, then (K0 - U0)/(K0 + U0) = -0.9
# K0 - U0 = -0.9 Omega_W0
# 2 K0 = 0.1 Omega_W0 => K0 = 0.05 Omega_W0
# U0 = 0.95 Omega_W0
K0 = 0.05 * Omega_W0
U0 = 0.95 * Omega_W0

y0 = [Omega_m0, K0]
a_vals = np.linspace(1.0, 0.4, 100)

sol = odeint(sys_a, y0, a_vals, args=(beta, U0))

rho_m_vals = sol[:, 0]
K_vals = sol[:, 1]
rho_W_vals = K_vals + U0
rho_tot = rho_m_vals + rho_W_vals
rho_m_std = Omega_m0 * a_vals**-3
rho_DE_eff = rho_tot - rho_m_std

w_eff_vals = np.zeros_like(a_vals)
for i in range(len(a_vals)-1):
    da = a_vals[i+1] - a_vals[i]
    drho = rho_DE_eff[i+1] - rho_DE_eff[i]
    w_eff_vals[i] = -1.0 - (a_vals[i]/3.0) * (1.0 / rho_DE_eff[i]) * (drho / da)
w_eff_vals[-1] = w_eff_vals[-2]

x_vals = 1.0 - a_vals
X = np.vstack([np.ones_like(x_vals), x_vals]).T
w0_pred, wa_pred = np.linalg.lstsq(X, w_eff_vals, rcond=None)[0]

print(f"w0 = {w0_pred:.4f}, wa = {wa_pred:.4f}")
