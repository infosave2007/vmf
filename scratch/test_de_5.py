import numpy as np
from scipy.integrate import odeint

Omega_m0 = 0.315
Omega_W0 = 1.0 - Omega_m0
beta = 0.12

def coupled_system(y, a, beta):
    rho_m, rho_W = y
    H = np.sqrt(abs(rho_m + rho_W))
    
    Q = beta * H * rho_m
    
    # K is driven by the melting coupling in the tracking limit: 6HK = Q
    # actually if 6HK = Q => K = Q / 6H = beta * rho_m / 6
    # But Q is energy added to W-field? No, Q is extracted from matter.
    # W-field EOM: \dot{rho}_W + 3H(2K) = -Q
    # If the potential U is extremely flat, \dot{rho}_W = \dot{K} + \dot{U} ~ \dot{K}
    # If the field is slowly rolling, U is dominant.
    # Let's dynamically calculate w_W(a) = (K - U) / (K + U)
    # where K(a) = beta * rho_m(a) / 6.0
    K = (beta / 6.0) * rho_m
    U = rho_W - K
    w_W = (K - U) / (K + U)
    
    drho_m_da = Q / (a * H) - 3.0 * rho_m / a
    drho_W_da = -Q / (a * H) - 3.0 * (1.0 + w_W) * rho_W / a
    
    return [drho_m_da, drho_W_da]

y0 = [Omega_m0, Omega_W0]
a_vals = np.linspace(1.0, 0.4, 100)

sol = odeint(coupled_system, y0, a_vals, args=(beta,))

rho_m_vals = sol[:, 0]
rho_W_vals = sol[:, 1]
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
