import numpy as np
from scipy.integrate import odeint

def coupled_system(y, a, beta_melt):
    rho_m, rho_W = y
    H = np.sqrt(abs(rho_m + rho_W))
    Q = beta_melt * H * rho_m
    K = (beta_melt / 6.0) * rho_m
    U = rho_W - K
    w_W = (K - U) / (K + U) if (K + U) > 0 else -1.0
    drho_m_da = Q / (a * H) - 3.0 * rho_m / a
    drho_W_da = -Q / (a * H) - 3.0 * (1.0 + w_W) * rho_W / a
    return [drho_m_da, drho_W_da]

def derive_w0_wa(beta_melt):
    Omega_m0 = 0.315
    Omega_W0 = 1.0 - Omega_m0
    y0 = [Omega_m0, Omega_W0]
    a_vals = np.linspace(1.0, 0.4, 100)
    sol = odeint(coupled_system, y0, a_vals, args=(beta_melt,))
    
    rho_m_vals = sol[:, 0]
    rho_W_vals = sol[:, 1]
    rho_tot = rho_m_vals + rho_W_vals
    
    rho_m_std = Omega_m0 * a_vals**-3
    rho_DE_eff = rho_tot - rho_m_std
    
    H_std = np.sqrt(Omega_m0 * a_vals**-3 + Omega_W0)
    rho_DE_eff_clean = rho_DE_eff
    
    w_eff = -1.0 - (1.0/3.0) * (np.gradient(rho_DE_eff_clean, a_vals) * a_vals / rho_DE_eff_clean)
    
    z_vals = 1.0 / a_vals - 1.0
    def cpl_model(a, w0, wa):
        return w0 + wa * (1 - a)
    
    from scipy.optimize import curve_fit
    popt, _ = curve_fit(cpl_model, a_vals, w_eff)
    w0, wa = popt
    return w0, wa

print(f"beta=0.12 -> {derive_w0_wa(0.12)}")
print(f"beta=0.05 -> {derive_w0_wa(0.05)}")
print(f"beta=0.20 -> {derive_w0_wa(0.20)}")
print(f"beta=0.30 -> {derive_w0_wa(0.30)}")
print(f"beta=1.00 -> {derive_w0_wa(1.00)}")
