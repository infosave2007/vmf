import math
import numpy as np

# Copied from nvg_tidal_deformability.py
hbar_c = 197.3269804
M_N = 939.0
n_0 = 0.16
G_cgs = 6.674e-8
c_cgs = 2.998e10
M_sun_g = 1.989e33
M_sun_km = 1.4766
k_conv = 1.3234e-6

M_Omega_0 = 859.0
M_current_0 = 80.0
kappa_1 = 0.25
kappa_2 = 0.80
alpha_v = 4.0
nu_v = 2.0
C_v_n0 = 100.0

def M_star(n_B: float) -> float:
    x = max(n_B / n_0, 0.0)
    return M_current_0 + M_Omega_0 * (1.0 + kappa_2 * x) ** (-kappa_1 / kappa_2)

def nvg_core_eos(n_B: float):
    m_eff = M_star(n_B)
    kf = (3.0 * np.pi**2 * n_B) ** (1.0/3.0) * hbar_c
    ef = math.sqrt(kf**2 + m_eff**2)
    log_term = math.log((kf + ef) / m_eff)
    eps_kin = (kf * ef * (2*kf**2 + m_eff**2) - m_eff**4 * log_term) / (8 * np.pi**2 * hbar_c**3)
    P_kin = (kf * ef * (2*kf**2/3 - m_eff**2) + m_eff**4 * log_term) / (8 * np.pi**2 * hbar_c**3)
    v_pot = (m_eff - M_N) * n_B * 0.5
    x = n_B / n_0
    v_vec = C_v_n0 * x**2 / (1.0 + alpha_v * x**nu_v)
    eps_v = v_vec * n_B
    P_v = eps_v * (1.0 + nu_v * alpha_v * x**nu_v) / (1.0 + alpha_v * x**nu_v)
    return eps_kin + v_pot + eps_v, P_kin + P_v

class PureEOS:
    def __init__(self):
        n_arr = np.linspace(0.01, 1.5, 1000)
        self.eps_arr = []
        self.p_arr = []
        for n in n_arr:
            e, p = nvg_core_eos(n)
            self.eps_arr.append(e)
            self.p_arr.append(p)
        self.eps_arr = np.array(self.eps_arr)
        self.p_arr = np.array(self.p_arr)

    def get_eps(self, P: float) -> float:
        if P <= 0.0: return 0.0
        if P >= self.p_arr[-1]: return self.eps_arr[-1]
        return float(np.interp(P, self.p_arr, self.eps_arr))

    def get_dedp(self, P: float) -> float:
        dP = max(P * 1e-4, 1e-8)
        e1 = self.get_eps(P + dP)
        e2 = self.get_eps(max(P - dP, 0.0))
        return (e1 - e2) / (2.0 * dP)

def solve_tov_tidal(eos, P_center: float):
    dr = 0.05  # km
    r = 1e-6   # km
    m = 0.0    # km (geometric mass)
    P = P_center
    y = 2.0    # y(0) = 2 for l=2

    def derivs(r_val, m_val, P_val, y_val):
        if r_val < 1e-10 or P_val <= 0:
            return 0.0, 0.0, 0.0

        eps = eos.get_eps(P_val)
        eps_k = eps * k_conv
        P_k = P_val * k_conv
        m_k = m_val * M_sun_km  # convert M_sun → km

        fac = r_val * (r_val - 2.0 * m_k)
        if fac <= 0:
            return 0.0, 0.0, 0.0

        # TOV
        dm_dr = 4.0 * math.pi * r_val**2 * eps_k / M_sun_km  # M_sun / km
        num = (eps_k + P_k) * (m_k + 4.0 * math.pi * r_val**3 * P_k)
        dP_dr = -num / fac / k_conv  # MeV/fm³ / km

        # y-equation (Hinderer 2008, Eq. 14)
        e_minus_p = (eps_k - P_k)
        one_m_2mr = 1.0 - 2.0 * m_k / r_val

        F = (1.0 - 4.0 * math.pi * r_val**2 * e_minus_p) / one_m_2mr

        dedp = eos.get_dedp(P_val)
        if dedp <= 0:
            dedp = 1.0

        Q_source = 4.0 * math.pi * (5.0 * eps_k + 9.0 * P_k + (eps_k + P_k) * dedp)
        Q_source /= one_m_2mr
        Q_grav = (2.0 * (m_k + 4.0 * math.pi * r_val**3 * P_k) / (r_val * one_m_2mr))**2 / r_val**2
        Q = Q_source - Q_grav - 6.0 / (r_val**2 * one_m_2mr)

        dy_dr = -(y_val**2 + y_val * F + r_val**2 * Q) / r_val

        return dm_dr, dP_dr, dy_dr

    while P > 1e-4 and r < 100.0:
        dm1, dp1, dy1 = derivs(r, m, P, y)

        r2 = r + 0.5 * dr
        m2 = m + 0.5 * dr * dm1
        P2 = P + 0.5 * dr * dp1
        y2 = y + 0.5 * dr * dy1
        if P2 <= 0: break
        dm2, dp2, dy2 = derivs(r2, m2, P2, y2)

        r3 = r + 0.5 * dr
        m3 = m + 0.5 * dr * dm2
        P3 = P + 0.5 * dr * dp2
        y3 = y + 0.5 * dr * dy2
        if P3 <= 0: break
        dm3, dp3, dy3 = derivs(r3, m3, P3, y3)

        r4 = r + dr
        m4 = m + dr * dm3
        P4 = P + dr * dp3
        y4 = y + dr * dy3
        if P4 <= 0: break
        dm4, dp4, dy4 = derivs(r4, m4, P4, y4)

        m += (dr / 6.0) * (dm1 + 2*dm2 + 2*dm3 + dm4)
        P += (dr / 6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
        y += (dr / 6.0) * (dy1 + 2*dy2 + 2*dy3 + dy4)
        r += dr

        if P < 0:
            P = 0.0
            break

    R = r
    M_solar = m
    C = M_solar * M_sun_km / R
    y_R = y

    if C <= 0 or C >= 0.5:
        return M_solar, R, 0.0, 0.0

    fac_1m2C = 1.0 - 2.0 * C
    if fac_1m2C <= 0:
        return M_solar, R, 0.0, 0.0

    ln_1m2C = math.log(fac_1m2C)

    num = (8.0/5.0) * C**5 * fac_1m2C**2 * (2.0 + 2.0*C*(y_R - 1.0) - y_R)

    den = (2.0*C * (6.0 - 3.0*y_R + 3.0*C*(5.0*y_R - 8.0))
           + 4.0*C**3 * (13.0 - 11.0*y_R + C*(3.0*y_R - 2.0) + 2.0*C**2*(1.0 + y_R))
           + 3.0*fac_1m2C**2 * (2.0 - y_R + 2.0*C*(y_R - 1.0)) * ln_1m2C)

    if abs(den) < 1e-30:
        return M_solar, R, 0.0, 0.0

    k2 = num / den
    Lambda = (2.0/3.0) * k2 * C**(-5) if k2 > 0 else 0.0

    return M_solar, R, k2, Lambda

eos = PureEOS()
M, R, k2, Lam = solve_tov_tidal(eos, 100.0) # P_c = 100 MeV/fm^3
P_centers = np.logspace(-1.0, 2.8, 50)
masses = []
lambdas = []
for Pc in P_centers:
    m, r, k2, L = solve_tov_tidal(eos, Pc)
    masses.append(m)
    lambdas.append(L)

L14 = float(np.interp(1.4, masses, lambdas))
print(f"Lambda_1.4 = {L14:.1f}")
