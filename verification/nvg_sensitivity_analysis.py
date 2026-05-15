import numpy as np
import math

# Re-implementing the EOS directly to allow dynamic M_Omega_0 modification
M_N = 939.0
n_0 = 0.16

# EOS Parameters (best point from full EOS)
kappa_1 = 0.25
kappa_2 = 0.80
C_v_n0 = 230.0  # Approx to give correct binding
alpha_v = 4.0
nu_v = 2.0
C_rho_n0 = 150.0

# Phase transition parameters
n_trans = 2.0
delta_eps = 350.0

def get_nvg_core_eos(n_B_fm3, M_Omega_0):
    if n_B_fm3 <= 0.0:
        return 0.0, 0.0
        
    M_current_0 = M_N - M_Omega_0
    
    n_B = n_B_fm3
    x = n_B / n_0
    
    # 1. Mass modification
    m_omega_nB = M_Omega_0 * (1.0 + kappa_2 * x)**(-kappa_1 / kappa_2)
    # Linear chiral drop for current mass (low-density approx)
    sigma_piN = 44.0
    f_pi = 93.0
    m_pi = 140.0
    # term = sigma_piN / (f_pi^2 m_pi^2) * n_B * (hbar c)^3
    # Approximated by empirical drop
    m_current_nB = M_current_0 * max(0.0, 1.0 - 0.2 * x) 
    
    m_eff = m_current_nB + m_omega_nB
    
    # 2. Kinetic energy (Fermi gas)
    k_F = (1.5 * math.pi**2 * n_B)**(1/3.0)
    # converting to fm^-1: k_F is in fm^-1
    hbarc = 197.3
    
    # Integrate E_kin
    # eps_kin = \frac{1}{\pi^2} \int_0^{k_F} k^2 \sqrt{k^2 + (m_{eff}/hbarc)^2} dk * hbarc
    # Analytical:
    m_eff_fm = m_eff / hbarc
    x_F = k_F / m_eff_fm
    
    if x_F > 0.01:
        eps_kin = (m_eff**4 / (8 * math.pi**2 * hbarc**3)) * (
            x_F * math.sqrt(1 + x_F**2) * (2*x_F**2 + 1) - math.asinh(x_F)
        )
        P_kin = (m_eff**4 / (24 * math.pi**2 * hbarc**3)) * (
            x_F * math.sqrt(1 + x_F**2) * (2*x_F**2 - 3) + 3 * math.asinh(x_F)
        )
    else:
        eps_kin = m_eff * n_B
        P_kin = 0.0
        
    # Potential scalar term
    v_pot = (m_eff - M_N) * n_B * 0.5 
    
    # Vector repulsion (saturated)
    v_vector = C_v_n0 * x**2 / (1.0 + alpha_v * x**nu_v)
    eps_v = v_vector * n_B
    P_v = eps_v * (1.0 + nu_v * alpha_v * x**nu_v) / (1.0 + alpha_v * x**nu_v)
    
    eps = eps_kin + v_pot + eps_v
    P = P_kin + P_v
    
    return eps, P

class UnifiedEOS:
    def __init__(self, M_Omega_0):
        self.n_trans = n_trans * n_0
        self.delta_eps = delta_eps
        
        n_grid = np.logspace(-4, 1.5, 500) * n_0
        self.eps_arr = []
        self.p_arr = []
        
        eps_t, P_t = get_nvg_core_eos(self.n_trans, M_Omega_0)
        
        for n in n_grid:
            if n <= self.n_trans:
                eps, P = get_nvg_core_eos(n, M_Omega_0)
                self.eps_arr.append(eps)
                self.p_arr.append(P)
            else:
                eps_quark_start = eps_t + self.delta_eps
                c2 = 1.0/3.0
                eps = eps_quark_start * (n / self.n_trans)**(1.0 + c2)
                P = P_t + c2 * (eps - eps_quark_start)
                self.eps_arr.append(eps)
                self.p_arr.append(P)
                
        self.eps_arr = np.array(self.eps_arr)
        self.p_arr = np.array(self.p_arr)
        
    def get_eps(self, P):
        if P <= self.p_arr[0]:
            return 0.0
        if P >= self.p_arr[-1]:
            return self.eps_arr[-1]
        return np.interp(P, self.p_arr, self.eps_arr)

def solve_tov(eos, P_center):
    def rk4_step(r, m, p, dr):
        if p <= 0: return m, 0
        eps = eos.get_eps(p)
        def dp_dr(r_val, m_val, p_val, eps_val):
            if r_val < 1e-10: return 0.0, 0.0
            k = 1.3234e-6
            eps_k, p_k = eps_val * k, p_val * k
            m_k = m_val * 1.4766
            dm_dr_k = 4.0 * math.pi * r_val**2 * eps_k
            num = (eps_k + p_k) * (m_k + 4.0 * math.pi * r_val**3 * p_k)
            den = r_val * (r_val - 2.0 * m_k)
            dp_dr_k = -num / den if den > 0 else 0.0
            return dm_dr_k / 1.4766, dp_dr_k / k

        k1_m, k1_p = dp_dr(r, m, p, eps)
        eps_mid = eos.get_eps(p + 0.5 * dr * k1_p)
        k2_m, k2_p = dp_dr(r + 0.5*dr, m + 0.5*dr*k1_m, p + 0.5*dr*k1_p, eps_mid)
        eps_mid = eos.get_eps(p + 0.5 * dr * k2_p)
        k3_m, k3_p = dp_dr(r + 0.5*dr, m + 0.5*dr*k2_m, p + 0.5*dr*k2_p, eps_mid)
        eps_end = eos.get_eps(p + dr * k3_p)
        k4_m, k4_p = dp_dr(r + dr, m + dr*k3_m, p + dr*k3_p, eps_end)
        
        return m + (dr/6.0) * (k1_m + 2*k2_m + 2*k3_m + k4_m), p + (dr/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        
    r = 1e-6
    m = 0.0
    p = P_center
    dr = 0.05
    while p > 1e-4 and r < 50.0:
        m, p = rk4_step(r, m, p, dr)
        r += dr
    return m, r

def main():
    print("="*60)
    print("SENSITIVITY ANALYSIS: EOS vs Lattice QCD Sigma Terms")
    print("="*60)
    
    # Baseline
    sigma_piN_base = 44.0
    sigma_sN_base = 30.0
    sigma_heavy_base = 6.0
    
    def calc_M_Omega(piN, sN):
        sigma_tot = piN + sN + sigma_heavy_base
        f_Omega = 1.0 - (sigma_tot / M_N)
        return f_Omega * M_N

    # Calculate points
    scenarios = [
        ("Lower Bound (-1 sigma)", 44 - 3, 30 - 7),
        ("Baseline (Central)", 44, 30),
        ("Upper Bound (+1 sigma)", 44 + 3, 30 + 7)
    ]
    
    print(f"{'Scenario':<25} | {'σ_πN':<6} | {'σ_sN':<6} | {'M_Ω,0 (MeV)':<11} | {'M_max (M_sun)':<13}")
    print("-" * 70)
    
    for name, piN, sN in scenarios:
        M_Omega = calc_M_Omega(piN, sN)
        eos = UnifiedEOS(M_Omega)
        
        # Scan pressures to find M_max
        pressures = np.logspace(1.5, 3.0, 15)
        M_max = 0.0
        for Pc in pressures:
            M, R = solve_tov(eos, Pc)
            if M > M_max:
                M_max = M
                
        print(f"{name:<25} | {piN:<6.1f} | {sN:<6.1f} | {M_Omega:<11.1f} | {M_max:<13.3f}")

if __name__ == "__main__":
    main()
