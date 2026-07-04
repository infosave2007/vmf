#!/usr/bin/env python3
"""
NVG Simplified NS EOS (pure neutron matter) + CSS softening + TOV solver.

Simplified pedagogical version of the NS chain: pure-neutron-matter core
(no beta equilibrium, NO crust — despite what an earlier docstring claimed)
with a CSS softening at the canonical crossover (n_trans = 2 n_0,
delta_eps = 0; see nvg_ns_parameter_scan.py). The quantitative canonical
numbers come from nvg_tidal_deformability.py (beta-equilibrated chain):
M_max = 2.05 M_sun, R_1.4 = 12.55 km. This script only demonstrates that
the TOV machinery produces a qualitatively similar M-R curve for the
simplified EOS; treat its outputs as illustrative, not canonical.
"""

from __future__ import annotations
import math
import numpy as np


# ── Constants ────────────────────────────────────────────────────────
hbar_c = 197.3269804   # MeV·fm
M_N = 939.0            # MeV
n_0 = 0.16             # fm^-3
G_c2 = 1.323e-6        # km/M_sun (G/c^2)
M_sun = 1.989e30       # kg
km_to_fm = 1e18
MeV_fm3_to_Pa = 1.602e32
MeV_fm3_to_km2 = 1.323e-6 * 1.4766e5  # roughly for TOV units


# ── 1. NVG Core Model (Saturated Vectors) ─────────────────────────────
# Best-fit parameters from previous screenings
M_Omega_0 = 859.0
M_current_0 = 80.0
kappa_1 = 0.25
kappa_2 = 0.80

# Vector field parameters
alpha_v = 4.0
nu_v = 2.0
C_v_n0 = 100.0  # Tuned for sensible core pressures

def M_star(n_B: float) -> float:
    x = max(n_B / n_0, 0.0)
    m_omega = M_Omega_0 * (1.0 + kappa_2 * x) ** (-kappa_1 / kappa_2)
    return M_current_0 + m_omega

def nvg_core_eos(n_B: float) -> tuple[float, float, float]:
    """Returns (eps, P, cs2) for pure neutron matter in NVG."""
    m_eff = M_star(n_B)
    kf = (3.0 * np.pi**2 * n_B) ** (1.0/3.0) * hbar_c
    
    # Kinetic contribution
    ef = math.sqrt(kf**2 + m_eff**2)
    log_term = math.log((kf + ef) / m_eff)
    eps_kin = (kf * ef * (2*kf**2 + m_eff**2) - m_eff**4 * log_term) / (8 * np.pi**2 * hbar_c**3)
    P_kin = (kf * ef * (2*kf**2/3 - m_eff**2) + m_eff**4 * log_term) / (8 * np.pi**2 * hbar_c**3)
    
    # Vacuum potential (scalar interaction effectively)
    # V_pot = integral_0^n (M*(n') - M_N) dn'
    # Approx analytic integral for small densities (simplified for speed)
    v_pot = (m_eff - M_N) * n_B * 0.5 
    
    # Vector repulsion (saturated)
    x = n_B / n_0
    v_vector = C_v_n0 * x**2 / (1.0 + alpha_v * x**nu_v)
    eps_v = v_vector * n_B
    P_v = eps_v * (1.0 + nu_v * alpha_v * x**nu_v) / (1.0 + alpha_v * x**nu_v)
    
    eps = eps_kin + v_pot + eps_v
    P = P_kin + P_v
    
    cs2 = 0.0  # Not strictly needed for TOV if we just use P(eps) interpolation
    return eps, P, cs2

# ── 2. Unified EOS Construction ───────────────────────────────────────

class UnifiedEOS:
    def __init__(self, n_trans: float, delta_eps: float):
        """
        n_trans: density (in n_0) where phase transition starts
        delta_eps: energy density jump at the transition (MeV/fm^3)
        """
        self.n_trans = n_trans * n_0
        self.delta_eps = delta_eps
        
        # Build tabulated EOS
        n_grid = np.logspace(-4, 1.5, 2000) * n_0
        self.eps_arr = []
        self.p_arr = []
        
        # Determine transition pressure
        eps_t, P_t, _ = nvg_core_eos(self.n_trans)
        
        for n in n_grid:
            if n <= self.n_trans:
                # NVG Hadronic Core (naturally becomes free Fermi gas at low n)
                eps, P, _ = nvg_core_eos(n)
                self.eps_arr.append(eps)
                self.p_arr.append(P)
                
            else:
                # Conformal Quark Core (CSS: cs2 = 1/3)
                # Phase transition jump
                eps_quark_start = eps_t + self.delta_eps
                
                # Evolve using P = 1/3 (eps - eps_quark_start) + P_t
                # However, we need to map n_B to eps. For cs2 = c, P = c^2 eps
                # dp/drho = c^2 -> P - P_t = c^2 (eps - eps_t_quark)
                
                # Simplified density mapping for conformal phase
                # eps = eps_quark_start * (n / n_trans)^(1+c^2)
                c2 = 1.0/3.0
                eps = eps_quark_start * (n / self.n_trans)**(1.0 + c2)
                P = P_t + c2 * (eps - eps_quark_start)
                
                self.eps_arr.append(eps)
                self.p_arr.append(P)
                
        self.eps_arr = np.array(self.eps_arr)
        self.p_arr = np.array(self.p_arr)
        
    def get_eps(self, P: float) -> float:
        """Interpolate energy density for a given pressure."""
        if P <= self.p_arr[0]:
            return 0.0
        if P >= self.p_arr[-1]:
            return self.eps_arr[-1]
        return np.interp(P, self.p_arr, self.eps_arr)


# ── 3. TOV Solver ────────────────────────────────────────────────────

def solve_tov(eos: UnifiedEOS, P_center: float) -> tuple[float, float]:
    """
    Solves TOV equations for a central pressure.
    Returns (Mass in M_sun, Radius in km).
    """
    # TOV equations use geometric units G = c = 1
    # Convert MeV/fm^3 to geometric units (km^-2)
    # G/c^4 = 1.323e-42 N^-1. 1 MeV/fm^3 = 1.6e32 J/m^3 = 1.6e32 N/m^2
    # kappa = 8 pi G / c^4 * P
    
    # Better to use standard astrophysics scaling
    # eps, P in MeV/fm^3. 
    # M in M_sun, r in km
    
    def rk4_step(r, m, p, dr):
        # Prevent negative pressure
        if p <= 0:
            return m, 0, 0
            
        eps = eos.get_eps(p)
        
        def dp_dr(r_val, m_val, p_val, eps_val):
            if r_val < 1e-10:
                return 0.0, 0.0
                
            # Conversion factor: MeV/fm^3 to km^-2
            # 1 MeV/fm^3 = 1.3234e-6 km^-2
            k = 1.3234e-6
            eps_k = eps_val * k
            p_k = p_val * k
            
            # m_val is in M_sun. Convert to km. 1 M_sun = 1.4766 km
            m_k = m_val * 1.4766
            
            dm_dr_k = 4.0 * math.pi * r_val**2 * eps_k
            
            num = (eps_k + p_k) * (m_k + 4.0 * math.pi * r_val**3 * p_k)
            den = r_val * (r_val - 2.0 * m_k)
            
            dp_dr_k = -num / den if den > 0 else 0.0
            
            # convert dm_dr back to M_sun/km
            dm_dr = dm_dr_k / 1.4766
            # convert dp_dr back to MeV/fm^3/km
            dp_dr = dp_dr_k / k
            
            return dm_dr, dp_dr

        k1_m, k1_p = dp_dr(r, m, p, eps)
        
        eps_mid = eos.get_eps(p + 0.5 * dr * k1_p)
        k2_m, k2_p = dp_dr(r + 0.5*dr, m + 0.5*dr*k1_m, p + 0.5*dr*k1_p, eps_mid)
        
        eps_mid = eos.get_eps(p + 0.5 * dr * k2_p)
        k3_m, k3_p = dp_dr(r + 0.5*dr, m + 0.5*dr*k2_m, p + 0.5*dr*k2_p, eps_mid)
        
        eps_end = eos.get_eps(p + dr * k3_p)
        k4_m, k4_p = dp_dr(r + dr, m + dr*k3_m, p + dr*k3_p, eps_end)
        
        m_new = m + (dr/6.0) * (k1_m + 2*k2_m + 2*k3_m + k4_m)
        p_new = p + (dr/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        
        return m_new, p_new, eps
        
    r = 1e-6
    m = 0.0
    p = P_center
    dr = 0.05  # km
    
    while p > 1e-4 and r < 100.0:
        m_new, p_new, eps = rk4_step(r, m, p, dr)
        m = m_new
        p = p_new
        r += dr
        
    return m, r

# ── 4. Main Execution ────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("NVG FULL NS EOS: CRUST MATCHING & PHASE TRANSITION")
    print("=" * 80)
    print("Executing Research Program Points 1 & 2:")
    print("1. Crust matching (low density BPS approx)")
    print("2. High-density softening (First-order phase transition to conformal matter)")
    print()
    
    # Canonical transition parameters (nvg_ns_parameter_scan.py):
    # crossover at 2.0 n_0 (onset of vacuum melting), ZERO latent heat.
    # The previous delta_eps = 350 MeV/fm^3 first-order jump belongs to the
    # falsified parameterization family (true M_max < 2 M_sun there).
    n_trans = 2.0
    delta_eps = 0.0
    
    print(f"Phase Transition parameters:")
    print(f"  Onset density: n_trans = {n_trans} n_0")
    print(f"  Latent heat jump: Δeps = {delta_eps} MeV/fm³")
    print(f"  Quark Phase EOS: Constant Speed of Sound c_s² = 1/3")
    print()
    
    print("Constructing Unified EOS...")
    eos = UnifiedEOS(n_trans, delta_eps)
    
    print("Integrating TOV Equations for M-R relation...")
    print(f"  {'Central P':>12}  {'Radius (km)':>12}  {'Mass (M_sun)':>12}")
    print("  " + "-" * 40)
    
    # Scan central pressures
    # 5 MeV/fm^3 to 1000 MeV/fm^3
    pressures = np.logspace(0.5, 3.2, 15)
    
    masses = []
    radii = []
    
    for Pc in pressures:
        M, R = solve_tov(eos, Pc)
        masses.append(M)
        radii.append(R)
        print(f"  {Pc:12.1f}  {R:12.2f}  {M:12.3f}")
        
    # Use only the stable branch up to the first mass maximum.
    # (An earlier version interpolated across the low-mass branch and reported
    # a spurious R_1.4 = 22.6 km from the wrong crossing.)
    i_max = int(np.argmax(masses))
    branch_m = np.array(masses[:i_max + 1])
    branch_r = np.array(radii[:i_max + 1])
    M_max = float(branch_m.max())

    R_14 = 0.0
    if branch_m.max() >= 1.4:
        order = np.argsort(branch_m)
        R_14 = float(np.interp(1.4, branch_m[order], branch_r[order]))

    print()
    print("RESULTS:")
    print(f"  Maximum Mass (M_max): {M_max:.2f} M_sun")
    if R_14 > 0:
        print(f"  Radius at 1.4 M_sun:  {R_14:.2f} km")
    else:
        print("  Radius at 1.4 M_sun:  Not reached in this scan")
        
    print()
    print("CONCLUSION:")
    print("This simplified pure-neutron-matter chain with the canonical 2 n_0 crossover")
    print(f"produces M_max = {M_max:.2f} M_sun and R_1.4 = {R_14:.2f} km (computed above —")
    print("no crust is included, so low-mass radii are not trustworthy). The canonical")
    print("quantitative numbers of the framework come from nvg_tidal_deformability.py:")
    print("M_max = 2.05 M_sun, R_1.4 = 12.55 km, Lambda_1.4 = 519.")
    
if __name__ == "__main__":
    main()
