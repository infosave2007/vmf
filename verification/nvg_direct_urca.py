#!/usr/bin/env python3
"""
NVG Verification: Direct Urca Cooling Threshold from VMF
-------------------------------------------------------
Calculates the proton fraction x_p as a function of baryon density n_B under VMF
using self-consistent beta-equilibrium, checks it against the Lattimer threshold
x_p >= 1/(1 + (1 + x_e^(1/3))^3), and maps only the mass range covered by the
tabulated EOS.  The table does not supply an independent mass threshold claim.
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# Constants
hbar_c = 197.3269804  # MeV*fm
M_N = 939.0           # MeV
n_0 = 0.16            # fm^-3
f_pi = 93.0           # MeV
m_pi = 140.0          # MeV
m_e = 0.511           # MeV
m_mu = 105.658        # MeV
MeV_fm3_to_geo = 1.3234e-6
M_sun_km = 1.4766

# QCD Anchor / Vacuum calibration
sigma_piN = 44.0
sigma_sN = 30.0
sigma_heavy = 6.0
sigma_total = sigma_piN + sigma_sN + sigma_heavy
M_Omega_0 = M_N - sigma_total
M_current_0 = sigma_total

# VMF/NVG Model Parameters
k1 = 0.25
k2 = 0.80
c_s = 300.0           # Scalar coupling MeV*fm^3
c_rho = 90.0  # MeV*fm^3, standard isovector RMF value, cf. Typel et al. (2010) DD2
# This illustrative chain uses a vector-saturation value distinct from the
# canonical EOS.  It is not an independently calibrated stellar-mass result.
alpha_v = 0.05        # Vector saturation parameter of this illustrative chain
nu_v = 2.0            # Vector saturation power

def M_Omega(n_b):
    x = max(n_b / n_0, 0.0)
    return M_Omega_0 * (1.0 + k2 * x) ** (-k1 / k2)

def current_ratio_linear(n_b):
    correction = sigma_piN * n_b * hbar_c**3 / (f_pi**2 * m_pi**2)
    return max(1.0 - correction, 0.0)

def M_current(n_b):
    return M_current_0 * current_ratio_linear(n_b)

def M_base(n_b):
    return M_current(n_b) + M_Omega(n_b)

def kf_from_density(n_b):
    if n_b <= 0.0:
        return 0.0
    return (3.0 * np.pi**2 * n_b) ** (1.0 / 3.0) * hbar_c

def fermion_energy_density(n_b, mass):
    if n_b <= 0.0:
        return 0.0
    kf = kf_from_density(n_b)
    ef = math.sqrt(kf * kf + mass * mass)
    log_term = math.log((kf + ef) / mass)
    num = kf * ef * (2.0 * kf * kf + mass * mass) - mass**4 * log_term
    return num / (8.0 * np.pi**2 * hbar_c**3)

def fermion_scalar_density(n_b, mass):
    if n_b <= 0.0:
        return 0.0
    kf = kf_from_density(n_b)
    ef = math.sqrt(kf * kf + mass * mass)
    log_term = math.log((kf + ef) / mass)
    return mass * (kf * ef - mass * mass * log_term) / (2.0 * np.pi**2 * hbar_c**3)

def lepton_density_from_mu(mu_l, mass_l):
    if mu_l <= mass_l:
        return 0.0
    kf = math.sqrt(mu_l * mu_l - mass_l * mass_l)
    return kf**3 / (3.0 * np.pi**2 * hbar_c**3)

def solve_dirac_mass(n_n, n_p, m_ref):
    if c_s <= 0.0:
        return m_ref

    def residual(m_dirac):
        ns_total = fermion_scalar_density(n_n, m_dirac) + fermion_scalar_density(n_p, m_dirac)
        return m_dirac - m_ref + c_s * ns_total

    low = 5.0
    high = max(m_ref, low + 1.0)
    try:
        f_low = residual(low)
        f_high = residual(high)
        if not (np.isfinite(f_low) and np.isfinite(f_high)):
            return None
        if f_low * f_high > 0.0:
            return None
        return brentq(residual, low, high, xtol=1.0e-8, rtol=1.0e-8, maxiter=100)
    except Exception:
        return None

def beta_equilibrium_state(n_b):
    m_ref = M_base(n_b)

    def charge_residual(y_p):
        n_p = y_p * n_b
        n_n = (1.0 - y_p) * n_b
        m_dirac = solve_dirac_mass(n_n, n_p, m_ref)
        if m_dirac is None:
            raise ValueError("Dirac gap failed")
        mu_n_star = math.sqrt(kf_from_density(n_n) ** 2 + m_dirac**2)
        mu_p_star = math.sqrt(kf_from_density(n_p) ** 2 + m_dirac**2)
        delta_iso = 2.0 * c_rho * (n_n - n_p)
        mu_e = max(mu_n_star - mu_p_star + delta_iso, m_e)
        n_e = lepton_density_from_mu(mu_e, m_e)
        n_mu = lepton_density_from_mu(mu_e, m_mu)
        return n_p - n_e - n_mu

    y_lo = 1.0e-6
    y_hi = 0.5 - 1.0e-6
    try:
        f_lo = charge_residual(y_lo)
        f_hi = charge_residual(y_hi)
    except ValueError:
        return None
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return None
    if f_lo * f_hi > 0.0:
        return None

    y_p = brentq(charge_residual, y_lo, y_hi, xtol=1.0e-8, rtol=1.0e-8, maxiter=100)
    n_p = y_p * n_b
    n_n = (1.0 - y_p) * n_b
    m_dirac = solve_dirac_mass(n_n, n_p, m_ref)
    if m_dirac is None:
        return None

    mu_n_star = math.sqrt(kf_from_density(n_n) ** 2 + m_dirac**2)
    mu_p_star = math.sqrt(kf_from_density(n_p) ** 2 + m_dirac**2)
    mu_e = max(mu_n_star - mu_p_star + 2.0 * c_rho * (n_n - n_p), m_e)
    n_e = lepton_density_from_mu(mu_e, m_e)
    n_mu = lepton_density_from_mu(mu_e, m_mu)

    eps_baryons = fermion_energy_density(n_n, m_dirac) + fermion_energy_density(n_p, m_dirac)
    eps_leptons = fermion_energy_density(n_e, m_e) + fermion_energy_density(n_mu, m_mu)
    eps_scalar = 0.5 * (m_ref - m_dirac) ** 2 / c_s if c_s > 0.0 else 0.0
    eps_rho = 0.5 * c_rho * (n_n - n_p) ** 2

    return {
        "y_p": y_p,
        "n_n": n_n,
        "n_p": n_p,
        "n_e": n_e,
        "n_mu": n_mu,
        "mu_e": mu_e,
        "m_ref": m_ref,
        "m_dirac": m_dirac,
        "eps_no_vec": eps_baryons + eps_leptons + eps_scalar + eps_rho,
    }

def vector_factor(n_b):
    x = n_b / n_0
    if x <= 1.0:
        return 1.0
    return 1.0 / (1.0 + alpha_v * (x - 1.0) ** nu_v)

def vector_energy_density(n_b, c_omega0):
    return 0.5 * c_omega0 * vector_factor(n_b) * n_b * n_b

def calibrate_c_omega0():
    state_n0 = beta_equilibrium_state(n_0)
    if state_n0 is None:
        raise ValueError("Failed to calibrate at n0")
    c_omega0 = 2.0 * (M_N - 16.0 - state_n0["eps_no_vec"] / n_0) / n_0
    return c_omega0

def build_eos(c_omega0):
    narr = np.geomspace(0.01 * n_0, 8.0 * n_0, 60)
    eps = []
    pressure = []
    
    valid_n = []
    for n_b in narr:
        state = beta_equilibrium_state(n_b)
        if state is None:
            continue
        eps_val = state["eps_no_vec"] + vector_energy_density(n_b, c_omega0)
        eps.append(eps_val)
        valid_n.append(n_b)
        
    valid_n = np.array(valid_n)
    eps = np.array(eps)
    
    e_per_baryon = eps / valid_n
    pressure = valid_n * valid_n * np.gradient(e_per_baryon, valid_n)
    
    return valid_n, eps, pressure

def solve_tov(eps_of_p, p_c):
    conv = MeV_fm3_to_geo
    r0 = 1.0e-3
    # ``eps_of_p`` is a strict interpolator.  Its lower endpoint is the
    # lowest pressure represented by the density derivative table; below it
    # the toy chain has no EOS and we stop at vacuum rather than extrapolate.
    p_min = float(np.asarray(getattr(eps_of_p, "x", [0.0]))[0])
    p_max = float(np.asarray(getattr(eps_of_p, "x", [np.inf]))[-1])
    if not np.isfinite(p_c) or not (p_min < p_c <= p_max):
        raise ValueError(f"central pressure {p_c!r} outside EOS domain ({p_min}, {p_max}]")
    e_c = float(eps_of_p(p_c))
    
    # We run integration in physical units, but convert internally to geometric units
    def rhs(radius, state):
        mass, pressure = state
        if pressure <= p_min:
            return [0.0, 0.0]
        energy = float(eps_of_p(pressure))
        
        eps_k = energy * conv
        p_k = pressure * conv
        m_k = mass * M_sun_km
        
        denom = radius * (radius - 2.0 * m_k)
        if denom <= 0.0:
            return [0.0, 0.0]
            
        dmdr_k = 4.0 * np.pi * radius**2 * eps_k
        dpdr_k = -(eps_k + p_k) * (m_k + 4.0 * np.pi * radius**3 * p_k) / denom
        
        # Convert dm_dr back to M_sun/km
        dmdr = dmdr_k / M_sun_km
        # Convert dp_dr back to MeV/fm^3/km
        dpdr = dpdr_k / conv
        
        return [dmdr, dpdr]

    def stop(radius, state):
        return state[1] - p_min

    stop.terminal = True
    stop.direction = -1

    sol = solve_ivp(
        rhs, 
        [r0, 40.0], 
        [0.0, p_c], 
        events=stop, 
        max_step=0.2, 
        rtol=1.0e-5, 
        atol=1.0e-8
    )
    
    if sol.t_events[0].size > 0:
        radius = float(sol.t_events[0][0])
        mass = float(sol.y_events[0][0][0])
    else:
        radius = float(sol.t[-1])
        mass = float(sol.y[0, -1])
        
    return mass, radius

def main():
    print("==========================================================================")
    print("  NVG: DIRECT URCA PROTON-FRACTION SCAN (VMF ILLUSTRATIVE CHAIN)")
    print("==========================================================================")
    
    # Calibrate vector parameter
    c_omega0 = calibrate_c_omega0()
    
    # Construct EOS
    narr, eps, pressure = build_eos(c_omega0)
    good = pressure > 0
    narr_g = narr[good]
    eps_g = eps[good]
    pressure_g = pressure[good]
    
    if len(pressure_g) < 2 or not np.all(np.isfinite(pressure_g)) or not np.all(np.isfinite(eps_g)):
        raise ValueError("insufficient finite EOS points for pressure interpolation")
    if not (float(np.min(pressure_g)) < float(np.max(pressure_g))):
        raise ValueError("EOS pressure domain is empty")
    # The derivative-based toy chain is not monotone at its highest-density
    # tail.  Sort and deduplicate pressure before interpolation so the strict
    # domain is explicit; no endpoint-filled state is returned.
    order = np.argsort(pressure_g)
    pressure_g = pressure_g[order]
    eps_g = eps_g[order]
    narr_g = narr_g[order]
    pressure_g, unique_idx = np.unique(pressure_g, return_index=True)
    eps_g = eps_g[unique_idx]
    narr_g = narr_g[unique_idx]
    eps_of_p = interp1d(pressure_g, eps_g, bounds_error=True)
    
    # Direct Urca opens when the proton fraction satisfies the Lattimer threshold:
    # x_p >= 1 / (1 + (1 + x_e^(1/3))^3)
    
    print(f"Calibrated c_omega0 vector coupling: {c_omega0:.2f} MeV*fm^3")
    print(f"{'Density (n_B/n_0)':<20} | {'Proton Fraction (x_p)':<25} | {'Lattimer Threshold':<20} | {'Direct Urca status'}")
    print("-" * 92)
    
    ratios = [1.0, 2.0, 3.0, 4.0, 5.0]
    xp_values = {}
    x_du_values = {}
    
    for r in ratios:
        state = beta_equilibrium_state(r * n_0)
        y_p = state["y_p"]
        x_e = state["n_e"] / state["n_p"] if state["n_p"] > 0 else 1.0
        x_du = 1.0 / (1.0 + (1.0 + x_e**(1.0/3.0))**3)
        status = "OPEN (Fast Cooling)" if y_p >= x_du else "CLOSED (Slow Cooling)"
        print(f"{r:<20.1f} | {y_p*100:<24.2f}% | {x_du*100:<19.2f}% | {status}")
        xp_values[r] = y_p
        x_du_values[r] = x_du
        
    print("-" * 92)
    
    # Solve TOV for central densities
    # Find stellar masses corresponding to central densities
    print("Integrating TOV equations to map central density to stellar mass...")
    p_min = float(np.min(pressure_g))
    p_grid = np.logspace(-0.5, 2.5, 12)
    p_grid = p_grid[p_grid > p_min]
    p_grid = p_grid[p_grid <= float(np.max(pressure_g))]
    if len(p_grid) < 2:
        raise ValueError("EOS pressure domain does not cover the TOV scan")
    masses = []
    central_densities = []
    
    for pc in p_grid:
        m, r = solve_tov(eps_of_p, pc)
        nc = np.interp(pc, pressure_g, narr_g)
        masses.append(m)
        central_densities.append(nc / n_0)
        print(f"  Pc = {pc:6.1f} MeV/fm3 | Central density = {nc/n_0:5.2f} n_0 | Mass = {m:6.3f} M_sun")
        
    # Interpolate mass for M = 1.45 M_sun only if this valid EOS scan reaches
    # that mass.  Otherwise report the threshold as unsupported by this table.
    m_sorted_idx = np.argsort(masses)
    masses_s = np.array(masses)[m_sorted_idx]
    densities_s = np.array(central_densities)[m_sorted_idx]
    
    if masses_s[0] <= 1.45 <= masses_s[-1]:
        nc_at_145 = float(np.interp(1.45, masses_s, densities_s))
        state_at_threshold = beta_equilibrium_state(nc_at_145 * n_0)
        xp_at_threshold = state_at_threshold["y_p"]
        xe_at_threshold = state_at_threshold["n_e"] / state_at_threshold["n_p"]
        xdu_at_threshold = 1.0 / (1.0 + (1.0 + xe_at_threshold**(1.0/3.0))**3)
        print(f"Central density for M = 1.45 M_sun: {nc_at_145:.2f} n_0")
        print(f"Proton fraction at 1.45 M_sun      : {xp_at_threshold*100:.2f}% (Lattimer threshold: {xdu_at_threshold*100:.2f}%)")
    else:
        nc_at_145 = None
        print(
            f"Mass threshold M = 1.45 M_sun is unsupported by the valid EOS scan "
            f"([{masses_s[0]:.3f}, {masses_s[-1]:.3f}] M_sun)."
        )
    print("-" * 92)

    du_open = [ratio for ratio in ratios if xp_values[ratio] >= x_du_values[ratio]]
    print(
        "Status: descriptive proton-fraction scan; Direct Urca is open at "
        f"densities {du_open} n_0 and no stellar-mass threshold is inferred."
    )
    print("==========================================================================")

if __name__ == "__main__":
    main()
