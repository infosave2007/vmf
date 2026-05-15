#!/usr/bin/env python3
"""
NVG dense matter: checked beta-equilibrated core EOS prototype.

This script extends the fast proof-of-concept in two ways:
1. adds a self-consistent scalar-attraction sector through a Walecka-like gap equation;
2. replaces symmetric matter by beta-equilibrated npe mu matter with an isovector term.

It is still a core-EOS prototype rather than a final neutron-star model:
- no crust matching;
- no nonlinear meson self-interactions;
- no Bayesian calibration.
"""

import argparse
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq


# === Constants ===
hbar_c = 197.3269804  # MeV*fm
M_N = 939.0
n_0 = 0.16  # fm^-3
f_pi = 93.0
m_pi = 140.0
m_e = 0.511
m_mu = 105.658
MeV_fm3_to_geo = 1.3234e-6
M_sun_km = 1.4766


# === Vacuum calibration ===
sigma_piN = 44.0
sigma_sN = 30.0
sigma_heavy = 6.0
sigma_total = sigma_piN + sigma_sN + sigma_heavy
M_Omega_0 = M_N - sigma_total
M_current_0 = sigma_total


def M_Omega(n_b, k1, k2):
    x = max(n_b / n_0, 0.0)
    return M_Omega_0 * (1.0 + k2 * x) ** (-k1 / k2)


def current_ratio_linear(n_b):
    correction = sigma_piN * n_b * hbar_c**3 / (f_pi**2 * m_pi**2)
    return max(1.0 - correction, 0.0)


def M_current(n_b):
    return M_current_0 * current_ratio_linear(n_b)


def M_base(n_b, k1, k2):
    return M_current(n_b) + M_Omega(n_b, k1, k2)


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


def solve_dirac_mass(n_n, n_p, m_ref, c_s):
    if c_s <= 0.0:
        return m_ref

    def residual(m_dirac):
        ns_total = fermion_scalar_density(n_n, m_dirac) + fermion_scalar_density(n_p, m_dirac)
        return m_dirac - m_ref + c_s * ns_total

    low = 5.0
    high = max(m_ref, low + 1.0)
    f_low = residual(low)
    f_high = residual(high)
    if not (np.isfinite(f_low) and np.isfinite(f_high)):
        return None
    if f_low * f_high > 0.0:
        return None
    return brentq(residual, low, high, xtol=1.0e-8, rtol=1.0e-8, maxiter=100)


def beta_equilibrium_state(n_b, k1, k2, c_s, c_rho):
    """Solve beta-equilibrated npe mu matter at fixed baryon density."""
    m_ref = M_base(n_b, k1, k2)

    def charge_residual(y_p):
        n_p = y_p * n_b
        n_n = (1.0 - y_p) * n_b
        m_dirac = solve_dirac_mass(n_n, n_p, m_ref, c_s)
        if m_dirac is None:
            raise ValueError("Dirac-mass gap failed")

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
    m_dirac = solve_dirac_mass(n_n, n_p, m_ref, c_s)
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


def calibrate_c_omega(k1, k2, c_s, c_rho):
    state_n0 = beta_equilibrium_state(n_0, k1, k2, c_s, c_rho)
    if state_n0 is None:
        return None, None
    c_omega = 2.0 * (M_N - 16.0 - state_n0["eps_no_vec"] / n_0) / n_0
    if not np.isfinite(c_omega) or c_omega <= 0.0:
        return None, None
    return c_omega, state_n0


def build_eos(k1, k2, c_s, c_rho, c_omega, nn=180):
    narr = np.geomspace(0.02 * n_0, 8.0 * n_0, nn)
    eps = np.full_like(narr, np.nan)
    mdirac = np.full_like(narr, np.nan)
    yprot = np.full_like(narr, np.nan)
    mu_e = np.full_like(narr, np.nan)

    for i, n_b in enumerate(narr):
        state = beta_equilibrium_state(n_b, k1, k2, c_s, c_rho)
        if state is None:
            continue
        eps[i] = state["eps_no_vec"] + 0.5 * c_omega * n_b * n_b
        mdirac[i] = state["m_dirac"]
        yprot[i] = state["y_p"]
        mu_e[i] = state["mu_e"]

    good = np.isfinite(eps)
    if good.sum() < 40:
        return None

    narr_g = narr[good]
    eps_g = eps[good]
    mdirac_g = mdirac[good]
    yprot_g = yprot[good]
    mu_e_g = mu_e[good]
    e_per_baryon = eps_g / narr_g
    pressure_g = narr_g * narr_g * np.gradient(e_per_baryon, narr_g)
    cs2_g = np.gradient(pressure_g, narr_g) / np.gradient(eps_g, narr_g)
    return narr_g, eps_g, pressure_g, mdirac_g, yprot_g, mu_e_g, cs2_g


def solve_tov(eps_of_p, p_c):
    conv = MeV_fm3_to_geo
    r0 = 1.0e-3
    e_c = float(eps_of_p(p_c))
    m0 = 4.0 * np.pi * r0**3 * e_c * conv / 3.0

    def rhs(radius, state):
        mass, pressure = state
        if pressure <= 0.0:
            return [0.0, 0.0]
        energy = float(eps_of_p(pressure))
        denom = radius * (radius - 2.0 * mass)
        if denom <= 0.0:
            return [0.0, 0.0]
        dmdr = 4.0 * np.pi * radius**2 * energy * conv
        dpdr = -(energy * conv + pressure * conv) * (mass + 4.0 * np.pi * radius**3 * pressure * conv) / denom
        return [dmdr, dpdr]

    def stop(radius, state):
        return state[1]

    stop.terminal = True
    stop.direction = -1

    sol = solve_ivp(
        rhs,
        [r0, 40.0],
        [m0, p_c],
        events=stop,
        max_step=0.15,
        rtol=1.0e-6,
        atol=1.0e-10,
    )
    if sol.t_events[0].size > 0:
        radius = float(sol.t_events[0][0])
        mass = float(sol.y_events[0][0][0]) / M_sun_km
    else:
        radius = float(sol.t[-1])
        mass = float(sol.y[0, -1]) / M_sun_km
    return mass, radius


def tov_scan(p_sorted, e_sorted):
    eps_of_p = interp1d(
        p_sorted,
        e_sorted,
        bounds_error=False,
        fill_value=(float(e_sorted[0]), float(e_sorted[-1])),
    )
    p_grid = np.logspace(np.log10(max(p_sorted[1], 1.0e-4)), np.log10(p_sorted[-1] * 0.9), 18)
    masses = []
    radii = []
    for p_c in p_grid:
        try:
            mass, radius = solve_tov(eps_of_p, p_c)
        except Exception:
            continue
        if 0.0 < mass < 5.0 and 5.0 < radius < 25.0:
            masses.append(mass)
            radii.append(radius)
    if len(masses) < 5:
        return None, None
    masses = np.array(masses)
    radii = np.array(radii)
    return masses, radii


def run_model(k1, k2, c_s, c_rho, with_tov=False):
    c_omega, state_n0 = calibrate_c_omega(k1, k2, c_s, c_rho)
    if c_omega is None:
        return None
    eos = build_eos(k1, k2, c_s, c_rho, c_omega)
    if eos is None:
        return None

    narr, eps, pressure, mdirac, yprot, mu_e, cs2 = eos
    good = (pressure > 0.0) & np.isfinite(cs2)
    if good.sum() < 20:
        return None
    p_sorted = pressure[good]
    e_sorted = eps[good]
    order = np.argsort(p_sorted)
    p_sorted = p_sorted[order]
    e_sorted = e_sorted[order]
    mono = np.concatenate([[True], np.diff(p_sorted) > 0.0])
    p_sorted = p_sorted[mono]
    e_sorted = e_sorted[mono]
    if len(p_sorted) < 20:
        return None

    p_n0 = float(np.interp(n_0, narr, pressure))
    mdirac_n0 = float(state_n0["m_dirac"])
    yp_n0 = float(state_n0["y_p"])
    mu_e_n0 = float(state_n0["mu_e"])

    result = {
        "k1": k1,
        "k2": k2,
        "Cs": c_s,
        "Crho": c_rho,
        "Cw": float(c_omega),
        "p_n0": p_n0,
        "mdirac_n0": mdirac_n0,
        "yp_n0": yp_n0,
        "mu_e_n0": mu_e_n0,
        "cs2_max": float(np.nanmax(cs2[good])),
        "causal": bool(np.nanmax(cs2[good]) <= 1.0),
        "mmax": None,
        "r14": None,
    }

    if not with_tov:
        return result

    masses, radii = tov_scan(p_sorted, e_sorted)
    if masses is None:
        return result
    imax = int(np.argmax(masses))
    result["mmax"] = float(masses[imax])
    stable = np.arange(imax + 1)
    if len(stable) > 3 and masses[stable].min() < 1.4 < masses[stable].max():
        try:
            result["r14"] = float(interp1d(masses[stable], radii[stable])(1.4))
        except Exception:
            result["r14"] = None
    return result


def core_flags(result):
    m_ratio = result["mdirac_n0"] / M_N
    sat_ok = abs(result["p_n0"]) <= 5.0
    mstar_ok = 0.55 <= m_ratio <= 0.80
    yp_ok = 0.01 <= result["yp_n0"] <= 0.20
    core_ok = result["causal"] and mstar_ok and yp_ok
    return core_ok, sat_ok, mstar_ok, yp_ok


def main():
    parser = argparse.ArgumentParser(description="Checked beta-equilibrated NVG core EOS prototype")
    parser.add_argument(
        "--with-tov",
        action="store_true",
        help="also run an expensive TOV scan for models that pass the core screening",
    )
    args = parser.parse_args()

    print("=" * 76)
    print("NVG BETA-CHECKED: SCALAR + ISOVECTOR + LEPTON CORE EOS")
    if args.with_tov:
        print("fixed lattice input -> beta-equilibrated core EOS -> optional TOV sanity scan")
    else:
        print("fixed lattice input -> beta-equilibrated core EOS screening")
    print("NOTE: this is still a core-only prototype; no crust matching yet")
    print("=" * 76)
    print(f"Sigma total = {sigma_total:.1f} MeV")
    print(f"M_Omega,0  = {M_Omega_0:.1f} MeV")
    print(f"f_Omega,0  = {M_Omega_0 / M_N:.4f}")
    print(f"Linear current-mass ratio at n0 = {current_ratio_linear(n_0):.4f}")
    print()

    tests = [
        (0.20, 0.60, 600.0, 300.0),
        (0.20, 0.60, 900.0, 600.0),
        (0.25, 0.80, 900.0, 600.0),
        (0.25, 0.80, 1200.0, 900.0),
        (0.30, 0.80, 1200.0, 600.0),
        (0.30, 1.00, 1400.0, 900.0),
        (0.35, 1.00, 1400.0, 600.0),
        (0.35, 1.20, 1600.0, 900.0),
        (0.40, 1.20, 1800.0, 1200.0),
        (0.45, 1.20, 1800.0, 900.0),
    ]

    results = []
    for k1, k2, c_s, c_rho in tests:
        result = run_model(k1, k2, c_s, c_rho, with_tov=False)
        if result is not None:
            results.append(result)

    if not results:
        print("No numerically viable beta-equilibrated core EOS points were found.")
        return 1

    print(f"Scanned models: {len(tests)}")
    print(f"Numerically viable beta-equilibrated EOS curves: {len(results)}")
    print()
    print(
        f"{'k1':>4} {'k2':>4} {'Cs':>6} {'Crho':>6} {'M_D/M':>7} {'Yp(n0)':>7} "
        f"{'P(n0)':>9} {'c_s^2':>7} {'Core':>6} {'Sat':>5}"
    )
    print("-" * 86)

    core_passes = []
    for result in results:
        core_ok, sat_ok, _, _ = core_flags(result)
        if core_ok:
            core_passes.append(result)
        print(
            f"{result['k1']:4.2f} {result['k2']:4.2f} {result['Cs']:6.0f} {result['Crho']:6.0f} "
            f"{result['mdirac_n0']/M_N:7.3f} {result['yp_n0']:7.3f} {result['p_n0']:9.3f} "
            f"{result['cs2_max']:7.3f} {'yes' if core_ok else 'no':>6} {'yes' if sat_ok else 'no':>5}"
        )

    print("-" * 86)
    print(f"Core-screening passes: {len(core_passes)}")

    if args.with_tov and core_passes:
        print("Running TOV sanity scan for core-passing models...")
        tov_results = []
        for result in core_passes[:4]:
            rerun = run_model(result['k1'], result['k2'], result['Cs'], result['Crho'], with_tov=True)
            if rerun is not None:
                tov_results.append(rerun)
        if tov_results:
            print()
            print(f"{'k1':>4} {'k2':>4} {'Cs':>6} {'Crho':>6} {'Mmax':>7} {'R1.4':>6}")
            print("-" * 44)
            for result in tov_results:
                r14_str = f"{result['r14']:.1f}" if result['r14'] is not None else "  -- "
                mmax_str = f"{result['mmax']:.3f}" if result['mmax'] is not None else "  --  "
                print(
                    f"{result['k1']:4.2f} {result['k2']:4.2f} {result['Cs']:6.0f} {result['Crho']:6.0f} "
                    f"{mmax_str:>7} {r14_str:>6}"
                )
        else:
            print("TOV scan did not return valid stellar branches for the screened models.")
    elif not args.with_tov:
        print("TOV scan: skipped by default; use --with-tov after core screening succeeds.")

    if core_passes:
        best = sorted(core_passes, key=lambda item: (abs(item['mdirac_n0']/M_N - 0.70), abs(item['p_n0'])))[0]
        print()
        print("Best beta-equilibrated core-screening point:")
        print(f"  k1 = {best['k1']:.2f}, k2 = {best['k2']:.2f}")
        print(f"  Cs = {best['Cs']:.0f} MeV*fm^3, Crho = {best['Crho']:.0f} MeV*fm^3")
        print(f"  M_D(n0)/M_N = {best['mdirac_n0']/M_N:.3f}")
        print(f"  Yp(n0) = {best['yp_n0']:.3f}")
        print(f"  P(n0) = {best['p_n0']:.3f} MeV/fm^3")
        print(f"  c_s^2,max = {best['cs2_max']:.3f}")
    else:
        print()
        print("No point passed the beta-equilibrated core screening.")
        print("In this parameterization the idea remains disfavored before a full NS fit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())