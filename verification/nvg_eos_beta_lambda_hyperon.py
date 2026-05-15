#!/usr/bin/env python3
"""NVG dense matter: pilot Lambda-hyperon onset inside the beta-equilibrium solver.

This is the next step after the external softening tests. Instead of appending a
separate high-density branch at the TOV level, this script adds a neutral
Lambda hyperon directly into the composition solver built on top of the best
saturated-vector hadronic baseline.

The goal is narrow and testable:
can a minimal in-medium Lambda degree of freedom reduce the overmassive TOV
branches more effectively than external CSS/Maxwell appendages?
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import nvg_eos_beta_saturated_vector as base


M_LAMBDA = 1115.683

BEST_BASELINE = {
    "k1": 0.25,
    "k2": 0.80,
    "Cs": 900.0,
    "Crho": 600.0,
    "alpha_v": 4.0,
    "nu_v": 2.0,
}


def lambda_effective_mass(m_n_star, m_ref, x_sigma):
    sigma_shift = max(m_ref - m_n_star, 0.0)
    return max(M_LAMBDA - x_sigma * sigma_shift, 50.0)


def solve_dirac_mass_with_lambda(n_n, n_p, n_lam, m_ref, c_s, x_sigma):
    if c_s <= 0.0:
        return m_ref

    def residual(m_n_star):
        m_lam_star = lambda_effective_mass(m_n_star, m_ref, x_sigma)
        ns_total = (
            base.fermion_scalar_density(n_n, m_n_star)
            + base.fermion_scalar_density(n_p, m_n_star)
            + x_sigma * base.fermion_scalar_density(n_lam, m_lam_star)
        )
        return m_n_star - m_ref + c_s * ns_total

    low = 5.0
    high = max(m_ref, low + 1.0)
    f_low = residual(low)
    f_high = residual(high)
    if not (np.isfinite(f_low) and np.isfinite(f_high)):
        return None
    if f_low * f_high > 0.0:
        return None
    return brentq(residual, low, high, xtol=1.0e-8, rtol=1.0e-8, maxiter=100)


def state_for_lambda_fraction(n_b, y_lam, k1, k2, c_s, c_rho, c_omega0, alpha_v, nu_v, x_sigma, x_omega):
    if not (0.0 <= y_lam < 1.0):
        return None

    m_ref = base.M_base(n_b, k1, k2)
    c_omega_eff = c_omega0 * base.vector_factor(n_b, alpha_v, nu_v)

    def charge_residual(y_p):
        n_lam = y_lam * n_b
        n_p = y_p * n_b
        n_n = n_b - n_p - n_lam
        if n_n <= 0.0:
            raise ValueError("negative neutron density")

        m_n_star = solve_dirac_mass_with_lambda(n_n, n_p, n_lam, m_ref, c_s, x_sigma)
        if m_n_star is None:
            raise ValueError("Dirac gap failed")
        m_lam_star = lambda_effective_mass(m_n_star, m_ref, x_sigma)

        vec_source = n_n + n_p + x_omega * n_lam
        sigma_omega_n = c_omega_eff * vec_source
        delta_rho = c_rho * (n_n - n_p)

        mu_n = math.sqrt(base.kf_from_density(n_n) ** 2 + m_n_star**2) + sigma_omega_n + delta_rho
        mu_p = math.sqrt(base.kf_from_density(n_p) ** 2 + m_n_star**2) + sigma_omega_n - delta_rho
        mu_e = max(mu_n - mu_p, base.m_e)
        n_e = base.lepton_density_from_mu(mu_e, base.m_e)
        n_mu = base.lepton_density_from_mu(mu_e, base.m_mu)
        return n_p - n_e - n_mu

    y_p_lo = 1.0e-6
    y_p_hi = min(0.5 - 1.0e-6, 1.0 - y_lam - 1.0e-6)
    if y_p_hi <= y_p_lo:
        return None

    try:
        f_lo = charge_residual(y_p_lo)
        f_hi = charge_residual(y_p_hi)
    except ValueError:
        return None
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return None
    if f_lo * f_hi > 0.0:
        return None

    y_p = brentq(charge_residual, y_p_lo, y_p_hi, xtol=1.0e-8, rtol=1.0e-8, maxiter=100)

    n_lam = y_lam * n_b
    n_p = y_p * n_b
    n_n = n_b - n_p - n_lam
    if n_n <= 0.0:
        return None

    m_n_star = solve_dirac_mass_with_lambda(n_n, n_p, n_lam, m_ref, c_s, x_sigma)
    if m_n_star is None:
        return None
    m_lam_star = lambda_effective_mass(m_n_star, m_ref, x_sigma)

    vec_source = n_n + n_p + x_omega * n_lam
    sigma_omega_n = c_omega_eff * vec_source
    sigma_omega_lam = x_omega * sigma_omega_n
    delta_rho = c_rho * (n_n - n_p)

    mu_n = math.sqrt(base.kf_from_density(n_n) ** 2 + m_n_star**2) + sigma_omega_n + delta_rho
    mu_p = math.sqrt(base.kf_from_density(n_p) ** 2 + m_n_star**2) + sigma_omega_n - delta_rho
    mu_lam = math.sqrt(base.kf_from_density(n_lam) ** 2 + m_lam_star**2) + sigma_omega_lam
    mu_e = max(mu_n - mu_p, base.m_e)
    n_e = base.lepton_density_from_mu(mu_e, base.m_e)
    n_mu = base.lepton_density_from_mu(mu_e, base.m_mu)

    eps_baryons = (
        base.fermion_energy_density(n_n, m_n_star)
        + base.fermion_energy_density(n_p, m_n_star)
        + base.fermion_energy_density(n_lam, m_lam_star)
    )
    eps_leptons = base.fermion_energy_density(n_e, base.m_e) + base.fermion_energy_density(n_mu, base.m_mu)
    eps_scalar = 0.5 * (m_ref - m_n_star) ** 2 / c_s if c_s > 0.0 else 0.0
    eps_rho = 0.5 * c_rho * (n_n - n_p) ** 2
    eps_vector = 0.5 * c_omega_eff * vec_source * vec_source

    return {
        "y_p": y_p,
        "y_lam": y_lam,
        "n_n": n_n,
        "n_p": n_p,
        "n_lam": n_lam,
        "mu_n": mu_n,
        "mu_p": mu_p,
        "mu_lam": mu_lam,
        "mu_e": mu_e,
        "m_ref": m_ref,
        "m_n_star": m_n_star,
        "m_lam_star": m_lam_star,
        "eps_total": eps_baryons + eps_leptons + eps_scalar + eps_rho + eps_vector,
    }


def beta_equilibrium_lambda_state(n_b, k1, k2, c_s, c_rho, c_omega0, alpha_v, nu_v, x_sigma, x_omega):
    nucleonic = state_for_lambda_fraction(n_b, 0.0, k1, k2, c_s, c_rho, c_omega0, alpha_v, nu_v, x_sigma, x_omega)
    if nucleonic is None:
        return None

    residual0 = nucleonic["mu_n"] - nucleonic["mu_lam"]
    if residual0 <= 0.0:
        return nucleonic

    def lambda_residual(y_lam):
        state = state_for_lambda_fraction(n_b, y_lam, k1, k2, c_s, c_rho, c_omega0, alpha_v, nu_v, x_sigma, x_omega)
        if state is None:
            raise ValueError("invalid lambda state")
        return state["mu_n"] - state["mu_lam"]

    scan_points = np.linspace(1.0e-4, 0.45, 18)
    previous_y = 0.0
    previous_r = residual0
    for y_lam in scan_points:
        try:
            current_r = lambda_residual(y_lam)
        except ValueError:
            continue
        if not np.isfinite(current_r):
            continue
        if previous_r * current_r <= 0.0:
            y_root = brentq(lambda_residual, previous_y, y_lam, xtol=1.0e-8, rtol=1.0e-8, maxiter=100)
            return state_for_lambda_fraction(
                n_b, y_root, k1, k2, c_s, c_rho, c_omega0, alpha_v, nu_v, x_sigma, x_omega
            )
        previous_y = y_lam
        previous_r = current_r

    return None


def build_lambda_eos(k1, k2, c_s, c_rho, alpha_v, nu_v, x_sigma, x_omega, nn=120):
    c_omega0, state_n0 = base.calibrate_c_omega0(k1, k2, c_s, c_rho)
    if c_omega0 is None:
        return None

    narr = np.geomspace(0.02 * base.n_0, 8.0 * base.n_0, nn)
    eps = np.full_like(narr, np.nan)
    yprot = np.full_like(narr, np.nan)
    ylam = np.full_like(narr, np.nan)
    mdirac = np.full_like(narr, np.nan)

    for index, n_b in enumerate(narr):
        state = beta_equilibrium_lambda_state(n_b, k1, k2, c_s, c_rho, c_omega0, alpha_v, nu_v, x_sigma, x_omega)
        if state is None:
            continue
        eps[index] = state["eps_total"]
        yprot[index] = state["y_p"]
        ylam[index] = state["y_lam"]
        mdirac[index] = state["m_n_star"]

    good = np.isfinite(eps)
    if good.sum() < 40:
        return None

    narr_g = narr[good]
    eps_g = eps[good]
    yprot_g = yprot[good]
    ylam_g = ylam[good]
    mdirac_g = mdirac[good]
    e_per_baryon = eps_g / narr_g
    pressure_g = narr_g * narr_g * np.gradient(e_per_baryon, narr_g)
    cs2_g = np.gradient(pressure_g, narr_g) / np.gradient(eps_g, narr_g)
    return narr_g, eps_g, pressure_g, yprot_g, ylam_g, mdirac_g, cs2_g, c_omega0, state_n0


def run_model(x_sigma, x_omega, with_tov=True):
    eos = build_lambda_eos(
        BEST_BASELINE["k1"],
        BEST_BASELINE["k2"],
        BEST_BASELINE["Cs"],
        BEST_BASELINE["Crho"],
        BEST_BASELINE["alpha_v"],
        BEST_BASELINE["nu_v"],
        x_sigma,
        x_omega,
    )
    if eos is None:
        return None

    narr, eps, pressure, yprot, ylam, mdirac, cs2, c_omega0, state_n0 = eos
    good = (pressure > 0.0) & np.isfinite(eps)
    if good.sum() < 8:
        return None

    narr_good = narr[good]
    eps_good = eps[good]
    pressure_good = pressure[good]
    order = np.argsort(narr_good)
    narr_good = narr_good[order]
    eps_good = eps_good[order]
    pressure_good = pressure_good[order]

    running_max = -np.inf
    keep = np.zeros_like(pressure_good, dtype=bool)
    for index, value in enumerate(pressure_good):
        if value > running_max + 1.0e-8:
            keep[index] = True
            running_max = value

    if keep.sum() < 8:
        return None

    narr_mono = narr_good[keep]
    eps_mono = eps_good[keep]
    pressure_mono = pressure_good[keep]
    order_p = np.argsort(pressure_mono)
    p_sorted = pressure_mono[order_p]
    e_sorted = eps_mono[order_p]
    mono = np.concatenate([[True], np.diff(p_sorted) > 0.0])
    p_sorted = p_sorted[mono]
    e_sorted = e_sorted[mono]
    if len(p_sorted) < 8:
        return None

    cs2_branch = np.gradient(pressure_mono, narr_mono) / np.gradient(eps_mono, narr_mono)
    finite_cs2 = cs2_branch[np.isfinite(cs2_branch)]
    cs2_max = float(np.nanmax(finite_cs2)) if finite_cs2.size > 0 else float("nan")

    onset_density = None
    onset_mask = ylam > 1.0e-4
    if np.any(onset_mask):
        onset_density = float(narr[np.argmax(onset_mask)])

    result = {
        "x_sigma": x_sigma,
        "x_omega": x_omega,
        "c_omega0": c_omega0,
        "mdirac_n0": float(np.interp(base.n_0, narr, mdirac)),
        "yp_n0": float(np.interp(base.n_0, narr, yprot)),
        "ylam_max": float(np.nanmax(ylam)),
        "onset_ratio": onset_density / base.n_0 if onset_density is not None else None,
        "p_n0": float(np.interp(base.n_0, narr, pressure)),
        "cs2_max": cs2_max,
        "causal": bool(np.isfinite(cs2_max) and cs2_max <= 1.0),
        "mmax": None,
        "r14": None,
    }

    if not with_tov:
        return result

    masses, radii = base.tov_scan(p_sorted, e_sorted)
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


def screening_flags(result):
    m_ratio = result["mdirac_n0"] / base.M_N
    sat_ok = abs(result["p_n0"]) <= 5.0
    mstar_ok = 0.55 <= m_ratio <= 0.80
    yp_ok = 0.01 <= result["yp_n0"] <= 0.20
    hyper_ok = result["onset_ratio"] is None or 1.0 <= result["onset_ratio"] <= 6.0
    core_ok = result["causal"] and mstar_ok and yp_ok and hyper_ok
    return core_ok, sat_ok


def main():
    print("=" * 88)
    print("NVG BETA-SATURATED + LAMBDA HYPERON PILOT")
    print("best saturated-vector baseline -> add Lambda onset inside beta equilibrium -> TOV")
    print("=" * 88)
    print(f"Baseline: k1 = {BEST_BASELINE['k1']:.2f}, k2 = {BEST_BASELINE['k2']:.2f}")
    print(f"alpha_v = {BEST_BASELINE['alpha_v']:.1f}, nu_v = {BEST_BASELINE['nu_v']:.1f}")
    print()

    tests = [
        (1.00, 0.20),
        (1.10, 0.20),
        (1.20, 0.20),
        (1.20, 0.00),
    ]

    results = []
    for index, (x_sigma, x_omega) in enumerate(tests, start=1):
        print(f"Running {index}/{len(tests)}: x_sigma = {x_sigma:.2f}, x_omega = {x_omega:.2f}", flush=True)
        result = run_model(x_sigma, x_omega, with_tov=True)
        if result is None:
            print("  -> no viable Lambda EOS branch", flush=True)
            continue
        results.append(result)
        onset_str = f"{result['onset_ratio']:.2f}" if result['onset_ratio'] is not None else "--"
        r14_str = f"{result['r14']:.2f}" if result['r14'] is not None else "--"
        mmax_str = f"{result['mmax']:.3f}" if result['mmax'] is not None else "--"
        print(
            f"  -> onset/n0 = {onset_str}, yLambda,max = {result['ylam_max']:.3f}, "
            f"Mmax = {mmax_str}, R1.4 = {r14_str}, c_s^2,max = {result['cs2_max']:.3f}",
            flush=True,
        )

    if not results:
        print("No Lambda-hyperon branches were produced.")
        return 1

    print()
    print(f"Scanned models: {len(tests)}")
    print(f"Valid Lambda branches: {len(results)}")
    print()
    print(
        f"{'x_s':>4} {'x_w':>4} {'onset/n0':>8} {'yL,max':>7} {'M_D/M':>7} {'P(n0)':>9} {'c_s^2':>7} {'Mmax':>7} {'R1.4':>6}"
    )
    print("-" * 86)
    for result in results:
        onset_str = f"{result['onset_ratio']:.2f}" if result['onset_ratio'] is not None else "  --  "
        mmax_str = f"{result['mmax']:.3f}" if result['mmax'] is not None else "  --  "
        r14_str = f"{result['r14']:.2f}" if result['r14'] is not None else "  --  "
        print(
            f"{result['x_sigma']:4.2f} {result['x_omega']:4.2f} {onset_str:>8} {result['ylam_max']:7.3f} "
            f"{result['mdirac_n0']/base.M_N:7.3f} {result['p_n0']:9.3f} {result['cs2_max']:7.3f} {mmax_str:>7} {r14_str:>6}"
        )

    best = sorted(results, key=lambda item: item['mmax'] if item['mmax'] is not None else 99.0)[0]
    print()
    if best['mmax'] is not None and best['mmax'] <= 2.6:
        print("A Lambda-hyperon branch entered the target mass window.")
    else:
        print("No Lambda-hyperon branch reached the target mass window yet.")
    print(f"Lowest Mmax found = {best['mmax']:.3f} Msun" if best['mmax'] is not None else "Lowest Mmax found = --")
    print(f"  x_sigma = {best['x_sigma']:.2f}, x_omega = {best['x_omega']:.2f}")
    onset_best = f"{best['onset_ratio']:.2f}" if best['onset_ratio'] is not None else "--"
    print(f"  onset / n0 = {onset_best}, yLambda,max = {best['ylam_max']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())