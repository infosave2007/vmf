#!/usr/bin/env python3
"""
Fast checked proof-of-concept for the NVG dense-matter idea.

This version is intentionally narrower than a full neutron-star EOS:
- it fixes the unit mismatch in the linear-density current-mass scaling;
- it replaces slow quadratures by analytic zero-temperature Fermi-gas formulas;
- it treats the output as a symmetric-matter core-EOS screening test,
  not as a final beta-equilibrated neutron-star calculation.
"""

import argparse
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq


# === Constants (hbar = c = 1, energies in MeV, densities in fm^-3) ===
hbar_c = 197.3269804  # MeV*fm
M_N = 939.0
n_0 = 0.16
f_pi = 93.0
m_pi = 140.0
MeV_fm3_to_geo = 1.3234e-6  # MeV/fm^3 -> km^-2
M_sun_km = 1.4766


# === Vacuum calibration (fixed lattice-style input) ===
sigma_piN = 44.0
sigma_sN = 30.0
sigma_heavy = 6.0
sigma_total = sigma_piN + sigma_sN + sigma_heavy
M_Omega_0 = M_N - sigma_total
M_current_0 = sigma_total


def M_Omega(n_b, k1, k2):
    """Residual-sector mass for kappa_Omega(x) = -k1 x / (1 + k2 x)."""
    x = max(n_b / n_0, 0.0)
    return M_Omega_0 * (1.0 + k2 * x) ** (-k1 / k2)


def current_ratio_linear(n_b):
    """Leading-density condensate-inspired scaling with the missing unit conversion fixed."""
    correction = sigma_piN * n_b * hbar_c**3 / (f_pi**2 * m_pi**2)
    return max(1.0 - correction, 0.0)


def M_current(n_b):
    return M_current_0 * current_ratio_linear(n_b)


def M_star(n_b, k1, k2):
    return M_current(n_b) + M_Omega(n_b, k1, k2)


def kF_symmetric(n_b):
    """Symmetric-matter Fermi momentum for degeneracy g = 4."""
    return (1.5 * np.pi**2 * n_b) ** (1.0 / 3.0) * hbar_c


def fg_kinetic_energy_density(n_b, mass_eff):
    if n_b <= 0.0:
        return 0.0
    kf = kF_symmetric(n_b)
    ef = math.sqrt(kf * kf + mass_eff * mass_eff)
    log_term = math.log((kf + ef) / mass_eff)
    num = kf * ef * (2.0 * kf * kf + mass_eff * mass_eff) - mass_eff**4 * log_term
    return num / (4.0 * np.pi**2 * hbar_c**3)


def build_eos(k1, k2, c_omega, nn=320):
    narr = np.geomspace(1.0e-4 * n_0, 8.0 * n_0, nn)
    eps = np.zeros_like(narr)
    ms = np.zeros_like(narr)

    for i, n_b in enumerate(narr):
        mass_eff = M_star(n_b, k1, k2)
        ms[i] = mass_eff
        eps[i] = fg_kinetic_energy_density(n_b, mass_eff) + 0.5 * c_omega * n_b * n_b

    e_per_baryon = eps / narr
    pressure = narr * narr * np.gradient(e_per_baryon, narr)
    cs2 = np.gradient(pressure, narr) / np.gradient(eps, narr)
    return narr, eps, pressure, ms, cs2


def binding_residual(c_omega, k1, k2):
    mass_eff = M_star(n_0, k1, k2)
    eps0 = fg_kinetic_energy_density(n_0, mass_eff) + 0.5 * c_omega * n_0 * n_0
    return eps0 / n_0 - M_N + 16.0


def calibrate_c_omega(k1, k2):
    lo = 0.0
    hi = 2.0e4
    f_lo = binding_residual(lo, k1, k2)
    f_hi = binding_residual(hi, k1, k2)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi):
        return None
    for _ in range(8):
        if f_lo == 0.0:
            return lo
        if f_lo * f_hi < 0.0:
            return brentq(lambda value: binding_residual(value, k1, k2), lo, hi)
        hi *= 2.0
        f_hi = binding_residual(hi, k1, k2)
        if not np.isfinite(f_hi):
            return None
    return None


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
        max_step=0.2,
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


def run_model(k1, k2, with_tov=False):
    c_omega = calibrate_c_omega(k1, k2)
    if c_omega is None:
        return None

    narr, eps, pressure, ms, cs2 = build_eos(k1, k2, c_omega)
    ok = (pressure > 0.0) & np.isfinite(cs2)
    if ok.sum() < 20:
        return None

    p_ok = pressure[ok]
    e_ok = eps[ok]
    order = np.argsort(p_ok)
    p_sorted = p_ok[order]
    e_sorted = e_ok[order]
    mono = np.concatenate([[True], np.diff(p_sorted) > 0.0])
    p_sorted = p_sorted[mono]
    e_sorted = e_sorted[mono]
    if len(p_sorted) < 20:
        return None

    p_n0 = float(np.interp(n_0, narr, pressure))
    mstar_n0 = float(M_star(n_0, k1, k2))
    result = {
        "k1": k1,
        "k2": k2,
        "Cw": float(c_omega),
        "mstar_n0": mstar_n0,
        "p_n0": p_n0,
        "cs2_max": float(np.nanmax(cs2[ok])),
        "causal": bool(np.nanmax(cs2[ok]) <= 1.0),
    }

    if not with_tov:
        result["mmax"] = None
        result["r14"] = None
        return result

    eps_of_p = interp1d(
        p_sorted,
        e_sorted,
        bounds_error=False,
        fill_value=(float(e_sorted[0]), float(e_sorted[-1])),
    )

    pc_grid = np.logspace(np.log10(max(p_sorted[1], 1.0e-4)), np.log10(p_sorted[-1] * 0.9), 20)
    masses = []
    radii = []
    for p_c in pc_grid:
        try:
            mass, radius = solve_tov(eps_of_p, p_c)
        except Exception:
            continue
        if 0.0 < mass < 5.0 and 5.0 < radius < 25.0:
            masses.append(mass)
            radii.append(radius)

    if len(masses) < 5:
        result["mmax"] = None
        result["r14"] = None
        return result

    masses = np.array(masses)
    radii = np.array(radii)
    imax = int(np.argmax(masses))
    result["mmax"] = float(masses[imax])
    result["r14"] = None
    stable = np.arange(imax + 1)
    if len(stable) > 3 and masses[stable].min() < 1.4 < masses[stable].max():
        try:
            result["r14"] = float(interp1d(masses[stable], radii[stable])(1.4))
        except Exception:
            result["r14"] = None
    return result


def verdict_flags(result):
    ms_ratio = result["mstar_n0"] / M_N
    core_ok = result["causal"] and 0.60 <= ms_ratio <= 0.80
    sat_ok = abs(result["p_n0"]) <= 5.0
    return core_ok, sat_ok


def main():
    parser = argparse.ArgumentParser(description="Checked NVG dense-matter proof-of-concept")
    parser.add_argument(
        "--with-tov",
        action="store_true",
        help="also run the expensive TOV sanity scan for each viable core EOS",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("NVG EOS CHECKED: FAST SYMMETRIC-MATTER CORE SCREENING")
    if args.with_tov:
        print("fixed lattice input -> corrected units -> fast EOS -> TOV sanity scan")
    else:
        print("fixed lattice input -> corrected units -> fast EOS core screening")
    print("NOTE: this is not yet a beta-equilibrated neutron-star EOS")
    print("=" * 72)
    print(f"Sigma total = {sigma_total:.1f} MeV")
    print(f"M_Omega,0  = {M_Omega_0:.1f} MeV")
    print(f"f_Omega,0  = {M_Omega_0 / M_N:.4f}")
    print(f"Linear current-mass ratio at n0 = {current_ratio_linear(n_0):.4f}")
    print()

    tests = [
        (0.20, 0.40), (0.20, 0.80),
        (0.30, 0.40), (0.30, 0.80),
        (0.40, 0.40), (0.40, 0.80),
        (0.50, 0.80), (0.60, 1.20),
    ]

    results = []
    for k1, k2 in tests:
        result = run_model(k1, k2, with_tov=args.with_tov)
        if result is not None:
            results.append(result)

    if not results:
        print("No numerically viable models were produced.")
        return 1

    print(f"Scanned models: {len(tests)}")
    print(f"Numerically viable EOS curves: {len(results)}")
    print()
    print(
        f"{'k1':>4} {'k2':>4} {'M*/M':>7} {'P(n0)':>10} {'Cw':>8} "
        f"{'Mmax':>7} {'R1.4':>6} {'cs2max':>7} {'Core':>6} {'Sat':>5}"
    )
    print("-" * 78)

    core_passes = []
    for result in results:
        core_ok, sat_ok = verdict_flags(result)
        if core_ok:
            core_passes.append(result)
        r14_str = f"{result['r14']:.1f}" if result["r14"] is not None else "  -- "
        mmax_str = f"{result['mmax']:.3f}" if result["mmax"] is not None else "  --  "
        print(
            f"{result['k1']:4.2f} {result['k2']:4.2f} {result['mstar_n0']/M_N:7.3f} "
            f"{result['p_n0']:10.3f} {result['Cw']:8.1f} {mmax_str:>7} "
            f"{r14_str:>6} {result['cs2_max']:7.3f} {'yes' if core_ok else 'no':>6} {'yes' if sat_ok else 'no':>5}"
        )

    print("-" * 78)
    print(f"Core-screening passes: {len(core_passes)}")
    if not args.with_tov:
        print("TOV scan: skipped by default; use --with-tov for the expensive sanity check.")

    if core_passes:
        best = sorted(core_passes, key=lambda item: (abs(item['mstar_n0'] / M_N - 0.70), abs(item['p_n0'])))[0]
        print()
        print("Best core-screening point:")
        print(f"  k1 = {best['k1']:.2f}, k2 = {best['k2']:.2f}")
        print(f"  M*(n0)/M_N = {best['mstar_n0']/M_N:.3f}")
        print(f"  P(n0) = {best['p_n0']:.3f} MeV/fm^3")
        if best['mmax'] is not None:
            print(f"  Mmax = {best['mmax']:.3f} M_sun")
        if best['r14'] is not None:
            print(f"  R1.4 = {best['r14']:.2f} km  (core-only, approximate)")
        print(f"  cs2_max = {best['cs2_max']:.3f}")
    else:
        print()
        print("No point passed the core-only screening criteria.")
        print("In this parameterization the idea is disfavored even before a full NS treatment.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())