#!/usr/bin/env python3
"""
Hawking evaporation in the Hayward-core model: temperature and horizon map.

The regular-core length is computed from the declared QCD anchor and the exact
Hayward metric is solved for its outer horizon and Hawking temperature:

  T_H = (hbar c / k_B) * f'(r_+) / (4 pi),   f(r) = 1 - 2Mr^2/(r^3 + 2Ml^2)

The script therefore reports the model-derived extremal mass and temperature
ceiling, plus geometric remnant sensitivities.  A CMB history, PBH formation
abundance, detector response, and dark-matter likelihood are not implemented;
standard-PBH limits are not silently promoted to an NVG exclusion or allowance.
Independent evaporation bursts, Hawking-background components, or sub-solar
black-hole horizons remain falsifier protocols rather than evaluated data.
"""
from __future__ import annotations
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

# Anchors (CODATA + repo QCD anchor)
G = 6.674e-8            # cgs
c = 2.998e10
hbar = 1.0546e-27
k_B = 1.3807e-16
M_sun = 1.989e33

M_Omega = 859.0                                   # MeV
hbar_c = 197.327                                  # MeV fm
rho_c = M_Omega**4 / hbar_c**3 * 1.7827e12        # g/cm^3
l_h = math.sqrt(3 * c**2 / (8 * math.pi * G * rho_c))   # Hayward length, cm

M_CRIT = (3 * math.sqrt(3) / 4) * l_h * c**2 / G        # extremal mass, g


def _validated_mass(M_g):
    mass = float(M_g)
    if not math.isfinite(mass) or mass < 0:
        raise ValueError("mass must be finite and nonnegative")
    return mass


def horizon(M_g):
    """Outer horizon in cm, including the double root at M_CRIT.

    The horizonless branch is 0 <= M < M_CRIT. Near extremality solve
    delta = t^2(3+2t)/(2+6t+3t^2), where t=r/(sqrt(3)*l)-1 and
    delta=(M-M_CRIT)/M_CRIT, to avoid cancellation in f at its double root.
    """
    M_g = _validated_mass(M_g)
    if M_g < M_CRIT:
        return None
    r_crit = math.sqrt(3.0) * l_h
    if M_g == M_CRIT:
        return r_crit
    delta = (M_g - M_CRIT) / M_CRIT
    if delta <= 1.0:
        equation = lambda t: t * t * (3 + 2 * t) / (2 + 6 * t + 3 * t * t) - delta
        t = brentq(equation, 0.0, 1.5 * (1.0 + delta) - 1.0,
                   xtol=np.nextafter(0.0, 1.0), rtol=8 * np.finfo(float).eps)
        return r_crit * (1.0 + t)
    # Scaling by the Schwarzschild radius keeps very large finite masses safe.
    r_s = 2 * G * M_g / c**2
    beta = (l_h / r_s)**2
    x = brentq(lambda x: x * x * (x - 1.0) + beta, 2.0 / 3.0, 1.0,
               xtol=np.nextafter(0.0, 1.0), rtol=8 * np.finfo(float).eps)
    return r_s * x


def hawking_T(M_g):
    """Exact Hayward Hawking temperature in Kelvin; 0 if no horizon."""
    r = horizon(M_g)
    if r is None:
        return 0.0
    r_crit = math.sqrt(3.0) * l_h
    # On a horizon f'=(r^2-3l^2)/r^3. Factoring the numerator gives exact
    # zero at the represented double root and avoids powers of a huge radius.
    fprime = ((r - r_crit) / r) * ((r + r_crit) / r) / r
    return (hbar * c / k_B) * fprime / (4 * math.pi)


def schwarzschild_T(M_g):
    M_g = _validated_mass(M_g)
    if M_g == 0:
        raise ValueError("Schwarzschild temperature requires positive mass")
    return hbar * c**3 / (8 * math.pi * G * M_g * k_B)


def main():
    print("=" * 92)
    print("  NVG HAYWARD TEMPERATURE / HORIZON MAP  (exact T_H on the QCD anchor)")
    print("=" * 92)
    print(f"  rho_c = {rho_c:.3e} g/cm^3   l = {l_h/1e5:.4f} km   "
          f"M_crit = {M_CRIT/M_sun:.4f} M_sun")

    # Sanity checks for the exact metric implementation (not evidence tests).
    for M in (1e3 * M_CRIT, 1e6 * M_CRIT):
        ratio = hawking_T(M) / schwarzschild_T(M)
        if abs(ratio - 1) >= 1e-3:
            raise RuntimeError(f"Schwarzschild limit broken: {ratio}")
    if hawking_T(M_CRIT * 1.0000001) >= 1e-3 * schwarzschild_T(M_CRIT):
        raise RuntimeError("extremal Hayward temperature check failed")

    print("-" * 92)
    print(f"  {'M/M_crit':>10} {'M (M_sun)':>12} {'r_+ (km)':>10} {'T_H (K)':>12} "
          f"{'T_Schw (K)':>12} {'T_H/T_Schw':>11}")
    for x in (1.0001, 1.001, 1.01, 1.05, 1.2, 2.0, 5.0, 30.0, 63.0, 1e4, 1e9):
        M = x * M_CRIT
        r = horizon(M)
        T = hawking_T(M)
        Ts = schwarzschild_T(M)
        print(f"  {x:>10.4g} {M/M_sun:>12.4g} {r/1e5:>10.3f} {T:>12.3e} "
              f"{Ts:>12.3e} {T/Ts:>11.4f}")

    # global temperature ceiling
    res = minimize_scalar(lambda lx: -hawking_T(math.exp(lx) * M_CRIT),
                          bounds=(math.log(1.0001), math.log(50.0)), method="bounded")
    M_peak = math.exp(res.x) * M_CRIT
    T_max = -res.fun
    print("-" * 92)
    print(f"  TEMPERATURE CEILING: T_max = {T_max:.3e} K at M = {M_peak/M_sun:.3f} M_sun "
          f"({M_peak/M_CRIT:.3f} M_crit)")
    print(f"  T_CMB today = 2.725 K  ->  T_max/T_CMB = {T_max/2.725:.1e}")
    t_cross_gyr = 17.5 * math.log(2.725 / T_max)   # Lambda-era e-folding H^-1 ~ 17.5 Gyr
    print(f"  Declared Lambda-era timescale places CMB crossing at ~{t_cross_gyr:.0f} Gyr")
    print("  in this model comparison; a cosmological mass-loss history is not solved here.")
    print("  Masses below M_crit are routed to the horizonless Hayward branch, so")
    print("  standard-PBH evaporation limits are not automatically transferable.")

    print("-" * 92)
    print("  SUB-M_crit REMNANTS AS MACRO DARK MATTER (the asteroid-window rung):")
    print(f"  {'M (g)':>10} {'r_0 (cm)':>10} {'GM/(c^2 r_0)':>13} {'sigma/M (cm^2/g)':>17}")
    for M in (1e15, 1e17, 1e20, 1e23):
        r0 = (3 * M / (4 * math.pi * rho_c)) ** (1 / 3)
        compact = G * M / (c**2 * r0)
        sig_m = math.pi * r0**2 / M
        print(f"  {M:>10.0e} {r0:>10.3f} {compact:>13.2e} {sig_m:>17.2e}")
    print("  These geometric ratios are model sensitivities only; no macro-DM or")
    print("  microlensing likelihood is evaluated for the horizonless branch.")

    print("-" * 92)
    print("  CONSUMER NOTE: the PBH ladder must route sub-M_crit objects through")
    print("  the horizonless-remnant branch; standard evaporation benchmarks are only")
    print("  comparison inputs and do not apply to this Hayward model.")
    print("-" * 92)
    print("  FALSIFIER PROTOCOLS (not evaluated here):")
    print("   * independently established PBH evaporation burst incompatible with the computed")
    print("     temperature ceiling -> falsifies this fixed-anchor Hayward hypothesis")
    print("   * independently established Hawking component in the MeV gamma background")
    print("     incompatible with that ceiling -> falsifies this fixed-anchor Hayward hypothesis")
    print(f"   * independently established black-hole horizon with mass below computed M_crit = "
          f"{M_CRIT/M_sun:.6f} M_sun -> falsifies this fixed-anchor Hayward hypothesis")
    print("  Alternative core-density scales and other NVG realizations require their own")
    print("  predictions; these protocols do not test all such hypotheses together.")
    print("  EVIDENCE STATUS: MODEL_DERIVED_NO_EVAPORATION_LIKELIHOOD")
    print("  The falsifiers above are test protocols; no current-data likelihood is evaluated.")
    print("=" * 92)
    return {
        "M_crit_g": float(M_CRIT),
        "M_peak_g": float(M_peak),
        "T_max_K": float(T_max),
        "evidence_status": "MODEL_DERIVED_NO_EVAPORATION_LIKELIHOOD",
        "observed_likelihood": None,
        "limitation": (
            "No cosmological CMB/mass-loss history, PBH formation model, or "
            "dark-matter detector likelihood is present."
        ),
    }


if __name__ == "__main__":
    main()
