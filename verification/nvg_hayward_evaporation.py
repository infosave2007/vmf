#!/usr/bin/env python3
"""
Hawking evaporation in NVG: the temperature ceiling and its consequences.

The NVG regular core fixes the Hayward length l = sqrt(3c^2/(8 pi G rho_c)) =
1.128 km from the QCD anchor rho_c = M_Omega^4/(hbar c)^3, giving the extremal
mass M_crit ~ 0.99 M_sun (README rows 35/54). This script computes what that
does to Hawking evaporation -- exactly, from the metric:

  T_H = (hbar c / k_B) * f'(r_+) / (4 pi),   f(r) = 1 - 2Mr^2/(r^3 + 2Ml^2)

Consequences derived below:
  (1) T_H(M) -> 0 as M -> M_crit^+ (extremal), and has a GLOBAL maximum T_max
      over all masses. Every black hole that can exist in NVG is colder than
      T_max ~ 1e-8 K -- categorically colder than the CMB for the lifetime of
      the current cycle. NO NVG black hole ever loses net mass to Hawking
      radiation; evaporation is switched off universe-wide.
  (2) Below M_crit there is NO horizon at all: sub-solar "PBHs" are naked
      de Sitter remnants with ZERO Hawking flux. Therefore the evaporation
      bounds that close the standard PBH dark-matter window below ~1e17 g
      (Voyager e+-, EGRET/Fermi gamma background, 511 keV) DO NOT APPLY.
      NVG dark matter may occupy masses absolutely forbidden for ordinary PBHs.
      NOTE: nvg_pbh_dark_matter.py currently applies those bounds to the NVG
      spectrum -- internally inconsistent with rows 35/54; flagged here.
  (3) Sharp falsifier, live today: a single confirmed PBH evaporation burst
      (HAWC/CTA/Fermi burst searches) or a confirmed Hawking component of the
      gamma-ray background kills the NVG regular core outright.
  (4) Viability of sub-M_crit remnants as macro dark matter: at the repo's
      abundance peak (~1e20 g) the remnant is a ~6 cm nuclear-density ball,
      compactness ~1e-9, geometric sigma/M ~ 1e-18 cm^2/g -- orders below all
      macro-DM bounds, and below the microlensing floor (~1e-11 M_sun).
      Formation abundance remains the calibrated (not derived) part.
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


def horizon(M_g):
    """Outer horizon of Hayward f(r)=1-2mr^2/(r^3+2ml^2), m=GM/c^2. None if M<M_crit."""
    m = G * M_g / c**2
    f = lambda r: 1.0 - 2 * m * r**2 / (r**3 + 2 * m * l_h**2)
    r_s = 2 * m
    if M_g <= M_CRIT:
        return None
    # outer root lies between the extremal (double-root) radius sqrt(3)*l and r_s
    return brentq(f, math.sqrt(3.0) * l_h, 1.001 * r_s, xtol=1e-12 * r_s)


def hawking_T(M_g):
    """Exact Hayward Hawking temperature in Kelvin; 0 if no horizon."""
    r = horizon(M_g)
    if r is None:
        return 0.0
    m = G * M_g / c**2
    d = r**3 + 2 * m * l_h**2
    # f'(r) = -2m[2r*d - 3r^4]/d^2 = (2m r^4 - 8 m^2 r l^2)/d^2
    fprime = (2 * m * r**4 - 8 * m**2 * r * l_h**2) / d**2
    return (hbar * c / k_B) * fprime / (4 * math.pi)


def schwarzschild_T(M_g):
    return hbar * c**3 / (8 * math.pi * G * M_g * k_B)


def main():
    print("=" * 92)
    print("  NVG HAWKING-EVAPORATION SHUTDOWN  (exact Hayward T_H on the QCD anchor)")
    print("=" * 92)
    print(f"  rho_c = {rho_c:.3e} g/cm^3   l = {l_h/1e5:.4f} km   "
          f"M_crit = {M_CRIT/M_sun:.4f} M_sun")

    # sanity: Schwarzschild limit at large M
    for M in (1e3 * M_CRIT, 1e6 * M_CRIT):
        ratio = hawking_T(M) / schwarzschild_T(M)
        assert abs(ratio - 1) < 1e-3, f"Schwarzschild limit broken: {ratio}"
    # sanity: extremal limit
    assert hawking_T(M_CRIT * 1.0000001) < 1e-3 * schwarzschild_T(M_CRIT)

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
    print(f"  CMB cools below T_max only after ~{t_cross_gyr:.0f} Gyr of Lambda expansion --")
    print(f"  far beyond the cycle turnaround; within the NVG cyclic history EVERY black")
    print(f"  hole is always colder than the radiation bath: net Hawking mass loss NEVER")
    print(f"  happens. The 'evaporating PBH' population (1e15-1e17 g, the one Voyager/")
    print(f"  EGRET/511keV/burst searches target) does not even have horizons here.")

    print("-" * 92)
    print("  SUB-M_crit REMNANTS AS MACRO DARK MATTER (the asteroid-window rung):")
    print(f"  {'M (g)':>10} {'r_0 (cm)':>10} {'GM/(c^2 r_0)':>13} {'sigma/M (cm^2/g)':>17}")
    for M in (1e15, 1e17, 1e20, 1e23):
        r0 = (3 * M / (4 * math.pi * rho_c)) ** (1 / 3)
        compact = G * M / (c**2 * r0)
        sig_m = math.pi * r0**2 / M
        print(f"  {M:>10.0e} {r0:>10.3f} {compact:>13.2e} {sig_m:>17.2e}")
    print("  All far below macro-DM bounds (sigma/M < ~1e-3 cm^2/g scale) and below the")
    print("  HSC microlensing floor (~1e-11 M_sun = 2e22 g): the range 1e10-1e17 g,")
    print("  CLOSED for ordinary PBHs by evaporation, is OPEN for NVG remnants.")

    print("-" * 92)
    print("  CONSISTENCY FLAG: verification/nvg_pbh_dark_matter.py applies the standard")
    print("  Hawking-radiation constraint (get_hawking_limit, Voyager/EGRET) to the NVG")
    print("  spectrum. Per rows 35/54 those objects are horizonless and emit nothing --")
    print("  the constraint does not apply and should be removed/relabeled there.")
    print("-" * 92)
    print("  FALSIFIERS (live, zero-cost):")
    print("   * one confirmed PBH evaporation burst (HAWC/CTA/Fermi searches)  -> NVG dead")
    print("   * confirmed Hawking component in the MeV gamma background        -> NVG dead")
    print("   * any confirmed sub-solar BLACK HOLE (horizon, e.g. ringdown)    -> NVG dead")
    print("  STATUS: consistent with all current data (all such searches are null to date).")
    print("=" * 92)


if __name__ == "__main__":
    main()
