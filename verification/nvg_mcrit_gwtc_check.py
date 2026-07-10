#!/usr/bin/env python3
"""
NVG Critical-Horizon-Mass consistency check against the GWTC catalog.

NVG prediction (parameter-free, anchored to M_Omega,0 = 859 +/- 8 MeV):
a Hayward regular object has an event horizon only above the extremal mass

    M_crit = (3 sqrt(3) / 4) * l / (G/c^2),   l = sqrt(3 c^2 / (8 pi G rho_c)),

with rho_c = M_Omega,0^4 / (hbar c)^3. Numerically M_crit ~ 0.99 M_sun, and
because M_crit ∝ M_Omega,0^{-2} the lattice error +/- 8 MeV gives the band
[0.97, 1.01] M_sun. Objects lighter than this CANNOT form a horizon in NVG:
they would be horizonless regular de Sitter remnants, not black holes.

Falsifiable content: a confirmed sub-critical (< ~1 M_sun) compact object with a
black-hole signature (ringdown / zero tidal deformability) would exclude NVG;
a horizonless ~1 M_sun remnant would support it. This script quantifies where the
current GWTC catalog sits relative to the band (a consistency / null test today).
"""
from __future__ import annotations
import csv
import math
import os

# ---- Physics: M_crit from rho_c (no cosmology, no H_0) ----------------------
RHO_C_MEV_FM3 = 7.09e4          # QCD vacuum saturation density (M_Omega^4/(hbar c)^3)
RHO_C_GEOM = RHO_C_MEV_FM3 * 1.323e-6   # -> km^-2
G_C2 = 1.476                    # G/c^2 in km / M_sun


def mcrit_msun(rho_geom: float) -> float:
    l = math.sqrt(3.0 / (8.0 * math.pi * rho_geom))       # km
    return (3.0 * math.sqrt(3.0) / 4.0) * l / G_C2


M_OMEGA = 859.0
M_CRIT = mcrit_msun(RHO_C_GEOM)
# M_crit ∝ M_Omega^{-2}: propagate the +/- 8 MeV lattice error
M_CRIT_HI = M_CRIT * (M_OMEGA / (M_OMEGA - 8.0)) ** 2      # lighter M_Omega -> larger M_crit
M_CRIT_LO = M_CRIT * (M_OMEGA / (M_OMEGA + 8.0)) ** 2
BAND = (M_CRIT_LO, M_CRIT_HI)


def _f(row: dict, key: str):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "data", "gwtc_events.csv")

    print("=" * 78)
    print("  NVG CRITICAL HORIZON MASS  vs  GWTC CATALOG")
    print("=" * 78)
    print(f"  rho_c            = {RHO_C_MEV_FM3:.3e} MeV/fm^3   (M_Omega = {M_OMEGA:.0f} MeV)")
    print(f"  l (vacuum scale) = {math.sqrt(3.0/(8.0*math.pi*RHO_C_GEOM)):.4f} km")
    print(f"  M_crit           = {M_CRIT:.4f} M_sun")
    print(f"  M_crit band      = [{BAND[0]:.4f}, {BAND[1]:.4f}] M_sun  (M_Omega +/- 8 MeV)")
    print("  Prediction: NO true (horizon-bearing) black hole below this band.")
    print("-" * 78)

    if not os.path.exists(path):
        print(f"  Catalog not found at {path}")
        return

    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            m1, m2 = _f(row, "mass_1_source"), _f(row, "mass_2_source")
            if m1 is None or m2 is None:
                continue
            # 90% credible lower bound on the LIGHTER component
            m2_lo_err = _f(row, "mass_2_source_lower")     # stored negative
            m2_lo = m2 + m2_lo_err if m2_lo_err is not None else m2
            m_final = _f(row, "final_mass_source")
            rows.append({
                "name": row.get("commonName") or row.get("id"),
                "m2": m2, "m2_lo": m2_lo, "m1": m1, "m_final": m_final,
            })

    if not rows:
        print("  No usable mass rows in catalog.")
        return

    n = len(rows)
    by_light = sorted(rows, key=lambda r: r["m2"])
    lightest = by_light[0]

    print(f"  Events with usable component masses: {n}")
    print(f"  Lightest component overall: {lightest['name']}  "
          f"m2 = {lightest['m2']:.2f} (-> 90% lower {lightest['m2_lo']:.2f}) M_sun")
    print()
    print("  Five lightest secondary components (m2, 90% lower bound):")
    print(f"    {'event':<16} {'m2':>7} {'m2_low':>8} {'margin to band top':>20}")
    for r in by_light[:5]:
        margin = r["m2_lo"] - BAND[1]
        print(f"    {r['name']:<16} {r['m2']:>7.2f} {r['m2_lo']:>8.2f} {margin:>18.2f}")

    # Any component whose 90% lower bound reaches into or below the critical band?
    breaches = [r for r in rows if r["m2_lo"] <= BAND[1]]
    subcrit = [r for r in rows if r["m2_lo"] < BAND[0]]

    print("-" * 78)
    print(f"  Components with 90% lower bound <= band top ({BAND[1]:.2f}): {len(breaches)}")
    print(f"  Components with 90% lower bound <  band bottom ({BAND[0]:.2f}): {len(subcrit)}")
    print()
    if not breaches:
        gap = lightest["m2_lo"] - BAND[1]
        print("  VERDICT: CONSISTENT (null). No catalog object reaches the critical band;")
        print(f"           the lightest sits {gap:.2f} M_sun above it. NVG's 'no sub-solar")
        print("           black hole' prediction is not yet probed by these confident events.")
        print("           Decisive test requires the O3/O4 sub-solar-mass (SSM) searches.")
    else:
        print("  VERDICT: TENSION TO CHECK. One or more components reach the critical band.")
        print("           If any is confirmed horizon-bearing (ringdown / Lambda=0), NVG is")
        print("           excluded; if horizonless, it is supported. Follow up per event:")
        for r in breaches:
            print(f"           - {r['name']}: m2 = {r['m2']:.2f} (90% low {r['m2_lo']:.2f}) M_sun")
    print("=" * 78)


if __name__ == "__main__":
    main()
