#!/usr/bin/env python3
"""
NVG: machine-readable GW background template (bounce bump + cycle comb)
========================================================================
Assembles the full predicted stochastic-background spectrum from the two
derived components and writes it to verification/data/nvg_gw_template.txt:

  1. RECONDENSATION BUMP of the most recent bounce: peak at 18-42 microHz
     with Omega_GW h^2 ~ 1e-9 (central case f* = 28 microHz, Omega = 1.2e-9),
     spectral shape x^3 (7/(4+3x^2))^{7/2} — derived in
     nvg_recondensation_dynamics.py / nvg_gw_comb_amplitude.py.

  2. CYCLE COMB: replicas of the bump from earlier bounces, spaced by the
     corrected Tolman factor 2^{-1/3} = 0.794 in frequency and diluted by
     2^{4/3} = 2.52 per cycle in amplitude (nvg_tolman_law_derivation.py).
     Sub-teeth at 0.79f (40%), 0.63f (16%), 0.50f (6%), ... — the log-2
     periodicity is the fingerprint of cyclicity, distinguishing the signal
     from any single phase transition.

Sensitivity anchors for context (power-law integrated, order of magnitude):
  LISA:   Omega_GW h^2 ~ 1e-12 near 3 mHz, degrading toward 1e-9 at ~2 microHz
  muAres: Omega_GW h^2 ~ 1e-13 near 10-100 microHz (proposed)
  -> the central prediction sits 3-4 orders ABOVE the muAres floor and at
     the edge of LISA's low-frequency wall: muAres-class missions test it
     decisively; LISA marginally.

Output columns: f [Hz], Omega_GW h^2 (total), Omega (last bounce), Omega (comb tail)
"""

from __future__ import annotations
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nvg_gw_template.txt')

F_PEAK = 28e-6          # Hz, central derived exit (18-42 microHz band)
OMEGA_PEAK = 1.2e-9     # Omega_GW h^2 at peak (alpha ~ 2, beta/H ~ 900)
SPACING = 2.0 ** (-1.0 / 3.0)   # per-cycle frequency ratio (derived)
DILUTION = 2.0 ** (4.0 / 3.0)   # per-cycle amplitude dilution (derived)
N_TEETH = 12            # earlier cycles retained (beyond that: < 1e-14)


def shape(x):
    """Bubble-collision spectral shape, x = f/f_peak (envelope approximation)."""
    return x ** 3 * (7.0 / (4.0 + 3.0 * x * x)) ** 3.5


def omega_total(f):
    last = OMEGA_PEAK * shape(f / F_PEAK)
    comb = 0.0
    for k in range(1, N_TEETH + 1):
        comb += OMEGA_PEAK / DILUTION ** k * shape(f / (F_PEAK * SPACING ** k))
    return last, comb


def main():
    print("=" * 78)
    print("  NVG: GW BACKGROUND TEMPLATE (bounce bump + cycle comb)")
    print("=" * 78)
    print(f"\n  Peak: {F_PEAK*1e6:.0f} microHz, Omega h^2 = {OMEGA_PEAK:.1e}")
    print(f"  Comb: spacing {SPACING:.3f}, dilution {DILUTION:.2f} per cycle "
          f"(log-2 fingerprint)")

    fs = [10 ** (-7.0 + 4.0 * i / 400.0) for i in range(401)]   # 0.1-1000 microHz
    lines = ["# f[Hz]  Omega_total  Omega_last_bounce  Omega_comb_tail",
             f"# NVG derived template: f* = {F_PEAK:.2e} Hz, Omega* = {OMEGA_PEAK:.2e},",
             f"# comb spacing 2^(-1/3) = {SPACING:.4f}, dilution 2^(4/3) = {DILUTION:.4f}"]
    o_max, f_max = 0.0, 0.0
    for f in fs:
        last, comb = omega_total(f)
        tot = last + comb
        if tot > o_max:
            o_max, f_max = tot, f
        lines.append(f"{f:.6e}  {tot:.6e}  {last:.6e}  {comb:.6e}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write("\n".join(lines) + "\n")

    # comb visibility: tooth-to-envelope contrast at the first sub-tooth
    f1 = F_PEAK * SPACING
    last1, comb1 = omega_total(f1)
    contrast = comb1 / last1
    print(f"\n  Template written: {os.path.relpath(OUT)} ({len(fs)} points, 0.1-1000 microHz)")
    print(f"  Peak of total spectrum: {o_max:.2e} at {f_max*1e6:.0f} microHz")
    print(f"  First sub-tooth contrast (comb/envelope at 0.79 f*): {contrast:.2f}")
    print(f"\n  Mission context: muAres floor ~1e-13 (factor {OMEGA_PEAK/1e-13:.0e} below peak);")
    print(f"  LISA low-f wall ~1e-9 at 2 microHz — marginal. The log-2 comb contrast")
    print(f"  of ~{contrast:.0%} at 0.79 f* is the cyclicity discriminator.")
    print("=" * 78)

    assert 1e-10 < o_max < 1e-8
    assert abs(f_max - F_PEAK) / F_PEAK < 0.2
    assert 0.1 < contrast < 1.0


if __name__ == "__main__":
    main()
