#!/usr/bin/env python3
"""
NVG: the DESI DR3 binary test — w = -1 exactly, and what DR3 does to it
========================================================================
After the melting-sector retirement, NVG's dark-energy prediction is not
"some w(z)" — it is SHARP. Both dark-sector modes are heavy at the vacuum:
the radial mode m_W = sqrt(2 lambda_v) W_0 = 1.22 GeV and the phase is the
eta' (958 MeV; theta-sector audit). A field with Compton time ~1e-24 s
cannot roll on Hubble times, so the residual vacuum energy has
    w = -1 identically, at all z.
This turns DESI into a clean binary test of the whole dark sector:
any CONFIRMED w(z) != -1 falsifies it outright — no parameter to adjust.

This script quantifies the current status and the DR3 decision table:
  - current tension of w = -1 against the DESI DR2 (w0, wa) contour
    (Gaussian approximation, same covariance as nvg_dark_energy_desi.py);
  - forecast: how the exclusion scales if DR3 shrinks errors by a factor
    s with the central value (a) unchanged, (b) drifting halfway to
    LCDM, (c) drifting fully to LCDM;
  - the survival threshold: how far the DR3 central value must move
    toward (-1, 0) for NVG to remain viable at 2 sigma.
"""

from __future__ import annotations
import math
import numpy as np

W0_D, WA_D = -0.730, -0.680
S_W0, S_WA, RHO = 0.057, 0.200, -0.85
COV = np.array([[S_W0**2, RHO*S_W0*S_WA], [RHO*S_W0*S_WA, S_WA**2]])
ICOV = np.linalg.inv(COV)
LCDM = np.array([-1.0, 0.0])
CENTER = np.array([W0_D, WA_D])


def chi2_of(point, center, shrink=1.0):
    d = np.asarray(point) - np.asarray(center)
    return float(d @ (ICOV * shrink ** 2) @ d)


def sigma_equiv(chi2, dof=2):
    """Gaussian-equivalent significance of a chi^2 with dof=2."""
    p = math.exp(-chi2 / 2.0)                     # exact for 2 dof
    if p < 1e-15:
        return math.sqrt(chi2)                    # asymptotic
    # invert two-sided normal
    from math import erfc
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if erfc(mid / math.sqrt(2.0)) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    print("=" * 78)
    print("  NVG: DESI DR3 BINARY TEST OF w = -1 (the sharpened dark sector)")
    print("=" * 78)

    print(f"\n  NVG prediction: w = -1 identically (m_W = 1.22 GeV, theta = eta'")
    print(f"  — no light mode can roll; no adjustable dark-energy parameter).")

    c_now = chi2_of(LCDM, CENTER)
    print(f"\n  Current status vs DESI DR2 (Gaussian contour):")
    print(f"    chi2(w = -1) = {c_now:.1f}  ->  {sigma_equiv(c_now):.1f} sigma disfavored")
    print(f"    (Gaussian approximation; the collaboration's full likelihood")
    print(f"     quotes 2.8-4.2 sigma depending on the SNe set — same message.)")

    print(f"\n  DR3 decision table (error shrink s, central drift d toward LCDM):")
    print(f"    {'s':>5} {'drift':>22} {'chi2':>7} {'sigma':>6}   verdict for NVG")
    for s in (1.2, 1.4, 1.7):
        for label, frac in (("unchanged", 0.0), ("halfway to LCDM", 0.5),
                            ("fully to LCDM", 1.0)):
            center = CENTER + frac * (LCDM - CENTER)
            c = chi2_of(LCDM, center, shrink=s)
            sig = sigma_equiv(c)
            verdict = ("DEAD" if sig > 5 else
                       "critical" if sig > 3 else
                       "viable" if sig < 2 else "strained")
            print(f"    {s:>5.1f} {label:>22} {c:>7.1f} {sig:>6.1f}   {verdict}")

    # survival threshold at DR2-like errors shrunk x1.4
    s = 1.4
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        center = CENTER + mid * (LCDM - CENTER)
        if sigma_equiv(chi2_of(LCDM, center, shrink=s)) > 2.0:
            lo = mid
        else:
            hi = mid
    frac_needed = 0.5 * (lo + hi)

    print(f"""
  SURVIVAL THRESHOLD: with DR3-scale errors (x1.4 tighter), the central
  value must move {100*frac_needed:.0f}% of the way from the DR2 best fit toward
  (-1, 0) for NVG to remain 2-sigma viable. Anything less kills the dark
  sector with no parameter to hide behind — a genuinely binary outcome.
""")
    print("=" * 78)

    assert 3.5 < sigma_equiv(c_now) < 6.0
    assert 0.3 < frac_needed < 0.95


if __name__ == "__main__":
    main()
