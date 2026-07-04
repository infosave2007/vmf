#!/usr/bin/env python3
"""
NVG: line-shape feasibility of the architecture selector at HADES
==================================================================
The HP2026 HADES overview (S. Kim, Hard Probes 2026) establishes that in
baryon-dominated matter the rho is COMPLETELY MELTED: the thermal
dilepton excess in 0.2-0.7 GeV is an exponential continuum
dN/dM ~ M^{3/2} exp(-M/T) with kT = 72-82 MeV — there is no vacuum-like
peak whose position could be compared to a prediction.  Peak-position
selectors (all earlier letters, and the 18-MeV fork separation) are
therefore inapplicable; the physical discriminator is the SHAPE of the
excess spectrum.

This script quantifies that discriminator.  Templates for the two
consistent NVG architectures:

  fork A (residual melting ~2%): in-medium pole ~ vacuum (770 MeV),
         melted by collisional broadening (Gamma_eff ~ 350 MeV);
  fork B (scalar-field melting, NA60-bounded mapping lambda ~ 0.09):
         pole pulled to ~735 MeV at the time-averaged density
         (<n> ~ 1.5 n_0, m*/M = 0.55), same broadening.

Both are folded with thermal weight (T_eff = 80 MeV per HADES) and the
spectrometer resolution, normalized over 0.2-0.8 GeV, and compared:

  N_3sigma = 9 / chi2_per_event

gives the number of EXCESS counts needed for a 3-sigma shape
discrimination, for two resolution assumptions (sigma = 20 MeV ~ 6%
FWHM-derived, and a conservative sigma = 46 MeV).

Independence note (methodological): the melt fraction f(n) is fixed by
nuclear saturation (scalar sector); the mapping exponent lambda is
bounded by NA60 dileptons (meson sector).  The two inputs are calibrated
on disjoint data sets; their product shapes the template but neither is
tuned to HADES.
"""

from __future__ import annotations
import math
import numpy as np

T_EFF = 80.0          # MeV, HADES fireball temperature (HP2026: 72-82)
GAMMA = 350.0         # MeV, effective in-medium width (melted regime)
POLE_A = 770.0        # fork A: vacuum-like pole (rho-baryon broadening only)
POLE_B = 735.0        # fork B: (m*/M)^lambda pull at <n> ~ 1.5 n_0
M_LO, M_HI = 200.0, 800.0


def template(m, pole, sigma_res):
    """Thermal weight x relativistic BW, folded with Gaussian resolution."""
    grid = np.linspace(M_LO - 100, M_HI + 100, 1200)
    bw = grid ** 2 * GAMMA ** 2 / ((grid ** 2 - pole ** 2) ** 2 +
                                   grid ** 2 * GAMMA ** 2)
    thermal = grid ** 1.5 * np.exp(-grid / T_EFF)
    raw = bw * thermal
    out = np.zeros_like(m)
    for i, mm in enumerate(m):
        k = np.exp(-0.5 * ((grid - mm) / sigma_res) ** 2)
        out[i] = np.trapz(raw * k, grid)
    return out


def discrimination(sigma_res):
    m = np.linspace(M_LO, M_HI, 61)
    a = template(m, POLE_A, sigma_res)
    b = template(m, POLE_B, sigma_res)
    a /= np.trapz(a, m)
    b /= np.trapz(b, m)
    # chi2 per unit total count for binned shapes (Poisson, expected = a)
    dm = m[1] - m[0]
    pa, pb = a * dm, b * dm
    chi2_per_event = float(np.sum((pb - pa) ** 2 / np.maximum(pa, 1e-12)))
    n3 = 9.0 / chi2_per_event
    # where the difference lives
    j = int(np.argmax(np.abs(pb - pa) / np.sqrt(np.maximum(pa, 1e-12))))
    return n3, m[j], chi2_per_event


def main():
    print("=" * 78)
    print("  NVG: ARCHITECTURE SELECTOR AT HADES — LINE-SHAPE FEASIBILITY")
    print("=" * 78)
    print(f"\n  Observable reality (HP2026): no in-medium rho peak exists —")
    print(f"  the excess is an exponential continuum (kT = 72-82 MeV).")
    print(f"  Peak-position selection is INAPPLICABLE; shape analysis only.")
    print(f"\n  Templates: pole {POLE_A:.0f} (fork A) vs {POLE_B:.0f} MeV (fork B),")
    print(f"  Gamma = {GAMMA:.0f} MeV, T_eff = {T_EFF:.0f} MeV, window "
          f"{M_LO:.0f}-{M_HI:.0f} MeV.")

    print(f"\n  {'sigma_res':>10} {'N_excess(3 sigma)':>18} {'best-lever M':>13}")
    for sig in (20.0, 46.0):
        n3, mbest, c = discrimination(sig)
        print(f"  {sig:>10.0f} {n3:>18,.0f} {mbest:>10.0f} MeV")

    n3_20, _, _ = discrimination(20.0)
    n3_46, _, _ = discrimination(46.0)
    print(f"""
  CONTEXT: the published Au+Au 2.4 GeV excess (Nature Phys. 15, 1040)
  contains ~1e4 excess pairs; the Gen2 campaigns (2024-25, ~4.4e8
  triggered events) should increase this by an order of magnitude.
  Required {n3_20:,.0f}-{n3_46:,.0f} excess counts for a 3-sigma shape
  separation is therefore {'WITHIN' if n3_46 < 1e5 else 'BEYOND'} realistic reach —
  PROVIDED the analysis fits the full line shape, not a peak position.

  PRACTICAL CONSEQUENCE for the collaboration request: ask for the
  in-medium spectral-shape fit (pole-mass parameter free) on the Gen2
  excess spectra, not for a peak position.  The architecture selector
  survives, but only in shape space; its timeline follows the announced
  2026-27 in-medium vector-meson analysis.
""")
    print("=" * 78)

    assert n3_46 > n3_20 > 0


if __name__ == "__main__":
    main()
