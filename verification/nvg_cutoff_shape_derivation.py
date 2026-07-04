#!/usr/bin/env python3
"""
NVG: what cutoff SHAPE does the action actually predict?
=========================================================
The IR cutoff scale k_c = 1/R_H0 is derived from the horizon chain; the
SHAPE exp(-(k_c/k)^8) was so far selected by scan (TT prefers the sharpest,
Delta chi^2 = +1.77; TE adds +0.75). This script confronts the two
action-level shape candidates with data and geometry:

CANDIDATE A — closed-instanton geometry (hard cut).
  A closed bounce patch quantizes perturbations on S^3: physical modes
  start at k_3 = sqrt(8)/R_curv, with NOTHING below — an infinitely hard
  cutoff (the n -> infinity limit of the scanned family, which is exactly
  where the scan pointed). BUT the scale is then the comoving CURVATURE
  radius, and identifying the observed cutoff k_c = 2.425e-4 Mpc^-1 with
  k_3 fixes R_curv = sqrt(8)/k_c, i.e. a closed universe with
      Omega_k = -(R_H0 / R_curv)^2  ~  -0.125,
  while Planck+BAO give Omega_k = 0.0007 +/- 0.0019. Computed below:
  excluded at the ~65 sigma level. For geometry to hide within the
  Omega_k bound, the geometric cut must sit at ell < 1 — unobservable.
  VERDICT: the "hard geometric boundary" reading of the low-ell deficit
  is quantitatively DEAD, despite matching the preferred sharpness.

CANDIDATE B — causal (finite-patch) white noise.
  Modes longer than the causally prepared patch carry uncorrelated
  (white-noise) amplitudes, P_zeta(k) -> const, i.e. the dimensionless
  power falls as Delta^2 ~ k^3 below k_c. That is the 'cubic' shape
  1/(1 + (k_c/k)^3) already present in the scan family — the only shape
  in the family with an action-level derivation.

The script recomputes the low-ell TT Gamma-likelihood for {no cutoff,
cubic, sharp8} at fixed Planck background and reports whether the data
can distinguish the DERIVED causal shape from the phenomenologically
preferred near-hard one.
"""

from __future__ import annotations
import math

from nvg_cmb_lowl_refit import (load_planck_tt, run_camb, chi2_lowl,
                                KC_NVG, H0_PLANCK, OMCH2, R_H0_MPC)

OMEGA_K_PLANCK, OMEGA_K_ERR = 0.0007, 0.0019


def main():
    print("=" * 78)
    print("  NVG: CUTOFF SHAPE FROM THE ACTION — GEOMETRY vs CAUSALITY vs DATA")
    print("=" * 78)

    # ── Candidate A: geometric hard cut vs Omega_k ──────────────────────
    r_curv = math.sqrt(8.0) / KC_NVG
    omega_k = -(R_H0_MPC / r_curv) ** 2
    nsig = abs(omega_k - OMEGA_K_PLANCK) / OMEGA_K_ERR
    r_curv_min = R_H0_MPC / math.sqrt(5e-3)
    k3_max = math.sqrt(8.0) / r_curv_min
    ell_geo = k3_max * 13800.0
    print(f"\nA. Closed-instanton hard cut:")
    print(f"   k_3 = sqrt(8)/R_curv = k_c  =>  R_curv = {r_curv:,.0f} Mpc")
    print(f"   => Omega_k = {omega_k:+.3f} vs Planck+BAO {OMEGA_K_PLANCK} +/- {OMEGA_K_ERR}")
    print(f"   => excluded at {nsig:.0f} sigma.")
    print(f"   Conversely, |Omega_k| < 0.005 forces R_curv > {r_curv_min:,.0f} Mpc,")
    print(f"   i.e. a geometric cut at ell < {ell_geo:.1f} — below the quadrupole,")
    print(f"   unobservable. The geometric reading of the deficit is CLOSED.")

    # ── Candidate B: causal k^3 vs the phenomenological sharp8 ──────────
    print(f"\nB. Causal white-noise shape (cubic, the derived candidate) vs data:")
    ell, Dl, sig, dm, dp = load_planck_tt()
    chi2 = {}
    for label, kc, shape in (("no cutoff", 0.0, 'exp2'),
                             ("cubic (derived)", KC_NVG, 'cubic'),
                             ("sharp8 (scan best)", KC_NVG, 'sharp8')):
        Dl_m = run_camb(H0_PLANCK, OMCH2, kc, shape=shape)
        chi2[label] = chi2_lowl(Dl_m, ell, Dl)
        print(f"   {label:<20} low-ell chi2_Gamma = {chi2[label]:.2f}")

    d_cubic = chi2["no cutoff"] - chi2["cubic (derived)"]
    d_sharp = chi2["no cutoff"] - chi2["sharp8 (scan best)"]
    d_between = chi2["cubic (derived)"] - chi2["sharp8 (scan best)"]

    print(f"""
   Delta chi^2 vs no cutoff:  cubic {d_cubic:+.2f},  sharp8 {d_sharp:+.2f}
   cubic vs sharp8:           {d_between:+.2f}

VERDICT:
  - The hard geometric boundary — the natural reading of the scan's
    sharp8 preference — is excluded by Omega_k at the {nsig:.0f}-sigma level.
  - The surviving action-level shape is the CAUSAL k^3 suppression
    ('cubic'), and the data separate it from the near-hard shape by only
    |Delta chi^2| = {abs(d_between):.2f} — indistinguishable at current precision.
  - Honest status: the derived shape is k^3 (causality); the scan's
    sharp8 preference over it is not significant; LiteBIRD-class low-ell
    polarization is the realistic discriminator.
""")
    print("=" * 78)

    assert nsig > 20.0, "geometric candidate should be decisively excluded"
    assert abs(d_between) < 6.0, "shape discrimination should be weak at low ell"


if __name__ == "__main__":
    main()
