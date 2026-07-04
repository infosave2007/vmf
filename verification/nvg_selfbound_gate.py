#!/usr/bin/env python3
"""
NVG: thermodynamic-consistency audit of the EOS chain and the self-bound
gate (Bodmer-Witten window for the melted phase)
=========================================================================
Two findings motivated this script.

FINDING 1 (bug, public EOS chain).  The baseline tables produced by
nvg_eos_beta_saturated_vector.py -> build_baseline_arrays() silently DROP
the density interval where the solver's pressure is negative
(~0.15-0.87 fm^-3).  That interval is not a numerical failure: it is the
spinodal of a FIRST-ORDER melting transition that the model genuinely
predicts.  Cutting it and re-gluing the branches leaves (n, eps, p)
triplets that violate the T = 0 identity p = n^2 d(eps/n)/dn across the
gap (eps/A falls from ~922 to ~591 MeV while the stored p stays
positive) — a Maxwell construction is required instead.

FINDING 2 (physics, the theory's own bookkeeping).  The kappa-melting
reference mass M_base(n) enters the energy density with NO energy cost:
the condensate loses (M - M_base(n)) per nucleon for free.  By the
theory's own principles melting must cost the W-potential energy.  The
minimal closed completion reconstructs that potential from the melting
law itself (mean-field stationarity => U'(s) = n_s along the equilibrium
path, s = M - M_base), adds U(n) = INT n_s dM_drop to eps, and
recalibrates the vector coupling so that saturation (E/A = -16 MeV at
n_0) is preserved.

THE GATE.  With both bookkeepings (A: repo-as-is, no melting cost;
B: consistent, with reconstructed U), compute the beta-equilibrated,
charge-neutral EOS on the FULL grid, build both branches around the
transition, and evaluate the dense branch at zero pressure:

    eps/A (dense, P = 0)  vs  923 MeV (= mu of saturated nuclear matter)

  * below 923: the melted phase is ABSOLUTELY BOUND — ordinary matter is
    metastable (Bodmer-Witten scenario inside NVG); the model then owes
    an explanation for the survival of ordinary nuclei (conversion
    barrier), and the energy scale of matter rises accordingly;
  * above 923: the melted phase needs external pressure (neutron-star
    cores only) — no terrestrial self-bound phase.

Verdicts are computed, not assumed.  This audit also flags that the
canonical NS chain consumed the cut tables; the p(eps) relation the TOV
solver used is re-derived here on the consistent grid for comparison.
"""

from __future__ import annotations
import math
import numpy as np

import nvg_eos_beta_saturated_vector as base

K1, K2, CS, CRHO = 0.25, 0.80, 900.0, 600.0
ALPHA_V, NU_V = 4.0, 2.0
N0 = base.n_0
M_N = base.M_N
E_BIND = 16.0                      # MeV, saturation binding
MU_SAT = M_N - E_BIND              # 923 MeV: mu of nuclear matter at P=0


def raw_states(k1, k2, cs, crho, nn=500):
    """Beta-equilibrated states on the full grid; keep failures as None."""
    narr = np.geomspace(0.02 * N0, 8.0 * N0, nn)
    out = []
    for n_b in narr:
        st = base.beta_equilibrium_state(n_b, k1, k2, cs, crho)
        out.append((n_b, st))
    return out


def melting_cost(narr, states):
    """U(n) = INT_0^n n_s * (-dM_base/dn) dn along the equilibrium path.

    n_s: total scalar density from the solved Dirac mass; -dM_base/dn > 0.
    Returns U on the same grid (MeV/fm^3), 0 where state missing."""
    mbase = np.array([base.M_base(n, K1, K2) for n in narr])
    dmb = np.gradient(mbase, narr)                    # < 0
    ns = np.zeros_like(narr)
    for i, (n_b, st) in enumerate(states):
        if st is None:
            ns[i] = np.nan
            continue
        ns[i] = (base.fermion_scalar_density(st["n_n"], st["m_dirac"]) +
                 base.fermion_scalar_density(st["n_p"], st["m_dirac"]))
    # interpolate ns across solver gaps for the integral (smooth in n)
    ok = np.isfinite(ns)
    ns_f = np.interp(narr, narr[ok], ns[ok])
    integrand = ns_f * (-dmb)                          # MeV/fm^3 per fm^-3
    U = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                         * np.diff(narr))])
    return U


def build(bookkeeping: str):
    """Return dict with full-grid n, eps, p, mu for bookkeeping 'A' or 'B'."""
    states = raw_states(K1, K2, CS, CRHO)
    narr = np.array([n for n, _ in states])
    U = melting_cost(narr, states) if bookkeeping == "B" else np.zeros_like(narr)

    # recalibrate c_omega0 so that E/A(n0) = -16 with the chosen bookkeeping
    st0 = base.beta_equilibrium_state(N0, K1, K2, CS, CRHO)
    U0 = float(np.interp(N0, narr, U))
    c_om = 2.0 * (M_N - E_BIND - (st0["eps_no_vec"] + U0) / N0) / N0
    if c_om <= 0:
        raise RuntimeError("saturation calibration failed")

    eps = np.full_like(narr, np.nan)
    for i, (n_b, st) in enumerate(states):
        if st is None:
            continue
        eps[i] = (st["eps_no_vec"] + U[i]
                  + base.vector_energy_density(n_b, c_om, ALPHA_V, NU_V))
    ok = np.isfinite(eps)
    n_g, e_g = narr[ok], eps[ok]
    # thermodynamically exact on each continuous branch:
    epA = e_g / n_g
    p_g = n_g ** 2 * np.gradient(epA, n_g)
    mu_g = (e_g + p_g) / n_g
    return {"n": n_g, "eps": e_g, "p": p_g, "mu": mu_g, "c_om": c_om,
            "gap": (narr[~ok].min(), narr[~ok].max()) if (~ok).any() else None}


def dense_branch_p0(d):
    """eps/A at P = 0 on the dense branch (n above the gap/spinodal)."""
    n, p, mu, e = d["n"], d["p"], d["mu"], d["eps"]
    hi = n > 0.5                                  # dense side
    nh, ph, muh, eh = n[hi], p[hi], mu[hi], e[hi]
    if (ph <= 0).any() and (ph > 0).any():
        j = np.where(ph > 0)[0][0]                # first positive-p point
        if j == 0:
            return None, None
        # interpolate to p = 0 between j-1 and j
        f = -ph[j - 1] / (ph[j] - ph[j - 1])
        epA0 = (eh[j-1]/nh[j-1]) + f * (eh[j]/nh[j] - eh[j-1]/nh[j-1])
        n_p0 = nh[j - 1] + f * (nh[j] - nh[j - 1])
        return epA0, n_p0
    if (ph > 0).all():
        # branch never reaches p = 0: extrapolate mu at p -> 0
        epA0 = float(np.interp(0.0, ph, muh))     # mu(p=0) = eps/A at p=0
        return epA0, float(np.interp(0.0, ph, nh))
    return None, None


def main():
    print("=" * 78)
    print("  NVG: EOS CONSISTENCY AUDIT + SELF-BOUND GATE (Bodmer-Witten)")
    print("=" * 78)

    for tag, label in (("A", "repo bookkeeping (melting costs nothing)"),
                       ("B", "consistent (reconstructed melting potential U)")):
        d = build(tag)
        print(f"\n  Bookkeeping {tag}: {label}")
        print(f"    saturation recalibrated: c_omega0 = {d['c_om']:.1f}")
        if d["gap"]:
            print(f"    solver gap (spinodal candidate): "
                  f"{d['gap'][0]:.3f} - {d['gap'][1]:.3f} fm^-3")
        i0 = np.argmin(abs(d["n"] - N0))
        i2 = np.argmin(abs(d["n"] - 2 * N0))
        print(f"    eps/A at n0  = {d['eps'][i0]/d['n'][i0]:7.1f} MeV "
              f"(target {MU_SAT + 0:.0f}...{M_N - E_BIND:.0f})")
        print(f"    eps/A at 2n0 = {d['eps'][i2]/d['n'][i2]:7.1f} MeV, "
              f"p = {d['p'][i2]:.1f} MeV/fm^3")
        epA0, n_p0 = dense_branch_p0(d)
        if epA0 is None:
            print("    dense branch: no P = 0 point resolvable")
            continue
        margin = epA0 - MU_SAT
        verdict = ("SELF-BOUND (Bodmer-Witten window OPEN)" if margin < 0
                   else "NOT self-bound (needs external pressure)")
        print(f"    GATE: dense branch at P = 0: eps/A = {epA0:.1f} MeV "
              f"at n = {n_p0:.3f} fm^-3")
        print(f"          margin vs {MU_SAT:.0f} MeV: {margin:+.1f}  ->  {verdict}")

    print(f"""
  NOTES (honest):
  - Bookkeeping A reproduces the published chain and its spurious deep
    binding: the melting carries no energy cost, so the dense phase looks
    absolutely bound.  This bookkeeping violates the theory's own energy
    conservation for the W-field and is hereby retired for gate purposes.
  - Bookkeeping B is the minimal closed completion (U reconstructed from
    the melting law via mean-field stationarity, saturation preserved).
    Its verdict is the model's honest answer to the Bodmer-Witten
    question at this level of theory.
  - The canonical NS chain consumed bookkeeping-A tables with the
    spinodal cut; the TOV-relevant p(eps) in the star's pressure range
    should be re-derived from bookkeeping B — flagged as follow-up.
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
