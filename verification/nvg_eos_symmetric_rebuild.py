#!/usr/bin/env python3
"""
NVG: EOS rebuild with proper symmetric-matter calibration
==========================================================
Fixes bug #3 of the EOS audit: saturation conditions belong to SYMMETRIC
nuclear matter (y_p = 1/2, no leptons), not to beta-equilibrated matter.

Pipeline per parameter point (kappa_1, kappa_2, alpha_v, nu_v):
  1. symmetric-matter states with the melting law and its energy cost
     U_sym(n) (consistent bookkeeping);
  2. calibrate (c_s, c_omega) from E/A(n_0) = -16 MeV and p_sym(n_0) = 0;
  3. calibrate c_rho from the symmetry energy J = 32 MeV at n_0;
  4. build beta-equilibrated, charge-neutral EOS with the calibrated
     couplings and the melting cost along the beta path;
  5. screens: nuclear existence (no self-bound u,d phase below
     930 MeV/baryon), causality, heavy-ion softness 5 < p(2n_0) < 60.

Control point alpha_v = 0 (no vector suppression) isolates the culprit:
if it passes existence while alpha_v > 0 fails, the saturating-vector
softening device — not the melting — drives the spurious collapse.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import fsolve, brentq

import nvg_eos_beta_saturated_vector as base

N0, M_NUC = base.n_0, base.M_N
E_BIND, J_SYM = 16.0, 32.0
MU_2FL = 930.0

GRID = [
    (0.10, 0.8, 0.0, 2.0),
    (0.10, 0.8, 1.0, 2.0),
    (0.10, 0.8, 4.0, 2.0),
    (0.25, 0.8, 0.0, 2.0),
    (0.25, 0.8, 1.0, 2.0),
    (0.25, 0.8, 4.0, 2.0),
    (0.25, 2.5, 0.0, 2.0),
    (0.25, 2.5, 1.0, 2.0),
    (0.10, 2.5, 1.0, 2.0),
    (0.05, 0.8, 4.0, 2.0),
]


def sym_state(n_b, k1, k2, c_s):
    """Symmetric matter: n_n = n_p = n_b/2, no leptons, no rho."""
    m_ref = base.M_base(n_b, k1, k2)
    m_d = base.solve_dirac_mass(n_b / 2, n_b / 2, m_ref, c_s)
    if m_d is None:
        return None
    ns = 2.0 * base.fermion_scalar_density(n_b / 2, m_d)
    eps = (2.0 * base.fermion_energy_density(n_b / 2, m_d) +
           (0.5 * (m_ref - m_d) ** 2 / c_s if c_s > 0 else 0.0))
    return {"m_d": m_d, "ns": ns, "eps_kin_sc": eps}


def u_cost_sym(narr, k1, k2, c_s):
    mb = np.array([base.M_base(n, k1, k2) for n in narr])
    dmb = np.gradient(mb, narr)
    ns = np.array([(sym_state(n, k1, k2, c_s) or {"ns": np.nan})["ns"]
                   for n in narr])
    ok = np.isfinite(ns)
    ns_f = np.interp(narr, narr[ok], ns[ok])
    integ = ns_f * (-dmb)
    return np.concatenate([[0.0], np.cumsum(
        0.5 * (integ[1:] + integ[:-1]) * np.diff(narr))])


def calibrate(k1, k2):
    """(c_s, c_omega) from symmetric saturation with melting cost."""
    n_loc = np.linspace(0.75 * N0, 1.05 * N0, 9)

    def resid(x):
        c_s, c_om = abs(x[0]), abs(x[1])
        U = u_cost_sym(n_loc, k1, k2, c_s)
        eps = []
        for i, nb in enumerate(n_loc):
            st = sym_state(nb, k1, k2, c_s)
            if st is None:
                return [1e3, 1e3]
            eps.append(st["eps_kin_sc"] + U[i] + 0.5 * c_om * nb * nb)
        eps = np.array(eps)
        epA = eps / n_loc
        j0 = np.argmin(abs(n_loc - N0))
        p0 = n_loc[j0] ** 2 * np.gradient(epA, n_loc)[j0]
        return [epA[j0] - M_NUC + E_BIND, p0]

    best = None
    for guess in ([700.0, 900.0], [400.0, 500.0], [1100.0, 1400.0]):
        x, info, ier, _ = fsolve(resid, guess, full_output=True,
                                 xtol=1e-9, maxfev=300)
        if ier == 1:
            c_s, c_om = abs(x[0]), abs(x[1])
            if 50 < c_s < 2600 and 50 < c_om < 4000:
                best = (c_s, c_om)
                break
    return best


def c_rho_from_J(k1, k2, c_s):
    st = sym_state(N0, k1, k2, c_s)
    kf = base.kf_from_density(N0 / 2)
    ef = math.sqrt(kf ** 2 + st["m_d"] ** 2)
    j_kin = kf ** 2 / (6.0 * ef)
    c_rho = 2.0 * (J_SYM - j_kin) / N0
    return max(c_rho, 0.0), j_kin


def beta_eos(k1, k2, c_s, c_rho, c_om, av, nv, nn=340):
    narr = np.geomspace(0.05 * N0, 8.0 * N0, nn)
    states = [base.beta_equilibrium_state(n, k1, k2, c_s, c_rho)
              for n in narr]
    mb = np.array([base.M_base(n, k1, k2) for n in narr])
    dmb = np.gradient(mb, narr)
    ns = np.array([
        (base.fermion_scalar_density(st["n_n"], st["m_dirac"]) +
         base.fermion_scalar_density(st["n_p"], st["m_dirac"]))
        if st else np.nan for st in states])
    ok0 = np.isfinite(ns)
    if ok0.sum() < 40:
        return None
    ns_f = np.interp(narr, narr[ok0], ns[ok0])
    integ = ns_f * (-dmb)
    U = np.concatenate([[0.0], np.cumsum(
        0.5 * (integ[1:] + integ[:-1]) * np.diff(narr))])
    eps = np.array([
        (st["eps_no_vec"] + U[i] +
         base.vector_energy_density(narr[i], c_om, av, nv))
        if st else np.nan for i, st in enumerate(states)])
    ok = np.isfinite(eps)
    n, e = narr[ok], eps[ok]
    epA = e / n
    p = n ** 2 * np.gradient(epA, n)
    cs2 = np.gradient(p, n) / np.maximum(np.gradient(e, n), 1e-12)
    return {"n": n, "eps": e, "p": p, "epA": epA, "cs2": cs2}


def main():
    print("=" * 78)
    print("  NVG: EOS REBUILD — SYMMETRIC-MATTER CALIBRATION (bug #3 fixed)")
    print("=" * 78)
    print(f"\n  {'k1':>5} {'k2':>4} {'a_v':>4} {'c_s':>6} {'c_om':>6} {'c_rho':>6} "
          f"{'p_b(n0)':>8} {'p(2n0)':>8} {'exist':>6} {'caus':>5}")

    survivors = []
    for k1, k2, av, nv in GRID:
        cal = calibrate(k1, k2)
        if cal is None:
            print(f"  {k1:>5.2f} {k2:>4.1f} {av:>4.1f}   calibration failed")
            continue
        c_s, c_om = cal
        c_rho, _ = c_rho_from_J(k1, k2, c_s)
        d = beta_eos(k1, k2, c_s, c_rho, c_om, av, nv)
        if d is None:
            print(f"  {k1:>5.2f} {k2:>4.1f} {av:>4.1f} {c_s:>6.0f} {c_om:>6.0f} "
                  f"{c_rho:>6.0f}   beta build failed")
            continue
        i0 = np.argmin(abs(d["n"] - N0))
        i2 = np.argmin(abs(d["n"] - 2 * N0))
        pb0, p2 = d["p"][i0], d["p"][i2]
        sel = d["n"] > 1.5 * N0
        bad = (d["p"][sel] <= 0) & (d["epA"][sel] < MU_2FL)
        exist = not bad.any()
        caus = bool((d["cs2"][d["p"] > 1.0] <= 1.05).all())
        soft = 5.0 <= p2 <= 60.0
        flag = "PASS" if (exist and caus and soft) else ""
        print(f"  {k1:>5.2f} {k2:>4.1f} {av:>4.1f} {c_s:>6.0f} {c_om:>6.0f} "
              f"{c_rho:>6.0f} {pb0:>8.2f} {p2:>8.1f} "
              f"{'yes' if exist else 'NO':>6} {'ok' if caus else 'NO':>5}  {flag}")
        if exist and caus and soft:
            survivors.append((k1, k2, av, nv, c_s, c_om, c_rho, d))

    if survivors:
        print(f"\n  {len(survivors)} SURVIVOR(S). Candidate (largest k1):")
        survivors.sort(key=lambda s: -s[0])
        k1, k2, av, nv, c_s, c_om, c_rho, d = survivors[0]
        print(f"    kappa_1 = {k1}, kappa_2 = {k2}, alpha_v = {av}, "
              f"c_s = {c_s:.0f}, c_omega = {c_om:.0f}, c_rho = {c_rho:.0f}")
        w = (1 + k2 * 2) ** (-k1 / k2)
        print(f"    VMF melt factor at 2n0: {(1-w)*100:.1f}% "
              f"(was 20.1% with old canon)")
    else:
        print("\n  NO survivors: even with correct calibration the family fails —")
        print("  the softening device and/or the melting law need structural revision.")
    print("=" * 78)


if __name__ == "__main__":
    main()
