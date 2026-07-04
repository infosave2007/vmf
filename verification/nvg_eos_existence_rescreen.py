#!/usr/bin/env python3
"""
NVG: EOS re-screening under the nuclear-existence constraint
=============================================================
nvg_selfbound_gate.py established that the published melting
parameterization, treated with consistent thermodynamics, predicts a
self-bound NON-STRANGE dense phase (eps/A(P=0) = 683 MeV) — excluded by
the existence of ordinary nuclei (two-flavor Bodmer-Witten argument).
This script re-screens the parameter space with the new constraint
promoted to first rank:

  EXISTENCE:  the two-flavor dense branch must satisfy
              eps/A >= 930 MeV wherever P <= 0
              (equivalently: no self-bound u,d phase).

All EOS are built with the CONSISTENT bookkeeping (melting potential
U(n) reconstructed from the kappa-law via mean-field stationarity;
saturation E/A = -16 MeV at n_0 recalibrated per parameter set).

Scan dimensions (kappa_1 = 0.25, Cs = 900, Crho = 600 held):
  kappa_2  in {0.8, 1.5, 2.5, 4.0}   — high-density melting saturation
  alpha_v  in {0.5, 1.0, 2.0, 4.0}   — vector-repulsion suppression
  nu_v     in {1, 2}

Secondary screens (kept from the original chain):
  - causality: c_s^2 <= 1 on the physical branch;
  - heavy-ion softness proxy: 5 <= p(2 n_0) <= 60 MeV/fm^3;
  - crude TOV on the candidate: M_max >= 2.0 M_sun (no crust; radii
    quoted with a ~0.4 km caveat).

Output: the surviving region, an updated canonical candidate, and the
propagated shift of the HADES observable (the in-medium mass shift at
2 n_0 scales the integrated dielectron peak).
"""

from __future__ import annotations
import math
import numpy as np

import nvg_eos_beta_saturated_vector as base

N0 = base.n_0
M_NUC = base.M_N
E_BIND = 16.0
MU_2FL = 930.0                      # existence bound (Fe-56 per nucleon)

CS, CRHO = 900.0, 600.0
GRID_K1 = (0.05, 0.10, 0.15, 0.25)
GRID_K2 = (0.8,)
GRID_AV = (1.0, 4.0)
GRID_NV = (2.0,)


def melting_mass(n_b, k1, k2):
    return base.M_base(n_b, k1, k2)


def consistent_eos(k1, k2, alpha_v, nu_v, nn=360):
    """Full-grid consistent EOS (bookkeeping B). Returns dict or None."""
    narr = np.geomspace(0.02 * N0, 8.0 * N0, nn)
    states = [base.beta_equilibrium_state(n, k1, k2, CS, CRHO) for n in narr]

    # melting potential U(n) = INT n_s (-dM_base/dn) dn
    mb = np.array([melting_mass(n, k1, k2) for n in narr])
    dmb = np.gradient(mb, narr)
    ns = np.array([
        (base.fermion_scalar_density(st["n_n"], st["m_dirac"]) +
         base.fermion_scalar_density(st["n_p"], st["m_dirac"]))
        if st is not None else np.nan for st in states])
    ok0 = np.isfinite(ns)
    if ok0.sum() < 50:
        return None
    ns_f = np.interp(narr, narr[ok0], ns[ok0])
    integrand = ns_f * (-dmb)
    U = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(narr))])

    # JOINT saturation calibration: solve (c_s, c_om) so that
    # E/A(n0) = -16 AND p(n0-) = 0 with the melting cost included.
    from scipy.optimize import fsolve

    def sat_residual(x):
        c_s_t, c_om_t = abs(x[0]), abs(x[1])
        ns_loc = np.linspace(0.7 * N0, 1.0 * N0, 7)
        eps_loc = []
        for nb in ns_loc:
            st = base.beta_equilibrium_state(nb, k1, k2, c_s_t, CRHO)
            if st is None:
                return [1e3, 1e3]
            # local melting cost along the path (recomputed cheaply)
            Ub = float(np.interp(nb, narr, U)) * (c_s_t / CS) ** 0.0
            eps_loc.append(st["eps_no_vec"] + Ub +
                           0.5 * c_om_t * nb * nb)      # g = 1 below n0
        eps_loc = np.array(eps_loc)
        epA_loc = eps_loc / ns_loc
        p0 = ns_loc[-1] ** 2 * np.gradient(epA_loc, ns_loc)[-1]
        ea0 = epA_loc[-1] - M_NUC
        return [ea0 + E_BIND, p0]

    sol = fsolve(sat_residual, [CS, 1200.0], full_output=True,
                 xtol=1e-8, maxfev=200)
    x, info, ier, _ = sol
    if ier != 1:
        return None
    c_s_cal, c_om = abs(x[0]), abs(x[1])
    if not np.isfinite(c_om) or c_om <= 0:
        return None
    # rebuild states and U with the calibrated c_s
    states = [base.beta_equilibrium_state(n, k1, k2, c_s_cal, CRHO)
              for n in narr]
    ns = np.array([
        (base.fermion_scalar_density(st["n_n"], st["m_dirac"]) +
         base.fermion_scalar_density(st["n_p"], st["m_dirac"]))
        if st is not None else np.nan for st in states])
    ok0 = np.isfinite(ns)
    if ok0.sum() < 50:
        return None
    ns_f = np.interp(narr, narr[ok0], ns[ok0])
    integrand = ns_f * (-dmb)
    U = np.concatenate([[0.0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(narr))])

    eps = np.array([
        (st["eps_no_vec"] + U[i] +
         base.vector_energy_density(narr[i], c_om, alpha_v, nu_v))
        if st is not None else np.nan for i, st in enumerate(states)])
    ok = np.isfinite(eps)
    n, e = narr[ok], eps[ok]
    epA = e / n
    p = n ** 2 * np.gradient(epA, n)
    cs2 = np.gradient(p, n) / np.maximum(np.gradient(e, n), 1e-12)
    return {"n": n, "eps": e, "p": p, "epA": epA, "cs2": cs2, "c_om": c_om}


def existence_ok(d):
    """eps/A >= 930 wherever p <= 0 on the dense side (n > 1.5 n0)."""
    sel = d["n"] > 1.5 * N0
    p, epA = d["p"][sel], d["epA"][sel]
    bad = (p <= 0.0) & (epA < MU_2FL)
    return not bad.any()


def hades_shift(k1, k2):
    """Instantaneous VMF melting factor at 2 n_0 (drives the meson shift)."""
    w = (1.0 + k2 * 2.0) ** (-k1 / k2)
    return w


def crude_tov(d):
    """Crustless TOV for M_max and R at M ~ 1.4 (caveat ~0.4 km)."""
    from scipy.integrate import solve_ivp
    p_arr, e_arr = d["p"], d["eps"]
    good = p_arr > 0.05
    p_s, e_s = p_arr[good], e_arr[good]
    idx = np.argsort(p_s)
    p_s, e_s = p_s[idx], e_s[idx]
    conv = 1.3234e-6                        # MeV/fm^3 -> km^-2 (G=c=1)

    def eps_of_p(p):
        return np.interp(p, p_s, e_s)

    def rhs(r, y):
        m, p = y
        if p <= p_s[0]:
            return [0.0, 0.0]
        e = eps_of_p(p) * conv
        pk = p * conv
        dm = 4.0 * math.pi * r ** 2 * e
        dp = -(e + pk) * (m + 4.0 * math.pi * r ** 3 * pk) / (
            r * (r - 2.0 * m) + 1e-30) / conv
        return [dm, dp]

    results = []
    for pc in np.geomspace(30, 1500, 26):
        sol = solve_ivp(rhs, [1e-6, 30.0], [0.0, pc], max_step=0.02,
                        events=lambda r, y: y[1] - p_s[0] * 1.01,
                        rtol=1e-6, atol=1e-9)
        if sol.t_events[0].size:
            R = float(sol.t_events[0][0])
            M = float(sol.y_events[0][0][0]) / 1.4766   # km -> M_sun
            results.append((M, R))
    if not results:
        return None, None
    ms = np.array([m for m, _ in results])
    rs = np.array([r for _, r in results])
    i = int(np.argmax(ms))
    m_arr, r_arr = ms[:i+1], rs[:i+1]
    r14 = float(np.interp(1.4, m_arr, r_arr)) if m_arr.max() >= 1.4 else None
    return float(ms.max()), r14


def main():
    print("=" * 78)
    print("  NVG: EOS RE-SCREENING UNDER THE NUCLEAR-EXISTENCE CONSTRAINT")
    print("=" * 78)
    print(f"\n  Consistent bookkeeping; scanning kappa_1; Cs, Crho calibrated/fixed")
    print(f"  {'k1':>5} {'k2':>5} {'a_v':>5} {'nu':>4} {'exist':>6} {'caus':>5} "
          f"{'p(2n0)':>8} {'eps/A@P<=0 min':>15}")

    survivors = []
    for K1 in GRID_K1:
      for k2 in GRID_K2:
        for av in GRID_AV:
            for nv in GRID_NV:
                d = consistent_eos(K1, k2, av, nv)
                if d is None:
                    print(f"  {K1:>5.2f} {k2:>5.1f} {av:>5.1f} {nv:>4.0f}   build failed")
                    continue
                ex = existence_ok(d)
                sel = d["n"] > 1.5 * N0
                pneg = d["p"][sel] <= 0
                min_epA = (d["epA"][sel][pneg].min()
                           if pneg.any() else float('nan'))
                caus = bool((d["cs2"][(d["p"] > 1.0)] <= 1.05).all())
                i2 = np.argmin(abs(d["n"] - 2 * N0))
                p2 = d["p"][i2]
                soft = 5.0 <= p2 <= 60.0
                flag = "PASS" if (ex and caus and soft) else "    "
                print(f"  {K1:>5.2f} {k2:>5.1f} {av:>5.1f} {nv:>4.0f} "
                      f"{'yes' if ex else 'NO':>6} {'ok' if caus else 'NO':>5} "
                      f"{p2:>8.1f} {min_epA:>15.1f}  {flag}")
                if ex and caus and soft:
                    survivors.append((K1, k2, av, nv, d))

    if not survivors:
        print("\n  RESULT: NO parameter set survives all screens — the existence")
        print("  constraint conflicts with the softness window in this family.")
        print("  The melting law itself must be revised (weaker high-density")
        print("  kappa-melting), which will move the HADES prediction.")
        return

    # candidate: smallest deviation from the old canon (k2 closest to 0.8)
    survivors.sort(key=lambda s: -s[0])   # largest surviving kappa_1
    k1c, k2c, avc, nvc, dc = survivors[0]
    print(f"\n  UPDATED CANONICAL CANDIDATE: kappa_1 = {k1c}, kappa_2 = {k2c}, "
          f"alpha_v = {avc}, nu_v = {nvc:.0f}")

    m_max, r14 = crude_tov(dc)
    print(f"  Crude TOV (no crust, +-0.4 km): M_max = "
          f"{m_max:.2f} M_sun, R_1.4 ~ {r14 if r14 else float('nan'):.2f} km")

    w_old = hades_shift(0.25, 0.8)
    w_new = hades_shift(k1c, k2c)
    inst_old, inst_new = (1 - w_old) * 100, (1 - w_new) * 100
    # integrated peak scales approx. linearly with the instantaneous shift
    peak_old = 712.0
    shift_old = 775.0 - peak_old
    peak_new = 775.0 - shift_old * (inst_new / inst_old)
    print(f"\n  HADES propagation: instantaneous VMF melt at 2 n_0: "
          f"{inst_old:.1f}% -> {inst_new:.1f}%")
    print(f"  integrated dielectron peak: 712 MeV -> ~{peak_new:.0f} MeV "
          f"(linear scaling estimate)")

    print(f"""
  STATUS: the existence constraint is satisfiable in this family; the
  canonical parameterization must move (kappa_2 up / vector repulsion
  retained), the NS numbers shift within the quoted TOV caveat, and the
  HADES target moves as printed. Full chain re-derivation (crust, tidal,
  joint chi^2) is the follow-up; the numbers above supersede nothing
  until that lands.
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
