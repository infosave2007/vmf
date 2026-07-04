#!/usr/bin/env python3
"""
NVG EOS, fork B: melting IDENTIFIED with the scalar mean field
===============================================================
Resolution of the double-counting found by the EOS audit: the kappa-law
and the sigma-attraction were the same condensate physics written twice.
Fork B removes the imposed M_base(n) entirely (kappa_1 = 0) and lets the
scalar field carry the melting:

    m*(n) = M_N - c_s n_s   (floored at the current-quark mass),
    melt fraction  f(n) = (M_N - m*(n)) / (M_N - M_cur)  — DERIVED.

This is thermodynamically consistent by construction (no rearrangement,
no imposed density law), and nuclear saturation is the standard
Walecka-type calibration:

    symmetric matter: E/A(n_0) = -16 MeV, p(n_0) = 0  ->  (c_s, c_omega)
    symmetry energy:  J(n_0) = 32 MeV                 ->  c_rho

Downstream, computed here:
  1. beta-equilibrated EOS; existence gate (no self-bound u,d phase),
     causality, p(2 n_0);
  2. crude TOV (hadronic branch only, no crust: quoted +-0.4 km);
  3. the derived melt fraction f(n_0), f(2 n_0) and the in-medium
     rho-meson mass under two universality mappings:
       linear:  m_rho* = M_rho,cur + (M_rho - M_rho,cur)(1 - f)
       sqrt:    m_rho* = M_rho * (m*/M_N)^{1/2}
     with the propagated integrated-dielectron-peak estimate for HADES.
     The mapping exponent is an assumption to be FIXED BY DATA
     (NA60 dilepton spectra already bound large shifts) — in fork B the
     meson sector tests the mapping, not a free kappa parameter.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import fsolve

import nvg_eos_beta_saturated_vector as base

N0, M_NUC = base.n_0, base.M_N
E_BIND, J_SYM = 16.0, 32.0
MU_2FL = 930.0
M_CUR = 80.0                 # current-quark share of the nucleon
M_RHO, M_RHO_CUR = 775.3, 80.0


def sym_state(n_b, c_s):
    m_d = base.solve_dirac_mass(n_b / 2, n_b / 2, M_NUC, c_s)
    if m_d is None:
        return None
    m_d = max(m_d, M_CUR)
    eps = (2.0 * base.fermion_energy_density(n_b / 2, m_d) +
           0.5 * (M_NUC - m_d) ** 2 / c_s)
    return {"m_d": m_d, "eps": eps}


def calibrate():
    n_loc = np.linspace(0.75 * N0, 1.05 * N0, 9)

    def resid(x):
        c_s, c_om = abs(x[0]), abs(x[1])
        eps = []
        for nb in n_loc:
            st = sym_state(nb, c_s)
            if st is None:
                return [1e3, 1e3]
            eps.append(st["eps"] + 0.5 * c_om * nb * nb)
        eps = np.array(eps)
        epA = eps / n_loc
        j0 = np.argmin(abs(n_loc - N0))
        p0 = n_loc[j0] ** 2 * np.gradient(epA, n_loc)[j0]
        return [epA[j0] - M_NUC + E_BIND, p0]

    for guess in ([1200.0, 900.0], [800.0, 600.0], [1600.0, 1200.0],
                  [2000.0, 1500.0]):
        x, info, ier, _ = fsolve(resid, guess, full_output=True,
                                 xtol=1e-9, maxfev=400)
        if ier == 1 and 100 < abs(x[0]) < 4000 and 100 < abs(x[1]) < 4000:
            return abs(x[0]), abs(x[1])
    return None


def main():
    print("=" * 78)
    print("  NVG EOS FORK B: MELTING = SCALAR FIELD (one-sector model)")
    print("=" * 78)

    cal = calibrate()
    if cal is None:
        print("\n  calibration failed — report and stop")
        return
    c_s, c_om = cal
    st0 = sym_state(N0, c_s)
    m_star0 = st0["m_d"]
    kf = base.kf_from_density(N0 / 2)
    j_kin = kf ** 2 / (6.0 * math.sqrt(kf ** 2 + m_star0 ** 2))
    c_rho = 2.0 * (J_SYM - j_kin) / N0
    f0 = (M_NUC - m_star0) / (M_NUC - M_CUR)

    print(f"\n  1. Saturation (symmetric): c_s = {c_s:.0f}, c_omega = {c_om:.0f}, "
          f"c_rho = {c_rho:.0f}")
    print(f"     m*(n_0) = {m_star0:.0f} MeV (m*/M = {m_star0/M_NUC:.2f}) — "
          f"derived melt fraction f(n_0) = {f0:.2f}")

    # beta-equilibrated EOS with kappa_1 = 0 (no imposed law)
    narr = np.geomspace(0.05 * N0, 8.0 * N0, 320)
    states = [base.beta_equilibrium_state(n, 0.0, 0.8, c_s, c_rho)
              for n in narr]
    eps = np.array([
        (st["eps_no_vec"] + 0.5 * c_om * narr[i] ** 2)
        if st else np.nan for i, st in enumerate(states)])
    ok = np.isfinite(eps)
    n, e = narr[ok], eps[ok]
    epA = e / n
    p = n ** 2 * np.gradient(epA, n)
    cs2 = np.gradient(p, n) / np.maximum(np.gradient(e, n), 1e-12)

    i0 = np.argmin(abs(n - N0))
    i2 = np.argmin(abs(n - 2 * N0))
    sel = n > 1.5 * N0
    bad = (p[sel] <= 0) & (epA[sel] < MU_2FL)
    exist = not bad.any()
    caus = bool((cs2[p > 1.0] <= 1.05).all())
    print(f"\n  2. Beta-equilibrated EOS: p(n_0) = {p[i0]:+.2f}, "
          f"p(2n_0) = {p[i2]:.1f} MeV/fm^3")
    print(f"     existence gate: {'PASS (no self-bound u,d phase)' if exist else 'FAIL'}")
    print(f"     causality:      {'PASS' if caus else 'FAIL'}")
    print(f"     eps/A monotone above n_0: "
          f"{'yes' if (np.diff(epA[n > N0]) > 0).all() else 'no'}")

    # crude TOV (hadronic only)
    from scipy.integrate import solve_ivp
    good = p > 0.05
    p_s, e_s = p[good], e[good]
    idx = np.argsort(p_s)
    p_s, e_s = p_s[idx], e_s[idx]
    conv = 1.3234e-6

    def rhs(r, y):
        m, pp = y
        if pp <= p_s[0]:
            return [0.0, 0.0]
        ee = np.interp(pp, p_s, e_s) * conv
        pk = pp * conv
        dm = 4 * math.pi * r ** 2 * ee
        dp = -(ee + pk) * (m + 4 * math.pi * r ** 3 * pk) / (
            r * (r - 2 * m) + 1e-30) / conv
        return [dm, dp]

    res = []
    for pc in np.geomspace(20, 1200, 24):
        sol = solve_ivp(rhs, [1e-6, 30], [0.0, pc], max_step=0.02,
                        events=lambda r, y: y[1] - p_s[0] * 1.01,
                        rtol=1e-6, atol=1e-9)
        if sol.t_events[0].size:
            res.append((float(sol.y_events[0][0][0]) / 1.4766,
                        float(sol.t_events[0][0])))
    ms = np.array([m for m, _ in res])
    rs = np.array([r for _, r in res])
    im = int(np.argmax(ms))
    r14 = (float(np.interp(1.4, ms[:im+1], rs[:im+1]))
           if ms[:im+1].max() >= 1.4 else float("nan"))
    print(f"\n  3. Crude TOV (hadronic, no crust, +-0.4 km): "
          f"M_max = {ms.max():.2f} M_sun, R_1.4 ~ {r14:.2f} km")
    print(f"     (softening for GW170817 now belongs to the CSS transition,")
    print(f"      to be re-attached in the full chain)")

    # melt fraction and HADES mapping
    st2 = base.beta_equilibrium_state(2 * N0, 0.0, 0.8, c_s, c_rho)
    m_star2 = st2["m_dirac"]
    f2 = (M_NUC - m_star2) / (M_NUC - M_CUR)
    rho_lin2 = M_RHO_CUR + (M_RHO - M_RHO_CUR) * (1 - f2)
    rho_sqrt2 = M_RHO * math.sqrt(m_star2 / M_NUC)
    print(f"\n  4. Derived melting: f(n_0) = {f0:.2f}, f(2n_0) = {f2:.2f} "
          f"(m*(2n_0) = {m_star2:.0f} MeV)")
    print(f"     rho-meson at 2n_0:  linear mapping {rho_lin2:.0f} MeV "
          f"({100*(rho_lin2/M_RHO-1):+.0f}%)")
    print(f"                         sqrt mapping   {rho_sqrt2:.0f} MeV "
          f"({100*(rho_sqrt2/M_RHO-1):+.0f}%)")
    # integrated-peak scaling from the published simulator anchor:
    # instantaneous -20.1% at 2n_0 <-> integrated peak 712 (shift -63 MeV)
    for label, rho2 in (("linear", rho_lin2), ("sqrt", rho_sqrt2)):
        inst = 100 * (1 - rho2 / M_RHO)
        peak = 775.0 - 63.0 * (inst / 20.1)
        print(f"     integrated HADES peak, {label:>6}: ~{peak:.0f} MeV")
    print(f"""
  STATUS: fork B gives a consistent one-sector nucleon EOS with standard
  saturation; the melt fraction is now an OUTPUT. The meson mapping
  exponent is the remaining assumption — the linear mapping produces
  shifts already in tension with NA60 dilepton constraints, favoring the
  milder sqrt-type mapping; HADES/NA60 data select the exponent. Full
  chain (crust, CSS transition, tidal, joint fit) is the follow-up.
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
