#!/usr/bin/env python3
"""
NVG EOS, fork A: imposed kappa-melting WITH its energy cost — kappa_1
pinned by nuclear saturation
======================================================================
Fork A keeps the kappa-law as a separate density-driven melting force,
but bookkeeps its condensate cost (audit bookkeeping B).  The audit
showed the binding depth on the saturation locus is then controlled
almost solely by kappa_1 — so saturation itself PINS the melting:

    E/A(p = 0 locus) = -16 MeV   ==>   kappa_1 = kappa_1*(c_s)

This script solves the pin exactly, checks its stability against the
residual c_s freedom, builds the full beta-equilibrated EOS at the
pinned value, and propagates to the observables:

  - existence gate (no self-bound u,d phase), causality, p(2 n_0);
  - crude TOV (M_max, R_1.4; no crust, +-0.4 km);
  - the pinned instantaneous VMF melt at 2 n_0 and a folded dielectron
    line-shape template.  No observed HADES likelihood is available here.

The interesting outcome either way: fork A converts the melting from a
fitted dial into a saturation-derived constant, and the meson shift
becomes a sharp, small, falsifiable number.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import fsolve

import nvg_eos_beta_saturated_vector as base
from nvg_eos_symmetric_rebuild import sym_state, u_cost_sym

N0, M_NUC = base.n_0, base.M_N
E_BIND, J_SYM = 16.0, 32.0
MU_2FL = 930.0
K2 = 0.8


def pressure_to_energy_checked(pressure, pressure_grid, energy_grid):
    """Interpolate EOS energy only on its domain; vacuum tail is explicit."""
    try:
        pressure = float(pressure)
    except (TypeError, ValueError) as exc:
        raise ValueError("pressure must be a finite scalar") from exc
    p_grid = np.asarray(pressure_grid, dtype=float)
    e_grid = np.asarray(energy_grid, dtype=float)
    if (not np.isfinite(pressure) or pressure < 0.0 or
            p_grid.ndim != 1 or e_grid.ndim != 1 or p_grid.size < 2 or
            p_grid.size != e_grid.size or not np.isfinite(p_grid).all() or
            not np.isfinite(e_grid).all() or np.any(np.diff(p_grid) <= 0.0)):
        raise ValueError("pressure lookup has an invalid EOS domain")
    if pressure < p_grid[0]:
        return 0.0  # explicit vacuum continuation, not np.interp clamping
    if pressure > p_grid[-1]:
        raise ValueError(
            f"pressure {pressure:g} is outside EOS domain "
            f"[{p_grid[0]:g}, {p_grid[-1]:g}]"
        )
    return float(np.interp(pressure, p_grid, e_grid))


def hades_shape_summary(pole_mev: float) -> dict[str, float | str]:
    """Fold the solved pole through the preregistered HADES template.

    This is a forward line-shape calculation only.  There is no observed
    HADES likelihood here, and no conversion from an instantaneous shift to a
    preferred peak is permitted.
    """

    import nvg_hades_lineshape_feasibility as hades

    masses = np.linspace(hades.M_LO, hades.M_HI, 121)
    shape = hades.template(masses, float(pole_mev), 20.0)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    norm = float(integrate(shape, masses))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("HADES template has non-positive normalization")
    centroid = float(integrate(masses * shape, masses) / norm)
    return {"pole_mev": float(pole_mev), "centroid_mev": centroid,
            "status": "FORWARD_ONLY_NO_HADES_LIKELIHOOD"}


def pin_kappa1(c_s, k2=K2):
    """Solve (kappa_1, c_omega) from E/A = -16 and p = 0 at n_0."""
    n_loc = np.linspace(0.75 * N0, 1.05 * N0, 9)

    def resid(x):
        k1, c_om = abs(x[0]), abs(x[1])
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

    for guess in ([0.02, 700.0], [0.05, 900.0], [0.01, 500.0]):
        x, info, ier, _ = fsolve(resid, guess, full_output=True,
                                 xtol=1e-10, maxfev=400)
        if ier == 1 and 0.0 < abs(x[0]) < 0.2 and 50 < abs(x[1]) < 4000:
            return abs(x[0]), abs(x[1])
    return None, None


def main():
    print("=" * 78)
    print("  NVG EOS FORK A: KAPPA-MELTING PINNED BY SATURATION")
    print("=" * 78)

    print(f"\n  1. The pin kappa_1*(c_s)  [kappa_2 = {K2}]:")
    pins = {}
    for c_s in (300.0, 600.0, 900.0, 1200.0):
        k1, c_om = pin_kappa1(c_s)
        pins[c_s] = (k1, c_om)
        if k1 is None:
            print(f"     c_s = {c_s:5.0f}:  no pin found")
        else:
            print(f"     c_s = {c_s:5.0f}:  kappa_1* = {k1:.4f}, "
                  f"c_omega = {c_om:.0f}")
    vals = [v[0] for v in pins.values() if v[0] is not None]
    if not vals:
        print("  pin failed everywhere — stop")
        return
    k1_lo, k1_hi = min(vals), max(vals)
    print(f"     stability: kappa_1* in [{k1_lo:.4f}, {k1_hi:.4f}] across c_s")

    # canonical fork-A point
    c_s = 600.0
    k1, c_om = pins[c_s]
    st0 = sym_state(N0, k1, K2, c_s)
    kf = base.kf_from_density(N0 / 2)
    j_kin = kf ** 2 / (6.0 * math.sqrt(kf ** 2 + st0["m_d"] ** 2))
    c_rho = 2.0 * (J_SYM - j_kin) / N0
    print(f"\n  2. Canonical fork-A point: c_s = {c_s:.0f}, "
          f"kappa_1* = {k1:.4f}, c_omega = {c_om:.0f}, c_rho = {c_rho:.0f}")
    print(f"     m*(n_0) = {st0['m_d']:.0f} MeV")

    # full beta EOS with melting cost
    narr = np.geomspace(0.05 * N0, 8.0 * N0, 320)
    states = [base.beta_equilibrium_state(n, k1, K2, c_s, c_rho)
              for n in narr]
    mb = np.array([base.M_base(n, k1, K2) for n in narr])
    dmb = np.gradient(mb, narr)
    ns = np.array([
        (base.fermion_scalar_density(st["n_n"], st["m_dirac"]) +
         base.fermion_scalar_density(st["n_p"], st["m_dirac"]))
        if st else np.nan for st in states])
    ok0 = np.isfinite(ns)
    ns_f = np.interp(narr, narr[ok0], ns[ok0])
    integ = ns_f * (-dmb)
    U = np.concatenate([[0.0], np.cumsum(
        0.5 * (integ[1:] + integ[:-1]) * np.diff(narr))])
    eps = np.array([
        (st["eps_no_vec"] + U[i] + 0.5 * c_om * narr[i] ** 2)
        if st else np.nan for i, st in enumerate(states)])
    ok = np.isfinite(eps)
    n, e = narr[ok], eps[ok]
    epA = e / n
    p = n ** 2 * np.gradient(epA, n)
    cs2 = np.gradient(p, n) / np.maximum(np.gradient(e, n), 1e-12)

    i0 = np.argmin(abs(n - N0))
    i2 = np.argmin(abs(n - 2 * N0))
    sel = n > 1.5 * N0
    exist = not ((p[sel] <= 0) & (epA[sel] < MU_2FL)).any()
    caus = bool((cs2[p > 1.0] <= 1.05).all())
    print(f"\n  3. Beta EOS: p(n_0) = {p[i0]:+.2f}, p(2n_0) = {p[i2]:.1f}; "
          f"existence_gate={'ok' if exist else 'fail'}, "
          f"causality_gate={'ok' if caus else 'fail'}")

    # crude TOV
    from scipy.integrate import solve_ivp
    good = p > 0.05
    p_s, e_s = p[good], e[good]
    idx = np.argsort(p_s)
    p_s, e_s = p_s[idx], e_s[idx]
    conv = 1.3234e-6

    if p_s.size < 2 or not np.isfinite(p_s).all() or not np.isfinite(e_s).all():
        print("     TOV unsupported: EOS pressure/energy table is not finite")
        return
    pc_hi = min(1200.0, float(p_s[-1]))
    if pc_hi < 20.0:
        print(f"     TOV unsupported: central-pressure range [20, {pc_hi:.3g}] is empty")
        return

    def rhs(r, y):
        m, pp = y
        if not np.isfinite(pp):
            raise ValueError("non-finite pressure during TOV integration")
        if pp <= 0.0:
            return [0.0, 0.0]
        ee = pressure_to_energy_checked(pp, p_s, e_s) * conv
        pk = pp * conv
        return [4 * math.pi * r ** 2 * ee,
                -(ee + pk) * (m + 4 * math.pi * r ** 3 * pk) /
                (r * (r - 2 * m) + 1e-30) / conv]

    res = []
    for pc in np.geomspace(20, pc_hi, 24):
        sol = solve_ivp(rhs, [1e-6, 30], [0.0, pc], max_step=0.02,
                        events=lambda r, y: y[1] - p_s[0] * 1.01,
                        rtol=1e-6, atol=1e-9)
        if sol.t_events[0].size:
            res.append((float(sol.y_events[0][0][0]) / 1.4766,
                        float(sol.t_events[0][0])))
    if not res:
        print("     TOV unsupported: no resolved central-pressure solution")
        return
    ms = np.array([m for m, _ in res])
    rs = np.array([r for _, r in res])
    im = int(np.argmax(ms))
    r14 = (float(np.interp(1.4, ms[:im+1], rs[:im+1]))
           if ms[:im+1].max() >= 1.4 else float("nan"))
    print(f"     TOV (no crust, +-0.4 km): M_max = {ms.max():.2f}, "
          f"R_1.4 ~ {r14:.2f} km")

    # Expose the raw solved pole and fold it through the independent
    # preregistered line-shape calculation.  The former post-hoc peak anchor
    # is intentionally absent.
    w2 = (1 + K2 * 2) ** (-k1 / K2)
    inst = 100 * (1 - w2)
    rho_pole = 80.0 + (775.3 - 80.0) * w2
    line_shape = hades_shape_summary(rho_pole)
    print(f"\n  4. PINNED melting at 2n_0: {inst:.2f}% instantaneous "
          f"(kappa-universality)")
    print(f"     raw rho pole: {line_shape['pole_mev']:.1f} MeV; "
          f"folded template centroid: {line_shape['centroid_mev']:.1f} MeV; "
          f"status={line_shape['status']}")
    print(f"""
  READING: fork A's instantaneous melt and rho pole are derived from its
  saturation solve. The folded line shape is a forward template only;
  acceptance-corrected HADES data and a likelihood are still required.
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
