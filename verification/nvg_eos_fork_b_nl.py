#!/usr/bin/env python3
"""
NVG EOS, fork B (nonlinear): melting = scalar field with standard
cubic-quartic self-interactions
==================================================================
The linear model saturates but is acausal at high density and too stiff
(Walecka's known disease).  The standard cure is the sigma self-interaction
U(s) = (b/3) s^3 + (c/4) s^4 with s = M - m*.  The intended four-coupling
calibration targets four symmetric-matter observables:

    E/A(n_0) = -16 MeV,  p(n_0) = 0,  m*/M = 0.65,  K = 240 MeV.

This entry point uses a robust three-condition calibration with a fixed small
quartic coefficient and reports the resulting K as a diagnostic; it does not
claim that the four-condition empirical calibration is complete.

(c_rho from J = 32 MeV as before).  The Dirac mass follows from the
stationarity of the energy:  n_s = s/c_s + b s^2 + c s^3.

Downstream (as in nvg_eos_fork_b.py): beta-equilibrated EOS, existence
gate, causality, crude TOV, derived melt fraction and folded meson templates
for HADES (without an observational fit).
"""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import fsolve, brentq

import nvg_eos_beta_saturated_vector as base

N0, M_NUC = base.n_0, base.M_N
E_BIND, J_SYM, K_TARGET, MSTAR_T = 16.0, 32.0, 240.0, 0.65
MU_2FL = 930.0
M_CUR = 80.0
M_RHO, M_RHO_CUR = 775.3, 80.0


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
    """Fold a solved rho pole through the preregistered HADES template."""

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


def solve_s(ns_total, c_s, b, c):
    """s = M - m* from n_s = s/c_s + b s^2 + c s^3 (smallest positive root)."""
    def f(s):
        return s / c_s + b * s * s + c * s ** 3 - ns_total
    hi = M_NUC - M_CUR
    if f(1e-6) > 0:
        return 0.0
    if f(hi) < 0:
        return hi
    return brentq(f, 1e-6, hi, xtol=1e-10)


def sym_state_nl(n_b, c_s, b, c):
    """Symmetric matter with nonlinear sigma; self-consistent m*."""
    s = 0.5 * (M_NUC - M_CUR)
    for _ in range(200):
        m = M_NUC - s
        ns = 2.0 * base.fermion_scalar_density(n_b / 2, m)
        s_new = solve_s(ns, c_s, b, c)
        if abs(s_new - s) < 1e-8:
            break
        s = 0.6 * s + 0.4 * s_new
    m = M_NUC - s
    eps_sigma = s * s / (2 * c_s) + (b / 3) * s ** 3 + (c / 4) * s ** 4
    eps = 2.0 * base.fermion_energy_density(n_b / 2, m) + eps_sigma
    return {"m_d": m, "s": s, "eps": eps}


def observables(x):
    c_s, c_om, b, c = abs(x[0]), abs(x[1]), x[2], abs(x[3])
    n_loc = np.linspace(0.7 * N0, 1.1 * N0, 13)
    eps = []
    for nb in n_loc:
        st = sym_state_nl(nb, c_s, b, c)
        eps.append(st["eps"] + 0.5 * c_om * nb * nb)
    eps = np.array(eps)
    epA = eps / n_loc
    j0 = np.argmin(abs(n_loc - N0))
    p_loc = n_loc ** 2 * np.gradient(epA, n_loc)
    p0 = p_loc[j0]
    K = 9.0 * np.gradient(p_loc, n_loc)[j0]     # standard: K = 9 dp/dn at sat
    m0 = sym_state_nl(N0, c_s, b, c)["m_d"]
    return epA[j0] - M_NUC, p0, m0, K


def calibrate():
    """Unknowns scaled: y = (c_s, c_om, b*1e7, c*1e10); b may be negative."""
    def resid(y):
        x = [abs(y[0]), abs(y[1]), y[2] * 1e-7, abs(y[3]) * 1e-10]
        ea, p0, m0, K = observables(x)
        return [(ea + E_BIND) / 5.0, p0 / 2.0,
                (m0 / M_NUC - MSTAR_T) * 20.0, (K - K_TARGET) / 100.0]

    # 4-condition solve is numerically fragile (K surface from discrete
    # gradients); use the robust 3-condition calibration (E/A, p, m*)
    # with the quartic fixed small.  Report the runtime K value honestly;
    # no claim of a completed empirical calibration is made.
    C_FIX = 2e-10

    def resid3(y):
        x = [abs(y[0]), abs(y[1]), y[2] * 1e-7, C_FIX]
        ea, p0, m0, K = observables(x)
        return [(ea + E_BIND) / 5.0, p0 / 2.0,
                (m0 / M_NUC - MSTAR_T) * 20.0]

    for g in ([2600.0, 1500.0, 1.0], [2400.0, 1400.0, -2.0],
              [2800.0, 1700.0, 3.0]):
        y, info, ier, _ = fsolve(resid3, g, full_output=True,
                                 xtol=1e-10, maxfev=600)
        if ier == 1:
            return abs(y[0]), abs(y[1]), y[2] * 1e-7, C_FIX
    return None


def main():
    print("=" * 78)
    print("  NVG EOS FORK B-NL: NONLINEAR SIGMA (runtime K diagnostic)")
    print("=" * 78)

    cal = calibrate()
    if cal is None:
        print("\n  calibration failed")
        return
    c_s, c_om, b, c = cal
    ea, p0, m0, K = observables([c_s, c_om, b, c])
    kf = base.kf_from_density(N0 / 2)
    j_kin = kf ** 2 / (6.0 * math.sqrt(kf ** 2 + m0 ** 2))
    c_rho = 2.0 * (J_SYM - j_kin) / N0
    f0 = (M_NUC - m0) / (M_NUC - M_CUR)
    print(f"\n  1. Calibration: c_s = {c_s:.0f}, c_omega = {c_om:.0f}, "
          f"b = {b:.2e}, c = {c:.2e}, c_rho = {c_rho:.0f}")
    print(f"     checks: E/A = {ea:+.2f}, p = {p0:+.3f}, m*/M = {m0/M_NUC:.3f}, "
          f"K = {K:.0f} MeV")
    print(f"     derived melt fraction f(n_0) = {f0:.2f}")

    # beta-equilibrated EOS: reuse base machinery by mapping the nonlinear
    # sigma onto an effective density-dependent c_s_eff(n) = s/ns (exact
    # for the mass equation); energy computed directly here.
    narr = np.geomspace(0.05 * N0, 8.0 * N0, 300)
    eps = np.full_like(narr, np.nan)
    m_arr = np.full_like(narr, np.nan)
    for i, nb in enumerate(narr):
        # beta equilibrium: iterate y_p with nonlinear sigma
        y_p = 0.05
        for _ in range(80):
            n_p = y_p * nb
            n_n = nb - n_p
            s = 0.3 * (M_NUC - M_CUR)
            for _ in range(120):
                m = M_NUC - s
                ns = (base.fermion_scalar_density(n_n, m) +
                      base.fermion_scalar_density(n_p, m))
                s_new = solve_s(ns, c_s, b, c)
                if abs(s_new - s) < 1e-8:
                    break
                s = 0.6 * s + 0.4 * s_new
            m = M_NUC - s
            mun = math.sqrt(base.kf_from_density(n_n) ** 2 + m * m)
            mup = math.sqrt(base.kf_from_density(n_p) ** 2 + m * m)
            mue = max(mun - mup + 2 * c_rho * (n_n - n_p), base.m_e)
            n_e = base.lepton_density_from_mu(mue, base.m_e)
            n_mu = base.lepton_density_from_mu(mue, base.m_mu)
            resid_q = n_p - n_e - n_mu
            y_p = min(max(y_p - 0.3 * resid_q / nb, 1e-6), 0.4)
            if abs(resid_q) < 1e-7 * nb:
                break
        eps_sigma = s * s / (2 * c_s) + (b / 3) * s ** 3 + (c / 4) * s ** 4
        eps[i] = (base.fermion_energy_density(n_n, m) +
                  base.fermion_energy_density(n_p, m) +
                  base.fermion_energy_density(n_e, base.m_e) +
                  base.fermion_energy_density(n_mu, base.m_mu) +
                  eps_sigma + 0.5 * c_rho * (n_n - n_p) ** 2 +
                  0.5 * c_om * nb * nb)
        m_arr[i] = m
    ok = np.isfinite(eps)
    n, e, m_arr = narr[ok], eps[ok], m_arr[ok]
    epA = e / n
    p = n ** 2 * np.gradient(epA, n)
    cs2 = np.gradient(p, n) / np.maximum(np.gradient(e, n), 1e-12)

    i0 = np.argmin(abs(n - N0))
    i2 = np.argmin(abs(n - 2 * N0))
    sel = n > 1.5 * N0
    exist = not ((p[sel] <= 0) & (epA[sel] < MU_2FL)).any()
    caus = bool((cs2[p > 1.0] <= 1.05).all())
    print(f"\n  2. Beta EOS: p(n_0) = {p[i0]:+.2f}, p(2n_0) = {p[i2]:.1f}; "
          f"existence_gate={'ok' if exist else 'fail'}, "
          f"causality_gate={'ok' if caus else 'fail'}")

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
        mm, pp = y
        if not np.isfinite(pp):
            raise ValueError("non-finite pressure during TOV integration")
        if pp <= 0.0:
            return [0.0, 0.0]
        ee = pressure_to_energy_checked(pp, p_s, e_s) * conv
        pk = pp * conv
        return [4 * math.pi * r ** 2 * ee,
                -(ee + pk) * (mm + 4 * math.pi * r ** 3 * pk) /
                (r * (r - 2 * mm) + 1e-30) / conv]

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
    ms = np.array([mm for mm, _ in res])
    rs = np.array([rr for _, rr in res])
    im = int(np.argmax(ms))
    r14 = (float(np.interp(1.4, ms[:im+1], rs[:im+1]))
           if ms[:im+1].max() >= 1.4 else float("nan"))
    print(f"     TOV (no crust, +-0.4 km): M_max = {ms.max():.2f}, "
          f"R_1.4 ~ {r14:.2f} km")

    m2 = m_arr[i2]
    f2 = (M_NUC - m2) / (M_NUC - M_CUR)
    rho_lin = M_RHO_CUR + (M_RHO - M_RHO_CUR) * (1 - f2)
    rho_sqrt = M_RHO * math.sqrt(m2 / M_NUC)
    print(f"\n  3. Derived melt: f(2n_0) = {f2:.2f} (m* = {m2:.0f} MeV)")
    for label, r2 in (("linear", rho_lin), ("sqrt", rho_sqrt)):
        inst = 100 * (1 - r2 / M_RHO)
        line_shape = hades_shape_summary(r2)
        print(f"     rho at 2n_0 ({label:>6}): raw pole={line_shape['pole_mev']:.1f} MeV "
              f"(instantaneous shift {inst:+.1f}%); folded centroid="
              f"{line_shape['centroid_mev']:.1f} MeV; status={line_shape['status']}")
    print(f"""
  STATUS: fork B-NL is a nonlinear scalar-sector calibration.  Its K and
  melt fraction are runtime outputs of the calibration, not a standard
  completion.  The meson mappings and folded spectra are forward templates;
  an acceptance-corrected HADES/NA60 likelihood is required before selecting
  an exponent or architecture.
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
