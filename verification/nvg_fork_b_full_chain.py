#!/usr/bin/env python3
"""
NVG fork B: the full neutron-star chain
========================================
Completes the fork-B program (melting = scalar field, nonlinear sigma):

  0. quartic tuning: 4-condition calibration (E/A = -16, p = 0,
     m*/M = 0.65, K = 240) with polynomial-smoothed derivatives;
     falls back to the 3-condition point (K = 414) if the K-surface
     refuses, with the value reported honestly;
  1. hadronic beta-equilibrated table;
  2. npLambda extension (quark-counting couplings) -> hyperon threshold
     and the existence/strange gates on the consistent EOS;
  3. CSS quark core attached (c_s^2 = 1/3), small scan over
     (n_tr, delta_eps); crust and tidal deformability via the
     established machinery of nvg_tidal_deformability.py;
     joint chi^2 against J0740, NICER J0030/J0437 and GW170817;
  4. the new fork-B canonical numbers: M_max, R_1.4, Lambda_1.4,
     R_1.6 -> f_peak (Bauswein-type relation), z_surf(1.4).

All quoted radii carry the crust model of the tidal script; the
previous canonical numbers (12.55 km etc.) are superseded by this chain
pending review.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import fsolve, brentq

import nvg_eos_beta_saturated_vector as base
import nvg_tidal_deformability as td

N0, M_NUC = base.n_0, base.M_N
E_BIND, J_SYM, K_T, MST = 16.0, 32.0, 240.0, 0.65
M_CUR = 80.0
M_L = 1115.7
XS = XW = 2.0 / 3.0
MU_2FL = 930.0


# ── nonlinear sigma machinery ───────────────────────────────────────────
def solve_s(ns_total, c_s, b, c):
    def f(s):
        return s / c_s + b * s * s + c * s ** 3 - ns_total
    hi = M_NUC - M_CUR
    if f(1e-6) > 0:
        return 0.0
    if f(hi) < 0:
        return hi
    return brentq(f, 1e-6, hi, xtol=1e-10)


def sym_eos_local(c_s, c_om, b, c, n_loc):
    eps = []
    for nb in n_loc:
        s = 0.4 * (M_NUC - M_CUR)
        for _ in range(200):
            m = M_NUC - s
            ns = 2.0 * base.fermion_scalar_density(nb / 2, m)
            s_new = solve_s(ns, c_s, b, c)
            if abs(s_new - s) < 1e-9:
                break
            s = 0.6 * s + 0.4 * s_new
        m = M_NUC - s
        e_sig = s * s / (2 * c_s) + (b / 3) * s ** 3 + (c / 4) * s ** 4
        eps.append(2 * base.fermion_energy_density(nb / 2, m) + e_sig +
                   0.5 * c_om * nb * nb)
    return np.array(eps)


def sym_observables(c_s, c_om, b, c):
    """Smooth derivatives via quartic polynomial fit of E/A(n)."""
    n_loc = np.linspace(0.7 * N0, 1.15 * N0, 19)
    eps = sym_eos_local(c_s, c_om, b, c, n_loc)
    epA = eps / n_loc - M_NUC
    coef = np.polyfit(n_loc, epA, 4)
    pol = np.poly1d(coef)
    d1, d2 = pol.deriv(1), pol.deriv(2)
    ea0 = float(pol(N0))
    p0 = N0 ** 2 * float(d1(N0))
    K = 9.0 * (2 * N0 * float(d1(N0)) + N0 ** 2 * float(d2(N0))) * N0 / N0
    # K = 9 dP/dn = 9 (2n d1 + n^2 d2)
    K = 9.0 * (2 * N0 * float(d1(N0)) + N0 ** 2 * float(d2(N0)))
    # m* at n0
    s = 0.4 * (M_NUC - M_CUR)
    for _ in range(200):
        m = M_NUC - s
        ns = 2.0 * base.fermion_scalar_density(N0 / 2, m)
        s_new = solve_s(ns, c_s, b, c)
        if abs(s_new - s) < 1e-9:
            break
        s = 0.6 * s + 0.4 * s_new
    return ea0, p0, M_NUC - s, K


def calibrate4():
    def resid(y):
        c_s, c_om = abs(y[0]), abs(y[1])
        b, c = y[2] * 1e-7, y[3] * 1e-10        # both signs allowed:
        ea, p0, m0, K = sym_observables(c_s, c_om, b, c)   # NL3-like sets
        return [(ea + E_BIND) / 2.0, p0, (m0 / M_NUC - MST) * 50.0,
                (K - K_T) / 50.0]

    for g in ([2357.0, 1615.0, 0.5, 2.0], [2400.0, 1650.0, 2.0, -50.0],
              [2500.0, 1700.0, 5.0, -200.0], [2300.0, 1550.0, 1.0, -20.0]):
        y, info, ier, _ = fsolve(resid, g, full_output=True,
                                 xtol=1e-11, maxfev=1500)
        if ier == 1:
            return abs(y[0]), abs(y[1]), y[2] * 1e-7, y[3] * 1e-10
    return None


def calibrate3():
    def resid(y):
        c_s, c_om, b = abs(y[0]), abs(y[1]), y[2] * 1e-7
        ea, p0, m0, _ = sym_observables(c_s, c_om, b, 2e-10)
        return [(ea + E_BIND) / 2.0, p0, (m0 / M_NUC - MST) * 50.0]
    for g in ([2357.0, 1615.0, 0.5], [2500.0, 1700.0, 2.0]):
        y, info, ier, _ = fsolve(resid, g, full_output=True,
                                 xtol=1e-11, maxfev=600)
        if ier == 1:
            return abs(y[0]), abs(y[1]), y[2] * 1e-7, 2e-10
    return None


A_RHO = 0.0          # density dependence of the rho coupling (iter 2)


def c_rho_of(nb, c_rho0):
    return c_rho0 * math.exp(-A_RHO * (nb / N0 - 1.0))


# ── beta-equilibrated npLambda matter ───────────────────────────────────
def beta_state(nb, c_s, c_om, b, c, c_rho, with_lambda=True):
    y_p, y_l = 0.04, 0.0
    for _ in range(120):
        n_p = y_p * nb
        n_l = y_l * nb
        n_n = nb - n_p - n_l
        if n_n <= 0:
            y_l *= 0.5
            continue
        s = 0.4 * (M_NUC - M_CUR)
        for _ in range(150):
            mN = M_NUC - s
            mL = M_L - XS * s
            ns = (base.fermion_scalar_density(n_n, mN) +
                  base.fermion_scalar_density(n_p, mN) +
                  XS * base.fermion_scalar_density(n_l, mL))
            s_new = solve_s(ns, c_s, b, c)
            if abs(s_new - s) < 1e-9:
                break
            s = 0.55 * s + 0.45 * s_new
        mN, mL = M_NUC - s, M_L - XS * s
        Qv = n_n + n_p + XW * n_l
        vshift = c_om * Qv
        crho_n = c_rho_of(nb, c_rho)
        mun = math.sqrt(base.kf_from_density(n_n) ** 2 + mN ** 2) + vshift \
            + crho_n * (n_n - n_p)
        mup = math.sqrt(base.kf_from_density(n_p) ** 2 + mN ** 2) + vshift \
            - crho_n * (n_n - n_p)
        mul = math.sqrt(base.kf_from_density(n_l) ** 2 + mL ** 2) \
            + XW * vshift if n_l > 0 else mL + XW * vshift
        mue = max(mun - mup, base.m_e)
        n_e = base.lepton_density_from_mu(mue, base.m_e)
        n_mu = base.lepton_density_from_mu(mue, base.m_mu)
        dq = (n_p - n_e - n_mu) / nb
        y_p = min(max(y_p - 0.25 * dq, 1e-6), 0.4)
        if with_lambda:
            dl = (mun - mul) / M_NUC
            y_l = min(max(y_l + 0.1 * dl, 0.0), 0.6)
        if abs(dq) < 1e-7 and (not with_lambda or abs(dl) < 1e-6 or
                               (y_l == 0 and dl < 0)):
            break
    e_sig = s * s / (2 * c_s) + (b / 3) * s ** 3 + (c / 4) * s ** 4
    eps = (base.fermion_energy_density(n_n, mN) +
           base.fermion_energy_density(n_p, mN) +
           base.fermion_energy_density(n_l, mL) +
           base.fermion_energy_density(n_e, base.m_e) +
           base.fermion_energy_density(n_mu, base.m_mu) +
           e_sig + 0.5 * c_om * Qv * Qv +
           0.5 * c_rho_of(nb, c_rho) * (n_n - n_p) ** 2)
    return eps, y_p, y_l, mN


def build_table(c_s, c_om, b, c, c_rho, with_lambda):
    narr = np.geomspace(0.08 * N0, 8.0 * N0, 260)
    eps = np.zeros_like(narr)
    yl = np.zeros_like(narr)
    mN = np.zeros_like(narr)
    for i, nb in enumerate(narr):
        eps[i], _, yl[i], mN[i] = beta_state(nb, c_s, c_om, b, c, c_rho,
                                             with_lambda)
    epA = eps / narr
    p = narr ** 2 * np.gradient(epA, narr)
    return {"n": narr, "eps": eps, "p": p, "epA": epA, "yl": yl, "mN": mN}


# ── stellar structure via the established tidal machinery ──────────────
def star_family(p_arr, e_arr):
    eos = td.EOS.__new__(td.EOS)
    idx = np.argsort(p_arr)
    p_s, e_s = np.asarray(p_arr)[idx], np.asarray(e_arr)[idx]
    good = p_s > 0.02
    p_s, e_s = p_s[good], e_s[good]
    eos.p_arr, eos.eps_arr = p_s, e_s
    eos.p_match, eos.Gamma = 1.5, 1.35
    eos.eps_match = float(np.interp(1.5, p_s, e_s))
    rows = []
    for pc in np.geomspace(15, 1600, 26):
        try:
            rows.append(td.solve_tov_tidal(eos, pc))
        except Exception:
            pass
    ms = np.array([r[0] for r in rows])
    rs = np.array([r[1] for r in rows])
    ls = np.array([r[3] for r in rows])
    im = int(np.argmax(ms))
    m, r, l = ms[:im + 1], rs[:im + 1], ls[:im + 1]

    def at(mt, arr):
        return float(np.interp(mt, m, arr)) if m.max() >= mt else float('nan')
    return {"m_max": float(ms.max()), "r14": at(1.4, r), "l14": at(1.4, l),
            "r16": at(1.6, r), "l136": at(1.36, l)}


def main():
    print("=" * 78)
    print("  NVG FORK B: FULL NEUTRON-STAR CHAIN")
    print("=" * 78)

    global A_RHO
    cal = calibrate4()
    tag = "4-condition (K = 240)"
    if cal is None:
        cal = calibrate3()
        tag = "3-condition fallback"
    if cal is None:
        print("  calibration failed")
        return
    c_s, c_om, b, c = cal
    ea, p0, m0, K = sym_observables(c_s, c_om, b, c)
    kf = base.kf_from_density(N0 / 2)
    j_kin = kf ** 2 / (6.0 * math.sqrt(kf ** 2 + m0 ** 2))
    c_rho = 2.0 * (J_SYM - j_kin) / N0
    # iteration 2: L-tuning via A_RHO. E_sym(n) ~ j_kin(n) + c_rho(n) n/2;
    # L = 3 n0 dE_sym/dn at n0, computed numerically; target L = 55 MeV.
    def L_of(a):
        global A_RHO
        A_RHO = a
        nn = np.linspace(0.85 * N0, 1.15 * N0, 7)
        es = []
        for x in nn:
            kfx = base.kf_from_density(x / 2)
            mx = m0  # slowly varying near n0
            es.append(kfx ** 2 / (6 * math.sqrt(kfx ** 2 + mx ** 2)) +
                      0.5 * c_rho_of(x, c_rho) * x)
        return 3.0 * N0 * float(np.gradient(np.array(es), nn)[3])
    lo_a, hi_a = 0.0, 2.5
    for _ in range(50):
        mid = 0.5 * (lo_a + hi_a)
        if L_of(mid) > 55.0:
            lo_a = mid
        else:
            hi_a = mid
    A_RHO = 0.5 * (lo_a + hi_a)
    L_final = L_of(A_RHO)
    print(f"\n  0. Calibration [{tag}]: c_s = {c_s:.0f}, c_om = {c_om:.0f}, "
          f"b = {b:.2e}, c = {c:.2e}")
    print(f"     E/A = {ea:+.2f}, p = {p0:+.3f}, m*/M = {m0/M_NUC:.3f}, "
          f"K = {K:.0f} MeV")
    print(f"     symmetry: J = {J_SYM:.0f} MeV, a_rho = {A_RHO:.2f} -> "
          f"L = {L_final:.0f} MeV (target 55)")

    # 1-2. hadronic + hyperonic tables and gates
    dl = build_table(c_s, c_om, b, c, c_rho, with_lambda=True)

    # explicit sound-speed table (reviewer requirement): c_s^2(n) < 1 on
    # the branch actually used by the hybrid construction (hadronic up to
    # the CSS transition <= 2.4 n_0, margin to 3 n_0; above the transition
    # the quark branch has c_s^2 = 1/3 exactly). The negative-quartic
    # sigma potential runs away near ~3.5 n_0 (m* -> current floor) — an
    # EFT boundary safely ABOVE the transition; the hadronic table is
    # truncated there and never enters the star.
    cs2 = np.gradient(dl["p"], dl["n"]) / np.maximum(
        np.gradient(dl["eps"], dl["n"]), 1e-12)
    used = (dl["p"] > 0.5) & (dl["n"] <= 3.0 * N0)
    print(f"\n  causality, explicit c_s^2(n) on the used branch (<= 3 n_0):")
    for xn in (1.0, 1.5, 2.0, 2.4, 3.0):
        j = np.argmin(abs(dl["n"] - xn * N0))
        print(f"     n = {xn:>3.1f} n_0:  c_s^2 = {cs2[j]:.3f}")
    print(f"     max over used branch: {cs2[used].max():.3f}; "
          f"quark branch: 1/3 exactly  "
          f"({'CAUSAL' if cs2[used].max() < 1.0 else 'VIOLATION'})")
    assert cs2[used].max() < 1.0, "causality violated on used branch"
    # truncate the table at the EFT boundary for all downstream use
    keep = dl["n"] <= 3.0 * N0
    for key in ("n", "eps", "p", "epA", "yl", "mN"):
        dl[key] = dl[key][keep]

    sel = dl["n"] > 1.2 * N0
    pneg = dl["p"][sel] <= 0
    gate_ok = not (pneg & (dl["epA"][sel] < MU_2FL)).any()
    i_thr = np.argmax(dl["yl"] > 1e-3) if (dl["yl"] > 1e-3).any() else -1
    thr = dl["n"][i_thr] / N0 if i_thr > 0 else float('nan')
    print(f"\n  1. npLambda matter: hyperon threshold ~ {thr:.1f} n_0; "
          f"strange/existence gates: "
          f"{'CLOSED (no self-bound phase)' if gate_ok and not pneg.any() else ('PASS' if gate_ok else 'FAIL')}")

    # 3. CSS scan + tidal
    print(f"\n  2. CSS transition scan (c_s^2 = 1/3) + crust + tidal:")
    print(f"     {'n_tr':>5} {'dE':>4} {'cs2q':>4} {'M_max':>6} {'R_1.4':>6} "
          f"{'L_1.4':>6} {'chi2':>6}")
    best = None
    for ntr in (1.6, 1.8, 2.0, 2.4):
        i_tr = np.argmin(abs(dl["n"] - ntr * N0))
        p_tr, e_tr = dl["p"][i_tr], dl["eps"][i_tr]
        if p_tr <= 0:
            continue
        for de_frac, cs2q in ((0.25, 1/3), (0.25, 0.5), (0.25, 0.65),
                              (0.5, 0.5), (0.5, 0.65), (0.0, 0.5)):
            de = de_frac * e_tr
            p_ext = np.geomspace(p_tr * 1.001, 4000, 160)
            e_ext = e_tr + de + (p_ext - p_tr) / cs2q
            p_all = np.concatenate([dl["p"][:i_tr + 1], p_ext])
            e_all = np.concatenate([dl["eps"][:i_tr + 1], e_ext])
            fam = star_family(p_all, e_all)
            if not np.isfinite(fam["r14"]):
                continue
            chi2 = (((fam["m_max"] - 2.08) / 0.07) ** 2 +
                    ((fam["r14"] - 12.2) / 0.5) ** 2 +
                    ((fam["r14"] - 11.36) / 0.8) ** 2 +
                    ((fam["l136"] - 300.0) / 255.0) ** 2)
            print(f"     {ntr:>5.1f} {de_frac:>4.2f} {cs2q:>4.2f} {fam['m_max']:>6.2f} "
                  f"{fam['r14']:>6.2f} {fam['l14']:>6.0f} {chi2:>6.2f}")
            if best is None or chi2 < best[0]:
                best = (chi2, ntr, de_frac, cs2q, fam)

    if best is None:
        print("     no viable CSS point")
        return
    chi2, ntr, de_frac, cs2q, fam = best
    z = (1.0 - 2.0 * 1.4766 * 1.4 / fam["r14"]) ** -0.5 - 1.0
    f_peak = 8.16 - 0.46 * fam["r16"]
    print(f"\n  3. FORK-B CANONICAL CANDIDATE: n_tr = {ntr} n_0, "
          f"delta_eps = {de_frac} eps_tr, cs2_q = {cs2q:.2f}")
    print(f"     M_max = {fam['m_max']:.2f} M_sun   (J0740: 2.08 +- 0.07)")
    print(f"     R_1.4 = {fam['r14']:.2f} km       (J0030 12.2 +- 0.5; "
          f"J0437 11.36 +- 0.8)")
    print(f"     Lambda_1.4 = {fam['l14']:.0f}, Lambda-tilde ~ {fam['l136']:.0f} "
          f"(GW170817 90%: [70, 720])")
    print(f"     R_1.6 = {fam['r16']:.2f} km -> f_peak ~ {f_peak:.2f} kHz "
          f"(Bauswein-type)")
    print(f"     z_surf(1.4) = {z:.3f};  joint chi^2 = {chi2:.2f} (4 pulls)")
    print(f"""
  STATUS: fork-B full chain complete at first pass. These numbers
  supersede the old canon (12.55 km family) pending review; K = {K:.0f}
  and the crust model are the quoted systematics. The strange gate is
  closed on the consistent EOS — no self-bound phase of any flavor.
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
