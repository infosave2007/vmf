#!/usr/bin/env python3
"""
NVG: the STRANGE gate — is hyperonic melted matter self-bound?
===============================================================
Sequel to nvg_selfbound_gate.py / nvg_eos_existence_rescreen.py.  The
two-flavor (u,d) gate must be CLOSED by the existence of nuclei; the
legitimate Bodmer-Witten question is the strange one: with hyperons
admitted, matter below eps/A = 930 MeV at P = 0 does NOT threaten nuclei
(conversion requires ~A simultaneous weak reactions), so a self-bound
strange-melted phase is allowed physics — and would set the true energy
ceiling of matter within NVG.

MODEL (minimal npLambda extension of the consistent EOS):
  - Lambda (1115.7 MeV) melts by the SAME kappa-law on its VMF fraction
    (the repo's hadron-universality claim), with a current/strange core
    M_Lambda_cur ~ 190 MeV (sensitivity scanned +-60);
  - quark-counting couplings: x_sigma = x_omega = 2/3, no rho coupling;
  - shared scalar field: m_i* = M_base_i(n) - x_i c_s n_s_eff, with
    n_s_eff = n_sN + (2/3) n_sL; shared vector charge Q_v = n_N + (2/3) n_L;
  - composition (y_p, y_Lambda) from minimizing eps at fixed n_b
    (equivalent to mu_Lambda = mu_n + charge neutrality);
  - melting-potential cost U(n) included by a two-pass iteration along
    the equilibrium trajectory (bookkeeping B of the audit);
  - pressure from the exact identity p = n^2 d(eps/n)/dn.

GATE: eps/A on the dense branch at P = 0, vs 930 MeV.
  open  (< 930): NVG predicts absolutely stable strange-melted matter —
                 the Witten hypothesis realized inside the model; the
                 conversion barrier becomes the next computation;
  closed (>= 930): no self-bound phase; condensate energy is reachable
                 only under external (gravitational) pressure.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize

import nvg_eos_beta_saturated_vector as base

N0, M_N = base.n_0, base.M_N
M_L = 1115.7
E_BIND = 16.0
GATE = 930.0
XS, XW = 2.0 / 3.0, 2.0 / 3.0


def m_base_lambda(n_b, k1, k2, m_cur_L):
    w = (1.0 + k2 * (n_b / N0)) ** (-k1 / k2)
    return m_cur_L + (M_L - m_cur_L) * w


def dirac_masses(n_n, n_p, n_l, mrefN, mrefL, c_s):
    """Self-consistent (m_N*, m_L*) with shared scalar field."""
    mN, mL = mrefN, mrefL
    for _ in range(60):
        nsN = (base.fermion_scalar_density(n_n, mN) +
               base.fermion_scalar_density(n_p, mN))
        nsL = base.fermion_scalar_density(n_l, mL)
        ns_eff = nsN + XS * nsL
        mN_new = max(mrefN - c_s * ns_eff, 5.0)
        mL_new = max(mrefL - XS * c_s * ns_eff, 5.0)
        if abs(mN_new - mN) < 1e-6 and abs(mL_new - mL) < 1e-6:
            break
        mN = 0.5 * (mN + mN_new)
        mL = 0.5 * (mL + mL_new)
    return mN, mL


def eps_at(n_b, y_p, y_l, pars, U_n):
    k1, k2, c_s, c_rho, c_om, av, nv, mcurL = pars
    y_p = min(max(y_p, 0.0), 0.6)
    y_l = min(max(y_l, 0.0), 0.8)
    if y_p + y_l > 0.95:
        return 1e9
    n_p = y_p * n_b
    n_l = y_l * n_b
    n_n = n_b - n_p - n_l
    if n_n < 0:
        return 1e9
    mrefN = base.M_base(n_b, k1, k2)
    mrefL = m_base_lambda(n_b, k1, k2, mcurL)
    mN, mL = dirac_masses(n_n, n_p, n_l, mrefN, mrefL, c_s)
    # charge neutrality: electrons (+muons) match protons
    kfe = base.kf_from_density(n_p) if n_p > 0 else 0.0
    eps_b = (base.fermion_energy_density(n_n, mN) +
             base.fermion_energy_density(n_p, mN) +
             base.fermion_energy_density(n_l, mL))
    eps_lep = base.fermion_energy_density(n_p, base.m_e)   # e density = n_p
    eps_sc = 0.5 * (mrefN - mN) ** 2 / c_s if c_s > 0 else 0.0
    eps_rho = 0.5 * c_rho * (n_n - n_p) ** 2
    Qv = (n_n + n_p) + XW * n_l
    g = base.vector_factor(n_b, av, nv)
    eps_vec = 0.5 * c_om * g * Qv * Qv
    return eps_b + eps_lep + eps_sc + eps_rho + eps_vec + U_n


def build_strange(pars_dict, nn=200):
    k1, k2 = pars_dict["k1"], pars_dict["k2"]
    c_s, c_rho = pars_dict["Cs"], pars_dict["Crho"]
    av, nv = pars_dict["alpha_v"], pars_dict["nu_v"]
    mcurL = pars_dict["mcurL"]

    # saturation calibration (npe matter at n0, no Lambda, U(n0) small)
    st0 = base.beta_equilibrium_state(N0, k1, k2, c_s, c_rho)
    c_om = 2.0 * (M_N - E_BIND - st0["eps_no_vec"] / N0) / N0

    narr = np.geomspace(0.3 * N0, 9.0 * N0, nn)
    pars = (k1, k2, c_s, c_rho, c_om, av, nv, mcurL)

    def solve_grid(U_arr):
        eps, comp = np.zeros(nn), np.zeros((nn, 2))
        guess = np.array([0.05, 0.0])
        for i, n_b in enumerate(narr):
            res = minimize(lambda v: eps_at(n_b, v[0], v[1], pars, U_arr[i]),
                           guess, method="Nelder-Mead",
                           options={"xatol": 1e-5, "fatol": 1e-4,
                                    "maxiter": 400})
            eps[i] = res.fun
            comp[i] = np.clip(res.x, 0.0, 0.8)
            guess = comp[i] + 1e-3
        return eps, comp

    # two-pass melting-cost iteration
    U = np.zeros(nn)
    eps, comp = solve_grid(U)
    for _ in range(2):
        mbN = np.array([base.M_base(n, k1, k2) for n in narr])
        mbL = np.array([m_base_lambda(n, k1, k2, mcurL) for n in narr])
        dmbN, dmbL = np.gradient(mbN, narr), np.gradient(mbL, narr)
        nsN = np.zeros(nn)
        nsL = np.zeros(nn)
        for i, n_b in enumerate(narr):
            y_p, y_l = comp[i]
            n_p, n_l = y_p * n_b, y_l * n_b
            n_n = max(n_b - n_p - n_l, 0.0)
            mN, mL = dirac_masses(n_n, n_p, n_l, mbN[i], mbL[i], c_s)
            nsN[i] = (base.fermion_scalar_density(n_n, mN) +
                      base.fermion_scalar_density(n_p, mN))
            nsL[i] = base.fermion_scalar_density(n_l, mL)
        integrand = nsN * (-dmbN) + nsL * (-dmbL)
        U = np.concatenate([[0.0], np.cumsum(
            0.5 * (integrand[1:] + integrand[:-1]) * np.diff(narr))])
        eps, comp = solve_grid(U)

    epA = eps / narr
    p = narr ** 2 * np.gradient(epA, narr)
    return {"n": narr, "eps": eps, "p": p, "epA": epA,
            "yl": comp[:, 1], "c_om": c_om}


def gate(d):
    sel = d["n"] > 1.5 * N0
    n, p, epA, yl = d["n"][sel], d["p"][sel], d["epA"][sel], d["yl"][sel]
    if (p <= 0).any():
        i = np.where(p <= 0)[0]
        j = i[np.argmin(epA[i])]
        return epA[j], n[j], yl[j], True
    j = int(np.argmin(epA))
    return epA[j], n[j], yl[j], False


def main():
    print("=" * 78)
    print("  NVG: THE STRANGE GATE — npLambda MELTED MATTER AT P = 0")
    print("=" * 78)

    cases = [
        ("old canon (k2 = 0.8) — for reference, 2-flavor-excluded family",
         dict(k1=0.25, k2=0.8, Cs=900.0, Crho=600.0, alpha_v=4.0, nu_v=2.0,
              mcurL=190.0)),
        ("existence-constrained candidate (k2 = 2.5, alpha_v = 1)",
         dict(k1=0.25, k2=2.5, Cs=900.0, Crho=600.0, alpha_v=1.0, nu_v=2.0,
              mcurL=190.0)),
        ("candidate, strange-core sensitivity mcurL = 250",
         dict(k1=0.25, k2=2.5, Cs=900.0, Crho=600.0, alpha_v=1.0, nu_v=2.0,
              mcurL=250.0)),
        ("candidate, mcurL = 130",
         dict(k1=0.25, k2=2.5, Cs=900.0, Crho=600.0, alpha_v=1.0, nu_v=2.0,
              mcurL=130.0)),
    ]

    for label, pd in cases:
        try:
            d = build_strange(pd)
        except Exception as ex:
            print(f"\n  {label}: build failed ({ex})")
            continue
        epA0, n0p, yl0, selfb = gate(d)
        status = ("OPEN  (self-bound strange matter)" if
                  (selfb and epA0 < GATE) else "closed")
        print(f"\n  {label}")
        print(f"    dense-branch minimum eps/A = {epA0:.1f} MeV at "
              f"n = {n0p:.2f} fm^-3, Y_Lambda = {yl0:.2f}, "
              f"{'P<=0 reached' if selfb else 'P>0 everywhere'}")
        print(f"    GATE vs {GATE:.0f} MeV: {epA0-GATE:+.1f}  ->  {status}")

    print(f"""
  READING: the gate verdict for the existence-constrained candidate is
  the model's honest answer to the Witten question. OPEN means NVG
  predicts absolutely stable strange-melted matter (energy release
  930 - eps/A per baryon on conversion, weak-suppressed barrier — the
  physics-legal 'condensate fuel'); closed means condensate energy is
  reachable only in neutron-star interiors. Either way the number
  supersedes speculation. Approximations: npLambda only (no Sigma/Xi),
  quark-counting couplings, crude leptons, two-pass U iteration.
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
