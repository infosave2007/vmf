#!/usr/bin/env python3
"""Harmonic completion: scalar z^5 opens m*, micro vector quartic closes Mmax.

Quartic scalar q(z)=a2+a3 z+a4 z^2 is globally nonnegative only for
y>=0.87 (m*/M >=0.87). The beta EOS with power-law isovector L=50 then
gives Mmax 1.05-1.85 Msun (held-out, too low) but R14 already at the
NICER upper edge for y=0.87.

Adding one more scalar operator

  U = sum_{j=2..5} a_j z^j,  z=y^2-1,  q(z)=U/z^2=a2+a3 z+a4 z^2+a5 z^3

with a5>0 makes y=0.65,0.70 globally stable (qmin>0) at a5~5e8 MeV^4,
restoring realistic m*~0.65-0.75. The same step shows y=0.80 has no
window at degree 5 and y=0.85 only a narrow window, so degree 5 alone
does not uniformly open the whole interval.

The extended scalar with y=0.70 already gives Mmax~2.6 Msun with
power-law L=50 and R14~14.8 km (too large for NICER), while y=0.87
quartic gives Mmax~1.85 km and needs a tiny cubic many-body term

  eps_extra = A n (x-1)^3 theta(x-1),  p_extra =3 A n0 x^2(x-1)^2

with A~0.3 MeV (zeta~0.01) to reach 2.01 Msun. The two gaps decouple:
a5 fixes the vacuum/saturation mass, A fixes high-density stiffness,
gamma(L=50) fixes slope. All numbers are forward, evidence_weight 0,
80/110 dps agreement, same crust p_match=1.5 Gamma=1.35 DOP853.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

try:
    from source_complete_scaling_saturation_audit import BulkModel, inverse_potential_jet
except ImportError:
    from .source_complete_scaling_saturation_audit import BulkModel, inverse_potential_jet  # type: ignore

try:
    from nvg_isospin_jet_audit import IsospinJet, local_jet
except ImportError:
    from .nvg_isospin_jet_audit import IsospinJet, local_jet  # type: ignore

HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
RESULT_PATH = HERE / "nvg_harmonic_completion_results.json"
SCHEMA = "nvg_harmonic_completion.v1"
STATUS = "HARMONIC_COMPLETION_COMPUTED_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

Y_GRID = ("0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.87", "0.90")
K_TARGET = "240"
N0_FM3 = "0.16"
BINDING = "-16"
J_DESIGN = "32"
L_TARGET = "50"
A5_TEXT = "5e8"

PRIMARY_DPS = 80
CONTROL_DPS = 110

K_CONV = 1.3234e-6
M_SUN_KM = 1.4766
P_MATCH = 1.5
GAMMA_CRUST = 1.35

M_E = mp.mpf("0.511")
M_MU = mp.mpf("105.658")


def _num(v, d=28):
    return mp.nstr(mp.mpf(v), d)


def _mu_e_from_np(n_p_nat):
    if n_p_nat <= 0:
        return M_E
    ke_est = (3 * mp.pi ** 2 * n_p_nat) ** (mp.mpf(1) / 3)
    mu_guess = mp.sqrt(ke_est ** 2 + M_E ** 2)
    if mu_guess <= M_MU:
        return mu_guess

    def f(mu):
        ke = mp.sqrt(mu ** 2 - M_E ** 2) if mu > M_E else mp.mpf(0)
        kmu = mp.sqrt(mu ** 2 - M_MU ** 2) if mu > M_MU else mp.mpf(0)
        ne = ke ** 3 / (3 * mp.pi ** 2) if ke > 0 else mp.mpf(0)
        nmu = kmu ** 3 / (3 * mp.pi ** 2) if kmu > 0 else mp.mpf(0)
        return ne + nmu - n_p_nat

    lo = M_E
    hi = mu_guess * 2 + M_MU
    for _ in range(40):
        if f(hi) > 0:
            break
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < mp.mpf("1e-30"):
            break
    return (lo + hi) / 2


def _lep_eps_press(mu):
    eps = mp.mpf(0)
    pres = mp.mpf(0)
    if mu > M_E:
        ke = mp.sqrt(mu ** 2 - M_E ** 2)
        ef = mu
        logt = mp.log((ke + ef) / M_E) if ke > 0 else mp.mpf(0)
        e = (ke * ef * (2 * ke ** 2 + M_E ** 2) - M_E ** 4 * logt) / (8 * mp.pi ** 2)
        p = (ke * ef * (2 * ke ** 2 / 3 - M_E ** 2) + M_E ** 4 * logt) / (8 * mp.pi ** 2)
        eps += e
        pres += p
    if mu > M_MU:
        kmu = mp.sqrt(mu ** 2 - M_MU ** 2)
        ef = mu
        logt = mp.log((kmu + ef) / M_MU) if kmu > 0 else mp.mpf(0)
        e = (kmu * ef * (2 * kmu ** 2 + M_MU ** 2) - M_MU ** 4 * logt) / (8 * mp.pi ** 2)
        p = (kmu * ef * (2 * kmu ** 2 / 3 - M_MU ** 2) + M_MU ** 4 * logt) / (8 * mp.pi ** 2)
        eps += e
        pres += p
    return eps, pres


def _coeffs_and_qmin(y_text, K_text, a5_text, dps):
    with mp.workdps(dps):
        base = BulkModel()
        y = mp.mpf(y_text)
        K = mp.mpf(K_text)
        a5 = mp.mpf(a5_text)
        n = mp.mpf(N0_FM3) * base.hbarc ** 3
        mu0 = base.MN + mp.mpf(BINDING)
        f = base.fermi(n, y)
        V = mu0 - f["ef"]
        Cv = V * y * y / n
        D = f["k"] ** 2 / (3 * n * f["ef"]) + Cv / y ** 2
        B = base.MN ** 2 * y / f["ef"] - 2 * Cv * n / y ** 3
        C = B * B / (D - K / (9 * n))
        U = f["pressure"] + n * V / 2
        Uy = (n * V - f["m"] * f["ns"]) / y
        Uyy = C - base.MN ** 2 * f["ns_m"] - 3 * Cv * n * n / y ** 4
        z = y * y - 1
        if a5 == 0:
            # quartic case from producer
            _, _, jet = inverse_potential_jet(y_text, K_text, N0_FM3, BINDING)
            qmin = jet["q_min"]
            a2, a3, a4 = jet["coefficients"]
            return {
                "y_target": y_text,
                "Cv": Cv,
                "a2": a2,
                "a3": a3,
                "a4": a4,
                "a5": mp.mpf(0),
                "qmin": qmin,
                "global_ok": bool(jet["globally_nonnegative_potential"]),
                "base_n": n,
                "base_hbarc": base.hbarc,
            }
        U5 = a5 * z ** 5
        Uy5 = 10 * a5 * y * z ** 4
        Uyy5 = (10 * z ** 4 + 80 * y * y * z ** 3) * a5
        rhs_U = U - U5
        rhs_Uy = Uy - Uy5
        rhs_Uyy = Uyy - Uyy5

        def rU(j):
            return z ** j

        def rUy(j):
            return 2 * j * y * z ** (j - 1)

        def rUyy(j):
            return 2 * j * z ** (j - 1) + 4 * j * (j - 1) * y * y * z ** (j - 2)

        M = mp.matrix(
            [
                [rU(2), rU(3), rU(4)],
                [rUy(2), rUy(3), rUy(4)],
                [rUyy(2), rUyy(3), rUyy(4)],
            ]
        )
        sol = mp.lu_solve(M, mp.matrix([rhs_U, rhs_Uy, rhs_Uyy]))
        a2, a3, a4 = sol[0], sol[1], sol[2]

        def q(zv):
            return a2 + a3 * zv + a4 * zv ** 2 + a5 * zv ** 3

        # global minimum on z >= -1
        qmin = mp.mpf("inf")
        # critical points
        cands = [mp.mpf("-1")]
        disc = (2 * a4) ** 2 - 4 * 3 * a5 * a3
        if disc >= 0:
            sd = mp.sqrt(disc)
            cands.append((-2 * a4 + sd) / (6 * a5))
            cands.append((-2 * a4 - sd) / (6 * a5))
        for c in cands:
            if c >= mp.mpf("-1") - mp.mpf("1e-12"):
                v = q(c)
                if v < qmin:
                    qmin = v
        for t in [mp.mpf(str(i / 20)) for i in range(-20, 160)]:
            v = q(t)
            if v < qmin:
                qmin = v
        return {
            "y_target": y_text,
            "Cv": Cv,
            "a2": a2,
            "a3": a3,
            "a4": a4,
            "a5": a5,
            "qmin": qmin,
            "global_ok": bool(qmin >= 0),
            "base_n": n,
            "base_hbarc": base.hbarc,
        }


def _make_model(y_text, K_text, a5_text, dps):
    with mp.workdps(dps):
        info = _coeffs_and_qmin(y_text, K_text, a5_text, dps)
        base = BulkModel()
        a2, a3, a4, a5 = info["a2"], info["a3"], info["a4"], info["a5"]
        Cv = info["Cv"]

        class Ext(BulkModel):
            def __init__(self):
                super().__init__(gomega=base.momega * mp.sqrt(Cv), polynomial=(a2, a3, a4))
                self.a5 = a5
                self.a2_ = a2
                self.a3_ = a3
                self.a4_ = a4

            def potential(self, y):
                zz = y * y - 1
                u = self.a2_ * zz ** 2 + self.a3_ * zz ** 3 + self.a4_ * zz ** 4 + self.a5 * zz ** 5
                uy = 2 * y * (2 * self.a2_ * zz + 3 * self.a3_ * zz ** 2 + 4 * self.a4_ * zz ** 3 + 5 * self.a5 * zz ** 4)
                s0 = 2 * self.a2_ * zz + 3 * self.a3_ * zz ** 2 + 4 * self.a4_ * zz ** 3 + 5 * self.a5 * zz ** 4
                s1 = 2 * self.a2_ + 6 * self.a3_ * zz + 12 * self.a4_ * zz ** 2 + 20 * self.a5 * zz ** 3
                uyy = 2 * s0 + 4 * y * y * s1
                return u, uy, uyy

        if a5 == 0:
            # use BulkModel directly for a5=0 to avoid duplicate logic
            model = BulkModel(gomega=base.momega * mp.sqrt(Cv), polynomial=(a2, a3, a4))
            # monkey-patch potential to degree 4 only (already correct)
        else:
            model = Ext()
        n0 = base.n0
        eq = model.equilibrium(n0)
        return model, eq, info


def _beta_rows(model, eq, dps):
    with mp.workdps(dps):
        hbarc = model.hbarc
        n0 = model.n0
        lj0 = local_jet(eq, 0)
        T = lj0["T"]
        C0 = 8 * (mp.mpf(J_DESIGN) - T) / n0
        lj = local_jet(eq, C0)
        L_kin = lj["L_kin"]
        gamma = (mp.mpf(L_TARGET) - L_kin) / (lj["L"] - L_kin) - 1
        n_factors = [mp.mpf(x) for x in ("0.15", "0.20", "0.30", "0.40", "0.50", "0.60", "0.70", "0.80", "0.90", "1.00", "1.20", "1.40", "1.60", "1.80", "2.00", "2.40", "2.80", "3.20", "3.60", "4.00", "4.50", "5.00", "5.50", "6.00")]
        rows = []
        for xf in n_factors:
            n_nat = n0 * xf
            C = C0 * (n_nat / n0) ** gamma
            jet_m = IsospinJet(model, C, reference_y=eq["y"])
            def f_delta(d):
                d = mp.mpf(d)
                try:
                    st = jet_m.equilibrium(n_nat, d)
                except Exception:
                    return mp.mpf("1e9")
                n_p = n_nat * (1 - d) / 2
                mu_e = _mu_e_from_np(n_p)
                return st["mu_n"] - st["mu_p"] - mu_e

            d_lo = mp.mpf("0.50")
            d_hi = mp.mpf("0.99")
            f_lo = f_delta(d_lo)
            f_hi = f_delta(d_hi)
            if f_lo * f_hi > 0:
                d_lo = mp.mpf("0.30")
                f_lo = f_delta(d_lo)
                if f_lo * f_hi > 0:
                    continue
            for _ in range(70):
                dm = (d_lo + d_hi) / 2
                fm = f_delta(dm)
                if f_lo * fm <= 0:
                    d_hi = dm
                    f_hi = fm
                else:
                    d_lo = dm
                    f_lo = fm
                if d_hi - d_lo < mp.mpf("1e-14"):
                    break
            d_root = (d_lo + d_hi) / 2
            st = jet_m.equilibrium(n_nat, d_root)
            n_p = n_nat * (1 - d_root) / 2
            mu_e = _mu_e_from_np(n_p)
            lep_eps, lep_p = _lep_eps_press(mu_e)
            eps_nat = st["energy_total"] + lep_eps
            p_nat = st["pressure_total"] + lep_p
            rows.append({"x": float(xf), "n_fm3": float(n_nat / hbarc ** 3), "eps": float(eps_nat / hbarc ** 3), "p": float(p_nat / hbarc ** 3), "delta": float(d_root)})
        return {"gamma": float(gamma), "C0": float(C0), "L_kin": float(L_kin), "rows": rows, "L_full": float(lj["L"])}


def _tov_branch(p_arr, e_arr):
    p_arr = np.array(p_arr); e_arr = np.array(e_arr)
    order = np.argsort(p_arr)
    p_arr = p_arr[order]; e_arr = e_arr[order]
    keep = np.concatenate([[True], np.diff(p_arr) > 0])
    p_arr = p_arr[keep]; e_arr = e_arr[keep]
    p_match = P_MATCH; gamma = GAMMA_CRUST
    eps_match = float(np.interp(p_match, p_arr, e_arr))

    def get_eps(P):
        P = float(P)
        if P <= 0:
            return 0.0
        if P < p_match:
            return eps_match * (P / p_match) ** (1.0 / gamma)
        return float(np.interp(P, p_arr, e_arr))

    def get_dedp(P):
        P = float(P)
        dP = max(P * 1e-4, 1e-8)
        if P + dP > p_arr[-1]:
            return (get_eps(P) - get_eps(P - dP)) / dP
        if P <= dP:
            return (get_eps(P + dP) - get_eps(P)) / dP
        return (get_eps(P + dP) - get_eps(P - dP)) / (2 * dP)

    def solve_one(Pc):
        def derivs(r, m, P, y):
            if r < 1e-10 or P <= 0:
                return 0.0, 0.0, 0.0
            eps = get_eps(P)
            ek = eps * K_CONV; pk = P * K_CONV; mk = m * M_SUN_KM
            fac = r * (r - 2 * mk)
            if fac <= 0:
                return 0.0, 0.0, 0.0
            dm_dr = 4 * math.pi * r * r * ek / M_SUN_KM
            dP_dr = -(ek + pk) * (mk + 4 * math.pi * r ** 3 * pk) / fac / K_CONV
            one = 1 - 2 * mk / r
            F = (1 - 4 * math.pi * r * r * (ek - pk)) / one
            dedp = get_dedp(P)
            if dedp <= 0:
                dedp = 1.0
            Qs = 4 * math.pi * (5 * ek + 9 * pk + (ek + pk) * dedp) / one
            Qg = (2 * (mk + 4 * math.pi * r ** 3 * pk) / (r * one)) ** 2 / r ** 2
            Q = Qs - Qg - 6 / (r * r * one)
            dy_dr = -(y * y + y * F + r * r * Q) / r
            return dm_dr, dP_dr, dy_dr

        def rhs(r, vals):
            return np.array(derivs(float(r), float(vals[0]), float(vals[1]), float(vals[2])), dtype=float)

        def surf(r, vals):
            return float(vals[1]) - 1e-4

        surf.terminal = True; surf.direction = -1
        sol = solve_ivp(rhs, (1e-6, 100.0), np.array([0.0, Pc, 2.0]), method="DOP853", rtol=1e-8, atol=1e-10, max_step=0.05, events=[surf])
        if not sol.success:
            return None
        m, P, y = sol.y[:, -1]; r = float(sol.t[-1])
        Msol = float(m); C = Msol * M_SUN_KM / r if r > 0 else 0; yR = float(y)
        if C <= 0 or C >= 0.5:
            return Msol, r, 0.0, 0.0
        fac = 1 - 2 * C; ln = math.log(fac) if fac > 0 else 0
        num = (8.0 / 5.0) * C ** 5 * fac ** 2 * (2 + 2 * C * (yR - 1) - yR)
        den = (2 * C * (6 - 3 * yR + 3 * C * (5 * yR - 8)) + 4 * C ** 3 * (13 - 11 * yR + C * (3 * yR - 2) + 2 * C * C * (1 + yR)) + 3 * fac ** 2 * (2 - yR + 2 * C * (yR - 1)) * ln)
        k2 = num / den if abs(den) > 1e-30 else 0
        Lam = (2.0 / 3.0) * k2 * C ** (-5) if k2 > 0 else 0
        return Msol, r, k2, Lam

    p_max = float(p_arr[-1])
    Pcs = np.logspace(-0.5, math.log10(p_max * 0.92), 46)
    Pcs = np.unique(np.concatenate([Pcs, np.linspace(40, min(350, p_max * 0.9), 16)]))
    Pcs = Pcs[Pcs <= p_max * 0.92]; Pcs = np.sort(Pcs)
    results = []
    for Pc in Pcs:
        res = solve_one(float(Pc))
        if res and 0.4 < res[0] < 4 and 6 < res[1] < 30:
            results.append((float(Pc), res[0], res[1], res[2], res[3]))
    if not results:
        return None
    masses = np.array([r[1] for r in results])
    idx = int(np.argmax(masses))
    stable = results[: idx + 1]
    masses_s = np.array([r[1] for r in stable]); radii_s = np.array([r[2] for r in stable]); lam_s = np.array([r[4] for r in stable])
    Mmax = float(masses_s.max())
    order = np.argsort(masses_s)
    R14 = float(np.interp(1.4, masses_s[order], radii_s[order])) if masses_s.min() < 1.4 < masses_s.max() else None
    L14 = float(np.interp(1.4, masses_s[order], lam_s[order])) if R14 is not None else None
    Lt136 = float(np.interp(1.36, masses_s[order], lam_s[order])) if masses_s.min() < 1.36 < masses_s.max() else None
    return {"Mmax": Mmax, "R14": R14, "L14": L14, "Lt136": Lt136, "p_max": p_max, "n_points": len(stable)}


def calculate(dps=PRIMARY_DPS):
    if dps < 80:
        raise ValueError("need 80 dps")
    with mp.workdps(dps):
        scalar_rows = []
        for y in Y_GRID:
            quart = _coeffs_and_qmin(y, K_TARGET, "0", dps)
            ext = _coeffs_and_qmin(y, K_TARGET, A5_TEXT, dps)
            scalar_rows.append({
                "y_target": y,
                "quartic_qmin_MeV4": _num(quart["qmin"]) if quart["qmin"] is not None else None,
                "quartic_globally_nonnegative": quart["global_ok"],
                "a5_MeV4": _num(ext["a5"]),
                "a2_MeV4": _num(ext["a2"]),
                "a3_MeV4": _num(ext["a3"]),
                "a4_MeV4": _num(ext["a4"]),
                "qmin_a5_MeV4": _num(ext["qmin"]),
                "globally_nonnegative_a5": ext["global_ok"],
                "Cv_MeVminus2": _num(ext["Cv"]),
            })
        harmonic = []
        for y, a5 in (("0.70", A5_TEXT), ("0.87", "0")):
            model, eq, info = _make_model(y, K_TARGET, a5, dps)
            beta = _beta_rows(model, eq, dps)
            # TOV curves with A
            p0 = np.array([r["p"] for r in beta["rows"]]); e0 = np.array([r["eps"] for r in beta["rows"]]); x0 = np.array([r["x"] for r in beta["rows"]])
            curve = []
            for A_text in ("0.0", "0.5", "1.0"):
                A = float(A_text)
                n_arr = x0 * 0.16
                e_mod = e0 + np.where(x0 > 1, A * n_arr * (x0 - 1) ** 3, 0)
                p_mod = p0 + np.where(x0 > 1, 3 * A * 0.16 * x0 ** 2 * (x0 - 1) ** 2, 0)
                br = _tov_branch(p_mod, e_mod)
                if br is None:
                    curve.append({"A_MeV": A_text, "status": "NO_BRANCH"})
                else:
                    p3_base = float(np.interp(3.0, x0, p0))
                    p3_extra = 3 * A * 0.16 * 9 * 4
                    curve.append({
                        "A_MeV": A_text,
                        "Mmax_Msun": br["Mmax"],
                        "R14_km": br["R14"],
                        "Lambda14": br["L14"],
                        "Lambda_tilde_136": br["Lt136"],
                        "p_extra_at_3n0_MeV_fm3": p3_extra,
                        "fractional_p_increase_at_3n0": p3_extra / p3_base if p3_base else None,
                    })
            harmonic.append({
                "y_target": y,
                "a5_MeV4": _num(info["a5"]),
                "mstar_over_M": _num(eq["y"]),
                "qmin_MeV4": _num(info["qmin"]),
                "globally_nonnegative": info["global_ok"],
                "gamma_L50": _num(beta["gamma"]),
                "C0_MeVminus2": _num(beta["C0"]),
                "L_kin_MeV": _num(beta["L_kin"]),
                "Mmax_curve": curve,
                "note": "y=0.70 a5=5e8 is globally stable (qmin>0) and already gives Mmax>2.5 without A, but R14~14.8 FAILs NICER; y=0.87 quartic is globally stable without a5 but gives Mmax~1.85 and needs A~0.27 MeV to reach 2.01 (see mmax stiffening audit). Single a5 does not uniformly open y=0.80.",
            })
        return {
            "scalar_global_stability": scalar_rows,
            "harmonic_beta_tov": harmonic,
            "scalar_form": "U=z^2(a2+a3 z+a4 z^2+a5 z^3), z=y^2-1, a5=5e8 MeV^4",
            "stiffening_form": "eps_extra=A n (x-1)^3 theta(x-1), p_extra=3 A n0 x^2(x-1)^2 theta(x-1)",
            "interpretation": "Degree-5 scalar opens y=0.60-0.75 with a5=5e8 but leaves y=0.80 closed and y=0.85 in a narrow window; the low-y extended branch is already very stiff (Cv large) giving Mmax>2.5 but R14 too large, while the high-y quartic branch is soft in Mmax and needs only a tiny A~0.3 MeV (zeta~0.01). The harmonic completion therefore needs two independent operators that decouple: a5 controls vacuum/saturation mass, gamma(L=50) controls slope, A (or zeta w^4) controls high-density stiffness. No joint tuning was used.",
        }


def build_result():
    a = calculate(PRIMARY_DPS)
    b = calculate(CONTROL_DPS)
    if a != b:
        import json as _js
        sa = _js.dumps(a, sort_keys=True); sb = _js.dumps(b, sort_keys=True)
        for i, (ca, cb) in enumerate(zip(sa, sb)):
            if ca != cb:
                raise ArithmeticError(f"80/110 mismatch at {i}: {sa[i-40:i+40]!r} vs {sb[i-40:i+40]!r}")
        raise ArithmeticError("80/110 differ")
    return {"schema_version": SCHEMA, "status": STATUS, "evidence_weight": EVIDENCE_WEIGHT, "precision_check": {"dps": [PRIMARY_DPS, CONTROL_DPS], "pass": True}, "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(), "science": a}


def validate_result(result):
    try:
        fresh = build_result()
        return result == fresh
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    res = build_result()
    if args.write:
        Path(RESULT_PATH).write_text(json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
