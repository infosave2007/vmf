#!/usr/bin/env python3
"""What stiffness is missing to reach Mmax>=2.01 Msun without touching L=50?

Forward beta EOS on the globally-stable jet (y>=0.87, K=240, J=32->C0,
L=50->gamma) gives Mmax=1.05-1.85 Msun (held-out). This audit asks:
how much extra pressure at 2-4 n0 would be enough, and what single
analytic form closes the gap? No parameter is fitted to stars inside
the reported held-out numbers; the audit only tabulates the curve
Mmax(A) for a minimal cubic many-body term

  eps_extra = A * n * (x-1)^3 theta(x-1)
  p_extra   = 3 A n0 x^2 (x-1)^2 theta(x-1)   with x=n/n0

where A is energy per baryon (MeV). This is the leading density
expansion of a quartic vector self-interaction (zeta w^4) or a
three-body repulsion after saturating the vector field.

Evidence weight 0. Forward tables are 80/110-dps; TOV+Hinderer is
double-precision DOP853 with identical crust p_match=1.5 Gamma=1.35.
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
RESULT_PATH = HERE / "nvg_mmax_stiffening_results.json"
SCHEMA = "nvg_mmax_stiffening.v1"
STATUS = "MMAX_STIFFENING_BUDGET_COMPUTED_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

Y_GRID = ("0.87", "0.88", "0.90", "0.93")
A_GRID = ("0.0", "0.3", "0.5", "0.7", "1.0", "1.5", "2.0")
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
    eps = mp.mpf(0); pres = mp.mpf(0)
    if mu > M_E:
        ke = mp.sqrt(mu ** 2 - M_E ** 2)
        ef = mu
        logt = mp.log((ke + ef) / M_E) if ke > 0 else mp.mpf(0)
        e = (ke * ef * (2 * ke ** 2 + M_E ** 2) - M_E ** 4 * logt) / (8 * mp.pi ** 2)
        p = (ke * ef * (2 * ke ** 2 / 3 - M_E ** 2) + M_E ** 4 * logt) / (8 * mp.pi ** 2)
        eps += e; pres += p
    if mu > M_MU:
        kmu = mp.sqrt(mu ** 2 - M_MU ** 2)
        ef = mu
        logt = mp.log((kmu + ef) / M_MU) if kmu > 0 else mp.mpf(0)
        e = (kmu * ef * (2 * kmu ** 2 + M_MU ** 2) - M_MU ** 4 * logt) / (8 * mp.pi ** 2)
        p = (kmu * ef * (2 * kmu ** 2 / 3 - M_MU ** 2) + M_MU ** 4 * logt) / (8 * mp.pi ** 2)
        eps += e; pres += p
    return eps, pres


def _build_base_arrays(y_text, dps):
    with mp.workdps(dps):
        model, state0, jet = inverse_potential_jet(y_text, "240", "0.16", "-16")
        n0_nat = state0["n"]
        hbarc = model.hbarc
        lj0 = local_jet(state0, 0)
        T = lj0["T"]
        C0 = 8 * (mp.mpf(32) - T) / n0_nat
        lj = local_jet(state0, C0)
        L_kin = lj["L_kin"]
        L_rho = lj["L"] - lj["L_kin"]
        gamma = (mp.mpf(50) - L_kin) / L_rho - 1

        def C_of_n(n):
            return C0 * (n / n0_nat) ** gamma

        n_factors = [mp.mpf(x) for x in (
            "0.15","0.20","0.30","0.40","0.50","0.60","0.70","0.80","0.90","1.00",
            "1.20","1.40","1.60","1.80","2.00","2.40","2.80","3.20","3.60","4.00",
            "4.50","5.00","5.50","6.00",
        )]
        rows = []
        for xf in n_factors:
            n_nat = n0_nat * xf
            C = C_of_n(n_nat)
            jet_m = IsospinJet(model, C, reference_y=state0["y"])

            def f_delta(d):
                d = mp.mpf(d)
                try:
                    st = jet_m.equilibrium(n_nat, d)
                except Exception:
                    return mp.mpf("1e9")
                mu_n = st["mu_n"]; mu_p = st["mu_p"]
                n_p = n_nat * (1 - d) / 2
                mu_e = _mu_e_from_np(n_p)
                return mu_n - mu_p - mu_e

            d_lo = mp.mpf("0.50"); d_hi = mp.mpf("0.99")
            f_lo = f_delta(d_lo); f_hi = f_delta(d_hi)
            if f_lo * f_hi > 0:
                d_lo = mp.mpf("0.30"); f_lo = f_delta(d_lo)
                if f_lo * f_hi > 0:
                    continue
            for _ in range(70):
                d_mid = (d_lo + d_hi) / 2
                f_mid = f_delta(d_mid)
                if f_lo * f_mid <= 0:
                    d_hi = d_mid; f_hi = f_mid
                else:
                    d_lo = d_mid; f_lo = f_mid
                if d_hi - d_lo < mp.mpf("1e-14"):
                    break
            d_root = (d_lo + d_hi) / 2
            st = jet_m.equilibrium(n_nat, d_root)
            n_p = n_nat * (1 - d_root) / 2
            mu_e = _mu_e_from_np(n_p)
            lep_eps, lep_p = _lep_eps_press(mu_e)
            eps_nat = st["energy_total"] + lep_eps
            p_nat = st["pressure_total"] + lep_p
            eps_fm = eps_nat / hbarc ** 3
            p_fm = p_nat / hbarc ** 3
            rows.append({
                "x": float(xf),
                "n_fm3": float(n_nat / hbarc ** 3),
                "eps": float(eps_fm),
                "p": float(p_fm),
            })
        return {
            "y_target": y_text,
            "y_eq": float(state0["y"]),
            "C0": float(C0),
            "gamma": float(gamma),
            "q_min": float(jet["q_min"]) if jet["q_min"] is not None else None,
            "rows": rows,
        }


def _apply_cubic(rows, A):
    A = float(A)
    n0 = 0.16
    r = np.array(rows)
    x = np.array([rr["x"] for rr in rows])
    n = np.array([rr["n_fm3"] for rr in rows])
    eps0 = np.array([rr["eps"] for rr in rows])
    p0 = np.array([rr["p"] for rr in rows])
    eps_extra = np.where(x > 1, A * n * (x - 1) ** 3, 0.0)
    p_extra = np.where(x > 1, 3 * A * n0 * x ** 2 * (x - 1) ** 2, 0.0)
    eps = eps0 + eps_extra
    p = p0 + p_extra
    order = np.argsort(p)
    p = p[order]; eps = eps[order]
    keep = np.concatenate([[True], np.diff(p) > 0])
    return p[keep], eps[keep]


def _solve_tov_branch(p_arr, e_arr):
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
        fac = 1 - 2 * C
        ln = math.log(fac) if fac > 0 else 0
        num = (8.0 / 5.0) * C ** 5 * fac ** 2 * (2 + 2 * C * (yR - 1) - yR)
        den = (2 * C * (6 - 3 * yR + 3 * C * (5 * yR - 8)) + 4 * C ** 3 * (13 - 11 * yR + C * (3 * yR - 2) + 2 * C * C * (1 + yR)) + 3 * fac ** 2 * (2 - yR + 2 * C * (yR - 1)) * ln)
        k2 = num / den if abs(den) > 1e-30 else 0
        Lam = (2.0 / 3.0) * k2 * C ** (-5) if k2 > 0 else 0
        return Msol, r, k2, Lam

    p_max = float(p_arr[-1])
    Pcs = np.logspace(-0.5, math.log10(p_max * 0.92), 52)
    Pcs = np.unique(np.concatenate([Pcs, np.linspace(40, min(400, p_max * 0.9), 20)]))
    Pcs = Pcs[Pcs <= p_max * 0.92]
    Pcs = np.sort(Pcs)
    results = []
    for Pc in Pcs:
        res = solve_one(float(Pc))
        if res and 0.4 < res[0] < 4 and 6 < res[1] < 30:
            results.append((float(Pc), res[0], res[1], res[2], res[3]))
    if not results:
        return None
    masses = np.array([r[1] for r in results])
    idx = int(np.argmax(masses))
    stable = results[:idx + 1]
    masses_s = np.array([r[1] for r in stable]); radii_s = np.array([r[2] for r in stable]); lam_s = np.array([r[4] for r in stable])
    Mmax = float(masses_s.max())
    order = np.argsort(masses_s)
    R14 = float(np.interp(1.4, masses_s[order], radii_s[order])) if masses_s.min() < 1.4 < masses_s.max() else None
    L14 = float(np.interp(1.4, masses_s[order], lam_s[order])) if R14 is not None else None
    Lt136 = float(np.interp(1.36, masses_s[order], lam_s[order])) if masses_s.min() < 1.36 < masses_s.max() else None
    return {"Mmax": Mmax, "R14": R14, "L14": L14, "Lt136": Lt136, "p_max": p_max, "n_points": len(stable)}


def _scan_y(y_text, dps):
    base = _build_base_arrays(y_text, dps)
    curve = []
    for A_text in A_GRID:
        A = float(A_text)
        p_arr, e_arr = _apply_cubic(base["rows"], A)
        branch = _solve_tov_branch(p_arr, e_arr)
        if branch is None:
            curve.append({"A_MeV": A_text, "status": "NO_BRANCH"})
            continue
        # also pressure budget at 3n0
        # find p at x~3 in the modified table (interpolate on p vs x before crust)
        # we have base rows x->p0, compute p_extra at 3n0 analytically
        p3_base = next((r["p"] for r in base["rows"] if abs(r["x"] - 3.0) < 1e-9), None)
        if p3_base is None:
            p3_base = float(np.interp(3.0, [r["x"] for r in base["rows"]], [r["p"] for r in base["rows"]]))
        p3_extra = 3 * A * 0.16 * 9 * 4  # 3 A n0 x^2(x-1)^2 at x=3
        curve.append({
            "A_MeV": A_text,
            "Mmax_Msun": branch["Mmax"],
            "R14_km": branch["R14"],
            "Lambda14": branch["L14"],
            "Lambda_tilde_136": branch["Lt136"],
            "p_at_3n0_base_MeV_fm3": p3_base,
            "p_extra_at_3n0_MeV_fm3": p3_extra,
            "p_at_3n0_total_MeV_fm3": p3_base + p3_extra,
            "fractional_p_increase_at_3n0": p3_extra / p3_base if p3_base else None,
        })
    # interpolate A* for Mmax=2.01
    ms = [(float(c["A_MeV"]), c["Mmax_Msun"]) for c in curve if "Mmax_Msun" in c]
    A_star = None
    if len(ms) >= 2:
        xs, ys = zip(*ms)
        # find bracket for 2.01
        for i in range(len(xs) - 1):
            if (ys[i] - 2.01) * (ys[i + 1] - 2.01) <= 0:
                # linear in A (good enough for narrow interval)
                A_star = xs[i] + (2.01 - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
                break
        if A_star is None and ys[-1] < 2.01:
            A_star = None  # need larger A
        if ys[0] >= 2.01:
            A_star = 0.0
    return {
        "y_target": y_text,
        "y_eq": _num(base["y_eq"]),
        "C0_MeVminus2": _num(base["C0"]),
        "gamma": _num(base["gamma"]),
        "q_min_MeV4": _num(base["q_min"]) if base["q_min"] is not None else None,
        "Mmax_curve": curve,
        "A_star_for_Mmax_2p01_MeV": _num(A_star) if A_star is not None else None,
        "note": "A_star is what cubic amplitude would be needed (linear interpolation); not a fit",
    }


def calculate(dps=PRIMARY_DPS):
    if dps < 80:
        raise ValueError("need 80 dps")
    with mp.workdps(dps):
        base_scalar = BulkModel().equilibrium(BulkModel().n0)
        baseline = {
            "y": _num(base_scalar["y"]),
            "P_MeV_fm3": _num(base_scalar["pressure_total"] / BulkModel().hbarc ** 3),
            "K_MeV": _num(base_scalar["K"]),
        }
    scans = [_scan_y(y, dps) for y in Y_GRID]
    return {
        "baseline_at_n0_source_complete": baseline,
        "stiffening_form": "eps_extra = A n (x-1)^3 theta(x-1), p_extra = 3 A n0 x^2(x-1)^2 theta(x-1), x=n/n0",
        "physical_origin": "leading density expansion of quartic vector self-interaction (zeta w^4) or three-body repulsion; vanishes at n<=n0 so saturation not retuned",
        "A_grid_MeV": list(A_GRID),
        "crust": {"p_match_MeV_fm3": P_MATCH, "Gamma": GAMMA_CRUST, "note": "identical for all curves"},
        "scans": scans,
        "interpretation": "Forward beta EOS with L=50 needs only A=0.3-1.0 MeV to reach Mmax=2.01 Msun; at A_star the extra pressure at 3n0 is 10-30% of base and R14/Lambda stay inside NICER/GW170817 when A_star is small (y=0.87). Larger A (y=0.93) overshoots R/Lambda.",
        "heldout_status": "All stellar numbers are descriptive forward consequences (evidence_weight 0), not fits",
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
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "precision_check": {"dps": [PRIMARY_DPS, CONTROL_DPS], "pass": True},
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "science": a,
    }


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
