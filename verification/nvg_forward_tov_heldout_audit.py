#!/usr/bin/env python3
"""Held-out TOV + Hinderer tidal audit on the forward beta EOS (no fit).

Consumes verification/nvg_forward_beta_eos_results.json (read-only).
For each y* (0.87,0.88,0.90,0.93) the table p(eps) produced with
gamma(L=50) is taken as a forward consequence — no parameter is re-tuned
to M_max, R or Lambda. A single explicit crust (p_match=1.5 Gamma=1.35)
is used identically for all branches. TOV + Hinderer are solved with
DOP853, and NICER/GW170817/J0740 numbers are reported as descriptive
held-out comparisons, not fits.

Evidence weight 0. Precision check is 80/110 for the underlying table;
TOV itself is double-precision with an explicit rtol/atol ladder as a
convergence diagnostic (not a fit).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
RESULT_PATH = HERE / "nvg_forward_tov_heldout_results.json"
SCHEMA = "nvg_forward_tov_heldout.v1"
STATUS = "FORWARD_TOV_HELDOUT_COMPUTED_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

FORWARD_RESULT = HERE / "nvg_forward_beta_eos_results.json"
P_MATCH = 1.5  # MeV/fm3
GAMMA_CRUST = 1.35
K_CONV = 1.3234e-6  # MeV/fm3 -> km^-2
M_SUN_KM = 1.4766

PRIMARY_DPS = 80
CONTROL_DPS = 110


def _load_forward(dps_tag="primary"):
    data = json.loads(FORWARD_RESULT.read_text(encoding="utf-8"))
    # validate dps provenance (table was 80/110)
    assert data["precision_check"]["pass"] is True
    assert data["precision_check"]["dps"] == [PRIMARY_DPS, CONTROL_DPS]
    return data["science"]["forward_tables"]


class ForwardEOS:
    def __init__(self, table, p_match=P_MATCH, gamma=GAMMA_CRUST):
        # table is list of dicts with eps_MeV_fm3, P_MeV_fm3
        pts = [(float(r["P_MeV_fm3"]), float(r["eps_MeV_fm3"])) for r in table if "P_MeV_fm3" in r and "eps_MeV_fm3" in r]
        pts = sorted(pts)
        # keep strictly increasing P
        p_arr = np.array([p for p, _ in pts])
        e_arr = np.array([e for _, e in pts])
        keep = np.concatenate([[True], np.diff(p_arr) > 0])
        p_arr = p_arr[keep]
        e_arr = e_arr[keep]
        self.p_arr = p_arr
        self.e_arr = e_arr
        self.p_match = float(p_match)
        self.gamma = float(gamma)
        if not (p_arr[0] < p_match < p_arr[-1]):
            raise ValueError(f"p_match {p_match} not inside table [{p_arr[0]}, {p_arr[-1]}]")
        self.eps_match = float(np.interp(p_match, p_arr, e_arr))
        self.p_min = 0.0
        self.p_max = float(p_arr[-1])

    def get_eps(self, P):
        P = float(P)
        if P < 0 or P > self.p_max:
            raise ValueError(f"P {P} outside [{0}, {self.p_max}]")
        if P == 0:
            return 0.0
        if P < self.p_match:
            return self.eps_match * (P / self.p_match) ** (1.0 / self.gamma)
        return float(np.interp(P, self.p_arr, self.e_arr))

    def get_dedp(self, P):
        P = float(P)
        if P < 0 or P > self.p_max:
            raise ValueError("outside domain")
        dP = max(P * 1e-4, 1e-8)
        if P <= dP:
            return (self.get_eps(P + dP) - self.get_eps(P)) / dP
        if P + dP > self.p_max:
            return (self.get_eps(P) - self.get_eps(P - dP)) / dP
        return (self.get_eps(P + dP) - self.get_eps(P - dP)) / (2 * dP)


def solve_tov_tidal(eos, P_center, rtol=1e-8, atol=1e-10, dr=0.05):
    P_center = float(P_center)
    eos.get_eps(P_center)  # fail closed
    # Hinderer TOV + y
    def derivs(r, m, P, y):
        if r < 1e-10 or P <= 0:
            return 0.0, 0.0, 0.0
        eps = eos.get_eps(P)
        ek = eps * K_CONV
        pk = P * K_CONV
        mk = m * M_SUN_KM
        fac = r * (r - 2 * mk)
        if fac <= 0:
            return 0.0, 0.0, 0.0
        dm_dr = 4 * math.pi * r * r * ek / M_SUN_KM
        dP_dr = -(ek + pk) * (mk + 4 * math.pi * r ** 3 * pk) / fac / K_CONV
        # Hinderer
        one = 1 - 2 * mk / r
        F = (1 - 4 * math.pi * r * r * (ek - pk)) / one
        dedp = eos.get_dedp(P)
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
    surf.terminal = True
    surf.direction = -1

    cap = 0.05 if rtol <= 1e-4 else 0.10
    max_step = max(dr, cap)
    sol = solve_ivp(rhs, (1e-6, 100.0), np.array([0.0, P_center, 2.0], dtype=float),
                    method="DOP853", rtol=rtol, atol=atol, max_step=max_step, events=[surf])
    if not sol.success:
        raise RuntimeError(sol.message)
    m, P, y = sol.y[:, -1]
    r = float(sol.t[-1])
    # Love
    Msol = float(m)
    C = Msol * M_SUN_KM / r if r > 0 else 0
    yR = float(y)
    if C <= 0 or C >= 0.5:
        return Msol, r, 0.0, 0.0
    fac = 1 - 2 * C
    if fac <= 0:
        return Msol, r, 0.0, 0.0
    ln = math.log(fac)
    num = (8.0/5.0) * C**5 * fac**2 * (2 + 2*C*(yR-1) - yR)
    den = (2*C*(6 -3*yR +3*C*(5*yR-8)) + 4*C**3*(13 -11*yR + C*(3*yR-2) +2*C*C*(1+yR)) + 3*fac**2*(2 - yR +2*C*(yR-1))*ln)
    if abs(den) < 1e-30:
        return Msol, r, 0.0, 0.0
    k2 = num/den
    Lam = (2.0/3.0)*k2*C**(-5) if k2>0 else 0.0
    return Msol, r, k2, Lam


def branch_for_y(y_target, rtol=1e-8, atol=1e-10):
    tables = _load_forward()
    entry = next(t for t in tables if t["y_target"] == y_target)
    eos = ForwardEOS(entry["table"])
    # scan Pc
    Pcs = np.logspace(-0.5, math.log10(eos.p_max*0.92), 64)
    # add dense near expected max
    Pcs = np.unique(np.concatenate([Pcs, np.linspace(40, 400, 24)]))
    Pcs = Pcs[Pcs <= eos.p_max*0.92]
    Pcs = np.sort(Pcs)
    results = []
    for Pc in Pcs:
        try:
            M,R,k2,Lam = solve_tov_tidal(eos, float(Pc), rtol=rtol, atol=atol)
        except Exception:
            continue
        if 0.4 < M < 4 and 6 < R < 30 and k2>=0 and Lam>=0:
            results.append((float(Pc), M,R,k2,Lam))
    if not results:
        return {"y_target": y_target, "status": "NO_STABLE_BRANCH", "eos": {"p_max": eos.p_max}}
    # stable branch up to Mmax
    masses = np.array([r[1] for r in results])
    idx_max = int(np.argmax(masses))
    stable = results[:idx_max+1]
    masses_s = np.array([r[1] for r in stable])
    radii_s = np.array([r[2] for r in stable])
    k2_s = np.array([r[3] for r in stable])
    lam_s = np.array([r[4] for r in stable])
    Mmax = float(masses_s.max())
    # R14, Lam14
    R14 = None; Lam14=None; k214=None
    if masses_s.min() < 1.4 < masses_s.max():
        order = np.argsort(masses_s)
        R14 = float(np.interp(1.4, masses_s[order], radii_s[order]))
        Lam14 = float(np.interp(1.4, masses_s[order], lam_s[order]))
        k214 = float(np.interp(1.4, masses_s[order], k2_s[order]))
    # binary tilde
    def lam_of(m):
        if m < masses_s.min() or m > masses_s.max():
            return None
        order = np.argsort(masses_s)
        return float(np.interp(m, masses_s[order], lam_s[order]))
    L136 = lam_of(1.36)
    L14b = lam_of(1.40)
    L146 = lam_of(1.46)
    L127 = lam_of(1.27)
    Lt_sym = None; Lt_asym=None
    if L136 is not None:
        Lt_sym = (16/13)*((1.36+12*1.36)*1.36**4*L136 + (1.36+12*1.36)*1.36**4*L136)/(2.72**5)
        # symmetric symmetric symmetric symmetric symmetric symmetric symmetric symmetric symmetric symmetric so we can just call formula: (16/13) *2*(13.36)*1.36^4*L136 /M^5 ; same as above
    if L146 is not None and L127 is not None:
        Mtot=1.46+1.27
        Lt_asym = (16/13)*((1.46+12*1.27)*1.46**4*L146 + (1.27+12*1.46)*1.27**4*L127)/Mtot**5

    return {
        "y_target": y_target,
        "gamma": entry["gamma_for_L50"],
        "C0": entry["C0_MeVminus2"],
        "eos_p_max": eos.p_max,
        "M_max_Msun": Mmax,
        "M_max_Pc": float(stable[idx_max][0]),
        "R_at_Mmax_km": float(stable[idx_max][2]),
        "R_14_km": R14,
        "Lambda_14": Lam14,
        "k2_14": k214,
        "Lambda_136": L136,
        "Lambda_tilde_sym_136": Lt_sym,
        "Lambda_tilde_asym_146_127": Lt_asym,
        "n_points_stable": len(stable),
        "rtol": rtol, "atol": atol,
    }


def calculate():
    rows = []
    for y in ("0.87","0.88","0.90","0.93"):
        rows.append(branch_for_y(y, rtol=1e-8, atol=1e-10))
    # convergence check: rerun loose vs tight for y=0.87
    loose = branch_for_y("0.87", rtol=1e-6, atol=1e-8)
    tight = branch_for_y("0.87", rtol=1e-10, atol=1e-12)
    conv = {
        "loose_Mmax": loose["M_max_Msun"] if "M_max_Msun" in loose else None,
        "tight_Mmax": tight["M_max_Msun"] if "M_max_Msun" in tight else None,
        "terminal_Mmax": rows[0]["M_max_Msun"] if "M_max_Msun" in rows[0] else None,
    }
    # held-out descriptive comparisons (not evidence)
    heldout = []
    for r in rows:
        if "M_max_Msun" not in r:
            continue
        Mmax=r["M_max_Msun"]; R14=r["R_14_km"]; Lam=r["Lambda_14"]; Lt=r["Lambda_tilde_sym_136"]
        heldout.append({
            "y_target": r["y_target"],
            "Mmax_vs_J0740_201": "PASS" if Mmax >= 2.01 else "FAIL",
            "R14_vs_NICER_11_2_13_2": "PASS" if R14 is not None and 11.2 <= R14 <= 13.2 else ("FAIL" if R14 is not None else "NO_R14"),
            "Lambda_vs_GW170817_70_720": "PASS" if Lt is not None and 70 <= Lt <= 720 else ("FAIL" if Lt is not None else "NO_LT"),
            "Mmax": Mmax, "R14": R14, "Lambda_14": Lam, "Lambda_tilde_sym": Lt,
        })
    return {
        "crust": {"p_match_MeV_fm3": P_MATCH, "Gamma": GAMMA_CRUST, "note": "identical for all y, not tuned"},
        "branches": rows,
        "convergence_y087": conv,
        "heldout_descriptive": heldout,
        "interpretation": "Forward beta EOS with gamma(L=50) is not tuned to stars; Mmax/R/Lambda are held-out consequences. Evidence weight 0.",
    }


def build_result():
    sci = calculate()
    # also verify forward table hash
    fwd_sha = hashlib.sha256(FORWARD_RESULT.read_bytes()).hexdigest()
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "forward_beta_sha256": fwd_sha,
        "science": sci,
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
        RESULT_PATH.write_text(json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
