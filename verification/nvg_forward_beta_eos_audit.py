#!/usr/bin/env python3
"""Forward beta-equilibrated EOS on the globally-stable isoscalar jet + density-dependent isovector term.

No fit to data is performed. The isoscalar sector is read-only from
source_complete_scaling_saturation_audit (inverse_potential_jet at fixed y*,
K=240, n0=0.16, E/A=-16). The isovector coupling at n0 is fixed by J=32
and the running exponent gamma is the tabulated value that brings L to 50
(L = L_kin + 3 C0 n0/8 (1+gamma)). The EOS at n != n0 is then a pure
forward prediction: beta equilibrium (mu_n = mu_p + mu_e) with electrons
and muons, no re-tuning of Cv or the scalar potential.

Evidence weight is 0. S(2n0) and the p(eps) table are forward consequences,
not calibration targets. A crust is not matched here; low-density points
are kept as symmetric/beta matter and the TOV integration is deferred to a
separate held-out step that will consume this table with an explicit crust.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mpmath as mp

try:
    from source_complete_scaling_saturation_audit import BulkModel, inverse_potential_jet
except ImportError:
    from .source_complete_scaling_saturation_audit import BulkModel, inverse_potential_jet

try:
    from nvg_isospin_jet_audit import IsospinJet, local_jet
except ImportError:
    from .nvg_isospin_jet_audit import IsospinJet, local_jet  # type: ignore

HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
RESULT_PATH = HERE / "nvg_forward_beta_eos_results.json"
SCHEMA = "nvg_forward_beta_eos.v1"
STATUS = "FORWARD_BETA_EOS_COMPUTED_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

Y_GRID = ("0.87", "0.88", "0.90", "0.93")
PRIMARY_DPS = 80
CONTROL_DPS = 110

M_E = mp.mpf("0.511")
M_MU = mp.mpf("105.658")


def _num(v, digits=28):
    return mp.nstr(mp.mpf(v), digits)


def _mu_e_from_np(n_p_nat):
    if n_p_nat <= 0:
        return M_E
    # solve ne + nmu = n_p for common mu
    # quick estimate
    ke_est = (3 * mp.pi ** 2 * n_p_nat) ** (mp.mpf(1) / 3)
    mu_guess = mp.sqrt(ke_est ** 2 + M_E ** 2)
    if mu_guess <= M_MU:
        # only electrons needed, analytic
        # mu = sqrt( (3 pi^2 n_p)^{2/3} + m_e^2 )
        return mu_guess
    # both species: bisection on mu
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
    # bisection
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < mp.mpf("1e-30"):
            break
    return (lo + hi) / 2


def _lepton_eps_press(mu):
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


def _build_beta_table(y_text, dps):
    with mp.workdps(dps):
        model, state0, jet = inverse_potential_jet(y_text, "240", "0.16", "-16")
        n0_nat = state0["n"]
        # fixed-J isovector coupling at n0
        lj0 = local_jet(state0, 0)
        T = lj0["T"]
        C0 = 8 * (mp.mpf(32) - T) / n0_nat
        lj = local_jet(state0, C0)
        L_kin = lj["L_kin"]
        L_rho = lj["L"] - lj["L_kin"]
        gamma = (mp.mpf(50) - L_kin) / L_rho - 1 if L_rho != 0 else mp.mpf("nan")
        hbarc = model.hbarc

        def C_of_n(n):
            return C0 * (n / n0_nat) ** gamma

        # density grid: 0.15 .. 6 n0, 48 points (enough for EOS shape without heavy cost)
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
                mu_n = st["mu_n"]
                mu_p = st["mu_p"]
                n_p = n_nat * (1 - d) / 2
                mu_e = _mu_e_from_np(n_p)
                return mu_n - mu_p - mu_e

            # bracket delta in [0.5, 0.99]
            d_lo = mp.mpf("0.50")
            d_hi = mp.mpf("0.99")
            f_lo = f_delta(d_lo)
            f_hi = f_delta(d_hi)
            # expand if not bracketed (should be bracketed for this density range)
            if f_lo * f_hi > 0:
                # try wider
                d_lo = mp.mpf("0.30")
                f_lo = f_delta(d_lo)
                if f_lo * f_hi > 0:
                    rows.append({"x_n0": _num(xf), "status": "NO_BRACKET", "f_lo": _num(f_lo), "f_hi": _num(f_hi)})
                    continue
            # bisection
            for _ in range(70):
                d_mid = (d_lo + d_hi) / 2
                f_mid = f_delta(d_mid)
                if f_lo * f_mid <= 0:
                    d_hi = d_mid
                    f_hi = f_mid
                else:
                    d_lo = d_mid
                    f_lo = f_mid
                if d_hi - d_lo < mp.mpf("1e-14"):
                    break
            d_root = (d_lo + d_hi) / 2
            st = jet_m.equilibrium(n_nat, d_root)
            n_p = n_nat * (1 - d_root) / 2
            mu_e = _mu_e_from_np(n_p)
            lep_eps, lep_p = _lepton_eps_press(mu_e)
            eps_nat = st["energy_total"] + lep_eps
            p_nat = st["pressure_total"] + lep_p
            eps_fm = eps_nat / hbarc ** 3
            p_fm = p_nat / hbarc ** 3
            rows.append({
                "x_n0": _num(xf),
                "n_fm3": _num(n_nat / hbarc ** 3),
                "delta_beta": _num(d_root),
                "y_beta": _num(st["y"]),
                "mu_e_MeV": _num(mu_e),
                "C_MeVminus2": _num(C),
                "eps_MeV_fm3": _num(eps_fm),
                "P_MeV_fm3": _num(p_fm),
                "eps_nat": _num(eps_nat),
                "P_nat": _num(p_nat),
            })

        # derived diagnostics: cs2 via finite diff on p(eps)
        # use central diff where possible
        cs2_rows = []
        for i, r in enumerate(rows):
            if "P_MeV_fm3" not in r or "eps_MeV_fm3" not in r:
                cs2_rows.append(None)
                continue
            if i == 0 or i == len(rows) - 1:
                cs2_rows.append(None)
                continue
            # need neighbours valid
            if "P_MeV_fm3" not in rows[i-1] or "P_MeV_fm3" not in rows[i+1]:
                cs2_rows.append(None)
                continue
            p_m = mp.mpf(rows[i-1]["P_MeV_fm3"])
            p_p = mp.mpf(rows[i+1]["P_MeV_fm3"])
            e_m = mp.mpf(rows[i-1]["eps_MeV_fm3"])
            e_p = mp.mpf(rows[i+1]["eps_MeV_fm3"])
            if e_p == e_m:
                cs2_rows.append(None)
            else:
                cs2 = (p_p - p_m) / (e_p - e_m)
                cs2_rows.append(_num(cs2))
        # attach
        for r, cs2 in zip(rows, cs2_rows):
            if cs2 is not None:
                r["cs2_fd"] = cs2

        # extract n0 row for quick check
        n0_row = next((r for r in rows if r["x_n0"] == _num(mp.mpf("1.00"))), None)
        x2_row = next((r for r in rows if r["x_n0"] == _num(mp.mpf("2.00"))), None)
        return {
            "y_target": y_text,
            "y_eq": _num(state0["y"]),
            "C0_MeVminus2": _num(C0),
            "gamma_for_L50": _num(gamma),
            "alpha_for_L50": _num(-gamma),
            "C_ratio_at_2n0": _num(C_of_n(2 * n0_nat) / C0),
            "globally_nonnegative_quartic": bool(jet["globally_nonnegative_potential"]),
            "q_min_MeV4": _num(jet["q_min"]) if jet["q_min"] is not None else None,
            "P_at_n0_beta_MeV_fm3": n0_row["P_MeV_fm3"] if n0_row and "P_MeV_fm3" in n0_row else None,
            "eps_at_n0_beta_MeV_fm3": n0_row["eps_MeV_fm3"] if n0_row else None,
            "delta_at_n0_beta": n0_row["delta_beta"] if n0_row else None,
            "P_at_2n0_beta_MeV_fm3": x2_row["P_MeV_fm3"] if x2_row and "P_MeV_fm3" in x2_row else None,
            "max_cs2_fd": max([float(mp.mpf(r["cs2_fd"])) for r in rows if "cs2_fd" in r], default=None),
            "table": rows,
        }


def calculate(dps=PRIMARY_DPS):
    if dps < 80:
        raise ValueError("need at least 80 dps")
    with mp.workdps(dps):
        base = BulkModel()
        s0 = base.equilibrium(base.n0)
        baseline = {
            "y": _num(s0["y"]),
            "P_MeV_fm3": _num(s0["pressure_total"] / base.hbarc ** 3),
            "binding_MeV": _num(s0["energy_total"] / s0["n"] - base.MN),
            "K_MeV": _num(s0["K"]),
        }
        tables = []
        for y_text in Y_GRID:
            tables.append(_build_beta_table(y_text, dps))
        return {
            "baseline_at_n0_source_complete": baseline,
            "empirical_context": {
                "n0_fm3": "0.16",
                "J_MeV": "32 (fixes C0)",
                "L_MeV": "50 (fixes gamma, not fitted beyond)",
                "K_MeV": "240 (fixes jet)",
                "note": "All three are inputs to the read-only jet; EOS at n!=n0 is forward, P(n0)_beta !=0 is expected (neutron-rich matter not symmetric)",
            },
            "forward_tables": tables,
            "identities": "BulkModel read-only; IsospinJet equilibrium; beta mu_n=mu_p+mu_e solved by bisection; C(n)=C0 (n/n0)^gamma",
            "what_is_not_done": "No crust matching; no TOV integration in this audit; TOV+Hinderer held-out is a separate step consuming this p(eps) with an explicit low-density crust",
        }


def build_result():
    a = calculate(PRIMARY_DPS)
    b = calculate(CONTROL_DPS)
    if a != b:
        import json as _js
        sa = _js.dumps(a, sort_keys=True)
        sb = _js.dumps(b, sort_keys=True)
        for i, (ca, cb) in enumerate(zip(sa, sb)):
            if ca != cb:
                raise ArithmeticError(f"80/110 digit mismatch at {i}: {sa[i-40:i+40]!r} vs {sb[i-40:i+40]!r}")
        raise ArithmeticError("80/110 digit results differ")
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
    p.add_argument("--write", action="store_true", help="write verification/nvg_forward_beta_eos_results.json")
    args = p.parse_args()
    res = build_result()
    if args.write:
        data = json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        RESULT_PATH.write_text(data, encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
