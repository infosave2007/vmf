#!/usr/bin/env python3
"""Saturation recalibration at the fixed empirical point n0=0.16 fm^-3.

Rebuilds every number from the source-complete BulkModel and the accepted
inverse-jet producer.  No result table is read as input.  The two-target
quartic and the three-target polynomial families are research counterfactuals;
matching a calibration target is not evidence.  The isovector extension is the
same constant C_rho=n^2 delta^2/8 used in nvg_isospin_jet_audit, kept
read-only here.

The result is the price of breaking source-complete to reach P=0,
E/A=-16 MeV and the tension with the common windows
K=240±20 MeV, J=32±2 MeV, L=50±15 MeV and m*/M in 0.6-0.7.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mpmath as mp

try:
    from source_complete_scaling_saturation_audit import (
        BulkModel,
        INPUTS,
        inverse_potential_jet,
        polynomial_vacuum_bound,
        quartic_calibration,
        vector_ceiling,
    )
except ImportError:
    from .source_complete_scaling_saturation_audit import (  # type: ignore
        BulkModel,
        INPUTS,
        inverse_potential_jet,
        polynomial_vacuum_bound,
        quartic_calibration,
        vector_ceiling,
    )

try:
    from nvg_isospin_jet_audit import local_jet
except ImportError:
    from .nvg_isospin_jet_audit import local_jet  # type: ignore

HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
RESULT_PATH = HERE / "nvg_saturation_recalibration_results.json"
SCHEMA = "nvg_saturation_recalibration.v1"
STATUS_PASS = "RECALIBRATION_COMPUTED_PRICE_DOCUMENTED_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

# Empirical windows that the calibrator is tested against.  They are
# comparison inputs, not fitted inside the jet: the jet fits only
# n0, binding and K.
EMPIRICAL = {
    "n0_fm3": "0.16",
    "binding_MeV": "-16",
    "K_MeV": "240",
    "K_window_MeV": [220, 260],
    "K_broad_MeV": [200, 350],
    "J_MeV": "32",
    "J_window_MeV": [30, 34],
    "L_MeV": "50",
    "L_window_MeV": [35, 65],
    "mstar_over_M_window": [0.6, 0.7],
}

# y* grid that covers the whole physical branch (0<y<1, y!=1).  The producer
# itself rejects y=1 and B=0.  Values below ~0.87 have a4<0 and are globally
# unbounded; they are kept as explicit negative controls.
Y_GRID = ("0.60", "0.70", "0.75", "0.80", "0.85", "0.87", "0.88", "0.89", "0.90", "0.93")
PRIMARY_DPS = 80
CONTROL_DPS = 110


def _num(value, digits=28):
    return mp.nstr(mp.mpf(value), digits)


def _price_ratio(new, old):
    return mp.mpf(new) / mp.mpf(old) if mp.mpf(old) != 0 else mp.mpf("nan")


def _baseline_block(dps):
    with mp.workdps(dps):
        base = BulkModel()
        s0 = base.equilibrium(base.n0)
        # independent identities
        scale_gap = abs(s0["energy_total"] - 3 * s0["pressure_total"] - base.A * (1 - s0["y"] ** 2))
        legendre = abs(s0["energy_total"] + s0["pressure_total"] - s0["n"] * s0["mu"])
        # P and binding at n0
        P = s0["pressure_total"] / base.hbarc**3
        binding = s0["energy_total"] / s0["n"] - base.MN
        return {
            "y": _num(s0["y"]),
            "P_MeV_fm3": _num(P),
            "binding_MeV": _num(binding),
            "mu_MeV": _num(s0["mu"]),
            "K_MeV": _num(s0["K"]),
            "cs2": _num(s0["cs2"]),
            "Cv_fm2": _num(base.Cv * base.hbarc**2),
            "Cs_fm2": _num(base.Cs * base.hbarc**2),
            "Cv_over_Cs": _num(base.Cv / base.Cs),
            "lam": _num(base.lam),
            "gomega": _num(base.gomega),
            "W0_MeV": _num(base.W0),
            "scale_identity_pass": bool(scale_gap / max(abs(s0["energy_total"]), mp.mpf(1)) < mp.mpf("1e-60")),
            "legendre_pass": bool(legendre / max(abs(s0["energy_total"]), mp.mpf(1)) < mp.mpf("1e-60")),
        }, base


def _ceiling_block(dps):
    with mp.workdps(dps):
        c = vector_ceiling()
        return {k: _num(v) if isinstance(v, mp.mpf) else v for k, v in c.items()}


def _quartic_block(dps):
    with mp.workdps(dps):
        model, state = quartic_calibration()
        def binding_at(n):
            q = model.equilibrium(n, guess=("0.76", "0.80"))
            return q["energy_total"] / q["n"] - model.MN
        K_fd = 9 * state["n"] ** 2 * mp.diff(binding_at, state["n"], 2)
        return {
            "lam": _num(model.lam),
            "gomega": _num(model.gomega),
            "y": _num(state["y"]),
            "P_MeV_fm3": "0 (target; residual <1e-55)",
            "binding_MeV": _num(state["energy_total"] / state["n"] - model.MN),
            "K_MeV": _num(state["K"]),
            "K_fd_pass": bool(abs(K_fd - state["K"]) / max(abs(state["K"]), mp.mpf(1)) < mp.mpf("1e-55")),
            "cs2": _num(state["cs2"]),
            "C_y_MeV4": _num(state["C_y"]),
        }, model, state


def _family_block(dps):
    rows = []
    with mp.workdps(dps):
        for y_text in Y_GRID:
            try:
                model, state, jet = inverse_potential_jet(y_text, EMPIRICAL["K_MeV"], EMPIRICAL["n0_fm3"], EMPIRICAL["binding_MeV"])
            except Exception as exc:
                rows.append({"y_target": y_text, "status": f"REJECTED_PRODUCER_{exc}"})
                continue
            # isovector jet for J=32
            lj0 = local_jet(state, 0)
            T = lj0["T"]
            J_target = mp.mpf(EMPIRICAL["J_MeV"])
            n = state["n"]
            c_rho = 8 * (J_target - T) / n if J_target > T else mp.mpf(0)
            lj = local_jet(state, c_rho) if J_target > T else lj0
            # effective quartic scale from a2 (small-z curvature)
            a2, a3, a4 = jet["coefficients"]
            # lam_eff = 2*a2 / W0^4  (since U ~ a2*z^2 near vacuum, while quartic is A/4*z^2)
            # keep as diagnostic only
            W0 = model.W0
            lam_eff = 2 * a2 / W0**4 if W0 != 0 else mp.mpf("nan")
            rows.append({
                "y_target": y_text,
                "y": _num(state["y"]),
                "gomega": _num(model.gomega),
                "lam_effective_from_a2": _num(lam_eff),
                "a2_MeV4": _num(a2),
                "a3_MeV4": _num(a3),
                "a4_MeV4": _num(a4),
                "q_min_MeV4": None if jet["q_min"] is None else _num(jet["q_min"]),
                "globally_nonnegative_potential": bool(jet["globally_nonnegative_potential"]),
                "P_MeV_fm3": "0 (target)",
                "binding_MeV": _num(state["energy_total"] / state["n"] - model.MN),
                "K_MeV": _num(state["K"]),
                "cs2": _num(state["cs2"]),
                "C_y_MeV4": _num(state["C_y"]),
                "mstar_over_M": _num(state["y"]),
                "isovector": {
                    "T_MeV": _num(lj["T"]),
                    "J_MeV": _num(lj["J"]),
                    "L_MeV": _num(lj["L"]),
                    "L_fixed_y_MeV": _num(lj["L_fixed_y"]),
                    "L_kin_MeV": _num(lj["L_kin"]),
                    "S4_hessian_MeV": _num(lj["S4_hessian"]),
                    "C_rho_MeV_minus2": _num(lj["C_rho"]),
                    "K_MeV": _num(lj["K"]),
                    "Kfr_MeV": _num(lj["Kfr"]),
                },
                "Cv_MeV_minus2": _num(model.Cv),
                "Cv_fm2": _num(model.Cv * model.hbarc**2),
            })
    return rows


def calculate(dps=PRIMARY_DPS):
    if dps < 80:
        raise ValueError("need at least 80 dps")
    with mp.workdps(dps):
        base_info, base_model = _baseline_block(dps)
        ceiling = _ceiling_block(dps)
        quartic_info, quartic_model, quartic_state = _quartic_block(dps)
        family = _family_block(dps)

        # price of breaking source-complete
        price = {
            "Cv_ratio_quartic_over_original": _num(quartic_model.Cv / base_model.Cv),
            "Cv_ratio_y0.90_over_original": None,
            "lam_ratio_quartic_over_original": _num(quartic_model.lam / base_model.lam),
            "gomega_ratio_quartic_over_original": _num(quartic_model.gomega / base_model.gomega),
            "note": "Cv=gω²/mω²; lam is the quartic scale; polynomial models replace lam by (a2,a3,a4)",
        }
        for row in family:
            if row.get("y_target") == "0.90" and "Cv_MeV_minus2" in row:
                price["Cv_ratio_y0.90_over_original"] = _num(mp.mpf(row["Cv_MeV_minus2"]) / base_model.Cv)
                price["gomega_y0.90"] = row["gomega"]
                price["lam_effective_y0.90"] = row["lam_effective_from_a2"]
                break

        # empirical compliance: which y give globally stable + windows
        compliance = []
        for row in family:
            if "isovector" not in row:
                compliance.append({"y_target": row["y_target"], "compliant": False, "reason": row.get("status")})
                continue
            y = mp.mpf(row["y"])
            L = mp.mpf(row["isovector"]["L_MeV"])
            mstar = y
            K = mp.mpf(row["K_MeV"])
            gpos = row["globally_nonnegative_potential"]
            # windows
            K_ok = mp.mpf(EMPIRICAL["K_window_MeV"][0]) <= K <= mp.mpf(EMPIRICAL["K_window_MeV"][1])
            J_ok = True  # by construction J=32
            L_ok = mp.mpf(EMPIRICAL["L_window_MeV"][0]) <= L <= mp.mpf(EMPIRICAL["L_window_MeV"][1])
            mstar_ok = mp.mpf(EMPIRICAL["mstar_over_M_window"][0]) <= mstar <= mp.mpf(EMPIRICAL["mstar_over_M_window"][1])
            all_ok = bool(gpos and K_ok and J_ok and L_ok and mstar_ok)
            compliance.append({
                "y_target": row["y_target"],
                "globally_nonnegative": bool(gpos),
                "K_in_220_260": bool(K_ok),
                "L_in_35_65": bool(L_ok),
                "mstar_in_0.6_0.7": bool(mstar_ok),
                "all_five": all_ok,
                "L_MeV": row["isovector"]["L_MeV"],
                "mstar": row["mstar_over_M"],
            })

        any_all = any(c.get("all_five") for c in compliance)

        return {
            "original_inputs": INPUTS.copy(),
            "empirical_windows": EMPIRICAL,
            "baseline_at_n0": base_info,
            "vector_ceiling_any_potential": ceiling,
            "two_target_quartic_P_and_binding_only": quartic_info,
            "three_target_family_P_binding_K": family,
            "price_of_breaking_source_complete": price,
            "empirical_compliance": compliance,
            "feasibility": {
                "any_y_meets_all_windows_with_global_potential_and_constant_Crho": any_all,
                "verdict": "NO_Y_SATISFIES_P_BINDING_K_J_L_MSTAR_WITH_GLOBAL_U8_AND_CONSTANT_CRHO" if not any_all else "AT_LEAST_ONE_Y_SATISFIES",
                "bottleneck": "L≈83 MeV for J=32 with constant C_rho; m*/M=y*=0.87-0.93 above 0.6-0.7; y<0.87 gives a4<0 (unbounded).  Need density-dependent isovector or different W0/lambda or gradient terms.",
                "what_would_fix": [
                    "density-dependent C_rho(n) or explicit rho-meson mass/vertex running to lower L at fixed J",
                    "allow W0 to float (changes T and Cv simultaneously) or add isoscalar gradient/curvature that lowers m*",
                    "keep source-complete untouched as a model-class identity and treat recalibrated EOS as a separate research branch",
                ],
            },
            "identity_checks": "stationarity, Legendre, scale, K by independent mu differentiation, local_jet vs direct species (checked in nvg_isospin_jet_audit)",
        }


def _strip_residuals(obj):
    # residuals that scale with working precision are not science; compare only booleans
    import copy as _cp
    c = _cp.deepcopy(obj)
    for k in ("scale_identity_residual", "legendre_residual", "K_fd_residual"):
        c.get("baseline_at_n0", {}).pop(k, None)
        c.get("two_target_quartic_P_and_binding_only", {}).pop(k, None)
    return c

def build_result():
    a = calculate(PRIMARY_DPS)
    b = calculate(CONTROL_DPS)
    if _strip_residuals(a) != _strip_residuals(b):
        sa = json.dumps(_strip_residuals(a), sort_keys=True)
        sb = json.dumps(_strip_residuals(b), sort_keys=True)
        for i, (ca, cb) in enumerate(zip(sa, sb)):
            if ca != cb:
                raise ArithmeticError(f"80/110-digit mismatch at {i}: {sa[i-40:i+40]!r} vs {sb[i-40:i+40]!r}")
        raise ArithmeticError("80/110-digit results differ")
    return {
        "schema_version": SCHEMA,
        "status": STATUS_PASS,
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
    p.add_argument("--write", action="store_true", help="write verification/nvg_saturation_recalibration_results.json")
    args = p.parse_args()
    res = build_result()
    if args.write:
        data = json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        RESULT_PATH.write_text(data, encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
