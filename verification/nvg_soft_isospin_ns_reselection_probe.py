#!/usr/bin/env python3
"""Soft-isospin NS-family re-selection probe (conditional, zero evidence).

Follow-up to the isovector transferability probe (2026-09-17): transferring the
finite-branch constant j = 11.136 MeV into the two-component saturated-vector
family (C_rho -> 2 j / n0 = 139.2002 MeV fm^3) keeps M_max >= 2.01 (it rises to
2.095) but breaks the secondary screening bands (R_1.4 = 13.45 km > 13.2;
Lambda_tilde ~ 940/870 > 720) because the tied saturation calibration
compensates the softer isospin by raising c_omega0 (1794 -> 2074 MeV fm^3).

Pre-registered question (fixed here before any chain evaluation):

    With the isovector coupling pinned to the identified j (C_rho = 139.2002),
    the saturation calibration intact, and alpha_v/nu_v/cs2_q at their
    canonical values, does the saturated-vector + CSS family admit ANY
    configuration on the declared grid below that passes all three canonical
    screening constraints (J0740 M_max >= 2.01; GW170817 70 <= Lambda_tilde
    <= 720 with Lambda_tilde = max of the two binary configurations; NICER
    11.2 <= R_1.4 <= 13.2)?

Pre-registered grid (declared before any evaluation in this probe):

    Stage 2 (full family): k1 x k2 x Cs x n_trans_ratio x delta_eps_ratio
        k1    in (0.15, 0.20, 0.25, 0.30, 0.35)
        k2    in (0.60, 0.80, 1.00)
        Cs    in (700, 800, 900, 1000, 1100) MeV fm^3
        n_tr  in (1.4, 1.6, 1.8, 2.0, 2.2, 2.4)      (x n_0)
        de    in (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)  (x eps_trans)
    Stage 1 (drop-in transition re-selection) is the subset with the baseline
    isoscalar shape (k1=0.25, k2=0.80, Cs=900): it answers whether the
    transition parameters ALONE can absorb the stiffness shift.

Pre-registered verdict bands:

    SOFT_ISOSPIN_SURVIVOR_FOUND  - at least one evaluated Stage-2 chain passes
                                   all three constraints at the same 72-point
                                   sequence resolution as the retained
                                   transferability artifact.
    NO_SURVIVOR_STIFFNESS_FLOOR  - no evaluated chain passes; the best-margin
                                   point and its binding constraint are
                                   reported instead.

Selection score: the canonical audit margin (nsa._constraint_margin over the
three constraints), identical to the canonical in-sample selection.  All
constraints are SELECTION INPUTS (J0740/GW170817/NICER), exactly as for the
canonical chain: this probe is conditional/in-sample with evidence weight zero
and is not an independent test of anything.  Production files and the
canonical NS chain are untouched; every baseline is built through a temporary
patch of the soft-module parameter dict, always restored.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_eos_beta_css_softening as soft
import nvg_isovector_transferability_probe as tp
import nvg_ns_predictive_audit as nsa

SCHEMA = "nvg_soft_isospin_ns_reselection_probe.v1"
STATUS = "COMPUTED_SOFT_ISOSPIN_RESELECTION_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

TRANSFER_RESULT_REL = Path("Lunacy/runs/isovector-transferability-2026-09-17/isovector_transferability_result.json")

# Identified isovector constant transferred into the NS sector (live-verified
# against the retained transferability artifact in _load_transfer_control).
C_RHO_TRANSFER = 2.0 * tp.J_BAR_MEV / tp.N0_FM3

# Baseline isoscalar shape (production BEST_BASELINE values).
BASELINE_ISOSCALAR = {
    "k1": float(soft.BEST_BASELINE["k1"]),
    "k2": float(soft.BEST_BASELINE["k2"]),
    "Cs": float(soft.BEST_BASELINE["Cs"]),
}
# Canonical vector self-interaction and CSS slope stay fixed (declared
# limitation of this probe: no scan over alpha_v/nu_v/cs2_q).
FIXED_VECTOR = {
    "alpha_v": float(soft.BEST_BASELINE["alpha_v"]),
    "nu_v": float(soft.BEST_BASELINE["nu_v"]),
}
CS2_Q = 1.0 / 3.0

# --- Pre-registered grids (fixed before any chain evaluation) ---------------
K1_GRID = (0.15, 0.20, 0.25, 0.30, 0.35)
K2_GRID = (0.60, 0.80, 1.00)
CS_GRID = (700.0, 800.0, 900.0, 1000.0, 1100.0)
N_TRANS_GRID = (1.4, 1.6, 1.8, 2.0, 2.2, 2.4)
DE_GRID = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
SEQUENCE_POINTS = 72  # same resolution as the retained transferability chains
ROBUSTNESS_POINTS = 120  # descriptive re-evaluation of the best point only

# Retained transferability controls (fail-closed reproduction targets).
CONTROL_BASELINE = {
    "c_rho": 600.0,
    "n_trans_ratio": 2.0,
    "delta_eps_ratio": 0.0,
    "M_max_msun": 2.047494978891693,
    "R_1.4_km": 12.534205277621984,
    "Lambda_1.4": 550.647620611349,
    "Lambda_tilde_symmetric_1.36": 644.6649557881896,
    "Lambda_tilde_asymmetric_1.46_1.27": 633.5883237604179,
}
CONTROL_TRANSFERRED = {
    "c_rho": C_RHO_TRANSFER,
    "n_trans_ratio": 2.0,
    "delta_eps_ratio": 0.0,
    "M_max_msun": 2.094671565238407,
    "R_1.4_km": 13.450001000000057,
    "Lambda_1.4": 791.7311030479473,
    "Lambda_tilde_symmetric_1.36": 940.4935584561267,
    "Lambda_tilde_asymmetric_1.46_1.27": 869.6277461578152,
}
CONTROL_REL_TOL = 1.0e-9

CONSTRAINT_NAMES = ("J0740_M_max", "GW170817_Lambda_tilde", "NICER_R14")


class SoftIsospinError(RuntimeError):
    """Fail-closed error for rejected builds, sequences or controls."""


def _num(value: Any, digits: int = 17) -> float:
    return float(format(float(value), f".{digits}g"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


@contextmanager
def _patched_baseline(**overrides: float):
    """Temporarily patch ``soft.BEST_BASELINE``; always restored."""

    missing = set(overrides) - set(soft.BEST_BASELINE)
    if missing:
        raise SoftIsospinError(f"unknown baseline parameters: {sorted(missing)}")
    saved = {key: soft.BEST_BASELINE[key] for key in overrides}
    try:
        for key, value in overrides.items():
            soft.BEST_BASELINE[key] = float(value)
        yield
    finally:
        soft.BEST_BASELINE.update(saved)


def _load_transfer_control() -> dict[str, Any]:
    """Verify the transferred coupling against the retained artifact."""

    path = ROOT / TRANSFER_RESULT_REL
    if not path.is_file():
        raise SoftIsospinError(f"retained transferability artifact not present: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "nvg_isovector_transferability_probe.v1":
        raise SoftIsospinError("retained transferability artifact schema mismatch")
    part_b = payload.get("part_b_ns_sector_transfer")
    if not isinstance(part_b, Mapping):
        raise SoftIsospinError("retained transferability artifact has no part B")
    transfer = part_b.get("cross_branch_transfer")
    if not isinstance(transfer, Mapping):
        raise SoftIsospinError("retained transferability artifact has no cross-branch block")
    cited = transfer.get("c_rho_transfer_MeV_fm3")
    if not isinstance(cited, (int, float)) or abs(float(cited) - C_RHO_TRANSFER) > 1.0e-12:
        raise SoftIsospinError("retained c_rho_transfer does not match 2 j / n0")
    return {
        "path": str(path),
        "c_rho_transfer_MeV_fm3": float(cited),
        "retuned_chain_M_max_msun": float(part_b["retuned_chain"]["M_max_msun"]),
        "baseline_chain_M_max_msun": float(part_b["baseline_chain"]["M_max_msun"]),
    }


def _chain_row(k1: float, k2: float, cs: float, n_tr: float, de: float,
                c_rho: float | None = None) -> dict[str, Any]:
    """One grid point: baseline build + hybrid + 72-point TOV sequence."""

    c_rho = C_RHO_TRANSFER if c_rho is None else float(c_rho)
    params = {"k1": float(k1), "k2": float(k2), "Cs": float(cs), "Crho": c_rho}
    row: dict[str, Any] = {
        "k1": _num(k1),
        "k2": _num(k2),
        "Cs": _num(cs),
        "c_rho": _num(c_rho),
        "n_trans_ratio": _num(n_tr),
        "delta_eps_ratio": _num(de),
    }
    try:
        with _patched_baseline(**params):
            baseline = soft.build_baseline_arrays(nn=220)
    except Exception as exc:  # noqa: BLE001 - recorded, never silent
        row["status"] = f"BASELINE_FAILED: {type(exc).__name__}"
        return row
    if baseline is None:
        row["status"] = "BASELINE_FAILED"
        return row
    row["c_omega0_MeV_fm3"] = _num(baseline["c_omega0"])
    row["y_p_at_n0"] = _num(baseline["state_n0"]["y_p"])
    try:
        eos = nsa._make_hybrid_eos(baseline, float(n_tr), float(de), CS2_Q)
    except Exception as exc:  # noqa: BLE001
        row["status"] = f"HYBRID_FAILED: {type(exc).__name__}"
        return row
    if eos is None:
        row["status"] = "HYBRID_FAILED"
        return row
    try:
        summary = nsa.evaluate_sequence(eos, central_pressure_points=SEQUENCE_POINTS)
    except Exception as exc:  # noqa: BLE001
        row["status"] = f"SEQUENCE_FAILED: {type(exc).__name__}"
        return row

    lambda_values = [
        summary.get("Lambda_tilde_symmetric_1.36"),
        summary.get("Lambda_tilde_asymmetric_1.46_1.27"),
    ]
    lambda_values = [float(v) for v in lambda_values if v is not None and nsa._finite(v)]
    constraint_summary = dict(summary)
    constraint_summary["Lambda_tilde"] = max(lambda_values) if lambda_values else None
    flags = {name: bool(nsa._constraint_passes(constraint_summary, name)) for name in CONSTRAINT_NAMES}
    margin = nsa._constraint_margin(constraint_summary, CONSTRAINT_NAMES)
    # R_1.4 / Lambda_1.4 / Lambda_tilde are None when the sequence never
    # reaches 1.4 M_sun (very soft configurations); such rows are evaluated,
    # fail every constraint that needs the missing observable, and never pass.
    row.update({
        "status": "EVALUATED",
        "M_max_msun": _num(summary["M_max"]),
        "R_1.4_km": _num(summary["R_1.4"]) if summary.get("R_1.4") is not None else None,
        "Lambda_1.4": _num(summary["Lambda_1.4"]) if summary.get("Lambda_1.4") is not None else None,
        "Lambda_tilde_symmetric_1.36": _num(lambda_values[0]) if len(lambda_values) > 0 else None,
        "Lambda_tilde_asymmetric_1.46_1.27": _num(lambda_values[1]) if len(lambda_values) > 1 else None,
        "branch_selection_status": str(summary.get("branch_selection_status", "")),
        "flags": flags,
        "all_constraints_pass": all(flags.values()),
        "margin": None if margin == float("-inf") else _num(margin),
    })
    return row


def _binding_constraint(row: Mapping[str, Any]) -> str:
    """Constraint with the smallest canonical margin component.

    Only constraints with a defined observable participate: when R_1.4 or
    Lambda_tilde is missing the sequence never reached 1.4 M_sun, which is
    already the (more severe) J0740 failure; an undefined component is a
    failure by construction, not a measure of closeness.
    """

    if row.get("status") != "EVALUATED":
        return "BUILD"
    lt_values = [
        row.get("Lambda_tilde_symmetric_1.36"),
        row.get("Lambda_tilde_asymmetric_1.46_1.27"),
    ]
    lt_values = [float(v) for v in lt_values if v is not None]
    lt = max(lt_values) if lt_values else None
    r14 = row.get("R_1.4_km")
    components: dict[str, float] = {
        "J0740_M_max": (float(row["M_max_msun"]) - 2.01) / 0.07,
    }
    if lt is not None:
        components["GW170817_Lambda_tilde"] = min((lt - 70.0) / 70.0, (720.0 - lt) / 200.0)
    if r14 is not None:
        components["NICER_R14"] = min((float(r14) - 11.2) / 0.5, (13.2 - float(r14)) / 0.5)
    return min(components, key=components.get)


def _control_chain(control: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce one retained transferability chain (fail-closed)."""

    row = _chain_row(
        BASELINE_ISOSCALAR["k1"],
        BASELINE_ISOSCALAR["k2"],
        BASELINE_ISOSCALAR["Cs"],
        float(control["n_trans_ratio"]),
        float(control["delta_eps_ratio"]),
        c_rho=float(control["c_rho"]),
    )
    if row.get("status") != "EVALUATED":
        raise SoftIsospinError(f"control chain did not evaluate: {row.get('status')}")
    for key in ("M_max_msun", "R_1.4_km", "Lambda_1.4",
                "Lambda_tilde_symmetric_1.36", "Lambda_tilde_asymmetric_1.46_1.27"):
        got = float(row[key])  # type: ignore[index]
        want = float(control[key])  # type: ignore[index]
        if abs(got - want) > CONTROL_REL_TOL * max(1.0, abs(want)):
            raise SoftIsospinError(f"control {key}: {got!r} != retained {want!r}")
    return row


def verdict(stage1_survivors: int, stage2_survivors: int) -> str:
    if stage2_survivors > 0:
        return "SOFT_ISOSPIN_SURVIVOR_FOUND"
    return "NO_SURVIVOR_STIFFNESS_FLOOR"


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    evaluated = [row for row in rows if row.get("status") == "EVALUATED"]
    if not evaluated:
        return None
    survivors = [row for row in evaluated if row["all_constraints_pass"]]
    pool = survivors if survivors else evaluated
    return max(pool, key=lambda row: (row["margin"] if row["margin"] is not None else float("-inf")))


def _robustness_recheck(row: Mapping[str, Any]) -> dict[str, Any]:
    """Descriptive 120-point re-evaluation of one grid point."""

    params = {
        "k1": float(row["k1"]),
        "k2": float(row["k2"]),
        "Cs": float(row["Cs"]),
        "Crho": C_RHO_TRANSFER,
    }
    with _patched_baseline(**params):
        baseline = soft.build_baseline_arrays(nn=220)
    if baseline is None:
        raise SoftIsospinError("robustness baseline failed")
    eos = nsa._make_hybrid_eos(baseline, float(row["n_trans_ratio"]), float(row["delta_eps_ratio"]), CS2_Q)
    if eos is None:
        raise SoftIsospinError("robustness hybrid failed")
    summary = nsa.evaluate_sequence(eos, central_pressure_points=ROBUSTNESS_POINTS)
    return {
        "sequence_points": ROBUSTNESS_POINTS,
        "M_max_msun": _num(summary["M_max"]),
        "R_1.4_km": _num(summary["R_1.4"]) if summary.get("R_1.4") is not None else None,
        "Lambda_1.4": _num(summary["Lambda_1.4"]) if summary.get("Lambda_1.4") is not None else None,
        "Lambda_tilde_symmetric_1.36": (
            _num(summary["Lambda_tilde_symmetric_1.36"])
            if summary.get("Lambda_tilde_symmetric_1.36") is not None else None
        ),
        "Lambda_tilde_asymmetric_1.46_1.27": (
            _num(summary["Lambda_tilde_asymmetric_1.46_1.27"])
            if summary.get("Lambda_tilde_asymmetric_1.46_1.27") is not None else None
        ),
        "note": "descriptive resolution robustness of the best point; verdict bands apply to the 72-point scan values",
    }


def calculate() -> dict[str, Any]:
    transfer_control = _load_transfer_control()
    control_baseline = _control_chain(CONTROL_BASELINE)
    control_transferred = _control_chain(CONTROL_TRANSFERRED)

    rows: list[dict[str, Any]] = []
    for k1 in K1_GRID:
        for k2 in K2_GRID:
            for cs in CS_GRID:
                for n_tr in N_TRANS_GRID:
                    for de in DE_GRID:
                        rows.append(_chain_row(k1, k2, cs, n_tr, de))

    stage1_rows = [
        row for row in rows
        if (row["k1"], row["k2"], row["Cs"]) == (
            _num(BASELINE_ISOSCALAR["k1"]),
            _num(BASELINE_ISOSCALAR["k2"]),
            _num(BASELINE_ISOSCALAR["Cs"]),
        )
    ]
    evaluated = [row for row in rows if row.get("status") == "EVALUATED"]
    stage1_evaluated = [row for row in stage1_rows if row.get("status") == "EVALUATED"]
    stage1_survivors = [row for row in stage1_evaluated if row["all_constraints_pass"]]
    stage2_survivors = [row for row in evaluated if row["all_constraints_pass"]]
    best = _best_row(rows)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "with the isovector coupling pinned to the identified j "
            "(C_rho = 2 j / n0), does the saturated-vector + CSS family admit "
            "a configuration passing all three canonical screening constraints "
            "on the pre-registered grid?"
        ),
        "retained_inputs": {"isovector_transferability": transfer_control},
        "fixed_inputs": {
            "c_rho_transfer_MeV_fm3": _num(C_RHO_TRANSFER),
            "j_bar_MeV": _num(tp.J_BAR_MEV),
            "alpha_v": _num(FIXED_VECTOR["alpha_v"]),
            "nu_v": _num(FIXED_VECTOR["nu_v"]),
            "cs2_q": _num(CS2_Q),
            "sequence_points": SEQUENCE_POINTS,
            "baseline_isoscalar": {key: _num(value) for key, value in BASELINE_ISOSCALAR.items()},
        },
        "pre_registered_grid": {
            "k1": list(K1_GRID),
            "k2": list(K2_GRID),
            "Cs": list(CS_GRID),
            "n_trans_ratio": list(N_TRANS_GRID),
            "delta_eps_ratio": list(DE_GRID),
            "stage1_subset": "baseline isoscalar shape (k1=0.25, k2=0.80, Cs=900)",
        },
        "constraints": {
            name: {
                "label": nsa.OBSERVABLE_CONSTRAINTS[name]["label"],
                "lower": nsa.OBSERVABLE_CONSTRAINTS[name]["lower"],
                "upper": nsa.OBSERVABLE_CONSTRAINTS[name]["upper"],
            }
            for name in CONSTRAINT_NAMES
        },
        "controls": {
            "baseline_chain_C_rho_600": control_baseline,
            "transferred_chain_C_rho_139_2": control_transferred,
            "rel_tol": CONTROL_REL_TOL,
            "note": "both retained transferability chains reproduced exactly before any scan row was accepted",
        },
        "scan": {
            "evaluated": len(evaluated),
            "grid_total": len(rows),
            "build_failures": len(rows) - len(evaluated),
            "stage1_evaluated": len(stage1_evaluated),
            "stage1_survivors": len(stage1_survivors),
            "stage2_survivors": len(stage2_survivors),
        },
        "rows": rows,
    }

    if best is not None:
        best_point = dict(best)
        best_point.pop("flags", None)
        best_point["binding_constraint"] = _binding_constraint(best)
        best_point["robustness_120_points"] = _robustness_recheck(best)
        with _patched_baseline(
            k1=float(best["k1"]), k2=float(best["k2"]), Cs=float(best["Cs"]), Crho=C_RHO_TRANSFER
        ):
            symmetry = tp.symmetry_energy_summary(C_RHO_TRANSFER)
        best_point["symmetry_energy_descriptive"] = {
            "J_diff_MeV": _num(symmetry["J_diff_MeV"]),
            "J_curv_MeV": _num(symmetry["J_curv_MeV"]),
            "L_MeV": _num(symmetry["L_MeV"]),
            "nonparabolicity_MeV": _num(symmetry["nonparabolicity_MeV"]),
        }
        payload["best_point"] = best_point
    else:
        payload["best_point"] = None

    payload["verdict"] = verdict(len(stage1_survivors), len(stage2_survivors))
    payload["verdict_note"] = (
        "conditional/in-sample family-viability diagnostic: the three constraints are "
        "selection inputs exactly as for the canonical chain; evidence weight zero"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="opt-in path for the strict-JSON result artifact")
    args = parser.parse_args(argv)
    payload = _jsonable(calculate())
    text = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"artifact written: {args.output}")
    print(json.dumps({
        "schema": payload["schema"],
        "status": payload["status"],
        "verdict": payload["verdict"],
        "scan": payload["scan"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
