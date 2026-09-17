#!/usr/bin/env python3
"""Isovector transferability probe: re-anchored finite branch and NS-sector consistency.

Part A (finite-nucleus static branch, W8.93):
    The discriminating-set probe demonstrated the universal-j collapse at the
    FIXED production scale s* = 0.2268, leaving the Ca40 anchor drifted by
    -0.0024 MeV per nucleon.  This probe re-anchors: a secant recalibration
    finds s*_rho such that the Ca40 binding per nucleon returns to the accepted
    R3 anchor value under the rho contact at the universal design
    J = 23.78013451414814 MeV, then re-solves the four held-out nuclei at
    s*_rho.  Question: does the ~60x residual collapse survive a clean
    re-anchoring of the only calibration parameter?

Part B (neutron-star sector):
    The canonical NS chain (tidal.EOS, M_max = 2.047950740578197 Msun) is a
    one-component isoscalar EOS with no proton/neutron split, so the isovector
    term cannot enter it by a parameter change; this is recorded explicitly.
    The two-component dense-matter family in the repository is the
    saturated-vector beta EOS (eps_rho = 0.5 C_rho (n_n - n_p)^2,
    C_rho = 600 MeV fm^3 in the canonical baseline).  The probe (1) computes
    that model's symmetry energy S(n), J and L directly from its functional;
    (2) quantifies the cross-branch isospin mismatch against the finite-branch
    identification j = 11.136 MeV; (3) transfers the identified interaction
    piece into the NS sector (C_rho -> 2 j / n0) and re-runs the identical
    hybrid + TOV pipeline, reporting M_max, R_1.4, Lambda_1.4 against the
    pre-registered transferability bands.

Verdict bands are fixed in code before any solve in this probe.  Every result
here is a conditional in-model diagnostic with evidence weight zero: nothing
in this file is observational evidence or a theory change.  Production
solver files and parameters are not modified; the rho branch is entered only
through the same context managers as the earlier probes.
"""

from __future__ import annotations

import argparse
import json
import math
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
import nvg_eos_beta_saturated_vector as eosbase
import nvg_finite_static_bridge as bridge
import nvg_isovector_formfactor_probe as ivp
import nvg_ns_predictive_audit as nsa
import nvg_sn132_discriminating_set_probe as sn
import nvg_tidal_deformability as tidal

SCHEMA = "nvg_isovector_transferability_probe.v1"
STATUS = "COMPUTED_ISOVECTOR_TRANSFERABILITY_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

SN132_RESULT_REL = Path("Lunacy/runs/sn132-full-set-2026-09-16/sn132_discriminating_set_result.json")
NSA_RESULTS_REL = HERE / "nvg_ns_predictive_audit_results.json"

# ---------------------------------------------------------------------------
# Cited anchors (each is live-verified against its retained artifact).
# ---------------------------------------------------------------------------

# Accepted R3 no-rho anchor value for Ca40 (ivp.BASELINE_EXPECTED); the rho
# branch is re-calibrated to return exactly this binding per nucleon.
CA40_ANCHOR_BA_MEV = 8.551607091172059
# Universal isovector constant from the completed discriminating set
# (retained artifact universal_j_demonstration.j_bar_MeV).
J_BAR_MEV = 11.13601607117819
# Canonical one-component NS chain (nvg_ns_predictive_audit_results.json
# /canonical): the flagship M_max the transferability question must not ruin.
CANONICAL_CHAIN = {
    "M_max_msun": 2.047950740578197,
    "R_1.4_km": 12.550001000000044,
    "Lambda_1.4": 519.4223807918132,
}
NS_RHO_BASELINE = float(soft.BEST_BASELINE["Crho"])  # 600 MeV fm^3
N0_FM3 = float(eosbase.n_0)

# ---------------------------------------------------------------------------
# Pre-registered verdict bands (fixed before any solve in this probe).
# ---------------------------------------------------------------------------

# Part A: re-anchoring must be numerically exact and small; the collapse must
# survive with per-nucleus and total residuals far below the pre-rho ones
# (pre-rho totals were 12.7 / 95.3 / 16.0 / 78.7 MeV for Zr/Pb/Ca48/Sn132).
ANCHOR_RECAL_ABS_TOL_BA_MEV = 1.0e-4
DS_OVER_S_ABS_BAND = (0.0, 2.0e-3)
PART_A_TOTAL_ABS_RESIDUAL_BAND_MEV = (0.0, 10.0)
PART_A_PER_NUCLEUS_ABS_BAND_MEV = (0.0, 5.0)
PART_A_MAX_REFINEMENTS = 4

# Part B: cross-branch and TOV transferability.
BASELINE_REPRO_REL_TOL = 1.0e-9
MMAX_TRANSFERABLE_MIN_MSUN = 2.01  # J0740 declared 1-sigma low edge
MMAX_TENSION_MIN_MSUN = 1.90
J_BASELINE_SANITY_BAND_MEV = (25.0, 80.0)
L_BASELINE_SANITY_BAND_MEV = (0.0, 200.0)
SYMMETRY_GRID_MIN_FM3 = 0.08
SYMMETRY_GRID_MAX_FM3 = 0.32
SYMMETRY_GRID_STEP_FM3 = 0.02
SYMMETRY_L_STEP_FM3 = 2.0e-3

HELDOUT_NUCLEI = ("Zr90", "Pb208", "Ca48", "Sn132")
ALL_NUCLEI = ("Ca40",) + HELDOUT_NUCLEI


class TransferabilityError(RuntimeError):
    """Fail-closed error for any rejected solve or failed control."""


def _num(value: Any, digits: int = 17) -> float:
    return float(format(float(value), f".{digits}g"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# Retained-artifact verification (fail closed).
# ---------------------------------------------------------------------------


def _load_sn132_artifact() -> dict[str, Any]:
    path = ROOT / SN132_RESULT_REL
    if not path.is_file():
        raise TransferabilityError(f"retained Sn132 artifact not present: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "nvg_sn132_discriminating_set_probe.v1":
        raise TransferabilityError("retained Sn132 artifact schema mismatch")
    demo = payload.get("universal_j_demonstration")
    if not isinstance(demo, Mapping):
        raise TransferabilityError("retained Sn132 artifact has no universal_j_demonstration")
    j_bar = demo.get("j_bar_MeV")
    if not isinstance(j_bar, (int, float)) or abs(float(j_bar) - J_BAR_MEV) > 1.0e-12:
        raise TransferabilityError("retained j_bar does not match the cited constant")
    j_design = demo.get("J_design_universal_MeV")
    expected_design = sn.universal_design_j(J_BAR_MEV)
    if not isinstance(j_design, (int, float)) or abs(float(j_design) - float(expected_design)) > 1.0e-12:
        raise TransferabilityError("retained J_design_universal does not match T_CONTACT + j_bar")
    cases = demo.get("cases")
    if not isinstance(cases, Mapping) or set(cases) != set(ALL_NUCLEI):
        raise TransferabilityError("retained universal-j cases do not cover the five nuclei")
    return {
        "path": str(path),
        "j_bar_MeV": float(j_bar),
        "J_design_universal_MeV": float(j_design),
        "t_contact_MeV": float(sn.T_CONTACT_MEV),
        "cases": {
            name: {
                "binding_per_A_MeV": float(row["binding_per_A_MeV"]),
                "residual_after_per_A_MeV": float(row["residual_after_per_A_MeV"]),
                "residual_after_total_MeV": float(row["residual_after_total_MeV"]),
            }
            for name, row in cases.items()
        },
        "ca40_anchor_drift": dict(demo.get("ca40_anchor_drift") or {}),
    }


def _load_nsa_canonical() -> dict[str, Any]:
    if not NSA_RESULTS_REL.is_file():
        raise TransferabilityError(f"canonical NS audit artifact not present: {NSA_RESULTS_REL}")
    payload = json.loads(NSA_RESULTS_REL.read_text(encoding="utf-8"))
    canonical = payload.get("canonical")
    if not isinstance(canonical, Mapping):
        raise TransferabilityError("canonical NS audit artifact has no /canonical block")
    for key, expected in (
        ("M_max", CANONICAL_CHAIN["M_max_msun"]),
        ("R_1.4", CANONICAL_CHAIN["R_1.4_km"]),
        ("Lambda_1.4", CANONICAL_CHAIN["Lambda_1.4"]),
    ):
        value = canonical.get(key)
        if not isinstance(value, (int, float)) or abs(float(value) - float(expected)) > 1.0e-12:
            raise TransferabilityError(f"canonical NS artifact {key} does not match the cited constant")
    return {"path": str(NSA_RESULTS_REL), "canonical": dict(canonical)}


# ---------------------------------------------------------------------------
# Part A: re-anchored finite-nucleus branch.
# ---------------------------------------------------------------------------


def _solve_metrics(nucleus: str, scale: float, j_design: float) -> dict[str, Any]:
    """One terminal solve of ``nucleus`` at ``scale`` under the rho contact."""

    if nucleus in sn.HELDOUT:
        outer = sn._heldout_context()
    else:
        from contextlib import nullcontext

        outer = nullcontext()
    with outer:
        with ivp._rho_context(ivp.CONTACT_RHO), ivp._j_design_context(j_design):
            case = bridge._terminal_case(ivp.FAMILY, float(scale), nucleus, multi_seed=True)
        if case.get("terminal_protocol_acceptance") is not True:
            raise TransferabilityError(f"{nucleus}: terminal protocol rejected at scale {scale}")
        # _terminal_metrics reads the nucleus table, which for Ca48/Sn132 only
        # exists inside the held-out patch context.
        metrics = ivp._terminal_metrics(nucleus, case)
    if not isinstance(metrics, Mapping):
        raise TransferabilityError(f"{nucleus}: no terminal metrics")
    return dict(metrics)


def _recalibrate_anchor(j_design: float) -> dict[str, Any]:
    """Secant re-anchoring of s* on Ca40 under the rho contact.

    The first probe step uses the envelope identity dE/ds = 2 T_W / s used by
    the earlier probes; the secant then converges on the exact scale.
    """

    target = CA40_ANCHOR_BA_MEV
    s0 = float(ivp.ROOT_SCALE)
    metrics0 = _solve_metrics("Ca40", s0, j_design)
    b0 = float(metrics0["binding_per_A_MeV"])
    a40 = int(bridge._load_inputs()["binding"]["values"]["Ca40"]["A"])
    t40 = float(metrics0["T_W_MeV"])
    # dB_per_A/ds = -2 T_W / (s A)  ->  linear step towards the anchor.
    ds_linear = -(target - b0) * s0 * a40 / (2.0 * t40)
    points: list[dict[str, float]] = [{"scale": s0, "binding_per_A_MeV": b0}]
    s = s0 + ds_linear
    s_star = None
    b_star = None
    for _ in range(PART_A_MAX_REFINEMENTS):
        metrics = _solve_metrics("Ca40", s, j_design)
        b = float(metrics["binding_per_A_MeV"])
        points.append({"scale": _num(s), "binding_per_A_MeV": _num(b)})
        if abs(b - target) <= ANCHOR_RECAL_ABS_TOL_BA_MEV:
            s_star, b_star = s, b
            break
        if len(points) < 2:
            raise TransferabilityError("recalibration needs two points for a secant")
        (sa, ba), (sb, bb) = points[-2], points[-1]
        slope = (bb - ba) / (sb - sa)
        if slope == 0.0 or not math.isfinite(slope):
            raise TransferabilityError("recalibration secant slope degenerated")
        s = sb + (target - bb) / slope
    if s_star is None or b_star is None:
        raise TransferabilityError("anchor recalibration did not converge within the refinement budget")
    return {
        "target_binding_per_A_MeV": target,
        "s_production": _num(s0),
        "s_rho_reanchored": _num(s_star),
        "ds_over_s": _num((s_star - s0) / s0),
        "ds_linear_estimate_over_s": _num(ds_linear / s0),
        "anchor_residual_per_A_MeV": _num(b_star - target),
        "anchor_abs_tol_per_A_MeV": ANCHOR_RECAL_ABS_TOL_BA_MEV,
        "secant_points": points,
    }


def verdict_part_a(*, anchor_ok: bool, ds_ok: bool, total_ok: bool, per_nucleus_ok: bool) -> str:
    if anchor_ok and ds_ok and total_ok and per_nucleus_ok:
        return "REANCHORED_UNIVERSAL_J_SURVIVES"
    return "REANCHORED_UNIVERSAL_J_BREAKS_BANDS"


def _mass_number(name: str) -> int:
    """Nucleus mass number from the production table or the probe rows."""

    production = bridge._load_inputs()["binding"]["values"]
    if name in production and "A" in production[name]:
        return int(production[name]["A"])
    if name == "Ca48":
        return 48
    if name == "Sn132":
        return int(sn.SN132_NUCLEUS_ROW["A"])
    raise TransferabilityError(f"{name}: no mass number available")


def _part_a(retained: dict[str, Any]) -> dict[str, Any]:
    j_design = float(sn.universal_design_j(J_BAR_MEV))
    recal = _recalibrate_anchor(j_design)
    s_star = float(recal["s_rho_reanchored"])
    targets = sn._targets()

    cases: dict[str, dict[str, Any]] = {}
    # Ca40 row from a fresh solve at the converged s*_rho (self-consistent record).
    ca40 = _solve_metrics("Ca40", s_star, j_design)
    target = float(targets["Ca40"]["B_nuc_target_per_A_MeV"])
    binding = float(ca40["binding_per_A_MeV"])
    cases["Ca40"] = {
        "status": "ACCEPTED",
        "binding_per_A_MeV": _num(binding),
        "target_per_A_MeV": _num(target),
        "residual_per_A_MeV": _num(binding - target),
        "residual_total_MeV": _num((binding - target) * 40),
        "T_W_MeV": _num(ca40["T_W_MeV"]),
        "E_rho_MeV": _num(ca40["E_rho_MeV"]),
        "rms_neutron_radius_fm": _num(ca40["rms_neutron_radius_fm"]),
        "rms_point_proton_radius_fm": _num(ca40["rms_point_proton_radius_fm"]),
    }
    for name in HELDOUT_NUCLEI:
        metrics = _solve_metrics(name, s_star, j_design)
        a_nuc = _mass_number(name)
        target = float(targets[name]["B_nuc_target_per_A_MeV"])
        binding = float(metrics["binding_per_A_MeV"])
        residual_pa = binding - target
        cases[name] = {
            "status": "ACCEPTED",
            "A": a_nuc,
            "binding_per_A_MeV": _num(binding),
            "target_per_A_MeV": _num(target),
            "residual_per_A_MeV": _num(residual_pa),
            "residual_total_MeV": _num(residual_pa * a_nuc),
            "T_W_MeV": _num(metrics["T_W_MeV"]),
            "E_rho_MeV": _num(metrics["E_rho_MeV"]),
            "rms_neutron_radius_fm": _num(metrics["rms_neutron_radius_fm"]),
            "rms_point_proton_radius_fm": _num(metrics["rms_point_proton_radius_fm"]),
            "retained_fixed_scale_residual_total_MeV": _num(
                retained["cases"][name]["residual_after_total_MeV"]
            ),
        }

    heldout_totals = [cases[name]["residual_total_MeV"] for name in HELDOUT_NUCLEI]
    total_sum = sum(heldout_totals)
    anchor_ok = abs(recal["anchor_residual_per_A_MeV"]) <= ANCHOR_RECAL_ABS_TOL_BA_MEV
    ds_ok = DS_OVER_S_ABS_BAND[0] <= abs(recal["ds_over_s"]) <= DS_OVER_S_ABS_BAND[1]
    total_ok = PART_A_TOTAL_ABS_RESIDUAL_BAND_MEV[0] <= abs(total_sum) <= PART_A_TOTAL_ABS_RESIDUAL_BAND_MEV[1]
    per_nucleus_ok = all(
        PART_A_PER_NUCLEUS_ABS_BAND_MEV[0] <= abs(value) <= PART_A_PER_NUCLEUS_ABS_BAND_MEV[1]
        for value in heldout_totals
    )
    bands = {
        "anchor_within_tol": anchor_ok,
        "ds_over_s_within_band": ds_ok,
        "total_residual_within_band": total_ok,
        "per_nucleus_residuals_within_band": per_nucleus_ok,
    }
    return {
        "question": (
            "does the universal-j residual collapse survive re-anchoring s* on Ca40 "
            "under the rho contact at the universal design J?"
        ),
        "j_design_MeV": _num(j_design),
        "j_bar_MeV": _num(J_BAR_MEV),
        "recalibration": recal,
        "cases": cases,
        "heldout_total_residual_MeV": _num(total_sum),
        "retained_fixed_scale_heldout_total_MeV": _num(
            sum(retained["cases"][name]["residual_after_total_MeV"] for name in HELDOUT_NUCLEI)
        ),
        "bands": bands,
        "verdict": verdict_part_a(
            anchor_ok=anchor_ok, ds_ok=ds_ok, total_ok=total_ok, per_nucleus_ok=per_nucleus_ok
        ),
    }


# ---------------------------------------------------------------------------
# Part B: NS-sector symmetry-energy audit and transfer.
# ---------------------------------------------------------------------------


def _energy_per_baryon_no_leptons(n_b: float, delta: float, c_rho: float) -> float:
    """E/A of the saturated-vector functional at isospin asymmetry ``delta``.

    Leptons and the isoscalar vector part are excluded: both are absent in the
    pure nuclear-matter definition of the symmetry energy, and the isoscalar
    vector cancels in any SNM/PNM difference anyway.
    """

    k1 = float(soft.BEST_BASELINE["k1"])
    k2 = float(soft.BEST_BASELINE["k2"])
    c_s = float(soft.BEST_BASELINE["Cs"])
    n_n = n_b * (1.0 + delta) / 2.0
    n_p = n_b * (1.0 - delta) / 2.0
    m_ref = eosbase.M_base(n_b, k1, k2)
    m_dirac = eosbase.solve_dirac_mass(n_n, n_p, m_ref, c_s)
    if m_dirac is None:
        raise TransferabilityError(f"Dirac gap solve failed at n_b={n_b}, delta={delta}")
    eps = eosbase.fermion_energy_density(n_n, m_dirac) + eosbase.fermion_energy_density(n_p, m_dirac)
    eps += 0.5 * (m_ref - m_dirac) ** 2 / c_s
    eps += 0.5 * c_rho * (n_n - n_p) ** 2
    return eps / n_b


def symmetry_energy_summary(c_rho: float) -> dict[str, Any]:
    """Symmetry energy of the saturated-vector model at ``c_rho``.

    ``J_diff`` is the PNM-SNM energy difference at saturation (the standard
    J definition); ``J_curv`` is the parabolic-curvature estimate from the
    delta = 0.5 point; their gap measures non-parabolicity.  ``L`` uses a
    central difference of the difference definition.
    """

    def s_diff(n_b: float) -> float:
        return _energy_per_baryon_no_leptons(n_b, 1.0, c_rho) - _energy_per_baryon_no_leptons(n_b, 0.0, c_rho)

    n0 = N0_FM3
    h = SYMMETRY_L_STEP_FM3
    j_diff = s_diff(n0)
    j_curv = 4.0 * (
        _energy_per_baryon_no_leptons(n0, 0.5, c_rho) - _energy_per_baryon_no_leptons(n0, 0.0, c_rho)
    )
    slope = (s_diff(n0 + h) - s_diff(n0 - h)) / (2.0 * h)
    curve = []
    steps = int(round((SYMMETRY_GRID_MAX_FM3 - SYMMETRY_GRID_MIN_FM3) / SYMMETRY_GRID_STEP_FM3))
    for index in range(steps + 1):
        n_b = SYMMETRY_GRID_MIN_FM3 + index * SYMMETRY_GRID_STEP_FM3
        curve.append({"n_b_fm3": _num(n_b, 6), "S_diff_MeV": _num(s_diff(n_b))})
    return {
        "c_rho_MeV_fm3": _num(c_rho),
        "j_contact_MeV": _num(0.5 * c_rho * n0),
        "J_diff_MeV": _num(j_diff),
        "J_curv_MeV": _num(j_curv),
        "nonparabolicity_MeV": _num(j_curv - j_diff),
        "L_MeV": _num(3.0 * n0 * slope),
        "S_diff_curve": curve,
    }


@contextmanager
def _patched_crho(c_rho: float):
    key = "Crho"
    original = soft.BEST_BASELINE[key]
    if abs(float(original) - float(c_rho)) < 1.0e-15:
        yield original
        return
    soft.BEST_BASELINE[key] = float(c_rho)
    try:
        yield float(c_rho)
    finally:
        soft.BEST_BASELINE[key] = original


def _constraint_flags(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the audit's screening constraint evaluation."""

    lambda_values = [
        summary.get("Lambda_tilde_symmetric_1.36"),
        summary.get("Lambda_tilde_asymmetric_1.46_1.27"),
    ]
    lambda_values = [float(value) for value in lambda_values if value is not None and nsa._finite(value)]
    constraint_summary = dict(summary)
    constraint_summary["Lambda_tilde"] = max(lambda_values) if lambda_values else None
    flags = {name: bool(nsa._constraint_passes(constraint_summary, name)) for name in nsa.OBSERVABLE_CONSTRAINTS}
    return {
        "flags": flags,
        "Lambda_tilde_symmetric_1.36": _num(lambda_values[0]) if len(lambda_values) > 0 else None,
        "Lambda_tilde_asymmetric_1.46_1.27": _num(lambda_values[1]) if len(lambda_values) > 1 else None,
    }


def _ns_chain(c_rho: float) -> dict[str, Any]:
    """Canonical hybrid pipeline (saturated-vector baseline + CSS transition)."""

    with _patched_crho(c_rho):
        baseline = soft.build_baseline_arrays(nn=220)
    if baseline is None:
        raise TransferabilityError(f"baseline EOS could not be built at C_rho={c_rho}")
    eos = nsa._make_hybrid_eos(baseline, 2.0, 0.0, 1.0 / 3.0)
    if eos is None:
        raise TransferabilityError(f"hybrid EOS could not be built at C_rho={c_rho}")
    summary = nsa.evaluate_sequence(eos, central_pressure_points=72)
    observables = {key: summary.get(key) for key in ("M_max", "R_1.4", "Lambda_1.4")}
    for key, value in observables.items():
        if value is None or not nsa._finite(value):
            raise TransferabilityError(f"C_rho={c_rho}: observable {key} is not finite")
    return {
        "c_rho_MeV_fm3": _num(c_rho),
        "c_omega0_calibration": _num(baseline["c_omega0"]),
        "y_p_at_n0": _num(baseline["state_n0"]["y_p"]),
        "M_max_msun": _num(observables["M_max"]),
        "R_1.4_km": _num(observables["R_1.4"]),
        "Lambda_1.4": _num(observables["Lambda_1.4"]),
        "branch_selection_status": summary.get("branch_selection_status"),
        "constraints": _constraint_flags(summary),
    }


def verdict_part_b(m_max_msun: float) -> str:
    if m_max_msun >= MMAX_TRANSFERABLE_MIN_MSUN:
        return "TRANSFERABLE_TO_NS_SECTOR"
    if m_max_msun >= MMAX_TENSION_MIN_MSUN:
        return "PARTIAL_TENSION_WITH_J0740"
    return "NOT_TRANSFERABLE_MAX_MASS_LOST"


def _canonical_chain_control() -> dict[str, Any]:
    """Reproduce the flagship one-component chain as a pipeline control."""

    canonical_eos = tidal.EOS(p_match=1.5, Gamma=1.35)
    summary = nsa.evaluate_sequence(canonical_eos, central_pressure_points=120, include_sequence=False)
    values = {
        "M_max": float(summary["M_max"]),
        "R_1.4": float(summary["R_1.4"]),
        "Lambda_1.4": float(summary["Lambda_1.4"]),
    }
    checks = {
        "M_max": abs(values["M_max"] - CANONICAL_CHAIN["M_max_msun"]) <= BASELINE_REPRO_REL_TOL * abs(CANONICAL_CHAIN["M_max_msun"]),
        "R_1.4": abs(values["R_1.4"] - CANONICAL_CHAIN["R_1.4_km"]) <= BASELINE_REPRO_REL_TOL * abs(CANONICAL_CHAIN["R_1.4_km"]),
        "Lambda_1.4": abs(values["Lambda_1.4"] - CANONICAL_CHAIN["Lambda_1.4"]) <= BASELINE_REPRO_REL_TOL * abs(CANONICAL_CHAIN["Lambda_1.4"]),
    }
    if not all(checks.values()):
        raise TransferabilityError(f"canonical chain control failed to reproduce the frozen values: {values}")
    return {
        "note": (
            "one-component isoscalar EOS (tidal.EOS p_match=1.5 Gamma=1.35): no proton/neutron "
            "split, so the isovector term cannot enter this chain by a parameter change"
        ),
        "M_max_msun": _num(values["M_max"]),
        "R_1.4_km": _num(values["R_1.4"]),
        "Lambda_1.4": _num(values["Lambda_1.4"]),
        "reproduces_frozen_canonical": True,
    }


def _part_b() -> dict[str, Any]:
    canonical_control = _canonical_chain_control()
    baseline_symmetry = symmetry_energy_summary(NS_RHO_BASELINE)
    if not (J_BASELINE_SANITY_BAND_MEV[0] <= baseline_symmetry["J_diff_MeV"] <= J_BASELINE_SANITY_BAND_MEV[1]):
        raise TransferabilityError(
            f"baseline J={baseline_symmetry['J_diff_MeV']} outside the pre-registered sanity band"
        )
    if not (L_BASELINE_SANITY_BAND_MEV[0] <= baseline_symmetry["L_MeV"] <= L_BASELINE_SANITY_BAND_MEV[1]):
        raise TransferabilityError(
            f"baseline L={baseline_symmetry['L_MeV']} outside the pre-registered sanity band"
        )

    j_contact_baseline = 0.5 * NS_RHO_BASELINE * N0_FM3
    c_rho_transfer = 2.0 * J_BAR_MEV / N0_FM3
    cross_branch = {
        "j_contact_NS_baseline_MeV": _num(j_contact_baseline),
        "j_identified_finite_branch_MeV": _num(J_BAR_MEV),
        "cross_branch_ratio": _num(j_contact_baseline / J_BAR_MEV),
        "c_rho_transfer_MeV_fm3": _num(c_rho_transfer),
        "transfer_identity": "C_rho -> 2 j / n0 matches the finite-branch contact (8 j / n0)/4; only the interaction piece is transferred",
    }

    baseline_chain = _ns_chain(NS_RHO_BASELINE)
    retuned_symmetry = symmetry_energy_summary(c_rho_transfer)
    retuned_chain = _ns_chain(c_rho_transfer)
    verdict = verdict_part_b(float(retuned_chain["M_max_msun"]))

    return {
        "question": (
            "is the finite-branch isovector interaction piece consistent with the two-component "
            "NS-sector family, and does transferring it keep the canonical hybrid chain viable?"
        ),
        "canonical_chain_control": canonical_control,
        "baseline_symmetry_energy": baseline_symmetry,
        "cross_branch_transfer": cross_branch,
        "baseline_chain": baseline_chain,
        "retuned_symmetry_energy": retuned_symmetry,
        "retuned_chain": retuned_chain,
        "pre_registered_bands": {
            "M_max_transferable_min_msun": MMAX_TRANSFERABLE_MIN_MSUN,
            "M_max_tension_min_msun": MMAX_TENSION_MIN_MSUN,
            "J_baseline_sanity_band_MeV": list(J_BASELINE_SANITY_BAND_MEV),
            "L_baseline_sanity_band_MeV": list(L_BASELINE_SANITY_BAND_MEV),
            "constraints_source": "J0740/GW170817/NICER screening bands from nvg_ns_predictive_audit (in-sample selection inputs)",
        },
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def calculate() -> dict[str, Any]:
    retained = _load_sn132_artifact()
    canonical = _load_nsa_canonical()
    part_a = _part_a(retained)
    part_b = _part_b()
    if part_a["verdict"] not in ("REANCHORED_UNIVERSAL_J_SURVIVES", "REANCHORED_UNIVERSAL_J_BREAKS_BANDS"):
        raise TransferabilityError("part A verdict is not a registered value")
    if part_b["verdict"] not in (
        "TRANSFERABLE_TO_NS_SECTOR",
        "PARTIAL_TENSION_WITH_J0740",
        "NOT_TRANSFERABLE_MAX_MASS_LOST",
    ):
        raise TransferabilityError("part B verdict is not a registered value")
    return _jsonable(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "evidence_weight": EVIDENCE_WEIGHT,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "retained_inputs": {
                "sn132_discriminating_set": {
                    "path": retained["path"],
                    "j_bar_MeV": _num(retained["j_bar_MeV"]),
                    "J_design_universal_MeV": _num(retained["J_design_universal_MeV"]),
                    "t_contact_MeV": _num(retained["t_contact_MeV"]),
                },
                "ns_canonical_audit": canonical,
            },
            "pre_registered_bands": {
                "part_a": {
                    "anchor_abs_tol_per_A_MeV": ANCHOR_RECAL_ABS_TOL_BA_MEV,
                    "ds_over_s_abs_band": list(DS_OVER_S_ABS_BAND),
                    "total_abs_residual_band_MeV": list(PART_A_TOTAL_ABS_RESIDUAL_BAND_MEV),
                    "per_nucleus_abs_band_MeV": list(PART_A_PER_NUCLEUS_ABS_BAND_MEV),
                },
                "part_b": {
                    "M_max_transferable_min_msun": MMAX_TRANSFERABLE_MIN_MSUN,
                    "M_max_tension_min_msun": MMAX_TENSION_MIN_MSUN,
                    "baseline_repro_rel_tol": BASELINE_REPRO_REL_TOL,
                },
            },
            "part_a_reanchored_finite_branch": part_a,
            "part_b_ns_sector_transfer": part_b,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path for the strict-JSON result artifact",
    )
    args = parser.parse_args(argv)
    payload = calculate()
    text = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print("Isovector transferability probe")
    print(f"schema={payload['schema']}")
    part_a = payload["part_a_reanchored_finite_branch"]
    part_b = payload["part_b_ns_sector_transfer"]
    print(f"part_a: s*_rho={part_a['recalibration']['s_rho_reanchored']} (ds/s={part_a['recalibration']['ds_over_s']})")
    print(f"part_a: heldout_total_residual={part_a['heldout_total_residual_MeV']} MeV -> {part_a['verdict']}")
    print(f"part_b: baseline J={part_b['baseline_symmetry_energy']['J_diff_MeV']} MeV, L={part_b['baseline_symmetry_energy']['L_MeV']} MeV")
    print(
        "part_b: cross-branch ratio j_NS/j_finite="
        f"{part_b['cross_branch_transfer']['cross_branch_ratio']}"
    )
    print(
        f"part_b: baseline M_max={part_b['baseline_chain']['M_max_msun']} -> "
        f"retuned M_max={part_b['retuned_chain']['M_max_msun']} Msun -> {part_b['verdict']}"
    )
    if args.output is not None:
        print(f"artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
