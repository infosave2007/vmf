#!/usr/bin/env python3
"""PREX-II/CREX external-consistency probe (conditional, zero evidence).

First external-data comparison of the finite-nucleus program: the point
neutron skins of Pb208 and Ca48 predicted by the two declared branches are
compared against the published PREX-II and CREX bands.

Branches compared (both already computed and retained elsewhere; re-solved
live here through the same terminal protocol):

* ``no_rho``       - the production W8.93 branch (s* = 0.2268, Ca40 anchor);
* ``identified_j`` - the rho contact at the identified universal constant
  (J_design = 23.7801345 MeV, j = 11.136 MeV) with the re-anchored scale
  s*_rho = 0.2267063 from the transferability probe.

Pre-registered question (fixed here before any solve in this probe):

    Is the point neutron skin R_n - R_p of Pb208 (PREX-II) or Ca48 (CREX)
    predicted by EITHER branch consistent with the published band at the
    pre-registered two-sigma level?

Pre-registered published bands (frozen constants with citations):

    PREX-II  Pb208: R_n - R_p = 0.283 +- 0.071 fm
             (Adhikari et al., PRL 126, 172502 (2021), arXiv:2102.10767;
              combined PREX-I+II extraction)
    CREX     Ca48:  R_n - R_p = 0.121 +- 0.026(exp) +- 0.024(model) fm
             (Adhikari et al., PRL 129, 042501 (2022), arXiv:2205.11593);
             exp and model errors combined in quadrature (conservative).

Pre-registered verdict bands:

    SKINS_CONSISTENT_SOME_BRANCH - at least one (branch, experiment) pair
                                    is consistent at two sigma.
    SKINS_EXCLUDED_BOTH_BRANCHES - every pair is excluded at two sigma.

Secondary descriptive quantities (no verdict weight, declared limitations):
the plane-wave weak form factor F_W(q) and parity-violating asymmetry A_PV
at the measured kinematics (PREX-II Q^2 = 0.00616 GeV^2; CREX
q = 0.8733 fm^-1) compared with the published central values, and the naive
linear contact strength that would be required to reach each skin.

The experimental values are public knowledge, so this is NOT a blind
prediction; it is out-of-sample only in the sense that neither j nor s* was
ever tuned to a skin.  No production file is modified; the rho branch and
the re-anchored scale are entered only through the same context managers as
the earlier probes, and every solve is a full multi-seed terminal case.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_finite_monopole as static
import nvg_finite_static_bridge as bridge
import nvg_isovector_formfactor_probe as ivp
import nvg_sn132_discriminating_set_probe as sn

SCHEMA = "nvg_prex_crex_external_consistency_probe.v1"
STATUS = "COMPUTED_PREX_CREX_EXTERNAL_CONSISTENCY_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

TRANSFER_RESULT_REL = Path("Lunacy/runs/isovector-transferability-2026-09-17/isovector_transferability_result.json")
FORMFACTOR_RESULT_REL = Path("Lunacy/runs/isovector-formfactor-2026-09-16/probe_result.json")
CA48_RESULT_REL = Path("Lunacy/runs/ca48-quantum-surface-2026-09-16/ca48_quantum_surface_result.json")

# ---------------------------------------------------------------------------
# Published experimental inputs (frozen; each cited).
# ---------------------------------------------------------------------------

PREX2 = {
    "nucleus": "Pb208",
    "skin_fm": 0.283,
    "skin_sigma_fm": 0.071,
    "F_W": 0.368,
    "F_W_sigma": 0.013,
    "Q2_GeV2": 0.00616,
    "A_PV_ppb": 550.0,
    "A_PV_stat_ppb": 16.0,
    "A_PV_syst_ppb": 8.0,
    "citation": (
        "PREX-II Collaboration (Adhikari et al.), Phys. Rev. Lett. 126, "
        "172502 (2021), arXiv:2102.10767: A_PV = 550 +- 16(stat) +- 8(syst) ppb; "
        "F_W(Q^2 = 0.00616 GeV^2) = 0.368 +- 0.013; combined PREX-I+II "
        "R_n - R_p = 0.283 +- 0.071 fm"
    ),
}
CREX = {
    "nucleus": "Ca48",
    "skin_fm": 0.121,
    "skin_sigma_exp_fm": 0.026,
    "skin_sigma_model_fm": 0.024,
    "F_W": 0.1304,
    "F_W_stat_sigma": 0.0052,
    "F_W_syst_sigma": 0.0020,
    "q_fm": 0.8733,
    "A_PV_ppb": 2668.0,
    "A_PV_stat_ppb": 106.0,
    "A_PV_syst_ppb": 40.0,
    "citation": (
        "CREX Collaboration (Adhikari et al.), Phys. Rev. Lett. 129, "
        "042501 (2022), arXiv:2205.11593: A_PV = 2668 +- 106(stat) +- 40(syst) "
        "ppb; F_W(q = 0.8733 fm^-1) = 0.1304 +- 0.0052(stat) +- 0.0020(syst); "
        "R_n - R_p = 0.121 +- 0.026(exp) +- 0.024(model) fm"
    ),
}
# Pre-registered: skins are tested at two sigma; exp+model errors in quadrature.
SKIN_SIGMA_LEVEL = 2.0

# ---------------------------------------------------------------------------
# Identified-j branch point (retained transferability artifact; live-verified).
# ---------------------------------------------------------------------------

J_BAR_MEV = 11.13601607117819
J_DESIGN_IDENTIFIED = 23.78013451414814
S_RHO_REANCHORED = 0.2267063450812736

# Fail-closed reproduction targets (retained artifacts, full precision).
IDENTIFIED_BRANCH_RADII = {
    "Ca40": {
        "rms_point_proton_radius_fm": 2.953294270498702,
        "rms_neutron_radius_fm": 2.9371019221719705,
    },
    "Pb208": {
        "rms_point_proton_radius_fm": 5.317782299151776,
        "rms_neutron_radius_fm": 5.292192489681611,
    },
    "Ca48": {
        "rms_point_proton_radius_fm": 3.136262999162403,
        "rms_neutron_radius_fm": 3.158890245884534,
    },
}
NO_RHO_CONTROL_RADII = {
    "Pb208": {  # isovector form-factor probe (2026-09-16) baseline
        "rms_point_proton_radius_fm": 5.319947436693366,
        "rms_neutron_radius_fm": 5.238221091766043,
    },
    "Ca48": {  # Ca48 quantum-surface probe (2026-09-16) no-rho case
        "rms_point_proton_radius_fm": 3.138338956744131,
        "rms_neutron_radius_fm": 3.1376216393230663,
    },
}
CONTROL_REL_TOL = 1.0e-9

# Kinematics (derived from the quoted central values).
Q_PREX2_FM = math.sqrt(PREX2["Q2_GeV2"]) / ivp.HBARC_GEV_FM
Q_CREX_FM = float(CREX["q_fm"])
FF_Q_GRID = (Q_PREX2_FM, Q_CREX_FM)

BRANCHES = ("no_rho", "identified_j")
SKIN_NUCLEI = ("Pb208", "Ca48")


class PrexCrexError(RuntimeError):
    """Fail-closed error for rejected solves, controls or artifacts."""


def _num(value: Any, digits: int = 17) -> float:
    return float(format(float(value), f".{digits}g"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, np.bool_)) or value is None:
        return bool(value) if isinstance(value, np.bool_) else value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# Retained-artifact verification (fail closed).
# ---------------------------------------------------------------------------


def _load_transferability_control() -> dict[str, Any]:
    path = ROOT / TRANSFER_RESULT_REL
    if not path.is_file():
        raise PrexCrexError(f"retained transferability artifact not present: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "nvg_isovector_transferability_probe.v1":
        raise PrexCrexError("retained transferability artifact schema mismatch")
    part_a = payload.get("part_a_reanchored_finite_branch")
    if not isinstance(part_a, Mapping):
        raise PrexCrexError("retained transferability artifact has no part A")
    if abs(float(part_a["j_bar_MeV"]) - J_BAR_MEV) > 1.0e-12:
        raise PrexCrexError("retained j_bar does not match the identified constant")
    recal = part_a.get("recalibration")
    if not isinstance(recal, Mapping):
        raise PrexCrexError("retained transferability artifact has no recalibration block")
    if abs(float(recal["s_rho_reanchored"]) - S_RHO_REANCHORED) > 1.0e-12:
        raise PrexCrexError("retained s_rho_reanchored does not match the probe constant")
    cases = part_a.get("cases")
    if not isinstance(cases, Mapping):
        raise PrexCrexError("retained transferability artifact has no case table")
    for nucleus, target in IDENTIFIED_BRANCH_RADII.items():
        row = cases.get(nucleus)
        if not isinstance(row, Mapping):
            raise PrexCrexError(f"retained transferability artifact lacks {nucleus}")
        for key, want in target.items():
            if abs(float(row[key]) - float(want)) > 1.0e-12:
                raise PrexCrexError(f"retained {nucleus} {key} drifted from the probe constant")
    return {
        "path": str(path),
        "j_bar_MeV": _num(part_a["j_bar_MeV"]),
        "j_design_MeV": _num(part_a["j_design_MeV"]),
        "s_rho_reanchored": _num(recal["s_rho_reanchored"]),
    }


def _load_formfactor_control() -> dict[str, Any]:
    path = ROOT / FORMFACTOR_RESULT_REL
    if not path.is_file():
        raise PrexCrexError(f"retained form-factor artifact not present: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "nvg_isovector_formfactor_probe.v1":
        raise PrexCrexError("retained form-factor artifact schema mismatch")
    baseline = payload.get("baseline")
    if not isinstance(baseline, Mapping) or not isinstance(baseline.get("Pb208"), Mapping):
        raise PrexCrexError("retained form-factor artifact has no Pb208 baseline")
    row = baseline["Pb208"]
    for key, want in NO_RHO_CONTROL_RADII["Pb208"].items():
        if abs(float(row[key]) - float(want)) > 1.0e-12:
            raise PrexCrexError(f"retained form-factor Pb208 {key} drifted from the probe constant")
    return {"path": str(path), "Pb208_point_skin_fm": _num(row["point_neutron_skin_fm"])}


def _load_ca48_control() -> dict[str, Any]:
    path = ROOT / CA48_RESULT_REL
    if not path.is_file():
        raise PrexCrexError(f"retained Ca48 artifact not present: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    part_a = payload.get("part_a_ca48_heldout")
    if not isinstance(part_a, Mapping):
        raise PrexCrexError("retained Ca48 artifact has no part A")
    cases = part_a.get("cases")
    if not isinstance(cases, Mapping) or not isinstance(cases.get("no_rho"), Mapping):
        raise PrexCrexError("retained Ca48 artifact has no no_rho case")
    row = cases["no_rho"]
    for key, want in NO_RHO_CONTROL_RADII["Ca48"].items():
        if abs(float(row[key]) - float(want)) > 1.0e-12:
            raise PrexCrexError(f"retained Ca48 no_rho {key} drifted from the probe constant")
    return {"path": str(path), "Ca48_point_skin_fm": _num(row["point_neutron_skin_fm"])}


# ---------------------------------------------------------------------------
# Live branch solves (full multi-seed terminal protocol).
# ---------------------------------------------------------------------------


def _solve_branch_point(nucleus: str, branch: str) -> dict[str, Any]:
    """One terminal solve of ``nucleus`` on the declared branch."""

    if branch == "no_rho":
        scale, rho_choice, j_design = ivp.ROOT_SCALE, ivp.BASELINE_RHO, None
    elif branch == "identified_j":
        scale, rho_choice, j_design = S_RHO_REANCHORED, ivp.CONTACT_RHO, J_DESIGN_IDENTIFIED
    else:
        raise PrexCrexError(f"unknown branch: {branch}")
    outer = sn._heldout_context() if nucleus in sn.HELDOUT else nullcontext()
    with outer:
        with ivp._rho_context(rho_choice), ivp._j_design_context(j_design):
            case = bridge._terminal_case(ivp.FAMILY, float(scale), nucleus, multi_seed=True)
        if case.get("terminal_protocol_acceptance") is not True:
            raise PrexCrexError(
                f"terminal protocol rejected {nucleus} on branch {branch}"
            )
        metrics = ivp._terminal_metrics(nucleus, case)
    if not isinstance(metrics, Mapping):
        raise PrexCrexError(f"{nucleus}: no terminal metrics on branch {branch}")
    return dict(metrics)


def _verify_radii(nucleus: str, metrics: Mapping[str, Any], target: Mapping[str, float]) -> None:
    for key, want in target.items():
        got = float(metrics[key])
        if abs(got - float(want)) > CONTROL_REL_TOL * max(1.0, abs(float(want))):
            raise PrexCrexError(
                f"{nucleus} live {key} {got!r} does not reproduce the retained {float(want)!r}"
            )


# ---------------------------------------------------------------------------
# Operator stack (plane-wave Born; declared approximation of ivp).
# ---------------------------------------------------------------------------


def _ff_lookup(table: Mapping[str, float], q_fm: float) -> float:
    for key, value in table.items():
        if abs(float(key) - q_fm) <= 1.0e-9:
            return float(value)
    raise PrexCrexError(f"form-factor table lacks q = {q_fm}")


def _apv_ppb(q_fm: float, f_ch: float, f_w: float, z: int, n_n: int) -> float:
    """Plane-wave A_PV in ppb from the normalized F_W and F_ch (ivp convention)."""

    q_gev = q_fm * ivp.HBARC_GEV_FM
    prefactor = ivp.GF_GEV_MINUS2 * q_gev * q_gev / (
        4.0 * math.pi * static.ALPHA * math.sqrt(2.0)
    )
    weak_total = ivp.Q_WEAK_P * z + ivp.Q_WEAK_N * n_n
    return -prefactor * weak_total * f_w / (z * f_ch) * 1.0e9


def _branch_row(nucleus: str, branch: str, control: Mapping[str, float] | None) -> dict[str, Any]:
    metrics = _solve_branch_point(nucleus, branch)
    if control is not None:
        _verify_radii(nucleus, metrics, control)
    profile = metrics.get("_profile")
    if not isinstance(profile, Mapping):
        raise PrexCrexError(f"{nucleus} ({branch}): solution lacks profiles")
    z, n_n = int(metrics["Z"]), int(metrics["N"])
    ff = ivp.charge_weak_form_factors(
        profile["r_fm"], profile["nn_fm3"], profile["np_fm3"], n_n, z, FF_Q_GRID
    )
    f_w_prex = _ff_lookup(ff["F_W_of_q"], Q_PREX2_FM)
    f_w_crex = _ff_lookup(ff["F_W_of_q"], Q_CREX_FM)
    f_ch_prex = _ff_lookup(ff["F_ch_of_q"], Q_PREX2_FM)
    f_ch_crex = _ff_lookup(ff["F_ch_of_q"], Q_CREX_FM)
    return {
        "nucleus": nucleus,
        "branch": branch,
        "A": int(metrics["A"]),
        "Z": z,
        "N": n_n,
        "binding_per_A_MeV": _num(metrics["binding_per_A_MeV"]),
        "rms_point_proton_radius_fm": _num(metrics["rms_point_proton_radius_fm"]),
        "rms_neutron_radius_fm": _num(metrics["rms_neutron_radius_fm"]),
        "point_neutron_skin_fm": _num(metrics["point_neutron_skin_fm"]),
        "conserved_NZ_relative_error": _num(metrics["conserved_NZ_relative_error"]),
        "F_W_at_PREX2_q": _num(f_w_prex),
        "F_W_at_CREX_q": _num(f_w_crex),
        "F_ch_at_PREX2_q": _num(f_ch_prex),
        "F_ch_at_CREX_q": _num(f_ch_crex),
        "A_PV_pw_ppb_at_PREX2_q": _num(_apv_ppb(Q_PREX2_FM, f_ch_prex, f_w_prex, z, n_n)),
        "A_PV_pw_ppb_at_CREX_q": _num(_apv_ppb(Q_CREX_FM, f_ch_crex, f_w_crex, z, n_n)),
        "rms_charge_from_slope_fm": ff["rms_charge_from_slope_fm"],
        "rms_weak_from_slope_fm": ff["rms_weak_from_slope_fm"],
        "operator_convention": ff["A_PV_pw_convention"],
    }


# ---------------------------------------------------------------------------
# Pre-registered comparison logic.
# ---------------------------------------------------------------------------


def skin_consistency(skin_fm: float, center_fm: float, sigma_fm: float) -> dict[str, Any]:
    delta = float(skin_fm) - float(center_fm)
    n_sigma = delta / float(sigma_fm)
    consistent = abs(delta) <= SKIN_SIGMA_LEVEL * float(sigma_fm)
    return {
        "skin_fm": _num(skin_fm),
        "center_fm": _num(center_fm),
        "sigma_fm": _num(sigma_fm),
        "delta_fm": _num(delta),
        "n_sigma": _num(n_sigma),
        "consistent_at_2sigma": bool(consistent),
        "band": "CONSISTENT_2SIGMA" if consistent else "EXCLUDED_2SIGMA",
    }


def verdict(pairs: Mapping[str, Mapping[str, Any]]) -> str:
    if any(pair["consistent_at_2sigma"] for pair in pairs.values()):
        return "SKINS_CONSISTENT_SOME_BRANCH"
    return "SKINS_EXCLUDED_BOTH_BRANCHES"


def _apv_published_sigma(stat: float, syst: float) -> float:
    return math.sqrt(stat * stat + syst * syst)


def _descriptive_weak_comparison(label: str, published: Mapping[str, Any],
                                 row: Mapping[str, Any], fw_key: str,
                                 apv_key: str, q_fm: float) -> dict[str, Any]:
    fw_sigma = (
        math.sqrt(published["F_W_stat_sigma"] ** 2 + published["F_W_syst_sigma"] ** 2)
        if "F_W_stat_sigma" in published
        else float(published["F_W_sigma"])
    )
    fw = float(row[fw_key])
    apv = float(row[apv_key])
    apv_sigma = _apv_published_sigma(published["A_PV_stat_ppb"], published["A_PV_syst_ppb"])
    return {
        "label": label,
        "q_fm": _num(q_fm),
        "F_W_published": _num(published["F_W"]),
        "F_W_sigma": _num(fw_sigma),
        "F_W_theory_pw": _num(fw),
        "F_W_delta_n_sigma": _num((fw - float(published["F_W"])) / fw_sigma),
        "A_PV_published_ppb": _num(published["A_PV_ppb"]),
        "A_PV_sigma_ppb": _num(apv_sigma),
        "A_PV_theory_pw_ppb": _num(apv),
        "A_PV_delta_n_sigma": _num((apv - float(published["A_PV_ppb"])) / apv_sigma),
        "weight": "descriptive only; plane-wave Born, no Coulomb distortion, "
                  "acceptance or radiative corrections",
    }


def _naive_j_for_skin(skin_no_rho: float, skin_identified: float,
                      center_fm: float) -> dict[str, Any]:
    response = (skin_identified - skin_no_rho) / J_BAR_MEV
    if response <= 0.0:
        return {
            "response_fm_per_MeV": _num(response),
            "j_naive_MeV": None,
            "note": "contact response does not increase the skin; no linear extrapolation exists",
        }
    j_naive = J_BAR_MEV + (center_fm - skin_identified) / response
    return {
        "response_fm_per_MeV": _num(response),
        "j_naive_MeV": _num(j_naive),
        "note": "naive linear extrapolation far beyond the identified j; "
                "the response is not linear over such a range and this is "
                "descriptive only",
    }


def calculate() -> dict[str, Any]:
    transfer_control = _load_transferability_control()
    formfactor_control = _load_formfactor_control()
    ca48_control = _load_ca48_control()

    rows: dict[str, dict[str, Any]] = {}
    # no_rho branch: Pb208 and Ca48, fail-closed against the retained artifacts.
    rows[("no_rho", "Pb208")] = _branch_row(
        "Pb208", "no_rho", NO_RHO_CONTROL_RADII["Pb208"]
    )
    rows[("no_rho", "Ca48")] = _branch_row(
        "Ca48", "no_rho", NO_RHO_CONTROL_RADII["Ca48"]
    )
    # identified-j branch: Ca40 (anchor control) plus the two skin nuclei.
    rows[("identified_j", "Ca40")] = _branch_row(
        "Ca40", "identified_j", IDENTIFIED_BRANCH_RADII["Ca40"]
    )
    rows[("identified_j", "Pb208")] = _branch_row(
        "Pb208", "identified_j", IDENTIFIED_BRANCH_RADII["Pb208"]
    )
    rows[("identified_j", "Ca48")] = _branch_row(
        "Ca48", "identified_j", IDENTIFIED_BRANCH_RADII["Ca48"]
    )

    pairs: dict[str, dict[str, Any]] = {}
    for experiment, published in (("PREX2_Pb208", PREX2), ("CREX_Ca48", CREX)):
        sigma = (
            math.sqrt(published["skin_sigma_exp_fm"] ** 2 + published["skin_sigma_model_fm"] ** 2)
            if "skin_sigma_exp_fm" in published
            else float(published["skin_sigma_fm"])
        )
        per_branch = {}
        for branch in BRANCHES:
            row = rows[(branch, published["nucleus"])]
            per_branch[branch] = skin_consistency(row["point_neutron_skin_fm"],
                                                  published["skin_fm"], sigma)
        pairs[experiment] = {
            "center_fm": _num(published["skin_fm"]),
            "sigma_fm": _num(sigma),
            "two_sigma_band_fm": (
                _num(published["skin_fm"] - SKIN_SIGMA_LEVEL * sigma),
                _num(published["skin_fm"] + SKIN_SIGMA_LEVEL * sigma),
            ),
            "no_rho": per_branch["no_rho"],
            "identified_j": per_branch["identified_j"],
        }

    weak_comparisons = [
        _descriptive_weak_comparison(
            "PREX2_Pb208", PREX2, rows[("identified_j", "Pb208")],
            "F_W_at_PREX2_q", "A_PV_pw_ppb_at_PREX2_q", Q_PREX2_FM,
        ),
        _descriptive_weak_comparison(
            "PREX2_Pb208_no_rho", PREX2, rows[("no_rho", "Pb208")],
            "F_W_at_PREX2_q", "A_PV_pw_ppb_at_PREX2_q", Q_PREX2_FM,
        ),
        _descriptive_weak_comparison(
            "CREX_Ca48", CREX, rows[("identified_j", "Ca48")],
            "F_W_at_CREX_q", "A_PV_pw_ppb_at_CREX_q", Q_CREX_FM,
        ),
        _descriptive_weak_comparison(
            "CREX_Ca48_no_rho", CREX, rows[("no_rho", "Ca48")],
            "F_W_at_CREX_q", "A_PV_pw_ppb_at_CREX_q", Q_CREX_FM,
        ),
    ]

    naive_j = {
        "PREX2_Pb208": _naive_j_for_skin(
            rows[("no_rho", "Pb208")]["point_neutron_skin_fm"],
            rows[("identified_j", "Pb208")]["point_neutron_skin_fm"],
            PREX2["skin_fm"],
        ),
        "CREX_Ca48": _naive_j_for_skin(
            rows[("no_rho", "Ca48")]["point_neutron_skin_fm"],
            rows[("identified_j", "Ca48")]["point_neutron_skin_fm"],
            CREX["skin_fm"],
        ),
    }

    flat_pairs = {
        f"{experiment}:{branch}": pairs[experiment][branch]
        for experiment in pairs
        for branch in BRANCHES
    }
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "Is the point neutron skin of Pb208 (PREX-II) or Ca48 (CREX) "
            "predicted by either declared branch (no_rho production; "
            "identified-j contact with re-anchored scale) consistent with "
            "the published band at the pre-registered two-sigma level?"
        ),
        "pre_registered": {
            "skin_sigma_level": SKIN_SIGMA_LEVEL,
            "verdict_names": ["SKINS_CONSISTENT_SOME_BRANCH", "SKINS_EXCLUDED_BOTH_BRANCHES"],
            "crex_error_convention": "exp and model sigmas combined in quadrature",
        },
        "published_inputs": {
            "PREX2": {key: (value if isinstance(value, str) else _num(value)) for key, value in PREX2.items()},
            "CREX": {key: (value if isinstance(value, str) else _num(value)) for key, value in CREX.items()},
        },
        "kinematics": {
            "q_PREX2_fm": _num(Q_PREX2_FM),
            "q_CREX_fm": _num(Q_CREX_FM),
            "derivation": "q_PREX2 = sqrt(0.00616 GeV^2)/hbar c; q_CREX quoted directly",
        },
        "retained_inputs": {
            "isovector_transferability": transfer_control,
            "isovector_formfactor_baseline_Pb208": formfactor_control,
            "ca48_quantum_surface_no_rho": ca48_control,
        },
        "branch_rows": {
            f"{branch}:{nucleus}": row
            for (branch, nucleus), row in sorted(rows.items())
        },
        "skin_comparison": pairs,
        "weak_form_factor_and_apv_descriptive": weak_comparisons,
        "naive_j_for_skins_descriptive": naive_j,
        "verdict": verdict(flat_pairs),
        "limits": [
            "published experimental values are public knowledge: this is an "
            "external-consistency audit, not a blind prediction; neither j "
            "nor s* was tuned to any skin",
            "the three screening constraints of the NS sector are unrelated "
            "inputs here; no selection used skin data",
            "plane-wave Born operators only: Coulomb distortion, experimental "
            "acceptance and radiative corrections are not included, so F_W and "
            "A_PV comparisons are descriptive",
            "CREX skin extraction carries a declared model dependence; the "
            "two-sigma band uses the conservative quadrature combination",
            "single accepted family (W8.93); shell structure is absent by "
            "construction, and the isotopic Ca40/Ca48 charge-radius anomaly "
            "already documents the surface-physics boundary of this branch",
        ],
    }
    return _jsonable(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PREX-II/CREX external-consistency probe (zero evidence)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="opt-in path for the strict-JSON result artifact",
    )
    args = parser.parse_args(argv)
    payload = calculate()
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(f"verdict={payload['verdict']}")
    for experiment, block in payload["skin_comparison"].items():
        for branch in BRANCHES:
            pair = block[branch]
            print(
                f"{experiment} {branch}: skin={pair['skin_fm']:+.4f} fm "
                f"(center {block['center_fm']:.3f} +- {block['sigma_fm']:.4f}) "
                f"-> {pair['band']} ({pair['n_sigma']:+.2f} sigma)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
