#!/usr/bin/env python3
"""Sn132 and the complete discriminating set: single-constant isovector verdict.

Closes the discriminating program of ``NVG_RESIDUAL_RESEARCH_DIRECTIONS_RU.md``
on the accepted finite-static baseline (W8.93, ``no_rho``, s*=0.2268, Ca40
anchor).  132Sn (doubly magic, spherical, delta = 32/132 = 0.2424 - the most
neutron-rich accessible doubly-magic held-out nucleus) completes the
pre-registered set {Zr90, Pb208, Ca48, Sn132} and separates the A- and
delta-dependence of the missing isovector term.

Pre-registered protocol (fixed in this file before any 132Sn solve):

1. Reproduce the three accepted baselines (fail closed against R3).
2. Re-solve 48Ca (no_rho and the declared J=20 contact probe) as a cross-run
   stability witness against the retained 2026-09-16 artifact.
3. From the Zr90/Pb208/Ca48 anchored tangents of THIS run, register the
   delta^2-form prediction band for the 132Sn residual BEFORE solving it.
4. Solve 132Sn through the full terminal protocol (no_rho; J=20 chord probe).
5. Verdict on the single-constant hypothesis from the four anchored tangents
   with bands registered here in code: SUPPORTED if every
   ``|j_i/j_mean - 1| <= 0.15``, REFUTED if any ``|j_i/j_mean - 1| > 0.35``,
   PARTIAL otherwise.
6. Universal-j demonstration: with ONE constant ``j_bar`` (the four-tangent
   mean; no per-nucleus tuning), solve all five nuclei self-consistently at
   the corresponding declared contact design and report the residual collapse
   (self-consistent vs the fixed-scale linear prediction).
7. Volume/surface operator patterns on all four held-out profiles.

No coefficient enters the theory: the universal-j solves are declared probes
through the existing contact machinery (context-patched, always restored), and
production files are untouched.  Cited 132Sn inputs only; the result is a
conditional diagnostic with ``evidence_weight=0``.  Importing this module
performs no numerical work and no network access; the CLI prints strict JSON
to stdout and ``--output`` is an explicit opt-in write.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # Both direct-script and package-style imports are supported.
    import nvg_finite_monopole as static
    import nvg_finite_static_bridge as bridge
    import nvg_isovector_formfactor_probe as ivp
    import nvg_ca48_quantum_surface_probe as ca48p
except ImportError:  # pragma: no cover - package import support.
    from . import nvg_finite_monopole as static
    from . import nvg_finite_static_bridge as bridge
    from . import nvg_isovector_formfactor_probe as ivp
    from . import nvg_ca48_quantum_surface_probe as ca48p

HERE = Path(__file__).resolve().parent
SCHEMA = "nvg_sn132_discriminating_set_probe.v1"
STATUS = "COMPUTED_SN132_DISCRIMINATING_SET_VERDICT_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

SN132 = "Sn132"
SN132_NUCLEUS_ROW = {"A": 132, "Z": 50, "N": 82}

# Cited 132Sn inputs (declared evaluated data only; never model outputs).
# B_atom/A from AME2020/ENSDF via the IAEA LiveChart ground-state service
# (extraction 2026-09-16): binding 8354.8726 +- 0.0150 keV per nucleon,
# atomic mass 131.917823898 u.  132Sn is beta-unstable (t1/2 = 39.7 s,
# B- to 132Sb, Q_beta = 3088.7 keV); a static strong-interaction bound state
# remains well defined, and the beta Q value does not enter the binding
# comparison (declared caveat, not a correction).
SN132_B_ATOM_PER_A_MEV = 8.3548726
SN132_B_ATOM_PER_A_SIGMA_MEV = 0.0000150
SN132_B_ATOM_SOURCE = (
    "AME2020/ENSDF via IAEA LiveChart (nds.iaea.org/relnsd/v1/data, extraction "
    "2026-09-16): binding 8354.8726 +- 0.0150 keV per nucleon, atomic mass "
    "131.917823898 u; beta-unstable (t1/2 = 39.7 s, B- to 132Sb, Q_beta = 3088.7 keV)"
)
# Charge radius: Angeli-Marinova 2013 published value (IAEA NDS radii table).
SN132_R_CH_FM = 4.7093
SN132_R_CH_SIGMA_FM = 0.0076
SN132_R_CH_SOURCE = "Angeli & Marinova, ADNDT 99 (2013) 69 (https://nds.iaea.org/radii/, published 2013 values)"
SN132_ROLE = "pre_registered_heldout_descriptive_comparison_only"

# Pre-registered verdict bands (relative deviation of each anchored tangent
# from the four-nucleus mean; fixed before any 132Sn solve).
VERDICT_BAND_SUPPORTED = 0.15
VERDICT_BAND_REFUTED = 0.35

HELDOUT = ("Zr90", "Pb208", "Ca48", "Sn132")
PRODUCTION_NUCLEI = ("Ca40", "Zr90", "Pb208")

CA48_RETAINED_REL = Path("Lunacy/runs/ca48-quantum-surface-2026-09-16/ca48_quantum_surface_result.json")
J20_PROBE_MEV = ivp.J20_PROBE_MEV


class Sn132ProbeError(ValueError):
    """Fail-closed error for protocol failures or non-reproducible baselines."""


def _number(value: Any, digits: int = 17) -> float:
    return ivp._number(value, digits)


def _jsonable(value: Any) -> Any:
    return ivp._jsonable(value)


# Contact j relation, verified exactly linear on this branch:
# j_added = J_design - T_CONTACT with T_CONTACT = 12.64411844296995 MeV
# (the kinetic part of the declared contact at saturation).


def _t_contact_mev() -> float:
    """T = J - j from the two declared probe designs (must agree exactly)."""

    j32 = ivp._contact_j_added(ivp.J32_DESIGN_MEV)
    j20 = ivp._contact_j_added(J20_PROBE_MEV)
    t32 = float(ivp.J32_DESIGN_MEV) - j32
    t20 = float(J20_PROBE_MEV) - j20
    if abs(t32 - t20) > 1.0e-12:
        raise Sn132ProbeError("contact j is not exactly linear in the design J")
    return t32


T_CONTACT_MEV = _t_contact_mev()


def universal_design_j(j_target: float) -> float:
    """Declared contact design J that realizes an added symmetry energy j."""

    return float(j_target) + T_CONTACT_MEV


# ---------------------------------------------------------------------------
# Held-out nucleus-table patch (always restored; never mutates files).
# ---------------------------------------------------------------------------


def _heldout_context():
    """Temporarily add the 48Ca and 132Sn rows to the live nucleus table."""

    rows = {
        ca48p.CA48: dict(ca48p.CA48_NUCLEUS_ROW),
        SN132: dict(SN132_NUCLEUS_ROW),
    }

    class _Patch:
        def __enter__(self):
            for name in rows:
                if name in static.BRIDGE_NUCLEI:
                    raise Sn132ProbeError(f"{name} is already present in the bridge nucleus table")
            for name, row in rows.items():
                static.BRIDGE_NUCLEI[name] = dict(row)
            self._added = tuple(rows)
            return self

        def __exit__(self, *exc):
            for name in self._added:
                static.BRIDGE_NUCLEI.pop(name, None)
            return False

    return _Patch()


def sn132_binding_target() -> dict[str, Any]:
    """Declared 132Sn binding target with the production electron convention."""

    row = {
        "A": SN132_NUCLEUS_ROW["A"],
        "Z": SN132_NUCLEUS_ROW["Z"],
        "B_atom_per_A_MeV": SN132_B_ATOM_PER_A_MEV,
    }
    correction = bridge.electron_correction_MeV(int(row["Z"]))
    return {
        "A": int(row["A"]),
        "Z": int(row["Z"]),
        "N": int(SN132_NUCLEUS_ROW["N"]),
        "B_atom_per_A_MeV": _number(SN132_B_ATOM_PER_A_MEV),
        "B_atom_per_A_sigma_MeV": _number(SN132_B_ATOM_PER_A_SIGMA_MEV),
        "B_atom_source": SN132_B_ATOM_SOURCE,
        "role": SN132_ROLE,
        "electron_correction_total_MeV": _number(correction),
        "electron_correction_per_A_MeV": _number(correction / float(row["A"])),
        "B_nuc_target_per_A_MeV": _number(bridge.bare_binding_per_A_MeV(row)),
    }


def sn132_patched_inputs() -> dict[str, Any]:
    """Deep copy of the cited input bundle with the 132Sn rows added."""

    inputs = bridge._load_inputs()
    inputs["binding"]["values"][SN132] = {
        "A": SN132_NUCLEUS_ROW["A"],
        "Z": SN132_NUCLEUS_ROW["Z"],
        "N": SN132_NUCLEUS_ROW["N"],
        "B_atom_per_A_MeV": SN132_B_ATOM_PER_A_MEV,
        "B_atom_per_A_sigma_MeV": SN132_B_ATOM_PER_A_SIGMA_MEV,
        "role": SN132_ROLE,
    }
    inputs["charge_radii"]["values"][SN132] = {
        "R_ch_fm": SN132_R_CH_FM,
        "R_ch_sigma_fm": SN132_R_CH_SIGMA_FM,
        "role": SN132_ROLE,
    }
    return inputs


def delta2_of(nucleus_row: Mapping[str, int]) -> float:
    """((N-Z)/A)^2 for a nucleus row."""

    a = int(nucleus_row["A"])
    n = int(nucleus_row["N"])
    z = int(nucleus_row["Z"])
    if a != n + z or a <= 0:
        raise Sn132ProbeError(f"invalid nucleus row {dict(nucleus_row)}")
    return ((n - z) / a) ** 2


def verdict_bands(j_values: Mapping[str, float]) -> dict[str, Any]:
    """Pre-registered single-constant verdict from anchored tangents."""

    values = {str(k): float(v) for k, v in j_values.items() if v is not None and math.isfinite(float(v))}
    if len(values) < 2:
        return {
            "status": "INSUFFICIENT_TANGENTS",
            "registered_bands": {"supported": VERDICT_BAND_SUPPORTED, "refuted": VERDICT_BAND_REFUTED},
        }
    j_mean = float(np.mean(list(values.values())))
    deviations = {name: value / j_mean - 1.0 for name, value in values.items()}
    max_dev = max(abs(value) for value in deviations.values())
    if max_dev <= VERDICT_BAND_SUPPORTED:
        status = "UNIVERSAL_CONSTANT_SUPPORTED"
    elif max_dev > VERDICT_BAND_REFUTED:
        status = "UNIVERSAL_CONSTANT_REFUTED"
    else:
        status = "PARTIAL_UNIVERSALITY"
    return {
        "status": status,
        "j_values_MeV": {name: _number(value) for name, value in values.items()},
        "j_mean_MeV": _number(j_mean),
        "relative_deviations": {name: _number(value, 10) for name, value in deviations.items()},
        "max_relative_deviation": _number(max_dev, 10),
        "registered_bands": {"supported_max_dev": VERDICT_BAND_SUPPORTED, "refuted_min_dev": VERDICT_BAND_REFUTED},
        "bands_note": "bands fixed in nvg_sn132_discriminating_set_probe.py before any 132Sn solve",
    }


# ---------------------------------------------------------------------------
# Terminal-case runners (fail-closed per case, evidence retained).
# ---------------------------------------------------------------------------


def _rejection_evidence(case: Mapping[str, Any]) -> dict[str, Any]:
    domain = case.get("domain40_check", {})
    fresh = case.get("fresh24_check", {})
    return {
        "terminal_classification": case.get("terminal_classification"),
        "domain40_converged": bool(domain.get("converged")),
        "domain40_stationarity_acceptance": domain.get("stationarity_acceptance"),
        "domain40_localized_vacuum_exterior": domain.get("localized_vacuum_exterior"),
        "domain40_conserved_NZ_relative_error": domain.get("conserved_NZ_relative_error"),
        "fresh24_converged": bool(fresh.get("converged")),
        "fresh24_stationarity_acceptance": fresh.get("stationarity_acceptance"),
    }


def _solve_case(
    nucleus: str,
    rho: str,
    j_design: float | None,
    *,
    patched: bool,
    inputs_for_radius: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, np.ndarray] | None]:
    """One terminal case; returns (public entry, metrics, profile).

    Rejected cases keep their evidence and return None metrics/profile; the
    caller decides whether a rejection is fatal (baselines) or informative.
    """

    entry: dict[str, Any] = {"rho_choice": rho, "j_design_MeV": j_design}
    patch_ctx = _heldout_context() if patched else nullcontext()
    with patch_ctx:
        spec = dict(static.BRIDGE_NUCLEI[nucleus])
        with ivp._rho_context(rho), ivp._j_design_context(j_design):
            case = bridge._terminal_case(ivp.FAMILY, ivp.ROOT_SCALE, nucleus, multi_seed=True)
        if case.get("terminal_protocol_acceptance") is not True:
            entry.update({"status": "TERMINAL_REJECTED", "rejection_evidence": _rejection_evidence(case)})
            return entry, None, None
        metrics = ivp._terminal_metrics(nucleus, case)
        profile = ca48p._profiles_with_y(case["_domain_solution"])
        radius = None
        if inputs_for_radius is not None:
            radius = bridge._radius_mapping(case["domain40_check"], nucleus, inputs_for_radius)
    entry.update({
        "status": "ACCEPTED",
        "A": int(spec["A"]),
        "N": int(spec["N"]),
        "Z": int(spec["Z"]),
        "terminal_classification": metrics["terminal_classification"],
        "binding_per_A_MeV": _number(metrics["binding_per_A_MeV"]),
        "T_W_MeV": _number(metrics["T_W_MeV"]),
        "E_rho_MeV": _number(metrics["E_rho_MeV"]),
        "total_E_MeV": _number(metrics["total_E_MeV"]),
        "rms_neutron_radius_fm": _number(metrics["rms_neutron_radius_fm"]),
        "rms_point_proton_radius_fm": _number(metrics["rms_point_proton_radius_fm"]),
        "point_neutron_skin_fm": _number(metrics["point_neutron_skin_fm"]),
        "central_density_fm_minus3": _number(metrics["central_density_fm_minus3"]),
        "conserved_NZ_relative_error": _number(metrics["conserved_NZ_relative_error"]),
        "isovector": metrics["isovector"],
    })
    if radius is not None:
        entry["radius_mapping"] = radius
    return entry, metrics, profile


# ---------------------------------------------------------------------------
# Main calculation.
# ---------------------------------------------------------------------------


def _load_retained_ca48() -> dict[str, Any]:
    path = HERE.parent / CA48_RETAINED_REL
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "UNAVAILABLE", "path": str(path), "error": str(exc)}
    return {"status": "LOADED", "path": str(path), "data": payload}


def _targets() -> dict[str, dict[str, Any]]:
    production = bridge._load_inputs()
    targets: dict[str, dict[str, Any]] = {}
    for nucleus in PRODUCTION_NUCLEI:
        row = production["binding"]["values"][nucleus]
        targets[nucleus] = {
            "B_nuc_target_per_A_MeV": bridge.bare_binding_per_A_MeV(row),
            "source": "production cited inputs (finite_static_observables_2026.json)",
        }
    targets["Ca48"] = {
        "B_nuc_target_per_A_MeV": ca48p.ca48_binding_target()["B_nuc_target_per_A_MeV"],
        "source": ca48p.CA48_B_ATOM_SOURCE,
    }
    targets[SN132] = {
        "B_nuc_target_per_A_MeV": sn132_binding_target()["B_nuc_target_per_A_MeV"],
        "source": SN132_B_ATOM_SOURCE,
    }
    return targets


def calculate() -> dict[str, Any]:
    """Run the full discriminating-set probe; strict JSON-serializable output."""

    targets = _targets()

    # --- 1. Accepted baselines (fail closed) --------------------------------
    baselines: dict[str, Any] = {}
    for nucleus in PRODUCTION_NUCLEI:
        case = ivp._terminal_case(nucleus, ivp.BASELINE_RHO)
        metrics = ivp._terminal_metrics(nucleus, case)
        expected = ivp.BASELINE_EXPECTED[nucleus]
        for key, value in expected.items():
            actual = float(metrics[key])
            if not math.isfinite(actual) or abs(actual - float(value)) > ivp.REPRO_TOL:
                raise Sn132ProbeError(
                    f"baseline reproduction failed for {nucleus}.{key}: {actual!r} vs {value!r}"
                )
        baselines[nucleus] = {
            "metrics": metrics,
            "profile": ca48p._profiles_with_y(case["_domain_solution"]),
            "case": case,
        }
    anchor = baselines["Ca40"]["metrics"]
    t40 = float(anchor["T_W_MeV"])
    i40 = float(anchor["isovector"]["I_A_dimensionless"])

    # --- 2. 48Ca re-solve (cross-run stability witness) --------------------
    ca48_entry, ca48_metrics, ca48_profile = _solve_case(ca48p.CA48, ivp.BASELINE_RHO, None, patched=True)
    ca48_j20_entry, ca48_j20_metrics, _ = _solve_case(ca48p.CA48, ivp.CONTACT_RHO, J20_PROBE_MEV, patched=True)
    if ca48_metrics is None:
        raise Sn132ProbeError("48Ca no_rho re-solve was rejected; baselines are not comparable")
    j20_added = ivp._contact_j_added(J20_PROBE_MEV)

    # --- 3. Pre-registered 132Sn prediction (before any 132Sn solve) -------
    def _row_of(nucleus: str) -> dict[str, int]:
        if nucleus in static.BRIDGE_NUCLEI:
            spec = static.BRIDGE_NUCLEI[nucleus]
        elif nucleus == ca48p.CA48:
            spec = ca48p.CA48_NUCLEUS_ROW
        elif nucleus == SN132:
            spec = SN132_NUCLEUS_ROW
        else:
            spec = static.BRIDGE_NUCLEI[nucleus]
        return {"A": int(spec["A"]), "N": int(spec["N"]), "Z": int(spec["Z"])}

    def _tangent(nucleus: str, metrics: Mapping[str, Any]) -> tuple[float, float]:
        row = _row_of(nucleus)
        a = int(row["A"])
        i_a = float(metrics["isovector"]["I_A_dimensionless"])
        t_a = float(metrics["T_W_MeV"])
        slope = ivp.anchored_binding_slope(i_a, t_a, i40, t40, a)
        residual = float(metrics["binding_per_A_MeV"]) - float(targets[nucleus]["B_nuc_target_per_A_MeV"])
        return slope, residual

    prior_tangents: dict[str, float] = {}
    prior_rows: dict[str, dict[str, Any]] = {}
    for nucleus in ("Zr90", "Pb208", "Ca48"):
        metrics = ca48_metrics if nucleus == "Ca48" else baselines[nucleus]["metrics"]
        slope, residual = _tangent(nucleus, metrics)
        j_req = residual / abs(slope) if slope != 0.0 else float("nan")
        prior_tangents[nucleus] = j_req
        prior_rows[nucleus] = {
            "residual_per_A_MeV": _number(residual),
            "anchored_slope": _number(slope),
            "j_tangent_MeV": _number(j_req),
            "delta2": _number(delta2_of(_row_of(nucleus))),
        }
    delta2_sn = delta2_of(SN132_NUCLEUS_ROW)
    j_min_prior = min(prior_tangents.values())
    j_max_prior = max(prior_tangents.values())
    target_sn = float(targets[SN132]["B_nuc_target_per_A_MeV"])
    pre_registered = {
        "statement": "registered from the Zr90/Pb208/Ca48 anchored tangents of this run before any 132Sn solve",
        "prior_tangents_MeV": {name: _number(value) for name, value in prior_tangents.items()},
        "prior_rows": prior_rows,
        "delta2_sn132": _number(delta2_sn),
        "delta2_form_residual_band_per_A_MeV": [
            _number(j_min_prior * delta2_sn, 10),
            _number(j_max_prior * delta2_sn, 10),
        ],
        "delta2_form_binding_band_per_A_MeV": [
            _number(target_sn + j_min_prior * delta2_sn, 10),
            _number(target_sn + j_max_prior * delta2_sn, 10),
        ],
        "note": "delta^2 form is the verified approximation (anchored slope ~ -delta^2); the exact tangent for 132Sn needs its own profile and is computed after the solve",
    }

    # --- 4. 132Sn solves ----------------------------------------------------
    sn_inputs = sn132_patched_inputs()
    sn_entry, sn_metrics, sn_profile = _solve_case(
        SN132, ivp.BASELINE_RHO, None, patched=True, inputs_for_radius=sn_inputs
    )
    sn_j20_entry, sn_j20_metrics, _ = _solve_case(SN132, ivp.CONTACT_RHO, J20_PROBE_MEV, patched=True)

    sn132_resolved = sn_metrics is not None
    full_set: dict[str, dict[str, Any]] = {}
    verdict_input: dict[str, float] = {}
    sn_prediction_check: dict[str, Any] = {"status": "SN132_TERMINAL_UNRESOLVED"}
    if sn132_resolved:
        slope_sn, residual_sn = _tangent(SN132, sn_metrics)
        j_sn = residual_sn / abs(slope_sn) if slope_sn != 0.0 else float("nan")
        row_sn = _row_of(SN132)
        fixed_slope_sn = -float(sn_metrics["isovector"]["I_A_dimensionless"]) / float(row_sn["A"])
        full_set[SN132] = {
            "A": int(row_sn["A"]), "N": int(row_sn["N"]), "Z": int(row_sn["Z"]),
            "delta": _number((row_sn["N"] - row_sn["Z"]) / row_sn["A"], 10),
            "delta2": _number(delta2_sn),
            "residual_per_A_MeV": _number(residual_sn),
            "residual_total_MeV": _number(residual_sn * row_sn["A"]),
            "T_W_MeV": _number(sn_metrics["T_W_MeV"]),
            "I_A": _number(sn_metrics["isovector"]["I_A_dimensionless"]),
            "surface_fraction": _number(sn_metrics["isovector"]["surface_fraction"]),
            "anchored_slope": _number(slope_sn),
            "fixed_scale_slope": _number(fixed_slope_sn),
            "j_tangent_MeV": _number(j_sn),
            "binding_per_A_MeV": _number(sn_metrics["binding_per_A_MeV"]),
            "point_neutron_skin_fm": _number(sn_metrics["point_neutron_skin_fm"]),
            "rms_neutron_radius_fm": _number(sn_metrics["rms_neutron_radius_fm"]),
            "rms_point_proton_radius_fm": _number(sn_metrics["rms_point_proton_radius_fm"]),
        }
        verdict_input[SN132] = j_sn
        band = pre_registered["delta2_form_residual_band_per_A_MeV"]
        sn_prediction_check = {
            "status": "COMPUTED",
            "actual_residual_per_A_MeV": _number(residual_sn),
            "registered_band_per_A_MeV": band,
            "inside_band": bool(float(band[0]) <= residual_sn <= float(band[1])),
            "deviation_from_band_center_per_A_MeV": _number(
                residual_sn - 0.5 * (float(band[0]) + float(band[1])), 10
            ),
            "actual_j_tangent_MeV": _number(j_sn),
            "prior_j_band_MeV": [_number(j_min_prior), _number(j_max_prior)],
            "j_inside_prior_band": bool(j_min_prior <= j_sn <= j_max_prior),
        }
    for nucleus in ("Zr90", "Pb208", "Ca48"):
        metrics = ca48_metrics if nucleus == "Ca48" else baselines[nucleus]["metrics"]
        slope, residual = _tangent(nucleus, metrics)
        row = _row_of(nucleus)
        full_set[nucleus] = {
            "A": int(row["A"]), "N": int(row["N"]), "Z": int(row["Z"]),
            "delta": _number((row["N"] - row["Z"]) / row["A"], 10),
            "delta2": _number(delta2_of(row)),
            "residual_per_A_MeV": _number(residual),
            "residual_total_MeV": _number(residual * row["A"]),
            "T_W_MeV": _number(metrics["T_W_MeV"]),
            "I_A": _number(metrics["isovector"]["I_A_dimensionless"]),
            "surface_fraction": _number(metrics["isovector"]["surface_fraction"]),
            "anchored_slope": _number(slope),
            "fixed_scale_slope": _number(-float(metrics["isovector"]["I_A_dimensionless"]) / row["A"]),
            "j_tangent_MeV": _number(residual / abs(slope)) if slope != 0.0 else None,
            "binding_per_A_MeV": _number(metrics["binding_per_A_MeV"]),
            "point_neutron_skin_fm": _number(metrics["point_neutron_skin_fm"]),
            "rms_neutron_radius_fm": _number(metrics["rms_neutron_radius_fm"]),
            "rms_point_proton_radius_fm": _number(metrics["rms_point_proton_radius_fm"]),
        }
        verdict_input[nucleus] = residual / abs(slope) if slope != 0.0 else float("nan")

    # Chords from the J=20 probes (self-consistent two-point estimates).
    chords: dict[str, Any] = {}
    for nucleus, base_metrics, j20_metrics in (
        ("Ca48", ca48_metrics, ca48_j20_metrics),
        (SN132, sn_metrics, sn_j20_metrics),
    ):
        if base_metrics is None or j20_metrics is None:
            chords[nucleus] = None
            continue
        delta_b = float(base_metrics["binding_per_A_MeV"]) - float(j20_metrics["binding_per_A_MeV"])
        if delta_b == 0.0:
            chords[nucleus] = None
            continue
        slope_chord = delta_b / j20_added
        residual = float(base_metrics["binding_per_A_MeV"]) - float(targets[nucleus]["B_nuc_target_per_A_MeV"])
        chords[nucleus] = {
            "chord_slope_dB_dj": _number(slope_chord),
            "j_chord_MeV": _number(residual / abs(slope_chord)),
        }
        if nucleus in full_set:
            full_set[nucleus]["j_chord_J20_MeV"] = chords[nucleus]["j_chord_MeV"]

    # --- 5. Verdict ----------------------------------------------------------
    verdict = verdict_bands(verdict_input)

    # --- 6. Universal-j demonstration ---------------------------------------
    universal: dict[str, Any] = {
        "construction": "j_bar is the mean of the four anchored tangents; one constant for all nuclei, no per-nucleus tuning",
        "t_contact_MeV": _number(T_CONTACT_MEV),
        "note": "declared probe through the existing contact machinery at fixed s*; not an addition to the theory",
    }
    universal_rows: dict[str, Any] = {}
    if sn132_resolved and verdict.get("j_mean_MeV") is not None:
        j_bar = float(verdict["j_mean_MeV"])
        j_design_universal = universal_design_j(j_bar)
        universal.update({
            "j_bar_MeV": _number(j_bar),
            "J_design_universal_MeV": _number(j_design_universal),
        })
        case_specs = (
            ("Ca40", False),
            ("Zr90", False),
            ("Pb208", False),
            ("Ca48", True),
            (SN132, True),
        )
        for nucleus, patched in case_specs:
            entry, metrics, _profile = _solve_case(nucleus, ivp.CONTACT_RHO, j_design_universal, patched=patched)
            row_out: dict[str, Any] = {"rho_choice": ivp.CONTACT_RHO, "j_design_MeV": _number(j_design_universal)}
            if metrics is None:
                row_out.update({"status": entry["status"], "rejection_evidence": entry["rejection_evidence"]})
                universal_rows[nucleus] = row_out
                continue
            target = float(targets[nucleus]["B_nuc_target_per_A_MeV"])
            residual_after = float(metrics["binding_per_A_MeV"]) - target
            # Fixed-scale linear prediction from the no-rho state: dB/dj|_s = -I_A/A
            # (for Ca40 this is the anchor-drift estimate).
            row = _row_of(nucleus)
            if nucleus == SN132:
                base_metrics = sn_metrics
            elif nucleus == "Ca48":
                base_metrics = ca48_metrics
            else:
                base_metrics = baselines[nucleus]["metrics"]
            i_a = float(base_metrics["isovector"]["I_A_dimensionless"])
            residual_before = float(base_metrics["binding_per_A_MeV"]) - target
            predicted_after = residual_before - j_bar * (i_a / row["A"])
            row_out.update({
                "status": "ACCEPTED",
                "binding_per_A_MeV": _number(metrics["binding_per_A_MeV"]),
                "residual_before_per_A_MeV": _number(residual_before),
                "residual_after_per_A_MeV": _number(residual_after),
                "residual_after_total_MeV": _number(residual_after * row["A"]),
                "predicted_after_linear_per_A_MeV": _number(predicted_after),
                "nonlinearity_per_A_MeV": _number(residual_after - predicted_after),
                "point_neutron_skin_fm": _number(metrics["point_neutron_skin_fm"]),
            })
            universal_rows[nucleus] = row_out
        # Anchor drift that a re-calibration would remove (scale diagnostic).
        ca40_after = universal_rows.get("Ca40", {})
        if ca40_after.get("status") == "ACCEPTED":
            delta_b40 = float(ca40_after["binding_per_A_MeV"]) - float(anchor["binding_per_A_MeV"])
            delta_e40_total = -float(anchor["A"]) * delta_b40
            universal["ca40_anchor_drift"] = {
                "delta_B_per_A_MeV": _number(delta_b40),
                "ds_over_s_to_restore_anchor": _number(-delta_e40_total / (2.0 * t40), 10),
                "identity": "dE/ds = 2 T_W/s => ds/s = -dE/(2 T_W)",
            }
        accepted_after = [row for row in universal_rows.values() if row.get("status") == "ACCEPTED" and "residual_after_per_A_MeV" in row]
        if accepted_after:
            before = [abs(float(row["residual_before_per_A_MeV"])) for row in accepted_after]
            after = [abs(float(row["residual_after_per_A_MeV"])) for row in accepted_after]
            universal["residual_collapse"] = {
                "accepted_case_count": len(accepted_after),
                "max_abs_residual_before_per_A_MeV": _number(max(before), 10),
                "max_abs_residual_after_per_A_MeV": _number(max(after), 10),
                "reduction_factor": _number(max(before) / max(after), 10) if max(after) > 0 else None,
            }
    universal["cases"] = universal_rows

    # --- 7. Operator patterns on the four held-out profiles -----------------
    operator_profiles: dict[str, dict[str, np.ndarray]] = {
        "Zr90": baselines["Zr90"]["profile"],
        "Pb208": baselines["Pb208"]["profile"],
        "Ca48": ca48_profile,
    }
    if sn_profile is not None:
        operator_profiles[SN132] = sn_profile
    operator_values: dict[str, dict[str, float]] = {}
    for nucleus, profile in operator_profiles.items():
        if profile is None:
            continue
        grad = ca48p.gradient_isovector_integrals(profile["r_fm"], profile["nn_fm3"], profile["np_fm3"])
        iso = full_set[nucleus]
        operator_values[nucleus] = {
            "bulk": float(iso["I_A"]),
            "surface_geometric": float(iso["I_A"]) * float(iso["surface_fraction"]),
            "grad_full": float(grad["I_grad_full_fm_minus2"]),
            "grad_masked_eps1": float(grad["masked_by_eps"]["1.0"]["I_grad_fm_minus2"]),
            "T_W": float(iso["T_W_MeV"]),
            "A": int(iso["A"]),
        }
    operator_split: dict[str, Any] = {"per_nucleus_operators": {
        nucleus: {key: _number(value) for key, value in values.items() if key != "A"} | {"A": values["A"]}
        for nucleus, values in operator_values.items()
    }}
    if "Ca40" in baselines:
        ref_profile = baselines["Ca40"]["profile"]
        ref_grad = ca48p.gradient_isovector_integrals(ref_profile["r_fm"], ref_profile["nn_fm3"], ref_profile["np_fm3"])
        ref_values = {
            "bulk": i40,
            "surface_geometric": i40 * float(anchor["isovector"]["surface_fraction"]),
            "grad_full": float(ref_grad["I_grad_full_fm_minus2"]),
            "grad_masked_eps1": float(ref_grad["masked_by_eps"]["1.0"]["I_grad_fm_minus2"]),
            "T_W": t40,
            "A": int(anchor["A"]),
        }
        slopes: dict[str, dict[str, float]] = {}
        required: dict[str, Any] = {}
        normalized: dict[str, Any] = {}
        for operator in ("bulk", "surface_geometric", "grad_full", "grad_masked_eps1"):
            op_slopes = {}
            op_required = {}
            for nucleus, values in operator_values.items():
                slope = ivp.anchored_binding_slope(
                    values[operator], values["T_W"], ref_values[operator], ref_values["T_W"], values["A"]
                )
                op_slopes[nucleus] = _number(slope)
                residual = float(full_set[nucleus]["residual_per_A_MeV"])
                op_required[nucleus] = _number(residual / abs(float(slope))) if float(slope) != 0.0 else None
            slopes[operator] = op_slopes
            required[operator] = {
                "values": op_required,
                "units": "MeV fm^2" if operator.startswith("grad_") else "MeV",
            }
            zr = op_required.get("Zr90")
            if zr not in (None, 0.0):
                normalized[operator] = {
                    nucleus: _number(float(value) / float(zr), 10)
                    for nucleus, value in op_required.items()
                    if value is not None
                }
        operator_split.update({
            "anchored_slopes_per_unit_coupling": slopes,
            "required_coupling_to_annihilate_residual": required,
            "normalized_required_coupling_over_Zr90": normalized,
        })

    # --- 8. 132Sn radius comparison ------------------------------------------
    sn_radius: dict[str, Any] = {"status": "SN132_TERMINAL_UNRESOLVED"}
    if sn132_resolved and "radius_mapping" in sn_entry:
        mapping = sn_entry["radius_mapping"]
        sn_radius = {
            "status": "COMPUTED",
            "model_R1b_fm": mapping.get("R1b_fm"),
            "evaluated_R_ch_fm": _number(SN132_R_CH_FM),
            "evaluated_R_ch_sigma_fm": _number(SN132_R_CH_SIGMA_FM),
            "delta_required_fm2": mapping.get("delta_required_fm2"),
            "R_ch_source": SN132_R_CH_SOURCE,
            "model_point_skin_fm": _number(sn_metrics["point_neutron_skin_fm"]),
            "skin_data_note": "no parity-violating skin measurement exists for 132Sn; skins are not part of this test",
            "beta_instability_caveat": SN132_B_ATOM_SOURCE,
        }

    # --- 9. Controls ----------------------------------------------------------
    retained = _load_retained_ca48()
    stability: dict[str, Any] = {"retained_status": retained["status"], "path": retained.get("path")}
    if retained["status"] == "LOADED":
        retained_case = retained["data"]["part_a_ca48_heldout"]["cases"].get("no_rho", {})
        if retained_case.get("status") == "ACCEPTED":
            stability["ca48_cross_run_witness"] = {
                key: {
                    "this_run": _number(ca48_metrics[key2]),
                    "retained": _number(retained_case[key]),
                    "difference": _number(float(ca48_metrics[key2]) - float(retained_case[key])),
                }
                for key, key2 in (
                    ("binding_per_A_MeV", "binding_per_A_MeV"),
                    ("rms_neutron_radius_fm", "rms_neutron_radius_fm"),
                    ("rms_point_proton_radius_fm", "rms_point_proton_radius_fm"),
                    ("point_neutron_skin_fm", "point_neutron_skin_fm"),
                )
            }
            stability["deterministic_match_expected"] = True
    decimation_sn = None
    if sn_profile is not None:
        decimation_sn = ca48p.decimation_control(sn_profile)

    return _jsonable({
        "schema": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "provenance": {
            "family": ivp.FAMILY,
            "root_scale": _number(ivp.ROOT_SCALE),
            "root_scale_source": ivp.ROOT_SCALE_SOURCE,
            "baseline_rho": ivp.BASELINE_RHO,
            "anchor": "Ca40 sole conditional calibration anchor (unchanged; no retuning here)",
            "heldout_set": list(HELDOUT),
            "sn132_registration": "132Sn added to complete the pre-registered discriminating set; prediction registered before its solve; not used in any calibration",
            "no_new_interaction_declared": True,
            "no_coefficient_tuned_to_residuals": True,
        },
        "baseline_reproduction": {
            nucleus: {
                "binding_per_A_MeV": _number(baselines[nucleus]["metrics"]["binding_per_A_MeV"]),
                "T_W_MeV": _number(baselines[nucleus]["metrics"]["T_W_MeV"]),
                "reproduced_vs_R3": True,
            }
            for nucleus in PRODUCTION_NUCLEI
        },
        "targets": {
            nucleus: {"B_nuc_target_per_A_MeV": _number(targets[nucleus]["B_nuc_target_per_A_MeV"]), "source": targets[nucleus]["source"]}
            for nucleus in ("Ca40", "Ca48", "Zr90", "Pb208", SN132)
        },
        "ca48_re_run": {
            "no_rho": {k: v for k, v in ca48_entry.items()},
            "J20": {k: v for k, v in ca48_j20_entry.items()},
            "j20_added_MeV": _number(j20_added),
        },
        "pre_registered_prediction": pre_registered,
        "sn132_cases": {
            "no_rho": {k: v for k, v in sn_entry.items()},
            "J20": {k: v for k, v in sn_j20_entry.items()},
        },
        "full_set_table": full_set,
        "chords_J20": chords,
        "sn132_prediction_check": sn_prediction_check,
        "verdict": verdict,
        "universal_j_demonstration": universal,
        "operator_split": operator_split,
        "sn132_radius": sn_radius,
        "controls": {
            "ca48_stability_witness": stability,
            "decimation_sn132": decimation_sn,
            "analytic_validation": ca48p.analytic_validation(),
            "t_contact_mev": _number(T_CONTACT_MEV),
        },
        "limits": [
            "conditional diagnostic of one accepted branch; evidence_weight=0; no statistical claim",
            "132Sn is held out from calibration, but its binding is well measured: descriptive out-of-sample comparison, not a blind prediction",
            "verdict bands (0.15/0.35) are registered conventions, not statistical confidence intervals",
            "required couplings and the universal j_bar are conditional arithmetic, not parameters of the theory; the universal-j solves are declared probes at fixed s*, not a re-calibration",
            "the delta^2-form prediction is the verified linear approximation; the exact 132Sn tangent uses its own post-solve profile",
            "132Sn is beta-unstable (t1/2 = 39.7 s); the static bound state and the AME2020 binding remain well defined, and the beta Q value does not enter the comparison",
            "radii and skins are outside the scope of this test: no PV skin measurement exists for 132Sn, and the static TF branch fails skins/shell isotope shifts by construction",
            "spin-orbit, pairing and shell structure are absent from the branch; the verdict concerns the bulk isovector (symmetry-energy) channel only",
        ],
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional explicit path for the strict JSON result",
    )
    args = parser.parse_args(argv)
    result = calculate()
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
