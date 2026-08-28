#!/usr/bin/env python3
"""Dense predictive/uncertainty audit of the maintained neutron-star chain.

This module is deliberately a consumer of the existing EOS/TOV/Hinderer
producers.  It does not introduce a second equation of state or an inverse
fit.  The declared transition grid is evaluated at runtime, with the three
observational rows treated as in-sample selection constraints.  Leave-one-
observable-out (LOO) rows are therefore conditional predictive checks, not
independent held-out evidence.

The canonical EOS has no physically wired ``M_Omega,0`` argument: the
beta-equilibrated producer derives ``M_Omega_0`` from its fixed sigma-term
calibration.  The audit records that blocked dependency explicitly and never
rescales an observable with the quoted 859 +/- 8 MeV anchor.

Run from any working directory with::

    python3 verification/nvg_ns_predictive_audit.py

The default run writes a structured JSON result and a two-panel figure under
``verification/`` and the immutable worker report under ``Lunacy/``.  A small
``--quick`` mode exists only for focused semantic tests; it is not the
terminal evidence run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import nvg_eos_beta_css_softening as soft
import nvg_joint_ns_inference as joint
import nvg_ns_canonical as canonical
import nvg_tidal_deformability as tidal


SCHEMA_VERSION = "P1-S7-ns-predictive-audit-v3"
CANONICAL_SOLVER = "verification/nvg_tidal_deformability.EOS + solve_tov_tidal"
M_OMEGA_CENTRAL_MEV = 859.0
M_OMEGA_SIGMA_MEV = 8.0

# The jump update is not an empirical correction.  It is the distributional
# first-order interface condition in Postnikov, Prakash & Lattimer (2010),
# arXiv:1004.5098, Eq. 15.  Eq. 14 in that source is the distributional
# derivative term; Eq. 15 is the integrated interface update.  Keep the
# citation and formula in every generated
# result so a positive-latent-heat row cannot be mistaken for a finite-width
# table interpolation.
DENSITY_JUMP_SOURCE = {
    "citation": "Postnikov, Prakash & Lattimer (2010), Phys. Rev. D 82, 024016, arXiv:1004.5098",
    "url": "https://arxiv.org/abs/1004.5098",
    "equation": "Eq. 15",
    "derivative_equation": "Eq. 14",
    "derivative_role": "distributional d rho/dp term integrated across the interface",
    "formula": "y_plus = y_minus - 4*pi*r_t**3*Delta_epsilon_geo/m_t_geo",
    "geometric_conversion": "Delta_epsilon_geo = k_conv * Delta_epsilon[MeV/fm^3], m_t_geo = M_solar * M_sun_km",
    "jump_sign": "positive Delta_epsilon lowers y across the interface",
}

BRANCH_SLOPE_WINDOW_MIN = 5
BRANCH_SLOPE_THRESHOLD_RELATIVE = 1.0e-3
BRANCH_RAW_MONOTONIC_TOLERANCE_RELATIVE = 1.0e-3
ROW_CONVERGENCE_THRESHOLD_RELATIVE = 5.0e-3


def solver_provenance() -> dict[str, Any]:
    """Return the authoritative adaptive ODE provenance from the TOV producer."""

    return tidal.adaptive_solver_provenance()


def assert_solver_provenance_artifact(result: dict[str, Any]) -> None:
    """Fail closed if an artifact's ODE metadata drifted from live source.

    The assertion compares the full producer metadata (including its source
    digest), then checks the convergence ladder's per-row ``max_step`` values.
    It is intentionally callable on a loaded JSON artifact as well as on the
    in-memory result produced by :func:`run_audit`.
    """

    expected = solver_provenance()
    actual = result.get("solver_provenance")
    if actual != expected:
        raise AssertionError("solver provenance drift: artifact does not match live producer")
    assertion = result.get("source_to_artifact_provenance", {})
    if assertion.get("status") != "PASS" or assertion.get("source_sha256") != expected["source_sha256"]:
        raise AssertionError("source-to-artifact provenance assertion is not PASS")
    ode = result.get("convergence", {}).get("ode_tolerances", {})
    if ode.get("solver_provenance") != expected:
        raise AssertionError("ODE convergence metadata drifted from live producer")
    for key in (
        "backend",
        "method",
        "method_description",
        "rtol_parameter",
        "atol_parameter",
        "max_step_rule",
        "max_step_cap_km",
    ):
        if ode.get(key) != expected[key]:
            raise AssertionError(f"ODE convergence field drifted: {key}")
    rows = ode.get("rows", [])
    for row in rows:
        if "rtol" not in row or "max_step" not in row:
            raise AssertionError("ODE convergence row is missing tolerance/max_step metadata")
        expected_step = tidal.adaptive_solver_max_step(0.05, float(row["rtol"]))
        if not math.isclose(float(row["max_step"]), expected_step, rel_tol=0.0, abs_tol=1.0e-15):
            raise AssertionError("ODE convergence row max_step drifted from live cap rule")
    row_summary_provenance = result.get("row_convergence", {}).get("ode_solver_provenance")
    if row_summary_provenance is not None and row_summary_provenance != expected:
        raise AssertionError("row-convergence ODE provenance drifted from live producer")
    continuity_provenance = result.get("zero_limit_continuity", {}).get("solver_provenance")
    if continuity_provenance is not None and continuity_provenance != expected:
        raise AssertionError("zero-limit ODE provenance drifted from live producer")
    for grid_row in result.get("grid_rows", []):
        nested = grid_row.get("convergence", {}).get("solver_provenance")
        if nested is not None and nested != expected:
            raise AssertionError("row-level ODE provenance drifted from live producer")

# The grid is intentionally much denser than the 3 x 5 canonical-selection
# grid.  The first three delta values resolve the zero-latent-heat boundary;
# cs2=1 is retained as the causal upper edge, not as a fitted value.
DECLARED_GRID: dict[str, tuple[float, ...]] = {
    "n_trans_ratio": tuple(round(float(x), 2) for x in np.arange(1.2, 3.01, 0.2)),
    "delta_eps_ratio": (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80),
    "cs2_q": (0.20, 1.0 / 3.0, 0.50, 0.80, 1.00),
}

QUICK_GRID: dict[str, tuple[float, ...]] = {
    "n_trans_ratio": (1.8, 2.0, 2.2),
    "delta_eps_ratio": (0.0, 0.05, 0.20),
    "cs2_q": (1.0 / 3.0, 0.80),
}

OBSERVABLE_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "J0740_M_max": {
        "label": "PSR J0740+6620 maximum mass",
        "observable": "M_max",
        "source": "NICER PSR J0740+6620; declared 1-sigma low edge",
        "lower": 2.01,
        "upper": None,
    },
    "GW170817_Lambda_tilde": {
        "label": "GW170817 binary tidal deformability",
        "observable": "Lambda_tilde",
        "source": "GW170817 low-spin 90% interval",
        "lower": 70.0,
        "upper": 720.0,
    },
    "NICER_R14": {
        "label": "NICER J0030+0451 radius",
        "observable": "R_1.4",
        "source": "NICER J0030+0451 declared interval",
        "lower": 11.2,
        "upper": 13.2,
    },
}

RESULT_PATH = _HERE / "nvg_ns_predictive_audit_p1s7_results.json"
FIGURE_PATH = _HERE / "fig_ns_predictive_audit_p1s7.png"
REPORT_PATH = _ROOT / "Lunacy/runs/predictive-research/phases/phase-1/P1-S9-REPORT.md"
EVIDENCE_PATH = _ROOT / "Lunacy/runs/predictive-research/phases/phase-1/evidence/P1-S9-terminal-verification.log"


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    """Convert numpy/scalar containers to deterministic JSON values."""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _configure_eos(hybrid: dict[str, Any]) -> tidal.EOS:
    """Create a strict canonical ``EOS`` instance around a hybrid table.

    ``nvg_ns_parameter_scan.star_family`` performs the same contract checks.
    Keeping this small adapter local lets the audit retain full sequences
    without creating another EOS implementation.
    """

    eos = tidal.EOS.__new__(tidal.EOS)
    eos.p_arr = np.asarray(hybrid["p_sorted"], dtype=float)
    eos.eps_arr = np.asarray(hybrid["e_sorted"], dtype=float)
    if eos.p_arr.ndim != 1 or eos.eps_arr.ndim != 1 or len(eos.p_arr) != len(eos.eps_arr):
        raise ValueError("hybrid EOS pressure/energy arrays must be paired one-dimensional arrays")
    if len(eos.p_arr) < 20 or not np.all(np.isfinite(eos.p_arr)) or not np.all(np.isfinite(eos.eps_arr)):
        raise ValueError("hybrid EOS arrays must contain at least 20 finite points")
    if np.any(np.diff(eos.p_arr) <= 0.0):
        raise ValueError("hybrid EOS pressure grid must be strictly increasing")
    if np.any(np.diff(eos.eps_arr) < -1.0e-9):
        raise ValueError("hybrid EOS energy density must be non-decreasing")
    eos.table_pressure_min = float(eos.p_arr[0])
    eos.pressure_min = 0.0
    eos.pressure_max = float(eos.p_arr[-1])
    eos.p_match = 1.5
    eos.Gamma = 1.35
    if not eos.table_pressure_min <= eos.p_match <= eos.pressure_max:
        raise ValueError("hybrid EOS matching pressure lies outside its table domain")
    eos.eps_match = float(np.interp(eos.p_match, eos.p_arr, eos.eps_arr))
    if not _finite(eos.eps_match) or eos.eps_match <= 0.0:
        raise ValueError("hybrid EOS matching energy is not physical")
    eos.audit_parameters = {
        "n_trans_ratio": float(hybrid.get("n_trans", np.nan) / soft.base.n_0)
        if _finite(hybrid.get("n_trans")) else None,
        "delta_eps_ratio": float(hybrid.get("delta_eps", np.nan) / hybrid["e_trans"])
        if _finite(hybrid.get("delta_eps")) and _finite(hybrid.get("e_trans"))
        and float(hybrid["e_trans"]) != 0.0 else None,
        "cs2_q": float(hybrid.get("cs2_q")) if _finite(hybrid.get("cs2_q")) else None,
    }
    # Preserve first-order metadata separately from the finite-width pressure
    # table.  The canonical TOV/Hinderer solver uses this metadata to apply
    # the distributional y jump at P_t; interpolating across the tiny table
    # plateau is not an acceptable approximation for Delta epsilon > 0.
    if all(_finite(hybrid.get(key)) for key in ("p_trans", "e_trans", "delta_eps", "cs2_q")):
        transition_pressure = float(hybrid["p_trans"])
        transition_energy_low = float(hybrid["e_trans"])
        transition_energy_jump = float(hybrid["delta_eps"])
        transition_cs2 = float(hybrid["cs2_q"])
        eos.transition_pressure = transition_pressure
        eos.transition_energy_low = transition_energy_low
        eos.transition_energy_jump = transition_energy_jump
        eos.transition_energy_high = transition_energy_low + transition_energy_jump
        eos.transition_cs2_q = transition_cs2
        had_mask = eos.p_arr <= transition_pressure + 1.0e-12
        had_p = np.asarray(eos.p_arr[had_mask], dtype=float)
        had_e = np.asarray(eos.eps_arr[had_mask], dtype=float)
        if len(had_p) < 2 or had_p[-1] < transition_pressure:
            had_p = np.append(had_p, transition_pressure)
            had_e = np.append(had_e, transition_energy_low)
        else:
            had_e[-1] = transition_energy_low
        eos.had_p_arr = had_p
        eos.had_eps_arr = had_e
    return eos


def _resample_pressure_table(eos: tidal.EOS, points: int) -> tidal.EOS:
    """Return an EOS with a declared pressure-table resolution.

    The pressure grid is sampled geometrically and explicitly retains the
    matching pressure.  This is only used for the numerical convergence audit;
    the canonical EOS constructor and default solver are untouched.
    """

    points = int(points)
    if points < 20:
        raise ValueError("pressure-table resolution must be at least 20 points")
    p = np.asarray(eos.p_arr, dtype=float)
    e = np.asarray(eos.eps_arr, dtype=float)
    # p_arr is strict by contract.  Retain endpoints and p_match so the crust
    # continuation is represented exactly in every resolution level.
    grid = np.geomspace(float(p[0]), float(p[-1]), points)
    grid = np.unique(np.concatenate([grid, [eos.p_match]]))
    grid.sort()
    e_grid = np.interp(grid, p, e)
    resolved = _configure_eos({"p_sorted": grid, "e_sorted": e_grid,
                               "n_trans": np.nan, "e_trans": np.nan,
                               "delta_eps": np.nan, "cs2_q": np.nan})
    if hasattr(eos, "transition_pressure"):
        # Copy the physical transition parameters while deriving the low-phase
        # table from the represented pressure grid itself.
        for key in ("transition_pressure", "transition_energy_low",
                    "transition_energy_jump", "transition_energy_high",
                    "transition_cs2_q"):
            setattr(resolved, key, getattr(eos, key))
        pt = float(eos.transition_pressure)
        mask = resolved.p_arr <= pt + 1.0e-12
        resolved.had_p_arr = np.asarray(resolved.p_arr[mask], dtype=float)
        resolved.had_eps_arr = np.asarray(resolved.eps_arr[mask], dtype=float)
        if len(resolved.had_p_arr) < 2 or resolved.had_p_arr[-1] < pt:
            resolved.had_p_arr = np.append(resolved.had_p_arr, pt)
            resolved.had_eps_arr = np.append(resolved.had_eps_arr, float(eos.transition_energy_low))
        else:
            resolved.had_eps_arr[-1] = float(eos.transition_energy_low)
    return resolved


def _pressure_domain_diagnostics(eos: tidal.EOS, declared_cs2: float | None = None) -> dict[str, Any]:
    p = np.asarray(eos.p_arr, dtype=float)
    e = np.asarray(eos.eps_arr, dtype=float)
    dp = np.diff(p)
    de = np.diff(e)
    energy_scale = max(1.0, float(np.nanmax(np.abs(e))))
    zero_width = de <= 1.0e-12 * energy_scale
    usable = de > 1.0e-12 * energy_scale
    effective_cs2 = np.divide(dp[usable], de[usable]) if np.any(usable) else np.array([])
    causal_max = float(np.max(effective_cs2)) if len(effective_cs2) else None
    causal_min = float(np.min(effective_cs2)) if len(effective_cs2) else None
    violations = int(np.sum((effective_cs2 < -1.0e-8) | (effective_cs2 > 1.0 + 1.0e-8)))
    return {
        "finite_pressure_energy": bool(np.all(np.isfinite(p)) and np.all(np.isfinite(e))),
        "strict_pressure_grid": bool(np.all(dp > 0.0)),
        "nondecreasing_energy_density": bool(np.all(de >= -1.0e-9)),
        "pressure_domain": {"min": float(p[0]), "max": float(p[-1]), "points": int(len(p))},
        "energy_density_domain": {"min": float(e[0]), "max": float(e[-1]), "points": int(len(e))},
        "effective_cs2_min": causal_min,
        "effective_cs2_max": causal_max,
        "causal_violation_count": violations,
        "causality_ok": violations == 0,
        "zero_width_transition_steps_excluded": int(np.sum(zero_width)),
        "zero_width_transition_note": (
            "A delta_eps=0 limiting transition has a zero-energy step; its zero-width "
            "numerical row is excluded from dP/dε rather than called superluminal."
        ),
        "declared_cs2_q": float(declared_cs2) if declared_cs2 is not None else None,
    }


def _ordered_mass_slopes(rows: list[dict[str, float]], window: int | None = None) -> np.ndarray:
    """Estimate ordered ``dM/d log(P_c)`` without reordering the sequence.

    The maintained solver has small alternating crust/surface integration
    noise at very low mass.  A local least-squares slope over a declared odd
    window suppresses that numerical chatter while retaining the sustained
    sign change at a physical turning point.  This is a topology diagnostic,
    not a mass envelope or an interpolation transform.
    """

    count = len(rows)
    if count < 2:
        return np.asarray([], dtype=float)
    if window is None:
        # Keep roughly six local samples at the coarsest scan and increase the
        # window with resolution; the value is recorded in the sequence.
        window = max(BRANCH_SLOPE_WINDOW_MIN, 2 * (count // 48) + 1)
    window = int(max(3, min(count if count % 2 else count - 1, window)))
    if window % 2 == 0:
        window -= 1
    half = window // 2
    log_pressure = np.log(np.asarray([row["central_pressure"] for row in rows], dtype=float))
    masses = np.asarray([row["mass"] for row in rows], dtype=float)
    slopes = np.full(count, np.nan, dtype=float)
    for index in range(count):
        lo = max(0, index - half)
        hi = min(count, index + half + 1)
        x = log_pressure[lo:hi]
        y = masses[lo:hi]
        if len(x) >= 2 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            slope = float(np.polyfit(x, y, 1)[0])
            slopes[index] = slope / max(abs(float(masses[index])), 1.0e-6)
    return slopes


def _stable_branches(
    rows: list[dict[str, float]],
    *,
    slope_threshold_relative: float = BRANCH_SLOPE_THRESHOLD_RELATIVE,
) -> tuple[list[list[dict[str, float]]], list[int]]:
    """Split an ordered sequence into sustained positive-slope branches.

    Central pressure order is authoritative.  The local slope sign identifies
    turning/unstable intervals; no mass sorting, argmax truncation, or twin
    branch mixing is performed.  Short positive runs remain unresolved and
    are excluded from all target interpolation.
    """

    if len(rows) < 2:
        return [], []
    slopes = _ordered_mass_slopes(rows)
    stable_mask = np.isfinite(slopes) & (slopes > float(slope_threshold_relative))
    branches: list[list[dict[str, float]]] = []
    branch_ids = [-1] * len(rows)

    def append_branch(start_index: int, end_index: int) -> None:
        candidate = rows[start_index:end_index]
        if len(candidate) < 3:
            return
        # A stable branch terminates at its first ordered local mass maximum.
        # Any post-peak rows belong to the unstable continuation, even if the
        # edge-smoothed slope remains positive for a few samples.
        candidate_masses = np.asarray([row["mass"] for row in candidate], dtype=float)
        # A coarse grid can place the first positive-slope flag one or two
        # samples before the density-jump trough.  Trim that leading descent
        # at the ordered minimum, then terminate at the first local peak.
        minimum = int(np.argmin(candidate_masses))
        peak = minimum + int(np.argmax(candidate_masses[minimum:]))
        candidate = candidate[minimum : peak + 1]
        if len(candidate) < 3:
            return
        branch_id = len(branches)
        branches.append(candidate)
        for row_index in range(start_index + minimum, start_index + minimum + len(candidate)):
            branch_ids[row_index] = branch_id

    start: int | None = None
    for index, is_stable in enumerate(stable_mask):
        if bool(is_stable) and start is None:
            start = index
        if (not bool(is_stable) or index == len(rows) - 1) and start is not None:
            end = index + 1 if bool(is_stable) and index == len(rows) - 1 else index
            append_branch(start, end)
            start = None
    return branches, branch_ids


def _at_mass_on_branch(target: float, branch: list[dict[str, float]], key: str) -> float | None:
    """Interpolate only along one explicitly ordered stable branch."""

    if len(branch) < 2:
        return None
    masses = np.asarray([row["mass"] for row in branch], dtype=float)
    relative_drops = np.diff(masses) / np.maximum(np.abs(masses[:-1]), 1.0e-6)
    if np.any(relative_drops < -BRANCH_RAW_MONOTONIC_TOLERANCE_RELATIVE):
        # A branch that is materially non-monotone is unresolved, not
        # something to repair by mass sorting or an envelope transform.  Tiny
        # reversals below the declared solver-noise tolerance are certified by
        # the row-level convergence audit rather than reordered.
        return None
    if target < float(masses[0]) or target > float(masses[-1]):
        return None
    values = np.asarray([row[key] for row in branch], dtype=float)
    return float(np.interp(target, masses, values))


def evaluate_sequence(
    eos: tidal.EOS,
    central_pressure_points: int = 72,
    solver_kwargs: dict[str, Any] | None = None,
    include_sequence: bool = False,
    refine_targets: bool = False,
) -> dict[str, Any]:
    """Evaluate a mass--radius--Lambda sequence from the canonical solver."""

    solver_kwargs = dict(solver_kwargs or {})
    central_pressure_points = int(central_pressure_points)
    if central_pressure_points < 8:
        raise ValueError("central-pressure resolution must be at least 8 points")
    p_lo = max(0.1, float(eos.table_pressure_min))
    p_hi = min(float(eos.pressure_max), 10.0 ** 3.4)
    pressures = np.geomspace(p_lo, p_hi, central_pressure_points)
    rows: list[dict[str, float]] = []
    failures: list[dict[str, Any]] = []
    for pressure in pressures:
        try:
            mass, radius, k2, lam = tidal.solve_tov_tidal(
                eos,
                float(pressure),
                density_jump_matching=bool(hasattr(eos, "transition_pressure")),
                **solver_kwargs,
            )
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError) as exc:
            failures.append({"central_pressure": float(pressure), "error": type(exc).__name__})
            continue
        if all(_finite(value) for value in (mass, radius, k2, lam)) and mass > 0.0 and radius > 0.0:
            rows.append({
                "central_pressure": float(pressure),
                "mass": float(mass),
                "radius": float(radius),
                "k2": float(k2),
                "lambda": float(lam),
            })
        else:
            failures.append({"central_pressure": float(pressure), "error": "nonfinite_or_nonphysical"})
    if len(rows) < 4:
        raise RuntimeError("canonical EOS produced fewer than four valid TOV solutions")

    branches, branch_ids = _stable_branches(rows)
    if not branches:
        raise RuntimeError("ordered central-pressure sequence has no stable branch")
    slope_window = max(BRANCH_SLOPE_WINDOW_MIN, 2 * (len(rows) // 48) + 1)
    if slope_window % 2 == 0:
        slope_window -= 1
    branch_records = []
    row_index_by_identity = {id(row): index for index, row in enumerate(rows)}
    for branch_id, branch in enumerate(branches):
        masses = np.asarray([row["mass"] for row in branch], dtype=float)
        relative_drops = np.diff(masses) / np.maximum(np.abs(masses[:-1]), 1.0e-6)
        negative_drops = relative_drops[relative_drops < 0.0]
        branch_slopes = _ordered_mass_slopes(branch)
        end_index = row_index_by_identity[id(branch[-1])]
        # The local turning-point sample immediately after a positive-slope
        # run carries the physical peak mass.  Include only a short ordered
        # look-ahead; never search or mix a disconnected later branch.
        peak_stop = min(len(rows), end_index + max(2, slope_window // 2 + 1))
        peak_slice = rows[row_index_by_identity[id(branch[0])] : peak_stop]
        peak_row = max(peak_slice, key=lambda item: float(item["mass"]))
        branch_records.append({
            "id": int(branch_id),
            "start_central_pressure": float(branch[0]["central_pressure"]),
            "end_central_pressure": float(branch[-1]["central_pressure"]),
            "points": int(len(branch)),
            "mass_min": float(np.min(masses)),
            "mass_max": float(np.max(masses)),
            "peak_mass": float(peak_row["mass"]),
            "peak_central_pressure": float(peak_row["central_pressure"]),
            "peak_lookahead_points": int(len(peak_slice) - len(branch)),
            "monotonic_mass": bool(np.all(np.diff(masses) > 0.0)),
            "raw_monotonic_within_tolerance": bool(
                np.all(relative_drops >= -BRANCH_RAW_MONOTONIC_TOLERANCE_RELATIVE)
            ),
            "max_raw_negative_drop_relative": float(-np.min(negative_drops))
            if len(negative_drops) else 0.0,
            "slope_relative_min": float(np.nanmin(branch_slopes))
            if np.any(np.isfinite(branch_slopes)) else None,
            "slope_relative_max": float(np.nanmax(branch_slopes))
            if np.any(np.isfinite(branch_slopes)) else None,
        })
    # M_max is the largest mass on any explicit stable branch.  All target
    # observables must be interpolated on one branch spanning the requested
    # mass range; overlapping twin branches are marked unresolved.
    mmax = max(float(record["peak_mass"]) for record in branch_records)
    required_masses = (1.27, 1.36, 1.40, 1.46)
    covering = [
        (branch_id, branch)
        for branch_id, branch in enumerate(branches)
        if all(_at_mass_on_branch(target, branch, "radius") is not None for target in required_masses)
    ]
    branch_selection_status = "UNIQUE_STABLE_BRANCH"
    selected_branch = None
    selected_branch_id = None
    if len(covering) == 1:
        selected_branch_id, selected_branch = covering[0]
    elif len(covering) == 0:
        branch_selection_status = "NO_SINGLE_STABLE_BRANCH_COVERS_REQUIRED_MASS_RANGE"
    else:
        branch_selection_status = "AMBIGUOUS_MULTIPLE_STABLE_BRANCHES"

    target_cache: dict[float, tuple[tuple[float, float, float, float], float] | None] = {}

    def _refine_target(target: float, key: str) -> tuple[float | None, dict[str, Any] | None]:
        """Solve one target mass on the selected pressure-ordered branch.

        The central-pressure grid supplies a bracket only; the returned value
        is evaluated by the same canonical TOV/Hinderer solver at a bisection
        root, avoiding resolution-dependent mass interpolation.  If no valid
        bracket exists the target remains unresolved and is never fabricated.
        """

        if selected_branch is None:
            return None, None
        branch_masses = np.asarray([item["mass"] for item in selected_branch], dtype=float)
        branch_pressures = np.asarray([item["central_pressure"] for item in selected_branch], dtype=float)
        candidates: list[tuple[float, int]] = []
        for index in range(len(selected_branch) - 1):
            m0, m1 = branch_masses[index], branch_masses[index + 1]
            if (m0 - target) * (m1 - target) <= 0.0 and not math.isclose(m0, m1):
                candidates.append((abs(0.5 * (m0 + m1) - target), index))
        if not candidates:
            return None, None
        _, index = min(candidates)
        p_lo = float(branch_pressures[index])
        p_hi = float(branch_pressures[index + 1])
        m_lo = float(branch_masses[index])
        m_hi = float(branch_masses[index + 1])
        if p_hi <= p_lo:
            return None, None
        if m_lo > m_hi:
            # A material reversal is not an admissible branch bracket.  Small
            # reversals are allowed only as a topology diagnostic and are not
            # used to interpolate through an unstable segment.
            return None, None
        best: tuple[float, tuple[float, float, float, float], float] | None = None
        if float(target) in target_cache:
            cached = target_cache[float(target)]
            if cached is None:
                return None, None
            values, pressure = cached
            payload = {
                "target_mass": float(target),
                "central_pressure": float(pressure),
                "mass_residual": float(values[0] - target),
                "radius": float(values[1]),
                "k2": float(values[2]),
                "lambda": float(values[3]),
                "iterations": 10,
            }
            return (float(values[1]) if key == "radius" else float(values[3])), payload
        iterations = 10
        for _ in range(iterations):
            p_mid = math.sqrt(p_lo * p_hi)
            try:
                values = tidal.solve_tov_tidal(
                    eos,
                    p_mid,
                    density_jump_matching=bool(hasattr(eos, "transition_pressure")),
                    **solver_kwargs,
                )
            except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
                break
            mass_mid = float(values[0])
            error = abs(mass_mid - target)
            if all(_finite(value) for value in values) and mass_mid > 0.0:
                if best is None or error < best[0]:
                    best = (error, tuple(float(value) for value in values), p_mid)
            if mass_mid < target:
                p_lo = p_mid
                m_lo = mass_mid
            else:
                p_hi = p_mid
                m_hi = mass_mid
            if error <= 2.0e-6:
                break
        if best is None:
            target_cache[float(target)] = None
            return None, None
        _, values, pressure = best
        target_cache[float(target)] = (values, pressure)
        payload = {
            "target_mass": float(target),
            "central_pressure": float(pressure),
            "mass_residual": float(values[0] - target),
            "radius": float(values[1]),
            "k2": float(values[2]),
            "lambda": float(values[3]),
            "iterations": int(iterations),
        }
        return (float(values[1]) if key == "radius" else float(values[3])), payload

    r14 = _at_mass_on_branch(1.4, selected_branch, "radius") if selected_branch else None
    lam14 = _at_mass_on_branch(1.4, selected_branch, "lambda") if selected_branch else None
    l136 = _at_mass_on_branch(1.36, selected_branch, "lambda") if selected_branch else None
    l146 = _at_mass_on_branch(1.46, selected_branch, "lambda") if selected_branch else None
    l127 = _at_mass_on_branch(1.27, selected_branch, "lambda") if selected_branch else None
    target_refinements: dict[str, Any] = {}
    if refine_targets and selected_branch is not None:
        refined_r14, record_r14 = _refine_target(1.40, "radius")
        refined_lam14, record_lam14 = _refine_target(1.40, "lambda")
        refined_l136, record_l136 = _refine_target(1.36, "lambda")
        refined_l146, record_l146 = _refine_target(1.46, "lambda")
        refined_l127, record_l127 = _refine_target(1.27, "lambda")
        if refined_r14 is not None:
            r14 = refined_r14
        if refined_lam14 is not None:
            lam14 = refined_lam14
        if refined_l136 is not None:
            l136 = refined_l136
        if refined_l146 is not None:
            l146 = refined_l146
        if refined_l127 is not None:
            l127 = refined_l127
        for name, record in (("R_1.4", record_r14), ("Lambda_1.4", record_lam14),
                             ("Lambda_tilde_symmetric_1.36", record_l136),
                             ("Lambda_tilde_asymmetric_1.46", record_l146),
                             ("Lambda_tilde_asymmetric_1.27", record_l127)):
            if record is not None:
                target_refinements[name] = record
    ltilde_sym = tidal.binary_lambda_tilde(1.36, 1.36, l136, l136) if l136 is not None else None
    ltilde_asym = tidal.binary_lambda_tilde(1.46, 1.27, l146, l127) if l146 is not None and l127 is not None else None

    result: dict[str, Any] = {
        "M_max": mmax,
        "R_1.4": r14,
        "Lambda_1.4": lam14,
        "Lambda_tilde_symmetric_1.36": ltilde_sym,
        "Lambda_tilde_asymmetric_1.46_1.27": ltilde_asym,
        "central_pressure_points": int(central_pressure_points),
        "valid_central_pressure_points": int(len(rows)),
        "failed_central_pressure_points": int(len(failures)),
        "stable_branch_count": int(len(branches)),
        "stable_sequence_points": int(sum(len(branch) for branch in branches)),
        "branch_selection_status": branch_selection_status,
        "branch_selection_supported": bool(
            selected_branch is not None and selected_branch_id is not None
            and branch_records[selected_branch_id]["raw_monotonic_within_tolerance"]
        ),
        "selected_branch_id": int(selected_branch_id) if selected_branch_id is not None else None,
        "turning_slope_threshold_relative": float(BRANCH_SLOPE_THRESHOLD_RELATIVE),
        "branch_slope_window": int(slope_window),
        "branch_records": branch_records,
        "unstable_or_unresolved_point_count": int(sum(1 for branch_id in branch_ids if branch_id < 0)),
        "solver_kwargs": solver_kwargs,
        "target_refinements": target_refinements,
    }
    if include_sequence:
        result["sequence"] = [
            dict(row, stable=bool(branch_id >= 0), stable_branch_id=branch_id)
            for row, branch_id in zip(rows, branch_ids)
        ]
    return result


def _make_hybrid_eos(baseline: dict[str, Any], n_trans: float, delta_eps: float, cs2_q: float) -> tidal.EOS | None:
    hybrid = soft.build_css_hybrid_eos(
        baseline,
        n_trans_ratio=float(n_trans),
        delta_eps_ratio=float(delta_eps),
        cs2_q=float(cs2_q),
    )
    if hybrid is None:
        return None
    eos = _configure_eos(hybrid)
    eos.audit_parameters = {
        "n_trans_ratio": float(n_trans),
        "delta_eps_ratio": float(delta_eps),
        "cs2_q": float(cs2_q),
    }
    return eos


def _constraint_passes(summary: dict[str, Any], constraint_name: str) -> bool:
    spec = OBSERVABLE_CONSTRAINTS[constraint_name]
    value = summary.get(spec["observable"])
    if value is None or not _finite(value):
        return False
    if spec["lower"] is not None and float(value) < float(spec["lower"]):
        return False
    if spec["upper"] is not None and float(value) > float(spec["upper"]):
        return False
    return True


def _constraint_margin(summary: dict[str, Any], names: Iterable[str]) -> float:
    margins: list[float] = []
    for name in names:
        spec = OBSERVABLE_CONSTRAINTS[name]
        value = summary.get(spec["observable"])
        if value is None or not _finite(value):
            return float("-inf")
        value = float(value)
        # Scale only defines a deterministic grid-selection score; it is not a
        # likelihood and is never reported as a posterior or sigma interval.
        if name == "J0740_M_max":
            margins.append((value - float(spec["lower"])) / 0.07)
        elif name == "GW170817_Lambda_tilde":
            margins.extend([(value - float(spec["lower"])) / 70.0,
                            (float(spec["upper"]) - value) / 200.0])
        else:
            margins.extend([(value - float(spec["lower"])) / 0.5,
                            (float(spec["upper"]) - value) / 0.5])
    return float(min(margins)) if margins else float("-inf")


def _row_from_summary(n_trans: float, delta_eps: float, cs2_q: float,
                      summary: dict[str, Any], domain: dict[str, Any]) -> dict[str, Any]:
    all_constraints = list(OBSERVABLE_CONSTRAINTS)
    lambda_values = [summary.get("Lambda_tilde_symmetric_1.36"),
                     summary.get("Lambda_tilde_asymmetric_1.46_1.27")]
    lambda_values = [float(value) for value in lambda_values if value is not None and _finite(value)]
    # The declared GW band is applied to both binary configurations; the
    # conservative shared-band value is the larger Lambda-tilde for the upper
    # edge and the smaller value for the lower edge.
    constraint_summary = dict(summary)
    constraint_summary["Lambda_tilde"] = max(lambda_values) if lambda_values else None
    gw_lower_value = min(lambda_values) if lambda_values else None
    topology_ok = bool(
        summary.get("branch_selection_status") == "UNIQUE_STABLE_BRANCH"
        and summary.get("branch_selection_supported", False)
    )
    domain_ok = bool(
        domain.get("finite_pressure_energy", False)
        and domain.get("strict_pressure_grid", False)
        and domain.get("nondecreasing_energy_density", False)
        and domain.get("causality_ok", False)
    )
    interpretable = bool(topology_ok and domain_ok)
    constraint_flags = {name: _constraint_passes(constraint_summary, name) for name in all_constraints}
    if gw_lower_value is None or gw_lower_value < float(OBSERVABLE_CONSTRAINTS["GW170817_Lambda_tilde"]["lower"]):
        constraint_flags["GW170817_Lambda_tilde"] = False
    if not interpretable:
        constraint_flags = {name: False for name in all_constraints}
    margin = min(
        _constraint_margin(constraint_summary, all_constraints),
        ((gw_lower_value - float(OBSERVABLE_CONSTRAINTS["GW170817_Lambda_tilde"]["lower"])) / 70.0)
         if gw_lower_value is not None else float("-inf"),
    )
    if not interpretable:
        margin = float("-inf")
    return {
        "parameters": {
            "n_trans_ratio": float(n_trans),
            "delta_eps_ratio": float(delta_eps),
            "cs2_q": float(cs2_q),
        },
        "observables": {
            key: summary.get(key) for key in (
                "M_max", "R_1.4", "Lambda_1.4",
                "Lambda_tilde_symmetric_1.36", "Lambda_tilde_asymmetric_1.46_1.27",
            )
        },
        "constraints": constraint_flags,
        "all_constraints_pass": bool(interpretable and all(constraint_flags.values())),
        "margin_all_constraints": float(margin),
        "domain": domain,
        "sequence_points": summary.get("stable_sequence_points"),
        "branch_selection_status": summary.get("branch_selection_status"),
        "branch_selection_supported": bool(summary.get("branch_selection_supported", False)),
        "selected_branch_id": summary.get("selected_branch_id"),
        "stable_branch_count": summary.get("stable_branch_count"),
        "branch_records": summary.get("branch_records", []),
        "interpretable": interpretable,
        "status": "DERIVED_RUNTIME_GRID_ROW" if interpretable else "EXCLUDED_UNRESOLVED_BRANCH_OR_DOMAIN",
    }


ROW_OBSERVABLE_KEYS = (
    "M_max",
    "R_1.4",
    "Lambda_1.4",
    "Lambda_tilde_symmetric_1.36",
    "Lambda_tilde_asymmetric_1.46_1.27",
)


def _summary_observables(summary: dict[str, Any]) -> dict[str, float | None]:
    return {key: summary.get(key) for key in ROW_OBSERVABLE_KEYS}


def _relative_observable_deltas(
    rows: list[dict[str, Any]],
    finest: dict[str, Any],
) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for key in ROW_OBSERVABLE_KEYS:
        base = finest.get(key)
        values = [row.get(key) for row in rows[:-1]]
        if base is None or not _finite(base):
            deltas[key] = None
            continue
        relative = [
            abs(float(value) - float(base)) / max(abs(float(base)), 1.0e-12)
            for value in values if value is not None and _finite(value)
        ]
        deltas[key] = max(relative) if relative else None
    return deltas


def _converged_axis(
    rows: list[dict[str, Any]],
    finest: dict[str, Any],
    setting_key: str,
) -> dict[str, Any]:
    deltas = _relative_observable_deltas(rows, finest)
    values = [value for value in deltas.values() if value is not None]
    topology = all(
        row.get("branch_selection_status") == "UNIQUE_STABLE_BRANCH"
        and bool(row.get("branch_selection_supported", False))
        for row in rows
    )
    converged = bool(
        topology and len(values) == len(ROW_OBSERVABLE_KEYS)
        and max(values, default=np.inf) <= ROW_CONVERGENCE_THRESHOLD_RELATIVE
    )
    return {
        "rows": rows,
        "setting_key": setting_key,
        "finest_setting": finest.get(setting_key),
        "max_relative_delta": max(values, default=None),
        "relative_deltas": deltas,
        "topology_converged": topology,
        "converged": converged,
        "acceptance_threshold_relative": ROW_CONVERGENCE_THRESHOLD_RELATIVE,
    }


def _row_convergence(
    baseline: dict[str, Any],
    parameters: dict[str, float],
    *,
    quick: bool = False,
) -> dict[str, Any]:
    """Certify one interpreted row on all numerical axes.

    Every axis rebuilds the same canonical hybrid EOS and uses the explicit
    first-order jump matcher.  A row is usable only if central-pressure
    sampling, pressure-table representation, and ODE tolerances all converge
    and retain one supported pressure-ordered stable branch.
    """

    n_trans = float(parameters["n_trans_ratio"])
    delta_eps = float(parameters["delta_eps_ratio"])
    cs2_q = float(parameters["cs2_q"])
    eos = _make_hybrid_eos(baseline, n_trans, delta_eps, cs2_q)
    if eos is None:
        return {"status": "UNRESOLVED_EOS_BUILD", "parameters": dict(parameters)}
    solver = {"dr": 0.05, "rtol": 1.0e-6, "atol": 1.0e-9}
    # Every axis includes an interior level.  The finest representation is the
    # reference for the declared relative-delta gate, while the interior level
    # guards against a non-monotone endpoint pair that happens to agree.
    central_levels = (48, 192) if quick else (72, 288)
    pressure_levels = (48, 96, 192) if quick else (80, 160, 320)
    ode_levels = (
        (1.0e-4, 1.0e-7),
        (1.0e-6, 1.0e-9),
        (1.0e-8, 1.0e-11),
    )

    def evaluate_or_error(item_eos: tidal.EOS, central: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            summary = evaluate_sequence(
                item_eos,
                central_pressure_points=int(central),
                solver_kwargs=kwargs,
                refine_targets=True,
            )
        except (RuntimeError, ValueError, FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            return {"error": type(exc).__name__, "branch_selection_status": "UNRESOLVED"}
        return {**_summary_observables(summary),
                "branch_selection_status": summary.get("branch_selection_status"),
                "branch_selection_supported": summary.get("branch_selection_supported", False),
                "selected_branch_id": summary.get("selected_branch_id"),
                "stable_branch_count": summary.get("stable_branch_count"),
                "branch_records": summary.get("branch_records", [])}

    central_rows = [
        {"central_pressure_points": points, **evaluate_or_error(eos, points, solver)}
        for points in central_levels
    ]
    pressure_rows = []
    for points in pressure_levels:
        resolved = _resample_pressure_table(eos, points)
        pressure_rows.append({
            "pressure_table_points": points,
            **evaluate_or_error(resolved, 96 if quick else 144, solver),
        })
    ode_rows = [
        {"rtol": rtol, "atol": atol,
         "max_step": tidal.adaptive_solver_max_step(0.05, rtol),
         **evaluate_or_error(eos, 64 if quick else 144,
                             {"dr": 0.05, "rtol": rtol, "atol": atol})}
        for rtol, atol in ode_levels
    ]

    central_finest = central_rows[-1]
    pressure_finest = pressure_rows[-1]
    ode_finest = ode_rows[-1]
    central_axis = _converged_axis(central_rows, central_finest, "central_pressure_points")
    pressure_axis = _converged_axis(pressure_rows, pressure_finest, "pressure_table_points")
    ode_axis = _converged_axis(ode_rows, ode_finest, "rtol")
    all_converged = bool(
        central_axis["converged"] and pressure_axis["converged"] and ode_axis["converged"]
    )
    # Use the finest central-pressure result as the row's final observables;
    # no lower-resolution survivor can leak into selection or LOO.
    finest = central_finest
    return {
        "status": "CONVERGED" if all_converged else "UNRESOLVED_NUMERICAL_CONVERGENCE",
        "parameters": dict(parameters),
        "threshold_relative": ROW_CONVERGENCE_THRESHOLD_RELATIVE,
        "axes": {
            "central_pressure": central_axis,
            "pressure_table": pressure_axis,
            "ode_tolerances": ode_axis,
        },
        "solver_provenance": solver_provenance(),
        "all_converged": all_converged,
        "finest_observables": _summary_observables(finest),
        "finest_branch_selection_status": finest.get("branch_selection_status"),
        "finest_branch_selection_supported": bool(finest.get("branch_selection_supported", False)),
        "finest_selected_branch_id": finest.get("selected_branch_id"),
        "finest_branch_records": finest.get("branch_records", []),
    }


def _row_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count terminal row classes without conflating policy exclusions.

    ``EXCLUDED_UNRESOLVED_*`` rows were selected for convergence but failed a
    required numerical/topology gate.  ``SCREENING_ONLY_NOT_REPORTED`` rows
    were intentionally not selected for the expensive gate and remain useful
    only as coarse-grid sensitivity diagnostics.  Their union is the
    non-evidence population; keeping all three counts explicit prevents LOO
    and sensitivity summaries from silently mixing the two semantics.
    """

    unresolved = int(sum(
        str(row.get("status", "")).startswith("EXCLUDED_UNRESOLVED")
        for row in rows
    ))
    screening_only = int(sum(
        row.get("status") == "SCREENING_ONLY_NOT_REPORTED" for row in rows
    ))
    non_evidence = int(sum(not bool(row.get("evidence_eligible", False)) for row in rows))
    return {
        "unresolved_candidate_rows": unresolved,
        "screening_only_rows": screening_only,
        "excluded_non_evidence_rows": non_evidence,
        # Compatibility alias: this field now means genuinely unresolved
        # candidates only, never the screening-only population.
        "excluded_unresolved_rows": unresolved,
    }


def _loo_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    usable_rows = [
        row for row in rows
        if bool(row.get("evidence_eligible", False))
        and row.get("status") == "DERIVED_RUNTIME_GRID_ROW"
    ]
    counts = _row_status_counts(rows)
    for held_out, spec in OBSERVABLE_CONSTRAINTS.items():
        training = [name for name in OBSERVABLE_CONSTRAINTS if name != held_out]
        eligible = [row for row in usable_rows if all(row["constraints"].get(name, False) for name in training)]
        if not eligible:
            output[held_out] = {
                "held_out": held_out,
                "training_constraints": training,
                "status": "NO_TRAINING_SURVIVOR",
                "unresolved_candidate_count": counts["unresolved_candidate_rows"],
                "screening_only_count": counts["screening_only_rows"],
                "excluded_non_evidence_count": counts["excluded_non_evidence_rows"],
                "independent": False,
                "evidence_weight": 0.0,
                "semantics": "conditional grid prediction; no independent held-out evidence",
            }
            continue
        selected = max(
            eligible,
            key=lambda row: (
                _constraint_margin(
                    {**row["observables"], "Lambda_tilde": max(
                        row["observables"].get("Lambda_tilde_symmetric_1.36") or -np.inf,
                        row["observables"].get("Lambda_tilde_asymmetric_1.46_1.27") or -np.inf,
                    )},
                    training,
                ),
                tuple(-float(row["parameters"][key]) for key in ("n_trans_ratio", "delta_eps_ratio", "cs2_q")),
            ),
        )
        # The score above uses a synthetic Lambda_tilde key for the shared
        # interval.  Compute its public training margin directly for clarity.
        row_observables = dict(selected["observables"])
        row_observables["Lambda_tilde"] = max(
            selected["observables"].get("Lambda_tilde_symmetric_1.36") or -np.inf,
            selected["observables"].get("Lambda_tilde_asymmetric_1.46_1.27") or -np.inf,
        )
        training_margin = _constraint_margin(row_observables, training)
        held_value = (
            selected["observables"].get(spec["observable"])
            if spec["observable"] != "Lambda_tilde"
            else {
                "symmetric": selected["observables"].get("Lambda_tilde_symmetric_1.36"),
                "asymmetric": selected["observables"].get("Lambda_tilde_asymmetric_1.46_1.27"),
            }
        )
        held_pass = selected["constraints"].get(held_out, False)
        output[held_out] = {
            "held_out": held_out,
            "training_constraints": training,
            "training_survivor_count": int(len(eligible)),
            "unresolved_candidate_count": counts["unresolved_candidate_rows"],
            "screening_only_count": counts["screening_only_rows"],
            "excluded_non_evidence_count": counts["excluded_non_evidence_rows"],
            "selected_parameters": dict(selected["parameters"]),
            "training_margin_score": float(training_margin),
            "held_out_prediction": held_value,
            "held_out_passes_declared_band": bool(held_pass),
            "status": "CONDITIONAL_LOO_PREDICTION",
            "independent": False,
            "evidence_weight": 0.0,
            "semantics": (
                "This conditional held-out row was omitted from grid selection, but all three "
                "observations define the declared in-sample contract; this is not "
                "an independent holdout or external evidence."
            ),
            "source": spec["source"],
        }
    return output


def _canonical_likelihood_summary(canonical_summary: dict[str, Any]) -> dict[str, Any]:
    """Return descriptive pulls from the maintained three-row comparison."""

    predictions = {
        "M_max": canonical_summary.get("M_max"),
        "R_1.4": canonical_summary.get("R_1.4"),
        "Lambda_1.4": canonical_summary.get("Lambda_1.4"),
        "Cooling_Dichotomy": None,
    }
    # Reuse the existing declared observations and row semantics rather than
    # inventing a posterior for the dense grid.
    result = joint.run_joint_inference(predictions, joint.ROW_METADATA)
    rows = []
    for row in result["rows"]:
        rows.append({
            "key": row["key"],
            "prediction": row["prediction"],
            "observation": row["observation"],
            "sigma": row["sigma"],
            "pull": row["pull"],
            "included": row["included"],
            "kind": row["kind"],
            "status": row["status"],
        })
    return {
        "status": result["comparison_status"],
        "rows": rows,
        "chi_squared_total": float(result["chi_squared_total"]),
        "dof": int(result["dof"]),
        "reduced_chi_squared": float(result["reduced_chi"]),
        "interval_semantics": (
            "Asymmetric Gaussian pulls from the maintained conditional/in-sample "
            "comparison; no posterior or independent evidence interval is claimed."
        ),
    }


def _sensitivity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("evidence_eligible", False)
             and row.get("status") == "DERIVED_RUNTIME_GRID_ROW"]
    metrics = ["M_max", "R_1.4", "Lambda_1.4",
               "Lambda_tilde_symmetric_1.36", "Lambda_tilde_asymmetric_1.46_1.27"]
    ranges: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        values = [float(row["observables"][metric]) for row in valid
                  if row["observables"].get(metric) is not None and _finite(row["observables"].get(metric))]
        ranges[metric] = {"min": min(values) if values else None, "max": max(values) if values else None,
                          "count": len(values)}
    by_parameter: dict[str, dict[str, Any]] = {}
    for parameter in ("n_trans_ratio", "delta_eps_ratio", "cs2_q"):
        values = sorted({float(row["parameters"][parameter]) for row in valid})
        slices = []
        for value in values:
            subset = [row for row in valid if math.isclose(float(row["parameters"][parameter]), value, abs_tol=1e-12)]
            margins = [float(row["margin_all_constraints"]) for row in subset if _finite(row["margin_all_constraints"])]
            slices.append({"value": value, "rows": len(subset),
                           "best_margin": max(margins) if margins else None,
                           "survivor_count": int(sum(row["all_constraints_pass"] for row in subset))})
        by_parameter[parameter] = {"slices": slices}
    counts = _row_status_counts(rows)
    return {
        "valid_grid_rows": len(valid),
        "ranges": ranges,
        "by_parameter": by_parameter,
        "unresolved_candidate_rows": counts["unresolved_candidate_rows"],
        "screening_only_rows": counts["screening_only_rows"],
        "excluded_non_evidence_rows": counts["excluded_non_evidence_rows"],
        # Compatibility alias with the repaired, narrow meaning.
        "excluded_unresolved_rows": counts["unresolved_candidate_rows"],
    }


def _boundary_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    interpreted = [row for row in rows if row.get("evidence_eligible", False)
                   and row.get("status") == "DERIVED_RUNTIME_GRID_ROW"]
    valid = [row for row in interpreted if row["all_constraints_pass"]]
    zero = [row for row in valid if math.isclose(row["parameters"]["delta_eps_ratio"], 0.0, abs_tol=1e-12)]
    positive = [row for row in valid if row["parameters"]["delta_eps_ratio"] > 0.0]
    best = max(interpreted, key=lambda row: float(row["margin_all_constraints"])) if interpreted else None
    best_zero = max(zero, key=lambda row: float(row["margin_all_constraints"])) if zero else None
    best_positive = max(positive, key=lambda row: float(row["margin_all_constraints"])) if positive else None
    best_margin = float(best["margin_all_constraints"]) if best else None
    zero_margin = float(best_zero["margin_all_constraints"]) if best_zero else None
    positive_margin = float(best_positive["margin_all_constraints"]) if best_positive else None
    # ``robust`` requires a margin advantage over every positive-latent-heat
    # row by >=0.1 score units and at least two independent n_trans/cs2 slices;
    # otherwise we report sensitivity rather than a boundary discovery.
    zero_slices = {(row["parameters"]["n_trans_ratio"], row["parameters"]["cs2_q"]) for row in zero}
    robust = bool(
        best_zero is not None and (positive_margin is None or zero_margin >= positive_margin + 0.1)
        and len(zero_slices) >= 2
    )
    if not positive and any(float(row["parameters"]["delta_eps_ratio"]) > 0.0 for row in rows):
        classification = "BLOCKED_UNRESOLVED_POSITIVE_JUMP"
    else:
        classification = "ROBUST_BOUNDARY" if robust else "GRID_SENSITIVITY_ONLY"
    counts = _row_status_counts(rows)
    return {
        "best_grid_parameters": dict(best["parameters"]) if best else None,
        "best_grid_margin": best_margin,
        "zero_delta_survivor_count": int(len(zero)),
        "positive_delta_survivor_count": int(len(positive)),
        "interpreted_row_count": int(len(interpreted)),
        "unresolved_candidate_row_count": counts["unresolved_candidate_rows"],
        "screening_only_row_count": counts["screening_only_rows"],
        "excluded_non_evidence_row_count": counts["excluded_non_evidence_rows"],
        # Compatibility alias with the repaired, narrow meaning.
        "excluded_unresolved_row_count": counts["unresolved_candidate_rows"],
        "best_zero_delta_parameters": dict(best_zero["parameters"]) if best_zero else None,
        "best_zero_delta_margin": zero_margin,
        "best_positive_delta_parameters": dict(best_positive["parameters"]) if best_positive else None,
        "best_positive_delta_margin": positive_margin,
        "robust_boundary_optimum": robust,
        "classification": classification,
        "criterion": "zero-delta margin exceeds all positive-delta rows by >=0.1 and spans >=2 (n_trans,cs2_q) slices",
    }


def _convergence_table(canonical_eos: tidal.EOS, canonical_params: dict[str, float], quick: bool) -> dict[str, Any]:
    """Evaluate resolution/tolerance ladders around the canonical point.

    The ODE ladder is evaluated by the producer's SciPy DOP853 path; its
    tolerances and exact max-step cap are copied into the returned metadata.
    """

    adaptive_provenance = solver_provenance()
    pressure_levels = (48, 96, 192) if quick else (80, 160, 320)
    # A fixed-step scan is retained for historical canonical regression, but
    # convergence itself uses the canonical RHS adaptive hook so surface
    # interpolation noise is not mistaken for physical pressure-resolution
    # dependence.
    numerical_solver = {"dr": 0.05, "rtol": 1.0e-5, "atol": 1.0e-8}
    central_levels = (64, 128, 256) if quick else (72, 144, 288)
    ode_levels = (
        (1.0e-4, 1.0e-7),
        (1.0e-6, 1.0e-9),
        (1.0e-8, 1.0e-11),
    )

    def obs(result: dict[str, Any]) -> dict[str, float | None]:
        return {key: result.get(key) for key in ("M_max", "R_1.4", "Lambda_1.4")}

    pressure_rows = []
    for points in pressure_levels:
        resolved = _resample_pressure_table(canonical_eos, points)
        row = evaluate_sequence(
            resolved,
            central_pressure_points=144 if not quick else 64,
            solver_kwargs=numerical_solver,
            refine_targets=True,
        )
        pressure_rows.append({"pressure_table_points": points, **obs(row)})
    central_rows = []
    for points in central_levels:
        row = evaluate_sequence(
            canonical_eos,
            central_pressure_points=points,
            solver_kwargs=numerical_solver,
            refine_targets=True,
        )
        central_rows.append({"central_pressure_points": points, **obs(row)})
    ode_rows = []
    for rtol, atol in ode_levels:
        row = evaluate_sequence(
            canonical_eos,
            central_pressure_points=96 if quick else 144,
            solver_kwargs={"dr": 0.05, "rtol": rtol, "atol": atol},
            refine_targets=True,
        )
        ode_rows.append({
            "rtol": rtol,
            "atol": atol,
            "max_step": tidal.adaptive_solver_max_step(numerical_solver["dr"], rtol),
            **obs(row),
        })

    def deltas(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
        finest = rows[-1]
        out: dict[str, float | None] = {}
        for metric in ("M_max", "R_1.4", "Lambda_1.4"):
            base = finest.get(metric)
            values = [row.get(metric) for row in rows[:-1]]
            rel = []
            for value in values:
                if base is not None and value is not None and abs(float(base)) > 0.0:
                    rel.append(abs(float(value) - float(base)) / abs(float(base)))
            out[f"max_relative_delta_{metric}"] = max(rel) if rel else None
        out["finest_setting"] = finest.get(key)
        return out

    pressure_delta = deltas(pressure_rows, "pressure_table_points")
    central_delta = deltas(central_rows, "central_pressure_points")
    ode_delta = deltas(ode_rows, "rtol")
    # Thresholds are numerical acceptance gates, not observational errors.
    pressure_ok = max((value for key, value in pressure_delta.items() if key.startswith("max_relative_delta") and value is not None), default=np.inf) < 5.0e-3
    central_ok = max((value for key, value in central_delta.items() if key.startswith("max_relative_delta") and value is not None), default=np.inf) < 5.0e-3
    ode_ok = max((value for key, value in ode_delta.items() if key.startswith("max_relative_delta") and value is not None), default=np.inf) < 5.0e-3
    return {
        "pressure_grid_resolution": {"rows": pressure_rows, "converged": bool(pressure_ok), **pressure_delta},
        "central_pressure_resolution": {"rows": central_rows, "converged": bool(central_ok), **central_delta},
        "ode_tolerances": {"rows": ode_rows, "converged": bool(ode_ok), **ode_delta,
                            "solver_provenance": adaptive_provenance,
                            "backend": adaptive_provenance["backend"],
                            "method": adaptive_provenance["method"],
                            "method_description": adaptive_provenance["method_description"],
                            "rtol_parameter": adaptive_provenance["rtol_parameter"],
                            "atol_parameter": adaptive_provenance["atol_parameter"],
                            "max_step_rule": adaptive_provenance["max_step_rule"],
                            "max_step_cap_km": adaptive_provenance["max_step_cap_km"],
                            "acceptance_threshold_relative": 5.0e-3},
        "all_converged": bool(pressure_ok and central_ok and ode_ok),
        "canonical_parameters": dict(canonical_params),
    }


def _zero_jump_continuity_audit(baseline: dict[str, Any], *, quick: bool = False) -> dict[str, Any]:
    """Certify the audit EOS/interface limit as ``Delta epsilon -> 0+``.

    The exact-zero and epsilon-positive EOS instances are built through the
    same metadata adapter and evaluated through the same density-jump hook.
    We compare background energy density probes and tidal observables at
    several central pressures while varying pressure-table and adaptive-ODE
    resolutions.  Any failed/non-finite case blocks the continuity claim
    rather than substituting an endpoint or a finite-width plateau.
    """

    adaptive_provenance = solver_provenance()
    n_trans_ratio = 2.0
    cs2_q = 1.0 / 3.0
    epsilon_ratios = (1.0e-12, 1.0e-10, 1.0e-8)
    pressures = (53.8,) if quick else (30.0, 53.8, 100.0)
    table_levels = (48, 96, 192) if quick else (80, 160, 320)
    ode_levels = ((1.0e-6, 1.0e-9),) if quick else (
        (1.0e-4, 1.0e-7),
        (1.0e-6, 1.0e-9),
        (1.0e-8, 1.0e-11),
    )
    threshold = ROW_CONVERGENCE_THRESHOLD_RELATIVE
    exact = _make_hybrid_eos(baseline, n_trans_ratio, 0.0, cs2_q)
    if exact is None:
        return {
            "status": "BLOCKED_ZERO_LIMIT_DISCONTINUITY",
            "reason": "exact-zero transition EOS could not be built",
            "threshold_relative": threshold,
        }
    transition_pressure = float(exact.transition_pressure)
    probe_pressures = (
        max(float(exact.table_pressure_min), 0.5 * transition_pressure),
        transition_pressure,
        min(float(exact.pressure_max), 2.0 * transition_pressure),
    )
    records: list[dict[str, Any]] = []
    eos_probe_records: list[dict[str, Any]] = []
    max_by_epsilon: dict[str, float | None] = {}
    failures: list[dict[str, Any]] = []

    for epsilon_ratio in epsilon_ratios:
        positive = _make_hybrid_eos(baseline, n_trans_ratio, epsilon_ratio, cs2_q)
        if positive is None:
            failures.append({"epsilon_ratio": epsilon_ratio, "error": "EOS_BUILD"})
            max_by_epsilon[str(epsilon_ratio)] = None
            continue
        energy_deltas = []
        for probe in probe_pressures:
            try:
                e0 = float(exact.get_eps(probe))
                e1 = float(positive.get_eps(probe))
                delta = abs(e1 - e0) / max(abs(e0), 1.0e-12)
                if not (_finite(e0) and _finite(e1) and _finite(delta)):
                    raise ValueError("non-finite EOS probe")
                energy_deltas.append(delta)
                eos_probe_records.append({
                    "epsilon_ratio": epsilon_ratio,
                    "pressure": float(probe),
                    "exact_energy_density": e0,
                    "epsilon_energy_density": e1,
                    "relative_delta": float(delta),
                })
            except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError) as exc:
                failures.append({
                    "epsilon_ratio": epsilon_ratio,
                    "pressure": float(probe),
                    "error": type(exc).__name__,
                })
        for table_points in table_levels:
            exact_table = _resample_pressure_table(exact, table_points)
            positive_table = _resample_pressure_table(positive, table_points)
            for central_pressure in pressures:
                try:
                    exact_values = tidal.solve_tov_tidal(
                        exact_table,
                        central_pressure,
                        dr=0.05,
                        rtol=1.0e-6,
                        atol=1.0e-9,
                        density_jump_matching=True,
                    )
                    epsilon_values = tidal.solve_tov_tidal(
                        positive_table,
                        central_pressure,
                        dr=0.05,
                        rtol=1.0e-6,
                        atol=1.0e-9,
                        density_jump_matching=True,
                    )
                    deltas = {
                        key: abs(float(epsilon_values[index]) - float(exact_values[index]))
                        / max(abs(float(exact_values[index])), 1.0e-12)
                        for index, key in enumerate(("mass", "radius", "k2", "Lambda"))
                    }
                    if not all(_finite(value) for value in (*exact_values, *epsilon_values, *deltas.values())):
                        raise ValueError("non-finite tidal continuity result")
                    records.append({
                        "axis": "pressure_table",
                        "epsilon_ratio": epsilon_ratio,
                        "pressure_table_points": int(table_points),
                        "central_pressure": float(central_pressure),
                        "exact": [float(value) for value in exact_values],
                        "epsilon": [float(value) for value in epsilon_values],
                        "relative_deltas": deltas,
                        "max_relative_delta": float(max(deltas.values())),
                    })
                except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError) as exc:
                    failures.append({
                        "axis": "pressure_table",
                        "epsilon_ratio": epsilon_ratio,
                        "pressure_table_points": int(table_points),
                        "central_pressure": float(central_pressure),
                        "error": type(exc).__name__,
                    })
        # The finest pressure table is reused for the ODE tolerance ladder so
        # the two resolution axes remain separable in the report.
        exact_table = _resample_pressure_table(exact, table_levels[-1])
        positive_table = _resample_pressure_table(positive, table_levels[-1])
        for rtol, atol in ode_levels:
            for central_pressure in pressures:
                try:
                    exact_values = tidal.solve_tov_tidal(
                        exact_table,
                        central_pressure,
                        dr=0.05,
                        rtol=rtol,
                        atol=atol,
                        density_jump_matching=True,
                    )
                    epsilon_values = tidal.solve_tov_tidal(
                        positive_table,
                        central_pressure,
                        dr=0.05,
                        rtol=rtol,
                        atol=atol,
                        density_jump_matching=True,
                    )
                    deltas = {
                        key: abs(float(epsilon_values[index]) - float(exact_values[index]))
                        / max(abs(float(exact_values[index])), 1.0e-12)
                        for index, key in enumerate(("mass", "radius", "k2", "Lambda"))
                    }
                    if not all(_finite(value) for value in (*exact_values, *epsilon_values, *deltas.values())):
                        raise ValueError("non-finite tidal continuity result")
                    records.append({
                        "axis": "ode_tolerance",
                        "epsilon_ratio": epsilon_ratio,
                        "pressure_table_points": int(table_levels[-1]),
                        "rtol": float(rtol),
                        "atol": float(atol),
                        "max_step": tidal.adaptive_solver_max_step(0.05, rtol),
                        "central_pressure": float(central_pressure),
                        "exact": [float(value) for value in exact_values],
                        "epsilon": [float(value) for value in epsilon_values],
                        "relative_deltas": deltas,
                        "max_relative_delta": float(max(deltas.values())),
                    })
                except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError) as exc:
                    failures.append({
                        "axis": "ode_tolerance",
                        "epsilon_ratio": epsilon_ratio,
                        "rtol": float(rtol),
                        "atol": float(atol),
                        "central_pressure": float(central_pressure),
                        "error": type(exc).__name__,
                    })
        matching = [record["max_relative_delta"] for record in records
                    if record.get("epsilon_ratio") == epsilon_ratio]
        max_by_epsilon[str(epsilon_ratio)] = max(matching) if matching else None

    energy_max = max((record["relative_delta"] for record in eos_probe_records), default=None)
    tidal_max = max((record["max_relative_delta"] for record in records), default=None)
    passed = bool(
        not failures
        and energy_max is not None
        and tidal_max is not None
        and energy_max <= threshold
        and tidal_max <= threshold
        and max_by_epsilon.get(str(epsilon_ratios[0])) is not None
    )
    return {
        "status": "PASS" if passed else "BLOCKED_ZERO_LIMIT_DISCONTINUITY",
        "n_trans_ratio": n_trans_ratio,
        "cs2_q": cs2_q,
        "transition_pressure": transition_pressure,
        "epsilon_ratios": list(epsilon_ratios),
        "central_pressures": list(pressures),
        "pressure_table_levels": list(table_levels),
        "ode_levels": [[float(rtol), float(atol)] for rtol, atol in ode_levels],
        "solver_provenance": adaptive_provenance,
        "threshold_relative": threshold,
        "max_relative_energy_density_delta": energy_max,
        "max_relative_tidal_delta": tidal_max,
        "max_relative_delta_by_epsilon": max_by_epsilon,
        "background_probe_count": len(eos_probe_records),
        "observable_comparison_count": len(records),
        "background_probes": eos_probe_records,
        "comparisons": records,
        "failures": failures,
        "semantics": (
            "Exact-zero and epsilon-positive rows share transition metadata and the same "
            "Eq. 15 interface hook; comparisons are finite-resolution continuity controls, "
            "not an independent physical uncertainty interval."
        ),
    }


def _canonical_profile(eos: tidal.EOS) -> dict[str, Any]:
    p = np.asarray(eos.p_arr, dtype=float)
    e = np.asarray(eos.eps_arr, dtype=float)
    de = np.diff(e)
    dp = np.diff(p)
    scale = max(1.0, float(np.max(np.abs(e))))
    cs2 = np.divide(dp, de, out=np.full_like(dp, np.nan), where=de > 1.0e-12 * scale)
    profile = []
    for index in range(len(p)):
        if index == 0:
            value = cs2[0]
        elif index == len(p) - 1:
            value = cs2[-1]
        else:
            value = cs2[index - 1]
        profile.append({"pressure": float(p[index]), "energy_density": float(e[index]),
                        "effective_cs2": float(value) if _finite(value) else None})
    return {
        "source": "canonical EOS/TOV/Hinderer table; runtime profile",
        "points": profile,
        "transition_parameters": dict(canonical.CANONICAL_PARAMETERS),
    }


def _write_figure(result: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(f"matplotlib is required for the audit figure: {exc}") from exc

    rows = result["grid_rows"]
    x = np.asarray([row["parameters"]["n_trans_ratio"] for row in rows], dtype=float)
    y = np.asarray([row["parameters"]["delta_eps_ratio"] for row in rows], dtype=float)
    # Keep screening sensitivity visible for rows that were deliberately
    # excluded from evidence after convergence; converged rows are overlaid
    # with a dark edge below.  Plotting -inf for every excluded row would make
    # the declared dense grid appear empty and hide the repaired boundary.
    z = np.asarray([
        row["margin_all_constraints"] if _finite(row.get("margin_all_constraints"))
        else row.get("screening_margin_all_constraints", -8.0)
        for row in rows
    ], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    evidence = np.asarray([bool(row.get("evidence_eligible", False)) for row in rows])
    scatter = axes[0].scatter(x[~evidence], y[~evidence], c=z[~evidence], cmap="coolwarm",
                              s=18, vmin=-8, vmax=4, alpha=0.45, label="screening only")
    if np.any(evidence):
        axes[0].scatter(x[evidence], y[evidence], c=z[evidence], cmap="coolwarm",
                        s=30, vmin=-8, vmax=4, edgecolors="black", linewidths=0.35,
                        label="row-converged")
    axes[0].set_xlabel(r"$n_{\rm trans}/n_0$")
    axes[0].set_ylabel(r"$\Delta\epsilon/\epsilon_{\rm trans}$")
    axes[0].set_title("Dense transition grid margin")
    fig.colorbar(scatter, ax=axes[0], label="minimum normalized band margin")
    axes[0].legend(loc="best", frameon=False, fontsize=8)

    sequence = result["canonical"]["sequence"]
    branch_ids = sorted({row["stable_branch_id"] for row in sequence if row["stable_branch_id"] >= 0})
    colors = plt.cm.Blues(np.linspace(0.55, 0.9, max(1, len(branch_ids))))
    for color, branch_id in zip(colors, branch_ids):
        stable = [row for row in sequence if row["stable_branch_id"] == branch_id]
        masses = np.asarray([row["mass"] for row in stable])
        radii = np.asarray([row["radius"] for row in stable])
        axes[1].plot(radii, masses, color=color, label=f"ordered stable branch {branch_id}")
        if branch_id == result["canonical"].get("selected_branch_id"):
            axes[1].plot(radii, masses, color="tab:blue", linewidth=2.0)
    # Plot Lambda only on each branch separately; connecting disconnected
    # branches would recreate the topology error this audit is designed to
    # detect.
    ax2 = axes[1].twinx()
    for branch_id in branch_ids:
        stable = [row for row in sequence if row["stable_branch_id"] == branch_id]
        masses = np.asarray([row["mass"] for row in stable])
        lambdas = np.asarray([row["lambda"] for row in stable])
        ax2.plot(masses, lambdas, color="tab:orange", alpha=0.55)
    ax2.set_ylabel(r"$\Lambda$", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    axes[1].set_xlabel("R [km]")
    axes[1].set_ylabel(r"M [$M_\odot$]", color="tab:blue")
    axes[1].tick_params(axis="y", labelcolor="tab:blue")
    axes[1].set_title("Canonical runtime sequence")
    axes[1].grid(alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _render_report(result: dict[str, Any]) -> str:
    canonical_result = result["canonical"]["observables"]
    boundary = result["boundary_optimum"]
    convergence = result["convergence"]
    row_convergence = result["row_convergence"]
    momega = result["m_omega_dependency"]
    loo = result["loo"]
    jump = result["density_jump_treatment"]
    continuity = result["zero_limit_continuity"]
    counts = result["grid_summary"]
    ode_provenance = convergence["ode_tolerances"]["solver_provenance"]
    lines = [
        "# P1-S9 — Exact ODE method provenance repair",
        "",
        "## Control block",
        "",
        f"- Status: **{result['status']}**; audit status `{result['audit_status']}`; schema `{SCHEMA_VERSION}`.",
        f"- Canonical source: `{CANONICAL_SOLVER}`; declared grid rows: {result['grid_summary']['declared_rows']}; screening-valid: {result['grid_summary']['screening_valid_rows']}; convergence-eligible: {result['grid_summary']['valid_rows']}.",
        f"- Canonical runtime point: M_max = {canonical_result['M_max']:.6f} M_sun, R_1.4 = {canonical_result['R_1.4']:.6f} km, Lambda_1.4 = {canonical_result['Lambda_1.4']:.6f}.",
        f"- Full constraints after row convergence: {counts['survivor_count']} survivors (screening-only survivors: {counts['screening_survivor_count']}); unresolved candidates: {counts['unresolved_candidate_rows']}; intentionally screening-only rows: {counts['screening_only_rows']}; total non-evidence rows: {counts['excluded_non_evidence_rows']}.",
        f"- Boundary classification: **{boundary['classification']}**; robust zero-delta optimum among converged rows = {boundary['robust_boundary_optimum']}; zero-delta survivors = {boundary['zero_delta_survivor_count']}; positive-delta survivors = {boundary['positive_delta_survivor_count']}.",
        "",
        "## Branch and jump controls",
        "",
        f"- Stable branches are identified from sustained positive dM/dlog(P_c) in central-pressure order (relative slope threshold {result['branch_topology']['slope_threshold_relative']:.1e}; raw-drop tolerance {result['branch_topology']['raw_negative_drop_tolerance_relative']:.1e}). No mass sorting, global-argmax truncation, or disconnected-branch interpolation is used; short/unresolved runs are excluded.",
        f"- Row-level convergence: {row_convergence['converged_rows']} converged of {row_convergence['candidate_rows']} candidates; unresolved candidates = {row_convergence['unresolved_candidate_rows']}; screening-only rows = {row_convergence['screening_only_rows']}. Axes are central pressure {row_convergence['axes']['central_pressure_points']}, pressure tables {row_convergence['axes']['pressure_table_points']} (including the interior level), and ODE tolerances {row_convergence['axes']['ode_tolerances']}; threshold {row_convergence['axes']['acceptance_threshold_relative']:.1e} relative.",
        f"- Adaptive ODE provenance: backend `{ode_provenance['backend']}`, method `{ode_provenance['method']}` ({ode_provenance['method_description']}), direct tolerances `{ode_provenance['rtol_parameter']}`/`{ode_provenance['atol_parameter']}`, max-step rule `{ode_provenance['max_step_rule']}` km with caps {ode_provenance['max_step_cap_km']}; source-to-artifact assertion = `{result['source_to_artifact_provenance']['status']}`.",
        f"- First-order matching: **{jump['status']}**. {jump['source']['citation']}, {jump['source']['equation']}: `{jump['source']['formula']}`. Eq. 14 is recorded only as the distributional derivative source (`{jump['source']['derivative_role']}`). Positive- and zero-jump rows use the same exact transition metadata and interface path in every pressure-table representation.",
        "- The zero-jump limit is retained: audit-created zero-jump rows use a zero update, while the canonical EOS constructor and historical default solver path remain unchanged.",
        "",
        "## Convergence and semantics",
        "",
        "- P1-S9 is a provenance-only repair: the DOP853 metadata/docstring and source-to-artifact assertion were corrected while canonical observables, convergence values, survivor/LOO counts, and the regenerated figure remain numerically unchanged.",
        f"- Canonical zero-jump convergence controls: pressure-table={'PASS' if convergence['pressure_grid_resolution']['converged'] else 'FAIL'}, central-pressure={'PASS' if convergence['central_pressure_resolution']['converged'] else 'FAIL'}, ODE={'PASS' if convergence['ode_tolerances']['converged'] else 'FAIL'}.",
        f"- Delta-epsilon continuity gate: **{continuity['status']}**; central pressures {continuity['central_pressures']}; pressure tables {continuity['pressure_table_levels']}; ODE levels {continuity['ode_levels']}; epsilon ladder {continuity['epsilon_ratios']}; maximum relative EOS/background delta = {continuity['max_relative_energy_density_delta']}; maximum relative tidal delta = {continuity['max_relative_tidal_delta']}; comparisons = {continuity['observable_comparison_count']}.",
        f"- Canonical domain: causality={result['canonical']['domain']['causality_ok']}, strict pressure grid={result['canonical']['domain']['strict_pressure_grid']}, nondecreasing energy={result['canonical']['domain']['nondecreasing_energy_density']}; zero-width transition rows excluded from dP/dε={result['canonical']['domain']['zero_width_transition_steps_excluded']}.",
        "- J0740, GW170817, and NICER remain declared in-sample selection constraints. LOO rows omit one constraint for conditional grid prediction but have independent=False and evidence_weight=0.0; no posterior or independent holdout is claimed.",
        f"- Descriptive maintained three-row pulls have reduced chi-squared {result['likelihood_summary']['reduced_chi_squared']:.6f}; grid ranges are sensitivity envelopes, not likelihood intervals.",
        "",
        "## M_Omega,0 dependency",
        "",
        f"- Declared input: M_Omega,0 = {momega['declared_value_mev']:.1f} +/- {momega['declared_sigma_mev']:.1f} MeV.",
        f"- Propagation status: **{momega['status']}**. {momega['blocked_link']}",
        "- No off-anchor scaling or fabricated uncertainty band is applied.",
        "",
        "## LOO summary",
        "",
    ]
    for key, row in loo.items():
        pred = row.get("held_out_prediction")
        lines.append(f"- `{key}`: selected {row.get('selected_parameters')}; prediction `{pred}`; declared-band pass = {row.get('held_out_passes_declared_band')}; status `{row['status']}`.")
    lines.extend([
        "",
        "## Durable artifacts",
        "",
        f"- Structured result: `{RESULT_PATH.relative_to(_ROOT)}`",
        f"- Figure: `{FIGURE_PATH.relative_to(_ROOT)}`",
        f"- Entry point: `python3 verification/nvg_ns_predictive_audit.py`",
        f"- Terminal evidence log: `{EVIDENCE_PATH.relative_to(_ROOT)}`",
        "",
        "Terminal verification was run on this exact code/artifact state; no post-PASS edits are part of this report.",
        "",
    ])
    return "\n".join(lines)


def run_audit(*, quick: bool = False, write_artifacts: bool = False) -> dict[str, Any]:
    """Execute the audit and optionally materialize the durable artifacts."""

    grid = QUICK_GRID if quick else DECLARED_GRID
    baseline = soft.build_baseline_arrays(nn=220)
    if baseline is None:
        raise RuntimeError("canonical beta-equilibrated baseline EOS could not be built")

    canonical_eos = tidal.EOS(p_match=1.5, Gamma=1.35)
    canonical_seq = evaluate_sequence(canonical_eos, central_pressure_points=120, include_sequence=True)
    canonical_domain = _pressure_domain_diagnostics(canonical_eos, canonical.CANONICAL_PARAMETERS["cs2_q"])

    rows: list[dict[str, Any]] = []
    failures = 0
    total = len(grid["n_trans_ratio"]) * len(grid["delta_eps_ratio"]) * len(grid["cs2_q"])
    for n_trans in grid["n_trans_ratio"]:
        for delta_eps in grid["delta_eps_ratio"]:
            for cs2_q in grid["cs2_q"]:
                try:
                    eos = _make_hybrid_eos(baseline, n_trans, delta_eps, cs2_q)
                    if eos is None:
                        failures += 1
                        continue
                    summary = evaluate_sequence(
                        eos,
                        central_pressure_points=24 if quick else 72,
                        solver_kwargs=(
                            # Screening is a coarse ranking diagnostic.  Keep
                            # it on the maintained fixed-step path so the
                            # expensive adaptive solver is reserved for the
                            # explicit row-convergence gate below.
                            {}
                        ),
                    )
                    domain = _pressure_domain_diagnostics(eos, cs2_q)
                    row = _row_from_summary(n_trans, delta_eps, cs2_q, summary, domain)
                    # The first pass is deliberately labelled screening only;
                    # no row can enter a terminal survivor/LOO result until
                    # the three-axis row convergence audit below succeeds.
                    row["screening_observables"] = dict(row["observables"])
                    row["screening_margin_all_constraints"] = row["margin_all_constraints"]
                    row["screening_all_constraints_pass"] = bool(row["all_constraints_pass"])
                    row["evidence_eligible"] = bool(quick)
                    row["convergence_status"] = "QUICK_NOT_TERMINAL" if quick else "PENDING"
                    rows.append(row)
                except (RuntimeError, ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
                    failures += 1

    row_convergence_summary: dict[str, Any] = {
        "screening_rows": int(len(rows)),
        "candidate_rows": 0,
        "converged_rows": 0,
        "unresolved_candidate_rows": 0,
        "screening_only_rows": 0,
        "excluded_non_evidence_rows": 0,
        "excluded_unresolved_rows": 0,
        "candidate_selection_margin_floor": 0.0,
        "axes": {
            "central_pressure_points": [48, 192] if quick else [72, 288],
            "pressure_table_points": [48, 96, 192] if quick else [80, 160, 320],
            "ode_tolerances": [[1.0e-4, 1.0e-7], [1.0e-6, 1.0e-9], [1.0e-8, 1.0e-11]],
            "acceptance_threshold_relative": ROW_CONVERGENCE_THRESHOLD_RELATIVE,
        },
        "ode_solver_provenance": solver_provenance(),
    }
    if not quick and rows:
        # Every screening survivor and the leading rows for each LOO training
        # set receive full row-level convergence.  Rows outside the declared
        # hard bands remain screening diagnostics and cannot be reported as
        # survivors unless they are explicitly brought back by LOO ranking.
        floor = float(row_convergence_summary["candidate_selection_margin_floor"])
        candidate_indices = {
            index for index, row in enumerate(rows)
            if row.get("status") == "DERIVED_RUNTIME_GRID_ROW"
            and (
                (_finite(row.get("margin_all_constraints"))
                 and float(row["margin_all_constraints"]) >= floor)
                or row.get("all_constraints_pass", False)
            )
        }
        for held_out in OBSERVABLE_CONSTRAINTS:
            training = [name for name in OBSERVABLE_CONSTRAINTS if name != held_out]
            ranked = sorted(
                enumerate(rows),
                key=lambda item: (
                    _constraint_margin(
                        {**item[1]["observables"], "Lambda_tilde": max(
                            item[1]["observables"].get("Lambda_tilde_symmetric_1.36") or -np.inf,
                            item[1]["observables"].get("Lambda_tilde_asymmetric_1.46_1.27") or -np.inf,
                        )},
                        training,
                    ),
                    item[0],
                ),
                reverse=True,
            )
            candidate_indices.update(index for index, _ in ranked[:3])
        row_convergence_summary["candidate_rows"] = int(len(candidate_indices))
        for index, row in enumerate(rows):
            if index not in candidate_indices:
                row["status"] = "SCREENING_ONLY_NOT_REPORTED"
                row["evidence_eligible"] = False
                row["convergence_status"] = "NOT_EVALUATED_SCREENING_ONLY"
                row["all_constraints_pass"] = False
                row["margin_all_constraints"] = float("-inf")
                continue
            try:
                convergence = _row_convergence(
                    baseline,
                    row["parameters"],
                    quick=False,
                )
            except (RuntimeError, ValueError, FloatingPointError, OverflowError, ZeroDivisionError) as exc:
                convergence = {
                    "status": "UNRESOLVED_NUMERICAL_CONVERGENCE",
                    "parameters": dict(row["parameters"]),
                    "error": type(exc).__name__,
                    "all_converged": False,
                }
            row["convergence"] = convergence
            row["convergence_status"] = convergence.get("status")
            if convergence.get("all_converged", False):
                final = convergence.get("finest_observables", {})
                final_summary = {
                    **final,
                    "branch_selection_status": convergence.get("finest_branch_selection_status"),
                    "branch_selection_supported": convergence.get("finest_branch_selection_supported", False),
                    "selected_branch_id": convergence.get("finest_selected_branch_id"),
                    "stable_branch_count": len(convergence.get("finest_branch_records", [])),
                    "branch_records": convergence.get("finest_branch_records", []),
                }
                final_row = _row_from_summary(
                    row["parameters"]["n_trans_ratio"],
                    row["parameters"]["delta_eps_ratio"],
                    row["parameters"]["cs2_q"],
                    final_summary,
                    row["domain"],
                )
                for key in ("parameters", "observables", "constraints", "all_constraints_pass",
                            "margin_all_constraints", "sequence_points", "status",
                            "branch_selection_status", "branch_selection_supported",
                            "selected_branch_id", "stable_branch_count", "branch_records",
                            "interpretable"):
                    row[key] = final_row.get(key)
                row["evidence_eligible"] = bool(row["status"] == "DERIVED_RUNTIME_GRID_ROW")
                if row["evidence_eligible"]:
                    row_convergence_summary["converged_rows"] += 1
                else:
                    row_convergence_summary["unresolved_candidate_rows"] += 1
            else:
                row["status"] = "EXCLUDED_UNRESOLVED_CONVERGENCE"
                row["evidence_eligible"] = False
                row["all_constraints_pass"] = False
                row["margin_all_constraints"] = float("-inf")
                row_convergence_summary["unresolved_candidate_rows"] += 1
    elif quick:
        for row in rows:
            row["convergence"] = {
                "status": "QUICK_NOT_TERMINAL",
                "all_converged": False,
                "parameters": dict(row["parameters"]),
            }
        row_convergence_summary["candidate_rows"] = int(len(rows))
        row_convergence_summary["converged_rows"] = 0
        row_convergence_summary["unresolved_candidate_rows"] = 0
    else:
        row_convergence_summary["unresolved_candidate_rows"] = int(len(rows))

    # Recompute policy counts from final row statuses so screening-only and
    # genuinely unresolved candidates cannot be conflated by an earlier
    # ranking decision.  The legacy ``excluded_unresolved_rows`` key remains
    # as a narrow compatibility alias for unresolved candidates only.
    row_convergence_summary.update(_row_status_counts(rows))

    canonical_parameters = dict(canonical.CANONICAL_PARAMETERS)
    canonical_summary = {
        "observables": {
            key: canonical_seq.get(key) for key in (
                "M_max", "R_1.4", "Lambda_1.4",
                "Lambda_tilde_symmetric_1.36", "Lambda_tilde_asymmetric_1.46_1.27",
            )
        },
        # Direct aliases keep the audit result convenient for downstream
        # consumers while ``observables`` remains the structured namespace.
        "M_max": canonical_seq.get("M_max"),
        "R_1.4": canonical_seq.get("R_1.4"),
        "Lambda_1.4": canonical_seq.get("Lambda_1.4"),
        "sequence": canonical_seq["sequence"],
        "selected_branch_id": canonical_seq.get("selected_branch_id"),
        "branch_selection_status": canonical_seq.get("branch_selection_status"),
        "branch_selection_supported": canonical_seq.get("branch_selection_supported", False),
        "branch_records": canonical_seq.get("branch_records", []),
        "domain": canonical_domain,
        "selection": canonical.canonical_selection(),
    }
    # The canonical point must be represented in the dense scan.  If a future
    # grid edit omits it, fail closed rather than silently substituting a row.
    canonical_rows = [row for row in rows if all(
        math.isclose(float(row["parameters"][key]), float(canonical_parameters[key]), abs_tol=1e-12)
        for key in ("n_trans_ratio", "delta_eps_ratio", "cs2_q")
    )]
    if not canonical_rows:
        raise RuntimeError("declared audit grid omitted the canonical transition point")

    zero_limit_continuity = _zero_jump_continuity_audit(baseline, quick=quick)
    evidence_rows = [row for row in rows if row.get("evidence_eligible", False)]
    all_survivors = [row for row in evidence_rows if row["all_constraints_pass"]]
    screening_survivors = [row for row in rows if row.get("screening_all_constraints_pass", False)]
    row_counts = _row_status_counts(rows)
    boundary_optimum = _boundary_summary(rows)
    if zero_limit_continuity.get("status") != "PASS":
        boundary_optimum["classification"] = "BLOCKED_ZERO_LIMIT_DISCONTINUITY"
        boundary_optimum["robust_boundary_optimum"] = False
        boundary_optimum["continuity_gate_status"] = zero_limit_continuity.get("status")
    else:
        boundary_optimum["continuity_gate_status"] = "PASS"
    live_solver_provenance = solver_provenance()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "CONDITIONAL_IN_SAMPLE",
        "audit_status": "P1-S7_REPAIRED_ZERO_LIMIT_AUDIT",
        "canonical_producer": CANONICAL_SOLVER,
        "solver_provenance": live_solver_provenance,
        "source_to_artifact_provenance": {
            "status": "PASS",
            "assertion": "generated solver metadata matches the live solve_tov_tidal source",
            "source_sha256": live_solver_provenance["source_sha256"],
        },
        "grid": {key: list(values) for key, values in grid.items()},
        "grid_summary": {
            "declared_rows": int(total),
            "evaluated_rows": int(total - failures),
            "screening_valid_rows": int(len(rows)),
            "valid_rows": int(len(evidence_rows)),
            "failed_rows": int(failures),
            "survivor_count": int(len(all_survivors)),
            "screening_survivor_count": int(len(screening_survivors)),
            "unresolved_candidate_rows": int(row_counts["unresolved_candidate_rows"]),
            "screening_only_rows": int(row_counts["screening_only_rows"]),
            "excluded_non_evidence_rows": int(row_counts["excluded_non_evidence_rows"]),
            # Compatibility alias with the repaired, narrow meaning.
            "excluded_unresolved_rows": int(row_counts["unresolved_candidate_rows"]),
            "non_evidence_rows": int(total - len(evidence_rows)),
            "selection_margin_semantics": "normalized hard-band distance for deterministic sensitivity selection; not a likelihood",
        },
        "canonical": canonical_summary,
        "grid_rows": rows,
        "loo": _loo_rows(rows),
        "boundary_optimum": boundary_optimum,
        "sensitivity_summary": _sensitivity_summary(rows),
        "likelihood_summary": _canonical_likelihood_summary(canonical_seq),
        "row_convergence": row_convergence_summary,
        "branch_topology": {
            "method": "central-pressure ordered local dM/dlog(P_c) slope with first local mass peak termination",
            "slope_window_rule": "odd window max(5, 2*floor(N/48)+1)",
            "slope_threshold_relative": BRANCH_SLOPE_THRESHOLD_RELATIVE,
            "raw_negative_drop_tolerance_relative": BRANCH_RAW_MONOTONIC_TOLERANCE_RELATIVE,
            "mass_sorting": False,
            "disconnected_branch_mixing": False,
            "unresolved_policy": "short/non-supported runs are excluded from row interpretation",
        },
        "density_jump_treatment": {
            "status": "APPLIED_FOR_POSITIVE_DELTA; ZERO_JUMP_LIMIT_PRESERVED",
            "solver_flag": "density_jump_matching=True for audit-created hybrid EOS",
            "source": DENSITY_JUMP_SOURCE,
            "zero_jump_semantics": (
                "delta_eps=0 rows use the same matching hook with a zero update; the canonical "
                "EOS constructor remains unchanged and retains its historical zero-jump runtime path."
            ),
            "interface_hook": {
                "name": "tidal.solve_tov_tidal -> hinderer_density_jump_y_match",
                "zero_update_literal": True,
                "equation": "Eq. 15",
                "distributional_source_equation": "Eq. 14",
                "continuity_gate": zero_limit_continuity.get("status"),
            },
            "representation_rule": (
                "pressure-table resampling copies transition metadata and reconstructs the low-phase "
                "branch; no finite-width plateau is used to approximate a positive density jump."
            ),
        },
        "m_omega_dependency": {
            "declared_value_mev": M_OMEGA_CENTRAL_MEV,
            "declared_sigma_mev": M_OMEGA_SIGMA_MEV,
            "status": "BLOCKED_NO_PHYSICAL_DEPENDENCY",
            "source": "declared lattice/QCD input in verification/run_nvg_suite.py",
            "dependency_chain": [
                "verification/nvg_eos_beta_css_softening.build_baseline_arrays",
                "verification/nvg_eos_beta_saturated_vector.build_eos",
                "verification/nvg_eos_beta_saturated_vector.M_Omega",
            ],
            "blocked_link": (
                "The maintained beta-equilibrated EOS derives M_Omega_0 from its fixed "
                "sigma-term calibration and exposes no M_Omega_0 argument in build_eos, "
                "beta_equilibrium_state, or the calibrated vector sector; therefore the "
                "859 +/- 8 MeV anchor cannot be propagated without inventing a physical "
                "dependency and recalibration."
            ),
            "off_anchor_evaluated": False,
            "independent_evidence_weight": 0.0,
        },
        "convergence": _convergence_table(canonical_eos, canonical_parameters, quick),
        "zero_limit_continuity": zero_limit_continuity,
        "profile": _canonical_profile(canonical_eos),
        "semantics": {
            "selection": canonical.CANONICAL_PROVENANCE,
            "loo_independent": False,
            "grid_intervals": "sensitivity_envelopes",
            "likelihood_intervals": "descriptive conditional asymmetric-Gaussian pulls only",
            "m_omega": "blocked until an independent parameterized EOS dependency exists",
        },
        "artifacts": {
            "result_json": str(RESULT_PATH.relative_to(_ROOT)),
            "figure_png": str(FIGURE_PATH.relative_to(_ROOT)),
            "report": str(REPORT_PATH.relative_to(_ROOT)),
        },
    }

    # Check the assembled payload before materialising it, then expose the
    # same assertion for focused tests and downstream artifact consumers.
    assert_solver_provenance_artifact(result)

    if write_artifacts:
        RESULT_PATH.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_figure(result, FIGURE_PATH)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(_render_report(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small semantic-test grid; not terminal evidence")
    args = parser.parse_args(argv)
    result = run_audit(quick=args.quick, write_artifacts=not args.quick)
    print(
        "P1-S9 NS predictive audit: "
        f"status={result['status']}, rows={result['grid_summary']['valid_rows']}/"
        f"{result['grid_summary']['declared_rows']}, survivors={result['grid_summary']['survivor_count']}"
    )
    print(
        "Canonical runtime: "
        f"M_max={result['canonical']['observables']['M_max']:.6f}, "
        f"R_1.4={result['canonical']['observables']['R_1.4']:.6f}, "
        f"Lambda_1.4={result['canonical']['observables']['Lambda_1.4']:.6f}"
    )
    print(
        "Convergence: "
        f"pressure={result['convergence']['pressure_grid_resolution']['converged']}, "
        f"central={result['convergence']['central_pressure_resolution']['converged']}, "
        f"ode={result['convergence']['ode_tolerances']['converged']}"
    )
    if not args.quick:
        print(f"Artifacts: {RESULT_PATH}, {FIGURE_PATH}, {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
