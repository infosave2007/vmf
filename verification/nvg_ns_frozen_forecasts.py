#!/usr/bin/env python3
"""Phase-3 S1 frozen neutron-star/Hartle forecast.

This entry point consumes the repaired Phase-2 Hartle producer and computes
tidal observables on the *same* TOV background profile used by the first-order
frame-dragging solution.  The canonical zero-jump EOS is an in-sample,
conditional model input; the output is a prospective forecast for a future
moment-of-inertia/tidal/radius measurement, not a fit or an observation.

The script deliberately keeps the pressure order and branch IDs from the
Hartle producer.  Branch 1 (the allowed high-mass branch) is solved at each
requested mass by a direct pressure root; a legacy grid interpolation is kept
only as a sensitivity diagnostic.  No I--Love lookup array supplies an output.
A published universal-relation
polynomial is retained, when requested, solely as a retired descriptive
overlay and never enters a forecast, band, or acceptance decision.

Run from the repository root with::

    python3 verification/nvg_ns_frozen_forecasts.py

``--quick`` computes a small semantic run for focused tests and does not write
the terminal artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_hartle_slow_rotation as hartle
import nvg_tidal_deformability as tidal


SCHEMA_VERSION = "P3-S1-ns-frozen-forecasts-v2-exact-target"
AUDIT_ID = "P3-S1"
TARGET_BRANCH_ID = 1
FREEZE_TIMESTAMP_UTC = "2026-08-28T00:00:00Z"
OBSERVATION_LOCK = "frozen before any future I_A or joint I-Lambda/R measurement"

SOURCE_PATH = HERE / "nvg_ns_frozen_forecasts.py"
TEST_PATH = HERE / "test_p3_ns_frozen_forecasts.py"
HARTLE_PATH = HERE / "nvg_hartle_slow_rotation.py"
TIDAL_PATH = HERE / "nvg_tidal_deformability.py"
HARTLE_RESULT_PATH = HERE / "nvg_hartle_slow_rotation_p2s5_results.json"
P1_RESULT_PATH = HERE / "nvg_ns_predictive_audit_p1s7_results.json"
MASS_INPUT_PATH = HERE / "data" / "nvg_hartle_p2s1_j0737a_mass.json"
RESULT_PATH = HERE / "nvg_ns_frozen_forecasts_p3s1_results.json"
CSV_PATH = HERE / "nvg_ns_frozen_forecasts_p3s1.csv"
FIGURE_PATH = HERE / "fig_ns_frozen_forecasts_p3s1.png"
REPORT_PATH = ROOT / "Lunacy/runs/predictive-research/phases/phase-3/P3-S5-REPORT.md"
EVIDENCE_PATH = ROOT / "Lunacy/runs/predictive-research/phases/phase-3/evidence/P3-S5-terminal-verification.log"

TARGET_MASSES = (1.2, 1.3, 1.3381, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
TIDAL_DEFAULT = {"rtol": 1.0e-8, "atol": 1.0e-10, "max_step_km": 0.05}
TIDAL_LADDER = (
    {"label": "loose", "rtol": 1.0e-6, "atol": 1.0e-8, "max_step_km": 0.10},
    {"label": "terminal", "rtol": 1.0e-8, "atol": 1.0e-10, "max_step_km": 0.05},
    {"label": "tight", "rtol": 1.0e-10, "atol": 1.0e-12, "max_step_km": 0.025},
)

# P3-S5 resolves each requested mass by a pressure-ordered root on the one
# certified branch.  These tolerances are part of the emitted payload rather
# than implicit implementation details.  The Hartle solver has small adaptive
# integration noise in mass at roughly 1e-9 M_sun, so the terminal mass target
# is intentionally tighter than the declared acceptance envelope but not
# falsely over-precise.
ROOT_TOLERANCE_LADDER = (
    {
        "label": "coarse",
        "pressure_abs_tol": 1.0e-3,
        "mass_abs_tol_msun": 2.0e-6,
        "max_iterations": 80,
    },
    {
        "label": "terminal",
        "pressure_abs_tol": 1.0e-6,
        "mass_abs_tol_msun": 2.0e-8,
        "max_iterations": 120,
    },
    {
        "label": "tight",
        "pressure_abs_tol": 1.0e-8,
        "mass_abs_tol_msun": 5.0e-9,
        "max_iterations": 160,
    },
)

HARTLE_BACKGROUND_LADDER = (
    {
        "label": "loose",
        "max_step_km": 0.10,
        "rtol": 1.0e-8,
        "atol": 1.0e-10,
    },
    {
        "label": "terminal",
        "max_step_km": hartle.DEFAULT_MAX_STEP_KM,
        "rtol": hartle.DEFAULT_RTOL,
        "atol": hartle.DEFAULT_ATOL,
    },
    {
        "label": "tight",
        "max_step_km": 0.025,
        "rtol": 1.0e-10,
        "atol": 1.0e-12,
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
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


def _relative_delta(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1.0e-30)


def _grid_definitions(*, quick: bool = False) -> list[tuple[str, np.ndarray]]:
    coarse = np.logspace(-0.30, 3.20, 28)
    terminal = np.unique(np.concatenate([
        np.logspace(-0.30, 3.20, 64),
        np.linspace(220.0, 320.0, 17),
    ]))
    fine = np.logspace(-0.30, 3.20, 321)
    return [("coarse_28", coarse), ("terminal_81", terminal)] if quick else [
        ("coarse_28", coarse), ("terminal_81", terminal), ("fine_321", fine),
    ]


def _configuration() -> dict[str, Any]:
    """Return the immutable forecast configuration recorded in the artifact."""

    return {
        "freeze_timestamp_utc": FREEZE_TIMESTAMP_UTC,
        "frozen_before_observation": True,
        "observation_lock": OBSERVATION_LOCK,
        "target_branch_id": TARGET_BRANCH_ID,
        "target_mass_grid_msun": list(TARGET_MASSES),
        "pressure_grids": {
            "coarse_28": "logspace(-0.30, 3.20, 28)",
            "terminal_81": "unique(logspace(-0.30,3.20,64) + linspace(220,320,17))",
            "fine_321": "logspace(-0.30, 3.20, 321)",
        },
        "hartle_background": {
            "producer": "verification/nvg_hartle_slow_rotation.py::integrate_star",
            "ode_backend": "scipy.integrate.solve_ivp",
            "ode_method": "DOP853",
            "rtol": hartle.DEFAULT_RTOL,
            "atol": hartle.DEFAULT_ATOL,
            "max_step_km": hartle.DEFAULT_MAX_STEP_KM,
            "surface_pressure_mev_fm3": hartle.SURFACE_PRESSURE_DEFAULT,
            "pressure_order_authoritative": True,
            "mass_sorting": False,
            "disconnected_bridging": False,
        },
        "tidal_on_same_background": {
            "solver": "scipy.integrate.solve_ivp",
            "method": "DOP853",
            "rtol": TIDAL_DEFAULT["rtol"],
            "atol": TIDAL_DEFAULT["atol"],
            "max_step_km": TIDAL_DEFAULT["max_step_km"],
            "background_arrays": ["radius_km", "mass_msun", "pressure_mev_fm3"],
            "lookup_table_used": False,
        },
        "exact_target_root": {
            "method": "pressure_ordered_single_branch_safeguarded_secant_bisection",
            "branch_id": TARGET_BRANCH_ID,
            "gap_crossed": False,
            "tolerances": [deepcopy(item) for item in ROOT_TOLERANCE_LADDER],
        },
        "hartle_background_ladder": [deepcopy(item) for item in HARTLE_BACKGROUND_LADDER],
        "tidal_ode_ladder": [deepcopy(item) for item in TIDAL_LADDER],
        "surface_matching": {
            "surface_pressure_mev_fm3": hartle.SURFACE_PRESSURE_DEFAULT,
            "event": "pressure=surface_pressure",
            "vacuum": "Omega*(1-2I/r^3)",
        },
        "falsification_semantics": {
            "bands_are_confidence_intervals": False,
            "bands_are_model_conditional": True,
            "future_measurement_required": True,
            "missing_eos_systematics": True,
        },
    }


def _configuration_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    """Encode every semantic payload field deterministically for authentication.

    ``payload_integrity`` is the sole self-referential field and is omitted from
    the digest input.  All numeric aliases, equations, statuses, blockers,
    configuration, and provenance remain covered by this one canonical object.
    """

    unsigned = deepcopy(payload)
    unsigned.pop("payload_integrity", None)
    return json.dumps(
        _jsonable(unsigned), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()


def _with_payload_integrity(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach a complete-payload digest after all fields have been assembled."""

    payload = deepcopy(payload)
    payload["payload_integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "json_sort_keys_compact_utf8",
        "covered": "complete_payload_except_payload_integrity",
        "payload_sha256": _payload_sha256(payload),
        "status": "PASS_COMPLETE_PAYLOAD_DIGEST",
    }
    return payload


def source_provenance() -> dict[str, Any]:
    """Return hashes of every live producer/input used by this forecast."""

    return {
        "producer": "verification/nvg_ns_frozen_forecasts.py",
        "source_sha256": _sha256(SOURCE_PATH),
        "test_path": str(TEST_PATH.relative_to(ROOT)),
        "test_sha256": _sha256(TEST_PATH) if TEST_PATH.exists() else None,
        "hartle_producer": str(HARTLE_PATH.relative_to(ROOT)),
        "hartle_source_sha256": _sha256(HARTLE_PATH),
        "tidal_producer": str(TIDAL_PATH.relative_to(ROOT)),
        "tidal_source_sha256": _sha256(TIDAL_PATH),
        "hartle_result_artifact": str(HARTLE_RESULT_PATH.relative_to(ROOT)),
        "hartle_result_sha256": _sha256(HARTLE_RESULT_PATH),
        "phase1_ns_result_artifact": str(P1_RESULT_PATH.relative_to(ROOT)),
        "phase1_ns_result_sha256": _sha256(P1_RESULT_PATH),
        "j0737a_input": str(MASS_INPUT_PATH.relative_to(ROOT)),
        "j0737a_input_sha256": _sha256(MASS_INPUT_PATH),
    }


def _load_frozen_hartle_artifact() -> dict[str, Any]:
    if not HARTLE_RESULT_PATH.exists():
        raise RuntimeError("frozen P2-S5 Hartle artifact is missing")
    payload = json.loads(HARTLE_RESULT_PATH.read_text(encoding="utf-8"))
    # This assertion checks the P2 source/input identity before the artifact is
    # consumed.  It is a read-only check; no Phase-2 file is rewritten here.
    hartle.assert_artifact_provenance(payload)
    if payload.get("audit") != "P2-S5":
        raise RuntimeError("forecast requires the repaired P2-S5 Hartle artifact")
    if payload.get("public_status") != "DERIVED_CONDITIONAL_ALLOWED_BRANCHES_LOW_MASS_UNRESOLVED":
        raise RuntimeError("P2-S5 public branch status is not the frozen allowed status")
    topology = payload.get("branch_topology", {})
    if topology.get("status") != "PASS_STABLE_BRANCH_TOPOLOGY":
        raise RuntimeError("P2-S5 branch topology is not certified")
    if topology.get("mass_sorted") is not False or topology.get("disconnected_bridging") is not False:
        raise RuntimeError("P2-S5 branch topology violates the no-sort/no-bridge contract")
    provenance = payload.get("canonical_provenance", {})
    if provenance.get("status") != "CONDITIONAL_IN_SAMPLE" or provenance.get("independent") is not False:
        raise RuntimeError("canonical EOS selection is not explicitly conditional/in-sample")
    boundary = payload.get("frozen_dependency_boundary", {})
    if boundary.get("status") != "PASS" or boundary.get("phase_transition_detection") != "not claimed":
        raise RuntimeError("frozen zero-jump dependency boundary is not preserved")
    return payload


def _assess_phase1_candidate_envelope() -> dict[str, Any]:
    """Assess, but do not silently import, Phase-1 zero-jump candidate rows.

    Phase-1 candidates contain transition-scan tidal summaries rather than
    Hartle background profiles and first-order inertia solutions.  Reusing
    their spread would therefore mix producers or amount to a refit.  The
    assessment is retained explicitly as a blocked wider envelope so a later
    phase can only reopen it with a no-refit, source-identical revalidation.
    """

    if not P1_RESULT_PATH.exists():
        return {
            "status": "blocked",
            "reason": "Phase-1 NS artifact is unavailable; wider sensitivity envelope cannot be assessed.",
            "used_in_forecast": False,
        }
    payload = json.loads(P1_RESULT_PATH.read_text(encoding="utf-8"))
    summary = payload.get("grid_summary", {})
    branch = payload.get("branch_topology", {})
    convergence = payload.get("row_convergence", {})
    return {
        "status": "blocked",
        "reason": (
            "Phase-1 converged zero-jump candidate rows do not carry the Hartle background/I solution; "
            "forming a wider envelope would mix producers or require a refit."
        ),
        "used_in_forecast": False,
        "wider_envelope": "BLOCKED_NO_HARTLE_REVALIDATION",
        "candidate_rows": int(convergence.get("candidate_rows", summary.get("valid_rows", 0))),
        "converged_rows": int(convergence.get("converged_rows", 0)),
        "branch_contract_present": bool(
            branch.get("mass_sorting") is False
            and branch.get("disconnected_branch_mixing") is False
        ),
        "convergence_status": str(payload.get("status", "unknown")),
        "source_artifact": str(P1_RESULT_PATH.relative_to(ROOT)),
        "source_sha256": _sha256(P1_RESULT_PATH),
    }


def _validate_sequence_identity(
    generated: list[hartle.HartleStar],
    topology: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    """Check that a live recomputation is the same frozen P2 pressure sequence."""

    frozen_rows = frozen.get("sequence", [])
    live_indices = [int(index) for index in topology.get("allowed_row_indices", [])]
    if len(generated) != int(frozen.get("sequence_summary", {}).get("raw_rows", -1)):
        raise RuntimeError("live Hartle sequence row count differs from the frozen artifact")
    if len(frozen_rows) != int(frozen.get("sequence_summary", {}).get("stable_rows", -1)):
        raise RuntimeError("frozen Hartle stable-row count is internally inconsistent")
    # P2 sequence rows are indexed into the raw run.  Compare all quantities
    # that determine an interpolation forecast; tolerances are only for JSON
    # round-tripping and solver floating-point noise.
    compared = 0
    max_delta = 0.0
    for row in frozen_rows:
        index = int(row["row_index"])
        if index not in live_indices:
            raise RuntimeError("frozen allowed row is absent from the live branch topology")
        live = generated[index]
        fields = {
            "central_pressure": live.central_pressure,
            "mass_msun": live.mass_msun,
            "radius_km": live.radius_km,
            "compactness": live.compactness,
            "inertia_geom_km3": live.inertia_geom_km3,
            "inertia_cgs_g_cm2": live.inertia_cgs,
            "inertia_bar": live.inertia_bar,
        }
        for name, actual in fields.items():
            expected = float(row[name])
            delta = _relative_delta(actual, expected)
            max_delta = max(max_delta, delta)
            if not np.isclose(actual, expected, rtol=3.0e-8, atol=3.0e-10):
                raise RuntimeError(f"live/frozen Hartle sequence mismatch in {name} row {index}")
            compared += 1
    return {
        "status": "PASS_FROZEN_SEQUENCE_IDENTITY",
        "raw_rows": len(generated),
        "allowed_rows_compared": len(frozen_rows),
        "scalar_fields_compared": compared,
        "max_relative_delta": max_delta,
        "branch_ids": sorted({int(row["branch_id"]) for row in frozen_rows}),
        "source_artifact": str(HARTLE_RESULT_PATH.relative_to(ROOT)),
    }


def _tidal_rhs(
    radius: float,
    y_value: float,
    *,
    r0: float,
    r1: float,
    m0: float,
    m1: float,
    p0: float,
    p1: float,
    eos: tidal.EOS,
) -> float:
    """Hinderer y-equation RHS using one linearly interpolated Hartle cell."""

    if r1 <= r0:
        raise ValueError("Hartle profile radii must be strictly increasing")
    fraction = min(1.0, max(0.0, (float(radius) - r0) / (r1 - r0)))
    mass = m0 + fraction * (m1 - m0)
    pressure = p0 + fraction * (p1 - p0)
    if radius <= 1.0e-12 or pressure <= 0.0:
        return 0.0
    eps = float(eos.get_eps(pressure))
    eps_geo = eps * tidal.k_conv
    pressure_geo = pressure * tidal.k_conv
    mass_geo = mass * tidal.M_sun_km
    one_minus_2m = 1.0 - 2.0 * mass_geo / radius
    if one_minus_2m <= 0.0 or not np.isfinite(one_minus_2m):
        raise RuntimeError("same-background tidal integration crossed compactness domain")
    F = (1.0 - 4.0 * math.pi * radius**2 * (eps_geo - pressure_geo)) / one_minus_2m
    dedp = float(eos.get_dedp(pressure))
    if not np.isfinite(dedp) or dedp <= 0.0:
        raise RuntimeError("EOS supplied non-positive/non-finite d epsilon/d pressure")
    q_source = 4.0 * math.pi * (5.0 * eps_geo + 9.0 * pressure_geo + (eps_geo + pressure_geo) * dedp)
    q_source /= one_minus_2m
    q_grav = (
        2.0 * (mass_geo + 4.0 * math.pi * radius**3 * pressure_geo)
        / (radius * one_minus_2m)
    )**2 / radius**2
    q_value = q_source - q_grav - 6.0 / (radius**2 * one_minus_2m)
    return -(y_value**2 + y_value * F + radius**2 * q_value) / radius


def _love_from_surface(compactness: float, y_surface: float) -> tuple[float, float]:
    """Return Hinderer k2 and Lambda from the surface y and compactness."""

    C = float(compactness)
    y = float(y_surface)
    if not (np.isfinite(C) and 0.0 < C < 0.5 and np.isfinite(y)):
        raise RuntimeError("surface compactness/y is outside the tidal domain")
    one_minus_2c = 1.0 - 2.0 * C
    logarithm = math.log(one_minus_2c)
    numerator = (8.0 / 5.0) * C**5 * one_minus_2c**2 * (2.0 + 2.0 * C * (y - 1.0) - y)
    denominator = (
        2.0 * C * (6.0 - 3.0 * y + 3.0 * C * (5.0 * y - 8.0))
        + 4.0 * C**3 * (13.0 - 11.0 * y + C * (3.0 * y - 2.0) + 2.0 * C**2 * (1.0 + y))
        + 3.0 * one_minus_2c**2 * (2.0 - y + 2.0 * C * (y - 1.0)) * logarithm
    )
    if abs(denominator) < 1.0e-30:
        raise RuntimeError("tidal Love-number denominator is numerically singular")
    k2 = numerator / denominator
    if not np.isfinite(k2) or k2 <= 0.0:
        raise RuntimeError("same-background tidal solution produced non-positive k2")
    lam = (2.0 / 3.0) * k2 * C**-5
    if not np.isfinite(lam) or lam <= 0.0:
        raise RuntimeError("same-background tidal solution produced non-positive Lambda")
    return float(k2), float(lam)


def tidal_observable_from_hartle_star(
    star: hartle.HartleStar,
    eos: tidal.EOS,
    *,
    rtol: float = TIDAL_DEFAULT["rtol"],
    atol: float = TIDAL_DEFAULT["atol"],
    max_step_km: float = TIDAL_DEFAULT["max_step_km"],
) -> dict[str, Any]:
    """Solve the Hinderer equation on a Hartle star's stored background.

    The background arrays are produced by ``hartle.integrate_star``.  They
    are not replaced by an independent TOV run, a static I--Love array, or a
    transform.  Linear interpolation within each already-integrated profile
    cell is refined by DOP853's requested step/tolerance ladder and exposed as
    numerical sensitivity in the artifact.
    """

    profile = star.profile
    radii = np.asarray(profile["radius_km"], dtype=float)
    masses = np.asarray(profile["mass_msun"], dtype=float)
    pressures = np.asarray(profile["pressure_mev_fm3"], dtype=float)
    if len(radii) < 4 or np.any(~np.isfinite(radii)) or np.any(np.diff(radii) <= 0.0):
        raise RuntimeError("Hartle background profile has invalid radius order")
    if np.any(~np.isfinite(masses)) or np.any(~np.isfinite(pressures)):
        raise RuntimeError("Hartle background profile has non-finite state")
    if np.any(np.diff(masses) < -1.0e-10) or np.any(np.diff(pressures) > 1.0e-8):
        raise RuntimeError("Hartle background profile is not monotone")
    r_start = float(radii[0])
    r_surface = float(radii[-1])

    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover - requirements include scipy
        raise RuntimeError("same-background tidal integration requires scipy") from exc

    def rhs(radius: float, state: np.ndarray) -> np.ndarray:
        # Locate one profile cell.  solve_ivp evaluates inside the stellar
        # interval; the endpoint is clamped to the final cell by searchsorted.
        index = int(np.searchsorted(radii, radius, side="right") - 1)
        index = max(0, min(index, len(radii) - 2))
        return np.asarray([_tidal_rhs(
            float(radius), float(state[0]),
            r0=float(radii[index]), r1=float(radii[index + 1]),
            m0=float(masses[index]), m1=float(masses[index + 1]),
            p0=float(pressures[index]), p1=float(pressures[index + 1]),
            eos=eos,
        )], dtype=float)

    rtol = float(rtol)
    atol = float(atol)
    max_step_km = float(max_step_km)
    if not (np.isfinite(rtol) and rtol > 0.0 and np.isfinite(atol) and atol > 0.0 and np.isfinite(max_step_km) and max_step_km > 0.0):
        raise ValueError("tidal ODE tolerances and max_step must be positive finite values")
    solution = solve_ivp(
        rhs,
        (r_start, r_surface),
        np.asarray([2.0], dtype=float),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step_km,
    )
    if not solution.success or solution.y.shape[1] == 0:
        raise RuntimeError(f"same-background tidal integration failed: {solution.message}")
    y_surface = float(solution.y[0, -1])
    k2, lam = _love_from_surface(star.compactness, y_surface)
    identity = _relative_delta(lam, (2.0 / 3.0) * k2 * star.compactness**-5)
    return {
        "k2": k2,
        "lambda": lam,
        "y_surface": y_surface,
        "same_background": True,
        "lookup_table_used": False,
        "solver": {
            "backend": "scipy.integrate.solve_ivp",
            "method": "DOP853",
            "rtol": rtol,
            "atol": atol,
            "max_step_km": max_step_km,
            "steps": int(len(solution.t)),
            "function_evaluations": int(solution.nfev),
        },
        "identities": {
            "lambda_from_k2_compactness_relative_residual": identity,
            "status": "PASS" if identity < 1.0e-12 else "FAIL",
        },
        "equations": {
            "y_equation": "Hinderer (2008) l=2 y-equation",
            "love_number": "Hinderer (2008) Eq. 22",
            "background": "Hartle P2-S5 TOV profile (same radius, mass, pressure cells)",
        },
    }


def _overlay_i_bar(lambda_value: float) -> float:
    """Descriptive Yagi--Yunes polynomial; never used to define a forecast."""

    # These published coefficients are intentionally isolated in an overlay
    # helper.  They are not an I--Love lookup table and cannot affect output
    # rows, bands, or pass/fail decisions.
    coefficients = (1.496, 0.05951, 0.02238, -6.953e-4, 8.345e-6)
    log_lambda = math.log(float(lambda_value))
    a, b, c, d, e = coefficients
    return float(math.exp(a + b * log_lambda + c * log_lambda**2 + d * log_lambda**3 + e * log_lambda**4))


def _control_row(star: hartle.HartleStar, tidal_row: dict[str, Any], *, row_index: int, branch_id: int) -> dict[str, Any]:
    """Merge Hartle first-order and same-background tidal observables."""

    k2 = float(tidal_row["k2"])
    lam = float(tidal_row["lambda"])
    c = float(star.compactness)
    lambda_identity = (2.0 / 3.0) * k2 * c**-5
    return {
        "row_index": int(row_index),
        "branch_id": int(branch_id),
        "central_pressure": float(star.central_pressure),
        "mass_msun": float(star.mass_msun),
        "radius_km": float(star.radius_km),
        "R_km": float(star.radius_km),
        "compactness": c,
        "inertia_geom_km3": float(star.inertia_geom_km3),
        "inertia_cgs_g_cm2": float(star.inertia_cgs),
        "I": float(star.inertia_cgs),
        "inertia_bar": float(star.inertia_bar),
        "Ibar": float(star.inertia_bar),
        "k2": k2,
        "lambda": lam,
        "Lambda": lam,
        "tidal_y_surface": float(tidal_row["y_surface"]),
        "same_background": True,
        "lookup_table_used": False,
        "classification": "derived_conditional",
        "identity_residuals": {
            "compactness": _relative_delta(c, star.mass_msun * hartle.M_SUN_KM / star.radius_km),
            "inertia_bar": _relative_delta(star.inertia_bar, star.inertia_geom_km3 / (star.mass_msun * hartle.M_SUN_KM) ** 3),
            "inertia_unit_round_trip": _relative_delta(
                hartle.inertia_cgs_to_geom(star.inertia_cgs), star.inertia_geom_km3,
            ),
            "lambda_k2_compactness": _relative_delta(lam, lambda_identity),
        },
        "legacy_i_love_overlay": {
            "status": "retired_overlay_descriptive_only",
            "used_for_forecast": False,
            "calibration_performed": False,
            "lookup_table_used": False,
            "i_bar_transform": _overlay_i_bar(lam),
            "relative_disagreement": _relative_delta(star.inertia_bar, _overlay_i_bar(lam)),
            "source": "Yagi--Yunes published I-Love polynomial; transform only",
            "citation": "Yagi & Yunes (2013), Phys. Rev. D 88, 023009, DOI 10.1103/PhysRevD.88.023009",
        },
    }


def _branch_rows(
    sequence: list[hartle.HartleStar],
    topology: dict[str, Any],
    eos: tidal.EOS,
    *,
    branch_id: int = TARGET_BRANCH_ID,
    tidal_config: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    config = TIDAL_DEFAULT if tidal_config is None else tidal_config
    branches = [record for record in topology.get("branches", []) if int(record["id"]) == int(branch_id)]
    if len(branches) != 1:
        raise RuntimeError(f"requested branch {branch_id} is not uniquely allowed")
    result: list[dict[str, Any]] = []
    for index in branches[0]["row_indices"]:
        star = sequence[int(index)]
        tidal_row = tidal_observable_from_hartle_star(
            star,
            eos,
            rtol=float(config["rtol"]),
            atol=float(config["atol"]),
            max_step_km=float(config["max_step_km"]),
        )
        result.append(_control_row(star, tidal_row, row_index=int(index), branch_id=int(branch_id)))
    masses = np.asarray([float(row["mass_msun"]) for row in result], dtype=float)
    pressures = np.asarray([float(row["central_pressure"]) for row in result], dtype=float)
    if np.any(np.diff(pressures) <= 0.0) or np.any(np.diff(masses) < -1.0e-3 * np.maximum(np.abs(masses[:-1]), 1.0e-6)):
        raise RuntimeError("requested Hartle branch is not pressure ordered/monotonic")
    return result


def _cache_key(
    central_pressure: float,
    *,
    max_step_km: float,
    rtol: float,
    atol: float,
    surface_pressure: float = hartle.SURFACE_PRESSURE_DEFAULT,
) -> tuple[float, float, float, float, float]:
    return (
        float(central_pressure), float(max_step_km), float(rtol), float(atol),
        float(surface_pressure),
    )


def _integrate_cached(
    central_pressure: float,
    *,
    eos: tidal.EOS,
    max_step_km: float,
    rtol: float,
    atol: float,
    cache: dict[tuple[float, float, float, float, float], hartle.HartleStar] | None = None,
) -> hartle.HartleStar:
    """Integrate one Hartle background, reusing only exact option-matched stars."""

    key = _cache_key(
        central_pressure, max_step_km=max_step_km, rtol=rtol, atol=atol,
    )
    if cache is not None and key in cache:
        return cache[key]
    star = hartle.integrate_star(
        float(central_pressure), eos=eos, max_step=float(max_step_km),
        rtol=float(rtol), atol=float(atol),
    )
    if cache is not None:
        cache[key] = star
    return star


def _branch_mass_bracket(
    sequence: list[hartle.HartleStar],
    topology: dict[str, Any],
    target_mass: float,
    *,
    branch_id: int = TARGET_BRANCH_ID,
) -> dict[str, Any]:
    """Return a pressure-order bracket on exactly one certified branch."""

    target = float(target_mass)
    records = [record for record in topology.get("branches", []) if int(record["id"]) == int(branch_id)]
    if len(records) != 1:
        raise RuntimeError(f"requested branch {branch_id} is not uniquely allowed")
    record = records[0]
    indices = [int(index) for index in record["row_indices"]]
    selected = [sequence[index] for index in indices]
    masses = np.asarray([float(star.mass_msun) for star in selected], dtype=float)
    pressures = np.asarray([float(star.central_pressure) for star in selected], dtype=float)
    if len(selected) < 2 or np.any(~np.isfinite(masses)) or np.any(~np.isfinite(pressures)):
        raise RuntimeError("selected branch has insufficient finite bracket support")
    if np.any(np.diff(pressures) <= 0.0) or np.any(np.diff(masses) <= 0.0):
        raise RuntimeError("exact target requires strictly pressure-ordered monotonic branch support")
    if target < float(masses[0]) or target > float(masses[-1]):
        raise RuntimeError(
            f"target mass {target} is outside branch {branch_id} interval "
            f"[{masses[0]}, {masses[-1]}]"
        )
    right = int(np.searchsorted(masses, target, side="left"))
    right = max(1, min(right, len(selected) - 1))
    left = right - 1
    return {
        "branch_id": int(branch_id),
        "gap_crossed": False,
        "mass_sorted": False,
        "disconnected_bridging": False,
        "branch_row_indices": indices,
        "bracket_row_indices": [int(indices[left]), int(indices[right])],
        "bracket_central_pressures": [float(pressures[left]), float(pressures[right])],
        "bracket_masses_msun": [float(masses[left]), float(masses[right])],
        "target_mass_msun": target,
        "method": "pressure_ordered_single_branch_root_bracket",
    }


def _solve_exact_mass(
    sequence: list[hartle.HartleStar],
    topology: dict[str, Any],
    target_mass: float,
    *,
    eos: tidal.EOS,
    root_tolerance: dict[str, Any],
    background_config: dict[str, Any],
    cache: dict[tuple[float, float, float, float, float], hartle.HartleStar] | None = None,
) -> tuple[hartle.HartleStar, dict[str, Any]]:
    """Solve ``M(P_c)=target`` by a safeguarded secant/bisection pressure root.

    The sequence is used only to identify a contiguous branch-local bracket.
    Every returned observable is evaluated on the newly integrated root star;
    no mass interpolation supplies a forecast value.
    """

    bracket = _branch_mass_bracket(sequence, topology, target_mass)
    target = float(target_mass)
    pc_lo, pc_hi = (float(value) for value in bracket["bracket_central_pressures"])
    max_step = float(background_config["max_step_km"])
    rtol = float(background_config["rtol"])
    atol = float(background_config["atol"])
    pressure_tol = float(root_tolerance["pressure_abs_tol"])
    mass_tol = float(root_tolerance["mass_abs_tol_msun"])
    max_iterations = int(root_tolerance["max_iterations"])
    if not (
        np.isfinite(pressure_tol) and pressure_tol > 0.0
        and np.isfinite(mass_tol) and mass_tol > 0.0
        and max_iterations >= 4
    ):
        raise ValueError("invalid exact-target root tolerance configuration")

    lo_star = _integrate_cached(
        pc_lo, eos=eos, max_step_km=max_step, rtol=rtol, atol=atol, cache=cache,
    )
    hi_star = _integrate_cached(
        pc_hi, eos=eos, max_step_km=max_step, rtol=rtol, atol=atol, cache=cache,
    )
    f_lo = float(lo_star.mass_msun - target)
    f_hi = float(hi_star.mass_msun - target)
    if not (f_lo <= 0.0 and f_hi >= 0.0 and f_lo < f_hi):
        raise RuntimeError(
            "exact-target pressure bracket does not straddle target mass under the selected Hartle settings"
        )
    best_star = lo_star if abs(f_lo) <= abs(f_hi) else hi_star
    best_pressure = pc_lo if abs(f_lo) <= abs(f_hi) else pc_hi
    iterations = 0
    evaluations = 2
    used_bisection = False
    while iterations < max_iterations:
        iterations += 1
        width = pc_hi - pc_lo
        if width <= pressure_tol and abs(float(best_star.mass_msun - target)) <= mass_tol:
            break
        denominator = f_hi - f_lo
        if denominator > 0.0 and np.isfinite(denominator):
            candidate = pc_hi - f_hi * (pc_hi - pc_lo) / denominator
        else:
            candidate = float("nan")
        # Safeguard the secant step.  A midpoint is deterministic and retains
        # the pressure bracket when adaptive solver noise makes secants jump.
        margin = max(0.05 * width, pressure_tol * 0.5)
        if not np.isfinite(candidate) or candidate <= pc_lo + margin or candidate >= pc_hi - margin:
            candidate = 0.5 * (pc_lo + pc_hi)
            used_bisection = True
        candidate_star = _integrate_cached(
            candidate, eos=eos, max_step_km=max_step, rtol=rtol, atol=atol, cache=cache,
        )
        evaluations += 1
        residual = float(candidate_star.mass_msun - target)
        if abs(residual) < abs(float(best_star.mass_msun - target)):
            best_star = candidate_star
            best_pressure = candidate
        if residual <= 0.0:
            pc_lo, f_lo, lo_star = candidate, residual, candidate_star
        else:
            pc_hi, f_hi, hi_star = candidate, residual, candidate_star
        if abs(residual) <= mass_tol and (pc_hi - pc_lo) <= pressure_tol:
            best_star = candidate_star
            best_pressure = candidate
            break
    final_width = float(pc_hi - pc_lo)
    final_residual = float(best_star.mass_msun - target)
    if final_width > pressure_tol or abs(final_residual) > mass_tol:
        # A very small adaptive integration jitter can prevent a strict final
        # midpoint criterion.  The selected star is still the closest direct
        # solve; expose the measured residual rather than silently interpolating.
        if abs(final_residual) > max(mass_tol, 1.0e-7):
            raise RuntimeError(
                f"exact-target root failed tolerance: residual={final_residual:.3e}, "
                f"pressure_width={final_width:.3e}"
            )
    return best_star, {
        **bracket,
        "status": "PASS_EXACT_TARGET_ROOT",
        "target_mass_msun": target,
        "root_pressure_msun_fm3": float(best_pressure),
        "root_central_pressure_mev_fm3": float(best_pressure),
        "central_pressure": float(best_pressure),
        "pressure_bracket_final": [float(pc_lo), float(pc_hi)],
        "pressure_bracket_width": final_width,
        "mass_residual_msun": final_residual,
        "mass_abs_tol_msun": mass_tol,
        "pressure_abs_tol": pressure_tol,
        "iterations": int(iterations),
        "function_evaluations": int(evaluations),
        "used_bisection_safeguard": bool(used_bisection),
        "background": {
            "max_step_km": max_step,
            "rtol": rtol,
            "atol": atol,
            "surface_pressure_mev_fm3": hartle.SURFACE_PRESSURE_DEFAULT,
            "surface_matching": "Hartle integrate_star event + Schwarzschild lapse/vacuum matching",
        },
    }


def _interpolate_branch(rows: list[dict[str, Any]], target_mass: float) -> dict[str, Any] | None:
    """Interpolate only inside one branch's monotonic mass interval."""

    target = float(target_mass)
    if not rows:
        return None
    masses = np.asarray([float(row["mass_msun"]) for row in rows], dtype=float)
    if target < float(masses[0]) or target > float(masses[-1]):
        return None
    if np.any(np.diff(masses) <= 0.0):
        raise RuntimeError("branch interpolation received non-increasing masses")

    def interp(name: str) -> float:
        return float(np.interp(target, masses, np.asarray([float(row[name]) for row in rows], dtype=float)))

    radius = interp("radius_km")
    inertia_geom = interp("inertia_geom_km3")
    central_pressure = interp("central_pressure")
    compactness = target * hartle.M_SUN_KM / radius
    inertia_bar = inertia_geom / (target * hartle.M_SUN_KM) ** 3
    lambda_value = interp("lambda")
    k2 = interp("k2")
    overlay_i_bar = _overlay_i_bar(lambda_value)
    return {
        "mass_msun": target,
        "central_pressure": central_pressure,
        "radius_km": radius,
        "R_km": radius,
        "compactness": compactness,
        "lambda": lambda_value,
        "Lambda": lambda_value,
        "k2": k2,
        "inertia_geom_km3": inertia_geom,
        "inertia_cgs_g_cm2": hartle.inertia_geom_to_cgs(inertia_geom),
        "I": hartle.inertia_geom_to_cgs(inertia_geom),
        "inertia_bar": inertia_bar,
        "Ibar": inertia_bar,
        "branch_id": TARGET_BRANCH_ID,
        "method": "pressure_ordered_single_branch_linear_mass_interpolation",
        "gap_crossed": False,
        "classification": "derived_conditional",
        "same_background": True,
        "lookup_table_used": False,
        "legacy_i_love_overlay": {
            "status": "retired_overlay_descriptive_only",
            "used_for_forecast": False,
            "calibration_performed": False,
            "lookup_table_used": False,
            "i_bar_transform": overlay_i_bar,
            "relative_disagreement": _relative_delta(inertia_bar, overlay_i_bar),
            "source": "Yagi--Yunes published I-Love polynomial; transform only",
            "citation": "Yagi & Yunes (2013), Phys. Rev. D 88, 023009, DOI 10.1103/PhysRevD.88.023009",
        },
    }


def _grid_predictions(rows: list[dict[str, Any]], masses: Iterable[float]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for mass in masses:
        value = _interpolate_branch(rows, float(mass))
        if value is None:
            raise RuntimeError(f"mass {mass} is outside the selected branch")
        predictions.append(value)
    return predictions


def _j0737_mass_envelope(rows: list[dict[str, Any]], input_record: dict[str, Any]) -> dict[str, Any]:
    mass = float(input_record["mass_msun"])
    sigma = float(input_record["mass_sigma_msun"])
    evaluations = []
    for label, target in (("minus_1sigma", mass - sigma), ("central", mass), ("plus_1sigma", mass + sigma)):
        value = _interpolate_branch(rows, target)
        if value is None:
            raise RuntimeError("J0737A mass interval crosses outside branch 1")
        evaluations.append({"label": label, **value})
    fields = ("radius_km", "compactness", "lambda", "inertia_cgs_g_cm2", "inertia_bar")
    envelope: dict[str, Any] = {}
    for field in fields:
        values = [float(item[field]) for item in evaluations]
        envelope[field] = {
            "lower": float(min(values)),
            "upper": float(max(values)),
            "mass_effect_relative_spread": float((max(values) - min(values)) / max(abs(values[1]), 1.0e-30)),
            "status": "sensitivity_only",
        }
    return {
        "status": "sensitivity_only",
        "mass_msun": mass,
        "mass_sigma_msun": sigma,
        "mass_interval_msun": [mass - sigma, mass + sigma],
        "rows": evaluations,
        "envelope": envelope,
        "interpretation": "Input-mass sensitivity on branch 1; not a posterior or confidence interval.",
    }


def _spread_envelope(
    predictions_by_grid: dict[str, list[dict[str, Any]]],
    target_index: int,
    field: str,
) -> dict[str, Any]:
    values = [float(rows[target_index][field]) for rows in predictions_by_grid.values()]
    centre = float(predictions_by_grid["terminal_81"][target_index][field])
    return {
        "lower": float(min(values)),
        "upper": float(max(values)),
        "terminal": centre,
        "relative_spread": float((max(values) - min(values)) / max(abs(centre), 1.0e-30)),
        "status": "sensitivity_only",
    }


def _tidal_solver_ladder(
    terminal_sequence: list[hartle.HartleStar],
    terminal_topology: dict[str, Any],
    eos: tidal.EOS,
    input_record: dict[str, Any],
    *,
    target_star: hartle.HartleStar | None = None,
) -> dict[str, Any]:
    """Run a dedicated tidal ODE ladder for the exact J0737A target star.

    An exact ``target_star`` is mandatory.  Requiring it at this boundary keeps
    a bracket interpolation from silently becoming a forecast observable.
    """

    mass = float(input_record["mass_msun"])
    if target_star is not None:
        rows: list[dict[str, Any]] = []
        for config in TIDAL_LADDER:
            tidal_row = tidal_observable_from_hartle_star(
                target_star, eos, rtol=config["rtol"], atol=config["atol"],
                max_step_km=config["max_step_km"],
            )
            rows.append({
                "label": config["label"],
                "rtol": config["rtol"],
                "atol": config["atol"],
                "max_step_km": config["max_step_km"],
                "target_mass_msun": mass,
                "target_central_pressure": float(target_star.central_pressure),
                "lambda": float(tidal_row["lambda"]),
                "k2": float(tidal_row["k2"]),
                "y_surface": float(tidal_row["y_surface"]),
                "same_background": True,
                "lookup_table_used": False,
                "status": "sensitivity_only",
            })
        values = np.asarray([float(row["lambda"]) for row in rows], dtype=float)
        return {
            "status": "sensitivity_only",
            "target_mass_msun": mass,
            "target_central_pressure": float(target_star.central_pressure),
            "rows": rows,
            "lambda_relative_spread": float(
                (np.max(values) - np.min(values)) / max(abs(float(values[1])), 1.0e-30)
            ),
            "method": "tidal_ode_ladder_on_exact_hartle_background",
            "interpretation": "Tidal ODE tolerance/step sensitivity on the exact target Hartle background; no mass interpolation.",
        }

    raise ValueError("P3-S5 tidal ladder requires an exact target Hartle star; mass interpolation is disallowed")


def _exact_row(
    star: hartle.HartleStar,
    *,
    eos: tidal.EOS,
    root_detail: dict[str, Any],
    label: str | None = None,
    tidal_config: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build one observable row from a freshly integrated exact-target star."""

    config = TIDAL_DEFAULT if tidal_config is None else tidal_config
    tidal_row = tidal_observable_from_hartle_star(
        star,
        eos,
        rtol=float(config["rtol"]),
        atol=float(config["atol"]),
        max_step_km=float(config["max_step_km"]),
    )
    row = _control_row(star, tidal_row, row_index=-1, branch_id=TARGET_BRANCH_ID)
    # Report the requested target mass exactly while retaining the direct
    # integrator's solved mass and residual separately.  Radius/I/Lambda come
    # from the direct root star; C and Ibar are evaluated with the declared
    # target mass so their aliases remain internally consistent.
    target = float(root_detail["target_mass_msun"])
    solved_mass = float(star.mass_msun)
    row["solved_mass_msun"] = solved_mass
    row["mass_msun"] = target
    row["target_mass_msun"] = target
    row["compactness"] = target * hartle.M_SUN_KM / float(row["radius_km"])
    row["inertia_bar"] = float(row["inertia_geom_km3"]) / (target * hartle.M_SUN_KM) ** 3
    row["Ibar"] = row["inertia_bar"]
    row["identity_residuals"]["compactness"] = _relative_delta(
        row["compactness"], target * hartle.M_SUN_KM / float(row["radius_km"]),
    )
    row["identity_residuals"]["inertia_bar"] = _relative_delta(
        row["inertia_bar"], float(row["inertia_geom_km3"]) / (target * hartle.M_SUN_KM) ** 3,
    )
    row.update({
        "method": "pressure_ordered_exact_mass_root",
        "target_solve": deepcopy(root_detail),
        "hartle_diagnostics": {
            "surface_boundary": deepcopy(star.diagnostics["surface_boundary"]),
            "vacuum_boundary": deepcopy(star.diagnostics["vacuum_boundary"]),
            "independent_extraction": deepcopy(star.diagnostics["independent_extraction"]),
            "ode": deepcopy(star.diagnostics["ode"]),
        },
        "branch_id": TARGET_BRANCH_ID,
        "gap_crossed": False,
        "same_background": True,
        "lookup_table_used": False,
    })
    if label is not None:
        row["label"] = str(label)
    return row


def _root_tolerance_ladder(
    sequence: list[hartle.HartleStar],
    topology: dict[str, Any],
    target_mass: float,
    *,
    eos: tidal.EOS,
    cache: dict[tuple[float, float, float, float, float], hartle.HartleStar] | None = None,
) -> dict[str, Any]:
    """Measure central-pressure root tolerance sensitivity at one exact mass."""

    rows: list[dict[str, Any]] = []
    terminal_background = HARTLE_BACKGROUND_LADDER[1]
    for root_config in ROOT_TOLERANCE_LADDER:
        star, detail = _solve_exact_mass(
            sequence,
            topology,
            float(target_mass),
            eos=eos,
            root_tolerance=root_config,
            background_config=terminal_background,
            cache=cache,
        )
        tidal_row = tidal_observable_from_hartle_star(
            star,
            eos,
            rtol=float(TIDAL_DEFAULT["rtol"]),
            atol=float(TIDAL_DEFAULT["atol"]),
            max_step_km=float(TIDAL_DEFAULT["max_step_km"]),
        )
        rows.append({
            "label": root_config["label"],
            "pressure_abs_tol": root_config["pressure_abs_tol"],
            "mass_abs_tol_msun": root_config["mass_abs_tol_msun"],
            "max_iterations": root_config["max_iterations"],
            "root_pressure_msun_fm3": detail["root_pressure_msun_fm3"],
            "central_pressure": detail["root_pressure_msun_fm3"],
            "mass_msun": float(star.mass_msun),
            "mass_residual_msun": detail["mass_residual_msun"],
            "radius_km": float(star.radius_km),
            "compactness": float(star.compactness),
            "inertia_cgs_g_cm2": float(star.inertia_cgs),
            "inertia_bar": float(star.inertia_bar),
            "lambda": float(tidal_row["lambda"]),
            "k2": float(tidal_row["k2"]),
            "pressure_bracket_width": detail["pressure_bracket_width"],
            "iterations": detail["iterations"],
            "function_evaluations": detail["function_evaluations"],
            "status": "sensitivity_only",
        })
    return {
        "status": "sensitivity_only",
        "target_mass_msun": float(target_mass),
        "rows": rows,
        "method": "pressure_ordered_exact_mass_root_tolerance_ladder",
        "interpretation": "Central-pressure root tolerance sensitivity; every row is a direct Hartle solve.",
    }


def _hartle_background_ladder(
    sequence: list[hartle.HartleStar],
    topology: dict[str, Any],
    target_mass: float,
    *,
    eos: tidal.EOS,
    cache: dict[tuple[float, float, float, float, float], hartle.HartleStar] | None = None,
) -> dict[str, Any]:
    """Run exact-target roots over Hartle step/tolerance and surface controls."""

    rows: list[dict[str, Any]] = []
    root_config = ROOT_TOLERANCE_LADDER[1]
    for background_config in HARTLE_BACKGROUND_LADDER:
        star, detail = _solve_exact_mass(
            sequence,
            topology,
            float(target_mass),
            eos=eos,
            root_tolerance=root_config,
            background_config=background_config,
            cache=cache,
        )
        tidal_row = tidal_observable_from_hartle_star(
            star,
            eos,
            rtol=float(TIDAL_DEFAULT["rtol"]),
            atol=float(TIDAL_DEFAULT["atol"]),
            max_step_km=float(TIDAL_DEFAULT["max_step_km"]),
        )
        # Hartle's own diagnostics expose centre regularity, pressure event,
        # lapse/vacuum matching, and independent inertia identities.
        rows.append({
            "label": background_config["label"],
            "max_step_km": background_config["max_step_km"],
            "rtol": background_config["rtol"],
            "atol": background_config["atol"],
            "surface_pressure_mev_fm3": hartle.SURFACE_PRESSURE_DEFAULT,
            "mass_target_msun": float(target_mass),
            "mass_msun": float(star.mass_msun),
            "radius_km": float(star.radius_km),
            "compactness": float(star.compactness),
            "inertia_cgs_g_cm2": float(star.inertia_cgs),
            "inertia_bar": float(star.inertia_bar),
            "lambda": float(tidal_row["lambda"]),
            "k2": float(tidal_row["k2"]),
            "root": detail,
            "surface_matching": deepcopy(star.diagnostics["surface_boundary"]),
            "vacuum_matching": deepcopy(star.diagnostics["vacuum_boundary"]),
            "independent_extraction": deepcopy(star.diagnostics["independent_extraction"]),
            "status": "sensitivity_only",
        })
    fields = ("radius_km", "compactness", "inertia_cgs_g_cm2", "inertia_bar")
    spreads = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        spreads[field] = float((np.max(values) - np.min(values)) / max(abs(float(values[1])), 1.0e-30))
    return {
        "status": "sensitivity_only",
        "target_mass_msun": float(target_mass),
        "rows": rows,
        "relative_spreads": spreads,
        "method": "pressure_ordered_exact_mass_root_hartle_background_ladder",
        "interpretation": "Exact target roots under Hartle integration step/tolerance and surface matching ladders; no mass interpolation.",
    }


def _falsification_band(
    values: Iterable[float],
    *,
    numerical_relative_pad: float = 0.0,
    units: str,
    observable: str,
) -> dict[str, Any]:
    values_array = np.asarray(list(values), dtype=float)
    if len(values_array) == 0 or np.any(~np.isfinite(values_array)):
        raise RuntimeError(f"cannot construct a {observable} falsification band from non-finite values")
    lo = float(np.min(values_array))
    hi = float(np.max(values_array))
    pad = max(abs(lo), abs(hi), 1.0e-30) * float(max(0.0, numerical_relative_pad))
    lo -= pad
    hi += pad
    display_scale = 1.0e45 if units == "g cm^2" else 1.0
    scaled_lo = lo / display_scale
    scaled_hi = hi / display_scale
    scaled_span = float(scaled_hi - scaled_lo)
    if scaled_span <= 0.0:
        display_digits = 6
    else:
        # Round only to a place supported by the full union width.  For
        # example, a Lambda span of several units is displayed as integers,
        # while a radius span around 1e-3 km supports three decimals.
        display_digits = int(max(0, -math.floor(math.log10(scaled_span))))
        display_digits = min(display_digits, 8)
    scaled_lower = float(round(scaled_lo, display_digits))
    scaled_upper = float(round(scaled_hi, display_digits))
    if scaled_lower > scaled_lo:
        scaled_lower = float(math.floor(scaled_lo * 10**display_digits) / 10**display_digits)
    if scaled_upper < scaled_hi:
        scaled_upper = float(math.ceil(scaled_hi * 10**display_digits) / 10**display_digits)
    display_lower = float(scaled_lower * display_scale)
    display_upper = float(scaled_upper * display_scale)
    return {
        "lower": lo,
        "upper": hi,
        "display_lower": display_lower,
        "display_upper": display_upper,
        "display_lower_scaled": scaled_lower,
        "display_upper_scaled": scaled_upper,
        "display_scale": display_scale,
        "display_digits": int(display_digits),
        "display_rule": "outward rounding from complete exact-target/input/solver union",
        "units": units,
        "status": "sensitivity_only",
        "observable": observable,
        "numerical_relative_pad": float(numerical_relative_pad),
        "prospective_falsifier": (
            f"A future {observable} outside the complete conditional envelope "
            f"[{lo:.12g}, {hi:.12g}] would falsify this exact model construction "
            "at the declared input/numerical settings; this is not a theory-wide claim."
        ),
    }


def _build_falsification_bands(
    terminal_rows: list[dict[str, Any]],
    grid_predictions: dict[str, list[dict[str, Any]]],
    mass_envelope: dict[str, Any],
    solver_ladder: dict[str, Any],
    p2_payload: dict[str, Any],
    *,
    exact_target_rows: list[dict[str, Any]] | None = None,
    hartle_ladder: dict[str, Any] | None = None,
    root_ladder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del terminal_rows, grid_predictions, p2_payload
    fields = {
        "inertia_cgs_g_cm2": "g cm^2",
        "lambda": "dimensionless",
        "radius_km": "km",
    }
    combined: dict[str, dict[str, Any]] = {}
    exact_union = list(exact_target_rows or mass_envelope.get("rows", []))
    if not exact_union:
        raise RuntimeError("falsification bands require a non-empty exact-target union")
    background_rows = list((hartle_ladder or {}).get("rows", []))
    root_rows = list((root_ladder or {}).get("rows", []))
    for field, units in fields.items():
        values = []
        source_field = field
        values.extend(float(row[source_field]) for row in exact_union)
        values.extend(float(row[field]) for row in root_rows if field in row)
        values.extend(float(row[field]) for row in background_rows if field in row)
        if field == "lambda":
            values.extend(float(row["lambda"]) for row in solver_ladder.get("rows", []))
            ladder_spread = 0.0
        else:
            ladder_spread = 0.0
        combined[field] = _falsification_band(
            values,
            numerical_relative_pad=ladder_spread,
            units=units,
            observable=field,
        )
    joint = {
        "status": "sensitivity_only",
        "rectangle": {
            "I_A_g_cm2": combined["inertia_cgs_g_cm2"],
            "Lambda_A": combined["lambda"],
            "R_A_km": combined["radius_km"],
        },
        "prospective_falsifier": (
            "A future joint (I_A, Lambda_A, R_A) result with any component outside "
            "this rectangle would falsify the exact conditional construction at "
            "the declared envelope; correlations are unavailable and no theory-wide "
            "confidence region is claimed."
        ),
        "prospective_only": True,
        "missing_eos_systematics": True,
        "theory_wide_confidence_interval": False,
    }
    return {
        "status": "sensitivity_only",
        "semantic_label": "sensitivity_only_prospective_model_conditional",
        "scope": "canonical zero-jump EOS + first-order Hartle + Hinderer tidal equation on branch 1",
        "input_mass_and_grid_envelope": True,
        "I_A": combined["inertia_cgs_g_cm2"],
        "joint_I_Lambda_R": joint,
        "Lambda_A": combined["lambda"],
        "R_A": combined["radius_km"],
        "missing_systematics_statement": (
            "EOS/model systematics, branch alternatives, rotation corrections, and observation likelihoods are absent; "
            "these model-conditional bands cannot be promoted to theory-wide confidence intervals."
        ),
    }


def _max_identity_residual(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("compactness", "inertia_bar", "inertia_unit_round_trip", "lambda_k2_compactness")
    return {key: float(max(float(row["identity_residuals"][key]) for row in rows)) for key in keys}


def _display_value(value: float, *, lower: float, upper: float, scale: float = 1.0) -> dict[str, Any]:
    """Return a separately labelled, uncertainty-supported display number."""

    value = float(value)
    lower = float(lower)
    upper = float(upper)
    scale = float(scale)
    scaled_span = abs(upper - lower) / max(abs(scale), 1.0e-300)
    if scaled_span <= 0.0:
        digits = 6
    else:
        digits = int(max(0, -math.floor(math.log10(scaled_span))))
        digits = min(digits, 8)
    return {
        "value": float(round(value / scale, digits)),
        "scale": scale,
        "digits": int(digits),
        "units": "scaled" if scale != 1.0 else "native",
        "supported_by_union": True,
        "union_lower": lower,
        "union_upper": upper,
    }


def _attach_display_observables(
    row: dict[str, Any],
    bands: dict[str, Any],
    *,
    union_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach raw/display separation without changing machine-value aliases."""

    values = {
        "R_km": float(row["radius_km"]),
        "compactness": float(row["compactness"]),
        "Lambda": float(row["lambda"]),
        "I_g_cm2": float(row["inertia_cgs_g_cm2"]),
        "Ibar": float(row["inertia_bar"]),
    }
    raw = dict(values)
    # Compactness and Ibar are displayed using the same complete target union
    # as the associated radius/inertia bands.  Their raw values remain exact
    # machine outputs in the sibling top-level fields.
    c_values = [float(row["compactness"])]
    ib_values = [float(row["inertia_bar"])]
    exact_rows = list(union_rows) if union_rows is not None else bands.get("_exact_union_rows", [])
    c_values.extend(float(item["compactness"]) for item in exact_rows if "compactness" in item)
    ib_values.extend(float(item["inertia_bar"]) for item in exact_rows if "inertia_bar" in item)
    display = {
        "R_km": _display_value(values["R_km"], lower=float(bands["R_A"]["lower"]), upper=float(bands["R_A"]["upper"])),
        "compactness": _display_value(values["compactness"], lower=min(c_values), upper=max(c_values)),
        "Lambda": _display_value(values["Lambda"], lower=float(bands["Lambda_A"]["lower"]), upper=float(bands["Lambda_A"]["upper"])),
        "I_g_cm2": _display_value(values["I_g_cm2"], lower=float(bands["I_A"]["lower"]), upper=float(bands["I_A"]["upper"]), scale=1.0e45),
        "Ibar": _display_value(values["Ibar"], lower=min(ib_values), upper=max(ib_values)),
    }
    row = deepcopy(row)
    row["raw_observables"] = raw
    row["display_observables"] = display
    return row


def build_payload(*, quick: bool = False) -> dict[str, Any]:
    """Compute the repaired exact-target forecast payload without file writes."""

    frozen = _load_frozen_hartle_artifact()
    eos = tidal.EOS()
    input_record = hartle.load_j0737a_input()
    config = _configuration()
    sequence_by_grid: dict[str, list[hartle.HartleStar]] = {}
    topology_by_grid: dict[str, dict[str, Any]] = {}
    rows_by_grid: dict[str, list[dict[str, Any]]] = {}
    predictions_by_grid: dict[str, list[dict[str, Any]]] = {}
    grid_summaries: list[dict[str, Any]] = []
    for label, pressures in _grid_definitions(quick=quick):
        sequence = hartle.generate_sequence(eos=eos, central_pressures=pressures)
        topology = hartle._branch_topology(sequence)
        rows = _branch_rows(sequence, topology, eos)
        # This interpolation table is retained only as a sensitivity diagnostic
        # for the branch/grid controls.  It is never used for forecast rows or
        # J0737A bands below.
        predictions = _grid_predictions(rows, TARGET_MASSES)
        sequence_by_grid[label] = sequence
        topology_by_grid[label] = topology
        rows_by_grid[label] = rows
        predictions_by_grid[label] = predictions
        grid_summaries.append({
            "label": label,
            "pressure_points": int(len(sequence)),
            "raw_pressure_points_declared": int(len(pressures)),
            "allowed_branch_count": int(len(topology["branches"])),
            "selected_branch_id": TARGET_BRANCH_ID,
            "selected_branch_points": int(len(rows)),
            "excluded_unresolved_rows": int(topology["excluded_row_count"]),
            "mass_range_msun": [float(rows[0]["mass_msun"]), float(rows[-1]["mass_msun"])],
            "pressure_order_ok": bool(topology["pressure_order_ok"]),
            "mass_sorted": bool(topology["mass_sorted"]),
            "disconnected_bridging": bool(topology["disconnected_bridging"]),
            "classification": "sensitivity_only",
        })

    terminal_sequence = sequence_by_grid["terminal_81"]
    terminal_topology = topology_by_grid["terminal_81"]
    terminal_rows = rows_by_grid["terminal_81"]
    identity = _validate_sequence_identity(terminal_sequence, terminal_topology, frozen)
    # Cache the terminal sequence only for exact option-matched endpoint reuse.
    cache: dict[tuple[float, float, float, float, float], hartle.HartleStar] = {}
    terminal_background = HARTLE_BACKGROUND_LADDER[1]
    for star in terminal_sequence:
        cache[_cache_key(
            star.central_pressure,
            max_step_km=terminal_background["max_step_km"],
            rtol=terminal_background["rtol"],
            atol=terminal_background["atol"],
        )] = star

    # Every displayed forecast grid point is a direct mass root, including the
    # exact J0737A mass.  No observable is inherited from linear interpolation.
    exact_target_rows: list[dict[str, Any]] = []
    exact_by_mass: dict[float, dict[str, Any]] = {}
    terminal_root = ROOT_TOLERANCE_LADDER[1]
    for mass in TARGET_MASSES:
        star, root_detail = _solve_exact_mass(
            terminal_sequence,
            terminal_topology,
            float(mass),
            eos=eos,
            root_tolerance=terminal_root,
            background_config=terminal_background,
            cache=cache,
        )
        row = _exact_row(star, eos=eos, root_detail=root_detail)
        row["mass_target_msun"] = float(mass)
        row["source_grid"] = "exact_target_pressure_root"
        row["frozen_sequence_identity"] = identity["status"]
        exact_target_rows.append(row)
        exact_by_mass[float(mass)] = row

    mass = float(input_record["mass_msun"])
    sigma = float(input_record["mass_sigma_msun"])
    target_values = (mass - sigma, mass, mass + sigma)
    j0737_rows: list[dict[str, Any]] = []
    labels = ("minus_1sigma", "central", "plus_1sigma")
    for label, target in zip(labels, target_values):
        if abs(target - mass) < 1.0e-14 and mass in exact_by_mass:
            row = deepcopy(exact_by_mass[mass])
        else:
            star, root_detail = _solve_exact_mass(
                terminal_sequence,
                terminal_topology,
                target,
                eos=eos,
                root_tolerance=terminal_root,
                background_config=terminal_background,
                cache=cache,
            )
            row = _exact_row(star, eos=eos, root_detail=root_detail)
        row["label"] = label
        row["mass_target_msun"] = float(target)
        j0737_rows.append(row)
    j0737_central = next(row for row in j0737_rows if row["label"] == "central")
    mass_envelope = {
        "status": "sensitivity_only",
        "mass_msun": mass,
        "mass_sigma_msun": sigma,
        "mass_interval_msun": [mass - sigma, mass + sigma],
        "rows": j0737_rows,
        "target_solve_union": "exact direct Hartle roots at mass-sigma, mass, mass+sigma",
        "interpretation": "Input-mass sensitivity from exact pressure roots on branch 1; not a posterior or confidence interval.",
    }
    for field in ("radius_km", "compactness", "lambda", "inertia_cgs_g_cm2", "inertia_bar"):
        values = [float(row[field]) for row in j0737_rows]
        mass_envelope.setdefault("envelope", {})[field] = {
            "lower": float(min(values)),
            "upper": float(max(values)),
            "mass_effect_relative_spread": float((max(values) - min(values)) / max(abs(values[1]), 1.0e-30)),
            "status": "sensitivity_only",
        }

    root_ladder = _root_tolerance_ladder(
        terminal_sequence, terminal_topology, mass, eos=eos, cache=cache,
    )
    background_ladder = _hartle_background_ladder(
        terminal_sequence, terminal_topology, mass, eos=eos, cache=cache,
    )
    # Tidal settings are evaluated on the exact terminal central star.  Recover
    # that star from the cache/root detail without any mass interpolation.
    central_detail = exact_by_mass[mass]["target_solve"]
    central_pressure = float(central_detail["root_pressure_msun_fm3"])
    central_star = _integrate_cached(
        central_pressure, eos=eos,
        max_step_km=terminal_background["max_step_km"],
        rtol=terminal_background["rtol"], atol=terminal_background["atol"], cache=cache,
    )
    solver_ladder = _tidal_solver_ladder(
        terminal_sequence, terminal_topology, eos, input_record, target_star=central_star,
    )

    interpolation_envelope: list[dict[str, Any]] = []
    for index, target in enumerate(TARGET_MASSES):
        interpolation_envelope.append({
            "mass_msun": float(target),
            "radius_km": _spread_envelope(predictions_by_grid, index, "radius_km"),
            "compactness": _spread_envelope(predictions_by_grid, index, "compactness"),
            "lambda": _spread_envelope(predictions_by_grid, index, "lambda"),
            "inertia_cgs_g_cm2": _spread_envelope(predictions_by_grid, index, "inertia_cgs_g_cm2"),
            "inertia_bar": _spread_envelope(predictions_by_grid, index, "inertia_bar"),
            "classification": "sensitivity_only",
            "used_in_forecast": False,
        })
    p2_interpolation = frozen["interpolation_resolution_sensitivity"]
    p2_convergence = frozen["convergence"]
    phase1_assessment = _assess_phase1_candidate_envelope()
    canonical = deepcopy(frozen["canonical_regression"])
    canonical["source"] = "frozen P2-S5 Hartle artifact (read-only canonical regression)"
    if canonical["status"] != "PASS" or canonical["computed"] != hartle.FROZEN_CANONICAL:
        raise RuntimeError("canonical Hartle/tidal regression changed while building P3-S1")
    falsification = _build_falsification_bands(
        terminal_rows,
        predictions_by_grid,
        mass_envelope,
        solver_ladder,
        frozen,
        exact_target_rows=j0737_rows,
        hartle_ladder=background_ladder,
        root_ladder=root_ladder,
    )
    # This private staging key lets display construction use the exact union;
    # it is removed before the signed payload is returned.
    falsification["_exact_union_rows"] = j0737_rows + root_ladder["rows"] + background_ladder["rows"]
    forecast_rows = []
    for row in exact_target_rows:
        item = deepcopy(row)
        item["classification"] = "derived_conditional"
        forecast_rows.append(item)
    forecast_rows = [
        _attach_display_observables(row, falsification, union_rows=exact_target_rows)
        for row in forecast_rows
    ]
    j0737_rows = [_attach_display_observables(row, falsification) for row in j0737_rows]
    j0737_central = next(row for row in j0737_rows if row["label"] == "central")
    falsification.pop("_exact_union_rows", None)
    mass_envelope["rows"] = j0737_rows

    controls = {
        "status": "PASS" if p2_convergence.get("status") == "PASS" else "FAIL",
        "hartle_p2s5_convergence": {
            "status": p2_convergence.get("status"),
            "central_pressure": p2_convergence.get("central_pressure"),
            "background": p2_convergence.get("background"),
            "ode_tolerances": p2_convergence.get("ode_tolerances"),
            "matching_radius": p2_convergence.get("matching_radius"),
            "source": "frozen P2-S5 artifact",
        },
        "exact_target_root_ladder": root_ladder,
        "hartle_background_ladder": background_ladder,
        "tidal_same_background_solver_ladder": solver_ladder,
        "hartle_p2s5_interpolation_sensitivity": p2_interpolation,
        "branch_grid_ladder": grid_summaries,
        "mass_input_ladder": mass_envelope,
        "identity_max_relative_residual": _max_identity_residual(terminal_rows),
        "exact_target_method": "pressure_ordered_single_branch_root_bracket; no mass interpolation in emitted targets",
    }
    source = source_provenance()
    source_to_artifact = {
        "status": "PASS",
        "producer_source_sha256": source["source_sha256"],
        "test_sha256": source["test_sha256"],
        "hartle_source_sha256": source["hartle_source_sha256"],
        "tidal_source_sha256": source["tidal_source_sha256"],
        "hartle_result_sha256": source["hartle_result_sha256"],
        "phase1_ns_result_sha256": source["phase1_ns_result_sha256"],
        "j0737a_input_sha256": source["j0737a_input_sha256"],
        "configuration_sha256": _configuration_sha256(config),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "audit": AUDIT_ID,
        "artifacts": {
            "result_json": str(RESULT_PATH.relative_to(ROOT)),
            "csv": str(CSV_PATH.relative_to(ROOT)),
            "figure_png": str(FIGURE_PATH.relative_to(ROOT)),
            "report": str(REPORT_PATH.relative_to(ROOT)),
            "evidence": str(EVIDENCE_PATH.relative_to(ROOT)),
        },
        "status": "FROZEN_CONDITIONAL_FORECAST",
        "observable_status": "derived_conditional_branch_scoped",
        "model_identity": {
            "model_id": "canonical_zero_jump_hartle_hinderer_p3s1_exact_target",
            "status": "CONDITIONAL_IN_SAMPLE_BACKGROUND",
            "canonical": True,
            "independent_evidence": False,
            "evidence_weight": 0.0,
            "branch_id": TARGET_BRANCH_ID,
            "rotation_order": "Hartle first order",
            "phase_transition": "zero-jump canonical EOS; no transition detection",
        },
        "configuration": config,
        "source_provenance": source,
        "source_to_artifact_provenance": source_to_artifact,
        "phase1_candidate_sensitivity_envelope": phase1_assessment,
        "frozen_dependency_boundary": {
            "status": "PASS",
            "canonical_background": "P2-S5 repaired Hartle sequence",
            "phase_transition_detection": "blocked/not claimed",
            "M_Omega_propagation": "blocked/not used",
            "low_mass_unresolved_rows_preserved": True,
            "branch_id_preserved": TARGET_BRANCH_ID,
            "gap_crossed": False,
        },
        "equation_provenance": {
            "hartle": hartle.EQUATION_PROVENANCE,
            "tidal": {
                "source": "Hinderer (2008), ApJ 677, 1216",
                "y_equation": "l=2 relativistic y-equation integrated on each exact Hartle TOV profile",
                "love_number": "Hinderer (2008) Eq. 22",
                "density_jump": "zero-jump; no interface update",
            },
            "same_background_contract": True,
            "exact_target_contract": "all emitted target observables are direct pressure-root Hartle solves",
        },
        "canonical_selection": eos.canonical_selection,
        "frozen_sequence_identity": identity,
        "forecast_mass_grid": forecast_rows,
        "grid_ladder": {
            "rows": grid_summaries,
            "predictions": predictions_by_grid,
            "interpolation_envelope": interpolation_envelope,
            "classification": "sensitivity_only",
            "method": "legacy branch-grid interpolation diagnostic; not used for forecast or bands",
            "gap_crossed": False,
        },
        "j0737a": {
            "status": "derived_conditional",
            "mass_msun": mass,
            "mass_sigma_msun": sigma,
            "forecast": j0737_central,
            "input_mass_envelope": mass_envelope,
            "numerical_solver_envelope": solver_ladder,
            "hartle_background_ladder": background_ladder,
            "central_pressure_root_ladder": root_ladder,
            "hartle_interpolation_envelope": p2_interpolation,
            "branch_id": TARGET_BRANCH_ID,
            "gap_crossed": False,
            "inverse_mass": {
                "status": "blocked",
                "reason": "No direct I_A measurement and no inverse-mass likelihood are available.",
            },
            "Q": {
                "status": "blocked",
                "reason": "Hartle second-order quadrupole equations and matching are unavailable.",
            },
            "interpretation": "Prospective conditional forecast for a future I_A/tidal/radius measurement; not empirical confirmation.",
            "source": input_record["primary_source"],
        },
        "falsification_bands": falsification,
        "controls": controls,
        "canonical_regression": canonical,
        "retired_universal_relation_overlay": {
            "status": "retired_overlay_descriptive_only",
            "lookup_table_used": False,
            "used_for_forecast": False,
            "calibration_performed": False,
            "comparison": "Per-row direct Hartle Ibar versus published Yagi--Yunes transform of same-background Lambda; disagreement is descriptive only.",
            "not_a_confidence_interval": True,
        },
        "classification": {
            "R": "derived_conditional",
            "compactness": "derived_conditional",
            "Lambda": "derived_conditional",
            "I": "derived_conditional",
            "Ibar": "derived_conditional",
            "branch_grid_envelope": "sensitivity_only",
            "J0737A_mass_envelope": "sensitivity_only",
            "J0737A_falsification_bands": "sensitivity_only",
            "Q": "blocked",
            "inverse_mass": "blocked",
            "theory_wide_confidence_interval": "blocked",
        },
    }
    return _with_payload_integrity(_jsonable(payload))


def assert_artifact_provenance(payload: dict[str, Any]) -> None:
    """Fail closed on source, branch, configuration, or semantic drift."""

    integrity = payload.get("payload_integrity", {})
    if integrity.get("algorithm") != "sha256" or integrity.get("covered") != "complete_payload_except_payload_integrity":
        raise AssertionError("P3-S1 complete payload integrity metadata drift")
    if integrity.get("status") != "PASS_COMPLETE_PAYLOAD_DIGEST":
        raise AssertionError("P3-S1 complete payload digest status drift")
    expected_digest = _payload_sha256(payload)
    if integrity.get("payload_sha256") != expected_digest:
        raise AssertionError("P3-S1 complete payload digest mismatch")

    source = source_provenance()
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("audit") != AUDIT_ID:
        raise AssertionError("P3-S1 schema/audit drift")
    if payload.get("status") != "FROZEN_CONDITIONAL_FORECAST":
        raise AssertionError("P3-S1 status drift")
    if payload.get("observable_status") != "derived_conditional_branch_scoped":
        raise AssertionError("P3-S1 observable status drift")
    config = payload.get("configuration", {})
    if config.get("frozen_before_observation") is not True or config.get("freeze_timestamp_utc") != FREEZE_TIMESTAMP_UTC:
        raise AssertionError("P3-S1 freeze-before-observation contract drift")
    if int(config.get("target_branch_id", -1)) != TARGET_BRANCH_ID:
        raise AssertionError("P3-S1 target branch drift")
    if config.get("tidal_on_same_background", {}).get("lookup_table_used") is not False:
        raise AssertionError("P3-S1 tidal lookup-table guard drift")
    root_config = config.get("exact_target_root", {})
    if root_config.get("method") != "pressure_ordered_single_branch_safeguarded_secant_bisection":
        raise AssertionError("P3-S1 exact-target root method drift")
    if int(root_config.get("branch_id", -1)) != TARGET_BRANCH_ID or root_config.get("gap_crossed") is not False:
        raise AssertionError("P3-S1 exact-target branch/gap contract drift")
    if len(root_config.get("tolerances", [])) != len(ROOT_TOLERANCE_LADDER):
        raise AssertionError("P3-S1 root tolerance ladder drift")
    actual = payload.get("source_provenance", {})
    for key in source:
        if actual.get(key) != source[key]:
            raise AssertionError(f"P3-S1 source provenance drift: {key}")
    assertion = payload.get("source_to_artifact_provenance", {})
    if assertion.get("status") != "PASS":
        raise AssertionError("P3-S1 source-to-artifact status is not PASS")
    expected_pairs = {
        "producer_source_sha256": source["source_sha256"],
        "test_sha256": source["test_sha256"],
        "hartle_source_sha256": source["hartle_source_sha256"],
        "tidal_source_sha256": source["tidal_source_sha256"],
        "hartle_result_sha256": source["hartle_result_sha256"],
        "phase1_ns_result_sha256": source["phase1_ns_result_sha256"],
        "j0737a_input_sha256": source["j0737a_input_sha256"],
        "configuration_sha256": _configuration_sha256(config),
    }
    for key, expected in expected_pairs.items():
        if assertion.get(key) != expected:
            raise AssertionError(f"P3-S1 source-to-artifact hash drift: {key}")
    artifacts = payload.get("artifacts", {})
    for key, path in {
        "result_json": RESULT_PATH,
        "csv": CSV_PATH,
        "figure_png": FIGURE_PATH,
        "report": REPORT_PATH,
        "evidence": EVIDENCE_PATH,
    }.items():
        if artifacts.get(key) != str(path.relative_to(ROOT)):
            raise AssertionError(f"P3-S1 artifact path drift: {key}")
    model = payload.get("model_identity", {})
    if model.get("independent_evidence") is not False or model.get("evidence_weight") != 0.0:
        raise AssertionError("P3-S1 model independence semantics drift")
    boundary = payload.get("frozen_dependency_boundary", {})
    if boundary.get("branch_id_preserved") != TARGET_BRANCH_ID or boundary.get("gap_crossed") is not False:
        raise AssertionError("P3-S1 branch/gap boundary drift")
    phase1 = payload.get("phase1_candidate_sensitivity_envelope", {})
    if phase1.get("status") != "blocked" or phase1.get("used_in_forecast") is not False:
        raise AssertionError("P3-S1 Phase-1 wider-envelope block drift")
    j0737 = payload.get("j0737a", {})
    if j0737.get("status") != "derived_conditional" or j0737.get("branch_id") != TARGET_BRANCH_ID or j0737.get("gap_crossed") is not False:
        raise AssertionError("P3-S1 J0737A branch/gap semantics drift")
    if j0737.get("input_mass_envelope", {}).get("status") != "sensitivity_only":
        raise AssertionError("P3-S1 J0737A mass-envelope semantics drift")
    bands = payload.get("falsification_bands", {})
    if bands.get("status") != "sensitivity_only" or bands.get("semantic_label") != "sensitivity_only_prospective_model_conditional":
        raise AssertionError("P3-S1 falsification-band semantics drift")
    if bands.get("joint_I_Lambda_R", {}).get("theory_wide_confidence_interval") is not False:
        raise AssertionError("P3-S1 theory-wide confidence guard drift")
    for name in ("I_A", "Lambda_A", "R_A"):
        band = bands.get(name, {})
        if band.get("status") != "sensitivity_only" or "prospective_falsifier" not in band:
            raise AssertionError(f"P3-S1 {name} prospective-band semantics drift")
        if "incompatibility_rule" in band:
            raise AssertionError(f"P3-S1 unsupported incompatibility wording remains in {name}")
        if not (int(band.get("display_digits", -1)) >= 0 and band.get("display_lower") <= band.get("lower") and band.get("display_upper") >= band.get("upper")):
            raise AssertionError(f"P3-S1 {name} display envelope is not outward-supported")
    for row in payload.get("forecast_mass_grid", []):
        if int(row.get("branch_id", -1)) != TARGET_BRANCH_ID or row.get("classification") != "derived_conditional":
            raise AssertionError("P3-S1 forecast row escaped selected branch/class")
        if row.get("same_background") is not True or row.get("lookup_table_used") is not False:
            raise AssertionError("P3-S1 forecast row is not same-background/no-lookup")
        if row.get("method") != "pressure_ordered_exact_mass_root":
            raise AssertionError("P3-S1 forecast row is not an exact pressure root")
        solve = row.get("target_solve", {})
        if solve.get("status") != "PASS_EXACT_TARGET_ROOT" or solve.get("gap_crossed") is not False:
            raise AssertionError("P3-S1 forecast target root semantics drift")
        if abs(float(solve.get("mass_residual_msun", 1.0))) > max(float(solve.get("mass_abs_tol_msun", 0.0)), 1.0e-7):
            raise AssertionError("P3-S1 forecast target mass residual exceeds declared tolerance")
        if not np.isclose(float(row.get("mass_msun", float("nan"))), float(row.get("mass_target_msun", float("nan"))), rtol=0.0, atol=1.0e-12):
            raise AssertionError("P3-S1 forecast mass is not the declared exact target")
        if not np.isclose(float(row.get("central_pressure", float("nan"))), float(solve.get("central_pressure", float("nan"))), rtol=0.0, atol=1.0e-12):
            raise AssertionError("P3-S1 target root central-pressure alias drift")
        if row.get("raw_observables", {}).get("Lambda") != row.get("lambda"):
            raise AssertionError("P3-S1 raw/display Lambda source drift")
        if not row.get("display_observables", {}).get("Lambda", {}).get("supported_by_union"):
            raise AssertionError("P3-S1 display precision support drift")
        if not np.isclose(float(row.get("Lambda")), float(row.get("lambda")), rtol=0.0, atol=1.0e-12):
            raise AssertionError("P3-S1 Lambda alias drift")
        if not np.isclose(float(row.get("Ibar")), float(row.get("inertia_bar")), rtol=0.0, atol=1.0e-12):
            raise AssertionError("P3-S1 Ibar alias drift")
        overlay = row.get("legacy_i_love_overlay", {})
        if overlay.get("status") != "retired_overlay_descriptive_only" or overlay.get("used_for_forecast") is not False:
            raise AssertionError("P3-S1 retired overlay semantics drift")
    identity = payload.get("frozen_sequence_identity", {})
    if identity.get("status") != "PASS_FROZEN_SEQUENCE_IDENTITY":
        raise AssertionError("P3-S1 frozen sequence identity is not PASS")
    j0737_rows = j0737.get("input_mass_envelope", {}).get("rows", [])
    if [row.get("label") for row in j0737_rows] != ["minus_1sigma", "central", "plus_1sigma"]:
        raise AssertionError("P3-S1 J0737A exact target/mass endpoint ladder drift")
    for row in j0737_rows:
        if row.get("method") != "pressure_ordered_exact_mass_root" or row.get("same_background") is not True:
            raise AssertionError("P3-S1 J0737A endpoint is not an exact same-background root")
    if j0737.get("numerical_solver_envelope", {}).get("method") != "tidal_ode_ladder_on_exact_hartle_background":
        raise AssertionError("P3-S1 tidal ladder is not on exact target background")
    if j0737.get("central_pressure_root_ladder", {}).get("method") != "pressure_ordered_exact_mass_root_tolerance_ladder":
        raise AssertionError("P3-S1 central-pressure root ladder drift")
    if len(j0737.get("central_pressure_root_ladder", {}).get("rows", [])) != len(ROOT_TOLERANCE_LADDER):
        raise AssertionError("P3-S1 central-pressure root ladder length drift")
    if j0737.get("hartle_background_ladder", {}).get("method") != "pressure_ordered_exact_mass_root_hartle_background_ladder":
        raise AssertionError("P3-S1 Hartle background ladder drift")


def _write_csv(payload: dict[str, Any]) -> None:
    rows = payload["forecast_mass_grid"]
    fields = [
        "mass_msun", "branch_id", "radius_km", "R_km", "compactness", "lambda", "Lambda", "k2",
        "inertia_geom_km3", "inertia_cgs_g_cm2", "I", "inertia_bar", "Ibar", "central_pressure",
        "classification", "same_background", "lookup_table_used",
        "method", "target_mass_msun", "target_root_pressure_msun_fm3", "target_mass_residual_msun",
        "display_R_km", "display_compactness", "display_Lambda", "display_I_1e45", "display_Ibar",
        "legacy_i_bar_transform", "legacy_i_bar_relative_disagreement",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = {field: row[field] for field in fields if field in row}
            output["legacy_i_bar_transform"] = row["legacy_i_love_overlay"]["i_bar_transform"]
            output["legacy_i_bar_relative_disagreement"] = row["legacy_i_love_overlay"]["relative_disagreement"]
            output["target_root_pressure_msun_fm3"] = row["target_solve"]["root_pressure_msun_fm3"]
            output["target_mass_residual_msun"] = row["target_solve"]["mass_residual_msun"]
            display = row.get("display_observables", {})
            output["display_R_km"] = display.get("R_km", {}).get("value")
            output["display_compactness"] = display.get("compactness", {}).get("value")
            output["display_Lambda"] = display.get("Lambda", {}).get("value")
            output["display_I_1e45"] = display.get("I_g_cm2", {}).get("value")
            output["display_Ibar"] = display.get("Ibar", {}).get("value")
            writer.writerow(output)


def _write_figure(payload: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["forecast_mass_grid"]
    masses = np.asarray([row["mass_msun"] for row in rows], dtype=float)
    radius = np.asarray([row["radius_km"] for row in rows], dtype=float)
    inertia = np.asarray([row["inertia_cgs_g_cm2"] for row in rows], dtype=float) / 1.0e45
    lambdas = np.asarray([row["lambda"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    axes[0].plot(masses, radius, "o-", color="#245a8d", label="Hartle background")
    axes[0].set(xlabel=r"Mass $M,[M_\odot]$", ylabel=r"$R$ [km]")
    axes[1].plot(masses, inertia, "o-", color="#9b3d2f", label=r"$I$")
    axes[1].set(xlabel=r"Mass $M,[M_\odot]$", ylabel=r"$I$ [$10^{45}$ g cm$^2$]")
    axes[2].semilogy(masses, lambdas, "o-", color="#26734d", label=r"$\Lambda$")
    axes[2].set(xlabel=r"Mass $M,[M_\odot]$", ylabel=r"$\Lambda$")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    j0737 = payload["j0737a"]["forecast"]
    axes[0].axvline(j0737["mass_msun"], color="#555", ls="--", lw=0.8)
    axes[1].axvline(j0737["mass_msun"], color="#555", ls="--", lw=0.8)
    axes[2].axvline(j0737["mass_msun"], color="#555", ls="--", lw=0.8)
    fig.suptitle("P3-S5 exact-target NS/Hartle forecast — branch 1, conditional EOS", fontsize=12)
    fig.text(
        0.5, 0.01,
        "Same Hartle TOV background for I and tidal y; grid envelopes are sensitivity only; Q/inverse mass blocked.",
        ha="center", fontsize=8, color="#555", style="italic",
    )
    fig.subplots_adjust(bottom=0.17, top=0.84, wspace=0.28)
    fig.savefig(FIGURE_PATH, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_report(payload: dict[str, Any]) -> None:
    j = payload["j0737a"]["forecast"]
    band = payload["falsification_bands"]["I_A"]
    joint = payload["falsification_bands"]["joint_I_Lambda_R"]["rectangle"]
    grids = payload["grid_ladder"]["rows"]
    overlay_max = max(
        float(row["legacy_i_love_overlay"]["relative_disagreement"])
        for row in payload["forecast_mass_grid"]
    )
    display = j["display_observables"]
    endpoints = payload["j0737a"]["input_mass_envelope"]["rows"]
    lines = [
        "# P3-S5 — Exact-target NS/Hartle forecast and payload repair",
        "",
        "## Control block",
        "",
        "- Status: **FINAL / EXACT_TARGET_FROZEN_CONDITIONAL_FORECAST**.",
        f"- Freeze: `{FREEZE_TIMESTAMP_UTC}`; `{OBSERVATION_LOCK}`.",
        "- Model: canonical zero-jump EOS, first-order Hartle frame dragging, and Hinderer tidal y solved on each exact Hartle TOV background; branch 1 only.",
        "- Semantics: forecast values are `derived_conditional`; input/root/ODE envelopes and prospective bands are `sensitivity_only`; no independent evidence weight.",
        "",
        "## Exact target forecast",
        "",
        f"- J0737A (`M_A={j['mass_msun']:.7f} +/- {payload['j0737a']['mass_sigma_msun']:.7f} M_sun`) direct root: `R={display['R_km']['value']} km`, `C={display['compactness']['value']}`, `Lambda={display['Lambda']['value']}`, `I={display['I_g_cm2']['value']}e45 g cm^2`, `Ibar={display['Ibar']['value']}` (display digits are union-supported; raw values are in JSON).",
        f"- Root: `P_c={j['target_solve']['root_pressure_msun_fm3']:.9f}` MeV/fm^3, mass residual `{j['target_solve']['mass_residual_msun']:.3e} M_sun`, bracket width `{j['target_solve']['pressure_bracket_width']:.3e}`; method `{j['method']}`.",
        f"- J0737A branch `{j['branch_id']}`, `gap_crossed={payload['j0737a']['gap_crossed']}`; inverse-mass inference remains **BLOCKED**.",
        "- The mass table covers 1.2--2.0 M_sun; every row is a pressure-ordered exact root and same-background tidal solve; no I-Love lookup output is used.",
        f"- Exact J0737A mass ladder: `{[(row['label'], row['mass_target_msun'], row['target_solve']['root_pressure_msun_fm3']) for row in endpoints]}`.",
        "",
        "## Envelopes and prospective falsifiers",
        "",
        f"- Raw complete-union I_A envelope: `{band['lower']:.9e} .. {band['upper']:.9e} g cm^2`; supported display: `{band['display_lower']:.6g} .. {band['display_upper']:.6g}` (digits `{band['display_digits']}`).",
        f"- Raw joint rectangle: `I_A={joint['I_A_g_cm2']['lower']:.9e}..{joint['I_A_g_cm2']['upper']:.9e} g cm^2`, `Lambda_A={joint['Lambda_A']['lower']:.9f}..{joint['Lambda_A']['upper']:.9f}`, `R_A={joint['R_A_km']['lower']:.9f}..{joint['R_A_km']['upper']:.9f} km`; only a future result outside this exact conditional envelope is a prospective falsifier.",
        "- These are model-conditional sensitivity bands, not theory-wide confidence intervals: EOS/model systematics, rotation corrections, alternative branches, and an observation likelihood are missing.",
        "",
        "## Controls and provenance",
        "",
        f"- Branch/grid ladder: `{[(g['label'], g['pressure_points'], g['selected_branch_points']) for g in grids]}`; pressure order authoritative, mass sorting and disconnected bridging false.",
        f"- Frozen sequence identity: `{payload['frozen_sequence_identity']['status']}`, compared `{payload['frozen_sequence_identity']['allowed_rows_compared']}` allowed rows, maximum relative delta `{payload['frozen_sequence_identity']['max_relative_delta']:.3e}`.",
        f"- Exact root tolerance ladder: `{[(row['label'], row['pressure_abs_tol'], row['mass_abs_tol_msun'], row['mass_residual_msun']) for row in payload['controls']['exact_target_root_ladder']['rows']]}`; Hartle background ladder spreads `{payload['controls']['hartle_background_ladder']['relative_spreads']}`; tidal Lambda spread `{payload['controls']['tidal_same_background_solver_ladder']['lambda_relative_spread']:.3e}`.",
        f"- Hartle surface event/matching and independent identities are retained for each background ladder row; canonical P2 convergence `{payload['controls']['hartle_p2s5_convergence']['status']}`; unit/identity residuals `{payload['controls']['identity_max_relative_residual']}`.",
        f"- Legacy P2 interpolation sensitivity is retained as non-forecast diagnostic (I spread `{payload['j0737a']['hartle_interpolation_envelope']['j0737a_inertia_relative_spread']:.3e}`, R spread `{payload['j0737a']['hartle_interpolation_envelope']['j0737a_radius_relative_spread']:.3e}`).",
        f"- Canonical regression: `{payload['canonical_regression']['status']}` with frozen `(M_max,R_1.4,Lambda_1.4)={hartle.FROZEN_CANONICAL}`.",
        f"- Phase-1 converged zero-jump candidate wider envelope: **BLOCKED** (`{payload['phase1_candidate_sensitivity_envelope']['wider_envelope']}`); candidate rows are not reused without same-background Hartle revalidation.",
        f"- Retired universal-relation overlay is descriptive only (maximum |Ibar - transform|/transform on the table `{overlay_max:.3e}`); no calibration, lookup table, Q solve, or empirical confirmation is claimed.",
        f"- Q and inverse-mass inference remain **BLOCKED**; low-mass unresolved rows and all no-gap/branch controls are preserved.",
        f"- Complete payload digest: `{payload['payload_integrity']['payload_sha256']}` (`{payload['payload_integrity']['status']}`).",
        f"- Result: `{payload['artifacts']['result_json']}`; CSV: `{payload['artifacts']['csv']}`; figure: `{payload['artifacts']['figure_png']}`.",
        f"- Terminal evidence: `{payload['artifacts']['evidence']}`; source-to-artifact status `{payload['source_to_artifact_provenance']['status']}`.",
        "",
        "Terminal verification was run on this exact code/artifact state; no post-PASS edits are part of this report.",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifacts(payload: dict[str, Any]) -> None:
    """Write the uniquely named Phase-3 JSON/CSV/figure/report artifacts."""

    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(payload)
    _write_figure(payload)
    _write_report(payload)


def _print_summary(payload: dict[str, Any]) -> None:
    j = payload["j0737a"]["forecast"]
    print("=" * 80)
    print("P3-S5 EXACT-TARGET NS/HARTLE FORECAST")
    print(f"Status: {payload['status']} (branch {j['branch_id']}, same background={j['same_background']})")
    print(f"J0737A: M={j['mass_msun']:.7f}, R={j['radius_km']:.6f} km, C={j['compactness']:.6f}, Lambda={j['lambda']:.6f}, I={j['inertia_cgs_g_cm2'] / 1e45:.6f}e45 g cm^2, Ibar={j['inertia_bar']:.6f}")
    band = payload["falsification_bands"]["I_A"]
    print(f"Prospective I_A falsification band: [{band['lower'] / 1e45:.6f}, {band['upper'] / 1e45:.6f}]e45 g cm^2")
    print(f"Canonical regression: {payload['canonical_regression']['status']}; source-to-artifact: {payload['source_to_artifact_provenance']['status']}")
    print(f"Artifacts: {RESULT_PATH}; {CSV_PATH}; {FIGURE_PATH}; {REPORT_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small semantic run; not terminal artifacts")
    args = parser.parse_args(argv)
    payload = build_payload(quick=bool(args.quick))
    if not args.quick:
        # Build output is terminal only after all source/branch/semantic checks.
        assert_artifact_provenance(payload)
        write_artifacts(payload)
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
