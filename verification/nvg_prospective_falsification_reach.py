#!/usr/bin/env python3
"""Phase 3 prospective reach and critical-boundary ledger.

This module joins the *frozen structured outputs* from the Phase 1/2 echo,
S8, and PBH audits.  It does not fetch data, fit parameters, or import the
NS/Hartle worker's unfinished Phase 3 output.  The three calculations are
therefore deliberately different from a forecast made after observing a
target:

* the echo curve conditions on the deterministic synthetic-null likelihood
  and varies only a predeclared catalog-prefix exposure;
* the S8 probe expands the already-declared ``Omega_m``/CPL grid and reports
  the first *grid boundary* with an overlap, not a continuous best fit; and
* the PBH table solves algebraically for the rate/abundance factor required
  to equal the maintained PTA benchmark, while retaining the JWST and dark
  matter boxes as separate scope tests.

All values are labelled ``conditional``, ``sensitivity``, or ``blocked``.
The inherited Phase 1/2 files are inputs and are never rewritten.  Run from
the repository root with::

    python3 verification/nvg_prospective_falsification_reach.py

The default command writes uniquely named Phase 3 JSON/CSV/PNG artifacts.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# The Phase 1/2 producers are read-only dependencies.  In particular, do not
# import the P3-S1 Hartle producer: its target is merged by the Phase 3 gate.
import nvg_echo_hierarchical_upper_limit as echo
import nvg_pbh_nanograv_audit as pbh
import nvg_pbh_two_population as pbh_two
import nvg_s8_no_go as s8


SCHEMA_VERSION = "P3-S3-prospective-falsification-reach-v1"
ECHO_RESULT_PATH = HERE / "nvg_echo_hierarchical_upper_limit_p2s3_results.json"
S8_RESULT_PATH = HERE / "nvg_s8_no_go_map.json"
PBH_RESULT_PATH = ROOT / "Lunacy/runs/predictive-research/phases/phase-1/evidence/P1-S3-pbh-nanograv-audit.json"

ECHO_SOURCE_PATH = HERE / "nvg_echo_hierarchical_upper_limit.py"
ECHO_TEST_PATH = HERE / "test_p2_echo_hierarchical_upper_limit.py"
ECHO_CONFIG_PATH = HERE / "data/p2_s3_hierarchical_echo/analysis_config.json"
ECHO_MANIFEST_PATH = HERE / "data/p2_s3_hierarchical_echo/provenance_manifest.json"
S8_SOURCE_PATH = HERE / "nvg_s8_no_go.py"
S8_CANONICAL_SOURCE_PATH = HERE / "nvg_black_hole_entropy.py"
S8_LEGACY_OVERLAY_SOURCE_PATH = HERE / "nvg_desi_s8_joint_map.py"
PBH_AUDIT_SOURCE_PATH = HERE / "nvg_pbh_nanograv_audit.py"
PBH_LADDER_SOURCE_PATH = HERE / "nvg_pbh_mass_spectrum.py"
PBH_TWO_SOURCE_PATH = HERE / "nvg_pbh_two_population.py"
PBH_PROFILE_SOURCE_PATH = HERE / "nvg_pbh_dark_matter.py"

RESULT_PATH = HERE / "nvg_prospective_falsification_reach_p3s3_results.json"
CSV_PATH = HERE / "p3_s3_prospective_reach.csv"
FIGURE_PATH = HERE / "fig_p3_s3_prospective_reach.png"
P3_SOURCE_PATH = HERE / "nvg_prospective_falsification_reach.py"
P3_TEST_PATH = HERE / "test_p3_prospective_falsification_reach.py"

# The complete JSON payload is authenticated independently of the inherited
# producer/source pins.  ``payload_integrity`` is excluded from the canonical
# representation to avoid a circular digest; every other field, including
# blocker text, rows, precision metadata, and the adversarial controls, is
# covered.  This is an integrity/closedness control, not a claim that the
# underlying sensitivity inputs are observational evidence.
PAYLOAD_INTEGRITY_KEY = "payload_integrity"
PAYLOAD_DIGEST_ALGORITHM = "sha256-canonical-json-v1"

# The prefix counts are declared analysis coordinates, not a target-tuned
# choice.  Event order is inherited from the final producer (descending
# catalog network SNR with deterministic event-name tie breaking).
ECHO_PREFIX_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 259)
S8_OMEGA_BOUNDARY_GRID = (0.28, 0.29, 0.30, 0.305, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36)
S8_BOUNDARY_RESOLUTIONS = (25, 51, 101)
S8_BOUNDARY_W0 = (-1.50, -0.30)
S8_BOUNDARY_WA = (-2.50, 1.50)
S8_MAINTAINED_OMEGA = 0.315
S8_NORMALIZATION_CONVENTIONS = (
    ("omega_m_0.315", 0.315, "S8=s8_planck*growth*sqrt(Omega_m/0.315)"),
    ("omega_m_0.300", 0.300, "S8=s8_planck*growth*sqrt(Omega_m/0.300)"),
    ("none", None, "S8=s8_planck*growth"),
)
S8_NORMALIZATION_PROBE_OMEGAS = (0.300, 0.305, 0.3075, 0.310, 0.3125, 0.315)
PBH_SEED_CYCLES = (9, 10, 11)
PBH_EXPANDED_CYCLES = tuple(range(7, 19))
PBH_SEED_DENSITIES = (float(pbh_two.N_SEED_LO), float(pbh_two.N_SEED_HI))


class ProvenanceError(RuntimeError):
    """Raised when a frozen input or live producer assertion is not closed."""


def sha256_file(path: Path) -> str:
    """Return the byte-level SHA-256 digest of ``path``."""

    if not path.is_file():
        raise ProvenanceError(f"required input is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProvenanceError(f"required structured artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read structured artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProvenanceError(f"structured artifact is not a JSON object: {path}")
    return value


def _finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _close(left: Any, right: Any, *, rel: float = 1.0e-9, abs_tol: float = 1.0e-12) -> bool:
    try:
        return bool(math.isclose(float(left), float(right), rel_tol=rel, abs_tol=abs_tol))
    except (TypeError, ValueError):
        return False


def _round_report(value: float | None, decimals: int) -> float | None:
    """Round a raw diagnostic only when its declared precision is supported."""

    return None if value is None else float(np.round(float(value), int(decimals)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(value: Any) -> str:
    """Return the deterministic JSON representation used for payload pins."""

    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_digest(payload: dict[str, Any]) -> str:
    """Hash every payload field except the self-authentication block."""

    body = dict(payload)
    body.pop(PAYLOAD_INTEGRITY_KEY, None)
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _csv_payload_digest(result: dict[str, Any]) -> str:
    """Hash the canonical flattened CSV rows without depending on CSV bytes."""

    return hashlib.sha256(_canonical_json(_csv_rows(result)).encode("utf-8")).hexdigest()


def _attach_payload_integrity(result: dict[str, Any]) -> dict[str, Any]:
    """Attach a complete canonical digest and deterministic CSV-row digest."""

    payload = dict(result)
    # ``_csv_rows`` intentionally consumes only already-built fields.  It is
    # defined later in this module but available when build_result executes.
    payload[PAYLOAD_INTEGRITY_KEY] = {
        "algorithm": PAYLOAD_DIGEST_ALGORITHM,
        "canonical_sha256": _payload_digest(payload),
        "csv_rows_sha256": _csv_payload_digest(payload),
        "covered_scope": "all JSON fields except payload_integrity; flattened CSV rows included by csv_rows_sha256",
        "status": "PASS_COMPLETE_PAYLOAD_DIGEST",
    }
    return payload


def _assert_payload_integrity(result: dict[str, Any]) -> None:
    """Reject any mutation of a material emitted JSON/CSV field."""

    integrity = result.get(PAYLOAD_INTEGRITY_KEY)
    if not isinstance(integrity, dict):
        raise ProvenanceError("complete payload integrity block is missing")
    if integrity.get("algorithm") != PAYLOAD_DIGEST_ALGORITHM:
        raise ProvenanceError("unsupported complete payload digest algorithm")
    if integrity.get("status") != "PASS_COMPLETE_PAYLOAD_DIGEST":
        raise ProvenanceError("complete payload digest status is not PASS")
    expected = integrity.get("canonical_sha256")
    if not isinstance(expected, str) or expected != _payload_digest(result):
        raise ProvenanceError("complete canonical payload digest mismatch")
    expected_csv = integrity.get("csv_rows_sha256")
    if not isinstance(expected_csv, str) or expected_csv != _csv_payload_digest(result):
        raise ProvenanceError("complete canonical CSV-row digest mismatch")


def _file_pin(path: Path, role: str) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": sha256_file(path), "role": role}


def _assert_interval(interval: Sequence[Any], name: str) -> list[float]:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ProvenanceError(f"{name} must be a two-element interval")
    try:
        values = [float(interval[0]), float(interval[1])]
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"{name} is not numeric") from exc
    if not all(math.isfinite(item) for item in values) or values[0] > values[1]:
        raise ProvenanceError(f"{name} is not finite and ordered")
    return values


def _compact_s8_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Retain only boundary coordinates/diagnostics, not a hidden fit table."""

    if row is None:
        return None
    keys = (
        "w0",
        "wa",
        "omega_m",
        "s8",
        "growth_ratio_gamma055",
        "distance_to_declared_lensing",
        "desi_overlay_chi2",
        "lensing_1sigma",
        "desi_overlay_2sigma",
        "early_matter_fraction",
        "background_ok",
    )
    return {key: row[key] for key in keys if key in row}


def _canonical_q_digest(
    events: Sequence[dict[str, Any]], amplitudes: Sequence[float], q_matrix: np.ndarray
) -> str:
    """Digest event order, amplitude coordinates, and every q value."""

    q = np.asarray(q_matrix, dtype=float)
    payload = {
        "shape": list(q.shape),
        "events": [
            {"event": row.get("event"), "network_snr": float(row.get("network_snr"))}
            for row in events
        ],
        "amplitudes": [float(value) for value in amplitudes],
        "efficiency": q.tolist(),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_echo_input(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the live P2-S6 assertion and extract a closed deterministic q matrix.

    P2 source/config/catalog hashes authenticate the evaluator, but the P2
    artifact historically did not authenticate every emitted efficiency value.
    Recompute all 259 x 15 entries from the live deterministic evaluator here,
    then compare the complete matrix (and amplitude grid) before using it.
    """

    try:
        echo.assert_result_provenance(result)
    except (AssertionError, ProvenanceError, OSError, ValueError) as exc:
        raise ProvenanceError(f"echo artifact/live producer assertion failed: {exc}") from exc
    if result.get("schema_version") != "P2-S3-hierarchical-echo-v1":
        raise ProvenanceError("unsupported final echo artifact schema")
    if result.get("observational", {}).get("status") != "BLOCKED_MISSING_STRAIN_POSTERIOR_NOISE_PRODUCTS":
        raise ProvenanceError("echo observational block was not preserved")
    sensitivity = result.get("sensitivity", {})
    events = sensitivity.get("events")
    if not isinstance(events, list) or not events:
        raise ProvenanceError("echo artifact has no selected-event rows")
    injections = sensitivity.get("calibration", {}).get("injections", {})
    amplitudes = np.asarray(injections.get("amplitudes", []), dtype=float)
    efficiency_rows = injections.get("efficiency_rows", [])
    if amplitudes.ndim != 1 or len(amplitudes) < 2 or np.any(~np.isfinite(amplitudes)):
        raise ProvenanceError("echo amplitude grid is missing or non-finite")
    if not isinstance(efficiency_rows, list) or len(efficiency_rows) != len(events):
        raise ProvenanceError("echo efficiency rows do not match selected events")
    q_matrix = np.asarray([row.get("efficiency", []) for row in efficiency_rows], dtype=float)
    if q_matrix.shape != (len(events), len(amplitudes)) or np.any(~np.isfinite(q_matrix)):
        raise ProvenanceError("echo deterministic efficiency matrix has an invalid shape")
    if np.any(q_matrix < -1.0e-12) or np.any(q_matrix > 1.0 + 1.0e-12):
        raise ProvenanceError("echo efficiency matrix leaves [0,1]")
    if np.any(np.diff(q_matrix, axis=1) < -1.0e-12):
        raise ProvenanceError("echo efficiency matrix is not monotone")
    for event, efficiency in zip(events, efficiency_rows):
        if event.get("event") != efficiency.get("event") or not _close(event.get("network_snr"), efficiency.get("network_snr"), rel=0.0, abs_tol=1.0e-10):
            raise ProvenanceError("echo event/efficiency identity mismatch")
    if int(result.get("n_events", -1)) != len(events):
        raise ProvenanceError("echo n_events does not match selected rows")
    # Recompute the entire q payload from the live P2 evaluator and pinned
    # configuration.  A shape/range/monotonicity check alone permits arbitrary
    # monotone replacements and was the decisive P3-S4 failure.
    config = echo.load_config()
    live_events, live_catalog_audit = echo.load_catalog(config)
    live_calibration = echo.synthetic_calibration(live_events, config, quick=False)
    live_amplitudes = np.asarray(live_calibration["amplitudes"], dtype=float)
    if live_amplitudes.shape != amplitudes.shape or not np.array_equal(live_amplitudes, amplitudes):
        raise ProvenanceError("echo amplitude grid does not match live deterministic evaluator")
    live_q = np.asarray(live_calibration["q_matrix"], dtype=float)
    if live_q.shape != q_matrix.shape or not np.array_equal(live_q, q_matrix):
        max_delta = float(np.max(np.abs(live_q - q_matrix))) if live_q.shape == q_matrix.shape else math.inf
        raise ProvenanceError(f"echo efficiency matrix does not match live deterministic evaluator (max_delta={max_delta:.3e})")
    if len(live_events) != len(events) or live_catalog_audit.get("selection", {}).get("selected") != len(events):
        raise ProvenanceError("echo live catalog selection count differs from artifact")
    if [row.get("event") for row in live_events] != [row.get("event") for row in events]:
        raise ProvenanceError("echo selected-event order differs from live catalog")
    uncertainty = sensitivity.get("calibration_uncertainty", {})
    union = _assert_interval(uncertainty.get("union_interval"), "echo numerical union")
    raw_union = _assert_interval(uncertainty.get("overall_interval_raw"), "echo raw numerical union")
    if any(not _close(left, right, rel=0.0, abs_tol=1.0e-15) for left, right in zip(union, raw_union)):
        raise ProvenanceError("echo numerical union aliases drift")
    posterior = sensitivity.get("posterior", {})
    raw_a90 = float(posterior.get("upper90_raw"))
    if not (union[0] - 1.0e-15 <= raw_a90 <= union[1] + 1.0e-15):
        raise ProvenanceError("echo configured A90 lies outside its numerical union")
    live_by_name = {row["event"]: row for row in live_events}
    for row in events:
        live = live_by_name.get(row.get("event"))
        if live is None or not _close(live.get("network_snr"), row.get("network_snr"), rel=0.0, abs_tol=1.0e-10):
            raise ProvenanceError("echo live catalog event identity mismatch")
    return q_matrix, amplitudes, np.asarray([float(row["network_snr"]) for row in events], dtype=float), {
        "events": events,
        "config": config,
        "catalog_audit": live_catalog_audit,
        "numerical_union": union,
        "raw_a90": raw_a90,
        "live_q_matrix": live_q,
        "live_amplitudes": live_amplitudes,
        "q_payload_digest": _canonical_q_digest(events, amplitudes, q_matrix),
        "live_q_payload_digest": _canonical_q_digest(live_events, live_amplitudes, live_q),
    }


def _echo_reach(result: dict[str, Any]) -> dict[str, Any]:
    q_matrix, amplitudes, snr, details = _validate_echo_input(result)
    sensitivity = result["sensitivity"]
    population = sensitivity["population_parameter"]
    prior_min = float(population["prior_min"])
    prior_max = float(population["prior_max"])
    posterior_points = int(population["posterior_grid_points"])
    if posterior_points < 2 or prior_min < 0.0 or prior_max <= prior_min:
        raise ProvenanceError("echo posterior grid/prior is invalid")
    parameter_grid = np.linspace(prior_min, prior_max, posterior_points)
    reference_network_snr = float(details["config"]["synthetic_detector_model"].get("reference_network_snr", 20.0))
    if not math.isfinite(reference_network_snr) or reference_network_snr <= 0.0:
        raise ProvenanceError("echo reference network SNR is invalid")
    prefix_counts = tuple(count for count in ECHO_PREFIX_COUNTS if count <= len(snr))
    if not prefix_counts or prefix_counts[-1] != len(snr):
        prefix_counts = (*prefix_counts, len(snr))
    rows: list[dict[str, Any]] = []
    for count in prefix_counts:
        posterior = echo.posterior_from_efficiencies(
            q_matrix[:count], amplitudes, parameter_grid,
            prior=str(population["prior_family"]), prior_max=prior_max,
        )
        exposure_snr2 = float(np.sum(np.square(snr[:count])))
        exposure_normalized = float(exposure_snr2 / (reference_network_snr**2))
        rows.append(
            {
                "selected_event_count": int(count),
                "catalog_prefix_rule": "final-producer-selected order; fixed prefix, no target optimization",
                "exposure_network_snr2": exposure_snr2,
                "exposure_normalized_to_reference_snr": exposure_normalized,
                "a90_raw": float(posterior["upper90"]),
                # Subset rows are raw diagnostics only: the complete P2
                # numerical union is certified for the full selected set.
                "a90_display": None,
                "display_precision_decimal_places": None,
                "a90_status": "sensitivity",
                "scope": "catalog-conditioned deterministic synthetic-null y_i=0; not observational",
            }
        )
    full_row = rows[-1]
    full_union = details["numerical_union"]
    supported_digits = int(sensitivity["calibration_uncertainty"].get("supported_upper90_decimal_places", 0))
    full_row["a90_display"] = _round_report(full_row["a90_raw"], supported_digits)
    full_row["display_precision_decimal_places"] = supported_digits
    full_row["numerical_union_raw"] = list(full_union)
    full_row["numerical_union_scope"] = "full selected-event artifact only; subset rows have no newly recomputed union"
    for row in rows:
        row["asymptotic_scaled_a90"] = float(row["a90_raw"] * math.sqrt(row["exposure_network_snr2"]))
    # In the declared likelihood the signal non-centrality is (A rho eta)^2.
    # Thus its small-A information is proportional to sum(rho^2), yielding the
    # exact asymptotic exponent -1/2.  The finite prefix ladder is reported as
    # a diagnostic and is not forced to that exponent.
    if len(rows) >= 2:
        previous, last = rows[-2], rows[-1]
        finite_exponent = float(
            math.log(last["a90_raw"] / previous["a90_raw"])
            / math.log(last["exposure_network_snr2"] / previous["exposure_network_snr2"])
        )
    else:
        finite_exponent = None
    return {
        "status": "sensitivity",
        "producer_identity": {
            "producer": "verification/nvg_echo_hierarchical_upper_limit.py",
            "focused_test": "verification/test_p2_echo_hierarchical_upper_limit.py",
            "artifact": _relative(ECHO_RESULT_PATH),
            "live_assertion": "echo.assert_result_provenance + load_config + load_catalog",
            "assertion_status": "PASS",
        },
        "selected_event_count": int(len(snr)),
        "selected_event_count_scope": "391-row public catalog conditioned to 259 selected rows; no population-rate inference",
        "observational_blocks": {
            "status": result.get("observational", {}).get("status"),
            "observational_likelihood": result.get("observational", {}).get("observational_likelihood"),
            "missing": list(result.get("observational", {}).get("missing", [])),
            "time_slide_independence": dict(result.get("observational", {}).get("time_slide_independence", {})),
            "scope": "synthetic/catalog-conditioned sensitivity only; real strain/posterior/PSD and empirical slides remain blocked",
        },
        "deterministic_likelihood": {
            "event_likelihood": str(population.get("event_likelihood")),
            "observations": "all y_i=0 synthetic null; no event statistic",
            "amplitude_definition": str(population.get("definition")),
            "reference_network_snr": reference_network_snr,
            "prefix_order": "final artifact's selected-event order (network SNR descending; event name tie break)",
        },
        "q_payload_integrity": {
            "status": "PASS_RECOMPUTED_LIVE_FULL_MATRIX",
            "algorithm": "sha256-canonical-json-v1",
            "shape": [int(q_matrix.shape[0]), int(q_matrix.shape[1])],
            "amplitude_points": int(len(amplitudes)),
            "artifact_q_sha256": details["q_payload_digest"],
            "live_q_sha256": details["live_q_payload_digest"],
            "matches_live": bool(details["q_payload_digest"] == details["live_q_payload_digest"]),
            "source": "echo.synthetic_calibration(load_catalog(load_config()))",
            "scope": "all selected events and all configured efficiency amplitudes; q-row mutations fail closed",
        },
        "numerical_union": {
            "raw_interval": list(full_union),
            "displayed_interval": list(sensitivity["calibration_uncertainty"].get("overall_interval", full_union)),
            "supported_decimal_places": int(sensitivity["calibration_uncertainty"].get("supported_upper90_decimal_places", 0)),
            "axis_order": list(sensitivity["calibration_uncertainty"].get("axis_order", [])),
            "source_scope": "P2-S8 full-set quadrature/seed/posterior/amplitude/monotonicity union",
        },
        "exposure_curve": rows,
        "convergence": {
            "numerical_union_axes": {
                axis: sensitivity["calibration_uncertainty"].get("axes", {}).get(axis, {}).get("status")
                for axis in sensitivity["calibration_uncertainty"].get("axis_order", [])
            },
            "posterior_grid": sensitivity.get("numerical_convergence", {}),
            "status": "PASS_FROZEN_P2_NUMERICAL_UNION",
        },
        "asymptotic_scaling": {
            "variable": "exposure = sum_i rho_i^2",
            "noncentrality_identity": "lambda_i=(A*rho_i*eta_i)^2",
            "expected_small_amplitude_exponent": -0.5,
            "expected_relation": "A90 proportional to exposure^(-1/2) as A approaches zero at fixed nuisance/quadrature",
            "finite_prefix_endpoint_exponent": finite_exponent,
            "finite_prefix_is_not_asymptotic_claim": True,
            "diagnostic_status": "SENSITIVITY_ONLY",
        },
        "falsifier": "A future event-level strain/posterior/PSD likelihood with a non-null event statistic can displace the synthetic-null projection.",
        "minimal_prospective_requirement": "Provide keyed strain, posterior samples, detector PSDs, and an empirical time-slide bank; rerun the same hierarchical likelihood with an independently preregistered statistic.",
        "dominant_uncertainty": "No real event products; catalog selection and the complete P2-S8 raw A90 union are sensitivity inputs.",
        "scope": "conditional on the P2 deterministic synthetic detector model and catalog selection; not observational and never an observed echo upper limit",
        "source_artifact_provenance": {
            "result": _file_pin(ECHO_RESULT_PATH, "P2-S8 final structured echo artifact"),
            "producer": _file_pin(ECHO_SOURCE_PATH, "live P2 echo producer"),
            "focused_test": _file_pin(ECHO_TEST_PATH, "live P2 echo focused test"),
            "configuration": _file_pin(ECHO_CONFIG_PATH, "pinned P2 echo configuration"),
            "manifest": _file_pin(ECHO_MANIFEST_PATH, "pinned P2 echo provenance manifest"),
        },
    }


def _validate_s8_input(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("schema_version") != 1:
        raise ProvenanceError("unsupported final S8 artifact schema")
    if result.get("provenance", {}).get("desi_overlay_is_observed_likelihood") is not False:
        raise ProvenanceError("S8 legacy overlay was not marked non-observational")
    claim_scope = result.get("claim_scope", {})
    if claim_scope.get("observational_claim") is not False:
        raise ProvenanceError("S8 observational claim flag is not false")
    if result.get("status", "").split(";")[0] != "SCOPED_NO_GO_MAINTAINED_OMEGA_M":
        raise ProvenanceError("S8 maintained-slice no-go status drifted")
    live = s8.compute_s8_test()
    s8._CANONICAL = dict(live)
    maintained = result.get("maintained_point", {})
    live_point = s8.point_record(float(live["w_0"]), float(live["w_a"]), S8_MAINTAINED_OMEGA)
    for key, live_key in (("s8", "s8_nvg"), ("growth_ratio_gamma055", "sigma8_ratio")):
        # The maintained producer integrates a left/right rectangular rule,
        # while the audit map uses ``trapz``; this is a documented numerical
        # identity check, not an observational discrepancy.
        if not _close(maintained.get(key), live_point.get(key), rel=0.0, abs_tol=3.0e-7):
            raise ProvenanceError(f"S8 maintained point does not match live producer: {key}")
        if not _close(maintained.get(key), live.get(live_key), rel=0.0, abs_tol=3.0e-7):
            raise ProvenanceError(f"S8 maintained point does not match canonical source: {key}")
    rows = result.get("map_resolution_convergence", {}).get("rows", [])
    if not rows or not all(int(row.get("one_sigma_overlay_overlap", -1)) == 0 for row in rows):
        raise ProvenanceError("S8 maintained map resolution no-go is not zero")
    return {"live": live, "maintained": maintained}


def _s8_scan_at(omega_m: float, resolution: int) -> dict[str, Any]:
    domain = s8.Domain(
        f"prospective_boundary_omega_{omega_m:.6f}",
        S8_BOUNDARY_W0,
        S8_BOUNDARY_WA,
        (float(omega_m), float(omega_m)),
        (int(resolution), int(resolution), 1),
    )
    # Retaining this temporary point list lets us correct the inherited
    # producer's duplicated ``background_ok`` counter without changing that
    # read-only Phase 1 source.  The points themselves are not emitted.
    scan = s8.scan_domain(domain, n_steps=1000, retain_points=True)
    counts = scan["counts"]
    overlap = int(counts["both_lensing_1sigma_and_desi_overlay"])
    true_background_ok = int(sum(bool(row.get("background_ok")) for row in scan.get("points", [])))
    return {
        "omega_m": float(omega_m),
        "w0_range": list(S8_BOUNDARY_W0),
        "wa_range": list(S8_BOUNDARY_WA),
        "w0wa_resolution": int(resolution),
        "one_sigma_overlay_overlap": overlap,
        "two_sigma_overlay_overlap": int(counts["both_lensing_2sigma_and_desi_overlay"]),
        "background_ok_count": true_background_ok,
        "total": int(counts["total"]),
        "best_overlap_row": _compact_s8_row(scan.get("best_1sigma_overlay_overlap")),
        "boundary_status": "crossing_grid_cell" if overlap else "no_crossing_at_grid_cell",
    }


def _s8_normalization_probe(
    omega_values: Sequence[float], *, resolution: int = 101, n_steps: int = 1000
) -> dict[str, Any]:
    """Recompute overlap counts under an explicit S8 normalization ladder.

    The growth and legacy-overlay cuts are held fixed while only the
    normalization reference is changed.  This is the decisive P3-S4 probe:
    a crossing that changes under this convention ladder cannot be promoted
    to a physical critical Omega_m threshold.
    """

    if int(resolution) < 2:
        raise ValueError("S8 normalization probe resolution must be at least two")
    rows = [
        _s8_normalization_probe_single(float(omega_m), resolution=resolution, n_steps=n_steps)
        for omega_m in omega_values
    ]
    return {
        "resolution": int(resolution),
        "w0_range": list(S8_BOUNDARY_W0),
        "wa_range": list(S8_BOUNDARY_WA),
        "normalization_conventions": [
            {"name": name, "reference_omega_m": reference, "formula": formula}
            for name, reference, formula in S8_NORMALIZATION_CONVENTIONS
        ],
        "rows": rows,
        "scope": "same broad CPL/legacy-overlay grid; normalization convention only; non-observational sensitivity",
        "status": "NOT_IDENTIFIABLE_GRID_AND_NORMALIZATION_DEPENDENT",
    }


def _s8_normalization_probe_single(omega_m: float, *, resolution: int = 101, n_steps: int = 1000) -> dict[str, Any]:
    """Compute one Omega_m row for the normalization convention ladder."""

    canonical = s8._CANONICAL
    s8_planck = float(canonical["s8_planck"])
    s8_lensing = float(canonical["s8_lensing"])
    counts = {name: {"one_sigma_overlap": 0, "two_sigma_overlap": 0, "background_ok": 0, "total": 0} for name, _, _ in S8_NORMALIZATION_CONVENTIONS}
    w0_values = np.linspace(*S8_BOUNDARY_W0, int(resolution))
    wa_values = np.linspace(*S8_BOUNDARY_WA, int(resolution))
    for w0 in w0_values:
        for wa in wa_values:
            growth = s8.growth_index_ratio(float(w0), float(wa), float(omega_m), n_steps=n_steps)
            omega_at_min, _, _ = s8.background(np.asarray([s8.A_MIN]), float(w0), float(wa), float(omega_m))
            background_ok = bool(float(omega_at_min[0]) >= s8.EARLY_MATTER_FRACTION_MIN)
            overlay_ok = bool(s8.desi_overlay_chi2(float(w0), float(wa)) <= s8.DESI_CHI2_LIMIT)
            for name, reference, _formula in S8_NORMALIZATION_CONVENTIONS:
                value = s8_planck * growth if reference is None else s8_planck * growth * math.sqrt(float(omega_m) / float(reference))
                one_sigma = bool(abs(value - s8_lensing) <= s8.S8_LENSING_SIGMA)
                two_sigma = bool(abs(value - s8_lensing) <= 2.0 * s8.S8_LENSING_SIGMA)
                cell = counts[name]
                cell["total"] += 1
                cell["background_ok"] += int(background_ok)
                cell["one_sigma_overlap"] += int(background_ok and overlay_ok and one_sigma)
                cell["two_sigma_overlap"] += int(background_ok and overlay_ok and two_sigma)
    return {"omega_m": float(omega_m), "counts": counts}


def _s8_reach(result: dict[str, Any]) -> dict[str, Any]:
    validation = _validate_s8_input(result)
    # The inherited maintained resolution rows are retained exactly as a
    # regression input; the expanded probes below are recomputed read-only.
    inherited_rows = [
        {
            "resolution": int(row["resolution"]),
            "one_sigma_overlay_overlap": int(row["one_sigma_overlay_overlap"]),
            "two_sigma_overlay_overlap": int(row["two_sigma_overlay_overlap"]),
        }
        for row in result["map_resolution_convergence"]["rows"]
    ]
    boundary_rows = [_s8_scan_at(omega, 25) for omega in S8_OMEGA_BOUNDARY_GRID]
    nearest = min(
        (row for row in boundary_rows if row["one_sigma_overlay_overlap"] > 0),
        key=lambda row: abs(row["omega_m"] - S8_MAINTAINED_OMEGA),
        default=None,
    )
    resolution_rows = []
    for resolution in S8_BOUNDARY_RESOLUTIONS:
        probes = [_s8_scan_at(omega, resolution) for omega in (0.300, 0.305, 0.310)]
        candidates = [row for row in probes if row["one_sigma_overlay_overlap"] > 0]
        nearest_probe = min(candidates, key=lambda row: abs(row["omega_m"] - S8_MAINTAINED_OMEGA), default=None)
        resolution_rows.append(
            {
                "w0wa_resolution": int(resolution),
                "probes": probes,
                "nearest_probe_with_overlap": nearest_probe,
            }
        )
    # A resolution-dependent first crossing is intentionally not interpolated
    # to a continuous Omega_m threshold.
    crossing_omegas = [
        float(row["nearest_probe_with_overlap"]["omega_m"])
        for row in resolution_rows
        if row["nearest_probe_with_overlap"] is not None
    ]
    boundary_resolution_dependent = len(set(crossing_omegas)) > 1
    normalization_probe = _s8_normalization_probe(S8_NORMALIZATION_PROBE_OMEGAS, resolution=101, n_steps=1000)
    normalization_counts = {
        str(row["omega_m"]): {
            name: dict(counts)
            for name, counts in row["counts"].items()
        }
        for row in normalization_probe["rows"]
    }
    normalization_changed = any(
        len({int(counts["one_sigma_overlap"]) for counts in row["counts"].values()}) > 1
        for row in normalization_probe["rows"]
    )
    legacy = result["inputs"]["desi_overlay"]
    return {
        "status": "sensitivity",
        "producer_identity": {
            "canonical_producer": "verification/nvg_black_hole_entropy.py::compute_s8_test",
            "maintained_map_producer": "verification/nvg_s8_no_go.py",
            "artifact": _relative(S8_RESULT_PATH),
            "live_assertion": "compute_s8_test + point_record + maintained zero-overlap rows",
            "assertion_status": "PASS",
        },
        "maintained_slice": {
            "omega_m": S8_MAINTAINED_OMEGA,
            "inherited_resolution_rows": inherited_rows,
            "one_sigma_overlap_count": 0,
            "no_go_status": "conditional",
            "scoped_scalar_no_go": True,
            "global_statement_status": "BLOCKED_NO_SOURCED_JOINT_LIKELIHOOD",
            "scope": "declared scalar S8/lensing target on the maintained Omega_m=0.315 slice",
        },
        "convergence": {
            "canonical": result.get("convergence", {}),
            "map_resolution": result.get("map_resolution_convergence", {}),
            "status": "PASS_INHERITED_CANONICAL_AND_MAINTAINED_MAP_CONTROLS",
        },
        "boundary_expansion": {
            "omega_grid": list(S8_OMEGA_BOUNDARY_GRID),
            "w0_range": list(S8_BOUNDARY_W0),
            "wa_range": list(S8_BOUNDARY_WA),
            "w0wa_resolution": 25,
            "rows": boundary_rows,
            "nearest_grid_boundary": nearest,
            "nearest_omega_departure_from_maintained": None if nearest is None else float(abs(nearest["omega_m"] - S8_MAINTAINED_OMEGA)),
            "nearest_boundary_is_continuous_inference": False,
            "interpretation": "first grid cell with a maintained one-sigma S8 plus legacy-overlay overlap; not a fitted or optimized boundary",
            "critical_boundary_status": "NOT_IDENTIFIABLE_GRID_AND_NORMALIZATION_DEPENDENT",
            "display_precision": {
                "omega_m_decimal_places": 3,
                "raw_coordinates_retained": True,
                "note": "grid coordinates are sensitivity diagnostics, not a continuous threshold",
            },
        },
        "resolution_probe": {
            "rows": resolution_rows,
            "nearest_crossing_omegas": crossing_omegas,
            "boundary_resolution_dependent": bool(boundary_resolution_dependent),
            "status": "NOT_IDENTIFIABLE_GRID_AND_NORMALIZATION_DEPENDENT",
            "interpretation": "crossing coordinate is a resolution-dependent grid candidate; it is not an invariant critical Omega_m",
        },
        "normalization_probe": {
            **normalization_probe,
            "one_sigma_counts_by_omega": normalization_counts,
            "normalization_changes_one_sigma_overlap": bool(normalization_changed),
            "critical_boundary_status": "NOT_IDENTIFIABLE_GRID_AND_NORMALIZATION_DEPENDENT",
        },
        "legacy_overlay_scope": {
            "source": result["provenance"].get("desi_overlay_source"),
            "observed_likelihood": False,
            "provenance_status": "unknown_provenance_sensitivity_overlay",
            "chi2_limit": float(legacy["chi2_limit"]),
            "target_semantics": "declared scalar one-sigma lensing band plus inherited chi2 overlay; no DESI likelihood reconstructed",
        },
        "global_statement_status": "BLOCKED_NO_SOURCED_JOINT_LIKELIHOOD",
        "falsifier": "A sourced, independent S8/lensing likelihood at the maintained Omega_m slice that admits the claimed overlap would falsify the scoped no-go; a revised Omega_m prior would falsify its global interpretation.",
        "minimal_prospective_requirement": "Supply independently sourced S8/lensing and (w0,wa,Omega_m) likelihood products, then preregister the domain and evaluate a joint likelihood without the legacy overlay shortcut.",
        "dominant_uncertainty": "Unknown-provenance legacy overlay, growth-index approximation, and fixed Omega_m slice; the nearest crossing shifts with w0/wa grid resolution.",
        "scope": "critical boundary is a sensitivity-only grid discovery candidate; no global observational no-go is claimed",
        "source_artifact_provenance": {
            "result": _file_pin(S8_RESULT_PATH, "P1-S2 final structured S8 artifact"),
            "maintained_map_producer": _file_pin(S8_SOURCE_PATH, "live P1 S8 map producer"),
            "canonical_producer": _file_pin(S8_CANONICAL_SOURCE_PATH, "live maintained S8 producer"),
            "legacy_overlay_source": _file_pin(S8_LEGACY_OVERLAY_SOURCE_PATH, "retired/unknown-provenance overlay source"),
        },
        "live_maintained_point": validation["live"],
    }


def _values_equal(left: Any, right: Any, *, path: str = "") -> bool:
    """Strictly compare a structured PBH row/summary with live output."""

    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_values_equal(left[key], right[key], path=f"{path}.{key}") for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_values_equal(a, b, path=path) for a, b in zip(left, right))
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        # JSON round-trips preserve these values; a very tight tolerance allows
        # only platform-level libm noise, never a material row mutation.
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-30)
    return left == right


def _assert_live_scan_matches_artifact(
    artifact: dict[str, Any], live: dict[str, Any], label: str
) -> None:
    """Authenticate every emitted P1 scan field used by the P3 ledger."""

    # The P1 artifact is a read-only dependency.  Compare all scan metadata,
    # extrema, and the complete max row; do not merely check its mass ladder.
    for key in (
        "cycles", "density_grid_points", "abundance_grid_points", "rate_grid_points",
        "rows_evaluated", "density_range_mpc3", "max_amplitude_strain",
        "max_deficit_factor", "max_at_mass_boundary", "max_at_abundance_boundary",
        "external_exclusion_applied", "status",
    ):
        if key in artifact or key in live:
            if not _values_equal(artifact.get(key), live.get(key), path=f"{label}.{key}"):
                raise ProvenanceError(f"PBH live {label} field drift: {key}")
    if not _values_equal(artifact.get("max_row"), live.get("max_row"), path=f"{label}.max_row"):
        raise ProvenanceError(f"PBH live {label} max_row drift")


def _live_pbh_scans(result: dict[str, Any]) -> dict[str, Any]:
    """Recompute all P1 extrema from live producers and declared scan grids."""

    contract = result.get("scan_contract", {})
    if contract.get("seed_cycles") != list(PBH_SEED_CYCLES) or contract.get("expanded_cycles") != list(PBH_EXPANDED_CYCLES):
        raise ProvenanceError("PBH seed/expanded rung contract drifted")
    seed_artifact = result.get("jwst_calibrated_seed_band", {})
    expanded_artifact = result.get("jwst_calibrated_expanded_boundary", {})
    budget_seed_artifact = result.get("internal_budget_seed_band", {})
    budget_expanded_artifact = result.get("internal_budget_expanded_boundary", {})
    # These are the terminal P1 configuration values.  They are explicit in
    # the artifact and are also checked against the maintained producer API.
    density_points = int(seed_artifact.get("density_grid_points", -1))
    rate_points = int(seed_artifact.get("rate_grid_points", -1))
    abundance_points = int(budget_expanded_artifact.get("abundance_grid_points", -1))
    if density_points != 17 or rate_points != 3 or abundance_points != 33:
        raise ProvenanceError("PBH scan configuration is not the frozen endpoint-inclusive ladder")
    if int(expanded_artifact.get("density_grid_points", -1)) != density_points or int(budget_seed_artifact.get("rate_grid_points", -1)) != rate_points:
        raise ProvenanceError("PBH scan configuration rows disagree")
    live_seed = pbh._scan_jwst_calibrated(cycles=PBH_SEED_CYCLES, density_points=density_points, rate_points=rate_points)
    live_expanded = pbh._scan_jwst_calibrated(cycles=PBH_EXPANDED_CYCLES, density_points=density_points, rate_points=rate_points)
    live_budget_seed = pbh._scan_internal_budget(cycles=PBH_SEED_CYCLES, abundance_points=abundance_points, rate_points=rate_points)
    live_budget_expanded = pbh._scan_internal_budget(cycles=PBH_EXPANDED_CYCLES, abundance_points=abundance_points, rate_points=rate_points)
    _assert_live_scan_matches_artifact(seed_artifact, live_seed, "jwst_calibrated_seed_band")
    _assert_live_scan_matches_artifact(expanded_artifact, live_expanded, "jwst_calibrated_expanded_boundary")
    _assert_live_scan_matches_artifact(budget_seed_artifact, live_budget_seed, "internal_budget_seed_band")
    _assert_live_scan_matches_artifact(budget_expanded_artifact, live_budget_expanded, "internal_budget_expanded_boundary")
    return {
        "seed": live_seed,
        "expanded": live_expanded,
        "budget_seed": live_budget_seed,
        "budget_expanded": live_budget_expanded,
        "configuration": {
            "density_grid_points": density_points,
            "abundance_grid_points": abundance_points,
            "rate_grid_points": rate_points,
        },
    }


def _validate_pbh_input(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "CONDITIONAL_JWST_BAND_DEFICIT_NO_INTERNAL_BUDGET_NO_GO":
        raise ProvenanceError("unsupported final PBH artifact status")
    benchmark = result.get("benchmark", {})
    live_benchmark = pbh.benchmark_metadata()
    if not _close(benchmark.get("amplitude_strain"), live_benchmark.get("amplitude_strain"), rel=0.0, abs_tol=1.0e-30):
        raise ProvenanceError("PBH benchmark does not match live producer")
    if benchmark.get("observed_likelihood") is not None or benchmark.get("target_calibration_used") is not False:
        raise ProvenanceError("PBH benchmark semantics were upgraded to a likelihood")
    contract = result.get("scan_contract", {})
    if contract.get("external_exclusions_used") is not False:
        raise ProvenanceError("PBH external exclusion was applied")
    if contract.get("seed_cycles") != list(PBH_SEED_CYCLES):
        raise ProvenanceError("PBH declared seed rung band drifted")
    for cycle in PBH_EXPANDED_CYCLES:
        live_mass = pbh.canonical_mass(cycle)
        # Identity is checked against the final artifact's all-DM rung rows.
        rows = result.get("internal_budget_expanded_boundary", {}).get("per_cycle_all_dm", [])
        row = next((item for item in rows if int(item.get("cycle", -1)) == cycle), None)
        if row is None or not _close(row.get("mass_msun"), live_mass, rel=0.0, abs_tol=1.0e-8):
            raise ProvenanceError(f"PBH mass-ladder identity missing for cycle {cycle}")
    scans = _live_pbh_scans(result)
    return {"benchmark": benchmark, "live_benchmark": live_benchmark, "live_scans": scans}


def _pbh_requirement_row(cycle: int, number_density: float, benchmark: float) -> dict[str, Any]:
    mass = pbh.canonical_mass(int(cycle))
    if number_density <= 0.0 or not math.isfinite(number_density):
        raise ProvenanceError("PBH requirement density must be finite and positive")
    # ``number_density = f_PBH * rho_DM / M`` in the canonical producer.
    fraction_dm = float(number_density * mass / pbh_two.RHO_DM_MSUN_MPC3)
    unit_rate_amplitude = pbh.amplitude_strain(fraction_dm=fraction_dm, mass_msun=mass)
    if unit_rate_amplitude <= 0.0:
        required_rate_product = math.inf
    else:
        required_rate_product = float((benchmark / unit_rate_amplitude) ** 2)
    return {
        "cycle": int(cycle),
        "mass_msun": float(mass),
        "number_density_mpc3": float(number_density),
        "fraction_dm": fraction_dm,
        "unit_rate_amplitude_strain": float(unit_rate_amplitude),
        "required_rate_product": required_rate_product,
        "rate_factor_box": [0.0, 1.0],
        "required_rate_inside_declared_box": bool(math.isfinite(required_rate_product) and 0.0 <= required_rate_product <= 1.0),
        "inside_jwst_seed_density_band": bool(PBH_SEED_DENSITIES[0] <= number_density <= PBH_SEED_DENSITIES[1]),
        "inside_declared_jwst_rung_band": bool(cycle in PBH_SEED_CYCLES),
        "inside_dm_budget": bool(0.0 <= fraction_dm <= 1.0),
        "coordinate_scan_boundary": False,
        "solution_scan_boundary": False,
        "required_factor_if_other_two_rates_are_unit": required_rate_product,
        "individual_rate_factors_identifiable": False,
        "status": "sensitivity",
    }


def _pbh_reach(result: dict[str, Any]) -> dict[str, Any]:
    validation = _validate_pbh_input(result)
    benchmark = float(validation["benchmark"]["amplitude_strain"])
    live_scans = validation["live_scans"]
    seed_rows = [_pbh_requirement_row(cycle, density, benchmark) for cycle in PBH_SEED_CYCLES for density in PBH_SEED_DENSITIES]
    for row in seed_rows:
        # Endpoint density is a declared coordinate, not an optimized value.
        row["coordinate_scan_boundary"] = bool(row["number_density_mpc3"] in PBH_SEED_DENSITIES)
        row["solution_scan_boundary"] = bool(row["required_rate_inside_declared_box"] and row["coordinate_scan_boundary"])

    # Solve for the abundance required at unit rate.  This is useful because a
    # rate product above one at the JWST density does not tell us whether the
    # separate all-DM envelope can reach the target.
    expanded_rows: list[dict[str, Any]] = []
    for cycle in PBH_EXPANDED_CYCLES:
        mass = pbh.canonical_mass(cycle)
        required_density = float(pbh_two.N_SMBH * (pbh_two.M_SMBH / mass) ** (5.0 / 3.0))
        required_fraction = float(required_density * mass / pbh_two.RHO_DM_MSUN_MPC3)
        row = {
            "cycle": int(cycle),
            "mass_msun": float(mass),
            "required_number_density_at_unit_rate_mpc3": required_density,
            "required_fraction_dm_at_unit_rate": required_fraction,
            "inside_jwst_seed_density_band": bool(PBH_SEED_DENSITIES[0] <= required_density <= PBH_SEED_DENSITIES[1]),
            "inside_declared_jwst_rung_band": bool(cycle in PBH_SEED_CYCLES),
            "inside_dm_budget": bool(0.0 <= required_fraction <= 1.0),
            "rate_product_at_required_density": 1.0,
            "individual_rate_factors_identifiable": False,
            "rate_at_upper_declared_boundary": True,
            "rung_at_expanded_scan_boundary": bool(cycle in (min(PBH_EXPANDED_CYCLES), max(PBH_EXPANDED_CYCLES))),
            "density_at_jwst_scan_boundary": bool(math.isclose(required_density, PBH_SEED_DENSITIES[0], rel_tol=0.0, abs_tol=1.0e-15) or math.isclose(required_density, PBH_SEED_DENSITIES[1], rel_tol=0.0, abs_tol=1.0e-15)),
            "solution_scan_boundary": bool(
                PBH_SEED_DENSITIES[0] <= required_density <= PBH_SEED_DENSITIES[1]
                and 0.0 <= required_fraction <= 1.0
            ),
            "status": "sensitivity",
        }
        expanded_rows.append(row)

    # A second rung scan uses the declared JWST upper density and unit rate to
    # show the first expanded rung that can reach the benchmark.  No parameter
    # is adjusted to make the row pass; this is an endpoint-inclusive ledger.
    upper_density_rows = []
    for cycle in PBH_EXPANDED_CYCLES:
        row = _pbh_requirement_row(cycle, PBH_SEED_DENSITIES[1], benchmark)
        row["scan_coordinate"] = "JWST upper density, rate product=1 envelope"
        row["solution_reachable_at_unit_rate"] = bool(row["required_rate_product"] <= 1.0)
        row["rate_at_upper_declared_boundary"] = bool(row["required_rate_inside_declared_box"])
        row["rung_at_expanded_scan_boundary"] = bool(cycle in (min(PBH_EXPANDED_CYCLES), max(PBH_EXPANDED_CYCLES)))
        row["density_at_jwst_scan_boundary"] = True
        row["solution_scan_boundary"] = bool(
            row["solution_reachable_at_unit_rate"]
            and (row["rung_at_expanded_scan_boundary"] or row["density_at_jwst_scan_boundary"] or row["rate_at_upper_declared_boundary"])
        )
        upper_density_rows.append(row)
    reachable = [row for row in upper_density_rows if row["solution_reachable_at_unit_rate"]]
    first_reachable = min(reachable, key=lambda row: row["cycle"], default=None)
    # Use the just-recomputed live scan, never a copied P1 max row.
    expanded_max = live_scans["budget_expanded"]["max_row"]
    expanded_max_boundary = {
        "cycle": int(expanded_max["cycle"]),
        "fraction_dm": float(expanded_max["fraction_dm"]),
        "binary_pair_fraction": float(expanded_max["binary_pair_fraction"]),
        "duty_cycle": float(expanded_max["duty_cycle"]),
        "merger_rate_ratio": float(expanded_max["merger_rate_ratio"]),
        "max_amplitude_strain": float(expanded_max["amplitude_strain"]),
        "at_upper_cycle_boundary": bool(expanded_max["cycle"] == max(PBH_EXPANDED_CYCLES)),
        "at_upper_abundance_boundary": bool(math.isclose(float(expanded_max["fraction_dm"]), 1.0, rel_tol=0.0, abs_tol=1.0e-15)),
        "at_upper_rate_boundaries": bool(all(math.isclose(float(expanded_max[name]), 1.0, rel_tol=0.0, abs_tol=1.0e-15) for name in ("binary_pair_fraction", "duty_cycle", "merger_rate_ratio"))),
    }
    seed_max = live_scans["seed"]["max_row"]
    return {
        "status": "sensitivity",
        "producer_identity": {
            "audit_producer": "verification/nvg_pbh_nanograv_audit.py",
            "mass_ladder": "verification/nvg_pbh_mass_spectrum.py:get_pbh_mass",
            "two_population": "verification/nvg_pbh_two_population.py",
            "artifact": _relative(PBH_RESULT_PATH),
            "live_assertion": "benchmark_metadata + canonical_mass + amplitude_strain + final rung rows",
            "assertion_status": "PASS",
        },
        "benchmark": {
            "amplitude_strain": benchmark,
            "source": validation["benchmark"].get("source_path"),
            "observed_likelihood": None,
            "target_role": "benchmark requirement only; never fit or selection criterion",
        },
        "declared_boxes": {
            "jwst_seed_density_mpc3": list(PBH_SEED_DENSITIES),
            "jwst_seed_rungs": list(PBH_SEED_CYCLES),
            "expanded_rungs": list(PBH_EXPANDED_CYCLES),
            "dm_fraction": [0.0, 1.0],
            "rate_factor_names": list(pbh.RATE_FACTOR_NAMES),
            "rate_factor_box": list(pbh.RATE_FACTOR_RANGE),
            "external_exclusions_used": False,
        },
        "seed_band_requirements": {
            "rows": seed_rows,
            "all_rate_requirements_outside_box": bool(all(not row["required_rate_inside_declared_box"] for row in seed_rows)),
            "minimum_required_rate_product": float(min(row["required_rate_product"] for row in seed_rows)),
            "minimum_row": min(seed_rows, key=lambda row: row["required_rate_product"]),
            "seed_band_max_row_live_recomputed": {
                "cycle": int(seed_max["cycle"]),
                "number_density_mpc3": float(seed_max["number_density_mpc3"]),
                "amplitude_strain": float(seed_max["amplitude_strain"]),
                "deficit_factor": float(seed_max["deficit_factor"]),
                "inside_dm_budget": bool(float(seed_max["fraction_dm"]) <= 1.0),
            },
            # Retain the old key for consumers, but make its provenance
            # explicit: values are live-recomputed, not trusted copies.
            "seed_band_max_row_from_frozen_artifact": {
                "cycle": int(seed_max["cycle"]),
                "number_density_mpc3": float(seed_max["number_density_mpc3"]),
                "amplitude_strain": float(seed_max["amplitude_strain"]),
                "deficit_factor": float(seed_max["deficit_factor"]),
                "inside_dm_budget": bool(float(seed_max["fraction_dm"]) <= 1.0),
                "source": "live_recomputed_pbh_scan; compatibility alias only",
            },
            "scope": "all three declared JWST rungs and both density endpoints; rate product is an algebraic requirement, not a fit",
        },
        "unit_rate_abundance_requirements": {
            "rows": expanded_rows,
            "first_rung_with_required_density_inside_jwst_band": next((row["cycle"] for row in expanded_rows if row["inside_jwst_seed_density_band"]), None),
            "first_rung_with_required_density_inside_dm_budget": next((row["cycle"] for row in expanded_rows if row["inside_dm_budget"]), None),
            "scope": "expanded rung sensitivity; unit rate is the upper declared rate product",
        },
        "expanded_jwst_upper_density_scan": {
            "rows": upper_density_rows,
            "first_reachable_rung": None if first_reachable is None else int(first_reachable["cycle"]),
            "first_reachable_row": first_reachable,
            "status": "SENSITIVITY_ONLY_RUNG_BOUNDARY_SCAN",
        },
        "convergence": result.get("convergence", {}),
        "analytic_scaling": result.get("analytic_scaling", {}),
        "frozen_expanded_envelope_boundary": expanded_max_boundary,
        "live_scan_configuration": live_scans["configuration"],
        "max_row_integrity": {
            "status": "PASS_LIVE_RECOMPUTED_COMPLETE_ROWS",
            "seed_max_row_source": "pbh._scan_jwst_calibrated",
            "expanded_max_row_source": "pbh._scan_internal_budget",
            "scan_maxima_copied_from_frozen_artifact": False,
            "seed_max_row_live_recomputed": dict(live_scans["seed"]["max_row"]),
            "expanded_max_row_live_recomputed": dict(live_scans["budget_expanded"]["max_row"]),
            "complete_row_fields_checked": [
                "cycle", "mass_msun", "fraction_dm", "number_density_mpc3",
                "binary_pair_fraction", "duty_cycle", "merger_rate_ratio",
                "target_scale", "amplitude_strain", "deficit_factor",
            ],
        },
        "falsifier": "A sourced JWST occupation/abundance and PBH binary-rate likelihood that reaches the PTA benchmark within the declared rung, density, DM, and rate boxes would falsify the seed-band deficit.",
        "minimal_prospective_requirement": "Supply an independently sourced seed occupation and merger-rate model; demonstrate a rate product in [0,1] for rungs 9--11 and a fraction f_PBH consistent with the DM budget, with no target-tuned abundance.",
        "dominant_uncertainty": "JWST occupation/duty cycle and PBH binary-merger efficiency are absent; the PTA number is an internal benchmark, and expanded maxima sit on scan boundaries.",
        "scope": "calibrated abundance/rate sensitivity only; no external CMB exclusion or global PBH--PTA likelihood",
        "source_artifact_provenance": {
            "result": _file_pin(PBH_RESULT_PATH, "P1-S3 final structured PBH artifact"),
            "audit_producer": _file_pin(PBH_AUDIT_SOURCE_PATH, "live P1 PBH audit producer"),
            "mass_ladder": _file_pin(PBH_LADDER_SOURCE_PATH, "live PBH mass ladder producer"),
            "two_population": _file_pin(PBH_TWO_SOURCE_PATH, "live PBH two-population constants"),
            "population_profile": _file_pin(PBH_PROFILE_SOURCE_PATH, "live PBH profile producer"),
        },
        "target_leakage": {
            "benchmark_used_only_for_algebraic_required_factor": True,
            "parameters_optimized_against_target": False,
            "scan_maxima_copied_from_frozen_artifact": False,
            "scan_maxima_live_recomputed": True,
            "status": "PASS_NO_TARGET_FIT_AND_LIVE_MAX_ROW_RECOMPUTATION",
        },
    }


def _target_leakage_controls(echo_result: dict[str, Any], s8_result: dict[str, Any], pbh_result: dict[str, Any]) -> dict[str, Any]:
    """Make target roles explicit and fail closed on common shortcut mutations."""

    echo_status = echo_result.get("sensitivity", {}).get("sensitivity_status")
    s8_overlay_observed = s8_result.get("provenance", {}).get("desi_overlay_is_observed_likelihood")
    pbh_external = pbh_result.get("scan_contract", {}).get("external_exclusions_used")
    checks = {
        "echo_synthetic_null_only": echo_status == "SENSITIVITY_ONLY_CONDITIONAL_SYNTHETIC_NULL",
        "echo_no_observational_likelihood": echo_result.get("observational", {}).get("observational_likelihood") == "BLOCKED_NO_EVENT_LEVEL_STRAIN_POSTERIOR_NOISE",
        "s8_overlay_not_observed": s8_overlay_observed is False,
        "pbh_external_exclusions_not_applied": pbh_external is False,
        "pbh_benchmark_not_likelihood": pbh_result.get("benchmark", {}).get("observed_likelihood") is None,
        "pbh_target_calibration_false": pbh_result.get("benchmark", {}).get("target_calibration_used") is False,
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ProvenanceError(f"target-leakage guard failed: {failed}")
    return {
        "checks": checks,
        "echo_parameter_selection": "fixed catalog-prefix counts; no target-dependent event selection",
        "s8_parameter_selection": "fixed endpoint-inclusive CPL/Omega_m grids; no fitted threshold or optimizer",
        "pbh_parameter_selection": "benchmark enters only inverse requirement algebra; no target-calibrated scan maximum",
        "ns_hartle": "not imported; pending P3-S1 merge",
        "status": "PASS_NO_TARGET_FIT_OR_OBSERVATION_RELABELLING",
    }


def _expect_provenance_error(callback: Any) -> bool:
    """Return True only when a deliberate mutation is rejected."""

    try:
        callback()
    except (ProvenanceError, AssertionError, ValueError, TypeError, KeyError):
        return True
    return False


def _adversarial_controls(
    echo_artifact: dict[str, Any], pbh_artifact: dict[str, Any], result_without_integrity: dict[str, Any]
) -> dict[str, Any]:
    """Re-run the exact P3-S4 q/max-row and payload mutation probes."""

    echo_mutation_checks: dict[str, bool] = {}
    base_rows = echo_artifact["sensitivity"]["calibration"]["injections"]["efficiency_rows"]
    mutations: dict[str, Any] = {}
    low = copy.deepcopy(echo_artifact)
    for row in low["sensitivity"]["calibration"]["injections"]["efficiency_rows"]:
        row["efficiency"] = [1.0e-9] * len(row["efficiency"])
    mutations["echo_q_all_low"] = low
    flat = copy.deepcopy(echo_artifact)
    for row in flat["sensitivity"]["calibration"]["injections"]["efficiency_rows"]:
        row["efficiency"] = [0.9] * len(row["efficiency"])
    mutations["echo_q_all_flat_high"] = flat
    step = copy.deepcopy(echo_artifact)
    for row in step["sensitivity"]["calibration"]["injections"]["efficiency_rows"]:
        row["efficiency"] = [0.0] + [1.0] * (len(row["efficiency"]) - 1)
    mutations["echo_q_step_all_rows"] = step
    first = copy.deepcopy(echo_artifact)
    first["sensitivity"]["calibration"]["injections"]["efficiency_rows"][0]["efficiency"] = [0.0] + [1.0] * (len(base_rows[0]["efficiency"]) - 1)
    mutations["echo_q_step_first_row"] = first
    for name, payload in mutations.items():
        echo_mutation_checks[name] = _expect_provenance_error(lambda payload=payload: _validate_echo_input(payload))

    pbh_mutation_checks: dict[str, bool] = {}
    expanded_fraction = copy.deepcopy(pbh_artifact)
    expanded_fraction["internal_budget_expanded_boundary"]["max_row"]["fraction_dm"] = 0.123456789
    pbh_mutation_checks["pbh_expanded_max_fraction"] = _expect_provenance_error(lambda: _validate_pbh_input(expanded_fraction))
    expanded_amplitude = copy.deepcopy(pbh_artifact)
    expanded_amplitude["internal_budget_expanded_boundary"]["max_row"]["amplitude_strain"] = 9.87654321
    pbh_mutation_checks["pbh_expanded_max_amplitude"] = _expect_provenance_error(lambda: _validate_pbh_input(expanded_amplitude))
    seed_amplitude = copy.deepcopy(pbh_artifact)
    seed_amplitude["jwst_calibrated_seed_band"]["max_row"]["amplitude_strain"] = 9.87654321
    pbh_mutation_checks["pbh_seed_max_amplitude"] = _expect_provenance_error(lambda: _validate_pbh_input(seed_amplitude))
    seed_deficit = copy.deepcopy(pbh_artifact)
    seed_deficit["jwst_calibrated_seed_band"]["max_row"]["deficit_factor"] = 0.123456789
    pbh_mutation_checks["pbh_seed_max_deficit"] = _expect_provenance_error(lambda: _validate_pbh_input(seed_deficit))

    payload_mutation = copy.deepcopy(result_without_integrity)
    payload_mutation["echo"]["exposure_curve"][0]["a90_raw"] = 1.0
    payload_digest_check = _payload_digest(payload_mutation) != _payload_digest(result_without_integrity)
    blocker_mutation = copy.deepcopy(result_without_integrity)
    blocker_mutation["s8"]["normalization_probe"]["status"] = "MUTATED_BLOCKER"
    blocker_digest_check = _payload_digest(blocker_mutation) != _payload_digest(result_without_integrity)
    checks = {
        **echo_mutation_checks,
        **pbh_mutation_checks,
        "complete_payload_catches_numeric_mutation": payload_digest_check,
        "complete_payload_catches_blocker_mutation": blocker_digest_check,
    }
    return {
        "status": "PASS_FAIL_CLOSED_FORMER_ADVERSARIAL_PROBES" if all(checks.values()) else "FAIL_MUTATION_ACCEPTED",
        "checks": checks,
        "echo_q_shape": [len(base_rows), len(base_rows[0]["efficiency"])],
        "pbh_rows_checked": ["jwst_calibrated_seed_band.max_row", "internal_budget_expanded_boundary.max_row"],
        "payload_digest_scope": "complete JSON payload and canonical flattened CSV rows",
    }


def build_result() -> dict[str, Any]:
    """Read frozen inputs and build the complete P3-S3 ledger."""

    echo_artifact = _read_json(ECHO_RESULT_PATH)
    s8_artifact = _read_json(S8_RESULT_PATH)
    pbh_artifact = _read_json(PBH_RESULT_PATH)
    echo_projection = _echo_reach(echo_artifact)
    s8_projection = _s8_reach(s8_artifact)
    pbh_projection = _pbh_reach(pbh_artifact)
    target_controls = _target_leakage_controls(echo_artifact, s8_artifact, pbh_artifact)
    result = _jsonable(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE_WITH_PROSPECTIVE_SENSITIVITY_AND_PENDING_NS_MERGE",
            "scope": "Prospective critical-boundary ledger; no new observation, fit, or cross-worker NS import",
            "producer_identity": {
                "echo": echo_projection["producer_identity"],
                "s8": s8_projection["producer_identity"],
                "pbh": pbh_projection["producer_identity"],
            },
            "target_leakage_controls": target_controls,
            "echo": echo_projection,
            "s8": s8_projection,
            "pbh": pbh_projection,
            "ns_hartle_pending_merge": {
                "status": "blocked",
                "producer": "verification/nvg_hartle_slow_rotation.py (P3-S1 target not imported)",
                "phase_3_artifact": "P3-S1 frozen NS/Hartle forecast table",
                "merge_rule": "Phase 3 gate merges P3-S1 only after its own terminal provenance/convergence proof; P3-S3 has no cross-worker dependency and performs no import.",
                "falsifier": "A P3-S1 re-gate failure or unresolved branch/interpolation identity blocks any NS/Hartle target merge.",
                "minimal_prospective_requirement": "P3-S1 final report/artifact with branch IDs, convergence, and source-to-artifact identity; merge in P3-S4/P4 gate.",
                "dominant_uncertainty": "P3-S1 is not available to this step by contract.",
                "scope": "pending merge only; no NS/Hartle value appears in this ledger",
            },
            "source_artifact_provenance": {
                "echo_result": _file_pin(ECHO_RESULT_PATH, "read-only P2 final artifact"),
                "s8_result": _file_pin(S8_RESULT_PATH, "read-only P1 final artifact"),
                "pbh_result": _file_pin(PBH_RESULT_PATH, "read-only P1 final artifact"),
                "p3_producer": _file_pin(P3_SOURCE_PATH, "live P3-S7 producer"),
                "p3_focused_test": _file_pin(P3_TEST_PATH, "live P3-S7 focused test"),
            },
        }
    )
    result["adversarial_mutations"] = _adversarial_controls(echo_artifact, pbh_artifact, result)
    return _attach_payload_integrity(result)


def _csv_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten decisive ledger rows into a review-friendly CSV."""

    rows: list[dict[str, Any]] = []
    def add(projection: str, key: str, value: Any, status: str, scope: str, falsifier: str, requirement: str, uncertainty: str) -> None:
        rows.append(
            {
                "projection": projection,
                "quantity": key,
                "value": value,
                "status": status,
                "scope": scope,
                "falsifier": falsifier,
                "minimal_prospective_requirement": requirement,
                "dominant_uncertainty": uncertainty,
            }
        )

    echo_result = result["echo"]
    for row in echo_result["exposure_curve"]:
        add("echo", f"a90_raw_N{row['selected_event_count']}", row["a90_raw"], row["a90_status"], row["scope"], echo_result["falsifier"], echo_result["minimal_prospective_requirement"], echo_result["dominant_uncertainty"])
        add("echo", f"exposure_snr2_N{row['selected_event_count']}", row["exposure_network_snr2"], row["a90_status"], row["scope"], echo_result["falsifier"], echo_result["minimal_prospective_requirement"], echo_result["dominant_uncertainty"])
    s8_result = result["s8"]
    for row in s8_result["boundary_expansion"]["rows"]:
        add("s8", f"one_sigma_overlap_omega_{row['omega_m']:.6f}", row["one_sigma_overlay_overlap"], s8_result["status"], s8_result["scope"], s8_result["falsifier"], s8_result["minimal_prospective_requirement"], s8_result["dominant_uncertainty"])
    pbh_result = result["pbh"]
    for row in pbh_result["seed_band_requirements"]["rows"]:
        add("pbh", f"required_rate_product_N{row['cycle']}_n{row['number_density_mpc3']:.1e}", row["required_rate_product"], row["status"], pbh_result["scope"], pbh_result["falsifier"], pbh_result["minimal_prospective_requirement"], pbh_result["dominant_uncertainty"])
    return rows


def assert_artifact_provenance(result: dict[str, Any]) -> None:
    """Verify complete payload integrity and live input/projection identities."""

    try:
        _assert_payload_integrity(result)
    except (ProvenanceError, TypeError, ValueError, KeyError) as exc:
        raise AssertionError(str(exc)) from exc
    if result.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("unsupported P3-S7 result schema")
    if result.get("status") != "COMPLETE_WITH_PROSPECTIVE_SENSITIVITY_AND_PENDING_NS_MERGE":
        raise AssertionError("P3-S3 scope/status drifted")
    source_bundle = result.get("source_artifact_provenance", {})
    for key, path in (("p3_producer", P3_SOURCE_PATH), ("p3_focused_test", P3_TEST_PATH)):
        entry = source_bundle.get(key, {})
        if entry.get("path") != _relative(path) or entry.get("sha256") != sha256_file(path):
            raise AssertionError(f"P3 source provenance drift: {key}")
    q_integrity = result.get("echo", {}).get("q_payload_integrity", {})
    if q_integrity.get("status") != "PASS_RECOMPUTED_LIVE_FULL_MATRIX" or q_integrity.get("matches_live") is not True:
        raise AssertionError("echo q payload is not marked live-recomputed")
    if q_integrity.get("shape") != [259, 15] or q_integrity.get("artifact_q_sha256") != q_integrity.get("live_q_sha256"):
        raise AssertionError("echo q payload integrity metadata drifted")
    echo_blocks = result.get("echo", {}).get("observational_blocks", {})
    if echo_blocks.get("status") != "BLOCKED_MISSING_STRAIN_POSTERIOR_NOISE_PRODUCTS" or echo_blocks.get("time_slide_independence", {}).get("status") != "BLOCKED_NO_EMPIRICAL_TIME_SLIDE_BANK":
        raise AssertionError("echo observational/time-slide blocks drifted")
    s8 = result.get("s8", {})
    if s8.get("maintained_slice", {}).get("one_sigma_overlap_count") != 0:
        raise AssertionError("maintained Omega_m slice no-go drifted")
    if s8.get("maintained_slice", {}).get("scoped_scalar_no_go") is not True or s8.get("global_statement_status") != "BLOCKED_NO_SOURCED_JOINT_LIKELIHOOD":
        raise AssertionError("S8 global/no-go scope drifted")
    if s8.get("legacy_overlay_scope", {}).get("provenance_status") != "unknown_provenance_sensitivity_overlay" or s8.get("legacy_overlay_scope", {}).get("observed_likelihood") is not False:
        raise AssertionError("S8 DESI overlay provenance scope drifted")
    for section in (s8.get("boundary_expansion", {}), s8.get("resolution_probe", {}), s8.get("normalization_probe", {})):
        if section.get("critical_boundary_status") != "NOT_IDENTIFIABLE_GRID_AND_NORMALIZATION_DEPENDENT" and section.get("status") != "NOT_IDENTIFIABLE_GRID_AND_NORMALIZATION_DEPENDENT":
            raise AssertionError("S8 critical boundary was not withdrawn")
    if s8.get("normalization_probe", {}).get("normalization_changes_one_sigma_overlap") is not True:
        raise AssertionError("S8 normalization convention probe did not survive")
    pbh = result.get("pbh", {})
    if pbh.get("target_leakage", {}).get("scan_maxima_copied_from_frozen_artifact") is not False:
        raise AssertionError("PBH max rows remain copied from frozen artifact")
    if pbh.get("max_row_integrity", {}).get("status") != "PASS_LIVE_RECOMPUTED_COMPLETE_ROWS":
        raise AssertionError("PBH max-row integrity is not live-recomputed")
    adversarial = result.get("adversarial_mutations", {})
    if adversarial.get("status") != "PASS_FAIL_CLOSED_FORMER_ADVERSARIAL_PROBES" or not all(adversarial.get("checks", {}).values()):
        raise AssertionError("former P3-S4 adversarial mutation probe failed")

    # Re-authenticate the source inputs used for the two previously failing
    # producer payloads.  This catches an artifact replacement whose P3 digest
    # was regenerated outside the declared live evaluator.
    echo_artifact = _read_json(ECHO_RESULT_PATH)
    _, _, _, echo_details = _validate_echo_input(echo_artifact)
    if echo_details["live_q_payload_digest"] != q_integrity.get("live_q_sha256"):
        raise AssertionError("P3 echo q digest does not match live P2 evaluator")
    pbh_artifact = _read_json(PBH_RESULT_PATH)
    live_pbh = _validate_pbh_input(pbh_artifact)["live_scans"]
    envelope = pbh["frozen_expanded_envelope_boundary"]
    expected_max = live_pbh["budget_expanded"]["max_row"]
    if not _values_equal(envelope["cycle"], expected_max["cycle"]) or not _values_equal(envelope["fraction_dm"], expected_max["fraction_dm"]) or not _values_equal(envelope["max_amplitude_strain"], expected_max["amplitude_strain"]):
        raise AssertionError("P3 PBH envelope does not match live max row")


def write_artifacts(result: dict[str, Any], *, json_path: Path | None = None, csv_path: Path | None = None, figure_path: Path | None = None) -> tuple[Path, Path, Path]:
    """Write uniquely named JSON, CSV, and diagnostic figure artifacts."""

    _assert_payload_integrity(result)
    json_path = json_path or RESULT_PATH
    csv_path = csv_path or CSV_PATH
    figure_path = figure_path or FIGURE_PATH
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    rows = _csv_rows(result)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)
    echo_rows = result["echo"]["exposure_curve"]
    axes[0].loglog([row["exposure_network_snr2"] for row in echo_rows], [row["a90_raw"] for row in echo_rows], marker="o", color="#264653")
    axes[0].set(xlabel=r"exposure $\sum\rho_i^2$", ylabel="synthetic A90", title="Echo reach (sensitivity)")
    axes[0].grid(True, which="both", alpha=0.25)
    s8_rows = result["s8"]["boundary_expansion"]["rows"]
    axes[1].plot([row["omega_m"] for row in s8_rows], [row["one_sigma_overlay_overlap"] for row in s8_rows], marker="o", color="#457b9d")
    axes[1].axvline(S8_MAINTAINED_OMEGA, color="#d1495b", linestyle="--", label="maintained 0.315")
    axes[1].set(xlabel=r"$\Omega_m$", ylabel=r"1$\sigma$ overlap cells", title="S8 boundary (grid only)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.25)
    pbh_rows = result["pbh"]["expanded_jwst_upper_density_scan"]["rows"]
    axes[2].semilogy([row["cycle"] for row in pbh_rows], [row["unit_rate_amplitude_strain"] for row in pbh_rows], marker="o", color="#2a9d8f")
    axes[2].axhline(result["pbh"]["benchmark"]["amplitude_strain"], color="#d1495b", linestyle="--", label="internal PTA benchmark")
    axes[2].axvspan(min(PBH_SEED_CYCLES) - 0.35, max(PBH_SEED_CYCLES) + 0.35, color="#f4a261", alpha=0.25, label="JWST rung band")
    axes[2].set(xlabel="mass-ladder cycle", ylabel="strain at JWST upper density", title="PBH rung reach (sensitivity)")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(True, which="both", alpha=0.25)
    fig.suptitle("P3-S3 prospective falsification reach; no observational claim")
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return json_path, csv_path, figure_path


# Keep the naming used by the Phase 1/2 producers available to independent
# gates and downstream tests.
assert_result_provenance = assert_artifact_provenance


def _terminal_summary(result: dict[str, Any], paths: tuple[Path, Path, Path]) -> str:
    nearest = result["s8"]["boundary_expansion"]["nearest_grid_boundary"]
    first_pbh = result["pbh"]["expanded_jwst_upper_density_scan"]["first_reachable_rung"]
    return "\n".join(
        (
            "P3-S3 PROSPECTIVE FALSIFICATION REACH",
            f"STATUS: {result['status']}",
            f"Echo selected events/exposure: {result['echo']['selected_event_count']} / {result['echo']['exposure_curve'][-1]['exposure_network_snr2']:.6g}; raw union={result['echo']['numerical_union']['raw_interval']}",
            f"S8 nearest crossing grid cell: {None if nearest is None else nearest['omega_m']} (resolution-dependent={result['s8']['resolution_probe']['boundary_resolution_dependent']})",
            f"S8 boundary semantics: {result['s8']['normalization_probe']['status']}",
            f"PBH seed-band min required rate product: {result['pbh']['seed_band_requirements']['minimum_required_rate_product']:.6g}; first expanded JWST-upper-density reachable rung: {first_pbh}",
            f"Integrity: echo_q={result['echo']['q_payload_integrity']['status']}; PBH_max={result['pbh']['max_row_integrity']['status']}; mutations={result['adversarial_mutations']['status']}",
            f"NS/Hartle: {result['ns_hartle_pending_merge']['status']} (P3-S1 pending merge; no import)",
            f"JSON: {paths[0]}",
            f"CSV: {paths[1]}",
            f"FIGURE: {paths[2]}",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="build the ledger without writing JSON/CSV/PNG")
    args = parser.parse_args(argv)
    result = build_result()
    assert_artifact_provenance(result)
    paths = (RESULT_PATH, CSV_PATH, FIGURE_PATH)
    if not args.no_write:
        paths = write_artifacts(result)
    print(_terminal_summary(result, paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
