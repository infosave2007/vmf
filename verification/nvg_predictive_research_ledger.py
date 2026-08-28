#!/usr/bin/env python3
"""Build the fail-closed machine ledger for the predictive-research program.

The ledger is an integration surface, not a second scientific producer.  It
loads the final Phase 1--3 artifacts, authenticates them against their live
producers, and copies only the statuses and values authorised by the P3-S8
gate.  In particular, conditional, synthetic, sensitivity and blocked rows
always carry zero independent-evidence weight.

Run from any working directory with::

    python3 verification/nvg_predictive_research_ledger.py

The output is deterministic JSON.  Any source/artifact drift, changed central
value, altered blocker, changed numerical band, q-table/max-row mutation, or
S8 boundary-status mutation fails closed before the output is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_beta_hyperon_urca_audit as beta
import nvg_cooling_gmode_dependency_audit as cooling
import nvg_echo_hierarchical_upper_limit as echo
import nvg_hartle_slow_rotation as hartle
import nvg_pbh_nanograv_audit as pbh_audit
import nvg_ns_frozen_forecasts as forecast
import nvg_ns_predictive_audit as ns_audit
import nvg_prospective_falsification_reach as prospective
import nvg_s8_no_go as s8


SCHEMA_VERSION = "P4-S1-predictive-research-ledger-v1"
AUDIT_ID = "P4-S1"
RESULT_PATH = HERE / "predictive_research_ledger.json"

P1_NS_RESULT_PATH = HERE / "nvg_ns_predictive_audit_p1s7_results.json"
P1_NS_OLD_RESULT_PATH = HERE / "nvg_ns_predictive_audit_results.json"
P1_NS_P1S5_RESULT_PATH = HERE / "nvg_ns_predictive_audit_p1s5_results.json"
S8_RESULT_PATH = HERE / "nvg_s8_no_go_map.json"
P2_HARTLE_RESULT_PATH = HERE / "nvg_hartle_slow_rotation_p2s5_results.json"
P2_HARTLE_OLD_RESULT_PATH = HERE / "nvg_hartle_slow_rotation_p2s1_results.json"
P2_BETA_RESULT_PATH = HERE / "nvg_beta_hyperon_urca_audit_p2s2_results.json"
P2_ECHO_RESULT_PATH = HERE / "nvg_echo_hierarchical_upper_limit_p2s3_results.json"
P3_NS_RESULT_PATH = HERE / "nvg_ns_frozen_forecasts_p3s1_results.json"
P3_NS_CSV_PATH = HERE / "nvg_ns_frozen_forecasts_p3s1.csv"
P3_COOLING_RESULT_PATH = HERE / "nvg_cooling_gmode_dependency_audit_p3s2_results.json"
P3_REACH_RESULT_PATH = HERE / "nvg_prospective_falsification_reach_p3s3_results.json"
P3_REACH_CSV_PATH = HERE / "p3_s3_prospective_reach.csv"
J0737_INPUT_PATH = HERE / "data" / "nvg_hartle_p2s1_j0737a_mass.json"
ECHO_CONFIG_PATH = HERE / "data" / "p2_s3_hierarchical_echo" / "analysis_config.json"
ECHO_MANIFEST_PATH = HERE / "data" / "p2_s3_hierarchical_echo" / "provenance_manifest.json"
GWTC_CATALOG_PATH = HERE / "data" / "gwtc_events.csv"
# The P1 PBH audit is promoted byte-for-byte into the tracked verification
# surface.  Keep the expected digests here so a replacement or stale copy
# fails closed before any claim is assembled.  The digests were established by
# a live producer regeneration and byte comparison with the immutable P1
# control artifact; that ignored control tree is intentionally not opened by
# this ledger at runtime.
PBH_RESULT_PATH = HERE / "P1-S3-pbh-nanograv-audit.json"
PBH_FIGURE_PATH = HERE / "P1-S3-pbh-nanograv-boundary.png"
PBH_RESULT_SHA256 = "810243886f2cbee5de18845ec5c910bd9b3fbe0e632fe8dbe352efab18e616f7"
PBH_FIGURE_SHA256 = "b38461668a090eef2542badbcb4bb8d5735a5791571015c1fdc544673a5b42e0"


def _jsonable(value: Any) -> Any:
    """Return a JSON-native, deterministic representation."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    # Avoid importing numpy solely for scalar conversion.  Numpy scalars expose
    # ``item`` and are used by the Phase 1/S8 builders.
    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_payload_digest(payload: Mapping[str, Any]) -> str:
    """Hash a payload with its self-referential integrity block removed."""

    unsigned = copy.deepcopy(_jsonable(payload))
    if isinstance(unsigned, dict):
        unsigned.pop("ledger_integrity", None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise AssertionError(f"required predictive file is unreadable: {path}") from exc


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"required predictive artifact is missing: {_relative(path)}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError(f"invalid predictive JSON {_relative(path)}: {exc}") from exc
    if not isinstance(value, dict):
        raise AssertionError(f"predictive JSON must be an object: {_relative(path)}")
    return value


def _pin(path: Path, role: str, *, required: bool = True) -> dict[str, Any]:
    if required or path.exists():
        return {"path": _relative(path), "role": role, "sha256": sha256_file(path)}
    return {"path": _relative(path), "role": role, "sha256": None, "status": "missing_optional"}


def _assert_close(actual: Any, expected: float, name: str, *, rel: float = 1.0e-12, abs_tol: float = 1.0e-12) -> None:
    try:
        if not math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=abs_tol):
            raise AssertionError(f"{name} changed: {actual!r} != {expected!r}")
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{name} is not numeric") from exc


def _assert_p1_ns_core(payload: Mapping[str, Any]) -> None:
    """Assert the P3-S8-authorized Phase-1 central values and counts."""

    if payload.get("schema_version") != ns_audit.SCHEMA_VERSION:
        raise AssertionError("Phase-1 final NS schema drift")
    if payload.get("status") != "CONDITIONAL_IN_SAMPLE":
        raise AssertionError("Phase-1 NS status drift")
    canonical = payload.get("canonical", {})
    _assert_close(canonical.get("M_max"), 2.047950740578197, "Phase-1 M_max")
    _assert_close(canonical.get("R_1.4"), 12.550001000000044, "Phase-1 R_1.4")
    _assert_close(canonical.get("Lambda_1.4"), 519.4223807918132, "Phase-1 Lambda_1.4")
    summary = payload.get("grid_summary", {})
    expected_counts = {
        "declared_rows": 450,
        "evaluated_rows": 450,
        "screening_valid_rows": 450,
        "valid_rows": 87,
        "survivor_count": 83,
        "screening_survivor_count": 115,
        "unresolved_candidate_rows": 33,
        "screening_only_rows": 330,
        "non_evidence_rows": 363,
    }
    for key, expected in expected_counts.items():
        if summary.get(key) != expected:
            raise AssertionError(f"Phase-1 grid count {key} changed")
    boundary = payload.get("boundary_optimum", {})
    if boundary.get("classification") != "GRID_SENSITIVITY_ONLY":
        raise AssertionError("Phase-1 boundary status drift")
    if boundary.get("robust_boundary_optimum") is not False:
        raise AssertionError("Phase-1 robust-boundary flag was promoted")
    if boundary.get("zero_delta_survivor_count") != 9 or boundary.get("positive_delta_survivor_count") != 74:
        raise AssertionError("Phase-1 zero/positive survivor counts changed")
    continuity = payload.get("zero_limit_continuity", {})
    if continuity.get("status") != "PASS":
        raise AssertionError("Phase-1 zero-limit continuity status drift")
    _assert_close(continuity.get("max_relative_tidal_delta"), 2.2432552239096867e-05, "Phase-1 tidal continuity")
    m_omega = payload.get("m_omega_dependency", {})
    if m_omega.get("status") != "BLOCKED_NO_PHYSICAL_DEPENDENCY" or m_omega.get("independent_evidence_weight") != 0.0:
        raise AssertionError("M_Omega blocker/evidence semantics drifted")
    ns_audit.assert_solver_provenance_artifact(dict(payload))


def _assert_hartle_core(payload: Mapping[str, Any]) -> None:
    hartle.assert_artifact_provenance(dict(payload))
    if payload.get("audit") != "P2-S5" or payload.get("public_status") != "DERIVED_CONDITIONAL_ALLOWED_BRANCHES_LOW_MASS_UNRESOLVED":
        raise AssertionError("P2-S5 Hartle status drift")
    regression = payload.get("canonical_regression", {})
    if regression.get("status") != "PASS":
        raise AssertionError("P2-S5 canonical regression is not PASS")
    frozen = regression.get("frozen", {})
    _assert_close(frozen.get("M_max_msun"), 2.047950740578197, "Hartle frozen M_max")
    _assert_close(frozen.get("R14_km"), 12.550001000000044, "Hartle frozen R_1.4")
    _assert_close(frozen.get("Lambda14"), 519.4223807918132, "Hartle frozen Lambda_1.4")
    j0737 = payload.get("j0737a", {})
    if j0737.get("status") != "derived_conditional" or j0737.get("branch_id") != 1 or j0737.get("gap_crossed") is not False:
        raise AssertionError("Hartle J0737A branch/gap semantics drifted")


def _assert_s8_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("S8 artifact schema drift")
    if payload.get("status") != "SCOPED_NO_GO_MAINTAINED_OMEGA_M; EXPANDED_OMEGA_M_SENSITIVITY_OVERLAP":
        raise AssertionError("S8 scoped status drift")
    # The S8 producer has no self-digest; compare the complete normalized
    # result to a fresh live construction so maps, counts, and formulas cannot
    # be replaced by a stale/static table.
    live = _jsonable(s8.build_result())
    if _canonical_json(payload) != _canonical_json(live):
        raise AssertionError("S8 artifact does not equal live recomputation")
    boundary = payload.get("boundary_conclusion", {})
    if boundary.get("scoped_no_go") is not True or boundary.get("expanded_omega_m_overlay_overlap_count") != 20:
        raise AssertionError("S8 overlap semantics changed")


def _assert_authorized_inputs() -> dict[str, dict[str, Any]]:
    """Load and authenticate the final P1--P3 artifacts.

    The returned mapping is deliberately kept private to the builder.  Claims
    below copy only the small P3-S8-authorized subset.
    """

    p1 = _read_json(P1_NS_RESULT_PATH)
    _assert_p1_ns_core(p1)

    hartle_result = _read_json(P2_HARTLE_RESULT_PATH)
    _assert_hartle_core(hartle_result)

    beta_result = _read_json(P2_BETA_RESULT_PATH)
    beta.assert_artifact_provenance(beta_result)
    if beta_result.get("status") != "COMPLETE_WITH_BLOCKED_PHYSICAL_HYPERON_PHASE_URCA_DEPENDENCIES":
        raise AssertionError("P2-S2 hyperon/Urca status drift")

    echo_result = _read_json(P2_ECHO_RESULT_PATH)
    echo.assert_result_provenance(echo_result)

    s8_result = _read_json(S8_RESULT_PATH)
    _assert_s8_artifact(s8_result)

    forecast_result = _read_json(P3_NS_RESULT_PATH)
    forecast.assert_artifact_provenance(forecast_result)
    if forecast_result.get("status") != "FROZEN_CONDITIONAL_FORECAST":
        raise AssertionError("P3-S5 frozen forecast status drift")

    cooling_result = _read_json(P3_COOLING_RESULT_PATH)
    cooling.assert_artifact_provenance(cooling_result)

    reach_result = _read_json(P3_REACH_RESULT_PATH)
    # P3-S7's validator predates the tracked PBH promotion and keeps its own
    # source-path constant.  Redirect that one read-only validation call to the
    # tracked byte-identical artifact; restore the imported module immediately
    # so no global path mutation leaks into callers or tests.
    previous_pbh_path = prospective.PBH_RESULT_PATH
    prospective.PBH_RESULT_PATH = PBH_RESULT_PATH
    try:
        prospective.assert_artifact_provenance(reach_result)
    finally:
        prospective.PBH_RESULT_PATH = previous_pbh_path

    pbh_artifact = _read_json(PBH_RESULT_PATH)
    if sha256_file(PBH_RESULT_PATH) != PBH_RESULT_SHA256:
        raise AssertionError("tracked P1 PBH artifact digest drift")
    if sha256_file(PBH_FIGURE_PATH) != PBH_FIGURE_SHA256:
        raise AssertionError("tracked P1 PBH figure digest drift")
    # Authenticate every PBH field against a fresh live producer result.  This
    # compares the complete JSON payload (not just the extrema consumed below)
    # while leaving the producer's canonical output path untouched.
    live_pbh_artifact = pbh_audit.run_audit()
    if _canonical_json(pbh_artifact) != _canonical_json(live_pbh_artifact):
        raise AssertionError("tracked P1 PBH artifact does not equal live regeneration")
    scan_contract = pbh_artifact.get("scan_contract", {})
    if scan_contract.get("external_exclusions_used") is not False:
        raise AssertionError("PBH source artifact external-exclusion control drift")

    return {
        "p1_ns": p1,
        "hartle": hartle_result,
        "beta": beta_result,
        "echo": echo_result,
        "s8": s8_result,
        "forecast": forecast_result,
        "cooling": cooling_result,
        "reach": reach_result,
        "pbh_artifact": pbh_artifact,
    }


def _source_provenance() -> dict[str, Any]:
    """Pin every source/config/data/artifact consumed by the ledger."""

    # P3-S7's own provenance bundle already authenticates the PBH artifact and
    # all live PBH producers.  The explicit pins here keep the integration
    # ledger self-contained without registering private control files.
    paths = {
        "p1_ns_producer": (HERE / "nvg_ns_predictive_audit.py", "P1 final NS audit producer"),
        "p1_ns_test": (HERE / "test_p1_s9_ns_provenance.py", "P1 final NS provenance test"),
        "p1_ns_result": (P1_NS_RESULT_PATH, "P1-S7/P1-S9 final NS artifact"),
        "s8_producer": (HERE / "nvg_s8_no_go.py", "P1 S8 map producer"),
        "s8_canonical_producer": (HERE / "nvg_black_hole_entropy.py", "maintained S8 producer"),
        "s8_result": (S8_RESULT_PATH, "P1-S2 final S8 artifact"),
        "pbh_producer": (HERE / "nvg_pbh_nanograv_audit.py", "P1 PBH audit producer"),
        "pbh_result": (PBH_RESULT_PATH, "P1-S3 tracked PBH artifact; byte-identical promotion of final control"),
        "pbh_figure": (PBH_FIGURE_PATH, "P1-S3 tracked PBH boundary figure; byte-identical promotion of final control"),
        "hartle_producer": (HERE / "nvg_hartle_slow_rotation.py", "P2-S5 Hartle producer"),
        "hartle_test": (HERE / "test_p2_hartle_slow_rotation.py", "P2 Hartle focused test"),
        "hartle_result": (P2_HARTLE_RESULT_PATH, "P2-S5 final Hartle artifact"),
        "j0737_input": (J0737_INPUT_PATH, "pinned J0737A timing input"),
        "beta_producer": (HERE / "nvg_beta_hyperon_urca_audit.py", "P2-S2 composition producer"),
        "beta_test": (HERE / "test_p2_beta_hyperon_urca_audit.py", "P2-S2 focused test"),
        "beta_result": (P2_BETA_RESULT_PATH, "P2-S2 final composition artifact"),
        "echo_producer": (HERE / "nvg_echo_hierarchical_upper_limit.py", "P2-S8 echo producer"),
        "echo_test": (HERE / "test_p2_echo_hierarchical_upper_limit.py", "P2 echo focused test"),
        "echo_result": (P2_ECHO_RESULT_PATH, "P2-S8 final echo artifact"),
        "echo_config": (ECHO_CONFIG_PATH, "pinned echo analysis configuration"),
        "echo_manifest": (ECHO_MANIFEST_PATH, "pinned echo provenance manifest"),
        "gwtc_catalog": (GWTC_CATALOG_PATH, "public GWTC catalog selection input"),
        "forecast_producer": (HERE / "nvg_ns_frozen_forecasts.py", "P3-S5 forecast producer"),
        "forecast_test": (HERE / "test_p3_ns_frozen_forecasts.py", "P3 forecast focused test"),
        "forecast_result": (P3_NS_RESULT_PATH, "P3-S5 final forecast artifact"),
        "forecast_csv": (P3_NS_CSV_PATH, "P3-S5 forecast table"),
        "cooling_producer": (HERE / "nvg_cooling_gmode_dependency_audit.py", "P3-S6 cooling/g-mode producer"),
        "cooling_test": (HERE / "test_p3_cooling_gmode_dependency_audit.py", "P3 cooling/g-mode focused test"),
        "cooling_result": (P3_COOLING_RESULT_PATH, "P3-S6 final cooling/g-mode artifact"),
        "reach_producer": (HERE / "nvg_prospective_falsification_reach.py", "P3-S7 reach producer"),
        "reach_test": (HERE / "test_p3_prospective_falsification_reach.py", "P3 reach focused test"),
        "reach_result": (P3_REACH_RESULT_PATH, "P3-S7 final prospective artifact"),
        "reach_csv": (P3_REACH_CSV_PATH, "P3-S7 reach table"),
    }
    return {key: _pin(path, role) for key, (path, role) in paths.items()}


def _inventory() -> dict[str, Any]:
    """List predictive-program surfaces without importing immutable controls."""

    producers = [
        "verification/nvg_ns_predictive_audit.py",
        "verification/nvg_s8_no_go.py",
        "verification/nvg_pbh_nanograv_audit.py",
        "verification/nvg_hartle_slow_rotation.py",
        "verification/nvg_beta_hyperon_urca_audit.py",
        "verification/nvg_echo_hierarchical_upper_limit.py",
        "verification/nvg_ns_frozen_forecasts.py",
        "verification/nvg_cooling_gmode_dependency_audit.py",
        "verification/nvg_prospective_falsification_reach.py",
    ]
    tests = [
        "verification/test_p1_s1_ns_predictive_audit.py",
        "verification/test_p1_s2_s8_no_go.py",
        "verification/test_p1_s3_pbh_nanograv_audit.py",
        "verification/test_p1_s5_ns_repair.py",
        "verification/test_p1_s7_ns_repair.py",
        "verification/test_p1_s9_ns_provenance.py",
        "verification/test_p2_beta_hyperon_urca_audit.py",
        "verification/test_p2_echo_hierarchical_upper_limit.py",
        "verification/test_p2_hartle_slow_rotation.py",
        "verification/test_p3_cooling_gmode_dependency_audit.py",
        "verification/test_p3_ns_frozen_forecasts.py",
        "verification/test_p3_prospective_falsification_reach.py",
        "verification/test_p4_predictive_research_integration.py",
    ]
    artifacts = [
        _relative(path)
        for path in (
            P1_NS_RESULT_PATH,
            P1_NS_OLD_RESULT_PATH,
            P1_NS_P1S5_RESULT_PATH,
            HERE / "fig_ns_predictive_audit.png",
            HERE / "fig_ns_predictive_audit_p1s5.png",
            HERE / "fig_ns_predictive_audit_p1s7.png",
            S8_RESULT_PATH,
            HERE / "fig_s8_no_go_map.png",
            PBH_RESULT_PATH,
            PBH_FIGURE_PATH,
            P2_HARTLE_OLD_RESULT_PATH,
            P2_HARTLE_RESULT_PATH,
            HERE / "fig_hartle_slow_rotation_p2s1.png",
            HERE / "fig_hartle_slow_rotation_p2s5.png",
            P2_BETA_RESULT_PATH,
            HERE / "fig_p2_s2_beta_hyperon_urca_audit.png",
            P2_ECHO_RESULT_PATH,
            HERE / "fig_echo_hierarchical_upper_limit_p2s3.png",
            P3_NS_RESULT_PATH,
            P3_NS_CSV_PATH,
            HERE / "fig_ns_frozen_forecasts_p3s1.png",
            P3_COOLING_RESULT_PATH,
            HERE / "fig_p3_s2_cooling_gmode_dependency_audit.png",
            P3_REACH_RESULT_PATH,
            P3_REACH_CSV_PATH,
            HERE / "fig_p3_s3_prospective_reach.png",
            RESULT_PATH,
        )
    ]
    inputs = [
        _relative(J0737_INPUT_PATH),
        _relative(GWTC_CATALOG_PATH),
        _relative(ECHO_CONFIG_PATH),
    ]
    manifests = [
        "verification/data/p2_s3_hierarchical_echo/provenance_manifest.json",
        "verification/artifact_manifest.json",
        "verification/data/provenance.json",
        "verification/registry.json",
    ]
    return {
        "predictive_producers": producers,
        "predictive_tests": tests,
        "durable_artifacts": artifacts,
        "source_inputs": inputs,
        "manifests": manifests,
        "counts": {
            "producers": len(producers),
            "tests": len(tests),
            "durable_artifacts": len(artifacts),
            "source_inputs": len(inputs),
            "manifests": len(manifests),
        },
        "immutable_controls_excluded": True,
    }


def _claims(payloads: Mapping[str, Mapping[str, Any]], provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract only the statuses/values explicitly allowed by P3-S8."""

    p1 = payloads["p1_ns"]
    hartle_result = payloads["hartle"]
    beta_result = payloads["beta"]
    echo_result = payloads["echo"]
    s8_result = payloads["s8"]
    frozen = payloads["forecast"]
    cooling_result = payloads["cooling"]
    reach = payloads["reach"]
    pbh_artifact = payloads["pbh_artifact"]
    j0737 = frozen["j0737a"]
    forecast_row = j0737["forecast"]
    bands = frozen["falsification_bands"]
    null = cooling_result["buoyancy"]["null_limit"]
    physical_no_go = cooling_result["physical_no_go"]
    pbh = reach["pbh"]
    pbh_seed = pbh["seed_band_requirements"]
    pbh_expanded = pbh["expanded_jwst_upper_density_scan"]
    pbh_integrity = pbh["max_row_integrity"]
    s8_reach = reach["s8"]
    echo_reach = reach["echo"]

    claims: list[dict[str, Any]] = [
        {
            "id": "ns_canonical_selection",
            "domain": "NS/selection",
            "status": "conditional_in_sample",
            "source_artifact_status": p1["status"],
            "evidence_weight": 0.0,
            "values": {
                "M_max_msun": p1["canonical"]["M_max"],
                "R_1.4_km": p1["canonical"]["R_1.4"],
                "Lambda_1.4": p1["canonical"]["Lambda_1.4"],
                "declared_rows": p1["grid_summary"]["declared_rows"],
                "converged_rows": p1["grid_summary"]["valid_rows"],
                "survivor_count": p1["grid_summary"]["survivor_count"],
                "unresolved_candidate_rows": p1["grid_summary"]["unresolved_candidate_rows"],
                "screening_only_rows": p1["grid_summary"]["screening_only_rows"],
            },
            "boundary_status": p1["boundary_optimum"]["classification"],
            "source": provenance["p1_ns_result"],
        },
        {
            "id": "ns_hartle_j0737a",
            "domain": "NS/Hartle",
            "status": "derived_conditional",
            "source_artifact_status": frozen["observable_status"],
            "evidence_weight": 0.0,
            "scope": "canonical zero-jump EOS; first-order Hartle plus same-background Hinderer; pressure-ordered branch 1",
            "branch_id": j0737["branch_id"],
            "gap_crossed": j0737["gap_crossed"],
            "values": {
                "mass_msun": j0737["mass_msun"],
                "R_A_km": forecast_row["R_km"],
                "compactness": forecast_row["compactness"],
                "Lambda_A": forecast_row["Lambda"],
                "I_A_g_cm2": forecast_row["I"],
                "Ibar_A": forecast_row["Ibar"],
                "central_pressure_mev_fm3": forecast_row["central_pressure"],
            },
            "supported_display": {
                "I_A_g_cm2": {
                    "lower": bands["I_A"]["display_lower"],
                    "upper": bands["I_A"]["display_upper"],
                    "digits": bands["I_A"]["display_digits"],
                },
                "Lambda_A": {
                    "lower": bands["Lambda_A"]["display_lower"],
                    "upper": bands["Lambda_A"]["display_upper"],
                    "digits": bands["Lambda_A"]["display_digits"],
                },
                "R_A_km": {
                    "lower": bands["R_A"]["display_lower"],
                    "upper": bands["R_A"]["display_upper"],
                    "digits": bands["R_A"]["display_digits"],
                },
            },
            "raw_bands": {
                "I_A_g_cm2": [bands["I_A"]["lower"], bands["I_A"]["upper"]],
                "Lambda_A": [bands["Lambda_A"]["lower"], bands["Lambda_A"]["upper"]],
                "R_A_km": [bands["R_A"]["lower"], bands["R_A"]["upper"]],
            },
            "blocked": {
                "Q": j0737["Q"]["status"],
                "inverse_mass": j0737["inverse_mass"]["status"],
                "phase1_wider_envelope": frozen["phase1_candidate_sensitivity_envelope"]["wider_envelope"],
            },
            "payload_sha256": frozen["payload_integrity"]["payload_sha256"],
            "source": provenance["forecast_result"],
        },
        {
            "id": "ns_quadrupole_Q",
            "domain": "NS/Hartle",
            "status": j0737["Q"]["status"],
            "evidence_weight": 0.0,
            "blocker": j0737["Q"]["reason"],
            "source": provenance["forecast_result"],
        },
        {
            "id": "ns_inverse_mass",
            "domain": "NS/Hartle",
            "status": j0737["inverse_mass"]["status"],
            "evidence_weight": 0.0,
            "blocker": j0737["inverse_mass"].get("reason", "No direct I_A measurement is present."),
            "source": provenance["forecast_result"],
        },
        {
            "id": "M_Omega",
            "domain": "NS/microphysics",
            "status": p1["m_omega_dependency"]["status"],
            "evidence_weight": 0.0,
            "values": {
                "declared_value_mev": p1["m_omega_dependency"]["declared_value_mev"],
                "declared_sigma_mev": p1["m_omega_dependency"]["declared_sigma_mev"],
            },
            "blocker": p1["m_omega_dependency"]["blocked_link"],
            "source": provenance["p1_ns_result"],
        },
        {
            "id": "barotropic_null",
            "domain": "cooling/g-modes",
            "status": null["status"],
            "evidence_weight": 0.0,
            "values": {
                "max_abs_N2_km2": null["max_abs_n2_km2"],
                "max_abs_discriminant": null["max_abs_discriminant"],
                "grid_points": null["grid_points"],
            },
            "physical_frequency_status": null["physical_frequency_status"],
            "source": provenance["cooling_result"],
        },
        {
            "id": "cooling_and_gmode",
            "domain": "cooling/g-modes",
            "status": "blocked",
            "evidence_weight": 0.0,
            "blockers": {
                "g_mode_frequency": physical_no_go["g_mode_frequency"],
                "cooling_curves": physical_no_go["cooling_curves"],
                "Cas_A_fit": physical_no_go["Cas_A_fit"],
                "hyperon_Urca_cooling": physical_no_go["hyperon_Urca_cooling"],
                "NVG_confirmation": physical_no_go["NVG_confirmation"],
            },
            "source": provenance["cooling_result"],
            "payload_sha256": cooling_result["payload_integrity"]["payload_sha256"],
        },
        {
            "id": "hyperon_fractions_and_phase",
            "domain": "hyperon/Urca",
            "status": beta_result["dependency_audit"]["physical_hyperon_status"],
            "evidence_weight": 0.0,
            "reference_model_status": beta_result["dependency_audit"]["reference_model_status"],
            "claim_statuses": {
                "physical_hyperon_fractions": beta_result["claims"]["physical_hyperon_fractions"],
                "phase_transition": beta_result["claims"]["phase_transition"],
                "NVG_hyperon_prediction": beta_result["claims"]["NVG_hyperon_prediction"],
            },
            "source": provenance["beta_result"],
        },
        {
            "id": "hyperon_urca",
            "domain": "hyperon/Urca",
            "status": beta_result["urca"]["emissivity_status"],
            "evidence_weight": 0.0,
            "cooling_status": beta_result["urca"]["cooling_claim"],
            "source": provenance["beta_result"],
        },
        {
            "id": "echo",
            "domain": "GW echo",
            "status": "synthetic_sensitivity",
            "source_artifact_status": echo_result["sensitivity"]["sensitivity_status"],
            "evidence_weight": 0.0,
            "selected_event_count": echo_reach["selected_event_count"],
            "raw_A90_union": echo_reach["numerical_union"]["raw_interval"],
            "display_A90_interval": echo_reach["numerical_union"]["displayed_interval"],
            "supported_decimal_places": echo_reach["numerical_union"]["supported_decimal_places"],
            "q_payload_integrity": echo_reach["q_payload_integrity"],
            "observational_status": echo_reach["observational_blocks"]["status"],
            "time_slide_status": echo_reach["observational_blocks"]["time_slide_independence"]["status"],
            "source": provenance["echo_result"],
        },
        {
            "id": "s8",
            "domain": "S8",
            "status": "conditional_sensitivity",
            "source_artifact_status": s8_result["status"],
            "evidence_weight": 0.0,
            "maintained_omega_m": s8_reach["maintained_slice"]["omega_m"],
            "maintained_one_sigma_overlap_count": s8_reach["maintained_slice"]["one_sigma_overlap_count"],
            "resolution_crossings": s8_reach["resolution_probe"]["nearest_crossing_omegas"],
            "critical_boundary_status": s8_reach["boundary_expansion"]["critical_boundary_status"],
            "normalization_status": s8_reach["normalization_probe"]["status"],
            "normalization_changes_one_sigma_overlap": s8_reach["normalization_probe"]["normalization_changes_one_sigma_overlap"],
            "normalization_conventions": s8_reach["normalization_probe"]["normalization_conventions"],
            "normalization_one_sigma_counts_by_omega": s8_reach["normalization_probe"]["one_sigma_counts_by_omega"],
            "resolution_probe_status": s8_reach["resolution_probe"]["status"],
            "resolution_is_grid_dependent": s8_reach["resolution_probe"]["boundary_resolution_dependent"],
            "global_statement_status": s8_reach["global_statement_status"],
            "desi_overlay_status": s8_reach["legacy_overlay_scope"]["provenance_status"],
            "payload_sha256": reach["payload_integrity"]["canonical_sha256"],
            "source": provenance["s8_result"],
        },
        {
            "id": "pbh",
            "domain": "PBH/PTA",
            "status": "algebraic_sensitivity",
            "source_artifact_status": pbh["status"],
            "evidence_weight": 0.0,
            "seed_band": {
                "minimum_required_rate_product": pbh_seed["minimum_required_rate_product"],
                "minimum_cycle": pbh_seed["minimum_row"]["cycle"],
                "minimum_amplitude_strain": pbh_seed["minimum_row"]["unit_rate_amplitude_strain"],
                "all_rate_requirements_outside_box": pbh_seed["all_rate_requirements_outside_box"],
            },
            "expanded_scan": {
                "first_reachable_rung": pbh_expanded["first_reachable_rung"],
                "boundary_cycle": pbh["frozen_expanded_envelope_boundary"]["cycle"],
                "boundary_fraction_dm": pbh["frozen_expanded_envelope_boundary"]["fraction_dm"],
                "boundary_max_amplitude_strain": pbh["frozen_expanded_envelope_boundary"]["max_amplitude_strain"],
            },
            "max_row_integrity": {
                "status": pbh_integrity["status"],
                "scan_maxima_copied_from_frozen_artifact": pbh_integrity["scan_maxima_copied_from_frozen_artifact"],
                "seed_max_row": pbh_integrity["seed_max_row_live_recomputed"],
                "expanded_max_row": pbh_integrity["expanded_max_row_live_recomputed"],
            },
            "target_leakage": pbh["target_leakage"],
            "external_exclusions_used": pbh_artifact["scan_contract"]["external_exclusions_used"],
            "pta_benchmark_is_internal": True,
            "source": provenance["pbh_result"],
        },
    ]
    return claims


def build_ledger() -> dict[str, Any]:
    """Authenticate final live artifacts and construct the machine ledger."""

    payloads = _assert_authorized_inputs()
    provenance = _source_provenance()
    inventory = _inventory()
    claims = _claims(payloads, provenance)
    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit": AUDIT_ID,
        "status": "FINAL_P4_S1_MACHINE_LEDGER",
        "scope": "P3-S8-authorized predictive integration; registry presence is not empirical evidence",
        "evidence_policy": {
            "independent_evidence_weight": 0.0,
            "conditional_synthetic_sensitivity_blocked_zero_weight": True,
            "registry_presence_is_not_evidence": True,
        },
        "inventory": inventory,
        "provenance": provenance,
        "claims": claims,
        "claim_count": len(claims),
    }
    ledger["ledger_integrity"] = {
        "algorithm": "sha256-canonical-json-v1",
        "canonicalization": "json_sort_keys_compact_utf8",
        "covered": "complete ledger except ledger_integrity",
        "payload_sha256": canonical_payload_digest(ledger),
        "status": "PASS_COMPLETE_LEDGER_DIGEST",
    }
    return ledger


def assert_ledger(payload: Mapping[str, Any]) -> None:
    """Fail closed if a ledger or any of its authenticated source inputs drift."""

    integrity = payload.get("ledger_integrity")
    if not isinstance(integrity, Mapping):
        raise AssertionError("ledger_integrity metadata is missing")
    expected_digest = canonical_payload_digest(payload)
    if integrity.get("payload_sha256") != expected_digest:
        raise AssertionError("predictive ledger digest mismatch")
    if integrity.get("status") != "PASS_COMPLETE_LEDGER_DIGEST":
        raise AssertionError("predictive ledger integrity status drift")
    expected = build_ledger()
    if _canonical_json(payload) != _canonical_json(expected):
        raise AssertionError("predictive ledger does not equal live fail-closed regeneration")


def write_ledger(payload: Mapping[str, Any] | None = None, path: Path = RESULT_PATH) -> Path:
    """Authenticate and write a deterministic ledger JSON artifact."""

    value = dict(payload) if payload is not None else build_ledger()
    if payload is not None:
        # Do not write an unverified caller-supplied mapping.
        assert_ledger(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="validate/build without writing the ledger")
    args = parser.parse_args(argv)
    try:
        ledger = build_ledger()
        if not args.no_write:
            write_ledger(ledger)
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"P4-S1 predictive ledger: FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "P4-S1 predictive ledger: "
        f"status={ledger['status']}, claims={ledger['claim_count']}, "
        f"digest={ledger['ledger_integrity']['payload_sha256']}"
    )
    if not args.no_write:
        print(f"Ledger artifact: {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
