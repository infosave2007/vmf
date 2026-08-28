#!/usr/bin/env python3
"""Hierarchical GW-echo sensitivity audit with a fail-closed data boundary.

This entry point implements the smallest likelihood that can be justified by the
files in this checkout.  The local GWTC table is used only as catalog-selection
metadata (and is hash checked against its canonical provenance manifest).  No
strain, event posterior, detector PSD, or independent time-slide products are
present, so an observational echo likelihood is *blocked*.  A controlled
two-detector Gaussian matched-filter model is nevertheless run to provide a
calibrated sensitivity projection and to exercise the hierarchical machinery.

The common population parameter is ``echo_amplitude_ratio`` (A >= 0): the
network matched-filter amplitude of an echo comb divided by the catalog merger
network SNR for each event.  For a selected event i, the synthetic signal
non-centrality is ``A * rho_i`` times a log-normal event-level nuisance factor.
The event likelihood is Bernoulli in a detection indicator y_i, with the
detection efficiency marginalized over that nuisance.  Efficiencies are
evaluated by deterministic Gauss--Hermite quadrature of the exact non-central
chi-square detection probability and projected onto the physically required
monotone curve.  The sensitivity result conditions on y_i=0 for every selected
catalog event; it is not an upper limit on real GW data.  Synthetic coverage
uses the full Bernoulli likelihood and therefore does not silently relabel mock
injections as observations.

Run from any working directory with::

    python3 verification/nvg_echo_hierarchical_upper_limit.py

The default run writes a uniquely named JSON result and figure under
``verification/``.  ``--quick`` is intended for focused tests only and is not
the terminal evidence run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = HERE / "data" / "p2_s3_hierarchical_echo"
CONFIG_PATH = DATA_DIR / "analysis_config.json"
MANIFEST_PATH = DATA_DIR / "provenance_manifest.json"
CATALOG_PATH = HERE / "data" / "gwtc_events.csv"
CANONICAL_MANIFEST_PATH = HERE / "data" / "provenance.json"
SOURCE_PATH = HERE / "nvg_echo_hierarchical_upper_limit.py"
TEST_PATH = HERE / "test_p2_echo_hierarchical_upper_limit.py"
RESULT_PATH = HERE / "nvg_echo_hierarchical_upper_limit_p2s3_results.json"
FIGURE_PATH = HERE / "fig_echo_hierarchical_upper_limit_p2s3.png"
REPORT_PATH = ROOT / "Lunacy/runs/predictive-research/phases/phase-2/P2-S3-REPORT.md"
EVIDENCE_PATH = ROOT / "Lunacy/runs/predictive-research/phases/phase-2/evidence/P2-S3-terminal-verification.log"

SCHEMA_VERSION = "P2-S3-hierarchical-echo-v1"
CONFIG_SCHEMA_VERSION = "P2-S3-hierarchical-echo-config-v1"
CALIBRATION_METHOD = "controlled_deterministic_gauss_hermite_noncentral_chi_square"
# Every numerical choice that can move the reported A90 is a first-class
# uncertainty axis.  Keep this list explicit so a future producer change
# cannot silently add a ladder without adding it to the union envelope.
NUMERICAL_UNCERTAINTY_AXES = (
    "quadrature",
    "seed",
    "posterior_grid",
    "amplitude_efficiency_grid",
    "monotonicity",
)
REQUIRED_COLUMNS = (
    "commonName",
    "version",
    "GPS",
    "network_matched_filter_snr",
    "p_astro",
    "final_mass_source",
)


class ProvenanceError(RuntimeError):
    """Raised when an input or configuration cannot be authenticated."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file without normalizing its bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_content_digest(manifest: dict[str, Any]) -> str:
    """Hash the manifest payload while excluding its self-digest field.

    The self-digest is deliberately excluded from the payload being hashed so
    that the manifest can authenticate itself without a circular hash.  The
    canonical JSON encoding makes the check independent of key order while the
    byte digest recorded in results still detects any live-file drift.
    """

    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_bytes_digest(path: Path = MANIFEST_PATH) -> str:
    """Hash manifest bytes after masking both self-digest values.

    Unlike the canonical content digest, this check intentionally remains
    sensitive to whitespace, key ordering, and any other byte-level drift in
    the live manifest.  The masked values avoid a circular self-reference.
    """

    raw = path.read_bytes()
    text = raw.decode("utf-8")
    masked, first = re.subn(r'("manifest_sha256"\s*:\s*)"[0-9a-fA-F]{64}"', r'\1"' + "0" * 64 + '"', text, count=1)
    if first != 1:
        raise ProvenanceError("P2-S3 provenance manifest lacks manifest_sha256 field")
    masked, second = re.subn(r'("manifest_bytes_sha256"\s*:\s*)"[0-9a-fA-F]{64}"', r'\1"' + "0" * 64 + '"', masked, count=1)
    if second != 1:
        raise ProvenanceError("P2-S6 provenance manifest lacks manifest_bytes_sha256 field")
    return hashlib.sha256(masked.encode("utf-8")).hexdigest()


def _read_p2s3_manifest() -> dict[str, Any]:
    """Read and validate the P2-S6 input/code provenance manifest."""

    if not MANIFEST_PATH.exists():
        raise ProvenanceError("P2-S3 provenance manifest is missing")
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read P2-S3 provenance manifest: {exc}") from exc
    if manifest.get("schema_version") != "P2-S6-hierarchical-echo-data-manifest-v2":
        raise ProvenanceError("unsupported P2-S3/P2-S6 provenance manifest schema")
    expected_self = manifest.get("manifest_sha256")
    if not expected_self or expected_self != _manifest_content_digest(manifest):
        raise ProvenanceError("P2-S3 provenance manifest self-digest does not match live content")
    expected_bytes = manifest.get("manifest_bytes_sha256")
    try:
        actual_bytes = _manifest_bytes_digest()
    except (OSError, UnicodeError, ProvenanceError) as exc:
        raise ProvenanceError(f"cannot verify P2-S6 manifest byte digest: {exc}") from exc
    if not expected_bytes or expected_bytes != actual_bytes:
        raise ProvenanceError("P2-S6 provenance manifest byte digest does not match live bytes")

    inputs = manifest.get("inputs", {})
    catalog_entry = inputs.get("gwtc_events.csv", {})
    if catalog_entry.get("trust_status") != "verified_public":
        raise ProvenanceError("P2-S3 manifest does not mark GWTC catalog verified_public")
    if not CATALOG_PATH.exists() or catalog_entry.get("sha256") != sha256_file(CATALOG_PATH):
        raise ProvenanceError("P2-S3 manifest GWTC hash does not match canonical catalog")
    config_entry = inputs.get("analysis_config.json", {})
    if not CONFIG_PATH.exists() or config_entry.get("sha256") != sha256_file(CONFIG_PATH):
        raise ProvenanceError("analysis_config.json hash does not match P2-S3 provenance manifest")
    code = manifest.get("code_provenance", {})
    producer_entry = code.get("producer", {})
    test_entry = code.get("focused_test", {})
    expected_producer = producer_entry.get("sha256")
    expected_test = test_entry.get("sha256")
    if producer_entry.get("path") != str(SOURCE_PATH.relative_to(ROOT)) or not expected_producer:
        raise ProvenanceError("P2-S6 manifest lacks the producer source pin")
    if test_entry.get("path") != str(TEST_PATH.relative_to(ROOT)) or not expected_test:
        raise ProvenanceError("P2-S6 manifest lacks the focused-test source pin")
    if expected_producer != sha256_file(SOURCE_PATH):
        raise ProvenanceError("P2-S6 producer source hash does not match live source")
    if expected_test != sha256_file(TEST_PATH):
        raise ProvenanceError("P2-S6 focused-test source hash does not match live test")
    return manifest


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    """Convert NumPy containers and non-finite floats for deterministic JSON."""

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


def load_config() -> dict[str, Any]:
    """Load the pinned analysis configuration and verify its manifest hash."""

    if not CONFIG_PATH.exists() or not MANIFEST_PATH.exists():
        raise ProvenanceError("P2-S3 configuration or provenance manifest is missing")
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read P2-S3 configuration provenance: {exc}") from exc
    # Validate the complete live manifest, including producer/test source pins,
    # before trusting any configuration value.
    manifest = _read_p2s3_manifest()
    expected = manifest.get("inputs", {}).get("analysis_config.json", {}).get("sha256")
    actual = sha256_file(CONFIG_PATH)
    if not expected or expected == "TO_BE_FILLED_BY_TERMINAL_VERIFICATION":
        raise ProvenanceError("analysis_config.json has no pinned SHA-256 in provenance manifest")
    if actual != expected:
        raise ProvenanceError("analysis_config.json hash does not match P2-S3 provenance manifest")
    catalog_entry = manifest.get("inputs", {}).get("gwtc_events.csv", {})
    if not catalog_entry.get("source_url") or not catalog_entry.get("release_version"):
        raise ProvenanceError("P2-S3 manifest lacks GWTC primary-source/version provenance")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ProvenanceError("unsupported P2-S3 configuration schema")
    calibration = config.get("calibration", {})
    if calibration.get("calibration_method") != CALIBRATION_METHOD:
        raise ProvenanceError("P2-S6 deterministic calibration method is not pinned")
    nodes = calibration.get("quadrature_nodes")
    if not isinstance(nodes, int) or nodes < 8:
        raise ProvenanceError("P2-S6 quadrature_nodes must be an integer >= 8")
    reported_digits = calibration.get("reported_upper90_decimal_places")
    if not isinstance(reported_digits, int) or reported_digits < 0:
        raise ProvenanceError("reported_upper90_decimal_places must be a non-negative integer")
    for key in ("posterior_grid_resolution_points", "amplitude_efficiency_grid_resolution_points"):
        ladder = calibration.get(key)
        if not isinstance(ladder, list) or not ladder or any(not isinstance(value, int) or value < 2 for value in ladder):
            raise ProvenanceError(f"{key} must be a non-empty integer ladder")
        if any(left >= right for left, right in zip(ladder, ladder[1:])):
            raise ProvenanceError(f"{key} must be strictly increasing")
    axes = calibration.get("uncertainty_axes")
    if axes != list(NUMERICAL_UNCERTAINTY_AXES):
        raise ProvenanceError("calibration uncertainty_axes do not cover every numerical axis")
    precision_rule = calibration.get("precision_rule", {})
    if precision_rule.get("name") != "shared_endpoint_round_half_even":
        raise ProvenanceError("unsupported A90 precision rule")
    if not isinstance(precision_rule.get("max_decimal_places"), int) or precision_rule["max_decimal_places"] < reported_digits:
        raise ProvenanceError("precision rule max_decimal_places is invalid")
    return config


def _canonical_catalog_provenance() -> dict[str, Any]:
    """Validate the inherited canonical GWTC hash and return its metadata."""

    if not CATALOG_PATH.exists() or not CANONICAL_MANIFEST_PATH.exists():
        raise ProvenanceError("canonical GWTC catalog or provenance manifest is missing")
    try:
        manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"cannot read canonical GWTC provenance: {exc}") from exc
    entry = manifest.get("entries", {}).get("gwtc_events.csv", {})
    if entry.get("trust_status") != "verified_public":
        raise ProvenanceError("GWTC catalog is not marked verified_public")
    expected = entry.get("sha256")
    actual = sha256_file(CATALOG_PATH)
    if not expected or actual != expected:
        raise ProvenanceError("GWTC catalog hash does not match canonical provenance manifest")
    return {
        "path": str(CATALOG_PATH.relative_to(ROOT)),
        "sha256": actual,
        "trust_status": entry.get("trust_status"),
        "source": entry.get("source"),
        "source_url": entry.get("source_url"),
        "version": entry.get("version"),
        "reference": "row-level reference/jsonurl fields are retained in the local export",
    }


def source_provenance() -> dict[str, Any]:
    """Return the complete live P2-S6 source/input digest bundle.

    Explicit aliases mirror the provenance vocabulary used by the other Phase 2
    producers (``source_sha256``, ``producer_source_sha256`` and
    ``test_source_sha256``) so independent gates can verify the artifact without
    relying on one particular nested field name.
    """

    # _read_p2s3_manifest performs the fail-closed source/config/catalog checks.
    manifest = _read_p2s3_manifest()
    source_hash = sha256_file(SOURCE_PATH)
    test_hash = sha256_file(TEST_PATH)
    manifest_hash = sha256_file(MANIFEST_PATH)
    config_hash = sha256_file(CONFIG_PATH)
    catalog_hash = sha256_file(CATALOG_PATH)
    return {
        "producer": str(SOURCE_PATH.relative_to(ROOT)),
        "source_sha256": source_hash,
        "producer_source_sha256": source_hash,
        "test_path": str(TEST_PATH.relative_to(ROOT)),
        "test_sha256": test_hash,
        "test_source_sha256": test_hash,
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_sha256": manifest_hash,
        "input_manifest_sha256": manifest_hash,
        "manifest_content_sha256": manifest["manifest_sha256"],
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": config_hash,
        "input_config_sha256": config_hash,
        "catalog_path": str(CATALOG_PATH.relative_to(ROOT)),
        "catalog_sha256": catalog_hash,
        "input_catalog_sha256": catalog_hash,
        "catalog_trust_status": "verified_public",
    }


def _as_float(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _version_number(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("version", "0")))
    except (TypeError, ValueError):
        return -1


def load_catalog(config: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load one latest row per event and return selected rows plus a skip ledger.

    Selection is explicitly conditional on the public catalog.  A missing or
    malformed field is a skip, never a default mass/SNR.  Duplicate versions are
    audited and only the highest integer version per ``commonName`` is retained.
    """

    config = config or load_config()
    provenance = _canonical_catalog_provenance()
    criteria = config["catalog_selection"]
    with CATALOG_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing_columns = [key for key in REQUIRED_COLUMNS if key not in (reader.fieldnames or [])]
        if missing_columns:
            raise ProvenanceError(f"GWTC catalog missing required columns: {missing_columns}")
        rows = list(reader)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        name = (row.get("commonName") or "").strip()
        if name:
            grouped.setdefault(name, []).append(row)
    latest: dict[str, dict[str, str]] = {}
    duplicate_events: dict[str, int] = {}
    for name, versions in grouped.items():
        if len(versions) > 1:
            duplicate_events[name] = len(versions)
        latest[name] = max(versions, key=_version_number)

    selected: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}

    def skip(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for name in sorted(latest):
        row = latest[name]
        snr = _as_float(row, "network_matched_filter_snr")
        if snr is None or snr < float(criteria["min_network_matched_filter_snr"]):
            skip("missing_or_below_network_snr")
            continue
        p_astro = _as_float(row, "p_astro")
        if p_astro is None or p_astro < float(criteria["min_p_astro"]):
            skip("missing_or_below_p_astro")
            continue
        mass = _as_float(row, "final_mass_source")
        if mass is None or mass < float(criteria["min_final_mass_source_msun"]):
            skip("missing_or_below_final_mass")
            continue
        gps = _as_float(row, "GPS")
        if gps is None or gps <= 0.0:
            skip("missing_or_invalid_gps")
            continue
        selected.append(
            {
                "event": name,
                "catalog_version": _version_number(row),
                "gps": gps,
                "network_snr": snr,
                "p_astro": p_astro,
                "final_mass_source_msun": mass,
                "catalog_conditioned": True,
            }
        )
    selected.sort(key=lambda item: (-float(item["network_snr"]), item["event"]))
    audit = {
        "provenance": provenance,
        "raw_rows": len(rows),
        "unique_events": len(latest),
        "duplicate_event_groups": len(duplicate_events),
        "duplicate_rows_removed": int(sum(duplicate_events.values()) - len(duplicate_events)),
        "duplicate_event_examples": dict(sorted(duplicate_events.items())[:8]),
        "selection": {
            "criteria": criteria,
            "selected": len(selected),
            "skipped": int(sum(reasons.values())),
            "skip_reasons": reasons,
            "selection_is_conditional": True,
            "population_rate_inference": False,
        },
    }
    return selected, audit


def check_real_data_products() -> dict[str, Any]:
    """Report the exact real-data dependency boundary without fetching data."""

    expected = {
        "strain": DATA_DIR / "strain",
        "posterior_samples": DATA_DIR / "posterior",
        "noise_psd_or_time_slides": DATA_DIR / "noise",
    }
    found: dict[str, list[str]] = {}
    missing: list[str] = []
    for key, directory in expected.items():
        files = sorted(str(path.relative_to(ROOT)) for path in directory.rglob("*") if path.is_file()) if directory.exists() else []
        found[key] = files
        if not files:
            missing.append(key)
    return {
        "status": "BLOCKED_MISSING_STRAIN_POSTERIOR_NOISE_PRODUCTS" if missing else "AVAILABLE_PENDING_VALIDATION",
        "required_products": {key: str(path.relative_to(ROOT)) for key, path in expected.items()},
        "found": found,
        "missing": missing,
        "existing_summary_inputs_rejected": [
            "verification/data/nvg_residual_stack_results.csv (generated_unverified summary, not raw strain)",
            "verification/nvg_echo_timeslide_background.py (network/PyCBC fetch entry point; no local time-slide bank)",
            "verification/nvg_echo_upper_limit.py (single-event injection script; no retained strain/PSD product)",
        ],
        "observational_likelihood": "BLOCKED_NO_EVENT_LEVEL_STRAIN_POSTERIOR_NOISE",
        "time_slide_independence": {
            "status": "BLOCKED_NO_EMPIRICAL_TIME_SLIDE_BANK",
            "limitation": "nvg_echo_timeslide_background.py can generate L1-vs-H1 slides only after detector strain is fetched; no local strain or retained slide bank is available",
            "synthetic_background_is_independent": True,
        },
    }


def _draw_background(rng: np.random.Generator, count: int, n_trials: int) -> np.ndarray:
    """Compatibility helper for an explicitly synthetic null draw.

    Production calibration does not use this finite sample: the null threshold
    is evaluated from the exact order-statistic CDF below.  Keeping the helper
    available makes the synthetic/mock boundary explicit for focused tests and
    downstream audits without allowing it to define the reported efficiency.
    """

    if count <= 0 or n_trials <= 0:
        raise ValueError("background count and delay-trial count must be positive")
    noise = rng.normal(size=(int(count), int(n_trials), 4))
    return np.sum(noise * noise, axis=2).max(axis=1)


def _isotonic_non_decreasing(values: Sequence[float]) -> np.ndarray:
    """Return the equally weighted pool-adjacent-violators projection."""

    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or np.any(~np.isfinite(y)):
        raise ValueError("isotonic calibration requires finite one-dimensional values")
    if len(y) < 2:
        return y.copy()
    # Blocks carry their weighted mean and the original span.  Equal weights
    # are appropriate because every deterministic amplitude evaluation has the
    # same quadrature error control; no noisy point is given special leverage.
    means: list[float] = []
    weights: list[int] = []
    spans: list[tuple[int, int]] = []
    for index, value in enumerate(y):
        means.append(float(value))
        weights.append(1)
        spans.append((index, index + 1))
        while len(means) >= 2 and means[-2] > means[-1]:
            total_weight = weights[-2] + weights[-1]
            means[-2] = (weights[-2] * means[-2] + weights[-1] * means[-1]) / total_weight
            weights[-2] = total_weight
            spans[-2] = (spans[-2][0], spans[-1][1])
            means.pop()
            weights.pop()
            spans.pop()
    result = np.empty_like(y)
    for mean, (start, stop) in zip(means, spans):
        result[start:stop] = mean
    return result


def _deterministic_efficiency(
    event_snr: float,
    amplitudes: Sequence[float],
    threshold: float,
    n_trials: int,
    nuisance_sigma: float,
    quadrature_nodes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate recovery efficiency from the exact non-central chi-square law.

    The first (zero-delay) trial has a non-central chi-square statistic with
    four degrees of freedom and non-centrality ``(A rho eta)^2``.  The remaining
    delay trials are central chi-square variables.  A Gauss--Hermite rule then
    integrates the declared log-normal nuisance exactly up to the reported
    quadrature order.  This has no finite-injection RNG noise; isotonic PAVA is
    only a numerical safeguard for the physically required non-decreasing
    efficiency curve.
    """

    from scipy.special import roots_hermitenorm
    from scipy.stats import chi2, ncx2

    if not _finite(event_snr) or float(event_snr) <= 0.0 or not _finite(threshold) or float(threshold) <= 0.0:
        raise ValueError("invalid synthetic detector/event configuration")
    if int(n_trials) <= 0 or int(quadrature_nodes) < 4:
        raise ValueError("delay trials and quadrature_nodes must be positive")
    if not _finite(nuisance_sigma) or float(nuisance_sigma) < 0.0:
        raise ValueError("nuisance sigma must be finite and non-negative")
    amplitudes_array = np.asarray(amplitudes, dtype=float)
    if amplitudes_array.ndim != 1 or np.any(~np.isfinite(amplitudes_array)) or np.any(amplitudes_array < 0.0):
        raise ValueError("injection amplitudes must be finite and non-negative")

    nodes, weights = roots_hermitenorm(int(quadrature_nodes))
    weights = np.asarray(weights, dtype=float) / math.sqrt(2.0 * math.pi)
    eta = np.exp(-0.5 * float(nuisance_sigma) ** 2 + float(nuisance_sigma) * np.asarray(nodes, dtype=float))
    central_cdf = float(chi2.cdf(float(threshold), df=4))
    central_other_trials = central_cdf ** (int(n_trials) - 1)
    noncentrality = np.square(amplitudes_array[:, None] * float(event_snr) * eta[None, :])
    signal_cdf = ncx2.cdf(float(threshold), df=4, nc=noncentrality)
    raw = 1.0 - central_other_trials * (signal_cdf @ weights)
    raw = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
    calibrated = np.clip(_isotonic_non_decreasing(raw), 0.0, 1.0)
    raw_differences = np.diff(raw)
    calibrated_differences = np.diff(calibrated)
    diagnostics = {
        "raw_downward_pairs": int(np.sum(raw_differences < -1.0e-12)),
        "calibrated_downward_pairs": int(np.sum(calibrated_differences < -1.0e-12)),
        "raw_min_step": float(np.min(raw_differences)) if len(raw_differences) else 0.0,
        "calibrated_min_step": float(np.min(calibrated_differences)) if len(calibrated_differences) else 0.0,
        "max_isotonic_adjustment": float(np.max(np.abs(calibrated - raw))) if len(raw) else 0.0,
        "quadrature_nodes": int(quadrature_nodes),
        "quadrature_family": "probabilists_hermite_normal_expectation",
    }
    return calibrated, diagnostics


def _draw_injection_statistics(
    rng: np.random.Generator | None,
    event_snr: float,
    amplitudes: Sequence[float],
    threshold: float,
    n_injections: int,
    n_trials: int,
    nuisance_sigma: float,
    quadrature_nodes: int = 64,
) -> np.ndarray:
    """Compatibility wrapper for the deterministic efficiency evaluator.

    ``rng`` and ``n_injections`` are accepted for callers of the P2-S3 API but
    intentionally ignored: finite seeded injection points were the source of
    the previous non-monotone calibration and are no longer production inputs.
    """

    del rng, n_injections
    return _deterministic_efficiency(
        event_snr, amplitudes, threshold, n_trials, nuisance_sigma, quadrature_nodes
    )[0]


def _analytic_background_threshold(n_trials: int, quantile: float) -> tuple[float, float, float, float]:
    """Return exact max-statistic threshold, FAP, median and high quantile."""

    from scipy.stats import chi2

    if int(n_trials) <= 0 or not (0.0 < float(quantile) < 1.0):
        raise ValueError("invalid synthetic null configuration")
    central_target = float(quantile) ** (1.0 / int(n_trials))
    threshold = float(chi2.ppf(central_target, df=4))
    central_cdf = float(chi2.cdf(threshold, df=4))
    fap = float(1.0 - central_cdf ** int(n_trials))
    median = float(chi2.ppf(0.5 ** (1.0 / int(n_trials)), df=4))
    high_quantile = float(chi2.ppf((1.0 - 1.0e-12) ** (1.0 / int(n_trials)), df=4))
    return threshold, fap, median, high_quantile


def synthetic_calibration(
    events: Sequence[dict[str, Any]],
    config: dict[str, Any],
    quick: bool = False,
    *,
    seed_override: int | None = None,
    quadrature_nodes_override: int | None = None,
    amplitude_grid_override: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Calibrate synthetic efficiencies with a controlled deterministic rule."""

    detector = config["synthetic_detector_model"]
    calibration = config["calibration"]
    seed = int(calibration["rng_seed"] if seed_override is None else seed_override)
    configured_nodes = int(calibration["quadrature_nodes"])
    quadrature_nodes = int(configured_nodes if quadrature_nodes_override is None else quadrature_nodes_override)
    if quick:
        quadrature_nodes = min(quadrature_nodes, 24)
    n_trials = int(detector["n_delay_trials"])
    if amplitude_grid_override is None:
        amplitudes = np.asarray(detector["injection_amplitude_grid"], dtype=float)
        amplitude_grid_kind = "configured_exact_evaluator_grid"
    else:
        amplitudes = np.asarray(amplitude_grid_override, dtype=float)
        amplitude_grid_kind = "exact_evaluator_resolution_probe"
    if amplitudes.ndim != 1 or len(amplitudes) < 2 or np.any(~np.isfinite(amplitudes)):
        raise ValueError("amplitude efficiency grid must contain at least two finite values")
    if np.any(amplitudes < 0.0) or np.any(np.diff(amplitudes) <= 0.0):
        raise ValueError("amplitude efficiency grid must be strictly increasing and non-negative")
    threshold, null_fap, background_median, background_high = _analytic_background_threshold(
        n_trials, float(detector["background_quantile"])
    )
    efficiency_rows: list[dict[str, Any]] = []
    q_matrix = np.empty((len(events), len(amplitudes)), dtype=float)
    diagnostics_rows: list[dict[str, Any]] = []
    for i, event in enumerate(events):
        q, diagnostics = _deterministic_efficiency(
            float(event["network_snr"]),
            amplitudes,
            threshold,
            n_trials,
            float(detector["event_log_nuisance_sigma"]),
            quadrature_nodes,
        )
        q_matrix[i] = q
        diagnostics_rows.append({"event": event["event"], **diagnostics})
        efficiency_rows.append(
            {
                "event": event["event"],
                "network_snr": event["network_snr"],
                "efficiency": q.tolist(),
            }
        )
    raw_downward = int(sum(row["raw_downward_pairs"] for row in diagnostics_rows))
    calibrated_downward = int(sum(row["calibrated_downward_pairs"] for row in diagnostics_rows))
    max_adjustment = float(max((row["max_isotonic_adjustment"] for row in diagnostics_rows), default=0.0))
    return {
        "seed": seed,
        "seed_used_for_calibration": False,
        "method": CALIBRATION_METHOD,
        "deterministic": True,
        "resolution": {
            "quadrature_nodes": quadrature_nodes,
            "configured_nodes": configured_nodes,
            "amplitude_grid_points": int(len(amplitudes)),
            "amplitude_grid_kind": amplitude_grid_kind,
            "quadrature_error_control": "resolution ladder is evaluated in run_sensitivity",
        },
        "background": {
            "kind": "synthetic_gaussian_null_exact_order_statistic",
            "not_observational": True,
            "draws": 0,
            "configured_draws_not_used": int(detector.get("n_background", 0)),
            "delay_trials": n_trials,
            "quantile": detector["background_quantile"],
            "threshold": threshold,
            "median": background_median,
            "max": background_high,
            "null_calibration": {
                "independent_draws": 0,
                "false_alarm_fraction": null_fap,
                "target": float(1.0 - detector["background_quantile"]),
                "evaluation": "exact central-chi-square max CDF",
                "status": "PASS_SYNTHETIC_NULL" if abs(null_fap - (1.0 - float(detector["background_quantile"]))) <= 1.0e-12 else "REVIEW_SYNTHETIC_NULL",
            },
        },
        "injections": {
            "amplitudes": amplitudes.tolist(),
            "per_event_draws": 0,
            "event_log_nuisance_sigma": detector["event_log_nuisance_sigma"],
            "efficiency_model": "noncentral_chi_square_max_with_log_normal_nuisance",
            "quadrature_nodes": quadrature_nodes,
            "efficiency_rows": efficiency_rows,
            "event_diagnostics": diagnostics_rows,
            "raw_downward_pairs": raw_downward,
            "calibrated_downward_pairs": calibrated_downward,
            "max_isotonic_adjustment": max_adjustment,
            "status": "PASS_SYNTHETIC_INJECTION_RECOVERY" if calibrated_downward == 0 else "REVIEW_SYNTHETIC_INJECTION_RECOVERY",
        },
        "q_matrix": q_matrix,
        "amplitudes": amplitudes,
    }


def posterior_from_efficiencies(
    q_matrix: np.ndarray,
    efficiency_amplitudes: Sequence[float],
    parameter_grid: Sequence[float],
    prior: str = "uniform",
    prior_max: float = 2.0,
    observations: Sequence[int] | None = None,
    monotone_calibration: bool = True,
) -> dict[str, Any]:
    """Evaluate the Bernoulli hierarchical posterior and its 90% upper quantile.

    Efficiency rows are monotone-calibrated with the same PAVA projection used
    by the deterministic injection model.  This prevents a caller from
    reintroducing the seed-noisy downward steps that invalidated the P2-S3
    artifact while retaining an opt-out for diagnostic reconstruction.
    """

    q_matrix = np.asarray(q_matrix, dtype=float)
    efficiency_amplitudes = np.asarray(efficiency_amplitudes, dtype=float)
    grid = np.asarray(parameter_grid, dtype=float)
    if q_matrix.ndim != 2 or q_matrix.shape[1] != len(efficiency_amplitudes):
        raise ValueError("q_matrix and efficiency amplitudes have incompatible shapes")
    if grid.ndim != 1 or len(grid) < 2 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("posterior parameter grid must be finite and strictly increasing")
    if prior_max <= 0.0 or grid[0] < 0.0 or grid[-1] > prior_max + 1.0e-12:
        raise ValueError("posterior grid lies outside the declared prior domain")
    if observations is None:
        y = np.zeros(q_matrix.shape[0], dtype=int)
    else:
        y = np.asarray(observations, dtype=int)
        if y.shape != (q_matrix.shape[0],) or np.any((y != 0) & (y != 1)):
            raise ValueError("observations must be one binary value per event")
    if np.any(np.diff(efficiency_amplitudes) <= 0.0):
        raise ValueError("efficiency amplitudes must be strictly increasing")
    if np.any(~np.isfinite(q_matrix)) or np.any(q_matrix < -1.0e-12) or np.any(q_matrix > 1.0 + 1.0e-12):
        raise ValueError("efficiency values must be finite and lie in [0, 1]")
    calibrated_rows = [_isotonic_non_decreasing(row) if monotone_calibration else np.asarray(row, dtype=float) for row in q_matrix]
    q = np.vstack([np.interp(grid, efficiency_amplitudes, row, left=row[0], right=row[-1]) for row in calibrated_rows])
    q = np.clip(q, 1.0e-9, 1.0 - 1.0e-9)
    log_likelihood = np.sum(np.where(y[:, None] == 1, np.log(q), np.log1p(-q)), axis=0)
    log_likelihood -= float(np.max(log_likelihood))
    likelihood = np.exp(log_likelihood)
    prior_density = np.zeros_like(grid)
    if prior == "uniform":
        prior_density[:] = 1.0 / float(prior_max)
    elif prior == "uniform_half":
        prior_density[:] = 1.0 / float(prior_max / 2.0)
        prior_density[grid > prior_max / 2.0] = 0.0
    elif prior == "half_normal":
        scale = max(1.0e-9, float(prior_max) / 4.0)
        prior_density = np.exp(-0.5 * (grid / scale) ** 2)
        prior_density[0] = 1.0
    else:
        raise ValueError(f"unknown prior family: {prior}")
    posterior = likelihood * prior_density
    normalizer = float(np.trapz(posterior, grid))
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("posterior normalization failed")
    posterior /= normalizer
    cdf = np.zeros_like(grid)
    cdf[1:] = np.cumsum(0.5 * (posterior[1:] + posterior[:-1]) * np.diff(grid))
    cdf /= float(cdf[-1])
    upper90 = float(np.interp(0.90, cdf, grid))
    return {
        "grid": grid,
        "q_matrix": q,
        "posterior": posterior,
        "cdf": cdf,
        "upper90": upper90,
        "normalizer": normalizer,
        "prior": prior,
        "prior_max": float(prior_max),
        "observations": y,
        "monotone_calibration": bool(monotone_calibration),
    }


def _round_report(value: float | None, decimals: int) -> float | None:
    """Round a reported scalar to the precision supported by calibration."""

    return None if value is None else float(np.round(float(value), int(decimals)))


def _shared_rounding_precision(interval: Sequence[float], max_decimal_places: int = 12) -> int | None:
    """Return the finest shared round-half-even decimal precision.

    The printed A90 is allowed only when both ends of the complete numerical
    envelope round to the same value at that precision.  This is deliberately
    an endpoint rule: it never turns a broad interval into a falsely precise
    point estimate, and a missing/invalid interval returns ``None`` so callers
    fail closed.
    """

    if len(interval) != 2:
        return None
    try:
        lower, upper = (float(interval[0]), float(interval[1]))
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        return None
    for decimals in range(int(max_decimal_places), -1, -1):
        if _round_report(lower, decimals) == _round_report(upper, decimals):
            return decimals
    return None


def _finite_interval(values: Sequence[float]) -> list[float]:
    """Return a finite min/max interval, rejecting an empty numerical axis."""

    raw = [float(value) for value in values]
    if not raw or any(not math.isfinite(value) for value in raw):
        raise ValueError("numerical uncertainty axis has no finite raw values")
    return [min(raw), max(raw)]


def _outward_rounded_interval(interval: Sequence[float], decimals: int) -> list[float]:
    """Round an interval outward for display without narrowing its evidence."""

    lower, upper = (float(interval[0]), float(interval[1]))
    scale = 10.0 ** int(decimals)
    return [math.floor(lower * scale) / scale, math.ceil(upper * scale) / scale]


def _validate_uncertainty_bundle(uncertainty: dict[str, Any]) -> None:
    """Fail closed unless every numerical axis is represented in the union.

    This validation is intentionally independent of the producer's current
    ladder implementation.  Removing an axis, replacing raw values with only
    rounded values, or shrinking the union below any axis interval therefore
    becomes a hard failure during both generation and provenance audits.
    """

    axes = uncertainty.get("axes")
    if not isinstance(axes, dict) or set(axes) != set(NUMERICAL_UNCERTAINTY_AXES):
        raise ValueError("calibration uncertainty must expose every numerical axis")
    union = uncertainty.get("union_interval")
    try:
        union_interval = [float(union[0]), float(union[1])]
    except (TypeError, ValueError, IndexError):
        raise ValueError("calibration uncertainty union interval is missing") from None
    if not (math.isfinite(union_interval[0]) and math.isfinite(union_interval[1]) and union_interval[0] <= union_interval[1]):
        raise ValueError("calibration uncertainty union interval is invalid")
    axis_intervals: dict[str, list[float]] = {}
    for axis in NUMERICAL_UNCERTAINTY_AXES:
        entry = axes.get(axis)
        if not isinstance(entry, dict):
            raise ValueError(f"calibration uncertainty axis is missing: {axis}")
        rows = entry.get("raw_ladder")
        interval = entry.get("interval")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"calibration uncertainty raw ladder is missing: {axis}")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"calibration uncertainty interval is missing: {axis}")
        raw_values: list[float] = []
        for row in rows:
            if not isinstance(row, dict) or "upper90_raw" not in row:
                raise ValueError(f"calibration uncertainty raw A90 is missing: {axis}")
            try:
                value = float(row["upper90_raw"])
            except (TypeError, ValueError):
                raise ValueError(f"calibration uncertainty raw A90 is invalid: {axis}") from None
            if not math.isfinite(value):
                raise ValueError(f"calibration uncertainty raw A90 is non-finite: {axis}")
            raw_values.append(value)
        observed = _finite_interval(raw_values)
        declared = [float(interval[0]), float(interval[1])]
        if not (math.isfinite(declared[0]) and math.isfinite(declared[1]) and declared[0] <= declared[1]):
            raise ValueError(f"calibration uncertainty interval is invalid: {axis}")
        if declared[0] > observed[0] + 1.0e-15 or declared[1] < observed[1] - 1.0e-15:
            raise ValueError(f"calibration uncertainty interval omits raw values: {axis}")
        if declared[0] < union_interval[0] - 1.0e-15 or declared[1] > union_interval[1] + 1.0e-15:
            raise ValueError(f"calibration uncertainty union omits axis: {axis}")
        axis_intervals[axis] = declared
    declared_axes = uncertainty.get("axis_intervals")
    if declared_axes != axis_intervals:
        raise ValueError("calibration uncertainty axis intervals do not match declared intervals")
    overall = uncertainty.get("overall_interval")
    if not isinstance(overall, (list, tuple)) or len(overall) != 2:
        raise ValueError("calibration uncertainty overall interval is missing")
    try:
        overall_interval = [float(overall[0]), float(overall[1])]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("calibration uncertainty overall interval is invalid") from None
    if not all(math.isfinite(value) for value in overall_interval):
        raise ValueError("calibration uncertainty overall interval is invalid")
    if overall_interval[0] > union_interval[0] + 1.0e-15 or overall_interval[1] < union_interval[1] - 1.0e-15:
        raise ValueError("calibration uncertainty overall interval omits union envelope")
    if uncertainty.get("overall_interval_raw") != uncertainty.get("union_interval"):
        raise ValueError("calibration uncertainty raw overall/union interval drift")
    supported = uncertainty.get("supported_upper90_decimal_places")
    if not isinstance(supported, int) or supported < 0:
        raise ValueError("calibration uncertainty supported precision is missing")
    calculated = _shared_rounding_precision(union_interval, int(uncertainty.get("precision_rule", {}).get("max_decimal_places", 12)))
    if calculated is None or supported > calculated:
        raise ValueError("calibration uncertainty precision exceeds complete union envelope")


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float | None]:
    """Return a two-sided Wilson 95% interval for synthetic coverage."""

    if trials <= 0:
        return [None, None]
    n = float(trials)
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def run_sensitivity(events: Sequence[dict[str, Any]], config: dict[str, Any], quick: bool = False) -> dict[str, Any]:
    """Run synthetic calibration and all required sensitivity controls."""

    calibration = synthetic_calibration(events, config, quick=quick)
    detector = config["synthetic_detector_model"]
    population = config["population_parameter"]
    calibration_config = config["calibration"]
    report_decimals = int(calibration_config["reported_upper90_decimal_places"])
    n_grid = min(int(population["posterior_grid_points"]), 401 if quick else int(population["posterior_grid_points"]))
    parameter_grid = np.linspace(float(population["prior_min"]), float(population["prior_max"]), n_grid)
    posterior = posterior_from_efficiencies(
        calibration["q_matrix"], calibration["amplitudes"], parameter_grid,
        prior=str(population["prior_family"]), prior_max=float(population["prior_max"]),
    )
    raw_upper90 = float(posterior["upper90"])

    # Prior sensitivity is evaluated on the same calibrated q matrix.
    prior_rows = {}
    for prior, pmax in (("uniform", 2.0), ("uniform_half", 2.0), ("half_normal", 2.0)):
        prior_rows[prior] = _round_report(posterior_from_efficiencies(
            calibration["q_matrix"], calibration["amplitudes"], parameter_grid,
            prior=prior, prior_max=pmax,
        )["upper90"], report_decimals)

    # Event-inclusion sensitivity: strict O1--O3 and a louder catalog cut.
    gps_cut = float(config["catalog_selection"]["gps_cut_o1_o3"])
    variants = {
        "all_selected": np.ones(len(events), dtype=bool),
        "o1_o3_only": np.array([float(e["gps"]) < gps_cut for e in events], dtype=bool),
        "snr_ge_10": np.array([float(e["network_snr"]) >= 10.0 for e in events], dtype=bool),
    }
    inclusion = {}
    for name, mask in variants.items():
        if not np.any(mask):
            inclusion[name] = {"events": 0, "upper90": None, "status": "SKIPPED_EMPTY_SELECTION"}
            continue
        row = posterior_from_efficiencies(
            calibration["q_matrix"][mask], calibration["amplitudes"], parameter_grid,
            prior="uniform", prior_max=float(population["prior_max"]),
        )
        inclusion[name] = {
            "events": int(np.sum(mask)),
            "upper90": _round_report(row["upper90"], report_decimals),
            "status": "SENSITIVITY_ONLY",
        }

    # Leave-one-event-out stability of the all-null projection.
    loo_values = []
    for i, event in enumerate(events):
        mask = np.ones(len(events), dtype=bool)
        mask[i] = False
        row = posterior_from_efficiencies(
            calibration["q_matrix"][mask], calibration["amplitudes"], parameter_grid,
            prior="uniform", prior_max=float(population["prior_max"]),
        )
        loo_values.append({"event": event["event"], "upper90": _round_report(row["upper90"], report_decimals)})
    loo_upper = np.asarray([row["upper90"] for row in loo_values], dtype=float)

    # Posterior-grid resolution is an independent numerical axis.  Keep every
    # raw value (not only rounded display values) in the artifact.
    configured_posterior_points = [int(x) for x in calibration_config["posterior_grid_resolution_points"]]
    posterior_grid_points = sorted(set(configured_posterior_points + [int(n_grid)]))
    if quick:
        posterior_grid_points = [x for x in posterior_grid_points if x <= 3201]
        if n_grid not in posterior_grid_points:
            posterior_grid_points.insert(0, n_grid)
        posterior_grid_points = sorted(set(posterior_grid_points))
    convergence_rows = []
    for points in posterior_grid_points:
        grid = np.linspace(float(population["prior_min"]), float(population["prior_max"]), points)
        row = posterior_from_efficiencies(
            calibration["q_matrix"], calibration["amplitudes"], grid,
            prior="uniform", prior_max=float(population["prior_max"]),
        )
        convergence_rows.append({
            "grid_points": points,
            "upper90_raw": float(row["upper90"]),
            "upper90": _round_report(row["upper90"], report_decimals),
            "monotone": bool(np.all(np.diff(row["cdf"]) >= -1.0e-12)),
        })
    conv_values = np.asarray([r["upper90_raw"] for r in convergence_rows], dtype=float)
    conv_rel = float(np.max(np.abs(np.diff(conv_values)) / max(conv_values[-1], 1.0e-12)))

    # Synthetic coverage of the full Bernoulli procedure at multiple truths.
    coverage_trials = min(int(calibration_config["coverage_trials"]), 20 if quick else int(calibration_config["coverage_trials"]))
    coverage_truths = [float(x) for x in calibration_config["coverage_truth_amplitudes"]]
    rng = np.random.default_rng(int(calibration_config["rng_seed"]) + 7919 + (1 if quick else 0))
    q_on_grid = posterior["q_matrix"]
    coverage_rows = []
    for truth in coverage_truths:
        q_truth = np.asarray([np.interp(truth, parameter_grid, row) for row in q_on_grid], dtype=float)
        covered = 0
        for _ in range(coverage_trials):
            observations = (rng.random(len(events)) < q_truth).astype(int)
            row = posterior_from_efficiencies(
                calibration["q_matrix"], calibration["amplitudes"], parameter_grid,
                prior="uniform", prior_max=float(population["prior_max"]), observations=observations,
            )
            covered += int(truth <= row["upper90"] + 1.0e-12)
        fraction = covered / float(coverage_trials)
        coverage_rows.append({
            "true_amplitude": truth,
            "trials": coverage_trials,
            "covered": covered,
            "coverage_fraction": fraction,
            "coverage_interval_95": _wilson_interval(covered, coverage_trials),
            "target": 0.90,
            "status": "PASS_SYNTHETIC_COVERAGE" if fraction >= 0.75 else "REVIEW_SYNTHETIC_COVERAGE",
        })

    # Resolution and seed probes are part of the reported calibration contract.
    # The analytic evaluator is seed-independent; changing the seed therefore
    # must leave A90 unchanged, while the quadrature ladder bounds deterministic
    # integration error.
    configured_resolutions = [int(x) for x in calibration_config.get("quadrature_resolution_nodes", [8, 16, 32, calibration["resolution"]["quadrature_nodes"]])]
    resolution_nodes = list(dict.fromkeys(min(int(calibration["resolution"]["quadrature_nodes"]), max(4, x)) for x in configured_resolutions))
    resolution_nodes.append(int(calibration["resolution"]["quadrature_nodes"]))
    resolution_nodes = sorted(set(resolution_nodes))
    if quick:
        resolution_nodes = [x for x in resolution_nodes if x <= 24] or [min(24, calibration["resolution"]["quadrature_nodes"])]
    resolution_rows = []
    for nodes in resolution_nodes:
        probe = synthetic_calibration(events, config, quick=quick, quadrature_nodes_override=nodes)
        probe_post = posterior_from_efficiencies(
            probe["q_matrix"], probe["amplitudes"], parameter_grid,
            prior="uniform", prior_max=float(population["prior_max"]),
        )
        resolution_rows.append({
            "quadrature_nodes": nodes,
            "upper90_raw": float(probe_post["upper90"]),
            "upper90": _round_report(probe_post["upper90"], report_decimals),
            "monotone": bool(np.all(np.diff(probe["q_matrix"], axis=1) >= -1.0e-12)),
        })
    seed_values = [int(x) for x in calibration_config.get("seed_probe_values", [calibration["seed"]])]
    seed_rows = []
    for seed in list(dict.fromkeys(seed_values)):
        probe = synthetic_calibration(events, config, quick=quick, seed_override=seed)
        probe_post = posterior_from_efficiencies(
            probe["q_matrix"], probe["amplitudes"], parameter_grid,
            prior="uniform", prior_max=float(population["prior_max"]),
        )
        seed_rows.append({
            "seed": seed,
            "upper90_raw": float(probe_post["upper90"]),
            "upper90": _round_report(probe_post["upper90"], report_decimals),
            "monotone": bool(np.all(np.diff(probe["q_matrix"], axis=1) >= -1.0e-12)),
        })

    # Exact amplitude-efficiency-grid probes isolate the interpolation choice
    # that the previous gate found missing.  The q matrix for each dense grid
    # is recomputed by the exact non-central chi-square evaluator, never by
    # interpolating the configured 15-point efficiency table.
    configured_amplitudes = np.asarray(calibration["amplitudes"], dtype=float)
    configured_amp_points = int(len(configured_amplitudes))
    configured_amp_ladder = [int(x) for x in calibration_config["amplitude_efficiency_grid_resolution_points"]]
    amplitude_grid_points = sorted(set(configured_amp_ladder + [configured_amp_points]))
    if quick:
        amplitude_grid_points = [x for x in amplitude_grid_points if x <= 201]
        if configured_amp_points not in amplitude_grid_points:
            amplitude_grid_points.insert(0, configured_amp_points)
        amplitude_grid_points = sorted(set(amplitude_grid_points))
    amplitude_rows = []
    for points in amplitude_grid_points:
        if points == configured_amp_points:
            probe = calibration
            exact_amplitudes = configured_amplitudes
        else:
            exact_amplitudes = np.linspace(float(population["prior_min"]), float(population["prior_max"]), points)
            probe = synthetic_calibration(
                events,
                config,
                quick=quick,
                amplitude_grid_override=exact_amplitudes,
            )
        probe_post = posterior_from_efficiencies(
            probe["q_matrix"], probe["amplitudes"], parameter_grid,
            prior="uniform", prior_max=float(population["prior_max"]),
        )
        amplitude_rows.append({
            "grid_points": int(points),
            "amplitude_min": float(exact_amplitudes[0]),
            "amplitude_max": float(exact_amplitudes[-1]),
            "grid_kind": "configured_exact_evaluator_grid" if points == configured_amp_points else "uniform_exact_evaluator_grid",
            "exact_evaluator": True,
            "quadrature_nodes": int(probe["resolution"]["quadrature_nodes"]),
            "upper90_raw": float(probe_post["upper90"]),
            "upper90": _round_report(probe_post["upper90"], report_decimals),
            "monotone": bool(np.all(np.diff(probe["q_matrix"], axis=1) >= -1.0e-12)),
            "raw_downward_pairs": int(probe["injections"]["raw_downward_pairs"]),
            "calibrated_downward_pairs": int(probe["injections"]["calibrated_downward_pairs"]),
        })

    resolution_raw = [float(row["upper90_raw"]) for row in resolution_rows]
    seed_raw = [float(row["upper90_raw"]) for row in seed_rows]
    posterior_raw = [float(row["upper90_raw"]) for row in convergence_rows]
    amplitude_raw = [float(row["upper90_raw"]) for row in amplitude_rows]

    # Summarize injection recovery across events without interpreting it as data.
    mean_eff = np.mean(calibration["q_matrix"], axis=0) if events else np.zeros(len(calibration["amplitudes"]))
    injection_summary = [
        {
            "amplitude": float(a),
            "mean_efficiency": float(e),
            "min_efficiency": float(np.min(calibration["q_matrix"][:, j])) if events else None,
            "max_efficiency": float(np.max(calibration["q_matrix"][:, j])) if events else None,
        }
        for j, (a, e) in enumerate(zip(calibration["amplitudes"], mean_eff))
    ]
    calibrated_downward = int(np.sum(np.diff(calibration["q_matrix"], axis=1) < -1.0e-12)) if events else 0
    raw_downward = int(calibration["injections"]["raw_downward_pairs"])

    # Monotonicity is a validation axis in its own right.  It is represented by
    # raw and PAVA-calibrated rows and participates in the union (with no A90
    # movement when the exact evaluator is already monotone).
    monotonicity_rows = [
        {
            "mode": "raw_exact_evaluator",
            "upper90_raw": raw_upper90,
            "raw_downward_pairs": raw_downward,
            "calibrated_downward_pairs": calibrated_downward,
            "max_isotonic_adjustment": float(calibration["injections"]["max_isotonic_adjustment"]),
        },
        {
            "mode": "pava_calibrated",
            "upper90_raw": raw_upper90,
            "raw_downward_pairs": raw_downward,
            "calibrated_downward_pairs": calibrated_downward,
            "max_isotonic_adjustment": float(calibration["injections"]["max_isotonic_adjustment"]),
        },
    ]
    axes = {
        "quadrature": {
            "raw_ladder": resolution_rows,
            "interval": _finite_interval(resolution_raw),
            "span": float(max(resolution_raw) - min(resolution_raw)),
            "status": "PASS_QUADRATURE_CONVERGED" if resolution_raw and max(resolution_raw) - min(resolution_raw) <= 5.0e-4 * max(raw_upper90, 1.0e-12) else "REVIEW_QUADRATURE_CONVERGENCE",
        },
        "seed": {
            "raw_ladder": seed_rows,
            "interval": _finite_interval(seed_raw),
            "span": float(max(seed_raw) - min(seed_raw)),
            "status": "PASS_SEED_INVARIANT" if seed_raw and max(seed_raw) - min(seed_raw) <= 1.0e-12 else "REVIEW_SEED_DEPENDENCE",
        },
        "posterior_grid": {
            "raw_ladder": convergence_rows,
            "interval": _finite_interval(posterior_raw),
            "span": float(max(posterior_raw) - min(posterior_raw)),
            "status": "PASS_NUMERICAL_GRID" if conv_rel <= 5.0e-3 else "REVIEW_NUMERICAL_GRID",
        },
        "amplitude_efficiency_grid": {
            "raw_ladder": amplitude_rows,
            "interval": _finite_interval(amplitude_raw),
            "span": float(max(amplitude_raw) - min(amplitude_raw)),
            "status": "PASS_AMPLITUDE_GRID_MONOTONE" if all(row["monotone"] for row in amplitude_rows) else "REVIEW_AMPLITUDE_GRID_MONOTONICITY",
        },
        "monotonicity": {
            "raw_ladder": monotonicity_rows,
            "interval": _finite_interval([float(row["upper90_raw"]) for row in monotonicity_rows]),
            "span": 0.0,
            "status": "PASS_MONOTONE_CALIBRATION" if calibrated_downward == 0 else "REVIEW_MONOTONE_CALIBRATION",
        },
    }
    axis_intervals = {axis: axes[axis]["interval"] for axis in NUMERICAL_UNCERTAINTY_AXES}
    all_probe_raw = [
        value
        for axis in NUMERICAL_UNCERTAINTY_AXES
        for value in (float(row["upper90_raw"]) for row in axes[axis]["raw_ladder"])
    ]
    union_interval = _finite_interval(all_probe_raw)
    precision_rule = calibration_config["precision_rule"]
    supported_digits = _shared_rounding_precision(union_interval, int(precision_rule["max_decimal_places"]))
    if supported_digits is None:
        raise ValueError("complete numerical union has no supported decimal precision")
    if report_decimals > supported_digits:
        raise ValueError(
            f"configured A90 precision {report_decimals} exceeds complete numerical union support {supported_digits}"
        )
    uncertainty = {
        "method": "deterministic_union_of_quadrature_seed_posterior_amplitude_and_monotonicity_axes",
        "axis_order": list(NUMERICAL_UNCERTAINTY_AXES),
        "axes": axes,
        "axis_intervals": axis_intervals,
        "union_interval": union_interval,
        "overall_interval_raw": union_interval,
        "supported_upper90_decimal_places": int(supported_digits),
        "precision_rule": {
            "name": precision_rule["name"],
            "max_decimal_places": int(precision_rule["max_decimal_places"]),
            "description": "Choose the largest d for which round-half-even(lower,d) equals round-half-even(upper,d) for the complete union interval; fail closed otherwise.",
            "configured_decimal_places": int(report_decimals),
        },
        # Backward-compatible name retained as an outward-rounded display
        # envelope.  The raw union_interval above is the authoritative bound.
        "overall_interval": _outward_rounded_interval(union_interval, int(supported_digits)),
        "overall_span": float(union_interval[1] - union_interval[0]),
        "seed_sensitivity": axes["seed"],
        "resolution_sensitivity": axes["quadrature"],
        "posterior_grid_sensitivity": axes["posterior_grid"],
        "amplitude_efficiency_grid_sensitivity": axes["amplitude_efficiency_grid"],
        "monotonicity_sensitivity": axes["monotonicity"],
    }
    _validate_uncertainty_bundle(uncertainty)
    return {
        "sensitivity_status": "SENSITIVITY_ONLY_CONDITIONAL_SYNTHETIC_NULL",
        "population_parameter": {
            **population,
            "event_likelihood": "product_i [q_i(A)^y_i (1-q_i(A))^(1-y_i)]",
            "event_nuisance": "log-normal eta_i marginalized by deterministic quadrature",
            "observations_used": "all y_i=0 is a synthetic null projection; no observed statistic",
        },
        "events": list(events),
        "n_events": len(events),
        "calibration": {
            key: value for key, value in calibration.items() if key not in ("q_matrix", "amplitudes")
        },
        "injection_recovery_summary": injection_summary,
        "monotonicity": {
            "status": "PASS_MONOTONE_CALIBRATION" if calibrated_downward == 0 else "REVIEW_MONOTONE_CALIBRATION",
            "raw_downward_pairs": raw_downward,
            "calibrated_downward_pairs": calibrated_downward,
            "max_isotonic_adjustment": calibration["injections"]["max_isotonic_adjustment"],
        },
        "calibration_uncertainty": uncertainty,
        "posterior": {
            "upper90": _round_report(raw_upper90, report_decimals),
            "upper90_raw": raw_upper90,
            "grid_points": len(parameter_grid),
            "prior": posterior["prior"],
            "prior_max": posterior["prior_max"],
            "reported_precision_decimal_places": report_decimals,
        },
        "prior_sensitivity": prior_rows,
        "event_inclusion_sensitivity": inclusion,
        "leave_one_event_out": {
            "n_rows": len(loo_values),
            "min_upper90": _round_report(float(np.min(loo_upper)), report_decimals) if len(loo_upper) else None,
            "max_upper90": _round_report(float(np.max(loo_upper)), report_decimals) if len(loo_upper) else None,
            "median_upper90": _round_report(float(np.median(loo_upper)), report_decimals) if len(loo_upper) else None,
            "rows": loo_values,
        },
        "numerical_convergence": {
            "rows": convergence_rows,
            "max_relative_step": conv_rel,
            "status": "PASS_NUMERICAL_GRID" if conv_rel <= 5.0e-3 else "REVIEW_NUMERICAL_GRID",
        },
        "coverage": {
            "rows": coverage_rows,
            "status": "PASS_SYNTHETIC_COVERAGE" if all(row["status"] == "PASS_SYNTHETIC_COVERAGE" for row in coverage_rows) else "REVIEW_SYNTHETIC_COVERAGE",
        },
        "blinded_or_synthetic_recovery": {
            "status": "PASS_SYNTHETIC_ONLY",
            "blinded_real_data": "NOT_RUN_BLOCKED_NO_STRAIN",
            "description": "recovery uses an exact synthetic matched-filter distribution; coverage uses seeded Bernoulli draws; no mock row is promoted to an observed event",
        },
        "selection_effects": {
            "status": "CONDITIONAL_ON_CATALOG_SELECTION",
            "catalog_rate_or_search_volume": "BLOCKED_MISSING_SEARCH_SELECTION_FUNCTION",
            "event_inclusion_variants": list(inclusion),
        },
    }


def assert_result_provenance(result: dict[str, Any]) -> None:
    """Assert all live source/input digests and the fail-closed status.

    This is intentionally stricter than a result self-consistency check: an
    artifact generated before a producer, focused test, manifest, config, or
    catalog edit must fail when it is audited against the current checkout.
    """

    try:
        manifest = _read_p2s3_manifest()
        live_source = source_provenance()
        live = {
            "catalog": sha256_file(CATALOG_PATH),
            "analysis_config": sha256_file(CONFIG_PATH),
            "data_manifest": sha256_file(MANIFEST_PATH),
            "producer_source": sha256_file(SOURCE_PATH),
            "focused_test": sha256_file(TEST_PATH),
        }
    except (OSError, ProvenanceError) as exc:
        raise AssertionError(f"live P2-S6 provenance is not closed: {exc}") from exc
    provenance = result.get("provenance", {})
    catalog = provenance.get("catalog", {})
    if catalog.get("path") != str(CATALOG_PATH.relative_to(ROOT)) or catalog.get("sha256") != live["catalog"] or catalog.get("trust_status") != "verified_public":
        raise AssertionError("result catalog provenance does not match live canonical input")
    config = provenance.get("analysis_config", {})
    if config.get("path") != str(CONFIG_PATH.relative_to(ROOT)) or config.get("sha256") != live["analysis_config"] or config.get("manifest_sha256") != live["data_manifest"]:
        raise AssertionError("result analysis configuration provenance does not match live input manifest/config")
    manifest_result = provenance.get("data_manifest", {})
    if manifest_result.get("sha256") != live["data_manifest"] or manifest_result.get("path") != str(MANIFEST_PATH.relative_to(ROOT)):
        raise AssertionError("result data-manifest provenance does not match live manifest")
    for key in ("producer_source", "focused_test"):
        entry = provenance.get(key, {})
        if entry.get("sha256") != live[key]:
            raise AssertionError(f"result {key} provenance does not match live source")
    declared_source = provenance.get("source_provenance", {})
    if declared_source != live_source:
        raise AssertionError("result source/input provenance bundle does not match live files")
    assertion = provenance.get("source_to_artifact_provenance", {})
    if assertion.get("status") != "PASS":
        raise AssertionError("result source-to-artifact provenance assertion is not PASS")
    for field in (
        "source_sha256", "producer_source_sha256", "test_sha256", "test_source_sha256",
        "manifest_sha256", "input_manifest_sha256", "config_sha256", "input_config_sha256",
        "catalog_sha256", "input_catalog_sha256",
    ):
        if assertion.get(field) != live_source.get(field):
            raise AssertionError(f"result source-to-artifact {field} drift")
    code = manifest.get("code_provenance", {})
    if code.get("producer", {}).get("sha256") != live["producer_source"] or code.get("focused_test", {}).get("sha256") != live["focused_test"]:
        raise AssertionError("live manifest code pins do not match source/test")
    if result.get("observational", {}).get("status") != "BLOCKED_MISSING_STRAIN_POSTERIOR_NOISE_PRODUCTS":
        raise AssertionError("result must retain the observational data block")
    if result.get("sensitivity", {}).get("sensitivity_status") != "SENSITIVITY_ONLY_CONDITIONAL_SYNTHETIC_NULL":
        raise AssertionError("result sensitivity status is not explicitly synthetic/conditional")
    if result.get("observational", {}).get("time_slide_independence", {}).get("status") != "BLOCKED_NO_EMPIRICAL_TIME_SLIDE_BANK":
        raise AssertionError("result must retain the empirical time-slide block")
    calibration = result.get("sensitivity", {}).get("calibration", {})
    if calibration.get("method") != CALIBRATION_METHOD or not calibration.get("deterministic"):
        raise AssertionError("result calibration method is not the pinned deterministic evaluator")
    monotonicity = result.get("sensitivity", {}).get("monotonicity", {})
    if monotonicity.get("status") != "PASS_MONOTONE_CALIBRATION" or monotonicity.get("calibrated_downward_pairs") != 0:
        raise AssertionError("result efficiency calibration is not monotone")
    uncertainty = result.get("sensitivity", {}).get("calibration_uncertainty", {})
    try:
        _validate_uncertainty_bundle(uncertainty)
    except (TypeError, ValueError, KeyError) as exc:
        raise AssertionError(f"result numerical uncertainty is not a complete union envelope: {exc}") from exc
    posterior = result.get("sensitivity", {}).get("posterior", {})
    if posterior.get("reported_precision_decimal_places") != uncertainty.get("supported_upper90_decimal_places"):
        raise AssertionError("result posterior precision does not match complete union support")
    try:
        displayed = float(posterior["upper90"])
        raw = float(posterior["upper90_raw"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AssertionError("result posterior lacks raw/displayed A90 values") from exc
    if displayed != _round_report(raw, int(uncertainty["supported_upper90_decimal_places"])):
        raise AssertionError("result displayed A90 does not follow the declared rounding rule")
    if not (float(uncertainty["union_interval"][0]) <= raw <= float(uncertainty["union_interval"][1])):
        raise AssertionError("result posterior raw A90 lies outside the complete union envelope")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("result schema version drift")


def _make_figure(result: dict[str, Any], posterior: dict[str, Any]) -> None:
    """Write a compact posterior/efficiency figure; figure is sensitivity-only."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    grid = posterior["grid"]
    density = posterior["posterior"]
    axes[0].plot(grid, density, color="#244b74", lw=2)
    upper90_raw = float(result["sensitivity"]["posterior"].get("upper90_raw", posterior["upper90"]))
    upper90_display = result["sensitivity"]["posterior"]["upper90"]
    axes[0].axvline(upper90_raw, color="#a33", ls="--", label=f"90% = {upper90_display:.{result['sensitivity']['posterior']['reported_precision_decimal_places']}f}")
    axes[0].set(xlabel="common echo amplitude ratio A", ylabel="synthetic posterior density", title="Conditional null projection")
    axes[0].legend(frameon=False, fontsize=8)
    inj = result["sensitivity"]["injection_recovery_summary"]
    axes[1].plot([row["amplitude"] for row in inj], [row["mean_efficiency"] for row in inj], marker="o", ms=3, color="#3b7f4d")
    axes[1].axhline(0.9, color="#888", ls=":")
    axes[1].set(xlabel="injected A", ylabel="mean synthetic recovery", ylim=(-0.02, 1.02), title="Injection recovery (mock only)")
    fig.suptitle("P2-S3 GW-echo hierarchical sensitivity; real-data upper limit BLOCKED", fontsize=10)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def build_result(config: dict[str, Any], events: Sequence[dict[str, Any]], audit: dict[str, Any], sensitivity: dict[str, Any], products: dict[str, Any]) -> dict[str, Any]:
    manifest_sha256 = sha256_file(MANIFEST_PATH)
    live_source = source_provenance()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED_OBSERVATIONAL_WITH_MAXIMAL_SENSITIVITY",
        "provenance": {
            "catalog": audit["provenance"],
            "analysis_config": {
                "path": str(CONFIG_PATH.relative_to(ROOT)),
                "sha256": sha256_file(CONFIG_PATH),
                "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
                "manifest_sha256": manifest_sha256,
                "trust_status": "declared_local_configuration",
            },
            "data_manifest": {
                "path": str(MANIFEST_PATH.relative_to(ROOT)),
                "sha256": manifest_sha256,
                "synthetic_not_observational": True,
            },
            "producer_source": {
                "path": str(SOURCE_PATH.relative_to(ROOT)),
                "sha256": sha256_file(SOURCE_PATH),
            },
            "focused_test": {
                "path": str(TEST_PATH.relative_to(ROOT)),
                "sha256": sha256_file(TEST_PATH),
            },
            "source_provenance": live_source,
            "source_to_artifact_provenance": {
                "status": "PASS",
                "source_sha256": live_source["source_sha256"],
                "producer_source_sha256": live_source["producer_source_sha256"],
                "test_sha256": live_source["test_sha256"],
                "test_source_sha256": live_source["test_source_sha256"],
                "manifest_sha256": live_source["manifest_sha256"],
                "input_manifest_sha256": live_source["input_manifest_sha256"],
                "config_sha256": live_source["config_sha256"],
                "input_config_sha256": live_source["input_config_sha256"],
                "catalog_sha256": live_source["catalog_sha256"],
                "input_catalog_sha256": live_source["input_catalog_sha256"],
            },
        },
        "catalog_audit": audit,
        "n_events": len(events),
        "observational": products,
        "sensitivity": sensitivity,
        "claims": {
            "observational_upper_limit": "BLOCKED_NO_EVENT_LEVEL_STRAIN_POSTERIOR_NOISE",
            "joint_credible_limit": "not_returned_as_observation; sensitivity-only A90 is conditional on synthetic y_i=0",
            "interesting_excess": "none evaluated; no event statistic exists in this checkout",
        },
    }


def run(quick: bool = False, write_artifacts: bool = True) -> dict[str, Any]:
    config = load_config()
    events, audit = load_catalog(config)
    products = check_real_data_products()
    sensitivity = run_sensitivity(events, config, quick=quick)
    result = build_result(config, events, audit, sensitivity, products)
    # Attach provenance before the assertion; this assertion is also used by tests.
    assert_result_provenance(result)
    if write_artifacts:
        RESULT_PATH.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Reconstruct only the posterior needed for the figure, avoiding any change to
        # the deterministic result payload after it has been validated.
        calibration = synthetic_calibration(events, config, quick=quick)
        population = config["population_parameter"]
        grid = np.linspace(float(population["prior_min"]), float(population["prior_max"]), min(int(population["posterior_grid_points"]), 401 if quick else int(population["posterior_grid_points"])))
        posterior = posterior_from_efficiencies(calibration["q_matrix"], calibration["amplitudes"], grid, prior="uniform", prior_max=float(population["prior_max"]))
        _make_figure(result, posterior)
    return result


def _terminal_summary(result: dict[str, Any]) -> str:
    sensitivity = result["sensitivity"]
    products = result["observational"]
    return "\n".join(
        [
            f"status={result['status']}",
            f"catalog_events={result['catalog_audit']['raw_rows']} unique={result['catalog_audit']['unique_events']} selected={result['sensitivity']['n_events']} skipped={result['catalog_audit']['selection']['skipped']}",
            f"duplicates={result['catalog_audit']['duplicate_event_groups']} groups rows_removed={result['catalog_audit']['duplicate_rows_removed']}",
            f"observational={products['status']} missing={','.join(products['missing'])}",
            f"synthetic_threshold={sensitivity['calibration']['background']['threshold']:.6f} null_fap={sensitivity['calibration']['background']['null_calibration']['false_alarm_fraction']:.4f}",
            f"synthetic_A90={sensitivity['posterior']['upper90']:.{sensitivity['posterior']['reported_precision_decimal_places']}f} raw={sensitivity['posterior']['upper90_raw']:.9f} prior={sensitivity['posterior']['prior']} events={sensitivity['n_events']}",
            f"numerical_union=[{sensitivity['calibration_uncertainty']['union_interval'][0]:.9f},{sensitivity['calibration_uncertainty']['union_interval'][1]:.9f}] supported_digits={sensitivity['calibration_uncertainty']['supported_upper90_decimal_places']}",
            f"coverage={sensitivity['coverage']['status']} convergence={sensitivity['numerical_convergence']['status']}",
            f"result={RESULT_PATH.relative_to(ROOT)} figure={FIGURE_PATH.relative_to(ROOT)}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small deterministic calibration for focused tests (not terminal evidence)")
    args = parser.parse_args(argv)
    try:
        result = run(quick=args.quick, write_artifacts=True)
    except (ProvenanceError, ValueError, AssertionError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(_terminal_summary(result))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI verification
    raise SystemExit(main())
