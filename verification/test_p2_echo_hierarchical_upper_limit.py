"""Focused semantic tests for the predictive-research P2-S3 echo audit."""

from __future__ import annotations

import math
import copy
import sys
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_echo_hierarchical_upper_limit as audit


def test_config_and_catalog_provenance_are_hash_checked():
    config = audit.load_config()
    selected, ledger = audit.load_catalog(config)
    assert config["schema_version"] == "P2-S3-hierarchical-echo-config-v1"
    assert config["calibration"]["calibration_method"] == audit.CALIBRATION_METHOD
    assert config["calibration"]["uncertainty_axes"] == list(audit.NUMERICAL_UNCERTAINTY_AXES)
    assert config["calibration"]["posterior_grid_resolution_points"]
    assert config["calibration"]["amplitude_efficiency_grid_resolution_points"]
    assert audit.sha256_file(audit.CATALOG_PATH) == ledger["provenance"]["sha256"]
    assert ledger["provenance"]["trust_status"] == "verified_public"
    assert len(selected) == ledger["selection"]["selected"]
    assert len({row["event"] for row in selected}) == len(selected)
    assert ledger["selection"]["population_rate_inference"] is False


def test_live_source_and_manifest_provenance_is_closed():
    manifest = audit._read_p2s3_manifest()
    assert manifest["manifest_sha256"] == audit._manifest_content_digest(manifest)
    assert manifest["manifest_bytes_sha256"] == audit._manifest_bytes_digest()
    assert manifest["code_provenance"]["producer"]["sha256"] == audit.sha256_file(audit.SOURCE_PATH)
    assert manifest["code_provenance"]["focused_test"]["sha256"] == audit.sha256_file(audit.TEST_PATH)


def test_real_data_boundary_fails_closed_without_strain_products():
    products = audit.check_real_data_products()
    assert products["status"] == "BLOCKED_MISSING_STRAIN_POSTERIOR_NOISE_PRODUCTS"
    assert set(products["missing"]) == {"strain", "posterior_samples", "noise_psd_or_time_slides"}
    assert products["observational_likelihood"].startswith("BLOCKED_")


def test_synthetic_null_and_injection_recovery_have_declared_separation():
    config = audit.load_config()
    events, _ = audit.load_catalog(config)
    calibration = audit.synthetic_calibration(events[:6], config, quick=True)
    null = calibration["background"]["null_calibration"]
    assert null["status"] == "PASS_SYNTHETIC_NULL"
    assert 0.0 <= null["false_alarm_fraction"] <= 0.10
    q = calibration["q_matrix"]
    assert q.shape == (6, len(calibration["amplitudes"]))
    assert np.all((q >= 0.0) & (q <= 1.0))
    assert float(np.mean(q[:, -1])) > float(np.mean(q[:, 0]))
    assert calibration["injections"]["status"] == "PASS_SYNTHETIC_INJECTION_RECOVERY"
    assert calibration["method"] == audit.CALIBRATION_METHOD
    assert calibration["seed_used_for_calibration"] is False
    assert calibration["injections"]["per_event_draws"] == 0
    assert np.all(np.diff(q, axis=1) >= -1.0e-12)
    assert calibration["injections"]["calibrated_downward_pairs"] == 0


def test_deterministic_calibration_is_seed_invariant_and_resolution_controlled():
    config = audit.load_config()
    events, _ = audit.load_catalog(config)
    first = audit.synthetic_calibration(events[:4], config, quick=True, seed_override=20260828, quadrature_nodes_override=16)
    second = audit.synthetic_calibration(events[:4], config, quick=True, seed_override=20261001, quadrature_nodes_override=16)
    assert np.array_equal(first["q_matrix"], second["q_matrix"])
    coarse = audit.synthetic_calibration(events[:4], config, quick=True, quadrature_nodes_override=8)
    assert np.all(np.diff(coarse["q_matrix"], axis=1) >= -1.0e-12)
    assert np.max(np.abs(first["q_matrix"] - coarse["q_matrix"])) < 2.0e-4


def test_hierarchical_posterior_normalizes_and_handles_event_outcomes():
    amplitudes = np.array([0.0, 0.5, 1.0, 2.0])
    q = np.array([[0.0, 0.4, 0.8, 1.0], [0.0, 0.2, 0.7, 1.0]])
    grid = np.linspace(0.0, 2.0, 401)
    null = audit.posterior_from_efficiencies(q, amplitudes, grid, prior_max=2.0)
    detection = audit.posterior_from_efficiencies(q, amplitudes, grid, prior_max=2.0, observations=[1, 0])
    assert math.isclose(float(np.trapz(null["posterior"], grid)), 1.0, rel_tol=0.0, abs_tol=2e-3)
    assert 0.0 < null["upper90"] <= 2.0
    assert 0.0 < detection["upper90"] <= 2.0
    assert detection["upper90"] >= null["upper90"]


def test_quick_run_is_sensitivity_only_and_observationally_blocked():
    result = audit.run(quick=True, write_artifacts=False)
    assert result["status"] == "BLOCKED_OBSERVATIONAL_WITH_MAXIMAL_SENSITIVITY"
    assert result["observational"]["status"] == "BLOCKED_MISSING_STRAIN_POSTERIOR_NOISE_PRODUCTS"
    assert result["sensitivity"]["sensitivity_status"] == "SENSITIVITY_ONLY_CONDITIONAL_SYNTHETIC_NULL"
    assert result["sensitivity"]["calibration"]["injections"]["status"] == "PASS_SYNTHETIC_INJECTION_RECOVERY"
    assert result["sensitivity"]["coverage"]["status"] == "PASS_SYNTHETIC_COVERAGE"
    assert result["sensitivity"]["numerical_convergence"]["status"] in {"PASS_NUMERICAL_GRID", "REVIEW_NUMERICAL_GRID"}
    assert result["sensitivity"]["monotonicity"]["status"] == "PASS_MONOTONE_CALIBRATION"
    uncertainty = result["sensitivity"]["calibration_uncertainty"]
    assert uncertainty["axis_order"] == list(audit.NUMERICAL_UNCERTAINTY_AXES)
    assert set(uncertainty["axes"]) == set(audit.NUMERICAL_UNCERTAINTY_AXES)
    for axis in audit.NUMERICAL_UNCERTAINTY_AXES:
        entry = uncertainty["axes"][axis]
        assert entry["raw_ladder"]
        assert len(entry["interval"]) == 2
        raw_values = [float(row["upper90_raw"]) for row in entry["raw_ladder"]]
        assert min(raw_values) >= entry["interval"][0] - 1.0e-15
        assert max(raw_values) <= entry["interval"][1] + 1.0e-15
    assert uncertainty["union_interval"] == uncertainty["overall_interval_raw"]
    assert uncertainty["supported_upper90_decimal_places"] == 2
    assert result["sensitivity"]["posterior"]["upper90"] == round(result["sensitivity"]["posterior"]["upper90_raw"], 2)
    assert uncertainty["axes"]["amplitude_efficiency_grid"]["status"] == "PASS_AMPLITUDE_GRID_MONOTONE"
    assert uncertainty["seed_sensitivity"]["status"] == "PASS_SEED_INVARIANT"
    assert uncertainty["resolution_sensitivity"]["status"] == "PASS_QUADRATURE_CONVERGED"
    assert uncertainty["overall_interval"][0] <= result["sensitivity"]["posterior"]["upper90"] <= uncertainty["overall_interval"][1]
    audit.assert_result_provenance(result)
    assert result["claims"]["observational_upper_limit"].startswith("BLOCKED_")


def test_numerical_uncertainty_fails_closed_when_an_axis_is_omitted():
    result = audit.run(quick=True, write_artifacts=False)
    uncertainty = copy.deepcopy(result["sensitivity"]["calibration_uncertainty"])
    del uncertainty["axes"]["posterior_grid"]
    with pytest.raises(ValueError):
        audit._validate_uncertainty_bundle(uncertainty)


def test_exact_amplitude_and_posterior_ladders_expose_raw_values():
    result = audit.run(quick=True, write_artifacts=False)
    uncertainty = result["sensitivity"]["calibration_uncertainty"]
    posterior_rows = uncertainty["axes"]["posterior_grid"]["raw_ladder"]
    amplitude_rows = uncertainty["axes"]["amplitude_efficiency_grid"]["raw_ladder"]
    assert all("upper90_raw" in row and "upper90" in row for row in posterior_rows)
    assert all(row["exact_evaluator"] for row in amplitude_rows)
    assert all(row["monotone"] for row in amplitude_rows)
    assert len({row["grid_points"] for row in amplitude_rows}) >= 2


def test_bad_binary_observation_and_bad_prior_fail_closed():
    q = np.array([[0.2, 0.8]])
    with pytest.raises(ValueError):
        audit.posterior_from_efficiencies(q, [0.0, 1.0], [0.0, 1.0], observations=[2])
    with pytest.raises(ValueError):
        audit.posterior_from_efficiencies(q, [0.0, 1.0], [0.0, 1.0], prior="unknown")
