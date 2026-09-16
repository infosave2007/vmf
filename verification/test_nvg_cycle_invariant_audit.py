"""Focused tests for the raw 2x2 cycle-invariant diagnostic."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

import nvg_cycle_invariant_audit as audit


@pytest.fixture(scope="module")
def full_result():
    return audit.compute_audit()


def test_exact_symbolic_identity_is_strictly_checked():
    rows = audit.symbolic_cycle_invariant_checks()
    assert audit.rows_pass(rows)
    assert rows["transfer_identity"] == {"residual": "0", "passed": True}
    assert rows["determinant_identity"] == {"residual": "0", "passed": True}


@pytest.mark.parametrize(
    ("matrix", "classification"),
    [
        ([[1.0, 1.0], [0.0, 1.0]], "nontrivial_jordan"),
        ([[-1.0, 1.0], [0.0, -1.0]], "nontrivial_jordan"),
        ([[2.0, 0.0], [0.0, 0.5]], "hyperbolic"),
    ],
)
def test_structural_classes_are_recomputed(matrix, classification):
    result = audit.analyze_transfer(matrix)
    assert result["classification"] == classification
    assert result["determinant"] == 1.0
    assert result["algebraic_identity_residual"] == 0.0
    assert result["certificate_passed"] is False


@pytest.mark.parametrize(
    ("matrix", "classification"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], "exact_plus_identity"),
        ([[-1.0, 0.0], [0.0, -1.0]], "exact_minus_identity"),
    ],
)
def test_exact_identities_have_positive_identity_invariant(matrix, classification):
    result = audit.analyze_transfer(matrix)
    assert result["classification"] == classification
    assert result["exact_structural"]["determinant_is_one"] is True
    assert result["candidate_available"] is True
    assert result["candidate"]["g_matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert result["candidate"]["spd"] is True
    assert result["candidate"]["min_eigenvalue"] == 1.0
    assert result["candidate"]["condition_number"] == 1.0
    assert result["candidate"]["invariance_residual"] == 0.0
    assert result["certificate_passed"] is True
    assert result["strict_exact_certificate"] is True


def test_exact_structural_predicates_do_not_use_float_determinant_or_trace():
    # The binary64 determinant rounds to one and the binary64 trace/discriminant
    # round to the parabolic values, but the supplied dyadic matrix is not an
    # exact area-preserving Jordan block.
    matrix = [[1.0, 1.0e-9], [-1.0e-9, 1.0]]
    result = audit.analyze_transfer(matrix)
    exact = result["exact_structural"]
    assert result["determinant"] == 1.0
    assert result["trace"] == 2.0
    assert result["classification"] == "near_boundary_unresolved"
    assert exact["determinant"] != "1"
    assert exact["discriminant"] != "0"
    assert exact["determinant_is_one"] is False
    assert exact["trace_is_plus_two"] is True
    assert exact["q_determinant_is_zero"] is False
    assert exact["is_nontrivial_jordan"] is False
    assert result["candidate_available"] is False
    assert result["certificate_passed"] is False


def test_float_rotation_with_unit_rounded_determinant_is_not_exact_certificate():
    matrix = audit._rotation(0.37)
    result = audit.analyze_transfer(matrix)
    assert result["determinant"] == 1.0
    assert result["exact_structural"]["determinant_is_one"] is False
    assert result["classification"] == "elliptic"
    assert result["numerical_certificate_passed"] is True
    assert result["strict_exact_certificate"] is False
    assert result["certificate_passed"] is False


def test_rotation_and_conjugated_rotation_have_spd_raw_candidates():
    examples = {row["label"]: row for row in audit.mathematical_examples()}
    for label in ("analytical_rotation", "conjugated_rotation"):
        analysis = examples[label]["analysis"]
        assert analysis["classification"] == "elliptic"
        assert analysis["candidate_available"] is True
        assert analysis["candidate"]["spd"] is True
        assert analysis["candidate"]["min_eigenvalue"] > 0.0
        assert analysis["candidate"]["condition_number"] >= 1.0
        assert analysis["candidate"]["invariance_residual"] < 1e-12
        assert analysis["numerical_certificate_passed"] is True
        assert analysis["certificate_passed"] is False


def test_near_boundary_and_determinant_drift_never_receive_certificate():
    examples = {row["label"]: row for row in audit.mathematical_examples()}
    near = examples["near_boundary_rotation"]["analysis"]
    drift = examples["determinant_drift"]["analysis"]
    assert near["classification"] == "near_boundary_unresolved"
    assert near["certificate_passed"] is False
    assert near["numerical_acceptance"] is False
    assert drift["classification"] == "determinant_invalid"
    assert drift["determinant_error"] > audit.DETERMINANT_TOLERANCE
    assert drift["candidate"] is None
    assert drift["certificate_passed"] is False


def test_q_identity_uses_raw_determinant_without_normalization():
    base = audit._rotation(0.4)
    matrix = [[1.0000005 * value for value in row] for row in base]
    result = audit.analyze_transfer(matrix)
    assert result["determinant"] != 1.0
    assert result["determinant_error"] < audit.DETERMINANT_TOLERANCE
    assert result["q_determinant_formula"] == pytest.approx(
        result["determinant"] - result["trace"] ** 2 / 4.0)
    assert result["candidate"]["determinant_contract"].startswith("raw determinant")
    assert result["numerical_certificate_passed"] is True
    assert result["certificate_passed"] is False
    assert result["algebraic_identity_residual"] < 1e-15


@pytest.mark.parametrize(
    "bad",
    [None, True, False, "nan", "inf", [[1.0, 2.0]], [[1.0, 2.0], [3.0, float("nan")]],
],
)
def test_input_validation_rejects_bad_shapes_and_nonfinite_values(bad):
    with pytest.raises(ValueError):
        audit.analyze_transfer(bad)
    with pytest.raises(ValueError):
        audit.quadratic_form(bad)


def test_full_result_contains_live_controls_and_visible_hyperbolic_mode(full_result):
    assert full_result["mathematical_checks_passed"] is True
    assert audit.acceptance_from_result(full_result)
    assert len(full_result["live_controls"]) == 4
    labels = [row["label"] for row in full_result["live_controls"]]
    assert labels == ["scalar_ell_2", "scalar_ell_6", "tensor_w_0_n_3", "tensor_w_1_n_3"]
    assert all(row["producer_gate"] is True for row in full_result["live_controls"])
    assert all(row["accepted"] is True for row in full_result["live_controls"])
    classes = {row["label"]: row["analysis"]["classification"]
               for row in full_result["live_controls"]}
    assert classes["scalar_ell_2"] == "elliptic"
    assert classes["scalar_ell_6"] == "elliptic"
    assert classes["tensor_w_0_n_3"] == "elliptic"
    assert classes["tensor_w_1_n_3"] == "hyperbolic"
    assert full_result["summary"]["q_identity_is_algebraic_not_solver_validation"]


def test_acceptance_fails_closed_on_tampering(full_result):
    missing = copy.deepcopy(full_result)
    del missing["live_controls"][0]["analysis"]["determinant"]
    assert not audit.acceptance_from_result(missing)

    matrix_mutation = copy.deepcopy(full_result)
    matrix_mutation["mathematical_examples"][0]["analysis"]["matrix"][0][0] += 1e-3
    assert not audit.acceptance_from_result(matrix_mutation)

    stale_classification = copy.deepcopy(full_result)
    stale_classification["live_controls"][3]["analysis"]["classification"] = "elliptic"
    assert not audit.acceptance_from_result(stale_classification)

    stale_summary = copy.deepcopy(full_result)
    stale_summary["summary"]["live_control_count"] = 3
    assert not audit.acceptance_from_result(stale_summary)


def test_cli_is_json_only_and_does_not_write_default_files(tmp_path):
    command = [sys.executable, audit.SOURCE_PATH]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, env=environment,
        cwd=str(tmp_path),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema"] == audit.SCHEMA
    assert payload["mathematical_checks_passed"] is True
    assert payload["evidence_weight"] == 0
    assert completed.stderr == ""
    assert not list(tmp_path.iterdir())
