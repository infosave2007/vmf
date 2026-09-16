"""Semantic tests for the independent EOS/HVH closure audit."""

import json
import copy
from pathlib import Path

import numpy as np
import pytest

import eos_hvh_closure_audit as audit


def test_analytic_powerlaw_reference_is_closed():
    result = audit.analytic_powerlaw_benchmark()
    assert result["status"] == "PASS"
    assert result["max_relative_error"] < 1.0e-8


def test_ideal_beta_reference_has_charge_and_hvh_closure():
    result = audit.ideal_reference()
    controls = result["controls"]
    assert controls["max_charge_residual_fm3"] < 1.0e-9
    assert controls["max_beta_residual_mev"] < 1.0e-8
    assert controls["closure_max_relative"] < 1.0e-5
    assert controls["derivative_convergence_max_relative"] < 1.0e-5


def test_interacting_reconstruction_exposes_rearrangement_and_stationarity():
    result = audit.interacting_reference()
    controls = result["controls"]
    assert controls["closure_gate"]
    assert controls["max_stationarity_residual_mev"] < 1.0e-5
    assert controls["max_charge_residual_fm3"] < 2.0e-8
    assert controls["max_beta_residual_mev"] < 1.0e-8
    assert controls["max_scalar_rearrangement_mev"] > 1.0
    assert controls["max_vector_rearrangement_mev"] > 1.0
    assert controls["naive_closure_max_relative_smooth_mass_branch"] > 10.0 * controls[
        "closure_max_relative_smooth_mass_branch"
    ]


def test_interacting_derivative_convergence_is_explicit():
    result = audit.interacting_reference()
    controls = result["controls"]
    assert controls["sample_count"] == 18
    assert controls["smooth_sample_count"] < controls["sample_count"]
    assert controls["derivative_convergence_max_relative"] < 1.0e-4
    assert set(result["derivative_pressure_mev_fm3"]) == {"0.02", "0.01", "0.005"}


def test_inventory_fails_closed_when_maintained_path_is_absent():
    inventory = audit.maintained_inventory()
    assert inventory["required_path_available"] is False
    assert "verification/eos_hvh_closure_audit.py" not in inventory["consumers_found_by_source_scan"]
    assert inventory["consumer_count"] >= 10
    assert "ABSENT" in inventory["chemical_potential_path"]
    assert "ABSENT" in inventory["stress_tensor_path"]
    result = audit.run_audit()
    assert result["status"] == "FAIL_CLOSED_MAINTAINED_STRESS_CHEMICAL_PATH_ABSENT"
    assert result["terminal_controls"]["maintained_path_gate"] is False
    assert result["terminal_controls"]["overall_scientific_pass"] is False


def test_status_derivation_mutations_fail_closed():
    result = audit.run_audit()
    inventory = result["producer_inventory"]
    ideal = result["ideal_reference"]
    interacting = result["interacting_nvg"]

    # A consistent synthetic AVAILABLE API proves that the status is not
    # permanently pinned to the current missing-path finding.
    available = copy.deepcopy(inventory)
    available["chemical_potential_path"] = "AVAILABLE"
    available["stress_tensor_path"] = "AVAILABLE"
    available["required_path_available"] = True
    status, controls = audit.derive_terminal_controls(available, ideal, interacting)
    assert status == "PASS_HVH_CLOSURE"
    assert controls["maintained_path_gate"] is True
    assert controls["overall_scientific_pass"] is True

    # Contradictory inventory and nested closure drift must never leave a stale
    # top-level PASS.
    contradictory = copy.deepcopy(available)
    contradictory["stress_tensor_path"] = "ABSENT"
    status, controls = audit.derive_terminal_controls(contradictory, ideal, interacting)
    assert status == "FAIL_CLOSED_INVALID_MAINTAINED_INVENTORY"
    assert controls["overall_scientific_pass"] is False

    closure_drift = copy.deepcopy(interacting)
    closure_drift["controls"]["closure_gate"] = False
    status, controls = audit.derive_terminal_controls(available, ideal, closure_drift)
    assert status == "FAIL_CLOSED_INTERACTING_CLOSURE_GATE"
    assert controls["interacting_closure_pass_on_smooth_branch"] is False
    assert controls["overall_scientific_pass"] is False

    analytic_drift = copy.deepcopy(ideal)
    analytic_drift["analytic_powerlaw"]["status"] = "FAIL"
    status, controls = audit.derive_terminal_controls(available, analytic_drift, interacting)
    assert status == "FAIL_CLOSED_ANALYTIC_REFERENCE"
    assert controls["overall_scientific_pass"] is False


def test_malformed_mutations_fail_closed_without_exceptions():
    result = audit.run_audit()
    inventory = copy.deepcopy(result["producer_inventory"])
    ideal = copy.deepcopy(result["ideal_reference"])
    interacting = copy.deepcopy(result["interacting_nvg"])
    inventory["required_path_available"] = "yes"
    status, controls = audit.derive_terminal_controls(inventory, ideal, interacting)
    assert status == "FAIL_CLOSED_INVALID_MAINTAINED_INVENTORY"
    assert controls["overall_scientific_pass"] is False
    ideal["controls"]["closure_max_relative"] = "not-a-number"
    status, controls = audit.derive_terminal_controls(inventory, ideal, interacting)
    assert status == "FAIL_CLOSED_INVALID_MAINTAINED_INVENTORY"
    assert controls["ideal_closure_pass"] is False


def test_generated_result_and_report_are_machine_readable_after_cli_run():
    result_path = Path(audit.RESULT_PATH)
    report_path = Path(audit.REPORT_PATH)
    assert result_path.exists()
    assert report_path.exists()
    # The original deep-physics report lives under the untracked Lunacy
    # evidence tree and does not exist on CI checkouts; the generated
    # artifacts above stay machine-readable there.
    if not Path(audit.ORIGINAL_REPORT_PATH).exists():
        pytest.skip("original Lunacy deep-physics report not present (untracked)")
    assert Path(audit.ORIGINAL_REPORT_PATH).exists()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert result["audit"] == "eos_hvh_closure_audit"
    assert result["status"] in report
    assert "FAIL_CLOSED" in report
    assert np.isfinite(result["interacting_nvg"]["controls"]["closure_max_relative_all"])
