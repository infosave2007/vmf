"""Focused regressions for the source-complete Q4 finite-TF droplet audit."""

from __future__ import annotations

import copy
import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_source_complete_q4_droplet_audit as audit
from source_complete_solution_audit import PARAMS


@pytest.fixture(scope="module")
def result() -> dict:
    return audit.build_result()


def test_live_action_parameters_and_sufficient_constant_are_not_fitted():
    values = audit.action_inputs()
    assert values["d"] == 4
    assert values["g_s_M_N_over_W0"] == pytest.approx(values["M_N_MeV"] / values["W0_MeV"])
    assert values["q_phi_momega_over_W0"] == pytest.approx(values["m_omega_MeV"] / values["W0_MeV"])
    assert audit.scalar_coercivity_constant() == pytest.approx(2.7216394942434)
    assert audit.C_CRITICAL == pytest.approx(0.25)
    assert audit.scalar_coercivity_constant() > audit.C_CRITICAL
    certificate = audit.scalar_coercivity_rational_certificate()
    assert certificate["strict_inequality_used"] == "pi > 3"
    assert certificate["strict_lower_C_exceeds_C_critical"] is True


def test_exact_fermi_and_jensen_plus_quartic_bound_hold_on_representative_domains():
    C = audit.scalar_coercivity_constant()
    for y in (0.0, 1.0e-7, 0.1, 0.73, 0.999, 1.0, 1.5):
        for z in (1.0e-5, 0.01, 0.2, 1.0, 10.0):
            exact_gap = (
                audit.fermi_energy_per_baryon_over_M(y, z)
                + audit.potential_per_baryon_over_M(y, z, C)
                - 1.0
            )
            sufficient_gap = audit.sufficient_local_gap_lower_bound(y, z, C)
            assert exact_gap > 0.0
            assert sufficient_gap >= -1.0e-13
            assert exact_gap + 1.0e-11 * max(1.0, abs(exact_gap)) >= sufficient_gap


def test_small_ratio_fermi_series_avoids_closed_form_cancellation():
    ratio = 1.0e-3
    expected = (
        1.0
        + 3.0 * ratio**2 / 10.0
        - 3.0 * ratio**4 / 56.0
        + ratio**6 / 48.0
        - 15.0 * ratio**8 / 1408.0
    )
    assert audit.fermi_energy_per_baryon_over_M(1.0, ratio) == pytest.approx(expected, abs=1.0e-15)


def test_sharp_jensen_threshold_has_the_known_equality_and_fails_below_it():
    # At y=0,z=1 the exact d=4 Fermi value is 3/4.  U/(Mn)=C there,
    # so C=1/4 is the sharp threshold of this certificate, not a fitted
    # convenience constant.
    fermi = audit.fermi_energy_per_baryon_over_M(0.0, 1.0)
    at_threshold = fermi + audit.potential_per_baryon_over_M(0.0, 1.0, 0.25) - 1.0
    below_threshold = fermi + audit.potential_per_baryon_over_M(0.0, 1.0, 0.249999) - 1.0
    assert fermi == pytest.approx(0.75)
    assert at_threshold == pytest.approx(0.0, abs=1.0e-15)
    assert below_threshold < 0.0


def test_full_certificate_has_a_scoped_no_self_binding_result(result: dict):
    assert result["schema_version"] == audit.SCHEMA
    assert result["status"] == audit.STATUS
    assert result["evidence_weight"] == 0.0
    assert result["analytic_variational_bound"]["pass"]
    assert result["analytic_variational_bound"]["criterion_is_sufficient_not_fitted"]
    assert result["analytic_variational_bound"]["sample_count"] > 1000
    assert "full NVG" in result["conclusion"]["not_claimed"][0]
    assert "unbound" in result["conclusion"]["not_claimed"][2]


def test_finite_radial_controls_resolve_gauss_and_never_promote_profiles_to_droplets(result: dict):
    controls = result["finite_radial_numerical_controls"]
    expected_rows = (
        len(audit.N_B_LADDER)
        * len(audit.BOXES_FM)
        * len(audit.GRID_CELLS_PER_FM)
        * len(audit.PROFILE_CONTROLS)
    )
    assert controls["row_count"] == expected_rows
    assert controls["pass"]
    assert controls["all_Gauss_constraints_pass"]
    assert controls["all_positive_gap_checks_pass"]
    assert controls["grid_and_box_control_pass"]
    for row in controls["rows"]:
        assert row["nonstationary_control_only"] is True
        assert row["conserved_N_relative_error"] <= 2.0e-15
        assert row["total_E_minus_NM_MeV"] > 0.0
        assert row["E_over_N_minus_M_N_MeV"] > 0.0
        assert row["Gauss_constraint"]["gauss_residual_relative"] <= audit.GAUSS_RELATIVE_LIMIT
        assert row["Gauss_constraint"]["gauss_identity_relative"] <= audit.GAUSS_RELATIVE_LIMIT
        assert row["Gauss_constraint"]["min_vector_energy_positive"] is True


def test_mutated_action_assumptions_fail_closed():
    with pytest.raises(audit.Q4DropletAuditError):
        audit.action_inputs(replace(PARAMS, degeneracy=2))
    with pytest.raises(audit.Q4DropletAuditError):
        audit.action_inputs(replace(PARAMS, g_omega=-1.0))
    below = audit.analytic_bound_controls(replace(PARAMS, lam=0.09))
    assert below["criterion_C_strictly_gt_1_over_4"] is False
    assert below["sufficient_lower_bound_available"] is False
    assert below["pass"] is False
    with pytest.raises(audit.Q4DropletAuditError, match="inconclusive"):
        audit.sufficient_local_gap_lower_bound(0.7, 0.2, 0.24)


def test_result_validation_rejects_any_static_or_semantic_substitution(result: dict):
    assert audit.validate_result(result)
    parsed = json.loads(json.dumps(result, allow_nan=False))
    assert audit.validate_result(parsed)
    for mutate in (
        lambda payload: payload.__setitem__("status", "DROPLET_FOUND"),
        lambda payload: payload["analytic_variational_bound"].__setitem__("C", 0.0),
        lambda payload: payload["finite_radial_numerical_controls"].__setitem__("all_Gauss_constraints_pass", False),
        lambda payload: payload.__setitem__("source_code_sha256", "0" * 64),
    ):
        changed = copy.deepcopy(result)
        mutate(changed)
        assert audit.validate_result(changed) is False


def test_no_w8_or_inverse_producer_is_reused():
    source = inspect.getsource(audit)
    forbidden = (
        "source_complete_scaling_saturation_audit",
        "inverse_potential_jet",
        "nvg_finite_droplet_audit",
        "target_y",
    )
    assert all(token not in source for token in forbidden)
    assert "from source_complete_solution_audit import PARAMS, Params" in source


def test_cli_writes_only_exact_regeneration_and_validation_is_read_only(tmp_path: Path):
    output = tmp_path / "q4.json"
    completed = subprocess.run(
        [sys.executable, "-B", str(HERE / "nvg_source_complete_q4_droplet_audit.py"), "--output", str(output)],
        cwd=HERE.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    stdout_payload = json.loads(completed.stdout)
    disk_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload == disk_payload
    before = output.read_bytes()
    validated = subprocess.run(
        [sys.executable, "-B", str(HERE / "nvg_source_complete_q4_droplet_audit.py"), "--validate", str(output)],
        cwd=HERE.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    assert validated.stdout.strip() == "VALIDATION_PASS"
    assert output.read_bytes() == before


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
