"""Focused tests for the source-action minimal-vacuum tree-cut audit."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_source_complete_minimal_vacuum_closure_audit as audit  # noqa: E402
from source_complete_solution_audit import PARAMS  # noqa: E402


@pytest.fixture(scope="module")
def result() -> dict:
    return audit.build_result()


def _fraction(record: dict) -> Fraction:
    return Fraction(int(record["exact_numerator"]), int(record["exact_denominator"]))


def test_live_action_surface_and_exact_mass_relations(result: dict):
    exact = audit.exact_parameters_from_live()
    assert exact.W0 == Fraction("859")
    assert exact.lambda_baseline == Fraction("1.05")
    assert exact.M_N == Fraction("939")
    assert exact.m_omega == Fraction("782.6")
    assert exact.g_s == Fraction("939") / Fraction("859")
    assert exact.q_phi == Fraction("782.6") / Fraction("859")
    assert exact.m_A_squared == Fraction("782.6") ** 2
    assert exact.m_sigma_squared == 2 * Fraction("1.05") * Fraction("859") ** 2

    derived = result["action_inputs"]["derived_from_bound_action"]
    assert _fraction(derived["m_A_squared_MeV2"]) == exact.m_A_squared
    assert _fraction(derived["m_sigma_squared_MeV2"]) == exact.m_sigma_squared
    assert mp.mpf(derived["m_A_MeV_display"]) == mp.mpf("782.6")
    assert abs(mp.mpf(derived["m_sigma_MeV_display"]) - mp.sqrt(mp.mpf("1549550.1"))) < mp.mpf("1e-28")


def test_minimal_sector_contract_excludes_unprovided_or_unphysical_channels(result: dict):
    contract = result["minimal_sector_contract"]
    assert contract["parameter_fit_performed"] is False
    assert contract["empirical_target_or_data_read"] is False
    assert contract["physical_claim_permitted"] is False
    assert any("theta" in item for item in contract["not_counted_as_physical_final_particles"])
    assert any("A0" in item for item in contract["not_counted_as_physical_final_particles"])
    assert any("L_spectator" in item for item in contract["excluded"])
    assert any("sigma*A_mu*A^mu" in item for item in contract["displayed_vertices_used"])


def test_exact_mass_ordering_and_all_displayed_two_body_cuts_are_closed(result: dict):
    ordering = result["mass_ordering"]
    assert ordering["strict_mass_ordering"] == "m_A < M_N < m_sigma < 2*m_A < 2*M_N"
    gaps = ordering["squared_gap_certificates"]
    assert _fraction(gaps["M_N_squared_minus_m_A_squared"]) == Fraction("269258.24")
    assert _fraction(gaps["m_sigma_squared_minus_M_N_squared"]) == Fraction("667829.1")
    assert _fraction(gaps["four_m_A_squared_minus_m_sigma_squared"]) == Fraction("900300.94")
    assert all(item["strictly_positive"] is True for item in gaps.values())

    cuts = result["leading_tree_two_body_cuts"]
    assert cuts["all_listed_cuts_closed"] is True
    channels = {row["identifier"]: row for row in cuts["channels"]}
    assert set(channels) == {
        "sigma_to_N_Nbar",
        "A_to_N_Nbar",
        "sigma_to_A_A",
        "N_to_N_sigma",
        "N_to_N_A",
        "A_to_A_sigma",
        "sigma_to_sigma_sigma",
    }
    assert _fraction(channels["sigma_to_N_Nbar"]["closure_gap_squared"]) == Fraction("1977333.9")
    assert _fraction(channels["A_to_N_Nbar"]["closure_gap_squared"]) == Fraction("2914421.24")
    assert _fraction(channels["sigma_to_A_A"]["closure_gap_squared"]) == Fraction("900300.94")
    assert _fraction(channels["N_to_N_sigma"]["positive_emitted_mass_squared"]) == Fraction("1549550.1")
    assert _fraction(channels["N_to_N_A"]["positive_emitted_mass_squared"]) == Fraction("612462.76")
    assert _fraction(channels["A_to_A_sigma"]["positive_emitted_mass_squared"]) == Fraction("1549550.1")
    assert _fraction(channels["sigma_to_sigma_sigma"]["positive_emitted_mass_squared"]) == Fraction("1549550.1")
    for channel in channels.values():
        assert channel["status"] == "KINEMATICALLY_CLOSED"
        certificate = channel.get("closure_gap_squared", channel.get("positive_emitted_mass_squared"))
        assert certificate["strictly_positive"] is True
        assert channel["phase_space_theta"].endswith("=0")
        assert channel["minimal_tree_vacuum_width"].startswith("0 ")


def test_lambda_diagnostic_gates_are_derived_not_fitted(result: dict):
    exact = audit.exact_parameters_from_live()
    sigma_AA = audit.lambda_sigma_to_AA_threshold(exact)
    sigma_NN = audit.lambda_sigma_to_NN_threshold(exact)
    assert sigma_AA == 2 * exact.q_phi**2
    assert sigma_NN == 2 * exact.g_s**2
    assert exact.lambda_baseline < sigma_AA < sigma_NN

    gates = result["hypothetical_lambda_diagnostic_gates"]
    assert _fraction(gates["sigma_to_A_A_closure_threshold"]) == sigma_AA
    assert _fraction(gates["sigma_to_N_Nbar_closure_threshold"]) == sigma_NN
    assert gates["baseline_is_below_both_opening_thresholds"] is True
    assert "not a parameter recommendation" in gates["lambda_only_hypothetical_deformation"]


def test_canonical_result_rejects_semantic_and_provenance_mutations(result: dict):
    assert audit.validate_result(result) is None
    parsed = json.loads(json.dumps(result, allow_nan=False))
    assert audit.validate_result(parsed) is None
    mutations = (
        lambda payload: payload.__setitem__("status", "VACUUM_PARTICLE_STABLE"),
        lambda payload: payload["minimal_sector_contract"].__setitem__("parameter_fit_performed", True),
        lambda payload: payload["leading_tree_two_body_cuts"].__setitem__("all_listed_cuts_closed", False),
        lambda payload: payload["leading_tree_two_body_cuts"]["channels"][0].__setitem__("status", "OPEN"),
        lambda payload: payload["provenance"].__setitem__("producer_sha256", "0" * 64),
    )
    for mutate in mutations:
        changed = copy.deepcopy(result)
        mutate(changed)
        with pytest.raises(audit.MinimalVacuumClosureError):
            audit.validate_result(changed)


def test_nonlive_or_broken_parameter_surfaces_fail_closed():
    with pytest.raises(audit.MinimalVacuumClosureError):
        audit.build_result(replace(PARAMS, lam=0.1))
    with pytest.raises(audit.MinimalVacuumClosureError):
        audit.exact_parameters_from_live(replace(PARAMS, W0=0.0))
    with pytest.raises(audit.MinimalVacuumClosureError):
        audit.exact_parameters_from_live(replace(PARAMS, M_N=0.0))


def test_ambient_precision_cannot_change_canonical_result():
    before = mp.mp.dps
    try:
        mp.mp.dps = 10
        low = audit._canonical_json(audit.build_result())
        mp.mp.dps = 120
        high = audit._canonical_json(audit.build_result())
    finally:
        mp.mp.dps = before
    assert low == high


def test_cli_regeneration_and_validation_are_canonical(tmp_path: Path):
    output = tmp_path / "minimal-vacuum-closure.json"
    script = HERE / "nvg_source_complete_minimal_vacuum_closure_audit.py"
    generated = subprocess.run(
        [sys.executable, "-B", str(script), "--output", str(output)],
        cwd=HERE.parent,
        text=True,
        capture_output=True,
        check=True,
    )
    stdout = json.loads(generated.stdout)
    disk = json.loads(output.read_text(encoding="utf-8"))
    assert stdout == disk
    before = output.read_bytes()
    checked = subprocess.run(
        [sys.executable, "-B", str(script), "--validate", str(output)],
        cwd=HERE.parent,
        text=True,
        capture_output=True,
        check=True,
    )
    assert checked.stdout.strip() == "VALIDATION_PASS"
    assert output.read_bytes() == before


def test_source_is_self_contained_and_does_not_import_fit_or_network_surfaces():
    source = (HERE / "nvg_source_complete_minimal_vacuum_closure_audit.py").read_text(encoding="utf-8")
    assert "from source_complete_solution_audit import PARAMS, Params" in source
    for forbidden in (
        "source_complete_scaling_saturation_audit",
        "inverse_potential_jet",
        "nvg_finite_droplet_audit",
        "requests",
        "urllib",
    ):
        assert forbidden not in source


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
