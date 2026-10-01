"""Focused tests for the conditional source-action vortex-sector audit."""

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

import nvg_source_complete_conditional_vortex_sector_audit as audit  # noqa: E402
from source_complete_solution_audit import PARAMS  # noqa: E402


@pytest.fixture(scope="module")
def result() -> dict:
    return audit.build_result()


def _fraction(record: dict) -> Fraction:
    return Fraction(int(record["exact_numerator"]), int(record["exact_denominator"]))


def test_live_higgs_mass_relations_and_distinct_couplings(result: dict):
    exact = audit.exact_parameters_from_live()
    assert exact.W0 == Fraction("859")
    assert exact.lambda_baseline == Fraction("1.05")
    assert exact.q_phi == Fraction("782.6") / Fraction("859")
    assert exact.g_s == Fraction("939") / Fraction("859")
    assert exact.m_A_squared == Fraction("782.6") ** 2
    assert exact.m_sigma_squared == 2 * Fraction("1.05") * Fraction("859") ** 2
    assert exact.q_phi != Fraction("10.12")

    source_free = result["source_free_local_truncation"]
    assert source_free["no_new_operator_or_particle_introduced"] is True
    assert source_free["parameter_fit_performed"] is False
    assert source_free["empirical_target_or_data_read"] is False
    assert source_free["physical_claim_permitted"] is False
    assert "q_phi" in source_free["action"]


def test_exact_bps_and_type_ii_certificates(result: dict):
    exact = audit.exact_parameters_from_live()
    bps = audit.lambda_BPS(exact)
    assert bps == exact.q_phi**2 / 2
    assert exact.lambda_baseline > bps
    assert exact.beta == 2 * exact.lambda_baseline / exact.q_phi**2
    assert exact.kappa_GL_squared == exact.lambda_baseline / exact.q_phi**2
    assert exact.beta > 1
    assert exact.kappa_GL_squared > Fraction(1, 2)

    classification = result["local_abelian_higgs_classification"]
    assert classification["local_classification"] == "TYPE_II_NOT_BPS"
    assert classification["BPS_equals_scalar_vector_mass_equality_threshold"] is True
    assert _fraction(classification["lambda_BPS"]) == bps
    assert _fraction(classification["beta"]) == exact.beta
    assert _fraction(classification["kappa_GL_squared"]) == exact.kappa_GL_squared
    assert _fraction(classification["beta_minus_one"]) > 0
    assert _fraction(classification["kappa_GL_squared_minus_one_half"]) > 0
    assert mp.mpf(classification["kappa_GL_display"]) > 1 / mp.sqrt(2)
    assert mp.mpf(classification["mass_ratio_m_sigma_over_m_A_display"]) > 1


def test_global_completion_and_boundary_problem_remain_explicitly_conditional(result: dict):
    bvp = result["conditional_vortex_boundary_problem"]
    assert bvp["global_completion_status"] == "UNDECLARED_BY_BOUND_CONTRACT"
    assert bvp["numerical_vortex_solution_published_by_this_audit"] is False
    assert bvp["baseline_BPS_equations_or_tension_claimed"] is False
    assert any("compact U(1)" in item for item in bvp["required_extra_global_assumptions"])
    assert any("charge lattice" in item for item in bvp["not_declared_by_bound_contract"])
    assert len(bvp["conditional_BVP"]) == 2
    assert bvp["conditional_flux_quantum_for_integer_n"] == "Phi_B=2*pi*n/q_phi"
    assert mp.mpf(bvp["conditional_flux_unit_display"]) > 0
    assert mp.mpf(bvp["BPS_tension_unit_MeV2_display_not_baseline_result"]) > 0


def test_canonical_result_rejects_semantic_and_provenance_mutations(result: dict):
    assert audit.validate_result(result) is None
    parsed = json.loads(json.dumps(result, allow_nan=False))
    assert audit.validate_result(parsed) is None
    mutations = (
        lambda payload: payload.__setitem__("status", "VORTEX_DISCOVERED"),
        lambda payload: payload["source_free_local_truncation"].__setitem__("parameter_fit_performed", True),
        lambda payload: payload["local_abelian_higgs_classification"].__setitem__("local_classification", "BPS"),
        lambda payload: payload["conditional_vortex_boundary_problem"].__setitem__("global_completion_status", "PROVEN"),
        lambda payload: payload["provenance"].__setitem__("producer_sha256", "0" * 64),
    )
    for mutate in mutations:
        changed = copy.deepcopy(result)
        mutate(changed)
        with pytest.raises(audit.ConditionalVortexSectorError):
            audit.validate_result(changed)


def test_nonlive_or_broken_parameter_surfaces_fail_closed():
    with pytest.raises(audit.ConditionalVortexSectorError):
        audit.build_result(replace(PARAMS, lam=0.1))
    with pytest.raises(audit.ConditionalVortexSectorError):
        audit.exact_parameters_from_live(replace(PARAMS, W0=0.0))
    with pytest.raises(audit.ConditionalVortexSectorError):
        audit.exact_parameters_from_live(replace(PARAMS, m_omega=0.0))


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
    output = tmp_path / "conditional-vortex.json"
    script = HERE / "nvg_source_complete_conditional_vortex_sector_audit.py"
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
    source = (HERE / "nvg_source_complete_conditional_vortex_sector_audit.py").read_text(encoding="utf-8")
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
