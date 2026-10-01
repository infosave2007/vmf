"""Focused semantic tests for the source-complete static NN exchange audit."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_source_complete_nn_static_audit as audit  # noqa: E402
from source_complete_solution_audit import PARAMS  # noqa: E402


def _fraction(row: dict) -> Fraction:
    return Fraction(int(row["exact_numerator"]), int(row["exact_denominator"]))


class TestSourceCompleteNNStaticAudit:
    def test_live_action_derivation_and_rational_certificates(self) -> None:
        result = audit.build_result()
        audit.validate_result(result)

        inputs = result["action_inputs"]
        base = inputs["base_live_decimal_inputs"]
        derived = inputs["derived_from_bound_action"]
        assert _fraction(base["W0_MeV"]) == Fraction("859")
        assert _fraction(base["lambda"]) == Fraction("1.05")
        assert _fraction(base["M_N_MeV"]) == Fraction("939")
        assert _fraction(base["m_omega_MeV"]) == Fraction("782.6")
        assert _fraction(base["g_omega"]) == Fraction("10.12")
        assert _fraction(derived["g_s_M_N_over_W0"]) == Fraction("939") / Fraction("859")
        assert _fraction(derived["q_phi_momega_over_W0"]) == Fraction("782.6") / Fraction("859")
        assert _fraction(derived["m_sigma_squared_MeV2"]) == 2 * Fraction("1.05") * Fraction("859") ** 2
        assert inputs["coupling_assignment"]["vector_exchange_nucleon_vertex"] == "g_omega (not q_phi)"

        expansion = result["quadratic_vacuum_expansion"]
        assert expansion["no_sigma_A_bilinear"] is True
        assert "2 lambda W0^2" in expansion["radial_mass_definition"]
        assert "g_omega" in expansion["leading_vertices"]
        assert "q_phi" not in expansion["leading_vertices"]

        exact = audit.exact_parameters_from_live(PARAMS)
        inequalities = result["certificate"]["exact_rational_inequalities"]
        assert _fraction(inequalities["g_omega_squared_minus_g_s_squared"]) == exact.g_omega_squared - exact.g_s_squared
        assert _fraction(inequalities["m_sigma_squared_minus_m_A_squared"]) == exact.m_sigma_squared - exact.m_A_squared
        assert all(row["strictly_positive"] is True for row in inequalities.values())
        assert result["certificate"]["proof"]["coordinate_strictly_positive"] is True
        assert result["certificate"]["proof"]["momentum_strictly_positive"] is True
        assert result["certificate"]["proof"]["static_hamiltonian_nonnegative"] is True

    def test_dimensionless_diagnostics_are_live_and_physically_scoped(self) -> None:
        result = audit.build_result()
        diagnostics = result["diagnostics"]
        assert mp.mpf(diagnostics["coordinate_repulsion_to_attraction_infimum"]) > 1
        assert mp.mpf(diagnostics["momentum_repulsion_to_attraction_infimum"]) > 1
        assert mp.mpf(diagnostics["mass_ratio_m_sigma_over_m_A"]) > 1
        assert mp.mpf(diagnostics["static_contact_net_MeV_minus2"]) > 0
        assert mp.mpf(diagnostics["tree_loop_counting_gomega2_over_16pi2"]) > mp.mpf("0.6")
        assert "not error bars" in diagnostics["loop_counting_interpretation"]
        assert result["scope"]["parameter_fit_performed"] is False
        assert result["scope"]["not_an_empirical_comparison"] is True
        assert result["evidence_weight"] == 0.0
        assert result["claim_scope"] == audit.CLAIM_SCOPE
        assert any("No empirical NN table" in item for item in result["caveats"])
        assert any("not a physical NN-channel prediction" in item for item in result["caveats"])
        assert any("gauge-invariant dressed" in item for item in result["caveats"])

    def test_ambient_precision_cannot_change_exact_serialization(self) -> None:
        before = mp.mp.dps
        try:
            mp.mp.dps = 10
            low = audit._canonical_json(audit.build_result())
            mp.mp.dps = 120
            high = audit._canonical_json(audit.build_result())
        finally:
            mp.mp.dps = before
        assert low == high

    @pytest.mark.parametrize(
        "path",
        [
            ("action_inputs", "coupling_assignment", "vector_exchange_nucleon_vertex"),
            ("certificate", "proof", "momentum_strictly_positive"),
            ("scope", "parameter_fit_performed"),
        ],
    )
    def test_semantic_mutations_fail_closed(self, path: tuple[str, str, str]) -> None:
        result = copy.deepcopy(audit.build_result())
        value: dict = result
        for key in path[:-1]:
            value = value[key]
        leaf = path[-1]
        if isinstance(value[leaf], bool):
            value[leaf] = not value[leaf]
        else:
            value[leaf] = "q_phi (wrong)"
        with pytest.raises(audit.StaticExchangeAuditError):
            audit.validate_result(result, check_provenance=False)

    def test_exact_inequality_mutation_fails_closed(self) -> None:
        result = copy.deepcopy(audit.build_result())
        row = result["certificate"]["exact_rational_inequalities"]["static_contact_vector_minus_scalar"]
        row["exact_numerator"] = str(int(row["exact_numerator"]) + 1)
        with pytest.raises(audit.StaticExchangeAuditError):
            audit.validate_result(result, check_provenance=False)

    @pytest.mark.parametrize(
        "path,value",
        [
            (("evidence_weight",), False),
            (("diagnostics", "alpha_vector_gomega2_over_4pi"), "bogus"),
            (("diagnostics", "range_vector_hbarc_over_m_A_fm"), "-999"),
            (("quadratic_vacuum_expansion", "quadratic_lagrangian"), "L2=wrong vector sign"),
            (("quadratic_vacuum_expansion", "leading_vertices"), "L_int=+g_s sigma bar N N"),
            (("scope", "included"), ["full fitted NN/QCD theory"]),
        ],
    )
    def test_every_serialized_claim_is_freshly_rederived(
        self, path: tuple[str, ...], value: object
    ) -> None:
        result = copy.deepcopy(audit.build_result())
        target: dict = result
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(audit.StaticExchangeAuditError):
            audit.validate_result(result, check_provenance=False)

    def test_provenance_digest_mutation_fails_when_validating_saved_artifact(self) -> None:
        result = copy.deepcopy(audit.build_result())
        result["provenance"]["source_complete_sha256"] = "0" * 64
        with pytest.raises(audit.StaticExchangeAuditError):
            audit.validate_result(result)

    @pytest.mark.parametrize(
        "container,key,value",
        [
            ("action_inputs", "m_sigma_squared_MeV2", None),
            ("quadratic_vacuum_expansion", "no_sigma_A_bilinear", False),
            ("certificate", "coordinate_natural", "V(r)=[-g_omega^2 exp(-m_A r)-g_s^2 exp(-m_sigma r)]/(4*pi*r)"),
        ],
    )
    def test_mass_mixing_and_vector_sign_mutations_fail_closed(
        self, container: str, key: str, value: object
    ) -> None:
        result = copy.deepcopy(audit.build_result())
        if container == "action_inputs":
            row = result[container]["derived_from_bound_action"][key]
            row["exact_numerator"] = str(int(row["exact_numerator"]) + 1)
        elif container == "certificate":
            result[container]["kernel_definitions"][key] = value
        else:
            result[container][key] = value
        with pytest.raises(audit.StaticExchangeAuditError):
            audit.validate_result(result, check_provenance=False)

    def test_cli_regeneration_and_validation_are_canonical(self, tmp_path: Path) -> None:
        output = tmp_path / "nn-static.json"
        script = HERE / "nvg_source_complete_nn_static_audit.py"
        generated = subprocess.run(
            [sys.executable, "-B", str(script), "--output", str(output)],
            cwd=HERE.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        assert generated.returncode == 0, generated.stderr
        first = output.read_bytes()
        checked = subprocess.run(
            [sys.executable, "-B", str(script), "--validate", str(output)],
            cwd=HERE.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        assert checked.returncode == 0, checked.stderr
        regenerated = subprocess.run(
            [sys.executable, "-B", str(script), "--output", str(output)],
            cwd=HERE.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        assert regenerated.returncode == 0, regenerated.stderr
        assert output.read_bytes() == first
        payload = json.loads(first)
        assert payload["schema"] == audit.SCHEMA
        assert payload["status"] == audit.STATUS

    def test_source_does_not_import_legacy_eos_or_empirical_data(self) -> None:
        source = (HERE / "nvg_source_complete_nn_static_audit.py").read_text(encoding="utf-8")
        assert "from source_complete_solution_audit import PARAMS, Params" in source
        assert "source_complete_scaling_saturation_audit" not in source
        assert "nvg_isospin_jet_audit" not in source
        assert "nvg_finite_droplet_audit" not in source
        assert "requests" not in source
        assert "urllib" not in source
