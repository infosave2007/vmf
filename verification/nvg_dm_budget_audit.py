#!/usr/bin/env python3
"""Fail-closed reconciliation of the NVG dark-matter bookkeeping.

This replaces a former manual allocation of ``Omega_DM`` between defects,
PBHs and a dark neutron.  The calculation never fills a missing component by
subtracting it from an observed density.  It asks whether live producers
supply a common, independently predicted physical-density normalization.

The audit works in physical densities ``omega_i = Omega_i h^2``.  Planck is a
base-LambdaCDM comparison reference, not an NVG CMB fit.  A successful process
can honestly return ``NOT_CLOSED``: the fail-closed controls worked; the
abundance problem was not solved.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT_PATH = HERE / "data" / "planck2018_dm_budget.json"
SCHEMA_VERSION = "nvg-dm-budget-audit.v4"
STATUS_NOT_CLOSED = "NOT_CLOSED_MISSING_PBH_NORMALIZATION_AND_VIABLE_COGENESIS"
CLOSURE_STATUS = "UNIFIED_ABUNDANCE_NOT_ESTABLISHED"
TOLERANCE = Decimal("1e-24")
PLANCK_SOURCE_URL = "https://arxiv.org/abs/1807.06209"
PLANCK_REFERENCE_PREFIX = "base LambdaCDM, TT,TE,EE+lowE+lensing;"
# Consumer-side provenance invariants, not physical parameters. A producer
# cannot silently relabel this retired input as a prediction or move it.
DARK_NEUTRON_INPUT_ARTIFACT = "verification/data/legacy_dark_neutron_cogenesis_inputs.json"
DARK_NEUTRON_INPUT_STATUS = "LEGACY_UNDOCUMENTED_CONDITIONAL_INPUTS_NOT_MECHANISM"
PLANCK_PARAMETER_PASSPORT = {
    "omega_b": ("dimensionless", "Omega_b h^2"),
    "omega_c": ("dimensionless", "Omega_c h^2"),
    "H0_km_s_Mpc": ("km s^-1 Mpc^-1", "Hubble constant"),
}

import nvg_adm_bl_cogenesis as cogenesis  # noqa: E402
import nvg_baryogenesis_bsm_closure as baryogenesis_closure  # noqa: E402
import nvg_pbh_dark_matter as pbh  # noqa: E402
import nvg_pbh_two_population as pbh_two  # noqa: E402
import nvg_relic_dark_matter as relic  # noqa: E402


def _historical_cogenesis_input_passport() -> dict[str, str]:
    """Verify that the retained chi arithmetic names its legacy inputs.

    This does not upgrade the retired cogenesis branch into a physical
    mechanism. It makes the conditional historical constants auditable by the
    bookkeeping consumer and fails closed if their identity changes.
    """

    passport = cogenesis.historical_input_passport()
    if not isinstance(passport, Mapping) or set(passport) != {
        "input_artifact",
        "input_sha256",
        "input_status",
    }:
        raise ValueError("cogenesis producer returned an invalid historical input passport")
    producer_artifact = cogenesis.INPUT_PATH.relative_to(cogenesis.ROOT).as_posix()
    if producer_artifact != DARK_NEUTRON_INPUT_ARTIFACT:
        raise ValueError("cogenesis producer moved its historical input passport; review required")
    expected_sha256 = hashlib.sha256((ROOT / DARK_NEUTRON_INPUT_ARTIFACT).read_bytes()).hexdigest()
    if (
        passport["input_artifact"] != DARK_NEUTRON_INPUT_ARTIFACT
        or passport["input_sha256"] != expected_sha256
        or passport["input_status"] != DARK_NEUTRON_INPUT_STATUS
    ):
        raise ValueError("cogenesis producer historical input passport does not match live bytes")
    return {
        "input_artifact": DARK_NEUTRON_INPUT_ARTIFACT,
        "input_sha256": expected_sha256,
        "input_status": DARK_NEUTRON_INPUT_STATUS,
    }


def _decimal(value: Any, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"not a finite decimal: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"not a finite decimal: {value!r}")
    if positive and result <= 0:
        raise ValueError(f"must be positive: {value!r}")
    if nonnegative and result < 0:
        raise ValueError(f"must be nonnegative: {value!r}")
    return result


def _number(value: Decimal) -> str:
    """Publish at most the precision warranted by the declared input passport."""
    with localcontext() as context:
        context.prec = 60
        return format(+_decimal(value), ".12g")


def _payload_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _artifact_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_input(path: Path = INPUT_PATH) -> dict[str, Any]:
    """Load one explicit Planck-2018 comparison passport or fail closed.

    ``--input`` is supported for a copied/reviewed passport, not as a way to
    silently swap a different cosmological analysis under the same schema.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "planck2018-dark-matter-budget-input.v1":
        raise ValueError("unrecognized Planck DM-budget input schema")
    source = payload.get("source")
    params = payload.get("parameters")
    reference_model = payload.get("reference_model")
    if not isinstance(source, dict):
        raise ValueError("Planck input misses source passport")
    if source.get("source_url") != PLANCK_SOURCE_URL:
        raise ValueError("Planck input must cite the declared Planck-2018 primary source URL")
    if not isinstance(source.get("citation"), str) or "Planck 2018" not in source["citation"]:
        raise ValueError("Planck input has an unexpected citation")
    if not isinstance(source.get("version"), str) or not source["version"].startswith("arXiv v4"):
        raise ValueError("Planck input must declare the reviewed arXiv v4 record")
    if not isinstance(reference_model, str) or not reference_model.startswith(PLANCK_REFERENCE_PREFIX):
        raise ValueError("Planck input has an unexpected reference-model scope")
    if not isinstance(params, dict):
        raise ValueError("Planck input misses parameters")
    for key, (units, definition) in PLANCK_PARAMETER_PASSPORT.items():
        entry = params.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"Planck input misses {key}")
        value = _decimal(entry.get("value"), positive=True)
        uncertainty = _decimal(entry.get("uncertainty_68_percent"), positive=True)
        if entry.get("units") != units or entry.get("definition") != definition:
            raise ValueError(f"Planck input has an unexpected passport for {key}")
        if uncertainty >= value:
            raise ValueError(f"Planck input has an implausibly large uncertainty for {key}")
    omega_b = _decimal(params["omega_b"]["value"])
    omega_c = _decimal(params["omega_c"]["value"])
    h0 = _decimal(params["H0_km_s_Mpc"]["value"])
    if not (Decimal("0.01") < omega_b < Decimal("0.04") and Decimal("0.05") < omega_c < Decimal("0.2") and Decimal("40") < h0 < Decimal("100")):
        raise ValueError("Planck input values are outside the declared physical comparison envelope")
    return payload


def _conditional_dark_neutron_mass_GeV() -> Decimal:
    low = _decimal(cogenesis.M_N, positive=True) - _decimal(cogenesis.S_N_BE9, positive=True)
    high = _decimal(cogenesis.M_P, positive=True) + _decimal(cogenesis.M_E, positive=True)
    if high <= low:
        raise ValueError("cogenesis mass corridor is invalid")
    return (low + high) / Decimal("2000")


def _relic_calibration_state() -> tuple[dict[str, float | str], Decimal, Decimal, Decimal]:
    """Consume producer-returned calibration values and verify their passport."""
    state = relic.run_dm_verification()
    if state.get("calibration_status") != relic.CALIBRATION_STATUS:
        raise ValueError("relic producer lost its explicit calibration-only status")
    artifact = state.get("calibration_input_artifact")
    if artifact != relic.CALIBRATION_PATH.relative_to(relic.ROOT).as_posix():
        raise ValueError("relic producer returned an unexpected calibration artifact")
    observed_hash = hashlib.sha256(relic.CALIBRATION_PATH.read_bytes()).hexdigest()
    if state.get("calibration_input_sha256") != observed_hash:
        raise ValueError("relic producer calibration passport checksum does not match live bytes")
    h = _decimal(state.get("calibration_H0_km_s_Mpc"), positive=True) / Decimal("100")
    target_omega = _decimal(state.get("calibration_target_Omega_DM"), positive=True)
    returned_omega_h2 = _decimal(state.get("calibration_target_omega_h2"), positive=True)
    if _decimal(state.get("Omega_DM"), positive=True) != target_omega:
        raise ValueError("relic producer's historical Omega return disagrees with its calibration passport")
    if abs(returned_omega_h2 - target_omega * h * h) > TOLERANCE:
        raise ValueError("relic producer's target omega_h2 disagrees with its own calibration values")
    return state, h, target_omega, returned_omega_h2


def _has_absolute_pbh_omega_contract(state: Mapping[str, Any]) -> bool:
    """Require a typed, sourced abundance contract; key names alone never count."""
    required = {
        "physical_omega_h2",
        "physical_omega_h2_units",
        "formation_solver",
        "abundance_source_provenance",
    }
    if not required.issubset(state):
        return False
    try:
        omega = _decimal(state["physical_omega_h2"], nonnegative=True)
    except ValueError:
        return False
    provenance = state["abundance_source_provenance"]
    return (
        omega >= 0
        and state["physical_omega_h2_units"] == "dimensionless"
        and isinstance(state["formation_solver"], str)
        and bool(state["formation_solver"])
        and isinstance(provenance, Mapping)
        and isinstance(provenance.get("source_url"), str)
        and provenance["source_url"].startswith("https://")
    )


def _build_result(input_path: Path = INPUT_PATH) -> dict[str, Any]:
    with localcontext() as context:
        context.prec = 70
        data = _load_input(input_path)
        params = data["parameters"]
        omega_b = _decimal(params["omega_b"]["value"], positive=True)
        omega_c = _decimal(params["omega_c"]["value"], positive=True)
        h_planck = _decimal(params["H0_km_s_Mpc"]["value"], positive=True) / Decimal("100")
        omega_b_uncertainty = _decimal(params["omega_b"]["uncertainty_68_percent"], positive=True)
        omega_c_uncertainty = _decimal(params["omega_c"]["uncertainty_68_percent"], positive=True)
        h0_uncertainty = _decimal(params["H0_km_s_Mpc"]["uncertainty_68_percent"], positive=True)
        omega_b_fraction = omega_b / (h_planck * h_planck)
        omega_c_fraction = omega_c / (h_planck * h_planck)

        relic_state, relic_h, relic_target, relic_omega_target = _relic_calibration_state()
        relic_planck_ratio = relic_omega_target / omega_c

        pbh_state = pbh.compute_spectrum()
        pbh_shape_sum = _decimal(pbh_state["fraction_sum"], positive=True)
        if pbh_state.get("evidence_status") != "CALIBRATED_GRID_NO_LIKELIHOOD":
            raise ValueError("PBH-A producer changed its declared evidence status; review required")
        if pbh_state.get("observed_likelihood") is not None:
            raise ValueError("PBH-A producer unexpectedly acquired a likelihood; review required")
        pbh_a_absolute_contract = _has_absolute_pbh_omega_contract(pbh_state)

        pbh_b_state = pbh_two.compute_population_b_calibrated_trace()
        if pbh_b_state.get("status") != pbh_two.STATUS:
            raise ValueError("PBH-B producer changed its calibration-only status; review required")
        if pbh_b_state.get("calibration_status") != pbh_two.CALIBRATION_STATUS:
            raise ValueError("PBH-B producer lost its explicit legacy-calibration status")
        if pbh_b_state.get("calibration_input_artifact") != pbh_two.CALIBRATION_PATH.relative_to(pbh_two.ROOT).as_posix():
            raise ValueError("PBH-B producer returned an unexpected calibration artifact")
        if pbh_b_state.get("calibration_input_sha256") != hashlib.sha256(pbh_two.CALIBRATION_PATH.read_bytes()).hexdigest():
            raise ValueError("PBH-B producer calibration passport checksum does not match live bytes")
        if pbh_b_state.get("physical_omega_h2") is not None or pbh_b_state.get("physical_fraction_of_dark_matter") is not None:
            raise ValueError("PBH-B calibration must not masquerade as a physical abundance")
        pbh_b_absolute_contract = _has_absolute_pbh_omega_contract(pbh_b_state)

        baryogenesis_state = baryogenesis_closure.compute_channels()
        if baryogenesis_state.get("evidence_status") != "RETIRED_MISSING_BARYOGENESIS_SOURCE":
            raise ValueError("baryogenesis status changed; review cogenesis eligibility")
        cogenesis_input = _historical_cogenesis_input_passport()
        dark_neutron_mass = _conditional_dark_neutron_mass_GeV()
        proton_mass = _decimal(cogenesis.M_P, positive=True) / Decimal("1000")
        conditional_chi_omega = dark_neutron_mass / proton_mass * omega_b
        conditional_chi_fraction = conditional_chi_omega / omega_c
        diagnostic_gap = omega_c - conditional_chi_omega
        if diagnostic_gap <= 0:
            raise ValueError("conditional dark-neutron identity exceeds Planck CDM reference")

        composition: None = None
        closure_requirements = {
            "relic_independently_predicts_omega": False,
            "pbh_population_A_has_absolute_abundance_contract": pbh_a_absolute_contract,
            "pbh_population_B_has_absolute_abundance_contract": pbh_b_absolute_contract,
            "cogenesis_is_a_viable_source_complete_abundance_mechanism": False,
            "all_components_share_one_predeclared_cosmology": False,
        }
        closure_established = all(closure_requirements.values())

        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS_NOT_CLOSED,
            "unified_abundance_status": CLOSURE_STATUS,
            "independent_evidence_weight": 0.0,
            "fit_used": False,
            "input_artifact": _artifact_label(input_path),
            "input_artifact_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "source_provenance": data["source"],
            "reference_scope": data["reference_model"],
            "planck_reference": {
                "h": _number(h_planck),
                "H0_km_s_Mpc": _number(h_planck * Decimal("100")),
                "H0_uncertainty_68_percent_km_s_Mpc": _number(h0_uncertainty),
                "omega_b": _number(omega_b),
                "omega_b_uncertainty_68_percent": _number(omega_b_uncertainty),
                "omega_c": _number(omega_c),
                "omega_c_uncertainty_68_percent": _number(omega_c_uncertainty),
                "Omega_b_derived_for_comparison_only": _number(omega_b_fraction),
                "Omega_c_derived_for_comparison_only": _number(omega_c_fraction),
                "derived_Omega_c_caveat": "This is arithmetic from correlated base-LambdaCDM inputs; no uncertainty/covariance propagation or NVG CMB fit is claimed.",
            },
            "component_audit": {
                "relic_defect_inversion": {
                    "calibration_status": relic_state["calibration_status"],
                    "calibration_input_artifact": relic_state["calibration_input_artifact"],
                    "calibration_input_sha256": relic_state["calibration_input_sha256"],
                    "declared_target_Omega_at_legacy_h": _number(relic_target),
                    "legacy_h": _number(relic_h),
                    "implied_target_omega": _number(relic_omega_target),
                    "Planck_omega_c_ratio": _number(relic_planck_ratio),
                    "signed_difference_from_Planck_omega_c": _number(relic_omega_target - omega_c),
                    "relative_difference_from_Planck_omega_c": _number(relic_planck_ratio - Decimal(1)),
                    "lambda_v_inferred_after_inputting_target": _number(_decimal(relic_state["required_lambda_v"], positive=True)),
                    "m_W_MeV_inferred_after_inputting_target": _number(_decimal(relic_state["m_W"], positive=True)),
                    "status": "LEGACY_CALIBRATION_NOT_AN_INDEPENDENT_ABUNDANCE_PREDICTION",
                },
                "pbh_population_A": {
                    "normalized_profile_sum": _number(pbh_shape_sum),
                    "physical_omega_h2": None,
                    "status": "UNIDENTIFIABLE_NO_ABSOLUTE_ABUNDANCE_NORMALIZATION",
                    "producer_evidence_status": pbh_state["evidence_status"],
                    "limitation": pbh_state["limitation"],
                },
                "pbh_population_B": {
                    "physical_omega_h2": None,
                    "status": pbh_b_state["status"],
                    "calibration_status": pbh_b_state["calibration_status"],
                    "calibration_input_artifact": pbh_b_state["calibration_input_artifact"],
                    "calibration_input_sha256": pbh_b_state["calibration_input_sha256"],
                    "canonical_ladder_nearest_cycle": pbh_b_state["canonical_ladder_nearest_cycle"],
                    "conditional_seed_fraction_range": [
                        _number(_decimal(row["conditional_fraction_relative_to_legacy_rho_dm"], nonnegative=True))
                        for row in pbh_b_state["rows"]
                    ],
                    "limitation": pbh_b_state["limitation"],
                },
                "dark_neutron_chi": {
                    "historical_input_artifact": cogenesis_input["input_artifact"],
                    "historical_input_sha256": cogenesis_input["input_sha256"],
                    "historical_input_status": cogenesis_input["input_status"],
                    "conditional_mass_GeV": _number(dark_neutron_mass),
                    "conditional_identity": "omega_chi=(m_chi/m_p)*omega_b, conditional on N_chi=N_B",
                    "conditional_omega_chi": _number(conditional_chi_omega),
                    "conditional_fraction_of_Planck_omega_c": _number(conditional_chi_fraction),
                    "unassigned_reference_gap_for_diagnostic_only": _number(diagnostic_gap),
                    "status": "CONDITIONAL_ON_RETIRED_COGENESIS_EXTENSION",
                    "baryogenesis_evidence_status": baryogenesis_state["evidence_status"],
                    "missing_components": baryogenesis_state["missing_components"],
                },
            },
            "manual_remainder_allocation": {
                "status": "REJECTED_NOT_DERIVED",
                "reason": "A calibrated relic target, an unnormalized PBH shape and a conditional chi identity cannot be repaired by assigning an observed-density remainder to another sector.",
            },
            "composition": composition,
            "closure_requirements": closure_requirements,
            "why_no_sum_is_reported": [
                "The relic module explicitly inverts a legacy, undocumented calibration target; it does not predict omega.",
                "PBH population A normalizes a shape only, while PBH population B is a seed/reference-density calibration with no shared physical omega_PBH.",
                "The dark-neutron fraction is conditional on a cogenesis extension whose baryogenesis parent is explicitly retired for missing source dynamics.",
            ],
            "next_non_ad_hoc_calculation": [
                "Derive an absolute PBH/defect production yield or formation probability from a predeclared model, rather than normalize a profile or invert omega_c.",
                "Provide a viable, source-complete baryogenesis/cogenesis mechanism before using N_chi=N_B as a component abundance.",
                "Choose one background cosmology and jointly calculate CMB/BBN/structure observables before comparing its omega_i values with Planck.",
                "Only then test sum_i omega_i=omega_c; no component may be filled in by a remainder allocation.",
            ],
            "controls": {
                "Planck_physical_density_identity_is_arithmetic_only": {
                    "pass": abs(omega_c_fraction * h_planck * h_planck - omega_c) < TOLERANCE,
                    "meaning": "Same-input arithmetic identity, not a physical precision or independent cosmological test.",
                },
                "relic_producer_returns_its_own_calibration_passport": {
                    "pass": relic_state["calibration_status"] == relic.CALIBRATION_STATUS,
                },
                "relic_is_not_promoted_to_a_prediction_even_if_its_target_changes": {"pass": True},
                "PBH_A_shape_is_not_mistaken_for_absolute_abundance": {
                    "pass": Decimal("0.999999") < pbh_shape_sum < Decimal("1.000001") and not pbh_a_absolute_contract,
                },
                "PBH_B_calibration_is_not_mistaken_for_absolute_abundance": {
                    "pass": not pbh_b_absolute_contract and pbh_b_state["physical_omega_h2"] is None,
                },
                "cogenesis_parent_is_retired": {"pass": baryogenesis_state["evidence_status"] == "RETIRED_MISSING_BARYOGENESIS_SOURCE"},
                "cogenesis_historical_input_passport_is_explicit_and_conditional": {
                    "pass": cogenesis_input["input_status"] == DARK_NEUTRON_INPUT_STATUS,
                },
                "conditional_chi_identity_is_arithmetic_only": {
                    "pass": conditional_chi_omega > 0 and conditional_chi_fraction > 0 and diagnostic_gap > 0,
                },
                "manual_composition_is_not_emitted": {"pass": not closure_established and composition is None},
            },
            "source_sha256": {
                "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "relic_module": hashlib.sha256(Path(relic.__file__).read_bytes()).hexdigest(),
                "relic_calibration_input": hashlib.sha256(relic.CALIBRATION_PATH.read_bytes()).hexdigest(),
                "pbh_population_A_module": hashlib.sha256(Path(pbh.__file__).read_bytes()).hexdigest(),
                "pbh_population_B_module": hashlib.sha256(Path(pbh_two.__file__).read_bytes()).hexdigest(),
                "pbh_population_B_calibration_input": hashlib.sha256(pbh_two.CALIBRATION_PATH.read_bytes()).hexdigest(),
                "cogenesis_module": hashlib.sha256(Path(cogenesis.__file__).read_bytes()).hexdigest(),
                "cogenesis_historical_input": cogenesis_input["input_sha256"],
                "baryogenesis_closure_module": hashlib.sha256(Path(baryogenesis_closure.__file__).read_bytes()).hexdigest(),
            },
        }
        result["audit_controls_pass"] = all(bool(control["pass"]) for control in result["controls"].values())
        if not result["audit_controls_pass"] or closure_established:
            raise AssertionError("fail-closed abundance audit control violation")
        result["integrity_sha256"] = hashlib.sha256(_payload_bytes(result)).hexdigest()
        return result


def validate_result(result: Any, input_path: Path = INPUT_PATH) -> bool:
    try:
        return isinstance(result, Mapping) and dict(result) == _build_result(input_path)
    except (ArithmeticError, InvalidOperation, KeyError, OSError, TypeError, ValueError, AssertionError, json.JSONDecodeError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=HERE / "nvg_dm_budget_audit_results.json")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.validate:
            valid = validate_result(json.loads(args.validate.read_text(encoding="utf-8")), args.input)
            print("PASS" if valid else "FAIL")
            return 0 if valid else 1
        result = _build_result(args.input)
    except (ArithmeticError, InvalidOperation, KeyError, OSError, TypeError, ValueError, AssertionError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{result['status']} (audit controls: PASS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
