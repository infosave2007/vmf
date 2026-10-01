#!/usr/bin/env python3
"""Fail-closed minimal-vacuum tree-cut audit of the bound source action.

The bound local-U(1) action fixes the vacuum masses and a small set of
unitary-gauge vertices without adding a detector, material table, damping
model, or empirical target.  This audit asks only whether the leading
*intrinsic vacuum two-body cuts* enabled by those displayed vertices are open
for the live source-complete baseline.

It is deliberately not a lifetime, quality-factor, dark-matter, device, or
all-orders stability calculation.  Spectator fields, renormalized pole shifts,
loops, finite-density/thermal damping, gravity, and open-system losses are
outside the declared minimal sector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import mpmath as mp

try:  # Direct execution from ``verification``.
    from source_complete_solution_audit import PARAMS, Params
except ImportError:  # pragma: no cover - package-style import support.
    from .source_complete_solution_audit import PARAMS, Params


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULT_PATH = HERE / "nvg_source_complete_minimal_vacuum_closure_results.json"
SOURCE_PATH = HERE / "source_complete_solution_audit.py"
ACTION_PATH = HERE / "contracts" / "source_complete_action.md"
SCHEMA = "nvg_source_complete_minimal_vacuum_closure_audit.v1"
STATUS = "PASS_MINIMAL_VACUUM_TREE_TWO_BODY_CUTS_CLOSED"
EVIDENCE_WEIGHT = 0.0


class MinimalVacuumClosureError(ValueError):
    """Raised when the source contract or a canonical result fails closed."""


@dataclass(frozen=True)
class ExactParameters:
    """Exact public-decimal reconstruction of the live source parameters."""

    W0: Fraction
    lambda_baseline: Fraction
    M_N: Fraction
    m_omega: Fraction
    g_omega: Fraction
    hbar_c: Fraction

    @property
    def g_s(self) -> Fraction:
        return self.M_N / self.W0

    @property
    def q_phi(self) -> Fraction:
        return self.m_omega / self.W0

    @property
    def m_A_squared(self) -> Fraction:
        return self.m_omega * self.m_omega

    @property
    def m_sigma_squared(self) -> Fraction:
        return 2 * self.lambda_baseline * self.W0 * self.W0


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:  # pragma: no cover - direct propagation.
        raise MinimalVacuumClosureError(f"cannot read provenance path: {path}") from exc


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise MinimalVacuumClosureError(f"{label} cannot be bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MinimalVacuumClosureError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise MinimalVacuumClosureError(f"{label} is not finite")
    return number


def _live_fraction(value: Any, label: str) -> Fraction:
    number = _finite_float(value, label)
    if number <= 0.0:
        raise MinimalVacuumClosureError(f"{label} must be strictly positive")
    try:
        return Fraction(repr(number))
    except (ValueError, ZeroDivisionError) as exc:  # pragma: no cover
        raise MinimalVacuumClosureError(f"{label} has no usable decimal spelling") from exc


def _mp_fraction(value: Fraction) -> mp.mpf:
    return mp.mpf(value.numerator) / mp.mpf(value.denominator)


def _number(value: Any, digits: int = 32) -> str:
    """Render a deterministic decimal without ambient-mp precision drift."""

    with mp.workdps(max(80, digits + 24)):
        rendered = _mp_fraction(value) if isinstance(value, Fraction) else mp.mpf(value)
        if not mp.isfinite(rendered):
            raise MinimalVacuumClosureError("non-finite scientific output")
        return mp.nstr(rendered, digits)


def _fraction_record(value: Fraction, unit: str, derivation: str) -> dict[str, str]:
    return {
        "decimal": _number(value),
        "derivation": derivation,
        "exact_denominator": str(value.denominator),
        "exact_numerator": str(value.numerator),
        "unit": unit,
    }


def _positive_fraction_record(value: Fraction, unit: str, derivation: str) -> dict[str, Any]:
    if value <= 0:
        raise MinimalVacuumClosureError(f"nonpositive certificate: {derivation}")
    record: dict[str, Any] = _fraction_record(value, unit, derivation)
    record["strictly_positive"] = True
    return record


def exact_parameters_from_live(p: Params = PARAMS) -> ExactParameters:
    """Read only the live action surface and guard its algebraic identities."""

    exact = ExactParameters(
        W0=_live_fraction(p.W0, "W0"),
        lambda_baseline=_live_fraction(p.lam, "lambda"),
        M_N=_live_fraction(p.M_N, "M_N"),
        m_omega=_live_fraction(p.m_omega, "m_omega"),
        g_omega=_live_fraction(p.g_omega, "g_omega"),
        hbar_c=_live_fraction(p.hbar_c, "hbar_c"),
    )
    with mp.workdps(80):
        expected_gs = float(_mp_fraction(exact.g_s))
        expected_qphi = float(_mp_fraction(exact.q_phi))
        expected_m_sigma = float(mp.sqrt(_mp_fraction(exact.m_sigma_squared)))
    for actual, expected, relation in (
        (p.g_s, expected_gs, "g_s=M_N/W0"),
        (p.q_phi, expected_qphi, "q_phi=m_omega/W0"),
        (p.scalar_mass, expected_m_sigma, "m_sigma=sqrt(2*lambda)*W0"),
    ):
        if abs(_finite_float(actual, relation) - expected) / max(1.0, abs(expected)) > 2.0e-14:
            raise MinimalVacuumClosureError(f"live source no longer satisfies {relation}")
    return exact


def _require_live_baseline(p: Params) -> None:
    if p != PARAMS:
        raise MinimalVacuumClosureError(
            "this artifact is defined only for the live source-complete baseline"
        )


def lambda_sigma_to_AA_threshold(exact: ExactParameters) -> Fraction:
    """Return the exact gate m_sigma=2*m_A under a lambda-only deformation."""

    threshold = 2 * exact.q_phi * exact.q_phi
    if threshold <= 0:
        raise MinimalVacuumClosureError("nonpositive sigma-to-AA lambda gate")
    return threshold


def lambda_sigma_to_NN_threshold(exact: ExactParameters) -> Fraction:
    """Return the exact gate m_sigma=2*M_N under a lambda-only deformation."""

    threshold = 2 * exact.g_s * exact.g_s
    if threshold <= 0:
        raise MinimalVacuumClosureError("nonpositive sigma-to-NN lambda gate")
    return threshold


def _provenance() -> dict[str, str]:
    paths = {
        "action": ACTION_PATH,
        "source_complete": SOURCE_PATH,
        "producer": Path(__file__),
    }
    return {f"{label}_path": str(path.relative_to(REPO_ROOT)) for label, path in paths.items()} | {
        f"{label}_sha256": _sha256(path) for label, path in paths.items()
    }


def _action_inputs(exact: ExactParameters) -> dict[str, Any]:
    with mp.workdps(100):
        m_A = mp.sqrt(_mp_fraction(exact.m_A_squared))
        m_sigma = mp.sqrt(_mp_fraction(exact.m_sigma_squared))
        return {
            "live_decimal_inputs": {
                "lambda": _fraction_record(exact.lambda_baseline, "1", "live Params.lam"),
                "M_N_MeV": _fraction_record(exact.M_N, "MeV", "live Params.M_N"),
                "W0_MeV": _fraction_record(exact.W0, "MeV", "live Params.W0"),
                "m_omega_MeV": _fraction_record(exact.m_omega, "MeV", "live Params.m_omega"),
                "g_omega": _fraction_record(
                    exact.g_omega, "1", "live Params.g_omega; matter-current coupling"
                ),
                "hbar_c_MeV_fm": _fraction_record(exact.hbar_c, "MeV fm", "live Params.hbar_c"),
            },
            "derived_from_bound_action": {
                "g_s_M_N_over_W0": _fraction_record(exact.g_s, "1", "M_N/W0"),
                "q_phi_momega_over_W0": _fraction_record(exact.q_phi, "1", "m_omega/W0"),
                "m_A_squared_MeV2": _fraction_record(
                    exact.m_A_squared, "MeV^2", "(q_phi*W0)^2=m_omega^2"
                ),
                "m_sigma_squared_MeV2": _fraction_record(
                    exact.m_sigma_squared, "MeV^2", "2*lambda*W0^2"
                ),
                "m_A_MeV_display": _number(m_A),
                "m_sigma_MeV_display": _number(m_sigma),
            },
        }


def _mass_ordering(exact: ExactParameters) -> dict[str, Any]:
    """Certify the displayed minimal-sector order with squared mass gaps."""

    gaps = {
        "M_N_squared_minus_m_A_squared": (
            exact.M_N * exact.M_N - exact.m_A_squared,
            "M_N^2-m_A^2; establishes m_A<M_N",
        ),
        "m_sigma_squared_minus_M_N_squared": (
            exact.m_sigma_squared - exact.M_N * exact.M_N,
            "m_sigma^2-M_N^2; establishes M_N<m_sigma",
        ),
        "four_m_A_squared_minus_m_sigma_squared": (
            4 * exact.m_A_squared - exact.m_sigma_squared,
            "4*m_A^2-m_sigma^2; establishes m_sigma<2*m_A",
        ),
    }
    return {
        "strict_mass_ordering": "m_A < M_N < m_sigma < 2*m_A < 2*M_N",
        "squared_gap_certificates": {
            label: _positive_fraction_record(value, "MeV^2", derivation)
            for label, (value, derivation) in gaps.items()
        },
    }


def _closed_channel(
    *,
    identifier: str,
    parent: str,
    final_state: str,
    closure_gap: Fraction,
    derivation: str,
) -> dict[str, Any]:
    """Record an exact closed two-body cut with no numerical width model."""

    return {
        "identifier": identifier,
        "parent": parent,
        "final_state": final_state,
        "open_threshold": f"m({parent}) > total rest mass({final_state})",
        "closure_gap_squared": _positive_fraction_record(
            closure_gap, "MeV^2", derivation
        ),
        "phase_space_theta": "Theta(m_parent-total_final_rest_mass)=0",
        "minimal_tree_vacuum_width": "0 (kinematically closed; no coupling, line shape, or damping fit evaluated)",
        "status": "KINEMATICALLY_CLOSED",
    }


def _positive_emission_closure(
    *,
    identifier: str,
    parent: str,
    final_state: str,
    emitted_mass_squared: Fraction,
    emitted_label: str,
) -> dict[str, Any]:
    """Record a crossed 1->2 process closed by an exactly positive emitted mass."""

    return {
        "identifier": identifier,
        "parent": parent,
        "final_state": final_state,
        "open_threshold": f"m({parent}) > m({parent})+m({emitted_label})",
        "positive_emitted_mass_squared": _positive_fraction_record(
            emitted_mass_squared,
            "MeV^2",
            f"m_{emitted_label}^2>0; therefore m({parent})<m({parent})+m({emitted_label})",
        ),
        "phase_space_theta": "Theta(m_parent-total_final_rest_mass)=0",
        "minimal_tree_vacuum_width": "0 (kinematically closed; no coupling, line shape, or damping fit evaluated)",
        "status": "KINEMATICALLY_CLOSED",
    }


def _tree_two_body_cuts(exact: ExactParameters) -> list[dict[str, Any]]:
    """Return every 1->2 crossing of the displayed unitary-gauge vertices.

    Charge-conjugate Nbar emission has the same mass certificate as the N
    row and is explicitly represented there rather than duplicated.
    """

    channels = [
        _closed_channel(
            identifier="sigma_to_N_Nbar",
            parent="sigma",
            final_state="N+Nbar",
            closure_gap=4 * exact.M_N * exact.M_N - exact.m_sigma_squared,
            derivation="(2*M_N)^2-m_sigma^2",
        ),
        _closed_channel(
            identifier="A_to_N_Nbar",
            parent="A",
            final_state="N+Nbar",
            closure_gap=4 * exact.M_N * exact.M_N - exact.m_A_squared,
            derivation="(2*M_N)^2-m_A^2",
        ),
        _closed_channel(
            identifier="sigma_to_A_A",
            parent="sigma",
            final_state="A+A",
            closure_gap=4 * exact.m_A_squared - exact.m_sigma_squared,
            derivation="(2*m_A)^2-m_sigma^2",
        ),
        _positive_emission_closure(
            identifier="N_to_N_sigma",
            parent="N",
            final_state="N+sigma (and charge-conjugate Nbar process)",
            emitted_mass_squared=exact.m_sigma_squared,
            emitted_label="sigma",
        ),
        _positive_emission_closure(
            identifier="N_to_N_A",
            parent="N",
            final_state="N+A (and charge-conjugate Nbar process)",
            emitted_mass_squared=exact.m_A_squared,
            emitted_label="A",
        ),
        _positive_emission_closure(
            identifier="A_to_A_sigma",
            parent="A",
            final_state="A+sigma",
            emitted_mass_squared=exact.m_sigma_squared,
            emitted_label="sigma",
        ),
        _positive_emission_closure(
            identifier="sigma_to_sigma_sigma",
            parent="sigma",
            final_state="sigma+sigma",
            emitted_mass_squared=exact.m_sigma_squared,
            emitted_label="sigma",
        ),
    ]
    if not all(row["status"] == "KINEMATICALLY_CLOSED" for row in channels):  # pragma: no cover
        raise MinimalVacuumClosureError("minimal tree cut unexpectedly opened")
    return channels


def _lambda_gates(exact: ExactParameters) -> dict[str, Any]:
    sigma_AA = lambda_sigma_to_AA_threshold(exact)
    sigma_NN = lambda_sigma_to_NN_threshold(exact)
    if not (exact.lambda_baseline < sigma_AA and exact.lambda_baseline < sigma_NN):
        raise MinimalVacuumClosureError("live baseline does not close the declared scalar two-body gates")
    return {
        "lambda_only_hypothetical_deformation": (
            "m_sigma(lambda_d)^2=2*lambda_d*W0^2 with W0, M_N, m_A, g_s, and q_phi held fixed; "
            "this is a diagnostic identity, not a parameter recommendation"
        ),
        "sigma_to_A_A_opens_only_if": "lambda_d > 2*q_phi^2",
        "sigma_to_A_A_closure_threshold": _fraction_record(
            sigma_AA, "1", "2*q_phi^2 from m_sigma=2*m_A"
        ),
        "sigma_to_N_Nbar_opens_only_if": "lambda_d > 2*g_s^2",
        "sigma_to_N_Nbar_closure_threshold": _fraction_record(
            sigma_NN, "1", "2*g_s^2 from m_sigma=2*M_N"
        ),
        "baseline_is_below_both_opening_thresholds": True,
    }


def _fresh_result_payload(p: Params = PARAMS) -> dict[str, Any]:
    _require_live_baseline(p)
    exact = exact_parameters_from_live(p)
    channels = _tree_two_body_cuts(exact)
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "provenance": _provenance(),
        "action_inputs": _action_inputs(exact),
        "minimal_sector_contract": {
            "vacuum": "source-free gauge-neutral vacuum; unitary gauge W=W0+sigma, theta=0, A_mu=0",
            "physical_modes_counted": [
                "N and Nbar with rest mass M_N",
                "one radial scalar sigma with m_sigma^2=2*lambda*W0^2",
                "one massive vector A with three polarizations and m_A=q_phi*W0=m_omega",
            ],
            "displayed_vertices_used": [
                "-g_s*sigma*Nbar*N",
                "-g_omega*A_mu*Nbar*gamma^mu*N",
                "+(m_A^2/W0)*sigma*A_mu*A^mu",
                "-lambda*W0*sigma^3",
            ],
            "not_counted_as_physical_final_particles": [
                "theta is the eaten Goldstone in unitary gauge",
                "A0 is a Gauss constraint, not an additional particle",
            ],
            "excluded": [
                "L_spectator content not specified by the displayed action",
                "new operators or particles",
                "external/open-system damping and drive",
                "loops, renormalized pole shifts, finite-density or finite-temperature damping, and gravity",
            ],
            "parameter_fit_performed": False,
            "empirical_target_or_data_read": False,
            "physical_claim_permitted": False,
        },
        "mass_ordering": _mass_ordering(exact),
        "leading_tree_two_body_cuts": {
            "scope": "only physical two-body cuts enabled by the displayed minimal-sector unitary-gauge vertices",
            "channels": channels,
            "all_listed_cuts_closed": True,
        },
        "hypothetical_lambda_diagnostic_gates": _lambda_gates(exact),
        "conclusion": {
            "minimal_tree_vacuum_only": (
                "All listed intrinsic vacuum two-body cuts in the displayed minimal (N,sigma,A) sector are "
                "kinematically closed at the live baseline. Therefore this action fragment supplies no nonzero "
                "tree-level intrinsic width from those cuts."
            ),
            "not_claimed": [
                "a numerical lifetime, linewidth, quality factor, or stored energy",
                "dark matter, a stable real-world particle, or all-orders stability",
                "a device, energy source, gain, or free-energy mechanism",
                "a statement about unspecified spectators, gravity, external ports, loops, or matter-induced damping",
            ],
        },
        "caveats": [
            "A kinematically closed displayed tree cut is not an all-orders pole-width calculation.",
            "The scalar cubic vertex does not allow sigma->sigma+sigma for positive m_sigma; its threshold is always above its parent mass.",
            "No physical AAA or A-sigma-sigma vertex exists in the displayed Abelian unitary-gauge minimal sector.",
            "This audit does not identify the formal A or sigma mode with an observed particle or experimental resonance.",
            "No empirical mass, decay, material, source, damping, detector, or likelihood table is read.",
        ],
    }


def build_result(p: Params = PARAMS) -> dict[str, Any]:
    result = _fresh_result_payload(p)
    validate_result(result, p=p, check_provenance=False)
    return result


def validate_result(payload: Any, p: Params = PARAMS, *, check_provenance: bool = True) -> None:
    """Accept only an exact fresh derivation for the live source action."""

    if not isinstance(payload, Mapping):
        raise MinimalVacuumClosureError("result payload must be an object")
    expected = _fresh_result_payload(p)
    if _canonical_json(payload) != _canonical_json(expected):
        raise MinimalVacuumClosureError("payload is not an exact fresh minimal-vacuum derivation")
    if check_provenance and payload.get("provenance") != _provenance():
        raise MinimalVacuumClosureError("provenance changed")


def write_result(result: Mapping[str, Any], path: Path = RESULT_PATH) -> None:
    validate_result(result)
    path.write_bytes(_canonical_json(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT_PATH, help="JSON output path")
    parser.add_argument("--validate", type=Path, help="validate an existing result and write no files")
    args = parser.parse_args(argv)
    if args.validate is not None:
        try:
            payload = json.loads(args.validate.read_text(encoding="utf-8"))
            validate_result(payload)
        except (OSError, json.JSONDecodeError, MinimalVacuumClosureError) as exc:
            print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
            return 1
        print("VALIDATION_PASS")
        return 0
    try:
        result = build_result()
        write_result(result, args.output)
    except MinimalVacuumClosureError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 1
    print(_canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
