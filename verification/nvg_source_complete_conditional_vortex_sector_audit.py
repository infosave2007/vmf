#!/usr/bin/env python3
"""Fail-closed conditional classical-vortex sector audit of the bound action.

The displayed source-complete action contains a local Abelian-Higgs subsector.
With matter, spectators, and A0 set to zero, it fixes its mass ratio and the
classical type-I/type-II/BPS algebra without fitting an input.  A globally
quantized vortex, however, additionally needs a compact-U(1) / winding
completion that the bound contract does not declare.  This artifact therefore
records only the conditional local-sector consequence and refuses to turn it
into a physical cosmic-string, superconductor, neutron-star, or energy claim.
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
RESULT_PATH = HERE / "nvg_source_complete_conditional_vortex_sector_results.json"
SOURCE_PATH = HERE / "source_complete_solution_audit.py"
ACTION_PATH = HERE / "contracts" / "source_complete_action.md"
SCHEMA = "nvg_source_complete_conditional_vortex_sector_audit.v1"
STATUS = "CONDITIONAL_LOCAL_TYPE_II_VORTEX_SECTOR_GLOBAL_COMPLETION_UNDECLARED"
EVIDENCE_WEIGHT = 0.0


class ConditionalVortexSectorError(ValueError):
    """Raised when live action relations or a canonical artifact fail closed."""


@dataclass(frozen=True)
class ExactParameters:
    """Exact public-decimal reconstruction of live bound-action inputs."""

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

    @property
    def beta(self) -> Fraction:
        """m_sigma^2/m_A^2 = 2 lambda / q_phi^2."""

        return self.m_sigma_squared / self.m_A_squared

    @property
    def kappa_GL_squared(self) -> Fraction:
        """kappa_GL^2 = m_sigma^2/(2 m_A^2) = lambda/q_phi^2."""

        return self.beta / 2


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:  # pragma: no cover - direct propagation.
        raise ConditionalVortexSectorError(f"cannot read provenance path: {path}") from exc


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ConditionalVortexSectorError(f"{label} cannot be bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConditionalVortexSectorError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ConditionalVortexSectorError(f"{label} is not finite")
    return number


def _live_fraction(value: Any, label: str) -> Fraction:
    number = _finite_float(value, label)
    if number <= 0.0:
        raise ConditionalVortexSectorError(f"{label} must be strictly positive")
    try:
        return Fraction(repr(number))
    except (ValueError, ZeroDivisionError) as exc:  # pragma: no cover
        raise ConditionalVortexSectorError(f"{label} has no usable decimal spelling") from exc


def _mp_fraction(value: Fraction) -> mp.mpf:
    return mp.mpf(value.numerator) / mp.mpf(value.denominator)


def _number(value: Any, digits: int = 32) -> str:
    with mp.workdps(max(80, digits + 24)):
        rendered = _mp_fraction(value) if isinstance(value, Fraction) else mp.mpf(value)
        if not mp.isfinite(rendered):
            raise ConditionalVortexSectorError("non-finite scientific output")
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
        raise ConditionalVortexSectorError(f"nonpositive certificate: {derivation}")
    record: dict[str, Any] = _fraction_record(value, unit, derivation)
    record["strictly_positive"] = True
    return record


def exact_parameters_from_live(p: Params = PARAMS) -> ExactParameters:
    """Read live values and protect the action's Higgs mass identities."""

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
            raise ConditionalVortexSectorError(f"live source no longer satisfies {relation}")
    return exact


def _require_live_baseline(p: Params) -> None:
    if p != PARAMS:
        raise ConditionalVortexSectorError(
            "this artifact is defined only for the live source-complete baseline"
        )


def lambda_BPS(exact: ExactParameters) -> Fraction:
    """Return lambda=q_phi^2/2, where m_sigma=m_A in this convention."""

    threshold = exact.q_phi * exact.q_phi / 2
    if threshold <= 0:
        raise ConditionalVortexSectorError("nonpositive BPS threshold")
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
        return {
            "live_decimal_inputs": {
                "lambda": _fraction_record(exact.lambda_baseline, "1", "live Params.lam"),
                "W0_MeV": _fraction_record(exact.W0, "MeV", "live Params.W0"),
                "m_omega_MeV": _fraction_record(exact.m_omega, "MeV", "live Params.m_omega"),
                "g_omega": _fraction_record(
                    exact.g_omega, "1", "live Params.g_omega; distinct matter-current coupling"
                ),
            },
            "derived_from_bound_action": {
                "q_phi_momega_over_W0": _fraction_record(exact.q_phi, "1", "m_omega/W0"),
                "m_A_squared_MeV2": _fraction_record(
                    exact.m_A_squared, "MeV^2", "q_phi^2*W0^2=m_omega^2"
                ),
                "m_sigma_squared_MeV2": _fraction_record(
                    exact.m_sigma_squared, "MeV^2", "2*lambda*W0^2"
                ),
                "m_A_MeV_display": _number(mp.sqrt(_mp_fraction(exact.m_A_squared))),
                "m_sigma_MeV_display": _number(mp.sqrt(_mp_fraction(exact.m_sigma_squared))),
            },
        }


def _classification(exact: ExactParameters) -> dict[str, Any]:
    bps = lambda_BPS(exact)
    beta_minus_one = exact.beta - 1
    kappa_sq_minus_half = exact.kappa_GL_squared - Fraction(1, 2)
    lambda_excess = exact.lambda_baseline - bps
    if not (beta_minus_one > 0 and kappa_sq_minus_half > 0 and lambda_excess > 0):
        raise ConditionalVortexSectorError("live baseline is not in the declared local type-II regime")
    with mp.workdps(100):
        beta = _mp_fraction(exact.beta)
        kappa = mp.sqrt(_mp_fraction(exact.kappa_GL_squared))
        mass_ratio = mp.sqrt(beta)
    return {
        "convention": {
            "beta": "m_sigma^2/m_A^2=2*lambda/q_phi^2",
            "kappa_GL": "m_sigma/(sqrt(2)*m_A)=sqrt(lambda/q_phi^2)",
            "BPS_condition": "lambda=q_phi^2/2, equivalently beta=1 and kappa_GL=1/sqrt(2)",
        },
        "lambda_BPS": _fraction_record(
            bps, "1", "q_phi^2/2=m_omega^2/(2*W0^2), equivalently m_sigma=m_A"
        ),
        "BPS_equals_scalar_vector_mass_equality_threshold": True,
        "baseline_minus_lambda_BPS": _positive_fraction_record(
            lambda_excess, "1", "lambda_baseline-q_phi^2/2"
        ),
        "beta": _fraction_record(exact.beta, "1", "m_sigma^2/m_A^2=2*lambda/q_phi^2"),
        "beta_minus_one": _positive_fraction_record(beta_minus_one, "1", "beta-1"),
        "kappa_GL_squared": _fraction_record(
            exact.kappa_GL_squared, "1", "lambda/q_phi^2"
        ),
        "kappa_GL_squared_minus_one_half": _positive_fraction_record(
            kappa_sq_minus_half, "1", "kappa_GL^2-1/2"
        ),
        "mass_ratio_m_sigma_over_m_A_display": _number(mass_ratio),
        "kappa_GL_display": _number(kappa),
        "local_classification": "TYPE_II_NOT_BPS",
    }


def _conditional_boundary_problem(exact: ExactParameters) -> dict[str, Any]:
    """Record equations, but intentionally do not publish an unqualified solution."""

    with mp.workdps(100):
        flux_unit = 2 * mp.pi / _mp_fraction(exact.q_phi)
        bps_tension_unit = mp.pi * _mp_fraction(exact.W0) ** 2
    return {
        "required_extra_global_assumptions": [
            "compact U(1) global completion with allowed integer winding n",
            "finite-energy boundary condition on a flat transverse R^2",
            "source-free truncation N=J_B=A0=0 and unspecified spectators absent",
        ],
        "not_declared_by_bound_contract": [
            "compactness/charge lattice and large-gauge transformation sector",
            "UV completion and quantum renormalization",
            "gravitational sector or physical string interpretation",
        ],
        "conditional_ansatz": "W=W0*f(r), theta=n*varphi, A_phi=n*a(r)/q_phi; f(0)=a(0)=0, f(infinity)=a(infinity)=1",
        "dimensionless_coordinate": "x=m_A*r",
        "conditional_BVP": [
            "f''+f'/x-n^2*(1-a)^2*f/x^2-(beta/2)*f*(f^2-1)=0",
            "a''-a'/x+f^2*(1-a)=0",
        ],
        "conditional_flux_quantum_for_integer_n": "Phi_B=2*pi*n/q_phi",
        "conditional_flux_unit_display": _number(flux_unit),
        "BPS_tension_only_at_lambda_BPS": "T_n=pi*W0^2*abs(n)",
        "BPS_tension_unit_MeV2_display_not_baseline_result": _number(bps_tension_unit),
        "baseline_BPS_equations_or_tension_claimed": False,
        "numerical_vortex_solution_published_by_this_audit": False,
        "global_completion_status": "UNDECLARED_BY_BOUND_CONTRACT",
    }


def _fresh_result_payload(p: Params = PARAMS) -> dict[str, Any]:
    _require_live_baseline(p)
    exact = exact_parameters_from_live(p)
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "provenance": _provenance(),
        "action_inputs": _action_inputs(exact),
        "source_free_local_truncation": {
            "action": (
                "S0=integral[-F^2/4+(partial W)^2/2+W^2*(partial theta-q_phi*A)^2/2"
                "-lambda*(W^2-W0^2)^2/4]"
            ),
            "fixed_conditions": ["N=J_B=0", "A0=0", "all unspecified spectators absent"],
            "no_new_operator_or_particle_introduced": True,
            "parameter_fit_performed": False,
            "empirical_target_or_data_read": False,
            "physical_claim_permitted": False,
        },
        "local_abelian_higgs_classification": _classification(exact),
        "conditional_vortex_boundary_problem": _conditional_boundary_problem(exact),
        "conclusion": {
            "local_math_only": (
                "The displayed source-free local action has a type-II, non-BPS Abelian-Higgs parameter ratio at "
                "the live baseline. A classical winding-vortex BVP is available only conditional on the listed "
                "global completion and boundary assumptions."
            ),
            "not_claimed": [
                "an unconditionally topologically protected quantum sector",
                "a solved baseline vortex profile, tension, flux tube, or physical string",
                "a superconductor, Abrikosov lattice, neutron-star vortex, or cosmic-string population",
                "a generator, energy source, dark-matter candidate, or experimental prediction",
            ],
        },
        "caveats": [
            "The bound contract specifies a local U(1) action but does not itself declare compactness, a charge lattice, or large gauge transformations.",
            "The homogeneous finite-density W^-2 no-go is not applied to the source-free A0=J_B=0 vortex truncation.",
            "Conversely, the source-free conditional BVP says nothing about a vortex in baryonic matter; that needs full inhomogeneous matter equations.",
            "q_phi=m_omega/W0 fixes the Higgs vector mass; g_omega is a different matter-current coupling and is not substituted into beta or the BPS gate.",
            "No numerical BVP solution is generated here, and no empirical data, material parameter, external field, or fitted input is read.",
        ],
    }


def build_result(p: Params = PARAMS) -> dict[str, Any]:
    result = _fresh_result_payload(p)
    validate_result(result, p=p, check_provenance=False)
    return result


def validate_result(payload: Any, p: Params = PARAMS, *, check_provenance: bool = True) -> None:
    if not isinstance(payload, Mapping):
        raise ConditionalVortexSectorError("result payload must be an object")
    expected = _fresh_result_payload(p)
    if _canonical_json(payload) != _canonical_json(expected):
        raise ConditionalVortexSectorError("payload is not an exact fresh conditional-vortex derivation")
    if check_provenance and payload.get("provenance") != _provenance():
        raise ConditionalVortexSectorError("provenance changed")


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
        except (OSError, json.JSONDecodeError, ConditionalVortexSectorError) as exc:
            print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
            return 1
        print("VALIDATION_PASS")
        return 0
    try:
        result = build_result()
        write_result(result, args.output)
    except ConditionalVortexSectorError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 1
    print(_canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
