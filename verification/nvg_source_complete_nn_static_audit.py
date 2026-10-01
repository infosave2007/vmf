#!/usr/bin/env python3
"""No-fit static two-baryon exchange audit of the bound source-complete action.

The calculation deliberately stops at the part that the adjudicated action
fixes without an added nuclear model: the vacuum quadratic spectrum and the
leading, static, nonrelativistic one-boson-exchange kernel between two
same-sign ``N`` currents.  It imports the live source-complete parameters; it
contains no NN data, form factor, cutoff, contact operator, pion/rho sector,
or fitted quantity.

A positive static kernel is an exact statement *within this tree/static
reduction*.  It is not promoted to an all-orders NN, QCD, deuteron, or
phenomenological-nuclear conclusion.  In particular, the vector loop-counting
parameter is deliberately reported because it is not small enough to call the
tree result a precision quantum prediction.
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
except ImportError:  # pragma: no cover - package-style execution support.
    from .source_complete_solution_audit import PARAMS, Params


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULT_PATH = HERE / "nvg_source_complete_nn_static_results.json"
SOURCE_PATH = HERE / "source_complete_solution_audit.py"
ACTION_PATH = HERE / "contracts" / "source_complete_action.md"
SCHEMA = "nvg_source_complete_nn_static_audit.v1"
STATUS = "PASS_SOURCE_COMPLETE_STATIC_EXCHANGE_REPULSION_CERTIFICATE"
EVIDENCE_WEIGHT = 0.0
CLAIM_SCOPE = "ANALYTIC_TREE_STATIC_NONRELATIVISTIC_ONLY"


class StaticExchangeAuditError(ValueError):
    """Raised when a source/action or serialized certificate is invalid."""


@dataclass(frozen=True)
class ExactParameters:
    """Exact-decimal reconstruction of the live source parameter surface.

    ``Params`` stores the declared numbers as Python floats.  We read those
    live values, convert their public decimal spellings to rationals, and only
    then form algebraic identities.  This avoids falsely treating the binary
    representation of 1.05 or 10.12 as a physics input.
    """

    W0: Fraction
    lam: Fraction
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
        return 2 * self.lam * self.W0 * self.W0

    @property
    def g_omega_squared(self) -> Fraction:
        return self.g_omega * self.g_omega

    @property
    def g_s_squared(self) -> Fraction:
        return self.g_s * self.g_s


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:  # pragma: no cover - direct error propagation.
        raise StaticExchangeAuditError(f"cannot read {path}") from exc


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise StaticExchangeAuditError(f"{label} must be a finite real number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StaticExchangeAuditError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise StaticExchangeAuditError(f"{label} is not finite")
    return number


def _live_fraction(value: Any, label: str, *, positive: bool = True) -> Fraction:
    """Convert a finite live parameter to the exact public decimal rational."""
    number = _finite_float(value, label)
    if positive and number <= 0.0:
        raise StaticExchangeAuditError(f"{label} must be positive")
    # ``repr`` is the public round-trippable spelling of the live float.  The
    # base source uses finite decimal model inputs, so this is a transparent
    # audit encoding rather than a copied parameter table.
    try:
        return Fraction(repr(number))
    except (ValueError, ZeroDivisionError) as exc:  # pragma: no cover
        raise StaticExchangeAuditError(f"{label} has no finite decimal representation") from exc


def exact_parameters_from_live(p: Params = PARAMS) -> ExactParameters:
    """Read all source inputs live and construct their algebraic derivatives."""
    exact = ExactParameters(
        W0=_live_fraction(p.W0, "W0"),
        lam=_live_fraction(p.lam, "lambda"),
        M_N=_live_fraction(p.M_N, "M_N"),
        m_omega=_live_fraction(p.m_omega, "m_omega"),
        g_omega=_live_fraction(p.g_omega, "g_omega"),
        hbar_c=_live_fraction(p.hbar_c, "hbar_c"),
    )
    # Confirm that the source's live derived properties retain their declared
    # action definitions.  These are guards, not extra parameter inputs.
    # These conversions are part of the source-contract guard too.  Make
    # them independent of a caller that has deliberately lowered mp.dps.
    with mp.workdps(80):
        expected_gs = float(_mp_fraction(exact.g_s))
        expected_q = float(_mp_fraction(exact.q_phi))
        expected_ms = float(mp.sqrt(_mp_fraction(exact.m_sigma_squared)))
    checks = (
        (p.g_s, expected_gs, "g_s=M_N/W0"),
        (p.q_phi, expected_q, "q_phi=m_omega/W0"),
        (p.scalar_mass, expected_ms, "m_sigma=sqrt(2*lambda)*W0"),
    )
    for actual, expected, name in checks:
        scale = max(1.0, abs(float(expected)))
        if abs(_finite_float(actual, name) - float(expected)) / scale > 2.0e-14:
            raise StaticExchangeAuditError(f"live source no longer satisfies {name}")
    return exact


def _mp_fraction(value: Fraction) -> mp.mpf:
    return mp.mpf(value.numerator) / mp.mpf(value.denominator)


def _number(value: Any, digits: int = 30) -> str:
    # Rendering itself is part of the certificate.  Do not let a caller's
    # ambient mpmath precision turn an exact Fraction such as 253/25 into a
    # binary-looking decimal during validation.
    with mp.workdps(max(80, digits + 16)):
        if isinstance(value, Fraction):
            value = _mp_fraction(value)
        else:
            value = mp.mpf(value)
        if not mp.isfinite(value):
            raise StaticExchangeAuditError("nonfinite scientific output")
        return mp.nstr(value, digits)


def _fraction_record(value: Fraction, unit: str, derivation: str) -> dict[str, str]:
    return {
        "decimal": _number(value),
        "derivation": derivation,
        "exact_denominator": str(value.denominator),
        "exact_numerator": str(value.numerator),
        "unit": unit,
    }


def _positive_fraction_record(value: Fraction, unit: str, derivation: str) -> dict[str, Any]:
    record: dict[str, Any] = _fraction_record(value, unit, derivation)
    record["strictly_positive"] = bool(value > 0)
    return record


def _provenance() -> dict[str, str]:
    return {
        "action_path": str(ACTION_PATH.relative_to(REPO_ROOT)),
        "action_sha256": _sha256(ACTION_PATH),
        "producer_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "producer_sha256": _sha256(Path(__file__)),
        "source_complete_path": str(SOURCE_PATH.relative_to(REPO_ROOT)),
        "source_complete_sha256": _sha256(SOURCE_PATH),
    }


def _action_input_records(exact: ExactParameters) -> dict[str, Any]:
    """Serialize live inputs and only action-derived quantities."""
    with mp.workdps(80):
        m_sigma = mp.sqrt(_mp_fraction(exact.m_sigma_squared))
        m_A = mp.sqrt(_mp_fraction(exact.m_A_squared))
        return {
            "base_live_decimal_inputs": {
                "M_N_MeV": _fraction_record(exact.M_N, "MeV", "live Params.M_N"),
                "W0_MeV": _fraction_record(exact.W0, "MeV", "live Params.W0"),
                "g_omega": _fraction_record(exact.g_omega, "1", "live Params.g_omega; nucleon current coupling"),
                "hbar_c_MeV_fm": _fraction_record(exact.hbar_c, "MeV fm", "live Params.hbar_c"),
                "lambda": _fraction_record(exact.lam, "1", "live Params.lam"),
                "m_omega_MeV": _fraction_record(exact.m_omega, "MeV", "live Params.m_omega"),
            },
            "derived_from_bound_action": {
                "g_s_M_N_over_W0": _fraction_record(exact.g_s, "1", "M_N/W0"),
                "m_A_MeV": _number(m_A),
                "m_A_squared_MeV2": _fraction_record(exact.m_A_squared, "MeV^2", "(q_phi W0)^2=m_omega^2"),
                "m_sigma_MeV": _number(m_sigma),
                "m_sigma_squared_MeV2": _fraction_record(exact.m_sigma_squared, "MeV^2", "2*lambda*W0^2"),
                "q_phi_momega_over_W0": _fraction_record(exact.q_phi, "1", "m_omega/W0"),
            },
            "coupling_assignment": {
                "scalar_exchange_nucleon_vertex": "g_s=M_N/W0",
                "vector_exchange_nucleon_vertex": "g_omega (not q_phi)",
                "why": "The bound action and live source use D_N=partial+i*g_omega*A; q_phi only fixes the Higgs-generated vector mass.",
            },
        }


def _certificates(exact: ExactParameters) -> dict[str, Any]:
    """Create exact-rational positivity certificates for q^2 and r domains."""
    gs2 = exact.g_s_squared
    gw2 = exact.g_omega_squared
    mA2 = exact.m_A_squared
    ms2 = exact.m_sigma_squared
    q2_coefficient = gw2 - gs2
    constant_coefficient = gw2 * ms2 - gs2 * mA2
    contact_vector = gw2 / mA2
    contact_scalar = gs2 / ms2
    contact_difference = contact_vector - contact_scalar
    coupling_ratio = gw2 / gs2
    mass_squared_difference = ms2 - mA2

    if not all(value > 0 for value in (
        gs2, gw2, mA2, ms2, q2_coefficient, constant_coefficient,
        contact_vector, contact_scalar, contact_difference,
        coupling_ratio - 1, mass_squared_difference,
    )):
        raise StaticExchangeAuditError("source parameters do not satisfy the static positivity inequalities")

    return {
        "domain": {
            "coordinate": "r>0, two equal-sign N currents, hbar=c=1 before the fm conversion",
            "momentum": "Q2=|q|^2>=0 for the static spatial momentum transfer",
        },
        "kernel_definitions": {
            "coordinate_natural": "V(r)=[g_omega^2 exp(-m_A r)-g_s^2 exp(-m_sigma r)]/(4*pi*r)",
            "coordinate_fm": "V(r_fm)=hbar_c*[g_omega^2 exp(-m_A*r_fm/hbar_c)-g_s^2 exp(-m_sigma*r_fm/hbar_c)]/(4*pi*r_fm)",
            "momentum": "V_tilde(Q2)=g_omega^2/(Q2+m_A^2)-g_s^2/(Q2+m_sigma^2)",
            "momentum_factored": "V_tilde(Q2)=[(g_omega^2-g_s^2)Q2+(g_omega^2*m_sigma^2-g_s^2*m_A^2)]/[(Q2+m_A^2)(Q2+m_sigma^2)]",
            "coordinate_factored": "V(r)=g_s^2 exp(-m_sigma r)*[(g_omega^2/g_s^2) exp((m_sigma-m_A)r)-1]/(4*pi*r)",
        },
        "exact_rational_inequalities": {
            "g_omega_squared_minus_g_s_squared": _positive_fraction_record(
                q2_coefficient, "1", "coefficient of Q2 in the factored momentum numerator"
            ),
            "g_omega_squared_m_sigma_squared_minus_g_s_squared_m_A_squared": _positive_fraction_record(
                constant_coefficient, "MeV^2", "constant term in the factored momentum numerator"
            ),
            "m_sigma_squared_minus_m_A_squared": _positive_fraction_record(
                mass_squared_difference, "MeV^2", "positive masses imply m_sigma>m_A"
            ),
            "g_omega_squared_over_g_s_squared_minus_one": _positive_fraction_record(
                coupling_ratio - 1, "1", "coordinate-space bracket is already positive as r approaches zero"
            ),
            "static_contact_vector_minus_scalar": _positive_fraction_record(
                contact_difference, "MeV^-2", "V_tilde(0)"
            ),
        },
        "proof": {
            "coordinate_strictly_positive": True,
            "coordinate_reason": (
                "For r>0, m_sigma>m_A and g_omega^2/g_s^2>1; therefore "
                "(g_omega^2/g_s^2) exp((m_sigma-m_A)r)-1>0."
            ),
            "momentum_strictly_positive": True,
            "momentum_reason": (
                "For Q2>=0, both denominator factors are positive and both exact "
                "coefficients of the factored numerator are positive."
            ),
            "static_hamiltonian_nonnegative": True,
            "static_hamiltonian_reason": (
                "Within the leading static nonrelativistic reduction, -nabla^2/(2*mu)+V "
                "has nonnegative kinetic and strictly positive r>0 potential terms, so it has no E<0 bound state."
            ),
        },
    }


def _quadratic_vacuum_expansion() -> dict[str, Any]:
    """State the action expansion that fixes the exchange vertices and signs.

    This is deliberately textual rather than a second copied action kernel:
    the numeric identities themselves are reconstructed from ``Params`` in
    :func:`_action_input_records` and the exact certificates.
    """
    return {
        "field_definition": "unitary gauge: W=W0+sigma and g_mu=-q_phi A_mu",
        "quadratic_lagrangian": (
            "L2=(partial sigma)^2/2-m_sigma^2 sigma^2/2-F^2/4+m_A^2 A_mu A^mu/2"
            "+bar N(i slash partial-M_N)N"
        ),
        "leading_vertices": "L_int=-g_s sigma bar N N-g_omega A_mu bar N gamma^mu N+...",
        "no_sigma_A_bilinear": True,
        "radial_mass_definition": "m_sigma^2=2 lambda W0^2 (not mu0^2=lambda W0^2)",
        "vector_mass_definition": "m_A^2=q_phi^2 W0^2=m_omega^2",
        "vector_longitudinal_scope": (
            "For the conserved equal-mass on-shell current used in the leading static exchange, "
            "q_mu bar u(p') gamma^mu u(p)=0; the longitudinal propagator term does not alter the kernel."
        ),
    }


def _diagnostics(exact: ExactParameters) -> dict[str, Any]:
    """Return dimensionless ratios without fitting or importing any NN datum."""
    with mp.workdps(80):
        gs2 = _mp_fraction(exact.g_s_squared)
        gw2 = _mp_fraction(exact.g_omega_squared)
        mA2 = _mp_fraction(exact.m_A_squared)
        ms2 = _mp_fraction(exact.m_sigma_squared)
        hbar_c = _mp_fraction(exact.hbar_c)
        pi = mp.pi
        mA = mp.sqrt(mA2)
        ms = mp.sqrt(ms2)
        alpha_vector = gw2 / (4 * pi)
        alpha_scalar = gs2 / (4 * pi)
        coupling_ratio = gw2 / gs2
        contact_vector = gw2 / mA2
        contact_scalar = gs2 / ms2
        contact_ratio = contact_vector / contact_scalar
        contact_difference = contact_vector - contact_scalar
        return {
            "alpha_vector_gomega2_over_4pi": _number(alpha_vector),
            "alpha_scalar_gs2_over_4pi": _number(alpha_scalar),
            "coordinate_repulsion_to_attraction_infimum": _number(coupling_ratio),
            "coordinate_repulsion_to_attraction_statement": (
                "V_vector/abs(V_scalar)=(g_omega^2/g_s^2)*exp((m_sigma-m_A)r); "
                "its infimum over r>0 is g_omega^2/g_s^2."
            ),
            "mass_ratio_m_sigma_over_m_A": _number(ms / mA),
            "momentum_repulsion_to_attraction_infimum": _number(coupling_ratio),
            "momentum_repulsion_to_attraction_statement": (
                "Vtilde_vector/abs(Vtilde_scalar)=(g_omega^2/g_s^2)*(Q2+m_sigma^2)/(Q2+m_A^2); "
                "its infimum over Q2>=0 is g_omega^2/g_s^2."
            ),
            "range_vector_hbarc_over_m_A_fm": _number(hbar_c / mA),
            "range_scalar_hbarc_over_m_sigma_fm": _number(hbar_c / ms),
            "static_contact_vector_MeV_minus2": _number(contact_vector),
            "static_contact_scalar_MeV_minus2": _number(contact_scalar),
            "static_contact_ratio_vector_over_scalar": _number(contact_ratio),
            "static_contact_net_MeV_minus2": _number(contact_difference),
            "static_contact_net_fm2": _number(contact_difference * hbar_c * hbar_c),
            "tree_loop_counting_gomega2_over_16pi2": _number(gw2 / (16 * pi * pi)),
            "tree_loop_counting_gs2_over_16pi2": _number(gs2 / (16 * pi * pi)),
            "tree_loop_counting_lambda_over_16pi2": _number(_mp_fraction(exact.lam) / (16 * pi * pi)),
            "loop_counting_interpretation": (
                "These conventional dimensionless loop-counting combinations are diagnostics, not error bars. "
                "The vector value is not small, so a tree/static result is not a precision all-orders prediction."
            ),
        }


def _fresh_result_payload(p: Params = PARAMS) -> dict[str, Any]:
    """Build the deterministic payload without validating a prior payload.

    Keeping this constructor separate lets :func:`validate_result` compare an
    untrusted loaded artifact against a newly derived canonical payload
    without recursive validation.
    """
    exact = exact_parameters_from_live(p)
    result = {
        "action_inputs": _action_input_records(exact),
        "caveats": [
            "Only the vacuum, tree-level, static, leading nonrelativistic two-baryon exchange kernel is calculated.",
            "The theorem concerns two equal-sign N gauge currents. N-anti-N, finite-density screening, and dynamical external sources are outside this audit.",
            "A charged N probe in a Higgs phase needs a gauge-invariant dressed/static-source interpretation; this is not the closed homogeneous gauge-neutral ensemble.",
            "No pion, rho/isovector sector, contact operator, form factor, cutoff, compositeness model, electromagnetic sector, or fitted NN quantity is present.",
            "The bound source action does not define a complete proton/neutron flavor, spin-isospin, or QCD scattering model; this output is not a physical NN-channel prediction.",
            "g_omega^2/(16*pi^2) is not small and the upstream passport blocks the required renormalization records; do not promote this tree/static result to an all-orders quantum conclusion.",
            "No empirical NN table, binding energy, phase shift, scattering length, likelihood, or fit is read or used.",
        ],
        "certificate": _certificates(exact),
        "claim_scope": CLAIM_SCOPE,
        "diagnostics": _diagnostics(exact),
        "evidence_weight": EVIDENCE_WEIGHT,
        "provenance": _provenance(),
        "quadratic_vacuum_expansion": _quadratic_vacuum_expansion(),
        "schema": SCHEMA,
        "scope": {
            "included": [
                "bound local-U(1) source-complete action in its source-free vacuum",
                "M*(W)=g_s W with g_s=M_N/W0",
                "m_A=q_phi W0=m_omega and m_sigma=sqrt(2*lambda)W0",
                "same-sign nucleon current coupling D_N=partial+i*g_omega*A",
                "analytic static momentum and coordinate exchange kernels",
            ],
            "not_an_empirical_comparison": True,
            "parameter_fit_performed": False,
        },
        "status": STATUS,
    }
    return result


def build_result(p: Params = PARAMS) -> dict[str, Any]:
    """Build a fresh no-fit certificate and self-check its semantic surface."""
    result = _fresh_result_payload(p)
    validate_result(result, p=p, check_provenance=False)
    return result


def _as_fraction(record: Mapping[str, Any], label: str) -> Fraction:
    try:
        numerator = record["exact_numerator"]
        denominator = record["exact_denominator"]
        value = Fraction(int(str(numerator)), int(str(denominator)))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise StaticExchangeAuditError(f"invalid exact fraction record: {label}") from exc
    if str(record.get("decimal")) != _number(value):
        raise StaticExchangeAuditError(f"inconsistent decimal rendering: {label}")
    return value


def validate_result(result: Mapping[str, Any], p: Params = PARAMS, *, check_provenance: bool = True) -> None:
    """Fail closed on semantic, exact-certificate, or provenance drift."""
    if not isinstance(result, Mapping):
        raise StaticExchangeAuditError("result must be an object")
    # This is deliberately a byte-level comparison rather than ``dict ==``:
    # in Python, False == 0.0, which would otherwise allow a boolean to
    # masquerade as the numerical evidence weight.  It also validates every
    # explanatory field, diagnostic, formula, and scope string rather than a
    # selective subset of them.  The expected side is a fresh action-derived
    # payload and never reads the saved JSON artifact.
    expected_payload = _fresh_result_payload(p)
    actual_payload = dict(result)
    if not check_provenance:
        actual_payload.pop("provenance", None)
        expected_payload.pop("provenance", None)
    if _canonical_json(actual_payload) != _canonical_json(expected_payload):
        raise StaticExchangeAuditError("serialized payload differs from a fresh action-derived certificate")
    required = {"action_inputs", "caveats", "certificate", "claim_scope", "diagnostics", "evidence_weight", "provenance", "quadratic_vacuum_expansion", "schema", "scope", "status"}
    if set(result) != required:
        raise StaticExchangeAuditError("unexpected result keys")
    if result.get("schema") != SCHEMA or result.get("status") != STATUS:
        raise StaticExchangeAuditError("schema/status mismatch")
    if result.get("evidence_weight") != EVIDENCE_WEIGHT:
        raise StaticExchangeAuditError("evidence-weight mutation")
    if result.get("claim_scope") != CLAIM_SCOPE:
        raise StaticExchangeAuditError("claim-scope mutation")
    caveats = result.get("caveats")
    if not isinstance(caveats, list) or len(caveats) != 7 or not all(isinstance(item, str) for item in caveats):
        raise StaticExchangeAuditError("missing scientific caveats")
    if not all(token in " ".join(caveats).lower() for token in ("tree", "renormalization", "empirical")):
        raise StaticExchangeAuditError("caveat scope was weakened")

    expansion = result.get("quadratic_vacuum_expansion")
    expected_expansion_keys = {
        "field_definition", "quadratic_lagrangian", "leading_vertices", "no_sigma_A_bilinear",
        "radial_mass_definition", "vector_mass_definition", "vector_longitudinal_scope",
    }
    if not isinstance(expansion, Mapping) or set(expansion) != expected_expansion_keys:
        raise StaticExchangeAuditError("quadratic-expansion structure invalid")
    if expansion.get("no_sigma_A_bilinear") is not True:
        raise StaticExchangeAuditError("quadratic sigma-A mixing was introduced")
    if "2 lambda W0^2" not in str(expansion.get("radial_mass_definition")):
        raise StaticExchangeAuditError("wrong radial mass definition")
    if "g_omega" not in str(expansion.get("leading_vertices")) or "q_phi" in str(expansion.get("leading_vertices")):
        raise StaticExchangeAuditError("wrong leading vector vertex")

    exact = exact_parameters_from_live(p)
    inputs = result.get("action_inputs")
    if not isinstance(inputs, Mapping):
        raise StaticExchangeAuditError("action inputs missing")
    base = inputs.get("base_live_decimal_inputs")
    derived = inputs.get("derived_from_bound_action")
    assignment = inputs.get("coupling_assignment")
    if not isinstance(base, Mapping) or not isinstance(derived, Mapping) or not isinstance(assignment, Mapping):
        raise StaticExchangeAuditError("action input structure invalid")
    expected_base = {
        "M_N_MeV": exact.M_N,
        "W0_MeV": exact.W0,
        "g_omega": exact.g_omega,
        "hbar_c_MeV_fm": exact.hbar_c,
        "lambda": exact.lam,
        "m_omega_MeV": exact.m_omega,
    }
    if set(base) != set(expected_base):
        raise StaticExchangeAuditError("base input set changed")
    for key, expected in expected_base.items():
        if _as_fraction(base[key], key) != expected:
            raise StaticExchangeAuditError(f"base input drift: {key}")
    if _as_fraction(derived.get("g_s_M_N_over_W0", {}), "g_s") != exact.g_s:
        raise StaticExchangeAuditError("g_s derivation drift")
    if _as_fraction(derived.get("q_phi_momega_over_W0", {}), "q_phi") != exact.q_phi:
        raise StaticExchangeAuditError("q_phi derivation drift")
    if _as_fraction(derived.get("m_A_squared_MeV2", {}), "m_A_squared") != exact.m_A_squared:
        raise StaticExchangeAuditError("m_A derivation drift")
    if _as_fraction(derived.get("m_sigma_squared_MeV2", {}), "m_sigma_squared") != exact.m_sigma_squared:
        raise StaticExchangeAuditError("m_sigma derivation drift")
    with mp.workdps(80):
        expected_mA = _number(mp.sqrt(_mp_fraction(exact.m_A_squared)))
        expected_ms = _number(mp.sqrt(_mp_fraction(exact.m_sigma_squared)))
    if derived.get("m_A_MeV") != expected_mA or derived.get("m_sigma_MeV") != expected_ms:
        raise StaticExchangeAuditError("derived mass rendering drift")
    if assignment.get("vector_exchange_nucleon_vertex") != "g_omega (not q_phi)":
        raise StaticExchangeAuditError("wrong vector source coupling")

    certificate = result.get("certificate")
    if not isinstance(certificate, Mapping):
        raise StaticExchangeAuditError("certificate missing")
    inequalities = certificate.get("exact_rational_inequalities")
    proof = certificate.get("proof")
    if not isinstance(inequalities, Mapping) or not isinstance(proof, Mapping):
        raise StaticExchangeAuditError("certificate structure invalid")
    expected_inequalities = {
        "g_omega_squared_minus_g_s_squared": exact.g_omega_squared - exact.g_s_squared,
        "g_omega_squared_m_sigma_squared_minus_g_s_squared_m_A_squared": exact.g_omega_squared * exact.m_sigma_squared - exact.g_s_squared * exact.m_A_squared,
        "m_sigma_squared_minus_m_A_squared": exact.m_sigma_squared - exact.m_A_squared,
        "g_omega_squared_over_g_s_squared_minus_one": exact.g_omega_squared / exact.g_s_squared - 1,
        "static_contact_vector_minus_scalar": exact.g_omega_squared / exact.m_A_squared - exact.g_s_squared / exact.m_sigma_squared,
    }
    if set(inequalities) != set(expected_inequalities):
        raise StaticExchangeAuditError("exact inequality set changed")
    for key, expected in expected_inequalities.items():
        row = inequalities[key]
        if not isinstance(row, Mapping) or row.get("strictly_positive") is not True:
            raise StaticExchangeAuditError(f"inequality flag false: {key}")
        if _as_fraction(row, key) != expected or expected <= 0:
            raise StaticExchangeAuditError(f"inequality certificate mismatch: {key}")
    for key in ("coordinate_strictly_positive", "momentum_strictly_positive", "static_hamiltonian_nonnegative"):
        if proof.get(key) is not True:
            raise StaticExchangeAuditError(f"proof gate false: {key}")
    kernels = certificate.get("kernel_definitions")
    expected_kernels = {
        "coordinate_natural": "V(r)=[g_omega^2 exp(-m_A r)-g_s^2 exp(-m_sigma r)]/(4*pi*r)",
        "coordinate_fm": "V(r_fm)=hbar_c*[g_omega^2 exp(-m_A*r_fm/hbar_c)-g_s^2 exp(-m_sigma*r_fm/hbar_c)]/(4*pi*r_fm)",
        "momentum": "V_tilde(Q2)=g_omega^2/(Q2+m_A^2)-g_s^2/(Q2+m_sigma^2)",
        "momentum_factored": "V_tilde(Q2)=[(g_omega^2-g_s^2)Q2+(g_omega^2*m_sigma^2-g_s^2*m_A^2)]/[(Q2+m_A^2)(Q2+m_sigma^2)]",
        "coordinate_factored": "V(r)=g_s^2 exp(-m_sigma r)*[(g_omega^2/g_s^2) exp((m_sigma-m_A)r)-1]/(4*pi*r)",
    }
    if not isinstance(kernels, Mapping) or dict(kernels) != expected_kernels:
        raise StaticExchangeAuditError("momentum kernel mutation")

    diagnostics = result.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise StaticExchangeAuditError("diagnostics missing")
    required_diagnostics = {
        "alpha_vector_gomega2_over_4pi", "alpha_scalar_gs2_over_4pi",
        "coordinate_repulsion_to_attraction_infimum", "coordinate_repulsion_to_attraction_statement",
        "mass_ratio_m_sigma_over_m_A", "momentum_repulsion_to_attraction_infimum",
        "momentum_repulsion_to_attraction_statement", "range_vector_hbarc_over_m_A_fm",
        "range_scalar_hbarc_over_m_sigma_fm", "static_contact_vector_MeV_minus2",
        "static_contact_scalar_MeV_minus2", "static_contact_ratio_vector_over_scalar",
        "static_contact_net_MeV_minus2", "static_contact_net_fm2",
        "tree_loop_counting_gomega2_over_16pi2", "tree_loop_counting_gs2_over_16pi2",
        "tree_loop_counting_lambda_over_16pi2", "loop_counting_interpretation",
    }
    if set(diagnostics) != required_diagnostics:
        raise StaticExchangeAuditError("diagnostic set changed")
    with mp.workdps(80):
        checks = {
            "coordinate_repulsion_to_attraction_infimum": _mp_fraction(exact.g_omega_squared / exact.g_s_squared),
            "momentum_repulsion_to_attraction_infimum": _mp_fraction(exact.g_omega_squared / exact.g_s_squared),
            "mass_ratio_m_sigma_over_m_A": mp.sqrt(_mp_fraction(exact.m_sigma_squared / exact.m_A_squared)),
            "static_contact_ratio_vector_over_scalar": _mp_fraction((exact.g_omega_squared / exact.m_A_squared) / (exact.g_s_squared / exact.m_sigma_squared)),
            "tree_loop_counting_gomega2_over_16pi2": _mp_fraction(exact.g_omega_squared) / (16 * mp.pi**2),
        }
        for key, expected in checks.items():
            actual = mp.mpf(str(diagnostics[key]))
            if not mp.almosteq(actual, expected, rel_eps=mp.mpf("1e-27"), abs_eps=mp.mpf("1e-27")):
                raise StaticExchangeAuditError(f"diagnostic drift: {key}")
        if not (mp.mpf(str(diagnostics["coordinate_repulsion_to_attraction_infimum"])) > 1 and
                mp.mpf(str(diagnostics["mass_ratio_m_sigma_over_m_A"])) > 1 and
                mp.mpf(str(diagnostics["static_contact_net_MeV_minus2"])) > 0):
            raise StaticExchangeAuditError("positivity diagnostic lost")

    scope = result.get("scope")
    if not isinstance(scope, Mapping) or scope.get("parameter_fit_performed") is not False or scope.get("not_an_empirical_comparison") is not True:
        raise StaticExchangeAuditError("scope mutation")
    if check_provenance:
        provenance = result.get("provenance")
        if not isinstance(provenance, Mapping) or provenance != _provenance():
            raise StaticExchangeAuditError("provenance drift")


def write_result(path: Path = RESULT_PATH, p: Params = PARAMS) -> dict[str, Any]:
    result = build_result(p)
    payload = _canonical_json(result)
    # A second fresh build must have byte-identical output; the producer never
    # treats a previous result file as an input.
    if payload != _canonical_json(build_result(p)):
        raise StaticExchangeAuditError("non-deterministic result generation")
    path.write_bytes(payload)
    return result


def load_and_validate(path: Path, p: Params = PARAMS) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticExchangeAuditError(f"cannot load result: {path}") from exc
    validate_result(result, p=p, check_provenance=True)
    if _canonical_json(result) != path.read_bytes():
        raise StaticExchangeAuditError("result is not canonical JSON")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=Path, help="write a fresh canonical JSON result")
    group.add_argument("--validate", type=Path, help="validate an existing canonical JSON result")
    args = parser.parse_args(argv)
    try:
        if args.validate is not None:
            load_and_validate(args.validate)
            print(f"PASS: {args.validate}")
        else:
            path = args.output if args.output is not None else RESULT_PATH
            write_result(path)
            print(f"WROTE: {path}")
    except StaticExchangeAuditError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
