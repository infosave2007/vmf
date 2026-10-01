#!/usr/bin/env python3
"""Fail-closed cross-scale entry gate for a symbolic one-parameter lambda deformation.

This audit combines only two analytic consequences of the *unmodified* bound
source action, reconstructed from live ``source_complete_solution_audit.Params``:

* the sharp d=4, s=1 Thomas--Fermi Q4 no-self-binding certificate; and
* the tree/static q=0 scalar-versus-vector Yukawa exchange sign; and
* the related coordinate-space range ordering of that same static kernel.

It does not fit or choose a new lambda.  Instead, it states the necessary
entry inequalities a hypothetical positive ``lambda_d`` would have to satisfy
    before the first two particular obstructions cease to apply.  The
    coordinate-space result is a sign map for the same reduced kernel, not a
    third binding criterion.  Passing any gate is not a binding, NN, nuclear,
    or physical-theory result.
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
RESULT_PATH = HERE / "nvg_source_complete_scalar_cross_scale_gate_results.json"
SOURCE_PATH = HERE / "source_complete_solution_audit.py"
ACTION_PATH = HERE / "contracts" / "source_complete_action.md"
Q4_AUDIT_PATH = HERE / "nvg_source_complete_q4_droplet_audit.py"
NN_AUDIT_PATH = HERE / "nvg_source_complete_nn_static_audit.py"
SCHEMA = "nvg_source_complete_scalar_cross_scale_gate.v1"
STATUS = "PASS_ENTRY_GATE_BASELINE_OUTSIDE_WINDOW"
EVIDENCE_WEIGHT = 0.0

# Exact classical bounds used only to prove ordering without treating a
# binary64 approximation to pi as a certificate.
PI_LOWER = Fraction(3, 1)       # pi > 3
PI_UPPER = Fraction(22, 7)      # pi < 22/7
Q4_C_CRITICAL = Fraction(1, 4)  # sharp Jensen certificate threshold


class CrossScaleGateError(ValueError):
    """Raised when the action surface or an artifact fails closed."""


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
    def m_sigma_squared_baseline(self) -> Fraction:
        return 2 * self.lambda_baseline * self.W0 * self.W0


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:  # pragma: no cover - direct propagation.
        raise CrossScaleGateError(f"cannot read provenance path: {path}") from exc


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CrossScaleGateError(f"{label} cannot be bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CrossScaleGateError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise CrossScaleGateError(f"{label} is not finite")
    return number


def _live_fraction(value: Any, label: str) -> Fraction:
    """Read a positive live parameter through its public decimal spelling."""

    number = _finite_float(value, label)
    if number <= 0.0:
        raise CrossScaleGateError(f"{label} must be strictly positive")
    try:
        return Fraction(repr(number))
    except (ValueError, ZeroDivisionError) as exc:  # pragma: no cover
        raise CrossScaleGateError(f"{label} has no usable decimal spelling") from exc


def _mp_fraction(value: Fraction) -> mp.mpf:
    return mp.mpf(value.numerator) / mp.mpf(value.denominator)


def _number(value: Any, digits: int = 32) -> str:
    """Precision-independent decimal rendering for a result artifact."""

    with mp.workdps(max(80, digits + 20)):
        if isinstance(value, Fraction):
            converted = _mp_fraction(value)
        else:
            converted = mp.mpf(value)
        if not mp.isfinite(converted):
            raise CrossScaleGateError("non-finite scientific output")
        return mp.nstr(converted, digits)


def _fraction_record(value: Fraction, unit: str, derivation: str) -> dict[str, str]:
    return {
        "decimal": _number(value),
        "derivation": derivation,
        "exact_numerator": str(value.numerator),
        "exact_denominator": str(value.denominator),
        "unit": unit,
    }


def _strict_fraction_record(value: Fraction, unit: str, derivation: str) -> dict[str, Any]:
    record: dict[str, Any] = _fraction_record(value, unit, derivation)
    record["strictly_positive"] = bool(value > 0)
    return record


def exact_parameters_from_live(p: Params = PARAMS) -> ExactParameters:
    """Read only live action parameters and guard all derived action identities."""

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
        expected_q = float(_mp_fraction(exact.q_phi))
        expected_m_sigma = float(mp.sqrt(_mp_fraction(exact.m_sigma_squared_baseline)))
    for actual, expected, relation in (
        (p.g_s, expected_gs, "g_s=M_N/W0"),
        (p.q_phi, expected_q, "q_phi=m_omega/W0"),
        (p.scalar_mass, expected_m_sigma, "m_sigma=sqrt(2*lambda)*W0"),
    ):
        if abs(_finite_float(actual, relation) - expected) / max(1.0, abs(expected)) > 2.0e-14:
            raise CrossScaleGateError(f"live source no longer satisfies {relation}")
    if int(p.degeneracy) != 4:
        raise CrossScaleGateError("cross-scale Q4 gate is defined only for d=4")
    return exact


def lambda_q4_display(exact: ExactParameters) -> mp.mpf:
    """Display the exact-pi Q4 threshold at controlled precision.

    Q4's sharp local certificate is C(lambda)>1/4, with
    C=3*pi^2 lambda W0^4/(8 M_N^4).  Hence lambda_Q4=2M_N^4/(3*pi^2W0^4).
    The formal ordering checks use rational pi bounds below instead.
    """

    with mp.workdps(100):
        return 2 * _mp_fraction(exact.M_N) ** 4 / (3 * mp.pi**2 * _mp_fraction(exact.W0) ** 4)


def lambda_q4_bounds(exact: ExactParameters) -> tuple[Fraction, Fraction]:
    """Return strict lower/upper rational bounds for lambda_Q4.

    pi<22/7 gives lambda_Q4 > lower; pi>3 gives lambda_Q4 < upper.
    """

    numerator = 2 * exact.M_N**4
    lower = numerator / (3 * PI_UPPER**2 * exact.W0**4)
    upper = numerator / (3 * PI_LOWER**2 * exact.W0**4)
    if not (lower > 0 and upper > lower):
        raise CrossScaleGateError("invalid rational Q4 threshold bounds")
    return lower, upper


def lambda_q0_attraction(exact: ExactParameters) -> Fraction:
    """Exact lambda at which Vtilde_static(Q2=0) is zero.

    With all non-lambda action parameters fixed,
      Vtilde(0)=g_omega^2/m_omega^2-M_N^2/(2 lambda_d W0^4).
    It is attractive iff 0<lambda_d<lambda_q0 below.
    """

    threshold = exact.M_N**2 * exact.m_omega**2 / (2 * exact.g_omega**2 * exact.W0**4)
    if threshold <= 0:
        raise CrossScaleGateError("nonpositive q=0 attraction threshold")
    return threshold


def lambda_coordinate_tail(exact: ExactParameters) -> Fraction:
    """Return the static-coordinate range threshold exactly.

    The existing NN static reduction has

      V(r)=[g_omega^2 exp(-m_omega r)-g_s^2 exp(-m_sigma r)]/(4*pi*r).

    The live source has ``g_omega^2>g_s^2``.  Therefore it is positive at
    every r>0 when m_sigma>=m_omega, and has one negative long-range tail
    when m_sigma<m_omega.  Under the symbolic lambda deformation the latter
    mass ordering switches at lambda=m_omega^2/(2 W0^2).
    """

    if exact.g_omega * exact.g_omega <= exact.g_s * exact.g_s:
        raise CrossScaleGateError("source does not certify g_omega^2>g_s^2 for the coordinate sign map")
    threshold = exact.m_omega**2 / (2 * exact.W0**2)
    if threshold <= 0:
        raise CrossScaleGateError("nonpositive coordinate-tail threshold")
    return threshold


def tail_crossover_radius_fm(lambda_d: Fraction | mp.mpf, exact: ExactParameters) -> mp.mpf:
    """Return the unique static-coordinate sign crossover for 0<lambda_d<lambda_range.

    This is deliberately a diagnostic of the tree/static coordinate kernel,
    not a classical turning point or a bound-state radius.
    """

    threshold = lambda_coordinate_tail(exact)
    with mp.workdps(100):
        lam = _mp_fraction(lambda_d) if isinstance(lambda_d, Fraction) else mp.mpf(lambda_d)
        threshold_mp = _mp_fraction(threshold)
        if not mp.isfinite(lam) or not (0 < lam < threshold_mp):
            raise CrossScaleGateError("tail crossover exists only for 0<lambda_d<lambda_range")
        coupling_ratio = _mp_fraction(exact.g_omega * exact.g_omega / (exact.g_s * exact.g_s))
        if coupling_ratio <= 1:
            raise CrossScaleGateError("coordinate tail crossover requires g_omega^2>g_s^2")
        m_sigma = mp.sqrt(2 * lam * _mp_fraction(exact.W0) ** 2)
        denominator = _mp_fraction(exact.m_omega) - m_sigma
        if denominator <= 0:
            raise CrossScaleGateError("coordinate tail crossover denominator is not positive")
        return _mp_fraction(exact.hbar_c) * mp.log(coupling_ratio) / denominator


def _provenance() -> dict[str, str]:
    paths = {
        "action": ACTION_PATH,
        "source_complete": SOURCE_PATH,
        "q4_audit": Q4_AUDIT_PATH,
        "nn_static_audit": NN_AUDIT_PATH,
        "producer": Path(__file__),
    }
    return {
        f"{label}_path": str(path.relative_to(REPO_ROOT))
        for label, path in paths.items()
    } | {
        f"{label}_sha256": _sha256(path)
        for label, path in paths.items()
    }


def _action_records(exact: ExactParameters) -> dict[str, Any]:
    with mp.workdps(80):
        return {
            "baseline_live_decimal_inputs": {
                "lambda": _fraction_record(exact.lambda_baseline, "1", "live Params.lam"),
                "M_N_MeV": _fraction_record(exact.M_N, "MeV", "live Params.M_N"),
                "W0_MeV": _fraction_record(exact.W0, "MeV", "live Params.W0"),
                "m_omega_MeV": _fraction_record(exact.m_omega, "MeV", "live Params.m_omega"),
                "g_omega": _fraction_record(exact.g_omega, "1", "live Params.g_omega"),
                "hbar_c_MeV_fm": _fraction_record(exact.hbar_c, "MeV fm", "live Params.hbar_c"),
            },
            "derived_action_relations": {
                "g_s_M_N_over_W0": _fraction_record(exact.g_s, "1", "M_N/W0"),
                "q_phi_momega_over_W0": _fraction_record(exact.q_phi, "1", "m_omega/W0"),
                "m_A_squared_MeV2": _fraction_record(exact.m_A_squared, "MeV^2", "q_phi^2 W0^2=m_omega^2"),
                "m_sigma_squared_under_deformation": "m_sigma(lambda_d)^2=2*lambda_d*W0^2",
            },
        }


def _q4_gate(exact: ExactParameters) -> dict[str, Any]:
    lower, upper = lambda_q4_bounds(exact)
    display = lambda_q4_display(exact)
    baseline_above = exact.lambda_baseline > upper
    if not baseline_above:
        raise CrossScaleGateError("baseline lambda cannot be certified above sharp Q4 threshold")
    with mp.workdps(100):
        C_baseline = (
            3
            * mp.pi**2
            * _mp_fraction(exact.lambda_baseline)
            * _mp_fraction(exact.W0) ** 4
            / (8 * _mp_fraction(exact.M_N) ** 4)
        )
        scalar_mass = mp.sqrt(2 * display * _mp_fraction(exact.W0) ** 2)
        scalar_range = _mp_fraction(exact.hbar_c) / scalar_mass
        ratio = _mp_fraction(exact.lambda_baseline) / display
    return {
        "origin": "sharp Q4 d=4,s=1 TF no-self-binding certificate C(lambda_d)>1/4",
        "coefficient": "C(lambda_d)=3*pi^2*lambda_d*W0^4/(8*M_N^4)",
        "strict_no_go_condition": "lambda_d > lambda_Q4",
        "condition_merely_to_evade_strict_certificate": "0 < lambda_d <= lambda_Q4",
        "lambda_Q4_exact_pi_display": _number(display),
        "lambda_Q4_rational_bounds": {
            "strict_lower_from_pi_lt_22_over_7": _strict_fraction_record(
                lower, "1", "2*M_N^4/[3*(22/7)^2*W0^4] < lambda_Q4"
            ),
            "strict_upper_from_pi_gt_3": _strict_fraction_record(
                upper, "1", "lambda_Q4 < 2*M_N^4/(27*W0^4)"
            ),
            "proof_baseline_lambda_above_lambda_Q4": baseline_above,
        },
        "baseline": {
            "lambda_baseline": _fraction_record(exact.lambda_baseline, "1", "live Params.lam"),
            "C_baseline_display": _number(C_baseline),
            "C_critical": "1/4",
            "baseline_over_lambda_Q4_display": _number(ratio),
            "status": "STRICT_Q4_NO_SELF_BINDING_CERTIFICATE_ACTIVE",
        },
        "threshold_scalar_diagnostics_not_predictions": {
            "m_sigma_at_lambda_Q4_MeV_display": _number(scalar_mass),
            "scalar_range_at_lambda_Q4_fm_display": _number(scalar_range),
        },
        "interpretation": (
            "At lambda_d=lambda_Q4 the sharp local result weakens from E>N_B M_N to E>=N_B M_N; "
            "lambda_d below it only removes this sufficient strict certificate and does not establish binding."
        ),
    }


def _q0_gate(exact: ExactParameters) -> dict[str, Any]:
    threshold = lambda_q0_attraction(exact)
    vector = exact.g_omega**2 / exact.m_omega**2
    scalar_baseline = exact.M_N**2 / (2 * exact.lambda_baseline * exact.W0**4)
    net_baseline = vector - scalar_baseline
    if not (vector > 0 and scalar_baseline > 0 and net_baseline > 0):
        raise CrossScaleGateError("baseline q=0 static kernel cannot be certified repulsive")
    with mp.workdps(100):
        scalar_mass = mp.sqrt(2 * _mp_fraction(threshold) * _mp_fraction(exact.W0) ** 2)
        scalar_range = _mp_fraction(exact.hbar_c) / scalar_mass
        baseline_over = _mp_fraction(exact.lambda_baseline) / _mp_fraction(threshold)
    return {
        "origin": "tree/static equal-sign N-current Yukawa kernel at Q2=0",
        "kernel": "Vtilde_lambda_d(0)=g_omega^2/m_omega^2-M_N^2/(2*lambda_d*W0^4)",
        "zero_kernel_threshold": _fraction_record(
            threshold, "1", "M_N^2*m_omega^2/(2*g_omega^2*W0^4)"
        ),
        "attraction_condition": "0 < lambda_d < lambda_q0",
        "equality_condition": "lambda_d=lambda_q0 gives Vtilde(0)=0",
        "repulsion_condition": "lambda_d > lambda_q0",
        "baseline": {
            "lambda_baseline_over_lambda_q0": _number(baseline_over),
            "static_vector_term_MeV_minus2": _fraction_record(vector, "MeV^-2", "g_omega^2/m_omega^2"),
            "static_scalar_term_baseline_MeV_minus2": _fraction_record(
                scalar_baseline, "MeV^-2", "M_N^2/(2*lambda_baseline*W0^4)"
            ),
            "static_net_kernel_baseline_MeV_minus2": _strict_fraction_record(
                net_baseline, "MeV^-2", "vector term minus scalar term"
            ),
            "status": "STATIC_Q0_REPULSIVE",
        },
        "threshold_scalar_diagnostics_not_predictions": {
            "m_sigma_at_lambda_q0_MeV_display": _number(scalar_mass),
            "scalar_range_at_lambda_q0_fm_display": _number(scalar_range),
        },
        "scope": (
            "This is only the leading tree/static q=0 Yukawa sign. It is not a coordinate-space, "
            "all-orders NN, QCD, deuteron, or finite-density binding calculation."
        ),
    }


def _coordinate_static_gate(exact: ExactParameters) -> dict[str, Any]:
    """Map the exact range sign of the same tree/static Yukawa kernel.

    This uses no coordinate grid or NN datum.  It is the direct algebraic
    consequence of the two Yukawa terms, the fixed vertex ordering, and the
    lambda-dependent scalar mass.  It deliberately does not solve a
    Schrodinger equation or infer a bound state.
    """

    threshold = lambda_coordinate_tail(exact)
    coupling_difference = exact.g_omega**2 - exact.g_s**2
    baseline_all_r_repulsive = exact.lambda_baseline >= threshold
    if not (coupling_difference > 0 and baseline_all_r_repulsive):
        raise CrossScaleGateError("baseline coordinate static kernel cannot be certified all-r repulsive")

    q4_lower, q4_upper = lambda_q4_bounds(exact)
    q0 = lambda_q0_attraction(exact)
    q4_below_range = q4_upper < threshold
    q0_below_q4 = q0 < q4_lower
    if not (q4_below_range and q0_below_q4):
        raise CrossScaleGateError("cannot certify coordinate-tail consequences of the Q4/q=0 thresholds")
    with mp.workdps(100):
        q4_display = lambda_q4_display(exact)
        q4_crossing = tail_crossover_radius_fm(q4_display, exact)
        q0_crossing = tail_crossover_radius_fm(q0, exact)
        baseline_over = _mp_fraction(exact.lambda_baseline) / _mp_fraction(threshold)
    return {
        "origin": "same tree/static coordinate Yukawa kernel used by the NN static audit",
        "kernel": (
            "V_lambda_d(r)=hbar_c*[g_omega^2*exp(-m_omega*r/hbar_c)"
            "-g_s^2*exp(-m_sigma(lambda_d)*r/hbar_c)]/(4*pi*r), r>0"
        ),
        "fixed_vertex_ordering": _strict_fraction_record(
            coupling_difference, "1", "g_omega^2-(M_N/W0)^2 > 0"
        ),
        "lambda_range": _fraction_record(
            threshold, "1", "m_omega^2/(2*W0^2), equivalently m_sigma(lambda_range)=m_omega"
        ),
        "all_r_repulsion_condition": "lambda_d >= lambda_range",
        "long_range_attractive_tail_condition": "0 < lambda_d < lambda_range",
        "tail_crossover_when_present": {
            "formula_fm": (
                "r_cross=hbar_c*ln(g_omega^2/g_s^2)/"
                "[m_omega-sqrt(2*lambda_d)*W0]"
            ),
            "signs": "V(r)>0 for 0<r<r_cross and V(r)<0 for r>r_cross",
            "at_lambda_Q4_fm_display_not_prediction": _number(q4_crossing),
            "at_lambda_q0_fm_display_not_prediction": _number(q0_crossing),
        },
        "baseline": {
            "lambda_baseline_over_lambda_range": _number(baseline_over),
            "status": "STATIC_COORDINATE_ALL_R_REPULSIVE",
        },
        "cross_scale_consequence": (
            "The rational ordering lambda_Q4<lambda_range proves that every positive lambda_d "
            "which merely evades the strict Q4 certificate has this tree/static negative long-range tail. "
            "A negative tail is not a binding proof."
        ),
        "scope": (
            "This is only a leading tree/static coordinate-kernel sign statement. It is not a coordinate-space "
            "bound-state solution, all-orders NN result, QCD calculation, or finite-density prediction."
        ),
    }


def _cross_scale_gate(exact: ExactParameters) -> dict[str, Any]:
    q0 = lambda_q0_attraction(exact)
    coordinate_range = lambda_coordinate_tail(exact)
    q4_lower, q4_upper = lambda_q4_bounds(exact)
    nested = q0 < q4_lower
    q4_below_coordinate_range = q4_upper < coordinate_range
    range_below_baseline = coordinate_range < exact.lambda_baseline
    if not (nested and q4_below_coordinate_range and range_below_baseline):
        raise CrossScaleGateError("cannot certify the source-action cross-scale regime ordering")
    with mp.workdps(100):
        q4_display = lambda_q4_display(exact)
        q0_mp = _mp_fraction(q0)
        ratio = q4_display / q0_mp
    return {
        "logical_intersection": "0 < lambda_d < lambda_q0",
        "why_q0_gate_is_stricter": (
            "lambda_q0 is strictly below a rational lower bound on lambda_Q4, so any positive "
            "lambda_d that gives q=0 attraction also merely evades the strict Q4 certificate."
        ),
        "exact_ordering_certificate": {
            "lambda_q0": _fraction_record(q0, "1", "q=0 zero-kernel threshold"),
            "lambda_Q4_strict_lower_bound": _fraction_record(
                q4_lower, "1", "from pi<22/7"
            ),
            "lambda_Q4_strict_upper_bound": _fraction_record(
                q4_upper, "1", "from pi>3"
            ),
            "lambda_range": _fraction_record(
                coordinate_range, "1", "coordinate static range threshold"
            ),
            "lambda_baseline": _fraction_record(exact.lambda_baseline, "1", "live Params.lam"),
            "lambda_q0_strictly_less_than_lambda_Q4": nested,
            "lambda_Q4_strictly_less_than_lambda_range": q4_below_coordinate_range,
            "lambda_range_strictly_less_than_baseline": range_below_baseline,
            "certified_ordering": "lambda_q0 < lambda_Q4 < lambda_range < lambda_baseline",
        },
        "display_ratio_lambda_Q4_over_lambda_q0": _number(ratio),
        "baseline_is_outside_intersection": bool(exact.lambda_baseline >= q0),
        "gate_status": "BASELINE_OUTSIDE_ENTRY_WINDOW",
        "analytic_regime_map_not_binding": [
            {
                "interval": "lambda_d >= lambda_range",
                "tree_static_signs": "V_tilde(Q2)>0 for Q2>=0 and V(r)>0 for every r>0",
                "Q4_status": "strict Q4 no-self-binding certificate active",
            },
            {
                "interval": "lambda_Q4 < lambda_d < lambda_range",
                "tree_static_signs": "V_tilde(Q2)>0 for Q2>=0, while V(r) has a negative long-range tail",
                "Q4_status": "strict Q4 no-self-binding certificate active",
            },
            {
                "interval": "lambda_q0 <= lambda_d <= lambda_Q4",
                "tree_static_signs": "V_tilde(0)>=0 and V(r) has a negative long-range tail",
                "Q4_status": "strict certificate evaded only; binding is undecided",
            },
            {
                "interval": "0 < lambda_d < lambda_q0",
                "tree_static_signs": "V_tilde(0)<0 and V(r) has a negative long-range tail",
                "Q4_status": "strict certificate evaded only; binding is undecided",
            },
        ],
        "not_a_solution": (
            "The intersection is necessary only for these two obstructions to stop deciding the question; "
            "the coordinate tail is only an additional sign consequence. Neither is a parameter recommendation, "
            "fit, completed action, or evidence that a bound state exists."
        ),
    }


def _fresh_result_payload(p: Params = PARAMS) -> dict[str, Any]:
    exact = exact_parameters_from_live(p)
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "provenance": _provenance(),
        "action_inputs": _action_records(exact),
        "one_parameter_deformation_contract": {
            "symbol": "lambda_d",
            "domain": "lambda_d>0",
            "changed": ["lambda in U=lambda(W^2-W0^2)^2/4 and m_sigma^2=2*lambda*W0^2"],
            "held_fixed": ["W0", "M_N", "m_omega", "g_omega", "g_s=M_N/W0", "q_phi=m_omega/W0", "all omitted sectors"],
            "parameter_fit_performed": False,
            "no_new_operator_introduced": True,
            "hypothetical_deformed_action_member": True,
            "physical_action_replacement_claimed": False,
            "meaning": "symbolic necessary-entry analysis only; the live baseline is not changed",
        },
        "q4_tf_gate": _q4_gate(exact),
        "static_q0_yukawa_gate": _q0_gate(exact),
        "static_coordinate_yukawa_gate": _coordinate_static_gate(exact),
        "cross_scale_entry_gate": _cross_scale_gate(exact),
        "conclusion": {
            "baseline": (
                "The live lambda=1.05 is rigorously above the Q4 strict-no-go threshold and above the q=0 "
                "attraction and coordinate-range thresholds; baseline tree/static exchange is repulsive at q=0 "
                "and for every coordinate separation r>0."
            ),
            "entry_gate_only": (
                "A hypothetical deformation would need 0<lambda_d<lambda_q0 merely to evade the strict Q4 "
                "certificate and make this tree/static q=0 sign attractive."
            ),
            "not_claimed": [
                "existence of a self-bound droplet, nucleus, or NN bound state",
                "a preferred or fitted value of lambda_d",
                "a new interaction, energy source, or completed physical theory",
                "an all-orders, finite-q, coordinate-space, QCD, or experimental prediction",
            ],
        },
        "caveats": [
            "The Q4 result is a static zero-temperature d=4,s=1 Thomas-Fermi certificate, not a full quantum calculation.",
            "The q=0 result is the leading tree/static equal-sign-current Yukawa sign only; it does not by itself determine a finite-range bound-state spectrum.",
            "A coordinate-space negative tail or sign crossover is not a bound-state proof; a kinetic term and full controlled two-body calculation are still required.",
            "Changing lambda_d would alter the source action's scalar potential and mass; it is not a numerical repair and requires a separate derivation and validation before being called a model.",
            "No empirical nuclear/NN data, binding target, radius, scattering observable, cutoff, form factor, likelihood, or fit is read or used.",
            "Coulomb, isovector, pion/rho, density-dependent, transfer, gravity, pairing, shell, RPA, and finite-temperature sectors remain outside this gate.",
        ],
    }
    return result


def build_result(p: Params = PARAMS) -> dict[str, Any]:
    result = _fresh_result_payload(p)
    validate_result(result, p=p, check_provenance=False)
    return result


def validate_result(payload: Any, p: Params = PARAMS, *, check_provenance: bool = True) -> None:
    """Accept only an exact fresh derivation; all mutations fail closed."""

    if not isinstance(payload, Mapping):
        raise CrossScaleGateError("result payload must be an object")
    expected = _fresh_result_payload(p)
    if _canonical_json(payload) != _canonical_json(expected):
        raise CrossScaleGateError("payload is not an exact fresh source-action derivation")
    if check_provenance and payload.get("provenance") != _provenance():
        raise CrossScaleGateError("provenance changed")


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
        except (OSError, json.JSONDecodeError, CrossScaleGateError) as exc:
            print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
            return 1
        print("VALIDATION_PASS")
        return 0
    try:
        result = build_result()
        write_result(result, args.output)
    except CrossScaleGateError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 1
    print(_canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
