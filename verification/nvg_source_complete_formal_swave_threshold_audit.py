#!/usr/bin/env python3
"""Fail-closed formal S-wave threshold audit for the source-complete Yukawa reduction.

This is deliberately a calculation in a *formal* static, tree-level,
spinless two-body reduction of the bound source action.  It holds all live
source parameters fixed except for a symbolic positive deformation
``lambda -> lambda_d`` of the quartic scalar potential, hence
``m_sigma(lambda_d)^2 = 2 lambda_d W0^2``.  It locates the first zero-energy
S-wave threshold of that reduced Hamiltonian with nodeless shooting and
independent finite-difference inertia signs.

It is not a NN/deuteron/QCD calculation and does not select or license a
revised physical action.  It is not a parameter fit or physical evidence. The source action does not supply the omitted
spin, isospin, pion/rho, form-factor, relativistic two-body, loop, or matching
sectors; in particular its tree loop expansion is not assumed controlled.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import mpmath as mp

try:
    from scipy.integrate import solve_ivp
except ImportError as exc:  # pragma: no cover - scipy is in requirements.txt.
    raise RuntimeError("formal S-wave audit requires scipy.integrate.solve_ivp") from exc

try:  # Direct execution from ``verification``.
    from source_complete_solution_audit import PARAMS, Params
except ImportError:  # pragma: no cover - package-style import support.
    from .source_complete_solution_audit import PARAMS, Params


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULT_PATH = HERE / "nvg_source_complete_formal_swave_threshold_results.json"
SOURCE_PATH = HERE / "source_complete_solution_audit.py"
ACTION_PATH = HERE / "contracts" / "source_complete_action.md"
NN_AUDIT_PATH = HERE / "nvg_source_complete_nn_static_audit.py"
CROSS_SCALE_PATH = HERE / "nvg_source_complete_scalar_cross_scale_gate.py"
SCHEMA = "nvg_source_complete_formal_swave_threshold_audit.v1"
STATUS = "PASS_FORMAL_SWAVE_THRESHOLD_NOT_PHYSICAL"
EVIDENCE_WEIGHT = 0.0

# Exact elementary pi bounds are used only for the analytic comparator
# certificate.  Numerical shooting never treats a rounded pi as a proof.
PI_LOWER = Fraction(3, 1)       # pi > 3
PI_UPPER = Fraction(22, 7)      # pi < 22/7

# These are declared numerical controls, not fitted physical inputs.  The
# shooting coordinate is x=m_sigma(lambda_d) r_nat = m_sigma r_fm/(hbar c).
# Every outer endpoint is at least thirty scalar ranges at the threshold.
@dataclass(frozen=True)
class ShootingControl:
    identifier: str
    x_sigma_max: float
    rtol: float
    atol: float
    x_start: float
    max_step: float


SHOOTING_CONTROLS = (
    ShootingControl(
        "coarse_x30_rtol1e-9_step1e-1", 30.0, 1.0e-9, 1.0e-11, 1.0e-8, 1.0e-1
    ),
    ShootingControl(
        "medium_x50_rtol1e-11_step7p5e-2", 50.0, 1.0e-11, 1.0e-13, 1.0e-8, 7.5e-2
    ),
    ShootingControl(
        "fine_x80_rtol2e-13_step5e-2", 80.0, 2.0e-13, 2.0e-15, 1.0e-8, 5.0e-2
    ),
)

# The independent operator uses z=m_omega r_nat and finite Dirichlet walls.
# They are operator controls, not physical radii or a finite-volume model.
@dataclass(frozen=True)
class FiniteDifferenceControl:
    identifier: str
    z_omega_max: float
    dz: float


FINITE_DIFFERENCE_CONTROLS = (
    FiniteDifferenceControl("dirichlet_z500_dz0p2", 500.0, 0.2),
    FiniteDifferenceControl("dirichlet_z2000_dz0p1", 2000.0, 0.1),
    FiniteDifferenceControl("dirichlet_z8000_dz0p05", 8000.0, 0.05),
)

SEARCH_RATIO = 0.8
ROOT_RELATIVE_TOLERANCE = 2.0e-10
NODE_BRACKET_MAX_ITERATIONS = 96
ROOT_MAX_ITERATIONS = 96
SEARCH_MAX_ITERATIONS = 96
CONVERGENCE_RELATIVE_LIMIT = 1.0e-6
PIVOT_RELATIVE_FLOOR = 1.0e-10
INITIAL_START_REPLAYS = (1.0e-6, 1.0e-10)

# A predeclared, non-optimized trial state only witnesses that the formal
# Hamiltonian is bound well below the numerical threshold.  It is not fitted.
WITNESS_LAMBDA_DIVISOR = 10
WITNESS_BETA_OVER_M_SIGMA = Fraction(2, 5)


class FormalSWaveAuditError(ValueError):
    """Raised when the source contract or a numerical certificate fails closed."""


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
    def reduced_mass(self) -> Fraction:
        return self.M_N / 2

    @property
    def m_A_squared(self) -> Fraction:
        return self.m_omega * self.m_omega

    @property
    def m_sigma_squared_baseline(self) -> Fraction:
        return 2 * self.lambda_baseline * self.W0 * self.W0


@dataclass(frozen=True)
class ShootingState:
    """One regular zero-energy radial shooting endpoint."""

    lambda_d: float
    x_sigma_max: float
    x_start: float
    endpoint_u: float
    endpoint_du_dx: float
    endpoint_log_derivative: float
    node_count: int
    nfev: int
    physical_outer_radius_fm: float


_DEFAULT_CACHE: tuple[str, dict[str, Any]] | None = None


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:  # pragma: no cover - direct propagation.
        raise FormalSWaveAuditError(f"cannot read provenance path: {path}") from exc


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise FormalSWaveAuditError(f"{label} cannot be bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormalSWaveAuditError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise FormalSWaveAuditError(f"{label} is not finite")
    return number


def _live_fraction(value: Any, label: str) -> Fraction:
    number = _finite_float(value, label)
    if number <= 0.0:
        raise FormalSWaveAuditError(f"{label} must be strictly positive")
    try:
        return Fraction(repr(number))
    except (ValueError, ZeroDivisionError) as exc:  # pragma: no cover
        raise FormalSWaveAuditError(f"{label} has no usable decimal spelling") from exc


def _mp_fraction(value: Fraction) -> mp.mpf:
    return mp.mpf(value.numerator) / mp.mpf(value.denominator)


def _number(value: Any, digits: int = 16) -> str:
    """Precision-independent decimal serialization for a canonical artifact."""

    with mp.workdps(max(80, digits + 24)):
        if isinstance(value, Fraction):
            rendered = _mp_fraction(value)
        elif isinstance(value, float):
            # Preserve the public deterministic decimal spelling, rather than
            # serializing an implementation-dependent binary expansion.
            rendered = mp.mpf(repr(value))
        else:
            rendered = mp.mpf(value)
        if not mp.isfinite(rendered):
            raise FormalSWaveAuditError("non-finite scientific output")
        return mp.nstr(rendered, digits)


def _fraction_record(value: Fraction, unit: str, derivation: str) -> dict[str, str]:
    return {
        "decimal": _number(value, 32),
        "derivation": derivation,
        "exact_numerator": str(value.numerator),
        "exact_denominator": str(value.denominator),
        "unit": unit,
    }


def _positive_fraction_record(value: Fraction, unit: str, derivation: str) -> dict[str, Any]:
    record: dict[str, Any] = _fraction_record(value, unit, derivation)
    record["strictly_positive"] = bool(value > 0)
    return record


def exact_parameters_from_live(p: Params = PARAMS) -> ExactParameters:
    """Read live action inputs and guard every derived source relation used."""

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
        expected_m_sigma = float(mp.sqrt(_mp_fraction(exact.m_sigma_squared_baseline)))
    for actual, expected, relation in (
        (p.g_s, expected_gs, "g_s=M_N/W0"),
        (p.q_phi, expected_qphi, "q_phi=m_omega/W0"),
        (p.scalar_mass, expected_m_sigma, "m_sigma=sqrt(2*lambda)*W0"),
    ):
        scale = max(1.0, abs(expected))
        if abs(_finite_float(actual, relation) - expected) / scale > 2.0e-14:
            raise FormalSWaveAuditError(f"live source no longer satisfies {relation}")
    if exact.g_omega * exact.g_omega <= exact.g_s * exact.g_s:
        raise FormalSWaveAuditError("formal static kernel requires live g_omega^2>g_s^2")
    return exact


def _require_live_baseline(p: Params) -> None:
    """Do not silently turn an audit into a calculation for another action."""

    if p != PARAMS:
        raise FormalSWaveAuditError("this artifact is defined only for the live source-complete baseline")


def lambda_q0_attraction(exact: ExactParameters) -> Fraction:
    """The existing tree/static q=0 sign threshold, reconstructed live."""

    threshold = exact.M_N**2 * exact.m_omega**2 / (2 * exact.g_omega**2 * exact.W0**4)
    if threshold <= 0:
        raise FormalSWaveAuditError("nonpositive q=0 threshold")
    return threshold


def lambda_coordinate_range(exact: ExactParameters) -> Fraction:
    """Static-coordinate range-sign threshold m_sigma=m_omega."""

    threshold = exact.m_omega**2 / (2 * exact.W0**2)
    if threshold <= 0:
        raise FormalSWaveAuditError("nonpositive coordinate range threshold")
    return threshold


def lambda_bargmann_display(exact: ExactParameters) -> mp.mpf:
    """Display the formal Bargmann comparator boundary at high precision.

    For V_comp=-alpha_s exp(-m_sigma r)/r with alpha_s=g_s^2/(4*pi),
    2*mu*alpha_s/m_sigma<1 is sufficient for no S-wave bound state.
    """

    with mp.workdps(100):
        return _mp_fraction(exact.M_N) ** 6 / (32 * mp.pi**2 * _mp_fraction(exact.W0) ** 6)


def lambda_bargmann_bounds(exact: ExactParameters) -> tuple[Fraction, Fraction]:
    """Strict rational lower/upper bounds for lambda_B using 3<pi<22/7."""

    numerator = exact.M_N**6
    lower = numerator / (32 * PI_UPPER**2 * exact.W0**6)
    upper = numerator / (32 * PI_LOWER**2 * exact.W0**6)
    if not (lower > 0 and upper > lower):
        raise FormalSWaveAuditError("invalid Bargmann rational bounds")
    return lower, upper


def scalar_mass_mev(lambda_d: float, exact: ExactParameters) -> float:
    lam = _finite_float(lambda_d, "lambda_d")
    if lam <= 0.0:
        raise FormalSWaveAuditError("lambda_d must be positive")
    mass = math.sqrt(2.0 * lam) * float(exact.W0)
    if not math.isfinite(mass) or mass <= 0.0:
        raise FormalSWaveAuditError("invalid deformed scalar mass")
    return mass


def _sigma_coordinate_potential(x_sigma: float, lambda_d: float, exact: ExactParameters) -> float:
    """Dimensionless U in [-u''+Uu]=epsilon*u, x_sigma=m_sigma*r_nat."""

    if x_sigma <= 0.0 or not math.isfinite(x_sigma):
        raise FormalSWaveAuditError("x_sigma must be finite and positive")
    m_sigma = scalar_mass_mev(lambda_d, exact)
    beta = float(exact.m_omega) / m_sigma
    mu = float(exact.reduced_mass)
    coefficient = mu / (2.0 * math.pi * m_sigma)
    value = coefficient * (
        float(exact.g_omega) ** 2 * math.exp(-beta * x_sigma)
        - float(exact.g_s) ** 2 * math.exp(-x_sigma)
    ) / x_sigma
    if not math.isfinite(value):
        raise FormalSWaveAuditError("non-finite sigma-coordinate potential")
    return value


def _omega_coordinate_potential(z_omega: float, lambda_d: float, exact: ExactParameters) -> float:
    """Same dimensionless operator in z_omega=m_omega*r_nat for FD checks."""

    if z_omega <= 0.0 or not math.isfinite(z_omega):
        raise FormalSWaveAuditError("z_omega must be finite and positive")
    m_sigma = scalar_mass_mev(lambda_d, exact)
    eta = m_sigma / float(exact.m_omega)
    mu = float(exact.reduced_mass)
    coefficient = mu / (2.0 * math.pi * float(exact.m_omega))
    value = coefficient * (
        float(exact.g_omega) ** 2 * math.exp(-z_omega)
        - float(exact.g_s) ** 2 * math.exp(-eta * z_omega)
    ) / z_omega
    if not math.isfinite(value):
        raise FormalSWaveAuditError("non-finite omega-coordinate potential")
    return value


def _regular_initial_conditions(lambda_d: float, exact: ExactParameters, x_start: float) -> tuple[float, float]:
    """Use the regular O(x^3) series rather than evaluating the 1/x origin.

    U(x)=A/x+B+O(x), so u=x+(A/2)x^2+(A^2/12+B/6)x^3+O(x^4).
    """

    if not (math.isfinite(x_start) and 0.0 < x_start < 1.0):
        raise FormalSWaveAuditError("x_start must lie in (0,1)")
    m_sigma = scalar_mass_mev(lambda_d, exact)
    beta = float(exact.m_omega) / m_sigma
    coefficient = float(exact.reduced_mass) / (2.0 * math.pi * m_sigma)
    A = coefficient * (float(exact.g_omega) ** 2 - float(exact.g_s) ** 2)
    B = coefficient * (float(exact.g_s) ** 2 - beta * float(exact.g_omega) ** 2)
    cubic = A * A / 12.0 + B / 6.0
    u = x_start + 0.5 * A * x_start * x_start + cubic * x_start**3
    du_dx = 1.0 + A * x_start + 3.0 * cubic * x_start * x_start
    if not (math.isfinite(u) and math.isfinite(du_dx) and u > 0.0 and du_dx > 0.0):
        raise FormalSWaveAuditError("regular origin series failed")
    return u, du_dx


def shoot_zero_energy(lambda_d: float, control: ShootingControl, exact: ExactParameters) -> ShootingState:
    """Integrate the nodeless zero-energy S-wave equation with DOP853.

    The endpoint log derivative has the sign of the asymptotic slope on the
    nodeless branch.  At the first zero-energy threshold it tends to zero.
    """

    lam = _finite_float(lambda_d, "lambda_d")
    if lam <= 0.0:
        raise FormalSWaveAuditError("shooting requires lambda_d>0")
    if control.x_sigma_max < 30.0:
        raise FormalSWaveAuditError("shooting outer control must cover at least 30 scalar ranges")
    if not (
        0.0 < control.x_start < control.x_sigma_max
        and 0.0 < control.rtol < 1.0
        and 0.0 < control.atol < 1.0
        and 0.0 < control.max_step <= control.x_sigma_max
    ):
        raise FormalSWaveAuditError("invalid shooting control")
    initial_u, initial_du = _regular_initial_conditions(lam, exact, control.x_start)

    def rhs(x_sigma: float, y: list[float]) -> tuple[float, float]:
        return (y[1], _sigma_coordinate_potential(x_sigma, lam, exact) * y[0])

    def node_event(_x_sigma: float, y: list[float]) -> float:
        return y[0]

    node_event.terminal = False  # type: ignore[attr-defined]
    node_event.direction = 0.0  # type: ignore[attr-defined]
    solution = solve_ivp(
        rhs,
        (control.x_start, control.x_sigma_max),
        (initial_u, initial_du),
        method="DOP853",
        rtol=control.rtol,
        atol=control.atol,
        max_step=control.max_step,
        events=node_event,
    )
    if not solution.success or abs(solution.t[-1] - control.x_sigma_max) > 1.0e-12 * control.x_sigma_max:
        raise FormalSWaveAuditError(f"zero-energy shooting failed: {solution.message}")
    endpoint_u = _finite_float(solution.y[0, -1], "shooting endpoint u")
    endpoint_du = _finite_float(solution.y[1, -1], "shooting endpoint derivative")
    if endpoint_u == 0.0:
        raise FormalSWaveAuditError("shooting endpoint landed exactly on a node")
    accepted_steps = (
        solution.t[index + 1] - solution.t[index]
        for index in range(len(solution.t) - 1)
    )
    if any(step > control.max_step * (1.0 + 1.0e-12) for step in accepted_steps):
        raise FormalSWaveAuditError("shooting exceeded its declared maximum step")
    log_derivative = endpoint_du / endpoint_u
    if not math.isfinite(log_derivative):
        raise FormalSWaveAuditError("non-finite shooting log derivative")
    nodes = len(solution.t_events[0])
    m_sigma = scalar_mass_mev(lam, exact)
    return ShootingState(
        lambda_d=lam,
        x_sigma_max=control.x_sigma_max,
        x_start=control.x_start,
        endpoint_u=endpoint_u,
        endpoint_du_dx=endpoint_du,
        endpoint_log_derivative=log_derivative,
        node_count=int(nodes),
        nfev=int(solution.nfev),
        physical_outer_radius_fm=control.x_sigma_max * float(exact.hbar_c) / m_sigma,
    )


def _nodeless_positive(state: ShootingState) -> bool:
    return state.node_count == 0 and state.endpoint_u > 0.0


def _state_record(state: ShootingState) -> dict[str, Any]:
    return {
        "lambda_d": _number(state.lambda_d),
        "x_sigma_max": _number(state.x_sigma_max),
        "x_start": _number(state.x_start),
        "endpoint_u": _number(state.endpoint_u),
        "endpoint_du_dx": _number(state.endpoint_du_dx),
        "endpoint_log_derivative": _number(state.endpoint_log_derivative),
        "node_count": state.node_count,
        "nfev": state.nfev,
        "physical_outer_radius_fm": _number(state.physical_outer_radius_fm),
    }


def _node_adjacent_negative_log_derivative(
    lambda_with_node: float,
    state_with_node: ShootingState,
    lambda_nodeless: float,
    state_nodeless: ShootingState,
    control: ShootingControl,
    exact: ExactParameters,
) -> tuple[float, ShootingState, int]:
    """Approach the first outer node from the nodeless side without crossing it."""

    if not (lambda_with_node < lambda_nodeless):
        raise FormalSWaveAuditError("invalid node bracket orientation")
    if not (state_with_node.node_count == 1 and state_with_node.endpoint_u < 0.0):
        raise FormalSWaveAuditError("lower node bracket must contain exactly the first endpoint node")
    if not (_nodeless_positive(state_nodeless) and state_nodeless.endpoint_log_derivative > 0.0):
        raise FormalSWaveAuditError("upper node bracket must be nodeless with positive slope")
    low_lambda = lambda_with_node
    high_lambda, high_state = lambda_nodeless, state_nodeless
    for iteration in range(NODE_BRACKET_MAX_ITERATIONS):
        midpoint = 0.5 * (low_lambda + high_lambda)
        state = shoot_zero_energy(midpoint, control, exact)
        if _nodeless_positive(state):
            high_lambda, high_state = midpoint, state
            # As u(R)->0+ while u'(R)<0, L=u'/u must become negative.
            if state.endpoint_log_derivative < 0.0:
                return high_lambda, high_state, iteration + 1
        elif state.node_count == 1 and state.endpoint_u < 0.0:
            low_lambda = midpoint
        else:
            raise FormalSWaveAuditError("node refinement left the first-node branch")
    raise FormalSWaveAuditError("could not obtain a nodeless negative-log-derivative bracket")


def locate_first_zero_energy_threshold(control: ShootingControl, exact: ExactParameters) -> dict[str, Any]:
    """Find the largest-lambda, nodeless zero-energy S-wave threshold.

    The search begins at a rationally certified Bargmann no-binding point and
    descends geometrically.  A *detected* jump across more than the first
    node fails closed rather than being relabelled as an excited-state
    threshold.  Node detection, bounded integration steps, and the separate
    finite-difference check are numerical controls, not an interval-ODE
    certificate.
    """

    _, bargmann_upper = lambda_bargmann_bounds(exact)
    current_lambda = float(bargmann_upper)
    current_state = shoot_zero_energy(current_lambda, control, exact)
    if not (_nodeless_positive(current_state) and current_state.endpoint_log_derivative > 0.0):
        raise FormalSWaveAuditError("Bargmann-certified upper search state is not nodeless and slope-positive")
    search_iterations = 0
    node_refinement_iterations = 0
    lower_lambda: float | None = None
    lower_state: ShootingState | None = None
    upper_lambda: float | None = None
    upper_state: ShootingState | None = None

    for search_iterations in range(1, SEARCH_MAX_ITERATIONS + 1):
        candidate_lambda = current_lambda * SEARCH_RATIO
        candidate_state = shoot_zero_energy(candidate_lambda, control, exact)
        if _nodeless_positive(candidate_state):
            if candidate_state.endpoint_log_derivative > 0.0:
                current_lambda, current_state = candidate_lambda, candidate_state
                continue
            if candidate_state.endpoint_log_derivative < 0.0:
                lower_lambda, lower_state = candidate_lambda, candidate_state
                upper_lambda, upper_state = current_lambda, current_state
                break
            raise FormalSWaveAuditError("search landed exactly at an unresolved zero log derivative")
        if candidate_state.node_count == 1 and candidate_state.endpoint_u < 0.0:
            lower_lambda, lower_state, node_refinement_iterations = _node_adjacent_negative_log_derivative(
                candidate_lambda,
                candidate_state,
                current_lambda,
                current_state,
                control,
                exact,
            )
            upper_lambda, upper_state = current_lambda, current_state
            break
        raise FormalSWaveAuditError("geometric search crossed more than the first S-wave node or lost regularity")
    else:  # pragma: no cover - the live action is asserted in focused tests.
        raise FormalSWaveAuditError("could not bracket the first S-wave threshold")

    if lower_lambda is None or lower_state is None or upper_lambda is None or upper_state is None:
        raise FormalSWaveAuditError("incomplete first-threshold bracket")
    if not (
        lower_lambda < upper_lambda
        and _nodeless_positive(lower_state)
        and _nodeless_positive(upper_state)
        and lower_state.endpoint_log_derivative < 0.0
        and upper_state.endpoint_log_derivative > 0.0
    ):
        raise FormalSWaveAuditError("invalid nodeless log-derivative threshold bracket")

    root_iterations = 0
    for root_iterations in range(1, ROOT_MAX_ITERATIONS + 1):
        midpoint = 0.5 * (lower_lambda + upper_lambda)
        state = shoot_zero_energy(midpoint, control, exact)
        if not _nodeless_positive(state):
            raise FormalSWaveAuditError("root refinement left the nodeless first-threshold branch")
        if state.endpoint_log_derivative > 0.0:
            upper_lambda, upper_state = midpoint, state
        elif state.endpoint_log_derivative < 0.0:
            lower_lambda, lower_state = midpoint, state
        else:
            # This is an exceptionally lucky machine-zero.  Keep a symmetric
            # bracket one bisection level wide instead of serializing an exact
            # numerical equality as physics.
            half_width = max(abs(midpoint) * ROOT_RELATIVE_TOLERANCE / 4.0, sys.float_info.epsilon)
            lower_lambda = midpoint - half_width
            upper_lambda = midpoint + half_width
            lower_state = shoot_zero_energy(lower_lambda, control, exact)
            upper_state = shoot_zero_energy(upper_lambda, control, exact)
            if not (
                _nodeless_positive(lower_state)
                and _nodeless_positive(upper_state)
                and lower_state.endpoint_log_derivative < 0.0
                and upper_state.endpoint_log_derivative > 0.0
            ):
                raise FormalSWaveAuditError("machine-zero fallback could not restore a sign bracket")
            break
        relative_width = (upper_lambda - lower_lambda) / (0.5 * (upper_lambda + lower_lambda))
        if relative_width <= ROOT_RELATIVE_TOLERANCE:
            break
    else:  # pragma: no cover - guarded by the fixed controls.
        raise FormalSWaveAuditError("zero-energy threshold bisection did not converge")

    midpoint = 0.5 * (lower_lambda + upper_lambda)
    midpoint_state = shoot_zero_energy(midpoint, control, exact)
    if not _nodeless_positive(midpoint_state):
        raise FormalSWaveAuditError("threshold midpoint is not nodeless")
    return {
        "control": {
            "identifier": control.identifier,
            "coordinate": "x_sigma=m_sigma(lambda_d)*r_nat=m_sigma*r_fm/(hbar*c)",
            "x_sigma_max": _number(control.x_sigma_max),
            "x_sigma_min": _number(control.x_start),
            "rtol": _number(control.rtol),
            "atol": _number(control.atol),
            "max_step": _number(control.max_step),
            "outer_control_has_at_least_30_scalar_ranges": bool(control.x_sigma_max >= 30.0),
        },
        "lambda_bracket": {
            "lower_negative_log_derivative": _number(lower_lambda),
            "upper_positive_log_derivative": _number(upper_lambda),
            "relative_width": _number((upper_lambda - lower_lambda) / (0.5 * (upper_lambda + lower_lambda))),
        },
        "midpoint_lambda_display": _number(midpoint),
        "lower_state": _state_record(lower_state),
        "upper_state": _state_record(upper_state),
        "midpoint_state": _state_record(midpoint_state),
        "search_iterations": search_iterations,
        "node_refinement_iterations": node_refinement_iterations,
        "root_iterations": root_iterations,
        "first_threshold_nodeless": True,
    }


def _float_from_record(record: Mapping[str, Any], key: str) -> float:
    try:
        return float(record[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalSWaveAuditError(f"malformed numeric record {key}") from exc


def _decimal_enclosure(lower: float, upper: float, places: int = 7) -> dict[str, str]:
    """Give a deliberately rounded enclosing interval, never a fitted value."""

    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
        raise FormalSWaveAuditError("invalid numerical threshold interval")
    scale = 10**places
    outer_lower = math.floor(lower * scale) / scale
    outer_upper = math.ceil(upper * scale) / scale
    if not (outer_lower < lower < upper < outer_upper):
        raise FormalSWaveAuditError("could not form a strict rounded threshold enclosure")
    return {
        "strict_lower": format(outer_lower, f".{places}f"),
        "strict_upper": format(outer_upper, f".{places}f"),
        "statement": f"{format(outer_lower, f'.{places}f')} < lambda_c_formal < {format(outer_upper, f'.{places}f')}",
    }


def _finite_difference_inertia(lambda_d: float, control: FiniteDifferenceControl, exact: ExactParameters) -> dict[str, Any]:
    """Count negative Dirichlet finite-difference eigenvalues by LDL inertia.

    This does not solve the same shooting IVP.  The tridiagonal matrix is the
    independently discretized operator -d^2/dz^2+U(z) on 0<z<z_max with
    Dirichlet walls.  The returned sign is a finite-box control only.
    """

    lam = _finite_float(lambda_d, "finite-difference lambda_d")
    if lam <= 0.0:
        raise FormalSWaveAuditError("finite-difference lambda_d must be positive")
    intervals = int(round(control.z_omega_max / control.dz))
    if intervals < 4 or abs(intervals * control.dz - control.z_omega_max) > 1.0e-12:
        raise FormalSWaveAuditError("finite-difference controls must form an exact declared grid")
    n_interior = intervals - 1
    step = control.z_omega_max / intervals
    inverse_step_squared = 1.0 / (step * step)
    off_diagonal_squared = inverse_step_squared * inverse_step_squared
    pivot: float | None = None
    negative_count = 0
    minimum_abs_pivot = math.inf
    maximum_abs_pivot = 0.0
    for index in range(1, n_interior + 1):
        z_omega = index * step
        diagonal = 2.0 * inverse_step_squared + _omega_coordinate_potential(z_omega, lam, exact)
        if pivot is None:
            pivot = diagonal
        else:
            if pivot == 0.0 or not math.isfinite(pivot):
                raise FormalSWaveAuditError("singular finite-difference LDL pivot")
            pivot = diagonal - off_diagonal_squared / pivot
        if not math.isfinite(pivot):
            raise FormalSWaveAuditError("non-finite finite-difference LDL pivot")
        minimum_abs_pivot = min(minimum_abs_pivot, abs(pivot))
        maximum_abs_pivot = max(maximum_abs_pivot, abs(pivot))
        if pivot < 0.0:
            negative_count += 1
    relative_pivot = minimum_abs_pivot / max(1.0, maximum_abs_pivot)
    if relative_pivot < PIVOT_RELATIVE_FLOOR:
        raise FormalSWaveAuditError("finite-difference inertia is too close to a singular pivot")
    return {
        "lambda_d": _number(lam),
        "negative_eigenvalue_count": negative_count,
        "matrix_dimension": n_interior,
        "z_omega_max": _number(control.z_omega_max),
        "dz": _number(step),
        "physical_outer_radius_fm": _number(control.z_omega_max * float(exact.hbar_c) / float(exact.m_omega)),
        "minimum_abs_ldl_pivot": _number(minimum_abs_pivot),
        "relative_minimum_abs_ldl_pivot": _number(relative_pivot),
        "finite_box_interpretation": (
            "negative count means this declared Dirichlet operator has that many E<0 eigenvalues; "
            "zero count is not an infinite-volume no-binding proof"
        ),
    }


def _finite_difference_sign_controls(fine_midpoint: float, exact: ExactParameters) -> dict[str, Any]:
    lambda_below = 0.9 * fine_midpoint
    lambda_above = 1.1 * fine_midpoint
    if not (0.0 < lambda_below < fine_midpoint < lambda_above):
        raise FormalSWaveAuditError("invalid finite-difference sign offsets")
    rows = []
    for control in FINITE_DIFFERENCE_CONTROLS:
        below = _finite_difference_inertia(lambda_below, control, exact)
        above = _finite_difference_inertia(lambda_above, control, exact)
        if below["negative_eigenvalue_count"] != 1 or above["negative_eigenvalue_count"] != 0:
            raise FormalSWaveAuditError("finite-difference sign control did not separate the first threshold")
        rows.append(
            {
                "identifier": control.identifier,
                "coordinate": "z_omega=m_omega*r_nat=m_omega*r_fm/(hbar*c)",
                "below_shooting_midpoint": below,
                "above_shooting_midpoint": above,
                "sign_control_pass": True,
            }
        )
    return {
        "method": "Dirichlet finite-difference tridiagonal LDL inertia at E=0",
        "lambda_below_shooting_midpoint": _number(lambda_below),
        "lambda_above_shooting_midpoint": _number(lambda_above),
        "controls_are_operator_not_droplet_boundaries": True,
        "rows": rows,
        "interpretation": (
            "The stable negative/zero finite-box inertia signs independently agree with a first threshold between "
            "the two deliberately separated formal lambdas. They do not replace the infinite-volume shooting definition."
        ),
    }


def _variational_witness(exact: ExactParameters) -> dict[str, Any]:
    """Evaluate one declared exponential S-wave trial state without optimization."""

    q0 = lambda_q0_attraction(exact)
    witness_lambda = q0 / WITNESS_LAMBDA_DIVISOR
    with mp.workdps(100):
        W0 = _mp_fraction(exact.W0)
        M_N = _mp_fraction(exact.M_N)
        m_omega = _mp_fraction(exact.m_omega)
        g_omega = _mp_fraction(exact.g_omega)
        g_s = _mp_fraction(exact.g_s)
        mu = M_N / 2
        m_sigma = mp.sqrt(2 * _mp_fraction(witness_lambda) * W0**2)
        beta = _mp_fraction(WITNESS_BETA_OVER_M_SIGMA) * m_sigma
        alpha_omega = g_omega**2 / (4 * mp.pi)
        alpha_sigma = g_s**2 / (4 * mp.pi)
        rayleigh_mev = beta**2 / (2 * mu) + 4 * beta**3 * (
            alpha_omega / (2 * beta + m_omega) ** 2 - alpha_sigma / (2 * beta + m_sigma) ** 2
        )
    if not rayleigh_mev < mp.mpf("-1e-6"):
        raise FormalSWaveAuditError("declared formal variational witness is not safely negative")
    return {
        "trial_state": "psi_beta(r)=(beta^3/pi)^(1/2)*exp(-beta*r), normalized in natural units",
        "lambda_witness": _fraction_record(witness_lambda, "1", "lambda_q0/10; predeclared non-fitted control"),
        "beta_MeV_display": _number(beta),
        "beta_definition": "(2/5)*m_sigma(lambda_witness)",
        "rayleigh_energy_MeV_display": _number(rayleigh_mev),
        "status": "NEGATIVE_DECLARED_FORMAL_TRIAL_WITNESS",
        "scope": (
            "Within this formal self-adjoint central Hamiltonian, a negative trial Rayleigh quotient witnesses at least "
            "one E<0 S-wave state. It is not a physical NN/deuteron claim."
        ),
    }


def _provenance() -> dict[str, str]:
    paths = {
        "action": ACTION_PATH,
        "source_complete": SOURCE_PATH,
        "nn_static_audit": NN_AUDIT_PATH,
        "cross_scale_gate": CROSS_SCALE_PATH,
        "producer": Path(__file__),
    }
    return {f"{label}_path": str(path.relative_to(REPO_ROOT)) for label, path in paths.items()} | {
        f"{label}_sha256": _sha256(path) for label, path in paths.items()
    }


def _action_records(exact: ExactParameters) -> dict[str, Any]:
    return {
        "live_decimal_inputs": {
            "lambda_baseline": _fraction_record(exact.lambda_baseline, "1", "live Params.lam"),
            "M_N_MeV": _fraction_record(exact.M_N, "MeV", "live Params.M_N"),
            "W0_MeV": _fraction_record(exact.W0, "MeV", "live Params.W0"),
            "m_omega_MeV": _fraction_record(exact.m_omega, "MeV", "live Params.m_omega"),
            "g_omega": _fraction_record(exact.g_omega, "1", "live Params.g_omega; nucleon-current vertex"),
            "hbar_c_MeV_fm": _fraction_record(exact.hbar_c, "MeV fm", "live Params.hbar_c"),
        },
        "derived_action_relations": {
            "g_s_M_N_over_W0": _fraction_record(exact.g_s, "1", "M_N/W0"),
            "q_phi_momega_over_W0": _fraction_record(exact.q_phi, "1", "m_omega/W0"),
            "reduced_mass_M_N_over_2": _fraction_record(exact.reduced_mass, "MeV", "fixed formal two-body reduced mass"),
            "m_sigma_deformation": "m_sigma(lambda_d)^2=2*lambda_d*W0^2",
            "static_potential": (
                "V_lambda_d(r_nat)=[g_omega^2 exp(-m_omega r_nat)-g_s^2 exp(-m_sigma(lambda_d) r_nat)]/(4*pi*r_nat)"
            ),
        },
    }


def _analytic_formal_certificates(exact: ExactParameters, witness: Mapping[str, Any]) -> dict[str, Any]:
    q0 = lambda_q0_attraction(exact)
    coordinate_range = lambda_coordinate_range(exact)
    bargmann_lower, bargmann_upper = lambda_bargmann_bounds(exact)
    bargmann_display = lambda_bargmann_display(exact)
    if not (q0 < bargmann_lower < bargmann_upper < coordinate_range < exact.lambda_baseline):
        raise FormalSWaveAuditError("expected live formal threshold ordering is not certified")
    return {
        "monotonicity": {
            "pointwise_statement": (
                "For lambda_2>lambda_1>0 and r>0, V_lambda_2(r)-V_lambda_1(r)>0 because "
                "m_sigma(lambda) increases and only the negative scalar Yukawa term changes."
            ),
            "min_max_consequence": (
                "Every discrete eigenvalue of this formal central Hamiltonian is nondecreasing with lambda_d; "
                "the first S-wave bound-state region, if present, is an interval below its first threshold."
            ),
        },
        "bargmann_comparator": {
            "comparator": "V_comp(r)=-alpha_s exp(-m_sigma r)/r, alpha_s=g_s^2/(4*pi)",
            "sufficient_no_binding_condition": "2*mu*alpha_s/m_sigma(lambda_d)<1",
            "lambda_B_exact_pi_display": _number(bargmann_display, 32),
            "lambda_B_rational_bounds": {
                "strict_lower_from_pi_lt_22_over_7": _positive_fraction_record(
                    bargmann_lower, "1", "M_N^6/[32*(22/7)^2*W0^6] < lambda_B"
                ),
                "strict_upper_from_pi_gt_3": _positive_fraction_record(
                    bargmann_upper, "1", "lambda_B < M_N^6/(288*W0^6)"
                ),
            },
            "strict_no_binding_for_lambda_d_at_least_rational_upper": True,
            "why": (
                "The full formal potential is pointwise no more attractive than V_comp; Bargmann N_0<1 for the "
                "comparator excludes an S-wave bound state there as well."
            ),
        },
        "declared_negative_variational_witness": dict(witness),
        "cross_scale_thresholds": {
            "lambda_q0": _fraction_record(q0, "1", "tree/static q=0 zero-kernel threshold"),
            "lambda_range": _fraction_record(
                coordinate_range, "1", "m_omega^2/(2*W0^2), tree/static coordinate range-sign threshold"
            ),
            "certified_analytic_ordering": "lambda_witness < lambda_q0 < lambda_B < lambda_range < lambda_baseline",
            "q0_strictly_below_Bargmann_lower_bound": bool(q0 < bargmann_lower),
            "baseline_above_coordinate_range": bool(exact.lambda_baseline > coordinate_range),
        },
    }


def _initial_start_replay(
    fine_threshold: Mapping[str, Any],
    exact: ExactParameters,
) -> list[dict[str, Any]]:
    fine_control = SHOOTING_CONTROLS[-1]
    bracket = fine_threshold["lambda_bracket"]
    lower = _float_from_record(bracket, "lower_negative_log_derivative")
    upper = _float_from_record(bracket, "upper_positive_log_derivative")
    rows = []
    for x_start in INITIAL_START_REPLAYS:
        replay_control = replace(fine_control, x_start=x_start)
        lower_state = shoot_zero_energy(lower, replay_control, exact)
        upper_state = shoot_zero_energy(upper, replay_control, exact)
        if not (
            _nodeless_positive(lower_state)
            and _nodeless_positive(upper_state)
            and lower_state.endpoint_log_derivative < 0.0
            and upper_state.endpoint_log_derivative > 0.0
        ):
            raise FormalSWaveAuditError("origin-series replay does not preserve the fine threshold sign bracket")
        rows.append(
            {
                "x_start": _number(x_start),
                "lower_log_derivative": _number(lower_state.endpoint_log_derivative),
                "upper_log_derivative": _number(upper_state.endpoint_log_derivative),
                "nodes_lower_and_upper": [lower_state.node_count, upper_state.node_count],
                "sign_bracket_preserved": True,
            }
        )
    return rows


def _threshold_diagnostics(fine_threshold: Mapping[str, Any], exact: ExactParameters) -> dict[str, Any]:
    midpoint = _float_from_record(fine_threshold, "midpoint_lambda_display")
    q0 = lambda_q0_attraction(exact)
    m_sigma = scalar_mass_mev(midpoint, exact)
    scalar_range = float(exact.hbar_c) / m_sigma
    coupling_ratio = float(exact.g_omega * exact.g_omega / (exact.g_s * exact.g_s))
    if not coupling_ratio > 1.0 or not float(exact.m_omega) > m_sigma:
        raise FormalSWaveAuditError("threshold diagnostics lost the static coordinate-tail domain")
    tail_crossing = float(exact.hbar_c) * math.log(coupling_ratio) / (float(exact.m_omega) - m_sigma)
    lower = _float_from_record(fine_threshold["lambda_bracket"], "lower_negative_log_derivative")
    upper = _float_from_record(fine_threshold["lambda_bracket"], "upper_positive_log_derivative")
    return {
        "m_sigma_at_lambda_c_MeV_display": _number(m_sigma),
        "scalar_range_hbarc_over_msigma_fm_display": _number(scalar_range),
        "tree_static_coordinate_tail_crossover_fm_display_not_prediction": _number(tail_crossing),
        "lambda_c_over_lambda_q0_display": _number(midpoint / float(q0)),
        "lambda_baseline_over_lambda_c_display": _number(float(exact.lambda_baseline) / midpoint),
        "rounded_numerical_enclosure": _decimal_enclosure(lower, upper),
    }


def _fresh_result_payload(p: Params = PARAMS) -> dict[str, Any]:
    _require_live_baseline(p)
    exact = exact_parameters_from_live(p)
    witness = _variational_witness(exact)
    analytic = _analytic_formal_certificates(exact, witness)
    shooting = [locate_first_zero_energy_threshold(control, exact) for control in SHOOTING_CONTROLS]
    fine = shooting[-1]
    roots = [_float_from_record(row, "midpoint_lambda_display") for row in shooting]
    relative_spread = (max(roots) - min(roots)) / roots[-1]
    if relative_spread > CONVERGENCE_RELATIVE_LIMIT:
        raise FormalSWaveAuditError("shooting box/tolerance controls did not converge")
    initial_replay = _initial_start_replay(fine, exact)
    finite_difference = _finite_difference_sign_controls(roots[-1], exact)
    diagnostics = _threshold_diagnostics(fine, exact)
    coordinate_range = lambda_coordinate_range(exact)
    q0 = lambda_q0_attraction(exact)
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "runtime_control_surface": _runtime_control_surface(),
        "provenance": _provenance(),
        "action_inputs": _action_records(exact),
        "formal_model_contract": {
            "symbolic_deformation": "lambda -> lambda_d>0 only in U=lambda(W^2-W0^2)^2/4 and m_sigma^2=2*lambda_d*W0^2",
            "held_fixed": ["W0", "M_N", "m_omega", "g_omega", "g_s=M_N/W0", "q_phi=m_omega/W0", "mu=M_N/2", "all omitted sectors"],
            "parameter_fit_performed": False,
            "no_new_operator_introduced": True,
            "hypothetical_deformed_action_member": True,
            "physical_action_replacement_claimed": False,
            "empirical_target_or_data_read": False,
            "physical_claim_permitted": False,
            "scope": "formal static tree-level central S-wave Hamiltonian only",
        },
        "analytic_formal_certificates": analytic,
        "zero_energy_shooting": {
            "equation": "[-d^2/dx_sigma^2+U_lambda_d(x_sigma)]u=0, u(0)=0, u'(0)=1",
            "threshold_definition": "first nodeless zero-energy threshold: u'(infinity)=0; at threshold it is a zero-energy resonance, not a normalizable bound state",
            "outer_coordinate": "x_sigma=m_sigma(lambda_d)*r_fm/(hbar*c)",
            "controls": shooting,
            "relative_spread_across_declared_box_tolerance_controls": _number(relative_spread),
            "relative_spread_limit": _number(CONVERGENCE_RELATIVE_LIMIT),
            "box_tolerance_convergence_pass": True,
            "origin_series_replays": initial_replay,
            "fine_control_identifier": fine["control"]["identifier"],
            "first_threshold_nodeless": True,
        },
        "finite_difference_sign_controls": finite_difference,
        "formal_threshold_diagnostics": diagnostics,
        "baseline_classification": {
            "lambda_baseline": _fraction_record(exact.lambda_baseline, "1", "live Params.lam"),
            "lambda_range": _fraction_record(coordinate_range, "1", "m_sigma=m_omega"),
            "lambda_q0": _fraction_record(q0, "1", "tree/static q=0 zero-kernel threshold"),
            "status": "FORMAL_BASELINE_POINTWISE_STATIC_REPULSIVE_AND_ABOVE_FIRST_THRESHOLD",
            "reason": "lambda_baseline>lambda_range and lambda_baseline>lambda_c_formal in this formal reduction",
        },
        "conclusion": {
            "formal_only": (
                "The declared formal central Hamiltonian has a controlled numerical first S-wave zero-energy threshold "
                "near lambda_d=0.0011028573. Its bound-state side is lambda_d below that threshold, by formal monotonicity."
            ),
            "cross_scale_interpretation": (
                "The threshold lies below lambda_q0, so a negative static q=0 sign or a coordinate-space negative tail is "
                "not sufficient to create an S-wave bound state in this reduction."
            ),
            "not_claimed": [
                "a physical NN, deuteron, nuclear, or QCD prediction",
                "a fitted or preferred replacement for the source-action lambda",
                "a new interaction, energy source, or completed physical theory",
                "a controlled loop, relativistic, spin, isospin, pion/rho, form-factor, Coulomb, or experimental result",
            ],
        },
        "caveats": [
            "The numerical threshold is controlled shooting plus finite-difference agreement, not an interval-ODE proof of an infinite-volume eigenvalue.",
            "Finite Dirichlet boxes are operator controls only: a positive finite-box spectrum does not prove infinite-volume nonbinding.",
            "The formal potential is tree/static and spinless; the source action does not make its loop expansion a controlled precision NN approximation.",
            "The zero-energy threshold is a resonance condition, not itself a normalizable bound-state wavefunction.",
            "No empirical NN/nuclear binding energy, scattering datum, radius, cutoff, form factor, likelihood, or fit is used.",
        ],
    }


def _runtime_control_surface() -> dict[str, Any]:
    """Serialize every mutable numerical control that can affect a result.

    The result cache is an optimization only.  It must never make a runtime
    mutation of a declared tolerance, search rule, or witness look like the
    original calculation during ``validate_result``.
    """

    return {
        "schema": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "pi_bounds": [str(PI_LOWER), str(PI_UPPER)],
        "shooting_controls": [
            {
                "identifier": control.identifier,
                "x_sigma_max": repr(control.x_sigma_max),
                "rtol": repr(control.rtol),
                "atol": repr(control.atol),
                "x_start": repr(control.x_start),
                "max_step": repr(control.max_step),
            }
            for control in SHOOTING_CONTROLS
        ],
        "finite_difference_controls": [
            {
                "identifier": control.identifier,
                "z_omega_max": repr(control.z_omega_max),
                "dz": repr(control.dz),
            }
            for control in FINITE_DIFFERENCE_CONTROLS
        ],
        "search": {
            "ratio": repr(SEARCH_RATIO),
            "node_bracket_max_iterations": NODE_BRACKET_MAX_ITERATIONS,
            "root_max_iterations": ROOT_MAX_ITERATIONS,
            "search_max_iterations": SEARCH_MAX_ITERATIONS,
            "root_relative_tolerance": repr(ROOT_RELATIVE_TOLERANCE),
            "convergence_relative_limit": repr(CONVERGENCE_RELATIVE_LIMIT),
            "pivot_relative_floor": repr(PIVOT_RELATIVE_FLOOR),
            "initial_start_replays": [repr(value) for value in INITIAL_START_REPLAYS],
        },
        "witness": {
            "lambda_divisor": WITNESS_LAMBDA_DIVISOR,
            "beta_over_m_sigma": str(WITNESS_BETA_OVER_M_SIGMA),
        },
    }


def _default_cache_key() -> str:
    exact = exact_parameters_from_live(PARAMS)
    payload = {
        "inputs": [str(exact.W0), str(exact.lambda_baseline), str(exact.M_N), str(exact.m_omega), str(exact.g_omega), str(exact.hbar_c)],
        "runtime_control_surface": _runtime_control_surface(),
        "provenance": _provenance(),
    }
    return _canonical_json(payload).decode("utf-8")


def _expected_payload(p: Params = PARAMS) -> dict[str, Any]:
    """Build once per unchanged live source surface; return an isolated copy."""

    global _DEFAULT_CACHE
    _require_live_baseline(p)
    key = _default_cache_key()
    if _DEFAULT_CACHE is not None and _DEFAULT_CACHE[0] == key:
        return copy.deepcopy(_DEFAULT_CACHE[1])
    payload = _fresh_result_payload(p)
    _DEFAULT_CACHE = (key, copy.deepcopy(payload))
    return payload


def build_result(p: Params = PARAMS) -> dict[str, Any]:
    """Return a fresh-or-cached canonical derivation for the unchanged live action."""

    return _expected_payload(p)


def validate_result(payload: Any, p: Params = PARAMS, *, check_provenance: bool = True) -> None:
    """Accept only an exact regenerated source-action derivation."""

    if not isinstance(payload, Mapping):
        raise FormalSWaveAuditError("result payload must be an object")
    expected = _expected_payload(p)
    if _canonical_json(payload) != _canonical_json(expected):
        raise FormalSWaveAuditError("payload is not an exact fresh formal source-action derivation")
    if check_provenance and payload.get("provenance") != _provenance():
        raise FormalSWaveAuditError("provenance changed")


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
        except (OSError, json.JSONDecodeError, FormalSWaveAuditError) as exc:
            print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
            return 1
        print("VALIDATION_PASS")
        return 0
    try:
        result = build_result()
        write_result(result, args.output)
    except FormalSWaveAuditError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 1
    print(_canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
