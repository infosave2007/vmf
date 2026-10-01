#!/usr/bin/env python3
"""Fail-closed finite-droplet audit for the unmodified source-complete Q4 action.

Scope is deliberately narrow: a static, no-current, zero-temperature
Thomas--Fermi (TF) gas with one d=4 baryon species and the action's full
``M*(W)=M_N W/W0`` scalar response (``s=1``).  It adds no Coulomb,
isovector, W8, density-dependent, fitted, gravitational, shell, or pairing
sector.

The central result is an analytic variational inequality, not a scan-derived
claim.  The accompanying finite radial calculations only independently check
that a nonlocal Gauss solve, positive field energy, and the declared
box/grid/profile controls behave as expected.  Those deterministic profiles
are *not* stationary solutions and are never promoted to physical droplets.
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
from typing import Any, Iterable, Mapping

try:  # Direct execution from ``verification``.
    from source_complete_solution_audit import PARAMS, Params
except ImportError:  # pragma: no cover - package-style execution support.
    from .source_complete_solution_audit import PARAMS, Params


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "nvg_source_complete_q4_droplet_results.json"
SCHEMA = "nvg_source_complete_q4_droplet_audit.v1"
STATUS = "PASS_SOURCE_COMPLETE_Q4_TF_SELF_BINDING_NO_GO"

# These are numerical controls, not calibration values or inferred nuclear
# radii.  The ladder is intentionally small enough to be independently
# regenerated in a focused unit test.
# Synthetic geometric controls, deliberately not isotope labels.
N_B_LADDER = (1, 8, 64, 512)
BOXES_FM = (18.0, 24.0, 30.0)
GRID_CELLS_PER_FM = (("coarse", 24), ("fine", 36))
GRID_RELATIVE_LIMIT = 2.0e-3
BOX_RELATIVE_LIMIT = 2.0e-3
GAUSS_RELATIVE_LIMIT = 1.0e-12

# ``radius_factor`` and ``scalar_depression`` specify deterministic starting
# profiles for quadrature/Gauss controls only.  They never enter the analytic
# no-go and are not chosen to match any nucleus or binding datum.
PROFILE_CONTROLS = (
    ("compact_control", 0.65, 0.45),
    ("reference_control", 1.00, 0.20),
    ("broad_control", 1.35, 0.05),
)
# Dimensionless radius scale in x=W0*r/(hbar*c), not a nuclear-density or
# empirical radius input.  Every profile radius is this number times N^(1/3).
PROFILE_RADIUS_X_PER_N_CUBERT = 2.5

# This is the sharp threshold for the Jensen certificate implemented below.
C_CRITICAL_FRACTION = Fraction(1, 4)
C_CRITICAL = float(C_CRITICAL_FRACTION)


class Q4DropletAuditError(ValueError):
    """Raised for non-finite or contract-breaking numerical states."""


@dataclass(frozen=True)
class RadialGrid:
    """Interior-node radial grid in x=W0 r/(hbar c)."""

    box_fm: float
    cells_per_fm: int
    x_max: float
    intervals: int
    dx: float
    x: tuple[float, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class ProfileControl:
    """A declared profile used only as a numerical adversarial control."""

    identifier: str
    radius_factor: float
    scalar_depression: float


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _as_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Q4DropletAuditError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise Q4DropletAuditError(f"{label} is not finite")
    return number


def action_inputs(p: Params = PARAMS) -> dict[str, float | int]:
    """Return exactly the action-derived inputs consumed by this audit.

    The parent module is intentionally used only for its frozen ``Params``
    object.  No result JSON, inverse W8 design, fit, or legacy density table
    is imported.
    """

    fields: dict[str, float | int] = {
        "W0_MeV": _as_float(p.W0, "W0"),
        "lambda": _as_float(p.lam, "lambda"),
        "M_N_MeV": _as_float(p.M_N, "M_N"),
        "m_omega_MeV": _as_float(p.m_omega, "m_omega"),
        "g_omega": _as_float(p.g_omega, "g_omega"),
        "hbar_c_MeV_fm": _as_float(p.hbar_c, "hbar_c"),
        "d": int(p.degeneracy),
    }
    if fields["W0_MeV"] <= 0.0 or fields["lambda"] <= 0.0 or fields["M_N_MeV"] <= 0.0:
        raise Q4DropletAuditError("source-complete scalar inputs must be positive")
    if fields["m_omega_MeV"] <= 0.0 or fields["g_omega"] <= 0.0 or fields["hbar_c_MeV_fm"] <= 0.0:
        raise Q4DropletAuditError("source-complete vector/unit inputs must be positive")
    if fields["d"] != 4:
        raise Q4DropletAuditError("this audit is frozen to a d=4 baryon gas")
    fields["g_s_M_N_over_W0"] = fields["M_N_MeV"] / fields["W0_MeV"]
    fields["q_phi_momega_over_W0"] = fields["m_omega_MeV"] / fields["W0_MeV"]
    return fields


def scalar_coercivity_constant(p: Params = PARAMS) -> float:
    """Return C=3*pi^2 lambda W0^4/(8 M_N^4) from the unmodified action."""

    values = action_inputs(p)
    return (
        3.0
        * math.pi**2
        * float(values["lambda"])
        * float(values["W0_MeV"]) ** 4
        / (8.0 * float(values["M_N_MeV"]) ** 4)
    )


def scalar_coercivity_rational_certificate(p: Params = PARAMS) -> dict[str, Any]:
    """Strict exact-rational certificate for C>1/4 using only pi>3.

    Decimal literals from the action are parsed as ``Fraction``.  Since
    pi>3, the exact action coefficient obeys

       C=3*pi^2*lambda*W0^4/(8*M_N^4)
        > 27*lambda*W0^4/(8*M_N^4).

    Thus a successful rational comparison is stronger than a binary64 or
    sampled numerical check.  If it fails, the audit reports the sharp
    Jensen certificate as inconclusive; it never substitutes a fit.
    """

    values = action_inputs(p)
    W0 = Fraction(str(values["W0_MeV"]))
    mass = Fraction(str(values["M_N_MeV"]))
    lam = Fraction(str(values["lambda"]))
    lower = Fraction(27, 8) * lam * W0**4 / mass**4
    return {
        "decimal_input_interpretation": {
            "W0": str(values["W0_MeV"]),
            "M_N": str(values["M_N_MeV"]),
            "lambda": str(values["lambda"]),
        },
        "strict_inequality_used": "pi > 3",
        "C_strict_lower_bound_formula": "27*lambda*W0^4/(8*M_N^4)",
        "C_strict_lower_bound_rational": {
            "numerator": str(lower.numerator),
            "denominator": str(lower.denominator),
            "binary64_display_only": float(lower),
        },
        "C_critical_rational": {
            "numerator": str(C_CRITICAL_FRACTION.numerator),
            "denominator": str(C_CRITICAL_FRACTION.denominator),
        },
        "strict_lower_C_exceeds_C_critical": bool(lower > C_CRITICAL_FRACTION),
    }


def fermi_energy_per_baryon_over_M(y: float, z: float) -> float:
    """Exact zero-T d=4 Fermi energy per baryon divided by M_N.

    ``y=W/W0 >= 0`` and ``z=k_F/M_N > 0``.  The d=4 degeneracy cancels in
    this per-particle expression; it remains essential in the n<->kF map.
    A small-ratio series avoids cancellation in the closed form.
    """

    if y < 0.0 or z <= 0.0 or not math.isfinite(y) or not math.isfinite(z):
        raise Q4DropletAuditError("Fermi ratio requires finite y>=0 and z>0")
    if y == 0.0:
        return 0.75 * z
    ratio = z / y
    if ratio <= 1.0e-2:
        r2 = ratio * ratio
        # 3 int_0^1 u^2 sqrt(1+r^2 u^2) du through O(r^8).
        return y * (
            1.0
            + 3.0 * r2 / 10.0
            - 3.0 * r2 * r2 / 56.0
            + r2**3 / 48.0
            - 15.0 * r2**4 / 1408.0
        )
    energy = math.sqrt(y * y + z * z)
    bracket = z * energy * (2.0 * z * z + y * y) - y**4 * math.asinh(ratio)
    return 3.0 * bracket / (8.0 * z**3)


def potential_per_baryon_over_M(y: float, z: float, C: float) -> float:
    """U/(M_N n) for U=lambda W0^4 (y^2-1)^2/4."""

    if C < 0.0 or y < 0.0 or z <= 0.0:
        raise Q4DropletAuditError("invalid potential-ratio domain")
    return C * (y * y - 1.0) ** 2 / z**3


def jensen_fermi_lower_bound_over_M(y: float, z: float) -> float:
    """3 int u^2 sqrt(y^2+z^2u^2)du >= sqrt(y^2+9z^2/16)."""

    if y < 0.0 or z <= 0.0:
        raise Q4DropletAuditError("invalid Jensen-bound domain")
    return math.sqrt(y * y + 9.0 * z * z / 16.0)


def sufficient_local_gap_lower_bound(y: float, z: float, C: float) -> float:
    """A nonnegative lower bound on (F_F+U-M_N n)/(M_N n).

    This is the sharp Jensen certificate.  For ``R=sqrt(y^2+9z^2/16)<1``,
    set delta=1-y^2 and r=sqrt((9/16)z^2/delta).  At C=1/4,
    (1-R)/(delta^2/(4z^3)) <= (256/27)r^3(1-r) <= 1, so U/(Mn)>=1-R.
    At C>1/4 the source-complete baseline is strictly above this threshold.
    """

    if C < C_CRITICAL:
        raise Q4DropletAuditError(
            "C<1/4: this sharp sufficient lower-bound certificate is inconclusive"
        )
    kinetic = jensen_fermi_lower_bound_over_M(y, z)
    # If R>=1 the Fermi Jensen term alone is enough.  In the remaining
    # branch the sharp C=1/4 proof supplies the nonnegative lower bound zero.
    return max(0.0, kinetic - 1.0)


def analytic_bound_controls(p: Params = PARAMS) -> dict[str, Any]:
    """Evaluate the sufficient bound and an independent exact-function grid."""

    C = scalar_coercivity_constant(p)
    rational_certificate = scalar_coercivity_rational_certificate(p)
    # This is an algorithmic mesh, not an empirical data table.
    y_values = (0.0,) + tuple(10.0 ** (-6.0 + 6.0 * i / 30.0) for i in range(31)) + tuple(
        1.0 + 0.05 * i for i in range(1, 41)
    )
    z_values = tuple(10.0 ** (-5.0 + 7.0 * i / 100.0) for i in range(101))
    criterion_pass = rational_certificate["strict_lower_C_exceeds_C_critical"]
    worst_exact_gap = math.inf
    worst_sufficient_gap: float | None = math.inf if criterion_pass else None
    worst_excess: float | None = math.inf if criterion_pass else None
    exact_pass = True
    for y in y_values:
        for z in z_values:
            exact_gap = (
                fermi_energy_per_baryon_over_M(y, z)
                + potential_per_baryon_over_M(y, z, C)
                - 1.0
            )
            if exact_gap <= 0.0:
                exact_pass = False
            if criterion_pass:
                sufficient_gap = sufficient_local_gap_lower_bound(y, z, C)
                excess = exact_gap - sufficient_gap
                scale = max(1.0, abs(exact_gap), abs(sufficient_gap))
                if sufficient_gap < -2.0e-14 * scale or excess < -5.0e-13 * scale:
                    exact_pass = False
                assert worst_sufficient_gap is not None and worst_excess is not None
                worst_sufficient_gap = min(worst_sufficient_gap, sufficient_gap)
                worst_excess = min(worst_excess, excess)
            worst_exact_gap = min(worst_exact_gap, exact_gap)
    return {
        "C": C,
        "C_critical": C_CRITICAL,
        "C_minus_C_critical": C - C_CRITICAL,
        "rational_certificate": rational_certificate,
        "criterion_C_strictly_gt_1_over_4": criterion_pass,
        "criterion_is_sufficient_not_fitted": True,
        "proof_domain": "n>0, y=W/W0>=0, z=k_F/M_N>0, d=4, static zero-temperature TF",
        "mesh_is_algorithmic_control_not_input_data": True,
        "y_point_count": len(y_values),
        "z_point_count": len(z_values),
        "sample_count": len(y_values) * len(z_values),
        "minimum_exact_local_gap_over_Mn": worst_exact_gap,
        "sufficient_lower_bound_available": criterion_pass,
        "minimum_sufficient_lower_gap_over_Mn": worst_sufficient_gap,
        "minimum_exact_minus_sufficient_gap_over_Mn": worst_excess,
        "exact_function_control_pass": exact_pass,
        "pass": bool(criterion_pass and exact_pass),
    }


def make_grid(box_fm: float, cells_per_fm: int, p: Params = PARAMS) -> RadialGrid:
    if box_fm <= 0.0 or cells_per_fm < 2:
        raise Q4DropletAuditError("invalid box/grid declaration")
    intervals = int(round(box_fm * cells_per_fm))
    if intervals < 8:
        raise Q4DropletAuditError("too few radial intervals")
    x_max = p.W0 * float(box_fm) / p.hbar_c
    dx = x_max / intervals
    x = tuple((index + 1) * dx for index in range(intervals - 1))
    weights = tuple(4.0 * math.pi * value * value * dx for value in x)
    return RadialGrid(
        box_fm=float(box_fm),
        cells_per_fm=int(cells_per_fm),
        x_max=x_max,
        intervals=intervals,
        dx=dx,
        x=x,
        weights=weights,
    )


def _solve_tridiagonal(lower: list[float], diagonal: list[float], upper: list[float], rhs: list[float]) -> list[float]:
    """Thomas solve for a strictly positive finite-volume Gauss operator."""

    size = len(diagonal)
    if size == 0 or len(rhs) != size or len(lower) != size - 1 or len(upper) != size - 1:
        raise Q4DropletAuditError("malformed tridiagonal Gauss system")
    diag = diagonal.copy()
    out_rhs = rhs.copy()
    out_upper = upper.copy()
    for index in range(1, size):
        if diag[index - 1] <= 0.0 or not math.isfinite(diag[index - 1]):
            raise Q4DropletAuditError("nonpositive Gauss pivot")
        factor = lower[index - 1] / diag[index - 1]
        diag[index] -= factor * out_upper[index - 1]
        out_rhs[index] -= factor * out_rhs[index - 1]
    if diag[-1] <= 0.0 or not math.isfinite(diag[-1]):
        raise Q4DropletAuditError("nonpositive final Gauss pivot")
    solution = [0.0] * size
    solution[-1] = out_rhs[-1] / diag[-1]
    for index in range(size - 2, -1, -1):
        if diag[index] <= 0.0 or not math.isfinite(diag[index]):
            raise Q4DropletAuditError("nonpositive back-substitution pivot")
        solution[index] = (out_rhs[index] - out_upper[index] * solution[index + 1]) / diag[index]
    if not all(math.isfinite(value) for value in solution):
        raise Q4DropletAuditError("non-finite Gauss solution")
    return solution


def _profile_for_control(
    grid: RadialGrid,
    target_N: int,
    control: ProfileControl,
    p: Params = PARAMS,
) -> tuple[list[float], list[float], float]:
    """Generate a normalized, predeclared numerical-control profile.

    It is deliberately not a variational ansatz and is never optimized.  Its
    only jobs are to exercise the finite radial Gauss solve and show that the
    analytic positive-gap inequality survives box/grid/profile changes.
    """

    radius_x = (
        control.radius_factor
        * PROFILE_RADIUS_X_PER_N_CUBERT
        * float(target_N) ** (1.0 / 3.0)
    )
    radius_fm = radius_x * p.hbar_c / p.W0
    shape = []
    for x_value in grid.x:
        shape.append(math.exp(-((x_value / radius_x) ** 4)))
    raw_number = sum(weight * value for weight, value in zip(grid.weights, shape))
    if raw_number <= 0.0 or not math.isfinite(raw_number):
        raise Q4DropletAuditError("profile normalization failed")
    normalization = float(target_N) / raw_number
    density = [normalization * value for value in shape]
    y = [1.0 - control.scalar_depression * value for value in shape]
    if min(y) <= 0.0 or min(density) < 0.0:
        raise Q4DropletAuditError("control profile left the positive TF domain")
    return density, y, radius_fm


def _fermi_density_dimensionless(n_dim: float, y: float, p: Params = PARAMS) -> tuple[float, float, float]:
    """Return (F/W0^4, z=kF/M_N, exact F/(M_N n)) at one positive cell."""

    if n_dim <= 0.0 or y < 0.0:
        raise Q4DropletAuditError("finite control requires n>0 and y>=0")
    k_over_W0 = (6.0 * math.pi**2 * n_dim / p.degeneracy) ** (1.0 / 3.0)
    z = k_over_W0 / p.g_s
    ratio = fermi_energy_per_baryon_over_M(y, z)
    return n_dim * p.g_s * ratio, z, ratio


def _scalar_gradient_dimensionless(grid: RadialGrid, y: list[float]) -> float:
    """4*pi int x^2 (y')^2/2 dx with y'(0)=0 and y(X)=1."""

    if len(y) != len(grid.x):
        raise Q4DropletAuditError("scalar profile/grid size mismatch")
    nodes = [y[0], *y, 1.0]
    total = 0.0
    for index in range(grid.intervals):
        midpoint = (index + 0.5) * grid.dx
        derivative = (nodes[index + 1] - nodes[index]) / grid.dx
        total += 4.0 * math.pi * 0.5 * midpoint * midpoint * derivative * derivative * grid.dx
    return total


def _gauss_solve_and_energy(
    grid: RadialGrid,
    density: list[float],
    y: list[float],
    p: Params = PARAMS,
) -> dict[str, float]:
    """Re-solve (-laplacian+q^2W^2)A=g*n in the declared finite box.

    We use u=x a so the regular radial equation is a positive tridiagonal
    system: (-u''+q^2 y^2 u)=g*x*n, u(0)=u(X)=0.  The matrix identity is the
    discrete Gauss constraint/positive vector-energy identity.
    """

    count = len(grid.x)
    if len(density) != count or len(y) != count:
        raise Q4DropletAuditError("Gauss profile/grid size mismatch")
    inverse_dx2 = 1.0 / (grid.dx * grid.dx)
    diagonal = [2.0 * inverse_dx2 + p.q_phi * p.q_phi * value * value for value in y]
    lower = [-inverse_dx2] * (count - 1)
    upper = [-inverse_dx2] * (count - 1)
    rhs = [p.g_omega * x_value * n_value for x_value, n_value in zip(grid.x, density)]
    u = _solve_tridiagonal(lower, diagonal, upper, rhs)
    a = [u_value / x_value for u_value, x_value in zip(u, grid.x)]
    Ku = []
    for index in range(count):
        value = diagonal[index] * u[index]
        if index:
            value += lower[index - 1] * u[index - 1]
        if index + 1 < count:
            value += upper[index] * u[index + 1]
        Ku.append(value)
    residual = [value - target for value, target in zip(Ku, rhs)]
    residual_scale = max(1.0, *(abs(value) for value in rhs), *(abs(value) for value in Ku))
    residual_relative = max(abs(value) for value in residual) / residual_scale
    matrix_quadratic = 4.0 * math.pi * grid.dx * sum(left * right for left, right in zip(u, Ku))
    source_integral = 4.0 * math.pi * grid.dx * sum(left * right for left, right in zip(rhs, u))
    vector_energy = 0.5 * matrix_quadratic
    identity_relative = abs(2.0 * vector_energy - source_integral) / max(1.0, abs(source_integral))
    if vector_energy < -1.0e-12 or min(a) < -1.0e-12:
        raise Q4DropletAuditError("positive Gauss operator gave an unphysical negative result")
    return {
        "gauss_residual_relative": residual_relative,
        "gauss_identity_relative": identity_relative,
        "vector_energy_dimensionless": vector_energy,
        "source_integral_dimensionless": source_integral,
        "min_a_A_over_W0": min(a),
        "max_a_A_over_W0": max(a),
        "min_vector_energy_positive": bool(vector_energy >= 0.0),
    }


def finite_profile_control_row(
    target_N: int,
    box_fm: float,
    resolution_label: str,
    cells_per_fm: int,
    control: ProfileControl,
    p: Params = PARAMS,
) -> dict[str, Any]:
    """Evaluate one deterministic, nonstationary profile control."""

    grid = make_grid(box_fm, cells_per_fm, p)
    density, y, radius_fm = _profile_for_control(grid, target_N, control, p)
    number = sum(weight * value for weight, value in zip(grid.weights, density))
    if abs(number - target_N) / target_N > 2.0e-15:
        raise Q4DropletAuditError("fixed-N normalization did not close")
    C = scalar_coercivity_constant(p)
    fermi_energy = 0.0
    potential_energy = 0.0
    local_exact_gap = 0.0
    local_sufficient_gap = 0.0
    z_min = math.inf
    z_max = 0.0
    for weight, n_value, y_value in zip(grid.weights, density, y):
        # A numerically underflowed Gaussian tail represents the exact
        # n=0 vacuum limit.  It contributes no local energy/gap and is not
        # an input to the n>0 analytic inequality.
        if n_value == 0.0:
            continue
        fermi, z, ratio = _fermi_density_dimensionless(n_value, y_value, p)
        potential = p.lam * (y_value * y_value - 1.0) ** 2 / 4.0
        exact_gap = ratio + potential_per_baryon_over_M(y_value, z, C) - 1.0
        sufficient_gap = sufficient_local_gap_lower_bound(y_value, z, C)
        fermi_energy += weight * fermi
        potential_energy += weight * potential
        local_exact_gap += weight * p.g_s * n_value * exact_gap
        local_sufficient_gap += weight * p.g_s * n_value * sufficient_gap
        z_min = min(z_min, z)
        z_max = max(z_max, z)
    scalar_gradient = _scalar_gradient_dimensionless(grid, y)
    gauss = _gauss_solve_and_energy(grid, density, y, p)
    total_energy = scalar_gradient + potential_energy + fermi_energy + gauss["vector_energy_dimensionless"]
    free_threshold = p.g_s * target_N
    total_gap = total_energy - free_threshold
    outer_number = sum(
        weight * n_value
        for weight, n_value, x_value in zip(grid.weights, density, grid.x)
        if x_value > 0.8 * grid.x_max
    )
    row = {
        "target_N_B": int(target_N),
        "box_fm": float(box_fm),
        "resolution": resolution_label,
        "cells_per_fm": int(cells_per_fm),
        "radial_intervals": grid.intervals,
        "x_box": grid.x_max,
        "dx": grid.dx,
        "profile_control": control.identifier,
        "profile_radius_fm": radius_fm,
        "profile_radius_x": radius_fm * p.W0 / p.hbar_c,
        "profile_scalar_depression": control.scalar_depression,
        "nonstationary_control_only": True,
        "finite_Dirichlet_positive_operator_control_only": True,
        "finite_boundary_condition": "u(0)=u(X)=0; no physical Yukawa-tail matching is claimed",
        "conserved_N_B": number,
        "conserved_N_relative_error": abs(number - target_N) / target_N,
        "min_y_W_over_W0": min(y),
        "max_y_W_over_W0": max(y),
        "z_kF_over_M_min": z_min,
        "z_kF_over_M_max": z_max,
        "outer_number_fraction_r_gt_0_8_box": outer_number / number,
        "energy_components_MeV": {
            "scalar_gradient": scalar_gradient * p.W0,
            "scalar_potential": potential_energy * p.W0,
            "fermi": fermi_energy * p.W0,
            "Gauss_vector": gauss["vector_energy_dimensionless"] * p.W0,
            "total": total_energy * p.W0,
            "free_N_times_M_N": free_threshold * p.W0,
        },
        "local_exact_gap_MeV": local_exact_gap * p.W0,
        "local_sufficient_gap_lower_bound_MeV": local_sufficient_gap * p.W0,
        "total_E_minus_NM_MeV": total_gap * p.W0,
        "E_over_N_minus_M_N_MeV": total_gap * p.W0 / target_N,
        "Gauss_constraint": gauss,
        "positive_gap_checks": {
            "local_exact_gap_positive": bool(local_exact_gap > 0.0),
            "local_sufficient_gap_nonnegative": bool(local_sufficient_gap >= -1.0e-12),
            "scalar_gradient_nonnegative": bool(scalar_gradient >= 0.0),
            "Gauss_vector_nonnegative": bool(gauss["vector_energy_dimensionless"] >= 0.0),
            "total_E_gt_NM": bool(total_gap > 0.0),
        },
    }
    return row


def _find_row(rows: Iterable[Mapping[str, Any]], target_N: int, box_fm: float, resolution: str, profile: str) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["target_N_B"] == target_N
        and row["box_fm"] == box_fm
        and row["resolution"] == resolution
        and row["profile_control"] == profile
    ]
    if len(matches) != 1:
        raise Q4DropletAuditError("numerical control row is not uniquely addressable")
    return matches[0]


def numerical_controls(p: Params = PARAMS) -> dict[str, Any]:
    """Run fixed finite box/grid/profile controls independent of the proof."""

    controls = [ProfileControl(*values) for values in PROFILE_CONTROLS]
    rows: list[dict[str, Any]] = []
    for target_N in N_B_LADDER:
        for box_fm in BOXES_FM:
            for label, cells_per_fm in GRID_CELLS_PER_FM:
                for control in controls:
                    rows.append(finite_profile_control_row(target_N, box_fm, label, cells_per_fm, control, p))
    gauss_pass = all(
        row["Gauss_constraint"]["gauss_residual_relative"] <= GAUSS_RELATIVE_LIMIT
        and row["Gauss_constraint"]["gauss_identity_relative"] <= GAUSS_RELATIVE_LIMIT
        and row["Gauss_constraint"]["min_vector_energy_positive"]
        for row in rows
    )
    positivity_pass = all(
        all(row["positive_gap_checks"].values())
        and row["conserved_N_relative_error"] <= 2.0e-15
        and row["min_y_W_over_W0"] > 0.0
        for row in rows
    )
    grid_changes = []
    for target_N in N_B_LADDER:
        for box_fm in BOXES_FM:
            for control in controls:
                coarse = _find_row(rows, target_N, box_fm, "coarse", control.identifier)
                fine = _find_row(rows, target_N, box_fm, "fine", control.identifier)
                coarse_gap = float(coarse["total_E_minus_NM_MeV"])
                fine_gap = float(fine["total_E_minus_NM_MeV"])
                relative = abs(fine_gap - coarse_gap) / max(1.0, abs(fine_gap))
                grid_changes.append(
                    {
                        "target_N_B": target_N,
                        "box_fm": box_fm,
                        "profile_control": control.identifier,
                        "coarse_to_fine_relative_change_E_minus_NM": relative,
                    }
                )
    box_changes = []
    for target_N in N_B_LADDER:
        for control in controls:
            values = [
                float(_find_row(rows, target_N, box_fm, "fine", control.identifier)["total_E_minus_NM_MeV"])
                for box_fm in BOXES_FM
            ]
            relative = max(values) - min(values)
            relative /= max(1.0, abs(values[-1]))
            box_changes.append(
                {
                    "target_N_B": target_N,
                    "profile_control": control.identifier,
                    "boxes_fm": list(BOXES_FM),
                    "fine_box_span_relative_change_E_minus_NM": relative,
                }
            )
    max_grid_change = max(row["coarse_to_fine_relative_change_E_minus_NM"] for row in grid_changes)
    max_box_change = max(row["fine_box_span_relative_change_E_minus_NM"] for row in box_changes)
    convergence_pass = bool(max_grid_change <= GRID_RELATIVE_LIMIT and max_box_change <= BOX_RELATIVE_LIMIT)
    return {
        "method": "finite-Dirichlet positive-operator Gauss control on prescribed nonstationary profiles",
        "not_a_stationary_solver_or_droplet_search": True,
        "not_a_vacuum_tail_matched_finite_droplet_calculation": True,
        "N_B_ladder": list(N_B_LADDER),
        "boxes_fm": list(BOXES_FM),
        "grid_cells_per_fm": {label: cells for label, cells in GRID_CELLS_PER_FM},
        "profile_controls": [
            {
                "identifier": control.identifier,
                "radius_factor": control.radius_factor,
                "scalar_depression": control.scalar_depression,
                "role": "initial-condition/profile robustness control only; not fitted",
            }
            for control in controls
        ],
        "profile_radius_definition": "radius_x=radius_factor*2.5*N_B^(1/3), x=W0*r/(hbar*c); no n0 or empirical radius input",
        "row_count": len(rows),
        "rows": rows,
        "grid_changes": grid_changes,
        "box_changes": box_changes,
        "limits": {
            "Gauss_relative": GAUSS_RELATIVE_LIMIT,
            "grid_relative_E_minus_NM": GRID_RELATIVE_LIMIT,
            "box_relative_E_minus_NM": BOX_RELATIVE_LIMIT,
        },
        "max_grid_relative_change_E_minus_NM": max_grid_change,
        "max_box_relative_change_E_minus_NM": max_box_change,
        "all_Gauss_constraints_pass": gauss_pass,
        "all_positive_gap_checks_pass": positivity_pass,
        "grid_and_box_control_pass": convergence_pass,
        "pass": bool(gauss_pass and positivity_pass and convergence_pass),
    }


def build_result(p: Params = PARAMS) -> dict[str, Any]:
    """Build a deterministic, source-complete Q4 TF no-go certificate."""

    inputs = action_inputs(p)
    analytic = analytic_bound_controls(p)
    numerical = numerical_controls(p)
    if not analytic["pass"] or not numerical["pass"]:
        raise Q4DropletAuditError("required analytic or numerical control failed")
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_weight": 0.0,
        "source_code_sha256": _source_sha256(),
        "action_inputs": inputs,
        "scope": {
            "included": [
                "unmodified source-complete scalar potential lambda(W^2-W0^2)^2/4",
                "M*(W)=M_N W/W0 (s=1)",
                "one d=4 zero-temperature no-current Thomas-Fermi baryon gas",
                "static finite configurations with W>=0",
                "nonlocal Gauss constraint (-nabla^2+q_phi^2 W^2)A0=g_omega n_B",
            ],
            "excluded": [
                "Coulomb and isovector sectors",
                "W8 or any inverse/fitted potential",
                "density-dependent couplings and transfer vertices",
                "pairing, shell, spin-orbit, RPA and beyond-Thomas-Fermi corrections",
                "gravity, rotation, finite temperature and external reservoirs",
            ],
        },
        "analytic_variational_bound": analytic,
        "finite_radial_numerical_controls": numerical,
        "conclusion": {
            "statement": (
                "Within the stated static d=4, s=1 source-complete zero-temperature Thomas-Fermi functional, "
                "every finite configuration with N_B>0 has E>N_B M_N; therefore it cannot be self-bound and "
                "no finite self-bound stationary droplet exists in this restricted functional."
            ),
            "why": (
                "The sharp Jensen Fermi-plus-quartic inequality is strictly above M_N n_B for C>1/4, "
                "while scalar-gradient and on-shell Gauss energies are nonnegative."
            ),
            "not_claimed": [
                "a proof about the full NVG theory or nature",
                "a numerical proof concerning omitted quantum or added interaction sectors",
                "nonexistence of unbound finite-box stationary branches",
                "a prediction for any real nucleus",
            ],
            "numerical_role": (
                "The finite radial rows re-solve Gauss and test box/grid/profile robustness, but the conclusion "
                "does not depend on treating those prescribed profiles as solutions."
            ),
        },
    }


def validate_result(payload: Any) -> bool:
    """Fail closed: accept only an exact regeneration by current source code."""

    if not isinstance(payload, dict):
        return False
    try:
        expected = build_result()
    except (ArithmeticError, OSError, Q4DropletAuditError, ValueError):
        return False
    return _canonical_json(payload) == _canonical_json(expected)


def write_result(result: Mapping[str, Any], path: Path = RESULT_PATH) -> None:
    if not validate_result(dict(result)):
        raise Q4DropletAuditError("refusing to write a non-regenerated result")
    path.write_text(_canonical_json(result), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT_PATH, help="output JSON path")
    parser.add_argument("--validate", type=Path, help="validate an existing JSON artifact and write no files")
    args = parser.parse_args(argv)
    if args.validate is not None:
        try:
            payload = json.loads(args.validate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
            return 1
        if not validate_result(payload):
            print("VALIDATION_FAILED: artifact is not an exact current regeneration", file=sys.stderr)
            return 1
        print("VALIDATION_PASS")
        return 0
    result = build_result()
    write_result(result, args.output)
    print(_canonical_json(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
