#!/usr/bin/env python3
"""Constrained static finite-basis stability audit for the accepted W8 droplets.

This is a deliberately bounded second-variation calculation.  It reuses the
read-only finite-droplet producer for the two accepted ``y*=.90/.93, N=208``
backgrounds, then assembles the analytic constrained Hessian in real radial
spherical-harmonic sectors ``ell=0..6``.  The vector field is eliminated with
a positive self-adjoint radial Gauss operator, so the induced nonlocal term is
positive and the field-dependent scalar/vector mixing is retained.

The command line emits strict JSON and has no file-writing side effect.  A
positive finite-basis spectrum is only a restricted static statement; a
robust negative direction is an instability witness, not a frequency.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    from scipy.integrate import simpson
    from scipy.special import spherical_jn
except ImportError as exc:  # pragma: no cover - requirements pin SciPy.
    raise RuntimeError("SciPy is required for the droplet stability audit") from exc

sys.dont_write_bytecode = True

try:
    import nvg_finite_droplet_audit as finite
except ImportError:  # pragma: no cover - package-style import support.
    from . import nvg_finite_droplet_audit as finite


HERE = Path(__file__).resolve().parent
SCHEMA = "nvg_droplet_stability_audit.v1"
STATUS = "COMPUTED_FINITE_W8_STATIC_STABILITY_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0
TARGETS = ("0.90", "0.93")
TARGET_N = 208
ELL_VALUES = tuple(range(7))
BASIS_SIZES = (6, 10, 14, 18)
FINE_INTERVALS = 1600
GRID_INTERVALS = (800, 1600, 3200)
DOMAIN_BOXES_FM = (24.0, 32.0)
BACKGROUND_NODES = 1400
BACKGROUND_TOLERANCE = 1.0e-5
GRAM_RELATIVE_CUTOFF = 1.0e-10
SIGN_ERROR_MULTIPLIER = 5.0
SIGN_RESOLUTION_FLOOR = 1.0e-8
BASIS_RELATIVE_DRIFT_LIMIT = 0.10
DISCRETIZATION_RELATIVE_DRIFT_LIMIT = 0.05
FD_AMPLITUDES = (0.02, 0.01, 0.005)
FD_AGREEMENT_RELATIVE_LIMIT = 0.08
FIXED_N_RELATIVE_LIMIT = 1.0e-10
NUMBER_CONSTRAINT_RELATIVE_LIMIT = 1.0e-6
PROJECTION_ORTHOGONALITY_LIMIT = 1.0e-10
TRANSLATION_RELATIVE_LIMIT = 2.0e-2
BACKGROUND_TIGHT_NODES = 1600
BACKGROUND_TIGHT_TOLERANCE = 1.0e-8
TRANSLATION_DIRECTION = "u=-n_prime, v=-y_prime, w=-a_prime"
TRANSLATION_REPRESENTATION = "ell=1 displacement translation column plus scalar -y_prime column; their sum is the joint mode"
TRANSLATION_CONSISTENCY = "MEASURED_FINITE_DOMAIN_IDENTITY"
FD_DIRECTION_NAMES = ("scalar_only_compact", "coupled_zero_integral_density_scalar")


class StabilityError(ValueError):
    """Fail-closed error for malformed or incomplete stability evidence."""


def _number(value: Any, digits: int = 17) -> Any:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StabilityError("non-numeric stability output") from exc
    if not math.isfinite(value):
        raise StabilityError("non-finite stability output")
    return float(format(value, f".{digits}g"))


def _finite_scalar(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and not isinstance(value, bool)
    except (TypeError, ValueError, OverflowError):
        return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not isinstance(value, bool):
        return _number(value)
    return value


def _integral(x: np.ndarray, values: np.ndarray) -> float:
    """Use Simpson integration on a uniform radial grid."""

    return float(simpson(np.asarray(values, dtype=float), x=np.asarray(x, dtype=float)))


def _integral_matrix(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Integrate a stack of pair products along its first axis."""

    return np.asarray(
        simpson(np.asarray(values, dtype=float), x=np.asarray(x, dtype=float), axis=0),
        dtype=float,
    )


def _compact_bump(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """C-infinity radial bump with compact support."""

    z = (x - center) / width
    out = np.zeros_like(x, dtype=float)
    inside = np.abs(z) < 1.0
    q = 1.0 - z[inside] ** 2
    out[inside] = np.exp(-1.0 / np.maximum(q, 1.0e-300))
    return out


def _smooth_tail_cutoff(x: np.ndarray, x_max: float, start_fraction: float = 0.72) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-minus-cosine cutoff that is exactly one in the interior."""

    start = float(start_fraction) * x_max
    width = x_max - start
    t = (x - start) / width
    c = np.ones_like(x, dtype=float)
    cp = np.zeros_like(x, dtype=float)
    cpp = np.zeros_like(x, dtype=float)
    high = t >= 1.0
    transition = (t > 0.0) & (t < 1.0)
    c[high] = 0.0
    c[transition] = 0.5 * (1.0 + np.cos(math.pi * t[transition]))
    cp[transition] = -0.5 * math.pi / width * np.sin(math.pi * t[transition])
    cpp[transition] = -0.5 * (math.pi / width) ** 2 * np.cos(math.pi * t[transition])
    return c, cp, cpp


def _bessel_second(ell: int, z: np.ndarray, first: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Second derivative of spherical j_l, with the origin limit supplied."""

    z = np.asarray(z, dtype=float)
    second = np.zeros_like(z)
    nonzero = np.abs(z) > 1.0e-8
    second[nonzero] = (
        -2.0 * first[nonzero] / z[nonzero]
        - (1.0 - ell * (ell + 1.0) / z[nonzero] ** 2) * value[nonzero]
    )
    if np.any(~nonzero):
        if ell == 0:
            second[~nonzero] = -1.0 / 3.0
        elif ell == 2:
            second[~nonzero] = 2.0 / 15.0
        else:
            second[~nonzero] = 0.0
    return second


@dataclass
class Background:
    target_y: str
    design: Any
    box_fm: float
    solution: Any
    solver_row: Mapping[str, Any]
    x: np.ndarray
    y: np.ndarray
    yp: np.ndarray
    ypp: np.ndarray
    a: np.ndarray
    ap: np.ndarray
    app: np.ndarray
    n: np.ndarray
    nprime: np.ndarray
    ns: np.ndarray
    fermi_f: np.ndarray
    fermi_margin: np.ndarray
    mu: float

    @classmethod
    def from_solution(
        cls,
        design: Any,
        box_fm: float,
        solution: Any,
        solver_row: Mapping[str, Any],
        intervals: int,
    ) -> "Background":
        x_max = design.W0 * float(box_fm) / design.hbarc
        x = np.linspace(0.0, x_max, int(intervals) + 1)
        values = np.asarray(solution.sol(x), dtype=float)
        first = np.asarray(solution.sol(x, 1), dtype=float)
        try:
            second = np.asarray(solution.sol(x, 2), dtype=float)
        except (TypeError, ValueError):  # pragma: no cover - old SciPy.
            second = np.vstack([
                np.gradient(first[index], x, edge_order=2)
                for index in range(first.shape[0])
            ])
        y, yp, a, ap = values[:4]
        matter = design.matter(y, a, float(solution.p[0]))
        n = np.maximum(np.asarray(matter["n"], dtype=float), 0.0)
        ns = np.asarray(matter["ns"], dtype=float)
        fermi_f = np.asarray(matter["f"], dtype=float)
        ypp, app = second[0], second[2]
        # Differentiate the local TF closure analytically on the occupied
        # branch.  This retains the moving edge instead of flooring n.
        e = float(solution.p[0]) / design.W0 - design.gomega * a
        mass = design.gs * y
        k = np.sqrt(np.maximum(e * e - mass * mass, 0.0))
        active = e > mass
        fermi_margin = e - mass
        eprime = -design.gomega * ap
        mprime = design.gs * yp
        nprime = np.zeros_like(x)
        nprime[active] = (
            design.d * k[active]
            * (e[active] * eprime[active] - mass[active] * mprime[active])
            / (2.0 * math.pi**2)
        )
        return cls(
            target_y=design.target_y,
            design=design,
            box_fm=float(box_fm),
            solution=solution,
            solver_row=solver_row,
            x=x,
            y=y,
            yp=yp,
            ypp=ypp,
            a=a,
            ap=ap,
            app=app,
            n=n,
            nprime=nprime,
            ns=ns,
            fermi_f=fermi_f,
            fermi_margin=fermi_margin,
            mu=float(solution.p[0]),
        )

    @property
    def x_max(self) -> float:
        return float(self.x[-1])

    @property
    def edge_x(self) -> float:
        positive = np.flatnonzero(self.fermi_margin > 0.0)
        if positive.size == 0:
            return 0.0
        last = int(positive[-1])
        if last >= self.x.size - 1:
            return float(self.x[-1])
        left = float(self.fermi_margin[last])
        right = float(self.fermi_margin[last + 1])
        if right >= 0.0 or left == right:
            return float(self.x[last])
        fraction = left / (left - right)
        return float(self.x[last] + fraction * (self.x[last + 1] - self.x[last]))

    def summary(self) -> dict[str, Any]:
        row = self.solver_row
        return {
            "box_fm": self.box_fm,
            "converged": bool(row.get("converged") is True),
            "solver_status": row.get("solver_status"),
            "nodes_used": row.get("nodes_used"),
            "chemical_potential_MeV": self.mu,
            "conserved_N_relative_error": row.get("conserved_N_relative_error"),
            "field_residual_relative": row.get("field_residual_relative"),
            "virial_relative": row.get("virial_relative"),
            "gauss_energy_identity_relative": row.get("gauss_energy_identity_relative"),
            "energy_per_particle_minus_M_MeV": row.get("energy_per_particle_minus_M_MeV"),
            "rms_baryon_radius_fm": row.get("rms_baryon_radius_fm"),
            "edge_x": self.edge_x,
            "central_y": float(self.y[0]),
            "central_density_fm_minus3": row.get("central_density_fm_minus3"),
        }


def _occupied_integral_matrix(background: Background, values: np.ndarray) -> np.ndarray:
    """Integrate occupied-cell products with an interpolated moving edge.

    The TF density vanishes at the crossing ``nu=m``.  Products containing
    ``F_nn`` are therefore integrated only to that crossing, rather than
    allowing Simpson's rule to straddle the discontinuous active mask.
    """

    x = background.x
    edge = background.edge_x
    if edge <= x[0]:
        return np.zeros(values.shape[1:], dtype=float)
    last = int(np.searchsorted(x, edge, side="right") - 1)
    last = max(0, min(last, x.size - 1))
    xx = x[: last + 1]
    vv = np.asarray(values[: last + 1], dtype=float)
    if edge > xx[-1] and last + 1 < x.size:
        fraction = (edge - xx[-1]) / (x[last + 1] - xx[-1])
        edge_value = vv[-1] + fraction * (values[last + 1] - vv[-1])
        xx = np.concatenate((xx, np.asarray([edge], dtype=float)))
        vv = np.concatenate((vv, edge_value[None, ...]), axis=0)
    return _integral_matrix(xx, vv)


def _edge_aware_number_integral(background: Background, density: np.ndarray) -> float:
    """Independently integrate ``x^2 density`` to the moving TF edge."""

    x = np.asarray(background.x, dtype=float)
    density = np.asarray(density, dtype=float).reshape(-1)
    if density.size != x.size:
        raise StabilityError("number-constraint vector has wrong grid size")
    edge = float(background.edge_x)
    if edge <= x[0]:
        return 0.0
    last = int(np.searchsorted(x, edge, side="right") - 1)
    last = max(0, min(last, x.size - 1))
    xx = x[: last + 1]
    dd = density[: last + 1]
    if edge > xx[-1] and last + 1 < x.size:
        fraction = (edge - xx[-1]) / (x[last + 1] - xx[-1])
        edge_density = dd[-1] + fraction * (density[last + 1] - dd[-1])
        xx = np.concatenate((xx, np.asarray([edge], dtype=float)))
        dd = np.concatenate((dd, np.asarray([edge_density], dtype=float)))
    return float(simpson(xx**2 * dd, x=xx))


def _number_constraint_diagnostics(background: Background, u: np.ndarray) -> dict[str, Any]:
    """Measure displacement-number residuals without a density floor."""

    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    residuals: list[float] = []
    denominators: list[float] = []
    for column in range(u.shape[1]):
        numerator = abs(_edge_aware_number_integral(background, u[:, column]))
        denominator = _edge_aware_number_integral(background, np.abs(u[:, column]))
        denominators.append(denominator)
        if denominator <= 1.0e-30:
            # An exactly zero displacement has an explicitly defined zero
            # residual; a nonzero numerator would instead be a failure.
            residuals.append(0.0 if numerator <= 1.0e-30 else math.inf)
        else:
            residuals.append(numerator / denominator)
    return {
        "number_constraint_residuals_relative": residuals,
        "number_constraint_denominators": denominators,
        "number_constraint_residual_max": max(residuals) if residuals else 0.0,
        "number_constraint_zero_vector_count": sum(denominator <= 1.0e-30 for denominator in denominators),
        "number_constraint_mode": "radial_integral" if u.shape[1] else "empty",
    }


def _backgrounds(target_y: str) -> tuple[Background, Background, Background]:
    """Recompute accepted backgrounds only through the upstream API."""

    design = finite.W8Design.from_target(target_y)
    row24, solution24 = finite._solve_once(
        design,
        TARGET_N,
        DOMAIN_BOXES_FM[0],
        BACKGROUND_NODES,
        BACKGROUND_TOLERANCE,
        factor=1.0,
    )
    if solution24 is None or row24.get("converged") is not True:
        raise StabilityError(f"24 fm background failed for y={target_y}: {row24}")
    row32, solution32 = finite._solve_once(
        design,
        TARGET_N,
        DOMAIN_BOXES_FM[1],
        BACKGROUND_NODES,
        BACKGROUND_TOLERANCE,
        previous=solution24,
    )
    if solution32 is None or row32.get("converged") is not True:
        raise StabilityError(f"32 fm background failed for y={target_y}: {row32}")
    row_tight, solution_tight = finite._solve_once(
        design,
        TARGET_N,
        DOMAIN_BOXES_FM[0],
        BACKGROUND_TIGHT_NODES,
        BACKGROUND_TIGHT_TOLERANCE,
        factor=1.0,
    )
    if solution_tight is None or row_tight.get("converged") is not True:
        raise StabilityError(f"tight 24 fm background failed for y={target_y}: {row_tight}")
    # Use a dense independent radial grid for all stability calculations.
    return (
        Background.from_solution(design, DOMAIN_BOXES_FM[0], solution24, row24, 3200),
        Background.from_solution(design, DOMAIN_BOXES_FM[1], solution32, row32, 3200),
        Background.from_solution(design, DOMAIN_BOXES_FM[0], solution_tight, row_tight, 3200),
    )


def _basis_function(
    x: np.ndarray,
    ell: int,
    index: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular smooth spherical-Bessel potential and its first two derivatives."""

    frequency = (index + 1.0) * math.pi / scale
    z = frequency * x
    value = spherical_jn(ell, z)
    first_z = spherical_jn(ell, z, derivative=True)
    second_z = _bessel_second(ell, z, first_z, value)
    t = x / scale
    cutoff = np.exp(-(t**4))
    cutoff_p = -4.0 * t**3 / scale * cutoff
    cutoff_pp = (16.0 * t**6 - 12.0 * t**2) / scale**2 * cutoff
    psi = value * cutoff
    psi_p = frequency * first_z * cutoff + value * cutoff_p
    psi_pp = frequency**2 * second_z * cutoff + 2.0 * frequency * first_z * cutoff_p + value * cutoff_pp
    return psi, psi_p, psi_pp


def _translation_potential(x: np.ndarray, x_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boundary-tapered translation potential plus a smooth scalar cutoff."""

    cutoff, cutoff_p, cutoff_pp = _smooth_tail_cutoff(x, x_max)
    psi = x * cutoff
    psi_p = cutoff + x * cutoff_p
    psi_pp = 2.0 * cutoff_p + x * cutoff_pp
    return psi, psi_p, psi_pp, cutoff, cutoff_p


def _displacement_density(
    x: np.ndarray,
    n: np.ndarray,
    nprime: np.ndarray,
    ell: int,
    psi: np.ndarray,
    psi_p: np.ndarray,
    psi_pp: np.ndarray,
) -> np.ndarray:
    """Continuity-generated fixed-N density direction, including n' at edge."""

    angular = float(ell * (ell + 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = -(nprime * psi_p + n * psi_pp + 2.0 * n * psi_p / x)
        u += angular * n * psi / x**2
    u = np.asarray(u, dtype=float)
    if x.size > 1:
        if ell == 0:
            u[0] = u[1]
        else:
            u[0] = 0.0
    u[~np.isfinite(u)] = 0.0
    # The physical branch is exactly zero outside the occupied support.  This
    # removes only roundoff tails, not a density floor.
    u[n <= 0.0] = 0.0
    return u


def _basis_columns(background: Background, ell: int, size: int) -> dict[str, Any]:
    x = background.x
    n = background.n
    nprime = background.nprime
    scale = min(background.design.W0 * 8.0 / background.design.hbarc, 0.45 * background.x_max)
    psi_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    scalar_data: list[tuple[np.ndarray, np.ndarray]] = []
    translation_cutoff = None
    if ell == 1:
        psi, psi_p, psi_pp, cutoff, cutoff_p = _translation_potential(x, background.x_max)
        psi_data.append((psi, psi_p, psi_pp))
        scalar_data.append((-background.yp * cutoff, -(background.ypp * cutoff + background.yp * cutoff_p)))
        translation_cutoff = cutoff
    for index in range(size - len(psi_data)):
        psi_data.append(_basis_function(x, ell, index, scale))
        phi, phi_p, _ = _basis_function(x, ell, index, scale)
        scalar_data.append((phi, phi_p))
    u_columns: list[np.ndarray] = []
    v_columns: list[np.ndarray] = []
    labels: list[str] = []
    for index, data in enumerate(psi_data):
        u_columns.append(_displacement_density(x, n, nprime, ell, *data))
        v_columns.append(np.zeros_like(x))
        labels.append(f"density_displacement_{index + 1}")
    for index, data in enumerate(scalar_data):
        u_columns.append(np.zeros_like(x))
        v_columns.append(np.asarray(data[0], dtype=float))
        labels.append(f"scalar_{index + 1}")
    if ell == 0 and u_columns:
        # The continuum displacement identity gives delta N=0 by a boundary
        # term.  The sampled BVP derivative and moving-edge interpolation do
        # not preserve that cancellation to binary64 accuracy, especially at
        # the dilute TF edge.  Project each sampled density column onto the
        # independently integrated fixed-number hyperplane.  The correction
        # is a smooth interior density shape, so it does not introduce an
        # edge floor or alter the scalar block; it only removes quadrature
        # leakage from the declared constraint.
        correction = _compact_bump(x, 0.45 * max(background.edge_x, 4.0 * (x[1] - x[0])),
                                   0.25 * max(background.edge_x, 4.0 * (x[1] - x[0])))
        correction *= max(float(np.max(np.abs(n))), 1.0e-30)
        correction_integral = _edge_aware_number_integral(background, correction)
        if not math.isfinite(correction_integral) or abs(correction_integral) <= 1.0e-30:
            raise StabilityError("failed to construct fixed-number projection")
        for index, column in enumerate(u_columns):
            residual = _edge_aware_number_integral(background, column)
            u_columns[index] = column - (residual / correction_integral) * correction
    translation_vector = np.zeros(len(u_columns), dtype=float)
    if ell == 1:
        translation_vector[0] = 1.0
        translation_vector[len(psi_data)] = 1.0
    return {
        "u": np.column_stack(u_columns),
        "v": np.column_stack(v_columns),
        "vprime": np.column_stack([
            np.zeros_like(x) if index < len(psi_data) else scalar_data[index - len(psi_data)][1]
            for index in range(len(u_columns))
        ]),
        "labels": labels,
        "displacement_count": len(psi_data),
        "scalar_count": len(scalar_data),
        "translation_basis_present": bool(ell == 1),
        "translation_coefficient_vector": translation_vector,
        "translation_cutoff_min": None if translation_cutoff is None else float(np.min(translation_cutoff)),
    }


def _operator_tridiagonal(
    design: Any,
    x: np.ndarray,
    y: np.ndarray,
    ell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Assemble a positive self-adjoint P1 radial operator in weak form."""

    intervals = x.size - 1
    h = float(x[1] - x[0])
    global_nodes = np.arange(0, intervals) if ell == 0 else np.arange(1, intervals)
    index_of = {int(node): index for index, node in enumerate(global_nodes)}
    n_unknown = global_nodes.size
    lower = np.zeros(max(n_unknown - 1, 0), dtype=float)
    diagonal = np.zeros(n_unknown, dtype=float)
    upper = np.zeros(max(n_unknown - 1, 0), dtype=float)

    for element in range(intervals):
        left = float(x[element])
        right = float(x[element + 1])
        step = right - left
        i2 = (right**3 - left**3) / 3.0
        i3 = (right**4 - left**4) / 4.0
        i4 = (right**5 - left**5) / 5.0
        m2_ll = (right**2 * i2 - 2.0 * right * i3 + i4) / step**2
        m2_lr = ((left + right) * i3 - i4 - left * right * i2) / step**2
        m2_rr = (i4 - 2.0 * left * i3 + left**2 * i2) / step**2
        m0_ll = step / 3.0
        m0_lr = step / 6.0
        m0_rr = step / 3.0
        gradient = ((left + right) * 0.5) ** 2 / step
        potential = design.q**2 * ((y[element] + y[element + 1]) * 0.5) ** 2
        angular = float(ell * (ell + 1))
        local = np.asarray([
            [gradient + angular * m0_ll + potential * m2_ll,
             -gradient + angular * m0_lr + potential * m2_lr],
            [-gradient + angular * m0_lr + potential * m2_lr,
             gradient + angular * m0_rr + potential * m2_rr],
        ])
        nodes = (element, element + 1)
        for i_local, node_i in enumerate(nodes):
            if node_i not in index_of:
                continue
            i_global = index_of[node_i]
            diagonal[i_global] += local[i_local, i_local]
            for j_local, node_j in enumerate(nodes):
                if j_local == i_local or node_j not in index_of:
                    continue
                j_global = index_of[node_j]
                if j_global == i_global + 1:
                    upper[i_global] += local[i_local, j_local]
                elif j_global == i_global - 1:
                    lower[j_global] += local[i_local, j_local]
    # The algebraic operator is symmetric by construction.  Thomas pivots
    # below provide an inexpensive positive-definiteness diagnostic.
    pivots = diagonal.copy()
    for index in range(1, pivots.size):
        pivots[index] -= lower[index - 1] * upper[index - 1] / pivots[index - 1]
    return lower, diagonal, upper, global_nodes, float(np.min(pivots)) if pivots.size else math.nan


def _solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve one or many RHS vectors and return minimum Thomas pivot."""

    d = np.asarray(diagonal, dtype=float).copy()
    lo = np.asarray(lower, dtype=float)
    up = np.asarray(upper, dtype=float)
    b = np.asarray(rhs, dtype=float).copy()
    if b.ndim == 1:
        b = b[:, None]
    pivots = d.copy()
    for index in range(1, d.size):
        factor = lo[index - 1] / d[index - 1]
        d[index] -= factor * up[index - 1]
        b[index] -= factor * b[index - 1]
        pivots[index] = d[index]
    solution = np.empty_like(b)
    solution[-1] = b[-1] / d[-1]
    for index in range(d.size - 2, -1, -1):
        solution[index] = (b[index] - up[index] * solution[index + 1]) / d[index]
    return solution if rhs.ndim == 2 else solution[:, 0], float(np.min(pivots))


def _operator_solve(
    design: Any,
    x: np.ndarray,
    y: np.ndarray,
    ell: int,
    sources: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Apply K_l^{-1} to source columns with regular-origin/Dirichlet-tail BC."""

    lower, diagonal, upper, nodes, min_pivot = _operator_tridiagonal(design, x, y, ell)
    src = np.asarray(sources, dtype=float)
    if src.ndim == 1:
        src = src[:, None]
    h = float(x[1] - x[0])
    weights = np.zeros_like(x)
    weights[1:-1] = h * x[1:-1] ** 2
    weights[0] = 0.5 * h * x[0] ** 2
    weights[-1] = 0.5 * h * x[-1] ** 2
    rhs = weights[nodes, None] * src[nodes, :]
    solved, solve_pivot = _solve_tridiagonal(lower, diagonal, upper, rhs)
    out = np.zeros((x.size, src.shape[1]), dtype=float)
    out[nodes, :] = solved
    positive = bool(min_pivot > 0.0 and solve_pivot > 0.0 and np.all(np.isfinite(out)))
    return out, min(min_pivot, solve_pivot), positive


def _matter_hessian_fields(background: Background) -> dict[str, np.ndarray]:
    design = background.design
    y = background.y
    a = background.a
    n = background.n
    fermi = design.matter(y, a, background.mu)
    k = np.asarray(fermi["k"], dtype=float) / design.W0
    e = np.asarray(fermi["ef"], dtype=float) / design.W0
    mass = design.gs * y
    positive = n > 0.0
    A = np.zeros_like(n)
    A[positive] = k[positive] ** 2 / (3.0 * n[positive] * e[positive])
    B = np.divide(design.gs**2 * y, e, out=np.zeros_like(y), where=e > 0.0)
    ratio = np.divide(k, mass, out=np.zeros_like(k), where=mass > 0.0)
    # F_yy uses a cancellation-safe series at the dilute edge and the exact
    # hyperbolic expression on the occupied interior.
    integral = np.zeros_like(k)
    small = positive & (ratio <= 0.2)
    if np.any(small):
        r = ratio[small]
        term = r**5 / 5.0
        coefficient = 1.0
        for j in range(1, 18):
            # (1+t^2)^(-3/2) = 1 - 3*t^2/2 + 15*t^4/8 - ...
            coefficient *= (-(2.0 * j + 1.0)) / (2.0 * j)
            term += coefficient * r ** (5 + 2 * j) / (5 + 2 * j)
        integral[small] = mass[small] ** 2 * term
    direct = positive & ~small
    if np.any(direct):
        r = ratio[direct]
        integral[direct] = (
            k[direct] * e[direct] / 2.0
            + mass[direct] ** 2 * k[direct] / e[direct]
            - 1.5 * mass[direct] ** 2 * np.arcsinh(r)
        )
    F_yy = design.gs**2 * design.d / (2.0 * math.pi**2) * integral
    C = F_yy + design.potential_yy(y) - design.q**2 * a**2
    return {"A": A, "B": B, "F_yy": F_yy, "C": C, "k": k, "e": e}


def _hessian_quadratic(
    background: Background,
    ell: int,
    u: np.ndarray,
    v: np.ndarray,
    vprime: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Evaluate Q_l/W0 for arbitrary radial direction columns."""

    x = background.x
    design = background.design
    fields = _matter_hessian_fields(background)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    vprime = np.asarray(vprime, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    if v.ndim == 1:
        v = v[:, None]
    if vprime.ndim == 1:
        vprime = vprime[:, None]
    ncols = max(u.shape[1], v.shape[1])
    if u.shape[1] != ncols:
        u = np.broadcast_to(u, (u.shape[0], ncols)).copy()
    if v.shape[1] != ncols:
        v = np.broadcast_to(v, (v.shape[0], ncols)).copy()
    if vprime.shape[1] != ncols:
        vprime = np.broadcast_to(vprime, (vprime.shape[0], ncols)).copy()
    x2 = x * x
    A, B, C = fields["A"], fields["B"], fields["C"]
    L = float(ell * (ell + 1))
    density_products = x2[:, None, None] * A[:, None, None] * u[:, :, None] * u[:, None, :]
    cross_products = x2[:, None, None] * B[:, None, None] * (
        u[:, :, None] * v[:, None, :] + v[:, :, None] * u[:, None, :]
    )
    scalar_products = (
        x2[:, None, None] * C[:, None, None] * v[:, :, None] * v[:, None, :]
        + x2[:, None, None] * vprime[:, :, None] * vprime[:, None, :]
        + L * v[:, :, None] * v[:, None, :]
    )
    # A and B are occupied-branch derivatives.  Splitting at the moving TF
    # edge is essential because A diverges as n -> 0 even though the combined
    # admissible density products remain integrable.
    local = (
        _occupied_integral_matrix(background, density_products)
        + _occupied_integral_matrix(background, cross_products)
        + _integral_matrix(x, scalar_products)
    )
    source = design.gomega * u - 2.0 * design.q**2 * background.y[:, None] * background.a[:, None] * v
    solved, min_pivot, positive_operator = _operator_solve(
        design, x, background.y, ell, source
    )
    nonlocal_matrix = _integral_matrix(
        x,
        x2[:, None, None] * source[:, :, None] * solved[:, None, :],
    )
    nonlocal_matrix = 0.5 * (nonlocal_matrix + nonlocal_matrix.T)
    hessian = 0.5 * ((local + nonlocal_matrix) + (local + nonlocal_matrix).T)
    return hessian, {
        "local_matrix": local,
        "nonlocal_matrix": nonlocal_matrix,
        "operator_min_pivot": min_pivot,
        "positive_operator": positive_operator,
        "source_columns": source,
    }


def _gram_matrix(background: Background, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    x = background.x
    scale = max(background.design.n0_dim, 1.0e-30)
    gram_products = x[:, None, None] ** 2 * (
        u[:, :, None] * u[:, None, :] / scale**2
        + v[:, :, None] * v[:, None, :]
    )
    gram = _integral_matrix(x, gram_products)
    return 0.5 * (gram + gram.T)


def _solve_generalized_problem(
    hessian: np.ndarray,
    gram: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Solve a Gram-normalized symmetric eigenproblem.

    The residual is evaluated after whitening by the positive Gram form, so
    its absolute value has the same Q/W0 units as the eigenvalue and is not a
    coordinate-dependent relative quantity.
    """

    gram = 0.5 * (np.asarray(gram, dtype=float) + np.asarray(gram, dtype=float).T)
    hessian = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    gram_values, gram_vectors = np.linalg.eigh(gram)
    finite_positive = np.isfinite(gram_values) & (gram_values > 0.0)
    if not np.any(finite_positive):
        raise StabilityError("basis Gram matrix has no positive rank")
    largest = float(np.max(gram_values[finite_positive]))
    keep = finite_positive & (gram_values >= GRAM_RELATIVE_CUTOFF * largest)
    kept_indices = np.flatnonzero(keep)
    transform = gram_vectors[:, kept_indices] / np.sqrt(gram_values[kept_indices])[None, :]
    reduced = transform.T @ hessian @ transform
    reduced = 0.5 * (reduced + reduced.T)
    eigenvalues, reduced_vectors = np.linalg.eigh(reduced)
    coefficients = transform @ reduced_vectors
    absolute_residuals = [
        float(np.linalg.norm(reduced @ reduced_vectors[:, index] - value * reduced_vectors[:, index]))
        for index, value in enumerate(eigenvalues)
    ]
    relative_residuals = [
        residual / max(abs(float(value)), SIGN_RESOLUTION_FLOOR)
        for residual, value in zip(absolute_residuals, eigenvalues)
    ]
    return {
        "eigenvalues_Q_over_W0": eigenvalues,
        "eigenvectors": coefficients,
        "eigen_residual_absolute": absolute_residuals,
        "eigen_residual_relative": relative_residuals,
        "eigen_residual_absolute_max": max(absolute_residuals) if absolute_residuals else math.inf,
        "gram_eigenvalues": gram_values,
        "gram_condition_number": float(np.max(gram_values[finite_positive]) / np.min(gram_values[keep])),
        "gram_rank": int(np.count_nonzero(keep)),
        "gram_size": int(gram.shape[0]),
        "rank_truncated": bool(np.count_nonzero(keep) < gram.shape[0]),
        "labels": labels,
    }


def _generalized_spectrum(
    hessian: np.ndarray,
    gram: np.ndarray,
    labels: list[str],
    excluded_vector: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return raw Ritz values and, when requested, a symmetry-projected set."""

    raw = _solve_generalized_problem(hessian, gram, labels)
    output: dict[str, Any] = {
        "raw": raw,
        "internal": raw,
        "projection_applied": False,
        "projection_removed_dimension": 0,
        "projection_basis_dimension": int(np.asarray(gram).shape[0]),
        "projection_orthogonality_relative": 0.0,
    }
    if excluded_vector is None:
        output.update(raw)
        return output

    gram = 0.5 * (np.asarray(gram, dtype=float) + np.asarray(gram, dtype=float).T)
    hessian = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    vector = np.asarray(excluded_vector, dtype=float).reshape(-1)
    if vector.size != gram.shape[0]:
        raise StabilityError("excluded symmetry vector has wrong basis dimension")
    vector_norm = float(vector @ gram @ vector)
    if not math.isfinite(vector_norm) or vector_norm <= 0.0:
        raise StabilityError("excluded symmetry vector has no positive Gram norm")
    row = (vector @ gram)[None, :]
    _, _, vh = np.linalg.svd(row, full_matrices=True)
    projection = vh[1:].T
    orthogonality = float(np.linalg.norm(vector @ gram @ projection) / max(
        np.linalg.norm(vector @ gram) * max(np.linalg.norm(projection), 1.0),
        1.0e-30,
    ))
    if projection.shape[1] != gram.shape[0] - 1:
        raise StabilityError("symmetry projection did not remove exactly one direction")
    projected_hessian = projection.T @ hessian @ projection
    projected_gram = projection.T @ gram @ projection
    internal = _solve_generalized_problem(projected_hessian, projected_gram, labels[1:])
    output.update({
        **{key: value for key, value in raw.items() if key not in {"eigenvectors"}},
        "internal": internal,
        "projection_applied": True,
        "projection_removed_dimension": 1,
        "projection_basis_dimension": int(projection.shape[1]),
        "projection_orthogonality_relative": orthogonality,
        "excluded_vector_gram_norm": vector_norm,
        "projected_gram_rank": internal["gram_rank"],
        "projected_gram_size": internal["gram_size"],
    })
    return output


def _spectrum_record(
    background: Background,
    ell: int,
    size: int,
    intervals: int,
    spectrum_transform: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if intervals != background.x.size - 1:
        # Re-evaluate the background on the requested nested grid using its
        # existing BVP solution; no saved stability result is an input.
        grid_background = Background.from_solution(
            background.design,
            background.box_fm,
            background.solution,
            background.solver_row,
            intervals,
        )
    else:
        grid_background = background
    columns = _basis_columns(grid_background, ell, size)
    hessian, details = _hessian_quadratic(
        grid_background,
        ell,
        columns["u"],
        columns["v"],
        columns["vprime"],
    )
    gram = _gram_matrix(grid_background, columns["u"], columns["v"])
    excluded_vector = columns["translation_coefficient_vector"] if ell == 1 else None
    transform_metadata: Mapping[str, Any] | None = None
    if spectrum_transform is not None:
        transformed = spectrum_transform(hessian, gram, excluded_vector)
        if not isinstance(transformed, tuple) or len(transformed) != 4:
            raise StabilityError("spectrum transform must return H, G, translation and metadata")
        hessian, gram, excluded_vector, transform_metadata = transformed
        if not isinstance(transform_metadata, Mapping):
            raise StabilityError("spectrum transform metadata must be a mapping")
    spectrum = _generalized_spectrum(
        hessian,
        gram,
        columns["labels"],
        excluded_vector=excluded_vector,
    )
    raw = spectrum["raw"]
    internal = spectrum["internal"]
    raw_values = np.asarray(raw["eigenvalues_Q_over_W0"], dtype=float)
    internal_values = np.asarray(internal["eigenvalues_Q_over_W0"], dtype=float)
    number_constraints = _number_constraint_diagnostics(grid_background, columns["u"])
    # Keep only compact JSON diagnostics; vectors/matrices are recomputed by
    # the producer and are deliberately not persisted as claimed result input.
    record = {
        "ell": int(ell),
        "basis_size_each_block": int(size),
        "basis_dimension": int(2 * size),
        "intervals": int(intervals),
        "box_fm": background.box_fm,
        "eigenvalues_Q_over_W0": raw_values,
        "smallest_eigenvalues_Q_over_W0": raw_values[: min(4, raw_values.size)],
        "eigen_residual_absolute_Q_over_W0": raw["eigen_residual_absolute"],
        "eigen_residual_absolute_max_Q_over_W0": raw["eigen_residual_absolute_max"],
        "eigen_residual_relative_max": float(max(raw["eigen_residual_relative"]) if raw["eigen_residual_relative"] else math.inf),
        "gram_condition_number": raw["gram_condition_number"],
        "gram_rank": raw["gram_rank"],
        "gram_size": raw["gram_size"],
        "rank_truncated": raw["rank_truncated"],
        "internal_eigenvalues_Q_over_W0": internal_values,
        "smallest_internal_eigenvalues_Q_over_W0": internal_values[: min(4, internal_values.size)],
        "internal_eigen_residual_absolute_Q_over_W0": internal["eigen_residual_absolute"],
        "internal_eigen_residual_absolute_max_Q_over_W0": internal["eigen_residual_absolute_max"],
        "internal_gram_condition_number": internal["gram_condition_number"],
        "internal_gram_rank": internal["gram_rank"],
        "internal_gram_size": internal["gram_size"],
        "internal_rank_truncated": internal["rank_truncated"],
        "internal_gram_rank_loss": int(internal["gram_size"] - internal["gram_rank"]),
        "projection_applied": spectrum["projection_applied"],
        "projection_removed_dimension": spectrum["projection_removed_dimension"],
        "projection_basis_dimension": spectrum["projection_basis_dimension"],
        "projection_orthogonality_relative": spectrum["projection_orthogonality_relative"],
        "excluded_vector_gram_norm": spectrum.get("excluded_vector_gram_norm"),
        "projected_gram_rank": spectrum.get("projected_gram_rank", internal["gram_rank"]),
        "projected_gram_size": spectrum.get("projected_gram_size", internal["gram_size"]),
        "number_constraint_residuals_relative": number_constraints["number_constraint_residuals_relative"],
        "number_constraint_denominators": number_constraints["number_constraint_denominators"],
        "number_constraint_residual_max": number_constraints["number_constraint_residual_max"],
        "number_constraint_zero_vector_count": number_constraints["number_constraint_zero_vector_count"],
        "number_constraint_mode": "radial_integral" if ell == 0 else "angular_integral_zero",
        "gram_rank_loss": int(raw["gram_size"] - raw["gram_rank"]),
        "operator_min_pivot": details["operator_min_pivot"],
        "positive_operator": details["positive_operator"],
        "translation_basis_present": columns["translation_basis_present"],
        "translation_cutoff_min": columns["translation_cutoff_min"],
    }
    if transform_metadata is not None:
        record["spectrum_transform_metadata"] = dict(transform_metadata)
    return record


def _vector_difference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    count: int = 4,
    key: str = "smallest_eigenvalues_Q_over_W0",
) -> float:
    a = np.asarray(left[key], dtype=float)[:count]
    b = np.asarray(right[key], dtype=float)[:count]
    length = min(a.size, b.size)
    return float(np.max(np.abs(a[:length] - b[:length]))) if length else math.inf


def _rayleigh_ritz_monotonicity(
    rows: list[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    """Check nested-basis Ritz ordering with measured absolute residual allowance."""

    violations: list[float] = []
    comparisons: list[dict[str, Any]] = []
    for left, right in zip(rows[:-1], rows[1:]):
        a = np.asarray(left[key], dtype=float)
        b = np.asarray(right[key], dtype=float)
        count = min(a.size, b.size)
        upward = np.maximum(b[:count] - a[:count], 0.0)
        maximum = float(np.max(upward)) if count else math.inf
        allowance = max(
            SIGN_RESOLUTION_FLOOR,
            float(left["internal_eigen_residual_absolute_max_Q_over_W0"]),
            float(right["internal_eigen_residual_absolute_max_Q_over_W0"]),
        )
        violations.append(maximum)
        comparisons.append({
            "from_basis_size": left["basis_size_each_block"],
            "to_basis_size": right["basis_size_each_block"],
            "max_upward_violation_Q_over_W0": maximum,
            "allowance_Q_over_W0": allowance,
            "passes": bool(maximum <= allowance),
        })
    maximum_violation = max(violations) if violations else 0.0
    maximum_allowance = max(
        [float(item["allowance_Q_over_W0"]) for item in comparisons]
        or [SIGN_RESOLUTION_FLOOR]
    )
    return {
        "key": key,
        "comparisons": comparisons,
        "max_upward_violation_Q_over_W0": maximum_violation,
        "numerical_allowance_Q_over_W0": maximum_allowance,
        "monotone_with_allowance": bool(all(item["passes"] for item in comparisons)),
    }


def _spectrum_sector(
    background24: Background,
    background32: Background,
    background_tight24: Background,
    ell: int,
    spectrum_transform: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    spectrum_key = "smallest_internal_eigenvalues_Q_over_W0" if ell == 1 else "smallest_eigenvalues_Q_over_W0"
    def record(background: Background, size: int, intervals: int) -> dict[str, Any]:
        if spectrum_transform is None:
            return _spectrum_record(background, ell, size, intervals)
        return _spectrum_record(
            background,
            ell,
            size,
            intervals,
            spectrum_transform=spectrum_transform,
        )

    basis_sweep = [record(background24, size, FINE_INTERVALS) for size in BASIS_SIZES]
    grid_sweep = [record(background24, BASIS_SIZES[-1], intervals) for intervals in GRID_INTERVALS]
    domain_sweep = [
        record(background24, BASIS_SIZES[-1], FINE_INTERVALS),
        record(background32, BASIS_SIZES[-1], FINE_INTERVALS),
    ]
    tighter_background = record(background_tight24, BASIS_SIZES[-1], FINE_INTERVALS)
    basis_delta = _vector_difference(basis_sweep[-1], basis_sweep[-2], key=spectrum_key)
    grid_delta = max(
        _vector_difference(grid_sweep[index + 1], grid_sweep[index], key=spectrum_key)
        for index in range(len(grid_sweep) - 1)
    )
    domain_delta = _vector_difference(domain_sweep[1], domain_sweep[0], key=spectrum_key)
    background_delta = _vector_difference(tighter_background, domain_sweep[0], key=spectrum_key)
    ranks = [row["gram_rank"] for row in basis_sweep + grid_sweep + domain_sweep]
    internal_ranks = [row["internal_gram_rank"] for row in basis_sweep + grid_sweep + domain_sweep]
    uncertainties = max(
        SIGN_RESOLUTION_FLOOR,
        basis_delta,
        grid_delta,
        domain_delta,
        background_delta,
        max(
            max(
                float(row["eigen_residual_absolute_max_Q_over_W0"]),
                float(row["internal_eigen_residual_absolute_max_Q_over_W0"]),
            )
            for row in basis_sweep + grid_sweep + domain_sweep + [tighter_background]
        ),
    )
    terminal = domain_sweep[1]
    values = np.asarray(terminal[spectrum_key], dtype=float)
    margins = [abs(float(value)) / (SIGN_ERROR_MULTIPLIER * uncertainties) for value in values]
    signs = ["positive" if value > 0.0 else "negative" if value < 0.0 else "zero" for value in values]
    sign_resolved = [bool(margin >= 1.0) for margin in margins]
    terminal_scale = np.maximum(np.abs(values), SIGN_RESOLUTION_FLOOR)
    basis_relative_drift = float(
        np.max(np.abs(
            np.asarray(basis_sweep[-1][spectrum_key])
            - np.asarray(basis_sweep[-2][spectrum_key])
        ) / terminal_scale)
    )
    grid_relative_drift = float(grid_delta / max(float(np.max(terminal_scale)), SIGN_RESOLUTION_FLOOR))
    domain_relative_drift = float(domain_delta / max(float(np.max(terminal_scale)), SIGN_RESOLUTION_FLOOR))
    refinement_ranks = ranks[len(basis_sweep):]
    refinement_internal_ranks = internal_ranks[len(basis_sweep):]
    monotonicity = _rayleigh_ritz_monotonicity(basis_sweep, spectrum_key)
    convergence = bool(
        len(set(refinement_ranks)) == 1
        and len(set(refinement_internal_ranks)) == 1
        and monotonicity["monotone_with_allowance"]
        and all(row["gram_rank_loss"] == 0 for row in basis_sweep)
        and all(row["internal_gram_rank_loss"] == 0 for row in basis_sweep)
        and tighter_background["gram_rank_loss"] == 0
        and tighter_background["internal_gram_rank_loss"] == 0
        and basis_relative_drift <= BASIS_RELATIVE_DRIFT_LIMIT
        and grid_relative_drift <= DISCRETIZATION_RELATIVE_DRIFT_LIMIT
        and domain_relative_drift <= DISCRETIZATION_RELATIVE_DRIFT_LIMIT
        and background_delta / max(float(np.max(terminal_scale)), SIGN_RESOLUTION_FLOOR) <= DISCRETIZATION_RELATIVE_DRIFT_LIMIT
    )
    robust_negative = bool(any(sign == "negative" and resolved for sign, resolved in zip(signs, sign_resolved)))
    robust_positive = bool(all(sign == "positive" and resolved for sign, resolved in zip(signs, sign_resolved)))
    if robust_negative:
        conclusion = "FINITE_BASIS_ROBUST_NEGATIVE_DIRECTION"
    elif robust_positive and convergence:
        conclusion = "FINITE_BASIS_POSITIVE_ONLY"
    else:
        conclusion = "UNRESOLVED_FINITE_BASIS_SIGN_OR_CONVERGENCE"
    return {
        "ell": int(ell),
        "basis_sweep": basis_sweep,
        "grid_sweep_largest_basis": grid_sweep,
        "domain_sweep_largest_basis": domain_sweep,
        "terminal_largest_basis": terminal,
        "basis_delta_max": basis_delta,
        "grid_delta_max": grid_delta,
        "domain_delta_max": domain_delta,
        "background_delta_max": background_delta,
        "background_relative_drift_max": float(
            background_delta / max(float(np.max(terminal_scale)), SIGN_RESOLUTION_FLOOR)
        ),
        "basis_relative_drift_max": basis_relative_drift,
        "grid_relative_drift_max": grid_relative_drift,
        "domain_relative_drift_max": domain_relative_drift,
        "measured_uncertainty_conservative_base": uncertainties,
        "sign_error_multiplier": SIGN_ERROR_MULTIPLIER,
        "terminal_signs": signs,
        "terminal_sign_margins": margins,
        "terminal_sign_resolved": sign_resolved,
        "rank_values": ranks,
        "internal_rank_values": internal_ranks,
        "spectrum_key_for_conclusion": spectrum_key,
        "tighter_background_same_domain": tighter_background,
        "rayleigh_ritz_monotonicity": monotonicity,
        "gram_rank_loss_significant": any(
            row["gram_rank_loss"] > 0
            or row["internal_gram_rank_loss"] > 0
            for row in basis_sweep + grid_sweep + domain_sweep + [tighter_background]
        ),
        "basis_and_grid_domain_convergence": convergence,
        "conclusion": conclusion,
    }


def _translation_metrics(background: Background) -> dict[str, Any]:
    """Independent ell=1 translation and differentiated-equation checks."""

    x = background.x
    design = background.design
    fields = _matter_hessian_fields(background)
    u = -background.nprime
    v = -background.yp
    w = -background.ap
    vprime = -background.ypp
    qform_matrix, details = _hessian_quadratic(background, 1, u, v, vprime)
    qform = float(qform_matrix[0, 0])
    norm = _integral(x, x**2 * (u**2 / max(design.n0_dim, 1.0e-30) ** 2 + v**2))
    # Strong-form residuals exclude the origin and edge cells where the
    # finite-box singular/TF-edge representation is least reliable.
    first_u = np.gradient(u, x, edge_order=2)
    second_u = np.gradient(first_u, x, edge_order=2)
    first_v = np.gradient(v, x, edge_order=2)
    second_v = np.gradient(first_v, x, edge_order=2)
    first_w = np.gradient(w, x, edge_order=2)
    second_w = np.gradient(first_w, x, edge_order=2)
    interior = (x > 5.0 * (x[1] - x[0])) & (x < background.edge_x * 0.95)
    if not np.any(interior):
        interior = x > 5.0 * (x[1] - x[0])
    scalar_matter_residual = fields["A"] * u + fields["B"] * v + design.gomega * w
    scalar_field_residual = (
        fields["B"] * u
        + (-second_v - 2.0 * first_v / np.maximum(x, 1.0e-30) + 2.0 * v / np.maximum(x, 1.0e-30) ** 2)
        + (fields["F_yy"] + design.potential_yy(background.y) - design.q**2 * background.a**2) * v
        - 2.0 * design.q**2 * background.y * background.a * w
    )
    gauss_translation_residual = (
        -second_w - 2.0 * first_w / np.maximum(x, 1.0e-30)
        + 2.0 * w / np.maximum(x, 1.0e-30) ** 2
        + design.q**2 * background.y**2 * w
        - (design.gomega * u - 2.0 * design.q**2 * background.y * background.a * v)
    )

    def weighted_norm(values: np.ndarray) -> float:
        return math.sqrt(max(_integral(x[interior], x[interior] ** 2 * values[interior] ** 2), 0.0))

    matter_terms = (
        fields["A"] * u,
        fields["B"] * v,
        design.gomega * w,
    )
    scalar_terms = (
        fields["B"] * u,
        -second_v - 2.0 * first_v / np.maximum(x, 1.0e-30) + 2.0 * v / np.maximum(x, 1.0e-30) ** 2,
        (fields["F_yy"] + design.potential_yy(background.y) - design.q**2 * background.a**2) * v,
        -2.0 * design.q**2 * background.y * background.a * w,
    )
    gauss_terms = (
        -second_w - 2.0 * first_w / np.maximum(x, 1.0e-30) + 2.0 * w / np.maximum(x, 1.0e-30) ** 2,
        design.q**2 * background.y**2 * w,
        -(design.gomega * u - 2.0 * design.q**2 * background.y * background.a * v),
    )

    def term_scale(terms: tuple[np.ndarray, ...]) -> float:
        return max(1.0e-30, math.sqrt(sum(weighted_norm(term) ** 2 for term in terms)))

    matter_scale = term_scale(matter_terms)
    scalar_scale = term_scale(scalar_terms)
    gauss_scale = term_scale(gauss_terms)
    identity_errors = {
        "matter_Au_Bv_gw_weighted_L2_relative": weighted_norm(scalar_matter_residual) / matter_scale,
        "scalar_equation_weighted_L2_relative": weighted_norm(scalar_field_residual) / scalar_scale,
        "Gauss_K1w_source_weighted_L2_relative": weighted_norm(gauss_translation_residual) / gauss_scale,
    }
    boundary_scale = max(1.0e-30, abs(float(background.yp[0])) + abs(float(background.ap[0])) + 1.0)
    return {
        "ell": 1,
        "translation_direction": TRANSLATION_DIRECTION,
        "translation_norm": norm,
        "translation_Q_over_W0": qform,
        "translation_rayleigh_Q_over_W0": qform / max(norm, 1.0e-30),
        "translation_identity_residuals_relative_weighted_L2": identity_errors,
        "translation_identity_scales_weighted_L2": {
            "matter": matter_scale,
            "scalar": scalar_scale,
            "Gauss": gauss_scale,
        },
        "translation_boundary_values": {
            "u_at_box": float(u[-1]),
            "v_at_box": float(v[-1]),
            "w_at_box": float(w[-1]),
            "v_over_boundary_scale": float(abs(v[-1]) / boundary_scale),
            "w_over_boundary_scale": float(abs(w[-1]) / boundary_scale),
        },
        "translation_basis_in_trial_space": True,
        "translation_basis_representation": TRANSLATION_REPRESENTATION,
        "operator_positive": details["positive_operator"],
        "translation_consistency": TRANSLATION_CONSISTENCY if max(identity_errors.values()) <= TRANSLATION_RELATIVE_LIMIT else "UNRESOLVED_TRANSLATION_RESIDUAL",
    }


def _radial_energy(
    background: Background,
    y: np.ndarray,
    yp: np.ndarray,
    n: np.ndarray,
    a: np.ndarray,
    ap: np.ndarray,
) -> float:
    """Full radial energy in MeV for an angularly uniform profile."""

    design = background.design
    f = design.fermi_from_density(np.maximum(n, 0.0), np.maximum(y, 1.0e-8))["f"]
    integrand = (
        0.5 * yp**2
        + design.potential(y)
        + f
        + 0.5 * ap**2
        + 0.5 * design.q**2 * y**2 * a**2
    )
    return 4.0 * math.pi * design.W0 * _integral(background.x, background.x**2 * integrand)


def _gauss_for_profile(background: Background, y: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool]:
    solved, pivot, positive = _operator_solve(
        background.design,
        background.x,
        np.asarray(y, dtype=float),
        0,
        background.design.gomega * np.asarray(n, dtype=float),
    )
    a = solved[:, 0]
    ap = np.gradient(a, background.x, edge_order=2)
    return a, ap, pivot, positive


def _fd_test_directions(background: Background) -> list[dict[str, np.ndarray]]:
    """Two compact-interior admissible directions for an independent oracle."""

    x = background.x
    edge = max(background.edge_x, 8.0)
    b1 = _compact_bump(x, 0.28 * edge, 0.16 * edge)
    b2 = _compact_bump(x, 0.55 * edge, 0.16 * edge)
    i1 = _integral(x, x**2 * b1)
    i2 = _integral(x, x**2 * b2)
    if i1 <= 0.0 or i2 <= 0.0:
        raise StabilityError("failed to construct compact FD bumps")
    u = b1 - (i1 / i2) * b2
    u *= 0.15 * max(background.design.n0_dim, 1.0e-30) / max(float(np.max(np.abs(u))), 1.0e-30)
    scalar_bump = _compact_bump(x, 0.42 * edge, 0.20 * edge)
    scalar_bump /= max(float(np.max(np.abs(scalar_bump))), 1.0e-30)
    v = 0.10 * scalar_bump
    zero = np.zeros_like(x)
    vprime = np.gradient(v, x, edge_order=2)
    coupled_v = 0.08 * scalar_bump
    coupled_vprime = np.gradient(coupled_v, x, edge_order=2)
    return [
        {"name": "scalar_only_compact", "u": zero, "v": v, "vprime": vprime},
        {"name": "coupled_zero_integral_density_scalar", "u": u, "v": coupled_v, "vprime": coupled_vprime},
    ]


def _fd_energy_check(background: Background, direction: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Compare Q to centered finite differences with Gauss re-solves."""

    x = background.x
    scale = math.sqrt(4.0 * math.pi)
    u = np.asarray(direction["u"], dtype=float)
    v = np.asarray(direction["v"], dtype=float)
    vp = np.asarray(direction["vprime"], dtype=float)
    qform_matrix, details = _hessian_quadratic(background, 0, u, v, vp)
    analytic = float(qform_matrix[0, 0] * background.design.W0)
    a0, ap0, pivot0, positive0 = _gauss_for_profile(background, background.y, background.n)
    baseline_energy = _radial_energy(background, background.y, background.yp, background.n, a0, ap0)
    baseline_number = 4.0 * math.pi * _integral(x, x**2 * background.n)
    values = []
    for amplitude in FD_AMPLITUDES:
        plus_y = background.y + amplitude * v / scale
        minus_y = background.y - amplitude * v / scale
        plus_n = background.n + amplitude * u / scale
        minus_n = background.n - amplitude * u / scale
        admissible = bool(
            np.all(plus_y > 0.0)
            and np.all(minus_y > 0.0)
            and np.all(plus_n >= 0.0)
            and np.all(minus_n >= 0.0)
            and abs(_integral(x, x**2 * u)) <= 1.0e-8 * max(_integral(x, x**2 * np.abs(u)), 1.0e-30)
        )
        if not admissible:
            values.append({"amplitude": amplitude, "admissible": False})
            continue
        plus_a, plus_ap, plus_pivot, plus_positive = _gauss_for_profile(background, plus_y, plus_n)
        minus_a, minus_ap, minus_pivot, minus_positive = _gauss_for_profile(background, minus_y, minus_n)
        plus_number = 4.0 * math.pi * _integral(x, x**2 * plus_n)
        minus_number = 4.0 * math.pi * _integral(x, x**2 * minus_n)
        plus_energy = _radial_energy(
            background, plus_y, background.yp + amplitude * vp / scale, plus_n, plus_a, plus_ap
        )
        minus_energy = _radial_energy(
            background, minus_y, background.yp - amplitude * vp / scale, minus_n, minus_a, minus_ap
        )
        fd_value = (plus_energy + minus_energy - 2.0 * baseline_energy) / amplitude**2
        values.append({
            "amplitude": amplitude,
            "admissible": True,
            "finite_difference_Q_MeV": fd_value,
            "plus_energy_MeV": plus_energy,
            "minus_energy_MeV": minus_energy,
            "plus_gauss_pivot": plus_pivot,
            "minus_gauss_pivot": minus_pivot,
            "plus_gauss_positive": plus_positive,
            "minus_gauss_positive": minus_positive,
            "plus_number": plus_number,
            "minus_number": minus_number,
            "plus_number_relative_error": abs(plus_number - baseline_number) / max(abs(baseline_number), 1.0e-30),
            "minus_number_relative_error": abs(minus_number - baseline_number) / max(abs(baseline_number), 1.0e-30),
        })
    valid = [value for value in values if value.get("admissible")]
    if len(valid) < 2:
        return {
            "name": direction["name"],
            "analytic_Q_MeV": analytic,
            "values": values,
            "available": False,
            "reason": "fewer than two admissible amplitudes",
        }
    errors = [abs(float(value["finite_difference_Q_MeV"]) - analytic) for value in valid]
    relative = [error / max(abs(analytic), 1.0e-8) for error in errors]
    # A centered second difference should approach a common limit as the
    # amplitude shrinks; require the finest two to be closer than the coarse.
    finest_change = abs(
        float(valid[-1]["finite_difference_Q_MeV"])
        - float(valid[-2]["finite_difference_Q_MeV"])
    )
    agreement = bool(
        relative[-1] <= FD_AGREEMENT_RELATIVE_LIMIT
        and finest_change <= max(0.5 * abs(analytic) * FD_AGREEMENT_RELATIVE_LIMIT, 1.0e-6)
    )
    return {
        "name": direction["name"],
        "analytic_Q_MeV": analytic,
        "values": values,
        "errors_MeV": errors,
        "relative_errors": relative,
        "finest_amplitude_change_MeV": finest_change,
        "available": True,
        "agreement": agreement,
        "baseline_energy_MeV": baseline_energy,
        "baseline_number": baseline_number,
        "fixed_N_relative_error_max": max(
            max(float(value["plus_number_relative_error"]), float(value["minus_number_relative_error"]))
            for value in valid
        ),
        "baseline_grid_gauss_pivot": pivot0,
        "baseline_grid_gauss_positive": positive0,
        "hessian_operator_positive": details["positive_operator"],
    }


def _translation_controls_pass(translation: Mapping[str, Any]) -> bool:
    """Validate the complete joint-translation control at aggregation time."""

    if not isinstance(translation, Mapping):
        return False
    if (
        translation.get("ell") != 1
        or translation.get("translation_direction") != TRANSLATION_DIRECTION
        or translation.get("translation_basis_in_trial_space") is not True
        or translation.get("operator_positive") is not True
        or translation.get("translation_basis_representation") != TRANSLATION_REPRESENTATION
        or translation.get("translation_consistency") != TRANSLATION_CONSISTENCY
        or not _finite_scalar(translation.get("translation_norm"))
        or float(translation.get("translation_norm", 0.0)) <= 0.0
        or not _finite_scalar(translation.get("translation_Q_over_W0"))
        or not _finite_scalar(translation.get("translation_rayleigh_Q_over_W0"))
    ):
        return False
    residuals = translation.get("translation_identity_residuals_relative_weighted_L2")
    scales = translation.get("translation_identity_scales_weighted_L2")
    if (
        not isinstance(residuals, Mapping)
        or set(residuals) != {
            "matter_Au_Bv_gw_weighted_L2_relative",
            "scalar_equation_weighted_L2_relative",
            "Gauss_K1w_source_weighted_L2_relative",
        }
        or not isinstance(scales, Mapping)
        or set(scales) != {"matter", "scalar", "Gauss"}
        or not all(
            _finite_scalar(value) and float(value) <= TRANSLATION_RELATIVE_LIMIT
            for value in residuals.values()
        )
        or not all(_finite_scalar(value) and float(value) > 0.0 for value in scales.values())
    ):
        return False
    boundary = translation.get("translation_boundary_values")
    if (
        not isinstance(boundary, Mapping)
        or not all(
            _finite_scalar(boundary.get(key))
            for key in ("u_at_box", "v_at_box", "w_at_box", "v_over_boundary_scale", "w_over_boundary_scale")
        )
    ):
        return False
    return True


def _fd_check_pass(check: Mapping[str, Any], name: str) -> bool:
    """Require one complete prescribed finite-difference trajectory."""

    if not isinstance(check, Mapping):
        return False
    values = check.get("values")
    errors = check.get("errors_MeV")
    relative_errors = check.get("relative_errors")
    for key in (
        "analytic_Q_MeV",
        "finest_amplitude_change_MeV",
        "baseline_energy_MeV",
        "baseline_number",
    ):
        if not _finite_scalar(check.get(key)):
            return False
    if (
        check.get("name") != name
        or check.get("available") is not True
        or check.get("agreement") is not True
        or check.get("hessian_operator_positive") is not True
        or check.get("baseline_grid_gauss_positive") is not True
        or not _finite_scalar(check.get("baseline_grid_gauss_pivot"))
        or float(check.get("baseline_grid_gauss_pivot", 0.0)) <= 0.0
        or not _finite_scalar(check.get("fixed_N_relative_error_max"))
        or float(check.get("fixed_N_relative_error_max", math.inf)) > FIXED_N_RELATIVE_LIMIT
        or not isinstance(values, list)
        or len(values) != len(FD_AMPLITUDES)
        or not isinstance(errors, list)
        or len(errors) != len(values)
        or not all(_finite_scalar(value) and float(value) >= 0.0 for value in errors)
        or not isinstance(relative_errors, list)
        or len(relative_errors) != len(values)
        or not all(_finite_scalar(value) and float(value) >= 0.0 for value in relative_errors)
        or max(relative_errors, default=math.inf) > FD_AGREEMENT_RELATIVE_LIMIT
        or float(check["finest_amplitude_change_MeV"]) > max(
            0.5 * abs(float(check["analytic_Q_MeV"])) * FD_AGREEMENT_RELATIVE_LIMIT,
            1.0e-6,
        )
    ):
        return False
    analytic = float(check["analytic_Q_MeV"])
    expected_scale = max(abs(analytic), 1.0e-8)
    for value, amplitude in zip(values, FD_AMPLITUDES):
        if (
            not isinstance(value, Mapping)
            or value.get("amplitude") != amplitude
            or value.get("admissible") is not True
            or value.get("plus_gauss_positive") is not True
            or value.get("minus_gauss_positive") is not True
            or not _finite_scalar(value.get("finite_difference_Q_MeV"))
            or not _finite_scalar(value.get("plus_energy_MeV"))
            or not _finite_scalar(value.get("minus_energy_MeV"))
            or not _finite_scalar(value.get("plus_number"))
            or not _finite_scalar(value.get("minus_number"))
            or not _finite_scalar(value.get("plus_number_relative_error"))
            or not _finite_scalar(value.get("minus_number_relative_error"))
            or not _finite_scalar(value.get("plus_gauss_pivot"))
            or not _finite_scalar(value.get("minus_gauss_pivot"))
            or float(value.get("plus_gauss_pivot", 0.0)) <= 0.0
            or float(value.get("minus_gauss_pivot", 0.0)) <= 0.0
            or float(value.get("plus_number_relative_error", math.inf)) > FIXED_N_RELATIVE_LIMIT
            or float(value.get("minus_number_relative_error", math.inf)) > FIXED_N_RELATIVE_LIMIT
        ):
            return False
    expected_errors = [
        abs(float(value["finite_difference_Q_MeV"]) - analytic)
        for value in values
    ]
    expected_relative = [error / expected_scale for error in expected_errors]
    if any(
        abs(float(actual) - expected) > 1.0e-12 * max(1.0, expected)
        for actual, expected in zip(errors, expected_errors)
    ) or any(
        abs(float(actual) - expected) > 1.0e-12 * max(1.0, expected)
        for actual, expected in zip(relative_errors, expected_relative)
    ):
        return False
    expected_fixed_n = max(
        max(float(value["plus_number_relative_error"]), float(value["minus_number_relative_error"]))
        for value in values
    )
    if abs(float(check["fixed_N_relative_error_max"]) - expected_fixed_n) > 1.0e-12 * max(1.0, expected_fixed_n):
        return False
    expected_finest_change = abs(
        float(values[-1]["finite_difference_Q_MeV"])
        - float(values[-2]["finite_difference_Q_MeV"])
    )
    if abs(float(check["finest_amplitude_change_MeV"]) - expected_finest_change) > 1.0e-12 * max(1.0, expected_finest_change):
        return False
    return True


def _spectrum_controls_pass(sector: Mapping[str, Any], ell: int) -> bool:
    """Check mandatory spectrum coverage and record controls before conclusions."""

    if not isinstance(sector, Mapping):
        return False
    # The live assembly contains NumPy scalar ranks and booleans until the
    # result is serialized.  Validate the same canonical JSON-shaped record
    # that the CLI and public validator will receive, without mutating the
    # producer-owned object.
    try:
        sector = _jsonable(sector)
    except (ArithmeticError, StabilityError, TypeError, ValueError):
        return False
    basis_sweep = sector.get("basis_sweep")
    grid_sweep = sector.get("grid_sweep_largest_basis")
    domain_sweep = sector.get("domain_sweep_largest_basis")
    tighter = sector.get("tighter_background_same_domain")
    terminal = sector.get("terminal_largest_basis")
    if (
        not isinstance(basis_sweep, list)
        or len(basis_sweep) != len(BASIS_SIZES)
        or not isinstance(grid_sweep, list)
        or len(grid_sweep) != len(GRID_INTERVALS)
        or not isinstance(domain_sweep, list)
        or len(domain_sweep) != len(DOMAIN_BOXES_FM)
        or not isinstance(tighter, Mapping)
        or not isinstance(terminal, Mapping)
        or [row.get("basis_size_each_block") for row in basis_sweep if isinstance(row, Mapping)] != list(BASIS_SIZES)
        or [row.get("intervals") for row in grid_sweep if isinstance(row, Mapping)] != list(GRID_INTERVALS)
        or [row.get("box_fm") for row in domain_sweep if isinstance(row, Mapping)] != list(DOMAIN_BOXES_FM)
    ):
        return False
    if not all(
        _validate_spectrum_record(row, ell, size, FINE_INTERVALS, DOMAIN_BOXES_FM[0])
        for row, size in zip(basis_sweep, BASIS_SIZES)
    ):
        return False
    if not all(
        _validate_spectrum_record(row, ell, BASIS_SIZES[-1], intervals, DOMAIN_BOXES_FM[0])
        for row, intervals in zip(grid_sweep, GRID_INTERVALS)
    ):
        return False
    if not all(
        _validate_spectrum_record(row, ell, BASIS_SIZES[-1], FINE_INTERVALS, box)
        for row, box in zip(domain_sweep, DOMAIN_BOXES_FM)
    ):
        return False
    if (
        not _validate_spectrum_record(tighter, ell, BASIS_SIZES[-1], FINE_INTERVALS, DOMAIN_BOXES_FM[0])
        or not _validate_spectrum_record(terminal, ell, BASIS_SIZES[-1], FINE_INTERVALS, DOMAIN_BOXES_FM[1])
        or terminal != domain_sweep[-1]
    ):
        return False
    expected_key = "smallest_internal_eigenvalues_Q_over_W0" if ell == 1 else "smallest_eigenvalues_Q_over_W0"
    monotonicity = sector.get("rayleigh_ritz_monotonicity")
    comparisons = monotonicity.get("comparisons") if isinstance(monotonicity, Mapping) else None
    if (
        sector.get("spectrum_key_for_conclusion") != expected_key
        or not isinstance(monotonicity, Mapping)
        or monotonicity.get("key") != expected_key
        or monotonicity.get("monotone_with_allowance") is not True
        or not isinstance(comparisons, list)
        or len(comparisons) != len(BASIS_SIZES) - 1
        or not all(isinstance(item, Mapping) and item.get("passes") is True for item in comparisons)
        or sector.get("basis_and_grid_domain_convergence") not in {True, False}
    ):
        return False
    return True


def _case_result(
    target_y: str,
    spectrum_transform: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    background24, background32, background_tight24 = _backgrounds(target_y)
    if spectrum_transform is None:
        sectors = {
            str(ell): _spectrum_sector(background24, background32, background_tight24, ell)
            for ell in ELL_VALUES
        }
    else:
        sectors = {
            str(ell): _spectrum_sector(
                background24,
                background32,
                background_tight24,
                ell,
                spectrum_transform=spectrum_transform,
            )
            for ell in ELL_VALUES
        }
    translation = _translation_metrics(background24)
    fd_checks = [_fd_energy_check(background24, direction) for direction in _fd_test_directions(background24)]
    all_planned_ell_sectors_reported = (
        len(sectors) == len(ELL_VALUES)
        and set(sectors) == {str(ell) for ell in ELL_VALUES}
    )
    all_spectrum_available = bool(
        all_planned_ell_sectors_reported
        and all(
            isinstance(sector, Mapping) and isinstance(sector.get("terminal_largest_basis"), Mapping)
            for sector in sectors.values()
        )
    )
    fd_names = [check.get("name") if isinstance(check, Mapping) else None for check in fd_checks]
    fd_controls_pass = bool(
        isinstance(fd_checks, list)
        and len(fd_checks) == len(FD_DIRECTION_NAMES)
        and fd_names == list(FD_DIRECTION_NAMES)
        and all(_fd_check_pass(check, name) for check, name in zip(fd_checks, FD_DIRECTION_NAMES))
    )
    fixed_n_checks_pass = bool(fd_controls_pass)
    gauss_checks_pass = bool(fd_controls_pass)
    checks_by_name = {
        check.get("name"): check
        for check in fd_checks
        if isinstance(check, Mapping)
    }
    scalar_check = checks_by_name.get("scalar_only_compact", {})
    translation_controls_pass = _translation_controls_pass(translation)
    control_failures: list[str] = []
    background_summaries = (background24.summary(), background32.summary(), background_tight24.summary())
    if len(background_summaries) != 3 or not all(
        isinstance(background, Mapping) and background.get("converged") is True
        for background in background_summaries
    ):
        control_failures.append("background_convergence")
    if not fixed_n_checks_pass:
        control_failures.append("fixed_N")
    if not gauss_checks_pass:
        control_failures.append("Gauss_re_solve")
    if not fd_controls_pass:
        control_failures.append("energy_difference_paths")
    if not translation_controls_pass:
        control_failures.append("translation_identity")
    if not all_planned_ell_sectors_reported:
        control_failures.append("ell_sector_coverage")
    if not all_spectrum_available:
        control_failures.append("spectrum_coverage")
    for ell, sector in sectors.items():
        try:
            ell_value = int(ell)
        except (TypeError, ValueError):
            control_failures.append(f"spectrum_l{ell}")
            continue
        if not _spectrum_controls_pass(sector, ell_value):
            control_failures.append(f"spectrum_l{ell_value}")
            continue
        records = (
            sector["basis_sweep"]
            + sector["grid_sweep_largest_basis"]
            + sector["domain_sweep_largest_basis"]
            + [sector["tighter_background_same_domain"]]
        )
        if not records or not all(record.get("positive_operator") is True for record in records):
            control_failures.append(f"operator_l{ell}")
        if any(not _finite_scalar(record.get("eigen_residual_absolute_max_Q_over_W0"))
               or not _finite_scalar(record.get("internal_eigen_residual_absolute_max_Q_over_W0"))
               for record in records):
            control_failures.append(f"eigen_residual_l{ell}")
        if ell == "0" and any(
            not _finite_scalar(record.get("number_constraint_residual_max"))
            or record.get("number_constraint_residual_max", math.inf) > NUMBER_CONSTRAINT_RELATIVE_LIMIT
            for record in records
        ):
            control_failures.append("number_constraint_l0")
        if ell == "1" and any(
            record.get("projection_applied") is not True
            or record.get("projection_removed_dimension") != 1
            or record.get("projection_basis_dimension") != record.get("basis_dimension", 0) - 1
            or record.get("projection_orthogonality_relative", math.inf) > PROJECTION_ORTHOGONALITY_LIMIT
            for record in records
        ):
            control_failures.append("translation_projection")
    terminal_controls_pass = not control_failures
    # A sector with an incomplete refinement control cannot retain either a
    # positive-only label or a negative-witness label.  Global control faults
    # similarly invalidate every physical classification while preserving the
    # raw diagnostics for the failure report.
    for sector in sectors.values():
        if (
            not terminal_controls_pass
            or not isinstance(sector, Mapping)
            or sector.get("basis_and_grid_domain_convergence") is not True
        ) and isinstance(sector, dict):
            if sector.get("conclusion") in {
                "FINITE_BASIS_POSITIVE_ONLY",
                "FINITE_BASIS_ROBUST_NEGATIVE_DIRECTION",
            }:
                sector["conclusion"] = "UNRESOLVED_FINITE_BASIS_SIGN_OR_CONVERGENCE"
    max_relative_number_error = max(
        (
            float(check.get("fixed_N_relative_error_max", math.inf))
            for check in fd_checks
            if isinstance(check, Mapping) and _finite_scalar(check.get("fixed_N_relative_error_max"))
        ),
        default=math.inf,
    )
    return {
        "target_y": target_y,
        "target_N": TARGET_N,
        "backgrounds": {
            "24fm": background24.summary(),
            "32fm": background32.summary(),
            "24fm_tighter": background_tight24.summary(),
            "background_grid_intervals": 3200,
            "background_nodes_requested": BACKGROUND_NODES,
            "background_tolerance": BACKGROUND_TOLERANCE,
            "background_tight_nodes_requested": BACKGROUND_TIGHT_NODES,
            "background_tight_tolerance": BACKGROUND_TIGHT_TOLERANCE,
        },
        "sectors": sectors,
        "translation_metrics_24fm": translation,
        "independent_energy_difference_checks_24fm": fd_checks,
        "terminal_consistency_checks": {
            "fixed_N_additive_paths": {
                "relative_number_limit": FIXED_N_RELATIVE_LIMIT,
                "all_checks_pass": fixed_n_checks_pass,
                "max_relative_number_error": max_relative_number_error,
            },
            "global_Gauss_re_solves": {
                "all_checks_pass": gauss_checks_pass,
                "all_profiles_re_solved_positive": gauss_checks_pass,
            },
            "scalar_only_energy_difference": {
                "available": scalar_check.get("available") is True,
                "agreement": scalar_check.get("agreement") is True,
            },
        },
        "terminal_controls_pass": terminal_controls_pass,
        "terminal_control_failures": control_failures,
        "all_planned_ell_sectors_reported": all_planned_ell_sectors_reported,
        "all_spectrum_records_available": all_spectrum_available,
        "scope_conclusion": "FINITE_BASIS_STATIC_ONLY",
    }


def calculate() -> dict[str, Any]:
    cases = {target: _case_result(target) for target in TARGETS}
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "model_scope": {
            "zero_temperature_symmetric_coulomb_free": True,
            "accepted_backgrounds_only": True,
            "background_target_N": TARGET_N,
            "ell_sectors": list(ELL_VALUES),
            "basis_sizes_each_block": list(BASIS_SIZES),
            "grid_intervals": list(GRID_INTERVALS),
            "domain_boxes_fm": list(DOMAIN_BOXES_FM),
            "no_new_interactions_or_parameters": True,
            "curvatures_are_not_frequencies": True,
            "positive_finite_basis_is_not_full_stability": True,
        },
        "hessian_contract": {
            "formula": "Q_l/W0=int x^2[A u^2+2Buv+C v^2+v'^2+ell(ell+1)v^2/x^2]+<s,K_l^-1 s>",
            "A": "F_nn=k^2/(3*n*e) on occupied cells; no density floor",
            "B": "F_ny=gs^2*y/e",
            "C": "F_yy+U_yy-q^2*a^2",
            "F_yy": "gs^2*d/(2*pi^2)*int_0^k p^4/(p^2+(gs*y)^2)^(3/2) dp",
            "source": "s=g*u-2*q^2*y*a*v",
            "operator": "K_l=-x^-2*d_x(x^2*d_x)+ell(ell+1)/x^2+q^2*y^2",
            "harmonic_normalization": "integral |Y_lm|^2 dOmega=1; no extra 4*pi in Q",
            "normalization": "int x^2[(u/n0_dim)^2+v^2] dx",
            "positive_nonlocal_term": True,
            "translation_joint_mode_in_trial": True,
            "ell0_number_constraint": "edge-aware |int x^2 u|/int x^2|u| <= 1e-6 with explicit zero-vector handling",
            "ell1_internal_projection": "P spans null(t^T G) for t=(1,0,...;1,0,...), exactly one known joint translation direction",
            "eigen_residual_units": "absolute residual after Gram whitening, Q/W0",
        },
        "acceptance_limits": {
            "gram_relative_cutoff": GRAM_RELATIVE_CUTOFF,
            "sign_error_multiplier": SIGN_ERROR_MULTIPLIER,
            "basis_relative_drift": BASIS_RELATIVE_DRIFT_LIMIT,
            "grid_domain_relative_drift": DISCRETIZATION_RELATIVE_DRIFT_LIMIT,
            "fd_agreement_relative": FD_AGREEMENT_RELATIVE_LIMIT,
            "fixed_N_relative": FIXED_N_RELATIVE_LIMIT,
            "number_constraint_relative": NUMBER_CONSTRAINT_RELATIVE_LIMIT,
            "projection_orthogonality_relative": PROJECTION_ORTHOGONALITY_LIMIT,
            "translation_identity_relative": TRANSLATION_RELATIVE_LIMIT,
            "background_tight_nodes": BACKGROUND_TIGHT_NODES,
            "background_tight_tolerance": BACKGROUND_TIGHT_TOLERANCE,
        },
        "input_provenance": {
            "finite_droplet_producer": "verification/nvg_finite_droplet_audit.py:W8Design,_solve_once",
            "authorized_upstream_derivative_repair": "W8Design.potential_yy only; U, Uy and calibration parameters unchanged",
            "finite_droplet_source_sha256": hashlib.sha256(
                (HERE / "nvg_finite_droplet_audit.py").read_bytes()
            ).hexdigest(),
            "no_saved_stability_result_used_as_input": True,
        },
        "cases": cases,
    }


_CACHE: dict[str, Any] | None = None


def build_result() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _jsonable(calculate())
    # Keep the trusted producer cache private.  Callers frequently use result
    # copies for fault injection; mutating one must never poison later builds.
    return copy.deepcopy(_CACHE)


def _validate_spectrum_record(
    row: Mapping[str, Any],
    ell: int,
    basis_size: int,
    intervals: int,
    box_fm: float,
) -> bool:
    """Validate one complete spectral record, including all numerical guards."""

    if not isinstance(row, Mapping):
        return False
    if (
        row.get("ell") != ell
        or row.get("basis_size_each_block") != basis_size
        or row.get("basis_dimension") != 2 * basis_size
        or row.get("intervals") != intervals
        or row.get("box_fm") != box_fm
        or row.get("positive_operator") is not True
    ):
        return False
    raw_values = row.get("eigenvalues_Q_over_W0")
    internal_values = row.get("internal_eigenvalues_Q_over_W0")
    raw_residuals = row.get("eigen_residual_absolute_Q_over_W0")
    internal_residuals = row.get("internal_eigen_residual_absolute_Q_over_W0")
    raw_rank = row.get("gram_rank")
    raw_size = row.get("gram_size")
    internal_rank = row.get("internal_gram_rank")
    internal_size = row.get("internal_gram_size")
    if (
        not isinstance(raw_values, list)
        or not isinstance(internal_values, list)
        or not isinstance(raw_residuals, list)
        or not isinstance(internal_residuals, list)
        or not isinstance(raw_rank, int)
        or isinstance(raw_rank, bool)
        or not isinstance(raw_size, int)
        or isinstance(raw_size, bool)
        or not isinstance(internal_rank, int)
        or isinstance(internal_rank, bool)
        or not isinstance(internal_size, int)
        or isinstance(internal_size, bool)
        or raw_size != 2 * basis_size
        or len(raw_values) != raw_rank
        or len(raw_residuals) != raw_rank
        or len(internal_values) != internal_rank
        or len(internal_residuals) != internal_rank
        or not (0 < raw_rank <= raw_size)
        or not (0 < internal_rank <= internal_size)
        or not all(_finite_scalar(value) for value in raw_values + internal_values)
        or not all(_finite_scalar(value) and float(value) >= 0.0 for value in raw_residuals + internal_residuals)
    ):
        return False
    raw_loss = row.get("gram_rank_loss")
    internal_loss = row.get("internal_gram_rank_loss")
    if (
        not isinstance(raw_loss, int)
        or isinstance(raw_loss, bool)
        or raw_loss != raw_size - raw_rank
        or not isinstance(internal_loss, int)
        or isinstance(internal_loss, bool)
        or internal_loss != internal_size - internal_rank
        or row.get("rank_truncated") is not (raw_loss > 0)
        or row.get("internal_rank_truncated") is not (internal_loss > 0)
    ):
        return False
    for key in (
        "eigen_residual_absolute_max_Q_over_W0",
        "eigen_residual_relative_max",
        "internal_eigen_residual_absolute_max_Q_over_W0",
        "gram_condition_number",
        "internal_gram_condition_number",
        "operator_min_pivot",
    ):
        if not _finite_scalar(row.get(key)):
            return False
    if (
        abs(float(row["eigen_residual_absolute_max_Q_over_W0"]) - max(map(float, raw_residuals), default=0.0))
        > 1.0e-12 * max(1.0, abs(float(row["eigen_residual_absolute_max_Q_over_W0"])))
        or abs(float(row["internal_eigen_residual_absolute_max_Q_over_W0"]) - max(map(float, internal_residuals), default=0.0))
        > 1.0e-12 * max(1.0, abs(float(row["internal_eigen_residual_absolute_max_Q_over_W0"])))
    ):
        return False
    projection_applied = row.get("projection_applied")
    expected_projection = ell == 1
    if (
        projection_applied is not expected_projection
        or row.get("translation_basis_present") is not expected_projection
        or row.get("projection_removed_dimension") != (1 if expected_projection else 0)
        or row.get("projection_basis_dimension") != (2 * basis_size - 1 if expected_projection else 2 * basis_size)
        or row.get("projected_gram_size") != internal_size
        or row.get("projected_gram_rank") != internal_rank
        or not _finite_scalar(row.get("projection_orthogonality_relative"))
        or float(row["projection_orthogonality_relative"]) > PROJECTION_ORTHOGONALITY_LIMIT
    ):
        return False
    if expected_projection:
        if (
            not _finite_scalar(row.get("excluded_vector_gram_norm"))
            or float(row["excluded_vector_gram_norm"]) <= 0.0
            or not _finite_scalar(row.get("translation_cutoff_min"))
        ):
            return False
        if internal_size != 2 * basis_size - 1:
            return False
    elif internal_size != 2 * basis_size:
        return False
    constraints = row.get("number_constraint_residuals_relative")
    denominators = row.get("number_constraint_denominators")
    if (
        not isinstance(constraints, list)
        or not isinstance(denominators, list)
        or len(constraints) != 2 * basis_size
        or len(denominators) != 2 * basis_size
        or not all(_finite_scalar(value) and float(value) >= 0.0 for value in constraints + denominators)
        or not _finite_scalar(row.get("number_constraint_residual_max"))
        or float(row["number_constraint_residual_max"]) < 0.0
        or not isinstance(row.get("number_constraint_zero_vector_count"), int)
        or isinstance(row.get("number_constraint_zero_vector_count"), bool)
        or row.get("number_constraint_zero_vector_count")
        != sum(float(value) <= 1.0e-30 for value in denominators)
        or abs(float(row["number_constraint_residual_max"]) - max(map(float, constraints), default=0.0))
        > 1.0e-12 * max(1.0, abs(float(row["number_constraint_residual_max"])))
    ):
        return False
    expected_mode = "radial_integral" if ell == 0 else "angular_integral_zero"
    if row.get("number_constraint_mode") != expected_mode:
        return False
    if ell == 0 and float(row["number_constraint_residual_max"]) > NUMBER_CONSTRAINT_RELATIVE_LIMIT:
        return False
    return True


def _validate_result_shape(result: Mapping[str, Any]) -> bool:
    if not isinstance(result, Mapping):
        return False
    if (
        result.get("schema_version") != SCHEMA
        or result.get("status") != STATUS
        or result.get("evidence_weight") != 0.0
    ):
        return False
    limits = result.get("acceptance_limits")
    expected_limits = {
        "gram_relative_cutoff": GRAM_RELATIVE_CUTOFF,
        "sign_error_multiplier": SIGN_ERROR_MULTIPLIER,
        "basis_relative_drift": BASIS_RELATIVE_DRIFT_LIMIT,
        "grid_domain_relative_drift": DISCRETIZATION_RELATIVE_DRIFT_LIMIT,
        "fd_agreement_relative": FD_AGREEMENT_RELATIVE_LIMIT,
        "fixed_N_relative": FIXED_N_RELATIVE_LIMIT,
        "number_constraint_relative": NUMBER_CONSTRAINT_RELATIVE_LIMIT,
        "projection_orthogonality_relative": PROJECTION_ORTHOGONALITY_LIMIT,
        "translation_identity_relative": TRANSLATION_RELATIVE_LIMIT,
        "background_tight_nodes": BACKGROUND_TIGHT_NODES,
        "background_tight_tolerance": BACKGROUND_TIGHT_TOLERANCE,
    }
    if not isinstance(limits, Mapping) or any(limits.get(key) != value for key, value in expected_limits.items()):
        return False
    cases = result.get("cases")
    if not isinstance(cases, Mapping):
        return False
    for target in TARGETS:
        case = cases.get(target)
        if not isinstance(case, Mapping):
            return False
        if case.get("target_y") != target or case.get("target_N") != TARGET_N:
            return False
        backgrounds = case.get("backgrounds")
        if not isinstance(backgrounds, Mapping):
            return False
        for box in ("24fm", "32fm", "24fm_tighter"):
            background = backgrounds.get(box)
            if (
                not isinstance(background, Mapping)
                or background.get("converged") is not True
                or not _finite_scalar(background.get("chemical_potential_MeV"))
                or not _finite_scalar(background.get("conserved_N_relative_error"))
                or not _finite_scalar(background.get("field_residual_relative"))
                or not _finite_scalar(background.get("virial_relative"))
                or not _finite_scalar(background.get("gauss_energy_identity_relative"))
                or not _finite_scalar(background.get("edge_x"))
            ):
                return False
        if (
            backgrounds.get("background_grid_intervals") != 3200
            or backgrounds.get("background_nodes_requested") != BACKGROUND_NODES
            or backgrounds.get("background_tolerance") != BACKGROUND_TOLERANCE
            or backgrounds.get("background_tight_nodes_requested") != BACKGROUND_TIGHT_NODES
            or backgrounds.get("background_tight_tolerance") != BACKGROUND_TIGHT_TOLERANCE
        ):
            return False
        sectors = case.get("sectors")
        if not isinstance(sectors, Mapping) or set(sectors) != {str(ell) for ell in ELL_VALUES}:
            return False
        if case.get("all_planned_ell_sectors_reported") is not True:
            return False
        if case.get("all_spectrum_records_available") is not True:
            return False
        for ell in ELL_VALUES:
            sector = sectors.get(str(ell))
            if not isinstance(sector, Mapping):
                return False
            basis_sweep = sector.get("basis_sweep")
            grid_sweep = sector.get("grid_sweep_largest_basis")
            domain_sweep = sector.get("domain_sweep_largest_basis")
            if not isinstance(basis_sweep, list) or len(basis_sweep) != len(BASIS_SIZES):
                return False
            if not isinstance(grid_sweep, list) or len(grid_sweep) != len(GRID_INTERVALS):
                return False
            if not isinstance(domain_sweep, list) or len(domain_sweep) != len(DOMAIN_BOXES_FM):
                return False
            if not all(isinstance(row, Mapping) for row in basis_sweep + grid_sweep + domain_sweep):
                return False
            if [row.get("basis_size_each_block") for row in basis_sweep] != list(BASIS_SIZES):
                return False
            if [row.get("intervals") for row in grid_sweep] != list(GRID_INTERVALS):
                return False
            if [row.get("box_fm") for row in domain_sweep] != list(DOMAIN_BOXES_FM):
                return False
            if not all(
                _validate_spectrum_record(row, ell, size, FINE_INTERVALS, DOMAIN_BOXES_FM[0])
                for row, size in zip(basis_sweep, BASIS_SIZES)
            ):
                return False
            if not all(
                _validate_spectrum_record(row, ell, BASIS_SIZES[-1], intervals, DOMAIN_BOXES_FM[0])
                for row, intervals in zip(grid_sweep, GRID_INTERVALS)
            ):
                return False
            if not all(
                _validate_spectrum_record(row, ell, BASIS_SIZES[-1], FINE_INTERVALS, box)
                for row, box in zip(domain_sweep, DOMAIN_BOXES_FM)
            ):
                return False
            tighter = sector.get("tighter_background_same_domain")
            if not _validate_spectrum_record(
                tighter, ell, BASIS_SIZES[-1], FINE_INTERVALS, DOMAIN_BOXES_FM[0]
            ):
                return False
            expected_key = (
                "smallest_internal_eigenvalues_Q_over_W0"
                if ell == 1
                else "smallest_eigenvalues_Q_over_W0"
            )
            if sector.get("spectrum_key_for_conclusion") != expected_key:
                return False
            monotonicity = sector.get("rayleigh_ritz_monotonicity")
            comparisons = monotonicity.get("comparisons") if isinstance(monotonicity, Mapping) else None
            if (
                not isinstance(monotonicity, Mapping)
                or monotonicity.get("key") != expected_key
                or monotonicity.get("monotone_with_allowance") is not True
                or not _finite_scalar(monotonicity.get("max_upward_violation_Q_over_W0"))
                or not _finite_scalar(monotonicity.get("numerical_allowance_Q_over_W0"))
                or not isinstance(comparisons, list)
                or len(comparisons) != len(BASIS_SIZES) - 1
                or not all(isinstance(item, Mapping) for item in comparisons)
                or not all(item.get("passes") is True for item in comparisons)
                or not all(
                    _finite_scalar(item.get("max_upward_violation_Q_over_W0"))
                    and _finite_scalar(item.get("allowance_Q_over_W0"))
                    for item in comparisons
                )
            ):
                return False
            for left, right, item in zip(basis_sweep[:-1], basis_sweep[1:], comparisons):
                if (
                    item.get("from_basis_size") != left.get("basis_size_each_block")
                    or item.get("to_basis_size") != right.get("basis_size_each_block")
                ):
                    return False
            terminal = sector.get("terminal_largest_basis")
            if not isinstance(terminal, Mapping):
                return False
            if terminal.get("ell") != ell or terminal.get("basis_size_each_block") != BASIS_SIZES[-1]:
                return False
            if not _validate_spectrum_record(
                terminal, ell, BASIS_SIZES[-1], FINE_INTERVALS, DOMAIN_BOXES_FM[1]
            ) or terminal != domain_sweep[-1]:
                return False
            if (
                not _finite_scalar(sector.get("basis_delta_max"))
                or not _finite_scalar(sector.get("grid_delta_max"))
                or not _finite_scalar(sector.get("domain_delta_max"))
                or not _finite_scalar(sector.get("background_delta_max"))
                or not _finite_scalar(sector.get("background_relative_drift_max"))
                or not _finite_scalar(sector.get("measured_uncertainty_conservative_base"))
                or float(sector.get("measured_uncertainty_conservative_base", 0.0)) <= 0.0
                or sector.get("sign_error_multiplier") != SIGN_ERROR_MULTIPLIER
                or not isinstance(sector.get("gram_rank_loss_significant"), bool)
                or sector.get("gram_rank_loss_significant")
                != any(
                    row.get("gram_rank_loss", 0) > 0
                    or row.get("internal_gram_rank_loss", 0) > 0
                    for row in basis_sweep + grid_sweep + domain_sweep + [tighter]
                )
                or not isinstance(sector.get("basis_and_grid_domain_convergence"), bool)
            ):
                return False
            terminal_values = terminal.get(expected_key)
            signs = sector.get("terminal_signs")
            margins = sector.get("terminal_sign_margins")
            resolved = sector.get("terminal_sign_resolved")
            uncertainty = float(sector["measured_uncertainty_conservative_base"])
            if (
                not isinstance(terminal_values, list)
                or not isinstance(signs, list)
                or not isinstance(margins, list)
                or not isinstance(resolved, list)
                or len(signs) != len(terminal_values)
                or len(margins) != len(terminal_values)
                or len(resolved) != len(terminal_values)
                or not all(_finite_scalar(item) for item in margins)
                or not all(isinstance(item, bool) for item in resolved)
            ):
                return False
            expected_signs = [
                "positive" if float(value) > 0.0 else "negative" if float(value) < 0.0 else "zero"
                for value in terminal_values
            ]
            expected_margins = [
                abs(float(value)) / (SIGN_ERROR_MULTIPLIER * uncertainty)
                for value in terminal_values
            ]
            if (
                signs != expected_signs
                or resolved != [margin >= 1.0 for margin in expected_margins]
                or any(abs(float(actual) - expected) > 1.0e-12 * max(1.0, expected)
                       for actual, expected in zip(margins, expected_margins))
            ):
                return False
            if sector.get("conclusion") not in {
                "FINITE_BASIS_POSITIVE_ONLY",
                "FINITE_BASIS_ROBUST_NEGATIVE_DIRECTION",
                "UNRESOLVED_FINITE_BASIS_SIGN_OR_CONVERGENCE",
            }:
                return False
            if sector.get("conclusion") == "FINITE_BASIS_ROBUST_NEGATIVE_DIRECTION":
                signs = sector.get("terminal_signs")
                resolved = sector.get("terminal_sign_resolved")
                if not isinstance(signs, list) or not isinstance(resolved, list):
                    return False
                if not any(sign == "negative" and flag is True for sign, flag in zip(signs, resolved)):
                    return False
        if case.get("terminal_controls_pass") is not True:
            return False
        if case.get("terminal_control_failures") != []:
            return False
        fd_checks = case.get("independent_energy_difference_checks_24fm")
        fd_names = [check.get("name") if isinstance(check, Mapping) else None for check in fd_checks] \
            if isinstance(fd_checks, list) else []
        if (
            not isinstance(fd_checks, list)
            or len(fd_checks) != len(FD_DIRECTION_NAMES)
            or fd_names != list(FD_DIRECTION_NAMES)
            or not all(
                _fd_check_pass(check, name)
                for check, name in zip(fd_checks, FD_DIRECTION_NAMES)
            )
        ):
            return False
        translation = case.get("translation_metrics_24fm")
        if not _translation_controls_pass(translation):
            return False
        checks = case.get("terminal_consistency_checks")
        fixed_checks = checks.get("fixed_N_additive_paths") if isinstance(checks, Mapping) else None
        gauss_checks = checks.get("global_Gauss_re_solves") if isinstance(checks, Mapping) else None
        scalar_check = checks.get("scalar_only_energy_difference") if isinstance(checks, Mapping) else None
        if (
            not isinstance(checks, Mapping)
            or not isinstance(fixed_checks, Mapping)
            or not isinstance(gauss_checks, Mapping)
            or not isinstance(scalar_check, Mapping)
            or fixed_checks.get("all_checks_pass") is not True
            or gauss_checks.get("all_checks_pass") is not True
            or scalar_check.get("available") is not True
            or scalar_check.get("agreement") is not True
        ):
            return False
    return True


def validate_result(result: Mapping[str, Any]) -> bool:
    if not isinstance(result, Mapping):
        return False
    try:
        if not _validate_result_shape(result):
            return False
        fresh = build_result()
        return _validate_result_shape(fresh) and result == fresh
    except (ArithmeticError, AttributeError, IndexError, KeyError, StabilityError, TypeError, ValueError):
        return False


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - parser path.
        raise StabilityError(message)


def main(argv: list[str] | None = None) -> int:
    parser = JsonArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = build_result()
        if not _validate_result_shape(result):
            raise StabilityError("invalid or incomplete droplet stability result")
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    except (ArithmeticError, AttributeError, IndexError, KeyError, RuntimeError, StabilityError, TypeError, ValueError) as exc:
        print(json.dumps({
            "schema_version": SCHEMA,
            "status": "INVALID_OR_FAILED_DROPLET_STABILITY_AUDIT",
            "evidence_weight": EVIDENCE_WEIGHT,
            "error": str(exc),
        }, ensure_ascii=False, allow_nan=False))
        return 2
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
