#!/usr/bin/env python3
"""Finite-N spherical W8 Thomas--Fermi droplet audit.

This is a new, deliberately bounded numerical surface for the two already
declared inverse W8 designs ``y*=0.90`` and ``y*=0.93``.  The producer
``source_complete_scaling_saturation_audit.inverse_potential_jet`` is read
only: its density, binding and incompressibility inputs are calibration
inputs, not predictions.

The radial calculation uses ``x=W0*r/(hbar*c)`` and ``a=A/W0``.  It solves

    y''+2 y'/x = Uy/W0^4 + (M/W0) ns/W0^3 - q^2*y*a^2,
    a''+2 a'/x = q^2*y^2*a - g*n/W0^3,

with a fixed-N integral state and an unknown chemical potential.  The
vector equation is solved as the nonlocal Gauss saddle; no local ``Cv*n^2``
replacement is used for an inhomogeneous profile.  The command line emits
strict JSON and has no file-writing side effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:
    from scipy.integrate import simpson, solve_bvp
    from scipy.interpolate import CubicSpline
except ImportError as exc:  # pragma: no cover - requirements pin SciPy.
    raise RuntimeError("SciPy is required for the finite droplet audit") from exc

sys.dont_write_bytecode = True

try:
    from source_complete_scaling_saturation_audit import (
        INPUTS,
        inverse_potential_jet,
    )
except ImportError:  # pragma: no cover - package-style import support.
    from .source_complete_scaling_saturation_audit import (
        INPUTS,
        inverse_potential_jet,
    )


HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
SCHEMA = "nvg_finite_droplet_audit.v1"
STATUS = "COMPUTED_FINITE_N_W8_SPHERICAL_TF_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0
TARGETS = ("0.90", "0.93")
NUMBERS = (40, 208)
SEED_FACTORS = (0.8, 1.0, 1.2)
# The 16 fm box is the primary finite-domain calculation.  The 24 fm box is
# deliberately larger and is only a domain/tail diagnostic.
PRIMARY_BOX_FM = 16.0
DOMAIN_BOX_FM = 24.0
PROTOCOL_LEVELS = (
    {"nodes": 450, "tolerance": 1.0e-3, "label": "coarse"},
    {"nodes": 700, "tolerance": 2.0e-4, "label": "medium"},
    {"nodes": 1000, "tolerance": 5.0e-5, "label": "fine"},
)
DOMAIN_LEVEL = {"nodes": 1400, "tolerance": 1.0e-5, "label": "larger_domain"}
MAX_NODES_FACTOR = 20
FIELD_RESIDUAL_LIMIT = 1.0e-4
N_RELATIVE_LIMIT = 1.0e-6
CHEMICAL_EQUILIBRIUM_LIMIT_MEV = 1.0e-2
ENERGY_REFINEMENT_LIMIT_MEV = 2.0e-2
RMS_REFINEMENT_LIMIT_FM = 2.0e-2
VIRIAL_RELATIVE_LIMIT = 1.0e-3
GAUSS_IDENTITY_RELATIVE_LIMIT = 1.0e-6
SCALE_CONTROLLED_N_RELATIVE_LIMIT = 1.0e-10


class FiniteDropletError(ValueError):
    """Fail-closed error for malformed or incomplete numerical evidence."""


def _finite_float(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _number(value: Any, digits: int = 17) -> Any:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FiniteDropletError("non-numeric scientific output") from exc
    if not math.isfinite(value):
        raise FiniteDropletError("non-finite scientific output")
    # Keep binary64 values round-trip stable.  Aggregated diagnostics such as
    # a near-unity normalization factor and its tiny correction are recomputed
    # by validate_result, so lossy decimal shortening is not coherent.
    return float(format(value, f".{digits}g"))


def _jsonable(value: Any) -> Any:
    """Convert numerical diagnostics while rejecting NaN/Infinity."""

    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not isinstance(value, bool):
        return _number(value)
    return value


def _stable_series_integrals(ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dimensionless energy, pressure and scalar-density integrals.

    The small-ratio series avoids cancellation at a dilute surface.  The
    returned integrals are respectively
    ``int(t^2*sqrt(1+t^2))``, ``int(t^4/sqrt(1+t^2))`` and
    ``int(t^2/sqrt(1+t^2))`` from zero to ``ratio``.
    """

    ratio = np.asarray(ratio, dtype=float)
    positive = np.maximum(ratio, 0.0)
    ie = positive**3 / 3.0
    ip = positive**5 / 5.0
    ins = positive**3 / 3.0
    ce = np.ones_like(positive)
    cp = np.ones_like(positive)
    cs = np.ones_like(positive)
    # The largest ratio in the physical branch is modest, but keep enough
    # terms for the dilute edge and use direct formulas above r=.45.
    for j in range(1, 20):
        ce *= (0.5 - (j - 1)) / j
        cp *= (-(2.0 * j - 1.0)) / (2.0 * j)
        cs *= (-(2.0 * j - 1.0)) / (2.0 * j)
        ie += ce * positive ** (3 + 2 * j) / (3 + 2 * j)
        ip += cp * positive ** (5 + 2 * j) / (5 + 2 * j)
        ins += cs * positive ** (3 + 2 * j) / (3 + 2 * j)
    return ie, ip, ins


@dataclass
class W8Design:
    """Floating-point adapter around one exact-decimal inverse W8 design."""

    target_y: str
    M: float
    W0: float
    hbarc: float
    momega: float
    gomega: float
    q: float
    gs: float
    d: float
    n0_dim: float
    coefficients_dim: np.ndarray
    target_mu: float

    @classmethod
    def from_target(cls, target_y: str) -> "W8Design":
        if target_y not in TARGETS:
            raise FiniteDropletError(f"unsupported W8 target {target_y}")
        # Keep producer evaluation at a controlled precision and immediately
        # convert to a numerical solver adapter.  No saved result table is
        # read as an input.
        import mpmath as mp

        with mp.workdps(90):
            model, state, jet = inverse_potential_jet(
                target_y, "240", "0.16", "-16"
            )
            W0 = float(model.W0)
            M = float(model.MN)
            hbarc = float(model.hbarc)
            momega = float(model.momega)
            gomega = float(model.gomega)
            n0_dim = float(state["n"] / model.W0**3)
            coefficients = np.asarray(
                [float(value / model.W0**4) for value in jet["coefficients"]],
                dtype=float,
            )
        return cls(
            target_y=str(target_y), M=M, W0=W0, hbarc=hbarc,
            momega=momega, gomega=gomega, q=momega / W0, gs=M / W0,
            d=4.0, n0_dim=n0_dim, coefficients_dim=coefficients,
            target_mu=M - 16.0,
        )

    def potential(self, y: np.ndarray | float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        z = y * y - 1.0
        a2, a3, a4 = self.coefficients_dim
        return a2 * z**2 + a3 * z**3 + a4 * z**4

    def potential_y(self, y: np.ndarray | float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        z = y * y - 1.0
        a2, a3, a4 = self.coefficients_dim
        return 4.0 * a2 * y * z + 6.0 * a3 * y * z**2 + 8.0 * a4 * y * z**3

    def potential_yy(self, y: np.ndarray | float) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        z = y * y - 1.0
        a2, a3, a4 = self.coefficients_dim
        return (
            a2 * (4.0 * z + 8.0 * y * y)
            + a3 * (6.0 * z**2 + 24.0 * y * y * z)
            + a4 * (8.0 * z**3 + 48.0 * y * y * z**2)
        )

    def fermi_from_density(self, n_dim: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """Fermi observables for dimensionless density and scalar y."""

        n_dim = np.maximum(np.asarray(n_dim, dtype=float), 0.0)
        y_safe = np.maximum(np.asarray(y, dtype=float), 1.0e-8)
        mass = self.M * y_safe
        k = (6.0 * math.pi**2 * self.W0**3 * n_dim / self.d) ** (1.0 / 3.0)
        ef = np.sqrt(k * k + mass * mass)
        ratio = np.divide(k, mass, out=np.zeros_like(k), where=mass > 0.0)
        ie, ip, ins = _stable_series_integrals(ratio)
        direct = ratio > 0.45
        if np.any(direct):
            rr = ratio[direct]
            kk, mm, ee = k[direct], mass[direct], ef[direct]
            asinh = np.arcsinh(rr)
            energy = (
                kk * ee * (2.0 * kk * kk + mm * mm) - mm**4 * asinh
            ) / (8.0 * mm**4)
            pressure = (
                kk * ee * (2.0 * kk * kk - 3.0 * mm * mm)
                + 3.0 * mm**4 * asinh
            ) / (8.0 * mm**4)
            scalar = (rr * np.sqrt(1.0 + rr * rr) - asinh) / 2.0
            ie[direct], ip[direct], ins[direct] = energy, pressure, scalar
        f_dim = self.d * mass**4 * ie / (2.0 * math.pi**2 * self.W0**4)
        p_dim = self.d * mass**4 * ip / (6.0 * math.pi**2 * self.W0**4)
        ns_dim = self.d * mass**3 * ins / (2.0 * math.pi**2 * self.W0**3)
        return {"k": k, "mass": mass, "ef": ef, "f": f_dim, "p": p_dim, "ns": ns_dim}

    def matter(self, y: np.ndarray, a: np.ndarray, mu: float) -> dict[str, np.ndarray]:
        """Local KKT density from ``nu=mu-g*A``; inactive cells have n=0."""

        y = np.asarray(y, dtype=float)
        a = np.asarray(a, dtype=float)
        y_safe = np.maximum(y, 1.0e-8)
        mass = self.M * y_safe
        nu = float(mu) - self.gomega * self.W0 * a
        active = nu > mass
        k = np.sqrt(np.maximum(nu * nu - mass * mass, 0.0))
        n_dim = np.where(active, self.d * k**3 / (6.0 * math.pi**2 * self.W0**3), 0.0)
        values = self.fermi_from_density(n_dim, y_safe)
        values.update({"nu": nu, "active": active, "n": n_dim})
        return values

    def bvp_fun(self, x: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
        y, yp, a, ap, number_state = Y
        del number_state
        matter = self.matter(y, a, float(p[0]))
        out = np.empty_like(Y)
        out[0] = yp
        out[1] = self.potential_y(y) + self.gs * matter["ns"] - self.q**2 * y * a * a
        out[2] = ap
        out[3] = self.q**2 * y * y * a - self.gomega * matter["n"]
        out[4] = x * x * matter["n"]
        return out

    @staticmethod
    def bvp_bc(ya: np.ndarray, yb: np.ndarray, p: np.ndarray, target_N: float) -> np.ndarray:
        del p
        return np.asarray(
            [ya[1], ya[3], ya[4], yb[0] - 1.0, yb[2], 4.0 * math.pi * yb[4] - target_N],
            dtype=float,
        )

    def seed(self, x: np.ndarray, target_N: int, factor: float, mu: float | None = None) -> np.ndarray:
        radius = (3.0 * target_N / (4.0 * math.pi * self.n0_dim)) ** (1.0 / 3.0)
        radius *= float(factor)
        width = 2.0
        t = np.tanh((x - radius) / width)
        y = float(self.target_y) + (1.0 - float(self.target_y)) * 0.5 * (1.0 + t)
        yp = (1.0 - float(self.target_y)) * 0.5 / width * (1.0 - t * t)
        a0 = self.gomega * self.n0_dim / (self.q * self.q * float(self.target_y) ** 2)
        a = 0.5 * a0 * (1.0 - t)
        ap = -0.5 * a0 / width * (1.0 - t * t)
        number_state = self.n0_dim * np.minimum(x, radius) ** 3 / 3.0
        return np.vstack([y, yp, a, ap, number_state])


def _transfer_guess(solution: Any, x: np.ndarray, target_N: int | None = None) -> np.ndarray:
    """Interpolate a converged BVP, padding the larger box by vacuum tails."""

    old_x = np.asarray(solution.x, dtype=float)
    values = np.asarray(solution.sol(np.minimum(x, old_x[-1])), dtype=float)
    outside = x > old_x[-1]
    if np.any(outside):
        values[0, outside] = 1.0
        values[1, outside] = 0.0
        values[2, outside] = 0.0
        values[3, outside] = 0.0
        values[4, outside] = values[4, np.flatnonzero(~outside)[-1]] if np.any(~outside) else 0.0
    if target_N is not None and values.shape[1]:
        old_N = max(4.0 * math.pi * values[4, -1], 1.0e-12)
        if abs(old_N - float(target_N)) > 1.0e-10:
            values[4] *= float(target_N) / old_N
    return values


def _solve_once(
    design: W8Design,
    target_N: int,
    box_fm: float,
    nodes: int,
    tolerance: float,
    *,
    factor: float | None = None,
    previous: Any | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """Run one BVP attempt and return a serializable row plus private solution."""

    x_max = design.W0 * float(box_fm) / design.hbarc
    x = np.linspace(0.0, x_max, int(nodes))
    if previous is None:
        if factor is None:
            raise FiniteDropletError("seed factor required for a fresh BVP")
        guess = design.seed(x, target_N, factor)
        p0 = np.asarray([design.target_mu], dtype=float)
    else:
        guess = _transfer_guess(previous, x, target_N)
        p0 = np.asarray(previous.p, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            solution = solve_bvp(
                design.bvp_fun,
                lambda ya, yb, p: design.bvp_bc(ya, yb, p, target_N),
                x,
                guess,
                p=p0,
                S=np.diag([0.0, -2.0, 0.0, -2.0, 0.0]),
                tol=float(tolerance),
                max_nodes=max(4000, int(nodes) * MAX_NODES_FACTOR),
                verbose=0,
            )
        except (ArithmeticError, FloatingPointError, RuntimeError, ValueError) as exc:
            return {
                "solver_status": -1,
                "solver_message": str(exc),
                "converged": False,
                "box_fm": float(box_fm),
                "nodes_requested": int(nodes),
                "tolerance": float(tolerance),
                "seed_factor": factor,
            }, None
    row = {
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "converged": bool(solution.status == 0),
        "box_fm": float(box_fm),
        "x_box": float(x_max),
        "nodes_requested": int(nodes),
        "nodes_used": int(solution.x.size),
        "tolerance": float(tolerance),
        "seed_factor": factor,
        "collocation_max_rms": float(np.max(solution.rms_residuals)) if solution.rms_residuals.size else math.inf,
        "chemical_potential_MeV": float(solution.p[0]) if solution.p.size else math.nan,
    }
    if not row["converged"]:
        return row, solution
    try:
        diagnostics = _diagnostics(design, target_N, solution)
    except (ArithmeticError, FloatingPointError, RuntimeError, ValueError) as exc:
        row.update({"converged": False, "diagnostic_failure": str(exc)})
        return row, solution
    row.update(diagnostics)
    return row, solution


def _diagnostics(design: W8Design, target_N: int, solution: Any) -> dict[str, Any]:
    """Independently integrate a converged BVP and evaluate all obligations."""

    if solution.status != 0 or not solution.p.size:
        raise FiniteDropletError("diagnostics require a converged BVP")
    x_max = float(solution.x[-1])
    # Dense integration is intentionally independent of the collocation mesh.
    x = np.linspace(0.0, x_max, max(8001, int(solution.x.size) * 6))
    Y = np.asarray(solution.sol(x), dtype=float)
    first = np.asarray(solution.sol(x, 1), dtype=float)
    try:
        second = np.asarray(solution.sol(x, 2), dtype=float)
    except (TypeError, ValueError):  # pragma: no cover - old SciPy fallback.
        second = np.vstack([np.gradient(first[i], x, edge_order=2) for i in range(5)])
    y, yp, a, ap = Y[:4]
    mu = float(solution.p[0])
    matter = design.matter(y, a, mu)
    n, ns, f, pressure = matter["n"], matter["ns"], matter["f"], matter["p"]
    # The radial Laplacians are evaluated away from x=0; the first point is
    # replaced by its regular limit and is omitted from the max if needed.
    active_grid = x > max(1.0e-7, x_max * 1.0e-8)
    scalar_lap = second[0] + np.divide(2.0 * yp, x, out=np.zeros_like(x), where=x > 0.0)
    gauss_lap = second[2] + np.divide(2.0 * ap, x, out=np.zeros_like(x), where=x > 0.0)
    scalar_expected = design.potential_y(y) + design.gs * ns - design.q**2 * y * a * a
    gauss_expected = gauss_lap * -1.0 + design.q**2 * y * y * a - design.gomega * n
    scalar_residual = scalar_lap - scalar_expected
    gauss_residual = gauss_expected
    # At x=0 solve_bvp's singular term is represented in S rather than in
    # the sampled second derivative.  Its regular residual is the source
    # equation itself and is checked on the first nonzero point.
    if np.any(active_grid):
        sr = scalar_residual[active_grid]
        gr = gauss_residual[active_grid]
    else:  # pragma: no cover - impossible for a positive box.
        sr, gr = scalar_residual, gauss_residual
    scalar_scale = max(1.0e-3, float(np.max(np.abs(scalar_expected))))
    gauss_scale = max(1.0e-3, float(np.max(np.abs(design.q**2 * y * y * a - design.gomega * n))))
    scalar_abs = float(np.max(np.abs(sr)))
    gauss_abs = float(np.max(np.abs(gr)))
    scalar_relative = scalar_abs / scalar_scale
    gauss_relative = gauss_abs / gauss_scale
    field_relative = max(scalar_relative, gauss_relative)
    # Independent number integral, not the BVP auxiliary state.
    number = 4.0 * math.pi * float(simpson(x * x * n, x=x))
    number_relative = abs(number - target_N) / target_N
    occupied = matter["active"] & (n > 1.0e-12 * max(design.n0_dim, 1.0e-12))
    if np.any(occupied):
        equilibrium_error = float(np.max(np.abs(matter["ef"][occupied] - matter["nu"][occupied])))
    else:
        equilibrium_error = 0.0
    weight = 4.0 * math.pi * design.W0 * x * x
    t_w = float(simpson(weight * 0.5 * yp * yp, x=x))
    v_u = float(simpson(weight * design.potential(y), x=x))
    e_f = float(simpson(weight * f, x=x))
    t_a = float(simpson(weight * 0.5 * ap * ap, x=x))
    v_a = float(simpson(weight * 0.5 * design.q**2 * y * y * a * a, x=x))
    source = float(simpson(weight * design.gomega * n * a, x=x))
    boundary_flux = 4.0 * math.pi * design.W0 * x_max**2 * a[-1] * ap[-1]
    gauss_identity_residual = 2.0 * (t_a + v_a) - source - boundary_flux
    energy = t_w + v_u + e_f + t_a + v_a
    p_integral = float(simpson(weight * pressure, x=x))
    virial = -t_w - 3.0 * v_u + 3.0 * p_integral + t_a + 3.0 * v_a
    rms = design.hbarc / design.W0 * math.sqrt(
        float(simpson(x**4 * n, x=x) / simpson(x * x * n, x=x))
    )
    outer = x > 0.8 * x_max
    outer_number_fraction = float(
        4.0 * math.pi * simpson(x[outer] ** 2 * n[outer], x=x[outer]) / max(number, 1.0e-30)
    ) if np.count_nonzero(outer) > 2 else 0.0
    density_boundary = float(n[-1])
    localized = bool(
        mu < design.M
        and density_boundary <= 1.0e-10
        and outer_number_fraction <= 1.0e-8
    )
    energy_per_particle_minus_mass = energy / number - design.M
    virial_relative = abs(virial) / max(abs(energy), 1.0)
    scalar_boundary_derivative = float(first[0, -1])
    gauss_boundary_derivative = float(first[2, -1])
    energy_mu_identity_residual = energy - mu * number - (
        2.0 / 3.0 * (t_w - t_a) - virial / 3.0
    )
    gauss_energy = t_a + v_a
    gauss_identity_lhs = 2.0 * gauss_energy
    gauss_identity_positive_physical_domain = bool(
        np.all(np.isfinite(y))
        and np.all(np.isfinite(a))
        and np.all(np.isfinite(n))
        and np.all(y > 0.0)
        and np.all(n >= 0.0)
        and math.isfinite(gauss_energy)
        and gauss_energy > 0.0
    )
    if gauss_identity_positive_physical_domain:
        gauss_identity_relative = abs(gauss_identity_residual) / gauss_identity_lhs
    else:
        gauss_identity_relative = math.inf
    gauss_identity_acceptance = bool(
        gauss_identity_positive_physical_domain
        and math.isfinite(gauss_identity_residual)
        and math.isfinite(gauss_identity_relative)
        and gauss_identity_relative <= GAUSS_IDENTITY_RELATIVE_LIMIT
    )
    return {
        "chemical_potential_MeV": mu,
        "conserved_N_integrated": number,
        "conserved_N_relative_error": number_relative,
        "central_y": float(y[0]),
        "central_a_A_over_W0": float(a[0]),
        "central_density_fm_minus3": float(n[0] * design.W0**3 / design.hbarc**3),
        "max_density_fm_minus3": float(np.max(n) * design.W0**3 / design.hbarc**3),
        "scalar_residual_max_abs_dimensionless": scalar_abs,
        "gauss_residual_max_abs_dimensionless": gauss_abs,
        "scalar_residual_relative": scalar_relative,
        "gauss_residual_relative": gauss_relative,
        "field_residual_relative": field_relative,
        "chemical_equilibrium_max_error_MeV": equilibrium_error,
        "energy_components_MeV": {
            "T_W": t_w, "V_U": v_u, "E_F": e_f, "T_A": t_a, "V_A": v_a,
            "source_integral_gnA": source,
            "Gauss_boundary_flux": boundary_flux,
            "total_E": energy,
        },
        "energy_per_particle_minus_M_MeV": energy_per_particle_minus_mass,
        "rms_baryon_radius_fm": rms,
        "integrated_fermi_pressure_MeV": p_integral,
        "virial_D_MeV": virial,
        "virial_relative": virial_relative,
        "gauss_energy_identity_residual_MeV": gauss_identity_residual,
        "gauss_energy_identity_relative": gauss_identity_relative,
        "gauss_energy_identity_scale_MeV": gauss_identity_lhs,
        "gauss_identity_positive_physical_domain": gauss_identity_positive_physical_domain,
        "gauss_energy_identity_acceptance": gauss_identity_acceptance,
        "energy_minus_muN_identity_residual_MeV": energy_mu_identity_residual,
        "boundary_derivatives_dimensionless": {
            "y_x_at_box": scalar_boundary_derivative,
            "a_x_at_box": gauss_boundary_derivative,
        },
        "outer_number_fraction_r_gt_0.8_box": outer_number_fraction,
        "boundary_density_fm_minus3": density_boundary * design.W0**3 / design.hbarc**3,
        "localized_vacuum_exterior": localized,
        "binding_condition_E_per_N_lt_M": bool(energy_per_particle_minus_mass < 0.0),
        "stationarity_acceptance": bool(
            number_relative <= N_RELATIVE_LIMIT
            and field_relative <= FIELD_RESIDUAL_LIMIT
            and equilibrium_error <= CHEMICAL_EQUILIBRIUM_LIMIT_MEV
            and virial_relative <= VIRIAL_RELATIVE_LIMIT
            and gauss_identity_acceptance
        ),
        "profile_grid_x": x,
        "profile_y": y,
        "profile_a": a,
        "profile_n": n,
        "profile_yp": yp,
        "profile_ap": ap,
    }


def _strip_profiles(row: Mapping[str, Any]) -> dict[str, Any]:
    """Drop private dense arrays before JSON serialization."""

    return {key: value for key, value in row.items() if not key.startswith("profile_")}


def _branch_signature(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(round(float(row.get("chemical_potential_MeV", 0.0)) * 10.0)),
        int(round(float(row.get("central_y", 0.0)) * 1000.0)),
        int(round(float(row.get("max_density_fm_minus3", 0.0)) * 10.0)),
    )


def _same_branch(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Merge seed paths that converge to the same physical profile."""

    return bool(
        abs(float(left.get("chemical_potential_MeV", 0.0))
            - float(right.get("chemical_potential_MeV", 0.0))) <= 0.2
        and abs(float(left.get("central_y", 0.0))
                - float(right.get("central_y", 0.0))) <= 2.0e-3
        and abs(float(left.get("max_density_fm_minus3", 0.0))
                - float(right.get("max_density_fm_minus3", 0.0))) <= 5.0e-3
    )


def _classify(row: Mapping[str, Any]) -> str:
    if not row.get("converged"):
        return "FAILED_NUMERICAL_PROTOCOL"
    if not row.get("localized_vacuum_exterior"):
        return (
            "NUMERICAL_STATIONARY_UNBOUND_NONLOCALIZED"
            if row.get("stationarity_acceptance") is True
            else "CONVERGED_UNBOUND_NONLOCALIZED_RESIDUAL_LIMIT_MISSED"
        )
    if row.get("stationarity_acceptance") is not True:
        return "CONVERGED_BUT_RESIDUAL_OR_VIRIAL_LIMIT_MISSED"
    if row.get("localized_vacuum_exterior") and row.get("binding_condition_E_per_N_lt_M"):
        return "NUMERICAL_STATIONARY_BOUND_CANDIDATE"
    return "NUMERICAL_STATIONARY_UNBOUND_LOCALIZED"


def _gauss_resolve_for_scaled_profile(
    design: W8Design,
    x: np.ndarray,
    y_scaled: np.ndarray,
    n_scaled: np.ndarray,
    a_guess: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-solve the positive Gauss operator for a fixed scaled profile."""

    def fun(xx: np.ndarray, AA: np.ndarray) -> np.ndarray:
        yv = np.interp(xx, x, y_scaled)
        nv = np.interp(xx, x, n_scaled)
        return np.vstack([AA[1], design.q**2 * yv * yv * AA[0] - design.gomega * nv])

    def bc(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray([left[1], right[0]], dtype=float)

    initial = np.vstack([a_guess, np.gradient(a_guess, x, edge_order=2)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solved = solve_bvp(
            fun, bc, x, initial, S=np.diag([0.0, -2.0]),
            tol=2.0e-8, max_nodes=max(3000, 4 * x.size), verbose=0,
        )
    if solved.status != 0:
        raise FiniteDropletError(f"scaled Gauss solve failed: {solved.message}")
    vals = np.asarray(solved.sol(x), dtype=float)
    return vals[0], vals[1]


def _scaled_energy(
    design: W8Design,
    profile: Mapping[str, np.ndarray],
    lam: float,
    target_N: float,
) -> tuple[float, float, float, float, np.ndarray]:
    x = np.asarray(profile["x"], dtype=float)
    y0 = np.asarray(profile["y"], dtype=float)
    n0 = np.asarray(profile["n"], dtype=float)
    a0 = np.asarray(profile["a"], dtype=float)
    yp0 = np.asarray(profile["yp"], dtype=float)
    xp = lam * x
    # Cubic interpolation suppresses the O(dx^2)/h amplification that a
    # piecewise-linear profile would introduce in the independent derivative
    # when h is reduced.  Vacuum tails are imposed exactly outside the box.
    y_spline = CubicSpline(x, y0, extrapolate=False)
    n_spline = CubicSpline(x, n0, extrapolate=False)
    # Differentiate the interpolated profile itself.  Interpolating a
    # separately sampled derivative and then multiplying by lambda creates a
    # first-order mismatch in the scalar-gradient contribution.
    yp_spline = y_spline.derivative()
    inside = xp <= x[-1]
    y_scaled = np.ones_like(x)
    n_scaled = np.zeros_like(x)
    yp_scaled = np.zeros_like(x)
    y_scaled[inside] = y_spline(xp[inside])
    n_scaled[inside] = np.maximum(n_spline(xp[inside]), 0.0) * lam**3
    yp_scaled[inside] = yp_spline(xp[inside]) * lam
    raw_number = 4.0 * math.pi * float(simpson(x * x * n_scaled, x=x))
    if not math.isfinite(raw_number) or raw_number <= 0.0:
        raise FiniteDropletError("scaled profile has no finite positive number")
    normalization_factor = float(target_N) / raw_number
    n_scaled *= normalization_factor
    controlled_number = 4.0 * math.pi * float(simpson(x * x * n_scaled, x=x))
    a_scaled, ap_scaled = _gauss_resolve_for_scaled_profile(
        design, x, y_scaled, n_scaled, np.where(inside, np.interp(xp, x, a0, right=0.0), 0.0)
    )
    fermi = design.fermi_from_density(n_scaled, y_scaled)
    weight = 4.0 * math.pi * design.W0 * x * x
    energy = float(simpson(
        weight * (
            0.5 * yp_scaled**2 + design.potential(y_scaled) + fermi["f"]
            + 0.5 * ap_scaled**2 + 0.5 * design.q**2 * y_scaled**2 * a_scaled**2
        ), x=x
    ))
    return energy, raw_number, controlled_number, normalization_factor, a_scaled


def _scale_derivative(
    design: W8Design, row: Mapping[str, Any], target_N: int
) -> dict[str, Any]:
    """Independent fixed-N scale finite differences with Gauss re-solves."""

    if not row.get("converged"):
        return {"available": False, "reason": "base solution did not converge"}
    profile = {
        "x": np.asarray(row["profile_grid_x"], dtype=float),
        "y": np.asarray(row["profile_y"], dtype=float),
        "n": np.asarray(row["profile_n"], dtype=float),
        "a": np.asarray(row["profile_a"], dtype=float),
        "yp": np.asarray(row["profile_yp"], dtype=float),
    }
    values = []
    for h in (0.02, 0.01, 0.005, 0.0025):
        plus, n_plus_raw, n_plus_controlled, plus_factor, _ = _scaled_energy(
            design, profile, math.exp(h), target_N
        )
        minus, n_minus_raw, n_minus_controlled, minus_factor, _ = _scaled_energy(
            design, profile, math.exp(-h), target_N
        )
        values.append({
            "step_dln_lambda": h,
            "finite_difference_D_MeV": (plus - minus) / (2.0 * h),
            "E_plus_MeV": plus,
            "E_minus_MeV": minus,
            # Keep the old names as raw quadrature diagnostics and expose the
            # controlled values used for the Gauss re-solves explicitly.
            "N_plus": n_plus_raw,
            "N_minus": n_minus_raw,
            "N_plus_raw": n_plus_raw,
            "N_minus_raw": n_minus_raw,
            "N_plus_controlled": n_plus_controlled,
            "N_minus_controlled": n_minus_controlled,
            "N_plus_raw_drift_relative": abs(n_plus_raw - target_N) / target_N,
            "N_minus_raw_drift_relative": abs(n_minus_raw - target_N) / target_N,
            "N_plus_controlled_drift_relative": abs(n_plus_controlled - target_N) / target_N,
            "N_minus_controlled_drift_relative": abs(n_minus_controlled - target_N) / target_N,
            "N_plus_normalization_factor": plus_factor,
            "N_minus_normalization_factor": minus_factor,
            "N_plus_normalization_correction_relative": abs(plus_factor - 1.0),
            "N_minus_normalization_correction_relative": abs(minus_factor - 1.0),
        })
    analytic = float(row["virial_D_MeV"])
    fd = [float(value["finite_difference_D_MeV"]) for value in values]
    successive = abs(fd[-1] - fd[-2])
    # The two finest centered differences carry the expected O(h^2) error.
    # Richardson extrapolation makes the independent comparison meaningful
    # at MeV-scale energies without amplifying the last Simpson ulps.
    richardson = (4.0 * fd[-1] - fd[-2]) / 3.0
    richardson_error = abs(richardson - fd[-1])
    agreement = abs(richardson - analytic)
    raw_drift = max(
        max(float(value["N_plus_raw_drift_relative"]), float(value["N_minus_raw_drift_relative"]))
        for value in values
    )
    controlled_drift = max(
        max(
            float(value["N_plus_controlled_drift_relative"]),
            float(value["N_minus_controlled_drift_relative"]),
        )
        for value in values
    )
    normalization_correction = max(
        max(
            float(value["N_plus_normalization_correction_relative"]),
            float(value["N_minus_normalization_correction_relative"]),
        )
        for value in values
    )
    return {
        "available": True,
        "gauss_resolved_for_each_lambda": True,
        "target_N": int(target_N),
        "fixed_N_normalization_enforced": True,
        "steps": values,
        "analytic_D_MeV": analytic,
        "finest_D_MeV": fd[-1],
        "finest_vs_analytic_MeV": abs(fd[-1] - analytic),
        "richardson_extrapolated_D_MeV": richardson,
        "richardson_vs_analytic_MeV": agreement,
        "successive_finite_difference_change_MeV": successive,
        "richardson_error_estimate_MeV": richardson_error,
        "raw_number_drift_relative_max": raw_drift,
        "controlled_number_drift_relative_max": controlled_drift,
        "normalization_correction_relative_max": normalization_correction,
        "fixed_N_number_control_passed": bool(
            controlled_drift <= SCALE_CONTROLLED_N_RELATIVE_LIMIT
        ),
        "finite_difference_converged": bool(
            richardson_error <= 0.1
            and controlled_drift <= SCALE_CONTROLLED_N_RELATIVE_LIMIT
        ),
        "analytic_and_re_solved_agree": bool(agreement <= 0.1),
    }


def _refinement_summary(
    rows: Iterable[Mapping[str, Any]],
    domain_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require every declared level and every adjacent/domain comparison."""

    rows = list(rows)
    declared_labels = [str(level["label"]) for level in PROTOCOL_LEVELS]
    actual_labels = [str(row.get("protocol_label")) for row in rows]
    levels_complete = bool(
        len(rows) == len(PROTOCOL_LEVELS)
        and actual_labels == declared_labels
        and all(row.get("converged") is True for row in rows)
    )
    summary: dict[str, Any] = {
        "available": False,
        "protocol_levels_complete": levels_complete,
        "all_declared_levels_converged": levels_complete,
        "consecutive_energy_changes_MeV": [],
        "consecutive_rms_changes_fm": [],
        "fine_to_domain_energy_change_MeV": None,
        "fine_to_domain_rms_change_fm": None,
        "domain_check_converged": bool(
            isinstance(domain_row, Mapping) and domain_row.get("converged") is True
        ),
        "domain_energy_refinement_passed": False,
        "domain_rms_refinement_passed": False,
        "E_per_N_change_MeV": None,
        "rms_change_fm": None,
        "energy_refinement_passed": False,
        "rms_refinement_passed": False,
    }
    if not levels_complete:
        summary["reason"] = "all three declared refinements did not converge"
        return summary
    consecutive_energy = [
        abs(
            float(rows[index + 1]["energy_per_particle_minus_M_MeV"])
            - float(rows[index]["energy_per_particle_minus_M_MeV"])
        )
        for index in range(len(rows) - 1)
    ]
    consecutive_rms = [
        abs(
            float(rows[index + 1]["rms_baryon_radius_fm"])
            - float(rows[index]["rms_baryon_radius_fm"])
        )
        for index in range(len(rows) - 1)
    ]
    summary["consecutive_energy_changes_MeV"] = consecutive_energy
    summary["consecutive_rms_changes_fm"] = consecutive_rms
    summary["E_per_N_change_MeV"] = abs(
        float(rows[-1]["energy_per_particle_minus_M_MeV"])
        - float(rows[0]["energy_per_particle_minus_M_MeV"])
    )
    summary["rms_change_fm"] = abs(
        float(rows[-1]["rms_baryon_radius_fm"])
        - float(rows[0]["rms_baryon_radius_fm"])
    )
    if summary["domain_check_converged"]:
        fine = rows[-1]
        summary["fine_to_domain_energy_change_MeV"] = abs(
            float(domain_row["energy_per_particle_minus_M_MeV"])
            - float(fine["energy_per_particle_minus_M_MeV"])
        )
        summary["fine_to_domain_rms_change_fm"] = abs(
            float(domain_row["rms_baryon_radius_fm"])
            - float(fine["rms_baryon_radius_fm"])
        )
        summary["domain_energy_refinement_passed"] = bool(
            summary["fine_to_domain_energy_change_MeV"] <= ENERGY_REFINEMENT_LIMIT_MEV
        )
        summary["domain_rms_refinement_passed"] = bool(
            summary["fine_to_domain_rms_change_fm"] <= RMS_REFINEMENT_LIMIT_FM
        )
    summary["available"] = bool(
        summary["domain_check_converged"]
        and summary["domain_energy_refinement_passed"]
        and summary["domain_rms_refinement_passed"]
    )
    summary["energy_refinement_passed"] = bool(
        all(change <= ENERGY_REFINEMENT_LIMIT_MEV for change in consecutive_energy)
        and summary["domain_energy_refinement_passed"]
    )
    summary["rms_refinement_passed"] = bool(
        all(change <= RMS_REFINEMENT_LIMIT_FM for change in consecutive_rms)
        and summary["domain_rms_refinement_passed"]
    )
    return summary


def _gauss_identity_passed(row: Mapping[str, Any]) -> bool:
    """Recompute the positive-energy Gauss identity from serialized pieces."""

    if not isinstance(row, Mapping):
        return False
    components = row.get("energy_components_MeV")
    if not isinstance(components, Mapping):
        return False
    try:
        t_a = float(components["T_A"])
        v_a = float(components["V_A"])
        source = float(components["source_integral_gnA"])
        boundary = float(components["Gauss_boundary_flux"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    lhs = 2.0 * (t_a + v_a)
    residual = lhs - source - boundary
    if not all(math.isfinite(value) for value in (lhs, residual, t_a, v_a, source, boundary)):
        return False
    if lhs <= 0.0 or row.get("gauss_identity_positive_physical_domain") is not True:
        return False
    relative = abs(residual) / lhs
    try:
        reported_relative = float(row["gauss_energy_identity_relative"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(reported_relative) or not math.isclose(
        reported_relative, relative, rel_tol=1.0e-8, abs_tol=1.0e-13
    ):
        return False
    return bool(
        row.get("gauss_energy_identity_acceptance") is True
        and math.isfinite(relative)
        and relative <= GAUSS_IDENTITY_RELATIVE_LIMIT
    )


def _fixed_N_scale_passed(row: Mapping[str, Any]) -> bool:
    """Recompute fixed-N control from every finite-difference step."""

    if not isinstance(row, Mapping):
        return False
    if row.get("available") is not True:
        return False
    if row.get("gauss_resolved_for_each_lambda") is not True:
        return False
    if row.get("fixed_N_normalization_enforced") is not True:
        return False
    try:
        target_N = float(row["target_N"])
        steps = row["steps"]
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(target_N) or target_N <= 0.0 or not isinstance(steps, list):
        return False
    if len(steps) != 4:
        return False
    controlled_drift = 0.0
    raw_drift = 0.0
    normalization_correction = 0.0
    for step in steps:
        if not isinstance(step, Mapping):
            return False
        try:
            controlled = (
                float(step["N_plus_controlled"]),
                float(step["N_minus_controlled"]),
            )
            raw = (
                float(step["N_plus_raw"]),
                float(step["N_minus_raw"]),
            )
            factors = (
                float(step["N_plus_normalization_factor"]),
                float(step["N_minus_normalization_factor"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not all(math.isfinite(value) for value in (*controlled, *raw, *factors)):
            return False
        controlled_drift = max(
            controlled_drift,
            *(abs(value - target_N) / target_N for value in controlled),
        )
        raw_drift = max(
            raw_drift,
            *(abs(value - target_N) / target_N for value in raw),
        )
        normalization_correction = max(
            normalization_correction,
            *(abs(value - 1.0) for value in factors),
        )
    try:
        reported_raw_drift = float(row["raw_number_drift_relative_max"])
        reported_controlled_drift = float(row["controlled_number_drift_relative_max"])
        reported_correction = float(row["normalization_correction_relative_max"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not all(
        math.isfinite(value)
        for value in (reported_raw_drift, reported_controlled_drift, reported_correction)
    ):
        return False
    if not all(
        math.isclose(actual, reported, rel_tol=1.0e-8, abs_tol=1.0e-14)
        for actual, reported in (
            (raw_drift, reported_raw_drift),
            (controlled_drift, reported_controlled_drift),
            (normalization_correction, reported_correction),
        )
    ):
        return False
    return bool(
        row.get("fixed_N_number_control_passed") is True
        and math.isfinite(controlled_drift)
        and controlled_drift <= SCALE_CONTROLLED_N_RELATIVE_LIMIT
    )


def _protocol_levels_complete(rows: Iterable[Mapping[str, Any]]) -> bool:
    rows = list(rows)
    expected = [str(level["label"]) for level in PROTOCOL_LEVELS]
    return bool(
        len(rows) == len(expected)
        and [str(row.get("protocol_label")) for row in rows] == expected
        and all(row.get("converged") is True for row in rows)
    )


def _full_protocol_stationarity_predicate(
    domain_row: Mapping[str, Any],
    refinement: Mapping[str, Any],
    fd_row: Mapping[str, Any],
    refinement_rows: Iterable[Mapping[str, Any]] | None = None,
) -> bool:
    """Single acceptance predicate shared by terminal label and boolean."""

    if not all(isinstance(value, Mapping) for value in (domain_row, refinement, fd_row)):
        return False
    levels_ok = (
        refinement.get("protocol_levels_complete") is True
        and refinement.get("all_declared_levels_converged") is True
    )
    if refinement_rows is not None:
        levels_ok = levels_ok and _protocol_levels_complete(refinement_rows)
    return bool(
        domain_row.get("converged") is True
        and domain_row.get("stationarity_acceptance") is True
        and _gauss_identity_passed(domain_row)
        and refinement.get("available") is True
        and levels_ok
        and refinement.get("energy_refinement_passed") is True
        and refinement.get("rms_refinement_passed") is True
        and _fixed_N_scale_passed(fd_row)
        and fd_row.get("finite_difference_converged") is True
        and fd_row.get("analytic_and_re_solved_agree") is True
    )


def _terminal_classification(
    domain_row: Mapping[str, Any],
    refinement: Mapping[str, Any],
    fd_row: Mapping[str, Any],
    refinement_rows: Iterable[Mapping[str, Any]] | None = None,
) -> str:
    """Classify only the full terminal protocol; never fall back to 16 fm."""

    if not isinstance(domain_row, Mapping) or domain_row.get("converged") is not True:
        return "FAILED_NUMERICAL_PROTOCOL"
    if _full_protocol_stationarity_predicate(
        domain_row, refinement, fd_row, refinement_rows
    ):
        return _classify(domain_row)
    if not domain_row.get("localized_vacuum_exterior"):
        return "CONVERGED_UNBOUND_NONLOCALIZED_RESIDUAL_LIMIT_MISSED"
    return "CONVERGED_BUT_RESIDUAL_OR_VIRIAL_LIMIT_MISSED"


def _run_case(design: W8Design, target_N: int) -> dict[str, Any]:
    """Run seed atlas, deterministic refinements and one larger-box check."""

    coarse_rows: list[dict[str, Any]] = []
    coarse_solutions: list[tuple[dict[str, Any], Any]] = []
    for factor in SEED_FACTORS:
        row, solved = _solve_once(
            design, target_N, PRIMARY_BOX_FM,
            PROTOCOL_LEVELS[0]["nodes"], PROTOCOL_LEVELS[0]["tolerance"],
            factor=factor,
        )
        row = dict(row)
        row["protocol_label"] = PROTOCOL_LEVELS[0]["label"]
        if solved is not None and row.get("converged"):
            row["classification"] = _classify(row)
            coarse_solutions.append((row, solved))
        coarse_rows.append(_strip_profiles(row))
    distinct: list[dict[str, Any]] = []
    for coarse, solved in coarse_solutions:
        signature = _branch_signature(coarse)
        if any(_same_branch(coarse, branch["coarse_reference"]) for branch in distinct):
            for branch in distinct:
                if _same_branch(coarse, branch["coarse_reference"]):
                    branch["seed_factors"].append(float(coarse["seed_factor"]))
            continue
        levels: list[dict[str, Any]] = [coarse]
        last_solution = solved
        for level in PROTOCOL_LEVELS[1:]:
            row, last_solution = _solve_once(
                design, target_N, PRIMARY_BOX_FM,
                level["nodes"], level["tolerance"], previous=last_solution,
            )
            row = dict(row)
            row["protocol_label"] = level["label"]
            if row.get("converged"):
                row["classification"] = _classify(row)
                levels.append(row)
            else:
                levels.append(_strip_profiles(row))
                break
        final = levels[-1]
        domain_raw, domain_solution = _solve_once(
            design, target_N, DOMAIN_BOX_FM,
            DOMAIN_LEVEL["nodes"], DOMAIN_LEVEL["tolerance"], previous=last_solution,
        )
        domain_raw["protocol_label"] = DOMAIN_LEVEL["label"]
        if domain_raw.get("converged"):
            domain_raw["classification"] = _classify(domain_raw)
        domain_row = _strip_profiles(domain_raw)
        # The larger box is the terminal tail-controlled profile whenever it
        # converges.  Using it for the independent scale oracle avoids turning
        # a visible finite-box scalar tail into a false virial discrepancy.
        scale_base = domain_raw if domain_raw.get("converged") else levels[-1]
        fd_row = _scale_derivative(design, scale_base, target_N)
        refinement = _refinement_summary(levels, domain_row)
        terminal_classification = _terminal_classification(
            domain_row, refinement, fd_row, levels
        )
        domain_row["classification"] = terminal_classification
        terminal_protocol_acceptance = _full_protocol_stationarity_predicate(
            domain_row, refinement, fd_row, levels
        )
        branch = {
            "signature": signature,
            "coarse_reference": _strip_profiles(coarse),
            "branch_id": f"branch_{len(distinct) + 1}",
            "seed_factors": [float(coarse["seed_factor"])],
            "refinements": [_strip_profiles(level) for level in levels],
            "larger_domain_check": domain_row,
            "fixed_N_scale_check": _jsonable(fd_row),
            "refinement_summary": refinement,
            "terminal_classification": terminal_classification,
            "terminal_protocol_acceptance": terminal_protocol_acceptance,
            "terminal_stationarity_candidate": bool(
                terminal_classification == "NUMERICAL_STATIONARY_BOUND_CANDIDATE"
            ),
        }
        distinct.append(branch)
    for branch in distinct:
        # The reference row is only an internal deduplication aid and is not
        # part of the public result contract.
        branch.pop("coarse_reference", None)
    status = "NO_CONVERGED_BRANCH" if not distinct else "CONVERGED_BRANCHES_REPORTED"
    return {
        "target_y": design.target_y,
        "target_N": int(target_N),
        "protocol": {
            "primary_box_fm": PRIMARY_BOX_FM,
            "larger_domain_box_fm": DOMAIN_BOX_FM,
            "seed_factors": list(SEED_FACTORS),
            "levels": [dict(level) for level in PROTOCOL_LEVELS],
            "larger_domain_level": dict(DOMAIN_LEVEL),
            "continuation_only_changes_initialization": True,
        },
        "seed_attempts": coarse_rows,
        "distinct_branches": distinct,
        "case_status": status,
        "all_seed_outcomes_reported": len(coarse_rows) == len(SEED_FACTORS),
    }


def _design_metadata(design: W8Design) -> dict[str, Any]:
    return {
        "target_y": design.target_y,
        "n0_fm_minus3": 0.16,
        "binding_input_MeV": -16.0,
        "K_input_MeV": 240.0,
        "mu_input_MeV": design.target_mu,
        "M_MeV": design.M,
        "W0_MeV": design.W0,
        "momega_MeV": design.momega,
        "gomega": design.gomega,
        "q_momega_over_W0": design.q,
        "degeneracy": design.d,
        "coefficients_U8_over_W0^4": design.coefficients_dim.tolist(),
        "calibration_inputs_not_predictions": True,
    }


def calculate() -> dict[str, Any]:
    designs = {target: W8Design.from_target(target) for target in TARGETS}
    cases = {}
    for target, design in designs.items():
        cases[target] = {
            "design": _design_metadata(design),
            "N40": _run_case(design, 40),
            "N208": _run_case(design, 208),
        }
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "model_scope": {
            "zero_temperature": True,
            "symmetric_coulomb_free": True,
            "nonlocal_vector_gauss_operator": "-Laplace + momega^2*y^2",
            "finite_N_controls": [40, 208],
            "no_coulomb_rho_pairing_shell_or_new_interactions": True,
            "finite_box_approximation": True,
            "uneliminated_A_is_a_maximum_constraint": True,
        },
        "input_provenance": {
            "producer": "verification/source_complete_scaling_saturation_audit.py:inverse_potential_jet",
            "producer_source_sha256": hashlib.sha256(
                (HERE / "source_complete_scaling_saturation_audit.py").read_bytes()
            ).hexdigest(),
            "baseline_inputs": dict(INPUTS),
            "no_saved_result_table_used_as_input": True,
        },
        "acceptance_limits": {
            "N_relative": N_RELATIVE_LIMIT,
            "field_relative": FIELD_RESIDUAL_LIMIT,
            "chemical_equilibrium_MeV": CHEMICAL_EQUILIBRIUM_LIMIT_MEV,
            "E_per_N_refinement_MeV": ENERGY_REFINEMENT_LIMIT_MEV,
            "rms_refinement_fm": RMS_REFINEMENT_LIMIT_FM,
            "virial_relative": VIRIAL_RELATIVE_LIMIT,
            "gauss_identity_relative_to_positive_gauss_energy": GAUSS_IDENTITY_RELATIVE_LIMIT,
            "controlled_scale_N_relative": SCALE_CONTROLLED_N_RELATIVE_LIMIT,
        },
        "cases": cases,
    }


_CACHE: dict[str, Any] | None = None


def build_result() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _jsonable(calculate())
    return _CACHE


def _validate_terminal_aggregates(result: Mapping[str, Any]) -> bool:
    """Fail closed if a claimed terminal result lacks protocol evidence."""

    cases = result.get("cases")
    if not isinstance(cases, Mapping):
        return False
    for target in TARGETS:
        target_cases = cases.get(target)
        if not isinstance(target_cases, Mapping):
            return False
        for case_name in ("N40", "N208"):
            case = target_cases.get(case_name)
            if not isinstance(case, Mapping):
                return False
            branches = case.get("distinct_branches")
            if not isinstance(branches, list):
                return False
            for branch in branches:
                if not isinstance(branch, Mapping):
                    return False
                domain = branch.get("larger_domain_check")
                refinement = branch.get("refinement_summary")
                fd = branch.get("fixed_N_scale_check")
                levels = branch.get("refinements")
                if not all(isinstance(value, Mapping) for value in (domain, refinement, fd)):
                    return False
                if not isinstance(levels, list) or not all(
                    isinstance(level, Mapping) for level in levels
                ):
                    return False
                expected_protocol = _full_protocol_stationarity_predicate(
                    domain, refinement, fd, levels
                )
                expected_classification = _terminal_classification(
                    domain, refinement, fd, levels
                )
                if branch.get("terminal_protocol_acceptance") is not expected_protocol:
                    return False
                if branch.get("terminal_classification") != expected_classification:
                    return False
                if domain.get("classification") != expected_classification:
                    return False
                expected_candidate = (
                    expected_classification == "NUMERICAL_STATIONARY_BOUND_CANDIDATE"
                )
                if branch.get("terminal_stationarity_candidate") is not expected_candidate:
                    return False
    return True


def validate_result(result: Mapping[str, Any]) -> bool:
    """Recompute this process's numerical result and fail closed on mutation."""

    if not isinstance(result, Mapping):
        return False
    try:
        if not _validate_terminal_aggregates(result):
            return False
        fresh = build_result()
        return _validate_terminal_aggregates(fresh) and result == fresh
    except (ArithmeticError, FiniteDropletError, KeyError, TypeError, ValueError):
        return False


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - parser path.
        raise FiniteDropletError(message)


def main(argv: list[str] | None = None) -> int:
    parser = JsonArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = build_result()
    except (ArithmeticError, FiniteDropletError, RuntimeError, ValueError) as exc:
        print(json.dumps({
            "schema_version": SCHEMA,
            "status": "INVALID_OR_FAILED_FINITE_DROPLET_AUDIT",
            "evidence_weight": EVIDENCE_WEIGHT,
            "error": str(exc),
        }, ensure_ascii=False, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
