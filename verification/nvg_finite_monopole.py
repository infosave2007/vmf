#!/usr/bin/env python3
"""Finite two-species Thomas--Fermi states and a conditional monopole.

This module is intentionally narrower than a finite-nucleus RPA.  It solves
the spherical d=2-per-species KKT problem for the declared Q4/W8 actions,
including the non-local longitudinal vector equation, a Coulomb Maxwell field
with its exterior tail, and (for the separately labelled J=32 alternative) a
local covariant contact-rho term.  On localized stationary rows it evaluates
the one-coordinate, adiabatic TD--TF dilation

``n_i(eta,r) = exp(3*eta) n_i(exp(eta)*r)``.

The command line is a strict no-write JSON producer.  The numerical output is
zero independent empirical weight: the RCNP records carried in the output are
descriptive provenance only and are never solver inputs.  In particular, the
reported pole is not a finite-nucleus Dirac/RRPA spectrum, a fitted width, or
an experimental centroid.  The static solver subset is also reused by the
separate finite-static bridge; this does not promote the older monopole
command or its dynamic rows to an accepted spectrum calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from scipy.integrate import cumulative_trapezoid, simpson, solve_bvp
    from scipy.interpolate import CubicSpline, PchipInterpolator
except ImportError as exc:  # pragma: no cover - requirements pin SciPy.
    raise RuntimeError("SciPy is required for the finite monopole calculation") from exc

sys.dont_write_bytecode = True

try:  # Both direct-script and package-style imports are useful to focused tests.
    import nvg_finite_droplet_audit as finite
    from nvg_isospin_jet_audit import constant_rho_coupling
    from source_complete_scaling_saturation_audit import BulkModel, INPUTS, inverse_potential_jet
except ImportError:  # pragma: no cover - package import support.
    from . import nvg_finite_droplet_audit as finite
    from .nvg_isospin_jet_audit import constant_rho_coupling
    from .source_complete_scaling_saturation_audit import BulkModel, INPUTS, inverse_potential_jet


HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
CONTRACT_PATH = HERE / "contracts" / "finite_monopole.md"
DATA_PATH = HERE / "data" / "rcnp_isgmr_2025_summaries.json"
SCHEMA = "nvg_finite_monopole.v1"
STATUS = "COMPUTED_FINITE_TD_TF_ADIABATIC_MONOPOLE_ZERO_EVIDENCE_NOT_EMPIRICAL"
EVIDENCE_WEIGHT = 0.0

# These are protocol values, not user-tunable defaults.  ``calculate`` accepts
# no numerical knobs; ``calculate_with_options`` exists only for focused tests
# and requires an explicit options object.
SCALES = ("0.5", "1", "2")
FAMILIES = ("Q4", "W8.90", "W8.93")
NUCLEI = {
    "Zr90": {"A": 90, "Z": 40, "N": 50},
    "Pb208": {"A": 208, "Z": 82, "N": 126},
}
# Keep the frozen legacy matrix above unchanged.  The bridge passes this
# explicit extension to the static solver instead of mutating the old global
# protocol or making Ca40 part of the historical 30-row command.
BRIDGE_NUCLEI = {
    **NUCLEI,
    "Ca40": {"A": 40, "Z": 20, "N": 20},
}
RHO_CHOICES = ("no_rho", "covariant_contact_J32")
SEED_FACTORS = (0.8, 1.0, 1.2)
ALPHA = 0.0072973525643  # CODATA 2022, https://physics.nist.gov/cuu/pdf/wall_2022.pdf
J_DESIGN_MEV = 32.0
N0_FM3 = 0.16
MASS_MEV = 939.0

STATIC_LEVELS = (
    {"label": "coarse16", "box_fm": 16.0, "nodes": 401, "tol": 2.0e-6},
    {"label": "fine24", "box_fm": 24.0, "nodes": 801, "tol": 2.0e-8},
    {"label": "domain40", "box_fm": 40.0, "nodes": 1201, "tol": 2.0e-8},
)
TIGHT24 = {"label": "tight24_independent", "box_fm": 24.0, "nodes": 1201, "tol": 2.0e-8}
MAX_NODES_FACTOR = 8

N_RELATIVE_LIMIT = 1.0e-6
FIELD_RESIDUAL_LIMIT = 1.0e-4
KKT_LIMIT_MEV = 1.0e-2
GAUSS_LIMIT = 1.0e-6
COULOMB_LIMIT = 1.0e-6
# Diagnostics are evaluated on one canonical ladder and one fixed nested
# ladder.  The bridge may tighten the BVP mesh, but it never post-selects a
# phase-shifted sample of an unchanged interpolant.
DIAGNOSTIC_INTERVALS = 12000
DIAGNOSTIC_NESTED_INTERVALS = 24000
ENERGY_DOMAIN_LIMIT_MEV = 2.0e-2
RADIUS_DOMAIN_LIMIT_FM = 2.0e-2
MESH_LIMIT_RELATIVE = 2.0e-2

ETA_STEPS = (0.01, 0.005, 0.0025)
FAST_GAP_TOLERANCES = (160, 320)
OMEGA_RELATIVE_LIMIT = 2.0e-2
INERTIA_RELATIVE_LIMIT = 5.0e-3
CURVATURE_RELATIVE_LIMIT = 5.0e-3
FAST_EPSILON_LIMIT = 5.0e-2


class FiniteMonopoleError(ValueError):
    """Fail-closed error for malformed protocol or scientific output."""


def _number(value: Any, digits: int = 17) -> float:
    if isinstance(value, bool):
        raise FiniteMonopoleError("boolean is not a scientific number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FiniteMonopoleError("non-numeric scientific output") from exc
    if not math.isfinite(result):
        raise FiniteMonopoleError("non-finite scientific output")
    return float(format(result, f".{digits}g"))


def _jsonable(value: Any) -> Any:
    """Convert numerical containers while rejecting NaN/Infinity."""

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
    if isinstance(value, float):
        return _number(value)
    return value


def _safe_rel(left: Any, right: Any, floor: float = 1.0e-30) -> float:
    a, b = float(left), float(right)
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b) / max(abs(a), abs(b), floor)


def _integral(x: np.ndarray, values: np.ndarray) -> float:
    return float(simpson(np.asarray(values, dtype=float), x=np.asarray(x, dtype=float)))


def _spherical_integral(x: np.ndarray, values: np.ndarray) -> float:
    return 4.0 * math.pi * _integral(x, np.asarray(x, dtype=float) ** 2 * values)


def _species_density_from_nu(nu: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d=2 zero-T density and ``dn/d(nu)`` in W0 units.

    The density is exactly zero on the inactive branch.  The derivative is
    one-sided there and is intentionally zero; this is the KKT closure rather
    than a density floor.
    """

    nu = np.asarray(nu, dtype=float)
    # solve_bvp may probe an intermediate Newton iterate far outside the
    # physical branch.  Keep that trial evaluation finite; the final root is
    # still subjected to explicit y>0/residual checks and is never accepted
    # merely because a trial was clipped.
    nu = np.nan_to_num(nu, nan=0.0, posinf=1.0e5, neginf=-1.0e5)
    nu = np.clip(nu, -1.0e5, 1.0e5)
    mass = np.nan_to_num(np.asarray(mass, dtype=float), nan=1.0e5, posinf=1.0e5, neginf=1.0e5)
    mass = np.clip(mass, 1.0e-12, 1.0e5)
    active = nu > mass
    with np.errstate(over="ignore", invalid="ignore"):
        k = np.sqrt(np.maximum(nu * nu - mass * mass, 0.0))
    n = np.where(active, k**3 / (3.0 * math.pi**2), 0.0)
    dn = np.where(active, nu * k / math.pi**2, 0.0)
    return n, dn


def solve_contact_density_difference(
    nu_n: Any,
    nu_p: Any,
    mass: Any,
    kappa: Any,
    *,
    iterations: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the local covariant-contact KKT equation elementwise.

    ``eta = kappa*(n_n-n_p)``, ``n_n=f(nu_n-eta)`` and
    ``n_p=f(nu_p+eta)``.  For ``kappa>=0`` the derivative of the residual is
    ``1+kappa*(dn_n+dn_p)>0``.  Safeguarded Newton therefore has a unique
    monotone root; bisection bounds are retained for cells where a Newton
    step leaves the active interval.  The exact ``kappa=0`` and symmetric
    ``nu_n=nu_p`` reductions are handled without iteration.
    """

    if isinstance(kappa, bool):
        raise FiniteMonopoleError("kappa must be a finite nonnegative scalar")
    try:
        kap = float(kappa)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FiniteMonopoleError("kappa must be a finite nonnegative scalar") from exc
    if not math.isfinite(kap) or kap < 0.0:
        raise FiniteMonopoleError("kappa must be a finite nonnegative scalar")
    un, up, mm = np.broadcast_arrays(
        np.asarray(nu_n, dtype=float), np.asarray(nu_p, dtype=float), np.asarray(mass, dtype=float)
    )
    if not np.all(np.isfinite(un)) or not np.all(np.isfinite(up)) or not np.all(np.isfinite(mm)):
        raise FiniteMonopoleError("contact inputs must be finite")
    if kap == 0.0:
        nn, dn = _species_density_from_nu(un, mm)
        np_, dp = _species_density_from_nu(up, mm)
        return np.zeros_like(un), nn, np_, dn + dp
    if np.allclose(un, up, rtol=0.0, atol=0.0):
        # Odd isovector density vanishes exactly in the symmetric limit.
        eta = np.zeros_like(un)
        nn, dn = _species_density_from_nu(un, mm)
        return eta, nn, nn.copy(), 2.0 * dn

    # Monotonic residual R(eta)=eta-kappa*(f(un-eta)-f(up+eta)).  A root is
    # bracketed by the zeros of either density and by conservative analytic
    # bounds; expanding the bracket handles both active and inactive seas.
    eta = np.zeros_like(un)
    span = kap * (np.maximum(un, mm) ** 3 + np.maximum(up, mm) ** 3) / (3.0 * math.pi**2)
    lo = -np.maximum(1.0, span + np.abs(kap * (un - up)))
    hi = np.maximum(1.0, span + np.abs(kap * (un - up)))

    def residual(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nn, dn = _species_density_from_nu(un - value, mm)
        pp, dp = _species_density_from_nu(up + value, mm)
        return value - kap * (nn - pp), 1.0 + kap * (dn + dp)

    rlo, _ = residual(lo)
    rhi, _ = residual(hi)
    # A finite expansion is deterministic; with the cubic density growth the
    # bracket is normally already generous, but retain a fail-closed guard.
    for _ in range(8):
        bad_lo, bad_hi = rlo > 0.0, rhi < 0.0
        if not np.any(bad_lo | bad_hi):
            break
        lo = np.where(bad_lo, lo * 2.0, lo)
        hi = np.where(bad_hi, hi * 2.0, hi)
        rlo, _ = residual(lo)
        rhi, _ = residual(hi)
    if np.any(rlo > 0.0) or np.any(rhi < 0.0):
        raise FiniteMonopoleError("contact KKT bracket failed")
    done = np.zeros_like(eta, dtype=bool)
    for _ in range(max(4, int(iterations))):
        rr, derivative = residual(eta)
        scale = np.maximum(1.0, np.abs(eta))
        done |= np.abs(rr) <= 1.0e-13 * scale
        trial = eta - rr / np.maximum(derivative, 1.0e-30)
        outside = (trial <= lo) | (trial >= hi) | ~np.isfinite(trial)
        trial = np.where(outside, 0.5 * (lo + hi), trial)
        rtrial, _ = residual(trial)
        lo = np.where(rtrial > 0.0, lo, trial)
        hi = np.where(rtrial > 0.0, trial, hi)
        eta = np.where(done, eta, trial)
        done |= np.abs(rtrial) <= 1.0e-13 * np.maximum(1.0, np.abs(trial))
    rr, derivative = residual(eta)
    if np.max(np.abs(rr)) > 5.0e-12 * max(1.0, float(np.max(np.abs(eta)))):
        # A final bracket midpoint is more robust at a moving TF edge.
        eta = 0.5 * (lo + hi)
        rr, derivative = residual(eta)
    if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(rr)):
        raise FiniteMonopoleError("contact KKT solve is nonfinite")
    nn, dn = _species_density_from_nu(un - eta, mm)
    pp, dp = _species_density_from_nu(up + eta, mm)
    return eta, nn, pp, dn + dp


@dataclass(frozen=True)
class FiniteDesign:
    """Numerical adapter preserving the live BulkModel/W8 potential."""

    family: str
    target_y: str | None
    scale: str
    M: float
    W0: float
    hbarc: float
    momega: float
    gomega: float
    q: float
    gs: float
    d: float
    n0_dim: float
    coefficients_dim: tuple[float, float, float]
    reference_y: float
    reference_mu: float

    @classmethod
    def from_family(
        cls,
        family: str,
        scale: str | float,
        *,
        allow_continuous: bool = False,
    ) -> "FiniteDesign":
        if family not in FAMILIES:
            raise FiniteMonopoleError(f"unsupported family {family}")
        scale_label = str(scale)
        if allow_continuous:
            try:
                scale_value = float(scale)
            except (TypeError, ValueError, OverflowError) as exc:
                raise FiniteMonopoleError("continuous scale must be finite and positive") from exc
            if not math.isfinite(scale_value) or scale_value <= 0.0:
                raise FiniteMonopoleError("continuous scale must be finite and positive")
            # Keep historical labels (e.g. ``1`` and ``0.5``) stable while
            # giving bridge-only values a deterministic representation.
            scale_label = format(scale_value, ".17g")
        elif scale_label not in SCALES:
            raise FiniteMonopoleError(f"unsupported correlated scale {scale}")
        s = float(scale_label)
        if family == "Q4":
            model = BulkModel()
            with np.errstate(all="ignore"):
                state = model.equilibrium(model.n0)
            W0 = float(model.W0) * s
            M = float(model.MN)
            hbarc = float(model.hbarc)
            momega = float(model.momega)
            gomega = float(model.gomega)
            # This is the live BulkModel quartic in z^2, not an invented
            # coefficient table.  Scaling changes its dimensionless units.
            coefficients = (float(model.A / (4 * model.W0**4)) / s**4, 0.0, 0.0)
            reference_y = float(state["y"])
            reference_mu = float(state["mu"])
            target = None
        else:
            target = "0." + family.split(".", 1)[1]
            source = finite.W8Design.from_target(target)
            W0 = float(source.W0) * s
            M = float(source.M)
            hbarc = float(source.hbarc)
            momega = float(source.momega)
            gomega = float(source.gomega)
            coefficients = tuple(float(value) / s**4 for value in source.coefficients_dim)
            reference_y = float(target)
            reference_mu = M - 16.0
        return cls(
            family=family,
            target_y=target,
            scale=scale_label,
            M=M,
            W0=W0,
            hbarc=hbarc,
            momega=momega,
            gomega=gomega,
            q=momega / W0,
            gs=M / W0,
            d=2.0,
            n0_dim=N0_FM3 * hbarc**3 / W0**3,
            coefficients_dim=coefficients,
            reference_y=reference_y,
            reference_mu=reference_mu,
        )

    @classmethod
    def from_continuous(cls, family: str, scale: str | float) -> "FiniteDesign":
        """Construct the same live design at an explicit bridge scale.

        The frozen legacy matrix continues to accept only ``SCALES``.  This
        opt-in constructor is the narrow seam used by the finite binding
        bridge for its bounded inverse search.
        """

        return cls.from_family(family, scale, allow_continuous=True)

    def adapter(self) -> Any:
        """Return the existing d=2 Fermi/potential adapter."""

        return finite.W8Design(
            target_y=self.target_y or "Q4",
            M=self.M,
            W0=self.W0,
            hbarc=self.hbarc,
            momega=self.momega,
            gomega=self.gomega,
            q=self.q,
            gs=self.gs,
            d=self.d,
            n0_dim=self.n0_dim,
            coefficients_dim=np.asarray(self.coefficients_dim, dtype=float),
            target_mu=self.reference_mu,
        )

    def potential(self, y: Any) -> np.ndarray:
        z = np.asarray(y, dtype=float) ** 2 - 1.0
        a2, a3, a4 = self.coefficients_dim
        return a2 * z**2 + a3 * z**3 + a4 * z**4

    def potential_y(self, y: Any) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        z = y * y - 1.0
        a2, a3, a4 = self.coefficients_dim
        return 4.0 * a2 * y * z + 6.0 * a3 * y * z**2 + 8.0 * a4 * y * z**3

    def potential_yy(self, y: Any) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        z = y * y - 1.0
        a2, a3, a4 = self.coefficients_dim
        return (
            a2 * (4.0 * z + 8.0 * y * y)
            + a3 * (6.0 * z**2 + 24.0 * y * y * z)
            + a4 * (8.0 * z**3 + 48.0 * y * y * z**2)
        )

    def fermi(self, n: Any, y: Any) -> dict[str, np.ndarray]:
        # Reuse the production adapter's cancellation-safe Fermi formulas.
        return self.adapter().fermi_from_density(np.asarray(n, dtype=float), np.asarray(y, dtype=float))

    def seed(
        self,
        x: np.ndarray,
        A: int,
        factor: float,
        rho_choice: str,
        *,
        Z: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct a regular two-sea/Maxwell seed with no result-table use."""

        radius = (3.0 * A / (4.0 * math.pi * self.n0_dim)) ** (1.0 / 3.0) * float(factor)
        width = max(1.0, 0.10 * radius)
        t = np.tanh((x - radius) / width)
        y = self.reference_y + (1.0 - self.reference_y) * 0.5 * (1.0 + t)
        yp = (1.0 - self.reference_y) * 0.5 / width * (1.0 - t * t)
        # Q4's local equilibrium y is close to one; keep the seed positive.
        y = np.maximum(y, 0.35)
        a0 = self.gomega * self.n0_dim / max(self.q**2 * self.reference_y**2, 1.0e-12)
        a = 0.5 * a0 * (1.0 - t)
        ap = -0.5 * a0 / width * (1.0 - t * t)
        number = self.n0_dim * np.minimum(x, radius) ** 3 / 3.0
        # Coulomb seed v=e*A/W0 for an approximately uniform proton sphere.
        if Z is None:
            # Historical callers only supplied A and retain their old
            # Zr90/Pb208 convention.  Bridge callers pass Z explicitly,
            # including Ca40.
            Z = 40 if A == 90 else 82
        c = np.where(
            x <= radius,
            ALPHA * Z / max(2.0 * radius, 1.0e-12) * (3.0 - (x / max(radius, 1.0e-12)) ** 2),
            ALPHA * Z / np.maximum(x, 1.0e-9),
        )
        cp = np.where(
            x <= radius,
            -ALPHA * Z * x / max(radius**3, 1.0e-12),
            -ALPHA * Z / np.maximum(x, 1.0e-9) ** 2,
        )
        return np.vstack((y, yp, a, ap, c, cp, number * (A - Z) / A, number * Z / A)), np.asarray((self.reference_mu, self.reference_mu), dtype=float)


@dataclass
class StaticSolution:
    design: FiniteDesign
    nucleus: str
    rho_choice: str
    N: int
    Z: int
    c_rho: float
    kappa: float
    solution: Any | None
    row: dict[str, Any]


def _rho_coupling(design: FiniteDesign, rho_choice: str) -> tuple[float, float]:
    if rho_choice == "no_rho":
        return 0.0, 0.0
    if rho_choice != "covariant_contact_J32":
        raise FiniteMonopoleError(f"unsupported rho choice {rho_choice}")
    # Reuse the prior homogeneous jet's coupling.  The covariant completion
    # is explicit in the contract; no range or new coefficient is introduced.
    # ``constant_rho_coupling`` expects a symmetric live BulkModel state.  We
    # reconstruct that state from the source producer for this call rather
    # than reading a previously serialized jet/result row.  The physical
    # coupling is invariant under the correlated W0 reparameterization.
    n0_nat = N0_FM3 * design.hbarc**3
    if design.family == "Q4":
        source_model = BulkModel()
        source_state = source_model.equilibrium(source_model.n0)
    else:
        source_model, source_state, _ = inverse_potential_jet(
            design.target_y, "240", "0.16", "-16"
        )
    coupling = float(constant_rho_coupling(source_model, source_state, J_DESIGN_MEV))
    if not math.isfinite(coupling) or coupling < 0.0:
        raise FiniteMonopoleError("declared J32 rho coupling is invalid")
    # eta=kappa*n3 in dimensionless KKT variables.
    return coupling, coupling * design.W0**2 / 4.0


def _make_static_functions(design: FiniteDesign, c_rho: float, kappa: float, N: int, Z: int):
    adapter = design.adapter()

    def matter(y: np.ndarray, a: np.ndarray, c: np.ndarray, mu: np.ndarray) -> dict[str, np.ndarray]:
        y = np.asarray(y, dtype=float)
        mass = design.gs * np.maximum(y, 1.0e-8)
        nu_n = float(mu[0]) / design.W0 - design.gomega * a
        nu_p = float(mu[1]) / design.W0 - design.gomega * a - c
        eta, nn, pp, _ = solve_contact_density_difference(nu_n, nu_p, mass, kappa)
        fn = adapter.fermi_from_density(nn, y)
        fp = adapter.fermi_from_density(pp, y)
        return {
            "nn": nn,
            "np": pp,
            "n": nn + pp,
            "n3": nn - pp,
            "eta_rho": eta,
            "ns": fn["ns"] + fp["ns"],
            "f": fn["f"] + fp["f"],
            "p": fn["p"] + fp["p"],
            "efn": fn["ef"],
            "efp": fp["ef"],
            "nu_n": nu_n,
            "nu_p": nu_p,
            "mass": mass,
        }

    def fun(x: np.ndarray, U: np.ndarray, mu: np.ndarray) -> np.ndarray:
        y, yp, a, ap, c, cp, qn, qp = U
        values = matter(y, a, c, mu)
        return np.vstack(
            (
                yp,
                design.potential_y(y) + design.gs * values["ns"] - design.q**2 * y * a * a,
                ap,
                design.q**2 * y * y * a - design.gomega * values["n"],
                cp,
                -4.0 * math.pi * ALPHA * values["np"],
                x * x * values["nn"],
                x * x * values["np"],
            )
        )

    def bc(left: np.ndarray, right: np.ndarray, mu: np.ndarray, X: float) -> np.ndarray:
        del mu
        return np.asarray(
            (
                left[1], left[3], left[5], left[6], left[7],
                right[0] - 1.0, right[2], right[5] + right[4] / X,
                4.0 * math.pi * right[6] - N, 4.0 * math.pi * right[7] - Z,
            ),
            dtype=float,
        )

    return matter, fun, bc


def _static_diagnostics_single(
    static: StaticSolution,
    x: np.ndarray,
    *,
    dense_intervals: int | None = None,
) -> dict[str, Any]:
    """Evaluate one deterministic diagnostic ladder for a converged BVP."""

    solution = static.solution
    if solution is None or solution.status != 0:
        raise FiniteMonopoleError("diagnostics require converged BVP")
    design, N, Z = static.design, static.N, static.Z
    matter, _, _ = _make_static_functions(design, static.c_rho, static.kappa, N, Z)
    if dense_intervals is None:
        dense_intervals = DIAGNOSTIC_INTERVALS
    xx = np.linspace(0.0, float(x[-1]), int(dense_intervals) + 1)
    U = np.asarray(solution.sol(xx), dtype=float)
    first = np.asarray(solution.sol(xx, 1), dtype=float)
    try:
        second = np.asarray(solution.sol(xx, 2), dtype=float)
    except (TypeError, ValueError):  # pragma: no cover - old SciPy fallback.
        second = np.vstack([np.gradient(first[i], xx, edge_order=2) for i in range(U.shape[0])])
    y, yp, a, ap, c, cp = U[:6]
    values = matter(y, a, c, np.asarray(solution.p, dtype=float))
    active = xx > max(1.0e-7, xx[-1] * 1.0e-8)
    scalar_lap = second[0] + np.divide(2.0 * yp, xx, out=np.zeros_like(xx), where=xx > 0.0)
    vector_lap = second[2] + np.divide(2.0 * ap, xx, out=np.zeros_like(xx), where=xx > 0.0)
    coul_lap = second[4] + np.divide(2.0 * cp, xx, out=np.zeros_like(xx), where=xx > 0.0)
    scalar_rhs = design.potential_y(y) + design.gs * values["ns"] - design.q**2 * y * a * a
    vector_rhs = design.q**2 * y * y * a - design.gomega * values["n"]
    coul_rhs = -4.0 * math.pi * ALPHA * values["np"]
    # First-order consistency is independent of the second derivative
    # interpolation residual.  The origin is excluded from 2/x terms because
    # regularity is already imposed by the BVP boundary conditions.
    first_state_y_abs = float(np.max(np.abs(first[0][active] - yp[active])))
    first_state_vector_abs = float(np.max(np.abs(first[2][active] - ap[active])))
    first_state_coul_abs = float(np.max(np.abs(first[4][active] - cp[active])))
    first_state_abs = float(max(first_state_y_abs, first_state_vector_abs, first_state_coul_abs))
    scalar_first = first[1] + np.divide(2.0 * yp, xx, out=np.zeros_like(xx), where=xx > 0.0) - scalar_rhs
    vector_first = first[3] + np.divide(2.0 * ap, xx, out=np.zeros_like(xx), where=xx > 0.0) - vector_rhs
    coul_first = first[5] + np.divide(2.0 * cp, xx, out=np.zeros_like(xx), where=xx > 0.0) - coul_rhs
    scalar_first_abs = float(np.max(np.abs(scalar_first[active])))
    vector_first_abs = float(np.max(np.abs(vector_first[active])))
    first_field_abs = float(max(scalar_first_abs, vector_first_abs))
    first_coul_abs = float(np.max(np.abs(coul_first[active])))
    field_abs = float(max(np.max(np.abs(scalar_lap[active] - scalar_rhs[active])), np.max(np.abs(vector_lap[active] - vector_rhs[active]))))
    field_scale = max(1.0e-3, float(np.max(np.abs(scalar_rhs))), float(np.max(np.abs(vector_rhs))))
    field_relative = field_abs / field_scale
    first_field_relative = first_field_abs / field_scale
    coul_abs = float(np.max(np.abs(coul_lap[active] - coul_rhs[active])))
    coul_scale = max(1.0e-3, float(np.max(np.abs(coul_rhs))))
    coul_relative = coul_abs / coul_scale
    first_coul_relative = first_coul_abs / coul_scale
    number_n = _spherical_integral(xx, values["nn"])
    number_p = _spherical_integral(xx, values["np"])
    number_rel = max(abs(number_n - N) / N, abs(number_p - Z) / Z)
    active_n = values["nn"] > 1.0e-12 * max(design.n0_dim, 1.0e-12)
    active_p = values["np"] > 1.0e-12 * max(design.n0_dim, 1.0e-12)
    if np.any(active_n) or np.any(active_p):
        kkt_n = np.abs(values["efn"][active_n] / design.W0 - values["nu_n"][active_n] + static.kappa * values["n3"][active_n]) * design.W0 if np.any(active_n) else np.zeros(1)
        kkt_p = np.abs(values["efp"][active_p] / design.W0 - values["nu_p"][active_p] - static.kappa * values["n3"][active_p]) * design.W0 if np.any(active_p) else np.zeros(1)
        kkt = float(max(np.max(kkt_n), np.max(kkt_p)))
    else:
        kkt = 0.0
    weight = 4.0 * math.pi * design.W0 * xx * xx
    t_w = _integral(xx, weight * 0.5 * yp * yp)
    v_u = _integral(xx, weight * design.potential(y))
    e_f = _integral(xx, weight * values["f"])
    t_a = _integral(xx, weight * 0.5 * ap * ap)
    v_a = _integral(xx, weight * 0.5 * design.q**2 * y * y * a * a)
    e_rho = _integral(xx, weight * static.kappa * values["n3"] ** 2 / 2.0)
    source_a = _integral(xx, weight * design.gomega * values["n"] * a)
    flux_a = 4.0 * math.pi * design.W0 * xx[-1] ** 2 * a[-1] * ap[-1]
    gauss_resid = 2.0 * (t_a + v_a) - source_a - flux_a
    # Maxwell energy includes the external Coulomb tail.  c is e*A/W0, so
    # W0/(8*pi*alpha) * [4*pi int x² c'^2 + 4*pi X c(X)^2].
    coul_field = design.W0 / (8.0 * math.pi * ALPHA) * (
        4.0 * math.pi * _integral(xx, xx * xx * cp * cp) + 4.0 * math.pi * xx[-1] * c[-1] ** 2
    )
    coul_source = design.W0 * 0.5 * _spherical_integral(xx, values["np"] * c)
    coul_identity = coul_source - coul_field
    energy = t_w + v_u + e_f + t_a + v_a + e_rho + coul_field
    pressure = _integral(xx, weight * values["p"])
    virial = -t_w - 3.0 * v_u + 3.0 * pressure + t_a + 3.0 * v_a + 3.0 * e_rho + coul_field
    rms = design.hbarc / design.W0 * math.sqrt(
        _integral(xx, xx**4 * values["n"]) / max(_integral(xx, xx**2 * values["n"]), 1.0e-30)
    )
    # Keep species moments separate.  The historical matrix only exposed the
    # baryon moment; the bridge needs the raw point-proton and neutron moments
    # before any finite-nucleon/charge-operator mapping is applied.
    n_den = _integral(xx, xx**2 * values["nn"])
    p_den = _integral(xx, xx**2 * values["np"])
    if n_den <= 0.0 or p_den <= 0.0:
        raise FiniteMonopoleError("species radius requires positive N and Z integrals")
    radius_scale = design.hbarc / design.W0
    rms_n = radius_scale * math.sqrt(_integral(xx, xx**4 * values["nn"]) / n_den)
    rms_p = radius_scale * math.sqrt(_integral(xx, xx**4 * values["np"]) / p_den)
    edge_n, edge_p = float(values["nn"][-1]), float(values["np"][-1])
    outer = xx > 0.8 * xx[-1]
    outer_fraction = max(
        _spherical_integral(xx[outer], values["nn"][outer]) / max(number_n, 1.0e-30),
        _spherical_integral(xx[outer], values["np"][outer]) / max(number_p, 1.0e-30),
    ) if np.count_nonzero(outer) > 3 else math.inf
    mu_n, mu_p = map(float, solution.p[:2])
    localized = bool(
        mu_n < design.M and mu_p < design.M and edge_n <= 1.0e-10 and edge_p <= 1.0e-10 and outer_fraction <= 1.0e-8
    )
    # A Coulomb turning radius is reported whenever mu_p exceeds the vacuum
    # threshold; a finite box edge is not allowed to masquerade as localization.
    turning = None
    if mu_p > design.M:
        turning = ALPHA * design.hbarc * Z / max(mu_p - design.M, 1.0e-30)
    stationarity = bool(
        number_rel <= N_RELATIVE_LIMIT
        and field_relative <= FIELD_RESIDUAL_LIMIT
        # Each first-order state identity is a separate absolute dimensionless
        # gate.  Do not let the aggregate maximum or a stored flag conceal a
        # bad scalar, vector, or Coulomb component.
        and all(
            math.isfinite(value) and value <= FIELD_RESIDUAL_LIMIT
            for value in (first_state_y_abs, first_state_vector_abs, first_state_coul_abs)
        )
        and first_field_relative <= FIELD_RESIDUAL_LIMIT
        and first_coul_relative <= 1.0e-4
        # The independent Coulomb source/field identity below is the Gauss
        # acceptance check.  A pointwise second derivative sampled through a
        # moving TF edge is a looser diagnostic and is not used to reject an
        # otherwise converged Maxwell solution.
        and coul_relative <= 1.0e-4
        and kkt <= KKT_LIMIT_MEV
        and abs(gauss_resid) / max(abs(2.0 * (t_a + v_a)), 1.0) <= GAUSS_LIMIT
        and abs(coul_identity) / max(abs(coul_field), 1.0) <= COULOMB_LIMIT
        and np.all(y > 0.0)
        and np.all(values["nn"] >= 0.0)
        and np.all(values["np"] >= 0.0)
    )
    return {
        "box_fm": float(xx[-1] * design.hbarc / design.W0),
        "x_box": float(xx[-1]),
        "chemical_potentials_MeV": {"neutron": mu_n, "proton": mu_p},
        "conserved_N_integrated": number_n,
        "conserved_Z_integrated": number_p,
        "conserved_NZ_relative_error": number_rel,
        "central_y": float(y[0]),
        "central_density_fm_minus3": float(values["n"][0] * design.W0**3 / design.hbarc**3),
        "min_y": float(np.min(y)),
        "field_residual_relative": field_relative,
        "first_order_state_residual_max_abs_dimensionless": first_state_abs,
        "first_order_state_y_residual_max_abs_dimensionless": first_state_y_abs,
        "first_order_state_vector_residual_max_abs_dimensionless": first_state_vector_abs,
        "first_order_state_coulomb_residual_max_abs_dimensionless": first_state_coul_abs,
        "first_order_field_residual_max_abs_dimensionless": first_field_abs,
        "first_order_scalar_ode_residual_max_abs_dimensionless": scalar_first_abs,
        "first_order_vector_ode_residual_max_abs_dimensionless": vector_first_abs,
        "first_order_field_residual_relative": first_field_relative,
        "scalar_residual_max_abs_dimensionless": float(np.max(np.abs(scalar_lap[active] - scalar_rhs[active]))),
        "vector_residual_max_abs_dimensionless": float(np.max(np.abs(vector_lap[active] - vector_rhs[active]))),
        "coulomb_residual_relative": coul_relative,
        "first_order_coulomb_residual_max_abs_dimensionless": first_coul_abs,
        "first_order_coulomb_residual_relative": first_coul_relative,
        "kkt_max_error_MeV": kkt,
        "energy_components_MeV": {
            "T_W": t_w, "V_U": v_u, "E_F": e_f, "T_A": t_a, "V_A": v_a,
            "E_rho": e_rho, "E_Coulomb_field_interior_plus_tail": coul_field,
            "E_Coulomb_source_half": coul_source, "total_E": energy,
        },
        "gauss_identity_residual_MeV": gauss_resid,
        "gauss_identity_relative": abs(gauss_resid) / max(abs(2.0 * (t_a + v_a)), 1.0),
        "coulomb_identity_residual_MeV": coul_identity,
        "coulomb_identity_relative": abs(coul_identity) / max(abs(coul_field), 1.0),
        "rms_baryon_radius_fm": rms,
        "integrated_fermi_pressure_MeV": pressure,
        "virial_D_MeV": virial,
        "virial_relative": abs(virial) / max(abs(energy), 1.0),
        "species_radius_integrals_dimensionless": {
            "neutron_x2": n_den,
            "neutron_x4": _integral(xx, xx**4 * values["nn"]),
            "proton_x2": p_den,
            "proton_x4": _integral(xx, xx**4 * values["np"]),
        },
        "rms_neutron_radius_fm": rms_n,
        "rms_proton_radius_fm": rms_p,
        "rms_point_proton_radius_fm": rms_p,
        "boundary_fields": {"a": float(a[-1]), "a_x": float(ap[-1]), "A_C_dimensionless": float(c[-1]), "A_C_x": float(cp[-1])},
        "boundary_densities_dimensionless": {"neutron": edge_n, "proton": edge_p},
        "outer_number_fraction_max": outer_fraction,
        "localized_vacuum_exterior": localized,
        "proton_turning_radius_fm": turning,
        "binding_condition_E_per_A_lt_M": bool(energy / max(N + Z, 1) < design.M),
        "energy_per_A_minus_M_MeV": energy / max(N + Z, 1) - design.M,
        "stationarity_acceptance": stationarity,
        "profile_grid_x": xx,
        "profile_y": y,
        "profile_yp": yp,
        "profile_a": a,
        "profile_ap": ap,
        "profile_c": c,
        "profile_cp": cp,
        "profile_nn": values["nn"],
        "profile_np": values["np"],
    }


def _static_diagnostics(
    static: StaticSolution,
    x: np.ndarray,
    *,
    dense_intervals: int | None = None,
) -> dict[str, Any]:
    """Run the fixed diagnostic ladder and its denser nested control.

    The default path always evaluates both declared ladders.  Residuals and
    gate booleans use the worst value/AND of the two; the nested ladder is
    never searched for a favourable phase or substituted as a hidden repair.
    An explicit ``dense_intervals`` is retained only as a focused-test seam
    for inspecting one ladder and does not alter production acceptance.
    """

    intervals = DIAGNOSTIC_INTERVALS if dense_intervals is None else int(dense_intervals)
    primary = _static_diagnostics_single(static, x, dense_intervals=intervals)
    if dense_intervals is not None:
        return primary
    nested = _static_diagnostics_single(static, x, dense_intervals=DIAGNOSTIC_NESTED_INTERVALS)
    residual_keys = (
        "conserved_NZ_relative_error",
        "field_residual_relative",
        "first_order_state_residual_max_abs_dimensionless",
        "first_order_state_y_residual_max_abs_dimensionless",
        "first_order_state_vector_residual_max_abs_dimensionless",
        "first_order_state_coulomb_residual_max_abs_dimensionless",
        "first_order_field_residual_max_abs_dimensionless",
        "first_order_scalar_ode_residual_max_abs_dimensionless",
        "first_order_vector_ode_residual_max_abs_dimensionless",
        "first_order_field_residual_relative",
        "scalar_residual_max_abs_dimensionless",
        "vector_residual_max_abs_dimensionless",
        "coulomb_residual_relative",
        "first_order_coulomb_residual_max_abs_dimensionless",
        "first_order_coulomb_residual_relative",
        "kkt_max_error_MeV",
        "gauss_identity_relative",
        "coulomb_identity_relative",
    )
    combined = dict(primary)
    for key in residual_keys:
        if key in primary and key in nested:
            combined[key] = max(float(primary[key]), float(nested[key]))
    # The second ladder has its own stationarity predicate; requiring both is
    # the fixed nested-control rule.  The primary ladder remains the canonical
    # source of energies/moments while the nested metrics govern acceptance.
    combined["stationarity_acceptance"] = bool(
        primary.get("stationarity_acceptance") is True
        and nested.get("stationarity_acceptance") is True
    )
    combined["localized_vacuum_exterior"] = bool(
        primary.get("localized_vacuum_exterior") is True
        and nested.get("localized_vacuum_exterior") is True
    )
    combined["binding_condition_E_per_A_lt_M"] = bool(
        primary.get("binding_condition_E_per_A_lt_M") is True
        and nested.get("binding_condition_E_per_A_lt_M") is True
    )
    combined["diagnostic_rule"] = {
        "primary_intervals": int(intervals),
        "nested_intervals": int(DIAGNOSTIC_NESTED_INTERVALS),
        "gate_combination": "worst_residual_and_boolean_AND",
        "phase_search": False,
    }
    combined["nested_diagnostic"] = {
        key: value for key, value in nested.items() if not str(key).startswith("profile_")
    }
    return combined


def _strip_profiles(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not str(key).startswith("profile_")}


def _solve_static_once(
    design: FiniteDesign,
    nucleus: str,
    rho_choice: str,
    *,
    box_fm: float,
    nodes: int,
    tol: float,
    factor: float | None = None,
    previous: StaticSolution | None = None,
    nucleus_data: Mapping[str, Mapping[str, Any]] | None = None,
) -> StaticSolution:
    table = NUCLEI if nucleus_data is None else nucleus_data
    try:
        spec = table[nucleus]
        A, Z, N = int(spec["A"]), int(spec["Z"]), int(spec["N"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FiniteMonopoleError(f"unsupported nucleus {nucleus}") from exc
    if A <= 0 or Z <= 0 or N < 0 or A != N + Z:
        raise FiniteMonopoleError(f"invalid N/Z bookkeeping for {nucleus}")
    c_rho, kappa = _rho_coupling(design, rho_choice)
    x_max = design.W0 * float(box_fm) / design.hbarc
    x = np.linspace(0.0, x_max, int(nodes))
    matter, fun, bc = _make_static_functions(design, c_rho, kappa, N, Z)
    seed_factor = None if factor is None else float(factor)
    if previous is None:
        if factor is None:
            raise FiniteMonopoleError("fresh solve needs a seed factor")
        guess, p0 = design.seed(x, A, factor, rho_choice, Z=Z)
        # Q4's homogeneous chemical potential is high; let the seed's live
        # branch provide its scale while W8 starts near its declared mu.
        if design.family == "Q4":
            p0 = np.asarray((design.M + 40.0, design.M + 40.0), dtype=float)
    else:
        old = previous.solution
        if old is None:
            raise FiniteMonopoleError("continuation source has no BVP")
        # A correlated scale changes W0 and therefore the dimensionless x
        # coordinate.  Continue on common physical r, not on an unchanged x
        # grid: x_old = x_new * W0_old/W0_new is the old coordinate at the
        # same radius, while a and c (and their derivatives) carry the inverse
        # W0 factors shown below.
        old_design = previous.design
        old_to_new = float(old_design.W0 / design.W0)
        x_old = x * old_to_new
        inside = x_old <= old.x[-1]
        sample_x = np.minimum(x_old, old.x[-1])
        old_state = np.asarray(old.sol(sample_x), dtype=float)
        old_first = np.asarray(old.sol(sample_x, 1), dtype=float)
        guess = np.empty_like(old_state)
        guess[0] = old_state[0]
        guess[1] = old_to_new * old_first[0]
        guess[2] = old_to_new * old_state[2]
        guess[3] = old_to_new**2 * old_first[2]
        guess[4] = old_to_new * old_state[4]
        guess[5] = old_to_new**2 * old_first[4]
        guess[6:] = old_state[6:]
        p0 = np.asarray(old.p, dtype=float)
        outside = ~inside
        if np.any(outside):
            guess[0, outside] = 1.0
            guess[1:4, outside] = 0.0
            guess[4, outside] = ALPHA * Z / np.maximum(x[outside], 1.0e-9)
            guess[5, outside] = -ALPHA * Z / np.maximum(x[outside], 1.0e-9) ** 2
            guess[6:, outside] = guess[6:, np.flatnonzero(~outside)[-1]][:, None]
    def boundary(left: np.ndarray, right: np.ndarray, p: np.ndarray) -> np.ndarray:
        return bc(left, right, p, x_max)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            solution = solve_bvp(
                fun,
                boundary,
                x,
                guess,
                p=p0,
                S=np.diag([0.0, -2.0, 0.0, -2.0, 0.0, -2.0, 0.0, 0.0]),
                tol=float(tol),
                max_nodes=max(int(nodes) * MAX_NODES_FACTOR, int(nodes) + 100),
                verbose=0,
            )
    except (ArithmeticError, FloatingPointError, RuntimeError, ValueError) as exc:
        row = {
            "solver_status": -1,
            "solver_message": str(exc),
            "converged": False,
            "box_fm": float(box_fm),
            "nodes_requested": int(nodes),
            "tolerance": float(tol),
            "seed_factor": seed_factor,
            "protocol_label": None,
        }
        return StaticSolution(design, nucleus, rho_choice, N, Z, c_rho, kappa, None, row)
    row: dict[str, Any] = {
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
        "converged": bool(solution.status == 0),
        "box_fm": float(box_fm),
        "x_box": float(x_max),
        "nodes_requested": int(nodes),
        "nodes_used": int(solution.x.size),
        "tolerance": float(tol),
        "seed_factor": seed_factor,
        "collocation_max_rms": float(np.max(solution.rms_residuals)) if solution.rms_residuals.size else math.inf,
        "chemical_potentials_MeV": [float(v) for v in solution.p],
        "protocol_label": None,
    }
    if solution.status == 0:
        try:
            row.update(_static_diagnostics(StaticSolution(design, nucleus, rho_choice, N, Z, c_rho, kappa, solution, row), x))
        except (ArithmeticError, FloatingPointError, RuntimeError, ValueError, FiniteMonopoleError) as exc:
            row.update({"converged": False, "diagnostic_failure": str(exc)})
    return StaticSolution(design, nucleus, rho_choice, N, Z, c_rho, kappa, solution, row)


def _profile_metrics(static: StaticSolution) -> dict[str, Any]:
    row = static.row
    return {
        "energy_per_A_minus_M_MeV": row.get("energy_per_A_minus_M_MeV"),
        "rms_baryon_radius_fm": row.get("rms_baryon_radius_fm"),
        "mu_n_MeV": (row.get("chemical_potentials_MeV") or [None, None])[0],
        "mu_p_MeV": (row.get("chemical_potentials_MeV") or [None, None])[1],
        "localized": row.get("localized_vacuum_exterior") is True,
        "stationary": row.get("stationarity_acceptance") is True,
    }


def _static_case(
    design: FiniteDesign,
    nucleus: str,
    rho_choice: str,
    *,
    nucleus_data: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the seed atlas and fixed refinement/domain protocol for one row."""

    table = NUCLEI if nucleus_data is None else nucleus_data
    try:
        spec = table[nucleus]
        N, Z = int(spec["N"]), int(spec["Z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FiniteMonopoleError(f"unsupported nucleus {nucleus}") from exc
    attempts: list[dict[str, Any]] = []
    solved: list[StaticSolution] = []
    coarse = STATIC_LEVELS[0]
    for factor in SEED_FACTORS:
        result = _solve_static_once(
            design, nucleus, rho_choice,
            box_fm=coarse["box_fm"], nodes=coarse["nodes"], tol=coarse["tol"], factor=factor,
            nucleus_data=table,
        )
        result.row["protocol_label"] = coarse["label"]
        attempts.append(_strip_profiles(result.row))
        if result.solution is not None and result.row.get("converged") is True:
            solved.append(result)
    # Merge seed paths only when their live observables are demonstrably one
    # branch; every original seed outcome remains in ``seed_attempts``.
    branches: list[StaticSolution] = []
    branch_seed_factors: dict[int, list[float]] = {}
    def _mu_n(row: Mapping[str, Any]) -> float:
        value = row.get("chemical_potentials_MeV", [0.0, 0.0])
        if isinstance(value, Mapping):
            return float(value.get("neutron", 0.0))
        return float(value[0]) if isinstance(value, (list, tuple)) and value else 0.0
    for item in solved:
        if any(
            abs(_mu_n(item.row) - _mu_n(existing.row)) < 0.2
            and abs(float(item.row.get("central_y", 0.0)) - float(existing.row.get("central_y", 0.0))) < 2.0e-3
            for existing in branches
        ):
            existing = next(
                existing for existing in branches
                if abs(_mu_n(item.row) - _mu_n(existing.row)) < 0.2
                and abs(float(item.row.get("central_y", 0.0)) - float(existing.row.get("central_y", 0.0))) < 2.0e-3
            )
            branch_seed_factors[id(existing)].append(float(item.row.get("seed_factor")))
            continue
        branches.append(item)
        branch_seed_factors[id(item)] = [float(item.row.get("seed_factor"))]
    branch_rows: list[dict[str, Any]] = []
    for index, branch in enumerate(branches, start=1):
        continuation = branch
        refinements: list[dict[str, Any]] = []
        for level in STATIC_LEVELS[1:]:
            next_row = _solve_static_once(
                design, nucleus, rho_choice,
                box_fm=level["box_fm"], nodes=level["nodes"], tol=level["tol"], previous=continuation,
                nucleus_data=table,
            )
            next_row.row["protocol_label"] = level["label"]
            refinements.append(_strip_profiles(next_row.row))
            continuation = next_row if next_row.solution is not None and next_row.row.get("converged") is True else continuation
        # The tightened 24-fm solve is deliberately fresh, not a continuation
        # of the larger domain.  It separates mesh/tolerance from tail error.
        tight = _solve_static_once(
            design, nucleus, rho_choice,
            box_fm=TIGHT24["box_fm"], nodes=TIGHT24["nodes"], tol=TIGHT24["tol"], factor=1.0,
            nucleus_data=table,
        )
        tight.row["protocol_label"] = TIGHT24["label"]
        domain = refinements[-1] if refinements and refinements[-1].get("converged") else _strip_profiles(continuation.row)
        fine = refinements[0] if refinements else _strip_profiles(branch.row)
        tight_row = _strip_profiles(tight.row)
        energy_domain = _safe_rel(domain.get("energy_per_A_minus_M_MeV", math.inf), fine.get("energy_per_A_minus_M_MeV", math.inf), floor=1.0e-3)
        # Absolute energy/radius changes are the declared protocol quantities.
        energy_change = abs(float(domain.get("energy_per_A_minus_M_MeV", math.inf)) - float(fine.get("energy_per_A_minus_M_MeV", math.inf)))
        radius_change = abs(float(domain.get("rms_baryon_radius_fm", math.inf)) - float(fine.get("rms_baryon_radius_fm", math.inf)))
        mesh_energy = abs(float(tight_row.get("energy_per_A_minus_M_MeV", math.inf)) - float(fine.get("energy_per_A_minus_M_MeV", math.inf)))
        mesh_radius = abs(float(tight_row.get("rms_baryon_radius_fm", math.inf)) - float(fine.get("rms_baryon_radius_fm", math.inf)))
        protocol = {
            "fine_to_domain_energy_change_MeV": energy_change,
            "fine_to_domain_radius_change_fm": radius_change,
            "tight24_to_fine_energy_change_MeV": mesh_energy,
            "tight24_to_fine_radius_change_fm": mesh_radius,
            "domain_converged": bool(domain.get("converged") is True),
            "tight24_converged": bool(tight_row.get("converged") is True),
            "domain_mesh_and_tail_pass": bool(energy_change <= ENERGY_DOMAIN_LIMIT_MEV and radius_change <= RADIUS_DOMAIN_LIMIT_FM),
            "tight_mesh_pass": bool(mesh_energy <= ENERGY_DOMAIN_LIMIT_MEV and mesh_radius <= RADIUS_DOMAIN_LIMIT_FM),
        }
        full = bool(
            protocol["domain_converged"]
            and protocol["tight24_converged"]
            and protocol["domain_mesh_and_tail_pass"]
            and protocol["tight_mesh_pass"]
            and domain.get("stationarity_acceptance") is True
        )
        classification = "UNRESOLVED_NUMERICAL_PROTOCOL"
        if domain.get("converged") is not True:
            classification = "FAILED_NUMERICAL_PROTOCOL"
        elif domain.get("localized_vacuum_exterior") is not True:
            classification = "CONVERGED_UNBOUND_OR_NONLOCALIZED"
        elif not full:
            classification = "CONVERGED_BUT_REFINEMENT_OR_RESIDUAL_LIMIT_MISSED"
        elif domain.get("binding_condition_E_per_A_lt_M") is True:
            classification = "NUMERICAL_STATIONARY_BOUND_CANDIDATE"
        else:
            classification = "NUMERICAL_STATIONARY_UNBOUND_LOCALIZED"
        dynamic: dict[str, Any]
        if classification == "NUMERICAL_STATIONARY_BOUND_CANDIDATE" and continuation.solution is not None:
            try:
                dynamic = _dynamic_response(continuation)
            except (ArithmeticError, FloatingPointError, RuntimeError, ValueError, FiniteMonopoleError) as exc:
                dynamic = {
                    "status": "DYNAMIC_CALCULATION_FAILED",
                    "reason": str(exc),
                    "Omega_MeV": None,
                    "C_MeV": None,
                    "B_total_MeV_minus1": None,
                }
        else:
            dynamic = {
                "status": "NOT_RUN_NONLOCALIZED_OR_UNRESOLVED",
                "claim_boundary": "conditional_dynamics_requires_a_localized_stationary_candidate",
                "Omega_MeV": None,
                "C_MeV": None,
                "B_total_MeV_minus1": None,
            }
        branch_rows.append({
            "branch_id": f"branch_{index}",
            "seed_factors": branch_seed_factors.get(id(branch), [float(branch.row.get("seed_factor"))]),
            "refinements": refinements,
            "larger_domain_check": dict(domain),
            "tight24_check": tight_row,
            "refinement_summary": protocol,
            "terminal_classification": classification,
            "terminal_protocol_acceptance": full,
            "terminal_stationarity_candidate": bool(classification == "NUMERICAL_STATIONARY_BOUND_CANDIDATE"),
            "proton_turning_diagnostic_required": bool(domain.get("proton_turning_radius_fm") is not None),
            "dynamics": dynamic,
        })
    if not branches:
        branch_rows.append({
            "branch_id": "no_converged_branch",
            "seed_factors": [],
            "refinements": [],
            "larger_domain_check": {"converged": False},
            "tight24_check": {"converged": False},
            "refinement_summary": {"domain_converged": False, "tight24_converged": False},
            "terminal_classification": "FAILED_NUMERICAL_PROTOCOL",
            "terminal_protocol_acceptance": False,
            "terminal_stationarity_candidate": False,
            "proton_turning_diagnostic_required": False,
            "dynamics": {
                "status": "NOT_RUN_NONLOCALIZED_OR_UNRESOLVED",
                "Omega_MeV": None,
                "C_MeV": None,
                "B_total_MeV_minus1": None,
            },
        })
    return {
        "family": design.family,
        "target_y": design.target_y,
        "scale": design.scale,
        "rho_choice": rho_choice,
        "rho_coupling_MeV_minus2": branches[0].c_rho if branches else _rho_coupling(design, rho_choice)[0],
        "kappa_dimensionless": branches[0].kappa if branches else _rho_coupling(design, rho_choice)[1],
        "nucleus": nucleus,
        "N": N,
        "Z": Z,
        "seed_attempts": attempts,
        "all_seed_outcomes_reported": len(attempts) == len(SEED_FACTORS),
        "distinct_branches": branch_rows,
        "case_status": "CONVERGED_BRANCHES_REPORTED" if branches else "NO_CONVERGED_BRANCH",
    }


def _density_derivative(x: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Derivative of a non-negative profile without introducing an edge floor."""

    x = np.asarray(x, dtype=float)
    density = np.maximum(np.asarray(density, dtype=float), 0.0)
    if x.size < 4:
        return np.gradient(density, x, edge_order=1)
    # PCHIP is shape-preserving at the TF edge and avoids cubic overshoot into
    # negative densities.  The derivative is still a diagnostic, not a new
    # physical smoothing scale.
    return np.asarray(PchipInterpolator(x, density, extrapolate=False).derivative()(x), dtype=float)


def _solve_coulomb_density(
    design: FiniteDesign,
    x: np.ndarray,
    np_p: np.ndarray,
    *,
    signed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact radial Maxwell solution for a prescribed spherical proton source."""

    x = np.asarray(x, dtype=float)
    np_p = np.asarray(np_p, dtype=float)
    if not signed:
        np_p = np.maximum(np_p, 0.0)
    if x.size < 3 or np.any(np.diff(x) <= 0.0):
        raise FiniteMonopoleError("Coulomb grid must be increasing")
    charge = cumulative_trapezoid(x * x * np_p, x, initial=0.0)
    # Integral from x to X of t*n_p(t) dt; reverse cumulative trapezoid is
    # numerically equivalent and keeps the boundary Robin condition explicit.
    tail = -cumulative_trapezoid((x * np_p)[::-1], x[::-1], initial=0.0)[::-1]
    c = 4.0 * math.pi * ALPHA * (
        np.divide(charge, x, out=np.zeros_like(x), where=x > 0.0) + tail
    )
    c[0] = 4.0 * math.pi * ALPHA * tail[0]
    cp = -4.0 * math.pi * ALPHA * np.divide(charge, x * x, out=np.zeros_like(x), where=x > 0.0)
    cp[0] = 0.0
    return c, cp


def _fixed_density_fields(
    design: FiniteDesign,
    x: np.ndarray,
    nn: np.ndarray,
    np_: np.ndarray,
    y_guess: np.ndarray,
    a_guess: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Re-solve scalar and Gauss fields at fixed prescribed densities."""

    adapter = design.adapter()
    nn_i = PchipInterpolator(x, np.maximum(nn, 0.0), extrapolate=False)
    np_i = PchipInterpolator(x, np.maximum(np_, 0.0), extrapolate=False)
    grid = np.linspace(float(x[0]), float(x[-1]), max(801, min(1601, x.size)))

    def fun(xx: np.ndarray, U: np.ndarray) -> np.ndarray:
        y, yp, a, ap = U
        nng = np.maximum(nn_i(xx), 0.0)
        npg = np.maximum(np_i(xx), 0.0)
        fm_n = adapter.fermi_from_density(nng, y)
        fm_p = adapter.fermi_from_density(npg, y)
        return np.vstack((
            yp,
            design.potential_y(y) + design.gs * (fm_n["ns"] + fm_p["ns"]) - design.q**2 * y * a * a,
            ap,
            design.q**2 * y * y * a - design.gomega * (nng + npg),
        ))

    def bc(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray((left[1], left[3], right[0] - 1.0, right[2]), dtype=float)

    # Interpolate the previous family onto the solve grid; no fixed result
    # table is involved, only the current live stationary profile.
    y0 = np.asarray(CubicSpline(x, y_guess)(grid), dtype=float)
    a0 = np.asarray(CubicSpline(x, a_guess)(grid), dtype=float)
    yp0 = np.asarray(CubicSpline(x, _density_derivative(x, y_guess))(grid), dtype=float)
    ap0 = np.asarray(CubicSpline(x, _density_derivative(x, a_guess))(grid), dtype=float)
    initial = np.vstack((y0, yp0, a0, ap0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solved = solve_bvp(
            fun,
            bc,
            grid,
            initial,
            S=np.diag([0.0, -2.0, 0.0, -2.0]),
            tol=2.0e-8,
            max_nodes=max(5000, 8 * grid.size),
            verbose=0,
        )
    if solved.status != 0:
        raise FiniteMonopoleError(f"fixed-density field solve failed: {solved.message}")
    fields = np.asarray(solved.sol(x), dtype=float)
    return fields[0], fields[1], fields[2], fields[3], solved


def _scaled_profile(
    static: StaticSolution,
    eta: float,
) -> dict[str, np.ndarray]:
    """Build exact-number dilated densities from the current live profile."""

    row = static.row
    x = np.asarray(row["profile_grid_x"], dtype=float)
    y0 = np.asarray(row["profile_y"], dtype=float)
    a0 = np.asarray(row["profile_a"], dtype=float)
    nn0 = np.maximum(np.asarray(row["profile_nn"], dtype=float), 0.0)
    np0 = np.maximum(np.asarray(row["profile_np"], dtype=float), 0.0)
    lam = math.exp(float(eta))
    xx = lam * x
    inside = xx <= x[-1]
    # PCHIP keeps the physical density nonnegative at the moving TF edge.
    ni = PchipInterpolator(x, nn0, extrapolate=False)
    pi = PchipInterpolator(x, np0, extrapolate=False)
    yi = CubicSpline(x, y0, extrapolate=False)
    ai = CubicSpline(x, a0, extrapolate=False)
    nn = np.zeros_like(x)
    np_ = np.zeros_like(x)
    y_seed = np.ones_like(x)
    a_seed = np.zeros_like(x)
    nn[inside] = np.maximum(ni(xx[inside]), 0.0) * lam**3
    np_[inside] = np.maximum(pi(xx[inside]), 0.0) * lam**3
    y_seed[inside] = yi(xx[inside])
    a_seed[inside] = ai(xx[inside])
    # Finite-grid interpolation of a compactly supported profile can lose a
    # tiny number at the moving edge.  Correct it explicitly and report the
    # correction; it is not a fit and does not alter the shape.
    for values, target in ((nn, static.N), (np_, static.Z)):
        current = _spherical_integral(x, values)
        if not math.isfinite(current) or current <= 0.0:
            raise FiniteMonopoleError("dilated profile has no positive number")
        values *= float(target) / current
    return {"x": x, "nn": nn, "np": np_, "y_seed": y_seed, "a_seed": a_seed, "lambda": np.asarray(lam)}


def _profile_energy(
    static: StaticSolution,
    profile: Mapping[str, np.ndarray],
    y: np.ndarray,
    yp: np.ndarray,
    a: np.ndarray,
    ap: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
    """Evaluate the full reduced energy, including the Coulomb exterior tail."""

    design = static.design
    x = np.asarray(profile["x"], dtype=float)
    nn = np.maximum(np.asarray(profile["nn"], dtype=float), 0.0)
    np_ = np.maximum(np.asarray(profile["np"], dtype=float), 0.0)
    adapter = design.adapter()
    fn = adapter.fermi_from_density(nn, y)
    fp = adapter.fermi_from_density(np_, y)
    c, cp = _solve_coulomb_density(design, x, np_)
    n3 = nn - np_
    weight = 4.0 * math.pi * design.W0 * x * x
    pieces = {
        "T_W": _integral(x, weight * 0.5 * yp * yp),
        "V_U": _integral(x, weight * design.potential(y)),
        "E_F": _integral(x, weight * (fn["f"] + fp["f"])),
        "T_A": _integral(x, weight * 0.5 * ap * ap),
        "V_A": _integral(x, weight * 0.5 * design.q**2 * y * y * a * a),
        "E_rho": _integral(x, weight * static.kappa * n3 * n3 / 2.0),
    }
    pieces["E_Coulomb_field_interior_plus_tail"] = design.W0 / (8.0 * math.pi * ALPHA) * (
        4.0 * math.pi * _integral(x, x * x * cp * cp) + 4.0 * math.pi * x[-1] * c[-1] ** 2
    )
    pieces["total_E"] = float(sum(pieces.values()))
    pieces["coulomb_potential"] = c
    pieces["coulomb_derivative"] = cp
    pieces["fermi_neutron_energy"] = fn["ef"]
    pieces["fermi_proton_energy"] = fp["ef"]
    return pieces["total_E"], pieces


def _linearized_fields(
    static: StaticSolution,
    x: np.ndarray,
    nn: np.ndarray,
    np_: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    density_derivative_n: np.ndarray | None = None,
    density_derivative_p: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the independent first-variation scalar/Gauss field BVP."""

    design = static.design
    adapter = design.adapter()
    if density_derivative_n is None:
        dnn = 3.0 * nn + x * _density_derivative(x, nn)
    else:
        dnn = np.asarray(density_derivative_n, dtype=float)
    if density_derivative_p is None:
        dnp = 3.0 * np_ + x * _density_derivative(x, np_)
    else:
        dnp = np.asarray(density_derivative_p, dtype=float)
    fm_n = adapter.fermi_from_density(nn, y)
    fm_p = adapter.fermi_from_density(np_, y)
    mass_dim = design.gs * np.maximum(y, 1.0e-8)
    en_n = fm_n["ef"] / design.W0
    en_p = fm_p["ef"] / design.W0
    nsprime_n = np.divide(3.0 * fm_n["ns"], mass_dim, out=np.zeros_like(y), where=mass_dim > 0.0) - np.divide(3.0 * nn, en_n, out=np.zeros_like(y), where=en_n > 0.0)
    nsprime_p = np.divide(3.0 * fm_p["ns"], mass_dim, out=np.zeros_like(y), where=mass_dim > 0.0) - np.divide(3.0 * np_, en_p, out=np.zeros_like(y), where=en_p > 0.0)
    nsprime = nsprime_n + nsprime_p
    scalar_diag = design.potential_yy(y) + design.gs**2 * nsprime - design.q**2 * a * a
    h = -2.0 * design.q**2 * y * a
    # At fixed scalar mass, ``d n_s / d n = m/E_F`` for each sea.  The
    # scalar-field source therefore sees the weighted species density
    # variation, not simply ``d(n_n+n_p)`` (the latter is the vector source).
    scalar_source = design.gs * (
        np.divide(mass_dim, en_n, out=np.zeros_like(y), where=en_n > 0.0) * dnn
        + np.divide(mass_dim, en_p, out=np.zeros_like(y), where=en_p > 0.0) * dnp
    )
    vector_source = dnn + dnp
    scalar_i = PchipInterpolator(x, scalar_source, extrapolate=False)
    vector_i = PchipInterpolator(x, vector_source, extrapolate=False)
    grid = np.linspace(float(x[0]), float(x[-1]), max(801, min(1601, x.size)))
    sy = CubicSpline(x, scalar_diag, extrapolate=False)
    hy = CubicSpline(x, h, extrapolate=False)
    yy = CubicSpline(x, y, extrapolate=False)
    aa = CubicSpline(x, a, extrapolate=False)

    def fun(xx: np.ndarray, U: np.ndarray) -> np.ndarray:
        dy, dyp, da, dap = U
        return np.vstack((
            dyp,
            sy(xx) * dy + hy(xx) * da + scalar_i(xx),
            dap,
            design.q**2 * yy(xx) ** 2 * da + 2.0 * design.q**2 * yy(xx) * aa(xx) * dy - design.gomega * vector_i(xx),
        ))

    def bc(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray((left[1], left[3], right[0], right[2]), dtype=float)

    # Use zero as a stable first guess; this is a linear problem and all
    # sources are finite on the compact numerical domain.
    initial = np.zeros((4, grid.size), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solved = solve_bvp(
            fun,
            bc,
            grid,
            initial,
            S=np.diag([0.0, -2.0, 0.0, -2.0]),
            tol=2.0e-8,
            max_nodes=max(5000, 8 * grid.size),
            verbose=0,
        )
    if solved.status != 0:
        raise FiniteMonopoleError(f"linearized field solve failed: {solved.message}")
    vals = np.asarray(solved.sol(x), dtype=float)
    return vals[0], vals[1], vals[2], vals[3]


def _analytic_curvature(
    static: StaticSolution,
    profile: Mapping[str, np.ndarray],
    y: np.ndarray,
    a: np.ndarray,
    dy: np.ndarray,
    dyp: np.ndarray,
    da: np.ndarray,
    dap: np.ndarray,
    density_derivative_n: np.ndarray | None = None,
    density_derivative_p: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Constrained Hessian curvature using the independent linearized BVP."""

    design = static.design
    x = np.asarray(profile["x"], dtype=float)
    nn = np.asarray(profile["nn"], dtype=float)
    np_ = np.asarray(profile["np"], dtype=float)
    dnn = 3.0 * nn + x * _density_derivative(x, nn) if density_derivative_n is None else np.asarray(density_derivative_n, dtype=float)
    dnp = 3.0 * np_ + x * _density_derivative(x, np_) if density_derivative_p is None else np.asarray(density_derivative_p, dtype=float)
    adapter = design.adapter()
    fn = adapter.fermi_from_density(nn, y)
    fp = adapter.fermi_from_density(np_, y)
    en_n, en_p = fn["ef"] / design.W0, fp["ef"] / design.W0
    mass_dim = design.gs * np.maximum(y, 1.0e-8)
    nsprime_n = np.divide(3.0 * fn["ns"], mass_dim, out=np.zeros_like(y), where=mass_dim > 0.0) - np.divide(3.0 * nn, en_n, out=np.zeros_like(y), where=en_n > 0.0)
    nsprime_p = np.divide(3.0 * fp["ns"], mass_dim, out=np.zeros_like(y), where=mass_dim > 0.0) - np.divide(3.0 * np_, en_p, out=np.zeros_like(y), where=en_p > 0.0)
    f_nn = np.divide(
        (fn["k"] / design.W0) ** 2,
        3.0 * nn * en_n,
        out=np.zeros_like(nn),
        where=nn > 1.0e-18,
    )
    f_pp = np.divide(
        (fp["k"] / design.W0) ** 2,
        3.0 * np_ * en_p,
        out=np.zeros_like(np_),
        where=np_ > 1.0e-18,
    )
    f_ny = np.divide(design.gs**2 * y, en_n, out=np.zeros_like(y), where=en_n > 0.0)
    f_py = np.divide(design.gs**2 * y, en_p, out=np.zeros_like(y), where=en_p > 0.0)
    scalar_matter = design.gs**2 * (nsprime_n + nsprime_p)
    # E'' of the fixed-density Fermi pieces plus scalar/vector Hessian.
    fermion = f_nn * dnn**2 + f_pp * dnp**2 + 2.0 * (f_ny * dnn + f_py * dnp) * dy + scalar_matter * dy**2
    scalar = dyp**2 + design.potential_yy(y) * dy**2
    vector = dap**2 + design.q**2 * (a * a * dy**2 + 4.0 * y * a * dy * da + y * y * da**2)
    rho = static.kappa * (dnn - dnp) ** 2
    weight = 4.0 * math.pi * design.W0 * x * x
    parts = {
        "fermion": _integral(x, weight * fermion),
        "scalar": _integral(x, weight * scalar),
        "vector": _integral(x, weight * vector),
        "rho": _integral(x, weight * rho),
    }
    # Maxwell is a quadratic form in the potential.  Re-solve the exact
    # radial source for the linearized proton density to retain the exterior
    # Coulomb tail in the Hessian.
    _, dcp = _solve_coulomb_density(design, x, dnp, signed=True)
    dcharge = cumulative_trapezoid(x * x * dnp, x, initial=0.0)
    dc = 4.0 * math.pi * ALPHA * (
        np.divide(dcharge, x, out=np.zeros_like(x), where=x > 0.0)
        - cumulative_trapezoid((x * dnp)[::-1], x[::-1], initial=0.0)[::-1]
    )
    dc[0] = -4.0 * math.pi * ALPHA * cumulative_trapezoid((x * dnp)[::-1], x[::-1], initial=0.0)[::-1][0]
    parts["coulomb"] = design.W0 / (4.0 * math.pi * ALPHA) * (
        4.0 * math.pi * _integral(x, x * x * dcp * dcp) + 4.0 * math.pi * x[-1] * dc[-1] ** 2
    )
    return float(sum(parts.values())), parts


def _dynamic_response(static: StaticSolution) -> dict[str, Any]:
    """Evaluate the conditional adiabatic TD--TF dilation on one static row."""

    row = static.row
    if row.get("converged") is not True or row.get("localized_vacuum_exterior") is not True or row.get("stationarity_acceptance") is not True:
        return {
            "status": "NOT_RUN_NONLOCALIZED_OR_UNRESOLVED",
            "claim_boundary": "conditional_dynamics_requires_a_localized_stationary_candidate",
            "Omega_MeV": None,
            "C_MeV": None,
            "B_total_MeV_minus1": None,
        }
    profile = {
        "x": np.asarray(row["profile_grid_x"], dtype=float),
        "nn": np.asarray(row["profile_nn"], dtype=float),
        "np": np.asarray(row["profile_np"], dtype=float),
    }
    x = profile["x"]
    y0 = np.asarray(row["profile_y"], dtype=float)
    yp0 = np.asarray(row["profile_yp"], dtype=float)
    a0 = np.asarray(row["profile_a"], dtype=float)
    ap0 = np.asarray(row["profile_ap"], dtype=float)
    # Base energy is recomputed from the same reduced functional used at eta=0.
    y_relaxed, yp_relaxed, a_relaxed, ap_relaxed, _ = _fixed_density_fields(
        static.design, x, profile["nn"], profile["np"], y0, a0
    )
    e0, eparts0 = _profile_energy(static, profile, y_relaxed, yp_relaxed, a_relaxed, ap_relaxed)
    derivative_rows: list[dict[str, Any]] = []
    profile_cache: dict[float, tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for h in ETA_STEPS:
        vals = []
        for eta in (h, -h):
            scaled = _scaled_profile(static, eta)
            ys, yps, aas, aaps, _ = _fixed_density_fields(
                static.design, x, scaled["nn"], scaled["np"], scaled["y_seed"], scaled["a_seed"]
            )
            ee, _parts = _profile_energy(static, scaled, ys, yps, aas, aaps)
            vals.append(ee)
            profile_cache[float(eta)] = (scaled, ys, yps, aas, aaps, ee)
        ep, em = vals
        derivative_rows.append({"h": h, "D_MeV": (ep - em) / (2.0 * h), "C_MeV": (ep - 2.0 * e0 + em) / h**2})
    # Richardson is applied to the two smallest steps, with O(h^2) error.
    c_richardson = derivative_rows[-1]["C_MeV"] + (derivative_rows[-1]["C_MeV"] - derivative_rows[-2]["C_MeV"]) / 3.0
    d_richardson = derivative_rows[-1]["D_MeV"] + (derivative_rows[-1]["D_MeV"] - derivative_rows[-2]["D_MeV"]) / 3.0
    fd_profile = profile_cache[float(-ETA_STEPS[-1])]
    fd_plus = profile_cache[float(ETA_STEPS[-1])]
    # The density derivative is taken from the declared finite-grid dilation
    # itself, including its exact-number normalization.  The field derivative
    # is then obtained independently from the linearized BVP (rather than by
    # differencing its solved fields), which cleanly separates the two checks.
    dnn_input = (fd_plus[0]["nn"] - fd_profile[0]["nn"]) / (2.0 * ETA_STEPS[-1])
    dnp_input = (fd_plus[0]["np"] - fd_profile[0]["np"]) / (2.0 * ETA_STEPS[-1])
    dy, dyp, da, dap = _linearized_fields(
        static, x, profile["nn"], profile["np"], y_relaxed, a_relaxed,
        dnn_input, dnp_input,
    )
    c_analytic, c_parts = _analytic_curvature(
        static, profile, y_relaxed, a_relaxed, dy, dyp, da, dap,
        dnn_input, dnp_input,
    )
    delta_y_fd = (fd_plus[1] - fd_profile[1]) / (2.0 * ETA_STEPS[-1])
    delta_a_fd = (fd_plus[3] - fd_profile[3]) / (2.0 * ETA_STEPS[-1])
    dy_error = float(np.max(np.abs(dy - delta_y_fd)) / max(float(np.max(np.abs(dy))), 1.0e-8))
    da_error = float(np.max(np.abs(da - delta_a_fd)) / max(float(np.max(np.abs(da))), 1.0e-8))
    # Low-frequency fluid inertia in natural units.  Every term is an actual
    # non-negative quadratic contribution; no phenomenological width appears.
    adapter = static.design.adapter()
    fn = adapter.fermi_from_density(profile["nn"], y_relaxed)
    fp = adapter.fermi_from_density(profile["np"], y_relaxed)
    efn, efp = fn["ef"] / static.design.W0, fp["ef"] / static.design.W0
    b_f = 4.0 * math.pi / static.design.W0 * _integral(x, x**4 * (profile["nn"] * efn + profile["np"] * efp))
    b_w = 4.0 * math.pi / static.design.W0 * _integral(x, x**2 * dy**2)
    # B_A includes the induced longitudinal-vector contribution.  The scalar
    # profile is the relaxed derivative, not a rigid rescaling.
    b_a = 4.0 * math.pi / static.design.W0 * _integral(
        x,
        x**4 * (static.design.gomega * (profile["nn"] + profile["np"]) + dap / np.maximum(x, 1.0e-30)) ** 2
        / np.maximum(static.design.q**2 * y_relaxed**2, 1.0e-30),
    )
    b_rho = 4.0 * math.pi / static.design.W0 * _integral(
        x, x**4 * static.kappa * (profile["nn"] - profile["np"]) ** 2
    )
    b_total = b_f + b_w + b_a + b_rho
    C = float(c_richardson)
    omega = math.sqrt(C / b_total) if C > 0.0 and b_total > 0.0 else None
    # Fast-field diagnostic on two radial meshes; it is a low-frequency
    # omission check only, not a full-plane stability theorem.
    fast = _fast_field_gap(static, x, profile["nn"], profile["np"], y_relaxed, a_relaxed, omega)
    fd_curvature_relative = abs(C - c_analytic) / max(abs(C), abs(c_analytic), 1.0e-12)
    inertia_controls = bool(math.isfinite(b_total) and b_total > 0.0 and all(v >= 0.0 and math.isfinite(v) for v in (b_f, b_w, b_a, b_rho)))
    controls = bool(
        omega is not None
        and C > 0.0
        and inertia_controls
        and abs(d_richardson) <= 2.0e-2
        and fd_curvature_relative <= CURVATURE_RELATIVE_LIMIT
        and dy_error <= CURVATURE_RELATIVE_LIMIT
        and da_error <= CURVATURE_RELATIVE_LIMIT
        and fast.get("available") is True
        and fast.get("epsilon_ad", math.inf) <= FAST_EPSILON_LIMIT
    )
    Q = 4.0 * math.pi / static.design.W0**2 * _integral(x, x**4 * (profile["nn"] + profile["np"]))
    transition = 3.0 * (profile["nn"] + profile["np"]) + x * (
        _density_derivative(x, profile["nn"]) + _density_derivative(x, profile["np"])
    )
    samples = np.linspace(0, x.size - 1, 24, dtype=int)
    transition_rows = [
        {"r_fm": float(x[idx] * static.design.hbarc / static.design.W0), "delta_natural": float(transition[idx])}
        for idx in samples
    ]
    return {
        "status": "COMPUTED_SINGLE_ADIABATIC_TD_TF_POLE" if controls else "COMPUTED_BUT_DYNAMIC_CONTROLS_UNRESOLVED",
        "claim_boundary": "one_coordinate_adiabatic_TD_TF_trial_pole_not_full_RRPA_or_measured_spectrum",
        "eta_definition": "ln(lambda)",
        "density_dilation": "n_i(eta,r)=exp(3eta)*n_i(exp(eta)*r)",
        "scalar_field_treatment": "relaxed_at_each_prescribed_density; rigid_scalar_is_not_the_default",
        "base_energy_MeV": e0,
        "derivative_rows": derivative_rows,
        "Richardson_D_MeV": d_richardson,
        "Richardson_C_MeV": C,
        "analytic_C_MeV": c_analytic,
        "analytic_C_components_MeV": c_parts,
        "curvature_fd_vs_analytic_relative": fd_curvature_relative,
        "linearized_deltaW_fd_relative": dy_error,
        "linearized_deltaV_fd_relative": da_error,
        "B_components_MeV_minus1": {"B_F": b_f, "B_W": b_w, "B_V": b_a, "B_rho": b_rho, "B_total": b_total},
        "C_MeV": C,
        "Omega_MeV": omega,
        "static_susceptibility_4Q2_over_C": 4.0 * Q * Q / C if C > 0.0 else None,
        "Q_MeV_minus2": Q,
        "pole_response": (
            {"form": "4 Q^2/(C-B z^2)", "weight_Q2_over_B_MeV3": 4.0 * Q * Q / b_total}
            if omega is not None else None
        ),
        "transition_density_samples": transition_rows,
        "fast_field_control": fast,
        "controls_pass": controls,
        "width_MeV": None,
        "fit_or_likelihood": None,
    }


def _fast_field_gap(
    static: StaticSolution,
    x: np.ndarray,
    nn: np.ndarray,
    np_: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    omega: float | None,
) -> dict[str, Any]:
    """Compute the declared ell=0 adiabatic field gap on two P1 meshes."""

    if omega is None or static.design.q <= 0.0:
        return {"available": False, "reason": "missing_positive_mode_or_vector_mass", "mesh_rows": [], "epsilon_ad": None}
    design = static.design
    adapter = design.adapter()
    rows: list[dict[str, Any]] = []
    try:
        fm_n = adapter.fermi_from_density(nn, y)
        fm_p = adapter.fermi_from_density(np_, y)
        mass_dim = design.gs * np.maximum(y, 1.0e-8)
        en_n, en_p = fm_n["ef"] / design.W0, fm_p["ef"] / design.W0
        nsprime = (
            np.divide(3.0 * fm_n["ns"], mass_dim, out=np.zeros_like(y), where=mass_dim > 0.0)
            - np.divide(3.0 * nn, en_n, out=np.zeros_like(y), where=en_n > 0.0)
            + np.divide(3.0 * fm_p["ns"], mass_dim, out=np.zeros_like(y), where=mass_dim > 0.0)
            - np.divide(3.0 * np_, en_p, out=np.zeros_like(y), where=en_p > 0.0)
        )
        scalar_diag = design.potential_yy(y) + design.gs**2 * nsprime - design.q**2 * a * a
        hfield = -2.0 * design.q**2 * y * a
        for n_intervals in FAST_GAP_TOLERANCES:
            if n_intervals < 8:
                raise FiniteMonopoleError("fast-field mesh too small")
            dx = float(x[-1]) / (n_intervals + 1)
            grid = dx * np.arange(1, n_intervals + 1)
            yy = CubicSpline(x, y)(grid)
            aa = CubicSpline(x, a)(grid)
            sw = CubicSpline(x, scalar_diag)(grid)
            hh = CubicSpline(x, hfield)(grid)
            N = n_intervals
            lap = np.diag(np.full(N, 2.0 / dx**2)) + np.diag(np.full(N - 1, -1.0 / dx**2), 1) + np.diag(np.full(N - 1, -1.0 / dx**2), -1)
            KA = lap + np.diag(design.q**2 * yy**2)
            KW = lap + np.diag(sw)
            inv_h = np.linalg.solve(KA, np.diag(hh))
            K = KW + np.diag(hh) @ inv_h
            # A symmetric finite-difference derivative for the induced metric.
            G = np.zeros((N, N), dtype=float)
            G[np.arange(N), np.arange(N)] = -1.0 / np.maximum(grid, 1.0e-30)
            G[np.arange(N - 1), np.arange(1, N)] += 1.0 / (2.0 * dx)
            G[np.arange(1, N), np.arange(N - 1)] += -1.0 / (2.0 * dx)
            gx = G @ inv_h
            metric = np.eye(N) + gx.T @ np.diag(1.0 / np.maximum(design.q**2 * yy**2, 1.0e-30)) @ gx
            # ``eigh`` is imported lazily to keep the module's base import
            # lightweight for no-write API tests.
            from scipy.linalg import eigh
            eig = eigh(0.5 * (K + K.T), 0.5 * (metric + metric.T), subset_by_index=(0, 0), check_finite=True)[0][0]
            gap = design.W0 * math.sqrt(float(eig)) if eig > 0.0 else None
            rows.append({"intervals": int(n_intervals), "lambda_dimless": float(eig), "gap_MeV": gap})
        if not all(row["gap_MeV"] is not None and math.isfinite(float(row["gap_MeV"])) for row in rows):
            return {"available": False, "reason": "nonpositive_fast_field_gap", "mesh_rows": rows, "epsilon_ad": None}
        gap_values = [float(row["gap_MeV"]) for row in rows]
        gap = gap_values[-1]
        mesh_relative = abs(gap_values[-1] - gap_values[-2]) / max(gap_values[-1], gap_values[-2], 1.0e-12)
        min_mass = design.momega * float(np.min(y))
        epsilon = max((omega / gap) ** 2, (omega / max(min_mass, 1.0e-12)) ** 2)
        return {
            "available": bool(mesh_relative <= 0.01 and math.isfinite(epsilon)),
            "mesh_rows": rows,
            "gap_MeV": gap,
            "mesh_relative": mesh_relative,
            "min_vector_mass_MeV": min_mass,
            "epsilon_ad": epsilon,
            "positive_lowest_generalized_eigenvalue": True,
        }
    except (ArithmeticError, FloatingPointError, np.linalg.LinAlgError, RuntimeError, ValueError) as exc:
        return {"available": False, "reason": str(exc), "mesh_rows": rows, "epsilon_ad": None}


def load_reference_data(path: Path | str = DATA_PATH) -> dict[str, Any]:
    """Load descriptive RCNP summaries, kept strictly outside solver inputs."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiniteMonopoleError(f"cannot read source-data record {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise FiniteMonopoleError("source-data record requires schema_version=1")
    summaries = payload.get("summaries")
    if not isinstance(summaries, dict) or set(summaries) != set(NUCLEI):
        raise FiniteMonopoleError("source-data record must contain Zr90 and Pb208 summaries")
    for nucleus, summary in summaries.items():
        if not isinstance(summary, dict):
            raise FiniteMonopoleError(f"source summary {nucleus} must be an object")
        for key in ("E_center_MeV", "Gamma_MeV", "m1_over_m0_MeV", "sqrt_m1_over_mminus1_MeV", "sqrt_m3_over_m1_MeV", "EWSR_percent"):
            if key not in summary:
                raise FiniteMonopoleError(f"source summary {nucleus} missing {key}")
        # Values are source summaries; asymmetric confidence bars are allowed
        # but every numerical endpoint must be finite.
        for value in summary.values():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise FiniteMonopoleError("source summary contains non-finite number")
    # Return a deep JSON copy so callers cannot mutate the loaded record.
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def _design_metadata(design: FiniteDesign) -> dict[str, Any]:
    return {
        "family": design.family,
        "target_y": design.target_y,
        "correlated_scale": design.scale,
        "M_MeV": design.M,
        "W0_MeV": design.W0,
        "momega_MeV": design.momega,
        "gomega": design.gomega,
        "gs": design.gs,
        "q_momega_over_W0": design.q,
        "degeneracy_per_species": design.d,
        "n0_fm_minus3": N0_FM3,
        "coefficients_U_over_W0^4_z2_z3_z4": list(design.coefficients_dim),
        "potential_source": (
            "source_complete_scaling_saturation_audit.BulkModel.potential"
            if design.family == "Q4"
            else "source_complete_scaling_saturation_audit.inverse_potential_jet"
        ),
        "calibration_inputs_not_predictions": True,
        "scale_is_conditional_alternative_not_fit": True,
    }


def _calculate_matrix(
    families: Sequence[str],
    scales: Sequence[str],
    nuclei: Sequence[str],
    *,
    include_dynamics: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for family in families:
        for scale in scales:
            design = FiniteDesign.from_family(family, scale)
            rho_values = ("no_rho",) if family == "Q4" else RHO_CHOICES
            for nucleus in nuclei:
                for rho_choice in rho_values:
                    case = _static_case(design, nucleus, rho_choice)
                    case["design"] = _design_metadata(design)
                    # A focused test may request static-only execution.  The
                    # default production matrix keeps dynamics on every
                    # accepted localized candidate.
                    if not include_dynamics:
                        for branch in case.get("distinct_branches", []):
                            if isinstance(branch, Mapping):
                                branch["dynamics"] = {
                                    "status": "NOT_REQUESTED_IN_FOCUSED_RUN",
                                    "Omega_MeV": None,
                                    "C_MeV": None,
                                    "B_total_MeV_minus1": None,
                                }
                    rows.append(case)
    return {"cases": rows, "static_case_count": len(rows)}


def calculate_with_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Run an explicitly scoped focused calculation for private tests.

    The ordinary public protocol is ``calculate()``.  This seam requires an
    explicit non-empty family/scale/nucleus selection and is intentionally not
    used by the default CLI, preventing accidental parameter selection from
    becoming a public result.
    """

    if not isinstance(options, Mapping):
        raise FiniteMonopoleError("focused options must be a mapping")
    required = ("families", "scales", "nuclei", "include_dynamics")
    if any(key not in options for key in required):
        raise FiniteMonopoleError("focused options must explicitly name families, scales, nuclei and include_dynamics")
    families = tuple(str(value) for value in options["families"])
    scales = tuple(str(value) for value in options["scales"])
    nuclei = tuple(str(value) for value in options["nuclei"])
    if not families or not scales or not nuclei or not isinstance(options["include_dynamics"], bool):
        raise FiniteMonopoleError("focused selections must be non-empty and include_dynamics must be bool")
    if any(value not in FAMILIES for value in families) or any(value not in SCALES for value in scales) or any(value not in NUCLEI for value in nuclei):
        raise FiniteMonopoleError("focused selections contain an unsupported protocol value")
    return _calculate_matrix(families, scales, nuclei, include_dynamics=options["include_dynamics"])


def calculate() -> dict[str, Any]:
    """Run the frozen 30-case matrix and conditional projected dynamics."""

    matrix = _calculate_matrix(FAMILIES, SCALES, tuple(NUCLEI), include_dynamics=True)
    references = load_reference_data()
    candidate_count = sum(
        1
        for case in matrix["cases"]
        for branch in case.get("distinct_branches", [])
        if branch.get("terminal_stationarity_candidate") is True
    )
    dynamic_count = sum(
        1
        for case in matrix["cases"]
        for branch in case.get("distinct_branches", [])
        if branch.get("dynamics", {}).get("status") == "COMPUTED_SINGLE_ADIABATIC_TD_TF_POLE"
    )
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "model_scope": {
            "finite_spherical_two_species_zero_temperature_TF": True,
            "species_degeneracy": 2,
            "nuclei": {name: dict(values) for name, values in NUCLEI.items()},
            "rho_choices": {"no_rho": "source-complete baseline", "covariant_contact_J32": "separate previously calibrated EFT alternative"},
            "coulomb": "Hartree radial Maxwell with exterior Robin/tail energy",
            "shell_spin_orbit_pairing": False,
            "finite_range_rho": False,
            "full_finite_nucleus_RRPA": False,
            "full_experimental_spectrum": False,
            "no_parameter_refit_or_width": True,
        },
        "protocol": {
            "families": list(FAMILIES),
            "correlated_scales": list(SCALES),
            "seed_factors": list(SEED_FACTORS),
            "static_levels": [dict(level) for level in STATIC_LEVELS],
            "tight24_independent": dict(TIGHT24),
            "eta_steps": list(ETA_STEPS),
            "fast_field_mesh_intervals": list(FAST_GAP_TOLERANCES),
            "alpha_CODATA_2022": ALPHA,
            "equations": {
                "kkt": "mu_n=E_n+gomega V0+C_rho n3/4; mu_p=E_p+gomega V0+e A_C-C_rho n3/4",
                "scalar": "-Lap W+U'(W)+gs rho_s-qphi^2 W V0^2=0",
                "vector": "(-Lap+qphi^2 W^2)V0=gomega(n_n+n_p)",
                "coulomb": "-Lap A_C=e n_p with A_C'(R)+A_C(R)/R=0",
                "dilation": "n_i(eta,r)=exp(3eta)n_i(exp(eta)r)",
                "response": "chi_QQ(z)=4 Q^2/(C-B z^2)",
            },
            "acceptance_limits": {
                "N_Z_relative": N_RELATIVE_LIMIT,
                "field_relative": FIELD_RESIDUAL_LIMIT,
                "KKT_MeV": KKT_LIMIT_MEV,
                "Gauss_relative": GAUSS_LIMIT,
                "Coulomb_relative": COULOMB_LIMIT,
                "energy_domain_MeV": ENERGY_DOMAIN_LIMIT_MEV,
                "radius_domain_fm": RADIUS_DOMAIN_LIMIT_FM,
                "curvature_and_inertia_relative": CURVATURE_RELATIVE_LIMIT,
                "fast_epsilon": FAST_EPSILON_LIMIT,
            },
        },
        "input_provenance": {
            "potential_producer": "verification/source_complete_scaling_saturation_audit.py",
            "finite_adapter": "verification/nvg_finite_droplet_audit.py:W8Design",
            "rho_producer": "verification/nvg_isospin_jet_audit.py:constant_rho_coupling",
            "source_sha256": hashlib.sha256((HERE / "source_complete_scaling_saturation_audit.py").read_bytes()).hexdigest(),
            "finite_adapter_sha256": hashlib.sha256((HERE / "nvg_finite_droplet_audit.py").read_bytes()).hexdigest(),
            "rho_source_sha256": hashlib.sha256((HERE / "nvg_isospin_jet_audit.py").read_bytes()).hexdigest(),
            "baseline_inputs": dict(INPUTS),
            "no_saved_result_table_used_as_solver_input": True,
            "experimental_data_used_as_solver_input": False,
        },
        "static_case_count": matrix["static_case_count"],
        "candidate_count": candidate_count,
        "dynamic_pole_count": dynamic_count,
        "cases": matrix["cases"],
        "experimental_reference": {
            "source_record": "verification/data/rcnp_isgmr_2025_summaries.json",
            "data_are_descriptive_metadata_only": True,
            "moment_energy_window": None,
            "moment_window_status": "MISSING_IN_PRIMARY_PRODUCT; NOT_DIRECTLY_COMPARABLE",
            "summaries": references["summaries"],
            "comparison_status": "SCALAR_SUMMARY_ONLY_NO_SPECTRUM_LIKELIHOOD_OR_COVARIANCE",
        },
        "scientific_limits": [
            "A bound row is a stationary candidate, not a proof of a global ground state or full stability.",
            "Positive proton chemical potential is reported as a Coulomb turning-radius diagnostic, not localization.",
            "The conditional pole is a one-coordinate adiabatic TD-TF trial and is not a finite-nucleus Dirac/RRPA spectrum.",
            "No width, fit centroid, likelihood, or empirical validation is produced without a machine-readable spectrum and covariance.",
        ],
    }


_CACHE: dict[str, Any] | None = None


def build_result() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _jsonable(calculate())
    return _CACHE


def _is_finite_json(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_is_finite_json(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite_json(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _validate_structure(result: Mapping[str, Any]) -> bool:
    if not isinstance(result, Mapping) or result.get("schema_version") != SCHEMA or result.get("status") != STATUS or result.get("evidence_weight") != 0.0:
        return False
    if not _is_finite_json(result):
        return False
    cases = result.get("cases")
    if not isinstance(cases, list) or len(cases) != 30 or result.get("static_case_count") != 30:
        return False
    expected = {
        (family, scale, nucleus, rho)
        for family in FAMILIES
        for scale in SCALES
        for nucleus in NUCLEI
        for rho in (("no_rho",) if family == "Q4" else RHO_CHOICES)
    }
    actual = set()
    for case in cases:
        if not isinstance(case, Mapping):
            return False
        key = (case.get("family"), case.get("scale"), case.get("nucleus"), case.get("rho_choice"))
        if key in actual or key not in expected:
            return False
        actual.add(key)
        if case.get("all_seed_outcomes_reported") is not True or not isinstance(case.get("seed_attempts"), list) or len(case["seed_attempts"]) != 3:
            return False
        branches = case.get("distinct_branches")
        if not isinstance(branches, list) or not branches:
            return False
        for branch in branches:
            if not isinstance(branch, Mapping) or "terminal_classification" not in branch or "dynamics" not in branch:
                return False
            if branch.get("terminal_stationarity_candidate") is True:
                dyn = branch.get("dynamics")
                if not isinstance(dyn, Mapping):
                    return False
                if dyn.get("status") not in {"COMPUTED_SINGLE_ADIABATIC_TD_TF_POLE", "COMPUTED_BUT_DYNAMIC_CONTROLS_UNRESOLVED", "DYNAMIC_CALCULATION_FAILED"}:
                    return False
    return actual == expected


def validate_result(result: Mapping[str, Any]) -> bool:
    """Fail closed on altered rows, controls, or coverage."""

    try:
        if not _validate_structure(result):
            return False
        return result == build_result()
    except (ArithmeticError, FiniteMonopoleError, KeyError, TypeError, ValueError):
        return False


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - parser path.
        raise FiniteMonopoleError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = JsonArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = build_result()
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ArithmeticError, FiniteMonopoleError, RuntimeError, ValueError) as exc:
        print(json.dumps({
            "schema_version": SCHEMA,
            "status": "INVALID_OR_FAILED_FINITE_MONOPOLE",
            "evidence_weight": EVIDENCE_WEIGHT,
            "error": str(exc),
        }, ensure_ascii=False, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
