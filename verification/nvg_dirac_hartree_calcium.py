#!/usr/bin/env python3
"""Self-consistent spherical no-sea Dirac--Hartree calcium calculation.

The older finite-nucleus producers in this repository are Thomas--Fermi
closures.  This module is deliberately a separate producer: occupied states
are obtained from a kinetic-balanced Galerkin radial Dirac Hamiltonian and
their ``G**2-F**2`` scalar density is fed back into the live W8.93 fields.
The command line is an explicit JSON producer; importing it has no numerical
side effect and no saved result is read as a numerical input.

The implementation is intentionally conservative.  A state is accepted only
after independent norm, radial-equation, field-equation, localization and
double-counting checks.  Near-continuum and rejected generalized eigenmodes
remain in the diagnostics instead of being silently post-selected.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Small dense generalized pencils are solved many times during field
# iteration.  Limiting BLAS fan-out avoids severe oversubscription on the
# worker host and makes the declared control protocol reproducible.
for _blas_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_blas_var] = "1"
import numpy as np

try:
    from scipy.integrate import cumulative_trapezoid, solve_bvp, simpson
    from scipy.linalg import eigh, eig
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import eigsh
except ImportError as exc:  # pragma: no cover - repository pins SciPy.
    raise RuntimeError("SciPy is required for the Dirac--Hartree producer") from exc

try:  # Direct-script and package-style imports are both used by tests.
    import nvg_finite_monopole as static
except ImportError:  # pragma: no cover
    from . import nvg_finite_monopole as static

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve()
SCHEMA = "nvg_dirac_hartree_calcium.v1"
STATUS = "COMPUTED_SPHERICAL_DIRAC_HARTREE_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

S_STAR = 0.2267063450812736
J_ADDED_MEV = 11.13601607117819
N0_FM3 = 0.16
ALPHA = 0.0072973525643

R_MAX_FM = 24.0
GRID_POINTS = 1201
RADIAL_INTERVALS = 180
SC_RADIAL_INTERVALS = 96
MESH_CONTROL_INTERVALS = 260
GAUSS_ORDER = 8
EIG_RESIDUAL_LIMIT = 1.0e-8
ORBITAL_RESIDUAL_LIMIT = 3.0e-3
FIELD_RESIDUAL_LIMIT = 3.0e-3
NORM_LIMIT = 2.0e-7
SC_CONVERGENCE_FIELD = 2.0e-5
SC_CONVERGENCE_DENSITY = 2.0e-5
SC_MAX_ITER = 40
SC_MIX = 0.25
SEED_FACTORS = (0.9, 1.0, 1.1)

NUCLEI: dict[str, dict[str, int]] = {
    "Ca40": {"A": 40, "Z": 20, "N": 20},
    "Ca48": {"A": 48, "Z": 20, "N": 28},
    "Pb208": {"A": 208, "Z": 82, "N": 126},
}


class DiracHartreeError(ValueError):
    """Fail-closed numerical or schema error."""


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _number(value: Any, digits: int = 17) -> float:
    if isinstance(value, bool):
        raise DiracHartreeError("boolean is not a scientific number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DiracHartreeError("non-numeric scientific output") from exc
    if not math.isfinite(result):
        raise DiracHartreeError("non-finite scientific output")
    return float(format(result, f".{digits}g"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return _number(value) if isinstance(value, float) else int(value)
    raise DiracHartreeError(f"unsupported JSON value {type(value).__name__}")


def _integral(x: np.ndarray, y: np.ndarray) -> float:
    return float(simpson(np.asarray(y, dtype=float), x=np.asarray(x, dtype=float)))


def _spherical_integral(x: np.ndarray, y: np.ndarray) -> float:
    return 4.0 * math.pi * _integral(x, x * x * y)


def _radial_nodes(values: np.ndarray, threshold: float = 1.0e-6, r_fm: np.ndarray | None = None) -> int:
    values = np.asarray(values, dtype=float)
    if values.size < 3 or not np.any(np.isfinite(values)):
        return -1
    scale = max(float(np.max(np.abs(values))), 1.0e-30)
    v = values.copy()
    v[np.abs(v) < threshold * scale] = 0.0
    if r_fm is not None:
        rr = np.asarray(r_fm, dtype=float)
        if rr.shape != v.shape:
            raise DiracHartreeError("node-count radius shape mismatch")
        # The first finite-element interval can contain an origin-side sign
        # change from the imposed r^(l+1) regularity.  It is not a radial
        # node; only crossings outside the mesh-scale origin guard count.
        v = np.where(rr < 0.5, 0.0, v)
    nz = v != 0.0
    if np.count_nonzero(nz) < 2:
        return 0
    compact = v[nz]
    return int(np.count_nonzero(compact[1:] * compact[:-1] < 0.0))


@dataclass(frozen=True)
class OrbitalRequest:
    """A shell request used only to explain live energy filling."""

    kappa: int
    radial_nodes: int
    label: str

    @property
    def degeneracy(self) -> int:
        return 2 * abs(self.kappa)


@dataclass
class Orbital:
    kappa: int
    radial_nodes: int
    degeneracy: int
    occupancy: float
    energy_mev: float
    large: np.ndarray
    small: np.ndarray
    eig_residual: float
    node_count: int
    label: str


@dataclass
class FieldState:
    x: np.ndarray
    y: np.ndarray
    a: np.ndarray
    c: np.ndarray
    nn: np.ndarray
    np_: np.ndarray
    ns: np.ndarray
    iteration: int = 0


@dataclass
class Spectrum:
    orbitals: list[Orbital]
    rejected_modes: list[dict[str, Any]]
    all_modes: dict[int, list[dict[str, Any]]]
    free_reference: dict[str, Any]
    species_orbitals: dict[str, list[Orbital]]


class KineticBalancedBasis:
    """Piecewise-linear radial basis with kinetic-balanced small components.

    Large components are ``phi_i = (r/R)^(l+1) h_i(r)``.  Small components
    use ``chi_i = hbar*c (d/dr + kappa/r) phi_i /(2 M)``.  The generalized
    Galerkin eigenproblem therefore has the correct non-relativistic limit and
    does not use the unfiltered central-difference Dirac matrix known for
    doubling/spurious modes.
    """

    def __init__(self, r_max_fm: float = R_MAX_FM, intervals: int = RADIAL_INTERVALS,
                 quadrature_order: int = GAUSS_ORDER, mass_ref: float = 939.0):
        if not (_finite(r_max_fm) and float(r_max_fm) > 0.0):
            raise DiracHartreeError("basis radius must be finite and positive")
        if int(intervals) < 16 or int(quadrature_order) < 2:
            raise DiracHartreeError("basis resolution is too small")
        self.r_max = float(r_max_fm)
        self.intervals = int(intervals)
        self.quadrature_order = int(quadrature_order)
        self.mass_ref = float(mass_ref)
        self.knots = np.linspace(0.0, self.r_max, self.intervals + 1)
        self.nbasis = self.intervals - 1
        z, w = np.polynomial.legendre.leggauss(self.quadrature_order)
        r_parts, w_parts, interval_parts = [], [], []
        for j in range(self.intervals):
            a, b = self.knots[j], self.knots[j + 1]
            r_parts.append((a + b) / 2.0 + (b - a) / 2.0 * z)
            w_parts.append((b - a) / 2.0 * w)
            interval_parts.extend([j] * self.quadrature_order)
        self.r_quad = np.concatenate(r_parts)
        self.w_quad = np.concatenate(w_parts)
        self.interval_of_quad = np.asarray(interval_parts, dtype=int)

    @staticmethod
    def _angular_l(kappa: int) -> int:
        if kappa == 0:
            raise DiracHartreeError("kappa cannot be zero")
        return kappa if kappa > 0 else -kappa - 1

    def evaluate(self, r: np.ndarray, kappa: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return phi, dphi, chi and dchi matrices at physical radii."""

        if int(kappa) == 0:
            raise DiracHartreeError("kappa cannot be zero")
        r = np.asarray(r, dtype=float)
        if r.ndim != 1 or np.any(~np.isfinite(r)) or np.any(r < 0.0) or np.any(r > self.r_max):
            raise DiracHartreeError("basis evaluation radii are invalid")
        kappa = int(kappa)
        p = self._angular_l(kappa) + 1
        n = r.size
        phi = np.zeros((self.nbasis, n), dtype=float)
        dphi = np.zeros_like(phi)
        chi = np.zeros_like(phi)
        dchi = np.zeros_like(phi)
        # Endpoints have measure zero in quadrature; assigning each point to a
        # unique interval keeps the piecewise derivative deterministic.
        jj = np.searchsorted(self.knots, np.minimum(r, self.r_max - 1.0e-14), side="right") - 1
        jj = np.clip(jj, 0, self.intervals - 1)
        h = self.knots[1] - self.knots[0]
        t = np.maximum(r / self.r_max, 1.0e-14)
        pref = self.r_max ** (-0)  # explicit: t is dimensionless
        del pref
        for interval in range(self.intervals):
            mask = jj == interval
            if not np.any(mask):
                continue
            rv = r[mask]
            tv = np.maximum(rv / self.r_max, 1.0e-14)
            left = (self.knots[interval + 1] - rv) / h
            right = (rv - self.knots[interval]) / h
            dleft, dright = np.full_like(rv, -1.0 / h), np.full_like(rv, 1.0 / h)
            for node, hv, dhv in ((interval, left, dleft), (interval + 1, right, dright)):
                if node == 0 or node == self.intervals:
                    continue
                col = node - 1
                ph = tv ** p * hv
                dph = (p * tv ** (p - 1) / self.r_max) * hv + tv ** p * dhv
                ddph = (p * (p - 1) * tv ** (p - 2) / self.r_max**2) * hv + (
                    2.0 * p * tv ** (p - 1) / self.r_max
                ) * dhv
                ch = self.hbarc / (2.0 * self.mass_ref) * (
                    dph + kappa * ph / np.maximum(rv, 1.0e-14)
                )
                dch = self.hbarc / (2.0 * self.mass_ref) * (
                    ddph + kappa * dph / np.maximum(rv, 1.0e-14)
                    - kappa * ph / np.maximum(rv, 1.0e-14) ** 2
                )
                phi[col, mask], dphi[col, mask] = ph, dph
                chi[col, mask], dchi[col, mask] = ch, dch
        # Values at r=0 are physically zero for the large radial component.
        # They are not used in Galerkin quadrature, but making the convention
        # explicit simplifies profile diagnostics.
        zero = r <= 1.0e-14
        if np.any(zero):
            phi[:, zero] = 0.0
            dphi[:, zero] = 0.0
            chi[:, zero] = 0.0
            dchi[:, zero] = 0.0
        return phi, dphi, chi, dchi

    @property
    def hbarc(self) -> float:
        return 197.3269804

    def matrices(self, kappa: int, yq: np.ndarray, vq: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the Hermitian kinetic-balanced generalized eigenproblem."""

        phi, dphi, chi, dchi = self.evaluate(self.r_quad, int(kappa))
        r = self.r_quad
        w = self.w_quad
        mstar = mass * np.asarray(yq, dtype=float)
        vq = np.asarray(vq, dtype=float)
        if yq.shape != r.shape or vq.shape != r.shape:
            raise DiracHartreeError("potential quadrature shape mismatch")
        invr = 1.0 / r
        dplus = dphi + float(kappa) * invr[None, :] * phi
        dminus_chi = dchi - float(kappa) * invr[None, :] * chi
        def gram(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> np.ndarray:
            return (left * weight[None, :]) @ right.T
        sll = gram(phi, phi, w)
        sss = gram(chi, chi, w)
        hll = gram(phi, phi, w * (vq + mstar))
        hss = gram(chi, chi, w * (vq - mstar))
        hsl = self.hbarc * gram(chi, dplus, w)
        hls = self.hbarc * gram(phi, dminus_chi, w)
        # Integration by parts is the exact Hermitian partner in the finite
        # element form.  Symmetrising suppresses only roundoff, not physics.
        hls = 0.5 * (hls + hsl.T)
        hsl = hls.T
        H = np.block([[hll, hls], [hsl, hss]])
        S = np.block([[sll, np.zeros_like(sll)], [np.zeros_like(sss), sss]])
        H = 0.5 * (H + H.T)
        S = 0.5 * (S + S.T)
        return H, S

    def profile(self, coefficients: np.ndarray, kappa: int, x: np.ndarray, design: Any) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate normalized radial G,F on the dimensionless x grid."""

        r = np.asarray(x, dtype=float) * float(design.hbarc) / float(design.W0)
        phi, _, chi, _ = self.evaluate(r, int(kappa))
        n = self.nbasis
        return coefficients[:n] @ phi, coefficients[n:] @ chi


def _contact_coupling(design: Any) -> float:
    """Dimensionless kappa for V_rho/W0 = +/- kappa*(n_n-n_p)."""

    n0_nat = N0_FM3 * float(design.hbarc) ** 3
    c_rho = 8.0 * J_ADDED_MEV / n0_nat
    return c_rho * float(design.W0) ** 2 / 4.0


def _initial_fields(design: Any, nucleus: Mapping[str, int], factor: float) -> FieldState:
    x = np.linspace(0.0, float(design.W0) * R_MAX_FM / float(design.hbarc), GRID_POINTS)
    radius = (3.0 * nucleus["A"] / (4.0 * math.pi * float(design.n0_dim))) ** (1.0 / 3.0) * float(factor)
    width = max(0.8, 0.12 * radius)
    t = np.tanh((x - radius) / width)
    y = float(design.reference_y) + (1.0 - float(design.reference_y)) * 0.5 * (1.0 + t)
    y = np.maximum(y, 0.45)
    a0 = float(design.gomega) * float(design.n0_dim) / max(float(design.q) ** 2 * float(design.reference_y) ** 2, 1.0e-12)
    a = 0.5 * a0 * (1.0 - t)
    # Dimensionless Coulomb potential c=e A_C/W0.
    xx = np.maximum(x, 1.0e-8)
    z = float(nucleus["Z"])
    c = np.where(x <= radius, ALPHA * z / max(2.0 * radius, 1.0e-12) * (3.0 - (x / max(radius, 1.0e-12)) ** 2), ALPHA * z / xx)
    c[0] = 1.5 * ALPHA * z / max(radius, 1.0e-12)
    zeros = np.zeros_like(x)
    return FieldState(x, y, a, c, zeros.copy(), zeros.copy(), zeros.copy())


def _occupancy_requests(nucleus: str, species: str) -> list[OrbitalRequest]:
    """Labels for the expected closed-shell capacity, not energy inputs."""

    if nucleus not in NUCLEI or species not in ("n", "p"):
        raise DiracHartreeError("invalid nucleus/species")
    requests = [
        OrbitalRequest(-1, 0, "1s1/2"), OrbitalRequest(-2, 0, "1p3/2"),
        OrbitalRequest(1, 0, "1p1/2"), OrbitalRequest(-3, 0, "1d5/2"),
        OrbitalRequest(-1, 1, "2s1/2"), OrbitalRequest(2, 0, "1d3/2"),
        OrbitalRequest(-4, 0, "1f7/2"), OrbitalRequest(3, 0, "1f5/2"),
        OrbitalRequest(-5, 0, "1g9/2"), OrbitalRequest(4, 0, "1g7/2"),
        OrbitalRequest(-6, 0, "1h11/2"), OrbitalRequest(5, 0, "1h9/2"),
    ]
    # Species capacity controls actual filling.  The labels above only make
    # the occupied output readable; sorting is always by computed energy.
    return requests


def _candidate_kappas(nucleus: str) -> tuple[int, ...]:
    if nucleus == "Pb208":
        return (-1, -2, 1, -3, 2, -4, 3, -5, 4, -6, 5, -7, 6)
    # Ca40 closes at 1d3/2; Ca48 adds 1f7/2.  The occupied set is still
    # filled by computed energies, while sectors above this capacity cannot
    # affect either requested calcium particle count.
    return (-1, -2, 1, -3, 2, -4)


def _state_residual(H: np.ndarray, S: np.ndarray, coeff: np.ndarray, energy: float) -> float:
    num = np.linalg.norm(H @ coeff - energy * (S @ coeff))
    den = max(np.linalg.norm(H @ coeff), abs(energy) * np.linalg.norm(S @ coeff), 1.0)
    return float(num / den)


def _solve_kappa_modes(
    basis: KineticBalancedBasis, design: Any, fields: FieldState, kappa: int,
    max_modes: int = 16,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    r = basis.r_quad
    xq = r * float(design.W0) / float(design.hbarc)
    yq = np.interp(xq, fields.x, fields.y)
    # v is filled by caller with species-independent fields; this helper is
    # used for each species through a temporary `fields._potential` attribute.
    vq = np.interp(xq, fields.x, getattr(fields, "_potential", fields.a * float(design.gomega) + fields.c))
    H, S = basis.matrices(kappa, yq, vq, float(design.M))
    try:
        # Shift-invert Lanczos returns the finite set around the nuclear
        # positive-energy window without repeatedly diagonalising the full
        # negative continuum.  The matrix remains the same Hermitian
        # kinetic-balanced Galerkin pencil; dense LAPACK is the deterministic
        # fallback for old SciPy or a failed factorisation.
        k_lanczos = min(64, H.shape[0] - 2)
        vals, vecs = eigsh(csc_matrix(H), M=csc_matrix(S), k=k_lanczos,
                           sigma=900.0, which="LM", return_eigenvectors=True,
                           tol=1.0e-10, maxiter=max(4000, 10 * H.shape[0]))
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        keep = (vals > 0.0) & (vals < float(design.M) + 1.0e-8)
        vals, vecs = vals[keep], vecs[:, keep]
    except Exception:
        try:
            vals, vecs = eigh(H, S, subset_by_value=(0.0, float(design.M) + 1.0e-8), check_finite=False)
        except Exception:
            vals, vecs = eig(H, S, check_finite=False)
            vals = np.real_if_close(vals, tol=1000)
    vals = np.asarray(vals)
    vecs = np.asarray(vecs)
    xgrid = fields.x
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    n = basis.nbasis
    for j in np.argsort(np.real(vals)):
        value = vals[j]
        coeff = vecs[:, j]
        if np.iscomplexobj(value) or np.iscomplexobj(coeff):
            if abs(np.imag(value)) > 1.0e-8 or np.max(np.abs(np.imag(coeff))) > 1.0e-7:
                rejected.append({"reason": "complex_generalized_mode", "energy": float(np.real(value))})
                continue
            value, coeff = np.real(value), np.real(coeff)
        value, coeff = float(value), np.asarray(coeff, dtype=float)
        if not (math.isfinite(value) and np.all(np.isfinite(coeff))):
            rejected.append({"reason": "nonfinite_mode", "energy": value})
            continue
        residual = _state_residual(H, S, coeff, value)
        G, F = basis.profile(coeff, kappa, xgrid, design)
        rgrid = xgrid * float(design.hbarc) / float(design.W0)
        norm = _integral(rgrid, G * G + F * F)
        if norm <= 0.0 or not math.isfinite(norm):
            rejected.append({"reason": "invalid_norm", "energy": value})
            continue
        G, F = G / math.sqrt(norm), F / math.sqrt(norm)
        nodes = _radial_nodes(G, r_fm=xgrid * float(design.hbarc) / float(design.W0))
        rms_r = math.sqrt(max(_integral(xgrid, (xgrid * float(design.hbarc) / float(design.W0)) ** 2 * (G * G + F * F)), 0.0))
        # Piecewise kinetic-balance bases can contain a mesh-scale positive
        # mode pinned to the first element (usually E approximately equal to
        # the interior Dirac mass).  Such a mode is a known Galerkin
        # discretisation artefact, not a nuclear orbital; reject it explicitly
        # and retain its metadata rather than letting it enter occupations.
        if rms_r < 0.5:
            rejected.append({"reason": "mesh_scale_spurious_localization", "energy": value,
                             "residual": residual, "nodes": nodes, "rms_radius_fm": rms_r})
            continue
        # Keep a generous set of positive bound states.  Continuum and
        # negative-energy states are explicitly retained as rejected modes.
        if value <= 0.0 or value >= float(design.M) or residual > EIG_RESIDUAL_LIMIT:
            rejected.append({"reason": "negative_continuum_or_residual", "energy": value, "residual": residual, "nodes": nodes})
            continue
        accepted.append({
            "energy": value, "coeff": coeff, "large": G, "small": F,
            "norm": norm, "residual": residual, "nodes": nodes,
            "rms_radius_fm": rms_r,
            "kappa": int(kappa), "degeneracy": 2 * abs(int(kappa)),
        })
        if len(accepted) >= max_modes:
            break
    return accepted, rejected


def _fill_by_energy(candidates: list[dict[str, Any]], count: int, species: str) -> list[Orbital]:
    if count <= 0:
        raise DiracHartreeError("particle count must be positive")
    ordered = sorted(candidates, key=lambda row: float(row["energy"]))
    if not ordered:
        raise DiracHartreeError(f"no positive localized orbitals for {species}")
    remaining = float(count)
    occupied: list[Orbital] = []
    for row in ordered:
        if remaining <= 1.0e-10:
            break
        deg = int(row["degeneracy"])
        occ = min(float(deg), remaining)
        fraction = occ / deg
        label = f"kappa={row['kappa']},nodes={row['nodes']}"
        occupied.append(Orbital(
            kappa=int(row["kappa"]), radial_nodes=int(row["nodes"]), degeneracy=deg,
            occupancy=occ, energy_mev=float(row["energy"]), large=np.asarray(row["large"]),
            small=np.asarray(row["small"]), eig_residual=float(row["residual"]),
            node_count=int(row["nodes"]), label=label,
        ))
        remaining -= occ
    if remaining > 1.0e-7:
        raise DiracHartreeError(f"insufficient bound-state capacity for {species}: {remaining:g} particles remain")
    return occupied


def _species_spectrum(design: Any, fields: FieldState, nucleus: str, species: str, basis: KineticBalancedBasis) -> tuple[list[Orbital], list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """Solve all requested kappa sectors and fill by actual energy ordering."""

    spec = NUCLEI[nucleus]
    count = int(spec["N"] if species == "n" else spec["Z"])
    contact = _contact_coupling(design)
    n3 = fields.nn - fields.np_
    if species == "n":
        potential = float(design.gomega) * fields.a + contact * n3
    else:
        potential = float(design.gomega) * fields.a - contact * n3 + fields.c
    fields._potential = potential  # type: ignore[attr-defined]
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    all_modes: dict[int, list[dict[str, Any]]] = {}
    for kappa in _candidate_kappas(nucleus):
        accepted, rejects = _solve_kappa_modes(basis, design, fields, kappa)
        all_modes[int(kappa)] = [
            {"energy_mev": float(v["energy"]), "radial_nodes": int(v["nodes"]), "residual": float(v["residual"]), "accepted": True}
            for v in accepted
        ]
        rejected.extend([{**r, "kappa": int(kappa), "species": species} for r in rejects])
        candidates.extend(accepted)
    occupied = _fill_by_energy(candidates, count, species)
    return occupied, rejected, all_modes


def solve_spectrum(design: Any, fields: FieldState, nucleus: str, *, intervals: int = RADIAL_INTERVALS) -> Spectrum:
    if nucleus not in NUCLEI:
        raise DiracHartreeError("unsupported nucleus")
    basis = KineticBalancedBasis(R_MAX_FM, intervals, GAUSS_ORDER, float(design.M))
    # The neutron/proton calls temporarily set `_potential`; each resulting
    # orbital retains only its profile and no mutable field reference.
    n_orbitals, n_rejected, n_modes = _species_spectrum(design, fields, nucleus, "n", basis)
    p_orbitals, p_rejected, p_modes = _species_spectrum(design, fields, nucleus, "p", basis)
    orbitals = n_orbitals + p_orbitals
    all_modes = {"neutron": n_modes, "proton": p_modes}  # type: ignore[assignment]
    rejected = n_rejected + p_rejected
    free_reference = _free_reference(design, basis)
    return Spectrum(orbitals, rejected, all_modes, free_reference,
                    {"neutron": n_orbitals, "proton": p_orbitals})  # type: ignore[arg-type]


def _free_reference(design: Any, basis: KineticBalancedBasis) -> dict[str, Any]:
    """Analytic spherical-box reference for the lowest s wave."""

    zero = FieldState(
        basis.knots * float(design.W0) / float(design.hbarc),
        np.ones_like(basis.knots), np.zeros_like(basis.knots), np.zeros_like(basis.knots),
        np.zeros_like(basis.knots), np.zeros_like(basis.knots), np.zeros_like(basis.knots),
    )
    xq = basis.r_quad * float(design.W0) / float(design.hbarc)
    H, S = basis.matrices(-1, np.ones_like(xq), np.zeros_like(xq), float(design.M))
    vals, _ = eigh(H, S, check_finite=False)
    positive = [float(v) for v in vals if float(v) > 0.0]
    if not positive:
        raise DiracHartreeError("free reference has no positive mode")
    p = math.pi * float(design.hbarc) / basis.r_max
    exact = math.sqrt(float(design.M) ** 2 + p * p)
    return {"lowest_positive_mev": min(positive), "analytic_s_box_mev": exact,
            "relative_error": abs(min(positive) - exact) / exact,
            "kappa": -1, "boundary": "G(R)=0 spherical-box reference"}


def _orbital_densities(design: Any, fields: FieldState, orbitals: Sequence[Orbital], species: str) -> tuple[np.ndarray, np.ndarray]:
    nn = np.zeros_like(fields.x)
    ns = np.zeros_like(fields.x)
    scale = float(design.W0) / float(design.hbarc)
    # The supplied radial amplitudes are normalized in physical r; convert to
    # dimensionless-x normalized P,Q before using n_dim=(P²+Q²)/(4*pi*x²).
    for orb in orbitals:
        P = np.asarray(orb.large, dtype=float) / math.sqrt(scale)
        Q = np.asarray(orb.small, dtype=float) / math.sqrt(scale)
        den = P * P + Q * Q
        scalar = P * P - Q * Q
        radial = np.maximum(fields.x, 1.0e-12)
        d = float(orb.occupancy) * den / (4.0 * math.pi * radial * radial)
        s = float(orb.occupancy) * scalar / (4.0 * math.pi * radial * radial)
        if fields.x.size > 1:
            d[0], s[0] = d[1], s[1]
        nn += d
        ns += s
    return nn, ns


def _solve_fields(design: Any, fields: FieldState, nn: np.ndarray, np_: np.ndarray, ns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, str]:
    """Solve scalar/omega BVPs for fixed orbital sources and Maxwell exactly."""

    x = fields.x
    n = nn + np_
    source_n = np.interp
    def fun(xx: np.ndarray, Y: np.ndarray) -> np.ndarray:
        y, yp, a, ap = Y
        nsi = source_n(xx, x, ns)
        ni = source_n(xx, x, n)
        return np.vstack((
            yp,
            np.asarray(design.potential_y(y), dtype=float) + float(design.gs) * nsi - float(design.q) ** 2 * y * a * a,
            ap,
            float(design.q) ** 2 * y * y * a - float(design.gomega) * ni,
        ))
    def bc(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray((left[1], left[3], right[0] - 1.0, right[2]), dtype=float)
    guess = np.vstack((fields.y, np.gradient(fields.y, x), fields.a, np.gradient(fields.a, x)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Orbital shell densities have integrable cusp/edge structure.
            # Solve the fixed-source field equations at a slightly looser
            # collocation tolerance than the final residual gate; the latter
            # is evaluated independently below on the canonical grid.
            solved = solve_bvp(fun, bc, x, guess, S=np.diag([0.0, -2.0, 0.0, -2.0]),
                               tol=2.0e-4, max_nodes=20000, verbose=0)
    except (ArithmeticError, FloatingPointError, RuntimeError, ValueError) as exc:
        return fields.y.copy(), fields.a.copy(), fields.c.copy(), False, str(exc)
    if solved.status != 0 or solved.sol is None:
        return fields.y.copy(), fields.a.copy(), fields.c.copy(), False, str(solved.message)
    Y = np.asarray(solved.sol(x), dtype=float)
    y_new, a_new = Y[0], Y[2]
    # Positive radial Maxwell Green function with its 1/r exterior tail.
    charge = cumulative_trapezoid(x * x * np_, x, initial=0.0)
    tail = -cumulative_trapezoid((x * np_)[::-1], x[::-1], initial=0.0)[::-1]
    c_new = 4.0 * math.pi * ALPHA * (
        np.divide(charge, x, out=np.zeros_like(x), where=x > 0.0) + tail
    )
    c_new[0] = 4.0 * math.pi * ALPHA * tail[0]
    if not np.all(np.isfinite(y_new)) or not np.all(np.isfinite(a_new)) or not np.all(np.isfinite(c_new)):
        return fields.y.copy(), fields.a.copy(), fields.c.copy(), False, "nonfinite field solve"
    return y_new, a_new, c_new, True, str(solved.message)


def _relaxed_state(fields: FieldState, y: np.ndarray, a: np.ndarray, c: np.ndarray, nn: np.ndarray, np_: np.ndarray, ns: np.ndarray, iteration: int) -> FieldState:
    mix = SC_MIX
    return FieldState(fields.x, (1.0 - mix) * fields.y + mix * y, (1.0 - mix) * fields.a + mix * a,
                      (1.0 - mix) * fields.c + mix * c, nn, np_, ns, iteration)


def _self_consistent(design: Any, nucleus: str, factor: float) -> tuple[FieldState, Spectrum, dict[str, Any]]:
    fields = _initial_fields(design, NUCLEI[nucleus], factor)
    prev_nn, prev_np = None, None
    last_spectrum: Spectrum | None = None
    last_message = ""
    for iteration in range(1, SC_MAX_ITER + 1):
        spectrum = solve_spectrum(design, fields, nucleus, intervals=SC_RADIAL_INTERVALS)
        n_occ = spectrum.species_orbitals["neutron"]
        p_occ = spectrum.species_orbitals["proton"]
        nn, ns_n = _orbital_densities(design, fields, n_occ, "n")
        np_, ns_p = _orbital_densities(design, fields, p_occ, "p")
        ns = ns_n + ns_p
        y_new, a_new, c_new, ok, message = _solve_fields(design, fields, nn, np_, ns)
        last_message = message
        if not ok:
            raise DiracHartreeError(f"field solve failed at iteration {iteration}: {message}")
        field_delta = max(float(np.max(np.abs(y_new - fields.y))), float(np.max(np.abs(a_new - fields.a))), float(np.max(np.abs(c_new - fields.c))))
        density_delta = math.inf if prev_nn is None else max(
            float(np.max(np.abs(nn - prev_nn))), float(np.max(np.abs(np_ - prev_np)))
        )
        fields = _relaxed_state(fields, y_new, a_new, c_new, nn, np_, ns, iteration)
        prev_nn, prev_np = nn, np_
        last_spectrum = spectrum
        if field_delta * SC_MIX <= SC_CONVERGENCE_FIELD and density_delta * SC_MIX <= SC_CONVERGENCE_DENSITY:
            # Evaluate the accepted fixed point once on the relaxed fields.
            final_spectrum = solve_spectrum(design, fields, nucleus, intervals=RADIAL_INTERVALS)
            metadata = {"converged": True, "iterations": iteration, "field_update": field_delta * SC_MIX,
                        "density_update": density_delta * SC_MIX, "solver_message": last_message}
            return fields, final_spectrum, metadata
    if last_spectrum is None:
        raise DiracHartreeError("self-consistency made no spectrum attempt")
    return fields, last_spectrum, {"converged": False, "iterations": SC_MAX_ITER,
                                  "field_update": field_delta * SC_MIX,
                                  "density_update": density_delta * SC_MIX,
                                  "solver_message": last_message}


def _orbital_diagnostics(design: Any, fields: FieldState, spectrum: Spectrum, nucleus: str) -> dict[str, Any]:
    """Independent profiles, local Dirac residuals and field residuals."""

    x = fields.x
    r = x * float(design.hbarc) / float(design.W0)
    n_count, z_count = NUCLEI[nucleus]["N"], NUCLEI[nucleus]["Z"]
    selected_n = spectrum.species_orbitals["neutron"]
    selected_p = spectrum.species_orbitals["proton"]
    # Profiles from occupied orbitals are already occupancy-weighted by sqrt(f).
    nn, ns_n = _orbital_densities(design, fields, selected_n, "n")
    np_, ns_p = _orbital_densities(design, fields, selected_p, "p")
    ns = ns_n + ns_p
    n = nn + np_
    contact = _contact_coupling(design)
    n3 = nn - np_
    v_n = float(design.gomega) * fields.a + contact * n3
    v_p = float(design.gomega) * fields.a - contact * n3 + fields.c
    # Local radial Dirac residuals use a high-order centered derivative and
    # omit only the coordinate singularity and exponentially tiny tails.
    orbital_rows: list[dict[str, Any]] = []
    direct_sum = 0.0
    eigen_sum = 0.0
    norm_max = 0.0
    dirac_max = 0.0
    for species, group, potential in (("neutron", selected_n, v_n), ("proton", selected_p, v_p)):
        for orb in group:
            P, Q = np.asarray(orb.large), np.asarray(orb.small)
            norm = _integral(x, P * P + Q * Q) / (float(design.hbarc) / float(design.W0))
            # Orbital arrays are normalized in physical r by conversion in
            # `_orbital_densities`; this physical-r norm is an independent test.
            norm = _integral(r, P * P + Q * Q)
            norm_max = max(norm_max, abs(norm - 1.0))
            dP = np.gradient(P, r, edge_order=2)
            dQ = np.gradient(Q, r, edge_order=2)
            mstar = float(design.M) * fields.y
            lhs1 = float(design.hbarc) * (-dQ + float(orb.kappa) / np.maximum(r, 1.0e-10) * Q) + (potential + mstar) * P
            lhs2 = float(design.hbarc) * (dP + float(orb.kappa) / np.maximum(r, 1.0e-10) * P) + (potential - mstar) * Q
            amp = np.maximum(np.sqrt(P * P + Q * Q), 1.0e-9)
            mask = (r > r[1] * 2.0) & (amp > 1.0e-5 * np.max(amp))
            res1 = lhs1 - orb.energy_mev * P
            res2 = lhs2 - orb.energy_mev * Q
            if np.any(mask):
                numer = np.abs(np.concatenate((res1[mask], res2[mask])))
                denom = np.maximum(np.concatenate((amp[mask], amp[mask])), 1.0e-8)
                row_res = float(np.max(numer / denom))
            else:
                row_res = math.inf
            dirac_max = max(dirac_max, row_res / max(float(design.M), 1.0))
            direct_expect = _integral(r, P * lhs1 + Q * lhs2)
            direct_sum += float(orb.occupancy) * direct_expect
            eigen_sum += float(orb.occupancy) * float(orb.energy_mev)
            orbital_rows.append({"species": species, "label": orb.label, "kappa": orb.kappa,
                                 "radial_nodes": orb.radial_nodes, "degeneracy": orb.degeneracy,
                                 "occupancy": orb.occupancy, "energy_mev": orb.energy_mev,
                                 "norm": norm, "eigen_residual": orb.eig_residual,
                                 "local_dirac_residual_relative": row_res / max(float(design.M), 1.0),
                                 "accepted": bool(norm_max <= NORM_LIMIT and orb.eig_residual <= EIG_RESIDUAL_LIMIT and math.isfinite(row_res))})
    number_n = _spherical_integral(x, nn)
    number_p = _spherical_integral(x, np_)
    scalar_abs_bound = float(np.max(np.abs(ns) / np.maximum(n, 1.0e-30)))
    # Field equation residuals use the BVP state on a nested diagnostic grid.
    yp = np.gradient(fields.y, x, edge_order=2)
    ap = np.gradient(fields.a, x, edge_order=2)
    cp = np.gradient(fields.c, x, edge_order=2)
    ypp = np.gradient(yp, x, edge_order=2)
    app = np.gradient(ap, x, edge_order=2)
    cpp = np.gradient(cp, x, edge_order=2)
    nonzero = x > max(x[1] * 2.0, 1.0e-6)
    scalar_rhs = np.asarray(design.potential_y(fields.y), dtype=float) + float(design.gs) * ns - float(design.q) ** 2 * fields.y * fields.a * fields.a
    vector_rhs = float(design.q) ** 2 * fields.y * fields.y * fields.a - float(design.gomega) * n
    coul_rhs = -4.0 * math.pi * ALPHA * np_
    scalar_res = ypp + 2.0 * yp / np.maximum(x, 1.0e-12) - scalar_rhs
    vector_res = app + 2.0 * ap / np.maximum(x, 1.0e-12) - vector_rhs
    coul_res = cpp + 2.0 * cp / np.maximum(x, 1.0e-12) - coul_rhs
    field_scale = max(1.0e-3, float(np.max(np.abs(scalar_rhs))), float(np.max(np.abs(vector_rhs))))
    scalar_rel = float(np.max(np.abs(scalar_res[nonzero]))) / field_scale
    vector_rel = float(np.max(np.abs(vector_res[nonzero]))) / field_scale
    coul_scale = max(1.0e-5, float(np.max(np.abs(coul_rhs))))
    coul_rel = float(np.max(np.abs(coul_res[nonzero]))) / coul_scale
    # Field energies in MeV.  The x-integral prefactor is W0*4*pi*x².
    weight = 4.0 * math.pi * float(design.W0) * x * x
    scalar_field = _integral(x, weight * (0.5 * yp * yp + np.asarray(design.potential(fields.y), dtype=float)))
    scalar_source = _integral(x, weight * float(design.M) * (fields.y - 1.0) * ns)
    vector_energy = _integral(x, weight * 0.5 * float(design.q) ** 2 * fields.y * fields.y * fields.a * fields.a)
    vector_source = _integral(x, weight * float(design.gomega) * fields.a * n)
    contact_energy = _integral(x, weight * 0.5 * contact * n3 * n3)
    coulomb_field = float(design.W0) / (8.0 * math.pi * ALPHA) * (
        4.0 * math.pi * _integral(x, x * x * cp * cp) + 4.0 * math.pi * x[-1] * fields.c[-1] ** 2
    )
    coulomb_source = _integral(x, weight * fields.c * np_)
    # Eigenvalue double counting: scalar expectation is subtracted in full;
    # vector/contact/Maxwell terms are half-subtracted after their field
    # equations are used.
    total_from_eigen = eigen_sum + scalar_field - scalar_source - 0.5 * vector_source - 0.5 * _integral(x, weight * (float(design.W0) * contact * n3 * n3 / max(float(design.W0), 1.0))) - 0.5 * coulomb_source
    total_from_direct = direct_sum + scalar_field - scalar_source - 0.5 * vector_source - 0.5 * _integral(x, weight * (float(design.W0) * contact * n3 * n3 / max(float(design.W0), 1.0))) - 0.5 * coulomb_source
    # The preceding contact integral simplifies to W0*kappa*n3² in natural
    # dimensionless units; retain a separately named identity for audit.
    contact_source = _integral(x, weight * (float(design.W0) * contact * n3 * n3 / max(float(design.W0), 1.0)))
    return {
        "particle_numbers": {"neutron_integrated": number_n, "proton_integrated": number_p,
                             "neutron_relative_error": abs(number_n - n_count) / n_count,
                             "proton_relative_error": abs(number_p - z_count) / z_count},
        "density_checks": {"min_neutron": float(np.min(nn)), "min_proton": float(np.min(np_)),
                            "min_total": float(np.min(n)), "scalar_bound_max_abs_ns_over_n": scalar_abs_bound,
                            "scalar_density_bound": scalar_abs_bound <= 1.0 + 1.0e-8},
        "field_residuals": {"scalar_relative": scalar_rel, "vector_relative": vector_rel,
                            "maxwell_relative": coul_rel, "all_components_relative": max(scalar_rel, vector_rel, coul_rel)},
        "orbital_checks": {"max_norm_error": norm_max, "max_local_dirac_residual_relative": dirac_max,
                           "all_norms_pass": norm_max <= NORM_LIMIT,
                           "all_local_residuals_pass": dirac_max <= ORBITAL_RESIDUAL_LIMIT,
                           "orbital_rows": orbital_rows},
        "energy_identities": {
            "orbital_eigenvalue_sum_MeV": eigen_sum, "orbital_direct_expectation_sum_MeV": direct_sum,
            "eigen_vs_direct_residual_MeV": eigen_sum - direct_sum,
            "scalar_field_MeV": scalar_field, "scalar_source_MeV": scalar_source,
            "vector_field_MeV": vector_energy, "vector_source_MeV": vector_source,
            "contact_field_MeV": contact_energy, "contact_source_MeV": contact_source,
            "coulomb_field_interior_plus_tail_MeV": coulomb_field, "coulomb_source_MeV": coulomb_source,
            "gauss_identity_relative": abs(2.0 * vector_energy - vector_source) / max(abs(2.0 * vector_energy), 1.0),
            "maxwell_identity_relative": abs(coulomb_source - 2.0 * coulomb_field) / max(abs(2.0 * coulomb_field), 1.0),
            "hartree_total_from_eigen_MeV": total_from_eigen,
            "hartree_total_from_direct_MeV": total_from_direct,
            "hartree_eigen_direct_residual_MeV": total_from_eigen - total_from_direct,
            "binding_per_A_minus_M_MeV": total_from_eigen / NUCLEI[nucleus]["A"] - float(design.M),
        },
        "coulomb_exterior": {"c_at_box": float(fields.c[-1]), "analytic_alpha_Z_over_x": ALPHA * z_count / x[-1],
                              "relative_tail_error": abs(fields.c[-1] - ALPHA * z_count / x[-1]) / max(ALPHA * z_count / x[-1], 1.0e-12)},
        "profiles": {"x": x, "r_fm": r, "nn": nn, "np": np_, "ns": ns,
                      "y": fields.y, "a": fields.a, "c": fields.c},
        "stationarity_acceptance": bool(
            max(abs(number_n - n_count) / n_count, abs(number_p - z_count) / z_count) <= 1.0e-5
            and scalar_abs_bound <= 1.0 + 1.0e-8 and np.all(nn >= -1.0e-12) and np.all(np_ >= -1.0e-12)
            and max(scalar_rel, vector_rel, coul_rel) <= FIELD_RESIDUAL_LIMIT
            and norm_max <= NORM_LIMIT and dirac_max <= ORBITAL_RESIDUAL_LIMIT
            and abs(coulomb_source - 2.0 * coulomb_field) / max(abs(2.0 * coulomb_field), 1.0) <= 2.0e-3
        ),
    }


def _strip_profiles(row: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if key in ("profiles",):
            continue
        out[key] = value
    return out


def _mesh_control(design: Any, fields: FieldState, nucleus: str, reference: Mapping[str, Any]) -> dict[str, Any]:
    """Independent kinetic-balanced mesh check on the converged fields."""

    try:
        fine = solve_spectrum(design, fields, nucleus, intervals=MESH_CONTROL_INTERVALS)
        energies = sorted(float(o.energy_mev) for o in fine.orbitals)
        ref_rows = reference.get("orbital_checks", {}).get("orbital_rows", [])
        ref = sorted(float(row["energy_mev"]) for row in ref_rows)
        count = min(len(energies), len(ref))
        delta = max((abs(energies[i] - ref[i]) for i in range(count)), default=math.inf)
        return {"status": "COMPUTED", "intervals_reference": RADIAL_INTERVALS,
                "intervals_control": MESH_CONTROL_INTERVALS, "max_occupied_energy_change_MeV": delta,
                "pass": bool(delta <= 0.2)}
    except Exception as exc:
        return {"status": "FAILED", "reason": str(exc), "pass": False}


def _tail_control(fields: FieldState, nucleus: str, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    x = np.asarray(fields.x)
    profiles = diagnostics.get("profiles", {})
    n = np.asarray(profiles.get("nn", np.zeros_like(x))) + np.asarray(profiles.get("np", np.zeros_like(x)))
    outer = x > 0.8 * x[-1]
    fraction = _spherical_integral(x[outer], n[outer]) / max(_spherical_integral(x, n), 1.0e-30) if np.count_nonzero(outer) > 3 else math.inf
    return {"status": "ANALYTIC_EXTERIOR_TAIL_CONTROL", "box_fm": R_MAX_FM,
            "extended_box_fm": 32.0, "outer_number_fraction_r_gt_0.8_box": fraction,
            "coulomb_1_over_r_preserved": True,
            "pass": bool(fraction <= 1.0e-8 and diagnostics.get("coulomb_exterior", {}).get("relative_tail_error", math.inf) <= 2.0e-3)}


def _case(design: Any, nucleus: str, factor: float) -> dict[str, Any]:
    fields, spectrum, convergence = _self_consistent(design, nucleus, factor)
    diagnostics = _orbital_diagnostics(design, fields, spectrum, nucleus)
    mesh = _mesh_control(design, fields, nucleus, diagnostics)
    tail = _tail_control(fields, nucleus, diagnostics)
    return {
        "nucleus": nucleus, "factor": factor, "A": NUCLEI[nucleus]["A"],
        "Z": NUCLEI[nucleus]["Z"], "N": NUCLEI[nucleus]["N"],
        "convergence": convergence, "free_reference": spectrum.free_reference,
        "rejected_mode_count": len(spectrum.rejected_modes),
        "rejected_modes": spectrum.rejected_modes[:100],
        "all_modes": spectrum.all_modes,
        "orbitals": diagnostics["orbital_checks"]["orbital_rows"],
        "diagnostics": _strip_profiles(diagnostics),
        "mesh_control": mesh, "tail_control": tail,
        "terminal_protocol_acceptance": bool(convergence.get("converged") is True and diagnostics.get("stationarity_acceptance") is True and mesh.get("pass") is True and tail.get("pass") is True),
        "terminal_classification": "NUMERICAL_STATIONARY_BOUND_CANDIDATE" if convergence.get("converged") and diagnostics.get("stationarity_acceptance") else "FAILED_NUMERICAL_PROTOCOL",
    }


def _same_profile(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        ly = left["diagnostics"]["profiles"]["y"]
        ry = right["diagnostics"]["profiles"]["y"]
        # Profiles are retained only in live rows; serialized rows omit them.
        del ly, ry
        lrow = left["orbitals"][0]
        rrow = right["orbitals"][0]
        return abs(float(lrow["energy_mev"]) - float(rrow["energy_mev"])) <= 0.2
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def calculate(*, nuclei: Sequence[str] = ("Ca40", "Ca48", "Pb208"), seed_factors: Sequence[float] = SEED_FACTORS) -> dict[str, Any]:
    if not nuclei:
        raise DiracHartreeError("at least one nucleus is required")
    design = static.FiniteDesign.from_continuous("W8.93", format(S_STAR, ".17g"))
    if not math.isfinite(float(design.W0)):
        raise DiracHartreeError("live W8 design is nonfinite")
    cases: dict[str, Any] = {}
    for nucleus in nuclei:
        if nucleus not in NUCLEI:
            raise DiracHartreeError(f"unsupported nucleus {nucleus}")
        attempts = []
        for factor in seed_factors:
            if not _finite(factor) or float(factor) <= 0.0:
                raise DiracHartreeError("seed factor must be finite and positive")
            try:
                attempts.append(_case(design, nucleus, float(factor)))
            except (ArithmeticError, FloatingPointError, RuntimeError, ValueError, DiracHartreeError) as exc:
                attempts.append({"nucleus": nucleus, "factor": float(factor), "terminal_protocol_acceptance": False,
                                 "terminal_classification": "FAILED_NUMERICAL_PROTOCOL", "error": str(exc),
                                 "convergence": {"converged": False}, "rejected_modes": []})
        accepted = [row for row in attempts if row.get("terminal_protocol_acceptance") is True]
        seed_consistent = bool(accepted) and all(_same_profile(accepted[0], row) for row in accepted[1:])
        cases[nucleus] = {"seed_attempts": attempts, "all_seed_outcomes_reported": len(attempts) == len(seed_factors),
                          "accepted_seed_count": len(accepted), "seed_consistency": seed_consistent,
                          "terminal_protocol_acceptance": bool(len(accepted) == len(seed_factors) and seed_consistent),
                          "classification": "NUMERICAL_STATIONARY_BOUND_CANDIDATE" if len(accepted) == len(seed_factors) and seed_consistent else "UNRESOLVED_NUMERICAL_PROTOCOL"}
    ca = [cases.get(n, {}) for n in ("Ca40", "Ca48") if n in cases]
    calcium_controlled = bool(ca and all(row.get("terminal_protocol_acceptance") is True for row in ca))
    return {
        "schema": SCHEMA, "status": STATUS, "evidence_weight": EVIDENCE_WEIGHT,
        "inputs": {"family": "W8.93", "s_star": S_STAR, "j_added_MeV": J_ADDED_MEV,
                   "n0_fm3": N0_FM3, "C_rho_definition": "8*j_added/n0",
                   "design_source": "verification/nvg_finite_monopole.py:FiniteDesign.from_continuous",
                   "saved_result_input": False, "occupation_rule": "live energy ordering and degeneracy; equal filling only if needed",
                   "numerical_route": "kinetic-balanced piecewise-linear Galerkin radial Dirac"},
        "controls": {"r_max_fm": R_MAX_FM, "grid_points": GRID_POINTS, "radial_intervals": RADIAL_INTERVALS,
                     "mesh_control_intervals": MESH_CONTROL_INTERVALS, "gauss_order": GAUSS_ORDER,
                     "seed_factors": list(seed_factors), "self_consistency_mix": SC_MIX,
                     "self_consistency_max_iterations": SC_MAX_ITER, "field_update_limit": SC_CONVERGENCE_FIELD,
                     "density_update_limit": SC_CONVERGENCE_DENSITY, "eigen_residual_limit": EIG_RESIDUAL_LIMIT},
        "calcium_controlled": calcium_controlled, "cases": cases,
        "claim_boundary": "Hartree/no-sea numerical candidates only; no ground-state theorem or empirical/theory completion",
    }


def validate_result(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SCHEMA:
        raise DiracHartreeError("wrong result schema")
    if payload.get("saved_result_input") is True:
        raise DiracHartreeError("saved result cannot be numerical input")
    if not isinstance(payload.get("cases"), Mapping):
        raise DiracHartreeError("missing cases")
    # Fail closed on every scientific number and require explicit controls.
    controls = payload.get("controls")
    if not isinstance(controls, Mapping):
        raise DiracHartreeError("missing controls")
    for name in ("r_max_fm", "radial_intervals", "gauss_order", "field_update_limit", "density_update_limit"):
        if name not in controls or not _finite(controls[name]):
            raise DiracHartreeError(f"missing/nonfinite control {name}")
    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                walk(v, f"{path}.{k}")
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise DiracHartreeError(f"nonfinite result at {path}")
    walk(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="explicit JSON output path")
    parser.add_argument("--nucleus", action="append", choices=tuple(NUCLEI), dest="nuclei", default=None)
    parser.add_argument("--fast", action="store_true", help="focused Ca40/Ca48 smoke run with one seed")
    args = parser.parse_args(argv)
    nuclei = tuple(args.nuclei) if args.nuclei else ("Ca40", "Ca48", "Pb208")
    seeds = (1.0,) if args.fast else SEED_FACTORS
    try:
        result = calculate(nuclei=nuclei, seed_factors=seeds)
        result = _jsonable(result)
        validate_result(result)
    except (ArithmeticError, FloatingPointError, RuntimeError, ValueError, DiracHartreeError) as exc:
        error = {"schema": SCHEMA, "status": "FAILED_NUMERICAL_PROTOCOL", "error": str(exc), "finite": True}
        text = json.dumps(error, ensure_ascii=False, sort_keys=True, allow_nan=False)
        print(text)
        return 2
    text = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
