#!/usr/bin/env python3
"""Executable audit of the bound source-complete W/Gauss contract.

This module intentionally owns a new ``source_complete`` surface.  It does not
import the maintained W producer: all branches, thermodynamic quantities and
passport controls are recomputed from the adjudicated action in
``verification/contracts/source_complete_action.md``.
"""

from __future__ import annotations

import json
import hashlib
import inspect
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import mpmath as mp


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
RESULT_PATH = ROOT / "source_complete_solution_audit_results.json"
REPORT_PATH = ROOT / "source_complete_solution_audit_report.md"
EULER_WARD_BACKEND_SHA256 = "52af615468f91e7ebd53dce63069890e93252423fc8aaaf7cd37fe8564d1e797"


@dataclass(frozen=True)
class Params:
    W0: float = 859.0
    lam: float = 1.05
    M_N: float = 939.0
    n0_fm3: float = 0.16
    hbar_c: float = 197.3269804
    g_omega: float = 10.12
    m_omega: float = 782.6
    degeneracy: int = 4

    @property
    def g_s(self) -> float:
        return self.M_N / self.W0

    @property
    def q_phi(self) -> float:
        return self.m_omega / self.W0

    @property
    def scalar_mass(self) -> float:
        return math.sqrt(2.0 * self.lam) * self.W0

    @property
    def mu0(self) -> float:
        return math.sqrt(self.lam) * self.W0

    @property
    def n0_nat(self) -> float:
        return self.n0_fm3 * self.hbar_c**3


PARAMS = Params()


def fermi_momentum(n_nat: float, p: Params = PARAMS) -> float:
    if n_nat <= 0.0:
        return 0.0
    return (6.0 * math.pi**2 * n_nat / p.degeneracy) ** (1.0 / 3.0)


def fermi_energy(n_nat: float, mass: float, p: Params = PARAMS) -> float:
    return math.sqrt(fermi_momentum(n_nat, p) ** 2 + mass**2)


def fermi_free_energy(n_nat: float, mass: float, p: Params = PARAMS) -> float:
    """Zero-temperature free Fermi-gas energy density in MeV^4."""
    if n_nat <= 0.0:
        return 0.0
    k = fermi_momentum(n_nat, p)
    E = math.sqrt(k * k + mass * mass)
    if mass == 0.0:
        # The m -> 0 limit of the finite-mass integral is d k_F^4/(8 pi^2).
        return p.degeneracy * k**4 / (8.0 * math.pi**2)
    log_term = math.asinh(k / mass)
    return p.degeneracy / (16.0 * math.pi**2) * (
        k * E * (2.0 * k * k + mass * mass) - mass**4 * log_term
    )


def fermi_pressure(n_nat: float, mass: float, p: Params = PARAMS) -> float:
    """Zero-temperature kinetic pressure for the same Fermi functional."""
    if n_nat <= 0.0:
        return 0.0
    k = fermi_momentum(n_nat, p)
    E = math.sqrt(k * k + mass * mass)
    if mass == 0.0:
        return p.degeneracy * k**4 / (24.0 * math.pi**2)
    return p.degeneracy / (48.0 * math.pi**2) * (
        k * E * (2.0 * k * k - 3.0 * mass * mass)
        + 3.0 * mass**4 * math.asinh(k / mass)
    )


def scalar_density(n_nat: float, mass: float, p: Params = PARAMS) -> float:
    """Scalar density ∂F/∂m for the same Fermi functional (MeV^3)."""
    if n_nat <= 0.0 or mass == 0.0:
        return 0.0
    k = fermi_momentum(n_nat, p)
    E = math.sqrt(k * k + mass * mass)
    return p.degeneracy * mass / (4.0 * math.pi**2) * (
        k * E - mass * mass * math.asinh(k / mass)
    )


def scalar_potential(W: float, p: Params = PARAMS) -> float:
    return p.lam * (W * W - p.W0 * p.W0) ** 2 / 4.0


def matter_mass(W: float, p: Params = PARAMS) -> float:
    return p.g_s * W


def gauss_A0(n_nat: float, W: float, p: Params = PARAMS) -> float:
    if n_nat == 0.0:
        return 0.0
    return p.g_omega * n_nat / (p.q_phi * p.q_phi * W * W)


def W_stationarity(W: float, n_nat: float, p: Params = PARAMS) -> float:
    """∂(eliminated generator)/∂W, in MeV^3."""
    m = matter_mass(W, p)
    ns = scalar_density(n_nat, m, p)
    vector = 0.0 if n_nat == 0.0 else p.g_omega**2 * n_nat**2 / (
        p.q_phi**2 * W**3
    )
    return p.lam * W * (W * W - p.W0 * p.W0) + p.g_s * ns - vector


def solve_W(
    n_nat: float,
    p: Params = PARAMS,
    tol: float = 1.0e-11,
    max_iter: int = 300,
) -> float:
    """Bracket the positive coupled W root and solve by bisection."""
    if n_nat == 0.0:
        return p.W0
    lo = max(p.W0 * 1.0e-8, 1.0e-8)
    flo = W_stationarity(lo, n_nat, p)
    hi = max(2.0 * p.W0, p.W0 + 1.0)
    fhi = W_stationarity(hi, n_nat, p)
    while fhi <= 0.0:
        hi *= 2.0
        fhi = W_stationarity(hi, n_nat, p)
        if hi > 1.0e9:
            raise RuntimeError("failed to bracket positive W branch")
    if flo * fhi > 0.0:
        raise RuntimeError("unexpected W bracket sign")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = W_stationarity(mid, n_nat, p)
        # ``tol`` is an absolute W-width/residual control.  A relative-width
        # stopping rule would leave MeV-scale drift when the requested
        # convergence is 1e-8 MeV.
        if abs(fm) <= tol or (hi - lo) <= tol:
            return mid
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def energy_density(n_nat: float, W: float, p: Params = PARAMS) -> float:
    m = matter_mass(W, p)
    vector = 0.0 if n_nat == 0.0 else p.g_omega**2 * n_nat**2 / (
        2.0 * p.q_phi**2 * W * W
    )
    return fermi_free_energy(n_nat, m, p) + scalar_potential(W, p) + vector


def hilbert_stress(n_nat: float, W: float, p: Params = PARAMS) -> tuple[float, float]:
    """Homogeneous (T00, P) reduction of metric variation of the same action."""
    m = matter_mass(W, p)
    vector = 0.0 if n_nat == 0.0 else p.g_omega**2 * n_nat**2 / (
        2.0 * p.q_phi**2 * W * W
    )
    return energy_density(n_nat, W, p), fermi_pressure(n_nat, m, p) - scalar_potential(W, p) + vector


def chemical_potential(n_nat: float, W: float, p: Params = PARAMS) -> float:
    m = matter_mass(W, p)
    vector = 0.0 if n_nat == 0.0 else p.g_omega**2 * n_nat / (
        p.q_phi**2 * W * W
    )
    return fermi_energy(n_nat, m, p) + vector


def generator_G(n_nat: float, W: float, mu_B: float, p: Params = PARAMS) -> float:
    """Stationary generator before eliminating A0, evaluated at Gauss A0."""
    A0 = gauss_A0(n_nat, W, p)
    m = matter_mass(W, p)
    omega = -p.q_phi * A0  # unitary gauge: g_0=-q A_0
    return (
        fermi_free_energy(n_nat, m, p)
        + scalar_potential(W, p)
        - 0.5 * W * W * omega * omega
        + p.g_omega * A0 * n_nat
        - mu_B * n_nat
    )


ETA = np.diag([1.0, -1.0, -1.0, -1.0])
_I2 = np.eye(2, dtype=complex)
_Z2 = np.zeros((2, 2), dtype=complex)
_SIGMA = (
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
)
GAMMA = [np.block([[_I2, _Z2], [_Z2, -_I2]])]
GAMMA.extend(np.block([[_Z2, sigma], [-sigma, _Z2]]) for sigma in _SIGMA)


def manufactured_fields() -> tuple[complex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Deterministic nonzero off-shell fields at one spacetime point."""
    phi = 0.71 + 0.23j
    dphi = np.array([0.13 - 0.07j, -0.21 + 0.11j, 0.17 + 0.05j, -0.09 - 0.14j])
    N = np.array([0.42 + 0.17j, -0.31 + 0.28j, 0.19 - 0.37j, 0.27 + 0.08j])
    dN = np.array(
        [
            [0.11 + 0.02j, -0.07 + 0.05j, 0.13 - 0.09j, 0.03 + 0.12j],
            [-0.04 + 0.08j, 0.15 - 0.03j, 0.06 + 0.02j, -0.12 + 0.04j],
            [0.09 - 0.06j, 0.01 + 0.13j, -0.16 + 0.07j, 0.05 - 0.02j],
            [-0.08 + 0.11j, 0.03 - 0.04j, 0.14 + 0.06j, 0.02 + 0.09j],
        ],
        dtype=complex,
    )
    A = np.array([0.19, -0.23, 0.07, 0.16], dtype=float)
    alpha = 0.37
    d_alpha = np.array([0.23, -0.17, 0.31, -0.11], dtype=float)
    return phi, dphi, N, dN, A, alpha, d_alpha


def _transform_manufactured_fields(
    phi: complex,
    dphi: np.ndarray,
    N: np.ndarray,
    dN: np.ndarray,
    A: np.ndarray,
    alpha: float,
    d_alpha: np.ndarray,
    p: Params = PARAMS,
    scale: float = 1.0,
) -> tuple[complex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the adjudicated local-U(1) transformation, including derivatives."""
    phase_phi = np.exp(1.0j * p.q_phi * scale * alpha)
    phase_N = np.exp(-1.0j * p.g_omega * scale * alpha)
    phi_t = phase_phi * phi
    dphi_t = phase_phi * (dphi + 1.0j * p.q_phi * scale * d_alpha * phi)
    N_t = phase_N * N
    dN_t = phase_N * (dN - 1.0j * p.g_omega * scale * d_alpha[:, None] * N[None, :])
    A_t = A + scale * d_alpha
    return phi_t, dphi_t, N_t, dN_t, A_t


def manufactured_action_density(
    phi: complex,
    dphi: np.ndarray,
    N: np.ndarray,
    dN: np.ndarray,
    A: np.ndarray,
    p: Params = PARAMS,
) -> float:
    """Flat-space scalar+Dirac action density with one covariant N coupling.

    ``D_N=nabla+i*g_omega*A`` is the sole nucleon gauge coupling here.  There
    is deliberately no second explicit baryon-current term in this local
    action density; the stationary generator's Routhian contains its required
    ``+g_omega A_0 n_B`` term separately.
    """
    W = math.sqrt(2.0) * abs(phi)
    D_phi = dphi - 1.0j * p.q_phi * A * phi
    scalar = float(np.real(np.sum(np.diag(ETA) * np.conjugate(D_phi) * D_phi)))
    scalar -= scalar_potential(W, p)
    D_N = dN + 1.0j * p.g_omega * A[:, None] * N[None, :]
    bar_N = np.conjugate(N) @ GAMMA[0]
    slash_D = sum(GAMMA[mu] @ D_N[mu] for mu in range(4))
    dirac = bar_N @ (1.0j * slash_D - matter_mass(W, p) * N)
    return scalar + float(np.real(dirac))


def _manufactured_A_dependent_density(
    phi: complex,
    dphi: np.ndarray,
    N: np.ndarray,
    dN: np.ndarray,
    A: np.ndarray,
    p: Params = PARAMS,
) -> float:
    """A-dependent part of ``manufactured_action_density``.

    The omitted scalar potential and Dirac mass are independent of ``A`` and
    therefore cancel identically in an A-variation.  Keeping the variation in
    this numerically centered form avoids subtracting two O(1e11) action
    densities when checking the source current.
    """
    W = math.sqrt(2.0) * abs(phi)
    D_phi = dphi - 1.0j * p.q_phi * A * phi
    scalar = float(np.real(np.sum(np.diag(ETA) * np.conjugate(D_phi) * D_phi)))
    D_N = dN + 1.0j * p.g_omega * A[:, None] * N[None, :]
    bar_N = np.conjugate(N) @ GAMMA[0]
    slash_D = sum(GAMMA[mu] @ D_N[mu] for mu in range(4))
    # The mass term is A independent and is intentionally omitted above.
    dirac = bar_N @ (1.0j * slash_D)
    return scalar + float(np.real(dirac))


def manufactured_current(
    phi: complex,
    dphi: np.ndarray,
    N: np.ndarray,
    dN: np.ndarray,
    A: np.ndarray,
    p: Params = PARAMS,
) -> np.ndarray:
    """Contravariant source current defined as ``-delta L/delta A_mu``."""
    D_phi = dphi - 1.0j * p.q_phi * A * phi
    j_phi = np.array(
        [
            np.real(-1.0j * p.q_phi * ETA[mu, mu] * (np.conjugate(phi) * D_phi[mu] - np.conjugate(D_phi[mu]) * phi))
            for mu in range(4)
        ]
    )
    D_N = dN + 1.0j * p.g_omega * A[:, None] * N[None, :]
    bar_N = np.conjugate(N) @ GAMMA[0]
    j_b = np.array([np.real(bar_N @ GAMMA[mu] @ N) for mu in range(4)])
    return j_phi + p.g_omega * j_b


def manufactured_gauge_controls(p: Params = PARAMS) -> dict[str, Any]:
    phi, dphi, N, dN, A, alpha, d_alpha = manufactured_fields()
    transformed = _transform_manufactured_fields(phi, dphi, N, dN, A, alpha, d_alpha, p)
    phi_t, dphi_t, N_t, dN_t, A_t = transformed
    D_phi = dphi - 1.0j * p.q_phi * A * phi
    D_N = dN + 1.0j * p.g_omega * A[:, None] * N[None, :]
    D_phi_t = dphi_t - 1.0j * p.q_phi * A_t * phi_t
    D_N_t = dN_t + 1.0j * p.g_omega * A_t[:, None] * N_t[None, :]
    expected_phi = np.exp(1.0j * p.q_phi * alpha) * D_phi
    expected_N = np.exp(-1.0j * p.g_omega * alpha) * D_N
    L = manufactured_action_density(phi, dphi, N, dN, A, p)
    L_t = manufactured_action_density(phi_t, dphi_t, N_t, dN_t, A_t, p)
    scale_L = max(1.0, abs(L), abs(L_t))
    covariance_phi = float(np.max(np.abs(D_phi_t - expected_phi)))
    covariance_N = float(np.max(np.abs(D_N_t - expected_N)))
    action_rel = abs(L_t - L) / scale_L

    current = manufactured_current(phi, dphi, N, dN, A, p)
    derivative = []
    h = 1.0e-5
    for mu in range(4):
        delta = np.zeros(4)
        delta[mu] = h
        lp = _manufactured_A_dependent_density(phi, dphi, N, dN, A + delta, p)
        lm = _manufactured_A_dependent_density(phi, dphi, N, dN, A - delta, p)
        derivative.append((lp - lm) / (2.0 * h))
    derivative = np.array(derivative)
    current_rel = float(np.max(np.abs(derivative + current)) / max(1.0, float(np.max(np.abs(current)))))

    def directional(alpha_value: float, d_alpha_value: np.ndarray) -> float:
        eps = 1.0e-6
        plus = _transform_manufactured_fields(phi, dphi, N, dN, A, alpha_value, d_alpha_value, p, eps)
        minus = _transform_manufactured_fields(phi, dphi, N, dN, A, alpha_value, d_alpha_value, p, -eps)
        lp = manufactured_action_density(*plus, p)
        lm = manufactured_action_density(*minus, p)
        return (lp - lm) / (2.0 * eps)

    ward_global = directional(alpha, np.zeros(4))
    ward_connection = directional(0.0, d_alpha)
    ward_full = directional(alpha, d_alpha)
    ward_scale = max(1.0, abs(L), float(np.max(np.abs(current))))
    source_scan = inspect.getsource(manufactured_action_density)
    single_coupling = source_scan.count("D_N = dN + 1.0j * p.g_omega") == 1 and "J_B" not in source_scan
    def stable(value: float) -> float:
        # BLAS reduction order can vary in the last ulp for exact-zero
        # covariance residuals; report those sub-1e-14 values canonically.
        return 0.0 if abs(value) < 1.0e-14 else float(f"{value:.15e}")
    return {
        "covariance_phi_max": stable(covariance_phi),
        "covariance_N_max": stable(covariance_N),
        "action_invariance_relative": stable(action_rel),
        "A_variation_current_relative": stable(current_rel),
        "ward_global_relative": stable(abs(ward_global) / ward_scale),
        "ward_connection_relative": stable(abs(ward_connection) / ward_scale),
        "ward_full_relative": stable(abs(ward_full) / ward_scale),
        "single_covariant_nucleon_coupling": single_coupling,
        "no_duplicate_explicit_current": "- g_omega A_mu J_B" not in source_scan and "J_B" not in source_scan,
        "all_pass": covariance_phi < 1.0e-12 and covariance_N < 1.0e-12 and action_rel < 1.0e-12 and current_rel < 1.0e-8 and max(abs(ward_global), abs(ward_connection), abs(ward_full)) / ward_scale < 1.0e-10 and single_coupling,
    }


def _manufactured_offshell_density(state: tuple[Any, ...], p: Params = PARAMS) -> complex:
    """Independent off-shell density with Phi/bar-Phi and N/bar-N varied separately."""
    phi, phibar, N, bar_N, dphi, dphibar, dN, _dbar_N, A = state
    rho = phibar * phi
    W = np.sqrt(2.0 * rho)
    D_phi = dphi - 1.0j * p.q_phi * A * phi
    D_phibar = dphibar + 1.0j * p.q_phi * A * phibar
    scalar = np.sum(np.diag(ETA) * D_phibar * D_phi)
    scalar -= p.lam * (rho - p.W0 * p.W0 / 2.0) ** 2
    D_N = dN + 1.0j * p.g_omega * A[:, None] * N[None, :]
    slash_D = sum(GAMMA[mu] @ D_N[mu] for mu in range(4))
    dirac = bar_N @ (1.0j * slash_D - p.g_s * W * N)
    return complex(scalar + dirac)


def _five_point_complex(function, h: float) -> complex:
    """Fourth-order centered finite difference for a complex-valued function."""
    return (
        function(-2.0 * h)
        - 8.0 * function(-h)
        + 8.0 * function(h)
        - function(2.0 * h)
    ) / (12.0 * h)


def _complex_euler_partial(state: tuple[Any, ...], field_index: int, component: int, p: Params, h: float = 1.0e-4) -> complex:
    """Finite-difference Wirtinger derivative, treating each barred field independently."""
    def value(delta: complex) -> complex:
        values = list(state)
        field = np.array(values[field_index], copy=True)
        field.flat[component] += delta
        values[field_index] = field
        return _manufactured_offshell_density(tuple(values), p)

    d_real = _five_point_complex(lambda t: value(t), h)
    d_imag = _five_point_complex(lambda t: value(1.0j * t), h)
    return 0.5 * (d_real - 1.0j * d_imag)


def _real_connection_partial(state: tuple[Any, ...], component: int, p: Params, h: float = 1.0e-4) -> complex:
    def value(delta: float) -> complex:
        values = list(state)
        A = np.array(values[8], copy=True)
        A[component] += delta
        values[8] = A
        return _manufactured_offshell_density(tuple(values), p)

    return _five_point_complex(value, h)


def manufactured_euler_ward_controls(p: Params = PARAMS, connection_sign: float = 1.0) -> dict[str, Any]:
    """Pinned analytic/nested-mp.diff control; no binary64 cancellation stencil.

    The dependency digest keeps the existing single-file upstream pin
    transitively bound to the actual control implementation. An implementation
    change requires explicit regeneration, not a silent backend substitution.
    """
    import nvg_euler_ward_precision as precision

    path = Path(precision.__file__).resolve()
    expected_path = (ROOT / "nvg_euler_ward_precision.py").resolve()
    if path != expected_path or hashlib.sha256(path.read_bytes()).hexdigest() != EULER_WARD_BACKEND_SHA256:
        raise RuntimeError("Euler/Ward precision backend identity changed; reviewed regeneration required")
    controls = precision.manufactured_euler_ward_controls(p, connection_sign)
    return {**controls, "backend_source_sha256": EULER_WARD_BACKEND_SHA256}


def _legacy_manufactured_euler_ward_controls(p: Params = PARAMS, connection_sign: float = 1.0) -> dict[str, Any]:
    """Explicit finite-dimensional Euler/Noether identity on a manufactured jet.

    Historical binary64 diagnostic only; never used by the canonical producer.
    Field and first-derivative jets are sampled at neighboring points to make
    every divergence term explicit.  Euler partials and canonical momenta are
    independently finite-differenced; the transformed-action controls above
    remain a separate invariance smoke test.
    """
    rng = np.random.default_rng(1701)
    phi = p.W0 / math.sqrt(2.0) * (1.0 + 0.013j)
    dphi = np.array([0.17 - 0.09j, -0.11 + 0.07j, 0.08 + 0.04j, -0.06 - 0.05j])
    phibar = np.conjugate(phi)
    dphibar = np.conjugate(dphi)
    N = np.array([0.42 + 0.17j, -0.31 + 0.28j, 0.19 - 0.37j, 0.27 + 0.08j])
    dN = np.array(
        [
            [0.11 + 0.02j, -0.07 + 0.05j, 0.13 - 0.09j, 0.03 + 0.12j],
            [-0.04 + 0.08j, 0.15 - 0.03j, 0.06 + 0.02j, -0.12 + 0.04j],
            [0.09 - 0.06j, 0.01 + 0.13j, -0.16 + 0.07j, 0.05 - 0.02j],
            [-0.08 + 0.11j, 0.03 - 0.04j, 0.14 + 0.06j, 0.02 + 0.09j],
        ],
        dtype=complex,
    )
    bar_N = np.conjugate(N) @ GAMMA[0]
    dbar_N = np.array([np.conjugate(dN[mu]) @ GAMMA[0] for mu in range(4)])
    A = np.array([0.19, -0.23, 0.07, 0.16], dtype=float)
    second_phi = (rng.normal(size=(4, 4)) + 0.3j * rng.normal(size=(4, 4))) * 0.01
    second_phibar = np.conjugate(second_phi)
    second_N = (rng.normal(size=(4, 4, 4)) + 0.3j * rng.normal(size=(4, 4, 4))) * 0.01
    second_bar_N = np.empty_like(second_N)
    for mu in range(4):
        for nu in range(4):
            second_bar_N[mu, nu] = np.conjugate(second_N[mu, nu]) @ GAMMA[0]
    first_A = rng.normal(size=(4, 4)) * 0.01
    state = (phi, phibar, N, bar_N, dphi, dphibar, dN, dbar_N, A)

    def shifted(mu: int, distance: float) -> tuple[Any, ...]:
        values = list(state)
        values[0] = phi + dphi[mu] * distance + 0.5 * second_phi[mu, mu] * distance**2
        values[1] = phibar + dphibar[mu] * distance + 0.5 * second_phibar[mu, mu] * distance**2
        values[2] = N + dN[mu] * distance + 0.5 * second_N[mu, mu] * distance**2
        values[3] = bar_N + dbar_N[mu] * distance + 0.5 * second_bar_N[mu, mu] * distance**2
        values[4] = dphi + second_phi[:, mu] * distance
        values[5] = dphibar + second_phibar[:, mu] * distance
        values[6] = dN + second_N[:, :, mu] * distance
        values[7] = dbar_N + second_bar_N[:, :, mu] * distance
        values[8] = A + first_A[:, mu] * distance
        return tuple(values)

    h = 1.0e-3
    partial_phi = _complex_euler_partial(state, 0, 0, p, h)
    partial_phibar = _complex_euler_partial(state, 1, 0, p, h)
    partial_N = np.array([_complex_euler_partial(state, 2, j, p, h) for j in range(4)])
    partial_bar_N = np.array([_complex_euler_partial(state, 3, j, p, h) for j in range(4)])

    def momentum(field_index: int, st: tuple[Any, ...], component: int) -> complex:
        return _complex_euler_partial(st, field_index, component, p, h)

    div_P_phi = 0.0j
    div_P_phibar = 0.0j
    div_P_N = np.zeros(4, dtype=complex)
    div_P_bar_N = np.zeros(4, dtype=complex)
    div_E_A = 0.0j
    for mu in range(4):
        div_P_phi += _five_point_complex(lambda s: momentum(4, shifted(mu, s), mu), h)
        div_P_phibar += _five_point_complex(lambda s: momentum(5, shifted(mu, s), mu), h)
        for j in range(4):
            div_P_N[j] += _five_point_complex(lambda s, j=j: momentum(6, shifted(mu, s), 4 * mu + j), h)
            div_P_bar_N[j] += _five_point_complex(lambda s, j=j: momentum(7, shifted(mu, s), 4 * mu + j), h)
        div_E_A += _five_point_complex(lambda s: _real_connection_partial(shifted(mu, s), mu, p, h), h)

    E_phi = partial_phi - div_P_phi
    E_phibar = partial_phibar - div_P_phibar
    E_N = partial_N - div_P_N
    E_bar_N = partial_bar_N - div_P_bar_N
    identity = (
        -connection_sign * div_E_A
        + 1.0j * p.q_phi * (E_phi * phi - E_phibar * phibar)
        - 1.0j * p.g_omega * np.dot(E_N, N)
        + 1.0j * p.g_omega * np.dot(bar_N, E_bar_N)
    )
    scale = max(1.0, abs(div_E_A), abs(p.q_phi * E_phi * phi), abs(p.g_omega * np.dot(E_N, N)))
    residual = abs(identity) / scale
    # Last-ulp BLAS/complex-reduction differences are immaterial to the gate
    # but would otherwise make generated JSON nondeterministic.  Report a
    # conservative 1e-11 quantum while retaining the unrounded value for the
    # boolean acceptance test.
    residual_report = math.ceil(residual / 1.0e-11) * 1.0e-11
    return {
        "identity_relative": float(residual_report),
        "connection_divergence_abs": float(f"{abs(div_E_A):.12e}"),
        "euler_phi_abs": float(f"{abs(E_phi):.6f}"),
        "euler_phibar_abs": float(f"{abs(E_phibar):.6f}"),
        "euler_N_abs": float(f"{np.linalg.norm(E_N):.6f}"),
        "euler_bar_N_abs": float(f"{np.linalg.norm(E_bar_N):.6f}"),
        "finite_difference_step": h,
        "all_pass": bool(residual < 1.0e-10),
    }


def finite_difference_mu(n_nat: float, p: Params = PARAMS) -> float:
    if n_nat <= 0.0:
        return p.M_N
    h = max(1.0e-5 * n_nat, 1.0e-9 * p.n0_nat)
    lo = max(1.0e-12 * p.n0_nat, n_nat - h)
    hi = n_nat + h
    W_lo = solve_W(lo, p, tol=1.0e-12)
    W_hi = solve_W(hi, p, tol=1.0e-12)
    return (energy_density(hi, W_hi, p) - energy_density(lo, W_lo, p)) / (hi - lo)


def branch_row(x: float, p: Params = PARAMS) -> dict[str, float]:
    n_nat = x * p.n0_nat
    W = solve_W(n_nat, p)
    A0 = gauss_A0(n_nat, W, p)
    mu_B = chemical_potential(n_nat, W, p)
    eps = energy_density(n_nat, W, p)
    stress_eps, stress_pressure = hilbert_stress(n_nat, W, p)
    G = generator_G(n_nat, W, mu_B, p)
    J_total = -p.q_phi**2 * W * W * A0 + p.g_omega * n_nat
    gauss_res = p.q_phi**2 * W * W * A0 - p.g_omega * n_nat
    w_res = W_stationarity(W, n_nat, p)
    hvh_mu = finite_difference_mu(n_nat, p) if x > 0.0 else p.M_N
    scale = max(1.0, abs(p.g_omega * n_nat), abs(p.q_phi**2 * W * W * A0))
    return {
        "n_over_n0": x,
        "n_fm3": x * p.n0_fm3,
        "W_MeV": W,
        "W_over_W0": W / p.W0,
        "Mstar_MeV": matter_mass(W, p),
        "A0_MeV": A0,
        "g0_MeV": -p.q_phi * A0,
        "scalar_density_fm3": scalar_density(n_nat, matter_mass(W, p), p) / p.hbar_c**3,
        "epsilon_MeV_fm3": eps / p.hbar_c**3,
        "mu_B_MeV": mu_B,
        "mu_B_fd_MeV": hvh_mu,
        "pressure_MeV_fm3": -G / p.hbar_c**3,
        "pressure_legendre_MeV_fm3": (mu_B * n_nat - eps) / p.hbar_c**3,
        "stress_energy_MeV_fm3": stress_eps / p.hbar_c**3,
        "stress_pressure_MeV_fm3": stress_pressure / p.hbar_c**3,
        "generator_residual_MeV4": (G - (eps - mu_B * n_nat)),
        "w_stationarity_residual_MeV3": w_res,
        "w_stationarity_relative": abs(w_res) / max(1.0, abs(p.lam * W**3)),
        "gauss_residual_MeV3": gauss_res,
        "total_current_residual_MeV3": J_total,
        "total_current_relative": abs(J_total) / scale,
        "hvh_mu_relative": abs(mu_B - hvh_mu) / max(1.0, abs(mu_B)),
        "hvh_pressure_relative": abs((-G) - (mu_B * n_nat - eps)) / max(1.0, abs(eps)),
        "stress_pressure_relative": abs(stress_pressure - (mu_B * n_nat - eps)) / max(1.0, abs(eps)),
    }


def legacy_row(x: float, p: Params = PARAMS) -> dict[str, float]:
    """Recompute the maintained fixed-frequency/map branch for comparison only."""
    q_legacy = 47.04047619047619
    mu_theta = p.m_omega
    mu_legacy = math.sqrt(mu_theta * mu_theta + p.lam * p.W0 * p.W0)
    gamma = q_legacy * p.g_omega * p.n0_fm3 * p.hbar_c**3 / (p.m_omega * p.m_omega)
    omega = mu_theta - gamma * x
    W_legacy = math.sqrt(max(0.0, mu_legacy * mu_legacy - omega * omega) / p.lam)
    return {
        "n_over_n0": x,
        "gamma_MeV": gamma,
        "Omega_MeV": omega,
        "W_legacy_MeV": W_legacy,
        "legacy_melting_branch": bool(W_legacy == 0.0),
    }


REQUIRED_PASSPORT_FIELDS = {
    "input_id",
    "producer",
    "quantity",
    "value",
    "uncertainty",
    "unit",
    "role",
    "source",
    "citation",
    "scheme",
    "scale",
    "operator_definition",
    "transformation",
    "covariance_group",
    "metadata_status",
}


def passport() -> dict[str, Any]:
    records = [
        {
            "input_id": "W0",
            "producer": "article/NVG_VACUUM_W_FIELD_DERIVATION_RU.md",
            "quantity": "vacuum condensate anchor",
            "value": PARAMS.W0,
            "uncertainty": None,
            "unit": "MeV",
            "role": "declared anchor",
            "source": "article/NVG_VACUUM_W_FIELD_DERIVATION_RU.md",
            "citation": "article/NVG_VACUUM_W_FIELD_DERIVATION_RU.md",
            "scheme": "model-defined",
            "scale": "model",
            "operator_definition": "W amplitude",
            "transformation": "identity in declared model convention",
            "covariance_group": "anchors",
            "metadata_status": "DECLARED_NO_UNCERTAINTY",
        },
        {
            "input_id": "lambda",
            "producer": "article/NVG_VACUUM_W_FIELD_DERIVATION_RU.md",
            "quantity": "scalar self-coupling",
            "value": PARAMS.lam,
            "uncertainty": None,
            "unit": "1",
            "role": "declared model input",
            "source": "article/NVG_VACUUM_W_FIELD_DERIVATION_RU.md",
            "citation": "article/NVG_VACUUM_W_FIELD_DERIVATION_RU.md",
            "scheme": "model-defined",
            "scale": "model",
            "operator_definition": "quartic centered at W0",
            "transformation": "identity in declared model convention",
            "covariance_group": "anchors",
            "metadata_status": "DECLARED_NO_UNCERTAINTY",
        },
        {
            "input_id": "M_N",
            "producer": "article/NVG_SCIENTIFIC_ARTICLE_EN.md",
            "quantity": "nucleon mass anchor",
            "value": PARAMS.M_N,
            "uncertainty": None,
            "unit": "MeV",
            "role": "mass response anchor",
            "source": "article/NVG_SCIENTIFIC_ARTICLE_EN.md",
            "citation": "article/NVG_SCIENTIFIC_ARTICLE_EN.md",
            "scheme": "declared bookkeeping convention",
            "scale": "declared model scale",
            "operator_definition": "full nucleon mass",
            "transformation": "full-mass model identity",
            "covariance_group": "W0_shared",
            "metadata_status": "BOOKKEEPING_INPUT",
        },
        {
            "input_id": "m_omega",
            "producer": "verification/nvg_vacuum_w_field_derivation.py",
            "quantity": "vector mass anchor",
            "value": PARAMS.m_omega,
            "uncertainty": None,
            "unit": "MeV",
            "role": "Higgs charge derivation",
            "source": "verification/nvg_vacuum_w_field_derivation.py",
            "citation": "verification/nvg_vacuum_w_field_derivation.py",
            "scheme": "model-defined",
            "scale": "model",
            "operator_definition": "canonical vector mass",
            "transformation": "identity of canonical vector mass",
            "covariance_group": "W0_shared",
            "metadata_status": "BOOKKEEPING_INPUT",
        },
        {
            "input_id": "g_omega",
            "producer": "verification/nvg_vacuum_w_field_derivation.py",
            "quantity": "nucleon-vector coupling",
            "value": PARAMS.g_omega,
            "uncertainty": None,
            "unit": "1",
            "role": "matter-current coupling",
            "source": "verification/nvg_vacuum_w_field_derivation.py",
            "citation": "verification/nvg_vacuum_w_field_derivation.py",
            "scheme": "model-defined",
            "scale": "model",
            "operator_definition": "A_mu J_B^mu",
            "transformation": "charge convention bound",
            "covariance_group": "vector_matter",
            "metadata_status": "BOOKKEEPING_INPUT",
        },
        {
            "input_id": "n0",
            "producer": "verification/nvg_vacuum_w_field_derivation.py",
            "quantity": "reference baryon density",
            "value": PARAMS.n0_fm3,
            "uncertainty": None,
            "unit": "fm^-3",
            "role": "grid scale",
            "source": "verification/nvg_vacuum_w_field_derivation.py",
            "citation": "verification/nvg_vacuum_w_field_derivation.py",
            "scheme": "unit conversion",
            "scale": "nuclear reference",
            "operator_definition": "baryon number density",
            "transformation": "n_nat=n_fm3(hbar c)^3",
            "covariance_group": "units",
            "metadata_status": "KINEMATIC_INPUT",
        },
        {
            "input_id": "hbar_c",
            "producer": "verification/nvg_vacuum_w_field_derivation.py",
            "quantity": "unit conversion",
            "value": PARAMS.hbar_c,
            "uncertainty": None,
            "unit": "MeV fm",
            "role": "unit conversion",
            "source": "verification/source_complete_solution_audit.py",
            "citation": "verification/source_complete_solution_audit.py",
            "scheme": "natural units",
            "scale": "exact",
            "operator_definition": "fm^-1 to MeV",
            "transformation": "n_nat=n_fm3(hbar c)^3",
            "covariance_group": "units",
            "metadata_status": "KINEMATIC_INPUT",
        },
    ]
    derived = [
        {
            "input_id": "q_phi",
            "producer": "verification/source_complete_solution_audit.py",
            "quantity": "Higgs gauge charge",
            "value": PARAMS.q_phi,
            "uncertainty": None,
            "unit": "1",
            "role": "derived vector calibration",
            "source": "verification/contracts/source_complete_action.md",
            "citation": "verification/contracts/source_complete_action.md",
            "scheme": "canonical A_mu normalization",
            "scale": "model",
            "operator_definition": "m_A=q_phi W0",
            "transformation": "Jacobian propagated",
            "covariance_group": "W0_shared",
            "metadata_status": "DERIVED_CORRELATED",
            "formula": "m_omega/W0",
        },
        {
            "input_id": "g_s",
            "producer": "verification/source_complete_solution_audit.py",
            "quantity": "full nucleon scalar response",
            "value": PARAMS.g_s,
            "uncertainty": None,
            "unit": "1",
            "role": "derived mass response",
            "source": "verification/contracts/source_complete_action.md",
            "citation": "verification/contracts/source_complete_action.md",
            "scheme": "full-mass convention",
            "scale": "model",
            "operator_definition": "M*(W)=g_s W",
            "transformation": "Jacobian propagated",
            "covariance_group": "W0_shared",
            "metadata_status": "DERIVED_CORRELATED",
            "formula": "M_N/W0",
        },
    ]
    for rec in records + derived:
        missing = REQUIRED_PASSPORT_FIELDS - rec.keys()
        if missing:
            raise AssertionError(f"passport missing {sorted(missing)}")
    return {
        "records": records,
        "derived_records": derived,
        "renormalization_block": {
            "record_count": 37,
            "metadata_status": "BLOCKED_MISSING_SCHEME_SCALE_OPERATOR_TRANSFORMATION",
            "missing_fields": ["scheme", "scale", "operator_definition", "transformation"],
            "parameter_refit": "NOT_PERFORMED",
        },
    }


PASSPORT_TOP_KEYS = {"records", "derived_records", "renormalization_block"}
BASE_IDS = {"W0", "lambda", "M_N", "m_omega", "g_omega", "n0", "hbar_c"}
DERIVED_IDS = {"q_phi", "g_s"}
PASSPORT_STATUS_ALLOWLIST = {
    "DECLARED_NO_UNCERTAINTY",
    "BOOKKEEPING_INPUT",
    "KINEMATIC_INPUT",
    "DERIVED_CORRELATED",
}
PASSPORT_UNITS = {
    "W0": "MeV",
    "lambda": "1",
    "M_N": "MeV",
    "m_omega": "MeV",
    "g_omega": "1",
    "n0": "fm^-3",
    "hbar_c": "MeV fm",
    "q_phi": "1",
    "g_s": "1",
}
FORBIDDEN_PLACEHOLDERS = {"", "unknown", "not supplied", "none declared", "n/a", "na"}


def _markdown_anchor_slugs(text: str) -> set[str]:
    """Return exact GitHub-style heading slugs present in Markdown text."""
    slugs: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        heading = re.sub(r"<[^>]*>", "", match.group(1)).strip().lower()
        heading = re.sub(r"[^\w\s-]", "", heading, flags=re.UNICODE)
        heading = re.sub(r"[\s-]+", "-", heading).strip("-")
        if heading:
            slugs.add(heading)
    return slugs


def _markdown_anchor_resolves(target: Path, anchor: str) -> bool:
    """Resolve an anchor only to an exact heading slug or explicit HTML id."""
    if not anchor or anchor.strip() != anchor:
        return False
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if anchor in _markdown_anchor_slugs(text):
        return True
    html_ids = set(re.findall(r"\bid\s*=\s*([\"'])([^\"']+)\1", text))
    return any(value == anchor for _, value in html_ids)


def _resolved_reference(ref: Any) -> bool:
    if not isinstance(ref, str) or not ref.strip():
        return False
    path, _, anchor = ref.partition("#")
    target = REPO_ROOT / path
    if not target.is_file():
        return False
    if anchor:
        return _markdown_anchor_resolves(target, anchor)
    if "#" in ref:
        # An explicitly supplied trailing '#' is an empty anchor, not a
        # no-anchor reference.
        return False
    return True


def validate_passport(payload: Any) -> dict[str, Any]:
    """Strict semantic passport gate; returns diagnostics instead of raising."""
    errors: list[str] = []
    if not isinstance(payload, dict) or set(payload) != PASSPORT_TOP_KEYS:
        return {"pass": False, "errors": ["top-level keys must be exactly records, derived_records, renormalization_block"]}
    records = payload.get("records")
    derived = payload.get("derived_records")
    block = payload.get("renormalization_block")
    if not isinstance(records, list) or not isinstance(derived, list):
        return {"pass": False, "errors": ["records and derived_records must be lists"]}
    if len(records) != 7 or len(derived) != 2:
        errors.append("record count must be exactly seven base plus two derived")
    ids = [r.get("input_id") for r in records + derived if isinstance(r, dict)]
    if len(ids) != len(set(ids)):
        errors.append("duplicate record ID")
    if set(ids[: len(records)]) != BASE_IDS or set(ids[len(records) :]) != DERIVED_IDS:
        errors.append("base/derived ID sets do not match exact contract")
    for index, rec in enumerate(records + derived):
        is_derived = index >= len(records)
        if not isinstance(rec, dict):
            errors.append(f"record {index} is not an object")
            continue
        expected = REQUIRED_PASSPORT_FIELDS | ({"formula"} if is_derived else set())
        if set(rec) != expected:
            errors.append(f"record {index} fields are not exact")
        ident = rec.get("input_id")
        if ident not in PASSPORT_UNITS or rec.get("unit") != PASSPORT_UNITS.get(ident):
            errors.append(f"record {index} has invalid unit")
        if not isinstance(rec.get("value"), (int, float)) or isinstance(rec.get("value"), bool) or not math.isfinite(float(rec.get("value"))):
            errors.append(f"record {index} value is not finite")
        for field in REQUIRED_PASSPORT_FIELDS - {"value", "uncertainty"}:
            value = rec.get(field)
            if not isinstance(value, str) or value.strip().lower() in FORBIDDEN_PLACEHOLDERS:
                errors.append(f"record {index} field {field} is blank/placeholder")
        uncertainty = rec.get("uncertainty")
        if uncertainty is not None and (not isinstance(uncertainty, (int, float)) or not math.isfinite(float(uncertainty)) or float(uncertainty) < 0.0):
            errors.append(f"record {index} uncertainty is invalid")
        if rec.get("metadata_status") not in PASSPORT_STATUS_ALLOWLIST:
            errors.append(f"record {index} metadata status is not allowlisted")
        if not _resolved_reference(rec.get("producer")) or not _resolved_reference(rec.get("citation")) or not _resolved_reference(rec.get("source")):
            errors.append(f"record {index} producer/source/citation target does not resolve")
        if any(token in " ".join(str(rec.get(field, "")).lower() for field in ("role", "source", "citation", "scheme", "scale", "operator_definition", "transformation")) for token in ("fit", "threshold")):
            errors.append(f"record {index} carries forbidden fit/threshold status")
        if is_derived and rec.get("covariance_group") != "W0_shared":
            errors.append(f"derived record {index} covariance group must be W0_shared")
    if isinstance(block, dict):
        if set(block) != {"record_count", "metadata_status", "missing_fields", "parameter_refit"}:
            errors.append("renormalization block fields are not exact")
        if block.get("record_count") != 37:
            errors.append("renormalization record count must be 37")
        if block.get("metadata_status") != "BLOCKED_MISSING_SCHEME_SCALE_OPERATOR_TRANSFORMATION":
            errors.append("renormalization status changed")
        if set(block.get("missing_fields", [])) != {"scheme", "scale", "operator_definition", "transformation"}:
            errors.append("renormalization missing-field set changed")
        if block.get("parameter_refit") != "NOT_PERFORMED":
            errors.append("parameter refit must remain NOT_PERFORMED")
    else:
        errors.append("renormalization block must be an object")
    base_values = {
        rec.get("input_id"): rec.get("value")
        for rec in records
        if isinstance(rec, dict)
    }
    expected_derived: dict[str, float] = {}
    try:
        W0_payload = float(base_values["W0"])
        if not math.isfinite(W0_payload) or W0_payload == 0.0:
            raise ValueError("W0 is nonfinite or zero")
        expected_derived = {
            "q_phi": float(base_values["m_omega"]) / W0_payload,
            "g_s": float(base_values["M_N"]) / W0_payload,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("derived values cannot be recomputed from payload base records")
    expected_formulas = {"q_phi": "m_omega/W0", "g_s": "M_N/W0"}
    for rec in derived if isinstance(derived, list) else []:
        ident = rec.get("input_id") if isinstance(rec, dict) else None
        if ident in expected_formulas:
            if not isinstance(rec.get("formula"), str) or rec.get("formula") != expected_formulas[ident]:
                errors.append(f"derived formula for {ident} is invalid")
            if expected_derived:
                try:
                    if abs(float(rec.get("value", float("nan"))) - expected_derived[ident]) / max(1.0, abs(expected_derived[ident])) > 1.0e-15:
                        errors.append(f"derived value for {ident} does not recompute from payload")
                except (TypeError, ValueError):
                    errors.append(f"derived value for {ident} does not recompute from payload")
    return {"pass": not errors, "errors": errors, "record_count": len(ids)}


def covariance_control() -> dict[str, Any]:
    """Pure control of correlated derived inputs; not a physical uncertainty fit."""
    x = [939.0, 782.6, 859.0]
    C = [[4.0, 0.6, 0.8], [0.6, 9.0, 1.2], [0.8, 1.2, 16.0]]
    J = [[0.0, 1.0 / x[2], -x[1] / (x[2] ** 2)], [1.0 / x[2], 0.0, -x[0] / (x[2] ** 2)]]
    # Explicit compensated scalar summation is stable across the Python 3.12
    # change to builtin sum. Do not use this for the matrix-valued Dirac sums.
    cy = [[math.fsum(J[i][a] * C[a][b] * J[j][b] for a in range(3) for b in range(3)) for j in range(2)] for i in range(2)]
    K = [[1.1, 0.1, 0.0], [0.0, 0.9, 0.05], [0.02, 0.0, 1.2]]
    # x'=Kx, C'=K C K^T, J'=J K^-1.  A tiny hand-coded inverse keeps this
    # control dependency-free and makes the tested covariance identity explicit.
    det = (
        K[0][0] * (K[1][1] * K[2][2] - K[1][2] * K[2][1])
        - K[0][1] * (K[1][0] * K[2][2] - K[1][2] * K[2][0])
        + K[0][2] * (K[1][0] * K[2][1] - K[1][1] * K[2][0])
    )
    Ki = [[0.0] * 3 for _ in range(3)]
    Ki[0][0] = (K[1][1] * K[2][2] - K[1][2] * K[2][1]) / det
    Ki[0][1] = (K[0][2] * K[2][1] - K[0][1] * K[2][2]) / det
    Ki[0][2] = (K[0][1] * K[1][2] - K[0][2] * K[1][1]) / det
    Ki[1][0] = (K[1][2] * K[2][0] - K[1][0] * K[2][2]) / det
    Ki[1][1] = (K[0][0] * K[2][2] - K[0][2] * K[2][0]) / det
    Ki[1][2] = (K[0][2] * K[1][0] - K[0][0] * K[1][2]) / det
    Ki[2][0] = (K[1][0] * K[2][1] - K[1][1] * K[2][0]) / det
    Ki[2][1] = (K[0][1] * K[2][0] - K[0][0] * K[2][1]) / det
    Ki[2][2] = (K[0][0] * K[1][1] - K[0][1] * K[1][0]) / det
    Cp = [[math.fsum(K[i][a] * C[a][b] * K[j][b] for a in range(3) for b in range(3)) for j in range(3)] for i in range(3)]
    Jp = [[math.fsum(J[i][a] * Ki[a][b] for a in range(3)) for b in range(3)] for i in range(2)]
    cyp = [[math.fsum(Jp[i][a] * Cp[a][b] * Jp[j][b] for a in range(3) for b in range(3)) for j in range(2)] for i in range(2)]
    err = max(abs(cy[i][j] - cyp[i][j]) for i in range(2) for j in range(2))
    return {
        "control_only": True,
        "jacobian": J,
        "covariance": C,
        "derived_covariance": cy,
        "reparameterized_max_abs_error": err,
        "invariant_pass": err < 1.0e-12,
    }


def massless_controls(p: Params = PARAMS) -> dict[str, Any]:
    rows = []
    for x in (1.0, 10.0):
        n_nat = x * p.n0_nat
        k = fermi_momentum(n_nat, p)
        exact_e = p.degeneracy * k**4 / (8.0 * math.pi**2)
        exact_p = p.degeneracy * k**4 / (24.0 * math.pi**2)
        e0 = fermi_free_energy(n_nat, 0.0, p)
        p0 = fermi_pressure(n_nat, 0.0, p)
        tiny_mass = 1.0e-6 * k
        e_small = fermi_free_energy(n_nat, tiny_mass, p)
        p_small = fermi_pressure(n_nat, tiny_mass, p)
        rows.append(
            {
                "n_over_n0": x,
                "energy_exact_MeV4": exact_e,
                "pressure_exact_MeV4": exact_p,
                "energy_formula_relative": abs(e0 - exact_e) / exact_e,
                "pressure_formula_relative": abs(p0 - exact_p) / exact_p,
                "energy_continuity_relative": abs(e_small - exact_e) / exact_e,
                "pressure_continuity_relative": abs(p_small - exact_p) / exact_p,
                "massless_trace_relative": abs(e0 - 3.0 * p0) / exact_e,
            }
        )
    return {
        "rows": rows,
        "exact_pass": all(row["energy_formula_relative"] < 1.0e-14 and row["pressure_formula_relative"] < 1.0e-14 and row["massless_trace_relative"] < 1.0e-14 for row in rows),
        "continuity_pass": all(row["energy_continuity_relative"] < 1.0e-9 and row["pressure_continuity_relative"] < 1.0e-9 for row in rows),
    }


def convergence_controls(p: Params = PARAMS) -> dict[str, Any]:
    rows = []
    for x in (0.0, 0.1, 1.0, 2.5, 10.0):
        n_nat = x * p.n0_nat
        values = [solve_W(n_nat, p, tol=t) for t in (1.0e-8, 1.0e-10, 1.0e-12)]
        rows.append({"n_over_n0": x, "W_by_tolerance": values, "max_delta_MeV": max(values) - min(values)})
    return {
        "rows": rows,
        "max_delta_MeV": max(row["max_delta_MeV"] for row in rows),
        "pass": max(row["max_delta_MeV"] for row in rows) < 1.0e-7,
    }


def reduced_lapse_scale_action(lapse: float, scale: float, B_nat: float, W: float, A0: float, p: Params = PARAMS) -> float:
    """Reduced covariant action at fixed coordinate baryon number B.

    ``A0`` is the covariant one-form component.  The lapse/scale factors are
    varied before the fields are re-solved, so holding ``n`` fixed here would
    be an incorrect metric variation.
    """
    n_nat = B_nat / scale**3
    # Evaluate the stationary action in extended precision.  The closed-form
    # Fermi integral subtracts two ~1e11 terms for nonrelativistic nucleons;
    # ordinary double precision would inject a 1e-9-level noise into the
    # pressure derivative at the 0.1 n0 row.
    # Restore the caller's precision on both normal and exceptional exits.
    with mp.workdps(50):
        mm = mp.mpf
        n_mp, W_mp, A_mp = mm(n_nat), mm(W), mm(A0)
        k_mp = (mm(6.0) * mp.pi**2 * n_mp / mm(p.degeneracy)) ** (mm(1.0) / mm(3.0)) if n_nat > 0.0 else mm(0.0)
        m_mp = mm(p.g_s) * W_mp
        if n_nat > 0.0:
            e_mp = mm(p.degeneracy) / (mm(16.0) * mp.pi**2) * (
                k_mp * mp.sqrt(k_mp**2 + m_mp**2) * (mm(2.0) * k_mp**2 + m_mp**2)
                - m_mp**4 * mp.asinh(k_mp / m_mp)
            )
        else:
            e_mp = mm(0.0)
        v_mp = mm(p.lam) * (W_mp**2 - mm(p.W0)**2) ** 2 / mm(4.0)
        # The final source term is simply -g_omega*A0*B; keep it explicit to
        # avoid a 0/0 branch at vacuum.
        out = (
            -mm(lapse) * mm(scale)**3 * (e_mp + v_mp)
            + mm(scale)**3 * mm(p.q_phi)**2 * W_mp**2 * A_mp**2 / (mm(2.0) * mm(lapse))
            - mm(p.g_omega) * A_mp * mm(B_nat)
        )
        return float(out)


def solve_metric_fields(lapse: float, scale: float, B_nat: float, p: Params = PARAMS, tol: float = 1.0e-11) -> tuple[float, float]:
    """Re-solve W and coordinate A0 at each lapse/scale perturbation."""
    n_nat = B_nat / scale**3
    W = solve_W(n_nat, p, tol=tol)
    A0 = lapse * gauss_A0(n_nat, W, p)
    return W, A0


def metric_variation_observables(
    x: float,
    p: Params = PARAMS,
    solver_tol: float = 1.0e-11,
    metric_step: float = 2.5e-5,
) -> dict[str, float]:
    """Independent centered lapse/scale variation of the stationary action."""
    B_nat = x * p.n0_nat
    W, _ = solve_metric_fields(1.0, 1.0, B_nat, p, solver_tol)

    def stationary_lapse(lapse: float) -> float:
        Wp, A0p = solve_metric_fields(lapse, 1.0, B_nat, p, solver_tol)
        return reduced_lapse_scale_action(lapse, 1.0, B_nat, Wp, A0p, p)

    def stationary_scale(scale: float) -> float:
        Wp, A0p = solve_metric_fields(1.0, scale, B_nat, p, solver_tol)
        return reduced_lapse_scale_action(1.0, scale, B_nat, Wp, A0p, p)

    h = metric_step
    # Five-point centered derivatives retain the mandated perturbation scales
    # while removing the O(h^2) truncation that otherwise masks the required
    # 1e-9 convergence at the finest 2.5e-5 scale.
    def five_point(f, center: float) -> float:
        # The stationary action carries an O(1e8) constant at sub-saturation
        # density.  Subtracting the center value before the stencil preserves
        # the derivative (the coefficients sum to zero) while avoiding loss of
        # significance in the alternating sum.
        f0 = f(center)
        return (
            (f(center - 2.0 * h) - f0)
            - 8.0 * (f(center - h) - f0)
            + 8.0 * (f(center + h) - f0)
            - (f(center + 2.0 * h) - f0)
        ) / (12.0 * h)

    epsilon = -five_point(stationary_lapse, 1.0)
    pressure = five_point(stationary_scale, 1.0) / 3.0
    mu_B = chemical_potential(B_nat, W, p)
    return {
        "W_MeV": W,
        "epsilon_metric_MeV_fm3": epsilon / p.hbar_c**3,
        "pressure_metric_MeV_fm3": pressure / p.hbar_c**3,
        "mu_B_MeV": mu_B,
    }


def metric_variation_controls(p: Params = PARAMS) -> dict[str, Any]:
    xs = (0.0, 0.1, 1.0, 2.5, 10.0)
    tolerances = (1.0e-8, 1.0e-10, 1.0e-12)
    steps = (1.0e-4, 5.0e-5, 2.5e-5)
    rows: list[dict[str, Any]] = []
    convergence_pass = True
    for x in xs:
        by_tol = []
        by_step = []
        for tol in tolerances:
            vals = [metric_variation_observables(x, p, tol, step) for step in steps]
            finest = vals[-1]
            by_step.append(
                {
                    "solver_tol": tol,
                    "metric_steps": steps,
                    "epsilon_metric_by_step": [v["epsilon_metric_MeV_fm3"] for v in vals],
                    "pressure_metric_by_step": [v["pressure_metric_MeV_fm3"] for v in vals],
                    "epsilon_step_relative": (max(v["epsilon_metric_MeV_fm3"] for v in vals) - min(v["epsilon_metric_MeV_fm3"] for v in vals)) / max(1.0, max(abs(v["epsilon_metric_MeV_fm3"]) for v in vals)),
                    "pressure_step_relative": (max(v["pressure_metric_MeV_fm3"] for v in vals) - min(v["pressure_metric_MeV_fm3"] for v in vals)) / max(1.0, max(abs(v["pressure_metric_MeV_fm3"]) for v in vals)),
                }
            )
            by_tol.append(finest)
        def spread(key: str) -> float:
            values = [v[key] for v in by_tol]
            return (max(values) - min(values)) / max(1.0, max(abs(v) for v in values))

        action = branch_row(x, p)
        row = {
            "n_over_n0": x,
            "by_solver_tolerance": by_tol,
            "by_metric_step": by_step,
            "W_spread_MeV": max(v["W_MeV"] for v in by_tol) - min(v["W_MeV"] for v in by_tol),
            "epsilon_metric_spread_relative": spread("epsilon_metric_MeV_fm3"),
            "mu_B_spread_relative": spread("mu_B_MeV"),
            "pressure_metric_spread_relative": spread("pressure_metric_MeV_fm3"),
            "epsilon_metric_action_relative": abs(by_tol[-1]["epsilon_metric_MeV_fm3"] - action["epsilon_MeV_fm3"]) / max(1.0, abs(action["epsilon_MeV_fm3"])),
            "pressure_metric_action_relative": abs(by_tol[-1]["pressure_metric_MeV_fm3"] - action["pressure_MeV_fm3"]) / max(1.0, abs(action["pressure_MeV_fm3"])),
        }
        convergence_pass = convergence_pass and row["W_spread_MeV"] < 1.0e-7 and row["epsilon_metric_spread_relative"] < 1.0e-9 and row["mu_B_spread_relative"] < 1.0e-9 and row["pressure_metric_spread_relative"] < 1.0e-9
        convergence_pass = convergence_pass and all(s["epsilon_step_relative"] < 1.0e-6 and s["pressure_step_relative"] < 1.0e-6 for s in by_step)
        convergence_pass = convergence_pass and row["epsilon_metric_action_relative"] < 1.0e-6 and row["pressure_metric_action_relative"] < 1.0e-6
        rows.append(row)
    return {"solver_tolerances": tolerances, "metric_steps": steps, "rows": rows, "pass": convergence_pass}


def analytic_controls(p: Params = PARAMS) -> dict[str, Any]:
    vacuum_W = solve_W(0.0, p)
    vacuum = {
        "W_MeV": vacuum_W,
        "W_expected_MeV": p.W0,
        "A0_MeV": gauss_A0(0.0, vacuum_W, p),
        "Mstar_MeV": matter_mass(vacuum_W, p),
        "M_expected_MeV": p.M_N,
        "epsilon_MeV4": energy_density(0.0, vacuum_W, p),
        "stationarity_MeV3": W_stationarity(vacuum_W, 0.0, p),
    }
    gs0 = replace(p, M_N=0.0)
    n_nat = gs0.n0_nat
    W0_vector = solve_W(n_nat, gs0)
    vector_only = {
        "W_MeV": W0_vector,
        "stationarity_MeV3": W_stationarity(W0_vector, n_nat, gs0),
        "gs": gs0.g_s,
    }
    divergence = []
    for factor in (1.0e-1, 1.0e-2, 1.0e-3):
        W = p.W0 * factor
        divergence.append({"W_over_W0": factor, "energy_MeV4": energy_density(n_nat, W, p)})
    return {
        "vacuum": vacuum,
        "vector_only_limit": vector_only,
        "W_to_zero_energy_samples": divergence,
        "vacuum_pass": abs(vacuum_W - p.W0) < 1.0e-12 and abs(vacuum["Mstar_MeV"] - p.M_N) < 1.0e-12 and abs(vacuum["epsilon_MeV4"]) < 1.0e-8,
        # The vector-only polynomial is an analytic-limit control; use a
        # sub-1e-4 MeV^3 residual, well below the MeV^3 scale of the terms.
        "vector_only_stationarity_pass": abs(vector_only["stationarity_MeV3"]) < 1.0e-4,
        "W_to_zero_diverges": divergence[2]["energy_MeV4"] > divergence[0]["energy_MeV4"] * 1.0e2,
    }


def build_result(p: Params = PARAMS) -> dict[str, Any]:
    xs = (0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
    rows = [branch_row(x, p) for x in xs]
    analytic = analytic_controls(p)
    massless = massless_controls(p)
    convergence = convergence_controls(p)
    metric = metric_variation_controls(p)
    gauge = manufactured_gauge_controls(p)
    euler_ward = manufactured_euler_ward_controls(p)
    cov = covariance_control()
    pp = passport()
    passport_validation = validate_passport(pp)
    max_current = max(r["total_current_relative"] for r in rows)
    max_stationarity = max(r["w_stationarity_relative"] for r in rows)
    max_hvh_mu = max(r["hvh_mu_relative"] for r in rows if r["n_over_n0"] > 0.0)
    max_hvh_pressure = max(r["hvh_pressure_relative"] for r in rows)
    max_stress_pressure = max(r["stress_pressure_relative"] for r in rows)
    legacy = [legacy_row(x, p) for x in (0.0, 1.0, 2.5)]
    renorm_matching_available = pp["renormalization_block"]["metadata_status"] == "COMPLETE"
    legacy_threshold_supported = not any(row["legacy_melting_branch"] for row in legacy)
    W_zero_at_finite_density = any(row["W_MeV"] <= 0.0 for row in rows if row["n_over_n0"] > 0.0)
    gates = {
        "analytic_limits": analytic["vacuum_pass"] and analytic["vector_only_stationarity_pass"] and analytic["W_to_zero_diverges"] and massless["exact_pass"] and massless["continuity_pass"],
        "coupled_branch_stationarity": max_stationarity < 1.0e-8,
        "gauss_total_current": max_current < 1.0e-14,
        "hvh_legendre_closure": max_hvh_mu < 1.0e-7 and max_hvh_pressure < 1.0e-12 and max_stress_pressure < 1.0e-11,
        "grid_root_convergence": convergence["pass"],
        "metric_stress_convergence": metric["pass"],
        "local_u1_covariance_current_ward": gauge["all_pass"] and euler_ward["all_pass"],
        "euler_noether_ward_identity": euler_ward["all_pass"],
        "correlated_derived_input_covariance": cov["invariant_pass"],
        "passport_complete_for_contract": passport_validation["pass"],
        "renorm_matching_available": renorm_matching_available,
        "legacy_threshold_supported": legacy_threshold_supported,
        "W_zero_at_finite_density": W_zero_at_finite_density,
    }
    status = (
        "PASS_CONTRACT_FAIL_CLOSED_MELTING_NO_GO"
        if all(gates[k] for k in ("analytic_limits", "coupled_branch_stationarity", "gauss_total_current", "hvh_legendre_closure", "grid_root_convergence", "metric_stress_convergence", "local_u1_covariance_current_ward", "euler_noether_ward_identity", "correlated_derived_input_covariance", "passport_complete_for_contract"))
        and not gates["renorm_matching_available"]
        and not gates["legacy_threshold_supported"]
        and not gates["W_zero_at_finite_density"]
        else "FAIL_CLOSED_SOURCE_COMPLETE_AUDIT"
    )
    return {
        "schema": "source_complete_solution_audit.v1",
        "status": status,
        "contract": {
            "metric": "(+---)",
            "ensemble": "zero-temperature fixed baryon density, source-free gauge-neutral",
            "potential": "lambda*(W^2-W0^2)^2/4",
            "mass_response": "Mstar=g_s*W, g_s=M_N/W0",
            "vector_mass": "m_A=q_phi*W, q_phi=m_omega/W0",
            "vector_source": "dynamical Gauss equation; no external A0",
            "transfer_branch": "NOT_IMPLEMENTED_BY_BINDING",
            "fitted_threshold": "NOT_PERFORMED",
        },
        "parameters": {
            "W0_MeV": p.W0,
            "lambda": p.lam,
            "M_N_MeV": p.M_N,
            "g_s": p.g_s,
            "m_omega_MeV": p.m_omega,
            "q_phi": p.q_phi,
            "g_omega": p.g_omega,
            "mu_scalar_MeV": p.mu0,
            "scalar_mass_MeV": p.scalar_mass,
            "n0_fm3": p.n0_fm3,
            "hbar_c_MeV_fm": p.hbar_c,
            "degeneracy": p.degeneracy,
        },
        "analytic_controls": analytic,
        "massless_controls": massless,
        "convergence": convergence,
        "metric_variation": metric,
        "manufactured_gauge_controls": gauge,
        "manufactured_euler_ward_controls": euler_ward,
        "rows": rows,
        "legacy_comparison": legacy,
        "closure_summary": {
            "max_current_relative": max_current,
            "max_W_stationarity_relative": max_stationarity,
            "max_HVH_mu_relative": max_hvh_mu,
            "max_HVH_pressure_relative": max_hvh_pressure,
            "max_stress_pressure_relative": max_stress_pressure,
            "max_generator_residual_MeV4": max(abs(r["generator_residual_MeV4"]) for r in rows),
        },
        "passport": pp,
        "passport_validation": passport_validation,
        "covariance_control": cov,
        "gates": gates,
        "scientific_boundary": {
            "candidate_branch": "W>0 for all scanned n_B; mild low-density suppression and vector-driven growth",
            "W_zero_claim": "NO_GO: eliminated energy has g_omega^2*n_B^2/(2*q_phi^2*W^2) divergence",
            "legacy_threshold": "NOT_SUPPORTED: maintained 2.5 n0 belongs to external fixed-frequency branch",
            "renormalization": "BLOCKED: 37 maintained records lack scheme/scale/operator transformation",
            "parameter_refit": "NOT_PERFORMED",
        },
    }


def serialize_result(result: dict[str, Any]) -> bytes:
    return (json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def render_report(result: dict[str, Any]) -> str:
    p = result["parameters"]
    c = result["closure_summary"]
    g = result["gates"]
    rows = result["rows"]
    legacy = result["legacy_comparison"]
    massless = result["massless_controls"]
    gauge = result["manufactured_gauge_controls"]
    euler_ward = result["manufactured_euler_ward_controls"]
    metric = result["metric_variation"]
    passport_validation = result["passport_validation"]
    return "\n".join(
        [
            "# Source-complete solution audit",
            "",
            "## Control block",
            "",
            f"- Status: **{result['status']}**.",
            "- Contract: exact local U(1), same-scalar Higgs vector mass, dynamical Gauss-neutral field, fixed baryon density, one stationary generator; no external source, transfer vertex, bare Proca term, or fitted threshold.",
            "- Owned artifacts: `verification/source_complete_solution_audit.py`, `test_source_complete_solution_audit.py`, generated JSON/report; the maintained action snapshot is `verification/contracts/source_complete_action.md`.",
            "",
            "## Reconstructed solution",
            "",
            f"`W0={p['W0_MeV']:.3f} MeV`, `lambda={p['lambda']:.5f}`, `g_s={p['g_s']:.9f}`, and `q_phi={p['q_phi']:.9f}` are derived from the bound action. The source-free vacuum is `W=W0`, `A0=0`, `M*=M_N`, `epsilon=0`; the scalar mass parameter is `mu=sqrt(lambda)W0={p['mu_scalar_MeV']:.6f} MeV`.",
            f"The coupled stationarity equation is solved with `A0=g_omega*n_B/(q_phi^2 W^2)` and `lambda W(W^2-W0^2)+g_s n_s-g_omega^2 n_B^2/(q_phi^2 W^3)=0`. The bound branch stays positive (a small scalar-density suppression at nuclear density, followed by vector-driven growth); at `2.5 n0`, `W={next(r['W_MeV'] for r in rows if r['n_over_n0']==2.5):.6f} MeV` and `W/W0={next(r['W_over_W0'] for r in rows if r['n_over_n0']==2.5):.6f}`.",
            "",
            "## Closure and convergence",
            "",
            f"- Analytic vacuum/vector-only limits: **{'PASS' if g['analytic_limits'] else 'FAIL'}**; `W -> 0` energy divergence: **{'PASS' if result['analytic_controls']['W_to_zero_diverges'] else 'FAIL'}**.",
            f"- Coupled W stationarity max relative residual: `{c['max_W_stationarity_relative']:.3e}`; Gauss/total-current max relative residual: `{c['max_current_relative']:.3e}`.",
            f"- HVH/Legendre closure: max `|mu_B-d epsilon/dn|/|mu_B|={c['max_HVH_mu_relative']:.3e}`; pressure identity max relative `{c['max_HVH_pressure_relative']:.3e}`; Hilbert stress-pressure residual `{c['max_stress_pressure_relative']:.3e}`; generator residual max `{c['max_generator_residual_MeV4']:.3e} MeV^4`.",
            f"- Root/grid convergence: max W spread across tolerances `{result['convergence']['max_delta_MeV']:.3e} MeV` (**{'PASS' if g['grid_root_convergence'] else 'FAIL'}**).",
            f"- Massless Fermi limits at `n/n0=1,10`: exact formulas and `epsilon=3P` trace **{'PASS' if massless['exact_pass'] else 'FAIL'}**; `m/k_F=1e-6` continuity **{'PASS' if massless['continuity_pass'] else 'FAIL'}** (max residual `{max(max(r['energy_continuity_relative'], r['pressure_continuity_relative']) for r in massless['rows']):.3e}`).",
            f"- Manufactured local-U(1) covariance/action/current controls: **{'PASS' if g['local_u1_covariance_current_ward'] else 'FAIL'}**; single covariant nucleon coupling/no duplicate explicit current **{'PASS' if gauge['single_covariant_nucleon_coupling'] and gauge['no_duplicate_explicit_current'] else 'FAIL'}**; A-variation current relative `{gauge['A_variation_current_relative']:.3e}`.",
            f"- Explicit off-shell Euler-derivative Noether/Ward identity (E_A, E_Phi, E_Phi*, E_N, E_barN): **{'PASS' if euler_ward['all_pass'] else 'FAIL'}**, normalized residual `{euler_ward['identity_relative']:.3e}`; analytic expressions and independent nested mp.diff at `{euler_ward['precision_dps']}` digits. The former binary64 cancellation stencil is not used; transformed-action directions remain a separate smoke test.",
            f"- Independent fixed-coordinate-`B` lapse/scale metric variation with re-solved `A0/W`: **{'PASS' if g['metric_stress_convergence'] else 'FAIL'}**; steps `{metric['metric_steps']}`, max metric/action closure `{max(max(r['epsilon_metric_action_relative'], r['pressure_metric_action_relative']) for r in metric['rows']):.3e}`.",
            f"- Strict passport validation: **{'PASS' if passport_validation['pass'] else 'FAIL'}**; exactly `{passport_validation['record_count']}` records (7 base + 2 derived), formula-to-ID mapping and resolved references enforced; renormalization block remains count `37` and `{result['passport']['renormalization_block']['metadata_status']}`.",
            "",
            "| n/n0 | W (MeV) | W/W0 | M* (MeV) | A0 (MeV) | epsilon (MeV/fm3) | mu_B (MeV) | P (MeV/fm3) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        + [
            f"| {r['n_over_n0']:.1f} | {r['W_MeV']:.6f} | {r['W_over_W0']:.6f} | {r['Mstar_MeV']:.6f} | {r['A0_MeV']:.6f} | {r['epsilon_MeV_fm3']:.6f} | {r['mu_B_MeV']:.6f} | {r['pressure_MeV_fm3']:.6f} |"
            for r in rows
        ]
        + [
            "",
            "## Legacy comparison and blockers",
            "",
            f"The maintained fixed-frequency/map formulas recompute `gamma={legacy[0]['gamma_MeV']:.9f} MeV` and melt at the legacy grid branch (`W_legacy(2.5 n0)={legacy[-1]['W_legacy_MeV']:.3f} MeV`). That branch is not generated by the bound action; the source-complete branch has no finite-energy `W=0` state for `g_omega n_B != 0` because the eliminated vector term scales as `W^-2`.",
            f"The Higgs calibration is `m_A=q_phi W0=m_omega={p['m_omega_MeV']:.3f} MeV`; the old `q=47.040476` is not reused. Correlated `(q_phi,g_s)` propagation passes the Jacobian covariance identity (control-only error `{result['covariance_control']['reparameterized_max_abs_error']:.3e}`).",
            f"Passport status is explicit: `{result['passport']['renormalization_block']['record_count']}` maintained renormalisation records remain `{result['passport']['renormalization_block']['metadata_status']}`; matching and refit are absent. Gates `renorm_matching_available`, `legacy_threshold_supported`, and `W_zero_at_finite_density` therefore remain false by contract.",
            "",
            "## Scientific boundary",
            "",
            "This executable surface supports a constructive equilibrium EFT contract and a density-melting no-go. It does not validate the maintained 2.5 n0 melting/bounce claim, substitute the legacy external A0 map, infer a transfer reservoir, or promote scheme-dependent mass fractions. Regenerate with `python verification/source_complete_solution_audit.py`; check independent byte regeneration with `python -m pytest -q verification/test_source_complete_solution_audit.py`.",
            "",
        ]
    )


def write_artifacts(result: dict[str, Any], result_path: Path = RESULT_PATH, report_path: Path = REPORT_PATH) -> None:
    result_path.write_bytes(serialize_result(result))
    report_path.write_text(render_report(result), encoding="utf-8")


def main() -> int:
    write_artifacts(build_result())
    print(f"status={build_result()['status']}")
    print(f"result={RESULT_PATH}")
    print(f"report={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
