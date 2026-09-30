#!/usr/bin/env python3
r"""Independent Hartle second-order (spin-quadrupole) producer.

This closes the repository's own declared blocker: the canonical first-order
Hartle producer (:mod:`nvg_hartle_slow_rotation`) reports
``quadrupole_status = BLOCKED`` because "the repository supplies no complete
second-order Hartle equations, gauge choice, centre data, or vacuum matching
conditions".  This module supplies exactly those dependencies from primary
sources and integrates them on the frozen canonical background.

Equation provenance (all in the repository's own convention
``g_tt = -exp(nu)`` single exponent, ``exp(lambda) = (1-2m/r)^{-1}``,
``j^2 = exp(-(nu+lambda))``, ``omega_tilde = Omega - omega``):

* l = 2 interior system ``(h2, K2, m2)`` and exterior closed forms:
  Yagi & Yunes (2013), PRD 88, 023009, arXiv:1303.1528 (Eqs. m2/k2R/h2R and
  h2-ext/k2-ext/m2-ext).  Symbolically verified in ``scratch/hartle_yy_verify.py``
  (sections A/B/D: 24 checks, residuals < 1e-24).
* l = 0 interior system ``(m0, p0)`` with centre data:
  Urbanec, Miller & Stuchlik (2013), MNRAS 433, 1906, arXiv:1301.5925.
  NOTE: their printed vacuum ``v2`` exterior carries a sign error
  (``+J^2/r^4``); the correct value ``-J^2/r^4`` is used here and is the one
  consistent with their own ``dv2/dr`` equation (verify B5/B6).  Their l = 0
  system and centre data are correct as printed.
* Newtonian validation target: Maclaurin spheroid,
  ``q_tilde -> (25/8) R/M``, ``Q -> -(1/2) Omega^2 R^5`` as ``C -> 0``
  (verify E1/E2).

Method (Hartle 1967): the l = 2 interior solution is built as a particular
integral (test centre amplitude ``B_p``) plus ``kappa`` times a complementary
(homogeneous) solution (test amplitude ``B_c``).  The two constants
``kappa`` and the exterior quadrupole constant ``A`` are fixed by requiring
``h2`` and ``K2`` to be continuous with the vacuum exterior at the surface.

The dimensionless quadrupole ``q_tilde = -Q_rot M / J^2`` is the conditional
forecast. It is spin independent within the slow-rotation approximation.
Its numerical gates use convergence and matching, not the descriptive
neutron-star band or I-Q relation. The canonical finite-pressure surface is
retained; zero-pressure extrapolation and EOS uncertainty are not claimed.
The l=0 bulk contribution ``m0(R)+J^2/R^3`` is diagnostic only: nonzero surface
density requires the additional term in Reina (2015), arXiv:1503.07835 Eq. (22).
A physical mass-change forecast remains blocked pending surface/ensemble validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

import nvg_hartle_slow_rotation as hr
import nvg_ns_frozen_forecasts as frozen_forecast
import nvg_tidal_deformability as tidal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT_PATH = HERE / "nvg_hartle_second_order_quadrupole_probe_results.json"
FIGURE_PATH = HERE / "nvg_hartle_second_order_quadrupole_probe.png"
SCHEMA_VERSION = "nvg_hartle_second_order_quadrupole_probe/2.0"
SOURCE_PATH = Path(__file__).resolve()
TEST_PATH = HERE / "test_hartle_second_order_quadrupole.py"
FROZEN_FORECAST_PATH = HERE / "nvg_ns_frozen_forecasts_p3s1_results.json"
FROZEN_FORECAST_SHA256 = "2c7abc805335c1ff1800bf21a2746f9e155fd9b3689e8ead90fa3f27a823943c"
CONVERGENCE_TOL = 1.0e-3
SPIN_TOL = 1.0e-4
MATCHING_TOL = 1.0e-8
STEP_LADDER = (0.05, 0.025, 0.0125)
RTOL_LADDER = (1.0e-9, 1.0e-10, 1.0e-11)
SEED_LADDER = (0.01, 0.005, 0.0025)
MACLAURIN_COMPACTNESSES = (0.05, 0.02, 0.008, 0.003, 0.0015)

# Frozen first-order J0737A target values from the predictive research ledger
# (claims[1] = ns_hartle_j0737a).  Pass-1 must reproduce these before any
# second-order number is trusted (fail closed).
LEDGER_J0737A = {
    "central_pressure_mev_fm3": 46.752078078168736,
    "mass_msun": 1.3381,
    "R_A_km": 12.466899931727045,
    "compactness": 0.15848675058116762,
    "I_A_g_cm2": 1.4856539538408488e+45,
    "Ibar_A": 14.303017715186515,
}
# Fail-closed relative tolerances for reproducing the frozen first-order state.
# The ledger values come from ``predict_j0737a``'s mass-interpolation over the
# canonical sequence; this probe reproduces them by a direct ``integrate_star``
# at the same central pressure, which agrees to ~5e-9 (9 digits).  1e-7 is far
# tighter than the ledger's own raw band (~1e-4) yet robust to that path gap.
REPRO_TOL = {
    "R_A_km": 1.0e-7,
    "compactness": 1.0e-7,
    "I_A_g_cm2": 1.0e-7,
    "Ibar_A": 1.0e-7,
}
# Pre-registered neutron-star dimensionless quadrupole band (Yagi & Yunes 2013:
# q_tilde ~ 2-9 for realistic EOS; -> 1 for a Kerr black hole).
QTILDE_BAND = (2.0, 9.0)

M_SUN_KM = hr.M_SUN_KM
K_CONV = tidal.k_conv          # MeV/fm^3 -> km^-2
INERTIA_CONV = hr.C_CGS ** 2 / hr.G_CGS * hr.KM3_TO_CM3  # km^3 -> g cm^2


# ── Legendre functions of the second kind (Condon-Shortley phase) ──────────
def _legendre_l2(x: float) -> dict[str, float]:
    r"""Return ``Q2^0``, ``Q2^1`` (Condon-Shortley) and ``Q2^2`` at ``x>1``.

    ``Q2^0 = (1/4)(3x^2-1)L - (3/2)x``, ``L = ln((x+1)/(x-1))``;
    ``Q2^1_cs = -sqrt(x^2-1) dQ2^0/dx``; ``Q2^2 = (x^2-1) d^2Q2^0/dx^2``.
    These closed forms are the ones symbolically verified in scratch
    sections A1/A2/A3 and used in the exterior matching (B4/B5).
    """
    x = float(x)
    if x <= 1.0:
        raise ValueError("Legendre Q2 argument must satisfy x = R/M - 1 > 1")
    L = math.log1p(2.0 / (x - 1.0))
    x2m1 = x * x - 1.0
    if x >= 10.0:
        # Convergent inverse-x series avoids cancellation in the weak-field
        # limit: Q2 = sum_{k>=1} 2k/[(2k+1)(2k+3)] x^(-2k-1).
        terms = [(2 * k, 2 * k + 1) for k in range(1, 17)]
        Q20 = math.fsum(a / (n * (n + 2)) * x ** (-n) for a, n in terms)
        dQ20 = -math.fsum(a / (n + 2) * x ** (-n - 1) for a, n in terms)
        d2Q20 = math.fsum(a * (n + 1) / (n + 2) * x ** (-n - 2) for a, n in terms)
    else:
        Q20 = 0.25 * (3.0 * x * x - 1.0) * L - 1.5 * x
        dQ20 = 1.5 * x * L - (3.0 * x * x - 1.0) / (2.0 * x2m1) - 1.5
        d2Q20 = 1.5 * L - 3.0 * x / x2m1 + 2.0 * x / (x2m1 * x2m1)
    Q21_cs = -math.sqrt(x2m1) * dQ20
    Q22 = x2m1 * d2Q20
    return {"x": x, "L": L, "Q20": Q20, "Q21_cs": Q21_cs, "Q22": Q22}


# ── Yagi-Yunes I-Q universal relation (non-authoritative cross-check) ───────
# ln(Ibar) = a + b ln(Qbar) + c (ln Qbar)^2 + d (ln Qbar)^3 + e (ln Qbar)^4
# Coefficients: Yagi & Yunes (2013) PRD 88 023009, Table II (row Ibar-Qbar) and
# Eq. (fit).  Qbar == -Q_rot M/S^2 == q_tilde (their Eq. bar-Q).  This relation
# is EOS independent only at the O(1%) level and is used strictly as a
# corroboration overlay, never as a substitute producer or authority.
YY_IQ_FIT = {"a": 1.35, "b": 0.697, "c": -0.143, "d": 9.94e-2, "e": -1.24e-2}


def yy_iq_ibar(qbar: float) -> float:
    r"""Return the universal-relation ``Ibar`` predicted for a given ``Qbar``."""
    L = math.log(float(qbar))
    lnI = (YY_IQ_FIT["a"] + YY_IQ_FIT["b"] * L + YY_IQ_FIT["c"] * L ** 2
           + YY_IQ_FIT["d"] * L ** 3 + YY_IQ_FIT["e"] * L ** 4)
    return math.exp(lnI)


# ── Background providers ────────────────────────────────────────────────────
class Background(Protocol):
    r"""Interior background fields on ``[r0, R]`` in geometric km units.

    Provides mass ``m`` [km], pressure/energy density ``p``, ``eps`` [km^-2],
    ``d eps / d P`` [dimensionless], the surface-matched ``exp(-nu)``,
    ``j^2 = exp(-(nu+lambda))``, and the normalised frame drag ``omega_tilde``
    [km^-1] with its radial derivative.
    """

    radius_km: float
    mass_km: float
    eps_c: float
    p_c: float
    dedp_c: float
    enu_inv_c: float

    def m(self, r: float) -> float: ...
    def p(self, r: float) -> float: ...
    def eps(self, r: float) -> float: ...
    def dedp(self, r: float) -> float: ...
    def enu_inv(self, r: float) -> float: ...
    def omega(self, r: float) -> float: ...
    def domega(self, r: float) -> float: ...


@dataclass
class SplineBackground:
    """Background interpolated from a frozen first-order :class:`HartleStar`."""

    star: hr.HartleStar
    eos: tidal.EOS
    radius_km: float
    mass_km: float
    eps_c: float
    p_c: float
    dedp_c: float
    enu_inv_c: float

    def __post_init__(self) -> None:
        prof = self.star.profile
        r = np.asarray(prof["radius_km"], dtype=float)
        m_msun = np.asarray(prof["mass_msun"], dtype=float)
        p_mev = np.asarray(prof["pressure_mev_fm3"], dtype=float)
        lapse = np.asarray(prof["lapse_log_gtt"], dtype=float)
        wt = np.asarray(prof["bar_omega"], dtype=float)
        dwt = np.asarray(prof["dbar_omega_dr"], dtype=float)
        self._r0 = float(r[0])
        self._R = float(r[-1])
        self._m = CubicSpline(r, m_msun * M_SUN_KM, extrapolate=True)
        self._p = CubicSpline(r, p_mev * K_CONV, extrapolate=True)
        eps_mev = np.asarray([self.eos.get_eps(float(pp)) for pp in p_mev], dtype=float)
        self._eps = CubicSpline(r, eps_mev * K_CONV, extrapolate=True)
        self._dedp = CubicSpline(
            r, np.asarray([self.eos.get_dedp(float(pp)) for pp in p_mev], dtype=float),
            extrapolate=True,
        )
        self._enu_inv = CubicSpline(r, np.exp(-lapse), extrapolate=True)
        self._wt = CubicSpline(r, wt, extrapolate=True)
        self._dwt = CubicSpline(r, dwt, extrapolate=True)

    def _clip(self, r: float) -> float:
        return float(min(max(r, self._r0), self._R))

    def m(self, r: float) -> float:
        return float(self._m(self._clip(r)))

    def p(self, r: float) -> float:
        return float(self._p(self._clip(r)))

    def eps(self, r: float) -> float:
        return float(self._eps(self._clip(r)))

    def dedp(self, r: float) -> float:
        return float(self._dedp(self._clip(r)))

    def enu_inv(self, r: float) -> float:
        return float(self._enu_inv(self._clip(r)))

    def omega(self, r: float) -> float:
        return float(self._wt(self._clip(r)))

    def domega(self, r: float) -> float:
        return float(self._dwt(self._clip(r)))


@dataclass
class ConstantDensityBackground:
    """Analytic interior-Schwarzschild background (uniform density).

    Reuses the same analytic profile as :func:`nvg_hartle_slow_rotation.constant_density_benchmark`
    so the second-order machinery is validated against an independent closed-form
    background that reduces to the Maclaurin spheroid as ``C -> 0``.  For uniform
    density ``d eps / d P = 0`` (incompressible), so the l = 0 stiffness term
    vanishes identically.
    """

    compactness: float
    radius_km: float
    omega_c: float
    domega_c: float
    omega_prof: Callable[[float], float]
    domega_prof: Callable[[float], float]
    mass_km: float
    eps_c: float
    p_c: float
    dedp_c: float = 0.0

    def __post_init__(self) -> None:
        C = float(self.compactness)
        R = float(self.radius_km)
        M = float(self.mass_km)
        self._C, self._R, self._M = C, R, M
        self._xs = math.sqrt(1.0 - 2.0 * C)
        self._lapse_half_c = 0.5 * (3.0 * self._xs - 1.0)
        self.radius_km = R
        self.eps_c = 3.0 * M / (4.0 * math.pi * R ** 3)
        self.p_c = self.eps_c * (1.0 - self._xs) / (3.0 * self._xs - 1.0)
        self.enu_inv_c = 1.0 / self._lapse_half_c ** 2

    def _x(self, r: float) -> float:
        return math.sqrt(max(1.0 - 2.0 * self._M * r * r / self._R ** 3, 1.0e-12))

    def m(self, r: float) -> float:
        return self._M * r ** 3 / self._R ** 3

    def p(self, r: float) -> float:
        x = self._x(r)
        return self.eps_c * (x - self._xs) / (3.0 * self._xs - x)

    def eps(self, r: float) -> float:
        return self.eps_c

    def dedp(self, r: float) -> float:
        return 0.0

    def enu_inv(self, r: float) -> float:
        lapse_half = 0.5 * (3.0 * self._xs - self._x(r))
        return 1.0 / (lapse_half * lapse_half)

    def omega(self, r: float) -> float:
        return float(self.omega_prof(r))

    def domega(self, r: float) -> float:
        return float(self.domega_prof(r))


def build_constant_density_background(
    *, compactness: float = 0.05, radius_km: float = 10.0,
    rtol: float = 1.0e-11, atol: float = 1.0e-13,
) -> ConstantDensityBackground:
    """Integrate the normalised frame drag on the analytic constant-density star.

    Returns a background whose ``omega``/``domega`` are the first-order
    ``bar_omega``-normalised drag (``omega_tilde(0) = 1``), matching the
    convention used for the canonical-EOS pass-1 profile.
    """
    C = float(compactness)
    R = float(radius_km)
    M = C * R
    eps = 3.0 * M / (4.0 * math.pi * R ** 3)
    xs = math.sqrt(1.0 - 2.0 * C)

    def profile(r: float) -> tuple[float, float, float]:
        x = math.sqrt(max(1.0 - 2.0 * M * r * r / R ** 3, 1.0e-12))
        pressure = eps * (x - xs) / (3.0 * xs - x)
        f = 1.0 - 2.0 * M * r * r / R ** 3
        return pressure, x, f

    r0 = 1.0e-6
    p0, _, _ = profile(0.0)
    source = eps + p0
    omega0 = 1.0 + (8.0 * math.pi / 5.0) * source * r0 ** 2
    domega0 = (16.0 * math.pi / 5.0) * source * r0

    def rhs(r: float, y: np.ndarray) -> np.ndarray:
        pressure, _, f = profile(r)
        drag = 4.0 * math.pi * r * (eps + pressure) / f
        return np.asarray([y[1], (drag - 4.0 / r) * y[1] + 4.0 * drag / r * y[0]])

    sol = solve_ivp(
        rhs, (r0, R), np.asarray([omega0, domega0]), method="DOP853",
        rtol=rtol, atol=atol, max_step=0.01, dense_output=True,
    )
    if not sol.success:
        raise RuntimeError("constant-density frame-dragging integration failed")
    omega_c = float(sol.sol(r0)[0])
    domega_c = float(sol.sol(r0)[1])
    return ConstantDensityBackground(
        compactness=C, radius_km=R, omega_c=omega_c, domega_c=domega_c,
        omega_prof=lambda r: float(sol.sol(min(max(r, r0), R))[0]),
        domega_prof=lambda r: float(sol.sol(min(max(r, r0), R))[1]),
        mass_km=M, eps_c=eps, p_c=p0,
    )



SECOND_ORDER_SEED_RADIUS_KM = 1.0e-2


def _l2_rhs(r: float, h2: float, K2: float, bg: Background) -> tuple[float, float]:
    r"""YY (2013) l = 2 interior system for ``(h2, K2)`` with algebraic ``m2``.

    Verified in scratch sections B (vacuum S/A families) and D (centre data).
    The coupled ``(dh2/dr, dK2/dr)`` pair is solved as an explicit 2x2 system.
    """
    m = bg.m(r)
    p = bg.p(r)
    eps = bg.eps(r)
    enu_inv = bg.enu_inv(r)
    wt = bg.omega(r)
    dwt = bg.domega(r)
    f = 1.0 - 2.0 * m / r
    e_lam = 1.0 / f
    e_2lam = e_lam * e_lam
    j2 = enu_inv * f
    ehp = eps + p
    m2 = -r * f * h2 + (r ** 4 / 6.0) * j2 * (r * f * dwt * dwt + 16.0 * math.pi * r * ehp * wt * wt)
    P = (r - 3.0 * m - 4.0 * math.pi * p * r ** 3) / r ** 2 * e_lam
    Qm = (r - m + 4.0 * math.pi * p * r ** 3) / r ** 3 * e_2lam
    beta = (r - m + 4.0 * math.pi * p * r ** 3) / r * e_lam
    Y = (
        (3.0 - 4.0 * math.pi * ehp * r * r) / r * e_lam * h2
        + 2.0 / r * e_lam * K2
        + (1.0 + 8.0 * math.pi * p * r * r) / r ** 2 * e_2lam * m2
        + r ** 3 / 12.0 * enu_inv * dwt * dwt
        - (4.0 * math.pi / 3.0) * ehp * r ** 3 * wt * wt * enu_inv * e_lam
    )
    Z = P * h2 + Qm * m2
    det = beta - 1.0
    dh2 = (beta * Z - Y) / det
    dK2 = (Y - Z) / det
    return dh2, dK2


def _l0_rhs(r: float, m0: float, p0: float, bg: Background) -> tuple[float, float]:
    r"""UMS (2013) l = 0 interior system for ``(m0, p0)``.

    Verified in scratch section C (centre coefficients C2/C3 and the C4 lapse
    identity used for ``(j^2)'``).  ``d eps / d P`` is supplied by the EOS.
    """
    m = bg.m(r)
    p = bg.p(r)
    eps = bg.eps(r)
    dedp = bg.dedp(r)
    enu_inv = bg.enu_inv(r)
    wt = bg.omega(r)
    dwt = bg.domega(r)
    f = 1.0 - 2.0 * m / r
    j2 = enu_inv * f
    ehp = eps + p
    r2m = r - 2.0 * m
    # (j^2)' = -j^2 (nu' + lambda') with nu'+lambda' = 8 pi r^2 (eps+p)/(r-2m)
    dj2 = -j2 * 8.0 * math.pi * r * r * ehp / r2m
    dm0 = (
        4.0 * math.pi * r * r * ehp * dedp * p0
        + (r ** 4 / 12.0) * j2 * dwt * dwt
        - (r ** 3 / 3.0) * wt * wt * dj2
    )
    # (1/3) d/dr [ r^3 j^2 wt^2 / (r-2m) ] expanded analytically.
    N = r ** 3 * j2 * wt * wt
    dN = 3.0 * r * r * j2 * wt * wt + r ** 3 * dj2 * wt * wt + r ** 3 * j2 * 2.0 * wt * dwt
    dD = 1.0 - 8.0 * math.pi * r * r * eps
    dterm = (dN * r2m - N * dD) / (r2m * r2m)
    dp0 = (
        -m0 * (1.0 + 8.0 * math.pi * r * r * p) / (r2m * r2m)
        - 4.0 * math.pi * ehp * r * r / r2m * p0
        + (r ** 4 / 12.0) * j2 / r2m * dwt * dwt
        + dterm / 3.0
    )
    return dm0, dp0


@dataclass
class SecondOrderSolution:
    """Second-order spin observables in geometric km units (seed normalisation)."""

    radius_km: float
    mass_km: float
    J_seed_km2: float          # angular momentum of the omega_tilde seed (=I*Omega_seed)
    Omega_seed_km1: float      # omega_infinity of the seed (bar_omega(0)=1)
    A_seed_km: float           # legacy field name: A is dimensionless, not km
    kappa: float               # complementary weight applied to the particular
    Q_rot_seed_km3: float      # YY Q^rot = -S^2/M - (8/5) A M^3
    q_tilde: float             # -Q_rot M / J^2 (spin independent)
    deltaM_seed_km: float      # m0(R) + J^2/R^3
    deltaM_over_MR2Om2: float  # bulk-only dimensionless contribution -> 2/5
    Q_over_Om2R5: float        # dimensionless -> -1/2 (Newtonian)
    I_over_MR2: float          # first-order dimensionless inertia -> 2/5 (Newtonian)
    surface_state: dict[str, float]


def solve_second_order(bg: Background, *, J_seed: float, Omega_seed: float,
                       I_seed: float, rtol: float = 1.0e-10, atol: float = 1.0e-12,
                       max_step: float = 0.05, r0: float = SECOND_ORDER_SEED_RADIUS_KM) -> SecondOrderSolution:
    r"""Integrate l = 0 and l = 2 to the surface and match to the vacuum exterior."""
    R = float(bg.radius_km)
    M = float(bg.mass_km)
    if not (0.0 < r0 < R):
        raise ValueError("second-order seed radius must satisfy 0 < r0 < R")
    if not (0.0 < 2.0 * M < R):
        raise ValueError("second-order background must satisfy 0 < 2M < R")
    for name, value in {"J_seed": J_seed, "Omega_seed": Omega_seed,
                        "I_seed": I_seed, "rtol": rtol, "atol": atol,
                        "max_step": max_step}.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    j2c = float(bg.enu_inv_c)          # j^2 at centre = e^{-nu_c} (lambda_c = 0)
    wt_c = float(bg.omega(r0))
    dedp_c = float(bg.dedp_c)
    eps_c = float(bg.eps_c)
    p_c = float(bg.p_c)

    # l = 0 centre data (UMS C2/C3).
    p0_0 = (1.0 / 3.0) * j2c * wt_c ** 2 * r0 ** 2
    m0_0 = (4.0 * math.pi / 15.0) * (eps_c + p_c) * (dedp_c + 2.0) * j2c * wt_c ** 2 * r0 ** 5

    def rhs(r: float, y: np.ndarray) -> np.ndarray:
        dm0, dp0 = _l0_rhs(r, y[0], y[1], bg)
        dhp, dKp = _l2_rhs(r, y[2], y[3], bg)          # particular (sources on)
        dmc = _l2_rhs(r, y[4], y[5], _HomogeneousView(bg))  # complementary
        return np.asarray([dm0, dp0, dhp, dKp, dmc[0], dmc[1]])

    # l = 2 centre data: h2 = B r^2, K2 = -B r^2 with test amplitudes B = 1.
    y0 = np.asarray([m0_0, p0_0, r0 ** 2, -r0 ** 2, r0 ** 2, -r0 ** 2])
    sol = solve_ivp(
        rhs, (r0, R), y0, method="DOP853", rtol=rtol, atol=atol, max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"second-order integration failed: {sol.message}")
    m0_R, p0_R, h2p_R, K2p_R, h2c_R, K2c_R = (float(v) for v in sol.y[:, -1])

    # Exterior matching for (kappa, A).
    x = R / M - 1.0
    leg = _legendre_l2(x)
    Q22 = leg["Q22"]
    Gk = 2.0 * M / math.sqrt(R * (R - 2.0 * M)) * leg["Q21_cs"] - Q22
    S = float(J_seed)
    rhs1 = (1.0 + M / R) * S * S / (M * R ** 3) - h2p_R
    rhs2 = -(1.0 + 2.0 * M / R) * S * S / (M * R ** 3) - K2p_R
    det = -h2c_R * Gk + Q22 * K2c_R
    if abs(det) < 1.0e-300:
        raise RuntimeError("second-order surface matching matrix is singular")
    kappa = (-Gk * rhs1 + Q22 * rhs2) / det
    A = (h2c_R * rhs2 - rhs1 * K2c_R) / det

    Q_rot = -S * S / M - (8.0 / 5.0) * A * M ** 3
    q_tilde = -Q_rot * M / (S * S)
    deltaM = m0_R + S * S / R ** 3
    Om2 = Omega_seed ** 2
    h_int, k_int = h2p_R + kappa * h2c_R, K2p_R + kappa * K2c_R
    h_ext = (1.0 + M / R) * S * S / (M * R ** 3) + A * Q22
    k_ext = -(1.0 + 2.0 * M / R) * S * S / (M * R ** 3) + A * Gk
    h_residual = abs(h_int - h_ext) / max(abs(h_int), abs(h_ext), 1e-30)
    k_residual = abs(k_int - k_ext) / max(abs(k_int), abs(k_ext), 1e-30)
    condition = float(np.linalg.cond([[h2c_R, -Q22], [K2c_R, -Gk]]))
    # Reina (2015), arXiv:1503.07835, Eq. (22). Only a zero-pressure
    # surface permits this interpretation; the canonical star stops at Ps>0.
    surface_deltaM = 4.0 * math.pi * R ** 3 / M * (R - 2.0 * M) * bg.eps(R) * p0_R
    return SecondOrderSolution(
        radius_km=R, mass_km=M, J_seed_km2=S, Omega_seed_km1=Omega_seed,
        A_seed_km=A, kappa=kappa, Q_rot_seed_km3=Q_rot, q_tilde=q_tilde,
        deltaM_seed_km=deltaM,
        deltaM_over_MR2Om2=deltaM / (M * R * R * Om2),
        Q_over_Om2R5=Q_rot / (Om2 * R ** 5),
        I_over_MR2=I_seed / (M * R * R),
        surface_state={
            "m0_R": m0_R, "p0_R": p0_R,
            "h2_particular_R": h2p_R, "K2_particular_R": K2p_R,
            "h2_complementary_R": h2c_R, "K2_complementary_R": K2c_R,
            "legendre_Q22": Q22, "legendre_Gk": Gk, "matching_det": det,
            "matching_condition_number": condition,
            "h2_interior": h_int, "h2_exterior": h_ext,
            "K2_interior": k_int, "K2_exterior": k_ext,
            "h2_relative_residual": h_residual, "K2_relative_residual": k_residual,
            "surface_pressure_geom": bg.p(R), "surface_density_geom": bg.eps(R),
            "deltaM_bulk_seed_km": deltaM,
            "deltaM_surface_candidate_seed_km": surface_deltaM,
            "deltaM_total_candidate_seed_km": deltaM + surface_deltaM,
        },
    )


class _HomogeneousView:
    """Background wrapper that zeroes the frame-drag sources (complementary l=2)."""

    __slots__ = ("_bg",)

    def __init__(self, bg: Background) -> None:
        self._bg = bg

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bg, name)

    def omega(self, r: float) -> float:
        return 0.0

    def domega(self, r: float) -> float:
        return 0.0



def background_from_star(star: hr.HartleStar, eos: tidal.EOS) -> SplineBackground:
    """Wrap a frozen first-order star into a spline second-order background."""
    prof = star.profile
    Pc = float(star.central_pressure)
    lapse = np.asarray(prof["lapse_log_gtt"], dtype=float)
    return SplineBackground(
        star=star, eos=eos,
        radius_km=float(star.radius_km),
        mass_km=float(star.mass_msun) * M_SUN_KM,
        eps_c=float(eos.get_eps(Pc)) * K_CONV,
        p_c=Pc * K_CONV,
        dedp_c=float(eos.get_dedp(Pc)),
        enu_inv_c=float(np.exp(-lapse[0])),
    )


def _seed_angular_from_background(bg: Background) -> tuple[float, float, float]:
    """Return ``(Omega_seed, I_seed, J_seed)`` from the surface drag, as in pass 1."""
    R = float(bg.radius_km)
    om_R = float(bg.omega(R))
    dom_R = float(bg.domega(R))
    Omega_seed = om_R + R * dom_R / 3.0
    I_seed = R ** 4 * dom_R / (6.0 * Omega_seed)
    return Omega_seed, I_seed, I_seed * Omega_seed


def _assert_close(label: str, got: float, want: float, rtol: float) -> None:
    denom = max(abs(want), 1.0e-300)
    if not math.isfinite(got) or not math.isfinite(want) or abs(got - want) > rtol * denom:
        raise AssertionError(
            f"fail-closed first-order drift in {label}: got {got!r}, ledger {want!r}, rtol {rtol}"
        )


def constant_density_second_order(
    *, compactness: float, radius_km: float = 10.0, r0: float = SECOND_ORDER_SEED_RADIUS_KM,
    rtol: float = 1.0e-10, atol: float = 1.0e-12,
) -> SecondOrderSolution:
    """Full second-order solve on the analytic uniform-density background."""
    bg = build_constant_density_background(compactness=compactness, radius_km=radius_km)
    Omega_seed, I_seed, J_seed = _seed_angular_from_background(bg)
    return solve_second_order(
        bg, J_seed=J_seed, Omega_seed=Omega_seed, I_seed=I_seed, rtol=rtol, atol=atol, r0=r0,
    )


def _richardson_c0(y1: float, c1: float, y2: float, c2: float) -> float:
    r"""Extrapolate ``Y(C) = Y0 + a C + O(C^2)`` to ``C -> 0`` from two points."""
    return (y1 * c2 - y2 * c1) / (c2 - c1)


def maclaurin_validation(*, radius_km: float = 10.0) -> dict[str, Any]:
    r"""Validate the second-order machinery against the Newtonian Maclaurin limit.

    As ``C -> 0`` the uniform-density relativistic star reduces to a Maclaurin
    spheroid.  The verified Newtonian targets are:

    * ``q_tilde -> (25/8) R/M``                (scratch E1)
    * ``Q_rot -> -(1/2) Omega^2 R^5``          (scratch E2)
    * ``I -> (2/5) M R^2``                     (first order; weak-field series)
    * ``delta_M -> (2/5) M R^2 Omega^2``       (l = 0; see note below)

    The ``2/5`` mass target validates ONLY the bulk integral ``m0+J^2/R^3``.
    Reina (2015), Eq. (22), requires an additional surface term when eps(R)>0.
    The pressure-preserving sequence used here has p0(0)=0; it is neither a
    fixed-baryon-number sequence nor, for incompressible matter, uniquely
    specified by fixed density alone. No total-mass Newtonian target is asserted.

    Every ratio converges only linearly in ``C``, so the ``C -> 0`` limit is
    obtained by Richardson extrapolation on the two smallest compactnesses; the
    gate is applied to the extrapolated limits and to a seed-radius refinement.
    """
    compactnesses = MACLAURIN_COMPACTNESSES
    rows: list[dict[str, Any]] = []
    for C in compactnesses:
        sol = constant_density_second_order(compactness=C, radius_km=radius_km)
        sol_fine = constant_density_second_order(
            compactness=C, radius_km=radius_km, r0=SECOND_ORDER_SEED_RADIUS_KM / 2.0,
        )
        target_q = 25.0 / 8.0 * sol.radius_km / sol.mass_km
        rows.append({
            "compactness": C,
            "R_over_M": sol.radius_km / sol.mass_km,
            "q_tilde": sol.q_tilde,
            "q_tilde_newtonian_target": target_q,
            "q_tilde_ratio": sol.q_tilde / target_q,
            "q_tilde_seed_radius_drift": abs(sol.q_tilde - sol_fine.q_tilde) / max(abs(sol.q_tilde), 1e-30),
            "Q_over_Om2R5": sol.Q_over_Om2R5,
            "Q_over_Om2R5_target": -0.5,
            "deltaM_over_MR2Om2": sol.deltaM_over_MR2Om2,
            "deltaM_over_MR2Om2_target": 0.4,
            "mass_correction_scope": "bulk_only_not_total_mass",
            "deltaM_surface_over_MR2Om2": sol.surface_state["deltaM_surface_candidate_seed_km"] / (
                sol.mass_km * sol.radius_km ** 2 * sol.Omega_seed_km1 ** 2),
            "deltaM_total_over_MR2Om2": sol.surface_state["deltaM_total_candidate_seed_km"] / (
                sol.mass_km * sol.radius_km ** 2 * sol.Omega_seed_km1 ** 2),
            "I_over_MR2": sol.I_over_MR2,
            "I_over_MR2_target": 0.4,
        })
    # Richardson extrapolation to C -> 0 from the two smallest compactnesses.
    c1, c2 = rows[-2]["compactness"], rows[-1]["compactness"]

    def _ext(key: str) -> float:
        return _richardson_c0(rows[-2][key], c1, rows[-1][key], c2)

    limits = {
        "q_tilde_ratio_C0": _ext("q_tilde_ratio"),
        "Q_over_Om2R5_C0": _ext("Q_over_Om2R5"),
        "deltaM_over_MR2Om2_C0": _ext("deltaM_over_MR2Om2"),
        "I_over_MR2_C0": _ext("I_over_MR2"),
    }
    tol = {"q_tilde_ratio": 2.0e-3, "Q_over_Om2R5": 2.0e-3,
           "deltaM_over_MR2Om2": 2.0e-3, "I_over_MR2": 2.0e-3}
    checks = {
        "q_tilde_ratio_to_1": abs(limits["q_tilde_ratio_C0"] - 1.0) < tol["q_tilde_ratio"],
        "Q_ratio_to_minus_half": abs(limits["Q_over_Om2R5_C0"] + 0.5) < tol["Q_over_Om2R5"],
        "deltaM_ratio_to_two_fifths": abs(limits["deltaM_over_MR2Om2_C0"] - 0.4) < tol["deltaM_over_MR2Om2"],
        "I_ratio_to_two_fifths": abs(limits["I_over_MR2_C0"] - 0.4) < tol["I_over_MR2"],
        "seed_radius_stable": all(r["q_tilde_seed_radius_drift"] < 1.0e-4 for r in rows),
    }
    previous_q = _richardson_c0(rows[-3]["q_tilde_ratio"], rows[-3]["compactness"],
                                rows[-2]["q_tilde_ratio"], rows[-2]["compactness"])
    checks["successive_extrapolations"] = (
        abs(previous_q - 1.0) < 0.002
        and abs(limits["q_tilde_ratio_C0"] - previous_q) < 0.002
    )
    # l=0 is a separate diagnostic and never authorizes the l=2 observable.
    quadrupole_checks = {k: v for k, v in checks.items() if k != "deltaM_ratio_to_two_fifths"}
    passed = all(quadrupole_checks.values())
    return {
        "status": "PASS_MACLAURIN_NEWTONIAN_LIMIT" if passed else "BLOCKED_MACLAURIN_LIMIT",
        "quadrupole_checks": quadrupole_checks,
        "previous_q_tilde_ratio_C0": previous_q,
        "extrapolation_difference": abs(limits["q_tilde_ratio_C0"] - previous_q),
        "mass_correction_scope": "bulk_only_not_total_mass",
        "surface_term_source": "https://arxiv.org/html/1503.07835v1 Eq. (22)",
        "radius_km": radius_km,
        "richardson_limits_C0": limits,
        "checks": checks,
        "tolerances": tol,
        "rows": rows,
        "provenance": (
            "uniform-density analytic interior-Schwarzschild background; q_tilde/Q "
            "targets from Maclaurin E1/E2; 2/5 validates only the l=0 bulk integral. "
            "The nonzero-density surface term is separately reported at fixed "
            "central pressure; no fixed-baryon-number or total-mass limit is claimed."
        ),
    }


class _SpinScaledBackground(_HomogeneousView):
    """Change only the arbitrary first-order spin normalization."""

    __slots__ = ("_scale",)

    def __init__(self, bg: Background, scale: float):
        super().__init__(bg)
        self._scale = scale

    def omega(self, r: float) -> float:
        return self._scale * self._bg.omega(r)

    def domega(self, r: float) -> float:
        return self._scale * self._bg.domega(r)


def _solve_star(star: hr.HartleStar, eos: tidal.EOS, **controls: float) -> SecondOrderSolution:
    return solve_second_order(
        background_from_star(star, eos), J_seed=star.angular_momentum_geom_km2,
        Omega_seed=star.omega_infinity, I_seed=star.inertia_geom_km3, **controls,
    )


def _solution_row(sol: SecondOrderSolution, control: float) -> dict[str, float]:
    row = {
        "control": control, "q_tilde": sol.q_tilde, "Q_rot_seed_km3": sol.Q_rot_seed_km3,
        "h2_matching_residual": sol.surface_state["h2_relative_residual"],
        "K2_matching_residual": sol.surface_state["K2_relative_residual"],
        "matching_condition_number": sol.surface_state["matching_condition_number"],
    }
    if not all(math.isfinite(v) for v in row.values()):
        raise AssertionError("nonfinite second-order convergence result")
    return row


def quadrupole_convergence(star: hr.HartleStar, eos: tidal.EOS,
                           baseline: SecondOrderSolution) -> dict[str, Any]:
    """Independent refinement axes; no empirical-fit or EOS-band gate."""
    ladders: dict[str, list[dict[str, float]]] = {}
    for axis, values in (("max_step_km", STEP_LADDER), ("rtol", RTOL_LADDER),
                         ("seed_radius_km", SEED_LADDER)):
        rows = []
        for value in values:
            controls = ({"max_step": value} if axis == "max_step_km" else
                        {"rtol": value, "atol": value / 100.0} if axis == "rtol" else
                        {"r0": value})
            rows.append(_solution_row(_solve_star(star, eos, **controls), value))
        ladders[axis] = rows
    # Background refinements are independent of the second-order controls.
    for axis, values in (("background_max_step_km", STEP_LADDER),
                         ("background_rtol", RTOL_LADDER)):
        rows = []
        for value in values:
            controls = ({"max_step": value} if axis == "background_max_step_km" else
                        {"rtol": value, "atol": value / 100.0})
            refined = hr.integrate_star(star.central_pressure, eos=eos, **controls)
            row = _solution_row(_solve_star(refined, eos), value)
            row.update({"background_points": len(refined.profile["radius_km"]),
                        "R_km": refined.radius_km, "Ibar": refined.inertia_bar})
            rows.append(row)
        ladders[axis] = rows
    differences = {
        axis: abs(rows[-1]["q_tilde"] - rows[-2]["q_tilde"]) / abs(rows[-1]["q_tilde"])
        for axis, rows in ladders.items()
    }
    bg = background_from_star(star, eos)
    spin_rows = []
    for scale in (0.5, 2.0):
        scaled = solve_second_order(
            _SpinScaledBackground(bg, scale),
            J_seed=star.angular_momentum_geom_km2 * scale,
            Omega_seed=star.omega_infinity * scale, I_seed=star.inertia_geom_km3,
        )
        row = _solution_row(scaled, scale)
        row["q_relative_drift"] = abs(scaled.q_tilde / baseline.q_tilde - 1.0)
        row["Q_quadratic_relative_drift"] = abs(
            scaled.Q_rot_seed_km3 / (baseline.Q_rot_seed_km3 * scale ** 2) - 1.0)
        spin_rows.append(row)
    all_rows = [r for rows in ladders.values() for r in rows] + spin_rows
    checks = {axis: drift < CONVERGENCE_TOL for axis, drift in differences.items()}
    checks["spin_normalization"] = all(
        max(r["q_relative_drift"], r["Q_quadratic_relative_drift"]) < SPIN_TOL for r in spin_rows)
    checks["surface_matching"] = all(
        max(r["h2_matching_residual"], r["K2_matching_residual"]) < MATCHING_TOL for r in all_rows)
    q_values = [r["q_tilde"] for rows in ladders.values() for r in rows]
    return {
        "status": "PASS_NUMERICAL_CONVERGENCE" if all(checks.values()) else "BLOCKED_NUMERICAL_CONVERGENCE",
        "checks": checks, "ladders": ladders, "finest_pair_relative_differences": differences,
        "spin_rescaling": spin_rows,
        "q_tilde_numerical_range": [min(q_values), max(q_values)],
        "tolerances": {"refinement": CONVERGENCE_TOL, "spin": SPIN_TOL, "matching": MATCHING_TOL},
        "interpretation": "Numerical sensitivity at fixed central pressure and frozen finite-pressure surface; not a confidence interval or EOS uncertainty.",
    }


def frozen_target() -> dict[str, float]:
    """Authenticate the immutable forecast directly, avoiding a ledger cycle."""
    if hr._sha256(FROZEN_FORECAST_PATH) != FROZEN_FORECAST_SHA256:
        raise AssertionError("frozen first-order forecast digest drift")
    payload = json.loads(FROZEN_FORECAST_PATH.read_text(encoding="utf-8"))
    frozen_forecast.assert_artifact_provenance(payload)
    j = payload["j0737a"]
    row = j["forecast"]
    target = {"mass_msun": j["mass_msun"], "central_pressure_mev_fm3": row["central_pressure"],
              "R_A_km": row["R_km"], "compactness": row["compactness"],
              "I_A_g_cm2": row["I"], "Ibar_A": row["Ibar"]}
    if target != LEDGER_J0737A or j["branch_id"] != 1 or j["gap_crossed"] is not False:
        raise AssertionError("frozen first-order target/branch drift")
    return target


def j0737a_second_order(*, eos: tidal.EOS | None = None) -> dict[str, Any]:
    r"""Fail-closed second-order quadrupole forecast for PSR J0737-3039A.

    Pass 1 must reproduce the frozen first-order ledger state (radius,
    compactness, ``I_A``, ``Ibar_A``) before any second-order number is emitted.
    ``q_tilde`` is spin independent and is checked against the pre-registered
    neutron-star band ``[2, 9]`` and the UMS fit ``3.64 x - 5.3`` (``x = R/2M``).
    """
    eos = tidal.EOS() if eos is None else eos
    Pc = frozen_target()["central_pressure_mev_fm3"]
    star = hr.integrate_star(Pc, eos=eos)
    # mass_msun is the rounded timing *input* (1.3381); the reproduced value
    # differs only by input rounding, so it is a loose sanity check.  The
    # derived first-order outputs below are the tight fail-closed targets.
    _assert_close("mass_msun", star.mass_msun, LEDGER_J0737A["mass_msun"], 1.0e-5)
    _assert_close("R_A_km", star.radius_km, LEDGER_J0737A["R_A_km"], REPRO_TOL["R_A_km"])
    _assert_close("compactness", star.compactness, LEDGER_J0737A["compactness"], REPRO_TOL["compactness"])
    _assert_close("I_A_g_cm2", star.inertia_cgs, LEDGER_J0737A["I_A_g_cm2"], REPRO_TOL["I_A_g_cm2"])
    _assert_close("Ibar_A", star.inertia_bar, LEDGER_J0737A["Ibar_A"], REPRO_TOL["Ibar_A"])

    bg = background_from_star(star, eos)
    sol = solve_second_order(
        bg, J_seed=float(star.angular_momentum_geom_km2),
        Omega_seed=float(star.omega_infinity), I_seed=float(star.inertia_geom_km3),
    )
    sol_fine = solve_second_order(
        bg, J_seed=float(star.angular_momentum_geom_km2),
        Omega_seed=float(star.omega_infinity), I_seed=float(star.inertia_geom_km3),
        r0=SECOND_ORDER_SEED_RADIUS_KM / 2.0,
    )
    convergence = quadrupole_convergence(star, eos, sol)
    q = float(sol.q_tilde)
    x_half = float(star.radius_km) / (2.0 * float(star.mass_msun) * M_SUN_KM)
    ums_fit = 3.64 * x_half - 5.3
    in_band = QTILDE_BAND[0] <= q <= QTILDE_BAND[1]
    seed_drift = abs(q - float(sol_fine.q_tilde)) / max(abs(q), 1e-30)
    # Independent cross-check: the frozen first-order Ibar and the second-order
    # Qbar (=q_tilde) must satisfy the universal I-Q relation to O(1%).
    ibar = float(star.inertia_bar)
    ibar_iq_fit = yy_iq_ibar(q)
    iq_rel_dev = abs(ibar - ibar_iq_fit) / abs(ibar_iq_fit)
    iloveq_crosscheck = {
        "status": "PASS_IQ_UNIVERSAL_1PCT" if iq_rel_dev < 0.01 else "REVIEW_IQ_UNIVERSALITY",
        "acceptance_gate": False,
        "Ibar_first_order": ibar,
        "Qbar_second_order": q,
        "Ibar_from_universal_IQ_fit": ibar_iq_fit,
        "relative_deviation": iq_rel_dev,
        "authority": "non-authoritative corroboration overlay (Yagi-Yunes 2013 Eq. fit, Table II)",
    }
    return {
        "status": "derived_conditional_second_order" if all(convergence["checks"].values()) else "BLOCKED_SECOND_ORDER",
        "convergence": convergence,
        "branch_id": 1,
        "gap_crossed": False,
        "mass_correction": {
            "status": "BLOCKED_SURFACE_AND_ENSEMBLE_VALIDATION",
            "reason": "Canonical star terminates at finite pressure; zero-pressure surface matching and a physical sequence are not validated.",
            "sequence": "p0(0)=0; fixed central pressure, not fixed baryon number",
            "bulk_seed_km": sol.deltaM_seed_km,
            "surface_candidate_seed_km": sol.surface_state["deltaM_surface_candidate_seed_km"],
            "source": "https://arxiv.org/html/1503.07835v1 Eq. (22)",
        },
        "first_order_reproduced": {
            "mass_msun": star.mass_msun, "R_A_km": star.radius_km,
            "compactness": star.compactness, "I_A_g_cm2": star.inertia_cgs,
            "Ibar_A": star.inertia_bar, "tolerances": REPRO_TOL,
        },
        "q_tilde": q,
        "q_tilde_band": list(QTILDE_BAND),
        "q_tilde_in_band": bool(in_band),
        "q_tilde_band_acceptance_gate": False,
        "q_tilde_seed_radius_drift": seed_drift,
        "x_R_over_2M": x_half,
        "ums_fit_3p64x_minus_5p3": ums_fit,
        "rel_dev_from_ums_fit": abs(q - ums_fit) / abs(ums_fit),
        "iloveq_crosscheck": iloveq_crosscheck,
        "Q_rot_geom_km3_seed": sol.Q_rot_seed_km3,
        "A_seed_km": sol.A_seed_km,
        "deltaM_geom_km_seed": sol.deltaM_seed_km,
        "deltaM_over_MR2Om2": sol.deltaM_over_MR2Om2,
        "kappa": sol.kappa,
        "surface_state": sol.surface_state,
        "interpretation": (
            "q_tilde = -Q_rot M / J^2 = Qbar (Yagi-Yunes Eq. bar-Q) is independent "
            "of the spin rate; it is a pure structure forecast for the frozen "
            "canonical EOS, not an empirical measurement (no direct quadrupole "
            "measurement is used). The finite-pressure surface convention is frozen. "
            "deltaM_geom_km_seed and deltaM_over_MR2Om2 are bulk-only diagnostics, "
            "not total physical mass changes. A_seed_km is a legacy key for dimensionless A."
        ),
    }


def source_provenance() -> dict[str, Any]:
    return {
        "producer": "verification/nvg_hartle_second_order_quadrupole_probe.py",
        "equation_sources": {
            "l2_interior_and_exterior": "Yagi & Yunes (2013) PRD 88 023009 arXiv:1303.1528",
            "l0_interior_and_centre": "Urbanec, Miller & Stuchlik (2013) MNRAS 433 1906 arXiv:1301.5925",
            "symbolic_verification": "scratch/hartle_yy_verify.py (A/B/C/D/E, 24 checks)",
            "newtonian_target": "Maclaurin spheroid (scratch E1/E2)",
        },
        "producer_sha256": hr._sha256(SOURCE_PATH),
        "test_sha256": hr._sha256(TEST_PATH),
        "first_order_dependency": hr.source_provenance(),
        "frozen_forecast_dependencies": frozen_forecast.source_provenance(),
        "eos_dependency": {
            "producer": "verification/nvg_tidal_deformability.py",
            "source_sha256": hr._sha256(hr.HERE / "nvg_tidal_deformability.py"),
        },
        "frozen_target": {
            "path": str(FROZEN_FORECAST_PATH.relative_to(ROOT)),
            "source_sha256": hr._sha256(FROZEN_FORECAST_PATH),
            "values": frozen_target(),
        },
        "configuration": {
            "second_order_seed_radius_km": SECOND_ORDER_SEED_RADIUS_KM,
            "step_ladder_km": list(STEP_LADDER), "rtol_ladder": list(RTOL_LADDER),
            "seed_ladder_km": list(SEED_LADDER),
            "maclaurin_compactnesses": list(MACLAURIN_COMPACTNESSES),
            "refinement_tolerance": CONVERGENCE_TOL, "spin_tolerance": SPIN_TOL,
            "matching_tolerance": MATCHING_TOL,
            "surface_convention": "frozen finite-pressure surface; no zero-pressure extrapolation claimed",
            "acceptance_uses_empirical_fit": False,
            "ode_method": "DOP853",
            "convention": "g_tt=-exp(nu) single exponent; j^2=exp(-(nu+lambda)); omega_tilde=Omega-omega",
            "ums_v2_exterior_sign_correction": "-J^2/r^4 (their printed +J^2/r^4 is inconsistent with their own dv2/dr)",
        },
    }


def build_payload(*, quick: bool = False) -> dict[str, Any]:
    eos = tidal.EOS()
    maclaurin = maclaurin_validation()
    j0737 = j0737a_second_order(eos=eos)
    derived = maclaurin["status"].startswith("PASS") and j0737["status"].startswith("derived")
    classification = "derived_conditional_second_order" if derived else "blocked"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "audit": "hartle-second-order-quadrupole",
        "artifacts": {
            "result_json": str(RESULT_PATH.relative_to(hr.ROOT)),
            "figure_png": str(FIGURE_PATH.relative_to(hr.ROOT)),
        },
        "public_status": (
            "DERIVED_CONDITIONAL_SECOND_ORDER"
            if (maclaurin["status"].startswith("PASS") and j0737["status"].startswith("derived"))
            else "BLOCKED_SECOND_ORDER"
        ),
        "blocker_resolution": {
            "previous_status": "BLOCKED",
            "previous_reason": hr.MODEL_IDENTITY["quadrupole_reason"],
            "resolved_by": "independent YY(l=2)+UMS(l=0) second-order producer, symbolically verified",
            "substitute_used": False,
        },
        "source_provenance": source_provenance(),
        "maclaurin_validation": maclaurin,
        "j0737a": j0737,
        "classification": {
            "quadrupole_Q": classification,
            "q_tilde": classification,
            "mass_correction": "blocked",
            "empirical_confirmation": "not_claimed",
            "independent_evidence_weight": 0.0,
        },
    }
    _reject_nonfinite(payload)
    payload = hr._jsonable(payload)
    payload["payload_integrity"] = {
        "algorithm": "sha256", "covered": "complete_payload_except_payload_integrity",
        "sha256": payload_digest(payload),
    }
    return payload


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_nonfinite(item)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise AssertionError("nonfinite numerical payload")


def payload_digest(payload: dict[str, Any]) -> str:
    _reject_nonfinite(payload)
    unsigned = {k: v for k, v in payload.items() if k != "payload_integrity"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _assert_live_equal(actual: Any, expected: Any, path: str = "payload") -> None:
    """Full-payload recomputation, with explicit float portability tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: key/schema drift")
        for key in expected:
            if key != "payload_integrity":
                _assert_live_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: row count drift")
        for i, (a, b) in enumerate(zip(actual, expected)):
            _assert_live_equal(a, b, f"{path}[{i}]")
    elif isinstance(expected, float):
        if (isinstance(actual, bool) or not isinstance(actual, (float, int))
                or not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10)):
            raise AssertionError(f"{path}: live numerical mismatch")
    elif type(actual) is not type(expected) or actual != expected:
        raise AssertionError(f"{path}: live semantic mismatch")


@lru_cache(maxsize=2)
def _live_validation_payload(provenance_json: str) -> dict[str, Any]:
    # In-process reuse only; any dependency/configuration hash change changes
    # this key. No stored artifact or caller-supplied value populates the cache.
    payload = build_payload()
    if json.dumps(payload["source_provenance"], sort_keys=True) != provenance_json:
        raise AssertionError("dependencies changed during live computation")
    return payload


def assert_artifact_provenance(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("second-order quadrupole schema drift")
    expected_integrity = {"algorithm": "sha256", "covered": "complete_payload_except_payload_integrity",
                          "sha256": payload_digest(payload)}
    if payload.get("payload_integrity") != expected_integrity:
        raise AssertionError("second-order quadrupole payload digest mismatch")
    provenance = source_provenance()
    if payload.get("source_provenance") != provenance:
        raise AssertionError("second-order quadrupole provenance drift")
    if payload.get("public_status") != "DERIVED_CONDITIONAL_SECOND_ORDER":
        raise AssertionError("second-order quadrupole gates are not PASS")
    if not all(payload.get("j0737a", {}).get("convergence", {}).get("checks", {}).values()):
        raise AssertionError("numerical convergence checks failed")
    live = _live_validation_payload(json.dumps(provenance, sort_keys=True))
    _assert_live_equal(payload, live)


def _write_figure(payload: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["maclaurin_validation"]["rows"]
    inv_R_over_M = [1.0 / r["R_over_M"] for r in rows]
    q_ratio = [r["q_tilde_ratio"] for r in rows]
    j0737 = payload["j0737a"]
    plt.rcParams.update({"font.family": "serif", "axes.linewidth": 1.1})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].plot(inv_R_over_M, q_ratio, "o-", color="#2c3e50", lw=1.8, ms=5)
    axes[0].axhline(1.0, color="#c0392b", ls="--", lw=1.2, label="Newtonian $25R/8M$")
    axes[0].set_xlabel(r"Compactness $C=M/R$")
    axes[0].set_ylabel(r"$\tilde q\,/\,(25R/8M)$")
    axes[0].set_title("Maclaurin limit of the second-order quadrupole")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].axhspan(*j0737["q_tilde_band"], color="#dfe6e9", label="pre-registered band [2, 9]")
    axes[1].scatter([j0737["x_R_over_2M"]], [j0737["q_tilde"]], marker="D", s=70,
                    color="#c0392b", zorder=4, label="J0737A forecast")
    xs = np.linspace(2.4, 4.0, 50)
    axes[1].plot(xs, 3.64 * xs - 5.3, color="#2980b9", lw=1.4, label="UMS fit $3.64x-5.3$")
    axes[1].set_xlabel(r"$x=R/2M$")
    axes[1].set_ylabel(r"$\tilde q=-Q_{\rm rot}M/J^2$")
    axes[1].set_title("J0737A dimensionless quadrupole")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Hartle second-order spin-quadrupole probe (frozen canonical EOS)", fontsize=12)
    fig.text(0.5, 0.01, "Second-order YY(l=2)+UMS(l=0); q_tilde is spin independent; forecast only, no empirical confirmation.",
             ha="center", fontsize=8, color="#555555", style="italic")
    fig.subplots_adjust(bottom=0.16, top=0.86, wspace=0.28)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_artifacts(payload: dict[str, Any]) -> None:
    assert_artifact_provenance(payload)
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_figure(payload)


def _print_summary(payload: dict[str, Any]) -> None:
    print("=" * 80)
    print("HARTLE SECOND-ORDER SPIN-QUADRUPOLE PROBE")
    print(f"Public status: {payload['public_status']}")
    mac = payload["maclaurin_validation"]
    print(f"Maclaurin validation: {mac['status']}")
    for row in mac["rows"]:
        print(f"  C={row['compactness']:.4f}  q~/(25R/8M)={row['q_tilde_ratio']:.6f}  "
              f"Q/Om^2R^5={row['Q_over_Om2R5']:.6f}  dM/MR^2Om^2={row['deltaM_over_MR2Om2']:.6f}  "
              f"I/MR^2={row['I_over_MR2']:.6f}")
    j0737 = payload["j0737a"]
    print(f"J0737A second-order: {j0737['status']}")
    print(f"  q_tilde = {j0737['q_tilde']:.6f}  (band {j0737['q_tilde_band']}, "
          f"in_band={j0737['q_tilde_in_band']})")
    print(f"  UMS fit 3.64x-5.3 at x={j0737['x_R_over_2M']:.4f} = {j0737['ums_fit_3p64x_minus_5p3']:.4f} "
          f"(rel dev {j0737['rel_dev_from_ums_fit']:.3e})")
    iq = j0737["iloveq_crosscheck"]
    print(f"  I-Q universal cross-check: {iq['status']}  Ibar={iq['Ibar_first_order']:.4f} vs "
          f"fit(Qbar)={iq['Ibar_from_universal_IQ_fit']:.4f}  (rel dev {iq['relative_deviation']:.3e})")
    convergence = j0737["convergence"]
    print(f"  Convergence: {convergence['status']}; Qbar range={convergence['q_tilde_numerical_range']}")
    print(f"  Finest-pair relative differences: {convergence['finest_pair_relative_differences']}")
    print(f"  A_seed={j0737['A_seed_km']:.6e} (dimensionless); mass correction: {j0737['mass_correction']['status']}")
    print(f"Artifact paths: {RESULT_PATH}; {FIGURE_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="semantic run; not terminal evidence")
    args = parser.parse_args(argv)
    payload = build_payload(quick=bool(args.quick))
    if not args.quick:
        assert_artifact_provenance(payload)
        write_artifacts(payload)
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
