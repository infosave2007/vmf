#!/usr/bin/env python3
"""Fail-closed symbolic and arbitrary-precision audit of Law 1:
Gauge-Invariant Quartic Vector Completion of ``source_complete``.

This module analyzes the unique missing dimension-4, Lorentz- and local-U(1)-
invariant operator built solely from the existing fields of ``source_complete``:

    L_g4 = +(zeta / 4) * (g_mu g^mu)^2,   g_mu = partial_mu theta - q_phi A_mu,

which reduces in unitary gauge (theta = 0) to +(zeta * q_phi^4 / 4) * (A_mu A^mu)^2.

Scope and scientific integrity (SPEC-v2):
- Pure mathematical / structural audit: ``evidence_weight = 0.0``.
- Reads live baseline parameters from ``source_complete_solution_audit.PARAMS``
  without hardcoded drift.
- Separates hypothetical nuclear saturation calibration targets (used solely to
  prove the single-scalar W + zeta structural no-go) from forward predictions.
- Cross-checks all key numerical certificates at 60 dps and 100 dps in ``mpmath``
  alongside exact symbolic proofs in ``sympy``.
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
import sympy as sp

try:  # Direct execution from ``verification``.
    from source_complete_solution_audit import PARAMS, Params
except ImportError:  # pragma: no cover - package-style import support.
    from .source_complete_solution_audit import PARAMS, Params


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULT_PATH = HERE / "nvg_quartic_vector_melting_results.json"
SOURCE_PATH = HERE / "source_complete_solution_audit.py"
SCHEMA = "nvg_quartic_vector_melting_audit.v1"
STATUS = "PASS_LAW1_QUARTIC_VECTOR_MELTING_AND_SINGLE_SCALAR_NOGO"
EVIDENCE_WEIGHT = 0.0

# Reference empirical pion mass (MeV) used solely as a qualitative scale
# benchmark in the single-scalar no-go diagnosis (not a fitted input).
M_PI_REF_MEV = Fraction("139.57039")
NUCLEAR_BINDING_REF_MEV = Fraction("16")


class QuarticVectorMeltingError(ValueError):
    """Raised when a mathematical certificate or canonical artifact fails closed."""


@dataclass(frozen=True)
class ExactBaseline:
    """Exact rational reconstruction of the live ``source_complete`` baseline."""

    W0: Fraction
    lam: Fraction
    M_N: Fraction
    m_omega: Fraction
    g_omega: Fraction
    n0_fm3: Fraction
    hbar_c: Fraction
    degeneracy: int

    @property
    def g_s(self) -> Fraction:
        return self.M_N / self.W0

    @property
    def q_phi(self) -> Fraction:
        return self.m_omega / self.W0

    @property
    def n0_nat(self) -> Fraction:
        return self.n0_fm3 * (self.hbar_c ** 3)

    @property
    def m_A_squared(self) -> Fraction:
        return self.m_omega * self.m_omega

    @property
    def m_sigma_squared(self) -> Fraction:
        return 2 * self.lam * self.W0 * self.W0


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:  # pragma: no cover
        raise QuarticVectorMeltingError(f"cannot read provenance path: {path}") from exc


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise QuarticVectorMeltingError(f"{label} cannot be bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QuarticVectorMeltingError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise QuarticVectorMeltingError(f"{label} is not finite")
    return number


def _live_fraction(value: Any, label: str) -> Fraction:
    number = _finite_float(value, label)
    if number <= 0.0:
        raise QuarticVectorMeltingError(f"{label} must be strictly positive")
    return Fraction(repr(number))


def _mp_frac(value: Fraction) -> mp.mpf:
    return mp.mpf(value.numerator) / mp.mpf(value.denominator)


def _fmt(value: Any, digits: int = 30) -> str:
    with mp.workdps(max(80, digits + 25)):
        val = _mp_frac(value) if isinstance(value, Fraction) else mp.mpf(value)
        if not mp.isfinite(val):
            raise QuarticVectorMeltingError("non-finite scientific output")
        return mp.nstr(val, digits)


def _fmt_sci(value: Any, digits: int = 12) -> str:
    with mp.workdps(max(80, digits + 25)):
        val = _mp_frac(value) if isinstance(value, Fraction) else mp.mpf(value)
        if not mp.isfinite(val):
            raise QuarticVectorMeltingError("non-finite scientific output")
        if val == 0:
            return "0.0"
        return mp.nstr(val, digits)


def exact_baseline_from_live(p: Params = PARAMS) -> ExactBaseline:
    """Read live ``PARAMS`` and enforce algebraic baseline consistency."""
    if not isinstance(p.degeneracy, int) or isinstance(p.degeneracy, bool) or p.degeneracy <= 0:
        raise QuarticVectorMeltingError("degeneracy must be a positive integer")
    exact = ExactBaseline(
        W0=_live_fraction(p.W0, "W0"),
        lam=_live_fraction(p.lam, "lam"),
        M_N=_live_fraction(p.M_N, "M_N"),
        m_omega=_live_fraction(p.m_omega, "m_omega"),
        g_omega=_live_fraction(p.g_omega, "g_omega"),
        n0_fm3=_live_fraction(p.n0_fm3, "n0_fm3"),
        hbar_c=_live_fraction(p.hbar_c, "hbar_c"),
        degeneracy=p.degeneracy,
    )
    with mp.workdps(80):
        checks = (
            (p.g_s, float(_mp_frac(exact.g_s)), "g_s=M_N/W0"),
            (p.q_phi, float(_mp_frac(exact.q_phi)), "q_phi=m_omega/W0"),
            (p.scalar_mass, float(mp.sqrt(_mp_frac(exact.m_sigma_squared))), "m_sigma=sqrt(2*lam)*W0"),
            (p.n0_nat, float(_mp_frac(exact.n0_nat)), "n0_nat=n0_fm3*hbar_c^3"),
        )
    for actual, expected, label in checks:
        if abs(_finite_float(actual, label) - expected) / max(1.0, abs(expected)) > 2.0e-14:
            raise QuarticVectorMeltingError(f"live PARAMS violated {label}")
    return exact


def _require_live_baseline(p: Params) -> None:
    if p != PARAMS:
        raise QuarticVectorMeltingError(
            "canonical audit artifact is bound exclusively to live source_complete PARAMS"
        )


# ---------------------------------------------------------------------------
# High-precision mpmath physics kernels (precision passed explicitly via dps)
# ---------------------------------------------------------------------------

def _mp_params(exact: ExactBaseline) -> dict[str, mp.mpf]:
    return {
        "W0": _mp_frac(exact.W0),
        "lam": _mp_frac(exact.lam),
        "M_N": _mp_frac(exact.M_N),
        "m_omega": _mp_frac(exact.m_omega),
        "g_omega": _mp_frac(exact.g_omega),
        "n0_fm3": _mp_frac(exact.n0_fm3),
        "hbar_c": _mp_frac(exact.hbar_c),
        "d": mp.mpf(exact.degeneracy),
        "g_s": _mp_frac(exact.g_s),
        "q_phi": _mp_frac(exact.q_phi),
        "n0_nat": _mp_frac(exact.n0_nat),
    }


def mp_fermi_k(n_nat: mp.mpf, d: mp.mpf) -> mp.mpf:
    if n_nat <= 0:
        return mp.mpf("0")
    return mp.cbrt(mp.mpf("6") * mp.pi**2 * n_nat / d)


def mp_fermi_energy(n_nat: mp.mpf, m: mp.mpf, d: mp.mpf) -> mp.mpf:
    k = mp_fermi_k(n_nat, d)
    return mp.sqrt(k * k + m * m)


def mp_fermi_free_energy(n_nat: mp.mpf, m: mp.mpf, d: mp.mpf) -> mp.mpf:
    if n_nat <= 0:
        return mp.mpf("0")
    k = mp_fermi_k(n_nat, d)
    if m == 0:
        return d * k**4 / (mp.mpf("8") * mp.pi**2)
    E = mp.sqrt(k * k + m * m)
    return d / (mp.mpf("16") * mp.pi**2) * (k * E * (mp.mpf("2") * k * k + m * m) - m**4 * mp.asinh(k / m))


def mp_fermi_pressure(n_nat: mp.mpf, m: mp.mpf, d: mp.mpf) -> mp.mpf:
    if n_nat <= 0:
        return mp.mpf("0")
    k = mp_fermi_k(n_nat, d)
    if m == 0:
        return d * k**4 / (mp.mpf("24") * mp.pi**2)
    E = mp.sqrt(k * k + m * m)
    return d / (mp.mpf("48") * mp.pi**2) * (
        k * E * (mp.mpf("2") * k * k - mp.mpf("3") * m * m) + mp.mpf("3") * m**4 * mp.asinh(k / m)
    )


def mp_scalar_density(n_nat: mp.mpf, m: mp.mpf, d: mp.mpf) -> mp.mpf:
    if n_nat <= 0 or m == 0:
        return mp.mpf("0")
    k = mp_fermi_k(n_nat, d)
    E = mp.sqrt(k * k + m * m)
    return d * m / (mp.mpf("4") * mp.pi**2) * (k * E - m * m * mp.asinh(k / m))


def mp_dns_dm(n_nat: mp.mpf, m: mp.mpf, d: mp.mpf) -> mp.mpf:
    """Exact partial derivative dn_s / dm at fixed n_B."""
    if n_nat <= 0:
        return mp.mpf("0")
    k = mp_fermi_k(n_nat, d)
    if m == 0:
        return d * k * k / (mp.mpf("4") * mp.pi**2)
    E = mp.sqrt(k * k + m * m)
    return d / (mp.mpf("4") * mp.pi**2) * (k * (k * k + mp.mpf("3") * m * m) / E - mp.mpf("3") * m * m * mp.asinh(k / m))


def mp_dns_dn(n_nat: mp.mpf, m: mp.mpf, d: mp.mpf) -> mp.mpf:
    """Exact partial derivative dn_s / dn_B = dE_F / dm = m / E_F at fixed m."""
    if n_nat <= 0 or m == 0:
        return mp.mpf("0")
    return m / mp_fermi_energy(n_nat, m, d)


def mp_solve_A0(n_nat: mp.mpf, W: mp.mpf, zeta: mp.mpf, q_phi: mp.mpf, g_omega: mp.mpf) -> mp.mpf:
    """Solve exact cubic Gauss constraint: q_phi^2 W^2 A0 + zeta q_phi^4 A0^3 = g_omega n_B."""
    if n_nat == 0:
        return mp.mpf("0")
    if zeta < 0:
        raise QuarticVectorMeltingError("negative zeta is acausal/unstable; Gauss solver requires zeta >= 0")
    if zeta == 0:
        if W <= 0:
            raise QuarticVectorMeltingError("W must be strictly positive when zeta = 0")
        return g_omega * n_nat / (q_phi * q_phi * W * W)
    a = zeta * q_phi**4
    b = q_phi**2 * W * W
    c = g_omega * n_nat
    if W == 0:
        return mp.cbrt(c / a)
    # Depressed cubic: A0^3 + p A0 - q = 0 with p > 0, q > 0
    pp = b / a
    qq = c / a
    half_q = qq / mp.mpf("2")
    disc = mp.sqrt(half_q * half_q + (pp / mp.mpf("3")) ** 3)
    u = mp.cbrt(half_q + disc)
    A0 = u - pp / (mp.mpf("3") * u)
    # Newton polish to full working precision
    for _ in range(12):
        f = b * A0 + a * A0**3 - c
        df = b + mp.mpf("3") * a * A0**2
        step = f / df
        A0 -= step
        if abs(step) <= mp.eps * max(mp.mpf("1"), abs(A0)) * mp.mpf("4"):
            break
    return A0


def mp_thermo_state(
    n_nat: mp.mpf,
    W: mp.mpf,
    zeta: mp.mpf,
    *,
    W0: mp.mpf,
    lam: mp.mpf,
    g_s: mp.mpf,
    q_phi: mp.mpf,
    g_omega: mp.mpf,
    d: mp.mpf,
) -> dict[str, mp.mpf]:
    """Compute full equilibrium thermodynamics and exact analytical c_s^2, c_L^2 at (n_nat, W)."""
    A0 = mp_solve_A0(n_nat, W, zeta, q_phi, g_omega)
    m = g_s * W
    U = lam * (W * W - W0 * W0) ** 2 / mp.mpf("4")
    Up = lam * W * (W * W - W0 * W0)
    FF = mp_fermi_free_energy(n_nat, m, d)
    PF = mp_fermi_pressure(n_nat, m, d)
    EF = mp_fermi_energy(n_nat, m, d)
    ns = mp_scalar_density(n_nat, m, d)
    zeta_A = zeta * q_phi**4
    mA2 = q_phi**2 * W * W
    vec_quad = mp.mpf("0.5") * mA2 * A0**2
    vec_quart = zeta_A * A0**4

    eps = FF + U + vec_quad + mp.mpf("0.75") * vec_quart
    P = PF - U + vec_quad + mp.mpf("0.25") * vec_quart
    mu_B = EF + g_omega * A0
    G = FF + U - vec_quad - mp.mpf("0.25") * vec_quart + g_omega * A0 * n_nat - mu_B * n_nat

    gauss_res = mA2 * A0 + zeta_A * A0**3 - g_omega * n_nat
    w_res = Up + g_s * ns - q_phi**2 * W * A0**2
    trace_lhs = eps - mp.mpf("3") * P
    trace_rhs = mp.mpf("4") * U - W * Up
    trace_res = trace_lhs - trace_rhs
    legendre_res = (mu_B * n_nat - eps) - P
    generator_res = (-G) - P

    # Vector longitudinal speed squared c_L^2
    M0_sq = mA2 + mp.mpf("3") * zeta_A * A0**2
    MT_sq = mA2 + zeta_A * A0**2
    c_L_sq = MT_sq / M0_sq if M0_sq > 0 else mp.mpf("1")

    # Exact analytical thermodynamic speed of sound c_s^2 = (n_B / mu_B) * (d mu_B / d n_B)
    # and incompressibility K = 9 * n_B * (d mu_B / d n_B).
    if n_nat == 0:
        dmu_dn = mp.mpf("0")
        cs2 = mp.mpf("0")
        dW_dn = mp.mpf("0")
    else:
        kF = mp_fermi_k(n_nat, d)
        # dE_F/dn_B at fixed W: dk_F/dn_B = 2 pi^2 / (d k_F^2), dE_F/dk_F = k_F / E_F
        dEF_dn = (mp.mpf("2") * mp.pi**2) / (d * kF * EF)
        dEF_dW = g_s * m / EF if m > 0 else mp.mpf("0")
        # From Gauss: (mA2 + 3 zeta_A A0^2) dA0 + 2 q_phi^2 W A0 dW = g_omega dn_B
        dA0_dn_fixedW = g_omega / M0_sq
        dA0_dW_fixedn = -mp.mpf("2") * q_phi**2 * W * A0 / M0_sq

        if W == 0:
            dW_dn = mp.mpf("0")
        else:
            # W stationarity F(W, n) = Up + g_s ns - q_phi^2 W A0^2 = 0
            dF_dW = (
                lam * (mp.mpf("3") * W * W - W0 * W0)
                + g_s**2 * mp_dns_dm(n_nat, m, d)
                - q_phi**2 * A0**2
                - mp.mpf("2") * q_phi**2 * W * A0 * dA0_dW_fixedn
            )
            dF_dn = g_s * mp_dns_dn(n_nat, m, d) - mp.mpf("2") * q_phi**2 * W * A0 * dA0_dn_fixedW
            dW_dn = -dF_dn / dF_dW

        dA0_dn = dA0_dn_fixedW + dA0_dW_fixedn * dW_dn
        dmu_dn = dEF_dn + dEF_dW * dW_dn + g_omega * dA0_dn
        cs2 = (n_nat / mu_B) * dmu_dn

    return {
        "n_nat": n_nat,
        "W": W,
        "A0": A0,
        "Mstar": m,
        "eps": eps,
        "P": P,
        "mu_B": mu_B,
        "G": G,
        "U": U,
        "Up": Up,
        "ns": ns,
        "gauss_res": gauss_res,
        "w_res": w_res,
        "trace_lhs": trace_lhs,
        "trace_rhs": trace_rhs,
        "trace_res": trace_res,
        "legendre_res": legendre_res,
        "generator_res": generator_res,
        "M0_sq": M0_sq,
        "MT_sq": MT_sq,
        "c_L_sq": c_L_sq,
        "dW_dn": dW_dn,
        "dmu_dn": dmu_dn,
        "cs2": cs2,
    }


def mp_melting_constants(exact: ExactBaseline) -> dict[str, mp.mpf]:
    """Compute exact analytical melting threshold constants at current mp.dps."""
    mp_p = _mp_params(exact)
    d = mp_p["d"]
    g_s = mp_p["g_s"]
    q_phi = mp_p["q_phi"]
    g_omega = mp_p["g_omega"]
    c_s = (d / (mp.mpf("4") * mp.pi**2)) * (mp.mpf("6") * mp.pi**2 / d) ** (mp.mpf("2") / mp.mpf("3")) * g_s**2
    zeta_melt = g_omega / (q_phi * c_s ** (mp.mpf("3") / mp.mpf("2")))
    zeta_A_melt = zeta_melt * q_phi**4
    zeta_RMF_melt = mp.mpf("6") * zeta_melt * (q_phi / g_omega) ** 4
    return {
        "c_s": c_s,
        "zeta_melt": zeta_melt,
        "zeta_A_melt": zeta_A_melt,
        "zeta_RMF_melt": zeta_RMF_melt,
    }


def mp_alpha_eff(zeta: mp.mpf, exact: ExactBaseline) -> mp.mpf:
    mp_p = _mp_params(exact)
    consts = mp_melting_constants(exact)
    c_v = (mp_p["g_omega"] / (zeta * mp_p["q_phi"])) ** (mp.mpf("2") / mp.mpf("3"))
    return consts["c_s"] - c_v


def mp_n_melt(zeta: mp.mpf, exact: ExactBaseline) -> mp.mpf:
    mp_p = _mp_params(exact)
    alpha = mp_alpha_eff(zeta, exact)
    if alpha <= 0:
        raise QuarticVectorMeltingError("no melting threshold for zeta <= zeta_melt")
    return (mp_p["lam"] * mp_p["W0"] ** 2 / alpha) ** (mp.mpf("3") / mp.mpf("2"))


def mp_solve_equilibrium_W(
    n_nat: mp.mpf,
    zeta: mp.mpf,
    exact: ExactBaseline,
    *,
    prefer_melted_at_spinodal: bool = True,
) -> mp.mpf:
    """Solve for the equilibrium condensate W(n_nat, zeta) to full mpmath precision.

    For n_nat == 0 returns W0.
    For zeta > 0, checks both the melted branch W = 0 and any positive root of
    g(W) = (d eps / d W) / W = lam*(W^2 - W0^2) + g_s*ns/W - q_phi^2*A0^2 = 0,
    returning the true minimum (or W = 0 at/above the exact spinodal n_melt when
    no nontrivial macroscopic branch exists).
    """
    mp_p = _mp_params(exact)
    W0 = mp_p["W0"]
    lam = mp_p["lam"]
    g_s = mp_p["g_s"]
    q_phi = mp_p["q_phi"]
    g_omega = mp_p["g_omega"]
    d = mp_p["d"]

    if n_nat == 0:
        return W0

    def g_over_W(W: mp.mpf) -> mp.mpf:
        if W == 0:
            if zeta == 0:
                return mp.mpf("-inf")
            return -lam * W0 * W0 + mp_alpha_eff(zeta, exact) * mp.cbrt(n_nat) ** 2
        A0 = mp_solve_A0(n_nat, W, zeta, q_phi, g_omega)
        m = g_s * W
        return lam * (W * W - W0 * W0) + (g_s * mp_scalar_density(n_nat, m, d)) / W - q_phi**2 * A0**2

    def dg_dW(W: mp.mpf) -> mp.mpf:
        A0 = mp_solve_A0(n_nat, W, zeta, q_phi, g_omega)
        m = g_s * W
        ns = mp_scalar_density(n_nat, m, d)
        dns_dm = mp_dns_dm(n_nat, m, d)
        M0_sq = q_phi**2 * W * W + mp.mpf("3") * zeta * q_phi**4 * A0**2
        dA0_dW = -mp.mpf("2") * q_phi**2 * W * A0 / M0_sq
        return (
            mp.mpf("2") * lam * W
            + g_s * (g_s * dns_dm * W - ns) / (W * W)
            - mp.mpf("2") * q_phi**2 * A0 * dA0_dW
        )

    g0 = g_over_W(mp.mpf("0"))

    # If g(0) >= 0 (within numerical rounding of the exact spinodal n_melt),
    # W = 0 is a local minimum (with only an O(1e-14) relativistic Fermi-log
    # pocket around W ~ 2 MeV inside n/n_melt in [1, 1 + 2e-6]).
    spinodal_tol = mp.eps * mp.mpf("1e4") * lam * W0 * W0
    if zeta > 0 and g0 >= -spinodal_tol and prefer_melted_at_spinodal:
        return mp.mpf("0")

    # Bracket positive root: g(lo) < 0, g(hi) > 0
    lo = mp.mpf("1e-8") * W0
    flo = g_over_W(lo)
    if flo >= 0:
        # Check if there is any negative dip at small W (the tiny Fermi-log pocket)
        for exp in (-6, -4, -3, -2, -1):
            cand = mp.power(10, exp) * W0
            fc = g_over_W(cand)
            if fc < 0:
                lo, flo = cand, fc
                break
        else:
            return mp.mpf("0")

    hi = max(mp.mpf("2") * W0, W0 + mp.mpf("1"))
    fhi = g_over_W(hi)
    while fhi <= 0:
        hi *= mp.mpf("2")
        fhi = g_over_W(hi)
        if hi > mp.mpf("1e9"):
            raise QuarticVectorMeltingError("failed to bracket positive W equilibrium root")

    # Bisect to tight bracket then polish with Newton to full precision
    for _ in range(90):
        mid = (lo + hi) / mp.mpf("2")
        fm = g_over_W(mid)
        if fm <= 0:
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= mp.mpf("1e-12") * max(mp.mpf("1"), mid):
            break

    W = (lo + hi) / mp.mpf("2")
    for _ in range(25):
        fw = g_over_W(W)
        dfw = dg_dW(W)
        if dfw == 0:
            break
        step = fw / dfw
        W_next = W - step
        if W_next <= 0:
            W_next = W / mp.mpf("2")
        W = W_next
        if abs(step) <= mp.eps * max(mp.mpf("1"), abs(W)) * mp.mpf("8"):
            break
    return W


# ---------------------------------------------------------------------------
# Section 1: SymPy Symbolic Proofs (Operators, Gauss, Trace, Causality, Threshold)
# ---------------------------------------------------------------------------

def run_symbolic_proofs() -> dict[str, Any]:
    """Execute exact symbolic derivations and identity proofs in SymPy."""
    # 1. Local U(1) gauge invariance of g_mu = d_mu theta - q_phi A_mu and L_g4
    q_phi, zeta = sp.symbols("q_phi zeta", positive=True, real=True)
    dtheta = sp.symbols("dtheta0:4", real=True)
    A = sp.symbols("A0:4", real=True)
    dalpha = sp.symbols("dalpha0:4", real=True)
    eta = [1, -1, -1, -1]

    g_vec = [dtheta[i] - q_phi * A[i] for i in range(4)]
    g_vec_prime = [(dtheta[i] + q_phi * dalpha[i]) - q_phi * (A[i] + dalpha[i]) for i in range(4)]
    g_inv_diffs = [sp.simplify(g_vec_prime[i] - g_vec[i]) for i in range(4)]

    g_sq = sum(eta[i] * g_vec[i] ** 2 for i in range(4))
    g_sq_prime = sum(eta[i] * g_vec_prime[i] ** 2 for i in range(4))
    L_g4 = sp.Rational(1, 4) * zeta * g_sq**2
    L_g4_prime = sp.Rational(1, 4) * zeta * g_sq_prime**2
    L_g4_gauge_diff = sp.simplify(L_g4_prime - L_g4)

    # Unitary gauge reduction (dtheta_mu = 0)
    L_g4_unitary = sp.simplify(L_g4.subs({dtheta[i]: 0 for i in range(4)}))
    A_sq = sum(eta[i] * A[i] ** 2 for i in range(4))
    L_g4_unitary_expected = sp.Rational(1, 4) * zeta * q_phi**4 * A_sq**2
    unitary_diff = sp.simplify(L_g4_unitary - L_g4_unitary_expected)

    # Metric variation / Hilbert stress tensor of L_g4 in homogeneous background g_mu = (-q_phi A0, 0, 0, 0)
    A0, W, W0, lam, g_s, g_omega, n_B, mu_B = sp.symbols(
        "A0 W W0 lam g_s g_omega n_B mu_B", positive=True, real=True
    )
    # T_{mu nu}^{(g4)} = zeta (g_alpha g^alpha) g_mu g_nu - (zeta/4) eta_{mu nu} (g_alpha g^alpha)^2
    g0 = -q_phi * A0
    g_sq_bg = g0**2
    T00_g4 = sp.simplify(zeta * g_sq_bg * g0**2 - sp.Rational(1, 4) * zeta * 1 * g_sq_bg**2)
    P_g4 = sp.simplify(zeta * g_sq_bg * 0 - sp.Rational(1, 4) * zeta * (-1) * g_sq_bg**2)
    trace_g4 = sp.simplify(T00_g4 - 3 * P_g4)

    # 2. Stationary generator, cubic Gauss constraint, energy density, pressure
    k_F, m, d = sp.symbols("k_F m d", positive=True, real=True)
    E_F_expr = sp.sqrt(k_F**2 + m**2)
    FF_expr = d / (16 * sp.pi**2) * (k_F * E_F_expr * (2 * k_F**2 + m**2) - m**4 * sp.asinh(k_F / m))
    PF_expr = d / (48 * sp.pi**2) * (k_F * E_F_expr * (2 * k_F**2 - 3 * m**2) + 3 * m**4 * sp.asinh(k_F / m))
    ns_expr = d * m / (4 * sp.pi**2) * (k_F * E_F_expr - m**2 * sp.asinh(k_F / m))
    n_B_of_kF = d * k_F**3 / (6 * sp.pi**2)

    # Prove exact free Fermi identities symbolically
    fermi_trace_diff = sp.simplify(FF_expr - 3 * PF_expr - m * ns_expr)
    fermi_legendre_diff = sp.simplify(FF_expr + PF_expr - E_F_expr * n_B_of_kF)
    fermi_dFF_dm_diff = sp.simplify(sp.diff(FF_expr, m) - ns_expr)

    # Symbolic stationary generator G with abstract FF, PF, ns satisfying exact Fermi identities
    FF_sym, PF_sym, ns_sym, EF_sym = sp.symbols("F_F P_F n_s E_F", real=True)
    U_sym = sp.Function("U")(W)
    G_sym = (
        FF_sym
        + U_sym
        - sp.Rational(1, 2) * q_phi**2 * W**2 * A0**2
        - sp.Rational(1, 4) * zeta * q_phi**4 * A0**4
        + g_omega * A0 * n_B
        - mu_B * n_B
    )
    dG_dA0 = sp.simplify(sp.diff(G_sym, A0))
    gauss_lhs = q_phi**2 * W**2 * A0 + zeta * q_phi**4 * A0**3
    gauss_rhs = g_omega * n_B
    gauss_eom_diff = sp.simplify(dG_dA0 + (gauss_lhs - gauss_rhs))

    # Substitute mu_B = EF + g_omega A0 and FF - EF n_B = -PF, and Gauss g_omega A0 n_B = gauss_lhs * A0
    P_from_G = sp.simplify(
        (-G_sym)
        .subs(mu_B, EF_sym + g_omega * A0)
        .subs(FF_sym, EF_sym * n_B - PF_sym)
    )
    P_expected = PF_sym - U_sym + sp.Rational(1, 2) * q_phi**2 * W**2 * A0**2 + sp.Rational(1, 4) * zeta * q_phi**4 * A0**4
    P_formula_diff = sp.simplify(P_from_G - P_expected)

    eps_from_G = sp.simplify(
        (G_sym + mu_B * n_B).subs(g_omega * n_B, gauss_lhs)
    )
    eps_expected = FF_sym + U_sym + sp.Rational(1, 2) * q_phi**2 * W**2 * A0**2 + sp.Rational(3, 4) * zeta * q_phi**4 * A0**4
    eps_formula_diff = sp.simplify(eps_from_G - eps_expected)

    # 3. Exact Equilibrium Trace Identity: eps - 3*P = 4*U(W) - W*U'(W)
    dG_dW_sym = g_s * ns_sym + sp.diff(U_sym, W) - q_phi**2 * W * A0**2
    trace_identity_expr = sp.simplify(
        (eps_expected - 3 * P_expected - (4 * U_sym - W * sp.diff(U_sym, W)))
        .subs(FF_sym - 3 * PF_sym, g_s * W * ns_sym)
        - W * dG_dW_sym
    )
    # Also verify directly by expanding FF_sym = 3*PF_sym + g_s*W*ns_sym
    trace_direct_diff = sp.simplify(
        (eps_expected - 3 * P_expected).subs(FF_sym, 3 * PF_sym + g_s * W * ns_sym)
        - (4 * U_sym - W * sp.diff(U_sym, W) + W * dG_dW_sym)
    )

    # Melted phase W = 0 with U(W) = lam/4 * (W^2 - W0^2)^2
    U_mex = lam * (W**2 - W0**2) ** 2 / 4
    trace_mex = sp.simplify(4 * U_mex - W * sp.diff(U_mex, W))
    trace_mex_expected = lam * W0**2 * (W0**2 - W**2)
    trace_mex_diff = sp.simplify(trace_mex - trace_mex_expected)
    trace_melted = sp.simplify(trace_mex.subs(W, 0))
    B_bag = sp.simplify(U_mex.subs(W, 0))
    melted_bag_diff = sp.simplify(trace_melted - 4 * B_bag)

    # Exact CSS speed of sound c_s^2 = 1/3 in melted phase W = 0
    kF_n = (6 * sp.pi**2 * n_B / d) ** sp.Rational(1, 3)
    A0_melted = (g_omega * n_B / (zeta * q_phi**4)) ** sp.Rational(1, 3)
    eps_melted_n = d * kF_n**4 / (8 * sp.pi**2) + B_bag + sp.Rational(3, 4) * zeta * q_phi**4 * A0_melted**4
    P_melted_n = d * kF_n**4 / (24 * sp.pi**2) - B_bag + sp.Rational(1, 4) * zeta * q_phi**4 * A0_melted**4
    cs2_melted_sym = sp.simplify(sp.diff(P_melted_n, n_B) / sp.diff(eps_melted_n, n_B))
    css_eos_diff = sp.simplify(P_melted_n - (eps_melted_n - 4 * B_bag) / 3)

    # 4. Causality & Stability (quadratic fluctuation Lagrangian around A_mu = (A0, 0))
    m_A, zeta_A, omega, k_mod = sp.symbols("m_A zeta_A omega k", positive=True, real=True)
    a0, aL, aT = sp.symbols("a0 aL aT", real=True)
    # Expand L_pot = 1/2 m_A^2 ((A0+a0)^2 - aL^2 - aT^2) + 1/4 zeta_A ((A0+a0)^2 - aL^2 - aT^2)^2
    A_pert_sq = (A0 + a0) ** 2 - aL**2 - aT**2
    L_pot_pert = sp.Rational(1, 2) * m_A**2 * A_pert_sq + sp.Rational(1, 4) * zeta_A * A_pert_sq**2
    M0_sq_sym = sp.simplify(sp.diff(L_pot_pert, a0, 2).subs({a0: 0, aL: 0, aT: 0}))
    MT_sq_sym = sp.simplify(-sp.diff(L_pot_pert, aL, 2).subs({a0: 0, aL: 0, aT: 0}))
    MT_sq_trans_sym = sp.simplify(-sp.diff(L_pot_pert, aT, 2).subs({a0: 0, aL: 0, aT: 0}))
    M0_sq_diff = sp.simplify(M0_sq_sym - (m_A**2 + 3 * zeta_A * A0**2))
    MT_sq_diff = sp.simplify(MT_sq_sym - (m_A**2 + zeta_A * A0**2))
    MT_iso_diff = sp.simplify(MT_sq_sym - MT_sq_trans_sym)

    # Eliminate constraint a0 in longitudinal sector:
    # L_L = 1/2 * (omega*aL - k*a0)^2 + 1/2 * M0^2 * a0^2 - 1/2 * MT^2 * aL^2
    L_L = sp.Rational(1, 2) * (omega * aL - k_mod * a0) ** 2 + sp.Rational(1, 2) * M0_sq_sym * a0**2 - sp.Rational(1, 2) * MT_sq_sym * aL**2
    a0_sol = sp.solve(sp.diff(L_L, a0), a0)[0]
    L_L_eff = sp.simplify(L_L.subs(a0, a0_sol))
    omega_L_sq_sol = sp.simplify(sp.solve(L_L_eff / aL**2, omega**2)[0])
    c_L_sq_sym = sp.simplify(sp.diff(omega_L_sq_sol, k_mod, 2) / 2)
    omega_L_k0_diff = sp.simplify(omega_L_sq_sol.subs(k_mod, 0) - MT_sq_sym)
    c_L_sq_expected = (m_A**2 + zeta_A * A0**2) / (m_A**2 + 3 * zeta_A * A0**2)
    c_L_sq_diff = sp.simplify(c_L_sq_sym - c_L_sq_expected) + omega_L_k0_diff
    c_L_minus_third = sp.factor(c_L_sq_expected - sp.Rational(1, 3))
    one_minus_c_L = sp.factor(1 - c_L_sq_expected)

    # 5. Exact small-W vacuum melting threshold
    ns_over_W_limit = sp.simplify(sp.limit((g_s * ns_expr.subs(m, g_s * W)) / W, W, 0, dir="+"))
    c_s_from_limit = sp.simplify(ns_over_W_limit.subs(k_F, kF_n) / n_B ** sp.Rational(2, 3))
    c_s_expected = (d / (4 * sp.pi**2)) * (6 * sp.pi**2 / d) ** sp.Rational(2, 3) * g_s**2
    c_s_diff = sp.simplify(c_s_from_limit - c_s_expected)

    c_v_from_limit = sp.simplify((q_phi**2 * A0_melted**2) / n_B ** sp.Rational(2, 3))
    c_v_expected = (g_omega / (zeta * q_phi)) ** sp.Rational(2, 3)
    c_v_diff = sp.simplify(c_v_from_limit - c_v_expected)

    c_s_sym = sp.Symbol("c_s", positive=True)
    zeta_melt_sol = sp.solve(c_s_sym - (g_omega / (zeta * q_phi)) ** sp.Rational(2, 3), zeta)[0]
    zeta_melt_expected = g_omega / (q_phi * c_s_sym ** sp.Rational(3, 2))
    zeta_melt_diff = sp.simplify(zeta_melt_sol - zeta_melt_expected)

    # 6. Single-scalar no-go trace-identity lock for m_sigma(y)
    y_sym, E_sat_sym, n0_sym = sp.symbols("y E_sat n0", positive=True, real=True)
    lam_from_trace = sp.solve(trace_mex_expected.subs(W, y_sym * W0) - E_sat_sym * n0_sym, lam)[0]
    m_sigma_sq_from_trace = sp.simplify(2 * lam_from_trace * W0**2)
    m_sigma_sq_expected = 2 * E_sat_sym * n0_sym / (W0**2 * (1 - y_sym**2))
    m_sigma_lock_diff = sp.simplify(m_sigma_sq_from_trace - m_sigma_sq_expected)

    all_exact_zero = (
        all(diff == 0 for diff in g_inv_diffs)
        and L_g4_gauge_diff == 0
        and unitary_diff == 0
        and trace_g4 == 0
        and fermi_trace_diff == 0
        and fermi_legendre_diff == 0
        and fermi_dFF_dm_diff == 0
        and gauss_eom_diff == 0
        and P_formula_diff == 0
        and eps_formula_diff == 0
        and trace_identity_expr == 0
        and trace_direct_diff == 0
        and trace_mex_diff == 0
        and melted_bag_diff == 0
        and cs2_melted_sym == sp.Rational(1, 3)
        and css_eos_diff == 0
        and M0_sq_diff == 0
        and MT_sq_diff == 0
        and MT_iso_diff == 0
        and c_L_sq_diff == 0
        and c_s_diff == 0
        and c_v_diff == 0
        and zeta_melt_diff == 0
        and m_sigma_lock_diff == 0
    )
    if not all_exact_zero:
        raise QuarticVectorMeltingError("one or more SymPy symbolic proofs failed")

    return {
        "all_symbolic_proofs_exact": True,
        "operator_uniqueness_and_invariance": {
            "operator": "L_g4 = +(zeta/4) * (g_mu g^mu)^2 with g_mu = partial_mu theta - q_phi A_mu",
            "unitary_gauge_form": str(L_g4_unitary),
            "g_mu_local_U1_invariance_exact_zero": all(diff == 0 for diff in g_inv_diffs),
            "L_g4_local_U1_invariance_exact_zero": L_g4_gauge_diff == 0,
            "unitary_gauge_reduction_exact_zero": unitary_diff == 0,
            "T00_g4": str(T00_g4),
            "P_g4": str(P_g4),
            "weyl_trace_T00_minus_3P_exact_zero": trace_g4 == 0,
        },
        "thermodynamics_and_trace_identity": {
            "fermi_trace_FF_minus_3PF_eq_m_ns_exact": fermi_trace_diff == 0,
            "fermi_legendre_FF_plus_PF_eq_EF_nB_exact": fermi_legendre_diff == 0,
            "fermi_dFF_dm_eq_ns_exact": fermi_dFF_dm_diff == 0,
            "cubic_gauss_eom_exact": gauss_eom_diff == 0,
            "pressure_formula_exact": P_formula_diff == 0,
            "energy_density_formula_exact": eps_formula_diff == 0,
            "equilibrium_trace_identity_unchanged_exact": trace_identity_expr == 0 and trace_direct_diff == 0,
            "mexican_hat_trace_4U_minus_WUp": str(trace_mex),
            "melted_W0_trace_equals_4B_bag_exact": melted_bag_diff == 0,
            "melted_W0_css_eos_exact": css_eos_diff == 0,
            "melted_W0_cs2_exact": str(cs2_melted_sym),
        },
        "causality_and_stability": {
            "M0_sq": str(M0_sq_sym),
            "MT_sq": str(MT_sq_sym),
            "longitudinal_dispersion_omega_L_sq": str(sp.simplify(omega_L_sq_sol)),
            "c_L_sq": str(c_L_sq_expected),
            "c_L_sq_minus_one_third": str(c_L_minus_third),
            "one_minus_c_L_sq": str(one_minus_c_L),
            "c_L_sq_derivation_exact": c_L_sq_diff == 0,
        },
        "melting_threshold_and_nogo": {
            "c_s_limit_exact": c_s_diff == 0,
            "c_v_limit_exact": c_v_diff == 0,
            "zeta_melt_formula_exact": zeta_melt_diff == 0,
            "single_scalar_m_sigma_sq_trace_lock": str(m_sigma_sq_expected),
            "single_scalar_m_sigma_sq_lock_exact": m_sigma_lock_diff == 0,
        },
    }


# ---------------------------------------------------------------------------
# Numerical Manufactured Local-U(1) Gauge Control at 60 dps
# ---------------------------------------------------------------------------

def manufactured_g4_gauge_control(exact: ExactBaseline, dps: int = 60) -> dict[str, Any]:
    """Verify off-shell local-U(1) invariance of g_mu and L_g4 at arbitrary precision."""
    with mp.workdps(dps):
        q_phi = _mp_frac(exact.q_phi)
        zeta = mp.mpf("35.620922")
        eta = (mp.mpf("1"), mp.mpf("-1"), mp.mpf("-1"), mp.mpf("-1"))
        dtheta = (mp.mpf("0.17"), mp.mpf("-0.23"), mp.mpf("0.11"), mp.mpf("-0.07"))
        A = (mp.mpf("0.31"), mp.mpf("-0.19"), mp.mpf("0.13"), mp.mpf("0.29"))
        dalpha = (mp.mpf("0.41"), mp.mpf("-0.37"), mp.mpf("0.23"), mp.mpf("-0.17"))

        g = [dtheta[i] - q_phi * A[i] for i in range(4)]
        g_t = [(dtheta[i] + q_phi * dalpha[i]) - q_phi * (A[i] + dalpha[i]) for i in range(4)]
        g_max_diff = max(abs(g_t[i] - g[i]) for i in range(4))

        g_sq = sum(eta[i] * g[i] ** 2 for i in range(4))
        g_sq_t = sum(eta[i] * g_t[i] ** 2 for i in range(4))
        L = mp.mpf("0.25") * zeta * g_sq**2
        L_t = mp.mpf("0.25") * zeta * g_sq_t**2
        L_rel_diff = abs(L_t - L) / max(mp.mpf("1"), abs(L))

        # Finite-difference variation dL_g4 / dA_mu = -zeta * q_phi * (g.g) * g^mu
        # Check analytically via mp.diff on A_0 component
        def L_of_A0(a0_val: mp.mpf) -> mp.mpf:
            gg = [dtheta[0] - q_phi * a0_val, g[1], g[2], g[3]]
            sq = sum(eta[i] * gg[i] ** 2 for i in range(4))
            return mp.mpf("0.25") * zeta * sq**2

        dL_dA0_num = mp.diff(L_of_A0, A[0])
        dL_dA0_exact = -zeta * q_phi * g_sq * g[0]
        dL_rel_err = abs(dL_dA0_num - dL_dA0_exact) / max(mp.mpf("1"), abs(dL_dA0_exact))

        passed = bool(g_max_diff < mp.mpf("1e-50") and L_rel_diff < mp.mpf("1e-50") and dL_rel_err < mp.mpf("1e-45"))
        if not passed:
            raise QuarticVectorMeltingError("manufactured L_g4 gauge invariance control failed")
        return {
            "precision_dps": dps,
            "g_mu_max_abs_diff": _fmt_sci(g_max_diff),
            "L_g4_relative_diff": _fmt_sci(L_rel_diff),
            "dL_dA0_variation_relative_error": _fmt_sci(dL_rel_err),
            "pass": passed,
        }


# ---------------------------------------------------------------------------
# Section 2, 3, 4: High-Precision Profiles, Melting, CSS, Causality, 60 vs 100 dps
# ---------------------------------------------------------------------------

def _evaluate_phase_suite(exact: ExactBaseline, dps: int) -> dict[str, Any]:
    """Compute threshold constants, melting profiles, trace checks, and causality at given dps."""
    with mp.workdps(dps):
        mp_p = _mp_params(exact)
        consts = mp_melting_constants(exact)
        zeta_melt = consts["zeta_melt"]
        W0 = mp_p["W0"]
        lam = mp_p["lam"]
        g_s = mp_p["g_s"]
        q_phi = mp_p["q_phi"]
        g_omega = mp_p["g_omega"]
        d = mp_p["d"]
        n0_nat = mp_p["n0_nat"]
        hbar_c3 = mp_p["hbar_c"] ** 3
        B_bag = lam * W0**4 / mp.mpf("4")
        four_B_bag = lam * W0**4

        # Baseline zeta = 0 check at n0 and 5*n0 to confirm trace identity also holds at zeta=0
        zeta0_states = []
        for x in (mp.mpf("1"), mp.mpf("5")):
            n_nat = x * n0_nat
            W = mp_solve_equilibrium_W(n_nat, mp.mpf("0"), exact)
            st = mp_thermo_state(
                n_nat, W, mp.mpf("0"), W0=W0, lam=lam, g_s=g_s, q_phi=q_phi, g_omega=g_omega, d=d
            )
            zeta0_states.append(st)

        # Profile 1: zeta = 0.5 * zeta_melt (no melting; W grows at high density)
        zeta_half = mp.mpf("0.5") * zeta_melt
        alpha_half = mp_alpha_eff(zeta_half, exact)
        sub_xs = (mp.mpf("0"), mp.mpf("1"), mp.mpf("5"), mp.mpf("10"), mp.mpf("100"), mp.mpf("1000"))
        sub_rows = []
        for x in sub_xs:
            n_nat = x * n0_nat
            W = mp_solve_equilibrium_W(n_nat, zeta_half, exact)
            st = mp_thermo_state(
                n_nat, W, zeta_half, W0=W0, lam=lam, g_s=g_s, q_phi=q_phi, g_omega=g_omega, d=d
            )
            sub_rows.append((x, st))

        # Profile 2 & 3: zeta = 2.0 * zeta_melt and zeta = 5.0 * zeta_melt (melting to W = 0)
        super_profiles: dict[str, dict[str, Any]] = {}
        for label, mult_str in (("zeta_2x_melt", "2"), ("zeta_5x_melt", "5")):
            mult = mp.mpf(mult_str)
            zeta_val = mult * zeta_melt
            alpha_eff = mp_alpha_eff(zeta_val, exact)
            n_melt_nat = mp_n_melt(zeta_val, exact)
            n_melt_fm3 = n_melt_nat / hbar_c3
            n_melt_over_n0 = n_melt_nat / n0_nat

            # Exact curvature a2(n) = d^2 eps / dW^2 |_{W=0} = -lam*W0^2 + alpha_eff*n^(2/3)
            a2_at_nmelt = -lam * W0**2 + alpha_eff * mp.cbrt(n_melt_nat) ** 2
            a2_below = -lam * W0**2 + alpha_eff * mp.cbrt(mp.mpf("0.99") * n_melt_nat) ** 2
            a2_above = -lam * W0**2 + alpha_eff * mp.cbrt(mp.mpf("1.01") * n_melt_nat) ** 2

            # Sample grid in units of n_melt
            ratios = (
                mp.mpf("0.001"),
                mp.mpf("0.1"),
                mp.mpf("0.5"),
                mp.mpf("0.9"),
                mp.mpf("0.99"),
                mp.mpf("1.0"),
                mp.mpf("1.01"),
                mp.mpf("1.5"),
                mp.mpf("2.0"),
                mp.mpf("5.0"),
            )
            rows = []
            for r in ratios:
                n_nat = r * n_melt_nat
                W = mp_solve_equilibrium_W(n_nat, zeta_val, exact, prefer_melted_at_spinodal=True)
                st = mp_thermo_state(
                    n_nat, W, zeta_val, W0=W0, lam=lam, g_s=g_s, q_phi=q_phi, g_omega=g_omega, d=d
                )
                rows.append((r, st))

            # Quantify the microscopic relativistic Fermi-log pocket right at n = n_melt
            # (d^2 eps/dW^2|_{W=0} = 0 at n_melt, while -d*g_s^4/(4 pi^2) W^2 ln(2 k_F/(g_s W))
            # creates a tiny W ~ 2 MeV minimum with |dE/E| ~ 1e-14 that vanishes by n/n_melt = 1 + 1e-5).
            W_pocket = mp_solve_equilibrium_W(n_melt_nat, zeta_val, exact, prefer_melted_at_spinodal=False)
            st_pocket = mp_thermo_state(
                n_melt_nat, W_pocket, zeta_val, W0=W0, lam=lam, g_s=g_s, q_phi=q_phi, g_omega=g_omega, d=d
            )
            st_melted_at_nmelt = mp_thermo_state(
                n_melt_nat, mp.mpf("0"), zeta_val, W0=W0, lam=lam, g_s=g_s, q_phi=q_phi, g_omega=g_omega, d=d
            )
            pocket_rel_depth = (st_pocket["eps"] - st_melted_at_nmelt["eps"]) / st_melted_at_nmelt["eps"]

            # Verify no positive minimum exists at n = (1 + 1e-5) * n_melt
            W_post_pocket = mp_solve_equilibrium_W(
                n_melt_nat * (mp.mpf("1") + mp.mpf("1e-5")),
                zeta_val,
                exact,
                prefer_melted_at_spinodal=False,
            )

            super_profiles[label] = {
                "multiplier": mult,
                "zeta": zeta_val,
                "zeta_A": zeta_val * q_phi**4,
                "zeta_RMF": mp.mpf("6") * zeta_val * (q_phi / g_omega) ** 4,
                "alpha_eff": alpha_eff,
                "n_melt_nat": n_melt_nat,
                "n_melt_fm3": n_melt_fm3,
                "n_melt_over_n0": n_melt_over_n0,
                "a2_at_nmelt": a2_at_nmelt,
                "a2_below": a2_below,
                "a2_above": a2_above,
                "rows": rows,
                "fermi_log_pocket_W_MeV": W_pocket,
                "fermi_log_pocket_rel_energy_depth": pocket_rel_depth,
                "W_unconstrained_at_1p1e5_nmelt": W_post_pocket,
            }

        # Negative control for causality & stability: zeta < 0
        zeta_neg = -zeta_melt
        zeta_A_neg = zeta_neg * q_phi**4
        mA_vac = q_phi * W0
        neg_cases = []
        # 3 distinct regimes of |zeta_A| A0^2 / m_A^2: 0.1 (superluminal), 0.5 (gradient/ghost), 2.0 (tachyonic/ghost)
        for regime, frac_str in (
            ("small_A0_superluminal", "0.1"),
            ("mid_A0_gradient_and_ghost", "0.5"),
            ("large_A0_tachyonic_and_ghost", "2.0"),
        ):
            frac = mp.mpf(frac_str)
            A0_test = mp.sqrt(frac * mA_vac**2 / abs(zeta_A_neg))
            M0_sq = mA_vac**2 + mp.mpf("3") * zeta_A_neg * A0_test**2
            MT_sq = mA_vac**2 + zeta_A_neg * A0_test**2
            c_L_sq = MT_sq / M0_sq
            neg_cases.append(
                {
                    "regime": regime,
                    "abs_zetaA_A0sq_over_mAsq": frac,
                    "A0_MeV": A0_test,
                    "M0_sq": M0_sq,
                    "MT_sq": MT_sq,
                    "c_L_sq": c_L_sq,
                    "violates_causality_or_stability": bool(M0_sq <= 0 or MT_sq <= 0 or c_L_sq > 1 or c_L_sq < 0),
                }
            )

        return {
            "dps": dps,
            "consts": consts,
            "B_bag": B_bag,
            "four_B_bag": four_B_bag,
            "zeta0_states": zeta0_states,
            "sub_threshold": {
                "zeta": zeta_half,
                "alpha_eff": alpha_half,
                "rows": sub_rows,
            },
            "super_profiles": super_profiles,
            "negative_control_causality": neg_cases,
        }


# ---------------------------------------------------------------------------
# Section 5: Closed-Form Single-Scalar W + zeta Saturation Scan (No-Go)
# ---------------------------------------------------------------------------

def _saturation_scan_point(
    y: mp.mpf,
    exact: ExactBaseline,
    E_sat: mp.mpf,
) -> dict[str, Any]:
    """Exact closed-form inversion for (lam, g_omega, zeta, K, n_melt) at fixed y = W(n0)/W0."""
    mp_p = _mp_params(exact)
    W0 = mp_p["W0"]
    g_s = mp_p["g_s"]
    q_phi = mp_p["q_phi"]
    d = mp_p["d"]
    n0 = mp_p["n0_nat"]

    W = y * W0
    Mstar = g_s * W
    EF = mp_fermi_energy(n0, Mstar, d)
    FF = mp_fermi_free_energy(n0, Mstar, d)
    PF = mp_fermi_pressure(n0, Mstar, d)
    ns = mp_scalar_density(n0, Mstar, d)

    # Exact trace-identity lock at P(n0) = 0, eps(n0) = E_sat * n0:
    # eps - 3P = lam * W0^4 * (1 - y^2) = E_sat * n0
    lam = (E_sat * n0) / (W0**4 * (mp.mpf("1") - y * y))
    m_sigma = mp.sqrt(mp.mpf("2") * lam) * W0

    # From W stationarity: X2 = q_phi^2 W^2 A0^2 = W * (lam * W * (W^2 - W0^2) + g_s * ns)
    X2 = W * (lam * W * (W * W - W0 * W0) + g_s * ns)
    # From Gauss + mu_B = E_sat: X2 + X4 = g_omega * A0 * n0 = (E_sat - EF) * n0
    X4 = (E_sat - EF) * n0 - X2

    U = lam * (W * W - W0 * W0) ** 2 / mp.mpf("4")
    P_res = PF - U + mp.mpf("0.5") * X2 + mp.mpf("0.25") * X4
    eps_res = (FF + U + mp.mpf("0.5") * X2 + mp.mpf("0.75") * X4) - E_sat * n0

    # Handle exact zeta = 0 boundary where X4 may be at working-precision rounding
    if abs(X4) <= mp.eps * mp.mpf("1e4") * max(mp.mpf("1"), abs(X2)):
        X4 = mp.mpf("0")

    physical = bool(X2 > 0 and X4 >= 0)
    if not physical:
        return {
            "y": y,
            "physical_causal": False,
            "lam": lam,
            "m_sigma_MeV": m_sigma,
            "X2_MeV4": X2,
            "X4_MeV4": X4,
            "P_residual_MeV4": P_res,
            "eps_residual_MeV4": eps_res,
        }

    A0 = mp.sqrt(X2) / (q_phi * W)
    g_omega = (E_sat - EF) / A0
    zeta = X4 / (q_phi**4 * A0**4) if X4 > 0 else mp.mpf("0")
    zeta_A = zeta * q_phi**4
    zeta_RMF = mp.mpf("6") * zeta * (q_phi / g_omega) ** 4

    # Exact analytical incompressibility K = 9 * n0 * (d mu_B / d n_B)|_{n0}
    st = mp_thermo_state(
        n0, W, zeta, W0=W0, lam=lam, g_s=g_s, q_phi=q_phi, g_omega=g_omega, d=d
    )
    K_MeV = mp.mpf("9") * n0 * st["dmu_dn"]

    # Exact melting density n_melt / n0 for this (lam, g_omega, zeta)
    c_s = (d / (mp.mpf("4") * mp.pi**2)) * (mp.mpf("6") * mp.pi**2 / d) ** (mp.mpf("2") / mp.mpf("3")) * g_s**2
    if zeta > 0:
        c_v = (g_omega / (zeta * q_phi)) ** (mp.mpf("2") / mp.mpf("3"))
        alpha_eff = c_s - c_v
        n_melt_over_n0 = ((lam * W0**2 / alpha_eff) ** (mp.mpf("3") / mp.mpf("2"))) / n0 if alpha_eff > 0 else None
        # Compare melted W=0 energy per baryon at n0
        st_melted = mp_thermo_state(
            n0, mp.mpf("0"), zeta, W0=W0, lam=lam, g_s=g_s, q_phi=q_phi, g_omega=g_omega, d=d
        )
        E_per_A_melted_n0 = st_melted["eps"] / n0
    else:
        alpha_eff = None
        n_melt_over_n0 = None
        E_per_A_melted_n0 = None

    return {
        "y": y,
        "physical_causal": True,
        "lam": lam,
        "m_sigma_MeV": m_sigma,
        "g_omega": g_omega,
        "A0_MeV": A0,
        "zeta": zeta,
        "zeta_A": zeta_A,
        "zeta_RMF": zeta_RMF,
        "K_MeV": K_MeV,
        "c_L_sq_at_n0": st["c_L_sq"],
        "n_melt_over_n0": n_melt_over_n0,
        "E_per_A_melted_at_n0_MeV": E_per_A_melted_n0,
        "P_residual_MeV4": P_res,
        "eps_residual_MeV4": eps_res,
        "trace_residual_MeV4": st["trace_res"],
        "w_residual_MeV3": st["w_res"],
    }


def _evaluate_saturation_nogo(exact: ExactBaseline, dps: int) -> dict[str, Any]:
    """Run closed-form single-scalar W + zeta nuclear saturation scan at given dps."""
    with mp.workdps(dps):
        E_sat = _mp_frac(exact.M_N - NUCLEAR_BINDING_REF_MEV)
        m_pi = _mp_frac(M_PI_REF_MEV)

        # Find exact upper causal boundary y_max where X4(y_max) = 0 (zeta = 0 limit)
        mp_p = _mp_params(exact)
        W0 = mp_p["W0"]
        g_s = mp_p["g_s"]
        d = mp_p["d"]
        n0 = mp_p["n0_nat"]

        def raw_X4(yv: mp.mpf) -> mp.mpf:
            Wv = yv * W0
            Ms = g_s * Wv
            EFv = mp_fermi_energy(n0, Ms, d)
            nsv = mp_scalar_density(n0, Ms, d)
            lamv = (E_sat * n0) / (W0**4 * (mp.mpf("1") - yv * yv))
            X2v = Wv * (lamv * Wv * (Wv * Wv - W0 * W0) + g_s * nsv)
            return (E_sat - EFv) * n0 - X2v

        lo, hi = mp.mpf("0.75"), mp.mpf("0.80")
        for _ in range(400):
            mid = (lo + hi) / mp.mpf("2")
            if raw_X4(mid) > 0:
                lo = mid
            else:
                hi = mid
            if (hi - lo) <= mp.eps * mp.mpf("8"):
                break
        y_zeta0 = (lo + hi) / mp.mpf("2")
        pt_zeta0 = _saturation_scan_point(y_zeta0, exact, E_sat)

        # Find exact y where K = 0 (spinodal boundary below which K < 0)
        lo_k, hi_k = mp.mpf("0.65"), mp.mpf("0.70")
        for _ in range(400):
            mid = (lo_k + hi_k) / mp.mpf("2")
            pt = _saturation_scan_point(mid, exact, E_sat)
            if pt["K_MeV"] < 0:
                lo_k = mid
            else:
                hi_k = mid
            if (hi_k - lo_k) <= mp.eps * mp.mpf("8"):
                break
        y_K0 = (lo_k + hi_k) / mp.mpf("2")

        scan_ys = (
            mp.mpf("0.55"),
            mp.mpf("0.60"),
            mp.mpf("0.65"),
            mp.mpf("0.70"),
            mp.mpf("0.71"),
            mp.mpf("0.72"),
            mp.mpf("0.75"),
            y_zeta0,
            mp.mpf("0.80"),  # Negative control: beyond causal boundary (zeta < 0)
        )
        points = [_saturation_scan_point(y, exact, E_sat) for y in scan_ys]

        # Independent finite-difference check of K(y=0.71) to validate analytical derivative
        pt_71 = next(pt for pt in points if pt["y"] == mp.mpf("0.71"))
        mp_p = _mp_params(exact)
        n0 = mp_p["n0_nat"]
        h = n0 * mp.mpf("1e-12")
        W_71 = mp.mpf("0.71") * mp_p["W0"]

        def mu_at_n(nn: mp.mpf) -> mp.mpf:
            # Newton-solve W near W_71 for the saturation parameters of y=0.71
            w = W_71
            lam_71 = pt_71["lam"]
            gom_71 = pt_71["g_omega"]
            z_71 = pt_71["zeta"]
            for _ in range(25):
                A0 = mp_solve_A0(nn, w, z_71, mp_p["q_phi"], gom_71)
                m = mp_p["g_s"] * w
                ns = mp_scalar_density(nn, m, mp_p["d"])
                f = lam_71 * w * (w * w - mp_p["W0"] ** 2) + mp_p["g_s"] * ns - mp_p["q_phi"] ** 2 * w * A0**2
                M0_sq = mp_p["q_phi"] ** 2 * w * w + mp.mpf("3") * z_71 * mp_p["q_phi"] ** 4 * A0**2
                dA0_dw = -mp.mpf("2") * mp_p["q_phi"] ** 2 * w * A0 / M0_sq
                df = (
                    lam_71 * (mp.mpf("3") * w * w - mp_p["W0"] ** 2)
                    + mp_p["g_s"] ** 2 * mp_dns_dm(nn, m, mp_p["d"])
                    - mp_p["q_phi"] ** 2 * A0**2
                    - mp.mpf("2") * mp_p["q_phi"] ** 2 * w * A0 * dA0_dw
                )
                step = f / df
                w -= step
                if abs(step) <= mp.eps * max(mp.mpf("1"), abs(w)) * mp.mpf("8"):
                    break
            A0 = mp_solve_A0(nn, w, z_71, mp_p["q_phi"], gom_71)
            return mp_fermi_energy(nn, mp_p["g_s"] * w, mp_p["d"]) + gom_71 * A0

        K_71_fd = mp.mpf("9") * n0 * (mu_at_n(n0 + h) - mu_at_n(n0 - h)) / (mp.mpf("2") * h)
        K_71_fd_rel_err = abs(K_71_fd - pt_71["K_MeV"]) / abs(pt_71["K_MeV"])

        all_causal_m_sigma_below_mpi = all(
            pt["m_sigma_MeV"] < m_pi for pt in points if pt["physical_causal"]
        )
        return {
            "dps": dps,
            "E_sat_MeV": E_sat,
            "y_zeta0_boundary": y_zeta0,
            "y_K0_boundary": y_K0,
            "pt_zeta0": pt_zeta0,
            "pt_y071": pt_71,
            "K_y071_fd_rel_err": K_71_fd_rel_err,
            "all_causal_m_sigma_below_mpi": all_causal_m_sigma_below_mpi,
            "points": points,
        }


# ---------------------------------------------------------------------------
# Serialization Helpers & Precision Cross-Check (60 dps vs 100 dps)
# ---------------------------------------------------------------------------

def _serialize_thermo_row(label_key: str, label_val: mp.mpf, st: dict[str, mp.mpf], exact: ExactBaseline) -> dict[str, Any]:
    with mp.workdps(80):
        hbar_c3 = _mp_frac(exact.hbar_c) ** 3
        W0 = _mp_frac(exact.W0)
        eps_scale = max(mp.mpf("1"), abs(st["eps"]))
        gauss_scale = max(mp.mpf("1"), abs(_mp_frac(exact.g_omega) * st["n_nat"]))
        w_scale = max(mp.mpf("1"), abs(_mp_frac(exact.lam) * W0**3))
        return {
            label_key: _fmt(label_val, 15),
            "n_fm3": _fmt(st["n_nat"] / hbar_c3, 25),
            "W_MeV": _fmt(st["W"], 28),
            "W_over_W0": _fmt(st["W"] / W0, 28),
            "A0_MeV": _fmt(st["A0"], 28),
            "epsilon_MeV_fm3": _fmt(st["eps"] / hbar_c3, 28),
            "pressure_MeV_fm3": _fmt(st["P"] / hbar_c3, 28),
            "mu_B_MeV": _fmt(st["mu_B"], 28),
            "cs2": _fmt(st["cs2"], 28),
            "c_L_sq": _fmt(st["c_L_sq"], 28),
            "trace_relative_residual": _fmt_sci(abs(st["trace_res"]) / eps_scale),
            "gauss_relative_residual": _fmt_sci(abs(st["gauss_res"]) / gauss_scale),
            "w_stationarity_relative_residual": _fmt_sci(abs(st["w_res"]) / w_scale),
            "legendre_relative_residual": _fmt_sci(abs(st["legendre_res"]) / eps_scale),
            "generator_relative_residual": _fmt_sci(abs(st["generator_res"]) / eps_scale),
        }


def _provenance() -> dict[str, str]:
    paths = {
        "source_complete": SOURCE_PATH,
        "producer": Path(__file__),
    }
    return {f"{k}_path": str(v.relative_to(REPO_ROOT)) for k, v in paths.items()} | {
        f"{k}_sha256": _sha256(v) for k, v in paths.items()
    }


def _fresh_result_payload(p: Params = PARAMS) -> dict[str, Any]:
    _require_live_baseline(p)
    exact = exact_baseline_from_live(p)

    # 1. Symbolic proofs
    symbolic = run_symbolic_proofs()

    # 2. Manufactured gauge control at 60 dps
    gauge_ctrl = manufactured_g4_gauge_control(exact, dps=60)

    # 3. Full phase suite and saturation no-go at BOTH 60 dps and 100 dps
    suite60 = _evaluate_phase_suite(exact, dps=60)
    suite100 = _evaluate_phase_suite(exact, dps=100)
    nogo60 = _evaluate_saturation_nogo(exact, dps=60)
    nogo100 = _evaluate_saturation_nogo(exact, dps=100)

    # Compute 60 vs 100 dps cross-check residuals
    with mp.workdps(100):
        def rel_diff(a: mp.mpf, b: mp.mpf) -> mp.mpf:
            return abs(a - b) / max(mp.mpf("1"), abs(b))

        dps_diffs = {
            "c_s": rel_diff(suite60["consts"]["c_s"], suite100["consts"]["c_s"]),
            "zeta_melt": rel_diff(suite60["consts"]["zeta_melt"], suite100["consts"]["zeta_melt"]),
            "zeta_A_melt": rel_diff(suite60["consts"]["zeta_A_melt"], suite100["consts"]["zeta_A_melt"]),
            "zeta_RMF_melt": rel_diff(suite60["consts"]["zeta_RMF_melt"], suite100["consts"]["zeta_RMF_melt"]),
            "n_melt_2x": rel_diff(
                suite60["super_profiles"]["zeta_2x_melt"]["n_melt_nat"],
                suite100["super_profiles"]["zeta_2x_melt"]["n_melt_nat"],
            ),
            "n_melt_5x": rel_diff(
                suite60["super_profiles"]["zeta_5x_melt"]["n_melt_nat"],
                suite100["super_profiles"]["zeta_5x_melt"]["n_melt_nat"],
            ),
            "W_0p5x_100n0": rel_diff(
                suite60["sub_threshold"]["rows"][-2][1]["W"],
                suite100["sub_threshold"]["rows"][-2][1]["W"],
            ),
            "W_2x_0p9nmelt": rel_diff(
                suite60["super_profiles"]["zeta_2x_melt"]["rows"][3][1]["W"],
                suite100["super_profiles"]["zeta_2x_melt"]["rows"][3][1]["W"],
            ),
            "K_y071": rel_diff(nogo60["pt_y071"]["K_MeV"], nogo100["pt_y071"]["K_MeV"]),
            "K_zeta0": rel_diff(nogo60["pt_zeta0"]["K_MeV"], nogo100["pt_zeta0"]["K_MeV"]),
            "m_sigma_y071": rel_diff(nogo60["pt_y071"]["m_sigma_MeV"], nogo100["pt_y071"]["m_sigma_MeV"]),
            "n_melt_over_n0_y071": rel_diff(
                nogo60["pt_y071"]["n_melt_over_n0"], nogo100["pt_y071"]["n_melt_over_n0"]
            ),
        }
        max_dps_diff = max(dps_diffs.values())

        # Collect worst-case trace, Gauss, stationarity, and CSS residuals across ALL 100-dps states
        all_100_states = (
            list(suite100["zeta0_states"])
            + [st for _, st in suite100["sub_threshold"]["rows"]]
            + [st for _, st in suite100["super_profiles"]["zeta_2x_melt"]["rows"]]
            + [st for _, st in suite100["super_profiles"]["zeta_5x_melt"]["rows"]]
        )
        max_trace_rel_100 = max(
            abs(st["trace_res"]) / max(mp.mpf("1"), abs(st["eps"])) for st in all_100_states
        )
        all_60_states = (
            list(suite60["zeta0_states"])
            + [st for _, st in suite60["sub_threshold"]["rows"]]
            + [st for _, st in suite60["super_profiles"]["zeta_2x_melt"]["rows"]]
            + [st for _, st in suite60["super_profiles"]["zeta_5x_melt"]["rows"]]
        )
        max_trace_rel_60 = max(
            abs(st["trace_res"]) / max(mp.mpf("1"), abs(st["eps"])) for st in all_60_states
        )
        max_gauss_rel_100 = max(
            abs(st["gauss_res"]) / max(mp.mpf("1"), abs(_mp_frac(exact.g_omega) * st["n_nat"]))
            for st in all_100_states
        )

        # Verify exact CSS limit (cs2 == 1/3 and eps - 3P == lam*W0^4) for all melted rows (r >= 1.0)
        melted_rows_100 = [
            st
            for prof in suite100["super_profiles"].values()
            for r, st in prof["rows"]
            if r >= mp.mpf("1.0")
        ]
        max_css_cs2_err = max(abs(st["cs2"] - mp.mpf("1") / mp.mpf("3")) for st in melted_rows_100)
        max_css_bag_rel_err = max(
            abs((st["eps"] - mp.mpf("3") * st["P"]) - suite100["four_B_bag"]) / suite100["four_B_bag"]
            for st in melted_rows_100
        )

        # Verify causality for all zeta > 0 states:
        #   W > 0, n > 0 => 1/3 < c_L^2 < 1
        #   W > 0, n == 0 => c_L^2 == 1
        #   W == 0, n > 0 => c_L^2 == 1/3
        zeta_pos_100_states = (
            [st for _, st in suite100["sub_threshold"]["rows"]]
            + [st for _, st in suite100["super_profiles"]["zeta_2x_melt"]["rows"]]
            + [st for _, st in suite100["super_profiles"]["zeta_5x_melt"]["rows"]]
        )
        causality_pos_pass = all(
            (st["M0_sq"] > 0 and st["MT_sq"] > 0)
            and (
                (st["W"] > 0 and st["n_nat"] > 0 and mp.mpf("1") / mp.mpf("3") < st["c_L_sq"] < mp.mpf("1"))
                or (st["W"] > 0 and st["n_nat"] == 0 and st["c_L_sq"] == mp.mpf("1"))
                or (st["W"] == 0 and st["n_nat"] > 0 and abs(st["c_L_sq"] - mp.mpf("1") / mp.mpf("3")) < mp.mpf("1e-90"))
            )
            for st in zeta_pos_100_states
        )
        neg_ctrl_all_violate = all(
            case["violates_causality_or_stability"] for case in suite100["negative_control_causality"]
        )

        # Verify sub-threshold behavior (0.5 * zeta_melt): W > 0 everywhere and W(1000 n0) > W0
        sub_rows_100 = suite100["sub_threshold"]["rows"]
        sub_no_melting_pass = (
            suite100["sub_threshold"]["alpha_eff"] < 0
            and all(st["W"] > 0 for _, st in sub_rows_100)
            and sub_rows_100[-1][1]["W"] > _mp_frac(exact.W0)
        )

        # Verify super-threshold behavior (2x, 5x): W > 0 below n_melt, W == 0 at/above n_melt
        super_melting_pass = True
        for prof in suite100["super_profiles"].values():
            cond = (
                prof["alpha_eff"] > 0
                and abs(prof["a2_at_nmelt"]) < mp.mpf("1e-90")
                and prof["a2_below"] < 0
                and prof["a2_above"] > 0
                and all(st["W"] > 0 for r, st in prof["rows"] if r < mp.mpf("1.0"))
                and all(st["W"] == 0 for r, st in prof["rows"] if r >= mp.mpf("1.0"))
                and prof["W_unconstrained_at_1p1e5_nmelt"] == 0
            )
            super_melting_pass = super_melting_pass and cond

        # Verify single-scalar no-go gates
        nogo_pass = (
            nogo100["all_causal_m_sigma_below_mpi"]
            and nogo100["pt_y071"]["K_MeV"] > mp.mpf("220")
            and nogo100["pt_y071"]["K_MeV"] < mp.mpf("235")
            and nogo100["pt_zeta0"]["K_MeV"] > mp.mpf("690")
            and nogo100["pt_zeta0"]["K_MeV"] < mp.mpf("695")
            and nogo100["pt_y071"]["m_sigma_MeV"] < mp.mpf("80")
            and nogo100["pt_y071"]["n_melt_over_n0"] < mp.mpf("1")
            and nogo100["K_y071_fd_rel_err"] < mp.mpf("1e-18")
            and not nogo100["points"][-1]["physical_causal"]
        )

        gates = {
            "symbolic_proofs_exact": symbolic["all_symbolic_proofs_exact"],
            "manufactured_g4_gauge_invariance": gauge_ctrl["pass"],
            "precision_60_vs_100_dps_agreement": bool(max_dps_diff < mp.mpf("1e-50")),
            "trace_identity_below_1e25": bool(max_trace_rel_60 < mp.mpf("1e-50") and max_trace_rel_100 < mp.mpf("1e-90")),
            "gauss_constraint_closure": bool(max_gauss_rel_100 < mp.mpf("1e-90")),
            "causality_and_stability_zeta_positive": causality_pos_pass,
            "negative_control_zeta_negative_fails_causality": neg_ctrl_all_violate,
            "sub_threshold_0p5x_no_melting": sub_no_melting_pass,
            "super_threshold_2x_5x_melting_and_css": bool(
                super_melting_pass and max_css_cs2_err < mp.mpf("1e-90") and max_css_bag_rel_err < mp.mpf("1e-90")
            ),
            "single_scalar_nogo_verified": nogo_pass,
        }

        if not all(gates.values()):
            failed = [k for k, v in gates.items() if not v]
            raise QuarticVectorMeltingError(f"one or more audit gates failed: {failed}")

        # Format output JSON payload cleanly using 100-dps authoritative values
        def fmt_super_profile(prof: dict[str, Any]) -> dict[str, Any]:
            return {
                "zeta_multiplier": _fmt(prof["multiplier"], 10),
                "zeta": _fmt(prof["zeta"], 30),
                "zeta_A": _fmt(prof["zeta_A"], 30),
                "zeta_RMF": _fmt(prof["zeta_RMF"], 30),
                "alpha_eff": _fmt(prof["alpha_eff"], 30),
                "n_melt_MeV3": _fmt(prof["n_melt_nat"], 30),
                "n_melt_fm3": _fmt(prof["n_melt_fm3"], 30),
                "n_melt_over_n0": _fmt(prof["n_melt_over_n0"], 30),
                "spinodal_curvature_at_nmelt_MeV2": _fmt_sci(prof["a2_at_nmelt"]),
                "spinodal_curvature_at_0p99_nmelt_MeV2": _fmt(prof["a2_below"], 20),
                "spinodal_curvature_at_1p01_nmelt_MeV2": _fmt(prof["a2_above"], 20),
                "relativistic_fermi_log_pocket": {
                    "note": (
                        "At the exact Landau spinodal n_melt, the relativistic Fermi-sea term "
                        "-d*g_s^4/(4*pi^2)*W^2*ln(2*k_F/(g_s*W)) creates a microscopic pocket "
                        "with relative energy depth O(1e-14) that vanishes by n/n_melt = 1 + 1e-5."
                    ),
                    "W_pocket_at_nmelt_MeV": _fmt(prof["fermi_log_pocket_W_MeV"], 15),
                    "relative_energy_depth": _fmt_sci(prof["fermi_log_pocket_rel_energy_depth"]),
                    "W_unconstrained_at_1p1e5_nmelt_MeV": _fmt(prof["W_unconstrained_at_1p1e5_nmelt"], 15),
                },
                "rows": [_serialize_thermo_row("n_over_n_melt", r, st, exact) for r, st in prof["rows"]],
            }

        def fmt_nogo_point(pt: dict[str, Any]) -> dict[str, Any]:
            base = {
                "y": _fmt(pt["y"], 20),
                "physical_causal": pt["physical_causal"],
                "lambda": _fmt(pt["lam"], 25),
                "m_sigma_MeV": _fmt(pt["m_sigma_MeV"], 25),
                "P_residual_MeV4": _fmt_sci(pt["P_residual_MeV4"]),
                "eps_residual_MeV4": _fmt_sci(pt["eps_residual_MeV4"]),
            }
            if not pt["physical_causal"]:
                base["X2_MeV4"] = _fmt(pt["X2_MeV4"], 20)
                base["X4_MeV4"] = _fmt(pt["X4_MeV4"], 20)
                base["failure_reason"] = "X4 < 0 forces zeta < 0 (acausal / unstable)"
                return base
            base.update(
                {
                    "g_omega": _fmt(pt["g_omega"], 25),
                    "A0_MeV": _fmt(pt["A0_MeV"], 25),
                    "zeta": _fmt(pt["zeta"], 25),
                    "zeta_A": _fmt(pt["zeta_A"], 25),
                    "zeta_RMF": _fmt(pt["zeta_RMF"], 25),
                    "K_MeV": _fmt(pt["K_MeV"], 25),
                    "c_L_sq_at_n0": _fmt(pt["c_L_sq_at_n0"], 25),
                    "n_melt_over_n0": (
                        _fmt(pt["n_melt_over_n0"], 25) if pt["n_melt_over_n0"] is not None else "INF_ZETA_ZERO"
                    ),
                    "E_per_A_melted_at_n0_MeV": (
                        _fmt(pt["E_per_A_melted_at_n0_MeV"], 25)
                        if pt["E_per_A_melted_at_n0_MeV"] is not None
                        else "INF_ZETA_ZERO"
                    ),
                    "m_sigma_below_mpi": bool(pt["m_sigma_MeV"] < _mp_frac(M_PI_REF_MEV)),
                    "n_melt_below_n0": bool(
                        pt["n_melt_over_n0"] is not None and pt["n_melt_over_n0"] < mp.mpf("1")
                    ),
                }
            )
            return base

        return {
            "schema": SCHEMA,
            "status": STATUS,
            "evidence_weight": EVIDENCE_WEIGHT,
            "provenance": _provenance(),
            "live_baseline_inputs": {
                "source": "verification/source_complete_solution_audit.py::PARAMS",
                "W0_MeV": _fmt(exact.W0, 15),
                "lambda": _fmt(exact.lam, 15),
                "M_N_MeV": _fmt(exact.M_N, 15),
                "m_omega_MeV": _fmt(exact.m_omega, 15),
                "g_omega": _fmt(exact.g_omega, 15),
                "n0_fm3": _fmt(exact.n0_fm3, 15),
                "hbar_c_MeV_fm": _fmt(exact.hbar_c, 15),
                "degeneracy": exact.degeneracy,
                "derived_g_s": _fmt(exact.g_s, 30),
                "derived_q_phi": _fmt(exact.q_phi, 30),
                "derived_m_sigma_MeV": _fmt(mp.sqrt(_mp_frac(exact.m_sigma_squared)), 30),
                "derived_B_bag_MeV4": _fmt(suite100["B_bag"], 30),
                "derived_4B_bag_MeV_fm3": _fmt(suite100["four_B_bag"] / (_mp_frac(exact.hbar_c) ** 3), 30),
            },
            "calibration_targets_vs_forward_predictions": {
                "law1_baseline_audit": {
                    "calibration_targets": [
                        "Live source_complete baseline inputs only (W0, lambda, M_N, m_omega, g_omega, n0, hbar_c, d); zero new fits performed."
                    ],
                    "forward_predictions": [
                        "Exact Weyl invariance of L_g4 and invariance of equilibrium trace identity eps - 3P = 4U(W) - W U'(W)",
                        "Exact melting threshold zeta_melt, zeta_A_melt, zeta_RMF_melt and spinodal n_melt(zeta)",
                        "Exact MIT-bag / CSS melted phase P = (eps - 4 B_bag)/3 with c_s^2 = 1/3",
                        "Exact longitudinal vector speed bound 1/3 < c_L^2 <= 1 for zeta > 0 and acausal/unstable breakdown for zeta < 0",
                    ],
                },
                "single_scalar_nogo_scan": {
                    "calibration_targets": [
                        "Hypothetical symmetric nuclear matter saturation point at n0 = 0.16 fm^-3: P(n0) = 0, E/A(n0) = M_N - 16 MeV = 923 MeV, parameterized by y = W(n0)/W0 with fixed W0=859 MeV, g_s=M_N/W0, q_phi=m_omega/W0."
                    ],
                    "forward_predictions": [
                        "Rigid trace-identity lock m_sigma(y) = sqrt(2*E_sat*n0 / (W0^2*(1-y^2))) < 89 MeV < m_pi across the entire causal domain",
                        "Exact incompressibility curve K(y): lowered from 692.07 MeV (zeta=0) to 227.02 MeV at y=0.71",
                        "Premature melted-vacuum spinodal n_melt(y)/n0 < 1 (0.3321 at y=0.71), proving a second chiral scalar condensate is required",
                    ],
                },
            },
            "symbolic_proofs": symbolic,
            "manufactured_gauge_control": gauge_ctrl,
            "precision_cross_check_60_vs_100_dps": {
                "max_relative_difference": _fmt_sci(max_dps_diff),
                "per_quantity_relative_differences": {k: _fmt_sci(v) for k, v in dps_diffs.items()},
                "max_trace_identity_relative_residual_60dps": _fmt_sci(max_trace_rel_60),
                "max_trace_identity_relative_residual_100dps": _fmt_sci(max_trace_rel_100),
                "max_gauss_relative_residual_100dps": _fmt_sci(max_gauss_rel_100),
                "max_melted_css_cs2_error_100dps": _fmt_sci(max_css_cs2_err),
                "max_melted_css_bag_relative_error_100dps": _fmt_sci(max_css_bag_rel_err),
            },
            "melting_threshold": {
                "c_s": _fmt(suite100["consts"]["c_s"], 30),
                "zeta_melt": _fmt(suite100["consts"]["zeta_melt"], 30),
                "zeta_A_melt": _fmt(suite100["consts"]["zeta_A_melt"], 30),
                "zeta_RMF_melt": _fmt(suite100["consts"]["zeta_RMF_melt"], 30),
                "formulas": {
                    "c_s": "(d / (4*pi^2)) * (6*pi^2 / d)^(2/3) * g_s^2",
                    "zeta_melt": "g_omega / (q_phi * c_s^(3/2))",
                    "zeta_A_melt": "zeta_melt * q_phi^4",
                    "zeta_RMF_melt": "6 * zeta_melt * (q_phi / g_omega)^4",
                    "alpha_eff": "c_s - (g_omega / (zeta * q_phi))^(2/3)",
                    "n_melt": "(lambda * W0^2 / alpha_eff)^(3/2)",
                },
            },
            "phase_profiles": {
                "zeta_0p5x_melt_no_melting": {
                    "zeta_multiplier": "0.5",
                    "zeta": _fmt(suite100["sub_threshold"]["zeta"], 30),
                    "alpha_eff": _fmt(suite100["sub_threshold"]["alpha_eff"], 30),
                    "melts": False,
                    "rows": [
                        _serialize_thermo_row("n_over_n0", x, st, exact)
                        for x, st in suite100["sub_threshold"]["rows"]
                    ],
                },
                "zeta_2x_melt": fmt_super_profile(suite100["super_profiles"]["zeta_2x_melt"]),
                "zeta_5x_melt": fmt_super_profile(suite100["super_profiles"]["zeta_5x_melt"]),
            },
            "causality_and_stability": {
                "positive_zeta_bound": "1/3 < c_L^2 <= 1 for W > 0, zeta > 0 (and c_L^2 = 1/3 exactly in melted phase W = 0)",
                "positive_zeta_all_states_pass": causality_pos_pass,
                "negative_control_zeta_negative": [
                    {
                        "regime": case["regime"],
                        "abs_zetaA_A0sq_over_mAsq": _fmt(case["abs_zetaA_A0sq_over_mAsq"], 10),
                        "A0_MeV": _fmt(case["A0_MeV"], 20),
                        "M0_sq_MeV2": _fmt(case["M0_sq"], 20),
                        "MT_sq_MeV2": _fmt(case["MT_sq"], 20),
                        "c_L_sq": _fmt(case["c_L_sq"], 20),
                        "violates_causality_or_stability": case["violates_causality_or_stability"],
                    }
                    for case in suite100["negative_control_causality"]
                ],
            },
            "single_scalar_saturation_nogo": {
                "reference_E_sat_MeV": _fmt(nogo100["E_sat_MeV"], 15),
                "reference_m_pi_MeV": _fmt(M_PI_REF_MEV, 15),
                "y_zeta0_causal_upper_bound": _fmt(nogo100["y_zeta0_boundary"], 25),
                "y_K0_spinodal_lower_bound": _fmt(nogo100["y_K0_boundary"], 25),
                "zeta0_limit_summary": {
                    "y": _fmt(nogo100["pt_zeta0"]["y"], 20),
                    "m_sigma_MeV": _fmt(nogo100["pt_zeta0"]["m_sigma_MeV"], 20),
                    "K_MeV": _fmt(nogo100["pt_zeta0"]["K_MeV"], 20),
                    "g_omega": _fmt(nogo100["pt_zeta0"]["g_omega"], 20),
                },
                "realistic_K_benchmark_y071": {
                    "y": "0.71",
                    "m_sigma_MeV": _fmt(nogo100["pt_y071"]["m_sigma_MeV"], 20),
                    "K_MeV": _fmt(nogo100["pt_y071"]["K_MeV"], 20),
                    "K_fd_relative_check": _fmt_sci(nogo100["K_y071_fd_rel_err"]),
                    "g_omega": _fmt(nogo100["pt_y071"]["g_omega"], 20),
                    "zeta": _fmt(nogo100["pt_y071"]["zeta"], 20),
                    "zeta_RMF": _fmt(nogo100["pt_y071"]["zeta_RMF"], 20),
                    "n_melt_over_n0": _fmt(nogo100["pt_y071"]["n_melt_over_n0"], 20),
                },
                "all_causal_points_have_m_sigma_below_mpi": nogo100["all_causal_m_sigma_below_mpi"],
                "structural_conclusion": (
                    "Because L_g4 is classically Weyl-invariant, the trace identity "
                    "eps - 3P = lam*W0^4*(1-y^2) = 1/2*m_sigma^2*W0^2*(1-y^2) is immune to zeta. "
                    "At nuclear saturation (P=0, eps=E_sat*n0), forcing g_s = M_N/W0 with W0=859 MeV "
                    "rigidly locks m_sigma < 89 MeV < m_pi across the entire causal domain zeta >= 0. "
                    "Although zeta lowers K from 692.07 MeV (zeta=0) to 227.02 MeV at y=0.71, it "
                    "simultaneously forces m_sigma = 78.75 MeV < m_pi and n_melt = 0.3321*n0 < n0, "
                    "proving mathematically that a second (chiral) scalar condensate is required."
                ),
                "scan_rows": [fmt_nogo_point(pt) for pt in nogo100["points"]],
            },
            "gates": gates,
            "honest_caveats": [
                "Pure mathematical / structural audit (evidence_weight = 0.0); no empirical likelihood or observational evidence is claimed.",
                "L_g4 = +(zeta/4)*(g_mu g^mu)^2 is dimension-4 in the polar/Stueckelberg representation (and unitary gauge), but is non-polynomial in Cartesian Phi at Phi = 0; UV completion above the Stueckelberg cutoff is outside this EFT audit.",
                "At the exact Landau spinodal n_melt, the relativistic Fermi-sea logarithm -d*g_s^4/(4*pi^2)*W^2*ln(2*k_F/(g_s*W)) produces a microscopic pocket of relative energy depth O(1e-14) at W ~ 2-3 MeV that vanishes by n/n_melt = 1 + 1e-5; for all n >= (1 + 1e-5)*n_melt, W = 0 is the unique global minimum.",
                "At the live baseline (lam = 1.05, W0 = 859 MeV), super-threshold melting occurs only at ultra-high densities (n_melt ~ 3951 n0 at 2*zeta_melt, 1666 n0 at 5*zeta_melt), far above neutron-star densities.",
                "The single-scalar W + zeta nuclear saturation scan proves a structural no-go: lowering K to ~227 MeV forces m_sigma ~ 79 MeV < m_pi and n_melt < n0, demonstrating that realistic nuclear saturation requires a second (chiral) scalar condensate.",
            ],
        }


def build_result(p: Params = PARAMS) -> dict[str, Any]:
    result = _fresh_result_payload(p)
    validate_result(result, p=p, check_provenance=False)
    return result


def validate_result(payload: Any, p: Params = PARAMS, *, check_provenance: bool = True) -> None:
    """Fail-closed validation requiring an exact fresh canonical derivation."""
    if not isinstance(payload, Mapping):
        raise QuarticVectorMeltingError("result payload must be a JSON object")
    if payload.get("schema") != SCHEMA:
        raise QuarticVectorMeltingError(f"unexpected schema: {payload.get('schema')!r}")
    if payload.get("status") != STATUS:
        raise QuarticVectorMeltingError(f"unexpected status: {payload.get('status')!r}")
    if payload.get("evidence_weight") != EVIDENCE_WEIGHT:
        raise QuarticVectorMeltingError(f"evidence_weight must be {EVIDENCE_WEIGHT}")
    expected = _fresh_result_payload(p)
    if _canonical_json(payload) != _canonical_json(expected):
        raise QuarticVectorMeltingError("payload does not match exact fresh canonical derivation")
    if check_provenance and payload.get("provenance") != _provenance():
        raise QuarticVectorMeltingError("provenance digest mismatch")


def write_result(result: Mapping[str, Any], path: Path = RESULT_PATH) -> None:
    validate_result(result)
    path.write_bytes(_canonical_json(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT_PATH, help="JSON output path")
    parser.add_argument("--validate", type=Path, help="validate an existing result JSON without writing")
    args = parser.parse_args(argv)
    if args.validate is not None:
        try:
            payload = json.loads(args.validate.read_text(encoding="utf-8"))
            validate_result(payload)
        except (OSError, json.JSONDecodeError, QuarticVectorMeltingError) as exc:
            print(f"VALIDATION_FAILED: {exc}", file=sys.stderr)
            return 1
        print("VALIDATION_PASS")
        return 0
    try:
        result = build_result()
        write_result(result, args.output)
    except QuarticVectorMeltingError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 1
    print(f"STATUS={result['status']} OUTPUT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
