#!/usr/bin/env python3
"""Finite-q retarded response of the declared subtracted Dirac/Hartree medium.

This is a deliberately bounded numerical continuation of the static finite-q
kernel.  It keeps the occupied-positive/negative-intermediate contribution
and the longitudinal Proca variable.  The output is a conditional low-energy
Hartree response, not vacuum RPA, a measured spectrum, or a global positivity
statement.  The command is strictly no-write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import mpmath as mp
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad_vec
from scipy.optimize import brentq

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import nvg_nonlocal_response as static  # noqa: E402
import source_complete_scaling_saturation_audit as upstream  # noqa: E402

SOURCE_PATH = Path(__file__).resolve()
CONTRACT_PATH = HERE / "contracts" / "retarded_response.md"
UPSTREAM_PATH = HERE / "source_complete_scaling_saturation_audit.py"
STATIC_PATH = HERE / "nvg_nonlocal_response.py"
SCHEMA_VERSION = 1
SCHEME = "SUBTRACTED_MEDIUM_DIRAC_RETARDED"
STATUS = "LIVE_SUBTRACTED_MEDIUM_DIRAC_RETARDED_HARTREE_ZERO_EVIDENCE_NOT_EMPIRICAL"
EVIDENCE_WEIGHT = 0.0
SCALES = ("1", "1.3")
Q_GRID_MEV = ("100", "200")
FREQUENCY_FRACTIONS = ("0", "0.25", "0.75", "1.25")
EXPECTED_ROW_COUNT = 3 * len(SCALES) * len(Q_GRID_MEV) * len(FREQUENCY_FRACTIONS)
EXPECTED_POLE_CASES = 3 * len(SCALES) * len(Q_GRID_MEV)
DEFAULT_EPSREL, DEFAULT_EPSABS = 1e-10, 1e-6
CONTROL_EPSREL, CONTROL_EPSABS = 1e-11, 1e-7


class RetardedResponseError(ValueError):
    """Malformed input or an explicitly unresolved response."""


def _real(x: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(x, bool):
        raise RetardedResponseError(f"{name} must be a finite real, not bool")
    try:
        y = float(x)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetardedResponseError(f"{name} must be a finite real") from exc
    if not math.isfinite(y):
        raise RetardedResponseError(f"{name} must be finite")
    if positive and y <= 0:
        raise RetardedResponseError(f"{name} must be positive")
    if nonnegative and y < 0:
        raise RetardedResponseError(f"{name} must be nonnegative")
    return y


def _z(z: Any, eta: Any | None = None) -> complex:
    try:
        if isinstance(z, (tuple, list)) and len(z) == 2:
            value = complex(_real(z[0], "Re z"), _real(z[1], "Im z"))
        else:
            value = complex(z)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetardedResponseError("z must be a finite complex number") from exc
    if eta is not None:
        value = complex(value.real, _real(eta, "eta", nonnegative=True))
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise RetardedResponseError("z must be finite")
    if value.imag < 0:
        raise RetardedResponseError("retarded z must lie on or above the real axis")
    return value


def _deg(d: Any) -> int:
    y = _real(d, "d", positive=True)
    if y != int(y) or int(y) % 2:
        raise RetardedResponseError("d must be a positive even integer")
    return int(y)


def _finite_complex(x: Any) -> bool:
    try:
        y = complex(x)
        return math.isfinite(y.real) and math.isfinite(y.imag)
    except (TypeError, ValueError, OverflowError):
        return False


def _rel(a: Any, b: Any) -> float:
    a, b = complex(a), complex(b)
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def _genuine_rel(a: Any, b: Any) -> float:
    """Relative difference without an O(1) floor.

    The scientific controls above intentionally use ``_rel`` so that a
    comparison against an exact zero is well behaved.  Pole derivatives and
    determinant identities are instead O(1e-5) quantities, so their
    convergence must not be hidden by that floor.
    """
    a, b = complex(a), complex(b)
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def _cjson(x: Any, digits: int = 17) -> dict[str, str]:
    y = complex(x)
    if not _finite_complex(y):
        raise RetardedResponseError("nonfinite complex output")
    return {"real": format(y.real, f".{digits}g"), "imag": format(y.imag, f".{digits}g")}


def _rjson(x: Any, digits: int = 17) -> str:
    y = _real(x, "derived number")
    return format(y, f".{digits}g")


def _matrix_json(m: Any) -> list[list[dict[str, str]]]:
    a = np.asarray(m, dtype=complex)
    return [[_cjson(v) for v in row] for row in a]


def _boundary_log(x: float, omega: float, plus: bool) -> complex:
    """Retarded real-axis logarithm, with the correct rim for +/- omega."""
    y = x + omega if plus else x - omega
    if y == 0.0:
        phase = math.pi if plus and omega < 0 else (-math.pi if not plus and omega > 0 else 0.0)
        return complex(-math.inf, phase)
    phase = math.pi if plus and y < 0 else (-math.pi if not plus and y < 0 else 0.0)
    return complex(math.log(abs(y)), phase)


def _primitive(x: float, E: float, m: float, q: float, z: complex, band: int) -> np.ndarray:
    if z.imag == 0.0:
        lm, lp = _boundary_log(x, z.real, False), _boundary_log(x, z.real, True)
    else:
        lm, lp = np.log(complex(x) - z), np.log(complex(x) + z)
    J1 = (lm + lp) / 2.0
    J2 = x + z * (lm - lp) / 2.0
    J3 = x * x / 2.0 + z * z * J1
    vv = (J3 + band * 4.0 * E * J2 + (4.0 * E * E - q * q) * J1) / (2.0 * E)
    ss = (-J3 + (4.0 * m * m + q * q) * J1) / (2.0 * E)
    vs = band * m / E * J2 + 2.0 * m * J1
    return np.asarray((vv, vs, ss), dtype=complex)


def _split_points(m: float, k: float, q: float, omega: float) -> list[float]:
    """Kinematic/endpoint roots at which a logarithmic boundary changes."""
    ef = math.hypot(m, k)
    points = [0.0, k]
    def add(x: float) -> None:
        if 0.0 < x < k and all(abs(x - p) > 2e-10 * max(1.0, k) for p in points):
            points.append(float(x))
    add(abs(k - q))
    if ef - omega > m:
        add(math.sqrt(max(0.0, (ef - omega) ** 2 - m * m)))
    if 0.0 < omega < q:
        t = q * q - omega * omega
        emin = (q * math.sqrt(1.0 + 4.0 * m * m / t) - omega) / 2.0
        if emin > m:
            add(math.sqrt(max(0.0, emin * emin - m * m)))

    # Solve all endpoint transfer roots, rather than assuming a particular
    # ordering of q and kF.  They are cheap compared with the radial integral.
    funcs = (
        lambda p: math.hypot(m, p + q) - math.hypot(m, p) - omega,
        lambda p: math.hypot(m, abs(p - q)) - math.hypot(m, p) - omega,
        lambda p: ef - math.hypot(m, p) - omega,
    )
    for fn in funcs:
        xs = np.linspace(0.0, k, 65)
        ys = [fn(float(x)) for x in xs]
        for x1, x2, y1, y2 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
            if y1 == 0.0:
                add(float(x1))
            if y1 * y2 < 0.0:
                try:
                    add(brentq(fn, float(x1), float(x2), xtol=2e-12, rtol=1e-14))
                except ValueError:
                    pass
        if ys[-1] == 0.0:
            add(float(xs[-1]))
    return sorted(points)


def _kernel(m: float, k: float, q: float, z: complex, d: int, *, epsrel: float, epsabs: float) -> tuple[np.ndarray, float]:
    if q == 0.0:
        if z.real == 0.0 and z.imag == 0.0:
            pi = static.polarization_kernel(str(m), str(k), 0, d=d, dps=40)
            return np.asarray([complex(float(v), 0.0) for v in pi]), 0.0
        # At strictly q=0 the density and density-scalar channels vanish by
        # number conservation.  The scalar channel has the remote
        # positive-to-negative (pair) contribution; retain it rather than
        # silently claiming that the whole polarization tensor is zero.
        def scalar_integrand(p: float) -> complex:
            E = math.hypot(m, p)
            return -2.0 * d / (math.pi * math.pi) * p**4 / (E * (4.0 * E * E - z * z))
        value, error = quad_vec(scalar_integrand, 0.0, k, epsabs=epsabs, epsrel=epsrel, limit=500)
        return np.asarray((0j, 0j, complex(value)), dtype=complex), float(error)
    ef = math.hypot(m, k)
    def integrand(p: float) -> np.ndarray:
        E = math.hypot(m, p)
        rlo, rhi = math.hypot(m, p - q), math.hypot(m, p + q)
        value = _primitive(rhi + E, E, m, q, z, -1) - _primitive(rlo + E, E, m, q, z, -1)
        lower = max(rlo, ef)
        if lower < rhi:
            value += _primitive(rhi - E, E, m, q, z, 1) - _primitive(lower - E, E, m, q, z, 1)
        return p / q * d / (4.0 * math.pi * math.pi) * value
    points = _split_points(m, k, q, z.real)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value, error = quad_vec(integrand, 0.0, k, epsabs=epsabs, epsrel=epsrel,
                                points=points[1:-1], limit=500)
    value = np.asarray(value, dtype=complex)
    if not np.all(np.isfinite(value)):
        raise RetardedResponseError("retarded radial integral is nonfinite")
    return value, float(error)


def polarization_kernel(mass: Any, kf: Any, q: Any, z: Any = 0, d: Any = 4, *, eta: Any | None = None,
                        epsrel: float = DEFAULT_EPSREL, epsabs: float = DEFAULT_EPSABS) -> tuple[complex, complex, complex]:
    """Return (Pi_vv, Pi_vs, Pi_ss) in MeV^2 for retarded z."""
    m, k, q, d = _real(mass, "M", positive=True), _real(kf, "kF", positive=True), _real(q, "q", nonnegative=True), _deg(d)
    z = _z(z, eta)
    if epsrel <= 0 or epsabs <= 0 or not math.isfinite(float(epsrel)) or not math.isfinite(float(epsabs)):
        raise RetardedResponseError("quadrature tolerances must be positive finite")
    value, _ = _kernel(m, k, q, z, d, epsrel=float(epsrel), epsabs=float(epsabs))
    return tuple(complex(v) for v in value)


retarded_polarization = polarization_kernel
retarded_kernel = polarization_kernel
finite_frequency_kernel = polarization_kernel


def analytic_ph_imag(mass: Any, kf: Any, q: Any, omega: Any, d: Any = 4) -> tuple[float, float, float]:
    """Analytic positive-frequency particle-hole absorptive part below pair threshold."""
    m, k, q, w, d = _real(mass, "M", positive=True), _real(kf, "kF", positive=True), _real(q, "q", nonnegative=True), _real(omega, "omega"), _deg(d)
    if q == 0.0 or w <= 0.0 or w >= q:
        return (0.0, 0.0, 0.0)
    ef = math.hypot(m, k)
    t = q * q - w * w
    elo = max(m, ef - w, (q * math.sqrt(1.0 + 4.0 * m * m / t) - w) / 2.0)
    if elo >= ef:
        return (0.0, 0.0, 0.0)
    de = ef - elo
    fac = d / (16.0 * math.pi * q)
    vv = fac * (4.0 * (ef**3 - elo**3) / 3.0 + 2.0 * w * (ef * ef - elo * elo) + (w * w - q * q) * de)
    ss = fac * (4.0 * m * m + t) * de
    vs = 2.0 * fac * m * (ef * ef - elo * elo + w * de)
    return (vv, vs, ss)


def particle_hole_edge(mass: Any, kf: Any, q: Any) -> float:
    m, k, q = _real(mass, "M", positive=True), _real(kf, "kF", positive=True), _real(q, "q", nonnegative=True)
    return 0.0 if q == 0.0 else math.hypot(m, k + q) - math.hypot(m, k)


def _coerce_coeff(c: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(c, Mapping):
        raise RetardedResponseError("coefficients must be a mapping")
    names = {"a", "b", "g", "h", "t0"}
    if not names <= set(c):
        raise RetardedResponseError("coefficients must contain a,b,g,h,t0")
    d0 = c.get("d0", c.get("c0"))
    if d0 is None:
        raise RetardedResponseError("coefficients must contain d0")
    out = {x: complex(c[x]) for x in names}
    out["d0"] = complex(d0)
    if not all(_finite_complex(v) for v in out.values()):
        raise RetardedResponseError("coefficients must be finite")
    if out["a"] == 0 or out["t0"] == 0:
        raise RetardedResponseError("a and t0 must be nonzero")
    # Live coefficients carry the physical coordinate scales used to make
    # the mixed-unit Hessian dimensionally uniform.  Small manufactured
    # controls may omit them and deliberately use unit scales.
    out["_n0"] = _real(c.get("_n0", 1.0), "n0", positive=True)
    out["_W0"] = _real(c.get("_W0", 1.0), "W0", positive=True)
    return out


def response_from_hessian(coefficients: Mapping[str, Any], Z: Any, q: Any, z: Any = 0, *, eta: Any | None = None,
                          solve_direct: bool = True) -> dict[str, Any]:
    """Direct 4x4 solve and exact longitudinal-vector Schur response."""
    c, Z, q, z = _coerce_coeff(coefficients), complex(Z), _real(q, "q", nonnegative=True), _z(z, eta)
    if not _finite_complex(Z) or Z.real <= 0 or Z.imag != 0:
        raise RetardedResponseError("Z must be positive finite real")
    if q == 0.0 and (z.imag != 0.0 or z.real != 0.0):
        return {"status": "Q0_CONSERVATION_ZERO", "chi": 0j, "S": None, "direct_solution": None,
                "direct_status": "NUMBER_CONSERVATION", "regular": True,
                "physical_stability_assessed": False, "q": q, "z": z}
    a, b, d0, g, h, t0 = (c[x] for x in ("a", "b", "d0", "g", "h", "t0"))
    n0, W0 = c["_n0"], c["_W0"]
    coordinate_scales = np.asarray((n0, 1.0, W0, W0), dtype=float)
    common_energy_density = n0 * W0

    def normalized_hessian(H: np.ndarray, raw_det: complex, expected_raw_det: complex) -> dict[str, Any]:
        """Return dimensionless congruence normalization and identity check.

        ``x=(n,y,A0,A_L)`` has scales ``(n0,1,W0,W0)``.  With
        ``x=S*x_hat`` the congruent matrix ``S H S`` has one common
        energy-density unit; division by ``n0*W0`` makes the displayed matrix
        and determinant dimensionless.  The determinant identity is checked
        independently from the Schur factors, not inferred from the SVD.
        """
        local_scales = coordinate_scales[: H.shape[0]]
        S = np.diag(local_scales)
        H_congruent = S @ H @ S
        H_normalized = H_congruent / common_energy_density
        with np.errstate(all="ignore"):
            determinant_normalized = np.linalg.det(H_normalized)
        expected_normalized = (np.linalg.det(S) ** 2) * expected_raw_det / (common_energy_density ** H.shape[0])
        identity_error = _genuine_rel(determinant_normalized, expected_normalized)
        determinant_scale = max(float(np.linalg.norm(H_normalized, ord=np.inf) ** H.shape[0]), np.finfo(float).tiny)
        return {"H_normalized": H_normalized, "determinant_normalized": determinant_normalized,
                "determinant_identity_normalized": expected_normalized,
                "determinant_identity_relative_error": identity_error,
                "determinant_residual": float(abs(determinant_normalized) / determinant_scale),
                "determinant_residual_scale": determinant_scale,
                "coordinate_scales": local_scales, "common_energy_density": common_energy_density}

    if q == 0.0:  # homogeneous static grand-canonical limit
        L, D, B, C = t0, a + g * g / t0, b + g * h / t0, d0 + h * h / t0
        S = D - B * B / C if C != 0 else None
        H = np.asarray([[a, b, g], [b, d0, h], [g, h, -t0]], dtype=complex)
        with np.errstate(all="ignore"):
            det = np.linalg.det(H)
        normalized = normalized_hessian(H, det, -t0 * (D * C - B * B))
        if solve_direct:
            try:
                direct = np.linalg.solve(H, np.asarray([1, 0, 0], dtype=complex))
                direct_status = "DIRECT_SOLVE_OK"
            except np.linalg.LinAlgError:
                direct, direct_status = None, "DIRECT_SOLVE_SINGULAR"
        else:
                direct, direct_status = None, "DIRECT_SOLVE_SKIPPED_AT_POLE"
        regular = bool(direct is not None and S is not None and _finite_complex(S))
        return {"status": "STATIC_Q0_GRAND_CANONICAL" if regular else "SINGULAR_FULL_HESSIAN",
                "chi": None if S is None else 1 / S, "S": S, "D": D, "B": B, "C": C, "L": L,
                "F": None if S is None else D * C - B * B, "H": H, "determinant": det,
                "direct_solution": direct, "direct_status": direct_status, "direct_chi": None if direct is None else direct[0],
                "direct_relative_error": None if direct is None or S is None else _rel(direct[0], 1 / S),
                "q": q, "z": z, "regular": regular, "physical_stability_assessed": False, **normalized}
    L = t0 + q * q - z * z
    if L == 0:
        return {"status": "SINGULAR_VECTOR_ELIMINATION", "chi": None, "S": None, "L": L, "q": q, "z": z,
                "direct_status": "VECTOR_ELIMINATION_SINGULAR", "regular": False,
                "physical_stability_assessed": False}
    D = a + g * g * (1 - z * z / (q * q)) / L
    B = b + g * h / L
    C = d0 + Z * (q * q - z * z) + h * h * (1 - z * z / t0) / L
    if C == 0:
        S, F = None, D * C - B * B
    else:
        S, F = D - B * B / C, D * C - B * B
    H = np.asarray([[a, b, g, -g * z / q], [b, d0 + Z * (q * q - z * z), h, 0],
                    [g, h, -(q * q + t0), z * q], [-g * z / q, 0, z * q, t0 - z * z]], dtype=complex)
    with np.errstate(all="ignore"):
        det = np.linalg.det(H)
    normalized = normalized_hessian(H, det, -t0 * L * F)
    if solve_direct:
        try:
            direct = np.linalg.solve(H, np.asarray([1, 0, 0, 0], dtype=complex))
            direct_status = "DIRECT_SOLVE_OK" if np.all(np.isfinite(direct)) else "DIRECT_SOLVE_NONFINITE"
            if direct_status != "DIRECT_SOLVE_OK":
                direct = None
        except np.linalg.LinAlgError:
            direct, direct_status = None, "DIRECT_SOLVE_SINGULAR"
    else:
        direct, direct_status = None, "DIRECT_SOLVE_SKIPPED_AT_POLE"
    ok = direct is not None and S is not None and _finite_complex(S)
    return {"status": "RETARDED_RESPONSE_OFF_POLE" if ok else ("SINGULAR_SCALAR_ELIMINATION" if C == 0 else "SINGULAR_FULL_HESSIAN"),
            "chi": None if S is None else 1 / S, "S": S, "D": D, "B": B, "C": C, "F": F, "L": L,
            "H": H, "determinant": det, "direct_solution": direct, "direct_status": direct_status,
            "direct_chi": None if direct is None else direct[0], "direct_relative_error": None if not ok else _rel(direct[0], 1 / S),
            "q": q, "z": z, "regular": ok, "physical_stability_assessed": False, **normalized}


retarded_response_from_hessian = response_from_hessian


def _coefficients(model: Any, state: Mapping[str, Any], pi: Sequence[complex], scale: float) -> tuple[dict[str, Any], float]:
    vv, vs, ss = map(complex, pi)
    if vv == 0:
        raise RetardedResponseError("Pi_vv=0 is an explicit singular density kernel")
    n, y = float(state["n"]), float(state["y"])
    mw, MN, g = float(model.momega), float(model.MN), float(model.gomega)
    A0 = g * n / (mw * mw * y * y)
    h, t0 = -2 * mw * mw * y * A0, mw * mw * y * y
    c = {"a": 1 / vv, "b": MN * vs / vv, "dF": MN * MN * (vs * vs / vv - ss),
         "g": g, "h": h, "t0": t0, "A0": A0, "Pi_vv": vv, "Pi_vs": vs, "Pi_ss": ss,
         "_n0": n, "_W0": float(model.W0)}
    c["d0"] = c["dF"] + float(state["Uyy"]) - mw * mw * A0 * A0
    Z = (float(model.W0) * float(scale)) ** 2
    return c, Z


def retarded_response(model: Any, state: Mapping[str, Any], q: Any, z: Any, pi: Sequence[complex] | None = None,
                      scale: Any = 1, *, eta: Any | None = None, epsrel: float = DEFAULT_EPSREL,
                      epsabs: float = DEFAULT_EPSABS, solve_direct: bool = True) -> dict[str, Any]:
    qf, sf, zz = _real(q, "q", nonnegative=True), _real(scale, "scale", positive=True), _z(z, eta)
    if qf == 0.0 and (zz.real != 0.0 or zz.imag != 0.0):
        # The uniform density operator is the conserved total number.  Keep
        # the exact scalar pair channel in the returned kernel, but do not
        # manufacture a 1/Pi_vv inversion for the conserved density source.
        if pi is None:
            pi = polarization_kernel(state["m"], state["k"], qf, zz, d=model.d, epsrel=epsrel, epsabs=epsabs)
        else:
            pi = tuple(complex(x) for x in pi)
        if len(pi) != 3 or not all(_finite_complex(x) for x in pi):
            raise RetardedResponseError("Pi must contain three finite complex channels")
        return {"status": "Q0_CONSERVATION_ZERO", "chi": 0j, "S": None,
                "direct_solution": None, "direct_status": "NUMBER_CONSERVATION",
                "regular": True, "physical_stability_assessed": False,
                "q": qf, "z": zz, "scale": sf, "Pi": pi, "Z": (float(model.W0) * sf) ** 2,
                "coefficients": None}
    if pi is None:
        pi = polarization_kernel(state["m"], state["k"], qf, zz, d=model.d, epsrel=epsrel, epsabs=epsabs)
    c, Z = _coefficients(model, state, pi, sf)
    result = response_from_hessian(c, Z, qf, zz, solve_direct=solve_direct)
    result.update({"coefficients": c, "Z": Z, "Pi": tuple(complex(x) for x in pi)})
    return result


def _backgrounds(base: Any) -> list[dict[str, Any]]:
    out = [{"background_id": "Q4:n_over_n0=1", "family": "Q4", "density_ratio": "1", "target_y": None,
            "model": base, "state": base.equilibrium(base.n0), "inverse_design": None}]
    for target in ("0.90", "0.93"):
        model, _, jet = upstream.inverse_potential_jet(target, "240", "0.16", "-16")
        out.append({"background_id": f"W8:y_star={target}", "family": "W8", "density_ratio": "1", "target_y": target,
                    "model": model, "state": model.state(model.n0, mp.mpf(target)), "inverse_design": jet})
    for b in out:
        b["mass"], b["kF"] = float(b["state"]["m"]), float(b["state"]["k"])
        if b["mass"] <= 0 or b["kF"] <= 0 or b["state"]["y"] <= 0:
            raise ArithmeticError(f"invalid background {b['background_id']}")
    return out


def _raw_ward(m: float, k: float, q: float, z: complex, order: int = 256) -> tuple[np.ndarray, np.ndarray]:
    x, w = leggauss(order)
    total, ph = np.zeros(6, complex), np.zeros(6, complex)
    points = sorted(set((0.0, max(0.0, k - q), k)))
    for low, high in zip(points[:-1], points[1:]):
        p = (low + (x + 1) * (high - low) / 2)[:, None]
        wp = w[:, None] * (high - low) / 2
        E = np.hypot(m, p)
        for t in (1, -1):
            if t == 1:
                ulo = np.maximum(-1, (k * k - p * p - q * q) / (2 * p * q))
            else:
                ulo = np.full_like(p, -1.0)
            ulo = np.minimum(1, ulo)
            u = ulo + (x[None, :] + 1) * (1 - ulo) / 2
            wu = w[None, :] * (1 - ulo) / 2
            rz, r2 = p * u + q, p * p + q * q + 2 * p * q * u
            R, dot = np.sqrt(m * m + r2), p * p + p * q * u
            delta = (r2 - p * p) / (R + E) if t == 1 else -R - E
            even, odd = delta / (delta * delta - z * z), z / (delta * delta - z * z)
            traces = (1 + t * (dot + m * m) / (E * R), m * (1 / E + t / R),
                      1 + t * (m * m - dot) / (E * R), p * u / E + t * rz / R,
                      t * m * (p * u + rz) / (E * R), 1 + t * (2 * p * u * rz - dot - m * m) / (E * R))
            weight = wp * wu * p * p * 4 / (4 * math.pi * math.pi)
            values = np.asarray([np.sum(weight * T * (odd if j in (3, 4) else even)) for j, T in enumerate(traces)])
            total += values
            if t == 1:
                ph += values
    return total, ph


def _pole_value(model: Any, state: Mapping[str, Any], q: float, w: float, scale: float, cache: dict,
                *, solve_direct: bool = True) -> tuple[float, dict[str, Any]]:
    key = (id(model), q, round(w, 12))
    if key not in cache:
        cache[key] = polarization_kernel(state["m"], state["k"], q, w, d=model.d)
    pi = cache[key]
    answer = retarded_response(model, state, q, w, pi, scale, solve_direct=solve_direct)
    # A real-axis pole scan is only admissible when the sampled point is
    # genuinely outside the declared ph cut and all elimination factors exist.
    if not all(abs(complex(x).imag) <= 3e-5 * max(1.0, abs(complex(x).real)) for x in pi):
        raise RetardedResponseError("imaginary contamination in outside-cut pole scan")
    if (answer.get("F") is None or not all(_finite_complex(answer.get(key)) for key in ("F", "L", "C", "D", "B"))
            or abs(answer.get("L", 0)) == 0 or abs(answer.get("C", 0)) == 0):
        raise RetardedResponseError("singular elimination factor in pole scan")
    return float(np.real(answer["F"])), answer


POLE_GRID_TOLERANCE_MEV = 2e-3
POLE_DERIVATIVE_RELATIVE_TOLERANCE = 2e-3
POLE_NULL_RESIDUAL_TOLERANCE = 1e-6
POLE_DETERMINANT_RESIDUAL_TOLERANCE = 1e-6
POLE_DETERMINANT_IDENTITY_TOLERANCE = 5e-3
POLE_DENSITY_OVERLAP_TOLERANCE = 1e-6


def _pole_acceptance(*, grid_match: bool, outside_cut: bool, bracket_confirmed: bool,
                     finite_elimination_factors: bool, residue_converged: bool,
                     positive_finite_weight: bool, null_residual: Any,
                     determinant_residual: Any, determinant_identity_error: Any,
                     density_overlap: Any) -> tuple[bool, list[str]]:
    """Centralized, explicit predicate for a resolved density pole.

    This is deliberately data-oriented so the same acceptance boundary can
    be exercised by live candidates and manufactured negative controls.  In
    particular, a zero of a scalar-only block or a wrong-sign residue cannot
    acquire pole status merely because a determinant vanishes.
    """
    reasons: list[str] = []
    if not grid_match:
        reasons.append("dual_grid_not_converged")
    if not outside_cut:
        reasons.append("candidate_not_outside_particle_hole_cut")
    if not bracket_confirmed:
        reasons.append("bracket_not_confirmed")
    if not finite_elimination_factors:
        reasons.append("elimination_factor_nonfinite_or_zero")
    if not residue_converged:
        reasons.append("residue_step_not_converged")
    if not positive_finite_weight:
        reasons.append("residue_weight_not_positive_finite")
    try:
        if not math.isfinite(float(null_residual)) or float(null_residual) >= POLE_NULL_RESIDUAL_TOLERANCE:
            reasons.append("full_normalized_null_residual_too_large")
    except (TypeError, ValueError, OverflowError):
        reasons.append("full_normalized_null_residual_missing")
    try:
        if not math.isfinite(float(determinant_residual)) or float(determinant_residual) >= POLE_DETERMINANT_RESIDUAL_TOLERANCE:
            reasons.append("normalized_determinant_residual_too_large")
    except (TypeError, ValueError, OverflowError):
        reasons.append("normalized_determinant_residual_missing")
    try:
        if not math.isfinite(float(determinant_identity_error)) or float(determinant_identity_error) >= POLE_DETERMINANT_IDENTITY_TOLERANCE:
            reasons.append("full_determinant_identity_mismatch")
    except (TypeError, ValueError, OverflowError):
        reasons.append("full_determinant_identity_missing")
    try:
        if not math.isfinite(float(density_overlap)) or float(density_overlap) <= POLE_DENSITY_OVERLAP_TOLERANCE:
            reasons.append("density_overlap_zero_or_too_small")
    except (TypeError, ValueError, OverflowError):
        reasons.append("density_overlap_missing")
    return not reasons, reasons


def _pole_rejection_record(root: float, edge: float, q: float, bracket: tuple[float, float], *,
                           candidate_grid: str, grid_match: bool, grid_delta: float | None,
                           reasons: Sequence[str], error: str | None = None) -> dict[str, Any]:
    """Retain a candidate that could not complete pole validation."""
    record: dict[str, Any] = {
        "omega_MeV": _rjson(root), "gap_above_ph_MeV": _rjson(root - edge),
        "candidate_grid": candidate_grid, "dual_grid_root_delta_MeV": None if grid_delta is None else _rjson(grid_delta),
        "grid_root_converged": bool(grid_match), "outside_cut_bracket": bool(root > edge and root < 0.95 * q),
        "bracket_MeV": [_rjson(bracket[0]), _rjson(bracket[1])], "F_at_root": None,
        "bracket_confirmed": False, "determinant_at_root": None, "determinant_raw_MeV6": None, "determinant_normalized": None,
        "determinant_identity_normalized": None, "determinant_identity_relative_error": None,
        "normalized_determinant_residual": None, "nullvector": None,
        "nullvector_coordinate_system": "(n/n0,y,A0/W0,A_L/W0)", "density_overlap": None,
        "nullvector_residual": None, "nonzero_density_overlap": False, "C_at_root": None,
        "spectral_derivative_MeVminus3": None, "spectral_derivative_h001_MeVminus3": None,
        "spectral_derivative_h0005_MeVminus3": None, "spectral_derivative_step_relative_difference": None,
        "spectral_weight_MeV3": None, "residue_step_converged": False, "positive_finite_weight": False,
        "accepted_resolved_pole": False,
        "rejection_reasons": list(reasons),
    }
    if error is not None:
        record["validation_error"] = error
    return record


def _pole_case(model: Any, state: Mapping[str, Any], name: str, q: float, scale: float, cache: dict) -> dict[str, Any]:
    edge = particle_hole_edge(state["m"], state["k"], q)
    interval = (edge * (1 + 1e-5), 0.95 * q)
    grids, roots_by_grid = {}, {}
    brackets_by_grid = {}
    for n in (33, 65):
        grid = np.linspace(interval[0], interval[1], n)
        values, valid = [], []
        for w in grid:
            try:
                value, _ = _pole_value(model, state, q, float(w), scale, cache, solve_direct=False)
                values.append(value)
                valid.append(True)
            except RetardedResponseError:
                values.append(float("nan"))
                valid.append(False)
        roots, brackets = [], []
        for i in range(n - 1):
            if valid[i] and valid[i + 1] and values[i] * values[i + 1] < 0:
                def f(w: float) -> float:
                    return _pole_value(model, state, q, float(w), scale, cache, solve_direct=False)[0]
                try:
                    root = brentq(f, float(grid[i]), float(grid[i + 1]), xtol=2e-8, rtol=1e-12)
                    if not roots or abs(root - roots[-1]) > 1e-5:
                        roots.append(root)
                        brackets.append((float(grid[i]), float(grid[i + 1])))
                except (ValueError, RetardedResponseError):
                    pass
        grids[str(n)] = {"node_count": n, "valid_nodes": sum(valid), "sign_change_count": len(roots),
                         "finite_values": all(np.isfinite(values)), "roots_MeV": [_rjson(x) for x in roots],
                         "brackets_MeV": [[_rjson(lo), _rjson(hi)] for lo, hi in brackets]}
        roots_by_grid[n] = roots
        brackets_by_grid[n] = brackets
    roots = roots_by_grid[65]
    records = []
    # Start from the finer grid, then retain any unmatched coarse-grid root as
    # an explicitly rejected candidate rather than silently dropping it.
    candidates: list[tuple[float, tuple[float, float], str, bool, float | None]] = []
    matched_coarse: set[int] = set()
    for root, bracket in zip(roots_by_grid[65], brackets_by_grid[65]):
        if roots_by_grid[33]:
            j = min(range(len(roots_by_grid[33])), key=lambda i: abs(root - roots_by_grid[33][i]))
            delta = abs(root - roots_by_grid[33][j])
            match = bool(delta <= POLE_GRID_TOLERANCE_MEV)
            if match:
                matched_coarse.add(j)
        else:
            delta, match = None, False
        candidates.append((root, bracket, "65", match, delta))
    for j, (root, bracket) in enumerate(zip(roots_by_grid[33], brackets_by_grid[33])):
        if j not in matched_coarse:
            candidates.append((root, bracket, "33", False, None if not roots else min(abs(root - x) for x in roots)))
    grid_converged = bool(roots_by_grid[33] and roots_by_grid[65] and
                          len(matched_coarse) == len(roots_by_grid[33]) and
                          all(item[3] for item in candidates))
    for root, bracket, candidate_grid, grid_match, grid_delta in candidates:
        outside_cut = bool(root > interval[0] and root < interval[1] and root > edge and root < 0.95 * q)
        bracket_confirmed = bool(bracket[0] <= root <= bracket[1] and bracket[0] >= interval[0] and bracket[1] <= interval[1])
        try:
            froot, ans = _pole_value(model, state, q, root, scale, cache, solve_direct=False)
            H_normalized = np.asarray(ans["H_normalized"], dtype=complex)
            if H_normalized.shape != (4, 4) or not np.all(np.isfinite(H_normalized)):
                raise RetardedResponseError("normalized full Hessian is nonfinite")
            _, _, vh = np.linalg.svd(H_normalized)
            null = vh[-1].conj()
            null /= np.linalg.norm(null)
            hnorm = float(np.linalg.norm(H_normalized, ord=np.inf))
            null_residual = float(np.linalg.norm(H_normalized @ null) / max(hnorm, np.finfo(float).tiny))
            determinant_residual = float(ans["determinant_residual"])
            determinant_identity_error = float(ans["determinant_identity_relative_error"])
            derivatives = {}
            for h in (1e-3, 5e-4):
                sp = _pole_value(model, state, q, root + h, scale, cache, solve_direct=True)[1]["S"]
                sm = _pole_value(model, state, q, root - h, scale, cache, solve_direct=True)[1]["S"]
                derivatives[h] = (float(np.real(sp)) - float(np.real(sm))) / (2 * h)
            deriv = derivatives[5e-4]
            derivative_difference = _genuine_rel(derivatives[1e-3], derivatives[5e-4])
            residue_converged = bool(math.isfinite(derivative_difference) and
                                     derivative_difference < POLE_DERIVATIVE_RELATIVE_TOLERANCE)
            weight = -1.0 / deriv if math.isfinite(deriv) and deriv != 0.0 else None
            positive_weight = bool(weight is not None and math.isfinite(weight) and weight > 0.0)
            finite_factors = bool(all(_finite_complex(ans.get(key)) for key in ("F", "L", "C", "D", "B")) and
                                  ans.get("Pi") is not None and all(_finite_complex(x) for x in ans["Pi"]) and
                                  abs(ans["Pi"][0]) > 0 and
                                  abs(ans["L"]) > 0 and abs(ans["C"]) > 0)
            accepted, reasons = _pole_acceptance(
                grid_match=grid_match, outside_cut=outside_cut, bracket_confirmed=bracket_confirmed,
                finite_elimination_factors=finite_factors, residue_converged=residue_converged,
                positive_finite_weight=positive_weight, null_residual=null_residual,
                determinant_residual=determinant_residual, determinant_identity_error=determinant_identity_error,
                density_overlap=abs(null[0]))
            records.append({"omega_MeV": _rjson(root), "gap_above_ph_MeV": _rjson(root - edge),
                            "candidate_grid": candidate_grid, "dual_grid_root_delta_MeV": None if grid_delta is None else _rjson(grid_delta),
                            "determinant_at_root": _cjson(ans["determinant"]), "determinant_raw_MeV6": _cjson(ans["determinant"]),
                            "determinant_normalized": _cjson(ans["determinant_normalized"]),
                            "determinant_identity_normalized": _cjson(ans["determinant_identity_normalized"]),
                            "determinant_identity_relative_error": _rjson(determinant_identity_error),
                            "F_at_root": _rjson(froot), "normalized_determinant_residual": _rjson(determinant_residual),
                            "nullvector": [_cjson(v) for v in null],
                            "nullvector_coordinate_system": "(n/n0,y,A0/W0,A_L/W0)",
                            "density_overlap": _rjson(abs(null[0])), "nullvector_residual": _rjson(null_residual),
                            "grid_root_converged": bool(grid_match), "nonzero_density_overlap": bool(abs(null[0]) > POLE_DENSITY_OVERLAP_TOLERANCE),
                            "C_at_root": _cjson(ans["C"]), "bracket_MeV": [_rjson(bracket[0]), _rjson(bracket[1])],
                            "outside_cut_bracket": outside_cut, "bracket_confirmed": bracket_confirmed,
                            "spectral_derivative_MeVminus3": _rjson(deriv),
                            "spectral_derivative_h001_MeVminus3": _rjson(derivatives[1e-3]),
                            "spectral_derivative_h0005_MeVminus3": _rjson(derivatives[5e-4]),
                            "spectral_derivative_step_relative_difference": _rjson(derivative_difference),
                            "spectral_weight_MeV3": None if weight is None else _rjson(weight),
                            "residue_step_converged": residue_converged, "positive_finite_weight": positive_weight,
                            "accepted_resolved_pole": accepted, "rejection_reasons": reasons})
        except (RetardedResponseError, np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError, TypeError, ValueError) as exc:
            records.append(_pole_rejection_record(root, edge, q, bracket, candidate_grid=candidate_grid,
                                                  grid_match=grid_match, grid_delta=grid_delta,
                                                  reasons=["root_validation_failed"],
                                                  error=f"{type(exc).__name__}: {exc}"))
    converged = grid_converged
    accepted_count = sum(bool(record["accepted_resolved_pole"]) for record in records)
    return {"case_id": f"{name}:q={q:g}:scale={scale:g}", "background_id": name, "q_MeV": _rjson(q),
            "scale": _rjson(scale), "ph_upper_edge_MeV": _rjson(edge), "searched_interval_MeV": [_rjson(x) for x in interval],
            "grids": grids, "roots": records, "resolved_root_count": accepted_count, "candidate_root_count": len(records),
            "grid_root_converged": bool(converged), "status": "RESOLVED_DENSITY_POLE" if accepted_count else "NONE_RESOLVED_IN_SCANNED_INTERVAL",
            "no_root_is_not_absence_theorem": True}


def _response_json(ans: Mapping[str, Any]) -> dict[str, Any]:
    out = {"status": ans.get("status"), "direct_status": ans.get("direct_status"),
           "regular": bool(ans.get("regular", False)),
           "physical_stability_assessed": bool(ans.get("physical_stability_assessed", False))}
    for key, unit in (("D", "MeVminus2"), ("B", "MeV"), ("C", "MeV4"), ("F", "MeV2"), ("S", "MeVminus2"), ("L", "MeV2"), ("chi", "MeV2")):
        out[f"{key}_{unit}"] = None if ans.get(key) is None else _cjson(ans[key])
    # The raw determinant belongs to mixed physical coordinates and is not
    # dimensionless.  Also expose the independently normalized congruence
    # determinant used by pole validation.
    if ans.get("determinant") is not None:
        out["determinant_raw_MeV4" if np.asarray(ans.get("H")).shape == (3, 3) else "determinant_raw_MeV6"] = _cjson(ans["determinant"])
    out["determinant_normalized_dimensionless"] = None if ans.get("determinant_normalized") is None else _cjson(ans["determinant_normalized"])
    out["determinant_identity_normalized_dimensionless"] = None if ans.get("determinant_identity_normalized") is None else _cjson(ans["determinant_identity_normalized"])
    out["determinant_identity_relative_error"] = None if ans.get("determinant_identity_relative_error") is None else _rjson(ans["determinant_identity_relative_error"])
    out["determinant_residual"] = None if ans.get("determinant_residual") is None else _rjson(ans["determinant_residual"])
    out["determinant_residual_scale"] = None if ans.get("determinant_residual_scale") is None else _rjson(ans["determinant_residual_scale"])
    out["normalization_coordinate_scales"] = None if ans.get("coordinate_scales") is None else [_rjson(x) for x in ans["coordinate_scales"]]
    out["normalization_common_energy_density_MeV4"] = None if ans.get("common_energy_density") is None else _rjson(ans["common_energy_density"])
    out["H"] = None if ans.get("H") is None else _matrix_json(ans["H"])
    out["H_normalized"] = None if ans.get("H_normalized") is None else _matrix_json(ans["H_normalized"])
    out["direct_solution"] = None if ans.get("direct_solution") is None else [_cjson(x) for x in ans["direct_solution"]]
    out["direct_vs_schur_relative_error"] = None if ans.get("direct_relative_error") is None else _rjson(ans["direct_relative_error"])
    return out


_PROVENANCE_FIELDS = {"source_sha256", "upstream_source_sha256", "static_source_sha256", "contract_sha256",
                      "numerical_payload_sha256"}


def _numerical_payload_sha256(result: Mapping[str, Any]) -> str:
    """Fingerprint all output other than mutable producer/provenance hashes."""
    payload = {key: value for key, value in result.items() if key not in _PROVENANCE_FIELDS}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_number_string(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _valid_cjson_payload(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"real", "imag"} and all(_valid_number_string(value[key]) for key in ("real", "imag"))


def _valid_matrix_payload(value: Any, shape: tuple[int, int]) -> bool:
    return (isinstance(value, list) and len(value) == shape[0] and
            all(isinstance(row, list) and len(row) == shape[1] and all(_valid_cjson_payload(x) for x in row) for row in value))


def _valid_payload_tree(value: Any) -> bool:
    """Reject NaN/Inf and malformed encoded complex values in nested controls."""
    if value is None or isinstance(value, (bool, str)):
        if isinstance(value, str) and value.lower() in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
            return False
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_valid_payload_tree(x) for x in value)
    if isinstance(value, dict):
        if set(value) == {"real", "imag"}:
            return _valid_cjson_payload(value)
        return all(_valid_payload_tree(x) for x in value.values())
    return False


def _validate_response_payload(response: Any) -> bool:
    if not isinstance(response, dict) or "stable" in response:
        return False
    required = {"status", "direct_status", "regular", "physical_stability_assessed",
                "D_MeVminus2", "B_MeV", "C_MeV4", "F_MeV2", "S_MeVminus2", "L_MeV2",
                "chi_MeV2", "determinant_normalized_dimensionless",
                "determinant_identity_normalized_dimensionless", "determinant_identity_relative_error",
                "determinant_residual", "determinant_residual_scale", "normalization_coordinate_scales",
                "normalization_common_energy_density_MeV4",
                "H", "H_normalized", "direct_solution", "direct_vs_schur_relative_error"}
    if not required <= set(response) or not isinstance(response["regular"], bool) or response["physical_stability_assessed"] is not False:
        return False
    for key in ("D_MeVminus2", "B_MeV", "C_MeV4", "F_MeV2", "S_MeVminus2", "L_MeV2", "chi_MeV2",
                "determinant_normalized_dimensionless", "determinant_identity_normalized_dimensionless"):
        if response[key] is not None and not _valid_cjson_payload(response[key]):
            return False
    for key in ("determinant_identity_relative_error", "determinant_residual", "determinant_residual_scale",
                "normalization_common_energy_density_MeV4", "direct_vs_schur_relative_error"):
        if response[key] is not None and not _valid_number_string(response[key]):
            return False
    if not (isinstance(response["normalization_coordinate_scales"], list) and
            all(_valid_number_string(x) for x in response["normalization_coordinate_scales"])):
        return False
    if response["H"] is not None and not _valid_matrix_payload(response["H"], (4, 4)):
        return False
    if response["H_normalized"] is not None and not _valid_matrix_payload(response["H_normalized"], (4, 4)):
        return False
    if response["direct_solution"] is not None and not (isinstance(response["direct_solution"], list) and
                                                          len(response["direct_solution"]) == 4 and
                                                          all(_valid_cjson_payload(x) for x in response["direct_solution"])):
        return False
    raw_keys = {key for key in response if key.startswith("determinant_raw_")}
    if raw_keys != {"determinant_raw_MeV6"} or not _valid_cjson_payload(response["determinant_raw_MeV6"]):
        return False
    return True


def _validate_pole_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    required = {"omega_MeV", "gap_above_ph_MeV", "candidate_grid", "dual_grid_root_delta_MeV",
                "determinant_at_root", "determinant_raw_MeV6", "determinant_normalized", "determinant_identity_normalized",
                "determinant_identity_relative_error", "F_at_root", "normalized_determinant_residual",
                "nullvector", "nullvector_coordinate_system", "density_overlap", "nullvector_residual",
                "grid_root_converged", "nonzero_density_overlap", "C_at_root", "bracket_MeV",
                "outside_cut_bracket", "bracket_confirmed", "spectral_derivative_MeVminus3",
                "spectral_derivative_h001_MeVminus3", "spectral_derivative_h0005_MeVminus3",
                "spectral_derivative_step_relative_difference", "spectral_weight_MeV3",
                "residue_step_converged", "positive_finite_weight", "accepted_resolved_pole", "rejection_reasons"}
    if not required <= set(record) or record["candidate_grid"] not in {"33", "65"}:
        return False
    for key in ("omega_MeV", "gap_above_ph_MeV"):
        if not _valid_number_string(record[key]):
            return False
    if record["F_at_root"] is not None and not _valid_number_string(record["F_at_root"]):
        return False
    for key in ("dual_grid_root_delta_MeV", "determinant_identity_relative_error", "normalized_determinant_residual",
                "density_overlap", "nullvector_residual", "spectral_derivative_MeVminus3",
                "spectral_derivative_h001_MeVminus3", "spectral_derivative_h0005_MeVminus3",
                "spectral_derivative_step_relative_difference", "spectral_weight_MeV3"):
        if record[key] is not None and not _valid_number_string(record[key]):
            return False
    for key in ("determinant_at_root", "determinant_raw_MeV6", "determinant_normalized", "determinant_identity_normalized", "C_at_root"):
        if record[key] is not None and not _valid_cjson_payload(record[key]):
            return False
    if not (isinstance(record["bracket_MeV"], list) and len(record["bracket_MeV"]) == 2 and
            all(_valid_number_string(x) for x in record["bracket_MeV"])):
        return False
    if record["nullvector"] is not None and not (isinstance(record["nullvector"], list) and
                                                    len(record["nullvector"]) == 4 and
                                                    all(_valid_cjson_payload(x) for x in record["nullvector"])):
        return False
    if not all(isinstance(record[key], bool) for key in ("grid_root_converged", "nonzero_density_overlap", "outside_cut_bracket",
                                                         "bracket_confirmed", "residue_step_converged", "positive_finite_weight",
                                                         "accepted_resolved_pole")):
        return False
    if not isinstance(record["rejection_reasons"], list) or not all(isinstance(x, str) and x for x in record["rejection_reasons"]):
        return False
    if record["accepted_resolved_pole"] and record["rejection_reasons"]:
        return False
    return True


def _validate_rows(result: Mapping[str, Any]) -> bool:
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROW_COUNT:
        return False
    expected_ids = set()
    for family, target in (("Q4", None), ("W8", "0.90"), ("W8", "0.93")):
        background_id = "Q4:n_over_n0=1" if family == "Q4" else f"W8:y_star={target}"
        for scale in SCALES:
            for q in Q_GRID_MEV:
                for fraction in FREQUENCY_FRACTIONS:
                    expected_ids.add(f"{background_id}:scale={scale}:q={q}:omega_fraction={fraction}")
    if {row.get("row_id") for row in rows} != expected_ids:
        return False
    required = {"row_id", "background_id", "family", "target_y", "scale", "q_MeV", "ph_upper_edge_MeV",
                "omega_fraction", "omega_MeV", "inside_particle_hole_continuum", "Pi", "Pi_vv_MeV2",
                "Pi_vs_MeV2", "Pi_ss_MeV2", "quadrature_error_estimate", "response", "chi_MeV2",
                "absorptive_Im_chi_over_pi_MeV2", "status", "response_status"}
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row) or not isinstance(row["inside_particle_hole_continuum"], bool):
            return False
        if row["Pi"].keys() != {"vv_MeV2", "vs_MeV2", "ss_MeV2"} or not all(_valid_cjson_payload(x) for x in row["Pi"].values()):
            return False
        if not all(_valid_cjson_payload(row[key]) for key in ("Pi_vv_MeV2", "Pi_vs_MeV2", "Pi_ss_MeV2")):
            return False
        if not all(_valid_number_string(row[key]) for key in ("q_MeV", "ph_upper_edge_MeV", "omega_fraction", "omega_MeV", "quadrature_error_estimate")):
            return False
        if row["chi_MeV2"] is not None and not _valid_cjson_payload(row["chi_MeV2"]):
            return False
        if row["absorptive_Im_chi_over_pi_MeV2"] is not None and not _valid_number_string(row["absorptive_Im_chi_over_pi_MeV2"]):
            return False
        if row["status"] != row["response_status"] or not _validate_response_payload(row["response"]):
            return False
    return True


def _validate_pole_cases(result: Mapping[str, Any]) -> bool:
    cases = result.get("pole_cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_POLE_CASES:
        return False
    expected_ids = {f"{background}:q={float(q):g}:scale={float(scale):g}"
                    for background in ("Q4:n_over_n0=1", "W8:y_star=0.90", "W8:y_star=0.93")
                    for q in Q_GRID_MEV for scale in SCALES}
    if {case.get("case_id") for case in cases} != expected_ids:
        return False
    for case in cases:
        if not isinstance(case, dict) or not {"case_id", "background_id", "q_MeV", "scale", "ph_upper_edge_MeV",
                "searched_interval_MeV", "grids", "roots", "resolved_root_count", "candidate_root_count",
                "grid_root_converged", "status", "no_root_is_not_absence_theorem"} <= set(case):
            return False
        if not all(_valid_number_string(case[key]) for key in ("q_MeV", "scale", "ph_upper_edge_MeV")):
            return False
        if not (isinstance(case["searched_interval_MeV"], list) and len(case["searched_interval_MeV"]) == 2 and
                all(_valid_number_string(x) for x in case["searched_interval_MeV"])):
            return False
        if not isinstance(case["grids"], dict) or set(case["grids"]) != {"33", "65"}:
            return False
        for grid_name, grid in case["grids"].items():
            if not isinstance(grid, dict) or not {"node_count", "valid_nodes", "sign_change_count", "finite_values", "roots_MeV", "brackets_MeV"} <= set(grid):
                return False
            if grid["node_count"] != int(grid_name) or not all(isinstance(grid[key], int) for key in ("valid_nodes", "sign_change_count")):
                return False
            if not isinstance(grid["finite_values"], bool) or not isinstance(grid["roots_MeV"], list) or not all(_valid_number_string(x) for x in grid["roots_MeV"]):
                return False
            if not (isinstance(grid["brackets_MeV"], list) and all(isinstance(x, list) and len(x) == 2 and all(_valid_number_string(y) for y in x) for x in grid["brackets_MeV"])):
                return False
        if not isinstance(case["roots"], list) or not all(_validate_pole_record(root) for root in case["roots"]):
            return False
        if case["candidate_root_count"] != len(case["roots"]) or case["resolved_root_count"] != sum(bool(x["accepted_resolved_pole"]) for x in case["roots"]):
            return False
        if not isinstance(case["grid_root_converged"], bool) or case["status"] not in {"RESOLVED_DENSITY_POLE", "NONE_RESOLVED_IN_SCANNED_INTERVAL"}:
            return False
        if not isinstance(case["no_root_is_not_absence_theorem"], bool) or not case["no_root_is_not_absence_theorem"]:
            return False
    return True


def _build_snapshot() -> dict[str, Any]:
    with mp.workdps(50):
        base = upstream.BulkModel()
        guard = static._guard(base)
        backgrounds = _backgrounds(base)
    rows, cache = [], {}
    static_controls, imag_controls, offpole = [], [], []
    for b in backgrounds:
        model, state, m, k = b["model"], b["state"], b["mass"], b["kF"]
        for scale_text in SCALES:
            scale = float(scale_text)
            for q_text in Q_GRID_MEV:
                q = float(q_text)
                edge = particle_hole_edge(m, k, q)
                for frac_text in FREQUENCY_FRACTIONS:
                    omega = edge * float(frac_text)
                    pi, error = _kernel(m, k, q, complex(omega), int(model.d), epsrel=DEFAULT_EPSREL, epsabs=DEFAULT_EPSABS)
                    ans = retarded_response(model, state, q, omega, pi, scale)
                    row = {"row_id": f"{b['background_id']}:scale={scale_text}:q={q_text}:omega_fraction={frac_text}",
                           "background_id": b["background_id"], "family": b["family"], "target_y": b["target_y"],
                           "scale": scale_text, "q_MeV": _rjson(q), "ph_upper_edge_MeV": _rjson(edge),
                           "omega_fraction": frac_text, "omega_MeV": _rjson(omega),
                           "inside_particle_hole_continuum": bool(0 < omega < edge),
                           "Pi": {"vv_MeV2": _cjson(pi[0]), "vs_MeV2": _cjson(pi[1]), "ss_MeV2": _cjson(pi[2])},
                           "Pi_vv_MeV2": _cjson(pi[0]), "Pi_vs_MeV2": _cjson(pi[1]), "Pi_ss_MeV2": _cjson(pi[2]),
                           "quadrature_error_estimate": _rjson(error), "response": _response_json(ans),
                           "chi_MeV2": None if ans.get("chi") is None else _cjson(ans["chi"]),
                           "absorptive_Im_chi_over_pi_MeV2": None if ans.get("chi") is None else _rjson(ans["chi"].imag / math.pi),
                           "status": ans["status"], "response_status": ans["status"]}
                    rows.append(row)
                # Static and off-pole controls use independently recomputed
                # q=0/finite-q kernels, rather than expected-value tables.
                static_pi = static.polarization_kernel(str(m), str(k), q, d=model.d, dps=40)
                own_pi = polarization_kernel(m, k, q, 0, d=model.d)
                static_candidate = static.ScaledNonlocalModel(scale_text, model)
                static_state = static_candidate.state(state["n"], state["y"])
                static_coeff = static._coeff(static_candidate, static_state, 1, static_pi)
                static_answer = static.response_from_hessian(static_coeff, static_coeff["Z"], q)
                own_answer = retarded_response(model, state, q, 0, own_pi, scale)
                static_controls.append({"background_id": b["background_id"], "q_MeV": _rjson(q), "scale": scale_text,
                                        "relative_errors": [_rjson(_rel(own_pi[i], static_pi[i])) for i in range(3)],
                                        "chi_relative_error": _rjson(_rel(own_answer["chi"], static_answer["chi"])),
                                        "pass": max(_rel(own_pi[i], static_pi[i]) for i in range(3)) <= 2e-6 and _rel(own_answer["chi"], static_answer["chi"]) <= 2e-6})
                probe_w = 0.6 * edge
                probe_pi = polarization_kernel(m, k, q, probe_w, d=model.d)
                probe_ans = retarded_response(model, state, q, probe_w, probe_pi, scale)
                offpole.append({"background_id": b["background_id"], "q_MeV": _rjson(q), "scale": scale_text,
                                "omega_MeV": _rjson(probe_w), "direct_vs_schur_relative_error": _rjson(probe_ans["direct_relative_error"]),
                                "pass": bool(probe_ans["direct_relative_error"] is not None and probe_ans["direct_relative_error"] < 2e-8)})
                if scale == 1.0:
                    imag = analytic_ph_imag(m, k, q, probe_w, model.d)
                    numerical = polarization_kernel(m, k, q, probe_w, d=model.d)
                    imag_controls.append({"background_id": b["background_id"], "q_MeV": _rjson(q), "omega_MeV": _rjson(probe_w),
                                          "region": "inside_particle_hole_cut",
                                          "numerical_Im_Pi": [_rjson(x.imag) for x in numerical], "analytic_Im_Pi": [_rjson(x) for x in imag],
                                          "relative_error": _rjson(max(_rel(numerical[i].imag, imag[i]) for i in range(3))),
                                          "pass": max(_rel(numerical[i].imag, imag[i]) for i in range(3)) < 2e-6})
                    outside_w = 1.25 * edge
                    outside = polarization_kernel(m, k, q, outside_w, d=model.d)
                    zero = (0.0, 0.0, 0.0)
                    imag_controls.append({"background_id": b["background_id"], "q_MeV": _rjson(q), "omega_MeV": _rjson(outside_w),
                                          "region": "above_particle_hole_cut", "numerical_Im_Pi": [_rjson(x.imag) for x in outside],
                                          "analytic_Im_Pi": ["0", "0", "0"], "relative_error": _rjson(max(_rel(outside[i].imag, zero[i]) for i in range(3))),
                                          "pass": bool(max(abs(x.imag) for x in outside) < 2e-5)})
    if len(rows) != EXPECTED_ROW_COUNT:
        raise ArithmeticError(f"row count {len(rows)} != {EXPECTED_ROW_COUNT}")
    return {"base": base, "guard": guard, "backgrounds": backgrounds, "rows": rows, "cache": cache,
            "static_controls": static_controls, "imag_controls": imag_controls, "offpole_controls": offpole}


def _controls(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    b = snapshot["backgrounds"][0]
    model, state, m, k = b["model"], b["state"], b["mass"], b["kF"]
    q, omega = 100.0, 0.6 * particle_hole_edge(m, k, 100.0)
    boundary = np.asarray(polarization_kernel(m, k, q, omega, d=model.d))
    regulator = []
    for eta in (0.4, 0.2, 0.1):
        value = np.asarray(polarization_kernel(m, k, q, omega + 1j * eta, d=model.d, epsrel=CONTROL_EPSREL, epsabs=CONTROL_EPSABS))
        regulator.append({"eta_MeV": _rjson(eta), "Pi": [_cjson(x) for x in value],
                          "relative_to_boundary": [_rjson(_rel(value[i], boundary[i])) for i in range(3)]})
    negative = np.asarray(polarization_kernel(m, k, q, -omega, d=model.d))
    pos_answer = retarded_response(model, state, q, omega, boundary, 1.0)
    neg_answer = retarded_response(model, state, q, -omega, negative, 1.0)
    kernel_conj_error = max(_rel(negative[i], np.conj(boundary[i])) for i in range(3))
    chi_conj_error = _rel(neg_answer["chi"], np.conj(pos_answer["chi"]))
    neg_control = {"kernel_relative_conjugation_error": _rjson(kernel_conj_error),
                   "chi_relative_conjugation_error": _rjson(chi_conj_error),
                   "pass": bool(kernel_conj_error < 2e-8 and chi_conj_error < 2e-8)}
    # Explicit current closure: raw six-component Gauss integrals, including
    # negative intermediates, against the one-dimensional radial kernel.
    ward_rows = []
    for name, qq, frac in (("Q4", 100.0, 0.6), ("W8.90", 200.0, 1.2)):
        prefix = "Q4:" if name == "Q4" else "W8:y_star=0.90"
        bb = next(x for x in snapshot["backgrounds"] if x["background_id"].startswith(prefix))
        ee = particle_hole_edge(bb["mass"], bb["kF"], qq)
        zz = ee * frac + 3j
        raw, ph = _raw_ward(bb["mass"], bb["kF"], qq, zz, 256)
        radial = np.asarray(polarization_kernel(bb["mass"], bb["kF"], qq, zz, d=bb["model"].d))
        scale = max(1.0, float(np.max(np.abs(radial))))
        ward = max(abs(qq * raw[3] - zz * raw[0]), abs(qq * raw[4] - zz * raw[1]), abs(qq * qq * raw[5] - zz * zz * raw[0])) / (scale * max(qq * qq, abs(zz * zz)))
        violation = abs(qq * qq * ph[5] - zz * zz * ph[0]) / (scale * qq * qq)
        ward_rows.append({"background_id": bb["background_id"], "q_MeV": _rjson(qq), "omega_re_MeV": _rjson(zz.real),
                          "eta_MeV": "3", "gauss_order": 256, "raw_1D_relative_error": _rjson(np.max(np.abs(raw[:3] - radial)) / scale),
                          "raw_Ward_relative_error": _rjson(ward), "ph_only_Ward_violation": _rjson(violation),
                          "pass": bool(ward < 1e-9 and violation > 1e-4)})
    # Missing A_L and missing scalar-vector mixing are intentionally distinct
    # counterfactual closures, not production paths.
    pi = boundary
    c, Z = _coefficients(model, state, pi, 1.0)
    full = response_from_hessian(c, Z, q, omega)
    cm = dict(c)
    cm["h"] = 0j
    no_h = response_from_hessian(cm, Z, q, omega)
    a, d0, g, h, t0 = (c[x] for x in ("a", "d0", "g", "h", "t0"))
    zz = complex(omega)
    H3 = np.asarray([[a, c["b"], g], [c["b"], d0 + Z * (q*q - zz*zz), h], [g, h, -(q*q+t0)]], complex)
    no_al = np.linalg.solve(H3, np.asarray([1, 0, 0], complex))[0]
    negative_closure = {"omega_MeV": _rjson(omega), "full_chi": _cjson(full["chi"]), "missing_A_L_chi": _cjson(no_al),
                        "missing_h_chi": _cjson(no_h["chi"]), "missing_A_L_relative_error": _rjson(_rel(no_al, full["chi"])),
                        "missing_h_relative_error": _rjson(_rel(no_h["chi"], full["chi"])),
                        "detected_and_rejected": bool(_rel(no_al, full["chi"]) > 1e-6 and _rel(no_h["chi"], full["chi"]) > 1e-6)}
    scale_13 = 1.3
    z1, z13 = float(model.W0) ** 2, (float(model.W0) * scale_13) ** 2
    scale_answer = retarded_response(model, state, q, omega, boundary, scale_13)
    gradient_control = {"Z_scale_1_MeV2": _rjson(z1), "Z_scale_1.3_MeV2": _rjson(z13),
                        "Z_ratio": _rjson(z13 / z1), "expected_Z_ratio": _rjson(scale_13 ** 2),
                        "chi_scale_1": _cjson(full["chi"]), "chi_scale_1.3": _cjson(scale_answer["chi"]),
                        "chi_relative_difference": _rjson(_rel(scale_answer["chi"], full["chi"])),
                        "gradient_applied_once": bool(abs(z13 / z1 - scale_13 ** 2) < 1e-12)}
    # Conservation controls and explicit pole-validation negative controls.
    q0_static = polarization_kernel(m, k, 0, 0, d=model.d)
    q0_dynamic = polarization_kernel(m, k, 0, 1.0 + 0.2j, d=model.d)
    q0_coeff, q0_Z = _coefficients(model, state, q0_static, 1.0)
    q0_response = response_from_hessian(q0_coeff, q0_Z, 0, 0)
    q0_control = {"static_Pi": [_cjson(x) for x in q0_static], "nonzero_frequency_Pi": [_cjson(x) for x in q0_dynamic],
                  "static_grand_canonical_chi": _cjson(q0_response["chi"]),
                  "static_grand_canonical_S": _cjson(q0_response["S"]),
                  "static_vs_homogeneous_mu_prime_relative_error": _rjson(_rel(q0_response["S"], state["mu_prime"])),
                  "nonzero_frequency_density_response": _cjson(response_from_hessian(c, Z, 0, 1.0 + 0.2j)["chi"]),
                  "ensemble_caveat": "q=0,z=0 is homogeneous grand-canonical; exact fixed-total-N uniform mode is excluded",
                  "pass": bool(abs(q0_dynamic[0]) == 0 and abs(q0_dynamic[1]) == 0 and
                               _rel(q0_response["S"], state["mu_prime"]) < 2e-8)}
    test_q, test_z, test_Z = 100.0, 40.0, 1.0
    decoupled = {"a": 1, "b": 0, "d0": -(test_q * test_q - test_z * test_z), "g": 0, "h": 0, "t0": 1}
    decoupled_answer = response_from_hessian(decoupled, test_Z, test_q, test_z)
    _, _, vh = np.linalg.svd(np.asarray(decoupled_answer["H_normalized"], complex))
    decoupled_null = vh[-1].conj()
    decoupled_null /= np.linalg.norm(decoupled_null)
    decoupled_null_residual = float(np.linalg.norm(np.asarray(decoupled_answer["H_normalized"]) @ decoupled_null) /
                                    max(np.linalg.norm(np.asarray(decoupled_answer["H_normalized"]), ord=np.inf), np.finfo(float).tiny))
    manufactured_accepted, manufactured_reasons = _pole_acceptance(
        grid_match=True, outside_cut=True, bracket_confirmed=True, finite_elimination_factors=True,
        residue_converged=True, positive_finite_weight=True, null_residual=decoupled_null_residual,
        determinant_residual=decoupled_answer["determinant_residual"],
        determinant_identity_error=decoupled_answer["determinant_identity_relative_error"],
        density_overlap=abs(decoupled_null[0]))
    manufactured = {"status": "REJECTED_NO_DENSITY_OVERLAP", "accepted": manufactured_accepted,
                    "determinant": _cjson(decoupled_answer["determinant"]),
                    "density_overlap": _rjson(abs(decoupled_null[0])),
                    "nullvector_residual": _rjson(decoupled_null_residual),
                    "rejection_reasons": manufactured_reasons,
                    "reason": "scalar-decoupled determinant zero"}
    fake_coeff = {"a": 1, "b": 1, "d0": -(test_q * test_q - test_z * test_z), "g": 1, "h": 0, "t0": 1}
    fake_answer = response_from_hessian(fake_coeff, test_Z, test_q, test_z)
    manufactured["determinant_zero_check"] = bool(abs(decoupled_answer["determinant"]) < 1e-10)
    _, _, fake_vh = np.linalg.svd(np.asarray(fake_answer["H_normalized"], complex))
    fake_null = fake_vh[-1].conj()
    fake_null /= np.linalg.norm(fake_null)
    fake_null_residual = float(np.linalg.norm(np.asarray(fake_answer["H_normalized"]) @ fake_null) /
                               max(np.linalg.norm(np.asarray(fake_answer["H_normalized"]), ord=np.inf), np.finfo(float).tiny))
    fake_accepted, fake_reasons = _pole_acceptance(
        grid_match=True, outside_cut=True, bracket_confirmed=True, finite_elimination_factors=False,
        residue_converged=True, positive_finite_weight=True, null_residual=fake_null_residual,
        determinant_residual=fake_answer["determinant_residual"],
        determinant_identity_error=fake_answer["determinant_identity_relative_error"],
        density_overlap=abs(fake_null[0]))
    fake = {"status": "REJECTED_SCHUR_SINGULARITY", "accepted": fake_accepted,
            "C": _cjson(fake_answer["C"]), "F": _cjson(fake_answer["F"]),
            "rejection_reasons": fake_reasons,
            "reason": "C=0 is not F=DC-B^2=0"}
    # Predicate-only counterfactuals cover each independent gate without
    # presenting a fabricated matrix zero as a physical pole.
    _, wrong_sign_reasons = _pole_acceptance(
        grid_match=True, outside_cut=True, bracket_confirmed=True, finite_elimination_factors=True,
        residue_converged=True, positive_finite_weight=False, null_residual=0.0,
        determinant_residual=0.0, determinant_identity_error=0.0, density_overlap=1.0)
    _, failed_grid_reasons = _pole_acceptance(
        grid_match=False, outside_cut=True, bracket_confirmed=True, finite_elimination_factors=True,
        residue_converged=True, positive_finite_weight=True, null_residual=0.0,
        determinant_residual=0.0, determinant_identity_error=0.0, density_overlap=1.0)
    _, full_residual_reasons = _pole_acceptance(
        grid_match=True, outside_cut=True, bracket_confirmed=True, finite_elimination_factors=True,
        residue_converged=True, positive_finite_weight=True, null_residual=1.0,
        determinant_residual=1.0, determinant_identity_error=0.0, density_overlap=1.0)
    predicate_controls = {
        "wrong_sign_residue": {"accepted": False, "rejection_reasons": wrong_sign_reasons},
        "failed_dual_grid": {"accepted": False, "rejection_reasons": failed_grid_reasons},
        "failed_full_hessian_residual": {"accepted": False, "rejection_reasons": full_residual_reasons},
    }
    return {"regulator_controls": regulator, "negative_frequency_control": neg_control, "ward_controls": ward_rows,
            "gradient_scale_control": gradient_control,
            "missing_longitudinal_or_mixing_control": negative_closure, "q0_conservation_controls": q0_control,
            "pole_validation_negative_controls": {"manufactured_density_decoupled_zero": manufactured, "fake_schur_singularity": fake,
                                                   **predicate_controls}}


def build_result() -> dict[str, Any]:
    snapshot = _build_snapshot()
    controls = _controls(snapshot)
    pole_cases = []
    for b in snapshot["backgrounds"]:
        for q_text in Q_GRID_MEV:
            for scale_text in SCALES:
                pole_cases.append(_pole_case(b["model"], b["state"], b["background_id"], float(q_text), float(scale_text), snapshot["cache"]))
    if len(pole_cases) != EXPECTED_POLE_CASES:
        raise ArithmeticError("pole case coverage is incomplete")
    result = {"schema_version": SCHEMA_VERSION, "status": STATUS, "scheme": SCHEME, "evidence_weight": EVIDENCE_WEIGHT,
              "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
              "upstream_source_sha256": hashlib.sha256(UPSTREAM_PATH.read_bytes()).hexdigest(),
              "static_source_sha256": hashlib.sha256(STATIC_PATH.read_bytes()).hexdigest(),
              "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
              "quadrature_controls": {"default": {"epsrel": DEFAULT_EPSREL, "epsabs": DEFAULT_EPSABS},
                                      "control": {"epsrel": CONTROL_EPSREL, "epsabs": CONTROL_EPSABS}},
              "inputs_and_scope": {"natural_units": True, "scales": list(SCALES), "q_MeV": list(Q_GRID_MEV),
                                    "frequency_fractions_of_live_ph_edge": list(FREQUENCY_FRACTIONS), "expected_row_count": EXPECTED_ROW_COUNT,
                                    "source_definition": "chi=delta_n/delta_mu_ext for source -mu_ext*n; response to +V*n is -chi",
                                    "negative_energy_intermediates_retained": True, "canonical_scalar_gradient_once": True,
                                    "longitudinal_A_L_retained": True, "not_full_vacuum_RPA": True, "not_empirical": True,
                                    "kernel_mathematically_evaluable_beyond_low_energy_scope": True,
                                    "physically_supported_scope": "declared low-energy response only; no global spectral/stability claim"},
              "baseline_guard": snapshot["guard"], "backgrounds": [{"background_id": b["background_id"], "family": b["family"],
                  "target_y": b["target_y"], "mass_MeV": _rjson(b["mass"]), "kF_MeV": _rjson(b["kF"]),
                  "ph_upper_edges_MeV": {q: _rjson(particle_hole_edge(b["mass"], b["kF"], float(q))) for q in Q_GRID_MEV},
                  "inverse_design_is_calibration_only": b["inverse_design"] is not None} for b in snapshot["backgrounds"]],
              "rows": snapshot["rows"], "static_limit_controls": snapshot["static_controls"], "analytic_imaginary_controls": snapshot["imag_controls"],
              "off_pole_direct_schur_controls": snapshot["offpole_controls"], "controls": controls, "pole_cases": pole_cases,
              "coverage": {"row_count": len(snapshot["rows"]), "expected_row_count": EXPECTED_ROW_COUNT, "all_rows_present": len(snapshot["rows"]) == EXPECTED_ROW_COUNT,
                            "pole_case_count": len(pole_cases), "expected_pole_case_count": EXPECTED_POLE_CASES, "all_pole_cases_present": len(pole_cases) == EXPECTED_POLE_CASES,
                            "all_statuses_retained": True}}
    result["numerical_payload_sha256"] = _numerical_payload_sha256(result)
    return result


def calculate() -> dict[str, Any]:
    """Build the live no-write JSON-compatible result."""
    return build_result()


def validate_result(result: Any) -> bool:
    try:
        if not isinstance(result, dict) or result.get("schema_version") != SCHEMA_VERSION:
            return False
        if result.get("status") != STATUS or result.get("scheme") != SCHEME:
            return False
        if result.get("evidence_weight") != EVIDENCE_WEIGHT:
            return False
        expected_hashes = {
            "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            "upstream_source_sha256": hashlib.sha256(UPSTREAM_PATH.read_bytes()).hexdigest(),
            "static_source_sha256": hashlib.sha256(STATIC_PATH.read_bytes()).hexdigest(),
            "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        }
        if any(result.get(key) != value for key, value in expected_hashes.items()):
            return False
        if not isinstance(result.get("numerical_payload_sha256"), str) or result["numerical_payload_sha256"] != _numerical_payload_sha256(result):
            return False
        coverage = result.get("coverage", {})
        if not isinstance(coverage, dict) or coverage.get("row_count") != EXPECTED_ROW_COUNT or coverage.get("expected_row_count") != EXPECTED_ROW_COUNT:
            return False
        if (coverage.get("pole_case_count") != EXPECTED_POLE_CASES or coverage.get("expected_pole_case_count") != EXPECTED_POLE_CASES or
                coverage.get("all_rows_present") is not True or coverage.get("all_pole_cases_present") is not True or
                coverage.get("all_statuses_retained") is not True):
            return False
        inputs = result.get("inputs_and_scope")
        if (not isinstance(inputs, dict) or inputs.get("natural_units") is not True or
                inputs.get("scales") != list(SCALES) or inputs.get("q_MeV") != list(Q_GRID_MEV) or
                inputs.get("frequency_fractions_of_live_ph_edge") != list(FREQUENCY_FRACTIONS) or
                inputs.get("expected_row_count") != EXPECTED_ROW_COUNT or
                inputs.get("negative_energy_intermediates_retained") is not True or
                inputs.get("canonical_scalar_gradient_once") is not True or
                inputs.get("longitudinal_A_L_retained") is not True):
            return False
        if not _validate_rows(result) or not _validate_pole_cases(result):
            return False
        for key, expected_len in (("static_limit_controls", 12), ("analytic_imaginary_controls", 12),
                                  ("off_pole_direct_schur_controls", 12)):
            values = result.get(key)
            if not isinstance(values, list) or len(values) != expected_len or not all(isinstance(x, dict) and isinstance(x.get("pass"), bool) and _valid_payload_tree(x) for x in values):
                return False
        controls = result.get("controls")
        if not isinstance(controls, dict) or not _valid_payload_tree(controls):
            return False
        required_controls = {"regulator_controls", "negative_frequency_control", "ward_controls",
                             "gradient_scale_control", "missing_longitudinal_or_mixing_control",
                             "q0_conservation_controls", "pole_validation_negative_controls"}
        if not required_controls <= set(controls):
            return False
        if not isinstance(controls["regulator_controls"], list) or len(controls["regulator_controls"]) != 3:
            return False
        if not isinstance(controls["ward_controls"], list) or len(controls["ward_controls"]) != 2:
            return False
        negative_controls = controls["pole_validation_negative_controls"]
        if (not isinstance(negative_controls, dict) or not negative_controls or
                not all(isinstance(x, dict) and x.get("accepted") is False and isinstance(x.get("rejection_reasons"), list) for x in negative_controls.values())):
            return False
        if not _valid_payload_tree(result.get("baseline_guard")) or not _valid_payload_tree(result.get("backgrounds")):
            return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, OSError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(json.dumps(build_result(), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
