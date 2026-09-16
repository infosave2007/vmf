#!/usr/bin/env python3
"""High-precision Euler/Noether control on the original manufactured binary jet.

This is a numerical action-identity check, not a physical solution or evidence.
The input generator reproduces the legacy seed-1701 NumPy operations; their
binary64 outputs (including derived couplings) are subsequently treated as exact
control inputs.  No finite-difference step or expected output is fitted.

The density is the original complex scalar plus unsymmetrized Dirac density,
with barred/unbarred commuting complex components varied independently.  The
ordered spinor contractions implement that manufactured-control convention;
this is not a new convention for quantum Grassmann functional integration.
The Maxwell Euler term has identically zero divergence by antisymmetry and is
absent from the legacy local density and this control alike.

Analytic field Euler expressions and source divergence are checked against
independent nested ``mp.diff`` variations of the density at 80 and 120 digits.
NumPy is used only for the declared binary input generator, never for reductions
or differentiation.  Runtime defaults are imported lazily to allow the original
producer to call this module without an import cycle.  CLI writes JSON only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import platform
from typing import Any

import mpmath as mp
import numpy as np


ETA = (1, -1, -1, -1)
GAMMA = (
    ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1)),
    ((0, 0, 0, 1), (0, 0, 1, 0), (0, -1, 0, 0), (-1, 0, 0, 0)),
    ((0, 0, 0, -1j), (0, 0, 1j, 0), (0, 1j, 0, 0), (-1j, 0, 0, 0)),
    ((0, 0, 1, 0), (0, 0, 0, -1), (-1, 0, 0, 0), (0, 1, 0, 0)),
)
REQUIRED_CHECKS = frozenset((
    "euler_phi", "euler_phibar", "euler_N", "euler_bar_N",
    "connection", "connection_divergence",
))
PARAMETER_NAMES = ("W0", "lam", "g_s", "q_phi", "g_omega")
CHECK_TOLERANCE = "1e-60"


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _precision(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer between 70 and 300")
    if not 70 <= value <= 300:
        raise ValueError(f"{name} must be between 70 and 300")
    return int(value)


def _parameters(p: Any = None) -> tuple[float, ...]:
    if p is None:
        from source_complete_solution_audit import PARAMS
        p = PARAMS
    values = []
    for name in PARAMETER_NAMES:
        value = _finite_real(getattr(p, name), name)
        if value <= 0:
            raise ValueError("control parameters must be positive")
        values.append(value)
    return tuple(values)


def _tuples(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _tuples(value.tolist())
    if isinstance(value, (tuple, list)):
        return tuple(_tuples(item) for item in value)
    return value


@dataclass(frozen=True)
class BinaryJet:
    parameters: tuple
    state: tuple
    second_phi: tuple
    second_phibar: tuple
    second_N: tuple
    second_bar_N: tuple
    first_A: tuple


def manufactured_jet(p: Any = None) -> BinaryJet:
    """Reproduce the legacy binary jet, with no call to its FD evaluator."""
    parameters = _parameters(p)
    W0 = parameters[0]
    rng = np.random.default_rng(1701)
    phi = W0 / math.sqrt(2.0) * (1.0 + 0.013j)
    dphi = np.array([0.17-0.09j, -0.11+0.07j, 0.08+0.04j, -0.06-0.05j])
    phibar, dphibar = np.conjugate(phi), np.conjugate(dphi)
    N = np.array([0.42+0.17j, -0.31+0.28j, 0.19-0.37j, 0.27+0.08j])
    dN = np.array([
        [0.11+0.02j, -0.07+0.05j, 0.13-0.09j, 0.03+0.12j],
        [-0.04+0.08j, 0.15-0.03j, 0.06+0.02j, -0.12+0.04j],
        [0.09-0.06j, 0.01+0.13j, -0.16+0.07j, 0.05-0.02j],
        [-0.08+0.11j, 0.03-0.04j, 0.14+0.06j, 0.02+0.09j],
    ], dtype=complex)
    gamma0 = np.array(GAMMA[0], dtype=complex)
    bar_N = np.conjugate(N) @ gamma0
    dbar_N = np.array([np.conjugate(dN[mu]) @ gamma0 for mu in range(4)])
    A = np.array([0.19, -0.23, 0.07, 0.16], dtype=float)
    second_phi = (rng.normal(size=(4, 4)) + 0.3j*rng.normal(size=(4, 4)))*0.01
    second_phibar = np.conjugate(second_phi)
    second_N = (rng.normal(size=(4, 4, 4)) + 0.3j*rng.normal(size=(4, 4, 4)))*0.01
    second_bar_N = np.empty_like(second_N)
    for mu in range(4):
        for nu in range(4):
            second_bar_N[mu, nu] = np.conjugate(second_N[mu, nu]) @ gamma0
    first_A = rng.normal(size=(4, 4))*0.01
    return BinaryJet(parameters, _tuples((phi, phibar, N, bar_N, dphi,
                     dphibar, dN, dbar_N, A)), _tuples(second_phi),
                     _tuples(second_phibar), _tuples(second_N),
                     _tuples(second_bar_N), _tuples(first_A))


def _mp_tree(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return tuple(_mp_tree(item) for item in value)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("jet entries must not be boolean")
    try:
        number = mp.mpc(value) if isinstance(value, (complex, np.complexfloating, mp.mpc)) else mp.mpf(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("jet entries must be numeric") from exc
    if not mp.isfinite(number):
        raise ValueError("jet entries must be finite")
    return number


def input_fingerprint(jet: BinaryJet) -> str:
    """Hash the complete exact-binary passport, not any calculated output."""
    def encode(value):
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if isinstance(value, complex):
            return [float(value.real).hex(), float(value.imag).hex()]
        return float(value).hex()
    payload = [encode(getattr(jet, name)) for name in jet.__dataclass_fields__]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _dot(left, right):
    return mp.fsum(a*b for a, b in zip(left, right))


def _bilinear(left, matrix, right):
    return mp.fsum(left[i]*matrix[i][j]*right[j]
                   for i in range(4) for j in range(4) if matrix[i][j])


def density(state: tuple, parameters: tuple):
    """Full legacy off-shell complex scalar+Dirac density, now in mp arithmetic."""
    W0, lam, gs, q, g = parameters
    phi, bar, N, barN, dp, db, dN, _dbarN, A = state
    W = mp.sqrt(2*bar*phi)
    scalar = mp.fsum(ETA[mu]*(db[mu]+1j*q*A[mu]*bar)*
                     (dp[mu]-1j*q*A[mu]*phi) for mu in range(4))
    scalar -= lam*(bar*phi-W0**2/2)**2
    dirac = mp.fsum(1j*_bilinear(barN, GAMMA[mu],
                      tuple(dN[mu][j]+1j*g*A[mu]*N[j] for j in range(4)))
                     for mu in range(4)) - gs*W*_dot(barN, N)
    return scalar + dirac


def _replace_component(state, field, component, value):
    values = list(state)
    if field in (0, 1):
        values[field] = value
    elif field in (6, 7):
        rows = [list(row) for row in values[field]]
        rows[component//4][component % 4] = value
        values[field] = tuple(map(tuple, rows))
    else:
        row = list(values[field])
        row[component] = value
        values[field] = tuple(row)
    return tuple(values)


def _partial(state, parameters, field, component=0):
    current = (state[field] if field in (0, 1) else
               state[field][component//4][component % 4] if field in (6, 7)
               else state[field][component])
    return mp.diff(lambda value: density(
        _replace_component(state, field, component, value), parameters), current)


def _shifted(jet, state, mu, t):
    phi, bar, N, barN, dp, db, dN, dbarN, A = state
    sp, sb, sN, sbarN, dA = (getattr(jet, name) for name in
        ("second_phi", "second_phibar", "second_N", "second_bar_N", "first_A"))
    return (phi+dp[mu]*t+sp[mu][mu]*t*t/2,
            bar+db[mu]*t+sb[mu][mu]*t*t/2,
            tuple(N[j]+dN[mu][j]*t+sN[mu][mu][j]*t*t/2 for j in range(4)),
            tuple(barN[j]+dbarN[mu][j]*t+sbarN[mu][mu][j]*t*t/2 for j in range(4)),
            tuple(dp[i]+sp[i][mu]*t for i in range(4)),
            tuple(db[i]+sb[i][mu]*t for i in range(4)),
            tuple(tuple(dN[i][j]+sN[i][j][mu]*t for j in range(4)) for i in range(4)),
            tuple(tuple(dbarN[i][j]+sbarN[i][j][mu]*t for j in range(4)) for i in range(4)),
            tuple(A[i]+dA[i][mu]*t for i in range(4)))


def analytic_eulers(jet: BinaryJet) -> dict:
    """Explicit Euler expressions; all arguments must already be mp numbers."""
    W0, lam, gs, q, g = jet.parameters
    phi, bar, N, barN, dp, db, dN, dbarN, A = jet.state
    rho, W = bar*phi, mp.sqrt(2*bar*phi)
    boxp = mp.fsum(ETA[i]*jet.second_phi[i][i] for i in range(4))
    boxb = mp.fsum(ETA[i]*jet.second_phibar[i][i] for i in range(4))
    divA = mp.fsum(ETA[i]*jet.first_A[i][i] for i in range(4))
    A2 = mp.fsum(ETA[i]*A[i]**2 for i in range(4))
    Adp = mp.fsum(ETA[i]*A[i]*dp[i] for i in range(4))
    Adb = mp.fsum(ETA[i]*A[i]*db[i] for i in range(4))
    nn = _dot(barN, N)
    Ep = -boxb-1j*q*divA*bar-2j*q*Adb+q*q*A2*bar-2*lam*(rho-W0**2/2)*bar-gs*bar/W*nn
    Eb = -boxp+1j*q*divA*phi+2j*q*Adp+q*q*A2*phi-2*lam*(rho-W0**2/2)*phi-gs*phi/W*nn
    EN = tuple(mp.fsum((-1j*dbarN[mu][i]-g*A[mu]*barN[i])*GAMMA[mu][i][j]
                      for mu in range(4) for i in range(4))-gs*W*barN[j] for j in range(4))
    EbarN = tuple(mp.fsum(GAMMA[mu][j][i]*(1j*dN[mu][i]-g*A[mu]*N[i])
                         for mu in range(4) for i in range(4))-gs*W*N[j] for j in range(4))
    EA = tuple(1j*q*ETA[mu]*(bar*dp[mu]-db[mu]*phi)+2*q*q*ETA[mu]*A[mu]*rho
               -g*_bilinear(barN, GAMMA[mu], N) for mu in range(4))
    divEA = (1j*q*(bar*boxp-phi*boxb)
             +2*q*q*mp.fsum(ETA[mu]*(jet.first_A[mu][mu]*rho+
                       A[mu]*(db[mu]*phi+bar*dp[mu])) for mu in range(4))
             -g*mp.fsum(_bilinear(dbarN[mu], GAMMA[mu], N)+
                        _bilinear(barN, GAMMA[mu], dN[mu]) for mu in range(4)))
    return dict(euler_phi=Ep, euler_phibar=Eb, euler_N=EN,
                euler_bar_N=EbarN, connection=EA, connection_divergence=divEA)


def differentiated_eulers(jet: BinaryJet) -> dict:
    """Nested variations of L itself, independent of the analytic formulas."""
    state, params = jet.state, jet.parameters
    output = {}
    for key, field, momentum, count in (
        ("euler_phi", 0, 4, 1), ("euler_phibar", 1, 5, 1),
        ("euler_N", 2, 6, 4), ("euler_bar_N", 3, 7, 4),
    ):
        components = []
        for j in range(count):
            divergence = mp.fsum(mp.diff(lambda t: _partial(
                _shifted(jet, state, mu, t), params, momentum,
                mu if count == 1 else 4*mu+j), 0) for mu in range(4))
            components.append(_partial(state, params, field, j)-divergence)
        output[key] = components[0] if count == 1 else tuple(components)
    output["connection"] = tuple(_partial(state, params, 8, mu) for mu in range(4))
    output["connection_divergence"] = mp.fsum(mp.diff(lambda t:
        _partial(_shifted(jet, state, mu, t), params, 8, mu), 0) for mu in range(4))
    return output


def _norm(value):
    return mp.sqrt(mp.fsum(abs(item)**2 for item in value)) if isinstance(value, (tuple, list)) else abs(value)


def _difference(left, right):
    return _norm(tuple(a-b for a, b in zip(left, right))) if isinstance(left, (tuple, list)) else abs(left-right)


def _ward(values, jet, connection_sign):
    _, _, _, q, g = jet.parameters
    phi, bar, N, barN = jet.state[:4]
    contraction = _dot(values["euler_N"], N)
    terms = (-connection_sign*values["connection_divergence"],
             1j*q*(values["euler_phi"]*phi-values["euler_phibar"]*bar),
             -1j*g*contraction, 1j*g*_dot(barN, values["euler_bar_N"]))
    scale = max(mp.mpf(1), abs(values["connection_divergence"]),
                abs(q*values["euler_phi"]*phi), abs(g*contraction))
    return abs(mp.fsum(terms))/scale


def _validate_shape(jet):
    def shaped(value, shape):
        if not shape:
            return not isinstance(value, (tuple, list))
        return (isinstance(value, (tuple, list)) and len(value) == shape[0]
                and all(shaped(item, shape[1:]) for item in value))
    if not isinstance(jet, BinaryJet) or len(jet.state) != 9 or len(jet.parameters) != 5:
        raise ValueError("invalid BinaryJet structure")
    shapes = ((), (), (4,), (4,), (4,), (4,), (4, 4), (4, 4), (4,))
    if not all(shaped(value, shape) for value, shape in zip(jet.state, shapes)):
        raise ValueError("invalid state dimensions")
    for name, shape in (("second_phi", (4, 4)), ("second_phibar", (4, 4)),
                        ("second_N", (4, 4, 4)), ("second_bar_N", (4, 4, 4)),
                        ("first_A", (4, 4))):
        if not shaped(getattr(jet, name), shape):
            raise ValueError(f"invalid {name} dimensions")


def evaluate(jet: BinaryJet, dps: int = 80, connection_sign: float = 1.0) -> dict:
    """Pure raw-mp evaluation, preserving the caller's precision context."""
    dps = _precision(dps, "dps")
    sign = _finite_real(connection_sign, "connection_sign")
    if sign not in (-1.0, 1.0):
        raise ValueError("connection_sign must be +1 or -1")
    _validate_shape(jet)
    with mp.workdps(dps):
        exact = BinaryJet(**{name: _mp_tree(getattr(jet, name)) for name in jet.__dataclass_fields__})
        if any(mp.im(value) != 0 or mp.re(value) <= 0 for value in exact.parameters):
            raise ValueError("control parameters must be positive")
        rho = exact.state[0]*exact.state[1]
        if mp.im(rho) != 0 or mp.re(rho) <= 0:
            raise ValueError("control jet must have positive real bar-phi phi")
        analytic = analytic_eulers(exact)
        independent = differentiated_eulers(exact)
        errors = {name: _difference(analytic[name], independent[name]) /
                  max(mp.mpf(1), _norm(analytic[name]), _norm(independent[name]))
                  for name in REQUIRED_CHECKS}
        _mp_tree(tuple(analytic.values()))
        _mp_tree(tuple(independent.values()))
        return dict(dps=dps, analytic=analytic, independent=independent,
                    errors=errors, analytic_ward=_ward(analytic, exact, sign),
                    independent_ward=_ward(independent, exact, sign),
                    wrong_sign_ward=_ward(independent, exact, -1))


def _decimal(value, digits=35):
    if not mp.isfinite(value):
        raise ValueError("nonfinite derived value")
    return mp.nstr(value, digits)


def require_checks(rows: Any) -> bool:
    if not isinstance(rows, dict) or set(rows) != REQUIRED_CHECKS:
        return False
    for row in rows.values():
        if not isinstance(row, dict) or row.get("passed") is not True:
            return False
        try:
            value = row["relative_error"]
            if isinstance(value, bool):
                return False
            number = mp.mpf(value)
            if not mp.isfinite(number) or not 0 <= number < mp.mpf(CHECK_TOLERANCE):
                return False
        except (TypeError, ValueError, KeyError):
            return False
    return True


def require_result(result: Any) -> bool:
    """Reject missing/truthy passports even if a caller supplies all_pass=True."""
    if not isinstance(result, dict) or result.get("all_pass") is not True:
        return False
    if not (require_checks(result.get("independent_checks"))
            and require_checks(result.get("precision_checks"))):
        return False
    norms = result.get("norms")
    if not isinstance(norms, dict) or set(norms) != REQUIRED_CHECKS:
        return False
    try:
        for value in norms.values():
            if isinstance(value, bool):
                return False
            number = mp.mpf(value)
            if not mp.isfinite(number) or number < 0:
                return False
        residual = result["identity_relative"]
        wrong = result["wrong_connection_sign_relative"]
        if isinstance(residual, bool) or isinstance(wrong, bool):
            return False
        residual, wrong = mp.mpf(residual), mp.mpf(wrong)
        return bool(mp.isfinite(residual) and mp.isfinite(wrong)
                    and 0 <= residual < mp.mpf(CHECK_TOLERANCE)
                    and wrong > mp.mpf("1e-10"))
    except (TypeError, ValueError, KeyError):
        return False


def compute_state(p: Any = None, connection_sign: float = 1.0,
                  dps: int = 80, verify_dps: int = 120) -> dict:
    """Return a JSON-safe, fail-closed passport for this one manufactured jet."""
    dps, verify_dps = _precision(dps, "dps"), _precision(verify_dps, "verify_dps")
    if verify_dps <= dps:
        raise ValueError("verify_dps must exceed dps")
    jet = manufactured_jet(p)
    coarse = evaluate(jet, dps, connection_sign)
    fine = evaluate(jet, verify_dps, connection_sign)
    with mp.workdps(verify_dps):
        tolerance = mp.mpf(CHECK_TOLERANCE)
        checks = {}
        precision_checks = {}
        for name in sorted(REQUIRED_CHECKS):
            error = max(coarse["errors"][name], fine["errors"][name])
            checks[name] = dict(relative_error=_decimal(error), passed=bool(error < tolerance))
            difference = _difference(coarse["independent"][name], fine["independent"][name])
            relative = difference/max(mp.mpf(1), _norm(fine["independent"][name]))
            precision_checks[name] = dict(relative_error=_decimal(relative), passed=bool(relative < tolerance))
        ward = max(coarse["analytic_ward"], coarse["independent_ward"],
                   fine["analytic_ward"], fine["independent_ward"])
        wrong = min(coarse["wrong_sign_ward"], fine["wrong_sign_ward"])
        all_pass = (require_checks(checks) and require_checks(precision_checks)
                    and mp.isfinite(ward) and ward < tolerance
                    and mp.isfinite(wrong) and wrong > mp.mpf("1e-10"))
        norms = {name: _decimal(_norm(fine["independent"][name]))
                 for name in sorted(REQUIRED_CHECKS)}
        return {
            "status": "manufactured_action_identity_precision_control_not_physical_evidence",
            "evidence_weight": 0,
            "method": "analytic_Euler_and_independent_nested_mpmath_diff",
            "input_passport": {"seed": 1701, "rng": "NumPy default_rng PCG64",
                "binary_inputs_are_exact": True, "sha256": input_fingerprint(jet),
                "parameters_binary_hex": {key: value.hex() for key, value in zip(PARAMETER_NAMES, jet.parameters)},
                "scope": "same legacy manufactured jet; not arbitrary-field physical dynamics"},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "mpmath": mp.__version__, "mpmath_backend": mp.libmp.BACKEND},
            "precision_dps": [dps, verify_dps], "relative_tolerance": CHECK_TOLERANCE,
            "independent_checks": checks, "precision_checks": precision_checks,
            "norms": norms, "identity_relative": _decimal(ward),
            "wrong_connection_sign_relative": _decimal(wrong),
            "all_pass": bool(all_pass),
        }


def manufactured_euler_ward_controls(p: Any = None, connection_sign: float = 1.0,
                                   dps: int = 80, verify_dps: int = 120) -> dict:
    """Migration interface: original observable fields, with truthful metadata.

    Norms retain the previous six-decimal output convention, now backed by the
    independent high-precision calculation.  No raw identity residual is clipped
    to zero. ``finite_difference_step=None`` explicitly marks the changed method.
    """
    result = compute_state(p, connection_sign, dps, verify_dps)
    with mp.workdps(verify_dps):
        def number(name, decimals=6):
            value = mp.mpf(result["norms"][name])
            if not mp.isfinite(value):
                raise ValueError("nonfinite norm")
            return _finite_real(round(value, decimals), name)
        return {
            "identity_relative": _finite_real(result["identity_relative"], "identity_relative"),
            "connection_divergence_abs": _finite_real(mp.nstr(mp.mpf(result["norms"]["connection_divergence"]), 13), "connection_divergence_abs"),
            "euler_phi_abs": number("euler_phi"), "euler_phibar_abs": number("euler_phibar"),
            "euler_N_abs": number("euler_N"), "euler_bar_N_abs": number("euler_bar_N"),
            "finite_difference_step": None,
            "method": result["method"], "precision_dps": result["precision_dps"],
            "input_sha256": result["input_passport"]["sha256"],
            "independent_variation_checks_pass": require_checks(result["independent_checks"]),
            "precision_convergence_pass": require_checks(result["precision_checks"]),
            "all_pass": require_result(result),
        }


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None) -> int:
    parser = JsonArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--verify-dps", type=int, default=120)
    try:
        args = parser.parse_args(argv)
        if args.help:
            print(json.dumps({"status": "help", "all_pass": False,
                              "evidence_weight": 0, "help": parser.format_help()}))
            return 0
        result = compute_state(dps=args.dps, verify_dps=args.verify_dps)
        result["all_pass"] = require_result(result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0 if result["all_pass"] is True else 1
    except Exception as exc:
        print(json.dumps({"all_pass": False, "evidence_weight": 0,
                          "error": str(exc)}, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
