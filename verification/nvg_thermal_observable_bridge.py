#!/usr/bin/env python3
"""Live finite-temperature thermal/susceptibility bridge for the W8/U16 action.

This producer rebuilds the accepted cold calibration and evaluates the same
homogeneous, symmetric (d=4) action in the grand-canonical ensemble.  Thermal
integrals are computed in the current process; the result JSON is an output
snapshot only and is never a physical input.  The public grid is deliberately
small and contextual: T=70 MeV and mu=775,875 MeV, for y*=0.90,0.93 and the
alpha=0/1 calibration-null models.

The implementation is local-branch thermodynamics.  Root discovery reports a
bounded set of starts and all distinct stable roots found in that set, but it
does not make a global phase-equilibrium claim.  Higher susceptibilities use
implicit differentiation of the two residual equations; finite differences
are retained only as independent controls.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# These are maintained public verification modules.  They contain the accepted
# calibration constructor; no run-workspace path or saved result is imported.
import nvg_nonlinear_calibration_response as nonlinear  # noqa: E402

SCHEMA_VERSION = "nvg_thermal_observable_bridge.v1"
STATUS_PASS = "PASS_LIVE_THERMAL_OBSERVABLE_BRIDGE"
STATUS_FAIL = "FAIL_LIVE_THERMAL_OBSERVABLE_BRIDGE"
EVIDENCE_WEIGHT = mp.mpf("0")
ANCHORS = ("0.90", "0.93")
MODEL_ORDER = ("w8", "u16")
MODEL_AMPLITUDES = {"w8": 0, "u16": 1}
TEMPERATURES = ("70",)
CHEMICAL_POTENTIALS = ("775", "875")
DEGENERACY = 4
HBARC_MEV_FM = "197.3269804"
N0_FM3 = "0.16"
BINDING_MEV = "-16"
K_TARGET_MEV = "240"
PRIMARY_DPS = 32
CONTROL_DPS = 38
PRIMARY_NQUAD = 80
CONTROL_NQUAD = 120
PRIMARY_CUTOFF_MEV = "4500"
CONTROL_CUTOFF_MEV = "5200"
TAIL_CUTOFF_TOL = mp.mpf("2e-12")
ROOT_REL_LIMIT = mp.mpf("2e-28")
IDENTITY_REL_LIMIT = mp.mpf("2e-20")
SCHUR_IDENTITY_LIMIT = mp.mpf("1e-20")
DERIVATIVE_REL_LIMIT = mp.mpf("5e-3")
PRECISION_REL_LIMIT = mp.mpf("3e-9")
FD_STEPS_MEV = ("2", "1")
STATIONARY_STEP_MEV = "0.5"
ROOT_Y_MIN = mp.mpf("0.10")
ROOT_Y_MAX = mp.mpf("3.0")
ROOT_NU_MARGIN = mp.mpf("240")
# A deliberately bounded root-start grid.  The starts are probes, not a claim
# that every stationary root is exposed; failures are retained in the output.
ROOT_GRID_NU_OFFSETS = (mp.mpf("0"),)
ROOT_GRID_Y = (mp.mpf("0.90"), mp.mpf("1.30"))
BRANCH_LINEAGE = "bounded_root_scan_nearest_anchor_local_stable_branch"
PRESSURE_UNITS = "MeV fm^-3"
DENSITY_UNITS = "fm^-3"
ENTROPY_UNITS = "fm^-3"
SUSCEPTIBILITY_UNITS = {
    "chi1": "fm^-3 MeV^-1",
    "chi2": "fm^-3 MeV^-2",
    "chi3": "fm^-3 MeV^-3",
}


class ThermalBridgeError(ValueError):
    """An invalid input or failed numerical/physics control."""


def _mp(value: Any) -> mp.mpf:
    if isinstance(value, mp.mpf):
        return value
    if isinstance(value, bool):
        raise ThermalBridgeError("boolean is not a finite scalar")
    try:
        value = mp.mpf(str(value))
    except (TypeError, ValueError) as exc:
        raise ThermalBridgeError("value must be a finite real scalar") from exc
    if not mp.isfinite(value):
        raise ThermalBridgeError("value must be finite")
    return value


def _finite(value: Any) -> bool:
    try:
        return bool(mp.isfinite(_mp(value)))
    except (TypeError, ValueError, ThermalBridgeError):
        return False


def _number(value: Any, digits: int = 36) -> str:
    value = _mp(value)
    if not mp.isfinite(value):
        raise ThermalBridgeError("scientific output is nonfinite")
    return mp.nstr(value, digits)


def _relative(a: Any, b: Any, *, floor: Any = "1e-45") -> mp.mpf:
    aa, bb = _mp(a), _mp(b)
    return abs(aa - bb) / max(abs(aa), abs(bb), _mp(floor))


def _relative_scale(value: Any, scale: Any, *, floor: Any = "1e-45") -> mp.mpf:
    return abs(_mp(value)) / max(abs(_mp(scale)), _mp(floor))


def _jsonable(value: Any) -> Any:
    if isinstance(value, mp.mpf):
        return _number(value)
    if isinstance(value, mp.mpc):
        raise ThermalBridgeError("complex scientific value")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _log1pexp(x: mp.mpf) -> mp.mpf:
    """Stable log(1+exp(x)) for real high-precision arguments."""
    x = _mp(x)
    if x > 40:
        return x + mp.log1p(mp.exp(-x))
    return mp.log1p(mp.exp(x))


def _fermi_argument(x: mp.mpf) -> mp.mpf:
    """Stable logistic 1/(exp(x)+1), where x=(E-s*nu)/T."""
    x = _mp(x)
    if x > 40:
        q = mp.exp(-x)
        return q / (1 + q)
    if x < -40:
        q = mp.exp(x)
        return 1 / (1 + q)
    return 1 / (mp.exp(x) + 1)


def _entropy_occupation(x: mp.mpf, f: mp.mpf) -> mp.mpf:
    # For f=(e^x+1)^-1 this is log(1+e^-x)+x f, with no log(0).
    return _log1pexp(-_mp(x)) + _mp(x) * _mp(f)


@lru_cache(maxsize=32)
def _gauss_grid(dps: int, count: int, cutoff_text: str, focus_text: str = "", focus_width_text: str = "") -> tuple[tuple[mp.mpf, mp.mpf], ...]:
    """Build a deterministic high-precision Gauss-Legendre grid.

    At low temperature ``focus_text`` is the Fermi momentum and the interval
    is split around p_F.  This is still fixed quadrature (not a hidden fit),
    but resolves the narrow Fermi edge in the T->0 control.
    """
    with mp.workdps(dps):
        cutoff = _mp(cutoff_text)
        focus = None if not focus_text else _mp(focus_text)
        if cutoff <= 0:
            raise ThermalBridgeError("momentum cutoff must be positive")
        if focus is None:
            intervals = ((mp.mpf(0), cutoff),)
            per_interval = count
        else:
            # Width is an explicit cache argument.  It must not come from
            # mutable module state because focused low-T grids are physics
            # inputs to the quadrature, not merely an implementation detail.
            width = _mp(focus_width_text or "30")
            if width <= 0:
                raise ThermalBridgeError("focused quadrature width must be positive")
            left = max(mp.mpf(0), focus - width)
            right = min(cutoff, focus + width)
            points = [mp.mpf(0), left, right, cutoff]
            intervals = tuple((points[i], points[i + 1]) for i in range(3) if points[i + 1] > points[i])
            per_interval = max(48, count // max(1, len(intervals)))
        nodes, weights = mp.gauss_quadrature(per_interval, "legendre")
        result: list[tuple[mp.mpf, mp.mpf]] = []
        for low, high in intervals:
            midpoint = (low + high) / 2
            half = (high - low) / 2
            result.extend((midpoint + half * nodes[i], half * weights[i]) for i in range(per_interval))
        return tuple(result)


@dataclass
class ThermalIntegrator:
    """Finite-T particle/antiparticle integrals in natural units."""

    temperature: mp.mpf
    cutoff: mp.mpf
    nquad: int
    focused: bool = False
    focus_p: mp.mpf | None = None

    def __post_init__(self) -> None:
        self.temperature = _mp(self.temperature)
        self.cutoff = _mp(self.cutoff)
        self.nquad = int(self.nquad)
        if self.temperature <= 0 or self.cutoff <= 0 or self.nquad < 32:
            raise ThermalBridgeError("invalid thermal quadrature configuration")
        focus_text = ""
        focus_width_text = ""
        if self.focused and self.focus_p is not None:
            focus_text = _number(self.focus_p, 30)
            focus_width_text = _number(max(mp.mpf("24") * self.temperature, mp.mpf("30")), 30)
        self._grid = _gauss_grid(mp.mp.dps, self.nquad, _number(self.cutoff, 30), focus_text, focus_width_text)
        self._cache: dict[tuple[str, str], dict[str, mp.mpf]] = {}

    def _key(self, nu: mp.mpf, y: mp.mpf) -> tuple[str, str]:
        # Never round away the tiny perturbations used by mp.findroot/mp.diff;
        # a dps-minus-five cache key can collapse a derivative stencil to one
        # quadrature point and make the Newton Jacobian spuriously singular.
        return (_number(nu, mp.mp.dps + 8), _number(y, mp.mp.dps + 8))

    def integrals(self, nu: Any, y: Any, *, mass_scale: Any | None = None) -> dict[str, mp.mpf]:
        """Return N, scalar density, kinetic P/e/s, and log-pressure."""
        nu, y = _mp(nu), _mp(y)
        m = self.mass(nu, y) if mass_scale is None else _mp(mass_scale)
        if m <= 0:
            raise ThermalBridgeError("effective mass must be positive")
        key = self._key(nu, y) if mass_scale is None else (self._key(nu, y)[0], _number(m, 30))
        if key in self._cache and mass_scale is None:
            return self._cache[key]
        t = self.temperature
        pref = mp.mpf(DEGENERACY) / (2 * mp.pi**2)
        p_sum_n: list[mp.mpf] = []
        p_sum_s: list[mp.mpf] = []
        p_sum_p: list[mp.mpf] = []
        p_sum_e: list[mp.mpf] = []
        p_sum_h: list[mp.mpf] = []
        p_sum_l: list[mp.mpf] = []
        for p, weight in self._grid:
            e = mp.sqrt(p * p + m * m)
            xp = (e - nu) / t
            xa = (e + nu) / t
            fp = _fermi_argument(xp)
            fa = _fermi_argument(xa)
            fs = fp + fa
            fd = fp - fa
            p2 = p * p
            p_sum_n.append(weight * p2 * fd)
            p_sum_s.append(weight * p2 * (m / e) * fs)
            p_sum_p.append(weight * p2 * p2 / e * fs)
            p_sum_e.append(weight * p2 * e * fs)
            p_sum_h.append(weight * p2 * (_entropy_occupation(xp, fp) + _entropy_occupation(xa, fa)))
            p_sum_l.append(weight * p2 * (_log1pexp(-xp) + _log1pexp(-xa)))
        result = {
            "n": pref * mp.fsum(p_sum_n),
            "ns": pref * mp.fsum(p_sum_s),
            "pressure": pref / 3 * mp.fsum(p_sum_p),
            "energy": pref * mp.fsum(p_sum_e),
            "entropy": pref * mp.fsum(p_sum_h),
            "pressure_log": pref * t * mp.fsum(p_sum_l),
        }
        if mass_scale is None:
            self._cache[key] = result
        return result

    @staticmethod
    def mass(nu: Any, y: Any) -> mp.mpf:
        del nu
        return _mp(_THERMAL_MASS_SCALE) * _mp(y)


# Set by each compute context; keeping this a string avoids an ambient float.
_THERMAL_MASS_SCALE = "939"


def _integrator_for(temperature: Any, *, cutoff: Any = PRIMARY_CUTOFF_MEV, nquad: int = PRIMARY_NQUAD, focused: bool = False, focus_p: Any | None = None) -> ThermalIntegrator:
    return ThermalIntegrator(_mp(temperature), _mp(cutoff), int(nquad), focused=focused, focus_p=None if focus_p is None else _mp(focus_p))


def _model_potential(model: Any, y: mp.mpf) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    values = model.potential(y)
    return tuple(_mp(value) for value in values)  # type: ignore[return-value]


def _residuals(integrator: ThermalIntegrator, model: Any, mu: mp.mpf, nu: mp.mpf, y: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    data = integrator.integrals(nu, y)
    _, uy, _ = _model_potential(model, y)
    number = nu + model.Cv * data["n"] / y**2 - mu
    gap = model.MN * data["ns"] + uy - model.Cv * data["n"]**2 / y**3
    return number, gap


def _root_residual_scale(integrator: ThermalIntegrator, model: Any, nu: mp.mpf, y: mp.mpf) -> mp.mpf:
    data = integrator.integrals(nu, y)
    _, uy, _ = _model_potential(model, y)
    return max(abs(nu), abs(model.MN * data["ns"]), abs(uy), abs(model.Cv * data["n"]**2 / y**3), mp.mpf(1))


def _solve_start(integrator: ThermalIntegrator, model: Any, mu: mp.mpf, nu0: mp.mpf, y0: mp.mpf) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    if y0 <= 0:
        raise ThermalBridgeError("root start y must be positive")
    cache: dict[tuple[str, str], tuple[mp.mpf, mp.mpf]] = {}

    def values(nu: Any, y: Any) -> tuple[mp.mpf, mp.mpf]:
        nu, y = _mp(nu), _mp(y)
        key = (_number(nu, mp.mp.dps + 8), _number(y, mp.mp.dps + 8))
        if key not in cache:
            cache[key] = _residuals(integrator, model, mu, nu, y)
        return cache[key]

    try:
        root = mp.findroot(
            (lambda nu, y: values(nu, y)[0], lambda nu, y: values(nu, y)[1]),
            (nu0, y0), solver="mdnewton", tol=mp.eps ** mp.mpf("0.70"), maxsteps=45, verify=False,
        )
    except Exception as exc:
        raise ThermalBridgeError(f"root solve failed: {type(exc).__name__}: {exc}") from exc
    nu, y = _mp(root[0]), _mp(root[1])
    if not (mp.isfinite(nu) and mp.isfinite(y) and ROOT_Y_MIN <= y <= ROOT_Y_MAX and abs(nu) < 5000):
        raise ThermalBridgeError("root leaves bounded thermal window")
    r1, r2 = values(nu, y)
    scale = _root_residual_scale(integrator, model, nu, y)
    rel = max(abs(r1), abs(r2)) / scale
    if rel > ROOT_REL_LIMIT:
        raise ThermalBridgeError(f"root residual too large: {_number(rel, 10)}")
    n = integrator.integrals(nu, y)["n"]
    if not (mp.isfinite(n) and n > 0):
        raise ThermalBridgeError("root has nonpositive net density")
    return nu, y, rel, n


def _same_root(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return abs(_mp(a["nu"]) - _mp(b["nu"])) < mp.mpf("1e-18") and abs(_mp(a["y"]) - _mp(b["y"])) < mp.mpf("1e-18")


def _partial(fun: Callable[..., mp.mpf], z: Sequence[mp.mpf], indices: Sequence[int]) -> mp.mpf:
    """High-precision automatic partial derivative of a scalar function."""
    if not indices:
        return _mp(fun(*z))
    index = int(indices[0])
    base = tuple(z)

    def reduced(x: Any) -> mp.mpf:
        point = list(base)
        point[index] = _mp(x)
        return _partial(fun, point, indices[1:])

    return _mp(mp.diff(reduced, base[index]))


def _contract2(tensor: Sequence[Sequence[mp.mpf]], a: Sequence[mp.mpf], b: Sequence[mp.mpf]) -> mp.mpf:
    return sum(tensor[i][j] * a[i] * b[j] for i in range(2) for j in range(2))


def _contract3(tensor: Sequence[Sequence[Sequence[mp.mpf]]], a: Sequence[mp.mpf], b: Sequence[mp.mpf], c: Sequence[mp.mpf]) -> mp.mpf:
    return sum(tensor[i][j][k] * a[i] * b[j] * c[k] for i in range(2) for j in range(2) for k in range(2))


def _solve_linear(matrix: Sequence[Sequence[mp.mpf]], rhs: Sequence[mp.mpf]) -> tuple[mp.mpf, mp.mpf]:
    solution = mp.lu_solve(mp.matrix(matrix), mp.matrix(rhs))
    return _mp(solution[0]), _mp(solution[1])


def _implicit_derivatives(integrator: ThermalIntegrator, model: Any, mu: mp.mpf, nu: mp.mpf, y: mp.mpf) -> dict[str, Any]:
    z = (nu, y)

    def g1(a: Any, b: Any) -> mp.mpf:
        return _residuals(integrator, model, mu, _mp(a), _mp(b))[0]

    def g2(a: Any, b: Any) -> mp.mpf:
        return _residuals(integrator, model, mu, _mp(a), _mp(b))[1]

    def number(a: Any, b: Any) -> mp.mpf:
        return integrator.integrals(_mp(a), _mp(b))["n"]

    functions = (g1, g2)
    jac: list[list[mp.mpf]] = [[_partial(fn, z, (i,)) for i in range(2)] for fn in functions]
    hess: list[list[list[mp.mpf]]] = [
        [[_partial(fn, z, (i, j)) for j in range(2)] for i in range(2)] for fn in functions
    ]
    third: list[list[list[list[mp.mpf]]]] = [
        [[[ _partial(fn, z, (i, j, k)) for k in range(2)] for j in range(2)] for i in range(2)] for fn in functions
    ]
    z1 = _solve_linear(jac, (mp.mpf(1), mp.mpf(0)))
    z2_rhs = tuple(-_contract2(hess[a], z1, z1) for a in range(2))
    z2 = _solve_linear(jac, z2_rhs)
    z3_rhs = tuple(
        -(_contract3(third[a], z1, z1, z1) + 3 * _contract2(hess[a], z1, z2)) for a in range(2)
    )
    z3 = _solve_linear(jac, z3_rhs)
    n1 = tuple(_partial(number, z, (i,)) for i in range(2))
    n2 = [[_partial(number, z, (i, j)) for j in range(2)] for i in range(2)]
    n3 = [[[ _partial(number, z, (i, j, k)) for k in range(2)] for j in range(2)] for i in range(2)]
    chi1 = sum(n1[i] * z1[i] for i in range(2))
    chi2 = _contract2(n2, z1, z1) + sum(n1[i] * z2[i] for i in range(2))
    chi3 = _contract3(n3, z1, z1, z1) + 3 * sum(n2[i][j] * z1[i] * z2[j] for i in range(2) for j in range(2)) + sum(n1[i] * z3[i] for i in range(2))
    return {
        "jacobian": jac,
        "z_mu": z1,
        "z_mu2": z2,
        "z_mu3": z3,
        "chi1": chi1,
        "chi2": chi2,
        "chi3": chi3,
        "method": "implicit_function_chain_rule_mp_diff_partial_integrals_orders_1_to_3",
    }


def _state(integrator: ThermalIntegrator, model: Any, mu: mp.mpf, nu: mp.mpf, y: mp.mpf, *, with_derivatives: bool = True) -> dict[str, Any]:
    data = integrator.integrals(nu, y)
    u, uy, uyy = _model_potential(model, y)
    n = data["n"]
    vector = model.Cv * n**2 / (2 * y**2)
    pressure = data["pressure"] - u + vector
    pressure_log = data["pressure_log"] - u + vector
    energy = data["energy"] + u + vector
    gap = model.MN * data["ns"] + uy - model.Cv * n**2 / y**3
    state: dict[str, Any] = {
        "T_MeV": integrator.temperature,
        "mu_MeV": mu,
        "nu_MeV": nu,
        "y": y,
        "mstar_MeV": model.MN * y,
        "n_MeV3": n,
        "n_fm3": n / model.hbarc**3,
        "ns_MeV3": data["ns"],
        "pressure_kinetic_MeV4": data["pressure"],
        "pressure_log_MeV4": data["pressure_log"],
        "pressure_MeV4": pressure,
        "pressure_MeV_fm3": pressure / model.hbarc**3,
        "energy_MeV4": energy,
        "energy_MeV_fm3": energy / model.hbarc**3,
        "entropy_MeV3": data["entropy"],
        "entropy_fm3": data["entropy"] / model.hbarc**3,
        "U_MeV4": u,
        "Uy_MeV4": uy,
        "Uyy_MeV4": uyy,
        "vector_MeV4": vector,
        "gap_residual": gap,
    }
    gap_scale = max(abs(model.MN * data["ns"]), abs(uy), abs(model.Cv * n**2 / y**3), mp.mpf(1))
    state["gap_residual_relative"] = abs(gap) / gap_scale
    state["pressure_kinetic_log_relative"] = _relative(data["pressure"], data["pressure_log"], floor="1e-35")
    gibbs = energy + pressure - mu * n - integrator.temperature * data["entropy"]
    state["gibbs_identity_relative"] = abs(gibbs) / max(abs(energy), abs(pressure), abs(mu * n), mp.mpf(1))
    # Canonical fixed-T curvature identities from the parent acceptance note.
    def n_nu(a: Any) -> mp.mpf:
        return integrator.integrals(_mp(a), y)["n"]

    def ns_nu(a: Any) -> mp.mpf:
        return integrator.integrals(_mp(a), y)["ns"]

    def ns_m(a: Any) -> mp.mpf:
        return integrator.integrals(nu, _mp(a) / model.MN)["ns"]

    nnu = _mp(mp.diff(n_nu, nu))
    snu = _mp(mp.diff(ns_nu, nu))
    sm = _mp(mp.diff(ns_m, model.MN * y))
    A = 1 / nnu + model.Cv / y**2
    B = model.MN * snu / nnu - 2 * model.Cv * n / y**3
    C = uyy + model.MN**2 * (sm + snu**2 / nnu) + 3 * model.Cv * n**2 / y**4
    f_nn = A - B**2 / C
    state.update({"N_nu": nnu, "S_nu": snu, "S_m": sm, "A_fnn": A, "B_fny": B, "C_yy_fixed_n": C, "f_nn": f_nn, "local_stable_C_positive": bool(C > 0), "local_stable_fnn_positive": bool(f_nn > 0)})
    if with_derivatives:
        derivatives = _implicit_derivatives(integrator, model, mu, nu, y)
        chi1, chi2, chi3 = derivatives["chi1"], derivatives["chi2"], derivatives["chi3"]
        state["derivatives"] = derivatives
        state["chi1_MeV2"] = chi1
        state["chi2_MeV"] = chi2
        state["chi3_dimensionless"] = chi3
        state["chi1_fm3_per_MeV"] = chi1 / model.hbarc**3
        state["chi2_fm3_per_MeV2"] = chi2 / model.hbarc**3
        state["chi3_fm3_per_MeV3"] = chi3 / model.hbarc**3
        state["chi1_vs_schur_relative"] = _relative(chi1, 1 / f_nn, floor="1e-35")
        state["R21"] = integrator.temperature * chi1 / n
        state["R32"] = integrator.temperature * chi2 / chi1
        state["R42"] = integrator.temperature**2 * chi3 / chi1
    return state


def _root_record(integrator: ThermalIntegrator, model: Any, mu: mp.mpf, nu0: mp.mpf, y0: mp.mpf, *, with_derivatives: bool = False) -> dict[str, Any]:
    nu, y, residual, n = _solve_start(integrator, model, mu, nu0, y0)
    state = _state(integrator, model, mu, nu, y, with_derivatives=with_derivatives)
    return {"nu": nu, "y": y, "residual_relative": residual, "n": n, "state": state, "stable": bool(state["local_stable_C_positive"] and state["local_stable_fnn_positive"]), "branch_lineage": BRANCH_LINEAGE}


def _discover_roots(integrator: ThermalIntegrator, model: Any, mu: mp.mpf, anchor_y: mp.mpf, *, starts: Sequence[tuple[mp.mpf, mp.mpf]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if starts is None:
        starts = tuple((mu + delta, y) for delta in ROOT_GRID_NU_OFFSETS for y in ROOT_GRID_Y)
    attempts: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    for nu0, y0 in starts:
        item: dict[str, Any] = {"nu_start_MeV": nu0, "y_start": y0}
        try:
            root = _root_record(integrator, model, mu, nu0, y0, with_derivatives=False)
            item.update({"status": "PASS", "nu_root_MeV": root["nu"], "y_root": root["y"], "residual_relative": root["residual_relative"], "positive_n": True, "local_stable": root["stable"]})
            if not any(_same_root(root, existing) for existing in roots):
                roots.append(root)
        except Exception as exc:
            item.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        attempts.append(item)
    if not roots:
        raise ThermalBridgeError("bounded root scan found no positive root")
    # Re-evaluate every distinct root with the stability curvature.  Keep roots
    # even if unstable so the report does not silently discard a branch.
    for root in roots:
        root["state"] = _state(integrator, model, mu, root["nu"], root["y"], with_derivatives=False)
        root["stable"] = bool(root["state"]["local_stable_C_positive"] and root["state"]["local_stable_fnn_positive"])
    return roots, attempts


def _select_root(roots: Sequence[dict[str, Any]], anchor_y: mp.mpf) -> dict[str, Any]:
    stable = [root for root in roots if root["stable"]]
    if not stable:
        raise ThermalBridgeError("no locally stable thermal root in bounded scan")
    # This is a declared nearest-anchor local selection, not a demonstrated
    # thermal/cold continuation path and not a grand-potential/global minimum
    # claim.  Every discovered root remains in the output.
    return min(stable, key=lambda root: (abs(root["y"] - anchor_y), abs(root["nu"])))


def _solve_selected(integrator: ThermalIntegrator, model: Any, mu: mp.mpf, anchor_y: mp.mpf, *, starts: Sequence[tuple[mp.mpf, mp.mpf]] | None = None) -> dict[str, Any]:
    roots, attempts = _discover_roots(integrator, model, mu, anchor_y, starts=starts)
    selected = _select_root(roots, anchor_y)
    state = _state(integrator, model, mu, selected["nu"], selected["y"], with_derivatives=True)
    return {"roots": roots, "attempts": attempts, "selected": selected, "state": state, "distinct_root_count": len(roots), "stable_root_count": sum(int(root["stable"]) for root in roots)}


def _state_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    # Preserve the primary state plus all controls; model objects never cross the
    # JSON boundary.  A shallow copy is enough because publicization is recursive.
    return dict(state)


def _finite_difference_controls(model: Any, base: dict[str, Any], temperature: mp.mpf, mu: mp.mpf, *, cutoff: mp.mpf, nquad: int) -> dict[str, Any]:
    selected = base["selected"]
    root_start = ((selected["nu"], selected["y"]),)
    state0 = base["state"]
    controls: list[dict[str, Any]] = []
    for h_text in FD_STEPS_MEV:
        h = _mp(h_text)
        states: dict[int, dict[str, Any]] = {}
        for multiple in (-2, -1, 1, 2):
            integrator = _integrator_for(temperature, cutoff=cutoff, nquad=nquad)
            target_mu = mu + multiple * h
            # The base branch is already identified by the bounded scan.  A
            # fresh one-start solve at each displaced chemical potential is a
            # continuation control, not a second expensive root enumeration.
            nu_guess = _mp(selected["nu"]) + (target_mu - mu)
            fresh = _root_record(integrator, model, target_mu, nu_guess, _mp(selected["y"]), with_derivatives=False)
            states[multiple] = fresh["state"]
        n0 = state0["n_MeV3"]
        nm2, nm1, np1, np2 = (states[multiple]["n_MeV3"] for multiple in (-2, -1, 1, 2))
        chi1_fd = (np1 - nm1) / (2 * h)
        chi2_fd = (np1 - 2 * n0 + nm1) / h**2
        chi3_fd = (-nm2 + 2 * nm1 - 2 * np1 + np2) / (2 * h**3)
        exact = (state0["chi1_MeV2"], state0["chi2_MeV"], state0["chi3_dimensionless"])
        estimates = (chi1_fd, chi2_fd, chi3_fd)
        errors = tuple(_relative(a, b, floor="1e-30") for a, b in zip(estimates, exact))
        controls.append({
            "h_MeV": h,
            "chi1_fd_MeV2": chi1_fd,
            "chi2_fd_MeV": chi2_fd,
            "chi3_fd_dimensionless": chi3_fd,
            "chi1_relative_error": errors[0],
            "chi2_relative_error": errors[1],
            "chi3_relative_error": errors[2],
            "max_relative_error": max(errors),
            "fresh_root_residual_max": max(states[m]["gap_residual_relative"] for m in states),
            "pass": bool(max(errors) <= DERIVATIVE_REL_LIMIT),
        })
    if not controls[1]["chi3_relative_error"] < controls[0]["chi3_relative_error"]:
        # A shallow finite difference can occasionally be dominated by quadrature
        # noise; retain the measured trend and fail explicitly rather than hiding it.
        controls[-1]["trend_pass"] = False
    else:
        controls[-1]["trend_pass"] = True
    return {"steps": controls, "all_pass": bool(all(row["pass"] for row in controls) and controls[-1]["trend_pass"])}


def _stationary_derivative_controls(model: Any, base: dict[str, Any], temperature: mp.mpf, mu: mp.mpf, *, cutoff: mp.mpf, nquad: int) -> dict[str, Any]:
    h = _mp(STATIONARY_STEP_MEV)
    selected = base["selected"]
    def pressure_at(temp: mp.mpf, chem: mp.mpf) -> dict[str, Any]:
        integrator = _integrator_for(temp, cutoff=cutoff, nquad=nquad)
        nu_guess = _mp(selected["nu"]) + (chem - mu)
        fresh = _root_record(integrator, model, chem, nu_guess, _mp(selected["y"]), with_derivatives=False)
        return fresh["state"]
    plus_mu, minus_mu = pressure_at(temperature, mu + h), pressure_at(temperature, mu - h)
    plus_t, minus_t = pressure_at(temperature + h, mu), pressure_at(temperature - h, mu)
    p_mu_fd = (plus_mu["pressure_MeV4"] - minus_mu["pressure_MeV4"]) / (2 * h)
    p_t_fd = (plus_t["pressure_MeV4"] - minus_t["pressure_MeV4"]) / (2 * h)
    return {
        "h_MeV": h,
        "P_mu_fd_MeV3": p_mu_fd,
        "P_T_fd_MeV3": p_t_fd,
        "n_MeV3": base["state"]["n_MeV3"],
        "s_MeV3": base["state"]["entropy_MeV3"],
        "P_mu_relative_error": _relative(p_mu_fd, base["state"]["n_MeV3"], floor="1e-25"),
        "P_T_relative_error": _relative(p_t_fd, base["state"]["entropy_MeV3"], floor="1e-25"),
        "pass": bool(_relative(p_mu_fd, base["state"]["n_MeV3"], floor="1e-25") < mp.mpf("2e-4") and _relative(p_t_fd, base["state"]["entropy_MeV3"], floor="1e-25") < mp.mpf("2e-4")),
    }


def _quadrature_controls(model: Any, selected: dict[str, Any], temperature: mp.mpf, mu: mp.mpf, anchor_y: mp.mpf, *, cutoff: mp.mpf, nquad: int, reference: Mapping[str, Any]) -> dict[str, Any]:
    # The alternate grid is a fresh root solve, not evaluation of serialized rows.
    fresh = _root_record(_integrator_for(temperature, cutoff=cutoff, nquad=nquad), model, mu, _mp(selected["nu"]), _mp(selected["y"]), with_derivatives=True)
    alt = fresh["state"]
    fields = ("n_fm3", "y", "pressure_MeV_fm3", "entropy_fm3", "R21", "R32", "R42")
    differences = {field: _relative(alt[field], reference[field], floor="1e-30") for field in fields}
    return {
        "alternate_nquad": nquad,
        "alternate_cutoff_MeV": cutoff,
        "alternate_state": alt,
        "relative_differences": differences,
        "max_relative_difference": max(differences.values()),
        "cutoff_tail_bound_claim": "cutoff tail numerically resolved by alternate cutoff; no analytic infinity claim",
        "pass": bool(max(differences.values()) < PRECISION_REL_LIMIT),
    }


def _model_info(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": info["id"],
        "amplitude": info["amplitude"],
        "target_y": info["target_y"],
        "eta_MeV4": info["eta_MeV4"],
        "design": info["design"],
    }


def _solve_fixed_density_lowT(model: Any, anchor_y: mp.mpf, temperature: mp.mpf, *, nquad: int = 240, cutoff: Any = "3000") -> dict[str, Any]:
    n_target = model.n0
    kf = (6 * mp.pi**2 * n_target / model.d) ** (mp.mpf(1) / 3)
    integrator = _integrator_for(temperature, cutoff=cutoff, nquad=nquad, focused=True, focus_p=kf)
    cold = model.state(n_target, anchor_y)
    nu0 = cold["mu"] - model.Cv * n_target / anchor_y**2
    cache: dict[tuple[str, str], tuple[mp.mpf, mp.mpf]] = {}
    def equations(nu: Any, y: Any) -> tuple[mp.mpf, mp.mpf]:
        nu, y = _mp(nu), _mp(y)
        key = (_number(nu, mp.mp.dps + 8), _number(y, mp.mp.dps + 8))
        if key not in cache:
            data = integrator.integrals(nu, y)
            _, uy, _ = _model_potential(model, y)
            cache[key] = (data["n"] - n_target, model.MN * data["ns"] + uy - model.Cv * data["n"]**2 / y**3)
        return cache[key]
    try:
        root = mp.findroot((lambda nu, y: equations(nu, y)[0], lambda nu, y: equations(nu, y)[1]), (nu0, anchor_y), solver="mdnewton", tol=mp.eps ** mp.mpf("0.65"), maxsteps=50, verify=False)
    except Exception as exc:
        raise ThermalBridgeError(f"low-T fixed-density root failed: {type(exc).__name__}: {exc}") from exc
    nu, y = _mp(root[0]), _mp(root[1])
    data = integrator.integrals(nu, y)
    u, uy, uyy = _model_potential(model, y)
    vector = model.Cv * data["n"]**2 / (2 * y**2)
    pressure = data["pressure"] - u + vector
    residual = max(abs(data["n"] - n_target), abs(model.MN * data["ns"] + uy - model.Cv * data["n"]**2 / y**3)) / max(model.MN * data["ns"], abs(uy), mp.mpf(1))
    return {
        "T_MeV": temperature,
        "nu_MeV": nu,
        "y": y,
        "n_fm3": data["n"] / model.hbarc**3,
        "pressure_MeV_fm3": pressure / model.hbarc**3,
        "entropy_fm3": data["entropy"] / model.hbarc**3,
        "cold_anchor_y": anchor_y,
        "cold_anchor_pressure_MeV_fm3": cold["pressure_total"] / model.hbarc**3,
        "y_difference": y - anchor_y,
        "root_residual_relative": residual,
        "quadrature": {"nquad": nquad, "cutoff_MeV": _mp(cutoff), "focused_at_pF_MeV": kf},
        "pass": bool(residual < mp.mpf("1e-20") and abs(data["n"] / model.hbarc**3 - _mp(N0_FM3)) < mp.mpf("1e-20")),
    }


def _ideal_controls(temperature: mp.mpf) -> dict[str, Any]:
    # Fixed-field ideal gas: no self-consistency or potential enters this toy.
    integrator = _integrator_for(temperature, cutoff=PRIMARY_CUTOFF_MEV, nquad=PRIMARY_NQUAD)
    y = mp.mpf("0.90")
    nu = mp.mpf("850")
    def number(a: Any) -> mp.mpf:
        return integrator.integrals(_mp(a), y)["n"]
    ideal = tuple(_mp(mp.diff(number, nu, order)) for order in (1, 2, 3))
    h_values = (_mp("2"), _mp("1"))
    fd_rows = []
    for h in h_values:
        nm2, nm1, np1, np2 = (number(nu + q * h) for q in (-2, -1, 1, 2))
        fd = ((np1 - nm1) / (2 * h), (np1 - 2 * number(nu) + nm1) / h**2, (-nm2 + 2 * nm1 - 2 * np1 + np2) / (2 * h**3))
        errs = tuple(_relative(a, b, floor="1e-30") for a, b in zip(fd, ideal))
        fd_rows.append({"h_MeV": h, "chi1_fd": fd[0], "chi2_fd": fd[1], "chi3_fd": fd[2], "relative_errors": errs, "pass": bool(max(errs) < mp.mpf("2e-3"))})
    return {"fixed_field": {"T_MeV": temperature, "nu_MeV": nu, "y": y, "chi1": ideal[0], "chi2": ideal[1], "chi3": ideal[2], "finite_difference": fd_rows}, "all_pass": all(row["pass"] for row in fd_rows)}


def _classical_controls(temperature: mp.mpf) -> dict[str, Any]:
    # Separate analytic controls make the particle/antiparticle signs visible.
    A = mp.mpf("1.7")
    nu = mp.mpf("-3") * temperature
    particle = A * mp.exp(nu / temperature)
    p_derivatives = (particle / temperature, particle / temperature**2, particle / temperature**3)
    particle_ratios = (temperature * p_derivatives[0] / particle, temperature * p_derivatives[1] / p_derivatives[0], temperature**2 * p_derivatives[2] / p_derivatives[0])
    ep, ea = mp.exp(nu / temperature), mp.exp(-nu / temperature)
    net = A * (ep - ea)
    net_d = (A * (ep + ea) / temperature, A * (ep - ea) / temperature**2, A * (ep + ea) / temperature**3)
    sign_identity = (net_d[0] * temperature - A * (ep + ea), net_d[1] * temperature**2 - net, net_d[2] * temperature**3 - A * (ep + ea))
    return {"particle_only": {"R21": particle_ratios[0], "R32": particle_ratios[1], "R42": particle_ratios[2], "all_one_relative": max(abs(value - 1) for value in particle_ratios)}, "particle_antiparticle_sign": {"net_density": net, "derivative_sign_identity_max": max(abs(value) for value in sign_identity)}, "pass": bool(max(abs(value - 1) for value in particle_ratios) < mp.mpf("1e-30") and max(abs(value) for value in sign_identity) < mp.mpf("1e-30"))}


def _deformation_effects(anchor_payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target, payload in anchor_payload.items():
        for key in ("775", "875"):
            for temperature in TEMPERATURES:
                row = payload[temperature][key]
                w = row["models"]["w8"]["state"]
                u = row["models"]["u16"]["state"]
                result[f"anchor{target}_T{temperature}_mu{key}"] = _classify_deformation_effect(
                    w,
                    u,
                    row["models"]["w8"]["quadrature_controls"],
                    row["models"]["u16"]["quadrature_controls"],
                )
    return result


def _classify_deformation_effect(
    w8_state: Mapping[str, Any],
    u16_state: Mapping[str, Any],
    w8_quadrature: Mapping[str, Any],
    u16_quadrature: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare W8/U16 using conservative absolute errors from both runs."""
    fields = ("n_fm3", "pressure_MeV_fm3", "entropy_fm3", "R21", "R32", "R42")
    deltas = {field: _mp(u16_state[field]) - _mp(w8_state[field]) for field in fields}
    w8_relative = w8_quadrature["relative_differences"]
    u16_relative = u16_quadrature["relative_differences"]
    w8_absolute = {field: abs(_mp(w8_state[field])) * _mp(w8_relative[field]) for field in fields}
    u16_absolute = {field: abs(_mp(u16_state[field])) * _mp(u16_relative[field]) for field in fields}
    combined_absolute = {field: w8_absolute[field] + u16_absolute[field] for field in fields}
    effect_relative = {field: abs(deltas[field]) / max(abs(_mp(w8_state[field])), abs(_mp(u16_state[field])), mp.mpf("1e-45")) for field in fields}
    resolved_fields = [field for field in fields if abs(deltas[field]) > mp.mpf("100") * combined_absolute[field]]
    return {
        "u16_minus_w8": deltas,
        "effect_relative": effect_relative,
        "w8_resolution_absolute": w8_absolute,
        "u16_resolution_absolute": u16_absolute,
        "combined_resolution_absolute": combined_absolute,
        "resolved_fields": resolved_fields,
        "classification": "resolved_live_difference" if resolved_fields else "numerically_unresolved",
    }


def _compute(dps: int, *, nquad: int = PRIMARY_NQUAD, cutoff: Any = PRIMARY_CUTOFF_MEV) -> dict[str, Any]:
    if dps < 28:
        raise ThermalBridgeError("thermal bridge requires at least 28 decimal digits")
    with mp.workdps(dps):
        anchors: dict[str, Any] = {}
        for target in ANCHORS:
            infos = nonlinear._build_models(target)
            anchor_payload: dict[str, Any] = {}
            for temperature_text in TEMPERATURES:
                temperature = _mp(temperature_text)
                anchor_payload[temperature_text] = {}
                for mu_text in CHEMICAL_POTENTIALS:
                    mu = _mp(mu_text)
                    combo: dict[str, Any] = {"T_MeV": temperature, "mu_MeV": mu, "models": {}}
                    for name in MODEL_ORDER:
                        info = infos[name]
                        model = info["model"]
                        integrator = _integrator_for(temperature, cutoff=cutoff, nquad=nquad)
                        solved = _solve_selected(integrator, model, mu, _mp(target))
                        selected = solved["selected"]
                        state = solved["state"]
                        fd = _finite_difference_controls(model, solved, temperature, mu, cutoff=_mp(cutoff), nquad=nquad)
                        station = _stationary_derivative_controls(model, solved, temperature, mu, cutoff=_mp(cutoff), nquad=nquad)
                        quad = _quadrature_controls(model, selected, temperature, mu, _mp(target), cutoff=_mp(CONTROL_CUTOFF_MEV), nquad=CONTROL_NQUAD, reference=state)
                        combo["models"][name] = {
                            **_model_info(info),
                            "root_discovery": {"attempted_starts": solved["attempts"], "distinct_roots": solved["roots"], "distinct_root_count": solved["distinct_root_count"], "stable_root_count": solved["stable_root_count"], "bounded_scan_window": {"nu_offset_MeV": [min(ROOT_GRID_NU_OFFSETS), max(ROOT_GRID_NU_OFFSETS)], "y": [ROOT_Y_MIN, ROOT_Y_MAX], "not_exhaustive": True}, "bracket_strategy": "bounded_2D_start_scan_with_explicit_residual_check; no scalar sign bracket asserted", "selected_lineage": BRANCH_LINEAGE, "global_phase_claim": False},
                            "state": state,
                            "finite_difference_controls": fd,
                            "stationary_pressure_controls": station,
                            "quadrature_controls": quad,
                        }
                    anchor_payload[temperature_text][mu_text] = combo
            anchors[target] = anchor_payload
        controls = {"ideal_fixed_field": _ideal_controls(_mp("70")), "classical_particle_and_antiparticle": _classical_controls(_mp("70"))}
        lowT: dict[str, Any] = {}
        for target in ANCHORS:
            model = nonlinear._build_models(target)["w8"]["model"]
            rows = []
            for t in (_mp("32"), _mp("16"), _mp("8")):
                rows.append(_solve_fixed_density_lowT(model, _mp(target), t))
            errors = [abs(row["y_difference"]) for row in rows]
            lowT[target] = {"model": "w8", "rows": rows, "monotone_y_error": bool(all(errors[index + 1] < errors[index] for index in range(len(errors) - 1))), "all_rows_pass": bool(all(row["pass"] for row in rows))}
        return {"anchors": anchors, "controls": controls, "low_temperature_fixed_density": lowT, "deformation_effects": _deformation_effects(anchors)}


def _precision_controls(primary: Mapping[str, Any], control: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for target in ANCHORS:
        for temperature in TEMPERATURES:
            for mu in CHEMICAL_POTENTIALS:
                for name in MODEL_ORDER:
                    p = primary["anchors"][target][temperature][mu]["models"][name]
                    c = control["anchors"][target][temperature][mu]["models"][name]
                    fields = ("n_fm3", "y", "pressure_MeV_fm3", "entropy_fm3", "R21", "R32", "R42", "chi1_fm3_per_MeV", "chi2_fm3_per_MeV2", "chi3_fm3_per_MeV3")
                    diffs = {field: _relative(p["state"][field], c["state"][field], floor="1e-25") for field in fields}
                    maximum = max(diffs.values())
                    checks.append({"anchor": target, "T_MeV": temperature, "mu_MeV": mu, "model": name, "primary_dps": PRIMARY_DPS, "control_dps": CONTROL_DPS, "max_relative_difference": maximum, "fields": diffs, "pass": bool(maximum <= PRECISION_REL_LIMIT)})
    return checks


def _all_controls_pass(tree: Mapping[str, Any]) -> bool:
    for anchor in tree["anchors"].values():
        for temp in anchor.values():
            for combo in temp.values():
                for model in combo["models"].values():
                    state = model["state"]
                    if not (state["local_stable_C_positive"] and state["local_stable_fnn_positive"]):
                        return False
                    if _mp(state["gap_residual_relative"]) > ROOT_REL_LIMIT or _mp(state["gibbs_identity_relative"]) > IDENTITY_REL_LIMIT or _mp(state["pressure_kinetic_log_relative"]) > IDENTITY_REL_LIMIT or _mp(state["chi1_vs_schur_relative"]) > SCHUR_IDENTITY_LIMIT:
                        return False
                    if not model["finite_difference_controls"]["all_pass"] or not model["stationary_pressure_controls"]["pass"] or not model["quadrature_controls"]["pass"]:
                        return False
    low_temperature = tree["low_temperature_fixed_density"]
    if not all(item.get("all_rows_pass", False) and all(row.get("pass", False) for row in item.get("rows", ())) and item.get("monotone_y_error", False) for item in low_temperature.values()):
        return False
    return bool(tree["controls"]["ideal_fixed_field"]["all_pass"] and tree["controls"]["classical_particle_and_antiparticle"]["pass"])


def _public_result(primary: Mapping[str, Any], control: Mapping[str, Any], precision: Sequence[Mapping[str, Any]], *, status: str) -> dict[str, Any]:
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    upstream_hash = hashlib.sha256(nonlinear.UPSTREAM_PATH.read_bytes()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "evidence_weight": EVIDENCE_WEIGHT,
        "protocol_version": "thermal-same-action-fixed-grid-v1",
        "protocol_provenance": "ADOPTED_P1_GRID_FIXED_AFTER_P0_PILOT; NUMERICAL_REFINEMENTS_NOT_FULLY_PREREGISTERED; CONTEXT_GRID_NOT_FREEZEOUT_FIT",
        "source_sha256": source_hash,
        "upstream_source_sha256": upstream_hash,
        "physical_inputs_fixed": True,
        "protocol_frozen_before_rows": False,
        "inputs": {"anchors": ANCHORS, "temperature_MeV": TEMPERATURES, "chemical_potential_MeV": CHEMICAL_POTENTIALS, "deformation_amplitudes": MODEL_AMPLITUDES, "degeneracy": DEGENERACY, "n0_fm3": N0_FM3, "binding_MeV": BINDING_MEV, "K_target_MeV": K_TARGET_MEV, "primary_dps": PRIMARY_DPS, "control_dps": CONTROL_DPS, "primary_nquad": PRIMARY_NQUAD, "control_nquad": CONTROL_NQUAD, "primary_cutoff_MeV": PRIMARY_CUTOFF_MEV, "control_cutoff_MeV": CONTROL_CUTOFF_MEV},
        "units_and_conventions": {"natural_units": "hbar=c=1; n,s in MeV^3; P,e in MeV^4", "density": DENSITY_UNITS, "pressure": PRESSURE_UNITS, "entropy": ENTROPY_UNITS, "susceptibilities": SUSCEPTIBILITY_UNITS, "hbarc_MeV_fm": HBARC_MEV_FM, "ensemble": "homogeneous symmetric grand-canonical medium, particle+antiparticle FD, vacuum term subtracted", "ratios": "R21=T*chi1/n; R32=T*chi2/chi1; R42=T^2*chi3/chi1; equilibrium total baryons, not accepted protons"},
        "anchors": primary["anchors"],
        "controls": primary["controls"],
        "low_temperature_fixed_density": primary["low_temperature_fixed_density"],
        "deformation_effects": primary["deformation_effects"],
        "precision_controls": list(precision),
        "precision_control_summary": {"all_pass": bool(all(item["pass"] for item in precision)), "scale_rule": "actual-value scaled; no max(1, observable) tolerance", "primary_nquad": PRIMARY_NQUAD, "control_nquad": CONTROL_NQUAD},
        "control_tree_summary": {"primary_all_controls_pass": _all_controls_pass(primary), "control_all_controls_pass": _all_controls_pass(control), "schur_identity_limit": SCHUR_IDENTITY_LIMIT, "low_temperature_requires_all_rows_and_each_step": True},
        "interpretation_limits": ["bounded local homogeneous branch; root scan is not a theorem of global phase equilibrium", "symmetric d=4 total baryons, not HADES accepted protons or a net-proton measurement", "T and mu grid points are contextual robustness probes, not fitted freezeout coordinates", "no chi-square, significance, refit, or experimental confirmation is claimed", "published R21/R32/R42 values in arXiv:2204.00137 are model-extracted rapidity-cut spreads, not experimental covariance"],
    }


def build_result() -> dict[str, Any]:
    global _LAST_RESULT_JSON
    primary = _compute(PRIMARY_DPS, nquad=PRIMARY_NQUAD, cutoff=PRIMARY_CUTOFF_MEV)
    control = _compute(CONTROL_DPS, nquad=CONTROL_NQUAD, cutoff=CONTROL_CUTOFF_MEV)
    precision = _precision_controls(primary, control)
    status = STATUS_PASS if _all_controls_pass(primary) and _all_controls_pass(control) and all(item["pass"] for item in precision) else STATUS_FAIL
    result = _jsonable(_public_result(primary, control, precision, status=status))
    _LAST_RESULT_JSON = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return result


_LAST_RESULT_JSON: str | None = None


def _expected_result_json() -> str:
    """Return the canonical tree from the latest live build.

    ``build_result`` stores this immutable JSON string after a complete live
    computation.  Validation therefore does not launch a second five-minute
    quadrature pass in the CLI/tests, while a caller that validates before any
    build still receives a fresh live rebuild.
    """
    global _LAST_RESULT_JSON
    if _LAST_RESULT_JSON is None:
        _LAST_RESULT_JSON = json.dumps(build_result(), sort_keys=True, ensure_ascii=False, allow_nan=False)
    return _LAST_RESULT_JSON


def validate_result(result: Any) -> bool:
    """Fail closed against the latest complete live build; JSON is not input."""
    try:
        if not isinstance(result, dict):
            return False
        if result.get("status") != STATUS_PASS:
            return False
        expected = json.loads(_expected_result_json())
        actual = json.loads(json.dumps(result, allow_nan=False))
        return actual == expected
    except (ArithmeticError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError, json.JSONDecodeError):
        return False


def _write_json(payload: Mapping[str, Any], path: Path | None) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path is not None:
        path.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional explicit JSON output path")
    args = parser.parse_args(argv)
    try:
        result = build_result()
        if result.get("status") != STATUS_PASS:
            _write_json(result, args.output)
            return 1
        if not validate_result(result):
            raise ThermalBridgeError("fresh result failed live validation")
        _write_json(result, args.output)
        return 0
    except Exception as exc:
        failure = {"schema_version": SCHEMA_VERSION, "status": STATUS_FAIL, "error": f"{type(exc).__name__}: {exc}"}
        _write_json(failure, args.output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
