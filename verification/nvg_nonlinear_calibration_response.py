#!/usr/bin/env python3
"""Live nonlinear inverse-chemical-potential response for the W8/U16 pair.

The producer rebuilds both predeclared calibration anchors from the accepted
``BulkModel`` implementation in the current process.  It never reads a result
table as an input.  Equilibrium values are obtained from the stationary scalar
root, while inverse chemical-potential rows solve back through that same branch.
The W8/U16 comparison is therefore a local nonlinear discriminator, not a
claim about a global phase, a dynamic resonance, or an experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import source_complete_scaling_saturation_audit as upstream  # noqa: E402
from nvg_density_universality_audit import PotentialAdapter  # noqa: E402

SCHEMA_VERSION = 1
STATUS_PASS = "PASS_LIVE_NONLINEAR_INVERSE_RESPONSE"
STATUS_FAIL = "FAIL_NONLINEAR_INVERSE_RESPONSE"
EVIDENCE_WEIGHT = 0.0
ANCHORS = ("0.90", "0.93")
MODEL_ORDER = ("w8", "u16")
AMPLITUDES = {"w8": 0, "u16": 1}
RESPONSE_U = ("-0.20", "-0.10", "0.10", "0.20")
FD_U_STEPS = ("0.005", "0.002")
PRIMARY_DPS = 90
CONTROL_DPS = 120
VALIDATION_DPS = 90
PRECISION_REL_LIMIT = mp.mpf("1e-45")
STATIONARITY_REL_LIMIT = mp.mpf("1e-70")
INVERSE_REL_LIMIT = mp.mpf("1e-60")
FD_CHI3_REL_LIMIT = mp.mpf("2e-3")
SEPARATION_FLOOR = mp.mpf("1e-30")
BRANCH_LOW = mp.mpf("0.5")
BRANCH_HIGH = mp.mpf("2.0")
BRACKET_LOW = mp.mpf("0.6")

# Validation tolerances are deliberately relative to the physical quantity,
# not to ``max(quantity, 1)``.  The latter turns a fourth derivative of order
# 1e-20 into an effectively unconstrained number and was the adversarial hole
# repaired in B1.  Exact-zero controls use a separately declared numerical
# bound; near-zero root residuals are checked by bounds below rather than by
# pretending that their last digits are physical data.
SCIENCE_REL_TOL = mp.mpf("1e-12")
ZERO_VALUE_TOL = mp.mpf("1e-45")
IDENTITY_LIMIT = mp.mpf("1e-25")
CALIBRATION_REL_LIMIT = mp.mpf("1e-60")

BRANCH_LINEAGE = "continued_from_anchor_positive_y_positive_C_positive_mu_prime"
DIMENSIONLESS_DEFINITION = {
    "x": "(n-n0)/n0",
    "u": "(mu-mu0)/(K/9)",
    "Z": "81*n0^4*d^4(epsilon/n)/dn^4 at n0",
}
INTERPRETATION = "nonlinear static branch discriminator only; no dynamic/resonance/measurement claim"
EXPECTED_UNITS = {
    "n": "natural MeV^3; n0=0.16 fm^-3 converted with hbarc",
    "epsilon": "MeV^4",
    "mu": "MeV",
    "epsilon_derivative_orders": "d^k epsilon/dn^k, no factorial absorbed",
    "chi": "chi_k=d^k n/dmu^k, no factorial absorbed",
    "x": "(n-n0)/n0",
    "u": "(mu-mu0)/(K/9)",
    "Z": "81*n0^4*d^4(epsilon/n)/dn^4 at n0, MeV",
}
EXPECTED_INTERPRETATION_LIMITS = [
    "local smooth positive-curvature homogeneous branch only",
    "not a global ground-state, phase-mixture, finite-density EOS, dynamic, resonance, or experimental claim",
    "DeltaU is a mathematical calibration-null counterexample with fixed positive coefficient",
]
CONTROL_NUMERIC_FIELDS = {
    "residual",
    "mu_residual_MeV",
    "stationarity_relative",
    "mu_relative",
    "max_inverse_root_relative",
    "epsilon1_equals_mu",
    "epsilon2_equals_mu_prime",
    "delta_chi3_identity_relative",
    "dimensionless_identity_relative",
}

UPSTREAM_PATH = HERE / "source_complete_scaling_saturation_audit.py"
PUBLIC_PROTOCOL_PATH = HERE / "data" / "nonlinear_calibration_protocol_2026.md"
# Keep this compatibility name local to the public verification tree.  It is
# intentionally not a path into the ignored run workspace.
PROTOCOL_PATH = PUBLIC_PROTOCOL_PATH
PROTOCOL_VERSION = "nonlinear-calibration-public-v1"
PROTOCOL_PROVENANCE = "PUBLIC_REPRODUCIBLE_NUMERICAL_SPECIFICATION_WITH_POST_FREEZE_FD_REFINEMENT"
PROTOCOL_MARKERS = (
    f"protocol_version: {PROTOCOL_VERSION}",
    "anchors: 0.90,0.93",
    "deformation_amplitudes: 0,1",
    "response_u: -0.20,-0.10,0.10,0.20",
    "fd_u_steps: 0.005,0.002",
    "primary_dps: 90",
    "independent_control_dps: 120",
    "stationarity_relative_limit: 1e-70",
    "inverse_relative_limit: 1e-60",
    "branch_window_ratio: 0.5,2.0",
)


def _public_protocol_bytes() -> bytes:
    """Read the required public protocol and reject missing/stale controls."""
    try:
        data = PROTOCOL_PATH.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ArithmeticError("required public protocol is missing or unreadable") from exc
    if any(marker not in text for marker in PROTOCOL_MARKERS):
        raise ArithmeticError("public protocol markers do not match live controls")
    return data


def _protocol_sha256() -> str:
    """Bind the artifact to public protocol bytes, never to an optional file."""
    return hashlib.sha256(_public_protocol_bytes()).hexdigest()


def _mp(value: Any) -> mp.mpf:
    if isinstance(value, mp.mpf):
        return value
    return mp.mpf(str(value))


def _finite(value: Any) -> bool:
    try:
        return bool(mp.isfinite(_mp(value)))
    except (TypeError, ValueError):
        return False


def _number(value: Any, digits: int = 90) -> str:
    value = _mp(value)
    if not mp.isfinite(value):
        raise ValueError("nonfinite scientific value")
    return mp.nstr(value, digits)


def _relative(a: Any, b: Any) -> mp.mpf:
    a, b = _mp(a), _mp(b)
    return abs(a - b) / max(abs(a), abs(b), mp.mpf(1))


def _relative_scale(value: Any, scale: Any) -> mp.mpf:
    value, scale = _mp(value), _mp(scale)
    return abs(value) / max(abs(scale), mp.mpf(1))


def _is_real(value: Any) -> bool:
    return not isinstance(value, mp.mpc) or abs(mp.im(value)) == 0


def _state(model: Any, n: Any, y: Any) -> dict[str, mp.mpf]:
    """Evaluate the accepted homogeneous functional with the active potential."""
    n, y = _mp(n), _mp(y)
    f = model.fermi(n, y)
    u, uy, uyy = model.potential(y)
    vector = model.Cv * n * n / (2 * y * y)
    energy = f["energy"] + u + vector
    pressure = f["pressure"] - u + vector
    mu = f["ef"] + model.Cv * n / (y * y)
    residual = model.MN * f["ns"] + uy - model.Cv * n * n / y**3
    c_y = uyy + model.MN**2 * f["ns_m"] + 3 * model.Cv * n * n / y**4
    b_y = model.MN**2 * y / f["ef"] - 2 * model.Cv * n / y**3
    d_nn = f["k"]**2 / (3 * n * f["ef"]) + model.Cv / y**2
    mu_prime = d_nn - b_y * b_y / c_y if c_y != 0 else mp.nan
    return dict(
        f,
        n=n,
        y=y,
        U=u,
        Uy=uy,
        Uyy=uyy,
        vector=vector,
        energy_total=energy,
        pressure_total=pressure,
        mu=mu,
        residual=residual,
        C_y=c_y,
        B_y=b_y,
        D_nn=d_nn,
        mu_prime=mu_prime,
        K=9 * n * mu_prime,
        cs2=n * mu_prime / mu,
    )


def _stationarity_scale(model: Any, state: dict[str, mp.mpf]) -> mp.mpf:
    return max(
        abs(model.MN * state["ns"]),
        abs(state["Uy"]),
        abs(model.Cv * state["n"] ** 2 / state["y"] ** 3),
        mp.mpf(1),
    )


def _root_tolerance() -> mp.mpf:
    # Leave guard digits for the nested derivatives used by mp.diff.
    return mp.eps ** mp.mpf("0.82")


def _stationary_root(model: Any, n: Any, guess: Any) -> tuple[mp.mpf, dict[str, mp.mpf], dict[str, Any]]:
    """Solve the local positive-curvature stationary branch from a lineage guess."""
    n, guess = _mp(n), _mp(guess)
    if n <= 0 or guess <= 0:
        raise ArithmeticError("stationary root requires positive n and y")

    def residual(y: Any) -> mp.mpf:
        return _state(model, n, y)["residual"]

    try:
        y = mp.findroot(residual, guess, solver="newton", tol=_root_tolerance(), maxsteps=80, verify=False)
    except Exception:
        # The second attempt is deliberately local; a distant root is not a
        # valid continuation of the accepted calibration branch.
        y = mp.findroot(
            residual,
            (guess * mp.mpf("0.995"), guess * mp.mpf("1.005")),
            solver="anderson",
            tol=_root_tolerance(),
            maxsteps=100,
            verify=False,
        )
    if not _finite(y) or not _finite(n) or y <= 0:
        raise ArithmeticError("nonfinite or nonpositive stationary root")
    state = _state(model, n, y)
    rel = abs(state["residual"]) / _stationarity_scale(model, state)
    if rel > STATIONARITY_REL_LIMIT:
        raise ArithmeticError("stationarity residual exceeds frozen limit")
    if not (_finite(state["C_y"]) and _finite(state["mu_prime"]) and state["C_y"] > 0 and state["mu_prime"] > 0):
        raise ArithmeticError("stationary root is not on the positive-curvature/compressibility branch")
    return y, state, {
        "stationarity_relative": rel,
        "branch_lineage": "continued_from_anchor_positive_y_positive_C_positive_mu_prime",
        "local_window": [BRANCH_LOW, BRANCH_HIGH],
    }


def _build_models(target: str) -> dict[str, dict[str, Any]]:
    """Rebuild one predeclared W8 anchor and its alpha=1 U16 companion live."""
    target_y = _mp(target)
    # These are the existing accepted inverse-potential construction inputs,
    # not values loaded from a result artifact and not a new optimization.
    w8, _, jet = upstream.inverse_potential_jet(
        target, K_target="240", n_fm3="0.16", binding="-16"
    )
    eta = _mp(jet["coefficients"][2])
    if not eta > 0:
        raise ArithmeticError("accepted U16 coefficient is not positive")
    u16 = PotentialAdapter(w8, target_y, eta)
    return {
        "w8": {
            "id": f"w8_{target[2:]}",
            "model": w8,
            "amplitude": 0,
            "design": "accepted W8 calibration anchor y*=" + target,
            "target_y": target_y,
            "eta_MeV4": eta,
        },
        "u16": {
            "id": f"u16_{target[2:]}",
            "model": u16,
            "amplitude": 1,
            "design": "W8 plus positive U16 calibration-null deformation y*=" + target,
            "target_y": target_y,
            "eta_MeV4": eta,
        },
    }


def _live_energy(model: Any, n: Any, anchor_y: Any) -> mp.mpf:
    """Energy on the stationary branch; used by all derivative calculations."""
    _, state, _ = _stationary_root(model, n, anchor_y)
    return state["energy_total"]


def _live_state(model: Any, n: Any, anchor_y: Any) -> dict[str, mp.mpf]:
    return _stationary_root(model, n, anchor_y)[1]


def _derivatives(model: Any, anchor_y: Any) -> dict[str, Any]:
    n0 = model.n0
    y0 = _mp(anchor_y)
    anchor_state = _live_state(model, n0, y0)
    # mp.diff differentiates the live root calculation.  This is intentionally
    # separate from the inverse-mu roots below; no serialized row is reused.
    eps = [mp.diff(lambda n: _live_energy(model, n, y0), n0, order, addprec=30) for order in range(1, 5)]
    eps1, eps2, eps3, eps4 = eps
    mu0 = anchor_state["mu"]
    chi1 = 1 / eps2
    chi2 = -eps3 / eps2**3
    chi3 = 3 * eps3**2 / eps2**5 - eps4 / eps2**4
    z_fourth = mp.diff(lambda n: _live_energy(model, n, y0) / n, n0, 4, addprec=30)
    z = 81 * n0**4 * z_fourth
    return {
        "anchor_state": anchor_state,
        "epsilon_derivatives": {
            "epsilon_1": eps1,
            "epsilon_2": eps2,
            "epsilon_3": eps3,
            "epsilon_4": eps4,
        },
        "mu0": mu0,
        "chi": {"chi1": chi1, "chi2": chi2, "chi3": chi3},
        "Z_MeV": z,
        "energy_per_particle_fourth_derivative": z_fourth,
        "checks": {
            "epsilon1_equals_mu": _relative(eps1, mu0),
            "epsilon2_equals_mu_prime": _relative(eps2, anchor_state["mu_prime"]),
            "stationarity_relative": abs(anchor_state["residual"]) / _stationarity_scale(model, anchor_state),
            "positive_curvature": bool(anchor_state["C_y"] > 0),
            "positive_compressibility": bool(eps2 > 0),
            "finite": all(_finite(v) for v in eps + [mu0, chi1, chi2, chi3, z, z_fourth]),
        },
        "method": "live_stationary_energy_mp.diff_orders_1_to_4",
    }


def _mu_at_n(model: Any, n: Any, anchor_y: Any) -> tuple[mp.mpf, dict[str, Any]]:
    y, state, controls = _stationary_root(model, n, anchor_y)
    return state["mu"], {"y": y, "state": state, **controls}


def _inverse_mu(model: Any, anchor_y: Any, mu_target: Any, mu0: Any, eps2: Any) -> dict[str, Any]:
    """Invert mu(n) on the positive branch with safeguarded continuation."""
    mu_target = _mp(mu_target)
    n0 = model.n0
    # Linearized guess only chooses the local starting point; the returned root
    # is bracketed and checked against a fresh stationary solve.
    guess = n0 * mp.exp((mu_target - mu0) / (eps2 * n0))
    guess = min(max(guess, n0 * BRANCH_LOW), n0 * BRANCH_HIGH)

    def f_n(n: Any) -> tuple[mp.mpf, dict[str, Any]]:
        mu, details = _mu_at_n(model, n, anchor_y)
        return mu - mu_target, details

    low, high = n0 * BRACKET_LOW, n0 * BRANCH_HIGH
    flo, _ = f_n(low)
    fhi, _ = f_n(high)
    if not (flo <= 0 <= fhi):
        raise ArithmeticError("inverse-mu target is outside the local monotone bracket")
    n = guess
    details = None
    for _ in range(160):
        fn, details = f_n(n)
        rel = abs(fn) / max(abs(mu_target), mp.mpf(1))
        if rel <= INVERSE_REL_LIMIT:
            break
        if fn < 0:
            low, flo = n, fn
        else:
            high, fhi = n, fn
        # Newton is fast on the smooth branch; bisection is the fail-safe.
        derivative = details["state"]["mu_prime"]
        candidate = n - fn / derivative if derivative > 0 else (low + high) / 2
        if not (_finite(candidate) and low < candidate < high):
            candidate = (low + high) / 2
        n = candidate
    else:
        raise ArithmeticError("inverse-mu root did not reach frozen residual")
    if details is None:
        raise ArithmeticError("inverse-mu root has no stationary details")
    fn, details = f_n(n)
    rel = abs(fn) / max(abs(mu_target), mp.mpf(1))
    if rel > INVERSE_REL_LIMIT:
        raise ArithmeticError("inverse-mu residual exceeds frozen limit")
    state = details["state"]
    if not (BRANCH_LOW * n0 <= n <= BRANCH_HIGH * n0 and state["C_y"] > 0 and state["mu_prime"] > 0):
        raise ArithmeticError("inverse-mu root left the local stable branch")
    return {
        "n": n,
        "y": details["y"],
        "state": state,
        "mu_target": mu_target,
        "mu_residual": fn,
        "mu_relative": rel,
        "stationarity_relative": details["stationarity_relative"],
        "branch_lineage": details["branch_lineage"],
        "window_accept": True,
    }


def _inverse_response(model: Any, anchor_y: Any, derivatives: dict[str, Any]) -> list[dict[str, Any]]:
    n0 = model.n0
    mu0 = derivatives["mu0"]
    eps2 = derivatives["epsilon_derivatives"]["epsilon_2"]
    K = 9 * n0 * eps2
    rows = []
    for u_text in RESPONSE_U:
        u = _mp(u_text)
        mu_target = mu0 + u * K / 9
        root = _inverse_mu(model, anchor_y, mu_target, mu0, eps2)
        x = (root["n"] - n0) / n0
        rows.append({
            "u": u,
            "x": x,
            "n_MeV3": root["n"],
            "y": root["y"],
            "mu_target_MeV": mu_target,
            "mu_solved_MeV": root["state"]["mu"],
            "mu_residual_MeV": root["mu_residual"],
            "mu_relative": root["mu_relative"],
            "stationarity_relative": root["stationarity_relative"],
            "C_y": root["state"]["C_y"],
            "mu_prime": root["state"]["mu_prime"],
            "K_local_MeV": 9 * root["n"] * root["state"]["mu_prime"],
            "branch_lineage": root["branch_lineage"],
            "window_accept": root["window_accept"],
        })
    return rows


def _inverse_fd_controls(model: Any, anchor_y: Any, derivatives: dict[str, Any]) -> list[dict[str, Any]]:
    """Independent finite-step inverse roots; truncation is visible in the trend."""
    n0 = model.n0
    mu0 = derivatives["mu0"]
    eps = derivatives["epsilon_derivatives"]
    chi = derivatives["chi"]
    K = 9 * n0 * eps["epsilon_2"]
    result = []
    for h_text in FD_U_STEPS:
        h = _mp(h_text)
        dmu = K / 9 * h
        roots = {}
        for sign in (-2, -1, 1, 2):
            root = _inverse_mu(model, anchor_y, mu0 + sign * dmu, mu0, eps["epsilon_2"])
            roots[sign] = root
        nm2, nm1, np1, np2 = (roots[index]["n"] for index in (-2, -1, 1, 2))
        chi1_fd = (np1 - nm1) / (2 * dmu)
        chi2_fd = (np1 - 2 * n0 + nm1) / dmu**2
        chi3_fd = (-nm2 + 2 * nm1 - 2 * np1 + np2) / (2 * dmu**3)
        errors = {
            "chi1": _relative(chi1_fd, chi["chi1"]),
            "chi2": _relative(chi2_fd, chi["chi2"]),
            "chi3": _relative(chi3_fd, chi["chi3"]),
        }
        max_root_rel = max(root["mu_relative"] for root in roots.values())
        result.append({
            "u_step": h,
            "dmu_step_MeV": dmu,
            "chi1_fd": chi1_fd,
            "chi2_fd": chi2_fd,
            "chi3_fd": chi3_fd,
            "chi1_relative_error": errors["chi1"],
            "chi2_relative_error": errors["chi2"],
            "chi3_relative_error": errors["chi3"],
            "max_inverse_root_relative": max_root_rel,
            "truncation_control": "decreasing_symmetric_five_point_step",
            "pass": bool(
                max_root_rel <= INVERSE_REL_LIMIT
                and errors["chi1"] < mp.mpf("2e-3")
                and errors["chi2"] < mp.mpf("2e-3")
                and errors["chi3"] <= FD_CHI3_REL_LIMIT
            ),
        })
    if not (result[1]["chi3_relative_error"] < result[0]["chi3_relative_error"]):
        raise ArithmeticError("decreasing inverse step did not reduce chi3 truncation")
    return result


def _deformation_summary(model_info: dict[str, Any]) -> dict[str, Any]:
    model = model_info["model"]
    target = model_info["target_y"]
    eta = model_info["eta_MeV4"]
    if model_info["amplitude"] == 0:
        return {
            "amplitude": 0,
            "eta_MeV4": eta,
            "delta_U_nonnegative_by_even_power_proof": True,
            "exact_zero_through_order_3_at_vacuum_and_anchor": True,
            "delta_U_fourth_derivative_at_anchor_MeV4": mp.mpf(0),
        }
    # The adapter's closed-form derivatives are used only as a zero/positivity
    # certificate; the response derivatives are obtained from live rooted energy.
    vacuum = model.deformation_derivatives(mp.mpf(1))
    anchor = model.deformation_derivatives(target)
    samples = [model.potential(point)[0] - model.base.potential(point)[0] for point in (mp.mpf("0.85"), mp.mpf("1.05"))]
    delta_u4 = mp.diff(lambda y: eta * (y * y - 1) ** 4 * (y * y - target * target) ** 4, target, 4)
    return {
        "amplitude": 1,
        "eta_MeV4": eta,
        "delta_U_nonnegative_by_even_power_proof": bool(eta > 0),
        "exact_zero_through_order_3_at_vacuum_and_anchor": all(v == 0 for v in vacuum + anchor),
        "vacuum_delta_U_derivatives_0_to_3": vacuum,
        "anchor_delta_U_derivatives_0_to_3": anchor,
        "delta_U_fourth_derivative_at_anchor_MeV4": delta_u4,
        "positive_sample_delta_U_MeV4": samples,
        "sample_points_y": (mp.mpf("0.85"), mp.mpf("1.05")),
    }


def _compute(dps: int) -> dict[str, Any]:
    with mp.workdps(dps):
        anchors: dict[str, Any] = {}
        for target in ANCHORS:
            target_y = _mp(target)
            infos = _build_models(target)
            models: dict[str, Any] = {}
            for name in MODEL_ORDER:
                info = infos[name]
                deriv = _derivatives(info["model"], target_y)
                responses = _inverse_response(info["model"], target_y, deriv)
                fd_controls = _inverse_fd_controls(info["model"], target_y, deriv)
                models[name] = {
                    **info,
                    "derivatives": deriv,
                    "responses": responses,
                    "inverse_fd_controls": fd_controls,
                    "deformation": _deformation_summary(info),
                }
            w8, u16 = models["w8"], models["u16"]
            wd, ud = w8["derivatives"], u16["derivatives"]
            weps, ueps = wd["epsilon_derivatives"], ud["epsilon_derivatives"]
            wchi, uchi = wd["chi"], ud["chi"]
            delta_eps4 = ueps["epsilon_4"] - weps["epsilon_4"]
            delta_chi3 = uchi["chi3"] - wchi["chi3"]
            delta_z = u16["derivatives"]["Z_MeV"] - w8["derivatives"]["Z_MeV"]
            n0 = infos["w8"]["model"].n0
            K = 9 * n0 * weps["epsilon_2"]
            delta_d3 = K**3 / (9**3 * n0) * delta_chi3
            delta_from_z = -delta_z / (9 * K)
            response_sep = []
            for wr, ur in zip(w8["responses"], u16["responses"]):
                response_sep.append({
                    "u": wr["u"],
                    "delta_x_u16_minus_w8": ur["x"] - wr["x"],
                    "w8_x": wr["x"],
                    "u16_x": ur["x"],
                    "same_target_mu": _relative(wr["mu_target_MeV"], ur["mu_target_MeV"]),
                })
            lower_fields = (
                "epsilon_1", "epsilon_2", "epsilon_3", "mu0"
            )
            lower = {}
            for field in lower_fields:
                if field == "mu0":
                    a, b = wd["mu0"], ud["mu0"]
                else:
                    a, b = wd["epsilon_derivatives"].get(field, wd.get(field)), ud["epsilon_derivatives"].get(field, ud.get(field))
                lower[field] = _relative(a, b)
            anchors[target] = {
                "anchor_inputs": {
                    "target_y": target_y,
                    "n0_fm3": mp.mpf("0.16"),
                    "binding_MeV": mp.mpf("-16"),
                    "K_target_MeV": mp.mpf("240"),
                    "ensemble": "cold homogeneous fixed composition",
                },
                "branch": {
                    "lineage": "positive-y stationary continuation through common n0 anchor",
                    "density_window_ratio": [BRANCH_LOW, BRANCH_HIGH],
                    "global_ground_state_claim": False,
                    "phase_mixture_claim": False,
                },
                "models": models,
                "comparison": {
                    "common_lower_jet_relative_differences": lower,
                    "delta_epsilon_4": delta_eps4,
                    "delta_chi3": delta_chi3,
                    "delta_Z_MeV": delta_z,
                    "delta_chi3_from_common_epsilon2": -delta_eps4 / weps["epsilon_2"]**4,
                    "delta_chi3_identity_relative": _relative(delta_chi3, -delta_eps4 / weps["epsilon_2"]**4),
                    "dimensionless_definition": {
                        "x": "(n-n0)/n0",
                        "u": "(mu-mu0)/(K/9)",
                        "Z": "81*n0^4*d^4(epsilon/n)/dn^4 at n0",
                    },
                    "delta_d3x_du3": delta_d3,
                    "delta_d3x_du3_from_delta_Z": delta_from_z,
                    "dimensionless_identity_relative": _relative(delta_d3, delta_from_z),
                    "response_separation": response_sep,
                    "max_abs_response_delta_x": max(abs(row["delta_x_u16_minus_w8"]) for row in response_sep),
                    "numerical_floor": SEPARATION_FLOOR,
                    "nonzero_separation_above_floor": bool(
                        abs(delta_chi3) > SEPARATION_FLOOR
                        and max(abs(row["delta_x_u16_minus_w8"]) for row in response_sep) > SEPARATION_FLOOR
                    ),
                    "interpretation": "nonlinear static branch discriminator only; no dynamic/resonance/measurement claim",
                },
            }
        return {"anchors": anchors}


def _publicize(value: Any) -> Any:
    """Remove live model objects before the snapshot crosses the JSON boundary."""
    if isinstance(value, dict):
        # ``model`` is a real live object in the per-model construction record,
        # but it is also a required string label in precision evidence.  Only
        # drop the former; silently deleting the latter would make the control
        # evidence unverifiable.
        return {
            str(key): _publicize(item)
            for key, item in value.items()
            if not (key == "model" and not isinstance(item, str))
        }
    if isinstance(value, (tuple, list)):
        return [_publicize(item) for item in value]
    if isinstance(value, mp.mpf):
        return _number(value)
    if isinstance(value, mp.mpc):
        raise ValueError("complex scientific value cannot be serialized")
    return value


def _jsonify(value: Any) -> Any:
    if isinstance(value, mp.mpf):
        return _number(value)
    if isinstance(value, mp.mpc):
        raise ValueError("complex scientific value cannot be serialized")
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonify(item) for item in value]
    return value


def _precision_controls(primary: dict[str, Any], control: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for target in ANCHORS:
        for name in MODEL_ORDER:
            p = primary["anchors"][target]["models"][name]
            c = control["anchors"][target]["models"][name]
            fields = {
                "epsilon_1": (p["derivatives"]["epsilon_derivatives"]["epsilon_1"], c["derivatives"]["epsilon_derivatives"]["epsilon_1"]),
                "epsilon_2": (p["derivatives"]["epsilon_derivatives"]["epsilon_2"], c["derivatives"]["epsilon_derivatives"]["epsilon_2"]),
                "epsilon_3": (p["derivatives"]["epsilon_derivatives"]["epsilon_3"], c["derivatives"]["epsilon_derivatives"]["epsilon_3"]),
                "epsilon_4": (p["derivatives"]["epsilon_derivatives"]["epsilon_4"], c["derivatives"]["epsilon_derivatives"]["epsilon_4"]),
                "chi3": (p["derivatives"]["chi"]["chi3"], c["derivatives"]["chi"]["chi3"]),
                "Z_MeV": (p["derivatives"]["Z_MeV"], c["derivatives"]["Z_MeV"]),
            }
            for index, (pr, cr) in enumerate(zip(p["responses"], c["responses"])):
                fields[f"response_x_{index}"] = (pr["x"], cr["x"])
            relative = {field: _relative(a, b) for field, (a, b) in fields.items()}
            maximum = max(relative.values())
            checks.append({
                "anchor": target,
                "model": name,
                "primary_dps": PRIMARY_DPS,
                "control_dps": CONTROL_DPS,
                "max_relative_difference": maximum,
                "fields": relative,
                "pass": bool(maximum <= PRECISION_REL_LIMIT),
            })
    return checks


def _toy_control() -> dict[str, Any]:
    """Independent one-dimensional analytic potential/root control."""
    n0 = mp.mpf(1)
    source_scale = mp.mpf("0.25")

    def potential(y):
        q = y - 1
        return q * q / 2 + q**4 / 24, q + q**3 / 6

    def state(n, y):
        u, uy = potential(y)
        source = source_scale * (n - 1)
        energy = (n - 1) ** 2 / 2 + mp.mpf("0.3") * n + u - source * y
        return {"energy": energy, "residual": uy - source, "curvature": 1 + (y - 1) ** 2 / 2}

    def root(n):
        y = mp.findroot(lambda q: state(n, q)["residual"], 1, tol=_root_tolerance(), verify=False)
        s = state(n, y)
        if y <= 0 or s["curvature"] <= 0:
            raise ArithmeticError("toy branch invalid")
        return y, s

    def energy(n):
        return root(n)[1]["energy"]

    e1 = mp.diff(energy, n0, 1, addprec=20)
    e2 = mp.diff(energy, n0, 2, addprec=20)
    # The envelope identity gives mu directly and is checked independently.
    y0, _ = root(n0)
    mu_envelope = (n0 - 1) + mp.mpf("0.3") - source_scale * y0
    target = e1 + mp.mpf("0.125")
    n_inverse = mp.findroot(lambda n: mp.diff(energy, n, 1, addprec=20) - target, 1.1, verify=False)
    return {
        "potential": "U=(y-1)^2/2+(y-1)^4/24; residual=U_y-0.25(n-1)",
        "root_y_at_n0": y0,
        "epsilon1": e1,
        "epsilon2": e2,
        "epsilon1_mu_envelope_relative": _relative(e1, mu_envelope),
        "inverse_target_mu": target,
        "inverse_n": n_inverse,
        "inverse_roundtrip_relative": _relative(mp.diff(energy, n_inverse, 1, addprec=20), target),
        "pass": bool(_relative(e1, mu_envelope) <= mp.mpf("1e-40") and _relative(mp.diff(energy, n_inverse, 1, addprec=20), target) <= mp.mpf("1e-35")),
    }


def _canonical_payload(result: dict[str, Any]) -> bytes:
    payload = copy.deepcopy(result)
    payload.pop("integrity_sha256", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _make_result() -> dict[str, Any]:
    protocol_sha256 = _protocol_sha256()
    primary = _compute(PRIMARY_DPS)
    control = _compute(CONTROL_DPS)
    precision = _precision_controls(primary, control)
    with mp.workdps(CONTROL_DPS):
        toy = _toy_control()
    if not all(item["pass"] for item in precision):
        raise ArithmeticError("independent precision control failed")
    if not toy["pass"]:
        raise ArithmeticError("analytic toy control failed")
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": STATUS_PASS,
        "evidence_weight": EVIDENCE_WEIGHT,
        # The physical scope is fixed, but the public numerical specification
        # records a post-freeze FD convergence refinement; do not claim a full
        # independently timestamped pre-row freeze.
        "protocol_frozen_before_rows": False,
        "physical_inputs_fixed": True,
        "protocol_provenance": PROTOCOL_PROVENANCE,
        "protocol_sha256": protocol_sha256,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_source_sha256": hashlib.sha256(UPSTREAM_PATH.read_bytes()).hexdigest(),
        "units_and_conventions": {
            "n": "natural MeV^3; n0=0.16 fm^-3 converted with hbarc",
            "epsilon": "MeV^4",
            "mu": "MeV",
            "epsilon_derivative_orders": "d^k epsilon/dn^k, no factorial absorbed",
            "chi": "chi_k=d^k n/dmu^k, no factorial absorbed",
            "x": "(n-n0)/n0",
            "u": "(mu-mu0)/(K/9)",
            "Z": "81*n0^4*d^4(epsilon/n)/dn^4 at n0, MeV",
        },
        "inputs": {
            "anchors": list(ANCHORS),
            "deformation_amplitudes": [0, 1],
            "response_u": list(RESPONSE_U),
            "inverse_fd_u_steps": list(FD_U_STEPS),
            "precision_digits": {"primary": PRIMARY_DPS, "independent_control": CONTROL_DPS},
            "branch_window_ratio": [_number(BRANCH_LOW), _number(BRANCH_HIGH)],
            "no_saved_results_as_inputs": True,
        },
        "anchors": _publicize(primary["anchors"]),
        "precision_controls": _publicize(precision),
        "analytic_toy_control": _publicize(toy),
        "interpretation_limits": [
            "local smooth positive-curvature homogeneous branch only",
            "not a global ground-state, phase-mixture, finite-density EOS, dynamic, resonance, or experimental claim",
            "DeltaU is a mathematical calibration-null counterexample with fixed positive coefficient",
        ],
    }
    result["integrity_sha256"] = hashlib.sha256(_canonical_payload(result)).hexdigest()
    return result


def build_result() -> dict[str, Any]:
    return _make_result()


def _walk_for_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return mp.isfinite(value)
    if isinstance(value, str) and value in {"NaN", "nan", "Infinity", "+Infinity", "-Infinity"}:
        return False
    if isinstance(value, dict):
        return all(_walk_for_nonfinite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_walk_for_nonfinite(v) for v in value)
    return True


def _numeric(value: Any) -> mp.mpf | None:
    """Parse a finite scalar without treating booleans as numbers."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = _mp(value)
    except (TypeError, ValueError):
        return None
    return parsed if mp.isfinite(parsed) else None


def _true_relative_close(given: Any, expected: Any, *, rel_tol: mp.mpf = SCIENCE_REL_TOL,
                         zero_tol: mp.mpf = ZERO_VALUE_TOL) -> bool:
    """Compare in the scale of the expected physical quantity."""
    g, e = _numeric(given), _numeric(expected)
    if g is None or e is None:
        return False
    if e == 0:
        return abs(g) <= zero_tol
    return abs(g - e) / abs(e) <= rel_tol


def _bounded(value: Any, limit: mp.mpf, *, signed: bool = False) -> bool:
    parsed = _numeric(value)
    if parsed is None:
        return False
    return abs(parsed) <= limit if signed else 0 <= parsed <= limit


def _exact_keys(value: Any, expected: tuple[str, ...] | set[str]) -> bool:
    return isinstance(value, dict) and set(value) == set(expected)


def _skip_tree_numeric(path: tuple[str, ...]) -> bool:
    last = path[-1] if path else ""
    if last in CONTROL_NUMERIC_FIELDS:
        return True
    # The calibration pressure and stationary residual are numerical controls,
    # not independent physical measurements.  They are checked by declared
    # bounds and by recomputing the state from the stored n,y below.
    return last == "pressure_total" and "anchor_state" in path


def _compare_public_tree(given: Any, expected: Any, path: tuple[str, ...] = ()) -> bool:
    """Schema-exact comparison of a JSON snapshot against a fresh live tree.

    Numeric leaves use true relative scale.  Only residual/identity controls
    explicitly listed in ``CONTROL_NUMERIC_FIELDS`` are exempted here; each is
    independently bounded and recomputed by the semantic checks below.
    """
    if isinstance(expected, dict):
        return (
            isinstance(given, dict)
            and set(given) == set(expected)
            and all(_compare_public_tree(given[key], expected[key], path + (str(key),)) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(given, list)
            and len(given) == len(expected)
            and all(_compare_public_tree(g, e, path + (str(i),)) for i, (g, e) in enumerate(zip(given, expected)))
        )
    if isinstance(expected, bool) or isinstance(given, bool):
        return isinstance(given, bool) and given == expected
    if expected is None or given is None:
        return given is None and expected is None
    expected_number = _numeric(expected)
    given_number = _numeric(given)
    if expected_number is not None and given_number is not None:
        if _skip_tree_numeric(path):
            return True
        return _true_relative_close(given_number, expected_number)
    return type(given) is type(expected) and given == expected


def _expected_top_contract() -> tuple[set[str], set[str], set[str]]:
    top = {
        "schema_version", "protocol_version", "status", "evidence_weight", "protocol_frozen_before_rows",
        "physical_inputs_fixed", "protocol_provenance",
        "protocol_sha256", "source_sha256", "upstream_source_sha256",
        "units_and_conventions", "inputs", "anchors", "precision_controls",
        "analytic_toy_control", "interpretation_limits", "integrity_sha256",
    }
    inputs = {
        "anchors", "deformation_amplitudes", "response_u", "inverse_fd_u_steps",
        "precision_digits", "branch_window_ratio", "no_saved_results_as_inputs",
    }
    anchor = {"anchor_inputs", "branch", "models", "comparison"}
    return top, inputs, anchor


def _validate_static_contract(result: dict[str, Any]) -> bool:
    """Reject labels, options, empty controls, and schema drift before rebuild."""
    top_keys, input_keys, anchor_keys = _expected_top_contract()
    if set(result) != top_keys:
        return False
    if result["status"] != STATUS_PASS or result["evidence_weight"] != EVIDENCE_WEIGHT:
        return False
    if (
        result["schema_version"] != SCHEMA_VERSION
        or result["protocol_version"] != PROTOCOL_VERSION
        or result["protocol_frozen_before_rows"] is not False
        or result["physical_inputs_fixed"] is not True
        or result["protocol_provenance"] != PROTOCOL_PROVENANCE
    ):
        return False
    if result["units_and_conventions"] != EXPECTED_UNITS:
        return False
    inputs = result["inputs"]
    if not _exact_keys(inputs, input_keys):
        return False
    if inputs["anchors"] != list(ANCHORS) or inputs["deformation_amplitudes"] != [0, 1]:
        return False
    if inputs["response_u"] != list(RESPONSE_U) or inputs["inverse_fd_u_steps"] != list(FD_U_STEPS):
        return False
    if inputs["precision_digits"] != {"primary": PRIMARY_DPS, "independent_control": CONTROL_DPS}:
        return False
    if inputs["no_saved_results_as_inputs"] is not True:
        return False
    window = inputs["branch_window_ratio"]
    if not isinstance(window, list) or len(window) != 2 or not _true_relative_close(window[0], BRANCH_LOW) or not _true_relative_close(window[1], BRANCH_HIGH):
        return False
    if set(result["anchors"]) != set(ANCHORS):
        return False
    for target in ANCHORS:
        anchor = result["anchors"][target]
        if not _exact_keys(anchor, anchor_keys):
            return False
        ai = anchor["anchor_inputs"]
        if not _exact_keys(ai, {"target_y", "n0_fm3", "binding_MeV", "K_target_MeV", "ensemble"}):
            return False
        if not _true_relative_close(ai["target_y"], _mp(target)) or not _true_relative_close(ai["n0_fm3"], "0.16"):
            return False
        if not _true_relative_close(ai["binding_MeV"], "-16") or not _true_relative_close(ai["K_target_MeV"], "240"):
            return False
        if ai["ensemble"] != "cold homogeneous fixed composition":
            return False
        branch = anchor["branch"]
        if not _exact_keys(branch, {"lineage", "density_window_ratio", "global_ground_state_claim", "phase_mixture_claim"}):
            return False
        if branch["lineage"] != "positive-y stationary continuation through common n0 anchor":
            return False
        if branch["density_window_ratio"] != ["0.5", "2.0"] or branch["global_ground_state_claim"] is not False or branch["phase_mixture_claim"] is not False:
            return False
        models = anchor["models"]
        if set(models) != set(MODEL_ORDER):
            return False
        for name in MODEL_ORDER:
            model = models[name]
            required = {"id", "design", "amplitude", "target_y", "eta_MeV4", "deformation", "derivatives", "responses", "inverse_fd_controls"}
            if not _exact_keys(model, required):
                return False
            if model["id"] != f"{name}_{target[2:]}" or model["amplitude"] != AMPLITUDES[name]:
                return False
            expected_design = (
                "accepted W8 calibration anchor y*=" + target
                if name == "w8"
                else "W8 plus positive U16 calibration-null deformation y*=" + target
            )
            if model["design"] != expected_design or not _true_relative_close(model["target_y"], _mp(target)):
                return False
            if len(model["responses"]) != len(RESPONSE_U) or len(model["inverse_fd_controls"]) != len(FD_U_STEPS):
                return False
            if [str(row.get("u")) for row in model["responses"]] != [str(_mp(u)) for u in RESPONSE_U]:
                # Numeric equality is required, while accepting mp's canonical
                # formatting (e.g. -0.20 -> -0.2) in a re-signed artifact.
                if any(not _true_relative_close(row.get("u"), u) for row, u in zip(model["responses"], RESPONSE_U)):
                    return False
            for row in model["responses"]:
                if row.get("branch_lineage") != BRANCH_LINEAGE or row.get("window_accept") is not True:
                    return False
            for fd, step in zip(model["inverse_fd_controls"], FD_U_STEPS):
                if fd.get("u_step") is None or not _true_relative_close(fd["u_step"], step):
                    return False
        comparison = anchor["comparison"]
        required_comparison = {
            "common_lower_jet_relative_differences", "delta_epsilon_4", "delta_chi3", "delta_Z_MeV",
            "delta_chi3_from_common_epsilon2", "delta_chi3_identity_relative", "dimensionless_definition",
            "delta_d3x_du3", "delta_d3x_du3_from_delta_Z", "dimensionless_identity_relative",
            "response_separation", "max_abs_response_delta_x", "numerical_floor",
            "nonzero_separation_above_floor", "interpretation",
        }
        if not _exact_keys(comparison, required_comparison) or len(comparison["response_separation"]) != len(RESPONSE_U):
            return False
        if comparison["dimensionless_definition"] != DIMENSIONLESS_DEFINITION or comparison["interpretation"] != INTERPRETATION:
            return False
    return True


def _validate_state_snapshot(stored: dict[str, Any], model: Any, *, anchor: bool) -> bool:
    expected_keys = {
        "k", "m", "ef", "energy", "pressure", "ns", "ns_m", "n", "y", "U", "Uy", "Uyy",
        "vector", "energy_total", "pressure_total", "mu", "residual", "C_y", "B_y", "D_nn",
        "mu_prime", "K", "cs2",
    }
    if not _exact_keys(stored, expected_keys):
        return False
    n, y = _numeric(stored["n"]), _numeric(stored["y"])
    if n is None or y is None or n <= 0 or y <= 0:
        return False
    try:
        live = _state(model, n, y)
    except (ArithmeticError, ValueError, ZeroDivisionError):
        return False
    if not (_finite(live["C_y"]) and _finite(live["mu_prime"]) and live["C_y"] > 0 and live["mu_prime"] > 0):
        return False
    scale = _stationarity_scale(model, live)
    if abs(live["residual"]) / scale > STATIONARITY_REL_LIMIT:
        return False
    # Every state number is tied to the live functional at the stored (n,y).
    # Calibration pressure and residual are bounded controls, not last-digit
    # observables, so they intentionally use declared scaled bounds.
    for key in expected_keys:
        if key == "residual":
            if abs(_numeric(stored[key]) or mp.inf) / scale > STATIONARITY_REL_LIMIT:
                return False
            continue
        if key == "pressure_total" and anchor:
            value = _numeric(stored[key])
            energy = _numeric(stored["energy_total"])
            if value is None or energy is None or abs(value) / max(abs(energy), mp.mpf(1)) > CALIBRATION_REL_LIMIT:
                return False
            continue
        if not _true_relative_close(stored[key], live[key]):
            return False
    return True


def _validate_derivative_snapshot(stored: dict[str, Any], model: Any, anchor_y: mp.mpf) -> bool:
    required = {"anchor_state", "epsilon_derivatives", "mu0", "chi", "Z_MeV", "energy_per_particle_fourth_derivative", "checks", "method"}
    if not _exact_keys(stored, required) or stored["method"] != "live_stationary_energy_mp.diff_orders_1_to_4":
        return False
    if not _validate_state_snapshot(stored["anchor_state"], model, anchor=True):
        return False
    state = stored["anchor_state"]
    if not _true_relative_close(state["n"], model.n0) or not _true_relative_close(state["y"], anchor_y):
        return False
    eps = stored["epsilon_derivatives"]
    if not _exact_keys(eps, {"epsilon_1", "epsilon_2", "epsilon_3", "epsilon_4"}):
        return False
    e1, e2, e3, e4 = (_numeric(eps[key]) for key in ("epsilon_1", "epsilon_2", "epsilon_3", "epsilon_4"))
    if any(value is None for value in (e1, e2, e3, e4)) or e2 <= 0 or e4 == 0:
        return False
    mu0 = _numeric(stored["mu0"])
    if mu0 is None or not _true_relative_close(mu0, e1) or not _true_relative_close(mu0, state["mu"]):
        return False
    chi = stored["chi"]
    if not _exact_keys(chi, {"chi1", "chi2", "chi3"}):
        return False
    chi1, chi2, chi3 = (_numeric(chi[key]) for key in ("chi1", "chi2", "chi3"))
    if any(value is None for value in (chi1, chi2, chi3)) or not _true_relative_close(chi1, 1 / e2) or not _true_relative_close(chi2, -e3 / e2**3) or not _true_relative_close(chi3, 3 * e3**2 / e2**5 - e4 / e2**4):
        return False
    z = _numeric(stored["Z_MeV"])
    z4 = _numeric(stored["energy_per_particle_fourth_derivative"])
    if z is None or z4 is None or z == 0:
        return False
    checks = stored["checks"]
    if not _exact_keys(checks, {"epsilon1_equals_mu", "epsilon2_equals_mu_prime", "stationarity_relative", "positive_curvature", "positive_compressibility", "finite"}):
        return False
    if not _bounded(checks["epsilon1_equals_mu"], IDENTITY_LIMIT) or not _bounded(checks["epsilon2_equals_mu_prime"], IDENTITY_LIMIT) or not _bounded(checks["stationarity_relative"], STATIONARITY_REL_LIMIT):
        return False
    if checks["positive_curvature"] is not True or checks["positive_compressibility"] is not True or checks["finite"] is not True:
        return False
    return True


def _validate_response_rows(model_info: dict[str, Any], anchor: dict[str, Any], model: Any, target_y: mp.mpf) -> bool:
    model_deriv = model_info["derivatives"]
    eps2 = _numeric(model_deriv["epsilon_derivatives"]["epsilon_2"])
    mu0 = _numeric(model_deriv["mu0"])
    if eps2 is None or mu0 is None:
        return False
    n0 = model.n0
    K = 9 * n0 * eps2
    rows = model_info["responses"]
    for row, expected_u_text in zip(rows, RESPONSE_U):
        expected_keys = {"u", "x", "n_MeV3", "y", "mu_target_MeV", "mu_solved_MeV", "mu_residual_MeV", "mu_relative", "stationarity_relative", "C_y", "mu_prime", "K_local_MeV", "branch_lineage", "window_accept"}
        if not _exact_keys(row, expected_keys):
            return False
        u = _numeric(row["u"])
        n = _numeric(row["n_MeV3"])
        y = _numeric(row["y"])
        if any(value is None for value in (u, n, y)) or not _true_relative_close(u, expected_u_text) or not (n0 * BRANCH_LOW <= n <= n0 * BRANCH_HIGH) or y <= 0:
            return False
        try:
            live = _state(model, n, y)
        except (ArithmeticError, ValueError, ZeroDivisionError):
            return False
        if not (_finite(live["C_y"]) and _finite(live["mu_prime"]) and live["C_y"] > 0 and live["mu_prime"] > 0):
            return False
        x_expected = (n - n0) / n0
        target_mu = mu0 + u * K / 9
        if not _true_relative_close(row["x"], x_expected) or not _true_relative_close(row["mu_target_MeV"], target_mu) or not _true_relative_close(row["mu_solved_MeV"], live["mu"]):
            return False
        mu_residual = _numeric(row["mu_residual_MeV"])
        if mu_residual is None:
            return False
        residual_scale = max(abs(target_mu), mp.mpf(1))
        if abs(live["residual"]) / _stationarity_scale(model, live) > STATIONARITY_REL_LIMIT or abs(mu_residual) / residual_scale > INVERSE_REL_LIMIT or abs(mu_residual - (live["mu"] - target_mu)) / residual_scale > INVERSE_REL_LIMIT:
            return False
        if not _bounded(row["mu_relative"], INVERSE_REL_LIMIT) or not _bounded(row["stationarity_relative"], STATIONARITY_REL_LIMIT):
            return False
        expected_mu_relative = abs(mu_residual) / residual_scale
        stored_mu_relative = _numeric(row["mu_relative"])
        if expected_mu_relative == 0:
            if stored_mu_relative is None or abs(stored_mu_relative) > ZERO_VALUE_TOL:
                return False
        elif not _true_relative_close(stored_mu_relative, expected_mu_relative):
            return False
        if not _true_relative_close(row["C_y"], live["C_y"]) or not _true_relative_close(row["mu_prime"], live["mu_prime"]) or not _true_relative_close(row["K_local_MeV"], 9 * n * live["mu_prime"]):
            return False
        if row["branch_lineage"] != BRANCH_LINEAGE or row["window_accept"] is not True:
            return False
    return True


def _validate_fd_controls(model_info: dict[str, Any]) -> bool:
    controls = model_info["inverse_fd_controls"]
    if len(controls) != len(FD_U_STEPS):
        return False
    for control, step in zip(controls, FD_U_STEPS):
        required = {"u_step", "dmu_step_MeV", "chi1_fd", "chi2_fd", "chi3_fd", "chi1_relative_error", "chi2_relative_error", "chi3_relative_error", "max_inverse_root_relative", "truncation_control", "pass"}
        if not _exact_keys(control, required) or not _true_relative_close(control["u_step"], step) or control["truncation_control"] != "decreasing_symmetric_five_point_step":
            return False
        numeric_values = [control[key] for key in required if key not in {"truncation_control", "pass"}]
        if any(_numeric(value) is None for value in numeric_values):
            return False
        if not _bounded(control["max_inverse_root_relative"], INVERSE_REL_LIMIT):
            return False
        expected_pass = (
            _numeric(control["chi1_relative_error"]) < mp.mpf("2e-3")
            and _numeric(control["chi2_relative_error"]) < mp.mpf("2e-3")
            and _numeric(control["chi3_relative_error"]) <= FD_CHI3_REL_LIMIT
        )
        if control["pass"] is not expected_pass:
            return False
    return _numeric(controls[-1]["chi3_relative_error"]) < _numeric(controls[0]["chi3_relative_error"])


def _validate_comparison(anchor: dict[str, Any], target: str) -> bool:
    comparison = anchor["comparison"]
    w8, u16 = anchor["models"]["w8"], anchor["models"]["u16"]
    weps, ueps = w8["derivatives"]["epsilon_derivatives"], u16["derivatives"]["epsilon_derivatives"]
    wchi, uchi = w8["derivatives"]["chi"], u16["derivatives"]["chi"]
    delta_e4 = _numeric(ueps["epsilon_4"]) - _numeric(weps["epsilon_4"])
    delta_chi3 = _numeric(uchi["chi3"]) - _numeric(wchi["chi3"])
    delta_z = _numeric(u16["derivatives"]["Z_MeV"]) - _numeric(w8["derivatives"]["Z_MeV"])
    n0 = _mp("0.16") * _mp("197.3269804") ** 3
    K = 9 * n0 * _numeric(weps["epsilon_2"])
    formula_chi3 = -delta_e4 / _numeric(weps["epsilon_2"]) ** 4
    d3 = K**3 / (9**3 * n0) * delta_chi3
    d3_z = -delta_z / (9 * K)
    if not _true_relative_close(comparison["delta_epsilon_4"], delta_e4) or not _true_relative_close(comparison["delta_chi3"], delta_chi3) or not _true_relative_close(comparison["delta_Z_MeV"], delta_z):
        return False
    if not _true_relative_close(comparison["delta_chi3_from_common_epsilon2"], formula_chi3) or not _true_relative_close(comparison["delta_d3x_du3"], d3) or not _true_relative_close(comparison["delta_d3x_du3_from_delta_Z"], d3_z):
        return False
    if not _bounded(comparison["delta_chi3_identity_relative"], IDENTITY_LIMIT) or not _bounded(comparison["dimensionless_identity_relative"], IDENTITY_LIMIT):
        return False
    lower = comparison["common_lower_jet_relative_differences"]
    if not _exact_keys(lower, {"epsilon_1", "epsilon_2", "epsilon_3", "mu0"}) or any(not _bounded(lower[key], IDENTITY_LIMIT) for key in lower):
        return False
    separations = comparison["response_separation"]
    if len(separations) != len(RESPONSE_U):
        return False
    max_sep = mp.mpf(0)
    for index, separation in enumerate(separations):
        if not _exact_keys(separation, {"u", "delta_x_u16_minus_w8", "w8_x", "u16_x", "same_target_mu"}) or not _true_relative_close(separation["u"], RESPONSE_U[index]) or not _bounded(separation["same_target_mu"], ZERO_VALUE_TOL):
            return False
        expected_delta = _numeric(u16["responses"][index]["x"]) - _numeric(w8["responses"][index]["x"])
        if not _true_relative_close(separation["delta_x_u16_minus_w8"], expected_delta) or not _true_relative_close(separation["w8_x"], w8["responses"][index]["x"]) or not _true_relative_close(separation["u16_x"], u16["responses"][index]["x"]):
            return False
        max_sep = max(max_sep, abs(expected_delta))
    floor = _numeric(comparison["numerical_floor"])
    if floor is None or not _true_relative_close(floor, SEPARATION_FLOOR) or not _true_relative_close(comparison["max_abs_response_delta_x"], max_sep):
        return False
    if comparison["nonzero_separation_above_floor"] is not bool(abs(delta_chi3) > floor and max_sep > floor):
        return False
    return abs(delta_e4) > floor and abs(delta_chi3) > floor and delta_z > floor and max_sep > floor


def validate_result(result: Any) -> bool:
    """Fail closed on a re-signed artifact by binding every scientific leaf."""
    try:
        if not isinstance(result, dict) or not _walk_for_nonfinite(result):
            return False
        if not _validate_static_contract(result):
            return False
        stored_integrity = result.get("integrity_sha256")
        if not isinstance(stored_integrity, str) or hashlib.sha256(_canonical_payload(result)).hexdigest() != stored_integrity:
            return False
        if result.get("source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            return False
        if result.get("upstream_source_sha256") != hashlib.sha256(UPSTREAM_PATH.read_bytes()).hexdigest():
            return False
        if result.get("protocol_sha256") != _protocol_sha256():
            return False
        with mp.workdps(CONTROL_DPS):
            fresh = _compute(PRIMARY_DPS)
            control = _compute(CONTROL_DPS)
            expected_anchors = _publicize(fresh["anchors"])
            if not _compare_public_tree(result["anchors"], expected_anchors):
                return False
            expected_precision = _publicize(_precision_controls(fresh, control))
            if not _compare_public_tree(result["precision_controls"], expected_precision):
                return False
            expected_toy = _publicize(_toy_control())
            if not _compare_public_tree(result["analytic_toy_control"], expected_toy):
                return False
            for target in ANCHORS:
                stored_anchor = result["anchors"][target]
                infos = _build_models(target)
                for name in MODEL_ORDER:
                    stored_model = stored_anchor["models"][name]
                    model = infos[name]["model"]
                    target_y = _mp(target)
                    if not _validate_derivative_snapshot(stored_model["derivatives"], model, target_y):
                        return False
                    if not _validate_response_rows(stored_model, stored_anchor, model, target_y) or not _validate_fd_controls(stored_model):
                        return False
                    deformation = stored_model["deformation"]
                    if stored_model["amplitude"] == 1:
                        if not deformation["delta_U_nonnegative_by_even_power_proof"] or not deformation["exact_zero_through_order_3_at_vacuum_and_anchor"] or not _bounded(deformation["delta_U_fourth_derivative_at_anchor_MeV4"], mp.inf):
                            return False
                    else:
                        if deformation["amplitude"] != 0:
                            return False
                if not _validate_comparison(stored_anchor, target):
                    return False
            # Precision rows are evidence, not trusted pass flags.  Recompute
            # their pass predicates from the stored values and protocol limits.
            for item in result["precision_controls"]:
                if not _exact_keys(item, {"anchor", "model", "primary_dps", "control_dps", "max_relative_difference", "fields", "pass"}):
                    return False
                if item["primary_dps"] != PRIMARY_DPS or item["control_dps"] != CONTROL_DPS or set(item["fields"]) != {"epsilon_1", "epsilon_2", "epsilon_3", "epsilon_4", "chi3", "Z_MeV", "response_x_0", "response_x_1", "response_x_2", "response_x_3"}:
                    return False
                maximum = max(_numeric(value) for value in item["fields"].values())
                if maximum is None or not _true_relative_close(item["max_relative_difference"], maximum) or item["pass"] is not bool(maximum <= PRECISION_REL_LIMIT):
                    return False
            return True
    except (ArithmeticError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


def _write_json(payload: dict[str, Any], path: Path | None) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path is None:
        sys.stdout.write(encoded)
    else:
        path.write_text(encoded, encoding="utf-8")
        sys.stdout.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional explicit JSON output path")
    args = parser.parse_args(argv)
    try:
        result = build_result()
        if not validate_result(result):
            raise ArithmeticError("fresh result failed fail-closed validation")
        _write_json(result, args.output)
        return 0
    except Exception as exc:  # CLI errors are strict JSON and never PASS.
        failure = {"schema_version": SCHEMA_VERSION, "status": STATUS_FAIL, "error": str(exc)}
        _write_json(failure, args.output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
