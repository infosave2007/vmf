#!/usr/bin/env python3
"""Fresh, no-write density-tail and calibration-null audit.

The calculation is deliberately narrower than an EOS or cosmology pipeline.
It reuses the accepted homogeneous BulkModel and reconstructs two positive
degree-16 deformations at the declared W8 calibration points.  All numbers
printed by the CLI are computed in the current process; no result table is an
input and the CLI never writes a file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mpmath as mp


HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
UPSTREAM_PATH = HERE / "source_complete_scaling_saturation_audit.py"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import source_complete_scaling_saturation_audit as upstream


SCHEMA = 1
EVIDENCE_WEIGHT = 0.0
DENSITY_RATIOS = ("1e2", "1e4", "1e8", "1e12", "1e18", "1e24", "1e36")
CONTROL_RATIOS = ("1e4", "1e36")
CASE_ORDER = ("original_quartic", "w8_90", "w8_93", "u16_90", "u16_93")
FD_STEPS = ("1e-5", "5e-6")
STATIONARITY_LIMIT = mp.mpf("1e-25")
IDENTITY_LIMIT = mp.mpf("1e-25")
FD_LIMIT = mp.mpf("1e-7")
ASYMPTOTIC_LIMIT = mp.mpf("1e-5")
ASYMPTOTIC_FIELD_ENERGY_LIMIT = mp.mpf("1e-4")
PRECISION_LIMIT = mp.mpf("1e-25")
NUMERIC_TIE_LIMIT = mp.mpf("1e-25")
VALIDATION_TARGETS = ("0.90", "0.93")
ROW_KEYS = ("density_ratio", "n_MeV3", "n_over_W0_cubed", "root", "state", "raw_evidence", "derived", "checks", "fd_checks")
STATE_KEYS = (
    "y", "energy_density_MeV4", "pressure_MeV4", "pressure_MeV_fm3", "mu_MeV",
    "binding_MeV", "U_MeV4", "Uy_MeV4", "Uyy_MeV4", "fermi_energy_MeV4",
    "vector_MeV4", "C_y_MeV4", "B_y", "D_nn", "mu_prime", "K_MeV", "cs2", "trace_MeV4",
)
ROOT_KEYS = (
    "y", "bracket_low", "bracket_high", "bracket_expansions",
    "sign_change_bracket_count_lower_bound", "positive_root_bracketed",
    "not_an_exhaustive_root_count", "residual_scale",
)
RAW_KEYS = ("residual", "stationarity_scale", "stationarity_terms", "legendre_residual", "trace_identity_residual")
STATIONARITY_TERM_KEYS = ("matter_derivative", "potential_derivative", "vector_derivative")
DERIVED_KEYS = (
    "pressure_MeV_fm3", "binding_MeV", "trace_MeV4", "trace_over_energy",
    "pressure_over_energy", "field_log_response", "energy_log_slope", "fractions",
    "asymptotic_field_scale", "asymptotic_energy_scale", "field_ratio", "energy_ratio",
)
CHECK_KEYS = ("stationarity_relative", "legendre_relative", "trace_identity_relative", "positive_C", "positive_mu", "finite")
FRACTION_KEYS = ("fermi", "potential", "vector")
FD_EVIDENCE_KEYS = ("n_minus", "n_plus", "y_minus", "y_plus", "mu_minus", "mu_plus")
FD_KEYS = FD_EVIDENCE_KEYS + (
    "mu_prime_fd", "mu_prime_analytic", "mu_relative_error", "field_log_response_fd",
    "field_log_response_analytic", "field_relative_error", "pass",
)
PRECISION_ROW_FIELDS = ("y", "energy_total", "pressure_total", "mu", "C_y", "K", "cs2")
PRECISION_CALIBRATION_FIELDS = ("energy_total", "pressure_total", "mu", "C_y", "K", "cs2")


def _mp(value):
    return value if isinstance(value, mp.mpf) else mp.mpf(str(value))


def _target_id(prefix, target):
    return prefix + str(target).replace("0.", "")


PRECISION_KEY_ORDER = tuple(
    (case_id, ratio) for case_id in CASE_ORDER for ratio in CONTROL_RATIOS
) + tuple(
    (case_id, "1")
    for target in VALIDATION_TARGETS
    for case_id in (_target_id("w8_", target), _target_id("u16_", target))
)


def _finite(value):
    try:
        return mp.isfinite(_mp(value))
    except (TypeError, ValueError):
        return False


def _number(value, digits=60):
    value = _mp(value)
    if not mp.isfinite(value):
        raise ValueError("nonfinite scientific value")
    return mp.nstr(value, digits)


def _relative(a, b):
    a, b = _mp(a), _mp(b)
    return abs(a - b) / max(abs(a), abs(b), mp.mpf(1))


def _relative_scale(value, scale):
    value, scale = _mp(value), _mp(scale)
    return abs(value) / max(abs(scale), mp.mpf(1))


def _tail_factor_derivatives(y, target_y):
    """Return f, df/dy, d2f/dy2, d3f/dy3 for f=(y2-1)^4(y2-t2)^4."""
    y, target_y = _mp(y), _mp(target_y)
    q = y * y
    a, b = q - 1, q - target_y * target_y
    f = a**4 * b**4
    f1 = 4 * a**3 * b**4 + 4 * a**4 * b**3
    f2 = 12 * a**2 * b**4 + 32 * a**3 * b**3 + 12 * a**4 * b**2
    f3 = 24 * a * b**4 + 144 * a**2 * b**3 + 144 * a**3 * b**2 + 24 * a**4 * b
    return f, 2 * y * f1, 2 * f1 + 4 * q * f2, 12 * y * f2 + 8 * y**3 * f3


def _solve_w8_coefficients(base, target_y):
    """Reconstruct the fixed three-jet polynomial without a matrix solve.

    The production builder uses the same three calibration equations through
    ``inverse_potential_jet``.  Validation deliberately uses the closed-form
    elimination below so it can check serialized rows without rebuilding that
    calibration matrix or rerunning the finite-density roots.
    """
    target_y = _mp(target_y)
    n = base.n0
    mu0 = base.MN - 16
    f = base.fermi(n, target_y)
    vector_gap = mu0 - f["ef"]
    cv = vector_gap * target_y**2 / n
    d_nn = f["k"]**2 / (3 * n * f["ef"]) + cv / target_y**2
    b_y = base.MN**2 * target_y / f["ef"] - 2 * cv * n / target_y**3
    denominator = d_nn - mp.mpf(240) / (9 * n)
    curvature = b_y**2 / denominator
    potential = f["pressure"] + n * vector_gap / 2
    potential_y = (n * vector_gap - f["m"] * f["ns"]) / target_y
    potential_yy = curvature - base.MN**2 * f["ns_m"] - 3 * cv * n**2 / target_y**4

    # Write U=z^2(a2+a3*z+a4*z^2), then solve for q=U/z^2,
    # r=U_y/(2*y*z), and h=U_yy/(4*y^2) after removing the first term.
    z = target_y**2 - 1
    q = potential / z**2
    r = (potential_y / (2 * target_y)) / z
    u_z = potential_y / (2 * target_y)
    u_zz = (potential_yy - 2 * u_z) / (4 * target_y**2)
    a4_z2 = (u_zz - 4 * r + 6 * q) / 2
    a3_z = 5 * r - 8 * q - u_zz
    a2 = 6 * q - 3 * r + u_zz / 2
    return (a2, a3_z / z, a4_z2 / z**2), {
        "Cv": cv,
        "U": potential,
        "Uy": potential_y,
        "Uyy": potential_yy,
        "K": mp.mpf(240),
    }


def _validation_cases():
    """Build the five fixed validation models from declared inputs only."""
    base = upstream.BulkModel()
    cases = {
        "original_quartic": {
            "id": "original_quartic",
            "model": base,
            "design": "original quartic",
            "target_y": None,
            "tail_degree": 4,
            "tail_coefficient": base.A / 4,
            "deformation_eta": None,
            "polynomial": None,
        }
    }
    for target in VALIDATION_TARGETS:
        target_y = _mp(target)
        coefficients, calibration = _solve_w8_coefficients(base, target_y)
        cv = calibration["Cv"]
        w8 = upstream.BulkModel(gomega=base.momega * mp.sqrt(cv), polynomial=coefficients)
        wid = _target_id("w8_", target)
        uid = _target_id("u16_", target)
        eta = coefficients[-1]
        cases[wid] = {
            "id": wid,
            "model": w8,
            "design": "inverse W8 y*=" + target,
            "target_y": target_y,
            "tail_degree": 8,
            "tail_coefficient": coefficients[-1],
            "deformation_eta": None,
            "polynomial": coefficients,
        }
        cases[uid] = {
            "id": uid,
            "model": PotentialAdapter(w8, target_y, eta),
            "design": "positive U16 deformation of W8 y*=" + target,
            "target_y": target_y,
            "tail_degree": 16,
            "tail_coefficient": eta,
            "deformation_eta": eta,
            "polynomial": coefficients,
            "base_model": w8,
        }
    return [cases[cid] for cid in CASE_ORDER]


class PotentialAdapter:
    """Small wrapper adding one declared nonnegative tail deformation."""

    def __init__(self, base, target_y, eta):
        self.base = base
        self.target_y = _mp(target_y)
        self.eta = _mp(eta)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def potential(self, y):
        u, uy, uyy = self.base.potential(y)
        t, ty, tyy, _ = _tail_factor_derivatives(y, self.target_y)
        return u + self.eta * t, uy + self.eta * ty, uyy + self.eta * tyy

    def deformation_derivatives(self, y):
        t, ty, tyy, tyyy = _tail_factor_derivatives(y, self.target_y)
        return tuple(self.eta * value for value in (t, ty, tyy, tyyy))


def _state(model, n, y):
    """BulkModel state, with the potential supplied by the active adapter."""
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
    mu_prime = d_nn - b_y * b_y / c_y if c_y else mp.nan
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
        D=d_nn,
        mu_prime=mu_prime,
        K=9 * n * mu_prime,
        cs2=n * mu_prime / mu,
    )


def _residual(model, n, y):
    return _state(model, n, y)["residual"]


def _bisect_function(function, lo, hi, *, tolerance="1e-55", iterations=650):
    lo, hi = _mp(lo), _mp(hi)
    flo, fhi = function(lo), function(hi)
    if not (flo < 0 < fhi):
        raise ArithmeticError("positive root is not bracketed with negative/positive endpoints")
    tol = _mp(tolerance)
    mid, fmid = lo, flo
    for _ in range(iterations):
        mid = (lo + hi) / 2
        fmid = function(mid)
        if abs(fmid) <= tol:
            return mid
        if fmid < 0:
            lo = mid
        else:
            hi = mid
    if abs(fmid) > tol:
        raise ArithmeticError("bracketed root did not reach the residual tolerance")
    return mid


def _bracket(model, n, guess):
    """Find a sign-changing positive interval by logarithmic expansion."""
    guess = _mp(guess)
    if guess <= 0:
        raise ArithmeticError("positive root guess is required")
    lo, hi = guess / 10, guess * 10
    flo, fhi = _residual(model, n, lo), _residual(model, n, hi)
    expansions = 0
    for _ in range(80):
        if flo < 0:
            break
        lo /= 10
        flo = _residual(model, n, lo)
        expansions += 1
    else:
        raise ArithmeticError("failed to find a negative lower endpoint")
    for _ in range(80):
        if fhi > 0:
            break
        hi *= 10
        fhi = _residual(model, n, hi)
        expansions += 1
    else:
        raise ArithmeticError("failed to find a positive upper endpoint")
    return lo, hi, expansions


def _root(model, n, guess, *, scan=True):
    lo, hi, expansions = _bracket(model, n, guess)
    lower_sign_changes = 0
    if scan:
        # This is a finite sign-change coverage diagnostic, not an exhaustive
        # theorem about all positive stationary roots.
        scan_lo, scan_hi = lo / 1000, hi * 1000
        previous = _residual(model, n, scan_lo)
        for j in range(1, 65):
            point = scan_lo * (scan_hi / scan_lo) ** (mp.mpf(j) / 64)
            current = _residual(model, n, point)
            if previous == 0 or current == 0 or previous * current < 0:
                lower_sign_changes += 1
            previous = current
    def scaled_residual(value):
        state = _state(model, n, value)
        scale = abs(model.MN * state["ns"]) + abs(state["Uy"]) + abs(model.Cv * n * n / value**3)
        return state["residual"] / max(scale, mp.mpf(1))

    y = _bisect_function(scaled_residual, lo, hi)
    state = _state(model, n, y)
    scale = abs(model.MN * state["ns"]) + abs(state["Uy"]) + abs(model.Cv * n * n / y**3)
    return y, {
        "bracket_low": lo,
        "bracket_high": hi,
        "bracket_expansions": expansions,
        "sign_change_bracket_count_lower_bound": lower_sign_changes,
        "positive_root_bracketed": True,
        "not_an_exhaustive_root_count": True,
        "residual_scale": scale,
    }


def _make_cases():
    base = upstream.BulkModel()
    cases = {
        "original_quartic": {
            "id": "original_quartic",
            "model": base,
            "design": "original quartic",
            "target_y": None,
            "tail_degree": 4,
            "tail_coefficient": base.A / 4,
            "deformation_eta": None,
            "calibration_input": False,
        }
    }
    for target in ("0.90", "0.93"):
        model, _, jet = upstream.inverse_potential_jet(target)
        wid = _target_id("w8_", target)
        uid = _target_id("u16_", target)
        target_y = _mp(target)
        eta = jet["coefficients"][2]
        cases[wid] = {
            "id": wid,
            "model": model,
            "design": "inverse W8 y*=" + target,
            "target_y": target_y,
            "tail_degree": 8,
            "tail_coefficient": model.polynomial[-1],
            "deformation_eta": None,
            "calibration_input": True,
            "jet": jet,
        }
        cases[uid] = {
            "id": uid,
            "model": PotentialAdapter(model, target_y, eta),
            "design": "positive U16 deformation of W8 y*=" + target,
            "target_y": target_y,
            "tail_degree": 16,
            "tail_coefficient": eta,
            "deformation_eta": eta,
            "calibration_input": True,
            "jet": jet,
            "base_id": wid,
        }
    return [cases[name] for name in CASE_ORDER]


def _limit_for(case):
    model = case["model"]
    p = case["tail_degree"]
    a = case["tail_coefficient"]
    if not (a > 0):
        raise ArithmeticError("tail coefficient must be positive")
    if p == 4:
        # s=n/W0^3 and y=z*s^(1/3).  This retains the full Fermi function;
        # dropping it is kept as an explicit negative control.
        n_unit = model.W0**3
        kinetic_free_z = (model.Cv * model.W0**6 / (4 * a)) ** (mp.mpf(1) / 6)

        def limiting_residual(z):
            f = model.fermi(n_unit, z)
            return model.MN * f["ns"] + 4 * a * z**3 - model.Cv * n_unit**2 / z**3

        def scaled_limiting_residual(z):
            f = model.fermi(n_unit, z)
            value = limiting_residual(z)
            scale = abs(model.MN * f["ns"]) + abs(4 * a * z**3) + abs(model.Cv * n_unit**2 / z**3)
            return value / max(scale, mp.mpf(1))

        lo, hi, _ = _bracket_generic(limiting_residual, kinetic_free_z)
        z = _bisect_function(scaled_limiting_residual, lo, hi)
        f = model.fermi(n_unit, z)
        fermi_coefficient = f["energy"] / model.W0**4
        potential_coefficient = a * z**4 / model.W0**4
        vector_coefficient = model.Cv * n_unit**2 / (2 * z**2 * model.W0**4)
        energy_coefficient = fermi_coefficient + potential_coefficient + vector_coefficient
        curvature = 12 * a * z**2 + model.MN**2 * f["ns_m"] + 3 * model.Cv * n_unit**2 / z**4
        kinetic_free_energy = a * kinetic_free_z**4 + model.Cv * n_unit**2 / (2 * kinetic_free_z**2)
        return {
            "kind": "full_fermi_p4_limit",
            "p": 4,
            "z": z,
            "energy_coefficient": energy_coefficient,
            "curvature": curvature,
            "kinetic_free_z": kinetic_free_z,
            "kinetic_free_energy_coefficient": kinetic_free_energy / model.W0**4,
            "expected_fermi_fraction": fermi_coefficient / energy_coefficient,
            "expected_potential_fraction": potential_coefficient / energy_coefficient,
            "expected_vector_fraction": vector_coefficient / energy_coefficient,
            "expected_trace_over_energy": mp.mpf(0),
            "expected_w": mp.mpf(1) / 3,
            "expected_cs2": mp.mpf(1) / 3,
            "nondegenerate": curvature > 0,
            "fermion_subleading": False,
        }
    c = (model.Cv / (p * a)) ** (mp.mpf(1) / (p + 2))
    energy_prefactor = (1 + mp.mpf(p) / 2) * a * c**p
    curvature_factor = (p + 2) * model.Cv
    return {
        "kind": "vector_potential_balance",
        "p": p,
        "c": c,
        "energy_prefactor": energy_prefactor,
        "curvature_factor": curvature_factor,
        "expected_w": (mp.mpf(p) - 2) / (p + 2),
        "expected_cs2": (mp.mpf(p) - 2) / (p + 2),
        "nondegenerate": True,
        "w_over_kf_exponent": (mp.mpf(4) - p) / (3 * (p + 2)),
        "fermi_over_energy_exponent": 2 * (mp.mpf(4) - p) / (3 * (p + 2)),
        "matter_derivative_over_vector_exponent": 4 * (mp.mpf(4) - p) / (3 * (p + 2)),
        "expected_fermi_fraction": mp.mpf(0),
        "expected_potential_fraction": mp.mpf(2) / (p + 2),
        "expected_vector_fraction": mp.mpf(p) / (p + 2),
        "expected_trace_over_energy": mp.mpf(2 * (4 - p)) / (p + 2),
        "fermion_subleading": True,
    }


def _bracket_generic(function, guess):
    guess = _mp(guess)
    if guess <= 0:
        raise ArithmeticError("positive limiting-root guess is required")
    lo, hi = guess / 10, guess * 10
    flo, fhi = function(lo), function(hi)
    expansions = 0
    for _ in range(80):
        if flo < 0:
            break
        lo /= 10
        flo = function(lo)
        expansions += 1
    else:
        raise ArithmeticError("failed to bracket limiting residual below")
    for _ in range(80):
        if fhi > 0:
            break
        hi *= 10
        fhi = function(hi)
        expansions += 1
    else:
        raise ArithmeticError("failed to bracket limiting residual above")
    return lo, hi, expansions


def _scales(case, limit, n):
    model = case["model"]
    n = _mp(n)
    if limit["p"] == 4:
        s = n / model.W0**3
        y_as = limit["z"] * s ** (mp.mpf(1) / 3)
        e_as = limit["energy_coefficient"] * model.W0**4 * s ** (mp.mpf(4) / 3)
    else:
        p = limit["p"]
        y_as = limit["c"] * n ** (mp.mpf(2) / (p + 2))
        e_as = limit["energy_prefactor"] * n ** (mp.mpf(2 * p) / (p + 2))
    return y_as, e_as


def _fd_check(case, row_state, limit):
    model, n, y = case["model"], row_state["n"], row_state["y"]
    checks = {}
    for htext in FD_STEPS:
        h = _mp(htext)
        n_minus, n_plus = n * mp.exp(-h), n * mp.exp(h)
        y_minus, _ = _root(model, n_minus, y, scan=False)
        y_plus, _ = _root(model, n_plus, y, scan=False)
        sm, sp = _state(model, n_minus, y_minus), _state(model, n_plus, y_plus)
        mu_fd = (sp["mu"] - sm["mu"]) / (2 * n * mp.sinh(h))
        field_fd = (mp.log(y_plus) - mp.log(y_minus)) / (2 * h)
        mu_error = _relative(mu_fd, row_state["mu_prime"])
        field_target = -n * row_state["B_y"] / (y * row_state["C_y"])
        field_error = _relative(field_fd, field_target)
        checks[htext] = {
            "n_minus": n_minus,
            "n_plus": n_plus,
            "y_minus": y_minus,
            "y_plus": y_plus,
            "mu_minus": sm["mu"],
            "mu_plus": sp["mu"],
            "mu_prime_fd": mu_fd,
            "mu_prime_analytic": row_state["mu_prime"],
            "mu_relative_error": mu_error,
            "field_log_response_fd": field_fd,
            "field_log_response_analytic": field_target,
            "field_relative_error": field_error,
            "pass": mu_error <= FD_LIMIT and field_error <= FD_LIMIT,
        }
    return checks


def _row(case, ratio, limit):
    model = case["model"]
    ratio_mp = _mp(ratio)
    n = model.n0 * ratio_mp
    y_as, e_as = _scales(case, limit, n)
    y, root_info = _root(model, n, y_as)
    s = _state(model, n, y)
    stationarity_scale = root_info["residual_scale"]
    legendre = s["energy_total"] + s["pressure_total"] - s["n"] * s["mu"]
    trace = s["energy_total"] - 3 * s["pressure_total"]
    trace_identity = trace - 4 * s["U"] + s["y"] * s["Uy"]
    row = {
        "density_ratio": str(ratio),
        "n_MeV3": n,
        "n_over_W0_cubed": n / model.W0**3,
        "root": {**root_info, "y": y},
        "state": s,
        "derived": {
            "pressure_MeV_fm3": s["pressure_total"] / model.hbarc**3,
            "binding_MeV": s["energy_total"] / n - model.MN,
            "trace_MeV4": trace,
            "trace_over_energy": trace / s["energy_total"],
            "pressure_over_energy": s["pressure_total"] / s["energy_total"],
            "field_log_response": -n * s["B_y"] / (y * s["C_y"]),
            "energy_log_slope": n * s["mu"] / s["energy_total"],
            "fractions": {
                "fermi": s["energy"] / s["energy_total"],
                "potential": s["U"] / s["energy_total"],
                "vector": s["vector"] / s["energy_total"],
            },
            "asymptotic_field_scale": y_as,
            "asymptotic_energy_scale": e_as,
            "field_ratio": y / y_as,
            "energy_ratio": s["energy_total"] / e_as,
        },
        "checks": {
            "stationarity_relative": abs(s["residual"]) / stationarity_scale,
            "legendre_relative": _relative_scale(legendre, s["energy_total"]),
            "trace_identity_relative": _relative_scale(trace_identity, s["energy_total"]),
            "positive_C": bool(s["C_y"] > 0),
            "positive_mu": bool(s["mu"] > 0),
            "finite": True,
        },
    }
    row["fd_checks"] = _fd_check(case, s, limit)
    return row


def _calibration(case_by_id):
    result = {}
    for target in ("0.90", "0.93"):
        target_y = _mp(target)
        entries = {}
        for cid in (_target_id("w8_", target), _target_id("u16_", target)):
            case = case_by_id[cid]
            model = case["model"]
            state = _state(model, model.n0, target_y)
            if isinstance(model, PotentialAdapter):
                delta = model.deformation_derivatives(target_y)
            else:
                delta = (mp.mpf(0),) * 4
            stationarity_scale = abs(model.MN * state["ns"]) + abs(state["Uy"])
            entries[cid] = {
                "target_y": target_y,
                "state": state,
                "delta_U_derivatives_0_to_3": delta,
                "stationarity_relative": abs(state["residual"]) / max(stationarity_scale, 1),
            }
        w8 = entries[_target_id("w8_", target)]
        u16 = entries[_target_id("u16_", target)]
        keys = ("energy_total", "pressure_total", "mu", "C_y", "K", "cs2")
        differences = {key: _relative(w8["state"][key], u16["state"][key]) for key in keys}
        result[target] = {
            "entries": entries,
            "w8_u16_relative_differences": differences,
            "jets_unchanged_through_order_3": all(value == 0 for value in u16["delta_U_derivatives_0_to_3"]),
            "positive_deformation_eta": u16["state"]["U"] >= w8["state"]["U"],
        }
    return result


def _serialize_state(state, model):
    fields = {
        "y": state["y"],
        "energy_density_MeV4": state["energy_total"],
        "pressure_MeV4": state["pressure_total"],
        "pressure_MeV_fm3": state["pressure_total"] / model.hbarc**3,
        "mu_MeV": state["mu"],
        "binding_MeV": state["energy_total"] / state["n"] - model.MN,
        "U_MeV4": state["U"],
        "Uy_MeV4": state["Uy"],
        "Uyy_MeV4": state["Uyy"],
        "fermi_energy_MeV4": state["energy"],
        "vector_MeV4": state["vector"],
        "C_y_MeV4": state["C_y"],
        "B_y": state["B_y"],
        "D_nn": state["D"],
        "mu_prime": state["mu_prime"],
        "K_MeV": state["K"],
        "cs2": state["cs2"],
        "trace_MeV4": state["energy_total"] - 3 * state["pressure_total"],
    }
    return {key: _number(value) for key, value in fields.items()}


def _serialize_row(row, model):
    state, root = row["state"], row["root"]
    derived = row["derived"]
    output = {
        "density_ratio": row["density_ratio"],
        "n_MeV3": _number(row["n_MeV3"]),
        "n_over_W0_cubed": _number(row["n_over_W0_cubed"]),
        "root": {
            "y": _number(root["y"]),
            "bracket_low": _number(root["bracket_low"]),
            "bracket_high": _number(root["bracket_high"]),
            "bracket_expansions": root["bracket_expansions"],
            "sign_change_bracket_count_lower_bound": root["sign_change_bracket_count_lower_bound"],
            "positive_root_bracketed": root["positive_root_bracketed"],
            "not_an_exhaustive_root_count": root["not_an_exhaustive_root_count"],
            "residual_scale": _number(root["residual_scale"]),
        },
        "state": _serialize_state(state, model),
        "raw_evidence": {
            "residual": _number(state["residual"]),
            "stationarity_scale": _number(root["residual_scale"]),
            "stationarity_terms": {
                "matter_derivative": _number(model.MN * state["ns"]),
                "potential_derivative": _number(state["Uy"]),
                "vector_derivative": _number(-model.Cv * state["n"] ** 2 / state["y"] ** 3),
            },
            "legendre_residual": _number(state["energy_total"] + state["pressure_total"] - state["n"] * state["mu"]),
            "trace_identity_residual": _number(
                state["energy_total"] - 3 * state["pressure_total"] - 4 * state["U"] + state["y"] * state["Uy"]
            ),
        },
        "derived": {
            "pressure_MeV_fm3": _number(derived["pressure_MeV_fm3"]),
            "binding_MeV": _number(derived["binding_MeV"]),
            "trace_MeV4": _number(derived["trace_MeV4"]),
            "trace_over_energy": _number(derived["trace_over_energy"]),
            "pressure_over_energy": _number(derived["pressure_over_energy"]),
            "field_log_response": _number(derived["field_log_response"]),
            "energy_log_slope": _number(derived["energy_log_slope"]),
            "fractions": {key: _number(value) for key, value in derived["fractions"].items()},
            "asymptotic_field_scale": _number(derived["asymptotic_field_scale"]),
            "asymptotic_energy_scale": _number(derived["asymptotic_energy_scale"]),
            "field_ratio": _number(derived["field_ratio"]),
            "energy_ratio": _number(derived["energy_ratio"]),
        },
        "checks": {
            "stationarity_relative": _number(row["checks"]["stationarity_relative"]),
            "legendre_relative": _number(row["checks"]["legendre_relative"]),
            "trace_identity_relative": _number(row["checks"]["trace_identity_relative"]),
            "positive_C": row["checks"]["positive_C"],
            "positive_mu": row["checks"]["positive_mu"],
            "finite": row["checks"]["finite"],
        },
        "fd_checks": {},
    }
    for htext, check in row["fd_checks"].items():
        output["fd_checks"][htext] = {
            key: _number(check[key]) for key in FD_EVIDENCE_KEYS
        }
        output["fd_checks"][htext].update({
            "mu_prime_fd": _number(check["mu_prime_fd"]),
            "mu_prime_analytic": _number(check["mu_prime_analytic"]),
            "mu_relative_error": _number(check["mu_relative_error"]),
            "field_log_response_fd": _number(check["field_log_response_fd"]),
            "field_log_response_analytic": _number(check["field_log_response_analytic"]),
            "field_relative_error": _number(check["field_relative_error"]),
            "pass": check["pass"],
        })
    return output


def _serialize_calibration(calibration, cases):
    out = {}
    for target, data in calibration.items():
        entries = {}
        for cid, entry in data["entries"].items():
            model = cases[cid]["model"]
            entries[cid] = {
                "target_y": _number(entry["target_y"]),
                "state": _serialize_state(entry["state"], model),
                "raw_residual": _number(entry["state"]["residual"]),
                "stationarity_relative": _number(entry["stationarity_relative"]),
                "delta_U_derivatives_0_to_3": [_number(value) for value in entry["delta_U_derivatives_0_to_3"]],
            }
        out[target] = {
            "entries": entries,
            "w8_u16_relative_differences": {
                key: _number(value) for key, value in data["w8_u16_relative_differences"].items()
            },
            "jets_unchanged_through_order_3": data["jets_unchanged_through_order_3"],
            "positive_deformation_eta": data["positive_deformation_eta"],
        }
    return out


def _serialize_snapshot(snapshot):
    cases = {}
    case_by_id = {case["id"]: case for case in snapshot["cases"]}
    for case in snapshot["cases"]:
        model = case["model"]
        limit = snapshot["limits"][case["id"]]
        metadata = {
            "design": case["design"],
            "tail_degree_in_y": case["tail_degree"],
            "tail_coefficient_MeV4": _number(case["tail_coefficient"]),
            "target_y": None if case["target_y"] is None else _number(case["target_y"]),
            "deformation_eta_MeV4": None if case["deformation_eta"] is None else _number(case["deformation_eta"]),
            "gomega": _number(model.gomega),
            "Cv_MeVminus2": _number(model.Cv),
            "limit": {
                key: (_number(value) if isinstance(value, mp.mpf) else value)
                for key, value in limit.items()
                if key not in ("z", "c", "energy_coefficient", "energy_prefactor", "curvature", "kinetic_free_z", "kinetic_free_energy_coefficient", "curvature_factor")
            },
        }
        for key in ("z", "c", "energy_coefficient", "energy_prefactor", "curvature", "kinetic_free_z", "kinetic_free_energy_coefficient", "curvature_factor"):
            if key in limit:
                metadata["limit"][key] = _number(limit[key])
        if case["id"] == "original_quartic":
            metadata["limit"]["full_fermi_is_retained"] = True
        final_row = snapshot["rows"][case["id"]][-1]
        final_expected = limit["expected_w"]
        final_w_error = abs(final_row["derived"]["pressure_over_energy"] - final_expected)
        final_cs_error = abs(final_row["state"]["cs2"] - limit["expected_cs2"])
        final_field_error = abs(final_row["derived"]["field_ratio"] - 1)
        final_energy_error = abs(final_row["derived"]["energy_ratio"] - 1)
        metadata["final_asymptotic_check"] = {
            "density_ratio": "1e36",
            "pressure_over_energy_absolute_error": _number(final_w_error),
            "cs2_absolute_error": _number(final_cs_error),
            "field_ratio_relative_error": _number(final_field_error),
            "energy_ratio_relative_error": _number(final_energy_error),
            "pass": bool(
                final_w_error <= ASYMPTOTIC_LIMIT
                and final_cs_error <= ASYMPTOTIC_LIMIT
                and final_field_error <= ASYMPTOTIC_FIELD_ENERGY_LIMIT
                and final_energy_error <= ASYMPTOTIC_FIELD_ENERGY_LIMIT
            ),
        }
        cases[case["id"]] = {
            **metadata,
            "rows": [_serialize_row(row, model) for row in snapshot["rows"][case["id"]]],
        }
    return cases, case_by_id


def _snapshot(dps, ratios):
    with mp.workdps(dps):
        cases = _make_cases()
        case_by_id = {case["id"]: case for case in cases}
        limits = {case["id"]: _limit_for(case) for case in cases}
        rows = {
            case["id"]: [_row(case, ratio, limits[case["id"]]) for ratio in ratios]
            for case in cases
        }
        calibration = _calibration(case_by_id)
        return {"dps": dps, "cases": cases, "limits": limits, "rows": rows, "calibration": calibration}


def _calibration_comparison(primary, control):
    comparisons = []
    for target in ("0.90", "0.93"):
        for cid in (_target_id("w8_", target), _target_id("u16_", target)):
            a = primary["calibration"][target]["entries"][cid]["state"]
            b = control["calibration"][target]["entries"][cid]["state"]
            fields = ("energy_total", "pressure_total", "mu", "C_y", "K", "cs2")
            differences = {field: _relative(a[field], b[field]) for field in fields}
            comparisons.append({
                "case_id": cid,
                "density_ratio": "1",
                "max_relative_difference": max(differences.values()),
                "fields": differences,
                "primary_values": {field: a[field] for field in fields},
                "control_values": {field: b[field] for field in fields},
            })
    return comparisons


def _precision_controls(primary, control):
    controls = []
    for case in primary["cases"]:
        cid = case["id"]
        primary_rows = {row["density_ratio"]: row for row in primary["rows"][cid]}
        control_rows = {row["density_ratio"]: row for row in control["rows"][cid]}
        for ratio in CONTROL_RATIOS:
            a, b = primary_rows[ratio]["state"], control_rows[ratio]["state"]
            fields = ("y", "energy_total", "pressure_total", "mu", "C_y", "K", "cs2")
            differences = {field: _relative(a[field], b[field]) for field in fields}
            controls.append({
                "case_id": cid,
                "density_ratio": ratio,
                "max_relative_difference": max(differences.values()),
                "fields": differences,
                "primary_values": {field: a[field] for field in fields},
                "control_values": {field: b[field] for field in fields},
            })
    controls.extend(_calibration_comparison(primary, control))
    return controls


def _serialize_precision(controls):
    return [
        {
            "case_id": item["case_id"],
            "density_ratio": item["density_ratio"],
            "max_relative_difference": _number(item["max_relative_difference"]),
            "fields": {key: _number(value) for key, value in item["fields"].items()},
            "primary_values": {key: _number(value) for key, value in item["primary_values"].items()},
            "control_values": {key: _number(value) for key, value in item["control_values"].items()},
            "pass": bool(item["max_relative_difference"] <= PRECISION_LIMIT),
        }
        for item in controls
    ]


def _model_checks(snapshot):
    checks = []
    for case in snapshot["cases"]:
        cid = case["id"]
        for row in snapshot["rows"][cid]:
            checks.append(row["checks"]["stationarity_relative"] <= STATIONARITY_LIMIT)
            checks.append(row["checks"]["legendre_relative"] <= IDENTITY_LIMIT)
            checks.append(row["checks"]["trace_identity_relative"] <= IDENTITY_LIMIT)
            checks.append(row["checks"]["positive_C"] and row["checks"]["positive_mu"])
            checks.extend(item["pass"] for item in row["fd_checks"].values())
    return all(checks)


def _deformation_summary(snapshot):
    result = {}
    for target in ("0.90", "0.93"):
        w8 = next(case for case in snapshot["cases"] if case["id"] == _target_id("w8_", target))
        u16 = next(case for case in snapshot["cases"] if case["id"] == _target_id("u16_", target))
        # Use the adapter's declared decimal object.  Re-parsing the same
        # text at a wider outer precision would create a tiny artificial
        # nonzero at the fourth-order zero.
        target_y = u16["target_y"]
        model = u16["model"]
        derivatives_vacuum = model.deformation_derivatives(mp.mpf(1))
        derivatives_target = model.deformation_derivatives(target_y)
        sample_points = (mp.mpf("0.5"), mp.mpf("1.1"), mp.mpf("2"))
        samples = [model.potential(point)[0] - w8["model"].potential(point)[0] for point in sample_points]
        result[target] = {
            "eta_MeV4": u16["deformation_eta"],
            "leading_degree_in_y": 16,
            "leading_coefficient_MeV4": u16["tail_coefficient"],
            "vacuum_delta_derivatives_0_to_3": derivatives_vacuum,
            "calibration_delta_derivatives_0_to_3": derivatives_target,
            "exact_zero_through_order_3": all(value == 0 for value in derivatives_vacuum + derivatives_target),
            "eta_positive": bool(u16["deformation_eta"] > 0),
            "delta_U_nonnegative_by_even_power_proof": bool(u16["deformation_eta"] > 0),
            "positive_sample_delta_U_MeV4": samples,
            "sample_points_y": sample_points,
        }
    return result


def _local_discriminator(snapshot):
    """First reduced-energy derivative that distinguishes the two branches."""
    result = {}
    for target in ("0.90", "0.93"):
        wid, uid = _target_id("w8_", target), _target_id("u16_", target)
        w8, u16 = next(case for case in snapshot["cases"] if case["id"] == wid), next(case for case in snapshot["cases"] if case["id"] == uid)
        y = w8["target_y"]
        state = _state(w8["model"], w8["model"].n0, y)
        # The fourth derivative of (y^2-1)^4(y^2-y*^2)^4 at y=y* is exact.
        delta_u4 = 384 * u16["deformation_eta"] * y**4 * (1 - y**2)**4
        dy_dlogn = -state["n"] * state["B_y"] / state["C_y"]
        delta_epsilon4 = delta_u4 * (dy_dlogn / state["n"])**4
        delta_z = 81 * delta_u4 * dy_dlogn**4 / state["n"]
        result[target] = {
            "dy_dlogn": dy_dlogn,
            "delta_U_fourth_derivative_MeV4": delta_u4,
            "delta_reduced_energy_fourth_density_derivative": delta_epsilon4,
            "delta_Z": delta_z,
            "symmetric_fourth_coefficient": delta_u4 * dy_dlogn**4 / 24,
            "positive_when_nonzero_slope": bool(delta_z > 0 and dy_dlogn != 0),
        }
    return result


def _serialize_deformation(summary):
    return {
        target: {
            "eta_MeV4": _number(item["eta_MeV4"]),
            "leading_degree_in_y": item["leading_degree_in_y"],
            "leading_coefficient_MeV4": _number(item["leading_coefficient_MeV4"]),
            "vacuum_delta_derivatives_0_to_3": [_number(value) for value in item["vacuum_delta_derivatives_0_to_3"]],
            "calibration_delta_derivatives_0_to_3": [_number(value) for value in item["calibration_delta_derivatives_0_to_3"]],
            "exact_zero_through_order_3": item["exact_zero_through_order_3"],
            "eta_positive": item["eta_positive"],
            "delta_U_nonnegative_by_even_power_proof": item["delta_U_nonnegative_by_even_power_proof"],
            "positive_sample_delta_U_MeV4": [_number(value) for value in item["positive_sample_delta_U_MeV4"]],
            "sample_points_y": [_number(value) for value in item["sample_points_y"]],
        }
        for target, item in summary.items()
    }


def _serialize_local_discriminator(summary):
    return {
        target: {
            key: (_number(value) if isinstance(value, mp.mpf) else value)
            for key, value in item.items()
        }
        for target, item in summary.items()
    }


def _serialize_limit_values(snapshot):
    result = {}
    for case in snapshot["cases"]:
        limit = snapshot["limits"][case["id"]]
        item = {}
        for key, value in limit.items():
            item[key] = _number(value) if isinstance(value, mp.mpf) else value
        result[case["id"]] = item
    return result


def _formula_controls():
    """Exact exponent-class controls, including the cold p=2 exception."""
    controls = []
    for p in (2, 4, 6, 8, 16):
        p_mp = mp.mpf(p)
        naive = (p_mp - 2) / (p_mp + 2)
        cold = mp.mpf(1) / 3 if p <= 4 else naive
        controls.append({
            "p": p,
            "cold_fermi_vector_w": cold,
            "cold_fermi_vector_cs2": cold,
            "naive_power_tail_ratio": naive,
            "fermion_retained_in_leading_balance": p <= 4,
            "power_tail_balance_applicable": p > 4,
            "p2_naive_ratio_rejected": p != 2 or cold != naive,
            "pass": True,
        })
    return controls


def _build_result():
    # Keep all mp operations in a high outer context while each snapshot uses
    # its declared working precision.  The two snapshots are independent
    # reconstructions of the equations and potentials.
    with mp.workdps(125):
        primary = _snapshot(70, DENSITY_RATIOS)
        control = _snapshot(110, CONTROL_RATIOS)
        precision_controls = _precision_controls(primary, control)
        if not _model_checks(primary):
            raise ArithmeticError("primary finite-density checks failed")
        if not all(item["max_relative_difference"] <= PRECISION_LIMIT for item in precision_controls):
            raise ArithmeticError("70/110 precision control failed")
        deformation = _deformation_summary(primary)
        local_discriminator = _local_discriminator(primary)
        source_hash = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
        upstream_hash = hashlib.sha256(UPSTREAM_PATH.read_bytes()).hexdigest()
        cases, _ = _serialize_snapshot(primary)
        calibration = _serialize_calibration(primary["calibration"], {case["id"]: case for case in primary["cases"]})
        # The p=4 kinetic-free comparison is deliberately part of the result;
        # it prevents a cold Fermi term from being silently discarded.
        p4_limit = primary["limits"]["original_quartic"]
        gravity = []
        for case in primary["cases"]:
            limit = primary["limits"][case["id"]]
            w = limit["expected_w"]
            kretschmann_factor = ((1 + 3 * w) ** 2 + 4) / 3
            gravity.append({
                "case_id": case["id"],
                "expected_w_limit": _number(w),
                "normalized_flat_flrw_kretschmann_factor": _number(kretschmann_factor),
            })
        return {
            "schema_version": SCHEMA,
            "status": "COMPUTED_FORMAL_TAIL_CLASSES_CALIBRATION_NULL_COUNTEREXAMPLE",
            "evidence_weight": EVIDENCE_WEIGHT,
            "source_sha256": source_hash,
            "upstream_source_sha256": upstream_hash,
            "units_and_inputs": {
                "state_variables": "y=W/W0 dimensionless; n in natural MeV^3; U and epsilon in MeV^4",
                "comparison_variables": "nbar=n/W0^3 and ebar=epsilon/W0^4",
                "W0_MeV": "859",
                "M_N_MeV": "939",
                "hbarc_MeV_fm": "197.3269804",
                "degeneracy": 4,
                "n0_fm3": "0.16",
                "binding_input_MeV": "-16",
                "K_input_MeV": "240",
                "density_ratios": list(DENSITY_RATIOS),
                "working_precision_digits": 70,
                "independent_control_precision_digits": 110,
            },
            "cases": cases,
            "calibration_point_comparisons": calibration,
            "deformation_certificate": _serialize_deformation(deformation),
            "local_fourth_density_discriminator": _serialize_local_discriminator(local_discriminator),
            "formula_controls": [
                {
                    key: (_number(value) if isinstance(value, mp.mpf) else value)
                    for key, value in item.items()
                }
                for item in _formula_controls()
            ],
            "precision_controls": _serialize_precision(precision_controls),
            "limit_controls": _serialize_limit_values(primary),
            "p4_full_fermi_vs_kinetic_free": {
                "full_fermi_z": _number(p4_limit["z"]),
                "kinetic_free_z": _number(p4_limit["kinetic_free_z"]),
                "relative_z_difference": _number(_relative(p4_limit["z"], p4_limit["kinetic_free_z"])),
                "full_fermi_energy_coefficient": _number(p4_limit["energy_coefficient"]),
                "kinetic_free_energy_coefficient": _number(p4_limit["kinetic_free_energy_coefficient"]),
                "conclusion": "kinetic-free substitution is a distinct limit, not the cold Fermi result",
            },
            "gravity_boundary": {
                "rho_plus_P_equals_n_mu": True,
                "mu_positive_on_all_reported_rows": True,
                "flat_GR_Hdot_negative_for_n_positive": True,
                "flat_GR_contraction_to_expansion_bounce_excluded": True,
                "curvature_factor_formula": "K/(8*pi*G_N)^2/epsilon^2=((1+3w)^2+4)/3",
                "cases": gravity,
                "interpretation": "trace/epsilon tending to zero does not imply a nonsingular flat-FLRW geometry",
            },
            "virial_connection": {
                "static_identity": "2 V_vector = p U in the p>4 tail balance",
                "oscillatory_identity": "2 <T> = p <U> for a rapidly oscillating canonical scalar",
                "shared_ratio": "(p-2)/(p+2) under the respective assumptions",
                "p2_counterexample": "cold homogeneous p=2 tail is Fermi/vector dominated with w=1/3; oscillatory quadratic scalar has w=0",
                "not_dynamical_equivalence": True,
            },
            "scope": {
                "formal_asymptotic_only": True,
                "finite_density_rows_are_stationary_candidates": True,
                "eft_validity_at_n_to_infinity_not_established": True,
                "not_empirical_validation": True,
                "not_full_eos_convexity_or_all_stationary_roots": True,
                "not_cosmological_tracking_or_frequency": True,
                "not_finite_nucleus_droplet_or_gravity_completion": True,
            },
        }


def build_result():
    """Build fresh numeric evidence; there is intentionally no result cache."""
    return _build_result()


def _mismatch(result, path, expected):
    value = result
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return True
        value = value[key]
    return value != expected


def _close(a, b, limit=NUMERIC_TIE_LIMIT):
    """Finite relative/absolute comparison for serialized decimal evidence."""
    if not (_finite(a) and _finite(b)):
        return False
    return _relative(a, b) <= limit


def _serialized_identity_close(reported, recomputed, operands):
    """Allow only the rounding budget implied by 60-digit serialization.

    Identities such as epsilon-3P-4U+yUy are cancellation-heavy at the last
    rows.  The producer's residual is evaluated before serialization, while
    ``recomputed`` uses 60-significant-digit fields.  A small explicit budget
    ties the two without making an arbitrary physical tolerance looser.
    """
    if not (_finite(reported) and _finite(recomputed)):
        return False
    scale = max(sum(abs(_mp(value)) for value in operands), mp.mpf(1))
    return abs(_mp(reported) - _mp(recomputed)) <= 100 * mp.mpf("1e-55") * scale


def _validation_state(model, n, y):
    """Evaluate the serialized point with a small independent state kernel."""
    n, y = _mp(n), _mp(y)
    f = model.fermi(n, y)
    u, uy, uyy = model.potential(y)
    vector = model.Cv * n**2 / (2 * y**2)
    energy = f["energy"] + u + vector
    pressure = f["pressure"] - u + vector
    mu = f["ef"] + model.Cv * n / y**2
    residual = model.MN * f["ns"] + uy - model.Cv * n**2 / y**3
    c_y = uyy + model.MN**2 * f["ns_m"] + 3 * model.Cv * n**2 / y**4
    b_y = model.MN**2 * y / f["ef"] - 2 * model.Cv * n / y**3
    d_nn = f["k"]**2 / (3 * n * f["ef"]) + model.Cv / y**2
    mu_prime = d_nn - b_y**2 / c_y if c_y else mp.nan
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
        D=d_nn,
        mu_prime=mu_prime,
        K=9 * n * mu_prime,
        cs2=n * mu_prime / mu,
    )


def _validation_state_mapping(state, model):
    return {
        "y": state["y"],
        "energy_density_MeV4": state["energy_total"],
        "pressure_MeV4": state["pressure_total"],
        "pressure_MeV_fm3": state["pressure_total"] / model.hbarc**3,
        "mu_MeV": state["mu"],
        "binding_MeV": state["energy_total"] / state["n"] - model.MN,
        "U_MeV4": state["U"],
        "Uy_MeV4": state["Uy"],
        "Uyy_MeV4": state["Uyy"],
        "fermi_energy_MeV4": state["energy"],
        "vector_MeV4": state["vector"],
        "C_y_MeV4": state["C_y"],
        "B_y": state["B_y"],
        "D_nn": state["D"],
        "mu_prime": state["mu_prime"],
        "K_MeV": state["K"],
        "cs2": state["cs2"],
        "trace_MeV4": state["energy_total"] - 3 * state["pressure_total"],
    }


def _validate_state_payload(payload, model, n, expected_y=None):
    """Check every serialized state field against the fixed model equations."""
    if not isinstance(payload, dict) or set(payload) != set(STATE_KEYS):
        return None
    try:
        if any(not _finite(value) for value in payload.values()):
            return None
        y = _mp(payload["y"])
        n = _mp(n)
        if not (n > 0 and y > 0):
            return None
        if expected_y is not None and not _close(y, expected_y):
            return None
        expected = _validation_state(model, n, y)
        mapping = _validation_state_mapping(expected, model)
        if any(not _close(payload[key], mapping[key]) for key in STATE_KEYS):
            return None
        return expected
    except Exception:
        return None


def _validation_limit(spec, limit_payload):
    """Reconstruct the declared asymptotic limits without a calibration solve."""
    if not isinstance(limit_payload, dict):
        return None
    try:
        model = spec["model"]
        p = spec["tail_degree"]
        a = spec["tail_coefficient"]
        if not (a > 0):
            return None
        if p == 4:
            z = _mp(limit_payload["z"])
            if not z > 0:
                return None
            n_unit = model.W0**3
            f = model.fermi(n_unit, z)
            residual = model.MN * f["ns"] + 4 * a * z**3 - model.Cv * n_unit**2 / z**3
            scale = abs(model.MN * f["ns"]) + abs(4 * a * z**3) + abs(model.Cv * n_unit**2 / z**3)
            if abs(residual) / max(scale, 1) > STATIONARITY_LIMIT:
                return None
            fermi_coefficient = f["energy"] / model.W0**4
            potential_coefficient = a * z**4 / model.W0**4
            vector_coefficient = model.Cv * n_unit**2 / (2 * z**2 * model.W0**4)
            energy_coefficient = fermi_coefficient + potential_coefficient + vector_coefficient
            curvature = 12 * a * z**2 + model.MN**2 * f["ns_m"] + 3 * model.Cv * n_unit**2 / z**4
            kinetic_free_z = (model.Cv * model.W0**6 / (4 * a)) ** (mp.mpf(1) / 6)
            kinetic_free_energy = a * kinetic_free_z**4 + model.Cv * n_unit**2 / (2 * kinetic_free_z**2)
            return {
                "kind": "full_fermi_p4_limit",
                "p": 4,
                "z": z,
                "energy_coefficient": energy_coefficient,
                "curvature": curvature,
                "kinetic_free_z": kinetic_free_z,
                "kinetic_free_energy_coefficient": kinetic_free_energy / model.W0**4,
                "expected_fermi_fraction": fermi_coefficient / energy_coefficient,
                "expected_potential_fraction": potential_coefficient / energy_coefficient,
                "expected_vector_fraction": vector_coefficient / energy_coefficient,
                "expected_trace_over_energy": mp.mpf(0),
                "expected_w": mp.mpf(1) / 3,
                "expected_cs2": mp.mpf(1) / 3,
                "nondegenerate": bool(curvature > 0),
                "fermion_subleading": False,
                "full_fermi_is_retained": True,
            }
        c = (model.Cv / (p * a)) ** (mp.mpf(1) / (p + 2))
        energy_prefactor = (1 + mp.mpf(p) / 2) * a * c**p
        return {
            "kind": "vector_potential_balance",
            "p": p,
            "c": c,
            "energy_prefactor": energy_prefactor,
            "curvature_factor": (p + 2) * model.Cv,
            "expected_w": (mp.mpf(p) - 2) / (p + 2),
            "expected_cs2": (mp.mpf(p) - 2) / (p + 2),
            "nondegenerate": True,
            "w_over_kf_exponent": (mp.mpf(4) - p) / (3 * (p + 2)),
            "fermi_over_energy_exponent": 2 * (mp.mpf(4) - p) / (3 * (p + 2)),
            "matter_derivative_over_vector_exponent": 4 * (mp.mpf(4) - p) / (3 * (p + 2)),
            "expected_fermi_fraction": mp.mpf(0),
            "expected_potential_fraction": mp.mpf(2) / (p + 2),
            "expected_vector_fraction": mp.mpf(p) / (p + 2),
            "expected_trace_over_energy": mp.mpf(2 * (4 - p)) / (p + 2),
            "fermion_subleading": True,
        }
    except Exception:
        return None


def _validate_limit(spec, case):
    limit = case.get("limit") if isinstance(case, dict) else None
    expected = _validation_limit(spec, limit)
    if expected is None or not isinstance(limit, dict) or set(limit) != set(expected):
        return None
    try:
        for key, value in expected.items():
            if isinstance(value, (str, bool, int)) and not isinstance(value, mp.mpf):
                if limit.get(key) != value:
                    return None
            elif not _close(limit.get(key), value):
                return None
        return expected
    except Exception:
        return None


def _validation_scales(spec, limit, n):
    model = spec["model"]
    n = _mp(n)
    if limit["p"] == 4:
        s = n / model.W0**3
        return limit["z"] * s ** (mp.mpf(1) / 3), limit["energy_coefficient"] * model.W0**4 * s ** (mp.mpf(4) / 3)
    return limit["c"] * n ** (mp.mpf(2) / (limit["p"] + 2)), limit["energy_prefactor"] * n ** (mp.mpf(2 * limit["p"]) / (limit["p"] + 2))


def _validation_root_evidence(model, n, guess):
    """Recompute bracket endpoints and finite sign-change coverage only."""
    guess = _mp(guess)
    if guess <= 0:
        raise ArithmeticError("validation root guess must be positive")
    lo, hi = guess / 10, guess * 10
    flo, fhi = _validation_state(model, n, lo)["residual"], _validation_state(model, n, hi)["residual"]
    expansions = 0
    for _ in range(80):
        if flo < 0:
            break
        lo /= 10
        flo = _validation_state(model, n, lo)["residual"]
        expansions += 1
    else:
        raise ArithmeticError("validation lower endpoint is not negative")
    for _ in range(80):
        if fhi > 0:
            break
        hi *= 10
        fhi = _validation_state(model, n, hi)["residual"]
        expansions += 1
    else:
        raise ArithmeticError("validation upper endpoint is not positive")
    scan_lo, scan_hi = lo / 1000, hi * 1000
    previous = _validation_state(model, n, scan_lo)["residual"]
    changes = 0
    for j in range(1, 65):
        point = scan_lo * (scan_hi / scan_lo) ** (mp.mpf(j) / 64)
        current = _validation_state(model, n, point)["residual"]
        if previous == 0 or current == 0 or previous * current < 0:
            changes += 1
        previous = current
    return {
        "bracket_low": lo,
        "bracket_high": hi,
        "bracket_expansions": expansions,
        "sign_change_bracket_count_lower_bound": changes,
    }


def _validate_numeric_row(row, spec, limit, expected_ratio):
    try:
        if not isinstance(row, dict) or set(row) != set(ROW_KEYS) or row.get("density_ratio") != expected_ratio:
            return False
        model = spec["model"]
        ratio = _mp(expected_ratio)
        n = _mp(row["n_MeV3"])
        expected_n = model.n0 * ratio
        if not _close(n, expected_n) or not _close(row["n_over_W0_cubed"], n / model.W0**3):
            return False
        if not (_finite(n) and n > 0):
            return False
        state_payload = row["state"]
        if not isinstance(state_payload, dict) or not _finite(state_payload.get("y")):
            return False
        y = _mp(state_payload["y"])
        expected_state = _validate_state_payload(state_payload, model, n)
        if expected_state is None:
            return False
        root = row["root"]
        if not isinstance(root, dict) or set(root) != set(ROOT_KEYS):
            return False
        if any(not _finite(root.get(key)) for key in ("y", "bracket_low", "bracket_high", "residual_scale")):
            return False
        if not (_mp(root["y"]) > 0 and _mp(root["bracket_low"]) > 0 and _mp(root["bracket_high"]) > _mp(root["bracket_low"]) and _mp(root["bracket_low"]) < _mp(root["y"]) < _mp(root["bracket_high"]) and _mp(root["residual_scale"]) > 0):
            return False
        if not isinstance(root["bracket_expansions"], int) or root["bracket_expansions"] < 0 or not isinstance(root["sign_change_bracket_count_lower_bound"], int) or root["sign_change_bracket_count_lower_bound"] < 1:
            return False
        if not _close(root["y"], y):
            return False
        y_as, _ = _validation_scales(spec, limit, n)
        expected_root = _validation_root_evidence(model, n, y_as)
        for key in ("bracket_low", "bracket_high"):
            if not _close(root[key], expected_root[key]):
                return False
        serialized_low_residual = _validation_state(model, n, _mp(root["bracket_low"]))["residual"]
        serialized_high_residual = _validation_state(model, n, _mp(root["bracket_high"]))["residual"]
        if not (serialized_low_residual < 0 < serialized_high_residual):
            return False
        if root["bracket_expansions"] != expected_root["bracket_expansions"] or root["sign_change_bracket_count_lower_bound"] != expected_root["sign_change_bracket_count_lower_bound"]:
            return False
        if root["positive_root_bracketed"] is not True or root["not_an_exhaustive_root_count"] is not True:
            return False

        raw = row["raw_evidence"]
        if not isinstance(raw, dict) or set(raw) != set(RAW_KEYS):
            return False
        terms = raw["stationarity_terms"]
        if not isinstance(terms, dict) or set(terms) != set(STATIONARITY_TERM_KEYS):
            return False
        expected_terms = {
            "matter_derivative": model.MN * expected_state["ns"],
            "potential_derivative": expected_state["Uy"],
            "vector_derivative": -model.Cv * n**2 / y**3,
        }
        if any(not _close(terms[key], expected_terms[key]) for key in STATIONARITY_TERM_KEYS):
            return False
        term_scale = sum(abs(expected_terms[key]) for key in STATIONARITY_TERM_KEYS)
        expected_residual = sum(expected_terms.values())
        expected_legendre_residual = expected_state["energy_total"] + expected_state["pressure_total"] - n * expected_state["mu"]
        expected_trace_residual = expected_state["energy_total"] - 3 * expected_state["pressure_total"] - 4 * expected_state["U"] + y * expected_state["Uy"]
        serialized_terms = [_mp(terms[key]) for key in STATIONARITY_TERM_KEYS]
        serialized_residual = _mp(raw["residual"])
        serialized_scale = _mp(raw["stationarity_scale"])
        if not _serialized_identity_close(serialized_residual, sum(serialized_terms), serialized_terms):
            return False
        if not _close(raw["stationarity_scale"], term_scale):
            return False
        if not _serialized_identity_close(raw["legendre_residual"], _mp(state_payload["energy_density_MeV4"]) + _mp(state_payload["pressure_MeV4"]) - n * _mp(state_payload["mu_MeV"]), (_mp(state_payload["energy_density_MeV4"]), _mp(state_payload["pressure_MeV4"]), n * _mp(state_payload["mu_MeV"]))):
            return False
        if not _serialized_identity_close(raw["trace_identity_residual"], _mp(state_payload["energy_density_MeV4"]) - 3 * _mp(state_payload["pressure_MeV4"]) - 4 * _mp(state_payload["U_MeV4"]) + _mp(state_payload["y"]) * _mp(state_payload["Uy_MeV4"]), (_mp(state_payload["energy_density_MeV4"]), 3 * _mp(state_payload["pressure_MeV4"]), 4 * _mp(state_payload["U_MeV4"]), _mp(state_payload["y"]) * _mp(state_payload["Uy_MeV4"]))):
            return False
        if not _close(root["residual_scale"], term_scale):
            return False

        energy = expected_state["energy_total"]
        pressure = expected_state["pressure_total"]
        mu = expected_state["mu"]
        stationarity = abs(serialized_residual) / max(abs(serialized_scale), 1)
        legendre = abs(_mp(raw["legendre_residual"])) / max(abs(_mp(state_payload["energy_density_MeV4"])), 1)
        trace_identity = abs(_mp(raw["trace_identity_residual"])) / max(abs(_mp(state_payload["energy_density_MeV4"])), 1)
        if abs(expected_residual) / max(abs(term_scale), 1) > STATIONARITY_LIMIT or abs(expected_legendre_residual) / max(abs(energy), 1) > IDENTITY_LIMIT or abs(expected_trace_residual) / max(abs(energy), 1) > IDENTITY_LIMIT:
            return False
        if stationarity > STATIONARITY_LIMIT or legendre > IDENTITY_LIMIT or trace_identity > IDENTITY_LIMIT:
            return False

        checks = row["checks"]
        if not isinstance(checks, dict) or set(checks) != set(CHECK_KEYS):
            return False
        if not _close(checks["stationarity_relative"], stationarity) or not _close(checks["legendre_relative"], legendre) or not _close(checks["trace_identity_relative"], trace_identity):
            return False
        if checks["positive_C"] is not bool(expected_state["C_y"] > 0) or checks["positive_mu"] is not bool(mu > 0) or checks["finite"] is not True:
            return False
        if not (mu > 0 and expected_state["C_y"] > 0):
            return False

        derived = row["derived"]
        if not isinstance(derived, dict) or set(derived) != set(DERIVED_KEYS):
            return False
        fractions = derived["fractions"]
        if not isinstance(fractions, dict) or set(fractions) != set(FRACTION_KEYS):
            return False
        expected_trace = energy - 3 * pressure
        expected_field = -n * expected_state["B_y"] / (y * expected_state["C_y"])
        expected_fractions = {
            "fermi": expected_state["energy"] / energy,
            "potential": expected_state["U"] / energy,
            "vector": expected_state["vector"] / energy,
        }
        expected_field_scale, expected_energy_scale = _validation_scales(spec, limit, n)
        expected_derived = {
            "pressure_MeV_fm3": pressure / model.hbarc**3,
            "binding_MeV": energy / n - model.MN,
            "trace_MeV4": expected_trace,
            "trace_over_energy": expected_trace / energy,
            "pressure_over_energy": pressure / energy,
            "field_log_response": expected_field,
            "energy_log_slope": n * mu / energy,
            "asymptotic_field_scale": expected_field_scale,
            "asymptotic_energy_scale": expected_energy_scale,
            "field_ratio": y / expected_field_scale,
            "energy_ratio": energy / expected_energy_scale,
        }
        if any(not _close(derived[key], expected_derived[key]) for key in expected_derived):
            return False
        if any(not _close(fractions[key], expected_fractions[key]) for key in FRACTION_KEYS):
            return False
        if _relative(sum(_mp(fractions[key]) for key in FRACTION_KEYS), 1) > mp.mpf("1e-20"):
            return False

        fd_checks = row["fd_checks"]
        if not isinstance(fd_checks, dict) or tuple(fd_checks) != FD_STEPS:
            return False
        for htext in FD_STEPS:
            check = fd_checks[htext]
            if not isinstance(check, dict) or set(check) != set(FD_KEYS):
                return False
            if any(not _finite(check[key]) for key in FD_EVIDENCE_KEYS + ("mu_prime_fd", "mu_prime_analytic", "mu_relative_error", "field_log_response_fd", "field_log_response_analytic", "field_relative_error")):
                return False
            h = _mp(htext)
            n_minus, n_plus = _mp(check["n_minus"]), _mp(check["n_plus"])
            y_minus, y_plus = _mp(check["y_minus"]), _mp(check["y_plus"])
            if not (_close(n_minus, n * mp.exp(-h)) and _close(n_plus, n * mp.exp(h)) and n_minus > 0 and n_plus > 0 and y_minus > 0 and y_plus > 0):
                return False
            sm, sp = _validation_state(model, n_minus, y_minus), _validation_state(model, n_plus, y_plus)
            for s in (sm, sp):
                residual_scale = abs(model.MN * s["ns"]) + abs(s["Uy"]) + abs(model.Cv * s["n"]**2 / s["y"]**3)
                if abs(s["residual"]) / max(residual_scale, 1) > STATIONARITY_LIMIT or s["C_y"] <= 0 or s["mu"] <= 0:
                    return False
            if not _close(check["mu_minus"], sm["mu"]) or not _close(check["mu_plus"], sp["mu"]):
                return False
            mu_fd = (sp["mu"] - sm["mu"]) / (2 * n * mp.sinh(h))
            field_fd = (mp.log(y_plus) - mp.log(y_minus)) / (2 * h)
            field_analytic = expected_field
            mu_error = _relative(mu_fd, expected_state["mu_prime"])
            field_error = _relative(field_fd, field_analytic)
            if not _close(check["mu_prime_fd"], mu_fd) or not _close(check["field_log_response_fd"], field_fd):
                return False
            if not _close(check["mu_prime_analytic"], expected_state["mu_prime"]) or not _close(check["field_log_response_analytic"], field_analytic):
                return False
            if _mp(check["mu_relative_error"]) < 0 or _mp(check["field_relative_error"]) < 0:
                return False
            if not _close(check["mu_relative_error"], mu_error) or not _close(check["field_relative_error"], field_error):
                return False
            expected_pass = bool(mu_error <= FD_LIMIT and field_error <= FD_LIMIT)
            if check["pass"] is not expected_pass or check["pass"] is not True:
                return False

        return True
    except Exception:
        return False


def _validate_case_metadata(case, spec):
    if not isinstance(case, dict):
        return False
    expected_keys = {"design", "tail_degree_in_y", "tail_coefficient_MeV4", "target_y", "deformation_eta_MeV4", "gomega", "Cv_MeVminus2", "limit", "final_asymptotic_check", "rows"}
    if set(case) != expected_keys:
        return False
    try:
        if case["design"] != spec["design"] or case["tail_degree_in_y"] != spec["tail_degree"]:
            return False
        if spec["target_y"] is None:
            if case["target_y"] is not None:
                return False
        elif not _close(case["target_y"], spec["target_y"]):
            return False
        if spec["deformation_eta"] is None:
            if case["deformation_eta_MeV4"] is not None:
                return False
        elif not _close(case["deformation_eta_MeV4"], spec["deformation_eta"]):
            return False
        model = spec["model"]
        return _close(case["tail_coefficient_MeV4"], spec["tail_coefficient"]) and _close(case["gomega"], model.gomega) and _close(case["Cv_MeVminus2"], model.Cv)
    except Exception:
        return False


def _validate_final_asymptotic_check(case, limit):
    final = case.get("final_asymptotic_check") if isinstance(case, dict) else None
    rows = case.get("rows") if isinstance(case, dict) else None
    expected_keys = {"density_ratio", "pressure_over_energy_absolute_error", "cs2_absolute_error", "field_ratio_relative_error", "energy_ratio_relative_error", "pass"}
    if not isinstance(final, dict) or set(final) != expected_keys or not isinstance(rows, list) or not rows:
        return False
    try:
        if final["density_ratio"] != "1e36":
            return False
        last = rows[-1]
        if not isinstance(last, dict) or not isinstance(last.get("derived"), dict):
            return False
        derived = last["derived"]
        errors = {
            "pressure_over_energy_absolute_error": abs(_mp(derived["pressure_over_energy"]) - limit["expected_w"]),
            "cs2_absolute_error": abs(_mp(last["state"]["cs2"]) - limit["expected_cs2"]),
            "field_ratio_relative_error": abs(_mp(derived["field_ratio"]) - 1),
            "energy_ratio_relative_error": abs(_mp(derived["energy_ratio"]) - 1),
        }
        for key, expected in errors.items():
            if not _finite(final.get(key)) or _mp(final[key]) < 0 or not _close(final[key], expected):
                return False
        expected_pass = bool(
            errors["pressure_over_energy_absolute_error"] <= ASYMPTOTIC_LIMIT
            and errors["cs2_absolute_error"] <= ASYMPTOTIC_LIMIT
            and errors["field_ratio_relative_error"] <= ASYMPTOTIC_FIELD_ENERGY_LIMIT
            and errors["energy_ratio_relative_error"] <= ASYMPTOTIC_FIELD_ENERGY_LIMIT
        )
        return final["pass"] is expected_pass and final["pass"] is True
    except Exception:
        return False


def _validate_calibration(result, specs):
    calibration = result.get("calibration_point_comparisons")
    if not isinstance(calibration, dict) or tuple(calibration) != VALIDATION_TARGETS:
        return False
    item_keys = {"entries", "w8_u16_relative_differences", "jets_unchanged_through_order_3", "positive_deformation_eta"}
    entry_keys = {"target_y", "state", "raw_residual", "stationarity_relative", "delta_U_derivatives_0_to_3"}
    field_map = {
        "energy_total": "energy_density_MeV4", "pressure_total": "pressure_MeV4", "mu": "mu_MeV",
        "C_y": "C_y_MeV4", "K": "K_MeV", "cs2": "cs2",
    }
    for target in VALIDATION_TARGETS:
        try:
            item = calibration[target]
            if not isinstance(item, dict) or set(item) != item_keys:
                return False
            expected_ids = {_target_id("w8_", target), _target_id("u16_", target)}
            entries = item["entries"]
            if not isinstance(entries, dict) or set(entries) != expected_ids:
                return False
            target_y = _mp(target)
            expected_delta = {}
            for cid in expected_ids:
                entry = entries[cid]
                if not isinstance(entry, dict) or set(entry) != entry_keys:
                    return False
                spec = specs[cid]
                model = spec["model"]
                expected_state = _validate_state_payload(entry["state"], model, model.n0, target_y)
                if expected_state is None or not _close(entry["target_y"], target_y):
                    return False
                if not _close(entry["raw_residual"], expected_state["residual"]):
                    return False
                expected_terms_scale = abs(model.MN * expected_state["ns"]) + abs(expected_state["Uy"])
                expected_stationarity = abs(expected_state["residual"]) / max(expected_terms_scale, 1)
                if not _close(entry["stationarity_relative"], expected_stationarity) or expected_stationarity > STATIONARITY_LIMIT:
                    return False
                if cid.startswith("u16_"):
                    expected_delta[cid] = tuple(model.deformation_derivatives(target_y))
                else:
                    expected_delta[cid] = (mp.mpf(0),) * 4
                delta = entry["delta_U_derivatives_0_to_3"]
                if not isinstance(delta, list) or len(delta) != 4 or any(not _finite(value) or _mp(value) != 0 for value in delta):
                    return False
                if any(not _close(delta[index], expected_delta[cid][index]) for index in range(4)):
                    return False
                state = entry["state"]
                if abs(_mp(state["pressure_MeV4"])) > IDENTITY_LIMIT * max(abs(_mp(state["energy_density_MeV4"])), 1):
                    return False
                if abs(_mp(state["binding_MeV"]) + 16) > mp.mpf("1e-20") or abs(_mp(state["K_MeV"]) - 240) > mp.mpf("1e-15"):
                    return False
                if not (_mp(state["y"]) > 0 and _mp(state["mu_MeV"]) > 0 and _mp(state["C_y_MeV4"]) > 0):
                    return False
            differences = item["w8_u16_relative_differences"]
            if not isinstance(differences, dict) or set(differences) != set(PRECISION_CALIBRATION_FIELDS):
                return False
            left = entries[_target_id("w8_", target)]["state"]
            right = entries[_target_id("u16_", target)]["state"]
            for field in PRECISION_CALIBRATION_FIELDS:
                expected_difference = _relative(left[field_map[field]], right[field_map[field]])
                if not _finite(differences[field]) or _mp(differences[field]) < 0 or not _close(differences[field], expected_difference) or _mp(differences[field]) > PRECISION_LIMIT:
                    return False
            expected_jets = all(value == 0 for value in expected_delta[_target_id("u16_", target)])
            expected_positive = bool(
                    _validation_state(specs[_target_id("u16_", target)]["model"], specs[_target_id("u16_", target)]["model"].n0, target_y)["U"]
                >= _validation_state(specs[_target_id("w8_", target)]["model"], specs[_target_id("w8_", target)]["model"].n0, target_y)["U"]
            )
            if item["jets_unchanged_through_order_3"] is not expected_jets or item["positive_deformation_eta"] is not expected_positive:
                return False
        except Exception:
            return False
    return True


def _precision_state_values(state, fields):
    mapping = {
        "y": "y", "energy_total": "energy_density_MeV4", "pressure_total": "pressure_MeV4",
        "mu": "mu_MeV", "C_y": "C_y_MeV4", "K": "K_MeV", "cs2": "cs2",
    }
    return {field: state[mapping[field]] for field in fields}


def _validate_precision_controls(result):
    controls = result.get("precision_controls")
    if not isinstance(controls, list) or len(controls) != len(PRECISION_KEY_ORDER):
        return False
    cases = result["cases"]
    calibration = result["calibration_point_comparisons"]
    for item, expected_key in zip(controls, PRECISION_KEY_ORDER):
        try:
            cid, ratio = expected_key
            expected_fields = PRECISION_CALIBRATION_FIELDS if ratio == "1" else PRECISION_ROW_FIELDS
            expected_item_keys = {"case_id", "density_ratio", "max_relative_difference", "fields", "primary_values", "control_values", "pass"}
            if not isinstance(item, dict) or set(item) != expected_item_keys or item["case_id"] != cid or item["density_ratio"] != ratio:
                return False
            primary_values, control_values = item["primary_values"], item["control_values"]
            if not isinstance(primary_values, dict) or not isinstance(control_values, dict) or set(primary_values) != set(expected_fields) or set(control_values) != set(expected_fields):
                return False
            if ratio == "1":
                target = "0.90" if cid.endswith("90") else "0.93"
                primary_state = calibration[target]["entries"][cid]["state"]
            else:
                row_index = list(DENSITY_RATIOS).index(ratio)
                primary_state = cases[cid]["rows"][row_index]["state"]
            expected_primary = _precision_state_values(primary_state, expected_fields)
            if any(not _close(primary_values[field], expected_primary[field]) for field in expected_fields):
                return False
            if any(not _finite(control_values[field]) for field in expected_fields):
                return False
            fields = item["fields"]
            if not isinstance(fields, dict) or set(fields) != set(expected_fields):
                return False
            expected_differences = {field: _relative(primary_values[field], control_values[field]) for field in expected_fields}
            for field in expected_fields:
                if not _finite(fields[field]) or _mp(fields[field]) < 0 or not _close(fields[field], expected_differences[field]) or _mp(fields[field]) > PRECISION_LIMIT:
                    return False
            expected_max = max(expected_differences.values())
            if not _finite(item["max_relative_difference"]) or _mp(item["max_relative_difference"]) < 0 or not _close(item["max_relative_difference"], expected_max) or _mp(item["max_relative_difference"]) > PRECISION_LIMIT:
                return False
            expected_pass = bool(expected_max <= PRECISION_LIMIT)
            if item["pass"] is not expected_pass or item["pass"] is not True:
                return False
        except Exception:
            return False
    return True


def _validation_precision(function):
    def wrapped(result):
        # Serialized strings carry 60 significant digits.  Parsing them at
        # the caller's default (~15 digits) would manufacture false identity
        # failures through cancellation in epsilon+P-n*mu.
        with mp.workdps(100):
            return function(result)

    return wrapped



@_validation_precision
def validate_result(result):
    """Validate serialized evidence against fixed equations and coverage.

    This path intentionally does not call ``build_result`` or any root/matrix
    calibration routine.  It reconstructs the five fixed model definitions,
    evaluates the equations at serialized states, and checks the finite
    difference/control evidence supplied by the producer.
    """
    try:
        top_keys = {
            "schema_version", "status", "evidence_weight", "source_sha256", "upstream_source_sha256",
            "units_and_inputs", "cases", "calibration_point_comparisons", "deformation_certificate",
            "local_fourth_density_discriminator", "formula_controls", "precision_controls", "limit_controls",
            "p4_full_fermi_vs_kinetic_free", "gravity_boundary", "virial_connection", "scope",
        }
        if not isinstance(result, dict) or set(result) != top_keys:
            return False
        if result.get("schema_version") != SCHEMA or result.get("status") != "COMPUTED_FORMAL_TAIL_CLASSES_CALIBRATION_NULL_COUNTEREXAMPLE" or result.get("evidence_weight") != 0.0:
            return False
        if result.get("source_sha256") != hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest() or result.get("upstream_source_sha256") != hashlib.sha256(UPSTREAM_PATH.read_bytes()).hexdigest():
            return False
        expected_units = {
            "state_variables": "y=W/W0 dimensionless; n in natural MeV^3; U and epsilon in MeV^4",
            "comparison_variables": "nbar=n/W0^3 and ebar=epsilon/W0^4",
            "W0_MeV": "859", "M_N_MeV": "939", "hbarc_MeV_fm": "197.3269804",
            "degeneracy": 4, "n0_fm3": "0.16", "binding_input_MeV": "-16", "K_input_MeV": "240",
            "density_ratios": list(DENSITY_RATIOS), "working_precision_digits": 70,
            "independent_control_precision_digits": 110,
        }
        units = result.get("units_and_inputs")
        if not isinstance(units, dict) or set(units) != set(expected_units) or any(units.get(key) != value for key, value in expected_units.items()):
            return False

        specs_list = _validation_cases()
        specs = {spec["id"]: spec for spec in specs_list}
        base_model = specs["original_quartic"]["model"]
        if not (_close(base_model.W0, units["W0_MeV"]) and _close(base_model.MN, units["M_N_MeV"]) and _close(base_model.hbarc, units["hbarc_MeV_fm"]) and _close(base_model.n0_fm3, units["n0_fm3"]) and base_model.d == units["degeneracy"]):
            return False
        cases = result.get("cases")
        if not isinstance(cases, dict) or tuple(cases) != CASE_ORDER or set(cases) != set(CASE_ORDER):
            return False
        limits = {}
        for cid in CASE_ORDER:
            spec = specs[cid]
            case = cases[cid]
            if not _validate_case_metadata(case, spec):
                return False
            limit = _validate_limit(spec, case)
            if limit is None:
                return False
            limits[cid] = limit
            rows = case.get("rows")
            if not isinstance(rows, list) or len(rows) != len(DENSITY_RATIOS) or [row.get("density_ratio") if isinstance(row, dict) else None for row in rows] != list(DENSITY_RATIOS):
                return False
            for row, ratio in zip(rows, DENSITY_RATIOS):
                if not _validate_numeric_row(row, spec, limit, ratio):
                    return False
            if not _validate_final_asymptotic_check(case, limit):
                return False

        limit_controls = result.get("limit_controls")
        if not isinstance(limit_controls, dict) or tuple(limit_controls) != CASE_ORDER or set(limit_controls) != set(CASE_ORDER):
            return False
        for cid in CASE_ORDER:
            declared, control = cases[cid]["limit"], limit_controls[cid]
            declared_control = {key: value for key, value in declared.items() if key != "full_fermi_is_retained"}
            if not isinstance(control, dict) or set(control) != set(declared_control):
                return False
            for key in declared_control:
                value = declared_control[key]
                if isinstance(value, (str, bool, int)) and not isinstance(value, mp.mpf):
                    if control[key] != value:
                        return False
                elif not _close(control[key], value):
                    return False

        if not _validate_calibration(result, specs) or not _validate_precision_controls(result):
            return False

        deformation = result.get("deformation_certificate")
        deformation_keys = {
            "eta_MeV4", "leading_degree_in_y", "leading_coefficient_MeV4", "vacuum_delta_derivatives_0_to_3",
            "calibration_delta_derivatives_0_to_3", "exact_zero_through_order_3", "eta_positive",
            "delta_U_nonnegative_by_even_power_proof", "positive_sample_delta_U_MeV4", "sample_points_y",
        }
        if not isinstance(deformation, dict) or tuple(deformation) != VALIDATION_TARGETS:
            return False
        for target in VALIDATION_TARGETS:
            item = deformation[target]
            uid, wid = _target_id("u16_", target), _target_id("w8_", target)
            spec = specs[uid]
            if not isinstance(item, dict) or set(item) != deformation_keys:
                return False
            if item["leading_degree_in_y"] != 16 or not _close(item["eta_MeV4"], spec["deformation_eta"]) or not _close(item["leading_coefficient_MeV4"], spec["deformation_eta"]):
                return False
            if item["sample_points_y"] != ["0.5", "1.1", "2.0"]:
                return False
            expected_vacuum = tuple(spec["model"].deformation_derivatives(mp.mpf(1)))
            expected_target = tuple(spec["model"].deformation_derivatives(_mp(target)))
            for key, expected_values in (("vacuum_delta_derivatives_0_to_3", expected_vacuum), ("calibration_delta_derivatives_0_to_3", expected_target)):
                values = item[key]
                if not isinstance(values, list) or len(values) != 4 or any(not _finite(value) or not _close(value, expected_values[index]) for index, value in enumerate(values)):
                    return False
                if any(_mp(value) != 0 for value in values):
                    return False
            sample_points = [mp.mpf(value) for value in item["sample_points_y"]]
            expected_samples = [spec["model"].potential(point)[0] - specs[wid]["model"].potential(point)[0] for point in sample_points]
            samples = item["positive_sample_delta_U_MeV4"]
            if not isinstance(samples, list) or len(samples) != len(expected_samples) or any(not _finite(value) or _mp(value) <= 0 or not _close(value, expected_samples[index]) for index, value in enumerate(samples)):
                return False
            if item["eta_positive"] is not bool(spec["deformation_eta"] > 0) or item["delta_U_nonnegative_by_even_power_proof"] is not bool(spec["deformation_eta"] > 0) or item["exact_zero_through_order_3"] is not True:
                return False

        discriminator = result.get("local_fourth_density_discriminator")
        discriminator_keys = {"dy_dlogn", "delta_U_fourth_derivative_MeV4", "delta_reduced_energy_fourth_density_derivative", "delta_Z", "symmetric_fourth_coefficient", "positive_when_nonzero_slope"}
        if not isinstance(discriminator, dict) or tuple(discriminator) != VALIDATION_TARGETS:
            return False
        for target in VALIDATION_TARGETS:
            item = discriminator[target]
            if not isinstance(item, dict) or set(item) != discriminator_keys:
                return False
            wid, uid = _target_id("w8_", target), _target_id("u16_", target)
            w8_spec, u16_spec = specs[wid], specs[uid]
            y = _mp(target)
            state = _validation_state(w8_spec["model"], w8_spec["model"].n0, y)
            delta_u4 = 384 * u16_spec["deformation_eta"] * y**4 * (1 - y**2)**4
            dy_dlogn = -state["n"] * state["B_y"] / state["C_y"]
            expected_discriminator = {
                "dy_dlogn": dy_dlogn,
                "delta_U_fourth_derivative_MeV4": delta_u4,
                "delta_reduced_energy_fourth_density_derivative": delta_u4 * (dy_dlogn / state["n"])**4,
                "delta_Z": 81 * delta_u4 * dy_dlogn**4 / state["n"],
                "symmetric_fourth_coefficient": delta_u4 * dy_dlogn**4 / 24,
            }
            if any(not _finite(item.get(key)) or not _close(item[key], value) for key, value in expected_discriminator.items()):
                return False
            if item["positive_when_nonzero_slope"] is not bool(expected_discriminator["delta_Z"] > 0 and dy_dlogn != 0) or _mp(item["delta_Z"]) <= 0 or _mp(item["symmetric_fourth_coefficient"]) <= 0:
                return False

        formula_controls = result.get("formula_controls")
        formula_keys = {"p", "cold_fermi_vector_w", "cold_fermi_vector_cs2", "naive_power_tail_ratio", "fermion_retained_in_leading_balance", "power_tail_balance_applicable", "p2_naive_ratio_rejected", "pass"}
        if not isinstance(formula_controls, list) or len(formula_controls) != 5 or [item.get("p") if isinstance(item, dict) else None for item in formula_controls] != [2, 4, 6, 8, 16]:
            return False
        for item in formula_controls:
            if not isinstance(item, dict) or set(item) != formula_keys:
                return False
            p = item["p"]
            if not isinstance(p, int) or p not in (2, 4, 6, 8, 16):
                return False
            naive = (mp.mpf(p) - 2) / (mp.mpf(p) + 2)
            cold = mp.mpf(1) / 3 if p <= 4 else naive
            if not _finite(item["cold_fermi_vector_w"]) or not _finite(item["cold_fermi_vector_cs2"]) or not _finite(item["naive_power_tail_ratio"]):
                return False
            if not _close(item["cold_fermi_vector_w"], cold, mp.mpf("1e-30")) or not _close(item["cold_fermi_vector_cs2"], cold, mp.mpf("1e-30")) or not _close(item["naive_power_tail_ratio"], naive, mp.mpf("1e-30")):
                return False
            if item["fermion_retained_in_leading_balance"] is not (p <= 4) or item["power_tail_balance_applicable"] is not (p > 4) or item["p2_naive_ratio_rejected"] is not (p != 2 or cold != naive) or item["pass"] is not True:
                return False

        p4 = result.get("p4_full_fermi_vs_kinetic_free")
        p4_keys = {"full_fermi_z", "kinetic_free_z", "relative_z_difference", "full_fermi_energy_coefficient", "kinetic_free_energy_coefficient", "conclusion"}
        if not isinstance(p4, dict) or set(p4) != p4_keys:
            return False
        p4_limit = limits["original_quartic"]
        expected_p4 = {
            "full_fermi_z": p4_limit["z"],
            "kinetic_free_z": p4_limit["kinetic_free_z"],
            "relative_z_difference": _relative(p4_limit["z"], p4_limit["kinetic_free_z"]),
            "full_fermi_energy_coefficient": p4_limit["energy_coefficient"],
            "kinetic_free_energy_coefficient": p4_limit["kinetic_free_energy_coefficient"],
        }
        if any(not _finite(p4.get(key)) or not _close(p4[key], value) for key, value in expected_p4.items()) or _mp(p4["relative_z_difference"]) <= mp.mpf("1e-3") or p4["conclusion"] != "kinetic-free substitution is a distinct limit, not the cold Fermi result":
            return False

        gravity = result.get("gravity_boundary")
        gravity_keys = {"rho_plus_P_equals_n_mu", "mu_positive_on_all_reported_rows", "flat_GR_Hdot_negative_for_n_positive", "flat_GR_contraction_to_expansion_bounce_excluded", "curvature_factor_formula", "cases", "interpretation"}
        if not isinstance(gravity, dict) or set(gravity) != gravity_keys:
            return False
        if gravity["rho_plus_P_equals_n_mu"] is not True or gravity["flat_GR_Hdot_negative_for_n_positive"] is not True or gravity["flat_GR_contraction_to_expansion_bounce_excluded"] is not True or gravity["curvature_factor_formula"] != "K/(8*pi*G_N)^2/epsilon^2=((1+3w)^2+4)/3" or gravity["interpretation"] != "trace/epsilon tending to zero does not imply a nonsingular flat-FLRW geometry":
            return False
        if gravity["mu_positive_on_all_reported_rows"] is not all(_mp(row["state"]["mu_MeV"]) > 0 for cid in CASE_ORDER for row in cases[cid]["rows"]):
            return False
        gravity_cases = gravity["cases"]
        if not isinstance(gravity_cases, list) or len(gravity_cases) != len(CASE_ORDER):
            return False
        for item, cid in zip(gravity_cases, CASE_ORDER):
            if not isinstance(item, dict) or set(item) != {"case_id", "expected_w_limit", "normalized_flat_flrw_kretschmann_factor"} or item["case_id"] != cid:
                return False
            w = limits[cid]["expected_w"]
            factor = ((1 + 3 * w) ** 2 + 4) / 3
            if not _close(item["expected_w_limit"], w) or not _finite(item["normalized_flat_flrw_kretschmann_factor"]) or _mp(item["normalized_flat_flrw_kretschmann_factor"]) < 0 or not _close(item["normalized_flat_flrw_kretschmann_factor"], factor):
                return False

        expected_virial = {
            "static_identity": "2 V_vector = p U in the p>4 tail balance",
            "oscillatory_identity": "2 <T> = p <U> for a rapidly oscillating canonical scalar",
            "shared_ratio": "(p-2)/(p+2) under the respective assumptions",
            "p2_counterexample": "cold homogeneous p=2 tail is Fermi/vector dominated with w=1/3; oscillatory quadratic scalar has w=0",
            "not_dynamical_equivalence": True,
        }
        expected_scope = {
            "formal_asymptotic_only": True, "finite_density_rows_are_stationary_candidates": True,
            "eft_validity_at_n_to_infinity_not_established": True, "not_empirical_validation": True,
            "not_full_eos_convexity_or_all_stationary_roots": True, "not_cosmological_tracking_or_frequency": True,
            "not_finite_nucleus_droplet_or_gravity_completion": True,
        }
        if result.get("virial_connection") != expected_virial or result.get("scope") != expected_scope:
            return False
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = build_result()
        if not validate_result(result):
            raise ArithmeticError("fresh result failed serialized fail-closed validation")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ArithmeticError, TypeError, ValueError, KeyError) as exc:
        # Error output remains strict finite JSON and no file is touched.
        print(json.dumps({"schema_version": SCHEMA, "status": "ERROR", "error": str(exc), "evidence_weight": 0.0}, ensure_ascii=False, allow_nan=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
