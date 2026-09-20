#!/usr/bin/env python3
"""Live finite-temperature two-component composition Hessian.

This is a deliberately bounded thermodynamic bridge.  It rebuilds the already
accepted W8/U16 calibration and finite-temperature integrals in the current
process, splits the maintained Dirac degeneracy ``d=4`` into two
spin-degenerate ``d=2`` species, and solves the canonical neutron/proton
composition at eight live thermal reference densities.  The frozen isovector
coefficient is the historical ``j`` transfer, not a fitted parameter or a new
force.

The result is a local homogeneous Hessian certificate only.  It is not a
global phase-equilibrium, charge/electron, isentropic-causality, TOV, cluster,
detector, or experimental calculation.  Saved JSON is an output snapshot and
is never used as a scientific input.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# These are maintained, pre-existing live producers.  The composition bridge
# intentionally does not import either P1 master-response product.
import nvg_nonlinear_calibration_response as nonlinear  # noqa: E402
import nvg_thermal_observable_bridge as thermal  # noqa: E402


SCHEMA_VERSION = "nvg_thermal_composition_hessian.v1"
STATUS_PASS = "PASS_LIVE_THERMAL_COMPOSITION_HESSIAN"
STATUS_FAIL = "FAIL_LIVE_THERMAL_COMPOSITION_HESSIAN"
EVIDENCE_WEIGHT = "0.0"

ANCHORS = ("0.90", "0.93")
TEMPERATURES = ("70",)
REFERENCE_CHEMICAL_POTENTIALS = ("775", "875")
MODEL_ORDER = ("w8", "u16")
DELTAS = ("0", "0.2")
CONTACT_ORDER = ("zero", "fixed_j")
MODEL_AMPLITUDES = {"w8": 0, "u16": 1}

# Frozen, conditional historical input.  It is deliberately not reconstructed
# from an output row and is never optimized by this producer.
J_MEV_TEXT = "11.13601607117819"
N0_FM3_TEXT = "0.16"
HBARC_MEV_FM_TEXT = "197.3269804"
with mp.workdps(80):
    J_MEV = mp.mpf(J_MEV_TEXT)
    N0_FM3 = mp.mpf(N0_FM3_TEXT)
    HBARC_MEV_FM = mp.mpf(HBARC_MEV_FM_TEXT)
    C_RHO_MEV_FM3 = 2 * J_MEV / N0_FM3
    C_RHO_NATURAL = C_RHO_MEV_FM3 / HBARC_MEV_FM**3

PRIMARY_DPS = 44
CONTROL_DPS = 52
PRIMARY_NQUAD = 80
CONTROL_NQUAD = 112
PRIMARY_CUTOFF_MEV = mp.mpf("4500")
CONTROL_CUTOFF_MEV = mp.mpf("5200")
FD_REL_STEPS = (mp.mpf("1e-3"), mp.mpf("5e-4"), mp.mpf("2.5e-4"))
ROOT_REL_LIMIT = mp.mpf("2e-28")
MAXWELL_REL_LIMIT = mp.mpf("2e-16")
FD_REL_LIMIT = mp.mpf("3e-6")
REFINEMENT_REL_LIMIT = mp.mpf("3e-8")
ROOT_Y_MIN = mp.mpf("0.10")
ROOT_Y_MAX = mp.mpf("3.0")

EXPECTED_REFERENCE_COUNT = 8
EXPECTED_COMPOSITION_COUNT = 16
EXPECTED_ROW_COUNT = 32


class CompositionBridgeError(ValueError):
    """Invalid input, failed local root, or failed numerical control."""


def _mp(value: Any) -> mp.mpf:
    if isinstance(value, mp.mpf):
        out = value
    elif isinstance(value, bool):
        raise CompositionBridgeError("boolean is not a finite scalar")
    else:
        try:
            out = mp.mpf(str(value))
        except (TypeError, ValueError) as exc:
            raise CompositionBridgeError("value must be a finite real scalar") from exc
    if not mp.isfinite(out):
        raise CompositionBridgeError("value must be finite")
    return out


def _number(value: Any, digits: int = 40) -> str:
    value = _mp(value)
    return mp.nstr(value, digits)


def _mu_key(value: Any) -> str:
    """Stable internal key for a reference chemical potential."""
    return _number(value, 30)


def _relative(a: Any, b: Any, *, floor: Any = "1e-45") -> mp.mpf:
    aa, bb = _mp(a), _mp(b)
    return abs(aa - bb) / max(abs(aa), abs(bb), _mp(floor))


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(key) and _finite_tree(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, str) or value is None or isinstance(value, bool):
        return True
    try:
        return bool(mp.isfinite(_mp(value)))
    except (TypeError, ValueError, CompositionBridgeError):
        return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, mp.mpf):
        return _number(value)
    if isinstance(value, mp.mpc):
        raise CompositionBridgeError("complex scientific value cannot be serialized")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _matrix_relative(a: Sequence[Sequence[Any]], b: Sequence[Sequence[Any]]) -> mp.mpf:
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return mp.inf
    return max((_relative(x, y, floor="1e-35") for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b)), default=mp.mpf(0))


def _matrix_inverse(matrix: Sequence[Sequence[mp.mpf]]) -> list[list[mp.mpf]]:
    try:
        inv = mp.inverse(mp.matrix(matrix))
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        raise CompositionBridgeError("Hessian is singular") from exc
    return [[_mp(inv[i, j]) for j in range(2)] for i in range(2)]


def _matrix_multiply(a: Sequence[Sequence[mp.mpf]], b: Sequence[Sequence[mp.mpf]]) -> list[list[mp.mpf]]:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _matrix_transpose(a: Sequence[Sequence[mp.mpf]]) -> list[list[mp.mpf]]:
    return [list(row) for row in zip(*a)]


def _matrix_det(a: Sequence[Sequence[mp.mpf]]) -> mp.mpf:
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]


def _species_integrals(integrator: Any, nu: mp.mpf, y: mp.mpf, *, mass_scale: Any | None = None) -> dict[str, mp.mpf]:
    """Halve the maintained d=4 particle/antiparticle integrals to d=2."""
    if mass_scale is None:
        values = integrator.integrals(nu, y)
    else:
        values = integrator.integrals(nu, y, mass_scale=mass_scale)
    return {key: _mp(value) / 2 for key, value in values.items()}


def _root_scale(model: Any, data_n: Mapping[str, mp.mpf], data_p: Mapping[str, mp.mpf], uy: mp.mpf, B: mp.mpf) -> mp.mpf:
    return max(
        abs(model.MN * (data_n["ns"] + data_p["ns"])),
        abs(uy),
        abs(model.Cv * B**2),
        abs(B),
        mp.mpf(1),
    )


@dataclass
class CompositionRoot:
    """Live canonical root and its thermodynamic quantities in natural units."""

    B: mp.mpf
    D: mp.mpf
    nu_n: mp.mpf
    nu_p: mp.mpf
    y: mp.mpf
    n_n: mp.mpf
    n_p: mp.mpf
    ns_n: mp.mpf
    ns_p: mp.mpf
    gap_residual: mp.mpf
    residual_relative: mp.mpf


def _canonical_root(
    integrator: Any,
    model: Any,
    B: Any,
    D: Any,
    *,
    nu_n0: Any,
    nu_p0: Any,
    y0: Any,
) -> CompositionRoot:
    """Solve the two species number equations plus the common scalar saddle."""
    B, D = _mp(B), _mp(D)
    nu_n0, nu_p0, y0 = _mp(nu_n0), _mp(nu_p0), _mp(y0)
    if B <= 0 or abs(D) >= B or y0 <= 0:
        raise CompositionBridgeError("canonical root requires B>0 and |D|<B")
    target_n = (B + D) / 2
    target_p = (B - D) / 2
    cache: dict[tuple[str, str, str], tuple[dict[str, mp.mpf], dict[str, mp.mpf], mp.mpf]] = {}

    def values(nu_n: Any, nu_p: Any, y: Any) -> tuple[dict[str, mp.mpf], dict[str, mp.mpf], mp.mpf]:
        nu_n, nu_p, y = _mp(nu_n), _mp(nu_p), _mp(y)
        key = (_number(nu_n, mp.mp.dps + 8), _number(nu_p, mp.mp.dps + 8), _number(y, mp.mp.dps + 8))
        if key not in cache:
            n_data = _species_integrals(integrator, nu_n, y)
            p_data = _species_integrals(integrator, nu_p, y)
            _, uy, _ = thermal._model_potential(model, y)
            gap = model.MN * (n_data["ns"] + p_data["ns"]) + uy - model.Cv * B**2 / y**3
            cache[key] = (n_data, p_data, gap)
        return cache[key]

    def residual_n(nu_n: Any, nu_p: Any, y: Any) -> mp.mpf:
        return values(nu_n, nu_p, y)[0]["n"] - target_n

    def residual_p(nu_n: Any, nu_p: Any, y: Any) -> mp.mpf:
        return values(nu_n, nu_p, y)[1]["n"] - target_p

    def residual_y(nu_n: Any, nu_p: Any, y: Any) -> mp.mpf:
        return values(nu_n, nu_p, y)[2]

    try:
        root = mp.findroot(
            (residual_n, residual_p, residual_y),
            (nu_n0, nu_p0, y0),
            solver="mdnewton",
            tol=mp.eps ** mp.mpf("0.70"),
            maxsteps=60,
            verify=False,
        )
    except Exception as exc:
        raise CompositionBridgeError(f"canonical root failed: {type(exc).__name__}: {exc}") from exc
    nu_n, nu_p, y = _mp(root[0]), _mp(root[1]), _mp(root[2])
    if not (mp.isfinite(nu_n) and mp.isfinite(nu_p) and mp.isfinite(y) and ROOT_Y_MIN <= y <= ROOT_Y_MAX):
        raise CompositionBridgeError("canonical root left the bounded positive-y branch")
    n_data, p_data, gap = values(nu_n, nu_p, y)
    scale = _root_scale(model, n_data, p_data, thermal._model_potential(model, y)[1], B)
    residual = max(abs(n_data["n"] - target_n), abs(p_data["n"] - target_p), abs(gap)) / scale
    if residual > ROOT_REL_LIMIT:
        raise CompositionBridgeError(f"canonical residual exceeds bound: {_number(residual, 12)}")
    if not (n_data["n"] > 0 and p_data["n"] > 0):
        raise CompositionBridgeError("canonical root has nonpositive species density")
    return CompositionRoot(B, D, nu_n, nu_p, y, n_data["n"], p_data["n"], n_data["ns"], p_data["ns"], gap, residual)


def _derivatives(integrator: Any, model: Any, root: CompositionRoot) -> dict[str, Any]:
    """Construct the analytic three-variable Legendre/Schur Hessian."""
    y, B = root.y, root.B
    nus = (root.nu_n, root.nu_p)
    n_data = (_species_integrals(integrator, root.nu_n, y), _species_integrals(integrator, root.nu_p, y))
    nnu: list[mp.mpf] = []
    snu: list[mp.mpf] = []
    sm: list[mp.mpf] = []
    for nu in nus:
        nnu.append(_mp(mp.diff(lambda value, nu=nu: _species_integrals(integrator, value, y)["n"], nu)))
        snu.append(_mp(mp.diff(lambda value, nu=nu: _species_integrals(integrator, value, y)["ns"], nu)))
        mass = model.MN * y
        sm.append(_mp(mp.diff(lambda value, nu=nu: _species_integrals(integrator, nu, y, mass_scale=value)["ns"], mass)))
    _, uy, uyy = thermal._model_potential(model, y)
    C = uyy + model.MN**2 * sum(sm[i] + snu[i] ** 2 / nnu[i] for i in range(2)) + 3 * model.Cv * B**2 / y**4
    cv_y = model.Cv / y**2
    b = [model.MN * snu[i] / nnu[i] - 2 * model.Cv * B / y**3 for i in range(2)]
    species: list[list[mp.mpf]] = []
    for i in range(2):
        row: list[mp.mpf] = []
        for k in range(2):
            row.append((1 / nnu[i] if i == k else mp.mpf(0)) + cv_y - b[i] * b[k] / C)
        species.append(row)
    R = [[mp.mpf("0.5"), mp.mpf("0.5")], [mp.mpf("0.5"), -mp.mpf("0.5")]]
    return {"nnu": nnu, "snu": snu, "sm": sm, "C_field": C, "b_field": b, "species_hessian_no_contact": species, "R_BD": R, "n_data": n_data, "uy": uy, "uyy": uyy}


def _hessian_and_state(integrator: Any, model: Any, root: CompositionRoot, derivatives: Mapping[str, Any], contact_nat: mp.mpf) -> dict[str, Any]:
    nnu = derivatives["nnu"]
    C = derivatives["C_field"]
    b = derivatives["b_field"]
    cv_y = model.Cv / root.y**2
    # v=(1,-1) is the isovector species direction.
    v = (mp.mpf(1), -mp.mpf(1))
    species: list[list[mp.mpf]] = []
    for i in range(2):
        row: list[mp.mpf] = []
        for k in range(2):
            contact = contact_nat * v[i] * v[k]
            row.append((1 / nnu[i] if i == k else mp.mpf(0)) + cv_y + contact - b[i] * b[k] / C)
        species.append(row)
    R = derivatives["R_BD"]
    H = _matrix_multiply(_matrix_multiply(_matrix_transpose(R), species), R)
    H_fm = [[value * HBARC_MEV_FM**3 for value in row] for row in H]
    det_fm = _matrix_det(H_fm)
    eig_min = (H_fm[0][0] + H_fm[1][1] - mp.sqrt((H_fm[0][0] - H_fm[1][1]) ** 2 + 4 * H_fm[0][1] ** 2)) / 2
    mu_n = root.nu_n + model.Cv * root.B / root.y**2 + contact_nat * root.D
    mu_p = root.nu_p + model.Cv * root.B / root.y**2 - contact_nat * root.D
    mu_B, mu_D = (mu_n + mu_p) / 2, (mu_n - mu_p) / 2
    data_n, data_p = derivatives["n_data"]
    pressure_kinetic = data_n["pressure"] + data_p["pressure"]
    U, _, _ = thermal._model_potential(model, root.y)
    vector = model.Cv * root.B**2 / (2 * root.y**2)
    contact_energy = contact_nat * root.D**2 / 2
    free_energy = root.nu_n * root.n_n + root.nu_p * root.n_p - pressure_kinetic + U + vector + contact_energy
    pressure = pressure_kinetic - U + vector + contact_energy
    pressure_from_mu = mu_B * root.B + mu_D * root.D - free_energy
    pressure_identity = _relative(pressure, pressure_from_mu, floor="1e-25")
    chi_fm = _matrix_inverse(H_fm)
    return {
        "mu_n_MeV": mu_n,
        "mu_p_MeV": mu_p,
        "mu_B_MeV": mu_B,
        "mu_D_MeV": mu_D,
        "H_BD_MeV_fm3": H_fm,
        "H_det_(MeV_fm3)^2": det_fm,
        "H_min_eigenvalue_MeV_fm3": eig_min,
        "susceptibility_BD_fm3_per_MeV": chi_fm,
        "free_energy_MeV4": free_energy,
        "free_energy_MeV_fm3": free_energy / HBARC_MEV_FM**3,
        "pressure_MeV4": pressure,
        "pressure_MeV_fm3": pressure / HBARC_MEV_FM**3,
        "pressure_legendre_MeV4": pressure_from_mu,
        "pressure_identity_relative": pressure_identity,
        "contact_energy_MeV4": contact_energy,
        "contact_energy_MeV_fm3": contact_energy / HBARC_MEV_FM**3,
        "vector_energy_MeV4": vector,
        "potential_MeV4": U,
        "kinetic_pressure_MeV4": pressure_kinetic,
        "field_curvature_MeV4": C,
        "local_stable": bool(C > 0 and H_fm[0][0] > 0 and H_fm[1][1] > 0 and det_fm > 0 and eig_min > 0),
        "contact_nat_MeVminus2": contact_nat,
    }


def _root_public(root: CompositionRoot, model: Any) -> dict[str, Any]:
    return {
        "B_MeV3": root.B,
        "D_MeV3": root.D,
        "B_fm3": root.B / HBARC_MEV_FM**3,
        "D_fm3": root.D / HBARC_MEV_FM**3,
        "nu_n_MeV": root.nu_n,
        "nu_p_MeV": root.nu_p,
        "y": root.y,
        "n_n_fm3": root.n_n / HBARC_MEV_FM**3,
        "n_p_fm3": root.n_p / HBARC_MEV_FM**3,
        "ns_n_MeV3": root.ns_n,
        "ns_p_MeV3": root.ns_p,
        "density_identity_relative": max(_relative(root.n_n + root.n_p, root.B, floor="1e-30"), _relative(root.n_n - root.n_p, root.D, floor="1e-30")),
        "gap_residual_relative": root.residual_relative,
        "gap_residual_MeV4": root.gap_residual,
        "branch": "positive_species_positive_y_bounded_local_canonical_branch",
        "model_n0_fm3": model.n0_fm3,
    }


def _chemical_pair(root: CompositionRoot, model: Any, contact_nat: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    mu_n = root.nu_n + model.Cv * root.B / root.y**2 + contact_nat * root.D
    mu_p = root.nu_p + model.Cv * root.B / root.y**2 - contact_nat * root.D
    return (mu_n + mu_p) / 2, (mu_n - mu_p) / 2


def _finite_difference_controls(
    integrator: Any,
    model: Any,
    base: CompositionRoot,
    derivatives: Mapping[str, Any],
    contact_nat: mp.mpf,
    *,
    include: bool = True,
    hessian: Sequence[Sequence[mp.mpf]] | None = None,
) -> dict[str, Any]:
    if not include:
        return {"steps": [], "all_pass": True, "omitted": True}
    steps: list[dict[str, Any]] = []
    base_mu_b, base_mu_d = _chemical_pair(base, model, contact_nat)
    del derivatives
    root_cache: dict[tuple[str, str], CompositionRoot] = {}

    def scalar_thermodynamics(root: CompositionRoot) -> dict[str, mp.mpf]:
        data_n = _species_integrals(integrator, root.nu_n, root.y)
        data_p = _species_integrals(integrator, root.nu_p, root.y)
        U, _, _ = thermal._model_potential(model, root.y)
        vector = model.Cv * root.B**2 / (2 * root.y**2)
        contact_energy = contact_nat * root.D**2 / 2
        pressure_kinetic = data_n["pressure"] + data_p["pressure"]
        free_energy = root.nu_n * root.n_n + root.nu_p * root.n_p - pressure_kinetic + U + vector + contact_energy
        pressure = pressure_kinetic - U + vector + contact_energy
        mu_b, mu_d = _chemical_pair(root, model, contact_nat)
        return {"free_energy": free_energy, "pressure": pressure, "mu_B": mu_b, "mu_D": mu_d}

    def shifted(dB: mp.mpf, dD: mp.mpf, guess: CompositionRoot) -> CompositionRoot:
        key = (_number(dB, mp.mp.dps + 8), _number(dD, mp.mp.dps + 8))
        if key not in root_cache:
            root_cache[key] = _canonical_root(
                integrator, model, base.B + dB, base.D + dD,
                nu_n0=guess.nu_n, nu_p0=guess.nu_p, y0=guess.y,
            )
        return root_cache[key]

    for rel in FD_REL_STEPS:
        hB, hD = rel * base.B, rel * base.B
        try:
            bp = shifted(hB, mp.mpf(0), base)
            bm = shifted(-hB, mp.mpf(0), base)
            dp = shifted(mp.mpf(0), hD, base)
            dm = shifted(mp.mpf(0), -hD, base)
            mub_p, mud_p = _chemical_pair(bp, model, contact_nat)
            mub_m, mud_m = _chemical_pair(bm, model, contact_nat)
            mub_dp, mud_dp = _chemical_pair(dp, model, contact_nat)
            mub_dm, mud_dm = _chemical_pair(dm, model, contact_nat)
            f_bp = scalar_thermodynamics(bp)["free_energy"]
            f_bm = scalar_thermodynamics(bm)["free_energy"]
            f_dp = scalar_thermodynamics(dp)["free_energy"]
            f_dm = scalar_thermodynamics(dm)["free_energy"]
            h_fd = [
                [(mub_p - mub_m) / (2 * hB), (mub_dp - mub_dm) / (2 * hD)],
                [(mud_p - mud_m) / (2 * hB), (mud_dp - mud_dm) / (2 * hD)],
            ]
            h_fd_fm = [[value * HBARC_MEV_FM**3 for value in row] for row in h_fd]
            if hessian is None:
                raise CompositionBridgeError("finite-difference control needs analytic Hessian")
            maxwell_fd = _relative(h_fd_fm[0][1], h_fd_fm[1][0], floor="1e-25")
            hessian_error = _matrix_relative(h_fd_fm, hessian)
            envelope_mu_b = (f_bp - f_bm) / (2 * hB)
            envelope_mu_d = (f_dp - f_dm) / (2 * hD)
            envelope_b_error = _relative(envelope_mu_b, base_mu_b, floor="1e-25")
            envelope_d_error = _relative(envelope_mu_d, base_mu_d, floor="1e-25")
            envelope_error = max(envelope_b_error, envelope_d_error)
            steps.append({
                "relative_step": rel,
                "h_B_MeV3": hB,
                "h_D_MeV3": hD,
                "H_fd_BD_MeV_fm3": h_fd_fm,
                "max_relative_error": hessian_error,
                "Maxwell_fd_relative": maxwell_fd,
                "envelope_muB_fd_MeV": envelope_mu_b,
                "envelope_muD_fd_MeV": envelope_mu_d,
                "envelope_muB_relative_error": envelope_b_error,
                "envelope_muD_relative_error": envelope_d_error,
                "envelope_max_relative_error": envelope_error,
                "shifted_root_count": 4,
                "pass": bool(hessian_error <= FD_REL_LIMIT and maxwell_fd <= FD_REL_LIMIT and envelope_error <= FD_REL_LIMIT),
            })
        except Exception as exc:
            steps.append({"relative_step": rel, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "pass": False})
    errors = [max(_mp(row.get("max_relative_error", "inf")), _mp(row.get("Maxwell_fd_relative", "inf")), _mp(row.get("envelope_max_relative_error", "inf"))) for row in steps if "max_relative_error" in row]
    trend_pass = bool(len(errors) >= 2 and all(errors[i + 1] < errors[i] for i in range(len(errors) - 1)))
    return {"steps": steps, "refinement_trend_pass": trend_pass, "all_pass": bool(steps and trend_pass and all(row.get("pass", False) for row in steps))}


def _reference_states(dps: int, *, nquad: int, cutoff: mp.mpf) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], Any]]:
    """Build eight thermal reference densities from live grand-canonical roots."""
    refs: list[dict[str, Any]] = []
    models: dict[tuple[str, str, str], Any] = {}
    with mp.workdps(dps):
        for anchor in ANCHORS:
            infos = nonlinear._build_models(anchor)
            for model_name in MODEL_ORDER:
                model = infos[model_name]["model"]
                integrator = thermal._integrator_for("70", cutoff=cutoff, nquad=nquad)
                for mu_text in REFERENCE_CHEMICAL_POTENTIALS:
                    mu = _mp(mu_text)
                    try:
                        root = thermal._root_record(integrator, model, mu, _mp(anchor), _mp(anchor), with_derivatives=False)
                    except Exception as exc:
                        raise CompositionBridgeError(f"reference root failed for {anchor}/{model_name}/{mu_text}: {exc}") from exc
                    if not root.get("stable"):
                        raise CompositionBridgeError(f"reference root is unstable for {anchor}/{model_name}/{mu_text}")
                    state = root["state"]
                    key = (anchor, model_name, _mu_key(mu))
                    refs.append({
                        "anchor": anchor,
                        "T_MeV": mp.mpf("70"),
                        "mu_reference_MeV": mu,
                        "model": model_name,
                        "model_id": infos[model_name]["id"],
                        "amplitude": infos[model_name]["amplitude"],
                        "target_y": infos[model_name]["target_y"],
                        "nu_reference_MeV": root["nu"],
                        "y_reference": root["y"],
                        "B_reference_MeV3": root["state"]["n_MeV3"],
                        "B_reference_fm3": root["state"]["n_fm3"],
                        "reference_root_residual_relative": root["residual_relative"],
                        "reference_local_fnn_MeVminus2": state["f_nn"],
                        "reference_local_fnn_MeV_fm3": state["f_nn"] * HBARC_MEV_FM**3,
                        "reference_branch": "bounded_thermal_positive_y_positive_curvature_local_branch",
                    })
                    models[key] = (model, integrator)
    return refs, models


def _make_row(
    reference: Mapping[str, Any],
    model: Any,
    integrator: Any,
    delta_text: str,
    contact_name: str,
    contact_nat: mp.mpf,
    *,
    include_fd: bool,
    root_cache: dict[str, CompositionRoot],
) -> dict[str, Any]:
    B = _mp(reference["B_reference_MeV3"])
    D = _mp(delta_text) * B
    row_id = f"{reference['anchor']}/70/{reference['mu_reference_MeV']}/{reference['model']}/delta{delta_text}/{contact_name}"
    try:
        root = root_cache.get(delta_text)
        if root is None:
            dnu = mp.mpf("20") if D else mp.mpf(0)
            root = _canonical_root(
                integrator, model, B, D,
                nu_n0=_mp(reference["nu_reference_MeV"]) + dnu,
                nu_p0=_mp(reference["nu_reference_MeV"]) - dnu,
                y0=_mp(reference["y_reference"]),
            )
            root_cache[delta_text] = root
        derivatives = _derivatives(integrator, model, root)
        state = _hessian_and_state(integrator, model, root, derivatives, contact_nat)
        fd = _finite_difference_controls(integrator, model, root, derivatives, contact_nat, include=include_fd, hessian=state["H_BD_MeV_fm3"])
        static_tie = None
        if delta_text == "0" and contact_name == "zero":
            static_tie = _relative(state["H_BD_MeV_fm3"][0][0], reference["reference_local_fnn_MeV_fm3"], floor="1e-20")
        return {
            "row_id": row_id,
            "status": "CANONICAL_ROOT_OK" if state["local_stable"] else "UNSTABLE_LOCAL_BRANCH",
            "pass": bool(state["local_stable"] and fd["all_pass"] and state["pressure_identity_relative"] <= MAXWELL_REL_LIMIT and (static_tie is None or static_tie <= MAXWELL_REL_LIMIT)),
            "anchor": reference["anchor"],
            "T_MeV": reference["T_MeV"],
            "mu_reference_MeV": reference["mu_reference_MeV"],
            "model": reference["model"],
            "amplitude": reference["amplitude"],
            "delta": delta_text,
            "contact": contact_name,
            "contact_j_MeV": J_MEV if contact_name == "fixed_j" else mp.mpf(0),
            "contact_Crho_MeV_fm3": C_RHO_MEV_FM3 if contact_name == "fixed_j" else mp.mpf(0),
            "root": _root_public(root, model),
            "thermodynamics": state,
            "derivative_controls": {
                "field_curvature_formula": "Uyy+MN^2 sum_i(Sm_i+Snu_i^2/Nnu_i)+3 Cv B^2/y^4",
                "analytic_hessian_formula": "R^T[diag(1/Nnu)+Cv/y^2*ones+C_rho vv^T-b b^T/C]R",
                "Maxwell_relative": _relative(state["H_BD_MeV_fm3"][0][1], state["H_BD_MeV_fm3"][1][0], floor="1e-35"),
                "static_q0_tie_relative": static_tie,
                "positive_definite": state["local_stable"],
                "pressure_identity_relative": state["pressure_identity_relative"],
            },
            "finite_difference_controls": fd,
            "units": {"B_D": "fm^-3 in public root; natural MeV^3 internally", "Hessian": "MeV fm^3", "susceptibility": "fm^-3 MeV^-1", "free_energy": "MeV fm^-3"},
        }
    except Exception as exc:
        return {
            "row_id": row_id,
            "status": "ROOT_FAILURE",
            "pass": False,
            "anchor": reference["anchor"],
            "T_MeV": reference["T_MeV"],
            "mu_reference_MeV": reference["mu_reference_MeV"],
            "model": reference["model"],
            "amplitude": reference["amplitude"],
            "delta": delta_text,
            "contact": contact_name,
            "contact_j_MeV": J_MEV if contact_name == "fixed_j" else mp.mpf(0),
            "contact_Crho_MeV_fm3": C_RHO_MEV_FM3 if contact_name == "fixed_j" else mp.mpf(0),
            "error": f"{type(exc).__name__}: {exc}",
            "branch_failure_preserved": True,
        }


def _exchange_controls(
    refs: Sequence[Mapping[str, Any]],
    models: Mapping[tuple[str, str, str], Any],
    primary_rows: Sequence[Mapping[str, Any]],
    *,
    include: bool,
) -> dict[str, Any]:
    """Bounded δ↔−δ species-exchange check, not a production grid extension."""
    if not include:
        return {"rows": [], "all_pass": True, "omitted": True}
    rows: list[dict[str, Any]] = []
    for ref in refs:
        model, integrator = models[(ref["anchor"], ref["model"], _mu_key(ref["mu_reference_MeV"]))]
        B = _mp(ref["B_reference_MeV3"])
        D = mp.mpf("0.2") * B
        try:
            plus = _canonical_root(integrator, model, B, D, nu_n0=_mp(ref["nu_reference_MeV"]) + 20, nu_p0=_mp(ref["nu_reference_MeV"]) - 20, y0=_mp(ref["y_reference"]))
            minus = _canonical_root(integrator, model, B, -D, nu_n0=plus.nu_p, nu_p0=plus.nu_n, y0=plus.y)
            plus_d = _derivatives(integrator, model, plus)
            minus_d = _derivatives(integrator, model, minus)
            plus_s = _hessian_and_state(integrator, model, plus, plus_d, mp.mpf(0))
            minus_s = _hessian_and_state(integrator, model, minus, minus_d, mp.mpf(0))
            plus_mu = _chemical_pair(plus, model, mp.mpf(0))
            minus_mu = _chemical_pair(minus, model, mp.mpf(0))
            hplus, hminus = plus_s["H_BD_MeV_fm3"], minus_s["H_BD_MeV_fm3"]
            checks = {
                "y_relative": _relative(plus.y, minus.y),
                "nu_exchange_relative": max(_relative(plus.nu_n, minus.nu_p), _relative(plus.nu_p, minus.nu_n)),
                "mu_B_relative": _relative(plus_mu[0], minus_mu[0]),
                "mu_D_odd_relative": _relative(plus_mu[1], -minus_mu[1], floor="1e-25"),
                "HBB_even_relative": _relative(hplus[0][0], hminus[0][0]),
                "HDD_even_relative": _relative(hplus[1][1], hminus[1][1]),
                "HBD_odd_relative": _relative(hplus[0][1], -hminus[0][1], floor="1e-25"),
            }
            rows.append({"reference": f"{ref['anchor']}/{ref['model']}/{ref['mu_reference_MeV']}", "checks": checks, "pass": bool(max(checks.values()) <= MAXWELL_REL_LIMIT)})
        except Exception as exc:
            rows.append({"reference": f"{ref['anchor']}/{ref['model']}/{ref['mu_reference_MeV']}", "status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "pass": False})
    del primary_rows
    return {"rows": rows, "all_pass": bool(rows and all(row.get("pass", False) for row in rows))}


def _contact_controls(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["anchor"], str(row["mu_reference_MeV"]), row["model"], row["delta"]), {})[row["contact"]] = row
    controls: list[dict[str, Any]] = []
    for key, pair in grouped.items():
        zero, fixed = pair.get("zero"), pair.get("fixed_j")
        if not zero or not fixed or "thermodynamics" not in zero or "thermodynamics" not in fixed:
            controls.append({"reference": key, "pass": False, "status": "MISSING_CONTACT_PAIR"})
            continue
        z, j = zero["thermodynamics"], fixed["thermodynamics"]
        crho = C_RHO_MEV_FM3
        h_delta = [[_mp(j["H_BD_MeV_fm3"][i][k]) - _mp(z["H_BD_MeV_fm3"][i][k]) for k in range(2)] for i in range(2)]
        expected = [[mp.mpf(0), mp.mpf(0)], [mp.mpf(0), crho]]
        contact_scale = max(abs(crho), mp.mpf(1))
        contact_error = max(abs(h_delta[i][k] - expected[i][k]) for i in range(2) for k in range(2)) / contact_scale
        checks = {
            "H_contact_identity_max_relative": contact_error,
            "mu_D_shift_relative": _relative(_mp(j["mu_D_MeV"]) - _mp(z["mu_D_MeV"]), C_RHO_NATURAL * _mp(zero["root"]["D_MeV3"]), floor="1e-25"),
            "free_energy_contact_relative": _relative(_mp(j["free_energy_MeV4"]) - _mp(z["free_energy_MeV4"]), C_RHO_NATURAL * _mp(zero["root"]["D_MeV3"])**2 / 2, floor="1e-30"),
            "pressure_contact_relative": _relative(_mp(j["pressure_MeV4"]) - _mp(z["pressure_MeV4"]), C_RHO_NATURAL * _mp(zero["root"]["D_MeV3"])**2 / 2, floor="1e-30"),
            "root_y_relative": _relative(fixed["root"]["y"], zero["root"]["y"]),
            "root_nu_n_relative": _relative(fixed["root"]["nu_n_MeV"], zero["root"]["nu_n_MeV"]),
            "root_nu_p_relative": _relative(fixed["root"]["nu_p_MeV"], zero["root"]["nu_p_MeV"]),
        }
        # At D=0 the contact energy and mu_D shift are exactly zero; using a
        # small absolute floor avoids declaring a false failure from 0/0.
        checks["free_energy_contact_relative"] = _relative(_mp(j["free_energy_MeV4"]) - _mp(z["free_energy_MeV4"]), C_RHO_NATURAL * _mp(zero["root"]["D_MeV3"])**2 / 2, floor="1e-30")
        controls.append({"reference": key, "checks": checks, "pass": bool(max(checks.values()) <= MAXWELL_REL_LIMIT)})
    return {"rows": controls, "all_pass": bool(controls and all(row.get("pass", False) for row in controls))}


def _charge_coordinate_controls(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check the thermodynamic map from ``(B,Q)`` to ``(B,D)``.

    The Jacobian is defined by ``x_BD = S x_BQ`` with ``D=B-2Q``.  Both
    Hessian and susceptibility transforms are checked independently; in
    particular, no fresh inverse is allowed to hide a corrupted exported
    ``chi_BD`` input.
    """
    S = [[mp.mpf(1), mp.mpf(0)], [mp.mpf(1), -mp.mpf(2)]]
    S_inv = _matrix_inverse(S)
    identity = [[mp.mpf(1), mp.mpf(0)], [mp.mpf(0), mp.mpf(1)]]

    def matrix_norm_residual(actual: Sequence[Sequence[Any]], expected: Sequence[Sequence[Any]], *, identity_target: bool = False) -> mp.mpf:
        """Return a max-absolute matrix residual in the published units.

        Identity products are dimensionless and therefore use ``||I||_max=1``
        rather than the scalar component floor used by the generic relative
        helper.  Transform comparisons are scaled by the max-absolute norm of
        the expected (true) matrix, so rounded zero components do not acquire
        an artificial 1e-35 denominator.
        """
        if len(actual) != len(expected) or any(len(a_row) != len(e_row) for a_row, e_row in zip(actual, expected)):
            return mp.inf
        difference = max(
            abs(_mp(a_value) - _mp(e_value))
            for a_row, e_row in zip(actual, expected)
            for a_value, e_value in zip(a_row, e_row)
        )
        if identity_target:
            return difference
        expected_norm = max((abs(_mp(value)) for row_values in expected for value in row_values), default=mp.mpf(0))
        if expected_norm == 0:
            return mp.mpf(0) if difference == 0 else mp.inf
        return difference / expected_norm

    checks: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "CANONICAL_ROOT_OK":
            checks.append({"row_id": row.get("row_id"), "pass": False})
            continue
        try:
            # Public snapshots store numbers as strings, while the live tree
            # uses mp.mpf values.  Normalize both forms before multiplying.
            H = [[_mp(value) for value in part] for part in row["thermodynamics"]["H_BD_MeV_fm3"]]
            chi_bd = [[_mp(value) for value in part] for part in row["thermodynamics"]["susceptibility_BD_fm3_per_MeV"]]
            h_bq = _matrix_multiply(_matrix_multiply(_matrix_transpose(S), H), S)
            chi_bq = _matrix_inverse(h_bq)
            chi_bq_from_bd = _matrix_multiply(
                _matrix_multiply(S_inv, chi_bd), _matrix_transpose(S_inv)
            )
            checks_by_name = {
                "susceptibility_BD_inverse_relative": matrix_norm_residual(chi_bd, _matrix_inverse(H)),
                "susceptibility_BD_identity_relative": matrix_norm_residual(_matrix_multiply(H, chi_bd), identity, identity_target=True),
                "susceptibility_BQ_transform_relative": matrix_norm_residual(chi_bq, chi_bq_from_bd),
                "susceptibility_BQ_identity_relative": matrix_norm_residual(_matrix_multiply(h_bq, chi_bq), identity, identity_target=True),
            }
            finite = all(mp.isfinite(value) for matrix in (H, chi_bd, h_bq, chi_bq, chi_bq_from_bd) for part in matrix for value in part)
            finite = bool(finite and all(mp.isfinite(value) for value in checks_by_name.values()))
            max_relative = max(checks_by_name.values())
            checks.append({
                "row_id": row["row_id"],
                "H_BQ_MeV_fm3": h_bq,
                "susceptibility_BQ_fm3_per_MeV": chi_bq,
                "susceptibility_BQ_from_BD_fm3_per_MeV": chi_bq_from_bd,
                "checks": checks_by_name,
                "finite": finite,
                "max_relative": max_relative,
                "pass": bool(finite and max_relative <= MAXWELL_REL_LIMIT),
            })
        except (ArithmeticError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError, CompositionBridgeError) as exc:
            checks.append({"row_id": row.get("row_id"), "pass": False, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
    return {"jacobian_BD_from_BQ": S, "rows": checks, "all_pass": bool(checks and all(row.get("pass", False) for row in checks)), "interpretation": "thermodynamic coordinate map only; no charge/electron or emission projection"}


def _static_bridge_controls(refs: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(row["anchor"], str(row["mu_reference_MeV"]), row["model"], row["delta"], row["contact"]): row for row in rows}
    checks = []
    for ref in refs:
        row = by_key.get((ref["anchor"], str(ref["mu_reference_MeV"]), ref["model"], "0", "zero"))
        if not row:
            checks.append({"reference": ref["anchor"], "pass": False})
            continue
        value = row["derivative_controls"]["static_q0_tie_relative"]
        checks.append({"reference": f"{ref['anchor']}/{ref['model']}/{ref['mu_reference_MeV']}", "relative": value, "pass": value is not None and _mp(value) <= MAXWELL_REL_LIMIT})
    return {"rows": checks, "all_pass": bool(checks and all(row.get("pass", False) for row in checks)), "meaning": "grand-canonical q=0 static density curvature tie, not covariance"}


def _precision_controls(primary: Sequence[Mapping[str, Any]], control: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    p_map = {row["row_id"]: row for row in primary}
    c_map = {row["row_id"]: row for row in control}
    fields = ("H_BD_MeV_fm3", "free_energy_MeV4", "pressure_MeV4", "mu_B_MeV", "mu_D_MeV", "y")
    checks: list[dict[str, Any]] = []
    for row_id, p in p_map.items():
        c = c_map.get(row_id)
        if not c or p.get("status") != "CANONICAL_ROOT_OK" or c.get("status") != "CANONICAL_ROOT_OK":
            checks.append({"row_id": row_id, "pass": False, "status": "MISSING_OR_FAILED_CONTROL"})
            continue
        values = {}
        for field in fields:
            if field == "H_BD_MeV_fm3":
                values[field] = _matrix_relative(p["thermodynamics"][field], c["thermodynamics"][field])
            elif field == "y":
                values[field] = _relative(p["root"][field], c["root"][field])
            else:
                values[field] = _relative(p["thermodynamics"][field], c["thermodynamics"][field])
        checks.append({"row_id": row_id, "fields": values, "max_relative_difference": max(values.values()), "pass": bool(max(values.values()) <= REFINEMENT_REL_LIMIT)})
    return {"rows": checks, "all_pass": bool(checks and all(row.get("pass", False) for row in checks)), "primary": {"dps": PRIMARY_DPS, "nquad": PRIMARY_NQUAD, "cutoff_MeV": PRIMARY_CUTOFF_MEV}, "control": {"dps": CONTROL_DPS, "nquad": CONTROL_NQUAD, "cutoff_MeV": CONTROL_CUTOFF_MEV}}


def _build_tree(dps: int, *, nquad: int, cutoff: mp.mpf, include_fd: bool, include_exchange: bool) -> dict[str, Any]:
    if dps < 36 or nquad < 32 or cutoff <= 0:
        raise CompositionBridgeError("invalid composition quadrature configuration")
    with mp.workdps(dps):
        refs, models = _reference_states(dps, nquad=nquad, cutoff=cutoff)
        rows: list[dict[str, Any]] = []
        for ref in refs:
            model, integrator = models[(ref["anchor"], ref["model"], _mu_key(ref["mu_reference_MeV"]))]
            root_cache: dict[str, CompositionRoot] = {}
            for delta in DELTAS:
                for contact in CONTACT_ORDER:
                    c_nat = C_RHO_NATURAL if contact == "fixed_j" else mp.mpf(0)
                    rows.append(_make_row(ref, model, integrator, delta, contact, c_nat, include_fd=include_fd, root_cache=root_cache))
        if len(refs) != EXPECTED_REFERENCE_COUNT or len(rows) != EXPECTED_ROW_COUNT:
            raise CompositionBridgeError("live composition coverage is incomplete")
        controls = {
            "reference_coverage": {"count": len(refs), "expected": EXPECTED_REFERENCE_COUNT, "all_pass": all(_mp(ref["reference_root_residual_relative"]) <= ROOT_REL_LIMIT for ref in refs)},
            "composition_coverage": {"count": len({(r["anchor"], str(r["mu_reference_MeV"]), r["model"], r["delta"]) for r in rows}), "expected": EXPECTED_COMPOSITION_COUNT, "all_pass": True},
            "row_coverage": {"count": len(rows), "expected": EXPECTED_ROW_COUNT, "all_pass": len(rows) == EXPECTED_ROW_COUNT},
            "finite_difference_refinement": {"rows": [r["row_id"] for r in rows if r.get("finite_difference_controls", {}).get("steps")], "all_pass": all(r.get("finite_difference_controls", {}).get("all_pass", False) for r in rows)},
            "contact_identity": _contact_controls(rows),
            "charge_coordinates": _charge_coordinate_controls(rows),
            "pressure_identity": {
                "rows": [r["row_id"] for r in rows if r.get("status") == "CANONICAL_ROOT_OK"],
                "all_pass": all(r.get("status") == "CANONICAL_ROOT_OK" and _mp(r.get("thermodynamics", {}).get("pressure_identity_relative", "inf")) <= MAXWELL_REL_LIMIT for r in rows),
                "identity": "P=P_kin-U+Cv B^2/(2y^2)+C_rho D^2/2=mu_B B+mu_D D-f",
            },
            "static_q0_bridge": {"deferred": True} if not include_fd else {},
        }
        if include_fd:
            controls["static_q0_bridge"] = _static_bridge_controls(refs, rows)
            controls["species_exchange"] = _exchange_controls(refs, models, rows, include=True)
        else:
            controls["species_exchange"] = {"rows": [], "all_pass": True, "omitted": True}
        return {"references": refs, "rows": rows, "controls": controls, "models": models}


def _public_tree(tree: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in tree.items() if key != "models"}


def _public_result(primary: Mapping[str, Any], control: Mapping[str, Any], precision: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_PASS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "protocol_provenance": "P2_ADOPTED_H6_LOCAL_COMPOSITION_BRIDGE_FROM_SAME_W8_U16_ACTION",
        "physical_inputs_fixed": True,
        "no_saved_results_as_inputs": True,
        "inputs": {
            "anchors": list(ANCHORS),
            "temperature_MeV": list(TEMPERATURES),
            "reference_chemical_potential_MeV": list(REFERENCE_CHEMICAL_POTENTIALS),
            "models": list(MODEL_ORDER),
            "deformation_amplitudes": MODEL_AMPLITUDES,
            "deltas": list(DELTAS),
            "contacts": {"zero": "C_rho=0", "fixed_j": "j historical input"},
            "j_MeV": J_MEV_TEXT,
            "n0_fm3": N0_FM3_TEXT,
            "C_rho_MeV_fm3": C_RHO_MEV_FM3,
            "C_rho_natural_MeVminus2": C_RHO_NATURAL,
            "hbarc_MeV_fm": HBARC_MEV_FM_TEXT,
            "species_degeneracy": 2,
            "maintained_total_degeneracy": 4,
            "primary_dps": PRIMARY_DPS,
            "control_dps": CONTROL_DPS,
            "primary_nquad": PRIMARY_NQUAD,
            "control_nquad": CONTROL_NQUAD,
            "primary_cutoff_MeV": PRIMARY_CUTOFF_MEV,
            "control_cutoff_MeV": CONTROL_CUTOFF_MEV,
            "finite_difference_relative_steps": FD_REL_STEPS,
        },
        "units_and_conventions": {
            "natural_units": "hbar=c=1; n,B,D in MeV^3, f/P/U in MeV^4; public density fm^-3",
            "hessian": "d mu_a/d x_b in MeV fm^3 for x=(B,D)",
            "pressure": "total P in MeV fm^-3; P=mu_B B+mu_D D-f at fixed T",
            "susceptibility": "inverse Hessian in fm^-3 MeV^-1",
            "functional": "f=sum_i(nu_i n_i-P_i)+U(y)+Cv B^2/(2y^2)+C_rho D^2/2",
            "species": "two d=2 particle+antiparticle FD species; each integral is half maintained d=4 integral",
            "branch": "bounded positive-y positive-species local canonical branch; no global-minimum claim",
        },
        "references": primary["references"],
        "rows": primary["rows"],
        "coverage": {"reference_count": len(primary["references"]), "expected_reference_count": EXPECTED_REFERENCE_COUNT, "composition_count": len({(r["anchor"], str(r["mu_reference_MeV"]), r["model"], r["delta"]) for r in primary["rows"]}), "expected_composition_count": EXPECTED_COMPOSITION_COUNT, "row_count": len(primary["rows"]), "expected_row_count": EXPECTED_ROW_COUNT, "all_rows_present": len(primary["rows"]) == EXPECTED_ROW_COUNT},
        "controls": primary["controls"],
        "precision_controls": precision,
        "control_tree_summary": {"reference_count": len(control["references"]), "row_count": len(control["rows"]), "all_rows_stable": all(r.get("pass", False) for r in control["rows"]), "quadrature": {"primary": PRIMARY_NQUAD, "control": CONTROL_NQUAD}},
        "interpretation_limits": [
            "Local homogeneous finite-T composition Hessian only; positive local curvature is not a global phase-equilibrium proof.",
            "C_rho=2j/n0 is a conditional historical transfer and a known contact identity, not a new force or universal prediction.",
            "B,Q coordinate conversion is a thermodynamic variable map only; electrons, beta equilibrium, Coulomb, clusters and detector acceptance are absent.",
            "No isentropic causality, TOV, neutron-star, HADES, cumulant, covariance or empirical-confirmation claim is made.",
            "Finite quadrature/cutoff controls approximate the infinite thermal integrals; local branch coverage is bounded and not exhaustive.",
        ],
        "source_sha256": {
            "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "thermal_observable_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(),
            "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest(),
        },
    }
    gate_names = ("reference_coverage", "composition_coverage", "row_coverage", "finite_difference_refinement", "contact_identity", "charge_coordinates", "pressure_identity", "static_q0_bridge", "species_exchange")
    result["all_controls_pass"] = bool(all(primary["controls"].get(name, {}).get("all_pass", False) for name in gate_names))
    result["precision_controls"]["all_pass"] = bool(precision.get("all_pass", False))
    result["status"] = STATUS_PASS if result["all_controls_pass"] and result["precision_controls"]["all_pass"] and all(row.get("pass", False) for row in primary["rows"]) else STATUS_FAIL
    return result


def _canonical_payload(result: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def build_result() -> dict[str, Any]:
    primary = _build_tree(PRIMARY_DPS, nquad=PRIMARY_NQUAD, cutoff=PRIMARY_CUTOFF_MEV, include_fd=True, include_exchange=True)
    control = _build_tree(CONTROL_DPS, nquad=CONTROL_NQUAD, cutoff=CONTROL_CUTOFF_MEV, include_fd=False, include_exchange=False)
    precision = _precision_controls(primary["rows"], control["rows"])
    result = _jsonable(_public_result(_public_tree(primary), _public_tree(control), _jsonable(precision)))
    result["integrity_sha256"] = hashlib.sha256(_canonical_payload(result).encode("utf-8")).hexdigest()
    return result


def _expected_row_keys() -> set[tuple[str, str, str, str, str, str]]:
    return {(a, _mu_key(mu), model, delta, contact, _mu_key("70")) for a in ANCHORS for mu in REFERENCE_CHEMICAL_POTENTIALS for model in MODEL_ORDER for delta in DELTAS for contact in CONTACT_ORDER}


def validate_result(result: Any) -> bool:
    """Validate a live result tree without reading the serialized JSON file."""
    try:
        if not isinstance(result, Mapping) or not _finite_tree(result):
            return False
        if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != STATUS_PASS:
            return False
        if result.get("evidence_weight") != EVIDENCE_WEIGHT or not result.get("physical_inputs_fixed") or not result.get("no_saved_results_as_inputs"):
            return False
        inputs = result.get("inputs", {})
        if inputs.get("j_MeV") != J_MEV_TEXT or inputs.get("n0_fm3") != N0_FM3_TEXT or inputs.get("hbarc_MeV_fm") != HBARC_MEV_FM_TEXT:
            return False
        rows = result.get("rows", [])
        if not isinstance(rows, list) or len(rows) != EXPECTED_ROW_COUNT:
            return False
        actual = {(str(row.get("anchor")), str(row.get("mu_reference_MeV")), row.get("model"), row.get("delta"), row.get("contact"), str(row.get("T_MeV"))) for row in rows if isinstance(row, Mapping)}
        if actual != _expected_row_keys() or len({row.get("row_id") for row in rows}) != EXPECTED_ROW_COUNT:
            return False
        if any(row.get("status") != "CANONICAL_ROOT_OK" or not row.get("pass") for row in rows):
            return False
        coverage = result.get("coverage", {})
        if coverage.get("reference_count") != EXPECTED_REFERENCE_COUNT or coverage.get("composition_count") != EXPECTED_COMPOSITION_COUNT or coverage.get("row_count") != EXPECTED_ROW_COUNT or coverage.get("all_rows_present") is not True:
            return False
        controls = result.get("controls", {})
        required = ("reference_coverage", "composition_coverage", "row_coverage", "finite_difference_refinement", "contact_identity", "charge_coordinates", "pressure_identity", "static_q0_bridge", "species_exchange")
        if any(name not in controls or controls[name].get("all_pass") is not True for name in required):
            return False
        # Recompute the coordinate control from the public 40-digit row
        # strings.  The serialized boolean is an output, not an authority.
        with mp.workdps(max(CONTROL_DPS, 65)):
            if _charge_coordinate_controls(rows).get("all_pass") is not True:
                return False
        if result.get("precision_controls", {}).get("all_pass") is not True:
            return False
        expected_hashes = {
            "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "thermal_observable_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(),
            "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest(),
        }
        if result.get("source_sha256") != expected_hashes:
            return False
        digest = result.get("integrity_sha256")
        copy_result = dict(result)
        copy_result.pop("integrity_sha256", None)
        return isinstance(digest, str) and hashlib.sha256(_canonical_payload(copy_result).encode("utf-8")).hexdigest() == digest
    except (ArithmeticError, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError, json.JSONDecodeError):
        return False


def _write_json(payload: Mapping[str, Any], path: Path | None) -> None:
    text = json.dumps(_jsonable(payload), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path is not None:
        path.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = build_result()
        if result.get("status") != STATUS_PASS or not validate_result(result):
            _write_json(result, args.output)
            return 1
        _write_json(result, args.output)
        return 0
    except Exception as exc:
        failure = {"schema_version": SCHEMA_VERSION, "status": STATUS_FAIL, "error": f"{type(exc).__name__}: {exc}", "physical_inputs_fixed": True, "no_saved_results_as_inputs": True}
        _write_json(failure, args.output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
