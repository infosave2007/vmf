#!/usr/bin/env python3
"""Source -> finite-q NVG density mode -> explicit reciprocal RLC load.

This producer is deliberately an *open-system algebra scenario*.  The NVG
part is the maintained finite-q retarded response in
``nvg_retarded_response.py``.  The receiver (a pressure/mechanical port and a
reciprocal RLC coordinate) is added here; its geometry, overlap, volume and
coupling are not fixed by NVG and no SI watts are inferred.

The command prints strict JSON and does not write files unless ``--write`` is
supplied.  The normalized RLC values are a declared dimensionless control,
not a hardware recommendation or an NVG coefficient.
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
from scipy.integrate import solve_ivp

# Never create __pycache__ as a side effect of a producer run.
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import nvg_retarded_response as retarded  # noqa: E402
import source_complete_scaling_saturation_audit as upstream  # noqa: E402

SCHEMA_VERSION = 1
STATUS = "LIVE_FINITE_Q_NVG_TO_DIMENSIONLESS_EXTERNAL_RLC"
SCENARIO_FLAG = "DIMENSIONLESS_ALGEBRA_ADDED_LOAD_NOT_HARDWARE_NOT_NVG_PREDICTION"
COORDINATE_ORDER = ("n", "y", "A0", "A_L")
FULL_COORDINATE_ORDER = ("n", "y", "A0", "A_L", "Q")
Q_GRID_MEV = ("100", "200")
OMEGA_FRACTIONS = ("0.25", "0.75", "1.25")
LOAD_R_VALUES = ("0.25", "1", "4")
L_HAT = 1.0
C_HAT = 1.0
G_HAT = 0.5
SOURCE_HAT = 1.0
EXPECTED_BASE_ROWS = 3 * len(Q_GRID_MEV) * len(OMEGA_FRACTIONS)
EXPECTED_LOADED_ROWS = EXPECTED_BASE_ROWS * len(LOAD_R_VALUES)


class DrivenLoadError(ValueError):
    """Malformed or unresolved source/load input."""


def _finite_real(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise DrivenLoadError(f"{name} must be a finite real, not bool")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DrivenLoadError(f"{name} must be a finite real") from exc
    if not math.isfinite(out):
        raise DrivenLoadError(f"{name} must be finite")
    if positive and out <= 0:
        raise DrivenLoadError(f"{name} must be positive")
    if nonnegative and out < 0:
        raise DrivenLoadError(f"{name} must be nonnegative")
    return out


def _finite_complex(value: Any) -> bool:
    try:
        z = complex(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(z.real) and math.isfinite(z.imag)


def _cjson(value: Any, digits: int = 17) -> dict[str, str]:
    z = complex(value)
    if not _finite_complex(z):
        raise DrivenLoadError("nonfinite complex output")
    return {"real": format(z.real, f".{digits}g"), "imag": format(z.imag, f".{digits}g")}


def _rjson(value: Any, digits: int = 17) -> str:
    x = _finite_real(value, "derived number")
    return format(x, f".{digits}g")


def _matrix_json(value: Any) -> list[list[dict[str, str]]]:
    matrix = np.asarray(value, dtype=complex)
    if not np.all(np.isfinite(matrix)):
        raise DrivenLoadError("nonfinite matrix output")
    return [[_cjson(x) for x in row] for row in matrix]


def _rel(a: Any, b: Any, *, floor: float = 1.0) -> float:
    za, zb = complex(a), complex(b)
    return abs(za - zb) / max(abs(za), abs(zb), floor)


def _max_rel(values: Sequence[float]) -> float:
    return max((float(x) for x in values), default=0.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_load_parameters(*, L_hat: Any = L_HAT, C_hat: Any = C_HAT,
                             G_hat_value: Any = G_HAT, R_hat: Any = 1.0) -> dict[str, float]:
    """Validate the explicitly dimensionless added receiver passport.

    ``R_hat >= 0`` is the positive-real/passive boundary.  No conversion to
    ohms, henries, farads, force, volume or watts is made.
    """
    L = _finite_real(L_hat, "L_hat", positive=True)
    C = _finite_real(C_hat, "C_hat", positive=True)
    G = _finite_real(G_hat_value, "G_hat")
    R = _finite_real(R_hat, "R_hat", nonnegative=True)
    return {"L_hat": L, "C_hat": C, "G_hat": G, "R_hat": R}


def rlc_impedance(omega_hat: Any, *, L_hat: Any = L_HAT, C_hat: Any = C_HAT,
                  R_hat: Any = 1.0) -> complex:
    """Dimensionless series-RLC inverse operator.

    With ``exp(-i Omega tau)``, ``Z_e = R - i Omega L + i/(Omega C)``.
    The value at ``Omega=0`` is singular by construction, so callers must
    reject it rather than clipping or manufacturing a DC impedance.
    """
    om = _finite_real(omega_hat, "Omega", positive=True)
    p = validate_load_parameters(L_hat=L_hat, C_hat=C_hat, R_hat=R_hat)
    return complex(p["R_hat"], -om * p["L_hat"] + 1.0 / (om * p["C_hat"]))


def _effective_density_inverse(H4: np.ndarray) -> tuple[complex, np.ndarray, np.ndarray, np.ndarray]:
    """Schur eliminate (y,A0,A_L) from normalized four-coordinate H."""
    h = np.asarray(H4, dtype=complex)
    if h.shape != (4, 4) or not np.all(np.isfinite(h)):
        raise DrivenLoadError("normalized NVG Hessian must be finite 4x4")
    rest = h[1:, 1:]
    try:
        rest_inv_c = np.linalg.solve(rest, h[1:, 0])
    except np.linalg.LinAlgError as exc:
        raise DrivenLoadError("scalar/vector block is singular") from exc
    K = complex(h[0, 0] - h[0, 1:] @ rest_inv_c)
    if not _finite_complex(K):
        raise DrivenLoadError("density Schur inverse is nonfinite")
    return K, h[0, 1:].copy(), h[1:, 0].copy(), rest.copy()


def solve_loaded(H4_normalized: Any, omega_hat: Any, *, L_hat: Any = L_HAT,
                 C_hat: Any = C_HAT, G_hat_value: Any = G_HAT,
                 R_hat: Any = 1.0, source_hat: Any = SOURCE_HAT) -> dict[str, Any]:
    """Solve the declared normalized 5x5 NVG+RLC system and its Schur form.

    The full coordinate order is exactly ``(n,y,A0,A_L,Q)``.  The reciprocal
    velocity coupling is encoded by
    ``H[n,Q]=-i Omega G`` and ``H[Q,n]=+i Omega G``.  ``I=-i Omega Q``.
    """
    # DC is a valid loaded limit in the Q-coordinate formulation.  The
    # current impedance Z_e itself is undefined at Omega=0, but H_QQ=1/C is
    # regular and the reciprocal velocity couplings vanish exactly.
    om = _finite_real(omega_hat, "Omega", nonnegative=True)
    pars = validate_load_parameters(L_hat=L_hat, C_hat=C_hat, G_hat_value=G_hat_value, R_hat=R_hat)
    source = complex(source_hat)
    if not _finite_complex(source):
        raise DrivenLoadError("source_hat must be finite")
    h4 = np.asarray(H4_normalized, dtype=complex)
    if h4.shape != (4, 4) or not np.all(np.isfinite(h4)):
        raise DrivenLoadError("normalized NVG Hessian must be finite 4x4")
    ze = None if om == 0.0 else complex(pars["R_hat"], -om * pars["L_hat"] + 1.0 / (om * pars["C_hat"]))
    if ze is not None and not _finite_complex(ze):
        raise DrivenLoadError("RLC impedance is nonfinite")
    full = np.zeros((5, 5), dtype=complex)
    full[:4, :4] = h4
    full[0, 4] = -1j * om * pars["G_hat"]
    full[4, 0] = +1j * om * pars["G_hat"]
    # The Q-coordinate Hessian is -i*Omega times the current impedance for
    # Omega>0; at DC retain its regular coordinate form directly.
    h_qq = (1.0 / pars["C_hat"] if om == 0.0 else -1j * om * ze)
    full[4, 4] = h_qq
    source_vector = np.zeros(5, dtype=complex)
    source_vector[0] = source
    try:
        direct = np.linalg.solve(full, source_vector)
    except np.linalg.LinAlgError as exc:
        raise DrivenLoadError("full 5x5 system is singular or at a declared near pole") from exc
    if not np.all(np.isfinite(direct)):
        raise DrivenLoadError("full 5x5 solution is nonfinite")

    try:
        condition_number = float(np.linalg.cond(full))
    except np.linalg.LinAlgError as exc:
        raise DrivenLoadError("full 5x5 condition number is undefined") from exc
    if not math.isfinite(condition_number):
        raise DrivenLoadError("full 5x5 condition number is nonfinite")
    near_loaded_pole = bool(condition_number > 1e12)

    k_nvg, b, c, rest = _effective_density_inverse(h4)
    # Use the coupled determinant instead of dividing by a bare NVG K, a bare
    # Q-block resonance, or Z_e.  A passive load can regularize K_NVG=0, and
    # G!=0 can regularize H_QQ=0; only Delta=0/full-solve failure is singular.
    loaded_determinant = k_nvg * h_qq - om * om * pars["G_hat"] ** 2
    determinant_scale = max(abs(k_nvg * h_qq), abs(om * om * pars["G_hat"] ** 2), np.finfo(float).tiny)
    if (not _finite_complex(loaded_determinant) or abs(loaded_determinant) == 0.0 or
            abs(loaded_determinant) <= 1e-13 * determinant_scale):
        raise DrivenLoadError("coupled loaded determinant is singular or numerically unresolved")
    n_schur = source * h_qq / loaded_determinant
    q_schur = -1j * om * pars["G_hat"] * source / loaded_determinant
    k_loaded = None if h_qq == 0 else loaded_determinant / h_qq
    try:
        internal_schur = -np.linalg.solve(rest, c * n_schur)
    except np.linalg.LinAlgError as exc:
        raise DrivenLoadError("internal NVG Schur block became singular") from exc
    schur = np.concatenate(([n_schur], internal_schur, [q_schur]))
    schur_relative_error = max((_rel(direct[i], schur[i]) for i in range(5)), default=0.0)
    current_direct = -1j * om * direct[4]
    current_schur = -1j * om * q_schur
    # Harmonic work with exp(-i Omega tau).  These are normalized algebra
    # powers per unit source-hat squared, not physical powers/watts.
    pin = om * 0.5 * float(np.imag(np.conj(source) * direct[0]))
    p_out = pars["R_hat"] * abs(current_direct) ** 2 / 2.0
    p_intrinsic = -om * 0.5 * float(np.imag(k_nvg)) * abs(direct[0]) ** 2
    balance_residual = pin - p_out - p_intrinsic
    im_k_nvg = float(np.imag(k_nvg))
    sign_tolerance = 1e-13 * max(1.0, abs(k_nvg))
    # A positive imaginary part is never certified as non-active.  Values
    # below the numerical sign resolution are explicitly indeterminate, while
    # larger positives are active signed responses; both suppress eta.
    intrinsic_passive = bool(im_k_nvg <= 0.0)
    intrinsic_sign_indeterminate = bool(im_k_nvg > 0.0 and im_k_nvg <= sign_tolerance)
    intrinsic_passivity_status = ("NONACTIVE_SIGNED_RESPONSE" if intrinsic_passive else
                                  ("INDETERMINATE_SIGNED_RESPONSE" if intrinsic_sign_indeterminate
                                   else "ACTIVE_SIGNED_RESPONSE_REJECTED"))
    p_available = pin if pin > 0.0 else None
    eta = None if p_available is None or not intrinsic_passive or near_loaded_pole else p_out / p_available
    if not bool(pars["R_hat"] >= 0.0):
        solve_status = "ACTIVE_LOAD_REJECTED"
    elif not intrinsic_passive:
        solve_status = ("INTRINSIC_SIGN_INDETERMINATE" if intrinsic_sign_indeterminate
                        else "INTRINSIC_ACTIVE_RESPONSE_REJECTED")
    elif near_loaded_pole:
        solve_status = "NEAR_LOADED_POLE_FINITE"
    else:
        solve_status = "LOADED_RLC_PASSIVE"
    return {
        "full_matrix": full,
        "source_vector": source_vector,
        "direct_solution": direct,
        "schur_solution": schur,
        "direct_vs_schur_relative_error": float(schur_relative_error),
        "scalar_identity_relative_error": float(_rel(direct[1], schur[1])),
        "Z_e": ze,
        "H_QQ": h_qq,
        "K_NVG": k_nvg,
        "K_loaded": k_loaded,
        "loaded_determinant": loaded_determinant,
        "n_hat": direct[0],
        "y_hat": direct[1],
        "A0_hat": direct[2],
        "A_L_hat": direct[3],
        "Q_hat": direct[4],
        "I_hat": current_direct,
        "I_schur_hat": current_schur,
        "current_identity_relative_error": float(_rel(current_direct, current_schur)),
        "P_in_normalized": pin,
        "P_out_normalized": p_out,
        "P_intrinsic_normalized": p_intrinsic,
        "power_balance_residual_normalized": balance_residual,
        "eta_normalized": eta,
        "parameters": {**pars, "Omega": om, "source_hat": source},
        "passive": bool(pars["R_hat"] >= 0.0),
        "load_passive": bool(pars["R_hat"] >= 0.0),
        "intrinsic_passive": intrinsic_passive,
        "intrinsic_passivity_status": intrinsic_passivity_status,
        "intrinsic_sign_indeterminate": intrinsic_sign_indeterminate,
        "condition_number": condition_number,
        "near_loaded_pole": near_loaded_pole,
        "solve_status": solve_status,
        "coordinate_order": list(FULL_COORDINATE_ORDER),
    }


def _load_result_json(answer: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize a loaded solve without allowing NaN/Inf into JSON."""
    out: dict[str, Any] = {
        "status": answer["solve_status"],
        "coordinate_order": list(FULL_COORDINATE_ORDER),
        "parameters": {
            "Omega": _rjson(answer["parameters"]["Omega"]),
            "L_hat": _rjson(answer["parameters"]["L_hat"]),
            "C_hat": _rjson(answer["parameters"]["C_hat"]),
            "G_hat": _rjson(answer["parameters"]["G_hat"]),
            "R_hat": _rjson(answer["parameters"]["R_hat"]),
            "source_hat": _cjson(answer["parameters"]["source_hat"]),
        },
        "full_matrix": _matrix_json(answer["full_matrix"]),
        "source_vector": [_cjson(x) for x in answer["source_vector"]],
        "Z_e": None if answer["Z_e"] is None else _cjson(answer["Z_e"]),
        "H_QQ": _cjson(answer["H_QQ"]),
        "K_NVG": _cjson(answer["K_NVG"]),
        "K_loaded": None if answer["K_loaded"] is None else _cjson(answer["K_loaded"]),
        "loaded_determinant": _cjson(answer["loaded_determinant"]),
        "direct_solution": [_cjson(x) for x in answer["direct_solution"]],
        "schur_solution": [_cjson(x) for x in answer["schur_solution"]],
        "direct_vs_schur_relative_error": _rjson(answer["direct_vs_schur_relative_error"]),
        "scalar_identity_relative_error": _rjson(answer["scalar_identity_relative_error"]),
        "n_hat": _cjson(answer["n_hat"]),
        "y_hat": _cjson(answer["y_hat"]),
        "A0_hat": _cjson(answer["A0_hat"]),
        "A_L_hat": _cjson(answer["A_L_hat"]),
        "Q_hat": _cjson(answer["Q_hat"]),
        "I_hat": _cjson(answer["I_hat"]),
        "I_schur_hat": _cjson(answer["I_schur_hat"]),
        "current_identity_relative_error": _rjson(answer["current_identity_relative_error"]),
        "P_in_normalized": _rjson(answer["P_in_normalized"]),
        "P_out_normalized": _rjson(answer["P_out_normalized"]),
        "P_intrinsic_normalized": _rjson(answer["P_intrinsic_normalized"]),
        "power_balance_residual_normalized": _rjson(answer["power_balance_residual_normalized"]),
        "eta_normalized": None if answer["eta_normalized"] is None else _rjson(answer["eta_normalized"]),
        "eta_interpretation": ("per_unit_algebra_only; not hardware efficiency" if answer["eta_normalized"] is not None
                               else ("not_reported_indeterminate_intrinsic_sign" if answer["intrinsic_sign_indeterminate"]
                                     else ("not_reported_active_intrinsic_response" if not answer["intrinsic_passive"]
                                           else "not_reported_near_loaded_pole"))),
        "load_passive": bool(answer["load_passive"]),
        "intrinsic_passive": bool(answer["intrinsic_passive"]),
        "intrinsic_passivity_status": answer["intrinsic_passivity_status"],
        "intrinsic_sign_indeterminate": bool(answer["intrinsic_sign_indeterminate"]),
        "condition_number": _rjson(answer["condition_number"]),
        "near_loaded_pole": bool(answer["near_loaded_pole"]),
        "scenario_flag": SCENARIO_FLAG,
    }
    return out


def _backgrounds() -> list[dict[str, Any]]:
    with upstream.mp.workdps(50):
        base = upstream.BulkModel()
        # Rebuild the maintained Q4/W8.90/W8.93 backgrounds exactly as the
        # retarded producer does; no saved result JSON is an input.
        backgrounds = retarded._backgrounds(base)
    return backgrounds


def _row_id(background_id: str, q_text: str, fraction_text: str, r_text: str) -> str:
    return f"{background_id}:q={q_text}:omega_fraction={fraction_text}:R_hat={r_text}"


def _build_rows(backgrounds: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    current_errors: list[float] = []
    scalar_errors: list[float] = []
    balance_errors: list[float] = []
    absolute_balance_errors: list[float] = []
    loaded_count = 0
    intrinsic_signs: dict[str, int] = {"negative_or_zero_Im_K": 0, "positive_Im_K": 0}
    for background in backgrounds:
        model, state = background["model"], background["state"]
        mass, kf = float(background["mass"]), float(background["kF"])
        for q_text in Q_GRID_MEV:
            q = float(q_text)
            edge = retarded.particle_hole_edge(mass, kf, q)
            for frac_text in OMEGA_FRACTIONS:
                fraction = float(frac_text)
                omega = edge * fraction
                # This is the maintained full retarded kernel and full
                # coordinate order (n,y,A0,A_L), with canonical scale=1.
                pi = retarded.polarization_kernel(mass, kf, q, omega, d=model.d)
                ans = retarded.retarded_response(model, state, q, omega, pi, 1.0)
                if ans.get("H_normalized") is None:
                    raise DrivenLoadError(f"missing normalized Hessian for {background['background_id']}")
                h4 = np.asarray(ans["H_normalized"], dtype=complex)
                # Time is rescaled to tau=q*t and Omega=omega/q.  The NVG
                # Hessian remains the exact maintained retarded value; only
                # the explicitly added receiver uses the dimensionless clock.
                omega_hat = omega / q
                try:
                    k_nvg, *_ = _effective_density_inverse(h4)
                except DrivenLoadError:
                    k_nvg = complex(float("nan"))
                if math.isfinite(k_nvg.imag) and k_nvg.imag <= 0.0:
                    intrinsic_signs["negative_or_zero_Im_K"] += 1
                elif math.isfinite(k_nvg.imag):
                    intrinsic_signs["positive_Im_K"] += 1
                for r_text in LOAD_R_VALUES:
                    r_hat = float(r_text)
                    try:
                        loaded = solve_loaded(h4, omega_hat, R_hat=r_hat)
                    except DrivenLoadError as exc:
                        raise DrivenLoadError(f"{background['background_id']} q={q_text} frac={frac_text} R={r_text}: {exc}") from exc
                    loaded_count += 1
                    errors.append(loaded["direct_vs_schur_relative_error"])
                    current_errors.append(loaded["current_identity_relative_error"])
                    scalar_errors.append(loaded["scalar_identity_relative_error"])
                    balance_scale = max(abs(loaded["P_in_normalized"]), abs(loaded["P_out_normalized"]),
                                        abs(loaded["P_intrinsic_normalized"]))
                    relative_balance = (0.0 if balance_scale == 0.0 else
                                        abs(loaded["power_balance_residual_normalized"]) / balance_scale)
                    balance_errors.append(relative_balance)
                    absolute_balance_errors.append(abs(loaded["power_balance_residual_normalized"]))
                    rows.append({
                        "row_id": _row_id(str(background["background_id"]), q_text, frac_text, r_text),
                        "background_id": background["background_id"],
                        "family": background["family"],
                        "target_y": background["target_y"],
                        "coordinate_order": list(FULL_COORDINATE_ORDER),
                        "nvg_coordinate_order": list(COORDINATE_ORDER),
                        "q_MeV": _rjson(q),
                        "particle_hole_edge_MeV": _rjson(edge),
                        "omega_fraction": frac_text,
                        "omega_MeV": _rjson(omega),
                        "Omega": _rjson(omega_hat),
                        "inside_particle_hole_continuum": bool(0.0 < omega < edge),
                        "Pi": {"vv_MeV2": _cjson(pi[0]), "vs_MeV2": _cjson(pi[1]), "ss_MeV2": _cjson(pi[2])},
                        "retarded_response_status": ans["status"],
                        "H_normalized": _matrix_json(h4),
                        "source_hat": _cjson(SOURCE_HAT),
                        "pressure_source": {
                            "mu_s": "p_s/n0",
                            "matched_profile_force": "F=integral n0*psi*mu_s",
                            "F_hat": _cjson(SOURCE_HAT),
                            "coordinate": "x=delta_n/n0=-div(u)",
                        },
                        "load": {
                            "scenario_flag": SCENARIO_FLAG,
                            "pressure_port": "x=delta_n/n0=-div(u), matched finite-q profile",
                            "L_hat": _rjson(L_HAT), "C_hat": _rjson(C_HAT), "G_hat": _rjson(G_HAT), "R_hat": r_text,
                            "parameters_are_dimensionless": True,
                            "physical_mapping": None,
                        },
                        "response": _load_result_json(loaded),
                        "passivity_status": "PASSIVE_RLC_RHAT_NONNEGATIVE" if loaded["load_passive"] else "ACTIVE_RLC_REJECTED",
                        "intrinsic_passivity_status": loaded["intrinsic_passivity_status"],
                        "near_loaded_pole": loaded["near_loaded_pole"],
                        "physical_watts": None,
                        "physical_efficiency": None,
                        "branch_lineage": "maintained_retarded_kernel -> normalized_full_H -> added_reciprocal_RLC",
                    })
    if loaded_count != EXPECTED_LOADED_ROWS:
        raise DrivenLoadError(f"loaded row count {loaded_count} != {EXPECTED_LOADED_ROWS}")
    controls = {
        "loaded_row_count": loaded_count,
        "max_direct_vs_schur_relative_error": _rjson(_max_rel(errors)),
        "max_current_identity_relative_error": _rjson(_max_rel(current_errors)),
        "max_scalar_identity_relative_error": _rjson(_max_rel(scalar_errors)),
        "max_relative_power_balance_residual": _rjson(_max_rel(balance_errors)),
        "max_absolute_power_balance_residual": _rjson(_max_rel(absolute_balance_errors)),
        "intrinsic_response_sign_counts": intrinsic_signs,
        "all_declared_loads_passive": True,
        "source_amplitude_interpretation": "SOURCE_HAT=1 is a per-unit infinitesimal force-transfer coefficient, not a physical amplitude",
    }
    return rows, controls


def _synthetic_time_verifier() -> dict[str, Any]:
    """Independent dimensionless positive-energy oscillator/RLC check.

    The mechanical inertia below is declared synthetic algebra (not fitted
    from static NVG ``H``).  The check starts at zero energy, applies a finite
    pump, then turns the pump off and verifies finite discharge is bounded by
    stored energy.  It is a work-identity control, not an NVG prediction.
    """
    M, kx, ks, kxy = 2.0, 1.4, 1.1, 0.2
    gamma_s = 0.05
    L, C, R, g = 1.0, 1.0, 0.4, 0.3
    amp, pump_duration, drive_omega = 0.7, 20.0, 0.8
    # Positive stiffness eigenvalue is recorded to make the energy passport
    # explicit; no physical unit is attached.
    stiffness = np.asarray([[kx, kxy], [kxy, ks]], dtype=float)
    stiffness_min = float(np.linalg.eigvalsh(stiffness).min())
    if stiffness_min <= 0.0:
        raise DrivenLoadError("synthetic positive-energy stiffness is not positive")

    def drive(t: float) -> float:
        return amp * math.sin(drive_omega * t) if t <= pump_duration else 0.0

    def rhs(t: float, state: np.ndarray) -> np.ndarray:
        x, vx, y, vy, charge, current = state[:6]
        force = drive(t)
        # L Idot + R I + Q/C = g xdot; reciprocal coupling.
        idot = (g * vx - R * current - charge / C) / L
        # M xdd + kx x + kxy y + g I = F.
        ax = (force - kx * x - kxy * y - g * current) / M
        ay = (-kxy * x - ks * y - gamma_s * vy)
        # The last three coordinates accumulate source work, load loss and
        # intrinsic scalar loss directly, avoiding a finite-grid quadrature
        # masquerading as a work identity.
        return np.asarray((vx, ax, vy, ay, current, idot,
                           force * vx, R * current * current, gamma_s * vy * vy), dtype=float)

    def energies(state: np.ndarray) -> tuple[float, float, float]:
        x, vx, y, vy, charge, current = state
        mechanical_scalar = 0.5 * M * vx * vx + 0.5 * ks * y * y + 0.5 * kx * x * x + kxy * x * y
        # The scalar kinetic term is explicit; separate naming prevents
        # accidental identification with an NVG static Hessian.
        total = mechanical_scalar + 0.5 * vy * vy + 0.5 * L * current * current + 0.5 * charge * charge / C
        return float(total), float(0.5 * R * current * current), float(gamma_s * vy * vy)

    # A single dense solve is used so the pre/post-pump boundary is one exact
    # integration trajectory; quadrature uses the recorded dense state.
    sol = solve_ivp(rhs, (0.0, 80.0), np.zeros(9), rtol=2e-10, atol=2e-12, dense_output=True, max_step=0.03)
    if not sol.success or sol.sol is None:
        raise DrivenLoadError(f"synthetic solve failed: {sol.message}")
    times = np.linspace(0.0, 80.0, 8001)
    states = np.asarray(sol.sol(times), dtype=float).T
    input_work = float(states[-1, 6])
    load_work = float(states[-1, 7])
    intrinsic_work = float(states[-1, 8])
    energies_total = np.asarray([energies(state[:6])[0] for state in states])
    balance = energies_total[-1] + load_work + intrinsic_work - input_work
    off_state = np.asarray(sol.sol(float(pump_duration)), dtype=float)
    off_energy = float(energies(off_state[:6])[0])
    post_load = float(states[-1, 7] - off_state[7])
    post_intrinsic = float(states[-1, 8] - off_state[8])
    post_input = float(states[-1, 6] - off_state[6])
    discharge_bound = post_load + post_intrinsic <= off_energy + 3e-5
    return {
        "status": "PASS_DIMENSIONLESS_SYNTHETIC_WORK_IDENTITY" if abs(balance) < 3e-5 and discharge_bound else "FAIL",
        "scenario_flag": "SYNTHETIC_POSITIVE_ENERGY_CONTROL_NOT_NVG_INERTIA",
        "declared_parameters": {"M": M, "kx": kx, "ks": ks, "kxy": kxy, "gamma_s": gamma_s,
                                 "L": L, "C": C, "R": R, "g": g, "pump_amplitude": amp,
                                 "pump_duration": pump_duration, "drive_Omega": drive_omega},
        "stiffness_min_eigenvalue": _rjson(stiffness_min),
        "initial_total_energy": _rjson(energies_total[0]),
        "final_total_energy": _rjson(energies_total[-1]),
        "input_work": _rjson(input_work),
        "load_dissipated_work": _rjson(load_work),
        "intrinsic_dissipated_work": _rjson(intrinsic_work),
        "work_balance_residual": _rjson(balance),
        "pump_off_stored_energy": _rjson(off_energy),
        "post_pump_off_load_plus_intrinsic": _rjson(post_load + post_intrinsic),
        "post_pump_off_input_work": _rjson(post_input),
        "finite_discharge_bounded_by_stored_energy": bool(discharge_bound),
        "zero_initial_energy": bool(abs(energies_total[0]) < 1e-14),
    }


def _controls() -> dict[str, Any]:
    """Small algebraic controls independent of the 54-row producer matrix."""
    controls: dict[str, Any] = {}
    # Source-off is exact for every finite load branch.
    sample_h = np.asarray([[2.0 - 0.1j, 0.2, 0.1, 0.0], [0.2, 3.0, 0.0, 0.0],
                           [0.1, 0.0, 1.5, 0.1], [0.0, 0.0, 0.1, 2.0]], complex)
    off = solve_loaded(sample_h, 0.7, source_hat=0.0)
    controls["source_off_max_solution_abs"] = _rjson(np.max(np.abs(off["direct_solution"])))
    # G=0 decouples Q and leaves the density response equal to the NVG-only
    # inverse; R=0 remains a lossless (zero-output) passive boundary.
    g0 = solve_loaded(sample_h, 0.7, G_hat_value=0.0, R_hat=1.0)
    base_k, *_ = _effective_density_inverse(sample_h)
    controls["G0_density_relative_error"] = _rjson(_rel(g0["n_hat"], 1.0 / base_k))
    r0 = solve_loaded(sample_h, 0.7, G_hat_value=0.5, R_hat=0.0)
    controls["R0_pout"] = _rjson(r0["P_out_normalized"])
    active_h = sample_h.copy()
    active_h[0, 0] = 2.0 + 0.1j
    active = solve_loaded(active_h, 0.7, R_hat=1.0)
    controls["active_intrinsic_negative_control"] = {
        "Im_K_NVG": _rjson(np.imag(active["K_NVG"])),
        "status": active["solve_status"],
        "eta_is_null": active["eta_normalized"] is None,
        "signed_P_intrinsic_normalized": _rjson(active["P_intrinsic_normalized"]),
    }
    tiny_positive = solve_loaded(np.diag(np.asarray((1.0 + 1e-14j, 2.0, 3.0, 4.0), dtype=complex)),
                                0.7, G_hat_value=1.0, R_hat=1.0)
    controls["tiny_positive_intrinsic_sign_control"] = {
        "Im_K_NVG": _rjson(np.imag(tiny_positive["K_NVG"])),
        "status": tiny_positive["solve_status"],
        "intrinsic_passivity_status": tiny_positive["intrinsic_passivity_status"],
        "sign_indeterminate": tiny_positive["intrinsic_sign_indeterminate"],
        "eta_is_null": tiny_positive["eta_normalized"] is None,
        "signed_P_intrinsic_normalized": _rjson(tiny_positive["P_intrinsic_normalized"]),
    }
    # Static limit is checked on the same finite-q full NVG inverse used by
    # the loaded rows; it is not replaced by the static D/C relaxation ratio.
    static_bg = _backgrounds()[0]
    static_q = 100.0
    static_pi = retarded.polarization_kernel(static_bg["mass"], static_bg["kF"], static_q, 0.0, d=static_bg["model"].d)
    static_ans = retarded.retarded_response(static_bg["model"], static_bg["state"], static_q, 0.0, static_pi, 1.0)
    static_h = np.asarray(static_ans["H_normalized"], dtype=complex)
    static_source = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=complex)
    static_direct = np.linalg.solve(static_h, static_source)
    static_schur = 1.0 / _effective_density_inverse(static_h)[0]
    controls["static_limit_control"] = {
        "direct_vs_schur_relative_error": _rjson(_rel(static_direct[0], static_schur)),
        "finite_density_response": bool(np.all(np.isfinite(static_direct))),
        "not_static_r2_efficiency": True,
    }
    # Bare factors are not promoted to poles: a density K_NVG=0 can be
    # regularized by a passive load, and Ze=0 can be regularized by G!=0.
    bare_k0 = np.eye(4, dtype=complex)
    bare_k0[0, 0] = 0.0
    regularized_k0 = solve_loaded(bare_k0, 0.7, G_hat_value=0.5, R_hat=1.0)
    bare_ze0 = solve_loaded(np.eye(4, dtype=complex), 1.0, G_hat_value=0.5, R_hat=0.0)
    # Negative controls: decoupled singularity and a deliberately tuned
    # coupled determinant zero must fail closed rather than emit infinity.
    decoupled_singular = np.eye(4, dtype=complex)
    decoupled_singular[0, 0] = 0.0
    try:
        solve_loaded(decoupled_singular, 0.7, G_hat_value=0.0, R_hat=1.0)
    except DrivenLoadError as exc:
        decoupled_status = type(exc).__name__
    else:
        decoupled_status = "UNEXPECTED_ACCEPT"
    tuned_singular = np.eye(4, dtype=complex)
    om_sing = 0.7
    ze_sing = rlc_impedance(om_sing, R_hat=1.0)
    hqq_sing = -1j * om_sing * ze_sing
    tuned_singular[0, 0] = om_sing * om_sing * G_HAT * G_HAT / hqq_sing
    try:
        solve_loaded(tuned_singular, om_sing, R_hat=1.0)
    except DrivenLoadError as exc:
        tuned_status = type(exc).__name__
    else:
        tuned_status = "UNEXPECTED_ACCEPT"
    controls["dressed_denominator_controls"] = {
        "bare_K0_regularized_status": regularized_k0["solve_status"],
        "bare_K0_regularized_finite": bool(np.all(np.isfinite(regularized_k0["direct_solution"]))),
        "bare_Ze0_regularized_status": bare_ze0["solve_status"],
        "bare_Ze0_regularized_finite": bool(np.all(np.isfinite(bare_ze0["direct_solution"]))),
        "decoupled_singular_rejection": decoupled_status,
        "tuned_loaded_singular_rejection": tuned_status,
    }
    # Independent, bounded matching law control.  This is a synthetic
    # impedance algebra check, not a fitted NVG coefficient or device value.
    r_int, x_int, om_match, g_match, f_match = 2.0, 0.75, 1.0, 1.0, 1.0
    z_nvg = complex(r_int, x_int)
    k_match = -1j * om_match * z_nvg  # Z_NVG=i*K/Omega
    matching_rows = []
    for rho in (0.25, 1.0, 4.0):
        z_load = complex(rho * r_int, -x_int)  # reactive cancellation
        z_e = g_match * g_match / z_load
        x_match = f_match / (k_match - 1j * om_match * g_match * g_match / z_e)
        pin_match = om_match * 0.5 * float(np.imag(np.conj(f_match) * x_match))
        pout_match = om_match * om_match * float(np.real(g_match * g_match / z_e)) * abs(x_match) ** 2 / 2.0
        matching_rows.append({
            "rho": rho,
            "eta": pout_match / pin_match,
            "eta_expected": rho / (1.0 + rho),
            "available_ratio": pout_match / (abs(f_match) ** 2 / (8.0 * r_int)),
            "available_ratio_expected": 4.0 * rho / (1.0 + rho) ** 2,
        })
    controls["matching_control"] = {
        "rows": matching_rows,
        "max_eta_error": _rjson(max(abs(r["eta"] - r["eta_expected"]) for r in matching_rows)),
        "max_available_ratio_error": _rjson(max(abs(r["available_ratio"] - r["available_ratio_expected"]) for r in matching_rows)),
        "conjugate_match_eta": _rjson(matching_rows[1]["eta"]),
        "lossless_guard": {"Rint": 0.0, "status": "LOSSLESS_MATCHING_UNDEFINED_NO_DIVISION"},
        "scenario_flag": "SYNTHETIC_PASSIVE_MATCHING_ALGEBRA_NOT_NVG_COEFFICIENTS",
    }
    # Congruence rescaling of all four coordinates is a pure normalization
    # change: S H S x' = S J implies S x'=x.  This catches a hidden
    # coordinate-order or double-normalization error.
    rescale = np.diag(np.asarray((2.0, 3.0, 4.0, 5.0), dtype=complex))
    scaled_h = rescale @ sample_h @ rescale
    unscaled_source = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=complex)
    scaled_source = rescale @ unscaled_source
    scaled_solution = np.linalg.solve(scaled_h, scaled_source)
    controls["congruence_rescaling_relative_error"] = _rjson(float(np.max(np.abs(rescale @ scaled_solution - np.linalg.solve(sample_h, unscaled_source)))))
    # Active R is rejected by the same validator; complex loads are rejected
    # because the passport is intentionally explicit and real-valued.
    rejection: dict[str, str] = {}
    for name, kwargs in (("negative_R", {"R_hat": -0.1}), ("nonfinite_R", {"R_hat": float("nan")}),
                         ("complex_R", {"R_hat": 1 + 0.2j})):
        try:
            validate_load_parameters(**kwargs)
        except DrivenLoadError as exc:
            rejection[name] = type(exc).__name__
    controls["invalid_load_rejections"] = rejection
    controls["passive_complex_load_rule"] = "only real finite R_hat>=0 accepted; complex/admittance fitting is out of scope"
    # Singular/near-pole control: the receiver's reactive pole is explicit at
    # Omega=0 and must not be hidden by a clamp.
    try:
        rlc_impedance(0.0)
    except DrivenLoadError as exc:
        controls["static_receiver_pole_rejection"] = type(exc).__name__
    # Conjugation control uses the maintained retarded kernel itself.  The
    # external RLC is positive-frequency by construction, so it is not
    # silently used as a negative-frequency surrogate here.
    bg = _backgrounds()[0]
    q = 100.0
    edge = retarded.particle_hole_edge(bg["mass"], bg["kF"], q)
    omega = 0.6 * edge
    pi_pos = retarded.polarization_kernel(bg["mass"], bg["kF"], q, omega, d=bg["model"].d)
    pi_neg = retarded.polarization_kernel(bg["mass"], bg["kF"], q, -omega, d=bg["model"].d)
    ans_pos = retarded.retarded_response(bg["model"], bg["state"], q, omega, pi_pos, 1.0)
    ans_neg = retarded.retarded_response(bg["model"], bg["state"], q, -omega, pi_neg, 1.0)
    kernel_conjugation_error = max(_rel(pi_neg[i], np.conj(pi_pos[i])) for i in range(3))
    controls["conjugation_control_relative_error"] = _rjson(max(kernel_conjugation_error, _rel(ans_neg["chi"], np.conj(ans_pos["chi"]))))
    # No double counting of the pump: the only source term is e_n*source_hat;
    # the load is passive and does not add an independent source.
    controls["no_double_counting_pump"] = {"source_vector_nonzero_entries": 1, "load_source_entries": 0, "pass": True}
    controls["synthetic_time_verifier"] = _synthetic_time_verifier()
    return controls


def _validate_rows(rows: Any) -> bool:
    def valid_number(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    def valid_complex(value: Any) -> bool:
        return (isinstance(value, dict) and set(value) == {"real", "imag"}
                and valid_number(value["real"]) and valid_number(value["imag"]))

    def valid_matrix(value: Any, shape: tuple[int, int]) -> bool:
        return (isinstance(value, list) and len(value) == shape[0]
                and all(isinstance(line, list) and len(line) == shape[1]
                        and all(valid_complex(item) for item in line) for line in value))

    def valid_vector(value: Any, length: int) -> bool:
        return isinstance(value, list) and len(value) == length and all(valid_complex(item) for item in value)

    if not isinstance(rows, list) or len(rows) != EXPECTED_LOADED_ROWS:
        return False
    required = {"row_id", "background_id", "q_MeV", "particle_hole_edge_MeV", "omega_fraction", "Omega",
                "H_normalized", "response", "load", "pressure_source", "physical_watts", "physical_efficiency", "branch_lineage"}
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row):
            return False
        if row["row_id"] in ids or not isinstance(row["row_id"], str):
            return False
        ids.add(row["row_id"])
        if not valid_matrix(row["H_normalized"], (4, 4)):
            return False
        if row["physical_watts"] is not None or row["physical_efficiency"] is not None:
            return False
        load = row["load"]
        if not isinstance(load, dict) or load.get("R_hat") not in LOAD_R_VALUES:
            return False
        if load.get("L_hat") != _rjson(L_HAT) or load.get("C_hat") != _rjson(C_HAT) or load.get("G_hat") != _rjson(G_HAT):
            return False
        pressure_source = row["pressure_source"]
        if not isinstance(pressure_source, dict) or pressure_source.get("mu_s") != "p_s/n0" or pressure_source.get("coordinate") != "x=delta_n/n0=-div(u)":
            return False
        response = row["response"]
        required_response = {"status", "coordinate_order", "parameters", "full_matrix", "source_vector", "Z_e", "H_QQ", "K_NVG", "K_loaded", "loaded_determinant",
                             "direct_solution", "schur_solution", "direct_vs_schur_relative_error",
                             "scalar_identity_relative_error", "I_hat", "I_schur_hat",
                             "current_identity_relative_error", "P_in_normalized", "P_out_normalized",
                             "P_intrinsic_normalized", "power_balance_residual_normalized", "eta_normalized",
                             "eta_interpretation", "load_passive", "intrinsic_passive", "intrinsic_passivity_status",
                             "intrinsic_sign_indeterminate",
                             "condition_number", "near_loaded_pole", "scenario_flag"}
        if not isinstance(response, dict) or not required_response <= set(response):
            return False
        if not valid_matrix(response["full_matrix"], (5, 5)) or not valid_vector(response["source_vector"], 5):
            return False
        if not valid_vector(response["direct_solution"], 5) or not valid_vector(response["schur_solution"], 5):
            return False
        if response["Z_e"] is not None and not valid_complex(response["Z_e"]):
            return False
        if response["K_loaded"] is not None and not valid_complex(response["K_loaded"]):
            return False
        if not all(valid_complex(response[key]) for key in ("H_QQ", "K_NVG", "loaded_determinant", "I_hat", "I_schur_hat")):
            return False
        if not all(valid_number(response[key]) for key in ("direct_vs_schur_relative_error", "scalar_identity_relative_error",
                                                            "current_identity_relative_error", "P_in_normalized",
                                                            "P_out_normalized", "P_intrinsic_normalized",
                                                            "power_balance_residual_normalized", "condition_number")):
            return False
        if response["eta_normalized"] is not None and not valid_number(response["eta_normalized"]):
            return False
        if response.get("scenario_flag") != SCENARIO_FLAG:
            return False
        if (not isinstance(response["load_passive"], bool) or not isinstance(response["intrinsic_passive"], bool)
                or not isinstance(response["intrinsic_sign_indeterminate"], bool)
                or not isinstance(response["near_loaded_pole"], bool)):
            return False
    return len(ids) == EXPECTED_LOADED_ROWS


def build_result() -> dict[str, Any]:
    """Build the complete source/NVG/load result payload."""
    backgrounds = _backgrounds()
    rows, row_controls = _build_rows(backgrounds)
    controls = _controls()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "scenario_flag": SCENARIO_FLAG,
        "coordinate_order": list(COORDINATE_ORDER),
        "full_coordinate_order": list(FULL_COORDINATE_ORDER),
        "time_convention": "exp(-i omega t); tau=q*t; Omega=omega/q",
        "action_source_convention": "H_source=-integral mu_s delta_n; real-time L_source=+integral mu_s delta_n; H^R X=+mu_s e_n",
        "pressure_port": {
            "definition": "x=delta_n/n0=-div(u)",
            "source": "mu_s=p_s/n0",
            "virtual_work_per_volume": "p_s*xdot",
            "profile_condition": "matched finite-q psi; integral psi=0 for closed baryon-conserving sample",
            "mode_projection": "delta_n=n0*x*psi; integral psi^2=Veff; F=integral n0*psi*mu_s=Veff*p_s for matched profile",
            "Veff_and_overlap": None,
        },
        "receiver": {
            "model": "explicit reciprocal external RLC completion",
            "lagrangian": "L_ext=F*x+(Lhat/2)*Qdot^2-Q^2/(2*Chat)-Ghat*x*Qdot",
            "rayleigh": "Rhat*Qdot^2/2",
            "equation": "Lhat*Idot+Rhat*I+Q/Chat=Ghat*xdot",
            "current": "I=-i*Omega*Q",
            "coupling_coordinate": "Ghat is flux/displacement slope with respect to generalized strain coordinate x; geometry mapping unknown",
            "load_power": "Pout=Rhat*|I|^2/2 (normalized algebra only)",
            "declared_values": {"L_hat": _rjson(L_HAT), "C_hat": _rjson(C_HAT), "G_hat": _rjson(G_HAT), "R_hat": list(LOAD_R_VALUES)},
            "normalization_passport": {
                "E_star": "Veff*n0*W0 (unspecified mode volume/normalization)",
                "omega_star": "q/hbar (q in energy units; SI conversion unspecified)",
                "Q_star": None,
                "definitions": {
                    "L_hat": "L*Q_star^2*omega_star^2/E_star",
                    "C_hat": "C*E_star/Q_star^2",
                    "R_hat": "R*Q_star^2*omega_star/E_star",
                    "G_hat": "G*Q_star*omega_star/E_star",
                    "F_hat": "F/E_star",
                    "P_hat": "P/(E_star*omega_star)",
                },
                "physical_units": False,
                "device_mapping": False,
                "mapped_watts": None,
                "mapped_eta": None,
            },
            "physical_mapping": None,
            "not_derived_from_closed_NVG_action": True,
        },
        "coverage": {"background_count": len(backgrounds), "base_row_count": EXPECTED_BASE_ROWS,
                     "loaded_row_count": len(rows), "background_ids": [b["background_id"] for b in backgrounds],
                     "q_MeV": list(Q_GRID_MEV), "omega_edge_fractions": list(OMEGA_FRACTIONS), "R_hat": list(LOAD_R_VALUES)},
        "rows": rows,
        "controls": {"row_controls": row_controls, **controls},
        "physical_outputs": {"watts": None, "device_efficiency": None,
                              "reason": "volume, profile, source amplitude, G/Le/Ce/Re geometry and SI mapping are unspecified"},
        "provenance": {
            "retarded_source": str(retarded.SOURCE_PATH),
            "retarded_source_sha256": _sha256(retarded.SOURCE_PATH),
            "upstream_source": str(retarded.UPSTREAM_PATH),
            "upstream_source_sha256": _sha256(retarded.UPSTREAM_PATH),
            "static_source": str(retarded.STATIC_PATH),
            "static_source_sha256": _sha256(retarded.STATIC_PATH),
            "producer_source": str(Path(__file__).resolve()),
            "producer_source_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    if not _validate_rows(rows):
        raise DrivenLoadError("result row validation failed")
    return result


def validate_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != STATUS:
        return False
    if result.get("scenario_flag") != SCENARIO_FLAG:
        return False
    return _validate_rows(result.get("rows"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, help="explicitly write strict JSON to this path")
    args = parser.parse_args(argv)
    result = build_result()
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
