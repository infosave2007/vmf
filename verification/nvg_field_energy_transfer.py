#!/usr/bin/env python3
"""Live NVG static density--scalar field-energy transfer calculation.

This is a small, source-derived calculation rather than a device model.  It
rebuilds the maintained zero-temperature Q4 branch and the two existing W8
inverse-design branches, evaluates the maintained finite-q Dirac kernel, and
uses the maintained vector-constraint Schur reduction.  The resulting
quadratic coefficients quantify redistribution of an *already supplied*
density perturbation.  They do not define a source, a load, a volume, a
damping law, power, efficiency, or an autonomous reservoir.

The producer never reads a saved result.  By default it emits strict JSON to
stdout only; ``--output`` is an explicit opt-in result write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import mpmath as mp

# Keep direct-script imports deterministic and avoid creating bytecode as an
# incidental side effect of the source-only CLI.
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:  # direct script and package-style imports
    import nvg_nonlocal_response as maintained
except ImportError:  # pragma: no cover - package import support
    from . import nvg_nonlocal_response as maintained


SOURCE_PATH = Path(__file__).resolve()
MAINTAINED_RESPONSE_PATH = HERE / "nvg_nonlocal_response.py"
MAINTAINED_STATIC_PATH = HERE / "nvg_spatial_identifiability_audit.py"
MAINTAINED_ACTION_PATH = HERE / "source_complete_scaling_saturation_audit.py"
MAINTAINED_CONTRACT_PATH = HERE / "contracts" / "spatial_identifiability.md"

SCHEMA_VERSION = "nvg_field_energy_transfer.v1"
STATUS = "LIVE_STATIC_NVG_FIELD_ENERGY_TRANSFER_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

ANCHORS = ("Q4:n_over_n0=1", "W8:y_star=0.90", "W8:y_star=0.93")
W8_TARGETS = ("0.90", "0.93")
Q_GRID_MEV = ("25", "100", "200", "400")
EXPECTED_BACKGROUND_COUNT = 3
EXPECTED_ROW_COUNT = EXPECTED_BACKGROUND_COUNT * len(Q_GRID_MEV)

# The live source uses high-precision decimal arithmetic.  The second pass is
# a precision control, not a fit and not an alternate parameter set.
PRIMARY_DPS = 60
CONTROL_DPS = 80
PRECISION_RELATIVE_LIMIT = mp.mpf("1e-35")
FINITE_DIFFERENCE_RELATIVE_LIMIT = mp.mpf("1e-24")


class FieldEnergyTransferError(ValueError):
    """Fail-closed error for malformed or unresolved source-derived output."""


def _mp(value: Any, name: str = "value", *, positive: bool = False,
        nonnegative: bool = False) -> mp.mpf:
    if isinstance(value, bool):
        raise FieldEnergyTransferError(f"{name} must be a finite real scalar")
    try:
        result = value if isinstance(value, mp.mpf) else mp.mpf(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise FieldEnergyTransferError(f"{name} must be a finite real scalar") from exc
    if not mp.isfinite(result):
        raise FieldEnergyTransferError(f"{name} must be finite")
    if positive and result <= 0:
        raise FieldEnergyTransferError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise FieldEnergyTransferError(f"{name} must be nonnegative")
    return result


def _number(value: Any, digits: int = 42) -> str:
    result = _mp(value, "derived number")
    return mp.nstr(result, digits)


def _finite_tree(value: Any) -> bool:
    """Return false for every non-finite numeric leaf (including strings)."""

    if isinstance(value, Mapping):
        return all(_finite_tree(key) and _finite_tree(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (mp.mpf, mp.mpc)):
        return bool(mp.isfinite(value))
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str):
        return value.lower() not in {
            "nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity"
        }
    return True


def _shown(value: Any) -> Any:
    """Serialize mpmath values as finite decimal strings, never JSON NaN."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, mp.mpc):
        if not mp.isfinite(value.real) or not mp.isfinite(value.imag):
            raise FieldEnergyTransferError("nonfinite complex output")
        return {"real": _number(value.real), "imag": _number(value.imag)}
    if isinstance(value, mp.mpf):
        return _number(value)
    if isinstance(value, mp.matrix):
        return [[_shown(value[i, j]) for j in range(value.cols)] for i in range(value.rows)]
    if isinstance(value, Mapping):
        return {str(key): _shown(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shown(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FieldEnergyTransferError("nonfinite floating output")
        return value
    return value


def _relative(left: Any, right: Any, *, floor: Any = "1e-50") -> mp.mpf:
    """Relative difference that remains meaningful for small Hessian entries."""

    left, right = _mp(left, "left"), _mp(right, "right")
    return abs(left - right) / max(abs(left), abs(right), _mp(floor, "floor", positive=True))


def _hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FieldEnergyTransferError(f"cannot hash maintained source {path}") from exc


def _source_hashes() -> dict[str, str]:
    return {
        "producer": _hash(MAINTAINED_RESPONSE_PATH),
        "static_schur": _hash(MAINTAINED_STATIC_PATH),
        "action": _hash(MAINTAINED_ACTION_PATH),
        "static_contract": _hash(MAINTAINED_CONTRACT_PATH),
    }


def _direct_solve(matrix: Any) -> tuple[tuple[mp.mpf, ...] | None, str]:
    """Solve the raw mixed-coordinate saddle independently of the Schur code."""

    if matrix is None:
        return None, "DIRECT_SOLVE_NO_MATRIX"
    try:
        solution = mp.lu_solve(matrix, mp.matrix([1, 0, 0]))
        vector = tuple(_mp(solution[i], "direct solution") for i in range(3))
    except (ArithmeticError, ValueError, ZeroDivisionError, TypeError, FieldEnergyTransferError):
        return None, "DIRECT_SOLVE_SINGULAR"
    return vector, "DIRECT_SOLVE_OK"


def _classify(D: mp.mpf, C: mp.mpf, S: mp.mpf | None,
              direct: tuple[mp.mpf, ...] | None) -> tuple[str, bool]:
    if C == 0:
        return "SINGULAR_SCALAR_CURVATURE", False
    if C < 0:
        return "UNSTABLE_SCALAR_CURVATURE", False
    if S is None or S == 0:
        return "SINGULAR_DENSITY_RESPONSE", False
    if S < 0:
        return "UNSTABLE_DENSITY_RESPONSE", False
    if direct is None:
        return "SINGULAR_FULL_HESSIAN", False
    # Preserve the maintained static-response status vocabulary for stable
    # rows; unstable/singular labels above are likewise the source labels.
    return "STABLE_STATIC_TF_RESPONSE", True


def _coerce_reduced(D: Any, B: Any, C: Any) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    return _mp(D, "D"), _mp(B, "B"), _mp(C, "C")


def _complex_mp(value: Any, name: str) -> mp.mpc:
    """Parse a Fourier amplitude without routing high precision through float."""

    if isinstance(value, mp.mpc):
        result = value
    elif isinstance(value, mp.mpf):
        result = mp.mpc(value, 0)
    elif isinstance(value, complex):
        result = mp.mpc(str(value.real), str(value.imag))
    else:
        try:
            result = mp.mpc(str(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise FieldEnergyTransferError(f"{name} must be finite") from exc
    if not mp.isfinite(result.real) or not mp.isfinite(result.imag):
        raise FieldEnergyTransferError(f"{name} must be finite")
    return result


def quadratic_energy(D: Any, B: Any, C: Any, dn: Any, dy: Any) -> mp.mpf:
    """Real quadratic Fourier-mode energy density for complex amplitudes.

    The declared convention is ``E/V = 1/2[D|dn|² + 2B Re(dn* dy) +
    C|dy|²]`` after the static vector constraint has been eliminated.
    """

    D, B, C = _coerce_reduced(D, B, C)
    dn_z, dy_z = _complex_mp(dn, "dn"), _complex_mp(dy, "dy")
    return D * abs(dn_z) ** 2 / 2 + B * mp.re(mp.conj(dn_z) * dy_z) + C * abs(dy_z) ** 2 / 2


def rescale_reduced_coefficients(D: Any, B: Any, C: Any,
                                 density_scale: Any, scalar_scale: Any) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Apply ``dn'=density_scale*dn, dy'=scalar_scale*dy`` by congruence."""

    D, B, C = _coerce_reduced(D, B, C)
    sn = _mp(density_scale, "density_scale", positive=True)
    sy = _mp(scalar_scale, "scalar_scale", positive=True)
    # x = diag(1/sn,1/sy) x'; H' = T^T H T.
    return D / sn**2, B / (sn * sy), C / sy**2


def _transfer_from_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """Derive all transfer quantities, retaining unstable values explicitly."""

    D = _mp(response["D"], "D")
    B = _mp(response["B"], "B")
    C = _mp(response["C"], "C")
    determinant = _mp(response["determinant"], "determinant")
    S = None if C == 0 else D - B**2 / C
    direct, direct_status = _direct_solve(response.get("H"))
    direct_chi = None if direct is None else direct[0]
    chi = None if S in (None, 0) else 1 / S
    direct_schur_error = None
    if direct_chi is not None and chi is not None:
        direct_schur_error = _relative(direct_chi, chi)

    frozen = D / 2
    relaxed = None if S is None else S / 2
    relaxation = None if C == 0 else B**2 / (2 * C)
    dy_dn = None if C == 0 else -B / C
    r2 = None if D == 0 or C == 0 else B**2 / (D * C)
    r2_from_chi = None
    if direct_chi is not None and D != 0 and direct_chi != 0:
        # Independent direct-inverse identity; this is deliberately not
        # clipped to [0,1], including on unstable or negative rows.
        r2_from_chi = 1 - 1 / (D * direct_chi)
    r2_error = None if r2 is None or r2_from_chi is None else _relative(r2, r2_from_chi)
    completion_error = None
    if relaxed is not None and relaxation is not None:
        completion_error = _relative(frozen - relaxed, relaxation)

    status, stable = _classify(D, C, S, direct)
    return {
        "D": D,
        "B": B,
        "C": C,
        "S": S,
        "F": None if C == 0 else D * C - B**2,
        "determinant": determinant,
        "direct_solution": direct,
        "direct_status": direct_status,
        "direct_chi": direct_chi,
        "chi": chi,
        "direct_vs_schur_relative": direct_schur_error,
        "status": status,
        "stable": stable,
        "frozen_coefficient": frozen,
        "relaxed_coefficient": relaxed,
        "relaxation_coefficient": relaxation,
        "dy_dn": dy_dn,
        "r2": r2,
        "r2_from_direct_chi": r2_from_chi,
        "r2_identity_relative": r2_error,
        "completion_square_relative": completion_error,
    }


def reduced_response_from_hessian(coefficients: Mapping[str, Any], Z: Any, q: Any) -> dict[str, Any]:
    """Evaluate maintained static H and return density--scalar transfer data.

    The maintained spatial-response implementation owns the finite-q
    convention and vector Schur elimination.  This wrapper only adds the
    energy bookkeeping and an independent direct solve, preserving negative
    and unstable rows instead of filtering or clipping them.
    """

    if not isinstance(coefficients, Mapping):
        raise FieldEnergyTransferError("coefficients must be a mapping")
    # Keep the derived Schur arithmetic in the same explicit high-precision
    # context as the maintained helper.  Otherwise a focused caller at the
    # ambient 15-digit context could round an otherwise exact identity before
    # the result is returned.
    with mp.workdps(max(mp.mp.dps, CONTROL_DPS)):
        try:
            response = maintained.response_from_hessian(coefficients, Z, q)
        except Exception as exc:  # source API errors are exposed as our fail-closed type
            if isinstance(exc, FieldEnergyTransferError):
                raise
            raise FieldEnergyTransferError(str(exc)) from exc
        if not isinstance(response, Mapping):
            raise FieldEnergyTransferError("maintained response is not a mapping")
        required = ("D", "B", "C", "determinant", "H")
        if not all(key in response for key in required):
            raise FieldEnergyTransferError("maintained response omitted required Schur fields")
        result = _transfer_from_response(response)
        result["H"] = response["H"]
        result["L"] = response.get("L")
        result["q"] = _mp(q, "q", nonnegative=True)
        result["Z"] = _mp(Z, "Z", positive=True)
        return result


# Descriptive aliases for focused callers.
static_field_response = reduced_response_from_hessian
field_energy_transfer = reduced_response_from_hessian


def _live_backgrounds(dps: int) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(dps, int) or not 40 <= dps <= 120:
        raise FieldEnergyTransferError("dps must be an integer in [40,120]")
    with mp.workdps(dps):
        # This guard is maintained-source validation, not an input table.  It
        # rejects drift before any row can be mistaken for a source result.
        base = maintained.upstream.BulkModel()
        guard = maintained._guard(base)
        backgrounds = maintained._backgrounds(base)
        if len(backgrounds) != EXPECTED_BACKGROUND_COUNT:
            raise FieldEnergyTransferError("maintained source returned an unexpected background count")
        expected = set(ANCHORS)
        actual = {str(item.get("background_id")) for item in backgrounds}
        if actual != expected:
            raise FieldEnergyTransferError(f"maintained source background lineage changed: {actual}")
        return base, backgrounds, guard


def _background_public(background: Mapping[str, Any]) -> dict[str, Any]:
    model, state = background["model"], background["state"]
    family = str(background["family"])
    target = background.get("target_y")
    lineage = (
        "BulkModel().equilibrium(BulkModel().n0)"
        if family == "Q4"
        else f"inverse_potential_jet(target_y={target!r}, K_target='240', n_fm3='0.16', binding='-16') -> model.state(model.n0, target_y)"
    )
    return {
        "background_id": background["background_id"],
        "family": family,
        "target_y": target,
        "density_ratio": background.get("density_ratio"),
        "n_fm3": _number(state["n"] / model.hbarc**3),
        "y": _number(state["y"]),
        "W_MeV": _number(model.W0 * state["y"]),
        "mass_MeV": _number(background["mass"]),
        "kF_MeV": _number(background["kF"]),
        "C_y_MeV4": _number(state["C_y"]),
        "B_y": _number(state["B_y"]),
        "D_nn_MeVminus2": _number(state["D"]),
        "branch_lineage": lineage,
        "W8_inverse_targets_are_not_held_out_evidence": bool(family == "W8"),
    }


def _row(background: Mapping[str, Any], q_text: str) -> dict[str, Any]:
    model, state = background["model"], background["state"]
    q = _mp(q_text, "q", positive=True)
    # This is the maintained finite-q kernel and coefficient assembly; no
    # output JSON or private run artifact is read here.
    coefficients = maintained.finite_q_coefficients(model, state, q, "1")
    Z = coefficients.get("Z")
    if Z is None:
        raise FieldEnergyTransferError("maintained finite-q coefficients omitted Z")
    response = reduced_response_from_hessian(coefficients, Z, q)
    return {
        "row_id": f"{background['background_id']}:q={q_text}",
        "background_id": background["background_id"],
        "family": background["family"],
        "target_y": background.get("target_y"),
        "q_MeV": q,
        "Pi_vv_MeV2": coefficients.get("Pi_vv"),
        "Pi_vs_MeV2": coefficients.get("Pi_vs"),
        "Pi_ss_MeV2": coefficients.get("Pi_ss"),
        "coefficients": coefficients,
        "response": response,
    }


def _raw_snapshot(dps: int) -> dict[str, Any]:
    with mp.workdps(dps):
        base, backgrounds, guard = _live_backgrounds(dps)
        rows = [_row(background, q_text) for background in backgrounds for q_text in Q_GRID_MEV]
        if len(rows) != EXPECTED_ROW_COUNT:
            raise FieldEnergyTransferError("live field-energy row count mismatch")
        return {"base": base, "backgrounds": backgrounds, "guard": guard, "rows": rows, "dps": dps}


def _control_rows(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Independent algebraic/finite-difference controls for one snapshot."""

    direct_rows = []
    completion_rows = []
    finite_difference_rows = []
    coordinate_rows = []
    with mp.workdps(max(int(snapshot["dps"]) + 20, CONTROL_DPS)):
        for row in snapshot["rows"]:
            data = row["response"]
            D, B, C = data["D"], data["B"], data["C"]
            S = data["S"]
            direct_rows.append({
                "row_id": row["row_id"],
                "relative_error": data["direct_vs_schur_relative"],
                "pass": data["direct_vs_schur_relative"] is not None and data["direct_vs_schur_relative"] <= PRECISION_RELATIVE_LIMIT,
            })
            if C != 0 and S is not None:
                dn = mp.mpf("1.234")
                dy = -B / C * dn
                trial = quadratic_energy(D, B, C, dn, dy)
                expected = S * dn**2 / 2
                completion_rows.append({
                    "row_id": row["row_id"],
                    "relative_error": _relative(trial, expected),
                    "pass": _relative(trial, expected) <= PRECISION_RELATIVE_LIMIT,
                })
                step = mp.mpf("1e-7")
                plus = quadratic_energy(D, B, C, dn, dy + step)
                minus = quadratic_energy(D, B, C, dn, dy - step)
                derivative = (plus - minus) / (2 * step)
                # Scale by the natural quadratic energy and a tiny floor; this
                # catches a sign/coordinate error without demanding exact zero.
                derivative_error = abs(derivative) / max(abs(D * dn), mp.mpf("1e-50"))
                finite_difference_rows.append({
                    "row_id": row["row_id"],
                    "relative_error": derivative_error,
                    "pass": derivative_error <= FINITE_DIFFERENCE_RELATIVE_LIMIT,
                })
                sn, sy = mp.mpf("3.0"), mp.mpf("0.25")
                Dp, Bp, Cp = rescale_reduced_coefficients(D, B, C, sn, sy)
                rp = None if Dp == 0 or Cp == 0 else Bp**2 / (Dp * Cp)
                coordinate_error = None if data["r2"] is None or rp is None else _relative(data["r2"], rp)
                coordinate_rows.append({
                    "row_id": row["row_id"],
                    "r2_relative_error": coordinate_error,
                    "pass": coordinate_error is not None and coordinate_error <= PRECISION_RELATIVE_LIMIT,
                })

    def summary(items: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
        values = [item[field] for item in items if item[field] is not None]
        maximum = max(values, default=mp.mpf(0))
        return {"count": len(items), "max_relative_error": maximum, "pass": bool(items and all(item["pass"] for item in items))}

    # Synthetic controls deliberately live in the calculation, not in the
    # public row table.  They exercise zero coupling and a stable near-soft
    # reduced mode without inventing a source parameter.
    zero_D, zero_B, zero_C = mp.mpf("2"), mp.mpf("0"), mp.mpf("3")
    zero_r2 = zero_B**2 / (zero_D * zero_C)
    zero_ok = bool(zero_r2 == 0 and zero_D - zero_B**2 / zero_C == zero_D)
    soft_D, soft_B, soft_C = mp.mpf("1"), mp.mpf("0.9"), mp.mpf("1")
    soft_S = soft_D - soft_B**2 / soft_C
    soft_r2 = soft_B**2 / (soft_D * soft_C)
    soft_ok = bool(soft_S > 0 and soft_r2 < 1 and soft_r2 > 0)

    guard_cases = []
    try:
        _mp("NaN", "invalid")
    except FieldEnergyTransferError:
        guard_cases.append("nonfinite")
    try:
        rescale_reduced_coefficients(1, 1, 1, 0, 1)
    except FieldEnergyTransferError:
        guard_cases.append("invalid_coordinate_scale")
    try:
        reduced_response_from_hessian({"a": "NaN"}, "1", "100")
    except FieldEnergyTransferError:
        guard_cases.append("indefinite_or_malformed_hessian")

    return {
        "direct_inverse_vs_schur": summary(direct_rows, "relative_error"),
        "completion_of_square": summary(completion_rows, "relative_error"),
        "finite_difference_scalar_minimum": summary(finite_difference_rows, "relative_error"),
        "coordinate_rescaling_invariance": summary(coordinate_rows, "r2_relative_error"),
        "zero_coupling": {"r2": zero_r2, "pass": zero_ok},
        "near_soft_stable_limit": {"S": soft_S, "r2": soft_r2, "pass": soft_ok},
        "invalid_nonfinite_guards": {"cases_caught": guard_cases, "expected_count": 3, "pass": len(guard_cases) == 3},
    }


def _precision_probes(snapshot: Mapping[str, Any]) -> dict[str, mp.mpf]:
    probes: dict[str, mp.mpf] = {}
    for row in snapshot["rows"]:
        response = row["response"]
        for key in ("D", "B", "C", "S", "F", "determinant", "direct_chi", "frozen_coefficient", "relaxed_coefficient", "relaxation_coefficient", "r2", "r2_from_direct_chi"):
            value = response.get(key)
            if value is not None:
                probes[f"{row['row_id']}.{key}"] = value
    return probes


def _precision_control(primary: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    left, right = _precision_probes(primary), _precision_probes(control)
    if set(left) != set(right):
        raise FieldEnergyTransferError("primary/control probe keys changed")
    errors = {key: _relative(left[key], right[key]) for key in left}
    maximum = max(errors.values(), default=mp.mpf(0))
    if maximum > PRECISION_RELATIVE_LIMIT:
        key = max(errors, key=errors.get)
        raise FieldEnergyTransferError(f"precision control failed at {key}: {maximum}")
    return {"dps": [primary["dps"], control["dps"]], "probes_compared": len(errors), "max_relative_error": maximum, "relative_limit": PRECISION_RELATIVE_LIMIT, "pass": True}


def _serialize_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    backgrounds = [_background_public(background) for background in snapshot["backgrounds"]]
    rows_out = []
    for row in snapshot["rows"]:
        response = row["response"]
        rows_out.append({
            "row_id": row["row_id"],
            "background_id": row["background_id"],
            "family": row["family"],
            "target_y": row["target_y"],
            "q_MeV": _number(row["q_MeV"]),
            "Pi_vv_MeV2": _number(row["Pi_vv_MeV2"]),
            "Pi_vs_MeV2": _number(row["Pi_vs_MeV2"]),
            "Pi_ss_MeV2": _number(row["Pi_ss_MeV2"]),
            "coefficients": _shown(row["coefficients"]),
            "response": {
                "D_MeVminus2": _number(response["D"]),
                "B_MeV": _number(response["B"]),
                "C_MeV4": _number(response["C"]),
                "S_MeVminus2": None if response["S"] is None else _number(response["S"]),
                "F_MeV2": None if response["F"] is None else _number(response["F"]),
                # The raw determinant is retained for the mixed-dimensional
                # H; no eigenvalues are emitted or interpreted as energies.
                "det": _number(response["determinant"]),
                "direct_chi_MeV2": None if response["direct_chi"] is None else _number(response["direct_chi"]),
                "chi_schur_MeV2": None if response["chi"] is None else _number(response["chi"]),
                "direct_solution": _shown(response["direct_solution"]),
                "direct_status": response["direct_status"],
                "direct_vs_schur_relative": None if response["direct_vs_schur_relative"] is None else _number(response["direct_vs_schur_relative"]),
                "H_raw_mixed_coordinate_units": _shown(response["H"]),
                "status": response["status"],
                "stable": response["stable"],
            },
            "energy_transfer": {
                "fourier_convention": "E/V = 1/2[D|dn|^2 + 2B Re(dn* dy) + C|dy|^2] after static vector Schur elimination",
                "frozen_coefficient_per_abs_dn2": _number(response["frozen_coefficient"]),
                "relaxed_coefficient_per_abs_dn2": None if response["relaxed_coefficient"] is None else _number(response["relaxed_coefficient"]),
                "relaxation_coefficient_per_abs_dn2": None if response["relaxation_coefficient"] is None else _number(response["relaxation_coefficient"]),
                "dy_eq_per_dn": None if response["dy_dn"] is None else _number(response["dy_dn"]),
                "r2": None if response["r2"] is None else _number(response["r2"]),
                "r2_from_1_minus_1_over_D_chi": None if response["r2_from_direct_chi"] is None else _number(response["r2_from_direct_chi"]),
                "r2_identity_relative": None if response["r2_identity_relative"] is None else _number(response["r2_identity_relative"]),
                "completion_square_relative": None if response["completion_square_relative"] is None else _number(response["completion_square_relative"]),
                "not_a_power_or_efficiency": True,
            },
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "source_sha256": _source_hashes(),
        "dps": snapshot["dps"],
        "inputs_and_scope": {
            "natural_units": True,
            "density_unit": "MeV^3",
            "scalar_coordinate": "y=W/W0 (dimensionless)",
            "q_unit": "MeV",
            "families": ["Q4", "W8"],
            "W8_target_y": list(W8_TARGETS),
            "q_MeV": list(Q_GRID_MEV),
            "expected_background_count": EXPECTED_BACKGROUND_COUNT,
            "expected_row_count": EXPECTED_ROW_COUNT,
            "source_definition": "chi=delta n/delta mu_ext for source -mu_ext*n",
            "vector_constraint": "static longitudinal vector/Gauss variable eliminated before (D,B,C)",
            "negative_rows_retained": True,
            "no_r2_clipping": True,
            "no_new_fitted_parameter": True,
            "no_finite_amplitude_or_real_frequency_power": True,
        },
        "baseline_guard": _shown(snapshot["guard"]),
        "branch_lineage": {
            "Q4": "maintained nvg_nonlocal_response._backgrounds -> BulkModel().equilibrium(BulkModel().n0)",
            "W8.90": "maintained nvg_nonlocal_response._backgrounds -> source_complete_scaling_saturation_audit.inverse_potential_jet('0.90','240','0.16','-16') -> model.state(model.n0,'0.90')",
            "W8.93": "maintained nvg_nonlocal_response._backgrounds -> source_complete_scaling_saturation_audit.inverse_potential_jet('0.93','240','0.16','-16') -> model.state(model.n0,'0.93')",
        },
        "backgrounds": backgrounds,
        "rows": rows_out,
        "coverage": {
            "background_count": len(backgrounds),
            "row_count": len(rows_out),
            "expected_row_count": EXPECTED_ROW_COUNT,
            "all_rows_present": len(rows_out) == EXPECTED_ROW_COUNT,
            "all_statuses_retained": True,
            "stable_row_count": sum(bool(row["response"]["stable"]) for row in rows_out),
            "nonstable_row_count": sum(not bool(row["response"]["stable"]) for row in rows_out),
        },
    }


def calculate(dps: int = PRIMARY_DPS) -> dict[str, Any]:
    """Recompute the live source-derived table at one explicit precision."""

    snapshot = _raw_snapshot(dps)
    result = _serialize_snapshot(snapshot)
    # Controls are useful in the single-pass CLI too; they are recomputed from
    # live objects before serialization and never read from a result file.
    result["controls"] = _shown(_control_rows(snapshot))
    if not _finite_tree(result):
        raise FieldEnergyTransferError("nonfinite serialized calculation")
    return result


def build_result() -> dict[str, Any]:
    """Build primary/control snapshots and return a strict JSON-compatible map."""

    with mp.workdps(CONTROL_DPS + 20):
        primary = _raw_snapshot(PRIMARY_DPS)
        control = _raw_snapshot(CONTROL_DPS)
        precision = _precision_control(primary, control)
        result = _serialize_snapshot(primary)
        result["precision_controls"] = _shown(precision)
        result["controls"] = _shown(_control_rows(primary))
        result["limitations"] = [
            "The quadratic coefficients are local per-mode work/storage bounds for an externally supplied density perturbation.",
            "No NVG source operator, finite volume, mode amplitude/overlap, damping, transport law, load, electrical port, or recharge law is supplied.",
            "W8 branches are inverse-design alternatives trained on declared local targets, not held-out evidence; Q4 is the maintained baseline.",
            "The vacuum-subtracted medium kernel is static and zero-temperature; no finite-amplitude, real-frequency, total spectral-passivity, or watt claim follows.",
            "The raw Hessian mixes coordinate units; its determinant is a consistency diagnostic, not an energy eigenvalue.",
        ]
        result["interpretation"] = "redistribution of previously injected compression/field energy; not power, efficiency, or an autonomous generator"
        result["scope"] = {"evidence_weight": EVIDENCE_WEIGHT, "source_only_rebuild": True, "result_json_is_output_only": True}
        if not _finite_tree(result):
            raise FieldEnergyTransferError("nonfinite result tree")
        return result


def validate_result(result: Any) -> bool:
    """Validate structure and finite values without reading a saved output."""

    try:
        if not isinstance(result, Mapping) or result.get("schema_version") != SCHEMA_VERSION:
            return False
        if result.get("status") != STATUS or result.get("evidence_weight") != EVIDENCE_WEIGHT:
            return False
        coverage = result.get("coverage")
        rows = result.get("rows")
        if not isinstance(coverage, Mapping) or not isinstance(rows, list):
            return False
        if coverage.get("row_count") != EXPECTED_ROW_COUNT or len(rows) != EXPECTED_ROW_COUNT:
            return False
        if not coverage.get("all_rows_present") or not coverage.get("all_statuses_retained"):
            return False
        hashes = result.get("source_sha256")
        if not isinstance(hashes, Mapping) or dict(hashes) != _source_hashes():
            return False
        if not _finite_tree(result):
            return False
        required = {"row_id", "background_id", "family", "q_MeV", "response", "energy_transfer"}
        return all(isinstance(row, Mapping) and required <= set(row) for row in rows)
    except (TypeError, ValueError, ArithmeticError, FieldEnergyTransferError):
        return False


def _write_json(result: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    except OSError as exc:
        raise FieldEnergyTransferError(f"cannot write explicit output {path}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dps", type=int, choices=range(40, 121), default=None,
                        help="single-pass precision; default runs primary/control precision")
    parser.add_argument("--output", type=Path, default=None,
                        help="explicitly write the strict JSON result to this path")
    args = parser.parse_args(argv)
    result = calculate(args.dps) if args.dps is not None else build_result()
    if args.output is not None:
        _write_json(result, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
