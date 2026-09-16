#!/usr/bin/env python3
"""Conditional finite-static binding bridge for the live W8 Thomas--Fermi solver.

This module is deliberately separate from :mod:`nvg_finite_monopole`.  It
uses only that module's static BVP and diagnostics, never its projected
monopole/dynamic path, to perform one bounded conditional inverse problem:

* probe the explicit correlated scale family ``s`` for W8.90 and W8.93;
* match the *bare-nucleus* 40Ca binding obtained from the AME atomic value
  after the explicitly approximate AME electron correction; and
* freeze that scale before computing 40Ca, 90Zr and 208Pb comparisons.

The result is a conditional finite-TF calculation, not a likelihood, a
precision nuclear mass model, a finite-nucleus spectrum, or empirical proof.
Importing this module performs no numerical work and no network access.  The
CLI prints strict JSON to stdout by default; ``--output`` is an explicit
opt-in write requested by the caller.
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

try:  # Both direct-script and package-style imports are supported.
    import nvg_finite_monopole as static
except ImportError:  # pragma: no cover - package import support.
    from . import nvg_finite_monopole as static


HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
DATA_PATH = HERE / "data" / "finite_static_observables_2026.json"

SCHEMA = "nvg_finite_static_bridge.v1"
STATUS = "COMPUTED_CONDITIONAL_FINITE_STATIC_BINDING_BRIDGE_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

FAMILIES = ("W8.90", "W8.93")
RHO_CHOICE = "no_rho"
BRIDGE_NUCLEI = ("Ca40", "Zr90", "Pb208")
EXPLORATORY_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625)
SCALE_MIN = 0.0625
SCALE_MAX = 1.0
SEED_FACTORS = (0.8, 1.0, 1.2)

# Keep the static solver's declared gates unchanged.  The bridge only names
# them here to make its acceptance predicate auditable in the JSON output.
COARSE_LEVEL = {"label": "exploratory_coarse16", "box_fm": 16.0, "nodes": 401, "tol": 2.0e-6}
FINE_LEVEL = {"label": "fine24", "box_fm": 24.0, "nodes": 801, "tol": 2.0e-8}
DOMAIN_LEVEL = {"label": "domain40", "box_fm": 40.0, "nodes": 1201, "tol": 2.0e-8}
FRESH24_LEVEL = {"label": "fresh24_independent", "box_fm": 24.0, "nodes": 1201, "tol": 2.0e-8}
# Recovery always means a new BVP solve at the same physical box.  These
# levels are fixed before a run and are never selected by a dense-grid phase
# scan.
RECOVERY_LEVELS = (
    {"label": "recovery2401", "box_fm": None, "nodes": 2401, "tol": 2.0e-10},
    {"label": "recovery4001", "box_fm": None, "nodes": 4001, "tol": 2.0e-11},
)
# Same-scale branch/lineage comparisons use a common physical radial ladder.
LINEAGE_PROFILE_POINTS = 257
LINEAGE_MAX_Y_DIFF = 1.0e-5
LINEAGE_MAX_VECTOR_DIFF_MEV = 5.0e-3
LINEAGE_MAX_COULOMB_DIFF_MEV = 5.0e-3
LINEAGE_MAX_DENSITY_DIFF_FM3 = 1.0e-4 * static.N0_FM3
LINEAGE_MAX_MU_DIFF_MEV = 5.0e-3
LINEAGE_MAX_BINDING_DIFF_MEV = 5.0e-3
LINEAGE_MAX_RADIUS_DIFF_FM = 5.0e-3
LINEAGE_MAX_SCALE_STEP = 2.0e-2
# Independent fresh controls deliberately reuse every declared seed factor.
# The factors are exploratory starts.  A branch needs one valid independent
# previous=None result matching it; failed starts remain visible but do not
# veto a branch when another independent start supplies that control.
FRESH_CONTROL_FACTORS = SEED_FACTORS
ENERGY_LIMIT_MEV = 2.0e-2
RADIUS_LIMIT_FM = 2.0e-2
ROOT_RESIDUAL_LIMIT_MEV = 5.0e-3
ROOT_WIDTH_LIMIT = 2.0e-4
DERIVATIVE_STEPS = (0.0025, 0.00125)


class FiniteStaticBridgeError(ValueError):
    """Fail-closed error for malformed inputs or unavailable static evidence."""


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _number(value: Any, digits: int = 17) -> float:
    if isinstance(value, bool):
        raise FiniteStaticBridgeError("boolean is not a scientific number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FiniteStaticBridgeError("non-numeric scientific output") from exc
    if not math.isfinite(result):
        raise FiniteStaticBridgeError("non-finite scientific output")
    return float(format(result, f".{digits}g"))


def _jsonable(value: Any) -> Any:
    """Convert diagnostics to finite JSON without silently replacing values."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    # The static rows should already be profile-stripped, but this keeps the
    # helper safe for NumPy scalar diagnostics in focused tests.
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _jsonable(value.tolist())
        if isinstance(value, np.generic):
            return _jsonable(value.item())
    except ImportError:  # pragma: no cover - NumPy is a declared dependency.
        pass
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return value


def _scale_label(scale: float | str) -> str:
    value = _number(scale)
    if value < SCALE_MIN or value > SCALE_MAX:
        raise FiniteStaticBridgeError(f"scale {value:g} is outside the bounded bridge domain")
    return format(value, ".17g")


def _load_inputs(path: Path | str = DATA_PATH) -> dict[str, Any]:
    """Read and validate the cited input bundle; never read model outputs."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiniteStaticBridgeError(f"cannot read bridge input data {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise FiniteStaticBridgeError("bridge input data require schema_version=1")
    if payload.get("status") != "CITED_EVALUATED_INPUTS_ONLY_NOT_MODEL_OUTPUTS":
        raise FiniteStaticBridgeError("bridge input data are not explicitly input-only")
    binding = payload.get("binding")
    radii = payload.get("charge_radii")
    if not isinstance(binding, Mapping) or not isinstance(radii, Mapping):
        raise FiniteStaticBridgeError("bridge input data require binding and charge-radii sections")
    values = binding.get("values")
    rvalues = radii.get("values")
    if not isinstance(values, Mapping) or not isinstance(rvalues, Mapping):
        raise FiniteStaticBridgeError("bridge input data require finite-system value maps")
    for nucleus in BRIDGE_NUCLEI:
        item = values.get(nucleus)
        ritem = rvalues.get(nucleus)
        if not isinstance(item, Mapping) or not isinstance(ritem, Mapping):
            raise FiniteStaticBridgeError(f"missing cited input row for {nucleus}")
        for key in ("A", "Z", "N", "B_atom_per_A_MeV", "B_atom_per_A_sigma_MeV"):
            if key not in item or not _finite(item[key]):
                raise FiniteStaticBridgeError(f"invalid binding input {nucleus}.{key}")
        for key in ("R_ch_fm", "R_ch_sigma_fm"):
            if key not in ritem or not _finite(ritem[key]):
                raise FiniteStaticBridgeError(f"invalid charge-radius input {nucleus}.{key}")
        if int(item["A"]) != int(item["N"]) + int(item["Z"]):
            raise FiniteStaticBridgeError(f"binding input N/Z mismatch for {nucleus}")
    electrons = payload.get("electron_binding")
    nucleons = payload.get("nucleon_charge_inputs")
    if not isinstance(electrons, Mapping) or not isinstance(nucleons, Mapping):
        raise FiniteStaticBridgeError("bridge input data require electron and nucleon sections")
    for key in ("proton_charge_radius_fm", "proton_charge_radius_sigma_fm", "neutron_mean_square_charge_radius_fm2", "neutron_mean_square_charge_radius_sigma_fm2", "common_mass_MeV", "hbarc_MeV_fm"):
        if key not in nucleons or not _finite(nucleons[key]):
            raise FiniteStaticBridgeError(f"invalid nucleon input {key}")
    # Return a deep copy so callers/tests cannot mutate the source object used
    # by a later calculation.
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def electron_binding_eV(Z: int | float) -> float:
    """AME Part II Eq.(2) approximate electron binding in eV."""

    if isinstance(Z, bool):
        raise FiniteStaticBridgeError("Z must be a positive integer")
    try:
        z = float(Z)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FiniteStaticBridgeError("Z must be a positive integer") from exc
    if not math.isfinite(z) or z < 1.0 or z != round(z):
        raise FiniteStaticBridgeError("Z must be a positive integer")
    return _number(14.4381 * z**2.39 + 1.55468e-6 * z**5.35)


def electron_correction_MeV(Z: int | float) -> float:
    """Return ``[Be(Z)-Z Be(1)]/1e6`` with the AME sign convention.

    The AME expression is in eV, while bridge binding energies are in MeV;
    the conversion is therefore ``1 eV = 1e-6 MeV`` (the Ca40 correction is
    about 18.3 keV, not 18.3 MeV).
    """

    if isinstance(Z, bool):
        raise FiniteStaticBridgeError("Z must be a positive integer")
    try:
        z_float = float(Z)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FiniteStaticBridgeError("Z must be a positive integer") from exc
    if not math.isfinite(z_float) or z_float < 1.0 or z_float != round(z_float):
        raise FiniteStaticBridgeError("Z must be a positive integer")
    z = int(z_float)
    correction = (electron_binding_eV(z) - z * electron_binding_eV(1)) / 1.0e6
    if correction < 0.0 or not math.isfinite(correction):
        raise FiniteStaticBridgeError("electron correction has invalid sign")
    return _number(correction)


def bare_binding_per_A_MeV(binding_row: Mapping[str, Any]) -> float:
    """Convert tabulated atomic-H binding to the declared bare-nucleus target."""

    if not isinstance(binding_row, Mapping):
        raise FiniteStaticBridgeError("binding row must be a mapping")
    try:
        A, Z, atom = int(binding_row["A"]), int(binding_row["Z"]), float(binding_row["B_atom_per_A_MeV"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise FiniteStaticBridgeError("malformed AME binding row") from exc
    if A <= 0 or Z <= 0 or not math.isfinite(atom):
        raise FiniteStaticBridgeError("malformed AME binding row")
    return _number(atom - electron_correction_MeV(Z) / A)


def _input_targets(inputs: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    binding = inputs["binding"]["values"]
    targets: dict[str, dict[str, float]] = {}
    for nucleus in BRIDGE_NUCLEI:
        row = binding[nucleus]
        correction = electron_correction_MeV(int(row["Z"]))
        targets[nucleus] = {
            "B_atom_per_A_MeV": _number(row["B_atom_per_A_MeV"]),
            "B_atom_per_A_sigma_MeV": _number(row["B_atom_per_A_sigma_MeV"]),
            "electron_correction_total_MeV": correction,
            "electron_correction_per_A_MeV": _number(correction / int(row["A"])),
            "B_nuc_target_per_A_MeV": bare_binding_per_A_MeV(row),
        }
    return targets


_TARGET_FIELDS = (
    "B_atom_per_A_MeV",
    "B_atom_per_A_sigma_MeV",
    "electron_correction_total_MeV",
    "electron_correction_per_A_MeV",
    "B_nuc_target_per_A_MeV",
)


def _same_number(actual: Any, expected: Any, *, tolerance: float = 1.0e-12) -> bool:
    """Compare serialized scientific numbers without accepting non-finite data."""

    if isinstance(actual, bool) or isinstance(expected, bool) or not _finite(actual) or not _finite(expected):
        return False
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError, OverflowError):
        return False


def _target_mapping_matches(actual: Any, expected: Mapping[str, Any]) -> bool:
    """Bind one public target mapping to the recomputed cited-input values."""

    if not isinstance(actual, Mapping) or set(actual) != set(_TARGET_FIELDS):
        return False
    return all(_same_number(actual[key], expected[key]) for key in _TARGET_FIELDS)


def _all_input_targets_match(actual: Any, expected: Mapping[str, Mapping[str, Any]]) -> bool:
    """Validate the complete result-level target witness, not just Ca40."""

    if not isinstance(actual, Mapping) or set(actual) != set(BRIDGE_NUCLEI):
        return False
    return all(_target_mapping_matches(actual[nucleus], expected[nucleus]) for nucleus in BRIDGE_NUCLEI)


def _make_design(family: str, scale: float | str) -> Any:
    if family not in FAMILIES:
        raise FiniteStaticBridgeError(f"unsupported bridge family {family}")
    label = _scale_label(scale)
    # This opt-in API does not mutate static.SCALES or the old 30-row matrix.
    return static.FiniteDesign.from_continuous(family, label)


def _strip_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return _jsonable(static._strip_profiles(row))


def _failure_row(exc: Exception, *, level: Mapping[str, Any], factor: float | None, nucleus: str) -> dict[str, Any]:
    return {
        "solver_status": -1,
        "solver_message": str(exc),
        "converged": False,
        "box_fm": float(level["box_fm"]),
        "nodes_requested": int(level["nodes"]),
        "tolerance": float(level["tol"]),
        "seed_factor": None if factor is None else float(factor),
        "protocol_label": str(level["label"]),
        "nucleus": str(nucleus),
    }


def _raw_solve(
    design: Any,
    nucleus: str,
    *,
    level: Mapping[str, Any],
    factor: float | None = None,
    previous: Any | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """Call one declared BVP level with explicit bridge nucleus custody."""

    try:
        solved = static._solve_static_once(
            design,
            nucleus,
            RHO_CHOICE,
            box_fm=float(level["box_fm"]),
            nodes=int(level["nodes"]),
            tol=float(level["tol"]),
            factor=factor,
            previous=previous,
            nucleus_data=static.BRIDGE_NUCLEI,
        )
        row = _strip_row(solved.row)
        row["protocol_label"] = str(level["label"])
        return row, solved
    except (ArithmeticError, RuntimeError, ValueError, static.FiniteMonopoleError) as exc:
        return _failure_row(exc, level=level, factor=factor, nucleus=nucleus), None


def _row_needs_recovery(row: Mapping[str, Any]) -> bool:
    """Identify state-consistency failures eligible for a new BVP.

    The fixed recovery ladder is intentionally reserved for an actual
    first-order state identity failure.  Second-derivative/field diagnostics
    remain hard terminal gates but do not launch an unstable high-mesh Newton
    path; those rows are retained as numerical failures.
    """

    if row.get("converged") is not True:
        # A status-zero diagnostic failure can still have a live state to
        # refine; a nonzero solver status is retained as a failed guess.
        return row.get("solver_status") in (0, None)
    for key in (
        "first_order_state_y_residual_max_abs_dimensionless",
        "first_order_state_vector_residual_max_abs_dimensionless",
        "first_order_state_coulomb_residual_max_abs_dimensionless",
    ):
        try:
            if not math.isfinite(float(row[key])) or float(row[key]) > static.FIELD_RESIDUAL_LIMIT:
                return True
        except (KeyError, TypeError, ValueError, OverflowError):
            return True
    return False


def _solve(
    design: Any,
    nucleus: str,
    *,
    level: Mapping[str, Any],
    factor: float | None = None,
    previous: Any | None = None,
    recover: bool = True,
) -> tuple[dict[str, Any], Any | None]:
    """Solve one level, then use only fixed tighter BVP recovery levels.

    A recovery is a fresh collocation solve at the same physical box.  The
    original row is always retained; no unchanged solution is re-diagnosed on
    a phase-shifted grid and no recovery row is accepted merely by changing a
    diagnostic sample.
    """

    original_row, original_solution = _raw_solve(
        design, nucleus, level=level, factor=factor, previous=previous
    )
    if original_row.get("converged") is not True and original_row.get("solver_status") not in (0, None):
        original_row["recovery_policy"] = "not_attempted_solver_failure"
        return original_row, original_solution
    if not recover or not _row_needs_recovery(original_row):
        if not recover and _row_needs_recovery(original_row):
            original_row["recovery_policy"] = "disabled_for_exploratory_coarse_level"
        return original_row, original_solution
    # A residual-failing solve must trigger an actual fixed BVP recovery even
    # for a fresh previous=None attempt.  The recovery ladder keeps that
    # attempt genuinely fresh (no failed collocation state is reused); a
    # successful row records its fresh custody explicitly below.
    recovery_attempts: list[dict[str, Any]] = []
    # When a caller supplies a physical predecessor, prefer it over a
    # residual-failing target solve as the recovery seed.  The predecessor is
    # mapped onto the new physical-r coordinate by _solve_static_once; this
    # avoids iterating a numerically unstable collocation state while keeping
    # every tighter solve an actual BVP at the declared level.
    continuation = previous if previous is not None else None
    for recovery_spec in RECOVERY_LEVELS:
        recovery_level = dict(recovery_spec)
        recovery_level["box_fm"] = float(level["box_fm"])
        recovery_row, recovery_solution = _raw_solve(
            design,
            nucleus,
            level=recovery_level,
            factor=factor,
            previous=continuation,
        )
        recovery_row["attempt_role"] = "tighter_bvp_recovery"
        recovery_attempts.append(recovery_row)
        # Always execute the complete predeclared recovery ladder.  A failed
        # first recovery is still an actual BVP attempt.  Only a physical
        # predecessor may seed the next continuation level; a fresh
        # previous=None recovery stays previous=None at every level, so no
        # failed fresh state can be promoted through warm substitution.
        if previous is not None and recovery_solution is not None and recovery_row.get("converged") is True:
            continuation = recovery_solution
        if (
            recovery_solution is not None
            and recovery_row.get("converged") is True
            and recovery_row.get("stationarity_acceptance") is True
        ):
            effective = dict(recovery_row)
            effective["original_attempt"] = original_row
            effective["recovery_attempts"] = recovery_attempts
            effective["recovery_selected"] = str(recovery_level["label"])
            effective["recovery_independence"] = (
                "fresh_previous_none" if previous is None else "continuation_from_predecessor"
            )
            return effective, recovery_solution
    failed = dict(original_row)
    failed["recovery_attempts"] = recovery_attempts
    failed["recovery_selected"] = None
    return failed, original_solution


def _is_localized(row: Mapping[str, Any]) -> bool:
    # Exploratory rows are allowed to be unbound, but all retained numerical
    # stationarity/localization/convergence metrics must still agree.
    return _row_numerical_gate(row, require_binding=False)


def _binding(row: Mapping[str, Any]) -> float | None:
    try:
        value = -float(row["energy_per_A_minus_M_MeV"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _signature(row: Mapping[str, Any]) -> tuple[float, float, float] | None:
    try:
        mu = row["chemical_potentials_MeV"]
        if isinstance(mu, Mapping):
            mu_n, mu_p = float(mu["neutron"]), float(mu["proton"])
        else:
            mu_n, mu_p = float(mu[0]), float(mu[1])
        central = float(row["central_y"])
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (mu_n, mu_p, central)):
        return None
    return mu_n, mu_p, central


def _same_branch(left: tuple[float, float, float] | None, right: tuple[float, float, float] | None) -> bool:
    if left is None or right is None:
        return False
    return bool(
        # Exploratory endpoints can be separated by the whole frozen grid
        # (e.g. s=.25 to .5); their smooth branch shifts chemical potentials
        # by several MeV while central y remains nearly identical.  Distinct
        # branches are still rejected by the central-field discriminator.
        abs(left[0] - right[0]) <= 12.0
        and abs(left[1] - right[1]) <= 12.0
        and abs(left[2] - right[2]) <= 2.0e-2
    )


def _sample_physical_profile(pair: tuple[Mapping[str, Any], Any], physical_grid: np.ndarray) -> dict[str, np.ndarray] | None:
    """Sample one static solution on a supplied common physical-r grid."""

    row, solved = pair
    if solved is None or getattr(solved, "solution", None) is None:
        return None
    try:
        design = solved.design
        solution = solved.solution
        x = np.asarray(physical_grid, dtype=float) * design.W0 / design.hbarc
        state = np.asarray(solution.sol(x), dtype=float)
        matter, _, _ = static._make_static_functions(design, solved.c_rho, solved.kappa, solved.N, solved.Z)
        values = matter(state[0], state[2], state[4], np.asarray(solution.p, dtype=float))
        density_scale = (design.W0 / design.hbarc) ** 3
        return {
            "y": state[0],
            "vector_MeV": design.gomega * design.W0 * state[2],
            "coulomb_MeV": design.W0 * state[4],
            "neutron_density_fm3": values["nn"] * density_scale,
            "proton_density_fm3": values["np"] * density_scale,
        }
    except (ArithmeticError, RuntimeError, TypeError, ValueError, FloatingPointError):
        return None


def _profile_consistency(
    left: tuple[Mapping[str, Any], Any], right: tuple[Mapping[str, Any], Any]
) -> dict[str, Any]:
    """Compare converged rows on a common physical grid, not dimensionless x."""

    left_row, left_solution = left
    right_row, right_solution = right
    if left_solution is None or right_solution is None:
        return {"passed": False, "reason": "missing_solution"}
    try:
        left_box = float(left_solution.solution.x[-1] * left_solution.design.hbarc / left_solution.design.W0)
        right_box = float(right_solution.solution.x[-1] * right_solution.design.hbarc / right_solution.design.W0)
        common_box = min(left_box, right_box)
        if not math.isfinite(common_box) or common_box <= 0.0:
            return {"passed": False, "reason": "invalid_common_physical_box"}
        physical_grid = np.linspace(0.0, common_box, LINEAGE_PROFILE_POINTS)
        left_profile = _sample_physical_profile(left, physical_grid)
        right_profile = _sample_physical_profile(right, physical_grid)
        if left_profile is None or right_profile is None:
            return {"passed": False, "reason": "profile_sampling_failed"}
        y_diff = float(np.max(np.abs(left_profile["y"] - right_profile["y"])))
        vector_diff = float(np.max(np.abs(left_profile["vector_MeV"] - right_profile["vector_MeV"])))
        coulomb_diff = float(np.max(np.abs(left_profile["coulomb_MeV"] - right_profile["coulomb_MeV"])))
        density_diff = float(max(
            np.max(np.abs(left_profile["neutron_density_fm3"] - right_profile["neutron_density_fm3"])),
            np.max(np.abs(left_profile["proton_density_fm3"] - right_profile["proton_density_fm3"])),
        ))
        left_mu, right_mu = _signature(left_row), _signature(right_row)
        if left_mu is None or right_mu is None:
            return {"passed": False, "reason": "missing_chemical_potentials"}
        mu_diff = max(abs(left_mu[0] - right_mu[0]), abs(left_mu[1] - right_mu[1]))
        left_binding, right_binding = _binding(left_row), _binding(right_row)
        if left_binding is None or right_binding is None:
            return {"passed": False, "reason": "missing_binding"}
        binding_diff = abs(left_binding - right_binding)
        radius_diffs = {}
        for name, key in (
            ("neutron", "rms_neutron_radius_fm"),
            ("point_proton", "rms_point_proton_radius_fm"),
        ):
            try:
                radius_diffs[name] = abs(float(left_row[key]) - float(right_row[key]))
            except (KeyError, TypeError, ValueError, OverflowError):
                return {"passed": False, "reason": f"missing_{name}_radius"}
        passed = bool(
            y_diff <= LINEAGE_MAX_Y_DIFF
            and vector_diff <= LINEAGE_MAX_VECTOR_DIFF_MEV
            and coulomb_diff <= LINEAGE_MAX_COULOMB_DIFF_MEV
            and density_diff <= LINEAGE_MAX_DENSITY_DIFF_FM3
            and mu_diff <= LINEAGE_MAX_MU_DIFF_MEV
            and binding_diff <= LINEAGE_MAX_BINDING_DIFF_MEV
            and all(value <= LINEAGE_MAX_RADIUS_DIFF_FM for value in radius_diffs.values())
        )
        return {
            "passed": passed,
            "common_physical_box_fm": common_box,
            "max_abs_y_diff": y_diff,
            "max_abs_vector_potential_diff_MeV": vector_diff,
            "max_abs_coulomb_potential_diff_MeV": coulomb_diff,
            "max_abs_species_density_diff_fm_minus3": density_diff,
            "max_abs_mu_diff_MeV": mu_diff,
            "abs_binding_per_A_diff_MeV": binding_diff,
            "species_radius_diffs_fm": radius_diffs,
            "limits": {
                "y": LINEAGE_MAX_Y_DIFF,
                "vector_MeV": LINEAGE_MAX_VECTOR_DIFF_MEV,
                "coulomb_MeV": LINEAGE_MAX_COULOMB_DIFF_MEV,
                "density_fm_minus3": LINEAGE_MAX_DENSITY_DIFF_FM3,
                "mu_MeV": LINEAGE_MAX_MU_DIFF_MEV,
                "binding_per_A_MeV": LINEAGE_MAX_BINDING_DIFF_MEV,
                "radius_fm": LINEAGE_MAX_RADIUS_DIFF_FM,
            },
        }
    except (ArithmeticError, RuntimeError, TypeError, ValueError, FloatingPointError):
        return {"passed": False, "reason": "profile_comparison_failed"}


def _profile_comparison_expected(comparison: Any) -> bool | None:
    """Recompute a retained profile comparison from fixed limits.

    Serialized ``passed`` and ``limits`` values are descriptive only.  The
    actual metric fields below are compared with this module's fixed R1
    bounds, so changing a metric while leaving a green flag cannot pass.
    """

    if not isinstance(comparison, Mapping):
        return None
    metric_limits = (
        ("common_physical_box_fm", 0.0, None),
        ("max_abs_y_diff", 0.0, LINEAGE_MAX_Y_DIFF),
        ("max_abs_vector_potential_diff_MeV", 0.0, LINEAGE_MAX_VECTOR_DIFF_MEV),
        ("max_abs_coulomb_potential_diff_MeV", 0.0, LINEAGE_MAX_COULOMB_DIFF_MEV),
        ("max_abs_species_density_diff_fm_minus3", 0.0, LINEAGE_MAX_DENSITY_DIFF_FM3),
        ("max_abs_mu_diff_MeV", 0.0, LINEAGE_MAX_MU_DIFF_MEV),
        ("abs_binding_per_A_diff_MeV", 0.0, LINEAGE_MAX_BINDING_DIFF_MEV),
    )
    values: dict[str, float] = {}
    for key, lower, upper in metric_limits:
        try:
            value = float(comparison[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value) or value < lower or (upper is not None and value > upper):
            return None
        values[key] = value
    radii = comparison.get("species_radius_diffs_fm")
    if not isinstance(radii, Mapping):
        return None
    radius_values: list[float] = []
    for key in ("neutron", "point_proton"):
        try:
            value = float(radii[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value) or value < 0.0 or value > LINEAGE_MAX_RADIUS_DIFF_FM:
            return None
        radius_values.append(value)
    expected = bool(
        values["max_abs_y_diff"] <= LINEAGE_MAX_Y_DIFF
        and values["max_abs_vector_potential_diff_MeV"] <= LINEAGE_MAX_VECTOR_DIFF_MEV
        and values["max_abs_coulomb_potential_diff_MeV"] <= LINEAGE_MAX_COULOMB_DIFF_MEV
        and values["max_abs_species_density_diff_fm_minus3"] <= LINEAGE_MAX_DENSITY_DIFF_FM3
        and values["max_abs_mu_diff_MeV"] <= LINEAGE_MAX_MU_DIFF_MEV
        and values["abs_binding_per_A_diff_MeV"] <= LINEAGE_MAX_BINDING_DIFF_MEV
        and all(value <= LINEAGE_MAX_RADIUS_DIFF_FM for value in radius_values)
    )
    return expected


def _profile_comparison_numeric_valid(comparison: Any) -> bool:
    """Check both the numerical result and the producer's consistency flag."""

    expected = _profile_comparison_expected(comparison)
    return expected is not None and comparison.get("passed") is expected


def _fresh_profile_entry_consistent(entry: Any) -> tuple[bool, bool]:
    """Return ``(has_valid_metrics_and_flag, recomputed_pass)`` for an entry."""

    if not isinstance(entry, Mapping):
        return False, False
    comparison = entry.get("profile_consistency", entry)
    expected = _profile_comparison_expected(comparison)
    if expected is None:
        return False, False
    if "profile_consistency" in entry and entry.get("passed") is not expected:
        return False, expected
    if "profile_consistency" not in entry and comparison.get("passed") is not expected:
        return False, expected
    return True, expected


def _fresh_profile_entry_valid(entry: Any) -> bool:
    """Validate a fresh-to-fine comparison including its outer flag."""

    consistent, expected = _fresh_profile_entry_consistent(entry)
    return bool(consistent and expected)


def _group_same_scale_candidates(
    candidates: Sequence[tuple[dict[str, Any], Any]],
) -> tuple[list[list[tuple[dict[str, Any], Any]]], list[dict[str, Any]]]:
    """Group rows only after full profile consistency, retaining comparisons."""

    groups: list[list[tuple[dict[str, Any], Any]]] = []
    comparisons: list[dict[str, Any]] = []
    for candidate in candidates:
        placed = False
        for group_index, group in enumerate(groups):
            heuristic_match = _same_branch(_signature(candidate[0]), _signature(group[0][0]))
            comparison = _profile_consistency(candidate, group[0]) if heuristic_match else {"passed": False, "reason": "heuristic_signature_screen"}
            comparisons.append({
                "candidate_seed_factor": candidate[0].get("seed_factor"),
                "reference_seed_factor": group[0][0].get("seed_factor"),
                "group_index": group_index + 1,
                "heuristic_signature_screen": heuristic_match,
                "profile_consistency": comparison,
            })
            if heuristic_match and comparison.get("passed") is True:
                group.append(candidate)
                placed = True
                break
        if not placed:
            groups.append([candidate])
    return groups, comparisons


def _coarse_probe(family: str, scale: float, nucleus: str = "Ca40") -> dict[str, Any]:
    """Run the required three exploratory seeds and retain every outcome."""

    label = _scale_label(scale)
    design = _make_design(family, label)
    attempts: list[dict[str, Any]] = []
    live: list[tuple[dict[str, Any], Any]] = []
    for factor in SEED_FACTORS:
        row, solved = _solve(design, nucleus, level=COARSE_LEVEL, factor=factor, recover=False)
        row.update({"family": family, "scale": label, "nucleus": nucleus, "seed_factor": float(factor)})
        row["attempt_role"] = "exploratory_seed"
        attempts.append(row)
        if solved is not None and row.get("converged") is True:
            live.append((row, solved))
    # Coarse rows are only guidance.  A representative is useful for branch
    # tracking but can never be a terminal calibrated state.  Signatures are
    # only a cheap screen; full profiles decide whether rows share a branch.
    groups, comparisons = _group_same_scale_candidates(live)
    representatives = [group[0] for group in groups]
    branch_rows = []
    for index, group in enumerate(groups, start=1):
        row = group[0][0]
        branch_rows.append({
            "branch_id": f"coarse_branch_{index}",
            "seed_factors": [item[0].get("seed_factor") for item in group],
            "localized": bool(row.get("localized_vacuum_exterior") is True),
            "stationary_gate": bool(row.get("stationarity_acceptance") is True),
            "binding_per_A_MeV": _binding(row),
            "signature": {
                "mu_n_MeV": _signature(row)[0] if _signature(row) else None,
                "mu_p_MeV": _signature(row)[1] if _signature(row) else None,
                "central_y": _signature(row)[2] if _signature(row) else None,
            },
            "same_scale_profile_consistent": all(
                item.get("profile_consistency", {}).get("passed") is True
                for item in comparisons
                if item.get("group_index") == index
            ) if len(group) > 1 else True,
        })
    return {
        "family": family,
        "scale": label,
        "nucleus": nucleus,
        "protocol_label": COARSE_LEVEL["label"],
        "seed_factors": list(SEED_FACTORS),
        "seed_attempts": attempts,
        "all_seed_outcomes_reported": len(attempts) == len(SEED_FACTORS),
        "coarse_branches": branch_rows,
        "same_scale_profile_comparisons": comparisons,
        "localized_branch_count": sum(1 for branch in branch_rows if branch["localized"]),
        "coarse_only_not_terminal": True,
        "_live": live,
        "_representatives": representatives,
    }


def _probe_binding(row: Mapping[str, Any]) -> float | None:
    value = _binding(row)
    if value is None or row.get("localized_vacuum_exterior") is not True:
        return None
    return value


def _adaptive_probe_series(family: str, fixed: Mapping[str, dict[str, Any]], target: float) -> list[dict[str, Any]]:
    """Fill a missing fixed-grid sign change with bounded live probes.

    The fixed scale list remains intact in ``fixed``.  This path is only used
    when coarse edge failures leave no localized sign change; every such
    attempt is reported explicitly rather than being silently promoted.
    """

    probes: list[dict[str, Any]] = []
    values = [
        (float(scale), probe)
        for scale, probe in fixed.items()
        if any(_probe_binding(row) is not None for row, _ in probe.get("_live", []))
    ]
    if not values:
        return probes
    # The physical branch's binding decreases with s.  Start from the valid
    # point nearest the target and move downward in bounded 4% steps; the
    # smaller step avoids jumping across a numerically delicate TF edge.
    valid_scale = min(values, key=lambda item: abs(_probe_binding(item[1]["_live"][0][0]) - target) if _probe_binding(item[1]["_live"][0][0]) is not None else math.inf)[0]
    previous_live = fixed[format(valid_scale, ".17g")].get("_representatives", [])
    previous_solution = previous_live[0][1] if previous_live else None
    scale = valid_scale
    for _ in range(24):
        existing = [float(item["scale"]) for item in probes]
        candidate = max(SCALE_MIN, scale * 0.96)
        if candidate >= scale - 1.0e-12 or any(abs(candidate - item) <= 1.0e-12 for item in existing):
            break
        probe = _coarse_probe(family, candidate)
        probe["probe_role"] = "adaptive_coarse_after_fixed_grid_failure"
        probes.append(probe)
        live = probe.get("_representatives", [])
        chosen = live[0] if live else None
        if chosen is None and previous_solution is not None:
            # A smaller step warm-start is explicitly allowed as a recovery
            # when the original fresh edge solve failed; retain that failure
            # in ``seed_attempts`` and label this row as a recovery attempt.
            design = _make_design(family, candidate)
            row, solved = _solve(design, "Ca40", level=COARSE_LEVEL, previous=previous_solution, recover=False)
            row.update({"family": family, "scale": _scale_label(candidate), "nucleus": "Ca40", "attempt_role": "adaptive_warm_start_recovery"})
            probe["recovery_attempts"] = [row]
            if solved is not None and row.get("converged") is True:
                chosen = (row, solved)
                probe["_representatives"] = [chosen]
        if chosen is not None:
            previous_solution = chosen[1]
            value = _probe_binding(chosen[0])
            if value is not None and (value - target) * (_probe_binding(values[0][1]["_live"][0][0]) - target if _probe_binding(values[0][1]["_live"][0][0]) is not None else -1.0) <= 0.0:
                break
        scale = candidate
        if candidate <= SCALE_MIN + 1.0e-12:
            break
    return probes


def _public_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in probe.items() if not str(key).startswith("_")}
    return _jsonable(result)


def _coarse_brackets(probes: Sequence[Mapping[str, Any]], target: float) -> list[dict[str, Any]]:
    entries: list[tuple[float, dict[str, Any], dict[str, Any], tuple[float, float, float] | None, Any]] = []
    for probe in probes:
        for row, solved in probe.get("_representatives", []):
            if row.get("localized_vacuum_exterior") is not True:
                continue
            value = _probe_binding(row)
            if value is None:
                continue
            entries.append((float(probe["scale"]), probe, row, _signature(row), solved))
    entries.sort(key=lambda item: item[0])
    brackets: list[dict[str, Any]] = []
    for index, left in enumerate(entries):
        for right in entries[index + 1 :]:
            f_left = _probe_binding(left[2])
            f_right = _probe_binding(right[2])
            if f_left is None or f_right is None or (f_left - target) * (f_right - target) > 0.0:
                continue
            if not _same_branch(left[3], right[3]):
                continue
            lo, hi = sorted((left[0], right[0]))
            brackets.append({
                "scale_lo": lo,
                "scale_hi": hi,
                "width": hi - lo,
                "binding_lo_MeV": _probe_binding(left[2]) if left[0] == lo else _probe_binding(right[2]),
                "binding_hi_MeV": _probe_binding(right[2]) if right[0] == hi else _probe_binding(left[2]),
                "target_B_nuc_per_A_MeV": target,
                # This is only a coarse heuristic screen.  The terminal root
                # must prove lineage at every insertion before acceptance.
                "same_tracked_branch": "PENDING_TERMINAL_LINEAGE",
                "heuristic_signature_screen": True,
                "coarse_profile_lineage_available": bool(left[1].get("_representatives") and right[1].get("_representatives")),
                "localized_endpoints": True,
                "source": "coarse_or_adaptive_localized_sign_change",
                # Private live solutions seed the first fine endpoint
                # lineage.  They are removed before JSON serialization.
                "_left_solution": left[4],
                "_right_solution": right[4],
            })
    brackets.sort(key=lambda item: (float(item["width"]), float(item["scale_lo"])))
    return brackets


def _public_bracket(bracket: Mapping[str, Any]) -> dict[str, Any]:
    """Drop live StaticSolution custody before serializing a bracket."""

    return _jsonable({key: value for key, value in bracket.items() if not str(key).startswith("_")})


def _species_radius(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _nested_control_numerical_gate(row: Mapping[str, Any]) -> bool:
    """Recompute the nested diagnostic gate from its retained numeric row."""

    nested = row.get("nested_diagnostic") if isinstance(row, Mapping) else None
    if not isinstance(nested, Mapping):
        return False
    residual_limits = (
        ("conserved_NZ_relative_error", static.N_RELATIVE_LIMIT),
        ("field_residual_relative", static.FIELD_RESIDUAL_LIMIT),
        ("first_order_state_residual_max_abs_dimensionless", static.FIELD_RESIDUAL_LIMIT),
        ("first_order_state_y_residual_max_abs_dimensionless", static.FIELD_RESIDUAL_LIMIT),
        ("first_order_state_vector_residual_max_abs_dimensionless", static.FIELD_RESIDUAL_LIMIT),
        ("first_order_state_coulomb_residual_max_abs_dimensionless", static.FIELD_RESIDUAL_LIMIT),
        ("first_order_field_residual_relative", static.FIELD_RESIDUAL_LIMIT),
        ("coulomb_residual_relative", 1.0e-4),
        ("first_order_coulomb_residual_relative", 1.0e-4),
        ("kkt_max_error_MeV", static.KKT_LIMIT_MEV),
        ("gauss_identity_relative", static.GAUSS_LIMIT),
        ("coulomb_identity_relative", static.COULOMB_LIMIT),
    )
    values: dict[str, float] = {}
    for key, limit in residual_limits:
        try:
            value = float(nested[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(value) or value < 0.0 or value > float(limit):
            return False
        values[key] = value
    for key in (
        "first_order_field_residual_max_abs_dimensionless",
        "first_order_scalar_ode_residual_max_abs_dimensionless",
        "first_order_vector_ode_residual_max_abs_dimensionless",
        "scalar_residual_max_abs_dimensionless",
        "vector_residual_max_abs_dimensionless",
    ):
        try:
            value = float(nested[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(value) or value < 0.0:
            return False
    try:
        energy = float(nested["energy_per_A_minus_M_MeV"])
        central_y = float(nested["central_y"])
        min_y = float(nested["min_y"])
        central_density = float(nested["central_density_fm_minus3"])
        mu = nested["chemical_potentials_MeV"]
        if isinstance(mu, Mapping):
            mu_n, mu_p = float(mu["neutron"]), float(mu["proton"])
        else:
            mu_n, mu_p = float(mu[0]), float(mu[1])
        edge = nested["boundary_densities_dimensionless"]
        edge_n, edge_p = float(edge["neutron"]), float(edge["proton"])
        outer = float(nested["outer_number_fraction_max"])
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return False
    if not all(math.isfinite(value) for value in (energy, central_y, min_y, central_density, mu_n, mu_p, edge_n, edge_p, outer)):
        return False
    if central_y <= 0.0 or min_y <= 0.0 or central_density < 0.0 or edge_n < 0.0 or edge_p < 0.0 or outer < 0.0:
        return False
    localized_expected = bool(
        mu_n < static.MASS_MEV
        and mu_p < static.MASS_MEV
        and edge_n <= 1.0e-10
        and edge_p <= 1.0e-10
        and outer <= 1.0e-8
    )
    binding_expected = bool(energy < 0.0)
    stationarity_expected = bool(
        values["conserved_NZ_relative_error"] <= static.N_RELATIVE_LIMIT
        and values["field_residual_relative"] <= static.FIELD_RESIDUAL_LIMIT
        and values["first_order_state_y_residual_max_abs_dimensionless"] <= static.FIELD_RESIDUAL_LIMIT
        and values["first_order_state_vector_residual_max_abs_dimensionless"] <= static.FIELD_RESIDUAL_LIMIT
        and values["first_order_state_coulomb_residual_max_abs_dimensionless"] <= static.FIELD_RESIDUAL_LIMIT
        and values["first_order_field_residual_relative"] <= static.FIELD_RESIDUAL_LIMIT
        and values["first_order_coulomb_residual_relative"] <= 1.0e-4
        and values["coulomb_residual_relative"] <= 1.0e-4
        and values["kkt_max_error_MeV"] <= static.KKT_LIMIT_MEV
        and values["gauss_identity_relative"] <= static.GAUSS_LIMIT
        and values["coulomb_identity_relative"] <= static.COULOMB_LIMIT
        and min_y > 0.0
        and central_density >= 0.0
    )
    return bool(
        nested.get("stationarity_acceptance") is stationarity_expected
        and nested.get("localized_vacuum_exterior") is localized_expected
        and nested.get("binding_condition_E_per_A_lt_M") is binding_expected
    )


def _row_numerical_gate(row: Mapping[str, Any], *, require_binding: bool = True) -> bool:
    """Apply the small shared numerical contract to one serialized row.

    This predicate intentionally reads the retained numerical metrics rather
    than accepting the producer's booleans as evidence.  It is used while
    aggregating live bridge branches and again by the serialized validator.
    The exploratory coarse atlas may retain rows outside this gate, but no
    such row can become a terminal result.
    """

    if not isinstance(row, Mapping) or row.get("converged") is not True:
        return False
    try:
        status = float(row["solver_status"])
        if not math.isfinite(status) or status != 0.0:
            return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False

    # These are the existing static limits.  The three first-order state
    # components are separate absolute gates, not merely an aggregate flag.
    residual_limits = (
        ("conserved_NZ_relative_error", static.N_RELATIVE_LIMIT),
        ("field_residual_relative", static.FIELD_RESIDUAL_LIMIT),
        ("first_order_state_residual_max_abs_dimensionless", static.FIELD_RESIDUAL_LIMIT),
        ("first_order_state_y_residual_max_abs_dimensionless", static.FIELD_RESIDUAL_LIMIT),
        ("first_order_state_vector_residual_max_abs_dimensionless", static.FIELD_RESIDUAL_LIMIT),
        ("first_order_state_coulomb_residual_max_abs_dimensionless", static.FIELD_RESIDUAL_LIMIT),
        ("first_order_field_residual_relative", static.FIELD_RESIDUAL_LIMIT),
        ("coulomb_residual_relative", 1.0e-4),
        ("first_order_coulomb_residual_relative", 1.0e-4),
        ("kkt_max_error_MeV", static.KKT_LIMIT_MEV),
        ("gauss_identity_relative", static.GAUSS_LIMIT),
        ("coulomb_identity_relative", static.COULOMB_LIMIT),
    )
    for key, limit in residual_limits:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(value) or value < 0.0 or value > float(limit):
            return False
    # Keep the first-order ODE and second-derivative diagnostics present and
    # finite.  Their historical acceptance limits remain encoded above in the
    # relative field/Coulomb quantities; no new tolerance is introduced here.
    for key in (
        "first_order_field_residual_max_abs_dimensionless",
        "first_order_scalar_ode_residual_max_abs_dimensionless",
        "first_order_vector_ode_residual_max_abs_dimensionless",
        "scalar_residual_max_abs_dimensionless",
        "vector_residual_max_abs_dimensionless",
    ):
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(value) or value < 0.0:
            return False

    try:
        energy = float(row["energy_per_A_minus_M_MeV"])
        central_y = float(row["central_y"])
        min_y = float(row["min_y"])
        central_density = float(row["central_density_fm_minus3"])
        collocation = float(row["collocation_max_rms"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(value) for value in (energy, central_y, min_y, central_density, collocation)):
        return False
    if min_y <= 0.0 or central_y <= 0.0 or central_density < 0.0 or collocation < 0.0:
        return False
    if any(
        radius is None or not math.isfinite(radius) or radius <= 0.0
        for radius in (
            _species_radius(row, "rms_neutron_radius_fm"),
            _species_radius(row, "rms_point_proton_radius_fm"),
        )
    ):
        return False

    signature = _signature(row)
    if signature is None:
        return False
    mu_n, mu_p, _ = signature
    try:
        edge = row["boundary_densities_dimensionless"]
        edge_n, edge_p = float(edge["neutron"]), float(edge["proton"])
        outer = float(row["outer_number_fraction_max"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(value) for value in (edge_n, edge_p, outer)):
        return False
    if edge_n < 0.0 or edge_p < 0.0 or outer < 0.0:
        return False
    localized_expected = bool(
        mu_n < static.MASS_MEV
        and mu_p < static.MASS_MEV
        and edge_n <= 1.0e-10
        and edge_p <= 1.0e-10
        and outer <= 1.0e-8
    )
    if row.get("localized_vacuum_exterior") is not localized_expected:
        return False

    # Binding is a derived comparison, not a trusted flag.  Keep the flag in
    # agreement even for unbound exploratory rows; terminal callers choose
    # require_binding=True.
    binding_expected = bool(energy < 0.0)
    if row.get("binding_condition_E_per_A_lt_M") is not binding_expected:
        return False
    if require_binding and not binding_expected:
        return False
    if not _nested_control_numerical_gate(row):
        return False

    stationarity_expected = bool(
        float(row["conserved_NZ_relative_error"]) <= static.N_RELATIVE_LIMIT
        and float(row["field_residual_relative"]) <= static.FIELD_RESIDUAL_LIMIT
        and float(row["first_order_state_y_residual_max_abs_dimensionless"]) <= static.FIELD_RESIDUAL_LIMIT
        and float(row["first_order_state_vector_residual_max_abs_dimensionless"]) <= static.FIELD_RESIDUAL_LIMIT
        and float(row["first_order_state_coulomb_residual_max_abs_dimensionless"]) <= static.FIELD_RESIDUAL_LIMIT
        and float(row["first_order_field_residual_relative"]) <= static.FIELD_RESIDUAL_LIMIT
        and float(row["first_order_coulomb_residual_relative"]) <= 1.0e-4
        and float(row["coulomb_residual_relative"]) <= 1.0e-4
        and float(row["kkt_max_error_MeV"]) <= static.KKT_LIMIT_MEV
        and float(row["gauss_identity_relative"]) <= static.GAUSS_LIMIT
        and float(row["coulomb_identity_relative"]) <= static.COULOMB_LIMIT
        and min_y > 0.0
        and central_density >= 0.0
    )
    return row.get("stationarity_acceptance") is stationarity_expected


def _terminal_row_valid(row: Mapping[str, Any], *, require_binding: bool = True) -> bool:
    """Require a row to pass the shared numerical terminal predicate."""

    return _row_numerical_gate(row, require_binding=require_binding)


def _original_attempt(row: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = row.get("original_attempt")
    return candidate if isinstance(candidate, Mapping) else row


def _fresh_attempt_valid(row: Any) -> bool:
    """Check an independent fresh row without trusting its container shape."""

    if not isinstance(row, Mapping) or row.get("independent_previous") is not None:
        return False
    if not _terminal_row_valid(row):
        return False
    original = _original_attempt(row)
    if _terminal_row_valid(original):
        return True
    # A failed initial numerical resolution may be repaired by a genuinely
    # fresh tighter solve.  The recovery is eligible only when it explicitly
    # records previous=None custody; warm continuation is never promoted.
    return row.get("recovery_independence") == "fresh_previous_none"


def _energy_and_radius_controls(
    fine: Mapping[str, Any],
    other: Mapping[str, Any],
) -> tuple[bool, float | None, dict[str, float | None]]:
    """Recompute refinement metrics from the two retained numerical rows."""

    try:
        fine_energy = float(fine["energy_per_A_minus_M_MeV"])
        other_energy = float(other["energy_per_A_minus_M_MeV"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, None, {"neutron": None, "point_proton": None}
    if not math.isfinite(fine_energy) or not math.isfinite(other_energy):
        return False, None, {"neutron": None, "point_proton": None}
    energy_change = abs(other_energy - fine_energy)
    radius_changes: dict[str, float | None] = {}
    for name, key in (
        ("neutron", "rms_neutron_radius_fm"),
        ("point_proton", "rms_point_proton_radius_fm"),
    ):
        left, right = _species_radius(fine, key), _species_radius(other, key)
        radius_changes[name] = None if left is None or right is None else abs(left - right)
    passed = bool(
        energy_change <= ENERGY_LIMIT_MEV
        and all(value is not None and math.isfinite(value) and value <= RADIUS_LIMIT_FM for value in radius_changes.values())
    )
    return passed, energy_change, radius_changes


def _branch_numeric_gate(
    fine: Mapping[str, Any],
    domain: Mapping[str, Any],
    fresh: Mapping[str, Any],
    fresh_initial: Mapping[str, Any],
    fresh_attempts: Any,
    domain_profile: Any,
    fresh_profiles: Any,
) -> dict[str, Any]:
    """Compute one branch's acceptance from actual rows and profile metrics."""

    fine_valid = _terminal_row_valid(fine)
    domain_valid = _terminal_row_valid(domain)
    fresh_valid = _fresh_attempt_valid(fresh)
    fresh_initial_valid = _terminal_row_valid(fresh_initial)
    reported_attempts = bool(
        isinstance(fresh_attempts, list)
        and len(fresh_attempts) == len(FRESH_CONTROL_FACTORS)
        and all(isinstance(item, Mapping) for item in fresh_attempts)
    )
    independent_rows = [item for item in (fresh_attempts if reported_attempts else []) if _fresh_attempt_valid(item)]
    fresh_rows_valid = bool(independent_rows)
    selected_factor = fresh.get("seed_factor") if isinstance(fresh, Mapping) else None
    selected_in_attempts = bool(
        selected_factor is not None
        and any(
            _finite(item.get("seed_factor"))
            and abs(float(item.get("seed_factor")) - float(selected_factor)) <= 1.0e-15
            for item in independent_rows
        )
    )
    profile_entries = fresh_profiles if isinstance(fresh_profiles, list) else []
    profile_consistent = []
    for item in profile_entries:
        consistent, passed = _fresh_profile_entry_consistent(item)
        profile_consistent.append((consistent, passed))
    fresh_profile_pass = bool(profile_consistent and all(item[0] for item in profile_consistent) and any(item[1] for item in profile_consistent))
    domain_profile_pass = _profile_comparison_numeric_valid(domain_profile)
    domain_refinement_pass, domain_energy_change, domain_radius_changes = _energy_and_radius_controls(fine, domain)
    fresh_refinement_pass, fresh_energy_change, fresh_radius_changes = _energy_and_radius_controls(fine, fresh)
    # Require the selected fresh row to be one of the independently valid
    # attempts and to have at least one matching, numerically valid profile.
    selected_profile_match = bool(
        selected_in_attempts
        and any(
            item[0]
            and item[1]
            and isinstance(entry, Mapping)
            and (
                selected_factor is None
                or not _finite(entry.get("fresh_seed_factor"))
                or abs(float(entry.get("fresh_seed_factor")) - float(selected_factor)) <= 1.0e-15
            )
            for item, entry in zip(profile_consistent, profile_entries)
        )
    )
    accepted = bool(
        fine_valid
        and domain_valid
        and fresh_valid
        and fresh_rows_valid
        and selected_profile_match
        and domain_profile_pass
        and domain_refinement_pass
        and fresh_refinement_pass
    )
    return {
        "accepted": accepted,
        "fine_valid": fine_valid,
        "domain_valid": domain_valid,
        "fresh_valid": fresh_valid,
        "fresh_initial_valid": fresh_initial_valid,
        "fresh_attempts_reported": reported_attempts,
        "fresh_independent_rows_valid": fresh_rows_valid,
        "fresh_profile_pass": fresh_profile_pass,
        "selected_fresh_in_attempts": selected_in_attempts,
        "selected_fresh_profile_match": selected_profile_match,
        "domain_profile_pass": domain_profile_pass,
        "domain_energy_change": domain_energy_change,
        "fresh_energy_change": fresh_energy_change,
        "domain_radius_changes": domain_radius_changes,
        "fresh_radius_changes": fresh_radius_changes,
        "domain_refinement_pass": domain_refinement_pass,
        "fresh_refinement_pass": fresh_refinement_pass,
        "profile_entries_consistent": all(item[0] for item in profile_consistent) if profile_consistent else False,
    }


def _fresh_branch_unmatched_factors(branches: Any, fresh_attempts: Any) -> list[float]:
    """Find valid fresh branches that no retained fine branch controls."""

    if not isinstance(branches, list) or not isinstance(fresh_attempts, list):
        return []
    valid_fresh = [item for item in fresh_attempts if _fresh_attempt_valid(item)]
    represented: set[float] = set()
    for branch in branches:
        if not isinstance(branch, Mapping):
            continue
        for entry in branch.get("fresh24_profile_comparisons", []):
            if not _fresh_profile_entry_valid(entry) or entry.get("passed") is not True:
                continue
            factor = entry.get("fresh_seed_factor")
            if _finite(factor):
                represented.add(float(factor))
    unmatched = []
    for item in valid_fresh:
        factor = item.get("seed_factor") if isinstance(item, Mapping) else None
        if _finite(factor) and not any(abs(float(factor) - other) <= 1.0e-15 for other in represented):
            unmatched.append(float(factor))
    return unmatched


def _terminal_case(
    family: str,
    scale: float,
    nucleus: str,
    *,
    multi_seed: bool,
    lineage_predecessors: Sequence[tuple[str, Any]] = (),
) -> dict[str, Any]:
    """Run fine/domain/fresh controls and explicit same-scale lineages."""

    label = _scale_label(scale)
    design = _make_design(family, label)
    factors = SEED_FACTORS if multi_seed else (1.0,)
    fine_attempts: list[dict[str, Any]] = []
    fine_live: list[tuple[dict[str, Any], Any]] = []
    for factor in factors:
        row, solved = _solve(design, nucleus, level=FINE_LEVEL, factor=factor)
        row.update({"family": family, "scale": label, "nucleus": nucleus, "seed_factor": float(factor)})
        row["attempt_role"] = "fine_terminal_seed" if multi_seed else "root_bracket_fine_probe"
        fine_attempts.append(row)
        if solved is not None and row.get("converged") is True:
            fine_live.append((row, solved))

    fine_groups, fine_comparisons = _group_same_scale_candidates(fine_live)
    fine_representatives = [group[0] for group in fine_groups]

    lineage_attempts: list[dict[str, Any]] = []
    lineage_live: list[tuple[dict[str, Any], Any]] = []
    for predecessor_label, predecessor in lineage_predecessors:
        if predecessor is None:
            lineage_attempts.append({
                "attempt_role": "lineage_continuation_missing_predecessor",
                "predecessor_label": str(predecessor_label),
                "converged": False,
            })
            continue
        row, solved = _solve(
            design,
            nucleus,
            level=FINE_LEVEL,
            factor=None,
            previous=predecessor,
        )
        row.update({
            "family": family,
            "scale": label,
            "nucleus": nucleus,
            "seed_factor": None,
            "attempt_role": "lineage_continuation_from_" + str(predecessor_label),
            "predecessor_label": str(predecessor_label),
        })
        lineage_attempts.append(row)
        if solved is not None and _terminal_row_valid(row):
            lineage_live.append((row, solved))

    lineage_comparisons: list[dict[str, Any]] = []
    if lineage_predecessors:
        if not fine_representatives:
            lineage_comparisons.append({
                "passed": False,
                "reason": "no_converged_fine_seed_for_lineage_comparison",
            })
        for lineage_pair in lineage_live:
            if not fine_representatives:
                break
            comparisons = [
                _profile_consistency(lineage_pair, reference)
                for reference in fine_representatives
            ]
            best = next((item for item in comparisons if item.get("passed") is True), comparisons[0])
            lineage_comparisons.append({
                "source": lineage_pair[0].get("predecessor_label"),
                "compared_to_seed_factors": [item[0].get("seed_factor") for item in fine_representatives],
                "profile_consistency": best,
                "passed": best.get("passed") is True,
            })
        if not lineage_live:
            lineage_comparisons.append({
                "passed": False,
                "reason": "no_converged_lineage_continuation",
            })
        for index, left in enumerate(lineage_live):
            for right in lineage_live[index + 1:]:
                comparison = _profile_consistency(left, right)
                lineage_comparisons.append({
                    "source_pair": [left[0].get("predecessor_label"), right[0].get("predecessor_label")],
                    "profile_consistency": comparison,
                    "passed": comparison.get("passed") is True,
                })
    lineage_ok = bool(
        not lineage_predecessors
        or lineage_comparisons
        and all(item.get("passed") is True for item in lineage_comparisons)
    )

    # The independent fresh control is deliberately a previous=None solve.
    # A tighter warm recovery may be retained in its row, but an initially
    # failed fresh solve can never be promoted to independent truth.  The
    # seed factors are exploratory guesses: one valid independent row is
    # sufficient, while every attempted outcome remains serialized.
    fresh_attempts: list[dict[str, Any]] = []
    fresh_live: list[tuple[dict[str, Any], Any]] = []
    for factor in FRESH_CONTROL_FACTORS:
        row, solved = _solve(design, nucleus, level=FRESH24_LEVEL, factor=factor, previous=None)
        row.update({
            "family": family,
            "scale": label,
            "nucleus": nucleus,
            "seed_factor": float(factor),
            "attempt_role": "fresh24_independent_control",
            "independent_previous": None,
        })
        fresh_attempts.append(row)
        if solved is not None and _fresh_attempt_valid(row):
            fresh_live.append((row, solved))

    fresh_independent_rows_valid = bool(fresh_live)

    # Build and control every distinct fine branch.  Distinct branches are
    # retained even when they later fail; selection is only among fully
    # controlled localized branches.
    branch_candidates: list[dict[str, Any]] = []
    for branch_index, group in enumerate(fine_groups, start=1):
        fine_pair = group[0]
        fine_row, fine_solution = fine_pair
        domain_row, domain_solution = _solve(
            design,
            nucleus,
            level=DOMAIN_LEVEL,
            factor=None,
            previous=fine_solution,
        )
        domain_row.update({
            "family": family,
            "scale": label,
            "nucleus": nucleus,
            "attempt_role": "domain_continuation_branch_%d" % branch_index,
            "branch_id": "fine_branch_%d" % branch_index,
        })
        domain_profile_comparison = _profile_consistency(
            fine_pair, (domain_row, domain_solution)
        ) if domain_solution is not None else {"passed": False, "reason": "domain_solution_missing"}
        fresh_matches: list[tuple[dict[str, Any], Any]] = []
        fresh_comparisons: list[dict[str, Any]] = []
        for fresh_pair in fresh_live:
            comparison = _profile_consistency(fine_pair, fresh_pair)
            fresh_comparisons.append({
                "fresh_seed_factor": fresh_pair[0].get("seed_factor"),
                "profile_consistency": comparison,
                "passed": comparison.get("passed") is True,
            })
            if comparison.get("passed") is True:
                fresh_matches.append(fresh_pair)
        fresh_pair = fresh_matches[0] if fresh_matches else None
        if fresh_pair is None:
            fresh_row: dict[str, Any] = {
                "converged": False,
                "solver_message": "no independent fresh24 row on this fine branch",
                "attempt_role": "fresh24_branch_match_failed",
            }
            fresh_solution = None
        else:
            fresh_row, fresh_solution = fresh_pair
        branch_gate = _branch_numeric_gate(
            fine_row,
            domain_row,
            fresh_row,
            _original_attempt(fresh_row),
            fresh_attempts,
            domain_profile_comparison,
            fresh_comparisons,
        )
        fine_valid = bool(branch_gate["fine_valid"])
        domain_valid = bool(branch_gate["domain_valid"])
        fresh_initial_valid = bool(branch_gate["fresh_initial_valid"])
        fresh_effective_valid = bool(branch_gate["fresh_valid"])
        domain_energy_change = branch_gate["domain_energy_change"]
        fresh_energy_change = branch_gate["fresh_energy_change"]
        domain_radius_changes = branch_gate["domain_radius_changes"]
        fresh_radius_changes = branch_gate["fresh_radius_changes"]
        domain_energy_pass = bool(branch_gate["domain_refinement_pass"])
        fresh_energy_pass = bool(
            fresh_energy_change is not None
            and fresh_energy_change <= ENERGY_LIMIT_MEV
        )
        domain_radius_pass = bool(
            all(value is not None and value <= RADIUS_LIMIT_FM for value in domain_radius_changes.values())
        )
        fresh_radius_pass = bool(
            all(value is not None and value <= RADIUS_LIMIT_FM for value in fresh_radius_changes.values())
        )
        fresh_branch_profiles_pass = bool(branch_gate["fresh_profile_pass"])
        branch_acceptance = bool(branch_gate["accepted"])
        branch_candidates.append({
            "branch_id": "fine_branch_%d" % branch_index,
            "fine_seed_factors": [item[0].get("seed_factor") for item in group],
            "fine_selected": _jsonable(fine_row),
            "domain40_check": _jsonable(domain_row),
            "fresh24_check": _jsonable(fresh_row),
            "fresh24_initial_attempt": _jsonable(_original_attempt(fresh_row)),
            "fresh24_profile_comparisons": _jsonable(fresh_comparisons),
            "fresh24_attempts": _jsonable(fresh_attempts),
            "domain_profile_comparison": _jsonable(domain_profile_comparison),
            "fine_profile_comparisons": _jsonable(fine_comparisons),
            "fine_converged": fine_valid,
            "domain_converged": domain_valid,
            "fresh24_independent_initial_pass": fresh_initial_valid,
            "fresh24_effective_pass": fresh_effective_valid,
            "fresh24_independent_rows_valid": fresh_independent_rows_valid,
            "fresh24_valid_attempt_count": len(fresh_live),
            "fresh24_failed_attempt_count": len(fresh_attempts) - len(fresh_live),
            "fresh24_independent_requirement": "at_least_one_valid_previous_none_attempt_matching_this_branch",
            "fresh24_branch_profiles_pass": fresh_branch_profiles_pass,
            "domain_energy_change_MeV": domain_energy_change,
            "fresh24_energy_change_MeV": fresh_energy_change,
            "domain_species_radius_changes_fm": domain_radius_changes,
            "fresh24_species_radius_changes_fm": fresh_radius_changes,
            "domain_energy_pass": domain_energy_pass,
            "fresh24_energy_pass": fresh_energy_pass,
            "domain_species_radius_pass": domain_radius_pass,
            "fresh24_species_radius_pass": fresh_radius_pass,
            "terminal_protocol_acceptance": branch_acceptance,
            "terminal_localized": bool(domain_row.get("localized_vacuum_exterior") is True),
            "_fine_solution": fine_solution,
            "_domain_solution": domain_solution,
            "_fresh_solution": fresh_solution,
            "_fine_row": fine_row,
            "_domain_row": domain_row,
            "_fresh_row": fresh_row,
        })

    # If no fine branch exists, retain a synthetic branch record carrying the
    # independent fresh attempt and its exact failure state.
    if not branch_candidates:
        branch_candidates.append({
            "branch_id": "fine_branch_none",
            "fine_seed_factors": [],
            "fine_selected": {"converged": False, "solver_message": "no converged fine24 branch"},
            "domain40_check": {"converged": False, "solver_message": "no eligible fine branch"},
            "fresh24_check": _jsonable(fresh_attempts[0]) if fresh_attempts else {"converged": False},
            "fresh24_initial_attempt": _jsonable(fresh_attempts[0]) if fresh_attempts else {"converged": False},
            "fresh24_profile_comparisons": [],
            "fresh24_attempts": _jsonable(fresh_attempts),
            "fine_profile_comparisons": _jsonable(fine_comparisons),
            "fine_converged": False,
            "domain_converged": False,
            "fresh24_independent_initial_pass": bool(fresh_live),
            "fresh24_effective_pass": bool(fresh_live),
            "fresh24_independent_rows_valid": fresh_independent_rows_valid,
            "fresh24_valid_attempt_count": len(fresh_live),
            "fresh24_failed_attempt_count": len(fresh_attempts) - len(fresh_live),
            "fresh24_independent_requirement": "at_least_one_valid_previous_none_attempt_matching_this_branch",
            "fresh24_branch_profiles_pass": False,
            "domain_energy_change_MeV": None,
            "fresh24_energy_change_MeV": None,
            "domain_species_radius_changes_fm": {"neutron": None, "point_proton": None},
            "fresh24_species_radius_changes_fm": {"neutron": None, "point_proton": None},
            "domain_energy_pass": False,
            "fresh24_energy_pass": False,
            "domain_species_radius_pass": False,
            "fresh24_species_radius_pass": False,
            "terminal_protocol_acceptance": False,
            "terminal_localized": False,
            "_fine_solution": None,
            "_domain_solution": None,
            "_fresh_solution": None,
            "_fine_row": {"converged": False},
            "_domain_row": {"converged": False},
            "_fresh_row": fresh_attempts[0] if fresh_attempts else {"converged": False},
        })

    unmatched_fresh_factors = _fresh_branch_unmatched_factors(branch_candidates, fresh_attempts)
    fresh_branch_ambiguity = {
        "status": "UNRESOLVED_FRESH_BRANCH_AMBIGUITY" if unmatched_fresh_factors else "NO_UNMATCHED_FRESH_BRANCH",
        "unmatched_valid_seed_factors": unmatched_fresh_factors,
        "valid_attempt_count": len(fresh_live),
        "all_valid_attempts_assigned_to_retained_fine_branch": not bool(unmatched_fresh_factors),
    }
    controlled = [item for item in branch_candidates if item.get("terminal_protocol_acceptance") is True]
    selected = None
    if controlled:
        # Lower E/A-M means lower total static energy at fixed A.  This is a
        # declared lowest-found selection, never a global-ground-state claim.
        selected = min(
            controlled,
            key=lambda item: float(item["_domain_row"].get("energy_per_A_minus_M_MeV", math.inf)),
        )
    else:
        # A deterministic representative keeps all failure evidence visible.
        selected = min(
            branch_candidates,
            key=lambda item: (
                not bool(item.get("fine_converged")),
                not bool(item.get("domain_converged")),
                item.get("branch_id", ""),
            ),
        )
    terminal_acceptance = bool(
        selected.get("terminal_protocol_acceptance") is True
        and lineage_ok
        and not unmatched_fresh_factors
    )
    domain_row = selected.get("_domain_row", {"converged": False})
    fine_row = selected.get("_fine_row", {"converged": False})
    fresh_row = selected.get("_fresh_row", {"converged": False})
    binding = _binding(domain_row)
    if not terminal_acceptance:
        if not bool(domain_row.get("converged") is True):
            classification = "FAILED_NUMERICAL_PROTOCOL"
        elif not fresh_independent_rows_valid:
            classification = "FAILED_INDEPENDENT_FRESH_CONTROL"
        elif not bool(selected.get("fresh24_branch_profiles_pass")):
            classification = "FAILED_FRESH_BRANCH_LINEAGE_CONTROL"
        elif unmatched_fresh_factors:
            classification = "UNRESOLVED_FRESH_BRANCH_AMBIGUITY"
        elif domain_row.get("localized_vacuum_exterior") is not True:
            classification = "CONVERGED_UNBOUND_OR_NONLOCALIZED"
        else:
            classification = "CONVERGED_BUT_REFINEMENT_OR_RESIDUAL_LIMIT_MISSED"
    elif domain_row.get("binding_condition_E_per_A_lt_M") is True:
        classification = "NUMERICAL_STATIONARY_BOUND_CANDIDATE"
    else:
        classification = "NUMERICAL_STATIONARY_UNBOUND_LOCALIZED"
    return {
        "family": family,
        "scale": label,
        "nucleus": nucleus,
        "rho_choice": RHO_CHOICE,
        "fine_seed_factors": list(factors),
        "fine_seed_attempts": [_jsonable(row) for row in fine_attempts],
        "all_seed_outcomes_reported": bool(len(fine_attempts) == len(factors)),
        "fresh24_seed_factors": list(FRESH_CONTROL_FACTORS),
        "fresh24_attempts": [_jsonable(row) for row in fresh_attempts],
        "all_independent_fresh_outcomes_reported": bool(len(fresh_attempts) == len(FRESH_CONTROL_FACTORS)),
        "selected_seed_factor": fine_row.get("seed_factor"),
        "fine_selected": _jsonable(fine_row),
        "domain40_check": _jsonable(domain_row),
        "fresh24_check": _jsonable(fresh_row),
        "fresh24_initial_attempt": _jsonable(_original_attempt(fresh_row)),
        "fresh24_recovery_attempt": None,
        "fresh24_independent_rows_valid": fresh_independent_rows_valid,
        "fresh24_valid_attempt_count": len(fresh_live),
        "fresh24_failed_attempt_count": len(fresh_attempts) - len(fresh_live),
        "fresh24_independent_requirement": "at_least_one_valid_previous_none_attempt_matching_selected_branch",
        "fresh24_attempts": [_jsonable(row) for row in fresh_attempts],
        "fresh24_profile_comparisons": _jsonable(selected.get("fresh24_profile_comparisons", [])),
        "domain_profile_comparison": _jsonable(selected.get("domain_profile_comparison", {})),
        "fresh_branch_ambiguity": fresh_branch_ambiguity,
        "branch_candidates": [_jsonable({key: value for key, value in item.items() if not str(key).startswith("_")}) for item in branch_candidates],
        "lineage": {
            "status": "CONSISTENT" if lineage_ok else "UNRESOLVED_LINEAGE_MISMATCH",
            "same_scale_continuation_count": len(lineage_attempts),
            "continuation_attempts": [_jsonable(row) for row in lineage_attempts],
            "profile_comparisons": _jsonable(lineage_comparisons),
            "heuristic_signature_only": False,
            "max_scale_step_required": LINEAGE_MAX_SCALE_STEP,
        },
        "fine_same_scale_profile_comparisons": _jsonable(fine_comparisons),
        "fresh_same_scale_profile_comparisons": _jsonable([]),
        "selected_branch_id": selected.get("branch_id"),
        "lowest_found_branch_id": selected.get("branch_id") if controlled else None,
        "branch_selection_basis": "lowest_energy_among_fully_controlled_localized_branches" if controlled else "no_fully_controlled_branch",
        "refinement_summary": {
            "fine_converged": bool(fine_row.get("converged") is True),
            "fine_stationarity": bool(fine_row.get("stationarity_acceptance") is True),
            "fine_localized": bool(fine_row.get("localized_vacuum_exterior") is True),
            "domain_converged": bool(domain_row.get("converged") is True),
            "domain_stationarity": bool(domain_row.get("stationarity_acceptance") is True),
            "domain_localized": bool(domain_row.get("localized_vacuum_exterior") is True),
            "fresh24_converged": bool(fresh_row.get("converged") is True),
            "fresh24_stationarity": bool(fresh_row.get("stationarity_acceptance") is True),
            "fresh24_localized": bool(fresh_row.get("localized_vacuum_exterior") is True),
            "fresh24_independent_initial_pass": bool(_terminal_row_valid(_original_attempt(fresh_row))),
            "fresh24_independent_rows_valid": bool(selected.get("fresh24_independent_rows_valid")),
            "fresh24_valid_attempt_count": int(selected.get("fresh24_valid_attempt_count", len(fresh_live))),
            "fresh24_failed_attempt_count": int(selected.get("fresh24_failed_attempt_count", len(fresh_attempts) - len(fresh_live))),
            "fresh24_branch_profiles_pass": bool(selected.get("fresh24_branch_profiles_pass")),
            "fine_to_domain_energy_change_MeV": selected.get("domain_energy_change_MeV"),
            "fresh24_to_fine_energy_change_MeV": selected.get("fresh24_energy_change_MeV"),
            "fine_to_domain_species_radius_changes_fm": selected.get("domain_species_radius_changes_fm"),
            "fresh24_to_fine_species_radius_changes_fm": selected.get("fresh24_species_radius_changes_fm"),
            "domain_energy_pass": bool(selected.get("domain_energy_pass")),
            "fresh24_energy_pass": bool(selected.get("fresh24_energy_pass")),
            "domain_species_radius_pass": bool(selected.get("domain_species_radius_pass")),
            "fresh24_species_radius_pass": bool(selected.get("fresh24_species_radius_pass")),
            "all_terminal_rows_converged_stationary_localized_bound": bool(
                _terminal_row_valid(fine_row)
                and _terminal_row_valid(domain_row)
                and _fresh_attempt_valid(fresh_row)
            ),
            "original_static_residual_gates_preserved": True,
        },
        "terminal_classification": classification,
        "terminal_protocol_acceptance": terminal_acceptance,
        "terminal_localized": bool(domain_row.get("localized_vacuum_exterior") is True),
        "binding_per_A_MeV": None if binding is None else _number(binding),
        "binding_residual_to_target_MeV": None,
        "_selected_solution": selected.get("_fine_solution"),
        "_domain_solution": selected.get("_domain_solution"),
    }


def _lineage_public_valid(lineage: Any) -> bool:
    """Recompute retained continuation/lineage checks from numeric rows."""

    if not isinstance(lineage, Mapping):
        return True
    status = lineage.get("status")
    if status not in {None, "CONSISTENT", "NOT_APPLICABLE"}:
        return False
    # A case that declares a same-scale lineage must retain every step and
    # comparison; status text alone is never a substitute for those metrics.
    required = bool(lineage.get("same_scale_lineage_required") is True or lineage.get("scale_walks") is not None)
    attempts = lineage.get("continuation_attempts")
    comparisons = lineage.get("profile_comparisons")
    walks = lineage.get("scale_walks")
    if required:
        if not isinstance(attempts, list) or not isinstance(comparisons, list) or not isinstance(walks, list):
            return False
        if not walks:
            return False
        for item in attempts:
            if not _terminal_row_valid(item):
                return False
        for item in comparisons:
            if not isinstance(item, Mapping):
                return False
            comparison = item.get("profile_consistency")
            if comparison is None or not _profile_comparison_numeric_valid(comparison):
                return False
            if item.get("passed") is not True:
                return False
        for walk in walks:
            if not isinstance(walk, Mapping) or walk.get("converged") is not True:
                return False
            walk_attempts = walk.get("attempts")
            if not isinstance(walk_attempts, list):
                return False
            try:
                source_scale = float(walk["source_scale"])
                target_scale = float(walk["target_scale"])
                step_count = int(walk["step_count"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return False
            if not all(math.isfinite(value) for value in (source_scale, target_scale)):
                return False
            if step_count != len(walk_attempts):
                return False
            max_abs_step = 0.0
            steps_ok = True
            cursor = source_scale
            # An endpoint sourced at exactly the target scale is a valid
            # zero-step lineage; inserted walks must retain at least one row.
            if not walk_attempts:
                try:
                    if abs(source_scale - target_scale) > 1.0e-12:
                        return False
                except (KeyError, TypeError, ValueError, OverflowError):
                    return False
            for item in walk_attempts:
                try:
                    step = float(item["scale_step"])
                    item_scale = float(item["scale"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    return False
                if not math.isfinite(step) or abs(step) > LINEAGE_MAX_SCALE_STEP + 1.0e-12:
                    return False
                if not math.isfinite(item_scale) or abs(item_scale - (cursor + step)) > 1.0e-10:
                    return False
                cursor = item_scale
                max_abs_step = max(max_abs_step, abs(step))
                steps_ok = steps_ok and abs(step) <= LINEAGE_MAX_SCALE_STEP + 1.0e-12
                if not _terminal_row_valid(item):
                    return False
            if abs(cursor - target_scale) > 1.0e-10:
                return False
            if walk.get("steps_within_limit") is not steps_ok:
                return False
            reported_max = walk.get("max_abs_step")
            if not walk_attempts:
                if reported_max is not None:
                    return False
            else:
                try:
                    if not math.isfinite(float(reported_max)) or abs(float(reported_max) - max_abs_step) > 1.0e-10:
                        return False
                except (TypeError, ValueError, OverflowError):
                    return False
        if status != "CONSISTENT":
            return False
    elif status == "CONSISTENT" and comparisons is not None:
        # Non-root prediction rows may omit a same-scale walk, but if any
        # comparisons are retained, their numeric flags still need checking.
        if not isinstance(comparisons, list) or any(
            not isinstance(item, Mapping)
            or item.get("passed") is not True
            or not _profile_comparison_numeric_valid(item.get("profile_consistency"))
            for item in comparisons
        ):
            return False
    return True


def _terminal_public(case: Mapping[str, Any]) -> dict[str, Any]:
    return _jsonable({key: value for key, value in case.items() if not str(key).startswith("_")})


def _terminal_valid(case: Mapping[str, Any]) -> bool:
    if not isinstance(case, Mapping):
        return False
    branches = case.get("branch_candidates")
    selected: Mapping[str, Any] | None = None
    if isinstance(branches, list):
        selected = next(
            (
                item for item in branches
                if isinstance(item, Mapping) and item.get("branch_id") == case.get("selected_branch_id")
            ),
            None,
        )
        if selected is None:
            return False
    else:
        controls = case.get("terminal_control_rows")
        if not isinstance(controls, Mapping):
            selected = case
        else:
            selected = {
                "fine_selected": controls.get("fine_selected", {}),
                "domain40_check": controls.get("domain40_check", {}),
                "fresh24_check": controls.get("fresh24_check", {}),
                "fresh24_initial_attempt": controls.get("fresh24_initial_attempt", {}),
                "fresh24_attempts": controls.get("fresh24_attempts", []),
                "domain_profile_comparison": controls.get("domain_profile_comparison", {}),
                "fresh24_profile_comparisons": controls.get("fresh24_profile_comparisons", []),
            }
    gate = _branch_numeric_gate(
        selected.get("fine_selected", {}),
        selected.get("domain40_check", {}),
        selected.get("fresh24_check", {}),
        selected.get("fresh24_initial_attempt", selected.get("fresh24_check", {})),
        selected.get("fresh24_attempts", []),
        selected.get("domain_profile_comparison", {}),
        selected.get("fresh24_profile_comparisons", []),
    )
    lineage = case.get("lineage", {})
    lineage_valid = _lineage_public_valid(lineage)
    unmatched_fresh_factors = _fresh_branch_unmatched_factors(
        branches,
        case.get("fresh24_attempts", selected.get("fresh24_attempts", [])),
    )
    expected = bool(gate["accepted"] and lineage_valid and not unmatched_fresh_factors)
    if case.get("terminal_protocol_acceptance") is not expected:
        return False
    expected_localized = bool(gate["domain_valid"] and selected.get("domain40_check", {}).get("localized_vacuum_exterior") is True)
    if case.get("terminal_localized") is not expected_localized:
        return False
    if expected:
        try:
            reported = float(case.get("binding_per_A_MeV"))
            expected_binding = -float(selected["domain40_check"]["energy_per_A_minus_M_MeV"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(reported) or abs(reported - expected_binding) > 1.0e-10:
            return False
    return expected


def _lineage_walk(
    family: str,
    nucleus: str,
    target_scale: float,
    source_label: str,
    source: Any,
) -> tuple[tuple[dict[str, Any], Any] | None, list[dict[str, Any]]]:
    """Continue an endpoint to a trial scale with steps no larger than .02."""

    if source is None:
        return None, [{"source": source_label, "converged": False, "reason": "missing_source"}]
    try:
        current_solution = source
        current_scale = float(source.design.scale)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None, [{"source": source_label, "converged": False, "reason": "invalid_source"}]
    attempts: list[dict[str, Any]] = []
    while abs(target_scale - current_scale) > 1.0e-12:
        delta = target_scale - current_scale
        step = math.copysign(min(abs(delta), LINEAGE_MAX_SCALE_STEP), delta)
        next_scale = current_scale + step
        design = _make_design(family, _scale_label(next_scale))
        row, solved = _solve(
            design,
            nucleus,
            level=FINE_LEVEL,
            factor=None,
            previous=current_solution,
        )
        row.update({
            "family": family,
            "scale": _scale_label(next_scale),
            "nucleus": nucleus,
            "seed_factor": None,
            "attempt_role": "lineage_scale_walk",
            "predecessor_label": source_label,
            "scale_step": step,
        })
        attempts.append(row)
        if solved is None or not _terminal_row_valid(row):
            return None, attempts
        current_solution = solved
        current_scale = next_scale
    final_row = attempts[-1] if attempts else {
        "family": family,
        "scale": _scale_label(target_scale),
        "nucleus": nucleus,
        "converged": True,
        "attempt_role": "lineage_same_scale_source",
    }
    return (final_row, current_solution), attempts


def _lineage_sources_for_case(case: Mapping[str, Any], label: str) -> tuple[str, Any] | None:
    if not isinstance(case, Mapping):
        return None
    source = case.get("_selected_solution") or case.get("_domain_solution")
    if source is None:
        return None
    return str(label), source


def _find_root(
    family: str,
    coarse_bracket: Mapping[str, Any],
    target: float,
    *,
    cache: dict[Any, dict[str, Any]],
) -> dict[str, Any]:
    """Refine a localized bracket with live branch-lineage controls."""

    evaluations: list[dict[str, Any]] = []
    walk_cache: dict[tuple[str, str, str, int], tuple[tuple[dict[str, Any], Any] | None, list[dict[str, Any]]]] = {}

    def source_token(source_specs: Sequence[tuple[str, Any]]) -> tuple[Any, ...]:
        tokens = []
        for label, source in source_specs:
            try:
                source_scale = float(source.design.scale)
            except (AttributeError, TypeError, ValueError, OverflowError):
                source_scale = None
            # Object identity distinguishes two same-scale branch solutions;
            # a scale label alone is not lineage provenance.
            tokens.append((str(label), source_scale, id(source)))
        return tuple(tokens)

    def prepare_lineage(scale: float, source_specs: Sequence[tuple[str, Any]]) -> tuple[list[tuple[str, Any]], list[dict[str, Any]], bool]:
        predecessors: list[tuple[str, Any]] = []
        walks: list[dict[str, Any]] = []
        all_passed = True
        for source_label, source in source_specs:
            try:
                source_scale = float(source.design.scale)
            except (AttributeError, TypeError, ValueError, OverflowError):
                source_scale = math.nan
            walk_key = (str(source_label), _scale_label(scale), format(source_scale, ".17g") if math.isfinite(source_scale) else "invalid", id(source))
            if walk_key not in walk_cache:
                walk_cache[walk_key] = _lineage_walk(family, "Ca40", scale, str(source_label), source)
            walked, attempts = walk_cache[walk_key]
            walks.append({
                "source": str(source_label),
                "source_scale": None if not math.isfinite(source_scale) else source_scale,
                "target_scale": _number(scale),
                "step_count": len(attempts),
                "max_abs_step": None if not attempts else _number(max(abs(float(item.get("scale_step", 0.0))) for item in attempts)),
                "steps_within_limit": all(abs(float(item.get("scale_step", 0.0))) <= LINEAGE_MAX_SCALE_STEP + 1.0e-12 for item in attempts),
                "attempts": [_jsonable(item) for item in attempts],
                "converged": walked is not None,
            })
            if walked is None:
                all_passed = False
            else:
                predecessors.append((str(source_label), walked[1]))
        return predecessors, walks, all_passed

    def evaluate(
        scale: float,
        *,
        multi_seed: bool = False,
        source_specs: Sequence[tuple[str, Any]] = (),
        trial_role: str = "root_trial",
    ) -> dict[str, Any]:
        token = source_token(source_specs)
        key = (family, _scale_label(scale), "Ca40", bool(multi_seed), token)
        if key not in cache:
            predecessors, walks, walk_ok = prepare_lineage(scale, source_specs)
            case = _terminal_case(
                family,
                scale,
                "Ca40",
                multi_seed=multi_seed,
                lineage_predecessors=predecessors,
            )
            lineage = dict(case.get("lineage", {}))
            lineage["trial_role"] = str(trial_role)
            lineage["scale_walks"] = walks
            lineage["scale_walks_passed"] = bool(walk_ok)
            lineage["same_scale_lineage_required"] = bool(source_specs)
            # Root evaluations are never allowed to become accepted merely
            # because independent seeds happened to converge.  A trial with
            # no tracked endpoint source has no lineage proof and therefore
            # remains unresolved; prediction-only terminal cases do not use
            # this evaluator and retain their independent-control semantics.
            if not source_specs:
                lineage["status"] = "UNRESOLVED_LINEAGE_SOURCE"
                lineage["same_scale_lineage_required"] = True
            if source_specs and not walk_ok:
                lineage["status"] = "UNRESOLVED_LINEAGE_STEP"
            case["lineage"] = lineage
            if not source_specs or (source_specs and not walk_ok):
                case["terminal_protocol_acceptance"] = False
                case["terminal_classification"] = "UNRESOLVED_LINEAGE_MISMATCH"
            cache[key] = case
        case = cache[key]
        if case not in evaluations:
            evaluations.append(case)
        return case

    lo = float(coarse_bracket["scale_lo"])
    hi = float(coarse_bracket["scale_hi"])
    if not (SCALE_MIN <= lo < hi <= SCALE_MAX):
        return {
            "status": "NO_LOCALIZED_BRACKET_UNRESOLVED",
            "reason": "coarse bracket outside bounded scale domain",
            "coarse_bracket": _public_bracket(coarse_bracket),
            "evaluations": [],
        }
    coarse_left = coarse_bracket.get("_left_solution")
    coarse_right = coarse_bracket.get("_right_solution")
    left_sources = [("coarse_lo", coarse_left)] if coarse_left is not None else []
    right_sources = [("coarse_hi", coarse_right)] if coarse_right is not None else []
    left = evaluate(lo, source_specs=left_sources, trial_role="coarse_left_endpoint")
    right = evaluate(hi, source_specs=right_sources, trial_role="coarse_right_endpoint")
    left_value = left.get("binding_per_A_MeV") if _terminal_valid(left) else None
    right_value = right.get("binding_per_A_MeV") if _terminal_valid(right) else None
    if left_value is None:
        left = evaluate(lo, multi_seed=True, source_specs=left_sources, trial_role="coarse_left_endpoint_multiseed")
        left_value = left.get("binding_per_A_MeV") if _terminal_valid(left) else None
    if right_value is None:
        right = evaluate(hi, multi_seed=True, source_specs=right_sources, trial_role="coarse_right_endpoint_multiseed")
        right_value = right.get("binding_per_A_MeV") if _terminal_valid(right) else None

    # Coarse endpoints only guide the search.  The interior ladder supplies
    # accepted terminal endpoints when a coarse domain residual fails, while
    # each trial is continued from a tracked endpoint with <=.02 scale steps.
    terminal_entries: list[tuple[float, dict[str, Any], float]] = []
    for scale_value, case, value in ((lo, left, left_value), (hi, right, right_value)):
        if value is not None and _terminal_valid(case):
            terminal_entries.append((scale_value, case, float(value)))
    if len(terminal_entries) < 2 or all((entry[2] - target) * (terminal_entries[0][2] - target) > 0.0 for entry in terminal_entries[1:]):
        # Walk from the low-scale endpoint in predeclared <=.02 steps.  This
        # avoids a broad fraction ladder while preserving the bounded
        # continuation lineage needed to locate the first terminal sign
        # change.
        interior_scales = []
        cursor = lo + min(LINEAGE_MAX_SCALE_STEP, hi - lo)
        while cursor < hi - 1.0e-12:
            interior_scales.append(cursor)
            cursor += min(LINEAGE_MAX_SCALE_STEP, hi - lo)
        for interior_scale in interior_scales:
            source_specs: list[tuple[str, Any]] = []
            for label, candidate in (("left_endpoint", left), ("right_endpoint", right)):
                if not _terminal_valid(candidate):
                    continue
                source = candidate.get("_selected_solution") or candidate.get("_domain_solution")
                if source is not None:
                    source_specs.append((label, source))
            case = evaluate(interior_scale, source_specs=source_specs, trial_role="interior_bracket_probe")
            value = case.get("binding_per_A_MeV") if _terminal_valid(case) else None
            if value is None:
                case = evaluate(interior_scale, multi_seed=True, source_specs=source_specs, trial_role="interior_bracket_probe_multiseed")
                value = case.get("binding_per_A_MeV") if _terminal_valid(case) else None
            if value is not None and _terminal_valid(case):
                terminal_entries.append((interior_scale, case, float(value)))
        terminal_entries.sort(key=lambda item: item[0])
        candidate_pairs = [
            (right_scale - left_scale, left_scale, right_scale, left_case, right_case, left_value_item, right_value_item)
            for i, (left_scale, left_case, left_value_item) in enumerate(terminal_entries)
            for right_scale, right_case, right_value_item in terminal_entries[i + 1 :]
            if (left_value_item - target) * (right_value_item - target) <= 0.0
        ]
        if candidate_pairs:
            _, lo, hi, left, right, left_value, right_value = min(candidate_pairs, key=lambda item: item[0])

    if left_value is None or right_value is None or (float(left_value) - target) * (float(right_value) - target) > 0.0:
        return {
            "status": "NO_TERMINAL_LOCALIZED_BRACKET_UNRESOLVED",
            "reason": "24/40-fm terminal controls did not preserve a localized sign change",
            "coarse_bracket": _public_bracket(coarse_bracket),
            "terminal_endpoint_scales": [lo, hi],
            "evaluations": [_terminal_public(item) for item in evaluations],
        }

    iterations = 0
    max_iterations = 32
    lineage_updates: list[dict[str, Any]] = []
    while hi - lo > ROOT_WIDTH_LIMIT and iterations < max_iterations:
        mid = 0.5 * (lo + hi)
        source_specs = []
        for label, candidate in (("left_endpoint", left), ("right_endpoint", right)):
            source = candidate.get("_selected_solution") or candidate.get("_domain_solution")
            if source is not None:
                source_specs.append((label, source))
        middle = evaluate(mid, source_specs=source_specs, trial_role="bisection_midpoint")
        middle_value = middle.get("binding_per_A_MeV") if _terminal_valid(middle) else None
        if middle_value is None:
            middle = evaluate(mid, multi_seed=True, source_specs=source_specs, trial_role="bisection_midpoint_multiseed")
            middle_value = middle.get("binding_per_A_MeV") if _terminal_valid(middle) else None
        if middle_value is None:
            denominator = float(right_value) - float(left_value)
            secant_fraction = 0.5 if denominator == 0.0 else (target - float(left_value)) / denominator
            candidate_scales = [
                lo + (hi - lo) * min(0.95, max(0.05, secant_fraction)),
                lo + 0.125 * (hi - lo),
                lo + 0.875 * (hi - lo),
            ]
            for candidate_scale in candidate_scales:
                if candidate_scale <= lo + 1.0e-12 or candidate_scale >= hi - 1.0e-12:
                    continue
                candidate = evaluate(candidate_scale, source_specs=source_specs, trial_role="safeguarded_interior_probe")
                candidate_value = candidate.get("binding_per_A_MeV") if _terminal_valid(candidate) else None
                if candidate_value is None:
                    candidate = evaluate(candidate_scale, multi_seed=True, source_specs=source_specs, trial_role="safeguarded_interior_probe_multiseed")
                    candidate_value = candidate.get("binding_per_A_MeV") if _terminal_valid(candidate) else None
                if candidate_value is not None:
                    middle, middle_value = candidate, candidate_value
                    mid = candidate_scale
                    break
        if middle_value is None:
            return {
                "status": "ROOT_UNRESOLVED_NUMERICAL_FAILURE",
                "reason": "terminal midpoint and safeguarded interior probes did not provide a localized accepted state",
                "coarse_bracket": _public_bracket(coarse_bracket),
                "remaining_bracket": [lo, hi],
                "evaluations": [_terminal_public(item) for item in evaluations],
            }
        lineage_updates.append({
            "scale": _number(mid),
            "lineage_status": middle.get("lineage", {}).get("status"),
            "terminal_valid": _terminal_valid(middle),
        })
        if (float(left_value) - target) * (float(middle_value) - target) <= 0.0:
            hi, right, right_value = mid, middle, middle_value
        else:
            lo, left, left_value = mid, middle, middle_value
        iterations += 1

    if hi - lo > ROOT_WIDTH_LIMIT:
        return {
            "status": "ROOT_UNRESOLVED_NUMERICAL_FAILURE",
            "reason": "root width cap was not reached within bounded iterations",
            "coarse_bracket": _public_bracket(coarse_bracket),
            "remaining_bracket": [lo, hi],
            "evaluations": [_terminal_public(item) for item in evaluations],
        }
    left_residual = abs(float(left_value) - target)
    right_residual = abs(float(right_value) - target)
    root_scale = lo if left_residual <= right_residual else hi
    final_sources = []
    for label, candidate in (("left_endpoint", left), ("right_endpoint", right)):
        source = candidate.get("_selected_solution") or candidate.get("_domain_solution")
        if source is not None:
            final_sources.append((label, source))
    final = evaluate(root_scale, multi_seed=True, source_specs=final_sources, trial_role="final_multiseed_replacement")
    final_value = final.get("binding_per_A_MeV") if _terminal_valid(final) else None
    final_residual = math.inf if final_value is None else abs(float(final_value) - target)
    lineage_ok = bool(
        _terminal_valid(left)
        and _terminal_valid(right)
        and _terminal_valid(final)
        and final.get("lineage", {}).get("status") == "CONSISTENT"
        and all(item.get("terminal_valid") is True and item.get("lineage_status") == "CONSISTENT" for item in lineage_updates)
    )
    accepted = bool(
        lineage_ok
        and final_residual <= ROOT_RESIDUAL_LIMIT_MEV
        and hi - lo <= ROOT_WIDTH_LIMIT
    )
    final["binding_residual_to_target_MeV"] = None if final_value is None else _number(float(final_value) - target)
    return {
        "status": "CALIBRATED_CONDITIONAL_ROOT" if accepted else "ROOT_RESIDUAL_OR_PROTOCOL_UNRESOLVED",
        "scale": _number(root_scale),
        "scale_label": _scale_label(root_scale),
        "target_B_nuc_per_A_MeV": _number(target),
        "binding_per_A_MeV": None if final_value is None else _number(final_value),
        "binding_residual_MeV": None if final_value is None else _number(float(final_value) - target),
        # Retain the exact coarse bracket used to start the terminal search so
        # the serialized validator can recompute its dimensions, target, and
        # sign change rather than trusting the refined endpoint alone.
        "coarse_bracket": _public_bracket(coarse_bracket),
        "bracket_final": [_number(lo), _number(hi)],
        "bracket_width": _number(hi - lo),
        "root_residual_limit_MeV": ROOT_RESIDUAL_LIMIT_MEV,
        "root_width_limit": ROOT_WIDTH_LIMIT,
        "iterations": iterations,
        "lineage_required": True,
        "lineage_consistency_verified": lineage_ok,
        "lineage_update_checks": lineage_updates,
        "terminal_protocol_acceptance": bool(final.get("terminal_protocol_acceptance") is True),
        "localized": bool(final.get("terminal_localized") is True),
        "accepted": accepted,
        "evaluations": [_terminal_public(item) for item in evaluations],
        "final_calibration_state": _terminal_public(final),
        "_final_case": final,
    }

def _radius_mapping(row: Mapping[str, Any], nucleus: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the declared one-body charge convention to raw species moments."""

    try:
        N = float(static.BRIDGE_NUCLEI[nucleus]["N"])
        Z = float(static.BRIDGE_NUCLEI[nucleus]["Z"])
        rpp = float(row["rms_point_proton_radius_fm"])
        rn = float(row["rms_neutron_radius_fm"])
        rb = float(row["rms_baryon_radius_fm"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise FiniteStaticBridgeError(f"species radii missing for {nucleus}") from exc
    ninputs = inputs["nucleon_charge_inputs"]
    rp = float(ninputs["proton_charge_radius_fm"])
    rn2 = float(ninputs["neutron_mean_square_charge_radius_fm2"])
    mass = float(ninputs["common_mass_MeV"])
    hbarc = float(ninputs["hbarc_MeV_fm"])
    r1b2 = rpp**2 + rp**2 + (N / Z) * rn2 + 3.0 * (hbarc / mass) ** 2 / 4.0
    if r1b2 <= 0.0 or not math.isfinite(r1b2):
        raise FiniteStaticBridgeError(f"one-body charge mapping is nonpositive for {nucleus}")
    rch_eval = float(inputs["charge_radii"]["values"][nucleus]["R_ch_fm"])
    delta_required = rch_eval**2 - r1b2
    # Sensitivities are derivatives of the reported squared-radius diagnostic;
    # they are not uncertainty propagation or a fitted nuisance model.
    sensitivities = {
        "delta_required_per_Rpp_fm": _number(-2.0 * rpp),
        "delta_required_per_rp_fm": _number(-2.0 * rp),
        "delta_required_per_rn2_fm2": _number(-N / Z),
        "delta_required_per_Rch_eval_fm": _number(2.0 * rch_eval),
        "delta_required_per_common_M_MeV": _number(3.0 * hbarc**2 / (2.0 * mass**3)),
    }
    return {
        "rms_point_proton_radius_fm": _number(rpp),
        "rms_neutron_radius_fm": _number(rn),
        "rms_baryon_radius_fm": _number(rb),
        "raw_species_radius_integrals_dimensionless": row.get("species_radius_integrals_dimensionless"),
        "one_body_charge_mapping": "R1b^2=Rpp^2+rp^2+(N/Z)rn^2+3(hbarc/MN)^2/4",
        "point_proton_square_fm2": _number(rpp**2),
        "proton_intrinsic_square_fm2": _number(rp**2),
        "neutron_mean_square_charge_fm2": _number(rn2),
        "darwin_foldy_square_fm2": _number(3.0 * (hbarc / mass) ** 2 / 4.0),
        "R1b_square_fm2": _number(r1b2),
        "R1b_fm": _number(math.sqrt(r1b2)),
        "Rch_evaluated_fm": _number(rch_eval),
        "Rch_evaluated_sigma_fm": _number(inputs["charge_radii"]["values"][nucleus]["R_ch_sigma_fm"]),
        "delta_required_fm2": _number(delta_required),
        "uncomputed_corrections": {
            "delta_SO_fm2": None,
            "delta_CM_fm2": None,
            "delta_2body_fm2": None,
            "relativistic_beyond_declared_order": None,
        },
        "common_mass_convention": "MN=939 MeV is a rest-mass/kinematic approximation; physical n-p mass differences are not computed",
        "sensitivity_of_delta_required": sensitivities,
        "uncertainty_status": "input uncertainties shown separately; no method/model error bar or empirical PASS",
    }


def _envelope_derivative(
    family: str,
    root_scale: float,
    final_case: Mapping[str, Any],
    *,
    cache: dict[Any, dict[str, Any]],
) -> dict[str, Any]:
    """Check the live envelope identity with two centered scale differences."""

    base = final_case.get("domain40_check", {})
    components = base.get("energy_components_MeV", {}) if isinstance(base, Mapping) else {}
    try:
        egrad = float(components["T_W"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"status": "UNRESOLVED_MISSING_SCALAR_GRADIENT_COMPONENT", "steps": []}
    rows: list[dict[str, Any]] = []
    root_source = final_case.get("_selected_solution")
    derivative_lineage = () if root_source is None else (("root_calibration", root_source),)
    for h in DERIVATIVE_STEPS:
        plus_scale, minus_scale = root_scale + h, root_scale - h
        if minus_scale < SCALE_MIN or plus_scale > SCALE_MAX:
            rows.append({"step_ds": h, "status": "OUTSIDE_BOUNDED_DOMAIN"})
            continue
        lineage_token = (("root_calibration", _number(root_scale)),) if root_source is not None else ()
        plus_key = (family, _scale_label(plus_scale), "Ca40", False, lineage_token)
        minus_key = (family, _scale_label(minus_scale), "Ca40", False, lineage_token)
        if plus_key not in cache:
            cache[plus_key] = _terminal_case(family, plus_scale, "Ca40", multi_seed=False, lineage_predecessors=derivative_lineage)
        if minus_key not in cache:
            cache[minus_key] = _terminal_case(family, minus_scale, "Ca40", multi_seed=False, lineage_predecessors=derivative_lineage)
        plus, minus = cache[plus_key], cache[minus_key]
        # Preserve a failed single-seed edge attempt, then use the explicitly
        # allowed multi-seed/warm-start recovery before declaring the centered
        # derivative unavailable.
        if not _terminal_valid(plus):
            recovery_key = (family, _scale_label(plus_scale), "Ca40", True, lineage_token)
            if recovery_key not in cache:
                cache[recovery_key] = _terminal_case(family, plus_scale, "Ca40", multi_seed=True, lineage_predecessors=derivative_lineage)
            plus = cache[recovery_key]
        if not _terminal_valid(minus):
            recovery_key = (family, _scale_label(minus_scale), "Ca40", True, lineage_token)
            if recovery_key not in cache:
                cache[recovery_key] = _terminal_case(family, minus_scale, "Ca40", multi_seed=True, lineage_predecessors=derivative_lineage)
            minus = cache[recovery_key]
        ep = plus.get("domain40_check", {}).get("energy_per_A_minus_M_MeV")
        em = minus.get("domain40_check", {}).get("energy_per_A_minus_M_MeV")
        accepted = bool(_terminal_valid(plus) and _terminal_valid(minus) and ep is not None and em is not None)
        if not accepted:
            rows.append({"step_ds": h, "status": "TERMINAL_SIDE_UNRESOLVED", "plus": _terminal_public(plus), "minus": _terminal_public(minus)})
            continue
        # Total E = A*(E/A-M)+A*M.  The centered difference therefore uses
        # per-nucleon energy multiplied by A; dB/ds is the negative of dE/ds.
        A = int(static.BRIDGE_NUCLEI["Ca40"]["A"])
        dE_ds = A * (float(ep) - float(em)) / (2.0 * h)
        predicted_dE_ds = 2.0 * egrad / root_scale
        rows.append({
            "step_ds": h,
            "status": "COMPUTED",
            "E_plus_per_A_minus_M_MeV": float(ep),
            "E_minus_per_A_minus_M_MeV": float(em),
            "dE_ds_centered_MeV": _number(dE_ds),
            "dB_ds_centered_MeV": _number(-dE_ds),
            "predicted_dE_ds_2Egrad_over_s_MeV": _number(predicted_dE_ds),
            "predicted_dB_ds_minus_2Egrad_over_s_MeV": _number(-predicted_dE_ds),
            "relative_error_dE_ds": _number(abs(dE_ds - predicted_dE_ds) / max(abs(dE_ds), abs(predicted_dE_ds), 1.0e-12)),
            "plus_terminal_acceptance": bool(plus.get("terminal_protocol_acceptance") is True),
            "minus_terminal_acceptance": bool(minus.get("terminal_protocol_acceptance") is True),
        })
    computed = [row for row in rows if row.get("status") == "COMPUTED"]
    return {
        "status": "COMPUTED_ENVELOPE_DERIVATIVE_CHECK" if len(computed) == len(DERIVATIVE_STEPS) else "UNRESOLVED_ENVELOPE_DERIVATIVE_CHECK",
        "identity": "dE_*/ds=2*T_W/s; dB_th/ds=-2*T_W/s",
        "E_grad_definition": "T_W scalar canonical gradient energy from live terminal domain40 state",
        "E_grad_MeV": _number(egrad),
        "steps": rows,
        "two_centered_steps_computed": len(computed) == len(DERIVATIVE_STEPS),
        "max_relative_error": None if not computed else _number(max(float(row["relative_error_dE_ds"]) for row in computed)),
        "repeat_root_domain_control": {
            "root_scale_used": _number(root_scale),
            "domain40_terminal_acceptance": bool(final_case.get("terminal_protocol_acceptance") is True),
            "fresh24_terminal_acceptance": _terminal_row_valid(final_case.get("fresh24_initial_attempt", {})),
            "domain_and_fresh24_controls_are_separate": True,
        },
        "not_a_stability_proof": True,
    }


def _prediction_case(family: str, scale: float, nucleus: str, *, cache: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    key = (family, _scale_label(scale), nucleus, True, ())
    if key not in cache:
        cache[key] = _terminal_case(family, scale, nucleus, multi_seed=True)
    case = cache[key]
    return case


def _frozen_prediction(family: str, scale: float, nucleus: str, inputs: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    domain = case.get("domain40_check", {})
    target = _input_targets(inputs)[nucleus]
    accepted = bool(case.get("terminal_protocol_acceptance") is True)
    model_binding = case.get("binding_per_A_MeV") if accepted else None
    unaccepted_model_binding = case.get("binding_per_A_MeV") if not accepted else None
    prediction = {
        "family": family,
        "scale": _scale_label(scale),
        "nucleus": nucleus,
        "rho_choice": RHO_CHOICE,
        "calibration_status": "FROZEN_SAME_S_AS_CA40" if nucleus != "Ca40" else "ANCHOR_STATE_NOT_USED_TO_SELECT_RADIUS",
        "terminal_classification": case.get("terminal_classification"),
        "terminal_protocol_acceptance": accepted,
        "localized_vacuum_exterior": bool(case.get("terminal_localized") is True),
        "lineage": case.get("lineage", {"status": "NOT_APPLICABLE"}),
        "fine_seed_factors": case.get("fine_seed_factors"),
        "fine_seed_attempts": case.get("fine_seed_attempts"),
        "all_seed_outcomes_reported": case.get("all_seed_outcomes_reported"),
        "fresh24_seed_factors": case.get("fresh24_seed_factors"),
        "fresh24_attempts": case.get("fresh24_attempts"),
        "all_independent_fresh_outcomes_reported": case.get("all_independent_fresh_outcomes_reported"),
        "fresh_branch_ambiguity": case.get("fresh_branch_ambiguity"),
        "model_binding_per_A_MeV": model_binding,
        "unaccepted_domain_binding_per_A_MeV": unaccepted_model_binding,
        "binding_comparison": {
            "B_atom_input_per_A_MeV": target["B_atom_per_A_MeV"],
            "B_nuc_target_per_A_MeV": target["B_nuc_target_per_A_MeV"],
            "model_minus_B_nuc_target_per_A_MeV": None if model_binding is None else _number(float(model_binding) - target["B_nuc_target_per_A_MeV"]),
            "input_role": "sole_calibration_anchor" if nucleus == "Ca40" else "heldout_descriptive_comparison_not_used_for_root",
            "experimental_uncertainty_MeV_per_A": target["B_atom_per_A_sigma_MeV"],
            "electronic_convention_correction_total_MeV": target["electron_correction_total_MeV"],
        },
        "radii": None,
        "all_seed_outcomes_reported": bool(case.get("all_seed_outcomes_reported") is True),
        "fine_seed_attempt_count": len(case.get("fine_seed_attempts", [])),
        # Keep every terminal row visible, including numerical failures.  The
        # rows are profile-stripped and preserve the gate evidence behind a
        # failed frozen comparison without promoting that row to a result.
        "terminal_control_rows": {
            "fine_selected": case.get("fine_selected"),
            "fine_seed_attempts": case.get("fine_seed_attempts"),
            "domain40_check": case.get("domain40_check"),
            "fresh24_check": case.get("fresh24_check"),
            "fresh24_initial_attempt": case.get("fresh24_initial_attempt"),
            "fresh24_recovery_attempt": case.get("fresh24_recovery_attempt"),
            "fresh24_seed_factors": case.get("fresh24_seed_factors"),
            "fresh24_attempts": case.get("fresh24_attempts"),
            "fresh24_profile_comparisons": case.get("fresh24_profile_comparisons"),
            "domain_profile_comparison": case.get("domain_profile_comparison"),
        },
        "refinement_summary": case.get("refinement_summary"),
        "control_boundary": "static TF only; no shell, spin-orbit, pairing, finite-nucleus Dirac or RRPA terms",
    }
    if case.get("terminal_protocol_acceptance") is True:
        prediction["radii"] = _radius_mapping(domain, nucleus, inputs)
    else:
        prediction["radii"] = {"status": "UNRESOLVED_NUMERICAL_PROTOCOL", "raw_species_radii_available": bool(domain.get("rms_point_proton_radius_fm") is not None)}
    return prediction


def _design_metadata(design: Any) -> dict[str, Any]:
    return _jsonable({
        "family": design.family,
        "target_y": design.target_y,
        "correlated_scale": design.scale,
        "M_MeV": design.M,
        "W0_MeV": design.W0,
        "momega_MeV": design.momega,
        "gomega": design.gomega,
        "gs": design.gs,
        "q_momega_over_W0": design.q,
        "degeneracy_per_species": design.d,
        "n0_fm_minus3": static.N0_FM3,
        "coefficients_U_over_W0^4_z2_z3_z4": list(design.coefficients_dim),
        "potential_source": "source_complete_scaling_saturation_audit.inverse_potential_jet",
        "correlated_scaling": "W0=s W0_base; coefficients=coefficients_base/s^4; gomega,MN,momega fixed",
        "calibration_inputs_not_predictions": True,
    })


def _family_calibration(family: str, inputs: Mapping[str, Any], *, cache: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    targets = _input_targets(inputs)
    target = targets["Ca40"]["B_nuc_target_per_A_MeV"]
    fixed: dict[str, dict[str, Any]] = {}
    for scale in EXPLORATORY_SCALES:
        probe = _coarse_probe(family, scale)
        probe["probe_role"] = "required_fixed_exploratory_scale"
        fixed[_scale_label(scale)] = probe
    adaptive = _adaptive_probe_series(family, fixed, target)
    all_probes = [fixed[_scale_label(scale)] for scale in EXPLORATORY_SCALES] + adaptive
    brackets = _coarse_brackets(all_probes, target)
    if not brackets:
        return {
            "family": family,
            "rho_choice": RHO_CHOICE,
            "calibration_status": "NO_LOCALIZED_BRACKET_UNRESOLVED",
            "target": targets["Ca40"],
            "exploratory_scales": [_number(scale) for scale in EXPLORATORY_SCALES],
            "exploratory_attempts": [_public_probe(probe) for probe in all_probes],
            "adaptive_attempt_count": len(adaptive),
            "localized_brackets": [],
            "root": {"status": "NO_LOCALIZED_BRACKET_UNRESOLVED", "accepted": False},
            "frozen_predictions": [],
        }
    root = _find_root(family, brackets[0], target, cache=cache)
    root_public = {key: value for key, value in root.items() if key != "_final_case"}
    root_case = root.get("_final_case")
    derivative = None
    predictions: list[dict[str, Any]] = []
    if root_case is not None and root.get("accepted") is True:
        root_scale = float(root["scale"])
        derivative = _envelope_derivative(family, root_scale, root_case, cache=cache)
        for nucleus in BRIDGE_NUCLEI:
            pred_case = _prediction_case(family, root_scale, nucleus, cache=cache)
            predictions.append(_frozen_prediction(family, root_scale, nucleus, inputs, pred_case))
    return {
        "family": family,
        "rho_choice": RHO_CHOICE,
        "calibration_status": "CALIBRATED_CONDITIONAL_ROOT" if root.get("accepted") is True else str(root.get("status")),
        "target": targets["Ca40"],
        "exploratory_scales": [_number(scale) for scale in EXPLORATORY_SCALES],
        "exploratory_attempts": [_public_probe(probe) for probe in all_probes],
        "adaptive_attempt_count": len(adaptive),
        "localized_brackets": [_public_bracket(bracket) for bracket in brackets],
        "root": _jsonable(root_public),
        "envelope_derivative": None if derivative is None else _jsonable(derivative),
        "frozen_predictions": predictions,
    }


def calculate() -> dict[str, Any]:
    """Run the complete conditional bridge computation with live solver rows."""

    inputs = _load_inputs()
    targets = _input_targets(inputs)
    cache: dict[Any, dict[str, Any]] = {}
    family_rows = [_family_calibration(family, inputs, cache=cache) for family in FAMILIES]
    calibrations = {row["family"]: row for row in family_rows}
    predictions = {
        family: {item["nucleus"]: item for item in row.get("frozen_predictions", [])}
        for family, row in calibrations.items()
    }
    designs = {
        family: _design_metadata(_make_design(family, calibrations[family].get("root", {}).get("scale", 0.25) if calibrations[family].get("root", {}).get("accepted") else 0.25))
        for family in FAMILIES
    }
    accepted_families = [family for family, row in calibrations.items() if row.get("calibration_status") == "CALIBRATED_CONDITIONAL_ROOT"]
    return _jsonable({
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "model_scope": {
            "finite_spherical_two_species_zero_temperature_TF": True,
            "species_degeneracy": 2,
            "families": list(FAMILIES),
            "excluded_families": {
                "Q4": "BULK_REJECTED; no finite fit or rescue attempted",
            },
            "rho_choice": RHO_CHOICE,
            "excluded_rho_choices": {
                "covariant_contact_J32": "separate EFT alternative; not fitted in this bridge",
            },
            "nuclei": {name: dict(values) for name, values in static.BRIDGE_NUCLEI.items()},
            "coulomb": "Hartree radial Maxwell with exterior Robin/tail energy",
            "shell_spin_orbit_pairing": False,
            "finite_range_rho": False,
            "full_finite_nucleus_RRPA": False,
            "full_experimental_spectrum": False,
            "new_interaction_or_global_refit": False,
            "static_subset_only": True,
        },
        "protocol": {
            "exploratory_scales": list(EXPLORATORY_SCALES),
            "bounded_scale_domain": [SCALE_MIN, SCALE_MAX],
            "seed_factors": list(SEED_FACTORS),
            "coarse_level": dict(COARSE_LEVEL),
            "fine_level": dict(FINE_LEVEL),
            "domain_level": dict(DOMAIN_LEVEL),
            "fresh24_level": dict(FRESH24_LEVEL),
            "recovery_levels": [dict(level) for level in RECOVERY_LEVELS],
            "diagnostic_rule": {
                "primary_intervals": static.DIAGNOSTIC_INTERVALS,
                "nested_intervals": static.DIAGNOSTIC_NESTED_INTERVALS,
                "gate_combination": "worst_residual_and_boolean_AND",
                "phase_search": False,
            },
            "fresh_control_factors": list(FRESH_CONTROL_FACTORS),
            "fresh_control_requirement": "at_least_one_valid_previous_none_attempt_matching_each_candidate_branch;_failed_exploratory_guesses_retained",
            "lineage_limits": {
                "max_scale_step": LINEAGE_MAX_SCALE_STEP,
                "max_abs_y_diff": LINEAGE_MAX_Y_DIFF,
                "max_abs_vector_potential_diff_MeV": LINEAGE_MAX_VECTOR_DIFF_MEV,
                "max_abs_coulomb_potential_diff_MeV": LINEAGE_MAX_COULOMB_DIFF_MEV,
                "max_abs_species_density_diff_fm_minus3": LINEAGE_MAX_DENSITY_DIFF_FM3,
                "max_abs_mu_diff_MeV": LINEAGE_MAX_MU_DIFF_MEV,
                "max_abs_binding_per_A_diff_MeV": LINEAGE_MAX_BINDING_DIFF_MEV,
                "max_abs_species_radius_diff_fm": LINEAGE_MAX_RADIUS_DIFF_FM,
            },
            "root_residual_limit_MeV_per_A": ROOT_RESIDUAL_LIMIT_MEV,
            "root_bracket_width_limit": ROOT_WIDTH_LIMIT,
            "energy_refinement_limit_MeV_per_A": ENERGY_LIMIT_MEV,
            "species_radius_refinement_limit_fm": RADIUS_LIMIT_FM,
            "derivative_steps_ds": list(DERIVATIVE_STEPS),
            "terminal_gates_reused_from_static_solver": {
                "N_Z_relative": static.N_RELATIVE_LIMIT,
                "field_relative": static.FIELD_RESIDUAL_LIMIT,
                "KKT_MeV": static.KKT_LIMIT_MEV,
                "Gauss_relative": static.GAUSS_LIMIT,
                "Coulomb_relative": static.COULOMB_LIMIT,
            },
        },
        "input_provenance": {
            "input_data": "verification/data/finite_static_observables_2026.json",
            "input_data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
            "static_solver": "verification/nvg_finite_monopole.py static subset only",
            "static_solver_sha256": hashlib.sha256(SOURCE_PATH.with_name("nvg_finite_monopole.py").read_bytes()).hexdigest(),
            "potential_producer": "verification/source_complete_scaling_saturation_audit.py:inverse_potential_jet",
            "finite_adapter": "verification/nvg_finite_droplet_audit.py:W8Design",
            "no_saved_model_result_used_as_input": True,
            "no_runtime_network": True,
            "experimental_data_used_for_root": "40Ca bare-binding target only after explicit electron convention",
            "heldout_binding_and_charge_radii_used_for_root": False,
        },
        "input_targets": targets,
        "designs_at_representative_scale": designs,
        "calibrations": calibrations,
        "families": family_rows,
        "predictions": predictions,
        "accepted_family_count": len(accepted_families),
        "accepted_families": accepted_families,
        "scientific_limits": [
            "The Ca40 binding match is one conditional fitted datum; it is not evidence for the fitted datum or a zero-fit claim.",
            "Common MN=939 MeV omits physical proton-neutron mass differences and nuclear mass-systematic terms.",
            "AME Eq.(2) electron binding is an explicitly approximate atomic convention; its unknown correction covariance is separate.",
            "Charge radii are descriptive IAEA/Angeli-Marinova inputs; one-body R1b omits delta_SO, delta_CM, delta_2body and relativistic corrections.",
            "Heldout 90Zr/208Pb binding and radii are not used for root/branch choice and are not silently pooled with newer muonic extractions.",
            "A localized stationary row is not a global-ground-state or full stability proof; no raw static-Hessian eigenvalue is promoted.",
            "The solver is finite spherical d=2 TF, not a finite-nucleus Dirac/RRPA spectrum or homogeneous complex response.",
        ],
    })


_CACHE: dict[str, Any] | None = None


def build_result() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = calculate()
    return _CACHE


def _finite_json(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _validate_terminal_public(case: Mapping[str, Any]) -> bool:
    """Recompute terminal acceptance from retained rows and fixed limits."""

    if not isinstance(case, Mapping):
        return False
    branches = case.get("branch_candidates")
    selected: Mapping[str, Any] | None = None
    if isinstance(branches, list):
        selected = next(
            (
                item for item in branches
                if isinstance(item, Mapping) and item.get("branch_id") == case.get("selected_branch_id")
            ),
            None,
        )
        if selected is None:
            return False
        # These duplicated top-level fields are part of the public witness;
        # an edited copy must not be allowed to differ from its selected branch.
        for key in (
            "fine_selected", "domain40_check", "fresh24_check", "fresh24_initial_attempt",
            "fresh24_attempts", "fresh24_profile_comparisons", "domain_profile_comparison",
        ):
            if key in case and case.get(key) != selected.get(key):
                return False
    else:
        controls = case.get("terminal_control_rows")
        if not isinstance(controls, Mapping):
            return False
        selected = {
            "fine_selected": controls.get("fine_selected", {}),
            "domain40_check": controls.get("domain40_check", {}),
            "fresh24_check": controls.get("fresh24_check", {}),
            "fresh24_initial_attempt": controls.get("fresh24_initial_attempt", {}),
            "fresh24_attempts": controls.get("fresh24_attempts", []),
            "domain_profile_comparison": controls.get("domain_profile_comparison", {}),
            "fresh24_profile_comparisons": controls.get("fresh24_profile_comparisons", []),
        }
    fine = selected.get("fine_selected", {})
    domain = selected.get("domain40_check", {})
    fresh = selected.get("fresh24_check", {})
    fresh_initial = selected.get("fresh24_initial_attempt", fresh)
    fresh_attempts = selected.get("fresh24_attempts", [])
    domain_profile = selected.get("domain_profile_comparison", {})
    fresh_profiles = selected.get("fresh24_profile_comparisons", [])
    # Every terminal witness retains the complete exploratory seed ledger.
    # Seed factors are guesses, so failed rows are allowed, but deleting an
    # attempt or changing its factor must not change the accepted branch.
    fine_factors = case.get("fine_seed_factors")
    fine_attempts = case.get("fine_seed_attempts")
    if not isinstance(fine_factors, list) or not fine_factors:
        return False
    if not isinstance(fine_attempts, list) or len(fine_attempts) != len(fine_factors):
        return False
    try:
        if any(
            not _finite(factor)
            or float(factor) not in tuple(float(item) for item in SEED_FACTORS)
            for factor in fine_factors
        ) or any(
            not isinstance(item, Mapping)
            or not _finite(item.get("seed_factor"))
            or abs(float(item["seed_factor"]) - float(factor)) > 1.0e-15
            for item, factor in zip(fine_attempts, fine_factors)
        ):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    if case.get("all_seed_outcomes_reported") is not True:
        return False
    fresh_factors = case.get("fresh24_seed_factors")
    if not isinstance(fresh_factors, list) or len(fresh_factors) != len(FRESH_CONTROL_FACTORS):
        return False
    try:
        if any(abs(float(left) - float(right)) > 1.0e-15 for left, right in zip(fresh_factors, FRESH_CONTROL_FACTORS)):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    if not isinstance(fresh_attempts, list) or len(fresh_attempts) != len(FRESH_CONTROL_FACTORS):
        return False
    if case.get("all_independent_fresh_outcomes_reported") is not True:
        return False
    if any(
        not isinstance(item, Mapping)
        or item.get("independent_previous") is not None
        or not _finite(item.get("seed_factor"))
        or abs(float(item["seed_factor"]) - float(factor)) > 1.0e-15
        for item, factor in zip(fresh_attempts, FRESH_CONTROL_FACTORS)
    ):
        return False
    if "fresh24_seed_factors" in case:
        factors = case.get("fresh24_seed_factors")
        if not isinstance(factors, list) or len(factors) != len(FRESH_CONTROL_FACTORS):
            return False
        try:
            if any(abs(float(left) - float(right)) > 1.0e-15 for left, right in zip(factors, FRESH_CONTROL_FACTORS)):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
    gate = _branch_numeric_gate(
        fine, domain, fresh, fresh_initial, fresh_attempts, domain_profile, fresh_profiles
    )
    summary = case.get("refinement_summary")
    if not isinstance(summary, Mapping):
        return False
    expected_summary = {
        "fine_converged": gate["fine_valid"],
        "fine_stationarity": _row_numerical_gate(fine, require_binding=False),
        "fine_localized": bool(isinstance(fine, Mapping) and fine.get("localized_vacuum_exterior") is True),
        "domain_converged": gate["domain_valid"],
        "domain_stationarity": _row_numerical_gate(domain, require_binding=False),
        "domain_localized": bool(isinstance(domain, Mapping) and domain.get("localized_vacuum_exterior") is True),
        "fresh24_converged": gate["fresh_valid"],
        "fresh24_stationarity": _row_numerical_gate(fresh, require_binding=False),
        "fresh24_localized": bool(isinstance(fresh, Mapping) and fresh.get("localized_vacuum_exterior") is True),
        "fresh24_independent_initial_pass": gate["fresh_initial_valid"],
        "fresh24_independent_rows_valid": gate["fresh_independent_rows_valid"],
        "fresh24_valid_attempt_count": sum(1 for item in fresh_attempts if _fresh_attempt_valid(item)) if isinstance(fresh_attempts, list) else 0,
        "fresh24_failed_attempt_count": (
            len(fresh_attempts) - sum(1 for item in fresh_attempts if _fresh_attempt_valid(item))
            if isinstance(fresh_attempts, list) else 0
        ),
        "fresh24_branch_profiles_pass": gate["fresh_profile_pass"],
        "domain_energy_pass": bool(gate["domain_energy_change"] is not None and gate["domain_energy_change"] <= ENERGY_LIMIT_MEV),
        "fresh24_energy_pass": bool(gate["fresh_energy_change"] is not None and gate["fresh_energy_change"] <= ENERGY_LIMIT_MEV),
        "domain_species_radius_pass": bool(all(value is not None and value <= RADIUS_LIMIT_FM for value in gate["domain_radius_changes"].values())),
        "fresh24_species_radius_pass": bool(all(value is not None and value <= RADIUS_LIMIT_FM for value in gate["fresh_radius_changes"].values())),
        "all_terminal_rows_converged_stationary_localized_bound": bool(
            gate["fine_valid"] and gate["domain_valid"] and gate["fresh_valid"]
        ),
    }
    for key, expected in expected_summary.items():
        if summary.get(key) is not expected:
            return False
    lineage_valid = _lineage_public_valid(case.get("lineage", {}))
    unmatched_fresh_factors = _fresh_branch_unmatched_factors(branches, fresh_attempts)
    if "fresh_branch_ambiguity" in case:
        ambiguity = case.get("fresh_branch_ambiguity")
        expected_ambiguity = {
            "status": "UNRESOLVED_FRESH_BRANCH_AMBIGUITY" if unmatched_fresh_factors else "NO_UNMATCHED_FRESH_BRANCH",
            "unmatched_valid_seed_factors": unmatched_fresh_factors,
            "valid_attempt_count": sum(1 for item in fresh_attempts if _fresh_attempt_valid(item)) if isinstance(fresh_attempts, list) else 0,
            "all_valid_attempts_assigned_to_retained_fine_branch": not bool(unmatched_fresh_factors),
        }
        if ambiguity != expected_ambiguity:
            return False
    expected_acceptance = bool(gate["accepted"] and lineage_valid and not unmatched_fresh_factors)
    if case.get("terminal_protocol_acceptance") is not expected_acceptance:
        return False
    expected_localized = bool(gate["domain_valid"] and domain.get("localized_vacuum_exterior") is True)
    if expected_acceptance and not isinstance(case.get("lineage"), Mapping):
        return False
    localized_key = "terminal_localized" if "terminal_localized" in case else "localized_vacuum_exterior"
    if case.get(localized_key) is not expected_localized:
        return False
    if expected_acceptance:
        try:
            expected_binding = -float(domain["energy_per_A_minus_M_MeV"])
            reported_binding = float(case.get("binding_per_A_MeV", case.get("model_binding_per_A_MeV")))
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(reported_binding) or abs(reported_binding - expected_binding) > 1.0e-10:
            return False

    if isinstance(branches, list):
        branch_states: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
        for branch in branches:
            if not isinstance(branch, Mapping):
                return False
            branch_factors = branch.get("fine_seed_factors")
            if not isinstance(branch_factors, list) or not branch_factors:
                return False
            if any(
                not _finite(item) or float(item) not in tuple(float(seed) for seed in SEED_FACTORS)
                for item in branch_factors
            ):
                return False
            branch_gate = _branch_numeric_gate(
                branch.get("fine_selected", {}),
                branch.get("domain40_check", {}),
                branch.get("fresh24_check", {}),
                branch.get("fresh24_initial_attempt", branch.get("fresh24_check", {})),
                branch.get("fresh24_attempts", []),
                branch.get("domain_profile_comparison", {}),
                branch.get("fresh24_profile_comparisons", []),
            )
            branch_states.append((branch, branch_gate))
            if branch.get("terminal_protocol_acceptance") is not branch_gate["accepted"]:
                return False
            if branch.get("terminal_localized") is not bool(branch_gate["domain_valid"] and branch.get("domain40_check", {}).get("localized_vacuum_exterior") is True):
                return False
            expected_branch_flags = {
                "fine_converged": branch_gate["fine_valid"],
                "domain_converged": branch_gate["domain_valid"],
                "fresh24_independent_initial_pass": branch_gate["fresh_initial_valid"],
                "fresh24_effective_pass": branch_gate["fresh_valid"],
                "fresh24_independent_rows_valid": branch_gate["fresh_independent_rows_valid"],
                "fresh24_valid_attempt_count": sum(1 for item in branch.get("fresh24_attempts", []) if _fresh_attempt_valid(item)) if isinstance(branch.get("fresh24_attempts", []), list) else 0,
                "fresh24_failed_attempt_count": (
                    len(branch.get("fresh24_attempts", [])) - sum(1 for item in branch.get("fresh24_attempts", []) if _fresh_attempt_valid(item))
                    if isinstance(branch.get("fresh24_attempts", []), list) else 0
                ),
                "fresh24_branch_profiles_pass": branch_gate["fresh_profile_pass"],
                "domain_energy_pass": bool(branch_gate["domain_energy_change"] is not None and branch_gate["domain_energy_change"] <= ENERGY_LIMIT_MEV),
                "fresh24_energy_pass": bool(branch_gate["fresh_energy_change"] is not None and branch_gate["fresh_energy_change"] <= ENERGY_LIMIT_MEV),
                "domain_species_radius_pass": bool(all(value is not None and value <= RADIUS_LIMIT_FM for value in branch_gate["domain_radius_changes"].values())),
                "fresh24_species_radius_pass": bool(all(value is not None and value <= RADIUS_LIMIT_FM for value in branch_gate["fresh_radius_changes"].values())),
            }
            for key, expected in expected_branch_flags.items():
                if branch.get(key) is not expected:
                    return False
            for key, expected in (
                ("domain_energy_change_MeV", branch_gate["domain_energy_change"]),
                ("fresh24_energy_change_MeV", branch_gate["fresh_energy_change"]),
            ):
                actual = branch.get(key)
                if expected is None:
                    if actual is not None:
                        return False
                else:
                    try:
                        if not math.isfinite(float(actual)) or abs(float(actual) - expected) > 1.0e-12:
                            return False
                    except (TypeError, ValueError, OverflowError):
                        return False
            for key, expected_map in (
                ("domain_species_radius_changes_fm", branch_gate["domain_radius_changes"]),
                ("fresh24_species_radius_changes_fm", branch_gate["fresh_radius_changes"]),
            ):
                actual_map = branch.get(key)
                if not isinstance(actual_map, Mapping):
                    return False
                for name, expected in expected_map.items():
                    actual = actual_map.get(name)
                    if expected is None:
                        if actual is not None:
                            return False
                    else:
                        try:
                            if not math.isfinite(float(actual)) or abs(float(actual) - expected) > 1.0e-12:
                                return False
                        except (TypeError, ValueError, OverflowError):
                            return False
        controlled = [branch for branch, branch_gate in branch_states if branch_gate["accepted"]]
        selected_branch_gate = next((gate_item for branch, gate_item in branch_states if branch is selected), None)
        if selected_branch_gate is None:
            return False
        if selected.get("terminal_protocol_acceptance") is True:
            if selected_branch_gate["accepted"] is not True or any(
                float(selected["domain40_check"]["energy_per_A_minus_M_MeV"])
                > float(branch["domain40_check"]["energy_per_A_minus_M_MeV"]) + 1.0e-12
                for branch in controlled
            ):
                return False
    return True


def _validate_root_public(root: Mapping[str, Any], *, expected_target: float | None = None) -> bool:
    """Recompute root endpoints, insertions, width, and final replacement."""

    if not isinstance(root, Mapping) or root.get("accepted") is not True:
        return False
    try:
        scale = float(root["scale"])
        target = float(root["target_B_nuc_per_A_MeV"])
        binding = float(root["binding_per_A_MeV"])
        residual = float(root["binding_residual_MeV"])
        bracket = root["bracket_final"]
        lo, hi = float(bracket[0]), float(bracket[1])
        width = float(root["bracket_width"])
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False
    if not all(math.isfinite(value) for value in (scale, target, binding, residual, lo, hi, width)):
        return False
    if expected_target is not None and not _same_number(target, expected_target):
        return False
    if not (SCALE_MIN <= lo < hi <= SCALE_MAX and lo <= scale <= hi):
        return False
    if abs(width - (hi - lo)) > 1.0e-12 or width > ROOT_WIDTH_LIMIT:
        return False
    if abs(residual - (binding - target)) > 1.0e-10 or abs(residual) > ROOT_RESIDUAL_LIMIT_MEV:
        return False
    if root.get("scale_label") != _scale_label(scale):
        return False
    coarse = root.get("coarse_bracket")
    if not isinstance(coarse, Mapping):
        return False
    try:
        coarse_lo = float(coarse["scale_lo"])
        coarse_hi = float(coarse["scale_hi"])
        coarse_width = float(coarse["width"])
        coarse_target = float(coarse["target_B_nuc_per_A_MeV"])
        coarse_binding_lo = float(coarse["binding_lo_MeV"])
        coarse_binding_hi = float(coarse["binding_hi_MeV"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(value) for value in (coarse_lo, coarse_hi, coarse_width, coarse_target, coarse_binding_lo, coarse_binding_hi)):
        return False
    if not (SCALE_MIN <= coarse_lo < coarse_hi <= SCALE_MAX and abs(coarse_width - (coarse_hi - coarse_lo)) <= 1.0e-12):
        return False
    if abs(coarse_target - target) > 1.0e-12:
        return False
    if (coarse_binding_lo - target) * (coarse_binding_hi - target) > 0.0:
        return False
    evaluations = root.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        return False
    evaluation_by_scale: dict[float, list[Mapping[str, Any]]] = {}
    for item in evaluations:
        if not isinstance(item, Mapping) or not _validate_terminal_public(item):
            return False
        try:
            value = float(item["scale"])
            item_binding = float(item["binding_per_A_MeV"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(value) or not math.isfinite(item_binding) or not (coarse_lo <= value <= coarse_hi):
            return False
        evaluation_by_scale.setdefault(value, []).append(item)
    final_state = root.get("final_calibration_state")
    if not isinstance(final_state, Mapping) or not _validate_terminal_public(final_state):
        return False
    try:
        final_scale = float(final_state["scale"])
        final_binding = float(final_state["binding_per_A_MeV"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if abs(final_scale - scale) > 1.0e-12 or abs(final_binding - binding) > 1.0e-10:
        return False
    updates = root.get("lineage_update_checks", [])
    if not isinstance(updates, list):
        return False
    try:
        if int(root.get("iterations")) != len(updates):
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    for update in updates:
        if not isinstance(update, Mapping):
            return False
        try:
            update_scale = float(update["scale"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(update_scale) or update_scale not in evaluation_by_scale:
            return False
        matching = evaluation_by_scale[update_scale]
        expected_valid = any(_validate_terminal_public(item) for item in matching)
        expected_status = next(
            (item.get("lineage", {}).get("status") for item in matching if _validate_terminal_public(item)),
            None,
        )
        if update.get("terminal_valid") is not expected_valid or update.get("lineage_status") != expected_status:
            return False
    if root.get("lineage_consistency_verified") is not True:
        return False
    if root.get("terminal_protocol_acceptance") is not final_state.get("terminal_protocol_acceptance"):
        return False
    if root.get("localized") is not final_state.get("terminal_localized"):
        return False
    return True


def _validate_derivative_public(derivative: Any, root: Mapping[str, Any]) -> bool:
    """Recompute the centered envelope check from retained side metrics."""

    if not isinstance(derivative, Mapping) or not isinstance(root, Mapping):
        return False
    steps = derivative.get("steps")
    if not isinstance(steps, list) or len(steps) != len(DERIVATIVE_STEPS):
        return False
    try:
        root_scale = float(root["scale"])
        e_grad = float(derivative["E_grad_MeV"])
        domain = root["final_calibration_state"]["domain40_check"]
        expected_grad = float(domain["energy_components_MeV"]["T_W"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(value) for value in (root_scale, e_grad, expected_grad)):
        return False
    if abs(e_grad - expected_grad) > 1.0e-10:
        return False
    repeat = derivative.get("repeat_root_domain_control")
    if not isinstance(repeat, Mapping):
        return False
    try:
        if abs(float(repeat["root_scale_used"]) - root_scale) > 1.0e-12:
            return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    expected_domain = root.get("terminal_protocol_acceptance") is True
    expected_fresh = _terminal_row_valid(root["final_calibration_state"].get("fresh24_initial_attempt", {}), require_binding=False)
    if repeat.get("domain40_terminal_acceptance") is not expected_domain:
        return False
    if repeat.get("fresh24_terminal_acceptance") is not expected_fresh:
        return False
    if repeat.get("domain_and_fresh24_controls_are_separate") is not True:
        return False

    computed_count = 0
    expected_errors: list[float] = []
    for expected_step, item in zip(DERIVATIVE_STEPS, steps):
        if not isinstance(item, Mapping):
            return False
        try:
            step = float(item["step_ds"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(step) or abs(step - expected_step) > 1.0e-15:
            return False
        status = item.get("status")
        if status == "COMPUTED":
            try:
                ep = float(item["E_plus_per_A_minus_M_MeV"])
                em = float(item["E_minus_per_A_minus_M_MeV"])
                d_e = float(item["dE_ds_centered_MeV"])
                d_b = float(item["dB_ds_centered_MeV"])
                predicted_e = float(item["predicted_dE_ds_2Egrad_over_s_MeV"])
                predicted_b = float(item["predicted_dB_ds_minus_2Egrad_over_s_MeV"])
                rel = float(item["relative_error_dE_ds"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return False
            if not all(math.isfinite(value) for value in (ep, em, d_e, d_b, predicted_e, predicted_b, rel)):
                return False
            expected_e = int(static.BRIDGE_NUCLEI["Ca40"]["A"]) * (ep - em) / (2.0 * expected_step)
            expected_predicted = 2.0 * e_grad / root_scale
            expected_rel = abs(expected_e - expected_predicted) / max(abs(expected_e), abs(expected_predicted), 1.0e-12)
            if abs(d_e - expected_e) > 1.0e-8 or abs(d_b + expected_e) > 1.0e-8:
                return False
            if abs(predicted_e - expected_predicted) > 1.0e-8 or abs(predicted_b + expected_predicted) > 1.0e-8:
                return False
            if abs(rel - expected_rel) > 1.0e-10:
                return False
            if item.get("plus_terminal_acceptance") is not True or item.get("minus_terminal_acceptance") is not True:
                return False
            expected_errors.append(expected_rel)
            computed_count += 1
        elif status == "TERMINAL_SIDE_UNRESOLVED":
            # Preserve unresolved side evidence, but do not let an edited
            # status/flag turn a failed side into a derivative result.
            plus, minus = item.get("plus"), item.get("minus")
            if not isinstance(plus, Mapping) or not isinstance(minus, Mapping):
                return False
            plus_valid = _validate_terminal_public(plus)
            minus_valid = _validate_terminal_public(minus)
            if item.get("plus_terminal_acceptance") is not plus_valid or item.get("minus_terminal_acceptance") is not minus_valid:
                return False
        elif status == "OUTSIDE_BOUNDED_DOMAIN":
            if expected_step + root_scale <= SCALE_MIN or root_scale + expected_step >= SCALE_MAX:
                return False
        else:
            return False
    all_computed = computed_count == len(DERIVATIVE_STEPS)
    expected_status = "COMPUTED_ENVELOPE_DERIVATIVE_CHECK" if all_computed else "UNRESOLVED_ENVELOPE_DERIVATIVE_CHECK"
    if derivative.get("status") != expected_status or derivative.get("two_centered_steps_computed") is not all_computed:
        return False
    max_error = None if not expected_errors else max(expected_errors)
    reported_max = derivative.get("max_relative_error")
    if max_error is None:
        if reported_max is not None:
            return False
    else:
        try:
            if not math.isfinite(float(reported_max)) or abs(float(reported_max) - max_error) > 1.0e-10:
                return False
        except (TypeError, ValueError, OverflowError):
            return False
    return True


def _validate_prediction(prediction: Mapping[str, Any]) -> bool:
    if not isinstance(prediction, Mapping):
        return False
    nucleus = prediction.get("nucleus")
    if nucleus not in BRIDGE_NUCLEI:
        return False
    try:
        inputs = _load_inputs()
        expected_target = _input_targets(inputs)[str(nucleus)]
    except (ArithmeticError, FiniteStaticBridgeError, KeyError, TypeError, ValueError, OverflowError):
        return False
    comparison = prediction.get("binding_comparison")
    if not isinstance(comparison, Mapping):
        return False
    # The displayed experimental reference and electron convention are
    # recomputed from the cited input bundle; labels and stored targets are
    # not accepted as a substitute for this linkage.
    for key in (
        "B_atom_input_per_A_MeV",
        "experimental_uncertainty_MeV_per_A",
        "electronic_convention_correction_total_MeV",
        "B_nuc_target_per_A_MeV",
    ):
        expected_key = {
            "B_atom_input_per_A_MeV": "B_atom_per_A_MeV",
            "experimental_uncertainty_MeV_per_A": "B_atom_per_A_sigma_MeV",
            "electronic_convention_correction_total_MeV": "electron_correction_total_MeV",
            "B_nuc_target_per_A_MeV": "B_nuc_target_per_A_MeV",
        }[key]
        if not _same_number(comparison.get(key), expected_target[expected_key]):
            return False
    expected_role = (
        "sole_calibration_anchor"
        if nucleus == "Ca40"
        else "heldout_descriptive_comparison_not_used_for_root"
    )
    if comparison.get("input_role") != expected_role:
        return False

    controls = prediction.get("terminal_control_rows")
    if not isinstance(controls, Mapping):
        return False
    domain = controls.get("domain40_check")
    if not isinstance(domain, Mapping):
        return False
    try:
        terminal_model_binding = -float(domain["energy_per_A_minus_M_MeV"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(terminal_model_binding):
        return False
    accepted = prediction.get("terminal_protocol_acceptance") is True
    if accepted:
        if not _same_number(prediction.get("model_binding_per_A_MeV"), terminal_model_binding, tolerance=1.0e-10):
            return False
        expected_residual = terminal_model_binding - expected_target["B_nuc_target_per_A_MeV"]
        if not _same_number(comparison.get("model_minus_B_nuc_target_per_A_MeV"), expected_residual, tolerance=1.0e-10):
            return False
        if prediction.get("unaccepted_domain_binding_per_A_MeV") is not None:
            return False
    else:
        if prediction.get("model_binding_per_A_MeV") is not None:
            return False
        if comparison.get("model_minus_B_nuc_target_per_A_MeV") is not None:
            return False
        if not _same_number(
            prediction.get("unaccepted_domain_binding_per_A_MeV"), terminal_model_binding,
            tolerance=1.0e-10,
        ):
            return False
    if prediction.get("terminal_protocol_acceptance") is True:
        if not isinstance(prediction.get("radii"), Mapping):
            return False
        if prediction["radii"].get("uncomputed_corrections", {}).get("delta_SO_fm2", "missing") is not None:
            return False
        try:
            expected_radii = _radius_mapping(domain, str(nucleus), inputs)
        except (ArithmeticError, FiniteStaticBridgeError, KeyError, TypeError, ValueError, OverflowError):
            return False
        if prediction.get("radii") != expected_radii:
            return False
    try:
        domain_localized = domain.get("localized_vacuum_exterior") is True
    except (KeyError, TypeError, AttributeError):
        return False
    if prediction.get("localized_vacuum_exterior") is not domain_localized:
        return False
    if not _validate_terminal_public(prediction):
        return False
    return True


def _validate_structure(result: Mapping[str, Any]) -> bool:
    if not isinstance(result, Mapping):
        return False
    if result.get("schema_version") != SCHEMA or result.get("status") != STATUS or result.get("evidence_weight") != 0.0:
        return False
    if not _finite_json(result):
        return False
    try:
        expected_inputs = _load_inputs()
        expected_targets = _input_targets(expected_inputs)
    except (ArithmeticError, FiniteStaticBridgeError, KeyError, TypeError, ValueError, OverflowError):
        return False
    if not _all_input_targets_match(result.get("input_targets"), expected_targets):
        return False
    calibrations = result.get("calibrations")
    if not isinstance(calibrations, Mapping) or set(calibrations) != set(FAMILIES):
        return False
    family_rows = result.get("families")
    if not isinstance(family_rows, list) or {item.get("family") for item in family_rows if isinstance(item, Mapping)} != set(FAMILIES):
        return False
    family_by_name = {item.get("family"): item for item in family_rows if isinstance(item, Mapping)}
    top_predictions = result.get("predictions")
    if not isinstance(top_predictions, Mapping) or set(top_predictions) != set(FAMILIES):
        return False
    expected_accepted_families = []
    for family in FAMILIES:
        row = calibrations[family]
        if not isinstance(row, Mapping):
            return False
        if family_by_name.get(family) != row:
            return False
        if row.get("rho_choice") != RHO_CHOICE:
            return False
        if row.get("exploratory_scales") != list(EXPLORATORY_SCALES):
            return False
        if not _target_mapping_matches(row.get("target"), expected_targets["Ca40"]):
            return False
        attempts = row.get("exploratory_attempts")
        if not isinstance(attempts, list) or len(attempts) < len(EXPLORATORY_SCALES):
            return False
        fixed = [item for item in attempts if isinstance(item, Mapping) and item.get("probe_role") == "required_fixed_exploratory_scale"]
        if len(fixed) != len(EXPLORATORY_SCALES):
            return False
        if any(
            not isinstance(item.get("seed_attempts"), list)
            or len(item["seed_attempts"]) != len(SEED_FACTORS)
            or item.get("all_seed_outcomes_reported") is not True
            for item in fixed
        ):
            return False
        root = row.get("root")
        if not isinstance(root, Mapping):
            return False
        if row.get("calibration_status") == "CALIBRATED_CONDITIONAL_ROOT":
            expected_accepted_families.append(family)
            if not _validate_root_public(root, expected_target=expected_targets["Ca40"]["B_nuc_target_per_A_MeV"]):
                return False
            if root.get("lineage_consistency_verified") is not True:
                return False
            if not _validate_derivative_public(row.get("envelope_derivative"), root):
                return False
            predictions = row.get("frozen_predictions")
            if not isinstance(predictions, list) or {item.get("nucleus") for item in predictions if isinstance(item, Mapping)} != set(BRIDGE_NUCLEI):
                return False
            if not all(_validate_prediction(item) for item in predictions):
                return False
            family_prediction_map = top_predictions.get(family)
            if not isinstance(family_prediction_map, Mapping) or set(family_prediction_map) != set(BRIDGE_NUCLEI):
                return False
            if any(
                family_prediction_map.get(item.get("nucleus")) != item
                for item in predictions
                if isinstance(item, Mapping)
            ):
                return False
        elif root.get("accepted") is True:
            return False
        else:
            if row.get("frozen_predictions") != []:
                return False
            if top_predictions.get(family) not in ({}, None):
                return False
    if result.get("accepted_families") != expected_accepted_families:
        return False
    if result.get("accepted_family_count") != len(expected_accepted_families):
        return False
    return True


def validate_result(result: Mapping[str, Any]) -> bool:
    """Fail closed on altered protocol, source role, or live rows."""

    try:
        return _validate_structure(result) and result == build_result()
    except (ArithmeticError, FiniteStaticBridgeError, KeyError, TypeError, ValueError):
        return False


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - parser path.
        raise FiniteStaticBridgeError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="explicitly write the JSON result to this path")
    args = parser.parse_args(argv)
    try:
        result = build_result()
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return 0
    except (ArithmeticError, FiniteStaticBridgeError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({
            "schema_version": SCHEMA,
            "status": "INVALID_OR_FAILED_FINITE_STATIC_BRIDGE",
            "evidence_weight": EVIDENCE_WEIGHT,
            "error": str(exc),
        }, ensure_ascii=False, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
