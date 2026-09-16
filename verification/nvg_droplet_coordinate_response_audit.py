#!/usr/bin/env python3
"""Coordinate-equilibrated finite-droplet replay and angular-response audit.

The calculation reuses the accepted finite-droplet stability producer.  It
replays the complete finite basis protocol after a declared diagonal Gram
equilibration, then checks coordinate-invariant quadratic forms, a finite
density Schur complement, and the angular resolvent identity on fixed smooth
profiles.  It is zero empirical weight and does not promote a finite replay
to an all-mode stability theorem.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

sys.dont_write_bytecode = True

try:
    import nvg_droplet_stability_audit as stability
except ImportError:  # pragma: no cover - package-style import support.
    from . import nvg_droplet_stability_audit as stability


HERE = Path(__file__).resolve().parent
SCHEMA = "nvg_droplet_coordinate_response_audit.v1"
STATUS = "COMPUTED_FINITE_W8_COORDINATE_RESPONSE_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0
TARGETS = tuple(stability.TARGETS)
TARGET_N = stability.TARGET_N
ELL_VALUES = tuple(stability.ELL_VALUES)
BASIS_SIZES = tuple(stability.BASIS_SIZES)
FINE_INTERVALS = stability.FINE_INTERVALS
GRID_INTERVALS = tuple(stability.GRID_INTERVALS)
DOMAIN_BOXES_FM = tuple(stability.DOMAIN_BOXES_FM)
TERMINAL_SIZE = BASIS_SIZES[-1]
TERMINAL_INTERVALS = FINE_INTERVALS
TERMINAL_DOMAIN_FM = DOMAIN_BOXES_FM[-1]
COORDINATE_REPLAY_EXPONENTS = (-6, 0, 6)
ANGULAR_PAIRS = (2, 4, 6, 10)
ANGULAR_PROFILES = (
    "density_only",
    "scalar_only",
    "coupled_density_scalar",
)
ANGULAR_CONTINUUM_LIMIT = stability.DISCRETIZATION_RELATIVE_DRIFT_LIMIT
ANGULAR_DISCRETE_LIMIT = 1.0e-10
SPECTRUM_COMPARISON_LIMIT = stability.DISCRETIZATION_RELATIVE_DRIFT_LIMIT


class CoordinateResponseError(ValueError):
    """Fail-closed error for malformed coordinate-response evidence."""


def _finite_scalar(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and not isinstance(value, bool)
    except (TypeError, ValueError, OverflowError):
        return False


def _jsonable(value: Any) -> Any:
    return stability._jsonable(value)


def _sym(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    return 0.5 * (array + array.T)


def _gram_rank(matrix: np.ndarray) -> tuple[int, int]:
    """Return rank under the existing stability Gram cutoff."""

    array = _sym(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise CoordinateResponseError("Gram matrix must be square")
    values = np.linalg.eigvalsh(array)
    positive = np.isfinite(values) & (values > 0.0)
    if not np.any(positive):
        return 0, int(array.shape[0])
    largest = float(np.max(values[positive]))
    keep = positive & (values >= stability.GRAM_RELATIVE_CUTOFF * largest)
    return int(np.count_nonzero(keep)), int(array.shape[0])


def _apply_diagonal_scale(
    hessian: np.ndarray,
    gram: np.ndarray,
    excluded_vector: np.ndarray | None,
    scale: np.ndarray,
    label: str,
    exponent: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Mapping[str, Any]]:
    """Apply c=S z to both members of a generalized eigenproblem."""

    hessian = _sym(hessian)
    gram = _sym(gram)
    if hessian.shape != gram.shape or hessian.ndim != 2:
        raise CoordinateResponseError("H and G must be matching square matrices")
    scale = np.asarray(scale, dtype=float).reshape(-1)
    if scale.size != gram.shape[0] or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise CoordinateResponseError("coordinate scale must be finite and positive")
    raw_rank, raw_size = _gram_rank(gram)
    hbar = _sym(scale[:, None] * hessian * scale[None, :])
    gbar = _sym(scale[:, None] * gram * scale[None, :])
    normalized_rank, normalized_size = _gram_rank(gbar)
    if excluded_vector is None:
        tbar = None
    else:
        vector = np.asarray(excluded_vector, dtype=float).reshape(-1)
        if vector.size != scale.size:
            raise CoordinateResponseError("translation vector has wrong dimension")
        # The physical vector is c=t, hence z=S^{-1}t.
        tbar = vector / scale
    metadata = {
        "transform": label,
        "scale_exponent": exponent,
        "scale_min": float(np.min(scale)),
        "scale_max": float(np.max(scale)),
        "scale_condition_number": float(np.max(scale) / np.min(scale)),
        "input_raw_gram_rank": raw_rank,
        "input_raw_gram_size": raw_size,
        "normalized_gram_rank": normalized_rank,
        "normalized_gram_size": normalized_size,
        "translation_transformed_as_inverse_scale": excluded_vector is not None,
    }
    return hbar, gbar, tbar, metadata


def _unit_gram_transform(
    hessian: np.ndarray,
    gram: np.ndarray,
    excluded_vector: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Mapping[str, Any]]:
    """Normalize every trial column to unit diagonal Gram scale."""

    diagonal = np.diag(_sym(gram))
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise CoordinateResponseError("unit Gram equilibration requires positive column norms")
    scale = 1.0 / np.sqrt(diagonal)
    return _apply_diagonal_scale(
        hessian,
        gram,
        excluded_vector,
        scale,
        "unit_diagonal_gram",
        None,
    )


def _power_scale_transform(exponent: int) -> Callable[..., Any]:
    """Construct the declared S=diag(10^(k*j/(m-1))) replay."""

    def transform(
        hessian: np.ndarray,
        gram: np.ndarray,
        excluded_vector: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Mapping[str, Any]]:
        size = _sym(gram).shape[0]
        if size < 2:
            raise CoordinateResponseError("coordinate replay needs at least two columns")
        scale = np.power(10.0, np.linspace(0.0, float(exponent), size))
        return _apply_diagonal_scale(
            hessian,
            gram,
            excluded_vector,
            scale,
            "declared_power_replay",
            int(exponent),
        )

    return transform


def _power_then_unit_transform(exponent: int) -> Callable[..., Any]:
    """Apply the declared power replay, then equilibrate its Gram diagonal.

    The order is intentional.  For a positive diagonal ``S`` the second
    normalization has scale ``D_S = D S^-1`` in exact arithmetic, so the
    composed physical coordinate map ``S D_S`` is the original unit map.  The
    intermediate power-congruent ranks are retained in the metadata; the
    returned pair is the representation whose spectrum is actually tested.
    """

    def transform(
        hessian: np.ndarray,
        gram: np.ndarray,
        excluded_vector: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Mapping[str, Any]]:
        size = _sym(gram).shape[0]
        if size < 2:
            raise CoordinateResponseError("coordinate replay needs at least two columns")
        scale = np.power(10.0, np.linspace(0.0, float(exponent), size))
        h_power, g_power, t_power, power_metadata = _apply_diagonal_scale(
            hessian,
            gram,
            excluded_vector,
            scale,
            "declared_power_replay",
            int(exponent),
        )
        hbar, gbar, tbar, unit_metadata = _unit_gram_transform(
            h_power,
            g_power,
            t_power,
        )
        raw_rank, raw_size = _gram_rank(gram)
        metadata = {
            "transform": "declared_power_replay_then_unit_diagonal_gram",
            "scale_exponent": int(exponent),
            "scale_min": power_metadata["scale_min"],
            "scale_max": power_metadata["scale_max"],
            "scale_condition_number": power_metadata["scale_condition_number"],
            "input_raw_gram_rank": raw_rank,
            "input_raw_gram_size": raw_size,
            "power_scaled_gram_rank": power_metadata["normalized_gram_rank"],
            "power_scaled_gram_size": power_metadata["normalized_gram_size"],
            "unit_input_raw_gram_rank": unit_metadata["input_raw_gram_rank"],
            "unit_input_raw_gram_size": unit_metadata["input_raw_gram_size"],
            "normalized_gram_rank": unit_metadata["normalized_gram_rank"],
            "normalized_gram_size": unit_metadata["normalized_gram_size"],
            "translation_transformed_as_inverse_scale": excluded_vector is not None,
            "translation_transformed_as_inverse_total_scale": excluded_vector is not None,
        }
        return hbar, gbar, tbar, metadata

    return transform


def _grid_background(background: stability.Background, intervals: int) -> stability.Background:
    if intervals == background.x.size - 1:
        return background
    return stability.Background.from_solution(
        background.design,
        background.box_fm,
        background.solution,
        background.solver_row,
        intervals,
    )


def _relative_vector_error(left: Any, right: Any) -> dict[str, Any]:
    a = np.asarray(left, dtype=float).reshape(-1)
    b = np.asarray(right, dtype=float).reshape(-1)
    if a.size != b.size:
        count = min(a.size, b.size)
        if count:
            maximum = float(np.max(np.abs(a[:count] - b[:count])))
        else:
            maximum = None
        return {
            "length_equal": False,
            "left_length": int(a.size),
            "right_length": int(b.size),
            "max_absolute_difference": maximum,
            "max_relative_difference": None,
            "passes": False,
        }
    if not a.size:
        return {
            "length_equal": True,
            "left_length": 0,
            "right_length": 0,
            "max_absolute_difference": 0.0,
            "max_relative_difference": 0.0,
            "passes": True,
        }
    differences = np.abs(a - b)
    scales = np.maximum(np.maximum(np.abs(a), np.abs(b)), stability.SIGN_RESOLUTION_FLOOR)
    maximum = float(np.max(differences))
    relative = float(np.max(differences / scales))
    return {
        "length_equal": True,
        "left_length": int(a.size),
        "right_length": int(b.size),
        "max_absolute_difference": maximum,
        "max_relative_difference": relative,
        "passes": bool(relative <= SPECTRUM_COMPARISON_LIMIT),
    }


def _record_pair_comparison(
    original: Mapping[str, Any],
    equilibrated: Mapping[str, Any],
    ell: int,
) -> dict[str, Any]:
    key = (
        "smallest_internal_eigenvalues_Q_over_W0"
        if ell == 1
        else "smallest_eigenvalues_Q_over_W0"
    )
    internal_key = "smallest_internal_eigenvalues_Q_over_W0"
    raw_pair = _relative_vector_error(
        original.get("eigenvalues_Q_over_W0", []),
        equilibrated.get("eigenvalues_Q_over_W0", []),
    )
    physical_pair = _relative_vector_error(original.get(key, []), equilibrated.get(key, []))
    internal_pair = _relative_vector_error(
        original.get(internal_key, []), equilibrated.get(internal_key, []),
    )
    ranks_equal = bool(
        original.get("gram_rank") == equilibrated.get("gram_rank")
        and original.get("internal_gram_rank") == equilibrated.get("internal_gram_rank")
    )
    return {
        "ell": int(ell),
        "spectrum_key": key,
        "raw_rank_original": original.get("gram_rank"),
        "raw_rank_equilibrated": equilibrated.get("gram_rank"),
        "internal_rank_original": original.get("internal_gram_rank"),
        "internal_rank_equilibrated": equilibrated.get("internal_gram_rank"),
        "rank_invariant": ranks_equal,
        "raw_eigenvalue_comparison": raw_pair,
        "physical_eigenvalue_comparison": physical_pair,
        "internal_eigenvalue_comparison": internal_pair,
        "passes": bool(
            ranks_equal
            and raw_pair["passes"]
            and physical_pair["passes"]
            and internal_pair["passes"]
        ),
    }


def _protocol_comparison(
    original: Mapping[str, Any],
    equilibrated: Mapping[str, Any],
) -> dict[str, Any]:
    sector_rows: list[dict[str, Any]] = []
    all_pass = True
    for ell in ELL_VALUES:
        original_sector = original.get("sectors", {}).get(str(ell), {})
        equilibrated_sector = equilibrated.get("sectors", {}).get(str(ell), {})
        row_pairs: list[dict[str, Any]] = []
        if not isinstance(original_sector, Mapping) or not isinstance(equilibrated_sector, Mapping):
            all_pass = False
            sector_rows.append({"ell": int(ell), "passes": False, "reason": "missing_sector"})
            continue
        for sweep_name in (
            "basis_sweep",
            "grid_sweep_largest_basis",
            "domain_sweep_largest_basis",
        ):
            left_rows = original_sector.get(sweep_name, [])
            right_rows = equilibrated_sector.get(sweep_name, [])
            if not isinstance(left_rows, list) or not isinstance(right_rows, list) or len(left_rows) != len(right_rows):
                row_pairs.append({"sweep": sweep_name, "passes": False, "reason": "coverage_or_length"})
                continue
            for index, (left, right) in enumerate(zip(left_rows, right_rows)):
                if isinstance(left, Mapping) and isinstance(right, Mapping):
                    pair = _record_pair_comparison(left, right, ell)
                    pair["sweep"] = sweep_name
                    pair["index"] = int(index)
                    row_pairs.append(pair)
                else:
                    row_pairs.append({"sweep": sweep_name, "index": int(index), "passes": False})
        left_tight = original_sector.get("tighter_background_same_domain")
        right_tight = equilibrated_sector.get("tighter_background_same_domain")
        if isinstance(left_tight, Mapping) and isinstance(right_tight, Mapping):
            pair = _record_pair_comparison(left_tight, right_tight, ell)
            pair["sweep"] = "tighter_background_same_domain"
            pair["index"] = 0
            row_pairs.append(pair)
        else:
            row_pairs.append({"sweep": "tighter_background_same_domain", "passes": False})
        terminal_pair = _record_pair_comparison(
            original_sector.get("terminal_largest_basis", {}),
            equilibrated_sector.get("terminal_largest_basis", {}),
            ell,
        )
        sector_pass = bool(
            all(item.get("passes") is True for item in row_pairs)
            and terminal_pair.get("passes") is True
        )
        all_pass = all_pass and sector_pass
        sector_rows.append({
            "ell": int(ell),
            "original_conclusion": original_sector.get("conclusion"),
            "equilibrated_raw_conclusion": equilibrated_sector.get("conclusion"),
            "terminal": terminal_pair,
            "protocol_rows": row_pairs,
            "passes": sector_pass,
        })
    controls_pass = bool(
        original.get("terminal_controls_pass") is True
        and equilibrated.get("terminal_controls_pass") is True
    )
    all_pass = bool(all_pass and controls_pass)
    return {
        "spectrum_comparison_limit": SPECTRUM_COMPARISON_LIMIT,
        "original_controls_pass": original.get("terminal_controls_pass") is True,
        "equilibrated_controls_pass": equilibrated.get("terminal_controls_pass") is True,
        "all_protocol_rows_pass": all_pass,
        "sectors": sector_rows,
    }


def _conservative_conclusion(
    original: Mapping[str, Any],
    equilibrated: Mapping[str, Any],
    comparison: Mapping[str, Any],
    ell: int,
    required_controls_pass: bool | None = None,
) -> str:
    """Classify only the computed normalized sector under live controls.

    The old/raw sector is retained for comparison diagnostics, but its rank
    must not veto an otherwise valid normalized calculation.  ``None`` keeps
    this helper useful as a direct aggregation boundary: source-case controls
    are still checked even when a caller has no replay summary.
    """

    unresolved = "UNRESOLVED_FINITE_BASIS_SIGN_OR_CONVERGENCE"
    right = equilibrated.get("sectors", {}).get(str(ell), {})
    comparison_sector = next(
        (
            item for item in comparison.get("sectors", [])
            if isinstance(item, Mapping) and item.get("ell") == ell
        ),
        {},
    )
    source_controls_pass = bool(
        original.get("terminal_controls_pass") is True
        and equilibrated.get("terminal_controls_pass") is True
    )
    if required_controls_pass is None:
        required_controls_pass = source_controls_pass
    if not required_controls_pass or not source_controls_pass:
        return unresolved
    if not isinstance(right, Mapping) or not isinstance(comparison_sector, Mapping):
        return unresolved
    if right.get("basis_and_grid_domain_convergence") is not True:
        return unresolved
    try:
        normalized_sector_controls = stability._spectrum_controls_pass(right, ell)
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError):
        normalized_sector_controls = False
    candidate = right.get("conclusion")
    if normalized_sector_controls and candidate in {
        "FINITE_BASIS_POSITIVE_ONLY",
        "FINITE_BASIS_ROBUST_NEGATIVE_DIRECTION",
    }:
        return str(candidate)
    return unresolved


def _physical_replay_controls(
    background: stability.Background,
    target: str,
    ell: int,
    exponent: int,
) -> dict[str, Any]:
    """Compare one normalized replay's physical vector with its reference.

    Eigenvectors are solved in each transformed coordinate system, then
    lifted back through the actual total diagonal map before all metrics are
    evaluated against the original physical ``H`` and ``G``.  This prevents a
    coordinate-space norm or translation test from being mistaken for a
    physical certificate.
    """

    grid = _grid_background(background, TERMINAL_INTERVALS)
    columns = stability._basis_columns(grid, ell, TERMINAL_SIZE)
    hessian, _ = stability._hessian_quadratic(
        grid,
        ell,
        columns["u"],
        columns["v"],
        columns["vprime"],
    )
    gram = stability._gram_matrix(grid, columns["u"], columns["v"])
    translation = columns["translation_coefficient_vector"] if ell == 1 else None
    labels = columns["labels"]
    h_reference, g_reference, t_reference, _ = _unit_gram_transform(
        hessian,
        gram,
        translation,
    )
    h_replay, g_replay, t_replay, _ = _power_then_unit_transform(exponent)(
        hessian,
        gram,
        translation,
    )
    reference_vector, reference_eigenvalue, reference_rank, reference_size, reference_orth, _ = _low_physical_vector(
        h_reference,
        g_reference,
        labels,
        t_reference,
    )
    replay_vector, replay_eigenvalue, replay_rank, replay_size, replay_orth, _ = _low_physical_vector(
        h_replay,
        g_replay,
        labels,
        t_replay,
    )
    diagonal = np.diag(_sym(gram))
    reference_scale = 1.0 / np.sqrt(diagonal)
    power_scale = np.power(10.0, np.linspace(0.0, float(exponent), diagonal.size))
    scaled_gram_diagonal = power_scale**2 * diagonal
    replay_scale = power_scale / np.sqrt(scaled_gram_diagonal)
    reference_physical = reference_scale * reference_vector
    replay_physical = replay_scale * replay_vector
    reference_metrics = _direct_vector_metrics(
        reference_physical,
        hessian,
        gram,
        translation,
    )
    replay_metrics = _direct_vector_metrics(
        replay_physical,
        hessian,
        gram,
        translation,
    )
    rayleigh_errors = [
        abs(float(reference_metrics["direct_rayleigh_Q_over_W0"]) - reference_eigenvalue),
        abs(float(replay_metrics["direct_rayleigh_Q_over_W0"]) - replay_eigenvalue),
    ]
    eigenvalue_scale = max(
        abs(reference_eigenvalue),
        abs(replay_eigenvalue),
        stability.SIGN_RESOLUTION_FLOOR,
    )
    rayleigh_relative = max(rayleigh_errors) / eigenvalue_scale
    eigenvalue_relative = abs(reference_eigenvalue - replay_eigenvalue) / eigenvalue_scale
    translation_pass = bool(
        ell != 1
        or (
            _finite_scalar(reference_metrics.get("translation_G_orthogonality_relative"))
            and _finite_scalar(replay_metrics.get("translation_G_orthogonality_relative"))
            and float(reference_metrics["translation_G_orthogonality_relative"])
            <= stability.PROJECTION_ORTHOGONALITY_LIMIT
            and float(replay_metrics["translation_G_orthogonality_relative"])
            <= stability.PROJECTION_ORTHOGONALITY_LIMIT
        )
    )
    norm_pass = bool(
        _finite_scalar(reference_metrics.get("physical_gram_norm"))
        and _finite_scalar(replay_metrics.get("physical_gram_norm"))
        and abs(float(reference_metrics["physical_gram_norm"]) - 1.0) <= SPECTRUM_COMPARISON_LIMIT
        and abs(float(replay_metrics["physical_gram_norm"]) - 1.0) <= SPECTRUM_COMPARISON_LIMIT
    )
    rayleigh_pass = bool(
        all(_finite_scalar(value) for value in rayleigh_errors)
        and _finite_scalar(rayleigh_relative)
        and rayleigh_relative <= SPECTRUM_COMPARISON_LIMIT
        and _finite_scalar(eigenvalue_relative)
        and eigenvalue_relative <= SPECTRUM_COMPARISON_LIMIT
    )
    return {
        "target_y": target,
        "ell": int(ell),
        "scale_exponent_k": int(exponent),
        "reference_coordinate_eigenvalue_Q_over_W0": reference_eigenvalue,
        "replayed_coordinate_eigenvalue_Q_over_W0": replay_eigenvalue,
        "reference_rank": reference_rank,
        "reference_size": reference_size,
        "replayed_rank": replay_rank,
        "replayed_size": replay_size,
        "reference_projection_orthogonality_relative": reference_orth,
        "replayed_projection_orthogonality_relative": replay_orth,
        "reference_physical_vector": reference_metrics,
        "replayed_physical_vector": replay_metrics,
        "rayleigh_absolute_error_max": max(rayleigh_errors),
        "rayleigh_relative_error_max": rayleigh_relative,
        "physical_eigenvalue_relative_difference": eigenvalue_relative,
        "physical_norm_pass": norm_pass,
        "rayleigh_pass": rayleigh_pass,
        "translation_pass": translation_pass,
        "passes": bool(norm_pass and rayleigh_pass and translation_pass),
    }


def _rank_replays(background: stability.Background, target: str) -> dict[str, Any]:
    """Retain raw scaling diagnostics and test all post-power unit replays."""

    raw_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for ell in ELL_VALUES:
        original = stability._spectrum_record(
            background, ell, TERMINAL_SIZE, TERMINAL_INTERVALS,
        )
        normalized_reference = stability._spectrum_record(
            background,
            ell,
            TERMINAL_SIZE,
            TERMINAL_INTERVALS,
            spectrum_transform=_unit_gram_transform,
        )
        for exponent in COORDINATE_REPLAY_EXPONENTS:
            raw_replay = stability._spectrum_record(
                background,
                ell,
                TERMINAL_SIZE,
                TERMINAL_INTERVALS,
                spectrum_transform=_power_scale_transform(exponent),
            )
            raw_metadata = raw_replay.get("spectrum_transform_metadata", {})
            raw_rank_invariant = bool(
                raw_replay.get("gram_rank") == original.get("gram_rank")
                and raw_replay.get("internal_gram_rank") == original.get("internal_gram_rank")
            )
            raw_record_controls = stability._validate_spectrum_record(
                stability._jsonable(raw_replay),
                ell,
                TERMINAL_SIZE,
                TERMINAL_INTERVALS,
                TERMINAL_DOMAIN_FM,
            )
            raw_row = {
                "target_y": target,
                "ell": int(ell),
                "basis_size_each_block": TERMINAL_SIZE,
                "intervals": TERMINAL_INTERVALS,
                "box_fm": TERMINAL_DOMAIN_FM,
                "scale_exponent_k": int(exponent),
                "replay_kind": "raw_power_only",
                "reference_representation": "original_raw",
                "original_raw_rank": original.get("gram_rank"),
                "original_internal_rank": original.get("internal_gram_rank"),
                "reference_raw_rank": original.get("gram_rank"),
                "reference_internal_rank": original.get("internal_gram_rank"),
                "replayed_raw_rank": raw_replay.get("gram_rank"),
                "replayed_internal_rank": raw_replay.get("internal_gram_rank"),
                "input_raw_rank_from_transform": raw_metadata.get("input_raw_gram_rank"),
                "normalized_rank_from_transform": raw_metadata.get("normalized_gram_rank"),
                "translation_transformed_as_inverse_scale": raw_metadata.get(
                    "translation_transformed_as_inverse_scale"
                ),
                "rank_invariant": raw_rank_invariant,
                "record_controls_pass": bool(raw_record_controls),
                "passes": bool(raw_rank_invariant and raw_record_controls),
            }
            raw_rows.append(raw_row)

            normalized_replay = stability._spectrum_record(
                background,
                ell,
                TERMINAL_SIZE,
                TERMINAL_INTERVALS,
                spectrum_transform=_power_then_unit_transform(exponent),
            )
            normalized_metadata = normalized_replay.get("spectrum_transform_metadata", {})
            normalized_rank_invariant = bool(
                normalized_replay.get("gram_rank") == normalized_reference.get("gram_rank")
                and normalized_replay.get("internal_gram_rank") == normalized_reference.get("internal_gram_rank")
            )
            normalized_record_controls = stability._validate_spectrum_record(
                stability._jsonable(normalized_replay),
                ell,
                TERMINAL_SIZE,
                TERMINAL_INTERVALS,
                TERMINAL_DOMAIN_FM,
            )
            physical = _physical_replay_controls(background, target, ell, exponent)
            normalized_row = {
                "target_y": target,
                "ell": int(ell),
                "basis_size_each_block": TERMINAL_SIZE,
                "intervals": TERMINAL_INTERVALS,
                "box_fm": TERMINAL_DOMAIN_FM,
                "scale_exponent_k": int(exponent),
                "replay_kind": "power_then_unit_equilibrated",
                "reference_representation": "unit_equilibrated",
                "original_raw_rank": original.get("gram_rank"),
                "original_internal_rank": original.get("internal_gram_rank"),
                "reference_raw_rank": normalized_reference.get("gram_rank"),
                "reference_internal_rank": normalized_reference.get("internal_gram_rank"),
                "replayed_raw_rank": normalized_replay.get("gram_rank"),
                "replayed_internal_rank": normalized_replay.get("internal_gram_rank"),
                "input_raw_rank_from_transform": normalized_metadata.get("input_raw_gram_rank"),
                "normalized_rank_from_transform": normalized_metadata.get("normalized_gram_rank"),
                "translation_transformed_as_inverse_scale": normalized_metadata.get(
                    "translation_transformed_as_inverse_scale"
                ),
                "translation_transformed_as_inverse_total_scale": normalized_metadata.get(
                    "translation_transformed_as_inverse_total_scale"
                ),
                "power_scaled_raw_rank": raw_replay.get("gram_rank"),
                "power_scaled_internal_rank": raw_replay.get("internal_gram_rank"),
                "unit_reference_raw_rank": normalized_reference.get("gram_rank"),
                "unit_reference_internal_rank": normalized_reference.get("internal_gram_rank"),
                "rank_invariant": normalized_rank_invariant,
                "record_controls_pass": bool(normalized_record_controls),
                "physical_checks": physical,
                "passes": bool(
                    normalized_rank_invariant
                    and normalized_record_controls
                    and physical.get("passes") is True
                ),
            }
            normalized_rows.append(normalized_row)
    raw_failures = [row for row in raw_rows if row.get("passes") is not True]
    normalized_failures = [row for row in normalized_rows if row.get("passes") is not True]
    rows = raw_rows + normalized_rows
    return {
        "target_y": target,
        "declared_scale": "S_jj=10^(k*j/(m-1))",
        "normalized_replay_transform": "S followed by D_S=diag(S^T G S)^(-1/2), including tbar=(S D_S)^(-1)t",
        "scale_exponents_k": list(COORDINATE_REPLAY_EXPONENTS),
        "terminal_pair": {
            "basis_size_each_block": TERMINAL_SIZE,
            "intervals": TERMINAL_INTERVALS,
            "box_fm": TERMINAL_DOMAIN_FM,
        },
        "row_count": len(rows),
        "expected_row_count": len(ELL_VALUES) * len(COORDINATE_REPLAY_EXPONENTS) * 2,
        "raw_row_count": len(raw_rows),
        "normalized_row_count": len(normalized_rows),
        "expected_raw_row_count": len(ELL_VALUES) * len(COORDINATE_REPLAY_EXPONENTS),
        "expected_normalized_row_count": len(ELL_VALUES) * len(COORDINATE_REPLAY_EXPONENTS),
        "raw_rank_invariance_failures": raw_failures,
        "normalized_rank_invariance_failures": normalized_failures,
        # Keep the historical field as an alias for the raw-only diagnostic;
        # final acceptance uses the explicit normalized field below.
        "rank_invariance_failures": raw_failures,
        "all_raw_rank_replays_pass": not raw_failures,
        "all_normalized_replays_pass": not normalized_failures,
        "all_rank_replays_pass": bool(not raw_failures and not normalized_failures),
        "raw_rows": raw_rows,
        "normalized_rows": normalized_rows,
        "rows": rows,
    }


def _normalized_replay_result_pass(replay: Mapping[str, Any] | None) -> bool:
    """Recompute normalized replay acceptance from the full live validator.

    Raw power-only rank loss remains a diagnostic, so the shared replay
    validator deliberately checks its structure without requiring the raw
    success flag.  It does, however, recompute every normalized row's rank
    linkage and physical certificate before this aggregation boundary can
    accept the normalized representation.
    """

    if not isinstance(replay, Mapping):
        return False
    target = replay.get("target_y")
    if target not in TARGETS:
        return False
    return _validate_rank_replay(replay, str(target))


def _synthetic_controls() -> dict[str, Any]:
    """Known-answer controls for signs, congruence and translation removal."""

    hessian = np.diag((1.0, -1.0e-12))
    gram = np.diag((1.0, 1.0e-12))
    hbar, gbar, _, metadata = _unit_gram_transform(hessian, gram, None)
    oracle = stability._generalized_spectrum(hbar, gbar, ["positive", "negative"])
    oracle_values = np.asarray(oracle["eigenvalues_Q_over_W0"], dtype=float)
    oracle_pass = bool(
        metadata["normalized_gram_rank"] == 2
        and np.allclose(oracle_values, np.array((-1.0, 1.0)), rtol=1.0e-12, atol=1.0e-12)
    )
    power_rows: list[dict[str, Any]] = []
    for exponent in COORDINATE_REPLAY_EXPONENTS:
        scaled_hessian, scaled_gram, _, scaled_metadata = _power_then_unit_transform(exponent)(
            hessian,
            gram,
            None,
        )
        scaled_oracle = stability._generalized_spectrum(
            scaled_hessian,
            scaled_gram,
            ["positive", "negative"],
        )
        scaled_values = np.asarray(scaled_oracle["eigenvalues_Q_over_W0"], dtype=float)
        power_rows.append({
            "scale_exponent_k": int(exponent),
            "computed_eigenvalues_after_power_then_unit": scaled_values,
            "normalized_rank": scaled_metadata["normalized_gram_rank"],
            "input_raw_rank": scaled_metadata["input_raw_gram_rank"],
            "power_scaled_rank": scaled_metadata["power_scaled_gram_rank"],
            "passes": bool(
                scaled_metadata["normalized_gram_rank"] == 2
                and np.allclose(scaled_values, np.array((-1.0, 1.0)), rtol=1.0e-12, atol=1.0e-12)
            ),
        })

    congruence = np.array(((1.0, 0.2), (0.1, 1.3)))
    h0 = np.diag((-0.7, 1.2))
    g0 = np.diag((1.0, 0.5))
    original = stability._generalized_spectrum(
        congruence.T @ h0 @ congruence,
        congruence.T @ g0 @ congruence,
        ["negative", "positive"],
    )
    h_congruent, g_congruent, _, _ = _unit_gram_transform(
        congruence.T @ h0 @ congruence,
        congruence.T @ g0 @ congruence,
        None,
    )
    transformed = stability._generalized_spectrum(
        h_congruent,
        g_congruent,
        ["negative", "positive"],
    )
    congruence_difference = _relative_vector_error(
        np.asarray(original["eigenvalues_Q_over_W0"], dtype=float),
        np.asarray(transformed["eigenvalues_Q_over_W0"], dtype=float),
    )
    congruence_pass = bool(congruence_difference["passes"])

    translated = stability._generalized_spectrum(
        np.diag((-1.0, 2.0)),
        np.eye(2),
        ["translation", "physical"],
        excluded_vector=np.array((1.0, 0.0)),
    )
    raw_low = float(np.asarray(translated["eigenvalues_Q_over_W0"], dtype=float)[0])
    internal_low = float(np.asarray(translated["internal"]["eigenvalues_Q_over_W0"], dtype=float)[0])
    translation_pass = bool(raw_low < 0.0 and internal_low > 0.0)
    return {
        "synthetic_mathematical_controls_only": True,
        "oracle_2x2": {
            "gram_diagonal": [1.0, 1.0e-12],
            "hessian_diagonal": [1.0, -1.0e-12],
            "computed_eigenvalues_after_unit_gram": oracle_values,
            "expected_eigenvalues": [-1.0, 1.0],
            "normalized_rank": metadata["normalized_gram_rank"],
            "passes": oracle_pass,
        },
        "oracle_2x2_power_then_unit": {
            "rows": power_rows,
            "expected_exponents": list(COORDINATE_REPLAY_EXPONENTS),
            "all_pass": all(row["passes"] for row in power_rows),
        },
        "nonorthogonal_well_conditioned_congruence": {
            "computed_relative_difference": congruence_difference,
            "passes": congruence_pass,
        },
        "translated_negative_mode": {
            "raw_low_eigenvalue": raw_low,
            "projected_internal_low_eigenvalue": internal_low,
            "passes": translation_pass,
        },
        "all_checks_pass": bool(
            oracle_pass
            and all(row["passes"] for row in power_rows)
            and congruence_pass
            and translation_pass
        ),
    }


def _simpson_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    intervals = x.size - 1
    if intervals < 2 or intervals % 2:
        raise CoordinateResponseError("positive Simpson weights require an even interval count")
    steps = np.diff(x)
    if not np.allclose(steps, steps[0], rtol=1.0e-12, atol=1.0e-14):
        raise CoordinateResponseError("weighted column check requires a uniform grid")
    weights = np.full(x.size, 2.0 * steps[0] / 3.0, dtype=float)
    weights[0] = steps[0] / 3.0
    weights[-1] = steps[0] / 3.0
    weights[1:-1:2] = 4.0 * steps[0] / 3.0
    return weights


def _weighted_column_independence(
    background: stability.Background,
    target: str,
    ell: int,
) -> dict[str, Any]:
    grid = _grid_background(background, TERMINAL_INTERVALS)
    columns = stability._basis_columns(grid, ell, TERMINAL_SIZE)
    scale = max(grid.design.n0_dim, 1.0e-30)
    features = np.vstack((columns["u"] / scale, columns["v"]))
    weights = _simpson_weights(grid.x)
    radial_weights = weights * grid.x**2
    weighted = features * np.sqrt(np.concatenate((radial_weights, radial_weights)))[:, None]
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    singular_values = np.asarray(singular_values, dtype=float)
    positive = singular_values > 0.0
    largest = float(np.max(singular_values[positive])) if np.any(positive) else 0.0
    cutoff = stability.GRAM_RELATIVE_CUTOFF * largest
    svd_rank = int(np.count_nonzero(np.isfinite(singular_values) & (singular_values >= cutoff)))
    gram = stability._gram_matrix(grid, columns["u"], columns["v"])
    gram_rank, gram_size = _gram_rank(gram)
    diagonal = np.diag(gram)
    unit_scale = 1.0 / np.sqrt(diagonal)
    normalized_weighted = weighted * unit_scale[None, :]
    normalized_singular_values = np.asarray(
        np.linalg.svd(normalized_weighted, compute_uv=False),
        dtype=float,
    )
    normalized_positive = normalized_singular_values > 0.0
    normalized_largest = (
        float(np.max(normalized_singular_values[normalized_positive]))
        if np.any(normalized_positive) else 0.0
    )
    normalized_cutoff = stability.GRAM_RELATIVE_CUTOFF * normalized_largest
    normalized_svd_rank = int(np.count_nonzero(
        np.isfinite(normalized_singular_values)
        & (normalized_singular_values >= normalized_cutoff)
    ))
    normalized_gram = _sym(unit_scale[:, None] * gram * unit_scale[None, :])
    normalized_gram_rank, normalized_gram_size = _gram_rank(normalized_gram)
    number = stability._number_constraint_diagnostics(grid, columns["u"])
    return {
        "target_y": target,
        "ell": int(ell),
        "basis_size_each_block": TERMINAL_SIZE,
        "intervals": TERMINAL_INTERVALS,
        "box_fm": TERMINAL_DOMAIN_FM,
        "column_count": int(features.shape[1]),
        "weighted_svd_rank": svd_rank,
        "gram_rank": gram_rank,
        "gram_size": gram_size,
        "rank_matches_actual_gram": bool(svd_rank == gram_rank),
        "weighted_svd_rank_raw": svd_rank,
        "gram_rank_raw": gram_rank,
        "weighted_svd_rank_normalized": normalized_svd_rank,
        "gram_rank_normalized": normalized_gram_rank,
        "gram_size_normalized": normalized_gram_size,
        "rank_matches_normalized_gram": bool(normalized_svd_rank == normalized_gram_rank),
        "raw_and_normalized_ranks_are_distinguished": True,
        "singular_value_max": largest,
        "singular_value_min_positive": float(np.min(singular_values[positive])) if np.any(positive) else None,
        "weighted_svd_condition_number": (
            float(np.max(singular_values[positive]) / np.min(singular_values[positive]))
            if np.any(positive) else None
        ),
        "simpson_weight_min_positive": float(np.min(radial_weights[radial_weights > 0.0])),
        "number_constraint_residual_max": number["number_constraint_residual_max"],
        "physical_column_independence_is_diagnostic_only": True,
    }


def _projection_basis(gram: np.ndarray, translation: np.ndarray | None) -> tuple[np.ndarray, float]:
    gram = _sym(gram)
    if translation is None:
        return np.eye(gram.shape[0]), 0.0
    vector = np.asarray(translation, dtype=float).reshape(-1)
    row = (vector @ gram)[None, :]
    _, _, vh = np.linalg.svd(row, full_matrices=True)
    projection = vh[1:].T
    orthogonality = float(
        np.linalg.norm(vector @ gram @ projection)
        / max(np.linalg.norm(vector @ gram) * max(np.linalg.norm(projection), 1.0), 1.0e-30)
    )
    return projection, orthogonality


def _low_physical_vector(
    hessian: np.ndarray,
    gram: np.ndarray,
    labels: list[str],
    translation: np.ndarray | None,
) -> tuple[np.ndarray, float, int, int, float, bool]:
    raw = stability._solve_generalized_problem(hessian, gram, labels)
    if translation is None:
        return (
            np.asarray(raw["eigenvectors"][:, 0], dtype=float),
            float(raw["eigenvalues_Q_over_W0"][0]),
            int(raw["gram_rank"]),
            int(raw["gram_size"]),
            0.0,
            False,
        )
    projection, orthogonality = _projection_basis(gram, translation)
    projected_hessian = projection.T @ _sym(hessian) @ projection
    projected_gram = projection.T @ _sym(gram) @ projection
    internal = stability._solve_generalized_problem(projected_hessian, projected_gram, labels[1:])
    return (
        projection @ np.asarray(internal["eigenvectors"][:, 0], dtype=float),
        float(internal["eigenvalues_Q_over_W0"][0]),
        int(internal["gram_rank"]),
        int(internal["gram_size"]),
        orthogonality,
        True,
    )


def _direct_vector_metrics(
    vector: np.ndarray,
    hessian: np.ndarray,
    gram: np.ndarray,
    translation: np.ndarray | None,
) -> dict[str, Any]:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    gram = _sym(gram)
    hessian = _sym(hessian)
    norm = float(vector @ gram @ vector)
    quadratic = float(vector @ hessian @ vector)
    rayleigh = quadratic / norm if norm > 0.0 else None
    if translation is None:
        orthogonality = 0.0
    else:
        t = np.asarray(translation, dtype=float).reshape(-1)
        orthogonality = float(
            abs(t @ gram @ vector)
            / max(np.linalg.norm(t @ gram) * math.sqrt(max(norm, 0.0)), 1.0e-30)
        )
    return {
        "physical_gram_norm": norm,
        "direct_quadratic_Q_over_W0": quadratic,
        "direct_rayleigh_Q_over_W0": rayleigh,
        "translation_G_orthogonality_relative": orthogonality,
    }


def _physical_eigenvector_check(
    background: stability.Background,
    target: str,
    ell: int,
) -> dict[str, Any]:
    grid = _grid_background(background, TERMINAL_INTERVALS)
    columns = stability._basis_columns(grid, ell, TERMINAL_SIZE)
    hessian, _ = stability._hessian_quadratic(
        grid,
        ell,
        columns["u"],
        columns["v"],
        columns["vprime"],
    )
    gram = stability._gram_matrix(grid, columns["u"], columns["v"])
    translation = columns["translation_coefficient_vector"] if ell == 1 else None
    labels = columns["labels"]
    original_vector, original_eigenvalue, original_rank, original_size, original_orth, projected = _low_physical_vector(
        hessian, gram, labels, translation,
    )
    hbar, gbar, tbar, metadata = _unit_gram_transform(hessian, gram, translation)
    normalized_vector, normalized_eigenvalue, normalized_rank, normalized_size, normalized_orth, _ = _low_physical_vector(
        hbar, gbar, labels, tbar,
    )
    diagonal = np.diag(gram)
    scale = 1.0 / np.sqrt(diagonal)
    equilibrated_vector = scale * normalized_vector
    original_metrics = _direct_vector_metrics(original_vector, hessian, gram, translation)
    equilibrated_metrics = _direct_vector_metrics(equilibrated_vector, hessian, gram, translation)
    rayleigh_errors = [
        abs(original_metrics["direct_rayleigh_Q_over_W0"] - original_eigenvalue),
        abs(equilibrated_metrics["direct_rayleigh_Q_over_W0"] - normalized_eigenvalue),
    ]
    value_scale = max(
        abs(original_eigenvalue),
        abs(normalized_eigenvalue),
        stability.SIGN_RESOLUTION_FLOOR,
    )
    return {
        "target_y": target,
        "ell": int(ell),
        "basis_size_each_block": TERMINAL_SIZE,
        "intervals": TERMINAL_INTERVALS,
        "box_fm": TERMINAL_DOMAIN_FM,
        "projection_lifted_through_joint_translation": bool(projected),
        "original_raw_or_internal_eigenvalue_Q_over_W0": original_eigenvalue,
        "equilibrated_coordinate_eigenvalue_Q_over_W0": normalized_eigenvalue,
        "original_rank": original_rank,
        "original_size": original_size,
        "equilibrated_rank": normalized_rank,
        "equilibrated_size": normalized_size,
        "original_projection_orthogonality_relative": original_orth,
        "equilibrated_projection_orthogonality_relative": normalized_orth,
        "equilibration_metadata": metadata,
        "original_physical_vector": original_metrics,
        "equilibrated_physical_vector": equilibrated_metrics,
        "rayleigh_absolute_error_max": max(rayleigh_errors),
        "rayleigh_relative_error_max": max(rayleigh_errors) / value_scale,
        "physical_eigenvalue_relative_difference": abs(
            original_eigenvalue - normalized_eigenvalue
        ) / value_scale,
        "passes": bool(
            all(_finite_scalar(value) for value in rayleigh_errors)
            and max(rayleigh_errors) / value_scale <= SPECTRUM_COMPARISON_LIMIT
            and _finite_scalar(original_metrics["physical_gram_norm"])
            and _finite_scalar(equilibrated_metrics["physical_gram_norm"])
            and abs(original_metrics["physical_gram_norm"] - 1.0) <= SPECTRUM_COMPARISON_LIMIT
            and abs(equilibrated_metrics["physical_gram_norm"] - 1.0) <= SPECTRUM_COMPARISON_LIMIT
        ),
    }


def _inertia(values: np.ndarray) -> dict[str, int]:
    values = np.asarray(values, dtype=float).reshape(-1)
    return {
        "positive": int(np.count_nonzero(values > 0.0)),
        "negative": int(np.count_nonzero(values < 0.0)),
        "zero": int(np.count_nonzero(values == 0.0)),
        "dimension": int(values.size),
    }


def _schur_diagnostic(
    background: stability.Background,
    target: str,
    ell: int,
) -> dict[str, Any]:
    if ell == 1:
        return {
            "target_y": target,
            "ell": 1,
            "status": "SKIPPED_JOINT_TRANSLATION_CONSTRAINT",
            "density_interpretation": "excluded_for_joint_ell1_translation_quotient",
            "all_checks_pass": True,
        }
    grid = _grid_background(background, TERMINAL_INTERVALS)
    columns = stability._basis_columns(grid, ell, TERMINAL_SIZE)
    hessian, _ = stability._hessian_quadratic(
        grid,
        ell,
        columns["u"],
        columns["v"],
        columns["vprime"],
    )
    hessian = _sym(hessian)
    density_count = int(columns["displacement_count"])
    scalar_count = int(columns["scalar_count"])
    S = hessian[:density_count, :density_count]
    R = hessian[:density_count, density_count:]
    T = hessian[density_count:, density_count:]
    s_values = np.linalg.eigvalsh(_sym(S))
    s_positive = bool(s_values.size and np.all(np.isfinite(s_values)) and np.all(s_values > 0.0))
    number = stability._number_constraint_diagnostics(grid, columns["u"])
    number_pass = bool(
        ell != 0
        or number["number_constraint_residual_max"] <= stability.NUMBER_CONSTRAINT_RELATIVE_LIMIT
    )
    output: dict[str, Any] = {
        "target_y": target,
        "ell": int(ell),
        "status": "COMPUTED_STRICT_POSITIVE_DENSITY_SCHUR" if s_positive else "UNRESOLVED_NONPOSITIVE_DENSITY_BLOCK",
        "density_column_count": density_count,
        "scalar_column_count": scalar_count,
        "density_block_min_eigenvalue": float(np.min(s_values)) if s_values.size else None,
        "density_block_max_eigenvalue": float(np.max(s_values)) if s_values.size else None,
        "density_block_positive": s_positive,
        "inverse_method": "strict_solve_only_no_pseudoinverse",
        "number_constraint_residual_max": number["number_constraint_residual_max"],
        "number_constraint_pass": number_pass,
        "density_interpretation": "all_existing_density_columns; ell1 handled separately",
    }
    if not s_positive:
        output.update({
            "reconstruction_relative_error_max": None,
            "full_inertia": None,
            "density_block_inertia": None,
            "reduced_inertia": None,
            "inertia_identity_pass": False,
            "reconstruction_pass": False,
            "all_checks_pass": False,
        })
        return output
    Hred = _sym(T - R.T @ np.linalg.solve(S, R))
    probes = [np.eye(scalar_count, dtype=float)[:, index] for index in range(scalar_count)]
    probes.append(np.linspace(-0.35, 0.65, scalar_count))
    cu = np.linspace(0.25, -0.45, density_count)
    errors: list[float] = []
    for cv in probes:
        full = np.concatenate((cu, cv))
        shifted = cu + np.linalg.solve(S, R @ cv)
        direct = float(full @ hessian @ full)
        reconstructed = float(shifted @ S @ shifted + cv @ Hred @ cv)
        errors.append(abs(direct - reconstructed) / max(1.0, abs(direct), abs(reconstructed)))
    full_values = np.linalg.eigvalsh(hessian)
    reduced_values = np.linalg.eigvalsh(Hred)
    density_inertia = _inertia(s_values)
    reduced_inertia = _inertia(reduced_values)
    full_inertia = _inertia(full_values)
    expected_inertia = {
        key: density_inertia[key] + reduced_inertia[key]
        for key in ("positive", "negative", "zero")
    }
    expected_inertia["dimension"] = density_inertia["dimension"] + reduced_inertia["dimension"]
    inertia_pass = bool(full_inertia == expected_inertia)
    reconstruction_max = max(errors) if errors else None
    reconstruction_pass = bool(reconstruction_max <= 1.0e-10)
    output.update({
        "full_inertia": full_inertia,
        "density_block_inertia": density_inertia,
        "reduced_inertia": reduced_inertia,
        "expected_full_inertia_from_Schur": expected_inertia,
        "inertia_identity_pass": inertia_pass,
        "reconstruction_relative_error_max": reconstruction_max,
        "reconstruction_pass": reconstruction_pass,
        "all_checks_pass": bool(s_positive and number_pass and inertia_pass and reconstruction_pass),
    })
    return output


def _p1_operator_and_angular_mass(
    design: Any,
    x: np.ndarray,
    y: np.ndarray,
    ell: int,
) -> Mapping[str, Any]:
    """Assemble exact P1 operator/mass bands without radial dense matrices."""

    lower, diagonal, upper, nodes, pivot = stability._operator_tridiagonal(design, x, y, ell)
    count = int(nodes.size)
    index_of = {int(node): index for index, node in enumerate(nodes)}
    mass_diagonal = np.zeros(count, dtype=float)
    mass_lower = np.zeros(max(count - 1, 0), dtype=float)
    mass_upper = np.zeros(max(count - 1, 0), dtype=float)
    for element in range(x.size - 1):
        step = float(x[element + 1] - x[element])
        local = np.asarray([[step / 3.0, step / 6.0], [step / 6.0, step / 3.0]])
        element_nodes = (element, element + 1)
        for i_local, node_i in enumerate(element_nodes):
            if node_i not in index_of:
                continue
            i_global = index_of[node_i]
            mass_diagonal[i_global] += local[i_local, i_local]
            for j_local, node_j in enumerate(element_nodes):
                if j_local == i_local or node_j not in index_of:
                    continue
                j_global = index_of[node_j]
                if j_global == i_global + 1:
                    mass_upper[i_global] += local[i_local, j_local]
                elif j_global == i_global - 1:
                    mass_lower[j_global] += local[i_local, j_local]
    step = float(x[1] - x[0])
    load_weights = np.zeros_like(x, dtype=float)
    load_weights[1:-1] = step * x[1:-1] ** 2
    load_weights[0] = 0.5 * step * x[0] ** 2
    load_weights[-1] = 0.5 * step * x[-1] ** 2
    return {
        "operator_lower": np.asarray(lower, dtype=float),
        "operator_diagonal": np.asarray(diagonal, dtype=float),
        "operator_upper": np.asarray(upper, dtype=float),
        "mass_lower": mass_lower,
        "mass_diagonal": mass_diagonal,
        "mass_upper": mass_upper,
        "load_weights": load_weights[nodes],
        "nodes": np.asarray(nodes, dtype=int),
        "pivot": float(pivot),
    }


def _band_bilinear(
    left: np.ndarray,
    diagonal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    right: np.ndarray,
) -> float:
    """Evaluate a symmetric/tridiagonal bilinear form from its bands."""

    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    diagonal = np.asarray(diagonal, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    if left.size != right.size or diagonal.size != left.size:
        raise CoordinateResponseError("band bilinear vectors have incompatible dimensions")
    if lower.size != max(left.size - 1, 0) or upper.size != max(left.size - 1, 0):
        raise CoordinateResponseError("band bilinear off-diagonal dimensions are invalid")
    value = float(np.dot(diagonal * left, right))
    if left.size > 1:
        value += float(np.dot(upper * left[:-1], right[1:]))
        value += float(np.dot(lower * left[1:], right[:-1]))
    return value


def _angular_profile(background: stability.Background, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge = float(background.edge_x)
    if not _finite_scalar(edge) or edge <= 0.0:
        raise CoordinateResponseError("angular profile requires a positive TF edge")
    bump = stability._compact_bump(background.x, 0.5 * edge, 0.25 * edge)
    bump /= max(float(np.max(np.abs(bump))), 1.0e-30)
    density = background.design.n0_dim * bump
    density[background.n <= 0.0] = 0.0
    scalar = bump
    if name == "density_only":
        u, v = density, np.zeros_like(scalar)
    elif name == "scalar_only":
        u, v = np.zeros_like(density), scalar
    elif name == "coupled_density_scalar":
        u, v = density, scalar
    else:
        raise CoordinateResponseError(f"unknown angular profile: {name}")
    vprime = np.gradient(v, background.x, edge_order=2)
    return np.asarray(u, dtype=float), np.asarray(v, dtype=float), np.asarray(vprime, dtype=float)


def _angular_record(
    background: stability.Background,
    target: str,
    intervals: int,
    ell: int,
    profile_name: str,
) -> dict[str, Any]:
    grid = _grid_background(background, intervals)
    x = grid.x
    design = grid.design
    u, v, vprime = _angular_profile(grid, profile_name)
    source = design.gomega * u - 2.0 * design.q**2 * grid.y * grid.a * v
    source_norm = float(math.sqrt(max(stability._integral(x, x**2 * source**2), 0.0)))
    source_field_control_pass = bool(_finite_scalar(source_norm) and source_norm > 0.0)
    w1, pivot1, positive1 = stability._operator_solve(design, x, grid.y, ell, source)
    w2, pivot2, positive2 = stability._operator_solve(design, x, grid.y, ell + 1, source)
    w1 = np.asarray(w1[:, 0], dtype=float)
    w2 = np.asarray(w2[:, 0], dtype=float)
    q1_matrix, details1 = stability._hessian_quadratic(
        grid, ell, u[:, None], v[:, None], vprime[:, None],
    )
    q2_matrix, details2 = stability._hessian_quadratic(
        grid, ell + 1, u[:, None], v[:, None], vprime[:, None],
    )
    q1 = float(q1_matrix[0, 0])
    q2 = float(q2_matrix[0, 0])
    tau1 = float(ell * (ell + 1))
    tau2 = float((ell + 1) * (ell + 2))
    delta_tau = tau2 - tau1
    scalar_angular = stability._integral(x, v * v)
    vector_cross = stability._integral(x, w2 * w1)
    continuum_prediction = delta_tau * (scalar_angular - vector_cross)
    continuum_difference = q2 - q1
    continuum_scale = max(
        abs(continuum_difference),
        abs(continuum_prediction),
        abs(delta_tau * scalar_angular),
        abs(delta_tau * vector_cross),
        stability.SIGN_RESOLUTION_FLOOR,
    )
    continuum_error = abs(continuum_difference - continuum_prediction) / continuum_scale
    operator1 = _p1_operator_and_angular_mass(design, x, grid.y, ell)
    operator2 = _p1_operator_and_angular_mass(design, x, grid.y, ell + 1)
    load_weights1 = np.asarray(operator1["load_weights"], dtype=float)
    load_weights2 = np.asarray(operator2["load_weights"], dtype=float)
    nodes1 = np.asarray(operator1["nodes"], dtype=int)
    nodes2 = np.asarray(operator2["nodes"], dtype=int)
    matrix_pivot1 = float(operator1["pivot"])
    matrix_pivot2 = float(operator2["pivot"])
    same_nodes = bool(np.array_equal(nodes1, nodes2))
    same_load = bool(np.allclose(load_weights1, load_weights2, rtol=0.0, atol=0.0))
    if not same_nodes or not same_load:
        raise CoordinateResponseError("angular P1 operators do not share node/load grids")
    load = load_weights1 * source[nodes1]
    w1_nodes, solve_pivot1 = stability._solve_tridiagonal(
        operator1["operator_lower"],
        operator1["operator_diagonal"],
        operator1["operator_upper"],
        load,
    )
    w2_nodes, solve_pivot2 = stability._solve_tridiagonal(
        operator2["operator_lower"],
        operator2["operator_diagonal"],
        operator2["operator_upper"],
        load,
    )
    mass_diagonal = 0.5 * (
        np.asarray(operator1["mass_diagonal"], dtype=float)
        + np.asarray(operator2["mass_diagonal"], dtype=float)
    )
    mass_lower = 0.5 * (
        np.asarray(operator1["mass_lower"], dtype=float)
        + np.asarray(operator2["mass_lower"], dtype=float)
    )
    mass_upper = 0.5 * (
        np.asarray(operator1["mass_upper"], dtype=float)
        + np.asarray(operator2["mass_upper"], dtype=float)
    )
    v_nodes = v[nodes1]
    scalar_angular_discrete = _band_bilinear(
        v_nodes,
        mass_diagonal,
        mass_lower,
        mass_upper,
        v_nodes,
    )
    vector_cross_discrete = _band_bilinear(
        w2_nodes,
        mass_diagonal,
        mass_lower,
        mass_upper,
        w1_nodes,
    )
    q1_discrete = tau1 * scalar_angular_discrete + float(load @ w1_nodes)
    q2_discrete = tau2 * scalar_angular_discrete + float(load @ w2_nodes)
    discrete_prediction = delta_tau * (scalar_angular_discrete - vector_cross_discrete)
    discrete_difference = q2_discrete - q1_discrete
    discrete_scale = max(
        abs(discrete_difference),
        abs(discrete_prediction),
        abs(delta_tau * scalar_angular_discrete),
        abs(delta_tau * vector_cross_discrete),
        1.0e-30,
    )
    discrete_error = abs(discrete_difference - discrete_prediction) / discrete_scale
    derivative = stability._integral(x, v * v - w1 * w1)
    vector_q1 = float(stability._integral(x, x**2 * source * w1))
    vector_q2 = float(stability._integral(x, x**2 * source * w2))
    return {
        "target_y": target,
        "domain_fm": grid.box_fm,
        "intervals": int(intervals),
        "ell": int(ell),
        "ell_next": int(ell + 1),
        "tau": [tau1, tau2],
        "profile": profile_name,
        "profile_definition": "bump center=0.5*edge, halfwidth=0.25*edge, peak-normalized; u occupied-only",
        "profile_peak": float(np.max(np.abs(v))) if np.any(v) else 0.0,
        "source_full_domain_L2": source_norm,
        "source_field_control_pass": source_field_control_pass,
        "operator_pivots": [float(pivot1), float(pivot2)],
        "operator_matrix_pivots": [float(matrix_pivot1), float(matrix_pivot2)],
        "operator_positive": bool(
            positive1
            and positive2
            and pivot1 > 0.0
            and pivot2 > 0.0
            and solve_pivot1 > 0.0
            and solve_pivot2 > 0.0
        ),
        "q1_Q_over_W0": q1,
        "q2_Q_over_W0": q2,
        "q_difference_Q_over_W0": continuum_difference,
        "scalar_angular_integral_v2_dx": scalar_angular,
        "vector_cross_integral_w2w1_dx": vector_cross,
        "continuum_prediction_Q_difference": continuum_prediction,
        "continuum_identity_relative_error": continuum_error,
        "continuum_identity_pass": bool(continuum_error <= ANGULAR_CONTINUUM_LIMIT),
        "fixed_pair_derivative_prediction_dQ_dtau": derivative,
        "vector_resolvent_q1": vector_q1,
        "vector_resolvent_q2": vector_q2,
        "vector_resolvent_difference": vector_q2 - vector_q1,
        "p1_scalar_angular_mass_identity_difference": discrete_difference,
        "p1_resolvent_identity_prediction": discrete_prediction,
        "p1_scalar_angular_integral_v2": scalar_angular_discrete,
        "p1_vector_cross_integral_w2w1": vector_cross_discrete,
        "p1_identity_relative_error": discrete_error,
        "p1_identity_pass": bool(discrete_error <= ANGULAR_DISCRETE_LIMIT),
        "p1_uses_exact_angular_mass_and_tridiagonal_load": True,
        "p1_operator_representation": "exact_tridiagonal_bands",
        "p1_mass_representation": "exact_symmetric_tridiagonal_bands",
        "p1_dense_reference": "tests_only_small_dense_oracle",
        "density_is_occupied_support_restricted": True,
        "operator_details_positive": bool(details1["positive_operator"] and details2["positive_operator"]),
        "pure_density_vector_decreases": bool(
            profile_name != "density_only" or vector_q2 < vector_q1
        ),
    }


def _continuum_refinement_diagnostics(
    rows: list[Mapping[str, Any]],
    target: str,
) -> list[dict[str, Any]]:
    """Compare each fixed profile/domain/ell continuum quantity across grids."""

    refinement_diagnostics: list[dict[str, Any]] = []
    refinement_groups = {
        (float(row["domain_fm"]), int(row["ell"]), row["profile"])
        for row in rows
    }
    refinement_quantities = (
        "q1_Q_over_W0",
        "q2_Q_over_W0",
        "q_difference_Q_over_W0",
        "scalar_angular_integral_v2_dx",
        "vector_cross_integral_w2w1_dx",
        "continuum_prediction_Q_difference",
        "continuum_identity_relative_error",
    )
    for domain_fm, ell, profile in sorted(refinement_groups):
        group = sorted(
            (
                row for row in rows
                if float(row["domain_fm"]) == domain_fm
                and int(row["ell"]) == ell
                and row["profile"] == profile
            ),
            key=lambda row: int(row["intervals"]),
        )
        quantities: dict[str, Any] = {}
        for quantity in refinement_quantities:
            values = [float(row[quantity]) for row in group]
            changes = [
                abs(values[index + 1] - values[index])
                for index in range(len(values) - 1)
            ]
            # An identity *error* can be close to zero; normalizing its
            # refinement change by that tiny error would turn harmless
            # roundoff into a spurious failure.  Use the declared unit scale
            # for the dimensionless error, while retaining its absolute
            # finest change explicitly below.
            baseline = 1.0 if quantity.endswith("relative_error") else stability.SIGN_RESOLUTION_FLOOR
            scale = max(max((abs(value) for value in values), default=0.0), baseline)
            finest_change = changes[-1] if changes else math.inf
            finest_relative = finest_change / scale
            quantities[quantity] = {
                "values_by_intervals": values,
                "absolute_changes_by_refinement": changes,
                "finest_change_absolute": finest_change,
                "finest_change_relative": finest_relative,
                "discretization_relative_threshold": ANGULAR_CONTINUUM_LIMIT,
                "passes": bool(finest_relative <= ANGULAR_CONTINUUM_LIMIT),
            }
        refinement_diagnostics.append({
            "target_y": target,
            "domain_fm": domain_fm,
            "ell": ell,
            "profile": profile,
            "intervals": [int(row["intervals"]) for row in group],
            "quantities": quantities,
            "passes": all(value["passes"] for value in quantities.values()),
        })
    return refinement_diagnostics


def _angular_response_for_domains(
    backgrounds: tuple[stability.Background, stability.Background],
    target: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for domain_background in backgrounds:
        for intervals in GRID_INTERVALS:
            for ell in ANGULAR_PAIRS:
                for profile in ANGULAR_PROFILES:
                    rows.append(_angular_record(domain_background, target, intervals, ell, profile))
    continuum_failures = [row for row in rows if row["continuum_identity_pass"] is not True]
    discrete_failures = [row for row in rows if row["p1_identity_pass"] is not True]
    operator_failures = [row for row in rows if row["operator_positive"] is not True]
    source_failures = [row for row in rows if row["source_field_control_pass"] is not True]
    last_grid = [row for row in rows if row["intervals"] == GRID_INTERVALS[-1]]
    domain_last_grid = {
        str(box): [
            row for row in last_grid
            if abs(float(row["domain_fm"]) - float(box)) <= 1.0e-12
        ]
        for box in DOMAIN_BOXES_FM
    }
    refinement_diagnostics = _continuum_refinement_diagnostics(rows, target)
    refinement_failures = [row for row in refinement_diagnostics if row["passes"] is not True]
    return {
        "target_y": target,
        "profile_definitions": {
            "names": list(ANGULAR_PROFILES),
            "bump": "center=0.5*edge, halfwidth=0.25*edge, peak-normalized",
            "density_rule": "u=n0_dim*b inside occupied support and zero in vacuum",
            "scalar_rule": "v=b over the full radial domain",
        },
        "angular_pairs": list(ANGULAR_PAIRS),
        "grid_intervals": list(GRID_INTERVALS),
        "domain_boxes_fm": list(DOMAIN_BOXES_FM),
        "row_count": len(rows),
        "expected_row_count": len(DOMAIN_BOXES_FM) * len(GRID_INTERVALS) * len(ANGULAR_PAIRS) * len(ANGULAR_PROFILES),
        "continuum_identity_limit": ANGULAR_CONTINUUM_LIMIT,
        "p1_identity_limit": ANGULAR_DISCRETE_LIMIT,
        "continuum_identity_failures": continuum_failures,
        "p1_identity_failures": discrete_failures,
        "operator_failures": operator_failures,
        "source_field_failures": source_failures,
        "all_continuum_identities_pass": not continuum_failures,
        "all_p1_identities_pass": not discrete_failures,
        "all_operator_controls_pass": not operator_failures,
        "all_source_field_controls_pass": not source_failures,
        "continuum_refinement_diagnostics": refinement_diagnostics,
        "continuum_refinement_failures": refinement_failures,
        "all_continuum_refinement_pass": not refinement_failures,
        "all_angular_controls_pass": bool(
            not continuum_failures
            and not discrete_failures
            and not operator_failures
            and not source_failures
            and not refinement_failures
        ),
        "last_grid_domain_checks": {
            box: {
                "row_count": len(domain_rows),
                "all_continuum_pass": all(row["continuum_identity_pass"] for row in domain_rows),
                "maximum_continuum_error": max(
                    (float(row["continuum_identity_relative_error"]) for row in domain_rows),
                    default=None,
                ),
            }
            for box, domain_rows in domain_last_grid.items()
        },
        "fixed_profile_pure_density_vector_decrease_all_pass": all(
            row["pure_density_vector_decreases"] for row in rows
        ),
        "rows": rows,
    }


def _case_result(
    target: str,
    normalized_replay: Mapping[str, Any] | None = None,
    physical_checks: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    original = stability._case_result(target)
    equilibrated = stability._case_result(target, spectrum_transform=_unit_gram_transform)
    comparison = _protocol_comparison(original, equilibrated)
    normalized_replay_pass = _normalized_replay_result_pass(normalized_replay)
    physical_ells = [
        row.get("ell") if isinstance(row, Mapping) else None
        for row in physical_checks or []
    ]
    physical_diagnostics_pass = bool(
        isinstance(physical_checks, list)
        and len(physical_checks) == len(ELL_VALUES)
        and all(_rank_value(ell) for ell in physical_ells)
        and sorted(int(ell) for ell in physical_ells) == list(ELL_VALUES)
        and all(
            _physical_eigenvector_certificate_pass(row, target=str(target), ell=int(row["ell"]))
            for row in physical_checks
        )
    )
    source_case_controls_pass = bool(
        original.get("terminal_controls_pass") is True
        and equilibrated.get("terminal_controls_pass") is True
    )
    required_controls_pass = bool(
        source_case_controls_pass
        and normalized_replay_pass
        and physical_diagnostics_pass
    )
    conservative_sectors = {
        str(ell): _conservative_conclusion(
            original,
            equilibrated,
            comparison,
            ell,
            required_controls_pass=required_controls_pass,
        )
        for ell in ELL_VALUES
    }
    return {
        "target_y": target,
        "target_N": TARGET_N,
        "original": original,
        "equilibrated": equilibrated,
        "coordinate_comparison": comparison,
        "conservative_sector_conclusions": conservative_sectors,
        "source_case_controls_pass": source_case_controls_pass,
        "normalized_replay_controls_pass": normalized_replay_pass,
        "physical_replay_diagnostics_pass": physical_diagnostics_pass,
        "all_conservative_controls_pass": required_controls_pass,
    }


def calculate() -> dict[str, Any]:
    cases: dict[str, Any] = {}
    rank_replays: list[dict[str, Any]] = []
    column_independence: list[dict[str, Any]] = []
    eigenvector_checks: list[dict[str, Any]] = []
    schur_diagnostics: list[dict[str, Any]] = []
    angular_responses: list[dict[str, Any]] = []
    for target in TARGETS:
        diagnostic_backgrounds = stability._backgrounds(target)
        rank_result = _rank_replays(diagnostic_backgrounds[1], target)
        rank_replays.append(rank_result)
        physical_rows: list[dict[str, Any]] = []
        for ell in ELL_VALUES:
            column_independence.append(
                _weighted_column_independence(diagnostic_backgrounds[1], target, ell)
            )
            physical_row = _physical_eigenvector_check(diagnostic_backgrounds[1], target, ell)
            eigenvector_checks.append(physical_row)
            physical_rows.append(physical_row)
            schur_diagnostics.append(
                _schur_diagnostic(diagnostic_backgrounds[1], target, ell)
            )
        cases[target] = _case_result(
            target,
            normalized_replay=rank_result,
            physical_checks=physical_rows,
        )
        angular_responses.append(
            _angular_response_for_domains(
                (diagnostic_backgrounds[0], diagnostic_backgrounds[1]),
                target,
            )
        )
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "model_scope": {
            "source_stability_schema": stability.SCHEMA,
            "accepted_backgrounds_only": True,
            "target_N": TARGET_N,
            "target_y_values": list(TARGETS),
            "original_ell_sectors": list(ELL_VALUES),
            "basis_sizes_each_block": list(BASIS_SIZES),
            "grid_intervals": list(GRID_INTERVALS),
            "domain_boxes_fm": list(DOMAIN_BOXES_FM),
            "coordinate_change": "c=D z; Hbar=D^T H D; Gbar=D^T G D; tbar=D^-1 t",
            "unit_gram_scale": "D_jj=1/sqrt(G_jj)",
            "raw_power_replay_is_diagnostic_only": True,
            "normalized_replay": "S followed by D_S=diag(S^T G S)^(-1/2); final acceptance uses this representation",
            "no_physics_or_parameter_change": True,
            "positive_finite_basis_is_not_full_stability": True,
            "curvatures_are_not_frequencies": True,
            "empirical_weight": 0.0,
        },
        "acceptance_limits": {
            "inherited_gram_relative_cutoff": stability.GRAM_RELATIVE_CUTOFF,
            "inherited_discretization_relative_drift": stability.DISCRETIZATION_RELATIVE_DRIFT_LIMIT,
            "inherited_projection_orthogonality_relative": stability.PROJECTION_ORTHOGONALITY_LIMIT,
            "angular_continuum_identity_relative": ANGULAR_CONTINUUM_LIMIT,
            "angular_discrete_identity_relative": ANGULAR_DISCRETE_LIMIT,
            "spectrum_comparison_relative": SPECTRUM_COMPARISON_LIMIT,
            "schur_reconstruction_relative": 1.0e-10,
        },
        "input_provenance": {
            "stability_source_sha256": hashlib.sha256(
                (HERE / "nvg_droplet_stability_audit.py").read_bytes()
            ).hexdigest(),
            "finite_droplet_source_sha256": hashlib.sha256(
                (HERE / "nvg_finite_droplet_audit.py").read_bytes()
            ).hexdigest(),
            "no_saved_result_used_as_input": True,
        },
        "synthetic_controls": _synthetic_controls(),
        "cases": cases,
        "rank_replays": rank_replays,
        "weighted_column_independence": column_independence,
        "physical_low_eigenvector_checks": eigenvector_checks,
        "density_schur_diagnostics": schur_diagnostics,
        "angular_resolvent_diagnostics": angular_responses,
    }


def _finite_tree(value: Any) -> bool:
    """Reject non-finite numeric leaves while permitting declared text/None."""

    if isinstance(value, Mapping):
        return all(_finite_tree(key) and _finite_tree(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _rank_value(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) >= 0


def _finite_field(mapping: Mapping[str, Any], key: str, *, allow_none: bool = False) -> bool:
    value = mapping.get(key)
    return (allow_none and value is None) or _finite_scalar(value)


def _stability_root(cases: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap child case records for the upstream producer's complete gate."""

    return {
        "schema_version": stability.SCHEMA,
        "status": stability.STATUS,
        "evidence_weight": stability.EVIDENCE_WEIGHT,
        "acceptance_limits": {
            "gram_relative_cutoff": stability.GRAM_RELATIVE_CUTOFF,
            "sign_error_multiplier": stability.SIGN_ERROR_MULTIPLIER,
            "basis_relative_drift": stability.BASIS_RELATIVE_DRIFT_LIMIT,
            "grid_domain_relative_drift": stability.DISCRETIZATION_RELATIVE_DRIFT_LIMIT,
            "fd_agreement_relative": stability.FD_AGREEMENT_RELATIVE_LIMIT,
            "fixed_N_relative": stability.FIXED_N_RELATIVE_LIMIT,
            "number_constraint_relative": stability.NUMBER_CONSTRAINT_RELATIVE_LIMIT,
            "projection_orthogonality_relative": stability.PROJECTION_ORTHOGONALITY_LIMIT,
            "translation_identity_relative": stability.TRANSLATION_RELATIVE_LIMIT,
            "background_tight_nodes": stability.BACKGROUND_TIGHT_NODES,
            "background_tight_tolerance": stability.BACKGROUND_TIGHT_TOLERANCE,
        },
        "cases": dict(cases),
    }


def _pair_shape_matches(
    pair: Mapping[str, Any],
    original: Mapping[str, Any],
    equilibrated: Mapping[str, Any],
    ell: int,
) -> bool:
    """Check a stored comparison against the two source records it summarizes."""

    try:
        expected = _record_pair_comparison(original, equilibrated, ell)
    except (ArithmeticError, TypeError, ValueError):
        return False
    if not isinstance(pair, Mapping):
        return False
    for key in (
        "ell",
        "spectrum_key",
        "raw_rank_original",
        "raw_rank_equilibrated",
        "internal_rank_original",
        "internal_rank_equilibrated",
        "rank_invariant",
        "raw_eigenvalue_comparison",
        "physical_eigenvalue_comparison",
        "internal_eigenvalue_comparison",
        "passes",
    ):
        if pair.get(key) != expected.get(key):
            return False
    return _finite_tree(pair)


def _validate_comparison_case(case: Mapping[str, Any], target: str) -> bool:
    original = case.get("original")
    equilibrated = case.get("equilibrated")
    comparison = case.get("coordinate_comparison")
    if (
        not isinstance(original, Mapping)
        or not isinstance(equilibrated, Mapping)
        or not isinstance(comparison, Mapping)
        or comparison.get("spectrum_comparison_limit") != SPECTRUM_COMPARISON_LIMIT
        or not isinstance(comparison.get("sectors"), list)
        or len(comparison["sectors"]) != len(ELL_VALUES)
        or not isinstance(comparison.get("original_controls_pass"), bool)
        or not isinstance(comparison.get("equilibrated_controls_pass"), bool)
        or comparison.get("original_controls_pass") != (original.get("terminal_controls_pass") is True)
        or comparison.get("equilibrated_controls_pass") != (equilibrated.get("terminal_controls_pass") is True)
        or not isinstance(comparison.get("all_protocol_rows_pass"), bool)
        or not isinstance(case.get("source_case_controls_pass"), bool)
        or not isinstance(case.get("normalized_replay_controls_pass"), bool)
        or not isinstance(case.get("physical_replay_diagnostics_pass"), bool)
        or not isinstance(case.get("all_conservative_controls_pass"), bool)
        or case.get("source_case_controls_pass") != (
            comparison.get("original_controls_pass") is True
            and comparison.get("equilibrated_controls_pass") is True
        )
        or case.get("all_conservative_controls_pass") != (
            case.get("source_case_controls_pass") is True
            and case.get("normalized_replay_controls_pass") is True
            and case.get("physical_replay_diagnostics_pass") is True
        )
    ):
        return False
    conservative = case.get("conservative_sector_conclusions")
    if not isinstance(conservative, Mapping) or set(conservative) != {str(ell) for ell in ELL_VALUES}:
        return False
    expected_comparison_all = bool(
        comparison["original_controls_pass"]
        and comparison["equilibrated_controls_pass"]
    )
    for ell, sector in zip(ELL_VALUES, comparison["sectors"]):
        left = original.get("sectors", {}).get(str(ell), {})
        right = equilibrated.get("sectors", {}).get(str(ell), {})
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or not isinstance(sector, Mapping):
            return False
        if sector.get("ell") != ell:
            return False
        if (
            sector.get("original_conclusion") != left.get("conclusion")
            or sector.get("equilibrated_raw_conclusion") != right.get("conclusion")
            or not isinstance(sector.get("protocol_rows"), list)
            or len(sector["protocol_rows"]) != len(BASIS_SIZES) + len(GRID_INTERVALS) + len(DOMAIN_BOXES_FM) + 1
            or not isinstance(sector.get("passes"), bool)
            or not isinstance(sector.get("terminal"), Mapping)
        ):
            return False
        expected_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any], str, int]] = []
        for sweep_name, expected_length in (
            ("basis_sweep", len(BASIS_SIZES)),
            ("grid_sweep_largest_basis", len(GRID_INTERVALS)),
            ("domain_sweep_largest_basis", len(DOMAIN_BOXES_FM)),
        ):
            left_rows = left.get(sweep_name)
            right_rows = right.get(sweep_name)
            if (
                not isinstance(left_rows, list)
                or not isinstance(right_rows, list)
                or len(left_rows) != expected_length
                or len(right_rows) != expected_length
            ):
                return False
            if not all(
                isinstance(left_row, Mapping) and isinstance(right_row, Mapping)
                for left_row, right_row in zip(left_rows, right_rows)
            ):
                return False
            expected_pairs.extend(
                (left_row, right_row, sweep_name, index)
                for index, (left_row, right_row) in enumerate(zip(left_rows, right_rows))
            )
        left_tight = left.get("tighter_background_same_domain")
        right_tight = right.get("tighter_background_same_domain")
        if not isinstance(left_tight, Mapping) or not isinstance(right_tight, Mapping):
            return False
        expected_pairs.append((left_tight, right_tight, "tighter_background_same_domain", 0))
        for pair, (left_row, right_row, sweep_name, index) in zip(sector["protocol_rows"], expected_pairs):
            if (
                not isinstance(pair, Mapping)
                or pair.get("sweep") != sweep_name
                or pair.get("index") != index
                or not _pair_shape_matches(pair, left_row, right_row, ell)
            ):
                return False
        terminal_expected = _record_pair_comparison(
            left.get("terminal_largest_basis", {}),
            right.get("terminal_largest_basis", {}),
            ell,
        )
        if not _pair_shape_matches(sector["terminal"], left.get("terminal_largest_basis", {}), right.get("terminal_largest_basis", {}), ell):
            return False
        if sector["terminal"] != terminal_expected:
            return False
        expected_sector_pass = bool(
            all(pair.get("passes") is True for pair in sector["protocol_rows"])
            and sector["terminal"].get("passes") is True
        )
        if sector.get("passes") != expected_sector_pass:
            return False
        expected_comparison_all = expected_comparison_all and expected_sector_pass
        expected_conclusion = _conservative_conclusion(
            original,
            equilibrated,
            comparison,
            ell,
            required_controls_pass=case.get("all_conservative_controls_pass") is True,
        )
        if conservative.get(str(ell)) != expected_conclusion:
            return False
    return comparison.get("all_protocol_rows_pass") == expected_comparison_all


def _validate_physical_replay_checks(
    physical: Mapping[str, Any],
    target: str,
    ell: int,
    exponent: int,
) -> bool:
    if (
        not isinstance(physical, Mapping)
        or physical.get("target_y") != target
        or physical.get("ell") != ell
        or physical.get("scale_exponent_k") != exponent
        or not all(_rank_value(physical.get(field)) for field in (
            "reference_rank", "reference_size", "replayed_rank", "replayed_size",
        ))
        or not all(_finite_field(physical, field) for field in (
            "reference_coordinate_eigenvalue_Q_over_W0",
            "replayed_coordinate_eigenvalue_Q_over_W0",
            "reference_projection_orthogonality_relative",
            "replayed_projection_orthogonality_relative",
            "rayleigh_absolute_error_max",
            "rayleigh_relative_error_max",
            "physical_eigenvalue_relative_difference",
        ))
        or not all(isinstance(physical.get(field), bool) for field in (
            "physical_norm_pass", "rayleigh_pass", "translation_pass", "passes",
        ))
    ):
        return False
    reference = physical.get("reference_physical_vector")
    replayed = physical.get("replayed_physical_vector")
    metric_fields = (
        "physical_gram_norm",
        "direct_quadratic_Q_over_W0",
        "direct_rayleigh_Q_over_W0",
        "translation_G_orthogonality_relative",
    )
    if (
        not isinstance(reference, Mapping)
        or not isinstance(replayed, Mapping)
        or not all(_finite_field(reference, field) for field in metric_fields)
        or not all(_finite_field(replayed, field) for field in metric_fields)
        or not _finite_tree(physical)
    ):
        return False
    reference_eigenvalue = float(physical["reference_coordinate_eigenvalue_Q_over_W0"])
    replayed_eigenvalue = float(physical["replayed_coordinate_eigenvalue_Q_over_W0"])
    rayleigh_errors = [
        abs(float(reference["direct_rayleigh_Q_over_W0"]) - reference_eigenvalue),
        abs(float(replayed["direct_rayleigh_Q_over_W0"]) - replayed_eigenvalue),
    ]
    eigenvalue_scale = max(
        abs(reference_eigenvalue),
        abs(replayed_eigenvalue),
        stability.SIGN_RESOLUTION_FLOOR,
    )
    expected_rayleigh_absolute = max(rayleigh_errors)
    expected_rayleigh_relative = expected_rayleigh_absolute / eigenvalue_scale
    expected_eigenvalue_relative = abs(reference_eigenvalue - replayed_eigenvalue) / eigenvalue_scale
    if (
        abs(float(physical["rayleigh_absolute_error_max"]) - expected_rayleigh_absolute)
        > 1.0e-12 * max(1.0, expected_rayleigh_absolute)
        or abs(float(physical["rayleigh_relative_error_max"]) - expected_rayleigh_relative)
        > 1.0e-12 * max(1.0, expected_rayleigh_relative)
        or abs(float(physical["physical_eigenvalue_relative_difference"]) - expected_eigenvalue_relative)
        > 1.0e-12 * max(1.0, expected_eigenvalue_relative)
    ):
        return False
    expected_norm_pass = bool(
        abs(float(reference["physical_gram_norm"]) - 1.0) <= SPECTRUM_COMPARISON_LIMIT
        and abs(float(replayed["physical_gram_norm"]) - 1.0) <= SPECTRUM_COMPARISON_LIMIT
    )
    expected_translation_pass = bool(
        ell != 1
        or (
            float(physical["reference_projection_orthogonality_relative"])
            <= stability.PROJECTION_ORTHOGONALITY_LIMIT
            and float(physical["replayed_projection_orthogonality_relative"])
            <= stability.PROJECTION_ORTHOGONALITY_LIMIT
            and float(reference["translation_G_orthogonality_relative"])
            <= stability.PROJECTION_ORTHOGONALITY_LIMIT
            and float(replayed["translation_G_orthogonality_relative"])
            <= stability.PROJECTION_ORTHOGONALITY_LIMIT
        )
    )
    expected_rayleigh_pass = bool(
        expected_rayleigh_relative <= SPECTRUM_COMPARISON_LIMIT
        and expected_eigenvalue_relative <= SPECTRUM_COMPARISON_LIMIT
    )
    return (
        physical.get("physical_norm_pass") == expected_norm_pass
        and physical.get("rayleigh_pass") == expected_rayleigh_pass
        and physical.get("translation_pass") == expected_translation_pass
        and physical.get("passes") == bool(
            expected_norm_pass and expected_rayleigh_pass and expected_translation_pass
        )
    )


def _physical_eigenvector_measurements(
    row: Mapping[str, Any],
    *,
    target: str | None = None,
    ell: int | None = None,
) -> tuple[bool, bool]:
    """Recompute one physical-vector record's controls and success flag.

    The first return value says that the record is internally consistent; the
    second is the success value implied by its measured norms, Rayleigh data,
    and translation metrics.  Keeping these separate lets serialization retain
    a truthful failing diagnostic while aggregation requires a passing
    certificate.
    """

    if not isinstance(row, Mapping):
        return False, False
    row_target = row.get("target_y")
    row_ell = row.get("ell")
    if (
        (target is not None and row_target != target)
        or (ell is not None and row_ell != ell)
        or not isinstance(row_ell, (int, np.integer))
        or isinstance(row_ell, bool)
        or row.get("basis_size_each_block") != TERMINAL_SIZE
        or row.get("intervals") != TERMINAL_INTERVALS
        or row.get("box_fm") != TERMINAL_DOMAIN_FM
        or not isinstance(row.get("projection_lifted_through_joint_translation"), bool)
        or row.get("projection_lifted_through_joint_translation") is not (int(row_ell) == 1)
        or not all(_rank_value(row.get(field)) for field in (
            "original_rank", "original_size", "equilibrated_rank", "equilibrated_size",
        ))
        or int(row["original_rank"]) > int(row["original_size"])
        or int(row["equilibrated_rank"]) > int(row["equilibrated_size"])
        or not isinstance(row.get("passes"), bool)
        or not all(_finite_field(row, field) for field in (
            "original_raw_or_internal_eigenvalue_Q_over_W0",
            "equilibrated_coordinate_eigenvalue_Q_over_W0",
            "original_projection_orthogonality_relative",
            "equilibrated_projection_orthogonality_relative",
            "rayleigh_absolute_error_max",
            "rayleigh_relative_error_max",
            "physical_eigenvalue_relative_difference",
        ))
    ):
        return False, False

    original_metrics = row.get("original_physical_vector")
    equilibrated_metrics = row.get("equilibrated_physical_vector")
    metric_fields = (
        "physical_gram_norm",
        "direct_quadratic_Q_over_W0",
        "direct_rayleigh_Q_over_W0",
        "translation_G_orthogonality_relative",
    )
    if (
        not isinstance(original_metrics, Mapping)
        or not isinstance(equilibrated_metrics, Mapping)
        or not all(_finite_field(original_metrics, field) for field in metric_fields)
        or not all(_finite_field(equilibrated_metrics, field) for field in metric_fields)
        or not _finite_tree(row)
    ):
        return False, False

    for metrics in (original_metrics, equilibrated_metrics):
        norm = float(metrics["physical_gram_norm"])
        quadratic = float(metrics["direct_quadratic_Q_over_W0"])
        direct_rayleigh = float(metrics["direct_rayleigh_Q_over_W0"])
        if norm <= 0.0:
            return False, False
        expected_direct_rayleigh = quadratic / norm
        if abs(direct_rayleigh - expected_direct_rayleigh) > 1.0e-12 * max(
            1.0, abs(expected_direct_rayleigh)
        ):
            return False, False

    original_eigenvalue = float(row["original_raw_or_internal_eigenvalue_Q_over_W0"])
    equilibrated_eigenvalue = float(row["equilibrated_coordinate_eigenvalue_Q_over_W0"])
    rayleigh_errors = [
        abs(float(original_metrics["direct_rayleigh_Q_over_W0"]) - original_eigenvalue),
        abs(float(equilibrated_metrics["direct_rayleigh_Q_over_W0"]) - equilibrated_eigenvalue),
    ]
    value_scale = max(
        abs(original_eigenvalue),
        abs(equilibrated_eigenvalue),
        stability.SIGN_RESOLUTION_FLOOR,
    )
    expected_rayleigh_absolute = max(rayleigh_errors)
    expected_rayleigh_relative = expected_rayleigh_absolute / value_scale
    expected_eigenvalue_relative = abs(original_eigenvalue - equilibrated_eigenvalue) / value_scale

    def close(measured: Any, expected: float) -> bool:
        return abs(float(measured) - expected) <= 1.0e-12 * max(1.0, abs(expected))

    if (
        not close(row["rayleigh_absolute_error_max"], expected_rayleigh_absolute)
        or not close(row["rayleigh_relative_error_max"], expected_rayleigh_relative)
        or not close(row["physical_eigenvalue_relative_difference"], expected_eigenvalue_relative)
    ):
        return False, False

    orthogonality_metrics = (
        float(row["original_projection_orthogonality_relative"]),
        float(row["equilibrated_projection_orthogonality_relative"]),
        float(original_metrics["translation_G_orthogonality_relative"]),
        float(equilibrated_metrics["translation_G_orthogonality_relative"]),
    )
    expected_norm_pass = bool(
        abs(float(original_metrics["physical_gram_norm"]) - 1.0) <= SPECTRUM_COMPARISON_LIMIT
        and abs(float(equilibrated_metrics["physical_gram_norm"]) - 1.0) <= SPECTRUM_COMPARISON_LIMIT
    )
    expected_translation_pass = bool(
        all(
            0.0 <= value <= stability.PROJECTION_ORTHOGONALITY_LIMIT
            for value in orthogonality_metrics
        )
    )
    expected_rayleigh_pass = bool(
        expected_rayleigh_relative <= SPECTRUM_COMPARISON_LIMIT
        and expected_eigenvalue_relative <= SPECTRUM_COMPARISON_LIMIT
    )
    expected_success = bool(
        expected_norm_pass and expected_rayleigh_pass and expected_translation_pass
    )
    return row.get("passes") is expected_success, expected_success


def _physical_eigenvector_certificate_pass(
    row: Mapping[str, Any],
    *,
    target: str | None = None,
    ell: int | None = None,
) -> bool:
    """Return true only for a consistent, passing physical-vector record."""

    consistent, success = _physical_eigenvector_measurements(
        row,
        target=target,
        ell=ell,
    )
    return bool(consistent and success)


def _validate_rank_replay(replay: Mapping[str, Any], target: str) -> bool:
    raw_count = len(ELL_VALUES) * len(COORDINATE_REPLAY_EXPONENTS)
    total_count = 2 * raw_count
    if (
        replay.get("target_y") != target
        or replay.get("declared_scale") != "S_jj=10^(k*j/(m-1))"
        or replay.get("normalized_replay_transform")
        != "S followed by D_S=diag(S^T G S)^(-1/2), including tbar=(S D_S)^(-1)t"
        or replay.get("scale_exponents_k") != list(COORDINATE_REPLAY_EXPONENTS)
        or replay.get("row_count") != total_count
        or replay.get("expected_row_count") != total_count
        or replay.get("raw_row_count") != raw_count
        or replay.get("normalized_row_count") != raw_count
        or replay.get("expected_raw_row_count") != raw_count
        or replay.get("expected_normalized_row_count") != raw_count
        or not isinstance(replay.get("rows"), list)
        or not isinstance(replay.get("raw_rows"), list)
        or not isinstance(replay.get("normalized_rows"), list)
        or len(replay["rows"]) != total_count
        or len(replay["raw_rows"]) != raw_count
        or len(replay["normalized_rows"]) != raw_count
        or replay.get("rows") != replay.get("raw_rows") + replay.get("normalized_rows")
        or not all(isinstance(replay.get(key), list) for key in (
            "rank_invariance_failures",
            "raw_rank_invariance_failures",
            "normalized_rank_invariance_failures",
        ))
        or not all(isinstance(replay.get(key), bool) for key in (
            "all_raw_rank_replays_pass",
            "all_normalized_replays_pass",
            "all_rank_replays_pass",
        ))
    ):
        return False
    terminal = replay.get("terminal_pair")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("basis_size_each_block") != TERMINAL_SIZE
        or terminal.get("intervals") != TERMINAL_INTERVALS
        or terminal.get("box_fm") != TERMINAL_DOMAIN_FM
    ):
        return False

    def validate_row(row: Any, kind: str, seen: set[tuple[int, int, str]]) -> bool:
        if not isinstance(row, Mapping):
            return False
        ell = row.get("ell")
        exponent = row.get("scale_exponent_k")
        key = (ell, exponent, kind)
        if key in seen or ell not in ELL_VALUES or exponent not in COORDINATE_REPLAY_EXPONENTS:
            return False
        seen.add(key)
        if (
            row.get("target_y") != target
            or row.get("basis_size_each_block") != TERMINAL_SIZE
            or row.get("intervals") != TERMINAL_INTERVALS
            or row.get("box_fm") != TERMINAL_DOMAIN_FM
            or row.get("replay_kind") != kind
            or not all(_rank_value(row.get(field)) for field in (
                "original_raw_rank",
                "original_internal_rank",
                "reference_raw_rank",
                "reference_internal_rank",
                "replayed_raw_rank",
                "replayed_internal_rank",
                "input_raw_rank_from_transform",
                "normalized_rank_from_transform",
            ))
            or not all(isinstance(row.get(field), bool) for field in (
                "rank_invariant", "record_controls_pass", "passes",
            ))
            or row.get("translation_transformed_as_inverse_scale") is not (ell == 1)
            or row.get("rank_invariant") != (
                row.get("reference_raw_rank") == row.get("replayed_raw_rank")
                and row.get("reference_internal_rank") == row.get("replayed_internal_rank")
            )
            or not _finite_tree(row)
        ):
            return False
        if kind == "raw_power_only":
            if (
                row.get("reference_representation") != "original_raw"
                or row.get("reference_raw_rank") != row.get("original_raw_rank")
                or row.get("reference_internal_rank") != row.get("original_internal_rank")
                or row.get("input_raw_rank_from_transform") != row.get("original_raw_rank")
                or row.get("normalized_rank_from_transform") != row.get("replayed_raw_rank")
                or "translation_transformed_as_inverse_total_scale" in row
                or row.get("passes") != (row.get("rank_invariant") and row.get("record_controls_pass"))
            ):
                return False
        else:
            physical = row.get("physical_checks")
            if (
                row.get("reference_representation") != "unit_equilibrated"
                or not _rank_value(row.get("power_scaled_raw_rank"))
                or not _rank_value(row.get("power_scaled_internal_rank"))
                or row.get("unit_reference_raw_rank") != row.get("reference_raw_rank")
                or row.get("unit_reference_internal_rank") != row.get("reference_internal_rank")
                or row.get("input_raw_rank_from_transform") != row.get("original_raw_rank")
                or row.get("normalized_rank_from_transform") != row.get("replayed_raw_rank")
                or row.get("translation_transformed_as_inverse_total_scale") is not (ell == 1)
                or not isinstance(physical, Mapping)
                or not _validate_physical_replay_checks(physical, target, int(ell), int(exponent))
                or physical.get("reference_rank") != (
                    row.get("reference_internal_rank")
                    if ell == 1 else row.get("reference_raw_rank")
                )
                or physical.get("replayed_rank") != (
                    row.get("replayed_internal_rank")
                    if ell == 1 else row.get("replayed_raw_rank")
                )
                or physical.get("reference_size") != TERMINAL_SIZE * 2 - int(ell == 1)
                or physical.get("replayed_size") != TERMINAL_SIZE * 2 - int(ell == 1)
                or row.get("passes") != (
                    row.get("rank_invariant")
                    and row.get("record_controls_pass")
                    and physical.get("passes") is True
                )
            ):
                return False
        return True

    seen: set[tuple[int, int, str]] = set()
    for row in replay["raw_rows"]:
        if not validate_row(row, "raw_power_only", seen):
            return False
    for row in replay["normalized_rows"]:
        if not validate_row(row, "power_then_unit_equilibrated", seen):
            return False
    expected_seen = {
        (ell, exponent, kind)
        for kind in ("raw_power_only", "power_then_unit_equilibrated")
        for ell in ELL_VALUES
        for exponent in COORDINATE_REPLAY_EXPONENTS
    }
    if seen != expected_seen:
        return False
    raw_by_key = {
        (row.get("ell"), row.get("scale_exponent_k")): row
        for row in replay["raw_rows"]
        if isinstance(row, Mapping)
    }
    for row in replay["normalized_rows"]:
        raw = raw_by_key.get((row.get("ell"), row.get("scale_exponent_k")))
        if (
            not isinstance(raw, Mapping)
            or row.get("power_scaled_raw_rank") != raw.get("replayed_raw_rank")
            or row.get("power_scaled_internal_rank") != raw.get("replayed_internal_rank")
        ):
            return False
    raw_failures = [row for row in replay["raw_rows"] if row.get("passes") is not True]
    normalized_failures = [row for row in replay["normalized_rows"] if row.get("passes") is not True]
    return (
        replay.get("raw_rank_invariance_failures") == raw_failures
        and replay.get("rank_invariance_failures") == raw_failures
        and replay.get("normalized_rank_invariance_failures") == normalized_failures
        and replay.get("all_raw_rank_replays_pass") == (not raw_failures)
        and replay.get("all_normalized_replays_pass") == (not normalized_failures)
        and replay.get("all_rank_replays_pass") == (not raw_failures and not normalized_failures)
    )


def _validate_angular_item(item: Mapping[str, Any], target: str) -> bool:
    expected_count = len(DOMAIN_BOXES_FM) * len(GRID_INTERVALS) * len(ANGULAR_PAIRS) * len(ANGULAR_PROFILES)
    profile_definitions = item.get("profile_definitions")
    if (
        item.get("target_y") != target
        or item.get("angular_pairs") != list(ANGULAR_PAIRS)
        or item.get("grid_intervals") != list(GRID_INTERVALS)
        or item.get("domain_boxes_fm") != list(DOMAIN_BOXES_FM)
        or item.get("continuum_identity_limit") != ANGULAR_CONTINUUM_LIMIT
        or item.get("p1_identity_limit") != ANGULAR_DISCRETE_LIMIT
        or not isinstance(profile_definitions, Mapping)
        or profile_definitions.get("names") != list(ANGULAR_PROFILES)
        or profile_definitions.get("bump") != "center=0.5*edge, halfwidth=0.25*edge, peak-normalized"
        or profile_definitions.get("density_rule") != "u=n0_dim*b inside occupied support and zero in vacuum"
        or profile_definitions.get("scalar_rule") != "v=b over the full radial domain"
        or item.get("row_count") != expected_count
        or item.get("expected_row_count") != expected_count
        or not isinstance(item.get("rows"), list)
        or len(item["rows"]) != expected_count
        or not all(isinstance(item.get(key), bool) for key in (
            "all_continuum_identities_pass",
            "all_p1_identities_pass",
            "all_operator_controls_pass",
            "all_source_field_controls_pass",
            "fixed_profile_pure_density_vector_decrease_all_pass",
            "all_continuum_refinement_pass",
            "all_angular_controls_pass",
        ))
        or not all(isinstance(item.get(key), list) for key in (
            "continuum_identity_failures",
            "p1_identity_failures",
            "operator_failures",
            "source_field_failures",
            "continuum_refinement_diagnostics",
            "continuum_refinement_failures",
        ))
    ):
        return False
    seen: set[tuple[float, int, int, str]] = set()
    continuum_failures: list[Mapping[str, Any]] = []
    p1_failures: list[Mapping[str, Any]] = []
    operator_failures: list[Mapping[str, Any]] = []
    source_failures: list[Mapping[str, Any]] = []
    pure_density = True
    for row in item["rows"]:
        if not isinstance(row, Mapping):
            return False
        key = (float(row.get("domain_fm")), int(row.get("intervals")), int(row.get("ell")), row.get("profile"))
        if key in seen or key[0] not in {float(box) for box in DOMAIN_BOXES_FM} or key[1] not in GRID_INTERVALS or key[2] not in ANGULAR_PAIRS or key[3] not in ANGULAR_PROFILES:
            return False
        seen.add(key)
        required = (
            "profile_peak",
            "source_full_domain_L2",
            "q1_Q_over_W0",
            "q2_Q_over_W0",
            "q_difference_Q_over_W0",
            "scalar_angular_integral_v2_dx",
            "vector_cross_integral_w2w1_dx",
            "continuum_prediction_Q_difference",
            "continuum_identity_relative_error",
            "fixed_pair_derivative_prediction_dQ_dtau",
            "vector_resolvent_q1",
            "vector_resolvent_q2",
            "vector_resolvent_difference",
            "p1_scalar_angular_mass_identity_difference",
            "p1_resolvent_identity_prediction",
            "p1_scalar_angular_integral_v2",
            "p1_vector_cross_integral_w2w1",
            "p1_identity_relative_error",
        )
        if (
            row.get("target_y") != target
            or row.get("ell_next") != row.get("ell", -1) + 1
            or not isinstance(row.get("tau"), list)
            or len(row["tau"]) != 2
            or not all(_finite_scalar(value) for value in row["tau"])
            or not all(_finite_scalar(row.get(field)) for field in required)
            or not all(isinstance(row.get(field), bool) for field in (
                "operator_positive",
                "source_field_control_pass",
                "continuum_identity_pass",
                "p1_identity_pass",
                "p1_uses_exact_angular_mass_and_tridiagonal_load",
                "density_is_occupied_support_restricted",
                "operator_details_positive",
                "pure_density_vector_decreases",
            ))
            or row.get("p1_uses_exact_angular_mass_and_tridiagonal_load") is not True
            or row.get("p1_operator_representation") != "exact_tridiagonal_bands"
            or row.get("p1_mass_representation") != "exact_symmetric_tridiagonal_bands"
            or row.get("p1_dense_reference") != "tests_only_small_dense_oracle"
            or row.get("density_is_occupied_support_restricted") is not True
            or not isinstance(row.get("operator_pivots"), list)
            or not isinstance(row.get("operator_matrix_pivots"), list)
            or len(row["operator_pivots"]) != 2
            or len(row["operator_matrix_pivots"]) != 2
            or not all(_finite_scalar(value) for value in row["operator_pivots"] + row["operator_matrix_pivots"])
        ):
            return False
        tau1, tau2 = (float(value) for value in row["tau"])
        delta_tau = tau2 - tau1
        diff = float(row["q_difference_Q_over_W0"])
        prediction = delta_tau * (
            float(row["scalar_angular_integral_v2_dx"])
            - float(row["vector_cross_integral_w2w1_dx"])
        )
        scale = max(
            abs(diff),
            abs(prediction),
            abs(delta_tau * float(row["scalar_angular_integral_v2_dx"])),
            abs(delta_tau * float(row["vector_cross_integral_w2w1_dx"])),
            stability.SIGN_RESOLUTION_FLOOR,
        )
        continuum_error = abs(diff - prediction) / scale
        p1_diff = float(row["p1_scalar_angular_mass_identity_difference"])
        p1_prediction = float(row["p1_resolvent_identity_prediction"])
        p1_scalar = float(row["p1_scalar_angular_integral_v2"])
        p1_vector = float(row["p1_vector_cross_integral_w2w1"])
        p1_scale = max(
            abs(p1_diff),
            abs(p1_prediction),
            abs(delta_tau * p1_scalar),
            abs(delta_tau * p1_vector),
            1.0e-30,
        )
        p1_error = abs(p1_diff - p1_prediction) / p1_scale
        if (
            abs(float(row["q2_Q_over_W0"]) - float(row["q1_Q_over_W0"]) - diff) > 1.0e-12 * max(1.0, abs(diff))
            or abs(float(row["continuum_prediction_Q_difference"]) - prediction) > 1.0e-12 * max(1.0, abs(prediction))
            or abs(float(row["continuum_identity_relative_error"]) - continuum_error) > 1.0e-12 * max(1.0, continuum_error)
            or abs(float(row["p1_identity_relative_error"]) - p1_error) > 1.0e-12 * max(1.0, p1_error)
            or row.get("source_field_control_pass") != (
                _finite_scalar(row.get("source_full_domain_L2"))
                and float(row["source_full_domain_L2"]) > 0.0
            )
            or row.get("continuum_identity_pass") != (continuum_error <= ANGULAR_CONTINUUM_LIMIT)
            or row.get("p1_identity_pass") != (p1_error <= ANGULAR_DISCRETE_LIMIT)
            or row.get("pure_density_vector_decreases") != (
                row.get("profile") != "density_only"
                or float(row["vector_resolvent_q2"]) < float(row["vector_resolvent_q1"])
            )
            or not _finite_tree(row)
        ):
            return False
        pure_density = pure_density and row.get("pure_density_vector_decreases") is True
        if row.get("continuum_identity_pass") is not True:
            continuum_failures.append(row)
        if row.get("p1_identity_pass") is not True:
            p1_failures.append(row)
        if row.get("operator_positive") is not True:
            operator_failures.append(row)
        if row.get("source_field_control_pass") is not True:
            source_failures.append(row)
    if seen != {
        (float(box), int(intervals), int(ell), profile)
        for box in DOMAIN_BOXES_FM
        for intervals in GRID_INTERVALS
        for ell in ANGULAR_PAIRS
        for profile in ANGULAR_PROFILES
    }:
        return False
    checks = item.get("last_grid_domain_checks")
    if not isinstance(checks, Mapping) or set(checks) != {str(box) for box in DOMAIN_BOXES_FM}:
        return False
    for box in DOMAIN_BOXES_FM:
        domain_rows = [row for row in item["rows"] if row.get("intervals") == GRID_INTERVALS[-1] and float(row.get("domain_fm")) == float(box)]
        value = checks.get(str(box))
        if not isinstance(value, Mapping) or value.get("row_count") != len(domain_rows) or not isinstance(value.get("all_continuum_pass"), bool):
            return False
        expected_max = max((float(row["continuum_identity_relative_error"]) for row in domain_rows), default=None)
        if value.get("all_continuum_pass") != all(row["continuum_identity_pass"] for row in domain_rows):
            return False
        if expected_max is None:
            if value.get("maximum_continuum_error") is not None:
                return False
        elif not _finite_scalar(value.get("maximum_continuum_error")) or abs(float(value["maximum_continuum_error"]) - expected_max) > 1.0e-12 * max(1.0, expected_max):
            return False
    try:
        expected_refinement = _jsonable(
            _continuum_refinement_diagnostics(item["rows"], target)
        )
    except (ArithmeticError, CoordinateResponseError, TypeError, ValueError):
        return False
    expected_refinement_failures = [
        row for row in expected_refinement if row.get("passes") is not True
    ]
    if (
        item.get("continuum_refinement_diagnostics") != expected_refinement
        or item.get("continuum_refinement_failures") != expected_refinement_failures
        or item.get("all_continuum_refinement_pass") != (not expected_refinement_failures)
        or item.get("all_angular_controls_pass") != bool(
            not continuum_failures
            and not p1_failures
            and not operator_failures
            and not source_failures
            and not expected_refinement_failures
        )
    ):
        return False
    return (
        item.get("continuum_identity_failures") == continuum_failures
        and item.get("p1_identity_failures") == p1_failures
        and item.get("operator_failures") == operator_failures
        and item.get("source_field_failures") == source_failures
        and item.get("all_continuum_identities_pass") == (not continuum_failures)
        and item.get("all_p1_identities_pass") == (not p1_failures)
        and item.get("all_operator_controls_pass") == (not operator_failures)
        and item.get("all_source_field_controls_pass") == (not source_failures)
        and item.get("fixed_profile_pure_density_vector_decrease_all_pass") == pure_density
    )


def _validate_synthetic_controls(controls: Any) -> bool:
    if not isinstance(controls, Mapping):
        return False
    try:
        expected = _jsonable(_synthetic_controls())
    except (ArithmeticError, TypeError, ValueError):
        return False
    return controls == expected


def _validate_shape(result: Mapping[str, Any]) -> bool:
    if not isinstance(result, Mapping) or not _finite_tree(result):
        return False
    if (
        result.get("schema_version") != SCHEMA
        or result.get("status") != STATUS
        or result.get("evidence_weight") != EVIDENCE_WEIGHT
    ):
        return False
    scope = result.get("model_scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("source_stability_schema") != stability.SCHEMA
        or scope.get("accepted_backgrounds_only") is not True
        or scope.get("target_N") != TARGET_N
        or scope.get("target_y_values") != list(TARGETS)
        or scope.get("original_ell_sectors") != list(ELL_VALUES)
        or scope.get("basis_sizes_each_block") != list(BASIS_SIZES)
        or scope.get("grid_intervals") != list(GRID_INTERVALS)
            or scope.get("domain_boxes_fm") != list(DOMAIN_BOXES_FM)
            or scope.get("coordinate_change") != "c=D z; Hbar=D^T H D; Gbar=D^T G D; tbar=D^-1 t"
            or scope.get("unit_gram_scale") != "D_jj=1/sqrt(G_jj)"
            or scope.get("raw_power_replay_is_diagnostic_only") is not True
            or scope.get("normalized_replay")
            != "S followed by D_S=diag(S^T G S)^(-1/2); final acceptance uses this representation"
            or scope.get("no_physics_or_parameter_change") is not True
        or scope.get("positive_finite_basis_is_not_full_stability") is not True
        or scope.get("curvatures_are_not_frequencies") is not True
        or scope.get("empirical_weight") != 0.0
    ):
        return False
    expected_limits = {
        "inherited_gram_relative_cutoff": stability.GRAM_RELATIVE_CUTOFF,
        "inherited_discretization_relative_drift": stability.DISCRETIZATION_RELATIVE_DRIFT_LIMIT,
        "inherited_projection_orthogonality_relative": stability.PROJECTION_ORTHOGONALITY_LIMIT,
        "angular_continuum_identity_relative": ANGULAR_CONTINUUM_LIMIT,
        "angular_discrete_identity_relative": ANGULAR_DISCRETE_LIMIT,
        "spectrum_comparison_relative": SPECTRUM_COMPARISON_LIMIT,
        "schur_reconstruction_relative": 1.0e-10,
    }
    limits = result.get("acceptance_limits")
    if not isinstance(limits, Mapping) or any(limits.get(key) != value for key, value in expected_limits.items()):
        return False
    if not _validate_synthetic_controls(result.get("synthetic_controls")):
        return False
    cases = result.get("cases")
    if not isinstance(cases, Mapping) or set(cases) != set(TARGETS):
        return False
    original_cases: dict[str, Any] = {}
    equilibrated_cases: dict[str, Any] = {}
    for target in TARGETS:
        case = cases.get(target)
        if not isinstance(case, Mapping):
            return False
        original_cases[target] = case.get("original")
        equilibrated_cases[target] = case.get("equilibrated")
    try:
        if (
            not stability._validate_result_shape(_stability_root(original_cases))
            or not stability._validate_result_shape(_stability_root(equilibrated_cases))
        ):
            return False
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False
    for target in TARGETS:
        case = cases.get(target)
        if (
            not isinstance(case, Mapping)
            or case.get("target_y") != target
            or case.get("target_N") != TARGET_N
            or not _validate_comparison_case(case, target)
        ):
            return False
    ranks = result.get("rank_replays")
    if not isinstance(ranks, list) or len(ranks) != len(TARGETS):
        return False
    for target, replay in zip(TARGETS, ranks):
        if not isinstance(replay, Mapping) or not _validate_rank_replay(replay, target):
            return False
    columns = result.get("weighted_column_independence")
    if not isinstance(columns, list) or len(columns) != len(TARGETS) * len(ELL_VALUES):
        return False
    expected_pairs = {(target, ell) for target in TARGETS for ell in ELL_VALUES}
    seen_columns: set[tuple[Any, Any]] = set()
    for row in columns:
        if not isinstance(row, Mapping):
            return False
        key = (row.get("target_y"), row.get("ell"))
        if key in seen_columns or key not in expected_pairs:
            return False
        seen_columns.add(key)
        if (
            not _rank_value(row.get("weighted_svd_rank"))
            or not _rank_value(row.get("gram_rank"))
            or not _rank_value(row.get("gram_size"))
            or not _rank_value(row.get("weighted_svd_rank_raw"))
            or not _rank_value(row.get("gram_rank_raw"))
            or not _rank_value(row.get("weighted_svd_rank_normalized"))
            or not _rank_value(row.get("gram_rank_normalized"))
            or not _rank_value(row.get("gram_size_normalized"))
            or row.get("rank_matches_actual_gram") != (row.get("weighted_svd_rank") == row.get("gram_rank"))
            or row.get("weighted_svd_rank_raw") != row.get("weighted_svd_rank")
            or row.get("gram_rank_raw") != row.get("gram_rank")
            or row.get("rank_matches_normalized_gram") != (
                row.get("weighted_svd_rank_normalized") == row.get("gram_rank_normalized")
            )
            or row.get("raw_and_normalized_ranks_are_distinguished") is not True
            or row.get("physical_column_independence_is_diagnostic_only") is not True
            or not all(_finite_field(row, field, allow_none=True) for field in (
                "singular_value_max",
                "singular_value_min_positive",
                "weighted_svd_condition_number",
                "simpson_weight_min_positive",
                "number_constraint_residual_max",
            ))
            or not _finite_tree(row)
        ):
            return False
    if seen_columns != expected_pairs:
        return False
    vectors = result.get("physical_low_eigenvector_checks")
    if not isinstance(vectors, list) or len(vectors) != len(TARGETS) * len(ELL_VALUES):
        return False
    seen_vectors: set[tuple[Any, Any]] = set()
    for row in vectors:
        if not isinstance(row, Mapping):
            return False
        key = (row.get("target_y"), row.get("ell"))
        if key in seen_vectors or key not in expected_pairs:
            return False
        seen_vectors.add(key)
        original_metrics = row.get("original_physical_vector")
        equilibrated_metrics = row.get("equilibrated_physical_vector")
        if (
            not _rank_value(row.get("original_rank"))
            or not _rank_value(row.get("original_size"))
            or not _rank_value(row.get("equilibrated_rank"))
            or not _rank_value(row.get("equilibrated_size"))
            or not isinstance(row.get("passes"), bool)
            or not isinstance(original_metrics, Mapping)
            or not isinstance(equilibrated_metrics, Mapping)
            or not _finite_field(row, "original_raw_or_internal_eigenvalue_Q_over_W0")
            or not _finite_field(row, "equilibrated_coordinate_eigenvalue_Q_over_W0")
            or not _finite_field(row, "original_projection_orthogonality_relative")
            or not _finite_field(row, "equilibrated_projection_orthogonality_relative")
            or not _finite_field(row, "rayleigh_absolute_error_max")
            or not _finite_field(row, "rayleigh_relative_error_max")
            or not _finite_field(row, "physical_eigenvalue_relative_difference")
            or not all(_finite_field(original_metrics, field, allow_none=True) for field in (
                "physical_gram_norm", "direct_quadratic_Q_over_W0", "direct_rayleigh_Q_over_W0", "translation_G_orthogonality_relative",
            ))
            or not all(_finite_field(equilibrated_metrics, field, allow_none=True) for field in (
                "physical_gram_norm", "direct_quadratic_Q_over_W0", "direct_rayleigh_Q_over_W0", "translation_G_orthogonality_relative",
            ))
            or not _finite_tree(row)
        ):
            return False
        if not _physical_eigenvector_measurements(
            row,
            target=str(row.get("target_y")),
            ell=int(row.get("ell")),
        )[0]:
            return False
    if seen_vectors != expected_pairs:
        return False
    for target in TARGETS:
        case = cases[target]
        replay = next(
            (item for item in ranks if isinstance(item, Mapping) and item.get("target_y") == target),
            None,
        )
        target_vectors = [
            row for row in vectors
            if isinstance(row, Mapping) and row.get("target_y") == target
        ]
        if not isinstance(replay, Mapping) or len(target_vectors) != len(ELL_VALUES):
            return False
        expected_source_controls = bool(
            case["original"].get("terminal_controls_pass") is True
            and case["equilibrated"].get("terminal_controls_pass") is True
        )
        expected_normalized_replays = _normalized_replay_result_pass(replay)
        expected_physical = bool(
            all(
                _physical_eigenvector_certificate_pass(
                    row,
                    target=target,
                    ell=int(row["ell"]),
                )
                for row in target_vectors
            )
        )
        expected_all_controls = bool(
            expected_source_controls
            and expected_normalized_replays
            and expected_physical
        )
        if (
            case.get("source_case_controls_pass") != expected_source_controls
            or case.get("normalized_replay_controls_pass") != expected_normalized_replays
            or case.get("physical_replay_diagnostics_pass") != expected_physical
            or case.get("all_conservative_controls_pass") != expected_all_controls
        ):
            return False
        for ell in ELL_VALUES:
            expected_conclusion = _conservative_conclusion(
                case["original"],
                case["equilibrated"],
                case["coordinate_comparison"],
                ell,
                required_controls_pass=expected_all_controls,
            )
            if case["conservative_sector_conclusions"].get(str(ell)) != expected_conclusion:
                return False
    schur = result.get("density_schur_diagnostics")
    if not isinstance(schur, list) or len(schur) != len(TARGETS) * len(ELL_VALUES):
        return False
    seen_schur: set[tuple[Any, Any]] = set()
    for row in schur:
        if not isinstance(row, Mapping):
            return False
        key = (row.get("target_y"), row.get("ell"))
        if key in seen_schur or key not in expected_pairs:
            return False
        seen_schur.add(key)
        if row.get("ell") == 1:
            if row.get("status") != "SKIPPED_JOINT_TRANSLATION_CONSTRAINT" or row.get("all_checks_pass") is not True:
                return False
            continue
        if (
            row.get("status") not in {"COMPUTED_STRICT_POSITIVE_DENSITY_SCHUR", "UNRESOLVED_NONPOSITIVE_DENSITY_BLOCK"}
            or not isinstance(row.get("all_checks_pass"), bool)
            or not isinstance(row.get("density_block_positive"), bool)
            or not isinstance(row.get("number_constraint_pass"), bool)
            or not isinstance(row.get("inertia_identity_pass"), bool)
            or not isinstance(row.get("reconstruction_pass"), bool)
            or row.get("status") == "COMPUTED_STRICT_POSITIVE_DENSITY_SCHUR" and row.get("density_block_positive") is not True
            or row.get("status") == "UNRESOLVED_NONPOSITIVE_DENSITY_BLOCK" and row.get("density_block_positive") is not False
            or row.get("all_checks_pass") != (
                row.get("density_block_positive")
                and row.get("number_constraint_pass")
                and row.get("inertia_identity_pass")
                and row.get("reconstruction_pass")
            )
            or not _finite_field(row, "density_block_min_eigenvalue", allow_none=True)
            or not _finite_field(row, "density_block_max_eigenvalue", allow_none=True)
            or not _finite_field(row, "number_constraint_residual_max", allow_none=True)
            or not _finite_field(row, "reconstruction_relative_error_max", allow_none=True)
            or not _finite_tree(row)
        ):
            return False
    if seen_schur != expected_pairs:
        return False
    angular = result.get("angular_resolvent_diagnostics")
    if not isinstance(angular, list) or len(angular) != len(TARGETS):
        return False
    for target, item in zip(TARGETS, angular):
        if not isinstance(item, Mapping) or not _validate_angular_item(item, target):
            return False
    provenance = result.get("input_provenance")
    if (
        not isinstance(provenance, Mapping)
        or not isinstance(provenance.get("stability_source_sha256"), str)
        or not isinstance(provenance.get("finite_droplet_source_sha256"), str)
        or provenance.get("no_saved_result_used_as_input") is not True
    ):
        return False
    return True


_CACHE: dict[str, Any] | None = None


def build_result() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _jsonable(calculate())
    return copy.deepcopy(_CACHE)


def validate_result(result: Mapping[str, Any]) -> bool:
    if not isinstance(result, Mapping):
        return False
    try:
        if not _validate_shape(result):
            return False
        fresh = build_result()
        return _validate_shape(fresh) and result == fresh
    except (ArithmeticError, AttributeError, CoordinateResponseError, IndexError, KeyError, TypeError, ValueError):
        return False


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - parser path.
        raise CoordinateResponseError(message)


def main(argv: list[str] | None = None) -> int:
    parser = JsonArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = build_result()
        if not _validate_shape(result):
            raise CoordinateResponseError("invalid or incomplete coordinate-response result")
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    except (ArithmeticError, AttributeError, CoordinateResponseError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({
            "schema_version": SCHEMA,
            "status": "INVALID_OR_FAILED_DROPLET_COORDINATE_RESPONSE_AUDIT",
            "evidence_weight": EVIDENCE_WEIGHT,
            "error": str(exc),
        }, ensure_ascii=False, allow_nan=False))
        return 2
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
