#!/usr/bin/env python3
"""Residual-guided finite-space enrichment for the accepted W8 droplets.

The implementation deliberately keeps the physical producer in
``nvg_droplet_stability_audit`` as the equation owner.  This module adds only
the numerical representation and compact trial-space layer required by the
sealed residual-enrichment continuation: a full physical whitening of the
existing 36 columns, deterministic compact candidates, a frozen greedy
selection, and an independent held-out replay.  It is a zero-write command;
the JSON output is a fresh calculation and is never used as an input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

sys.dont_write_bytecode = True

try:
    import nvg_droplet_stability_audit as stability
except ImportError:  # pragma: no cover - package-style import support.
    from . import nvg_droplet_stability_audit as stability


HERE = Path(__file__).resolve().parent
SCHEMA = "nvg_droplet_residual_enrichment_audit.v1"
STATUS = "COMPUTED_FINITE_W8_RESIDUAL_ENRICHMENT_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0
TARGETS = tuple(stability.TARGETS)
TARGET_N = stability.TARGET_N
ELL_VALUES = tuple(stability.ELL_VALUES)
BASIS_SIZES = tuple(stability.BASIS_SIZES)
BASE_BLOCK_SIZE = BASIS_SIZES[-1]
BASE_DIMENSION = 2 * BASE_BLOCK_SIZE
TRAINING_INTERVALS = stability.FINE_INTERVALS
TRAINING_BOX_FM = stability.DOMAIN_BOXES_FM[0]
GRID_INTERVALS = tuple(stability.GRID_INTERVALS)
DOMAIN_BOXES_FM = tuple(stability.DOMAIN_BOXES_FM)
VALIDATION_BACKGROUND_KEYS = ("24fm", "32fm", "24fm_tighter")
STAGE_COUNTS = (0, 4, 8)

GRAM_RELATIVE_CUTOFF = stability.GRAM_RELATIVE_CUTOFF
SIGN_ERROR_MULTIPLIER = stability.SIGN_ERROR_MULTIPLIER
SIGN_RESOLUTION_FLOOR = stability.SIGN_RESOLUTION_FLOOR
BASIS_RELATIVE_DRIFT_LIMIT = stability.BASIS_RELATIVE_DRIFT_LIMIT
DISCRETIZATION_RELATIVE_DRIFT_LIMIT = stability.DISCRETIZATION_RELATIVE_DRIFT_LIMIT
NUMBER_CONSTRAINT_RELATIVE_LIMIT = stability.NUMBER_CONSTRAINT_RELATIVE_LIMIT
PROJECTION_ORTHOGONALITY_LIMIT = stability.PROJECTION_ORTHOGONALITY_LIMIT
FIXED_N_RELATIVE_LIMIT = stability.FIXED_N_RELATIVE_LIMIT

# These are representation diagnostics, not new physics/tolerance inputs.
PHYSICAL_RECONSTRUCTION_LIMIT = 1.0e-8
PHYSICAL_ORTHOGONALITY_LIMIT = 1.0e-8
PHYSICAL_RAYLEIGH_LIMIT = 1.0e-7
CANDIDATE_RELATIVE_RANK = GRAM_RELATIVE_CUTOFF
HOLDOUT_RESIDUAL_LIMIT = DISCRETIZATION_RELATIVE_DRIFT_LIMIT

TRAINING_CENTER_FRACTIONS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15)
TRAINING_WIDTH_FRACTIONS = (0.12, 0.20)
HOLDOUT_CENTER_FRACTIONS = (0.325, 0.475, 0.625, 0.775, 0.925, 1.075)
HOLDOUT_WIDTH_FRACTION = 0.16


class ResidualEnrichmentError(ValueError):
    """Fail-closed error for malformed or internally invalid evidence."""


def _number(value: Any, digits: int = 17) -> Any:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResidualEnrichmentError("non-numeric residual-enrichment output") from exc
    if not math.isfinite(number):
        raise ResidualEnrichmentError("non-finite residual-enrichment output")
    return float(format(number, f".{digits}g"))


def _finite_scalar(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and not isinstance(value, bool)
    except (TypeError, ValueError, OverflowError):
        return False


def _jsonable(value: Any) -> Any:
    """Convert NumPy leaves while rejecting non-finite values."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return value


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(key) and _finite_tree(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def _integral(x: np.ndarray, values: np.ndarray) -> float:
    return stability._integral(np.asarray(x, dtype=float), np.asarray(values, dtype=float))


def _integral_matrix(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    return stability._integral_matrix(np.asarray(x, dtype=float), np.asarray(values, dtype=float))


def _simpson_weights(x: np.ndarray) -> np.ndarray:
    """Positive composite-Simpson weights for the actual sampled grid."""

    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < 3 or (x.size - 1) % 2:
        raise ResidualEnrichmentError("physical whitening requires an even-interval Simpson grid")
    steps = np.diff(x)
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ResidualEnrichmentError("physical whitening grid is not strictly increasing")
    if not np.allclose(steps, steps[0], rtol=2.0e-12, atol=0.0):
        raise ResidualEnrichmentError("physical whitening requires a uniform Simpson grid")
    weights = np.full(x.size, 2.0, dtype=float)
    weights[0] = weights[-1] = 1.0
    weights[1:-1:2] = 4.0
    weights *= float(steps[0]) / 3.0
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ResidualEnrichmentError("invalid Simpson weights")
    return weights


def _sym(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    return 0.5 * (array + array.T)


def _bump_with_derivatives(
    x: np.ndarray,
    center: float,
    width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The sealed polynomial bump and its analytic first two derivatives."""

    x = np.asarray(x, dtype=float)
    center = float(center)
    width = float(width)
    if not math.isfinite(center) or not math.isfinite(width) or width <= 0.0:
        raise ResidualEnrichmentError("invalid compact-bump parameters")
    z = (x - center) / width
    inside = np.abs(z) < 1.0
    b = np.zeros_like(x, dtype=float)
    bp = np.zeros_like(x, dtype=float)
    bpp = np.zeros_like(x, dtype=float)
    zz = z[inside]
    one_minus = 1.0 - zz**2
    b[inside] = one_minus**4
    bp[inside] = -8.0 * zz * one_minus**3 / width
    bpp[inside] = one_minus**2 * (56.0 * zz**2 - 8.0) / width**2
    return b, bp, bpp


# Short aliases make the mathematical seam easy to exercise in focused tests.
_analytic_bump = _bump_with_derivatives
_bump = _bump_with_derivatives


def _candidate_id(kind: str, center_fraction: float, width_fraction: float) -> str:
    center_token = format(float(center_fraction), ".6f").rstrip("0").rstrip(".").replace(".", "p")
    width_token = format(float(width_fraction), ".6f").rstrip("0").rstrip(".").replace(".", "p")
    return f"{kind}_c{center_token}_w{width_token}"


def _candidate_specs(*, heldout: bool = False, edge: float | None = None) -> list[dict[str, Any]]:
    """Return the fixed dictionary in deterministic order."""

    if heldout:
        centers = HOLDOUT_CENTER_FRACTIONS
        widths = (HOLDOUT_WIDTH_FRACTION,)
        dictionary_name = "heldout"
    else:
        centers = TRAINING_CENTER_FRACTIONS
        widths = TRAINING_WIDTH_FRACTIONS
        dictionary_name = "training"
    specs: list[dict[str, Any]] = []
    for center_fraction in centers:
        for width_fraction in widths:
            for kind in ("density", "scalar"):
                item: dict[str, Any] = {
                    "candidate_id": _candidate_id(kind, center_fraction, width_fraction),
                    "kind": kind,
                    "dictionary": dictionary_name,
                    "center_fraction_of_training_edge": float(center_fraction),
                    "width_fraction_of_training_edge": float(width_fraction),
                }
                if edge is not None:
                    item["center_x"] = float(edge) * float(center_fraction)
                    item["width_x"] = float(edge) * float(width_fraction)
                specs.append(item)
    return specs


def _edge_aware_number_integral(background: Any, density: np.ndarray) -> float:
    return stability._edge_aware_number_integral(background, np.asarray(density, dtype=float))


def _fixed_number_projection(
    background: Any,
    density: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the existing independent ell-0 fixed-N projection."""

    density = np.asarray(density, dtype=float).copy()
    residual_before = _edge_aware_number_integral(background, density)
    x = np.asarray(background.x, dtype=float)
    scale = max(float(background.edge_x), 4.0 * float(x[1] - x[0]))
    correction = stability._compact_bump(x, 0.45 * scale, 0.25 * scale)
    correction *= max(float(np.max(np.abs(background.n))), 1.0e-30)
    correction_integral = _edge_aware_number_integral(background, correction)
    if not _finite_scalar(correction_integral) or abs(correction_integral) <= 1.0e-30:
        raise ResidualEnrichmentError("failed to construct candidate fixed-N projection")
    coefficient = residual_before / correction_integral
    projected = density - coefficient * correction
    residual_after = _edge_aware_number_integral(background, projected)
    return projected, {
        "applied": True,
        "residual_before": float(residual_before),
        "correction_integral": float(correction_integral),
        "correction_coefficient": float(coefficient),
        "residual_after": float(residual_after),
        "relative_residual_after": float(
            abs(residual_after) / max(_edge_aware_number_integral(background, np.abs(projected)), 1.0e-30)
        ),
    }


@dataclass
class CandidateField:
    spec: dict[str, Any]
    u: np.ndarray
    v: np.ndarray
    vprime: np.ndarray
    correction: dict[str, Any]
    raw_norm: float
    support_left: float
    support_right: float
    support_region: str
    zero_support: bool


def _candidate_field(background: Any, ell: int, spec: Mapping[str, Any]) -> CandidateField:
    if "center_x" not in spec or "width_x" not in spec:
        raise ResidualEnrichmentError("candidate spec lacks frozen absolute center/width")
    x = np.asarray(background.x, dtype=float)
    center = float(spec["center_x"])
    width = float(spec["width_x"])
    bump, bump_prime, bump_second = _bump_with_derivatives(x, center, width)
    correction: dict[str, Any] = {
        "applied": False,
        "residual_before": 0.0,
        "correction_integral": 0.0,
        "correction_coefficient": 0.0,
        "residual_after": 0.0,
        "relative_residual_after": 0.0,
    }
    if spec.get("kind") == "density":
        u = stability._displacement_density(
            x,
            np.asarray(background.n, dtype=float),
            np.asarray(background.nprime, dtype=float),
            int(ell),
            bump,
            bump_prime,
            bump_second,
        )
        if int(ell) == 0:
            u, correction = _fixed_number_projection(background, u)
        v = np.zeros_like(x)
        vprime = np.zeros_like(x)
    elif spec.get("kind") == "scalar":
        u = np.zeros_like(x)
        v = bump
        vprime = bump_prime
    else:
        raise ResidualEnrichmentError("unknown residual candidate kind")
    density_scale = max(float(background.design.n0_dim), 1.0e-30)
    raw_norm_sq = _integral(x, x**2 * (u**2 / density_scale**2 + v**2))
    raw_norm = math.sqrt(max(float(raw_norm_sq), 0.0))
    support_left = center - width
    support_right = center + width
    if support_right <= float(background.edge_x):
        support_region = "interior"
    else:
        support_region = "edge"
    zero_support = bool(raw_norm <= 1.0e-30 or not np.any(np.abs(u) > 0.0) and not np.any(np.abs(v) > 0.0))
    candidate_spec = dict(spec)
    candidate_spec.update({
        "center_x": center,
        "width_x": width,
        "support_left_x": float(support_left),
        "support_right_x": float(support_right),
        "support_region": support_region,
    })
    return CandidateField(
        candidate_spec,
        np.asarray(u, dtype=float),
        np.asarray(v, dtype=float),
        np.asarray(vprime, dtype=float),
        correction,
        float(raw_norm),
        float(support_left),
        float(support_right),
        support_region,
        zero_support,
    )


def _physical_matrix(background: Any, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    x = np.asarray(background.x, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    if v.ndim == 1:
        v = v[:, None]
    if u.shape != v.shape:
        raise ResidualEnrichmentError("physical columns have inconsistent shapes")
    weights = _simpson_weights(x)
    scale = max(float(background.design.n0_dim), 1.0e-30)
    return np.vstack((np.sqrt(weights)[:, None] * x[:, None] * u / scale,
                      np.sqrt(weights)[:, None] * x[:, None] * v))


def _physical_whiten(
    background: Any,
    ell: int,
    columns: Mapping[str, Any],
) -> dict[str, Any]:
    """Full weighted thin-SVD representation of every base column.

    No singular value is discarded by the declared Gram cutoff.  If a base
    direction is genuinely unrepresentable, the function returns a measured
    invalid representation and an identity fallback so the caller can publish
    an unresolved result without manufacturing a positive classification.
    """

    raw_u = np.asarray(columns["u"], dtype=float)
    raw_v = np.asarray(columns["v"], dtype=float)
    raw_vprime = np.asarray(columns["vprime"], dtype=float)
    labels = list(columns.get("labels", []))
    n_columns = int(raw_u.shape[1]) if raw_u.ndim == 2 else 0
    if raw_u.ndim != 2 or raw_v.shape != raw_u.shape or raw_vprime.shape != raw_u.shape:
        raise ResidualEnrichmentError("existing base columns have inconsistent dimensions")
    if len(labels) != n_columns:
        raise ResidualEnrichmentError("physical-column labels do not match their dimension")
    F = _physical_matrix(background, raw_u, raw_v)
    raw_gram = _sym(F.T @ F)
    gram_values = np.linalg.eigvalsh(raw_gram)
    positive_values = gram_values[np.isfinite(gram_values) & (gram_values > 0.0)]
    raw_condition = (
        float(np.max(positive_values) / np.min(positive_values))
        if positive_values.size else math.inf
    )
    try:
        left, singular_values, right_transpose = np.linalg.svd(F, full_matrices=False)
        singular_values = np.asarray(singular_values, dtype=float)
        all_positive = bool(
            singular_values.size == n_columns
            and np.all(np.isfinite(singular_values))
            and np.all(singular_values > 0.0)
        )
        if all_positive:
            transform = np.asarray(right_transpose.T, dtype=float) / singular_values[None, :]
            inverse_transform = np.asarray(right_transpose.T, dtype=float) * singular_values[None, :]
            # inverse_transform above is not generally T^-1; solve below is
            # used for all measured reconstruction checks.
            inverse_transform = np.linalg.solve(transform, np.eye(n_columns))
        else:
            transform = np.eye(n_columns)
            inverse_transform = np.eye(n_columns)
    except (ArithmeticError, np.linalg.LinAlgError, TypeError, ValueError):
        left = np.zeros((F.shape[0], n_columns), dtype=float)
        singular_values = np.zeros(n_columns, dtype=float)
        transform = np.eye(n_columns)
        inverse_transform = np.eye(n_columns)
        all_positive = False
    F_white = F @ transform
    reconstruction = F_white @ inverse_transform - F
    reconstruction_relative = float(
        np.linalg.norm(reconstruction) / max(np.linalg.norm(F), 1.0e-30)
    )
    white_gram = _sym(F_white.T @ F_white)
    orthogonality_relative = float(
        np.linalg.norm(white_gram - np.eye(n_columns)) / max(float(n_columns), 1.0)
    )
    white_eigenvalues = np.linalg.eigvalsh(white_gram)
    white_largest = float(np.max(white_eigenvalues)) if white_eigenvalues.size else 0.0
    cutoff_rank = int(np.count_nonzero(
        np.isfinite(white_eigenvalues)
        & (white_eigenvalues > 0.0)
        & (white_eigenvalues >= GRAM_RELATIVE_CUTOFF * max(white_largest, 1.0e-300))
    ))
    raw_diag = np.diag(raw_gram)
    raw_nonzero_count = int(np.count_nonzero(np.isfinite(raw_diag) & (raw_diag > 0.0)))
    reliable = bool(
        all_positive
        and raw_nonzero_count == n_columns
        and cutoff_rank == n_columns
        and np.all(np.isfinite(F_white))
        and reconstruction_relative <= PHYSICAL_RECONSTRUCTION_LIMIT
        and orthogonality_relative <= PHYSICAL_ORTHOGONALITY_LIMIT
    )
    translation_old = np.asarray(columns.get("translation_coefficient_vector", np.zeros(n_columns)), dtype=float)
    if translation_old.size != n_columns:
        raise ResidualEnrichmentError("translation coefficient vector has wrong base dimension")
    try:
        translation_new = np.linalg.solve(transform, translation_old)
    except (ArithmeticError, np.linalg.LinAlgError, ValueError):
        translation_new = np.zeros(n_columns, dtype=float)
        reliable = False
    return {
        "raw_u": raw_u,
        "raw_v": raw_v,
        "raw_vprime": raw_vprime,
        "u": raw_u @ transform,
        "v": raw_v @ transform,
        "vprime": raw_vprime @ transform,
        "labels": [f"physical_{label}" for label in labels],
        "transform": transform,
        "inverse_transform": inverse_transform,
        "singular_values": singular_values,
        "raw_gram": raw_gram,
        "raw_gram_eigenvalues": gram_values,
        "raw_norms": np.sqrt(np.maximum(raw_diag, 0.0)),
        "raw_condition_number": raw_condition,
        "weighted_matrix_shape": [int(F.shape[0]), int(F.shape[1])],
        "raw_nonzero_column_count": raw_nonzero_count,
        "white_gram": white_gram,
        "white_gram_eigenvalues": white_eigenvalues,
        "white_gram_rank_after_unchanged_cutoff": cutoff_rank,
        "reconstruction_relative": reconstruction_relative,
        "gram_orthogonality_relative": orthogonality_relative,
        "translation_old": translation_old,
        "translation_new": translation_new,
        "all_nonzero_columns_used": True,
        "uses_pseudoinverse": False,
        "rank_truncated_by_unchanged_cutoff": bool(cutoff_rank < n_columns),
        "representation_reliable": reliable,
        "ell": int(ell),
        "base_dimension": n_columns,
    }


@dataclass
class CombinedSpace:
    background: Any
    ell: int
    base: dict[str, Any]
    candidates: list[CandidateField]
    u: np.ndarray
    v: np.ndarray
    vprime: np.ndarray
    hessian: np.ndarray
    gram: np.ndarray
    labels: list[str]
    translation: np.ndarray
    diagnostics: dict[str, Any]


def _assemble_space(
    background: Any,
    ell: int,
    candidate_specs: Sequence[Mapping[str, Any]],
) -> CombinedSpace:
    columns = stability._basis_columns(background, int(ell), BASE_BLOCK_SIZE)
    if (
        np.asarray(columns.get("u")).ndim != 2
        or np.asarray(columns.get("u")).shape[1] != BASE_DIMENSION
        or np.asarray(columns.get("v")).shape != np.asarray(columns.get("u")).shape
        or np.asarray(columns.get("vprime")).shape != np.asarray(columns.get("u")).shape
    ):
        raise ResidualEnrichmentError("existing size-18-per-block base is incomplete")
    base = _physical_whiten(background, int(ell), columns)
    candidates = [_candidate_field(background, int(ell), spec) for spec in candidate_specs]
    u = np.column_stack([base["u"]] + [candidate.u for candidate in candidates])
    v = np.column_stack([base["v"]] + [candidate.v for candidate in candidates])
    vprime = np.column_stack([base["vprime"]] + [candidate.vprime for candidate in candidates])
    # Reassemble the live Hessian/G from the transformed physical fields.  The
    # old stability module owns all matter, Gauss, edge and norm equations.
    hessian, details = stability._hessian_quadratic(background, int(ell), u, v, vprime)
    gram = stability._gram_matrix(background, u, v)
    hessian = _sym(hessian)
    gram = _sym(gram)
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(gram)):
        raise ResidualEnrichmentError("physical Hessian or norm contains non-finite values")
    # A direct old-column assembly is a small, independent representation
    # check.  It is deliberately diagnostic; no old source or threshold is
    # changed by this congruence.
    raw_hessian, _ = stability._hessian_quadratic(
        background,
        int(ell),
        base["raw_u"],
        base["raw_v"],
        base["raw_vprime"],
    )
    raw_gram = base["raw_gram"]
    expected_hessian = base["transform"].T @ raw_hessian @ base["transform"]
    expected_gram = base["transform"].T @ raw_gram @ base["transform"]
    base_hessian_error = float(
        np.linalg.norm(hessian[:BASE_DIMENSION, :BASE_DIMENSION] - expected_hessian)
        / max(np.linalg.norm(expected_hessian), 1.0e-30)
    )
    base_gram_error = float(
        np.linalg.norm(gram[:BASE_DIMENSION, :BASE_DIMENSION] - expected_gram)
        / max(np.linalg.norm(expected_gram), 1.0e-30)
    )
    base["rayleigh_consistency_relative"] = base_hessian_error
    base["gram_reassembly_consistency_relative"] = base_gram_error
    base["representation_reliable"] = bool(
        base["representation_reliable"]
        and base_hessian_error <= PHYSICAL_RAYLEIGH_LIMIT
        and base_gram_error <= PHYSICAL_RAYLEIGH_LIMIT
    )
    translation = np.concatenate((np.asarray(base["translation_new"], dtype=float),
                                  np.zeros(len(candidates), dtype=float)))
    labels = list(base["labels"]) + [str(candidate.spec["candidate_id"]) for candidate in candidates]
    diagnostics = {
        "operator_min_pivot": float(details["operator_min_pivot"]),
        "positive_operator": bool(details["positive_operator"]),
        "base_representation_reliable": bool(base["representation_reliable"]),
        "base_reconstruction_relative": float(base["reconstruction_relative"]),
        "base_gram_orthogonality_relative": float(base["gram_orthogonality_relative"]),
        "base_rayleigh_consistency_relative": float(base_hessian_error),
        "base_gram_reassembly_consistency_relative": float(base_gram_error),
        "base_raw_condition_number": float(base["raw_condition_number"]),
        "base_singular_values": np.asarray(base["singular_values"], dtype=float),
        "base_raw_norms": np.asarray(base["raw_norms"], dtype=float),
        "base_raw_gram_eigenvalues": np.asarray(base["raw_gram_eigenvalues"], dtype=float),
        "base_white_gram_eigenvalues": np.asarray(base["white_gram_eigenvalues"], dtype=float),
        "base_white_rank_after_unchanged_cutoff": int(base["white_gram_rank_after_unchanged_cutoff"]),
        "base_raw_nonzero_column_count": int(base["raw_nonzero_column_count"]),
        "base_dimension": BASE_DIMENSION,
        "all_nonzero_columns_used": True,
        "uses_pseudoinverse": False,
    }
    return CombinedSpace(
        background=background,
        ell=int(ell),
        base=base,
        candidates=candidates,
        u=u,
        v=v,
        vprime=vprime,
        hessian=hessian,
        gram=gram,
        labels=labels,
        translation=translation,
        diagnostics=diagnostics,
    )


def _identity_coefficients(space: CombinedSpace) -> np.ndarray:
    return np.eye(BASE_DIMENSION + len(space.candidates), BASE_DIMENSION)


def _coefficient_fields(space: CombinedSpace, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return space.u @ coefficients, space.v @ coefficients, space.vprime @ coefficients


def _coefficient_inner(space: CombinedSpace, left: np.ndarray, right: np.ndarray) -> float:
    return float(np.asarray(left, dtype=float) @ space.gram @ np.asarray(right, dtype=float))


def _candidate_complement(
    space: CombinedSpace,
    candidate_index: int,
    current: np.ndarray,
) -> dict[str, Any]:
    """Project a candidate from the whole current physical space."""

    candidate = space.candidates[int(candidate_index)]
    dimension = BASE_DIMENSION + len(space.candidates)
    raw = np.zeros(dimension, dtype=float)
    raw[BASE_DIMENSION + int(candidate_index)] = 1.0
    original_norm_sq = _coefficient_inner(space, raw, raw)
    result: dict[str, Any] = {
        "candidate_id": candidate.spec["candidate_id"],
        "candidate_index": int(candidate_index),
        "raw_norm": float(math.sqrt(max(original_norm_sq, 0.0))),
        "zero_support": bool(candidate.zero_support),
        "support_region": candidate.support_region,
        "kind": candidate.spec["kind"],
        "fixed_number_projection": candidate.correction,
        "eligible": False,
        "near_dependent": False,
        "translation_projection_relative": 0.0,
        "remaining_norm": 0.0,
        "remaining_norm_relative": 0.0,
        "reason": "",
    }
    if candidate.zero_support or not math.isfinite(original_norm_sq) or original_norm_sq <= 1.0e-60:
        result["reason"] = "ZERO_SUPPORT"
        return result
    vector = raw.copy()
    translation_norm_sq = _coefficient_inner(space, space.translation, space.translation)
    if space.ell == 1:
        if not math.isfinite(translation_norm_sq) or translation_norm_sq <= 0.0:
            result["reason"] = "INVALID_TRANSLATION_NORM"
            return result
        coefficient = _coefficient_inner(space, space.translation, vector) / translation_norm_sq
        vector = vector - coefficient * space.translation
        result["translation_projection_relative"] = float(
            abs(_coefficient_inner(space, space.translation, vector))
            / max(math.sqrt(translation_norm_sq) * max(math.sqrt(original_norm_sq), 1.0e-30), 1.0e-30)
        )
    try:
        current_gram = _sym(current.T @ space.gram @ current)
        current_inner = current.T @ space.gram @ vector
        projection_coefficients = np.linalg.solve(current_gram, current_inner)
    except (ArithmeticError, np.linalg.LinAlgError, TypeError, ValueError):
        result["reason"] = "INVALID_CURRENT_GRAM"
        return result
    vector = vector - current @ projection_coefficients
    remaining_norm_sq = _coefficient_inner(space, vector, vector)
    if not math.isfinite(remaining_norm_sq) or remaining_norm_sq < 0.0:
        result["reason"] = "INVALID_REMAINING_NORM"
        return result
    remaining_norm = math.sqrt(max(remaining_norm_sq, 0.0))
    # The declared rank resolution is a relative *Gram norm* resolution, so
    # compare G(c_perp,c_perp) with the original candidate Gram norm.  Keep
    # the amplitude ratio separately for readable diagnostics.
    relative = remaining_norm_sq / max(original_norm_sq, 1.0e-60)
    result["remaining_norm"] = float(remaining_norm)
    result["remaining_norm_relative"] = float(relative)
    result["remaining_norm_amplitude_relative"] = float(
        remaining_norm / max(math.sqrt(original_norm_sq), 1.0e-30)
    )
    result["near_dependent"] = bool(relative <= CANDIDATE_RELATIVE_RANK)
    if result["near_dependent"]:
        result["reason"] = "NEAR_DEPENDENT"
        return result
    if not math.isfinite(remaining_norm) or remaining_norm <= 0.0:
        result["reason"] = "ZERO_REMAINING_NORM"
        return result
    normalized = vector / remaining_norm
    result["eligible"] = True
    result["reason"] = "ELIGIBLE"
    result["normalized_coefficient"] = normalized
    result["projection_coefficients"] = projection_coefficients
    result["translation_orthogonality_relative"] = float(
        abs(_coefficient_inner(space, space.translation, normalized))
        / max(math.sqrt(translation_norm_sq), 1.0e-30)
        if space.ell == 1 else 0.0
    )
    return result


def _low_mode_count(values: Sequence[float], residuals: Sequence[float], requested: int = 3) -> int:
    values = [float(value) for value in values]
    residuals = [float(value) for value in residuals]
    count = min(int(requested), len(values))
    if count == 0:
        return 0
    while count < len(values):
        resolution = max(
            SIGN_RESOLUTION_FLOOR,
            abs(residuals[count - 1]) if count - 1 < len(residuals) else 0.0,
            abs(residuals[count]) if count < len(residuals) else 0.0,
        )
        if abs(values[count] - values[count - 1]) > resolution:
            break
        count += 1
    return count


def _spectrum_modes(
    space: CombinedSpace,
    coefficients: np.ndarray,
) -> dict[str, Any]:
    """Solve one small physical trial space and lift its modes."""

    hessian = _sym(coefficients.T @ space.hessian @ coefficients)
    gram = _sym(coefficients.T @ space.gram @ coefficients)
    labels = [f"stage_{index}" for index in range(coefficients.shape[1])]
    raw = stability._solve_generalized_problem(hessian, gram, labels)
    translation_stage = np.zeros(coefficients.shape[1], dtype=float)
    translation_stage[:BASE_DIMENSION] = space.base["translation_new"]
    internal = raw
    projector = np.eye(coefficients.shape[1])
    projection_relative = 0.0
    if space.ell == 1:
        translation_norm_sq = float(translation_stage @ gram @ translation_stage)
        if not math.isfinite(translation_norm_sq) or translation_norm_sq <= 0.0:
            raise ResidualEnrichmentError("ell-1 translation has no positive stage norm")
        row = (translation_stage @ gram)[None, :]
        _, _, vh = np.linalg.svd(row, full_matrices=True)
        projector = vh[1:].T
        projected_hessian = projector.T @ hessian @ projector
        projected_gram = projector.T @ gram @ projector
        internal = stability._solve_generalized_problem(
            projected_hessian,
            projected_gram,
            [f"internal_{index}" for index in range(projector.shape[1])],
        )
        projection_relative = float(
            np.linalg.norm(translation_stage @ gram @ projector)
            / max(np.linalg.norm(translation_stage @ gram) * max(np.linalg.norm(projector), 1.0), 1.0e-30)
        )
    mode_coefficients_stage = projector @ np.asarray(internal["eigenvectors"], dtype=float)
    mode_coefficients_combined = coefficients @ mode_coefficients_stage
    values = np.asarray(internal["eigenvalues_Q_over_W0"], dtype=float)
    residuals = np.asarray(internal["eigen_residual_absolute"], dtype=float)
    low_count = _low_mode_count(values, residuals)
    return {
        "hessian": hessian,
        "gram": gram,
        "raw": raw,
        "internal": internal,
        "projector": projector,
        "translation_stage": translation_stage,
        "translation_norm": math.sqrt(max(float(translation_stage @ gram @ translation_stage), 0.0)),
        "projection_orthogonality_relative": projection_relative,
        "mode_coefficients_stage": mode_coefficients_stage,
        "mode_coefficients_combined": mode_coefficients_combined,
        "eigenvalues": values,
        "residuals": residuals,
        "low_mode_count": low_count,
        "internal_rank": int(internal["gram_rank"]),
        "internal_size": int(internal["gram_size"]),
        "raw_rank": int(raw["gram_rank"]),
        "raw_size": int(raw["gram_size"]),
    }


def _residual_score(
    hessian: np.ndarray,
    gram: np.ndarray,
    candidate: np.ndarray,
    eigenvalues: Sequence[float],
    modes: np.ndarray,
    *,
    residuals: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute the invariant low-mode residual score for a normalized probe."""

    hessian = _sym(hessian)
    gram = _sym(gram)
    candidate = np.asarray(candidate, dtype=float).reshape(-1)
    modes = np.asarray(modes, dtype=float)
    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    if modes.ndim == 1:
        modes = modes[:, None]
    if modes.shape[0] != candidate.size or modes.shape[1] != values.size:
        raise ResidualEnrichmentError("residual-score dimensions do not match")
    norm_sq = float(candidate @ gram @ candidate)
    if not math.isfinite(norm_sq) or norm_sq <= 0.0:
        raise ResidualEnrichmentError("residual-score candidate has no positive norm")
    residual_values_for_resolution = (
        residuals if residuals is not None else [0.0] * values.size
    )
    count = _low_mode_count(values, residual_values_for_resolution)
    cross_h = modes[:, :count].T @ hessian @ candidate
    cross_g = modes[:, :count].T @ gram @ candidate
    residual_values = cross_h - values[:count] * cross_g
    score = float(np.sum(np.abs(residual_values) ** 2) / norm_sq)
    return {
        "score": score,
        "residuals_Q_over_W0": residual_values,
        "residual_count": int(count),
        "norm_squared": norm_sq,
        "normalized_residuals": np.asarray(
            [abs(float(value)) / max(abs(float(eigenvalue)), SIGN_RESOLUTION_FLOOR)
             for value, eigenvalue in zip(residual_values, values[:count])],
            dtype=float,
        ),
    }


# Explicit descriptive alias used by focused mathematical tests.
_candidate_residual_score = _residual_score


def _append_candidate(current: np.ndarray, normalized: np.ndarray) -> np.ndarray:
    normalized = np.asarray(normalized, dtype=float).reshape(-1)
    if current.shape[0] != normalized.size:
        raise ResidualEnrichmentError("candidate coefficient dimension mismatch")
    return np.column_stack((current, normalized))


def _candidate_public_record(
    complement: Mapping[str, Any],
    *,
    score: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        key: value
        for key, value in complement.items()
        if key not in {"normalized_coefficient", "projection_coefficients"}
    }
    if score is not None:
        record.update({
            "score": float(score["score"]),
            "score_residuals_Q_over_W0": np.asarray(score["residuals_Q_over_W0"], dtype=float),
            "score_normalized_residuals": np.asarray(score["normalized_residuals"], dtype=float),
            "score_residual_count": int(score["residual_count"]),
        })
    return _jsonable(record)


def _select_training_candidates(
    space: CombinedSpace,
) -> dict[str, Any]:
    """Greedy two-stage selection on the 24 fm/1600 training grid."""

    current = _identity_coefficients(space)
    selected: list[str] = []
    selected_indices: list[int] = []
    stage_records: list[dict[str, Any]] = []
    for stage_count in (0, 4):
        spectrum = _spectrum_modes(space, current)
        candidates: list[dict[str, Any]] = []
        remaining_indices = [
            index for index, candidate in enumerate(space.candidates)
            if candidate.spec["candidate_id"] not in selected
        ]
        scored: list[tuple[float, int, dict[str, Any]]] = []
        low_modes = spectrum["mode_coefficients_combined"][:, : spectrum["low_mode_count"]]
        low_values = spectrum["eigenvalues"][: spectrum["low_mode_count"]]
        low_residuals = spectrum["residuals"][: spectrum["low_mode_count"]]
        for index in remaining_indices:
            complement = _candidate_complement(space, index, current)
            if complement.get("eligible") is True:
                score = _residual_score(
                    space.hessian,
                    space.gram,
                    complement["normalized_coefficient"],
                    low_values,
                    low_modes,
                    residuals=low_residuals,
                )
                scored.append((float(score["score"]), -index, {
                    "complement": complement,
                    "score": score,
                }))
                candidates.append(_candidate_public_record(complement, score=score))
            else:
                candidates.append(_candidate_public_record(complement))
        # Highest residual score wins; dictionary order is the exact tie break.
        scored.sort(key=lambda item: (-item[0], -item[1]))
        additions: list[dict[str, Any]] = []
        for _, _, item in scored[:4]:
            complement = item["complement"]
            index = int(complement["candidate_index"])
            selected.append(str(complement["candidate_id"]))
            selected_indices.append(index)
            current = _append_candidate(current, complement["normalized_coefficient"])
            additions.append({
                "candidate_id": complement["candidate_id"],
                "candidate_index": index,
                "score": float(item["score"]["score"]),
            })
        stage_records.append({
            "stage_before_additions": int(stage_count),
            "low_mode_count": int(spectrum["low_mode_count"]),
            "current_internal_eigenvalues_Q_over_W0": spectrum["eigenvalues"][:5],
            "candidate_scores": candidates,
            "additions": additions,
            "eligible_count": int(sum(
                bool(item.get("eligible") is True) for item in candidates
            )),
            "zero_support_count": int(sum(
                bool(item.get("zero_support") is True) for item in candidates
            )),
            "near_dependent_count": int(sum(
                bool(item.get("near_dependent") is True) for item in candidates
            )),
        })
    planned = 8
    return {
        "selected_candidate_ids": selected,
        "selected_candidate_indices": selected_indices,
        "stage_additions": {
            "stage4": selected[:4],
            "stage8": selected[4:8],
        },
        "actual_selected_count": len(selected),
        "planned_selected_count": planned,
        "could_complete_planned_refinement": bool(len(selected) == planned),
        "selection_stages": stage_records,
    }


def _principal_angles(
    space: CombinedSpace,
    left_modes: np.ndarray,
    right_modes: np.ndarray,
    count: int = 3,
) -> list[float]:
    left = np.asarray(left_modes, dtype=float)[:, :count]
    right = np.asarray(right_modes, dtype=float)[:, :count]
    if left.shape[1] == 0 or right.shape[1] == 0:
        return []
    overlap = left.T @ space.gram @ right
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    singular_values = np.clip(np.asarray(singular_values, dtype=float), -1.0, 1.0)
    return [float(math.acos(float(value))) for value in singular_values]


def _monotonicity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for left, right in zip(rows[:-1], rows[1:]):
        left_values = np.asarray(left["internal_eigenvalues_Q_over_W0"], dtype=float)
        right_values = np.asarray(right["internal_eigenvalues_Q_over_W0"], dtype=float)
        count = min(left_values.size, right_values.size)
        upward = np.maximum(right_values[:count] - left_values[:count], 0.0)
        violation = float(np.max(upward)) if count else math.inf
        allowance = max(
            SIGN_RESOLUTION_FLOOR,
            float(left.get("internal_eigen_residual_absolute_max_Q_over_W0", math.inf)),
            float(right.get("internal_eigen_residual_absolute_max_Q_over_W0", math.inf)),
        )
        comparisons.append({
            "from_stage": left.get("stage"),
            "to_stage": right.get("stage"),
            "max_upward_violation_Q_over_W0": violation,
            "allowance_Q_over_W0": allowance,
            "passes": bool(violation <= allowance),
        })
    return {
        "comparisons": comparisons,
        "max_upward_violation_Q_over_W0": max(
            [float(item["max_upward_violation_Q_over_W0"]) for item in comparisons] or [0.0]
        ),
        "numerical_allowance_Q_over_W0": max(
            [float(item["allowance_Q_over_W0"]) for item in comparisons] or [SIGN_RESOLUTION_FLOOR]
        ),
        "monotone_with_allowance": bool(all(item["passes"] for item in comparisons)),
    }


def _holdout_stage(
    space: CombinedSpace,
    current: np.ndarray,
    spectrum: Mapping[str, Any],
    holdout_offset: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    low_count = int(spectrum["low_mode_count"])
    modes = np.asarray(spectrum["mode_coefficients_combined"], dtype=float)[:, :low_count]
    values = np.asarray(spectrum["eigenvalues"], dtype=float)[:low_count]
    residuals = np.asarray(spectrum["residuals"], dtype=float)[:low_count]
    by_type: dict[str, float] = {"density": 0.0, "scalar": 0.0}
    by_region: dict[str, float] = {"interior": 0.0, "edge": 0.0}
    eligible_count = 0
    zero_support_count = 0
    near_dependent_count = 0
    all_normalized: list[float] = []
    for index in range(holdout_offset, len(space.candidates)):
        complement = _candidate_complement(space, index, current)
        record = _candidate_public_record(complement)
        record["dictionary"] = "heldout"
        if complement.get("zero_support") is True:
            zero_support_count += 1
        if complement.get("near_dependent") is True:
            near_dependent_count += 1
        if complement.get("eligible") is True:
            eligible_count += 1
            score = _residual_score(
                space.hessian,
                space.gram,
                complement["normalized_coefficient"],
                values,
                modes,
                residuals=residuals,
            )
            normalized_values = np.asarray(score["normalized_residuals"], dtype=float)
            record.update({
                "residuals_Q_over_W0": np.asarray(score["residuals_Q_over_W0"], dtype=float),
                "normalized_residuals": normalized_values,
                "max_normalized_residual": float(np.max(normalized_values)) if normalized_values.size else 0.0,
                "residual_score": float(score["score"]),
            })
            maximum = float(record["max_normalized_residual"])
            all_normalized.append(maximum)
            by_type[str(complement["kind"])] += maximum**2
            by_region[str(complement["support_region"])] += maximum**2
        else:
            record.update({
                "residuals_Q_over_W0": [],
                "normalized_residuals": [],
                "max_normalized_residual": None,
                "residual_score": None,
            })
        records.append(record)
    maximum = max(all_normalized) if all_normalized else 0.0
    return {
        "dictionary": "heldout",
        "probe_records": records,
        "probe_count": len(records),
        "eligible_count": eligible_count,
        "zero_support_count": zero_support_count,
        "near_dependent_count": near_dependent_count,
        "max_normalized_residual": maximum,
        "normalized_residual_limit": HOLDOUT_RESIDUAL_LIMIT,
        "residual_gate_pass": bool(
            len(records) == len(HOLDOUT_CENTER_FRACTIONS) * 2
            and eligible_count + zero_support_count + near_dependent_count == len(records)
            and all(
                item.get("max_normalized_residual") is not None
                and _finite_scalar(item.get("max_normalized_residual"))
                for item in records if item.get("eligible") is True
            )
            and maximum <= HOLDOUT_RESIDUAL_LIMIT
        ),
        "attribution_sum_squared": {
            "density": float(by_type["density"]),
            "scalar": float(by_type["scalar"]),
            "interior": float(by_region["interior"]),
            "edge": float(by_region["edge"]),
        },
    }


def _stage_row(
    space: CombinedSpace,
    current: np.ndarray,
    stage: int,
    selected_replay: list[dict[str, Any]],
    holdout_offset: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spectrum = _spectrum_modes(space, current)
    values = np.asarray(spectrum["eigenvalues"], dtype=float)
    residuals = np.asarray(spectrum["residuals"], dtype=float)
    holdout = _holdout_stage(space, current, spectrum, holdout_offset)
    row = {
        "stage": int(stage),
        "added_count_planned": int(stage),
        "added_count_actual": int(current.shape[1] - BASE_DIMENSION),
        "basis_dimension": int(current.shape[1]),
        "internal_eigenvalues_Q_over_W0": values[:5],
        "smallest_internal_eigenvalues_Q_over_W0": values[:5],
        "internal_eigen_residual_absolute_Q_over_W0": residuals[:5],
        "internal_eigen_residual_absolute_max_Q_over_W0": float(np.max(residuals)) if residuals.size else math.inf,
        "raw_gram_rank": int(spectrum["raw_rank"]),
        "raw_gram_size": int(spectrum["raw_size"]),
        "internal_gram_rank": int(spectrum["internal_rank"]),
        "internal_gram_size": int(spectrum["internal_size"]),
        "gram_rank_loss": int(spectrum["raw_size"] - spectrum["raw_rank"]),
        "internal_gram_rank_loss": int(spectrum["internal_size"] - spectrum["internal_rank"]),
        "raw_rank_truncated": bool(spectrum["raw_rank"] < spectrum["raw_size"]),
        "internal_rank_truncated": bool(spectrum["internal_rank"] < spectrum["internal_size"]),
        "operator_min_pivot": float(space.diagnostics["operator_min_pivot"]),
        "positive_operator": bool(space.diagnostics["positive_operator"]),
        "projection_applied": bool(space.ell == 1),
        "projection_removed_dimension": 1 if space.ell == 1 else 0,
        "projection_basis_dimension": int(current.shape[1] - (1 if space.ell == 1 else 0)),
        "projection_orthogonality_relative": float(spectrum["projection_orthogonality_relative"]),
        "translation_norm": float(spectrum["translation_norm"]),
        "selected_replay": selected_replay,
        "heldout": holdout,
        "internal_modes_combined": spectrum["mode_coefficients_combined"],
        "internal_values_full": values,
        "internal_residuals_full": residuals,
    }
    return row, spectrum


def _replay_grid(
    background: Any,
    ell: int,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = [str(value) for value in selection.get("selected_candidate_ids", [])]
    selection_specs = [dict(item) for item in selection.get("candidate_specs", [])]
    selected_specs = [
        spec for spec in selection_specs if spec.get("candidate_id") in selected_ids
    ]
    # Holdout candidates are appended after selected candidates.  They are
    # never passed through the training selector.
    holdout_specs = [dict(item) for item in selection.get("holdout_specs", [])]
    all_specs = selected_specs + holdout_specs
    space = _assemble_space(background, int(ell), all_specs)
    current = _identity_coefficients(space)
    selected_records: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    stage_spectra: list[dict[str, Any]] = []
    for position, selected_id in enumerate(selected_ids):
        candidate_index = next(
            (index for index, candidate in enumerate(space.candidates)
             if candidate.spec.get("candidate_id") == selected_id),
            None,
        )
        if candidate_index is None:
            selected_records.append({
                "candidate_id": selected_id,
                "eligible": False,
                "reason": "MISSING_SELECTED_CANDIDATE",
            })
            continue
        complement = _candidate_complement(space, candidate_index, current)
        public = _candidate_public_record(complement)
        public["replay_position"] = int(position)
        selected_records.append(public)
        if complement.get("eligible") is True:
            current = _append_candidate(current, complement["normalized_coefficient"])
        # Record stage 4 and stage 8 after the corresponding additions.  If
        # the training selection had fewer directions, actual counts remain
        # visible and the later gate remains unresolved.
        if position + 1 in (4, 8):
            row, spectrum = _stage_row(
                space,
                current,
                position + 1,
                selected_records.copy(),
                len(selected_specs),
            )
            stage_rows.append(row)
            stage_spectra.append(spectrum)
    # Stage zero always exists and uses all original physical columns.
    stage0_row, stage0_spectrum = _stage_row(
        space,
        _identity_coefficients(space),
        0,
        [],
        len(selected_specs),
    )
    stage_rows.insert(0, stage0_row)
    stage_spectra.insert(0, stage0_spectrum)
    # If a selected stage was unavailable, create a measured row at the
    # largest available extension; the failed planned count is explicit.
    while len(stage_rows) < 3:
        stage = (4, 8)[len(stage_rows) - 1]
        row, spectrum = _stage_row(
            space,
            current,
            stage,
            selected_records.copy(),
            len(selected_specs),
        )
        stage_rows.append(row)
        stage_spectra.append(spectrum)
    monotonicity = _monotonicity(stage_rows)
    principal_angles = {
        "stage0_to_stage4": _principal_angles(
            space,
            stage_spectra[0]["mode_coefficients_combined"],
            stage_spectra[1]["mode_coefficients_combined"],
        ),
        "stage4_to_stage8": _principal_angles(
            space,
            stage_spectra[1]["mode_coefficients_combined"],
            stage_spectra[2]["mode_coefficients_combined"],
        ),
        "stage0_to_stage8": _principal_angles(
            space,
            stage_spectra[0]["mode_coefficients_combined"],
            stage_spectra[2]["mode_coefficients_combined"],
        ),
    }
    values4 = np.asarray(stage_rows[1]["internal_eigenvalues_Q_over_W0"], dtype=float)
    values8 = np.asarray(stage_rows[2]["internal_eigenvalues_Q_over_W0"], dtype=float)
    count = min(values4.size, values8.size)
    drift = float(
        np.max(np.abs(values8[:count] - values4[:count])
               / np.maximum(np.abs(values8[:count]), SIGN_RESOLUTION_FLOOR))
        if count else math.inf
    )
    stage8 = stage_rows[2]
    selected_complete = bool(
        len(selected_ids) == int(selection.get("planned_selected_count", 8))
        and len(selected_records) == len(selected_ids)
        and all(item.get("eligible") is True for item in selected_records)
    )
    stage_controls_pass = bool(
        space.diagnostics["base_representation_reliable"]
        and space.diagnostics["positive_operator"]
        and selected_complete
        and all(
            row["positive_operator"] is True
            and row["internal_gram_rank_loss"] == 0
            and row["gram_rank_loss"] == 0
            and _finite_scalar(row["internal_eigen_residual_absolute_max_Q_over_W0"])
            and row["projection_orthogonality_relative"] <= PROJECTION_ORTHOGONALITY_LIMIT
            for row in stage_rows
        )
        and monotonicity["monotone_with_allowance"]
        and drift <= BASIS_RELATIVE_DRIFT_LIMIT
    )
    return {
        "ell": int(ell),
        "intervals": int(background.x.size - 1),
        "box_fm": float(background.box_fm),
        "background_edge_x": float(background.edge_x),
        "stage_rows": stage_rows,
        "rayleigh_ritz_monotonicity": monotonicity,
        "principal_angles_first3_radians": principal_angles,
        "last4_to8_relative_drift_max": drift,
        "selected_replay_complete": selected_complete,
        "heldout_terminal_max_normalized_residual": stage8["heldout"].get("max_normalized_residual"),
        "base_physical_space": _jsonable(space.diagnostics),
        "stage_controls_pass": stage_controls_pass,
        "internal_modes_for_validation": [
            spectrum["mode_coefficients_combined"] for spectrum in stage_spectra
        ],
    }


def _background_summary_map(backgrounds: Sequence[Any]) -> dict[str, Any]:
    if len(backgrounds) != 3:
        raise ResidualEnrichmentError("expected 24 fm, 32 fm and tighter 24 fm backgrounds")
    return {
        "24fm": backgrounds[0].summary(),
        "32fm": backgrounds[1].summary(),
        "24fm_tighter": backgrounds[2].summary(),
        "background_grid_intervals": 3200,
        "background_nodes_requested": stability.BACKGROUND_NODES,
        "background_tolerance": stability.BACKGROUND_TOLERANCE,
        "background_tight_nodes_requested": stability.BACKGROUND_TIGHT_NODES,
        "background_tight_tolerance": stability.BACKGROUND_TIGHT_TOLERANCE,
    }


def _source_case_controls_pass(source_case: Mapping[str, Any], target: str) -> bool:
    """Recompute inherited source guards rather than trusting its booleans."""

    try:
        if (
            source_case.get("target_y") != target
            or source_case.get("target_N") != TARGET_N
            or source_case.get("all_planned_ell_sectors_reported") is not True
            or source_case.get("all_spectrum_records_available") is not True
        ):
            return False
        backgrounds = source_case.get("backgrounds")
        if not isinstance(backgrounds, Mapping):
            return False
        for key in VALIDATION_BACKGROUND_KEYS:
            item = backgrounds.get(key)
            if not isinstance(item, Mapping) or item.get("converged") is not True:
                return False
            for field in (
                "chemical_potential_MeV", "conserved_N_relative_error",
                "field_residual_relative", "virial_relative",
                "gauss_energy_identity_relative", "edge_x",
            ):
                if not _finite_scalar(item.get(field)):
                    return False
        sectors = source_case.get("sectors")
        if not isinstance(sectors, Mapping) or set(sectors) != {str(ell) for ell in ELL_VALUES}:
            return False
        for ell in ELL_VALUES:
            sector = sectors.get(str(ell))
            if not stability._spectrum_controls_pass(sector, ell):
                return False
        fd_checks = source_case.get("independent_energy_difference_checks_24fm")
        if not isinstance(fd_checks, list) or len(fd_checks) != len(stability.FD_DIRECTION_NAMES):
            return False
        if not all(
            stability._fd_check_pass(check, name)
            for check, name in zip(fd_checks, stability.FD_DIRECTION_NAMES)
        ):
            return False
        if not stability._translation_controls_pass(source_case.get("translation_metrics_24fm")):
            return False
        checks = source_case.get("terminal_consistency_checks")
        if not isinstance(checks, Mapping):
            return False
        fixed = checks.get("fixed_N_additive_paths")
        gauss = checks.get("global_Gauss_re_solves")
        scalar = checks.get("scalar_only_energy_difference")
        if not all(isinstance(item, Mapping) for item in (fixed, gauss, scalar)):
            return False
        return bool(
            fixed.get("all_checks_pass") is True
            and gauss.get("all_checks_pass") is True
            and scalar.get("available") is True
            and scalar.get("agreement") is True
            and source_case.get("terminal_control_failures") == []
            and source_case.get("terminal_controls_pass") is True
        )
    except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _source_summary(source_case: Mapping[str, Any], target: str) -> dict[str, Any]:
    canonical = json.dumps(_jsonable(source_case), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "target_y": target,
        "schema_version": source_case.get("schema_version", stability.SCHEMA),
        "status": source_case.get("status", stability.STATUS),
        "terminal_controls_pass": source_case.get("terminal_controls_pass"),
        "terminal_control_failures": source_case.get("terminal_control_failures"),
        "aggregate_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _sector_result(
    target: str,
    ell: int,
    backgrounds: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for background_key in VALIDATION_BACKGROUND_KEYS:
        background = backgrounds[background_key]
        for intervals in GRID_INTERVALS:
            grid_background = stability.Background.from_solution(
                background.design,
                background.box_fm,
                background.solution,
                background.solver_row,
                int(intervals),
            )
            replay = _replay_grid(grid_background, int(ell), selection)
            replay["target_y"] = target
            replay["background_key"] = background_key
            replay["background_summary"] = grid_background.summary()
            rows.append(replay)
    expected_row_count = len(VALIDATION_BACKGROUND_KEYS) * len(GRID_INTERVALS)
    complete_coverage = bool(
        len(rows) == expected_row_count
        and sorted((row.get("background_key"), row.get("intervals")) for row in rows)
        == sorted((key, intervals) for key in VALIDATION_BACKGROUND_KEYS for intervals in GRID_INTERVALS)
    )
    by_key = {(row["background_key"], row["intervals"]): row for row in rows}
    terminal = by_key.get(("32fm", max(GRID_INTERVALS)))
    if terminal is None:
        terminal = rows[-1] if rows else {}
    # The same stage is compared across measured grids/domains/backgrounds.
    stage8_values = {
        (row.get("background_key"), row.get("intervals")): np.asarray(
            row.get("stage_rows", [{}, {}, {}])[2].get("internal_eigenvalues_Q_over_W0", []), dtype=float
        )
        for row in rows if len(row.get("stage_rows", [])) == 3
    }
    grid_deltas: list[float] = []
    for key in VALIDATION_BACKGROUND_KEYS:
        for left_intervals, right_intervals in zip(GRID_INTERVALS[:-1], GRID_INTERVALS[1:]):
            left = stage8_values.get((key, left_intervals), np.asarray([], dtype=float))
            right = stage8_values.get((key, right_intervals), np.asarray([], dtype=float))
            count = min(left.size, right.size)
            grid_deltas.append(float(np.max(np.abs(right[:count] - left[:count]))) if count else math.inf)
    domain_delta = math.inf
    background_delta = math.inf
    if ("24fm", max(GRID_INTERVALS)) in stage8_values and ("32fm", max(GRID_INTERVALS)) in stage8_values:
        left = stage8_values[("24fm", max(GRID_INTERVALS))]
        right = stage8_values[("32fm", max(GRID_INTERVALS))]
        count = min(left.size, right.size)
        domain_delta = float(np.max(np.abs(right[:count] - left[:count]))) if count else math.inf
    if ("24fm", max(GRID_INTERVALS)) in stage8_values and ("24fm_tighter", max(GRID_INTERVALS)) in stage8_values:
        left = stage8_values[("24fm", max(GRID_INTERVALS))]
        right = stage8_values[("24fm_tighter", max(GRID_INTERVALS))]
        count = min(left.size, right.size)
        background_delta = float(np.max(np.abs(right[:count] - left[:count]))) if count else math.inf
    terminal_values = np.asarray(
        terminal.get("stage_rows", [{}, {}, {}])[2].get("internal_eigenvalues_Q_over_W0", []),
        dtype=float,
    )
    uncertainty_terms: list[float] = [SIGN_RESOLUTION_FLOOR]
    uncertainty_terms.extend(grid_deltas)
    uncertainty_terms.extend([domain_delta, background_delta])
    uncertainty_terms.extend(
        float(row.get("stage_rows", [{}, {}, {}])[2].get("internal_eigen_residual_absolute_max_Q_over_W0", math.inf))
        for row in rows if len(row.get("stage_rows", [])) == 3
    )
    uncertainty = max(uncertainty_terms) if uncertainty_terms else math.inf
    signs = [
        "positive" if value > 0.0 else "negative" if value < 0.0 else "zero"
        for value in terminal_values
    ]
    margins = [
        abs(float(value)) / (SIGN_ERROR_MULTIPLIER * uncertainty)
        for value in terminal_values
    ]
    sign_resolved = [margin >= 1.0 for margin in margins]
    convergence = bool(
        complete_coverage
        and all(row.get("stage_controls_pass") is True for row in rows)
        and all(row.get("last4_to8_relative_drift_max", math.inf) <= BASIS_RELATIVE_DRIFT_LIMIT for row in rows)
        and max(grid_deltas or [math.inf]) / max(float(np.max(np.maximum(np.abs(terminal_values), SIGN_RESOLUTION_FLOOR))) if terminal_values.size else 1.0, SIGN_RESOLUTION_FLOOR) <= DISCRETIZATION_RELATIVE_DRIFT_LIMIT
        and domain_delta / max(float(np.max(np.maximum(np.abs(terminal_values), SIGN_RESOLUTION_FLOOR))) if terminal_values.size else 1.0, SIGN_RESOLUTION_FLOOR) <= DISCRETIZATION_RELATIVE_DRIFT_LIMIT
        and background_delta / max(float(np.max(np.maximum(np.abs(terminal_values), SIGN_RESOLUTION_FLOOR))) if terminal_values.size else 1.0, SIGN_RESOLUTION_FLOOR) <= DISCRETIZATION_RELATIVE_DRIFT_LIMIT
        and bool(terminal.get("stage_rows", [{}, {}, {}])[2].get("heldout", {}).get("residual_gate_pass") is True)
        and bool(terminal.get("rayleigh_ritz_monotonicity", {}).get("monotone_with_allowance") is True)
    )
    robust_negative = bool(
        convergence and any(sign == "negative" and resolved for sign, resolved in zip(signs, sign_resolved))
    )
    robust_positive = bool(
        convergence and terminal_values.size > 0 and all(sign == "positive" and resolved for sign, resolved in zip(signs, sign_resolved))
    )
    if robust_negative:
        conclusion = "FINITE_ENRICHED_BASIS_ROBUST_NEGATIVE_DIRECTION"
    elif robust_positive:
        conclusion = "FINITE_ENRICHED_BASIS_POSITIVE_ONLY"
    else:
        conclusion = "UNRESOLVED_FINITE_ENRICHED_SPACE_SIGN_OR_CONVERGENCE"
    stage_conclusions: dict[str, str] = {}
    for index, stage in enumerate((0, 4, 8)):
        values = np.asarray(terminal.get("stage_rows", [{}, {}, {}])[index].get("internal_eigenvalues_Q_over_W0", []), dtype=float)
        if values.size and np.all(values > 0.0):
            stage_conclusions[f"stage{stage}"] = "POSITIVE_FINITE_STAGE_DIAGNOSTIC"
        elif values.size and np.any(values < 0.0):
            stage_conclusions[f"stage{stage}"] = "NEGATIVE_FINITE_STAGE_DIAGNOSTIC"
        else:
            stage_conclusions[f"stage{stage}"] = "UNRESOLVED_FINITE_STAGE_DIAGNOSTIC"
    terminal_holdout = terminal.get("stage_rows", [{}, {}, {}])[2].get("heldout", {})
    control_failures: list[str] = []
    if not complete_coverage:
        control_failures.append("validation_coverage")
    if not all(row.get("stage_controls_pass") is True for row in rows):
        control_failures.append("physical_or_refinement_controls")
    if not all(row.get("selected_replay_complete") is True for row in rows):
        control_failures.append("selected_candidate_replay")
    if not terminal_holdout.get("residual_gate_pass") is True:
        control_failures.append("heldout_residual_gate")
    return {
        "target_y": target,
        "ell": int(ell),
        "training_grid": {
            "box_fm": TRAINING_BOX_FM,
            "intervals": TRAINING_INTERVALS,
            "edge_x": selection.get("training_edge_x"),
        },
        "source_sector_conclusion": selection.get("source_sector_conclusion"),
        "selection": selection,
        "validation_rows": rows,
        "expected_validation_row_count": expected_row_count,
        "validation_row_count": len(rows),
        "all_validation_rows_reported": complete_coverage,
        "terminal_validation": terminal,
        "grid_delta_max_Q_over_W0": max(grid_deltas or [math.inf]),
        "domain_delta_max_Q_over_W0": domain_delta,
        "tighter_background_delta_max_Q_over_W0": background_delta,
        "measured_uncertainty_conservative_base": uncertainty,
        "terminal_eigenvalues_Q_over_W0": terminal_values,
        "terminal_signs": signs,
        "terminal_sign_margins": margins,
        "terminal_sign_resolved": sign_resolved,
        "basis_and_grid_domain_convergence": convergence,
        "heldout_terminal_max_normalized_residual": terminal_holdout.get("max_normalized_residual"),
        "heldout_terminal_gate_pass": terminal_holdout.get("residual_gate_pass"),
        "rayleigh_ritz_all_rows_pass": bool(all(
            row.get("rayleigh_ritz_monotonicity", {}).get("monotone_with_allowance") is True
            for row in rows
        )),
        "stage_conclusions": stage_conclusions,
        "conclusion": conclusion,
        "terminal_controls_pass": bool(not control_failures),
        "terminal_control_failures": control_failures,
    }


def _case_result(target: str) -> dict[str, Any]:
    source_case = stability._case_result(target)
    source_controls = _source_case_controls_pass(source_case, target)
    background_tuple = stability._backgrounds(target)
    background_map = dict(zip(VALIDATION_BACKGROUND_KEYS, background_tuple))
    sectors: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for ell in ELL_VALUES:
        training_background = stability.Background.from_solution(
            background_tuple[0].design,
            TRAINING_BOX_FM,
            background_tuple[0].solution,
            background_tuple[0].solver_row,
            TRAINING_INTERVALS,
        )
        edge = float(training_background.edge_x)
        training_specs = _candidate_specs(edge=edge)
        training_space = _assemble_space(training_background, int(ell), training_specs)
        selection = _select_training_candidates(training_space)
        source_sector = source_case.get("sectors", {}).get(str(ell), {})
        selection.update({
            "target_y": target,
            "ell": int(ell),
            "training_edge_x": edge,
            "training_box_fm": TRAINING_BOX_FM,
            "training_intervals": TRAINING_INTERVALS,
            "candidate_dictionary": "b(z)=(1-z^2)^4 for |z|<1; analytic b',b''",
            "candidate_specs": training_specs,
            "holdout_specs": _candidate_specs(heldout=True, edge=edge),
            "heldout_never_used_for_selection": True,
            "source_sector_conclusion": source_sector.get("conclusion"),
            "base_physical_space": _jsonable(training_space.diagnostics),
        })
        selections[str(ell)] = selection
        sectors[str(ell)] = _sector_result(target, int(ell), background_map, selection)
    global_failures: list[str] = []
    if not source_controls:
        global_failures.append("inherited_source_controls")
    if not all(sector.get("terminal_controls_pass") is True for sector in sectors.values()):
        global_failures.append("residual_enrichment_controls")
    if set(sectors) != {str(ell) for ell in ELL_VALUES}:
        global_failures.append("ell_sector_coverage")
    if not all(
        selection.get("could_complete_planned_refinement") is True
        for selection in selections.values()
    ):
        global_failures.append("planned_refinement_count")
    # The actual measured statuses are recomputed here.  A stale success flag
    # in any child row can therefore never create a positive or negative label.
    if global_failures:
        for sector in sectors.values():
            sector["conclusion"] = "UNRESOLVED_FINITE_ENRICHED_SPACE_SIGN_OR_CONVERGENCE"
    background_summaries = _background_summary_map(background_tuple)
    return {
        "target_y": target,
        "target_N": TARGET_N,
        "backgrounds": background_summaries,
        "source_case": _source_summary(source_case, target),
        "source_case_controls_pass": bool(source_controls),
        "selections": selections,
        "sectors": sectors,
        "terminal_controls_pass": bool(not global_failures),
        "terminal_control_failures": global_failures,
        "all_planned_ell_sectors_reported": bool(set(sectors) == {str(ell) for ell in ELL_VALUES}),
        "all_validation_records_available": bool(all(
            sector.get("all_validation_rows_reported") is True for sector in sectors.values()
        )),
        "scope_conclusion": "FINITE_BASIS_RESIDUAL_ENRICHMENT_ONLY",
    }


def _synthetic_controls() -> dict[str, Any]:
    """Independent controls for whitening, quotient, degeneracy and hidden directions."""

    controls: dict[str, Any] = {}
    try:
        # A weakly represented negative coordinate is recovered by physical
        # equilibration; no Gram cutoff is allowed to delete it.
        gram = np.diag((1.0, 1.0e-12))
        hessian = np.diag((1.0, -1.0e-12))
        transform = np.diag((1.0, 1.0e6))
        h_white = transform.T @ hessian @ transform
        g_white = transform.T @ gram @ transform
        weak_values = np.linalg.eigvalsh(np.linalg.solve(g_white, h_white))
        controls["weak_negative_whitening"] = {
            "eigenvalues": weak_values,
            "negative_recovered": bool(np.min(weak_values) < -0.9),
            "all_columns_retained": True,
        }
        # Equal low eigenvalues: rotate the included eigenspace and preserve
        # the residual sum exactly.
        low_hessian = np.diag((2.0, 2.0, 4.0))
        low_gram = np.eye(3)
        candidate = np.array((0.3, -0.4, 0.5))
        modes = np.eye(3)[:, :2]
        score_a = _residual_score(low_hessian, low_gram, candidate, (2.0, 2.0), modes)
        rotation = np.array(((0.0, 1.0), (1.0, 0.0)))
        score_b = _residual_score(
            low_hessian,
            low_gram,
            candidate,
            (2.0, 2.0),
            np.column_stack((modes[:, :2] @ rotation,)),
        )
        controls["degenerate_score_invariance"] = {
            "score_a": score_a["score"],
            "score_b": score_b["score"],
            "passes": bool(abs(score_a["score"] - score_b["score"]) <= 1.0e-12),
        }
        # Non-orthogonal translation quotient keeps a distinct negative mode.
        nonorthogonal_gram = np.array(((2.0, 0.3), (0.3, 1.0)))
        nonorthogonal_hessian = np.diag((0.4, -0.7))
        translation = np.array((1.0, 0.0))
        row = (translation @ nonorthogonal_gram)[None, :]
        _, _, vh = np.linalg.svd(row, full_matrices=True)
        projector = vh[1:].T
        quotient = float(
            (projector[:, 0] @ nonorthogonal_hessian @ projector[:, 0])
            / (projector[:, 0] @ nonorthogonal_gram @ projector[:, 0])
        )
        controls["nonorthogonal_translation_quotient"] = {
            "raw_minimum": float(np.linalg.eigvalsh(np.linalg.solve(nonorthogonal_gram, nonorthogonal_hessian))[0]),
            "quotient_rayleigh": quotient,
            "translation_removed_exactly_one": bool(projector.shape[1] == 1),
            "negative_preserved": bool(quotient < 0.0),
        }
        # A decoupled held-out negative direction has zero low-mode residual;
        # enrichment still exposes the negative finite-space curvature.
        hidden_hessian = np.diag((1.0, -1.0))
        hidden_gram = np.eye(2)
        hidden_score = _residual_score(
            hidden_hessian[:1, :1], hidden_gram[:1, :1], np.array((1.0,)), (1.0,), np.array((1.0,))[:, None]
        )
        enriched_values = np.linalg.eigvalsh(hidden_hessian)
        controls["hidden_outside_space_negative"] = {
            "old_low_mode_residual_score": hidden_score["score"],
            "enriched_eigenvalues": enriched_values,
            "zero_residual_does_not_prove_completeness": bool(hidden_score["score"] == 0.0),
            "negative_exposed_after_enrichment": bool(np.min(enriched_values) < 0.0),
        }
        # Nested spaces have nonincreasing ordered Ritz values.
        nested = [
            np.linalg.eigvalsh(np.diag((1.0, 3.0))),
            np.linalg.eigvalsh(np.diag((0.8, 1.0, 3.0))),
        ]
        controls["nested_ritz_monotonicity"] = {
            "passes": bool(np.all(nested[1][:2] <= nested[0] + 1.0e-14)),
            "stage_values": nested,
        }
        controls["all_checks_pass"] = bool(
            controls["weak_negative_whitening"]["negative_recovered"]
            and controls["degenerate_score_invariance"]["passes"]
            and controls["nonorthogonal_translation_quotient"]["translation_removed_exactly_one"]
            and controls["nonorthogonal_translation_quotient"]["negative_preserved"]
            and controls["hidden_outside_space_negative"]["zero_residual_does_not_prove_completeness"]
            and controls["hidden_outside_space_negative"]["negative_exposed_after_enrichment"]
            and controls["nested_ritz_monotonicity"]["passes"]
        )
    except (ArithmeticError, ResidualEnrichmentError, TypeError, ValueError):
        controls["all_checks_pass"] = False
    return _jsonable(controls)


def calculate() -> dict[str, Any]:
    cases = {target: _case_result(target) for target in TARGETS}
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "model_scope": {
            "zero_temperature_symmetric_coulomb_free": True,
            "accepted_backgrounds_only": True,
            "target_N": TARGET_N,
            "target_y_values": list(TARGETS),
            "ell_sectors": list(ELL_VALUES),
            "base_columns_each_block": BASE_BLOCK_SIZE,
            "base_raw_dimension": BASE_DIMENSION,
            "training_box_fm": TRAINING_BOX_FM,
            "training_intervals": TRAINING_INTERVALS,
            "validation_intervals": list(GRID_INTERVALS),
            "validation_domains_fm": list(DOMAIN_BOXES_FM),
            "validation_backgrounds": list(VALIDATION_BACKGROUND_KEYS),
            "enrichment_stages_added": list(STAGE_COUNTS),
            "curvatures_are_not_frequencies": True,
            "positive_finite_enriched_space_is_not_full_stability": True,
            "no_physics_or_parameter_change": True,
            "empirical_weight": 0.0,
        },
        "dictionary_contract": {
            "training_center_fractions": list(TRAINING_CENTER_FRACTIONS),
            "training_width_fractions": list(TRAINING_WIDTH_FRACTIONS),
            "heldout_center_fractions": list(HOLDOUT_CENTER_FRACTIONS),
            "heldout_width_fraction": HOLDOUT_WIDTH_FRACTION,
            "bump": "b=(1-z^2)^4 for |z|<1; zero otherwise",
            "b_prime": "-8*z*(1-z^2)^3/width",
            "b_second": "(1-z^2)^2*(56*z^2-8)/width^2",
            "density_direction": "existing continuity displacement derivative including n'",
            "scalar_direction": "v=b, vprime=b'",
            "fixed_number_projection": "existing independent ell0 edge-aware projection; no density floor",
            "heldout_never_used_for_selection": True,
        },
        "acceptance_limits": {
            "inherited_gram_relative_cutoff": GRAM_RELATIVE_CUTOFF,
            "inherited_sign_error_multiplier": SIGN_ERROR_MULTIPLIER,
            "inherited_sign_resolution_floor": SIGN_RESOLUTION_FLOOR,
            "inherited_basis_relative_drift": BASIS_RELATIVE_DRIFT_LIMIT,
            "inherited_discretization_relative_drift": DISCRETIZATION_RELATIVE_DRIFT_LIMIT,
            "inherited_number_constraint_relative": NUMBER_CONSTRAINT_RELATIVE_LIMIT,
            "inherited_projection_orthogonality_relative": PROJECTION_ORTHOGONALITY_LIMIT,
            "candidate_relative_rank_resolution": CANDIDATE_RELATIVE_RANK,
            "heldout_normalized_residual_limit": HOLDOUT_RESIDUAL_LIMIT,
            "physical_reconstruction_relative": PHYSICAL_RECONSTRUCTION_LIMIT,
            "physical_orthogonality_relative": PHYSICAL_ORTHOGONALITY_LIMIT,
            "physical_rayleigh_consistency_relative": PHYSICAL_RAYLEIGH_LIMIT,
        },
        "hessian_contract": {
            "source_stability_schema": stability.SCHEMA,
            "formula": "Q_l/W0=int x^2[A u^2+2Buv+C v^2+v'^2+l(l+1)v^2/x^2]+<s,K_l^-1s>",
            "physical_norm": "int x^2[(u/n0_dim)^2+v^2]dx",
            "candidate_score": "sum_i |H(z,phi_i)-lambda_i G(z,phi_i)|^2/G(z,z)",
            "translation_quotient": "ell1 removes exactly the known joint translation through physical G-orthogonality",
            "operator_solver": "existing positive O(N) tridiagonal Gauss solve",
        },
        "input_provenance": {
            "stability_source": "verification/nvg_droplet_stability_audit.py",
            "finite_droplet_source": "verification/nvg_finite_droplet_audit.py",
            "stability_source_sha256": hashlib.sha256(
                (HERE / "nvg_droplet_stability_audit.py").read_bytes()
            ).hexdigest(),
            "finite_droplet_source_sha256": hashlib.sha256(
                (HERE / "nvg_finite_droplet_audit.py").read_bytes()
            ).hexdigest(),
            "no_saved_result_used_as_input": True,
            "old_sources_read_only": True,
            "source_case_aggregate_sha256": {
                target: cases[target]["source_case"]["aggregate_sha256"] for target in TARGETS
            },
        },
        "synthetic_controls": _synthetic_controls(),
        "cases": cases,
    }


_CACHE: dict[str, Any] | None = None


def build_result() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _jsonable(calculate())
    return copy.deepcopy(_CACHE)


def _validate_stage_row(row: Mapping[str, Any], ell: int) -> bool:
    if not isinstance(row, Mapping):
        return False
    if (
        row.get("stage") not in STAGE_COUNTS
        or not isinstance(row.get("basis_dimension"), int)
        or row.get("basis_dimension") < BASE_DIMENSION
        or row.get("projection_applied") is not (ell == 1)
        or row.get("projection_removed_dimension") != (1 if ell == 1 else 0)
        or row.get("projection_basis_dimension") != row.get("basis_dimension") - (1 if ell == 1 else 0)
        or row.get("positive_operator") is not True
        or not _finite_scalar(row.get("operator_min_pivot"))
        or float(row.get("operator_min_pivot")) <= 0.0
    ):
        return False
    for key in (
        "internal_eigenvalues_Q_over_W0",
        "internal_eigen_residual_absolute_Q_over_W0",
    ):
        values = row.get(key)
        if not isinstance(values, (list, tuple, np.ndarray)) or not len(values) or not all(_finite_scalar(value) for value in values):
            return False
    for key in (
        "internal_eigen_residual_absolute_max_Q_over_W0",
        "projection_orthogonality_relative",
        "translation_norm",
    ):
        if not _finite_scalar(row.get(key)):
            return False
    values = row.get("internal_eigenvalues_Q_over_W0")
    smallest = row.get("smallest_internal_eigenvalues_Q_over_W0")
    if not isinstance(smallest, (list, tuple, np.ndarray)) or list(smallest) != list(values):
        return False
    residuals = row.get("internal_eigen_residual_absolute_Q_over_W0")
    if abs(
        float(row["internal_eigen_residual_absolute_max_Q_over_W0"])
        - max(map(float, residuals), default=0.0)
    ) > 1.0e-12 * max(1.0, abs(float(row["internal_eigen_residual_absolute_max_Q_over_W0"]))):
        return False
    if row.get("projection_orthogonality_relative", math.inf) > PROJECTION_ORTHOGONALITY_LIMIT:
        return False
    if not isinstance(row.get("heldout"), Mapping):
        return False
    return _finite_tree(row)


def _validate_selection(selection: Mapping[str, Any], ell: int) -> bool:
    if not isinstance(selection, Mapping):
        return False
    specs = selection.get("candidate_specs")
    holdout = selection.get("holdout_specs")
    selected = selection.get("selected_candidate_ids")
    if (
        not isinstance(specs, list)
        or len(specs) != len(TRAINING_CENTER_FRACTIONS) * len(TRAINING_WIDTH_FRACTIONS) * 2
        or not isinstance(holdout, list)
        or len(holdout) != len(HOLDOUT_CENTER_FRACTIONS) * 2
        or not isinstance(selected, list)
        or len(selected) != len(set(selected))
        or not all(item in {spec.get("candidate_id") for spec in specs} for item in selected)
        or any(item in {spec.get("candidate_id") for spec in holdout} for item in selected)
        or selection.get("heldout_never_used_for_selection") is not True
        or selection.get("ell") != ell
        or not _finite_scalar(selection.get("training_edge_x"))
    ):
        return False
    additions = selection.get("stage_additions")
    if not isinstance(additions, Mapping) or additions.get("stage4") != selected[:4] or additions.get("stage8") != selected[4:8]:
        return False
    stages = selection.get("selection_stages")
    if not isinstance(stages, list) or len(stages) != 2:
        return False
    return _finite_tree(selection)


def _validate_replay_row(row: Mapping[str, Any], ell: int) -> bool:
    if (
        not isinstance(row, Mapping)
        or row.get("ell") != ell
        or row.get("background_key") not in VALIDATION_BACKGROUND_KEYS
        or row.get("intervals") not in GRID_INTERVALS
        or row.get("box_fm") not in DOMAIN_BOXES_FM
        or not isinstance(row.get("stage_rows"), list)
        or len(row.get("stage_rows")) != 3
        or [item.get("stage") for item in row["stage_rows"]] != list(STAGE_COUNTS)
        or not _finite_scalar(row.get("heldout_terminal_max_normalized_residual"))
    ):
        return False
    if not all(_validate_stage_row(item, ell) for item in row["stage_rows"]):
        return False
    monotonicity = row.get("rayleigh_ritz_monotonicity")
    if (
        not isinstance(monotonicity, Mapping)
        or monotonicity.get("monotone_with_allowance") is not True
        or not isinstance(monotonicity.get("comparisons"), list)
        or len(monotonicity["comparisons"]) != 2
        or not all(item.get("passes") is True for item in monotonicity["comparisons"])
    ):
        return False
    if row.get("selected_replay_complete") is not True:
        return False
    heldout = row["stage_rows"][2].get("heldout")
    if (
        not isinstance(heldout, Mapping)
        or heldout.get("probe_count") != len(HOLDOUT_CENTER_FRACTIONS) * 2
        or not isinstance(heldout.get("residual_gate_pass"), bool)
        or not _finite_scalar(heldout.get("max_normalized_residual"))
    ):
        return False
    records = heldout.get("probe_records")
    if not isinstance(records, list) or len(records) != heldout.get("probe_count"):
        return False
    finite_maxima = [
        float(item.get("max_normalized_residual"))
        for item in records
        if isinstance(item, Mapping)
        and item.get("eligible") is True
        and _finite_scalar(item.get("max_normalized_residual"))
    ]
    expected_maximum = max(finite_maxima or [0.0])
    expected_gate = bool(
        len(records) == len(HOLDOUT_CENTER_FRACTIONS) * 2
        and all(isinstance(item, Mapping) for item in records)
        and all(
            item.get("eligible") is True
            or item.get("zero_support") is True
            or item.get("near_dependent") is True
            for item in records
        )
        and expected_maximum <= HOLDOUT_RESIDUAL_LIMIT
    )
    if (
        abs(float(heldout.get("max_normalized_residual")) - expected_maximum)
        > 1.0e-12 * max(1.0, expected_maximum)
        or heldout.get("residual_gate_pass") is not expected_gate
        or abs(
            float(row.get("heldout_terminal_max_normalized_residual"))
            - float(heldout.get("max_normalized_residual"))
        ) > 1.0e-12 * max(1.0, expected_maximum)
    ):
        return False
    return _finite_tree(row)


def _recomputed_monotonicity(row: Mapping[str, Any]) -> dict[str, Any] | None:
    stages = row.get("stage_rows") if isinstance(row, Mapping) else None
    if not isinstance(stages, list) or len(stages) != 3:
        return None
    try:
        return _monotonicity(stages)
    except (ArithmeticError, TypeError, ValueError):
        return None


def _mapping_numeric_close(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)) and not isinstance(a, bool) and not isinstance(b, bool):
            if abs(float(a) - float(b)) > 1.0e-12 * max(1.0, abs(float(a)), abs(float(b))):
                return False
        elif a != b:
            return False
    return True


def _monotonicity_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        return False
    if actual.get("monotone_with_allowance") is not expected.get("monotone_with_allowance"):
        return False
    for key in ("max_upward_violation_Q_over_W0", "numerical_allowance_Q_over_W0"):
        if not _finite_scalar(actual.get(key)) or not _finite_scalar(expected.get(key)):
            return False
        if abs(float(actual[key]) - float(expected[key])) > 1.0e-12 * max(1.0, abs(float(expected[key]))):
            return False
    actual_items, expected_items = actual.get("comparisons"), expected.get("comparisons")
    if not isinstance(actual_items, list) or not isinstance(expected_items, list) or len(actual_items) != len(expected_items):
        return False
    return all(_mapping_numeric_close(a, b) for a, b in zip(actual_items, expected_items))


def _recomputed_holdout_gate(heldout: Mapping[str, Any]) -> tuple[float, bool] | None:
    if not isinstance(heldout, Mapping):
        return None
    records = heldout.get("probe_records")
    if not isinstance(records, list):
        return None
    maxima: list[float] = []
    for item in records:
        if not isinstance(item, Mapping):
            return None
        if item.get("eligible") is True:
            if not _finite_scalar(item.get("max_normalized_residual")):
                return None
            maxima.append(float(item["max_normalized_residual"]))
        elif item.get("zero_support") is not True and item.get("near_dependent") is not True:
            return None
    maximum = max(maxima or [0.0])
    complete = bool(
        len(records) == len(HOLDOUT_CENTER_FRACTIONS) * 2
        and all(
            item.get("eligible") is True
            or item.get("zero_support") is True
            or item.get("near_dependent") is True
            for item in records
        )
    )
    return maximum, bool(complete and maximum <= HOLDOUT_RESIDUAL_LIMIT)


def _recomputed_replay_core_pass(row: Mapping[str, Any], ell: int) -> bool:
    """Recompute physical/refinement guards from measured replay fields."""

    if not isinstance(row, Mapping) or not _validate_replay_row(row, ell):
        return False
    stages = row["stage_rows"]
    expected_monotonicity = _recomputed_monotonicity(row)
    if expected_monotonicity is None or not _monotonicity_matches(row.get("rayleigh_ritz_monotonicity"), expected_monotonicity):
        return False
    if not row.get("selected_replay_complete") is True:
        return False
    drift = float(row.get("last4_to8_relative_drift_max", math.inf))
    if not _finite_scalar(drift) or drift > BASIS_RELATIVE_DRIFT_LIMIT:
        return False
    base = row.get("base_physical_space")
    if not isinstance(base, Mapping) or base.get("base_representation_reliable") is not True:
        return False
    expected_stage_pass = bool(
        all(item.get("positive_operator") is True for item in stages)
        and all(item.get("internal_gram_rank_loss") == 0 for item in stages)
        and all(item.get("gram_rank_loss") == 0 for item in stages)
        and all(
            _finite_scalar(item.get("internal_eigen_residual_absolute_max_Q_over_W0"))
            for item in stages
        )
        and all(
            item.get("projection_orthogonality_relative", math.inf)
            <= PROJECTION_ORTHOGONALITY_LIMIT
            for item in stages
        )
        and row.get("selected_replay_complete") is True
        and expected_monotonicity.get("monotone_with_allowance") is True
        and drift <= BASIS_RELATIVE_DRIFT_LIMIT
    )
    return row.get("stage_controls_pass") is expected_stage_pass


def _recomputed_sector_failures(sector: Mapping[str, Any], ell: int) -> list[str] | None:
    if not isinstance(sector, Mapping):
        return None
    rows = sector.get("validation_rows")
    expected_count = len(VALIDATION_BACKGROUND_KEYS) * len(GRID_INTERVALS)
    failures: list[str] = []
    if not isinstance(rows, list) or len(rows) != expected_count:
        failures.append("validation_coverage")
        return failures
    expected_keys = sorted(
        (key, intervals)
        for key in VALIDATION_BACKGROUND_KEYS for intervals in GRID_INTERVALS
    )
    actual_keys = sorted((row.get("background_key"), row.get("intervals")) for row in rows if isinstance(row, Mapping))
    if actual_keys != expected_keys:
        failures.append("validation_coverage")
    if not all(_recomputed_replay_core_pass(row, ell) for row in rows):
        failures.append("physical_or_refinement_controls")
    if not all(row.get("selected_replay_complete") is True for row in rows if isinstance(row, Mapping)):
        failures.append("selected_candidate_replay")
    terminal = sector.get("terminal_validation")
    terminal_holdout = (
        terminal.get("stage_rows", [{}, {}, {}])[2].get("heldout")
        if isinstance(terminal, Mapping) and len(terminal.get("stage_rows", [])) == 3
        else None
    )
    recomputed = _recomputed_holdout_gate(terminal_holdout)
    if recomputed is None:
        failures.append("heldout_residual_gate")
    else:
        maximum, gate = recomputed
        if (
            sector.get("heldout_terminal_max_normalized_residual") is None
            or not _finite_scalar(sector.get("heldout_terminal_max_normalized_residual"))
            or abs(float(sector["heldout_terminal_max_normalized_residual"]) - maximum)
            > 1.0e-12 * max(1.0, maximum)
            or sector.get("heldout_terminal_gate_pass") is not gate
            or gate is not True
        ):
            failures.append("heldout_residual_gate")
    return failures


def _validate_shape(result: Mapping[str, Any]) -> bool:
    if not isinstance(result, Mapping):
        return False
    if (
        result.get("schema_version") != SCHEMA
        or result.get("status") != STATUS
        or result.get("evidence_weight") != 0.0
        or not isinstance(result.get("cases"), Mapping)
        or set(result["cases"]) != set(TARGETS)
        or result.get("synthetic_controls", {}).get("all_checks_pass") is not True
    ):
        return False
    if not _finite_tree(result):
        return False
    for target in TARGETS:
        case = result["cases"].get(target)
        if (
            not isinstance(case, Mapping)
            or case.get("target_y") != target
            or case.get("target_N") != TARGET_N
            or not isinstance(case.get("backgrounds"), Mapping)
            or not isinstance(case.get("sectors"), Mapping)
            or set(case["sectors"]) != {str(ell) for ell in ELL_VALUES}
            or not isinstance(case.get("selections"), Mapping)
            or set(case["selections"]) != {str(ell) for ell in ELL_VALUES}
        ):
            return False
        source_summary = case.get("source_case")
        if (
            not isinstance(source_summary, Mapping)
            or source_summary.get("target_y") != target
            or source_summary.get("schema_version") != stability.SCHEMA
            or source_summary.get("status") != stability.STATUS
            or not isinstance(source_summary.get("terminal_controls_pass"), bool)
            or not isinstance(source_summary.get("terminal_control_failures"), list)
            or case.get("source_case_controls_pass") is not source_summary.get("terminal_controls_pass")
        ):
            return False
        for key in VALIDATION_BACKGROUND_KEYS:
            background = case["backgrounds"].get(key)
            if not isinstance(background, Mapping) or background.get("converged") is not True:
                return False
        if (
            case["backgrounds"].get("background_grid_intervals") != 3200
            or case["backgrounds"].get("background_nodes_requested") != stability.BACKGROUND_NODES
            or case["backgrounds"].get("background_tight_nodes_requested") != stability.BACKGROUND_TIGHT_NODES
            or case["backgrounds"].get("background_tolerance") != stability.BACKGROUND_TOLERANCE
            or case["backgrounds"].get("background_tight_tolerance") != stability.BACKGROUND_TIGHT_TOLERANCE
        ):
            return False
        for background_key in VALIDATION_BACKGROUND_KEYS:
            background = case["backgrounds"].get(background_key)
            if not all(_finite_scalar(background.get(field)) for field in (
                "chemical_potential_MeV", "conserved_N_relative_error", "field_residual_relative",
                "virial_relative", "gauss_energy_identity_relative", "edge_x",
            )):
                return False
        for ell in ELL_VALUES:
            selection = case["selections"][str(ell)]
            sector = case["sectors"][str(ell)]
            if not _validate_selection(selection, ell):
                return False
            if not isinstance(sector, Mapping) or sector.get("ell") != ell:
                return False
            rows = sector.get("validation_rows")
            if not isinstance(rows, list) or len(rows) != len(VALIDATION_BACKGROUND_KEYS) * len(GRID_INTERVALS):
                return False
            if not all(_validate_replay_row(row, ell) for row in rows):
                return False
            recomputed_sector_failures = _recomputed_sector_failures(sector, ell)
            if recomputed_sector_failures is None:
                return False
            if sector.get("terminal_control_failures") != recomputed_sector_failures:
                return False
            if sector.get("terminal_controls_pass") is not (not recomputed_sector_failures):
                return False
            if not isinstance(sector.get("terminal_signs"), list) or not isinstance(sector.get("terminal_sign_resolved"), list):
                return False
            values = sector.get("terminal_eigenvalues_Q_over_W0")
            if not isinstance(values, list) or len(values) != len(sector["terminal_signs"]):
                return False
            expected_signs = [
                "positive" if float(value) > 0.0 else "negative" if float(value) < 0.0 else "zero"
                for value in values
            ]
            if sector["terminal_signs"] != expected_signs:
                return False
            if not isinstance(sector.get("heldout_terminal_gate_pass"), bool):
                return False
            terminal_row = sector.get("terminal_validation")
            terminal_holdout = (
                terminal_row.get("stage_rows", [{}, {}, {}])[2].get("heldout")
                if isinstance(terminal_row, Mapping) and len(terminal_row.get("stage_rows", [])) == 3
                else None
            )
            if (
                not isinstance(terminal_holdout, Mapping)
                or sector.get("heldout_terminal_gate_pass")
                is not terminal_holdout.get("residual_gate_pass")
            ):
                return False
        # Recompute the top-level aggregate rather than trusting child flags.
        expected_failures: list[str] = []
        if case.get("source_case_controls_pass") is not True:
            expected_failures.append("inherited_source_controls")
        if not all(case["sectors"][str(ell)].get("terminal_controls_pass") is True for ell in ELL_VALUES):
            expected_failures.append("residual_enrichment_controls")
        if not all(case["selections"][str(ell)].get("could_complete_planned_refinement") is True for ell in ELL_VALUES):
            expected_failures.append("planned_refinement_count")
        if case.get("terminal_control_failures") != expected_failures:
            return False
        if case.get("terminal_controls_pass") is not (not expected_failures):
            return False
    return True


def validate_result(result: Mapping[str, Any]) -> bool:
    if not isinstance(result, Mapping):
        return False
    try:
        if not _validate_shape(result):
            return False
        fresh = build_result()
        return _validate_shape(fresh) and result == fresh
    except (ArithmeticError, AttributeError, IndexError, KeyError, ResidualEnrichmentError, TypeError, ValueError):
        return False


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # pragma: no cover - parser path.
        raise ResidualEnrichmentError(message)


def main(argv: list[str] | None = None) -> int:
    parser = JsonArgumentParser(description=__doc__)
    try:
        parser.parse_args(argv)
        result = build_result()
        if not _validate_shape(result):
            raise ResidualEnrichmentError("invalid or incomplete residual-enrichment result")
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    except (ArithmeticError, AttributeError, IndexError, KeyError, RuntimeError,
            ResidualEnrichmentError, TypeError, ValueError) as exc:
        print(json.dumps({
            "schema_version": SCHEMA,
            "status": "INVALID_OR_FAILED_DROPLET_RESIDUAL_ENRICHMENT_AUDIT",
            "evidence_weight": EVIDENCE_WEIGHT,
            "error": str(exc),
        }, ensure_ascii=False, allow_nan=False))
        return 2
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
