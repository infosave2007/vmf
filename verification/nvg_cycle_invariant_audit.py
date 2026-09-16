#!/usr/bin/env python3
"""Raw 2x2 cycle-invariant diagnostics for the declared transfer controls.

The algebraic identity

    T.T @ Q(T) @ T = det(T) * Q(T)

is useful for constructing a positive quadratic invariant in a robust
elliptic, area-preserving 2x2 block.  It is not an independent check of an
ODE solver: a Q built from the same reported T satisfies the identity by
construction.  This module therefore keeps the raw determinant, reports the
identity residual, and applies an explicit numerical boundary contract before
accepting a candidate.  Hyperbolic, Jordan, near-boundary, and determinant
invalid cases remain visible.

The four live rows are computed from the current scalar and tensor producers;
their outputs are not embedded result tables.  The command-line interface is
JSON-only and does not write files.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import sympy as sp


STATUS = "nvg_cycle_invariant_audit"
SCHEMA = "nvg_cycle_invariant_audit.v1"
EVIDENCE_WEIGHT = 0
SOURCE_PATH = Path(__file__).resolve()

# The tensor and scalar producers use a 3e-6 determinant gate.  This module
# adopts that gate verbatim for the raw one-cycle matrix and never rescales a
# matrix to force determinant one.
DETERMINANT_TOLERANCE = 3.0e-6
# A sign of det(Q) closer than this to zero is not a resolved Floquet type.
# It is deliberately much larger than ordinary double precision roundoff and
# is reported as near-boundary rather than promoted to a certificate.
BOUNDARY_TOLERANCE = 3.0e-7
IDENTITY_RESIDUAL_TOLERANCE = 3.0e-10
INVARIANCE_RESIDUAL_TOLERANCE = 3.0e-6

CLASSIFICATIONS = (
    "elliptic",
    "hyperbolic",
    "exact_plus_identity",
    "exact_minus_identity",
    "nontrivial_jordan",
    "near_boundary_unresolved",
    "determinant_invalid",
)

SOURCES = {
    "derivation_and_scope": "NVG_UNIVERSAL_FORMULAS_RESEARCH_RU.md",
    "scalar_producer":
        "verification/nvg_regular_scalar_perturbation_audit.py:closed_w1_transfer_convergence",
    "tensor_producer":
        "verification/nvg_cyclic_tensor_audit.py:tensor_mode_transfer",
}

SCOPE = {
    "mathematical": (
        "real 2x2 transfer identity with raw determinant and a quadratic "
        "candidate on a robust elliptic branch"),
    "determinant": (
        "actual determinant is retained; no determinant normalization or "
        "area repair is performed"),
    "boundary": (
        "det(Q) within the declared boundary tolerance is unresolved; a "
        "rounded near-Jordan matrix cannot receive a positive certificate"),
    "solver_validation": (
        "the Q identity is algebraic and is not independent solver validation"),
    "live_controls": (
        "scalar ell=2,6 and tensor (w,n)=(0,3),(1,3), each through its "
        "current producer convergence gate"),
    "interpretation": (
        "finite diagnostic controls only; no all-mode, nonlinear, quantum, "
        "observational, or NVG completion claim"),
}

ANALYSIS_REQUIRED_KEYS = frozenset({
    "matrix", "trace", "determinant", "determinant_error",
    "determinant_tolerance", "determinant_margin", "discriminant",
    "boundary_tolerance", "boundary_margin", "q_matrix",
    "q_determinant_direct", "q_determinant_formula",
    "q_determinant_consistency_error", "algebraic_identity_residual",
    "exact_structural",
    "classification", "candidate_available", "candidate",
    "numerical_acceptance", "numerical_certificate_passed",
    "certificate_passed",
    "strict_exact_certificate", "scope",
})

RESULT_REQUIRED_KEYS = frozenset({
    "status", "schema", "evidence_weight", "mathematical_checks_passed",
    "symbolic_checks", "mathematical_examples", "live_controls", "summary",
    "scope", "sources",
})

SYMBOLIC_REQUIRED_ROWS = frozenset({
    "Q_is_symmetric",
    "transfer_identity",
    "determinant_identity",
    "discriminant_identity",
})

MATHEMATICAL_EXAMPLE_LABELS = (
    "analytical_rotation", "conjugated_rotation", "exact_plus_identity",
    "exact_minus_identity", "nontrivial_jordan_plus",
    "nontrivial_jordan_minus", "hyperbolic_diagonal", "determinant_drift",
    "near_boundary_rotation",
)
LIVE_CONTROL_LABELS = (
    "scalar_ell_2", "scalar_ell_6", "tensor_w_0_n_3", "tensor_w_1_n_3",
)
SCALAR_CONVERGENCE_KEYS = frozenset({
    "ell", "epsilon", "rho_c_over_rho_ref", "coarse", "fine",
    "relative_matrix_difference", "passed",
})
SCALAR_TRANSFER_KEYS = frozenset({
    "ell", "lambda", "epsilon", "rho_c_over_rho_ref", "A_low", "A_end",
    "q_end", "matrix", "determinant", "determinant_error", "symplectic_error",
    "trace", "discriminant", "classification", "eigenvalues_real",
    "eigenvalues_imag", "classification_tolerance", "max_constraint_residual",
    "max_background_equation_residual", "min_A_s", "min_U", "min_Delta",
    "rhs_evaluations", "finite", "scope",
})

EXACT_STRUCTURAL_REQUIRED_KEYS = frozenset({
    "determinant", "trace", "discriminant", "q_determinant",
    "determinant_is_one", "trace_is_plus_two", "trace_is_minus_two",
    "q_determinant_is_zero", "is_plus_identity", "is_minus_identity",
    "is_nontrivial_jordan",
})


def _finite(value: Any, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    """Parse a finite real, rejecting bools and non-finite values."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _validate_matrix(matrix: Any, name: str = "matrix") -> List[List[float]]:
    """Validate and copy a finite, explicitly shaped 2x2 matrix."""
    if (not isinstance(matrix, (list, tuple)) or len(matrix) != 2
            or any(not isinstance(row, (list, tuple)) or len(row) != 2
                   for row in matrix)):
        raise ValueError(f"{name} must be a 2x2 matrix")
    return [[_finite(value, f"{name}[{i}][{j}]")
             for j, value in enumerate(row)]
            for i, row in enumerate(matrix)]


def _matrix_determinant(matrix: Sequence[Sequence[float]]) -> float:
    value = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if not math.isfinite(value):
        raise ValueError("matrix determinant exceeds finite arithmetic")
    return value


def _exact_structural_predicates(
        matrix: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Evaluate structural predicates over the supplied binary64 values.

    ``Fraction.from_float`` imports each already-validated entry as its exact
    dyadic value.  The resulting determinant, trace, discriminant, and
    boundary determinant are deliberately kept as Fractions until the
    structural comparisons are complete; converting a rational result back
    to float would recreate the equality bug this repair is intended to
    prevent.
    """
    exact = tuple(tuple(Fraction.from_float(value) for value in row)
                  for row in matrix)
    a, b = exact[0]
    c, d = exact[1]
    determinant = a * d - b * c
    trace = a + d
    discriminant = trace * trace - 4 * determinant
    q_determinant = determinant - trace * trace / 4
    plus_identity = exact == (
        (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    minus_identity = exact == (
        (Fraction(-1), Fraction(0)), (Fraction(0), Fraction(-1)))
    determinant_is_one = determinant == 1
    trace_is_parabolic = trace in (2, -2)
    q_determinant_is_zero = q_determinant == 0
    nontrivial_jordan = bool(
        determinant_is_one and trace_is_parabolic
        and q_determinant_is_zero and not plus_identity and not minus_identity)
    return {
        "determinant": str(determinant),
        "trace": str(trace),
        "discriminant": str(discriminant),
        "q_determinant": str(q_determinant),
        "determinant_is_one": determinant_is_one,
        "trace_is_plus_two": trace == 2,
        "trace_is_minus_two": trace == -2,
        "q_determinant_is_zero": q_determinant_is_zero,
        "is_plus_identity": plus_identity,
        "is_minus_identity": minus_identity,
        "is_nontrivial_jordan": nontrivial_jordan,
    }


def _exact_fraction(value: str) -> Fraction:
    """Parse one internal exact-rational diagnostic value."""
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(int(value), 1)


def exact_structural_predicates(matrix: Any) -> Dict[str, Any]:
    """Return structural predicates for the exact supplied binary64 matrix."""
    return _exact_structural_predicates(_validate_matrix(matrix))


def matrix_determinant(matrix: Any) -> float:
    """Return the raw determinant after strict input validation."""
    return _matrix_determinant(_validate_matrix(matrix))


def _matrix_residual(left: Sequence[Sequence[float]],
                     right: Sequence[Sequence[float]]) -> float:
    scale = max(1.0, *(abs(value) for row in right for value in row))
    residual = max(abs(left[i][j] - right[i][j])
                   for i in range(2) for j in range(2)) / scale
    if not math.isfinite(residual):
        raise ValueError("matrix residual exceeds finite arithmetic")
    return residual


def _transpose_q_product(matrix: Sequence[Sequence[float]],
                          q_matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    # Compute T.T Q T directly, retaining the declared determinant in the
    # comparison target rather than normalizing T first.
    tq = [[sum(matrix[k][i] * q_matrix[k][j] for k in range(2))
           for j in range(2)] for i in range(2)]
    result = [[sum(tq[i][k] * matrix[k][j] for k in range(2))
               for j in range(2)] for i in range(2)]
    if not all(math.isfinite(value) for row in result for value in row):
        raise ValueError("quadratic-form product exceeds finite arithmetic")
    return result


def quadratic_form(matrix: Any) -> List[List[float]]:
    """Return Q(T)=[[-c,(a-d)/2],[(a-d)/2,b]]."""
    validated = _validate_matrix(matrix)
    a, b = validated[0]
    c, d = validated[1]
    return [[-c, (a - d) / 2.0], [(a - d) / 2.0, b]]


# Descriptive aliases keep this diagnostic discoverable beside the producers.
invariant_quadratic_form = quadratic_form
q_matrix = quadratic_form


def _rows(expressions: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for name, expression in expressions.items():
        reduced = sp.factor(sp.trigsimp(sp.simplify(expression)))
        rows[name] = {"residual": str(reduced),
                      "passed": bool(reduced == 0)}
    return rows


def rows_pass(rows: Any,
              required_names: Iterable[str] = SYMBOLIC_REQUIRED_ROWS) -> bool:
    """Fail closed unless every declared exact row is literally zero."""
    if not isinstance(rows, dict) or set(rows) != set(required_names):
        return False
    return all(
        isinstance(row, dict)
        and set(row) == {"residual", "passed"}
        and isinstance(row["residual"], str)
        and isinstance(row["passed"], bool)
        and row["passed"] is True
        and row["residual"] == "0"
        for row in rows.values())


def symbolic_cycle_invariant_checks() -> Dict[str, Dict[str, Any]]:
    """Derive the Q identities exactly with independent symbolic variables."""
    a, b, c, d = sp.symbols("a b c d", real=True)
    transfer = sp.Matrix([[a, b], [c, d]])
    trace = sp.trace(transfer)
    determinant = transfer.det()
    q = sp.Matrix([[-c, (a - d) / 2], [(a - d) / 2, b]])
    transfer_residual = transfer.T * q * transfer - determinant * q
    rows = {
        "Q_is_symmetric": sum(item ** 2 for item in q - q.T),
        "transfer_identity": sum(item ** 2 for item in transfer_residual),
        "determinant_identity": q.det() - (determinant - trace ** 2 / 4),
        "discriminant_identity": q.det() + (trace ** 2 - 4 * determinant) / 4,
    }
    return _rows(rows)


# Short aliases used by focused callers.
symbolic_checks = symbolic_cycle_invariant_checks
cycle_invariant_symbolic_checks = symbolic_cycle_invariant_checks


def _classification(matrix: Sequence[Sequence[float]], determinant: float,
                    trace: float, q_determinant: float,
                    *, determinant_tolerance: float,
                    boundary_tolerance: float,
                    exact_structural: Optional[Mapping[str, Any]] = None,
                    numerical_q_determinant: Optional[float] = None) -> str:
    if abs(determinant - 1.0) > determinant_tolerance:
        return "determinant_invalid"
    # Structural cases are decided from the exact dyadic values of the
    # supplied binary64 entries.  In particular, float(det)==1.0 and a
    # rounded trace/discriminant are not exact mathematical evidence.
    exact = (exact_structural if exact_structural is not None
             else _exact_structural_predicates(matrix))
    if exact["is_plus_identity"]:
        return "exact_plus_identity"
    if exact["is_minus_identity"]:
        return "exact_minus_identity"
    if exact["is_nontrivial_jordan"]:
        return "nontrivial_jordan"
    # ``q_determinant`` is retained as the historical trace/determinant
    # formula.  The direct Q determinant is a better-conditioned numerical
    # candidate for the sign near a repeated-eigenvalue boundary, so callers
    # may supply it explicitly.  This is still only a numerical classification
    # when the exact area-preserving predicate above is false.
    numerical_q = (q_determinant if numerical_q_determinant is None
                   else numerical_q_determinant)
    if numerical_q > boundary_tolerance:
        return "elliptic"
    if numerical_q < -boundary_tolerance:
        return "hyperbolic"
    return "near_boundary_unresolved"


def classify_transfer(matrix: Any, *,
                      determinant_tolerance: Any = DETERMINANT_TOLERANCE,
                      boundary_tolerance: Any = BOUNDARY_TOLERANCE) -> str:
    """Classify a raw transfer without determinant normalization."""
    m = _validate_matrix(matrix)
    dt = _finite(determinant_tolerance, "determinant_tolerance", positive=True)
    bt = _finite(boundary_tolerance, "boundary_tolerance", positive=True)
    determinant = _matrix_determinant(m)
    trace = m[0][0] + m[1][1]
    q_determinant = determinant - trace * trace / 4.0
    q = [[-m[1][0], (m[0][0] - m[1][1]) / 2.0],
         [(m[0][0] - m[1][1]) / 2.0, m[0][1]]]
    q_direct = _matrix_determinant(q)
    if not math.isfinite(q_determinant) or not math.isfinite(q_direct):
        raise ValueError("transfer classification exceeds finite arithmetic")
    exact_structural = _exact_structural_predicates(m)
    return _classification(m, determinant, trace, q_determinant,
                           determinant_tolerance=dt,
                           boundary_tolerance=bt,
                           exact_structural=exact_structural,
                           numerical_q_determinant=q_direct)


def _symmetric_eigenvalues(matrix: Sequence[Sequence[float]]) -> Tuple[float, float]:
    x, y = matrix[0]
    y2, z = matrix[1]
    if abs(y - y2) > 1e-14 * max(1.0, abs(y), abs(y2)):
        raise ValueError("candidate invariant is not symmetric")
    center = (x + z) / 2.0
    radius = math.hypot((x - z) / 2.0, y)
    values = (center - radius, center + radius)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("candidate eigenvalues exceed finite arithmetic")
    return values


def analyze_transfer(matrix: Any, *,
                     determinant_tolerance: Any = DETERMINANT_TOLERANCE,
                     boundary_tolerance: Any = BOUNDARY_TOLERANCE,
                     invariance_tolerance: Any = INVARIANCE_RESIDUAL_TOLERANCE,
                     label: Optional[str] = None) -> Dict[str, Any]:
    """Analyze one raw 2x2 transfer and, only when resolved, build G."""
    m = _validate_matrix(matrix)
    dt = _finite(determinant_tolerance, "determinant_tolerance", positive=True)
    bt = _finite(boundary_tolerance, "boundary_tolerance", positive=True)
    it = _finite(invariance_tolerance, "invariance_tolerance", positive=True)
    determinant = _matrix_determinant(m)
    trace = m[0][0] + m[1][1]
    discriminant = trace * trace - 4.0 * determinant
    q = [[-m[1][0], (m[0][0] - m[1][1]) / 2.0],
         [(m[0][0] - m[1][1]) / 2.0, m[0][1]]]
    q_direct = _matrix_determinant(q)
    q_formula = determinant - trace * trace / 4.0
    if not all(math.isfinite(value) for value in
               (trace, discriminant, q_direct, q_formula)):
        raise ValueError("transfer diagnostics exceed finite arithmetic")
    exact_structural = _exact_structural_predicates(m)
    classification = _classification(
        m, determinant, trace, q_formula,
        determinant_tolerance=dt, boundary_tolerance=bt,
        exact_structural=exact_structural,
        numerical_q_determinant=q_direct)
    identity_target = [[determinant * q[i][j] for j in range(2)]
                       for i in range(2)]
    identity_lhs = _transpose_q_product(m, q)
    identity_residual = _matrix_residual(identity_lhs, identity_target)

    candidate: Optional[Dict[str, Any]] = None
    candidate_kind = classification in {
        "elliptic", "exact_plus_identity", "exact_minus_identity",
    }
    if candidate_kind:
        identity_candidate = classification in {
            "exact_plus_identity", "exact_minus_identity",
        }
        if identity_candidate:
            # Q(T) vanishes for +/-I, but the positive invariant is not absent:
            # G=I is an exact positive invariant for either identity transfer.
            scale: Optional[float] = None
            g = [[1.0, 0.0], [0.0, 1.0]]
        else:
            b = m[0][1]
            if b == 0.0 or q_direct <= bt:
                g = None
                scale = None
            else:
                scale = ((1.0 if b > 0.0 else -1.0)
                         / math.sqrt(q_direct))
                g = [[scale * q[i][j] for j in range(2)]
                     for i in range(2)]
        if g is not None:
            if not all(math.isfinite(value) for row in g for value in row):
                raise ValueError("candidate invariant exceeds finite arithmetic")
            eig_min, eig_max = _symmetric_eigenvalues(g)
            g_determinant = _matrix_determinant(g)
            g_target = g
            g_lhs = _transpose_q_product(m, g)
            invariance_target_residual = _matrix_residual(g_lhs, g_target)
            condition = (eig_max / eig_min
                         if eig_min > 0.0 else float("inf"))
            if not math.isfinite(condition) and eig_min > 0.0:
                raise ValueError("candidate condition number exceeds finite arithmetic")
            spd = bool(eig_min > 0.0 and math.isfinite(eig_max))
            determinant_valid = abs(determinant - 1.0) <= dt
            exact_q_resolved = (
                _exact_fraction(exact_structural["q_determinant"])
                > Fraction.from_float(bt))
            q_resolved = identity_candidate or q_direct > bt
            numerical_certificate = bool(
                determinant_valid and spd and q_resolved
                and identity_residual <= it
                and invariance_target_residual <= it)
            strict_certificate = bool(
                exact_structural["determinant_is_one"] and spd
                and (identity_candidate or exact_q_resolved)
                and identity_residual <= it
                and invariance_target_residual <= it)
            candidate = {
                "scale": scale,
                "g_matrix": g,
                "g_determinant": g_determinant,
                "min_eigenvalue": eig_min,
                "max_eigenvalue": eig_max,
                "condition_number": condition,
                "coordinate_norm_bound": math.sqrt(condition) if spd else None,
                "spd": spd,
                "invariance_residual": invariance_target_residual,
                "numerical_certificate": numerical_certificate,
                "strict_exact_certificate": strict_certificate,
                "determinant_contract": (
                    "raw determinant retained; candidate is numerically "
                    "area-preserving within the declared gate"),
            }

    determinant_valid = abs(determinant - 1.0) <= dt
    robust_classification = classification not in {
        "near_boundary_unresolved", "determinant_invalid",
    }
    numerical_acceptance = bool(
        determinant_valid and robust_classification
        and identity_residual <= it)
    strict_exact_certificate = bool(
        candidate is not None and candidate["strict_exact_certificate"]
        and classification in {
            "elliptic", "exact_plus_identity", "exact_minus_identity",
        })
    numerical_certificate_passed = bool(
        candidate is not None and candidate["numerical_certificate"]
        and classification in {
            "elliptic", "exact_plus_identity", "exact_minus_identity",
        })
    return {
        "matrix": m,
        "trace": trace,
        "determinant": determinant,
        "determinant_error": abs(determinant - 1.0),
        "determinant_tolerance": dt,
        "determinant_margin": dt - abs(determinant - 1.0),
        "discriminant": discriminant,
        "boundary_tolerance": bt,
        "boundary_margin": abs(q_formula) - bt,
        "q_matrix": q,
        "q_determinant_direct": q_direct,
        "q_determinant_formula": q_formula,
        "q_determinant_consistency_error": abs(q_direct - q_formula),
        "algebraic_identity_residual": identity_residual,
        "exact_structural": exact_structural,
        "classification": classification,
        "candidate_available": candidate is not None,
        "candidate": candidate,
        "numerical_acceptance": numerical_acceptance,
        # A positive *exact* certificate is reserved for a raw determinant
        # represented exactly as one.  Floating-point live rows expose a
        # separate tolerance-qualified candidate instead of silently calling
        # a non-area-preserving matrix exact.
        "numerical_certificate_passed": numerical_certificate_passed,
        "certificate_passed": strict_exact_certificate,
        "strict_exact_certificate": strict_exact_certificate,
        "scope": {
            "q_identity": (
                "algebraic identity only; not independent solver validation"),
            "determinant": "raw determinant; no normalization",
            "conditioning": "Euclidean 2-norm conditioning of reported G",
            "classification_tolerance": (
                "elliptic/hyperbolic only outside the boundary band"),
            "label": label,
        },
    }


# Convenient names for interactive use.
cycle_invariant = analyze_transfer
positive_invariant_candidate = analyze_transfer


def _rotation(theta: float) -> List[List[float]]:
    c, s = math.cos(_finite(theta, "theta")), math.sin(_finite(theta, "theta"))
    return [[c, s], [-s, c]]


def _matmul(left: Sequence[Sequence[float]],
            right: Sequence[Sequence[float]]) -> List[List[float]]:
    result = [[sum(left[i][k] * right[k][j] for k in range(2))
               for j in range(2)] for i in range(2)]
    if not all(math.isfinite(value) for row in result for value in row):
        raise ValueError("matrix product exceeds finite arithmetic")
    return result


def _inverse(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    determinant = _matrix_determinant(matrix)
    if determinant == 0.0:
        raise ValueError("conjugating matrix must be nonsingular")
    return [[matrix[1][1] / determinant, -matrix[0][1] / determinant],
            [-matrix[1][0] / determinant, matrix[0][0] / determinant]]


def mathematical_examples() -> List[Dict[str, Any]]:
    """Construct fixed mathematical controls and recompute their types."""
    rotation = _rotation(0.73)
    similarity = [[2.0, 1.0], [0.25, 0.75]]
    conjugated = _matmul(_matmul(similarity, _rotation(-0.61)),
                         _inverse(similarity))
    plus_jordan = [[1.0, 1.0], [0.0, 1.0]]
    minus_jordan = [[-1.0, 1.0], [0.0, -1.0]]
    hyperbolic = [[2.0, 0.0], [0.0, 0.5]]
    drifted = [[1.0001 * value for value in row] for row in rotation]
    near_boundary = _rotation(1.0e-6)
    controls = (
        ("analytical_rotation", rotation, "exact rotation control"),
        ("conjugated_rotation", conjugated, "similarity-transformed rotation control"),
        ("exact_plus_identity", [[1.0, 0.0], [0.0, 1.0]], "exact +I control"),
        ("exact_minus_identity", [[-1.0, 0.0], [0.0, -1.0]], "exact -I control"),
        ("nontrivial_jordan_plus", plus_jordan, "exact nontrivial + Jordan control"),
        ("nontrivial_jordan_minus", minus_jordan, "exact nontrivial - Jordan control"),
        ("hyperbolic_diagonal", hyperbolic, "exact hyperbolic control"),
        ("determinant_drift", drifted, "non-area-preserving negative control"),
        ("near_boundary_rotation", near_boundary, "unresolved near-Jordan boundary control"),
    )
    rows = []
    for label, matrix, scope in controls:
        analysis = analyze_transfer(matrix, label=label)
        rows.append({
            "label": label,
            "kind": "fixed_mathematical_control",
            "matrix": matrix,
            "analysis": analysis,
            "accepted": bool(analysis["numerical_acceptance"]),
            "scope": scope,
        })
    return rows


def _number_close(actual: Any, expected: Any, *, relative: float = 1.0e-10,
                  absolute: float = 1.0e-12) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return False
    try:
        a, b = float(actual), float(expected)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(a) or not math.isfinite(b):
        return False
    return abs(a - b) <= max(absolute, relative * max(1.0, abs(a), abs(b)))


def _deep_equivalent(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return _number_close(actual, expected)
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, str) or isinstance(expected, str):
        return actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return (set(actual) == set(expected)
                and all(_deep_equivalent(actual[key], expected[key])
                        for key in actual))
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return (len(actual) == len(expected)
                and all(_deep_equivalent(a, b)
                        for a, b in zip(actual, expected)))
    return actual == expected


def analysis_acceptance(analysis: Any) -> bool:
    """Recompute one analysis and reject stale, missing, or mutated fields."""
    if not isinstance(analysis, dict) or set(analysis) != set(ANALYSIS_REQUIRED_KEYS):
        return False
    if (not _number_close(analysis.get("determinant_tolerance"),
                          DETERMINANT_TOLERANCE, relative=0.0,
                          absolute=0.0)
            or not _number_close(analysis.get("boundary_tolerance"),
                                BOUNDARY_TOLERANCE, relative=0.0,
                                absolute=0.0)):
        return False
    exact_structural = analysis.get("exact_structural")
    if (not isinstance(exact_structural, dict)
            or set(exact_structural) != EXACT_STRUCTURAL_REQUIRED_KEYS):
        return False
    try:
        matrix = _validate_matrix(analysis["matrix"])
        dt = analysis["determinant_tolerance"]
        bt = analysis["boundary_tolerance"]
        scope = analysis.get("scope")
        label = scope.get("label") if isinstance(scope, dict) else None
        expected = analyze_transfer(
            matrix, determinant_tolerance=dt, boundary_tolerance=bt,
            invariance_tolerance=INVARIANCE_RESIDUAL_TOLERANCE, label=label)
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False
    # The public output does not expose invariance_tolerance; enforce its
    # declared value through the recomputation contract above.
    if not _deep_equivalent(analysis, expected):
        return False
    return analysis["classification"] in CLASSIFICATIONS


def _producer_matrix_gate(record: Any, matrix_name: str,
                          determinant_name: str,
                          determinant_error_name: str,
                          residual_name: str) -> bool:
    if not isinstance(record, dict):
        return False
    try:
        matrix = _validate_matrix(record[matrix_name], matrix_name)
        determinant = _matrix_determinant(matrix)
        reported = _finite(record[determinant_name], determinant_name)
        error = _finite(record[determinant_error_name], determinant_error_name,
                        nonnegative=True)
        residual = _finite(record[residual_name], residual_name,
                           nonnegative=True)
        return bool(
            _number_close(reported, determinant)
            and _number_close(error, abs(determinant - 1.0))
            and _number_close(residual, abs(determinant - 1.0))
            and record.get("finite") is True
            and isinstance(record.get("rhs_evaluations"), int)
            and not isinstance(record.get("rhs_evaluations"), bool)
            and record["rhs_evaluations"] > 0)
    except (KeyError, TypeError, ValueError):
        return False


def _scalar_producer_gate(row: Any, ell: int) -> bool:
    """Recheck the scalar producer's selected convergence row."""
    if (not isinstance(row, dict)
            or set(row) != set(SCALAR_CONVERGENCE_KEYS)
            or row.get("passed") is not True):
        return False
    if (row.get("ell") != ell or row.get("epsilon") != 0.1
            or row.get("rho_c_over_rho_ref") != 1.0):
        return False
    for nested in (row.get("coarse"), row.get("fine")):
        if (not isinstance(nested, dict)
                or set(nested) != set(SCALAR_TRANSFER_KEYS)):
            return False
        if not _producer_matrix_gate(
                nested, "matrix", "determinant", "determinant_error",
                "symplectic_error"):
            return False
        for name in ("A_low", "A_end", "q_end", "max_constraint_residual",
                     "max_background_equation_residual", "min_A_s", "min_U",
                     "min_Delta"):
            try:
                value = _finite(nested[name], name)
            except (KeyError, ValueError):
                return False
            if name in ("A_low", "A_end", "min_A_s", "min_U", "min_Delta") \
                    and value <= 0.0:
                return False
            if name in ("max_constraint_residual",
                        "max_background_equation_residual") and value < 0.0:
                return False
    try:
        coarse, fine = row["coarse"], row["fine"]
        refinement = max(abs(coarse["matrix"][i][j] - fine["matrix"][i][j])
                         / max(1.0, abs(fine["matrix"][i][j]))
                         for i in range(2) for j in range(2))
        reported_refinement = _finite(row["relative_matrix_difference"],
                                      "relative_matrix_difference",
                                      nonnegative=True)
        return bool(
            _number_close(reported_refinement, refinement)
            and refinement < 3.0e-5
            and fine["determinant_error"] < 3.0e-6
            # These are the scalar producer's own closed-transfer gates.
            and fine["max_constraint_residual"] < 3.0e-8
            and fine["max_background_equation_residual"] < 3.0e-8
            and fine["min_A_s"] > 0.0 and fine["min_U"] > 0.0
            and fine["min_Delta"] > 0.0
            and abs(fine["A_end"] - fine["A_low"]) < 3.0e-6
            and abs(fine["q_end"] - math.pi) < 3.0e-7)
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False


def _live_scalar(ell: int) -> Dict[str, Any]:
    try:
        from nvg_regular_scalar_perturbation_audit import (
            closed_w1_transfer_convergence)
    except ImportError:
        from verification.nvg_regular_scalar_perturbation_audit import (
            closed_w1_transfer_convergence)
    row = closed_w1_transfer_convergence(ell)
    producer_gate = _scalar_producer_gate(row, ell)
    analysis = analyze_transfer(row["fine"]["matrix"],
                                label=f"scalar_ell_{ell}")
    return {
        "label": f"scalar_ell_{ell}",
        "kind": "scalar",
        "control": {"ell": ell, "epsilon": 0.1,
                     "rho_c_over_rho_ref": 1.0},
        "producer": SOURCES["scalar_producer"],
        "producer_gate": producer_gate,
        "producer_record": copy.deepcopy(row),
        "analysis": analysis,
        "accepted": bool(producer_gate and analysis["numerical_acceptance"]),
        "scope": "live scalar control through the current closed-w=1 convergence gate",
    }


def _live_tensor(w: float, n: int) -> Dict[str, Any]:
    try:
        from nvg_cyclic_tensor_audit import mode_acceptance, tensor_mode_transfer
    except ImportError:
        from verification.nvg_cyclic_tensor_audit import (
            mode_acceptance, tensor_mode_transfer)
    row = tensor_mode_transfer(w=w, n=n)
    producer_gate = bool(mode_acceptance(row))
    analysis = analyze_transfer(row["matrix_fine"],
                                label=f"tensor_w_{w:g}_n_{n}")
    return {
        "label": f"tensor_w_{w:g}_n_{n}",
        "kind": "tensor",
        "control": {"w": w, "n": n, "epsilon": 0.1,
                     "rho_c_over_rho_ref": 1.0},
        "producer": SOURCES["tensor_producer"],
        "producer_gate": producer_gate,
        "producer_record": copy.deepcopy(row),
        "analysis": analysis,
        "accepted": bool(producer_gate and analysis["numerical_acceptance"]),
        "scope": "live tensor control through the current TT convergence gate",
    }


def live_controls() -> List[Dict[str, Any]]:
    """Compute only the four declared live controls from current producers."""
    return [_live_scalar(2), _live_scalar(6),
            _live_tensor(0.0, 3), _live_tensor(1.0, 3)]


compute_live_controls = live_controls


def _case_acceptance(case: Any) -> bool:
    if not isinstance(case, dict):
        return False
    required = {"label", "kind", "matrix", "analysis", "accepted", "scope"}
    if set(case) != required or case.get("kind") != "fixed_mathematical_control":
        return False
    if case.get("label") not in MATHEMATICAL_EXAMPLE_LABELS:
        return False
    try:
        matrix = _validate_matrix(case["matrix"])
    except ValueError:
        return False
    analysis = case.get("analysis")
    if not analysis_acceptance(analysis):
        return False
    return (analysis["matrix"] == matrix
            and analysis["scope"].get("label") == case["label"]
            and case["accepted"] is analysis["numerical_acceptance"]
            and isinstance(case.get("label"), str)
            and isinstance(case.get("scope"), str))


def _live_acceptance(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    required = {"label", "kind", "control", "producer", "producer_gate",
                "producer_record", "analysis", "accepted", "scope"}
    if set(row) != required or row.get("producer_gate") is not True:
        return False
    if not analysis_acceptance(row.get("analysis")):
        return False
    if row.get("accepted") is not (
            row["producer_gate"] and row["analysis"]["numerical_acceptance"]):
        return False
    if row.get("kind") == "scalar":
        control = row.get("control")
        if not isinstance(control, dict):
            return False
        if (row.get("label") not in LIVE_CONTROL_LABELS
                or row.get("producer") != SOURCES["scalar_producer"]
                or control.get("ell") not in (2, 6)
                or control.get("epsilon") != 0.1
                or control.get("rho_c_over_rho_ref") != 1.0
                or row.get("label") != f"scalar_ell_{control['ell']}"
                or not _scalar_producer_gate(row.get("producer_record"),
                                              control["ell"])):
            return False
        expected_matrix = row["producer_record"]["fine"]["matrix"]
    elif row.get("kind") == "tensor":
        control = row.get("control")
        if (not isinstance(control, dict)
                or row.get("producer") != SOURCES["tensor_producer"]
                or (control.get("w"), control.get("n")) not in
                ((0.0, 3), (1.0, 3))
                or control.get("epsilon") != 0.1
                or control.get("rho_c_over_rho_ref") != 1.0):
            return False
        expected_label = f"tensor_w_{control['w']:g}_n_{control['n']}"
        if row.get("label") != expected_label or row.get("label") not in LIVE_CONTROL_LABELS:
            return False
        try:
            from nvg_cyclic_tensor_audit import mode_acceptance
        except ImportError:
            from verification.nvg_cyclic_tensor_audit import mode_acceptance
        if not mode_acceptance(row.get("producer_record")):
            return False
        expected_matrix = row["producer_record"]["matrix_fine"]
    else:
        return False
    return (row["analysis"]["matrix"] == _validate_matrix(expected_matrix)
            and row["analysis"]["scope"].get("label") == row["label"])


def _summary(examples: Sequence[Mapping[str, Any]],
             live: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {name: 0 for name in CLASSIFICATIONS}
    for row in list(examples) + list(live):
        counts[row["analysis"]["classification"]] += 1
    return {
        "mathematical_example_count": len(examples),
        "live_control_count": len(live),
        "classification_counts": counts,
        "positive_certificate_labels": [
            row["label"] for row in list(examples) + list(live)
            if row["analysis"]["certificate_passed"]],
        "numerical_candidate_labels": [
            row["label"] for row in list(examples) + list(live)
            if row["analysis"]["numerical_certificate_passed"]],
        "near_boundary_labels": [
            row["label"] for row in list(examples) + list(live)
            if row["analysis"]["classification"] == "near_boundary_unresolved"],
        "determinant_invalid_labels": [
            row["label"] for row in list(examples) + list(live)
            if row["analysis"]["classification"] == "determinant_invalid"],
        "hyperbolic_labels": [
            row["label"] for row in list(examples) + list(live)
            if row["analysis"]["classification"] == "hyperbolic"],
        "all_live_controls_accepted": all(row["accepted"] for row in live),
        "finite_sample_only": True,
        "q_identity_is_algebraic_not_solver_validation": True,
    }


def compute_audit() -> Dict[str, Any]:
    """Run exact controls, fixed examples, and the four live integrations."""
    examples = mathematical_examples()
    live = live_controls()
    symbolic = symbolic_cycle_invariant_checks()
    summary = _summary(examples, live)
    passed = bool(
        rows_pass(symbolic)
        and all(_case_acceptance(row) for row in examples)
        and all(_live_acceptance(row) for row in live)
        and summary["all_live_controls_accepted"])
    return {
        "status": STATUS,
        "schema": SCHEMA,
        "evidence_weight": EVIDENCE_WEIGHT,
        "mathematical_checks_passed": passed,
        "symbolic_checks": symbolic,
        "mathematical_examples": examples,
        "live_controls": live,
        "summary": summary,
        "scope": SCOPE,
        "sources": SOURCES,
    }


run_audit = compute_audit
compute_state = compute_audit


def acceptance_from_result(result: Any) -> bool:
    """Fail closed on missing, stale, nonfinite, or mutated evidence."""
    if not isinstance(result, dict) or set(result) != set(RESULT_REQUIRED_KEYS):
        return False
    if (result.get("status") != STATUS or result.get("schema") != SCHEMA
            or result.get("evidence_weight") != EVIDENCE_WEIGHT
            or result.get("mathematical_checks_passed") is not True
            or result.get("scope") != SCOPE or result.get("sources") != SOURCES):
        return False
    if not rows_pass(result.get("symbolic_checks")):
        return False
    examples = result.get("mathematical_examples")
    live = result.get("live_controls")
    if (not isinstance(examples, list) or not examples
            or not isinstance(live, list) or len(live) != 4):
        return False
    if (any(not isinstance(row, dict) for row in examples)
            or any(not isinstance(row, dict) for row in live)):
        return False
    if (len(examples) != len(MATHEMATICAL_EXAMPLE_LABELS)
            or {row.get("label") for row in examples}
            != set(MATHEMATICAL_EXAMPLE_LABELS)
            or not all(_case_acceptance(row) for row in examples)):
        return False
    if ({row.get("label") for row in live} != set(LIVE_CONTROL_LABELS)
            or not all(_live_acceptance(row) for row in live)):
        return False
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return False
    expected_summary = _summary(examples, live)
    return summary == expected_summary and summary["all_live_controls_accepted"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArithmeticError("nonfinite JSON output")
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = _json_safe(compute_audit())
    except (ValueError, ArithmeticError, ImportError) as exc:
        result = {
            "status": "invalid_or_failed_audit",
            "schema": SCHEMA,
            "evidence_weight": EVIDENCE_WEIGHT,
            "mathematical_checks_passed": False,
            "error": str(exc),
        }
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["mathematical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
