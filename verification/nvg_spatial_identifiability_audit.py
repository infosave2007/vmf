#!/usr/bin/env python3
"""Static in-medium Thomas--Fermi response and W0 identifiability audit.

The producer recomputes the fixed homogeneous action and its local finite-wave
static Hessian.  It is deliberately narrower than a quantum RPA, a finite
nucleus, transport, or a dynamical spectrum.  The vector potential is a Gauss
constraint saddle; its negative Hessian entry is eliminated before a positive
static response is reported.

The command prints strict JSON and has no result-file side effect.  All rows
carry zero independent empirical weight.  ``chi`` is defined for a source
``-mu_ext*n``: ``delta n = chi*delta mu_ext``.  A source written instead as
``+V_ext*n`` has conventional response ``delta n/delta V_ext = -chi``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import mpmath as mp

# This audit is intentionally no-write, including no byte-code side effect.
sys.dont_write_bytecode = True

try:  # Direct script and verification-directory imports.
    import nvg_foundation_identifiability_audit as foundation
    import nvg_nuclear_closure_audit as nuclear
    import source_complete_scaling_saturation_audit as upstream
except ImportError:  # pragma: no cover - package-style fallback
    from . import nvg_foundation_identifiability_audit as foundation
    from . import nvg_nuclear_closure_audit as nuclear
    from . import source_complete_scaling_saturation_audit as upstream


HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
UPSTREAM_PATH = HERE / "source_complete_scaling_saturation_audit.py"
FOUNDATION_PATH = HERE / "nvg_foundation_identifiability_audit.py"
NUCLEAR_PATH = HERE / "nvg_nuclear_closure_audit.py"
CONTRACT_PATH = HERE / "contracts" / "spatial_identifiability.md"

SCHEMA_VERSION = 1
STATUS = "LIVE_STATIC_TF_SPATIAL_IDENTIFIABILITY_ZERO_EVIDENCE_NOT_EMPIRICAL"
EVIDENCE_WEIGHT = 0.0
PRIMARY_DPS = 80
CONTROL_DPS = 110
PRECISION_LIMIT = mp.mpf("1e-45")
IDENTITY_LIMIT = mp.mpf("1e-55")

# This grid is the complete predeclared response atlas: four Q4 states and two
# inverse-designed W8 alternatives, four correlated scales, and three waves.
DENSITY_RATIOS = ("0.37", "1", "2.75", "7.25")
W8_TARGETS = ("0.90", "0.93")
TARGETS = W8_TARGETS
SCALES = ("0.5", "1", "1.3", "2")
WAVE_NUMBERS_MEV = ("0", "100", "200")
EXPECTED_ROW_COUNT = (len(DENSITY_RATIOS) + len(W8_TARGETS)) * len(SCALES) * len(WAVE_NUMBERS_MEV)

BASELINE_BINDING_MEV = "-16"
CALIBRATION_K_MEV = "240"

HESSIAN_KEYS = ("a", "b", "d0", "g", "h", "t0")
STATE_COMPARISON_KEYS = (
    "energy_total",
    "pressure_total",
    "mu",
    "residual",
    "C_y",
    "B_y",
    "D",
    "K",
    "cs2",
)
DERIVATIVE_KEYS = HESSIAN_KEYS


class SpatialIdentifiabilityError(ValueError):
    """Fail-closed error for malformed inputs or incomplete response evidence."""


def _precision(dps: Any) -> int:
    if isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 200:
        raise SpatialIdentifiabilityError("dps must be an integer in [80,200]")
    return dps


def _mp(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> mp.mpf:
    if isinstance(value, bool):
        raise SpatialIdentifiabilityError(f"{name} must be a finite real, not bool")
    try:
        # Preserve an existing high-precision mpf.  Decimal strings are parsed
        # under the caller's explicit workdps and never through binary float.
        result = value if isinstance(value, mp.mpf) else mp.mpf(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SpatialIdentifiabilityError(f"{name} must be a finite real") from exc
    if not mp.isfinite(result):
        raise SpatialIdentifiabilityError(f"{name} must be finite")
    if positive and result <= 0:
        raise SpatialIdentifiabilityError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise SpatialIdentifiabilityError(f"{name} must be nonnegative")
    return result


def _finite(value: Any) -> bool:
    try:
        return bool(mp.isfinite(value if isinstance(value, mp.mpf) else mp.mpf(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _number(value: Any, digits: int = 70) -> str:
    value = _mp(value, "derived number")
    with mp.workdps(max(mp.mp.dps, digits + 12)):
        return mp.nstr(value, digits)


def _relative(left: Any, right: Any) -> mp.mpf:
    left, right = _mp(left, "left"), _mp(right, "right")
    return abs(left - right) / max(abs(left), abs(right), mp.mpf(1))


def _shown(value: Any) -> Any:
    """Convert numeric evidence to finite JSON-safe decimal strings."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, mp.matrix):
        return [[_shown(value[i, j]) for j in range(value.cols)] for i in range(value.rows)]
    if isinstance(value, mp.mpf):
        return _number(value)
    if isinstance(value, mp.mpc):
        return {"real": _number(value.real), "imag": _number(value.imag)}
    if isinstance(value, Mapping):
        return {str(key): _shown(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shown(item) for item in value]
    if isinstance(value, float):
        if not _finite(value):
            raise SpatialIdentifiabilityError("nonfinite floating-point output")
        return value
    return value


def parse_high_precision(value: Any, dps: int = CONTROL_DPS) -> mp.mpf:
    """Parse a serialized decimal while an explicit high-precision context is active."""
    dps = _precision(dps)
    with mp.workdps(dps):
        parsed = _mp(str(value), "serialized decimal")
        if not mp.isfinite(parsed):
            raise SpatialIdentifiabilityError("serialized decimal is not finite")
        return parsed


class ScaledResponseModel(upstream.BulkModel):
    """Actual correlated W0 rescaling, including W8 polynomial preservation.

    The homogeneous potential is expressed in y=W/W0.  For a quartic model
    lambda is transformed as s^-4; for an inverse W8 model the polynomial
    coefficients in y are copied unchanged.  Every dimensional invariant is
    recomputed from this candidate's actual inputs.
    """

    def __init__(self, scale: Any, reference: upstream.BulkModel):
        scale = _mp(scale, "scale", positive=True)
        polynomial = None if reference.polynomial is None else tuple(reference.polynomial)
        super().__init__(
            lam=reference.lam / scale**4,
            gomega=reference.gomega,
            polynomial=polynomial,
        )
        self.scale = scale
        self.W0 = reference.W0 * scale
        self.lam = reference.lam / scale**4
        self.MN = reference.MN
        self.momega = reference.momega
        self.gomega = reference.gomega
        self.hbarc = reference.hbarc
        self.n0_fm3 = reference.n0_fm3
        self.n0 = reference.n0
        self.d = reference.d
        self.A = self.lam * self.W0**4
        self.Cv = self.gomega**2 / self.momega**2
        self.Cs = self.MN**2 / (2 * self.A)


def _assert_baseline_contract(base: upstream.BulkModel) -> dict[str, Any]:
    """Reuse the accepted foundation passport/input guard, not a duplicate."""
    passport, passport_check = foundation._assert_baseline_contract(base)
    if not passport_check.get("pass"):
        raise ArithmeticError("foundation baseline passport validation failed")
    return {
        "guard_source": "verification/nvg_foundation_identifiability_audit.py",
        "passport_validation_passed": True,
        "passport_schema_version": passport.get("schema_version"),
    }


def _state_observable_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, mp.mpf]:
    return {key: _relative(left[key], right[key]) for key in STATE_COMPARISON_KEYS}


def _coerce_hessian(coefficients: Mapping[str, Any]) -> dict[str, mp.mpf]:
    if not isinstance(coefficients, Mapping) or not set(HESSIAN_KEYS) <= set(coefficients):
        raise SpatialIdentifiabilityError(f"coefficients must contain {HESSIAN_KEYS}")
    values = {key: _mp(coefficients[key], key) for key in HESSIAN_KEYS}
    _mp(values["a"], "a", positive=True)
    _mp(values["g"], "g", nonnegative=True)
    _mp(values["t0"], "t0", positive=True)
    return values


def _full_hessian_response(coefficients: Mapping[str, Any], Z: Any, k: Any) -> dict[str, Any]:
    """Evaluate H, its Schur complement, and an independent direct solve."""
    c = _coerce_hessian(coefficients)
    Z = _mp(Z, "Z", positive=True)
    k = _mp(k, "k", nonnegative=True)
    a, b, d0, g, h, t0 = (c[key] for key in HESSIAN_KEYS)
    L = t0 + k**2
    D = a + g**2 / L
    B = b + g * h / L
    C = d0 + h**2 / L + Z * k**2
    # Keep an explicit null for the undefined Schur complement at C=0.  This
    # lets callers classify a singular input without manufacturing NaN output
    # or accidentally treating it as a susceptibility.
    S = D - B**2 / C if C != 0 else None
    H = mp.matrix([[a, b, g], [b, d0 + Z * k**2, h], [g, h, -L]])
    determinant = mp.det(H)
    direct = None
    direct_status = "DIRECT_SOLVE_OK"
    try:
        if determinant == 0:
            raise ZeroDivisionError("singular Hessian")
        solution = mp.lu_solve(H, mp.matrix([1, 0, 0]))
        direct = tuple(solution[i] for i in range(3))
        if not all(_finite(item) for item in direct):
            direct_status = "DIRECT_SOLVE_NONFINITE"
            direct = None
    except (ZeroDivisionError, ValueError, ArithmeticError):
        direct_status = "DIRECT_SOLVE_SINGULAR"

    if C == 0:
        status = "SINGULAR_SCALAR_CURVATURE"
    elif C < 0:
        status = "UNSTABLE_SCALAR_CURVATURE"
    elif S == 0:
        status = "SINGULAR_DENSITY_RESPONSE"
    elif S is not None and S < 0:
        status = "UNSTABLE_DENSITY_RESPONSE"
    elif direct is None:
        status = "SINGULAR_FULL_HESSIAN"
    else:
        status = "STABLE_STATIC_TF_RESPONSE"
    stable = bool(C > 0 and S is not None and S > 0 and direct is not None)
    chi = 1 / S if stable else None
    direct_chi = direct[0] if stable else None
    direct_relative_error = None if not stable else _relative(direct_chi, chi)
    return {
        "coefficients": c,
        "Z": Z,
        "k": k,
        "L": L,
        "D": D,
        "B": B,
        "C": C,
        "S": S,
        "chi": chi,
        "H": H,
        "determinant": determinant,
        "direct_solution": direct,
        "direct_status": direct_status,
        "direct_chi": direct_chi,
        "direct_relative_error": direct_relative_error,
        "stable": stable,
        "status": status,
    }


def response_from_hessian(coefficients: Mapping[str, Any], Z: Any, k: Any) -> dict[str, Any]:
    """Public finite-real static response helper; no JSON coercion is applied."""
    # Serialized rows are decimal strings.  Give a direct API caller a
    # proper high-precision parse even when the ambient mpmath context is the
    # default 15 digits; existing mpf inputs are still preserved by _mp.
    with mp.workdps(max(mp.mp.dps, CONTROL_DPS)):
        return _full_hessian_response(coefficients, Z, k)


# A concise alias used by downstream focused checks.
static_response = response_from_hessian


def _full_saddle_derivatives(model: upstream.BulkModel, n: mp.mpf, y: mp.mpf, A0: mp.mpf, dps: int):
    """Reuse the maintained independent nested derivative implementation."""
    return nuclear._full_saddle_derivatives(model, n, y, A0, dps)


def _background_coefficients(
    model: upstream.BulkModel,
    state: Mapping[str, Any],
    dps: int,
    *,
    check_derivative: bool = True,
) -> dict[str, Any]:
    n, y = state["n"], state["y"]
    f = model.fermi(n, y)
    A0 = model.gomega * n / (model.momega**2 * y**2)
    a = f["k"]**2 / (3 * n * f["ef"])
    b = model.MN**2 * y / f["ef"]
    d0 = model.MN**2 * f["ns_m"] + model.potential(y)[2] - model.momega**2 * A0**2
    g = model.gomega
    h = -2 * model.momega**2 * y * A0
    t0 = model.momega**2 * y**2
    analytic = {"a": a, "b": b, "d0": d0, "g": g, "h": h, "t0": t0}
    # Existing helper names the y-y entry d; it is the same d0 here.  The
    # expensive nested differentiation is intentionally done once for each
    # physical background.  Correlated scale rows reuse the analytic values
    # after independently recomputing their candidate inputs.
    if check_derivative:
        independent_raw = _full_saddle_derivatives(model, n, y, A0, dps)
        independent = {"a": independent_raw["a"], "b": independent_raw["b"],
                       "d0": independent_raw["d"], "g": independent_raw["g"],
                       "h": independent_raw["h"], "t0": independent_raw["t0"]}
    else:
        independent = dict(analytic)
    derivative_errors = {key: _relative(independent[key], analytic[key]) for key in DERIVATIVE_KEYS}
    return {
        "analytic": analytic,
        "independent": independent,
        "relative_errors": derivative_errors,
        "passed": bool(all(error <= IDENTITY_LIMIT for error in derivative_errors.values())),
        "A0": A0,
        "Z": model.W0**2,
        "n": n,
        "y": y,
        "kF": f["k"],
    }


def _blind_wave_number(coefficients: Mapping[str, Any]) -> dict[str, Any]:
    c = _coerce_hessian(coefficients)
    b, g, h, t0 = c["b"], c["g"], c["h"], c["t0"]
    if b == 0:
        return {"status": "NO_BLIND_POINT_B_ZERO", "k_blind": None, "k_squared": None,
                "B_at_k_blind": None}
    k_squared = -g * h / b - t0
    if k_squared > 0:
        k_blind = mp.sqrt(k_squared)
        B_at = b + g * h / (t0 + k_blind**2)
        status = "POSITIVE_FINITE_BLIND_WAVE"
    elif k_squared == 0:
        k_blind = mp.mpf(0)
        B_at = b + g * h / t0
        status = "K0_BLIND_SCALE_POINT"
    else:
        k_blind = None
        B_at = None
        status = "NO_POSITIVE_BLIND_WAVE"
    return {"status": status, "k_blind": k_blind, "k_squared": k_squared,
            "B_at_k_blind": B_at}


blind_wave_number = _blind_wave_number


def invert_gradient_coefficient(coefficients: Mapping[str, Any], k: Any, chi: Any) -> mp.mpf:
    """Reconstruct Z_s from a nondegenerate stable response row.

    Required conditions are k>0, B(k)!=0, stable C/S, and two numerically
    separated subtractions.  Refusal is intentional at k=0, at a blind point,
    or when either subtraction cannot retain the declared precision.  These
    are numerical precision guards, not estimates of experimental uncertainty.
    """
    with mp.workdps(max(mp.mp.dps, CONTROL_DPS)):
        c = _coerce_hessian(coefficients)
        k = _mp(k, "k", nonnegative=True)
        chi = _mp(chi, "chi", positive=True)
        if k <= 0:
            raise SpatialIdentifiabilityError("gradient inversion is scale-blind at k=0")
        L = c["t0"] + k**2
        D = c["a"] + c["g"]**2 / L
        B = c["b"] + c["g"] * c["h"] / L
        if B == 0:
            raise SpatialIdentifiabilityError("gradient inversion is degenerate at B(k)=0")
        denominator = D - 1 / chi
        if not _finite(denominator) or denominator <= 0:
            raise SpatialIdentifiabilityError("gradient inversion denominator is nonpositive or singular")
        condition_ratio = abs(denominator) / max(abs(D), abs(1 / chi), mp.mpf("1e-100"))
        if condition_ratio <= PRECISION_LIMIT:
            raise SpatialIdentifiabilityError("gradient inversion denominator is numerically ill-conditioned")
        base = c["d0"] + c["h"]**2 / L
        reconstructed_base = B**2 / denominator
        separation = reconstructed_base - base
        subtraction_conditioning_ratio = abs(separation) / max(
            abs(reconstructed_base), abs(base), mp.mpf("1e-100")
        )
        if subtraction_conditioning_ratio <= PRECISION_LIMIT:
            raise SpatialIdentifiabilityError(
                "gradient inversion C subtraction is numerically ill-conditioned"
            )
        result = separation / k**2
        if not _finite(result) or result <= 0:
            raise SpatialIdentifiabilityError("reconstructed gradient coefficient is not positive finite")
        return result


reconstruct_gradient_coefficient = invert_gradient_coefficient


def _inversion_record(coefficients: Mapping[str, Any], physical_Z: Any, response: Mapping[str, Any]) -> dict[str, Any]:
    k = response["k"]
    if not response["stable"]:
        return {"status": "REFUSED_UNSTABLE_OR_SINGULAR", "reconstructed_Z": None,
                "relative_error": None, "conditioning_ratio": None,
                "subtraction_conditioning_ratio": None, "reason": response["status"]}
    D = response["D"]
    denominator = D - 1 / response["chi"]
    conditioning_ratio = abs(denominator) / max(abs(D), abs(1 / response["chi"]), mp.mpf("1e-100"))
    subtraction_conditioning_ratio = None
    if k > 0 and denominator > 0 and response["B"] != 0:
        c = _coerce_hessian(coefficients)
        base = c["d0"] + c["h"]**2 / response["L"]
        reconstructed_base = response["B"]**2 / denominator
        separation = reconstructed_base - base
        subtraction_conditioning_ratio = abs(separation) / max(
            abs(reconstructed_base), abs(base), mp.mpf("1e-100")
        )
    try:
        value = invert_gradient_coefficient(coefficients, k, response["chi"])
    except SpatialIdentifiabilityError as exc:
        reason = str(exc)
        if k == 0:
            status = "REFUSED_K0_SCALE_BLIND"
        elif "ill-conditioned" in reason:
            status = "REFUSED_ILL_CONDITIONED"
        else:
            status = "REFUSED_B_BLIND_OR_DEGENERATE"
        return {"status": status, "reconstructed_Z": None, "relative_error": None,
                "conditioning_ratio": conditioning_ratio,
                "subtraction_conditioning_ratio": subtraction_conditioning_ratio,
                "reason": reason}
    return {"status": "RECONSTRUCTED_NONDEGENERATE", "reconstructed_Z": value,
            "relative_error": _relative(value, physical_Z),
            "conditioning_ratio": conditioning_ratio,
            "subtraction_conditioning_ratio": subtraction_conditioning_ratio,
            "reason": None}


def _canonical_vacuum_mass(model: upstream.BulkModel) -> tuple[mp.mpf, mp.mpf]:
    """Return the physical canonical vacuum mass and U_yy(1).

    ``lambda`` and ``A`` are physical potential parameters only for Q4.  The
    W8 branch uses its retained polynomial directly, so its mass must come
    from the actual potential curvature in the canonical field W=W0*y.
    """
    W0 = _mp(model.W0, "W0", positive=True)
    _, _, curvature = model.potential(mp.mpf(1))
    curvature = _mp(curvature, "vacuum potential curvature Uyy", positive=True)
    mass_squared = curvature / W0**2
    mass_squared = _mp(mass_squared, "canonical vacuum mass squared", positive=True)
    mass = mp.sqrt(mass_squared)
    if not _finite(mass):
        raise SpatialIdentifiabilityError("canonical vacuum mass is not finite")
    return mass, curvature


def _parameter_record(reference: upstream.BulkModel, candidate: ScaledResponseModel) -> dict[str, Any]:
    gs0, gss = reference.MN / reference.W0, candidate.MN / candidate.W0
    q0, qs = reference.momega / reference.W0, candidate.momega / candidate.W0
    family = "Q4" if reference.polynomial is None else "W8"
    ms0, reference_curvature = _canonical_vacuum_mass(reference)
    mss, candidate_curvature = _canonical_vacuum_mass(candidate)
    z0, zs = reference.W0**2, candidate.W0**2
    ratio = {
        "W0": candidate.W0 / reference.W0,
        "g_s": gss / gs0,
        "q_phi": qs / q0,
        "m_sigma": mss / ms0,
        "gradient_coefficient": zs / z0,
        "Cv": candidate.Cv / reference.Cv,
    }
    expected = {
        "W0": candidate.scale,
        "g_s": candidate.scale**-1,
        "q_phi": candidate.scale**-1,
        "m_sigma": candidate.scale**-1,
        "gradient_coefficient": candidate.scale**2,
        "Cv": mp.mpf(1),
    }
    polynomial_same = (reference.polynomial is None and candidate.polynomial is None) or (
        reference.polynomial is not None and candidate.polynomial == reference.polynomial
    )
    if family == "Q4":
        ratio.update({"lambda": candidate.lam / reference.lam, "A": candidate.A / reference.A})
        expected.update({"lambda": candidate.scale**-4, "A": mp.mpf(1)})
        lambda_value = candidate.lam
        homogeneous_A = candidate.lam * candidate.W0**4
        potential_metadata = {
            "representation": "quartic_z2",
            "lambda_used_by_potential": True,
            "A_used_by_potential": True,
            "polynomial_coefficients_in_y": None,
            "vacuum_curvature_Uyy_MeV4": candidate_curvature,
            "reference_vacuum_curvature_Uyy_MeV4": reference_curvature,
            "unused_quartic_auxiliary": None,
        }
    else:
        ratio.update({"lambda": None, "A": None})
        expected.update({"lambda": None, "A": None})
        lambda_value = None
        homogeneous_A = None
        auxiliary = {
            "lambda": candidate.lam,
            "A_MeV4": candidate.A,
            "meaning": "unused BulkModel quartic auxiliaries; W8 potential is the retained polynomial",
        }
        potential_metadata = {
            "representation": "polynomial_z2_z3_z4",
            "lambda_used_by_potential": False,
            "A_used_by_potential": False,
            "polynomial_coefficients_in_y": candidate.polynomial,
            "vacuum_curvature_Uyy_MeV4": candidate_curvature,
            "reference_vacuum_curvature_Uyy_MeV4": reference_curvature,
            "unused_quartic_auxiliary": auxiliary,
        }
    return {
        "scale": candidate.scale,
        "potential_family": family,
        "potential_representation": potential_metadata["representation"],
        "W0_MeV": candidate.W0,
        "lambda": lambda_value,
        "g_s": gss,
        "q_phi": qs,
        "m_sigma_MeV": mss,
        "gradient_coefficient_MeV2": zs,
        "ratios": ratio,
        "expected_ratios": expected,
        "homogeneous_A_from_actual_inputs": homogeneous_A,
        "homogeneous_Cv_from_actual_inputs": candidate.gomega**2 / candidate.momega**2,
        "polynomial_coefficients_in_y_unchanged": polynomial_same,
        "potential_metadata": potential_metadata,
        "physical_fixed_inputs": {
            "M_N_MeV": reference.MN,
            "m_omega_MeV": reference.momega,
            "g_omega": reference.gomega,
            "degeneracy": int(reference.d),
        },
    }


def _wrong_mixed_hessian_control(background: Mapping[str, Any], dps: int) -> dict[str, Any]:
    c = background["coefficients"]
    actual = c["analytic"]
    physical = _full_hessian_response(actual, background["model"].W0**2, mp.mpf("100"))
    wrong = dict(actual)
    wrong["h"] = -actual["h"]
    omitted = dict(actual)
    omitted["h"] = mp.mpf(0)
    wrong_response = _full_hessian_response(wrong, background["model"].W0**2, mp.mpf("100"))
    omitted_response = _full_hessian_response(omitted, background["model"].W0**2, mp.mpf("100"))
    mismatch = abs(actual["h"] - wrong["h"])
    wrong_chi_error = None
    omitted_chi_error = None
    if physical["chi"] is not None and wrong_response["chi"] is not None:
        wrong_chi_error = _relative(wrong_response["chi"], physical["chi"])
    if physical["chi"] is not None and omitted_response["chi"] is not None:
        omitted_chi_error = _relative(omitted_response["chi"], physical["chi"])
    detected = bool(mismatch > IDENTITY_LIMIT and (
        (wrong_chi_error is not None and wrong_chi_error > IDENTITY_LIMIT)
        or (omitted_chi_error is not None and omitted_chi_error > IDENTITY_LIMIT)
    ))
    return {
        "physical_h": actual["h"],
        "wrong_sign_h": wrong["h"],
        "omitted_h": omitted["h"],
        "wrong_sign_derivative_mismatch": mismatch,
        "wrong_sign_response_relative_error": wrong_chi_error,
        "omitted_response_relative_error": omitted_chi_error,
        "detected_and_rejected": detected,
        "negative_control_precision_digits": dps,
    }


def _make_backgrounds(base: upstream.BulkModel, dps: int) -> list[dict[str, Any]]:
    backgrounds: list[dict[str, Any]] = []
    for ratio_text in DENSITY_RATIOS:
        ratio = _mp(ratio_text, "density_ratio", positive=True)
        n = base.n0 * ratio
        state = base.equilibrium(n)
        backgrounds.append({
            "background_id": f"Q4:n_over_n0={ratio_text}",
            "family": "Q4",
            "density_ratio": ratio_text,
            "target_y": None,
            "model": base,
            "state": state,
            "inverse_design": None,
        })
    for target_text in W8_TARGETS:
        model, _, jet = upstream.inverse_potential_jet(
            target_text, CALIBRATION_K_MEV, "0.16", BASELINE_BINDING_MEV
        )
        y = _mp(target_text, "target_y", positive=True)
        state = model.state(model.n0, y)
        backgrounds.append({
            "background_id": f"W8:y_star={target_text}",
            "family": "W8",
            "density_ratio": "1",
            "target_y": target_text,
            "model": model,
            "state": state,
            "inverse_design": jet,
        })
    for background in backgrounds:
        model, state = background["model"], background["state"]
        if state["C_y"] <= 0:
            raise ArithmeticError(f"background {background['background_id']} has C_y<=0")
        coeff = _background_coefficients(model, state, dps)
        if not coeff["passed"]:
            raise ArithmeticError(f"full Hessian derivative check failed for {background['background_id']}")
        background["coefficients"] = coeff
        blind = _blind_wave_number(coeff["analytic"])
        if blind["k_blind"] is not None:
            blind_response = _full_hessian_response(coeff["analytic"], model.W0**2, blind["k_blind"])
            blind.update({
                "response_status": blind_response["status"],
                "response_stable": blind_response["stable"],
                "response_chi": blind_response["chi"],
            })
        background["blind"] = blind
    return backgrounds


def _build_snapshot(dps: int) -> dict[str, Any]:
    dps = _precision(dps)
    with mp.workdps(dps):
        base = upstream.BulkModel()
        guard = _assert_baseline_contract(base)
        backgrounds = _make_backgrounds(base, dps)
        rows: list[dict[str, Any]] = []
        scale_laws: list[dict[str, Any]] = []
        scale_maps: dict[str, list[dict[str, Any]]] = {}
        k0_matches: list[dict[str, Any]] = []
        negative_controls: list[dict[str, Any]] = []

        for background in backgrounds:
            ref_model = background["model"]
            coeff = background["coefficients"]["analytic"]
            n, y = background["state"]["n"], background["state"]["y"]
            map_rows: list[dict[str, Any]] = []
            for scale_text in SCALES:
                candidate = ScaledResponseModel(scale_text, ref_model)
                candidate_state = candidate.state(n, y)
                state_diff = _state_observable_difference(candidate_state, background["state"])
                candidate_coeff = _background_coefficients(
                    candidate, candidate_state, dps, check_derivative=False
                )
                coeff_diff = {
                    key: _relative(candidate_coeff["analytic"][key], coeff[key])
                    for key in HESSIAN_KEYS
                }
                map_rows.append({
                    "scale_label": scale_text,
                    "model": candidate,
                    "state": candidate_state,
                    "state_relative_differences": state_diff,
                    "coefficient_relative_differences": coeff_diff,
                    "coefficients": candidate_coeff["analytic"],
                    "parameter": _parameter_record(ref_model, candidate),
                    "homogeneous_state_invariant": bool(all(v <= IDENTITY_LIMIT for v in state_diff.values())),
                    "homogeneous_hessian_invariant": bool(all(v <= IDENTITY_LIMIT for v in coeff_diff.values())),
                })
            scale_maps[background["background_id"]] = map_rows

            # Full 72-row atlas is emitted without dropping red or marginal rows.
            by_k: dict[str, list[dict[str, Any]]] = {k: [] for k in WAVE_NUMBERS_MEV}
            for map_row in map_rows:
                candidate = map_row["model"]
                ccoeff = map_row["coefficients"]
                for k_text in WAVE_NUMBERS_MEV:
                    response = _full_hessian_response(ccoeff, candidate.W0**2, _mp(k_text, "k", nonnegative=True))
                    inversion = _inversion_record(ccoeff, candidate.W0**2, response)
                    row = {
                        "row_id": f"{background['background_id']}:scale={map_row['scale_label']}:k={k_text}",
                        "background_id": background["background_id"],
                        "family": background["family"],
                        "density_ratio": background["density_ratio"],
                        "target_y": background["target_y"],
                        "scale": map_row["scale_label"],
                        "k_MeV": _mp(k_text, "k", nonnegative=True),
                        "model": candidate,
                        "coefficients": ccoeff,
                        "response": response,
                        "inversion": inversion,
                        "state": map_row["state"],
                    }
                    rows.append(row)
                    by_k[k_text].append(row)

            for k_text, k_rows in by_k.items():
                k = _mp(k_text, "k", nonnegative=True)
                pair_rows: list[dict[str, Any]] = []
                for i, first in enumerate(k_rows):
                    for second in k_rows[i + 1:]:
                        r1, r2 = first["response"], second["response"]
                        s1, s2 = _mp(first["scale"], "s1"), _mp(second["scale"], "s2")
                        B = r1["B"]
                        C1, C2 = r1["C"], r2["C"]
                        expected_delta = B**2 * ref_model.W0**2 * k**2 * (s2**2 - s1**2) / (C1 * C2)
                        actual_delta = r2["S"] - r1["S"]
                        delta_error = _relative(actual_delta, expected_delta)
                        chi_delta = None
                        if r1["chi"] is not None and r2["chi"] is not None:
                            chi_delta = r2["chi"] - r1["chi"]
                        pair_rows.append({
                            "s1": s1,
                            "s2": s2,
                            "S_difference": actual_delta,
                            "expected_S_difference": expected_delta,
                            "relative_error": delta_error,
                            "dS_dscale2_nonnegative": bool(actual_delta >= -IDENTITY_LIMIT),
                            "chi_difference": chi_delta,
                            "dchi_dscale2_nonpositive": None if chi_delta is None else bool(chi_delta <= IDENTITY_LIMIT),
                            "pass": bool(delta_error <= PRECISION_LIMIT and actual_delta >= -IDENTITY_LIMIT),
                        })
                derivative_rows = []
                for item in k_rows:
                    r = item["response"]
                    derivative = r["B"]**2 * ref_model.W0**2 * k**2 / r["C"]**2 if r["C"] != 0 else mp.nan
                    dchi = None if r["chi"] is None else -derivative * r["chi"]**2
                    derivative_rows.append({
                        "scale": item["scale"],
                        "dS_dscale2": derivative,
                        "dchi_dscale2": dchi,
                        "nonnegative_S_derivative": bool(_finite(derivative) and derivative >= 0),
                        "nonpositive_chi_derivative_on_stable_domain": None if dchi is None else bool(dchi <= IDENTITY_LIMIT),
                    })
                scale_laws.append({
                    "background_id": background["background_id"],
                    "k_MeV": k,
                    "pairs": pair_rows,
                    "derivatives": derivative_rows,
                    "all_pairs_pass": bool(all(pair["pass"] for pair in pair_rows)),
                })

            k0 = next(item for item in rows if item["background_id"] == background["background_id"] and item["scale"] == "1" and item["k_MeV"] == 0)
            k0_response = k0["response"]
            thermodynamic = background["state"]["mu_prime"]
            k0_matches.append({
                "background_id": background["background_id"],
                "S_k0": k0_response["S"],
                "dmu_dn_from_homogeneous_state": thermodynamic,
                "S_relative_error": _relative(k0_response["S"], thermodynamic),
                "chi_k0": k0_response["chi"],
                "inverse_thermodynamic_curvature": 1 / thermodynamic,
                "chi_relative_error": _relative(k0_response["chi"], 1 / thermodynamic),
                "pass": bool(_relative(k0_response["S"], thermodynamic) <= IDENTITY_LIMIT),
                "ensemble_note": "grand-canonical long-wavelength limit; exact uniform mode excluded at fixed total N",
            })
            negative_controls.append({
                "background_id": background["background_id"],
                **_wrong_mixed_hessian_control(background, dps),
            })

        if len(rows) != EXPECTED_ROW_COUNT:
            raise ArithmeticError(f"response atlas has {len(rows)} rows, expected {EXPECTED_ROW_COUNT}")
        return {
            "dps": dps,
            "base": base,
            "baseline_guard": guard,
            "backgrounds": backgrounds,
            "scale_maps": scale_maps,
            "rows": rows,
            "scale_laws": scale_laws,
            "k0_matches": k0_matches,
            "negative_controls": negative_controls,
        }


def _inverse_design_serialized(background: Mapping[str, Any]) -> dict[str, Any] | None:
    jet = background.get("inverse_design")
    if jet is None:
        return None
    state = background["state"]
    model = background["model"]
    coeff = jet.get("coefficients")
    if coeff is None:
        raise ArithmeticError("inverse W8 design did not expose coefficients")
    return {
        "target_y": background["target_y"],
        "training_targets": {
            "n0_fm3": "0.16",
            "binding_MeV": BASELINE_BINDING_MEV,
            "K_MeV": CALIBRATION_K_MEV,
            "pressure_MeV_fm3": "0",
        },
        "coefficients_z2_z3_z4_MeV4": [_number(value) for value in coeff],
        "reconstructed_state": {
            "y": _number(state["y"]),
            "binding_MeV": _number(state["energy_total"] / state["n"] - model.MN),
            "pressure_MeV_fm3": _number(state["pressure_total"] / model.hbarc**3),
            "K_MeV": _number(state["K"]),
            "stationarity_residual": _number(state["residual"]),
        },
        "not_adopted_physical_model": True,
        "not_held_out_evidence": True,
    }


def _serialize_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    base = snapshot["base"]
    backgrounds_out = []
    for background in snapshot["backgrounds"]:
        state, model, coeff = background["state"], background["model"], background["coefficients"]
        state_payload = {key: state[key] for key in (
            "y", "energy_total", "pressure_total", "mu", "residual",
            "C_y", "B_y", "D", "K", "cs2",
        )}
        # Upstream state has no W_MeV key; include dimensional W explicitly.
        state_payload["W_MeV"] = model.W0 * state["y"]
        state_payload["pressure_MeV_fm3"] = state["pressure_total"] / model.hbarc**3
        state_payload["binding_MeV"] = state["energy_total"] / state["n"] - model.MN
        state_payload.update({
            "energy_density_MeV4": state["energy_total"],
            "pressure_MeV4": state["pressure_total"],
            "mu_MeV": state["mu"],
            "residual_y_MeV4": state["residual"],
            "C_y_MeV4": state["C_y"],
            "D_nn_MeVminus2": state["D"],
            "K_MeV": state["K"],
        })
        backgrounds_out.append({
            "background_id": background["background_id"],
            "family": background["family"],
            "density_ratio": background["density_ratio"],
            "target_y": background["target_y"],
            "n_fm3": _number(base.n0_fm3 * _mp(background["density_ratio"], "density_ratio")),
            "state": _shown(state_payload),
            "hessian_coefficients": _shown(coeff["analytic"]),
            "full_hessian_derivative_check": {
                "analytic": _shown(coeff["analytic"]),
                "independent_nested_derivatives": _shown(coeff["independent"]),
                "relative_errors": _shown(coeff["relative_errors"]),
                "passed": coeff["passed"],
            },
            "blind_wave": _shown({
                **background["blind"],
                "kF_MeV": coeff["kF"],
                "B_at_k0": coeff["analytic"]["b"] + coeff["analytic"]["g"] * coeff["analytic"]["h"] / coeff["analytic"]["t0"],
                "dy_dn_at_k0": -(
                    coeff["analytic"]["b"] + coeff["analytic"]["g"] * coeff["analytic"]["h"] / coeff["analytic"]["t0"]
                ) / (coeff["analytic"]["d0"] + coeff["analytic"]["h"]**2 / coeff["analytic"]["t0"]),
                "dy_dn_sign_connection": -coeff["analytic"]["b"] - coeff["analytic"]["g"] * coeff["analytic"]["h"] / coeff["analytic"]["t0"],
            }),
            "inverse_design": _inverse_design_serialized(background),
        })

    scale_maps_out = {}
    for background_id, map_rows in snapshot["scale_maps"].items():
        scale_maps_out[background_id] = []
        for item in map_rows:
            scale_maps_out[background_id].append({
                "scale": item["scale_label"],
                "parameter": _shown(item["parameter"]),
                "state_relative_differences": _shown(item["state_relative_differences"]),
                "coefficient_relative_differences": _shown(item["coefficient_relative_differences"]),
                "homogeneous_state_invariant": item["homogeneous_state_invariant"],
                "homogeneous_hessian_invariant": item["homogeneous_hessian_invariant"],
            })

    rows_out = []
    for row in snapshot["rows"]:
        response = row["response"]
        rows_out.append({
            "row_id": row["row_id"],
            "background_id": row["background_id"],
            "family": row["family"],
            "density_ratio": row["density_ratio"],
            "target_y": row["target_y"],
            "scale": row["scale"],
            "k_MeV": _number(row["k_MeV"]),
            "Z_MeV2": _number(response["Z"]),
            "coefficients": _shown(row["coefficients"]),
            "L_MeV2": _number(response["L"]),
            "D_MeVminus2": _number(response["D"]),
            "B_MeV": _number(response["B"]),
            "C_MeV4": _number(response["C"]),
            "S_MeVminus2": None if response["S"] is None else _number(response["S"]),
            "chi_MeV2": None if response["chi"] is None else _number(response["chi"]),
            "H": _shown(response["H"]),
            "determinant": _number(response["determinant"]),
            "direct_solution": None if response["direct_solution"] is None else _shown(response["direct_solution"]),
            "direct_status": response["direct_status"],
            "direct_vs_schur_relative_error": None if response["direct_relative_error"] is None else _number(response["direct_relative_error"]),
            "stable": response["stable"],
            "status": response["status"],
            "gradient_inversion": _shown(row["inversion"]),
        })

    scale_laws_out = []
    for item in snapshot["scale_laws"]:
        scale_laws_out.append({
            "background_id": item["background_id"],
            "k_MeV": _number(item["k_MeV"]),
            "pairs": _shown(item["pairs"]),
            "derivatives": _shown(item["derivatives"]),
            "all_pairs_pass": item["all_pairs_pass"],
        })

    k0_out = _shown(snapshot["k0_matches"])
    negatives_out = _shown(snapshot["negative_controls"])
    return {
        "dps": snapshot["dps"],
        "inputs_and_scope": {
            "natural_units": True,
            "n_unit": "MeV^3",
            "k_unit": "MeV",
            "Z_unit": "MeV^2",
            "chi_unit": "MeV^2",
            "density_ratios_Q4": list(DENSITY_RATIOS),
            "W8_target_y": list(W8_TARGETS),
            "scales": list(SCALES),
            "wave_numbers_MeV": list(WAVE_NUMBERS_MEV),
            "expected_row_count": EXPECTED_ROW_COUNT,
            "source_definition": "chi=delta_n/delta_mu_ext for source -mu_ext*n",
            "opposite_source_definition": "delta_n/delta_V_ext=-chi for source +V_ext*n",
            "new_interaction_adopted": False,
            "calibration_targets_training_only": True,
            "not_full_RPA_or_dynamics": True,
            "not_finite_nucleus_or_transport": True,
            "high_k_rows_formal_local_TF_only": True,
            "fixed_total_N_uniform_mode_excluded": True,
            "evidence_weight": EVIDENCE_WEIGHT,
        },
        "baseline_guard": snapshot["baseline_guard"],
        "backgrounds": backgrounds_out,
        "scale_maps": scale_maps_out,
        "rows": rows_out,
        "scale_laws": scale_laws_out,
        "k0_thermodynamic_matches": k0_out,
        "negative_controls": negatives_out,
        "coverage": {
            "row_count": len(rows_out),
            "expected_row_count": EXPECTED_ROW_COUNT,
            "all_rows_present": len(rows_out) == EXPECTED_ROW_COUNT,
            "families": ["Q4", "W8"],
            "all_statuses_retained": True,
        },
    }


def _precision_probes(snapshot: Mapping[str, Any]) -> dict[str, mp.mpf]:
    probes: dict[str, mp.mpf] = {}
    for background in snapshot["backgrounds"]:
        bid = background["background_id"]
        state = background["state"]
        for key in ("y", "energy_total", "pressure_total", "mu", "C_y", "B_y", "D", "K", "cs2"):
            probes[f"{bid}.state.{key}"] = state[key]
        for key in HESSIAN_KEYS:
            probes[f"{bid}.hessian.{key}"] = background["coefficients"]["analytic"][key]
    for row in snapshot["rows"]:
        label = row["row_id"]
        for key in ("D", "B", "C", "S", "determinant"):
            if row["response"][key] is not None:
                probes[f"{label}.{key}"] = row["response"][key]
        if row["response"]["chi"] is not None:
            probes[f"{label}.chi"] = row["response"]["chi"]
    return probes


def _precision_compare(primary: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    a, b = _precision_probes(primary), _precision_probes(control)
    if set(a) != set(b):
        raise ArithmeticError("80/110 precision probe key set changed")
    errors = {key: _relative(a[key], b[key]) for key in a}
    maximum = max(errors.values(), default=mp.mpf(0))
    if maximum > PRECISION_LIMIT:
        bad = max(errors, key=errors.get)
        raise ArithmeticError(f"80/110 precision mismatch at {bad}: {maximum}")
    # Parse a representative serialized decimal inside an explicit high-dps
    # context; this catches accidental float round-trips in public output.
    representative = _number(next(iter(a.values())), 65)
    parsed = parse_high_precision(representative, CONTROL_DPS)
    parsed_error = _relative(parsed, next(iter(b.values())))
    if parsed_error > PRECISION_LIMIT:
        raise ArithmeticError("high-precision JSON decimal parse failed")
    return {
        "dps": [PRIMARY_DPS, CONTROL_DPS],
        "probes_compared": len(errors),
        "max_relative_error": maximum,
        "parsed_decimal_relative_error": parsed_error,
        "pass": True,
    }


def _raw_calculate(dps: int) -> dict[str, Any]:
    return _build_snapshot(dps)


def calculate(dps: int = PRIMARY_DPS) -> dict[str, Any]:
    """Recompute one complete atlas at the requested precision and serialize it."""
    return _serialize_snapshot(_raw_calculate(_precision(dps)))


def build_result() -> dict[str, Any]:
    """Recompute primary/control snapshots and return strict JSON-compatible data."""
    with mp.workdps(CONTROL_DPS + 20):
        primary = _raw_calculate(PRIMARY_DPS)
        control = _raw_calculate(CONTROL_DPS)
        precision = _precision_compare(primary, control)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "evidence_weight": EVIDENCE_WEIGHT,
            "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            "upstream_source_sha256": hashlib.sha256(UPSTREAM_PATH.read_bytes()).hexdigest(),
            "foundation_source_sha256": hashlib.sha256(FOUNDATION_PATH.read_bytes()).hexdigest(),
            "nuclear_source_sha256": hashlib.sha256(NUCLEAR_PATH.read_bytes()).hexdigest(),
            "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
            "precision_controls": _shown(precision),
            **_serialize_snapshot(primary),
            "scope": {
                "homogeneous_bulk_backgrounds": True,
                "canonical_scalar_gradient_scale_is_retained": True,
                "Gauss_vector_is_constraint_saddle": True,
                "new_field_interaction_adopted": False,
                "calibration_targets_are_training_only": True,
                "original_parameters_not_refit": True,
                "chi_source_sign_declared": True,
                "k0_is_grand_canonical_long_wavelength_limit": True,
                "fixed_total_N_uniform_mode_excluded": True,
                "high_k_blind_points_are_formal_local_TF": True,
                "not_empirical_validation": True,
                "not_full_dynamics_or_RPA": True,
                "not_finite_nucleus_or_transport": True,
                "not_a_universal_law_or_complete_action_symmetry": True,
                "tail_jet_ambiguity_not_resolved": True,
            },
        }
        return result


def validate_result(result: Any) -> bool:
    """Fail closed by fresh high-precision recomputation, never cache acceptance."""
    try:
        return isinstance(result, dict) and result == build_result()
    except (ArithmeticError, OSError, TypeError, ValueError, KeyError, SpatialIdentifiabilityError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dps", type=int, choices=range(80, 201), default=None,
                        help="print one live snapshot at this precision instead of the 80/110 atlas")
    args = parser.parse_args(argv)
    result = calculate(args.dps) if args.dps is not None else build_result()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
