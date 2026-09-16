#!/usr/bin/env python3
"""Live homogeneous identifiability and finite-momentum discriminator audit.

This is a deliberately bounded companion to
``source_complete_scaling_saturation_audit``.  It recomputes the maintained
homogeneous branch in the current process, checks the original model and the
two-target quartic calibration, and then applies the declared ``W0`` rescaling
at fixed physical ``M_N``, ``m_omega`` and ``g_omega``.  The rescaling is an
exact reparameterisation of the homogeneous quartic functional at fixed
``y=W/W0``; it is *not* asserted to be a symmetry of the canonically
normalised spatial dynamics.  In particular, the finite-momentum scalar
exchange kernel is evaluated at nonzero wave numbers.

The command prints a strict JSON calculation and never writes a result file.
The JSON is evidence of a numerical/model-class audit only and carries zero
independent empirical weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import mpmath as mp


HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
UPSTREAM_PATH = HERE / "source_complete_scaling_saturation_audit.py"
PASSPORT_PATH = HERE / "source_complete_solution_audit.py"
CONTRACT_PATH = HERE / "contracts" / "foundation_identifiability.md"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import source_complete_scaling_saturation_audit as upstream  # noqa: E402
import source_complete_solution_audit as passport_source  # noqa: E402


SCHEMA_VERSION = 1
EVIDENCE_WEIGHT = 0.0
PRIMARY_DPS = 80
CONTROL_DPS = 110
PRECISION_LIMIT = mp.mpf("1e-55")
IDENTITY_LIMIT = mp.mpf("1e-55")
QUADRATURE_LIMIT = mp.mpf("1e-45")
SCALES = ("0.5", "1", "1.3", "2")
# This is a sealed contract, not an output table.  The live upstream inputs,
# the values consumed by ``BulkModel``, and the numeric passport records are
# checked against it before any nominal audit result can be produced.
BASELINE_INPUTS = {
    "W0": {"value": "859", "model_attr": "W0", "passport_id": "W0", "unit": "MeV"},
    "lam": {"value": "1.05", "model_attr": "lam", "passport_id": "lambda", "unit": "1"},
    "MN": {"value": "939", "model_attr": "MN", "passport_id": "M_N", "unit": "MeV"},
    "momega": {"value": "782.6", "model_attr": "momega", "passport_id": "m_omega", "unit": "MeV"},
    "gomega": {"value": "10.12", "model_attr": "gomega", "passport_id": "g_omega", "unit": "1"},
    "hbarc": {"value": "197.3269804", "model_attr": "hbarc", "passport_id": "hbar_c", "unit": "MeV fm"},
    "n0_fm3": {"value": "0.16", "model_attr": "n0_fm3", "passport_id": "n0", "unit": "fm^-3"},
    "d": {"value": "4", "model_attr": "d", "passport_id": None, "unit": "1"},
}
# All entries here are off-grid relative to the n0 calibration point.  The
# n/n0=1 point is separately reported as original saturation, not hidden in
# this grid.
OFF_GRID_DENSITY_RATIOS = ("0.37", "0.73", "1.37", "2.75", "7.25")
FIXED_Y_VALUES = ("0.80", "0.95", "1.10")
WAVE_NUMBERS_MEV = ("0", "25", "200", "1000")
COMPARISON_FIELDS = (
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
STATE_FIELDS = (
    "y",
    "W_MeV",
    "energy_density_MeV4",
    "pressure_MeV4",
    "pressure_MeV_fm3",
    "binding_MeV",
    "mu_MeV",
    "residual_y_MeV4",
    "C_y_MeV4",
    "B_y",
    "D_nn",
    "K_MeV",
    "cs2",
)
QUADRATURE_FIELDS = (
    "energy_relative_error",
    "pressure_relative_error",
    "scalar_density_relative_error",
    "scalar_density_mass_derivative_relative_error",
)


def _mp(value: Any) -> mp.mpf:
    return value if isinstance(value, mp.mpf) else mp.mpf(str(value))


def _finite(value: Any) -> bool:
    try:
        return bool(mp.isfinite(_mp(value)))
    except (TypeError, ValueError):
        return False


def _number(value: Any, digits: int = 60) -> str:
    value = _mp(value)
    if not mp.isfinite(value):
        raise ValueError("nonfinite scientific value")
    return mp.nstr(value, digits)


def _relative(a: Any, b: Any) -> mp.mpf:
    a, b = _mp(a), _mp(b)
    return abs(a - b) / max(abs(a), abs(b), mp.mpf(1))


def _relative_scale(value: Any, scale: Any) -> mp.mpf:
    return abs(_mp(value)) / max(abs(_mp(scale)), mp.mpf(1))


def _decimal_equal(actual: Any, expected: Any) -> bool:
    """Compare decimal source values without first coercing through binary float."""
    try:
        actual_mp, expected_mp = _mp(actual), _mp(expected)
    except (TypeError, ValueError):
        return False
    return bool(mp.isfinite(actual_mp) and mp.isfinite(expected_mp) and actual_mp == expected_mp)


def _assert_baseline_contract(base: upstream.BulkModel) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject drift between the sealed inputs, consumed model, and passport."""
    if set(upstream.INPUTS) != set(BASELINE_INPUTS):
        raise ArithmeticError("upstream input key set drifted from sealed foundation contract")
    for input_id, spec in BASELINE_INPUTS.items():
        if not _decimal_equal(upstream.INPUTS[input_id], spec["value"]):
            raise ArithmeticError(f"upstream input {input_id} drifted from sealed foundation contract")
        consumed = getattr(base, spec["model_attr"], None)
        if not _decimal_equal(consumed, spec["value"]):
            raise ArithmeticError(f"consumed model input {input_id} drifted from sealed foundation contract")

    expected_n0 = _mp(BASELINE_INPUTS["n0_fm3"]["value"]) * _mp(BASELINE_INPUTS["hbarc"]["value"])**3
    if not _decimal_equal(base.n0, expected_n0):
        raise ArithmeticError("consumed natural-unit n0 drifted from sealed foundation contract")
    if not _decimal_equal(base.A, base.lam * base.W0**4):
        raise ArithmeticError("consumed scalar invariant A is incoherent")
    if not _decimal_equal(base.Cv, base.gomega**2 / base.momega**2):
        raise ArithmeticError("consumed vector invariant Cv is incoherent")
    if not _decimal_equal(base.Cs, base.MN**2 / (2 * base.A)):
        raise ArithmeticError("consumed scalar curvature invariant Cs is incoherent")

    passport = passport_source.passport()
    passport_check = passport_source.validate_passport(passport)
    if not passport_check.get("pass"):
        raise ArithmeticError("maintained source passport is invalid or incoherent")
    records = passport.get("records")
    if not isinstance(records, list):
        raise ArithmeticError("maintained source passport base records are malformed")
    records_by_id = {record.get("input_id"): record for record in records if isinstance(record, dict)}
    for spec in BASELINE_INPUTS.values():
        passport_id = spec["passport_id"]
        if passport_id is None:
            continue
        record = records_by_id.get(passport_id)
        if (
            record is None
            or record.get("unit") != spec["unit"]
            or not _decimal_equal(record.get("value"), spec["value"])
            or not _decimal_equal(record.get("value"), getattr(base, spec["model_attr"], None))
        ):
            raise ArithmeticError(f"passport record {passport_id} drifted from sealed foundation contract")
    return passport, passport_check


class ScaledBulkModel(upstream.BulkModel):
    """The declared W0 reparameterisation, without changing the shared API.

    ``BulkModel`` already uses ``y=W/W0`` and stores the homogeneous
    invariants ``A=lambda*W0**4`` and ``Cv=gomega**2/momega**2``.  This small
    local subclass only changes the dimensional field anchor and coupling.
    Its invariant combinations are recomputed from the actual transformed
    inputs; no reference value is injected into the candidate.
    """

    def __init__(self, scale: Any, reference: upstream.BulkModel):
        scale = _mp(scale)
        if not mp.isfinite(scale):
            raise ValueError("scale must be finite")
        if scale <= 0:
            raise ValueError("scale must be positive")
        super().__init__(lam=reference.lam / scale**4, gomega=reference.gomega)
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
        # Recompute the combinations from the actual candidate inputs.  The
        # resulting tiny high-precision rounding differences are part of the
        # invariance check, rather than being hidden by a reference pin.
        self.A = self.lam * self.W0**4
        self.Cv = self.gomega**2 / self.momega**2
        self.Cs = self.MN**2 / (2 * self.A)


def _state_payload(state: dict[str, Any], model: upstream.BulkModel) -> dict[str, str]:
    values = {
        "y": state["y"],
        "W_MeV": model.W0 * state["y"],
        "energy_density_MeV4": state["energy_total"],
        "pressure_MeV4": state["pressure_total"],
        "pressure_MeV_fm3": state["pressure_total"] / model.hbarc**3,
        "binding_MeV": state["energy_total"] / state["n"] - model.MN,
        "mu_MeV": state["mu"],
        "residual_y_MeV4": state["residual"],
        "C_y_MeV4": state["C_y"],
        "B_y": state["B_y"],
        "D_nn": state["D"],
        "K_MeV": state["K"],
        "cs2": state["cs2"],
    }
    return {key: _number(value) for key, value in values.items()}


def _raw_state_values(state: dict[str, Any]) -> dict[str, mp.mpf]:
    return {key: state[key] for key in COMPARISON_FIELDS}


def _finite_exchange(gs: Any, scalar_mass: Any, wave_number: Any) -> mp.mpf:
    """Tree-level vacuum scalar exchange, with wave number in MeV."""
    gs, scalar_mass, wave_number = _mp(gs), _mp(scalar_mass), _mp(wave_number)
    return -gs**2 / (wave_number**2 + scalar_mass**2)


def _parameter_record(reference: upstream.BulkModel, model: ScaledBulkModel) -> dict[str, Any]:
    s = model.scale
    gs0 = reference.MN / reference.W0
    gss = model.MN / model.W0
    qphi0 = reference.momega / reference.W0
    qphis = model.momega / model.W0
    ms0 = mp.sqrt(2 * reference.lam) * reference.W0
    mss = mp.sqrt(2 * model.lam) * model.W0
    gradient0, gradients = reference.W0**2, model.W0**2
    return {
        "scale": s,
        "W0_MeV": model.W0,
        "lambda": model.lam,
        "g_s": gss,
        "q_phi": qphis,
        "m_sigma_MeV": mss,
        "gradient_coefficient_W0_squared_MeV2": gradients,
        "ratios": {
            "W0": model.W0 / reference.W0,
            "lambda": model.lam / reference.lam,
            "g_s": gss / gs0,
            "q_phi": qphis / qphi0,
            "m_sigma": mss / ms0,
            "gradient_coefficient": gradients / gradient0,
            "A": model.A / reference.A,
            "Cv": model.Cv / reference.Cv,
        },
        "expected_ratios": {
            "W0": s,
            "lambda": s**-4,
            "g_s": s**-1,
            "q_phi": s**-1,
            "m_sigma": s**-1,
            "gradient_coefficient": s**2,
            "A": mp.mpf(1),
            "Cv": mp.mpf(1),
        },
        "physical_fixed_inputs": {
            "M_N_MeV": reference.MN,
            "m_omega_MeV": reference.momega,
            "g_omega": reference.gomega,
            "degeneracy": reference.d,
        },
    }


def _finite_q_record(reference: upstream.BulkModel, model: ScaledBulkModel) -> dict[str, Any]:
    s = model.scale
    gs0 = reference.MN / reference.W0
    gss = model.MN / model.W0
    ms0 = mp.sqrt(2 * reference.lam) * reference.W0
    mss = mp.sqrt(2 * model.lam) * model.W0
    rows: list[dict[str, Any]] = []
    for qtext in WAVE_NUMBERS_MEV:
        wave = _mp(qtext)
        base_kernel = _finite_exchange(gs0, ms0, wave)
        scaled_kernel = _finite_exchange(gss, mss, wave)
        # Algebraically multiply the transformed kernel denominator by s^2;
        # this is the finite-q form stated in the authority.
        expected_kernel = -gs0**2 / (s**2 * wave**2 + ms0**2)
        rows.append(
            {
                "wave_number_MeV": wave,
                "base_exchange_MeVminus2": base_kernel,
                "scaled_exchange_MeVminus2": scaled_kernel,
                "expected_scaled_exchange_MeVminus2": expected_kernel,
                "kernel_identity_relative_error": _relative(scaled_kernel, expected_kernel),
                "absolute_difference_MeVminus2": abs(scaled_kernel - base_kernel),
                "equal_at_zero_wave_number": bool(wave == 0 and _relative(scaled_kernel, base_kernel) <= IDENTITY_LIMIT),
                "nonzero_finite_q_difference": bool(wave > 0 and scaled_kernel != base_kernel),
            }
        )
    return {
        "wave_number_symbol": "k (MeV), not q_phi",
        "charge_symbol": "q_phi (dimensionless Higgs charge)",
        "scalar_exchange_formula": "-g_s^2/(k^2+m_sigma^2)",
        "transformed_formula": "-g_s^2/(s^2*k^2+m_sigma^2)",
        "rows": rows,
        "zero_wave_number_equal": bool(_relative(rows[0]["scaled_exchange_MeVminus2"], rows[0]["base_exchange_MeVminus2"]) <= IDENTITY_LIMIT),
        "finite_wave_number_nonzero_for_s_not_1": bool(s != 1 and all(row["nonzero_finite_q_difference"] for row in rows[1:])),
        "finite_wave_number_proof_reached": bool(any(row["wave_number_MeV"] > 0 for row in rows)),
    }


def _input_sensitivity_controls(base: upstream.BulkModel) -> dict[str, Any]:
    """Show that changing one input alone changes the computed invariant."""
    candidate = ScaledBulkModel("1.3", base)
    lambda_only = ScaledBulkModel("1.3", base)
    lambda_only.lam *= mp.mpf("1.01")
    lambda_only.A = lambda_only.lam * lambda_only.W0**4
    lambda_only.Cs = lambda_only.MN**2 / (2 * lambda_only.A)
    w0_only = ScaledBulkModel("1.3", base)
    w0_only.W0 *= mp.mpf("1.01")
    w0_only.A = w0_only.lam * w0_only.W0**4
    w0_only.Cs = w0_only.MN**2 / (2 * w0_only.A)
    candidate_a_from_inputs = candidate.lam * candidate.W0**4
    lambda_a_from_inputs = lambda_only.lam * lambda_only.W0**4
    w0_a_from_inputs = w0_only.lam * w0_only.W0**4
    return {
        "candidate_scale": candidate.scale,
        "candidate_A": candidate.A,
        "candidate_A_from_actual_inputs": candidate_a_from_inputs,
        "candidate_Cv": candidate.Cv,
        "candidate_Cv_from_actual_inputs": candidate.gomega**2 / candidate.momega**2,
        "lambda_only_A": lambda_only.A,
        "lambda_only_A_from_actual_inputs": lambda_a_from_inputs,
        "W0_only_A": w0_only.A,
        "W0_only_A_from_actual_inputs": w0_a_from_inputs,
        "lambda_only_relative_A_change": _relative(lambda_only.A, candidate.A),
        "W0_only_relative_A_change": _relative(w0_only.A, candidate.A),
        "pass": bool(
            _relative(candidate.A, candidate_a_from_inputs) <= IDENTITY_LIMIT
            and _relative(candidate.Cv, candidate.gomega**2 / candidate.momega**2) <= IDENTITY_LIMIT
            and lambda_only.A != candidate.A
            and w0_only.A != candidate.A
        ),
    }


def _quadrature_check(model: upstream.BulkModel, n: mp.mpf, y: mp.mpf) -> dict[str, Any]:
    """Independent momentum quadratures for a newly evaluated state."""
    f = model.fermi(n, y)
    factor = model.d / (2 * mp.pi**2)
    m = f["m"]
    k = f["k"]
    energy_quad = factor * mp.quad(lambda p: p**2 * mp.sqrt(p**2 + m**2), [0, k])
    pressure_quad = factor / 3 * mp.quad(lambda p: p**4 / mp.sqrt(p**2 + m**2), [0, k])
    scalar_quad = factor * mp.quad(lambda p: p**2 * m / mp.sqrt(p**2 + m**2), [0, k])
    scalar_mass_quad = factor * mp.quad(lambda p: p**4 / (p**2 + m**2) ** (mp.mpf(3) / 2), [0, k])
    errors = {
        "energy_relative_error": _relative(f["energy"], energy_quad),
        "pressure_relative_error": _relative(f["pressure"], pressure_quad),
        "scalar_density_relative_error": _relative(f["ns"], scalar_quad),
        "scalar_density_mass_derivative_relative_error": _relative(f["ns_m"], scalar_mass_quad),
    }
    return {
        "density_ratio": n / model.n0,
        "y": y,
        "state": _state_payload(model.state(n, y), model),
        "errors": errors,
        "pass": all(value <= QUADRATURE_LIMIT for value in errors.values()),
    }


def _independent_calibration_k(model: upstream.BulkModel, state: dict[str, Any]) -> mp.mpf:
    n0 = state["n"]
    y0 = state["y"]

    def binding(nn: mp.mpf) -> mp.mpf:
        nearby = model.equilibrium(nn, (y0 * mp.mpf("0.99"), y0 * mp.mpf("1.01")))
        return nearby["energy_total"] / nn - model.MN

    # At a pressure-zero calibration point, K = 9 n0^2 d2(E/A)/dn2
    # = 9 n0 d mu/dn.  Differentiate the binding energy here (rather than
    # reading the model's relaxed curvature) so this remains an independent
    # check of the two-target root.  The equivalent chemical-potential form
    # is used below for the non-saturated original point.
    return 9 * n0**2 * mp.diff(binding, n0, 2)


def _live_vector_ceiling(base: upstream.BulkModel, binding: Any = "-16") -> dict[str, mp.mpf]:
    """Recompute the necessary g_omega ceiling from the live BulkModel."""
    n = base.n0
    mu_target = base.MN + _mp(binding)
    k = base.fermi(n, mp.mpf(1))["k"]
    if not k < mu_target:
        raise ValueError("target chemical potential must exceed massless Fermi energy")
    e_star = (mu_target + mp.sqrt(mu_target**2 + 3 * k**2)) / 3
    y_star = mp.sqrt(e_star**2 - k**2) / base.MN
    cv_max = y_star**2 * (mu_target - e_star) / n
    y_min = mp.findroot(
        lambda y: base.MN**2 * y / base.fermi(n, y)["ef"] - 2 * base.Cv * n / y**3,
        (mp.mpf("0.6"), mp.mpf("0.9")),
    )
    mu_min = base.fermi(n, y_min)["ef"] + base.Cv * n / y_min**2
    cap = upstream.BulkModel(gomega=base.momega * mp.sqrt(cv_max))
    cap.W0 = base.W0
    cap.A = base.A
    cap.Cv = cv_max
    cap.Cs = base.Cs
    cap_state = cap.state(n, y_star)
    return {
        "gomega_max": base.momega * mp.sqrt(cv_max),
        "y_at_ceiling": y_star,
        "Cv_max": cv_max,
        "original_mu_min_MeV": mu_min,
        "original_y_at_mu_min": y_min,
        "mu_target_MeV": mu_target,
        "finite_curvature_K_at_ceiling_MeV": 3 * mu_target,
        "ceiling_state_B_y": cap_state["B_y"],
        "ceiling_state_K_MeV": cap_state["K"],
        "ceiling_state_mu_MeV": cap_state["mu"],
        "ceiling_state_mu_target_relative": _relative(cap_state["mu"], mu_target),
    }


def _raw_snapshot(dps: int) -> dict[str, Any]:
    with mp.workdps(dps):
        base = upstream.BulkModel()
        # Validate immediately after construction so an upstream-input drift
        # cannot be consumed by a long calculation and mislabeled as baseline.
        passport, passport_check = _assert_baseline_contract(base)
        original = base.equilibrium(base.n0)
        if original["C_y"] <= 0:
            raise ArithmeticError("original branch is not locally stable")

        ceiling = _live_vector_ceiling(base)
        calibration_model, calibration_state = upstream.quartic_calibration()
        if calibration_state["C_y"] <= 0:
            raise ArithmeticError("calibration branch is not locally stable")
        calibration_k = _independent_calibration_k(calibration_model, calibration_state)

        scales: list[dict[str, Any]] = []
        for stext in SCALES:
            model = ScaledBulkModel(stext, base)
            parameter = _parameter_record(base, model)
            fixed_y_rows: list[dict[str, Any]] = []
            equilibrium_rows: list[dict[str, Any]] = []
            for ratio_text in OFF_GRID_DENSITY_RATIOS:
                ratio = _mp(ratio_text)
                n = base.n0 * ratio
                for ytext in FIXED_Y_VALUES:
                    y = _mp(ytext)
                    reference_state = base.state(n, y)
                    scaled_state = model.state(n, y)
                    differences = {
                        key: _relative(scaled_state[key], reference_state[key])
                        for key in COMPARISON_FIELDS
                    }
                    fixed_y_rows.append(
                        {
                            "density_ratio": ratio,
                            "y": y,
                            "reference": reference_state,
                            "scaled": scaled_state,
                            "relative_differences": differences,
                        }
                    )
                reference_eq = base.equilibrium(n)
                scaled_eq = model.equilibrium(n)
                differences = {
                    key: _relative(scaled_eq[key], reference_eq[key])
                    for key in COMPARISON_FIELDS
                }
                equilibrium_rows.append(
                    {
                        "density_ratio": ratio,
                        "reference": reference_eq,
                        "scaled": scaled_eq,
                        "relative_differences": differences,
                    }
                )
            scales.append(
                {
                    "scale_label": stext,
                    "scale": model.scale,
                    "parameter": parameter,
                    "fixed_y_rows": fixed_y_rows,
                    "equilibrium_rows": equilibrium_rows,
                    "finite_q": _finite_q_record(base, model),
                }
            )

        sensitivity = _input_sensitivity_controls(base)

        quadrature_rows = []
        for stext, ratio_text, ytext in (
            ("1", "0.37", "0.80"),
            ("2", "2.75", "1.10"),
        ):
            model = base if stext == "1" else ScaledBulkModel(stext, base)
            quadrature_rows.append(_quadrature_check(model, model.n0 * _mp(ratio_text), _mp(ytext)))

        # The original point is not at P=0, so its compressibility is
        # unambiguously checked as K=9 n dmu/dn rather than by the binding
        # second derivative (which has an extra nonzero-pressure term).
        original_k_independent = 9 * original["n"] * mp.diff(
            lambda nn: base.equilibrium(nn, (original["y"] * mp.mpf("0.99"), original["y"] * mp.mpf("1.01")))["mu"],
            original["n"],
        )

        return {
            "dps": dps,
            "base": base,
            "original": original,
            "original_k_independent": original_k_independent,
            "ceiling": ceiling,
            "calibration_model": calibration_model,
            "calibration_state": calibration_state,
            "calibration_k_independent": calibration_k,
            "scales": scales,
            "sensitivity": sensitivity,
            "quadrature": quadrature_rows,
            "passport": passport,
            "passport_check": passport_check,
        }


def _precision_probes(snapshot: dict[str, Any]) -> dict[str, mp.mpf]:
    probes: dict[str, mp.mpf] = {}
    probes.update(
        {
            "original.energy_total": snapshot["original"]["energy_total"],
            "original.pressure_total": snapshot["original"]["pressure_total"],
            "original.mu": snapshot["original"]["mu"],
            "original.K": snapshot["original"]["K"],
            "original_k_independent": snapshot["original_k_independent"],
            "ceiling.gomega_max": snapshot["ceiling"]["gomega_max"],
            "ceiling.y_at_ceiling": snapshot["ceiling"]["y_at_ceiling"],
            "calibration.K": snapshot["calibration_state"]["K"],
            "calibration.K_independent": snapshot["calibration_k_independent"],
            "sensitivity.candidate_A": snapshot["sensitivity"]["candidate_A"],
            "sensitivity.lambda_only_A": snapshot["sensitivity"]["lambda_only_A"],
            "sensitivity.W0_only_A": snapshot["sensitivity"]["W0_only_A"],
        }
    )
    for item in snapshot["scales"]:
        # Use the declared decimal label, not str(mp_value), because the
        # latter exposes precision-dependent trailing roundoff for 1.3.
        tag = item["scale_label"]
        first = item["fixed_y_rows"][0]
        last = item["equilibrium_rows"][-1]
        probes[f"scale_{tag}.fixed_y_energy"] = first["scaled"]["energy_total"]
        probes[f"scale_{tag}.equilibrium_y"] = last["scaled"]["y"]
        probes[f"scale_{tag}.finite_q_delta"] = item["finite_q"]["rows"][2]["absolute_difference_MeVminus2"]
    return probes


def _serialize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    base = snapshot["base"]
    calibration_model = snapshot["calibration_model"]
    original = snapshot["original"]
    original_binding = original["energy_total"] / original["n"] - base.MN
    original_block = {
        "density_ratio": "1",
        "n_fm3": _number(base.n0_fm3),
        "state": _state_payload(original, base),
        "binding_MeV": _number(original_binding),
        "classification": "ORIGINAL_NOT_SATURATED_POSITIVE_BINDING",
        "target_binding_MeV": _number(_mp("-16")),
        "target_pressure_MeV_fm3": _number(mp.mpf(0)),
        "binding_target_match": bool(_relative(original_binding, -16) <= IDENTITY_LIMIT),
        "pressure_target_match": bool(_relative(original["pressure_total"] / base.hbarc**3, 0) <= IDENTITY_LIMIT),
        "Cv_fm2": _number(base.Cv * base.hbarc**2),
        "Cs_fm2": _number(base.Cs * base.hbarc**2),
        "Cv_over_Cs": _number(base.Cv / base.Cs),
        "global_no_binding_sufficient_condition": bool(base.Cv >= base.Cs),
        "independent_K_MeV": _number(snapshot["original_k_independent"]),
    }
    ceiling = {key: _number(value) for key, value in snapshot["ceiling"].items()}
    ceiling.update(
        {
            "original_gomega_dimensionless": _number(base.gomega),
            "original_exceeds_ceiling": bool(base.gomega > snapshot["ceiling"]["gomega_max"]),
            "ceiling_state_B_y_relative": _number(_relative_scale(snapshot["ceiling"]["ceiling_state_B_y"], snapshot["ceiling"]["ceiling_state_mu_MeV"])),
            "ceiling_state_mu_target_relative": _number(snapshot["ceiling"]["ceiling_state_mu_target_relative"]),
            "ceiling_curvature_identity_relative": _number(
                _relative(snapshot["ceiling"]["ceiling_state_K_MeV"], 3 * snapshot["ceiling"]["ceiling_state_mu_MeV"])
            ),
        }
    )
    calibration_state = _state_payload(snapshot["calibration_state"], calibration_model)
    calibration = {
        "training_targets": {"n_fm3": "0.16", "binding_MeV": "-16"},
        "fitted_lambda": _number(calibration_model.lam),
        "fitted_gomega": _number(calibration_model.gomega),
        "state": calibration_state,
        "binding_MeV": _number(snapshot["calibration_state"]["energy_total"] / snapshot["calibration_state"]["n"] - calibration_model.MN),
        "independent_K_MeV": _number(snapshot["calibration_k_independent"]),
        "K_from_state_MeV": _number(snapshot["calibration_state"]["K"]),
        "K_was_fitted": False,
        "K_negative_control": bool(snapshot["calibration_state"]["K"] > 350),
        "K_reference_ranges_MeV": [["220", "260"], ["200", "350"]],
        "deployed_to_baseline": False,
        "calibration_is_not_held_out_evidence": True,
    }
    scales: list[dict[str, Any]] = []
    for item in snapshot["scales"]:
        parameter = item["parameter"]
        parameter_serialized = {
            "scale": _number(parameter["scale"]),
            "W0_MeV": _number(parameter["W0_MeV"]),
            "lambda": _number(parameter["lambda"]),
            "g_s": _number(parameter["g_s"]),
            "q_phi": _number(parameter["q_phi"]),
            "m_sigma_MeV": _number(parameter["m_sigma_MeV"]),
            "gradient_coefficient_W0_squared_MeV2": _number(parameter["gradient_coefficient_W0_squared_MeV2"]),
            "ratios": {key: _number(value) for key, value in parameter["ratios"].items()},
            "expected_ratios": {key: _number(value) for key, value in parameter["expected_ratios"].items()},
            "physical_fixed_inputs": {
                key: (_number(value) if isinstance(value, mp.mpf) else value)
                for key, value in parameter["physical_fixed_inputs"].items()
            },
        }
        fixed_y_rows = []
        for row in item["fixed_y_rows"]:
            fixed_y_rows.append(
                {
                    "density_ratio": _number(row["density_ratio"]),
                    "y": _number(row["y"]),
                    "reference": _state_payload(row["reference"], base),
                    "scaled": _state_payload(row["scaled"], ScaledBulkModel(item["scale"], base)),
                    "relative_differences": {key: _number(value) for key, value in row["relative_differences"].items()},
                    "pass": bool(all(value <= IDENTITY_LIMIT for value in row["relative_differences"].values())),
                }
            )
        equilibrium_rows = []
        for row in item["equilibrium_rows"]:
            equilibrium_rows.append(
                {
                    "density_ratio": _number(row["density_ratio"]),
                    "reference": _state_payload(row["reference"], base),
                    "scaled": _state_payload(row["scaled"], ScaledBulkModel(item["scale"], base)),
                    "relative_differences": {key: _number(value) for key, value in row["relative_differences"].items()},
                    "pass": bool(all(value <= IDENTITY_LIMIT for value in row["relative_differences"].values())),
                }
            )
        finite_q = item["finite_q"]
        finite_q_serialized = {
            key: finite_q[key]
            for key in (
                "wave_number_symbol",
                "charge_symbol",
                "scalar_exchange_formula",
                "transformed_formula",
            )
        }
        finite_q_serialized.update(
            {
                "rows": [
                    {
                        "wave_number_MeV": _number(row["wave_number_MeV"]),
                        "base_exchange_MeVminus2": _number(row["base_exchange_MeVminus2"]),
                        "scaled_exchange_MeVminus2": _number(row["scaled_exchange_MeVminus2"]),
                        "expected_scaled_exchange_MeVminus2": _number(row["expected_scaled_exchange_MeVminus2"]),
                        "kernel_identity_relative_error": _number(row["kernel_identity_relative_error"]),
                        "absolute_difference_MeVminus2": _number(row["absolute_difference_MeVminus2"]),
                        "equal_at_zero_wave_number": row["equal_at_zero_wave_number"],
                        "nonzero_finite_q_difference": row["nonzero_finite_q_difference"],
                    }
                    for row in finite_q["rows"]
                ],
                "zero_wave_number_equal": finite_q["zero_wave_number_equal"],
                "finite_wave_number_nonzero_for_s_not_1": finite_q["finite_wave_number_nonzero_for_s_not_1"],
                "finite_wave_number_proof_reached": finite_q["finite_wave_number_proof_reached"],
            }
        )
        scales.append(
            {
                "scale": _number(item["scale"]),
                "parameter": parameter_serialized,
                "fixed_y_rows": fixed_y_rows,
                "equilibrium_rows": equilibrium_rows,
                "all_fixed_y_rows_pass": all(row["pass"] for row in fixed_y_rows),
                "all_equilibrium_rows_pass": all(row["pass"] for row in equilibrium_rows),
                "finite_q": finite_q_serialized,
                "homogeneous_eos_invariant_at_fixed_y": all(row["pass"] for row in fixed_y_rows),
                "full_dynamics_not_invariant": bool(
                    _relative(parameter["ratios"]["gradient_coefficient"], 1) > IDENTITY_LIMIT
                    or _relative(parameter["ratios"]["m_sigma"], 1) > IDENTITY_LIMIT
                    or finite_q["finite_wave_number_nonzero_for_s_not_1"]
                ),
            }
        )
    quadrature = []
    for row in snapshot["quadrature"]:
        quadrature.append(
            {
                "density_ratio": _number(row["density_ratio"]),
                "y": _number(row["y"]),
                "state": row["state"],
                "errors": {key: _number(value) for key, value in row["errors"].items()},
                "pass": row["pass"],
            }
        )
    sensitivity_raw = snapshot["sensitivity"]
    sensitivity = {
        "candidate_scale": _number(sensitivity_raw["candidate_scale"]),
        "candidate_A": _number(sensitivity_raw["candidate_A"]),
        "candidate_A_from_actual_inputs": _number(sensitivity_raw["candidate_A_from_actual_inputs"]),
        "candidate_Cv": _number(sensitivity_raw["candidate_Cv"]),
        "candidate_Cv_from_actual_inputs": _number(sensitivity_raw["candidate_Cv_from_actual_inputs"]),
        "lambda_only_A": _number(sensitivity_raw["lambda_only_A"]),
        "lambda_only_A_from_actual_inputs": _number(sensitivity_raw["lambda_only_A_from_actual_inputs"]),
        "W0_only_A": _number(sensitivity_raw["W0_only_A"]),
        "W0_only_A_from_actual_inputs": _number(sensitivity_raw["W0_only_A_from_actual_inputs"]),
        "lambda_only_relative_A_change": _number(sensitivity_raw["lambda_only_relative_A_change"]),
        "W0_only_relative_A_change": _number(sensitivity_raw["W0_only_relative_A_change"]),
        "pass": sensitivity_raw["pass"],
    }
    return {
        "dps": snapshot["dps"],
        "original": original_block,
        "vector_ceiling": ceiling,
        "two_target_quartic_calibration": calibration,
        "scale_identifiability": {
            "scales": scales,
            "density_ratios": list(OFF_GRID_DENSITY_RATIOS),
            "fixed_y_values": list(FIXED_Y_VALUES),
            "homogeneous_invariants": ["A=lambda*W0^4", "M_N", "Cv=g_omega^2/m_omega^2", "d"],
            "physical_fixed_inputs": ["M_N", "m_omega", "g_omega"],
            "transformation": {
                "W0": "s*W0",
                "lambda": "lambda/s^4",
                "g_s": "g_s/s",
                "q_phi": "q_phi/s",
            },
            "homogeneous_eos_claim": "invariant at fixed y and fixed physical density",
            "full_dynamics_claim": "not invariant with canonical kinetic terms and finite wave number",
        },
        "input_sensitivity_controls": sensitivity,
        "independent_quadrature": quadrature,
        "passport": snapshot["passport"],
        "passport_validation": snapshot["passport_check"],
    }


def _precision_result(primary: dict[str, Any], control: dict[str, Any]) -> list[dict[str, Any]]:
    left, right = _precision_probes(primary), _precision_probes(control)
    if set(left) != set(right):
        raise ArithmeticError("precision probe sets differ")
    result = []
    for key in left:
        difference = _relative(left[key], right[key])
        result.append(
            {
                "check_id": key,
                "primary_dps": PRIMARY_DPS,
                "control_dps": CONTROL_DPS,
                "max_relative_difference": _number(difference),
                "pass": bool(difference <= PRECISION_LIMIT),
            }
        )
    if not all(item["pass"] for item in result):
        raise ArithmeticError("80/110-digit precision control failed")
    return result


def _input_scope_payload(base: upstream.BulkModel) -> dict[str, Any]:
    """Serialize baseline metadata from the validated, consumed model."""
    return {
        "W0_MeV": _number(base.W0),
        "lambda": _number(base.lam),
        "M_N_MeV": _number(base.MN),
        "m_omega_MeV": _number(base.momega),
        "g_omega": _number(base.gomega),
        "hbarc_MeV_fm": _number(base.hbarc),
        "n0_fm3": _number(base.n0_fm3),
        "degeneracy": int(base.d),
        "original_saturation_target": {"binding_MeV": "-16", "pressure_MeV_fm3": "0"},
        "calibration_targets": {"binding_MeV": "-16", "pressure_MeV_fm3": "0"},
        "scales": list(SCALES),
        "off_grid_density_ratios": list(OFF_GRID_DENSITY_RATIOS),
        "fixed_y_values": list(FIXED_Y_VALUES),
        "wave_numbers_MeV": list(WAVE_NUMBERS_MEV),
        "primary_working_precision_digits": PRIMARY_DPS,
        "independent_control_precision_digits": CONTROL_DPS,
        "all_new_empirical_weights": 0.0,
    }


def build_result() -> dict[str, Any]:
    """Recompute the complete bounded audit without reading a result cache."""
    with mp.workdps(CONTROL_DPS + 15):
        primary = _raw_snapshot(PRIMARY_DPS)
        control = _raw_snapshot(CONTROL_DPS)
        serialized = _serialize_snapshot(primary)
        precision = _precision_result(primary, control)
        if not all(row["pass"] for row in serialized["independent_quadrature"]):
            raise ArithmeticError("independent quadrature control failed")
        if not serialized["input_sensitivity_controls"]["pass"]:
            raise ArithmeticError("single-input sensitivity control failed")
        if not serialized["passport_validation"].get("pass"):
            raise ArithmeticError("passport validation failed")
        if not all(item["homogeneous_eos_invariant_at_fixed_y"] for item in serialized["scale_identifiability"]["scales"]):
            raise ArithmeticError("fixed-y homogeneous invariance failed")
        # The identity member s=1 is, of course, identical to itself.  The
        # discriminator gate applies only to the two nontrivial rescalings.
        if not all(
            item["full_dynamics_not_invariant"]
            for item in serialized["scale_identifiability"]["scales"]
            if _mp(item["scale"]) != 1
        ):
            raise ArithmeticError("full-dynamics discriminator failed")
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "LIVE_FOUNDATION_IDENTIFIABILITY_AUDIT_CALIBRATION_NULL_NOT_EMPIRICAL",
            "evidence_weight": EVIDENCE_WEIGHT,
            "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            "upstream_source_sha256": hashlib.sha256(UPSTREAM_PATH.read_bytes()).hexdigest(),
            "passport_source_sha256": hashlib.sha256(PASSPORT_PATH.read_bytes()).hexdigest(),
            "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
            "precision_controls": precision,
            "inputs_and_scope": _input_scope_payload(primary["base"]),
            **serialized,
            "scope": {
                "homogeneous_bulk_only": True,
                "canonical_kinetic_terms_in_discriminator": True,
                "finite_q_is_tree_level_vacuum_exchange": True,
                "q_wave_number_distinct_from_q_phi": True,
                "new_field_interaction_adopted": False,
                "calibration_targets_are_training_only": True,
                "original_parameters_not_refit": True,
                "not_empirical_validation": True,
                "not_a_full_dynamics_symmetry": True,
                "not_a_finite_nucleus_or_transport_result": True,
                "not_a_universal_theory_claim": True,
            },
        }


def validate_result(result: Any) -> bool:
    """Fail closed by fresh recomputation, never by trusting success flags."""
    try:
        if not isinstance(result, dict):
            return False
        fresh = build_result()
        return result == fresh
    except (ArithmeticError, OSError, TypeError, ValueError, KeyError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = build_result()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
