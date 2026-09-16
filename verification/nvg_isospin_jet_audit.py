#!/usr/bin/env python3
"""Live local isospin-jet audit for the declared constant-rho extension.

The source-complete ``BulkModel`` and its inverse polynomial-jet producer are
kept read-only.  This module only supplies the two-spin-species bookkeeping,
the separately declared eliminated rho energy, and local derivatives at the
chosen stationary point.  All numerical rows are recomputed from those
producers when the JSON-only CLI is run; the input JSON contains design inputs
and descriptive source references, never answer tables.

The result is a conditional homogeneous local jet.  It is not a neutron-matter
EOS, a finite-nucleus calculation, a transport calculation, or an empirical
likelihood.  In particular, the ``J`` values below are declared design inputs:
``L`` and ``S4`` are consequences and are never used to select a residual-y
row.
"""

from __future__ import annotations

import argparse
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import mpmath as mp

# The CLI is a no-write audit; also prevent import bytecode from becoming a
# side effect when it is invoked from a read-only checkout.
sys.dont_write_bytecode = True

try:  # Allow both ``python verification/file.py`` and package-style imports.
    from source_complete_scaling_saturation_audit import (
        BulkModel,
        inverse_potential_jet,
        quartic_calibration,
    )
except ImportError:  # pragma: no cover - exercised by package importers
    from .source_complete_scaling_saturation_audit import (
        BulkModel,
        inverse_potential_jet,
        quartic_calibration,
    )


HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
INPUT_PATH = HERE / "data" / "nvg_isospin_jet_benchmarks.json"
SCHEMA = "nvg_isospin_jet_audit.v1"
EVIDENCE_WEIGHT = 0.0
PHYSICAL_INTERPRETATION = (
    "CONDITIONAL_LOCAL_HOMOGENEOUS_JET; no finite-nucleus, transport, "
    "beta-equilibrium, neutron-star or empirical-completion claim"
)

# P1 is a sealed executable protocol, rather than a general-purpose matrix
# generator.  Keep these values as decimal text so that spelling variants such
# as ``.90`` and ``0.900`` can be compared semantically without changing the
# declared physical inputs or the output labels.
_SEALED_N0_FM3 = "0.16"
_SEALED_BINDING_MEV = "-16"
_SEALED_KTARGET_MEV = "240"
_SEALED_J_DESIGN_MEV = "32"
_SEALED_J_REFERENCE_CONTROLS_MEV = ("31", "34")
_SEALED_RESIDUAL_Y = ("0.60", "0.75", "0.85", "0.90", "0.93")
_SEALED_SENSITIVITY_K_MEV = ("220", "260")
_SEALED_DIRECT_CHECK_Y = ("0.75", "0.90")
_SEALED_WORKING_PRECISIONS = (80, 110)

_EXPECTED_STATE_NUMERIC_FIELDS = (
    "n_fm3",
    "delta",
    "y",
    "pressure_MeV_fm3",
    "binding_MeV",
    "mu_fixed_delta_MeV",
    "mu_n_MeV",
    "mu_p_MeV",
    "C_y_MeV4",
    "stationarity_residual_relative",
)
_EXPECTED_JET_NUMERIC_FIELDS = (
    "C_rho",
    "T",
    "u",
    "C",
    "B",
    "D",
    "Kfr",
    "K",
    "Kfr_minus_K",
    "dy_dn",
    "y_delta2",
    "T_y",
    "S2",
    "J",
    "L_fixed_y",
    "L_kin",
    "L",
    "S4_kin",
    "S4_fixed_y",
    "S4_relaxation",
    "S4_hessian",
    "S4_relaxed",
    "Q",
    "S4_eliminated",
    "identity_residual_K",
)
_EXPECTED_THERMO_NUMERIC_FIELDS = (
    "energy_density",
    "pressure",
    "mu_fixed_delta",
    "mu_n",
    "mu_p",
)
_DIRECT_RESTATIONARIZATION_FIELDS = (
    "h_delta",
    "h_density_fraction",
    "S2_direct_richardson",
    "S4_direct_richardson",
    "L_direct_density_fd",
    "y_delta2_direct",
    "parity_energy_max",
    "S2_identity_residual",
    "S4_hessian_residual",
    "L_identity_residual",
    "y_delta2_residual",
)
_DIRECT_SPECIES_FIELDS = (
    "energy_relative_error",
    "pressure_relative_error",
    "scalar_density_relative_error",
)

# These are numerical acceptance limits for the implemented h=.002 / h=.001
# Richardson delta stencil, the n +/- 1e-4 density stencil, and 80/110-digit
# arithmetic.  The residuals are dimensionless only where the producer made
# them dimensionless; otherwise the scale is the corresponding computed
# observable in the same MeV convention.  They are deliberately much tighter
# than a unit-sized injected residual and are not physical uncertainties.
_DIRECT_RESIDUAL_LIMITS = {
    "parity_energy_max": ("1e-60", "0"),
    "S2_identity_residual": ("1e-8", "1e-12"),
    "S4_hessian_residual": ("1e-8", "1e-12"),
    "L_identity_residual": ("1e-4", "1e-12"),
    "y_delta2_residual": ("1e-8", "1e-12"),
}
_DIRECT_SPECIES_ERROR_LIMIT = mp.mpf("1e-50")
_ALLOWED_GLOBAL_STATUS_PREFIXES = (
    "BASELINE_",
    "TWO_TARGET_QUARTIC_",
    "LOCAL_RESPONSE_RETAINED_",
    "REJECTED_GLOBAL_POTENTIAL_",
)


class IsospinJetError(ValueError):
    """Fail-closed error for invalid physical or numerical jet requests."""


def _semantic_mp(value: Any, name: str) -> mp.mpf:
    """Parse a protocol scalar by decimal meaning, including Python floats."""

    decimal = _semantic_decimal(value, name)
    # Keep protocol validation independent of the caller's ambient mpmath
    # precision; otherwise a tiny nominal mutation could round back to .16.
    with mp.workdps(max(80, mp.mp.dps)):
        return mp.mpf(str(decimal))


def _semantic_decimal(value: Any, name: str) -> Decimal:
    """Parse an exact finite decimal for sealed-protocol equality tests."""

    if isinstance(value, bool):
        raise IsospinJetError(f"{name} must be a finite real, not bool")
    if isinstance(value, float):
        value = repr(value)
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise IsospinJetError(f"{name} must be a finite real") from exc
    if not decimal.is_finite():
        raise IsospinJetError(f"{name} must be finite")
    return decimal


def _semantic_text(value: Any) -> str:
    return str(_semantic_decimal(value, "protocol value"))


def _require_exact_scalar(value: Any, expected: str, name: str) -> mp.mpf:
    actual_decimal = _semantic_decimal(value, name)
    if actual_decimal != Decimal(expected):
        raise IsospinJetError(
            f"{name} is sealed at {expected}; received {_semantic_text(value)}"
        )
    return _semantic_mp(value, name)


def _require_exact_decimal_list(
    values: Any,
    name: str,
    expected: Sequence[str],
    *,
    positive: bool = False,
) -> tuple[str, ...]:
    """Require an order-independent, unique list of exact decimal values."""

    if not isinstance(values, list) or not values:
        raise IsospinJetError(f"{name} must be a nonempty list")
    parsed = [_semantic_decimal(value, f"{name} item") for value in values]
    if positive and any(value <= 0 for value in parsed):
        raise IsospinJetError(f"{name} items must be positive")
    duplicate_values = []
    for index, value in enumerate(parsed):
        if any(value == prior for prior in parsed[:index]):
            duplicate_values.append(_semantic_text(value))
    expected_decimal = [Decimal(value) for value in expected]
    missing = [
        value
        for value, expected_value in zip(expected, expected_decimal)
        if not any(actual == expected_value for actual in parsed)
    ]
    foreign = [
        _semantic_text(value)
        for value in parsed
        if not any(value == expected_value for expected_value in expected_decimal)
    ]
    if duplicate_values or missing or foreign or len(parsed) != len(expected):
        details = []
        if duplicate_values:
            details.append(f"duplicates={duplicate_values}")
        if missing:
            details.append(f"missing={missing}")
        if foreign:
            details.append(f"foreign={foreign}")
        raise IsospinJetError(
            f"{name} must contain exactly the sealed semantic set "
            f"{list(expected)} ({', '.join(details)})"
        )
    # Return the sealed order so output labels and precision snapshots are
    # deterministic even when an equivalent input list was reordered.
    return tuple(expected)


def _require_exact_precision_list(values: Any) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise IsospinJetError("working_precisions must be a nonempty list")
    parsed = []
    for value in values:
        if isinstance(value, bool):
            raise IsospinJetError("working_precisions items must be integer values >=80")
        parsed_value = _semantic_decimal(value, "working_precisions item")
        if parsed_value != parsed_value.to_integral_value() or parsed_value < 80:
            raise IsospinJetError(
                "working_precisions items must be distinct integer values >=80"
            )
        parsed.append(int(parsed_value))
    duplicates = sorted({value for value in parsed if parsed.count(value) > 1})
    expected = set(_SEALED_WORKING_PRECISIONS)
    if duplicates or len(parsed) != len(expected) or set(parsed) != expected:
        details = []
        if duplicates:
            details.append(f"duplicates={duplicates}")
        missing = sorted(expected - set(parsed))
        foreign = sorted(set(parsed) - expected)
        if missing:
            details.append(f"missing={missing}")
        if foreign:
            details.append(f"foreign={foreign}")
        raise IsospinJetError(
            "working_precisions must contain exactly the distinct sealed "
            f"values {list(_SEALED_WORKING_PRECISIONS)} ({', '.join(details)})"
        )
    return _SEALED_WORKING_PRECISIONS


def _protocol_calculation_inputs(calc: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the complete sealed calculation protocol."""

    required = (
        "n0_fm3",
        "binding_MeV",
        "Ktarget_MeV",
        "J_design_MeV",
        "J_reference_controls_MeV",
        "residual_y",
        "sensitivity_K_MeV",
        "direct_check_y",
        "working_precisions",
    )
    missing = [key for key in required if key not in calc]
    if missing:
        raise IsospinJetError(f"missing calculation input(s): {', '.join(missing)}")
    _require_exact_scalar(calc["n0_fm3"], _SEALED_N0_FM3, "n0_fm3")
    _require_exact_scalar(calc["binding_MeV"], _SEALED_BINDING_MEV, "binding_MeV")
    _require_exact_scalar(calc["Ktarget_MeV"], _SEALED_KTARGET_MEV, "Ktarget_MeV")
    _require_exact_scalar(calc["J_design_MeV"], _SEALED_J_DESIGN_MEV, "J_design_MeV")
    return {
        "n0_fm3": _SEALED_N0_FM3,
        "binding_MeV": _SEALED_BINDING_MEV,
        "Ktarget_MeV": _SEALED_KTARGET_MEV,
        "J_design_MeV": _SEALED_J_DESIGN_MEV,
        "J_reference_controls_MeV": list(
            _require_exact_decimal_list(
                calc["J_reference_controls_MeV"],
                "J_reference_controls_MeV",
                _SEALED_J_REFERENCE_CONTROLS_MEV,
                positive=True,
            )
        ),
        "residual_y": list(
            _require_exact_decimal_list(
                calc["residual_y"], "residual_y", _SEALED_RESIDUAL_Y, positive=True
            )
        ),
        "sensitivity_K_MeV": list(
            _require_exact_decimal_list(
                calc["sensitivity_K_MeV"],
                "sensitivity_K_MeV",
                _SEALED_SENSITIVITY_K_MEV,
                positive=True,
            )
        ),
        "direct_check_y": list(
            _require_exact_decimal_list(
                calc["direct_check_y"],
                "direct_check_y",
                _SEALED_DIRECT_CHECK_Y,
                positive=True,
            )
        ),
        "working_precisions": list(_require_exact_precision_list(calc["working_precisions"])),
    }


def _mp(value: Any, name: str) -> mp.mpf:
    if isinstance(value, bool):
        raise IsospinJetError(f"{name} must be a finite real, not bool")
    try:
        result = mp.mpf(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IsospinJetError(f"{name} must be a finite real") from exc
    if not mp.isfinite(result):
        raise IsospinJetError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> mp.mpf:
    result = _mp(value, name)
    if result <= 0:
        raise IsospinJetError(f"{name} must be positive")
    return result


def _nonnegative(value: Any, name: str) -> mp.mpf:
    result = _mp(value, name)
    if result < 0:
        raise IsospinJetError(f"{name} must be nonnegative")
    return result


def _number(value: Any, digits: int = 18) -> str:
    result = _mp(value, "scientific value")
    return mp.nstr(result, digits)


def _relative_error(a: Any, b: Any) -> mp.mpf:
    a, b = _mp(a, "left value"), _mp(b, "right value")
    return abs(a - b) / max(abs(a), abs(b), mp.mpf(1))


def _validate_delta(delta: Any) -> mp.mpf:
    result = _mp(delta, "delta")
    if abs(result) > 1:
        raise IsospinJetError("delta must lie in [-1,1]")
    return result


def _validate_coupling(c_rho: Any) -> mp.mpf:
    """Validate the nonnegative constant ``C_rho=g_rho^2/m_rho^2``."""

    return _nonnegative(c_rho, "C_rho")


def load_declared_inputs(path: Path | str = INPUT_PATH) -> dict[str, Any]:
    """Load and validate the compact design/source input record.

    The calculation consumes only ``calculation_inputs``.  Literature fields
    are deliberately carried separately so changing a comparison reference
    cannot mutate any prediction row.
    """

    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IsospinJetError(f"cannot read input JSON {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 1
    ):
        raise IsospinJetError("isospin input JSON requires schema_version=1")
    calc = payload.get("calculation_inputs")
    refs = payload.get("descriptive_benchmarks")
    if not isinstance(calc, dict) or not isinstance(refs, dict):
        raise IsospinJetError(
            "input JSON requires separate calculation_inputs and descriptive_benchmarks"
        )
    _protocol_calculation_inputs(calc)
    # Keep the source input structure intact for provenance, but expose only
    # copies to callers so no calculation can mutate the loaded record.
    return copy.deepcopy(payload)


def _species_fermi(model: BulkModel, density: mp.mpf, y: mp.mpf) -> dict[str, mp.mpf]:
    """Analytic zero-temperature Fermi integrals for one spin-2 species."""

    density = _mp(density, "species density")
    y = _positive(y, "y")
    if density < 0:
        raise IsospinJetError("species density must be nonnegative")
    zero = mp.mpf(0)
    if density == 0:
        return {key: zero for key in ("k", "m", "ef", "energy", "pressure", "ns", "ns_m")}
    k = (3 * mp.pi**2 * density) ** (mp.mpf(1) / 3)
    m = model.MN * y
    ef = mp.sqrt(k * k + m * m)
    a = mp.asinh(k / m)
    energy = 1 / (8 * mp.pi**2) * (k * ef * (2 * k * k + m * m) - m**4 * a)
    pressure = 1 / (24 * mp.pi**2) * (
        k * ef * (2 * k * k - 3 * m * m) + 3 * m**4 * a
    )
    ns = m / (2 * mp.pi**2) * (k * ef - m * m * a)
    ns_m = 3 * ns / m - 3 * density / ef
    return {
        "k": k,
        "m": m,
        "ef": ef,
        "energy": energy,
        "pressure": pressure,
        "ns": ns,
        "ns_m": ns_m,
    }


class IsospinJet:
    """Two-species energy functional with a constant, eliminated rho term."""

    def __init__(self, model: BulkModel, c_rho: Any = 0, reference_y: Any | None = None):
        self.model = model
        self.C_rho = _validate_coupling(c_rho)
        self.reference_y = None if reference_y is None else _positive(reference_y, "reference_y")

    def rho_energy(self, n: Any, delta: Any) -> mp.mpf:
        n = _positive(n, "n")
        delta = _validate_delta(delta)
        return self.C_rho * n * n * delta * delta / 8

    def state(self, n: Any, delta: Any, y: Any) -> dict[str, mp.mpf]:
        """Return direct two-sea thermodynamics at a specified scalar field."""

        n = _positive(n, "n")
        delta = _validate_delta(delta)
        y = _positive(y, "y")
        nn, np_ = n * (1 + delta) / 2, n * (1 - delta) / 2
        fn, fp = _species_fermi(self.model, nn, y), _species_fermi(self.model, np_, y)
        u, uy, uyy = self.model.potential(y)
        vector = self.model.Cv * n * n / (2 * y * y)
        rho = self.rho_energy(n, delta)
        kinetic = fn["energy"] + fp["energy"]
        pressure_kin = fn["pressure"] + fp["pressure"]
        residual = self.model.MN * (fn["ns"] + fp["ns"]) + uy - self.model.Cv * n * n / y**3
        curvature = (
            uyy
            + self.model.MN**2 * (fn["ns_m"] + fp["ns_m"])
            + 3 * self.model.Cv * n * n / y**4
        )
        mu_n = fn["ef"] + self.model.Cv * n / y**2 + self.C_rho * n * delta / 4
        mu_p = fp["ef"] + self.model.Cv * n / y**2 - self.C_rho * n * delta / 4
        mu_fixed_delta = ((1 + delta) * mu_n + (1 - delta) * mu_p) / 2
        energy_total = kinetic + u + vector + rho
        pressure_total = pressure_kin - u + vector + rho
        return {
            "n": n,
            "delta": delta,
            "y": y,
            "nn": nn,
            "np": np_,
            "kinetic": kinetic,
            "pressure_kinetic": pressure_kin,
            "U": u,
            "Uy": uy,
            "Uyy": uyy,
            "vector": vector,
            "rho_energy": rho,
            "energy_total": energy_total,
            "pressure_total": pressure_total,
            "mu_n": mu_n,
            "mu_p": mu_p,
            "mu_fixed_delta": mu_fixed_delta,
            "residual": residual,
            "C_y": curvature,
            "fn": fn,
            "fp": fp,
        }

    def equilibrium(
        self, n: Any, delta: Any = 0, guess: Any | None = None
    ) -> dict[str, mp.mpf]:
        """Re-stationarize the scalar field for density and asymmetry."""

        n = _positive(n, "n")
        delta = _validate_delta(delta)
        if guess is None:
            if self.reference_y is None:
                base = self.model.equilibrium(n)
                y0 = base["y"]
            else:
                y0 = self.reference_y
        else:
            y0 = _positive(guess[0] if isinstance(guess, (tuple, list)) else guess, "guess")
        # A secant pair centred on the already selected local branch avoids
        # importing a global-minimum assumption into the inverse design.
        roots = (y0 * mp.mpf("0.999"), y0 * mp.mpf("1.001"))
        try:
            y = mp.findroot(lambda yy: self.state(n, delta, yy)["residual"], roots)
        except (ValueError, ZeroDivisionError) as exc:
            raise IsospinJetError(f"scalar re-stationarization failed: {exc}") from exc
        result = self.state(n, delta, y)
        if result["y"] <= 0 or result["C_y"] <= 0:
            raise IsospinJetError("re-stationarized scalar state is not locally stable")
        return result


def rho_thermodynamics(c_rho: Any, n: Any, delta: Any) -> dict[str, mp.mpf]:
    """Explicit species chemical-potential and pressure contribution of rho."""

    c_rho = _validate_coupling(c_rho)
    n = _positive(n, "n")
    delta = _validate_delta(delta)
    energy = c_rho * n * n * delta * delta / 8
    return {
        "energy_density": energy,
        "pressure": energy,
        "mu_fixed_delta": c_rho * n * delta**2 / 4,
        "mu_n": c_rho * n * delta / 4,
        "mu_p": -c_rho * n * delta / 4,
    }


def _near_degenerate(value: mp.mpf, scale: mp.mpf) -> bool:
    # This is a conditioning guard, not a physical uncertainty.  The fixed
    # threshold keeps a tiny B from becoming a huge squared-over-gap ratio.
    return abs(value) <= max(mp.mpf("1e-30"), 100 * mp.eps) * max(mp.mpf(1), abs(scale))


def local_jet_from_derivatives(
    *,
    n: Any,
    y: Any,
    k: Any,
    ef: Any,
    c_rho: Any,
    C: Any,
    B: Any,
    D: Any,
    label: str | None = None,
) -> dict[str, Any]:
    """Compute the accepted identities from a stationary symmetric state.

    This lower-level entry point is also used for synthetic exact/near
    degeneracy controls.  It never divides by ``B`` or ``Kfr-K``; the direct
    Hessian formula remains available whenever finite ``C>0`` is supplied.
    """

    n, y, k, ef = (_positive(n, "n"), _positive(y, "y"), _positive(k, "k"), _positive(ef, "E_F"))
    c_rho = _validate_coupling(c_rho)
    C, B, D = _mp(C, "C"), _mp(B, "B"), _mp(D, "D")
    if C <= 0:
        raise IsospinJetError("unstable scalar curvature C<=0")
    T = k * k / (6 * ef)
    u = k * k / (ef * ef)
    if u > 1:
        raise IsospinJetError("E_F must satisfy E_F>=k")
    T_y = -T * (1 - u) / y
    Kfr = 9 * n * D
    K = 9 * n * (D - B * B / C)
    gap = Kfr - K
    gap_scale = max(mp.mpf(1), abs(Kfr), abs(K))
    if gap < -max(mp.mpf("1e-40"), 100 * mp.eps) * gap_scale:
        raise IsospinJetError("Kfr-K<0: locally unstable or inconsistent derivatives")
    y_n = -B / C
    J = T + c_rho * n / 8
    L_fixed_y = (2 - u) * T
    L_kin = L_fixed_y + 3 * n * T_y * y_n
    L = L_kin + 3 * c_rho * n / 8
    S4_kin = k * k * (4 + 3 * u + 3 * u * u) / (648 * ef)
    relaxation = n * T_y * T_y / (2 * C)
    S4_hessian = S4_kin - relaxation
    Q = L - 3 * J + (1 + u) * T
    near_B = _near_degenerate(B, max(mp.mpf(1), abs(B)))
    near_gap = _near_degenerate(gap, gap_scale)
    if near_B or near_gap:
        S4_eliminated = None
        degeneracy_status = "NEAR_DEGENERATE_DIRECT_HESSIAN_ONLY"
    else:
        S4_eliminated = S4_kin - Q * Q / (2 * gap)
        degeneracy_status = "ELIMINATED_MIXING_FORMULA_RESOLVED"
    return {
        "label": label,
        "n": n,
        "y": y,
        "k": k,
        "ef": ef,
        "T": T,
        "u": u,
        "C_rho": c_rho,
        "C": C,
        "B": B,
        "D": D,
        "Kfr": Kfr,
        "K": K,
        "Kfr_minus_K": gap,
        "dy_dn": y_n,
        "y_delta2": -n * T_y / C,
        "T_y": T_y,
        "S2": J,
        "J": J,
        "L_fixed_y": L_fixed_y,
        "L_kin": L_kin,
        "L": L,
        "S4_kin": S4_kin,
        "S4_fixed_y": S4_kin,
        "S4_relaxation": relaxation,
        "S4_hessian": S4_hessian,
        "S4_relaxed": S4_hessian,
        "Q": Q,
        "S4_eliminated": S4_eliminated,
        "degeneracy_status": degeneracy_status,
        "identity_residual_K": gap - 9 * n * B * B / C,
        "rho_action_normalization": "C_rho*n^2*delta^2/8",
    }


def local_jet(state: Mapping[str, Any], c_rho: Any = 0, label: str | None = None) -> dict[str, Any]:
    """Compute the local jet from a symmetric ``BulkModel`` state."""

    delta = _validate_delta(state.get("delta", 0))
    if delta != 0:
        raise IsospinJetError("local_jet requires the symmetric delta=0 state")
    return local_jet_from_derivatives(
        n=state["n"],
        y=state["y"],
        k=state["k"],
        ef=state["ef"],
        c_rho=c_rho,
        C=state.get("C_y", state.get("C_y_MeV4")),
        B=state.get("B_y", state.get("B")),
        D=state["D"],
        label=label,
    )


def constant_rho_coupling(model: BulkModel, state: Mapping[str, Any], J: Any) -> mp.mpf:
    """Return C_rho fixed by the declared local J design input."""

    J = _positive(J, "J design")
    T = state["k"] ** 2 / (6 * state["ef"])
    c_rho = 8 * (J - T) / state["n"]
    if not mp.isfinite(c_rho) or c_rho < 0:
        raise IsospinJetError("declared J gives a negative/nonfinite rho coupling")
    return c_rho


def _energy_even_coefficients(
    jet_model: IsospinJet,
    n: mp.mpf,
    y0: mp.mpf,
    h: mp.mpf,
) -> dict[str, mp.mpf]:
    states = {
        "minus_2": jet_model.equilibrium(n, -2 * h, guess=(y0, y0))["energy_total"],
        "minus": jet_model.equilibrium(n, -h, guess=(y0, y0))["energy_total"],
        "zero": jet_model.equilibrium(n, 0, guess=(y0, y0))["energy_total"],
        "plus": jet_model.equilibrium(n, h, guess=(y0, y0))["energy_total"],
        "plus_2": jet_model.equilibrium(n, 2 * h, guess=(y0, y0))["energy_total"],
    }
    ys = {
        key: jet_model.equilibrium(n, delta, guess=(y0, y0))["y"]
        for key, delta in (
            ("minus_2", -2 * h),
            ("minus", -h),
            ("zero", mp.mpf(0)),
            ("plus", h),
            ("plus_2", 2 * h),
        )
    }
    s2 = (states["plus"] + states["minus"] - 2 * states["zero"]) / (2 * n * h**2)
    s4 = (
        states["plus_2"]
        - 4 * states["plus"]
        + 6 * states["zero"]
        - 4 * states["minus"]
        + states["minus_2"]
    ) / (24 * n * h**4)
    return {
        "S2": s2,
        "S4": s4,
        "y_delta2": (ys["plus"] + ys["minus"] - 2 * ys["zero"]) / (2 * h**2),
        "parity_energy": abs(states["plus"] - states["minus"])
        / max(abs(states["plus"]), abs(states["minus"]), mp.mpf(1)),
    }


def direct_restationarization_check(
    jet_model: IsospinJet,
    state: Mapping[str, Any],
    *,
    h_delta: Any = "0.002",
    h_density_fraction: Any = "0.0001",
) -> dict[str, Any]:
    """Convergent finite-difference check of L, S4 and scalar re-equilibration."""

    n, y0 = state["n"], state["y"]
    h = _positive(h_delta, "h_delta")
    h_n_frac = _positive(h_density_fraction, "h_density_fraction")
    if h >= mp.mpf("0.1"):
        raise IsospinJetError("h_delta is too large for a local Taylor check")
    coarse = _energy_even_coefficients(jet_model, n, y0, h)
    fine = _energy_even_coefficients(jet_model, n, y0, h / 2)
    s2 = fine["S2"] + (fine["S2"] - coarse["S2"]) / 3
    s4 = fine["S4"] + (fine["S4"] - coarse["S4"]) / 3
    # At n +/- dn use the same small-delta Richardson estimator.  This
    # re-equilibrates the scalar branch at each density rather than taking a
    # derivative of a frozen-y expression.
    dn = n * h_n_frac
    density_s2 = {}
    for key, nn in (("minus", n - dn), ("plus", n + dn)):
        local0 = jet_model.equilibrium(nn, 0, guess=(y0, y0))
        c = _energy_even_coefficients(jet_model, nn, local0["y"], h / 2)
        density_s2[key] = c["S2"]
    L_fd = 3 * n * (density_s2["plus"] - density_s2["minus"]) / (2 * dn)
    expected = local_jet(state, jet_model.C_rho)
    y_delta2_expected = -n * expected["T_y"] / expected["C"]
    return {
        "h_delta": h,
        "h_density_fraction": h_n_frac,
        "S2_direct_richardson": s2,
        "S4_direct_richardson": s4,
        "L_direct_density_fd": L_fd,
        "y_delta2_direct": fine["y_delta2"] + (fine["y_delta2"] - coarse["y_delta2"]) / 3,
        "parity_energy_max": max(coarse["parity_energy"], fine["parity_energy"]),
        "S2_identity_residual": s2 - expected["J"],
        "S4_hessian_residual": s4 - expected["S4_hessian"],
        "L_identity_residual": L_fd - expected["L"],
        "y_delta2_residual": (
            fine["y_delta2"] + (fine["y_delta2"] - coarse["y_delta2"]) / 3
        ) - y_delta2_expected,
    }


def direct_species_quadrature_check(
    model: BulkModel,
    state: Mapping[str, Any],
    *,
    delta: Any = "0.2",
) -> dict[str, Any]:
    """Independent momentum quadratures for both asymmetric species."""

    delta = _validate_delta(delta)
    n, y = state["n"], state["y"]
    nn, np_ = n * (1 + delta) / 2, n * (1 - delta) / 2
    rows = []
    for name, density in (("neutron", nn), ("proton", np_)):
        analytic = _species_fermi(model, density, y)
        # spin degeneracy is two, hence d/(2*pi^2)=1/pi^2.
        factor = 1 / mp.pi**2
        energy_quad = factor * mp.quad(
            lambda momentum: momentum**2 * mp.sqrt(momentum**2 + analytic["m"] ** 2),
            [0, analytic["k"]],
        )
        pressure_quad = factor / 3 * mp.quad(
            lambda momentum: momentum**4 / mp.sqrt(momentum**2 + analytic["m"] ** 2),
            [0, analytic["k"]],
        )
        ns_quad = factor * mp.quad(
            lambda momentum: momentum**2
            * analytic["m"]
            / mp.sqrt(momentum**2 + analytic["m"] ** 2),
            [0, analytic["k"]],
        )
        rows.append(
            {
                "species": name,
                "energy_relative_error": _relative_error(analytic["energy"], energy_quad),
                "pressure_relative_error": _relative_error(analytic["pressure"], pressure_quad),
                "scalar_density_relative_error": _relative_error(analytic["ns"], ns_quad),
            }
        )
    return {"delta": delta, "species": rows}


def _shown_jet(jet: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "C_rho",
        "T",
        "u",
        "C",
        "B",
        "D",
        "Kfr",
        "K",
        "Kfr_minus_K",
        "dy_dn",
        "y_delta2",
        "T_y",
        "S2",
        "J",
        "L_fixed_y",
        "L_kin",
        "L",
        "S4_kin",
        "S4_fixed_y",
        "S4_relaxation",
        "S4_hessian",
        "S4_relaxed",
        "Q",
        "S4_eliminated",
        "identity_residual_K",
    )
    result = {key: (None if jet[key] is None else _number(jet[key])) for key in keys}
    result["degeneracy_status"] = jet["degeneracy_status"]
    result["rho_action_normalization"] = jet["rho_action_normalization"]
    return result


def _shown_state(state: Mapping[str, Any], model: BulkModel) -> dict[str, Any]:
    mu = state.get("mu_fixed_delta", state.get("mu"))
    mu_n = state.get("mu_n", mu)
    mu_p = state.get("mu_p", mu)
    return {
        "n_fm3": _number(state["n"] / model.hbarc**3),
        "delta": _number(state.get("delta", 0)),
        "y": _number(state["y"]),
        "pressure_MeV_fm3": _number(state["pressure_total"] / model.hbarc**3),
        "binding_MeV": _number(state["energy_total"] / state["n"] - model.MN),
        "mu_fixed_delta_MeV": _number(mu),
        "mu_n_MeV": _number(mu_n),
        "mu_p_MeV": _number(mu_p),
        "C_y_MeV4": _number(state["C_y"]),
        "stationarity_residual_relative": _number(
            abs(state["residual"]) / max(abs(state["energy_total"]), mp.mpf(1))
        ),
    }


def _candidate_row(
    *,
    label: str,
    model: BulkModel,
    state: Mapping[str, Any],
    c_rho: Any,
    J_role: str,
    global_status: str,
    direct_checks: bool = False,
) -> dict[str, Any]:
    jet_model = IsospinJet(model, c_rho, reference_y=state["y"])
    jet = local_jet(state, jet_model.C_rho, label=label)
    row: dict[str, Any] = {
        "label": label,
        "design_role": J_role,
        "physical_interpretation_status": PHYSICAL_INTERPRETATION,
        "global_candidate_status": global_status,
        "local_jet_status": "COMPUTED_LOCAL_STABLE" if state["C_y"] > 0 else "REJECTED_UNSTABLE_LOCAL",
        "isospin_local_status": (
            "S2_POSITIVE_LOCAL_ISOVECTOR_CURVATURE"
            if jet["J"] > 0
            else "S2_NONPOSITIVE_LOCAL_ISOVECTOR_CURVATURE"
        ),
        "S4_sign_status": (
            "S4_NONNEGATIVE_LOCAL_TAYLOR_COEFFICIENT"
            if jet["S4_hessian"] >= 0
            else "S4_NEGATIVE_LOCAL_TAYLOR_COEFFICIENT_NOT_BY_ITSELF_ISOSPIN_INSTABILITY"
        ),
        "state": _shown_state(state, model),
        "jet": _shown_jet(jet),
        "rho_thermodynamics_at_delta_0": {
            key: _number(value)
            for key, value in rho_thermodynamics(jet_model.C_rho, state["n"], 0).items()
        },
        "rho_thermodynamics_at_delta_0.2": {
            key: _number(value)
            for key, value in rho_thermodynamics(jet_model.C_rho, state["n"], "0.2").items()
        },
    }
    if direct_checks:
        direct = direct_restationarization_check(jet_model, state)
        row["direct_restationarization_check"] = {
            key: (None if value is None else _number(value))
            for key, value in direct.items()
        }
        row["direct_species_quadrature_check"] = {
            key: value
            if key not in {"delta", "species"}
            else (
                _number(value)
                if key == "delta"
                else [
                    {
                        item_key: (
                            item_value
                            if item_key == "species"
                            else _number(item_value)
                        )
                        for item_key, item_value in item.items()
                    }
                    for item in value
                ]
            )
            for key, value in direct_species_quadrature_check(model, state).items()
        }
    return row


def _benchmark_comparisons(
    refs: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build descriptive conditional references without feeding back to rows."""

    roca = refs.get("roca_maza_eq14")
    asy = refs.get("asy_eos")
    if not isinstance(roca, dict) or not isinstance(asy, dict):
        raise IsospinJetError("required descriptive benchmark records are missing")
    intercept = _mp(roca["intercept_MeV"], "Roca-Maza intercept")
    slope = _mp(roca["slope"], "Roca-Maza slope")
    intercept_sigma = _nonnegative(roca["intercept_theory_sigma_MeV"], "intercept sigma")
    slope_exp_sigma = _nonnegative(roca["slope_exp_sigma"], "slope exp sigma")
    slope_theory_sigma = _nonnegative(roca["slope_theory_sigma"], "slope theory sigma")
    j_ref = _positive(roca["reference_J_MeV"], "Roca-Maza reference J")
    j_ref_sigma = _nonnegative(roca["reference_J_sigma_MeV"], "Roca-Maza J sigma")
    roca_rows = []
    for row in rows:
        J = _mp(row["jet"]["J"], f"J in {row['label']}")
        roca_rows.append(
            {
                "label": row["label"],
                "J_MeV": _number(J),
                "L_conditional_central_MeV": _number(intercept + slope * J),
                "uncertainty_components_MeV": {
                    "intercept_theory": _number(intercept_sigma),
                    "slope_experimental_times_J": _number(slope_exp_sigma * J),
                    "slope_theory_times_J": _number(slope_theory_sigma * J),
                    "J_assumption_only_at_reference_J": _number(slope * j_ref_sigma)
                    if J == j_ref
                    else None,
                },
                "comparison_status": "DESCRIPTIVE_CONDITIONAL_REFERENCE_NOT_A_TEST",
            }
        )
    asy_J = _positive(asy["assumed_J_MeV"], "ASY-EOS assumed J")
    asy_L = _mp(asy["L_MeV"], "ASY-EOS L")
    asy_sigma = _nonnegative(asy["L_sigma_MeV"], "ASY-EOS sigma")
    asy_rows = []
    for row in rows:
        J = _mp(row["jet"]["J"], f"J in {row['label']}")
        matched = J == asy_J
        asy_rows.append(
            {
                "label": row["label"],
                "model_J_MeV": _number(J),
                "source_assumed_J_MeV": _number(asy_J),
                "L_MeV": _number(asy_L) if matched else None,
                "sigma_MeV": _number(asy_sigma) if matched else None,
                "comparison_status": (
                    "DESCRIPTIVE_ONLY_MATCHED_J_ASSUMPTION"
                    if matched
                    else "UNMATCHED_J_ASSUMPTION_NO_DIRECT_COMPARISON"
                ),
            }
        )
    narrative = []
    for key in ("drischler", "somasundaram"):
        item = refs.get(key)
        if not isinstance(item, dict):
            raise IsospinJetError(f"missing narrative benchmark: {key}")
        narrative.append(
            {
                "source": item["source"],
                "comparison_status": "NARRATIVE_DEFINITION_BOUNDARY_ONLY",
                "definition_boundary": item["definition_boundary"],
            }
        )
    return {
        "roca_maza_eq14": {
            "relation": f"L = {_number(intercept)} + {_number(slope)} J",
            "rows": roca_rows,
        },
        "asy_eos": {"rows": asy_rows},
        "narrative_only": narrative,
        "likelihood_used": False,
        "empirical_pass_fail_used": False,
        "transfer_to_homogeneous_local_eos": "UNVALIDATED",
    }


def _calculation_inputs(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _shown_calculation_inputs(
        _protocol_calculation_inputs(payload["calculation_inputs"])
    )


def _shown_calculation_inputs(calc: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical serialized form used at the result boundary."""

    return {
        "n0_fm3": _number(_mp(calc["n0_fm3"], "n0_fm3")),
        "binding_MeV": _number(_mp(calc["binding_MeV"], "binding_MeV")),
        "Ktarget_MeV": _number(_positive(calc["Ktarget_MeV"], "Ktarget_MeV")),
        "J_design_MeV": _number(_positive(calc["J_design_MeV"], "J_design_MeV")),
        "J_reference_controls_MeV": [
            _number(_positive(value, "J control"))
            for value in calc["J_reference_controls_MeV"]
        ],
        "residual_y": [
            _number(_positive(value, "residual y")) for value in calc["residual_y"]
        ],
        "sensitivity_K_MeV": [
            _number(_positive(value, "sensitivity K"))
            for value in calc["sensitivity_K_MeV"]
        ],
        "direct_check_y": [
            _number(_positive(value, "direct y")) for value in calc["direct_check_y"]
        ],
        "working_precisions": [int(value) for value in calc["working_precisions"]],
    }


def _validate_shown_calculation_inputs(
    shown: Any, calc: Mapping[str, Any], path: str
) -> None:
    """Validate serialized inputs by meaning, not by incidental mp.nstr spelling."""

    shown = _require_output_mapping(shown, path)
    _require_output_fields(
        shown,
        (
            "n0_fm3",
            "binding_MeV",
            "Ktarget_MeV",
            "J_design_MeV",
            "J_reference_controls_MeV",
            "residual_y",
            "sensitivity_K_MeV",
            "direct_check_y",
            "working_precisions",
        ),
        path,
    )
    for field in ("n0_fm3", "binding_MeV", "Ktarget_MeV", "J_design_MeV"):
        expected = _semantic_decimal(calc[field], f"{path}.{field} expected")
        actual = _semantic_decimal(shown[field], f"{path}.{field}")
        if actual != expected:
            raise IsospinJetError(f"{path}.{field} is not the sealed value")
    for field in (
        "J_reference_controls_MeV",
        "residual_y",
        "sensitivity_K_MeV",
        "direct_check_y",
    ):
        actual_values = shown[field]
        expected_values = calc[field]
        if not isinstance(actual_values, list) or len(actual_values) != len(expected_values):
            raise IsospinJetError(f"{path}.{field} has incomplete serialized values")
        for index, (actual, expected) in enumerate(zip(actual_values, expected_values)):
            if _semantic_decimal(actual, f"{path}.{field}[{index}]") != _semantic_decimal(
                expected, f"{path}.{field} expected[{index}]"
            ):
                raise IsospinJetError(f"{path}.{field}[{index}] is not the sealed value")
    actual_precisions = shown["working_precisions"]
    if actual_precisions != list(calc["working_precisions"]):
        raise IsospinJetError(f"{path}.working_precisions is not the sealed pair")


def _require_output_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise IsospinJetError(f"{path} must be an object")
    return value


def _require_output_fields(
    mapping: Mapping[str, Any], fields: Sequence[str], path: str
) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise IsospinJetError(f"{path} missing required field(s): {', '.join(missing)}")


def _finite_output_fields(
    mapping: Mapping[str, Any], fields: Sequence[str], path: str, *, allow_none: Sequence[str] = ()
) -> None:
    allowed_none = set(allow_none)
    for field in fields:
        if field not in mapping:
            raise IsospinJetError(f"{path}.{field} is missing")
        value = mapping[field]
        if value is None and field in allowed_none:
            continue
        if value is None:
            raise IsospinJetError(f"{path}.{field} must be finite, not null")
        _mp(value, f"{path}.{field}")


def _validate_output_rows(
    rows: Any, expected_labels: Sequence[str], path: str
) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise IsospinJetError(f"{path} must be a list")
    labels = []
    for index, row in enumerate(rows):
        row_mapping = _require_output_mapping(row, f"{path}[{index}]")
        label = row_mapping.get("label")
        if not isinstance(label, str) or not label:
            raise IsospinJetError(f"{path}[{index}].label must be a nonempty string")
        labels.append(label)
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    expected = set(expected_labels)
    actual = set(labels)
    if duplicates:
        raise IsospinJetError(f"{path} contains duplicate output row(s): {duplicates}")
    if len(labels) != len(expected) or actual != expected:
        missing = sorted(expected - actual)
        foreign = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if foreign:
            details.append(f"foreign={foreign}")
        raise IsospinJetError(
            f"{path} must contain the complete unique sealed row set "
            f"({', '.join(details)})"
        )
    return [row for row in rows]


def _validate_direct_restationarization(
    check: Any, path: str
) -> None:
    check = _require_output_mapping(check, path)
    _finite_output_fields(check, _DIRECT_RESTATIONARIZATION_FIELDS, path)
    h_delta = _positive(check["h_delta"], f"{path}.h_delta")
    h_density_fraction = _positive(
        check["h_density_fraction"], f"{path}.h_density_fraction"
    )
    if h_delta >= mp.mpf("0.1"):
        raise IsospinJetError(f"{path}.h_delta is outside the local Taylor stencil")
    if h_density_fraction >= mp.mpf("0.1"):
        raise IsospinJetError(
            f"{path}.h_density_fraction is outside the density finite-difference stencil"
        )
    scales = {
        "S2_identity_residual": max(
            mp.mpf(1), abs(_mp(check["S2_direct_richardson"], f"{path}.S2"))
        ),
        "S4_hessian_residual": max(
            mp.mpf(1), abs(_mp(check["S4_direct_richardson"], f"{path}.S4"))
        ),
        "L_identity_residual": max(
            mp.mpf(1), abs(_mp(check["L_direct_density_fd"], f"{path}.L"))
        ),
        "y_delta2_residual": max(
            mp.mpf(1), abs(_mp(check["y_delta2_direct"], f"{path}.y_delta2"))
        ),
        "parity_energy_max": mp.mpf(1),
    }
    for field, (absolute_text, relative_text) in _DIRECT_RESIDUAL_LIMITS.items():
        residual = abs(_mp(check[field], f"{path}.{field}"))
        limit = max(
            mp.mpf(absolute_text),
            mp.mpf(relative_text) * scales[field],
        )
        if residual > limit:
            raise IsospinJetError(
                f"{path}.{field} residual {mp.nstr(residual, 8)} exceeds "
                f"numerical gate {mp.nstr(limit, 8)}"
            )
    if _mp(check["parity_energy_max"], f"{path}.parity_energy_max") < 0:
        raise IsospinJetError(f"{path}.parity_energy_max must be nonnegative")


def _validate_direct_species_quadrature(check: Any, path: str) -> None:
    check = _require_output_mapping(check, path)
    _require_output_fields(check, ("delta", "species"), path)
    if _semantic_decimal(check["delta"], f"{path}.delta") != Decimal("0.2"):
        raise IsospinJetError(f"{path}.delta must be the prescribed 0.2 check")
    species = check["species"]
    if not isinstance(species, list):
        raise IsospinJetError(f"{path}.species must be a list")
    names = [
        item.get("species") if isinstance(item, dict) else None for item in species
    ]
    expected_names = {"neutron", "proton"}
    if len(names) != len(expected_names) or set(names) != expected_names:
        raise IsospinJetError(
            f"{path}.species must contain exactly neutron and proton rows"
        )
    if len(set(names)) != len(names):
        raise IsospinJetError(f"{path}.species contains duplicate species rows")
    for index, item in enumerate(species):
        item = _require_output_mapping(item, f"{path}.species[{index}]")
        _require_output_fields(item, ("species", *_DIRECT_SPECIES_FIELDS), f"{path}.species[{index}]")
        for field in _DIRECT_SPECIES_FIELDS:
            error = _mp(item[field], f"{path}.species[{index}].{field}")
            if error < 0 or error > _DIRECT_SPECIES_ERROR_LIMIT:
                raise IsospinJetError(
                    f"{path}.species[{index}].{field} exceeds quadrature gate "
                    f"{mp.nstr(_DIRECT_SPECIES_ERROR_LIMIT, 8)}"
                )


def _validate_candidate_row(
    row: Mapping[str, Any],
    path: str,
    *,
    chosen_y: str | None = None,
    sensitivity_K: str | None = None,
    direct_required: bool = False,
) -> None:
    _require_output_fields(
        row,
        (
            "label",
            "design_role",
            "physical_interpretation_status",
            "global_candidate_status",
            "local_jet_status",
            "isospin_local_status",
            "S4_sign_status",
            "state",
            "jet",
            "rho_thermodynamics_at_delta_0",
            "rho_thermodynamics_at_delta_0.2",
        ),
        path,
    )
    for field in (
        "label",
        "design_role",
        "physical_interpretation_status",
        "global_candidate_status",
        "local_jet_status",
        "isospin_local_status",
        "S4_sign_status",
    ):
        if not isinstance(row[field], str) or not row[field]:
            raise IsospinJetError(f"{path}.{field} must be a nonempty string")
    if row["physical_interpretation_status"] != PHYSICAL_INTERPRETATION:
        raise IsospinJetError(f"{path}.physical_interpretation_status is not the sealed scope")
    if not any(
        row["global_candidate_status"].startswith(prefix)
        for prefix in _ALLOWED_GLOBAL_STATUS_PREFIXES
    ):
        raise IsospinJetError(
            f"{path}.global_candidate_status cannot be collapsed into a numerical status"
        )
    if row["local_jet_status"] != "COMPUTED_LOCAL_STABLE":
        raise IsospinJetError(f"{path}.local_jet_status is not a certified local state")
    state = _require_output_mapping(row["state"], f"{path}.state")
    _finite_output_fields(state, _EXPECTED_STATE_NUMERIC_FIELDS, f"{path}.state")
    if _mp(state["n_fm3"], f"{path}.state.n_fm3") <= 0:
        raise IsospinJetError(f"{path}.state.n_fm3 must be positive")
    if abs(_mp(state["delta"], f"{path}.state.delta")) > 1:
        raise IsospinJetError(f"{path}.state.delta is outside [-1,1]")
    if _mp(state["C_y_MeV4"], f"{path}.state.C_y_MeV4") <= 0:
        raise IsospinJetError(f"{path}.state.C_y_MeV4 must be positive")
    jet = _require_output_mapping(row["jet"], f"{path}.jet")
    _finite_output_fields(jet, _EXPECTED_JET_NUMERIC_FIELDS, f"{path}.jet")
    if jet.get("S4_eliminated") is None and jet.get("degeneracy_status") != "NEAR_DEGENERATE_DIRECT_HESSIAN_ONLY":
        raise IsospinJetError(f"{path}.jet.S4_eliminated is missing outside degeneracy")
    if not isinstance(jet.get("degeneracy_status"), str) or not jet["degeneracy_status"]:
        raise IsospinJetError(f"{path}.jet.degeneracy_status must be a status string")
    if jet.get("rho_action_normalization") != "C_rho*n^2*delta^2/8":
        raise IsospinJetError(f"{path}.jet rho normalization is not the sealed one")
    for field in ("rho_thermodynamics_at_delta_0", "rho_thermodynamics_at_delta_0.2"):
        thermo = _require_output_mapping(row[field], f"{path}.{field}")
        _finite_output_fields(thermo, _EXPECTED_THERMO_NUMERIC_FIELDS, f"{path}.{field}")
    if chosen_y is not None:
        if "chosen_y" not in row:
            raise IsospinJetError(f"{path}.chosen_y is missing")
        if _semantic_decimal(row["chosen_y"], f"{path}.chosen_y") != Decimal(chosen_y):
            raise IsospinJetError(f"{path}.chosen_y does not match the sealed row identity")
    elif "chosen_y" in row:
        raise IsospinJetError(f"{path}.chosen_y is not allowed for this control row")
    if sensitivity_K is not None:
        if "K_design_sensitivity_MeV" not in row:
            raise IsospinJetError(f"{path}.K_design_sensitivity_MeV is missing")
        if _semantic_decimal(row["K_design_sensitivity_MeV"], f"{path}.K_design_sensitivity_MeV") != Decimal(sensitivity_K):
            raise IsospinJetError(f"{path}.K_design_sensitivity_MeV does not match the sealed row identity")
    elif "K_design_sensitivity_MeV" in row:
        raise IsospinJetError(f"{path}.K_design_sensitivity_MeV is not allowed for this row")
    direct_keys = (
        "direct_restationarization_check",
        "direct_species_quadrature_check",
    )
    if direct_required:
        for field in direct_keys:
            if field not in row:
                raise IsospinJetError(f"{path}.{field} is missing from a prescribed direct check")
        _validate_direct_restationarization(
            row["direct_restationarization_check"],
            f"{path}.direct_restationarization_check",
        )
        _validate_direct_species_quadrature(
            row["direct_species_quadrature_check"],
            f"{path}.direct_species_quadrature_check",
        )
    elif any(field in row for field in direct_keys):
        raise IsospinJetError(f"{path} contains an unprescribed direct-check row")


def _validate_descriptive_comparisons(
    comparisons: Any, expected_labels: Sequence[str], path: str
) -> None:
    comparisons = _require_output_mapping(comparisons, path)
    _require_output_fields(
        comparisons,
        (
            "roca_maza_eq14",
            "asy_eos",
            "narrative_only",
            "likelihood_used",
            "empirical_pass_fail_used",
            "transfer_to_homogeneous_local_eos",
        ),
        path,
    )
    if comparisons["likelihood_used"] is not False:
        raise IsospinJetError(f"{path}.likelihood_used must remain false")
    if comparisons["empirical_pass_fail_used"] is not False:
        raise IsospinJetError(f"{path}.empirical_pass_fail_used must remain false")
    if comparisons["transfer_to_homogeneous_local_eos"] != "UNVALIDATED":
        raise IsospinJetError(f"{path} has an unvalidated transfer status")
    roca = _require_output_mapping(comparisons["roca_maza_eq14"], f"{path}.roca_maza_eq14")
    _require_output_fields(roca, ("relation", "rows"), f"{path}.roca_maza_eq14")
    roca_rows = _validate_output_rows(roca["rows"], expected_labels, f"{path}.roca_maza_eq14.rows")
    for index, row in enumerate(roca_rows):
        row_path = f"{path}.roca_maza_eq14.rows[{index}]"
        _require_output_fields(
            row,
            ("label", "J_MeV", "L_conditional_central_MeV", "uncertainty_components_MeV", "comparison_status"),
            row_path,
        )
        _finite_output_fields(
            row,
            ("J_MeV", "L_conditional_central_MeV"),
            row_path,
        )
        if row["comparison_status"] != "DESCRIPTIVE_CONDITIONAL_REFERENCE_NOT_A_TEST":
            raise IsospinJetError(f"{row_path}.comparison_status promotes a reference")
        components = _require_output_mapping(row["uncertainty_components_MeV"], f"{row_path}.uncertainty_components_MeV")
        _require_output_fields(
            components,
            (
                "intercept_theory",
                "slope_experimental_times_J",
                "slope_theory_times_J",
                "J_assumption_only_at_reference_J",
            ),
            f"{row_path}.uncertainty_components_MeV",
        )
        _finite_output_fields(
            components,
            (
                "intercept_theory",
                "slope_experimental_times_J",
                "slope_theory_times_J",
            ),
            f"{row_path}.uncertainty_components_MeV",
        )
        if components["J_assumption_only_at_reference_J"] is not None:
            _mp(
                components["J_assumption_only_at_reference_J"],
                f"{row_path}.uncertainty_components_MeV.J_assumption_only_at_reference_J",
            )
    asy = _require_output_mapping(comparisons["asy_eos"], f"{path}.asy_eos")
    _require_output_fields(asy, ("rows",), f"{path}.asy_eos")
    asy_rows = _validate_output_rows(asy["rows"], expected_labels, f"{path}.asy_eos.rows")
    for index, row in enumerate(asy_rows):
        row_path = f"{path}.asy_eos.rows[{index}]"
        _require_output_fields(
            row,
            (
                "label",
                "model_J_MeV",
                "source_assumed_J_MeV",
                "L_MeV",
                "sigma_MeV",
                "comparison_status",
            ),
            row_path,
        )
        _finite_output_fields(row, ("model_J_MeV", "source_assumed_J_MeV"), row_path)
        if row["L_MeV"] is not None:
            _mp(row["L_MeV"], f"{row_path}.L_MeV")
        if row["sigma_MeV"] is not None:
            _mp(row["sigma_MeV"], f"{row_path}.sigma_MeV")
        if row["comparison_status"] not in {
            "DESCRIPTIVE_ONLY_MATCHED_J_ASSUMPTION",
            "UNMATCHED_J_ASSUMPTION_NO_DIRECT_COMPARISON",
        }:
            raise IsospinJetError(f"{row_path}.comparison_status is invalid")
    narrative = comparisons["narrative_only"]
    if not isinstance(narrative, list) or len(narrative) != 2:
        raise IsospinJetError(f"{path}.narrative_only must retain both source-boundary records")
    for index, item in enumerate(narrative):
        item = _require_output_mapping(item, f"{path}.narrative_only[{index}]")
        _require_output_fields(item, ("source", "comparison_status", "definition_boundary"), f"{path}.narrative_only[{index}]")
        if item["comparison_status"] != "NARRATIVE_DEFINITION_BOUNDARY_ONLY":
            raise IsospinJetError(f"{path}.narrative_only[{index}] promotes a source boundary")


def _validate_calculation_result(
    result: Any, payload: Mapping[str, Any], *, precision: int
) -> None:
    """Validate one complete precision state before it can be aggregated."""

    result = _require_output_mapping(result, f"precision_{precision}_result")
    if result.get("status") != "COMPUTED_LOCAL_ISOSPIN_JET_NO_EMPIRICAL_PASS":
        raise IsospinJetError(f"precision {precision} returned an invalid computational status")
    if result.get("schema") != SCHEMA:
        raise IsospinJetError(f"precision {precision} returned an invalid result schema")
    if isinstance(result.get("evidence_weight"), bool) or result.get("evidence_weight") != 0.0:
        raise IsospinJetError(f"precision {precision} returned a nonzero evidence weight")
    calc = _protocol_calculation_inputs(payload["calculation_inputs"])
    _validate_shown_calculation_inputs(
        result.get("calculation_inputs"), calc, f"precision_{precision}.calculation_inputs"
    )
    predictions = _require_output_mapping(result.get("predictions"), f"precision_{precision}.predictions")
    _require_output_fields(
        predictions,
        ("controls", "inverse_design_grid", "conditional_J_controls", "design_sensitivity"),
        f"precision_{precision}.predictions",
    )
    y_labels = [_number(value, 6) for value in calc["residual_y"]]
    j_design = _number(calc["J_design_MeV"], 6)
    j_labels = [j_design] + [_number(value, 6) for value in calc["J_reference_controls_MeV"]]
    inverse_labels = [f"inverse_y_{y}_J_{j_design}" for y in y_labels]
    conditional_labels = [
        f"inverse_y_{y}_conditional_J_{j}"
        for y in y_labels
        for j in j_labels
    ]
    sensitivity_y = _number("0.90", 6)
    sensitivity_K = [_number(value, 6) for value in calc["sensitivity_K_MeV"]]
    sensitivity_labels = [
        f"design_sensitivity_y_{sensitivity_y}_K_{value}" for value in sensitivity_K
    ]
    control_rows = _validate_output_rows(
        predictions["controls"],
        ("baseline_no_rho_original", "two_target_quartic_no_rho_control"),
        f"precision_{precision}.predictions.controls",
    )
    for index, row in enumerate(control_rows):
        _validate_candidate_row(row, f"precision_{precision}.predictions.controls[{index}]")
    inverse_rows = _validate_output_rows(
        predictions["inverse_design_grid"],
        inverse_labels,
        f"precision_{precision}.predictions.inverse_design_grid",
    )
    direct_targets = {_number(value, 6) for value in calc["direct_check_y"]}
    inverse_y_by_label = dict(zip(inverse_labels, y_labels))
    for index, row in enumerate(inverse_rows):
        row_y = inverse_y_by_label[row["label"]]
        _validate_candidate_row(
            row,
            f"precision_{precision}.predictions.inverse_design_grid[{index}]",
            chosen_y=row_y,
            direct_required=row_y in direct_targets,
        )
    conditional_rows = _validate_output_rows(
        predictions["conditional_J_controls"],
        conditional_labels,
        f"precision_{precision}.predictions.conditional_J_controls",
    )
    conditional_y_by_label = {
        label: y for label, y in zip(conditional_labels, (y for y in y_labels for _ in j_labels))
    }
    for index, row in enumerate(conditional_rows):
        row_y = conditional_y_by_label[row["label"]]
        _validate_candidate_row(
            row,
            f"precision_{precision}.predictions.conditional_J_controls[{index}]",
            chosen_y=row_y,
        )
    sensitivity_rows = _validate_output_rows(
        predictions["design_sensitivity"],
        sensitivity_labels,
        f"precision_{precision}.predictions.design_sensitivity",
    )
    sensitivity_K_by_label = dict(zip(sensitivity_labels, sensitivity_K))
    for index, row in enumerate(sensitivity_rows):
        _validate_candidate_row(
            row,
            f"precision_{precision}.predictions.design_sensitivity[{index}]",
            sensitivity_K=sensitivity_K_by_label[row["label"]],
        )
    comparison_labels = inverse_labels + conditional_labels
    _validate_descriptive_comparisons(
        result.get("descriptive_comparisons"),
        comparison_labels,
        f"precision_{precision}.descriptive_comparisons",
    )
    scope = _require_output_mapping(result.get("scope"), f"precision_{precision}.scope")
    _require_output_fields(
        scope,
        (
            "rho_extension",
            "baseline_source_complete_action_changed",
            "calibration_roles",
            "local_domain",
            "global_status_rule",
            "uncertainty_rule",
            "empirical_rule",
        ),
        f"precision_{precision}.scope",
    )
    if scope["baseline_source_complete_action_changed"] is not False:
        raise IsospinJetError(f"precision {precision} reports a modified baseline action")


def _run_calculation(dps: int, input_path: Path) -> dict[str, Any]:
    if dps < 80:
        raise IsospinJetError("verification requires at least 80 decimal digits")
    payload = load_declared_inputs(input_path)
    calc = _protocol_calculation_inputs(payload["calculation_inputs"])
    with mp.workdps(dps):
        n0_fm3 = _positive(calc["n0_fm3"], "n0_fm3")
        binding = _mp(calc["binding_MeV"], "binding_MeV")
        Ktarget = _positive(calc["Ktarget_MeV"], "Ktarget_MeV")
        j_design = _positive(calc["J_design_MeV"], "J_design_MeV")
        base = BulkModel()
        n0 = n0_fm3 * base.hbarc**3

        original = base.equilibrium(n0)
        original_row = _candidate_row(
            label="baseline_no_rho_original",
            model=base,
            state=original,
            c_rho=0,
            J_role="BASELINE_NO_RHO_CONTROL",
            global_status="BASELINE_GLOBAL_STATUS_PRESERVED_NOT_REASSESSED",
            direct_checks=False,
        )
        calibrated, calibrated_state = quartic_calibration(
            n_fm3=_number(n0_fm3), binding=_number(binding)
        )
        calibrated_row = _candidate_row(
            label="two_target_quartic_no_rho_control",
            model=calibrated,
            state=calibrated_state,
            c_rho=0,
            J_role="TWO_TARGET_QUARTIC_NO_RHO_CONTROL",
            global_status="TWO_TARGET_QUARTIC_CONTROL_GLOBAL_STATUS_PRESERVED",
            direct_checks=False,
        )

        inverse_rows: list[dict[str, Any]] = []
        inverse_models: dict[str, tuple[BulkModel, Mapping[str, Any], Mapping[str, Any]]] = {}
        for y_text in calc["residual_y"]:
            y_label = _number(_positive(y_text, "residual y"), 6)
            model, state, jet = inverse_potential_jet(
                y_label,
                K_target=_number(Ktarget),
                n_fm3=_number(n0_fm3),
                binding=_number(binding),
            )
            inverse_models[y_label] = (model, state, jet)
            global_status = (
                "LOCAL_RESPONSE_RETAINED_GLOBAL_POTENTIAL_NONNEGATIVE"
                if jet["globally_nonnegative_potential"]
                else "REJECTED_GLOBAL_POTENTIAL_UNSTABLE_LOCAL_CANDIDATE_RETAINED"
            )
            c_rho = constant_rho_coupling(model, state, j_design)
            row = _candidate_row(
                label=f"inverse_y_{y_label}_J_{_number(j_design, 6)}",
                model=model,
                state=state,
                c_rho=c_rho,
                J_role="NOMINAL_J32_DESIGN_INPUT",
                global_status=global_status,
                direct_checks=y_label in {_number(v, 6) for v in calc["direct_check_y"]},
            )
            row["chosen_y"] = y_label
            row["calibration_inputs"] = {
                "n_fm3": _number(n0_fm3),
                "binding_MeV": _number(binding),
                "Ktarget_MeV": _number(Ktarget),
                "J_design_MeV": _number(j_design),
                "chosen_y_is_design_not_observation": True,
            }
            inverse_rows.append(row)

        # Conditional J controls use the same predeclared isoscalar rows.  No
        # row is selected by its comparison output.
        conditional_rows: list[dict[str, Any]] = []
        j_values = [j_design] + [_positive(v, "J reference control") for v in calc["J_reference_controls_MeV"]]
        seen_j = set()
        for y_text in calc["residual_y"]:
            y_label = _number(_positive(y_text, "residual y"), 6)
            model, state, jet = inverse_models[y_label]
            global_status = (
                "LOCAL_RESPONSE_RETAINED_GLOBAL_POTENTIAL_NONNEGATIVE"
                if jet["globally_nonnegative_potential"]
                else "REJECTED_GLOBAL_POTENTIAL_UNSTABLE_LOCAL_CANDIDATE_RETAINED"
            )
            for J in j_values:
                J_text = _number(J, 6)
                if (y_label, J_text) in seen_j:
                    continue
                seen_j.add((y_label, J_text))
                c_rho = constant_rho_coupling(model, state, J)
                role = (
                    "NOMINAL_J32_DESIGN_INPUT"
                    if J == j_design
                    else "CONDITIONAL_REFERENCE_CONTROL_NOT_FIT"
                )
                row = _candidate_row(
                    label=f"inverse_y_{y_label}_conditional_J_{J_text}",
                    model=model,
                    state=state,
                    c_rho=c_rho,
                    J_role=role,
                    global_status=global_status,
                    direct_checks=False,
                )
                row["chosen_y"] = y_label
                conditional_rows.append(row)

        sensitivity_rows: list[dict[str, Any]] = []
        sensitivity_y = _number(_positive("0.90", "sensitivity y"), 6)
        sensitivity_model, sensitivity_state, sensitivity_jet = inverse_models[sensitivity_y]
        for K in calc["sensitivity_K_MeV"]:
            K = _positive(K, "sensitivity K")
            model, state, jet = inverse_potential_jet(
                sensitivity_y,
                K_target=_number(K),
                n_fm3=_number(n0_fm3),
                binding=_number(binding),
            )
            c_rho = constant_rho_coupling(model, state, j_design)
            row = _candidate_row(
                label=f"design_sensitivity_y_{sensitivity_y}_K_{_number(K, 6)}",
                model=model,
                state=state,
                c_rho=c_rho,
                J_role="K_SENSITIVITY_NOT_PROBABILISTIC_UNCERTAINTY",
                global_status=(
                    "LOCAL_RESPONSE_RETAINED_GLOBAL_POTENTIAL_NONNEGATIVE"
                    if jet["globally_nonnegative_potential"]
                    else "REJECTED_GLOBAL_POTENTIAL_UNSTABLE_LOCAL_CANDIDATE_RETAINED"
                ),
                direct_checks=False,
            )
            row["K_design_sensitivity_MeV"] = _number(K)
            sensitivity_rows.append(row)

        # Literature references consume only the finished rows and cannot feed
        # back into model construction.  Use nominal plus explicit conditional
        # rows for the descriptive comparison.
        comparison_rows = inverse_rows + conditional_rows
        comparisons = _benchmark_comparisons(payload["descriptive_benchmarks"], comparison_rows)
        # Compact live arithmetic guards retained in the JSON result.
        for row in inverse_rows:
            if row["local_jet_status"] != "COMPUTED_LOCAL_STABLE":
                raise IsospinJetError(f"local inverse row is not stable: {row['label']}")
            if row["jet"]["J"] is None or row["jet"]["L"] is None:
                raise IsospinJetError(f"nonfinite local jet row: {row['label']}")

        return {
            "status": "COMPUTED_LOCAL_ISOSPIN_JET_NO_EMPIRICAL_PASS",
            "schema": SCHEMA,
            "evidence_weight": EVIDENCE_WEIGHT,
            "calculation_inputs": _calculation_inputs(payload),
            "predictions": {
                "controls": [original_row, calibrated_row],
                "inverse_design_grid": inverse_rows,
                "conditional_J_controls": conditional_rows,
                "design_sensitivity": sensitivity_rows,
            },
            "descriptive_comparisons": comparisons,
            "scope": {
                "rho_extension": "separately_declared_constant_mass_constant_coupling_effective_extension",
                "baseline_source_complete_action_changed": False,
                "calibration_roles": "n0,binding,Ktarget,J are declared design/calibration inputs; L and S4 held out",
                "local_domain": "n=n0, delta expansion around 0, positive locally stable scalar branch",
                "global_status_rule": "unstable inverse polynomial rows remain visible and rejected globally",
                "uncertainty_rule": "precision and K sensitivity are not physical uncertainty; no mass-scheme uncertainty invented",
                "empirical_rule": "literature rows are descriptive conditional references only; no likelihood or empirical PASS",
            },
        }


def calculate(dps: int = 80, input_path: Path | str = INPUT_PATH) -> dict[str, Any]:
    """Run one live calculation at the requested working precision."""

    path = Path(input_path).resolve()
    return _run_calculation(int(dps), path)


def _compare_nested(left: Any, right: Any, *, tolerance: mp.mpf = mp.mpf("1e-12"), path: str = "root") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise IsospinJetError(f"precision output keys differ at {path}")
        for key in left:
            _compare_nested(left[key], right[key], tolerance=tolerance, path=f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise IsospinJetError(f"precision output lengths differ at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            _compare_nested(a, b, tolerance=tolerance, path=f"{path}[{index}]")
        return
    if left is None or right is None or isinstance(left, bool) or isinstance(right, bool):
        if left != right:
            raise IsospinJetError(f"precision output differs at {path}")
        return
    try:
        if _relative_error(_mp(left, path), _mp(right, path)) > tolerance:
            raise IsospinJetError(f"precision output differs at {path}")
    except IsospinJetError:
        if left != right:
            raise


def build_result(input_path: Path | str = INPUT_PATH) -> dict[str, Any]:
    """Run the full designated matrix and compare independent precisions."""

    path = Path(input_path).resolve()
    payload = load_declared_inputs(path)
    calc = _protocol_calculation_inputs(payload["calculation_inputs"])
    precisions = [int(v) for v in calc["working_precisions"]]
    low = calculate(precisions[0], path)
    high = calculate(precisions[1], path)
    # Validate each precision state independently before comparing or assigning
    # a computational-success status.  A common-mode bad residual, a dropped
    # fine-precision row, or a pre-existing ``pass`` flag must not disappear in
    # the aggregate comparison.
    _validate_calculation_result(low, payload, precision=precisions[0])
    _validate_calculation_result(high, payload, precision=precisions[1])
    _compare_nested(low["predictions"], high["predictions"], tolerance=mp.mpf("1e-12"))
    _compare_nested(
        low["descriptive_comparisons"],
        high["descriptive_comparisons"],
        tolerance=mp.mpf("1e-12"),
    )
    low["precision_check"] = {
        "working_precisions": precisions,
        "comparison": "all prediction fields agree to relative 1e-12",
        "mandatory_checks": {
            str(precisions[0]): "PASS",
            str(precisions[1]): "PASS",
        },
        "pass": True,
    }
    low["provenance"] = {
        "source_path": "verification/nvg_isospin_jet_audit.py",
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "input_path": "verification/data/nvg_isospin_jet_benchmarks.json",
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reused_producers": [
            "verification/source_complete_scaling_saturation_audit.py:BulkModel",
            "verification/source_complete_scaling_saturation_audit.py:quartic_calibration",
            "verification/source_complete_scaling_saturation_audit.py:inverse_potential_jet",
        ],
        "reused_producer_sha256": hashlib.sha256(
            (HERE / "source_complete_scaling_saturation_audit.py").read_bytes()
        ).hexdigest(),
        "original_action_modified": False,
        "answer_table_used": False,
    }
    return low


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    args = parser.parse_args(argv)
    try:
        result = build_result(args.input)
    except (IsospinJetError, ValueError, ArithmeticError) as exc:
        print(f"isospin-jet audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
