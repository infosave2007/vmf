#!/usr/bin/env python3
"""Ca48 held-out probe, volume/surface operator split and quantum-surface first look.

Continuation of ``NVG_RESIDUAL_RESEARCH_DIRECTIONS_RU.md`` after the isovector
form-factor probe (directions I and III, 2026-09-16).  Three declared parts on
the accepted finite-static baseline (W8.93, ``no_rho``, s*=0.2268, Ca40 anchor):

* **Part A - pre-registered held-out nucleus.**  48Ca (named by the research
  program as a discriminating input) is solved through the *full* terminal
  protocol at the frozen root scale with no retuning: the ``no_rho`` branch
  prediction plus both declared contact probes (J=20 and the live J=32).  The
  discriminating question: does the same bulk-isovector coupling j~10-12 MeV
  that annihilates the Zr90/Pb208 overbinding also annihilate the 48Ca
  residual, or does 48Ca falsify the single-mechanism reading?
* **Part B - volume vs surface operator sensitivity.**  On frozen accepted
  profiles: the bulk isovector integral ``I_A`` (with half-density split), the
  gradient isovector operator ``(1/n0)∫|∇(n_n-n_p)|^2 d^3r`` (full and
  WKB-validity masked), and the anchor-preserving slopes of each operator.
  The output is the A-pattern of the required coupling per operator: where
  bulk and surface operators make maximally different predictions.
* **Part C - quantum surface, fixed physical coefficients.**  The 2nd-order
  extended-Thomas-Fermi gradient kinetic correction ``ΔE = (ħc)^2/(36 m*)
  ∫ (∇n)^2/n d^3r`` (and the von-Weizsacker variant ``(ħc)^2/(8 m*)``) on
  frozen profiles with the local effective mass ``m*(r)=M y(r)``.  TF profiles
  have compact support, where the gradient expansion is singular; every
  integral is therefore reported both full and masked to the WKB-validity
  region ``|∇n| <= eps_threshold * n * k_F(r)``.  Anchored (Ca40-preserving)
  first-order predictions follow from the same envelope chain rule as the
  contact probe, including the anchor-retuning fraction ``ds/s``.

New experimental inputs are cited only (AME2020/ENSDF via IAEA LiveChart for
the 48Ca binding, Angeli-Marinova 2013 for the charge radius, CREX PRL 129,
042501 (2022) for the neutron skin).  No coefficient is tuned to any residual,
no new interaction is declared, nothing is an empirical confirmation, and the
result carries ``evidence_weight=0``.  Importing this module performs no
numerical work and no network access; the CLI prints strict JSON to stdout and
``--output`` is an explicit opt-in write.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.integrate import simpson

try:  # Both direct-script and package-style imports are supported.
    import nvg_finite_monopole as static
    import nvg_finite_static_bridge as bridge
    import nvg_isovector_formfactor_probe as ivp
except ImportError:  # pragma: no cover - package import support.
    from . import nvg_finite_monopole as static
    from . import nvg_finite_static_bridge as bridge
    from . import nvg_isovector_formfactor_probe as ivp

HERE = Path(__file__).resolve().parent
SCHEMA = "nvg_ca48_quantum_surface_probe.v1"
STATUS = "COMPUTED_CA48_HELDOUT_AND_QUANTUM_SURFACE_PROBE_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

CA48 = "Ca48"
CA48_NUCLEUS_ROW = {"A": 48, "Z": 20, "N": 28}

# Cited 48Ca inputs (declared evaluated data only; never model outputs).
# B_atom/A from AME2020/ENSDF via the IAEA LiveChart ground-state service
# (extraction 2026-09-16): binding 8666.6916 +- 0.0004 keV per nucleon,
# atomic mass 47.952522654 u.  The KAERI mirror of the older Audi-Wapstra
# evaluation (415991.169 keV total) is superseded and NOT used.
CA48_B_ATOM_PER_A_MEV = 8.6666916
CA48_B_ATOM_PER_A_SIGMA_MEV = 0.0000004
CA48_B_ATOM_SOURCE = "AME2020/ENSDF via IAEA LiveChart (nds.iaea.org/relnsd/v1/data, extraction 2026-09-16); atomic mass 47.952522654 u"
# Charge radius: Angeli-Marinova 2013 published value (IAEA NDS radii table).
CA48_R_CH_FM = 3.4771
CA48_R_CH_SIGMA_FM = 0.0020
CA48_R_CH_SOURCE = "Angeli & Marinova, ADNDT 99 (2013) 69 (https://nds.iaea.org/radii/, published 2013 values)"
# Neutron skin: CREX parity-violating measurement (model-extracted).
CA48_SKIN_FM = 0.121
CA48_SKIN_EXP_SIGMA_FM = 0.026
CA48_SKIN_MODEL_SIGMA_FM = 0.024
CA48_SKIN_SOURCE = "CREX Collaboration (Adhikari et al.), PRL 129, 042501 (2022), arXiv:2205.11593: Rn-Rp = 0.121 +- 0.026(exp) +- 0.024(model) fm"
CA48_ROLE = "pre_registered_heldout_descriptive_comparison_only"

# Fixed physical gradient-correction coefficients (no tuning): the systematic
# 2nd-order ETF term hbar^2/(36 m*) and the von-Weizsacker single-orbital form
# hbar^2/(8 m*).  The local effective mass is m*(r) = M * y(r) with the design
# nucleon mass M and the live scalar profile y (vacuum y = 1 exactly).
ETF2_COEFF = 1.0 / 36.0
VW_COEFF = 1.0 / 8.0
VW_OVER_ETF2 = VW_COEFF / ETF2_COEFF  # exactly 4.5
# WKB-validity thresholds for the gradient expansion: |grad n| <= eps * n * k_F
# with the per-species local Fermi momentum k_F = (3 pi^2 n)^(1/3) (d = 2).
EPS_VALIDITY_THRESHOLDS = (0.5, 1.0)
Y_VACUUM_TOL = 1.0e-6

PREVIOUS_SUMMARY_REL = Path("Lunacy/runs/isovector-formfactor-2026-09-16/probe_summary.json")

# Ca40 anchor and baseline conventions are imported from the accepted isovector
# probe (single source of truth): FAMILY, ROOT_SCALE, BASELINE_EXPECTED,
# REPRO_TOL, contact j values and the terminal/metrics machinery.
FAMILY = ivp.FAMILY
ROOT_SCALE = ivp.ROOT_SCALE
ROOT_SCALE_SOURCE = ivp.ROOT_SCALE_SOURCE
BASELINE_EXPECTED = ivp.BASELINE_EXPECTED
REPRO_TOL = ivp.REPRO_TOL
N0_FM3 = ivp.N0_FM3
J20_PROBE_MEV = ivp.J20_PROBE_MEV
J32_DESIGN_MEV = ivp.J32_DESIGN_MEV
BASELINE_RHO = ivp.BASELINE_RHO
CONTACT_RHO = ivp.CONTACT_RHO

CA48_CASES = (
    ("no_rho", BASELINE_RHO, None),
    ("J20", CONTACT_RHO, J20_PROBE_MEV),
    ("J32", CONTACT_RHO, J32_DESIGN_MEV),
)


class Ca48ProbeError(ValueError):
    """Fail-closed error for protocol failures or non-reproducible baselines."""


def _number(value: Any, digits: int = 17) -> float:
    return ivp._number(value, digits)


def _jsonable(value: Any) -> Any:
    return ivp._jsonable(value)


# ---------------------------------------------------------------------------
# Ca48 nucleus-table patch (always restored; never mutates production files).
# ---------------------------------------------------------------------------


def _ca48_nuclei_context():
    """Temporarily add the 48Ca row to the live bridge nucleus table.

    The bridge reads ``static.BRIDGE_NUCLEI`` at call time (seed generation,
    A/Z/N bookkeeping, radius mapping), so a context patch is the narrow seam
    for a pre-registered held-out nucleus.  Double-entry is refused, and the
    row is always removed on exit, including on exceptions.
    """

    class _Patch:
        def __enter__(self):
            if CA48 in static.BRIDGE_NUCLEI:
                raise Ca48ProbeError("Ca48 is already present in the bridge nucleus table")
            static.BRIDGE_NUCLEI[CA48] = dict(CA48_NUCLEUS_ROW)
            return self

        def __exit__(self, *exc):
            static.BRIDGE_NUCLEI.pop(CA48, None)
            return False

    return _Patch()


def ca48_binding_target() -> dict[str, Any]:
    """Declared 48Ca binding target with the production electron convention."""

    row = {
        "A": CA48_NUCLEUS_ROW["A"],
        "Z": CA48_NUCLEUS_ROW["Z"],
        "B_atom_per_A_MeV": CA48_B_ATOM_PER_A_MEV,
    }
    correction = bridge.electron_correction_MeV(int(row["Z"]))
    return {
        "A": int(row["A"]),
        "Z": int(row["Z"]),
        "N": int(CA48_NUCLEUS_ROW["N"]),
        "B_atom_per_A_MeV": _number(CA48_B_ATOM_PER_A_MEV),
        "B_atom_per_A_sigma_MeV": _number(CA48_B_ATOM_PER_A_SIGMA_MEV),
        "B_atom_source": CA48_B_ATOM_SOURCE,
        "role": CA48_ROLE,
        "electron_correction_total_MeV": _number(correction),
        "electron_correction_per_A_MeV": _number(correction / float(row["A"])),
        "B_nuc_target_per_A_MeV": _number(bridge.bare_binding_per_A_MeV(row)),
    }


def ca48_patched_inputs() -> dict[str, Any]:
    """Deep copy of the cited input bundle with 48Ca rows added for mappings."""

    inputs = bridge._load_inputs()
    inputs["binding"]["values"][CA48] = {
        "A": CA48_NUCLEUS_ROW["A"],
        "Z": CA48_NUCLEUS_ROW["Z"],
        "N": CA48_NUCLEUS_ROW["N"],
        "B_atom_per_A_MeV": CA48_B_ATOM_PER_A_MEV,
        "B_atom_per_A_sigma_MeV": CA48_B_ATOM_PER_A_SIGMA_MEV,
        "role": CA48_ROLE,
    }
    inputs["charge_radii"]["values"][CA48] = {
        "R_ch_fm": CA48_R_CH_FM,
        "R_ch_sigma_fm": CA48_R_CH_SIGMA_FM,
        "role": CA48_ROLE,
    }
    return inputs


# ---------------------------------------------------------------------------
# Profiles with the live scalar field (local effective mass).
# ---------------------------------------------------------------------------


def _profiles_with_y(solved: Any) -> dict[str, Any]:
    """Physical-radius species densities plus y(r) and m*(r) = M y(r)."""

    profile = ivp._profiles(solved)
    row = solved.row
    try:
        xx = np.asarray(row["profile_grid_x"], dtype=float)
        y = np.asarray(row["profile_y"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise Ca48ProbeError("solution row lacks a scalar-field profile") from exc
    if y.shape != xx.shape or y.ndim != 1 or not np.all(np.isfinite(y)):
        raise Ca48ProbeError("scalar-field profile is malformed")
    if float(y[-1]) <= 0.0:
        raise Ca48ProbeError("vacuum scalar field is not positive")
    if abs(float(y[-1]) - 1.0) > Y_VACUUM_TOL:
        raise Ca48ProbeError(f"vacuum scalar field deviates from 1: {float(y[-1])!r}")
    design = solved.design
    mass = float(design.M)
    if not math.isfinite(mass) or mass <= 0.0:
        raise Ca48ProbeError("design nucleon mass is invalid")
    profile["y"] = y
    profile["M_MeV"] = mass
    profile["m_star_MeV"] = mass * y
    return profile


def local_fermi_momentum(n_fm3: np.ndarray) -> np.ndarray:
    """Per-species local k_F = (3 pi^2 n)^(1/3) fm^-1 for d = 2 degeneracy."""

    n = np.asarray(n_fm3, dtype=float)
    return np.cbrt(3.0 * math.pi * math.pi * np.maximum(n, 0.0))


def wkb_validity_parameter(r_fm: np.ndarray, n_fm3: np.ndarray) -> np.ndarray:
    """eps(r) = |grad n| / (n k_F): gradient-expansion validity parameter.

    eps << 1 means the density varies slowly on the local de Broglie scale;
    the TF compact-support edge has eps -> infinity, where the gradient
    expansion is invalid by construction.
    """

    r = np.asarray(r_fm, dtype=float)
    n = np.asarray(n_fm3, dtype=float)
    dn = np.gradient(n, r, edge_order=2)
    positive = n > 0.0
    k_f = local_fermi_momentum(n)
    safe_n = np.where(positive, n, 1.0)
    safe_k = np.where(positive, k_f, 1.0)
    return np.where(positive, np.abs(dn) / (safe_n * safe_k), 0.0)


def _masked_spherical_integral(r: np.ndarray, integrand: np.ndarray, mask: np.ndarray) -> float:
    """Spherical integral of the integrand multiplied by a boolean mask."""

    r = np.asarray(r, dtype=float)
    f = np.asarray(integrand, dtype=float) * np.asarray(mask, dtype=bool)
    return 4.0 * math.pi * float(simpson(f * r**2, x=r))


# ---------------------------------------------------------------------------
# Part C: fixed-coefficient quantum-surface gradient corrections.
# ---------------------------------------------------------------------------


def quantum_surface_energy(
    r_fm: np.ndarray,
    n_fm3: np.ndarray,
    y: np.ndarray,
    mass_MeV: float,
    hbarc_MeV_fm: float,
    coeff_over_m: float,
    eps_thresholds: Sequence[float] = EPS_VALIDITY_THRESHOLDS,
) -> dict[str, Any]:
    """Gradient kinetic correction on one species profile (frozen, first order).

    ΔE = (ħc)^2 * coeff_over_m * ∫ (∇n)^2 / (n m*(r)) d^3r with the fixed
    physical coefficient (ETF-2 1/36 or von-Weizsacker 1/8) and the local
    effective mass m*(r) = M y(r).  Reported full and masked to the WKB
    validity region; the TF edge makes the full integral an upper-bound-flavored
    artifact, and the masked variant is the defensible first-order estimate.
    """

    r = np.asarray(r_fm, dtype=float)
    n = np.asarray(n_fm3, dtype=float)
    y = np.asarray(y, dtype=float)
    dn = np.gradient(n, r, edge_order=2)
    positive = n > 0.0
    safe_n = np.where(positive, n, 1.0)
    base = np.where(positive, dn**2 / safe_n, 0.0)  # fm^-2 integrand
    m_star = float(mass_MeV) * np.maximum(y, 1.0e-12)
    prefactor = (float(hbarc_MeV_fm) ** 2) * float(coeff_over_m)  # MeV^2 fm^2 / m*
    eps = wkb_validity_parameter(r, n)
    result: dict[str, Any] = {
        "coefficient_over_m": _number(coeff_over_m),
        "integral_full_fm_minus2": _number(ivp._spherical_integral(r, base)),
        "energy_full_MeV": _number(ivp._spherical_integral(r, prefactor * base / m_star)),
    }
    masked = {}
    for threshold in eps_thresholds:
        mask = positive & (eps <= float(threshold))
        cut_radii = r[mask]
        masked[str(_number(threshold, 10))] = {
            "integral_fm_minus2": _number(_masked_spherical_integral(r, base, mask)),
            "energy_MeV": _number(_masked_spherical_integral(r, prefactor * base / m_star, mask)),
            "outer_valid_radius_fm": None if cut_radii.size == 0 else _number(float(cut_radii[-1]), 10),
            "valid_grid_fraction": _number(float(np.count_nonzero(mask)) / float(r.size), 10),
        }
    result["masked_by_eps"] = masked
    return result


def nucleus_quantum_surface(
    profile: Mapping[str, np.ndarray],
    coeff_over_m: float = ETF2_COEFF,
    eps_thresholds: Sequence[float] = EPS_VALIDITY_THRESHOLDS,
) -> dict[str, Any]:
    """Per-species and total gradient correction for one accepted profile."""

    r = np.asarray(profile["r_fm"], dtype=float)
    y = np.asarray(profile["y"], dtype=float)
    mass = float(profile["M_MeV"])
    hbarc = float(profile["hbarc_MeV_fm"])
    species = {}
    total_full = 0.0
    totals_masked = {str(_number(t, 10)): 0.0 for t in eps_thresholds}
    for label, key in (("neutron", "nn_fm3"), ("proton", "np_fm3")):
        block = quantum_surface_energy(r, profile[key], y, mass, hbarc, coeff_over_m, eps_thresholds)
        species[label] = block
        total_full += float(block["energy_full_MeV"])
        for threshold in eps_thresholds:
            totals_masked[str(_number(threshold, 10))] += float(block["masked_by_eps"][str(_number(threshold, 10))]["energy_MeV"])
    return {
        "m_star_center_MeV": _number(float(mass * float(y[0])), 10),
        "m_star_vacuum_MeV": _number(float(mass * float(y[-1])), 10),
        "energy_total_full_MeV": _number(total_full),
        "energy_total_masked_MeV": {key: _number(value) for key, value in totals_masked.items()},
        "species": species,
    }


# ---------------------------------------------------------------------------
# Part B: volume vs surface isovector operators.
# ---------------------------------------------------------------------------


def gradient_isovector_integrals(
    r_fm: np.ndarray,
    nn_fm3: np.ndarray,
    np_fm3: np.ndarray,
    n0_fm3: float = N0_FM3,
    eps_thresholds: Sequence[float] = EPS_VALIDITY_THRESHOLDS,
) -> dict[str, Any]:
    """Gradient isovector operator (1/n0)∫|∇(n_n-n_p)|^2 d^3r in fm^-2.

    Reported full, split by the half-density radius of the total density, and
    masked to the region where BOTH species satisfy the WKB validity bound.
    A surface-coupled isovector operator j_s * I_grad carries j_s in MeV fm^2.
    """

    r = np.asarray(r_fm, dtype=float)
    nn = np.asarray(nn_fm3, dtype=float)
    np_ = np.asarray(np_fm3, dtype=float)
    n3 = nn - np_
    dn3 = np.gradient(n3, r, edge_order=2)
    integrand = dn3**2
    i_full = ivp._spherical_integral(r, integrand) / float(n0_fm3)
    r_half = ivp._half_density_radius(r, nn + np_)
    inside = r <= r_half
    i_bulk = _masked_spherical_integral(r, integrand / float(n0_fm3), inside)
    eps_n = wkb_validity_parameter(r, nn)
    eps_p = wkb_validity_parameter(r, np_)
    masked = {}
    for threshold in eps_thresholds:
        mask = (eps_n <= float(threshold)) & (eps_p <= float(threshold))
        masked[str(_number(threshold, 10))] = {
            "I_grad_fm_minus2": _number(_masked_spherical_integral(r, integrand / float(n0_fm3), mask)),
        }
    return {
        "I_grad_full_fm_minus2": _number(i_full),
        "I_grad_bulk_inside_half_fm_minus2": _number(i_bulk),
        "I_grad_surface_outside_half_fm_minus2": _number(i_full - i_bulk),
        "half_density_radius_fm": _number(r_half, 10),
        "masked_by_eps": masked,
        "units_note": "I_grad has units fm^-2; a coupling j_s in MeV fm^2 makes E = j_s * I_grad",
    }


def anchored_operator_slope(
    op_a: float, t_a: float, op_ref: float, t_ref: float, a: int
) -> float:
    """Anchor-preserving slope d(B_A/A)/d(coupling) for a frozen operator."""

    return ivp.anchored_binding_slope(op_a, t_a, op_ref, t_ref, a)


# ---------------------------------------------------------------------------
# Analytic validation (no BVP; fast import-safe controls).
# ---------------------------------------------------------------------------


def analytic_validation() -> dict[str, Any]:
    """Gaussian controls for the gradient and quantum-surface machinery."""

    r = np.linspace(0.0, 14.0, 5601)
    sigma = 1.9
    shape = np.exp(-(r**2) / (2.0 * sigma**2))
    n_gauss = shape  # unnormalized Gaussian density
    count = ivp._spherical_integral(r, n_gauss)
    y_flat = np.ones_like(r)
    mass = 939.0
    hbarc = 197.3269804
    # (∇n)^2/n integrates to 3 N / sigma^2 for a normalized Gaussian.
    dn = np.gradient(n_gauss, r, edge_order=2)
    positive = n_gauss > 0.0
    base = np.where(positive, dn**2 / np.where(positive, n_gauss, 1.0), 0.0)
    integral_full = ivp._spherical_integral(r, base)
    analytic_integral = 3.0 * count / sigma**2
    # ETF-2 energy with a constant effective mass.
    block = quantum_surface_energy(r, n_gauss, y_flat, mass, hbarc, ETF2_COEFF)
    analytic_energy = (hbarc**2 * ETF2_COEFF / mass) * analytic_integral
    # Isovector gradient control: n3 = a * exp(-r^2/(2 sigma^2)) gives
    # ∫|∇n3|^2 d^3r = (1/sigma^4) ∫ r^2 n3^2 d^3r with ∫ r^2 e^{-r^2/sigma^2} d^3r
    # = 4 pi * (3/8) sqrt(pi) sigma^5, hence I_grad = 3 pi^(3/2) a^2 sigma / (2 n0).
    amplitude = 0.03
    n3 = amplitude * shape
    dn3 = np.gradient(n3, r, edge_order=2)
    i_grad = ivp._spherical_integral(r, dn3**2) / N0_FM3
    analytic_i_grad = 3.0 * math.pi**1.5 * amplitude**2 * sigma / (2.0 * N0_FM3)
    # Masked variants must not exceed the full integrals, and the mask must
    # remove the outer tail where eps grows without bound.
    eps = wkb_validity_parameter(r, n_gauss)
    return {
        "gaussian_count": _number(count, 10),
        "grad_squared_over_n_integral": _number(integral_full, 10),
        "grad_squared_over_n_analytic": _number(analytic_integral, 10),
        "grad_squared_over_n_relative_error": _number(abs(integral_full - analytic_integral) / analytic_integral, 10),
        "etf2_energy_full_MeV": block["energy_full_MeV"],
        "etf2_energy_analytic_MeV": _number(analytic_energy, 10),
        "etf2_energy_relative_error": _number(abs(float(block["energy_full_MeV"]) - analytic_energy) / analytic_energy, 10),
        "i_grad_full_fm_minus2": _number(i_grad, 10),
        "i_grad_analytic_fm_minus2": _number(analytic_i_grad, 10),
        "i_grad_relative_error": _number(abs(i_grad - analytic_i_grad) / analytic_i_grad, 10),
        "eps_max": _number(float(np.max(eps)), 10),
        "eps_median_positive_region": _number(float(np.median(eps[positive])), 10),
        "masked_leq_full": bool(
            all(
                float(block["masked_by_eps"][str(_number(t, 10))]["energy_MeV"]) <= float(block["energy_full_MeV"]) + 1.0e-12
                for t in EPS_VALIDITY_THRESHOLDS
            )
        ),
    }


# ---------------------------------------------------------------------------
# Decimation stability control for the gradient quantities.
# ---------------------------------------------------------------------------


def decimation_control(profile: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Key gradient integrals on the full ladder and its 2x/4x decimations."""

    r = np.asarray(profile["r_fm"], dtype=float)
    y = np.asarray(profile["y"], dtype=float)
    nn = np.asarray(profile["nn_fm3"], dtype=float)
    np_ = np.asarray(profile["np_fm3"], dtype=float)
    mass = float(profile["M_MeV"])
    hbarc = float(profile["hbarc_MeV_fm"])
    rows = {}
    for stride, label in ((1, "full"), (2, "half"), (4, "quarter")):
        rr = r[::stride]
        yy = y[::stride]
        nnn = nn[::stride]
        ppp = np_[::stride]
        grad = gradient_isovector_integrals(rr, nnn, ppp)
        qs = nucleus_quantum_surface(
            {"r_fm": rr, "y": yy, "nn_fm3": nnn, "np_fm3": ppp, "M_MeV": mass, "hbarc_MeV_fm": hbarc}
        )
        rows[label] = {
            "points": int(rr.size),
            "I_grad_full_fm_minus2": grad["I_grad_full_fm_minus2"],
            "etf2_energy_total_full_MeV": qs["energy_total_full_MeV"],
            "etf2_energy_total_masked_eps1_MeV": qs["energy_total_masked_MeV"]["1.0"],
        }
    full = rows["full"]
    worst_grad = max(
        abs(float(rows[label]["I_grad_full_fm_minus2"]) - float(full["I_grad_full_fm_minus2"])) / max(abs(float(full["I_grad_full_fm_minus2"])), 1.0e-30)
        for label in ("half", "quarter")
    )
    worst_energy = max(
        abs(float(rows[label]["etf2_energy_total_full_MeV"]) - float(full["etf2_energy_total_full_MeV"])) / max(abs(float(full["etf2_energy_total_full_MeV"])), 1.0e-30)
        for label in ("half", "quarter")
    )
    return {
        "rows": rows,
        "max_relative_change_I_grad": _number(worst_grad, 10),
        "max_relative_change_etf2_energy": _number(worst_energy, 10),
        "note": "decimation of the fixed diagnostic ladder; stability of the masked integrals is limited by the sharp eps cut (declared)",
    }


# ---------------------------------------------------------------------------
# Main calculation.
# ---------------------------------------------------------------------------


def _baseline_block() -> dict[str, Any]:
    """Reproduce the three accepted no-rho baselines; fail closed."""

    baselines: dict[str, Any] = {}
    for nucleus in ivp.NUCLEI:
        case = ivp._terminal_case(nucleus, BASELINE_RHO)
        metrics = ivp._terminal_metrics(nucleus, case)
        expected = BASELINE_EXPECTED[nucleus]
        for key, value in expected.items():
            actual = float(metrics[key])
            if not math.isfinite(actual) or abs(actual - float(value)) > REPRO_TOL:
                raise Ca48ProbeError(
                    f"baseline reproduction failed for {nucleus}.{key}: {actual!r} vs {value!r}"
                )
        profile = _profiles_with_y(case["_domain_solution"])
        baselines[nucleus] = {
            "metrics": metrics,
            "profile": profile,
            "_case": case,
            "reproduced": True,
        }
    return baselines


def _ca48_case_block() -> dict[str, Any]:
    """All 48Ca terminal solves under the nucleus-table patch."""

    target = ca48_binding_target()
    inputs = ca48_patched_inputs()
    cases: dict[str, Any] = {}
    with _ca48_nuclei_context():
        for label, rho, j_design in CA48_CASES:
            entry: dict[str, Any] = {"rho_choice": rho, "j_design_MeV": j_design}
            with ivp._rho_context(rho), ivp._j_design_context(j_design):
                case = bridge._terminal_case(FAMILY, ROOT_SCALE, CA48, multi_seed=True)
            if case.get("terminal_protocol_acceptance") is not True:
                domain = case.get("domain40_check", {})
                entry.update({
                    "status": "CA48_TERMINAL_REJECTED",
                    "terminal_classification": case.get("terminal_classification"),
                    "rejection_evidence": {
                        "domain40_converged": bool(domain.get("converged")),
                        "domain40_stationarity_acceptance": domain.get("stationarity_acceptance"),
                        "domain40_localized_vacuum_exterior": domain.get("localized_vacuum_exterior"),
                        "domain40_conserved_NZ_relative_error": domain.get("conserved_NZ_relative_error"),
                        "fresh24_converged": bool(case.get("fresh24_check", {}).get("converged")),
                        "fresh24_stationarity_acceptance": case.get("fresh24_check", {}).get("stationarity_acceptance"),
                    },
                    "note": "the fail-closed terminal protocol refused to promote this state; no quantity from it is used anywhere",
                })
                cases[label] = entry
                continue
            metrics = ivp._terminal_metrics(CA48, case)
            profile = _profiles_with_y(case["_domain_solution"])
            radius = bridge._radius_mapping(case["domain40_check"], CA48, inputs)
            entry.update({
                "status": "ACCEPTED",
                "A": int(CA48_NUCLEUS_ROW["A"]),
                "N": int(CA48_NUCLEUS_ROW["N"]),
                "Z": int(CA48_NUCLEUS_ROW["Z"]),
                "terminal_classification": metrics["terminal_classification"],
                "binding_per_A_MeV": _number(metrics["binding_per_A_MeV"]),
                "residual_vs_B_nuc_target_per_A_MeV": _number(
                    float(metrics["binding_per_A_MeV"]) - float(target["B_nuc_target_per_A_MeV"])
                ),
                "T_W_MeV": _number(metrics["T_W_MeV"]),
                "E_rho_MeV": _number(metrics["E_rho_MeV"]),
                "total_E_MeV": _number(metrics["total_E_MeV"]),
                "rms_neutron_radius_fm": _number(metrics["rms_neutron_radius_fm"]),
                "rms_point_proton_radius_fm": _number(metrics["rms_point_proton_radius_fm"]),
                "point_neutron_skin_fm": _number(metrics["point_neutron_skin_fm"]),
                "central_density_fm_minus3": _number(metrics["central_density_fm_minus3"]),
                "conserved_NZ_relative_error": _number(metrics["conserved_NZ_relative_error"]),
                "isovector": metrics["isovector"],
                "radius_mapping": radius,
                "_profile": profile,
                "_case": case,
            })
            cases[label] = entry
    return {"target": target, "cases": cases}


def _load_previous_summary() -> dict[str, Any]:
    path = HERE.parent / PREVIOUS_SUMMARY_REL
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "PREVIOUS_SUMMARY_UNAVAILABLE", "path": str(path), "error": str(exc)}
    return {"status": "LOADED", "path": str(path), "data": payload}


def calculate() -> dict[str, Any]:
    """Run the full probe; strict JSON-serializable output."""

    # --- Baseline reproduction (fail closed) --------------------------------
    baselines = _baseline_block()
    baseline_public = {}
    for nucleus, block in baselines.items():
        metrics = block["metrics"]
        baseline_public[nucleus] = {
            "binding_per_A_MeV": _number(metrics["binding_per_A_MeV"]),
            "T_W_MeV": _number(metrics["T_W_MeV"]),
            "rms_neutron_radius_fm": _number(metrics["rms_neutron_radius_fm"]),
            "rms_point_proton_radius_fm": _number(metrics["rms_point_proton_radius_fm"]),
            "point_neutron_skin_fm": _number(metrics["point_neutron_skin_fm"]),
            "isovector": metrics["isovector"],
            "reproduced_vs_R3": True,
        }

    # --- Part A: 48Ca held-out cases ----------------------------------------
    ca48 = _ca48_case_block()
    target = ca48["target"]
    cases = ca48["cases"]
    j20_added = ivp._contact_j_added(J20_PROBE_MEV)
    j32_added = ivp._contact_j_added(J32_DESIGN_MEV)
    anchor_metrics = baselines["Ca40"]["metrics"]
    anchor_profile = baselines["Ca40"]["profile"]
    t40 = float(anchor_metrics["T_W_MeV"])
    i40 = float(anchor_metrics["isovector"]["I_A_dimensionless"])

    part_a: dict[str, Any] = {
        "target": target,
        "contact_probes_j_MeV": {"J20": _number(j20_added), "J32": _number(j32_added)},
        "cases": {label: {k: v for k, v in entry.items() if not k.startswith("_")} for label, entry in cases.items()},
    }

    baseline_case = cases.get("no_rho", {})
    if baseline_case.get("status") == "ACCEPTED":
        metrics48 = cases["no_rho"]
        a48 = int(target["A"])
        t48 = float(metrics48["T_W_MeV"])
        i48 = float(metrics48["isovector"]["I_A_dimensionless"])
        residual48 = float(metrics48["residual_vs_B_nuc_target_per_A_MeV"])
        anchored_slope48 = anchored_operator_slope(i48, t48, i40, t40, a48)
        fixed_slope48 = -i48 / float(a48)
        required: dict[str, Any] = {
            "anchored_tangent_MeV": _number(residual48 / abs(anchored_slope48)) if anchored_slope48 != 0.0 else None,
            "fixed_scale_tangent_MeV": _number(residual48 / abs(fixed_slope48)) if fixed_slope48 != 0.0 else None,
            "anchored_slope_dB_dj_MeV_per_MeV": _number(anchored_slope48),
            "fixed_scale_tangent_slope_MeV_per_MeV": _number(fixed_slope48),
        }
        for probe_label, j_added in (("J20", j20_added), ("J32", j32_added)):
            probe_case = cases.get(probe_label, {})
            if probe_case.get("status") == "ACCEPTED":
                delta = float(metrics48["binding_per_A_MeV"]) - float(probe_case["binding_per_A_MeV"])
                slope = delta / j_added
                required[f"chord_{probe_label}_MeV"] = _number(residual48 / slope) if slope != 0.0 else None
        part_a["required_j"] = required
        # Universality of the required bulk coupling across held-out nuclei:
        # anchored tangents recomputed on this run's own baselines.
        universality: dict[str, Any] = {}
        for nucleus in ("Zr90", "Pb208", CA48):
            if nucleus == CA48:
                if anchored_slope48 == 0.0:
                    continue
                universality[nucleus] = {
                    "residual_per_A_MeV": _number(residual48),
                    "anchored_slope": _number(anchored_slope48),
                    "j_required_tangent_MeV": _number(residual48 / abs(anchored_slope48)),
                }
            else:
                metrics = baselines[nucleus]["metrics"]
                a = int(metrics["A"])
                t = float(metrics["T_W_MeV"])
                i_a = float(metrics["isovector"]["I_A_dimensionless"])
                slope = anchored_operator_slope(i_a, t, i40, t40, a)
                # Residual against the production bare target.
                inputs = bridge._load_inputs()
                prod_target = bridge.bare_binding_per_A_MeV(inputs["binding"]["values"][nucleus])
                residual = float(metrics["binding_per_A_MeV"]) - prod_target
                universality[nucleus] = {
                    "residual_per_A_MeV": _number(residual),
                    "anchored_slope": _number(slope),
                    "j_required_tangent_MeV": _number(residual / abs(slope)) if slope != 0.0 else None,
                }
        tangents = [
            float(row["j_required_tangent_MeV"])
            for row in universality.values()
            if row.get("j_required_tangent_MeV") is not None
        ]
        if len(tangents) >= 2:
            spread = (max(tangents) - min(tangents)) / abs(np.mean(tangents))
            universality["spread_fraction_of_mean"] = _number(float(spread), 10)
        part_a["universality_of_required_j"] = universality
        # Radius and skin comparisons for 48Ca (Ca40 R1b is mapped below for
        # the isotope-difference comparison).
        radius48 = cases["no_rho"].get("radius_mapping", {})
        part_a["radius_comparison"] = {
            "model_R1b_fm": radius48.get("R1b_fm"),
            "evaluated_R_ch_fm": _number(CA48_R_CH_FM),
            "evaluated_R_ch_sigma_fm": _number(CA48_R_CH_SIGMA_FM),
            "delta_required_fm2": radius48.get("delta_required_fm2"),
            "R_ch_source": CA48_R_CH_SOURCE,
            "ca48_minus_ca40": {},
        }
        # Ca40 R1b for the isotope-difference comparison.
        inputs_prod = bridge._load_inputs()
        r40 = bridge._radius_mapping(baselines["Ca40"]["_case"]["domain40_check"], "Ca40", inputs_prod)
        part_a["radius_comparison"]["ca40_R1b_fm"] = r40.get("R1b_fm")
        if radius48.get("R1b_fm") is not None and r40.get("R1b_fm") is not None:
            part_a["radius_comparison"]["ca48_minus_ca40"]["model_R1b_difference_fm"] = _number(
                float(radius48["R1b_fm"]) - float(r40["R1b_fm"])
            )
        part_a["radius_comparison"]["ca48_minus_ca40"]["evaluated_R_ch_difference_fm"] = _number(
            CA48_R_CH_FM - float(inputs_prod["charge_radii"]["values"]["Ca40"]["R_ch_fm"])
        )
        part_a["skin_comparison"] = {
            "model_point_skin_fm": _number(float(metrics48["point_neutron_skin_fm"])),
            "CREX_skin_fm": _number(CA48_SKIN_FM),
            "CREX_exp_sigma_fm": _number(CA48_SKIN_EXP_SIGMA_FM),
            "CREX_model_sigma_fm": _number(CA48_SKIN_MODEL_SIGMA_FM),
            "CREX_source": CA48_SKIN_SOURCE,
            "caveat": "CREX extracts the skin from the weak form factor with declared model dependence; the comparison is descriptive",
        }
        # Naive linear j for a CREX-size skin (declared extrapolation).
        skin0 = float(metrics48["point_neutron_skin_fm"])
        skin_response = {}
        for probe_label, j_added in (("J20", j20_added), ("J32", j32_added)):
            probe_case = cases.get(probe_label, {})
            if probe_case.get("status") == "ACCEPTED":
                d_skin = float(probe_case["point_neutron_skin_fm"]) - skin0
                if d_skin != 0.0:
                    skin_response[probe_label] = _number(d_skin / j_added)
        if skin_response:
            mean_response = float(np.mean(list(skin_response.values())))
            if mean_response != 0.0:
                part_a["skin_comparison"]["j_for_CREX_skin_naive_linear_MeV"] = _number(
                    (CA48_SKIN_FM - skin0) / mean_response
                )
            part_a["skin_comparison"]["skin_response_per_j_fm_per_MeV"] = skin_response

    # --- Part B: volume vs surface isovector operator split ----------------
    profiles = {nucleus: baselines[nucleus]["profile"] for nucleus in baselines}
    if cases.get("no_rho", {}).get("status") == "ACCEPTED":
        profiles[CA48] = cases["no_rho"]["_profile"]
    part_b: dict[str, Any] = {"per_nucleus": {}}
    operator_values: dict[str, dict[str, float]] = {}
    for nucleus, profile in profiles.items():
        metrics = (
            cases["no_rho"] if nucleus == CA48 else baselines[nucleus]["metrics"]
        )
        grad = gradient_isovector_integrals(
            profile["r_fm"], profile["nn_fm3"], profile["np_fm3"]
        )
        iso = metrics["isovector"]
        a = int(metrics["A"])
        t = float(metrics["T_W_MeV"])
        row = {
            "A": a,
            "T_W_MeV": _number(t),
            "I_A_bulk_contact": _number(float(iso["I_A_dimensionless"])),
            "I_A_surface_fraction_half_density": _number(float(iso["surface_fraction"])),
            "I_grad": grad,
        }
        part_b["per_nucleus"][nucleus] = row
        operator_values[nucleus] = {
            "bulk": float(iso["I_A_dimensionless"]),
            "surface_geometric": float(iso["I_surface_dimensionless"]),
            "grad_full": float(grad["I_grad_full_fm_minus2"]),
            "grad_masked_eps1": float(grad["masked_by_eps"]["1.0"]["I_grad_fm_minus2"]),
            "T_W": t,
            "A": a,
        }
    # Anchored slopes and required couplings per operator.
    if "Ca40" in operator_values:
        ref = operator_values["Ca40"]
        slopes = {}
        for operator in ("bulk", "surface_geometric", "grad_full", "grad_masked_eps1"):
            operator_slopes = {}
            for nucleus, values in operator_values.items():
                if nucleus == "Ca40":
                    continue
                slope = anchored_operator_slope(
                    values[operator], values["T_W"], ref[operator], ref["T_W"], values["A"]
                )
                operator_slopes[nucleus] = _number(slope)
            slopes[operator] = operator_slopes
        part_b["anchored_slopes_per_unit_coupling"] = slopes
        # Required couplings to annihilate each binding residual.
        required_patterns: dict[str, Any] = {}
        residuals = {}
        inputs_prod = bridge._load_inputs()
        for nucleus, values in operator_values.items():
            if nucleus == "Ca40":
                continue
            if nucleus == CA48:
                residuals[nucleus] = float(cases["no_rho"]["residual_vs_B_nuc_target_per_A_MeV"])
            else:
                prod_target = bridge.bare_binding_per_A_MeV(inputs_prod["binding"]["values"][nucleus])
                residuals[nucleus] = float(baselines[nucleus]["metrics"]["binding_per_A_MeV"]) - prod_target
        for operator in ("bulk", "surface_geometric", "grad_full", "grad_masked_eps1"):
            pattern = {}
            for nucleus, residual in residuals.items():
                slope = slopes[operator][nucleus]
                pattern[nucleus] = _number(residual / abs(float(slope))) if float(slope) != 0.0 else None
            required_patterns[operator] = {
                "values": pattern,
                "units": "MeV fm^2" if operator.startswith("grad_") else "MeV",
            }
        part_b["required_coupling_to_annihilate_residual"] = required_patterns
        # Normalized pattern comparison (where do the operators diverge?).
        if "Zr90" in residuals:
            normalized = {}
            for operator in ("bulk", "surface_geometric", "grad_full", "grad_masked_eps1"):
                pattern = required_patterns[operator]["values"]
                zr_value = pattern.get("Zr90")
                if zr_value in (None, 0.0):
                    continue
                normalized[operator] = {
                    nucleus: _number(float(value) / float(zr_value))
                    for nucleus, value in pattern.items()
                    if value is not None
                }
            part_b["normalized_required_coupling_over_Zr90"] = normalized
            # Largest relative divergence between the bulk and masked-gradient
            # operator patterns across the held-out nuclei.
            divergence = {}
            for nucleus in residuals:
                bulk_value = normalized.get("bulk", {}).get(nucleus)
                grad_value = normalized.get("grad_masked_eps1", {}).get(nucleus)
                if bulk_value is None or grad_value is None or bulk_value == 0.0:
                    continue
                divergence[nucleus] = _number(float(grad_value) / float(bulk_value))
            part_b["surface_over_bulk_pattern_ratio"] = divergence

    # --- Part C: quantum surface with fixed physical coefficients -----------
    part_c: dict[str, Any] = {"per_nucleus": {}, "coefficient_variants": {
        "ETF2": "hbar^2/(36 m*) systematic 2nd-order ETF gradient term",
        "von_Weizsacker": "hbar^2/(8 m*) single-orbital form; exactly 4.5x ETF2",
        "von_Weizsacker_over_ETF2": _number(VW_OVER_ETF2, 10),
    }}
    etf2_energies: dict[str, float] = {}
    for nucleus, profile in profiles.items():
        block = nucleus_quantum_surface(profile, ETF2_COEFF)
        part_c["per_nucleus"][nucleus] = block
        etf2_energies[nucleus] = float(block["energy_total_masked_MeV"]["1.0"])
    if "Ca40" in etf2_energies:
        t40_c = float(baselines["Ca40"]["metrics"]["T_W_MeV"])
        e40 = etf2_energies["Ca40"]
        anchor_retuning = {
            "ds_over_s_for_Ca40_calibration": _number(-e40 / (2.0 * t40_c), 10),
            "identity": "dE/ds = 2 T_W/s => ds/s = -dE/(2 T_W)",
            "linearity_caveat": "first-order estimate; a large |ds/s| is outside the linear regime and is reported as a scale diagnostic only",
        }
        part_c["anchor_retuning"] = anchor_retuning
        predictions = {}
        inputs_prod = bridge._load_inputs()
        for nucleus, energy in etf2_energies.items():
            if nucleus == "Ca40":
                continue
            values = operator_values.get(nucleus)
            if values is None:
                continue
            a = values["A"]
            t = values["T_W"]
            shift = anchored_operator_slope(energy, t, e40, t40_c, a)
            if nucleus == CA48:
                residual = float(cases["no_rho"]["residual_vs_B_nuc_target_per_A_MeV"])
            else:
                prod_target = bridge.bare_binding_per_A_MeV(inputs_prod["binding"]["values"][nucleus])
                residual = float(baselines[nucleus]["metrics"]["binding_per_A_MeV"]) - prod_target
            predictions[nucleus] = {
                "dE_ETF2_masked_MeV": _number(energy),
                "anchored_shift_dB_per_A_MeV": _number(shift),
                "residual_before_MeV": _number(residual),
                "residual_after_first_order_MeV": _number(residual + shift),
                "fraction_of_residual_removed": _number(abs(shift) / abs(residual), 10) if residual != 0.0 else None,
            }
        part_c["anchored_predictions_ETF2_masked"] = predictions
        # Full-integral variant for transparency (declared edge-dominated).
        full_energies = {
            nucleus: float(part_c["per_nucleus"][nucleus]["energy_total_full_MeV"])
            for nucleus in part_c["per_nucleus"]
        }
        part_c["full_integral_variant_note"] = {
            "energies_full_MeV": {nucleus: _number(value) for nucleus, value in full_energies.items()},
            "note": "full integrals are dominated by the TF compact-support edge where the gradient expansion is invalid; the masked variant is the defensible estimate",
        }

    # --- Controls and provenance -------------------------------------------
    previous = _load_previous_summary()
    controls = {
        "analytic_validation": analytic_validation(),
        "decimation_Ca40": decimation_control(baselines["Ca40"]["profile"]),
        "y_vacuum_gate": "vacuum y=1 enforced at profile extraction for every accepted case",
        "previous_probe_summary": {
            "status": previous["status"],
            "path": previous.get("path"),
            "j_required_chords_MeV": (
                {
                    nucleus: previous["data"]["derived"][nucleus]
                    for nucleus in ("Zr90", "Pb208")
                    if nucleus in previous.get("data", {}).get("derived", {})
                }
                if previous["status"] == "LOADED"
                else None
            ),
        },
    }

    return _jsonable({
        "schema": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "provenance": {
            "family": FAMILY,
            "root_scale": _number(ROOT_SCALE),
            "root_scale_source": ROOT_SCALE_SOURCE,
            "baseline_rho": BASELINE_RHO,
            "anchor": "Ca40 sole conditional calibration anchor (unchanged; no retuning here)",
            "ca48_registration": "48Ca named as a discriminating held-out input by NVG_RESIDUAL_RESEARCH_DIRECTIONS_RU.md before this run; it was not used in any calibration",
            "no_new_interaction_declared": True,
            "no_coefficient_tuned_to_residuals": True,
        },
        "baseline_reproduction": baseline_public,
        "part_a_ca48_heldout": part_a,
        "part_b_operator_split": part_b,
        "part_c_quantum_surface": part_c,
        "controls": controls,
        "limits": [
            "conditional diagnostic of one accepted branch; evidence_weight=0; no statistical claim",
            "48Ca is held out from calibration, but its binding and radii are well-measured: this is a descriptive out-of-sample comparison, not a blind prediction",
            "required couplings are conditional arithmetic (residual/slope), not fitted parameters and not part of the theory",
            "frozen-profile first-order estimates only; the gradient corrections are not included self-consistently in the BVP",
            "TF profiles have compact support: the full gradient integrals are edge-dominated and invalid there; masked variants use the declared eps thresholds (0.5, 1.0)",
            "ETF-2/von-Weizsacker coefficients are the non-relativistic forms with the local effective mass M y(r); relativistic gradient completions (e.g. PRC 46, 230) and field-gradient cross terms are not evaluated",
            "spin-orbit, shell, pairing, center-of-mass and two-body terms are absent; the anchor-retuning ds/s is a scale diagnostic, not a re-calibration",
            "CREX skin is model-extracted from the weak form factor; the comparison is descriptive",
        ],
    })


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional explicit path for the strict JSON result",
    )
    args = parser.parse_args(argv)
    result = calculate()
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
