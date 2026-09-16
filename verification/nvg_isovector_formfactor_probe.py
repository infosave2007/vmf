#!/usr/bin/env python3
"""Isovector-contact sensitivity and form-factor probe on the accepted W8.93 branch.

This module executes the first declared physical step of
``NVG_RESIDUAL_RESEARCH_DIRECTIONS_RU.md`` (directions I and III) on the
accepted finite-static baseline (W8.93, ``no_rho``, s*=0.2268, Ca40 anchor):

* reproduce the three accepted no-rho baseline states and verify them
  against the retained R3 result before any new quantity is computed;
* compute the isovector integral ``I_A = (1/n0) ∫ (n_n - n_p)^2 d^3r`` with a
  bulk/surface split and the anchor-preserving binding sensitivity
  ``d(B_A/A)/dj|_{B40} = -[I_A - (T_A/T_40) I_40]/A`` (the symbolic identity
  of the research note, evaluated on live profiles; never fitted to Zr/Pb);
* measure the actual self-consistent response of energies, radii and the
  neutron skin to two declared contact probes: the existing
  ``covariant_contact_J32`` (j≈19.36 MeV) and one explicitly declared
  J=20 MeV probe (j≈7.36 MeV), each through the full terminal protocol at
  the unchanged scale s*;
* check the rho-on envelope identity ``dE/ds = 2 T_W / s`` by centered
  differences and derive the anchored skin response;
* compute point proton/neutron form factors, moments, first diffraction
  minima and declared plane-wave charge/weak convolutions with dipole and
  Galster nucleon electric form factors and tree-level CVC electric weak
  form factors.

No new interaction is declared, no coefficient is tuned to any Zr/Pb
discrepancy, and nothing here is an empirical confirmation.  The result is a
conditional diagnostic with ``evidence_weight=0``.  Importing this module
performs no numerical work and no network access.  The CLI prints strict
JSON to stdout; ``--output`` is an explicit opt-in write.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:  # Both direct-script and package-style imports are supported.
    import nvg_finite_monopole as static
    import nvg_finite_static_bridge as bridge
except ImportError:  # pragma: no cover - package import support.
    from . import nvg_finite_monopole as static
    from . import nvg_finite_static_bridge as bridge

from scipy.integrate import simpson

HERE = Path(__file__).resolve().parent
SCHEMA = "nvg_isovector_formfactor_probe.v1"
STATUS = "COMPUTED_ISOVECTOR_CONTACT_AND_FORMFACTOR_PROBE_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

FAMILY = "W8.93"
ROOT_SCALE = 0.22679999999999997
ROOT_SCALE_SOURCE = "Lunacy/runs/observable-bridge-2026-09-15/evidence_R3/bridge_result_r3.json"
R3_PATH = HERE.parent / ROOT_SCALE_SOURCE
BASELINE_RHO = "no_rho"
CONTACT_RHO = "covariant_contact_J32"
J32_DESIGN_MEV = 32.0  # live static.J_DESIGN_MEV protocol value
J20_PROBE_MEV = 20.0
NUCLEI = ("Ca40", "Zr90", "Pb208")
N0_FM3 = static.N0_FM3

ENVELOPE_DS = 0.0025
NUMBER_REL_TOL = 1.0e-6
REPRO_TOL = 1.0e-8
DOMAIN_LEVEL = {"box_fm": 40.0, "nodes": 1201, "tol": 2.0e-8}

# Accepted R3 values (frozen no-rho baseline); the probe fails closed if the
# live solver does not reproduce them.  They are reproducibility witnesses,
# not fitting targets: no quantity below is adjusted to any Zr/Pb residual.
BASELINE_EXPECTED = {
    "Ca40": {
        "binding_per_A_MeV": 8.551607091172059,
        "rms_neutron_radius_fm": 2.930642370107997,
        "rms_point_proton_radius_fm": 2.959371534110888,
        "T_W_MeV": 117.74879665320242,
        "total_E_MeV": 37217.93571635312,
    },
    "Zr90": {
        "binding_per_A_MeV": 8.850419693055073,
        "rms_neutron_radius_fm": 3.9010242180216093,
        "rms_point_proton_radius_fm": 3.9386636595786566,
        "T_W_MeV": 202.6191870459901,
        "total_E_MeV": 83713.46222762504,
    },
    "Pb208": {
        "binding_per_A_MeV": 8.322821939572577,
        "rms_neutron_radius_fm": 5.238221091766043,
        "rms_point_proton_radius_fm": 5.319947436693366,
        "T_W_MeV": 356.0198180612069,
        "total_E_MeV": 193580.8530365689,
    },
}

# Form-factor grids (fm^-1).  The q domain stays well inside the range where
# the TF density itself is defined; no claim is made beyond the declared grid.
FF_Q_GRID = tuple(round(0.05 * i, 10) for i in range(1, 33))  # 0.05..1.60
FF_ZERO_SCAN_STEP_FM = 0.002
FF_ZERO_SCAN_QMAX_FM = 2.5
FF_MOMENT_FIT_QMAX_FM = 0.10
MOMENT_Q_GRID = (0.005, 0.01, 0.015)  # dedicated very-small-q slope nodes

# Declared approximate charge/weak operator inputs.  Radiative corrections,
# spin-orbit, center-of-mass and two-body currents are NOT included; the
# neutron weak spatial distribution is approximated by its electric form
# factor scaled by the weak charge (declared limitation, not a prediction).
SIN2_THETA_W = 0.2312  # on-shell convention
Q_WEAK_P = 1.0 - 4.0 * SIN2_THETA_W
Q_WEAK_N = -1.0
GF_GEV_MINUS2 = 1.1663787e-5
HBARC_GEV_FM = 0.1973269804
DIPOLE_M2_GEV2 = 0.71
MU_NEUTRON = -1.913
GALSTER_B = 5.6
MASS_N_GEV = 0.939
APV_Q_LIST_FM = (0.30, 0.397, 0.50)  # PREX-II-like diagnostic kinematics

# Uniform-sphere first zero of 3 j1(x)/x in x = q R.
UNIFORM_SPHERE_FIRST_ZERO_X = 4.493409457909064


class IsovectorProbeError(ValueError):
    """Fail-closed error for protocol failures or non-reproducible baselines."""


def _number(value: Any, digits: int = 17) -> float:
    return bridge._number(value, digits)


def _jsonable(value: Any) -> Any:
    return bridge._jsonable(value)


# ---------------------------------------------------------------------------
# Live-solver access with explicit, always-restored protocol patches.
# ---------------------------------------------------------------------------


def _rho_context(rho_choice: str):
    """Temporarily select the bridge rho choice; always restore it."""

    class _Patch:
        def __enter__(self):
            self._saved = bridge.RHO_CHOICE
            bridge.RHO_CHOICE = rho_choice
            return self

        def __exit__(self, *exc):
            bridge.RHO_CHOICE = self._saved
            return False

    return _Patch()


def _j_design_context(j_design: float | None):
    """Temporarily override the declared rho-coupling design input."""

    if j_design is None:
        return nullcontext()

    class _Patch:
        def __enter__(self):
            self._saved = static.J_DESIGN_MEV
            static.J_DESIGN_MEV = float(j_design)
            return self

        def __exit__(self, *exc):
            static.J_DESIGN_MEV = self._saved
            return False

    return _Patch()


def _terminal_case(nucleus: str, rho_choice: str, j_design: float | None = None) -> dict[str, Any]:
    """Run the full terminal protocol at the frozen root scale.

    The bridge module reads its rho choice at call time, so an explicit
    context patch is the narrow seam for probe rho choices.  Fail closed if
    the terminal protocol does not accept the state.
    """

    with _rho_context(rho_choice), _j_design_context(j_design):
        case = bridge._terminal_case(FAMILY, ROOT_SCALE, nucleus, multi_seed=True)
    if case.get("terminal_protocol_acceptance") is not True:
        raise IsovectorProbeError(
            f"terminal protocol rejected {nucleus} at rho={rho_choice}, j_design={j_design}"
        )
    return case


def _contact_j_added(j_design: float | None = None) -> float:
    """The contact's added symmetry energy j (MeV) for a declared J design."""

    design = static.FiniteDesign.from_continuous(FAMILY, format(ROOT_SCALE, ".17g"))
    with _j_design_context(j_design):
        c_rho, _kappa = static._rho_coupling(design, CONTACT_RHO)
    n0_nat = N0_FM3 * design.hbarc**3
    j_added = float(c_rho) * n0_nat / 8.0
    if not math.isfinite(j_added) or j_added < 0.0:
        raise IsovectorProbeError("declared contact probe has invalid j")
    return j_added


def _continuation_solution(
    scale: float,
    nucleus: str,
    rho_choice: str,
    j_design: float | None,
    predecessor: Any,
) -> Any:
    """One continuation BVP at a shifted scale; fail closed on any gate."""

    design = static.FiniteDesign.from_continuous(FAMILY, format(scale, ".17g"))
    with _j_design_context(j_design):
        solved = static._solve_static_once(
            design,
            nucleus,
            rho_choice,
            box_fm=float(DOMAIN_LEVEL["box_fm"]),
            nodes=int(DOMAIN_LEVEL["nodes"]),
            tol=float(DOMAIN_LEVEL["tol"]),
            factor=None,
            previous=predecessor,
            nucleus_data=static.BRIDGE_NUCLEI,
        )
    row = solved.row
    if not row.get("converged") is True:
        raise IsovectorProbeError(f"continuation solve failed to converge ({nucleus}, s={scale})")
    for key in ("stationarity_acceptance", "localized_vacuum_exterior"):
        if row.get(key) is not True:
            raise IsovectorProbeError(f"continuation solve failed {key} ({nucleus}, s={scale})")
    if float(row.get("conserved_NZ_relative_error", math.inf)) > NUMBER_REL_TOL:
        raise IsovectorProbeError(f"continuation solve lost particle number ({nucleus}, s={scale})")
    return solved


# ---------------------------------------------------------------------------
# Profile-level observables.
# ---------------------------------------------------------------------------


def _profiles(solved: Any) -> dict[str, Any]:
    """Physical-radius species densities of a converged domain solution."""

    row = solved.row
    design = solved.design
    try:
        xx = np.asarray(row["profile_grid_x"], dtype=float)
        nn = np.asarray(row["profile_nn"], dtype=float)
        pp = np.asarray(row["profile_np"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise IsovectorProbeError("solution row lacks profile arrays") from exc
    if xx.ndim != 1 or xx.size < 100 or nn.shape != xx.shape or pp.shape != xx.shape:
        raise IsovectorProbeError("profile arrays are malformed")
    if not (np.all(np.isfinite(xx)) and np.all(np.isfinite(nn)) and np.all(np.isfinite(pp))):
        raise IsovectorProbeError("profile arrays are nonfinite")
    if np.any(nn < 0.0) or np.any(pp < 0.0):
        raise IsovectorProbeError("profile densities are negative")
    scale_fm = design.hbarc / design.W0
    n_to_fm3 = design.W0**3 / design.hbarc**3
    return {
        "r_fm": xx * scale_fm,
        "nn_fm3": nn * n_to_fm3,
        "np_fm3": pp * n_to_fm3,
        "W0_MeV": float(design.W0),
        "hbarc_MeV_fm": float(design.hbarc),
    }


def _spherical_integral(r: np.ndarray, f: np.ndarray) -> float:
    return 4.0 * math.pi * float(simpson(np.asarray(f, dtype=float) * r**2, x=r))


def _half_density_radius(r: np.ndarray, density: np.ndarray) -> float:
    """Linear interpolation of the first r where density crosses half-center."""

    center = float(density[0])
    if not math.isfinite(center) or center <= 0.0:
        raise IsovectorProbeError("central density is not positive")
    target = 0.5 * center
    below = np.flatnonzero(density < target)
    if below.size == 0:
        raise IsovectorProbeError("density never crosses half the central value")
    i = int(below[0])
    if i == 0:
        return float(r[0])
    r0, r1 = float(r[i - 1]), float(r[i])
    d0, d1 = float(density[i - 1]), float(density[i])
    if d0 == d1:
        return r1
    return r0 + (target - d0) * (r1 - r0) / (d1 - d0)


def isovector_metrics(profile: Mapping[str, np.ndarray], n0_fm3: float = N0_FM3) -> dict[str, Any]:
    """I_A with a half-density bulk/surface split (declared diagnostic split)."""

    r = np.asarray(profile["r_fm"], dtype=float)
    nn = np.asarray(profile["nn_fm3"], dtype=float)
    pp = np.asarray(profile["np_fm3"], dtype=float)
    n3 = nn - pp
    i_total = _spherical_integral(r, n3**2) / n0_fm3
    r_half = _half_density_radius(r, nn + pp)
    inside = r <= r_half
    i_bulk = _spherical_integral(r[inside], n3[inside] ** 2) / n0_fm3
    i_surface = i_total - i_bulk
    return {
        "I_A_dimensionless": float(i_total),
        "I_bulk_dimensionless": float(i_bulk),
        "I_surface_dimensionless": float(i_surface),
        "half_density_radius_fm": float(r_half),
        "surface_fraction": float(i_surface / i_total) if i_total > 0.0 else None,
    }


def anchored_binding_slope(
    i_a: float, t_a: float, i_ref: float, t_ref: float, a: int
) -> float:
    """d(B_A/A)/dj at fixed B_40 anchor (envelope-identity chain rule)."""

    if t_ref <= 0.0:
        raise IsovectorProbeError("reference scalar gradient T_ref must be positive")
    return -(i_a - (t_a / t_ref) * i_ref) / float(a)


def _terminal_metrics(nucleus: str, case: Mapping[str, Any]) -> dict[str, Any]:
    domain = case.get("domain40_check", {})
    solved = case.get("_domain_solution")
    if solved is None:
        raise IsovectorProbeError(f"terminal case lacks a live domain solution ({nucleus})")
    binding = bridge._binding(domain)
    if binding is None:
        raise IsovectorProbeError(f"binding unavailable for {nucleus}")
    components = domain.get("energy_components_MeV", {})
    profile = _profiles(solved)
    iso = isovector_metrics(profile)
    spec = static.BRIDGE_NUCLEI[nucleus]
    rms_n = float(domain["rms_neutron_radius_fm"])
    rms_p = float(domain["rms_point_proton_radius_fm"])
    return {
        "A": int(spec["A"]),
        "N": int(spec["N"]),
        "Z": int(spec["Z"]),
        "binding_per_A_MeV": float(binding),
        "T_W_MeV": float(components["T_W"]),
        "E_rho_MeV": float(components["E_rho"]),
        "total_E_MeV": float(components["total_E"]),
        "rms_neutron_radius_fm": rms_n,
        "rms_point_proton_radius_fm": rms_p,
        "point_neutron_skin_fm": rms_n - rms_p,
        "central_density_fm_minus3": float(domain["central_density_fm_minus3"]),
        "chemical_potentials_MeV": domain.get("chemical_potentials_MeV"),
        "conserved_NZ_relative_error": float(domain["conserved_NZ_relative_error"]),
        "terminal_classification": case.get("terminal_classification"),
        "isovector": iso,
        "_profile": profile,
        "_solved": solved,
    }


# ---------------------------------------------------------------------------
# Form factors (point species densities and declared nucleon operators).
# ---------------------------------------------------------------------------


def _spherical_bessel_j0(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1.0e-8
    out[small] = 1.0 - x[small] ** 2 / 6.0
    xs = x[~small]
    out[~small] = np.sin(xs) / xs
    return out


def species_form_factor(
    r: np.ndarray, density: np.ndarray, count: int, q_grid: Sequence[float]
) -> dict[str, Any]:
    """Normalized point-density form factor F(q)=4pi/N ∫ r^2 n j0(qr) dr."""

    r = np.asarray(r, dtype=float)
    density = np.asarray(density, dtype=float)
    count = float(count)
    if count <= 0.0:
        raise IsovectorProbeError("species count must be positive")
    norm = _spherical_integral(r, density)
    values = {}
    for q in q_grid:
        fq = _spherical_integral(r, density * _spherical_bessel_j0(q * r)) / count
        values[_number(q, 10)] = _number(fq)
    r2 = _spherical_integral(r, density * r**2) / count
    r4 = _spherical_integral(r, density * r**4) / count
    r6 = _spherical_integral(r, density * r**6) / count
    # First diffraction minimum: first sign change of F on the declared scan.
    scan = np.arange(FF_ZERO_SCAN_STEP_FM, FF_ZERO_SCAN_QMAX_FM + 0.5 * FF_ZERO_SCAN_STEP_FM, FF_ZERO_SCAN_STEP_FM)
    previous = 1.0
    first_zero = None
    for q in scan:
        fq = _spherical_integral(r, density * _spherical_bessel_j0(q * r)) / count
        if previous > 0.0 >= fq:
            # Linear interpolation of the crossing between the last two nodes.
            q0 = q - FF_ZERO_SCAN_STEP_FM
            f0 = previous
            first_zero = q0 + (0.0 - f0) * (q - q0) / (fq - f0) if fq != f0 else q
            break
        previous = fq
    # Small-q moment consistency: F(q) = 1 - q^2 r2/6 + q^4 r4/120 + O(q^6).
    checks = []
    for q in (0.05, 0.10, 0.15):
        fq = _spherical_integral(r, density * _spherical_bessel_j0(q * r)) / count
        taylor = 1.0 - q * q * r2 / 6.0 + q**4 * r4 / 120.0
        checks.append(abs(fq - taylor))
    return {
        "F_of_q": values,
        "integrated_count": _number(norm),
        "declared_count": _number(count),
        "count_relative_difference": _number(abs(norm - count) / max(abs(count), 1.0e-30)),
        "rms_from_moments_fm": math.sqrt(r2),
        "moment_r2_fm2": r2,
        "moment_r4_fm4": r4,
        "moment_r6_fm6": r6,
        "first_zero_q_fm": None if first_zero is None else float(first_zero),
        "first_zero_search": {
            "step_fm": FF_ZERO_SCAN_STEP_FM,
            "q_max_fm": FF_ZERO_SCAN_QMAX_FM,
            "none_found_note": "no sign change of F on the declared scan grid",
        },
        "small_q_taylor_max_abs_deviation": max(checks),
    }


def _nucleon_electric_form_factors(q_fm: float) -> tuple[float, float]:
    """Dipole G_E^p and Galster G_E^n at |q| in fm^-1 (declared approximation)."""

    q_gev = q_fm * HBARC_GEV_FM
    q2 = q_gev * q_gev
    dipole = (1.0 + q2 / DIPOLE_M2_GEV2) ** (-2)
    tau = q2 / (4.0 * MASS_N_GEV**2)
    g_ep = dipole
    g_en = -MU_NEUTRON * tau * dipole / (1.0 + GALSTER_B * tau)
    return g_ep, g_en


def _nucleon_weak_electric_form_factors(q_fm: float) -> tuple[float, float]:
    """Tree-level CVC electric weak form factors (strangeness neglected).

    With J_NC = J_3 - 2 sin^2(theta_W) J_EM, the electric weak form factors
    are G_EW,p = (1-4s2) G_Ep - G_En and G_EW,n = -G_Ep + (1-4s2) G_En.
    Both reduce to the weak charges Q_W^p = 1-4s2 and Q_W^n = -1 at q=0;
    using G_E^n alone for the neutron weak density would wrongly vanish at
    q=0 and destroy the F_W(0)=1 normalization.
    """

    g_ep, g_en = _nucleon_electric_form_factors(q_fm)
    g_ewp = Q_WEAK_P * g_ep - g_en
    g_ewn = -g_ep + Q_WEAK_P * g_en
    return g_ewp, g_ewn


def charge_weak_form_factors(
    r: np.ndarray,
    nn: np.ndarray,
    np_: np.ndarray,
    n_n: int,
    z: int,
    q_grid: Sequence[float],
) -> dict[str, Any]:
    """Declared plane-wave charge/weak convolutions with nucleon form factors.

    F_ch(q) = [Z G_Ep F_p + N G_En F_n]/Z and the weak density
    W(q) = Z G_EW,p F_p + N G_EW,n F_n with the tree-level CVC electric weak
    form factors, so F_W = W/Q_W^A satisfies F_W(0)=1 exactly.  No Coulomb
    distortion, no acceptance, no radiative/strange/spin-orbit/CM terms:
    these are operator diagnostics, not experimental predictions.
    """

    combined_q = sorted(set(float(q) for q in q_grid) | set(float(q) for q in APV_Q_LIST_FM) | set(MOMENT_Q_GRID))

    def point_ff(density: np.ndarray, count: float, q: float) -> float:
        return _spherical_integral(r, density * _spherical_bessel_j0(q * r)) / count

    f_p = {q: point_ff(np_, float(z), q) for q in combined_q}
    f_n = {q: point_ff(nn, float(n_n), q) for q in combined_q}
    f_ch = {}
    f_w = {}
    for q in combined_q:
        g_ep, g_en = _nucleon_electric_form_factors(q)
        f_ch[q] = (z * g_ep * f_p[q] + n_n * g_en * f_n[q]) / z
        g_ewp, g_ewn = _nucleon_weak_electric_form_factors(q)
        weak_total = Q_WEAK_P * z + Q_WEAK_N * n_n
        w_unnorm = z * g_ewp * f_p[q] + n_n * g_ewn * f_n[q]
        f_w[q] = w_unnorm / weak_total
    # Slope moments from a dedicated very-small-q linear fit in q^2.
    q2 = np.array([q * q for q in MOMENT_Q_GRID])
    fch_small = np.array([f_ch[q] for q in MOMENT_Q_GRID])
    fw_small = np.array([f_w[q] for q in MOMENT_Q_GRID])
    slope_ch = float(np.polyfit(q2, fch_small, 1)[0])
    slope_w = float(np.polyfit(q2, fw_small, 1)[0])
    r2_ch = -6.0 * slope_ch
    r2_w = -6.0 * slope_w
    # Plane-wave A_PV diagnostic with the full weak-charge normalization.
    # Value is expressed in ppm (the dimensionless asymmetry is ~1e-7 here).
    apv = {}
    for q in APV_Q_LIST_FM:
        q_gev = q * HBARC_GEV_FM
        q2_gev = q_gev * q_gev
        prefactor = GF_GEV_MINUS2 * q2_gev / (4.0 * math.pi * static.ALPHA * math.sqrt(2.0))
        g_ewp, g_ewn = _nucleon_weak_electric_form_factors(q)
        w_unnorm = z * g_ewp * f_p[q] + n_n * g_ewn * f_n[q]
        apv[_number(q, 10)] = _number(-prefactor * w_unnorm / (z * f_ch[q]) * 1.0e6)
    q_smallest = min(combined_q)
    return {
        "F_ch_of_q": {_number(q, 10): _number(v) for q, v in f_ch.items()},
        "F_W_of_q": {_number(q, 10): _number(v) for q, v in f_w.items()},
        "F_W_normalization_check": {
            "q_fm": _number(q_smallest, 10),
            "F_W_minus_one": _number(f_w[q_smallest] - 1.0),
            "note": "F_W(0)=1 by the CVC weak-charge normalization; the deviation at the smallest grid q is the physical slope",
        },
        "moment_r2_charge_fm2": r2_ch,
        "moment_r2_weak_fm2": r2_w,
        "rms_charge_from_slope_fm": math.sqrt(r2_ch) if r2_ch > 0 else None,
        "rms_weak_from_slope_fm": math.sqrt(r2_w) if r2_w > 0 else None,
        "weak_minus_charge_radius_fm": (
            math.sqrt(r2_w) - math.sqrt(r2_ch) if r2_w > 0 and r2_ch > 0 else None
        ),
        "moment_fit_note": "slope from a dedicated very-small-q fit; sub-percent residual curvature bias",
        "A_PV_pw_per_million": apv,
        "A_PV_pw_convention": (
            "A_PV=(sigma_R-sigma_L)/(sigma_R+sigma_L) plane-wave Born in ppm with the "
            "full weak-charge normalization Q_W^A=(1-4s2)Z-N; the sign matches the "
            "positive PREX convention via Q_W=N-(1-4s2)Z; Coulomb distortion, "
            "acceptance and radiative corrections are not included, so these are "
            "diagnostics, not PREX/CREX predictions"
        ),
        "operator_inputs": {
            "sin2_theta_w_on_shell": SIN2_THETA_W,
            "Q_weak_proton": Q_WEAK_P,
            "Q_weak_neutron": Q_WEAK_N,
            "G_E_proton": "dipole (1+q^2/0.71 GeV^2)^-2",
            "G_E_neutron": "Galster -mu_n tau G_D/(1+5.6 tau)",
            "G_E_weak_proton": "(1-4s2) G_Ep - G_En (tree-level CVC, strangeness neglected)",
            "G_E_weak_neutron": "-G_Ep + (1-4s2) G_En (tree-level CVC, strangeness neglected)",
        },
    }


def analytic_formfactor_validation() -> dict[str, Any]:
    """Gaussian and uniform-sphere controls for the form-factor code path."""

    r = np.linspace(0.0, 12.0, 4801)
    sigma = 1.7
    gauss = np.exp(-(r**2) / (2.0 * sigma**2))
    count_g = _spherical_integral(r, gauss)
    ff_g = species_form_factor(r, gauss, count_g, (0.1, 0.3, 0.6, 0.9))
    expected_g = {q: math.exp(-(q**2) * sigma**2 / 2.0) for q in (0.1, 0.3, 0.6, 0.9)}
    max_gauss = max(abs(ff_g["F_of_q"][_number(q, 10)] - v) for q, v in expected_g.items())
    r2_gauss_err = abs(ff_g["moment_r2_fm2"] - 3.0 * sigma**2)

    r_uni = np.linspace(0.0, 10.0, 4001)
    radius = 4.2
    inside = r_uni <= radius
    uni = np.where(inside, 1.0, 0.0)
    count_u = _spherical_integral(r_uni, uni)
    ff_u = species_form_factor(r_uni, uni, count_u, (0.1, 0.5, 1.0))

    def uniform_f(q: float) -> float:
        x = q * radius
        return 3.0 * (math.sin(x) - x * math.cos(x)) / x**3

    max_uni = max(abs(ff_u["F_of_q"][_number(q, 10)] - uniform_f(q)) for q in (0.1, 0.5, 1.0))
    r2_uni_err = abs(ff_u["moment_r2_fm2"] - 0.6 * radius**2)
    first_zero = ff_u["first_zero_q_fm"]
    zero_err = (
        abs(first_zero - UNIFORM_SPHERE_FIRST_ZERO_X / radius)
        if first_zero is not None
        else math.inf
    )
    return {
        "gaussian_max_abs_F_error": float(max_gauss),
        "gaussian_r2_error_fm2": float(r2_gauss_err),
        "uniform_sphere_max_abs_F_error": float(max_uni),
        "uniform_sphere_r2_error_fm2": float(r2_uni_err),
        "uniform_sphere_first_zero_error_fm": float(zero_err),
        "uniform_sphere_tolerance_note": (
            "the uniform-sphere tolerances absorb the O(h) Simpson error of a "
            "discontinuous edge on the fixed grid; the production TF densities "
            "are smooth, where the Gaussian control demonstrates fast convergence"
        ),
        "pass": bool(
            max_gauss < 1.0e-8
            and r2_gauss_err < 1.0e-6
            and max_uni < 1.0e-3
            and r2_uni_err < 1.0e-2
            and zero_err < 2.0 * FF_ZERO_SCAN_STEP_FM
        ),
    }


def anchored_identity_toy_check() -> dict[str, Any]:
    """Numeric control of the anchored slope formula on a toy envelope model.

    Toy energy E_A(s,j) = a_A s^2 + j i_A s^p reproduces the production
    envelope structure dE/ds = 2 T_A/s with T_A = a_A s^2, and dE/dj|_s =
    I_A = i_A s^p.  Holding B_40 fixed and central-differencing B_A/A in j
    must reproduce -(I_A - (T_A/T_40) I_40)/A to first order in the probe
    step.  This checks the implemented algebra, not the nuclear model.
    """

    a40, a90, i40, i90 = 6.0e5, 1.4e6, 0.02, 0.14
    p = 0.5
    s0 = 0.22
    a_ca, a_zr = 40.0, 90.0
    mass = 939.0

    def energy(a_coeff: float, i_coeff: float, s: float, j: float) -> float:
        return a_coeff * s**2 + j * i_coeff * s**p

    def binding(a_count: float, a_coeff: float, i_coeff: float, s: float, j: float) -> float:
        return a_count * mass - energy(a_coeff, i_coeff, s, j)

    b40_target = binding(a_ca, a40, i40, s0, 0.0)

    def anchor_scale(j: float) -> float:
        # B is strictly decreasing in s here, so bisection keeps the side
        # that still brackets the anchor-preserving root.
        lo, hi = 1.0e-3, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if binding(a_ca, a40, i40, mid, j) > b40_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    dj = 1.0e-5
    s_plus, s_minus = anchor_scale(dj), anchor_scale(-dj)
    b_plus = binding(a_zr, a90, i90, s_plus, dj) / a_zr
    b_minus = binding(a_zr, a90, i90, s_minus, -dj) / a_zr
    numeric = (b_plus - b_minus) / (2.0 * dj)
    t40 = a40 * s0**2
    t90 = a90 * s0**2
    i40_s0 = i40 * s0**p
    i90_s0 = i90 * s0**p
    formula = anchored_binding_slope(i90_s0, t90, i40_s0, t40, 90)
    rel = abs(numeric - formula) / max(abs(numeric), abs(formula), 1.0e-30)
    return {
        "numeric_dB_per_A_dj": _number(numeric),
        "formula_dB_per_A_dj": _number(formula),
        "relative_difference": _number(rel),
        "pass": bool(rel < 1.0e-6),
    }


# ---------------------------------------------------------------------------
# Main computation.
# ---------------------------------------------------------------------------


def _baseline_block() -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduce and verify the accepted no-rho baseline states."""

    rows: dict[str, Any] = {}
    profiles: dict[str, Any] = {}
    for nucleus in NUCLEI:
        case = _terminal_case(nucleus, BASELINE_RHO)
        metrics = _terminal_metrics(nucleus, case)
        expected = BASELINE_EXPECTED[nucleus]
        checks = {}
        for key, expected_value in expected.items():
            actual = metrics[key]
            tolerance = REPRO_TOL * max(1.0, abs(expected_value))
            checks[key] = {
                "expected": _number(expected_value),
                "actual": _number(actual),
                "abs_difference": _number(abs(actual - expected_value)),
                "pass": bool(abs(actual - expected_value) <= tolerance),
            }
            if abs(actual - expected_value) > tolerance:
                raise IsovectorProbeError(
                    f"baseline reproduction failed for {nucleus}/{key}: "
                    f"{actual!r} vs {expected_value!r}"
                )
        rows[nucleus] = {key: value for key, value in metrics.items() if not key.startswith("_")}
        rows[nucleus]["reproduction_checks_vs_R3"] = checks
        profiles[nucleus] = metrics["_profile"]
    return rows, profiles


def _sensitivity_block(baseline_rows: Mapping[str, Any]) -> dict[str, Any]:
    """Anchor-preserving slopes d(B/A)/dj on the baseline profiles."""

    ca = baseline_rows["Ca40"]
    i_ref = ca["isovector"]["I_A_dimensionless"]
    t_ref = ca["T_W_MeV"]
    out = {
        "identity": "d(B_A/A)/dj|_{B_40 anchor} = -[I_A - (T_A/T_40) I_40]/A",
        "reference_Ca40": {
            "I_40_dimensionless": _number(i_ref),
            "T_40_MeV": _number(t_ref),
            "note": "T_A is the scalar-field gradient energy T_W, not the full fermion kinetic energy",
        },
        "per_nucleus": {},
    }
    for nucleus in NUCLEI:
        row = baseline_rows[nucleus]
        i_a = row["isovector"]["I_A_dimensionless"]
        t_a = row["T_W_MeV"]
        slope = anchored_binding_slope(i_a, t_a, i_ref, t_ref, row["A"])
        surface_fraction = row["isovector"]["surface_fraction"]
        out["per_nucleus"][nucleus] = {
            "I_A_dimensionless": _number(i_a),
            "I_bulk_dimensionless": _number(row["isovector"]["I_bulk_dimensionless"]),
            "I_surface_dimensionless": _number(row["isovector"]["I_surface_dimensionless"]),
            "half_density_radius_fm": _number(row["isovector"]["half_density_radius_fm"]),
            "surface_fraction": None if surface_fraction is None else _number(surface_fraction),
            "T_A_MeV": _number(t_a),
            "d_binding_per_A_dj_MeV_per_MeV": _number(slope),
            "predicted_d_binding_per_A_for_j_plus_1_MeV": _number(slope * 1.0),
            "anchored_ds_dj": _number(-ROOT_SCALE * i_ref / (2.0 * t_ref)),
        }
    return out


def _contact_block() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Self-consistent response to the two declared contact probes."""

    j32 = _contact_j_added(None)
    j20 = _contact_j_added(J20_PROBE_MEV)
    block: dict[str, Any] = {
        "probes": {
            "J32_live_protocol": {
                "design_J_MeV": J32_DESIGN_MEV,
                "j_added_MeV": _number(j32),
                "rho_choice": CONTACT_RHO,
            },
            "J20_declared_probe": {
                "design_J_MeV": J20_PROBE_MEV,
                "j_added_MeV": _number(j20),
                "rho_choice": CONTACT_RHO,
            },
            "delta_j_MeV": _number(j32 - j20),
            "linearity_note": "j is exactly linear in the declared design J (checked analytically via constant_rho_coupling)",
        },
        "per_nucleus": {},
    }
    probe_metrics: dict[str, dict[str, Any]] = {}
    for nucleus in NUCLEI:
        entry: dict[str, Any] = {}
        for label, j_design in (("J32", None), ("J20", J20_PROBE_MEV)):
            case = _terminal_case(nucleus, CONTACT_RHO, j_design)
            metrics = _terminal_metrics(nucleus, case)
            entry[label] = {
                key: value
                for key, value in metrics.items()
                if not key.startswith("_") and key != "isovector"
            }
            entry[label]["isovector"] = metrics["isovector"]
            probe_metrics[(nucleus, label)] = metrics
        block["per_nucleus"][nucleus] = entry
    return block, probe_metrics


def _response_block(
    baseline_rows: Mapping[str, Any],
    probe_metrics: Mapping[tuple[str, str], Mapping[str, Any]],
    j32: float,
    j20: float,
) -> dict[str, Any]:
    """Fixed-scale finite-difference response and E_rho = j I_A control."""

    dj = j32 - j20
    per_nucleus = {}
    for nucleus in NUCLEI:
        base = baseline_rows[nucleus]
        m32 = probe_metrics[(nucleus, "J32")]
        m20 = probe_metrics[(nucleus, "J20")]
        d_binding = (m32["binding_per_A_MeV"] - m20["binding_per_A_MeV"]) / dj
        d_skin = (m32["point_neutron_skin_fm"] - m20["point_neutron_skin_fm"]) / dj
        # Internal consistency: E_rho(j) must equal j*I_A on the same profile.
        e_rho_control = {}
        for label, j in (("J32", j32), ("J20", j20)):
            metrics = probe_metrics[(nucleus, label)]
            j_i = j * metrics["isovector"]["I_A_dimensionless"]
            e_rho_control[label] = {
                "E_rho_MeV": _number(metrics["E_rho_MeV"]),
                "j_times_I_A_MeV": _number(j_i),
                "relative_difference": _number(
                    abs(metrics["E_rho_MeV"] - j_i)
                    / max(abs(metrics["E_rho_MeV"]), abs(j_i), 1.0e-30)
                ),
            }
        per_nucleus[nucleus] = {
            "d_binding_per_A_dj_fixed_scale_MeV_per_MeV": _number(d_binding),
            "d_point_skin_dj_fixed_scale_fm_per_MeV": _number(d_skin),
            "delta_binding_per_A_J32_vs_baseline_MeV": _number(
                probe_metrics[(nucleus, "J32")]["binding_per_A_MeV"]
                - base["binding_per_A_MeV"]
            ),
            "delta_point_skin_J32_vs_baseline_fm": _number(
                probe_metrics[(nucleus, "J32")]["point_neutron_skin_fm"]
                - base["point_neutron_skin_fm"]
            ),
            "delta_binding_per_A_J20_vs_baseline_MeV": _number(
                probe_metrics[(nucleus, "J20")]["binding_per_A_MeV"]
                - base["binding_per_A_MeV"]
            ),
            "delta_point_skin_J20_vs_baseline_fm": _number(
                probe_metrics[(nucleus, "J20")]["point_neutron_skin_fm"]
                - base["point_neutron_skin_fm"]
            ),
            "E_rho_equals_jI_A_control": e_rho_control,
            "fixed_scale_difference_note": (
                "finite-difference response over dj=12 MeV at unchanged s*; "
                "not an infinitesimal derivative and not an anchored response"
            ),
        }
    return {
        "per_nucleus": per_nucleus,
        "anchor_retuning_note": (
            "the anchored slope adds the s(j) retuning term; compare "
            "d_binding_per_A_dj_fixed_scale with the baseline anchored slope "
            "in sensitivities.per_nucleus to isolate the retuning correction"
        ),
    }


def _envelope_block(probe_metrics: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    """Rho-on envelope identity check and anchored skin response (Ca40)."""

    base = probe_metrics[("Ca40", "J32")]
    solved = base["_solved"]
    plus = _continuation_solution(ROOT_SCALE + ENVELOPE_DS, "Ca40", CONTACT_RHO, None, solved)
    minus = _continuation_solution(ROOT_SCALE - ENVELOPE_DS, "Ca40", CONTACT_RHO, None, solved)
    e_plus = float(plus.row["energy_components_MeV"]["total_E"])
    e_minus = float(minus.row["energy_components_MeV"]["total_E"])
    de_ds = (e_plus - e_minus) / (2.0 * ENVELOPE_DS)
    t_w = base["T_W_MeV"]
    predicted = 2.0 * t_w / ROOT_SCALE

    def skin(row: Mapping[str, Any]) -> float:
        return float(row["rms_neutron_radius_fm"]) - float(row["rms_point_proton_radius_fm"])

    skin_plus = skin(plus.row)
    skin_minus = skin(minus.row)
    d_skin_ds = (skin_plus - skin_minus) / (2.0 * ENVELOPE_DS)
    i_ca = base["isovector"]["I_A_dimensionless"]
    ds_dj = -ROOT_SCALE * i_ca / (2.0 * t_w)
    block = {
        "nucleus": "Ca40",
        "rho_choice": CONTACT_RHO,
        "probe": "J32",
        "ds": ENVELOPE_DS,
        "dE_ds_centered_MeV": _number(de_ds),
        "predicted_dE_ds_2TW_over_s_MeV": _number(predicted),
        "relative_error": _number(abs(de_ds - predicted) / max(abs(de_ds), abs(predicted), 1.0e-30)),
        "d_point_skin_ds_fm": _number(d_skin_ds),
        "ds_dj_at_anchor_rho_on": _number(ds_dj),
        "anchored_skin_retuning_term_fm_per_MeV": _number(d_skin_ds * ds_dj),
        "anchored_skin_note": (
            "full anchored skin response = fixed-scale response (fixed_scale_response) "
            "+ this retuning term; the fixed-scale part uses a dj=12 MeV finite "
            "difference while the retuning term is infinitesimal at the J32 probe"
        ),
        "identity": "dE/ds = 2*T_W/s evaluated on the rho-on branch; continuation solves only, full gates enforced",
    }
    return block


def _formfactor_block(profiles: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    """Point species form factors plus declared charge/weak convolutions."""

    per_nucleus = {}
    for nucleus in NUCLEI:
        profile = profiles[nucleus]
        r = profile["r_fm"]
        nn = profile["nn_fm3"]
        np_ = profile["np_fm3"]
        spec = static.BRIDGE_NUCLEI[nucleus]
        f_p = species_form_factor(r, np_, spec["Z"], FF_Q_GRID)
        f_n = species_form_factor(r, nn, spec["N"], FF_Q_GRID)
        operators = charge_weak_form_factors(r, nn, np_, spec["N"], spec["Z"], FF_Q_GRID)
        per_nucleus[nucleus] = {
            "point_proton": f_p,
            "point_neutron": f_n,
            "charge_weak_operators": operators,
        }
    return {
        "method": "F(q)=4pi/N ∫ r^2 n(r) j0(qr) dr on the diagnostic ladder grid; particle numbers controlled before the transform and never renormalized away",
        "q_grid_fm": [_number(q, 10) for q in FF_Q_GRID],
        "per_nucleus": per_nucleus,
        "analytic_validation": analytic_formfactor_validation(),
        "anchored_identity_toy_check": anchored_identity_toy_check(),
    }


def calculate() -> dict[str, Any]:
    """Run the full probe; fail closed on any protocol or reproduction failure."""

    inputs = bridge._load_inputs()
    targets = bridge._input_targets(inputs)
    baseline_rows, profiles = _baseline_block()
    sensitivities = _sensitivity_block(baseline_rows)
    contact_block, probe_metrics = _contact_block()
    j32 = contact_block["probes"]["J32_live_protocol"]["j_added_MeV"]
    j20 = contact_block["probes"]["J20_declared_probe"]["j_added_MeV"]
    response = _response_block(baseline_rows, probe_metrics, j32, j20)
    envelope = _envelope_block(probe_metrics)
    formfactors = _formfactor_block(profiles)

    target_summary = {
        nucleus: {
            "B_nuc_target_per_A_MeV": _number(targets[nucleus]["B_nuc_target_per_A_MeV"]),
            "baseline_residual_MeV": _number(
                baseline_rows[nucleus]["binding_per_A_MeV"]
                - targets[nucleus]["B_nuc_target_per_A_MeV"]
            ),
        }
        for nucleus in NUCLEI
    }

    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "model_scope": {
            "family": FAMILY,
            "rho_baseline": BASELINE_RHO,
            "correlated_scale": _number(ROOT_SCALE),
            "root_scale_source": ROOT_SCALE_SOURCE,
            "calibration": "unchanged Ca40 anchor from the accepted R3 run; no refit, no new tuned coefficient",
            "contact_probe_status": "declared diagnostic probes through the live covariant_contact_J32 machinery; not a fitted interaction",
        },
        "targets_from_cited_inputs": target_summary,
        "baseline": {nucleus: _jsonable(rows) for nucleus, rows in baseline_rows.items()},
        "anchored_sensitivities": _jsonable(sensitivities),
        "contact_probe": _jsonable(contact_block),
        "fixed_scale_response": _jsonable(response),
        "envelope_check_rho_on": _jsonable(envelope),
        "form_factors": _jsonable(formfactors),
        "limits": [
            "evidence_weight=0: conditional diagnostics inside one declared model branch, not empirical confirmation",
            "the anchored slope is a local envelope/chain-rule identity; finite j changes require the new solves performed here and still larger j would need re-validation",
            "the fixed-scale response uses a dj=12 MeV finite difference, not an infinitesimal derivative",
            "point form factors are not the full charge form factor: nucleon form factors are approximate (dipole/Galster, tree-level CVC weak), spin-orbit, center-of-mass, two-body and relativistic corrections are not computed",
            "A_PV values are plane-wave Born diagnostics with a declared sign convention; Coulomb distortion and experimental acceptance are not included, so they are not PREX/CREX predictions (distortion is known to change heavy-nucleus asymmetries by tens of percent at PREX kinematics)",
            "the half-density bulk/surface split of I_A is a declared geometric diagnostic, not a derived operator decomposition",
            "no coefficient was fitted to the Zr90 or Pb208 residuals at any point",
        ],
    }


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="explicitly write the JSON result to this path")
    args = parser.parse_args(argv)
    try:
        result = calculate()
        encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        return 0
    except (ArithmeticError, IsovectorProbeError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({
            "schema_version": SCHEMA,
            "status": "INVALID_OR_FAILED_ISOVECTOR_FORMFACTOR_PROBE",
            "evidence_weight": EVIDENCE_WEIGHT,
            "error": str(exc),
        }, ensure_ascii=False, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
