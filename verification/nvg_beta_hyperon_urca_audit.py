#!/usr/bin/env python3
"""P2-S2 beta-equilibrium, hyperon and direct-Urca audit.

This entry point is deliberately independent of the maintained toy EOS files.
It solves a zero-temperature *ideal* (zero-interaction) ``n p Lambda e mu``
reference in natural units, and then audits the existing NVG/hyperon/Urca
routes without importing their arbitrary couplings.  The ideal gas is useful
for numerical controls (conservation, chemical equilibrium, the pressure
identity and the triangle conditions); it is not a physical interacting
hyperon EOS and carries zero evidence weight for an NVG claim.

The interacting hyperon model, a phase construction, emissivities and an NVG
hyperon dependency are intentionally fail-closed.  No constants are fitted to
stellar masses and no cooling claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
N0 = 0.16  # fm^-3
HBAR_C = 197.3269804  # MeV fm
M_SUN_KM = 1.4766
MEV_FM3_TO_GEO = 1.3234e-6

# CODATA/PDG masses are used only as fixed conventional reference constants;
# no observational table or fitted interaction parameter enters the solver.
MASS_MEV = {
    "n": 939.56542052,
    "p": 938.27208816,
    "Lambda": 1115.683,
    "e": 0.51099895,
    "mu": 105.6583755,
}

CANONICAL_REGRESSION = {
    "M_max": 2.047950740578197,
    "R_1.4": 12.550001000000044,
    "Lambda_1.4": 519.4223807918132,
}

RESULT_PATH = ROOT / "verification" / "nvg_beta_hyperon_urca_audit_p2s2_results.json"
FIGURE_PATH = ROOT / "verification" / "fig_p2_s2_beta_hyperon_urca_audit.png"


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def density_from_mu(mu: float, mass: float) -> float:
    """Spin-two Fermi-gas density in fm^-3 for a chemical potential in MeV."""

    if not (_finite(mu) and _finite(mass) and mass > 0.0):
        raise ValueError("chemical potential and mass must be finite and mass positive")
    if mu <= mass:
        return 0.0
    kf = math.sqrt(mu * mu - mass * mass)
    return kf**3 / (3.0 * math.pi**2 * HBAR_C**3)


def fermi_momentum(density: float) -> float:
    """Fermi momentum in MeV for a spin-two density in fm^-3."""

    if not _finite(density) or density < 0.0:
        raise ValueError("density must be finite and non-negative")
    return HBAR_C * (3.0 * math.pi**2 * density) ** (1.0 / 3.0) if density else 0.0


def inverse_density(density: float, mass: float) -> float:
    if density <= 0.0:
        return mass
    return math.sqrt(fermi_momentum(density) ** 2 + mass * mass)


def _fermi_terms(mu: float, mass: float) -> tuple[float, float, float, float]:
    """Return ``(n, epsilon, pressure, k_F)`` for one free species."""

    if mu <= mass:
        return 0.0, 0.0, 0.0, 0.0
    kf = math.sqrt(mu * mu - mass * mass)
    log_term = math.log((kf + mu) / mass)
    pref = math.pi**2 * HBAR_C**3
    energy = (kf * mu * (2.0 * kf * kf + mass * mass) - mass**4 * log_term) / (8.0 * pref)
    pressure = (kf * mu * (2.0 * kf * kf - 3.0 * mass * mass) + 3.0 * mass**4 * log_term) / (24.0 * pref)
    density = kf**3 / (3.0 * pref)
    return density, energy, pressure, kf


def _solve_baryon_mu(n_b: float, mu_e: float) -> float | None:
    """Solve baryon-number conservation for ``mu_n`` at fixed ``mu_e``.

    The lower endpoint is the neutron rest mass.  A candidate with a proton
    density already above ``n_b`` at that endpoint is outside the physical
    charge-neutral bracket and is reported as invalid to the outer root.
    """

    m_n = MASS_MEV["n"]
    lo = m_n
    hi = max(m_n, MASS_MEV["p"] + mu_e, MASS_MEV["Lambda"]) + 2000.0

    def residual(mu_n: float) -> float:
        return (
            density_from_mu(mu_n, MASS_MEV["n"])
            + density_from_mu(mu_n - mu_e, MASS_MEV["p"])
            + density_from_mu(mu_n, MASS_MEV["Lambda"])
            - n_b
        )

    f_lo = residual(lo)
    if f_lo >= 0.0:
        return None
    f_hi = residual(hi)
    while f_hi <= 0.0 and hi < 20000.0:
        hi += 2000.0
        f_hi = residual(hi)
    if f_hi <= 0.0:
        raise RuntimeError("baryon chemical-potential bracket exhausted")
    return float(brentq(residual, lo, hi, xtol=1.0e-11, rtol=1.0e-12, maxiter=200))


def solve_composition(n_b: float) -> dict[str, Any]:
    """Solve cold ``n p Lambda e mu`` beta equilibrium and neutrality.

    The Lambda has zero interaction couplings in this reference.  When it is
    absent the equilibrium condition is the complementarity inequality
    ``mu_n <= m_Lambda``; once populated ``mu_Lambda = mu_n`` exactly.
    """

    if not _finite(n_b) or n_b <= 0.0:
        raise ValueError("baryon density must be finite and positive")

    def charge_residual(mu_e: float) -> float:
        terms_e = _fermi_terms(mu_e, MASS_MEV["e"])
        terms_mu = _fermi_terms(mu_e, MASS_MEV["mu"])
        mu_n = _solve_baryon_mu(n_b, mu_e)
        if mu_n is None:
            # The baryon minimum has already exceeded n_b; this side of the
            # bracket cannot satisfy charge neutrality and must be negative.
            return -(terms_e[0] + terms_mu[0] + n_b)
        n_p = density_from_mu(mu_n - mu_e, MASS_MEV["p"])
        return n_p - terms_e[0] - terms_mu[0]

    lo = MASS_MEV["e"]
    hi = max(inverse_density(n_b, MASS_MEV["e"]), MASS_MEV["e"] + 1.0) + 1000.0
    f_lo = charge_residual(lo)
    f_hi = charge_residual(hi)
    while f_lo * f_hi > 0.0 and hi < 20000.0:
        hi += 1000.0
        f_hi = charge_residual(hi)
    if f_lo * f_hi > 0.0:
        raise RuntimeError(f"charge-neutral bracket failed at n_b={n_b:g}")
    mu_e = float(brentq(charge_residual, lo, hi, xtol=1.0e-11, rtol=1.0e-12, maxiter=200))
    mu_n = _solve_baryon_mu(n_b, mu_e)
    if mu_n is None:
        raise RuntimeError("composition root landed outside baryon domain")
    mu_p = mu_n - mu_e
    terms = {
        "n": _fermi_terms(mu_n, MASS_MEV["n"]),
        "p": _fermi_terms(mu_p, MASS_MEV["p"]),
        "Lambda": _fermi_terms(mu_n, MASS_MEV["Lambda"]),
        "e": _fermi_terms(mu_e, MASS_MEV["e"]),
        "mu": _fermi_terms(mu_e, MASS_MEV["mu"]),
    }
    densities = {name: float(values[0]) for name, values in terms.items()}
    energies = {name: float(values[1]) for name, values in terms.items()}
    pressures = {name: float(values[2]) for name, values in terms.items()}
    fermi_momenta = {name: float(values[3]) for name, values in terms.items()}
    epsilon = float(sum(energies.values()))
    pressure = float(sum(pressures.values()))
    fractions = {name: densities[name] / n_b for name in densities}
    charge_resid = densities["p"] - densities["e"] - densities["mu"]
    beta_resid = mu_n - mu_p - mu_e
    lambda_gap = mu_n - MASS_MEV["Lambda"]
    lambda_resid = 0.0 if densities["Lambda"] > 0.0 else max(lambda_gap, 0.0)
    thermo_pressure = float(sum(mu * densities[name] for name, mu in {
        "n": mu_n,
        "p": mu_p,
        "Lambda": mu_n,
        "e": mu_e,
        "mu": mu_e,
    }.items()) - epsilon)
    return {
        "n_b": float(n_b),
        "mu_n": float(mu_n),
        "mu_p": float(mu_p),
        "mu_Lambda": float(mu_n),
        "mu_e": float(mu_e),
        "mu_mu": float(mu_e),
        "densities": densities,
        "fractions": fractions,
        "energies": energies,
        "pressures": pressures,
        "fermi_momenta": fermi_momenta,
        "epsilon": epsilon,
        "pressure": pressure,
        "thermo_pressure": thermo_pressure,
        "charge_residual": float(charge_resid),
        "beta_residual": float(beta_resid),
        "lambda_gap": float(lambda_gap),
        "lambda_equilibrium_residual": float(lambda_resid),
        "muon_on": bool(densities["mu"] > 0.0),
        "lambda_on": bool(densities["Lambda"] > 0.0),
    }


def _sequence_arrays(states: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    names = ("n", "p", "Lambda", "e", "mu")
    arrays: dict[str, np.ndarray] = {
        "n_b": np.asarray([row["n_b"] for row in states], dtype=float),
        "epsilon": np.asarray([row["epsilon"] for row in states], dtype=float),
        "pressure": np.asarray([row["pressure"] for row in states], dtype=float),
        "thermo_pressure": np.asarray([row["thermo_pressure"] for row in states], dtype=float),
        "mu_e": np.asarray([row["mu_e"] for row in states], dtype=float),
        "mu_n": np.asarray([row["mu_n"] for row in states], dtype=float),
        "charge_residual": np.asarray([row["charge_residual"] for row in states], dtype=float),
        "beta_residual": np.asarray([row["beta_residual"] for row in states], dtype=float),
        "lambda_gap": np.asarray([row["lambda_gap"] for row in states], dtype=float),
        "lambda_equilibrium_residual": np.asarray(
            [row["lambda_equilibrium_residual"] for row in states], dtype=float
        ),
    }
    for name in names:
        arrays[f"y_{name}"] = np.asarray([row["fractions"][name] for row in states], dtype=float)
        arrays[f"n_{name}"] = np.asarray([row["densities"][name] for row in states], dtype=float)
        arrays[f"kf_{name}"] = np.asarray([row["fermi_momenta"][name] for row in states], dtype=float)
    return arrays


def _onset_density(density: np.ndarray, signal: np.ndarray, threshold: float = 0.0) -> float | None:
    values = signal - threshold
    crossing = np.flatnonzero((values[:-1] <= 0.0) & (values[1:] > 0.0))
    if crossing.size == 0:
        return None
    i = int(crossing[0])
    if values[i + 1] == values[i]:
        return float(density[i + 1])
    weight = -values[i] / (values[i + 1] - values[i])
    return float(density[i] + weight * (density[i + 1] - density[i]))


def _triangle_margin(arrays: dict[str, np.ndarray], final: str) -> np.ndarray:
    if final == "e":
        return arrays["kf_p"] + arrays["kf_e"] - arrays["kf_n"]
    if final == "mu":
        return arrays["kf_p"] + arrays["kf_mu"] - arrays["kf_n"]
    if final == "Lambda_e":
        return arrays["kf_p"] + arrays["kf_e"] - arrays["kf_Lambda"]
    if final == "Lambda_mu":
        return arrays["kf_p"] + arrays["kf_mu"] - arrays["kf_Lambda"]
    raise ValueError(f"unknown triangle channel: {final}")


def _relative_max(a: np.ndarray, b: np.ndarray, floor: float = 1.0e-12) -> float:
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)
    return float(np.nanmax(np.abs(a - b) / denom))


def build_reference_sequence(
    grid_points: int = 241, n_min_ratio: float = 0.02, n_max_ratio: float = 10.0
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    if grid_points < 25 or n_min_ratio <= 0.0 or n_max_ratio <= n_min_ratio:
        raise ValueError("invalid density grid configuration")
    density = np.geomspace(n_min_ratio * N0, n_max_ratio * N0, int(grid_points))
    states = [solve_composition(float(value)) for value in density]
    arrays = _sequence_arrays(states)
    return states, arrays


def _tov_from_sequence(arrays: dict[str, np.ndarray], central_density: float) -> tuple[float, float] | None:
    pressure = arrays["pressure"]
    epsilon = arrays["epsilon"]
    order = np.argsort(pressure)
    p = pressure[order]
    e = epsilon[order]
    keep = np.concatenate(([True], np.diff(p) > 0.0))
    p = p[keep]
    e = e[keep]
    if len(p) < 12 or not (np.all(np.isfinite(p)) and np.all(np.isfinite(e))):
        return None
    target_p = float(np.interp(central_density, arrays["n_b"], arrays["pressure"]))
    if not (p[0] < target_p <= p[-1]):
        return None
    eps_of_p = interp1d(p, e, bounds_error=True)
    p_surface = float(p[0])
    r0 = 1.0e-3
    e_c = float(eps_of_p(target_p))
    m0 = 4.0 * math.pi * r0**3 * e_c * MEV_FM3_TO_GEO / 3.0

    def rhs(radius: float, state: np.ndarray) -> list[float]:
        mass_km, pressure_here = float(state[0]), float(state[1])
        if pressure_here <= p_surface:
            return [0.0, 0.0]
        energy_here = float(eps_of_p(pressure_here))
        denominator = radius * (radius - 2.0 * mass_km)
        if denominator <= 0.0:
            return [0.0, 0.0]
        dmdr = 4.0 * math.pi * radius**2 * energy_here * MEV_FM3_TO_GEO
        dpdr = -(energy_here + pressure_here) * MEV_FM3_TO_GEO * (
            mass_km + 4.0 * math.pi * radius**3 * pressure_here * MEV_FM3_TO_GEO
        ) / denominator
        return [dmdr, dpdr]

    def surface(radius: float, state: np.ndarray) -> float:
        return float(state[1] - p_surface)

    surface.terminal = True  # type: ignore[attr-defined]
    surface.direction = -1  # type: ignore[attr-defined]
    try:
        solution = solve_ivp(
            rhs,
            [r0, 100.0],
            [m0, target_p],
            events=surface,
            method="DOP853",
            max_step=0.25,
            rtol=2.0e-7,
            atol=1.0e-10,
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return None
    if solution.t_events[0].size:
        radius = float(solution.t_events[0][0])
        mass_km = float(solution.y_events[0][0][0])
    else:
        radius = float(solution.t[-1])
        mass_km = float(solution.y[0, -1])
    mass = mass_km / M_SUN_KM
    if not (_finite(mass) and _finite(radius) and mass > 0.0 and radius > 0.0):
        return None
    return mass, radius


def _stellar_mapping(arrays: dict[str, np.ndarray], onset_density: float | None) -> dict[str, Any]:
    central_ratios = np.geomspace(0.5, min(9.5, float(arrays["n_b"][-1] / N0)), 32)
    central_density = central_ratios * N0
    masses: list[float] = []
    radii: list[float] = []
    used_density: list[float] = []
    for density in central_density:
        solved = _tov_from_sequence(arrays, float(density))
        if solved is None:
            continue
        mass, radius = solved
        masses.append(mass)
        radii.append(radius)
        used_density.append(float(density))
    if len(masses) < 6:
        return {
            "status": "BLOCKED_NO_STABLE_TOV_BRANCH",
            "central_density_ratio": [],
            "mass_msun": [],
            "radius_km": [],
            "stable_count": 0,
            "onset_mass_msun": None,
            "onset_density_ratio": None if onset_density is None else onset_density / N0,
        }
    masses_arr = np.asarray(masses)
    density_arr = np.asarray(used_density)
    radii_arr = np.asarray(radii)
    maximum_index = int(np.argmax(masses_arr))
    mass_slope = np.gradient(masses_arr, density_arr)
    # A stable branch must begin at the low-central-density end with
    # dM/d(rho_c)>0.  The free-gas reference has the opposite slope because
    # it has no crust/binding prescription, so its stellar mapping is blocked
    # rather than relabelled as a physical maximum mass.
    if not bool(mass_slope[0] > 0.0):
        return {
            "status": "BLOCKED_NO_STABLE_TOV_BRANCH",
            "central_density_ratio": [float(x / N0) for x in density_arr],
            "mass_msun": [float(x) for x in masses_arr],
            "radius_km": [float(x) for x in radii_arr],
            "stable_count": 0,
            "first_mass_maximum_index": maximum_index,
            "unstable_sequence_max_msun": float(masses_arr[maximum_index]),
            "onset_mass_msun": None,
            "onset_density_ratio": None if onset_density is None else float(onset_density / N0),
            "onset_mapping_status": "BLOCKED_NO_STABLE_TOV_BRANCH",
        }
    stable = np.arange(maximum_index + 1)
    onset_mass = None
    onset_mapping_status = "DERIVED_REFERENCE_NOT_REACHED_ON_GRID"
    if onset_density is not None:
        if onset_density <= float(density_arr[maximum_index]):
            solved = _tov_from_sequence(arrays, onset_density)
            if solved is not None:
                onset_mass = float(solved[0])
                onset_mapping_status = "DERIVED_REFERENCE_STABLE_BRANCH_ONLY"
            else:
                onset_mapping_status = "BLOCKED_TOV_DOMAIN_OR_SOLVER"
        else:
            onset_mapping_status = "BLOCKED_ON_UNSTABLE_BRANCH"
    return {
        "status": "DERIVED_REFERENCE_STABLE_BRANCH_ONLY",
        "central_density_ratio": [float(x / N0) for x in density_arr],
        "mass_msun": [float(x) for x in masses_arr],
        "radius_km": [float(x) for x in radii_arr],
        "stable_count": int(len(stable)),
        "first_mass_maximum_index": maximum_index,
        "mmax_msun": float(masses_arr[maximum_index]),
        "onset_mass_msun": onset_mass,
        "onset_density_ratio": None if onset_density is None else float(onset_density / N0),
        "onset_mapping_status": onset_mapping_status,
    }


def _urca_summary(arrays: dict[str, np.ndarray], mapping: dict[str, Any]) -> dict[str, Any]:
    channels: dict[str, dict[str, Any]] = {}
    for channel in ("e", "mu", "Lambda_e", "Lambda_mu"):
        margin = _triangle_margin(arrays, channel)
        if channel == "mu":
            eligible = arrays["n_mu"] > 0.0
        elif channel == "Lambda_mu":
            eligible = (arrays["n_Lambda"] > 0.0) & (arrays["n_mu"] > 0.0)
        elif channel == "Lambda_e":
            eligible = arrays["n_Lambda"] > 0.0
        else:
            eligible = np.ones_like(margin, dtype=bool)
        masked = np.where(eligible, margin, np.nan)
        crossing = np.flatnonzero(eligible[:-1] & eligible[1:] & (margin[:-1] <= 0.0) & (margin[1:] > 0.0))
        onset = None
        if crossing.size:
            i = int(crossing[0])
            weight = -margin[i] / (margin[i + 1] - margin[i])
            onset = float(arrays["n_b"][i] + weight * (arrays["n_b"][i + 1] - arrays["n_b"][i]))
        else:
            # A channel can become eligible at a species onset.  In that
            # case there is no sign-changing pair because the preceding row
            # is intentionally masked; retain the first eligible positive
            # row as a grid-resolved kinematic onset.
            opening = np.flatnonzero(eligible & (margin >= 0.0))
            if opening.size:
                onset = float(arrays["n_b"][int(opening[0])])
        physical_status = (
            "BLOCKED_MISSING_INTERACTION_AND_EMISSIVITY_DEPENDENCIES"
            if channel.startswith("Lambda")
            else "DERIVED_FREE_GAS_KINEMATIC_REFERENCE"
        )
        channels[channel] = {
            "triangle_margin_max_mev": float(np.nanmax(masked)) if np.any(np.isfinite(masked)) else None,
            "triangle_margin_min_mev": float(np.nanmin(masked)) if np.any(np.isfinite(masked)) else None,
            "onset_density_ratio": None if onset is None else float(onset / N0),
            "onset_mass_msun": None,
            "status": physical_status,
        }
        if onset is not None:
            if onset <= float(mapping["central_density_ratio"][mapping["first_mass_maximum_index"]] * N0) if mapping.get("central_density_ratio") else False:
                solved = _tov_from_sequence(arrays, onset)
                if solved is not None:
                    channels[channel]["onset_mass_msun"] = float(solved[0])
    # The proton-fraction form is evaluated independently of the momentum sums.
    xp = arrays["y_p"]
    xe = np.divide(arrays["n_e"], arrays["n_p"], out=np.zeros_like(xp), where=arrays["n_p"] > 0.0)
    threshold = 1.0 / (1.0 + (1.0 + np.cbrt(np.maximum(xe, 0.0))) ** 3)
    proton_criterion = xp - threshold
    triangle_e = _triangle_margin(arrays, "e")
    fraction_open = proton_criterion >= 0.0
    triangle_open = triangle_e >= 0.0
    channels["nucleonic_fraction_identity"] = {
        "criterion_max": float(np.nanmax(proton_criterion)),
        "threshold_min": float(np.nanmin(threshold)),
        "boolean_disagreement_count": int(np.count_nonzero(fraction_open != triangle_open)),
        "boolean_identity": bool(np.array_equal(fraction_open, triangle_open)),
        "status": "DERIVED_FREE_GAS_KINEMATIC_REFERENCE",
        "independent_check": "p_Fn <= p_Fp + p_Fe evaluated separately from fraction criterion",
    }
    # Keep the raw independent margin for tests and the artifact.
    channels["nucleonic_fraction_identity"]["triangle_margin_max_mev"] = float(np.nanmax(triangle_e))
    return {
        "source": {
            "triangle_reference": {
                "citation": "Lattimer et al., Phys. Rev. Lett. 66, 2701 (1991)",
                "doi": "10.1103/PhysRevLett.66.2701",
                "url": "https://doi.org/10.1103/PhysRevLett.66.2701",
                "role": "independent direct-Urca Fermi-momentum triangle condition",
            }
        },
        "channels": channels,
        "emissivity_status": "BLOCKED_MISSING_WEAK_MATRIX_ELEMENTS_TEMPERATURE_AND_SUPERFLUID_GAPS",
        "cooling_claim": "BLOCKED_NO_COOLING_CURVE_AUTHORIZED_IN_P2-S2",
    }


def _audit_existing_dependencies() -> dict[str, Any]:
    paths = {
        "hyperon_eos": ROOT / "verification" / "nvg_eos_beta_lambda_hyperon.py",
        "direct_urca": ROOT / "verification" / "nvg_direct_urca.py",
        "hyperon_tov": ROOT / "verification" / "nvg_hyperon_puzzle_tov.py",
        "phase_toy": ROOT / "verification" / "nvg_full_ns_eos.py",
        "nvgbeta": ROOT / "verification" / "nvg_eos_beta_saturated_vector.py",
    }
    inventory: dict[str, Any] = {}
    for key, path in paths.items():
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        inventory[key] = {
            "path": str(path.relative_to(ROOT)) if path.exists() else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            "status": "AUDITED_READ_ONLY" if path.exists() else "BLOCKED_MISSING_SOURCE",
            "contains_arbitrary_coupling_or_branch": bool(
                any(token in text for token in ("x_sigma", "x_omega", "onset =", "P_add", "de Sitter boost"))
            ),
        }
    return {
        "inventory": inventory,
        "dependency_graph": [
            {
                "node": "interacting_hyperon_mean_fields",
                "status": "BLOCKED_MISSING_TRACEABLE_HYPERON_COUPLINGS",
                "evidence": "verification/nvg_eos_beta_lambda_hyperon.py",
                "required": [
                    "g_sigma_Lambda, g_omega_Lambda, g_rho_Lambda or a sourced U_Lambda(n0)",
                    "all hyperon species and density-dependent fields",
                    "thermodynamic rearrangement terms for any density-dependent couplings",
                ],
            },
            {
                "node": "phase_construction",
                "status": "BLOCKED_MISSING_COEXISTENCE_CONSTRUCTION",
                "evidence": [
                    "verification/nvg_full_ns_eos.py",
                    "verification/nvg_hyperon_puzzle_tov.py",
                ],
                "required": [
                    "equal-pressure/equal-chemical-potential Maxwell or Gibbs conditions",
                    "latent heat or mixed-phase prescription and surface-charge treatment",
                    "consistent epsilon(P) table with interface matching",
                ],
            },
            {
                "node": "hyperonic_emissivity",
                "status": "BLOCKED_MISSING_WEAK_MATRIX_ELEMENTS",
                "evidence": "verification/nvg_direct_urca.py",
                "required": [
                    "weak vertices and in-medium dispersion relations",
                    "temperature dependence and superfluid suppression factors",
                ],
            },
            {
                "node": "nvg_hyperon_dependency",
                "status": "BLOCKED_NO_PHYSICAL_NVG_DEPENDENCY",
                "evidence": [
                    "verification/nvg_eos_beta_saturated_vector.py",
                    "Lunacy/runs/predictive-research/phases/phase-1/P1-S7-REPORT.md",
                ],
                "required": [
                    "derived M_Omega(n) or W-field coupling for each hyperon",
                    "recalibration path linking the NVG field to the beta-EOS chemical potentials",
                ],
            },
        ],
        "physical_hyperon_status": "BLOCKED_MISSING_TRACEABLE_COUPLINGS_PHASE_AND_NVG_DEPENDENCY",
        "reference_model_status": "DERIVED_ZERO_INTERACTION_FREE_GAS_REFERENCE_ONLY",
    }


def _canonical_check() -> dict[str, Any]:
    path = ROOT / "verification" / "nvg_ns_predictive_audit_results.json"
    if not path.exists():
        return {"status": "BLOCKED_CANONICAL_ARTIFACT_MISSING", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = payload.get("canonical", {}).get("observables", {})
    checks = {
        name: bool(math.isclose(float(canonical.get(name)), expected, rel_tol=0.0, abs_tol=1.0e-12))
        for name, expected in CANONICAL_REGRESSION.items()
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL_CANONICAL_REGRESSION_DRIFT",
        "checks": checks,
        "values": {name: canonical.get(name) for name in CANONICAL_REGRESSION},
        "path": str(path.relative_to(ROOT)),
        "semantics": "read-only conditional/in-sample Phase-1 input; not independent evidence",
    }


def _source_provenance() -> dict[str, Any]:
    local = [
        "verification/nvg_beta_hyperon_urca_audit.py",
        "verification/nvg_eos_beta_lambda_hyperon.py",
        "verification/nvg_direct_urca.py",
        "verification/nvg_hyperon_puzzle_tov.py",
        "verification/nvg_full_ns_eos.py",
        "verification/nvg_eos_beta_saturated_vector.py",
    ]
    hashes = {}
    for rel in local:
        path = ROOT / rel
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return {
        "external_inputs": {
            "count": 0,
            "status": "NONE_USED",
            "note": "No downloaded tables, observational likelihoods or fitted couplings enter this audit.",
        },
        "equation_references": [
            {
                "citation": "Lattimer et al., Phys. Rev. Lett. 66, 2701 (1991)",
                "doi": "10.1103/PhysRevLett.66.2701",
                "url": "https://doi.org/10.1103/PhysRevLett.66.2701",
                "role": "direct-Urca triangle condition",
            }
        ],
        "local_source_sha256": hashes,
        "configuration": {
            "model": "T=0 ideal n-p-Lambda-e-mu gas; all interaction couplings exactly zero",
            "density_units": "fm^-3",
            "chemical_potential_units": "MeV",
            "pressure_energy_units": "MeV/fm^3",
            "density_grid_ratio": [0.02, 10.0],
            "grid_points_default": 241,
            "root_xtol": 1.0e-11,
            "root_rtol": 1.0e-12,
            "tov_method": "scipy.integrate.solve_ivp:DOP853",
            "tov_rtol": 2.0e-7,
            "tov_atol": 1.0e-10,
        },
    }


def assert_artifact_provenance(result: dict[str, Any]) -> None:
    """Reject an artifact whose declared inputs/configuration drifted."""

    assert result.get("audit") == "P2-S2"
    assert result.get("source_provenance", {}).get("external_inputs", {}).get("count") == 0
    assert result.get("source_provenance", {}).get("configuration", {}).get("model", "").startswith("T=0 ideal")
    assert result.get("model", {}).get("couplings", {}).get("all_interaction_couplings") == 0.0
    assert result.get("canonical_regression", {}).get("status") == "PASS"
    declared = result.get("source_provenance", {}).get("local_source_sha256", {})
    for relative, digest in declared.items():
        path = ROOT / relative
        assert path.exists(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, relative
    triangle = result.get("urca", {}).get("source", {}).get("triangle_reference", {})
    assert triangle.get("doi") == "10.1103/PhysRevLett.66.2701"
    assert triangle.get("url") == "https://doi.org/10.1103/PhysRevLett.66.2701"


def run_audit(quick: bool = False, grid_points: int | None = None) -> dict[str, Any]:
    points = int(grid_points or (121 if quick else 241))
    states, arrays = build_reference_sequence(points)
    pressure_derivative = arrays["n_b"] ** 2 * np.gradient(
        arrays["epsilon"] / arrays["n_b"], arrays["n_b"], edge_order=2
    )
    dpressure_dn = np.gradient(arrays["pressure"], arrays["n_b"], edge_order=2)
    depsilon_dn = np.gradient(arrays["epsilon"], arrays["n_b"], edge_order=2)
    cs2 = dpressure_dn / depsilon_dn
    arrays["pressure_derivative"] = pressure_derivative
    arrays["cs2"] = cs2
    onset_lambda = _onset_density(arrays["n_b"], arrays["lambda_gap"])
    onset_muon = _onset_density(arrays["n_b"], arrays["mu_e"], MASS_MEV["mu"])
    mapping = _stellar_mapping(arrays, onset_lambda)
    urca = _urca_summary(arrays, mapping)
    if quick:
        # The quick path still exercises the stable-branch integrator, but has
        # less central-density sampling for focused semantic tests.
        mapping["sampling"] = "quick"
    pressure_identity = _relative_max(arrays["pressure"], arrays["thermo_pressure"], floor=1.0e-12)
    derivative_identity = _relative_max(arrays["pressure"], arrays["pressure_derivative"], floor=1.0e-10)
    fractions_sum = arrays["y_n"] + arrays["y_p"] + arrays["y_Lambda"]
    lepton_charge = arrays["n_p"] - arrays["n_e"] - arrays["n_mu"]
    controls = {
        "charge_neutrality_max_abs_fm3": float(np.max(np.abs(lepton_charge))),
        "baryon_fraction_max_abs_residual": float(np.max(np.abs(fractions_sum - 1.0))),
        "beta_equilibrium_max_abs_mev": float(np.max(np.abs(arrays["beta_residual"]))),
        "lambda_complementarity_max_abs_mev": float(np.max(np.abs(arrays["lambda_equilibrium_residual"]))),
        "pressure_identity_max_relative": pressure_identity,
        "thermodynamic_derivative_max_relative": derivative_identity,
        "finite_all_arrays": bool(all(np.all(np.isfinite(values)) for values in arrays.values())),
        "pressure_monotone": bool(np.all(np.diff(arrays["pressure"]) > 0.0)),
        "energy_monotone": bool(np.all(np.diff(arrays["epsilon"]) > 0.0)),
        "causal_sound_speed": bool(np.all(np.isfinite(cs2)) and np.nanmin(cs2) >= -2.0e-3 and np.nanmax(cs2) <= 1.0 + 2.0e-3),
        "sound_speed_min": float(np.nanmin(cs2)),
        "sound_speed_max": float(np.nanmax(cs2)),
        "zero_density_pressure_limit": float(states[0]["pressure"]),
    }
    # A second grid is an explicitly labelled numerical sensitivity, not a fit.
    coarse_states, coarse_arrays = build_reference_sequence(max(61, points // 2))
    sample_density = np.geomspace(max(arrays["n_b"][0], coarse_arrays["n_b"][0]), min(arrays["n_b"][-1], coarse_arrays["n_b"][-1]), 24)
    sensitivity = {
        "status": "NUMERICAL_GRID_SENSITIVITY_ONLY_NO_FIT",
        "grid_points": [int(len(coarse_states)), int(len(states))],
        "max_relative_y_p": float(
            np.max(
                np.abs(
                    np.interp(sample_density, arrays["n_b"], arrays["y_p"])
                    - np.interp(sample_density, coarse_arrays["n_b"], coarse_arrays["y_p"])
                )
            )
        ),
        "max_relative_y_Lambda": float(
            np.max(
                np.abs(
                    np.interp(sample_density, arrays["n_b"], arrays["y_Lambda"])
                    - np.interp(sample_density, coarse_arrays["n_b"], coarse_arrays["y_Lambda"])
                )
            )
        ),
        "lambda_onset_ratio_coarse": None
        if _onset_density(coarse_arrays["n_b"], coarse_arrays["lambda_gap"]) is None
        else float(_onset_density(coarse_arrays["n_b"], coarse_arrays["lambda_gap"]) / N0),
        "lambda_onset_ratio_fine": None if onset_lambda is None else float(onset_lambda / N0),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "audit": "P2-S2",
        "status": "COMPLETE_WITH_BLOCKED_PHYSICAL_HYPERON_PHASE_URCA_DEPENDENCIES",
        "artifacts": {
            "result_json": "verification/nvg_beta_hyperon_urca_audit_p2s2_results.json",
            "figure_png": "verification/fig_p2_s2_beta_hyperon_urca_audit.png",
            "report": "Lunacy/runs/predictive-research/phases/phase-2/P2-S2-REPORT.md",
        },
        "model": {
            "name": "zero-interaction cold beta-equilibrated n-p-Lambda-e-mu reference",
            "status": "DERIVED_ZERO_INTERACTION_FREE_GAS_REFERENCE_ONLY",
            "couplings": {"all_interaction_couplings": 0.0},
            "temperature": 0.0,
            "species": ["n", "p", "Lambda", "e", "mu"],
        },
        "composition": {
            "density_ratio": [float(x / N0) for x in arrays["n_b"]],
            "fractions": {
                name: [float(x) for x in arrays[f"y_{name}"]] for name in ("n", "p", "Lambda", "e", "mu")
            },
            "chemical_potentials_mev": {
                "mu_n": [float(x) for x in arrays["mu_n"]],
                "mu_p": [float(x) for x in arrays["mu_n"] - arrays["mu_e"]],
                "mu_Lambda": [float(x) for x in arrays["mu_n"]],
                "mu_e": [float(x) for x in arrays["mu_e"]],
                "mu_mu": [float(x) for x in arrays["mu_e"]],
            },
            "onsets": {
                "muon_density_ratio": None if onset_muon is None else float(onset_muon / N0),
                "Lambda_zero_coupling_reference_density_ratio": None
                if onset_lambda is None
                else float(onset_lambda / N0),
                "physical_hyperon_onset_status": "BLOCKED_MISSING_TRACEABLE_HYPERON_COUPLINGS",
            },
            "status": "DERIVED_REFERENCE_ONLY_PHYSICAL_HYPERON_FRACTIONS_BLOCKED",
        },
        "eos": {
            "density_ratio": [float(x / N0) for x in arrays["n_b"]],
            "epsilon_mev_fm3": [float(x) for x in arrays["epsilon"]],
            "pressure_mev_fm3": [float(x) for x in arrays["pressure"]],
            "sound_speed_squared": [float(x) for x in cs2],
            "controls": controls,
            "status": "DERIVED_REFERENCE_THERMODYNAMICALLY_CONSISTENT_ZERO_INTERACTION",
        },
        "urca": urca,
        "stellar_mapping": mapping,
        "sensitivity": sensitivity,
        "dependency_audit": _audit_existing_dependencies(),
        "canonical_regression": _canonical_check(),
        "source_provenance": _source_provenance(),
        "claims": {
            "physical_hyperon_fractions": "BLOCKED",
            "phase_transition": "BLOCKED",
            "hyperonic_emissivity": "BLOCKED",
            "cooling_curve": "BLOCKED",
            "NVG_hyperon_prediction": "BLOCKED",
            "free_gas_kinematic_rows": "DERIVED_CONDITIONAL_REFERENCE_ONLY",
        },
    }
    return result


def write_artifacts(result: dict[str, Any], result_path: Path = RESULT_PATH, figure_path: Path = FIGURE_PATH) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    density = np.asarray(result["composition"]["density_ratio"])
    fractions = result["composition"]["fractions"]
    plt.close("all")
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 8.0), sharex=True, constrained_layout=True)
    for name, label, colour in (
        ("n", "n", "tab:blue"),
        ("p", "p", "tab:orange"),
        ("Lambda", r"$\Lambda$", "tab:green"),
        ("e", "e", "tab:red"),
        ("mu", r"$\mu$", "tab:purple"),
    ):
        axes[0].plot(density, fractions[name], label=label, color=colour, linewidth=1.8)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("fraction")
    axes[0].set_title("P2-S2 ideal beta-equilibrated reference (not an interacting NVG EOS)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=5, loc="upper left", fontsize=8)
    urca_channels = result["urca"]["channels"]
    # The threshold panel uses the reference channel extrema as horizontal
    # annotations and keeps the physical BLOCK explicit.
    labels = list(urca_channels)
    vals = [urca_channels[key].get("triangle_margin_max_mev") for key in labels]
    vals = [0.0 if value is None else float(value) for value in vals]
    axes[1].bar(np.arange(len(labels)), vals, color=["tab:blue", "tab:orange", "tab:green", "tab:red", "0.5"][: len(labels)])
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    axes[1].set_ylabel(r"max $(p_{F,\mathrm{final}}+p_{F,l}-p_{F,\mathrm{initial}})$ [MeV]")
    axes[1].set_title("Independent triangle controls; hyperonic/emissivity interpretation BLOCKED")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use a smaller grid for a focused smoke run")
    parser.add_argument("--grid-points", type=int, default=None, help="override the density-grid point count")
    parser.add_argument("--no-figure", action="store_true", help="write JSON only")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_audit(quick=args.quick, grid_points=args.grid_points)
    if args.no_figure:
        RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        write_artifacts(result)
    comp = result["composition"]["onsets"]
    print("P2-S2 beta-equilibrium/hyperon/Urca audit")
    print(f"status={result['status']}")
    print(f"reference Lambda onset={comp['Lambda_zero_coupling_reference_density_ratio']}")
    print(f"muon onset={comp['muon_density_ratio']}")
    print(f"thermodynamic derivative max relative={result['eos']['controls']['thermodynamic_derivative_max_relative']:.3e}")
    print(f"physical hyperon status={result['dependency_audit']['physical_hyperon_status']}")
    print(f"phase status={result['claims']['phase_transition']}")
    print(f"emissivity status={result['urca']['emissivity_status']}")
    print(f"canonical regression={result['canonical_regression']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
