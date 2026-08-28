#!/usr/bin/env python3
r"""Independent Hartle first-order slow-rotation observables.

This is the Phase-2 P2-S1/P2-S5 producer.  The canonical zero-jump EOS and its
selection record are imported read-only from :mod:`nvg_tidal_deformability`.
The stellar background (TOV) and the first-order frame-dragging equation are
integrated here independently; the retired ``nvg_moment_of_inertia_j0737``
route is deliberately not called.

The solved perturbation is Hartle's :math:`\bar{\omega}=\Omega-\omega`, with
the centre-normalised seed ``bar_omega(0)=1``.  In geometric units (km),

``(r**4*j*bar_omega')' + 4*r**3*j'*bar_omega = 0``

where ``j = exp(-nu/2)*sqrt(1-2m/r)`` and ``nu`` is the logarithm of the
``g_tt`` lapse.  The exterior solution is
``bar_omega = Omega*(1 - 2 I/r**3)``.  Surface matching therefore gives two
independent identities,

``Omega = bar_omega_R + R*bar_omega'_R/3`` and
``I = R**4*bar_omega'_R/(6*Omega)``.

Only first-order observables are solved.  The second-order quadrupole is
explicitly blocked because this repository has no complete Hartle second
order ``m_0,m_2,h_0,h_2,v_2`` equations, gauge choice, and boundary matching
data.  A Yagi--Yunes lookup is never used as a substitute.

Primary equation provenance:

* Hartle (1967), *ApJ* **150**, 1005--1029, DOI 10.1086/149400,
  https://articles.adsabs.harvard.edu/pdf/1967ApJ...150.1005H
* Kramer et al. (2006), *Tests of general relativity from timing the double
  pulsar*, arXiv:astro-ph/0609417,
  https://arxiv.org/abs/astro-ph/0609417 (the J0737A timing target
  ``1.3381(7) M_sun``).

Run from the repository root with ``python3 verification/nvg_hartle_slow_rotation.py``.
The default run writes uniquely named JSON and figure artifacts and the
worker report.  ``--quick`` is for focused tests only and does not replace
the terminal evidence run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_tidal_deformability as tidal


SCHEMA_VERSION = "P2-S5-hartle-slow-rotation-v2"
SOURCE_PATH = HERE / "nvg_hartle_slow_rotation.py"
TEST_PATH = HERE / "test_p2_hartle_slow_rotation.py"
MASS_INPUT_PATH = HERE / "data" / "nvg_hartle_p2s1_j0737a_mass.json"
RESULT_PATH = HERE / "nvg_hartle_slow_rotation_p2s5_results.json"
FIGURE_PATH = HERE / "fig_hartle_slow_rotation_p2s5.png"
REPORT_PATH = ROOT / "Lunacy/runs/predictive-research/phases/phase-2/P2-S5-REPORT.md"
EVIDENCE_PATH = ROOT / "Lunacy/runs/predictive-research/phases/phase-2/evidence/P2-S5-terminal-verification.log"

# Geometric conversion constants.  The canonical TOV producer uses the same
# M_sun geometric length and MeV/fm^3 conversion; retaining the aliases here
# makes every conversion visible at this independent solver boundary.
G_CGS = 6.67430e-8                    # cm^3 g^-1 s^-2
C_CGS = 2.99792458e10                 # cm s^-1
M_SUN_G = 1.98847e33                  # g (conversion only)
M_SUN_KM = float(tidal.M_sun_km)      # G M_sun / c^2 in km, canonical alias
K_CONV = float(tidal.k_conv)          # MeV/fm^3 -> km^-2, canonical alias
KM3_TO_CM3 = 1.0e15

SURFACE_PRESSURE_DEFAULT = 1.0e-4     # MeV/fm^3; same event convention as canonical TOV
DEFAULT_MAX_STEP_KM = 0.05
DEFAULT_RTOL = 1.0e-9
DEFAULT_ATOL = 1.0e-11
CENTRE_SEED_RADIUS_KM = 1.0e-5

# The branch contract is intentionally the same conservative pressure-ordered
# rule frozen by Phase 1.  Local regression slopes suppress tiny solver chatter;
# raw rows are still retained and any material descent is left unresolved.
BRANCH_SLOPE_WINDOW_MIN = 5
BRANCH_SLOPE_THRESHOLD_RELATIVE = 1.0e-3
BRANCH_RAW_MONOTONIC_TOLERANCE_RELATIVE = 1.0e-3
BRANCH_MIN_POINTS = 3

FROZEN_CANONICAL = {
    "M_max_msun": 2.047950740578197,
    "R14_km": 12.550001000000044,
    "Lambda14": 519.4223807918132,
}

EQUATION_PROVENANCE: dict[str, Any] = {
    "status": "PRIMARY_SOURCE_EQUATIONS",
    "frame_dragging": {
        "citation": "Hartle (1967), ApJ 150, 1005-1029",
        "doi": "10.1086/149400",
        "url": "https://articles.adsabs.harvard.edu/pdf/1967ApJ...150.1005H",
        "equation": "(r^4 j d(bar_omega)/dr)' + 4 r^3 j' bar_omega = 0",
        "j": "exp(-nu/2)*sqrt(1-2m/r), with g_tt=-exp(nu)",
        "exterior": "bar_omega = Omega*(1 - 2 I/r^3)",
    },
    "background": {
        "equations": "Tolman-Oppenheimer-Volkoff equations in G=c=1",
        "source": "Hartle (1967), Sec. II; canonical EOS pressure/energy table",
    },
    "j0737a_mass": {
        "citation": "Kramer et al. (2006), Tests of general relativity from timing the double pulsar",
        "identifier": "arXiv:astro-ph/0609417",
        "url": "https://arxiv.org/abs/astro-ph/0609417",
        "release_date": "2006-10-06",
        "value": "M_A = 1.3381(7) M_sun",
    },
}

MODEL_IDENTITY: dict[str, Any] = {
    "model_id": "canonical_zero_jump_hartle_p2s1",
    "status": "CONDITIONAL_IN_SAMPLE_BACKGROUND",
    "canonical": True,
    "independent_evidence": False,
    "evidence_weight": 0.0,
    "quadrupole_status": "BLOCKED",
    "quadrupole_reason": (
        "The repository supplies no complete second-order Hartle equations, gauge "
        "choice, centre data, or vacuum matching conditions; a universal-relation "
        "table cannot replace those dependencies."
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def source_provenance() -> dict[str, Any]:
    """Return hashes/configuration tied to the live producer and its inputs."""

    canonical_paths = {
        "verification/nvg_tidal_deformability.py": HERE / "nvg_tidal_deformability.py",
        "verification/nvg_eos_beta_css_softening.py": HERE / "nvg_eos_beta_css_softening.py",
        "verification/nvg_ns_canonical.py": HERE / "nvg_ns_canonical.py",
    }
    return {
        "producer": "verification/nvg_hartle_slow_rotation.py",
        "source_sha256": _sha256(SOURCE_PATH),
        # Keep explicit aliases for the three local inputs required to
        # reproduce this repair.  ``source_sha256`` is retained for backwards
        # compatibility with the P2-S1 artifact contract.
        "producer_source_sha256": _sha256(SOURCE_PATH),
        "test_path": "verification/test_p2_hartle_slow_rotation.py",
        "test_sha256": _sha256(TEST_PATH),
        "test_source_sha256": _sha256(TEST_PATH),
        "canonical_sources_sha256": {
            name: _sha256(path) for name, path in canonical_paths.items()
        },
        "input_path": "verification/data/nvg_hartle_p2s1_j0737a_mass.json",
        "input_sha256": _sha256(MASS_INPUT_PATH),
        "configuration": {
            "units": {
                "radius": "km",
                "mass": "M_sun for TOV state; km for geometric mass",
                "pressure_energy": "MeV/fm^3 and km^-2 internally",
                "inertia": "km^3 internally; g cm^2 at output",
            },
            "surface_pressure": SURFACE_PRESSURE_DEFAULT,
            "ode_backend": "scipy.integrate.solve_ivp",
            "ode_method": "DOP853",
            "ode_default_rtol": DEFAULT_RTOL,
            "ode_default_atol": DEFAULT_ATOL,
            "ode_default_max_step_km": DEFAULT_MAX_STEP_KM,
            "central_seed_radius_km": CENTRE_SEED_RADIUS_KM,
            "density_jump": "zero-jump canonical EOS; no interface jump applied",
            "branch_contract": {
                "pressure_order_authoritative": True,
                "slope_window_min": BRANCH_SLOPE_WINDOW_MIN,
                "slope_threshold_relative": BRANCH_SLOPE_THRESHOLD_RELATIVE,
                "raw_negative_drop_tolerance_relative": BRANCH_RAW_MONOTONIC_TOLERANCE_RELATIVE,
                "minimum_branch_points": BRANCH_MIN_POINTS,
                "mass_sorting": False,
                "disconnected_bridging": False,
            },
        },
    }


def load_j0737a_input() -> dict[str, Any]:
    """Load and validate the pinned primary-source timing target."""

    payload = json.loads(MASS_INPUT_PATH.read_text(encoding="utf-8"))
    required = ("object", "mass_msun", "mass_sigma_msun", "primary_source",
                "retrieved_utc", "configuration")
    if any(key not in payload for key in required):
        raise ValueError("J0737A input is missing required provenance fields")
    mass = float(payload["mass_msun"])
    sigma = float(payload["mass_sigma_msun"])
    if not (np.isfinite(mass) and mass > 0.0 and np.isfinite(sigma) and sigma > 0.0):
        raise ValueError("J0737A mass input must be positive and finite")
    source = payload["primary_source"]
    for key in ("citation", "identifier", "url", "release_date"):
        if not source.get(key):
            raise ValueError("J0737A primary-source provenance is incomplete")
    if payload["configuration"].get("independent_evidence") is not False:
        raise ValueError("J0737A target must remain non-independent in this step")
    payload["local_sha256"] = _sha256(MASS_INPUT_PATH)
    return payload


@dataclass
class HartleStar:
    """One TOV + first-order Hartle solution in canonical units."""

    central_pressure: float
    surface_pressure: float
    radius_km: float
    mass_msun: float
    compactness: float
    inertia_geom_km3: float
    inertia_cgs: float
    inertia_bar: float
    omega_surface: float
    domega_surface: float
    omega_infinity: float
    angular_momentum_geom_km2: float
    inertia_integral_geom_km3: float
    profile: dict[str, np.ndarray]
    diagnostics: dict[str, Any]

    def as_dict(self, *, include_profile: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "central_pressure": float(self.central_pressure),
            "surface_pressure": float(self.surface_pressure),
            "radius_km": float(self.radius_km),
            "mass_msun": float(self.mass_msun),
            "compactness": float(self.compactness),
            "inertia_geom_km3": float(self.inertia_geom_km3),
            "inertia_cgs_g_cm2": float(self.inertia_cgs),
            "inertia_bar": float(self.inertia_bar),
            "omega_surface": float(self.omega_surface),
            "domega_surface_per_km": float(self.domega_surface),
            "omega_infinity": float(self.omega_infinity),
            "angular_momentum_geom_km2": float(self.angular_momentum_geom_km2),
            "inertia_integral_geom_km3": float(self.inertia_integral_geom_km3),
            "diagnostics": _jsonable(self.diagnostics),
        }
        if include_profile:
            result["profile"] = _jsonable(self.profile)
        return result


def inertia_geom_to_cgs(inertia_geom_km3: float) -> float:
    """Convert geometric ``I`` [km^3] to cgs [g cm^2]."""

    value = float(inertia_geom_km3)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("geometric inertia must be positive and finite")
    return value * KM3_TO_CM3 * C_CGS**2 / G_CGS


def inertia_cgs_to_geom(inertia_cgs: float) -> float:
    """Convert cgs ``I`` [g cm^2] to geometric [km^3]."""

    value = float(inertia_cgs)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("cgs inertia must be positive and finite")
    return value * G_CGS / C_CGS**2 / KM3_TO_CM3


def _central_seed(eos: tidal.EOS, central_pressure: float, radius_km: float) -> tuple[float, float, float, float, float]:
    """Regular TOV/Hartle series at a small non-zero radius."""

    pressure = float(central_pressure)
    eps_geo = float(eos.get_eps(pressure)) * K_CONV
    pressure_geo = pressure * K_CONV
    radius = float(radius_km)
    mass_msun = 4.0 * math.pi * eps_geo * radius**3 / (3.0 * M_SUN_KM)
    pressure_seed = pressure - (
        (2.0 * math.pi / 3.0)
        * (eps_geo + pressure_geo)
        * (eps_geo + 3.0 * pressure_geo)
        * radius**2 / K_CONV
    )
    lapse_seed = 0.0
    source = eps_geo + pressure_geo
    omega_seed = 1.0 + (8.0 * math.pi / 5.0) * source * radius**2
    domega_seed = (16.0 * math.pi / 5.0) * source * radius
    return mass_msun, pressure_seed, lapse_seed, omega_seed, domega_seed


def integrate_star(
    central_pressure: float,
    *,
    eos: tidal.EOS | None = None,
    surface_pressure: float = SURFACE_PRESSURE_DEFAULT,
    max_step: float = DEFAULT_MAX_STEP_KM,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    central_seed_radius: float = CENTRE_SEED_RADIUS_KM,
) -> HartleStar:
    """Integrate one canonical TOV background and Hartle frame-dragging ODE.

    The state is ``(m[M_sun], P[MeV/fm^3], nu, bar_omega, bar_omega')``.
    ``nu`` is shifted after integration so that ``exp(nu_R)=1-2M/R`` before
    constructing ``j``.  No tidal or legacy moment-of-inertia solver is used.
    """

    if eos is None:
        eos = tidal.EOS()
    Pc = float(central_pressure)
    Ps = float(surface_pressure)
    hmax = float(max_step)
    rtol = float(rtol)
    atol = float(atol)
    r0 = float(central_seed_radius)
    if not np.isfinite(Pc) or Pc <= 0.0:
        raise ValueError("central pressure must be positive and finite")
    if not np.isfinite(Ps) or Ps <= 0.0 or Ps >= Pc:
        raise ValueError("surface pressure must be positive and below central pressure")
    if not np.isfinite(hmax) or hmax <= 0.0:
        raise ValueError("max_step must be positive and finite")
    if not np.isfinite(rtol) or rtol <= 0.0 or not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("ODE tolerances must be positive and finite")
    if not np.isfinite(r0) or r0 <= 0.0 or r0 >= 0.1:
        raise ValueError("central seed radius must lie in (0, 0.1) km")
    # Domain check before entering solve_ivp avoids hidden endpoint clamping.
    eos.get_eps(Pc)

    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover - requirements include scipy
        raise RuntimeError("Hartle integration requires scipy") from exc

    def rhs(radius: float, state: np.ndarray) -> np.ndarray:
        m_msun, pressure, lapse, omega, domega = (float(value) for value in state)
        if radius <= 1.0e-12 or pressure <= 0.0:
            return np.zeros(5, dtype=float)
        eps = float(eos.get_eps(pressure))
        eps_geo = eps * K_CONV
        pressure_geo = pressure * K_CONV
        mass_geo = m_msun * M_SUN_KM
        denominator = radius * (radius - 2.0 * mass_geo)
        if denominator <= 0.0 or not np.isfinite(denominator):
            return np.zeros(5, dtype=float)
        dm = 4.0 * math.pi * radius**2 * eps_geo / M_SUN_KM
        dpressure = -(
            (eps_geo + pressure_geo)
            * (mass_geo + 4.0 * math.pi * radius**3 * pressure_geo)
            / denominator / K_CONV
        )
        dlapse = 2.0 * (
            mass_geo + 4.0 * math.pi * radius**3 * pressure_geo
        ) / denominator
        # j'/j = -4*pi*r*(epsilon+p)/(1-2m/r).  Expanding Hartle's
        # divergence form yields the two terms below and is numerically regular
        # when combined with the centre seed.
        one_minus_2m = 1.0 - 2.0 * mass_geo / radius
        source = eps_geo + pressure_geo
        drag = 4.0 * math.pi * radius * source / one_minus_2m
        domega2 = (drag - 4.0 / radius) * domega + (4.0 * drag / radius) * omega
        return np.asarray([dm, dpressure, dlapse, domega, domega2], dtype=float)

    initial = np.asarray(_central_seed(eos, Pc, r0), dtype=float)

    def surface_event(radius: float, state: np.ndarray) -> float:
        del radius
        return float(state[1]) - Ps

    surface_event.terminal = True
    surface_event.direction = -1

    solution = solve_ivp(
        rhs,
        (r0, 100.0),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=hmax,
        events=surface_event,
    )
    if not solution.success:
        raise RuntimeError(f"Hartle/TOV integration failed: {solution.message}")
    if solution.t_events[0].size == 0:
        raise RuntimeError("Hartle/TOV surface event was not located")

    radii = np.asarray(solution.t, dtype=float)
    masses = np.asarray(solution.y[0], dtype=float)
    pressures = np.asarray(solution.y[1], dtype=float)
    lapse_raw = np.asarray(solution.y[2], dtype=float)
    omega = np.asarray(solution.y[3], dtype=float)
    domega = np.asarray(solution.y[4], dtype=float)
    radius = float(radii[-1])
    mass = float(masses[-1])
    if not (radius > 0.0 and mass > 0.0):
        raise RuntimeError("Hartle/TOV solution has non-positive surface M or R")
    compactness = mass * M_SUN_KM / radius
    if not np.isfinite(compactness) or compactness <= 0.0 or compactness >= 0.5:
        raise RuntimeError("Hartle/TOV surface is outside the regular compactness domain")

    # Match the lapse to Schwarzschild before forming j.  The centre lapse
    # integration constant is arbitrary and must not leak into the integral.
    exterior_lapse = math.log(1.0 - 2.0 * compactness)
    lapse = lapse_raw + (exterior_lapse - lapse_raw[-1])
    one_minus_2m = 1.0 - 2.0 * masses * M_SUN_KM / radii
    if np.any(one_minus_2m <= 0.0):
        raise RuntimeError("background crossed an apparent horizon")
    j = np.exp(-0.5 * lapse) * np.sqrt(one_minus_2m)

    omega_surface = float(omega[-1])
    domega_surface = float(domega[-1])
    omega_infinity = omega_surface + radius * domega_surface / 3.0
    if not np.isfinite(omega_infinity) or omega_infinity <= 0.0:
        raise RuntimeError("surface matching produced non-positive Omega")
    q = radius**4 * domega_surface / omega_surface
    inertia_ratio_geom = q / (6.0 + 2.0 * q / radius**3)
    inertia_geom = radius**4 * domega_surface / (6.0 * omega_infinity)
    if not (np.isfinite(inertia_geom) and inertia_geom > 0.0):
        raise RuntimeError("surface matching produced non-positive inertia")

    # Independent integral identity obtained by integrating the divergence
    # form.  This is not used to define I; it is a diagnostic cross-check.
    eps_geo = np.asarray([float(eos.get_eps(value)) * K_CONV for value in pressures])
    pressure_geo = pressures * K_CONV
    integrand = (eps_geo + pressure_geo) * j / one_minus_2m * (omega / omega_infinity) * radii**4
    # ``np.trapz`` is retained for compatibility with the repository's
    # NumPy 1.x runtime (``np.trapezoid`` was introduced later).
    inertia_integral = float((8.0 * math.pi / 3.0) * np.trapz(integrand, radii))

    omega_ext = omega_infinity * (1.0 - 2.0 * inertia_geom / radii[-1]**3)
    domega_ext = 6.0 * inertia_geom * omega_infinity / radii[-1]**4
    match_residual = max(
        abs(omega_surface - omega_ext) / max(abs(omega_surface), 1.0e-30),
        abs(domega_surface - domega_ext) / max(abs(domega_surface), 1.0e-30),
    )
    extraction_residual = abs(inertia_geom - inertia_ratio_geom) / inertia_geom
    integral_residual = abs(inertia_integral - inertia_geom) / inertia_geom

    central_eps_geo = eps_geo[0]
    central_pressure_geo = pressure_geo[0]
    central_source = central_eps_geo + central_pressure_geo
    seed_expected = (16.0 * math.pi / 5.0) * central_source * r0
    centre_regular = {
        "finite": bool(np.all(np.isfinite(solution.y))),
        "omega_positive": bool(omega[0] > 0.0),
        "mass_over_r3_finite": bool(np.isfinite(masses[0] / radii[0]**3)),
        "domega_seed_relative_error": float(abs(domega[0] - seed_expected) / max(abs(seed_expected), 1.0e-30)),
        "regularity_ok": bool(
            np.all(np.isfinite(solution.y))
            and omega[0] > 0.0
            and np.isfinite(masses[0] / radii[0]**3)
            and abs(domega[0] - seed_expected) / max(abs(seed_expected), 1.0e-30) < 5.0e-5
        ),
    }
    monotonic_pressure = bool(np.all(np.diff(pressures) <= 1.0e-9 * max(Pc, 1.0)))
    monotonic_mass = bool(np.all(np.diff(masses) >= -1.0e-12))
    diagnostics: dict[str, Any] = {
        "ode": {
            "backend": "scipy.integrate.solve_ivp",
            "method": "DOP853",
            "rtol": rtol,
            "atol": atol,
            "max_step_km": hmax,
            "steps": int(len(radii)),
        },
        "centre_boundary": centre_regular,
        "surface_boundary": {
            "event_pressure": float(pressures[-1]),
            "target_pressure": Ps,
            "pressure_event_relative_error": float(abs(pressures[-1] - Ps) / Ps),
            "lapse_match": float(abs(lapse[-1] - exterior_lapse)),
            "compactness_lt_half": bool(compactness < 0.5),
            "surface_ok": bool(abs(pressures[-1] - Ps) / Ps < 1.0e-8 and compactness < 0.5),
        },
        "background": {
            "finite": bool(np.all(np.isfinite(solution.y))),
            "pressure_monotonic": monotonic_pressure,
            "mass_monotonic": monotonic_mass,
            "horizon_free": bool(np.all(one_minus_2m > 0.0)),
            "background_ok": bool(np.all(np.isfinite(solution.y)) and monotonic_pressure and monotonic_mass),
        },
        "vacuum_boundary": {
            "j_surface": float(j[-1]),
            "surface_match_relative_residual": float(match_residual),
            "vacuum_solution": "Omega*(1-2I/r^3)",
            "vacuum_ok": bool(match_residual < 1.0e-10),
        },
        "independent_extraction": {
            "inertia_ratio_identity_geom_km3": float(inertia_ratio_geom),
            "surface_identity_relative_residual": float(extraction_residual),
            "integral_identity_geom_km3": float(inertia_integral),
            "integral_identity_relative_residual": float(integral_residual),
            "identity_ok": bool(extraction_residual < 1.0e-10 and integral_residual < 5.0e-3),
        },
        "positivity": {
            "inertia_positive": bool(inertia_geom > 0.0),
            "omega_positive": bool(np.all(omega > 0.0)),
            "finite_profile": bool(np.all(np.isfinite(solution.y))),
            "regularity_ok": centre_regular["regularity_ok"],
            "status": "PASS" if inertia_geom > 0.0 and np.all(omega > 0.0) and centre_regular["regularity_ok"] else "FAIL",
        },
    }
    profile = {
        "radius_km": radii,
        "mass_msun": masses,
        "pressure_mev_fm3": pressures,
        "lapse_log_gtt": lapse,
        "j": j,
        "bar_omega": omega,
        "dbar_omega_dr": domega,
    }
    return HartleStar(
        central_pressure=Pc,
        surface_pressure=Ps,
        radius_km=radius,
        mass_msun=mass,
        compactness=compactness,
        inertia_geom_km3=float(inertia_geom),
        inertia_cgs=inertia_geom_to_cgs(inertia_geom),
        inertia_bar=float(inertia_geom / (mass * M_SUN_KM) ** 3),
        omega_surface=omega_surface,
        domega_surface=domega_surface,
        omega_infinity=omega_infinity,
        angular_momentum_geom_km2=float(inertia_geom * omega_infinity),
        inertia_integral_geom_km3=inertia_integral,
        profile=profile,
        diagnostics=diagnostics,
    )


def _row_field(row: HartleStar | dict[str, Any], name: str) -> float:
    """Read a scalar field from a dataclass row or a test dictionary."""

    if isinstance(row, dict):
        # ``mass`` is the compact name used by the Phase-1 branch probes,
        # while Hartle control rows use ``mass_msun``.
        key = name
        if key not in row and name == "mass_msun":
            key = "mass"
        value = row[key]
    else:
        value = getattr(row, name)
    return float(value)


def _ordered_mass_slopes(
    rows: list[HartleStar] | list[dict[str, Any]],
    window: int | None = None,
) -> np.ndarray:
    """Estimate ordered ``dM/dlog(P_c)`` without reordering rows.

    The local least-squares slope is a topology diagnostic.  It suppresses
    small integration chatter while retaining a sustained sign change at a
    turning point.  Central-pressure order is authoritative throughout.
    """

    count = len(rows)
    if count < 2:
        return np.asarray([], dtype=float)
    pressures = np.asarray([_row_field(row, "central_pressure") for row in rows], dtype=float)
    masses = np.asarray([_row_field(row, "mass_msun") for row in rows], dtype=float)
    if window is None:
        window = max(BRANCH_SLOPE_WINDOW_MIN, 2 * (count // 48) + 1)
    window = int(max(3, min(count if count % 2 else count - 1, window)))
    if window % 2 == 0:
        window -= 1
    if np.any(~np.isfinite(pressures)) or np.any(pressures <= 0.0):
        return np.full(count, np.nan, dtype=float)
    log_pressure = np.log(pressures)
    slopes = np.full(count, np.nan, dtype=float)
    half = window // 2
    for index in range(count):
        lo = max(0, index - half)
        hi = min(count, index + half + 1)
        x = log_pressure[lo:hi]
        y = masses[lo:hi]
        if len(x) >= 2 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            slope = float(np.polyfit(x, y, 1)[0])
            slopes[index] = slope / max(abs(float(masses[index])), 1.0e-6)
    return slopes


def _branch_topology(
    rows: list[HartleStar] | list[dict[str, Any]],
    *,
    slope_threshold_relative: float = BRANCH_SLOPE_THRESHOLD_RELATIVE,
    raw_tolerance_relative: float = BRANCH_RAW_MONOTONIC_TOLERANCE_RELATIVE,
    min_points: int = BRANCH_MIN_POINTS,
) -> dict[str, Any]:
    """Classify pressure-ordered stable branches and every excluded row.

    A branch is a contiguous, sustained-positive local-slope run.  Material
    raw mass descents split the run; the descending rows remain ``branch_id=-1``
    with an explicit reason.  No mass sorting, global argmax, or disconnected
    branch interpolation is performed.
    """

    rows = list(rows)
    count = len(rows)
    if count == 0:
        return {
            "status": "NO_ROWS",
            "pressure_order_ok": False,
            "slope_window": None,
            "slope_threshold_relative": float(slope_threshold_relative),
            "raw_negative_drop_tolerance_relative": float(raw_tolerance_relative),
            "min_branch_points": int(min_points),
            "branches": [],
            "branch_ids": [],
            "rows": [],
            "allowed_row_indices": [],
            "excluded_row_count": 0,
            "unresolved_intervals": [],
        }

    pressure_values = np.asarray([_row_field(row, "central_pressure") for row in rows], dtype=float)
    mass_values = np.asarray([_row_field(row, "mass_msun") for row in rows], dtype=float)
    pressure_order_ok = bool(
        np.all(np.isfinite(pressure_values))
        and np.all(np.diff(pressure_values) > 0.0)
    )
    slope_window = max(BRANCH_SLOPE_WINDOW_MIN, 2 * (count // 48) + 1)
    slope_window = int(max(3, min(count if count % 2 else count - 1, slope_window)))
    if slope_window % 2 == 0:
        slope_window -= 1
    slopes = _ordered_mass_slopes(rows, window=slope_window)
    stable_mask = (
        pressure_order_ok
        & np.isfinite(slopes)
        & (slopes > float(slope_threshold_relative))
        & np.isfinite(mass_values)
    )

    # First split on local slope runs, then split any run at a material raw
    # mass descent.  The row after a descent may start a new branch, but the
    # interpolation code can never cross the excluded interval.
    candidate_segments: list[tuple[int, int]] = []
    unresolved_intervals: list[dict[str, Any]] = []
    run_start: int | None = None
    for index, is_stable in enumerate(stable_mask):
        if bool(is_stable) and run_start is None:
            run_start = index
        closes = run_start is not None and (not bool(is_stable) or index == count - 1)
        if closes:
            run_end = index + 1 if bool(is_stable) and index == count - 1 else index
            segment_start = int(run_start)
            for stop in range(segment_start + 1, run_end):
                previous = max(abs(float(mass_values[stop - 1])), 1.0e-6)
                relative_drop = (float(mass_values[stop]) - float(mass_values[stop - 1])) / previous
                if relative_drop < -float(raw_tolerance_relative):
                    unresolved_intervals.append({
                        "from_row_index": int(stop - 1),
                        "to_row_index": int(stop),
                        "from_central_pressure": float(pressure_values[stop - 1]),
                        "to_central_pressure": float(pressure_values[stop]),
                        "from_mass_msun": float(mass_values[stop - 1]),
                        "to_mass_msun": float(mass_values[stop]),
                        "relative_drop": float(relative_drop),
                        "reason": "DESCENDING_MASS_GAP",
                    })
                    candidate_segments.append((segment_start, stop))
                    segment_start = stop
            candidate_segments.append((segment_start, run_end))
            run_start = None

    # Also record descents that straddle a slope-mask boundary (for example,
    # the row immediately after a local minimum).  They remain explicit gap
    # intervals even when the post-turn row starts a new allowed branch.
    recorded_pairs = {
        (int(item["from_row_index"]), int(item["to_row_index"]))
        for item in unresolved_intervals
    }
    for stop in range(1, count):
        previous = max(abs(float(mass_values[stop - 1])), 1.0e-6)
        relative_drop = (float(mass_values[stop]) - float(mass_values[stop - 1])) / previous
        pair = (int(stop - 1), int(stop))
        if relative_drop < -float(raw_tolerance_relative) and pair not in recorded_pairs:
            unresolved_intervals.append({
                "from_row_index": int(stop - 1),
                "to_row_index": int(stop),
                "from_central_pressure": float(pressure_values[stop - 1]),
                "to_central_pressure": float(pressure_values[stop]),
                "from_mass_msun": float(mass_values[stop - 1]),
                "to_mass_msun": float(mass_values[stop]),
                "relative_drop": float(relative_drop),
                "reason": "DESCENDING_MASS_GAP",
            })

    # Assign IDs only to sufficiently long runs.  IDs follow pressure order,
    # never mass order.
    branch_indices: list[list[int]] = []
    branch_ids = [-1] * count
    for start, end in candidate_segments:
        if end - start < int(min_points):
            continue
        indices = list(range(start, end))
        if not indices:
            continue
        branch_id = len(branch_indices)
        branch_indices.append(indices)
        for row_index in indices:
            branch_ids[row_index] = branch_id

    branch_records: list[dict[str, Any]] = []
    for branch_id, indices in enumerate(branch_indices):
        branch_masses = mass_values[indices]
        branch_pressures = pressure_values[indices]
        relative_drops = np.diff(branch_masses) / np.maximum(np.abs(branch_masses[:-1]), 1.0e-6)
        branch_slopes = slopes[indices]
        following_turn = None
        for row_index in range(indices[-1] + 1, count):
            if (
                not np.isfinite(slopes[row_index])
                or slopes[row_index] <= float(slope_threshold_relative)
                or (
                    row_index > 0
                    and (mass_values[row_index] - mass_values[row_index - 1])
                    / max(abs(mass_values[row_index - 1]), 1.0e-6)
                    < -float(raw_tolerance_relative)
                )
            ):
                following_turn = row_index
                break
        peak_candidates = list(indices)
        if following_turn is not None:
            # A discrete grid may place the first non-positive local slope one
            # sample after the actual maximum.  Record the ordered peak for
            # reporting, while keeping the turning row out of interpolation.
            peak_candidates.append(int(following_turn))
        peak_index = max(peak_candidates, key=lambda index: float(mass_values[index]))
        branch_records.append({
            "id": int(branch_id),
            "status": "ALLOWED_STABLE_BRANCH",
            "row_indices": [int(index) for index in indices],
            "start_row_index": int(indices[0]),
            "end_row_index": int(indices[-1]),
            "start_central_pressure": float(branch_pressures[0]),
            "end_central_pressure": float(branch_pressures[-1]),
            "points": int(len(indices)),
            "mass_min": float(np.min(branch_masses)),
            "mass_max": float(np.max(branch_masses)),
            "peak_row_index": int(peak_index),
            "peak_central_pressure": float(pressure_values[peak_index]),
            "peak_mass": float(mass_values[peak_index]),
            "raw_monotonic_within_tolerance": bool(
                np.all(relative_drops >= -float(raw_tolerance_relative))
            ),
            "strict_mass_monotonic": bool(np.all(relative_drops > 0.0)),
            "max_raw_negative_drop_relative": float(-np.min(relative_drops[relative_drops < 0.0]))
            if np.any(relative_drops < 0.0) else 0.0,
            "slope_relative_min": float(np.nanmin(branch_slopes))
            if np.any(np.isfinite(branch_slopes)) else None,
            "slope_relative_max": float(np.nanmax(branch_slopes))
            if np.any(np.isfinite(branch_slopes)) else None,
            "turning_point_row_index": int(following_turn) if following_turn is not None else None,
            "turning_point_central_pressure": float(pressure_values[following_turn]) if following_turn is not None else None,
            "turning_point_mass_msun": float(mass_values[following_turn]) if following_turn is not None else None,
            "interpolation_gap_crossed": False,
        })

    topology_rows: list[dict[str, Any]] = []
    for index in range(count):
        if branch_ids[index] >= 0:
            status = "ALLOWED_STABLE"
            reason = None
        elif not pressure_order_ok:
            status = "EXCLUDED_UNRESOLVED"
            reason = "NON_INCREASING_OR_NONFINITE_CENTRAL_PRESSURE"
        elif not np.isfinite(slopes[index]):
            status = "EXCLUDED_UNRESOLVED"
            reason = "INSUFFICIENT_FINITE_LOCAL_SLOPE_SUPPORT"
        elif slopes[index] <= float(slope_threshold_relative):
            status = "EXCLUDED_UNRESOLVED"
            material_drop = (
                index > 0
                and np.isfinite(mass_values[index])
                and np.isfinite(mass_values[index - 1])
                and (mass_values[index] - mass_values[index - 1])
                / max(abs(mass_values[index - 1]), 1.0e-6)
                < -float(raw_tolerance_relative)
            )
            reason = "DESCENDING_OR_TURNING_POINT" if not material_drop else "DESCENDING_MASS_GAP"
        else:
            status = "EXCLUDED_UNRESOLVED"
            reason = "SHORT_OR_RAW_NONMONOTONE_POSITIVE_RUN"
        topology_rows.append({
            "row_index": int(index),
            "central_pressure": float(pressure_values[index]),
            "mass_msun": float(mass_values[index]),
            "dM_dlogPc_relative": float(slopes[index]) if np.isfinite(slopes[index]) else None,
            "branch_id": int(branch_ids[index]),
            "status": status,
            "exclusion_reason": reason,
        })

    return {
        "status": "PASS_STABLE_BRANCH_TOPOLOGY" if branch_indices else "BLOCKED_NO_STABLE_BRANCH",
        "pressure_order_ok": pressure_order_ok,
        "slope_window": int(slope_window),
        "slope_threshold_relative": float(slope_threshold_relative),
        "raw_negative_drop_tolerance_relative": float(raw_tolerance_relative),
        "min_branch_points": int(min_points),
        "branches": branch_records,
        "branch_ids": [int(value) for value in branch_ids],
        "rows": topology_rows,
        "allowed_row_indices": [int(index) for indices in branch_indices for index in indices],
        "excluded_row_count": int(sum(value < 0 for value in branch_ids)),
        "unresolved_intervals": unresolved_intervals,
        "mass_sorted": False,
        "disconnected_bridging": False,
    }


def _stable_prefix(rows: list[HartleStar]) -> list[HartleStar]:
    """Return all explicitly allowed branches in pressure order.

    The historical name is retained for callers, but this is no longer a
    global-argmax prefix.  Excluded/descending rows are never returned.
    """

    topology = _branch_topology(rows)
    return [rows[index] for index in topology["allowed_row_indices"]]


def _interpolate_star_detail(
    rows: list[HartleStar],
    target_mass: float,
    *,
    branch_id: int | None = None,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """Interpolate on one pressure-ordered branch, never across an exclusion."""

    target = float(target_mass)
    topology = _branch_topology(rows)
    branch_records = topology["branches"]
    candidates = [
        record for record in branch_records
        if float(record["mass_min"]) <= target <= float(record["mass_max"])
    ]
    if branch_id is not None:
        candidates = [record for record in candidates if int(record["id"]) == int(branch_id)]
    detail: dict[str, Any] = {
        "status": "BLOCKED",
        "target_mass_msun": target,
        "candidate_branch_ids": [int(record["id"]) for record in candidates],
        "selected_branch_id": None,
        "gap_crossed": False,
        "method": "pressure_ordered_single_branch_linear_mass_interpolation",
    }
    if not candidates:
        detail["reason"] = "TARGET_OUTSIDE_ALLOWED_BRANCHES" if branch_id is None else "REQUESTED_BRANCH_DOES_NOT_COVER_TARGET"
        return None, detail
    if len(candidates) != 1:
        detail["reason"] = "AMBIGUOUS_MULTIPLE_ALLOWED_BRANCHES"
        return None, detail
    selected_record = candidates[0]
    selected_id = int(selected_record["id"])
    detail["selected_branch_id"] = selected_id
    indices = [int(index) for index in selected_record["row_indices"]]
    selected = [rows[index] for index in indices]
    masses = np.asarray([_row_field(row, "mass_msun") for row in selected], dtype=float)
    relative_drops = np.diff(masses) / np.maximum(np.abs(masses[:-1]), 1.0e-6)
    if len(selected) < 2 or np.any(relative_drops < -BRANCH_RAW_MONOTONIC_TOLERANCE_RELATIVE):
        detail["reason"] = "SELECTED_BRANCH_RAW_MASS_ORDER_UNRESOLVED"
        return None, detail
    if target < float(masses[0]) or target > float(masses[-1]):
        detail["reason"] = "TARGET_OUTSIDE_STRICT_BRANCH_MASS_RANGE"
        return None, detail
    bracket = int(np.searchsorted(masses, target, side="left"))
    if bracket <= 0:
        bracket = 1
    if bracket >= len(masses):
        bracket = len(masses) - 1
    lo = bracket - 1
    hi = bracket
    detail.update({
        "status": "PASS_SINGLE_STABLE_BRANCH",
        "bracket_row_indices": [int(indices[lo]), int(indices[hi])],
        "bracket_central_pressures": [float(_row_field(selected[lo], "central_pressure")), float(_row_field(selected[hi], "central_pressure"))],
        "bracket_masses_msun": [float(masses[lo]), float(masses[hi])],
    })

    def interp(field: str) -> float:
        values = np.asarray([_row_field(row, field) for row in selected], dtype=float)
        return float(np.interp(target, masses, values))

    radius = interp("radius_km")
    inertia_geom = interp("inertia_geom_km3")
    return {
        "mass_msun": target,
        "radius_km": radius,
        "compactness": target * M_SUN_KM / radius,
        "inertia_geom_km3": inertia_geom,
        "inertia_cgs_g_cm2": inertia_geom_to_cgs(inertia_geom),
        "inertia_bar": inertia_geom / (target * M_SUN_KM) ** 3,
        "branch_id": selected_id,
    }, detail


def _interpolate_star(rows: list[HartleStar], target_mass: float) -> dict[str, float] | None:
    value, _ = _interpolate_star_detail(rows, target_mass)
    return value


def generate_sequence(
    *,
    eos: tidal.EOS | None = None,
    central_pressures: Iterable[float] | None = None,
    max_step: float = DEFAULT_MAX_STEP_KM,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    surface_pressure: float = SURFACE_PRESSURE_DEFAULT,
) -> list[HartleStar]:
    """Generate an ordered canonical sequence, retaining only valid stars."""

    if eos is None:
        eos = tidal.EOS()
    if central_pressures is None:
        # The logarithmic scan resolves the low/intermediate-mass branch; the
        # explicit local refinement resolves the first maximum rather than
        # reporting a grid-edge M_max.
        pressures = np.unique(np.concatenate([
            np.logspace(-0.30, 3.20, 64),
            np.linspace(220.0, 320.0, 17),
        ]))
    else:
        pressures = np.asarray(list(central_pressures), dtype=float)
    if len(pressures) < 4 or np.any(~np.isfinite(pressures)) or np.any(pressures <= 0.0):
        raise ValueError("central-pressure sequence must contain at least four positive finite values")
    rows: list[HartleStar] = []
    for pressure in pressures:
        try:
            rows.append(
                integrate_star(
                    float(pressure), eos=eos, max_step=max_step, rtol=rtol,
                    atol=atol, surface_pressure=surface_pressure,
                )
            )
        except (RuntimeError, ValueError):
            # Invalid high-pressure/post-horizon points are excluded from the
            # sequence, but the pressure order and the exclusion count are
            # retained in the artifact summary.
            continue
    if len(rows) < 4:
        raise RuntimeError("canonical EOS produced fewer than four Hartle stars")
    return rows


def canonical_regression(eos: tidal.EOS | None = None) -> dict[str, Any]:
    """Recompute the frozen Phase-1 regression point through the read-only producer."""

    if eos is None:
        eos = tidal.EOS()
    pressures = np.logspace(-1.0, 3.4, 120)
    raw: list[tuple[float, float, float, float]] = []
    for pressure in pressures:
        mass, radius, k2, lam = tidal.solve_tov_tidal(eos, float(pressure))
        if mass > 0.5 and radius > 5.0 and k2 > 0.0 and lam > 0.0:
            raw.append((float(mass), float(radius), float(k2), float(lam)))
    if not raw:
        raise RuntimeError("canonical tidal producer returned no regression rows")
    peak = int(np.argmax([row[0] for row in raw]))
    stable = sorted(raw[: peak + 1], key=lambda row: row[0])
    masses = np.asarray([row[0] for row in stable], dtype=float)
    radii = np.asarray([row[1] for row in stable], dtype=float)
    lambdas = np.asarray([row[3] for row in stable], dtype=float)
    values = {
        "M_max_msun": float(np.max(masses)),
        "R14_km": float(np.interp(1.4, masses, radii)),
        "Lambda14": float(np.interp(1.4, masses, lambdas)),
    }
    deltas = {key: float(values[key] - FROZEN_CANONICAL[key]) for key in values}
    tolerances = {"M_max_msun": 2.0e-6, "R14_km": 2.0e-6, "Lambda14": 2.0e-6}
    passed = all(abs(deltas[key]) <= tolerances[key] for key in values)
    return {
        "status": "PASS" if passed else "FAIL",
        "solver": "verification/nvg_tidal_deformability.EOS + solve_tov_tidal (read-only regression)",
        "computed": values,
        "frozen": dict(FROZEN_CANONICAL),
        "delta": deltas,
        "tolerance": tolerances,
        "sequence_rows": len(stable),
    }


def _control_row(star: HartleStar) -> dict[str, float]:
    return {
        "central_pressure": float(star.central_pressure),
        "mass_msun": float(star.mass_msun),
        "radius_km": float(star.radius_km),
        "compactness": float(star.compactness),
        "inertia_geom_km3": float(star.inertia_geom_km3),
        "inertia_cgs_g_cm2": float(star.inertia_cgs),
        "inertia_bar": float(star.inertia_bar),
    }


def _relative_spread(rows: list[dict[str, Any]], keys: Iterable[str]) -> float:
    values: list[float] = []
    for key in keys:
        vals = np.asarray([float(row[key]) for row in rows], dtype=float)
        if len(vals) and np.all(np.isfinite(vals)):
            scale = max(abs(float(vals[-1])), 1.0e-30)
            values.append(float(np.max(np.abs(vals - vals[-1])) / scale))
    return float(max(values)) if values else float("nan")


def convergence_controls(eos: tidal.EOS, sequence: list[HartleStar]) -> dict[str, Any]:
    """Run central-pressure, background, ODE and surface-match ladders."""

    target = _interpolate_star(sequence, 1.4)
    if target is None:
        raise RuntimeError("1.4 M_sun target is outside the Hartle sequence")
    masses = np.asarray([row.mass_msun for row in _stable_prefix(sequence)], dtype=float)
    stable = _stable_prefix(sequence)
    target_index = int(np.argmin(np.abs(masses - 1.4)))
    representative_pc = float(stable[target_index].central_pressure)

    # Central-pressure grid refinement: interpolate I at 1.4 M_sun from
    # progressively denser pressure brackets, with no parameter refit.
    central_rows: list[dict[str, Any]] = []
    for count in (12, 24, 48):
        grid = np.geomspace(representative_pc * 0.65, representative_pc * 1.35, count)
        local = generate_sequence(
            eos=eos, central_pressures=grid, max_step=0.10,
            rtol=1.0e-8, atol=1.0e-10,
        )
        value = _interpolate_star(local, 1.4)
        if value is None:
            continue
        central_rows.append({"grid_points": count, **value})
    central_spread = _relative_spread(central_rows, ("radius_km", "inertia_geom_km3", "inertia_bar"))

    background_rows: list[dict[str, Any]] = []
    for step in (0.20, 0.10, 0.05):
        background_rows.append(_control_row(integrate_star(representative_pc, eos=eos, max_step=step)))
    background_spread = _relative_spread(background_rows, ("mass_msun", "radius_km", "inertia_geom_km3"))

    ode_rows: list[dict[str, Any]] = []
    for rtol, atol in ((1.0e-6, 1.0e-8), (1.0e-8, 1.0e-10), (1.0e-10, 1.0e-12)):
        ode_rows.append(_control_row(integrate_star(
            representative_pc, eos=eos, max_step=0.05, rtol=rtol, atol=atol,
        )) | {"rtol": rtol, "atol": atol})
    ode_spread = _relative_spread(ode_rows, ("mass_msun", "radius_km", "inertia_geom_km3"))

    matching_rows: list[dict[str, Any]] = []
    # The low-pressure polytropic continuation contributes a slowly converging
    # crust radius.  Use a local factor-of-two event ladder rather than mixing
    # in an artificial vacuum atmosphere over two decades.
    for pressure in (2.0e-4, 1.0e-4, 5.0e-5):
        matching_rows.append(_control_row(integrate_star(
            representative_pc, eos=eos, surface_pressure=pressure,
            max_step=0.05, rtol=1.0e-9, atol=1.0e-11,
        )) | {"surface_pressure": pressure})
    matching_spread = _relative_spread(matching_rows, ("radius_km", "inertia_geom_km3", "inertia_bar"))

    # Thresholds are declared before looking at the values and are intentionally
    # modest relative to the canonical fixed-step background.
    controls = {
        "central_pressure": {
            "status": "PASS" if np.isfinite(central_spread) and central_spread < 5.0e-3 else "FAIL",
            "target_mass_msun": 1.4,
            "rows": central_rows,
            "max_relative_spread_vs_finest": central_spread,
            "threshold": 5.0e-3,
        },
        "background": {
            "status": "PASS" if np.isfinite(background_spread) and background_spread < 5.0e-3 else "FAIL",
            "rows": background_rows,
            "max_relative_spread_vs_finest": background_spread,
            "threshold": 5.0e-3,
        },
        "ode_tolerances": {
            "status": "PASS" if np.isfinite(ode_spread) and ode_spread < 5.0e-4 else "FAIL",
            "rows": ode_rows,
            "max_relative_spread_vs_finest": ode_spread,
            "threshold": 5.0e-4,
        },
        "matching_radius": {
            "status": "PASS" if np.isfinite(matching_spread) and matching_spread < 1.0e-2 else "FAIL",
            "rows": matching_rows,
            "max_relative_spread_vs_finest": matching_spread,
            "threshold": 1.0e-2,
            "matching_parameter": "surface pressure event (equivalent radius ladder)",
        },
    }
    controls["status"] = "PASS" if all(item["status"] == "PASS" for item in controls.values()) else "FAIL"
    controls["representative_central_pressure"] = representative_pc
    return controls


def constant_density_benchmark(
    *, compactness: float = 0.05, radius_km: float = 10.0,
    rtol: float = 1.0e-11, atol: float = 1.0e-13,
) -> dict[str, Any]:
    """Reproduce the low-compactness constant-density Hartle benchmark.

    The analytic interior-Schwarzschild pressure/lapse profile is used, so this
    benchmark does not depend on the canonical EOS or its sequence.  The
    published weak-field expansion is ``I/(MR^2) = (2/5)(1+6C/7+106C^2/105)``;
    at ``C=0.05`` the O(C^3) remainder is safely below the declared 2% gate.
    """

    C = float(compactness)
    R = float(radius_km)
    if not (np.isfinite(C) and 0.0 < C < 0.2 and np.isfinite(R) and R > 0.0):
        raise ValueError("constant-density benchmark requires 0<C<0.2 and R>0")
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("constant-density benchmark requires scipy") from exc
    M = C * R
    eps = 3.0 * M / (4.0 * math.pi * R**3)
    sqrt_surface = math.sqrt(1.0 - 2.0 * C)

    def profile(radius: float) -> tuple[float, float, float]:
        x = math.sqrt(1.0 - 2.0 * M * radius**2 / R**3)
        pressure = eps * (x - sqrt_surface) / (3.0 * sqrt_surface - x)
        lapse_half = 0.5 * (3.0 * sqrt_surface - x)
        f = 1.0 - 2.0 * M * radius**2 / R**3
        return pressure, lapse_half, f

    r0 = 1.0e-6
    p0, _, _ = profile(0.0)
    source = eps + p0
    omega0 = 1.0 + (8.0 * math.pi / 5.0) * source * r0**2
    domega0 = (16.0 * math.pi / 5.0) * source * r0

    def rhs(radius: float, state: np.ndarray) -> np.ndarray:
        pressure, _, f = profile(radius)
        drag = 4.0 * math.pi * radius * (eps + pressure) / f
        return np.asarray([state[1], (drag - 4.0 / radius) * state[1] + 4.0 * drag / radius * state[0]])

    solution = solve_ivp(
        rhs, (r0, R), np.asarray([omega0, domega0]), method="DOP853",
        rtol=float(rtol), atol=float(atol), max_step=0.01,
    )
    if not solution.success:
        raise RuntimeError("constant-density frame-dragging benchmark failed")
    omega_surface, domega_surface = (float(value) for value in solution.y[:, -1])
    omega_inf = omega_surface + R * domega_surface / 3.0
    inertia = R**4 * domega_surface / (6.0 * omega_inf)
    ratio = inertia / (M * R**2)
    series = 0.4 * (1.0 + 6.0 * C / 7.0 + 106.0 * C**2 / 105.0)
    relative = abs(ratio - series) / series
    return {
        "status": "validated" if relative < 2.0e-2 else "blocked",
        "compactness": C,
        "radius_km": R,
        "inertia_over_MR2": ratio,
        "weak_field_series": series,
        "relative_difference": relative,
        "threshold": 2.0e-2,
        "equation_provenance": EQUATION_PROVENANCE["frame_dragging"],
    }


def predict_j0737a(sequence: list[HartleStar], input_record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the conditional canonical sequence at the measured A mass.

    There is no published direct ``I_A`` measurement in the pinned input.  We
    therefore return a conditional model value for ``I_A`` and explicitly
    block an inverse mass inference; the timing mass itself is never refit.
    """

    mass = float(input_record["mass_msun"])
    prediction, interpolation = _interpolate_star_detail(sequence, mass)
    if prediction is None:
        return {
            "status": "blocked",
            "reason": interpolation.get("reason", "J0737A timing mass is unresolved on the allowed branches"),
            "mass_msun": mass,
            "interpolation": interpolation,
        }
    return {
        "status": "derived_conditional",
        "mass_msun": mass,
        "mass_sigma_msun": float(input_record["mass_sigma_msun"]),
        "predicted_radius_km": prediction["radius_km"],
        "predicted_compactness": prediction["compactness"],
        "predicted_inertia_geom_km3": prediction["inertia_geom_km3"],
        "predicted_inertia_cgs_g_cm2": prediction["inertia_cgs_g_cm2"],
        "predicted_inertia_bar": prediction["inertia_bar"],
        "branch_id": prediction["branch_id"],
        "interpolation": interpolation,
        "gap_crossed": bool(interpolation.get("gap_crossed", True)),
        "mass_prediction": {
            "status": "blocked",
            "reason": "No direct I_A observable is available for an independent inverse mass prediction",
            "inverse_parameter": "M_A(I_A)",
        },
        "interpretation": (
            "Conditional EOS evaluation at the primary timing mass; this is a forecast "
            "for the future I_A measurement, not empirical confirmation or a new mass fit."
        ),
        "source": input_record["primary_source"],
    }


def interpolation_resolution_sensitivity(
    eos: tidal.EOS,
    input_record: dict[str, Any],
    reference_sequence: list[HartleStar],
    *,
    quick: bool = False,
) -> dict[str, Any]:
    """Quantify J0737A interpolation drift across pressure-grid resolutions."""

    terminal_pressures = np.unique(np.concatenate([
        np.logspace(-0.30, 3.20, 64),
        np.linspace(220.0, 320.0, 17),
    ]))
    grids: list[tuple[str, np.ndarray | None]] = [
        ("coarse_28", np.logspace(-0.30, 3.20, 28)),
        ("terminal_81", None),
    ]
    if not quick:
        grids.append(("fine_321", np.logspace(-0.30, 3.20, 321)))
    rows: list[dict[str, Any]] = []
    for label, pressure_grid in grids:
        sequence = reference_sequence if pressure_grid is None else generate_sequence(
            eos=eos, central_pressures=pressure_grid,
        )
        # ``reference_sequence`` is the terminal 81-row run for full mode and
        # the 28-row run in quick mode; label it according to the actual grid.
        if pressure_grid is None and len(sequence) != len(terminal_pressures):
            sequence = generate_sequence(eos=eos, central_pressures=terminal_pressures)
        prediction = predict_j0737a(sequence, input_record)
        topology = _branch_topology(sequence)
        rows.append({
            "label": label,
            "pressure_points": int(len(sequence)),
            "allowed_branch_count": int(len(topology["branches"])),
            "prediction": prediction,
            "status": prediction.get("status"),
            "branch_id": prediction.get("branch_id"),
            "gap_crossed": prediction.get("gap_crossed", True),
        })
    valid = [row for row in rows if row["status"] == "derived_conditional"]
    inertia_values = np.asarray([
        float(row["prediction"]["predicted_inertia_cgs_g_cm2"]) for row in valid
    ], dtype=float)
    radius_values = np.asarray([
        float(row["prediction"]["predicted_radius_km"]) for row in valid
    ], dtype=float)
    if len(valid):
        inertia_spread = float((np.max(inertia_values) - np.min(inertia_values)) / max(abs(float(inertia_values[-1])), 1.0e-30))
        radius_spread = float((np.max(radius_values) - np.min(radius_values)) / max(abs(float(radius_values[-1])), 1.0e-30))
    else:
        inertia_spread = float("nan")
        radius_spread = float("nan")
    branch_ids = {row["branch_id"] for row in valid}
    passed = bool(
        len(valid) == len(rows)
        and len(branch_ids) == 1
        and all(row["gap_crossed"] is False for row in valid)
    )
    return {
        "status": "PASS_BRANCH_LOCAL_RESOLUTION" if passed else "BLOCKED_UNRESOLVED_J0737A_INTERPOLATION",
        "target_mass_msun": float(input_record["mass_msun"]),
        "rows": rows,
        "branch_id_consistency": sorted(int(value) for value in branch_ids) if branch_ids else [],
        "j0737a_inertia_relative_spread": inertia_spread,
        "j0737a_radius_relative_spread": radius_spread,
        "interpretation": "Grid/interpolation sensitivity only; no parameter refit and no empirical I_A claim.",
    }


def build_payload(*, quick: bool = False) -> dict[str, Any]:
    """Compute the structured P2-S5 result without writing files."""

    eos = tidal.EOS()
    input_record = load_j0737a_input()
    if quick:
        sequence_pressures = np.logspace(-0.30, 3.20, 28)
    else:
        sequence_pressures = np.unique(np.concatenate([
            np.logspace(-0.30, 3.20, 64),
            np.linspace(220.0, 320.0, 17),
        ]))
    sequence = generate_sequence(eos=eos, central_pressures=sequence_pressures)
    branch_topology = _branch_topology(sequence)
    stable_indices = [int(index) for index in branch_topology["allowed_row_indices"]]
    stable = [sequence[index] for index in stable_indices]
    target_masses = (1.2, 1.3381, 1.4, 1.6, 1.8, 2.0)
    observable_rows: list[dict[str, Any]] = []
    for mass in target_masses:
        value = _interpolate_star(sequence, mass)
        if value is not None:
            observable_rows.append(value)
    if not observable_rows:
        raise RuntimeError("Hartle sequence did not cover any requested mass")

    # A full terminal run includes every control.  Quick mode intentionally
    # keeps only the benchmark and one direct star for semantic tests.
    if quick:
        representative = integrate_star(100.0, eos=eos)
        controls: dict[str, Any] = {
            "status": "SKIPPED_QUICK_MODE",
            "representative": _control_row(representative),
        }
    else:
        controls = convergence_controls(eos, sequence)

    canonical = canonical_regression(eos)
    benchmark = constant_density_benchmark()
    j0737 = predict_j0737a(sequence, input_record)
    interpolation_sensitivity = interpolation_resolution_sensitivity(
        eos, input_record, sequence, quick=quick,
    )
    second_order = {
        "status": "blocked",
        "reason": MODEL_IDENTITY["quadrupole_reason"],
        "required_dependencies": [
            "Hartle second-order m_0/m_2/h_0/h_2 equations",
            "gauge and centre boundary conditions",
            "vacuum quadrupole matching and spin convention",
        ],
        "substitute_used": False,
    }

    branch_id_by_index = {
        int(record["row_index"]): int(record["branch_id"])
        for record in branch_topology["rows"]
        if int(record["branch_id"]) >= 0
    }
    sequence_rows = [
        _control_row(row) | {
            "row_index": int(index),
            "branch_id": int(branch_id_by_index[index]),
            "status": "ALLOWED_STABLE",
            "exclusion_reason": None,
        }
        for index, row in zip(stable_indices, stable)
    ]
    positivity = all(
        row.diagnostics["positivity"]["status"] == "PASS" for row in stable
    )
    boundaries = {
        "centre": "PASS" if all(row.diagnostics["centre_boundary"]["regularity_ok"] for row in stable) else "FAIL",
        "surface": "PASS" if all(row.diagnostics["surface_boundary"]["surface_ok"] for row in stable) else "FAIL",
        "vacuum": "PASS" if all(row.diagnostics["vacuum_boundary"]["vacuum_ok"] for row in stable) else "FAIL",
        "independent_identities": "PASS" if all(row.diagnostics["independent_extraction"]["identity_ok"] for row in stable) else "FAIL",
        "positivity_regular": "PASS" if positivity else "FAIL",
    }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit": "P2-S5",
        "artifacts": {
            "result_json": str(RESULT_PATH.relative_to(ROOT)),
            "figure_png": str(FIGURE_PATH.relative_to(ROOT)),
        },
        "model_identity": MODEL_IDENTITY,
        "observable_status": "derived_conditional_branch_scoped",
        "public_status": "DERIVED_CONDITIONAL_ALLOWED_BRANCHES_LOW_MASS_UNRESOLVED",
        "equation_provenance": EQUATION_PROVENANCE,
        "canonical_selection": eos.canonical_selection,
        "canonical_provenance": eos.canonical_provenance,
        "source_provenance": source_provenance(),
        "source_to_artifact_provenance": {
            "status": "PASS",
            "source_sha256": _sha256(SOURCE_PATH),
            "producer_source_sha256": _sha256(SOURCE_PATH),
            "test_sha256": _sha256(TEST_PATH),
            "test_source_sha256": _sha256(TEST_PATH),
            "input_sha256": _sha256(MASS_INPUT_PATH),
        },
        "frozen_dependency_boundary": {
            "status": "PASS",
            "background": "canonical zero-jump NS sequence only",
            "M_Omega_propagation": "not used; Phase-1 dependency remains blocked",
            "phase_transition_detection": "not claimed",
        },
        "observational_input": input_record,
        "sequence": sequence_rows,
        "branch_topology": branch_topology,
        "sequence_summary": {
            "raw_rows": len(sequence),
            "stable_rows": len(stable),
            "allowed_branch_count": int(len(branch_topology["branches"])),
            "excluded_unresolved_rows": int(branch_topology["excluded_row_count"]),
            "low_mass_rows_status": (
                "UNRESOLVED_EXCLUDED_GAP" if branch_topology["excluded_row_count"] else "RESOLVED"
            ),
            "mass_max_msun": float(max(record.get("peak_mass", record["mass_max"]) for record in branch_topology["branches"])),
            "targets": observable_rows,
            "sensitivity_without_refit": True,
            "retired_static_i_love_overlay": {
                "status": "not_used",
                "reason": "Existing I-Love-Q arrays are transform/retired overlays, not producers",
            },
        },
        "j0737a": j0737,
        "interpolation_resolution_sensitivity": interpolation_sensitivity,
        "second_order_quadrupole": second_order,
        "boundary_checks": boundaries,
        "convergence": controls,
        "constant_density_benchmark": benchmark,
        "canonical_regression": canonical,
        "classification": {
            "I(M)": "derived_conditional_on_allowed_pressure_ordered_branches",
            "compactness": "derived_conditional_on_allowed_pressure_ordered_branches",
            "dimensionless_Ibar": "derived_conditional_on_allowed_pressure_ordered_branches",
            "J0737A_conditional_I": "derived_conditional",
            "J0737A_inverse_mass": "blocked",
            "constant_density_benchmark": benchmark["status"],
            "quadrupole_Q": "blocked",
            "empirical_confirmation": "not_claimed",
        },
    }
    return _jsonable(payload)


def assert_artifact_provenance(payload: dict[str, Any]) -> None:
    """Fail closed if a result was generated by stale source/input files."""

    source = source_provenance()
    actual = payload.get("source_provenance", {})
    if actual != source:
        raise AssertionError("Hartle source/input provenance drift")
    assertion = payload.get("source_to_artifact_provenance", {})
    if assertion.get("status") != "PASS" or assertion.get("source_sha256") != source["source_sha256"]:
        raise AssertionError("Hartle source-to-artifact assertion is not PASS")
    if assertion.get("producer_source_sha256") != source["producer_source_sha256"]:
        raise AssertionError("Hartle producer source hash drift")
    if assertion.get("test_sha256") != source["test_sha256"]:
        raise AssertionError("Hartle focused-test hash drift")
    if assertion.get("test_source_sha256") != source["test_source_sha256"]:
        raise AssertionError("Hartle focused-test source hash drift")
    if assertion.get("input_sha256") != source["input_sha256"]:
        raise AssertionError("Hartle input hash drift")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("Hartle artifact schema drift")
    artifacts = payload.get("artifacts", {})
    if artifacts.get("result_json") != str(RESULT_PATH.relative_to(ROOT)):
        raise AssertionError("Hartle result artifact path drift")
    if artifacts.get("figure_png") != str(FIGURE_PATH.relative_to(ROOT)):
        raise AssertionError("Hartle figure artifact path drift")
    topology = payload.get("branch_topology", {})
    if topology.get("mass_sorted") is not False or topology.get("disconnected_bridging") is not False:
        raise AssertionError("Hartle branch topology permits forbidden sorting/bridging")
    if topology.get("status") != "PASS_STABLE_BRANCH_TOPOLOGY":
        raise AssertionError("Hartle branch topology is not certified")
    for row in topology.get("rows", []):
        if int(row.get("branch_id", -1)) < 0 and not row.get("exclusion_reason"):
            raise AssertionError("Hartle unresolved row lacks exclusion reason")
    for interval in topology.get("unresolved_intervals", []):
        if interval.get("reason") != "DESCENDING_MASS_GAP":
            raise AssertionError("Hartle unresolved interval lacks descending-gap reason")
    j0737 = payload.get("j0737a", {})
    if j0737.get("status") == "derived_conditional":
        if j0737.get("branch_id") is None or j0737.get("gap_crossed") is not False:
            raise AssertionError("J0737A interpolation is not branch-local")
        sensitivity = payload.get("interpolation_resolution_sensitivity", {})
        if sensitivity.get("status") != "PASS_BRANCH_LOCAL_RESOLUTION":
            raise AssertionError("J0737A interpolation-resolution control is not PASS")


def _write_figure(payload: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["sequence"]
    branch_ids = sorted({int(row["branch_id"]) for row in rows})
    plt.rcParams.update({"font.family": "serif", "axes.linewidth": 1.1})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    colors = plt.cm.Blues(np.linspace(0.55, 0.9, max(1, len(branch_ids))))
    for color, branch_id in zip(colors, branch_ids):
        branch_rows = [row for row in rows if int(row["branch_id"]) == branch_id]
        masses = np.asarray([row["mass_msun"] for row in branch_rows], dtype=float)
        inertia = np.asarray([row["inertia_cgs_g_cm2"] for row in branch_rows], dtype=float) / 1.0e45
        compactness = np.asarray([row["compactness"] for row in branch_rows], dtype=float)
        ibar = np.asarray([row["inertia_bar"] for row in branch_rows], dtype=float)
        axes[0].plot(masses, inertia, "o-", color=color, lw=1.8, ms=4.5, label=f"Hartle $I(M)$ branch {branch_id}")
        axes[1].plot(compactness, ibar, "o-", color=color, lw=1.8, ms=4.5, label=f"branch {branch_id}")
    axes[0].set_xlabel(r"Mass $M\,[M_\odot]$")
    axes[0].set_ylabel(r"$I\,[10^{45}\,\mathrm{g\,cm^2}]$")
    axes[0].grid(alpha=0.2)
    j0737 = payload["j0737a"]
    if j0737.get("status") == "derived_conditional":
        axes[0].scatter([j0737["mass_msun"]], [j0737["predicted_inertia_cgs_g_cm2"] / 1.0e45], marker="D", s=58, color="#c0392b", zorder=4, label="J0737A conditional target")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_xlabel(r"Compactness $C=GM/(Rc^2)$")
    axes[1].set_ylabel(r"Dimensionless $\bar I=I/M^3$")
    axes[1].grid(alpha=0.2)
    if branch_ids:
        axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("P2-S5 Hartle slow-rotation observables (conditional canonical EOS)", fontsize=12)
    fig.text(0.5, 0.01, "Pressure-ordered stable branches only; unresolved/descending gaps omitted; Q blocked; no empirical confirmation.", ha="center", fontsize=8, color="#555555", style="italic")
    fig.subplots_adjust(bottom=0.16, top=0.87, wspace=0.28)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_artifacts(payload: dict[str, Any]) -> None:
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_figure(payload)


def _print_summary(payload: dict[str, Any]) -> None:
    summary = payload["sequence_summary"]
    print("=" * 80)
    print("P2-S5 HARTLE SLOW-ROTATION OBSERVABLES")
    print(f"Status: {payload['public_status']} on canonical zero-jump background")
    print(f"Stable rows: {summary['stable_rows']} / raw rows: {summary['raw_rows']}")
    print(f"Allowed branches: {summary['allowed_branch_count']}; excluded/unresolved rows: {summary['excluded_unresolved_rows']}")
    print(f"M_max (Hartle background) = {summary['mass_max_msun']:.9f} M_sun")
    for row in summary["targets"]:
        print(f"M={row['mass_msun']:.4f}: R={row['radius_km']:.6f} km, C={row['compactness']:.6f}, I={row['inertia_cgs_g_cm2'] / 1e45:.6f}e45 g cm^2, Ibar={row['inertia_bar']:.6f}")
    j0737 = payload["j0737a"]
    if j0737.get("status") == "derived_conditional":
        print(f"J0737A conditional I(M_A=1.3381) = {j0737['predicted_inertia_cgs_g_cm2'] / 1e45:.6f}e45 g cm^2")
    print(f"J0737A inverse mass prediction: {j0737.get('mass_prediction', {}).get('status', 'blocked')}")
    print(f"Second-order Q: {payload['second_order_quadrupole']['status']}")
    print(f"Boundary checks: {payload['boundary_checks']}")
    print(f"Canonical regression: {payload['canonical_regression']['status']}")
    print(f"Artifacts: {RESULT_PATH}; {FIGURE_PATH}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small semantic run; not terminal evidence")
    args = parser.parse_args(argv)
    payload = build_payload(quick=bool(args.quick))
    if not args.quick:
        assert_artifact_provenance(payload)
        write_artifacts(payload)
    _print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
