#!/usr/bin/env python3
"""Neutron-star composition g-mode forward calculation.

The TOV profile is computed from the canonical EOS.  The composition
buoyancy parameter is only an assumed sensitivity input; no microphysical
composition solver or detector likelihood is present, so the result is a
forecast rather than a confirmed 66-ms observation.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The canonical tidal module owns the EOS/TOV chain used by the other NS
# entry points.  Keep the imported aliases for compatibility with callers.
from nvg_tidal_deformability import EOS, k_conv, M_sun_km, solve_tov_tidal

EVIDENCE_STATUS = "FORECAST_ASSUMED_COMPOSITION_NO_INDEPENDENT_G_MODE_LIKELIHOOD"


def integrate_tov_profiles(eos: EOS, P_center: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate mass, pressure and energy profiles for a central pressure."""

    if not np.isfinite(P_center) or P_center <= 0.0:
        raise ValueError("central pressure must be positive and finite")
    eos.get_eps(P_center)
    dr = 0.01
    r = 1e-6
    m = 0.0
    pressure = float(P_center)
    radii: list[float] = []
    masses: list[float] = []
    pressures: list[float] = []
    energies: list[float] = []

    def derivs(radius: float, mass: float, p_value: float) -> tuple[float, float]:
        if p_value <= 0.0:
            return 0.0, 0.0
        energy = eos.get_eps(p_value)
        ek = energy * k_conv
        pk = p_value * k_conv
        denom = radius * (radius - 2.0 * mass)
        if denom <= 0.0:
            return 0.0, 0.0
        dm = 4.0 * math.pi * radius ** 2 * ek
        dp = -(ek + pk) * (mass + 4.0 * math.pi * radius ** 3 * pk) / denom / k_conv
        return dm, dp

    while pressure > 1e-5 and r < 100.0:
        radii.append(r)
        masses.append(m)
        pressures.append(pressure)
        energies.append(float(eos.get_eps(pressure)))
        dm1, dp1 = derivs(r, m, pressure)
        dm2, dp2 = derivs(r + 0.5 * dr, m + 0.5 * dr * dm1, pressure + 0.5 * dr * dp1)
        dm3, dp3 = derivs(r + 0.5 * dr, m + 0.5 * dr * dm2, pressure + 0.5 * dr * dp2)
        dm4, dp4 = derivs(r + dr, m + dr * dm3, pressure + dr * dp3)
        m += dr * (dm1 + 2.0 * dm2 + 2.0 * dm3 + dm4) / 6.0
        pressure += dr * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4) / 6.0
        r += dr
        if not np.isfinite(pressure) or not np.isfinite(m):
            raise RuntimeError("non-finite TOV profile")
    return np.asarray(radii), np.asarray(masses), np.asarray(pressures), np.asarray(energies)


def trapezoidal_integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Integrate with NumPy's trapezoidal rule across supported versions."""

    integration = getattr(np, "trapezoid", None)
    if integration is None:
        integration = np.trapz
    return float(integration(values, coordinates))


def gmode_period_ms(radii: np.ndarray, masses: np.ndarray, pressures: np.ndarray,
                    energies: np.ndarray, delta_comp: float) -> float:
    """Compute the WKB l=2 period for one assumed composition parameter."""

    if not np.isfinite(delta_comp) or delta_comp <= 0.0:
        raise ValueError("delta_comp must be positive and finite")
    p_k = pressures * k_conv
    e_k = energies * k_conv
    denom = radii * (radii - 2.0 * masses)
    valid = (radii > 0.5) & (pressures > 0.0) & (denom > 0.0)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("insufficient resolved core profile")
    r_core = radii[valid]
    g_local = (masses[valid] + 4.0 * np.pi * r_core ** 3 * p_k[valid]) / denom[valid]
    e_minus_lambda = np.sqrt(np.maximum(1.0 - 2.0 * masses[valid] / r_core, 1e-12))
    n_geom = g_local * e_minus_lambda * np.sqrt(np.maximum((e_k[valid] + p_k[valid]) / p_k[valid] * delta_comp, 0.0))
    c_light = 2.99792e5
    integral = trapezoidal_integral(n_geom / r_core, r_core)
    return float((2.0 * np.pi ** 2 * np.sqrt(6.0)) / (integral * c_light) * 1000.0)


def compute_gmode_state() -> dict[str, Any]:
    eos = EOS(p_match=1.5, Gamma=1.35)
    p_lo, p_hi = 5.0, min(400.0, eos.pressure_max)
    if p_hi <= p_lo:
        raise RuntimeError("canonical EOS central-pressure bracket is empty")
    for _ in range(25):
        p_mid = 0.5 * (p_lo + p_hi)
        mass, _, _, _ = solve_tov_tidal(eos, p_mid)
        if mass < 1.4:
            p_lo = p_mid
        else:
            p_hi = p_mid
    profiles = integrate_tov_profiles(eos, p_mid)
    mass = profiles[1][-1] / M_sun_km
    radius = profiles[0][-1]
    composition_grid = np.array([1.5e-4, 2.0e-4, 2.5e-4])
    periods = np.array([gmode_period_ms(*profiles, value) for value in composition_grid])
    return {
        "central_pressure": float(p_mid),
        "mass_msun": float(mass),
        "radius_km": float(radius),
        "delta_comp": composition_grid,
        "period_ms": periods,
        "status": EVIDENCE_STATUS,
        "solver": "nvg_tidal_deformability.EOS + TOV profile + WKB",
        "missing": "composition microphysics and independent g-mode detector data/likelihood",
    }


def main() -> dict[str, Any]:
    state = compute_gmode_state()
    print("=" * 80)
    print("  NVG NEUTRON-STAR COMPOSITION g-MODE (RUNTIME FORECAST)")
    print("=" * 80)
    print(f"Canonical profile: M={state['mass_msun']:.3f} M_sun, R={state['radius_km']:.3f} km")
    for dc, period in zip(state["delta_comp"], state["period_ms"]):
        print(f"delta_comp={dc:.2e} -> T_g={period:.2f} ms")
    print(f"Status: {state['status']}")
    print(f"Missing: {state['missing']}")
    print("No detector confirmation is inferred from the WKB forecast.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
