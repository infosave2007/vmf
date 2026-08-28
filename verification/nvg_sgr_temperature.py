#!/usr/bin/env python3
"""SGR 1935 thermal-emission consistency calculation.

The XMM-Newton temperature/luminosity values are declared observational inputs.
The heating level, spot size, envelope relation and two core temperatures are
illustrative model inputs, so the derived spot temperature is a consistency
calculation rather than an independently predicted fit.  No thermal-evolution
solver or likelihood is available in this repository.
"""

from __future__ import annotations

import math
from typing import Any


T_OBS_KEV = 0.45
L_OBS_ERG_S = 1.0e34
M_SGR = 1.10
M_HEAVY = 1.60
AGE_YR = 3600.0
M_DU_THRESHOLD = 1.45
T_CORE_MURCA = 1.2e8
T_CORE_DURCA = 1.5e7
R_NS_CM = 12.0e5
R_SPOT_CM = 1.5e5
SIGMA_SB = 5.6704e-5


def compute_sgr_thermal() -> dict[str, Any]:
    """Compute passive/envelope and spot-heating quantities at runtime."""

    t_obs_K = T_OBS_KEV * 1e3 * 11604.5
    f_spot = (R_SPOT_CM / R_NS_CM) ** 2
    t_passive_murca = 1.0e6 * (T_CORE_MURCA / 1e8) ** 0.55
    t_passive_durca = 1.0e6 * (T_CORE_DURCA / 1e8) ** 0.55
    # Explicit consistency input: the adopted heating power is of the same
    # order as the quoted luminosity, not a fit performed by this script.
    l_heat = 1.1e34
    t_heat = (l_heat / (f_spot * 4.0 * math.pi * R_NS_CM**2 * SIGMA_SB)) ** 0.25
    t_light = (t_passive_murca**4 + t_heat**4) ** 0.25
    t_heavy = (t_passive_durca**4 + 0.10 * t_heat**4) ** 0.25
    l_light = f_spot * 4.0 * math.pi * R_NS_CM**2 * SIGMA_SB * t_light**4
    l_heavy = f_spot * 4.0 * math.pi * R_NS_CM**2 * SIGMA_SB * t_heavy**4
    return {
        "T_obs_keV": T_OBS_KEV,
        "T_obs_K": t_obs_K,
        "L_obs_erg_s": L_OBS_ERG_S,
        "M_sgr": M_SGR,
        "M_heavy": M_HEAVY,
        "M_DU_threshold": M_DU_THRESHOLD,
        "age_yr": AGE_YR,
        "f_spot": f_spot,
        "T_light_keV": t_light / (1e3 * 11604.5),
        "T_heavy_keV": t_heavy / (1e3 * 11604.5),
        "L_light_erg_s": l_light,
        "L_heavy_erg_s": l_heavy,
        "temperature_offset_sigma": (t_light / (1e3 * 11604.5) - T_OBS_KEV) / 0.05,
        "evidence_status": "CALIBRATED_CONSISTENCY_ONLY",
        "observed_likelihood": None,
        "limitation": "No magnetar thermal-evolution solver or XMM spectral likelihood is present.",
    }


def calculate_sgr_thermal() -> dict[str, Any]:
    """Compatibility entry point used by the historical CLI."""

    state = compute_sgr_thermal()
    print("=" * 74)
    print("  NVG/VMF SGR 1935 THERMAL CONSISTENCY CALCULATION")
    print("=" * 74)
    print(f"Declared XMM temperature                 : {state['T_obs_keV']:.2f} keV")
    print(f"Declared XMM luminosity                   : {state['L_obs_erg_s']:.2e} erg/s")
    print(f"Model masses (threshold input)            : {state['M_sgr']:.2f}, {state['M_heavy']:.2f} M_sun")
    print(f"Direct-Urca threshold input               : {state['M_DU_threshold']:.2f} M_sun")
    print(f"Spot area fraction                        : {state['f_spot'] * 100.0:.3f}%")
    print(f"Runtime light-model temperature           : {state['T_light_keV']:.3f} keV")
    print(f"Runtime heavy-model temperature           : {state['T_heavy_keV']:.3f} keV")
    print(f"Runtime light-model luminosity             : {state['L_light_erg_s']:.2e} erg/s")
    print(f"Runtime heavy-model luminosity             : {state['L_heavy_erg_s']:.2e} erg/s")
    print(f"Offset from declared temperature          : {state['temperature_offset_sigma']:+.2f} sigma (descriptive)")
    print("Evidence status                           : CALIBRATED_CONSISTENCY_ONLY")
    print("No independent thermal likelihood is available; no observational result is claimed.")
    print("=" * 74)
    return state


if __name__ == "__main__":
    calculate_sgr_thermal()
