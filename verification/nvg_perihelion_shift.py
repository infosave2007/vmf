#!/usr/bin/env python3
"""Strong-field perihelion-correction sensitivity.

The expression below is an explicit toy correction to the measured GR
periastron rate.  The repository contains neither a matched NVG binary solver
nor a timing residual likelihood, so the tiny number is not an observational
null-test result.
"""

from __future__ import annotations

import math
from typing import Any


def calculate_perihelion_correction(m_omega: float) -> tuple[float, float]:
    """Evaluate the declared vacuum-polarization ansatz at a QCD anchor."""

    if m_omega < 0.0:
        raise ValueError("m_omega must be non-negative")
    eps_eff = math.exp(math.log(0.135) * m_omega / 859.0)
    a_orbit_km = 8.8e5
    r_ns_km = 12.0
    ratio = (1.0 - eps_eff) * (r_ns_km / a_orbit_km) ** 2
    omega_dot_gr = 16.89947
    return ratio, omega_dot_gr * ratio


def compute_perihelion() -> dict[str, Any]:
    center = 859.0
    error = 8.0
    ratio, correction = calculate_perihelion_correction(center)
    ratio_lo, correction_lo = calculate_perihelion_correction(center + error)
    ratio_hi, correction_hi = calculate_perihelion_correction(center - error)
    return {
        "m_omega_center_MeV": center,
        "m_omega_error_MeV": error,
        "ratio": ratio,
        "correction_deg_per_yr": correction,
        "ratio_range": (ratio_hi, ratio_lo),
        "correction_range_deg_per_yr": (correction_hi, correction_lo),
        "precision_benchmark": 2.0e-4,
        "evidence_status": "MODEL_SENSITIVITY_ONLY",
        "observed_likelihood": None,
        "limitation": "No matched binary solver or PSR timing likelihood is present.",
    }


def main() -> dict[str, Any]:
    state = compute_perihelion()
    print("=" * 80)
    print(" NVG STRONG-FIELD PERIASTRON-CORRECTION SENSITIVITY")
    print("=" * 80)
    print(f"QCD anchor M_Omega                       : {state['m_omega_center_MeV']:.1f} ± {state['m_omega_error_MeV']:.1f} MeV")
    print(f"Runtime NVG/GR fractional correction      : {state['ratio']:.3e}")
    print(f"Fractional range                          : {state['ratio_range'][0]:.3e}–{state['ratio_range'][1]:.3e}")
    print(f"Absolute correction                        : {state['correction_deg_per_yr']:.3e} deg/yr")
    print(f"Timing precision benchmark                 : {state['precision_benchmark']:.3e}")
    print("Evidence status                            : MODEL_SENSITIVITY_ONLY")
    print("The benchmark comparison is not a timing-likelihood result.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
