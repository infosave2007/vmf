#!/usr/bin/env python3
"""Cooling and PBH model-chain surfaces with honest provenance.

Cooling fractions require a stellar density profile that this simplified
integrator does not compute.  They are therefore exposed as a sensitivity
parameter, never as fitted VMF fractions.  The PBH values are delegated to the
maintained discrete ladder and are not promoted to an abundance prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


YEARS_TO_SECONDS = 3.154e7
TIME_YEARS = np.logspace(0, 6, 80)


def simulate_cooling(mass: float, radius_km: float, direct_urca_fraction: float = 0.0) -> np.ndarray:
    """Integrate the pedagogical cooling ODE for an explicit sensitivity case."""
    if mass <= 0.0 or radius_km <= 0.0 or not 0.0 <= direct_urca_fraction <= 1.0:
        raise ValueError("mass/radius must be positive and DU fraction in [0, 1]")
    times = TIME_YEARS * YEARS_TO_SECONDS
    t9 = 1.0
    history = []
    now = 0.0
    for target in times:
        while now < target and t9 > 1.0e-8:
            l_mu = 1.0e21 * mass * t9**8
            l_du = 1.0e27 * mass * direct_urca_fraction * t9**6
            t_surface = 1.0e6 * (t9 * 10.0) ** 0.55
            l_gamma = 4.0 * np.pi * (radius_km * 1.0e5) ** 2 * 5.67e-5 * t_surface**4
            heat_capacity = 1.0e39 * mass * t9
            derivative = -(l_mu + l_du + l_gamma) / (heat_capacity * 1.0e9)
            if derivative == 0.0:
                now = target
                break
            step = min(times[0], abs(0.01 * t9 / derivative), target - now)
            t9 = max(1.0e-8, t9 + derivative * step)
            now += step
        history.append(1.0e6 * (t9 * 10.0) ** 0.55)
    return np.asarray(history)


def pbh_ladder(cycles: list[int]) -> list[dict]:
    verification_dir = Path(__file__).resolve().parent
    if str(verification_dir) not in sys.path:
        sys.path.insert(0, str(verification_dir))
    from nvg_pbh_mass_spectrum import get_pbh_mass

    return [{"cycle": int(cycle), "mass_msun": float(get_pbh_mass(int(cycle)))} for cycle in cycles]


def compute_results() -> dict:
    # DU fractions are a sensitivity grid: no density profile is inferred here.
    fractions = [0.0, 0.5, 1.0]
    cooling = {
        fraction: simulate_cooling(1.8, 11.5, fraction) for fraction in fractions
    }
    ladder = pbh_ladder([-21, 0, 10])
    return {
        "cooling": {
            "time_years": TIME_YEARS,
            "fraction_grid": fractions,
            "curves": cooling,
            "status": "SENSITIVITY_ONLY_FRACTIONS_NOT_INFERRED",
        },
        "pbh": {
            "rows": ladder,
            "status": "DISCRETE_LADDER_MODEL_OUTPUT_NO_ABUNDANCE_PREDICTION",
        },
    }


RESULTS = compute_results()


def main() -> None:
    print("=" * 72)
    print("  NVG: COOLING SENSITIVITY AND PBH LADDER")
    print("=" * 72)
    cooling = RESULTS["cooling"]
    print("Y. Cooling ODE sensitivity for M=1.8 M_sun, R=11.5 km:")
    for fraction in cooling["fraction_grid"]:
        curve = cooling["curves"][fraction]
        print(
            f"   DU fraction={fraction:.2f}: "
            f"T_surface(1 yr)={curve[0]:.3e} K, "
            f"T_surface(1 Myr)={curve[-1]:.3e} K"
        )
    print(f"   status={cooling['status']}")

    pbh = RESULTS["pbh"]
    print("Z. Discrete PBH ladder values from nvg_pbh_mass_spectrum:")
    for row in pbh["rows"]:
        print(f"   cycle={row['cycle']:>3d}, mass={row['mass_msun']:.6e} M_sun")
    print(f"   status={pbh['status']}")
    print("No cooling observation, DM fraction, or lensing compatibility is claimed.")


if __name__ == "__main__":
    main()
