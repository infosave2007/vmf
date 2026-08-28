#!/usr/bin/env python3
"""Additional model-chain examples (PBH ladder, echoes, cooling).

This script intentionally separates executable transforms from assumptions.
The PBH masses and echo delay are consumed from maintained calculators; PBH
abundances, echo reflectivity, and cooling fractions remain explicit
sensitivity inputs and are not presented as predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _verification_imports():
    verification_dir = Path(__file__).resolve().parent
    if str(verification_dir) not in sys.path:
        sys.path.insert(0, str(verification_dir))
    from nvg_cooling_dark_matter import simulate_cooling
    from nvg_gw_echoes import calculate_echo_delay
    from nvg_pbh_mass_spectrum import get_pbh_mass

    return simulate_cooling, calculate_echo_delay, get_pbh_mass


def compute_results() -> dict:
    simulate_cooling, calculate_echo_delay, get_pbh_mass = _verification_imports()

    # D. Use the maintained discrete ladder.  There is no abundance model here.
    pbh_rows = [{"cycle": cycle, "mass_msun": get_pbh_mass(cycle)} for cycle in (0, 10)]

    # E. Delay is computed from the Hayward core calculator; reflectivity is a
    # free template parameter and is shown only as a sensitivity grid.
    mass = 65.0  # event mass input for an illustrative template
    delay = float(calculate_echo_delay(mass)["delta_t_echo_s"])
    reflectivities = [0.0, 0.5, 1.0]
    echo_rows = [
        {"echo": n, "reflectivity": r, "delay_s": n * delay, "amplitude": r**n}
        for r in reflectivities
        for n in range(1, 4)
    ]

    # F. Cooling is a sensitivity case, not a mass-threshold inference.
    ages = np.array([10.0, 100.0, 1000.0, 10000.0])
    cooling_curves = {
        fraction: simulate_cooling(1.8, 11.5, fraction) for fraction in (0.0, 1.0)
    }
    return {
        "pbh": {
            "rows": pbh_rows,
            "status": "DISCRETE_LADDER_MODEL_OUTPUT_NO_ABUNDANCE_PREDICTION",
        },
        "echo": {
            "mass_msun": mass,
            "delay_s": delay,
            "reflectivity_grid": reflectivities,
            "rows": echo_rows,
            "status": "COMPUTED_DELAY_SENSITIVITY_NO_DETECTED_ECHO_CLAIM",
        },
        "cooling": {
            "age_years": ages,
            "curves": cooling_curves,
            "status": "SENSITIVITY_ONLY_NO_URCA_THRESHOLD_INFERENCE",
        },
    }


RESULTS = compute_results()


def main() -> None:
    print("=" * 72)
    print("  NVG: PBH LADDER, ECHO DELAY, AND COOLING SENSITIVITIES")
    print("=" * 72)
    pbh = RESULTS["pbh"]
    print("D. PBH mass ladder (maintained model output):")
    for row in pbh["rows"]:
        print(f"   cycle={row['cycle']:>3d}, mass={row['mass_msun']:.6e} M_sun")
    print(f"   status={pbh['status']}")

    echo = RESULTS["echo"]
    print(f"E. Computed delay for mass={echo['mass_msun']:.1f} M_sun: Δt={echo['delay_s']*1e3:.3f} ms")
    for row in echo["rows"]:
        print(
            f"   n={row['echo']}, R={row['reflectivity']:.2f}, "
            f"delay={row['delay_s']*1e3:.3f} ms, amplitude={row['amplitude']:.3f}"
        )
    print(f"   status={echo['status']}")

    cooling = RESULTS["cooling"]
    print("F. Cooling sensitivity (M=1.8 M_sun, R=11.5 km):")
    for fraction, curve in cooling["curves"].items():
        print(f"   DU fraction={fraction:.1f}, T_surface(10 yr)={curve[0]:.3e} K")
    print(f"   status={cooling['status']}")
    print("No DM abundance, echo detection, or cooling-observation fit is asserted.")


if __name__ == "__main__":
    main()
