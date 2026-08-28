#!/usr/bin/env python3
"""Audit the PBH ladder without overstating it as a continuous spectrum.

The maintained model defines discrete rungs ``M_N = 0.38 * 4**N``.  This
entry point computes those rungs and explicitly reports that occupancy,
abundance, and subsequent accretion are not supplied; therefore no PBH--SMBH
continuity or JWST resolution claim follows from this calculation alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def ladder(cycles: np.ndarray | list[int]) -> np.ndarray:
    verification_dir = Path(__file__).resolve().parent
    if str(verification_dir) not in sys.path:
        sys.path.insert(0, str(verification_dir))
    from nvg_pbh_mass_spectrum import get_pbh_mass

    return np.asarray([get_pbh_mass(int(cycle)) for cycle in cycles], dtype=float)


def compute_results() -> dict:
    cycles = np.arange(-28, 11, dtype=int)
    masses = ladder(cycles)
    if not np.all(np.diff(masses) > 0.0):
        raise RuntimeError("PBH ladder is not strictly increasing")
    rows = [
        {"cycle": int(cycle), "mass_msun": float(mass)}
        for cycle, mass in zip(cycles, masses)
    ]
    return {
        "rows": rows,
        "min_mass_msun": float(masses[0]),
        "max_mass_msun": float(masses[-1]),
        "status": "DISCRETE_LADDER_ONLY_NO_SMBH_CONTINUITY_PROOF",
    }


RESULTS = compute_results()


def main() -> None:
    print("=" * 70)
    print(" NVG PBH MASS LADDER AUDIT")
    print("=" * 70)
    print("Computed maintained rungs (selected cycles):")
    selected = {1, 10, 22, 30, 40, 50, 60, 70, 73, 75}
    for row in RESULTS["rows"]:
        if row["cycle"] in selected:
            print(f"  cycle {row['cycle']:>3d}: M={row['mass_msun']:.6e} M_sun")
    print(f"Mass range: {RESULTS['min_mass_msun']:.3e} .. {RESULTS['max_mass_msun']:.3e} M_sun")
    print(f"STATUS: {RESULTS['status']}")
    print("No abundance, occupancy, accretion, or observational-continuity result is computed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
