#!/usr/bin/env python3
"""Runtime Tolman-cycle horizon-chain calculation.

The QCD-anchor density, instanton radius, entropy ratio and derived cycle index
are recomputed from explicit constants.  The current Hubble value and the
per-cycle entropy factor are calibration inputs; no independent cyclic
microphysics or cosmological likelihood is present.
"""

from __future__ import annotations

import math
from typing import Any


G = 6.674e-8
c = 2.998e10
M_sun = 1.989e33
yr_to_s = 3.154e7
l_p = 1.616e-33
M_Omega_0 = 859.0
hbar_c = 197.327
eps_max = M_Omega_0**4 / hbar_c**3
MeV_fm3_to_gcm3 = 1.7827e12
rho_c = eps_max * MeV_fm3_to_gcm3


def compute_cyclic_state(
    *,
    H0_km_s_Mpc: float = 72.8,
    entropy_factor: float = 4.0,
    cycle_index: int = 77,
) -> dict[str, Any]:
    """Compute the horizon-chain quantities from declared calibration inputs."""

    if H0_km_s_Mpc <= 0.0 or entropy_factor <= 1.0 or cycle_index < 1:
        raise ValueError("H0/entropy factor must be positive and cycle_index >= 1")
    H_c = math.sqrt(8.0 * math.pi * G * rho_c / 3.0)
    r_c = c / H_c
    M_genesis = (4.0 / 3.0) * math.pi * r_c**3 * rho_c
    S_genesis = math.pi * r_c**2 / l_p**2
    H0_cgs = H0_km_s_Mpc * 1e5 / 3.086e24
    R_H0 = c / H0_cgs
    S_current = math.pi * R_H0**2 / l_p**2
    n_derived = 1.0 + (math.log10(S_current) - math.log10(S_genesis)) / math.log10(entropy_factor)
    M_turnaround = M_genesis * (2.0 ** (cycle_index - 1))
    lifetime_yr = math.pi * G * M_turnaround / c**3 / yr_to_s
    return {
        "rho_c_g_cm3": float(rho_c),
        "r_c_km": float(r_c / 1e5),
        "M_genesis_g": float(M_genesis),
        "S_genesis_log10": float(math.log10(S_genesis)),
        "S_current_log10": float(math.log10(S_current)),
        "n_derived": float(n_derived),
        "cycle_index": int(cycle_index),
        "M_turnaround_g": float(M_turnaround),
        "lifetime_yr": float(lifetime_yr),
        "H0_km_s_Mpc": float(H0_km_s_Mpc),
        "entropy_factor": float(entropy_factor),
        "evidence_status": "CALIBRATED_HORIZON_CHAIN",
        "observed_likelihood": None,
        "limitation": "The per-cycle growth factor and entropy law are calibration inputs; no microphysical solver/likelihood is present.",
    }


def main() -> dict[str, Any]:
    state = compute_cyclic_state()
    print("=" * 78)
    print("  NVG CYCLIC-COSMOLOGY HORIZON-CHAIN CALCULATION")
    print("=" * 78)
    print(f"QCD-anchor density                         : {state['rho_c_g_cm3']:.4e} g/cm^3")
    print(f"Runtime instanton radius                   : {state['r_c_km']:.4f} km")
    print(f"Genesis mass                               : {state['M_genesis_g']:.4e} g")
    print(f"Genesis log10 entropy                      : {state['S_genesis_log10']:.4f}")
    print(f"Current-horizon log10 entropy              : {state['S_current_log10']:.4f}")
    print(f"Derived cycle index                        : {state['n_derived']:.2f} (H0/entropy inputs)")
    print(f"Turnaround mass at cycle {state['cycle_index']}             : {state['M_turnaround_g']:.4e} g")
    print(f"Turnaround lifetime                        : {state['lifetime_yr']:.2e} yr")
    print("Evidence status                            : CALIBRATED_HORIZON_CHAIN")
    print("No independent cyclic-cosmology likelihood or microphysical growth solver is evaluated.")
    print("=" * 78)
    return state


if __name__ == "__main__":
    main()
