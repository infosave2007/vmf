#!/usr/bin/env python3
"""
NVG Verification: PBH Dark Matter Fraction & Observational Constraints
----------------------------------------------------------------------
Calculates the discrete PBH mass spectrum M_N = 0.38 * 4^N M_sun, models the
Dark Matter fraction f_PBH at each mass bin, and compares it against observational
constraints from Subaru HSC (microlensing), Hawking radiation (Voyager/Fermi), and LIGO.
"""

from __future__ import annotations
import math
import numpy as np

def get_subaru_limit(M_msun: float) -> float:
    """Analytical representation of the Subaru HSC microlensing upper bound on f_PBH."""
    if M_msun <= 1e-13:
        return 1.0
    elif M_msun < 1e-8:
        # Log-linear interpolation from (1e-13, 1.0) to (1e-8, 1e-3)
        log_M = math.log10(M_msun)
        log_1e13 = -13.0
        log_1e8 = -8.0
        frac = (log_M - log_1e13) / (log_1e8 - log_1e13)
        return 10.0 ** (-3.0 * frac)
    else:
        return 1e-3

def get_hawking_limit(M_msun: float) -> float:
    """Analytical representation of Hawking radiation constraints (Voyager 1 / EGRET)."""
    if M_msun >= 1e-16:
        return 1.0
    elif M_msun > 1e-18:
        # Extremely strong constraint below 1e-16 M_sun
        log_M = math.log10(M_msun)
        log_1e18 = -18.0
        log_1e16 = -16.0
        frac = (log_M - log_1e18) / (log_1e16 - log_1e18)
        return 10.0 ** (-8.0 * (1.0 - frac))
    else:
        return 1e-8

def main():
    print("=" * 80)
    print("     NVG PBH DARK MATTER SPECTRUM & OBSERVATIONAL LIMITS")
    print("=" * 80)

    # 1. PBH Discrete Mass Spectrum M_N = 0.38 * 4^N M_sun
    # We scan cycle indexes N from -30 to 12
    N_vals = np.arange(-30, 13)
    M_vals = 0.38 * (4.0 ** N_vals)

    # 2. VMF Relic Abundance Distribution
    # The abundance is peaked in the asteroid-mass window at N_peak = -21
    # M_{-21} = 0.38 * 4^-21 ~ 8.6e-14 M_sun
    N_peak = -21
    sigma_N = 1.3
    
    # Calculate unnormalized fraction f_N
    f_unnorm = np.exp(-((N_vals - N_peak) ** 2) / (2.0 * sigma_N ** 2))
    f_pbh = f_unnorm / np.sum(f_unnorm)  # Normalize so total fraction is 1.0 (100% of DM)

    print(f"Total modeled PBH Dark Matter fraction: {np.sum(f_pbh):.4f} (100% of DM)")
    print(f"Peak mass at cycle N={N_peak}                : {M_vals[N_vals == N_peak][0]:.2e} M_sun")
    print("-" * 80)

    # 3. Check constraints for each cycle
    print(f"  {'Cycle N':<8} | {'PBH Mass (M_sun)':<18} | {'f_PBH':<10} | {'Limit':<10} | {'Constraint Source':<20} | {'Status':<8}")
    print("  " + "-" * 76)

    all_passed = True
    for N, M, f in zip(N_vals, M_vals, f_pbh):
        # Determine relevant constraint
        if M < 1e-16:
            limit = get_hawking_limit(M)
            source = "Hawking Radiation"
        elif M < 1.0:
            limit = get_subaru_limit(M)
            source = "Subaru HSC / EROS"
        elif 1.0 <= M <= 100.0:
            limit = 1e-3  # LIGO constraint on stellar mass
            source = "LIGO Microlensing"
        else:
            limit = 1.0  # Weak constraints for very high masses (serve as seeds)
            source = "Unconstrained (Seeds)"

        status = "PASSED" if f <= limit else "FAILED"
        if f > limit:
            all_passed = False

        # Only print representative or significant cycles
        if N in [-28, -25, -22, -20, -18, -15, -10, 0, 3, 10]:
            print(f"  N = {N:<4d} | {M:<18.2e} | {f:<10.2e} | {limit:<10.2e} | {source:<20} | {status:<8}")

    print("-" * 80)
    print(f"Constraints Verification Status: {'✅ ALL PASSED' if all_passed else '❌ FAILED'}")
    assert all_passed, "PBH Dark Matter spectrum violates observational bounds!"
    print("=" * 80)

if __name__ == "__main__":
    main()
