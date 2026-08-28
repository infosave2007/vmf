#!/usr/bin/env python3
"""Conditional/in-sample comparison with the NICER J0437-4715 input.

The transition parameters of the canonical EOS were selected using J0740,
GW170817, and NICER constraints.  This entry point therefore reports a
descriptive runtime comparison only: it carries zero independent evidence
weight and must not be read as an independent NICER evidence test.
"""

import os
import sys
import numpy as np
import math

# Add local path to import EOS solving classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nvg_tidal_deformability import EOS, solve_tov_tidal


COMPARISON_METADATA = {
    "status": "CONDITIONAL_IN_SAMPLE",
    "kind": "derived_runtime_comparison",
    "independent": False,
    "evidence_weight": 0.0,
    "selected_on": ["J0740", "GW170817", "NICER"],
    "source": "nvg_tidal_deformability.EOS + solve_tov_tidal",
}

def run_nicer_check():
    print("==========================================================================")
    print("  NVG CONDITIONAL/IN-SAMPLE COMPARISON WITH NICER 2024 (PSR J0437-4715)")
    print("==========================================================================")
    print(
        "  status=CONDITIONAL_IN_SAMPLE; independent=False; evidence_weight=0.0 "
        "(selection inputs reused; no independent evidence)"
    )
    
    # Observed NICER 2024 values
    M_obs = 1.418
    M_err = 0.037
    R_obs = 11.36
    R_err = 0.8
    
    print(f"NICER 2024 Measurement for PSR J0437-4715:")
    print(f"  Mass M = {M_obs} ± {M_err} M_sun")
    print(f"  Radius R = {R_obs} ± {R_err} km")
    
    # Solve TOV for NVG EOS
    eos = EOS(p_match=1.5, Gamma=1.35)
    
    # центральные давления для поиска 1.418 M_sun
    P_centers = np.logspace(-1.0, 2.8, 200)
    results = []
    
    for Pc in P_centers:
        M, R, k2, Lam = solve_tov_tidal(eos, Pc)
        if M > 0.5 and R > 5.0:
            results.append((M, R))
            
    # Найти точку, ближайшую к M_obs = 1.418
    best_diff = 1e9
    R_nvg = 0.0
    M_nvg = 0.0
    for M, R in results:
        diff = abs(M - M_obs)
        if diff < best_diff:
            best_diff = diff
            R_nvg = R
            M_nvg = M
            
    print("\nRuntime canonical EOS output (conditional/in-sample; no independent evidence):")
    print(f"  Calculated Mass: {M_nvg:.4f} M_sun (diff: {best_diff:.4f})")
    print(f"  Radius output: {R_nvg:.2f} km")
    
    # Вычисление отклонения (Z-score)
    z_score = (R_nvg - R_obs) / R_err
    
    print(f"\nStatistical Compatibility Audit:")
    print(f"  Radius Difference: {R_nvg - R_obs:+.2f} km")
    print(f"  Z-score Deviation: {z_score:.2f}σ")
    
    # p-value для двустороннего Z-теста
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2.0))))
    print(
        f"  descriptive normal-approximation statistic: {p_val:.4f} "
        "(not an independent p-value/evidence claim)"
    )
    
    is_ok = abs(z_score) <= 1.5
    print(
        "  Status: "
        + (
            "COMPATIBLE (conditional/in-sample; zero independent evidence weight)"
            if is_ok
            else "TENSION (conditional/in-sample; zero independent evidence weight)"
        )
    )
    
    print("\nPhysics Context (honest reading):")
    print(f"The conditional canonical EOS output is R = {R_nvg:.2f} km at {M_nvg:.3f} M_sun, i.e.")
    print(f"{z_score:+.1f} sigma above the J0437 central value {R_obs} km — outside the 68% CI")
    print("but inside 95%. This is a descriptive conditional comparison, not independent evidence:")
    print("a future J0437 radius confirmed below ~12.0 km at high precision would stress")
    print("the canonical parameterization (see nvg_ns_parameter_scan.py); no independent claim is made.")
    print("==========================================================================")

    return {
        "metadata": COMPARISON_METADATA,
        "observed": {"mass_msun": M_obs, "mass_sigma": M_err, "radius_km": R_obs, "radius_sigma": R_err},
        "runtime": {"mass_msun": M_nvg, "radius_km": R_nvg},
        "z_score": z_score,
        "descriptive_statistic": p_val,
        "status": "COMPATIBLE" if is_ok else "TENSION",
    }

if __name__ == "__main__":
    run_nicer_check()
