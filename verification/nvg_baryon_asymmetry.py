#!/usr/bin/env python3
"""Spontaneous-baryogenesis source-term audit.

The module evaluates the DIGA suppression and the algebraic conversion from a
specified phase velocity to a baryon asymmetry.  It does not supply a
theta(t) solution, CP-violating source, washout network, or entropy history.
Consequently no baryogenesis value is treated as an NVG prediction.
"""

from __future__ import annotations

import math
from typing import Any


# Declared physics/literature inputs used by the sensitivity calculation.
T_B_MEV = 432.2
T_C_MEV = 157.0
ALPHA_S = 0.3
ETA_OBS = 6.1e-10
ETA_OBS_ERR = 0.3e-10
ZETA_3 = 1.20205


def baryon_asymmetry_coefficient(T_b_MeV: float, *, zeta_3: float = ZETA_3) -> float:
    """Return eta per MeV of an externally specified theta-dot source."""

    if T_b_MeV <= 0.0 or zeta_3 <= 0.0:
        raise ValueError("temperature and zeta(3) must be positive")
    n_gamma_over_T3 = 2.0 * zeta_3 / math.pi**2
    return 1.0 / (T_b_MeV * 6.0 * n_gamma_over_T3)


def calculate_baryon_asymmetry(
    *,
    T_b_MeV: float = T_B_MEV,
    T_c_MeV: float = T_C_MEV,
    alpha_s: float = ALPHA_S,
) -> dict[str, Any]:
    """Compute suppression and source coefficient without inventing theta dynamics."""

    if T_b_MeV <= 0.0 or T_c_MeV <= 0.0 or alpha_s <= 0.0:
        raise ValueError("temperatures and alpha_s must be positive")
    suppression = (T_c_MeV / T_b_MeV) ** 14
    theta_dot_required = ETA_OBS / baryon_asymmetry_coefficient(T_b_MeV)
    # DIGA supplies a suppression factor, not a normalization chosen to match
    # eta_obs.  Retain the unspecialized dimensional scale for sensitivity.
    theta_dot_suppressed = alpha_s**4 * T_b_MeV * suppression
    eta_proxy = theta_dot_suppressed * baryon_asymmetry_coefficient(T_b_MeV)
    return {
        "T_b_MeV": float(T_b_MeV),
        "T_c_MeV": float(T_c_MeV),
        "alpha_s": float(alpha_s),
        "diga_suppression": float(suppression),
        "theta_dot_suppressed_MeV": float(theta_dot_suppressed),
        "eta_proxy": float(eta_proxy),
        "eta_coefficient_per_MeV": float(baryon_asymmetry_coefficient(T_b_MeV)),
        "theta_dot_required_for_eta_obs_MeV": float(theta_dot_required),
        "eta_observed": ETA_OBS,
        "eta_observed_error": ETA_OBS_ERR,
        "evidence_status": "RETIRED_MISSING_BARYOGENESIS_SOURCE",
        "observed_likelihood": None,
        "missing_components": [
            "dynamical theta(t) solution and its initial conditions",
            "CP-violating baryon-number source and washout network",
            "reheating/entropy history converting n_B/s to the quoted eta",
        ],
    }


def main() -> dict[str, Any]:
    state = calculate_baryon_asymmetry()
    deviation = (state["eta_proxy"] - ETA_OBS) / ETA_OBS_ERR
    print("=" * 74)
    print("  NVG BARYOGENESIS SOURCE-TERM AUDIT")
    print("=" * 74)
    print(f"Bounce temperature T_b                 : {state['T_b_MeV']:.2f} MeV")
    print(f"QCD crossover T_c                      : {state['T_c_MeV']:.2f} MeV")
    print(f"DIGA suppression (T_c/T_b)^14          : {state['diga_suppression']:.3e}")
    print(f"Algebraic eta/theta-dot coefficient    : {state['eta_coefficient_per_MeV']:.3e} MeV^-1")
    print(f"theta-dot required for eta_obs         : {state['theta_dot_required_for_eta_obs_MeV']:.3e} MeV")
    print(f"Illustrative DIGA theta-dot source     : {state['theta_dot_suppressed_MeV']:.3e} MeV")
    print(f"Resulting source-only eta proxy        : {state['eta_proxy']:.3e}")
    print(f"Proxy-minus-observation deviation      : {deviation:+.2f} sigma (not a fit)")
    print("Evidence status                        : RETIRED_MISSING_BARYOGENESIS_SOURCE")
    print("No theta dynamics, CP/washout solver, or entropy likelihood is present.")
    print("=" * 74)
    return state


if __name__ == "__main__":
    main()
