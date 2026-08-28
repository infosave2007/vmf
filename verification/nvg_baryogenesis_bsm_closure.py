#!/usr/bin/env python3
"""Baryogenesis channel closure audit.

This module computes the suppression of electroweak sphalerons and the proton
lifetime implied by an illustrative dimension-six operator.  It also exposes
the algebraic source required by the spontaneous-anomaly route.  The latter is
not a closure: the repository has no dynamical theta solution, CP/washout
network, or entropy history, so the empirical baryon-asymmetry claim is
explicitly retired.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nvg_baryon_asymmetry import calculate_baryon_asymmetry


T_B = 0.432  # GeV, bounce temperature bound
E_SPH = 9000.0  # GeV, illustrative sphaleron barrier
M_PL = 1.22e19  # GeV
G_STAR = 20.0
M_THETA = 6.6e-3  # GeV, illustrative phase scale
T_STAR = 0.2  # GeV, illustrative transition temperature
ETA_TARGET = 8.6e-11  # dimensionless target n_B/s (declared observation)
M_P = 0.938  # GeV
TAU_P_BOUND_S = 2.4e34 * 3.156e7
HBAR_GEV_S = 6.58e-25


def compute_channels() -> dict[str, Any]:
    """Compute channel diagnostics and preserve missing closure information."""

    exp1 = -E_SPH / T_B
    h_star = 1.66 * math.sqrt(G_STAR) * T_STAR**2 / M_PL
    s_dens = (2.0 * math.pi**2 / 45.0) * G_STAR * T_STAR**3
    eta_eq = (M_THETA * T_STAR**2 / 6.0) / s_dens
    ratio_needed = ETA_TARGET / eta_eq
    lambda_eq = (T_STAR**3 * M_PL / (1.66 * math.sqrt(G_STAR))) ** 0.25
    lambda_b = lambda_eq * (1.0 / ratio_needed) ** 0.25
    tau_p_s = (lambda_b**4 / M_P**5) * HBAR_GEV_S
    deficit = TAU_P_BOUND_S / tau_p_s
    anomaly = calculate_baryon_asymmetry()
    return {
        "sphaleron_exponent": float(exp1),
        "bsmlambda_GeV": float(lambda_b),
        "proton_lifetime_s": float(tau_p_s),
        "proton_bound_s": float(TAU_P_BOUND_S),
        "proton_bound_ratio": float(deficit),
        "anomaly_source": anomaly,
        "evidence_status": "RETIRED_MISSING_BARYOGENESIS_SOURCE",
        "observed_likelihood": None,
        "missing_components": anomaly["missing_components"],
    }


def main() -> dict[str, Any]:
    state = compute_channels()
    print("=" * 78)
    print("  NVG BARYOGENESIS CHANNEL CLOSURE AUDIT")
    print("=" * 78)
    print(f"Bounce temperature bound                 : {T_B * 1e3:.0f} MeV")
    print(f"EW sphaleron exponent                   : {state['sphaleron_exponent']:.3e}")
    print("EW sphaleron channel                    : SUPPRESSED in this temperature bound")
    print(f"Dimension-six Lambda_B                  : {state['bsmlambda_GeV'] / 1e3:.1f} TeV")
    print(f"Ungated proton lifetime                  : {state['proton_lifetime_s']:.3e} s")
    print(f"Super-K bound / trial lifetime           : {state['proton_bound_ratio']:.3e}")
    print("Dimension-six channel                   : EXCLUDED by declared proton bound")
    anomaly = state["anomaly_source"]
    print(f"Anomaly source-only eta proxy            : {anomaly['eta_proxy']:.3e}")
    print(f"theta-dot needed for eta_obs             : {anomaly['theta_dot_required_for_eta_obs_MeV']:.3e} MeV")
    print("Anomaly channel                          : NO CLOSURE (source dynamics absent)")
    print("Evidence status                          : RETIRED_MISSING_BARYOGENESIS_SOURCE")
    print("No channel is promoted as an empirical baryogenesis mechanism.")
    print("=" * 78)
    return state


if __name__ == "__main__":
    main()
