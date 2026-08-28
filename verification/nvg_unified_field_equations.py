#!/usr/bin/env python3
"""Runtime algebraic checks for the NVG/VMF field-equation ansatz.

The bounce density and anomaly growth expressions are recomputed from the
declared equations.  Magnetic pressure and post-merger frequency coefficients
are explicit sensitivity inputs, not a solved magnetar or waveform model.
"""

from __future__ import annotations

import math
from typing import Any


c = 2.99792458e10
hbar = 1.0545718e-27
G_Newton = 6.67430e-8
alpha_EM = 1.0 / 137.035999
MeV_to_erg = 1.60217663e-6
fm_to_cm = 1e-13
M_omega_0 = 859.0


def verify_cosmological_bounce() -> tuple[float, float, float]:
    """Compute the QCD-anchor density and positive acceleration term."""

    hbar_c = 197.326979
    rho_mev_fm3 = M_omega_0**4 / hbar_c**3
    rho_c_cgs = rho_mev_fm3 * MeV_to_erg / fm_to_cm**3
    rho_mass = rho_c_cgs / c**2
    ddot_a = (16.0 * math.pi * G_Newton / (3.0 * c**2)) * rho_c_cgs
    return float(rho_mev_fm3), float(rho_mass), float(ddot_a)


def verify_magnetar_chiral_instability() -> tuple[float, float, float]:
    """Evaluate the declared anomaly/conductivity toy expression."""

    sigma_core = 1e29
    delta_w_erg = 0.20 * M_omega_0 * MeV_to_erg
    dot_theta = delta_w_erg / hbar
    gamma_topo = 2.0
    sigma_topo = gamma_topo * (alpha_EM / (2.0 * math.pi)) * dot_theta
    growth = sigma_topo**2 / (4.0 * sigma_core)
    return float(dot_theta), float(growth), float(1.0 / growth)


def verify_magnetic_backpressure() -> tuple[float, float, float]:
    """Evaluate magnetic-pressure sensitivity at an explicit field benchmark."""

    B_core = 2.82e16
    p_mag_cgs = B_core**2 / (8.0 * math.pi)
    p_mag_mev_fm3 = p_mag_cgs / 1.60217663e33
    f_pi, m_sigma = 93.0, 500.0
    stiffness = f_pi**2 * m_sigma**2
    p_mag_mev4 = p_mag_mev_fm3 * 197.326979**3
    delta_w_ratio = -p_mag_mev4 / stiffness
    df2_dpc = -20.0
    return float(p_mag_mev_fm3), float(delta_w_ratio), float(df2_dpc * p_mag_mev_fm3)


def compute_unified_state() -> dict[str, Any]:
    rho, rho_mass, ddot_a = verify_cosmological_bounce()
    dot_theta, growth, tau_efold = verify_magnetar_chiral_instability()
    p_mag, delta_w, delta_f2 = verify_magnetic_backpressure()
    return {
        "rho_c_MeV_fm3": rho,
        "rho_c_g_cm3": rho_mass,
        "ddot_a_over_a_s2": ddot_a,
        "dot_theta_rad_s": dot_theta,
        "growth_s-1": growth,
        "tau_efold_s": tau_efold,
        "P_mag_MeV_fm3": p_mag,
        "delta_W_ratio": delta_w,
        "delta_f2_Hz": delta_f2,
        "evidence_status": "FORMAL_MODEL_SENSITIVITY",
        "observed_likelihood": None,
        "limitation": "No UV-complete field solver, magnetar saturation solver, or GW likelihood is present.",
    }


def main() -> dict[str, Any]:
    state = compute_unified_state()
    print("=" * 80)
    print("      NVG UNIFIED FIELD-EQUATION ALGEBRAIC CHECKS")
    print("=" * 80)
    print("1. Cosmological bounce density and acceleration")
    print(f"   rho_c                                  : {state['rho_c_MeV_fm3']:.4e} MeV/fm^3")
    print(f"   mass density                            : {state['rho_c_g_cm3']:.4e} g/cm^3")
    print(f"   acceleration term                      : {state['ddot_a_over_a_s2']:.4e} s^-2")
    print("2. Chiral-instability sensitivity")
    print(f"   dot(theta)                             : {state['dot_theta_rad_s']:.4e} rad/s")
    print(f"   growth rate / e-fold                   : {state['growth_s-1']:.4e} s^-1 / {state['tau_efold_s']:.4e} s")
    print("3. Magnetic-backpressure sensitivity")
    print(f"   pressure                               : {state['P_mag_MeV_fm3']:.5f} MeV/fm^3")
    print(f"   gap shift ratio                        : {state['delta_W_ratio']:.4e}")
    print(f"   f2 coefficient result                  : {state['delta_f2_Hz']:.3f} Hz")
    print("Evidence status                           : FORMAL_MODEL_SENSITIVITY")
    print("Magnetar fields and GW shifts are not observationally inferred here.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
