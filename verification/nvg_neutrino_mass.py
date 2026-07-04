#!/usr/bin/env python3
"""
NVG Verification: Majorana Neutrino Mass Sum from Goldstone phase
-----------------------------------------------------------------
Calculates the suppressed Majorana neutrino mass from the Higgs seesaw
mechanism with the Goldstone theta-phase scaling factor. Compares the
sum of the three neutrino masses (under normal hierarchy and degenerate sum
interpretations) against the cosmological limit from Planck PR4 (sum m_ν < 0.12 eV).
"""

import math
import numpy as np

def calculate_neutrino_mass(m_omega: float) -> float:
    v_ew = 246.22e9     # eV (Electroweak Higgs VEV)
    # The Grand Unified Theory (GUT) scale is an external model prior (2e16 GeV):
    lambda_gut = 2.0e25 # eV
    m_pi = 139.57e6     # eV (pion mass)
    
    # Goldstone theta phase factor: theta_eff = 2 * pi * M_omega / m_pi
    theta_eff = 2.0 * math.pi * (m_omega * 1e6) / m_pi
    
    # Majorana mass scale predicted by VMF seesaw modulation:
    # m_nu = (v_ew^2 / lambda_gut) * theta_eff
    # This represents a topological scale estimate (null test), modulating the standard
    # seesaw scale by the vacuum Goldstone phase factor.
    return (v_ew**2 / lambda_gut) * theta_eff

def main():
    print("==========================================================================")
    print("  NVG/VMF CALCULATION: NEUTRINO MASS SUM VS PLANCK PR4 LIMIT")
    print("==========================================================================")
    
    m_omega_center = 859.0
    m_omega_err = 8.0
    
    m_nu_scale = calculate_neutrino_mass(m_omega_center)
    m_nu_scale_lower = calculate_neutrino_mass(m_omega_center - m_omega_err)
    m_nu_scale_upper = calculate_neutrino_mass(m_omega_center + m_omega_err)
    
    # Standard neutrino oscillation parameters (PDG 2024):
    # Delta m^2_21 = 7.53e-5 eV^2
    # Delta m^2_32 = 2.45e-3 eV^2 (for normal hierarchy)
    dm2_21 = 7.53e-5
    dm2_32 = 2.45e-3
    
    # ── Scenario A: VMF scale represents the SUM of neutrino masses (Σm_ν) ──
    sum_m_nu_scen_A = m_nu_scale
    sum_m_nu_scen_A_lower = m_nu_scale_lower
    sum_m_nu_scen_A_upper = m_nu_scale_upper
    
    # Under Scenario A, we solve for the individual masses in Normal Hierarchy:
    # m_1 + m_2 + m_3 = sum_m_nu
    # m_2 = sqrt(m_1^2 + dm2_21)
    # m_3 = sqrt(m_1^2 + dm2_32 + dm2_21)
    # We find m_1 numerically such that the sum is sum_m_nu
    m_1 = 0.0
    for m_1_try in np.linspace(0, 0.1, 10000):
        m_2_try = math.sqrt(m_1_try**2 + dm2_21)
        m_3_try = math.sqrt(m_1_try**2 + dm2_32 + dm2_21)
        if (m_1_try + m_2_try + m_3_try) >= sum_m_nu_scen_A:
            m_1 = m_1_try
            break
    m_2 = math.sqrt(m_1**2 + dm2_21)
    m_3 = math.sqrt(m_1**2 + dm2_32 + dm2_21)
    
    # ── Scenario B: VMF scale represents the HEAVIEST mass m_3 ──
    m_3_scen_B = m_nu_scale
    m_2_scen_B = math.sqrt(max(0, m_3_scen_B**2 - dm2_32))
    m_1_scen_B = math.sqrt(max(0, m_2_scen_B**2 - dm2_21))
    sum_m_nu_scen_B = m_1_scen_B + m_2_scen_B + m_3_scen_B
    
    # Cosmological upper limits on the neutrino mass sum
    planck_limit = 0.12   # eV (Planck PR4)
    desi_lcdm = 0.064     # eV (DESI DR2 BAO + CMB, LCDM, 95%; arXiv:2503.14744)
    desi_w0wa = 0.16      # eV (same data, w0-wa dynamical dark energy, 95%)
    katrin_limit = 0.45   # eV
    
    print(f"QCD Anchor M_Omega_0             : {m_omega_center} +/- {m_omega_err} MeV")
    print(f"External prior GUT scale         : 2.0e16 GeV (fixed standard prior)")
    print(f"VMF Majorana Mass Scale (m_nu)   : {m_nu_scale:.4f} eV")
    print("-" * 74)
    print(f"SCENARIO A (VMF scale maps to the sum of masses):")
    print(f"  Predicted sum (Σm_ν)           : {sum_m_nu_scen_A:.4f} eV (Range: {sum_m_nu_scen_A_lower:.4f} - {sum_m_nu_scen_A_upper:.4f})")
    print(f"  Individual masses (NH)         : m_1 = {m_1:.4f} eV, m_2 = {m_2:.4f} eV, m_3 = {m_3:.4f} eV")
    print(f"  Planck PR4 Limit (Σm_ν < 0.12) : {'✅ within' if sum_m_nu_scen_A < planck_limit else '❌ EXCEEDED'}")
    print(f"  DESI DR2, LCDM (< 0.064)       : {'✅ within' if sum_m_nu_scen_A < desi_lcdm else '❌ EXCLUDED at 95%'}")
    print(f"  DESI DR2, w0-wa (< 0.16)       : {'✅ within' if sum_m_nu_scen_A < desi_w0wa else '❌ EXCLUDED at 95%'}")
    print()
    print(f"SCENARIO B (VMF scale maps to the heaviest mass m_3):")
    print(f"  Predicted sum (Σm_ν)           : {sum_m_nu_scen_B:.4f} eV")
    print(f"  Individual masses (NH)         : m_1 = {m_1_scen_B:.4f} eV, m_2 = {m_2_scen_B:.4f} eV, m_3 = {m_3_scen_B:.4f} eV")
    print(f"  All cosmological limits        : {'✅ within' if sum_m_nu_scen_B < desi_lcdm else '❌ EXCLUDED (exceeds even the w0-wa limit)' if sum_m_nu_scen_B > desi_w0wa else '⚠️ tension'}")
    print()
    print(f"KATRIN test (m_ν,max < 0.45 eV)  : {'✅ PASSED' if m_nu_scale < katrin_limit else '❌ FAILED'}")
    print("-" * 74)
    print("CONCLUSION (honest status, 2025 data):")
    print("The theta_eff factor and the GUT-scale prior are choices, so this is a scale")
    print("estimate. Scenario B is EXCLUDED by cosmology. Scenario A (Σm_ν ≈ 0.117 eV)")
    print("is EXCLUDED at 95% within LCDM by DESI DR2 (< 0.064 eV) but remains allowed")
    print("in a w0-wa cosmology (< 0.16 eV) — and NVG itself predicts dynamical dark")
    print("energy. The neutrino sector is therefore viable ONLY jointly with the NVG")
    print("dark-energy claim, which makes the pair sharply co-testable.")
    print("==========================================================================")

if __name__ == "__main__":
    main()
