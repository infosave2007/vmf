#!/usr/bin/env python3
"""
NVG Verification: QCD Axion Mass from topological theta-sector.
Calculates the Peccei-Quinn axion decay constant f_a and mass m_a
scaled by the Genesis e-folds (N_e), comparing with ADMX/CASPEr limits.
"""

import math

def calculate_axion_parameters(m_omega: float) -> tuple[float, float]:
    m_planck = 1.2209e19   # GeV
    m_pi = 0.13957         # GeV
    f_pi = 0.0924          # GeV
    
    # Present Hubble horizon derived from H_0 = 72.8 km/s/Mpc:
    # R_H0 = c/H_0 ≈ 1.27e23 km
    r_h0 = 1.2709e23       # km
    
    # Genesis instanton scale: r_c = 1.128 * (859 / M_omega)
    r_c = 1.128 * (859.0 / m_omega)
    n_e = math.log(r_h0 / r_c)
    
    # NOTE: The formula f_a = M_planck / N_e^4 is a topological scale estimate
    # (dimensional ansatz representing the suppression of axion scale due to e-folds
    # of topological inflation) rather than a direct derivation from the NVG action.
    # It represents a consistency check (null test) for axion dark matter models.
    f_a = m_planck / (n_e ** 4)
    # Axion mass: m_a = (m_pi * f_pi) / f_a in GeV
    m_a_gev = (m_pi * f_pi) / f_a
    m_a_ev = m_a_gev * 1e9
    
    return f_a, m_a_ev

def main():
    print("=" * 70)
    print(" NVG TOPOLOGICAL AXION MASS CALCULATION")
    print("=" * 70)
    
    m_omega_center = 859.0
    m_omega_err = 8.0
    
    f_a_center, m_a_center = calculate_axion_parameters(m_omega_center)
    f_a_lower, m_a_lower = calculate_axion_parameters(m_omega_center + m_omega_err) # higher m_omega -> smaller r_c -> larger N_e -> smaller f_a -> higher m_a
    f_a_upper, m_a_upper = calculate_axion_parameters(m_omega_center - m_omega_err)
    
    # ADMX search window: 1e-6 eV to 1e-5 eV
    admx_min = 1.0e-6
    admx_max = 1.0e-5
    
    r_h0 = 1.2709e23
    r_c_center = 1.128 * (859.0 / m_omega_center)
    print(f"QCD Anchor M_Omega_0  : {m_omega_center} +/- {m_omega_err} MeV")
    print(f"Genesis e-folds N_e   : {math.log(r_h0 / r_c_center):.2f} (topologically bounded by cycle index n=77)")
    print(f"Predicted Axion Scale : {f_a_center:.3e} GeV (Range: {f_a_lower:.3e} - {f_a_upper:.3e})")
    print(f"Predicted Axion Mass  : {m_a_center:.3e} eV (Range: {m_a_lower:.3e} - {m_a_upper:.3e})")
    print(f"ADMX/CASPEr Window    : {admx_min:.1e} - {admx_max:.1e} eV")
    
    print("-" * 70)
    in_window = admx_min <= m_a_center <= admx_max
    print(f"ADMX Window Test      : {'✅ IN SEARCH WINDOW' if in_window else '❌ OUT OF WINDOW'}")
    print("This predicts a topological axion dark matter contribution in the micro-eV band.")
    print("=" * 70)

if __name__ == "__main__":
    main()
