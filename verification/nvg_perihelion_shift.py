#!/usr/bin/env python3
"""
NVG Verification: Strong-field Periastron Advance in PSR J0737-3039.
Calculates the NVG vacuum polarization correction to General Relativity's
predicted perihelion advance and validates it against the observed precision.
"""

import math

def calculate_perihelion_correction(m_omega: float) -> tuple[float, float]:
    # eps_eff = exp(ln(0.135) * m_omega / 859)
    eps_eff = math.exp(math.log(0.135) * m_omega / 859.0)
    
    # Binary system parameters: PSR J0737-3039
    a_orbit = 8.8e5       # km (orbital semi-major axis)
    r_ns = 12.0           # km (neutron star radius)
    
    # NVG correction relative to standard GR: delta_phi / delta_phi_GR = (1 - eps_eff) * (r_ns / a_orbit)^2
    ratio = (1.0 - eps_eff) * ((r_ns / a_orbit) ** 2)
    
    # Standard GR periastron advance for J0737-3039
    omega_dot_gr = 16.89947 # deg/year
    
    delta_omega_dot = omega_dot_gr * ratio
    return ratio, delta_omega_dot

def main():
    print("=" * 80)
    print(" NVG STRONG-FIELD PERIASTRON ADVANCE VERIFICATION")
    print("=" * 80)
    
    m_omega_center = 859.0
    m_omega_err = 8.0
    
    ratio_center, delta_center = calculate_perihelion_correction(m_omega_center)
    ratio_lower, delta_lower = calculate_perihelion_correction(m_omega_center + m_omega_err) # higher m_omega -> smaller eps_eff -> larger correction
    ratio_upper, delta_upper = calculate_perihelion_correction(m_omega_center - m_omega_err)
    
    # Observational precision of omega_dot in PSR J0737-3039 (Kramer et al. 2021: 0.02% error)
    precision_limit = 2.0e-4 # 0.02%
    
    print(f"QCD Anchor M_Omega_0 : {m_omega_center} +/- {m_omega_err} MeV")
    print(f"Core eps_eff/eps_0   : {math.exp(math.log(0.135) * m_omega_center / 859.0):.4f}")
    print(f"PSR J0737-3039 orbital separation a: 8.8e5 km")
    print(f"Observed GR Periastron Advance     : 16.89947 deg/yr")
    print(f"Predicted NVG/GR fractional diff   : {ratio_center:.3e} (Range: {ratio_upper:.3e} - {ratio_lower:.3e})")
    print(f"Absolute NVG correction            : {delta_center:.3e} deg/yr")
    print(f"Observational Precision Limit      : {precision_limit:.3e} (0.02%)")
    
    print("-" * 80)
    passed = ratio_center < precision_limit
    print(f"Strong-Field Null Test Status      : {'✅ PASSED (NVG correction is undetectable)' if passed else '❌ FAILED'}")
    print(f"Safety Margin                      : {precision_limit / ratio_center:.1e}x below current detection limits")
    print("=" * 80)

if __name__ == "__main__":
    main()
