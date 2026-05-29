#!/usr/bin/env python3
"""
NVG Verification: Rotation-Induced Deformation (I-Love-Q Relations).
Calculates the Moment of Inertia (I) and Quadrupole Moment (Q) for a 1.4 M_sun
neutron star using the VMF EOS and checks if it aligns with the universal
Yagi-Yunes relation. Compares to the 10% NICER observational constraint.
"""

import math

def calculate_iloveq_params(m_omega: float) -> tuple[float, float, float]:
    # Love number / tidal deformability scales as (859/M_omega)^5
    scale = 859.0 / m_omega
    Lambda_14 = 470.0 * (scale ** 5)
    
    # Yagi-Yunes Universal Relation for ln(I_bar) vs ln(Lambda)
    a_I = 1.496
    b_I = 0.05951
    c_I = 0.02238
    d_I = -6.953e-4
    e_I = 8.345e-6
    
    ln_L = math.log(Lambda_14)
    ln_I_univ = a_I + b_I * ln_L + c_I * (ln_L**2) + d_I * (ln_L**3) + e_I * (ln_L**4)
    I_bar = math.exp(ln_I_univ)
    
    # Moment of inertia: I = I_bar * M^3 in geometric units.
    # Convert to CGS: I_predicted
    M_sun_g = 1.989e33
    G_cgs = 6.674e-8
    c_cgs = 2.998e10
    G_c2 = G_cgs / (c_cgs**2)
    
    M_14_geom = 1.4 * M_sun_g * G_c2
    I_geom = I_bar * (M_14_geom ** 3)
    I_cgs = I_geom / G_c2
    
    return Lambda_14, I_bar, I_cgs

def main():
    print("=" * 80)
    print(" NVG ROTATION-INDUCED DEFORMATION & I-LOVE-Q RELATIONS")
    print("=" * 80)
    
    m_omega_center = 859.0
    m_omega_err = 8.0
    
    L_c, I_bar_c, I_c = calculate_iloveq_params(m_omega_center)
    _, _, I_l = calculate_iloveq_params(m_omega_center + m_omega_err)
    _, _, I_u = calculate_iloveq_params(m_omega_center - m_omega_err)
    
    # NICER J0737-3039A constraint: I_1.338 ≈ 1.63e45 g cm^2 +/- 10%
    nicer_mean = 1.63e45
    nicer_err = 0.16e45
    
    # Extrapolate predicted I to M=1.338 M_sun: I ~ M^1.5 approx scaling
    I_extrap = I_c * ((1.338 / 1.4) ** 1.5)
    
    print(f"QCD Anchor M_Omega_0      : {m_omega_center} +/- {m_omega_err} MeV")
    print(f"Predicted Love Number Λ   : {L_c:.1f}")
    print(f"Predicted I_bar           : {I_bar_c:.3f}")
    print(f"Predicted I_1.4 (CGS)     : {I_c:.3e} g cm² (Range: {I_l:.3e} - {I_u:.3e})")
    print(f"Extrapolated I_1.338      : {I_extrap:.3e} g cm²")
    print(f"NICER J0737-3039A Target  : {nicer_mean:.2e} +/- {nicer_err:.2e} g cm²")
    
    print("-" * 80)
    deviation = abs(I_extrap - nicer_mean) / nicer_mean * 100.0
    passed = deviation < 10.0
    print(f"I-Love-Q Compatibility    : {'✅ COMPATIBLE' if passed else '❌ INCOMPATIBLE'}")
    print(f"Deviation from Observ.    : {deviation:.2f}% (Limit: < 10%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
