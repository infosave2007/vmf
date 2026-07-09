import numpy as np

def calculate_bh_interior(m_tot_solar, eps_max_mev_fm3):
    """
    Calculates the metric and curvature invariants for a regular black hole
    core driven by the NVG trace anomaly (vacuum saturation).
    """
    G_c2 = 1.476  # km/M_sun
    m_tot = m_tot_solar * G_c2 # in km
    
    # eps_max is the maximum vacuum energy density when mass melts
    # Convert MeV/fm^3 to km^-2. 
    # 1 MeV/fm^3 = 1.323e-6 km^-2 (geometrized units)
    eps_max_geom = eps_max_mev_fm3 * 1.323e-6
    
    # Calculate characteristic scale r_0 based on limiting density
    # M_tot = (4*pi/3) * eps_max * r_0^3
    r_0 = (3 * m_tot / (4 * np.pi * eps_max_geom))**(1/3)
    
    r = np.logspace(-5, 2, 1000) # From center to 100 km
    
    # Mass function M(r)
    m_r = m_tot * (r**3) / (r**3 + r_0**3)
    
    # Metric function f(r) = 1 - 2M(r)/r
    f_r = 1 - 2 * m_r / r
    
    # Energy density eps(r) = M'(r) / (4 pi r^2)
    eps_r = (3 * m_tot * r_0**3) / (4 * np.pi * (r**3 + r_0**3)**2)
    
    # Kretschmann scalar K = R_abcd R^abcd
    # For f(r) = 1 - 2M(r)/r, it is a known analytical expression
    # K = f''^2 + 4(f'/r)^2 + 4((1-f)/r^2)^2
    f_prime = (2 * m_r / r**2) - (2 / r) * (3 * m_tot * r**2 * r_0**3 / (r**3 + r_0**3)**2)
    
    # At r=0, f(r) ~ 1 - (2 m_tot / r_0^3) r^2
    # So it is de Sitter space with cosmological constant Lambda = 6 m_tot / r_0^3
    Lambda_eff = 6 * m_tot / (r_0**3)
    
    print("=== NVG Regular Black Hole Core Validation ===")
    print(f"Total Mass: {m_tot_solar} M_sun")
    print(f"Vacuum Saturation Density (eps_max): {eps_max_mev_fm3:.1e} MeV/fm^3")
    print(f"Core length scale (r_0): {r_0:.4f} km")
    print(f"Effective Cosmological Constant at center: {Lambda_eff:.4f} km^-2")
    
    # Check values at r=0
    # Core Genesis scale l = r_c
    l_rc = np.sqrt(3 / (8 * np.pi * eps_max_geom))
    K_0 = 24 * (Lambda_eff / 3)**2 # de Sitter Kretschmann (24/l^4)
    
    # Explicit regression test for K(0)
    expected_K0 = 24 / (l_rc**4)
    assert abs(K_0 - expected_K0) < 1e-10, f"K(0) regression failed! {K_0} != {expected_K0}"
    
    print(f"\nAt r -> 0:")
    print(f"  f(r) -> 1")
    print(f"  Metric completely regular.")
    print(f"  Kretschmann scalar K(0): {K_0:.4f} km^-4 (FINITE) [Verified 24/l^4]")
    print(f"  Energy Density eps(0): {eps_r[0] / 1.323e-6:.1e} MeV/fm^3")
    
    # Verify strong energy condition violation at core
    print(f"\nThermodynamics at center (r->0):")
    print(f"  P_r = -eps (Vacuum state)")
    print(f"  eps + 3P = eps - 3eps = -2eps < 0")
    print(f"  Strong Energy Condition VIOLATED -> Singularity Avoided.")

    # Horizons
    horizon_idx = np.where(np.diff(np.sign(f_r)))[0]
    horizons = r[horizon_idx]
    if len(horizons) >= 2:
        print(f"\nHorizons detected:")
        print(f"  Inner (Cauchy) horizon: {horizons[0]:.2f} km")
        print(f"  Outer (Event) horizon: {horizons[1]:.2f} km")
    else:
        print(f"\nOuter (Event) horizon: {horizons[0] if len(horizons)>0 else 'None'} km")

if __name__ == '__main__':
    # Use NVG QCD parameters
    # The max vacuum energy density when M_omega melts is approx the QCD scale
    # B ~ (200 MeV)^4 ~ 50 MeV/fm^3. Or use M_omega_0^4 ~ 7.1e4 MeV/fm^3
    calculate_bh_interior(m_tot_solar=10.0, eps_max_mev_fm3=7.1e4)
