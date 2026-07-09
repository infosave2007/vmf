import numpy as np
import sympy as sp

def analyze_entropy():
    # Constants
    c = 299792458  # m/s
    G = 6.67430e-11  # m^3/kg/s^2
    M_sun = 1.98847e30  # kg
    k_B = 1.380649e-23  # J/K
    hbar = 1.054571817e-34  # J s
    
    # NVG core density
    rho_c_MeV_fm3 = 7.09e4  # MeV/fm^3
    rho_c_kg_m3 = rho_c_MeV_fm3 * 1.78266192e-30 * 1e45  # ~1.26e18 kg/m^3
    
    # Calculate de Sitter length scale l
    # l = sqrt(3 c^2 / (8 pi G rho_c))
    l_m = np.sqrt(3 * c**2 / (8 * np.pi * G * rho_c_kg_m3))
    l_km = l_m / 1000
    
    print(f"Universal core length scale l: {l_km:.6f} km")
    
    # 1. Analytical limit using SymPy
    r, M, l_sym = sp.symbols('r M l', real=True, positive=True)
    G_sym, c_sym = sp.symbols('G c', real=True, positive=True)
    
    # Horizon equation: epsilon * r^3 - r^2 + l^2 = 0 where epsilon = c^2 / (2GM)
    epsilon = sp.Symbol('epsilon', real=True, positive=True)
    eq = epsilon * r**3 - r**2 + l_sym**2
    
    print("\n--- Analytical Asymptotics for M -> infinity (epsilon -> 0) ---")
    # We substitute r = l + a*epsilon + b*epsilon**2
    a, b = sp.symbols('a b', real=True)
    r_expansion = l_sym + a * epsilon + b * epsilon**2
    eq_expanded = eq.subs(r, r_expansion).expand()
    
    # Group by powers of epsilon
    coeff_0 = eq_expanded.subs(epsilon, 0)
    coeff_1 = sp.diff(eq_expanded, epsilon).subs(epsilon, 0)
    
    print(f"0th order in epsilon: {coeff_0} == 0  => r_in -> l")
    
    # Solve for a
    a_sol = sp.solve(coeff_1, a)[0]
    print(f"1st order coefficient 'a': {a_sol}")
    
    # r_in(M) asymptotic formula:
    # epsilon = c^2 / (2GM)
    print(f"r_in(M) ~ l + (l^2 c^2) / (4 G M) + O(1/M^2)")
    
    # 2. Universal Entropy
    # S = k_B c^3 A / (4 G hbar)
    A_in = 4 * np.pi * l_m**2
    S_in = k_B * c**3 * A_in / (4 * G * hbar)
    S_in_bits = S_in / (k_B * np.log(2))
    
    print(f"\n--- Universal Entropy of the NVG Core ---")
    print(f"Area of inner horizon A_in: {A_in:.4e} m^2")
    print(f"Entropy S_in: {S_in:.4e} J/K")
    print(f"Information content S_in (bits): {S_in_bits:.4e} bits")
    
    # Compare with observable universe
    S_univ_bits = 1e122  # Penrose estimate
    print(f"Fraction of observable universe entropy: {S_in_bits / S_univ_bits:.1e}")
    
    # 3. Entropy Deficit
    print(f"\n--- Entropy Deficit ---")
    masses = [1.0, 10.0, 65.0, 1000.0]
    
    for mass in masses:
        # Exact numerical root finding
        M_kg = mass * M_sun
        eps_val = c**2 / (2 * G * M_kg)
        
        # solve eps_val * r^3 - r^2 + l_m^2 = 0
        coeffs = [eps_val, -1.0, 0.0, l_m**2]
        roots = np.roots(coeffs)
        real_roots = np.sort(np.real(roots[np.isreal(roots)]))
        positive_roots = real_roots[real_roots > 0]
        
        if len(positive_roots) >= 2:
            r_in = positive_roots[0]
            r_out = positive_roots[1]
            
            A_out = 4 * np.pi * r_out**2
            S_out = k_B * c**3 * A_out / (4 * G * hbar)
            S_out_bits = S_out / (k_B * np.log(2))
            
            r_schwarzschild = 2 * G * M_kg / c**2
            A_schwarzschild = 4 * np.pi * r_schwarzschild**2
            S_schwarzschild = k_B * c**3 * A_schwarzschild / (4 * G * hbar)
            S_schwarzschild_bits = S_schwarzschild / (k_B * np.log(2))
            
            # The deficit is how much entropy is "missing" compared to a classical black hole
            S_total_nvg = S_out_bits + S_in_bits
            deficit = S_schwarzschild_bits - S_total_nvg
            
            print(f"M = {mass:4.0f} M_sun | r_in = {r_in/1000:.5f} km | Deficit: {deficit:.4e} bits ({deficit/S_schwarzschild_bits*100:.6f}%)")

    # 4. Proof of Universality (r_0(M) vs l)
    print(f"\n--- 4. Proof of Universality (r_0 vs l) ---")
    print(f"The alternative formulation uses r_0(M) = (3M / (4 pi rho_c))^(1/3).")
    print(f"The Hayward horizon equation is: f(r) = 1 - 2GMr^2 / (c^2(r^3 + r_0^3)) = 0")
    print(f"This implies: c^2 r^3 - 2GMr^2 + c^2 r_0^3 = 0")
    print(f"Substitute r_0^3 = 3M / (4 pi rho_c):")
    print(f"c^2 r^3 - 2GMr^2 + c^2 * (3M / (4 pi rho_c)) = 0")
    print(f"Divide the entire equation by 2GM:")
    print(f"(c^2 / 2GM) r^3 - r^2 + (3 c^2) / (8 pi G rho_c) = 0")
    print(f"Notice the constant term is exactly (3 c^2) / (8 pi G rho_c) = l^2 !")
    print(f"Therefore, the equation is universally: epsilon * r^3 - r^2 + l^2 = 0")
    print(f"CONCLUSION: The asymptotic limit l = {l_km:.5f} km is universally robust, regardless of whether parameterized via l or r_0(M).")

if __name__ == '__main__':
    analyze_entropy()
