#!/usr/bin/env python3
import sympy as sp

def main():
    print("=" * 80)
    print("  NVG THEOREM: EXACT ENTROPY DEFICIT VIA PHANTOM HORIZON")
    print("=" * 80)
    
    eps = sp.Symbol('eps', positive=True)
    z = sp.Symbol('z')
    
    # The horizon equation in dimensionless form x = r/r_g
    # x^3 - x^2 + eps^2 = 0
    # Let the roots be x_in, x_out, and x_phantom
    x_in = sp.Symbol('x_in')
    x_out = sp.Symbol('x_out')
    x_phantom = sp.Symbol('x_phantom')
    
    print("1. Vieta's formulas for cubic z^3 - z^2 + 0*z + eps^2 = 0:")
    print("   Sum of roots:           x_in + x_out + x_phantom = 1")
    print("   Sum of pair products:   x_in*x_out + x_out*x_phantom + x_phantom*x_in = 0")
    print("   Product of roots:       x_in * x_out * x_phantom = -eps^2")
    
    print("\n2. The sum of squares of the roots is:")
    print("   x_in^2 + x_out^2 + x_phantom^2 = (x_in + x_out + x_phantom)^2 - 2*(pair products)")
    print("                                  = 1^2 - 2(0) = 1")
    
    print("\n3. Area and Entropy Deficit:")
    print("   A_Schw = 4*pi*r_g^2")
    print("   A_out + A_in = 4*pi*r_g^2 * (x_out^2 + x_in^2)")
    print("                = 4*pi*r_g^2 * (1 - x_phantom^2)")
    print("   Delta A = A_Schw - (A_out + A_in) = 4*pi*r_g^2 * x_phantom^2 = A_phantom")
    print("   Delta S = S_phantom !")
    
    print("\n4. Verifying against the perturbative series:")
    # We use our previous expansions
    # x_in = eps / sqrt(1-x_in)
    x = 0
    for i in range(15):
        x = eps * (1 - x)**(-sp.Rational(1, 2))
        x = x.series(eps, 0, 15).removeO()
        
    # x_out = 1 - y, where y = eps^2 / (1-y)^2
    delta = eps**2
    y = 0
    for i in range(10):
        y = delta * (1 - y)**(-2)
        y = y.series(eps, 0, 15).removeO()
        
    x_in_series = x
    x_out_series = 1 - y
    
    # Phantom root from sum of roots = 1
    x_phantom_series = 1 - x_in_series - x_out_series
    
    print("   x_phantom = y - x_in = ")
    print("  ", sp.expand(x_phantom_series))
    
    S_deficit = (2*y - y**2 - x**2) / eps**2
    S_deficit = S_deficit.series(eps, 0, 15).removeO()
    
    phantom_area_ratio = (x_phantom_series**2) / eps**2
    phantom_area_ratio = phantom_area_ratio.series(eps, 0, 15).removeO()
    
    print("\n   [Series] Delta S / S_in(infty):")
    print("  ", sp.expand(S_deficit))
    
    print("\n   [Series] (x_phantom / eps)^2:")
    print("  ", sp.expand(phantom_area_ratio))
    
    if sp.expand(S_deficit - phantom_area_ratio) == 0:
        print("\n   => EXACT MATCH CONFIRMED TO ALL ORDERS CALCUATED.")
        print("      The information loss is precisely the entropy of the phantom root!")
    print("=" * 80)

if __name__ == "__main__":
    main()
