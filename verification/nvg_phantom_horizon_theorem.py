#!/usr/bin/env python3
"""
NVG: Horizon-area deficit of the Hayward regular black hole (exact area identity).

For the dimensionless horizon equation  z^3 - z^2 + eps^2 = 0
(z = r/r_Sch, eps = l/r_Sch), Vieta's formulas give e1 = 1, e2 = 0, so the sum
of squares of the three roots is exactly p2 = e1^2 - 2 e2 = 1, independent of
mass. Two roots are the physical outer/inner horizons; the third is a negative
(unphysical) root. The identity p2 = 1 lets the outer+inner horizon-area deficit
relative to a Schwarzschild hole of the same mass be written in closed form:

    dA / A_Sch = 1 - (z_out^2 + z_in^2) = z_phantom^2 .

This is an exact but ELEMENTARY algebraic property of the cubic -- a compact
expression for the area deficit, NOT a physical conservation law. The third root
is negative (no horizon lives there), so no Bekenstein-Hawking entropy attaches
to it, and the identity says nothing about the unitarity of Hawking evaporation
or the information paradox (those concern evaporation dynamics, not static areas).
"""
import numpy as np
import sympy as sp


def main():
    print("=" * 78)
    print("  NVG: Hayward horizon-area deficit (exact area identity)")
    print("=" * 78)

    eps = sp.Symbol('eps', positive=True)

    # 1. Vieta / Newton: sum of squares of the roots is exactly 1 (no series needed).
    e1, e2, e3 = sp.Integer(1), sp.Integer(0), -eps**2   # z^3 - z^2 + 0*z + eps^2
    p2 = e1**2 - 2 * e2
    print("\n1. Vieta on z^3 - z^2 + eps^2 = 0:")
    print(f"   e1 = sum z_i         = {e1}")
    print(f"   e2 = sum z_i z_j     = {e2}")
    print(f"   e3 = z_out z_in z_ph = {e3}")
    print(f"   p2 = sum z_i^2 = e1^2 - 2 e2 = {p2}   (exact, mass-independent)")
    assert p2 == 1, "Sum of squares of the roots must be exactly 1."

    # 2. Numerical confirmation at several eps: real roots, sum of squares = 1,
    #    and dA/A_Sch = z_phantom^2 to machine precision. The extremal (double-root)
    #    case is at eps = 2/(3*sqrt(3)) ~ 0.3849, so all test values are below it.
    print("\n2. Numerical check (roots of the cubic):")
    header = (f"   {'eps':>9} | {'z_out':>12} | {'z_in':>12} | {'z_phantom':>12} | "
              f"{'sum z^2':>9} | {'dA/A - z_ph^2':>14}")
    print(header)
    print("   " + "-" * (len(header) - 3))
    for eps_val in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]:
        roots = np.sort(np.roots([1.0, -1.0, 0.0, eps_val ** 2]).real)
        z_ph, z_in, z_out = roots            # ascending: negative, small+, ~1
        sum_sq = float(np.sum(roots ** 2))
        dA_over_A = 1.0 - (z_out ** 2 + z_in ** 2)
        resid = dA_over_A - z_ph ** 2
        print(f"   {eps_val:>9.0e} | {z_out:>12.8f} | {z_in:>12.8f} | "
              f"{z_ph:>12.8f} | {sum_sq:>9.6f} | {resid:>14.2e}")
        assert abs(sum_sq - 1.0) < 1e-9,  "sum of squares must equal 1"
        assert abs(resid) < 1e-9,         "dA/A_Sch must equal z_phantom^2"
        assert z_ph < 0.0,                "third root must be negative (unphysical)"

    print("\n   => Confirmed: sum of squares = 1 and dA/A_Sch = z_phantom^2 exactly.")
    print("      The third root is negative (no horizon); this is algebraic")
    print("      bookkeeping for the area deficit, not a new conservation law.")
    print("=" * 78)


if __name__ == "__main__":
    main()
