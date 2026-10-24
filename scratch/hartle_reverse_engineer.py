#!/usr/bin/env python3
"""Reverse-engineer the vacuum Hartle l=2 O(omega^2) equations exactly.

The exterior Hartle-Thorne solution families are known in closed form:
  J^2 family (particular): h2 = J^2 (1/(M r^3) + 1/r^4),  v2 = J^2 / r^4
  K family (homogeneous):   h2 = Q_2^2(x),  v2 = 2M Q_2^1(x)/sqrt(r(r-2M)),
                            x = r/M - 1
with the vacuum first-order solution omega_tilde = Omega - 2J/r^3,
omega_tilde' = 6J/r^4, j = 1, (j^2)' = 0, nu' = 2M/(r(r-2M)).

The K family must satisfy the homogeneous vacuum system
  dh2/dr = alpha(r) v2 + beta(r) h2,   dv2/dr = delta(r) h2
and the J^2 family must satisfy the full affine system
  dh2/dr = alpha v2 + beta h2 + b_h2(r),  dv2/dr = delta h2 + b_v2(r).
This script evaluates candidate transcriptions to machine precision.
"""

from __future__ import annotations

import numpy as np


def Q20(x: float) -> float:
    return 0.25 * (3.0 * x * x - 1.0) * np.log((x + 1.0) / (x - 1.0)) - 1.5 * x


def Q21_nocs(x: float) -> float:
    """(x^2-1)^{1/2} dQ20/dx, no Condon-Shortley phase."""
    s = np.log((x + 1.0) / (x - 1.0))
    return np.sqrt(x * x - 1.0) * (
        1.5 * x * s - (3.0 * x * x - 1.0) / (2.0 * (x * x - 1.0)) - 1.5
    )


def Q22_nocs(x: float) -> float:
    """(x^2-1) d^2Q20/dx^2, no Condon-Shortley phase."""
    s = np.log((x + 1.0) / (x - 1.0))
    return 1.5 * (x * x - 1.0) * s - 3.0 * x + 2.0 * x / (x * x - 1.0)


def dnum(f, x: float, h: float = 1.0e-6) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


def main() -> None:
    from scipy.special import lqmn

    print("=== scipy lqmn cross-check at x = 9 ===")
    lq = lqmn(2, 2, 9.0)[0]
    for i in range(3):
        for j in range(3):
            print(f"  lqmn[{i},{j}] = {lq[i, j]: .12e}")
    print(f"  Q20(9)      = {Q20(9.0): .12e}")
    print(f"  Q21_nocs(9) = {Q21_nocs(9.0): .12e}")
    print(f"  Q22_nocs(9) = {Q22_nocs(9.0): .12e}")

    print("\n=== K-family homogeneous vacuum test ===")
    for M in (1.0, 1.5):
        for r in (10.0, 30.0):
            x = r / M - 1.0
            nu_p = 2.0 * M / (r * (r - 2.0 * M))
            S = np.sqrt(r * (r - 2.0 * M))
            for phase in (+1.0, -1.0):
                h2K = Q22_nocs(x)
                v2K = phase * 2.0 * M * Q21_nocs(x) / S
                dv2K = dnum(lambda rr: phase * 2.0 * M * Q21_nocs(rr / M - 1.0)
                            / np.sqrt(rr * (rr - 2.0 * M)), r)
                for name, delta in (("-2nu'", -2.0 * nu_p), ("-nu'", -1.0 * nu_p)):
                    ratio = dv2K / (delta * h2K) if delta * h2K != 0 else np.nan
                    print(f"  M={M} r={r} phase={phase:+.0f} delta={name:>6s}"
                          f" dv2K={dv2K: .6e} delta*h2K={delta*h2K: .6e}"
                          f" ratio={ratio: .9f}")

    print("\n=== J^2-family affine vacuum test (b_v2) ===")
    for M in (1.0, 1.5):
        for r in (10.0, 30.0):
            J = 0.7
            nu_p = 2.0 * M / (r * (r - 2.0 * M))
            v2J = J * J / r**4
            dv2J = -4.0 * J * J / r**5
            # candidate sources:  A: nu' * (1/6) r^4 wt'^2   B: (1/r+nu') * (1/6) r^4 wt'^2
            wtp2 = (6.0 * J / r**4) ** 2
            srcA = nu_p * (1.0 / 6.0) * r**4 * wtp2
            srcB = (1.0 / r + nu_p) * (1.0 / 6.0) * r**4 * wtp2
            for name, delta in (("-2nu'", -2.0 * nu_p), ("-nu'", -1.0 * nu_p)):
                h2J = J * J * (1.0 / (M * r**3) + 1.0 / r**4)
                need = dv2J - delta * h2J
                print(f"  M={M} r={r} delta={name:>6s} need_b={need: .9e}"
                      f" srcA={srcA: .9e} srcB={srcB: .9e}"
                      f" A/need={srcA/need: .9f} B/need={srcB/need: .9f}")

    print("\n=== K-family in the h2 equation: solve alpha, beta ===")
    # two sample radii, two unknowns alpha, beta per (M, phase):
    # dh2K/dr = alpha v2K + beta h2K  (homogeneous)
    for M in (1.0,):
        x1, x2 = 9.0, 29.0
        r1, r2 = M * (x1 + 1.0), M * (x2 + 1.0)
        for phase in (+1.0, -1.0):
            def h2K_of_r(rr: float) -> float:
                return Q22_nocs(rr / M - 1.0)

            def v2K_of_r(rr: float) -> float:
                return phase * 2.0 * M * Q21_nocs(rr / M - 1.0) / np.sqrt(rr * (rr - 2.0 * M))

            rows = []
            for rr in (r1, r2):
                dh = dnum(h2K_of_r, rr)
                rows.append((dh, h2K_of_r(rr), v2K_of_r(rr)))
            # [h2K_i, v2K_i] . (beta, alpha) = dh_i
            mat = np.array([[rows[i][1], rows[i][2]] for i in range(2)])
            rhs = np.array([rows[i][0] for i in range(2)])
            sol = np.linalg.solve(mat, rhs)
            beta, alpha = sol
            # verify at a third radius
            r3 = 50.0
            pred = beta * h2K_of_r(r3) + alpha * v2K_of_r(r3)
            exact = dnum(h2K_of_r, r3)
            print(f"  M={M} phase={phase:+.0f}: alpha={alpha: .9f} beta={beta: .9f}"
                  f" | r=50 check: pred={pred: .6e} exact={exact: .6e}"
                  f" relerr={abs(pred-exact)/abs(exact):.2e}")
            # compare with candidate forms
            for rr in (r1, r2, r3):
                nu_p = 2.0 * M / (rr * (rr - 2.0 * M))
                cand_alpha_ums = -2.0 / (rr * (rr - 2.0 * M) * nu_p)
                print(f"    r={rr}: alpha={alpha: .6e} cand_ums={cand_alpha_ums: .6e}")


if __name__ == "__main__":
    main()
