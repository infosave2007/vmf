#!/usr/bin/env python3
"""
NVG Theory: deriving the Tolman growth law from turnaround dynamics
====================================================================
The repo postulated a "x4 entropy snowball" and wrote it in THREE mutually
inconsistent forms: S x4 per cycle, M_n = M_1 * 4^(n-1), and R_n = r_c * 2^(n-1).
This script derives which combinations are dynamically consistent.

Setup: modified Friedmann with curvature,
    H^2 = (8 pi G / 3) rho (1 - rho/rho_c) - k c^2 / a^2 ,
with every bounce at the same rho_c (fixed by M_Omega). Let g be the
per-cycle growth factor of the bounce mass/entropy budget
(M_bounce = rho_c V_b, S_matter ~ V_b T_b^3 with T_b fixed):
    a_b -> g^{1/3} a_b  per cycle.

Turnaround scaling (H = 0 at rho << rho_c, curvature term dominates):
    radiation (rho ~ a^-4):  rho_b a_b^4 / a_t^4 = k c^2/a_t^2  =>  a_t ~ a_b^2
    matter    (rho ~ a^-3):  rho_b a_b^3 / a_t^3 = k c^2/a_t^2  =>  a_t ~ a_b^3 ~ M

Hence per cycle:
    a_t -> g^{2/3} a_t   (radiation-dominated turnaround)
    a_t -> g       a_t   (matter-dominated turnaround)
and the Gibbons-Hawking horizon entropy S_GH ~ a_t^2 grows as g^{4/3} or g^2.

RESULT. The repo's horizon chain (R_n = r_c 2^{n-1}, S_GH x4 — the basis of
the N_e in [52.68, 53.38] interval, the H_0 bound and the 77-cycle count) is
SELF-CONSISTENT: a_t x2  <=>  S_GH x4 identically. What it fixes is g:
    matter-era turnaround (the realistic case):  g = 2
    radiation-era turnaround:                    g = 2*sqrt(2) ~ 2.83
Either way the mass law "M x4 per cycle" is EXCLUDED — with matter-era
dynamics M x4 would give a_t x4 and S_GH x16, destroying the horizon chain.

Confirmation from the repo's own numbers: the README quotes M ~ 10^56 g for
cycle 77. With M x2 per cycle: 0.38 M_sun * 2^76 = 5.7e55 g ~ 10^56 ✓.
With M x4: 0.38 M_sun * 4^76 ~ 4e78 g — off by 22 orders of magnitude.
The concrete numbers in the repo were already using the x2 law; only the
formulas said x4.

CONSEQUENCES of the corrected mass law (matter-era, g = 2):
  - GW comb tooth spacing: r_b ~ M^{1/3} => f-ratio 2^{-1/3} = 0.794
    (previously 4^{-1/3} = 0.63 under the wrong M x4);
  - earlier-tooth dilution: (a_b ratio)^4 = 2^{4/3} = 2.52 (was 6.35);
  - PBH ladder: M_N = 0.38 * 2^N M_sun (denser rungs; the JWST seed mass
    4e5 M_sun sits at rung N ~ 20 instead of N = 10);
  - holographic cycle count n = (122 - 76)/log10(4) ~ 77 — UNCHANGED,
    because it uses S_GH x4, which survives.

REMAINING OPEN: the microscopic mechanism that fixes g = 2 (equivalently,
why the turnaround horizon doubles). Candidates measured against it:
  winding argument exp(2 pi/N_e) = 1.126 — fails; vacuum-energy release
  1 + lambda_v/4 = 1.26 — fails. A mechanism delivering exactly x2 in the
  bounce mass per cycle is the concrete target.

Runtime: seconds. Numerical verification below solves H = 0 exactly.
"""

from __future__ import annotations
import math
import numpy as np

RHO_C = 1.0          # units: rho_c = 1, 8 pi G/3 = 1


def turnaround(a_b, w, K=1e-9):
    """Largest root of H^2(a) = rho(1-rho) - K/a^2 for rho = rho_b (a_b/a)^{3(1+w)}.
    rho_b is fixed slightly below rho_c so that the bounce (small root) exists."""
    p = 3.0 * (1.0 + w)
    rho = lambda a: RHO_C * (a_b / a) ** p
    f = lambda a: rho(a) * (1.0 - rho(a)) - K / a ** 2
    # bracket the turnaround: start beyond the bounce, expand until f < 0
    a_lo = a_b * 1.5
    assert f(a_lo) > 0, "no expansion phase — K too large"
    a_hi = a_lo
    while f(a_hi) > 0:
        a_hi *= 1.5
    for _ in range(200):
        mid = math.sqrt(a_lo * a_hi)
        if f(mid) > 0:
            a_lo = mid
        else:
            a_hi = mid
    return math.sqrt(a_lo * a_hi)


def main():
    print("=" * 78)
    print("  NVG: TOLMAN GROWTH LAW FROM TURNAROUND DYNAMICS")
    print("=" * 78)

    # ── Numerical verification of the scaling exponents ────────────────
    print("\n1. Turnaround scaling a_t(a_b), exact roots of H = 0:")
    for w, name, expo in ((1.0 / 3.0, "radiation", 2.0), (0.0, "matter", 3.0)):
        a1, a2 = 1.0, 2.0 ** (1.0 / 3.0)          # bounce radii for g = 2
        t1, t2 = turnaround(a1, w), turnaround(a2, w)
        measured = math.log(t2 / t1) / math.log(a2 / a1)
        print(f"   {name:9s}: a_t ~ a_b^{measured:.3f} (analytic exponent {expo:.0f}) "
              f"-> per-cycle a_t ratio = {t2 / t1:.4f}")
        assert abs(measured - expo) < 0.01, f"{name} exponent mismatch"

    # ── Consistency table ───────────────────────────────────────────────
    print("\n2. Per-cycle factors for entropy-generation factor g:")
    print(f"   {'g':>6} {'a_b':>7} {'a_t(mat)':>9} {'S_GH(mat)':>10} {'a_t(rad)':>9} {'S_GH(rad)':>10}")
    for g in (2.0, 2.0 * math.sqrt(2.0), 4.0):
        ab = g ** (1.0 / 3.0)
        print(f"   {g:6.2f} {ab:7.3f} {g:9.3f} {g*g:10.3f} "
              f"{g**(2./3.):9.3f} {g**(4./3.):10.3f}")
    print("   Horizon chain requires a_t x2 and S_GH x4:")
    print("   -> matter-era turnaround requires g = 2 (mass x2 per cycle).")
    print("   -> 'M x4 per cycle' would give a_t x4, S_GH x16 — EXCLUDED.")
    print("   NOTE: g = 2 is an EMPIRICAL POSTULATE chosen to fit the current")
    print("   Universe mass (~10^56 g) at cycle 77. It is NOT derived from microphysics.")

    # ── Repo self-consistency check ─────────────────────────────────────
    M1_g = 0.38 * 1.989e33
    m77_x2 = M1_g * 2.0 ** 76
    m77_x4 = M1_g * 4.0 ** 76
    print(f"\n3. Cross-check vs the repo's own cycle-77 mass (~1e56 g):")
    print(f"   M x2 law: 0.38 M_sun * 2^76 = {m77_x2:.1e} g  ~ 10^56 ✓")
    print(f"   M x4 law: 0.38 M_sun * 4^76 = {m77_x4:.1e} g  — off by 22 orders")
    assert 1e55 < m77_x2 < 1e57
    assert m77_x4 > 1e75

    # ── Corrected downstream constants ──────────────────────────────────
    print(f"\n4. Corrected downstream constants (matter-era, g = 2):")
    print(f"   GW comb tooth spacing : 2^(-1/3) = {2**(-1/3.):.3f}  (was 0.63 under M x4)")
    print(f"   Earlier-tooth dilution: 2^(4/3)  = {2**(4/3.):.2f}   (was 6.35)")
    print(f"   PBH ladder            : M_N = 0.38 * 2^N M_sun (JWST seed rung N ~ 20)")
    print(f"   Holographic cycle count (from S_GH x4): unchanged, ~77 cycles")

    print("\n5. OPEN: a microscopic mechanism fixing g = 2 per cycle.")
    print("   Candidates evaluated in nvg_g2_mechanism.py: winding (1.126) and vacuum")
    print("   release (1.26) fail. Until derived, g=2 remains an EMPIRICAL ANSATZ.")
    print("=" * 78)


if __name__ == "__main__":
    main()
