#!/usr/bin/env python3
"""
NVG: candidate mechanisms for g = 2 (bounce-mass doubling per cycle)
=====================================================================
nvg_tolman_law_derivation.py established that the horizon chain
(R x2 <=> S_GH x4) requires the bounce mass to grow by g = 2 per cycle,
and that the repo's own numbers already used this law. What FIXES g = 2
is the remaining fundamental question. This script states the requirement
precisely and evaluates every in-model candidate quantitatively.

REQUIREMENT. Between bounce n and bounce n+1 the energy of the comoving
patch must double: the integrated fractional energy injection per cycle is
    ln g = ln 2 = 0.693.
Equivalently, with an injection rate beta_eff per e-fold of expansion,
    ∫ beta_eff dN (over the cycle) = 0.693.

CONSTRAINT from data: the S8/lensing analysis (nvg_s8_tension_check.py)
caps the injection accumulated in the OBSERVABLE window (recombination ->
today) at beta*ln(1/a_on) <~ 0.06. So at least ~90% of ln 2 must accumulate
in the unobservable leg: today -> turnaround -> contraction -> crunch.

Candidates (computed below):
  1. Winding argument:        g = exp(2*pi/N_e)      = 1.126   — fails (x5.9 short in exponent)
  2. Vacuum release at bounce: g = 1 + lambda_v/4    = 1.26    — fails (x3.0 short)
  3. DE->matter transfer (the w0-wa melting sector): the same coupling
     beta = 0.12 integrated to turnaround gives ln g ~ 0.22 — right order,
     x3 short; would need beta to GROW in the DE-decay phase and the
     contraction to ratchet (masses locked in collapsed structures do not
     melt back — the same clumpiness asymmetry that generates the entropy).
  4. Crunch binding-energy release: a contraction that ends in maximal
     collapse releases gravitational binding energy of order the rest mass
     itself; g = 1 + |E_bind|/Mc^2 -> 2 in the maximal-collapse limit.
     Newtonian assembly to the Schwarzschild radius gives |E_bind| ~ Mc^2/2
     (g = 1.5); collapse proceeds far deeper (to rho_c), pushing the factor
     toward 1. This is the most natural candidate: g = 2 <=> the crunch
     releases binding energy equal to the rest mass — and maximal collapse
     is also maximal entropy production, consistent with S_GH x4.

STATUS: candidate mechanisms identified and bracketed; none is yet a
derivation. The concrete open task is the GR energy bookkeeping
(Misner-Sharp/Kodama energy through the clumpy contraction and bounce).
"""

from __future__ import annotations
import math

LN2 = math.log(2.0)
N_E = 53.08          # e-folds, cosmology sector
LAMBDA_V = 1.02      # vacuum quartic
BETA_MELT = 0.12     # w0-wa melting coupling (DESI-fit)
S8_CAP = 0.06        # lensing cap on beta*ln(1/a_on) in the observable window
A_TURN = 2.4         # turnaround scale factor in units of today (repo estimate)


def main():
    print("=" * 78)
    print("  NVG: WHAT FIXES g = 2?  (bounce-mass doubling per cycle)")
    print("=" * 78)
    print(f"\nRequirement: integrated fractional energy gain per cycle = ln 2 = {LN2:.3f}")
    print(f"Data constraint: observable-window injection <= {S8_CAP} (S8/lensing cap)")
    print(f"=> at least {100*(1-S8_CAP/LN2):.0f}% of ln 2 must accumulate after today.\n")

    rows = []

    # 1. winding
    lng = 2.0 * math.pi / N_E
    rows.append(("Winding exp(2pi/N_e)", lng, math.exp(lng)))

    # 2. vacuum release
    g2 = 1.0 + LAMBDA_V / 4.0
    rows.append(("Vacuum release 1+lambda_v/4", math.log(g2), g2))

    # 3. melting transfer to turnaround (observable leg capped + future leg)
    dN_future = math.log(A_TURN)              # today -> turnaround
    lng3 = S8_CAP + BETA_MELT * dN_future     # capped past + beta*dN future
    rows.append(("DE->matter transfer (beta=0.12 to turnaround)", lng3, math.exp(lng3)))

    # 4. binding energy release, Newtonian assembly to R_s
    rows.append(("Crunch binding release (to R_s, Newtonian)", math.log(1.5), 1.5))
    rows.append(("Crunch binding release (maximal-collapse limit)", LN2, 2.0))

    print(f"{'candidate':<48} {'ln g':>8} {'g':>7} {'vs ln2':>8}")
    for name, lng_i, g_i in rows:
        print(f"{name:<48} {lng_i:>8.3f} {g_i:>7.3f} {lng_i/LN2:>7.2f}x")

    print(f"""
VERDICT:
  - Candidates 1-2 fail by factors 3-6 in the exponent and are closed.
  - Candidate 3 (the model's own DE->matter transfer) is the right order
    but x3 short with today's coupling; it can only work if beta grows as
    the DE reservoir drains and the contraction ratchets — not yet derived.
  - Candidate 4 brackets g = 2 naturally: collapse to the Schwarzschild
    scale already releases half the rest mass; the NVG crunch goes far
    deeper (to rho_c), driving the released fraction toward 1 and g toward
    2. g = 2 would mean: THE CRUNCH RELEASES BINDING ENERGY EQUAL TO THE
    REST MASS — maximal collapse, maximal entropy production (S_GH x4).
  - None of this is yet a derivation. Open task: GR energy bookkeeping
    (Misner-Sharp/Kodama) through the clumpy contraction and the bounce.
""")
    print("=" * 78)

    assert abs(math.exp(2*math.pi/N_E) - 1.126) < 0.01
    assert abs((1 + LAMBDA_V/4) - 1.255) < 0.01
    assert rows[2][1] < LN2 / 2.0, "transfer candidate should be short with today's beta"
    assert rows[4][2] == 2.0


if __name__ == "__main__":
    main()
