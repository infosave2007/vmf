#!/usr/bin/env python3
"""
*** SUPERSEDED (crude proxy) -- see nvg_melting_beta_horizon_chain.py. ***
This script evaluates the comoving Hubble radius R_H at an ARBITRARY melting-onset
density r* in {0.9, 0.99}; that point choice is what inflates ell_c to 5-7.6 and
motivates the "Planck ALSO constrains beta" reading below. The horizon-chain
treatment shows that reading does NOT hold: under the framework's committed
instanton cutoff ell_c is pinned to H_0 (a near-tautology), and the low-ell chi^2
gaps are sub-2 sigma anyway -- so the CMB does NOT constrain beta. Keep this file
only as the differential bounce-duration demonstration; ignore its CMB conclusion.

Task 1 — how the vacuum melting exponent beta feeds the cyclic bounce and the
CMB low-ell cutoff. The modified Friedmann bounce term is generalised from the
mean-field mean-field form (2 beta = 1) to (1 - rho/rho_c)^{2 beta}, where beta is
the order-parameter exponent (mean-field 0.5, 3D XY 0.349, 3D Ising 0.326).

Computed (differential, unit-free where possible):
  * bounce proper-time duration ratio vs mean field,
  * maximum comoving Hubble radius R_H ~ cutoff scale, hence the shift in
    k_c = 1/R_H and the CMB cutoff multipole ell_c ~ k_c * D_LS.
CAVEAT: the exact ell_c needs the full CAMB horizon-chain pipeline (not run
here); this reports the beta-DEPENDENCE (ratios) with explicit assumptions.
Reference (mean-field) values from the framework: ell_c = 3.42, k_c = 2.425e-4 Mpc^-1.
"""
import numpy as np
from scipy import integrate

D_LS = 13800.0          # Mpc, comoving distance to last scattering (framework value)
ELL_C_MF = 3.42         # framework cutoff multipole at 2beta=1
KC_MF = 2.425e-4        # Mpc^-1

BETAS = {"mean-field (sqrt-law)": 0.5, "3D XY": 0.349, "3D Ising": 0.326}

# radiation bounce: a ~ rho^{-1/4}; r = rho/rho_c in (0,1], bounce at r=1.
# H^2 ~ rho (1-r)^{2beta}  ->  |H| ~ r^{1/2}(1-r)^{beta};  a|H| ~ r^{1/4}(1-r)^{beta}
# comoving Hubble radius R_H = 1/(a|H|) ~ r^{-1/4}(1-r)^{-beta}.
def RH(r, beta):
    return r**(-0.25) * (1.0 - r)**(-beta)

# bounce proper time near the turnaround: dt = da/(a|H|) = ... integrate dr weight
# t_b ~ int_{r0}^{1} dr / (rho |H| (1+w)) ; with a~rho^-1/4, radiation w=1/3:
#   dt ~ dr * r^{-1} (1-r)^{-beta}   (up to r-independent const), integrand near r=1 ~ (1-r)^{-beta}
def bounce_time(beta, r0=0.5):
    f = lambda r: r**(-1.0) * (1.0 - r)**(-beta)
    val, _ = integrate.quad(f, r0, 1.0 - 1e-9, points=[1.0], limit=200)
    return val

print("="*74)
print("Task 1 — melting exponent beta -> bounce + CMB low-ell cutoff")
print("="*74)
print(f"\nGeneralised bounce term H^2 ~ rho (1 - rho/rho_c)^(2 beta):")

# cutoff scale is set at the melting-onset density r* (where W has dropped
# appreciably). Report the ell_c shift for a band of plausible r*.
print(f"\n{'beta':>18} {'2beta':>6}  bounce-time/MF   ell_c shift (r*=0.9 .. 0.99)")
tb_mf = bounce_time(0.5)
for name, b in BETAS.items():
    tb = bounce_time(b) / tb_mf
    # R_H(beta)/R_H(MF) at fixed r*  ->  k_c ratio = inverse  ->  ell_c ratio = inverse
    shifts = []
    for rstar in (0.9, 0.99):
        ratio_RH = RH(rstar, b) / RH(rstar, 0.5)          # (1-r*)^{-(beta-0.5)}
        ell_c = ELL_C_MF / ratio_RH                        # ell_c ~ 1/R_H
        shifts.append(ell_c)
    print(f"{name:>18} {2*b:6.3f}  {tb:12.3f}     {shifts[0]:5.2f} .. {shifts[1]:5.2f}")

print("\nReading:")
print(" * beta<1/2 makes H^2 fall LESS steeply toward the bounce, so the bounce is")
print("   sharper (shorter proper time) and the comoving Hubble radius at melting")
print("   onset is SMALLER -> k_c larger -> the CMB cutoff moves to HIGHER ell.")
print(f" * For 3D Ising (beta=0.326) the cutoff shifts from ell_c={ELL_C_MF} up to")
print(f"   ~{ELL_C_MF/(RH(0.9,0.326)/RH(0.9,0.5)):.1f}-{ELL_C_MF/(RH(0.99,0.326)/RH(0.99,0.5)):.1f}")
print("   depending on the melting-onset density r*.")
print(" * NAIVE CONSEQUENCE (SUPERSEDED): a larger ell_c would push the cutoff past the")
print("   quadrupole/octupole and seem to let Planck low-ell ALSO constrain beta. But the")
print("   ell_c shift here is a point-r* artefact: the horizon-chain treatment")
print("   (nvg_melting_beta_horizon_chain.py) shows ell_c is pinned to H_0 by the N_e")
print("   calibration and the low-ell chi^2 gaps are sub-2 sigma -> the CMB does NOT")
print("   constrain beta. Treat only the bounce-duration ratio above as robust.")
print("\nCAVEAT: exact ell_c requires re-running the CAMB pipeline (nvg_cmb_lowl_refit.py")
print("with camb installed) and the full horizon chain; the numbers above are the")
print("differential beta-dependence under the stated radiation-bounce scaling.")
