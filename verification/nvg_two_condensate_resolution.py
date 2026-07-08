#!/usr/bin/env python3
"""
Two-condensate HYPOTHESIS for the melting-exponent tension.

This is a physically-plausible MODEL INPUT, not a derived resolution: the split below
is assumed, and the NS calculation only tests its self-consistency, it does not derive it.

The tension (nvg_melting_beta_cmb_chi2.py + nvg_melting_ginzburg.py): one melting
law W(rho) with one exponent beta cannot be both a mean-field cosmological bounce
(CMB prefers beta=0.5) and a 3D-Ising QCD melting (heavy-ion/lattice want 0.326).

Proposed picture (speculative, assumed here): the bounce and the dense-matter/heavy-ion
transitions MIGHT involve DIFFERENT condensates at DIFFERENT scales, so no single beta
would be required.
  * deep vacuum condensate  W_deep -- rho_c^cosmo = 7.09e4 MeV/fm^3, mean-field
    beta=0.5, governs the bounce/CMB (l_c=3.42 fit).
  * dense-matter / chiral condensate -- critical near a few n0, 3D-Ising beta=0.326,
    probed by RHIC BES and (as its low-density tail) by neutron stars.

Two computed checks (both INPUT-DEPENDENT -- see caveats in the output):
  1. WEAK COUPLING (NOT decoupling): the melting law W=(1-rho/rho_c)^beta linearizes
     for rho<<rho_c to W ~ 1 - beta*(rho/rho_c), so its density slope dW/drho ~
     -beta/rho_c is PROPORTIONAL to beta. beta_deep (0.5 vs 0.326, a ~53% slope
     difference) therefore FORMALLY DOES enter the deep condensate's contribution to
     the NS EOS. That contribution is numerically negligible (~1% of a subdominant
     term) ONLY because the ASSUMED rho_c^cosmo=7.09e4 is ~65x the NS core energy
     density (eps_c/rho_c^cosmo=0.0153, ~0.8% melt). It is weak coupling from an
     assumed scale separation, not a genuine decoupling.
  2. NS density reach: neutron-star cores sit BELOW any melting critical point, so
     NS cannot pin the exponent of EITHER condensate (consistent with the
     identifiability study). This is agnostic about the two-condensate hypothesis --
     it neither confirms nor requires the split.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import nvg_eos_beta_saturated_vector as base
import nvg_tidal_deformability as td

n_0 = base.n_0; W0 = base.M_Omega_0; M_N = 939.0
RHO_C_COSMO = 7.09e4   # MeV/fm^3

def set_g(g): base.M_Omega = lambda n_b, k1, k2, _g=g: W0*_g(max(n_b/n_0, 0.0))

def central_density_of_Mmax():
    """peak baryon density in the maximum-mass star (n0 units), canonical EOS."""
    set_g(lambda x: (1.0+0.80*x)**(-0.25/0.80))
    eos = td.EOS(p_match=1.5, Gamma=1.35)
    best = (0.0, 0.0)
    for pc in np.logspace(-1.0, 3.4, 90):
        m, r, k2, lam = td.solve_tov_tidal(eos, pc)
        if m > best[0]: best = (m, eos.get_eps(pc))
    # convert central energy density to n_B/n0 via the tabulated EOS (n ~ eps/M_N crude)
    eps_c = best[1]
    return best[0], eps_c

print("="*74)
print("TWO-CONDENSATE RESOLUTION OF THE beta TENSION")
print("="*74)

Mmax, eps_c = central_density_of_Mmax()
n_c_over_n0 = (eps_c / M_N) / n_0          # OVERESTIMATE: uses ENERGY density, not baryon number
print(f"\nMaximum-mass star: M_max={Mmax:.2f} M_sun, central eps ~ {eps_c:.0f} MeV/fm^3")
print(f"  -> n_B ~ (eps/M_N)/n0 ~ {n_c_over_n0:.1f} n0  [OVERESTIMATE: eps is ENERGY density and")
print(f"     E/A >> M_N at the core, so the true central baryon density is ~5-6 n0]")

# 1. WEAK COUPLING (not decoupling): how much does the DEEP vacuum condensate melt?
W_deep = math.sqrt(max(1.0 - eps_c/RHO_C_COSMO, 0.0))    # sqrt(1-rho/rho_c^cosmo)
print(f"\n(1) WEAK COUPLING -- deep vacuum W_deep = sqrt(1 - rho/rho_c^cosmo):")
print(f"    rho_c^cosmo = {RHO_C_COSMO:.2e} MeV/fm^3;  eps_c/rho_c^cosmo = {eps_c/RHO_C_COSMO:.4f}")
print(f"    W_deep at the NS core = {W_deep:.4f}  -> only {100*(1-W_deep):.1f}% melted (marginal).")
print(f"    NOTE: W=(1-rho/rho_c)^beta linearizes to W ~ 1 - beta*(rho/rho_c), so its slope")
print(f"    dW/drho ~ -beta/rho_c is PROPORTIONAL to beta -- beta_deep (0.5 vs 0.326) DOES")
print(f"    FORMALLY enter the deep condensate's M* contribution. It is negligible (~1% of a")
print(f"    subdominant term) ONLY because the ASSUMED rho_c^cosmo is ~65x the core density;")
print(f"    this is WEAK COUPLING from a scale separation, NOT a decoupling. It FLIPS to")
print(f"    W_deep~0.92 (~8% melt, coupling ON) if rho_c^cosmo were ~10x lower.")

# 2. NS density reach vs the critical densities
print(f"\n(2) NS density reach vs the melting critical points:")
print(f"    NS core         ~ {n_c_over_n0:.1f} n0   ({eps_c:.0f} MeV/fm^3)  [E-density overest.; true ~5-6 n0]")
print(f"    chiral CP (est) ~ 3-5 n0                 (few x 10^2 MeV/fm^3)")
print(f"    deep vacuum rho_c^cosmo = {RHO_C_COSMO/(M_N*n_0):.0f} n0  ({RHO_C_COSMO:.1e} MeV/fm^3)")
print(f"    -> the NS core reaches at most the chiral tail; the deep vacuum CP is")
print(f"       ~{RHO_C_COSMO/eps_c:.0f}x beyond it. NS samples only tails, never a critical exponent.")

print("\n" + "="*74)
print("VERDICT (hypothesis assessment, NOT a derived resolution)")
print("="*74)
print(" * The deep vacuum condensate is only marginally melted (~0.8%, EOS-dependent) at")
print("   neutron-star densities, so beta_deep couples only WEAKLY into the NS EOS -- it is")
print("   NOT decoupled. This marginal result follows from the ASSUMED ~65x scale separation")
print("   (a speculative CMB low-ell fit rho_c^cosmo=7.09e4), NOT from NS physics, and it")
print("   FLIPS (W_deep~0.92, ~8% melt) if rho_c^cosmo were ~10x lower.")
print(" * The two-condensate split is a physically-plausible HYPOTHESIS / model input that")
print("   WOULD remove the tension IF true (it is not derived here):")
print("     - deep vacuum   : rho_c^cosmo = 7.09e4,  beta=0.5   -> CMB bounce (l_c=3.42)")
print("     - chiral/dense  : rho_c ~ few n0,        beta=0.326 -> RHIC BES-II (+ NS tail)")
print("   This is the physical form of the earlier 'two-scale rho_c' (~150x) result.")
print(" * Neutron stars sit below both critical points, so they pin NEITHER exponent")
print("   (matches the identifiability study) -- consistent with either value.")
print("\n NOT AN INDEPENDENT PREDICTION: beta=0.326 (BES) and beta=0.5 (CMB) are the very")
print(" inputs used to MOTIVATE the two-condensate split, so recovering them is post-hoc")
print(" accommodation, not a falsifiable discriminating test. A genuine test would need a")
print(" third, independent probe of either condensate's exponent.")
