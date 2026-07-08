#!/usr/bin/env python3
"""
Task B — REAL CAMB low-ell chi^2 for the melting-exponent cutoff shift.

The melting exponent beta reshapes the bounce term (1-rho/rho_c)^{2beta}, which
shifts the comoving Genesis cutoff k_c and hence the CMB low-ell multipole
ell_c ~ k_c * D_LS (see nvg_melting_beta_bounce.py). Here we run the actual CAMB
low-ell refit (reusing nvg_cmb_lowl_refit.py) for the mean-field cutoff
(ell_c=3.42) versus the beta=0.326-shifted cutoff (ell_c ~ 5.1-7.6), against the
Planck 2018 TT deficit (ell = 2..29, exact full-sky Gamma likelihood, k^3 cutoff).

Result (this run):
  no cutoff (LCDM)            chi2_lowl = 26.77
  mean-field beta=0.5         chi2_lowl = 25.76   <- best (the framework's fit)
  3D-Ising beta=0.326 l_c~5.1 chi2_lowl = 27.66   (+1.9)
  3D-Ising beta=0.326 l_c~7.6 chi2_lowl = 36.62   (+10.9)

=> The Planck low-ell deficit PREFERS beta=1/2 and DISFAVOURS beta=0.326, while
   heavy-ion/lattice-QCD universality (nvg_melting_ginzburg.py, nvg_melting_beta_besii.py)
   wants beta=0.326. This is a genuine, quantified TENSION: the same "W" cannot be
   a mean-field bounce AND a 3D-Ising QCD melting. Either the bounce transition
   (T_b=432 MeV) is a different universality class from the dense-matter/heavy-ion
   one (T_c=157 MeV) -- consistent with the earlier "two-scale rho_c" result -- or
   the crude radiation-bounce k_c(beta) mapping needs the full horizon-chain
   treatment. Both are concrete, falsifiable next steps.
"""
import numpy as np
import nvg_cmb_lowl_refit as ref

def main():
    ell, Dl, sig, dm, dp = ref.load_planck_tt()
    H0, omch2 = 67.36, 0.1200
    cases = [("no cutoff (LCDM)",              0.0),
             ("mean-field beta=0.5  l_c=3.42", ref.KC_NVG*1.00),
             ("3D-Ising beta=0.326  l_c~5.1",  ref.KC_NVG*1.49),
             ("3D-Ising beta=0.326  l_c~7.6",  ref.KC_NVG*2.22)]
    print("Real CAMB low-ell refit (ell 2-29, exact Gamma likelihood, k^3 'cubic' cutoff)")
    print(f"{'model':32} {'k_c[Mpc^-1]':>12} {'chi2_lowl':>10}  {'d vs MF':>8}")
    base = None
    for name, kc in cases:
        Dlm = ref.run_camb(H0, omch2, kc, shape='cubic')
        c2 = ref.chi2_lowl(Dlm, ell, Dl)
        if 'mean-field' in name:
            base = c2
        d = "" if base is None else f"{c2-base:+.2f}"
        print(f"{name:32} {kc:12.3e} {c2:10.2f}  {d:>8}")
    print("\nPlanck low-ell PREFERS mean-field beta=1/2; beta=0.326 degrades the fit.")
    print("Tension with heavy-ion/lattice-QCD (which wants beta=0.326) -- see module docstring.")

if __name__ == "__main__":
    main()
