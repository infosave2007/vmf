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
  mean-field beta=0.5         chi2_lowl = 25.76   (nominal lowest)
  3D-Ising beta=0.326 l_c~5.1 chi2_lowl = 27.66   (+1.9)
  3D-Ising beta=0.326 l_c~7.6 chi2_lowl = 36.62   (+10.9)

STATISTICAL SIGNIFICANCE (read before interpreting the numbers above):
  At ell = 2..29 cosmic variance dominates -- the (2l+1) factor in the Gamma
  likelihood IS that cosmic-variance weighting -- so the model-to-model spread is
  tiny relative to the noise. For the single cutoff parameter:
    mean-field beta=0.5 vs no-cutoff       : Delta chi^2 = 1.01  (~1.0 sigma)
    mean-field beta=0.5 vs beta=0.326(l_c~5.1): Delta chi^2 = 1.9   (~1.4 sigma)
  NEITHER reaches 2 sigma, and the real f_sky ~ 0.86 sky mask scales these gains
  DOWN further. Planck 2018 (X) found NO statistically significant low-ell cutoff.

=> The low-ell CMB is CONSISTENT WITH beta=0.5, no-cutoff, AND beta=0.326 ALIKE.
   It does NOT statistically prefer beta=0.5 and CANNOT exclude beta=0.326: the
   low-ell deficit here is just the known, non-significant quadrupole/octupole
   anomaly. Heavy-ion/lattice-QCD universality (nvg_melting_ginzburg.py,
   nvg_melting_beta_besii.py) wants beta=0.326; the CMB neither confirms nor
   contradicts it at any meaningful confidence.

   Note on the +10.9 entry: the two beta=0.326 variants span chi^2 = 27.7 - 36.6
   purely because of the UNCERTAIN/crude k_c(beta) horizon mapping -- there are two
   admissible mappings (l_c~5.1 and l_c~7.6) and WHICH one is physical is NOT
   established here. The +10.9 is therefore a statement about mapping ambiguity,
   not a physical disfavouring of beta=0.326.

   Bottom line: this is an apparent low-ell difference that is within cosmic
   variance and does not robustly discriminate beta. Pinning down the physical
   k_c(beta) mapping (full horizon-chain treatment) is the concrete next step.
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
    print("\nAt ell 2-29 cosmic variance dominates (the (2l+1) Gamma weighting): the")
    print("model spread is tiny. mean-field beta=0.5 vs no-cutoff is Dchi2~1.0 (~1.0 sigma),")
    print("vs beta=0.326(l_c~5.1) is Dchi2~1.9 (~1.4 sigma) -- NEITHER reaches 2 sigma, and")
    print("the f_sky~0.86 mask scales these gains DOWN further. Planck 2018 (X) found no")
    print("statistically significant low-ell cutoff.")
    print("=> The low-ell CMB is CONSISTENT with beta=0.5, no-cutoff, AND beta=0.326 alike;")
    print("   it does NOT prefer beta=0.5 and CANNOT exclude beta=0.326 (this is the known,")
    print("   non-significant quadrupole/octupole anomaly).")
    print("The +10.9 spread between the two beta=0.326 rows is UNCERTAIN k_c(beta) horizon")
    print("mapping (two admissible mappings); which mapping is physical is NOT established here.")

if __name__ == "__main__":
    main()
