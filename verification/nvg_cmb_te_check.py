#!/usr/bin/env python3
"""
NVG: independent low-ell TE polarization check of the Genesis IR cutoff
========================================================================
The TT re-fit (nvg_cmb_lowl_refit.py) found the data mildly prefer the
sharp IR cutoff exp(-(k_c/k)^8) at k_c = 2.425e-4 Mpc^-1 (Delta chi^2 =
+1.77 over 2 <= ell <= 29). TE at low ell carries partially independent
information about the same primordial cut. This script computes the same
comparison for TE against the actual Planck 2018 TE spectrum
(verification/data/planck2018_te_full.txt, COM_PowerSpect_CMB-TE-full,
downloaded from IRSA).

Method: CAMB LCDM baseline (Planck 2018 parameters) vs the same cosmology
with the primordial P(k) multiplied by exp(-(k_c/k)^8); Gaussian chi^2
over 2 <= ell <= 30 using the published symmetrized errors (a first-pass
approximation — the exact low-ell TE likelihood is non-Gaussian; stated
honestly in the verdict).
"""

from __future__ import annotations
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TE_FILE = os.path.join(HERE, 'data', 'planck2018_te_full.txt')
K_C = 2.425e-4     # Mpc^-1, IR cutoff = 1/R_H0 (cosmology sector)
ELL_MAX = 30

PARS = dict(H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544,
            As=2.1e-9, ns=0.9649)


def get_te(cutoff: bool):
    import camb
    from camb import model
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=PARS['H0'], ombh2=PARS['ombh2'], omch2=PARS['omch2'],
                       tau=PARS['tau'])
    pars.InitPower.set_params(As=PARS['As'], ns=PARS['ns'])
    pars.set_for_lmax(200, lens_potential_accuracy=0)
    pars.NonLinear = model.NonLinear_none
    if cutoff:
        ks = np.logspace(-6, 0, 2000)
        pk = (PARS['As'] * (ks / 0.05) ** (PARS['ns'] - 1.0)
              * np.exp(-(K_C / ks) ** 8))
        pars.set_initial_power_table(ks, pk, effective_ns_for_nonlinear=PARS['ns'])
    res = camb.get_results(pars)
    cl = res.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
    return cl[:, 3]        # TE column, D_ell


def main():
    print("=" * 78)
    print("  NVG: LOW-ELL TE CHECK OF THE GENESIS IR CUTOFF (real Planck TE data)")
    print("=" * 78)

    data = np.loadtxt(TE_FILE)
    ells = data[:, 0].astype(int)
    sel = (ells >= 2) & (ells <= ELL_MAX)
    l_d, d_d, err = ells[sel], data[sel, 1], 0.5 * (data[sel, 2] + data[sel, 3])

    te_lcdm = get_te(False)
    te_cut = get_te(True)

    def chi2(model_dl):
        m = np.array([model_dl[l] for l in l_d])
        return float(np.sum(((d_d - m) / err) ** 2))

    c_lcdm = chi2(te_lcdm)
    c_cut = chi2(te_cut)
    dchi = c_lcdm - c_cut

    print(f"\n  TE chi^2 over 2 <= ell <= {ELL_MAX} ({len(l_d)} points, Gaussian approx):")
    print(f"    LCDM (no cutoff):   {c_lcdm:.2f}")
    print(f"    IR cutoff (sharp8): {c_cut:.2f}")
    print(f"    Delta chi^2 = {dchi:+.2f}  ({'cutoff preferred' if dchi > 0 else 'no preference'})")
    print(f"\n  Context: TT gave Delta chi^2 = +1.77 for the same cutoff. TE is a")
    print(f"  partially independent probe of the same primordial cut; a Gaussian")
    print(f"  low-ell TE likelihood is approximate, so treat |Delta chi^2| < 1 as")
    print(f"  indistinguishable. Either way this is a ~1-sigma-scale diagnostic,")
    print(f"  not a confirmation — stated for completeness of the cutoff test.")
    print("=" * 78)

    assert abs(dchi) < 20.0, "TE low-ell shift should be modest"


if __name__ == "__main__":
    main()
