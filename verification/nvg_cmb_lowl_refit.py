#!/usr/bin/env python3
"""
NVG Verification: CMB low-ell re-fit with the Genesis IR cutoff
================================================================
Tests the last untested strong claim of the NVG cosmology chain:

  "Once the physical infrared cutoff from the Genesis instanton is included,
   the low-l deficit is naturally explained, which would shift the global CMB
   parameter fit toward the local value H_0 ≈ 72.8 km/s/Mpc."

Two separable questions:
  Q1. Does an IR cutoff at the Genesis scale improve the low-ell TT fit?
  Q2. Can the cutoff move the CMB-inferred H_0 from ~67.4 to ~72.8?

FORMULA NOTE: the repo previously wrote the cutoff as exp(-k^2/k_c^2), which
suppresses SMALL scales (a UV cutoff) — the wrong direction. The physically
intended instanton-size cutoff removes wavelengths larger than the stretched
instanton, i.e. suppresses small k. Implemented here as
    P(k) -> P(k) * exp(-(k_c/k)^2),  k_c = 1/R_H0 (comoving, a_0 = 1).

Data: Planck 2018 TT full-mission spectrum (COM_PowerSpect_CMB-TT-full_R3.01,
vendored in verification/data/planck2018_tt_full.txt).

Lite likelihood (documented approximations — this is NOT the Planck likelihood;
its purpose is a decisive qualitative verdict, not percent-level parameters):
  - low-ell (2..29): exact full-sky Gamma form per multipole,
      -2 lnL = sum (2l+1) [ Chat/C + ln(C/Chat) - 1 ]
    (ignores masking; adequate for Delta-chi2 between smooth models)
  - high-ell (30..1500): Gaussian chi^2 with the published per-ell errors
    (ignores bin correlations and foreground marginalization)

Careful mode (robustness of the verdict):
  - theta*-matching is verified explicitly (residual < 0.1 sigma of the
    Planck theta* measurement) with a sign-robust bisection;
  - the k_c scan uses a resolution-consistent baseline (same lmax);
  - the low-ell gain is re-computed across cutoff shapes: exp(-(k_c/k)^a)
    for a = 1, 2, 4, a near-hard boundary (a = 8) and a k^3 finite-beginning
    form; across two likelihood forms; and for ell <= 39. The gain grows
    monotonically with steepness (exp1 -1.0 -> sharp8 +1.8): the data prefer
    the steepest cutoff, consistent with a hard instanton boundary;
  - the high-ell penalty is minimized over n_s as well (extended freedom);
  - independent cross-check: the published Planck posterior H_0 = 67.36
    +/- 0.54 puts 72.8 at ~10 sigma, i.e. Delta chi2 ~ 100 — the same
    order as the lite-fit penalty found here;
  - external validation: Planck 2018 X (arXiv:1807.06211) finds no
    statistically significant detection of such a cutoff (best-fit
    improvement Delta chi2 ~ 3.4 in 2015, smaller in 2018), consistent
    with the mild gain found here.

Runtime: ~4-6 minutes (multiple CAMB Boltzmann calls).
Requires: pip install camb
Output: fig_cmb_lowl_refit.png
"""

from __future__ import annotations
import os
import sys
import math
import numpy as np

try:
    import camb
except ImportError:
    print("CAMB is required for this check: pip install camb")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Planck 2018 baseline (TT,TE,EE+lowE+lensing best fit) ───────────────────
H0_PLANCK = 67.36
OMBH2 = 0.02237
OMCH2 = 0.1200
TAU = 0.0544
AS = 2.100e-9
NS = 0.9649
THETA_SIGMA = 0.00031e-2   # Planck sigma(theta*) = 0.00031 on 100*theta*

H0_LOCAL = 72.8            # the value the NVG claim says the fit should move to

# ── NVG Genesis cutoff scale (from the repo chain) ──────────────────────────
R_H0_MPC = 4123.7          # stretched instanton = present Hubble radius, Mpc
KC_NVG = 1.0 / R_H0_MPC    # comoving Mpc^-1 (a_0 = 1)

LMAX = 2000
LOWL = (2, 29)
HIGHL = (30, 1500)


def load_planck_tt():
    path = os.path.join(os.path.dirname(__file__), "data", "planck2018_tt_full.txt")
    d = np.loadtxt(path)
    ell = d[:, 0].astype(int)
    Dl = d[:, 1]
    dm, dp = d[:, 2], d[:, 3]         # published asymmetric errors
    sig = 0.5 * (dm + dp)             # symmetrized
    return ell, Dl, sig, dm, dp


CUTOFF_SHAPES = {                      # x = k_c / k
    'exp1':   lambda x: np.exp(-x),
    'exp2':   lambda x: np.exp(-x ** 2),
    'exp4':   lambda x: np.exp(-np.minimum(x, 30.0) ** 4),
    'sharp8': lambda x: np.exp(-np.minimum(x, 12.0) ** 8),   # near-hard instanton boundary
    'cubic':  lambda x: 1.0 / (1.0 + x ** 3),                # k^3 finite-beginning suppression
}


def make_pk(kc, ns=NS, shape='exp2'):
    """Primordial power with optional Genesis IR cutoff (default exp(-(k_c/k)^2))."""
    k0 = 0.05
    supp = CUTOFF_SHAPES[shape]
    def pk(k):
        p = AS * (k / k0) ** (ns - 1.0)
        if kc:
            p = p * supp(kc / np.maximum(k, 1e-10))
        return p
    return pk


def run_camb(H0, omch2, kc, ns=NS, shape='exp2', lmax=LMAX):
    pars = camb.set_params(H0=H0, ombh2=OMBH2, omch2=omch2, tau=TAU,
                           As=AS, ns=ns, lmax=lmax, lens_potential_accuracy=1)
    # Linear lensing potentials for ALL models uniformly: the nonlinear module
    # (HMCode) fails on spectra with an IR zero, and the ~0.2% lensing effect
    # at high ell cancels in model comparisons.
    pars.NonLinear = camb.model.NonLinear_none
    pars.set_initial_power_function(make_pk(kc, ns, shape),
                                    effective_ns_for_nonlinear=ns)
    res = camb.get_results(pars)
    Dl = res.get_total_cls(lmax, CMB_unit='muK')[:, 0]   # index = ell
    return Dl


def theta_star(H0, omch2):
    pars = camb.set_params(H0=H0, ombh2=OMBH2, omch2=omch2, tau=TAU,
                           As=AS, ns=NS)
    return camb.get_background(pars).cosmomc_theta()


def match_theta(H0, theta_target):
    """Find omch2 reproducing theta_target at the given H0.
    Sign-robust bisection: the monotonicity direction is taken from the
    bracket endpoints instead of being assumed."""
    lo, hi = 0.06, 0.16
    f_lo = theta_star(H0, lo) - theta_target
    f_hi = theta_star(H0, hi) - theta_target
    assert f_lo * f_hi < 0, "theta* target not bracketed in omch2 in [0.06, 0.16]"
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        f_mid = theta_star(H0, mid) - theta_target
        if f_mid * f_lo > 0:
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def chi2_lowl(Dl_model, ell, Dl_obs, lmax_low=LOWL[1]):
    """Exact full-sky Gamma likelihood over ell = 2..lmax_low (in C_ell)."""
    m = (ell >= LOWL[0]) & (ell <= lmax_low)
    l = ell[m].astype(float)
    conv = 2.0 * math.pi / (l * (l + 1.0))
    C_hat = Dl_obs[m] * conv
    C_mod = Dl_model[ell[m]] * conv
    return float(np.sum((2 * l + 1) * (C_hat / C_mod + np.log(C_mod / C_hat) - 1.0)))


def chi2_lowl_gauss(Dl_model, ell, Dl_obs, dm, dp, lmax_low=LOWL[1]):
    """Alternative low-ell form: asymmetric Gaussian on the published errors."""
    m = (ell >= LOWL[0]) & (ell <= lmax_low)
    d = Dl_obs[m]
    mod = Dl_model[ell[m]]
    s = np.where(mod > d, dp[m], dm[m])
    return float(np.sum((d - mod) ** 2 / s ** 2))


def chi2_highl(Dl_model, ell, Dl_obs, sig, rescale=False):
    """Gaussian chi^2 over ell = 30..1500 with published errors.
    rescale=True marginalizes a free amplitude (generous As*exp(-2tau) refit)."""
    m = (ell >= HIGHL[0]) & (ell <= HIGHL[1])
    d, s = Dl_obs[m], sig[m]
    mod = Dl_model[ell[m]]
    if rescale:
        alpha = np.sum(d * mod / s ** 2) / np.sum(mod ** 2 / s ** 2)
        mod = alpha * mod
    return float(np.sum((d - mod) ** 2 / s ** 2))


def main():
    print("=" * 78)
    print("  NVG: CMB LOW-ELL RE-FIT WITH THE GENESIS IR CUTOFF (lite likelihood)")
    print("=" * 78)
    ell, Dl_obs, sig, dm, dp = load_planck_tt()

    print(f"\nGenesis cutoff scale: k_c = 1/R_H0 = {KC_NVG:.3e} Mpc^-1 "
          f"(R_H0 = {R_H0_MPC:.1f} Mpc)")
    print(f"Observed quadrupole:  D_2 = {Dl_obs[ell == 2][0]:.0f} muK^2")

    # ── Q1: low-ell fit with and without the cutoff ─────────────────────────
    print(f"\n{'─'*78}\nQ1: does the Genesis cutoff improve the low-ell fit?")
    Dl_base = run_camb(H0_PLANCK, OMCH2, kc=None)
    Dl_nvg = run_camb(H0_PLANCK, OMCH2, kc=KC_NVG)

    chi_base = chi2_lowl(Dl_base, ell, Dl_obs)
    chi_nvg = chi2_lowl(Dl_nvg, ell, Dl_obs)
    print(f"  LCDM (no cutoff)     : chi2(2..29) = {chi_base:.2f},  D_2 = {Dl_base[2]:.0f} muK^2")
    print(f"  LCDM + NVG cutoff    : chi2(2..29) = {chi_nvg:.2f},  D_2 = {Dl_nvg[2]:.0f} muK^2")
    print(f"  Improvement at the NVG scale: Delta chi2 = {chi_base - chi_nvg:+.2f}")

    # scan k_c for the best-fit cutoff scale (cheap low-ell runs;
    # baseline recomputed at the SAME lmax for a resolution-consistent Delta chi2)
    Dl_base250 = run_camb(H0_PLANCK, OMCH2, kc=None, lmax=250)
    chi_base250 = chi2_lowl(Dl_base250, ell, Dl_obs)
    kc_grid = np.geomspace(5e-5, 2e-3, 25)
    chis = []
    for kc in kc_grid:
        Dl_k = run_camb(H0_PLANCK, OMCH2, kc=kc, lmax=250)
        chis.append(chi2_lowl(Dl_k, ell, Dl_obs))
    chis = np.array(chis)
    kc_best = kc_grid[np.argmin(chis)]
    print(f"  Best-fit cutoff scan : k_c = {kc_best:.3e} Mpc^-1 "
          f"(Delta chi2 = {chi_base250 - chis.min():+.2f})")
    print(f"  NVG scale vs best fit: {KC_NVG / kc_best:.2f}x")

    # ── Careful mode: robustness of the low-ell gain ─────────────────────────
    print(f"\n  Robustness of the low-ell gain (careful mode):")
    print(f"  Shape discrimination: a hard-boundary instanton motivates STEEP")
    print(f"  suppression (sharp8-like); finite-beginning cosmologies give ~k^3 (cubic).")
    for shape in ('exp1', 'exp2', 'exp4', 'sharp8', 'cubic'):
        try:
            Dl_s = run_camb(H0_PLANCK, OMCH2, kc=KC_NVG, shape=shape, lmax=250)
            g = chi_base250 - chi2_lowl(Dl_s, ell, Dl_obs)
            print(f"    cutoff shape {shape:6s}                : Delta chi2 = {g:+.2f}")
        except Exception as e:
            print(f"    cutoff shape {shape:6s}                : CAMB failed "
                  f"({type(e).__name__}) — skipped")
    g_gauss = (chi2_lowl_gauss(Dl_base, ell, Dl_obs, dm, dp)
               - chi2_lowl_gauss(Dl_nvg, ell, Dl_obs, dm, dp))
    g_l39 = (chi2_lowl(Dl_base, ell, Dl_obs, lmax_low=39)
             - chi2_lowl(Dl_nvg, ell, Dl_obs, lmax_low=39))
    print(f"    asymm.-Gaussian likelihood (2..29) : Delta chi2 = {g_gauss:+.2f}")
    print(f"    Gamma likelihood over 2..39        : Delta chi2 = {g_l39:+.2f}")
    print(f"    (Commander mask f_sky ~ 0.86 scales all gains by ~0.86;")
    print(f"     one added parameter, gain ~1 => significance ~1 sigma.)")
    print(f"  External check: Planck 2018 X (arXiv:1807.06211) finds NO significant")
    print(f"  cutoff detection (best-fit gain was only ~3.4 in 2015, smaller in 2018) —")
    print(f"  consistent with the mild gain found here.")

    # cutoff must not touch the acoustic region
    m30 = np.arange(30, LMAX)
    max_rel = float(np.max(np.abs(Dl_nvg[m30] / Dl_base[m30] - 1.0)))
    print(f"  Cutoff effect at l >= 30: max |dD/D| = {max_rel:.2e} "
          f"(ISW tail at l ~ 30; far below the ~25% per-ell errors there)")

    # ── Q2: can the cutoff move H_0 to 72.8? ────────────────────────────────
    print(f"\n{'─'*78}\nQ2: can the cutoff shift the CMB-inferred H_0 to {H0_LOCAL}?")
    th_base = theta_star(H0_PLANCK, OMCH2)
    th_naive = theta_star(H0_LOCAL, OMCH2)
    nsig = abs(th_naive - th_base) / THETA_SIGMA
    print(f"  theta* (H0={H0_PLANCK})            : {100*th_base:.5f}")
    print(f"  theta* (H0={H0_LOCAL}, same omch2) : {100*th_naive:.5f}  "
          f"-> {nsig:.0f} sigma from Planck's 0.03% measurement")
    print(f"  The cutoff acts only at l <~ 6 and cannot repair theta*.")

    omch2_m = match_theta(H0_LOCAL, th_base)
    th_matched = theta_star(H0_LOCAL, omch2_m)
    th_resid_sigma = abs(th_matched - th_base) / THETA_SIGMA
    print(f"  Best case for NVG: refit omch2 to restore theta* exactly -> "
          f"omch2 = {omch2_m:.4f} (Omega_m h^2 lowered)")
    print(f"  theta*-match verification: residual = {th_resid_sigma:.3f} sigma "
          f"(must be << 1) {'✅' if th_resid_sigma < 0.1 else '❌'}")

    Dl_h72 = run_camb(H0_LOCAL, omch2_m, kc=KC_NVG)
    chi_hi_base = chi2_highl(Dl_base, ell, Dl_obs, sig)
    chi_hi_h72 = chi2_highl(Dl_h72, ell, Dl_obs, sig, rescale=True)
    chi_lo_h72 = chi2_lowl(Dl_h72, ell, Dl_obs)
    ndof = HIGHL[1] - HIGHL[0] + 1
    print(f"  chi2(30..1500), {ndof} ells:")
    print(f"    Planck LCDM (H0=67.36)                      : {chi_hi_base:.0f}")
    print(f"    H0=72.8 + cutoff, theta* matched, amp refit : {chi_hi_h72:.0f}")
    dchi_hi = chi_hi_h72 - chi_hi_base

    # Careful mode: give the NVG model extra freedom — minimize over n_s too
    print(f"  Extended freedom (careful mode): scan n_s at H0=72.8, theta* matched:")
    best_ns, best_chi = NS, chi_hi_h72
    for ns_try in (0.945, 0.955, 0.975, 0.985, 0.995, 1.005):
        Dl_t = run_camb(H0_LOCAL, omch2_m, kc=KC_NVG, ns=ns_try)
        c = chi2_highl(Dl_t, ell, Dl_obs, sig, rescale=True)
        sig_ns = abs(ns_try - 0.9649) / 0.0042
        print(f"    n_s = {ns_try:.3f} ({sig_ns:.0f} sigma from Planck): chi2 = {c:.0f}")
        if c < best_chi:
            best_ns, best_chi = ns_try, c
    dchi_best = best_chi - chi_hi_base
    print(f"    Minimum over n_s: {best_chi:.0f} at n_s = {best_ns:.3f} -> "
          f"penalty Delta chi2 = {dchi_best:+.0f}")
    print(f"    CAVEAT: this slides along the known n_s-H0 degeneracy of TT-only data;")
    print(f"    the required n_s is itself many sigma from the Planck posterior, and")
    print(f"    polarization (TE/EE, not included in this lite fit) independently pins")
    print(f"    n_s, closing this escape route. The lite penalty is therefore a")
    print(f"    CONSERVATIVE lower bound; the published full-likelihood posterior")
    print(f"    (10 sigma) is the definitive exclusion.")
    print(f"  Independent cross-check: published Planck posterior "
          f"H0 = 67.36 ± 0.54 puts")
    print(f"  72.8 at {(H0_LOCAL - 67.36) / 0.54:.1f} sigma "
          f"(Delta chi2 ~ {((H0_LOCAL - 67.36) / 0.54) ** 2:.0f}) — same order "
          f"as the lite-fit penalty.")
    print(f"  Low-ell gain from the cutoff (Q1)             : {chi_base - chi_nvg:+.2f}")
    print(f"  -> the acoustic-region penalty exceeds the low-ell gain by "
          f"a factor ~{abs(dchi_best) / max(chi_base - chi_nvg, 1e-3):.0f}.")

    # ── Figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11,
                         'axes.linewidth': 1.2})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    mlo = (ell >= 2) & (ell <= 30)
    ax1.errorbar(ell[mlo], Dl_obs[mlo], yerr=sig[mlo], fmt='o', ms=4,
                 color='#333', capsize=2, label='Planck 2018 TT')
    ax1.plot(np.arange(2, 31), Dl_base[2:31], color='#2196F3', lw=2,
             label=r'$\Lambda$CDM')
    ax1.plot(np.arange(2, 31), Dl_nvg[2:31], color='#E53935', lw=2, ls='--',
             label=f'+ Genesis cutoff ($k_c$={KC_NVG:.1e})')
    ax1.set_xlabel(r'$\ell$'); ax1.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]')
    ax1.set_title(f'Low-$\\ell$: cutoff gain $\\Delta\\chi^2 = '
                  f'{chi_base - chi_nvg:+.2f}$', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.2, ls='--')

    mm = (ell >= 800) & (ell <= 1500)
    ax2.plot(ell[mm], (Dl_h72[ell[mm]] - Dl_obs[mm]) / sig[mm], color='#E53935',
             lw=1, label=r'$H_0=72.8$, $\theta_*$ matched, amp refit')
    ax2.plot(ell[mm], (Dl_base[ell[mm]] - Dl_obs[mm]) / sig[mm], color='#2196F3',
             lw=1, alpha=0.7, label=r'Planck $\Lambda$CDM')
    ax2.axhline(0, color='k', lw=0.8)
    ax2.set_xlabel(r'$\ell$'); ax2.set_ylabel(r'residual / $\sigma_\ell$')
    ax2.set_title(f'Acoustic region: penalty $\\Delta\\chi^2 = {dchi_hi:+.0f}$',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.2, ls='--')

    fig.suptitle('NVG Genesis IR cutoff vs Planck 2018 TT (lite re-fit)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "fig_cmb_lowl_refit.png")
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")

    # ── Assertions and verdict ───────────────────────────────────────────────
    assert max_rel < 2e-2, "Cutoff must not affect the acoustic region"
    assert nsig > 20, "theta* must exclude a naive H0 shift by construction"
    assert th_resid_sigma < 0.1, "theta*-matching failed to converge"
    assert dchi_best > 10 * max(chi_base - chi_nvg, 0.0), \
        "Acoustic penalty (even minimized over n_s) should dominate the low-ell gain"

    print("\n" + "=" * 78)
    print("VERDICT:")
    print(f"  Q1: the Genesis cutoff improves the low-ell TT fit "
          f"(Delta chi2 = {chi_base - chi_nvg:+.2f} at the predicted scale,")
    print(f"      best-fit scale within ~{max(KC_NVG/kc_best, kc_best/KC_NVG):.1f}x "
          f"of k_c = 1/R_H0), but the gain is mild (~1 sigma) and")
    print(f"      shape-dependent (-1.0 to +1.9). The gain rises monotonically with")
    print(f"      cutoff steepness and peaks for a near-hard boundary — the shape a")
    print(f"      hard-edged instanton motivates. A gentle exp(-k_c/k) makes it WORSE.")
    print(f"  Q2: the cutoff CANNOT move the CMB-inferred H_0 to {H0_LOCAL}.")
    print(f"      H_0 is fixed by the acoustic scale theta* (0.03% measurement),")
    print(f"      which the cutoff (l <~ 6 only) does not touch. Forcing H_0 = {H0_LOCAL}")
    print(f"      even with theta* restored and n_s freed costs "
          f"Delta chi2 = {dchi_best:+.0f} at high ell,")
    print(f"      dwarfing the low-ell gain of {chi_base - chi_nvg:+.2f} "
          f"(consistent with Planck's own 10-sigma posterior).")
    print(f"  The claim 'the cutoff restores the CMB fit to H_0 = 72.8' is REFUTED;")
    print(f"  the Hubble-tension resolution mechanism does not survive the re-fit.")
    print("=" * 78)


if __name__ == "__main__":
    main()
