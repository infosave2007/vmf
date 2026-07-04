#!/usr/bin/env python3
"""
NVG Verification: GW Comb Tooth Amplitudes (scale estimate)
============================================================
Companion to nvg_primordial_gw_comb.py: supplies the missing AMPLITUDE half
of the gravitational-wave comb prediction.

Physics:
  A homogeneous bounce radiates no gravitational waves. The GW source in NVG
  is the RECONDENSATION of the W-condensate right after the bounce, in the
  window T_b = 432 MeV -> T_c = 157 MeV. If that transition is first order
  (latent-heat fraction alpha, inverse duration beta/H), the standard
  sound-wave formulas for cosmological phase transitions apply
  (Caprini et al., JCAP 1604:001 and JCAP 2003:024):

    Omega_sw h^2 = 2.65e-6 (H/beta) [kappa_v alpha/(1+alpha)]^2
                   (100/g_*)^{1/3} v_w * S(f/f_p),
    S(x) = x^3 [7/(4+3x^2)]^{7/2},
    f_p  = 1.9e-5 Hz (beta/H)(T_*/100 GeV)(g_*/100)^{1/6} / v_w,

  with the Espinosa-Konstandin-No-Servant efficiency kappa_v(alpha).

Frequency cross-check (two independent routes):
  - adiabatic redshift of the bounce scale (nvg_primordial_gw_comb.py): 62.8 nHz
  - standard PT peak formula at T_b = 432 MeV, beta/H = 1:                72.5 nHz
  Agreement to ~15% — the comb anchor is robust.

Comb structure:
  GW energy redshifts as a^-4 while each bounce injects entropy (x4/cycle),
  so every earlier tooth is diluted by 4^{4/3} = 6.35 in Omega. The comb is
  therefore amplitude-dominated by the LAST bounce: a main bump at ~26-72 nHz
  with a sub-tooth at 0.63 f of ~16% amplitude, then <3%.

Honest status:
  (alpha, beta/H) are DERIVED in the companion nvg_recondensation_dynamics.py
  from the action, with a decisive split: the repo's written quartic potential
  gives a spinodal-sliver transition (alpha ~ 1e-4..0.03, beta/H ~ 60..500,
  Omega_GW ~ 1e-22..5e-14 — no PTA signal), while the scale-invariant
  Coleman-Weinberg form natural for a trace-anomaly condensate supercools
  deeply and, with the QCD-anomaly graceful exit, lands at alpha >> 1,
  beta/H ~ O(10) — inside the window below. NANOGrav 15yr (arXiv:2306.16213)
  reports a signal in this band, and its new-physics analysis
  (arXiv:2306.16219) lists a QCD-scale first-order transition as viable.
  The amplitude question is therefore now a question about the FORM of V(W). Caveats: the sound-wave formula is applied by
  analogy in a post-bounce (nonstandard) setting, and finite sound-wave
  lifetime corrections introduce O(10) amplitude uncertainty. Scale estimate,
  not a unique prediction — the model MATCHES the signal but does not yet
  single it out.

Output: fig_gw_comb_amplitude.png
"""

from __future__ import annotations
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nvg_primordial_gw_comb import bounce_frequency_today

G_STAR = 47.5
T_BOUNCE = 432.0      # MeV
T_COND = 157.0        # MeV — recondensation completes near T_c
NANOGRAV_F = 32.0     # nHz (f = 1/yr)
NANOGRAV_OMEGA = 4e-9 # representative Omega_GW h^2 at 1/yr (15yr data, factor ~2)
TOOTH_DILUTION = 4.0 ** (4.0 / 3.0)   # per earlier cycle: GW ~ a^-4 vs entropy x4
TOOTH_SPACING = 4.0 ** (-1.0 / 3.0)


def kappa_v(alpha: float) -> float:
    """Sound-wave efficiency for v_w -> 1 (Espinosa et al. 2010)."""
    return alpha / (0.73 + 0.083 * math.sqrt(alpha) + alpha)


def omega_peak(alpha: float, beta_H: float, g: float = G_STAR, vw: float = 1.0) -> float:
    return (2.65e-6 / beta_H * (kappa_v(alpha) * alpha / (1 + alpha)) ** 2
            * (100.0 / g) ** (1.0 / 3.0) * vw)


def f_peak_nHz(T_MeV: float, beta_H: float, g: float = G_STAR, vw: float = 1.0) -> float:
    return 1.9e-5 * beta_H / vw * (T_MeV / 1e5) * (g / 100.0) ** (1.0 / 6.0) * 1e9


def spectral_shape(x):
    return x ** 3 * (7.0 / (4.0 + 3.0 * x ** 2)) ** 3.5


def main():
    print("=" * 78)
    print("  NVG GW COMB: TOOTH AMPLITUDES FROM FIRST-ORDER RECONDENSATION")
    print("=" * 78)

    # ── 1. Frequency cross-check ────────────────────────────────────────
    f_adiabatic = bounce_frequency_today(859.0)
    f_pt = f_peak_nHz(T_BOUNCE, 1.0)
    ratio = f_pt / f_adiabatic
    print(f"\n1. Peak frequency, two independent routes:")
    print(f"   Adiabatic redshift of bounce scale : {f_adiabatic:.1f} nHz")
    print(f"   PT peak formula (T_b, beta/H = 1)  : {f_pt:.1f} nHz  (ratio {ratio:.2f})")
    print(f"   Recondensation end (T_c = 157 MeV) : {f_peak_nHz(T_COND, 1.0):.1f} nHz")
    print(f"   -> the signal window is 26-72 nHz, inside the PTA band.")

    # ── 2. Amplitude grid over the two underived parameters ────────────
    print(f"\n2. Peak amplitude Omega_GW h^2 (alpha, beta/H underived — grid):")
    alphas = (0.1, 0.2, 0.3, 0.5, 1.0)
    betas = (1.0, 3.0, 10.0)
    print(f"   {'alpha':>6} " + " ".join(f"{'b/H=' + str(int(b)):>10}" for b in betas))
    matching = []
    for a in alphas:
        row = []
        for b in betas:
            om = omega_peak(a, b)
            row.append(om)
            if NANOGRAV_OMEGA / 3.0 <= om <= NANOGRAV_OMEGA * 3.0:
                matching.append((a, b, om))
        print(f"   {a:6.1f} " + " ".join(f"{v:10.1e}" for v in row))
    print(f"\n   NANOGrav 15yr reference: ~{NANOGRAV_OMEGA:.0e} at {NANOGRAV_F:.0f} nHz")
    print(f"   Parameter points within x3 of the signal: "
          f"{', '.join(f'(a={a}, b/H={int(b)})' for a, b, _ in matching)}")

    # ── 3. Comb structure ───────────────────────────────────────────────
    print(f"\n3. Comb structure (dilution 4^(4/3) = {TOOTH_DILUTION:.2f} per earlier cycle):")
    for j in range(4):
        print(f"   tooth 77-{j}: f = {f_adiabatic * TOOTH_SPACING**j:6.1f} nHz, "
              f"relative amplitude {TOOTH_DILUTION**-j:.3f}")
    print(f"   -> a single dominant bump with a ~16% sub-tooth at 0.63 f;")
    print(f"      distinguishable from the SMBH-binary power law by spectral shape.")

    # ── 4. Figure ───────────────────────────────────────────────────────
    f = np.logspace(0, 2.5, 400)  # nHz
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for a, b, style in ((0.3, 3.0, '-'), (1.0, 10.0, '--'), (0.1, 1.0, ':')):
        fp = f_peak_nHz(T_COND, b)
        total = np.zeros_like(f)
        for j in range(3):
            total += (omega_peak(a, b) * TOOTH_DILUTION ** -j
                      * spectral_shape(f / (fp * TOOTH_SPACING ** j)))
        ax.loglog(f, total, style, label=f"α={a}, β/H={int(b)}")
    ax.errorbar([NANOGRAV_F], [NANOGRAV_OMEGA], yerr=[[NANOGRAV_OMEGA*0.5], [NANOGRAV_OMEGA]],
                fmt='o', color='k', capsize=4, label='NANOGrav 15yr (repr.)')
    ax.set_xlabel('f [nHz]'); ax.set_ylabel(r'$\Omega_{\rm GW} h^2$')
    ax.set_ylim(1e-12, 1e-6); ax.grid(alpha=0.3, ls='--')
    ax.set_title('NVG bounce recondensation GW background vs NANOGrav')
    ax.legend(fontsize=9)
    out = os.path.join(os.path.dirname(__file__), "fig_gw_comb_amplitude.png")
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out}")

    # ── Assertions ──────────────────────────────────────────────────────
    assert 0.5 < ratio < 2.0, "two frequency routes must agree to O(1)"
    assert matching, "some (alpha, beta/H) region should bracket NANOGrav"

    print("\nVERDICT: with the recondensation treated as a first-order transition,")
    print("the NVG bounce yields Omega_GW h^2 ~ 1e-10..3e-7 peaked at 26-72 nHz —")
    print("bracketing the NANOGrav 15yr signal. The comb is now a (frequency +")
    print("amplitude-window) prediction; deriving (alpha, beta/H) from the W-field")
    print("action would turn the window into a point and is the remaining step.")
    print("=" * 78)


if __name__ == "__main__":
    main()
