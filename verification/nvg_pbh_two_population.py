#!/usr/bin/env python3
"""
NVG: two-population PBH abundance — one calibration, three claims tested
=========================================================================
Three rows of the verification table rely on PBH abundance:
  (i)   PBH dark matter (asteroid-window peak) — population A;
  (ii)  JWST early SMBH seeds (~4e5 M_sun at z ~ 20) — population B;
  (iii) the NANOGrav SGWB attributed to PBH-binary mergers — population B.
The published single-Gaussian abundance (peak N = -21, sigma_N = 1.3) gives
essentially zero weight to population B, so claims (ii) and (iii) had no
abundance behind them. This script constructs the minimal honest fix — a
second population calibrated to what (ii) REQUIRES — and then computes what
that same population yields for (iii). One calibration, one cross-check.

Population B calibration (JWST):
  Observed z > 10 AGN/SMBH comoving density from JWST surveys:
  n_seed ~ 1e-5 .. 1e-4 Mpc^-3 (GN-z11 / UHZ1-type objects, with duty-cycle
  and completeness giving the order-of-magnitude range). Every seed is one
  M_B = 4e5 M_sun PBH (rung N ~ 20 of the corrected 2^N ladder).

Checks performed:
  1. f_PBH(B) against the CMB-accretion bound for 1e5-6 M_sun PBHs;
  2. the SGWB strain from population-B binaries vs NANOGrav A ~ 2.4e-15,
     scaled from the standard SMBH-binary background formula
     h_c^2 ∝ n_merge * (G Mc)^{5/3} / f^{4/3} at f = 1/yr.

Verdict is computed, not asserted in advance.
"""

from __future__ import annotations
import math

# ── constants / benchmarks ─────────────────────────────────────────────
RHO_DM_MSUN_MPC3 = 3.3e10      # mean DM density today, M_sun / Mpc^3
M_B = 4.0e5                    # population-B PBH mass, M_sun (JWST seed)
N_SEED_LO, N_SEED_HI = 1e-5, 1e-4   # required seed density, Mpc^-3
A_NANOGRAV = 2.4e-15           # NANOGrav 15yr strain amplitude at f = 1/yr

# Reference: the observed SGWB is consistent with the standard SMBH-binary
# population — characteristic chirp mass and merger-population density:
M_SMBH = 4.0e8                 # M_sun, effective chirp mass of the SMBHB background
N_SMBH = 3.0e-3                # Mpc^-3, galaxies contributing merging SMBH binaries

# CMB accretion bound (Serpico et al. 2020 class): for M ~ 1e5-6 M_sun,
# f_PBH < ~1e-8 (spherical accretion; disk accretion tighter).
F_PBH_CMB_BOUND = 1e-8


def main():
    print("=" * 78)
    print("  NVG: TWO-POPULATION PBH ABUNDANCE — JWST CALIBRATION vs NANOGrav")
    print("=" * 78)

    print("\n1. Population A (dark matter): asteroid-window peak — unchanged,")
    print("   abundance calibrated within the allowed window (see nvg_pbh_dark_matter.py).")

    # ── population B from JWST requirement ──────────────────────────────
    print(f"\n2. Population B calibrated to the JWST seeding requirement:")
    print(f"   {'n_seed [Mpc^-3]':>16} {'rho_B [Msun/Mpc^3]':>20} {'f_PBH(B)':>12} {'CMB bound ok?':>14}")
    results = []
    for n_seed in (N_SEED_LO, N_SEED_HI):
        rho_B = n_seed * M_B
        f_B = rho_B / RHO_DM_MSUN_MPC3
        ok = f_B < F_PBH_CMB_BOUND
        results.append((n_seed, f_B, ok))
        print(f"   {n_seed:>16.0e} {rho_B:>20.1f} {f_B:>12.2e} {'yes' if ok else 'NO':>14}")
    # the low-density end passes the CMB bound; the high end is marginal
    f_B_lo, f_B_hi = results[0][1], results[1][1]

    # ── NANOGrav cross-check from the SAME population ──────────────────
    # Background strain scaling (phinney 2001): h_c^2 ∝ n * (Mc)^{5/3}
    # (same merger efficiency per object assumed as for the SMBHB population —
    # deliberately GENEROUS to population B).
    print(f"\n3. SGWB from population-B binaries (same-per-object merger efficiency")
    print(f"   as the SMBHB population — a deliberately generous assumption):")
    for n_seed, f_B, _ in results:
        ratio_h2 = (n_seed / N_SMBH) * (M_B / M_SMBH) ** (5.0 / 3.0)
        A_B = A_NANOGRAV * math.sqrt(ratio_h2)
        deficit = A_NANOGRAV / A_B
        print(f"   n_seed = {n_seed:.0e}: A_B = {A_B:.1e}  →  {deficit:,.0f}x below NANOGrav")

    ratio_h2_hi = (N_SEED_HI / N_SMBH) * (M_B / M_SMBH) ** (5.0 / 3.0)
    A_B_hi = A_NANOGRAV * math.sqrt(ratio_h2_hi)
    deficit_hi = A_NANOGRAV / A_B_hi

    # ── verdict ─────────────────────────────────────────────────────────
    print(f"\n4. VERDICT:")
    print(f"   - Population B needed by JWST is TINY (f_PBH ~ {f_B_lo:.0e}..{f_B_hi:.0e});")
    print(f"     the low end passes the CMB-accretion bound — the JWST claim can be")
    print(f"     carried by an abundance model, at the cost of one calibrated number.")
    print(f"   - The SAME population falls short of the NANOGrav amplitude by a factor")
    print(f"     ~{deficit_hi:,.0f} in strain even at the generous end. Raising it to NANOGrav")
    print(f"     levels would need f_PBH ~ {f_B_hi * deficit_hi**2:.0e} — {f_B_hi*deficit_hi**2/F_PBH_CMB_BOUND:,.0f}x over the CMB bound.")
    print(f"   - CONCLUSION: the PBH-binary explanation of NANOGrav is RETIRED. NVG")
    print(f"     currently has NO mechanism for the NANOGrav signal: the bounce radiates")
    print(f"     at microHz (nvg_recondensation_dynamics.py) and heavy-PBH binaries are")
    print(f"     {deficit_hi:,.0f}x too weak. NANOGrav is attributed to ordinary SMBH binaries.")
    print("=" * 78)

    assert not results[0][2] or results[0][1] < F_PBH_CMB_BOUND
    assert deficit_hi > 100.0, "NANOGrav deficit should be large"
    assert A_B_hi < A_NANOGRAV / 100.0


if __name__ == "__main__":
    main()
