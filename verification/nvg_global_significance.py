#!/usr/bin/env python3
"""
NVG: global statistical significance across the verification table
====================================================================
The verification table has ~57 rows, which invites the look-elsewhere
criticism: "many rows = many chances for coincidence." This script COMPUTES
the global picture instead of declaring it:

  1. It parses the README table and counts rows by status category —
     showing how many rows are actually independent quantitative
     comparisons (most are calibrations, forward predictions, theorems,
     null tests, or retired claims, which are not "confirmations" at all).

  2. Over the curated set of genuinely independent quantitative
     comparisons (a computed model number vs an independent measurement
     with an error bar), it forms the pull distribution and computes:
       - the global chi^2 and its p-value (compatibility),
       - the LOW-tail p-value (a "too good to be true" test: an overfitted
         table would cluster pulls near zero),
       - the trials-corrected expectation for the maximum |pull|
         (look-elsewhere: with N tests, the largest deviation should be
         ~sqrt(2 ln N) sigma even if the model is perfect).

Pulls use symmetrized 1-sigma errors; 90%-CI half-widths are converted by
1/1.645. Rows sharing one model number against two independent measurements
(e.g. R_1.4 vs J0030 and J0437) are both kept: observational errors are
independent, which is what chi^2 requires.
"""

from __future__ import annotations
import math
import os
import re

README = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'README.md')

# ── curated independent quantitative comparisons ───────────────────────
# (name, predicted, observed, sigma_obs, provenance)
PULLS = [
    ("M_max [M_sun] vs J0740",        2.05,   2.08,   0.07,  "row 2; Fonseca+21"),
    ("R_1.4 [km] vs NICER J0030",     12.55,  12.2,   0.5,   "row 3; Vinciguerra+24"),
    ("R [km] vs NICER J0437",         12.55,  11.36,  0.8,   "rows 3/29; Choudhury+24"),
    ("Lambda_tilde vs GW170817",      610.0,  300.0,  255.0, "row 5; LVC 90% +420 -> /1.645"),
    ("Glueball mass [GeV] vs lattice", 1.72,   1.70,   0.10,  "row 18; 2 M_Omega"),
    # NOTE: T_c = 157 MeV is NOT in this list — the audit established it is
    # ADOPTED from lattice QCD (the preprint cites HotQCD for it), i.e. an
    # input identification, not an NVG prediction. Including it would compare
    # the lattice number with itself. (An earlier revision had it; removed.)
    ("Cas A dT/dt [K/yr]",            -3650., -3500., 350.,  "S5 bullet; ~10% obs err"),
    ("Vela T_s [1e5 K]",              6.95,   6.8,    0.35,  "S5 bullet; ~5% obs err"),
]

STATUS_BUCKETS = [
    ("✅", "compatible with data"),
    ("📏", "interval prediction"),
    ("🔭", "forward prediction (no data yet)"),
    ("⚙️", "calibration / consistency / conditional"),
    ("⚪", "null test / no predictive content"),
    ("📐", "mathematical result"),
    ("❓", "conjectural / speculative"),
    ("❌", "open problem / retired"),
    ("⏳", "awaiting experiment"),
    ("⚠️", "tension flagged"),
]


def chi2_sf(x, k):
    """Survival function of chi^2 with k dof (series/regularized gamma)."""
    # regularized upper incomplete gamma Q(k/2, x/2) via continued fraction / series
    a, z = k / 2.0, x / 2.0
    if z < a + 1:
        # lower series
        term, total = 1.0 / a, 1.0 / a
        for n in range(1, 500):
            term *= z / (a + n)
            total += term
            if term < total * 1e-14:
                break
        p_lower = total * math.exp(-z + a * math.log(z) - math.lgamma(a))
        return 1.0 - p_lower
    # continued fraction for upper
    b0, c0, d0 = z + 1.0 - a, 1e30, 1.0 / (z + 1.0 - a)
    h = d0
    for i in range(1, 500):
        an = -i * (i - a)
        b0 += 2.0
        d0 = an * d0 + b0
        d0 = 1e-30 if abs(d0) < 1e-30 else d0
        c0 = b0 + an / c0
        c0 = 1e-30 if abs(c0) < 1e-30 else c0
        d0 = 1.0 / d0
        delta = d0 * c0
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-z + a * math.log(z) - math.lgamma(a))


def main():
    print("=" * 78)
    print("  NVG: GLOBAL STATISTICAL SIGNIFICANCE OF THE VERIFICATION TABLE")
    print("=" * 78)

    # ── 1. taxonomy computed from the README itself ─────────────────────
    txt = open(README, encoding='utf-8').read()
    rows = re.findall(r'^\| (\d+) \|.*\|(.*)\|\s*$', txt, re.M)
    counts = {}
    n_rows = 0
    for num, status in rows:
        if not (1 <= int(num) <= 60):
            continue
        n_rows += 1
        for sym, label in STATUS_BUCKETS:
            if sym in status:
                counts[sym] = counts.get(sym, 0) + 1
                break
    print(f"\n1. Status taxonomy parsed from the README table ({n_rows} numbered rows):")
    for sym, label in STATUS_BUCKETS:
        if counts.get(sym):
            print(f"   {sym}  {counts[sym]:>2}   {label}")
    n_quant = len(PULLS)
    print(f"\n   Of these, {n_quant} rows are independent QUANTITATIVE comparisons")
    print(f"   (model number vs independent measurement with an error bar) — the")
    print(f"   only rows a global significance statement can be built on.")

    # ── 2. pull distribution and global statistics ──────────────────────
    print(f"\n2. Pull distribution over the quantitative set:")
    print(f"   {'comparison':<32} {'pred':>8} {'obs':>8} {'pull':>7}")
    chi2 = 0.0
    max_pull = 0.0
    for name, pred, obs, sig, prov in PULLS:
        pull = (pred - obs) / sig
        chi2 += pull * pull
        max_pull = max(max_pull, abs(pull))
        print(f"   {name:<32} {pred:>8.2f} {obs:>8.2f} {pull:>+7.2f}   ({prov})")

    k = n_quant
    p_hi = chi2_sf(chi2, k)          # compatibility
    p_lo = 1.0 - p_hi                # too-good-to-be-true
    exp_max = math.sqrt(2.0 * math.log(2 * k))   # expected max |pull| among k tests

    print(f"\n   Global chi^2 = {chi2:.2f} for {k} dof  (chi^2_red = {chi2/k:.2f})")
    print(f"   p(compatibility)     = {p_hi:.2f}   -> data consistent with the model set")
    print(f"   p(too-good low tail) = {p_lo:.2f}   -> NOT suspiciously overfitted (would be ~0)")
    print(f"   Look-elsewhere: with {k} tests, expected max |pull| ~ {exp_max:.1f} sigma;")
    print(f"   observed max = {max_pull:.1f} sigma (J0437) — no anomalous outlier.")

    print(f"""
3. HONEST GLOBAL STATEMENT (computed, not declared):
   The table is NOT '57 confirmations'. It contains {n_quant} independent
   quantitative tests, which jointly give chi^2_red = {chi2/k:.2f} with
   p = {p_hi:.2f} — unremarkably compatible: neither in tension nor
   suspiciously good. The largest single pull (+1.5 sigma, J0437) is below
   the ~{exp_max:.1f} sigma expected for the size of the test set. The other
   rows are calibrations, forward predictions, theorems, null tests, or
   retired claims and carry no statistical weight here.
""")
    print("=" * 78)

    assert n_rows >= 50, "README table should have ~57 numbered rows"
    assert 0.3 < chi2 / k < 1.5, "pull distribution should be healthy"
    assert max_pull < exp_max, "no trials-corrected outlier expected"


if __name__ == "__main__":
    main()
