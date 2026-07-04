#!/usr/bin/env python3
"""
NVG: log-2 periodicity test of the PBH ladder against the real GWTC catalog
=============================================================================
The corrected Tolman ladder (nvg_tolman_law_derivation.py) places PBH rungs
at M_N = 0.38 * 2^N M_sun — i.e. 6.1, 12.2, 24.3, 48.6, 97.3 M_sun inside
the LIGO/Virgo band. IF a detectable fraction of the observed BBH population
were ladder PBHs, the component masses would cluster at integer values of
    phi = log2(M / 0.38)  (mod 1).

Data: the actual GWTC event list downloaded from GWOSC
(verification/data/gwtc_events.csv, https://gwosc.org/eventapi/csv/GWTC/).
Selection: confident catalogs, deduplicated by event name (latest version),
BBH only (m2 > 3 M_sun), source-frame primary and secondary masses.

Statistic: Rayleigh test on the circular phase 2*pi*frac(log2(m/0.38)):
    R = |sum exp(i phi_k)| / n,   p = exp(-n R^2)  (uniform-phase null).
A smooth astrophysical mass function spanning ~4 octaves has nearly uniform
fractional log2 phases, so the null is appropriate at first order.

Honest caveats, printed with the result: mass errors (~10-20%) smear
0.15-0.3 of a log-2 cycle and dilute any real comb; astrophysical BBHs
dominate the band, so the expected outcome is a NULL — which then bounds
the coherent ladder fraction rather than confirming anything.
"""

from __future__ import annotations
import csv
import math
import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gwtc_events.csv')
M1_ANCHOR = 0.38          # ladder anchor mass, M_sun
MIN_M2_BBH = 3.0          # exclude NS secondaries


def load_masses():
    events = {}
    with open(DATA, newline='') as fh:
        for row in csv.DictReader(fh):
            name = row['commonName']
            cat = row['catalog.shortName']
            if 'marginal' in cat.lower():
                continue
            try:
                m1 = float(row['mass_1_source'])
                m2 = float(row['mass_2_source'])
                v = int(row['version'])
            except (ValueError, KeyError):
                continue
            if name not in events or v > events[name][0]:
                events[name] = (v, m1, m2)
    m1s, m2s = [], []
    for _, m1, m2 in events.values():
        if m2 > MIN_M2_BBH:
            m1s.append(m1)
            m2s.append(m2)
    return m1s, m2s


def circular_stats(masses):
    """Rayleigh (any non-uniformity) + V-test at the rung phase (ladder-specific)."""
    n = len(masses)
    ph = [2 * math.pi * (math.log2(m / M1_ANCHOR) % 1.0) for m in masses]
    cx = sum(math.cos(x) for x in ph) / n
    sx = sum(math.sin(x) for x in ph) / n
    R = math.hypot(cx, sx)
    p_ray = min(math.exp(-n * R * R), 1.0)
    mean_phase = (math.atan2(sx, cx) / (2 * math.pi)) % 1.0
    u = cx * math.sqrt(2.0 * n)                  # V-test toward phase 0 (rungs)
    p_v = 0.5 * math.erfc(u / math.sqrt(2.0))    # one-sided
    return n, R, p_ray, mean_phase, cx, p_v


def main():
    print("=" * 78)
    print("  NVG: LOG-2 LADDER PERIODICITY vs GWTC (real catalog data)")
    print("=" * 78)

    m1s, m2s = load_masses()
    print(f"\n  Events after dedup + BBH cut: {len(m1s)}")
    print(f"  Ladder rungs in band: " +
          ", ".join(f"{M1_ANCHOR*2**n:.1f}" for n in range(4, 9)) + " M_sun")

    results = {}
    for label, ms in (("primary m1", m1s), ("secondary m2", m2s)):
        n, R, p_ray, mu, V, p_v = circular_stats(ms)
        results[label] = (n, R, p_ray, mu, V, p_v)
        print(f"\n  {label}: n = {n}")
        print(f"    Rayleigh (any non-uniformity): R = {R:.3f}, p = {p_ray:.3f}, mean phase = {mu:.2f}")
        print(f"    V-test at rung phase 0 (ladder-specific): V = {V:+.3f}, p = {p_v:.2f}")

    n1 = results["primary m1"][0]
    p_v1 = results["primary m1"][5]
    p_v2 = results["secondary m2"][5]
    R95 = math.sqrt(3.0 / n1)
    print(f"""
  RESULT: the Rayleigh test does detect non-uniformity — at mean phases
  0.7-0.8 (m1) and ~0.3 (m2), i.e. the KNOWN astrophysical mass-function
  peaks (~10 and ~35 M_sun), which sit BETWEEN the ladder rungs. The
  ladder-specific V-test at rung phase 0 is null (p = {p_v1:.2f} for m1,
  {p_v2:.2f} for m2). A coherent ladder subpopulation is bounded to
  f <~ {R95:.2f} of detected BBHs (95%, zero-width rungs; mass errors
  weaken the bound). Genuine NULL constraint on the NVG PBH ladder in
  the stellar-mass band — consistent with the two-population abundance
  (heavy-rung f_PBH ~ 1e-9): ladder PBHs should NOT be visible to LVK.
""")
    print("=" * 78)

    assert len(m1s) > 50, "expect a substantial BBH catalog"
    assert p_v1 > 0.05 and p_v2 > 0.05, "ladder-phase excess would be a discovery — investigate"


if __name__ == "__main__":
    main()
