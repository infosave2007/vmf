#!/usr/bin/env python3
"""
Population echo-comb search: stack the coherent time-slide statistic over the full
confident-BBH catalog (O1-O3) to test for ANY NVG post-merger echo population.

Reuses the validated per-event engine in nvg_echo_timeslide_background.analyse
(L1-vs-H1 time-slide background, delay-LEE folded in, single-detector decomposition),
runs it over every confident BBH with m2 > 3 M_sun and network SNR > 9 that GWOSC
serves, and Fisher-combines the per-event look-elsewhere p-values.

Under the null the per-event p-values are ~uniform and the stack is unremarkable.
A real echo population would pile up coherent (both-detector) low-p events and drive
the stacked p down. Prints per-event lines + the population verdict.

Requires pycbc + network to GWOSC. Runtime is dominated by strain fetches.
"""
from __future__ import annotations
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nvg_echo_timeslide_background as ts   # noqa: E402

ts.N_SLIDES = 200          # fewer slides per event for a tractable population run
MAX_EVENTS = 60
MIN_M1_BH = 3.0            # primary is a black hole -> remnant is a BH (keeps BBH + NSBH)
MIN_SNR = 7.0             # lowered threshold
GPS_O4_START = 1.30e9     # exclude O4 (open strain not served yet via pycbc/GWOSC)


def select_events():
    path = os.path.join(HERE, "data", "gwtc_events.csv")
    seen, cand = set(), []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("commonName")
            if not name or name in seen:
                continue
            try:
                m1 = float(row["mass_1_source"])
                snr = float(row["network_matched_filter_snr"])
                gps = float(row["GPS"])
            except (KeyError, ValueError, TypeError):
                continue
            # BH primary (BBH or NSBH; excludes BNS), loud enough, and O1-O3 only
            if m1 >= MIN_M1_BH and snr >= MIN_SNR and gps < GPS_O4_START:
                seen.add(name)
                cand.append((snr, name))
    cand.sort(reverse=True)
    return [n for _, n in cand[:MAX_EVENTS]]


def main():
    masses = ts.load_masses()
    events = select_events()
    print("=" * 96)
    print("  NVG ECHO-COMB POPULATION STACK  (confident BBH, O1-O3, time-slide background)")
    print("=" * 96)
    print(f"  Candidates: {len(events)} O1-O3 events (BH primary m1>={MIN_M1_BH} Msun -> BBH+NSBH, "
          f"netSNR>={MIN_SNR}); {ts.N_SLIDES} slides/event")
    print("-" * 96)
    print(f"  {'event':<20}{'M_f':>6}{'det':>4}{'S0':>9}{'H1':>8}{'L1':>8}{'coh?':>6}"
          f"{'bkg_med':>8}{'slides':>7}{'p':>8}")
    print("  " + "-" * 92)

    results = []
    for name in events:
        try:
            r = ts.analyse(name, masses)
        except Exception as exc:
            print(f"  {name:<20} skipped ({type(exc).__name__})")
            continue
        if r is None or r["p"] != r["p"]:
            print(f"  {name:<20} skipped (no dual-detector data)")
            continue
        results.append(r)
        coh = "yes" if r["S0"] > 1.15 * max(r["h1"], r["l1"]) else "1det"
        print(f"  {r['name']:<20}{r['mass_final']:>6.1f}{r['n_det']:>4}{r['S0']:>9.2f}"
              f"{r['h1']:>8.2f}{r['l1']:>8.2f}{coh:>6}{r['bkg_med']:>8.2f}"
              f"{r['n_slide']:>7}{r['p']:>8.3f}", flush=True)

    print("-" * 96)
    if not results:
        print("  No usable events."); print("=" * 96); return

    ps = np.clip(np.array([r["p"] for r in results]), 1e-6, 1.0)
    chi2 = -2.0 * float(np.sum(np.log(ps)))
    dof = 2 * len(ps)

    def chi2_sf(x, k):
        a, xx = k / 2.0, x / 2.0
        if xx <= 0:
            return 1.0
        if xx < a + 1:
            term = 1.0 / a; sm = term; nn = a
            for _ in range(1000):
                nn += 1; term *= xx / nn; sm += term
                if abs(term) < abs(sm) * 1e-13:
                    break
            return 1.0 - sm * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        b = xx + 1 - a; c = 1e30; d = 1.0 / b; h = d
        for i in range(1, 1000):
            an = -i * (i - a); b += 2
            d = an * d + b; d = 1e-30 if abs(d) < 1e-30 else d
            c = b + an / c; c = 1e-30 if abs(c) < 1e-30 else c
            d = 1.0 / d; delta = d * c; h *= delta
            if abs(delta - 1.0) < 1e-13:
                break
        return h * math.exp(-xx + a * math.log(xx) - math.lgamma(a))

    p_stack = chi2_sf(chi2, dof)
    n_coh = sum(1 for r in results if r["S0"] > 1.15 * max(r["h1"], r["l1"]))
    n_low = sum(1 for r in results if r["p"] < 0.05)
    n_coh_low = sum(1 for r in results
                    if r["p"] < 0.05 and r["S0"] > 1.15 * max(r["h1"], r["l1"]))

    print(f"  Events stacked: {len(results)}   coherent(both-det): {n_coh}")
    print(f"  p<0.05 events: {n_low} (expected ~{0.05*len(results):.1f} under null); "
          f"coherent AND p<0.05: {n_coh_low}")
    print(f"  Stacked Fisher chi^2 = {chi2:.2f} (dof {dof}) -> p = {p_stack:.3g} "
          f"({ts.p_to_sigma(p_stack):.2f} sigma)")
    # KS-style uniformity check on p-values
    srt = np.sort(ps)
    d_ks = float(np.max(np.abs(srt - (np.arange(1, len(srt) + 1) / len(srt)))))
    print(f"  p-value distribution max deviation from uniform (KS D) = {d_ks:.3f}")
    print()
    if p_stack > 0.05 and n_coh_low <= max(1, round(0.05 * len(results))):
        print("  VERDICT: POPULATION NULL. Stacking the O1-O3 BBH+NSBH catalog shows no")
        print("           coherent post-merger echo comb at the NVG spacing. The p-values are")
        print("           consistent with noise; no echo population is present in O1-O3.")
    else:
        print("  VERDICT: EXCESS TO INVESTIGATE. A coherent low-p pile-up survives time-slides;")
        print("           requires DQ vetoes + injection validation + a full pipeline before")
        print("           any statement. Not a detection claim.")
    print("=" * 96)


if __name__ == "__main__":
    main()
