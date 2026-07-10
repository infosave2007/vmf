#!/usr/bin/env python3
"""
O4 extension of the coherent echo-comb population search (GWTC-4.0 / GWTC-5.0).

pycbc's per-event catalog does not serve O4 strain yet, so we fetch O4a/O4b open
strain directly from GWOSC by GPS (nvg_o4_fetch.fetch_strain, lazy HTTP-range reads)
and feed it into the SAME validated coherent time-slide engine
(nvg_echo_timeslide_background.analyse_strains) used for the O1-O3 null. Nothing in
the statistics changes -- only the data source.

Selects confident O4 events (BH primary m1>3 -> BBH+NSBH, network SNR>7, p_astro>=0.5)
from the local GWTC catalog CSV, runs the coherent test per event, Fisher-combines the
O4 p-values, and reports the O4 stack plus a combination with the committed O1-O3 stack.
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
from nvg_o4_fetch import fetch_strain        # noqa: E402

ts.N_SLIDES = 200
MAX_EVENTS = 45
MIN_M1_BH = 3.0
MIN_SNR = 7.0
GPS_O4_START = 1.30e9
MIN_PASTRO = 0.5

# committed O1-O3 result (nvg_echo_population_stack.py, commit ef42f66): 52 events
O1O3_CHI2, O1O3_DOF, O1O3_N = 112.40, 104, 52


def chi2_sf(x, k):
    a, xx = k / 2.0, x / 2.0
    if xx <= 0:
        return 1.0
    if xx < a + 1:
        term = 1.0 / a; sm = term; nn = a
        for _ in range(2000):
            nn += 1; term *= xx / nn; sm += term
            if abs(term) < abs(sm) * 1e-13:
                break
        return 1.0 - sm * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    b = xx + 1 - a; c = 1e30; d = 1.0 / b; h = d
    for i in range(1, 2000):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1e-30 if abs(d) < 1e-30 else d
        c = b + an / c; c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < 1e-13:
            break
    return h * math.exp(-xx + a * math.log(xx) - math.lgamma(a))


def select_o4():
    path = os.path.join(HERE, "data", "gwtc_events.csv")
    seen, cand = set(), []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("commonName")
            if not name or name in seen:
                continue
            try:
                m1 = float(row["mass_1_source"]); snr = float(row["network_matched_filter_snr"])
                gps = float(row["GPS"])
            except (KeyError, ValueError, TypeError):
                continue
            try:
                pastro = float(row["p_astro"])
            except (KeyError, ValueError, TypeError):
                pastro = 1.0
            if gps >= GPS_O4_START and m1 >= MIN_M1_BH and snr >= MIN_SNR and pastro >= MIN_PASTRO:
                seen.add(name)
                cand.append((snr, name, gps))
    cand.sort(reverse=True)
    return cand[:MAX_EVENTS]


def main():
    masses = ts.load_masses()
    events = select_o4()
    print("=" * 96)
    print("  NVG ECHO-COMB SEARCH -- O4 EXTENSION (GWTC-4.0/5.0, direct GWOSC fetch)")
    print("=" * 96)
    print(f"  {len(events)} O4 candidates (m1>={MIN_M1_BH}, netSNR>={MIN_SNR}, p_astro>={MIN_PASTRO}); "
          f"{ts.N_SLIDES} slides/event")
    print("-" * 96)
    print(f"  {'event':<20}{'netSNR':>7}{'M_f':>6}{'det':>4}{'S0':>9}{'H1':>8}{'L1':>8}{'coh?':>6}"
          f"{'bkg_med':>8}{'p':>8}")
    print("  " + "-" * 90)

    results = []
    snrs = {}
    for snr, name, gps in events:
        dets = {}
        for det in ("H1", "L1"):
            s = fetch_strain(det, gps)
            if s is not None:
                try:
                    dets[det] = ts.condition(s)
                except Exception:
                    pass
        if "H1" not in dets or "L1" not in dets:
            print(f"  {name:<20} skipped (only {sorted(dets) or 'no'} detector open strain)",
                  flush=True)
            continue
        try:
            r = ts.analyse_strains(dets, gps, masses.get(name, 60.0), name)
        except Exception as exc:
            print(f"  {name:<20} skipped ({type(exc).__name__})", flush=True)
            continue
        if r is None or r["p"] != r["p"]:
            print(f"  {name:<20} skipped (no background)", flush=True); continue
        results.append(r)
        snrs[r["name"]] = snr
        coh = "yes" if r["S0"] > 1.15 * max(r["h1"], r["l1"]) else "1det"
        print(f"  {r['name']:<20}{snr:>7.1f}{r['mass_final']:>6.1f}{r['n_det']:>4}{r['S0']:>9.2f}"
              f"{r['h1']:>8.2f}{r['l1']:>8.2f}{coh:>6}{r['bkg_med']:>8.2f}{r['p']:>8.3f}",
              flush=True)

    print("-" * 96)
    if not results:
        print("  No O4 events with dual-detector open strain returned a background.")
        print("=" * 96); return

    ps = np.clip(np.array([r["p"] for r in results]), 1e-6, 1.0)
    chi2 = -2.0 * float(np.sum(np.log(ps)))
    dof = 2 * len(ps)
    p_o4 = chi2_sf(chi2, dof)
    n_coh = sum(1 for r in results if r["S0"] > 1.15 * max(r["h1"], r["l1"]))
    n_low = sum(1 for r in results if r["p"] < 0.05)
    srt = np.sort(ps)
    d_ks = float(np.max(np.abs(srt - (np.arange(1, len(srt) + 1) / len(srt)))))

    print(f"  O4 events stacked: {len(results)}  coherent(both-det): {n_coh}")
    print(f"  p<0.05: {n_low} (expected ~{0.05*len(results):.1f}); KS D={d_ks:.3f}")
    print(f"  O4 stack: Fisher chi^2={chi2:.2f} (dof {dof}) -> p={p_o4:.3g} "
          f"({ts.p_to_sigma(p_o4):.2f} sigma)")

    # combine with committed O1-O3 stack (independent events -> Fisher chi^2 adds)
    chi2_all = chi2 + O1O3_CHI2
    dof_all = dof + O1O3_DOF
    p_all = chi2_sf(chi2_all, dof_all)
    print(f"  O1-O4 combined ({len(results)+O1O3_N} events): Fisher chi^2={chi2_all:.1f} "
          f"(dof {dof_all}) -> p={p_all:.3g} ({ts.p_to_sigma(p_all):.2f} sigma)")

    # main-signal-leakage diagnostic: rank-correlate event loudness with -log(p).
    # If low p tracks high SNR, the coherent on-source excess is the PRIMARY signal
    # bleeding through the comb template, not an echo (needs residual subtraction).
    sn = np.array([snrs[r["name"]] for r in results])
    neglp = -np.log(ps)
    def rankcorr(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        ra = ra - ra.mean(); rb = rb - rb.mean()
        d = math.sqrt(float(np.sum(ra**2)) * float(np.sum(rb**2)))
        return float(np.sum(ra * rb) / d) if d else 0.0
    rho = rankcorr(sn, neglp)
    low = sorted(results, key=lambda r: r["p"])[:3]
    print()
    print(f"  DIAGNOSTIC (main-signal leakage): Spearman rho(netSNR, -log p) = {rho:+.2f}")
    print("    lowest-p events:", ", ".join(f"{r['name']}(SNR{snrs[r['name']]:.0f},p={r['p']:.3f})"
                                            for r in low))
    if rho > 0.4:
        print("    -> low p tracks loudness: coherent on-source excess is PRIMARY-SIGNAL")
        print("       leakage through the comb template, NOT echoes. A clean limit needs")
        print("       residual subtraction (best-fit merger removed) before the echo search.")
    print()
    coh_low = [r for r in results if r["p"] < 0.01 and r["S0"] > 1.15 * max(r["h1"], r["l1"])]
    if p_o4 > 0.05 and not coh_low:
        print("  VERDICT: O4 POPULATION NULL. The more sensitive O4 data show no coherent")
        print("           post-merger echo comb at the NVG spacing; combined with O1-O3 the")
        print("           non-detection is the deepest to date. Upper limit tightens with N.")
    else:
        print("  VERDICT: coherent low-p O4 event(s) survive time-slides -> full-pipeline")
        print("           follow-up (DQ vetoes, injections) required before any statement.")
    print("=" * 96)


if __name__ == "__main__":
    main()
