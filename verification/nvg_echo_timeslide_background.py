#!/usr/bin/env python3
"""
NVG echo-comb search, rigorously calibrated with a TIME-SLIDE background.

Follow-up to nvg_echo_stacked_search.py, which used a crude single-segment
off-source background and flagged GW150914 at p~0.005 (2.8 sigma). Here we replace
that with the gold-standard significance estimate for a coherent network search:

  * zero-lag network statistic  S0 = max_over_delays max_{|t|<8ms} (|SNR_H1|^2(t) + |SNR_L1|^2(t))
    evaluated at the merger (where an echo comb would sit);
  * BACKGROUND from time-slides: shift L1's SNR series relative to H1 by many
    non-physical lags, recompute the same network statistic. Sliding destroys any
    real coherence, so the slide distribution IS the null. p = (1+#{S_slide>=S0})/(1+N).
  * The delay scan (16-34 ms window) is maximised inside BOTH S0 and every slide,
    so look-elsewhere over delay is automatically folded into the background.
  * SINGLE-DETECTOR decomposition: a genuine echo must be coherent (both detectors
    contribute at the same merger-relative time). An excess carried by one detector
    is instrumental. We print H1-only and L1-only on-source peaks for diagnosis.

Requires pycbc (installed) + network to GWOSC.
"""
from __future__ import annotations
import csv
import math
import os
import sys

import numpy as np

try:
    from pycbc.catalog import Merger
    from pycbc.types import TimeSeries
    from pycbc.filter import matched_filter
    from pycbc.psd import interpolate, inverse_spectrum_truncation
except Exception as exc:  # pragma: no cover
    print("PyCBC unavailable:", exc)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))

EVENTS = [
    "GW150914", "GW170104", "GW170814", "GW170823",
    "GW190521", "GW190519_153544", "GW190602_175927", "GW190630_185205",
]
DELAY_MULT = [0.73, 0.87, 1.00, 1.14, 1.27, 1.41, 1.55]   # 16-34 ms @ 65 Msun
R_EFF = 0.95
ON_WIN = 0.008
EDGE = 4.0
DUR = 0.35
N_SLIDES = 400
SLIDE_MIN = 0.10        # s, minimum |lag| so slid L1 on-source maps to off-source


def load_masses() -> dict:
    path = os.path.join(HERE, "data", "gwtc_events.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("commonName") or ""
            for key, fac in (("final_mass_source", 1.0), ("total_mass_source", 0.95)):
                try:
                    out.setdefault(name, float(row[key]) * fac)
                    break
                except (KeyError, ValueError, TypeError):
                    continue
    return out


def echo_comb(mass_final, sr, duration, dt_echo):
    n = int(duration * sr)
    t = np.arange(n) / sr
    wf = np.zeros(n)
    f_qnm = 251.0 * (65.0 / mass_final)
    tau = 3.6e-3 * (mass_final / 65.0)
    for k in range(1, max(1, int(duration / dt_echo)) + 1):
        idx = t >= k * dt_echo
        te = t[idx] - k * dt_echo
        sign = -1.0 if (k % 2) else 1.0
        wf[idx] += sign * (R_EFF ** k) * np.exp(-te / tau) * np.sin(2 * np.pi * f_qnm * te)
    return wf


def condition(strain):
    strain = strain.highpass_fir(15, 512)
    psd = strain.psd(4)
    psd = interpolate(psd, strain.delta_f)
    psd = inverse_spectrum_truncation(psd, int(4 * strain.sample_rate),
                                      low_frequency_cutoff=15)
    return strain, psd


def snr2_on_grid(strain, psd, mass_final, dt_echo, grid, t0):
    tmpl = TimeSeries(echo_comb(mass_final, strain.sample_rate, DUR, dt_echo),
                      delta_t=strain.delta_t)
    tmpl.resize(len(strain))
    snr = matched_filter(tmpl, strain, psd=psd, low_frequency_cutoff=20).crop(EDGE + DUR, EDGE)
    rel = np.array(snr.sample_times) - t0
    return np.interp(grid, rel, np.abs(np.array(snr)) ** 2, left=0.0, right=0.0)


def p_to_sigma(p):
    if p <= 0:
        return float("inf")
    if p >= 1:
        return 0.0
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(mid / math.sqrt(2)) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def analyse(name, masses):
    """Fetch O1-O3 open strain via the pycbc catalog, then run the coherent search."""
    try:
        m = Merger(name)
    except Exception:
        m = None
        for src in ("gwtc-1", "gwtc-2", "gwtc-3"):
            try:
                m = Merger(name, source=src); break
            except Exception:
                pass
        if m is None:
            return None
    dets = {}
    for det in ("H1", "L1"):
        try:
            dets[det] = condition(m.strain(det))
        except Exception:
            continue
    if "H1" not in dets:
        return None
    return analyse_strains(dets, m.time, masses.get(name, 60.0), name)


def analyse_strains(dets, t0, mass_final, name):
    """Coherent time-slide search on conditioned {det: (strain, psd)} dicts.

    Reused by both the O1-O3 (catalog) and O4 (direct GWOSC fetch) drivers."""
    dt0 = 0.022 * (mass_final / 65.0)
    # shared merger-relative grid, taken from H1 at the central delay
    s, psd_h = dets["H1"]
    snr0 = matched_filter(_sized_template(mass_final, s, dt0), s,
                          psd=psd_h, low_frequency_cutoff=20).crop(EDGE + DUR, EDGE)
    grid = np.array(snr0.sample_times) - t0
    dt_grid = grid[1] - grid[0]

    series = {d: {} for d in dets}
    for det, (strain, psd) in dets.items():
        for mult in DELAY_MULT:
            series[det][mult] = snr2_on_grid(strain, psd, mass_final, dt0 * mult, grid, t0)

    on_mask = np.abs(grid) <= ON_WIN
    on_idx = np.where(on_mask)[0]

    def net_stat(shift_samps):
        """max over delays of max on-source of (H1(t)^2 + L1(t+shift)^2)."""
        best = 0.0
        for mult in DELAY_MULT:
            h = series["H1"][mult]
            tot = h.copy()
            if "L1" in series:
                tot = tot + np.roll(series["L1"][mult], shift_samps)
            best = max(best, float(tot[on_idx].max()))
        return best

    S0 = net_stat(0)
    # single-detector on-source peaks (coherence diagnosis)
    h1_only = max(float(series["H1"][mult][on_idx].max()) for mult in DELAY_MULT)
    l1_only = (max(float(series["L1"][mult][on_idx].max()) for mult in DELAY_MULT)
               if "L1" in series else 0.0)

    # time-slide background
    n = len(grid)
    min_shift = int(SLIDE_MIN / dt_grid)
    max_shift = n - min_shift
    if "L1" in series and max_shift > min_shift:
        shifts = np.unique(np.linspace(min_shift, max_shift, N_SLIDES).astype(int))
        bkg = np.array([net_stat(int(sh)) for sh in shifts])
    else:
        bkg = np.array([])

    if bkg.size:
        p = (1.0 + int(np.sum(bkg >= S0))) / (1.0 + bkg.size)
    else:
        p = float("nan")
    return {
        "name": name, "mass_final": mass_final, "n_det": len(dets),
        "S0": S0, "h1": h1_only, "l1": l1_only,
        "bkg_med": float(np.median(bkg)) if bkg.size else float("nan"),
        "bkg_max": float(bkg.max()) if bkg.size else float("nan"),
        "n_slide": int(bkg.size), "p": p,
    }


def _sized_template(mass_final, strain, dt_echo):
    tmpl = TimeSeries(echo_comb(mass_final, strain.sample_rate, DUR, dt_echo),
                      delta_t=strain.delta_t)
    tmpl.resize(len(strain))
    return tmpl


def main():
    masses = load_masses()
    print("=" * 96)
    print("  NVG ECHO-COMB SEARCH  --  TIME-SLIDE BACKGROUND (coherent-network significance)")
    print("=" * 96)
    print(f"  {N_SLIDES} L1-vs-H1 time-slides/event; delay LEE folded into background; "
          f"on-source +/-{ON_WIN*1e3:.0f} ms")
    print("-" * 96)
    print(f"  {'event':<18}{'M_f':>6}{'det':>4}{'S0(netSNR^2)':>13}{'H1only':>8}{'L1only':>8}"
          f"{'coh?':>6}{'bkg_med':>8}{'bkg_max':>8}{'slides':>7}{'p':>8}{'sigma':>7}")
    print("  " + "-" * 92)

    results = []
    for name in EVENTS:
        try:
            r = analyse(name, masses)
        except Exception as exc:
            print(f"  {name:<18} skipped ({type(exc).__name__})")
            continue
        if r is None:
            print(f"  {name:<18} skipped (no data)")
            continue
        results.append(r)
        # coherent if network exceeds the larger single-detector peak by a real margin
        coh = "yes" if r["S0"] > 1.15 * max(r["h1"], r["l1"]) else "1det"
        sig = p_to_sigma(r["p"]) if r["p"] == r["p"] else float("nan")
        print(f"  {r['name']:<18}{r['mass_final']:>6.1f}{r['n_det']:>4}{r['S0']:>13.2f}"
              f"{r['h1']:>8.2f}{r['l1']:>8.2f}{coh:>6}{r['bkg_med']:>8.2f}{r['bkg_max']:>8.2f}"
              f"{r['n_slide']:>7}{r['p']:>8.3f}{sig:>7.2f}")

    print("-" * 96)
    valid = [r for r in results if r["p"] == r["p"]]
    if not valid:
        print("  No dual-detector events with a time-slide background.")
        print("=" * 96); return

    ps = np.clip(np.array([r["p"] for r in valid]), 1e-6, 1.0)
    chi2 = -2.0 * float(np.sum(np.log(ps)))
    dof = 2 * len(ps)
    # chi^2 survival (regularized upper incomplete gamma)
    def chi2_sf(x, k):
        a, xx = k / 2.0, x / 2.0
        if xx <= 0:
            return 1.0
        if xx < a + 1:
            term = 1.0 / a; sm = term; nn = a
            for _ in range(500):
                nn += 1; term *= xx / nn; sm += term
                if abs(term) < abs(sm) * 1e-12:
                    break
            return 1.0 - sm * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        b = xx + 1 - a; c = 1e30; d = 1.0 / b; h = d
        for i in range(1, 500):
            an = -i * (i - a); b += 2
            d = an * d + b; d = 1e-30 if abs(d) < 1e-30 else d
            c = b + an / c; c = 1e-30 if abs(c) < 1e-30 else c
            d = 1.0 / d; delta = d * c; h *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        return h * math.exp(-xx + a * math.log(xx) - math.lgamma(a))

    p_stack = chi2_sf(chi2, dof)
    print(f"  Dual-detector events: {len(valid)}")
    print(f"  Stacked Fisher chi^2 = {chi2:.2f} (dof {dof}) -> p = {p_stack:.3g} "
          f"({p_to_sigma(p_stack):.2f} sigma)")
    loud = [r for r in valid if r["p"] < 0.05]
    print()
    print("  Coherence note: 'coh?=1det' means the on-source excess is carried by a single")
    print("  detector -> instrumental, NOT an astrophysical echo, regardless of its p-value.")
    print()
    if not loud and p_stack > 0.05:
        print("  VERDICT: NULL under the proper time-slide background. No coherent post-merger")
        print("           echo comb at the predicted spacing; the earlier off-source p-values")
        print("           (incl. GW150914) do not survive a real coincidence background.")
    elif any(r["p"] < 0.01 and r["S0"] > 1.15 * max(r["h1"], r["l1"]) for r in valid):
        print("  VERDICT: a COHERENT sub-1% event survives time-slides -- worth a full pipeline")
        print("           (DQ vetoes, injections, more slides) before any statement. Not a claim.")
    else:
        print("  VERDICT: NO detection. Any low p is single-detector (instrumental) and/or the")
        print("           stack is consistent with noise.")
    print("=" * 96)


if __name__ == "__main__":
    main()
