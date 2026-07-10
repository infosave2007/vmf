#!/usr/bin/env python3
"""
NVG post-merger echo-comb search on real LVK open data (stacked, with background).

Honest test of the ONE robust NVG echo prediction: a regular Hayward core reflects
a train of post-merger "echoes" spaced by a geometric delay dt ~ 0.022*(M/65 M_sun),
logarithmically uncertain over ~0.73..1.55 of that value (the 16-34 ms cutoff window
for a 65 M_sun remnant). We matched-filter a *pure echo comb* template (no primary
ringdown) against whitened H1/L1 strain, scan the delay over the allowed window, and
compare the on-source statistic to an OFF-SOURCE background from the same segments.

Method (deliberately simple and auditable, not a full LVK pipeline):
  * per event: fetch 32 s of H1+L1 open strain, highpass + PSD-whiten (matched_filter);
  * template = alternating-phase (pi flip) damped-sinusoid echo train, QNM freq/damping
    and delay all scaled to the event's remnant mass; amplitude R_eff^n (R_eff=0.95);
  * on-source statistic = peak network |SNR|^2 in a +/-8 ms window about the merger
    (where an echo comb starting right after merger would land);
  * background = the SAME statistic evaluated on off-source windows (|t-t_merger|>0.5 s),
    giving a per-event p-value and, stacked, a combined (Fisher) p-value;
  * delay scan is penalised as trials (look-elsewhere).

Outputs a table + an honest verdict. Requires: pycbc (installed), network to GWOSC.
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

# Loud, well-localised BBH events spanning a range of remnant masses.
EVENTS = [
    "GW150914", "GW170104", "GW170814", "GW170823",
    "GW190521", "GW190519_153544", "GW190602_175927", "GW190630_185205",
]

# Delay-scan multipliers on the central delay (the 16-34 ms cutoff window for 65 Msun
# is 0.73..1.55 of the 22 ms central value); penalised as independent trials.
DELAY_MULT = [0.73, 0.87, 1.00, 1.14, 1.27, 1.41, 1.55]
R_EFF = 0.95
ON_WIN = 0.008          # s, half-width of on-source window about merger
OFF_GUARD = 0.5         # s, exclude |t-t_merger| < OFF_GUARD from background
EDGE = 4.0              # s, filter-corrupted crop each side


def load_remnant_masses() -> dict:
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


def echo_comb(mass_final: float, sample_rate: float, duration: float,
              dt_echo: float) -> np.ndarray:
    """Pure echo train (no primary ringdown), remnant-mass-scaled, alternating pi phase."""
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    wf = np.zeros(n)
    f_qnm = 251.0 * (65.0 / mass_final)      # QNM frequency ∝ 1/M
    tau = 3.6e-3 * (mass_final / 65.0)       # damping time ∝ M
    n_echo = max(1, int(duration / dt_echo))
    for k in range(1, n_echo + 1):
        shift = k * dt_echo
        idx = t >= shift
        te = t[idx] - shift
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


def snr_series(strain, psd, mass_final, duration, dt_echo):
    tmpl = TimeSeries(echo_comb(mass_final, strain.sample_rate, duration, dt_echo),
                      delta_t=strain.delta_t)
    tmpl.resize(len(strain))
    snr = matched_filter(tmpl, strain, psd=psd, low_frequency_cutoff=20)
    return snr.crop(EDGE + duration, EDGE)


def search_event(name: str, masses: dict):
    try:
        m = Merger(name)
    except Exception:
        for src in ("gwtc-1", "gwtc-2", "gwtc-3"):
            try:
                m = Merger(name, source=src)
                break
            except Exception:
                m = None
        if m is None:
            return None
    t0 = m.time
    mass_final = masses.get(name, 60.0)
    dt_central = 0.022 * (mass_final / 65.0)

    # network |SNR|^2(t) on a common merger-relative time grid, summed over detectors
    per_det = []
    for det in ("H1", "L1"):
        try:
            strain = m.strain(det)
        except Exception:
            continue
        strain, psd = condition(strain)
        best = None
        for mult in DELAY_MULT:
            try:
                snr = snr_series(strain, psd, mass_final, 0.35, dt_central * mult)
            except Exception:
                continue
            rel = np.array(snr.sample_times) - t0
            a2 = np.abs(np.array(snr)) ** 2
            if best is None:
                best = {"rel": rel, "a2_by_mult": {}}
            # interpolate onto the first grid for stacking consistency
            best["a2_by_mult"][mult] = (rel, a2)
        if best:
            per_det.append(best)
    if not per_det:
        return None

    # Common grid = first detector's first delay grid
    grid = per_det[0]["a2_by_mult"][DELAY_MULT[0]][0]

    def net_a2(mult):
        tot = np.zeros_like(grid)
        for d in per_det:
            if mult in d["a2_by_mult"]:
                r, a2 = d["a2_by_mult"][mult]
                tot += np.interp(grid, r, a2, left=0.0, right=0.0)
        return tot

    on_mask = np.abs(grid) <= ON_WIN
    off_mask = np.abs(grid) >= OFF_GUARD

    # For each delay, on-source peak vs off-source background; take best delay,
    # penalise with a trials factor = number of delays scanned.
    best_delay = None
    for mult in DELAY_MULT:
        net = net_a2(mult)
        on_peak = float(net[on_mask].max()) if on_mask.any() else 0.0
        off = net[off_mask]
        # off-source local maxima in equal-size (2*ON_WIN) windows -> independent trials
        win = max(1, int((2 * ON_WIN) / (grid[1] - grid[0])))
        off_max = [off[i:i + win].max() for i in range(0, len(off) - win, win)]
        off_max = np.array(off_max) if off_max else np.array([0.0])
        p = (1.0 + np.sum(off_max >= on_peak)) / (1.0 + len(off_max))
        if best_delay is None or on_peak > best_delay["on_peak"]:
            best_delay = {"mult": mult, "on_peak": on_peak, "p_raw": p,
                          "off_mean": float(off_max.mean()), "off_max": float(off_max.max())}
    # look-elsewhere over delays
    p_trials = 1.0 - (1.0 - best_delay["p_raw"]) ** len(DELAY_MULT)
    return {
        "name": name, "mass_final": mass_final, "dt_ms": dt_central * best_delay["mult"] * 1e3,
        "on_peak": best_delay["on_peak"], "off_mean": best_delay["off_mean"],
        "off_max": best_delay["off_max"], "p": min(1.0, p_trials), "n_det": len(per_det),
    }


def main():
    masses = load_remnant_masses()
    print("=" * 92)
    print("  NVG POST-MERGER ECHO-COMB SEARCH  (real LVK open data, stacked, background-calibrated)")
    print("=" * 92)
    print(f"  Template: pure echo comb, R_eff={R_EFF}, delay dt=0.022*(M/65) scaled over "
          f"x{DELAY_MULT[0]}..{DELAY_MULT[-1]} (16-34 ms @65 Msun)")
    print(f"  On-source window +/-{ON_WIN*1e3:.0f} ms about merger; background |t-t_merger|>{OFF_GUARD}s")
    print("-" * 92)
    print(f"  {'event':<18} {'M_f':>6} {'dt(ms)':>7} {'det':>4} {'netSNR^2_on':>12} "
          f"{'bkg_mean':>9} {'bkg_max':>8} {'p(LEE)':>9}")
    print("  " + "-" * 88)

    results = []
    for name in EVENTS:
        try:
            r = search_event(name, masses)
        except Exception as exc:
            print(f"  {name:<18} skipped ({type(exc).__name__}: {exc})")
            continue
        if r is None:
            print(f"  {name:<18} skipped (no data)")
            continue
        results.append(r)
        snr_on = math.sqrt(max(0.0, r["on_peak"]))
        print(f"  {r['name']:<18} {r['mass_final']:>6.1f} {r['dt_ms']:>7.1f} {r['n_det']:>4d} "
              f"{r['on_peak']:>12.2f} {r['off_mean']:>9.2f} {r['off_max']:>8.2f} {r['p']:>9.3f}"
              f"   (netSNR_on={snr_on:.2f})")

    print("-" * 92)
    if not results:
        print("  No events returned usable data (network/catalog).")
        print("=" * 92)
        return

    # Stacked (Fisher) combination of per-event look-elsewhere p-values
    ps = np.clip(np.array([r["p"] for r in results]), 1e-6, 1.0)
    chi2 = -2.0 * np.sum(np.log(ps))
    dof = 2 * len(ps)
    # survival of chi^2 with dof degrees of freedom (no scipy dependency)
    from math import erfc, sqrt
    def chi2_sf(x, k):
        # regularized upper incomplete gamma via series/continued fraction
        a = k / 2.0
        xx = x / 2.0
        if xx <= 0:
            return 1.0
        if xx < a + 1:
            term = 1.0 / a
            s = term
            n = a
            for _ in range(500):
                n += 1
                term *= xx / n
                s += term
                if abs(term) < abs(s) * 1e-12:
                    break
            return 1.0 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        b = xx + 1 - a
        c = 1e30
        d = 1.0 / b
        h = d
        for i in range(1, 500):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        return h * math.exp(-xx + a * math.log(xx) - math.lgamma(a))

    p_stack = chi2_sf(chi2, dof)
    # equivalent one-sided Gaussian sigma
    def p_to_sigma(p):
        # invert survival of standard normal via bisection on erfc
        if p <= 0:
            return float("inf")
        if p >= 1:
            return 0.0
        lo, hi = 0.0, 40.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * erfc(mid / sqrt(2)) > p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    print(f"  Events used: {len(results)}")
    print(f"  Stacked Fisher chi^2 = {chi2:.2f} (dof {dof}) -> combined p = {p_stack:.3g} "
          f"({p_to_sigma(p_stack):.2f} sigma)")
    loud = [r for r in results if r["p"] < 0.05]
    print()
    if not loud and p_stack > 0.01:
        print("  VERDICT: NULL. No event shows a post-merger echo comb above background at the")
        print("           predicted spacing, and the stack is consistent with noise. This sets an")
        print("           upper limit on the echo amplitude (R_eff train) rather than a detection.")
    elif loud and p_stack < 1e-3:
        print("  VERDICT: STACKED EXCESS — investigate before ANY claim. Check: PSD lines, glitch")
        print("           vetoes (CAT2/3), single-detector vs coherent, injection recovery, and a")
        print("           proper time-slide background. Do NOT report as detection without these.")
    else:
        print("  VERDICT: INCONCLUSIVE / marginal. Individual p-values and stack do not cross a")
        print("           robust threshold; treat as no-detection pending a full pipeline.")
    print("=" * 92)


if __name__ == "__main__":
    main()
