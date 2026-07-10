#!/usr/bin/env python3
"""
Deep-dive on the low-p O4 events: echo, or primary-signal leakage through the comb?

The O4 run flagged a marginal 2.39-sigma stack from 4-5 low-p events, all loud/massive
confident BBH. The suspicion is primary-signal leakage: a loud inspiral-merger-ringdown
produces a broad matched-filter response to the 250 Hz comb template that reaches the
on-source window in both detectors, which the L1-vs-H1 time-slide then scores as a
coherent excess.

A naive time-domain gate (zeroing the merger) is INVALID here -- it injects a broadband
edge transient and corrupts the PSD, giving nonsensical SNR ~ 10^5 (verified). Instead we
use a template-based discriminant that needs no gating:

  DISCRIMINANT: drop the FIRST comb tooth and search only teeth at 2*dt, 3*dt, ...
  The primary signal sits at the merger and can only bleed into the first-tooth region;
  a GENUINE echo comb still has teeth at 2*dt, 3*dt, ... So:
     low-p vanishes when the first tooth is removed  ->  primary-signal leakage (no echo);
     low-p survives                                   ->  genuine multi-tooth echo comb.

Runs each low-p event with the full comb and the first-tooth-removed comb (1000 slides).
"""
from __future__ import annotations
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nvg_echo_timeslide_background as ts   # noqa: E402
from nvg_o4_fetch import fetch_strain        # noqa: E402

ts.N_SLIDES = 1000
EVENTS = ["GW250114_082203", "GW240705_053215", "GW241129_021832", "GW250118_170523"]

_ORIG_COMB = ts.echo_comb


def comb_skip_first(mass_final, sr, duration, dt_echo):
    """Echo comb with the first tooth removed (teeth at 2*dt, 3*dt, ...)."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    wf = np.zeros(n)
    f_qnm = 251.0 * (65.0 / mass_final)
    tau = 3.6e-3 * (mass_final / 65.0)
    for k in range(2, max(2, int(duration / dt_echo)) + 1):   # start at k=2
        idx = t >= k * dt_echo
        te = t[idx] - k * dt_echo
        sign = -1.0 if (k % 2) else 1.0
        wf[idx] += sign * (ts.R_EFF ** k) * np.exp(-te / tau) * np.sin(2 * np.pi * f_qnm * te)
    return wf


def load_gps():
    out = {}
    with open(os.path.join(HERE, "data", "gwtc_events.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["commonName"]] = float(row["GPS"])
            except (KeyError, ValueError, TypeError):
                pass
    return out


def analyse(name, gps, mass):
    dets = {}
    for det in ("H1", "L1"):
        s = fetch_strain(det, gps)
        if s is not None:
            dets[det] = ts.condition(s)
    if "H1" not in dets or "L1" not in dets:
        return None
    return ts.analyse_strains(dets, gps, mass, name)


def main():
    gps_of = load_gps()
    masses = ts.load_masses()
    print("=" * 90)
    print("  O4 LOW-p DEEP-DIVE: full comb  vs  first-tooth-removed comb (primary-leakage test)")
    print("=" * 90)
    print(f"  {ts.N_SLIDES} slides. Primary bleeds only into tooth 1; a real echo keeps teeth 2,3,...")
    print("-" * 90)
    print(f"  {'event':<20}{'M_f':>6}{'p(full)':>9}{'p(no 1st tooth)':>17}{'interpretation':>22}")
    print("  " + "-" * 80)

    rows = []
    for name in EVENTS:
        gps, mass = gps_of.get(name), masses.get(name, 60.0)
        ts.echo_comb = _ORIG_COMB
        full = analyse(name, gps, mass)
        ts.echo_comb = comb_skip_first
        skip = analyse(name, gps, mass)
        ts.echo_comb = _ORIG_COMB
        if full is None or skip is None:
            print(f"  {name:<20} skipped (data)", flush=True); continue
        interp = "LEAKAGE (tooth1 only)" if (full["p"] < 0.05 <= skip["p"]) else \
                 ("survives -> echo?" if skip["p"] < 0.05 else "null both")
        print(f"  {name:<20}{mass:>6.1f}{full['p']:>9.3f}{skip['p']:>17.3f}{interp:>22}", flush=True)
        rows.append((name, full["p"], skip["p"]))

    print("-" * 90)
    if rows:
        leak = sum(1 for _, f, s in rows if f < 0.05 <= s)
        surv = [n for n, f, s in rows if s < 0.05]
        print(f"  Of {len(rows)} low-p events: {leak} are pure first-tooth (primary leakage); "
              f"{len(surv)} keep multi-tooth significance.")
        print()
        if not surv:
            print("  VERDICT: the O4 excess is PRIMARY-SIGNAL LEAKAGE. Removing the first comb")
            print("           tooth (which alone overlaps the merger) erases the coherent excess")
            print("           in every case -> no genuine echo comb. The 2.39-sigma stack is a")
            print("           systematic of loud/massive BBH, not a signal. Deep echo limits need")
            print("           residual-subtracted (best-fit-waveform-removed) data.")
        else:
            print(f"  VERDICT: {surv} keep multi-tooth coherence -> genuine echo candidate(s);")
            print("           escalate to a full pipeline (residual PE, DQ vetoes, injections).")
    print("=" * 90)


if __name__ == "__main__":
    main()
