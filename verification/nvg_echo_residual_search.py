#!/usr/bin/env python3
"""
Residual-subtracted echo search on GW250114 (the O4 leakage diagnosis follow-up).

The naive O4 run (nvg_echo_o4_search.py) flagged GW250114 at p~0.001, which
nvg_o4_lowp_deepdive.py traced to PRIMARY-SIGNAL LEAKAGE: at network SNR ~79 the
matched-filter response of the inspiral-merger-ringdown itself reaches the +-8 ms
on-source window through the comb template in both detectors, and the L1-vs-H1
time-slide then scores it as coherent. The gap-tooth discriminant (p 0.001 -> 0.50)
removes the effect but also throws away the first comb tooth.

Here we do the clean version: subtract the best-fit IMR waveform from each
detector's strain (standard PyCBC recipe: matched-filter peak -> aligned, scaled
template -> time-domain subtraction, maximised over a small IMRPhenomD grid), then
run the UNMODIFIED coherent time-slide engine (all comb teeth intact) on the
residual. Side-by-side with the naive run on the same data.

A real echo lives in the post-merger data and is NOT removed by subtracting the
best-fit IMR model (which contains no echoes); leakage is. So:
  - leakage hypothesis  -> residual p becomes unremarkable;
  - echo hypothesis     -> residual p stays low with full-comb coherence.
"""
from __future__ import annotations
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import nvg_echo_timeslide_background as ts   # noqa: E402
from nvg_o4_fetch import fetch_strain        # noqa: E402

from pycbc.filter import matched_filter, sigma   # noqa: E402
from pycbc.waveform import get_td_waveform       # noqa: E402

EVENT = "GW250114_082203"
GPS = 1420878141.2
M_FINAL_SRC = 62.9          # GWTC-5.0 v2 final_mass_source (pipeline convention)
Z = 0.09                    # GWTC-5.0 v2 redshift
# detector-frame component masses around the catalog point (33.76, 32.26) x (1+z)
MC_DET_GRID = np.linspace(30.6, 32.0, 8)     # detector-frame chirp mass
Q_GRID = [1.0, 0.90, 0.80]                   # mass ratio m2/m1
CHI_GRID = [0.0, -0.05]                      # aligned spin (chi_eff ~ -0.03)
F_LOW = 20.0

ts.N_SLIDES = 200


def component_masses(mc, q):
    """(m1, m2) from chirp mass and q = m2/m1 <= 1."""
    m1 = mc * (1 + q) ** 0.2 / q ** 0.6
    return m1, q * m1


def best_fit_subtract(strain, psd, label):
    """Subtract the maximum-SNR IMRPhenomD template; return (residual, report)."""
    best = None
    for mc in MC_DET_GRID:
        for q in Q_GRID:
            for chi in CHI_GRID:
                m1, m2 = component_masses(mc, q)
                try:
                    hp, _ = get_td_waveform(approximant="IMRPhenomD",
                                            mass1=m1, mass2=m2,
                                            spin1z=chi, spin2z=chi,
                                            delta_t=strain.delta_t, f_lower=F_LOW)
                except Exception:
                    continue
                hp.resize(len(strain))
                template = hp.cyclic_time_shift(hp.start_time)
                template.start_time = strain.start_time
                snr = matched_filter(template, strain, psd=psd,
                                     low_frequency_cutoff=F_LOW).crop(ts.EDGE, ts.EDGE)
                peak = int(np.argmax(np.abs(np.array(snr))))
                z = complex(snr[peak])
                if best is None or abs(z) > abs(best["z"]):
                    best = {"z": z, "t": float(snr.sample_times[peak]),
                            "tmpl": template, "mc": mc, "q": q, "chi": chi}
    if best is None:
        raise RuntimeError("no template generated")

    tmpl = best["tmpl"]
    dt = best["t"] - float(strain.start_time)
    aligned = tmpl.cyclic_time_shift(dt)
    aligned.start_time = strain.start_time
    norm = sigma(aligned, psd=psd, low_frequency_cutoff=F_LOW)
    aligned_fd = aligned.to_frequencyseries() * (best["z"] / norm)
    fit = aligned_fd.to_timeseries()
    fit.start_time = strain.start_time
    residual = strain - fit

    # subtraction quality: re-filter the residual with the best template
    snr_res = matched_filter(tmpl, residual, psd=psd,
                             low_frequency_cutoff=F_LOW).crop(ts.EDGE, ts.EDGE)
    rel = np.abs(np.array(snr_res.sample_times) - best["t"]) < 0.05
    resid_snr = float(np.max(np.abs(np.array(snr_res))[rel]))
    print(f"    {label}: best (Mc_det={best['mc']:.2f}, q={best['q']:.2f}, "
          f"chi={best['chi']:+.2f})  SNR {abs(best['z']):.1f} -> residual {resid_snr:.1f}")
    return residual, abs(best["z"]), resid_snr


def inject_echo(strain, psd, snr_target):
    """Add an NVG echo comb (teeth at k*dt0 after merger) at a fixed per-detector SNR.

    Same-epoch injection in H1 and L1 (no antenna/arrival-time modelling): the
    +-8 ms on-source window absorbs the inter-site delay, matching what the
    search itself assumes. This is a method validation, not an astrophysical one.
    """
    from pycbc.types import TimeSeries
    sr = strain.sample_rate
    dt0 = 0.022 * (M_FINAL_SRC / 65.0)
    comb = np.zeros(len(strain))
    i0 = int(round((GPS - float(strain.start_time)) * sr))
    wf = ts.echo_comb(M_FINAL_SRC, sr, ts.DUR, dt0)
    comb[i0:i0 + len(wf)] = wf
    comb_ts = TimeSeries(comb, delta_t=strain.delta_t, epoch=strain.start_time)
    norm = sigma(comb_ts, psd=psd, low_frequency_cutoff=F_LOW)
    return strain + comb_ts * (snr_target / norm)


def run_variant(dets, tag):
    r = ts.analyse_strains(dets, GPS, M_FINAL_SRC, EVENT)
    coh = "yes" if r["S0"] > 1.15 * max(r["h1"], r["l1"]) else "1det"
    print(f"  {tag:<29}S0={r['S0']:>8.2f}  H1={r['h1']:>7.2f}  L1={r['l1']:>7.2f}  "
          f"coh={coh:<5} bkg_med={r['bkg_med']:>7.2f}  p={r['p']:.3f} "
          f"({ts.p_to_sigma(r['p']):.2f} sigma)")
    return r


def main():
    print("=" * 96)
    print(f"  RESIDUAL-SUBTRACTED ECHO SEARCH -- {EVENT} (netSNR~79, the O4 leakage case)")
    print("=" * 96)
    print(f"  template delay dt0 = 22 ms x ({M_FINAL_SRC}/65) = "
          f"{22.0 * M_FINAL_SRC / 65.0:.1f} ms (source-frame pipeline convention);")
    print(f"  detector-frame center would be {22.0 * M_FINAL_SRC * (1 + Z) / 65.0:.1f} ms "
          f"-- covered by the 0.73-1.55x delay scan")
    print("-" * 96)

    raw = {}
    for det in ("H1", "L1"):
        s = fetch_strain(det, GPS)
        if s is None:
            print(f"  {det}: open strain unavailable -- abort"); return
        raw[det] = ts.condition(s)
    print("  strain fetched and conditioned (H1, L1)")

    print("  subtracting best-fit IMRPhenomD per detector:")
    resid = {}
    for det, (strain, psd) in raw.items():
        residual, snr_before, snr_after = best_fit_subtract(strain, psd, det)
        resid[det] = (residual, psd)
        if snr_after > 10:
            print(f"    {det}: WARNING residual SNR {snr_after:.1f} > 10 -- grid-level fit "
                  f"incomplete, PE-grade parameters needed for a final statement")

    # method control: inject an echo comb at SNR 8/detector into the RAW strain,
    # run the identical subtract-then-search chain -> a real echo must survive.
    INJ_SNR = 8.0
    inj = {}
    print(f"  injection control (echo comb at SNR {INJ_SNR:.0f}/det + IMR subtraction):")
    for det, (strain, psd) in raw.items():
        injected = inject_echo(strain, psd, INJ_SNR)
        residual, _, _ = best_fit_subtract(injected, psd, f"{det}+inj")
        inj[det] = (residual, psd)

    print("-" * 96)
    print(f"  coherent time-slide test ({ts.N_SLIDES} slides, full comb, all teeth):")
    r_naive = run_variant(raw, "original strain")
    r_resid = run_variant(resid, "residual (IMR removed)")
    r_inj = run_variant(inj, "injected echo + IMR removed")
    if r_inj["p"] > 0.05:
        print("    WARNING: injected echo did NOT survive the chain -- method insensitive,")
        print("             the residual null above is NOT informative at this amplitude.")

    print("-" * 96)
    drop = r_naive["S0"] / r_resid["S0"] if r_resid["S0"] > 0 else float("inf")
    if r_resid["p"] > 0.05:
        print(f"  VERDICT: LEAKAGE CONFIRMED, ECHO NULL. Removing the best-fit merger drops the")
        print(f"           on-source network statistic by x{drop:.1f} and the full-comb p-value")
        print(f"           becomes unremarkable ({r_naive['p']:.3f} -> {r_resid['p']:.3f}).")
        print(f"           An astrophysical echo would SURVIVE IMR subtraction; it does not.")
        print(f"           This supersedes the gap-tooth workaround (which discards tooth 1).")
    else:
        print(f"  VERDICT: residual p={r_resid['p']:.3f} stays low after IMR subtraction --")
        print(f"           NOT explained by leakage alone; needs DQ vetoes, >1000 slides,")
        print(f"           and phase-coherent statistics before any statement.")
    print("=" * 96)


if __name__ == "__main__":
    main()
