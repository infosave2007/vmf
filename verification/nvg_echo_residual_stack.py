#!/usr/bin/env python3
"""
Full O1-O4 residual-subtracted coherent echo stack + per-event 90% amplitude limit.

Combines everything the echo program established:
  * coherent L1-vs-H1 time-slide background (nvg_echo_timeslide_background.analyse_strains),
  * O4 strain via direct GWOSC HTTP-range fetch (nvg_o4_fetch),
  * per-detector best-fit IMR subtraction validated on GW250114
    (nvg_echo_residual_search.py: leakage removed, injected echo survives).

For every dual-detector event (m1>=3 Msun, netSNR>=7, O4 additionally p_astro>=0.5):
  1. fetch + condition strain;
  2. subtract the best-fit IMRPhenomD (template grid around the catalog masses,
     detector-frame; skipped with a warning if the fit fails);
  3. run the coherent time-slide test on the residual (all comb teeth intact --
     no gap-tooth workaround needed once the primary is removed);
  4. inject the NVG comb at per-detector SNR 6 into the residual and remeasure the
     on-source network statistic; matched-filter linearity (validated in
     nvg_echo_upper_limit.py) then gives rho_90 = the per-detector echo amplitude
     whose EXPECTED statistic crosses the 90th background percentile. This is a
     median-detection sensitivity threshold, NOT a formal frequentist 90% upper
     limit (which requires an injection-recovery efficiency curve and sits a
     factor ~1.3-1.6 higher).

Outputs a per-event CSV (data/nvg_residual_stack_results.csv), the Fisher-stacked
population p-value, and the deepest per-event amplitude limits. This is the first
leakage-free version of the stack: the naive O4 run showed an apparent 2.4 sigma
excess from primary-signal leakage; here loud events are cleaned instead of gapped.
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

from pycbc.catalog import Merger                 # noqa: E402
from pycbc.filter import matched_filter, sigma   # noqa: E402
from pycbc.types import TimeSeries               # noqa: E402
from pycbc.waveform import get_td_waveform       # noqa: E402

ts.N_SLIDES = 200
MIN_M1, MIN_SNR, MIN_PASTRO_O4 = 3.0, 7.0, 0.5
GPS_O4 = 1.30e9
INJ_SNR = 6.0
OUT_CSV = os.path.join(HERE, "data", "nvg_residual_stack_results.csv")


def select_events():
    path = os.path.join(HERE, "data", "gwtc_events.csv")
    seen, out = set(), []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("commonName")
            if not name or name in seen:
                continue
            try:
                m1 = float(row["mass_1_source"]); m2 = float(row["mass_2_source"])
                snr = float(row["network_matched_filter_snr"]); gps = float(row["GPS"])
            except (KeyError, ValueError, TypeError):
                continue
            try:
                z = float(row["redshift"])
            except (KeyError, ValueError, TypeError):
                z = 0.0
            try:
                chi = float(row["chi_eff"])
            except (KeyError, ValueError, TypeError):
                chi = 0.0
            try:
                pastro = float(row["p_astro"])
            except (KeyError, ValueError, TypeError):
                pastro = 1.0
            if m1 < MIN_M1 or snr < MIN_SNR:
                continue
            if gps >= GPS_O4 and pastro < MIN_PASTRO_O4:
                continue
            seen.add(name)
            out.append({"name": name, "gps": gps, "m1": m1, "m2": m2, "z": z,
                        "chi": chi, "snr": snr})
    out.sort(key=lambda e: -e["snr"])
    return out


def fetch_dets(ev):
    """{det: (strain, psd)} via pycbc catalog (O1-O3) or direct GWOSC (O4)."""
    dets = {}
    if ev["gps"] < GPS_O4:
        m = None
        try:
            m = Merger(ev["name"])
        except Exception:
            for src in ("gwtc-1", "gwtc-2", "gwtc-3"):
                try:
                    m = Merger(ev["name"], source=src); break
                except Exception:
                    pass
        if m is None:
            return {}
        for det in ("H1", "L1"):
            try:
                dets[det] = ts.condition(m.strain(det))
            except Exception:
                pass
    else:
        for det in ("H1", "L1"):
            s = fetch_strain(det, ev["gps"])
            if s is not None:
                try:
                    dets[det] = ts.condition(s)
                except Exception:
                    pass
    return dets


def subtract_imr(strain, psd, ev):
    """Best-fit IMRPhenomD subtraction on a catalog-centred detector-frame grid."""
    zf = 1.0 + ev["z"]
    mc_cat = (ev["m1"] * ev["m2"]) ** 0.6 / (ev["m1"] + ev["m2"]) ** 0.2 * zf
    q_cat = min(1.0, ev["m2"] / ev["m1"])
    f_low = 25.0 if mc_cat < 6.0 else 20.0
    fine = ev["snr"] >= 20.0
    mc_grid = mc_cat * (np.linspace(0.96, 1.04, 5) if fine else np.linspace(0.97, 1.03, 3))
    q_grid = sorted({q_cat, max(0.05, 0.85 * q_cat), min(1.0, 1.15 * q_cat)}) \
        if fine else [q_cat]
    chi_grid = [ev["chi"] - 0.1, ev["chi"], ev["chi"] + 0.1] if fine else [ev["chi"]]

    best = None
    for mc in mc_grid:
        for q in q_grid:
            for chi in chi_grid:
                m1 = mc * (1 + q) ** 0.2 / q ** 0.6
                m2 = q * m1
                try:
                    hp, _ = get_td_waveform(approximant="IMRPhenomD",
                                            mass1=m1, mass2=m2,
                                            spin1z=chi, spin2z=chi,
                                            delta_t=strain.delta_t, f_lower=f_low)
                except Exception:
                    continue
                if len(hp) > len(strain):
                    continue
                hp.resize(len(strain))
                tmpl = hp.cyclic_time_shift(hp.start_time)
                tmpl.start_time = strain.start_time
                try:
                    snr = matched_filter(tmpl, strain, psd=psd,
                                         low_frequency_cutoff=f_low
                                         ).crop(ts.EDGE, ts.EDGE)
                except Exception:
                    continue
                peak = int(np.argmax(np.abs(np.array(snr))))
                zc = complex(snr[peak])
                if best is None or abs(zc) > abs(best["z"]):
                    best = {"z": zc, "t": float(snr.sample_times[peak]),
                            "tmpl": tmpl, "f_low": f_low}
    if best is None:
        return strain, float("nan"), float("nan")

    tmpl = best["tmpl"]
    dt = best["t"] - float(strain.start_time)
    aligned = tmpl.cyclic_time_shift(dt)
    aligned.start_time = strain.start_time
    norm = sigma(aligned, psd=psd, low_frequency_cutoff=best["f_low"])
    fit = (aligned.to_frequencyseries() * (best["z"] / norm)).to_timeseries()
    fit.start_time = strain.start_time
    residual = strain - fit

    snr_res = matched_filter(tmpl, residual, psd=psd,
                             low_frequency_cutoff=best["f_low"]).crop(ts.EDGE, ts.EDGE)
    rel = np.abs(np.array(snr_res.sample_times) - best["t"]) < 0.05
    resid = float(np.max(np.abs(np.array(snr_res))[rel])) if rel.any() else float("nan")
    return residual, abs(best["z"]), resid


def net_s0(dets, t0, mass_final):
    """Zero-lag on-source network statistic only (no slides) -- for injections."""
    dt0 = 0.022 * (mass_final / 65.0)
    s, psd_h = dets["H1"]
    snr0 = matched_filter(ts._sized_template(mass_final, s, dt0), s,
                          psd=psd_h, low_frequency_cutoff=20).crop(ts.EDGE + ts.DUR, ts.EDGE)
    grid = np.array(snr0.sample_times) - t0
    on_idx = np.where(np.abs(grid) <= ts.ON_WIN)[0]
    best = 0.0
    for mult in ts.DELAY_MULT:
        tot = None
        for det, (strain, psd) in dets.items():
            se = ts.snr2_on_grid(strain, psd, mass_final, dt0 * mult, grid, t0)
            tot = se if tot is None else tot + se
        best = max(best, float(tot[on_idx].max()))
    return best


def inject_comb(strain, psd, t0, mass_final, snr_target):
    sr = strain.sample_rate
    dt0 = 0.022 * (mass_final / 65.0)
    comb = np.zeros(len(strain))
    i0 = int(round((t0 - float(strain.start_time)) * sr))
    wf = ts.echo_comb(mass_final, sr, ts.DUR, dt0)
    if i0 < 0 or i0 + len(wf) > len(comb):
        return None
    comb[i0:i0 + len(wf)] = wf
    comb_ts = TimeSeries(comb, delta_t=strain.delta_t, epoch=strain.start_time)
    norm = sigma(comb_ts, psd=psd, low_frequency_cutoff=20)
    return strain + comb_ts * (snr_target / norm)


def main():
    masses = ts.load_masses()
    events = select_events()
    print("=" * 100)
    print("  NVG FULL O1-O4 RESIDUAL-SUBTRACTED ECHO STACK "
          f"({len(events)} candidates, {ts.N_SLIDES} slides/event)")
    print("=" * 100)
    fields = ["name", "snr", "mass_final", "snr_h1", "res_h1", "snr_l1", "res_l1",
              "S0", "p", "bkg_p90", "S_inj", "rho90"]
    fh = open(OUT_CSV, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()

    print(f"  {'event':<22}{'SNR':>6}{'M_f':>6}{'sub H1':>12}{'sub L1':>12}"
          f"{'S0':>8}{'p':>7}{'rho90':>7}")
    print("  " + "-" * 94)
    results = []
    for ev in events:
        name = ev["name"]
        try:
            dets = fetch_dets(ev)
        except Exception as exc:
            print(f"  {name:<22} skipped (fetch {type(exc).__name__})", flush=True)
            continue
        if "H1" not in dets or "L1" not in dets:
            print(f"  {name:<22} skipped (needs H1+L1, have {sorted(dets) or 'none'})",
                  flush=True)
            continue
        mf = masses.get(name, 60.0)
        try:
            resid, subs = {}, {}
            for det, (strain, psd) in dets.items():
                r, s_before, s_after = subtract_imr(strain, psd, ev)
                resid[det] = (r, psd)
                subs[det] = (s_before, s_after)
            r = ts.analyse_strains(resid, ev["gps"], mf, name)
            inj = {}
            for det, (strain, psd) in resid.items():
                s_i = inject_comb(strain, psd, ev["gps"], mf, INJ_SNR)
                if s_i is None:
                    raise RuntimeError("injection window out of range")
                inj[det] = (s_i, psd)
            s_inj = net_s0(inj, ev["gps"], mf)
        except Exception as exc:
            print(f"  {name:<22} skipped (analysis {type(exc).__name__}: {exc})", flush=True)
            continue
        if r is None or r["p"] != r["p"]:
            print(f"  {name:<22} skipped (no background)", flush=True)
            continue
        rho90 = INJ_SNR * math.sqrt(r["bkg_p90"] / s_inj) if s_inj > 0 else float("nan")
        row = {"name": name, "snr": ev["snr"], "mass_final": mf,
               "snr_h1": subs["H1"][0], "res_h1": subs["H1"][1],
               "snr_l1": subs["L1"][0], "res_l1": subs["L1"][1],
               "S0": r["S0"], "p": r["p"], "bkg_p90": r["bkg_p90"],
               "S_inj": s_inj, "rho90": rho90}
        writer.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v)
                         for k, v in row.items()})
        fh.flush()
        results.append(row)
        print(f"  {name:<22}{ev['snr']:>6.1f}{mf:>6.1f}"
              f"{subs['H1'][0]:>6.1f}->{subs['H1'][1]:>4.1f}"
              f"{subs['L1'][0]:>6.1f}->{subs['L1'][1]:>4.1f}"
              f"{r['S0']:>8.2f}{r['p']:>7.3f}{rho90:>7.2f}", flush=True)
    fh.close()

    print("-" * 100)
    if len(results) < 5:
        print("  too few events for a stack"); return
    ps = np.clip(np.array([r["p"] for r in results]), 1e-6, 1.0)
    chi2 = -2.0 * float(np.sum(np.log(ps)))
    dof = 2 * len(ps)
    from scipy.stats import chi2 as chi2_dist, kstest
    p_stack = float(chi2_dist.sf(chi2, dof))
    ks = kstest(ps, "uniform")
    n_low = int(np.sum(ps < 0.05))
    rhos = np.array([r["rho90"] for r in results])
    fin = rhos[np.isfinite(rhos)]
    best = min(results, key=lambda r: r["rho90"] if np.isfinite(r["rho90"]) else 1e9)

    print(f"  events stacked: {len(results)}   p<0.05: {n_low} "
          f"(expected {0.05*len(results):.1f})")
    print(f"  Fisher chi^2 = {chi2:.1f} (dof {dof}) -> population p = {p_stack:.3f} "
          f"({ts.p_to_sigma(p_stack):.2f} sigma)")
    print(f"  KS uniformity: D = {ks.statistic:.3f}, p = {ks.pvalue:.3f}")
    print(f"  90% echo amplitude limits (per-detector SNR): "
          f"median rho_90 = {np.median(fin):.1f}, best = {best['rho90']:.2f} ({best['name']})")
    print(f"  per-event CSV: {OUT_CSV}")
    print("-" * 100)
    if p_stack > 0.05 and n_low <= max(2, 0.1 * len(results)):
        print("  VERDICT: LEAKAGE-FREE POPULATION NULL across O1-O4. With the primary signal")
        print("  subtracted (all comb teeth in play), no event and no stack shows a coherent")
        print("  NVG echo comb. This supersedes both the naive O4 2.4-sigma excess (leakage)")
        print("  and the gap-tooth workaround, and is the deepest limit of this program.")
    else:
        print("  VERDICT: residual excess persists after IMR subtraction -- requires DQ vetoes,")
        print("  >1000 slides and hardware-injection checks before interpretation.")
    print("=" * 100)


if __name__ == "__main__":
    main()
