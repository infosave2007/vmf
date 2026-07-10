#!/usr/bin/env python3
"""
Upper limit on NVG post-merger echo amplitude via injection-recovery (GW150914).

The coherent time-slide search (nvg_echo_timeslide_background.py) found NO echo comb
(stack 0.0 sigma; GW150914's apparent excess was single-detector and below the slide
median). A non-detection is only scientific if we say what amplitude we could have seen.

Method (exact, using matched-filter linearity):
  * condition real H1+L1 strain for GW150914; build the NVG echo-comb template family
    (delay scan over the 16-34 ms cutoff window, mass-scaled);
  * z0_{d,delay}(t) = complex matched-filter SNR of the real data (the "noise");
  * u_{d,delay}(t)  = matched-filter response to a UNIT-amplitude comb injected at the
    reference time; an injection of network optimal SNR rho at time t_inj adds
    A * shift(u, t_inj) to z0 with A = rho / sqrt(sum_d sigma_d^2), sigma_d = optimal
    SNR of a unit comb in detector d. (Validated against a direct re-injection.)
  * detection statistic = max over delays of the on-source network |z0 + A*u|^2;
    threshold = 95th percentile of the L1-vs-H1 TIME-SLIDE background of the real data
    (false-alarm probability 5%);
  * efficiency eps(rho) = fraction of many off-source injection times detected; the
    90% upper limit rho_90 is where eps = 0.9. We report rho_90 and the implied
    echo-to-event amplitude ratio (GW150914 network SNR ~ 26).

Requires pycbc + network to GWOSC.
"""
from __future__ import annotations
import math
import os
import sys

import numpy as np

try:
    from pycbc.catalog import Merger
    from pycbc.types import TimeSeries
    from pycbc.filter import matched_filter, sigma
    from pycbc.psd import interpolate, inverse_spectrum_truncation
except Exception as exc:  # pragma: no cover
    print("PyCBC unavailable:", exc)
    sys.exit(1)

EVENT = "GW150914"
MASS_FINAL = 61.5          # remnant mass (M_sun)
EVENT_NET_SNR = 26.0       # published GW150914 network SNR (for context)
DELAY_MULT = [0.73, 0.87, 1.00, 1.14, 1.27, 1.41, 1.55]
R_EFF = 0.95
ON_WIN = 0.008
EDGE = 4.0
DUR = 0.35
N_SLIDES = 400
N_INJ = 60
RHO_GRID = [4, 6, 8, 10, 12, 15, 20, 25]
SLIDE_MIN = 0.10
INJ_GUARD = 1.0            # s, keep injections off-source


def echo_comb(sr, dt_echo):
    n = int(DUR * sr)
    t = np.arange(n) / sr
    wf = np.zeros(n)
    f_qnm = 251.0 * (65.0 / MASS_FINAL)
    tau = 3.6e-3 * (MASS_FINAL / 65.0)
    for k in range(1, max(1, int(DUR / dt_echo)) + 1):
        idx = t >= k * dt_echo
        te = t[idx] - k * dt_echo
        sign = -1.0 if (k % 2) else 1.0
        wf[idx] += sign * (R_EFF ** k) * np.exp(-te / tau) * np.sin(2 * np.pi * f_qnm * te)
    return wf


def template(strain, dt_echo):
    tmpl = TimeSeries(echo_comb(strain.sample_rate, dt_echo), delta_t=strain.delta_t)
    tmpl.resize(len(strain))
    return tmpl


def condition(strain):
    strain = strain.highpass_fir(15, 512)
    psd = strain.psd(4)
    psd = interpolate(psd, strain.delta_f)
    psd = inverse_spectrum_truncation(psd, int(4 * strain.sample_rate), low_frequency_cutoff=15)
    return strain, psd


def main():
    print("=" * 88)
    print(f"  NVG ECHO AMPLITUDE UPPER LIMIT  --  injection-recovery on real {EVENT} data")
    print("=" * 88)

    m = Merger(EVENT)
    t0 = m.time
    dt0 = 0.022 * (MASS_FINAL / 65.0)

    dets = {}
    for det in ("H1", "L1"):
        try:
            dets[det] = condition(m.strain(det))
        except Exception as exc:
            print(f"  {det}: no data ({exc})")
    if len(dets) < 2:
        print("  Need both detectors for a coherent limit."); return

    # complex SNR of data (z0) and unit-injection response (u), per detector/delay
    z0, u, sig2 = {}, {}, {}
    grid = None
    for det, (strain, psd) in dets.items():
        z0[det], u[det] = {}, {}
        # unit comb injected at reference time t0 (center): a zero series + comb at t0
        inj = strain * 0.0
        cen = echo_comb(strain.sample_rate, dt0)
        i0 = int((t0 - float(strain.start_time)) * strain.sample_rate)
        inj.data[i0:i0 + len(cen)] += cen[:max(0, len(inj) - i0)][:len(cen)]
        sig2[det] = sigma(template(strain, dt0), psd=psd, low_frequency_cutoff=20) ** 2
        for mult in DELAY_MULT:
            tm = template(strain, dt0 * mult)
            zc = matched_filter(tm, strain, psd=psd, low_frequency_cutoff=20).crop(EDGE + DUR, EDGE)
            uc = matched_filter(tm, inj, psd=psd, low_frequency_cutoff=20).crop(EDGE + DUR, EDGE)
            z0[det][mult] = np.array(zc)
            u[det][mult] = np.array(uc)
            if grid is None:
                grid = np.array(zc.sample_times) - t0
    dt_grid = grid[1] - grid[0]
    sig_net = math.sqrt(sum(sig2.values()))    # network optimal SNR of a unit comb
    n = len(grid)

    def net_stat(shift_data_samps, A=0.0, inj_shift=0):
        """max over delays of max on-source of |z_H1(+A u) + z_L1(+A u)|^2, with L1 time-slid."""
        best = 0.0
        # on-source window centered on the injection (index 0 == real merger)
        i_center = int(round(inj_shift)) + int(np.argmin(np.abs(grid)))
        w = max(1, int(ON_WIN / dt_grid))
        lo, hi = max(0, i_center - w), min(n, i_center + w + 1)
        for mult in DELAY_MULT:
            zh = z0["H1"][mult].copy()
            zl = np.roll(z0["L1"][mult], shift_data_samps)
            if A:
                uh = np.roll(u["H1"][mult], inj_shift)
                ul = np.roll(u["L1"][mult], inj_shift)  # coherent: same inj time both dets
                zh = zh + A * uh
                zl = zl + A * ul
            tot = np.abs(zh) ** 2 + np.abs(zl) ** 2
            best = max(best, float(tot[lo:hi].max()))
        return best

    # ---- time-slide background of the real data -> detection threshold (FAP 5%) ----
    min_sh, max_sh = int(SLIDE_MIN / dt_grid), n - int(SLIDE_MIN / dt_grid)
    shifts = np.unique(np.linspace(min_sh, max_sh, N_SLIDES).astype(int))
    bkg = np.array([net_stat(int(s)) for s in shifts])
    thr95 = float(np.percentile(bkg, 95))
    S0 = net_stat(0)
    p0 = (1.0 + int(np.sum(bkg >= S0))) / (1.0 + len(bkg))
    print(f"  Zero-lag S0 = {S0:.2f} netSNR^2   background median = {np.median(bkg):.2f}, "
          f"95th pct = {thr95:.2f}  ->  p(S0) = {p0:.3f}")
    print(f"  Detection threshold (FAP 5%): netSNR^2 > {thr95:.2f}")
    print(f"  Unit-comb network optimal SNR sigma_net = {sig_net:.3e} (H1^2+L1^2)")

    # self-check: linearity vs a direct injection at one off-source time
    t_inj_chk = 3.0
    ish = int(round(t_inj_chk / dt_grid))
    A_chk = 10.0 / sig_net
    lin = net_stat(0, A=A_chk, inj_shift=ish)
    print(f"  [self-check] linear-injection stat at rho=10, t=+3s: {lin:.2f} netSNR^2 "
          f"(should exceed threshold; ok if > {thr95:.1f})")

    # ---- injection-recovery efficiency ----
    off_shifts = [int(round(tt / dt_grid)) for tt in
                  np.linspace(INJ_GUARD, (n * dt_grid) / 2 - INJ_GUARD, N_INJ)]
    print("-" * 88)
    print(f"  {'rho_inj':>8} {'efficiency (FAP 5%)':>22} {'implied echo/event ratio':>28}")
    print("  " + "-" * 62)
    eff = {}
    for rho in RHO_GRID:
        A = rho / sig_net
        hits = 0
        for ish in off_shifts:
            stat = net_stat(0, A=A, inj_shift=ish)
            if stat > thr95:
                hits += 1
        eff[rho] = hits / len(off_shifts)
        print(f"  {rho:>8.1f} {eff[rho]:>22.2f} {rho / EVENT_NET_SNR:>28.3f}")

    # interpolate rho at 90% efficiency
    rr = np.array(RHO_GRID, float)
    ee = np.array([eff[r] for r in RHO_GRID])
    rho90 = None
    for i in range(len(rr) - 1):
        if ee[i] < 0.9 <= ee[i + 1]:
            rho90 = rr[i] + (0.9 - ee[i]) * (rr[i + 1] - rr[i]) / (ee[i + 1] - ee[i])
            break
    print("-" * 88)
    if rho90:
        print(f"  90% UPPER LIMIT: echo combs with network SNR > {rho90:.1f} are excluded "
              f"(FAP 5%).")
        print(f"  i.e. an echo train fainter than ~{rho90/EVENT_NET_SNR:.2f} x the GW150914 signal")
        print(f"  ({EVENT_NET_SNR:.0f} SNR) would have escaped detection -- so the non-detection")
        print("  constrains only relatively LOUD echoes; deep limits need the full O3/O4 stack.")
    else:
        print("  Efficiency did not reach 90% on the tested rho grid; extend RHO_GRID upward.")
    print("=" * 88)


if __name__ == "__main__":
    main()
