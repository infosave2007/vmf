#!/usr/bin/env python3
"""
NVG Verification: observer card -- post-merger echo numbers for the
LIGO/Virgo O4 candidates (concrete values for GW follow-up analyses).

For the four O4 BBH candidates of the ROADMAP echo program
(GW230518, GW230615, GW230922, GW231215, total mass ~ 62-70 M_sun) this
prints the complete template-parameter card:

  * predicted echo spacing Delta t_echo (Hayward-core model, same physics
    as nvg_ligo_o4_echo_candidates.py, rho_c from the single QCD anchor);
  * remnant-mass systematic (M_f = 0.95 M_tot vs total mass);
  * ringdown carrier frequency f_220 and damping time tau (repo scaling);
  * search window, bandpass, number of pulses and reflectivity decay for
    the comb template used by nvg_echo_upper_limit.py;
  * status against the repo's own null results (time-slide background
    stack 0.0 sigma; injection-recovery limit constrains only loud echoes).

Honest scope: Delta t_echo is logarithmically sensitive to the core
cutoff; the repo searches therefore scan a x0.73-x1.55 delay multiplier
window, which also covers the Kerr-tortoise variant (0.005 s at 65 M_sun,
nvg_gw_echo_prediction.py). Numbers below are template CENTERS.

Requires numpy/scipy (for the imported candidates module).
"""

from nvg_ligo_o4_echo_candidates import calculate_echo_delay

# --- repo template conventions (nvg_echo_upper_limit.py) ---
F_QNM_65 = 251.0        # Hz, carrier frequency at 65 M_sun remnant
TAU_65 = 3.6e-3         # s, damping time at 65 M_sun remnant
R_EFF = 0.95            # effective reflectivity per bounce (template input)
DUR = 0.35              # s, comb duration used by the searches
F_REM = 0.95            # remnant mass fraction of the total mass

# O4 ROADMAP candidates: name -> total source-frame mass (M_sun)
CANDIDATES = {
    "GW230518_174026": 65.4,
    "GW230615_091807": 61.8,
    "GW230922_191834": 70.2,
    "GW231215_223405": 63.5,
}


def card(name, m_tot):
    m_rem = F_REM * m_tot
    res_tot = calculate_echo_delay(m_tot)
    res_rem = calculate_echo_delay(m_rem)
    if res_tot is None or res_rem is None:
        return None
    dt, dt_rem = res_tot["delta_t_echo_s"], res_rem["delta_t_echo_s"]
    f_qnm = F_QNM_65 * (65.0 / m_rem)
    tau = TAU_65 * (m_rem / 65.0)
    n_pulses = int(DUR / dt)
    amp_last = R_EFF ** n_pulses
    return {
        "name": name, "m_tot": m_tot, "m_rem": m_rem,
        "dt_ms": dt * 1e3, "dt_rem_ms": dt_rem * 1e3,
        "f_qnm": f_qnm, "tau_ms": tau * 1e3,
        "n_pulses": n_pulses, "amp_last": amp_last,
        "f_rep": 1.0 / dt,
    }


def main():
    print("=" * 88)
    print("  NVG ECHO OBSERVER CARD -- LIGO/VIRGO O4 CANDIDATES (ROADMAP program)")
    print("=" * 88)
    hdr = (f"{'event':<18}{'M_tot':>6}{'M_rem':>6}{'dt_echo':>9}"
           f"{'dt(M_f)':>9}{'f_QNM':>7}{'tau':>7}{'N_puls':>7}{'R^N':>7}")
    print(hdr)
    print("-" * 88)
    rows = []
    for name, m in CANDIDATES.items():
        c = card(name, m)
        if c is None:
            continue
        rows.append(c)
        print(f"{c['name']:<18}{c['m_tot']:>6.1f}{c['m_rem']:>6.1f}"
              f"{c['dt_ms']:>8.1f}m{c['dt_rem_ms']:>8.1f}m"
              f"{c['f_qnm']:>7.0f}{c['tau_ms']:>6.1f}m{c['n_pulses']:>7d}"
              f"{c['amp_last']:>7.2f}")
    print("-" * 88)
    if rows:
        dt_lo = min(r["dt_rem_ms"] for r in rows)
        dt_hi = max(r["dt_ms"] for r in rows)
        print(f"  Delta t_echo range across the program: {dt_lo:.1f}-{dt_hi:.1f} ms")
        print(f"  (total-mass template; remnant-mass systematic x{F_REM} included)")
    print()
    print("  HOW TO SEARCH (matches the repo pipeline, nvg_echo_upper_limit.py):")
    print(f"    * comb template: {len(rows)} candidates, pulses at t_c + k*dt_echo,")
    print(f"      k = 1..N with alternating sign and amplitude R_eff^k, R_eff = {R_EFF}")
    print(f"    * per-pulse carrier: damped sinusoid at f_QNM above, tau listed")
    print(f"    * search window: t_c + dt_echo .. t_c + {DUR:.2f} s (DUR = {DUR} s)")
    print("    * bandpass: 20-500 Hz; delay scan x0.73-x1.55 around the tabulated")
    print("      dt_echo (covers the Kerr-tortoise variant, ~0.005 s at 65 M_sun)")
    print("    * statistic: coherent H1+L1 matched-filter comb SNR vs time-slide")
    print("      background (FAP 5%)")
    print("-" * 88)
    print("  STATUS AGAINST THE REPO'S OWN NULL RESULTS:")
    print("    * time-slide stack of real data: 0.0 sigma (no echo comb found);")
    print("    * GW150914 injection-recovery: only LOUD combs are excluded --")
    print("      the non-detection so far constrains amplitude, not the delay;")
    print("    * these four O4 events are the next targets: public GWOSC strain,")
    print("      the card above gives every template parameter needed.")
    print("-" * 88)
    print("  FALSIFIER: a coherent comb at ANY of the tabulated delays (within")
    print("  the x0.73-x1.55 scan) with network SNR above the time-slide")
    print("  threshold would be the first NVG signal; a null deep stack of the")
    print("  full O4 catalog at the tabulated delays pushes R_eff toward 0,")
    print("  i.e. falsifies the reflective-core picture of the remnant.")
    print("=" * 88)


if __name__ == "__main__":
    main()
