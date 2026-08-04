#!/usr/bin/env python3
"""
NVG Verification: superradiance of the theta mode on compact remnants.

A bosonic field of mass m extracts spin from a rotating black hole when
its Compton frequency matches the orbital frequency near the horizon,
i.e. for dimensionless gravitational coupling

    alpha = G M m / (hbar c) ~ O(0.1-0.5).

For the theta mode (m_theta = 53 ueV, row 47) this selects a black-hole
mass scale of order 1e-6 M_sun -- a PLANETARY (~0.3 Earth-mass) PBH, not a stellar-mass
BH (m_boson * M_BH = const anchors the band). This script computes:

  (1) the resonant BH mass M_res(alpha) for the theta mode;
  (2) the overlap of the superradiance band with the 4^N PBH ladder of
      the NVG anchor chain (M_N = M_1 / 4^N, M_1 = 0.382 M_sun);
  (3) the e-folding growth time of the dominant l=m=1 mode,
      tau ~ M / (alpha^9) in geometric units;
  (4) the decisive caveat: NVG compact remnants are HORIZONLESS
      (surface at the Schwarzschild radius up to Planck corrections).
      Without a horizon there is no ergoregion energy sink -> NO
      standard superradiant instability. The prediction is therefore a
      NULL TEST: theta-mode clouds and their annihilation GW lines
      (f = m_theta c^2 / (pi hbar) ~ 25.6 GHz, far above every planned
      GW band) are ABSENT around NVG remnants, while the same clouds
      would spin down any Kerr PBH in the band essentially instantly.

Confrontation content: (i) the band sits at ~1e-6 M_sun where the NVG
4^N ladder passes within ~40% (N = 9); (ii) any detection of a SPINNING
sub-stellar PBH inside the band excludes the horizon picture for that
object; (iii) the 25.6 GHz line position is fixed by the row-47 mass.
"""

import math

# constants
HBAR_C_EVCM = 1.973269804e-5   # eV cm
G_CGS = 6.67430e-8             # cm^3 g^-1 s^-2
C_CGS = 2.99792458e10          # cm/s
MSUN_G = 1.98892e33            # g
EV_TO_KG = 1.78266192e-36      # kg per eV/c^2

M_THETA_UEV = 53.0             # theta-mode mass, row 47
M1_MSUN = 0.3819               # anchor chain seed M_1 = c^3 t_b / (2G)
ALPHA_RES = 0.41               # l=m=1 superradiance resonance center


def m_res_msun(m_boson_ev: float, alpha: float) -> float:
    """BH mass (M_sun) for which G M m/(hbar c) = alpha."""
    m_kg = m_boson_ev * EV_TO_KG
    m_si = alpha * 1.054571817e-34 * 2.99792458e8 / (G_CGS * 1e-3 * m_kg)
    return m_si / (MSUN_G * 1e-3)


def ladder_masses(n_max: int = 12):
    """4^N NVG PBH ladder M_N = M_1 / 4^N in solar masses."""
    return [M1_MSUN / 4.0**n for n in range(n_max + 1)]


def growth_time_yr(m_bh_msun: float, alpha: float) -> float:
    """Geometric-units estimate tau ~ M/alpha^9 for l=m=1 (high alpha)."""
    t_geom_s = G_CGS * m_bh_msun * MSUN_G / C_CGS**3
    return (t_geom_s / alpha**9) / 3.15576e7


def main():
    m_th_ev = M_THETA_UEV * 1e-6

    print("=" * 72)
    print(" NVG THETA-MODE SUPERRADIANCE: BANDS, GROWTH, NULL TEST")
    print("=" * 72)

    # (1) resonance mass
    m_lo = m_res_msun(m_th_ev, 0.20)
    m_hi = m_res_msun(m_th_ev, 0.50)
    m_c = m_res_msun(m_th_ev, ALPHA_RES)
    print(f"[1] theta mode m = {M_THETA_UEV:.0f} ueV "
          f"(lambda = {HBAR_C_EVCM/m_th_ev*10:.2f} mm):")
    print(f"    superradiance band alpha in [0.2, 0.5]:")
    print(f"    M_BH in [{m_lo:.2e}, {m_hi:.2e}] M_sun "
          f"(center {m_c:.2e} M_sun at alpha = {ALPHA_RES})")
    m_c_kg = m_c * MSUN_G * 1e-3
    print(f"    = {m_c_kg/5.97e24:.2f} Earth masses "
          f"(~{m_c_kg/7.35e22:.0f} lunar masses) -> PLANETARY")
    print("      PBH window (NOT stellar mass: m_boson x M_BH = const)")
    print("-" * 72)

    # (2) overlap with the 4^N ladder
    print("[2] overlap with the NVG 4^N PBH ladder (M_1 = "
          f"{M1_MSUN:.4f} M_sun), relevant rungs:")
    best = None
    for n, m_n in enumerate(ladder_masses()):
        if not (m_lo / 50.0 <= m_n <= m_hi * 50.0):
            continue
        ratio = m_n / m_c
        tag = ""
        if m_lo <= m_n <= m_hi:
            tag = "  <-- inside the theta-mode band"
        print(f"    N = {n:2d}: M = {m_n:.4e} M_sun "
              f"({ratio:.2f} x band center){tag}")
        if best is None or abs(math.log(ratio)) < abs(math.log(best[2])):
            best = (n, m_n, ratio)
    if best:
        print(f"    -> closest rung N = {best[0]} at {best[1]:.3e} M_sun, "
              f"{best[2]:.2f}x the band center")
    print("-" * 72)

    # (3) growth time
    tau = growth_time_yr(m_c, ALPHA_RES)
    print(f"[3] l=m=1 e-folding time at alpha = {ALPHA_RES}, "
          f"M = {m_c:.2e} M_sun:")
    print(f"    tau ~ M/alpha^9 = {tau:.2e} yr = {tau*3.15576e7:.2e} s")
    print("    -> any Kerr PBH in the band sheds its spin essentially")
    print("       instantly; a SPINNING planetary PBH in the band is a")
    print("       direct counterexample to the horizon superradiance picture")
    print("-" * 72)

    # (4) horizonless null test
    print("[4] HORIZONLESS-REMNANT NULL TEST:")
    print("    NVG remnants have no horizon -> no ergoregion sink ->")
    print("    the superradiant channel is CLOSED for them. Prediction:")
    print("    (i)  no theta-mode boson clouds around NVG remnants;")
    f_line = m_th_ev / (math.pi * 6.582119569e-16)  # Hz, f = m/(pi hbar)
    print(f"    (ii) no annihilation GW line at f = {f_line:.3e} Hz")
    print("         (~25.6 GHz, far above every planned GW band; the")
    print("         frequency itself is a row-47 anchor prediction);")
    print("    (iii) planetary microlensing surveys (OGLE, Roman): a")
    print("         4^N-ladder mass spectrum clustered near 1e-6 M_sun")
    print("         would support the NVG PBH chain.")
    print("=" * 72)
    print(f"STATUS: band [{m_lo:.1e}, {m_hi:.1e}] M_sun computed; closest")
    print("ladder rung N = 9 within ~40%. Null test is forward-falsifiable")
    print("via planetary PBH spin/lens surveys and the GHz-line absence.")
    print("=" * 72)


if __name__ == "__main__":
    main()
