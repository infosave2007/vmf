#!/usr/bin/env python3
"""
NVG Verification: tidal heating of compact objects -- a null test of the
horizonless-remnant hypothesis.

A black hole absorbs tidal gravitational radiation through its horizon.
At leading (quadrupole, circular-orbit) order the absorbed flux is the
infinity flux suppressed by (v/c)^5 (Poisson-Sasaki / Alvi scaling):

    F_inf = (32/5) (c^5/G) (mu/M)^2 (v/c)^10,
    F_H   ~ F_inf * (v/c)^5   (Schwarzschild horizon, l = m = 2),

while a HORIZONLESS NVG remnant has no absorptive boundary condition:
its Planck-thin surface reflects instead of absorbing, so tidal heating
vanishes to leading order. The observable imprint is a GW phase deficit

    dPhi ~ 2 pi N_cyc * (F_H / F_inf) ~ 2 pi N_cyc (v/c)^5

in the inspiral, for a Kerr BH primary; it is ZERO for a remnant.

This script computes the fluxes and dephasing for:
  (1) a representative LISA EMRI (M = 1e6 M_sun, mu = 10 M_sun);
  (2) a stellar-mass LIGO binary (30 + 30 M_sun);
and states the forward falsifiable content.
"""

import math

# constants (CGS)
C = 2.99792458e10      # cm/s
G = 6.67430e-8         # cm^3 g^-1 s^-2
MSUN = 1.98892e33      # g


def flux_infinity(mu_g: float, m_g: float, v_over_c: float) -> float:
    """Quadrupole GW flux to infinity, erg/s (circular orbit)."""
    return (32.0 / 5.0) * C**5 / G * (mu_g / m_g)**2 * v_over_c**10


def flux_horizon(mu_g: float, m_g: float, v_over_c: float) -> float:
    """Horizon absorption flux, erg/s (leading (v/c)^5 suppression)."""
    return flux_infinity(mu_g, m_g, v_over_c) * v_over_c**5


def l_eddington(m_g: float) -> float:
    return 1.26e38 * (m_g / MSUN)


def report(label: str, m_msun: float, mu_msun: float, v: float,
           n_cycles: float):
    m, mu = m_msun * MSUN, mu_msun * MSUN
    f_inf = flux_infinity(mu, m, v)
    f_h = flux_horizon(mu, m, v)
    d_phi = 2.0 * math.pi * n_cycles * v**5
    print(f"[{label}]")
    print(f"    M = {m_msun:.0e} M_sun, mu = {mu_msun:.0f} M_sun, "
          f"v/c = {v:.2f}, N_cyc = {n_cycles:.0e}")
    if m_msun >= 1.0e4:
        print(f"    F_infinity = {f_inf:.2e} erg/s "
              f"({f_inf/l_eddington(m):.2e} L_Edd)")
    else:
        print(f"    F_infinity = {f_inf:.2e} erg/s "
              f"(L_Edd ratio not meaningful at merger)")
    print(f"    F_horizon  = {f_h:.2e} erg/s "
          f"(ratio F_H/F_inf = {v**5:.2e})")
    print(f"    dephasing for a Kerr BH primary:  dPhi ~ {d_phi:.1f} rad "
          f"({d_phi/(2*math.pi):.1f} cycles)")
    print(f"    dephasing for an NVG remnant:     dPhi = 0 (no horizon)")
    return d_phi


def main():
    print("=" * 72)
    print(" NVG TIDAL HEATING NULL TEST (HORIZONLESS REMNANTS)")
    print("=" * 72)

    d1 = report("1 LISA EMRI", 1.0e6, 10.0, 0.20, 1.0e5)
    print("-" * 72)
    d2 = report("2 LIGO stellar-mass merger", 30.0, 30.0, 0.30, 1.0e3)
    print("-" * 72)

    print("[3] VERDICT:")
    print(f"    EMRI phase deficit ~ {d1/(2*math.pi):.0f} cycles is at the")
    print("    edge of LISA waveform resolution (O(1) cycle) but competes")
    print("    with spin/eccentricity systematics; the stellar-mass case")
    print(f"    (~ {d2/(2*math.pi):.1f} cycles) is marginal.")
    print("    NULL PREDICTION: NVG remnants show NO horizon-absorption")
    print("    phase term and no associated low-frequency dissipation;")
    print("    a statistically significant Kerr-consistent absorption")
    print("    signature in EMRIs falsifies the remnant interpretation.")
    print("=" * 72)
    print("STATUS: forward null test; quantitative imprint computed above.")
    print("=" * 72)


if __name__ == "__main__":
    main()
