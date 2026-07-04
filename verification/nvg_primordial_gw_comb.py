#!/usr/bin/env python3
"""
NVG Verification: Primordial Gravitational Wave Background Comb.
Derives the redshifted bounce frequencies for the Tolman cycle sequence
and checks which cycles fall in the Pulsar Timing Array (PTA) nHz band.

Derivation (all quantities from the repo's own anchors):
  - Every bounce occurs at the same critical density rho_c = M_Omega^4/(hbar c)^3,
    so every bounce emits at the same physical frequency f_b ~ 1/t_b with
    t_b = (8 pi G rho_c / 3)^(-1/2) = 3.76e-6 s.
  - Redshift from the cycle-77 bounce (T_b = 432 MeV) to today (T_0 = 2.7255 K)
    with entropy conservation g_S a^3 T^3 = const:
      f_77 = (1/t_b) * (T_0/T_b) * (g_S0/g_Sb)^(1/3) = 62.9 nHz.
    (An earlier version used 145 nHz — that is (1/t_b)*(T_0/T_b) with the
    entropy g-factor omitted.)
  - Inter-cycle spacing from the corrected Tolman law (bounce mass x2 per
    cycle, derived from matter-era turnaround dynamics in
    nvg_tolman_law_derivation.py): bounce scale x2^(1/3), teeth spaced by
    2^(-1/3) = 0.794.
  - Convention caveat: the bounce emission peak may sit at 1/t_b or 1/(2 pi t_b),
    an O(2 pi) spread (anchor 10-63 nHz); either choice keeps the comb in the
    PTA band.

AMPLITUDES: see the companion nvg_gw_comb_amplitude.py — treating the
W-condensate recondensation after the bounce as a first-order transition,
the sound-wave mechanism gives Omega_GW h^2 ~ 1e-10..3e-7 peaked at
26-72 nHz, bracketing the NANOGrav 15yr signal; teeth from earlier cycles
are diluted by 4^(4/3) = 6.35 per step, so the comb appears as a dominant
bump with a ~16% sub-tooth at 0.63 f. The remaining underived inputs are
the transition parameters (alpha, beta/H).
"""

import math

# ── Repo anchors ──────────────────────────────────────────────────────
HBAR_C = 197.3269804     # MeV fm
G_CGS = 6.674e-8         # cm^3 g^-1 s^-2
C_CGS = 2.998e10         # cm/s
MEV_FM3_TO_GCM3 = 1.7827e12
T_0_MEV = 2.7255 * 8.617333e-11   # CMB temperature today, MeV
G_S_BOUNCE = 47.5        # entropy dof at the QGP bounce
G_S_TODAY = 3.91         # photons + neutrinos
# Corrected Tolman law (nvg_tolman_law_derivation.py): matter-era turnaround
# dynamics fixes the bounce-mass growth at x2 per cycle (not x4), so the
# bounce scale grows by 2^(1/3) and consecutive teeth are spaced by:
SPACING = 2.0 ** (-1.0 / 3.0)     # = 0.794


def bounce_frequency_today(m_omega: float) -> float:
    """f_77 in nHz, derived from the QCD anchor via t_b and adiabatic redshift."""
    eps_max = m_omega ** 4 / HBAR_C ** 3          # MeV/fm^3
    rho_c = eps_max * MEV_FM3_TO_GCM3             # g/cm^3
    t_b = (8.0 * math.pi * G_CGS * rho_c / 3.0) ** -0.5
    # Bounce temperature via Stefan-Boltzmann with g_* = 47.5 (as in the repo)
    t_b_mev = (30.0 * m_omega ** 4 / (math.pi ** 2 * G_S_BOUNCE)) ** 0.25
    g_factor = (G_S_TODAY / G_S_BOUNCE) ** (1.0 / 3.0)
    return (1.0 / t_b) * (T_0_MEV / t_b_mev) * g_factor * 1e9   # nHz


def calculate_comb_frequencies(m_omega: float) -> dict[int, float]:
    f_0 = bounce_frequency_today(m_omega)
    return {k: f_0 * SPACING ** (77 - k) for k in range(48, 78)}


def main():
    print("=" * 70)
    print(" NVG PRIMORDIAL GRAVITATIONAL WAVE BACKGROUND COMB (derived)")
    print("=" * 70)

    m_omega_center = 859.0
    m_omega_err = 8.0

    f_center = bounce_frequency_today(m_omega_center)
    freq_center = calculate_comb_frequencies(m_omega_center)
    freq_lower = calculate_comb_frequencies(m_omega_center - m_omega_err)
    freq_upper = calculate_comb_frequencies(m_omega_center + m_omega_err)

    pta_min, pta_max = 1.0, 1000.0

    print(f"QCD Anchor M_Omega_0  : {m_omega_center} +/- {m_omega_err} MeV")
    print(f"Derived anchor f_77   : {f_center:.1f} nHz "
          f"= (1/t_b)(T_0/T_b)(g_S0/g_Sb)^(1/3)")
    print(f"Derived tooth spacing : 4^(-1/3) = {SPACING:.3f} (Tolman M x4 per cycle)")
    print(f"Convention spread     : {f_center/(2*math.pi):.1f}-{f_center:.1f} nHz "
          f"(peak at 1/(2 pi t_b) vs 1/t_b)")
    print()
    print(f"  {'Cycle (k)':<12} | {'Derived Frequency (nHz)':<35} | {'In PTA Band?'}")
    print("-" * 68)

    pta_cycles = []
    for k in sorted(freq_center.keys()):
        val, low, upp = freq_center[k], freq_lower[k], freq_upper[k]
        in_pta = pta_min <= val <= pta_max
        if in_pta:
            pta_cycles.append(k)
        tag = "YES" if in_pta else ("NO (Too Low)" if val < pta_min else "NO (Too High)")
        print(f"  Cycle {k:<7} | {val:.2f} ({low:.2f} - {upp:.2f})"
              f"{'':<12} | {tag}")

    print("-" * 68)
    print(f"Result: cycles {pta_cycles[0]}-{pta_cycles[-1]} fall in the PTA band;")
    print(f"the teeth overlap the NANOGrav (2023) stochastic-signal band.")
    print("Amplitudes: nvg_gw_comb_amplitude.py (sound-wave mechanism; brackets the")
    print("NANOGrav 15yr signal for alpha ~ 0.2-0.5, beta/H ~ 1-10).")
    print("=" * 70)

    assert 40.0 < f_center < 90.0, "derived anchor should be ~63 nHz"
    assert len(pta_cycles) >= 8, "comb should populate the PTA band"


if __name__ == "__main__":
    main()
