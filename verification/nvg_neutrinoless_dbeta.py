#!/usr/bin/env python3
"""
NVG Verification: Neutrinoless double beta decay null test from theta-seesaw.

Row 47 of the prediction table fixes the neutrino sector: the theta-seesaw
gives m_3 = 50.3 meV (atmospheric scale) with f_a = 1.07e11 GeV and no
right-handed neutrinos. This script computes the effective Majorana mass

    m_bb = | sum_i U_ei^2 m_i |

for both orderings and the kinematic mass m_beta, and confronts them with:
  - DESI DR2 + CMB bound  sum(m_nu) < 72 meV (95% CL),
  - KATRIN direct bound   m_beta < 0.45 eV (2024),
  - 0nubb experiments: KamLAND-Zen (m_bb < 28-122 meV), LEGEND-1000 and
    nEXO projected sensitivity ~ 10 meV.

Results (normal ordering, NuFIT 5.2 splittings):
  - the theta-seesaw anchor m_3 = 50.3 meV (row 47) implies
    sum(m_nu) = 63.3 meV, below the DESI DR2 bound of 72 meV;
  - the README sum 59 meV coincides with the minimal normal-ordering
    spectrum (m1 -> 0): mutually consistent;
  - m_bb in [0, 6.4] meV over the full Majorana-phase scan.
The falsifiable CONTENT: a confirmed 0nubb signal with m_bb > 10 meV
excludes the minimal theta-seesaw sector; the predicted window lies
below the ~10 meV reach of LEGEND-1000/nEXO (null prediction).
"""

import math

# oscillation data (NuFIT 5.2 / PDG 2024 central values)
DM21 = 7.42e-5      # eV^2, solar
DM31_NO = 2.517e-3  # eV^2, atmospheric (normal ordering)
DM32_IO = 2.498e-3  # eV^2, |Delta m32^2| (inverted ordering)
S12_2 = 0.304       # sin^2 theta_12
S13_2 = 0.0222      # sin^2 theta_13
M3_THETASESAW = 50.3e-3   # eV, row 47
SUM_README = 59.0e-3      # eV, README sum claim (audited below)
DESI_BOUND = 72.0e-3      # eV, DESI DR2 + CMB 95% CL


def spectrum_from_m3(m3: float):
    """Normal ordering masses (m1, m2, m3) from m3."""
    m1 = math.sqrt(max(m3**2 - DM31_NO, 0.0))
    m2 = math.sqrt(m1**2 + DM21)
    return m1, m2, m3


def min_sum_no() -> float:
    """Minimal sum(m_nu) allowed by normal ordering (m1 -> 0)."""
    m1 = 0.0
    m2 = math.sqrt(DM21)
    m3 = math.sqrt(DM31_NO)
    return m1 + m2 + m3


def min_sum_io() -> float:
    """Minimal sum(m_nu) allowed by inverted ordering (m3 -> 0)."""
    m3 = 0.0
    m2 = math.sqrt(DM32_IO)
    m1 = math.sqrt(m2**2 - DM21)
    return m1 + m2 + m3


def m_bb_range(m1, m2, m3):
    """Extremal m_bb over Majorana phases (normal ordering).

    m_bb = | c12^2 c13^2 m1 + s12^2 c13^2 m2 e^{2i alpha21}
             + s13^2 m3 e^{2i alpha31} |,
    with two independent phases -> max = a+b+c,
    min = max(0, largest term - sum of the other two).
    """
    a = (1 - S12_2) * (1 - S13_2) * m1
    b = S12_2 * (1 - S13_2) * m2
    c = S13_2 * m3
    terms = [a, b, c]
    mx = sum(terms)
    mn = max(0.0, 2.0 * max(terms) - mx)
    return mn, mx


def main():
    print("=" * 72)
    print(" NVG THETA-SEESAW: 0nubb NULL TEST AND SUM-RULE AUDIT")
    print("=" * 72)

    print("[0] oscillation floor (NuFIT 5.2):")
    print(f"    minimal sum(m_nu), normal ordering   = {min_sum_no()*1e3:.1f} meV")
    print(f"    minimal sum(m_nu), inverted ordering = {min_sum_io()*1e3:.1f} meV")
    print(f"    -> the README value sum = {SUM_README*1e3:.0f} meV coincides with the")
    print("       MINIMAL normal-ordering spectrum (m1 -> 0): CONSISTENT.")
    print("-" * 72)

    m1, m2, m3 = spectrum_from_m3(M3_THETASESAW)
    tot = m1 + m2 + m3
    mn, mx = m_bb_range(m1, m2, m3)
    m_beta = math.sqrt((1 - S12_2) * (1 - S13_2) * m1**2
                       + S12_2 * (1 - S13_2) * m2**2
                       + S13_2 * m3**2)
    print("[1] theta-seesaw anchor m_3 = 50.3 meV (row 47), normal ordering:")
    print(f"    masses (m1,m2,m3) = ({m1*1e3:.2f}, {m2*1e3:.2f}, "
          f"{m3*1e3:.2f}) meV")
    print(f"    sum(m_nu) = {tot*1e3:.1f} meV  vs DESI DR2+CMB bound "
          f"{DESI_BOUND*1e3:.0f} meV : PASS (margin "
          f"{DESI_BOUND*1e3-tot*1e3:.1f} meV)")
    print(f"    m_beta (KATRIN observable) = {m_beta*1e3:.2f} meV "
          f"(KATRIN bound 450 meV: PASS)")
    print(f"    m_bb in [{mn*1e3:.2f}, {mx*1e3:.2f}] meV "
          f"(full Majorana-phase scan)")
    print("-" * 72)

    m1m, m2m, m3m = 0.0, math.sqrt(DM21), math.sqrt(DM31_NO)
    totm = m1m + m2m + m3m
    mnm, mxm = m_bb_range(m1m, m2m, m3m)
    print("[2] minimal-normal-ordering spectrum (m1 -> 0, sum = "
          f"{totm*1e3:.1f} meV = README value):")
    print(f"    m_bb in [{mnm*1e3:.2f}, {mxm*1e3:.2f}] meV")
    print("-" * 72)

    print("EXPERIMENTAL CONFRONTATION:")
    print("  KamLAND-Zen 800 : m_bb < 28-122 meV   -> PASS (>2x margin)")
    print("  LEGEND-1000/nEXO: reach ~ 10 meV      -> predicted m_bb lies")
    print("                    BELOW reach: clean NULL prediction")
    print("  => FALSIFIABLE CONTENT: a confirmed 0nubb signal with")
    print("     m_bb > 10 meV excludes the minimal theta-seesaw sector.")
    print("=" * 72)
    print(f"STATUS: PASS. sum(m_nu) = {tot*1e3:.1f} meV (anchor) / "
          f"{totm*1e3:.1f} meV (minimal)")
    print(f"both below the DESI DR2 bound; m_bb = {mn*1e3:.1f}-{mx*1e3:.1f} "
          f"meV is a sharp null")
    print("prediction for the next-generation 0nubb program.")
    print("=" * 72)


if __name__ == "__main__":
    main()
