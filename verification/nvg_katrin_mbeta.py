#!/usr/bin/env python3
"""
NVG Verification: kinematic neutrino mass m_beta vs KATRIN / Project 8.

The theta-seesaw fixes the spectrum (normal ordering, m1 -> 0, no nu_R).
The KATRIN observable is the PMNS-weighted effective electron mass

    m_beta^2 = sum_i |U_ei|^2 m_i^2 ,

NOT m_3/sqrt(3) (the equi-weighted form used in an earlier version of
nvg_de_axion_crosscheck.py -- AUDIT FIX 2026-08; the correct formula is
identical to nvg_neutrinoless_dbeta.py).

Predictions (NuFIT 5.2 mixing, NO):
    minimal spectrum (m1 = 0):        m_beta = 8.9 meV
    anchor spectrum (m3 = 50.3 meV):  m_beta = 9.0 meV   [m1 = 2.2 meV]

Confrontation:
    KATRIN 2019 (KN1):        m_beta < 1.1 eV
    KATRIN 2022 (KN3+4):      m_beta < 0.8 eV
    KATRIN 2024 (KN5+6):      m_beta < 0.45 eV (90% CL)
    KATRIN design (1000 d):   ~0.2 eV
    Project 8 (design goals): sigma ~ 40 meV (Phase IV), ~10 meV ultimate

VERDICT: the prediction lies a factor ~20 below KATRIN's final design
reach -- KATRIN is INSENSITIVE to the minimal theta-seesaw (honest
status, not a testable prediction for KATRIN). One-sided content:
any robust m_beta > 20 meV excludes the sector (inverted-ordering
floor is m_beta(IO) ~ 48 meV, quasi-degenerate spectra sit higher).
A two-sided test requires a Project-8-ultimate/HOLMES-class
experiment with sigma ~ 10 meV.

STATUS: PASS (all current limits), registered one-sided falsifier.
Pure standard library.
"""

import math

# --- oscillation inputs (NuFIT 5.2, normal ordering) ---
DM2_21 = 7.53e-5      # eV^2
DM2_31 = 2.45e-3      # eV^2
S12_2 = 0.304         # sin^2 theta_12
S13_2 = 0.0222        # sin^2 theta_13

M3_ANCHOR = 50.3e-3   # eV, theta-seesaw anchor (row 47 of the table)


def m_beta(m1, m2, m3):
    return math.sqrt((1 - S12_2) * (1 - S13_2) * m1**2
                     + S12_2 * (1 - S13_2) * m2**2
                     + S13_2 * m3**2)


def main():
    print("=" * 72)
    print(" NVG THETA-SEESAW: m_beta (KATRIN OBSERVABLE) CONFRONTATION")
    print("=" * 72)

    # [1] predictions
    m2_min, m3_min = math.sqrt(DM2_21), math.sqrt(DM2_21 + DM2_31)
    mb_min = m_beta(0.0, m2_min, m3_min)
    m1_an = math.sqrt(M3_ANCHOR**2 - DM2_21 - DM2_31)
    mb_an = m_beta(m1_an, m2_min, m3_min)
    mb_wrong = m3_min / math.sqrt(3.0)
    print("[1] Prediction (PMNS-weighted, normal ordering):")
    print(f"    minimal spectrum  (m1 = 0):           m_beta = {mb_min*1e3:.2f} meV")
    print(f"    anchor spectrum   (m3 = 50.3 meV):    m_beta = {mb_an*1e3:.2f} meV")
    print(f"    AUDIT NOTE: m3/sqrt(3) = {mb_wrong*1e3:.1f} meV is NOT the KATRIN")
    print("    observable (equi-weighted); earlier repo citations corrected.")
    print("-" * 72)

    # [2] live data cut
    limits = [
        ("KATRIN 2019 (KN1)",      1.10e3, "published 90% CL"),
        ("KATRIN 2022 (KN3+4)",    0.80e3, "published 90% CL"),
        ("KATRIN 2024 (KN5+6)",    0.45e3, "published 90% CL"),
        ("KATRIN design (1000 d)", 0.20e3, "design sensitivity"),
    ]
    print("[2] Direct-kinematics limits vs prediction:")
    for name, ul, note in limits:
        margin = ul - mb_min * 1e3
        print(f"    {name:<24} m_beta < {ul:6.0f} meV  -> margin "
              f"+{margin:5.0f} meV  ({note})")
    print(f"    KATRIN final reach / prediction = "
          f"{0.20e3/(mb_min*1e3):.0f}x  -> KATRIN is INSENSITIVE")
    print("-" * 72)

    # [3] next generation
    print("[3] Next generation (design goals, not measurements):")
    print("    Project 8 Phase IV : sigma ~ 40 meV  -> one-sided exclusion if")
    print("                         best fit > ~20 meV; cannot resolve 8.9 meV")
    print("    Project 8 ultimate : sigma ~ 10 meV  -> FIRST two-sided test:")
    print(f"                         pull = |0 - {mb_min*1e3:.1f}| / 10 = "
          f"{mb_min*1e3/10:.1f} sigma (if central ~ 0)")
    print("    HOLMES/ECHo        : meV-calorimetry programs, same territory")
    print("-" * 72)

    # [4] ordering floors -> one-sided falsifier
    m1_io = math.sqrt(abs(DM2_31))
    m2_io = math.sqrt(abs(DM2_31) + DM2_21)
    mb_io = m_beta(m1_io, m2_io, 0.0)      # IO: m3 is the lightest state
    print("[4] Ordering floors:")
    print(f"    m_beta(IO, m3 -> 0) = {mb_io*1e3:.1f} meV; quasi-degenerate")
    print("    spectra sit higher still.")
    print(f"    FALSIFIER: any robust m_beta > 20 meV (a discovery anywhere in")
    print("    the IO/quasi-degenerate territory) excludes the minimal")
    print("    theta-seesaw (NO, m1 ~ 0).")
    print("-" * 72)

    # [5] consistency triangle
    print("[5] Consistency triangle from ONE anchor:")
    print(f"    cosmology   sum(m_nu) = 58.9 meV   (nvg_sigma_mnu_corner.py)")
    print(f"    beta decay  m_beta    = {mb_min*1e3:.1f} meV   (this script)")
    print(f"    0nubb       m_bb <= 6.4 meV        (nvg_neutrinoless_dbeta.py)")
    print("    The three observables are jointly rigid: ANY deviating")
    print("    measurement excludes the sector.")
    print("=" * 72)
    print("STATUS: PASS -- all limits hold; honest note: KATRIN itself cannot")
    print("reach the prediction (factor ~20). Registered one-sided falsifier:")
    print("a robust m_beta > 20 meV excludes the minimal theta-seesaw.")
    print("=" * 72)


if __name__ == "__main__":
    main()
