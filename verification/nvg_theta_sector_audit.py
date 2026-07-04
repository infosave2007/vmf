#!/usr/bin/env python3
"""
NVG: the theta-sector identity audit — one field, four masses
==============================================================
A fresh cross-review of the post-audit repository found that "the theta
mode" carries FOUR incompatible (mass, decay-constant) assignments:

  A. preprint (CP section):   m = 958 MeV   (f = f_pi, "sqrt(chi)/f_pi")
  B. recondensation/cogenesis: m = 6.6 MeV   (sqrt(chi_full)/W_0)
  C. seesaw / ADMX row 52:     m = 53 microeV (f_a = 1.07e11 GeV)
  D. axion row 22:             m = 8.38 microeV (f_a = 1.54e12 GeV)

One field cannot have four masses. This script computes what each
assignment implies and which survive.

RESULTS (computed below):
  A. The 958 MeV value is the Witten-Veneziano eta-prime:
     sqrt(6 chi_quenched)/f_pi = 967 MeV vs m_eta' = 958. This is the
     physically CORRECT identification: the phase of the QCD quark
     condensate IS the eta' direction. No new particle exists at all.
  B. Taken as a real light ALP (m = 6.6 MeV, f = W_0 = 0.86 GeV), the
     photon coupling alpha/(2 pi W_0) ~ 1.4e-3 GeV^-1 is excluded by
     beam dumps (bounds ~1e-5 GeV^-1) by two orders of magnitude. The
     6.6 MeV number was a convention artifact (full-QCD chi divided by
     W_0 instead of the physical eta' normalization) — retired.
  C. Feeding the 53-microeV mass into the cogenesis bias (mu = theta-dot
     ~ m_theta) collapses Lambda to 5 TeV and raises Br(n -> chi gamma)
     to ~1e+2 — six orders above the experimental bound. The invisible
     axion CANNOT be the field that drives cogenesis.
  D. Row 22's (8.38 microeV, 1.54e12 GeV) contradicts row 52's
     (53 microeV, 1.07e11 GeV) by x6.3 in mass — same duplication
     pattern as the retired second neutrino sector.

RESOLUTION (single consistent assignment):
  - theta_condensate = eta' (958 MeV): drives recondensation, strong-CP,
    cogenesis. Cogenesis recomputed with the thermal cap mu ~ pi T:
    Lambda = 1.7 PeV, Br(n -> chi gamma) ~ 4e-8 — still a floor, but
    ~100x below the previous estimate (honest testability downgrade).
  - The ADMX/seesaw sector requires a SEPARATE invisible-axion field
    (f_a = 1.07e11 GeV): kept as an extension, at the cost of row 52's
    "no new U(1)" claim. Row 22 merges into it or retires.
  - Claims that used a light condensate theta at laboratory conditions
    (cell-scale coherence etc.) lose their carrier: the condensate phase
    is a 958-MeV meson with tau ~ 3e-21 s.
"""

from __future__ import annotations
import math

CHI_FULL = 75.5 ** 4     # MeV^4, full-QCD topological susceptibility
CHI_QUEN = 191.0 ** 4    # MeV^4, quenched (enters Witten-Veneziano)
F_PI, W0 = 92.4, 859.0   # MeV
M_ETA_P = 957.8          # MeV
ALPHA_EM = 1.0 / 137.036
BEAM_DUMP_BOUND = 1e-5   # GeV^-1, ALP-photon coupling at MeV masses
T_STAR = 200.0           # MeV
M_PL = 1.22e19           # GeV
G_STAR = 20.0
ETA_TARGET = 8.6e-11
BR_OLD, LAM_OLD = 3.6e-6, 5.56e5   # previous cogenesis numbers (m = 6.6 MeV)
BR_BOUND = 1e-4


def main():
    print("=" * 78)
    print("  NVG: THETA-SECTOR IDENTITY AUDIT — ONE FIELD, FOUR MASSES")
    print("=" * 78)

    # A. Witten-Veneziano check
    m_wv = math.sqrt(6.0 * CHI_QUEN) / F_PI
    print(f"\nA. Witten-Veneziano: sqrt(6 chi_quen)/f_pi = {m_wv:.0f} MeV "
          f"vs m_eta' = {M_ETA_P} — the condensate phase IS the eta'.")
    assert abs(m_wv - M_ETA_P) / M_ETA_P < 0.05

    # B. light-ALP exclusion
    m_b = math.sqrt(CHI_FULL) / W0
    g_agg = ALPHA_EM / (2.0 * math.pi * W0 * 1e-3)
    excl = g_agg / BEAM_DUMP_BOUND
    print(f"\nB. '6.6 MeV theta' as a real ALP: m = {m_b:.2f} MeV, "
          f"g_agg = {g_agg:.1e} GeV^-1")
    print(f"   beam-dump bound ~{BEAM_DUMP_BOUND:.0e} -> excluded x{excl:.0f}. "
          f"Convention artifact; retired.")
    assert excl > 50

    # C. cogenesis with each mass
    def cogenesis(mu_mev):
        mu = min(mu_mev, math.pi * T_STAR) * 1e-3        # GeV, thermal cap
        t = T_STAR * 1e-3
        s = (2 * math.pi ** 2 / 45) * G_STAR * t ** 3
        eta_eq = (mu * t ** 2 / 6.0) / s
        eff = ETA_TARGET / eta_eq
        lam_eq = (t ** 3 * M_PL / (1.66 * math.sqrt(G_STAR))) ** 0.25
        lam = lam_eq * eff ** -0.25
        br = BR_OLD * (LAM_OLD / lam) ** 4
        return lam, br

    print(f"\nC. Cogenesis under each theta identification:")
    for label, mu in (("eta' (958 MeV, capped at pi T)", 958.0),
                      ("6.6 MeV (retired)", 6.6),
                      ("53 microeV (invisible axion)", 53e-9)):
        lam, br = cogenesis(mu)
        ok = "OK" if br < BR_BOUND else "EXCLUDED (Br above bound)"
        print(f"   {label:<34} Lambda = {lam/1e3:8.1f} TeV, "
              f"Br(n->chi gamma) = {br:.1e}  {ok}")
    lam_ep, br_ep = cogenesis(958.0)
    _, br_inv = cogenesis(53e-9)
    assert br_ep < BR_BOUND and br_inv > BR_BOUND

    # D. rows 22 vs 52
    print(f"\nD. Axion rows: row 22 (8.38 microeV, f_a = 1.54e12 GeV) vs "
          f"row 52 (53 microeV, 1.07e11 GeV)")
    print(f"   -> x{53/8.38:.1f} in mass: duplicated sector, must merge or retire.")

    print(f"""
RESOLUTION: theta_condensate = eta' (no new light particle; correct QCD).
Cogenesis survives with Lambda = {lam_ep/1e3:.0f} TeV and a lab floor
Br(n -> chi gamma) ~ {br_ep:.0e} (x100 below the earlier 6.6-MeV estimate —
an honest testability downgrade). The ADMX/seesaw axion is a SEPARATE
extension field; row 22 merges into it. Light-theta laboratory-coherence
claims lose their carrier (the eta' lives 3e-21 s).
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
