#!/usr/bin/env python3
"""
NVG: B-L cogenesis with a dark-neutron partner — the surviving eta_B door
==========================================================================
nvg_baryogenesis_bsm_closure.py closed every GENERATION channel and stated
the requirement for a viable extension: B-violation below 432 MeV that is
proton-safe BY SYMMETRY. This script builds that extension explicitly and
computes every number it must survive.

CONSTRUCTION (three ingredients):
  1. A dark Dirac fermion chi with (B-L)_chi = +1; TOTAL B-L exactly
     conserved and equal to ZERO for all time — no proton-decay operator
     can ever be generated (symmetry protection, not gating).
  2. The neutron portal O = (u d d) chi / Lambda^2: converts visible
     baryons into chi (and back), conserving total B-L while violating
     B_vis - B_dark — the charge the theta-dot chemical potential biases.
     Spontaneous COGENESIS at the recondensation then separates a net
     zero into equal-and-opposite asymmetries: n_B = -n_chi charge-wise,
     N_B = N_chi in number.
  3. A theta-chi Yukawa (y ~ 0.02) to annihilate the symmetric chi
     component into theta-bosons (m_theta = 6.6 MeV) — required, because
     the portal alone leaves a 30x symmetric overabundance (computed).

PROTON/NEUTRON SAFETY — kinematic corridor (computed):
     m_n - S_n(Be-9)  <  m_chi  <  m_p + m_e
     937.900 MeV      <  m_chi  <  938.783 MeV        (width 0.883 MeV)
  Lower edge: bound neutrons in the most fragile nucleus cannot decay to
  chi (SNO/Borexino nuclear-stability bounds evaded kinematically).
  Upper edge: chi cannot decay to p e nu — absolutely stable dark matter.
  The dark-neutron mass is pinned to 0.05% by stability alone.

WHAT IT BUYS AND WHAT IT COSTS — all computed below:
  + eta_B is GENERATED (first viable mechanism after all closures);
  + the same Lambda fixed by eta_B predicts a free-neutron dark branching
    Br(n -> chi gamma) ~ 4e-6 — a genuine laboratory target (~30x below
    current bounds; falsifiable floor, since Br ~ 1/Lambda^4 rises if
    Lambda drops);
  + Omega_chi = (m_chi/m_p) Omega_b: an 18.6% dark-neutron component of DM.
  - The full-ADM reading fails: Omega_DM/Omega_b = 5.364 +/- 0.065 is NOT
    an integer multiple of m_chi/m_p (N=5 off by 5.6 sigma, N=6 by 9.8) —
    chi must be a SUBCOMPONENT and PBHs remain the dominant, calibrated DM.
  - Lambda ~ 556 TeV is calibrated to eta_B: one new scale, one number.
"""

from __future__ import annotations
import math

# ── masses and nuclear inputs (MeV unless noted) ────────────────────────
M_P, M_E, M_N = 938.272, 0.511, 939.565
S_N_BE9 = 1.6654            # neutron separation energy of Be-9 (tightest)
BETA_LAT = 0.0144           # GeV^3, lattice <0|udd|n>
ALPHA_EM = 1.0 / 137.036
TAU_N_GEV = 7.5e-28         # free-neutron width, GeV

# THETA IDENTIFICATION UPDATE (nvg_theta_sector_audit.py): the condensate
# phase is the eta-prime (958 MeV), not a 6.6-MeV mode; with the thermal cap
# mu ~ pi T the cogenesis scale becomes Lambda ~ 1.7 PeV and the laboratory
# floor Br(n -> chi gamma) ~ 4e-8 (x100 below the numbers printed by this
# script, which are retained as the m_theta = 6.6 MeV variant for record).

# ── cosmology / cogenesis inputs ────────────────────────────────────────
T_STAR = 0.2                # GeV
M_THETA = 6.6e-3            # GeV (theta-dot bias = m_theta)
M_PL = 1.22e19
G_STAR = 20.0
ETA_TARGET = 8.6e-11        # n_B/s
OMEGA_RATIO_OBS, OMEGA_RATIO_ERR = 0.1200 / 0.02237, 0.065
BR_BOUND_NOW = 1e-4         # current n -> chi gamma search sensitivity


def main():
    print("=" * 78)
    print("  NVG: B-L COGENESIS WITH A DARK NEUTRON — CONSTRUCTED AND TESTED")
    print("=" * 78)

    # 1. stability corridor
    lo, hi = M_N - S_N_BE9, M_P + M_E
    m_chi = 0.5 * (lo + hi)
    dm = (M_N - m_chi) * 1e-3            # GeV
    print(f"\n1. Kinematic corridor: {lo:.3f} < m_chi < {hi:.3f} MeV "
          f"(width {hi-lo:.3f} MeV)")
    print(f"   nucleons cannot decay to chi; chi absolutely stable — by masses,")
    print(f"   not by gating. Take mid-corridor m_chi = {m_chi:.3f} MeV.")
    assert hi > lo and (hi - lo) < 1.0

    # 2. cogenesis scale (same arithmetic as the closure, now safe)
    s_dens = (2.0 * math.pi ** 2 / 45.0) * G_STAR * T_STAR ** 3
    eta_eq = (M_THETA * T_STAR ** 2 / 6.0) / s_dens
    eff = ETA_TARGET / eta_eq
    lam_eq = (T_STAR ** 3 * M_PL / (1.66 * math.sqrt(G_STAR))) ** 0.25
    lam = lam_eq * eff ** -0.25
    print(f"\n2. Cogenesis: bias mu/T = {M_THETA/T_STAR:.3f}, required Gamma/H = {eff:.1e}")
    print(f"   => Lambda = {lam/1e3:.0f} TeV (calibrated to eta_B — one scale, one number)")
    print(f"   Total B-L = 0 for all time: proton-decay operators forbidden by symmetry.")

    # 3. neutron portal phenomenology at that Lambda
    theta_mix = BETA_LAT / (lam ** 2 * dm)
    br = theta_mix ** 2 * ALPHA_EM * dm ** 3 / (8.0 * (M_N * 1e-3) ** 2) / TAU_N_GEV
    print(f"\n3. Same Lambda predicts the free-neutron dark channel:")
    print(f"   n-chi mixing theta = {theta_mix:.1e}")
    print(f"   Br(n -> chi gamma) ~ {br:.1e}  (current bounds ~{BR_BOUND_NOW:.0e};")
    print(f"   a x{BR_BOUND_NOW/br:.0f} improvement tests the mechanism — a FLOOR, since")
    print(f"   lowering Lambda to raise eta_B raises Br as 1/Lambda^4)")
    assert br < BR_BOUND_NOW, "portal must pass current neutron bounds"

    # 4. symmetric component and the theta-Yukawa
    bias = M_THETA / T_STAR
    sym_over_asym = 1.0 / bias
    sv_needed = 2.6e-9                                  # GeV^-2 (thermal)
    y = (sv_needed * 16.0 * math.pi * (m_chi * 1e-3) ** 2) ** 0.25
    sig_self = y ** 4 * (m_chi * 1e-3) ** 2 / (4.0 * math.pi * M_THETA ** 4)
    som = sig_self * 0.389e-27 / (m_chi * 1e-3 * 1.783e-24)
    print(f"\n4. Symmetric chi component = {sym_over_asym:.0f}x the asymmetric one —")
    print(f"   the portal cannot annihilate it (computed in the closure script);")
    print(f"   a theta-chi Yukawa y >= {y:.3f} burns it via chi chi-bar -> theta theta.")
    print(f"   Induced self-interaction sigma/m = {som:.1e} cm^2/g (bound ~1) — safe.")
    assert som < 0.1

    # 5. relic prediction and the Omega ratio test
    print(f"\n5. Relic: N_chi = N_B exactly (cogenesis) => per dark flavor")
    print(f"   Omega_chi/Omega_b = m_chi/m_p = {m_chi/M_P:.4f} — an 18.6% DM subcomponent.")
    print(f"   Full-ADM reading (chi = ALL of DM, N flavors):")
    for N in (5, 6):
        r = N * m_chi / M_P
        pull = abs(r - OMEGA_RATIO_OBS) / OMEGA_RATIO_ERR
        print(f"     N = {N}: ratio {r:.3f} vs {OMEGA_RATIO_OBS:.3f} +/- {OMEGA_RATIO_ERR} "
              f"-> {pull:.1f} sigma — excluded")
        assert pull > 5.0
    print(f"   => chi is a SUBCOMPONENT; PBHs remain the dominant (calibrated) DM;")
    print(f"   Omega_DM/Omega_b = 5.364 is NOT derived by this construction.")

    print(f"""
VERDICT: the eta_B door closes on a WORKING mechanism — the first to
survive every computed constraint: cogenesis through a neutron portal
with exact B-L, proton-safe by symmetry, nucleon-safe by the 0.88-MeV
kinematic corridor, radiatively stable (no gate), neutron-phenomenology
safe by x{BR_BOUND_NOW/br:.0f}. Price: one calibrated scale (Lambda = {lam/1e3:.0f} TeV), one
pinned mass (938.34 +/- 0.44 MeV), one bounded Yukawa (y >= {y:.3f}).
Reward: a falsifiable laboratory floor Br(n -> chi gamma) ~ {br:.0e} and a
predicted dark-neutron DM fraction in units of 18.6%. The hoped-for full
derivation of Omega_DM/Omega_b = 5.36 does NOT materialize (integer
multiples excluded at >5 sigma) — stated plainly.
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
