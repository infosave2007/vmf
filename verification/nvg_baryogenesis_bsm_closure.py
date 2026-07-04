#!/usr/bin/env python3
"""
NVG: the BSM route to eta_B — every channel computed and closed
================================================================
Prior results: inheritance through cycles is excluded by 13 orders
(nvg_etab_inheritance.py), so the asymmetry must be REGENERATED each
cycle. This script computes every regeneration channel available to the
framework. The chain closes them all.

STRUCTURAL FACT (stated for the first time): the bounce caps the density
at rho_c, so the temperature of the NVG universe NEVER exceeds
T_b = 432 MeV — in this cycle or any other. Electroweak baryogenesis,
leptogenesis, GUT baryogenesis: unavailable IN PRINCIPLE, forever.
All B-violation must operate at T <= 432 MeV.

CHANNEL 1 — electroweak sphalerons at T_b: rate ~ exp(-E_sph/T) with
E_sph ~ 9 TeV: exponent ~ -21,000. Dead (computed).

CHANNEL 2 — electroweak instantons coupled to the theta winding
(theta W-Wdual would violate B+L): vacuum rate ~ exp(-4*pi/alpha_W),
exponent ~ -370. Dead. (The QCD anomaly the theta-mode does couple to
conserves B — SU(3) is vector-like. The "eta_B from topological choice
Q = +1" reading is closed by this line.)

CHANNEL 3 — a BSM dimension-6 operator (qqql)/Lambda^2 driven by
spontaneous baryogenesis (mu_B = theta-dot ~ m_theta during
recondensation). Computed below:
  - the scale reproducing eta_B is Lambda ~ 500 TeV (out-of-equilibrium
    regime, Gamma/H ~ 1e-7);
  - the SAME operator, ungated, gives a proton lifetime ~ 0.1 s vs the
    Super-K bound 2.4e34 yr — excluded by ~43 orders of magnitude;
  - the gate that would reconcile them must suppress the vacuum
    coefficient by ~1e-22 in amplitude. Both in-model gates fail,
    computed: (a) the W-melting gate (1 - W*^2/W0^2)^n has base ~ 0.27
    inside nucleons — needs n ~ 39 powers (non-EFT); and if the hadron
    interior restores W -> 0 (bag picture), the gate turns fully ON
    inside the proton — the wrong direction entirely; (b) the
    theta-dot derivative gate is not radiatively stable: the vacuum
    fluctuation <(d theta)^2> ~ W0^2/(16 pi^2) exceeds the gate
    normalization m_theta^2 by a factor ~100, regenerating the ungated
    operator with a LARGE coefficient.

VERDICT: within the current field content NVG has NO viable baryogenesis
mechanism. Combined with the inheritance closure, eta_B is promoted from
"scale estimate" to the model's sharpest OPEN PROBLEM / falsifier, and
the pi*sqrt(T_b/M_Pl) ansatz is withdrawn as a claim (a dimensional
coincidence, retained only as a note). An honest positive statement
survives: any future NVG extension must add B-violation that is (i)
active at T <= 432 MeV, (ii) proton-safe by symmetry (not by gating) —
e.g. an exactly conserved B-L with a dark-sector asymmetry partner.
"""

from __future__ import annotations
import math

T_B = 0.432          # GeV, bounce temperature (max temperature ever)
E_SPH = 9000.0       # GeV, sphaleron barrier
ALPHA_W = 1.0 / 30.0
M_PL = 1.22e19       # GeV
G_STAR = 20.0        # dof at T ~ 0.2 GeV
M_THETA = 6.6e-3     # GeV, theta-mode mass (= mu_B scale during roll)
T_STAR = 0.2         # GeV, exit temperature of the recondensation
ETA_TARGET = 8.6e-11  # n_B/s
M_P = 0.938          # GeV
TAU_P_BOUND_S = 2.4e34 * 3.156e7   # Super-K p -> e+ pi0, seconds
HBAR_GEV_S = 6.58e-25
W0 = 0.859           # GeV


def main():
    print("=" * 78)
    print("  NVG: BSM BARYOGENESIS — EVERY CHANNEL COMPUTED")
    print("=" * 78)

    print(f"\n  Structural fact: T_max = T_b = {T_B*1e3:.0f} MeV in EVERY cycle")
    print(f"  (the bounce caps rho at rho_c). All B-violation must act below it.")

    # ── channel 1: sphalerons ───────────────────────────────────────────
    exp1 = -E_SPH / T_B
    print(f"\n  1. EW sphalerons at T_b: rate ~ exp({exp1:,.0f}) — dead.")

    # ── channel 2: EW instantons via theta winding ──────────────────────
    exp2 = -4.0 * math.pi / ALPHA_W
    print(f"  2. EW instantons (theta-W-Wdual): rate ~ exp({exp2:.0f}) "
          f"(10^{exp2/math.log(10):.0f}) — dead.")
    print(f"     (QCD anomaly conserves B: SU(3) vector-like — the 'Q = +1")
    print(f"      topological choice' route closes here.)")

    # ── channel 3: dim-6 operator, spontaneous baryogenesis ─────────────
    print(f"\n  3. BSM operator (qqql)/Lambda^2 with mu_B = theta-dot ~ m_theta:")
    H_star = 1.66 * math.sqrt(G_STAR) * T_STAR ** 2 / M_PL
    s_dens = (2.0 * math.pi ** 2 / 45.0) * G_STAR * T_STAR ** 3
    eta_eq = (M_THETA * T_STAR ** 2 / 6.0) / s_dens
    ratio_needed = ETA_TARGET / eta_eq            # = Gamma/H required
    lam_eq = (T_STAR ** 3 * M_PL / (1.66 * math.sqrt(G_STAR))) ** 0.25
    lam = lam_eq * (1.0 / ratio_needed) ** 0.25
    print(f"     equilibrium eta_eq = {eta_eq:.1e}; required Gamma/H = {ratio_needed:.1e}")
    print(f"     => Lambda_B = {lam/1e3:.0f} TeV")

    tau_p_s = (lam ** 4 / M_P ** 5) * HBAR_GEV_S
    deficit = TAU_P_BOUND_S / tau_p_s
    print(f"     ungated proton lifetime: {tau_p_s:.2f} s vs bound {TAU_P_BOUND_S:.1e} s")
    print(f"     => excluded by {deficit:.1e} (~{math.log10(deficit):.0f} orders);")
    amp_gate = 1.0 / math.sqrt(deficit)
    print(f"     a rescuing gate must suppress the vacuum amplitude to {amp_gate:.0e}.")

    # gate (a): W-melting power gate
    base = 1.0 - 0.857 ** 2                       # in-medium W*/W0 at n_0
    n_needed = math.log(amp_gate) / math.log(base)
    print(f"\n     gate (a) W-melting (1 - W*^2/W0^2)^n: base = {base:.2f} in nucleons")
    print(f"       -> needs n = {n_needed:.0f} powers (non-EFT); and a bag-model")
    print(f"       interior (W -> 0) turns the gate fully ON inside the proton.")

    # gate (b): theta-dot derivative gate, radiative stability
    fluct = W0 ** 2 / (16.0 * math.pi ** 2)       # <(d theta)^2>, cutoff W0
    regen = fluct / M_THETA ** 2
    print(f"     gate (b) (d theta)^2/m_theta^2: vacuum <(d theta)^2> = {fluct:.1e} GeV^2")
    print(f"       -> regenerates the ungated operator with coefficient {regen:.0f} > 1:")
    print(f"       not radiatively stable.")

    print(f"""
  VERDICT: all three channels are closed by computation. Together with
  the inheritance closure (2^76 dilution), NVG as formulated has NO
  mechanism for the baryon asymmetry. Status change: eta_B row moves
  from "scale estimate" to OPEN PROBLEM / falsifier; the
  pi*sqrt(T_b/M_Pl) ansatz is withdrawn as a claim. Requirement on any
  future extension: B-violation active below 432 MeV that is
  proton-safe BY SYMMETRY (e.g. exact B-L with a dark-sector partner
  asymmetry), not by gating.
""")
    print("=" * 78)

    assert exp1 < -1e4 and exp2 < -300
    assert 100e3 < lam < 2e6, "operator scale should be O(100 TeV - 1 PeV)"
    assert deficit > 1e40
    assert regen > 10


if __name__ == "__main__":
    main()
