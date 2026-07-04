#!/usr/bin/env python3
"""
NVG: dark-matter budget audit — three sectors, one Omega
=========================================================
Fresh cross-review finding: three parts of the repository each size a
dark-matter component against (nearly) the FULL observed density,
independently of one another:

  1. relic condensate defects: lambda_v = 1.02 INVERTED from
     Omega_DM = 0.268 (nvg_relic_dark_matter.py);
  2. PBH ladder: abundance peak CALIBRATED to Omega_PBH ~ 0.26
     (nvg_pbh_dark_matter.py);
  3. dark neutrons chi: Omega_chi = 0.049 FIXED by cogenesis
     (nvg_adm_bl_cogenesis.py — not adjustable).

Sum: 0.577 vs the observed 0.264 — the budget is over-subscribed x2.19.
Each claim was written before the others existed; nobody added them up.

RESOLUTION (computed): declare the budget once. Since chi is fixed and
the PBH abundance is only REQUIRED at the JWST-seeding level
(f_PBH ~ 1e-9, nvg_pbh_two_population.py), the natural allocation is

    defects 0.215  +  chi 0.049  +  PBH trace  =  0.264.

The defect re-sizing is nearly free because the abundance depends on
lambda_v only logarithmically through the instanton action
S = 8 pi^2/(3 lambda_v):
    lambda_v: 1.02 -> 1.011  (a 0.9% shift),
    m_W = sqrt(2 lambda_v) W_0: 1229 -> 1222 MeV (f0(1370) association
    intact), vacuum-release factor 1 + lambda_v/4: 1.255 -> 1.253.
Downstream consequences: the asteroid-window PBH abundance peak is no
longer needed (PBHs become the trace JWST-seed population), and the
dark-matter composition statement becomes
    ~81% condensate defects + ~19% dark neutrons + trace PBH,
which keeps the direct-detection null test (neither component scatters
in nuclear-recoil detectors above current sensitivity) and the GWTC
null result (trace PBHs invisible to LVK) consistent.
"""

from __future__ import annotations
import math

OMEGA_DM = 0.264
CLAIMS = [("condensate defects (relic inversion)", 0.268),
          ("PBH ladder (calibrated peak)",          0.26),
          ("dark neutrons chi (cogenesis, fixed)",  0.049)]
LAMBDA_V_OLD = 1.02
W0 = 859.0


def main():
    print("=" * 78)
    print("  NVG: DARK-MATTER BUDGET AUDIT — THREE SECTORS, ONE OMEGA")
    print("=" * 78)

    total = sum(v for _, v in CLAIMS)
    print(f"\n  Claimed components (each sized independently):")
    for name, v in CLAIMS:
        print(f"    {name:<42} {v:.3f}")
    print(f"    {'SUM':<42} {total:.3f}  vs observed {OMEGA_DM}")
    print(f"    -> over-subscribed x{total/OMEGA_DM:.2f}")
    assert total > 1.5 * OMEGA_DM

    # resolution: defects take the remainder after chi; PBH -> trace
    omega_chi = CLAIMS[2][1]
    omega_def = OMEGA_DM - omega_chi
    S_old = 8.0 * math.pi ** 2 / (3.0 * LAMBDA_V_OLD)
    S_new = S_old + math.log(CLAIMS[0][1] / omega_def)
    lam_new = 8.0 * math.pi ** 2 / (3.0 * S_new)
    m_w_old = math.sqrt(2.0 * LAMBDA_V_OLD) * W0
    m_w_new = math.sqrt(2.0 * lam_new) * W0

    print(f"\n  Resolution: defects {omega_def:.3f} + chi {omega_chi:.3f} + PBH trace "
          f"= {OMEGA_DM}")
    print(f"    lambda_v : {LAMBDA_V_OLD} -> {lam_new:.3f}  "
          f"({100*(lam_new/LAMBDA_V_OLD-1):+.1f}%, log-insensitive)")
    print(f"    m_W      : {m_w_old:.0f} -> {m_w_new:.0f} MeV "
          f"(f0(1370) association intact)")
    print(f"    1+lam/4  : {1+LAMBDA_V_OLD/4:.3f} -> {1+lam_new/4:.3f} "
          f"(g-candidate bracket unchanged)")
    print(f"""
  Composition statement (single, repo-wide):
    ~{100*omega_def/OMEGA_DM:.0f}% condensate defects + ~{100*omega_chi/OMEGA_DM:.0f}% dark neutrons + trace PBH
  Consequences: the asteroid-window PBH abundance calibration is no
  longer needed (PBHs = trace JWST-seed population, f_PBH ~ 1e-9);
  direct-detection and GWTC null results remain consistent.
""")
    print("=" * 78)

    assert abs(lam_new - 1.011) < 0.005
    assert abs(m_w_new - 1222) < 5


if __name__ == "__main__":
    main()
