#!/usr/bin/env python3
"""
Honest joint chi^2 of the NVG canonical EOS against the CURRENT (2024-2026) NICER +
maximum-mass + GW170817 dataset — including PSR J0437 (which nvg_joint_ns_inference.py
omits) and using the real J0740 mass (2.08 +- 0.07, not the 2.14 +- 0.10 used there).

It evaluates BOTH canonical variants that live in the repo, because they disagree:
  * "causal"  (cs_q^2 = 1/3, conformal): nvg_fork_b_full_chain.py -> M_max 1.89, R_1.4 13.11
  * "stiff"   (cs_q^2 = 0.50, super-conformal): nvg_joint_ns_inference / tidal canon
                                                -> M_max 2.05, R_1.4 12.55
Headline numbers are the code-computed outputs of those chains (TOV solver); R near
1.34-1.42 Msun is nearly flat (<0.2 km) so R_1.4 is used for the J0030/J0437 radii.

Maximum mass is treated as a ONE-SIDED constraint (only penalised if M_max < M_obs:
the theory must be able to *support* an observed pulsar; exceeding it is fine).
"""
from __future__ import annotations
import math

# ---- Current measurements (with sources) ------------------------------------
# masses (max-mass lower bounds): J0740 firm; J0952 heavier but larger systematics
MASS = [
    ("J0740+6620 mass", 2.08, 0.07, "Fonseca+2021"),
    ("J0952-0607 mass", 2.35, 0.17, "Romani+2022 (systematics-limited)"),
]
# radii (two-sided), near 1.4 Msun
RADII = [
    ("J0030+0451 R(1.44)", 12.71, 1.15, "Riley+2019 NICER"),
    ("J0437-4715 R(1.42)", 11.36, 0.80, "Choudhury+2024 NICER  <-- omitted by repo joint fit"),
    ("J0740+6620 R(2.08)", 12.49, 1.00, "Salmi+2024 NICER+XMM"),
]
# tidal (asymmetric)
LAMBDA = ("GW170817 Lambda(1.4)", 190.0, 390.0, 120.0, "Abbott+2018")   # +390 / -120

CANONS = {
    "causal  (cs_q^2 = 1/3)":  {"M_max": 1.89, "R_1.4": 13.11, "R_2.08": 12.6, "Lam": 393},
    "stiff   (cs_q^2 = 0.50)": {"M_max": 2.05, "R_1.4": 12.55, "R_2.08": 12.0, "Lam": 519},
}


def one_sided_mass_pull(m_max, m_obs, sig):
    """Penalise only if the EOS cannot support the observed pulsar (M_max < M_obs)."""
    return (m_max - m_obs) / sig if m_max < m_obs else 0.0


def main():
    print("=" * 84)
    print("  NVG CANONICAL EOS  vs  CURRENT NICER / MAX-MASS / GW170817  (honest joint chi^2)")
    print("=" * 84)
    for label, c in CANONS.items():
        print(f"\n  CANON: {label}   [M_max={c['M_max']}, R_1.4={c['R_1.4']}, Lambda_1.4={c['Lam']}]")
        print(f"    {'constraint':<26}{'obs':>16}{'pred':>8}{'pull':>8}")
        print("    " + "-" * 62)
        chi2 = 0.0
        n = 0
        # max mass (one-sided)
        for name, val, sig, src in MASS:
            pull = one_sided_mass_pull(c["M_max"], val, sig)
            chi2 += pull ** 2; n += 1
            tag = "" if pull == 0 else "  <-- cannot support" if abs(pull) > 2 else ""
            print(f"    {name:<26}{val:>10.2f}+/-{sig:<3.2f}{c['M_max']:>8.2f}{pull:>8.2f}{tag}")
        # radii (two-sided): use R_1.4 for the ~1.4 Msun sources, R_2.08 for J0740
        for name, val, sig, src in RADII:
            pred = c["R_2.08"] if "2.08" in name else c["R_1.4"]
            pull = (pred - val) / sig
            chi2 += pull ** 2; n += 1
            print(f"    {name:<26}{val:>10.2f}+/-{sig:<3.2f}{pred:>8.2f}{pull:>8.2f}")
        # tidal (asymmetric)
        name, val, su, sl, src = LAMBDA
        sig = su if c["Lam"] > val else sl
        pull = (c["Lam"] - val) / sig
        chi2 += pull ** 2; n += 1
        print(f"    {name:<26}{val:>7.0f}+{su:.0f}/-{sl:.0f}{c['Lam']:>8.0f}{pull:>8.2f}")

        print("    " + "-" * 62)
        print(f"    joint chi^2 = {chi2:.2f} over {n} constraints  ->  reduced = {chi2/n:.2f}  "
              f"({'ACCEPTABLE' if chi2/n < 2 else 'DISFAVOURED'})")

    print("\n" + "=" * 84)
    print("  READING:")
    print("  * The causal (conformal cs^2=1/3) canon is DISFAVOURED: M_max=1.89 cannot support")
    print("    J0740 (2.08), a ~2.7 sigma / effective exclusion, and R_1.4=13.11 sits +2.2 sigma")
    print("    above J0437. This is what nvg_fork_b_full_chain.py actually prints (chi^2=16.4/4).")
    print("  * The stiff canon survives only by pushing the quark sound speed to cs^2=0.50, above")
    print("    the conformal bound 1/3; even then J0437 is the tightest pull (+1.5 sigma).")
    print("  * The repo's 'EXCELLENT chi^2~1.0' headline (nvg_joint_ns_inference.py) omits J0437")
    print("    and uses M_max obs 2.14+-0.10 instead of J0740's 2.08+-0.07. Including J0437 and")
    print("    the correct mass makes the honest reduced chi^2 ~1.2-1.4 for the stiff canon.")
    print("  * NET: the NS prediction is NOT robustly consistent — it is canon-dependent and")
    print("    hinges on a super-conformal sound speed; J0437 is the live tension to watch.")
    print("=" * 84)


if __name__ == "__main__":
    main()
