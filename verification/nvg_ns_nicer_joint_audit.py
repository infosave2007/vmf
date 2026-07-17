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
    # 2025 addition: edge-on 1.44-Msun pulsar, one of the most compact NICER radii.
    # Asymmetric errors +1.01/-0.86; predictions sit ABOVE the value, so the upper
    # error is the relevant one and is used here. Preferred-model dependence noted
    # in the paper itself (single hot-spot geometry class; EOS-ensemble impact they
    # quote is a milder ~ -300 m shift of the allowed M-R region).
    ("J0614-3329 R(1.44)", 10.29, 1.01, "Mauviard+2025 NICER+XMM (arXiv:2506.14883)"),
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

    # combined ~1.4-Msun NICER radius (inverse-variance, symmetrised errors)
    r14 = [(12.71, 1.15), (11.36, 0.80), (10.29, 0.5 * (1.01 + 0.86))]
    w = [1.0 / s**2 for _, s in r14]
    mean = sum(v * wi for (v, _), wi in zip(r14, w)) / sum(w)
    err = math.sqrt(1.0 / sum(w))
    print(f"\n  Combined NICER R(~1.4): {mean:.2f} +/- {err:.2f} km "
          f"(J0030 + J0437 + J0614, inverse-variance)")

    print("\n" + "=" * 84)
    print("  READING (updated 2026-07 with J0614-3329):")
    print("  * The causal (conformal cs^2=1/3) canon is now solidly DISFAVOURED: cannot support")
    print("    J0740, R_1.4=13.11 sits +2.2 sigma above J0437 and +2.8 sigma above J0614.")
    print("  * The stiff canon (cs^2=0.50, super-conformal) degrades from reduced chi^2 1.08 to")
    print("    ~1.6: J0614 (+2.2 sigma) overtakes J0437 (+1.5) as the tightest pull. Two")
    print("    independent NICER sources now pull the SAME direction (canon too large), and the")
    print("    combined ~1.4-Msun NICER radius sits ~2 sigma below the canonical 12.55 km.")
    print("  * Caveats: J0614 is a single preferred-geometry result (paper's own EOS-ensemble")
    print("    impact is a milder -300 m); the dashboard kill condition 'confirmed R_1.4<12.0'")
    print("    is NOT yet met (10.29+1.01 does not exclude 12.0). Trend, not verdict.")
    print("  * NVG cannot take the strange-quark-star escape route that J0614 papers discuss:")
    print("    the self-bound branch is excluded by the nuclear-existence constraint")
    print("    (nvg_selfbound_gate.py). A more compact canon must come from within")
    print("    the screened hadronic+crossover family, likely at the cost of M_max or Lambda.")
    print("=" * 84)


if __name__ == "__main__":
    main()
