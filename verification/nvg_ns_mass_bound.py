#!/usr/bin/env python3
"""
NVG Verification: Parameter-Free Neutron-Star Mass Bound from the M_Omega Anchor
---------------------------------------------------------------------------------
The bare Chandrasekhar-type scale built from the single frozen anchor
M_Omega = 859 +/- 8 MeV,

    M_bare = M_Pl^3 / M_Omega^2 = 2.211 +/- 0.041 M_sun,

is the only dimensional maximum-mass scale derivable from NVG without any
EOS modeling. This script registers it as a falsifiable upper bound on the
gravitational mass of cold non-rotating neutron stars and compares it with
the current observational sample.

Registered falsifier (SPEC-v2 rejection boundary):
    ANY robustly confirmed cold NS mass above the upper band
    (M_bare * (851/859)^2 = 2.252 M_sun) refutes the bound.

Also prints two honest internal audits found while preparing this claim:
    [A] nvg_hyperon_puzzle_tov.py rescales its TOV curves to hardcoded
        targets (2.99 / 2.91 M_sun), which sit well above the bound;
        those numbers are presentation choices, NOT TOV outputs.
    [B] nvg_moment_of_inertia_j0737.py gives a genuine fork-B TOV
        M_max = 2.29 M_sun, 3.6% above the bound -- a live internal
        tension, listed as OPEN.

STATUS UNDER SPEC-v2: dimensional bound from one frozen input; falsifiable
with published data. It is a REGISTERED CLAIM with a live comparison, not a
prediction derived from dynamics (SPEC-v2 item 7: dimensional coincidences
are not evidence -- but a sharp falsifier IS admissible content).
"""

import math

# ------------------------------------------------------------ frozen anchors
M_OMEGA   = 859.0      # MeV (NVG anchor, +/-8)
DM_OMEGA  = 8.0        # MeV
M_PL_KG   = math.sqrt(1.054571817e-34 * 2.99792458e8 / 6.67430e-11)
M_SUN_KG  = 1.98892e30
M_OMEGA_KG = M_OMEGA * 1.78266192e-30   # MeV -> kg

# ------------------------------------------------------------ [0] the bound
m_bare = M_PL_KG**3 / M_OMEGA_KG**2 / M_SUN_KG
dm_hi = m_bare * ((M_OMEGA / (M_OMEGA - DM_OMEGA))**2 - 1.0)   # M_Omega low
dm_lo = m_bare * (1.0 - (M_OMEGA / (M_OMEGA + DM_OMEGA))**2)   # M_Omega high
m_upper = m_bare * (M_OMEGA / (M_OMEGA - DM_OMEGA))**2
m_lower = m_bare * (M_OMEGA / (M_OMEGA + DM_OMEGA))**2

print("=" * 72)
print("  NVG PARAMETER-FREE NS MASS BOUND: M_Pl^3 / M_Omega^2")
print("=" * 72)
print(f"  M_bare = {m_bare:.3f} M_sun   (+{dm_hi:.3f}/-{dm_lo:.3f})")
print(f"  band   = [{m_lower:.3f}, {m_upper:.3f}] M_sun")
print(f"  relation to M_crit: M_bare = (8 sqrt(2 pi)/9) * M_crit "
      f"(factor 2.228)")
print()

# ------------------------------------------------------------ [1] live data
# Gravitational masses of the heaviest reliably measured NSs (literature).
sample = [
    # name,            M [M_sun],  sigma,  reference
    ("PSR J0740+6620", 2.08, 0.07, "NICER+XMM, Fargo+24"),
    ("PSR J1614-2230", 1.97, 0.04, "Shapiro delay, Fonseca+16"),
    ("PSR J0348+0432", 2.01, 0.04, "Shapiro delay, Antoniadis+13"),
    ("PSR J0952-0607", 2.35, 0.17, "spectroscopic, Romani+22 (disputed)"),
]
print("  Live comparison (bound: no cold NS above {:.3f} M_sun):".format(
    m_upper))
print(f"  {'source':<18} {'M [M_sun]':>10} {'margin':>8} {'sigma':>6}  status")
print("  " + "-" * 66)
for name, m, sig, ref in sample:
    margin = (m_upper - m) / sig          # sigmas below the upper band
    if "disputed" in ref:
        status = "TENSE*"
    else:
        status = "ALIVE" if margin > 0 else "REFUTED"
    print(f"  {name:<18} {m:>8.2f}+/-{sig:.2f} {margin:>+7.1f}  "
          f"{status}")
print("  " + "-" * 66)
print("  * TENSE: 0.6 sigma above the band but the measurement is")
print("  contested (companion spectroscopy); NOT treated as a refutation.")
print("  J0740+6620 keeps the bound alive with a ~2.5-sigma margin.")
print()

# ------------------------------------------------------------ [2] audits
print("  Internal audits:")
print("  [A] nvg_hyperon_puzzle_tov.py: TOV curves are rescaled to")
print("      hardcoded targets 2.99 (NL3) / 2.91 (SLy) M_sun. These are")
print("      presentation choices, not TOV outputs, and they violate the")
print("      bound; the figure should be regenerated or re-flagged.")
print("  [B] nvg_moment_of_inertia_j0737.py fork-B TOV M_max = 2.29 M_sun")
print(f"      exceeds the bound by {(2.29/m_bare - 1.0)*100:.1f}% -- OPEN "
      f"tension; resolution needs an EOS re-tune (the fork-B family in")
print("      nvg_fork_b_full_chain.py targets 2.07-2.08 and is consistent).")
print()
print("  Falsification path: a robust cold-NS mass >= 2.25 M_sun (e.g. a")
print("  J0952-class object confirmed by Shapiro delay) kills the bound.")
print("=" * 72)
print("Status: registered dimensional bound from one frozen anchor;")
print("live against current data; SPEC-v2 admissible (sharp falsifier).")
print("=" * 72)
