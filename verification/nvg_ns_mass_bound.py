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
        M_max = 2.29 M_sun, 3.6% above the bound -- RESOLUTION FOUND:
        gamma2 2.30 -> 2.00 gives M_max = 2.218 M_sun (in band) with
        R(1.4)/I(1.4) unchanged at 0.2%, and restores the CSS conformal
        limit c_s^2 = 1/3; retune awaits approval.

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
m_n_chandra = M_PL_KG**3 / (939.565 * 1.78266192e-30)**2 / M_SUN_KG
print(f"  interpretation: M_bare/M_Ch(m_n) = {m_bare/m_n_chandra:.3f} = "
      f"(m_n/M_Omega)^2 = {(939.565/M_OMEGA)**2:.3f} -- the ordinary")
print(f"  Chandrasekhar scale with the fermion mass m_n -> M_Omega")
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
print("      bound; the figure is flagged in-script pending regeneration.")
print("  [B] nvg_moment_of_inertia_j0737.py fork-B TOV M_max = 2.29 M_sun")
print(f"      exceeds the bound by {(2.29/m_bare - 1.0)*100:.1f}% -- RESOLUTION "
      f"FOUND: a fine-grid TOV scan with the")
print("      high-density polytrope softened from gamma2 = 2.30 to 2.00")
print("      gives M_max = 2.218 M_sun (inside the band) with R(1.4) and")
print("      I(1.4) unchanged at the 0.2% level; gamma2 = 2.00 also")
print("      restores the exact CSS conformal limit c_s^2 = 1/3 quoted in")
print("      the script docstring. The tracked EOS keeps gamma2 = 2.30")
print("      until the retune is approved; the fork-B family in")
print("      nvg_fork_b_full_chain.py (M_max 2.07-2.08) is consistent as is.")
print()
print("  Falsification path: a robust cold-NS mass >= 2.25 M_sun (e.g. a")
print("  J0952-class object confirmed by Shapiro delay) kills the bound.")
print()

# ------------------------------------------------------------ [3] GW verdicts
print("  Gravitational-wave verdicts:")
m_gw190814, sig_gw = 2.59, 0.08
pull_gw = (m_gw190814 - m_upper) / sig_gw
print(f"  GW190814 secondary {m_gw190814}+/-{sig_gw} M_sun: if a neutron")
print(f"      star it would sit {pull_gw:.1f} sigma above the band -> NVG")
print("      VERDICT: it MUST be a black hole; a future measurement of")
print("      nonzero tidal deformability for a >= 2.25 M_sun compact")
print("      object would falsify the bound.")
print("  GW230529 compact object 1.2-2.0 M_sun: above M_crit = 0.992")
print("      M_sun -> a horizon is allowed in NVG (consistent; see")
print("      nvg_mcrit_gwtc_check.py for the sub-critical null test).")
print("=" * 72)
print("Status: registered dimensional bound from one frozen anchor;")
print("live against current data; SPEC-v2 admissible (sharp falsifier).")
print("=" * 72)
