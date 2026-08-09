#!/usr/bin/env python3
"""
NVG Verification: Sigma m_nu = 59 meV -- the cornered theta-seesaw sum
------------------------------------------------------------------------
The minimal theta-seesaw (nvg_neutrino_seesaw.py: no nu_R, m_1 ~ 0, normal
ordering, f_a = 1.07e11 GeV) predicts a SHARP sum

    Sigma m_nu = m_2 + m_3 = sqrt(dm2_21) + sqrt(dm2_31) ~ 59.0 meV,

with essentially no free slack: the oscillation splittings fix it. This
script monitors the prediction against the live cosmological bounds.

Registered status (2026-08):
    - Planck 2018 alone:        UL 120 meV -> alive, margin +61 meV
    - DESI DR1 + CMB + SN:      UL  72 meV -> alive, margin only +13 meV
    - DESI DR2 2025 (disputed): Sigma = -75 +- 51 meV -> pull of the
      prediction 2.6 sigma; NOT treated as a refutation while the
      negative-mass posteriors remain contested (systematics of the
      BAO + lensing combination).
    - CMB-S4 forecast sigma(Sigma) ~ 15 meV: a ~4-sigma exclusion of the
      prediction if the central stays near zero; a ~2-sigma confirmation
      if it lands near 59 meV. Two-sided and near-term.

FALSIFIER (SPEC-v2 rejection boundary): a robust cosmological determination
Sigma m_nu < 45 meV (or a confirmed negative best fit) excludes the minimal
theta-seesaw; a measurement in [50, 70] meV confirms it against the
hierarchical alternatives that allow sums down to 58.9-60 meV only at
m_1 ~ 0. Note the prediction sits at the MINIMUM of the normal-ordering
band: any robust Sigma > 90 meV would also require m_1 > 0 and stress the
minimal (m_1 ~ 0) version.

STATUS: live and cornered -- one of the sharpest near-term tests of the
whole framework.
"""

import math

# ---- prediction (from nvg_neutrino_seesaw.py; oscillation data fixed) --
DM2_21 = 7.53e-5     # eV^2 (NuFIT 5.2, NO)
DM2_31 = 2.45e-3     # eV^2
SIGMA_PRED = math.sqrt(DM2_21) + math.sqrt(DM2_21 + DM2_31)  # eV, m_1 ~ 0

# ---- live bounds and forecasts -----------------------------------------
# name, kind, value [meV], error [meV], note
sample = [
    ("Planck 2018 (TT+lensing)",      "UL", 120.0, None, ""),
    ("DESI DR1 + CMB + SN, 95% UL",   "UL",  72.0, None, "margin only +13 meV"),
    ("DESI DR2 2025 + CMB",           "BF", -75.0, 51.0, "disputed"),
    ("CMB-S4 forecast",               "FC",   0.0, 15.0, "two-sided, ~2030"),
]

print("=" * 72)
print("  NVG THETA-SEESAW SUM: Sigma m_nu (cornered prediction)")
print("=" * 72)
print(f"  prediction: Sigma m_nu = {SIGMA_PRED*1e3:.1f} meV "
      f"(m_1 ~ 0, normal ordering; the minimum of the NO band)")
print()
print(f"  {'dataset':<34} {'result':>12}  verdict")
print("  " + "-" * 66)
for name, kind, val, err, note in sample:
    if kind == "UL":
        margin = val - SIGMA_PRED * 1e3
        status = "ALIVE" if margin > 0 else "EXCLUDED"
        print(f"  {name:<34} {'< ' + f'{val:.0f}':>10} meV  {status}"
              f" (margin {margin:+.0f} meV) {note}")
    elif kind == "BF":
        pull = (SIGMA_PRED * 1e3 - val) / err
        print(f"  {name:<34} {val:+.0f}+-{err:.0f} meV  "
              f"TENSE* (pull {pull:.1f} sigma) {note}")
    else:
        excl = SIGMA_PRED * 1e3 / err
        print(f"  {name:<34} {'sigma ' + f'{err:.0f}':>10} meV  "
              f"FORECAST: {excl:.1f} sigma exclusion if central ~0; "
              f"{note}")
print("  " + "-" * 66)
print("  * TENSE: negative-mass posteriors remain contested; NOT a")
print("  refutation until systematics of the BAO+lensing combination settle.")
print()
print("  Falsification paths:")
print("    - robust Sigma < 45 meV or confirmed negative best fit ->")
print("      minimal theta-seesaw excluded")
print("    - robust Sigma in [50, 70] meV -> confirmation; hierarchical")
print("      m_1 > 0 alternatives are simultaneously squeezed")
print("    - robust Sigma > 90 meV -> stresses the minimal m_1 ~ 0 form")
print("=" * 72)
print("Status: live, cornered; sharpest near-term cosmological test of")
print("the theta-seesaw sector (SPEC-v2 admissible, two-sided).")
print("=" * 72)
