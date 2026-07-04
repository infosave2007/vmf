#!/usr/bin/env python3
"""
NVG: can the baryon asymmetry be INHERITED through cycles? (closing test)
==========================================================================
The eta_B analysis (nvg_baryon_asymmetry.py) left a fork: either NVG
generates the asymmetry at each bounce (requires B-violation beyond the
Standard Model at QCD-scale temperatures), or eta_B is an initial condition
inherited through cycles. This script closes the second fork by arithmetic.

Setup. If baryon number is conserved through the bounce, the comoving baryon
number N_B = n_B a^3 is constant across cycles. But the comoving entropy
GROWS by the derived Tolman factor g = 2 per cycle (S_matter x2; see
nvg_tolman_law_derivation.py). The asymmetry per entropy therefore DILUTES:

    eta_n  =  (n_B/s)_n  =  eta_1 / 2^(n-1).

For the observed eta_77 = n_B/s ~ 8.6e-11 after 77 cycles, the first cycle
would need
    eta_1 = 8.6e-11 * 2^76  ~  6.5e12,
i.e. ~10^13 baryons per unit entropy — physically impossible (the asymmetry
parameter is bounded by ~1: at most one net baryon per photon-scale entropy
quantum; realistic bounds are far below 1).

CONCLUSION (computed below): inheritance is EXCLUDED by ~13 orders of
magnitude. The asymmetry must be REGENERATED at (or after) each bounce,
which — combined with frozen electroweak sphalerons at T << 130 GeV —
means NVG requires a B-violating mechanism beyond the Standard Model
operating at the bounce. The eta_B fork is now closed: the only viable
route is per-cycle generation, and the dimensional ansatz pi*sqrt(T_b/M_Pl)
must eventually be derived from that mechanism or withdrawn.

(Even under the superseded x4 entropy law the conclusion is identical and
stronger: eta_1 ~ 8.6e-11 * 4^76 ~ 5e35.)
"""

from __future__ import annotations
import math

ETA_OBS = 8.6e-11        # n_B/s today (from eta_B = 6.1e-10, factor 1/7.04)
G_ENTROPY = 2.0          # derived per-cycle matter-entropy growth factor
N_CYCLES = 77
ETA_MAX_PHYSICAL = 1.0   # absolute ceiling: |n_B|/s cannot exceed O(1)


def main():
    print("=" * 78)
    print("  NVG: ETA_B INHERITANCE THROUGH CYCLES — CLOSING TEST")
    print("=" * 78)

    dilution = G_ENTROPY ** (N_CYCLES - 1)
    eta_1_required = ETA_OBS * dilution
    deficit_orders = math.log10(eta_1_required / ETA_MAX_PHYSICAL)

    print(f"\n  Comoving baryon number conserved; comoving entropy x{G_ENTROPY:.0f} per cycle")
    print(f"  (derived Tolman factor). Dilution over {N_CYCLES} cycles: 2^76 = {dilution:.2e}")
    print(f"\n  Observed n_B/s today:          {ETA_OBS:.1e}")
    print(f"  Required at cycle 1:           {eta_1_required:.1e}")
    print(f"  Physical ceiling:              {ETA_MAX_PHYSICAL:.0e}  (one baryon per entropy quantum)")
    print(f"  → inheritance EXCLUDED by ~{deficit_orders:.0f} orders of magnitude.")

    eta_1_x4 = ETA_OBS * 4.0 ** (N_CYCLES - 1)
    print(f"  (Superseded x4 law would require {eta_1_x4:.0e} — same conclusion, stronger.)")

    print(f"""
  VERDICT: eta_B cannot be an inherited initial condition. It must be
  regenerated each cycle, and since electroweak sphalerons are frozen at
  the QCD-scale bounce, NVG requires baryon-number violation beyond the
  Standard Model. The eta_B fork is closed; the remaining task is to
  derive (or abandon) the pi*sqrt(T_b/M_Pl) ansatz from such a mechanism.
""")
    print("=" * 78)

    assert eta_1_required > 1e12
    assert deficit_orders > 10


if __name__ == "__main__":
    main()
