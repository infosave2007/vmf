#!/usr/bin/env python3
"""PBH stochastic-background cross-check for the NANOGrav claim.

The PBH abundance needed for JWST seeding is taken from the maintained
two-population calculation and propagated to the PTA amplitude.  The result
is a deficit, so the historical PBH=NANOGrav match is retired; no static
amplitude or p-value is presented as a confirmation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

EVIDENCE_STATUS = "RETIRED_PBH_NANOGRAV_CHANNEL_AFTER_ABUNDANCE_CROSSCHECK"


def run_nanograv_verification() -> dict[str, Any]:
    import nvg_pbh_two_population as pbh

    f_yr = 1.0 / (365.25 * 24.0 * 3600.0)
    h0 = 67.4 * 1e5 / 3.086e24
    observed_strain = float(pbh.A_NANOGRAV)
    observed_omega = (2.0 * math.pi ** 2 / (3.0 * h0 ** 2)) * f_yr ** 2 * observed_strain ** 2
    rows = []
    for density in (pbh.N_SEED_LO, pbh.N_SEED_HI):
        ratio_h2 = (density / pbh.N_SMBH) * (pbh.M_B / pbh.M_SMBH) ** (5.0 / 3.0)
        amplitude = observed_strain * math.sqrt(ratio_h2)
        rows.append({"seed_density_mpc3": float(density), "strain": float(amplitude),
                     "deficit_factor": float(observed_strain / amplitude),
                     "cmb_bound_ok": bool((density * pbh.M_B / pbh.RHO_DM_MSUN_MPC3) < pbh.F_PBH_CMB_BOUND)})
    return {
        "frequency_hz": f_yr,
        "observed_strain": observed_strain,
        "observed_omega": float(observed_omega),
        "rows": rows,
        "status": EVIDENCE_STATUS,
        "solver": "nvg_pbh_two_population abundance cross-check",
        "missing": "independent PBH binary formation/merger-rate likelihood in the PTA band",
    }


def main() -> dict[str, Any]:
    result = run_nanograv_verification()
    print("=" * 78)
    print("  NVG PBH STOCHASTIC BACKGROUND — NANOGrav CROSS-CHECK")
    print("=" * 78)
    print(f"Observed NANOGrav input: A={result['observed_strain']:.3e}, Omega={result['observed_omega']:.3e}")
    for row in result["rows"]:
        print(f"JWST-calibrated density={row['seed_density_mpc3']:.0e}: "
              f"A_PBH={row['strain']:.3e}, deficit={row['deficit_factor']:.3e}, "
              f"CMB-bound-ok={row['cmb_bound_ok']}")
    print(f"Status: {result['status']}")
    print("NANOGrav is not attributed to this PBH channel; no match claim is made.")
    print("=" * 78)
    return result


if __name__ == "__main__":
    main()
