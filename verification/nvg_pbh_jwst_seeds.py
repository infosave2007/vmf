#!/usr/bin/env python3
"""Conditional early-SMBH seed growth calculation.

JWST target masses/redshifts are declared literature inputs.  Growth is
recomputed with the explicit constant-duty Eddington model and the primordial
seed mass comes from the canonical PBH ladder.  No seed-occupation or survey
selection likelihood is present, so this remains a conditional forward model.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nvg_pbh_mass_spectrum import get_pbh_mass

H_0_km_s_Mpc = 67.4
Omega_m = 0.315
Omega_L = 0.685
km_s_Mpc_to_Gyr = 1.022689e-3
H_0_Gyr = H_0_km_s_Mpc * km_s_Mpc_to_Gyr

tau_Salpeter_Gyr = 0.045
eta = 0.1
f_Edd = 0.10
z_start = 20.0
M_seed_popIII = 100.0
M_seed_nvg = get_pbh_mass(10)

JWST_TARGETS = [
    {"name": "GN-z11", "redshift": 10.60, "observed_mass": 1.6e6, "ref": "Maiolino et al. (2023)"},
    {"name": "UHZ1", "redshift": 10.10, "observed_mass": 4.0e7, "ref": "Goulding et al. (2023)"},
    {"name": "J2236+0032", "redshift": 6.30, "observed_mass": 1.4e9, "ref": "Onoue et al. (2023)"},
]


def cosmic_age_Gyr(z: float) -> float:
    if z < -1.0:
        raise ValueError("redshift must be >= -1")
    factor = 2.0 / (3.0 * H_0_Gyr * math.sqrt(Omega_L))
    x = Omega_L / (Omega_m * (1.0 + z) ** 3)
    return factor * math.log(math.sqrt(x) + math.sqrt(1.0 + x))


def grow_black_hole(M_start: float, t_start: float, t_end: float) -> float:
    if M_start < 0.0:
        raise ValueError("seed mass must be non-negative")
    dt = t_end - t_start
    if dt <= 0.0:
        return M_start
    growth_rate = f_Edd * (1.0 - eta) / (eta * tau_Salpeter_Gyr)
    return M_start * math.exp(growth_rate * dt)


def compute_seed_growth() -> dict[str, Any]:
    t_start = cosmic_age_Gyr(z_start)
    rows = []
    for target in JWST_TARGETS:
        t_end = cosmic_age_Gyr(target["redshift"])
        pop_mass = grow_black_hole(M_seed_popIII, t_start, t_end)
        nvg_mass = grow_black_hole(M_seed_nvg, t_start, t_end)
        rows.append(
            {
                **target,
                "age_Myr": t_end * 1000.0,
                "popIII_final_mass": pop_mass,
                "nvg_final_mass": nvg_mass,
                "nvg_to_observed": nvg_mass / target["observed_mass"],
            }
        )
    return {
        "start_age_Myr": t_start * 1000.0,
        "rows": rows,
        "evidence_status": "CONDITIONAL_FORWARD_MODEL",
        "observed_likelihood": None,
        "canonical_producer": "verification/nvg_pbh_mass_spectrum.py:get_pbh_mass",
        "limitation": "No seed occupation, duty-cycle distribution, or JWST selection likelihood is present.",
    }


def main() -> dict[str, Any]:
    state = compute_seed_growth()
    print("=" * 80)
    print("      NVG PBH-LADDER VS JWST SEED-GROWTH FORWARD MODEL")
    print("=" * 80)
    print(f"Start redshift / age                      : z={z_start:.1f}, {state['start_age_Myr']:.1f} Myr")
    print(f"Pop-III seed mass                          : {M_seed_popIII:.1f} M_sun (declared scenario)")
    print(f"Canonical ladder seed (N=10)              : {M_seed_nvg:.1f} M_sun")
    print(f"Average Eddington ratio                   : {f_Edd * 100.0:.1f}% (declared scenario)")
    print(f"{'Target':<14} | {'z':<7} | {'Age Myr':<10} | {'PopIII final':<15} | {'Ladder final':<15} | {'Declared obs':<15}")
    print("-" * 100)
    for row in state["rows"]:
        print(
            f"{row['name']:<14} | {row['redshift']:<7.2f} | {row['age_Myr']:<10.1f} | "
            f"{row['popIII_final_mass']:<15.2e} | {row['nvg_final_mass']:<15.2e} | {row['observed_mass']:<15.2e}"
        )
    print("-" * 100)
    print("Evidence status: CONDITIONAL_FORWARD_MODEL")
    print("Target comparisons are descriptive; no seed-abundance or survey likelihood is evaluated.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
