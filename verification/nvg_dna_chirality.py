#!/usr/bin/env python3
"""DNA-chirality scale calculations with an explicit evidence boundary.

Temperature conversions, coherence lengths and thermalization times are
recomputed from declared constants.  The prior PVED value, reaction count and
CISS amplification did not have a source derivation or biological likelihood;
those quantities are therefore optional sensitivity inputs and the
homochirality claim is retired.
"""

from __future__ import annotations

import math
from typing import Any


hbar = 1.0545718e-34
hbar_ev = 6.5821195e-16
k_B_J = 1.380649e-23
k_B_eV = 8.617333e-5
c = 2.99792458e8
fm_to_m = 1e-15
hbar_c_mev_fm = 197.327
M_Omega_0 = 859.0
T_c_MeV = 157.3
xi_theta_0 = 1.254


def ev_to_kelvin(ev_val: float) -> float:
    return ev_val / k_B_eV


def kelvin_to_ev(k_val: float) -> float:
    return k_val * k_B_eV


def compute_scales(
    *,
    room_temperature_K: float = 300.0,
    physiological_temperature_K: float = 310.15,
    pved_eV: float | None = None,
    reactions: float | None = None,
) -> dict[str, Any]:
    """Compute anchored scales and optional, clearly labelled PVED sensitivity."""

    if room_temperature_K <= 0.0 or physiological_temperature_K <= 0.0:
        raise ValueError("temperatures must be positive")
    t_c_K = ev_to_kelvin(T_c_MeV * 1e6)
    derived_xi = hbar_c_mev_fm / T_c_MeV
    tau_room_fs = hbar / (k_B_J * room_temperature_K) * 1e15
    tau_phys_fs = hbar / (k_B_J * physiological_temperature_K) * 1e15
    xi_room_um = xi_theta_0 * (t_c_K / room_temperature_K) * 1e-9
    xi_phys_um = xi_theta_0 * (t_c_K / physiological_temperature_K) * 1e-9
    membrane_nm = 5.0
    membrane_fm = membrane_nm * 1e6
    theta_0_bio = 2.73e-9
    rho_mev_fm3 = (M_Omega_0**2 * theta_0_bio**2) / (hbar_c_mev_fm * membrane_fm**2)
    rho_j_m3 = rho_mev_fm3 * 1.60217663e32

    pved_sensitivity = None
    if pved_eV is not None:
        if pved_eV < 0.0:
            raise ValueError("pved_eV must be non-negative")
        advantage = pved_eV / kelvin_to_ev(room_temperature_K)
        pved_sensitivity = {
            "pved_eV": float(pved_eV),
            "advantage_per_interaction": float(advantage),
            "reactions": None if reactions is None else float(reactions),
            "enantiomeric_excess": (
                None if reactions is None else float(1.0 - math.exp(-advantage * reactions))
            ),
        }
    return {
        "T_c_K": float(t_c_K),
        "derived_xi_0_fm": float(derived_xi),
        "xi_room_um": float(xi_room_um),
        "xi_phys_um": float(xi_phys_um),
        "tau_room_fs": float(tau_room_fs),
        "tau_phys_fs": float(tau_phys_fs),
        "vacuum_battery_density_J_m3": float(rho_j_m3),
        "pved_sensitivity": pved_sensitivity,
        "evidence_status": "RETIRED_MISSING_PVED_SOURCE",
        "observed_likelihood": None,
        "missing_components": [
            "derived QCD-to-molecular PVED coupling",
            "CISS/chemical-reaction kinetics and environmental history",
            "independent chirality outcome likelihood",
        ],
    }


def main() -> dict[str, Any]:
    state = compute_scales()
    print("=" * 80)
    print(" NVG/VMF BIOLOGICAL SCALE CALCULATION")
    print("=" * 80)
    print(f"QCD transition temperature                 : {T_c_MeV:.1f} MeV = {state['T_c_K']:.3e} K")
    print(f"Coherence length at T_c                    : {xi_theta_0:.3f} fm")
    print(f"Runtime hbar*c/T_c                         : {state['derived_xi_0_fm']:.4f} fm")
    print(f"Room-temperature phase time                : {state['tau_room_fs']:.2f} fs")
    print(f"Physiological phase time                   : {state['tau_phys_fs']:.2f} fs")
    print(f"Room-temperature coherence scale           : {state['xi_room_um']:.3f} microns")
    print(f"Physiological coherence scale              : {state['xi_phys_um']:.3f} microns")
    print(f"Illustrative membrane energy density       : {state['vacuum_battery_density_J_m3']:.3e} J/m^3")
    print("Evidence status                            : RETIRED_MISSING_PVED_SOURCE")
    print("PVED and homochirality are not inferred: the molecular coupling and likelihood are absent.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
