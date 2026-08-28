#!/usr/bin/env python3
"""Advanced calculations with runtime provenance and honest boundaries.

The previous suite generated mock JWST/FRB catalogs, injected GW noise and
aggregated their p-values as if they were observations.  The retained rows
either reuse a canonical calculation or return a precise retirement reason.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

M_Omega_0 = 859.0  # declared lattice/QCD input (MeV)
hbar_c = 197.32698  # MeV fm

EVIDENCE_STATUS = {
    "jwst": "RETIRED_NO_MACHINE_READABLE_JWST_CATALOG_OR_LIKELIHOOD",
    "frb": "RETIRED_NO_REPEATER_DM_MASS_CATALOG_OR_LIKELIHOOD",
    "mass_ratio": "FORMAL_GOR_CALCULATION_NO_MASS_PREDICTION",
    "qcd_phase": "FORWARD_PARAMETERIZATION_NO_PHASE_DIAGRAM_LIKELIHOOD",
    "pta": "RETIRED_PBH_NANOGRAV_CLAIM_AFTER_ABUNDANCE_CROSSCHECK",
    "bounce_temperature": "DERIVED_FROM_DECLARED_ANCHOR_AND_GSTAR",
    "neutrino": "RETIRED_NO_NEUTRINO_MASS_SOLVER_OR_INDEPENDENT_LIKELIHOOD",
}


def cosmic_age_Gyr(z: float, H_0: float = 67.4, Omega_m: float = 0.315,
                   Omega_L: float = 0.685) -> float:
    """Flat-Lambda-CDM age used only by forward growth calculations."""

    if not np.isfinite(z) or z < -1.0:
        raise ValueError("redshift must be finite and >= -1")
    H_0_Gyr = float(H_0) * 1.022689e-3
    factor = 2.0 / (3.0 * H_0_Gyr * math.sqrt(float(Omega_L)))
    x = float(Omega_L) / (float(Omega_m) * (1.0 + float(z)) ** 3)
    return float(factor * math.log(math.sqrt(x) + math.sqrt(1.0 + x)))


def required_f_edd(M_seed: float, M_obs: float, t_start: float, t_end: float,
                   eta: float = 0.1, tau_Salpeter_Gyr: float = 0.045) -> float:
    """Compute the Eddington ratio required by a declared mass pair."""

    if min(M_seed, M_obs, eta, tau_Salpeter_Gyr) <= 0.0:
        raise ValueError("masses, efficiency and timescale must be positive")
    dt = float(t_end) - float(t_start)
    if dt <= 0.0:
        return 0.0
    return float(math.log(float(M_obs) / float(M_seed)) * float(eta) * float(tau_Salpeter_Gyr)
                 / ((1.0 - float(eta)) * dt))


def run_jwst_smbh_check() -> dict[str, Any]:
    """Reuse the two-population abundance cross-check, without mock objects."""

    import nvg_pbh_two_population as pbh

    rows = []
    for density in (pbh.N_SEED_LO, pbh.N_SEED_HI):
        rho = float(density) * pbh.M_B
        fraction = rho / pbh.RHO_DM_MSUN_MPC3
        rows.append({"seed_density_mpc3": float(density), "f_pbh": fraction,
                     "cmb_bound_ok": bool(fraction < pbh.F_PBH_CMB_BOUND)})
    return {
        "rows": rows,
        "status": EVIDENCE_STATUS["jwst"],
        "solver": "nvg_pbh_two_population abundance cross-check",
        "missing": "machine-readable JWST mass/redshift catalog and an object-level selection likelihood",
    }


def run_frb_dm_check() -> dict[str, Any]:
    return {
        "mean_repeater_dm": None,
        "mean_single_dm": None,
        "p_value": None,
        "status": EVIDENCE_STATUS["frb"],
        "missing": "CHIME repeater labels joined to host/redshift/DM and a selection-corrected mass likelihood",
    }


def run_mass_derivation_check() -> dict[str, Any]:
    """Evaluate the GOR relation as a formal dimensional calculation."""

    m_q = 3.45
    f_pi = 92.2
    condensate = 272.0 ** 3
    m_pi = math.sqrt(2.0 * m_q * condensate / f_pi ** 2)
    m_nucleon = M_Omega_0 + 3.0 * m_q
    return {
        "m_nucleon_mev": float(m_nucleon),
        "m_pion_mev": float(m_pi),
        "ratio_to_physical_pion": float(m_nucleon / 139.57),
        "status": EVIDENCE_STATUS["mass_ratio"],
    }


def run_qcd_phase_diagram() -> dict[str, Any]:
    """Build the declared crossover/melting parameterization in memory."""

    mu = np.linspace(0.0, 1200.0, 200)
    t_lattice = 165.0 * (1.0 - 0.013 * (mu / 165.0) ** 2)
    t_melt = 432.2 * np.maximum(1.0 - (mu / 1200.0) ** 4, 0.0) ** 0.25
    return {
        "mu_b_mev": mu,
        "lattice_crossover_mev": t_lattice,
        "vmf_melting_mev": t_melt,
        "status": EVIDENCE_STATUS["qcd_phase"],
        "missing": "finite-density lattice data or a likelihood selecting the VMF boundary",
    }


def run_pta_ligo_cross_correlation() -> dict[str, Any]:
    """Compute the PBH abundance cross-check that retires the PTA claim."""

    import nvg_pbh_two_population as pbh

    n_seed = float(pbh.N_SEED_HI)
    ratio_h2 = (n_seed / pbh.N_SMBH) * (pbh.M_B / pbh.M_SMBH) ** (5.0 / 3.0)
    amplitude = pbh.A_NANOGRAV * math.sqrt(ratio_h2)
    return {
        "pbh_amplitude": float(amplitude),
        "observed_amplitude": float(pbh.A_NANOGRAV),
        "deficit_factor": float(pbh.A_NANOGRAV / amplitude),
        "status": EVIDENCE_STATUS["pta"],
        "solver": "nvg_pbh_two_population abundance cross-check",
    }


def run_bounce_temperature_check() -> dict[str, Any]:
    g_star = 47.5
    eps_max = M_Omega_0 ** 4 / hbar_c ** 3
    temperature = (eps_max / (g_star * math.pi ** 2 / 30.0 / hbar_c ** 3)) ** 0.25
    return {"temperature_mev": float(temperature), "g_star": g_star,
            "status": EVIDENCE_STATUS["bounce_temperature"]}


def run_neutrino_mass_check() -> dict[str, Any]:
    return {
        "mass_ev": None,
        "katrin_limit_ev": 0.45,  # declared external limit input
        "status": EVIDENCE_STATUS["neutrino"],
        "missing": "a neutrino-sector mass-generation calculation and independent likelihood",
    }


def compute_suite() -> dict[str, Any]:
    return {
        "jwst": run_jwst_smbh_check(),
        "frb": run_frb_dm_check(),
        "mass_ratio": run_mass_derivation_check(),
        "qcd_phase": run_qcd_phase_diagram(),
        "pta": run_pta_ligo_cross_correlation(),
        "bounce_temperature": run_bounce_temperature_check(),
        "neutrino": run_neutrino_mass_check(),
    }


def main() -> int:
    state = compute_suite()
    print("=" * 80)
    print("  NVG ADVANCED CALCULATIONS (RUNTIME EVIDENCE LEDGER)")
    print("=" * 80)
    print(f"JWST abundance rows: {len(state['jwst']['rows'])}; status={state['jwst']['status']}")
    print(f"FRB DM statistic: p={state['frb']['p_value']}; status={state['frb']['status']}")
    print(f"GOR formal ratio: {state['mass_ratio']['ratio_to_physical_pion']:.3f}; status={state['mass_ratio']['status']}")
    print(f"QCD phase grid: {len(state['qcd_phase']['mu_b_mev'])} points; status={state['qcd_phase']['status']}")
    print(f"PTA PBH amplitude deficit: {state['pta']['deficit_factor']:.3e}; status={state['pta']['status']}")
    print(f"Bounce temperature: {state['bounce_temperature']['temperature_mev']:.2f} MeV; status={state['bounce_temperature']['status']}")
    print(f"Neutrino mass: status={state['neutrino']['status']}")
    print("A process run is not an observational verification of these rows.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    main()
