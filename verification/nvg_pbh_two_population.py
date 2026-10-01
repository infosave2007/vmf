#!/usr/bin/env python3
"""Conditional two-population PBH bookkeeping, not a cosmological abundance model.

Population A is the normalized *shape* produced by ``nvg_pbh_dark_matter``.
Population B is a legacy JWST-seed calibration: a stipulated seed number
range is divided by a stipulated present-day dark-matter density merely to
show its tiny conditional fraction.  Neither row contains a PBH formation
solver, a shared cosmology, an absolute physical ``omega_PBH`` prediction, or
an observational likelihood.  In particular, this module must not be used to
close the NVG dark-matter budget.

The former text called a 4e5-solar-mass seed ``N~20`` under a separate 2^N
convention.  The maintained canonical ladder is 0.38*4^N; its nearest runtime
rung is now derived below rather than hard-coded.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from nvg_pbh_mass_spectrum import get_pbh_mass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CALIBRATION_PATH = HERE / "data" / "pbh_population_b_seed_calibration.json"
CALIBRATION_SCHEMA_VERSION = "pbh-population-b-seed-calibration.v1"
CALIBRATION_STATUS = "LEGACY_UNDOCUMENTED_SEED_TRACE_NOT_ABSOLUTE_COSMOLOGY"
STATUS = "CALIBRATED_SEED_TRACE_NOT_ABSOLUTE_COSMOLOGY"


def _load_seed_calibration(path: Path = CALIBRATION_PATH) -> dict[str, Any]:
    """Load the historical seed trace without pretending it has provenance."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unrecognized PBH-B seed calibration schema")
    if payload.get("calibration_status") != CALIBRATION_STATUS:
        raise ValueError("PBH-B calibration status changed; review required")
    provenance = payload.get("provenance")
    params = payload.get("parameters")
    if not isinstance(provenance, dict) or not isinstance(params, dict):
        raise ValueError("PBH-B seed calibration passport is incomplete")
    if provenance.get("citation") is not None or provenance.get("source_url") is not None:
        raise ValueError("PBH-B seed calibration must not invent source provenance")
    if not isinstance(provenance.get("limitation"), str) or not provenance["limitation"]:
        raise ValueError("PBH-B seed calibration misses its limitation")
    expected_units = {
        "legacy_rho_dm_Msun_Mpc3": "M_sun Mpc^-3",
        "seed_mass_Msun": "M_sun",
        "seed_density_low_Mpc_minus3": "Mpc^-3",
        "seed_density_high_Mpc_minus3": "Mpc^-3",
        "legacy_nanograv_strain_at_1_per_year": "dimensionless",
        "legacy_smbh_chirp_mass_Msun": "M_sun",
        "legacy_smbh_merger_density_Mpc_minus3": "Mpc^-3",
        "legacy_cmb_fraction_benchmark": "dimensionless",
    }
    for key, units in expected_units.items():
        entry = params.get(key)
        if not isinstance(entry, dict) or entry.get("units") != units or not isinstance(entry.get("definition"), str):
            raise ValueError(f"PBH-B seed calibration misses passport for {key}")
        try:
            value = float(entry.get("value"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PBH-B seed calibration has invalid {key}") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"PBH-B seed calibration has nonpositive {key}")
    if float(params["seed_density_high_Mpc_minus3"]["value"]) <= float(params["seed_density_low_Mpc_minus3"]["value"]):
        raise ValueError("PBH-B seed density range is reversed")
    return payload


_CALIBRATION = _load_seed_calibration()
_PARAMETERS = _CALIBRATION["parameters"]
# Compatibility aliases for existing consumers; each is loaded from the
# explicit legacy passport rather than held as a hidden script constant.
RHO_DM_MSUN_MPC3 = float(_PARAMETERS["legacy_rho_dm_Msun_Mpc3"]["value"])
M_B = float(_PARAMETERS["seed_mass_Msun"]["value"])
N_SEED_LO = float(_PARAMETERS["seed_density_low_Mpc_minus3"]["value"])
N_SEED_HI = float(_PARAMETERS["seed_density_high_Mpc_minus3"]["value"])
A_NANOGRAV = float(_PARAMETERS["legacy_nanograv_strain_at_1_per_year"]["value"])
M_SMBH = float(_PARAMETERS["legacy_smbh_chirp_mass_Msun"]["value"])
N_SMBH = float(_PARAMETERS["legacy_smbh_merger_density_Mpc_minus3"]["value"])
F_PBH_CMB_BOUND = float(_PARAMETERS["legacy_cmb_fraction_benchmark"]["value"])


def _nearest_canonical_cycle(mass_msun: float) -> tuple[int, float, float]:
    """Return the nearest declared 0.38*4^N rung without changing the ladder."""
    if not math.isfinite(mass_msun) or mass_msun <= 0.0:
        raise ValueError("seed mass must be finite and positive")
    cycles = range(-30, 21)
    cycle = min(cycles, key=lambda candidate: abs(math.log(get_pbh_mass(candidate) / mass_msun)))
    ladder_mass = float(get_pbh_mass(cycle))
    return cycle, ladder_mass, ladder_mass / mass_msun - 1.0


def compute_population_b_calibrated_trace() -> dict[str, Any]:
    """Expose the B calibration while explicitly withholding physical abundance.

    ``conditional_fraction_relative_to_legacy_rho_dm`` is arithmetic using a
    declared historical density benchmark.  It is not ``f_PBH`` in a common
    NVG cosmology and never becomes a physical abundance component.
    """
    cycle, ladder_mass, ladder_relative_difference = _nearest_canonical_cycle(M_B)
    rows: list[dict[str, float | bool]] = []
    for seed_density in (N_SEED_LO, N_SEED_HI):
        mass_density = seed_density * M_B
        conditional_fraction = mass_density / RHO_DM_MSUN_MPC3
        ratio_h2 = (seed_density / N_SMBH) * (M_B / M_SMBH) ** (5.0 / 3.0)
        strain = A_NANOGRAV * math.sqrt(ratio_h2)
        rows.append(
            {
                "seed_density_Mpc_minus3": seed_density,
                "conditional_mass_density_Msun_Mpc_minus3": mass_density,
                "conditional_fraction_relative_to_legacy_rho_dm": conditional_fraction,
                "conditional_fraction_below_legacy_cmb_benchmark": conditional_fraction < F_PBH_CMB_BOUND,
                "conditional_strain_at_1_per_year": strain,
                "conditional_strain_deficit_vs_legacy_nanograv_benchmark": A_NANOGRAV / strain,
            }
        )
    return {
        "status": STATUS,
        "calibration_status": CALIBRATION_STATUS,
        "calibration_input_artifact": CALIBRATION_PATH.relative_to(ROOT).as_posix(),
        "calibration_input_sha256": hashlib.sha256(CALIBRATION_PATH.read_bytes()).hexdigest(),
        "physical_omega_h2": None,
        "physical_fraction_of_dark_matter": None,
        "observed_likelihood": None,
        "seed_mass_Msun": M_B,
        "canonical_ladder_nearest_cycle": cycle,
        "canonical_ladder_mass_Msun": ladder_mass,
        "canonical_ladder_relative_mass_difference": ladder_relative_difference,
        "calibration_inputs": {
            "legacy_rho_dm_Msun_Mpc3": RHO_DM_MSUN_MPC3,
            "seed_density_range_Mpc_minus3": [N_SEED_LO, N_SEED_HI],
            "legacy_nanograv_strain_at_1_per_year": A_NANOGRAV,
            "legacy_cmb_fraction_benchmark": F_PBH_CMB_BOUND,
        },
        "rows": rows,
        "limitation": (
            "Seed counts, reference density, CMB benchmark and strain reference are calibration inputs. "
            "No PBH formation solver, shared background cosmology, absolute omega_PBH prediction, "
            "JWST occupation likelihood or merger likelihood is supplied."
        ),
    }


def main() -> dict[str, Any]:
    state = compute_population_b_calibrated_trace()
    print("=" * 78)
    print("  NVG: CONDITIONAL PBH SEED TRACE — NOT A DM ABUNDANCE")
    print("=" * 78)
    print("Population A is a normalized profile only; it has no formation abundance.")
    print(
        "Population B uses calibrated seed/reference-density inputs; it is not "
        "a component of a common physical DM budget."
    )
    print(
        f"Seed mass {state['seed_mass_Msun']:.2e} M_sun; nearest canonical "
        f"0.38*4^N rung: N={state['canonical_ladder_nearest_cycle']} "
        f"({state['canonical_ladder_mass_Msun']:.2e} M_sun, "
        f"relative difference {state['canonical_ladder_relative_mass_difference']:+.3e})."
    )
    print(f"{'n_seed [Mpc^-3]':>16} {'conditional f':>16} {'A_B (conditional)':>20}")
    for row in state["rows"]:
        print(
            f"{row['seed_density_Mpc_minus3']:>16.0e} "
            f"{row['conditional_fraction_relative_to_legacy_rho_dm']:>16.2e} "
            f"{row['conditional_strain_at_1_per_year']:>20.1e}"
        )
    print("Evidence status:", state["status"])
    print("=" * 78)
    return state


if __name__ == "__main__":
    main()
