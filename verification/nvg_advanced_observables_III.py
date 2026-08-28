#!/usr/bin/env python3
"""Runtime forward calculations for the third advanced-observables set.

Only algebraic/formal checks and model templates are available for these
rows.  No detector likelihood is fabricated, and the old population-level
cooling conclusion is explicitly retired because no independent mass/age/
luminosity sample is present in the repository.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

EVIDENCE_STATUS = {
    "lorentz": "FORMAL_CONFORMAL_IDENTITY_NO_GRB_LIKELIHOOD",
    "meson": "FORWARD_ONLY_NO_IN_MEDIUM_DATA_LIKELIHOOD",
    "qnm": "SENSITIVITY_ONLY_NO_RINGDOWN_SOLVER_OR_DATA",
    "cooling": "RETIRED_NO_INDEPENDENT_COOLING_POPULATION_OR_LIKELIHOOD",
}


def lorentz_limit(energies_ev: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Evaluate the conformal-sector prediction in a homogeneous vacuum."""

    energies = np.asarray(energies_ev if energies_ev is not None else [1e6, 1e9, 1e12], dtype=float)
    if np.any(~np.isfinite(energies)) or np.any(energies <= 0.0):
        raise ValueError("photon energies must be finite and positive")
    return {"energies_ev": energies, "birefringence": np.zeros_like(energies), "dispersion": np.zeros_like(energies)}


def meson_forward_table(n_ratio: float = 2.0) -> list[dict[str, float | str]]:
    """Use the maintained FAIR/HADES mass-mapping producer for templates."""

    if not np.isfinite(n_ratio) or n_ratio < 0.0:
        raise ValueError("density ratio must be finite and non-negative")
    from nvg_fair_hades_link import HADRONS, in_medium_mass, n_0

    rows = []
    for name, (vacuum, current) in HADRONS.items():
        medium = float(in_medium_mass(vacuum, current, float(n_ratio) * n_0))
        rows.append({"name": name, "vacuum_mev": float(vacuum), "medium_mev": medium,
                     "shift_fraction": 1.0 - medium / float(vacuum)})
    return rows


def qnm_sensitivity(l_over_rs: float = 1.0e-35) -> dict[str, float]:
    """Return the leading core-size scaling, explicitly not a ringdown fit."""

    if not np.isfinite(l_over_rs) or l_over_rs < 0.0:
        raise ValueError("l_over_rs must be finite and non-negative")
    return {"l_over_rs": float(l_over_rs), "fractional_shift": float(l_over_rs) ** 3}


def compute_observables() -> dict[str, Any]:
    lorentz = lorentz_limit()
    mesons = meson_forward_table()
    return {
        "lorentz": {**lorentz, "status": EVIDENCE_STATUS["lorentz"],
                    "missing": "independent GRB polarization/time-of-flight likelihood"},
        "meson": {"rows": mesons, "status": EVIDENCE_STATUS["meson"],
                  "missing": "acceptance-corrected CBM/HADES line-shape data and likelihood"},
        "qnm": {**qnm_sensitivity(), "status": EVIDENCE_STATUS["qnm"],
                "missing": "perturbation solver on the specified regular metric and independent ringdown data"},
        "cooling": {"rows": [], "status": EVIDENCE_STATUS["cooling"],
                    "missing": "mass-, age-, envelope- and luminosity-linked pulsar catalog plus cooling likelihood"},
    }


def main() -> dict[str, Any]:
    state = compute_observables()
    print("=" * 72)
    print("  NVG: ADVANCED OBSERVABLES III (FORWARD / EVIDENCE-STATUS LEDGER)")
    print("=" * 72)
    l = state["lorentz"]
    print(f"Conformal W-sector: {len(l['energies_ev'])} energies, birefringence=0, dispersion=0")
    print(f"  status={l['status']}")
    print(f"Meson templates at 2 n0: {len(state['meson']['rows'])} runtime rows; status={state['meson']['status']}")
    print(f"QNM core-size sensitivity: delta={state['qnm']['fractional_shift']:.3e}; status={state['qnm']['status']}")
    print(f"Cooling population: {state['cooling']['status']}")
    print("No independent GRB, CBM/HADES, ringdown or cooling-population confirmation is claimed.")
    print("=" * 72)
    return state


if __name__ == "__main__":
    main()
