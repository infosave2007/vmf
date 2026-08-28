#!/usr/bin/env python3
"""Runtime-backed advanced observables (I).

This entry point used to print a hand-written neutron-star table, a tuned rho
mass and a synthetic cosmological ``proof``.  It now keeps the useful forward
calculations, but obtains the neutron-star numbers from the maintained TOV
producer and refuses to turn an unmeasured line shape or post-merger frequency
into evidence.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

EVIDENCE_STATUS = {
    "hades_dilepton": "FORWARD_ONLY_NO_HADES_LIKELIHOOD",
    "ns_redshift": "DERIVED_RUNTIME_CONDITIONAL_IN_SAMPLE",
    "postmerger_f_peak": "RETIRED_NO_POSTMERGER_SOLVER_OR_DATA",
    "tolman_cycles": "SENSITIVITY_ONLY_NO_ENTROPY_DYNAMICS_SOLVER",
}


def breit_wigner(mass: np.ndarray | float, pole: float, width: float) -> np.ndarray:
    """Relativistic Breit--Wigner template used only as a forward model."""

    m = np.asarray(mass, dtype=float)
    numerator = m * float(width)
    denominator = (m * m - float(pole) ** 2) ** 2 + (float(pole) * float(width)) ** 2
    return (2.0 * m / np.pi) * numerator / np.maximum(denominator, 1e-300)


def _hades_template_summary(pole: float, width: float = 350.0) -> dict[str, float]:
    """Compute a detector-style shape summary without fitting HADES data."""

    try:
        import nvg_hades_lineshape_feasibility as hades
    except ImportError:
        mass_grid = np.linspace(200.0, 800.0, 121)
        raw = breit_wigner(mass_grid, pole, width) * mass_grid ** 1.5 * np.exp(-mass_grid / 80.0)
        solver = "local Breit-Wigner thermal forward template"
    else:
        # The shared feasibility code supplies thermal and resolution folding;
        # replacing its pole with this run's raw model output avoids the old
        # post-hoc peak mapping.
        mass_grid = np.linspace(200.0, 800.0, 121)
        raw = hades.template(mass_grid, float(pole), 20.0)
        solver = "nvg_hades_lineshape_feasibility.template"
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    norm = float(integrate(raw, mass_grid))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("non-positive HADES forward-template integral")
    centroid = float(integrate(mass_grid * raw, mass_grid) / norm)
    return {"pole_mev": float(pole), "centroid_mev": centroid, "normalization": norm, "solver": solver}


def calculate_cycles(m_omega_mev: float) -> dict[str, float]:
    """Evaluate the documented scaling sensitivity; it is not a prediction."""

    if not np.isfinite(m_omega_mev) or m_omega_mev <= 0.0:
        raise ValueError("m_omega_mev must be positive and finite")
    s0 = 1.5e76 * (859.0 / float(m_omega_mev)) ** 8
    n_cycles = math.log(1.0e122 / s0) / math.log(4.0)
    n_e = 53.08 + 4.0 * math.log(float(m_omega_mev) / 859.0)
    return {"s0": s0, "n_cycles": n_cycles, "n_e": n_e}


def compute_observables() -> dict[str, Any]:
    """Compute all defensible rows and record unavailable mappings explicitly."""

    from nvg_fair_hades_link import HADRONS, in_medium_mass, n_0
    from nvg_joint_ns_inference import compute_nvg_predictions

    predictions = compute_nvg_predictions()[0]
    rho_vac, rho_cur = HADRONS["Rho (rho)"]
    rho_med = float(in_medium_mass(rho_vac, rho_cur, 2.0 * n_0))
    hades = _hades_template_summary(rho_med)
    mass = float(predictions["M_max"])
    radius = float(predictions["R_1.4"])
    compactness = mass * 1.4766 / radius
    z_surface = (1.0 - 2.0 * compactness) ** -0.5 - 1.0 if 0.0 < compactness < 0.5 else float("nan")

    return {
        "hades": {
            **hades,
            "status": EVIDENCE_STATUS["hades_dilepton"],
            "independent_data": False,
        },
        "ns_redshift": {
            "mass_msun": mass,
            "radius_km": radius,
            "z_surface": float(z_surface),
            "status": EVIDENCE_STATUS["ns_redshift"],
            "solver": "nvg_joint_ns_inference.compute_nvg_predictions",
        },
        "postmerger_f_peak": {
            "value_khz": None,
            "status": EVIDENCE_STATUS["postmerger_f_peak"],
            "missing": "relativistic post-merger simulation plus an independent detector likelihood",
        },
        "tolman_cycles": {
            "values": {str(m_omega): calculate_cycles(m_omega) for m_omega in (851.0, 859.0, 867.0)},
            "status": EVIDENCE_STATUS["tolman_cycles"],
            "missing": "entropy-production/turnaround dynamics solver and independent cycle observations",
        },
    }


def main() -> dict[str, Any]:
    state = compute_observables()
    print("=" * 72)
    print("  NVG: ADVANCED OBSERVABLES I (RUNTIME / EVIDENCE-STATUS LEDGER)")
    print("=" * 72)
    h = state["hades"]
    print(f"HADES forward rho template: pole={h['pole_mev']:.1f} MeV, centroid={h['centroid_mev']:.1f} MeV")
    print(f"  status={h['status']} (no observed HADES likelihood)")
    ns = state["ns_redshift"]
    print(f"Canonical NS: M_max={ns['mass_msun']:.3f} M_sun, R_1.4={ns['radius_km']:.3f} km, z={ns['z_surface']:.3f}")
    print(f"  status={ns['status']}")
    print(f"Post-merger f_peak: {state['postmerger_f_peak']['status']}")
    print(f"Tolman sensitivity: {state['tolman_cycles']['status']}")
    print("No independent observational confirmation is claimed by this entry point.")
    print("=" * 72)
    return state


if __name__ == "__main__":
    main()
