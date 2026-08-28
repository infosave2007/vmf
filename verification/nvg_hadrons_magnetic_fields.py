#!/usr/bin/env python3
"""Extended NVG calculations with explicit evidence boundaries.

The original "verification map" mixed dimensional estimates, calibration
inputs and unsupported confirmations.  This module now computes the same
useful quantities at runtime and labels each one according to the available
solver/data path.
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

hbar_c = 197.327
c_cgs = 2.998e10
G_cgs = 6.674e-8
M_Omega_0 = 859.0
M_N = 939.0
n_0 = 0.16
kappa_1 = 0.25
kappa_2 = 0.80
MeV_fm3_to_gcm3 = 1.7827e12

EVIDENCE_STATUS = {
    "magnetic_seed": "DIMENSIONAL_ESTIMATE_NO_MHD_SOLVER",
    "baryon_asymmetry": "RETIRED_NO_BARYON_NUMBER_OR_CP_VIOLATION_MECHANISM",
    "meson": "FORWARD_ONLY_NO_IN_MEDIUM_LIKELIHOOD",
    "entropy": "FORMULA_BOUND_NO_ENTROPY_TRANSFER_DYNAMICS",
    "summary": "RUNTIME_LEDGER_CONDITIONAL_OR_ZERO_EVIDENCE",
}


def M_Omega(n_B: float) -> float:
    """Density-dependent vacuum scale used by the declared VMF ansatz."""

    if not np.isfinite(n_B) or n_B < 0.0:
        raise ValueError("baryon density must be finite and non-negative")
    x = n_B / n_0
    return float(M_Omega_0 * (1.0 + kappa_2 * x) ** (-kappa_1 / kappa_2))


def magnetic_seed_estimate() -> dict[str, float | str]:
    """Compute a dimensional Biermann estimate; no field-evolution solve exists."""

    t_qcd = 1.0e-5  # declared epoch input
    t_qcd_mev = 155.0
    xi_fm = hbar_c / t_qcd_mev
    xi_cm = xi_fm * 1.0e-13
    d_horizon = c_cgs * t_qcd
    e_esu = 4.803e-10
    t_erg = t_qcd_mev * 1.602e-6
    dimensional_seed = t_erg ** 2 / (e_esu * c_cgs * xi_cm * 1.0e6)
    literature_seed = 1.0e-20  # declared literature order-of-magnitude input
    melt = 1.0 - M_Omega(2.0 * n_0) / M_Omega_0
    enhancement = 1.0 + 10.0 * melt
    b_epoch = literature_seed * enhancement
    b_today = b_epoch * (2.35e-10 / t_qcd_mev) ** 2
    return {
        "xi_fm": xi_fm,
        "horizon_cm": d_horizon,
        "dimensional_seed_gauss": dimensional_seed,
        "literature_seed_gauss": literature_seed,
        "melt_fraction_2n0": melt,
        "enhancement": enhancement,
        "epoch_seed_gauss": b_epoch,
        "today_gauss": b_today,
        "status": EVIDENCE_STATUS["magnetic_seed"],
    }


def baryon_asymmetry_audit(n_cycles: int = 76, growth: float = 1.35) -> dict[str, float | str]:
    """Quantify dilution of a relic asymmetry under the stated toy scaling."""

    if n_cycles < 0 or not np.isfinite(growth) or growth <= 0.0:
        raise ValueError("cycle count and growth factor must be valid")
    eta_observed = 6.1e-10  # Planck+BBN observation input
    eta_initial = eta_observed * float(growth) ** int(n_cycles)
    return {
        "eta_observed": eta_observed,
        "eta_initial_required": eta_initial,
        "cycles": float(n_cycles),
        "growth": float(growth),
        "status": EVIDENCE_STATUS["baryon_asymmetry"],
    }


def meson_forward_table() -> list[dict[str, float | str]]:
    """Reuse the maintained FAIR/HADES mass mapping for a forward table."""

    from nvg_fair_hades_link import HADRONS, in_medium_mass
    rows = []
    for name, (vacuum, current) in HADRONS.items():
        medium = float(in_medium_mass(vacuum, current, 2.0 * n_0))
        rows.append({"name": name, "vacuum_mev": float(vacuum), "medium_mev": medium,
                     "shift_fraction": 1.0 - medium / float(vacuum)})
    return rows


def entropy_bound_audit() -> dict[str, float | bool | str]:
    """Compare declared entropy inputs with the geometric bound."""

    radiation = 2.6e88
    black_holes = 1.0e104
    radius_cm = 1.13e5
    planck_cm = 1.616e-33
    bound = 4.0 * math.pi * radius_cm ** 2 / (4.0 * planck_cm ** 2)
    return {
        "radiation_entropy": radiation,
        "black_hole_entropy": black_holes,
        "bounce_bound": bound,
        "radiation_fits": bool(bound > radiation),
        "black_hole_entropy_fits": bool(bound > black_holes),
        "status": EVIDENCE_STATUS["entropy"],
    }


def compute_summary() -> dict[str, Any]:
    from nvg_joint_ns_inference import compute_nvg_predictions

    predictions, metadata = compute_nvg_predictions()
    return {
        "canonical_ns": predictions,
        "canonical_selection": metadata["canonical_selection"],
        "magnetic_seed": magnetic_seed_estimate(),
        "baryon_asymmetry": baryon_asymmetry_audit(),
        "mesons": {"rows": meson_forward_table(), "status": EVIDENCE_STATUS["meson"]},
        "entropy": entropy_bound_audit(),
        "status": EVIDENCE_STATUS["summary"],
    }


def main() -> dict[str, Any]:
    state = compute_summary()
    print("=" * 72)
    print("  NVG: EXTENDED PHYSICS TESTS (RUNTIME EVIDENCE LEDGER)")
    print("=" * 72)
    ns = state["canonical_ns"]
    print(f"Canonical NS: M_max={ns['M_max']:.3f} M_sun, R_1.4={ns['R_1.4']:.3f} km, Lambda_1.4={ns['Lambda_1.4']:.1f}")
    seed = state["magnetic_seed"]
    print(f"Magnetic seed estimate: B_epoch={seed['epoch_seed_gauss']:.3e} G, B_today={seed['today_gauss']:.3e} G; status={seed['status']}")
    eta = state["baryon_asymmetry"]
    print(f"Baryon asymmetry audit: eta_initial_required={eta['eta_initial_required']:.3e}; status={eta['status']}")
    print(f"Meson forward rows: {len(state['mesons']['rows'])}; status={state['mesons']['status']}")
    ent = state["entropy"]
    print(f"Entropy geometric bound: radiation_fits={ent['radiation_fits']}, black_hole_entropy_fits={ent['black_hole_entropy_fits']}; status={ent['status']}")
    print("No row in this map is an independent confirmation merely because it was computed.")
    print("=" * 72)
    return state


if __name__ == "__main__":
    main()
