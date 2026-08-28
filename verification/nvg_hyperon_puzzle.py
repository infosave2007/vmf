#!/usr/bin/env python3
"""Hyperon-threshold audit using runtime equations and raw TOV baselines.

The earlier unsupported hyperon conclusion came from approximate thresholds
and a static target mass.  This entry point now exposes the calculated
threshold proxy and reuses :mod:`nvg_hyperon_puzzle_tov` for raw baseline
curves.  It does not claim a complete hyperonic beta-equilibrium likelihood.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

n_0 = 0.16
M_N_vac = 939.0
M_Omega_N = 859.0
M_cur_N = 80.0
M_Lambda_vac = 1115.7
M_cur_Lambda = 200.0
M_Omega_Lambda = M_Lambda_vac - M_cur_Lambda
C_v_n0 = 100.0
alpha_v = 4.0
nu_v = 2.0
kappa_1 = 0.25
kappa_2 = 0.80

EVIDENCE_STATUS = "DERIVED_THRESHOLD_PROXY_NO_COMPLETE_HYPERON_LIKELIHOOD"


def vacuum_melt_factor(n_B: float) -> float:
    if not np.isfinite(n_B) or n_B < 0.0:
        raise ValueError("density must be finite and non-negative")
    x = n_B / n_0
    return float((1.0 + kappa_2 * x) ** (-kappa_1 / kappa_2))


def vector_potential(n_B: float) -> float:
    if not np.isfinite(n_B) or n_B < 0.0:
        raise ValueError("density must be finite and non-negative")
    x = n_B / n_0
    return float(C_v_n0 * x ** nu_v / (1.0 + x ** nu_v / alpha_v)) if x else 0.0


def nvg_core_thermodynamics(n_B: float) -> float:
    x = n_B / n_0
    melt = vacuum_melt_factor(n_B)
    m_star = M_cur_N + M_Omega_N * melt
    k_f = (1.5 * np.pi ** 2 * n_B) ** (1.0 / 3.0)
    e_f = np.sqrt(k_f ** 2 + m_star ** 2)
    return float(e_f + vector_potential(n_B))


def lambda_effective_mass(n_B: float) -> float:
    return float(M_cur_Lambda + M_Omega_Lambda * vacuum_melt_factor(n_B))


def compute_thresholds(n_min: float = 0.5, n_max: float = 3.5, points: int = 121) -> dict[str, Any]:
    grid = np.linspace(float(n_min), float(n_max), int(points)) * n_0
    mu = np.asarray([nvg_core_thermodynamics(n) for n in grid])
    mass = np.asarray([lambda_effective_mass(n) for n in grid])
    crossing = np.flatnonzero(mu >= mass)
    onset = float(grid[crossing[0]] / n_0) if crossing.size else None
    return {
        "density_ratio": grid / n_0,
        "mu_nucleon": mu,
        "lambda_mass": mass,
        "lambda_onset_n0": onset,
        "transition_n0": 2.0,
        "status": EVIDENCE_STATUS,
        "missing": "beta-equilibrated multi-species hyperon EOS, phase construction and independent likelihood",
    }


def raw_tov_summary() -> dict[str, Any]:
    """Run a small raw NL3 hyperon curve from the maintained TOV producer."""

    import nvg_hyperon_puzzle_tov as tov

    eps, pressure = tov.get_nl3_eos("hyperon")
    radii, masses = tov.generate_mr_curve(eps, pressure, p_max=320.0, n_points=24)
    if masses.size == 0:
        return {"m_max": None, "r14": None, "status": "NO_RAW_TOV_SOLUTIONS"}
    imax = int(np.argmax(masses))
    stable_m = masses[: imax + 1]
    stable_r = radii[: imax + 1]
    r14 = float(np.interp(1.4, np.sort(stable_m), stable_r[np.argsort(stable_m)])) if stable_m.max() >= 1.4 else None
    return {"m_max": float(masses[imax]), "r14": r14,
            "status": "DERIVED_RAW_TOV_NO_TARGET_RESCALING",
            "solver": "nvg_hyperon_puzzle_tov.generate_mr_curve"}


def main() -> dict[str, Any]:
    thresholds = compute_thresholds()
    raw = raw_tov_summary()
    onset = thresholds["lambda_onset_n0"]
    onset_text = f"{onset:.3f} n0" if onset is not None else "not reached on scanned grid"
    print("=" * 80)
    print("  NVG HYPERON THRESHOLD AUDIT (RUNTIME / RAW TOV)")
    print("=" * 80)
    print(f"Lambda threshold proxy: {onset_text}; transition={thresholds['transition_n0']:.1f} n0")
    print(f"Raw NL3+Lambda TOV: M_max={raw['m_max']}, R_1.4={raw['r14']}; status={raw['status']}")
    print(f"Evidence status: {thresholds['status']}")
    print(f"Missing: {thresholds['missing']}")
    print("No independent hyperon-resolution claim is emitted by this proxy calculation.")
    print("=" * 80)
    return {"thresholds": thresholds, "tov": raw}


if __name__ == "__main__":
    main()
