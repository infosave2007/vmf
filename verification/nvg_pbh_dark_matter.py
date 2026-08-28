#!/usr/bin/env python3
"""PBH ladder and benchmark-constraint bookkeeping.

The mass ladder is imported from the canonical nvg_pbh_mass_spectrum producer.
The abundance profile and constraint curves are explicit calibration/benchmark
inputs; this module has no formation solver or event-level likelihood, so it
reports comparisons without calling them observational exclusions or
detections.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nvg_hayward_evaporation import M_CRIT
from nvg_pbh_mass_spectrum import get_pbh_mass

M_SUN_G = 1.989e33


def get_subaru_limit(M_msun: float) -> float:
    """Declared log-linear microlensing benchmark (not an event likelihood)."""

    if M_msun <= 1e-13:
        return 1.0
    if M_msun < 1e-8:
        frac = (math.log10(M_msun) + 13.0) / 5.0
        return 10.0 ** (-3.0 * frac)
    return 1e-3


def get_hawking_limit(M_msun: float) -> float:
    """Standard-PBH evaporation benchmark, retained for comparison only."""

    if M_msun >= 1e-16:
        return 1.0
    if M_msun > 1e-18:
        frac = (math.log10(M_msun) + 18.0) / 2.0
        return 10.0 ** (-8.0 * (1.0 - frac))
    return 1e-8


def compute_spectrum(
    *,
    cycles: np.ndarray | list[int] | None = None,
    abundance_peak: int = -21,
    abundance_width: float = 1.3,
) -> dict[str, Any]:
    """Compute the calibrated ladder profile and benchmark margins."""

    N_vals = np.arange(-30, 13, dtype=int) if cycles is None else np.asarray(cycles, dtype=int)
    if N_vals.ndim != 1 or len(N_vals) == 0 or abundance_width <= 0.0:
        raise ValueError("cycles must be a non-empty vector and abundance_width positive")
    M_vals = np.asarray([get_pbh_mass(int(N)) for N in N_vals], dtype=float)
    f_unnorm = np.exp(-((N_vals - abundance_peak) ** 2) / (2.0 * abundance_width**2))
    f_pbh = f_unnorm / np.sum(f_unnorm)

    # Use the exact Hayward extremal mass from the canonical evaporation producer.
    mcrit_msun = float(M_CRIT / M_SUN_G)

    rows: list[dict[str, Any]] = []
    for N, M, f in zip(N_vals, M_vals, f_pbh):
        if M < mcrit_msun:
            limit = get_subaru_limit(float(M))
            source = "Subaru/EROS benchmark (gravitational)"
            horizon_class = "horizonless remnant in Hayward model"
        elif M <= 100.0:
            limit = 1e-3
            source = "stellar-PBH merger-rate benchmark"
            horizon_class = "horizon branch"
        else:
            limit = 1.0
            source = "no benchmark applied"
            horizon_class = "horizon branch"
        rows.append(
            {
                "cycle": int(N),
                "mass_msun": float(M),
                "fraction_profile": float(f),
                "benchmark_limit": float(limit),
                "benchmark_ratio": float(f / limit),
                "constraint_source": source,
                "horizon_class": horizon_class,
                "evidence_status": "CALIBRATED_GRID_NO_LIKELIHOOD",
            }
        )
    return {
        "rows": rows,
        "abundance_peak": int(abundance_peak),
        "abundance_width": float(abundance_width),
        "fraction_sum": float(np.sum(f_pbh)),
        "mcrit_msun": mcrit_msun,
        "evidence_status": "CALIBRATED_GRID_NO_LIKELIHOOD",
        "observed_likelihood": None,
        "limitation": "No PBH formation solver or independent abundance likelihood is present.",
    }


def main() -> dict[str, Any]:
    state = compute_spectrum()
    print("=" * 80)
    print("     NVG PBH LADDER & BENCHMARK-CONSTRAINT BOOKKEEPING")
    print("=" * 80)
    print(f"Profile fraction sum                 : {state['fraction_sum']:.6f} (calibrated profile)")
    print(f"Profile peak cycle                   : N={state['abundance_peak']} (declared calibration)")
    print(f"Hayward extremal mass                : {state['mcrit_msun']:.4f} M_sun (model output)")
    print(f"{'Cycle':<8} | {'Mass (M_sun)':<18} | {'f_profile':<12} | {'f/limit':<12} | {'Class':<34}")
    print("-" * 100)
    keep = {-28, -25, -22, -20, -18, -15, -10, 0, 3, 10}
    for row in state["rows"]:
        if row["cycle"] in keep:
            print(
                f"N = {row['cycle']:<4d} | {row['mass_msun']:<18.2e} | "
                f"{row['fraction_profile']:<12.2e} | {row['benchmark_ratio']:<12.2e} | "
                f"{row['horizon_class']:<34}"
            )
    print("-" * 100)
    print("Evidence status: CALIBRATED_GRID_NO_LIKELIHOOD")
    print("Benchmark ratios are not observational exclusions; formation and abundance likelihood are absent.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
