#!/usr/bin/env python3
"""de Sitter-core standing-wave forward calculation.

The core radius and scalar standing-wave frequency follow from the stated
constant-density Hayward-core ansatz.  They are model outputs only: no
GW150914 detector waveform, matched-filter response or echo likelihood is
included, so the frequency is not an empirical result.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def calculate_ds_core_properties(
    M_solar: float, M_omega_0: float = 859.0
) -> tuple[float, float, float]:
    """Compute core radius (km), fundamental frequency (Hz), and period (s)."""

    if M_solar <= 0.0 or M_omega_0 <= 0.0:
        raise ValueError("mass and QCD anchor must be positive")
    G_c2 = 1.476
    m_tot = M_solar * G_c2
    hbar_c = 197.327
    rho_c_mev_fm3 = M_omega_0**4 / hbar_c**3
    rho_c_geom = rho_c_mev_fm3 * 1.323e-6
    r_0 = (3.0 * m_tot / (4.0 * np.pi * rho_c_geom)) ** (1.0 / 3.0)
    f_1 = 299792.458 / (2.0 * r_0)
    return float(r_0), float(f_1), float(1.0 / f_1)


def compute_core_grid() -> dict[str, Any]:
    cases = [
        {"name": "Stellar black hole", "M": 10.0},
        {"name": "GW150914 remnant benchmark", "M": 65.0},
        {"name": "Supermassive seed", "M": 4e5},
        {"name": "M87* benchmark", "M": 6.5e9},
    ]
    rows = []
    for case in cases:
        r_0, f_1, period = calculate_ds_core_properties(case["M"])
        rows.append({"name": case["name"], "mass_msun": case["M"], "r0_km": r_0, "frequency_hz": f_1, "period_s": period})
    return {
        "rows": rows,
        "evidence_status": "FORWARD_MODEL_ONLY",
        "observed_likelihood": None,
        "limitation": "No detector waveform or echo likelihood is available.",
    }


def main() -> dict[str, Any]:
    state = compute_core_grid()
    print("=" * 80)
    print("     NVG REGULAR de SITTER CORE STANDING-WAVE FORWARD MODEL")
    print("=" * 80)
    print(f"{'Case':<30} | {'Mass':<12} | {'r_0 (km)':<12} | {'f_1 (Hz)':<14} | {'T_1 (ms)':<12}")
    print("-" * 90)
    for row in state["rows"]:
        print(f"{row['name']:<30} | {row['mass_msun']:<12.3g} | {row['r0_km']:<12.3f} | {row['frequency_hz']:<14.3g} | {row['period_s'] * 1e3:<12.3g}")
    print("-" * 90)
    print("Evidence status: FORWARD_MODEL_ONLY")
    print("Standing-wave frequencies are not matched to an observed GW echo.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
