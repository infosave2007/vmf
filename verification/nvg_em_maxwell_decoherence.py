#!/usr/bin/env python3
"""Formal Maxwell-sector dielectric-response calculation.

The exponential response is evaluated for an explicit dimensionless W-field
grid.  It is a formal EFT sensitivity table; matching to dense-matter photon
data or a magnetar observable is outside the available equations/data.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_dielectric_table(
    w_ratios: np.ndarray | None = None,
) -> dict[str, Any]:
    ratios = np.linspace(0.0, 1.0, 10) if w_ratios is None else np.asarray(w_ratios, dtype=float)
    if ratios.ndim != 1 or not np.all(np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError("w_ratios must be finite and non-negative")
    eps_ratios = np.exp(-2.0 * ratios)
    return {
        "w_ratios": ratios,
        "eps_ratios": eps_ratios,
        "evidence_status": "FORMAL_EFT_SENSITIVITY",
        "observed_likelihood": None,
        "limitation": "No dense-matter electromagnetic data or matching likelihood is present.",
    }


def main() -> dict[str, Any]:
    state = compute_dielectric_table()
    print("=" * 72)
    print("  NVG ELECTROMAGNETIC SECTOR FORMAL RESPONSE")
    print("=" * 72)
    print(f"{'Environment':>22} | {'(alpha W)/M_W':>15} | {'eps_eff/eps_0':>18}")
    print("-" * 62)
    for ratio, eps_ratio in zip(state["w_ratios"], state["eps_ratios"]):
        if ratio == 0.0:
            environment = "interstellar-vacuum limit"
        elif ratio == 1.0:
            environment = "dense-core model limit"
        else:
            environment = "intermediate model point"
        print(f"{environment:>22} | {ratio:15.2f} | {eps_ratio:18.4f}")
    print("-" * 72)
    print("Evidence status: FORMAL_EFT_SENSITIVITY")
    print("The table is not a magnetar-field solution or measured result.")
    print("=" * 72)
    return state


if __name__ == "__main__":
    main()
