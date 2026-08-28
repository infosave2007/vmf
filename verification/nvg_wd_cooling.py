#!/usr/bin/env python3
"""White-dwarf cooling sensitivity calculation.

The mass scaling and VMF melting fraction are explicit model inputs.  The
calculation reports the resulting fractional age shift but does not compare it
to a real Gaia/SDSS catalogue likelihood: no cooling solver, selection model or
catalogue is included.
"""

from __future__ import annotations

from typing import Any


M_SOLAR_REF = 0.6
DELTA_W_0 = 1.2e-6
ALPHA_VMF = 1.5
MASS_GRID = (0.4, 0.6, 0.8, 1.0, 1.2)


def compute_cooling_grid(
    masses: tuple[float, ...] = MASS_GRID,
    *,
    mass_ref: float = M_SOLAR_REF,
    delta_w0: float = DELTA_W_0,
    alpha_vmf: float = ALPHA_VMF,
) -> dict[str, Any]:
    """Compute the explicit WD transport sensitivity grid."""

    if mass_ref <= 0.0 or delta_w0 < 0.0 or alpha_vmf < 0.0:
        raise ValueError("model inputs must be non-negative and mass_ref positive")
    rows = []
    for mass in masses:
        if mass <= 0.0:
            raise ValueError("white-dwarf masses must be positive")
        rho_c = 1e6 * (mass / mass_ref) ** 2
        delta_w = delta_w0 * (mass / mass_ref) ** 2
        age_shift = -alpha_vmf * delta_w
        rows.append(
            {
                "mass_msun": float(mass),
                "core_density_g_cm3": float(rho_c),
                "melting_fraction": float(delta_w),
                "age_shift_fraction": float(age_shift),
                "benchmark_within_5pct": bool(abs(age_shift) < 0.05),
            }
        )
    return {
        "rows": rows,
        "evidence_status": "MODEL_SENSITIVITY_ONLY",
        "observed_likelihood": None,
        "limitation": "No Gaia/SDSS cooling-age catalogue and likelihood are present.",
    }


def main() -> dict[str, Any]:
    state = compute_cooling_grid()
    print("=" * 80)
    print("     NVG WHITE-DWARF COOLING SENSITIVITY GRID")
    print("=" * 80)
    print("Mass-dependent transport shifts from explicit VMF inputs; no catalogue fit.")
    print(f"{'Mass':<10} | {'Core density':<18} | {'Melting':<14} | {'Age shift':<14} | {'5% benchmark':<14}")
    print("-" * 80)
    for row in state["rows"]:
        print(
            f"{row['mass_msun']:<10.1f} | {row['core_density_g_cm3']:<18.2e} | "
            f"{row['melting_fraction']:<14.2e} | {row['age_shift_fraction']:<14.2e} | "
            f"{str(row['benchmark_within_5pct']):<14}"
        )
    print("-" * 80)
    print("Evidence status: MODEL_SENSITIVITY_ONLY")
    print("Gaia/SDSS compatibility is not evaluated without a cooling solver and catalogue likelihood.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
