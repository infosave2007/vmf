#!/usr/bin/env python3
"""Canonical discrete PBH mass ladder.

The ladder is a theory-level mapping from cycle index to mass.  It contains no
formation abundance, lensing selection, JWST occupation model, or merger-rate
likelihood.  Downstream consumers must therefore treat it as a model input and
not as an observed population.
"""

from __future__ import annotations

from typing import Any


MASS_ANCHOR_MSUN = 0.38
LADDER_BASE = 4.0


def get_pbh_mass(n_cycle: int) -> float:
    """Return the runtime mass (solar masses) for an integer ladder index."""

    if not isinstance(n_cycle, int):
        raise TypeError("n_cycle must be an integer")
    return MASS_ANCHOR_MSUN * LADDER_BASE ** n_cycle


def mass_ladder(cycles: list[int] | tuple[int, ...]) -> list[dict[str, Any]]:
    """Build a machine-readable ladder with explicit evidence semantics."""

    return [
        {
            "cycle": int(cycle),
            "mass_msun": float(get_pbh_mass(int(cycle))),
            "evidence_status": "THEORY_LADDER_ONLY",
        }
        for cycle in cycles
    ]


def main() -> dict[str, Any]:
    cycles = [-28, -25, -21, -15, 0, 10]
    rows = mass_ladder(cycles)
    print("=" * 78)
    print("  NVG DISCRETE PBH MASS LADDER (THEORY INPUT)")
    print("=" * 78)
    print(f"{'Cycle':<10} | {'Mass (M_sun)':<20} | {'Evidence status':<24}")
    print("-" * 78)
    for row in rows:
        print(f"N = {row['cycle']:<7d} | {row['mass_msun']:<20.4e} | {row['evidence_status']:<24}")
    print("-" * 78)
    print("The ladder is computed from M_N = 0.38 * 4^N at runtime.")
    print("No abundance, lensing, JWST occupation, or merger likelihood is supplied.")
    print("Evidence status: THEORY_LADDER_ONLY")
    print("=" * 78)
    return {
        "rows": rows,
        "evidence_status": "THEORY_LADDER_ONLY",
        "observed_likelihood": None,
        "limitation": "No formation-abundance, lensing/JWST occupation, or merger-rate likelihood is present.",
    }


if __name__ == "__main__":
    main()
