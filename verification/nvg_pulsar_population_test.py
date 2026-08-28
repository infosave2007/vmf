#!/usr/bin/env python3
"""ATNF cooling-population claim boundary.

The former script generated 10,000 stars and then treated the generated sample
as an ATNF test.  ATNF does not provide the required joint mass, age, envelope
and thermal-luminosity likelihood in this checkout, so the empirical claim is
retired instead of being simulated.
"""

from __future__ import annotations

from typing import Any

EVIDENCE_STATUS = "RETIRED_NO_ATNF_MASS_AGE_LUMINOSITY_LIKELIHOOD"
CATALOG_REFERENCE = "ATNF Pulsar Catalogue (Manchester et al.; catalogue metadata only)"


def run_pulsar_population_test() -> dict[str, Any]:
    return {
        "catalog": CATALOG_REFERENCE,
        "n_stars": None,
        "falsifiers": None,
        "mean_light_luminosity": None,
        "mean_heavy_luminosity": None,
        "status": EVIDENCE_STATUS,
        "missing": "mass/age/envelope-linked thermal luminosities and a selection-corrected cooling likelihood",
    }


def main() -> dict[str, Any]:
    result = run_pulsar_population_test()
    print("=" * 74)
    print("  NVG PULSAR POPULATION COOLING CHECK")
    print("=" * 74)
    print(f"Reference: {result['catalog']}")
    print("No synthetic ATNF population is generated.")
    print(f"Status: {result['status']}")
    print(f"Missing: {result['missing']}")
    print("No cooling dichotomy or falsifier count is reported.")
    print("=" * 74)
    return result


if __name__ == "__main__":
    main()
