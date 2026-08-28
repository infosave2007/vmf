#!/usr/bin/env python3
"""CHIME FRB claim boundary.

The repository contains no CHIME repeater-labelled DM/mass table.  A seeded
normal sample is therefore not a substitute for the catalogue and has been
removed.  The entry point records the missing mapping and does not emit a
p-value.
"""

from __future__ import annotations

from typing import Any

EVIDENCE_STATUS = "RETIRED_NO_CHIME_DM_MASS_MAPPING_OR_LIKELIHOOD"
CATALOG_REFERENCE = "CHIME/FRB Catalog 1 (Amiri et al., 2021, ApJS 257, 59)"


def run_chime_frb_check() -> dict[str, Any]:
    return {
        "catalog": CATALOG_REFERENCE,
        "catalog_rows": None,
        "repeater_rows": None,
        "p_value": None,
        "status": EVIDENCE_STATUS,
        "missing": "machine-readable repeater/non-repeater host DM and magnetar-mass mapping plus selection likelihood",
    }


def main() -> dict[str, Any]:
    result = run_chime_frb_check()
    print("=" * 74)
    print("  NVG CHIME FRB REPEATER/MASS CHECK")
    print("=" * 74)
    print(f"Reference: {result['catalog']}")
    print("No local repeater-linked DM/mass data are available; synthetic samples are not used.")
    print(f"Status: {result['status']}")
    print(f"Missing: {result['missing']}")
    print("No statistical distinction or confirmation is reported.")
    print("=" * 74)
    return result


if __name__ == "__main__":
    main()
