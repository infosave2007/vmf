"""Shared canonical neutron-star transition metadata.

The transition point is not an independently predicted parameter.  It is the
survivor selected by :mod:`nvg_ns_parameter_scan` using the same J0740,
GW170817, and NICER constraints that appear in the downstream comparisons.
Keeping the point and that provenance in one small, importable record prevents
the two tidal entry points and their consumers from silently drifting apart.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


# These are model inputs selected by the explicit scan, not measured values and
# not free parameters to be counted as independent evidence downstream.
CANONICAL_PARAMETERS: dict[str, float] = {
    "n_trans_ratio": 2.0,
    "delta_eps_ratio": 0.0,
    "cs2_q": 1.0 / 3.0,
}


# Machine-readable provenance required at every consumer boundary.  The
# selection observations are intentionally named here so reports cannot call
# their reuse an independent test without changing this record.
CANONICAL_PROVENANCE: dict[str, Any] = {
    "status": "CONDITIONAL_IN_SAMPLE",
    "kind": "in_sample_selection",
    "selected_by": "verification/nvg_ns_parameter_scan.py",
    "selection_method": "best_margin_survivor_on_explicit_grid",
    "independent": False,
    "held_out_observations": [],
    "observations_used_for_selection": ["J0740", "GW170817", "NICER"],
    "constraints": [
        {
            "name": "PSR J0740+6620 maximum mass",
            "source": "J0740",
            "criterion": "M_max >= 2.01 M_sun",
        },
        {
            "name": "GW170817 binary tidal deformability",
            "source": "GW170817",
            "criterion": "70 <= Lambda_tilde <= 720",
        },
        {
            "name": "NICER J0030+0451 radius",
            "source": "NICER",
            "criterion": "11.2 <= R_1.4 <= 13.2 km",
        },
    ],
    "selection_note": (
        "J0740/GW170817/NICER constraints are selection inputs; all downstream "
        "comparisons are conditional/in-sample and carry no independent evidence weight."
    ),
}


def canonical_selection() -> dict[str, Any]:
    """Return an isolated machine-readable canonical point and provenance."""

    parameters = deepcopy(CANONICAL_PARAMETERS)
    return {
        "parameters": parameters,
        "selected_point": deepcopy(parameters),
        "provenance": deepcopy(CANONICAL_PROVENANCE),
    }


__all__ = [
    "CANONICAL_PARAMETERS",
    "CANONICAL_PROVENANCE",
    "canonical_selection",
]
