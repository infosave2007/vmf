#!/usr/bin/env python3
"""Conditional neutrino-sector calculation for neutrinoless double beta decay.

Mass splittings and mixing angles are declared NuFIT/PDG inputs, while the
theta-seesaw m3 value is a model anchor.  The script recomputes masses and
Majorana-phase extrema, but it does not evaluate a nuclear matrix-element or
experimental event likelihood; the result is conditional phenomenology.
"""

from __future__ import annotations

import math
from typing import Any


# Declared oscillation inputs (NuFIT 5.2 / PDG-style central values).
DM21 = 7.42e-5
DM31_NO = 2.517e-3
DM32_IO = 2.498e-3
S12_2 = 0.304
S13_2 = 0.0222
M3_THETASESAW = 50.3e-3
SUM_README = 59.0e-3
DESI_BOUND = 72.0e-3
KATRIN_BOUND = 0.45
NUBB_PROJECTED_REACH = 10e-3


def spectrum_from_m3(m3: float) -> tuple[float, float, float]:
    """Normal-ordering masses from the lightest/atmospheric anchor m3."""

    if m3 < 0.0:
        raise ValueError("m3 must be non-negative")
    m1 = math.sqrt(max(m3**2 - DM31_NO, 0.0))
    m2 = math.sqrt(m1**2 + DM21)
    return m1, m2, m3


def min_sum_no() -> float:
    return math.sqrt(DM21) + math.sqrt(DM31_NO)


def min_sum_io() -> float:
    m2 = math.sqrt(DM32_IO)
    return math.sqrt(m2**2 - DM21) + m2


def m_bb_range(m1: float, m2: float, m3: float) -> tuple[float, float]:
    """Extremal effective Majorana mass over two independent phases."""

    terms = [
        (1.0 - S12_2) * (1.0 - S13_2) * m1,
        S12_2 * (1.0 - S13_2) * m2,
        S13_2 * m3,
    ]
    total = sum(terms)
    return max(0.0, 2.0 * max(terms) - total), total


def compute_neutrino_observables() -> dict[str, Any]:
    m1, m2, m3 = spectrum_from_m3(M3_THETASESAW)
    total = m1 + m2 + m3
    bb_min, bb_max = m_bb_range(m1, m2, m3)
    m_beta = math.sqrt(
        (1.0 - S12_2) * (1.0 - S13_2) * m1**2
        + S12_2 * (1.0 - S13_2) * m2**2
        + S13_2 * m3**2
    )
    min_m1, min_m2, min_m3 = 0.0, math.sqrt(DM21), math.sqrt(DM31_NO)
    min_total = min_m1 + min_m2 + min_m3
    min_bb = m_bb_range(min_m1, min_m2, min_m3)
    return {
        "anchor_masses_eV": (m1, m2, m3),
        "anchor_sum_eV": total,
        "anchor_m_beta_eV": m_beta,
        "anchor_m_bb_range_eV": (bb_min, bb_max),
        "minimal_no_sum_eV": min_total,
        "minimal_no_m_bb_range_eV": min_bb,
        "DESI_bound_eV": DESI_BOUND,
        "KATRIN_bound_eV": KATRIN_BOUND,
        "projected_nubb_reach_eV": NUBB_PROJECTED_REACH,
        "evidence_status": "CONDITIONAL_MODEL_COMPARISON",
        "observed_likelihood": None,
        "provenance": "NuFIT 5.2/PDG central values plus theta-seesaw m3 anchor",
        "limitation": "No nuclear matrix-element, detector-response, or event likelihood is present.",
    }


def main() -> dict[str, Any]:
    state = compute_neutrino_observables()
    m1, m2, m3 = state["anchor_masses_eV"]
    bb_min, bb_max = state["anchor_m_bb_range_eV"]
    min_bb_min, min_bb_max = state["minimal_no_m_bb_range_eV"]
    print("=" * 72)
    print(" NVG THETA-SEESAW NEUTRINO CONDITIONAL CALCULATION")
    print("=" * 72)
    print(f"Minimal normal-ordering sum               : {state['minimal_no_sum_eV'] * 1e3:.1f} meV")
    print(f"Minimal inverted-ordering sum             : {min_sum_io() * 1e3:.1f} meV")
    print(f"Anchor masses (m1,m2,m3)                  : {m1 * 1e3:.2f}, {m2 * 1e3:.2f}, {m3 * 1e3:.2f} meV")
    print(f"Anchor sum(m_nu)                           : {state['anchor_sum_eV'] * 1e3:.1f} meV")
    print(f"Anchor m_beta                              : {state['anchor_m_beta_eV'] * 1e3:.2f} meV")
    print(f"Anchor m_bb phase range                    : {bb_min * 1e3:.2f}–{bb_max * 1e3:.2f} meV")
    print(f"Minimal-NO m_bb phase range                : {min_bb_min * 1e3:.2f}–{min_bb_max * 1e3:.2f} meV")
    print(f"Declared DESI bound                        : {DESI_BOUND * 1e3:.0f} meV")
    print(f"Declared KATRIN bound                      : {KATRIN_BOUND * 1e3:.0f} meV")
    print(f"Projected 0nubb reach                      : {NUBB_PROJECTED_REACH * 1e3:.0f} meV")
    print("Evidence status                            : CONDITIONAL_MODEL_COMPARISON")
    print("No nuclear-matrix-element or detector likelihood is evaluated.")
    print("=" * 72)
    return state


if __name__ == "__main__":
    main()
