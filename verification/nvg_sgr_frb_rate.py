#!/usr/bin/env python3
"""Mass-scaling sensitivity for a magnetar FRB-rate hypothesis.

The code evaluates the stated R ∝ M^-4 toy scaling at explicit benchmark
masses.  No mass measurements or burst-rate likelihood for a matched magnetar
sample are available here, so the empirical correlation is not promoted.
"""

from __future__ import annotations

from typing import Any


M_SGR = 1.10
M_STANDARD = 1.45
M_HEAVY = 1.60


def compute_rate_scaling(
    *,
    m_sgr: float = M_SGR,
    m_standard: float = M_STANDARD,
    m_heavy: float = M_HEAVY,
    exponent: float = -4.0,
) -> dict[str, Any]:
    """Compute relative rates for an explicit toy mass-scaling scenario."""

    masses = (m_sgr, m_standard, m_heavy)
    if any(m <= 0.0 for m in masses):
        raise ValueError("benchmark masses must be positive")
    rate_sgr = m_sgr**exponent
    rate_standard = m_standard**exponent
    rate_heavy = m_heavy**exponent
    return {
        "m_sgr": float(m_sgr),
        "m_standard": float(m_standard),
        "m_heavy": float(m_heavy),
        "exponent": float(exponent),
        "rate_sgr": float(rate_sgr),
        "rate_standard": float(rate_standard),
        "rate_heavy": float(rate_heavy),
        "enhancement_vs_standard": float(rate_sgr / rate_standard),
        "enhancement_vs_heavy": float(rate_sgr / rate_heavy),
        "evidence_status": "MODEL_SENSITIVITY_ONLY",
        "observed_likelihood": None,
        "limitation": "No independent magnetar mass sample and burst-rate likelihood are present.",
    }


def run_sgr_frb_verification() -> dict[str, Any]:
    """Compatibility CLI entry point; reports a sensitivity calculation."""

    state = compute_rate_scaling()
    print("=" * 74)
    print("  NVG MAGNETAR FRB-RATE MASS-SCALING SENSITIVITY")
    print("=" * 74)
    print(f"Benchmark masses (SGR/standard/heavy) : {state['m_sgr']:.2f}/{state['m_standard']:.2f}/{state['m_heavy']:.2f} M_sun")
    print(f"Toy scaling exponent                    : R ∝ M^{state['exponent']:.1f}")
    print(f"Relative rate (SGR)                     : {state['rate_sgr']:.4f}")
    print(f"Relative rate (standard)                : {state['rate_standard']:.4f}")
    print(f"Relative rate (heavy)                   : {state['rate_heavy']:.4f}")
    print(f"Sensitivity ratio SGR/standard          : {state['enhancement_vs_standard']:.2f}x")
    print(f"Sensitivity ratio SGR/heavy              : {state['enhancement_vs_heavy']:.2f}x")
    print("Evidence status                          : MODEL_SENSITIVITY_ONLY")
    print("No burst-rate likelihood or mass inference is present; correlation remains untested.")
    print("=" * 74)
    return state


if __name__ == "__main__":
    run_sgr_frb_verification()
