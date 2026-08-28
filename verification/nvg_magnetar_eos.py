#!/usr/bin/env python3
"""Magnetar dense-matter response sensitivity.

The VMF melting fraction and conservative loop dielectric response are reused
from the maintained nvg_magnetar_closure module.  The seed field and
magnetar-field interpretation remain explicit scenario inputs; no population
likelihood or magnetic-field inference is performed here.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nvg_magnetar_closure import eps_eff_ratio_loop, m_omega_star, melted_fraction, N_0

M_Omega_0 = 859.0
n_0 = N_0
alpha_v = 4.0
kappa_1 = 0.25
kappa_2 = 0.80


def m_star(n_b: float) -> float:
    """Canonical VMF omega-sector mass used by the closure producer."""

    return float(m_omega_star(n_b))


def calculate_magnetar_fields(n_core_ratio: float = 3.0, B_seed: float = 1e14) -> tuple[float, float, float]:
    """Compute loop-response amplification for an explicit seed field scenario."""

    if n_core_ratio < 0.0 or B_seed < 0.0:
        raise ValueError("density ratio and seed field must be non-negative")
    n_b = n_core_ratio * n_0
    f_melt = melted_fraction(n_b)
    eps_ratio = float(eps_eff_ratio_loop(n_b))
    amplification = 1.0 / eps_ratio**0.5
    return eps_ratio, amplification, B_seed * amplification


def compute_magnetar_state() -> dict[str, Any]:
    ratio = 3.0
    seed = 4.0e14
    eps_ratio, amplification, field = calculate_magnetar_fields(ratio, seed)
    return {
        "density_ratio": ratio,
        "seed_field_G": seed,
        "m_star_MeV": m_star(ratio * n_0),
        "melted_fraction": melted_fraction(ratio * n_0),
        "eps_ratio": eps_ratio,
        "amplification": amplification,
        "core_field_G": field,
        "evidence_status": "CANONICAL_MODEL_SENSITIVITY",
        "observed_likelihood": None,
        "canonical_producer": "verification/nvg_magnetar_closure.py:m_omega_star,eps_eff_ratio_loop",
        "limitation": "No independent magnetar mass/field likelihood is present.",
    }


def main() -> dict[str, Any]:
    state = compute_magnetar_state()
    print("=" * 74)
    print("  NVG MAGNETAR CORE RESPONSE SENSITIVITY")
    print("=" * 74)
    print(f"QCD anchor M_Omega                      : {M_Omega_0:.1f} MeV")
    print(f"Core density                             : {state['density_ratio']:.1f} n_0")
    print(f"Canonical m_omega*                       : {state['m_star_MeV']:.1f} MeV")
    print(f"Melted fraction                          : {state['melted_fraction']:.3f}")
    print(f"Loop dielectric ratio                    : {state['eps_ratio']:.5f}")
    print(f"Amplification for seed {state['seed_field_G']:.2e} G : {state['amplification']:.3f}x")
    print(f"Scenario core field                      : {state['core_field_G']:.3e} G")
    print("Evidence status                          : CANONICAL_MODEL_SENSITIVITY")
    print("No observed magnetic-field likelihood is evaluated.")
    print("=" * 74)
    return state


if __name__ == "__main__":
    main()
