#!/usr/bin/env python3
"""Higgs-mass shift sensitivity from an explicit effective interaction.

The script evaluates the leading algebraic shift for a declared W-field scale
and coupling.  The coupling is not derived from a matched UV/EW model and no
Higgs likelihood is present, so the result is a model sensitivity rather than
an experimental verification.
"""

from __future__ import annotations

from typing import Any


W_0_GEV = 0.859
G_S_MH = 1.218
M_H_OBS_GEV = 125.25
SIG_M_H_GEV = 0.11


def compute_shift(
    *,
    W_0_GeV: float = W_0_GEV,
    g_s_mH: float = G_S_MH,
    m_H_obs_GeV: float = M_H_OBS_GEV,
    sigma_m_H_GeV: float = SIG_M_H_GEV,
) -> dict[str, Any]:
    """Compute the leading EFT mass shift from explicit model inputs."""

    if W_0_GeV < 0.0 or g_s_mH < 0.0 or m_H_obs_GeV <= 0.0 or sigma_m_H_GeV <= 0.0:
        raise ValueError("scale/coupling must be non-negative and masses positive")
    delta_m_h = (g_s_mH**2 * W_0_GeV**2) / (2.0 * m_H_obs_GeV)
    return {
        "W_0_GeV": float(W_0_GeV),
        "g_s_mH": float(g_s_mH),
        "m_H_obs_GeV": float(m_H_obs_GeV),
        "sigma_m_H_GeV": float(sigma_m_H_GeV),
        "delta_m_H_GeV": float(delta_m_h),
        "delta_m_H_MeV": float(delta_m_h * 1000.0),
        "uncertainty_ratio": float(delta_m_h / sigma_m_H_GeV),
        "evidence_status": "MODEL_SENSITIVITY_ONLY",
        "observed_likelihood": None,
        "limitation": "No matched Higgs-W EFT and no LHC pole-mass likelihood are present.",
    }


def main() -> dict[str, Any]:
    state = compute_shift()
    print("=" * 80)
    print("   NVG HIGGS-MASS SHIFT EFT SENSITIVITY")
    print("=" * 80)
    print(f"QCD vacuum scale W_0                    : {state['W_0_GeV']:.3f} GeV")
    print(f"Declared coupling g_s(m_H)              : {state['g_s_mH']:.3f}")
    print(f"Leading shift                            : {state['delta_m_H_MeV']:.3f} MeV")
    print(f"Fraction of declared LHC uncertainty    : {state['uncertainty_ratio'] * 100.0:.2f}%")
    print("Evidence status                          : MODEL_SENSITIVITY_ONLY")
    print("A pole-mass likelihood and matched renormalization are absent.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
