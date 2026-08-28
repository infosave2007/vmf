#!/usr/bin/env python3
"""Runtime hyperon threshold proxy (no claim of puzzle resolution).

The thresholds are useful sensitivity outputs of the simplified VMF ansatz,
but a complete multi-species thermodynamic solver and an observational
likelihood are absent.  Static phase thresholds and the former "resolved"
summary have therefore been removed.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

n_0 = 0.16
m_N, m_Lambda, m_Sigma, m_Xi = 939.0, 1115.7, 1192.6, 1314.9
alpha_N, alpha_L, alpha_S, alpha_X = 1.0, 0.66, 0.66, 0.33
EVIDENCE_STATUS = "DERIVED_THRESHOLD_PROXY_NO_COMPLETE_HYPERON_LIKELIHOOD"


def vmf_scaling(x: float | np.ndarray) -> float | np.ndarray:
    value = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("density ratios must be finite and non-negative")
    result = (1.0 + 0.8 * value) ** (-0.25 / 0.8)
    return float(result) if np.ndim(result) == 0 else result


def repulsion_energy(x: float | np.ndarray) -> float | np.ndarray:
    scale = vmf_scaling(x)
    result = 50.0 * np.asarray(x, dtype=float) / np.asarray(scale) ** 2
    return float(result) if np.ndim(result) == 0 else result


def compute_thresholds(points: int = 121) -> dict[str, Any]:
    x = np.linspace(1.0, 6.0, int(points))
    thresholds: dict[str, float | None] = {"Lambda": None, "Sigma": None, "Xi": None}
    masses = {"Lambda": (m_Lambda, alpha_L, 2.0 / 3.0),
              "Sigma": (m_Sigma, alpha_S, 2.0 / 3.0),
              "Xi": (m_Xi, alpha_X, 1.0 / 3.0)}
    rows = []
    for ratio in x:
        scale = float(vmf_scaling(ratio))
        m_n = m_N * scale ** alpha_N
        kf = (3.0 * np.pi ** 2 * (ratio * n_0)) ** (1.0 / 3.0) * 197.3
        e_fermi = np.sqrt(kf ** 2 + m_n ** 2) - m_n
        u_n = float(repulsion_energy(ratio))
        mu_n = m_n + e_fermi + u_n
        row = {"density_ratio": float(ratio), "mu_n": float(mu_n)}
        for name, (vac, alpha, u_fraction) in masses.items():
            effective = vac * scale ** alpha
            threshold_condition = mu_n > effective + u_fraction * u_n
            row[f"{name}_mass"] = float(effective)
            row[f"{name}_appears"] = bool(threshold_condition)
            if thresholds[name] is None and threshold_condition:
                thresholds[name] = float(ratio)
        rows.append(row)
    return {"rows": rows, "thresholds_n0": thresholds, "status": EVIDENCE_STATUS,
            "missing": "beta-equilibrated multi-species hyperon EOS, phase construction and independent likelihood"}


def main() -> dict[str, Any]:
    result = compute_thresholds()
    print("=" * 78)
    print("  NVG HYPERON THRESHOLD PROXY (RUNTIME / ZERO-EVIDENCE BOUNDARY)")
    print("=" * 78)
    for name, onset in result["thresholds_n0"].items():
        print(f"{name:>7} onset: {onset if onset is not None else 'not reached'} n0")
    print(f"Status: {result['status']}")
    print(f"Missing: {result['missing']}")
    print("No complete hyperonic EOS or puzzle-resolution claim is emitted.")
    print("=" * 78)
    return result


if __name__ == "__main__":
    main()
