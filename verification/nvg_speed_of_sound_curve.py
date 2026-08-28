#!/usr/bin/env python3
"""Runtime speed-of-sound profile from the maintained UnifiedEOS producer.

The derivative dP/dε is recomputed from the EOS table for every invocation.
Causality and conformal-limit rows are internal model checks; no observational
sound-speed likelihood is bundled.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from nvg_full_ns_eos import UnifiedEOS, n_0


def compute_speed_sound() -> dict[str, Any]:
    """Build the canonical simplified EOS and derive c_s^2 at runtime."""

    n_trans = 2.0
    delta_eps = 350.0
    eos = UnifiedEOS(n_trans, delta_eps)
    # UnifiedEOS builds this documented grid internally; recreate its coordinate
    # only to label the returned derivative samples.
    n_grid = np.logspace(-4, 1.5, len(eos.eps_arr)) * n_0
    deps = np.diff(eos.eps_arr)
    dp = np.diff(eos.p_arr)
    cs2 = np.zeros_like(deps)
    valid = deps > 0.0
    cs2[valid] = dp[valid] / deps[valid]
    n_mid_n0 = ((n_grid[:-1] + n_grid[1:]) / 2.0) / n_0
    max_cs2 = float(np.max(cs2))
    max_idx = int(np.argmax(cs2))
    hadronic_mask = n_mid_n0 < n_trans
    high_density_mask = n_mid_n0 > 4.0
    return {
        "n_mid_n0": n_mid_n0,
        "eps_arr": eos.eps_arr,
        "cs2": cs2,
        "max_cs2": max_cs2,
        "max_density_n0": float(n_mid_n0[max_idx]),
        "max_hadronic_cs2": float(np.max(cs2[hadronic_mask])) if np.any(hadronic_mask) else 0.0,
        "asymptotic_cs2": float(cs2[high_density_mask][-1]) if np.any(high_density_mask) else 0.0,
        "n_trans": n_trans,
        "delta_eps": delta_eps,
        "evidence_status": "CANONICAL_MODEL_CONSISTENCY",
        "observed_likelihood": None,
        "canonical_producer": "verification/nvg_full_ns_eos.py:UnifiedEOS",
        "limitation": "No dense-matter sound-speed observation or likelihood is present.",
    }


def main() -> dict[str, Any]:
    state = compute_speed_sound()
    n_mid_n0 = state["n_mid_n0"]
    eps_arr = state["eps_arr"]
    cs2 = state["cs2"]
    n_trans = state["n_trans"]
    print("=" * 80)
    print("     NVG SPEED OF SOUND PROFILE c_s^2(n_B) MODEL CHECK")
    print("=" * 80)
    print(f"Maximum c_s^2/c^2                     : {state['max_cs2']:.4f} (at {state['max_density_n0']:.2f} n_0)")
    print(f"Maximum hadronic c_s^2/c^2            : {state['max_hadronic_cs2']:.4f}")
    print(f"High-density c_s^2/c^2                : {state['asymptotic_cs2']:.4f} (conformal input 1/3)")
    print("-" * 80)
    print(f"{'Density (n_0)':<18} | {'Energy density':<20} | {'c_s^2/c^2':<14}")
    print("-" * 80)
    for density in (0.1, 0.5, 1.0, 1.5, 1.9, 2.5, 4.0, 6.0):
        idx = int(np.argmin(np.abs(n_mid_n0 - density)))
        print(f"{n_mid_n0[idx]:<18.2f} | {eps_arr[idx]:<20.2f} | {cs2[idx]:<14.4f}")

    plot_path = os.path.join(_HERE, "fig_speed_of_sound.png")
    plt.figure(figsize=(8, 5))
    plt.plot(n_mid_n0, cs2, color="#00aaff", linewidth=2.5, label=r"$c_s^2(n_B)$")
    plt.axhline(1.0 / 3.0, color="red", linestyle="--", label="Conformal input (1/3)")
    plt.axhline(1.0, color="gray", linestyle=":", label="Causality bound (1.0)")
    plt.axvline(n_trans, color="orange", linestyle="-.", label="Model transition (2.0 n_0)")
    plt.xlabel(r"Baryon Density $n_B / n_0$")
    plt.ylabel(r"Speed of Sound $c_s^2 / c^2$")
    plt.title("Speed of Sound Profile in the UnifiedEOS model")
    plt.xlim(0, 8.0)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved speed-of-sound plot to: {plot_path}")
    print("Evidence status: CANONICAL_MODEL_CONSISTENCY")
    print("No observational sound-speed claim is made by this model check.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
