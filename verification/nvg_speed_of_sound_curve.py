#!/usr/bin/env python3
"""
NVG Verification: Speed of Sound c_s^2(n_B) Profile
---------------------------------------------------
Calculates the speed of sound c_s^2 = dP/deps as a function of the baryon density
n_B (in units of saturation density n_0) using the unified VMF EOS (with BPS
crust matching and a first-order phase transition to a conformal quark core).
Verifies the causality bound (c_s^2 <= 1) and compliance with the conformal limit (c_s^2 = 1/3) at high densities.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add local path to import EOS solving classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nvg_full_ns_eos import UnifiedEOS, n_0

def main():
    print("=" * 80)
    print("     NVG SPEED OF SOUND PROFILE c_s^2(n_B) VERIFICATION")
    print("=" * 80)
    
    # Phase transition parameters (same as nvg_full_ns_eos.py)
    n_trans = 2.0
    delta_eps = 350.0
    
    print(f"Loading Unified VMF EOS (Onset: {n_trans} n_0, Latent Heat: {delta_eps} MeV/fm³)...")
    eos = UnifiedEOS(n_trans, delta_eps)
    
    # Calculate density grid
    n_grid = np.logspace(-4, 1.5, 2000) * n_0
    eps_arr = eos.eps_arr
    p_arr = eos.p_arr
    
    # Compute cs^2 = dP/deps numerically
    deps = np.diff(eps_arr)
    dp = np.diff(p_arr)
    
    cs2 = np.zeros_like(deps)
    valid = deps > 0.0
    cs2[valid] = dp[valid] / deps[valid]
    
    # Baryon density at midpoints (in units of n_0)
    n_mid_n0 = ((n_grid[:-1] + n_grid[1:]) / 2.0) / n_0
    
    # Identify maximum speed of sound
    max_cs2 = np.max(cs2)
    max_idx = np.argmax(cs2)
    max_density = n_mid_n0[max_idx]
    
    # Hadronic phase max speed of sound (n < n_trans)
    hadronic_mask = n_mid_n0 < n_trans
    max_hadronic_cs2 = np.max(cs2[hadronic_mask]) if np.any(hadronic_mask) else 0.0
    
    # Asymptotic quark phase speed of sound (at very high density)
    high_density_mask = n_mid_n0 > 4.0
    asymptotic_cs2 = cs2[high_density_mask][-1] if np.any(high_density_mask) else 0.0
    
    print("-" * 80)
    print(f"Maximum overall speed of sound c_s^2/c^2  : {max_cs2:.4f} (at {max_density:.2f} n_0)")
    print(f"Max speed of sound in Hadronic phase      : {max_hadronic_cs2:.4f}")
    print(f"Asymptotic speed of sound (quark core)    : {asymptotic_cs2:.4f} (expected: ~0.3333)")
    print("-" * 80)
    
    # Print representative points
    sample_densities = [0.1, 0.5, 1.0, 1.5, 1.9, 2.5, 4.0, 6.0]
    print(f"  {'Baryon Density (n_0)':<25} | {'Energy Density (MeV/fm³)':<25} | {'Speed of Sound c_s²/c²':<20}")
    print("  " + "-" * 75)
    for nd in sample_densities:
        # Find closest index
        idx = np.argmin(np.abs(n_mid_n0 - nd))
        print(f"  {n_mid_n0[idx]:<25.2f} | {eps_arr[idx]:<25.2f} | {cs2[idx]:<20.4f}")
        
    print("-" * 80)
    
    # Visualizing and saving plot
    plot_path = os.path.join(os.path.dirname(__file__), "fig_speed_of_sound.png")
    plt.figure(figsize=(8, 5))
    plt.plot(n_mid_n0, cs2, color='#00aaff', linewidth=2.5, label=r'$c_s^2(n_B)$')
    plt.axhline(1.0/3.0, color='red', linestyle='--', label='Conformal Bound (1/3)')
    plt.axhline(1.0, color='gray', linestyle=':', label='Causality Limit (1.0)')
    plt.axvline(n_trans, color='orange', linestyle='-.', label='Phase Transition (2.0 $n_0$)')
    plt.xlabel(r'Baryon Density $n_B / n_0$', fontsize=12)
    plt.ylabel(r'Speed of Sound $c_s^2 / c^2$', fontsize=12)
    plt.title('Speed of Sound Profile in VMF Unified EOS', fontsize=14, fontweight='bold')
    plt.xlim(0, 8.0)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved speed of sound plot to: {plot_path}")
    print("-" * 80)
    
    # Assertions for correctness
    assert max_cs2 < 1.0, "EOS violates causality (c_s^2 >= 1.0)!"
    assert abs(asymptotic_cs2 - 1.0/3.0) < 0.05, "Quark phase does not approach the conformal limit!"
    
    print("Speed of sound curve verification PASSED.")

if __name__ == "__main__":
    main()
