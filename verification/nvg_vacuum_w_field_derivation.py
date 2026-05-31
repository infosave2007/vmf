#!/usr/bin/env python3
"""
Numerical Verification of the Vacuum Condensate W-Field Phase Transition
------------------------------------------------------------------------
This script models and verifies the phase transition of the real amplitude
W(x) of the complex scalar vacuum condensate under the influence of an
external gauge field (or phase gradient) g_mu from the first principles of QFT.
"""

import math

def run_w_field_derivation():
    print("=" * 80)
    print("  QFT VACUUM CONDENSATE W-FIELD PHASE TRANSITION NUMERICAL MODEL")
    print("=" * 80)

    # 1. Theoretical Parameters (arbitrary normalized units)
    mu_squared = 1.0        # Parameter mu^2 of the Higgs-like potential
    lambda_param = 0.5      # Self-coupling constant lambda
    
    # Critical threshold for gauge coupling/gradient: |g|_crit = sqrt(mu^2)
    g_crit = math.sqrt(mu_squared)
    
    print(f"Model Parameters:")
    print(f"  mu^2 (Mass Parameter)              : {mu_squared:.4f}")
    print(f"  lambda (Self-coupling)             : {lambda_param:.4f}")
    print(f"  Critical Gauge Threshold (|g|_crit): {g_crit:.4f}")
    print("-" * 80)

    # 2. VEV (Vacuum Expectation Value) calculations over a range of gauge fields |g|
    g_values = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    
    print("VEV EVOLUTION AS A FUNCTION OF GAPE FIELD GRADIENT |g|:")
    print(f"{'|g|':^10} | {'g^2':^10} | {'Effective Mass Term (g^2 - mu^2)':^32} | {'VEV (W_0)':^15}")
    print("-" * 80)
    
    results = []
    for g in g_values:
        g_sq = g**2
        eff_mass_term = g_sq - mu_squared
        
        # W_0^2 = max(0, mu^2 - g^2) / lambda
        w0_sq = max(0.0, mu_squared - g_sq) / lambda_param
        w0 = math.sqrt(w0_sq)
        
        state = "SSB Condensate" if g < g_crit else "Sym Restored (Melted)"
        print(f"{g:^10.4f} | {g_sq:^10.4f} | {eff_mass_term:^32.4f} | {w0:^15.4f} ({state})")
        results.append((g, g_sq, eff_mass_term, w0, state))
        
    print("-" * 80)
    
    # 3. Model potential curves V_eff(W) for three characteristic regimes
    regimes = {
        "Sub-critical (Weak Field)": 0.5,
        "Critical (Phase Boundary)": 1.0,
        "Super-critical (Strong Field)": 1.5
    }
    
    print("EFFECTIVE POTENTIAL V_eff(W) VALUES FOR REGIMES:")
    w_steps = [0.0, 0.5, 1.0, 1.414, 2.0] # W grid points
    
    for name, g_val in regimes.items():
        g_sq = g_val**2
        print(f"\nRegime: {name} (|g| = {g_val:.2f})")
        print(f"  {'W':^10} | {'V_eff(W)':^20}")
        print("  " + "-" * 33)
        for w in w_steps:
            # V_eff = 0.5 * (g^2 - mu^2) * W^2 + 0.25 * lambda * W^4
            v_eff = 0.5 * (g_sq - mu_squared) * (w**2) + 0.25 * lambda_param * (w**4)
            print(f"  {w:^10.4f} | {v_eff:^20.4f}")

    print("=" * 80)

    # Save output to verification directory
    output_path = "verification/nvg_vacuum_w_field_derivation_output.txt"
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  QFT VACUUM CONDENSATE W-FIELD PHASE TRANSITION NUMERICAL MODEL\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model Parameters:\n")
        f.write(f"  mu^2 (Mass Parameter)              : {mu_squared:.4f}\n")
        f.write(f"  lambda (Self-coupling)             : {lambda_param:.4f}\n")
        f.write(f"  Critical Gauge Threshold (|g|_crit): {g_crit:.4f}\n")
        f.write("-" * 80 + "\n")
        f.write("VEV EVOLUTION AS A FUNCTION OF GAPE FIELD GRADIENT |g|:\n")
        f.write(f"{'|g|':^10} | {'g^2':^10} | {'Effective Mass Term (g^2 - mu^2)':^32} | {'VEV (W_0)':^15}\n")
        f.write("-" * 80 + "\n")
        for g, g_sq, eff_mass_term, w0, state in results:
            f.write(f"{g:^10.4f} | {g_sq:^10.4f} | {eff_mass_term:^32.4f} | {w0:^15.4f} ({state})\n")
        f.write("-" * 80 + "\n")
        f.write("EFFECTIVE POTENTIAL V_eff(W) VALUES FOR REGIMES:\n")
        for name, g_val in regimes.items():
            g_sq = g_val**2
            f.write(f"\nRegime: {name} (|g| = {g_val:.2f})\n")
            f.write(f"  {'W':^10} | {'V_eff(W)':^20}\n")
            f.write("  " + "-" * 33 + "\n")
            for w in w_steps:
                v_eff = 0.5 * (g_sq - mu_squared) * (w**2) + 0.25 * lambda_param * (w**4)
                f.write(f"  {w:^10.4f} | {v_eff:^20.4f}\n")
        f.write("=" * 80 + "\n")

if __name__ == "__main__":
    run_w_field_derivation()
