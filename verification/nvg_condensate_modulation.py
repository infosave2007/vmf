#!/usr/bin/env python3
"""
NVG Condensate Modulation (Podkletnov-Modanese Effect Simulator).

This script demonstrates how macroscopic quantum condensates (like
superconductors) interacting with the global QCD vacuum can produce
measurable weight-loss (Podkletnov 1992) or impulse forces (Modanese 2001).

Key equations:
  1. Static shielding (Podkletnov):
     F_g = m_0 * g * (1 - η * δΔ/Δ_0)
  2. Dynamic impulse (Modanese/Podkletnov):
     F_impulse ∝ m_0 * η * (dΔ/dt) / Δ_0

Where:
  - η is the vacuum coupling efficiency (small for YBCO, large for Graphene)
  - Δ is the superconducting/topological energy gap
"""

from __future__ import annotations


def podkletnov_shielding(mass_kg: float, eta: float, gap_reduction_fraction: float) -> tuple[float, float]:
    """
    Simulate the static weight reduction above a modulated condensate.
    Returns (effective_weight_N, weight_loss_percentage).
    """
    g = 9.80665
    weight_normal = mass_kg * g
    
    # F_g = m_0 * g * (1 - η * (Δ_0 - Δ_mod) / Δ_0)
    effective_gravity_factor = 1.0 - (eta * gap_reduction_fraction)
    effective_gravity_factor = max(effective_gravity_factor, 0.0) # cannot be negative
    
    effective_weight = weight_normal * effective_gravity_factor
    weight_loss_pct = (1.0 - effective_gravity_factor) * 100.0
    
    return effective_weight, weight_loss_pct


def impulse_generator(mass_kg: float, eta: float, gap_change_rate_eV_s: float, gap_0_eV: float) -> float:
    """
    Simulate the impulse force from a rapid condensate collapse (spark discharge).
    Returns relative force magnitude.
    """
    # F_impulse ∝ m_0 * η * (1/Δ_0) * (dΔ/dt)
    force_magnitude = mass_kg * eta * (gap_change_rate_eV_s / gap_0_eV)
    return force_magnitude


def main() -> None:
    print("=" * 80)
    print("NVG CONDENSATE MODULATION: PODKLETNOV-MODANESE EFFECT SIMULATOR")
    print("=" * 80)
    print()
    print("The NVG framework treats mass as vacuum tension. Macroscopic quantum")
    print("condensates (superconductors, topological insulators) can locally modulate")
    print("this tension, leading to measurable gravitational anomalies.")
    print()

    # 1. Podkletnov 1992 (Static Shielding)
    print("1. PODKLETNOV 1992 EXPERIMENT (STATIC SHIELDING)")
    print("   Setup: Rotating YBa2Cu3O7-x superconducting disk.")
    print("   Sample mass: 1.000 kg")
    print()
    print(f"   {'Material':>20}  {'η (Coupling)':>15}  {'Gap Reduc.':>12}  {'Weight Loss':>15}")
    print("   " + "-" * 66)
    
    scenarios = [
        ("YBCO (Podkletnov)", 0.02, 1.0),   # 2% coupling, full gap collapse
        ("YBCO (Weak)", 0.02, 0.1),         # 2% coupling, 10% gap collapse (poor rotation)
        ("Graphene (Future)", 0.50, 1.0),   # 50% coupling, full gap collapse
    ]
    
    test_mass = 1.0
    for name, eta, gap_red in scenarios:
        _, pct = podkletnov_shielding(test_mass, eta, gap_red)
        print(f"   {name:>20}  {eta:15.3f}  {gap_red*100:11.1f}%  {pct:14.2f}%")
        
    print()
    print("   CONCLUSION: The 2% weight loss reported by Podkletnov is consistent")
    print("   with a low-efficiency coupling (η ~ 0.02) expected from granular")
    print("   ceramic superconductors. Graphene could amplify this by 25x.")
    print()

    # 2. Podkletnov/Modanese 2001 (Impulse Generator)
    print("2. PODKLETNOV/MODANESE 2001 EXPERIMENT (IMPULSE BEAM)")
    print("   Setup: 1 MV, 10^4 A discharge through YBCO electrode.")
    print("   Effect: Beam of repulsive force, passes through walls, proportional to mass.")
    print()
    
    # Let's compare continuous rotation vs spark discharge
    gap_ybco = 0.03  # ~30 meV gap for YBCO
    
    # Rotation: small gap fluctuation over ~12 ms (83 Hz)
    rate_rotation = gap_ybco / 0.012  # eV/s
    
    # Discharge: gap collapses in ~10 ns
    rate_discharge = gap_ybco / 1e-8  # eV/s
    
    eta_ybco = 0.02
    
    f_rot = impulse_generator(1.0, eta_ybco, rate_rotation, gap_ybco)
    f_dis = impulse_generator(1.0, eta_ybco, rate_discharge, gap_ybco)
    
    print(f"   {'Modulation Method':>25}  {'dΔ/dt (eV/s)':>15}  {'Relative Force':>18}")
    print("   " + "-" * 62)
    print(f"   {'Mechanical Rotation (83Hz)':>25}  {rate_rotation:15.2e}  {f_rot:18.2e}")
    print(f"   {'Spark Discharge (10ns)':>25}  {rate_discharge:15.2e}  {f_dis:18.2e}")
    print()
    print(f"   Amplification factor: {f_dis/f_rot:.1e}x")
    print()
    print("   CONCLUSION: The spark discharge forces a catastrophic phase transition,")
    print("   creating a vacuum shockwave millions of times stronger than mechanical")
    print("   rotation. Because the wave is a vacuum deformation, it passes through")
    print("   all matter (walls) and exerts force proportional to the target's mass.")
    print()


if __name__ == "__main__":
    main()
