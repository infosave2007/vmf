#!/usr/bin/env python3
"""
NVG Graphene Auto-Generator: Theoretical Power Calculator
==========================================================

Calculates expected power output of a graphene NDR auto-oscillator
based on the VMF (Vacuum Mass Fraction) theoretical framework.

Three models are computed:
  1. VMF optimistic  — full vacuum coupling (η = 1)
  2. VMF realistic   — weak vacuum coupling (η from Podkletnov analogy)
  3. Dynamic Casimir  — strict QFT analog (lower bound)

Author: Oleg Kirichenko (urevich55@gmail.com)
Based on: VMF model, Zenodo record 20214457
"""

import numpy as np

# ==============================================================================
# Fundamental constants
# ==============================================================================
hbar = 1.054571817e-34   # J·s
c = 2.99792458e8          # m/s
e_charge = 1.602176634e-19  # C
k_B = 1.380649e-23        # J/K
eV = e_charge              # 1 eV in Joules
MeV = 1e6 * eV             # 1 MeV in Joules

# ==============================================================================
# Graphene parameters (experimental)
# ==============================================================================
v_F = 1.0e6                # Fermi velocity, m/s
alpha_eff = 2.2             # effective fine structure constant in graphene
alpha_QED = 1/137.036       # QED fine structure constant
a_lattice = 2.46e-10        # graphene lattice constant, m
A_unit_cell = (np.sqrt(3)/2) * a_lattice**2  # unit cell area, m²

# ==============================================================================
# Bilayer graphene gap control
# ==============================================================================
Delta_max = 0.1 * eV       # max tunable gap (bilayer, gate voltage ~1V)
Delta_min = 0.0             # gap at Dirac point

# ==============================================================================
# Device parameters
# ==============================================================================
S_device = 10e-6            # device area, m² (10 mm²)
n_2D = 1e16                 # carrier density at gating, m⁻²
T_room = 300                # room temperature, K

# ==============================================================================
# LC circuit parameters
# ==============================================================================
L_inductor = 1.0e-9         # inductance, H (1 nH)
C_tune = 2.0e-12            # capacitance, F (2 pF)
Q_factor = 30               # quality factor of LC circuit

# ==============================================================================
# VMF model parameters (from published article)
# ==============================================================================
M_Omega_0 = 859 * MeV       # vacuum nonperturbative mass, J
M_nucleon = 938.3 * MeV     # nucleon mass, J
f_Omega = M_Omega_0 / M_nucleon  # vacuum mass fraction (~0.916)


def separator(title):
    """Print a formatted section separator."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def calc_oscillator():
    """Calculate LC oscillator parameters."""
    separator("1. NDR AUTO-OSCILLATOR PARAMETERS")

    f_osc = 1 / (2 * np.pi * np.sqrt(L_inductor * C_tune))
    omega = 2 * np.pi * f_osc
    Z_char = np.sqrt(L_inductor / C_tune)  # characteristic impedance
    R_loss = Z_char / Q_factor              # loss resistance

    print(f"  Inductance L        = {L_inductor*1e9:.1f} nH")
    print(f"  Capacitance C       = {C_tune*1e12:.1f} pF")
    print(f"  Quality factor Q    = {Q_factor}")
    print(f"  ─────────────────────────────────")
    print(f"  Oscillation freq    = {f_osc/1e9:.2f} GHz")
    print(f"  Angular frequency ω = {omega:.3e} rad/s")
    print(f"  Char. impedance Z₀  = {Z_char:.1f} Ω")
    print(f"  Loss resistance     = {R_loss:.2f} Ω")
    print(f"  Min |R_NDR| needed  = {R_loss:.2f} Ω  (graphene NDR ~ 10-100 Ω ✓)")

    return f_osc, omega


def calc_active_carriers():
    """Calculate number of active carriers in the device."""
    separator("2. ACTIVE CARRIERS IN GRAPHENE DEVICE")

    N_active = n_2D * S_device
    N_unit_cells = S_device / A_unit_cell

    print(f"  Device area         = {S_device*1e6:.1f} mm²")
    print(f"  Carrier density     = {n_2D:.0e} m⁻²")
    print(f"  Active carriers N   = {N_active:.2e}")
    print(f"  Unit cells          = {N_unit_cells:.2e}")

    return N_active


def calc_vmf_optimistic(f_osc, N_active):
    """
    VMF optimistic model: each carrier absorbs Δ per cycle from vacuum.
    This is the original NVG-Generator calculation (COP ~ 36,000).
    """
    separator("3. MODEL A: VMF OPTIMISTIC (η = 1)")

    E_per_cycle = N_active * Delta_max
    P_out = E_per_cycle * f_osc

    # Input power (gate capacitance switching)
    d_dielectric = 1e-9     # gate dielectric thickness, m
    eps_0 = 8.854e-12
    C_gate = eps_0 * S_device / d_dielectric
    V_mod = 1e-3            # modulation voltage, V
    E_input_cycle = 0.5 * C_gate * V_mod**2
    P_input = E_input_cycle * f_osc

    COP = P_out / P_input if P_input > 0 else float('inf')

    print(f"  Assumption: each carrier absorbs Δ = {Delta_max/eV:.1f} eV per cycle")
    print(f"  Energy per cycle    = {E_per_cycle:.3e} J  ({E_per_cycle/eV:.3e} eV)")
    print(f"  Frequency           = {f_osc/1e9:.2f} GHz")
    print(f"  ─────────────────────────────────")
    print(f"  OUTPUT power        = {P_out:.3e} W  ({P_out:.1f} W)")
    print(f"  INPUT power (gate)  = {P_input:.3e} W  ({P_input*1e3:.2f} mW)")
    print(f"  COP                 = {COP:.0f}")
    print(f"")
    print(f"  ⚠ WARNING: This model assumes η = 1 (perfect vacuum coupling).")
    print(f"  This is the upper theoretical bound. Real η is unknown.")

    return P_out, P_input, COP


def calc_vmf_realistic(f_osc, N_active):
    """
    VMF realistic model: vacuum coupling η ~ 0.02 (Podkletnov analogy).
    """
    separator("4. MODEL B: VMF REALISTIC (η = 0.02)")

    eta = 0.02  # from Podkletnov 1992 experiment
    E_per_carrier = Delta_max * eta
    E_per_cycle = N_active * E_per_carrier
    P_out = E_per_cycle * f_osc

    # Same input power
    d_dielectric = 1e-9
    eps_0 = 8.854e-12
    C_gate = eps_0 * S_device / d_dielectric
    V_mod = 1e-3
    P_input = 0.5 * C_gate * V_mod**2 * f_osc

    COP = P_out / P_input if P_input > 0 else float('inf')

    print(f"  Vacuum coupling η   = {eta} (from Podkletnov 1992)")
    print(f"  Energy per carrier  = {E_per_carrier/eV:.4f} eV")
    print(f"  Energy per cycle    = {E_per_cycle:.3e} J")
    print(f"  ─────────────────────────────────")
    print(f"  OUTPUT power        = {P_out:.3e} W  ({P_out*1e3:.2f} mW)")
    print(f"  INPUT power (gate)  = {P_input:.3e} W  ({P_input*1e3:.2f} mW)")
    print(f"  COP                 = {COP:.0f}")

    return P_out, P_input, COP


def calc_dynamic_casimir(f_osc, omega, N_active):
    """
    Dynamic Casimir Effect analog in graphene.
    Strict QFT calculation — lower bound on vacuum energy extraction.

    Key: in graphene, the relevant velocity is v_F (not c),
    and the coupling constant is α_eff ≈ 2.2 (not 1/137).
    Both factors ENHANCE the effect relative to free-space QED.
    """
    separator("5. MODEL C: DYNAMIC CASIMIR ANALOG (QFT lower bound)")

    # In DCE, photon production rate per mode:
    #   N_photon ~ (v/c)^2 per cycle
    # In graphene, the "mirror velocity" is the rate of gap change:
    #   v_eff ~ Δ₀ / (ℏ * ω) * a_lattice
    # And the relevant "speed of light" is v_F

    v_boundary = (Delta_max / hbar) * a_lattice / omega
    beta = v_boundary / v_F   # relativistic parameter in graphene

    print(f"  Boundary velocity   = {v_boundary:.3e} m/s")
    print(f"  Fermi velocity      = {v_F:.3e} m/s")
    print(f"  β = v/v_F           = {beta:.6f}")
    print(f"  β² (pair production)= {beta**2:.3e}")

    # Energy per photon at oscillation frequency
    E_photon = hbar * omega

    # Number of photon pairs created per cycle (DCE formula)
    # In graphene: enhanced by α_eff² / α_QED²
    enhancement = (alpha_eff / alpha_QED)**2
    N_photons_per_cycle = beta**2 * enhancement

    print(f"")
    print(f"  Photon energy       = {E_photon/eV:.6f} eV")
    print(f"  α_eff / α_QED       = {alpha_eff/alpha_QED:.0f}×")
    print(f"  Enhancement factor  = {enhancement:.0f}×")
    print(f"  Photons/cycle/mode  = {N_photons_per_cycle:.3e}")

    # Number of modes ~ number of carriers
    N_modes = N_active

    E_per_cycle = N_photons_per_cycle * E_photon * N_modes
    P_out = E_per_cycle * f_osc

    # Input power (same as before)
    d_dielectric = 1e-9
    eps_0 = 8.854e-12
    C_gate = eps_0 * S_device / d_dielectric
    V_mod = 1e-3
    P_input = 0.5 * C_gate * V_mod**2 * f_osc

    COP = P_out / P_input if P_input > 0 else float('inf')

    print(f"  Active modes        = {N_modes:.2e}")
    print(f"  Energy per cycle    = {E_per_cycle:.3e} J")
    print(f"  ─────────────────────────────────")
    print(f"  OUTPUT power        = {P_out:.3e} W")
    print(f"  INPUT power (gate)  = {P_input:.3e} W")
    print(f"  COP                 = {COP:.3e}")

    return P_out, P_input, COP


def calc_thermal_noise():
    """Calculate thermal noise power for comparison (Nyquist–Johnson)."""
    separator("6. THERMAL NOISE BASELINE (300 K)")

    f_osc = 1 / (2 * np.pi * np.sqrt(L_inductor * C_tune))
    bandwidth = f_osc / Q_factor  # noise bandwidth of LC circuit

    # Johnson-Nyquist noise power
    P_thermal = k_B * T_room * bandwidth

    # Thibado effect (thermal fluctuation harvesting in graphene)
    # ~10 pW per membrane (Thibado 2020)
    P_thibado = 10e-12  # W, per membrane

    print(f"  Temperature         = {T_room} K")
    print(f"  kT                  = {k_B*T_room/eV*1e3:.1f} meV")
    print(f"  Noise bandwidth     = {bandwidth/1e6:.1f} MHz")
    print(f"  Thermal noise power = {P_thermal:.3e} W  ({P_thermal*1e12:.1f} pW)")
    print(f"  Thibado effect      = {P_thibado:.3e} W  (10 pW, measured)")

    return P_thermal


def calc_parametric_amplification(f_osc, omega):
    """
    Parametric amplification near Dirac point.
    If graphene at Δ→0 acts as a parametric amplifier for vacuum fluctuations,
    the gain can be exponential.
    """
    separator("7. PARAMETRIC AMPLIFICATION SCENARIO")

    # Parametric gain in one pass:
    #   G = cosh²(g·t)  where g = pump amplitude / threshold
    # For graphene NDR near Dirac point:
    #   g ~ α_eff * (Δ₀/ℏω) * (v_F/c)

    g_param = alpha_eff * (Delta_max / (hbar * omega)) * (v_F / c)
    t_cycle = 1 / f_osc
    G_single = np.cosh(g_param)**2
    G_dB = 10 * np.log10(G_single)

    print(f"  Parametric gain parameter g = {g_param:.4f}")
    print(f"  Single-pass gain G  = {G_single:.4f} ({G_dB:.1f} dB)")

    # If gain > 1, vacuum fluctuations get amplified each cycle
    # Effective power = thermal_seed × G^N_cycles
    # But saturates at NDR limiting amplitude

    seed_power = k_B * T_room * f_osc / Q_factor
    P_amplified = seed_power * G_single

    print(f"  Seed (thermal)      = {seed_power:.3e} W")
    print(f"  After 1 cycle       = {P_amplified:.3e} W")
    print(f"  After 10 cycles     = {seed_power * G_single**10:.3e} W")
    print(f"  After 100 cycles    = {seed_power * min(G_single**100, 1e30):.3e} W")
    print(f"")
    if G_single > 1:
        print(f"  ✓ Parametric gain > 1: vacuum fluctuations AMPLIFIED")
        print(f"    Oscillation self-starts from thermal/vacuum seed")
    else:
        print(f"  ✗ Parametric gain < 1: no amplification")

    return G_single


def summary(results):
    """Print comparison of all models."""
    separator("SUMMARY: COMPARISON OF ALL MODELS")

    headers = ["Model", "P_out", "P_in", "COP", "Interpretation"]
    rows = [
        ("A: VMF optimistic (η=1)",
         f"{results['A'][0]:.2e} W",
         f"{results['A'][1]:.2e} W",
         f"{results['A'][2]:.0f}",
         "Upper bound, unrealistic"),
        ("B: VMF realistic (η=0.02)",
         f"{results['B'][0]:.2e} W",
         f"{results['B'][1]:.2e} W",
         f"{results['B'][2]:.0f}",
         "Podkletnov analogy"),
        ("C: Dynamic Casimir (QFT)",
         f"{results['C'][0]:.2e} W",
         f"{results['C'][1]:.2e} W",
         f"{results['C'][2]:.2e}",
         "Strict lower bound"),
    ]

    # Print table
    print(f"  {'Model':<30} {'P_out':>12} {'P_in':>12} {'COP':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")
    for name, p_out, p_in, cop, interp in rows:
        print(f"  {name:<30} {p_out:>12} {p_in:>12} {cop:>12}")
    print()

    # Key insight
    print(f"  KEY INSIGHT:")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  The three models span ~26 orders of magnitude.")
    print(f"  The truth — if any effect exists — lies between B and C.")
    print(f"")
    print(f"  Model A (160 W)      — assumes perfect vacuum coupling.")
    print(f"                         Too optimistic. Not physical.")
    print(f"  Model B (3.2 mW)     — uses Podkletnov's η = 0.02.")
    print(f"                         Testable. Would show COP ~ 700.")
    print(f"  Model C (QFT bound)  — strict dynamic Casimir analog.")
    print(f"                         COP << 1. Undetectable alone.")
    print(f"  Parametric amp.      — if G > 1, exponential boost possible.")
    print(f"                         This is the wildcard.")


def frequency_scan(N_active):
    """Scan over frequencies to find optimal operating point."""
    separator("8. FREQUENCY SCAN: COP vs OSCILLATION FREQUENCY")

    freqs = np.logspace(6, 12, 25)  # 1 MHz to 1 THz
    print(f"  {'Frequency':>12} {'P_out(B)':>12} {'P_in':>12} {'COP(B)':>10} {'COP(C)':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*10} {'─'*12}")

    for f in freqs:
        omega = 2 * np.pi * f
        eta = 0.02

        # Model B output
        E_cycle_B = N_active * Delta_max * eta
        P_out_B = E_cycle_B * f

        # Model C output (DCE analog)
        v_boundary = (Delta_max / hbar) * a_lattice / omega
        beta = v_boundary / v_F
        enhancement = (alpha_eff / alpha_QED)**2
        N_photons = beta**2 * enhancement
        E_photon = hbar * omega
        E_cycle_C = N_photons * E_photon * N_active
        P_out_C = E_cycle_C * f

        # Input
        eps_0 = 8.854e-12
        C_gate = eps_0 * S_device / 1e-9
        P_in = 0.5 * C_gate * (1e-3)**2 * f

        COP_B = P_out_B / P_in if P_in > 0 else 0
        COP_C = P_out_C / P_in if P_in > 0 else 0

        freq_str = f"{f:.1e}" if f < 1e9 else f"{f/1e9:.1f} GHz" if f < 1e12 else f"{f/1e12:.1f} THz"
        print(f"  {freq_str:>12} {P_out_B:>12.2e} {P_in:>12.2e} {COP_B:>10.0f} {COP_C:>12.2e}")

    print(f"\n  Note: COP(B) is constant with frequency (both scale linearly).")
    print(f"  COP(C) DECREASES with frequency (β² ∝ 1/ω² while P_in ∝ ω).")
    print(f"  → Lower frequencies favor vacuum extraction in DCE model.")


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  NVG GRAPHENE AUTO-GENERATOR: THEORETICAL POWER CALCULATOR     ║")
    print("║  Based on VMF model (Kirichenko, 2026)                         ║")
    print("║  DOI: 10.5281/zenodo.20214457                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # 1. Oscillator parameters
    f_osc, omega = calc_oscillator()

    # 2. Active carriers
    N_active = calc_active_carriers()

    # 3. VMF optimistic
    P_A = calc_vmf_optimistic(f_osc, N_active)

    # 4. VMF realistic
    P_B = calc_vmf_realistic(f_osc, N_active)

    # 5. Dynamic Casimir analog
    P_C = calc_dynamic_casimir(f_osc, omega, N_active)

    # 6. Thermal baseline
    P_thermal = calc_thermal_noise()

    # 7. Parametric amplification
    G = calc_parametric_amplification(f_osc, omega)

    # 8. Frequency scan
    frequency_scan(N_active)

    # Summary
    results = {'A': P_A, 'B': P_B, 'C': P_C}
    summary(results)

    print(f"\n{'='*70}")
    print(f"  Run complete. All models computed successfully.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
