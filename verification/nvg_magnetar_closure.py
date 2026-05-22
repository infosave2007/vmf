#!/usr/bin/env python3
"""
NVG Magnetar Closure Checks

This script addresses the main open points of the NVG magnetar note with a
single quantitative pass:

1. Dense-matter EM bridge:
   Reuses the existing VMF melting law to estimate a conservative quark-loop
   dielectric correction and a benchmark gauge-kinetic amplification corridor.
2. Energetics:
   Compares rotational energy at birth against the supernova-remnant tension.
3. Magnetar viability:
   Computes the structural amplification required to reach magnetar fields from
   plausible fossil seeds without millisecond birth.
4. Long-period benchmark:
   Integrates dipole spin-down for a 1E 161348-5055-like system.

The goal is not to claim a first-principles proof of mu_eff collapse, but to
quantify exactly what amplification corridor is needed and which parts are
already consistent with existing NVG dense-matter benchmarks.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Core NVG / VMF dense-matter parameters
# ---------------------------------------------------------------------------
M_OMEGA_0 = 859.0  # MeV
N_0 = 0.16  # fm^-3
KAPPA_1 = 0.25
KAPPA_2 = 0.80


def m_omega_star(n_b: float) -> float:
    x = max(n_b / N_0, 0.0)
    return M_OMEGA_0 * (1.0 + KAPPA_2 * x) ** (-KAPPA_1 / KAPPA_2)


def melted_fraction(n_b: float) -> float:
    return 1.0 - m_omega_star(n_b) / M_OMEGA_0


# ---------------------------------------------------------------------------
# 1. Dense-matter EM bridge: conservative loop response vs benchmark coupling
# ---------------------------------------------------------------------------
ALPHA_EM = 1.0 / 137.036
M_U = 2.2
M_D = 4.7
M_S = 95.0


def constituent_mass(n_b: float, current_mass: float) -> float:
    return current_mass + m_omega_star(n_b) / 3.0


def delta_alpha_over_alpha(n_b: float) -> float:
    delta = 0.0
    species = [
        (2.0 / 3.0, M_U),
        (-1.0 / 3.0, M_D),
        (-1.0 / 3.0, M_S),
    ]
    for charge, current_mass in species:
        m_med = constituent_mass(n_b, current_mass)
        m_vac = constituent_mass(0.0, current_mass)
        delta += -(2.0 / (3.0 * math.pi)) * 3.0 * charge**2 * math.log(m_med / m_vac)
    return delta


def eps_eff_ratio_loop(n_b: float) -> float:
    return 1.0 / (1.0 + delta_alpha_over_alpha(n_b))


def gamma_struct_benchmark(n_b: float, coupling_strength: float) -> float:
    """
    Benchmark structural amplification from a gauge-kinetic coupling.

    Z_EM(W) is taken to vary exponentially with the melted VMF fraction.
    If B scales as 1/Z_EM across the dense core, this produces a simple
    amplification corridor. The parameter `coupling_strength` is dimensionless
    and encodes the unknown matching between the EFT order parameter and the
    effective constitutive response.
    """
    f_melt = melted_fraction(n_b)
    return math.exp(coupling_strength * f_melt)


# ---------------------------------------------------------------------------
# 2. Stellar magnetic-field bookkeeping
# ---------------------------------------------------------------------------
R_PROG = 3.0e8  # cm, iron-core scale
R_NS = 1.2e6  # cm, NS radius ~12 km
R_STAR = 1.0e11  # cm, magnetized massive-star radius scale


def flux_freezing_gain(r_prog: float = R_PROG, r_ns: float = R_NS) -> float:
    return (r_prog / r_ns) ** 2


def progenitor_surface_to_core_field(surface_field: float, r_star: float = R_STAR, r_core: float = R_PROG) -> float:
    return surface_field * (r_star / r_core) ** 2


def final_field(seed_field: float, gamma_struct: float) -> float:
    return seed_field * flux_freezing_gain() * gamma_struct


def required_gamma_struct(seed_field: float, target_field: float) -> float:
    return target_field / (seed_field * flux_freezing_gain())


# ---------------------------------------------------------------------------
# 3. Rotational energy and supernova-remnant tension
# ---------------------------------------------------------------------------
M_NS = 1.4 * 1.98847e33  # g
RADIUS_NS = 1.2e6  # cm
I_NS = 0.35 * M_NS * RADIUS_NS**2  # g cm^2


def rotational_energy(period_s: float) -> float:
    omega = 2.0 * math.pi / period_s
    return 0.5 * I_NS * omega**2


# ---------------------------------------------------------------------------
# 4. Dipole spin-down evolution
# ---------------------------------------------------------------------------
C_LIGHT = 2.99792458e10  # cm/s


def dipole_k_constant(field_g: float, radius_cm: float = RADIUS_NS, inertia: float = I_NS) -> float:
    return (8.0 * math.pi**2 * field_g**2 * radius_cm**6) / (3.0 * inertia * C_LIGHT**3)


def period_after_time(period0_s: float, field_g: float, time_s: float) -> float:
    k_const = dipole_k_constant(field_g)
    return math.sqrt(period0_s**2 + 2.0 * k_const * time_s)


def field_for_target_period(period0_s: float, target_period_s: float, time_s: float) -> float:
    numerator = (target_period_s**2 - period0_s**2) * 3.0 * I_NS * C_LIGHT**3
    denominator = 16.0 * math.pi**2 * RADIUS_NS**6 * time_s
    return math.sqrt(max(numerator / denominator, 0.0))


def characteristic_age(period_s: float, period_dot: float) -> float:
    return period_s / (2.0 * period_dot)


def dipole_period_dot(period_s: float, field_g: float) -> float:
    return dipole_k_constant(field_g) / period_s


def gauss_label(value: float) -> str:
    return f"{value:.2e} G"


def erg_label(value: float) -> str:
    return f"{value:.2e} erg"


def year_label(seconds: float) -> str:
    return f"{seconds / (365.25 * 24 * 3600):.2e} yr"


def main() -> None:
    print("=" * 84)
    print("NVG MAGNETAR CLOSURE CHECKS")
    print("=" * 84)

    print("\n1. DENSE-MATTER EM BRIDGE")
    print("-" * 84)
    print(f"{'n_B/n_0':>8} | {'melted frac':>11} | {'eps_eff(loop)':>13} | {'Gamma(c=1)':>10} | {'Gamma(c=2)':>10} | {'Gamma(c=3)':>10}")
    print("-" * 84)
    for ratio in [1.0, 2.0, 3.0, 5.0]:
        n_b = ratio * N_0
        melt = melted_fraction(n_b)
        eps_loop = eps_eff_ratio_loop(n_b)
        gamma_1 = gamma_struct_benchmark(n_b, 1.0)
        gamma_2 = gamma_struct_benchmark(n_b, 2.0)
        gamma_3 = gamma_struct_benchmark(n_b, 3.0)
        print(f"{ratio:8.1f} | {melt:11.3f} | {eps_loop:13.5f} | {gamma_1:10.3f} | {gamma_2:10.3f} | {gamma_3:10.3f}")

    print("""
Interpretation:
- The conservative quark-loop route gives a moderate O(10%) change in the
    effective dielectric response across the 1-5 n_0 corridor.
- A sizable structural amplification requires an additional EFT matching between
  the VMF order parameter and the EM constitutive response.
- The unknown is therefore not whether dense matter changes the EM sector at all,
  but how strongly that change feeds into the magnetic constitutive law.
""")

    print("\n2. REQUIRED STRUCTURAL AMPLIFICATION")
    print("-" * 84)
    gamma_flux = flux_freezing_gain()
    print(f"Flux-freezing gain from R_prog={R_PROG:.2e} cm to R_NS={R_NS:.2e} cm: {gamma_flux:.3e}")
    print(f"{'B_surface':>12} | {'B_core before collapse':>24} | {'B_NS from flux only':>20}")
    print("-" * 66)
    for surface_field in [1.0e2, 1.0e3, 3.0e3, 1.0e4]:
        b_core = progenitor_surface_to_core_field(surface_field)
        b_ns_flux = final_field(b_core, 1.0)
        print(f"{surface_field:12.2e} | {b_core:24.3e} | {b_ns_flux:20.3e}")

    print()
    seeds = [1.0e8, 3.0e8, 1.0e9, 3.0e9, 1.0e10]
    targets = [1.0e14, 3.0e14, 1.0e15]
    print(f"{'B_seed':>12} | {'B_target':>12} | {'Gamma_struct req.':>18}")
    print("-" * 50)
    for seed in seeds:
        for target in targets:
            gamma_req = required_gamma_struct(seed, target)
            print(f"{seed:12.2e} | {target:12.2e} | {gamma_req:18.3f}")

    print("""
Interpretation:
- If one starts from surface fossil fields on the progenitor, the pre-collapse
    iron-core field is already much larger after geometric concentration.
- For iron-core seed fields in the 10^8-10^10 G corridor, the extra structural
    amplification needed to reach magnetar strengths drops to order unity to
    several tens rather than many orders of magnitude.
- This is the narrowest quantitative version of the NVG claim.
""")

    print("\n3. ROTATIONAL ENERGY BUDGET")
    print("-" * 84)
    print(f"{'Birth period':>12} | {'E_rot':>16} | {'Relative to 1e51 erg':>24}")
    print("-" * 60)
    for period_ms in [1.0, 2.0, 3.0, 5.0, 10.0, 100.0, 1000.0]:
        period_s = period_ms * 1.0e-3
        e_rot = rotational_energy(period_s)
        print(f"{period_ms:12.1f} | {e_rot:16.3e} | {e_rot / 1.0e51:24.2f}")

    print("""
Interpretation:
- Millisecond birth periods imply rotational reservoirs near 10^52 erg.
- Birth periods in the 10-1000 ms corridor reduce the injected rotational energy
  by one to six orders of magnitude, which is precisely why a non-dynamo field
  amplification channel is attractive from the remnant-energetics perspective.
""")

    print("\n4. 1E 161348-5055-LIKE LONG-PERIOD CHECK")
    print("-" * 84)
    target_period = 6.67 * 3600.0  # s
    ages_yr = [2_000.0, 5_000.0, 10_000.0]
    birth_periods = [0.01, 0.1, 1.0, 5.0]  # s
    print(f"Target period: {target_period:.2e} s")
    print(f"{'Age (yr)':>10} | {'P0 (s)':>8} | {'B required':>14} | {'Pdot now':>12} | {'tau_c':>12}")
    print("-" * 72)
    for age_yr in ages_yr:
        age_s = age_yr * 365.25 * 24.0 * 3600.0
        for p0 in birth_periods:
            b_req = field_for_target_period(p0, target_period, age_s)
            pdot = dipole_period_dot(target_period, b_req)
            tau = characteristic_age(target_period, pdot)
            print(f"{age_yr:10.0f} | {p0:8.2f} | {b_req:14.3e} | {pdot:12.3e} | {tau / (365.25 * 24 * 3600):12.2e}")

    print("""
Interpretation:
- Pure vacuum dipole braking to 6.67 hr within 2-10 kyr requires ultra-strong
  effective dipole fields of order 10^17-10^18 G.
- Therefore, a rotation-independent field-amplification channel may help with
  field generation, but it does not by itself solve the extreme long-period case.
- For 1E 161348-5055, additional torques such as fallback-disk braking,
  propeller interaction, or early-time environmental coupling remain necessary.
""")

    print("\n5. SYNTHESIS")
    print("-" * 84)
    n_b = 5.0 * N_0
    gamma2 = gamma_struct_benchmark(n_b, 2.0)
    gamma3 = gamma_struct_benchmark(n_b, 3.0)
    sample_seed = 1.0e9
    print(f"At n_B = 5 n_0, melted fraction = {melted_fraction(n_b):.3f}.")
    print(f"A benchmark coupling corridor gives Gamma_struct ~ {gamma2:.2f} to {gamma3:.2f}.")
    print(
        f"For a fossil core field of {sample_seed:.1e} G, this implies final fields "
        f"B_NS ~ {gauss_label(final_field(sample_seed, gamma2))} to {gauss_label(final_field(sample_seed, gamma3))}."
    )
    print("""
Bottom line:
- The disputed step is now quantitatively isolated: the needed structural
  amplification is modest if the progenitor fossil field is already strong.
- The supernova-remnant energetics argument strongly favors avoiding obligatory
  millisecond birth.
- The long-period magnetar case still needs extra torque physics on top of any
  NVG structural amplification.
""")


if __name__ == "__main__":
    main()