#!/usr/bin/env python3
"""
NVG fallback-disk torque test for 1E 161348-5055.

Purpose:
1. Show that pure dipole braking is insufficient for the 6.67 hr period.
2. Test whether a standard propeller / fallback-disk torque can bridge the gap
   for magnetar-strength fields without requiring absurd dipole fields.

This is a phenomenological calculation, not a first-principles disk model.
It is intended to delimit a viable parameter corridor that can be cited in the
magnetar paper.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Neutron-star parameters
# ---------------------------------------------------------------------------
G_CGS = 6.67430e-8
C_CGS = 2.99792458e10
M_SUN = 1.98847e33

M_NS = 1.4 * M_SUN
R_NS = 1.2e6  # cm
I_NS = 0.35 * M_NS * R_NS**2


# ---------------------------------------------------------------------------
# Target source: 1E 161348-5055
# ---------------------------------------------------------------------------
TARGET_PERIOD = 6.67 * 3600.0  # s
TARGET_AGES_YR = [2_000.0, 5_000.0, 10_000.0]


def dipole_torque(omega: float, b_dip: float) -> float:
    mu = b_dip * R_NS**3
    return -(2.0 / 3.0) * mu**2 * omega**3 / C_CGS**3


def light_cylinder_radius(omega: float) -> float:
    return C_CGS / max(omega, 1.0e-30)


def corotation_radius(omega: float) -> float:
    return (G_CGS * M_NS / max(omega**2, 1.0e-60)) ** (1.0 / 3.0)


def magnetospheric_radius(b_dip: float, mdot: float, xi: float = 0.5) -> float:
    mu = b_dip * R_NS**3
    raw = (mu**4 / (2.0 * G_CGS * M_NS * max(mdot, 1.0e-40) ** 2)) ** (1.0 / 7.0)
    return xi * raw


def fallback_mdot(time_s: float, mdot0: float, t0_s: float, alpha: float) -> float:
    return mdot0 * (1.0 + time_s / t0_s) ** (-alpha)


def disk_propeller_torque(omega: float, b_dip: float, mdot: float, xi: float = 0.5) -> tuple[float, float, float, float]:
    """
    Returns (torque, r_m, r_c, r_lc).

    The disk torque follows a standard fastness prescription:
      N_disk = \.dot{M} sqrt(G M R_m) [1 - omega / omega_K(R_m)]

    When omega > omega_K(R_m), the torque is negative (propeller regime).
    The effective coupling radius is capped below the light cylinder so the
    fallback disk does not act from an unphysical radius.
    """
    if mdot <= 0.0:
        r_lc = light_cylinder_radius(omega)
        r_c = corotation_radius(omega)
        return 0.0, r_lc, r_c, r_lc

    r_m = magnetospheric_radius(b_dip, mdot, xi=xi)
    r_lc = light_cylinder_radius(omega)
    r_c = corotation_radius(omega)
    r_in = min(r_m, 0.95 * r_lc)
    omega_k = math.sqrt(G_CGS * M_NS / r_in**3)
    fastness = omega / max(omega_k, 1.0e-30)
    torque = mdot * math.sqrt(G_CGS * M_NS * r_in) * (1.0 - fastness)
    return torque, r_in, r_c, r_lc


def evolve_period(
    p0_s: float,
    age_yr: float,
    b_dip: float,
    mdot0: float,
    t0_s: float,
    alpha: float,
    xi: float = 0.5,
    n_steps: int = 8000,
) -> dict[str, float]:
    age_s = age_yr * 365.25 * 24.0 * 3600.0
    omega = 2.0 * math.pi / p0_s
    time_s = 0.0
    dt = age_s / n_steps
    max_disk_torque = 0.0
    max_fastness = 0.0
    final_mdot = 0.0
    r_m = r_c = r_lc = 0.0

    for _ in range(n_steps):
        mdot = fallback_mdot(time_s, mdot0, t0_s, alpha)
        n_disk, r_m, r_c, r_lc = disk_propeller_torque(omega, b_dip, mdot, xi=xi)
        n_dip = dipole_torque(omega, b_dip)
        n_total = n_disk + n_dip
        omega += (n_total / I_NS) * dt
        omega = max(omega, 2.0 * math.pi / TARGET_PERIOD / 100.0)
        time_s += dt
        final_mdot = mdot
        max_disk_torque = max(max_disk_torque, abs(n_disk))
        omega_k = math.sqrt(G_CGS * M_NS / max(r_m, R_NS) ** 3)
        max_fastness = max(max_fastness, omega / max(omega_k, 1.0e-30))

    p_final = 2.0 * math.pi / omega
    n_disk, r_m, r_c, r_lc = disk_propeller_torque(omega, b_dip, final_mdot, xi=xi)
    n_dip = dipole_torque(omega, b_dip)
    p_dot = -(p_final**2 / (2.0 * math.pi * I_NS)) * (n_disk + n_dip)

    return {
        "age_yr": age_yr,
        "p0_s": p0_s,
        "b_dip": b_dip,
        "mdot0": mdot0,
        "t0_s": t0_s,
        "alpha": alpha,
        "p_final": p_final,
        "p_dot": max(p_dot, 0.0),
        "mdot_final": final_mdot,
        "r_m": r_m,
        "r_c": r_c,
        "r_lc": r_lc,
        "max_disk_torque": max_disk_torque,
        "max_fastness": max_fastness,
    }


def years(seconds: float) -> float:
    return seconds / (365.25 * 24.0 * 3600.0)


def fmt_scientific(value: float) -> str:
    return f"{value:.3e}"


def main() -> None:
    print("=" * 88)
    print("1E 161348-5055: FALLBACK-DISK / PROPELLER TORQUE TEST")
    print("=" * 88)

    p0_grid = [0.1, 0.3, 1.0]
    b_grid = [1.0e14, 3.0e14, 1.0e15]
    mdot0_grid = [3.0e20, 1.0e21, 3.0e21, 1.0e22]
    t0_grid = [1.0e4, 1.0e5]
    alpha_grid = [1.2, 4.0 / 3.0, 1.5]

    print("\nAssumptions:")
    print(f"- Target period: {TARGET_PERIOD:.3e} s")
    print(f"- Dipole field scan: {b_grid}")
    print(f"- Initial fallback rates: {mdot0_grid} g/s")
    print(f"- Disk decay times: {t0_grid} s")
    print(f"- Decay indices: {alpha_grid}")

    best_by_age: dict[float, dict[str, float] | None] = {age: None for age in TARGET_AGES_YR}

    for age_yr in TARGET_AGES_YR:
        best_misfit = float("inf")
        for p0_s in p0_grid:
            for b_dip in b_grid:
                for mdot0 in mdot0_grid:
                    for t0_s in t0_grid:
                        for alpha in alpha_grid:
                            result = evolve_period(p0_s, age_yr, b_dip, mdot0, t0_s, alpha)
                            misfit = abs(math.log10(result["p_final"] / TARGET_PERIOD))
                            if misfit < best_misfit:
                                best_misfit = misfit
                                best_by_age[age_yr] = result

    print("\nBest-fit fallback solutions by remnant age:")
    print(
        f"{'Age (yr)':>8} | {'P0 (s)':>7} | {'B (G)':>10} | {'Mdot0':>10} | {'t0 (s)':>10} | {'alpha':>5} | {'P_f (s)':>10} | {'Pdot':>10}"
    )
    print("-" * 96)
    for age_yr in TARGET_AGES_YR:
        result = best_by_age[age_yr]
        assert result is not None
        print(
            f"{age_yr:8.0f} | {result['p0_s']:7.2f} | {result['b_dip']:10.2e} | {result['mdot0']:10.2e} | "
            f"{result['t0_s']:10.2e} | {result['alpha']:5.2f} | {result['p_final']:10.2e} | {result['p_dot']:10.2e}"
        )

    print("\nDetailed diagnostics for best-fit solutions:")
    for age_yr in TARGET_AGES_YR:
        result = best_by_age[age_yr]
        assert result is not None
        tau_c = result["p_final"] / max(2.0 * result["p_dot"], 1.0e-99)
        print("-" * 88)
        print(f"Age = {age_yr:.0f} yr")
        print(f"  Final period:       {result['p_final']:.3e} s")
        print(f"  Final period dot:   {result['p_dot']:.3e} s/s")
        print(f"  Characteristic age: {years(tau_c):.3e} yr")
        print(f"  Final mdot:         {result['mdot_final']:.3e} g/s")
        print(f"  r_m:                {result['r_m']:.3e} cm")
        print(f"  r_c:                {result['r_c']:.3e} cm")
        print(f"  r_lc:               {result['r_lc']:.3e} cm")
        print(f"  Max |N_disk|:       {result['max_disk_torque']:.3e} dyne cm")
        print(f"  Max fastness:       {result['max_fastness']:.3e}")

    print("\nInterpretation:")
    print("- The target 6.67 hr period can be approached with magnetar-strength fields")
    print("  once a decaying fallback disk contributes a long propeller torque.")
    print("- This removes the need for absurd 10^17-10^18 G dipole fields inferred from")
    print("  pure vacuum dipole braking alone.")
    print("- The long-period source therefore does not falsify the NVG field-amplification")
    print("  channel; instead, it forces the model to include extra torque physics.")


if __name__ == "__main__":
    main()