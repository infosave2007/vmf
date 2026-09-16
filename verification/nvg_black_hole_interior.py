#!/usr/bin/env python3
"""Thermodynamic audit of a phenomenological density-dependent mass ansatz.

This homogeneous, zero-temperature quasiparticle model uses one energy
e(n) = Fermi(n, m(n)). It is not a source-complete NVG action, a collapse
solution, or a measured decomposition of the nucleon mass. In particular,
ultrarelativistic matter (P/e -> 1/3) is not de Sitter matter (P/e = -1).
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence

import numpy as np
from decimal import Decimal, localcontext


hbar_c = 197.3269804  # MeV fm
M_N = 939.0  # MeV
n_0 = 0.16  # fm^-3

# Retained phenomenological inputs. Their sum is an ansatz, not a measured
# current/vacuum mass split; no observational fit is performed here.
sigma_piN = 44.0
sigma_sN = 30.0
sigma_heavy = 6.0
sigma_total = sigma_piN + sigma_sN + sigma_heavy
M_Omega_0 = M_N - sigma_total
M_current_0 = sigma_total
f_Omega_0 = M_Omega_0 / M_N
kappa_1 = 0.25
kappa_2 = 0.80
FERMION_DEGENERACY = 2  # One species with two spin states.

G_SI = 6.674e-11
C_SI = 2.998e8
M_SUN_KG = 1.989e30
MEV_FM3_TO_PA = 1.602e32

_CURRENT_SLOPE = sigma_piN * hbar_c**3 / (93.0**2 * 140.0**2)
CURRENT_MASS_KINK_DENSITY = 1.0 / _CURRENT_SLOPE
_nodes, _weights = np.polynomial.legendre.leggauss(128)
_Q = (_nodes + 1.0) / 2.0
_WEIGHTS = _weights / 2.0


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive_finite(value: float, name: str) -> float:
    value = _nonnegative_finite(value, name)
    if value == 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _at_current_mass_kink(n_B: float) -> bool:
    return math.isclose(n_B, CURRENT_MASS_KINK_DENSITY, rel_tol=4e-15)


def M_Omega(n_B: float) -> float:
    """Smooth component of the retained mass ansatz, in MeV."""
    n_B = _nonnegative_finite(n_B, "n_B")
    return M_Omega_0 * (1.0 + kappa_2 * n_B / n_0) ** (-kappa_1 / kappa_2)


def M_current(n_B: float) -> float:
    """Clipped phenomenological component; continuous but kinked at n_kink."""
    n_B = _nonnegative_finite(n_B, "n_B")
    return M_current_0 * max(1.0 - _CURRENT_SLOPE * n_B, 0.0)


def M_star(n_B: float) -> float:
    return M_current(n_B) + M_Omega(n_B)


def f_Omega(n_B: float) -> float:
    return M_Omega(n_B) / M_star(n_B)


def mass_derivatives(n_B: float) -> tuple[float, float, float]:
    """Return m, dm/dn, d2m/dn2; refuse the nondifferentiable clipping kink."""
    n_B = _nonnegative_finite(n_B, "n_B")
    if _at_current_mass_kink(n_B):
        raise ValueError("M_current has a nondifferentiable kink at this density")
    a = 1.0 + kappa_2 * n_B / n_0
    omega = M_Omega(n_B)
    first = -kappa_1 * omega / (n_0 * a)
    second = kappa_1 * (kappa_1 + kappa_2) * omega / (n_0 * a) ** 2
    if n_B < CURRENT_MASS_KINK_DENSITY:
        first -= M_current_0 * _CURRENT_SLOPE
    return M_star(n_B), first, second


def fermi_integrals(
    n_B: float, mass_mev: float, degeneracy: int = FERMION_DEGENERACY
) -> dict[str, float]:
    """Stable Fermi integrals, using n = g*kF^3/(6*pi^2*(hbar*c)^3).

    Return energy, kinetic pressure, partial_e/partial_m, and
    partial2_e/partial_m2 in MeV/fm^3, MeV/fm^3, fm^-3, and fm^-3/MeV.
    Quadrature on p/kF avoids subtraction of nearly equal massive terms.
    The exact massless branch uses the same degeneracy normalization.
    """
    n_B = _nonnegative_finite(n_B, "n_B")
    mass_mev = _nonnegative_finite(mass_mev, "mass_mev")
    if not isinstance(degeneracy, int) or isinstance(degeneracy, bool) or degeneracy <= 0:
        raise ValueError("degeneracy must be a positive integer")
    kf = (6.0 * math.pi**2 / degeneracy) ** (1.0 / 3.0) * n_B ** (1.0 / 3.0) * hbar_c
    # The scalar boundary energy propagates into cancellation-sensitive EOS
    # derivatives. Both legacy math.hypot and NumPy can differ by one ULP on
    # this domain. Compute it from exact binary inputs with guard digits;
    # independent mpmath tests check rounding without reading saved outputs.
    with localcontext() as context:
        context.prec = 80
        k_exact, m_exact = Decimal.from_float(kf), Decimal.from_float(mass_mev)
        ef = float((k_exact*k_exact + m_exact*m_exact).sqrt())
    if n_B == 0.0:
        energy = pressure = scalar = curvature = 0.0
    elif mass_mev == 0.0:
        energy = 0.75 * n_B * kf
        pressure = energy / 3.0
        scalar = 0.0
        curvature = 1.5 * n_B / kf
    else:
        momenta = kf * _Q
        energies = np.hypot(momenta, mass_mev)
        energy = 3.0 * n_B * float(np.dot(_WEIGHTS, _Q**2 * energies))
        pressure = n_B * kf**2 * float(np.dot(_WEIGHTS, _Q**4 / energies))
        scalar = 3.0 * n_B * float(np.dot(_WEIGHTS, _Q**2 * (mass_mev / energies)))
        curvature = 3.0 * n_B * float(np.dot(_WEIGHTS, _Q**2 * (momenta / energies) ** 2 / energies))
    result = {
        "kf_mev": kf,
        "fermi_energy_mev": ef,
        "energy_mev_fm3": energy,
        "kinetic_pressure_mev_fm3": pressure,
        "scalar_density_fm3": scalar,
        "mass_curvature_fm3_per_mev": curvature,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Fermi integrals exceed the finite floating-point range")
    return result


def energy_density(n_B: float) -> float:
    """The single energy functional, including at the continuous mass kink."""
    return fermi_integrals(n_B, M_star(n_B))["energy_mev_fm3"]


def quasiparticle_eos(
    n_B: float,
    mass_mev: float,
    mass_prime: float,
    mass_second: float,
    degeneracy: int = FERMION_DEGENERACY,
) -> dict[str, object]:
    """EOS for a specified smooth local mass law, with its rearrangement.

    mu = EF + e_m*m'; e'' = kF^2/(3*n*EF) + 2*m*m'/EF
                            + e_mm*(m')^2 + e_m*m''.
    P = n*mu-e is evaluated as P_kinetic+n*e_m*m' to avoid cancellation.
    cs2 = n*e''/mu is never clipped. Vacuum entries use one-sided limits;
    e'' itself diverges there and is represented by None, not a finite value.
    """
    n_B = _nonnegative_finite(n_B, "n_B")
    mass_mev = _nonnegative_finite(mass_mev, "mass_mev")
    mass_prime, mass_second = float(mass_prime), float(mass_second)
    if not math.isfinite(mass_prime) or not math.isfinite(mass_second):
        raise ValueError("mass derivatives must be finite")
    integrals = fermi_integrals(n_B, mass_mev, degeneracy)
    ef = integrals["fermi_energy_mev"]
    scalar = integrals["scalar_density_fm3"]
    mu_rearrangement = scalar * mass_prime
    mu = ef + mu_rearrangement
    pressure_rearrangement = n_B * mu_rearrangement
    pressure = integrals["kinetic_pressure_mev_fm3"] + pressure_rearrangement
    if n_B == 0.0:
        second = None
        cs2 = 0.0 if mass_mev > 0.0 else 1.0 / 3.0
        status = "vacuum_one_sided_limit"
    else:
        # Multiplication returns inf instead of the raw OverflowError from
        # m'**2. If only the square overflows, apply its small coefficient
        # first so a representable final curvature term is retained.
        prime_squared = mass_prime * mass_prime
        curvature_term = integrals["mass_curvature_fm3_per_mev"] * prime_squared
        if not math.isfinite(prime_squared):
            curvature_term = (integrals["mass_curvature_fm3_per_mev"] * mass_prime) * mass_prime
        second = (
            (integrals["kf_mev"] / ef) * (integrals["kf_mev"] / (3.0 * n_B))
            + 2.0 * (mass_mev / ef) * mass_prime
            + curvature_term
            + scalar * mass_second
        )
        if mu == 0.0:
            cs2 = None
        else:
            numerator = n_B * second
            # Preserve ordinary arithmetic, but divide first if n*e'' alone
            # overflows. This is an equivalent evaluation, never clipping.
            cs2 = (numerator / mu if math.isfinite(numerator)
                   else n_B * (second / mu))
        if mu <= 0.0:
            status = "nonpositive_chemical_potential"
        elif cs2 < 0.0:
            status = "mechanically_unstable"
        elif cs2 > 1.0:
            status = "superluminal"
        else:
            status = "locally_stable_and_subluminal"
    result = {
        **integrals,
        "density_fm3": n_B,
        "mass_mev": mass_mev,
        "mass_prime_mev_fm3": mass_prime,
        "mass_second_mev_fm6": mass_second,
        "chemical_potential_mev": mu,
        "chemical_rearrangement_mev": mu_rearrangement,
        "pressure_rearrangement_mev_fm3": pressure_rearrangement,
        "pressure_mev_fm3": pressure,
        "energy_second_mev_fm3": second,
        "cs2": cs2,
        "pressure_over_energy": pressure / integrals["energy_mev_fm3"] if integrals["energy_mev_fm3"] > 0.0 else None,
        "closure_residual_mev_fm3": integrals["energy_mev_fm3"] + pressure - n_B * mu,
        "status": status,
    }
    for name, value in result.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"derived EOS field {name} exceeds the finite floating-point range")
    if n_B > 0.0 and integrals["energy_mev_fm3"] == 0.0:
        raise ValueError("positive-density energy underflows the floating-point range")
    return result


def eos_state(n_B: float) -> dict[str, object]:
    """Thermodynamics of the retained mass ansatz; undefined at its kink."""
    return quasiparticle_eos(n_B, *mass_derivatives(n_B))


def conformal_EOS(n_B: float) -> tuple[float, float, float]:
    """Legacy name for the full quasiparticle EOS, not an imposed conformal EOS."""
    state = eos_state(n_B)
    if state["cs2"] is None:
        raise ValueError("sound speed is undefined where chemical potential vanishes")
    return state["energy_mev_fm3"], state["pressure_mev_fm3"], state["cs2"]


def epsilon_density_scale() -> float:
    """Dimensional scale M_Omega_0^4/(hbar*c)^3, not a density bound."""
    return M_Omega_0**4 / hbar_c**3


def epsilon_max_estimate() -> float:
    """Backward-compatible alias for epsilon_density_scale(); NOT a maximum."""
    return epsilon_density_scale()


def de_sitter_core_radius(
    M_bh_solar: float, *, model: str | None = None,
    assumed_core_density: float | None = None,
) -> float:
    """Optional Hayward crossover radius in km, not a derived collapse core.

    Requires model='assumed_hayward'. For f=1-Rs*r^2/(r^3+Rs*l^2),
    l^2=3*c^4/(8*pi*G*epsilon_core) and r0=(Rs*l^2)^(1/3).
    Omitting epsilon_core explicitly adopts epsilon_density_scale() as an
    illustrative assumption. Neither this density nor the metric follows
    from the quasiparticle EOS. r0 is a crossover, not a horizon radius.
    """
    mass = _positive_finite(M_bh_solar, "M_bh_solar")
    if model != "assumed_hayward":
        raise ValueError("core radius requires the explicit model='assumed_hayward' assumption")
    density = epsilon_density_scale() if assumed_core_density is None else assumed_core_density
    density = _positive_finite(density, "assumed_core_density")
    rs_m = 2.0 * G_SI * mass * M_SUN_KG / C_SI**2
    length_squared = 3.0 * C_SI**4 / (8.0 * math.pi * G_SI * density * MEV_FM3_TO_PA)
    return (rs_m * length_squared) ** (1.0 / 3.0) / 1000.0


def compute_state(
    density_ratios: Sequence[float] | None = None, *,
    core_model: str | None = None,
    black_hole_masses_solar: Sequence[float] = (3.0, 10.0, 30.0, 4.0e6, 6.5e9),
) -> dict[str, object]:
    """Pure, JSON-compatible report; statuses refer only to sampled densities."""
    if density_ratios is None:
        density_ratios = (0.0, 0.01, 0.5, 1.0, 2.0,
                          CURRENT_MASS_KINK_DENSITY / n_0,
                          5.0, 10.0, 20.0, 100.0, 500.0, 1.0e6)
    rows = []
    for ratio in density_ratios:
        ratio = _nonnegative_finite(ratio, "density ratio")
        density = ratio * n_0
        if _at_current_mass_kink(density):
            row = {
                "density_fm3": density, "mass_mev": M_star(density),
                "energy_mev_fm3": energy_density(density),
                "pressure_mev_fm3": None, "cs2": None,
                "pressure_over_energy": None,
                "status": "undefined_thermodynamics_at_mass_kink",
            }
        else:
            row = eos_state(density)
        rows.append({"density_ratio": ratio, **row})
    smooth = [row for row in rows if "closure_residual_mev_fm3" in row]
    positive = [row for row in smooth if row["density_fm3"] > 0.0]
    closure = max((abs(row["closure_residual_mev_fm3"]) / max(
        abs(row["energy_mev_fm3"]), abs(row["pressure_mev_fm3"]),
        abs(row["density_fm3"] * row["chemical_potential_mev"]), 1e-300)
        for row in smooth), default=None)
    unstable = [row["density_ratio"] for row in positive if row["status"] == "mechanically_unstable"]
    superluminal = [row["density_ratio"] for row in positive if row["cs2"] is not None and row["cs2"] > 1.0]
    exceeds_scale = [row["density_ratio"] for row in rows if row["energy_mev_fm3"] > epsilon_density_scale()]
    if core_model is None:
        geometry = {"status": "not_computed_no_geometry_model", "radii": []}
    elif core_model == "assumed_hayward":
        geometry = {
            "status": "assumed_hayward_crossover_not_collapse_solution",
            "assumed_core_density_mev_fm3": epsilon_density_scale(),
            "radii": [{"mass_solar": mass, "crossover_radius_km": de_sitter_core_radius(
                mass, model=core_model)} for mass in black_hole_masses_solar],
        }
    else:
        raise ValueError("unsupported core_model")
    return {
        "model_status": "phenomenological_quasiparticle_ansatz_not_source_complete",
        "inputs": {"n0_fm3": n_0, "nucleon_mass_mev": M_N,
                   "omega_mass_parameter_mev": M_Omega_0,
                   "current_mass_parameter_mev": M_current_0,
                   "kappa1": kappa_1, "kappa2": kappa_2,
                   "fermion_degeneracy": FERMION_DEGENERACY},
        "current_mass_kink_density_fm3": CURRENT_MASS_KINK_DENSITY,
        "density_scale_mev_fm3": epsilon_density_scale(),
        "density_scale_status": "dimensional_scale_not_an_upper_bound",
        "eos_rows": rows,
        "sampled_checks": {
            "closure_max_relative": closure,
            "closure_within_1e_minus_11": closure is not None and closure < 1e-11,
            "mechanically_unstable_density_ratios": unstable,
            "superluminal_density_ratios": superluminal,
            "energy_exceeds_density_scale_at_ratios": exceeds_scale,
            "differentiable_at_all_samples": len(smooth) == len(rows),
            "all_positive_samples_locally_stable_and_subluminal": bool(positive) and all(
                row["status"] == "locally_stable_and_subluminal" for row in positive),
        },
        "geometry": geometry,
        "uncomputed": ["collapse dynamics", "singularity resolution", "exterior matching",
                       "cosmological bounce", "observational confirmation"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the computed state as JSON")
    parser.add_argument("--assumed-hayward", action="store_true", help="include explicitly assumed Hayward crossover radii")
    args = parser.parse_args(argv)
    state = compute_state(core_model="assumed_hayward" if args.assumed_hayward else None)
    if args.json:
        print(json.dumps(state, indent=2, allow_nan=False))
        return
    print("PHENOMENOLOGICAL MASS ANSATZ: THERMODYNAMIC AUDIT")
    print(f"Model status: {state['model_status']}")
    print("The mass split and melting parameters are inputs, not measured decompositions.")
    print("n/n0          energy           pressure          cs2       calculated status")
    for row in state["eos_rows"]:
        pressure = "undefined" if row["pressure_mev_fm3"] is None else f"{row['pressure_mev_fm3']:.6g}"
        cs2 = "undefined" if row["cs2"] is None else f"{row['cs2']:.6g}"
        print(f"{row['density_ratio']:9.4g}  {row['energy_mev_fm3']:15.6g}  {pressure:>15}  {cs2:>11}  {row['status']}")
    print(f"Density scale: {state['density_scale_mev_fm3']:.6g} MeV/fm^3 ({state['density_scale_status']})")
    print("Sampled checks: " + json.dumps(state["sampled_checks"], allow_nan=False))
    print("Geometry: " + json.dumps(state["geometry"], allow_nan=False))
    print("Ultrarelativistic P/energy -> 1/3 does not imply de Sitter P/energy = -1.")
    print("Uncomputed: " + "; ".join(state["uncomputed"]))


if __name__ == "__main__":
    main()
