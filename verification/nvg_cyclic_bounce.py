#!/usr/bin/env python3
"""Conserved effective-fluid backgrounds, NOT a derivation of an NVG bounce.

For a separately conserved microscopic fluid, a prescribed energy map F(epsilon)
requires P_eff=(epsilon+P)*Fprime-F. The quadratic map is an extra assumption.
No torsion action, perturbation stability, entropy transfer, recollapse or
observational likelihood is derived. Rows are computed, not phase lookup tables.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.constants import Boltzmann, electron_volt
from scipy.integrate import solve_ivp

hbar_c = 197.3269804  # MeV fm
c_cgs, G_cgs = 2.998e10, 6.674e-8
MeV_fm3_to_gcm3 = 1.7827e12
M_Omega_0 = 859.0  # historical input scale, not measured vacuum-mass split
kB_MeV = Boltzmann / (1e6 * electron_volt)  # correct MeV/K


def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def mapped_fluid(epsilon, pressure, energy_map, map_derivative):
    """Background conservation/Legendre identity; not a perturbation theory."""
    e, p, f, fp = (finite(v, n) for v, n in zip(
        (epsilon, pressure, energy_map, map_derivative), ("epsilon", "P", "F", "Fprime")))
    if e < 0:
        raise ValueError("epsilon must be nonnegative")
    enthalpy = finite((e+p)*fp, "effective enthalpy")
    peff = finite(enthalpy-f, "effective pressure")
    return {"epsilon_eff": f, "pressure_eff": peff, "enthalpy_eff": enthalpy,
            "trace_eff": finite(f-3*peff, "effective trace")}


def density_map(epsilon, pressure, epsilon_c, *, kind="quadratic"):
    """Alternative ansatzes; neither is silently added to the NVG action."""
    e, ec = finite(epsilon, "epsilon"), finite(epsilon_c, "epsilon_c")
    if e < 0 or ec <= 0:
        raise ValueError("epsilon >= 0 and epsilon_c > 0 required")
    y = e/ec
    if kind == "quadratic":
        f, fp = e*(1-y), 1-2*y
    elif kind == "saturating":
        if e <= ec:
            f, fp = e/(1+y), 1/(1+y)**2
        else:
            inverse = ec/e
            f, fp = ec/(1+inverse), (inverse/(1+inverse))**2
        if e > 0 and (f == 0 or fp == 0):
            raise ValueError("positive saturating map/derivative underflows; use higher precision")
    else:
        raise ValueError("unknown energy map")
    return {**mapped_fluid(e, pressure, f, fp), "Fprime": fp}


def exact_bounce(tau, w=1/3):
    """Flat quadratic-map background; tau=t/tchar, tchar=(8piG*rho_c/3)^(-1/2).

    rho_c is mass-equivalent CGS density; epsilon_c=rho_c*c^2. a_min=1 is
    a normalization. Constant w>-1. This is one bounce, not repeated cycles.
    """
    w = finite(w, "w")
    if w <= -1:
        raise ValueError("this solution requires w > -1")
    tau = np.asarray(tau, dtype=float)
    if not np.all(np.isfinite(tau)):
        raise ValueError("tau must be finite")
    q = finite(3*(1+w), "3*(1+w)")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            z = q*tau/2
            z2 = z*z
            den = 1+z2
            # log1p retains small z^2 even when 1+z^2 rounds to one.
            a = np.exp(np.log1p(z2)/q)
            state = {"a": a, "density_ratio": 1/den,
                     "H_tchar": z/den,
                     "dH_dt_tchar2": (q/2)*((1-z2)/den)/den}
    except FloatingPointError as exc:
        raise ValueError("background exceeds finite floating-point range") from exc
    if any(not np.all(np.isfinite(v)) for v in state.values()):
        raise ValueError("background exceeds finite floating-point range")
    return state


def integrate_bounce(tau, w=1/3, *, rtol=2e-11, atol=2e-13):
    """Independent smooth Raychaudhuri+continuity IVP through H=0.

    No square-root branch, starting offset, clipping, or exact trajectory used.
    Initial data a=1, rho/rho_c=1, H=0 define the conditional background.
    """
    w = finite(w, "w")
    if w <= -1:
        raise ValueError("w > -1 required")
    t = np.asarray(tau, dtype=float)
    if t.ndim != 1 or not np.all(np.isfinite(t)):
        raise ValueError("tau must be a finite one-dimensional sequence")
    q = finite(3*(1+w), "3*(1+w)")

    def rhs(_, state):
        a, y, h = state
        return [a*h, -q*h*y, -(q/2)*y*(1-2*y)]

    out = np.empty((3, len(t)))
    for sign in (-1, 1):
        mask = t*sign > 0
        if np.any(mask):
            endpoint = sign*np.max(np.abs(t[mask]))
            sol = solve_ivp(rhs, (0, endpoint), (1, 1, 0), method="DOP853",
                            rtol=rtol, atol=atol, dense_output=True)
            if not sol.success:
                raise ArithmeticError(sol.message)
            out[:, mask] = sol.sol(t[mask])
    out[:, t == 0] = np.array([1, 1, 0])[:, None]
    return {"a": out[0], "density_ratio": out[1], "H_tchar": out[2]}


def gr_turning_point(epsilon, pressure, *, G=1.0):
    """Einstein gravity, c=1, total supplied fluid includes any Lambda.

    At H=0: a''/a=-4piG(epsilon+3P)/3. A k=+1 turning point with positive
    epsilon+3P is a maximum, not a bounce. No claim about omitted dynamical
    fields or off-equilibrium sectors.
    """
    e, p, g = (finite(v, n) for v, n in zip((epsilon, pressure, G), ("epsilon", "P", "G")))
    if e < 0 or g <= 0:
        raise ValueError("epsilon >= 0 and G > 0 required")
    return {"flat_H2": 8*math.pi*g*e/3,
            "acceleration_over_a_at_H0": -4*math.pi*g*(e+3*p)/3,
            "enthalpy": e+p}


def thermal_scale(epsilon, g_star=47.5):
    """Ideal relativistic-gas comparison with input g_star, not a QCD EOS."""
    e, g = finite(epsilon, "epsilon"), finite(g_star, "g_star")
    if e < 0 or g <= 0:
        raise ValueError("epsilon >= 0 and g_star > 0 required")
    tm = (30*e*hbar_c**3/(math.pi**2*g))**0.25
    return {"kBT_MeV": tm, "temperature_K": tm/kB_MeV, "g_star_input": g}


def compute_bounce_state():
    ec = M_Omega_0**4/hbar_c**3
    tau = np.linspace(-5, 5, 101)
    exact, numeric = exact_bounce(tau), integrate_bounce(tau)
    error = max(float(np.max(np.abs(numeric[k]-exact[k]))) for k in numeric)
    y, h = numeric["density_ratio"], numeric["H_tchar"]
    constraint = float(np.max(np.abs(h*h-y*(1-y))))
    if error > 1e-8 or constraint > 1e-9:
        raise ArithmeticError("conditional bounce IVP failed independent verification")
    return {
        "evidence_status": "CONDITIONAL_BACKGROUND_NOT_NVG_DERIVATION",
        "observed_likelihood": None, "epsilon_c_input_MeV_fm3": ec,
        "t_char_s": 1/math.sqrt(8*math.pi*G_cgs*ec*MeV_fm3_to_gcm3/3),
        "thermal_comparison": thermal_scale(ec),
        "max_ODE_vs_exact_absolute_error": error,
        "max_dimensionless_Friedmann_constraint_error": constraint,
        "radiation_bounce_effective_fluid_in_ec_units": density_map(1, 1/3, 1),
        "samples": [{"tau": t, **{k: float(v) for k,v in exact_bounce(t).items()}}
                    for t in (-5, -1, 0, 1, 5)],
        "missing": ["NVG action producing F and its coefficient",
                    "perturbation action, stability and propagation equations",
                    "recollapse/return map", "entropy and information transfer"],
        "slope_warning": "Fprime=0 at epsilon_c/2: effective barotropic slope is singular, not a certified physical sound speed.",
    }


def calc_critical_density():
    ec = M_Omega_0**4/hbar_c**3
    print(f"Declared density scale, NOT a proved maximum: {ec:.9g} MeV/fm^3")
    return ec, ec*MeV_fm3_to_gcm3


def calc_bounce_trajectory(rho_c_cgs):
    rho = finite(rho_c_cgs, "rho_c")
    if rho <= 0:
        raise ValueError("rho_c must be positive")
    for t in (-5, -1, 0, 1, 5):
        print(t, exact_bounce(t))
    print("Conditional quadratic-map solution; no NVG action derivation.")
    return 1/math.sqrt(8*math.pi*G_cgs*rho/3)


def calc_bounce_temperature(eps_max):
    state = thermal_scale(eps_max)
    print(state)
    return state["kBT_MeV"]


def main():
    import json
    state = compute_bounce_state()
    print(json.dumps(state, indent=2, allow_nan=False))
    return state


if __name__ == "__main__":
    main()
