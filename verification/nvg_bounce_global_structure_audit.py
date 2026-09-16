#!/usr/bin/env python3
"""Conditional global FLRW structure of NVG matter + reconstructed gravity.

The additional sine/cuscuton gravity and its density cap are assumptions.
Matter uses the original positive-energy, neutral, homogeneous, zero-T Fermi
closure. W is NOT constrained to its equilibrium minimum. Symbolic identities
and off-equilibrium point checks support the accompanying continuation proof;
they are not a numerical full-history integration or a perturbation analysis.
The CLI prints JSON only and never overwrites saved research artifacts.
"""
from __future__ import annotations

import argparse
import json

import mpmath as mp
import sympy as sp

from source_complete_scaling_saturation_audit import BulkModel


STATUS = "conditional_full_homogeneous_bounce_not_confirmed_NVG"
GN_INPUT = "6.70883e-45"  # MeV^-2, same declared input as the previous audit.
REQUIRED_SYMBOLIC_CHECKS = frozenset({
    "dynamic_enthalpy", "matter_continuity_off_equilibrium", "sine_constraint_propagation",
    "raychaudhuri_from_regular_q", "baryon_number", "bounce_positive_rate",
    "friedmann_without_sign_patch", "R_affine_in_enthalpy", "R0_nonnegative", "R0_upper",
    "R2_lower", "R2_upper", "K_convex_endpoint_bound", "K0_upper", "K2_upper",
    "dominant_energy_difference", "positive_enthalpy_sum",
})


def finite(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool is not a physical number")
    try:
        value = mp.mpf(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name}: finite real number required") from exc
    if not mp.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name}: finite {'positive ' if positive else ''}number required")
    return value


def number(value):
    return mp.nstr(finite(value, "derived value"), 35)


def symbolic_checks(*, scalar_force_sign=1, sine_rate_scale=1):
    """Derive constraint propagation and explicit polynomial certificates.

    The optional sign/scale are deliberate negative controls, not new models.
    """
    if isinstance(scalar_force_sign, bool) or scalar_force_sign not in (-1, 1):
        raise ValueError("scalar_force_sign must be -1 or +1")
    scale = sp.Rational(str(finite(sine_rate_scale, "sine_rate_scale")))
    n, W, M, rc, B, V = sp.symbols("n W M rc B V", positive=True)
    v, H, q = sp.symbols("v H q", real=True)
    E = sp.Function("E")(n, W)
    rho = v*v/2+E
    P = v*v/2+n*sp.diff(E, n)-E
    Q = rho+P
    alpha = sp.sqrt(3/(4*M*M*rc))
    Hq = alpha*rc*sp.sin(2*q)/3
    ndot, vdot = -3*H*n, -3*H*v-scalar_force_sign*sp.diff(E, W)
    qdot = scale*alpha*Q
    rhodot = sp.diff(rho, n)*ndot+sp.diff(rho, W)*v+sp.diff(rho, v)*vdot
    checks = {
        "dynamic_enthalpy": Q-v*v-n*sp.diff(E, n),
        "matter_continuity_off_equilibrium": rhodot+3*H*Q,
        "sine_constraint_propagation": rhodot.subs(H, Hq)+rc*sp.sin(2*q)*qdot,
        "raychaudhuri_from_regular_q": sp.diff(Hq, q)*qdot-Q*(2*sp.cos(q)**2-1)/(2*M*M),
        "baryon_number": (-3*H*n)*V+n*3*H*V,
        "bounce_positive_rate": (sp.diff(Hq, q)*qdot).subs(q, 0)-Q/(2*M*M),
        "friedmann_without_sign_patch": Hq**2-rc*sp.cos(q)**2*sp.sin(q)**2/(3*M*M),
    }
    # Each certificate is an equality to a manifestly nonnegative expression
    # on 0<=x<=1, 0<=u<=2. No sampling establishes these global inequalities.
    x, u = sp.symbols("x u", real=True)
    R = -3*u*x*(1-2*x)+4*x*(1-x)  # R_scalar * M^2 / rho_c.
    K = 12*x*x*(((1-x)/3-u*(1-2*x)/2)**2+(1-x)**2/9)
    R0, R2 = R.subs(u, 0), R.subs(u, 2)
    K0, K2 = K.subs(u, 0), K.subs(u, 2)
    checks.update({
        "R_affine_in_enthalpy": R-((1-u/2)*R0+u*R2/2),
        "R0_nonnegative": R0-4*x*(1-x),
        "R0_upper": 1-R0-(2*x-1)**2,
        "R2_lower": R2+sp.Rational(1, 8)-(8*x-1)**2/8,
        "R2_upper": 6-R2-2*(1-x)*(4*x+3),
        "K_convex_endpoint_bound": (1-u/2)*K0+u*K2/2-K-3*u*(2-u)*x*x*(2*x-1)**2,
        "K0_upper": sp.Rational(1, 6)-K0-(2*x-1)**2*(1+4*x*(1-x))/6,
        "K2_upper": 12-K2-sp.Rational(4, 3)*(1-x)*(26*x**3+4*x*x+9*x+9),
    })
    f, pf, potential, vector = sp.symbols("F PF U vector", nonnegative=True)
    r = v*v/2+f+potential+vector
    p = v*v/2+pf-potential+vector
    checks["dominant_energy_difference"] = r-p-(f-pf+2*potential)
    checks["positive_enthalpy_sum"] = r+p-(v*v+f+pf+2*vector)
    result = {}
    for name, value in checks.items():
        residual = sp.trigsimp(sp.simplify(value))
        result[name] = {"residual": str(residual), "passed": bool(residual == 0)}
    return result


def point_state(n_ratio="1", W_ratio="1", speed_ratio="0", rho_c_ratio="1",
                *, direction=1, dps=80):
    """An off-equilibrium constrained jet, not a prescribed exact trajectory.

    Ratios use n0, W0, W0^2 and W0^4 respectively. rho_c_ratio is an explicit
    external gravity input, NOT inferred from NVG. Both signs of H are tested.
    """
    if isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 300:
        raise ValueError("dps must be an integer from 80 to 300")
    if isinstance(direction, bool) or direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    with mp.workdps(dps):
        nr = finite(n_ratio, "n_ratio", positive=True)
        yr = finite(W_ratio, "W_ratio", positive=True)
        vr = finite(speed_ratio, "speed_ratio")
        cr = finite(rho_c_ratio, "rho_c_ratio", positive=True)
        model = BulkModel()
        n, W, v = nr*model.n0, yr*model.W0, vr*model.W0**2
        rc = cr*model.W0**4
        M2 = 1/(8*mp.pi*mp.mpf(GN_INPUT))
        alpha = mp.sqrt(3/(4*M2*rc))
        s = model.state(n, yr)
        rho = v*v/2+s["energy_total"]
        pressure = v*v/2+s["pressure_total"]
        if not 0 < rho <= rc:
            raise ValueError("initial matter energy must satisfy 0 < rho <= declared rho_c")
        x, Q = rho/rc, v*v+n*s["mu"]
        H = direction*mp.sqrt(rho*(1-x)/(3*M2))
        q = direction*mp.acos(mp.sqrt(x))
        force = s["residual"]/model.W0
        ndot, vdot, qdot = -3*H*n, -3*H*v-force, alpha*Q
        Hdot = -Q*(1-2*x)/(2*M2)
        # Independent momentum integrals: do not use saved tables or an
        # equilibrium root to certify this dynamically displaced point.
        k, mass = s["k"], s["m"]
        F = model.d/(2*mp.pi**2)*mp.quad(lambda z: z*z*mp.sqrt(z*z+mass*mass), [0, k])
        PF = model.d/(6*mp.pi**2)*mp.quad(lambda z: z**4/mp.sqrt(z*z+mass*mass), [0, k])
        ns = model.d*mass/(2*mp.pi**2)*mp.quad(lambda z: z*z/mp.sqrt(z*z+mass*mass), [0, k])
        e_n = mp.diff(lambda nn: model.state(nn, yr)["energy_total"], n)
        e_W = mp.diff(lambda ww: model.state(n, ww/model.W0)["energy_total"], W)
        rhodot = v*vdot+e_n*ndot+e_W*v
        hmax = mp.sqrt(rc/(12*M2))
        Gv = model.gomega**2/(model.momega/model.W0)**2
        CF = mp.mpf(3)/4*(6*mp.pi**2/model.d)**(mp.mpf(1)/3)
        nmax = (rc/CF)**(mp.mpf(3)/4)
        Wmax = mp.sqrt(model.W0**2+2*mp.sqrt(rc/model.lam))
        Wlower = n*mp.sqrt(Gv/(2*rc))
        Ricci = 6*(Hdot+2*H*H)
        Kretschmann = 12*((Hdot+H*H)**2+H**4)
        escale = max(rho, mp.mpf(1))
        # Compare rate residuals to rate scales, not MeV^4: a physical tiny G
        # must not make wrong equations pass merely by suppressing all rates.
        rscale = max(abs(v*vdot), abs(e_n*ndot), abs(e_W*v),
                     abs(3*H*Q), abs(rc*mp.sin(2*q)*qdot), mp.mpf("1e-100"))
        residuals = {
            "independent_fermi_energy": (s["energy"]-F)/escale,
            "independent_fermi_pressure": (s["pressure"]-PF)/escale,
            "independent_fermi_trace": (F-3*PF-mass*ns)/escale,
            "independent_mu_derivative": (s["mu"]-e_n)/max(abs(e_n), 1),
            "independent_scalar_force": (force-e_W)/max(abs(e_W), 1),
            "enthalpy": (rho+pressure-Q)/escale,
            "continuity": (rhodot+3*H*Q)/rscale,
            "constraint_propagation": (rhodot+rc*mp.sin(2*q)*qdot)/rscale,
            "q_friedmann": (alpha*rc*mp.sin(2*q)/3-H)/max(hmax, mp.mpf("1e-100")),
            "q_raychaudhuri": (2*alpha*rc*mp.cos(2*q)*qdot/3-Hdot)/(rc/M2),
        }
        flags = {
            "positive_enthalpy": 0 < Q <= 2*rho,
            "Fermi_massless_lower_bound": F >= CF*n**(mp.mpf(4)/3),
            "hubble_bound": abs(H) <= hmax*(1+mp.mpf("1e-70")),
            "hubble_rate_bound": abs(Hdot) <= rc/M2*(1+mp.mpf("1e-70")),
            "scalar_speed_bound": abs(v) <= mp.sqrt(2*rc),
            "density_bound": n <= nmax,
            "scalar_upper_bound": W <= Wmax,
            "charged_scalar_barrier": W >= Wlower > 0,
            "Ricci_bounds": -rc/(8*M2) <= Ricci <= 6*rc/M2,
            "Kretschmann_bound": 0 <= Kretschmann <= 12*rc*rc/M2**2,
        }
        max_residual = max(abs(finite(z, key)) for key, z in residuals.items())
        passed = max_residual < mp.mpf("1e-60") and all(flags.values())
        # If this is on the contracting branch, n >= n_initial up to the
        # bounce. On the expanding branch the same bound holds backwards.
        # Q >= n_initial*k_F_initial yields a finite proper-time bound.
        time_bound = abs(q)/(alpha*n*k)
        return {
            "status": STATUS, "mathematical_checks_passed": bool(passed),
            "stationarity_imposed": False, "full_history_integrated": False,
            "rho_c_is_external_input": True,
            "inputs": {"n_over_n0": number(nr), "W_over_W0": number(yr),
                       "Wdot_over_W0_squared": number(vr), "rho_c_over_W0_fourth": number(cr),
                       "G_Newton_MeV_minus2": GN_INPUT, "H_direction": direction, "dps": dps},
            "state": {"rho_MeV4": number(rho), "P_MeV4": number(pressure),
                      "rho_over_rho_c": number(x), "enthalpy_MeV4": number(Q),
                      "H_MeV": number(H), "Hdot_MeV2": number(Hdot),
                      "scalar_force_MeV3": number(force), "q": number(q)},
            "bounds": {"n_max_over_n0": number(nmax/model.n0),
                       "W_max_over_W0": number(Wmax/model.W0),
                       "W_lower_at_current_n_over_W0": number(Wlower/model.W0),
                       "a_min_over_a_initial": number((n/nmax)**(mp.mpf(1)/3)),
                       "time_to_bounce_upper_MeV_minus1": number(time_bound),
                       "time_direction_to_bounce": "future" if direction == -1 else "past",
                       "H_max_MeV": number(hmax)},
            "inequalities": {key: bool(value) for key, value in flags.items()},
            "residuals": {key: number(value) for key, value in residuals.items()},
            "maximum_scaled_residual": number(max_residual),
            "scope": "Homogeneous positive-energy Fermi mean-field closure plus assumed sine gravity; not full inhomogeneous/quantum NVG, not a black-hole theorem.",
        }


def audit(n_ratio="1", W_ratio="1", speed_ratio="0", rho_c_ratio="1", *, direction=1, dps=80):
    result = point_state(n_ratio, W_ratio, speed_ratio, rho_c_ratio, direction=direction, dps=dps)
    checks = symbolic_checks()
    checks_passed = (isinstance(checks, dict) and set(checks) == REQUIRED_SYMBOLIC_CHECKS and
                     all(isinstance(row, dict) and row.get("passed") is True and
                         row.get("residual") == "0" for row in checks.values()))
    result["mathematical_checks_passed"] = result["mathematical_checks_passed"] is True and checks_passed
    result["symbolic_checks"] = checks
    result["theorem_assumptions"] = [
        "Flat FLRW with complete Euclidean spatial slices and positive conserved baryon number.",
        "Positive canonical scalar kinetic energy and quartic potential; neutral vector energy retained.",
        "Original fixed couplings, homogeneous zero-T Fermi mean-field closure, no pair-production extension.",
        "Finite positive externally specified rho_c and reconstructed cuscuton gravity on its timelike branch.",
        "Initial data satisfy the gravitational constraint and W>0, n>0.",
    ]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ratio", default="1")
    parser.add_argument("--W-ratio", default="1")
    parser.add_argument("--speed-ratio", default="0")
    parser.add_argument("--rho-c-ratio", default="1")
    parser.add_argument("--direction", choices=(-1, 1), type=int, default=1)
    args = parser.parse_args(argv)
    try:
        result = audit(args.n_ratio, args.W_ratio, args.speed_ratio, args.rho_c_ratio, direction=args.direction)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        print(json.dumps({"status": "invalid_or_failed_audit", "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["mathematical_checks_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
