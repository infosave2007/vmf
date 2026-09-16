#!/usr/bin/env python3
"""Symbolic action reduction and conditional bounce reconstruction audit.

The accepted matter action and Einstein-Hilbert gravity forbid a flat bounce
in the homogeneous positive-energy fermion closure. A sine Hamiltonian is an
explicitly reconstructed DIFFERENT gravitational functional, not derived NVG.
All computations are pure; the CLI prints JSON and never saves an artifact.
"""
from __future__ import annotations

import json

import mpmath as mp
import sympy as sp

from source_complete_scaling_saturation_audit import BulkModel


def finite_number(value, name, *, positive=False, nonnegative=False):
    """Validate before arbitrary-precision arithmetic or decimal serialization."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        number = mp.mpf(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not mp.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def decimal(value):
    return mp.nstr(finite_number(value, "derived value"), 35)


def symbolic_checks(*, phase_sign=1, map_deformation=0):
    """Differentiate generating functionals; arguments expose negative controls.

    phase_sign=-1 changes the scalar-phase kinetic sign. map_deformation
    adds rho_c*eta*sin(alpha*p)^4. Neither is an accepted model option.
    The identities deliberately reject these changes where they matter.
    """
    if phase_sign not in (-1, 1):
        raise ValueError("phase_sign must be +1 or -1")
    finite_number(map_deformation, "map_deformation")
    eta = sp.Rational(str(map_deformation))
    V, lapse, W, B, q, g = sp.symbols("V lapse W B q g", positive=True)
    A, theta_dot, charge = sp.symbols("A theta_dot charge", real=True)
    residuals = {}

    # Start from both the covariant phase kinetic term and its matter source.
    phase_L = phase_sign*V*W**2*(theta_dot-q*A)**2/(2*lapse)-g*A*B
    solved_velocity = sp.solve(sp.diff(phase_L, theta_dot)-charge, theta_dot)[0]
    routh = sp.expand((phase_L-charge*theta_dot).subs(theta_dot, solved_velocity))
    gauss = sp.diff(routh, A)
    neutral_charge = sp.solve(gauss, charge)[0]
    reduced_phase = sp.simplify(routh.subs(charge, neutral_charge))
    vector_energy = sp.simplify(-sp.diff(reduced_phase, lapse)/V)
    residuals["gauss_neutrality"] = q*neutral_charge+g*B
    residuals["positive_vector_energy_from_routh"] = vector_energy-g**2*B**2/(2*q**2*V**2*W**2)
    residuals["gauss_multiplier_eliminated"] = sp.diff(reduced_phase, A)
    residuals["source_term_required"] = sp.diff(routh, A)+q*charge+g*B

    # Lapse variation is performed before choosing proper time. Matter number
    # B stays fixed under a-variation, so density B/a^3 also varies.
    a, ad, wd, Mpl2 = sp.symbols("a ad wd Mpl2", positive=True)
    k = sp.symbols("k", real=True)
    E = sp.Function("E")
    e_a = E(B/a**3, W)
    gravity_L = -3*Mpl2*a*ad**2/lapse+3*Mpl2*k*lapse*a
    matter_L = a**3*wd**2/(2*lapse)-lapse*a**3*e_a
    L = gravity_L+matter_L
    rho_lapse = sp.simplify(-sp.diff(matter_L, lapse)/a**3)
    pressure_a = sp.simplify(sp.diff(matter_L, a)/(3*lapse*a**2))
    n = sp.symbols("n", positive=True)
    En = sp.diff(E(n, W), n)
    P_legendre = wd**2/(2*lapse**2)+n*En-E(n, W)
    residuals["lapse_energy"] = rho_lapse-wd**2/(2*lapse**2)-e_a
    residuals["einstein_lapse_constraint_all_k"] = sp.diff(L, lapse)/a**3-(
        3*Mpl2*((ad/(lapse*a))**2+k/a**2)-rho_lapse)
    residuals["pressure_fixed_baryon_number"] = pressure_a-P_legendre.subs(n, B/a**3)
    pa, Pi = sp.symbols("pa Pi", real=True)
    momenta = (sp.diff(L, ad), sp.diff(L, wd))
    velocities = sp.solve((momenta[0]-pa, momenta[1]-Pi), (ad, wd))
    constraint = sp.simplify(((pa*ad+Pi*wd-L)/lapse).subs(velocities))
    residuals["einstein_legendre_transform"] = constraint-(
        -pa**2/(12*Mpl2*a)-3*Mpl2*k*a+Pi**2/(2*a**3)+a**3*e_a)

    # Generic flat Hamiltonian: volume momentum p and scalar momentum Pi
    # are independent canonical variables. Differentiate at fixed Pi,B,W.
    p = sp.symbols("p", real=True)
    f = sp.Function("f")
    C = -V*f(p)+Pi**2/(2*V)+V*E(B/V, W)
    Vdot, pdot = sp.diff(C, p), -sp.diff(C, V)
    Wdot, Pidot = sp.diff(C, Pi), -sp.diff(C, W)
    rho = Pi**2/(2*V**2)+E(B/V, W)
    pressure = (Pi**2/(2*V**2)+n*En-E(n, W)).subs(n, B/V)
    H = sp.simplify(Vdot/(3*V))
    Hdot = sp.diff(H, p)*pdot
    residuals["hamiltonian_volume_rate"] = H+sp.diff(f(p), p)/3
    residuals["hamiltonian_pressure_fixed_Pi_B_W"] = pdot-f(p)-pressure
    residuals["constraint_is_total_energy"] = C/V-(rho-f(p))
    # Eliminate only the value f=rho: substituting f inside its derivatives
    # would erase gravitational curvature and create a spurious zero identity.
    f_curvature = sp.symbols("f_curvature", real=True)
    constrained_Hdot = Hdot.subs(sp.diff(f(p), p, 2), f_curvature).subs(f(p), rho)
    residuals["hamiltonian_raychaudhuri_on_constraint"] = constrained_Hdot+f_curvature*(rho+pressure)/3
    rho_dot = sp.diff(rho, V)*Vdot+sp.diff(rho, W)*Wdot+sp.diff(rho, Pi)*Pidot
    residuals["continuity_without_H_division"] = rho_dot+3*H*(rho+pressure)
    residuals["baryon_continuity"] = sp.diff(B/V, V)*Vdot+3*H*B/V
    Wddot = sp.diff(Wdot, V)*Vdot+sp.diff(Wdot, Pi)*Pidot
    residuals["scalar_equation_from_hamiltonian"] = Wddot+3*H*Wdot+sp.diff(E(B/V, W), W)

    # The curved Einstein Hamiltonian has an explicit volume curvature term;
    # it must not be confused with the flat sine reconstruction below.
    EH_C = C.subs(f(p), 3*p**2/(4*Mpl2))-3*Mpl2*k*V**sp.Rational(1, 3)
    EH_H = sp.diff(EH_C, p)/(3*V)
    EH_Hdot = sp.diff(EH_H, p)*(-sp.diff(EH_C, V))
    p_squared_constraint = sp.solve(EH_C, p**2)[0]
    residuals["einstein_raychaudhuri_all_k"] = EH_Hdot.subs(p**2, p_squared_constraint)-(
        k/V**sp.Rational(2, 3)-(rho+pressure)/(2*Mpl2))

    # On-shell scalar equation and the independent free-Fermi trace relation.
    FF, PF, ns, gs, Gv = sp.symbols("FF PF ns gs Gv", real=True)
    speed, acceleration, hubble = sp.symbols("speed acceleration hubble", real=True)
    U = sp.Function("U")
    source_energy = FF+U(W)+Gv*n**2/(2*W**2)
    source_pressure = PF-U(W)+Gv*n**2/(2*W**2)
    rho_source = speed**2/2+source_energy
    P_source = speed**2/2+source_pressure
    EF = sp.symbols("EF", positive=True)
    residuals["dynamic_NEC_sum"] = (rho_source+P_source).subs(FF, n*EF-PF)-(
        speed**2+n*EF+Gv*n**2/W**2)
    trace = (rho_source-3*P_source).subs(FF, 3*PF+gs*W*ns)
    scalar_force = gs*ns+sp.diff(U(W), W)-Gv*n**2/W**3
    acceleration_on_shell = -3*hubble*speed-scalar_force
    divergence = speed**2+W*acceleration+3*hubble*W*speed
    residuals["dynamic_trace_completion"] = (trace-(
        4*U(W)-W*sp.diff(U(W), W)-divergence)).subs(acceleration, acceleration_on_shell)

    # Reconstruct a DIFFERENT flat gravitational functional and derive both
    # background equations from Hamilton derivatives, including at H=0.
    rc, alpha, GN = sp.symbols("rho_c alpha G_Newton", positive=True)
    enthalpy = sp.symbols("enthalpy", positive=True)
    sine_f = rc*(sp.sin(alpha*p)**2+eta*sp.sin(alpha*p)**4)
    sine_H = -sp.diff(sine_f, p)/3
    sine_Hdot = -sp.diff(sine_f, p, 2)*enthalpy/3
    alpha_squared = 6*sp.pi*GN/rc
    residuals["sine_friedmann_from_hamiltonian"] = (sine_H**2-
        8*sp.pi*GN*sine_f*(1-sine_f/rc)/3).subs(GN, rc*alpha**2/(6*sp.pi))
    residuals["sine_raychaudhuri_from_hamiltonian"] = (sine_Hdot+
        4*sp.pi*GN*enthalpy*(1-2*sine_f/rc)).subs(GN, rc*alpha**2/(6*sp.pi))
    bounce_p = sp.pi/(2*alpha)
    residuals["bounce_density_equals_rho_c"] = sine_f.subs(p, bounce_p)-rc
    residuals["bounce_H_zero"] = sine_H.subs(p, bounce_p)
    residuals["bounce_acceleration_without_dividing_by_H"] = (
        sine_Hdot.subs(p, bounce_p)-2*rc*alpha**2*enthalpy/3)
    residuals["GR_low_momentum_coefficient"] = sp.limit(sine_f/p**2, p, 0)-rc*alpha**2
    residuals["GR_match_Newton_constant"] = (rc*alpha**2-6*sp.pi*GN).subs(alpha**2, alpha_squared)
    # Generic maximum, independent of the sine example: f'=0, f''=-c<0.
    maximum_curvature = sp.symbols("maximum_curvature", positive=True)
    generic_H = -sp.diff(f(p), p)/3
    generic_acceleration = -sp.diff(f(p), p, 2)*enthalpy/3
    residuals["generic_stationary_H_zero"] = generic_H.subs(sp.diff(f(p), p), 0)
    residuals["generic_maximum_is_bounce_for_positive_enthalpy"] = (
        generic_acceleration.subs(sp.diff(f(p), p, 2), -maximum_curvature)
        -maximum_curvature*enthalpy/3)

    simplified = {name: sp.trigsimp(sp.simplify(value)) for name, value in residuals.items()}
    return {name: {"residual": str(value), "passed": bool(value == 0)}
            for name, value in simplified.items()}


def source_witness(n_ratio="0.1", W_ratio="1.1", Wdot="0"):
    """Off-equilibrium initial data, checked against independent Fermi integrals.

    This is a point witness, not a universal proof or a finite-time evolution.
    Wdot uses proper-time natural units MeV^2. No equilibrium root is imposed.
    """
    with mp.workdps(80):
        x = finite_number(n_ratio, "n_ratio", positive=True)
        y = finite_number(W_ratio, "W_ratio", positive=True)
        speed = finite_number(Wdot, "Wdot")
        model = BulkModel()
        for key in ("W0", "lam", "MN", "momega", "gomega", "hbarc", "n0", "d"):
            finite_number(getattr(model, key), key, positive=True)
        n, W = x*model.n0, y*model.W0
        gs, q = model.MN/model.W0, model.momega/model.W0
        Gv = model.gomega**2/q**2
        mass, kf = gs*W, (6*mp.pi**2*n/model.d)**(mp.mpf(1)/3)
        F = model.d/(2*mp.pi**2)*mp.quad(lambda z: z*z*mp.sqrt(z*z+mass*mass), [0, kf])
        PF = model.d/(6*mp.pi**2)*mp.quad(lambda z: z**4/mp.sqrt(z*z+mass*mass), [0, kf])
        ns = model.d*mass/(2*mp.pi**2)*mp.quad(lambda z: z*z/mp.sqrt(z*z+mass*mass), [0, kf])
        potential = model.lam*(W*W-model.W0**2)**2/4
        vector = Gv*n*n/(2*W*W)
        rho = speed*speed/2+F+potential+vector
        pressure = speed*speed/2+PF-potential+vector
        force = gs*ns+model.lam*W*(W*W-model.W0**2)-Gv*n*n/W**3
        baseline = model.state(n, y)
        for key in ("energy_total", "pressure_total", "residual", "y"):
            finite_number(baseline[key], f"BulkModel.{key}")
        scale = max(abs(rho), abs(pressure), mp.mpf(1))
        residuals = (
            (rho-baseline["energy_total"]-speed*speed/2)/scale,
            (pressure-baseline["pressure_total"]-speed*speed/2)/scale,
            (F-3*PF-mass*ns)/scale,
            (force-baseline["residual"]/model.W0)/max(abs(force), mp.mpf(1)),
        )
        for residual in residuals:
            finite_number(residual, "independent source residual")
        max_error = max(map(abs, residuals))
        if max_error >= mp.mpf("1e-60"):
            raise ArithmeticError("independent source functional comparison failed")
        enthalpy, sec = rho+pressure, rho+3*pressure
        for name, value in (("rho", rho), ("pressure", pressure), ("enthalpy", enthalpy),
                            ("rho_plus_3P", sec), ("scalar_force", force)):
            finite_number(value, name)
        if rho <= 0 or enthalpy <= 0:
            raise ArithmeticError("positive-energy homogeneous source witness failed")
        return {
            "n_over_n0": decimal(x), "W_over_W0": decimal(y), "Wdot_MeV2": decimal(speed),
            "rho_MeV4": decimal(rho), "P_MeV4": decimal(pressure),
            "rho_plus_P_MeV4": decimal(enthalpy), "rho_plus_3P_MeV4": decimal(sec),
            "scalar_force_MeV3": decimal(force),
            "independent_integral_max_relative_residual": decimal(max_error),
            "stationarity_imposed": False,
            "flat_H0_allowed": bool(rho == 0), "open_H0_allowed": bool(rho < 0),
            "closed_H0_allowed": bool(rho > 0),
            "closed_turning_point": "bounce" if sec < 0 else "maximum" if sec > 0 else "marginal",
            "closed_initial_constraint": "a_b^2=3*M_Pl^2/rho_b; choose B=n_b*a_b^3",
            "scope": "One off-equilibrium point; local turning criterion only, no full evolution or cycle.",
        }


def reconstructed_point(momentum, *, rho_c="1", G_Newton="1", enthalpy="1"):
    """Independent mpmath differentiation of the reconstructed sine functional."""
    with mp.workdps(80):
        p = finite_number(momentum, "momentum")
        rc = finite_number(rho_c, "rho_c", positive=True)
        gn = finite_number(G_Newton, "G_Newton", positive=True)
        rp = finite_number(enthalpy, "enthalpy", nonnegative=True)
        alpha = mp.sqrt(6*mp.pi*gn/rc)
        function = lambda v: rc*mp.sin(alpha*v)**2
        rho = function(p)
        first, second = mp.diff(function, p), mp.diff(function, p, 2)
        H, Hdot = -first/3, -second*rp/3
        friedmann_residual = H*H-8*mp.pi*gn*rho*(1-rho/rc)/3
        raychaudhuri_residual = Hdot+4*mp.pi*gn*rp*(1-2*rho/rc)
        values = {"rho": rho, "H": H, "Hdot": Hdot, "f_prime": first,
                  "f_second": second, "alpha": alpha,
                  "friedmann_residual": friedmann_residual,
                  "raychaudhuri_residual": raychaudhuri_residual}
        return {name: decimal(value) for name, value in values.items()}


def short_time_leading_order(*, rtol=2e-11, atol=2e-13, max_step=0.04):
    """DOP853 at leading order in delta, not full Einstein evolution.

    tau=m_sigma*t; y=W/W0; H=(4*pi*G*rho_b/m_sigma)*h;
    log(a/a_b)=delta*L. Density is constant only at order delta^0; R tracks
    its first-order total-energy consequence without subtracting 1 in double.
    The instantaneous cold fermion closure is formal and not validated by
    this integration. The scalar y is evolved, never frozen.
    """
    from scipy.integrate import solve_ivp

    relative = float(finite_number(rtol, "rtol", positive=True))
    absolute = float(finite_number(atol, "atol", positive=True))
    step_limit = float(finite_number(max_step, "max_step", positive=True))
    if any(not mp.isfinite(value) or value == 0 for value in (relative, absolute, step_limit)):
        raise ValueError("solver tolerances must fit finite positive floating-point numbers")
    with mp.workdps(40):
        model = BulkModel()
        n = mp.mpf("0.1")*model.n0
        msigma2 = 2*model.lam*model.W0**2
        scale = msigma2*model.W0**2
        initial = model.state(n, mp.mpf("1.1"))
        rho_b = finite_number(initial["energy_total"], "rho_b", positive=True)
        GN = mp.mpf("6.70883e-45")  # MeV^-2, declared Newton constant input.
        delta = 4*mp.pi*GN*rho_b/msigma2

        def matter(y, velocity):
            y = finite_number(y, "y", positive=True)
            velocity = finite_number(velocity, "dy/dtau")
            state = model.state(n, y)
            kinetic = scale*velocity**2/2
            energy = finite_number(state["energy_total"]+kinetic, "dynamic energy", positive=True)
            enthalpy_ratio = finite_number((2*kinetic+n*state["ef"]+2*state["vector"])/rho_b,
                                          "dynamic enthalpy ratio", positive=True)
            force = finite_number(state["residual"]/scale, "dimensionless scalar force")
            return float(force), float(enthalpy_ratio), energy

        def rhs(_, state):
            y, velocity, h, _, _ = state
            force, rp, _ = matter(y, velocity)
            values = (velocity, -force, -rp+2/3, h, -3*h*rp)
            for value in values:
                finite_number(value, "ODE derivative")
            return values

        def return_event(_, state):
            return state[2]

        return_event.direction = -1
        return_event.terminal = True
        solved = solve_ivp(rhs, (0.0, 2.0), (1.1, 0.0, 0.0, 0.0, 0.0),
                           method="DOP853", rtol=relative, atol=absolute,
                           events=return_event, dense_output=True, max_step=step_limit)
        if not solved.success or len(solved.t_events[0]) != 1:
            raise ArithmeticError("leading-order scalar evolution failed to resolve its return")
        returned = float(solved.t_events[0][0])
        end = solved.y_events[0][0]
        if returned <= 0 or rhs(returned, end)[2] >= 0 or end[3] <= 0:
            raise ArithmeticError("leading-order return is not expansion followed by recollapse")
        energy_error = constraint_error = 0.0
        max_mass_adiabatic_ratio = 0.0
        for index in range(121):
            y, velocity, h, L, R = solved.sol(returned*index/120)
            _, _, energy = matter(y, velocity)
            energy_error = max(energy_error, float(abs(energy/rho_b-1)))
            constraint_error = max(constraint_error, abs(h*h-(2/3)*(R+2*L)))
            kf = (6*mp.pi**2*n/model.d)**(mp.mpf(1)/3)
            mdot = model.MN*mp.sqrt(msigma2)*velocity
            ratio = abs(mdot)/(kf*kf+(model.MN*y)**2)
            max_mass_adiabatic_ratio = max(max_mass_adiabatic_ratio, float(ratio))
        numeric = {
            "rtol": relative, "atol": absolute, "max_step": step_limit,
            "accepted_steps": len(solved.t)-1, "rhs_evaluations": solved.nfev,
            "return_tau": returned,
            "L_at_return": float(end[3]), "R_at_return": float(end[4]),
            "W_over_W0_at_return": float(end[0]),
            "energy_max_relative_drift": energy_error,
            "scaled_constraint_max_absolute_residual": constraint_error,
            "mass_adiabatic_ratio_max": max_mass_adiabatic_ratio,
        }
        for name, value in numeric.items():
            finite_number(value, name)
        return {
            **numeric, "delta": decimal(delta), "log_a_growth_at_return": decimal(delta*end[3]),
            "status": "LEADING_ORDER_DELTA_FORMAL_FERMION_CLOSURE_NOT_FULL_EVOLUTION",
            "scaled_constraint": "h^2-(2/3)*(R+2*L)=0",
            "limitation": "Short-time expansion in delta only; no validation of instantaneous fermion occupations, macroscopic expansion or a recurrent cycle.",
        }


def build_result():
    checks = symbolic_checks()
    if not checks or not all(item.get("passed") is True and item.get("residual") == "0"
                             for item in checks.values()):
        raise ArithmeticError("action-derivation symbolic check failed")
    displaced = source_witness("0.1", "1.1")
    near_vacuum = source_witness("0.1", "1")
    if displaced["closed_turning_point"] != "bounce" or near_vacuum["closed_turning_point"] != "maximum":
        raise ArithmeticError("declared closed-universe witness or counterexample failed")
    coarse = short_time_leading_order(rtol=2e-8, atol=2e-10, max_step=0.15)
    fine = short_time_leading_order()
    event_difference = abs(coarse["return_tau"]-fine["return_tau"])
    if (event_difference >= 1e-6 or fine["energy_max_relative_drift"] >= 1e-8
            or fine["scaled_constraint_max_absolute_residual"] >= 1e-8):
        raise ArithmeticError("leading-order evolution convergence or conservation check failed")
    return {
        "schema_version": 1,
        "status": "MATHEMATICAL_DERIVATION_VERIFIED_NOT_NEW_PHYSICS",
        "evidence_weight": 0, "observed_likelihood": None,
        "symbolic_checks": checks,
        "accepted_action_consequence": {
            "energy": "rho=Wdot^2/2+F_F+U+G_v*n^2/(2*W^2)",
            "pressure": "P=Wdot^2/2+P_F-U+G_v*n^2/(2*W^2)",
            "NEC": "rho+P=Wdot^2+n*E_F+G_v*n^2/W^2>0 for n>0,W>0",
            "einstein_constraint": "H^2+k/a^2=rho/(3*M_Pl^2)",
            "flat_consequence": "Hdot<0; the accepted action does not produce a flat bounce",
            "dynamic_trace": "rho-3P=4U-W*Uprime-(1/a^3)*d_dtau(a^3*W*Wdot)",
        },
        "reconstructed_gravity": {
            "status": "RECONSTRUCTED_GRAVITY_NOT_DERIVED_FROM_NVG",
            "domain": "flat homogeneous Hamiltonian only; not a covariant or perturbative completion",
            "canonical_brackets": "{V,p}=1; {W,Pi}=1; fixed baryon number B",
            "constraint": "C=-V*f(p)+Pi^2/(2*V)+V*E(B/V,W)=0",
            "function": "f(p)=rho_c*sin(alpha*p)^2; alpha^2=6*pi*G_Newton/rho_c",
            "friedmann": "H^2=(8*pi*G_Newton/3)*rho*(1-rho/rho_c)",
            "raychaudhuri": "Hdot=-4*pi*G_Newton*(rho+P)*(1-2*rho/rho_c)",
            "generic_bounce_criterion": "f'=0 and f''<0 give Hdot>0 when rho+P>0",
            "free_input": "rho_c is not fixed by the accepted NVG action",
        },
        "closed_Einstein_witness": displaced,
        "closed_Einstein_nonbounce_counterexample": near_vacuum,
        "short_time_formal_closure": {"coarse": coarse, "fine": fine,
                                    "return_tau_difference": event_difference},
        "units": "hbar=c=1; G_Newton has units MeV^-2; G_v=gomega^2/qphi^2 is dimensionless",
        "limitations": [
            "Homogeneous mean field with positive-energy cold Fermi occupations; no quantum pair-production or vacuum-loop calculation.",
            "The Gauss-compensating phase charge is fixed by baryon charge, not an external scalar reservoir.",
            "Closed Einstein turning-point samples do not prove a complete bounce history, recurrence, or physical applicability.",
            "The reconstructed sine gravitational functional is additional model input; no NVG origin, covariant closure, perturbation stability, or observational confirmation is established.",
        ],
    }


def main():
    result = build_result()
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
