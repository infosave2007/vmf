#!/usr/bin/env python3
"""Pure audit of a reconstructed cuscuton extension, NOT a prediction of NVG.

Conventions: +---, X=(partial Psi)^2/2>0, increasing timelike Psi,
L_add=-sqrt(2X)-V(Psi), flat FLRW and Mpl^2=1/(8*pi*G_Newton).
Matter is separately conserved with positive enthalpy. Its homogeneous
description need not be barotropic; validity of the NVG fermion closure is
not established here. No accepted action, physical parameter or artifact
is changed. The CLI prints JSON only and never saves its results.

The perturbation calculation is ONLY the canonical massless scalar (w=1)
benchmark of https://arxiv.org/html/1911.06040, Eqs. (40), (42)-(43).
It is not a stability proof for the full NVG field/fermion system, nor a
causal UV completion or a quantum initial-state prescription.
"""
from __future__ import annotations

import argparse
import json
import math

import mpmath as mp
import sympy as sp


STATUS = "reconstructed_covariant_extension_not_NVG_prediction"
SOURCES = {
    "cuscuton_bounce": "https://arxiv.org/abs/1802.06818",
    "perturbation_coefficients": "https://arxiv.org/html/1911.06040",
    "covariant_reconstruction_context": "https://arxiv.org/abs/2405.08071",
    "nonlinear_scope_caution": "https://arxiv.org/abs/2503.01992",
}


def finite_number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        result = mp.mpf(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not mp.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def parametric_state(q, rho_c=1, Mpl=1, enthalpy=1):
    """Return mp values at the current precision; never clamp any field.

    Psi_b=0 fixes a translation convention. q is in the OPEN timelike
    branch (-pi/2, pi/2); its vacuum endpoints are not part of the model.
    rho_c is a new INPUT gravity scale, not inferred from NVG parameters.
    """
    q = finite_number(q, "q")
    rc = finite_number(rho_c, "rho_c", positive=True)
    M = finite_number(Mpl, "Mpl", positive=True)
    Q = finite_number(enthalpy, "enthalpy", positive=True)
    if not -mp.pi/2 < q < mp.pi/2:
        raise ValueError("q must lie strictly inside (-pi/2, pi/2)")
    alpha = mp.sqrt(3/(4*M*M*rc))
    x = mp.cos(q)**2
    H = alpha*rc*mp.sin(2*q)/3
    result = {
        "q": q, "rho_c": rc, "Mpl": M, "enthalpy": Q,
        "alpha": alpha, "x": x, "rho": rc*x,
        "Psi": (q+mp.sin(2*q)/2)/alpha,
        "V": -rc*x*x, "H": H, "Psi_q": 2*x/alpha,
        "V_Psi": 3*H, "V_PsiPsi": 3*(2-1/x)/(4*M*M),
        "constraint_determinant": 3/(2*x),
        "qdot": alpha*Q, "Psidot": 2*x*Q,
        "Hdot": -Q*(1-2*x)/(2*M*M),
        "rho_add": -rc*x*x, "pressure_add": -2*x*Q+rc*x*x,
        "total_enthalpy": Q*(1-2*x),
    }
    return {name: finite_number(value, name) for name, value in result.items()}


def inverse_field(Psi, rho_c=1, Mpl=1):
    """Bracketed inversion of the increasing map, excluding vacuum endpoints.

    Bisection termination uses q accuracy, not the vanishing endpoint slope.
    Precision remains the caller's responsibility arbitrarily near endpoints.
    """
    target = finite_number(Psi, "Psi")
    rc = finite_number(rho_c, "rho_c", positive=True)
    M = finite_number(Mpl, "Mpl", positive=True)
    alpha = mp.sqrt(3/(4*M*M*rc))
    scaled = target*alpha
    if not -mp.pi/2 < scaled < mp.pi/2:
        raise ValueError("Psi is outside the open single-valued field interval")
    lo, hi = -mp.pi/2, mp.pi/2
    for _ in range(mp.mp.prec+12):
        mid = (lo+hi)/2
        if mid == lo or mid == hi:
            break
        value = mid+mp.sin(2*mid)/2
        if value == scaled:
            return mid
        if value < scaled:
            lo = mid
        else:
            hi = mid
    result = (lo+hi)/2
    finite_number(result, "inverse q")
    if not -mp.pi/2 < result < mp.pi/2:
        raise ArithmeticError("insufficient precision to resolve the open branch")
    return result


def symbolic_checks(*, kinetic_sign=-1, potential_scale=1):
    """Differentiate an action and parametrization; expose two negative controls.

    Modified signs/scales are NOT physical options: they test whether the
    independent variation/constraint identities actually detect wrong models.
    """
    if isinstance(kinetic_sign, bool) or kinetic_sign not in (-1, 1):
        raise ValueError("kinetic_sign must be -1 or +1")
    finite_number(potential_scale, "potential_scale")
    scale = sp.Rational(str(potential_scale))
    X, a, lapse, velocity, mu2 = sp.symbols("X a lapse velocity mu2", positive=True)
    psi, Hvar = sp.symbols("Psi H", real=True)
    potential = sp.Function("V")
    pressure = kinetic_sign*sp.sqrt(2*X)-potential(psi)
    energy = 2*X*sp.diff(pressure, X)-pressure
    L = lapse*a**3*pressure.subs(X, velocity**2/(2*lapse**2))
    rho_lapse = -sp.diff(L, lapse)/a**3
    momentum = sp.diff(L, velocity)
    # d/dt p_Psi - dL/dPsi; momentum has no velocity dependence.
    euler = sp.diff(momentum, a)*lapse*a*Hvar-sp.diff(L, psi)
    residuals = {
        "kessence_energy_from_legendre": energy-potential(psi),
        "vanishing_background_kinetic_coefficient": sp.diff(pressure, X)+2*X*sp.diff(pressure, X, 2),
        "lapse_variation_energy": rho_lapse-energy,
        "scale_variation_pressure": sp.diff(L, a)/(3*lapse*a**2)-pressure.subs(X, velocity**2/(2*lapse**2)),
        "field_equation_from_variation": euler/(lapse*a**3)-3*kinetic_sign*Hvar-sp.diff(potential(psi), psi),
        "mu_normalization_redundant_without_other_couplings":
            kinetic_sign*mu2*sp.sqrt(2*X/mu2**2)-kinetic_sign*sp.sqrt(2*X),
    }
    q = sp.symbols("q", real=True)
    M, rc, Q = sp.symbols("Mpl rho_c enthalpy", positive=True)
    alpha = sp.sqrt(3/(4*M**2*rc))
    x = sp.cos(q)**2
    field = (q+sp.sin(2*q)/2)/alpha
    V = -scale*rc*sp.cos(q)**4
    rho = rc*x
    H = alpha*rc*sp.sin(2*q)/3
    field_q = sp.diff(field, q)
    Vp = sp.trigsimp(sp.diff(V, q)/field_q)
    Vpp = sp.trigsimp(sp.diff(Vp, q)/field_q)
    field_dot = field_q*alpha*Q
    Hdot = sp.diff(H, q)*alpha*Q
    euler_on_solution = (euler/(lapse*a**3)).subs(
        {sp.diff(potential(psi), psi): Vp, Hvar: H})
    residuals.update({
        "monotonic_field_jacobian": field_q-2*x/alpha,
        "potential_gradient_from_parametrization": Vp-3*H,
        "field_equation_on_reconstructed_solution": euler_on_solution,
        "friedmann_from_lapse_and_parametrization": 3*M**2*H**2-rho-V,
        "target_friedmann": H**2-rho*(1-rho/rc)/(3*M**2),
        "raychaudhuri_without_H_division": 2*M**2*Hdot+Q+kinetic_sign*field_dot,
        "target_raychaudhuri": Hdot+Q*(1-2*x)/(2*M**2),
        "matter_continuity_without_H_division": sp.diff(rho, q)*alpha*Q+3*H*Q,
        "added_field_continuity": Vp*field_dot+3*H*kinetic_sign*field_dot,
        "field_speed": field_dot-2*x*Q,
        "potential_curvature": Vpp-3*(2-1/x)/(4*M**2),
        "coupled_constraint_determinant": 3-2*M**2*Vpp-3/(2*x),
        "bounce_H_zero": H.subs(q, 0),
        "bounce_positive_acceleration": Hdot.subs(q, 0)-Q/(2*M**2),
        "fold_potential_curvature_zero": Vpp.subs(q, sp.pi/4),
        "fold_determinant_nonzero": (3-2*M**2*Vpp).subs(q, sp.pi/4)-3,
        "fold_field_speed_nonzero": field_dot.subs(q, sp.pi/4)-Q,
    })
    eps = sp.symbols("epsilon", positive=True)
    field_gap = sp.pi/(2*alpha)-field.subs(q, sp.pi/2-eps)
    residuals.update({
        "endpoint_field_gap_cubic": sp.limit(field_gap/eps**3, eps, 0)-2/(3*alpha),
        "endpoint_potential_quartic": sp.limit(V.subs(q, sp.pi/2-eps)/eps**4, eps, 0)+rc,
        "endpoint_first_derivative_zero": sp.limit(Vp, q, sp.pi/2, dir="-"),
        "endpoint_second_derivative_diverges_negative": sp.limit(x*Vpp, q, sp.pi/2, dir="-")+3/(4*M**2),
    })
    # Exact canonical bridge on any invertible J>0 branch; no f' or H
    # division. Integration by parts: -volume*Psidot -> Psi*volumedot.
    volume, volume_dot = sp.symbols("volume volume_dot", positive=True)
    p = sp.symbols("p", real=True)
    f = sp.Function("f")
    U = sp.Function("U")
    bridge_field = p-2*M**2*sp.diff(f(p), p)/3
    bridge_potential = M**2*sp.diff(f(p), p)**2/3-f(p)
    J = sp.diff(bridge_field, p)
    reduced_L = -M**2*volume_dot**2/(3*lapse*volume)+psi*volume_dot-lapse*volume*U(psi)
    momentum_volume = sp.diff(reduced_L, volume_dot)
    solved_volume_dot = sp.solve(momentum_volume-p, volume_dot)[0]
    canonical_gravity = sp.simplify(((p*volume_dot-reduced_L)/(lapse*volume)).subs(
        volume_dot, solved_volume_dot))
    eliminated = canonical_gravity.subs(U(psi), bridge_potential).subs(psi, bridge_field)
    sine_function = rc*sp.sin(alpha*p)**2
    residuals.update({
        "canonical_bridge_chain_rule_without_fprime_division":
            sp.diff(bridge_potential, p)+sp.diff(f(p), p)*J,
        "canonical_bridge_legendre_constraint":
            canonical_gravity+3*(p-psi)**2/(4*M**2)-U(psi),
        "canonical_bridge_exact_functional_elimination": eliminated+f(p),
        "canonical_bridge_sine_J_positive":
            (1-2*M**2*sp.diff(sine_function, p, 2)/3)-2*sine_function/rc,
        "canonical_bridge_sine_potential":
            M**2*sp.diff(sine_function, p)**2/3-sine_function+sine_function**2/rc,
    })
    result = {}
    for name, residual in residuals.items():
        value = sp.trigsimp(sp.simplify(residual))
        result[name] = {"passed": bool(value == 0), "residual": str(value)}
    return result


def _perturbation_polynomials(x, K):
    A = x*(1-x)/3
    # Positive form of 3+7*x-8*x^2 on 0<x<=1 avoids cancellation.
    N = A*K*K+2*x*x*(2*x+1)*K+x**3*(2+(1-x)*(8*x+1))
    D = A*K*K+2*x*x*K+3*x**3*(x+1)
    kinetic = 3*(K+3*x)/(K*(1-x)+3*x*(x+1))
    return N, D, kinetic


def scalar_massless_coefficients(x, K):
    """w=1 benchmark ONLY; K=(k/a)^2/(rho_c/Mpl^2), no causality claim."""
    x = finite_number(x, "x", positive=True)
    K = finite_number(K, "K", nonnegative=True)
    if x > 1:
        raise ValueError("x must lie in (0, 1]")
    N, D, kinetic = _perturbation_polynomials(x, K)
    result = {"x": x, "K": K, "N": N, "D": D,
              "cs2": N/D, "kinetic": kinetic}
    for name, value in result.items():
        finite_number(value, name)
    if D <= 0 or N <= 0 or kinetic <= 0:
        raise ArithmeticError("benchmark positivity violated")
    result["superluminal_phase_speed"] = bool(result["cs2"] > 1)
    return result


def perturbation_symbolic_checks(*, h_hddot_sign=-1):
    """Reconstruct Eqs.40,42-43 from background derivatives, not copied N/D.

    Eq.43 has -H*Hddot in A2. Flipping that sign is a negative control,
    independently rejected by the physical GR limit x->0, K=s*x.
    """
    if isinstance(h_hddot_sign, bool) or h_hddot_sign not in (-1, 1):
        raise ValueError("h_hddot_sign must be -1 or +1")
    x, K, ratio = sp.symbols("x K ratio", positive=True)
    H, e = sp.symbols("H e", real=True)
    A = x*(1-x)/3  # H^2/e, e=rho_c/Mpl^2.
    xdot = -6*H*x  # Independent w=1 continuity.
    Hdot = e*x*(2*x-1)
    HHddot = sp.expand(H*sp.diff(Hdot, x)*xdot).subs(H**2, e*A)/e**2
    Y, hd = x, Hdot/e
    A2 = Y*(12*A+3*hd+Y)+2*hd**2+h_hddot_sign*HHddot
    A0 = Y**2*(15*A+hd-Y)-Y*(12*A*hd-2*hd**2+3*HHddot)
    B2 = Y*(6*A+hd+Y)
    B0 = 3*Y**2*(3*A+hd+Y)
    N = sp.expand(A*K**2+A2*K+A0)
    D = sp.expand(A*K**2+B2*K+B0)
    kinetic = Y*(K+3*Y)/(K*A+Y*(3*A+hd+Y))
    expected_N = A*K**2+2*x**2*(2*x+1)*K+x**3*(3+7*x-8*x**2)
    expected_D = A*K**2+2*x**2*K+3*x**3*(x+1)
    lower_positive = 2*A*K**2+4*x**2*(3*x+1)*K+6*x**3*(1-x)*(4*x+1)
    upper_positive = 2*A*K**2+4*x**2*(1-x)*K+2*x**3*(4*x**2+x+3)
    residuals = {
        "HHddot_from_continuity": HHddot+6*A*x*(4*x-1),
        "A2_from_source_eq43": A2-2*x**2*(2*x+1),
        "A0_from_source_eq43": A0-x**3*(3+7*x-8*x**2),
        "B2_from_source_eq43": B2-2*x**2,
        "B0_from_source_eq43": B0-3*x**3*(x+1),
        "sound_numerator_from_source": N-expected_N,
        "sound_denominator_from_source": D-expected_D,
        "kinetic_from_source_eq40": kinetic-3*(K+3*x)/(K*(1-x)+3*x*(x+1)),
        "lower_bound_positive_decomposition": 3*N-D-lower_positive,
        "upper_bound_strict_positive_decomposition": 3*D-N-upper_positive,
        "bounce_sound_speed": (N/D).subs(x, 1)-(3*K+1)/(K+3),
        "bounce_superluminal_threshold": ((N-D)/D).subs(x, 1)-2*(K-1)/(K+3),
        "physical_GR_limit_K_proportional_x": sp.limit((N/D).subs(K, ratio*x), x, 0)-1,
    }
    return {name: {"passed": bool(sp.simplify(value) == 0),
                   "residual": str(sp.factor(value))}
            for name, value in residuals.items()}


def numerical_derivative_check(q, rho_c=1, Mpl=1, enthalpy=1):
    """Independent mp.diff of the parametric V and Psi; no analytic derivative input."""
    with mp.workdps(80):
        point = parametric_state(q, rho_c, Mpl, enthalpy)
        q = point["q"]
        phi = lambda z: parametric_state(z, rho_c, Mpl, enthalpy)["Psi"]
        pot = lambda z: parametric_state(z, rho_c, Mpl, enthalpy)["V"]
        v1, v2 = mp.diff(pot, q), mp.diff(pot, q, 2)
        p1, p2 = mp.diff(phi, q), mp.diff(phi, q, 2)
        gradient = v1/p1
        curvature = (v2*p1-v1*p2)/p1**3
        errors = {
            "gradient_error": abs(gradient-point["V_Psi"])/max(1, abs(gradient)),
            "curvature_error": abs(curvature-point["V_PsiPsi"])/max(1, abs(curvature)),
            "jacobian_error": abs(p1-point["Psi_q"])/max(1, abs(p1)),
        }
        for name, value in errors.items():
            finite_number(value, name)
        errors["passed"] = bool(max(errors.values()) < mp.mpf("1e-65"))
        return errors


def scalar_massless_transfer(kappa, *, tau_start=-5, tau_end=5,
                             rtol=1e-10, atol=1e-12, max_step=0.04):
    """Classical two-basis transfer across a finite interval, not a spectrum.

    tau=t*sqrt(3*rho_c/Mpl^2), x=1/(1+tau^2), a_b=1.  Pi is
    b*zeta', b=a^3*kinetic. The system is regular at the bounce and folds.
    """
    import numpy as np
    from scipy.integrate import solve_ivp

    values = {}
    for name, value in (("kappa", kappa), ("tau_start", tau_start),
                        ("tau_end", tau_end), ("rtol", rtol), ("atol", atol),
                        ("max_step", max_step)):
        checked = finite_number(value, name, positive=name in ("rtol", "atol", "max_step"),
                                nonnegative=name == "kappa")
        values[name] = float(checked)
        if not math.isfinite(values[name]) or (checked != 0 and values[name] == 0):
            raise ValueError(f"{name} is outside float range required by solve_ivp")
    if values["tau_start"] >= values["tau_end"]:
        raise ValueError("tau_start must precede tau_end")
    if values["rtol"] < 3e-14 or values["atol"] < 1e-300:
        raise ValueError("requested solver tolerance is outside the supported float range")

    rhs_calls = 0

    def rhs(tau, state):
        nonlocal rhs_calls
        rhs_calls += 1
        if rhs_calls > 200000:
            raise ArithmeticError("finite-interval transfer exceeded the bounded evaluation budget")
        x = 1/(1+tau*tau)
        a = (1+tau*tau)**(1/6)
        K = (values["kappa"]/a)**2
        N, D, kinetic = _perturbation_polynomials(x, K)
        b = a**3*kinetic
        frequency = N/D*K/3
        result = (state[1]/b, -b*frequency*state[0],
                  state[3]/b, -b*frequency*state[2])
        if not all(math.isfinite(v) for v in result):
            raise ArithmeticError("nonfinite perturbation evolution")
        return result

    try:
        sol = solve_ivp(rhs, (values["tau_start"], values["tau_end"]),
                        (1., 0., 0., 1.), method="DOP853", rtol=values["rtol"],
                        atol=values["atol"], max_step=values["max_step"])
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError("transfer inputs exceed finite float arithmetic") from exc
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise ArithmeticError("finite-interval perturbation integration failed")
    z1, p1, z2, p2 = sol.y[:, -1]
    matrix = [[float(z1), float(z2)], [float(p1), float(p2)]]
    determinant = float(z1*p2-z2*p1)
    if not math.isfinite(determinant):
        raise ArithmeticError("nonfinite transfer determinant")
    return {"kappa": values["kappa"], "interval": [values["tau_start"], values["tau_end"]],
            "matrix": matrix, "determinant_error": abs(determinant-1),
            "rhs_evaluations": int(sol.nfev), "finite": True,
            "scope": "canonical_massless_scalar_classical_basis_transfer_only"}


def transfer_convergence(kappa):
    coarse = scalar_massless_transfer(kappa, rtol=1e-8, atol=1e-10, max_step=0.15)
    fine = scalar_massless_transfer(kappa, rtol=1e-10, atol=1e-12, max_step=0.04)
    error = max(abs(coarse["matrix"][i][j]-fine["matrix"][i][j]) /
                max(1, abs(fine["matrix"][i][j])) for i in range(2) for j in range(2))
    return {"coarse": coarse, "fine": fine, "relative_matrix_difference": error,
            "passed": bool(error < 1e-6 and fine["determinant_error"] < 1e-7)}


def _json_values(value):
    if isinstance(value, mp.mpf):
        return mp.nstr(finite_number(value, "output"), 35)
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ArithmeticError("nonfinite JSON output")
    return value


def _symbolic_results_passed(rows):
    """Fail closed on missing checks, truthy non-booleans or nonzero residuals."""
    return (isinstance(rows, dict) and bool(rows) and
            all(isinstance(row, dict) and row.get("passed") is True and
                row.get("residual") == "0" for row in rows.values()))


def compute_state(q="0", rho_c="1", Mpl="1", enthalpy="1", *, include_transfer=True):
    """Pure report; defaults are dimensionless demonstration inputs, not a fit."""
    if not isinstance(include_transfer, bool):
        raise ValueError("include_transfer must be bool")
    with mp.workdps(80):
        point = parametric_state(q, rho_c, Mpl, enthalpy)
        background = symbolic_checks()
        perturbations = perturbation_symbolic_checks()
        derivative = numerical_derivative_check(q, rho_c, Mpl, enthalpy)
        transfers = [transfer_convergence(k) for k in (0, 0.1, 1, 10)] if include_transfer else []
        all_passed = (_symbolic_results_passed(background) and
                      _symbolic_results_passed(perturbations) and
                      isinstance(derivative, dict) and derivative.get("passed") is True and
                      all(isinstance(row, dict) and row.get("passed") is True
                          for row in transfers))
        return _json_values({
            "status": STATUS, "evidentiary_weight": 0,
            "mathematical_checks_passed": bool(all_passed), "point": point,
            "symbolic_background": background, "symbolic_w1_perturbations": perturbations,
            "independent_numeric_derivatives": derivative, "w1_transfer_checks": transfers,
            "scope": {
                "matter": "separately_conserved_positive_enthalpy; full_NVG_dynamics_not_validated",
                "geometry": "flat_FLRW_timelike_monotonic_Psi_open_field_interval",
                "extension": "added_negative_sign_cuscuton; not_derived_from_accepted_NVG_action",
                "parameters": "rho_c_is_new_input; mu_is_field_normalization_without_other_couplings",
                "canonical_bridge": "exact_minus_f_constraint_after_elimination_on_J>0_branch; not_NVG_derivation",
                "folds": "rho=rho_c/2_regular; V_PsiPsi=0_but_constraint_determinant=3",
                "vacuum_endpoints": "finite_field_endpoints_are_C1_not_C2; X=0_completion_not_provided",
                "linear_benchmark": "w=1_canonical_massless_scalar_only; kinetic>0_and_1/3<=cs2<3",
                "causality": "cs2_at_bounce>1_for_K>1; no_UV_causality_or_full_NVG_stability_claim",
                "excluded": "quantum_state_spectrum_strong_coupling_anisotropy_black_hole_core_and_cycle_count",
            }, "sources": SOURCES,
        })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", default="0")
    parser.add_argument("--rho-c", default="1")
    parser.add_argument("--Mpl", default="1")
    parser.add_argument("--enthalpy", default="1")
    parser.add_argument("--skip-transfer", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = compute_state(args.q, args.rho_c, args.Mpl, args.enthalpy,
                               include_transfer=not args.skip_transfer)
    except (ValueError, ArithmeticError) as exc:
        print(json.dumps({"status": "invalid_or_failed_audit", "evidentiary_weight": 0,
                          "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["mathematical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
