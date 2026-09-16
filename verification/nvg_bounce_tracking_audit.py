#!/usr/bin/env python3
"""Fixed-Newton-constant audit of a quasistatic NVG trial configuration.

This is an instantaneous jet/residual calculation, NOT an integrated full
trajectory, an error theorem, or a prediction of rho_c. Original BulkModel
parameters are used unchanged. x=rho_trial/rho_c includes the trial kinetic
energy. Hdot is assigned by Raychaudhuri at that jet, not obtained by
differentiating an arbitrary sequence of state(n_ratio,x) calls.

Units: hbar=c=1, MeV. The scalar gap is C=E_WW, not E_yy.
The homogeneous cold positive-energy fermion closure remains an assumption.
The CLI prints JSON only; no result files are written.
"""
from __future__ import annotations

import argparse
import json

import mpmath as mp
import sympy as sp

from source_complete_scaling_saturation_audit import BulkModel, INPUTS


G_NEWTON_MEV_MINUS2 = "6.70883e-45"
STATUS = "QUASISTATIC_TRIAL_DEFECT_AND_LEADING_CORRECTION_NOT_FULL_TRAJECTORY"
EXPECTED_SYMBOLIC_CHECKS = frozenset({
    "stationary_trial_residual_from_chain_rule",
    "leading_algebraic_correction_cancels_residual",
    "kinetic_completed_friedmann_constraint",
    "kinetic_completed_fraction",
    "generic_scalar_stress_identity",
    "stationary_trial_stress_completion",
    "constraint_propagation_defect",
    "bounce_trial_velocity_zero",
    "bounce_residual_is_finite_and_generally_nonzero",
    "bounce_leading_correction",
})


def finite_number(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        result = mp.mpf(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not mp.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _precision(dps):
    if isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 200:
        raise ValueError("dps must be an integer between 80 and 200")


def _relative_difference(a, b, scale):
    return abs(a-b)/max(abs(a), abs(b), scale)


def _stationary_jet(model, n):
    """Implicit derivatives from independently analytic third derivatives."""
    try:
        s = model.equilibrium(n)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("positive locally stable stationary branch could not be resolved") from exc
    for key in ("y", "C_y", "energy_total", "mu"):
        finite_number(s[key], key, positive=True)
    y, mass, ef, k = s["y"], s["m"], s["ef"], s["k"]
    # F=E_y: F_n=B_y, F_y=C_y. These are partial derivatives, not
    # derivatives along a fitted or numerically tabulated equation of state.
    ns_mm = 3*s["ns_m"]/mass-3*s["ns"]/mass**2+3*n*mass/ef**3
    F_nn = -model.MN**2*y*k**2/(3*n*ef**3)-2*model.Cv/y**3
    F_ny = model.MN**2*k**2/ef**3+6*model.Cv*n/y**4
    F_yy = 6*model.A*y+model.MN**3*ns_mm-12*model.Cv*n*n/y**5
    yn = -s["B_y"]/s["C_y"]
    ynn = -(F_nn+2*F_ny*yn+F_yy*yn*yn)/s["C_y"]
    result = {
        "n": n, "y": y, "W": model.W0*y,
        "E": s["energy_total"], "P_star": s["pressure_total"], "mu": s["mu"],
        "Q_star": n*s["mu"], "C": s["C_y"]/model.W0**2,
        "B": s["B_y"]/model.W0, "Wn": model.W0*yn, "Wnn": model.W0*ynn,
        "C_n_along_branch": (F_ny+F_yy*yn)/model.W0**2,
        "stationary_force": s["residual"]/model.W0,
        "E_Wnn": F_nn/model.W0, "E_WWn": F_ny/model.W0**2,
        "E_WWW": F_yy/model.W0**3,
    }
    for name, value in result.items():
        finite_number(value, name)
    return result


def _independent_jet_check(model, jet):
    """Differentiate newly solved equilibria and integrate the scalar curvature."""
    n, y = jet["n"], jet["y"]
    branch = lambda density: model.equilibrium(density, (y*mp.mpf(".999"), y*mp.mpf("1.001")))["y"]
    Wn = model.W0*mp.diff(branch, n)
    Wnn = model.W0*mp.diff(branch, n, 2)
    k = (6*mp.pi**2*n/model.d)**(mp.mpf(1)/3)
    mass = model.MN*y
    ns_m_integral = model.d/(2*mp.pi**2)*mp.quad(
        lambda z: z**4/(z*z+mass*mass)**mp.mpf("1.5"), [0, k])
    C_integral = (model.A*(3*y*y-1)+model.MN**2*ns_m_integral+
                  3*model.Cv*n*n/y**4)/model.W0**2
    residuals = {
        "Wn_relative_error": _relative_difference(Wn, jet["Wn"], model.W0/n),
        "Wnn_relative_error": _relative_difference(Wnn, jet["Wnn"], model.W0/n**2),
        "gap_quadrature_relative_error": _relative_difference(C_integral, jet["C"], mp.mpf(1)),
        "stationarity_scaled_residual": abs(jet["stationary_force"]*jet["W"])/jet["E"],
    }
    for name, value in residuals.items():
        finite_number(value, name)
    return {**residuals, "passed": bool(max(residuals.values()) < mp.mpf("1e-60"))}


def state(n_ratio="1", x="1", *, dps=80):
    """Compute a trial jet at fixed physical GN; no adjustable gravity argument.

    x=0 denotes the formal GR limit only: rho_c is None, not infinity in
    a numeric passport. Uniform bounds additionally require eta<1 so the
    kinetic-completed trial constraint is nonsingular for every x in [0,1].
    """
    _precision(dps)
    with mp.workdps(dps):
        n_ratio = finite_number(n_ratio, "n_ratio", positive=True)
        x = finite_number(x, "x")
        if not 0 <= x <= 1:
            raise ValueError("x must lie in [0,1]")
        model = BulkModel()
        jet = _stationary_jet(model, n_ratio*model.n0)
        independent = _independent_jet_check(model, jet)
        if independent["passed"] is not True:
            raise ArithmeticError("independent stationary-branch derivative check failed")
        M2 = 1/(8*mp.pi*mp.mpf(G_NEWTON_MEV_MINUS2))
        n, W, C, E, Q = (jet[key] for key in ("n", "W", "C", "E", "Q_star"))
        Wn, Wnn = jet["Wn"], jet["Wnn"]
        eta = 3*n*n*Wn*Wn/(2*M2)
        denominator = 1-eta*(1-x)
        if denominator <= 0 or eta >= 1:
            raise ValueError("trial kinetic denominator or uniform-in-x bound is not positive")
        H2 = E*(1-x)/(3*M2*denominator)
        H = mp.sqrt(H2)  # Expanding jet; contraction has H,Wdot,RhoDot opposite.
        Wdot = -3*H*n*Wn
        Ktrack = Wdot**2/2
        rho, pressure = E+Ktrack, jet["P_star"]+Ktrack
        enthalpy = Q+2*Ktrack
        Hdot = -enthalpy*(1-2*x)/(2*M2)
        R = 9*H2*n*n*Wnn-3*Hdot*n*Wn
        deltaW = -R/C
        m = mp.sqrt(C)
        ndot = -3*H*n
        H2max = E/(3*M2*(1-eta))
        Kmax = E*eta/(1-eta)
        Hdotmax = (Q+2*Kmax)/(2*M2)
        Rmax = 9*H2max*n*n*abs(Wnn)+3*Hdotmax*n*abs(Wn)
        numeric = {
            **jet, "n_ratio": n_ratio, "x": x, "G_Newton": mp.mpf(G_NEWTON_MEV_MINUS2),
            "Mpl": mp.sqrt(M2), "Mpl_over_W0": mp.sqrt(M2)/model.W0,
            "gap_MeV": m, "eta": eta, "kinetic_denominator": denominator,
            "H": H, "H_squared": H2, "Hdot_assigned_Raychaudhuri": Hdot,
            "Wdot_trial": Wdot, "Ktrack": Ktrack,
            "rho_trial": rho, "P_trial": pressure, "enthalpy_trial": enthalpy,
            "R_exact_trial_defect": R, "deltaW_leading": deltaW,
            "H_over_gap": abs(H)/m, "sqrt_abs_Hdot_over_gap": mp.sqrt(abs(Hdot))/m,
            "abs_R_over_CW": abs(R)/(C*W), "deltaW_over_W": deltaW/W,
            "Ktrack_over_E": Ktrack/E,
            "gap_adiabatic_ratio": abs(jet["C_n_along_branch"]*ndot)/(2*C**mp.mpf("1.5")),
            "trial_drift_over_gap_W": abs(Wdot)/(m*W),
            "stress_continuity_defect": Wdot*R,
            "friedmann_constraint_time_defect": -(1-2*x)*Wdot*R,
            "H_squared_leading_without_Ktrack": E*(1-x)/(3*M2),
        }
        for name, value in numeric.items():
            finite_number(value, name)
        bounds = {
            "H_squared_max": H2max, "Ktrack_max": Kmax,
            "abs_Hdot_max": Hdotmax, "abs_R_max": Rmax,
            "H_over_gap_max": mp.sqrt(H2max/C),
            "sqrt_abs_Hdot_over_gap_max": mp.sqrt(Hdotmax/C),
            "abs_R_over_CW_max": Rmax/(C*W), "Ktrack_over_E_max": eta/(1-eta),
        }
        for name, value in bounds.items():
            finite_number(value, name)
        return {"numeric": numeric, "uniform_x_bounds": bounds,
                "rho_c_conditional_input": None if x == 0 else rho/x,
                "rho_c_status": "GR_limit_not_finite_rho_c" if x == 0 else
                                "inferred_from_declared_trial_point_NOT_NVG_prediction",
                "independent_derivatives": independent}


def symbolic_checks(*, residual_hdot_sign=-1, correction_sign=-1, kinetic_sign=1):
    for name, sign in (("residual_hdot_sign", residual_hdot_sign),
                       ("correction_sign", correction_sign), ("kinetic_sign", kinetic_sign)):
        if isinstance(sign, bool) or sign not in (-1, 1):
            raise ValueError(f"{name} must be +1 or -1")
    n, M2, E, C = sp.symbols("n Mpl2 E C", positive=True)
    H, hd, x, Q = sp.symbols("H Hdot x Q", real=True)
    w = sp.Function("Wstar")
    wn, wnn = sp.diff(w(n), n), sp.diff(w(n), n, 2)
    ndot = -3*H*n
    u = wn*ndot
    udot = sp.diff(u, n)*ndot+sp.diff(u, H)*hd
    R_derived = udot+3*H*u
    R_trial = 9*H**2*n**2*wnn+residual_hdot_sign*3*hd*n*wn
    eta = 3*n*n*wn**2/(2*M2)
    h2_completed = E*(1-x)/(3*M2*(1-eta*(1-x)))
    k_completed = kinetic_sign*sp.Rational(9, 2)*h2_completed*n*n*wn*wn
    # Differentiate generic full trial stress first, then impose E_W=0.
    en, ew, scalar_u, scalar_a = sp.symbols("En Ew scalar_u scalar_a", real=True)
    rho_dot = scalar_u*scalar_a+en*ndot+ew*scalar_u
    total_enthalpy = scalar_u**2+n*en
    stress_defect = rho_dot+3*H*total_enthalpy
    jet_stress_defect = stress_defect.subs({ew: 0, scalar_u: u, scalar_a: udot})
    hd_from_ray = -Q*(1-2*x)/(2*M2)
    rho_dot_with_defect = -3*H*Q+u*R_derived
    constraint_derivative = 6*M2*H*hd_from_ray-(1-2*x)*rho_dot_with_defect
    residuals = {
        "stationary_trial_residual_from_chain_rule": R_derived-R_trial,
        "leading_algebraic_correction_cancels_residual": R_derived+C*correction_sign*R_trial/C,
        "kinetic_completed_friedmann_constraint": 3*M2*h2_completed-(E+k_completed)*(1-x),
        "kinetic_completed_fraction": k_completed/E-eta*(1-x)/(1-eta*(1-x)),
        "generic_scalar_stress_identity": stress_defect-scalar_u*(scalar_a+3*H*scalar_u+ew),
        "stationary_trial_stress_completion": jet_stress_defect-u*R_trial,
        "constraint_propagation_defect": constraint_derivative+(1-2*x)*u*R_trial,
        "bounce_trial_velocity_zero": u.subs(H, 0),
        "bounce_residual_is_finite_and_generally_nonzero":
            R_trial.subs({H: 0, hd: Q/(2*M2)})+3*n*wn*Q/(2*M2),
        "bounce_leading_correction":
            (correction_sign*R_trial/C).subs({H: 0, hd: Q/(2*M2)})-3*n*wn*Q/(2*M2*C),
    }
    return {name: {"passed": bool(sp.simplify(value) == 0),
                   "residual": str(sp.simplify(value))}
            for name, value in residuals.items()}


def _symbolic_passed(rows):
    return (isinstance(rows, dict) and set(rows) == EXPECTED_SYMBOLIC_CHECKS and
            all(isinstance(row, dict) and row.get("passed") is True and
                row.get("residual") == "0" for row in rows.values()))


def _shown(value):
    if isinstance(value, mp.mpf):
        # Do not reconstruct mpf outside workdps: that would round the
        # already high-precision value to the ambient 15-digit context.
        if not mp.isfinite(value):
            raise ArithmeticError("nonfinite output value")
        return mp.nstr(value, 45)
    if isinstance(value, dict):
        return {key: _shown(item) for key, item in value.items()}
    return value


def compute_state(n_ratio="1", x="1"):
    """Compare 80/120 digits and independent derivatives; return JSON-safe data."""
    coarse = state(n_ratio, x, dps=80)
    fine = state(n_ratio, x, dps=120)
    with mp.workdps(120):
        differences = {
            key: abs(coarse["numeric"][key]-value)/max(abs(value), mp.mpf("1e-100"))
            for key, value in fine["numeric"].items() if key != "stationary_force"
        }
        precision_error = max(differences.values())
        precision_passed = bool(precision_error < mp.mpf("1e-55"))
    checks = symbolic_checks()
    passed = (_symbolic_passed(checks) and precision_passed is True and
              coarse["independent_derivatives"].get("passed") is True and
              fine["independent_derivatives"].get("passed") is True)
    return _shown({
        "status": STATUS, "evidentiary_weight": 0,
        "mathematical_checks_passed": bool(passed), "original_matter_inputs": INPUTS.copy(),
        "point": fine, "symbolic_checks": checks,
        "precision_check": {"dps": "80_vs_120", "max_relative_difference": precision_error,
                            "passed": precision_passed},
        "scope": {
            "model": "original_source_complete_matter_plus_separately_reconstructed_cuscuton_gravity",
            "geometry": "flat_FLRW; expanding_jet; contraction_flips_H_and_Wdot_not_R_or_Ktrack",
            "x": "rho_trial/rho_c_including_Ktrack; rho_c_is_NOT_determined_by_NVG",
            "trial": "W=Wstar(n), E_W=0; generally_NOT_an_exact_scalar_trajectory",
            "Hdot": "assigned_Raychaudhuri_at_the_jet_NOT_derivative_of_arbitrary_state_line",
            "correction": "deltaW_leading=-R/C; formal_slow_variation_correction_NOT_validated_error_bound",
            "uniform_bounds": "pointwise_in_n_for_all_x_in_[0,1]_when_eta<1; NOT_global_trajectory_bounds",
            "bounce": "H=0_and_Wdot_trial=0_do_NOT_imply_R=0; no_division_by_H",
            "constraint": "pointwise_Friedmann_and_Raychaudhuri_do_NOT_establish_constraint_propagation",
            "tracking_conditions": "positive_gap; slow_gap_and_background_variation; small_nonlinearity; prepared_small_initial_free_oscillations",
            "initial_oscillations": "not_fixed_by_action_or_large_Mpl; contraction_can_amplify_free_modes",
            "closure": "homogeneous_instantaneous_cold_positive_energy_fermion_closure_NOT_microscopically_validated",
            "excluded": "full_trajectory_stability_arbitrary_initial_conditions_rho_c_prediction_and_observational_confirmation",
        },
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ratio", default="1")
    parser.add_argument("--x", default="1")
    args = parser.parse_args(argv)
    try:
        result = compute_state(args.n_ratio, args.x)
    except (ValueError, ArithmeticError) as exc:
        print(json.dumps({"status": "INVALID_OR_FAILED_TRACKING_AUDIT", "error": str(exc),
                          "evidentiary_weight": 0}, allow_nan=False))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["mathematical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
