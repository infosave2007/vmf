#!/usr/bin/env python3
"""Potential-independent homogeneous and curved-action bounce audit.

This module is a deliberately scoped mathematical audit.  It does not change
the accepted source-complete action, parameters, or saved research products.
The first part repeats the flat homogeneous continuation argument for a
non-negative C2 coercive potential and keeps the full ``E_W`` force in the
constraint identity.  The second part varies the *same* negative-branch
cuscuton action with a curved FLRW lapse and derives the curvature terms.

The optional closed-universe calculation uses the original stationary
one-component ``BulkModel`` EOS only as a declared mathematical control.  Its
q-periodic continuation crosses the finite-field endpoint where the
cuscuton action degenerates, so it is reported as a formal trajectory and
never as a healthy cyclic universe or an NVG prediction.  The CLI prints JSON
and writes no output files.

Conventions are +---, ``M2=(8*pi*G_Newton)**-1``,
``X=(partial chi)^2/2>0``, and the cuscuton addition is
``-mu2*sqrt(2X)-V(chi)`` on the increasing timelike branch.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mpmath as mp
import sympy as sp

from source_complete_scaling_saturation_audit import BulkModel
from nvg_binding_feasibility_audit import BindingDesign, INPUTS as BINDING_INPUTS


STATUS = "potential_and_curved_action_audit_not_NVG_completion"
EVIDENTIARY_WEIGHT = 0
GN_INPUT = "6.70883e-45"  # MeV^-2; unchanged declared input in prior audits.
SOURCE_PATH = Path(__file__).resolve()
SOURCES = {
    "accepted_flat_reconstruction": "verification/covariant_bounce_completion_audit.py",
    "accepted_homogeneous_bounds": "verification/nvg_bounce_global_structure_audit.py",
    "accepted_original_eos": "verification/source_complete_scaling_saturation_audit.py",
    "fixed_binding_design": "verification/nvg_binding_feasibility_audit.py",
    "action_contract": "verification/contracts/source_complete_action.md",
    "cuscuton_bounce": "https://arxiv.org/abs/1802.06818",
    "cuscuton_perturbations": "https://arxiv.org/html/1911.06040v2",
    "cuscuton_reconstruction_context": "https://arxiv.org/abs/2405.08071",
}


def finite(value, name, *, positive=False, nonnegative=False):
    """Return an mp number after rejecting booleans, NaN and infinities."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        number = mp.mpf(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not mp.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def decimal(value):
    return mp.nstr(finite(value, "derived value"), 35)


def _json_values(value):
    if isinstance(value, mp.mpf):
        return decimal(value)
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_values(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ArithmeticError("nonfinite JSON output")
    return value


def _symbolic_rows_pass(rows, required=None):
    """Fail closed on absent checks, non-boolean status, or nonzero residuals."""
    if not isinstance(rows, dict) or not rows:
        return False
    if required is not None and set(rows) != set(required):
        return False
    return all(isinstance(row, dict) and row.get("passed") is True
               and row.get("residual") == "0" for row in rows.values())


def _symbolic_residuals(expressions):
    """Reduce independently constructed symbolic equations to audit rows."""
    result = {}
    for name, expression in expressions.items():
        reduced = sp.trigsimp(sp.factor(sp.simplify(expression)))
        result[name] = {"residual": str(reduced),
                        "passed": bool(reduced == 0)}
    return result


def _finite_number(value):
    """Parse serialized evidence without accepting bool/NaN/Infinity."""
    if isinstance(value, bool):
        return None
    try:
        number = mp.mpf(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if mp.isfinite(number) else None


def _number_below(value, bound):
    number = _finite_number(value)
    return bool(number is not None and number < mp.mpf(bound))


def _abs_below(value, bound):
    number = _finite_number(value)
    return bool(number is not None and abs(number) < mp.mpf(bound))


def _relative_close(lhs, rhs, *scale_terms, tol="1e-24"):
    left = _finite_number(lhs)
    right = _finite_number(rhs)
    if left is None or right is None:
        return False
    scale = max(mp.mpf(1), abs(left), abs(right),
                *(abs(_finite_number(term)) for term in scale_terms
                  if _finite_number(term) is not None))
    return bool(abs(left-right) < mp.mpf(tol)*scale)


POTENTIAL_SYMBOLIC_REQUIRED = frozenset({
    "barrier_target_value", "barrier_target_slope",
    "barrier_target_curvature", "barrier_vacuum_value",
    "barrier_vacuum_slope", "barrier_vacuum_curvature",
    "barrier_tail_factorization", "tail_factor_derivative",
})
FLAT_SYMBOLIC_REQUIRED = frozenset({
    "matter_action_W_euler_lagrange", "matter_continuity_full_EW",
    "flat_constraint_propagation_full_EW",
    "full_force_cancellation", "W_force_equation_from_action",
    "vector_enthalpy_from_energy_derivative",
    "fermi_mass_derivative_integrand", "fermi_legendre_integrand_identity",
    "fermi_pressure_gap_integrand_identity",
    "flat_friedmann_without_sign_patch",
    "flat_raychaudhuri_from_regular_q",
})
CURVED_SYMBOLIC_REQUIRED = frozenset({
    "lapse_variation_friedmann", "cuscuton_EL_from_action",
    "scale_variation_raychaudhuri", "raychaudhuri_after_lapse_elimination",
    "candidate_potential_gradient", "candidate_cuscuton_EL_branch",
    "candidate_friedmann_curved", "candidate_raychaudhuri_curved",
    "candidate_constraint_derivative", "candidate_q0_H_zero",
    "candidate_q0_chi_q", "candidate_endpoint_chi_q_zero",
    "candidate_endpoint_chi_dot_zero", "candidate_endpoint_H_zero",
    "closed_F_derivative_from_conservation",
})
ENDPOINT_SYMBOLIC_REQUIRED = frozenset({
    "chi_endpoint_gap_cubic", "potential_endpoint_quartic",
    "potential_first_derivative_zero", "potential_second_derivative_diverges",
    "regularized_kinetic_combination", "regularized_LX_negative_asymptotic",
})
GR_SYMBOLIC_REQUIRED = frozenset({
    "flat_GR_raychaudhuri_solution", "flat_GR_equation_at_kappa_zero",
})


def coercive_potential(y, beta="1"):
    """A manufactured polynomial fixture, never the fixed BindingDesign U."""
    y = finite(y, "y", positive=True)
    beta = finite(beta, "beta", positive=True)
    return beta*(y*y-1)**4


def coercive_potential_derivatives(y, beta="1"):
    """Return U, dU/dy and d2U/dy2 without numerical differentiation."""
    y = finite(y, "y", positive=True)
    beta = finite(beta, "beta", positive=True)
    z = y*y-1
    return {
        "U": beta*z**4,
        "Uy": 8*beta*y*z**3,
        "Uyy": 8*beta*z**3+48*beta*y*y*z*z,
    }


def potential_symbolic_checks():
    """Exact identities for the fixed two-root barrier used by BindingDesign.

    Inequalities and envelope regularity are returned by the numerical
    ``actual_binding_design_certificate`` below.  They are deliberately not
    disguised as zero residuals in this symbolic group.
    """
    y, ys, beta = sp.symbols("y y_star beta", positive=True)
    z = sp.symbols("z", positive=True)
    barrier = beta*(y*y-1)**2*(y*y-ys*ys)**2
    tail_ratio = (z-1)**2*(z-ys*ys)**2/z**4
    rows = {
        "barrier_target_value": barrier.subs(y, ys),
        "barrier_target_slope": sp.diff(barrier, y).subs(y, ys),
        "barrier_target_curvature": sp.diff(barrier, y, 2).subs(y, ys)-
            8*beta*ys*ys*(ys*ys-1)**2,
        "barrier_vacuum_value": barrier.subs(y, 1),
        "barrier_vacuum_slope": sp.diff(barrier, y).subs(y, 1),
        "barrier_vacuum_curvature": sp.diff(barrier, y, 2).subs(y, 1)-
            8*beta*(ys*ys-1)**2,
        "barrier_tail_factorization": tail_ratio-
            ((z-1)/z)**4*((z-ys*ys)/(z-1))**2,
        "tail_factor_derivative": sp.diff((1-1/z)**4, z)-
            4*(z-1)**3/z**5,
    }
    return _symbolic_residuals(rows)


def manufactured_polynomial_fixture(*, beta="1", y="0.75"):
    """Return the old polynomial only as an explicit non-acceptance control."""
    be = finite(beta, "beta", positive=True)
    yy = finite(y, "y", positive=True)
    values = coercive_potential_derivatives(yy, be)
    return {
        "manufactured_polynomial_fixture": True,
        "used_for_actual_acceptance": False,
        "formula": "U_fixture=beta*(y^2-1)^4",
        "beta": decimal(be), "y": decimal(yy),
        **{key: decimal(value) for key, value in values.items()},
    }


def flat_symbolic_checks(*, scalar_force_sign=1, q_rate_scale=1):
    """Derive flat continuity/constraint identities with full ``E_W``.

    ``scalar_force_sign`` and ``q_rate_scale`` are deliberately wrong-model
    controls.  They are not physical options.  No scalar stationarity or
    on-shell ``E_W=0`` substitution is made in this derivation.
    """
    if isinstance(scalar_force_sign, bool) or scalar_force_sign not in (-1, 1):
        raise ValueError("scalar_force_sign must be -1 or +1")
    if isinstance(q_rate_scale, bool):
        raise ValueError("q_rate_scale must be a finite real number")
    scale = sp.Rational(str(finite(q_rate_scale, "q_rate_scale")))

    n, W, v, H, q = sp.symbols("n W v H q", positive=True, real=True)
    M, rc, Gv = sp.symbols("M rho_c Gv", positive=True)
    Efun = sp.Function("E")
    E = Efun(n, W)
    Ew = sp.diff(E, W)
    En = sp.diff(E, n)
    rho = v*v/2+E
    pressure = v*v/2+n*En-E
    Q = rho+pressure
    ndot = -3*H*n
    vdot = -3*H*v-scalar_force_sign*Ew
    rhodot = (sp.diff(rho, n)*ndot+sp.diff(rho, W)*v+
              sp.diff(rho, v)*vdot)
    alpha = sp.sqrt(3/(4*M*M*rc))
    Hq = alpha*rc*sp.sin(2*q)/3
    qdot = scale*alpha*Q

    # Independent Euler--Lagrange calculation for the actual proper-time
    # W action L=a^3*(Wdot^2/2-E(N/a^3,W)).
    a, ad, add, Wd, Wdd, Nbar = sp.symbols(
        "a adot addot Wdot Wdd Nbar", positive=True, real=True)
    n_action = Nbar/a**3
    E_action = Efun(n_action, W)
    L_action = a**3*(Wd**2/2-E_action)
    dL_dWd = sp.diff(L_action, Wd)
    d_dt_dL = (sp.diff(dL_dWd, a)*ad + sp.diff(dL_dWd, ad)*add +
               sp.diff(dL_dWd, W)*Wd + sp.diff(dL_dWd, Wd)*Wdd)
    action_el = (d_dt_dL-sp.diff(L_action, W))/a**3
    action_force = sp.diff(E_action, W).subs(n_action, n)

    # The Fermi identities below are local integrand identities.  Positivity
    # of their integrals is checked numerically in the fixed-design block.
    p, mass = sp.symbols("p mass", positive=True)
    ef = sp.sqrt(p*p+mass*mass)
    evec = Gv*n*n/(2*W*W)
    rows = {
        "matter_action_W_euler_lagrange": action_el.subs(
            {Nbar: a**3*n, Wd: v,
             Wdd: sp.Symbol("Wddot")})-
            (sp.Symbol("Wddot")+3*ad/a*v+action_force),
        "matter_continuity_full_EW": rhodot+3*H*Q,
        "flat_constraint_propagation_full_EW": rhodot.subs(H, Hq)+
            rc*sp.sin(2*q)*qdot,
        "full_force_cancellation":
            sp.expand(rhodot+3*H*Q),
        "W_force_equation_from_action": vdot+3*H*v+Ew,
        "vector_enthalpy_from_energy_derivative": n*sp.diff(evec, n)-
            Gv*n*n/W**2,
        "fermi_mass_derivative_integrand": sp.diff(ef, mass)-mass/ef,
        "fermi_legendre_integrand_identity": sp.diff(p**3*ef, p)-
            (3*p*p*ef+p**4/ef),
        "fermi_pressure_gap_integrand_identity": (
            3*p*p*ef-p**4/ef-p*p*(3*mass*mass+2*p*p)/ef),
        "flat_friedmann_without_sign_patch": Hq**2-
            rc*sp.cos(q)**2*(1-sp.cos(q)**2)/(3*M*M),
        "flat_raychaudhuri_from_regular_q": sp.diff(Hq, q)*qdot-
            Q*(2*sp.cos(q)**2-1)/(2*M*M),
    }
    return _symbolic_residuals(rows)


def flat_bound_certificate(*, rho_c="16", beta="1", n="0.5", Gv="1",
                           C_F="1", W0="1"):
    """Manufactured polynomial-fixture version of the flat continuation bounds.

    Inputs are mathematical control scales.  ``rho_c`` is not inferred from
    the matter action.  The potential bound uses the old polynomial fixture;
    it is intentionally not used for fixed BindingDesign acceptance.
    """
    rc = finite(rho_c, "rho_c", positive=True)
    be = finite(beta, "beta", positive=True)
    nn = finite(n, "n", positive=True)
    gv = finite(Gv, "Gv", nonnegative=True)
    cf = finite(C_F, "C_F", positive=True)
    w0 = finite(W0, "W0", positive=True)
    nmax = (rc/cf)**(mp.mpf(3)/4)
    y_tail = (16*rc/be)**(mp.mpf(1)/8)
    y_upper = max(mp.sqrt(2), y_tail)
    w_lower = nn*mp.sqrt(gv/(2*rc)) if gv else mp.mpf(0)
    # Equality at y=sqrt(2) verifies the chosen 1/16 constant; for larger y
    # monotonicity of (1-1/y^2)^4 proves the inequality.
    # Evaluate the exact algebraic endpoint in z=y^2, avoiding the roundoff
    # introduced by constructing sqrt(2) at finite precision.
    tail_ratio_at_2 = ((mp.mpf(2)-1)**4)/(mp.mpf(2)**4)
    return {
        "rho_c": decimal(rc), "beta": decimal(be), "n": decimal(nn),
        "C_F": decimal(cf), "Gv": decimal(gv), "W0": decimal(w0),
        "n_max": decimal(nmax), "y_tail_bound": decimal(y_tail),
        "y_upper_bound": decimal(y_upper),
        "W_lower_at_fixed_n": decimal(w_lower),
        "W_upper_bound": decimal(w0*y_upper),
        "W_lower_over_W0": decimal(w_lower/w0),
        "tail_constant_exact": bool(tail_ratio_at_2 == mp.mpf(1)/16),
        "n_bound_positive": bool(nmax > 0),
        "W_lower_positive_for_charge": bool(gv > 0 and w_lower > 0),
        "manufactured_polynomial_fixture": True,
        "used_for_actual_acceptance": False,
        "scope": "flat_constraint_rho<=rho_c; U>=0; conserved n>0; external rho_c",
    }


def _positive_fermi_integrals(n, y, design):
    """Evaluate the independent positive T=0 Fermi integrals.

    The BindingDesign helper is intentionally not used for this certificate:
    ``F``, ``P_F`` and ``n_s`` are recomputed from their positive momentum
    integrands.  This makes the Legendre and pressure-gap evidence sensitive
    to an action/EOS mutation rather than to a copied closed-form identity.
    """
    nn = finite(n, "n", nonnegative=True)
    yy = finite(y, "y", positive=True)
    d = mp.mpf(design.base.d)
    if nn == 0:
        return {"k": mp.mpf(0), "mass": design.base.MN*yy,
                "EF": design.base.MN*yy, "F": mp.mpf(0),
                "P_F": mp.mpf(0), "n_s": mp.mpf(0)}
    with mp.workdps(max(design.dps, mp.mp.dps)):
        k = (6*mp.pi**2*nn/d)**(mp.mpf(1)/3)
        mass = design.base.MN*yy
        # Rescale p=k*t only for numerical conditioning; every integrand is
        # the declared positive continuum integral.
        energy_integral = mp.quad(
            lambda t: t*t*mp.sqrt((k*t)**2+mass**2), [0, 1])
        pressure_integral = mp.quad(
            lambda t: t**4/mp.sqrt((k*t)**2+mass**2), [0, 1])
        scalar_integral = mp.quad(
            lambda t: t*t/mp.sqrt((k*t)**2+mass**2), [0, 1])
        F = d*k**3/(2*mp.pi**2)*energy_integral
        PF = d*k**5/(6*mp.pi**2)*pressure_integral
        ns = d*mass*k**3/(2*mp.pi**2)*scalar_integral
        return {"k": k, "mass": mass,
                "EF": mp.sqrt(k*k+mass*mass), "F": F,
                "P_F": PF, "n_s": ns}


def _actual_potential_jet(design, y):
    """Return U and its envelope-theorem derivative for fixed BindingDesign."""
    yy = finite(y, "y", positive=True)
    state = design.envelope_state(yy)
    nbar = state["n"]
    if nbar == 0:
        floor_y = mp.mpf(0)
        ns = mp.mpf(0)
    else:
        integrals = _positive_fermi_integrals(nbar, yy, design)
        ns = integrals["n_s"]
        floor_y = (-design.base.MN*ns + design.Cv*nbar*nbar/yy**3)
    z = yy*yy-1
    barrier = design.beta*z*z*(z-design.zs)**2
    barrier_y = 4*design.beta*yy*z*(z-design.zs)*(2*z-design.zs)
    return {"y": yy, "floor": state["value"], "barrier": barrier,
            "U": state["value"]+barrier, "floor_y": floor_y,
            "barrier_y": barrier_y, "Uy": floor_y+barrier_y,
            "envelope_n": nbar, "envelope_ns": ns}


def _actual_design_sample(design, y, n_ratio, *, velocity="0.37"):
    """Evaluate the full off-equilibrium W force and positive-energy bounds."""
    yy = finite(y, "sample y", positive=True)
    ratio = finite(n_ratio, "n_ratio", positive=True)
    vv = finite(velocity, "velocity")
    nn = ratio*design.n
    jet = _actual_potential_jet(design, yy)
    fermi = _positive_fermi_integrals(nn, yy, design)
    qphi = design.base.momega/design.base.W0
    Gv = design.gomega**2/(qphi*qphi)
    W = design.base.W0*yy
    vector = Gv*nn*nn/(2*W*W)
    direct_vector = design.Cv*nn*nn/(2*yy*yy)
    rho = vv*vv/2+fermi["F"]+jet["U"]+vector
    pressure = vv*vv/2+fermi["P_F"]-jet["U"]+vector
    Q = rho+pressure
    two_rho_minus_Q = 2*rho-Q
    Q_decomposition = vv*vv+nn*fermi["EF"]+Gv*nn*nn/(W*W)
    upper_bound_decomposition = fermi["F"]-fermi["P_F"]+2*jet["U"]
    # E_y is differentiated term-by-term from the actual matter action.
    Ey = (design.base.MN*fermi["n_s"]+jet["Uy"]-
          design.Cv*nn*nn/yy**3)
    EW = Ey/design.base.W0
    # A direct mp derivative of the fixed constructed U is a separate check
    # on the envelope-theorem force, not the value used to construct it.
    direct_Uy = mp.diff(design.potential, yy)
    direct_Ey = (design.base.MN*fermi["n_s"]+direct_Uy-
                 design.Cv*nn*nn/yy**3)
    return {
        "sample_id": f"y={mp.nstr(yy, 8)},n_ratio={mp.nstr(ratio, 8)}",
        "y": decimal(yy), "n_ratio": decimal(ratio), "n": decimal(nn),
        "velocity": decimal(vv), "W": decimal(W),
        "potential": decimal(jet["U"]), "floor": decimal(jet["floor"]),
        "barrier": decimal(jet["barrier"]),
        "envelope_density": decimal(jet["envelope_n"]),
        "F": decimal(fermi["F"]), "P_F": decimal(fermi["P_F"]),
        "n_s": decimal(fermi["n_s"]), "EF": decimal(fermi["EF"]),
        "vector_energy": decimal(vector),
        "vector_chain_residual": decimal(direct_vector-vector),
        "rho": decimal(rho), "pressure": decimal(pressure),
        "Q": decimal(Q), "two_rho_minus_Q": decimal(two_rho_minus_Q),
        "Q_decomposition": decimal(Q_decomposition),
        "Q_decomposition_residual": decimal(Q-Q_decomposition),
        "upper_bound_decomposition": decimal(upper_bound_decomposition),
        "upper_bound_decomposition_residual": decimal(
            two_rho_minus_Q-upper_bound_decomposition),
        "E_y": decimal(Ey), "E_W": decimal(EW),
        "direct_U_y": decimal(direct_Uy),
        "envelope_force_residual": decimal(jet["Uy"]-direct_Uy),
        "force_residual_relative": decimal((direct_Ey-Ey)/max(1, abs(Ey))),
        "fermi_legendre_residual": decimal(fermi["F"]+fermi["P_F"]-
                                            nn*fermi["EF"]),
        "fermi_pressure_gap": decimal(fermi["F"]-fermi["P_F"]),
        "finite": True,
        "potential_nonnegative": bool(jet["U"] >= 0),
        "rho_positive": bool(rho > 0),
        "Q_positive": bool(Q > 0),
        "Q_upper_bound": bool(Q <= 2*rho),
    }


def _actual_design_bounds(design, *, rho_c="1"):
    """Prove the fixed-design tail, density cap and vector lower bound."""
    rc = finite(rho_c, "rho_c", positive=True)
    ys2 = design.y*design.y
    z2 = mp.mpf(2)
    ratio_at_2 = ((z2-1)**2*(z2-ys2)**2)/(z2**4)
    C_F = mp.mpf(3)/4*(6*mp.pi**2/design.base.d)**(mp.mpf(1)/3)
    n_cap = (rc/C_F)**(mp.mpf(3)/4)
    qphi = design.base.momega/design.base.W0
    Gv = design.gomega**2/(qphi*qphi)
    fixed_n = design.n
    w_lower = fixed_n*mp.sqrt(Gv/(2*rc))
    y_tail = (16*rc/design.beta)**(mp.mpf(1)/8)
    y_upper = max(mp.sqrt(2), y_tail)
    # An actual U evaluation at the tail point includes the nonnegative
    # envelope floor; the barrier ratio is retained separately for the proof.
    tail_y = mp.sqrt(2)
    tail_jet = _actual_potential_jet(design, tail_y)
    tail_bound_gap = tail_jet["U"]-design.beta*tail_y**8/16
    vector_identity = Gv-design.Cv*design.base.W0**2
    return {
        "rho_c_control": decimal(rc), "C_F": decimal(C_F),
        "n_cap": decimal(n_cap), "n_cap_positive": bool(n_cap > 0),
        "massless_fermi_lower_bound":
            "F>=d*k_F^4/(8*pi^2)=C_F*n^(4/3), from sqrt(p^2+m^2)>=p",
        "massless_energy_gap_integrand_nonnegative": True,
        "density_cap_from_rho_le_rho_c": True,
        "envelope_definition": "U_floor(y)=sup_{n>=0}[mu*n-F(n,MN*y)-Cv*n^2/(2*y^2)]",
        "actual_potential_definition": "U_BindingDesign=U_floor(y)+beta*(y^2-1)^2*(y^2-y_star^2)^2",
        "zero_density_competitor_value": "0",
        "envelope_zero_density_competitor": True,
        "grand_density_hessian": "E_nn=k_F^2/(3*n*E_F)+Cv/y^2>0 for n>0,y>0",
        "strict_density_convexity_premise": True,
        "fixed_n": decimal(fixed_n), "Gv": decimal(Gv),
        "qphi": decimal(qphi), "vector_coefficient_identity": decimal(vector_identity),
        "vector_coefficient_identity_passed": bool(abs(vector_identity) <
                                                     mp.mpf("1e-65")*max(1, abs(Gv))),
        "fixed_n_W_lower": decimal(w_lower),
        "fixed_n_vector_bound_argument":
            "rho>=Gv*n^2/(2*W^2) and rho<=rho_c imply W>=n*sqrt(Gv/(2*rho_c))",
        "fixed_n_W_lower_positive": bool(w_lower > 0),
        "y_tail_from_U_bound": decimal(y_tail),
        "y_upper_bound": decimal(y_upper),
        "barrier_ratio_at_z2": decimal(ratio_at_2),
        "barrier_ratio_bound": decimal(mp.mpf(1)/16),
        "tail_bound_argument":
            "z>=2, y_star^2<=1: (z-y_star^2)^2>=(z-1)^2 and (1-1/z)^4>=1/16",
        "barrier_ratio_at_z2_ge_one_sixteenth": bool(ratio_at_2 >= mp.mpf(1)/16),
        "tail_factor_monotone_for_z_ge_2": True,
        "barrier_nonnegative_formula":
            "beta*(y^2-1)^2*(y^2-y_star^2)^2>=0 for beta>0",
        "barrier_nonnegative_factorization_premise": True,
        "ys2_le_one_premise": bool(ys2 <= 1),
        "beta_positive_premise": bool(design.beta > 0),
        "envelope_floor_nonnegative_premise": True,
        "tail_U_at_sqrt2": decimal(tail_jet["U"]),
        "tail_bound_gap": decimal(tail_bound_gap),
        "tail_U_bound_checked": bool(tail_bound_gap >= 0),
        "passed": bool(rc > 0 and C_F > 0 and n_cap > 0 and Gv > 0 and
                        abs(vector_identity) < mp.mpf("1e-65")*max(1, abs(Gv)) and
                        w_lower > 0 and
                        ratio_at_2 >= mp.mpf(1)/16 and ys2 <= 1 and
                        design.beta > 0 and tail_bound_gap >= 0),
        "scope": "fixed BindingDesign; rho_c is external; density cap assumes all other energy terms are nonnegative",
    }


def _envelope_regularity_certificate(design):
    """Check the actual envelope onset and its C2/not-C3 asymptotics."""
    onset = design.mu/design.base.MN
    coefficient_n = design.base.d*(2*design.mu)**(mp.mpf(3)/2)/(6*mp.pi**2)
    # P=d*k_F^5/(30*pi^2*m) and k_F^2=2*m*x, hence the leading grand
    # pressure is d*(2*mu)^(3/2)*x^(5/2)/(15*pi^2), not a fitted coefficient.
    coefficient_floor = design.base.d*(2*design.mu)**(mp.mpf(3)/2)/(15*mp.pi**2)
    deltas = (mp.mpf("1e-5"), mp.mpf("3e-6"), mp.mpf("1e-6"))
    samples = []
    for delta in deltas:
        yy = onset-delta
        state = design.envelope_state(yy)
        x = design.mu-design.base.MN*yy
        samples.append({
            "delta_y": decimal(delta), "x": decimal(x),
            "density_over_x_3_2": decimal(state["n"]/x**(mp.mpf(3)/2)),
            "floor_over_x_5_2": decimal(state["value"]/x**(mp.mpf(5)/2)),
            "floor_over_x_2": decimal(state["value"]/x**2),
            "floor_over_x_3": decimal(state["value"]/x**3),
            "finite": bool(mp.isfinite(state["n"]) and mp.isfinite(state["value"])),
        })
    at_onset = design.envelope_state(onset)
    below = samples[-1]
    n_ratio = _finite_number(below["density_over_x_3_2"])
    u_ratio = _finite_number(below["floor_over_x_5_2"])
    c2_values = [_finite_number(item["floor_over_x_2"]) for item in samples]
    c3_values = [_finite_number(item["floor_over_x_3"]) for item in samples]
    return {
        "onset_y": decimal(onset), "onset_density_zero": bool(at_onset["n"] == 0),
        "onset_floor_zero": bool(at_onset["value"] == 0),
        "onset_variable": "x=mu-MN*y down to 0",
        "leading_density_from_integral":
            "nbar=d*(2*mu)^(3/2)*x^(3/2)/(6*pi^2)+O(x^(5/2))",
        "leading_floor_from_pressure_integral":
            "U_floor=d*(2*mu)^(3/2)*x^(5/2)/(15*pi^2)+O(x^(7/2))",
        "vector_term_order": "Cv*nbar^2/(2*y^2)=O(x^3)",
        "regularity_basis": "positive Fermi integrands plus envelope theorem; C2 but not C3",
        "density_exponent": "3/2", "floor_exponent": "5/2",
        "density_leading_coefficient": decimal(coefficient_n),
        "floor_leading_coefficient_from_positive_pressure_integral": decimal(coefficient_floor),
        "below_onset_density_positive": bool(below["finite"] and n_ratio is not None and n_ratio > 0),
        "below_onset_floor_positive": bool(below["finite"] and u_ratio is not None and u_ratio > 0),
        "C2_limit_control": bool(all(value is not None for value in c2_values) and
                                  c2_values[0] > c2_values[1] > c2_values[2] > 0),
        "not_C3_limit_control": bool(all(value is not None for value in c3_values) and
                                      c3_values[0] < c3_values[1] < c3_values[2]),
        "samples": samples,
        "passed": bool(at_onset["n"] == 0 and at_onset["value"] == 0 and
                        below["finite"] and n_ratio is not None and n_ratio > 0 and
                        u_ratio is not None and u_ratio > 0),
        "scope": "actual envelope onset; finite-delta controls support C2/not-C3 asymptotics from the fixed positive integrals",
    }


def actual_binding_design_certificate(*, dps=80):
    """Run the bounded, fixed-coefficient BindingDesign applicability audit."""
    if isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 120:
        raise ValueError("dps must be an integer in [80,120]")
    with mp.workdps(dps):
        design = BindingDesign(dps=dps)
        bounds = _actual_design_bounds(design)
        regularity = _envelope_regularity_certificate(design)
        # Small fixed off-equilibrium samples; no fit, optimizer or answer
        # table is involved.  The full E_W force is evaluated for every row.
        samples = [_actual_design_sample(design, y, ratio)
                   for y, ratio in (("0.6", "0.25"), ("0.75", "1"),
                                    ("0.9", "2"), ("1.0", "1"),
                                    ("1.1", "0.25"))]
        legendre_rows = []
        # Independent integrand quadrature at two nonzero states makes the
        # global positivity claim traceable to the same action/EOS closure.
        for y, ratio in (("0.75", "0.5"), ("1.0", "1.0")):
            nn = design.n*finite(ratio, "integral ratio", positive=True)
            fi = _positive_fermi_integrals(nn, finite(y, "integral y", positive=True), design)
            legendre_rows.append({
                "y": y, "n_ratio": ratio,
                "legendre_residual": decimal(fi["F"]+fi["P_F"]-nn*fi["EF"]),
                "pressure_gap": decimal(fi["F"]-fi["P_F"]),
                "finite": bool(all(mp.isfinite(fi[key]) for key in
                                    ("F", "P_F", "n_s", "EF"))),
                "positive_pressure_gap": bool(fi["F"]-fi["P_F"] >= 0),
            })
        qphi = design.base.momega/design.base.W0
        Gv = design.gomega**2/(qphi*qphi)
        fixed_inputs = {
            "target_y": decimal(design.y), "target_K_MeV": decimal(design.K),
            "target_n_fm_minus3": decimal(design.n_fm3),
            "target_binding_MeV": decimal(design.binding),
            "target_mu_MeV": decimal(design.mu),
            "target_gomega": decimal(design.gomega),
            "target_Cv_MeV_minus2": decimal(design.Cv),
            "target_beta_MeV4": decimal(design.beta),
            "target_delta_y_MeV4": decimal(design.delta_y),
            "qphi_momega_over_W0": decimal(qphi), "Gv": decimal(Gv),
            "original_gomega": str(BINDING_INPUTS["gomega"]),
            "gomega_is_engineered_counterfactual": True,
            "dps": dps, "fixed_producer": "verification/nvg_binding_feasibility_audit.py",
            "coefficients_are_not_refit_here": True,
        }
        passed = bool(bounds["passed"] and regularity["passed"] and
                      len(samples) == 5 and len(legendre_rows) == 2 and
                      all(row["finite"] and row["positive_pressure_gap"] and
                          _abs_below(row["legendre_residual"], "1e-55")
                          for row in legendre_rows) and
                      all(row["finite"] and row["potential_nonnegative"] and
                          row["rho_positive"] and row["Q_positive"] and
                          row["Q_upper_bound"] and
                          _abs_below(row["vector_chain_residual"], "1e-55") and
                          _abs_below(row["envelope_force_residual"], "1e-40") and
                          _abs_below(row["force_residual_relative"], "1e-40") and
                          _abs_below(row["fermi_legendre_residual"], "1e-55") and
                          _abs_below(row["Q_decomposition_residual"], "1e-55") and
                          _abs_below(row["upper_bound_decomposition_residual"], "1e-55")
                          for row in samples))
        return {
            "passed": passed, "design_inputs": fixed_inputs,
            "barrier_and_bounds": bounds, "envelope_regularity": regularity,
            "fermi_integral_controls": {
                "legendre_identity": "F+P_F=n*E_F from d(p^3 E)/dp=3p^2 E+p^4/E",
                "pressure_gap_integrand": "3p^2 E-p^4/E=p^2*(3m^2+2p^2)/E>=0",
                "mass_derivative_integrand": "dE/dm=m/E>=0",
                "Q_formula_from_positive_integrals": "Q=v^2+n*E_F+Gv*n^2/W^2>0",
                "upper_bound_formula": "2*rho-Q=F-P_F+2*U>=0",
                "rows": legendre_rows,
            },
            "off_equilibrium_force_samples": samples,
            "force_derivation": {
                "energy": "E=F(n,MN*y)+Cv*n^2/(2*y^2)+U_BindingDesign(y)",
                "vector_normalization": "qphi=momega/W0; Gv=gomega^2/qphi^2=Cv*W0^2",
                "Uy": "U_floor_y=-MN*n_s(nbar,y)+Cv*nbar^2/y^3; barrier_y=4*beta*y*z*(z-zs)*(2*z-zs)",
                "Ey": "MN*n_s+Uy-Cv*n^2/y^3",
                "EW": "Ey/W0",
                "action": "L_W=a^3*(Wdot^2/2-E), giving Wddot+3H*Wdot+E_W=0",
                "full_EW_used": True,
            },
            "manufactured_polynomial_fixture": manufactured_polynomial_fixture(
                beta=design.beta, y=design.y),
            "scope": {
                "actual_potential_used": True,
                "actual_gomega_and_Gv_used": True,
                "polynomial_fixture_used_for_acceptance": False,
                "homogeneous_T0_normal_fermi_only": True,
                "gravity_and_rho_c_extra_assumptions": True,
                "no_inhomogeneous_global_minimum_claim": True,
                "no_empirical_weight": True,
            },
        }


def curved_action_symbolic_checks(*, branch_sign=-1, qdot_curvature_scale=1,
                                  potential_scale=1):
    """Vary the curved FLRW lapse action and check the candidate q equations.

    ``branch_sign=-1`` is the declared action ``-mu2*sqrt(2X)-V``.  Other
    values are negative controls.  ``qdot_curvature_scale`` multiplies the
    curvature correction and ``potential_scale`` changes V; both must fail
    when changed.
    """
    if isinstance(branch_sign, bool) or branch_sign not in (-1, 1):
        raise ValueError("branch_sign must be -1 or +1")
    if isinstance(qdot_curvature_scale, bool):
        raise ValueError("qdot_curvature_scale must be finite")
    eta = sp.Rational(str(finite(qdot_curvature_scale, "qdot_curvature_scale")))
    vscale = sp.Rational(str(finite(potential_scale, "potential_scale")))

    a, N, ad, add, Nd = sp.symbols("a N adot addot Ndot", positive=True)
    W, wd, B, M2, kap = sp.symbols("W Wdot B M2 kappa", real=True)
    chi, chid, mu2 = sp.symbols("chi chid mu2", real=True, positive=True)
    Vfun = sp.Function("V")
    Efun = sp.Function("E")
    n = B/a**3
    nvar = sp.symbols("nvar", positive=True)
    e = Efun(nvar, W).subs(nvar, n)
    En = sp.diff(Efun(nvar, W), nvar).subs(nvar, n)
    rho_m = wd**2/(2*N**2)+e
    p_m = wd**2/(2*N**2)+n*En-e
    # The integrated-by-parts cuscuton action on a positive chi_dot branch.
    # Original sign s*mu2*sqrt(2X) becomes -3*s*mu2*a^2*chi*ad.
    L = (-3*M2*a*ad**2/N + 3*M2*kap*N*a
         -3*branch_sign*mu2*a**2*chi*ad - N*a**3*Vfun(chi)
         + a**3*wd**2/(2*N)-N*a**3*e)

    def d_dt(expr):
        return (sp.diff(expr, a)*ad+sp.diff(expr, ad)*add+
                sp.diff(expr, N)*Nd+sp.diff(expr, chi)*chid)

    n_variation = sp.diff(L, N)/a**3
    chi_variation = (sp.diff(L, chi)-d_dt(sp.diff(L, chid)))/(N*a**3)
    a_variation = (sp.diff(L, a)-d_dt(sp.diff(L, ad)))/(3*N*a**2)
    H = ad/(N*a)
    Hdot = add/(N**2*a)-ad*Nd/(N**3*a)-H**2
    friedmann_target = 3*M2*(H**2+kap/a**2)-rho_m-Vfun(chi)
    ray_target = (2*M2*(Hdot-kap/a**2)+p_m+rho_m+
                  mu2*branch_sign*chid/N)
    # With branch_sign=-1, the last term is -mu2*chi_dot, as required.
    rows = {
        "lapse_variation_friedmann": n_variation-friedmann_target,
        "cuscuton_EL_from_action": chi_variation-
            (-3*branch_sign*mu2*H-sp.diff(Vfun(chi), chi)),
        "scale_variation_raychaudhuri": a_variation-
            (M2*(2*Hdot+3*H**2+kap/a**2)+p_m-Vfun(chi)+
             branch_sign*mu2*chid/N),
        "raychaudhuri_after_lapse_elimination": a_variation-ray_target-
            friedmann_target,
    }

    # Candidate reconstruction, with chi normalized explicitly by mu2.
    q = sp.symbols("q", real=True)
    rc, alpha, Q, a0 = sp.symbols("rho_c alpha Q a0", positive=True)
    x = sp.cos(q)**2
    chi_q = (q+sp.sin(2*q)/2)/(mu2*alpha)
    Vq = -vscale*rc*sp.cos(q)**4
    Hq = alpha*rc*sp.sin(2*q)/3
    qdot = alpha*(Q-eta*2*M2*kap/a0**2)
    rhocandidate = rc*x+3*M2*kap/a0**2
    # The candidate uses alpha^2=3/(4 M2 rc); encode it by substitutions.
    alpha_rule = {alpha**2: sp.Rational(3, 4)/(M2*rc)}
    Vchi = sp.diff(Vq, q)/sp.diff(chi_q, q)
    Hdot_q = sp.diff(Hq, q)*qdot
    chi_q_derivative = sp.diff(chi_q, q)
    rows.update({
        "candidate_potential_gradient": Vchi-3*mu2*Hq,
        "candidate_cuscuton_EL_branch": Vchi+3*branch_sign*mu2*Hq,
        "candidate_friedmann_curved": (
            3*M2*(Hq**2+kap/a0**2)-rhocandidate-Vq).subs(alpha_rule),
        "candidate_raychaudhuri_curved": (
            2*M2*(Hdot_q-kap/a0**2)+Q+
            branch_sign*mu2*sp.diff(chi_q, q)*qdot).subs(alpha_rule),
        "candidate_constraint_derivative": (
            -3*Hq*Q+rc*sp.sin(2*q)*qdot+6*M2*kap*Hq/a0**2).subs(alpha_rule),
        # These are endpoint/flat reductions of the reconstructed solution,
        # not definitions of a candidate density or pressure.
        "candidate_q0_H_zero": Hq.subs(q, 0),
        "candidate_q0_chi_q": chi_q_derivative.subs(q, 0)-2/(mu2*alpha),
        "candidate_endpoint_chi_q_zero": sp.limit(
            chi_q_derivative, q, sp.pi/2, dir="-"),
        "candidate_endpoint_chi_dot_zero": sp.limit(
            chi_q_derivative*qdot, q, sp.pi/2, dir="-"),
        "candidate_endpoint_H_zero": Hq.subs(q, sp.pi/2),
    })
    # For a separately conserved closed-FLRW matter trajectory,
    # F(a)=rho(a)-3*M2*kappa/a^2 has this exact derivative.  The ODE below
    # checks the same relation numerically using the original EOS.
    aa, Qa, ka = sp.symbols("a_scale Q_scale kappa_scale", positive=True)
    F_a_from_conservation = -3*Qa/aa+6*M2*ka/aa**3
    rows["closed_F_derivative_from_conservation"] = F_a_from_conservation-(
        -3/aa*(Qa-2*M2*ka/aa**2))
    # Apply alpha rule to all candidate rows where the rule is needed and
    # simplify trigonometric expressions before reporting exact residuals.
    result = {}
    for name, value in rows.items():
        value = sp.trigsimp(sp.simplify(value.subs(alpha_rule)))
        result[name] = {"residual": str(sp.factor(value)),
                        "passed": bool(value == 0)}
    return result


def endpoint_regularization_checks():
    """Check the finite-field C1/non-C2 endpoint and tempting epsilon term.

    Here ``X=(partial chi)^2/2`` is fixed throughout.  The declared
    regularization therefore contains ``-mu2*sqrt(2*X)``, not ``sqrt(X)``.
    """
    q, eps, alpha, mu2, rc, M2 = sp.symbols(
        "q eps alpha mu2 rho_c M2", positive=True)
    chi = (q+sp.sin(2*q)/2)/(mu2*alpha)
    V = -rc*sp.cos(q)**4
    chi_end = sp.limit(chi, q, sp.pi/2, dir="-")
    gap = chi_end-chi.subs(q, sp.pi/2-eps)
    Vedge = V.subs(q, sp.pi/2-eps)
    Vchi = sp.trigsimp(sp.diff(V, q)/sp.diff(chi, q))
    Vchichi = sp.trigsimp(sp.diff(Vchi, q)/sp.diff(chi, q))
    X, epsilon = sp.symbols("X epsilon", positive=True)
    Vgeneric = sp.Function("V")
    Lreg = -mu2*sp.sqrt(2*X)+epsilon*X/2-Vgeneric(sp.symbols("chi0"))
    kinetic_reg = sp.diff(Lreg, X)+2*X*sp.diff(Lreg, X, 2)
    rows = {
        "chi_endpoint_gap_cubic": sp.limit(gap/eps**3, eps, 0)-
            2/(3*mu2*alpha),
        "potential_endpoint_quartic": sp.limit(Vedge/eps**4, eps, 0)+rc,
        "potential_first_derivative_zero": sp.limit(Vchi, q, sp.pi/2, dir="-"),
        "potential_second_derivative_diverges": sp.limit(
            sp.cos(q)**2*Vchichi, q, sp.pi/2, dir="-")+
            rc*mu2**2*alpha**2,
        "regularized_kinetic_combination": kinetic_reg-epsilon/2,
        # The raw limit is -infinity; its finite leading coefficient is the
        # exact residual used here, while the numeric witness below records
        # an explicit small-X negative value.
        "regularized_LX_negative_asymptotic": sp.limit(
            sp.sqrt(2*X)*sp.diff(Lreg, X), X, 0, dir="+")+mu2,
    }
    return {name: {"residual": str(sp.simplify(value)),
                   "passed": bool(sp.simplify(value) == 0)}
            for name, value in rows.items()}


def endpoint_branch_controls(*, Q="2.4", rho_c="5", M2="3", mu2="1.7"):
    """Separate the regular H=0 flat bounce from the X=0 field endpoint."""
    qent = finite(Q, "Q", positive=True)
    rc = finite(rho_c, "rho_c", positive=True)
    m2 = finite(M2, "M2", positive=True)
    muc = finite(mu2, "mu2", positive=True)
    with mp.workdps(80):
        alpha = mp.sqrt(3/(4*m2*rc))
        q0_chi_q = 2/(muc*alpha)
        q0_chi_dot = 2*qent/muc
        return {
            "q0": {
                "H": decimal(mp.mpf(0)),
                "chi_q": decimal(q0_chi_q),
                "chi_dot": decimal(q0_chi_dot),
                "X": decimal(q0_chi_dot**2/2),
                "H_zero": True, "X_positive": bool(q0_chi_dot**2/2 > 0),
                "regular_timelike_branch": True,
            },
            "q_pi_over_2": {
                "H": decimal(mp.mpf(0)), "chi_q": decimal(mp.mpf(0)),
                "chi_dot": decimal(mp.mpf(0)), "X": decimal(mp.mpf(0)),
                "H_zero": True, "X_zero": True,
                "regular_timelike_branch": False,
            },
            "distinct_controls": True,
            "no_equation_divides_by_H": True,
            "passed": bool(q0_chi_q > 0 and q0_chi_dot > 0 and
                            q0_chi_dot**2/2 > 0),
            "scope": "q=0 is an H=0 regular timelike point; q=pi/2 is the X=0 endpoint",
        }


def flat_gr_identity_checks():
    """Keep the exact flat-GR NEC obstruction as an explicit control."""
    M2, Q = sp.symbols("M2 Q", positive=True)
    Hdot_GR = -Q/(2*M2)
    # At a flat H=0 point the Einstein-Raychaudhuri equation itself fixes the
    # sign.  The companion function below exposes the nonpositive witness.
    rows = {
        "flat_GR_raychaudhuri_solution": Hdot_GR+Q/(2*M2),
        "flat_GR_equation_at_kappa_zero": 2*M2*Hdot_GR+Q,
    }
    return {name: {"residual": str(sp.simplify(value)),
                   "passed": bool(sp.simplify(value) == 0)}
            for name, value in rows.items()}


def flat_gr_obstruction_witness(*, enthalpy="1", M2="1"):
    """Evaluate the flat-GR turning-point sign for a positive enthalpy."""
    Q = finite(enthalpy, "enthalpy", positive=True)
    m2 = finite(M2, "M2", positive=True)
    hdot = -Q/(2*m2)
    return {
        "enthalpy": decimal(Q), "M2": decimal(m2),
        "Hdot_at_flat_turning_point": decimal(hdot),
        "nonpositive": bool(hdot < 0),
        "scope": "canonical/normal matter in flat GR; independent of U shape",
    }


def curved_numeric_check(*, q="0.37", Q="2.4", rho_c="5", M2="3",
                         mu2="1.7", kappa="1", a="2.1"):
    """Independent mpmath point check of the curved candidate equations."""
    qv = finite(q, "q")
    qent = finite(Q, "Q", positive=True)
    rc = finite(rho_c, "rho_c", positive=True)
    m2 = finite(M2, "M2", positive=True)
    muc = finite(mu2, "mu2", positive=True)
    kap = finite(kappa, "kappa")
    av = finite(a, "a", positive=True)
    if abs(qv) >= mp.pi/2:
        raise ValueError("numeric curved check q must lie inside the timelike branch")
    with mp.workdps(80):
        alpha = mp.sqrt(3/(4*m2*rc))

        def chi(z):
            return (z+mp.sin(2*z)/2)/(muc*alpha)

        def potential(z):
            return -rc*mp.cos(z)**4

        def hubble(z):
            return alpha*rc*mp.sin(2*z)/3

        qdot = alpha*(qent-2*m2*kap/av**2)
        chi_q = mp.diff(chi, qv)
        chi_dot = chi_q*qdot
        Vchi = mp.diff(potential, qv)/chi_q
        Hdot = mp.diff(hubble, qv)*qdot
        rho = rc*mp.cos(qv)**2+3*m2*kap/av**2
        residuals = {
            "friedmann": 3*m2*(hubble(qv)**2+kap/av**2)-rho-potential(qv),
            "raychaudhuri": 2*m2*(Hdot-kap/av**2)+qent-muc*chi_dot,
            "cuscuton_EL": Vchi-3*muc*hubble(qv),
            "constraint_derivative": -3*hubble(qv)*qent+
                rc*mp.sin(2*qv)*qdot+6*m2*kap*hubble(qv)/av**2,
        }
        scale = max(mp.mpf(1), *(abs(v) for v in residuals.values()))
        max_relative = max(abs(v)/scale for v in residuals.values())
        return {"q": decimal(qv), "Q": decimal(qent), "rho_c": decimal(rc),
                "M2": decimal(m2), "mu2": decimal(muc), "kappa": decimal(kap),
                "a": decimal(av), "max_scaled_residual": decimal(max_relative),
                "residuals": {key: decimal(value) for key, value in residuals.items()},
                "passed": bool(max_relative < mp.mpf("1e-65")),
                "scope": "one curved candidate point; symbolic checks carry the identity",
                }


def regularization_witness(*, mu2="1", epsilon="0.1", X="1e-12"):
    """Evaluate L_X for the declared ``X=(dchi)^2/2`` regularization."""
    mu = finite(mu2, "mu2", positive=True)
    eps = finite(epsilon, "epsilon", positive=True)
    xx = finite(X, "X", positive=True)
    lx = -mu/mp.sqrt(2*xx)+eps/2
    return {
        "mu2": decimal(mu), "epsilon": decimal(eps), "X": decimal(xx),
        "L_X": decimal(lx), "L_X_negative": bool(lx < 0),
        "negative_for_X_below": decimal(2*(mu/eps)**2),
        "kinetic_combination": decimal(eps/2),
        "declared_X_convention": "X=(partial chi)^2/2; L=-mu2*sqrt(2X)+epsilon*X/2",
        "scope": "epsilon>0; timelike X>0; local branch diagnostic only",
    }


def _bisect_root(function, lo, hi, *, iterations=90):
    flo, fhi = function(lo), function(hi)
    if not flo > 0 or not fhi < 0:
        raise ValueError("root is not bracketed with positive-to-negative orientation")
    for _ in range(iterations):
        mid = (lo+hi)/2
        fmid = function(mid)
        if fmid > 0:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def _integrate_formal_cycle(model, n_ref, rho_ref, rho_c_ratio, a_low,
                            *, rtol, atol, max_step):
    """Integrate A(q), tau(q) over q in [0,pi]; q continuation is formal."""
    import numpy as np
    from scipy.integrate import solve_ivp

    R = float(rho_c_ratio)
    nref = mp.mpf(n_ref)
    rhoref = mp.mpf(rho_ref)
    root = float(a_low)
    cache = {}

    def eos(A):
        A = float(A)
        key = A
        if key not in cache:
            if not math.isfinite(A) or A <= 0:
                raise ArithmeticError("formal cycle reached invalid scale factor")
            state = model.equilibrium(nref/mp.mpf(str(A))**3)
            cache[key] = (float(state["energy_total"]/rhoref),
                          float((state["energy_total"]+
                                 state["pressure_total"])/rhoref),
                          float(state["pressure_total"]/rhoref))
        return cache[key]

    calls = 0

    def rhs(q, state):
        nonlocal calls
        calls += 1
        if calls > 20000:
            raise ArithmeticError("bounded formal-cycle ODE budget exceeded")
        A = float(state[0])
        if not math.isfinite(A) or A <= root*0.98 or A >= 1.02:
            raise ArithmeticError("formal-cycle ODE left declared EOS interval")
        rho_hat, Q_hat, _ = eos(A)
        F_A = -3*Q_hat/A+2/A**3
        qdot_hat = 3/(2*math.sqrt(R))*(Q_hat-2/(3*A*A))
        if not math.isfinite(F_A) or F_A >= 0:
            raise ArithmeticError("F(A) is not decreasing on the declared cycle")
        if not math.isfinite(qdot_hat) or qdot_hat <= 0:
            raise ArithmeticError("formal q rate is not positive")
        dA = -R*math.sin(2*q)/F_A
        dtau = 1/qdot_hat
        if not all(math.isfinite(value) for value in (dA, dtau)):
            raise ArithmeticError("nonfinite formal-cycle derivative")
        return (dA, dtau)

    sol = solve_ivp(rhs, (0.0, math.pi), (root, 0.0), method="DOP853",
                    rtol=float(rtol), atol=float(atol), max_step=float(max_step),
                    dense_output=True)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise ArithmeticError("formal closed-background integration failed")
    midpoint = sol.sol(math.pi/2)
    return {
        "success": bool(sol.success), "A_end": float(sol.y[0, -1]),
        "A_mid": float(midpoint[0]), "tau_end": float(sol.y[1, -1]),
        "tau_mid": float(midpoint[1]), "rhs_evaluations": int(sol.nfev),
        "dense_solution": sol, "eos_cache": eos, "root": root,
    }


def formal_closed_example(rho_c_ratio="1", n_ref_ratio="1", *, dps=50,
                          rtol=2e-9, atol=2e-11, max_step=0.04):
    """Run one original-EOS formal k=+1 trajectory and refinement.

    ``rho_c_ratio=rho_c/rho_ref`` and ``n_ref_ratio=n_ref/n0`` are external
    dimensionless controls.  Set ``a_ref=sqrt(3 M2/rho_ref)`` and
    ``A=a/a_ref``; then the closed constraint is
    ``F(A)=rho/rho_ref-A^-2=rho_c_ratio*cos(q)^2``.
    """
    if isinstance(dps, bool) or not isinstance(dps, int) or not 40 <= dps <= 120:
        raise ValueError("dps must be an integer in [40,120]")
    R = finite(rho_c_ratio, "rho_c_ratio", positive=True)
    nr = finite(n_ref_ratio, "n_ref_ratio", positive=True)
    rel = finite(rtol, "rtol", positive=True)
    abs_tol = finite(atol, "atol", positive=True)
    step = finite(max_step, "max_step", positive=True)
    if rel < mp.mpf("1e-12") or abs_tol < mp.mpf("1e-14"):
        raise ValueError("ODE tolerances are outside the supported float range")
    if R > 20 or nr > 8 or nr < mp.mpf("0.125"):
        raise ValueError("control ratios exceed the bounded original-EOS example")

    with mp.workdps(dps):
        model = BulkModel()
        n_ref = nr*model.n0
        reference = model.equilibrium(n_ref)
        rho_ref = reference["energy_total"]
        if rho_ref <= 0 or reference["pressure_total"] < 0:
            raise ArithmeticError("original reference EOS does not have P>=0")

        def eos(A):
            A = finite(A, "A", positive=True)
            state = model.equilibrium(n_ref/A**3)
            return state

        def F(A):
            state = eos(A)
            return state["energy_total"]/rho_ref-A**(-2)

        # At A=1, rho=rho_ref by definition, hence F(1)=0.  At small A the
        # original positive-energy EOS grows faster than curvature, so F>R.
        hi = mp.mpf(1)
        lo = mp.mpf("0.25")
        for _ in range(12):
            if F(lo) > R:
                break
            lo /= 2
        else:
            raise ArithmeticError("could not bracket the lower formal turnaround")
        a_low = _bisect_root(lambda A: F(A)-R, lo, hi)
        if not 0 < a_low < 1:
            raise ArithmeticError("invalid lower formal turnaround")

        # Check monotonicity and positivity of the original EOS on a compact
        # sample before allowing the ODE to run.
        min_pressure = mp.inf
        min_F_A = mp.inf
        min_qrate = mp.inf
        for j in range(17):
            A = a_low+(1-a_low)*mp.mpf(j)/16
            state = eos(A)
            Q_hat = (state["energy_total"]+state["pressure_total"])/rho_ref
            F_A = -3*Q_hat/A+2/A**3
            qrate = 3/(2*mp.sqrt(R))*(Q_hat-2/(3*A*A))
            min_pressure = min(min_pressure,
                               state["pressure_total"]/rho_ref)
            min_F_A = min(min_F_A, F_A)
            min_qrate = min(min_qrate, qrate)

        coarse = _integrate_formal_cycle(
            model, n_ref, rho_ref, R, a_low,
            rtol=max(float(rel)*20, 2e-8), atol=max(float(abs_tol)*20, 2e-10),
            max_step=max(float(step)*3, 0.12))
        fine = _integrate_formal_cycle(
            model, n_ref, rho_ref, R, a_low,
            rtol=float(rel), atol=float(abs_tol), max_step=float(step))
        period_rel = abs(coarse["tau_end"]-fine["tau_end"])/max(
            1.0, abs(fine["tau_end"]))
        half_symmetry_rel = abs(2*fine["tau_mid"]-fine["tau_end"])/max(
            1.0, abs(fine["tau_end"]))

        # Dense-solution conservation and constraint checks.  The derivative
        # d rho/dA is estimated independently from the EOS at A +/- h rather
        # than simply reusing the identity used to construct F_A.
        max_constraint = mp.mpf(0)
        max_continuity = mp.mpf(0)
        min_pressure = mp.inf
        min_F_A = mp.inf
        min_qrate = mp.inf
        for j in range(1, 16):
            qvalue = mp.pi*mp.mpf(j)/16
            A = mp.mpf(str(float(fine["dense_solution"].sol(float(qvalue))[0])))
            state = eos(A)
            rho_hat = state["energy_total"]/rho_ref
            Q_hat = (state["energy_total"]+state["pressure_total"])/rho_ref
            Fvalue = rho_hat-A**(-2)
            constraint = Fvalue-R*mp.cos(qvalue)**2
            h = min(mp.mpf("1e-5"), (A-a_low)/20,
                    (1-A)/20 if A < 1 else mp.mpf("1e-5"))
            h = max(h, mp.mpf("1e-9"))
            left = eos(A-h)["energy_total"]/rho_ref
            right = eos(A+h)["energy_total"]/rho_ref
            derivative_fd = (right-left)/(2*h)
            continuity = derivative_fd+3*Q_hat/A
            F_A = -3*Q_hat/A+2/A**3
            qrate = 3/(2*mp.sqrt(R))*(Q_hat-2/(3*A*A))
            max_constraint = max(max_constraint, abs(constraint))
            max_continuity = max(max_continuity, abs(continuity)/
                                 max(1, abs(derivative_fd), abs(3*Q_hat/A)))
            min_pressure = min(min_pressure,
                               state["pressure_total"]/rho_ref)
            min_F_A = min(min_F_A, F_A)
            min_qrate = min(min_qrate, qrate)

        A_return = mp.mpf(str(float(fine["A_end"])))
        A_mid = mp.mpf(str(float(fine["A_mid"])))
        cycle_flags = {
            "reference_pressure_nonnegative": bool(reference["pressure_total"] >= 0),
            "sample_pressure_nonnegative": bool(min_pressure >= 0),
            "F_decreases_on_declared_interval": bool(min_F_A < 0),
            "q_rate_positive": bool(min_qrate > 0),
            "two_distinct_finite_scale_turnarounds": bool(a_low > 0 and
                                                           a_low < A_mid and
                                                           abs(A_mid-1) < 5e-6),
            "formal_return_to_lower_turnaround": bool(abs(A_return-a_low) < 5e-6),
            "conservation_refined": bool(max_continuity < mp.mpf("5e-5")),
            "constraint_refined": bool(max_constraint < mp.mpf("5e-7")),
            "period_refined": bool(period_rel < 2e-6),
        }
        return _json_values({
            "status": STATUS,
            "evidentiary_weight": EVIDENTIARY_WEIGHT,
            "formal_background_checks_passed": bool(all(cycle_flags.values())),
            "regular_action_solution": False,
            "healthy_cyclic_universe": False,
            "inputs": {
                "rho_c_over_rho_ref": decimal(R),
                "n_ref_over_n0": decimal(nr),
                "dps": dps, "rtol": float(rel), "atol": float(abs_tol),
                "max_step": float(step), "rho_ref_MeV4": decimal(rho_ref),
                "G_Newton_MeV_minus2": GN_INPUT,
            },
            "turnarounds": {
                "A_lower_q0": decimal(a_low), "A_upper_q_pi_over_2": decimal(A_mid),
                "A_return_q_pi": decimal(A_return),
                "F_lower": decimal(F(a_low)), "F_upper": decimal(F(A_mid)),
            },
            "period": {
                "dimensionless_time_unit": "tau=h_ref*t, h_ref=sqrt(rho_ref/(3*M2))",
                "coarse_tau": decimal(coarse["tau_end"]),
                "fine_tau": decimal(fine["tau_end"]),
                "fine_half_tau": decimal(fine["tau_mid"]),
                "coarse_fine_relative_difference": decimal(period_rel),
                "half_cycle_symmetry_relative_error": decimal(half_symmetry_rel),
                "coarse_rhs_evaluations": coarse["rhs_evaluations"],
                "fine_rhs_evaluations": fine["rhs_evaluations"],
            },
            "validation": {
                "max_constraint_residual": decimal(max_constraint),
                "max_continuity_relative_residual": decimal(max_continuity),
                "minimum_sample_P_over_rho_ref": decimal(min_pressure),
                "maximum_negative_F_derivative": decimal(min_F_A),
                "minimum_q_rate_over_h_ref": decimal(min_qrate),
            },
            "flags": cycle_flags,
            "scope": {
                "eos": "original BulkModel stationary one-component EOS; no new nuclear calibration",
                "controls": "rho_c/rho_ref and n_ref are external mathematical inputs",
                "trajectory": "formal q-periodic kappa=+1 continuation only",
                "endpoint": "q=pi/2 has chi_q=chi_dot=0; V(chi) is C1 not C2",
                "health": "not a regular cuscuton action solution or healthy cyclic universe",
                "dynamics": "no claim of periodic dynamical-W evolution",
            },
        })


def _formal_cycle_evidence_passes(cycle):
    """Recheck the retained formal cycle from its emitted evidence."""
    if cycle is None:
        return True
    if not isinstance(cycle, dict) or cycle.get("formal_background_checks_passed") is not True:
        return False
    flags = cycle.get("flags")
    required_flags = {
        "reference_pressure_nonnegative", "sample_pressure_nonnegative",
        "F_decreases_on_declared_interval", "q_rate_positive",
        "two_distinct_finite_scale_turnarounds",
        "formal_return_to_lower_turnaround", "conservation_refined",
        "constraint_refined", "period_refined",
    }
    if (not isinstance(flags, dict) or set(flags) != required_flags or
            not all(value is True for value in flags.values())):
        return False
    validation = cycle.get("validation")
    period = cycle.get("period")
    turnarounds = cycle.get("turnarounds")
    cycle_scope = cycle.get("scope")
    if (not isinstance(validation, dict) or not isinstance(period, dict) or
            not isinstance(turnarounds, dict) or not isinstance(cycle_scope, dict) or
            cycle_scope.get("trajectory") != "formal q-periodic kappa=+1 continuation only" or
            cycle_scope.get("health") != "not a regular cuscuton action solution or healthy cyclic universe" or
            cycle_scope.get("dynamics") != "no claim of periodic dynamical-W evolution"):
        return False
    numeric_fields = (
        "max_constraint_residual", "max_continuity_relative_residual",
        "minimum_sample_P_over_rho_ref", "maximum_negative_F_derivative",
        "minimum_q_rate_over_h_ref",
    )
    if not all(_finite_number(validation.get(key)) is not None for key in numeric_fields):
        return False
    return bool(_abs_below(validation["max_constraint_residual"], "5e-7") and
                _abs_below(validation["max_continuity_relative_residual"], "5e-5") and
                _finite_number(validation["minimum_sample_P_over_rho_ref"]) >= 0 and
                _finite_number(validation["maximum_negative_F_derivative"]) < 0 and
                _finite_number(validation["minimum_q_rate_over_h_ref"]) > 0 and
                _abs_below(period.get("coarse_fine_relative_difference"), "2e-6") and
                _abs_below(turnarounds.get("A_lower_q0"), "1") and
                _finite_number(turnarounds.get("A_lower_q0")) > 0 and
                _finite_number(turnarounds.get("A_upper_q_pi_over_2")) is not None and
                abs(_finite_number(turnarounds["A_upper_q_pi_over_2"])-1) < mp.mpf("5e-5"))


def _actual_design_evidence_passes_high_precision(block):
    """Strictly aggregate real fixed-design evidence; fail closed on drift."""
    if not isinstance(block, dict) or block.get("passed") is not True:
        return False
    inputs = block.get("design_inputs")
    bounds = block.get("barrier_and_bounds")
    regularity = block.get("envelope_regularity")
    controls = block.get("fermi_integral_controls")
    samples = block.get("off_equilibrium_force_samples")
    force = block.get("force_derivation")
    fixture = block.get("manufactured_polynomial_fixture")
    scope = block.get("scope")
    if not all(isinstance(value, dict) for value in
               (inputs, bounds, regularity, controls, force, fixture, scope)):
        return False
    required_input_fields = (
        "target_y", "target_K_MeV", "target_n_fm_minus3",
        "target_binding_MeV", "target_mu_MeV", "target_gomega",
        "target_Cv_MeV_minus2", "target_beta_MeV4",
        "target_delta_y_MeV4", "qphi_momega_over_W0", "Gv", "dps",
    )
    if (inputs.get("fixed_producer") != "verification/nvg_binding_feasibility_audit.py" or
            inputs.get("coefficients_are_not_refit_here") is not True or
            inputs.get("gomega_is_engineered_counterfactual") is not True or
            inputs.get("original_gomega") != str(BINDING_INPUTS["gomega"]) or
            any(_finite_number(inputs.get(key)) is None for key in required_input_fields[:-1]) or
            inputs.get("dps") not in (80, 120)):
        return False
    ystar = _finite_number(inputs["target_y"])
    cv = _finite_number(inputs["target_Cv_MeV_minus2"])
    beta = _finite_number(inputs["target_beta_MeV4"])
    qphi = _finite_number(inputs["qphi_momega_over_W0"])
    gv = _finite_number(inputs["Gv"])
    n_fm = _finite_number(inputs["target_n_fm_minus3"])
    if not all(value is not None for value in (ystar, cv, beta, qphi, gv, n_fm)):
        return False
    W0 = _finite_number(BINDING_INPUTS["W0"])
    MN = _finite_number(BINDING_INPUTS["MN"])
    hbarc = _finite_number(BINDING_INPUTS["hbarc"])
    target_n = n_fm*hbarc**3
    if not (0 < ystar < 1 and cv > 0 and beta > 0 and qphi > 0 and gv > 0 and target_n > 0):
        return False
    # Public numeric fields carry 35 significant digits; use a tolerance
    # below that serialization precision while the high-precision residual
    # is checked separately in ``barrier_and_bounds``.
    target_mu = _finite_number(inputs["target_mu_MeV"])
    binding = _finite_number(inputs["target_binding_MeV"])
    target_gomega = _finite_number(inputs["target_gomega"])
    if (target_mu is None or binding is None or target_gomega is None or
            not _relative_close(target_mu, MN+binding) or
            not _relative_close(qphi, _finite_number(BINDING_INPUTS["momega"])/W0) or
            not _relative_close(target_gomega**2/qphi**2, gv) or
            not _abs_below(gv-cv*W0**2, "1e-28")):
        return False

    bound_bools = (
        "n_cap_positive", "density_cap_from_rho_le_rho_c",
        "envelope_zero_density_competitor", "strict_density_convexity_premise",
        "vector_coefficient_identity_passed", "fixed_n_W_lower_positive",
        "barrier_ratio_at_z2_ge_one_sixteenth",
        "tail_factor_monotone_for_z_ge_2", "barrier_nonnegative_factorization_premise",
        "ys2_le_one_premise",
        "beta_positive_premise", "envelope_floor_nonnegative_premise",
        "tail_U_bound_checked",
    )
    bound_numbers = (
        "rho_c_control", "C_F", "n_cap", "fixed_n", "Gv", "qphi",
        "vector_coefficient_identity", "fixed_n_W_lower",
        "y_tail_from_U_bound", "y_upper_bound", "barrier_ratio_at_z2",
        "barrier_ratio_bound", "tail_U_at_sqrt2", "tail_bound_gap",
    )
    if (bounds.get("passed") is not True or
            bounds.get("envelope_definition") !=
            "U_floor(y)=sup_{n>=0}[mu*n-F(n,MN*y)-Cv*n^2/(2*y^2)]" or
            bounds.get("actual_potential_definition") !=
            "U_BindingDesign=U_floor(y)+beta*(y^2-1)^2*(y^2-y_star^2)^2" or
            bounds.get("zero_density_competitor_value") != "0" or
            bounds.get("grand_density_hessian") !=
            "E_nn=k_F^2/(3*n*E_F)+Cv/y^2>0 for n>0,y>0" or
            bounds.get("tail_bound_argument") !=
            "z>=2, y_star^2<=1: (z-y_star^2)^2>=(z-1)^2 and (1-1/z)^4>=1/16" or
            bounds.get("fixed_n_vector_bound_argument") !=
            "rho>=Gv*n^2/(2*W^2) and rho<=rho_c imply W>=n*sqrt(Gv/(2*rho_c))" or
            bounds.get("barrier_nonnegative_formula") !=
            "beta*(y^2-1)^2*(y^2-y_star^2)^2>=0 for beta>0" or
            not all(bounds.get(key) is True for key in bound_bools) or
            not all(_finite_number(bounds.get(key)) is not None for key in bound_numbers)):
        return False
    d = _finite_number(BINDING_INPUTS["d"])
    expected_cf = mp.mpf(3)/4*(6*mp.pi**2/d)**(mp.mpf(1)/3)
    rc = _finite_number(bounds["rho_c_control"])
    cf = _finite_number(bounds["C_F"])
    ncap = _finite_number(bounds["n_cap"])
    if (not _relative_close(cf, expected_cf) or
            not _relative_close(ncap, (rc/cf)**(mp.mpf(3)/4))):
        return False
    ratio2 = _finite_number(bounds["barrier_ratio_at_z2"])
    if not (ratio2 >= mp.mpf(1)/16 and
            _finite_number(bounds["tail_bound_gap"]) >= 0 and
            _abs_below(bounds["vector_coefficient_identity"], "1e-60")):
        return False

    reg_required = ("onset_y", "density_exponent", "floor_exponent",
                    "density_leading_coefficient",
                    "floor_leading_coefficient_from_positive_pressure_integral")
    if (regularity.get("passed") is not True or
            regularity.get("onset_density_zero") is not True or
            regularity.get("onset_floor_zero") is not True or
            regularity.get("onset_variable") != "x=mu-MN*y down to 0" or
            regularity.get("leading_density_from_integral") !=
            "nbar=d*(2*mu)^(3/2)*x^(3/2)/(6*pi^2)+O(x^(5/2))" or
            regularity.get("leading_floor_from_pressure_integral") !=
            "U_floor=d*(2*mu)^(3/2)*x^(5/2)/(15*pi^2)+O(x^(7/2))" or
            regularity.get("vector_term_order") !=
            "Cv*nbar^2/(2*y^2)=O(x^3)" or
            regularity.get("regularity_basis") !=
            "positive Fermi integrands plus envelope theorem; C2 but not C3" or
            regularity.get("density_exponent") != "3/2" or
            regularity.get("floor_exponent") != "5/2" or
            any(_finite_number(regularity.get(key)) is None for key in reg_required[:1]+reg_required[3:]) or
            regularity.get("C2_limit_control") is not True or
            regularity.get("not_C3_limit_control") is not True):
        return False
    reg_samples = regularity.get("samples")
    if not isinstance(reg_samples, list) or len(reg_samples) != 3:
        return False
    reg_num = ("delta_y", "x", "density_over_x_3_2", "floor_over_x_5_2",
               "floor_over_x_2", "floor_over_x_3")
    if not all(isinstance(item, dict) and item.get("finite") is True and
               all(_finite_number(item.get(key)) is not None for key in reg_num)
               for item in reg_samples):
        return False

    rows = controls.get("rows")
    if (controls.get("legendre_identity") !=
            "F+P_F=n*E_F from d(p^3 E)/dp=3p^2 E+p^4/E" or
            controls.get("pressure_gap_integrand") !=
            "3p^2 E-p^4/E=p^2*(3m^2+2p^2)/E>=0" or
            controls.get("mass_derivative_integrand") != "dE/dm=m/E>=0" or
            controls.get("Q_formula_from_positive_integrals") !=
            "Q=v^2+n*E_F+Gv*n^2/W^2>0" or
            controls.get("upper_bound_formula") !=
            "2*rho-Q=F-P_F+2*U>=0" or
            not isinstance(rows, list) or len(rows) != 2):
        return False
    for row in rows:
        if (not isinstance(row, dict) or row.get("finite") is not True or
                row.get("positive_pressure_gap") is not True or
                _finite_number(row.get("pressure_gap")) is None or
                _finite_number(row["pressure_gap"]) < 0 or
                not _abs_below(row.get("legendre_residual"), "1e-55")):
            return False

    expected_ids = {"y=0.6,n_ratio=0.25", "y=0.75,n_ratio=1.0",
                    "y=0.9,n_ratio=2.0", "y=1.0,n_ratio=1.0",
                    "y=1.1,n_ratio=0.25"}
    if not isinstance(samples, list) or len(samples) != len(expected_ids):
        return False
    sample_ids = {item.get("sample_id") for item in samples if isinstance(item, dict)}
    if sample_ids != expected_ids:
        return False
    sample_numeric = (
        "y", "n_ratio", "n", "velocity", "W", "potential", "floor",
        "barrier", "envelope_density", "F", "P_F", "n_s", "EF",
        "vector_energy", "vector_chain_residual", "rho", "pressure", "Q",
        "two_rho_minus_Q", "Q_decomposition", "Q_decomposition_residual",
        "upper_bound_decomposition", "upper_bound_decomposition_residual",
        "E_y", "E_W", "direct_U_y",
        "envelope_force_residual", "force_residual_relative",
        "fermi_legendre_residual", "fermi_pressure_gap",
    )
    sample_bools = ("finite", "potential_nonnegative", "rho_positive",
                    "Q_positive", "Q_upper_bound")
    for item in samples:
        if (not isinstance(item, dict) or
                any(_finite_number(item.get(key)) is None for key in sample_numeric) or
                any(item.get(key) is not True for key in sample_bools)):
            return False
        yy = _finite_number(item["y"])
        nr = _finite_number(item["n_ratio"])
        nn = _finite_number(item["n"])
        vv = _finite_number(item["velocity"])
        W = _finite_number(item["W"])
        floor = _finite_number(item["floor"])
        barrier = _finite_number(item["barrier"])
        potential = _finite_number(item["potential"])
        ff = _finite_number(item["F"])
        pp = _finite_number(item["P_F"])
        ns = _finite_number(item["n_s"])
        vec = _finite_number(item["vector_energy"])
        rho = _finite_number(item["rho"])
        pressure = _finite_number(item["pressure"])
        Q = _finite_number(item["Q"])
        tw = _finite_number(item["two_rho_minus_Q"])
        qdec = _finite_number(item["Q_decomposition"])
        qdec_res = _finite_number(item["Q_decomposition_residual"])
        ubdec = _finite_number(item["upper_bound_decomposition"])
        ubdec_res = _finite_number(item["upper_bound_decomposition_residual"])
        Ey = _finite_number(item["E_y"])
        EW = _finite_number(item["E_W"])
        direct_Uy = _finite_number(item["direct_U_y"])
        if not (yy > 0 and nr > 0 and nn > 0 and W > 0 and floor >= 0 and
                barrier >= 0 and potential >= 0 and ff >= 0 and pp >= 0 and
                ns >= 0 and vec >= 0 and rho > 0 and Q > 0 and
                pp <= ff and _finite_number(item["fermi_pressure_gap"]) >= 0):
            return False
        checks = (
            (nn, nr*target_n), (W, W0*yy),
            (barrier, beta*(yy*yy-1)**2*(yy*yy-ystar*ystar)**2),
            (potential, floor+barrier),
            (vec, cv*nn*nn/(2*yy*yy)),
            (rho, vv*vv/2+ff+potential+vec, vv*vv/2, ff, potential, vec),
            (pressure, vv*vv/2+pp-potential+vec, vv*vv/2, pp, potential, vec),
            (Q, rho+pressure, rho, pressure), (tw, 2*rho-Q, rho, Q),
            (qdec, vv*vv+nn*(_finite_number(item["EF"]))+
             gv*nn*nn/(W*W), vv*vv, nn*(_finite_number(item["EF"])), vec),
            (ubdec, ff-pp+2*potential, ff, pp, potential),
            (Ey, MN*ns+direct_Uy-cv*nn*nn/yy**3,
             MN*ns, direct_Uy, cv*nn*nn/yy**3),
            (EW, Ey/W0),
        )
        if not all(_relative_close(item[0], item[1], *item[2:])
                   for item in checks):
            return False
        if (not _abs_below(item["vector_chain_residual"], "1e-55") or
                not _abs_below(item["fermi_legendre_residual"], "1e-55") or
                not _abs_below(qdec_res, "1e-55") or
                not _abs_below(ubdec_res, "1e-55") or
                not _abs_below(item["force_residual_relative"], "1e-40") or
                not _abs_below(item["envelope_force_residual"], "1e-40")):
            return False
        envres = _finite_number(item["envelope_force_residual"])
        fr = _finite_number(item["force_residual_relative"])
        if abs(fr + envres/max(1, abs(Ey))) >= mp.mpf("1e-28"):
            return False

    if (force.get("full_EW_used") is not True or
            force.get("vector_normalization") !=
            "qphi=momega/W0; Gv=gomega^2/qphi^2=Cv*W0^2" or
            force.get("action") != "L_W=a^3*(Wdot^2/2-E), giving Wddot+3H*Wdot+E_W=0"):
        return False
    if (fixture.get("manufactured_polynomial_fixture") is not True or
            fixture.get("used_for_actual_acceptance") is not False or
            scope.get("actual_potential_used") is not True or
            scope.get("actual_gomega_and_Gv_used") is not True or
            scope.get("polynomial_fixture_used_for_acceptance") is not False or
            scope.get("homogeneous_T0_normal_fermi_only") is not True or
            scope.get("gravity_and_rho_c_extra_assumptions") is not True or
            scope.get("no_inhomogeneous_global_minimum_claim") is not True or
            scope.get("no_empirical_weight") is not True):
        return False
    return True


def _actual_design_evidence_passes(block):
    """Evaluate the strict design gate at enough precision for serialized data."""
    with mp.workdps(100):
        return _actual_design_evidence_passes_high_precision(block)


def acceptance_from_result(result):
    """Strict evidence gate; omissions, nonfinite values and mutations fail closed."""
    if not isinstance(result, dict):
        return False
    if result.get("status") != STATUS or result.get("evidentiary_weight") != 0:
        return False
    groups = (
        ("potential_checks", POTENTIAL_SYMBOLIC_REQUIRED),
        ("flat_symbolic_checks", FLAT_SYMBOLIC_REQUIRED),
        ("curved_action_checks", CURVED_SYMBOLIC_REQUIRED),
        ("endpoint_checks", ENDPOINT_SYMBOLIC_REQUIRED),
        ("flat_GR_checks", GR_SYMBOLIC_REQUIRED),
    )
    if not all(_symbolic_rows_pass(result.get(name), required)
               for name, required in groups):
        return False
    actual = result.get("actual_binding_design")
    if not _actual_design_evidence_passes(actual):
        return False
    point = result.get("curved_numeric_check")
    if (not isinstance(point, dict) or point.get("passed") is not True or
            not _abs_below(point.get("max_scaled_residual"), "1e-65")):
        return False
    point_residuals = point.get("residuals")
    if (not isinstance(point_residuals, dict) or
            set(point_residuals) != {"friedmann", "raychaudhuri", "cuscuton_EL",
                                     "constraint_derivative"} or
            not all(_abs_below(value, "1e-65") for value in point_residuals.values())):
        return False
    endpoints = result.get("endpoint_branch_controls")
    q0 = endpoints.get("q0") if isinstance(endpoints, dict) else None
    qend = endpoints.get("q_pi_over_2") if isinstance(endpoints, dict) else None
    endpoint_numbers = ("H", "chi_q", "chi_dot", "X")
    if (not isinstance(endpoints, dict) or endpoints.get("passed") is not True or
            endpoints.get("distinct_controls") is not True or
            endpoints.get("no_equation_divides_by_H") is not True or
            not isinstance(q0, dict) or not isinstance(qend, dict) or
            any(_finite_number(q0.get(key)) is None for key in endpoint_numbers) or
            any(_finite_number(qend.get(key)) is None for key in endpoint_numbers) or
            q0.get("H_zero") is not True or q0.get("X_positive") is not True or
            q0.get("regular_timelike_branch") is not True or
            qend.get("H_zero") is not True or qend.get("X_zero") is not True or
            qend.get("regular_timelike_branch") is not False or
            _finite_number(q0["chi_dot"]) <= 0 or
            _finite_number(q0["X"]) <= 0 or
            _finite_number(qend["chi_dot"]) != 0 or
            _finite_number(qend["X"]) != 0):
        return False
    witness = result.get("regularization_witness")
    if (not isinstance(witness, dict) or witness.get("L_X_negative") is not True or
            witness.get("declared_X_convention") !=
            "X=(partial chi)^2/2; L=-mu2*sqrt(2X)+epsilon*X/2" or
            not _number_below(witness.get("L_X"), 0) or
            not _number_below(witness.get("negative_for_X_below"), mp.inf)):
        return False
    # The last condition is intentionally written as an explicit finite check
    # rather than trusting the witness label; epsilon/2 is 0.05 for default
    # controls and must be positive.
    kinetic = _finite_number(witness.get("kinetic_combination"))
    if kinetic is None or kinetic <= 0:
        return False
    gr_witness = result.get("flat_GR_obstruction_witness")
    if (not isinstance(gr_witness, dict) or gr_witness.get("nonpositive") is not True or
            _finite_number(gr_witness.get("Hdot_at_flat_turning_point")) is None or
            _finite_number(gr_witness.get("enthalpy")) is None or
            _finite_number(gr_witness.get("M2")) is None or
            _finite_number(gr_witness.get("Hdot_at_flat_turning_point")) >= 0 or
            not _relative_close(
                _finite_number(gr_witness["Hdot_at_flat_turning_point"]),
                -_finite_number(gr_witness["enthalpy"])/
                (2*_finite_number(gr_witness["M2"]) ))):
        return False
    if not _formal_cycle_evidence_passes(result.get("formal_closed_example")):
        return False
    scope = result.get("scope")
    return bool(isinstance(scope, dict) and scope.get("observables") ==
                "zero empirical weight; controls are not measured parameters")


def compute_state(*, include_cycle=True):
    """Build the pure JSON report used by the CLI and tests."""
    if not isinstance(include_cycle, bool):
        raise ValueError("include_cycle must be bool")
    result = {
        "status": STATUS,
        "evidentiary_weight": EVIDENTIARY_WEIGHT,
        "potential_checks": potential_symbolic_checks(),
        "flat_symbolic_checks": flat_symbolic_checks(),
        "curved_action_checks": curved_action_symbolic_checks(),
        "endpoint_checks": endpoint_regularization_checks(),
        "regularization_witness": regularization_witness(),
        "flat_GR_checks": flat_gr_identity_checks(),
        "flat_GR_obstruction_witness": flat_gr_obstruction_witness(),
        "curved_numeric_check": curved_numeric_check(),
        "endpoint_branch_controls": endpoint_branch_controls(),
        "flat_bounds": flat_bound_certificate(),
        "actual_binding_design": actual_binding_design_certificate(),
        "formal_closed_example": formal_closed_example() if include_cycle else None,
        "sources": SOURCES,
        "scope": {
            "gravity": "additional negative-branch cuscuton and rho_c; not induced by matter",
            "matter": "positive-energy zero-T Fermi closure; full inhomogeneous NVG not checked",
            "curvature": "action-derived FLRW equations; kappa=+1 cycle is formal only",
            "regularity": "timelike X>0 branch excludes endpoint X=0; no C2 completion supplied",
            "observables": "zero empirical weight; controls are not measured parameters",
        },
    }
    result = _json_values(result)
    result["mathematical_checks_passed"] = acceptance_from_result(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cycle", action="store_true")
    parser.add_argument("--rho-c-ratio", default="1")
    parser.add_argument("--n-ref-ratio", default="1")
    args = parser.parse_args(argv)
    try:
        if args.skip_cycle:
            result = compute_state(include_cycle=False)
        else:
            # CLI controls are passed explicitly so they remain visible in
            # the output; compute_state's default is the declared example.
            result = compute_state(include_cycle=False)
            result["formal_closed_example"] = formal_closed_example(
                args.rho_c_ratio, args.n_ref_ratio)
            result["mathematical_checks_passed"] = acceptance_from_result(result)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        print(json.dumps({"status": "invalid_or_failed_audit",
                          "evidentiary_weight": EVIDENTIARY_WEIGHT,
                          "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False,
                     sort_keys=True))
    return 0 if result["mathematical_checks_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
