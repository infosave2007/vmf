#!/usr/bin/env python3
"""Longitudinal quadratic-action audit in a frozen isentropic fluid closure.

The accepted source-complete scalar/vector couplings are unchanged. This
replaces fermions by a cold isentropic perfect fluid on frozen Minkowski
space; it is NOT collisionless NVG, the full cuscuton/gravity system, or
physical hydrodynamics at arbitrarily large wave number. No data are fitted
and no artifact is saved. The CLI prints JSON with evidentiary weight zero.

Derivation: scratch/nvg-longitudinal-fluid-derivation-2026-09-09.md.
Contract: verification/contracts/source_complete_action.md.
"""
from __future__ import annotations

import json
from functools import lru_cache

import mpmath as mp
import sympy as sp

from source_complete_scaling_saturation_audit import BulkModel, INPUTS
import nvg_matter_branch_certificate as branch_certificate


STATUS = "FROZEN_MINKOWSKI_ISENTROPIC_FLUID_NOT_FULL_NVG_STABILITY"
DENSITY_RATIOS = ("0.001", "0.01", "0.1", "1", "2", "4", "10", "100")
EXPECTED_DENSITY_RATIOS = DENSITY_RATIOS
EXPECTED_MODE_KEYS = frozenset((n, k) for n in ("0.001", "1", "100") for k in ("0.01", "1", "10"))
EXPECTED_INDEPENDENT_ERRORS = frozenset({
    "scalar_density_quadrature", "static_hessian", "acoustic_vs_existing_cs2",
    "stationarity", "on_shell_P1_identity", "P0_sound_factorization",
})
COEFFICIENT_NAMES = ("a", "b", "c", "g", "M2", "n", "mu_F", "s")
EXPECTED_SYMBOLIC_CHECKS = frozenset({
    "euler_matrix_from_quadratic_action", "dispersion_is_cubic",
    "gauss_elimination_from_legendre", "electric_momentum_minimum_hessian",
    "static_determinant_polynomial", "static_dispersion_identity",
    "quadratic_vertex_value", "quadratic_vertex_location",
    "zero_k_three_roots", "decoupled_three_modes", "acoustic_implicit_derivative",
    "zero_k_density_hessian", "zero_k_mixed_hessian", "zero_k_scalar_hessian",
    "total_chemical_potential", "fluid_inertia_not_total_mu",
    "on_shell_P1_positive_decomposition", "P0_hessian_determinant",
    "P0_acoustic_positive_factorization", "principal_characteristic_speeds",
    "transverse_euler_roots", "transverse_positive_hamiltonian",
})


def finite_number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite real, not bool")
    try:
        value = mp.mpf(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite real") from exc
    if not mp.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} outside finite numeric domain")
    return value


def _precision(dps):
    if isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 200:
        raise ValueError("dps must be an integer in [80,200]")


def _validated(coefficients):
    if not isinstance(coefficients, dict) or not set(COEFFICIENT_NAMES) <= set(coefficients):
        raise ValueError("missing longitudinal coefficients")
    return {key: finite_number(coefficients[key], key, positive=key in ("a", "M2", "n", "mu_F"))
            for key in COEFFICIENT_NAMES}


@lru_cache(maxsize=4)
def _symbolic_model(mixing_scale=1):
    Om, z = sp.symbols("Omega z", real=True)
    a, M2, n, mu = sp.symbols("a M2 n mu_F", positive=True)
    b, c, g, s = sp.symbols("b c g s", real=True)
    k = sp.symbols("k", positive=True)
    omega = sp.symbols("omega", nonzero=True, real=True)
    t, r = sp.symbols("t r", real=True)
    xi, w, phi, A = [sp.Function(name)(t, r) for name in ("xi", "w", "phi", "A")]
    dn = -n*sp.diff(xi, r)
    L2 = (n*mu*sp.diff(xi, t)**2/2-a*dn**2/2-b*dn*w+
          sp.diff(w, t)**2/2-sp.diff(w, r)**2/2-s*w*w/2+
          (sp.diff(A, t)+sp.diff(phi, r))**2/2+M2*(phi*phi-A*A)/2+
          (mixing_scale*c*w-g*dn)*phi+g*n*sp.diff(xi, t)*A)
    fields = (xi, w, phi, A)
    amplitudes = sp.symbols("xi_amp w_amp phi_amp A_amp")
    plane = {}
    for field, amp in zip(fields, amplitudes):
        for nt in range(3):
            for nr in range(3-nt):
                plane[sp.diff(field, t, nt, r, nr)] = (-sp.I*omega)**nt*(sp.I*k)**nr*amp
    eom = []
    for field in fields:
        expression = (sp.diff(sp.diff(L2, sp.diff(field, t)), t)+
                      sp.diff(sp.diff(L2, sp.diff(field, r)), r)-sp.diff(L2, field))
        eom.append(sp.expand(expression).xreplace(plane))
    density_amp, vector_L = sp.symbols("density_amp vector_L")
    substitutions = {amplitudes[0]: sp.I*density_amp/(k*n), amplitudes[3]: vector_L/(omega*k)}
    factors = (-k/(sp.I*n), 1, -1, omega*k)
    rows = [sp.cancel(expr.subs(substitutions)*factor).subs({omega**2: Om, k**2: z})
            for expr, factor in zip(eom, factors)]
    matrix = sp.Matrix(rows).jacobian((density_amp, amplitudes[1], amplitudes[2], vector_L))
    expected_matrix = sp.Matrix([
        [mu*Om/n-a*z, -b*z, -g*z, g], [b, z+s-Om, -c, 0],
        [-g, c, z+M2, -1], [-g*Om, 0, Om*z, M2-Om]])
    determinant = sp.factor(matrix.det())

    # Independent local Legendre transform. Integrate -Pi_A*phi_x by parts
    # before solving the nondynamical temporal-vector equation.
    d, w0, phi0, A0, u, wd, Ad, wx, phix = sp.symbols("d w phi A u wd Ad wx phix", real=True)
    px, pw, pA, pAx = sp.symbols("px pw pA pAx", real=True)
    Llocal = (n*mu*u*u/2-a*d*d/2-b*d*w0+wd*wd/2-wx*wx/2-s*w0*w0/2+
              (Ad+phix)**2/2+M2*(phi0*phi0-A0*A0)/2+
              (mixing_scale*c*w0-g*d)*phi0+g*n*u*A0)
    velocities = sp.solve((sp.diff(Llocal, u)-px, sp.diff(Llocal, wd)-pw,
                           sp.diff(Llocal, Ad)-pA), (u, wd, Ad))
    hamiltonian = sp.expand((px*u+pw*wd+pA*Ad-Llocal).subs(velocities))
    integrated = hamiltonian+pA*phix+pAx*phi0
    gauss_phi = sp.solve(sp.diff(integrated, phi0), phi0)[0]
    eliminated = sp.factor(integrated.subs(phi0, gauss_phi))
    J = g*d-c*w0
    expected_hamiltonian = ((px-g*n*A0)**2/(2*n*mu)+pw**2/2+pA**2/2+
                            (pAx+J)**2/(2*M2)+M2*A0*A0/2+
                            (a*d*d+2*b*d*w0+wx*wx+s*w0*w0)/2)
    fourier_H = eliminated.subs({pAx: k*pA, wx: k*w0})
    static = fourier_H.subs({px: g*n*A0, pw: 0, A0: 0})
    electric_minimum = sp.solve(sp.diff(static, pA), pA)[0]
    static_energy = sp.factor(static.subs(pA, electric_minimum))
    Kderived = sp.hessian(static_energy, (d, w0)).subs(k*k, z)
    K = sp.Matrix([[a+g*g/(z+M2), b-g*c/(z+M2)],
                   [b-g*c/(z+M2), z+s+c*c/(z+M2)]])
    # Real sine/cosine canonical amplitudes: xi,A,Pi_A are sine modes;
    # delta n,w are cosine modes, hence delta n=-n*k*xi.
    xi0 = sp.symbols("xi0", real=True)
    real_H = fourier_H.subs(d, -n*k*xi0)
    canonical_hessian = sp.hessian(real_H, (xi0, w0, A0, px, pw, pA))
    return dict(symbols=(a, b, c, g, M2, n, mu, s), Om=Om, z=z, k=k,
                omega=omega, original_eom=eom, amplitudes=amplitudes,
                matrix=matrix, expected_matrix=expected_matrix, determinant=determinant,
                eliminated_H=eliminated, expected_H=expected_hamiltonian,
                Kderived=Kderived, K=K, canonical_hessian=canonical_hessian)


def symbolic_checks(*, mixing_scale=1, scalar_curvature_sign=-1):
    if isinstance(mixing_scale, bool) or mixing_scale not in (0, 1, -1):
        raise ValueError("mixing_scale is a negative-control switch: 0,1,-1")
    if isinstance(scalar_curvature_sign, bool) or scalar_curvature_sign not in (-1, 1):
        raise ValueError("scalar_curvature_sign must be -1 or +1")
    model = _symbolic_model(mixing_scale)
    a, b, c, g, M2, n, mu, s = model["symbols"]
    Om, z, D, K = model["Om"], model["z"], model["determinant"], model["Kderived"]
    P1 = a*(M2+s)+g*g-b*b
    P0 = M2*(a*s-b*b)+a*c*c+g*g*s+2*b*g*c
    polynomial = a*z*z+P1*z+P0
    Enn, EnW, EWW = a+g*g/M2, b-g*c/M2, s+c*c/M2
    mu_total = mu+g*g*n/M2
    acoustic = -sp.diff(D, z).subs({Om: 0, z: 0})/sp.diff(D, Om).subs({Om: 0, z: 0})
    # Differentiate the accepted eliminated energy, then map kinetic Fermi
    # derivatives. s is fixed-A0 curvature, not E_WW.
    W, q = sp.symbols("W q", positive=True)
    F, U = sp.Function("F"), sp.Function("U")
    E = F(n, W)+U(W)+g*g*n*n/(2*q*q*W*W)
    direct_Enn = sp.diff(E, n, 2).subs(sp.diff(F(n, W), n, 2), a)
    direct_EnW = sp.diff(E, n, W).subs(sp.diff(F(n, W), n, W), b)
    direct_EWW = sp.diff(E, W, 2)
    A0 = g*n/(q*q*W*W)
    mapping = {M2: q*q*W*W, c: 2*mixing_scale*g*n/W,
               s: sp.diff(U(W), W, 2)+sp.diff(F(n, W), W, 2)+
                  scalar_curvature_sign*q*q*A0*A0}
    matrix_error = model["matrix"]-model["expected_matrix"]
    hessian_error = model["Kderived"]-model["K"]
    residuals = {
        "euler_matrix_from_quadratic_action": sum(entry**2 for entry in matrix_error),
        "dispersion_is_cubic": sp.Poly(D, Om).degree()-3,
        "gauss_elimination_from_legendre": model["eliminated_H"]-model["expected_H"],
        "electric_momentum_minimum_hessian": sum(entry**2 for entry in hessian_error),
        "static_determinant_polynomial": (z+M2)*K.det()-polynomial,
        "static_dispersion_identity": D.subs(Om, 0)+M2*z*polynomial,
        "quadratic_vertex_value": polynomial.subs(z, -P1/(2*a))-(P0-P1**2/(4*a)),
        "quadratic_vertex_location": sp.diff(polynomial, z).subs(z, -P1/(2*a)),
        "zero_k_three_roots": D.subs(z, 0)-M2*mu/n*Om*(Om-EWW)*(Om-M2-g*g*n/mu),
        "decoupled_three_modes": D.subs({g: 0, b: 0, c: 0})-
             M2*mu/n*(Om-n*a*z/mu)*(Om-z-s)*(Om-z-M2),
        "acoustic_implicit_derivative": acoustic-n/mu_total*(Enn-EnW**2/EWW),
        "zero_k_density_hessian": Enn.subs(mapping)-direct_Enn,
        "zero_k_mixed_hessian": EnW.subs(mapping)-direct_EnW,
        "zero_k_scalar_hessian": EWW.subs(mapping)-direct_EWW,
        "total_chemical_potential": mu_total.subs(mapping)-sp.diff(E, n).subs(sp.diff(F(n, W), n), mu),
        "fluid_inertia_not_total_mu": mu_total-mu-g*g*n/M2,
    }
    gs, mass, ns, ef, kf, lam = sp.symbols("gs mass ns EF kF lambda", positive=True)
    kinetic_a = kf*kf/(3*n*ef)
    shell_s = 2*lam*W*W+gs*gs*(2*ns/mass-3*n/ef)
    shell_P1 = kinetic_a*(M2+shell_s)+g*g-(gs*mass/ef)**2
    positive_P1 = g*g-gs*gs+kinetic_a*(M2+2*lam*W*W+2*gs*gs*ns/mass)
    v = sp.symbols("v", real=True)
    principal = sp.Poly(sp.expand(D.subs(Om, v*z)), z).coeff_monomial(z**3)
    # Transverse displacement is cyclic: its Omega=0 root is not a
    # positive-frequency wave, nor exponential growth of physical fields.
    xiT, AT, xid, Aid, xidd, Aidd, pxiT, pAT = sp.symbols("xiT AT xid Aid xidd Aidd pxiT pAT", real=True)
    LT = n*mu*xid*xid/2+Aid*Aid/2-(z+M2)*AT*AT/2+g*n*xid*AT
    mom_xi, mom_A = sp.diff(LT, xid), sp.diff(LT, Aid)
    eq_xi = sp.diff(mom_xi, xid)*xidd+sp.diff(mom_xi, AT)*Aid
    eq_A = sp.diff(mom_A, Aid)*Aidd-sp.diff(LT, AT)
    omega = model["omega"]
    transverse = sp.Matrix([eq_xi, eq_A]).subs(
        {xidd: -omega**2*xiT, Aidd: -omega**2*AT, xid: -sp.I*omega*xiT, Aid: -sp.I*omega*AT})
    transverse_det = sp.expand(transverse.jacobian((xiT, AT)).det()).subs(omega**2, Om)
    Tvel = sp.solve((mom_xi-pxiT, mom_A-pAT), (xid, Aid))
    HT = sp.expand((pxiT*xid+pAT*Aid-LT).subs(Tvel))
    residuals.update({
        "on_shell_P1_positive_decomposition": sp.factor((shell_P1-positive_P1)*ef**2).subs(ef**2, kf*kf+mass*mass),
        "P0_hessian_determinant": P0-M2*(Enn*EWW-EnW*EnW),
        "P0_acoustic_positive_factorization": P0-M2*EWW*mu_total*acoustic/n,
        "principal_characteristic_speeds": principal-M2/n*(v-1)**2*(mu*v-a*n),
        "transverse_euler_roots": transverse_det-n*mu*Om*(Om-z-M2-g*g*n/mu),
        "transverse_positive_hamiltonian": HT-(pxiT-g*n*AT)**2/(2*n*mu)-pAT*pAT/2-(z+M2)*AT*AT/2,
    })
    return {name: {"passed": bool(sp.simplify(value) == 0), "residual": str(sp.factor(value))}
            for name, value in residuals.items()}


def positivity_certificate(coefficients):
    c = _validated(coefficients)
    a, b, mix, g, M2, _, _, s = (c[key] for key in COEFFICIENT_NAMES)
    P1 = a*(M2+s)+g*g-b*b
    P0 = M2*(a*s-b*b)+a*mix*mix+g*g*s+2*b*g*mix
    vertex = -P1/(2*a) if P1 < 0 else mp.mpf(0)
    minimum = P0-P1*P1/(4*a) if P1 < 0 else P0
    for key, value in (("P1", P1), ("P0", P0), ("z_min", vertex), ("minimum", minimum)):
        finite_number(value, key)
    return {"P1": P1, "P0": P0, "minimum_at_z": vertex, "minimum_value": minimum,
            "strict_positive_all_z": bool(P0 > 0 and (P1 >= 0 or minimum > 0)),
            "status": "STRICT_POSITIVE" if minimum > 0 else "MARGINAL" if minimum == 0 else "NOT_POSITIVE",
            "scope": "all_mathematical_z>=0_at_this_density_in_frozen_fluid_closure_only"}


def static_hessian(coefficients, z):
    c = _validated(coefficients)
    z = finite_number(z, "z", nonnegative=True)
    den = z+c["M2"]
    values = [[c["a"]+c["g"]**2/den, c["b"]-c["g"]*c["c"]/den],
              [c["b"]-c["g"]*c["c"]/den, z+c["s"]+c["c"]**2/den]]
    for row in values:
        for value in row:
            finite_number(value, "static Hessian")
    return values


def coefficients(n_ratio, *, dps=80):
    _precision(dps)
    with mp.workdps(dps):
        ratio = finite_number(n_ratio, "n_ratio", positive=True)
        model = BulkModel()
        n = ratio*model.n0
        try:
            state = model.equilibrium(n)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("positive stationary branch could not be resolved") from exc
        W = finite_number(state["y"]*model.W0, "W", positive=True)
        g, q, gs = model.gomega, model.momega/model.W0, model.MN/model.W0
        M2 = q*q*W*W
        A0 = g*n/M2
        c = {"a": state["k"]**2/(3*n*state["ef"]), "b": gs*state["m"]/state["ef"],
             "c": 2*g*n/W, "g": g, "M2": M2, "n": n, "mu_F": state["ef"],
             "s": state["Uyy"]/model.W0**2+gs*gs*state["ns_m"]-q*q*A0*A0}
        c = _validated(c)
        # Scalar-density derivative independently integrated, not generated
        # from the closed-form expression used by BulkModel.
        k, mass = state["k"], state["m"]
        integral = model.d/(2*mp.pi**2)*mp.quad(
            lambda p: p**4/(p*p+mass*mass)**mp.mpf("1.5"), [0, k])
        integral_error = abs(integral-state["ns_m"])/max(1, abs(integral))
        K0 = static_hessian(c, 0)
        expected = (state["D"], state["B_y"]/model.W0, state["C_y"]/model.W0**2)
        actual = (K0[0][0], K0[0][1], K0[1][1])
        hessian_error = max(abs(x-y)/max(abs(y), mp.mpf("1e-50")) for x, y in zip(actual, expected))
        mu_total = c["mu_F"]+g*g*n/M2
        acoustic = n/mu_total*(K0[0][0]-K0[0][1]**2/K0[1][1])
        acoustic_error = abs(acoustic-state["cs2"])/max(abs(acoustic), mp.mpf("1e-50"))
        stationarity = abs(state["residual"])/max(1, abs(state["energy_total"]))
        errors = {"scalar_density_quadrature": integral_error, "static_hessian": hessian_error,
                  "acoustic_vs_existing_cs2": acoustic_error, "stationarity": stationarity}
        positive_P1 = g*g-gs*gs+c["a"]*(M2+2*model.lam*W*W+2*gs*gs*state["ns"]/mass)
        P = positivity_certificate(c)
        errors["on_shell_P1_identity"] = abs(P["P1"]-positive_P1)/max(1, abs(positive_P1))
        errors["P0_sound_factorization"] = abs(P["P0"]-M2*K0[1][1]*mu_total*acoustic/n)/max(1, abs(P["P0"]))
        for key, value in errors.items():
            finite_number(value, key)
        return {"n_ratio": ratio, "W_over_W0": state["y"], "coefficients": c,
                "mu_total": mu_total, "acoustic_cs2": acoustic,
                "zero_k_Omega": [mp.mpf(0), K0[1][1], M2+g*g*n/c["mu_F"]],
                "principal_squared_speeds": [n*c["a"]/c["mu_F"], mp.mpf(1), mp.mpf(1)],
                "positive_energy": P,
                "independent_errors": errors,
                "independent_checks_passed": bool(max(errors.values()) < mp.mpf("1e-60"))}


@lru_cache(maxsize=1)
def _numeric_functions():
    model = _symbolic_model()
    variables = (*model["symbols"], model["z"])
    polynomial = sp.Poly(model["determinant"], model["Om"])
    poly = sp.lambdify(variables, polynomial.all_coeffs(), "mpmath")
    hessian = sp.lambdify((*model["symbols"], model["k"]), model["canonical_hessian"], "mpmath")
    return poly, hessian


def dispersion_roots(coefficients, z, *, dps=80):
    """Raw complex roots plus independent canonical-generator eigenvalues.

    No root is clipped, dropped, or coerced to its real part. Classification
    records numerical resolution; an exact zero is marginal, not positive.
    The k=0 spectrum is a continuous-k limit, not an unconstrained uniform
    baryon-number perturbation. L=omega*k*A is not invertible at omega*k=0.
    """
    _precision(dps)
    with mp.workdps(dps):
        c = _validated(coefficients)
        z = finite_number(z, "z", nonnegative=True)
        poly_function, hessian_function = _numeric_functions()
        args = [c[key] for key in COEFFICIENT_NAMES]
        poly = [finite_number(v, "polynomial coefficient") for v in poly_function(*args, z)]
        scale = c["M2"]+abs(c["s"])+c["c"]**2/c["M2"]+c["g"]**2*c["n"]/c["mu_F"]+z
        if z == 0:
            roots = [mp.mpc(0), mp.mpc(c["s"]+c["c"]**2/c["M2"]),
                     mp.mpc(c["M2"]+c["g"]**2*c["n"]/c["mu_F"])]
        else:
            normalized = [poly[i]/poly[0]/scale**i for i in range(4)]
            try:
                roots = [r*scale for r in mp.polyroots(normalized, maxsteps=300,
                                                      extraprec=100, cleanup=False)]
            except mp.libmp.libhyper.NoConvergence as exc:
                raise ArithmeticError("dispersion roots did not converge") from exc
        for root in roots:
            finite_number(mp.re(root), "root real part")
            finite_number(mp.im(root), "root imaginary part")
        roots = sorted(roots, key=lambda r: (mp.re(r), mp.im(r)))
        H = mp.matrix(hessian_function(*args, mp.sqrt(z)))
        symplectic = mp.zeros(6)
        for i in range(3):
            symplectic[i, i+3] = 1
            symplectic[i+3, i] = -1
        for entry in H:
            finite_number(entry, "canonical Hessian entry")
        try:
            eigenvalues = mp.eig(symplectic*H, left=False, right=False)
        except mp.libmp.libhyper.NoConvergence as exc:
            raise ArithmeticError("canonical-generator eigenvalues did not converge") from exc
        for eigenvalue in eigenvalues:
            finite_number(mp.re(eigenvalue), "generator eigenvalue real part")
            finite_number(mp.im(eigenvalue), "generator eigenvalue imaginary part")
        generator_Omega = [-e*e for e in eigenvalues]
        error = max(min(abs(value-root) for root in roots)/scale for value in generator_Omega)
        residual = max(abs(mp.polyval(poly, root))/
                       max(mp.fsum(abs(poly[i])*abs(root)**(3-i) for i in range(4)), abs(poly[0])*scale**3)
                       for root in roots)
        finite_number(error, "generator comparison")
        finite_number(residual, "polynomial residual")
        tolerance = mp.mpf("1e-50")*scale
        if any(abs(mp.im(root)) > tolerance for root in roots):
            root_status = "COMPLEX_OMEGA"
        elif any(mp.re(root) < -tolerance for root in roots):
            root_status = "NEGATIVE_OMEGA_INSTABILITY"
        elif any(mp.re(root) <= tolerance for root in roots):
            root_status = "MARGINAL_OR_UNRESOLVED_NEAR_ZERO"
        else:
            root_status = "POSITIVE_REAL_WITHIN_NUMERIC_RESOLUTION"
        return {"z": z, "Omega_roots_raw": roots, "generator_Omega_raw": generator_Omega,
                "generator_max_relative_difference": error, "polynomial_max_scaled_residual": residual,
                "spectral_comparison_passed": bool(error < mp.mpf("1e-45") and residual < mp.mpf("1e-60")),
                "root_status": root_status, "strict_positive_energy": positivity_certificate(c)["strict_positive_all_z"],
                "zero_mode_boundary": "continuous_k_limit_and_original_variables_required_at_omega*k=0"}


def finite_wave_counterexample(*, dps=80):
    _precision(dps)
    with mp.workdps(dps):
        c = {"a": 1, "b": 1, "c": 2, "g": 1, "M2": 1, "n": 1,
             "mu_F": 1, "s": -mp.mpf(69)/20}
        return {"status": "FORMAL_COEFFICIENT_NEGATIVE_CONTROL_NOT_NVG_INPUTS",
                "coefficients": _validated(c), "K0": static_hessian(c, 0),
                "positive_energy": positivity_certificate(c),
                "zero_k": dispersion_roots(c, 0, dps=dps),
                "finite_k": dispersion_roots(c, 1, dps=dps)}


def global_energy_certificate(*, dps=80):
    """Analytic all-n result, not an extrapolation of the eight-point scan.

    Reuse the explicit quartic branch proof and validate its complete named
    identities. For original inputs R=Cv/Cs>5/4 implies C>0, mu>0 and
    0<cs2<1 on the unique positive stationary branch for EVERY n>0.
    Together with g^2-gs^2>0, the new on-shell P1/P0 factorizations imply
    the strict positive-energy condition for EVERY mathematical z>=0.
    """
    _precision(dps)
    with mp.workdps(dps):
        model = BulkModel()
        prior_checks = branch_certificate.symbolic_certificates()
        branch_certificate.require_symbolic_certificates(prior_checks)
        bounds = branch_certificate.sufficient_bounds(model.Cv/model.Cs, model.W0, model.lam)
        floor = model.gomega**2-(model.MN/model.W0)**2
        finite_number(floor, "P1 lower bound")
        proved = (bounds["unique_global_branch_certified"] is True and
                  bounds["positive_uniform_gap_certified"] is True and
                  bounds["zero_to_one_cs2_certified"] is True and floor > 0 and
                  _symbolic_passed(symbolic_checks()))
        return {"all_positive_density_all_wave_number_certified": bool(proved),
                "Cv_over_Cs": model.Cv/model.Cs, "P1_strict_lower_bound": floor,
                "uniform_scalar_gap_squared_lower_bound": bounds["C_W_lower_MeV2"],
                "prior_branch_symbolic_checks": prior_checks,
                "prior_branch_source": "verification/nvg_matter_branch_certificate.py",
                "P1_proof": "g^2-gs^2+a*(M2+2*lambda*W^2+2*gs^2*ns/m)>0",
                "P0_proof": "M2*E_WW*mu_total*cs2/n>0",
                "scope": "stationary_original_quartic_branch; frozen_isentropic_fluid; NOT_collisionless_or_gravitational_stability",
                "transverse": "positive_Hamiltonian_squares; Omega=0_and_z+M2+g^2*n/muF; zero_is_cyclic_displacement",
                "principal": "squared_characteristic_speeds={kF^2/(3*EF^2),1,1}; NOT_a_finite_k_group_velocity_bound"}


def _symbolic_passed(checks):
    return (isinstance(checks, dict) and set(checks) == EXPECTED_SYMBOLIC_CHECKS and
            all(isinstance(row, dict) and row.get("passed") is True and row.get("residual") == "0"
                for row in checks.values()))


def _numeric_passport_passed(points, modes):
    """Reject empty, truncated, duplicate or merely truthy numerical passports."""
    try:
        if (not isinstance(points, list) or len(points) != 8 or
                not isinstance(modes, list) or len(modes) != 9):
            return False
        ratios = {finite_number(p["n_ratio"], "density ratio", positive=True) for p in points}
        if ratios != {mp.mpf(r) for r in EXPECTED_DENSITY_RATIOS}:
            return False
        if {(m["n_ratio"], m["k_over_vector_mass"]) for m in modes} != EXPECTED_MODE_KEYS:
            return False
        for point in points:
            _validated(point["coefficients"])
            errors = point["independent_errors"]
            if (point.get("independent_checks_passed") is not True or
                    point["positive_energy"].get("strict_positive_all_z") is not True or
                    not isinstance(errors, dict) or set(errors) != EXPECTED_INDEPENDENT_ERRORS or
                    not all(abs(finite_number(v, "independent error")) < mp.mpf("1e-60") for v in errors.values())):
                return False
        for mode in modes:
            if (mode.get("spectral_comparison_passed") is not True or
                    mode.get("root_status") != "POSITIVE_REAL_WITHIN_NUMERIC_RESOLUTION" or
                    len(mode["Omega_roots_raw"]) != 3 or len(mode["generator_Omega_raw"]) != 6 or
                    abs(finite_number(mode["generator_max_relative_difference"], "spectral error")) >= mp.mpf("1e-45") or
                    abs(finite_number(mode["polynomial_max_scaled_residual"], "polynomial error")) >= mp.mpf("1e-60")):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _shown(value):
    if isinstance(value, mp.mpc):
        return {"real": _shown(value.real), "imag": _shown(value.imag)}
    if isinstance(value, mp.mpf):
        if not mp.isfinite(value):
            raise ArithmeticError("nonfinite numeric output")
        return mp.nstr(value, 40)
    if isinstance(value, dict):
        return {key: _shown(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shown(item) for item in value]
    return value


def compute_state():
    checks = symbolic_checks()
    points, mode_checks = [], []
    with mp.workdps(120):
        max_precision_difference = mp.mpf(0)
        for ratio in DENSITY_RATIOS:
            coarse, fine = coefficients(ratio, dps=80), coefficients(ratio, dps=120)
            delta = max(abs(coarse["coefficients"][key]-fine["coefficients"][key])/
                        max(abs(fine["coefficients"][key]), mp.mpf("1e-60")) for key in COEFFICIENT_NAMES)
            max_precision_difference = max(max_precision_difference, delta)
            points.append(fine)
            if ratio in ("0.001", "1", "100"):
                for k_ratio in ("0.01", "1", "10"):
                    z = fine["coefficients"]["M2"]*mp.mpf(k_ratio)**2
                    mode_checks.append({"n_ratio": ratio, "k_over_vector_mass": k_ratio,
                                        **dispersion_roots(fine["coefficients"], z, dps=80)})
        negative = finite_wave_counterexample(dps=80)
        global_certificate = global_energy_certificate(dps=120)
        negative_detected = (negative["positive_energy"]["strict_positive_all_z"] is False and
                             negative["finite_k"]["root_status"] == "NEGATIVE_OMEGA_INSTABILITY" and
                             negative["finite_k"]["spectral_comparison_passed"] is True)
        passed = (_symbolic_passed(checks) and global_certificate["all_positive_density_all_wave_number_certified"] is True and
                  max_precision_difference < mp.mpf("1e-60") and
                  _numeric_passport_passed(points, mode_checks) and
                  negative_detected)
        return _shown({
            "status": STATUS, "evidentiary_weight": 0, "mathematical_checks_passed": bool(passed),
            "original_inputs": INPUTS.copy(), "symbolic_checks": checks, "density_points": points,
            "coefficient_80_vs_120_max_relative_difference": max_precision_difference,
            "representative_mode_checks": mode_checks, "finite_k_negative_control": negative,
            "global_analytic_energy_certificate": global_certificate,
            "negative_control_detected": negative_detected,
            "scope": {
                "closure": "frozen_Minkowski_isentropic_cold_perfect_fluid; NOT_collisionless_NVG",
                "gravity": "lapse_shift_cuscuton_and_Jeans_modes_NOT_included",
                "density": "global_analytic_certificate_uses_quartic_branch_theorem; eight_numeric_points_are_independent_sanity_checks_only",
                "wave_number": "all_z_energy_certificate_is_mathematical_fluid_statement; NOT_hydrodynamic_validity_at_all_k",
                "roots": "raw_complex_values_retained; zero_and_negative_roots_not_clipped_or_discarded",
                "zero_k": "continuous_k_limit; fixed_global_baryon_number_and_singular_L_variable_need_separate_treatment",
                "excluded": "time_dependent_amplification_quantum_closure_strong_coupling_microcausality_and_observational_confirmation",
            },
        })


def main():
    try:
        result = compute_state()
        if not isinstance(result, dict):
            raise ValueError("audit report must be a dictionary")
        passed = result.get("mathematical_checks_passed") is True
        result = {**result, "mathematical_checks_passed": passed}
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    except Exception as exc:
        print(json.dumps({"status": "INVALID_OR_FAILED_LONGITUDINAL_AUDIT", "error": str(exc),
                          "mathematical_checks_passed": False, "evidentiary_weight": 0}, allow_nan=False))
        return 2
    print(rendered)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
