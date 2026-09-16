#!/usr/bin/env python3
"""Low-frequency effective response after the longitudinal Gauss constraint.

This module is a bounded mathematical audit of the frozen, homogeneous
source-complete matter action.  It keeps the real sine/cosine amplitudes of the
longitudinal vector and removes the nondynamical temporal vector first.  The
remaining finite-dimensional conservative system is written as

    D(omega) = H + i*omega*B - omega**2*M,

with real symmetric ``H`` and ``M`` and real antisymmetric ``B``.  A second
Schur reduction removes a gapped vector coordinate at low frequency.  The
result is a response/inertia statement for this local fluid closure, not a
stability theorem for the full NVG, cuscuton, gravitational, periodic, or
quantum problem.  The CLI prints JSON only and never writes an answer file.

The numerical NVG connection reuses the equilibrium coefficients produced by
``nvg_longitudinal_fluid_audit``.  Its fixed decimal inputs are assumptions,
not fitted constants or result tables.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

import mpmath as mp
import sympy as sp

import nvg_longitudinal_fluid_audit as fluid


STATUS = "CONDITIONAL_GYROSCOPIC_LOW_FREQUENCY_RESPONSE_NOT_FULL_NVG_STABILITY"
COEFFICIENT_NAMES = tuple(fluid.COEFFICIENT_NAMES)
REQUIRED_SYMBOLIC_CHECKS = frozenset({
    "gauss_solution_from_real_action",
    "real_action_reconstruction",
    "reduced_H_symmetric",
    "reduced_M_symmetric",
    "reduced_B_antisymmetric",
    "generic_H_block_diagonal",
    "generic_C_definition",
    "generic_Heff",
    "generic_Beff",
    "generic_Meff",
    "generic_second_order_schur",
    "longitudinal_determinant_bridge",
    "static_hessian_bridge",
    "acoustic_inertial_limit",
})
REPRESENTATIVE_DENSITIES = ("1", "10")
REFERENCES = (
    "scratch/nvg-longitudinal-fluid-derivation-2026-09-09.md",
    "verification/contracts/source_complete_action.md",
    "https://arxiv.org/html/2411.06960v1",
    "https://arxiv.org/abs/0910.3283",
)


def finite(value, name, *, positive=False, nonnegative=False):
    """Convert a scalar to finite mpmath real and reject bool/complex input."""
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool is not a physical number")
    if isinstance(value, complex):
        raise ValueError(f"{name}: real number required")
    try:
        result = mp.mpf(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name}: finite real number required") from exc
    if (not mp.isfinite(result) or (positive and result <= 0)
            or (nonnegative and result < 0)):
        raise ValueError(f"{name}: outside finite numeric domain")
    return result


def precision(dps):
    if isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 200:
        raise ValueError("dps must be an integer in [80,200]")


def number(value):
    """Stable decimal display for a finite real derived quantity."""
    return mp.nstr(finite(value, "derived number"), 35)


def _validated_coefficients(coefficients):
    if not isinstance(coefficients, dict) or not set(COEFFICIENT_NAMES) <= set(coefficients):
        raise ValueError("missing longitudinal coefficient")
    positive = {"a", "M2", "n", "mu_F"}
    return {key: finite(coefficients[key], key, positive=key in positive)
            for key in COEFFICIENT_NAMES}


def _matrix(value, name, *, size=None):
    """Copy a real square matrix into mpmath storage with shape validation."""
    if isinstance(value, mp.matrix):
        rows, cols = value.rows, value.cols
        if rows != cols:
            raise ValueError(f"{name}: square matrix required")
        result = mp.matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                result[i, j] = finite(value[i, j], f"{name}[{i},{j}]")
    else:
        try:
            rows = list(value)
        except TypeError as exc:
            raise ValueError(f"{name}: square matrix required") from exc
        if not rows or any(not hasattr(row, "__len__") for row in rows):
            raise ValueError(f"{name}: square matrix required")
        cols = len(rows)
        if len(rows) != cols or any(len(row) != cols for row in rows):
            raise ValueError(f"{name}: square matrix required")
        result = mp.matrix(cols, cols)
        for i, row in enumerate(rows):
            for j, entry in enumerate(row):
                result[i, j] = finite(entry, f"{name}[{i},{j}]")
    if size is not None and result.rows != size:
        raise ValueError(f"{name}: expected {size} by {size}")
    return result


def _block(matrix, row0, row1, col0, col1):
    result = mp.matrix(row1-row0, col1-col0)
    for i in range(row0, row1):
        for j in range(col0, col1):
            result[i-row0, j-col0] = matrix[i, j]
    return result


def matrix_norm(matrix):
    matrix = mp.matrix(matrix)
    return mp.sqrt(mp.fsum(abs(matrix[i, j])**2
                            for i in range(matrix.rows)
                            for j in range(matrix.cols)))


def _relative_matrix_error(left, right):
    scale = max(matrix_norm(left), matrix_norm(right), mp.mpf("1e-70"))
    return matrix_norm(mp.matrix(left)-mp.matrix(right))/scale


def _validate_system(H, M, B, q_dim):
    if isinstance(q_dim, bool) or not isinstance(q_dim, int):
        raise ValueError("q_dim must be an integer")
    H = _matrix(H, "H")
    size = H.rows
    if not 1 <= q_dim < size:
        raise ValueError("q_dim must leave at least one excluded coordinate")
    M, B = _matrix(M, "M", size=size), _matrix(B, "B", size=size)
    scale = max(matrix_norm(H), matrix_norm(M), matrix_norm(B), mp.mpf(1))
    tolerance = mp.mpf("1e-60")*scale
    if matrix_norm(H-H.T) > tolerance:
        raise ValueError("H must be real symmetric")
    if matrix_norm(M-M.T) > tolerance:
        raise ValueError("M must be real symmetric")
    if matrix_norm(B+B.T) > tolerance:
        raise ValueError("B must be real antisymmetric")
    return H, M, B, size


def _invertible(Hzz, name="Hzz"):
    determinant = mp.det(Hzz)
    scale = max(matrix_norm(Hzz), mp.mpf(1))**Hzz.rows
    if not mp.isfinite(determinant) or abs(determinant) <= mp.mpf("1e-70")*scale:
        raise ValueError(f"{name} is singular or unresolved")
    return Hzz**-1


def low_frequency_effective(H, M, B, q_dim, *, correction_sign=1):
    """Return the regular Schur expansion through O(omega**2).

    Coordinates are transformed with ``x = R*q + Ez*z`` where
    ``R=[I;-Hzz^-1*Hzq]``.  The returned convention is exactly
    ``Heff + i*omega*Beff - omega**2*Meff``.  ``correction_sign=-1`` is an
    explicit adverse-control mutation and is never accepted as the formula.
    """
    if isinstance(correction_sign, bool) or correction_sign not in (1, -1):
        raise ValueError("correction_sign must be +1 or -1")
    H, M, B, size = _validate_system(H, M, B, q_dim)
    z_dim = size-q_dim
    Hqq = _block(H, 0, q_dim, 0, q_dim)
    Hzq = _block(H, q_dim, size, 0, q_dim)
    Hzz = _block(H, q_dim, size, q_dim, size)
    Hzz_inv = _invertible(Hzz)
    R = mp.matrix(size, q_dim)
    for i in range(q_dim):
        for j in range(q_dim):
            R[i, j] = 1 if i == j else 0
    tail = -(Hzz_inv*Hzq)
    for i in range(z_dim):
        for j in range(q_dim):
            R[q_dim+i, j] = tail[i, j]
    Ez = mp.matrix(size, z_dim)
    for i in range(z_dim):
        Ez[q_dim+i, i] = 1
    C = R.T*B*Ez
    Heff = R.T*H*R
    Beff = R.T*B*R
    Mbase = R.T*M*R
    Meff = Mbase + correction_sign*(C*Hzz_inv*C.T)
    return {"H": H, "M": M, "B": B, "R": R, "Ez": Ez,
            "Hqq": Hqq, "Hzq": Hzq, "Hzz": Hzz, "Hzz_inv": Hzz_inv,
            "C": C, "Heff": Heff, "Beff": Beff, "Mbase": Mbase,
            "Meff": Meff, "correction_sign": correction_sign}


def exact_schur(H, M, B, omega, q_dim):
    """Exact finite-frequency Schur complement away from excluded poles."""
    H, M, B, size = _validate_system(H, M, B, q_dim)
    omega = finite(omega, "omega")
    D = H + mp.j*omega*B - omega**2*M
    Dqq = _block(D, 0, q_dim, 0, q_dim)
    Dqz = _block(D, 0, q_dim, q_dim, size)
    Dzq = _block(D, q_dim, size, 0, q_dim)
    Dzz = _block(D, q_dim, size, q_dim, size)
    Dzz_inv = _invertible(Dzz, "Dzz(omega)")
    return Dqq-Dqz*Dzz_inv*Dzq


def gauss_eliminated_matrices(coefficients, k):
    """Build the real sine/cosine action after eliminating ``phi=delta A0``.

    The retained real amplitudes are ``q=(xi_s,w_c,A_s)`` with
    ``delta n=-n*k*xi_s``.  The temporal vector solves
    ``phi_c=(k*dot(A_s)-(c*w_c+g*n*k*xi_s))/(k**2+M2)``.  No ``omega*k*A``
    amplitude is introduced, so this expression remains meaningful at
    ``omega=0`` and at ``k=0``.
    """
    c = _validated_coefficients(coefficients)
    k = finite(k, "k", nonnegative=True)
    a, b, cmix, g, M2, n, mu, s = (c[key] for key in COEFFICIENT_NAMES)
    kwave = k*k+M2
    H = mp.matrix([
        [n*n*k*k*(a+g*g/kwave), n*k*(-b+g*cmix/kwave), 0],
        [n*k*(-b+g*cmix/kwave), k*k+s+cmix*cmix/kwave, 0],
        [0, 0, M2],
    ])
    M = mp.matrix([[n*mu, 0, 0], [0, 1, 0], [0, 0, M2/kwave]])
    # For exp(-i omega t), D=H+i*omega*B-omega**2*M and B=G.T-G.
    B = mp.matrix([
        [0, 0, -g*n*M2/kwave],
        [0, 0, k*cmix/kwave],
        [g*n*M2/kwave, -k*cmix/kwave, 0],
    ])
    return {"H": H, "M": M, "B": B, "k": k, "k_squared": k*k,
            "wave_denominator": kwave, "phi_solution": "(k*dot(A)-c*w-g*n*k*xi)/(k^2+M2)"}


@lru_cache(maxsize=1)
def _fluid_slope_function():
    model = fluid._symbolic_model()
    Om, z = model["Om"], model["z"]
    slope = -sp.diff(model["determinant"], z).subs({Om: 0, z: 0}) / sp.diff(
        model["determinant"], Om).subs({Om: 0, z: 0})
    return sp.lambdify((*model["symbols"],), sp.factor(slope), "mpmath")


@lru_cache(maxsize=1)
def _fluid_determinant_function():
    model = fluid._symbolic_model()
    return sp.lambdify((*model["symbols"], model["Om"], model["z"]),
                       model["determinant"], "mpmath")


def determinant_bridge(coefficients, k, omega):
    """Compare the Gauss-reduced three-field determinant to the live 4x4 one."""
    c = _validated_coefficients(coefficients)
    k, omega = finite(k, "k", nonnegative=True), finite(omega, "omega")
    matrices = gauss_eliminated_matrices(c, k)
    D3 = matrices["H"]+mp.j*omega*matrices["B"]-omega**2*matrices["M"]
    det3 = mp.det(D3)
    args = [c[key] for key in COEFFICIENT_NAMES]
    det4 = _fluid_determinant_function()(*args, omega**2, k*k)
    expected = -c["n"]**2/(k*k+c["M2"])*det4
    residual = abs(det3-expected)/max(abs(det3), abs(expected), mp.mpf("1e-70"))
    return {"determinant_reduction_relative_error": residual,
            "det3": det3, "expected_det3": expected, "omega": omega, "k": k}


def static_bridge(coefficients, k):
    """Check Hqq=T^T K(k^2) T against the maintained static Hessian."""
    c = _validated_coefficients(coefficients)
    k = finite(k, "k", nonnegative=True)
    matrices = gauss_eliminated_matrices(c, k)
    Kfluid = mp.matrix(fluid.static_hessian(c, k*k))
    T = mp.matrix([[-c["n"]*k, 0], [0, 1]])
    expected = T.T*Kfluid*T
    actual = _block(matrices["H"], 0, 2, 0, 2)
    return {"static_hessian_relative_error": _relative_matrix_error(actual, expected),
            "Hqq": actual, "expected_Hqq": expected}


def acoustic_limit(coefficients, *, dps=80, n_ratio=None):
    """Derive the k->0 acoustic slope and compare to the current determinant.

    A source comparison exists only when a density ratio is supplied.  The
    determinant slope is calculated from the supplied coefficient dictionary
    in either case, but it is not an independent source value and therefore
    must not be copied into the source-comparison fields when ``n_ratio`` is
    omitted.
    """
    precision(dps)
    with mp.workdps(dps):
        c = _validated_coefficients(coefficients)
        matrices = gauss_eliminated_matrices(c, 0)
        low = low_frequency_effective(matrices["H"], matrices["M"], matrices["B"], 2)
        Hqq = low["Heff"]
        H00_over_k2 = c["n"]**2*(c["a"]+c["g"]**2/c["M2"])
        H01_over_k = -c["n"]*(c["b"]-c["g"]*c["c"]/c["M2"])
        H11 = Hqq[1, 1]
        inertial = low["Meff"][0, 0]
        relaxed_stiffness = H00_over_k2-H01_over_k**2/H11
        slope = relaxed_stiffness/inertial
        determinant_slope = _fluid_slope_function()(*[c[key] for key in COEFFICIENT_NAMES])
        source_comparison_checked = n_ratio is not None
        if source_comparison_checked:
            source = fluid.coefficients(n_ratio, dps=dps)
            source_slope = source["acoustic_cs2"]
            source_relative_error = abs(slope-source_slope)/max(
                abs(slope), abs(source_slope), mp.mpf("1e-70"))
        else:
            source_slope = None
            source_relative_error = None
        return {"acoustic_squared_speed": slope, "determinant_slope": determinant_slope,
                "source_acoustic_squared_speed": source_slope,
                "determinant_relative_error": abs(slope-determinant_slope)/max(abs(slope), abs(determinant_slope), mp.mpf("1e-70")),
                "source_relative_error": source_relative_error,
                "source_comparison_checked": source_comparison_checked,
                "low_frequency_inertial_coefficient": inertial,
                "expected_inertial_coefficient": c["n"]*c["mu_F"]+c["g"]**2*c["n"]**2/c["M2"],
                "inertial_relative_error": abs(inertial-(c["n"]*c["mu_F"]+c["g"]**2*c["n"]**2/c["M2"])) / max(abs(inertial), 1),
                "finite_k_group_velocity_not_claimed": True}


def finite_frequency_check(coefficients, k, *, dps=80, correction_sign=1):
    """Compare exact and O(omega^2) Schur responses at two non-pole frequencies."""
    precision(dps)
    c = _validated_coefficients(coefficients)
    k = finite(k, "k", nonnegative=True)
    with mp.workdps(dps):
        matrices = gauss_eliminated_matrices(c, k)
        low = low_frequency_effective(matrices["H"], matrices["M"], matrices["B"], 2,
                                      correction_sign=correction_sign)
        gap = mp.sqrt(k*k+c["M2"])
        errors = []
        pole_distances = []
        for ratio in (mp.mpf(1)/32, mp.mpf(1)/64):
            omega = ratio*gap
            exact = exact_schur(matrices["H"], matrices["M"], matrices["B"], omega, 2)
            approx = low["Heff"]+mp.j*omega*low["Beff"]-omega**2*low["Meff"]
            errors.append(_relative_matrix_error(exact, approx))
            pole_distances.append(1-ratio**2)
        pass_value = (correction_sign == 1 and errors[1] < errors[0]
                      and errors[1] < mp.mpf("1e-3"))
        return {"correction_sign": correction_sign, "frequency_ratios_of_vector_gap": ["1/32", "1/64"],
                "relative_errors_exact_vs_expanded": errors, "pole_distance_fractions": pole_distances,
                "away_from_excluded_poles": all(v > mp.mpf("0.9") for v in pole_distances),
                "expanded_response_passed": bool(pass_value)}


def _eigenvalues(matrix):
    matrix = mp.matrix(matrix)
    values = mp.eigsy(matrix, eigvals_only=True)
    return [values[i] for i in range(len(values))]


def counterexamples(*, dps=80):
    """Positive and saddle controls for the Schur positivity implication."""
    precision(dps)
    with mp.workdps(dps):
        H_positive = mp.matrix([[2, 1], [1, 3]])
        M_positive = mp.matrix([[1, 0], [0, 2]])
        B_positive = mp.matrix([[0, 2], [-2, 0]])
        positive = low_frequency_effective(H_positive, M_positive, B_positive, 1)
        wrong = low_frequency_effective(H_positive, M_positive, B_positive, 1,
                                        correction_sign=-1)
        omega = mp.mpf("1e-3")
        exact = exact_schur(H_positive, M_positive, B_positive, omega, 1)
        approx = positive["Heff"]+mp.j*omega*positive["Beff"]-omega**2*positive["Meff"]
        wrong_approx = wrong["Heff"]+mp.j*omega*wrong["Beff"]-omega**2*wrong["Meff"]
        H_indefinite = mp.matrix([[2, 0], [0, -1]])
        indefinite = low_frequency_effective(H_indefinite, M_positive, B_positive, 1)
        positive_meff = _eigenvalues(positive["Meff"])
        indefinite_meff = _eigenvalues(indefinite["Meff"])
        return {
            "positive_Hzz_and_M": {
                "min_M_eigenvalue": min(_eigenvalues(M_positive)),
                "Hzz": H_positive[1, 1],
                "Meff_eigenvalues": positive_meff,
                "positivity_implication_passed": bool(min(positive_meff) > 0),
                "exact_expansion_relative_error": _relative_matrix_error(exact, approx),
            },
            "indefinite_Hzz_counterexample": {
                "min_M_eigenvalue": min(_eigenvalues(M_positive)),
                "Hzz": H_indefinite[1, 1],
                "Meff_eigenvalues": indefinite_meff,
                "positivity_certificate_applicable": False,
                "negative_effective_inertia_visible": bool(min(indefinite_meff) < 0),
                "general_schur_algebra_still_defined": True,
            },
            "wrong_sign_control": {
                "corrected_formula_error": _relative_matrix_error(exact, approx),
                "minus_C_Hinv_Ct_error": _relative_matrix_error(exact, wrong_approx),
                "wrong_sign_rejected": bool(_relative_matrix_error(exact, wrong_approx)
                                             > _relative_matrix_error(exact, approx)*100),
            },
        }


def _symbolic_row(expression):
    residual = sp.factor(sp.simplify(expression))
    if isinstance(residual, sp.MatrixBase):
        passed = all(sp.simplify(entry) == 0 for entry in residual)
    else:
        passed = bool(residual == 0)
    return {"passed": bool(passed), "residual": "0" if passed else str(residual)}


@lru_cache(maxsize=4)
def _symbolic_payload():
    # Real sine/cosine reduction.  xi,A are sine amplitudes; w,phi are cosine
    # amplitudes, making delta n=-n*k*xi and E=dot(A)-k*phi real.
    a, b, cmix, g, M2, n, mu, s, k, omega = sp.symbols(
        "a b c g M2 n mu s k omega", nonzero=True, real=True)
    xi, w, A, phi, xid, wd, Ad = sp.symbols("xi w A phi xid wd Ad", real=True)
    K = k*k+M2
    J = cmix*w+g*n*k*xi
    L = (n*mu*xid*xid/2-a*n*n*k*k*xi*xi/2+b*n*k*xi*w
         +wd*wd/2-(k*k+s)*w*w/2+(Ad-k*phi)**2/2+M2*phi*phi/2
         -M2*A*A/2+J*phi+g*n*xid*A)
    phi_solution = sp.solve(sp.diff(L, phi), phi)[0]
    L_reduced = sp.factor(L.subs(phi, phi_solution))
    H = sp.Matrix([[n*n*k*k*(a+g*g/K), n*k*(-b+g*cmix/K), 0],
                   [n*k*(-b+g*cmix/K), k*k+s+cmix*cmix/K, 0], [0, 0, M2]])
    M = sp.diag(n*mu, 1, M2/K)
    G = sp.Matrix([[0, 0, g*n], [0, 0, 0], [g*n*k*k/K, k*cmix/K, 0]])
    B = G.T-G
    q = sp.Matrix([xi, w, A])
    dq = sp.Matrix([xid, wd, Ad])
    reconstructed = (dq.T*M*dq)[0]/2+(dq.T*G*q)[0]-(q.T*H*q)[0]/2

    # One retained and two excluded coordinates give a nontrivial generic
    # composition test for the block-Schur formula.
    h, u, v, p, r, t = sp.symbols("h u v p r t", real=True)
    m00, m01, m02, m11, m12, m22 = sp.symbols("m00 m01 m02 m11 m12 m22", real=True)
    b01, b02, b12 = sp.symbols("b01 b02 b12", real=True)
    Hg = sp.Matrix([[h, u, v], [u, p, r], [v, r, t]])
    Mg = sp.Matrix([[m00, m01, m02], [m01, m11, m12], [m02, m12, m22]])
    Bg = sp.Matrix([[0, b01, b02], [-b01, 0, b12], [-b02, -b12, 0]])
    Hzz = Hg[1:, 1:]
    Hzz_inv = Hzz.inv()
    Hzq = Hg[1:, 0]
    R = sp.Matrix.vstack(sp.Matrix([[1]]), -Hzz_inv*Hzq)
    Ez = sp.Matrix([[0, 0], [1, 0], [0, 1]])
    P = sp.Matrix.hstack(R, Ez)
    Hg_t, Mg_t, Bg_t = P.T*Hg*P, P.T*Mg*P, P.T*Bg*P
    Heff = (R.T*Hg*R)[0]
    Beff = (R.T*Bg*R)[0]
    C = R.T*Bg*Ez
    Mbase = (R.T*Mg*R)[0]
    correction = (C*Hzz_inv*C.T)[0]
    Dg = Hg_t+sp.I*omega*Bg_t-omega*omega*Mg_t
    Dqq, Dqz, Dzq, Dzz = Dg[0, 0], Dg[0, 1:], Dg[1:, 0], Dg[1:, 1:]
    Deff = sp.factor(Dqq-(Dqz*Dzz.inv()*Dzq)[0])

    # Use the maintained determinant object itself for the bridge identity.
    fm = fluid._symbolic_model()
    fa, fb, fc, fg, fM2, fn, fmu, fs = fm["symbols"]
    fk, fom = sp.symbols("fk fom", nonzero=True, real=True)
    fK = fk*fk+fM2
    fH = sp.Matrix([[fn*fn*fk*fk*(fa+fg*fg/fK), fn*fk*(-fb+fg*fc/fK), 0],
                    [fn*fk*(-fb+fg*fc/fK), fk*fk+fs+fc*fc/fK, 0], [0, 0, fM2]])
    fM = sp.diag(fn*fmu, 1, fM2/fK)
    fB = sp.Matrix([[0, 0, -fg*fn*fM2/fK], [0, 0, fk*fc/fK],
                    [fg*fn*fM2/fK, -fk*fc/fK, 0]])
    fD3 = fH+sp.I*fom*fB-fom*fom*fM
    fD3det = sp.factor(fD3.det())
    fD4det = fm["determinant"]
    relation = sp.factor(fD3det.subs({fom**2: fm["Om"], fk**2: fm["z"]})
                         + fn**2/(fM2+fm["z"])*fD4det)
    fKstatic = fm["K"].subs(fm["z"], fk*fk)
    T = sp.diag(-fn*fk, 1)
    static_relation = sp.factor((fH[:2, :2]-T.T*fKstatic*T))
    Dstatic = fa+fg*fg/fM2
    Bstatic = fb-fg*fc/fM2
    Cstatic = fs+fc*fc/fM2
    mu_total = fmu+fg*fg*fn/fM2
    inertial = fn*fmu+fg*fg*fn*fn/fM2
    acoustic_relation = sp.factor(
        (fn*fn*(Dstatic-Bstatic*Bstatic/Cstatic))/inertial
        -fn/mu_total*(Dstatic-Bstatic*Bstatic/Cstatic))
    return {"gauss_solution": sp.factor(phi_solution-(k*Ad-J)/K),
            "real_action": sp.factor(L_reduced-reconstructed),
            "H": H, "M": M, "B": B,
            "generic_Ht": Hg_t, "generic_Bt": Bg_t, "generic_Mt": Mg_t,
            "generic_Heff": Heff, "generic_Beff": Beff, "generic_C": C,
            "generic_Mbase": Mbase, "generic_correction": correction,
            "generic_Deff": Deff, "generic_Meff": Mbase+correction,
            "generic_H_block": Hg_t[0, 1:],
            "generic_B_qz": Bg_t[0, 1:],
            "generic_first": sp.diff(Deff, omega).subs(omega, 0)-sp.I*Beff,
            "generic_second": sp.diff(Deff, omega, 2).subs(omega, 0)
            +2*(Mbase+correction),
            "bridge": relation, "static": static_relation,
            "acoustic": acoustic_relation}


def symbolic_checks(*, correction_sign=1):
    """Return exact symbolic residuals; adverse sign is a deliberate control."""
    if isinstance(correction_sign, bool) or correction_sign not in (1, -1):
        raise ValueError("correction_sign must be +1 or -1")
    p = _symbolic_payload()
    H, M, B = p["H"], p["M"], p["B"]
    checks = {
        "gauss_solution_from_real_action": p["gauss_solution"],
        "real_action_reconstruction": p["real_action"],
        "reduced_H_symmetric": H-H.T,
        "reduced_M_symmetric": M-M.T,
        "reduced_B_antisymmetric": B+B.T,
        "generic_H_block_diagonal": p["generic_H_block"],
        "generic_C_definition": p["generic_B_qz"]-p["generic_C"],
        "generic_Heff": p["generic_Ht"][0, 0]-p["generic_Heff"],
        "generic_Beff": p["generic_first"],
        "generic_Meff": p["generic_second"] if correction_sign == 1 else
                         p["generic_second"]+4*p["generic_correction"],
        "generic_second_order_schur": p["generic_second"],
        "longitudinal_determinant_bridge": p["bridge"],
        "static_hessian_bridge": p["static"],
        "acoustic_inertial_limit": p["acoustic"],
    }
    return {name: _symbolic_row(value) for name, value in checks.items()}


def _symbolic_passed(checks):
    return (isinstance(checks, dict) and set(checks) == REQUIRED_SYMBOLIC_CHECKS
            and all(isinstance(row, dict) and row.get("passed") is True
                    and row.get("residual") == "0" for row in checks.values()))


def _shown_matrix(matrix):
    return [[number(matrix[i, j]) for j in range(matrix.cols)]
            for i in range(matrix.rows)]


def nvg_connection(n_ratio="1", k_MeV="50", *, dps=80):
    """Run the effective-response calculation at one live equilibrium state."""
    precision(dps)
    with mp.workdps(dps):
        source = fluid.coefficients(n_ratio, dps=dps)
        if not isinstance(source, dict):
            raise ValueError("source coefficient producer must return a dictionary")
        source_independent_checks_passed = source.get("independent_checks_passed")
        coefficients = {key: source["coefficients"][key] if key == "mu_F" else
                        source["coefficients"][key] for key in COEFFICIENT_NAMES}
        # The fluid producer uses exactly these names, including mu_F.
        k = finite(k_MeV, "k_MeV", nonnegative=True)
        matrices = gauss_eliminated_matrices(coefficients, k)
        low = low_frequency_effective(matrices["H"], matrices["M"], matrices["B"], 2)
        static = static_bridge(coefficients, k)
        frequencies = (mp.mpf(1)/32*mp.sqrt(k*k+coefficients["M2"]),
                       mp.mpf(1)/64*mp.sqrt(k*k+coefficients["M2"]))
        bridges = [determinant_bridge(coefficients, k, omega)
                   for omega in frequencies]
        expansion = finite_frequency_check(coefficients, k, dps=dps)
        acoustic = acoustic_limit(coefficients, dps=dps, n_ratio=n_ratio)
        M_eigs = _eigenvalues(matrices["M"])
        Hzz = matrices["H"][2, 2]
        checks_passed = (static["static_hessian_relative_error"] < mp.mpf("1e-60")
                         and all(row["determinant_reduction_relative_error"] < mp.mpf("1e-55")
                                 for row in bridges)
                         and expansion["expanded_response_passed"]
                         and acoustic["determinant_relative_error"] < mp.mpf("1e-55")
                         and acoustic["source_comparison_checked"] is True
                         and acoustic["source_relative_error"] < mp.mpf("1e-55")
                         and acoustic["inertial_relative_error"] < mp.mpf("1e-60")
                         and source_independent_checks_passed is True
                         and min(M_eigs) > 0 and Hzz > 0)
        return {
            "n_ratio": source["n_ratio"], "k_MeV": k,
            "units": "natural_hbar_c_1; k and omega in MeV",
            "source_producer": "verification/nvg_longitudinal_fluid_audit.py",
            "source_independent_checks_passed": source_independent_checks_passed,
            "Hzz_MeV2": Hzz, "M_eigenvalues": M_eigs,
            "H_effective": _shown_matrix(low["Heff"]),
            "B_effective": _shown_matrix(low["Beff"]),
            "M_effective": _shown_matrix(low["Meff"]),
            "static": {"relative_error": static["static_hessian_relative_error"]},
            "determinant_bridges": [{"omega_MeV": row["omega"],
                                     "relative_error": row["determinant_reduction_relative_error"]}
                                    for row in bridges],
            "finite_frequency": {key: value for key, value in expansion.items()
                                  if key != "correction_sign"},
            "acoustic": acoustic,
            "mathematical_checks_passed": bool(checks_passed),
            "scope": [
                "Frozen Minkowski local isentropic fluid closure at T=0, n>0, W>0.",
                "The temporal vector is a Gauss variable; the retained real A amplitude is not omega*k*A.",
                "Low-frequency Schur expansion is regular only away from det(Hzz)=0 and excluded finite-frequency poles.",
                "M>0 and Hzz>0 are checked here; no positivity claim is made for saddle constraints or singular/gapless limits.",
                "Finite-k response and k->0 acoustic scaling are reported separately.",
                "No periodic coefficients, cuscuton/gravity perturbations, quantum loops, collisions, or empirical fit are included.",
            ],
        }


def audit(n_ratio="1", k_MeV="50", *, dps=80):
    precision(dps)
    with mp.workdps(dps):
        symbolic = symbolic_checks()
        connection = nvg_connection(n_ratio, k_MeV, dps=dps)
        controls = counterexamples(dps=dps)
        wrong_symbolic = symbolic_checks(correction_sign=-1)
        passed = (_symbolic_passed(symbolic)
                  and not _symbolic_passed(wrong_symbolic)
                  and connection["mathematical_checks_passed"]
                  and controls["positive_Hzz_and_M"]["positivity_implication_passed"]
                  and controls["indefinite_Hzz_counterexample"]["negative_effective_inertia_visible"]
                  and controls["wrong_sign_control"]["wrong_sign_rejected"])
        return {
            "status": STATUS, "evidence_weight": 0,
            "mathematical_checks_passed": bool(passed),
            "symbolic_checks": symbolic,
            "connection": connection,
            "counterexamples": controls,
            "wrong_sign_symbolic_checks_passed": _symbolic_passed(wrong_symbolic),
            "assumptions": [
                "Stationary conservative action with real symmetric H,M and real antisymmetric B.",
                "Hzz is invertible and the finite-frequency Schur point is away from excluded poles.",
                "The response formula is an O(omega^2) local expansion, not an all-frequency elimination.",
                "If M>0 and Hzz>0, Meff>0; this implication is not extended to an indefinite Hzz.",
            ],
            "next_missing_closure": "A source-complete finite-k quantum/Vlasov response including collisions or particle-hole continua, then coupled lapse/shift/cuscuton/gravity constraints; neither is supplied by this local audit.",
            "references": list(REFERENCES),
        }


def compute_state(*, k_MeV="50", dps=80):
    """Run the same no-write audit at the two fixed representative densities."""
    precision(dps)
    with mp.workdps(dps):
        symbolic = symbolic_checks()
        connections = [nvg_connection(ratio, k_MeV, dps=dps)
                       for ratio in REPRESENTATIVE_DENSITIES]
        controls = counterexamples(dps=dps)
        wrong_symbolic = symbolic_checks(correction_sign=-1)
        passed = (_symbolic_passed(symbolic)
                  and all(row["mathematical_checks_passed"] for row in connections)
                  and not _symbolic_passed(wrong_symbolic)
                  and controls["positive_Hzz_and_M"]["positivity_implication_passed"]
                  and controls["indefinite_Hzz_counterexample"]["negative_effective_inertia_visible"]
                  and controls["wrong_sign_control"]["wrong_sign_rejected"])
        return {
            "status": STATUS, "evidence_weight": 0,
            "mathematical_checks_passed": bool(passed),
            "representative_density_ratios": list(REPRESENTATIVE_DENSITIES),
            "connections": connections,
            "symbolic_checks": symbolic,
            "counterexamples": controls,
            "wrong_sign_symbolic_checks_passed": _symbolic_passed(wrong_symbolic),
            "assumptions": [
                "Stationary conservative action with real symmetric H,M and real antisymmetric B.",
                "Hzz is invertible and the finite-frequency Schur point is away from excluded poles.",
                "The response formula is an O(omega^2) local expansion, not an all-frequency elimination.",
                "If M>0 and Hzz>0, Meff>0; this implication is not extended to an indefinite Hzz.",
            ],
            "next_missing_closure": "A source-complete finite-k quantum/Vlasov response including collisions or particle-hole continua, then coupled lapse/shift/cuscuton/gravity constraints; neither is supplied by this local audit.",
            "references": list(REFERENCES),
        }


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--n-ratio", default="1")
    parser.add_argument("--k-MeV", default="50")
    parser.add_argument("--dps", type=int, default=80)
    try:
        args = parser.parse_args(argv)
        result = audit(args.n_ratio, args.k_MeV, dps=args.dps)
    except (ValueError, ArithmeticError, RuntimeError, ZeroDivisionError) as exc:
        print(json.dumps({"status": "invalid_or_failed_effective_response_audit",
                          "evidence_weight": 0, "mathematical_checks_passed": False,
                          "error": str(exc)}, ensure_ascii=False, allow_nan=False))
        return 2
    if result.get("mathematical_checks_passed") is not True:
        result["mathematical_checks_passed"] = False
    # ``audit`` deliberately contains only JSON-native values and mpmath
    # numbers are converted by this recursive encoder at the output boundary.
    def encode(value):
        if isinstance(value, (mp.mpf, mp.mpc)):
            if isinstance(value, mp.mpc) and value.imag:
                return {"real": number(value.real), "imag": number(value.imag)}
            return number(value.real if isinstance(value, mp.mpc) else value)
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        return value
    print(json.dumps(encode(result), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["mathematical_checks_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
