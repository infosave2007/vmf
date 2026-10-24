#!/usr/bin/env python3
"""Symbolic verification of the second-order Hartle(-Thorne) system.

Sources compared in this script:

* Yagi & Yunes (2013), PRD 88, 023009, arXiv:1303.1528 (TeX: /tmp/yy2-tex)
  -- the l=2 interior system in (h2, K2, m2) and the exterior closed forms.
* Urbanec, Miller & Stuchlik (2013), MNRAS 433, 1906, arXiv:1301.5925
  (TeX: /tmp/ums-tex) -- the l=0 (m0, p0) system with centre data.

Conventions (Hartle 1967, same as verification/nvg_hartle_slow_rotation.py):
g_tt = -exp(nu) (single exponent), exp(lambda) = (1-2m/r)^{-1},
j^2 = exp(-(nu+lambda)), omega_tilde = Omega - omega (frame dragging complement).

Sections: A. Legendre closed forms; B. vacuum l=2 system vs exteriors;
C. l=0 system (vacuum + centre series, j_c^2 carried explicitly);
D. l=2 centre initial data (h2=B r^2, K2=-B r^2, m2=-B r^3) consistency
   with the interior ODEs at leading series orders; E. Maclaurin limit.

Each check passes when the residual simplifies to literal zero or evaluates
to < 1e-24 (30-digit arithmetic) at four independent rational sample points.
"""

from __future__ import annotations

import sympy as sp

R, M, S, A, Om = sp.symbols("R M S A Om", positive=True)
rho_c, p_c, cEpdP, om_c = sp.symbols("rho_c p_c cEpdP om_c", positive=True)

f = 1 - sp.Rational(2) * M / R                      # 1 - 2M/R
xs = sp.symbols("xs", positive=True)
Ls = sp.log((xs + 1) / (xs - 1))
Q2x = sp.Rational(1, 4) * (3 * xs**2 - 1) * Ls - sp.Rational(3, 2) * xs
Q2 = Q2x.subs(xs, R / M - 1)
dQ2 = sp.diff(Q2x, xs).subs(xs, R / M - 1)
d2Q2 = sp.diff(Q2x, xs, 2).subs(xs, R / M - 1)
x = R / M - 1
Q21_nocs = sp.sqrt(x**2 - 1) * dQ2                   # no Condon-Shortley
Q21_cs = -sp.sqrt(x**2 - 1) * dQ2                     # with Condon-Shortley
Q22 = (x**2 - 1) * d2Q2                               # m=2: CS == no-CS

FAILURES: list[str] = []
_POINTS = [
    {M: sp.Rational(1), R: sp.Rational(10), S: sp.Rational(7, 10), A: sp.Rational(3, 10), Om: sp.Rational(2)},
    {M: sp.Rational(3, 2), R: sp.Rational(30), S: sp.Rational(1, 3), A: sp.Rational(5, 7), Om: sp.Rational(9, 4)},
    {M: sp.Rational(1), R: sp.Rational(25, 4), S: sp.Rational(11, 5), A: sp.Rational(1, 9), Om: sp.Rational(1, 2)},
    {M: sp.Rational(5, 4), R: sp.Rational(40), S: sp.Rational(2, 7), A: sp.Rational(4, 5), Om: sp.Rational(7, 2)},
]


def check(name: str, expr: sp.Expr, subs: dict | None = None) -> None:
    e = sp.simplify(sp.together(sp.expand(expr)))
    if e != 0:
        pts = _POINTS if subs is None else [dict(p, **subs) for p in _POINTS]
        vals = []
        for p in pts:
            v = complex(sp.N(e.subs(p), 30))
            vals.append(abs(v))
        if max(vals) < 1e-24:
            print(f"[OK  ] {name}  (numeric zero, {max(vals):.1e})")
            return
        print(f"[FAIL] {name}  -> max |residual| = {max(vals):.3e}")
        FAILURES.append(name)
        return
    print(f"[OK  ] {name}")


# ---------------------------------------------------------------- A. Legendre identities
print("=== A. Legendre closed forms vs Yagi-Yunes exterior prints ===")
h2A_log = -sp.Rational(3) * R**2 / (M * (R - 2 * M)) * (
    1 - 3 * M / R + sp.Rational(4, 3) * M**2 / R**2
    + sp.Rational(2, 3) * M**3 / R**3 + R / (2 * M) * f**2 * sp.log(f)
)
check("A1  Q22 == YY h2^ext A-term (log form)", Q22 - h2A_log)

K2A_log = sp.Rational(3) * R / M * (
    1 + M / R - sp.Rational(2, 3) * M**2 / R**2
    + R / (2 * M) * (1 - 2 * M**2 / R**2) * sp.log(f)
)
check("A2  K2 A-term with Q21_cs == YY log form",
      A * (2 * M / sp.sqrt(R * (R - 2 * M)) * Q21_cs - Q22) - A * K2A_log)
nocs_resid = sp.simplify(
    A * (2 * M / sp.sqrt(R * (R - 2 * M)) * Q21_nocs - Q22) - A * K2A_log
)
if nocs_resid == 0:
    print("[FAIL] A2d Q21_nocs unexpectedly matches (CS diagnostic broken)")
    FAILURES.append("A2d")
else:
    print("[OK  ] A2d Q21_nocs does NOT match -> Condon-Shortley phase required")

m2A_log = sp.Rational(3) * R**2 / M * (
    1 - 3 * M / R + sp.Rational(4, 3) * M**2 / R**2
    + sp.Rational(2, 3) * M**3 / R**3 + R / (2 * M) * f**2 * sp.log(f)
)
check("A3  -R*f*Q22 == YY m2^ext A-term (log form)", -R * f * Q22 - m2A_log)

lim_Q22 = sp.limit(Q22.subs(M, 1) * R**3, R, sp.oo)
check("A4  lim [Q22 * R^3/M^3] == 8/5  (fixes Q = -S^2/M - 8/5 A M^3)",
      sp.Rational(8, 5) - lim_Q22)


# ---------------------------------------------------------------- B. vacuum l=2 system
print("\n=== B. YY l=2 interior system vs exterior (vacuum, rho=p=0) ===")
h2e = S**2 * (1 + M / R) / (M * R**3) + A * Q22
K2e = -S**2 * (1 + 2 * M / R) / (M * R**3) + A * (2 * M / sp.sqrt(R * (R - 2 * M)) * Q21_cs - Q22)
m2e = -S**2 * (1 - 7 * M / R + 10 * M**2 / R**2) / (M * R**2) + A * m2A_log
wtilde = Om - 2 * S / R**3                       # vacuum omega_tilde
dwt = 6 * S / R**4


def m2_of(h2, dwt2):
    return -R * f * h2 + sp.Rational(1, 6) * R**4 * (R * f * dwt2)


check("B1  (m2) S-family", m2e - m2_of(h2e, dwt**2))
check("B1h (m2) A-family (wt=0)", (m2e - m2_of(h2e, 0)).subs(S, 0))


def k2R_residual(h2, K2, m2):
    return sp.diff(K2, R) - (
        -sp.diff(h2, R)
        + (R - 3 * M) / R**2 * (1 / f) * h2
        + (R - M) / R**3 * (1 / f**2) * m2
    )


check("B2  (k2R) S-family", k2R_residual(h2e, K2e, m2e))
check("B2h (k2R) A-family", k2R_residual(h2e, K2e, m2e).subs(S, 0))


def h2R_residual(h2, K2, m2, wt, dwt):
    beta = (R - M) / R * (1 / f)
    Y = (
        3 / R * (1 / f) * h2
        + 2 / R * (1 / f) * K2
        + 1 / R**2 * (1 / f**2) * m2
        + R**3 / 12 * (1 / f) * dwt**2
    )
    return sp.diff(h2, R) - (-beta * sp.diff(K2, R) + Y)


check("B3  (h2R) S-family", h2R_residual(h2e, K2e, m2e, wtilde, dwt))
check("B3h (h2R) A-family", h2R_residual(h2e, K2e, m2e, 0, 0).subs(S, 0))

check("B4  UMS m2-relation (vacuum, S-family)",
      m2e / (R - 2 * M) - (-h2e + sp.Rational(1, 6) * R**4 * dwt**2))

v2e = h2e + K2e
v2e_expected = -S**2 / R**4 + A * 2 * M / sp.sqrt(R * (R - 2 * M)) * Q21_cs
check("B5  v2^ext == -S^2/R^4 + 2AM Q21_cs/sqrt(R(R-2M))", v2e - v2e_expected)

nu0p = M / (R * (R - 2 * M))                     # UMS half-convention nu0'
check("B6  UMS v2-eq (their convention) vs corrected exterior",
      sp.diff(v2e, R) - (-2 * nu0p * h2e + (1 / R + nu0p) * sp.Rational(1, 6) * R**4 * dwt**2))
check("B6h UMS v2-eq homogeneous vs A-family",
      (sp.diff(v2e, R) + 2 * nu0p * h2e).subs(S, 0))


# ---------------------------------------------------------------- C. l=0 system (UMS print)
print("\n=== C. UMS l=0 (m0, p0): vacuum + centre series ===")
check("C1  (m0) vacuum: 3 S^2/R^4 source",
      3 * S**2 / R**4 - sp.Rational(1, 12) * R**4 * dwt**2)

jc2 = sp.symbols("jc2", positive=True)              # e^{-nu_c} = j_c^2
b_coef, a_coef = sp.symbols("b_coef a_coef")
p0 = b_coef * R**2
m0 = a_coef * R**5
M_c = sp.Rational(4, 3) * sp.pi * rho_c * R**3
j2p = -jc2 * 8 * sp.pi * (rho_c + p_c) * R          # (j^2)' to leading order
wt = om_c * (1 + sp.Rational(8, 5) * sp.pi * (rho_c + p_c) * R**2)
dwt_c = sp.Rational(16, 5) * sp.pi * (rho_c + p_c) * om_c * R

_CSUBS = {rho_c: sp.Rational(3), p_c: sp.Rational(1, 2),
          om_c: sp.Rational(1, 3), jc2: sp.Rational(4, 5)}

dm0_rhs = (
    4 * sp.pi * R**2 * (rho_c + p_c) * cEpdP * p0
    + sp.Rational(1, 12) * R**4 * jc2 * dwt_c**2
    - sp.Rational(1, 3) * R**3 * wt**2 * j2p
)
a_eq = sp.Eq(sp.expand(sp.diff(m0, R) / R**4), sp.expand(sp.series(dm0_rhs, R, 0, 6).removeO() / R**4))
a_solved = sp.solve(a_eq, a_coef)[0]

dp0_rhs = (
    -m0 * (1 + 8 * sp.pi * R**2 * p_c) / (R - 2 * M_c) ** 2
    - 4 * sp.pi * (rho_c + p_c) * R**2 / (R - 2 * M_c) * p0
    + sp.Rational(1, 12) * R**4 * jc2 / (R - 2 * M_c) * dwt_c**2
    + sp.Rational(1, 3) * jc2 * sp.diff(R**3 * wt**2 / (R - 2 * M_c), R)
)
b_eq = sp.Eq(sp.expand(sp.diff(p0, R) / R), sp.expand(sp.series(dp0_rhs, R, 0, 2).removeO() / R))
b_solved = sp.solve(b_eq, b_coef)[0]
check("C3  p0 centre coefficient == (1/3) j_c^2 om_c^2",
      sp.simplify(b_solved - jc2 * om_c**2 / 3), subs=_CSUBS)

a_full = a_solved.subs(b_coef, b_solved)
a_expected = sp.Rational(4, 15) * sp.pi * (rho_c + p_c) * (cEpdP + 2) * jc2 * om_c**2
check("C2  m0 centre coefficient == (4 pi/15)(eps+p)(dEps/dP+2) j_c^2 om_c^2",
      sp.simplify(sp.expand(a_full - a_expected)), subs=dict(_CSUBS, cEpdP=sp.Rational(2)))

nup_full = 2 * (M_c + 4 * sp.pi * p_c * R**3) / (R * (R - 2 * M_c))
lamp_full = 2 * (4 * sp.pi * rho_c * R**3 - M_c) / (R * (R - 2 * M_c))
check("C4  nu'+lambda' == 8 pi R^2 (eps+p)/(R-2M)",
      sp.simplify(nup_full + lamp_full - 8 * sp.pi * R**2 * (rho_c + p_c) / (R - 2 * M_c)))

# interior equivalence, pointwise: UMS m2-relation wt^2-term vs YY (m2) 16 pi term.
# UMS: m2 = (R-2m)[-h2 + (1/6)R^4 j^2 wt'^2 - (1/3)R^3 wt^2 (j^2)'] with
# j^2 = e^{-(nu+lambda)} and (j^2)' = -j^2 (nu'+lambda')  [full convention]
j2c = sp.symbols("j2c", positive=True)
ums_om2_term = -sp.Rational(1, 3) * (R - 2 * M_c) * R**3 * wt**2 * (-j2c * (nup_full + lamp_full))
yy_om2_term = sp.Rational(8, 3) * sp.pi * R**5 * j2c * (rho_c + p_c) * wt**2
check("C5  UMS m2-relation wt^2-term == YY (m2) 16 pi term (interior)",
      sp.simplify(sp.expand(ums_om2_term - yy_om2_term)),
      subs=dict(_CSUBS, j2c=sp.Rational(4, 5)))


# ---------------------------------------------------------------- D. YY l=2 centre data
print("\n=== D. YY l=2 system: centre initial-data consistency ===")
B = sp.symbols("B", positive=True)
epsc = sp.symbols("epsc", positive=True)            # e^{-nu_c}
p2c = -sp.Rational(2, 3) * sp.pi * (rho_c + p_c) * (3 * p_c + rho_c)
p_ser = p_c + p2c * R**2
rho_ser = rho_c + cEpdP * p2c * R**2
m_ser = sp.Rational(4, 3) * sp.pi * rho_c * R**3
elam = 1 / (1 - 2 * m_ser / R)                     # e^{lambda} = (1-2m/R)^{-1}
enu = sp.exp(sp.Rational(4, 3) * sp.pi * (rho_c + p_c) * R**2) / epsc   # e^{nu}
wt_ser = om_c * (1 + sp.Rational(8, 5) * sp.pi * (rho_c + p_c) * R**2)
dwt_ser = sp.diff(wt_ser, R)
h2s, K2s, m2s = B * R**2, -B * R**2, -B * R**3
_DSUBS = dict(_CSUBS, epsc=sp.Rational(4, 5), cEpdP=sp.Rational(2), B=sp.Rational(3, 2))

res_k2 = sp.diff(K2s, R) - (
    -sp.diff(h2s, R)
    + (R - 3 * m_ser - 4 * sp.pi * p_ser * R**3) / R**2 * elam * h2s
    + (R - m_ser + 4 * sp.pi * p_ser * R**3) / R**3 * elam**2 * m2s
)
check("D1  k2R residual for (BR^2, -BR^2, -BR^3) has no terms below O(R^3)",
      sp.series(res_k2, R, 0, 3).removeO(), subs=_DSUBS)

res_h2 = sp.diff(h2s, R) - (
    -(R - m_ser + 4 * sp.pi * p_ser * R**3) / R * elam * sp.diff(K2s, R)
    + (3 - 4 * sp.pi * (rho_ser + p_ser) * R**2) / R * elam * h2s
    + 2 / R * elam * K2s
    + (1 + 8 * sp.pi * p_ser * R**2) / R**2 * elam**2 * m2s
    + R**3 / 12 / enu * dwt_ser**2
    - sp.Rational(4, 3) * sp.pi * (rho_ser + p_ser) * R**3 * wt_ser**2 / enu * elam
)
check("D2  h2R residual for (BR^2, -BR^2, -BR^3) has no terms below O(R^3)",
      sp.series(res_h2, R, 0, 3).removeO(), subs=_DSUBS)

m2_alg = -R * (1 - 2 * m_ser / R) * h2s + sp.Rational(1, 6) * R**4 / enu * (1 - 2 * m_ser / R) * (
    R * (1 - 2 * m_ser / R) * dwt_ser**2 + 16 * sp.pi * R * (rho_ser + p_ser) * wt_ser**2
)
check("D3  m2 algebraic: m2 + B R^3 has no terms below O(R^5)",
      sp.series(m2_alg + B * R**3, R, 0, 5).removeO(), subs=_DSUBS)


# ---------------------------------------------------------------- E. Maclaurin limit
print("\n=== E. Maclaurin (Newtonian uniform-density) validation target ===")
e_sym, rho_N, R_N, M_N, Om_N = sp.symbols("e rho_N R_N M_N Om_N", positive=True)
maclaurin = sp.sqrt(1 - e_sym**2) / e_sym**3 * (3 - 2 * e_sym**2) * sp.asin(e_sym) - 3 * (1 - e_sym**2) / e_sym**2
Om2_over = sp.series(2 * maclaurin, e_sym, 0, 4).removeO()
print(f"  Omega^2/(pi G rho) = {sp.simplify(Om2_over)}  (expect 8/15 e^2)")
check("E0  Maclaurin slow-rotation: Omega^2 == (8/15) pi G rho e^2",
      sp.simplify(Om2_over - sp.Rational(8, 15) * e_sym**2))

e2_val = sp.Rational(15, 8) * Om_N**2 / (sp.pi * rho_N)
Q_N = -M_N / 5 * R_N**2 * e2_val
J_N = sp.Rational(2, 5) * M_N * R_N**2 * Om_N
qtilde = sp.simplify(-Q_N * M_N / J_N**2).subs(rho_N, 3 * M_N / (4 * sp.pi * R_N**3))
check("E1  Maclaurin q_tilde == 25/8 R/M",
      sp.simplify(qtilde - sp.Rational(25, 8) * R_N / M_N))

Q_N_sub = Q_N.subs(rho_N, 3 * M_N / (4 * sp.pi * R_N**3))
check("E2  Q_Maclaurin == -(1/2) Omega^2 R^5  (k2_rot^N = 3/4)",
      sp.simplify(Q_N_sub + Om_N**2 * R_N**5 / 2))

print("\n" + "=" * 60)
if FAILURES:
    print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
    raise SystemExit(1)
print("ALL SYMBOLIC CHECKS PASSED")
