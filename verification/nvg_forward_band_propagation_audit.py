#!/usr/bin/env python3
"""Forward-band propagation audit: declared nuclear windows -> NVG forward chain -> held-out stars.

No parameter is chosen using a stellar observable.  The chain is

    scalar jet (y*, n0, E/A, K [, Q0 -> a5 ; minimal a6 as a separate structural class])
    -> global stability of U = z^2 q(z), z = y^2 - 1, q(z) >= 0 on z >= -1
    -> isovector contact C0 = 8 (J - T)/n0 ; running gamma(L) (power law) or alpha = -gamma (Pade)
    -> beta equilibrium n p e mu (mu_n - mu_p = mu_e, n_e + n_mu = n_p), scalar re-stationarised
    -> table eps(P) with the thermodynamically consistent P_rho = eps_rho (1 + n C'(n)/C)
       (the author's P_rho = eps_rho is kept as the labelled 'authorP' variant)
    -> crust: author's polytrope (p_match = 1.5 MeV/fm^3, Gamma = 1.35) or the physical SLy crust
       (Read, Lackey, Owen, Friedman 2009, Table II) matched by eps-continuity (or P-continuity with
       the Hinderer jump condition in the tidal equation)
    -> TOV + Hinderer (DOP853) -> M_max, R(M), Lambda(M), Lambda_tilde
    -> descriptive held-out comparison with declared star data (pass/fail at 1 and 2 sigma).

Float (numpy/scipy) transcription of the mpmath originals in
verification/source_complete_scaling_saturation_audit.py (BulkModel, inverse_potential_jet,
polynomial_vacuum_bound), verification/nvg_isospin_jet_audit.py (_species_fermi, local_jet),
verification/nvg_forward_beta_eos_audit.py (beta table) and
verification/nvg_forward_tov_heldout_audit.py (ForwardEOS, solve_tov_tidal).  The script reads no
saved result file as numerical input: every window, every structural principle and every held-out
star datum is a constant below with its source.

Modes: default = fast declared grid (24 samples); --full = declared design (Latin hypercube 400,
seed 20261030, plus the coarse y* x L x Q0 grid); --crust {author,sly} selects the crust used for
the band; the corner_and_author_points block always evaluates both crusts and both pressure
conventions.  Strict JSON to stdout; --write stores verification/nvg_forward_band_propagation_results.json.
Evidence weight 0: a forward consequence of declared windows, never a fit, and the pass/fail flags
are descriptive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.interpolate import CubicSpline, PchipInterpolator  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE_PATH = Path(__file__).resolve()
RESULT_PATH = HERE / "nvg_forward_band_propagation_results.json"
SCHEMA = "nvg_forward_band_propagation.v1"
STATUS = "FORWARD_BAND_PROPAGATION_COMPUTED_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

# ----------------------------------------------------------------------------- physical constants
HBARC = 197.3269804          # MeV fm
MN = 939.0                   # MeV, single nucleon mass (author's convention)
M_E = 0.511
M_MU = 105.658
PI2 = math.pi ** 2

_C = 299792458.0
_G = 6.67430e-11
_MEV_J = 1.602176634e-13
CONST = {
    # values hard-coded in verification/nvg_forward_tov_heldout_audit.py:39-40
    "author": {"K_CONV": 1.3234e-6, "M_SUN_KM": 1.4766},
    # G/c^4 * (1 MeV/fm^3) in km^-2 and G M_sun/c^2 in km (CODATA-2018 / IAU-2015 GM_sun)
    "codata": {"K_CONV": _G / _C ** 4 * _MEV_J * 1e45 * 1e6,
               "M_SUN_KM": 1.32712440018e20 / _C ** 2 / 1e3},
}

# verification/nvg_prex_crex_external_consistency_probe.py:128-129 (T_CONTACT 12.644118 + j_bar 11.136016)
J_DESIGN = 23.78013451414814

# ----------------------------------------------------------------------------- declared windows (fixed before any star number)
DECLARED_WINDOWS = {
    "n0": {"range": [0.15, 0.17], "unit": "fm^-3",
           "source": "Margueron, Hoffmann Casali, Gulminelli, PRC 97, 025805 (2018): n_sat = 0.155 +- 0.005; author's EMPIRICAL n0_fm3='0.16' (verification/nvg_saturation_recalibration.py:59-70)"},
    "EA": {"range": [-16.5, -15.5], "unit": "MeV",
           "source": "Margueron et al. 2018: E_sat = -15.8 +- 0.3; Dutra et al. PRC 85, 035201 (2012); author's binding_MeV='-16'"},
    "K": {"range": [220.0, 260.0], "unit": "MeV",
          "source": "Shlomo, Kolomietz, Colo, EPJA 30, 23 (2006): K = 240 +- 20; Garg & Colo, PPNP 101, 55 (2018); author's K_window_MeV=[220,260]"},
    "ystar": {"range": [0.60, 0.93], "unit": "m*/M",
              "source": "RMF Dirac effective mass 0.55-0.75 (Dutra et al. PRC 90, 055203 (2014); Glendenning 1997) = author's EMPIRICAL mstar_over_M_window=[0.6,0.7]; 0.75-0.93 is the author's own design grid Y_GRID (nvg_saturation_recalibration.py:75, nvg_forward_beta_eos_audit.py:44); declared window = union, subject to global stability of U"},
    "Q0": {"range": [-600.0, 200.0], "unit": "MeV",
           "source": "Margueron et al. 2018: Q_sat = 300 +- 400; Dutra et al. 2012 Skyrme systematics -700..-200; Cai & Chen PRC 95, 024302 (2017): -306 +- 400. Role: fixes a5 ONLY when the quartic q(z) is not globally stable; otherwise a5 = 0 and Q0 is a forward output"},
    "J": {"range": [30.0, 34.0], "unit": "MeV",
          "source": "Lattimer & Lim, ApJ 771, 51 (2013): 29.0-32.7; Li & Han, PLB 727, 276 (2013): 31.6 +- 2.7; Oertel et al. RMP 89, 015007 (2017): 31.7 +- 3.2; author's J_window_MeV=[30,34]"},
    "L": {"range": [35.0, 65.0], "unit": "MeV",
          "source": "Lattimer & Lim 2013: 40.5-61.9; Li & Han 2013: 58.9 +- 16; Oertel et al. 2017: 58.7 +- 28.1; author's L_window_MeV=[35,65]"},
}
WINDOW_ORDER = ("n0", "EA", "K", "ystar", "Q0", "J", "L")
CENTRAL = {"n0": 0.16, "EA": -16.0, "K": 240.0, "ystar": 0.87, "Q0": -200.0, "J": 32.0, "L": 50.0}
LHS_SEED = 20261030
LHS_N = 400
GRID_Y = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.93)
GRID_Q0 = (-600.0, -400.0, -200.0, 0.0, 200.0)
GRID_L = (35.0, 50.0, 65.0)
# fast default grid (<= 24 samples): central (n0, E/A, K, J), Q0 at the window centre, y* x L
FAST_GRID_Y = (0.60, 0.70, 0.80, 0.84, 0.86, 0.87, 0.90, 0.93)
FAST_GRID_L = (35.0, 50.0, 65.0)
FAST_GRID_Q0 = -200.0

STRUCTURAL_PRINCIPLES = [
    "U(y) = z^2 q(z), z = y^2-1 (vacuum y=1 with U(1)=U'(1)=0); global stability = q(z) >= 0 for all z >= -1 (polynomial_vacuum_bound, source_complete_scaling_saturation_audit.py)",
    "vector energy C_v n^2/(2 y^2), C_v = (mu0 - E_F) y*^2/n0 fixed by (y*, n0, E/A) (inverse_potential_jet); no vector self-interaction",
    "linear scalar mass m = M y, M = 939 MeV; rho term C_rho n^2 delta^2/8, C0 = 8 (J - T)/n0 at n0",
    "running of the isovector contact: power law C(n) = C0 (n/n0)^gamma, gamma = (L - L_kin)/(3 (J - T)) - 1 [class 1], or Pade C(n) = C0/(1 + alpha (n/n0 - 1)), alpha = -gamma [class 2]",
    "beta equilibrium n p e mu with charge neutrality; thermodynamically consistent pressure P_rho = eps_rho (1 + n C'(n)/C); the author's P_rho = eps_rho is the labelled 'authorP' variant",
    "scalar completion rule (declared): quartic if globally stable (a5 = 0, Q0 forward output); else quintic with a5 fixed by the declared Q0; else 'no_stable_model'. Separate structural class (counted, not from a window): minimal sextic completion a6_min",
    "stellar validity: causality max c_s^2 < 1 on the stellar branch; dP/dn > 0 above the crust match; M_max an interior maximum of M(P_c) within the table (x <= 15); 'strict' additionally requires the tracked scalar branch to be the global minimum of eps_N(n, delta, y) over y on the stellar branch",
]

# ----------------------------------------------------------------------------- held-out star data (declared, descriptive only)
# Accessed ONLY by heldout_compare (after the TOV solve).  Never read by the coefficient solve.
HELDOUT_STAR_DATA = {
    "Mmax_J0740_Fonseca2021": {"type": "Mmax_ge", "value": 2.01, "source": "Fonseca et al. 2021 ApJL 915 L12: M=2.08+-0.07 -> pass if M_max>=2.01"},
    "R207_J0740_Salmi2024": {"type": "radius", "M": 2.07, "c": 12.49, "sp": 1.28, "sm": 0.88, "source": "Salmi et al. 2024 ApJ 974 294 (68%)", "if_reached": True},
    "R134_J0030_Riley2019": {"type": "radius", "M": 1.34, "c": 12.71, "sp": 1.14, "sm": 1.19, "source": "Riley et al. 2019 ApJL 887 L21 (68%)"},
    "R14_band_11p2_13p2": {"type": "band", "M": 1.4, "lo": 11.2, "hi": 13.2, "source": "declared NICER band for R_1.4"},
    "R1418_J0437_Choudhury2024": {"type": "radius", "M": 1.418, "c": 11.36, "sp": 0.95, "sm": 0.63, "source": "Choudhury et al. 2024 ApJL 971 L20 (68%)"},
    "R144_J0614_Mauviard2025": {"type": "radius", "M": 1.44, "c": 10.29, "sp": 1.01, "sm": 0.86, "source": "Mauviard et al. 2025 (NICER, 68%)"},
    "Lt_GW170817_le720": {"type": "Lt_le", "value": 720.0, "source": "Abbott et al. 2018 PRL 121 161101 (low-spin 90%)"},
    "L14_GW170817_190_p390_m120": {"type": "band_L", "lo": 70.0, "hi": 580.0, "source": "Abbott et al. 2018 PRL 121 161101: Lambda_1.4 = 190 +390/-120 (90%)"},
}
THRESHOLD_NOTE = {
    "Mmax_J0740_Fonseca2021": "M_max >= 2.01 Msun = 2.08 - 0.07 (1-sigma lower edge); the SAME threshold is used for pass1 and pass2",
    "R207_J0740_Salmi2024": "R(2.07) in [12.49-2*0.88, 12.49+2*1.28] (pass2) / [12.49-0.88, 12.49+1.28] (pass1); None if M_max < 2.07",
    "R134_J0030_Riley2019": "R(1.34) in [12.71-2*1.19, 12.71+2*1.14] (pass2) / [12.71-1.19, 12.71+1.14] (pass1)",
    "R14_band_11p2_13p2": "R(1.4) in the fixed declared band [11.2, 13.2] km; same for pass1 and pass2",
    "R1418_J0437_Choudhury2024": "R(1.418) in [11.36-2*0.63, 11.36+2*0.95] (pass2) / [11.36-0.63, 11.36+0.95] (pass1)",
    "R144_J0614_Mauviard2025": "R(1.44) in [10.29-2*0.86, 10.29+2*1.01] = [8.57, 12.31] (pass2) / [9.43, 11.30] (pass1)",
    "Lt_GW170817_le720": "Lambda_tilde(1.36+1.36, q=1 from Mchirp 1.186) <= 720 (90% CL); same for pass1 and pass2",
    "L14_GW170817_190_p390_m120": "Lambda_1.4 in [70, 580] (190 +390/-120, 90% CL); same for pass1 and pass2",
}

# ----------------------------------------------------------------------------- SLy crust (Read+2009 Table II)
C_CGS = 2.99792458e10                 # cm/s (exact)
MEV_ERG = 1.602176634e-6              # erg per MeV (exact, CODATA-2018)
FM3_CM3 = 1e-39                       # cm^3 per fm^3
MEVFM3_TO_GCC = MEV_ERG / C_CGS ** 2 / FM3_CM3   # g/cm^3 per MeV/fm^3 (energy density / c^2)
M_B_G = 1.66e-24                      # g, baryon rest mass of the Read+2009 rest-mass-density convention
N_FM3_TO_RHO_GCC = 1e39 * M_B_G       # g/cm^3 per fm^-3
# (K_i [cgs, c=1], Gamma_i, rho_i upper dividing density [g/cm^3])
SLY_TABLE_II = (
    (6.80110e-9, 1.58425, 2.44034e7),
    (1.06186e-6, 1.28733, 3.78358e11),
    (5.32697e+1, 0.62223, 2.62780e12),
    (3.99874e-8, 1.35692, None),
)
CRUST_SOURCE = ("Read, Lackey, Owen, Friedman, Phys. Rev. D 79, 124032 (2009), Table II (piecewise-polytrope fit to "
                "the SLy EOS of Douchin & Haensel 2001, A&A 380, 151), eqs. (5)-(7) for eps(rho) with a_0 = 0")
JUMP_SOURCE = ("Damour & Nagar 2009 PRD 80 084035; Hinderer, Lackey, Lang, Read 2010 PRD 81 123016: "
               "y(r_d+) = y(r_d-) - 4 pi r_d^3 Deps/(m + 4 pi r_d^3 P)")
AUTHOR_CRUST = {"p_match_MeV_fm3": 1.5, "Gamma": 1.35, "p_surf_MeV_fm3": 1e-4,
                "source": "verification/nvg_forward_tov_heldout_audit.py:37-38 (P_MATCH, GAMMA_CRUST); eps = eps_match (P/p_match)^(1/Gamma) below p_match; surface event P = 1e-4"}
P_SURF_SLY = 1e-9


# ----------------------------------------------------------------------------- Fermi integrals
def fermi_s(n, m, g):
    """Zero-T Fermi integrals, degeneracy g (g=4 symmetric matter, g=2 one species).
    Returns k, E_F, eps, P, n_s, d n_s/d m (natural MeV units)."""
    if n <= 0.0:
        return 0.0, m, 0.0, 0.0, 0.0, 0.0
    k = (6.0 * PI2 * n / g) ** (1.0 / 3.0)
    ef = math.sqrt(k * k + m * m)
    a = math.asinh(k / m)
    m2 = m * m
    m4 = m2 * m2
    kef = k * ef
    eps = g / (16.0 * PI2) * (kef * (2.0 * k * k + m2) - m4 * a)
    p = g / (48.0 * PI2) * (kef * (2.0 * k * k - 3.0 * m2) + 3.0 * m4 * a)
    ns = g * m / (4.0 * PI2) * (kef - m2 * a)
    ns_m = 3.0 * ns / m - 3.0 * n / ef
    return k, ef, eps, p, ns, ns_m


def fermi_v(n, m, g):
    """Vectorised (numpy) version of fermi_s for arrays of m."""
    n = np.asarray(n, dtype=float)
    m = np.asarray(m, dtype=float)
    k = (6.0 * PI2 * n / g) ** (1.0 / 3.0)
    ef = np.sqrt(k * k + m * m)
    a = np.arcsinh(k / m)
    m2 = m * m
    m4 = m2 * m2
    kef = k * ef
    eps = g / (16.0 * PI2) * (kef * (2.0 * k * k + m2) - m4 * a)
    p = g / (48.0 * PI2) * (kef * (2.0 * k * k - 3.0 * m2) + 3.0 * m4 * a)
    ns = g * m / (4.0 * PI2) * (kef - m2 * a)
    return k, ef, eps, p, ns


# ----------------------------------------------------------------------------- scalar potential
def potential(coef, y):
    """U = sum_{j>=2} a_j z^j, z = y^2-1 ; returns U, U_y, U_yy (same algebra as BulkModel.potential)."""
    z = y * y - 1.0
    u = uy = uyy = 0.0
    for idx, a in enumerate(coef):
        j = idx + 2
        zj2 = z ** (j - 2)
        zj1 = zj2 * z
        u += a * zj1 * z
        uy += 2.0 * j * a * y * zj1
        uyy += 2.0 * j * a * zj1 + 4.0 * j * (j - 1) * a * y * y * zj2
    return u, uy, uyy


def potential_v(coef, y):
    z = y * y - 1.0
    u = np.zeros_like(y)
    for idx, a in enumerate(coef):
        u += a * z ** (idx + 2)
    return u


def _jet_rows(j, y, z):
    """Rows of the linear system for a_j: [U, U_y, U_yy, U_yyy] at (y, z=y^2-1)."""
    r0 = z ** j
    r1 = 2.0 * j * y * z ** (j - 1)
    r2 = 2.0 * j * z ** (j - 1) + 4.0 * j * (j - 1) * y * y * z ** (j - 2)
    r3 = 12.0 * j * (j - 1) * y * z ** (j - 2)
    if j >= 3:
        r3 += 8.0 * j * (j - 1) * (j - 2) * y ** 3 * z ** (j - 3)
    return r0, r1, r2, r3


def q_min(coef):
    """Global minimum of q(z) = sum a_j z^{j-2} on z >= -1 (U = z^2 q(z)).
    Returns (q_min, z_at_min, bounded); bounded=False when the leading coefficient is <= 0."""
    a = list(coef)
    while len(a) > 1 and a[-1] == 0.0:
        a.pop()
    deg = len(a) - 1
    if deg == 0:
        return (a[0], -1.0, a[0] >= 0)
    if a[-1] < 0.0:
        return (None, None, False)
    if deg == 1:
        return (a[0] - a[1], -1.0, True)
    dq = [(i) * a[i] for i in range(deg, 0, -1)]
    roots = np.roots(dq)
    cands = [-1.0]
    for r in roots:
        if abs(r.imag) < 1e-9 * max(1.0, abs(r.real)) and r.real >= -1.0:
            cands.append(float(r.real))

    def q(z):
        return sum(a[i] * z ** i for i in range(deg + 1))
    vals = [(q(z), z) for z in cands]
    qm, zm = min(vals)
    return (qm, zm, True)


# ----------------------------------------------------------------------------- scalar jet (coefficient solve; no star data)
def scalar_jet(ystar, n0_fm3, EA, K, Q0=None, a6=0.0):
    """Inverse local design at saturation: (y*, n0, E/A, K) fix (a2,a3,a4) [quartic]; with Q0 given,
    (a2..a5) are fixed by the 4x4 system including U_yyy(Q0); a6 is a declared extra coefficient whose
    contribution is subtracted from the right-hand side."""
    y = float(ystar)
    n = n0_fm3 * HBARC ** 3
    mu0 = MN + EA
    m = MN * y
    k, ef, eps, P, ns, ns_m = fermi_s(n, m, 4)
    V = mu0 - ef
    if V <= 0.0 or K <= 0.0 or y == 1.0:
        return {"status": "jet_invalid_V_or_K", "V": V}
    Cv = V * y * y / n
    D = k * k / (3.0 * n * ef) + Cv / y ** 2
    B = MN ** 2 * y / ef - 2.0 * Cv * n / y ** 3
    den = D - K / (9.0 * n)
    if den <= 0.0 or abs(B) < 1e-9 * (MN ** 2 * y / ef):
        return {"status": "jet_invalid_curvature", "den": den, "B": B}
    C = B * B / den
    U = P + n * V / 2.0
    Uy = (n * V - m * ns) / y
    Uyy = C - MN ** 2 * ns_m - 3.0 * Cv * n * n / y ** 4
    z = y * y - 1.0
    out = {"status": "ok", "ystar": y, "n0_nat": n, "n0_fm3": n0_fm3, "EA": EA, "K": K, "Cv": Cv,
           "k": k, "ef": ef, "D": D, "B": B, "C": C, "U": U, "Uy": Uy, "Uyy": Uyy, "m": m, "ns": ns, "ns_m": ns_m}
    e_nnn = -k * k / (9.0 * n * n * ef) * (1.0 + k * k / ef ** 2)
    e_nny = -k * k * MN ** 2 * y / (3.0 * n * ef ** 3) - 2.0 * Cv / y ** 3
    e_nyy = MN ** 2 * k * k / ef ** 3 + 6.0 * Cv * n / y ** 4
    ns_mm = 6.0 * ns / m ** 2 - 9.0 * n / (m * ef) + 3.0 * n * m / ef ** 3
    yp = -B / C
    if Q0 is None:
        rows = [_jet_rows(j, y, z) for j in (2, 3, 4)]
        A = np.array([[r[i] for r in rows] for i in range(3)])
        coef = np.linalg.solve(A, np.array([U, Uy, Uyy]))
        coef = [float(c) for c in coef]
        Uyyy = sum(c * _jet_rows(j, y, z)[3] for c, j in zip(coef, (2, 3, 4)))
        e_yyy = Uyyy + MN ** 3 * ns_mm - 12.0 * Cv * n * n / y ** 5
        ypp = -(e_nny + 2.0 * e_nyy * yp + e_yyy * yp ** 2) / C
        mu2 = e_nnn + 2.0 * e_nny * yp + e_nyy * yp ** 2 + B * ypp
        out["Q0_forward"] = 27.0 * n * n * mu2 - 9.0 * K
        out["degree"] = 4
    else:
        mu2 = (Q0 + 9.0 * K) / (27.0 * n * n)
        e_yyy = ((e_nnn + 2.0 * e_nny * yp + e_nyy * yp ** 2 - mu2) * C / B - e_nny - 2.0 * e_nyy * yp) / yp ** 2
        Uyyy = e_yyy - MN ** 3 * ns_mm + 12.0 * Cv * n * n / y ** 5
        rhs = np.array([U, Uy, Uyy, Uyyy])
        if a6 != 0.0:
            r6 = _jet_rows(6, y, z)
            rhs = rhs - a6 * np.array(r6)
        rows = [_jet_rows(j, y, z) for j in (2, 3, 4, 5)]
        A = np.array([[r[i] for r in rows] for i in range(4)])
        coef = [float(c) for c in np.linalg.solve(A, rhs)]
        if a6 != 0.0:
            coef.append(float(a6))
        out["Q0_forward"] = float(Q0)
        out["degree"] = 6 if a6 != 0.0 else 5
        out["Uyyy"] = Uyyy
    out["coef"] = coef
    qm, zm, bounded = q_min(coef)
    out["q_min"] = qm
    out["z_at_q_min"] = zm
    out["globally_stable"] = bool(bounded and qm is not None and qm >= 0.0)
    T = k * k / (6.0 * ef)
    u = k * k / (ef * ef)
    T_y = -T * (1.0 - u) / y
    out.update(T=T, u=u, T_y=T_y, dy_dn=yp, L_fixed_y=(2.0 - u) * T,
               L_kin=(2.0 - u) * T + 3.0 * n * T_y * yp)
    return out


def minimal_a6(ystar, n0_fm3, EA, K, Q0, a6_grid=None):
    """Smallest a6 >= 0 (log-grid scan then bisection) such that the sextic completion with (a2..a5)
    fixed by (U,U_y,U_yy,U_yyy(Q0)) has q(z) >= 0 on z >= -1.  Declared STRUCTURAL minimality, not a window."""
    if a6_grid is None:
        a6_grid = np.logspace(5, 14, 181)
    prev = 0.0
    j0 = scalar_jet(ystar, n0_fm3, EA, K, Q0=Q0, a6=0.0)
    if j0["status"] != "ok":
        return None
    if j0["globally_stable"]:
        return {"a6": 0.0, "jet": j0}
    found = None
    for a6 in a6_grid:
        j = scalar_jet(ystar, n0_fm3, EA, K, Q0=Q0, a6=float(a6))
        if j["status"] == "ok" and j["globally_stable"]:
            found = float(a6)
            break
        prev = float(a6)
    if found is None:
        return None
    lo, hi = prev, found
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        j = scalar_jet(ystar, n0_fm3, EA, K, Q0=Q0, a6=mid)
        if j["status"] == "ok" and j["globally_stable"]:
            hi = mid
        else:
            lo = mid
    j = scalar_jet(ystar, n0_fm3, EA, K, Q0=Q0, a6=hi)
    return {"a6": hi, "jet": j}


def scalar_spec(ystar, n0, EA, K, Q0, allow_sextic=True):
    """Declared rule: quartic if globally stable (a5=0, Q0 forward output); else quintic with a5 from Q0;
    else 'no_stable_model'.  Separately (structural class): minimal sextic completion a6_min."""
    j4 = scalar_jet(ystar, n0, EA, K)
    specs = {}
    if j4["status"] != "ok":
        specs["declared"] = {"status": "jet_invalid", "detail": j4}
        return specs
    if j4["globally_stable"]:
        specs["declared"] = {"status": "quartic_stable", "jet": j4, "a5": 0.0, "a6": 0.0}
    else:
        j5 = scalar_jet(ystar, n0, EA, K, Q0=Q0) if Q0 is not None else {"status": "no_Q0"}
        if j5["status"] == "ok" and j5["globally_stable"]:
            specs["declared"] = {"status": "quintic_stable", "jet": j5, "a5": j5["coef"][3], "a6": 0.0}
        else:
            specs["declared"] = {"status": "no_stable_model", "jet_quartic": j4, "jet_quintic": j5}
            if allow_sextic and Q0 is not None:
                m6 = minimal_a6(ystar, n0, EA, K, Q0)
                if m6 is not None:
                    specs["sextic_min"] = {"status": "sextic_min_stable", "jet": m6["jet"], "a5": m6["jet"]["coef"][3], "a6": m6["a6"]}
                else:
                    specs["sextic_min"] = {"status": "no_stable_model"}
    return specs


def _jet_summary(j):
    keys = ("ystar", "n0_fm3", "EA", "K", "Cv", "coef", "q_min", "z_at_q_min", "globally_stable", "Q0_forward", "degree",
            "T", "u", "L_kin", "dy_dn", "C", "B", "D")
    return {k: j.get(k) for k in keys}


# ----------------------------------------------------------------------------- isovector
def isovector(jet, J, L, form="power"):
    """C0 = 8 (J - T)/n0 ; L = L_kin + 3 C0 n0/8 (1+gamma)  =>  gamma = (L - L_kin)/(3 (J - T)) - 1.
    Pade: C(n) = C0 / (1 + alpha (x-1)), alpha = -gamma (same first-order L)."""
    T = jet["T"]
    n = jet["n0_nat"]
    if J <= T:
        return {"status": "J_le_T_no_positive_C0", "T": T}
    C0 = 8.0 * (J - T) / n
    L_rho = 3.0 * C0 * n / 8.0
    gamma = (L - jet["L_kin"]) / L_rho - 1.0
    return {"status": "ok", "J": J, "L": L, "T": T, "C0": C0, "L_kin": jet["L_kin"], "L_const": jet["L_kin"] + L_rho,
            "gamma": gamma, "alpha": -gamma, "form": form}


def make_C_of_x(iso):
    """Return (C(x), dlnC/dlnx (x), pole) for x = n/n0.  Pade follows the author's convention
    C = 1e-12 when the denominator is <= 0."""
    C0 = iso["C0"]
    if iso["form"] == "power":
        g = iso["gamma"]

        def C(x):
            return C0 * x ** g

        def dln(x):
            return g
        pole = None
    else:
        al = iso["alpha"]

        def C(x):
            d = 1.0 + al * (x - 1.0)
            return C0 / d if d > 0.0 else 1e-12

        def dln(x):
            d = 1.0 + al * (x - 1.0)
            return -al * x / d if d > 0.0 else 0.0
        pole = (1.0 - 1.0 / al) if al != 0.0 else None
    return C, dln, pole


# ----------------------------------------------------------------------------- leptons
def lepton_mu(n_p):
    """Common lepton chemical potential from n_e + n_mu = n_p ; returns (mu, d mu / d n_p)."""
    if n_p <= 0.0:
        return M_E, 0.0
    ke = (3.0 * PI2 * n_p) ** (1.0 / 3.0)
    mu = math.sqrt(ke * ke + M_E * M_E)
    if mu <= M_MU:
        return mu, PI2 / (mu * ke)
    for _ in range(100):
        ke = math.sqrt(mu * mu - M_E * M_E)
        kmu = math.sqrt(mu * mu - M_MU * M_MU) if mu > M_MU else 0.0
        g = (ke ** 3 + kmu ** 3) / (3.0 * PI2) - n_p
        dg = (mu * ke + mu * kmu) / PI2
        step = g / dg
        mu_new = mu - step
        if mu_new <= M_MU:
            mu_new = 0.5 * (mu + M_MU)
        if abs(mu_new - mu) < 1e-14 * mu:
            mu = mu_new
            break
        mu = mu_new
    ke = math.sqrt(mu * mu - M_E * M_E)
    kmu = math.sqrt(mu * mu - M_MU * M_MU) if mu > M_MU else 0.0
    return mu, PI2 / (mu * (ke + kmu))


def lepton_eps_p(mu):
    eps = p = 0.0
    for ml in (M_E, M_MU):
        if mu > ml:
            k = math.sqrt(mu * mu - ml * ml)
            lg = math.log((k + mu) / ml)
            eps += (k * mu * (2.0 * k * k + ml * ml) - ml ** 4 * lg) / (8.0 * PI2)
            p += (k * mu * (2.0 * k * k / 3.0 - ml * ml) + ml ** 4 * lg) / (8.0 * PI2)
    return eps, p


# ----------------------------------------------------------------------------- beta equilibrium
class BetaSolver:
    def __init__(self, coef, Cv, n0_nat, C_of_x):
        self.coef = coef
        self.Cv = Cv
        self.n0 = n0_nat
        self.C_of_x = C_of_x

    def residuals(self, n, C, y, d):
        """F1 = scalar stationarity, F2 = mu_n - mu_p - mu_e ; plus Jacobian pieces and state."""
        m = MN * y
        nn = n * (1.0 + d) / 2.0
        npr = n * (1.0 - d) / 2.0
        kn, efn, en, pn, nsn, nsmn = fermi_s(nn, m, 2)
        kp, efp, ep, pp, nsp, nsmp = fermi_s(npr, m, 2)
        U, Uy, Uyy = potential(self.coef, y)
        Cv = self.Cv
        F1 = MN * (nsn + nsp) + Uy - Cv * n * n / y ** 3
        Cy = Uyy + MN ** 2 * (nsmn + nsmp) + 3.0 * Cv * n * n / y ** 4
        mue, dmue = lepton_mu(npr)
        F2 = efn - efp + C * n * d / 2.0 - mue
        inv_efn = 1.0 / efn
        inv_efp = 1.0 / efp if efp > 0 else 0.0
        J11 = Cy
        J12 = MN * (n / 2.0) * m * (inv_efn - inv_efp)
        J21 = MN * m * (inv_efn - inv_efp)
        dEn = kn * kn / (3.0 * nn * efn) if nn > 0 else 0.0
        dEp = kp * kp / (3.0 * npr * efp) if npr > 0 else 1e300
        J22 = (n / 2.0) * (dEn + dEp) + C * n / 2.0 + (n / 2.0) * dmue
        st = {"y": y, "delta": d, "mu_e": mue, "Cy": Cy, "U": U,
              "eps_N": en + ep + U + Cv * n * n / (2.0 * y * y),
              "P_N": pn + pp - U + Cv * n * n / (2.0 * y * y),
              "eps_rho": C * n * n * d * d / 8.0,
              "mu_n": efn + Cv * n / y ** 2 + C * n * d / 4.0,
              "mu_p": efp + Cv * n / y ** 2 - C * n * d / 4.0}
        return F1, F2, (J11, J12, J21, J22), st

    def solve(self, n, C, y0, d0, tol=1e-11):
        """Damped 2D Newton from (y0, d0); falls back to nested brentq."""
        y, d = y0, d0
        scale1 = MN * n
        ok = False
        for _ in range(80):
            F1, F2, (J11, J12, J21, J22), st = self.residuals(n, C, y, d)
            if abs(F1) < tol * scale1 and abs(F2) < 1e-9:
                ok = True
                break
            det = J11 * J22 - J12 * J21
            if det == 0.0:
                break
            dy = -(F1 * J22 - F2 * J12) / det
            dd = -(J11 * F2 - J21 * F1) / det
            lam = 1.0
            while y + lam * dy <= 0.02 or d + lam * dd <= 1e-9 or d + lam * dd >= 1.0 - 1e-12:
                lam *= 0.5
                if lam < 1e-8:
                    break
            if abs(dy) > 0.2 * y:
                lam = min(lam, 0.2 * y / abs(dy))
            y += lam * dy
            d += lam * dd
        if ok and st["Cy"] > 0.0:
            return st
        return self.solve_nested(n, C, y0)

    def solve_nested(self, n, C, y0):
        """Outer brentq on delta of F2 with y re-stationarised (1D Newton + brentq fallback)."""
        def y_of(d, yg):
            yy = yg
            for _ in range(60):
                F1, F2, (J11, J12, J21, J22), st = self.residuals(n, C, yy, d)
                if abs(F1) < 1e-12 * MN * n:
                    return yy
                step = -F1 / J11
                if abs(step) > 0.2 * yy:
                    step = math.copysign(0.2 * yy, step)
                yy = max(0.02, yy + step)
            ys = np.linspace(0.03, 1.3, 128)
            vals = [self.residuals(n, C, float(v), d)[0] for v in ys]
            best = None
            for i in range(len(ys) - 1):
                if vals[i] == 0.0 or vals[i] * vals[i + 1] < 0:
                    r = brentq(lambda v: self.residuals(n, C, v, d)[0], ys[i], ys[i + 1], xtol=1e-14)
                    if best is None or abs(r - yg) < abs(best - yg):
                        best = r
            if best is None:
                raise RuntimeError("no scalar root")
            return best
        yg = [y0]

        def F2_of(d):
            yy = y_of(d, yg[0])
            yg[0] = yy
            return self.residuals(n, C, yy, d)[1]
        d_root = brentq(F2_of, 0.0, 1.0 - 1e-9, xtol=1e-15, maxiter=300)
        yy = y_of(d_root, yg[0])
        F1, F2, Jm, st = self.residuals(n, C, yy, d_root)
        if st["Cy"] <= 0.0:
            raise RuntimeError("C_y <= 0 at solution")
        return st

    def global_min_check(self, n, C, d, y_sol, eps_sol):
        """Is the tracked stationary point the global minimum of eps_N(n,delta,y) over y? (diagnostic)"""
        ys = np.linspace(0.02, 1.4, 139)
        m = MN * ys
        nn = n * (1.0 + d) / 2.0
        npr = n * (1.0 - d) / 2.0
        _, _, en, _, _ = fermi_v(nn, m, 2)
        _, _, ep, _, _ = fermi_v(npr, m, 2) if npr > 0 else (0, 0, 0 * m, 0, 0)
        e = en + ep + potential_v(self.coef, ys) + self.Cv * n * n / (2.0 * ys * ys)
        i = int(np.argmin(e))
        return bool(e[i] >= eps_sol - 1e-9 * abs(eps_sol) or abs(ys[i] - y_sol) < 0.03), float(ys[i]), float(e[i] - eps_sol)


def _beta_row(bs, C_of_x, dln, n0, x, y0, d0, global_check):
    n = n0 * x
    C = C_of_x(x)
    st = bs.solve(n, C, y0, d0)
    le, lp = lepton_eps_p(st["mu_e"])
    eps_nat = st["eps_N"] + st["eps_rho"] + le
    P_auth = st["P_N"] + st["eps_rho"] + lp
    P_cons = P_auth + dln(x) * st["eps_rho"]
    gm = bs.global_min_check(n, C, st["delta"], st["y"], st["eps_N"]) if global_check else (True, None, None)
    row = {"x": x, "n_fm3": n / HBARC ** 3, "delta": st["delta"], "y": st["y"], "mu_e": st["mu_e"], "C": C,
           "eps_nat": eps_nat, "P_nat_author": P_auth, "P_nat_cons": P_cons, "eps_rho_nat": st["eps_rho"],
           "Cy": st["Cy"], "global_min_ok": gm[0], "y_global_min": gm[1], "dE_global": gm[2],
           "mu_n": st["mu_n"], "mu_p": st["mu_p"]}
    return row, st


def build_beta_table(jet, iso, xs, rearrangement=True, global_check=True):
    """Beta-equilibrated table on x = n/n0 grid `xs`.  rearrangement=True selects the thermodynamic
    P_rho,cons = eps_rho (1 + dlnC/dlnn) for arr['P'] (author convention: P_rho = eps_rho -> False);
    both pressures are always stored (P_cons, P_author)."""
    C_of_x, dln, pole = make_C_of_x(iso)
    n0 = jet["n0_nat"]
    bs = BetaSolver(jet["coef"], jet["Cv"], n0, C_of_x)
    xs = np.asarray(sorted(set(float(x) for x in xs)))
    i1 = int(np.argmin(np.abs(xs - 1.0)))
    N = len(xs)
    rows = [None] * N

    def do(i, y0, d0):
        rows[i], st = _beta_row(bs, C_of_x, dln, n0, xs[i], y0, d0, global_check)
        return st
    st = do(i1, jet["ystar"], 0.9)
    y0, d0 = st["y"], st["delta"]
    for i in range(i1 + 1, N):
        st = do(i, y0, d0)
        y0, d0 = st["y"], st["delta"]
    y0, d0 = rows[i1]["y"], rows[i1]["delta"]
    for i in range(i1 - 1, -1, -1):
        st = do(i, y0, d0)
        y0, d0 = st["y"], st["delta"]
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in
           ("x", "n_fm3", "delta", "y", "mu_e", "C", "eps_nat", "P_nat_author", "P_nat_cons", "eps_rho_nat", "Cy", "mu_n", "mu_p")}
    arr["global_min_ok"] = np.array([r["global_min_ok"] for r in rows], dtype=bool)
    arr["eps"] = arr["eps_nat"] / HBARC ** 3
    arr["P"] = (arr["P_nat_cons"] if rearrangement else arr["P_nat_author"]) / HBARC ** 3
    arr["P_author"] = arr["P_nat_author"] / HBARC ** 3
    arr["P_cons"] = arr["P_nat_cons"] / HBARC ** 3
    arr["pade_pole_x"] = pole
    arr["_solver"] = (bs, C_of_x, dln)
    return arr


def symmetry_energy_S(n_nat, y, C):
    """S(n) = T(n, y) + C n/8 (local-jet definition at the beta-equilibrium y)."""
    k, ef, *_ = fermi_s(n_nat, MN * y, 4)
    return float(k * k / (6.0 * ef) + C * n_nat / 8.0)


def table_point_exact(tab, jet, x):
    """Verifier correction 1: beta-equilibrium quantities EXACTLY at x = n/n0 (direct solve started
    from the nearest table node), plus cubic interpolation of the table in x and the legacy nearest-node
    values, so the bias of the nearest-node convention is recorded."""
    bs, C_of_x, dln = tab["_solver"]
    X = tab["x"]
    i = int(np.argmin(np.abs(X - x)))
    row, st = _beta_row(bs, C_of_x, dln, jet["n0_nat"], float(x), float(tab["y"][i]), float(tab["delta"][i]), False)
    n_nat = jet["n0_nat"] * x
    out = {"x": float(x),
           "P_cons": row["P_nat_cons"] / HBARC ** 3, "P_author": row["P_nat_author"] / HBARC ** 3,
           "eps": row["eps_nat"] / HBARC ** 3, "delta": row["delta"], "y": row["y"], "mu_e": row["mu_e"],
           "S": symmetry_energy_S(n_nat, row["y"], row["C"])}
    lo, hi = max(i - 3, 0), min(i + 4, len(X))
    if hi - lo >= 4 and X[lo] <= x <= X[hi - 1]:
        out["P_cons_interp_cubic"] = float(CubicSpline(X[lo:hi], tab["P_cons"][lo:hi])(x))
        out["P_author_interp_cubic"] = float(CubicSpline(X[lo:hi], tab["P_author"][lo:hi])(x))
    out["nearest_node"] = {"x": float(X[i]), "P_cons": float(tab["P_cons"][i]), "P_author": float(tab["P_author"][i]),
                           "delta": float(tab["delta"][i]), "y": float(tab["y"][i]),
                           "S": symmetry_energy_S(tab["n_fm3"][i] * HBARC ** 3, tab["y"][i], tab["C"][i])}
    out["rel_bias_nearest_node_P_cons"] = out["nearest_node"]["P_cons"] / out["P_cons"] - 1.0 if out["P_cons"] else None
    return out


# ----------------------------------------------------------------------------- EOS with the author's polytropic crust
class EOS:
    """eps(P) with the author's polytropic crust below p_match (eps = eps_match (P/p_match)^(1/Gamma)).
    mode='author': np.interp on the strictly-increasing-P table + finite-difference de/dP (1:1 with
    ForwardEOS); mode='spline': cubic spline in (ln P, ln eps) on the monotone high-density tail,
    tabulated densely, with analytic de/dP."""

    def __init__(self, P, eps, n_fm3=None, p_match=1.5, gamma=1.35, mode="spline", n_dense=4000):
        P = np.asarray(P, float)
        eps = np.asarray(eps, float)
        self.mode = mode
        self.p_match = float(p_match)
        self.gamma = float(gamma)
        self.crust_kind = "author_polytrope"
        self.jumps = []
        if mode == "author":
            pts = sorted(zip(P.tolist(), eps.tolist()))
            p_arr = np.array([p for p, _ in pts])
            e_arr = np.array([e for _, e in pts])
            keep = np.concatenate([[True], np.diff(p_arr) > 0])
            self.p_arr, self.e_arr = p_arr[keep], e_arr[keep]
            if not (self.p_arr[0] < p_match < self.p_arr[-1]):
                raise ValueError("p_match outside table")
            self.eps_match = float(np.interp(p_match, self.p_arr, self.e_arr))
            self.p_max = float(self.p_arr[-1])
            self.n_spl = None
            self.i_tail = 0
            self.cs2_seg = np.diff(self.p_arr) / np.diff(self.e_arr)
            self.P_seg_hi = self.p_arr[1:]
            if n_fm3 is not None:
                nn = np.asarray(n_fm3, float)
                order = np.argsort(np.asarray(P, float))
                n_sorted = nn[order][keep]
                pos = (self.p_arr > 0) & (n_sorted > 0)
                if pos.sum() >= 4:
                    self.n_spl = CubicSpline(np.log(self.p_arr[pos]), np.log(n_sorted[pos]))
        else:
            i = len(P) - 1
            while i > 0 and P[i - 1] < P[i] and P[i - 1] > 0.0:
                i -= 1
            self.i_tail = i
            Pt, Et = P[i:], eps[i:]
            if not (Pt[0] < p_match < Pt[-1]):
                raise ValueError(f"p_match {p_match} outside monotone table [{Pt[0]}, {Pt[-1]}]")
            lnP, lnE = np.log(Pt), np.log(Et)
            cs = CubicSpline(lnP, lnE)
            self.eps_match = float(math.exp(cs(math.log(p_match))))
            self.p_max = float(Pt[-1])
            self.lnP_tab = np.linspace(math.log(p_match), math.log(self.p_max), n_dense)
            self.lnE_tab = cs(self.lnP_tab)
            self.dlnE_tab = cs(self.lnP_tab, 1)
            self._lnP = self.lnP_tab.tolist()
            self._lnE = self.lnE_tab.tolist()
            self._dlnE = self.dlnE_tab.tolist()
            self._h = (self.lnP_tab[-1] - self.lnP_tab[0]) / (n_dense - 1)
            self._lnP0 = float(self.lnP_tab[0])
            self._n = n_dense
            self.n_spl = CubicSpline(lnP, np.log(np.asarray(n_fm3, float)[i:])) if n_fm3 is not None else None
            self.cs2_seg = np.diff(Pt) / np.diff(Et)
            self.P_seg_hi = Pt[1:]
        self.P_m = self.p_match
        self.eps_core_m = self.eps_match
        self.eps_crust_m = self.eps_match
        self.eps_jump_rel = 0.0

    def _eps_author(self, P):
        if P == 0.0:
            return 0.0
        if P < self.p_match:
            return self.eps_match * (P / self.p_match) ** (1.0 / self.gamma)
        return float(np.interp(P, self.p_arr, self.e_arr))

    def _dedp_author(self, P):
        dP = max(P * 1e-4, 1e-8)
        if P <= dP:
            return (self._eps_author(P + dP) - self._eps_author(P)) / dP
        if P + dP > self.p_max:
            return (self._eps_author(P) - self._eps_author(P - dP)) / dP
        return (self._eps_author(P + dP) - self._eps_author(P - dP)) / (2 * dP)

    def _lookup_core(self, P):
        lp = math.log(P)
        t = (lp - self._lnP0) / self._h
        i = int(t)
        if i >= self._n - 1:
            i = self._n - 2
        f = t - i
        lnE = self._lnE[i] + f * (self._lnE[i + 1] - self._lnE[i])
        dln = self._dlnE[i] + f * (self._dlnE[i + 1] - self._dlnE[i])
        e = math.exp(lnE)
        return e, dln * e / P

    def _lookup(self, P):
        if P <= 0.0:
            return 0.0, 1.0
        if P < self.p_match:
            e = self.eps_match * (P / self.p_match) ** (1.0 / self.gamma)
            return e, e / (self.gamma * P)
        return self._lookup_core(P)

    def eps_dedp(self, P):
        if P > self.p_max * (1 + 1e-12):
            raise ValueError(f"P={P} above table p_max={self.p_max}")
        if self.mode == "author":
            return self._eps_author(P), self._dedp_author(P)
        return self._lookup(P)

    def n_of_P(self, P):
        if self.n_spl is None or P < self.p_match:
            return None
        return float(math.exp(self.n_spl(math.log(P))))

    def max_cs2_below(self, P):
        """max segment sound speed on the monotone tail for P_seg <= P (stellar-branch causality)."""
        mask = self.P_seg_hi <= P
        if not mask.any():
            return float(self.cs2_seg[0])
        return float(self.cs2_seg[mask].max())


# ----------------------------------------------------------------------------- SLy crust
class SlyCrust:
    """Piecewise-polytropic SLy crust (Read+2009 Table II) in c=1 cgs internally; API in MeV/fm^3, fm^-3.
    eps(rho) = (1 + a_i) rho + K_i rho^Gamma_i/(Gamma_i - 1), a_0 = 0, a_i from eps-continuity (eq. 7).
    P -> piece uses the LOWER piece's pressure at each dividing density (six-digit table mismatch
    1.5e-5 / -1.2e-4 / -1.6e-5 deep in the outer crust)."""

    def __init__(self, table=SLY_TABLE_II, m_B_g=M_B_G):
        self.K = [t[0] for t in table]
        self.G = [t[1] for t in table]
        self.rho_div = [t[2] for t in table[:-1]]
        self.m_B_g = m_B_g
        a = [0.0]
        for i in range(1, len(table)):
            r = self.rho_div[i - 1]
            eps_prev = self._eps_rho_piece(i - 1, r, a[i - 1])
            a.append(eps_prev / r - 1.0 - self.K[i] * r ** (self.G[i] - 1.0) / (self.G[i] - 1.0))
        self.a = a
        self.P_div = [self.K[i] * self.rho_div[i] ** self.G[i] for i in range(len(self.rho_div))]
        self.P_div_mismatch_rel = [self.K[i + 1] * self.rho_div[i] ** self.G[i + 1] / self.P_div[i] - 1.0
                                   for i in range(len(self.rho_div))]

    def _eps_rho_piece(self, i, rho, a=None):
        a = self.a[i] if a is None else a
        return (1.0 + a) * rho + self.K[i] * rho ** self.G[i] / (self.G[i] - 1.0)

    def piece_of_rho(self, rho):
        i = 0
        while i < len(self.rho_div) and rho > self.rho_div[i]:
            i += 1
        return i

    def piece_of_P(self, P):
        i = 0
        while i < len(self.P_div) and P > self.P_div[i]:
            i += 1
        return i

    def P_eps_of_rho_cgs(self, rho):
        i = self.piece_of_rho(rho)
        return self.K[i] * rho ** self.G[i], self._eps_rho_piece(i, rho)

    def rho_eps_dedP_of_P_cgs(self, P):
        i = self.piece_of_P(P)
        K, G, a = self.K[i], self.G[i], self.a[i]
        rho = (P / K) ** (1.0 / G)
        eps = (1.0 + a) * rho + K * rho ** G / (G - 1.0)
        dPdrho = G * K * rho ** (G - 1.0)
        dedrho = (1.0 + a) + G * K * rho ** (G - 1.0) / (G - 1.0)
        return rho, eps, dedrho / dPdrho

    def eps_dedp(self, P_mev):
        """eps [MeV/fm^3] and d eps/dP at pressure P [MeV/fm^3]."""
        if P_mev <= 0.0:
            return 0.0, 1.0
        rho, eps, dedp = self.rho_eps_dedP_of_P_cgs(P_mev * MEVFM3_TO_GCC)
        return eps / MEVFM3_TO_GCC, dedp

    def eps_of_P(self, P_mev):
        return self.eps_dedp(P_mev)[0]

    def n_of_P(self, P_mev):
        if P_mev <= 0.0:
            return 0.0
        rho = self.rho_eps_dedP_of_P_cgs(P_mev * MEVFM3_TO_GCC)[0]
        return rho / N_FM3_TO_RHO_GCC

    def P_eps_of_n(self, n_fm3):
        P, e = self.P_eps_of_rho_cgs(n_fm3 * N_FM3_TO_RHO_GCC)
        return P / MEVFM3_TO_GCC, e / MEVFM3_TO_GCC

    def describe(self):
        return {
            "source": CRUST_SOURCE,
            "pieces_cgs_c1": [{"K": self.K[i], "Gamma": self.G[i],
                               "rho_upper_gcc": (self.rho_div[i] if i < len(self.rho_div) else None),
                               "a_i": self.a[i]} for i in range(len(self.K))],
            "P_at_dividing_densities_MeVfm3": [p / MEVFM3_TO_GCC for p in self.P_div],
            "P_mismatch_rel_at_dividing_densities_from_6digit_table": self.P_div_mismatch_rel,
            "MEVFM3_TO_GCC": MEVFM3_TO_GCC, "m_B_g": self.m_B_g, "p_surf_MeV_fm3": P_SURF_SLY,
            "conventions": "P -> piece via the lower piece's pressure at each dividing density; rho = m_B n; "
                           "matching: 'eps' = eps-continuity root on the monotone core tail (spline convention), "
                           "'P' = P-continuity at x_m with the Hinderer jump condition (" + JUMP_SOURCE + ")",
        }


SLY = SlyCrust()


def monotone_tail(P):
    """Index i such that P[i:] is the strictly increasing, positive tail (same walk as EOS spline mode)."""
    P = np.asarray(P, float)
    i = len(P) - 1
    while i > 0 and P[i - 1] < P[i] and P[i - 1] > 0.0:
        i -= 1
    return i


class MergedEOS(EOS):
    """EOS (spline mode) whose region P < P_m is the SLy crust instead of the polytrope."""

    def __init__(self, P, eps, n_fm3, P_m, crust=SLY, n_dense=4000):
        super().__init__(P, eps, n_fm3, p_match=float(P_m), gamma=1.35, mode="spline", n_dense=n_dense)
        self.crust = crust
        self.crust_kind = "sly"
        self.P_m = float(P_m)
        self.eps_core_m = float(self.eps_match)
        self.eps_crust_m = float(crust.eps_of_P(self.P_m))
        self.eps_jump_rel = self.eps_core_m / self.eps_crust_m - 1.0
        self.jumps = [(self.P_m, self.eps_core_m - self.eps_crust_m)]

    def _lookup(self, P):
        if P <= 0.0:
            return 0.0, 1.0
        if P < self.P_m:
            return self.crust.eps_dedp(P)
        return self._lookup_core(P)

    def n_of_P(self, P):
        if P < self.P_m:
            return self.crust.n_of_P(P)
        return super().n_of_P(P)


def match_sly(P, eps, x, n_fm3, crust=SLY, method="eps", x_m=None, match_interp="spline", P_scan=(None, 3.0), n_scan=240):
    """Crust/core interface for a core table.  method='eps': lowest-P root of eps_core(P) - eps_SLy(P) on the
    monotone tail; method='P': P_m = P_core(x_m), eps discontinuous."""
    P = np.asarray(P, float); eps = np.asarray(eps, float); x = np.asarray(x, float); n = np.asarray(n_fm3, float)
    it = monotone_tail(P)
    Pt, Et, Xt, Nt = P[it:], eps[it:], x[it:], n[it:]
    lnP, lnE = np.log(Pt), np.log(Et)
    cs = CubicSpline(lnP, lnE)
    x_of_P = CubicSpline(lnP, np.log(Xt))

    def eps_core(Pm):
        if match_interp == "linear":
            return float(np.interp(Pm, Pt, Et))
        return float(math.exp(cs(math.log(Pm))))

    out = {"i_tail": int(it), "x_tail_start": float(Xt[0]), "P_tail_start": float(Pt[0]), "method": method,
           "match_interp": match_interp}
    if method == "eps":
        lo = P_scan[0] if P_scan[0] is not None else Pt[0] * (1.0 + 1e-9)
        hi = min(P_scan[1], Pt[-1] * (1.0 - 1e-9))
        grid = np.logspace(math.log10(lo), math.log10(hi), n_scan)

        def g(Pm):
            return eps_core(Pm) - crust.eps_of_P(Pm)
        vals = [g(pp) for pp in grid]
        roots = [brentq(g, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-13, maxiter=200)
                 for i in range(len(grid) - 1) if vals[i] == 0.0 or vals[i] * vals[i + 1] < 0.0]
        if not roots:
            raise ValueError("no eps-continuous SLy/core match on the monotone tail "
                             f"(g at ends: {vals[0]:.4g}, {vals[-1]:.4g})")
        P_m = float(roots[0])
        out.update(P_m=P_m, roots_P=[float(r) for r in roots],
                   roots_x=[float(math.exp(x_of_P(math.log(r)))) for r in roots],
                   x_m=float(math.exp(x_of_P(math.log(P_m)))), n_m_fm3=float(np.interp(P_m, Pt, Nt)))
    elif method == "P":
        if x_m is None:
            raise ValueError("method='P' needs x_m")
        if not (Xt[0] <= x_m <= Xt[-1]):
            raise ValueError(f"x_m={x_m} outside the monotone tail [{Xt[0]:.4f}, {Xt[-1]:.4f}]")
        P_m = float(np.interp(x_m, Xt, Pt))
        if P_m <= 0.0:
            raise ValueError("P_core(x_m) <= 0")
        out.update(P_m=P_m, x_m=float(x_m), roots_P=None, roots_x=None, n_m_fm3=float(np.interp(x_m, Xt, Nt)))
    else:
        raise ValueError(method)
    out["eps_core_m_spline"] = float(math.exp(cs(math.log(out["P_m"]))))
    out["eps_crust_m"] = float(crust.eps_of_P(out["P_m"]))
    out["eps_jump_rel"] = out["eps_core_m_spline"] / out["eps_crust_m"] - 1.0
    return out


def build_eos(tab, crust="author", match="eps", x_m=None, p_match=1.5, gamma=1.35, match_interp="spline",
              pressure="consistent", eos_mode="spline"):
    """EOS with crust from a build_beta_table dict.  pressure: 'consistent' | 'author'; crust: 'author' | 'sly'."""
    P = tab["P_cons"] if pressure == "consistent" else tab["P_author"]
    E, X, N = tab["eps"], tab["x"], tab["n_fm3"]
    if crust == "author":
        eos = EOS(P, E, N, p_match=p_match, gamma=gamma, mode=eos_mode)
        it = monotone_tail(P)
        info = {"method": "author_polytrope", "P_m": float(p_match), "gamma_crust": float(gamma),
                "x_m": float(np.interp(p_match, P[it:], X[it:])), "eps_core_m_spline": eos.eps_match,
                "eps_crust_m": eos.eps_match, "eps_jump_rel": 0.0, "i_tail": int(it), "x_tail_start": float(X[it]),
                "P_tail_start": float(P[it])}
        return eos, info
    if crust != "sly":
        raise ValueError(crust)
    info = match_sly(P, E, X, N, crust=SLY, method=match, x_m=x_m, match_interp=match_interp)
    eos = MergedEOS(P, E, N, info["P_m"], crust=SLY)
    info["eps_core_m_eos"] = eos.eps_core_m
    info["eps_jump_rel_eos"] = eos.eps_jump_rel
    return eos, info


# ----------------------------------------------------------------------------- TOV + Hinderer
def _tov_rhs_factory(eos, K_CONV, M_SUN_KM):
    eps_dedp = eos.eps_dedp
    fourpi = 4.0 * math.pi

    def rhs(r, v):
        m, P, y = v[0], v[1], v[2]
        if r < 1e-10 or P <= 0.0:
            return [0.0, 0.0, 0.0]
        eps, dedp = eps_dedp(P)
        ek = eps * K_CONV
        pk = P * K_CONV
        mk = m * M_SUN_KM
        fac = r * (r - 2.0 * mk)
        if fac <= 0.0:
            return [0.0, 0.0, 0.0]
        dm = fourpi * r * r * ek / M_SUN_KM
        dP = -(ek + pk) * (mk + fourpi * r ** 3 * pk) / fac / K_CONV
        one = 1.0 - 2.0 * mk / r
        F = (1.0 - fourpi * r * r * (ek - pk)) / one
        if dedp <= 0.0:
            dedp = 1.0
        Qs = fourpi * (5.0 * ek + 9.0 * pk + (ek + pk) * dedp) / one
        Qg = (2.0 * (mk + fourpi * r ** 3 * pk) / (r * one)) ** 2 / r ** 2
        Q = Qs - Qg - 6.0 / (r * r * one)
        dy = -(y * y + y * F + r * r * Q) / r
        return [dm, dP, dy]
    return rhs


def _love_from_surface(m, r, y, M_SUN_KM):
    Msol = float(m)
    C = Msol * M_SUN_KM / r if r > 0 else 0.0
    yR = float(y)
    if C <= 0 or C >= 0.5:
        return Msol, r, 0.0, 0.0
    fac = 1.0 - 2.0 * C
    ln = math.log(fac)
    num = (8.0 / 5.0) * C ** 5 * fac ** 2 * (2.0 + 2.0 * C * (yR - 1.0) - yR)
    den = (2.0 * C * (6.0 - 3.0 * yR + 3.0 * C * (5.0 * yR - 8.0))
           + 4.0 * C ** 3 * (13.0 - 11.0 * yR + C * (3.0 * yR - 2.0) + 2.0 * C * C * (1.0 + yR))
           + 3.0 * fac ** 2 * (2.0 - yR + 2.0 * C * (yR - 1.0)) * ln)
    if abs(den) < 1e-30:
        return Msol, r, 0.0, 0.0
    k2 = num / den
    Lam = (2.0 / 3.0) * k2 * C ** (-5) if k2 > 0 else 0.0
    return Msol, r, k2, Lam


def solve_tov_tidal(eos, P_center, rtol=1e-8, atol=1e-10, max_step=0.05, p_surf=1e-4, const="author", r0=1e-6):
    """1:1 transcription of nvg_forward_tov_heldout_audit.solve_tov_tidal (r km, m M_sun, P MeV/fm^3)."""
    K_CONV = CONST[const]["K_CONV"]
    M_SUN_KM = CONST[const]["M_SUN_KM"]
    eos.eps_dedp(P_center)  # fail closed
    rhs = _tov_rhs_factory(eos, K_CONV, M_SUN_KM)

    def surf(r, v):
        return v[1] - p_surf
    surf.terminal = True
    surf.direction = -1
    sol = solve_ivp(rhs, (r0, 100.0), [0.0, float(P_center), 2.0], method="DOP853",
                    rtol=rtol, atol=atol, max_step=max_step, events=[surf])
    if not sol.success:
        raise RuntimeError(sol.message)
    m, P, y = sol.y[:, -1]
    return _love_from_surface(m, float(sol.t[-1]), y, M_SUN_KM)


def solve_tov_tidal_jumps(eos, P_center, jumps, rtol=1e-8, atol=1e-10, max_step=0.05, p_surf=1e-4, const="author", r0=1e-6):
    """solve_tov_tidal with the Hinderer jump condition at pressure interfaces (jumps: (P_j, Deps_j inner-outer))."""
    K_CONV = CONST[const]["K_CONV"]
    M_SUN_KM = CONST[const]["M_SUN_KM"]
    eos.eps_dedp(P_center)
    rhs = _tov_rhs_factory(eos, K_CONV, M_SUN_KM)
    fourpi = 4.0 * math.pi
    r_cur, v = r0, [0.0, float(P_center), 2.0]
    for Pj, deps in sorted([j for j in jumps if p_surf < j[0] < P_center], key=lambda j: -j[0]):
        def ev(r, vv, Pj=Pj):
            return vv[1] - Pj
        ev.terminal = True
        ev.direction = -1
        sol = solve_ivp(rhs, (r_cur, 100.0), v, method="DOP853", rtol=rtol, atol=atol, max_step=max_step, events=[ev])
        if not sol.success:
            raise RuntimeError(sol.message)
        if sol.t_events[0].size == 0:
            raise RuntimeError("interface pressure not reached")
        m, P, y = sol.y[:, -1]
        r_cur = float(sol.t[-1])
        mk, pk, dek = m * M_SUN_KM, P * K_CONV, deps * K_CONV
        y = y - fourpi * r_cur ** 3 * dek / (mk + fourpi * r_cur ** 3 * pk)
        v = [float(m), float(P), float(y)]

    def surf(r, vv):
        return vv[1] - p_surf
    surf.terminal = True
    surf.direction = -1
    sol = solve_ivp(rhs, (r_cur, 100.0), v, method="DOP853", rtol=rtol, atol=atol, max_step=max_step, events=[surf])
    if not sol.success:
        raise RuntimeError(sol.message)
    m, P, y = sol.y[:, -1]
    return _love_from_surface(m, float(sol.t[-1]), y, M_SUN_KM)


def lambda_tilde(m1, m2, L1, L2):
    return (16.0 / 13.0) * ((m1 + 12.0 * m2) * m1 ** 4 * L1 + (m2 + 12.0 * m1) * m2 ** 4 * L2) / (m1 + m2) ** 5


def stellar_branch(eos, Pcs=None, n_pc=70, pc_min=2.0, pc_frac=0.97, masses=(1.27, 1.34, 1.36, 1.4, 1.418, 1.44, 1.46, 2.07),
                   tov=None, refine=True, solver=None):
    """Pc grid -> stable branch (up to the grid maximum of M) -> M_max refinement (3-point parabola in ln Pc +
    one solve) -> R(M), Lambda(M) at requested masses via PCHIP inversion of M(ln Pc) plus direct solves with
    secant correction."""
    tov = dict(tov or {})
    solve = solver or solve_tov_tidal
    if Pcs is None:
        Pcs = np.logspace(math.log10(pc_min), math.log10(eos.p_max * pc_frac), n_pc)
    res = []
    n_solves = 0
    for Pc in Pcs:
        try:
            M, R, k2, Lam = solve(eos, float(Pc), **tov)
            n_solves += 1
        except Exception:  # noqa: BLE001
            continue
        if 0.05 < M < 5 and 3 < R < 60 and k2 >= 0 and Lam >= 0:
            res.append((float(Pc), M, R, k2, Lam))
    if len(res) < 5:
        return {"status": "NO_BRANCH", "n_accepted": len(res)}
    arr = np.array(res)
    Ms = arr[:, 1]
    idx = int(np.argmax(Ms))
    out = {"status": "ok", "n_grid": int(len(Pcs)), "n_accepted": len(res), "M_max_grid": float(Ms[idx]),
           "M_max_grid_Pc": float(arr[idx, 0]), "M_max_on_grid_end": bool(idx == len(res) - 1),
           "M_max_on_grid_start": bool(idx == 0)}
    stable = arr[: idx + 1]
    Mmax, Pc_max, R_max = float(Ms[idx]), float(arr[idx, 0]), float(arr[idx, 2])
    if refine and 0 < idx < len(res) - 1:
        x = np.log(arr[idx - 1: idx + 2, 0])
        yv = arr[idx - 1: idx + 2, 1]
        A = np.vstack([x ** 2, x, np.ones(3)]).T
        c = np.linalg.solve(A, yv)
        if c[0] < 0:
            xv = -c[1] / (2 * c[0])
            if x[0] < xv < x[2]:
                try:
                    M, R, k2, Lam = solve(eos, float(math.exp(xv)), **tov)
                    n_solves += 1
                    if M > Mmax:
                        Mmax, Pc_max, R_max = M, float(math.exp(xv)), R
                        stable = np.vstack([stable, [Pc_max, M, R, k2, Lam]])
                except Exception:  # noqa: BLE001
                    pass
    out.update(M_max=Mmax, M_max_Pc=Pc_max, R_at_Mmax=R_max)
    nc = eos.n_of_P(Pc_max)
    out["n_c_Mmax_fm3"] = nc
    out["max_cs2_stellar_branch"] = eos.max_cs2_below(Pc_max)
    Pb, Mb, Rb, k2b, Lb = stable[:, 0], stable[:, 1], stable[:, 2], stable[:, 3], stable[:, 4]
    order = np.argsort(Pb)
    Pb, Mb, Rb, k2b, Lb = Pb[order], Mb[order], Rb[order], k2b[order], Lb[order]
    out["branch_M_monotone"] = bool(np.all(np.diff(Mb) > 0))
    cm = np.maximum.accumulate(Mb)
    keep = np.concatenate([[True], Mb[1:] > cm[:-1]])
    Pb, Mb, Rb, k2b, Lb = Pb[keep], Mb[keep], Rb[keep], k2b[keep], Lb[keep]
    out["M_min_branch"] = float(Mb[0])
    out["n_points_stable"] = int(len(Mb))
    out["R_of_M"] = {}
    out["Lambda_of_M"] = {}
    out["k2_of_M"] = {}
    out["M_err_of_M"] = {}
    if len(Mb) >= 4:
        pch = PchipInterpolator(Mb, np.log(Pb))
        for mt in masses:
            if not (Mb[0] < mt < Mmax):
                out["R_of_M"][str(mt)] = None
                out["Lambda_of_M"][str(mt)] = None
                out["k2_of_M"][str(mt)] = None
                continue
            try:
                if mt < Mb[-1]:
                    Pc_t = float(math.exp(pch(mt)))
                else:
                    Pc_t = float(math.exp(np.interp(mt, [Mb[-1], Mmax], [math.log(Pb[-1]), math.log(Pc_max)])))
                M1, R1, k21, L1 = solve(eos, Pc_t, **tov)
                n_solves += 1
                for _ in range(4):
                    if abs(M1 - mt) < 5e-5:
                        break
                    i = min(max(int(np.searchsorted(Mb, mt)), 1), len(Mb) - 1)
                    slope = (Mb[i] - Mb[i - 1]) / (math.log(Pb[i]) - math.log(Pb[i - 1]))
                    if slope <= 0:
                        break
                    Pc_t = float(math.exp(math.log(Pc_t) + (mt - M1) / slope))
                    if Pc_t >= Pc_max:
                        Pc_t = 0.5 * (Pc_t + Pc_max)
                    M1, R1, k21, L1 = solve(eos, Pc_t, **tov)
                    n_solves += 1
                if abs(M1 - mt) > 5e-5:
                    out["R_of_M"][str(mt)] = float(np.interp(mt, Mb, Rb))
                    out["Lambda_of_M"][str(mt)] = float(np.interp(mt, Mb, Lb))
                    out["k2_of_M"][str(mt)] = float(np.interp(mt, Mb, k2b))
                    out["M_err_of_M"][str(mt)] = float(M1 - mt)
                else:
                    out["R_of_M"][str(mt)] = R1
                    out["Lambda_of_M"][str(mt)] = L1
                    out["k2_of_M"][str(mt)] = k21
                    out["M_err_of_M"][str(mt)] = float(M1 - mt)
            except Exception as exc:  # noqa: BLE001
                out["R_of_M"][str(mt)] = float(np.interp(mt, Mb, Rb))
                out["Lambda_of_M"][str(mt)] = float(np.interp(mt, Mb, Lb))
                out["k2_of_M"][str(mt)] = float(np.interp(mt, Mb, k2b))
                out["M_err_of_M"][str(mt)] = f"interp_fallback: {exc}"
    out["n_tov_solves"] = n_solves
    out["R_14_interp"] = float(np.interp(1.4, Mb, Rb)) if Mb[0] < 1.4 < Mb[-1] else None
    out["Lambda_14_interp"] = float(np.interp(1.4, Mb, Lb)) if Mb[0] < 1.4 < Mb[-1] else None
    return out


def stellar_branch_for_eos(eos, s, p_surf, tidal_jump=False):
    """stellar_branch with a settings dict; tidal_jump routes every solve through the jump solver when the
    EOS carries density jumps (identical to the plain solver when eos.jumps is empty)."""
    tov = {"rtol": s["rtol"], "atol": s["atol"], "max_step": s["max_step"], "p_surf": p_surf, "const": s["const"]}
    jumps = getattr(eos, "jumps", None) or []
    solver = None
    if tidal_jump and jumps:
        def solver(e, Pc, **kw):
            return solve_tov_tidal_jumps(e, Pc, jumps, **kw)
    return stellar_branch(eos, n_pc=s["n_pc"], pc_min=s["pc_min"], pc_frac=s["pc_frac"], tov=tov, solver=solver)


# ----------------------------------------------------------------------------- held-out comparison (the ONLY consumer of star data)
def heldout_compare(br):
    """Pass/fail per declared criterion for a stellar-branch result; 2 sigma (and 1 sigma) with asymmetric
    errors.  Radii at masses the model cannot support => FAIL (the star exists)."""
    out = {}
    if br.get("status") != "ok":
        return {k: {"pass2": False, "pass1": False, "value": None, "note": "no branch"} for k in HELDOUT_STAR_DATA} | {"joint2": False, "joint1": False}
    Mmax = br["M_max"]
    RofM = br["R_of_M"]
    LofM = br["Lambda_of_M"]
    joint2 = True
    joint1 = True
    for key, spec in HELDOUT_STAR_DATA.items():
        t = spec["type"]
        if t == "Mmax_ge":
            v = Mmax
            p2 = p1 = bool(v >= spec["value"])
            out[key] = {"value": v, "pass2": p2, "pass1": p1}
        elif t == "radius":
            mt = spec["M"]
            if Mmax < mt:
                if spec.get("if_reached"):
                    out[key] = {"value": None, "pass2": None, "pass1": None, "note": "not reached (M_max < M)"}
                    continue
                out[key] = {"value": None, "pass2": False, "pass1": False, "note": "M_max < M: star unsupported"}
                p2 = p1 = False
            else:
                v = RofM.get(str(mt))
                if v is None:
                    out[key] = {"value": None, "pass2": False, "pass1": False, "note": "R(M) unavailable"}
                    p2 = p1 = False
                else:
                    lo2, hi2 = spec["c"] - 2 * spec["sm"], spec["c"] + 2 * spec["sp"]
                    lo1, hi1 = spec["c"] - spec["sm"], spec["c"] + spec["sp"]
                    p2 = bool(lo2 <= v <= hi2)
                    p1 = bool(lo1 <= v <= hi1)
                    out[key] = {"value": v, "pass2": p2, "pass1": p1, "band2": [lo2, hi2], "band1": [lo1, hi1]}
        elif t == "band":
            mt = spec["M"]
            v = RofM.get(str(mt)) if Mmax >= mt else None
            p2 = p1 = bool(v is not None and spec["lo"] <= v <= spec["hi"])
            out[key] = {"value": v, "pass2": p2, "pass1": p1}
        elif t == "Lt_le":
            L136 = LofM.get("1.36") if Mmax >= 1.36 else None
            v = lambda_tilde(1.36, 1.36, L136, L136) if L136 is not None else None
            p2 = p1 = bool(v is not None and v <= spec["value"])
            out[key] = {"value": v, "pass2": p2, "pass1": p1}
        elif t == "band_L":
            v = LofM.get("1.4") if Mmax >= 1.4 else None
            p2 = p1 = bool(v is not None and spec["lo"] <= v <= spec["hi"])
            out[key] = {"value": v, "pass2": p2, "pass1": p1}
        out[key]["threshold"] = THRESHOLD_NOTE.get(key)
        joint2 = joint2 and p2
        joint1 = joint1 and p1
    out["joint2"] = bool(joint2)
    out["joint1"] = bool(joint1)
    L146, L127 = LofM.get("1.46"), LofM.get("1.27")
    out["Lambda_tilde_asym_146_127"] = lambda_tilde(1.46, 1.27, L146, L127) if (L146 is not None and L127 is not None) else None
    out["fails2"] = [k for k, v in out.items() if isinstance(v, dict) and v.get("pass2") is False]
    out["fails1"] = [k for k, v in out.items() if isinstance(v, dict) and v.get("pass1") is False]
    out["not_reached"] = [k for k, v in out.items() if isinstance(v, dict) and v.get("pass2") is None]
    return out


# ----------------------------------------------------------------------------- settings
DEFAULT_SETTINGS = {
    "x_min": 0.10, "x_max": 15.0, "n_x": 220,
    "p_match": 1.5, "gamma_crust": 1.35, "eos_mode": "spline",
    "rtol": 1e-8, "atol": 1e-10, "max_step": 0.1, "p_surf": 1e-4, "const": "codata",
    "n_pc": 70, "pc_min": 2.0, "pc_frac": 0.97,
    "rearrangement": True, "global_check": True,
    "crust": "author",            # 'author' | 'sly' (band crust)
    "p_surf_sly": P_SURF_SLY,
}


def x_grid(settings):
    xs = np.logspace(math.log10(settings["x_min"]), math.log10(settings["x_max"]), settings["n_x"])
    return np.unique(np.concatenate([xs, [1.0]]))


def branch_summary(br, n0):
    R, L = br["R_of_M"], br["Lambda_of_M"]
    L136 = L.get("1.36")
    nc = br.get("n_c_Mmax_fm3")
    return {"M_max": br["M_max"], "M_max_Pc": br["M_max_Pc"], "R_at_Mmax": br["R_at_Mmax"],
            "R_14": R.get("1.4"), "R_134": R.get("1.34"), "R_1418": R.get("1.418"), "R_144": R.get("1.44"), "R_207": R.get("2.07"),
            "Lambda_14": L.get("1.4"), "Lambda_136": L136,
            "Lambda_tilde_136_q1": lambda_tilde(1.36, 1.36, L136, L136) if L136 is not None else None,
            "Lambda_tilde_146_127": (lambda_tilde(1.46, 1.27, L.get("1.46"), L.get("1.27"))
                                     if (L.get("1.46") is not None and L.get("1.27") is not None) else None),
            "max_cs2_stellar_branch": br["max_cs2_stellar_branch"],
            "x_c_Mmax": (nc / n0) if nc else None, "n_c_Mmax_fm3": nc,
            "M_max_on_grid_end": br["M_max_on_grid_end"], "branch_M_monotone": br["branch_M_monotone"],
            "n_tov_solves": br["n_tov_solves"], "M_err_of_M": br.get("M_err_of_M")}


# ----------------------------------------------------------------------------- one EOS of the band
def evaluate_eos(jet, J, L, form, settings, label=""):
    """Isovector + beta table + crust + TOV + held-out comparison for one scalar jet (band sample)."""
    s = dict(DEFAULT_SETTINGS)
    s.update(settings or {})
    t0 = time.time()
    iso = isovector(jet, J, L, form)
    rec = {"label": label, "form": form, "J": J, "L": L, "crust": s["crust"],
           "pressure": "consistent" if s["rearrangement"] else "author"}
    if iso["status"] != "ok":
        rec["status"] = iso["status"]
        return rec
    rec.update({k: iso[k] for k in ("C0", "gamma", "alpha", "L_kin", "L_const", "T")})
    rec["gamma_lt_minus1"] = bool(iso["gamma"] < -1.0)
    try:
        tab = build_beta_table(jet, iso, x_grid(s), rearrangement=s["rearrangement"], global_check=s["global_check"])
    except Exception as exc:  # noqa: BLE001
        rec["status"] = f"beta_table_failed: {exc!r}"
        return rec
    rec["pade_pole_x"] = tab["pade_pole_x"]
    rec["pade_pole_in_range"] = bool(tab["pade_pole_x"] is not None and s["x_min"] < tab["pade_pole_x"] < s["x_max"])
    rec["branch_global_min_ok_all"] = bool(tab["global_min_ok"].all())
    rec["branch_global_min_first_bad_x"] = float(tab["x"][~tab["global_min_ok"]][0]) if (~tab["global_min_ok"]).any() else None
    rec["Cy_min"] = float(tab["Cy"].min())
    P, E, X = tab["P"], tab["eps"], tab["x"]
    pkey = "P_cons" if s["rearrangement"] else "P_author"
    # verifier correction 1: exact evaluation at x = 1 and x = 2 (direct solve), not the nearest node
    try:
        at1 = table_point_exact(tab, jet, 1.0)
        at2 = table_point_exact(tab, jet, 2.0)
        rec["P_beta_n0"] = at1[pkey]; rec["eps_beta_n0"] = at1["eps"]; rec["delta_n0"] = at1["delta"]; rec["y_n0"] = at1["y"]; rec["S_n0"] = at1["S"]
        rec["P_2n0"] = at2[pkey]; rec["delta_2n0"] = at2["delta"]; rec["y_2n0"] = at2["y"]; rec["S_2n0"] = at2["S"]
        rec["P_author_conv_2n0"] = at2["P_author"]
        rec["P_2n0_interp_cubic"] = at2.get(pkey + "_interp_cubic")
        rec["P_2n0_nearest_node_legacy"] = at2["nearest_node"]
        rec["P_2n0_rel_bias_nearest_node"] = at2["rel_bias_nearest_node_P_cons"]
        rec["x_used_for_2n0"] = at2["x"]
    except Exception as exc:  # noqa: BLE001
        rec["P_2n0_exact_error"] = repr(exc)
    dP = np.diff(P); dE = np.diff(E)
    seg = dP / dE
    rec["table_nonmonotone_P_above_x"] = [float(X[i + 1]) for i in range(len(dP)) if dP[i] <= 0 and X[i + 1] > 0.9]
    rec["table_max_cs2_to_xmax"] = float(np.nanmax(seg[np.isfinite(seg)]))
    it = monotone_tail(P)
    rec["x_monotone_from"] = float(X[it])
    rec["P_at_monotone_start"] = float(P[it])
    crust = s["crust"]
    if crust == "author" and P[it] >= s["p_match"]:
        rec["status"] = "thermo_unstable_dPdn_le_0_above_crust"
        rec["thermo_unstable_x_range"] = [float(X[max(it - 1, 0)]), float(X[it])]
        return rec
    bad = (~tab["global_min_ok"]) & (X >= X[it])
    rec["metastable_branch_above_crust_x"] = [float(X[bad][0]), float(X[bad][-1])] if bad.any() else None
    try:
        eos, info = build_eos(tab, crust=crust, p_match=s["p_match"], gamma=s["gamma_crust"],
                              pressure="consistent" if s["rearrangement"] else "author", eos_mode=s["eos_mode"])
    except Exception as exc:  # noqa: BLE001
        rec["status"] = f"crust_match_failed: {exc!r}"
        return rec
    rec["matching"] = {k: info.get(k) for k in ("method", "P_m", "x_m", "eps_core_m_spline", "eps_crust_m", "eps_jump_rel", "x_tail_start", "P_tail_start")}
    rec["eps_match"] = eos.eps_match
    rec["p_max_table"] = eos.p_max
    p_surf = s["p_surf"] if crust == "author" else s["p_surf_sly"]
    rec["p_surf"] = p_surf
    try:
        br = stellar_branch_for_eos(eos, s, p_surf)
    except Exception as exc:  # noqa: BLE001
        rec["status"] = f"tov_failed: {exc!r}"
        return rec
    if br["status"] != "ok":
        rec["status"] = "no_branch"
        rec["branch_status"] = br
        return rec
    rec.update(branch_summary(br, jet["n0_fm3"]))
    rec["causal_stellar_branch"] = bool(br["max_cs2_stellar_branch"] is not None and br["max_cs2_stellar_branch"] < 1.0)
    rec["Mmax_reached"] = not br["M_max_on_grid_end"]
    xc = rec["x_c_Mmax"]
    mb = rec["metastable_branch_above_crust_x"]
    rec["metastable_on_stellar_branch"] = bool(mb is not None and xc is not None and mb[0] <= xc)
    rec["strict_valid"] = bool(rec["causal_stellar_branch"] and rec["Mmax_reached"] and not rec["metastable_on_stellar_branch"])
    rec["heldout"] = heldout_compare(br)
    rec["joint2"] = rec["heldout"]["joint2"]; rec["joint1"] = rec["heldout"]["joint1"]
    rec["fails2"] = rec["heldout"]["fails2"]; rec["fails1"] = rec["heldout"]["fails1"]
    rec["status"] = "ok"
    rec["wall_s"] = time.time() - t0
    return rec


def evaluate_sample(params, settings=None, forms=("power", "pade"), scenarios=("A", "B"), allow_sextic=True, label="",
                    author_conv_extra=True):
    """params: dict n0, EA, K, ystar, Q0, J, L.  Returns scalar specs and per (scalar variant, form, scenario)
    EOS records; author_conv_extra adds the power/A EOS with the author's P_rho = eps_rho convention."""
    p = params
    out = {"id": label, "params": {k: p[k] for k in WINDOW_ORDER}}
    specs = scalar_spec(p["ystar"], p["n0"], p["EA"], p["K"], p["Q0"], allow_sextic=allow_sextic)
    out["scalar"] = {}
    out["eos"] = {}
    for variant, sp in specs.items():
        out["scalar"][variant] = {"status": sp["status"], "a5": sp.get("a5"), "a6": sp.get("a6")}
        if "jet" in sp:
            out["scalar"][variant]["jet"] = _jet_summary(sp["jet"])
        else:
            for k in ("jet_quartic", "jet_quintic"):
                if k in sp and isinstance(sp[k], dict) and sp[k].get("status") == "ok":
                    out["scalar"][variant][k] = _jet_summary(sp[k])
            continue
        for scen in scenarios:
            J = p["J"] if scen == "A" else J_DESIGN
            for form in forms:
                key = f"{variant}|{form}|{scen}"
                out["eos"][key] = evaluate_eos(sp["jet"], J, p["L"], form, settings, label=key)
        if author_conv_extra:
            key = f"{variant}|power|A|authorP"
            s2 = dict(settings or {})
            s2["rearrangement"] = False
            out["eos"][key] = evaluate_eos(sp["jet"], p["J"], p["L"], "power", s2, label=key)
    return out


# ----------------------------------------------------------------------------- single-point evaluation (corner / author points / controls)
def scalar_model(params, scalar="declared"):
    """scalar='declared': scalar_spec rule; 'quartic': a5 = 0 regardless of stability (author's points);
    'quintic': a5 from Q0; 'sextic_min': minimal a6 completion."""
    p = params
    if scalar == "quartic":
        j = scalar_jet(p["ystar"], p["n0"], p["EA"], p["K"])
        return {"status": "quartic_forced" if j["status"] == "ok" else j["status"], "jet": j if j["status"] == "ok" else None,
                "a5": 0.0, "a6": 0.0, "globally_stable": j.get("globally_stable")}
    if scalar == "quintic":
        j = scalar_jet(p["ystar"], p["n0"], p["EA"], p["K"], Q0=p["Q0"])
        return {"status": "quintic_forced" if j["status"] == "ok" else j["status"], "jet": j if j["status"] == "ok" else None,
                "a5": j["coef"][3] if j["status"] == "ok" else None, "a6": 0.0, "globally_stable": j.get("globally_stable")}
    specs = scalar_spec(p["ystar"], p["n0"], p["EA"], p["K"], p.get("Q0"), allow_sextic=True)
    sp = (specs.get("sextic_min") or specs["declared"]) if scalar == "sextic_min" else specs["declared"]
    jet = sp.get("jet")
    return {"status": sp["status"], "jet": jet, "a5": sp.get("a5"), "a6": sp.get("a6"),
            "globally_stable": jet.get("globally_stable") if jet else None,
            "sextic_min_available": "sextic_min" in specs}


def core_table(params, scalar="declared", form="power", settings=None, J=None):
    s = dict(DEFAULT_SETTINGS)
    s.update(settings or {})
    sm = scalar_model(params, scalar)
    if sm["jet"] is None:
        return sm, None, None, s
    iso = isovector(sm["jet"], params["J"] if J is None else J, params["L"], form)
    if iso["status"] != "ok":
        sm["status"] = iso["status"]
        return sm, iso, None, s
    tab = build_beta_table(sm["jet"], iso, x_grid(s), rearrangement=True, global_check=s["global_check"])
    return sm, iso, tab, s


def x_negative_pressure(tab, pressure="consistent"):
    """Highest x < 1 at which the beta-equilibrium pressure crosses zero (linear interpolation on the grid)."""
    P = tab["P_cons"] if pressure == "consistent" else tab["P_author"]
    X = tab["x"]
    xz = None
    for i in range(len(X) - 1):
        if X[i] < 1.0 and P[i] <= 0.0 < P[i + 1]:
            xz = float(X[i] + (X[i + 1] - X[i]) * (-P[i]) / (P[i + 1] - P[i]))
    return xz


def evaluate_point(params, crust="sly", p_surf=None, match="eps", x_m=None, pressure="consistent", scalar="declared",
                   form="power", settings=None, tidal_jump=False, match_interp="spline", p_match=1.5, gamma_crust=1.35,
                   label=""):
    """Full forward evaluation of one parameter point with the chosen crust and pressure convention."""
    t0 = time.time()
    if p_surf is None:
        p_surf = P_SURF_SLY if crust == "sly" else AUTHOR_CRUST["p_surf_MeV_fm3"]
    rec = {"label": label, "params": dict(params), "crust": crust, "match": match if crust == "sly" else "author_polytrope",
           "x_m_requested": x_m, "p_surf": p_surf, "pressure": pressure, "scalar": scalar, "form": form,
           "tidal_jump": bool(tidal_jump), "match_interp": match_interp, "settings_override": dict(settings or {})}
    try:
        sm, iso, tab, s = core_table(params, scalar=scalar, form=form, settings=settings)
    except Exception as exc:  # noqa: BLE001
        rec["status"] = f"beta_table_failed: {exc!r}"
        rec["wall_s"] = time.time() - t0
        return rec
    rec["scalar_status"] = sm["status"]
    rec["a5"] = sm.get("a5"); rec["a6"] = sm.get("a6"); rec["globally_stable"] = sm.get("globally_stable")
    if sm["jet"] is None:
        rec["status"] = f"no_scalar_model:{sm['status']}"
        rec["wall_s"] = time.time() - t0
        return rec
    jet = sm["jet"]
    rec["coef"] = jet["coef"]; rec["Q0_forward"] = jet.get("Q0_forward"); rec["degree"] = jet.get("degree")
    if tab is None:
        rec["status"] = f"isovector_failed:{sm['status']}"
        rec["wall_s"] = time.time() - t0
        return rec
    rec["gamma"] = iso["gamma"]; rec["C0"] = iso["C0"]; rec["L_kin"] = iso["L_kin"]; rec["T"] = iso["T"]
    P = tab["P_cons"] if pressure == "consistent" else tab["P_author"]
    rec["x_neg_grid"] = x_negative_pressure(tab, pressure)
    it = monotone_tail(P)
    rec["x_monotone_from"] = float(tab["x"][it]); rec["P_monotone_from"] = float(P[it])
    pkey = "P_cons" if pressure == "consistent" else "P_author"
    at1 = table_point_exact(tab, jet, 1.0)
    at2 = table_point_exact(tab, jet, 2.0)
    rec["P_n0"] = at1[pkey]; rec["eps_n0"] = at1["eps"]; rec["delta_n0"] = at1["delta"]; rec["y_n0"] = at1["y"]; rec["S_n0"] = at1["S"]
    rec["P_2n0"] = at2[pkey]; rec["S_2n0"] = at2["S"]; rec["delta_2n0"] = at2["delta"]; rec["y_2n0"] = at2["y"]
    rec["P_2n0_nearest_node_legacy"] = at2["nearest_node"]
    rec["global_min_ok_all"] = bool(tab["global_min_ok"].all())
    try:
        eos, info = build_eos(tab, crust=crust, match=match, x_m=x_m, p_match=p_match, gamma=gamma_crust,
                              match_interp=match_interp, pressure=pressure)
    except Exception as exc:  # noqa: BLE001
        rec["status"] = f"crust_match_failed: {exc!r}"
        rec["wall_s"] = time.time() - t0
        return rec
    rec["matching"] = info
    rec["P_match"] = info["P_m"]; rec["x_match"] = info["x_m"]
    rec["eps_jump_rel_at_match"] = getattr(eos, "eps_jump_rel", 0.0)
    rec["density_inversion"] = bool(rec["eps_jump_rel_at_match"] < -1e-12)
    rec["p_max_table"] = eos.p_max
    if crust == "sly":
        n_m = eos.crust.n_of_P(info["P_m"])
        rec["n_SLy_at_P_match_fm3"] = n_m
        rec["x_SLy_at_P_match"] = n_m / params["n0"]
    try:
        br = stellar_branch_for_eos(eos, s, p_surf, tidal_jump=tidal_jump)
    except Exception as exc:  # noqa: BLE001
        rec["status"] = f"tov_failed: {exc!r}"
        rec["wall_s"] = time.time() - t0
        return rec
    if br["status"] != "ok":
        rec["status"] = "no_branch"
        rec["branch_status"] = br
        rec["wall_s"] = time.time() - t0
        return rec
    rec.update(branch_summary(br, params["n0"]))
    rec["causal_stellar_branch"] = bool(rec["max_cs2_stellar_branch"] is not None and rec["max_cs2_stellar_branch"] < 1.0)
    rec["Mmax_reached"] = not br["M_max_on_grid_end"]
    rec["heldout"] = heldout_compare(br)
    rec["joint2"] = rec["heldout"]["joint2"]; rec["joint1"] = rec["heldout"]["joint1"]
    rec["fails2"] = rec["heldout"]["fails2"]; rec["fails1"] = rec["heldout"]["fails1"]
    rec["status"] = "ok"
    rec["wall_s"] = time.time() - t0
    return rec


def evaluate_point_both_conventions(params, scalar, p_surf_author_extra=None, label=""):
    """Both crusts x both pressure conventions for one point.  When the eps-continuous SLy match does not
    exist (author-convention tables are lighter than SLy at every pressure), the declared fallback is
    P-continuity at x_m = 0.6 and 0.8 with the Hinderer jump condition, flagged density_inversion."""
    out = {}
    out["sly|consistent|eps_match|psurf1e-9"] = evaluate_point(params, crust="sly", pressure="consistent", scalar=scalar, label=f"{label}|sly|consistent")
    b = evaluate_point(params, crust="sly", pressure="author", scalar=scalar, label=f"{label}|sly|authorP")
    out["sly|authorP|eps_match|psurf1e-9"] = b
    if b["status"] != "ok":
        for xm in (0.6, 0.8):
            k = f"sly|authorP|Pcont_x{xm}_jump|psurf1e-9"
            out[k] = evaluate_point(params, crust="sly", match="P", x_m=xm, pressure="author", scalar=scalar, tidal_jump=True, label=f"{label}|{k}")
    out["author_polytrope|consistent|psurf1e-9"] = evaluate_point(params, crust="author", pressure="consistent", scalar=scalar, p_surf=1e-9, label=f"{label}|author|consistent|1e-9")
    out["author_polytrope|authorP|psurf1e-9"] = evaluate_point(params, crust="author", pressure="author", scalar=scalar, p_surf=1e-9, label=f"{label}|author|authorP|1e-9")
    out["author_polytrope|consistent|psurf1e-4"] = evaluate_point(params, crust="author", pressure="consistent", scalar=scalar, p_surf=1e-4, label=f"{label}|author|consistent|1e-4")
    out["author_polytrope|authorP|psurf1e-4"] = evaluate_point(params, crust="author", pressure="author", scalar=scalar, p_surf=1e-4, label=f"{label}|author|authorP|1e-4")
    return out


def compact(rec, keys=("label", "status", "scalar_status", "globally_stable", "a5", "a6", "gamma", "C0", "M_max", "R_14", "R_134", "R_1418", "R_144", "R_207",
                       "Lambda_14", "Lambda_tilde_136_q1", "max_cs2_stellar_branch", "causal_stellar_branch", "Mmax_reached", "x_c_Mmax",
                       "x_match", "P_match", "eps_jump_rel_at_match", "density_inversion", "x_neg_grid", "P_2n0", "S_2n0",
                       "joint2", "joint1", "fails2", "fails1", "wall_s")):
    return {k: rec.get(k) for k in keys}


# ----------------------------------------------------------------------------- design
def lhs_params(n, seed, windows=DECLARED_WINDOWS, tag="lhs"):
    from scipy.stats import qmc
    s = qmc.LatinHypercube(d=len(WINDOW_ORDER), seed=seed).random(n)
    lo = [windows[k]["range"][0] for k in WINDOW_ORDER]
    hi = [windows[k]["range"][1] for k in WINDOW_ORDER]
    u = qmc.scale(s, lo, hi)
    return [dict(zip(WINDOW_ORDER, map(float, row))) | {"id": f"{tag}{i:04d}"} for i, row in enumerate(u)]


def grid_params():
    """Coarse declared grid: y* x L x Q0 at the central (n0, E/A, K, J); Q0 is scanned only where the central
    quartic is not globally stable (otherwise a5 = 0 and Q0 is irrelevant)."""
    items = []
    for ys in GRID_Y:
        j4 = scalar_jet(ys, CENTRAL["n0"], CENTRAL["EA"], CENTRAL["K"])
        stable4 = j4["status"] == "ok" and j4["globally_stable"]
        for L in GRID_L:
            for Q0 in ([-200.0] if stable4 else GRID_Q0):
                items.append(dict(CENTRAL, ystar=ys, L=L, Q0=Q0, id=f"grid_y{ys:.2f}_L{L:.0f}_Q{Q0:+.0f}"))
    return items


def fast_grid_params():
    return [dict(CENTRAL, ystar=ys, L=L, Q0=FAST_GRID_Q0, id=f"fast_y{ys:.2f}_L{L:.0f}_Q{FAST_GRID_Q0:+.0f}")
            for ys in FAST_GRID_Y for L in FAST_GRID_L]


def _work(args):
    params, settings = args
    t0 = time.time()
    try:
        r = evaluate_sample(params, settings, label=params["id"])
    except Exception as exc:  # noqa: BLE001
        r = {"id": params["id"], "params": {k: params[k] for k in WINDOW_ORDER}, "status": f"worker_exception: {exc!r}", "scalar": {}, "eos": {}}
    r["wall_s"] = time.time() - t0
    return r


def run_samples(items, settings, workers):
    if workers <= 1:
        rows = [_work((p, settings)) for p in items]
    else:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            rows = list(pool.imap_unordered(_work, [(p, settings) for p in items], chunksize=1))
    rows.sort(key=lambda r: r["id"])
    return rows


# ----------------------------------------------------------------------------- aggregation
OBS = ["M_max", "R_14", "Lambda_14", "Lambda_tilde_136_q1", "R_134", "R_1418", "R_144", "R_207", "P_2n0", "S_2n0", "x_c_Mmax", "gamma", "max_cs2_stellar_branch"]
CRIT = list(HELDOUT_STAR_DATA.keys())


def band_stats(vals):
    v = np.array([x for x in vals if x is not None and isinstance(x, (int, float)) and math.isfinite(x)], float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "min": float(v.min()), "p16": float(np.percentile(v, 16)), "median": float(np.median(v)),
            "p84": float(np.percentile(v, 84)), "max": float(v.max())}


def prange(samples, key):
    v = np.array([s["params"][key] for s in samples], float)
    return [float(v.min()), float(v.max())] if len(v) else None


def analyse(rows, eos_key):
    variant = eos_key.split("|")[0]
    recs = [(d, d.get("eos", {}).get(eos_key), d.get("scalar", {}).get(variant, {}).get("status")) for d in rows]
    n_total = len(rows)
    scalar_status = {}
    eos_status = {}
    for d, e, st in recs:
        scalar_status[st or "absent"] = scalar_status.get(st or "absent", 0) + 1
        s = (e or {}).get("status", "no_eos") if e is not None else ("no_eos_" + (st or "absent"))
        s = s.split(":")[0]
        eos_status[s] = eos_status.get(s, 0) + 1
    ok = [(d, e) for d, e, st in recs if e and e.get("status") == "ok"]
    valid = [(d, e) for d, e in ok if e.get("causal_stellar_branch") and e.get("Mmax_reached")]
    strict = [(d, e) for d, e in valid if e.get("strict_valid")]
    out = {"eos_key": eos_key, "n_total": n_total, "scalar_status": scalar_status, "eos_status": eos_status,
           "n_ok": len(ok), "n_valid": len(valid), "n_strict_valid": len(strict),
           "n_metastable_on_branch": sum(1 for d, e in ok if e.get("metastable_on_stellar_branch")),
           "n_acausal": sum(1 for d, e in ok if not e.get("causal_stellar_branch")),
           "n_Mmax_not_reached": sum(1 for d, e in ok if not e.get("Mmax_reached")),
           "n_gamma_lt_minus1": sum(1 for d, e in ok if e.get("gamma_lt_minus1")),
           "n_pade_pole_in_range": sum(1 for d, e in ok if e.get("pade_pole_in_range"))}
    out["forward_band_valid"] = {k: band_stats([e.get(k) for d, e in valid]) for k in OBS}
    out["forward_band_strict"] = {k: band_stats([e.get(k) for d, e in strict]) for k in ("M_max", "R_14", "Lambda_14", "R_144")}
    pf = {}
    for c in CRIT:
        p2 = [e["heldout"][c].get("pass2") for d, e in valid]
        p1 = [e["heldout"][c].get("pass1") for d, e in valid]
        appl2 = [x for x in p2 if x is not None]
        pf[c] = {"n_applicable": len(appl2), "pass2_frac_valid": (sum(appl2) / len(appl2)) if appl2 else None,
                 "pass1_frac_valid": (sum(x for x in p1 if x is not None) / len(appl2)) if appl2 else None,
                 "pass2_frac_all": (sum(appl2) / n_total) if n_total else None,
                 "n_not_reached": sum(1 for x in p2 if x is None)}
    j2 = [e["heldout"]["joint2"] for d, e in valid]
    j1 = [e["heldout"]["joint1"] for d, e in valid]
    pf["joint"] = {"joint2_frac_valid": (sum(j2) / len(j2)) if j2 else None, "joint1_frac_valid": (sum(j1) / len(j1)) if j1 else None,
                   "joint2_frac_all": (sum(j2) / n_total) if n_total else None, "n_joint2": int(sum(j2)), "n_joint1": int(sum(j1))}
    jr = [all(e["heldout"][c].get("pass2") for c in CRIT if c not in ("Mmax_J0740_Fonseca2021", "R207_J0740_Salmi2024")) for d, e in valid]
    pf["joint_radii_tidal_only"] = {"frac_valid": (sum(jr) / len(jr)) if jr else None, "n": int(sum(jr))}
    out["pass_fractions"] = pf
    reg = {"_label": "star_selected_post_hoc: parameter ranges of the samples passing each criterion; descriptive, not a fit"}
    for c in CRIT:
        sel = [d for d, e in valid if e["heldout"][c].get("pass2")]
        reg[c] = {"n": len(sel), **{k: prange(sel, k) for k in WINDOW_ORDER}} if sel else {"n": 0}
    sel = [d for d, e in valid if e["heldout"]["joint2"]]
    reg["joint2"] = {"n": len(sel), **{k: prange(sel, k) for k in WINDOW_ORDER}, "ids": [d["id"] for d in sel][:50]} if sel else {"n": 0}
    sel = [d for d, e in valid if e["M_max"] >= 2.01]
    reg["Mmax_ge_2.01"] = {"n": len(sel), **{k: prange(sel, k) for k in WINDOW_ORDER}} if sel else {"n": 0}
    out["star_selected_region"] = reg
    out["valid_param_region"] = {k: prange([d for d, e in valid], k) for k in WINDOW_ORDER} if valid else None
    return out


def sample_compact(d):
    c = {"id": d["id"], "params": d["params"], "scalar": {}, "eos": {}}
    for variant, sc in d.get("scalar", {}).items():
        c["scalar"][variant] = {"status": sc.get("status"), "a5": sc.get("a5"), "a6": sc.get("a6"),
                                "coef": (sc.get("jet") or {}).get("coef"), "Q0_forward": (sc.get("jet") or {}).get("Q0_forward"),
                                "q_min": (sc.get("jet") or {}).get("q_min")}
    for key, e in d.get("eos", {}).items():
        c["eos"][key] = {k: e.get(k) for k in ("status", "gamma", "C0", "M_max", "R_14", "R_134", "R_1418", "R_144", "R_207", "Lambda_14",
                                                "Lambda_tilde_136_q1", "P_2n0", "S_2n0", "P_2n0_rel_bias_nearest_node", "x_c_Mmax",
                                                "max_cs2_stellar_branch", "causal_stellar_branch", "Mmax_reached", "strict_valid",
                                                "metastable_on_stellar_branch", "x_monotone_from", "joint2", "joint1", "fails2", "fails1")}
    if "status" in d:
        c["status"] = d["status"]
    return c


# ----------------------------------------------------------------------------- precision / convergence control (declared rule)
CONTROL_POINTS = [
    {"id": "control_y0.87_quartic", "params": dict(CENTRAL, ystar=0.87), "selected_by": "declared_fixed_ystar",
     "why": "author's forward design point (quartic globally stable); central windows"},
    {"id": "control_y0.60_quintic", "params": dict(CENTRAL, ystar=0.60), "selected_by": "declared_fixed_ystar",
     "why": "lower edge of the declared y* window (quintic from Q0 = -200, the Q0-window centre); central windows"},
]
CONTROL_LADDER = {"base": {}, "rtol1e-10": {"rtol": 1e-10, "atol": 1e-12}, "rtol1e-6": {"rtol": 1e-6, "atol": 1e-8},
                  "nx440": {"n_x": 440}, "npc140": {"n_pc": 140}}


def precision_control(crust):
    out = {"rule": "control points fixed by declared y* values (0.87: author's design point; 0.60: lower edge of the y* window) at the "
                   "central windows, declared scalar rule, power law, scenario A, consistent pressure, band crust; NEVER selected by a "
                   "star output (verifier correction 2)",
           "ladder": CONTROL_LADDER, "points": []}
    for cp in CONTROL_POINTS:
        row = {"id": cp["id"], "params": cp["params"], "selected_by": cp["selected_by"], "why": cp["why"], "runs": {}}
        base = None
        for name, cfg in CONTROL_LADDER.items():
            r = evaluate_point(cp["params"], crust=crust, scalar="declared", settings=cfg, label=f"{cp['id']}|{name}")
            row["runs"][name] = compact(r)
            if name == "base":
                base = r
        deltas = {}
        for name, r in row["runs"].items():
            if name == "base" or base is None or base.get("M_max") is None or r.get("M_max") is None:
                continue
            deltas[name] = {"dM_max": r["M_max"] - base["M_max"],
                            "dR_14": (r["R_14"] - base["R_14"]) if (r.get("R_14") is not None and base.get("R_14") is not None) else None,
                            "dLambda_14_rel": (r["Lambda_14"] / base["Lambda_14"] - 1.0) if (r.get("Lambda_14") and base.get("Lambda_14")) else None,
                            "dR_144": (r["R_144"] - base["R_144"]) if (r.get("R_144") is not None and base.get("R_144") is not None) else None}
        row["deltas_vs_base"] = deltas
        out["points"].append(row)
    worst = {}
    for row in out["points"]:
        for name, dl in row["deltas_vs_base"].items():
            for k, v in dl.items():
                if v is not None:
                    worst[k] = max(worst.get(k, 0.0), abs(v))
    out["worst_abs_delta_over_ladder"] = worst
    out["pass"] = bool(worst and worst.get("dM_max", 1) < 1e-3 and worst.get("dR_14", 1) < 2e-2)
    out["pass_criterion"] = "|dM_max| < 1e-3 Msun and |dR_1.4| < 0.02 km over the rtol/n_x/n_pc ladder (numerical control only, not evidence)"
    return out


# ----------------------------------------------------------------------------- corner and author points
CORNER = {"params": {"ystar": 0.83, "Q0": 55.0, "J": 34.0, "L": 35.0, "n0": 0.17, "EA": -15.5, "K": 260.0},
          "label": "star_selected_post_hoc",
          "note": "Five-edge corner (y*, K, n0, E/A, L at window edges; J at its upper edge) found POST HOC by a targeted scan after "
                  "comparing with the held-out stars.  Its evaluation is a re-evaluation of a star-selected point, NOT a forward prediction, "
                  "and carries no evidence weight; declared scalar rule gives the quintic (a5 from Q0 = +55), power law, scenario A."}
AUTHOR_POINTS = {"ystars": (0.87, 0.88, 0.90, 0.93), "params_common": {"n0": 0.16, "EA": -16.0, "K": 240.0, "J": 32.0, "L": 50.0, "Q0": None},
                 "label": "author_forward_design_points",
                 "note": "The author's four forward points (quartic a5 = 0 by design, verification/nvg_forward_beta_eos_audit.py Y_GRID); "
                         "the tracked artifact verification/nvg_forward_tov_heldout_results.json uses a 24-node table with the author's "
                         "P_rho = eps_rho convention and polytropic crust at p_surf = 1e-4 (variant 'author_polytrope|authorP|psurf1e-4' here, "
                         "on the 220-node table)."}


def corner_and_author_points():
    out = {"_note": "both crusts (author polytrope p_match=1.5 Gamma=1.35; SLy Read+2009 eps-continuity match) x both pressure "
                    "conventions (thermodynamically consistent P_rho = eps_rho (1 + n C'/C); author's P_rho = eps_rho).  "
                    "Pass flags are descriptive; nothing here is fitted."}
    c = dict(CORNER)
    c["evaluations"] = evaluate_point_both_conventions(CORNER["params"], scalar="declared", label="corner")
    # P-continuity conventions for the corner (consistent pressure): Lambda depends on the jump condition, M and R do not
    for xm in (0.6, 0.8):
        for tj in (False, True):
            k = f"sly|consistent|Pcont_x{xm}_{'jump' if tj else 'nojump'}|psurf1e-9"
            c["evaluations"][k] = evaluate_point(CORNER["params"], crust="sly", match="P", x_m=xm, pressure="consistent", scalar="declared",
                                                 tidal_jump=tj, label=f"corner|{k}")
    c["summary"] = {k: compact(v) for k, v in c["evaluations"].items()}
    out["corner"] = c
    ap = dict(AUTHOR_POINTS)
    ap["points"] = {}
    for y in AUTHOR_POINTS["ystars"]:
        p = dict(AUTHOR_POINTS["params_common"], ystar=y)
        ev = evaluate_point_both_conventions(p, scalar="quartic", label=f"author_y{y}")
        ap["points"][f"y{y}"] = {"params": p, "evaluations": ev, "summary": {k: compact(v) for k, v in ev.items()}}
    out["author_points"] = ap
    return out


# ----------------------------------------------------------------------------- JSON helpers
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_solver")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def declared_inputs_block(crust):
    return {
        "windows": DECLARED_WINDOWS,
        "central_point": CENTRAL,
        "scenario_B_J_design_MeV": {"value": J_DESIGN, "source": "verification/nvg_prex_crex_external_consistency_probe.py:128-129 (T_CONTACT 12.644118 + j_bar 11.136016); theory-internal identification replacing the empirical J window"},
        "structural_principles": STRUCTURAL_PRINCIPLES,
        "crust_models": {"author_polytrope": AUTHOR_CRUST, "sly": SLY.describe(), "band_crust_used": crust},
        "constants": {"HBARC": HBARC, "MN": MN, "M_E": M_E, "M_MU": M_MU, "TOV_geometric": CONST, "TOV_const_set_used": DEFAULT_SETTINGS["const"],
                      "MEVFM3_TO_GCC": MEVFM3_TO_GCC, "m_B_g": M_B_G},
        "heldout_star_data": HELDOUT_STAR_DATA,
        "heldout_thresholds": THRESHOLD_NOTE,
        "chain_settings_default": DEFAULT_SETTINGS,
        "lambda_tilde_convention": "Lambda_tilde(1.36) = lambda_tilde(1.36, 1.36, L(1.36), L(1.36)): equal-mass binary q = 1 at chirp mass 1.186",
    }


def design_block(full):
    return {
        "mode": "full_declared_design" if full else "fast_declared_grid",
        "fast_grid": {"rule": "central (n0, E/A, K, J); Q0 = -200 (centre of the declared Q0 window); y* x L grid",
                      "ystar": list(FAST_GRID_Y), "L": list(FAST_GRID_L), "Q0": FAST_GRID_Q0, "n_samples": len(FAST_GRID_Y) * len(FAST_GRID_L)},
        "full_design": {"lhs": {"n": LHS_N, "seed": LHS_SEED, "sampler": "scipy.stats.qmc.LatinHypercube(d=7)", "order": list(WINDOW_ORDER)},
                        "grid": {"ystar": list(GRID_Y), "L": list(GRID_L), "Q0": list(GRID_Q0),
                                 "rule": "Q0 scanned only where the central quartic is not globally stable (else a5 = 0, Q0 = -200 placeholder)"}},
        "eos_keys_per_sample": ["declared|power|A", "declared|pade|A", "declared|power|B", "declared|pade|B", "declared|power|A|authorP",
                                "sextic_min|... (same five, only when the declared rule gives no_stable_model)"],
        "multiplicity": {"scalar_variants": 2, "running_forms": 2, "scenarios": 2, "pressure_conventions": 2,
                         "combinations_counted": 10,
                         "note": "2 scalar variants (declared quartic|quintic rule, minimal sextic) x 2 running forms x 2 scenarios = 8, plus 2 author-pressure variants"},
    }


def build_result(full=False, crust="author", workers=2):
    t0 = time.time()
    settings = dict(DEFAULT_SETTINGS)
    settings["crust"] = crust
    items = (lhs_params(LHS_N, LHS_SEED) + grid_params()) if full else fast_grid_params()
    rows = run_samples(items, settings, workers)
    keys = sorted({k for d in rows for k in d.get("eos", {}).keys()})
    band = {"n_samples": len(rows), "crust": crust, "eos_keys": keys,
            "analysis_by_key": {k: analyse(rows, k) for k in keys},
            "samples": [sample_compact(d) for d in rows]}
    if full:
        lhs_rows = [d for d in rows if d["id"].startswith("lhs")]
        grid_rows = [d for d in rows if d["id"].startswith("grid")]
        band["analysis_lhs_only"] = {k: analyse(lhs_rows, k) for k in keys}
        band["analysis_grid_only"] = {k: analyse(grid_rows, k) for k in sorted({k for d in grid_rows for k in d.get("eos", {}).keys()})}
    cap = corner_and_author_points()
    ctrl = precision_control(crust)
    corr = {
        "1_exact_2n0": "P(2n0), S(2n0), delta(2n0), y(2n0) are evaluated by a direct beta-equilibrium solve EXACTLY at x = n/n0 = 2 (and x = 1), "
                       "with the cubic interpolation of the table and the legacy nearest-node value (x = 2.003 on the 220-node log grid) recorded "
                       "alongside as P_2n0_interp_cubic / P_2n0_nearest_node_legacy; the nearest-node convention biased p(2n0) by +0.3..+0.5%",
        "2_control_points_by_rule": "sensitivity/convergence control points are fixed y* values (0.87, 0.60) at the central windows, declared "
                                    "before any output; no control point is chosen by a star output such as M_max > 1.45",
        "source": "independent verifier audit (sampler_audit_findings 1 and 7 of the independent chain re-check)",
    }
    key_main = "declared|power|A"
    a = band["analysis_by_key"].get(key_main, {})
    interp = {
        "status_RU": "Распространение объявленных ядерных окон через forward-цепочку без подгонки: ни один параметр не выбирается по "
                     "наблюдаемой, с которой затем сравнивается; звёздные данные входят только в heldout_compare после решения TOV.",
        "band_RU": (f"Ключ {key_main}: образцов {a.get('n_total')}, валидных (причинная ветвь, M_max внутри таблицы) {a.get('n_valid')}, "
                    f"совместное прохождение 2σ: {a.get('pass_fractions', {}).get('joint', {}).get('n_joint2')} из {a.get('n_valid')}; "
                    f"полоса M_max (валидные): {a.get('forward_band_valid', {}).get('M_max')}.") if a else "нет данных",
        "corner_RU": "Угловая точка (y*=0.83, Q0=+55, K=260, n0=0.17, E/A=-15.5, L=35, J=34) — звёздно-отобранная (post hoc), помечена "
                     "star_selected_post_hoc; её прохождение 2σ при корке SLy (eps-непрерывная сшивка) и провал 1σ по R(1.44) — описание, не свидетельство.",
        "author_points_RU": "Четыре авторские точки (квартика, a5=0): при согласованном давлении и корке SLy M_max = 1.79/1.69/1.45/1.00 "
                            "для y* = 0.87/0.88/0.90/0.93 — ни одна не достигает 2.01 M_sun; авторская конвенция P_rho = eps_rho с коркой "
                            "SLy не допускает eps-непрерывной сшивки (материя легче SLy при любом давлении), резерв — P-сшивка с инверсией плотности.",
        "multiplicity_RU": "Множественность классов посчитана: 2 скалярных варианта × 2 бега × 2 сценария + 2 варианта авторского давления = 10 ключей EOS на образец; "
                           "корка — 2 модели (политропа автора, SLy) × конвенции сшивки.",
        "evidence_RU": "Вес свидетельства 0.0: forward-следствие объявленных окон; флаги pass/fail описательные; статус NOT_ESTABLISHED для любого "
                       "утверждения о согласии с нейтронными звёздами, если оно опирается на звёздно-отобранную точку или на выбор корки.",
    }
    res = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "mode": {"full": bool(full), "crust": crust, "workers": int(workers),
                 "command": "python verification/nvg_forward_band_propagation_audit.py" + (" --full" if full else "") + f" --crust {crust}"},
        "declared_inputs": declared_inputs_block(crust),
        "design": design_block(full),
        "band": band,
        "corner_and_author_points": cap,
        "precision_convergence_control": ctrl,
        "verifier_corrections_applied": corr,
        "interpretation": interp,
        "wall_s": time.time() - t0,
    }
    return _clean(res)


def recompute_scalar_block(result):
    """Cheap, deterministic re-derivation of the no-star part (scalar jets of every band sample) for validate_result."""
    out = {}
    for s in result["band"]["samples"]:
        p = s["params"]
        specs = scalar_spec(p["ystar"], p["n0"], p["EA"], p["K"], p["Q0"], allow_sextic=True)
        sc = {}
        for variant, sp in specs.items():
            jet = sp.get("jet") or {}
            sc[variant] = {"status": sp["status"], "a5": sp.get("a5"), "a6": sp.get("a6"), "coef": jet.get("coef"),
                           "Q0_forward": jet.get("Q0_forward"), "q_min": jet.get("q_min")}
        out[s["id"]] = _clean(sc)
    return out


def validate_result(result):
    """Fail-closed structural + partial-recompute check (the full band is too heavy to regenerate in a test)."""
    try:
        if result["schema_version"] != SCHEMA or result["status"] != STATUS or result["evidence_weight"] != EVIDENCE_WEIGHT:
            return False
        if result["source_sha256"] != hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest():
            return False
        crust = result["mode"]["crust"]
        if _clean(declared_inputs_block(crust)) != result["declared_inputs"]:
            return False
        if _clean(design_block(result["mode"]["full"])) != result["design"]:
            return False
        fresh = recompute_scalar_block(result)
        for s in result["band"]["samples"]:
            if fresh[s["id"]] != s["scalar"]:
                return False
        if result["corner_and_author_points"]["corner"]["label"] != "star_selected_post_hoc":
            return False
        for pt in result["precision_convergence_control"]["points"]:
            if pt["selected_by"] != "declared_fixed_ystar":
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write", action="store_true", help="write the strict-JSON artifact to verification/")
    p.add_argument("--full", action="store_true", help="declared design (LHS 400 + grid) instead of the fast 24-sample grid")
    p.add_argument("--crust", choices=("author", "sly"), default="author", help="crust used for the band (default: author polytrope, the closure baseline)")
    p.add_argument("--workers", type=int, default=2, help="worker processes for the band samples (results are independent of this)")
    args = p.parse_args(argv)
    res = build_result(full=args.full, crust=args.crust, workers=max(1, args.workers))
    text = json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.write:
        RESULT_PATH.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
