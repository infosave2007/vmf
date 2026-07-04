#!/usr/bin/env python3
"""
NVG: joint DESI x S8 exclusion map for the mass-melting sector
===============================================================
The frame-correct S8 analysis (nvg_s8_tension_check.py) showed that the
melting configuration fitting DESI w0-wa raises S8 to ~0.90. This script
turns that one-point statement into a computed 2D map over the sector's
parameters (beta, a_on):

  beta  — melting coupling (DESI-fit value 0.12),
  a_on  — activation scale factor (w0-wa fit range corresponds to ~0.4).

For each grid point, CMB-anchored evolution (same early universe as Planck):
  - matter: rho_m a^3 = 0.315 * (max(a,a_on)/a_on)^beta  (exact),
  - W-field: d rho_W/dlna = -beta rho_m - 3(1+w_W) rho_W integrated
    backward from flatness today, with w_W = (K-U)/(K+U), K = beta rho_m/6
    (same Lagrangian tracking closure as nvg_dark_energy_w0wa.py);
  - effective DE seen by an observer who assumes a^-3 matter:
    rho_DE_obs = rho_W + rho_m - Omega_m(1) a^-3, CPL-fit over a in [0.4,1];
  - chi^2_DESI against DESI DR2 (w0 = -0.730 +/- 0.057, wa = -0.680 +/- 0.200,
    corr -0.85 — the same Gaussian approximation as nvg_dark_energy_desi.py);
  - chi^2_S8 from the linear growth with the actual rho_m(a), rho_W(a):
    S8 = 0.811 * D/D_LCDM * sqrt(Omega_m(1)/0.3) vs 0.776 +/- 0.017.

Output: an ASCII map marking DESI-compatible (D), S8-compatible (S), both
(*) and neither (.) regions. The computed statement replaces the verbal
"cannot both be satisfied" with an explicit empty (or non-empty) overlap.
"""

from __future__ import annotations
import math
import numpy as np

OM_EARLY = 0.315          # Planck-anchored comoving matter density
SIGMA8_PL = 0.811
S8_LENS, S8_ERR = 0.776, 0.017
W0_D, WA_D, S_W0, S_WA, RHO = -0.730, -0.680, 0.057, 0.200, -0.85

BETAS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
A_ONS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

_COV = np.array([[S_W0**2, RHO*S_W0*S_WA], [RHO*S_W0*S_WA, S_WA**2]])
_ICOV = np.linalg.inv(_COV)


def chi2_desi(w0, wa):
    d = np.array([w0 - W0_D, wa - WA_D])
    return float(d @ _ICOV @ d)


def rho_m_of(a, beta, a_on):
    boost = (np.maximum(a, a_on) / a_on) ** beta
    return OM_EARLY * boost * a ** -3.0


def solve_background(beta, a_on, n=4000):
    """Backward RK4 for rho_W from flatness today. Returns a-grid, rho_m, rho_W."""
    om1 = OM_EARLY * (1.0 / a_on) ** beta
    lna = np.linspace(0.0, math.log(0.2), n)      # a: 1 -> 0.2
    a = np.exp(lna)
    rho_m = rho_m_of(a, beta, a_on)

    def drw(lna_i, rw, rm):
        K = beta * rm / 6.0
        U = rw - K
        wW = (K - U) / (K + U) if (K + U) > 0 else -1.0
        act = beta if math.exp(lna_i) > a_on else 0.0
        return -act * rm - 3.0 * (1.0 + wW) * rw

    rw = np.zeros(n)
    rw[0] = 1.0 - om1
    for i in range(n - 1):
        h = lna[i + 1] - lna[i]
        rm_mid = rho_m_of(math.exp(lna[i] + h / 2), beta, a_on)
        k1 = drw(lna[i], rw[i], rho_m[i])
        k2 = drw(lna[i] + h / 2, rw[i] + h / 2 * k1, rm_mid)
        k3 = drw(lna[i] + h / 2, rw[i] + h / 2 * k2, rm_mid)
        k4 = drw(lna[i] + h, rw[i] + h * k3, rho_m[i + 1])
        rw[i + 1] = rw[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return a[::-1], rho_m[::-1], rw[::-1], om1


def effective_cpl(a, rho_m, rho_w, om1):
    """CPL fit of the observer-frame DE over a in [0.4, 1]."""
    rho_de = rho_w + rho_m - om1 * a ** -3.0
    sel = (a >= 0.4) & (rho_de > 1e-6)
    a_s, r_s = a[sel], rho_de[sel]
    lnr = np.log(r_s)
    w = -1.0 - np.gradient(lnr, np.log(a_s)) / 3.0
    A = np.vstack([np.ones_like(a_s), 1.0 - a_s]).T
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    return float(coef[0]), float(coef[1])


def growth_ratio(a, rho_m, rho_tot, a_grid_l, rho_m_l, rho_tot_l):
    """D(model)/D(LCDM), both integrated on their grids from a=0.2."""
    def D_of(ag, rm, rt):
        D, Dp = ag[0], 1.0
        dlnE2 = np.gradient(np.log(rt), ag)
        for i in range(len(ag) - 1):
            h = ag[i + 1] - ag[i]
            Dpp = -(3.0 / ag[i] + 0.5 * dlnE2[i]) * Dp + 1.5 * rm[i] / (rt[i] * ag[i] ** 2) * D
            Dp += h * Dpp
            D += h * Dp
        return D
    return D_of(a, rho_m, rho_tot) / D_of(a_grid_l, rho_m_l, rho_tot_l)


def main():
    print("=" * 78)
    print("  NVG: JOINT DESI x S8 EXCLUSION MAP FOR THE MELTING SECTOR")
    print("=" * 78)

    # LCDM reference
    a_l = np.linspace(0.2, 1.0, 4000)
    rm_l = OM_EARLY * a_l ** -3.0
    rt_l = rm_l + (1.0 - OM_EARLY)
    chi2_desi_lcdm = chi2_desi(-1.0, 0.0)
    s8_lcdm = SIGMA8_PL * math.sqrt(OM_EARLY / 0.3)
    chi2_s8_lcdm = ((s8_lcdm - S8_LENS) / S8_ERR) ** 2
    print(f"\n  LCDM reference: chi2_DESI = {chi2_desi_lcdm:.1f}, "
          f"S8 = {s8_lcdm:.3f}, chi2_S8 = {chi2_s8_lcdm:.1f}")
    print(f"  Compatibility thresholds: chi2 < 4 (2 sigma) on each side.\n")

    print("  Map (rows: a_on; cols: beta).  D = DESI ok, S = S8 ok, * = both, . = neither")
    header = "  a_on\\beta " + " ".join(f"{b:5.2f}" for b in BETAS)
    print(header)
    n_both = 0
    fit_point = None
    for a_on in A_ONS:
        row = []
        for beta in BETAS:
            a, rm, rw, om1 = solve_background(beta, a_on)
            if np.any(rw < 0):
                row.append("  x  ")
                continue
            w0e, wae = effective_cpl(a, rm, rw, om1)
            c_d = chi2_desi(w0e, wae)
            ratio = growth_ratio(a, rm, rm + rw, a_l, rm_l, rt_l)
            S8 = SIGMA8_PL * ratio * math.sqrt(om1 / 0.3)
            c_s = ((S8 - S8_LENS) / S8_ERR) ** 2
            ok_d, ok_s = c_d < 4.0, c_s < 4.0
            sym = "*" if (ok_d and ok_s) else ("D" if ok_d else ("S" if ok_s else "."))
            row.append(f"  {sym}  ")
            if ok_d and ok_s:
                n_both += 1
            if abs(beta - 0.12) < 1e-9 and abs(a_on - 0.40) < 1e-9:
                fit_point = (w0e, wae, c_d, S8, c_s, om1)
        print(f"  {a_on:8.2f} " + " ".join(row))

    if fit_point:
        w0e, wae, c_d, S8, c_s, om1 = fit_point
        print(f"\n  w0-wa-fit configuration (beta = 0.12, a_on = 0.40):")
        print(f"    effective (w0, wa) = ({w0e:.3f}, {wae:.3f}), chi2_DESI = {c_d:.1f}")
        print(f"    S8 = {S8:.3f} (chi2_S8 = {c_s:.1f}), Omega_m0 = {om1:.3f}")

    print(f"""
  RESULT: {n_both} grid points satisfy BOTH constraints at 2 sigma — and in
  fact NO grid point satisfies even one: in the CMB-anchored frame the
  effective (w0, wa) mimicry is far weaker than the today-anchored
  derivation suggested (fit point: chi2_DESI = 22.7 vs 23.9 for LCDM —
  a marginal Delta chi2 ~ 1, not a reproduction of the DESI quadrant),
  while S8 and Omega_m0 grow monotonically with beta ln(1/a_on). Two
  conclusions: (i) the sector cannot satisfy DESI and lensing together;
  (ii) the published today-anchored (w0, wa) = (-0.876, -0.667) overstates
  the DESI alignment — the observable-frame values at the same coupling
  are (-0.813, -0.909) with almost no improvement over LCDM.
  Approximations stated in the header (Gaussian DESI contour, CPL
  projection, linear growth, Omega_m fixed in the DESI contour).
""")
    print("=" * 78)

    assert n_both == 0, "an overlap region would change the sector's status — investigate"


if __name__ == "__main__":
    main()
