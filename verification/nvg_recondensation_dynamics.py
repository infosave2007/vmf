#!/usr/bin/env python3
"""
NVG Theory: (alpha, beta/H) of the W-condensate recondensation from the action
================================================================================
Derives the phase-transition parameters that nvg_gw_comb_amplitude.py left as
unknowns, using the repo's own action inputs:

  W_0 = 859 MeV, lambda_v = 1.02 (relic-DM inversion; m_W = sqrt(2 lambda) W_0
  = 1.24 GeV -> f_0(1370), consistent), T_c = 157.3 MeV, g_* = 47.5.
  (NOTE: the Seven-Theorems preprint quotes W_0 = 432.2 MeV — a factor-2
  convention discrepancy with the rest of the repo; results below use the
  repo canon 859 and the sensitivity to this is reported.)

Method (standard finite-temperature field theory, one loop):
  V(W,T) = D (T^2 - T_0^2) W^2 - E T W^3 + (lambda/4) W^4        [branch A]
  with the cubic coefficient from the bosons whose masses scale with W
  (rho: 9 dof, omega: 3 dof, g_i = M_Omega_i / W_0 ~ 0.81):
     E = (1/12 pi) sum n_i g_i^3 = 0.17   (x3 strong-coupling band)
  and (D, T_0) FIXED by the repo anchors: degenerate minima at T_c = 157.3
  and vacuum minimum W_0 at T = 0. The O(3) bounce action S_3(T) is computed
  numerically (overshoot/undershoot shooting); nucleation at
  S_3/T = 4 ln(T/H); alpha = Delta V / rho_rad; beta/H = T d(S_3/T)/dT.

  Branch B replaces the quartic by the scale-invariant Coleman-Weinberg form
  natural for a trace-anomaly (dilaton-like) condensate:
  V(W,T) = D_cw T^2 W^2 + (lambda/4) W^4 [ln(W/W_0) - 1/4],
  D_cw from the same particle content. For this shape W = 0 stays metastable
  at all T (no spinodal), producing deep supercooling — the classic
  near-conformal scenario (Creminelli-Nicolis-Rattazzi; Randall-Servant).

Honest caveats: one-loop Landau theory at QCD coupling strengths is
qualitative; lattice QCD finds a crossover for the chiral/deconfinement
transition at mu_B ~ 0 — branch A's weakness is consistent with that. The
discriminating structure (polynomial => thin spinodal window, no signal;
scale-invariant => supercooling, PTA-band signal) is robust beyond one loop
because it depends on the potential's shape near W = 0, not on loop factors.

Runtime: ~1-2 minutes. Output: fig_recondensation_dynamics.png
"""

from __future__ import annotations
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nvg_gw_comb_amplitude import omega_peak, kappa_v

# ── Repo anchors ─────────────────────────────────────────────────────────
W0 = 859.0            # MeV (repo canon; preprint's 432.2 flagged in docstring)
LAM = 1.02            # lambda_v from the relic-DM inversion
TC = 157.3            # MeV
G_STAR = 47.5
M_PL = 1.2209e22      # MeV
RHO_RAD = lambda T: (math.pi ** 2 / 30.0) * G_STAR * T ** 4   # MeV^4

# Cubic and quadratic thermal coefficients from the W-coupled content
G_RHO, N_RHO = 695.3 / W0, 9      # rho meson (M_Omega part / W_0)
G_OMG, N_OMG = 702.6 / W0, 3      # omega meson
G_NUC, N_NUC = 859.0 / W0, 8      # nucleons (fermions: no cubic term)
E_CENTRAL = (N_RHO * G_RHO ** 3 + N_OMG * G_OMG ** 3) / (12.0 * math.pi)
D_CW = (N_RHO * G_RHO ** 2 + N_OMG * G_OMG ** 2
        + 0.5 * N_NUC * G_NUC ** 2) / 24.0


def hubble(T, delta_v=0.0):
    """H in MeV, including vacuum energy during supercooling."""
    rho = RHO_RAD(T) + delta_v
    return math.sqrt(8.0 * math.pi * rho / 3.0) / M_PL


def nucleation_target(T, delta_v=0.0):
    """Gamma ~ T^4 exp(-S3/T) = H^4  =>  S3/T = 4 ln(T/H)."""
    return 4.0 * math.log(T / hubble(T, delta_v))


# ── O(3) bounce action by shooting ───────────────────────────────────────
def bounce_S3(V, dV, W_top, W_min):
    """S_3 for tunneling from W = 0 (V(0) = 0) through a barrier at W_top."""
    m2 = abs(2.0 * (V(W_top * 1.05) - 2 * V(W_top) + V(W_top * 0.95))
             / (0.05 * W_top) ** 2) + 1e-12
    m = math.sqrt(m2)
    dr = 0.02 / m
    n_max = 60000

    def shoot(phi0, collect=False):
        phi, dphi, r = phi0, 0.0, 1e-8 / m
        S = 0.0
        for _ in range(n_max):
            def acc(p, dp, rr):
                return dV(p) - 2.0 * dp / rr

            k1p, k1d = dphi, acc(phi, dphi, r)
            k2p, k2d = dphi + 0.5 * dr * k1d, acc(phi + 0.5 * dr * k1p,
                                                  dphi + 0.5 * dr * k1d, r + 0.5 * dr)
            k3p, k3d = dphi + 0.5 * dr * k2d, acc(phi + 0.5 * dr * k2p,
                                                  dphi + 0.5 * dr * k2d, r + 0.5 * dr)
            k4p, k4d = dphi + dr * k3d, acc(phi + dr * k3p, dphi + dr * k3d, r + dr)
            if collect:
                S += 4.0 * math.pi * r * r * (0.5 * dphi ** 2 + V(phi)) * dr
            phi += dr / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
            dphi += dr / 6.0 * (k1d + 2 * k2d + 2 * k3d + k4d)
            r += dr
            if phi < -0.02 * W_top:
                return +1, S           # overshoot
            if dphi > 0 and phi > 0.02 * W_top:
                return -1, S           # undershoot
            if abs(phi) < 1e-4 * W_top and abs(dphi) * r < 1e-4 * W_top:
                return 0, S            # converged
        return 0, S

    lo, hi = W_top * 1.0001, W_min * 0.9999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        flag, _ = shoot(mid)
        if flag > 0:
            hi = mid
        else:
            lo = mid
    _, S3 = shoot(0.5 * (lo + hi), collect=True)
    return S3


def extrema(V, dV, W_hi):
    """Barrier top and broken minimum of V on (0, W_hi]."""
    Ws = np.linspace(W_hi * 1e-3, W_hi, 4000)
    dVs = np.array([dV(w) for w in Ws])
    sign_flips = np.where(np.diff(np.sign(dVs)) != 0)[0]
    roots = [0.5 * (Ws[i] + Ws[i + 1]) for i in sign_flips]
    if len(roots) < 2:
        return None, None
    return roots[0], roots[-1]        # top, minimum


def analyze_branch(name, V_of_T, W_search, T_grid, delta_v_of_T=None):
    """Scan T: find nucleation point, return (T*, alpha, beta/H) or None."""
    print(f"\n{'─' * 78}\nBranch {name}:")
    prev = None
    results = []
    for T in T_grid:
        V, dV = V_of_T(T)
        top, wmin = extrema(V, dV, W_search)
        if top is None or V(wmin) >= 0:
            continue
        S3 = bounce_S3(V, dV, top, wmin)
        dv = -V(wmin)
        target = nucleation_target(T, dv if delta_v_of_T else 0.0)
        results.append((T, S3 / T, target, dv))
        if prev and prev[1] > prev[2] and S3 / T <= target:  # crossed nucleation
            # interpolate T*, compute beta/H by finite difference
            T1, s1 = prev[0], prev[1]
            T2, s2 = T, S3 / T
            Tstar = T2 + (T1 - T2) * (target - s2) / (s1 - s2 + 1e-30)
            beta_H = abs(Tstar * (s1 - s2) / (T1 - T2))
            alpha = dv / RHO_RAD(Tstar)
            print(f"  NUCLEATION at T* = {Tstar:.2f} MeV: "
                  f"alpha = {alpha:.3g}, beta/H = {beta_H:.3g}")
            return Tstar, alpha, beta_H, results
        prev = (T, S3 / T, target)
    return None, None, None, results


# ── Branch B exit: QCD-anomaly (quark-condensate) tilt ───────────────────
# Linear tilt from the Yukawa coupling of W to light quarks:
#   V_tilt = -b f_on(T) W,  b = 2 g_Q |<qbar q>|,  g_Q = (M_Omega/3)/W_0 = 1/3,
#   <qbar q> = -(272 MeV)^3;  f_on models the chiral-condensate onset below
#   T_chi ~ 132 MeV with sharpness n (lattice: growth over ~10-20% in T).
B_TILT = 2.0 * (1.0 / 3.0) * 272.0 ** 3     # MeV^3, anchored
DV_CW = LAM * W0 ** 4 / 16.0                # depth of the CW true vacuum


def V_of_T_exit(T, b_scale=1.0, T_chi=132.0, n=12):
    a2 = D_CW * T * T
    tilt = B_TILT * b_scale / (1.0 + (T / T_chi) ** n)
    V = lambda w: (a2 * w * w + 0.25 * LAM * w ** 4
                   * (math.log(max(w, 1e-6) / W0) - 0.25) - tilt * w)
    dV = lambda w: (2 * a2 * w + LAM * w ** 3
                    * math.log(max(w, 1e-6) / W0) - tilt)
    return V, dV


def extrema_general(dV, W_hi):
    """(W_false, W_top, W_true) for a tilted potential; None if no barrier."""
    Ws = np.geomspace(W_hi * 1e-6, W_hi, 6000)
    dVs = np.array([dV(w) for w in Ws])
    flips = np.where(np.diff(np.sign(dVs)) != 0)[0]
    roots = [0.5 * (Ws[i] + Ws[i + 1]) for i in flips]
    if dVs[0] > 0:                      # origin is the false minimum
        if len(roots) < 2:
            return None
        return 0.0, roots[0], roots[-1]
    if len(roots) < 3:                  # tilted origin: need min, top, min
        return None
    return roots[0], roots[1], roots[-1]


def bounce_S3_general(V, dV, W_f, W_top, W_true):
    """O(3) bounce from a (possibly displaced) false vacuum W_f."""
    Vs = lambda w: V(w) - V(W_f)
    span = W_top - W_f
    m2 = abs(2.0 * (Vs(W_top + 0.05 * span) - 2 * Vs(W_top)
                    + Vs(W_top - 0.05 * span)) / (0.05 * span) ** 2) + 1e-12
    m = math.sqrt(m2)
    dr = 0.02 / m
    n_max = 50000

    def shoot(phi0, collect=False):
        phi, dphi, r = phi0, 0.0, 1e-8 / m
        S = 0.0
        for _ in range(n_max):
            def acc(p, dp, rr):
                return dV(p) - 2.0 * dp / rr
            k1p, k1d = dphi, acc(phi, dphi, r)
            k2p, k2d = (dphi + 0.5 * dr * k1d,
                        acc(phi + 0.5 * dr * k1p, dphi + 0.5 * dr * k1d,
                            r + 0.5 * dr))
            k3p, k3d = (dphi + 0.5 * dr * k2d,
                        acc(phi + 0.5 * dr * k2p, dphi + 0.5 * dr * k2d,
                            r + 0.5 * dr))
            k4p, k4d = (dphi + dr * k3d,
                        acc(phi + dr * k3p, dphi + dr * k3d, r + dr))
            if collect:
                S += 4.0 * math.pi * r * r * (0.5 * dphi ** 2 + Vs(phi)) * dr
            phi += dr / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
            dphi += dr / 6.0 * (k1d + 2 * k2d + 2 * k3d + k4d)
            r += dr
            if phi < W_f - 0.05 * span:
                return +1, S
            if dphi > 0 and phi > W_f + 0.05 * span:
                return -1, S
            if abs(phi - W_f) < 1e-4 * span and abs(dphi) * r < 1e-4 * span:
                return 0, S
        return 0, S

    lo, hi = W_top + 1e-4 * span, W_true * 0.9999
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        flag, _ = shoot(mid)
        if flag > 0:
            hi = mid
        else:
            lo = mid
    _, S3 = shoot(0.5 * (lo + hi), collect=True)
    return S3


def find_exit(b_scale=1.0, T_chi=132.0, n=12, verbose=True):
    """T*, alpha, beta/H for the tilted dilaton potential."""
    # 1. barrier-disappearance temperature T_spin (bisection, cheap)
    def has_barrier(T):
        _, dV = V_of_T_exit(T, b_scale, T_chi, n)
        return extrema_general(dV, 1.5 * W0) is not None
    lo, hi = 60.0, 210.0
    if not has_barrier(hi):
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if has_barrier(mid):
            hi = mid
        else:
            lo = mid
    T_spin = hi

    # 2. nucleation scan just above T_spin
    prev = None
    for T in T_spin * np.geomspace(1.5, 1.002, 16):
        if T > 209.0:
            continue
        V, dV = V_of_T_exit(T, b_scale, T_chi, n)
        ext = extrema_general(dV, 1.5 * W0)
        if ext is None:
            continue
        W_f, W_top, W_true = ext
        dv = V(W_f) - V(W_true)
        if dv <= 0:
            continue
        S3 = bounce_S3_general(V, dV, W_f, W_top, W_true)
        target = nucleation_target(T, dv)
        if prev and prev[1] > prev[2] and S3 / T <= target:
            T1, s1, _ = prev
            T2, s2 = T, S3 / T
            Tstar = T2 + (T1 - T2) * (target - s2) / (s1 - s2 + 1e-30)
            beta_H = abs(Tstar * (s1 - s2) / (T1 - T2))
            alpha = dv / RHO_RAD(Tstar)
            return Tstar, alpha, beta_H, dv
        prev = (T, S3 / T, target)
    # no crossing: spinodal completion; beta/H from the last slope
    V, dV = V_of_T_exit(T_spin * 1.001, b_scale, T_chi, n)
    ext = extrema_general(dV, 1.5 * W0)
    if ext is None:
        return None
    W_f, W_top, W_true = ext
    dv = V(W_f) - V(W_true)
    beta_H = (abs(prev[0] * (prev[1]) / (prev[0] - T_spin))
              if prev and prev[0] != T_spin else 100.0)
    return T_spin, dv / RHO_RAD(T_spin), min(beta_H, 1e4), dv


def main():
    print("=" * 78)
    print("  NVG: (alpha, beta/H) OF THE RECONDENSATION FROM THE W-FIELD ACTION")
    print("=" * 78)
    print(f"\nInputs: W_0 = {W0} MeV, lambda_v = {LAM}, T_c = {TC} MeV, "
          f"g_* = {G_STAR}")
    print(f"Cubic coefficient from meson content: E = {E_CENTRAL:.3f} "
          f"(rho + omega, g ~ 0.81)")

    summary = {}

    # ── Branch A: the repo's quartic Mexican-hat action ────────────────
    for label, E in (("A (quartic action, E central)", E_CENTRAL),
                     ("A (quartic action, E x3 strong-coupling)", 3 * E_CENTRAL)):
        D = LAM * W0 ** 2 / (2 * TC ** 2) + 2 * E ** 2 / LAM
        T0 = math.sqrt(LAM * W0 ** 2 / (2 * D))
        print(f"\n{label}: D = {D:.2f}, spinodal T_0 = {T0:.2f} MeV "
              f"(window below T_c: {(TC - T0) / TC * 100:.2f}%)")
        print(f"  Transition strength W_c/T_c = 2E/lambda = {2 * E / LAM:.2f}")

        def V_of_T(T, D=D, T0=T0, E=E):
            a2, a3, a4 = D * (T ** 2 - T0 ** 2), E * T, LAM / 4.0
            V = lambda w: a2 * w * w - a3 * w ** 3 + a4 * w ** 4
            dV = lambda w: 2 * a2 * w - 3 * a3 * w * w + 4 * a4 * w ** 3
            return V, dV

        T_grid = T0 + (TC - T0) * np.linspace(0.98, 0.02, 25)
        Tstar, alpha, beta_H, _ = analyze_branch(label, V_of_T, 4 * TC, T_grid)
        if Tstar is None:
            # barrier disappears before nucleation: spinodal completion
            Tstar = T0
            V, dV = V_of_T(T0 * 1.0001)
            _, wmin = extrema(V, dV, 4 * TC)
            alpha = -V(wmin) / RHO_RAD(T0)
            beta_H = TC / (TC - T0)
            print(f"  No nucleation above T_0 -> SPINODAL completion at "
                  f"{T0:.1f} MeV: alpha = {alpha:.3g}, beta/H ~ {beta_H:.0f}")
        om = omega_peak(alpha, beta_H)
        print(f"  => Omega_GW h^2 ~ {om:.1e}")
        summary[label] = (alpha, beta_H, om)

    # ── Branch B: scale-invariant (dilaton/CW) potential ────────────────
    print(f"\nBranch B thermal mass from content: D_cw = {D_CW:.2f}")

    def V_of_T_cw(T):
        a2 = D_CW * T ** 2
        V = lambda w: (a2 * w * w + (LAM / 4.0) * w ** 4
                       * (math.log(max(w, 1e-6) / W0) - 0.25))
        dV = lambda w: (2 * a2 * w + LAM * w ** 3
                        * math.log(max(w, 1e-6) / W0))
        return V, dV

    T_grid = np.geomspace(TC, 2.0, 40)
    Tstar, alpha, beta_H, res = analyze_branch(
        "B (Coleman-Weinberg / dilaton, no exit)", V_of_T_cw, 1.5 * W0, T_grid,
        delta_v_of_T=True)
    if Tstar is None:
        print("  No nucleation in the pure CW branch (over-supercooling) —")
        print("  a graceful exit is REQUIRED, supplied below by the QCD tilt.")

    # ── Branch B with the QCD-anomaly (quark-condensate) exit ───────────
    print(f"\n{'─' * 78}\nBranch B + QCD tilt exit "
          f"(b anchored = 2 g_Q |<qq>| = ({B_TILT ** (1/3.):.0f} MeV)^3):")
    from nvg_gw_comb_amplitude import spectral_shape, f_peak_nHz
    variations = [
        ("central          (b x1, T_chi=132, n=12)", 1.0, 132.0, 12),
        ("weak tilt        (b x0.3)               ", 0.3, 132.0, 12),
        ("strong tilt      (b x3)                 ", 3.0, 132.0, 12),
        ("early onset      (T_chi=150)            ", 1.0, 150.0, 12),
        ("late onset       (T_chi=120)            ", 1.0, 120.0, 12),
        ("gradual onset    (n=6)                  ", 1.0, 132.0, 6),
        ("sharp onset      (n=24)                 ", 1.0, 132.0, 24),
    ]
    om32_band = []
    for label, bs, tchi, nn in variations:
        out = find_exit(bs, tchi, nn)
        if out is None:
            print(f"  {label}: no exit found")
            continue
        Ts, al, bh, dv = out
        T_reh = max(Ts, (30.0 * dv / (math.pi ** 2 * G_STAR)) ** 0.25)
        f_p = f_peak_nHz(T_reh, max(bh, 1.0))
        om_p = omega_peak(min(al, 1e3), max(bh, 1.0))
        om32 = om_p * spectral_shape(32.0 / f_p)
        om32_band.append(om32)
        print(f"  {label}: T* = {Ts:6.1f} MeV, alpha = {al:6.2f}, "
              f"beta/H = {bh:7.1f}")
        print(f"    -> T_reh = {T_reh:.0f} MeV, f_peak = {f_p:7.0f} nHz, "
              f"Omega_peak = {om_p:.1e}, Omega(32 nHz) = {om32:.1e}")
        if label.startswith("central"):
            summary["B (dilaton + QCD exit)"] = (al, bh, om32)
    if om32_band:
        print(f"\n  POINT PREDICTION at 32 nHz: Omega_GW h^2 = "
              f"{min(om32_band):.1e} .. {max(om32_band):.1e}")
        print(f"  NANOGrav 15yr reference:     ~4e-09")
        print(f"  Spectral slope in the PTA band: rising ~f^3 flank of the peak")
        print(f"  (SMBH binaries: Omega ~ f^(2/3) — distinguishable in shape).")

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}\nVERDICT:")
    print("Branch A — the repo's written quartic action — gives a transition")
    print("confined to a thin spinodal sliver below T_c: alpha ~ 1e-4..1e-2,")
    print("beta/H ~ 1e2..1e5, hence Omega_GW far below any PTA sensitivity.")
    print("The NANOGrav-band amplitude of nvg_gw_comb_amplitude.py is NOT")
    print("reproduced by the current action.")
    print()
    print("Branch B — the scale-invariant CW form supercools (no nucleation without")
    print("an exit), and the ANCHORED QCD tilt b = 2 g_Q |<qq>| = (238 MeV)^3 supplies")
    print("one — but its high-T tail triggers the exit at T* = 149-186 MeV, BEFORE")
    print("deep supercooling develops: alpha = 1.3-3.6 and beta/H ~ 540-1230. The")
    print("fast transition puts the peak at 18-42 microHz with Omega_peak ~ 1e-9;")
    print("the PTA band receives only the f^3 tail, Omega(32 nHz) ~ 3e-18..6e-17.")
    print()
    print("CONCLUSION: with (alpha, beta/H) actually DERIVED, neither branch")
    print("reproduces the NANOGrav signal — the amplitude window of")
    print("nvg_gw_comb_amplitude.py required beta/H ~ 1-10, and the action")
    print("delivers ~500+ (quartic: spinodal sliver; dilaton: anchored-tilt")
    print("exit). The surviving falsifiable prediction moves to the microHz")
    print("band: Omega_GW h^2 ~ 1e-9 peaked at ~20-40 microHz — in reach of")
    print("proposed microHz missions (muAres) and the low-frequency edge of")
    print("LISA. Conversely, if PTAs establish the ~30 nHz signal as a")
    print("QCD-scale phase transition, that would DISFAVOR the derived NVG")
    print("parameters — the bounce, per this derivation, does not ring there.")
    print("=" * 78)

    # ── Figure: S3/T vs nucleation target for both branches ────────────
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if res:
        Ts = [r[0] for r in res]
        ax.semilogy(Ts, [r[1] for r in res], 'o-', ms=3,
                    label=r'$S_3/T$ (branch B, dilaton)')
        ax.semilogy(Ts, [r[2] for r in res], '--', color='k',
                    label=r'nucleation target $4\ln(T/H)$')
    ax.set_xlabel('T [MeV]')
    ax.set_ylabel(r'$S_3/T$')
    ax.set_title('Recondensation nucleation: bounce action vs target')
    ax.grid(alpha=0.3, ls='--')
    ax.legend(fontsize=9)
    out = os.path.join(os.path.dirname(__file__),
                       "fig_recondensation_dynamics.png")
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")

    # ── Assertions ──────────────────────────────────────────────────────
    a_quartic = summary["A (quartic action, E central)"][0]
    assert a_quartic < 0.05, "quartic action should give a weak transition"
    om_q = summary["A (quartic action, E central)"][2]
    assert om_q < 1e-11, "quartic-action GW signal should be far below PTA"
    exit_al, exit_bh, exit_om32 = summary["B (dilaton + QCD exit)"]
    assert exit_al > 1.0, "anchored-tilt exit should still be a strong transition"
    assert exit_om32 < 1e-12, "derived PTA-band tail must be negligible"


if __name__ == "__main__":
    main()
