#!/usr/bin/env python3
"""
NVG Verification: the triple-hierarchy identity from rho_c r_c^2 = 3c^2/8piG.

The bounce identity (nvg_anchor_identities.py, PASS at 2e-16) and the
critical-density identity rho_crit R_H0^2 = 3c^2/8piG are the SAME constant,
so the three cyclic-cosmology hierarchies are closed forms of one ratio
R_H0/r_c = exp(N_e):

  [A] density:  rho_c/rho_DE   = exp(2 N_e)/Omega_DE      (~1.85e46)
  [B] entropy:  S_now/S_gen    = exp(2 N_e) = 4^n,  n = N_e/ln2
  [C] mass:     M_U/M_1        = exp(N_e)  = 2^n   (Tolman x2/cycle closure)

Audit of README claim #15 is printed in [D]: the quoted "77.2 cycles" is
log4(2.6e122/1e76) = 77.09 built on rounded E&L entropy anchors, and "+/-0.3"
is the cycle-interval half-width ln2/2, not the QCD sensitivity (+/-0.013).

Block [E] audits the PBH ladder M_N = M_1 x 4^N: the shared spacing is
structural, but the peak rung N = -21 is calibrated to the asteroid DM
window (formation T ~ 642 TeV matches no natural scale) and the geometric-
mean closure hypothesis sqrt(m_Pl x M_1) is refuted by x1.3e6.

STATUS UNDER SPEC-v2: CLOSED IDENTITY WITH A CALIBRATED INPUT. The algebra is
tier I, but N_e uses the calibrated local H_0 = 72.8 and the Tolman x2 law is
a repo derivation, not data. These are consistency closures that RE-EXPRESS the
hierarchies; they do not explain the smallness of rho_DE and are not evidence.
Pure standard library.
"""

import math

# --- constants (SI) ---
C = 2.99792458e8
G = 6.67430e-11
HBAR = 1.054571817e-34
HC_MEVFM = 197.3269804
MEV_J = 1.602176634e-13
MSUN = 1.98892e30
MPC_M = 3.0857e22

M_OMEGA = 859.0        # MeV, single QCD anchor
OMEGA_DE = 0.685       # Planck 2018
H0_CAL = 72.8          # km/s/Mpc, repo calibrated anchor


def main():
    print("=" * 72)
    print(" TRIPLE-HIERARCHY IDENTITY (closed forms; calibrated-H0 input)")
    print("=" * 72)

    # bounce side from M_Omega alone
    rho_mevfm3 = M_OMEGA**4 / HC_MEVFM**3
    rho_c = rho_mevfm3 * MEV_J / 1e-45 / C**2          # kg/m^3
    r_c = C / math.sqrt(8.0 * math.pi * G * rho_c / 3.0)

    # today side from the calibrated H_0
    h0 = H0_CAL * 1e3 / MPC_M
    r_h0 = C / h0
    rho_crit = 3.0 * h0**2 / (8.0 * math.pi * G)
    rho_de = OMEGA_DE * rho_crit
    n_e = math.log(r_h0 / r_c)
    n_cycles = n_e / math.log(2.0)

    # the shared geometric constant
    lhs = rho_c * r_c**2
    rhs = rho_crit * r_h0**2
    geom = 3.0 * C**2 / (8.0 * math.pi * G)
    print(f"[0] rho_c r_c^2 = {lhs:.6e} ; rho_crit R_H0^2 = {rhs:.6e}")
    print(f"    3c^2/8piG   = {geom:.6e}  (rel diffs "
          f"{abs(lhs-geom)/geom:.1e} / {abs(rhs-geom)/geom:.1e})")
    print(f"    N_e = ln(R_H0/r_c) = {n_e:.3f} ; n = N_e/ln2 = {n_cycles:.2f}")

    # [A] density hierarchy
    ratio_a = rho_c / rho_de
    ratio_a_x = math.exp(2.0 * n_e) / OMEGA_DE
    print(f"[A] rho_c/rho_DE = {ratio_a:.4e} = exp(2N_e)/Omega_DE "
          f"(rel diff {abs(ratio_a-ratio_a_x)/ratio_a_x:.1e})")

    # [B] entropy hierarchy
    l_pl = math.sqrt(HBAR * G / C**3)
    s_gen = math.pi * r_c**2 / l_pl**2
    s_now = math.pi * r_h0**2 / l_pl**2
    print(f"[B] S_gen = {s_gen:.3e} ; S_now = {s_now:.3e} ; ratio = "
          f"{s_now/s_gen:.4e} = 4^({n_cycles:.2f}) "
          f"(rel diff {abs(s_now/s_gen-math.exp(2*n_e))/math.exp(2*n_e):.1e})")

    # [C] mass hierarchy (Tolman x2 closure)
    m1 = C**2 * r_c / (2.0 * G)
    m_u = C**2 * r_h0 / (2.0 * G)
    print(f"[C] M_1 = {m1/MSUN:.4f} M_sun ; M_U = c^2 R_H0/(2G) = "
          f"{m_u/MSUN:.4e} M_sun")
    print(f"    M_1 x 2^n = {m1*2.0**n_cycles/MSUN:.4e} M_sun = M_U "
          f"(rel diff {abs(m1*2.0**n_cycles-m_u)/m_u:.1e}); "
          f"Hubble-sphere budgets ~8.5e52 kg")

    # [D] audit of README claim #15
    print("[D] audit of README claim #15 (was: 77.2 cycles, N_e 53.16-53.24,")
    print("    QCD sensitivity +/-0.3):")
    dn = (8.0 / M_OMEGA) / math.log(2.0)
    print(f"    closed form: n = {n_cycles:.2f} +/- {dn:.4f} at H_0 = {H0_CAL}")
    print(f"    integer cycle index 77 brackets N_e in "
          f"[76ln2, 77ln2] = [52.68, 53.38] (nvg_hubble_tension.py)")
    print(f"    '77.2' = log4(2.6e122/1e76) = "
          f"{math.log(2.6e122/1e76)/math.log(4):.2f} (rounded E&L anchors)")
    print(f"    '+/-0.3' = ln2/2 = {0.5*math.log(2):.3f} = cycle-interval "
          f"half-width, NOT the QCD sensitivity (+/-{dn:.3f})")

    # [E] PBH ladder audit: same 4^N spacing, peak NOT derived
    print("[E] PBH ladder M_N = M_1 x 4^N audit:")
    m_pl_msun = math.sqrt(HBAR * C / G) / MSUN
    t_b = r_c / C
    m_peak = 0.38 * 4.0 ** (-21)
    # radiation era: t = 0.301 g_*^-1/2 M_Pl/T^2  ->  T = sqrt(c^4/(G hbar))
    # convention-independent; T in MeV with t in seconds
    g_star = 47.5
    m1_kg = m1                                   # kg: c^2 r_c/(2G)
    t_form = t_b * (m_peak * MSUN) / m1_kg       # horizon mass is linear in t
    t_pl_s = math.sqrt(HBAR * G / C**5)
    m_pl_mev = math.sqrt(HBAR * C**5 / G) / MEV_J
    t_form_mev = math.sqrt(0.301 / math.sqrt(g_star)) * math.sqrt(
        t_pl_s / t_form) * m_pl_mev
    print(f"    rung N = -21 (ledger peak) = {m_peak:.3e} M_sun "
          f"({m_peak*MSUN*1e3:.2e} g); horizon formation at t = "
          f"{t_form:.2e} s, T ~ {t_form_mev:.2e} MeV (g_* = {g_star})")
    print(f"    bounce rung N = 0 at t_b = {t_b:.3e} s (repo T_b = 432 MeV "
          f"convention); ladder spacing x2 in t, x1/2 in T per rung")
    m_peak_kg = m_peak * MSUN
    t_h = HBAR * C**3 / (8.0 * math.pi * G * m_peak_kg * 1.380649e-23)
    tau = 5120.0 * math.pi * G**2 * m_peak_kg**3 / (HBAR * C**4)
    print(f"    peak PBH: T_H = {t_h:.2e} K > T_CMB; Hawking lifetime "
          f"{tau/3.156e7:.1e} yr (t_univ = 1.38e10 yr) -> comfortably stable "
          f"as DM")
    # evaporation boundary: tau(M) = age of the universe, solved directly
    t_univ_s = 1.38e10 * 3.156e7
    m_evap_kg = (t_univ_s * HBAR * C**4 /
                 (5120.0 * math.pi * G**2)) ** (1.0 / 3.0)
    n_evap = math.log(m_evap_kg / m1_kg) / math.log(4.0)
    print(f"    evaporation boundary {m_evap_kg:.2e} kg sits at N = "
          f"{n_evap:.1f} ({4.0**n_evap*m1_kg/MSUN:.2e} M_sun)")
    gm = math.sqrt(m_pl_msun * m1 / MSUN)
    n_star = -3.0 * n_e / (4.0 * math.log(2.0))
    print(f"    REFUTED closure hypothesis: sqrt(m_Pl x M_1) = {gm:.3e} M_sun "
          f"misses the peak by x{m_peak/gm:.2e}; exact-N form -3N_e/(4ln2) = "
          f"{n_star:.1f}")
    print(f"    VERDICT: peak rung N = -21 is CALIBRATED to the asteroid DM "
          f"window (~1.7e20 g), not derived; only the shared 4^N spacing is "
          f"structural")

    print("=" * 72)
    print("Status: closed identity, calibrated-H0 input; consistency closure,")
    print("not evidence under SPEC-v2 (re-expression, not explanation).")
    print("=" * 72)


if __name__ == "__main__":
    main()
