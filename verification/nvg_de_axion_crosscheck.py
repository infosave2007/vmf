#!/usr/bin/env python3
"""
NVG Verification: dark-energy <-> pseudoscalar-sector cross-checks.

Cross-section against the dark-dimension / swampland literature window
(R ~ 0.2-100 um) and an internal sensitivity identity of the topological
axion ansatz. Blocks:

  [A] rho_DE in natural units; Lambda^{1/4} and its length scale
  [B] KK-tower kinematics m = hbar c / R over the literature window
  [C] Compton radii of the NVG pseudoscalars vs the dark-dimension window
      -> geometric identification test (expected: REFUTED)
  [D] Internal identity: d ln m_a / d ln rho_DE = 2/N_e and the inverse anchor
  [E] Registered falsification channels (DESI w-running, KATRIN, haloscopes,
      fifth force)

STATUS UNDER SPEC-v2: DIAGNOSTIC / CONJECTURED.
Block [D] is a mathematical sensitivity of the ansatz f_a = M_Pl/N_e^4
(explicitly flagged as a dimensional ansatz in nvg_axion_mass.py). Both
N_e (via R_H0) and rho_DE descend from the same calibrated local H_0, so
[D] is a reverse-solution structure per SPEC-v2 rejection boundary item 2
and is NOT evidence for or against NVG. Blocks [A]-[C], [E] are unit
conversions and literature anchors.

Inputs: Planck 2018 Omega_DE = 0.685 with H_0 = 70 km/s/Mpc for rho_DE
(literature convention); repo axion anchors imported from nvg_axion_mass.
Pure standard library.
"""

import math

from nvg_axion_mass import calculate_axion_parameters

# --- constants ---
HBAR_C_EV_M = 1.973269804e-7     # eV * m
G_CM3       = 6.67430e-8         # cm^3 g^-1 s^-2
EV_PER_G    = 5.609588603e32     # eV per gram (c^2)
CM3_PER_M3  = 1.0e6
MPC_CM      = 3.0857e24          # cm

H0_LIT      = 70.0               # km/s/Mpc (rho_DE convention)
OMEGA_DE    = 0.685              # Planck 2018

M_OMEGA     = 859.0              # MeV (single QCD anchor)

# theta-seesaw anchors (nvg_neutrino_seesaw.py): C = (alpha_s/4pi)^2 from
# the two-loop QCD anomaly diagram, v_EW = 246.22 GeV, alpha_s(M_Z) = 0.1184
FA_THETA    = 1.07e11            # GeV
MA_REF_EV   = 5.691e-6           # eV at f_a = 1e12 GeV
ALPHA_S_MZ  = 0.1184
C_SEE       = (ALPHA_S_MZ / (4.0 * math.pi))**2
V_EW        = 246.22             # GeV


def rho_de_ev4() -> float:
    """Dark-energy density in eV^4 from H_0 = 70, Omega_DE = 0.685."""
    h0_s = H0_LIT * 1e5 / MPC_CM                     # 1/s
    rho_crit = 3.0 * h0_s**2 / (8.0 * math.pi * G_CM3)   # g/cm^3
    rho_de = OMEGA_DE * rho_crit                     # g/cm^3
    rho_de *= EV_PER_G * CM3_PER_M3                  # eV/m^3
    return rho_de * HBAR_C_EV_M**3                   # eV^4


def main():
    print("=" * 72)
    print(" NVG x DARK-ENERGY CROSS-CHECK  (dark-dimension window; SPEC-v2")
    print(" diagnostic -- CONJECTURED status, not evidence)")
    print("=" * 72)

    # [A] dark-energy scale
    rho4 = rho_de_ev4()
    e_de = rho4**0.25                                  # eV
    l_de = HBAR_C_EV_M / e_de                          # m
    print(f"[A] rho_DE = {rho4:.3e} eV^4 ; Lambda^(1/4) = {e_de*1e3:.3f} meV")
    print(f"    hbar c / Lambda^(1/4) = {l_de*1e6:.1f} um  "
          f"(dark-dimension window: 0.2-100 um)")

    # [B] KK kinematics over the literature window
    print("[B] KK tower kinematics m = hbar c / R:")
    for r_um in (0.2, 1.0, 100.0):
        m_kk = HBAR_C_EV_M / (r_um * 1e-6)
        print(f"    R = {r_um:>6.1f} um -> m = {m_kk:.4e} eV")
    print("    (window masses 2 meV - 1 eV: meV-eV haloscope/5th-force "
          "territory, not the NVG micro-eV band)")

    # [C] NVG pseudoscalar Compton radii vs the window
    print("[C] Geometric identification test:")
    f_a_top, m_a_ev = calculate_axion_parameters(M_OMEGA)
    r_a_mm = HBAR_C_EV_M / m_a_ev * 1e3
    m_th_ev = MA_REF_EV * 1e12 / FA_THETA
    r_th_mm = HBAR_C_EV_M / m_th_ev * 1e3
    print(f"    topological axion: f_a = {f_a_top:.3e} GeV, "
          f"m_a = {m_a_ev:.3e} eV -> R_a = {r_a_mm:.1f} mm")
    print(f"    theta-seesaw:      f_a = {FA_THETA:.2e} GeV, "
          f"m_th = {m_th_ev*1e6:.1f} ueV -> R_th = {r_th_mm:.2f} mm")
    print(f"    window 0.2-100 um: R_a off by x{r_a_mm * 1e3 / 100.0:.0f}"
          f" (top), R_th off by x{r_th_mm * 1e3 / 1.0:.0f} (1 um anchor)")
    print("    -> geometric identification with the dark dimension: REFUTED")

    # [D] internal sensitivity identity of the ansatz
    print("[D] Internal identity (ansatz sensitivity, NOT evidence):")
    r_h0_km = 1.2709e23
    r_c_km = 1.128 * (859.0 / M_OMEGA)
    n_e = math.log(r_h0_km / r_c_km)
    sens = 2.0 / n_e
    print(f"    m_a ~ N_e^4, N_e = ln(R_H0/r_c) = {n_e:.2f}, "
          f"rho_DE ~ H_0^2, R_H0 ~ H_0^-1")
    print(f"    d ln m_a / d ln rho_DE = -2/N_e = -{sens*100:.2f}% per "
          f"ln-unit (higher rho_DE -> higher H_0 -> smaller m_a)")
    print(f"    inverse anchor: m_a to 1% -> N_e to +/-0.25% (+/-0.13 at "
          f"N_e = {n_e:.1f}) -> H_0 to ~+/-0.25% (+/-0.18 km/s/Mpc at 72.8)")
    print("    CAVEAT: N_e (via R_H0) and rho_DE share the same calibrated")
    print("    H_0 -- reverse-solution structure (SPEC-v2 sec.4 item 2).")

    # [E] falsification channels
    print("[E] Registered falsification channels:")
    m_nu3 = C_SEE * V_EW**2 / FA_THETA * 1e9         # eV
    # KATRIN observable is PMNS-weighted, NOT m_3/sqrt(3) (AUDIT FIX 2026-08,
    # consistent with nvg_neutrinoless_dbeta.py): m1 -> 0, normal ordering.
    dm2_21, dm2_31 = 7.53e-5, 2.45e-3                # eV^2 (NuFIT 5.2, NO)
    s12_2, s13_2 = 0.304, 0.0222
    m2 = math.sqrt(dm2_21)
    m3 = math.sqrt(dm2_21 + dm2_31)
    m_beta = math.sqrt(s12_2 * (1 - s13_2) * m2**2 + s13_2 * m3**2)
    print(f"    DESI: NVG predicts w = -1 exactly; DR2 dynamical-DE pull")
    print(f"          (3.1sigma primary) is the sharpest live falsifier.")
    print(f"    KATRIN: m_beta = {m_beta*1e3:.1f} meV (seesaw, NO, PMNS-weighted) --")
    print(f"            below KATRIN's ~200 meV design reach; a discovery")
    print(f"            m_beta > 20 meV excludes the minimal theta-seesaw")
    print(f"            (two-sided test belongs to Project 8, sigma ~ 10 meV);")
    print(f"            RH-neutrino signals at O(100) meV also exclude NO.")
    print(f"    Haloscopes: NVG band 8-53 ueV (ADMX/HAYSTAC territory) is")
    print(f"          disjoint from the KK meV-eV band -> band detection")
    print(f"          discriminates NVG from dark-dimension DM.")
    print(f"    Fifth force: Eot-Wash excludes scalar forces to ~52 um;")
    print(f"          the theta monopole channel is identically zero.")
    print("=" * 72)
    print("Terminal status: DIAGNOSTIC / CONJECTURED under SPEC-v2.")
    print("=" * 72)


if __name__ == "__main__":
    main()
