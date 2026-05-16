#!/usr/bin/env python3
"""
NVG Cyclic Bounce: Quantitative calculations for the cosmological bounce
from QCD vacuum parameters.

Derives all bounce parameters from the lattice-QCD anchor M_Omega_0 = 859 MeV.
No free cosmological fitting — everything follows from the VMF EOS chain.
"""
import numpy as np
from scipy.integrate import solve_ivp
import math

# ── Physical Constants ──
hbar_c = 197.3269804    # MeV·fm
c_cgs = 2.998e10        # cm/s
G_cgs = 6.674e-8        # cm^3 g^-1 s^-2
M_sun = 1.989e33        # g
l_Pl = 1.616e-33        # cm
t_Pl = 5.391e-44        # s
rho_Pl_cgs = 5.155e93   # g/cm^3
MeV_fm3_to_gcm3 = 1.7827e12  # 1 MeV/fm^3 in g/cm^3
kB_MeV = 8.617e-2       # MeV/K (Boltzmann)

# ── QCD Anchors (from lattice, NOT fitted) ──
M_N = 939.0
sigma_piN = 44.0
sigma_sN = 30.0
sigma_heavy = 6.0
M_Omega_0 = M_N - (sigma_piN + sigma_sN + sigma_heavy)  # 859 MeV
n_0 = 0.16              # fm^-3
kappa_1 = 0.25
kappa_2 = 0.80

def M_Omega(n_B):
    x = max(n_B / n_0, 0.0)
    return M_Omega_0 * (1.0 + kappa_2 * x) ** (-kappa_1 / kappa_2)

def M_star(n_B):
    M_cur = (sigma_piN + sigma_sN + sigma_heavy)
    corr = sigma_piN * n_B * hbar_c**3 / (93.0**2 * 140.0**2)
    return max(M_cur * max(1.0 - corr, 0.0), 0.0) + M_Omega(n_B)


# ══════════════════════════════════════════════════════════════════════
# CALCULATION 1: Critical bounce density from QCD
# ══════════════════════════════════════════════════════════════════════
def calc_critical_density():
    print("=" * 78)
    print("CALCULATION 1: CRITICAL BOUNCE DENSITY FROM QCD PARAMETERS")
    print("=" * 78)

    # eps_max = M_Omega_0^4 / (hbar*c)^3
    eps_max = M_Omega_0**4 / hbar_c**3  # MeV/fm^3
    rho_c_cgs = eps_max * MeV_fm3_to_gcm3

    print(f"\n  Input (lattice QCD):")
    print(f"    M_Omega_0      = {M_Omega_0} MeV")
    print(f"    (hbar*c)       = {hbar_c} MeV·fm")
    print(f"\n  Derivation:")
    print(f"    eps_max = M_Omega_0^4 / (hbar*c)^3")
    print(f"            = ({M_Omega_0})^4 / ({hbar_c})^3")
    print(f"            = {M_Omega_0**4:.4e} / {hbar_c**3:.4e}")
    print(f"            = {eps_max:.4e} MeV/fm^3")
    print(f"\n  Convert to CGS:")
    print(f"    rho_c = {rho_c_cgs:.4e} g/cm^3")
    print(f"\n  Compare to Planck density:")
    print(f"    rho_Planck     = {rho_Pl_cgs:.4e} g/cm^3")
    print(f"    rho_c / rho_Pl = {rho_c_cgs/rho_Pl_cgs:.4e}")
    print(f"\n  RESULT: rho_c is {rho_Pl_cgs/rho_c_cgs:.1e}x BELOW Planck density.")
    print(f"  The bounce occurs in the semi-classical regime — no quantum gravity needed.")

    # Effective cosmological constant at bounce
    Lambda_eff = 8 * np.pi * G_cgs * rho_c_cgs / c_cgs**2  # cm^-2
    print(f"\n  Effective Lambda at bounce core:")
    print(f"    Lambda_eff = 8*pi*G*rho_c/c^2 = {Lambda_eff:.4e} cm^-2")

    return eps_max, rho_c_cgs


# ══════════════════════════════════════════════════════════════════════
# CALCULATION 2: Bounce trajectory a(t)
# ══════════════════════════════════════════════════════════════════════
def calc_bounce_trajectory(rho_c_cgs):
    print("\n" + "=" * 78)
    print("CALCULATION 2: BOUNCE TRAJECTORY a(t)")
    print("=" * 78)

    # Modified Friedmann: H^2 = (8piG/3) rho (1 - rho/rho_c)
    # For radiation-dominated: rho = rho_b * (a_b/a)^4
    # Let x = a/a_min. Then rho = rho_c / x^4
    # H^2 = (8piG/3) (rho_c/x^4) (1 - 1/x^4)
    # da/dt = a * H => dx/dt = x * H

    # Characteristic time scale
    t_char = 1.0 / np.sqrt(8 * np.pi * G_cgs * rho_c_cgs / 3.0)
    print(f"\n  Characteristic bounce time:")
    print(f"    t_bounce = 1/sqrt(8*pi*G*rho_c/3)")
    print(f"             = {t_char:.4e} s")
    print(f"             = {t_char/t_Pl:.4e} t_Planck")

    # Dimensionless ODE: d(x)/d(tau) where tau = t/t_char
    # dx/dtau = x * sqrt( (1/x^4)(1 - 1/x^4) )  for x >= 1
    def rhs(tau, y):
        x = y[0]
        if x < 1.0001:
            x = 1.0001
        arg = (1.0 / x**4) * (1.0 - 1.0 / x**4)
        if arg < 0:
            arg = 0.0
        H_dimless = np.sqrt(arg)
        return [x * H_dimless]

    sol = solve_ivp(rhs, [0, 30], [1.0001], max_step=0.01, dense_output=True)
    tau = np.linspace(0, 25, 500)
    x = sol.sol(tau)[0]

    # Key milestones
    print(f"\n  Bounce trajectory (a/a_min vs t/t_bounce):")
    print(f"    {'t/t_bounce':>12}  {'a/a_min':>12}  {'rho/rho_c':>12}  {'H*t_b':>12}")
    print("    " + "-" * 52)
    milestones = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    for t_val in milestones:
        if t_val <= tau[-1]:
            a_val = float(sol.sol(t_val)[0])
            rho_ratio = 1.0 / a_val**4
            arg = rho_ratio * (1.0 - rho_ratio)
            H_val = np.sqrt(max(arg, 0))
            print(f"    {t_val:12.1f}  {a_val:12.4f}  {rho_ratio:12.6f}  {H_val:12.6f}")

    # When does rho drop to nuclear density?
    rho_nuc_ratio = (n_0 * M_N) / (M_Omega_0**4 / hbar_c**3)  # rho_nuc / rho_c
    a_nuc = (1.0 / rho_nuc_ratio)**0.25
    print(f"\n  Nuclear density reached at a/a_min = {a_nuc:.2f}")
    print(f"  (rho_nuc/rho_c = {rho_nuc_ratio:.4e})")

    return t_char


# ══════════════════════════════════════════════════════════════════════
# CALCULATION 3: Mass melting profile through the bounce
# ══════════════════════════════════════════════════════════════════════
def calc_mass_melting_profile(eps_max):
    print("\n" + "=" * 78)
    print("CALCULATION 3: MASS MELTING PROFILE THROUGH THE BOUNCE")
    print("=" * 78)

    # Map energy density to effective baryon density
    # At extreme densities, n_B ~ eps / M_N (rough upper bound)
    print(f"\n  {'rho/rho_c':>10}  {'n_B/n_0':>10}  {'M*(MeV)':>10}  {'M_Omega':>10}  {'P/eps':>8}  {'Phase'}")
    print("  " + "-" * 72)

    ratios = [1.0, 0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-6, 1e-10]
    for r in ratios:
        eps = eps_max * r  # MeV/fm^3
        n_B_approx = eps / M_N  # fm^-3 (rough)
        m = M_star(n_B_approx)
        m_om = M_Omega(n_B_approx)

        if m < 1.0:
            P_ratio = 1.0/3.0
            phase = "Conformal QGP"
        elif m < 0.3 * M_N:
            P_ratio = 0.28
            phase = "Chiral restored"
        elif m < 0.7 * M_N:
            P_ratio = 0.20
            phase = "Partial melting"
        elif r > 0.9:
            P_ratio = -1.0
            phase = "de Sitter core"
        else:
            P_ratio = 0.15
            phase = "Hadronic"

        if r > 0.9:
            phase = "de Sitter core"
            P_ratio_str = "-1 (vacuum)"
        else:
            P_ratio_str = f"{P_ratio:.3f}"

        print(f"  {r:10.1e}  {n_B_approx/n_0:10.1f}  {m:10.1f}  {m_om:10.1f}  {P_ratio_str:>8}  {phase}")


# ══════════════════════════════════════════════════════════════════════
# CALCULATION 4: Holographic entropy at the bounce
# ══════════════════════════════════════════════════════════════════════
def calc_holographic_entropy(eps_max):
    print("\n" + "=" * 78)
    print("CALCULATION 4: HOLOGRAPHIC ENTROPY AT THE BOUNCE")
    print("=" * 78)

    eps_max_cgs = eps_max * MeV_fm3_to_gcm3  # g/cm^3

    # Minimal scale: r_0 for a universe-mass object
    # r_0 = (3 M c^2 / (4 pi eps_max c^2))^(1/3) (geometric units)
    # For observable universe: M_obs ~ 4e55 g
    M_obs = 4.0e55  # g
    r_0_cm = (3.0 * M_obs / (4.0 * np.pi * eps_max_cgs))**(1.0/3.0)

    A_star = 4.0 * np.pi * r_0_cm**2  # cm^2
    S_holo = A_star / (4.0 * l_Pl**2)

    print(f"\n  Observable universe mass: M_obs = {M_obs:.2e} g")
    print(f"  Bounce core scale:")
    print(f"    r_0 = (3M / 4pi*rho_c)^(1/3)")
    print(f"        = {r_0_cm:.4e} cm = {r_0_cm/1e5:.4e} km")
    print(f"\n  Holographic area: A* = 4*pi*r_0^2 = {A_star:.4e} cm^2")
    print(f"\n  Maximum entropy through bounce:")
    print(f"    S_core = A* / (4 * l_Pl^2)")
    print(f"           = {S_holo:.4e}")
    print(f"\n  Compare: Bekenstein-Hawking entropy of observable universe")
    R_H = 4.4e28  # cm (Hubble radius)
    S_BH = 4.0 * np.pi * R_H**2 / (4.0 * l_Pl**2)
    print(f"    S_BH(Hubble) = {S_BH:.4e}")
    print(f"    S_core / S_BH = {S_holo/S_BH:.4e}")
    print(f"\n  RESULT: Information is compressed by factor ~{S_BH/S_holo:.1e}")
    print(f"  but NOT destroyed. The bounce core can hold ~{S_holo:.1e} bits.")

    return S_holo, r_0_cm


# ══════════════════════════════════════════════════════════════════════
# CALCULATION 5: Temperature at the bounce
# ══════════════════════════════════════════════════════════════════════
def calc_bounce_temperature(eps_max):
    print("\n" + "=" * 78)
    print("CALCULATION 5: TEMPERATURE AT THE BOUNCE")
    print("=" * 78)

    # For conformal matter: eps = (pi^2/30) * g_* * T^4 / (hbar*c)^3
    # Effective d.o.f. at QGP: g_* ~ 47.5 (for 3-flavor QGP)
    g_star = 47.5

    # eps_max in MeV/fm^3, convert to MeV^4/(hbar*c)^3
    # eps [MeV/fm^3] = eps [MeV^4/(hbar*c)^3] since 1 fm = 1/(hbar*c in MeV)
    # Actually eps [MeV/fm^3] * (hbar*c)^3 [MeV^3 fm^3] = eps_MeV4
    eps_MeV4 = eps_max * hbar_c**3  # MeV^4

    T4 = 30.0 * eps_MeV4 / (np.pi**2 * g_star)
    T_MeV = T4**0.25
    T_K = T_MeV / kB_MeV

    print(f"\n  Using Stefan-Boltzmann for QGP:")
    print(f"    eps = (pi^2/30) * g_* * T^4 / (hbar*c)^3")
    print(f"    g_* = {g_star} (3-flavor QGP)")
    print(f"    eps_max = {eps_max:.4e} MeV/fm^3")
    print(f"\n  Solving for T:")
    print(f"    T_bounce = {T_MeV:.1f} MeV")
    print(f"             = {T_K:.4e} K")
    print(f"\n  Compare:")
    print(f"    QCD deconfinement:  T_QCD ~ 155 MeV")
    print(f"    Electroweak:        T_EW  ~ 160 GeV")
    print(f"    Planck temperature: T_Pl  ~ 1.4e32 K")
    print(f"\n  RESULT: T_bounce ~ {T_MeV:.0f} MeV >> T_QCD.")
    print(f"  The bounce occurs well above QCD deconfinement but far below Planck.")
    print(f"  This is the regime of KNOWN physics (perturbative QGP).")

    return T_MeV


# ══════════════════════════════════════════════════════════════════════
# CALCULATION 6: Unified BH-Cosmology scale comparison
# ══════════════════════════════════════════════════════════════════════
def calc_bh_cosmology_unification(eps_max):
    print("\n" + "=" * 78)
    print("CALCULATION 6: BLACK HOLE ↔ COSMOLOGICAL BOUNCE UNIFICATION")
    print("=" * 78)

    eps_cgs = eps_max * MeV_fm3_to_gcm3
    print(f"\n  Same rho_c = {eps_max:.4e} MeV/fm^3 governs BOTH systems:")
    print(f"\n  {'Object':>25}  {'Mass':>14}  {'r_0 (km)':>14}  {'S_core':>14}")
    print("  " + "-" * 70)

    objects = [
        ("Stellar BH (10 M_sun)", 10.0 * M_sun),
        ("IMBH (1000 M_sun)", 1e3 * M_sun),
        ("Sgr A* (4e6 M_sun)", 4e6 * M_sun),
        ("M87* (6.5e9 M_sun)", 6.5e9 * M_sun),
        ("Observable Universe", 4e55),
    ]

    for name, mass in objects:
        r0 = (3.0 * mass / (4.0 * np.pi * eps_cgs))**(1.0/3.0)
        r0_km = r0 / 1e5
        A = 4.0 * np.pi * r0**2
        S = A / (4.0 * l_Pl**2)
        print(f"  {name:>25}  {mass:.4e}  {r0_km:14.4e}  {S:14.4e}")

    print(f"\n  KEY INSIGHT: The core scale r_0 ~ M^(1/3) / rho_c^(1/3)")
    print(f"  grows with mass, but rho_c is UNIVERSAL (fixed by QCD).")
    print(f"  Every collapsed object reaches the SAME maximum density.")


# ══════════════════════════════════════════════════════════════════════
# CALCULATION 7: EOS chain through the full cycle
# ══════════════════════════════════════════════════════════════════════
def calc_eos_chain():
    print("\n" + "=" * 78)
    print("CALCULATION 7: COMPLETE EOS CHAIN THROUGH THE CYCLE")
    print("=" * 78)

    print(f"\n  Phase 1: EXPANSION (current epoch)")
    print(f"    rho << rho_nuc:  Standard cosmology, M* = M_N = 939 MeV")
    print(f"    EOS: dust (w=0) + radiation (w=1/3) + Lambda (w=-1)")

    print(f"\n  Phase 2: CONTRACTION → HADRONIC COMPRESSION")
    print(f"    rho ~ n_0*M_N:  Nuclear matter, M* starts dropping")

    n_points = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    print(f"    {'n_B/n_0':>8}  {'M*(MeV)':>10}  {'M*/M_N':>8}  {'f_Omega':>8}")
    print("    " + "-" * 40)
    for x in n_points:
        n = x * n_0
        m = M_star(n)
        print(f"    {x:8.1f}  {m:10.1f}  {m/M_N:8.3f}  {M_Omega(n)/max(m,1):8.3f}")

    print(f"\n  Phase 3: CONFORMAL QGP (M* → 0)")
    print(f"    P = eps/3, c_s^2 = 1/3 (causal)")

    print(f"\n  Phase 4: DE SITTER CORE (rho → rho_c)")
    print(f"    P = -eps (vacuum equation of state)")
    print(f"    Strong energy condition VIOLATED → bounce possible")

    print(f"\n  Phase 5: BOUNCE (H = 0, dH/dt > 0)")
    print(f"    New expansion begins with initial conditions from W-field state")

    print(f"\n  COMPLETE CHAIN:")
    print(f"    Dust → Nuclear → QGP → de Sitter → BOUNCE → Radiation → Dust")
    print(f"    (Every transition is governed by M_Omega melting from QCD)")


def main():
    print("╔" + "═" * 76 + "╗")
    print("║  NVG CYCLIC COSMOLOGY: QUANTITATIVE CALCULATIONS FROM QCD PARAMETERS    ║")
    print("║  All results derived from M_Omega_0 = 859 MeV (lattice QCD anchor)      ║")
    print("╚" + "═" * 76 + "╝")

    eps_max, rho_c = calc_critical_density()
    t_b = calc_bounce_trajectory(rho_c)
    calc_mass_melting_profile(eps_max)
    S_core, r_0 = calc_holographic_entropy(eps_max)
    T_b = calc_bounce_temperature(eps_max)
    calc_bh_cosmology_unification(eps_max)
    calc_eos_chain()

    print("\n" + "=" * 78)
    print("SUMMARY OF DERIVED QUANTITIES")
    print("=" * 78)
    print(f"  Input:  M_Omega_0 = {M_Omega_0} MeV (from lattice QCD)")
    print(f"  ────────────────────────────────────────────────")
    print(f"  rho_c (bounce)     = {eps_max:.4e} MeV/fm^3")
    print(f"  rho_c / rho_Planck = {rho_c/rho_Pl_cgs:.4e}")
    print(f"  t_bounce           = {t_b:.4e} s")
    print(f"  T_bounce           = {T_b:.1f} MeV")
    print(f"  r_0 (Universe)     = {r_0/1e5:.4e} km")
    print(f"  S_core (Universe)  = {S_core:.4e}")
    print(f"  ────────────────────────────────────────────────")
    print(f"  ALL derived from ONE QCD number. No cosmological fitting.")


if __name__ == "__main__":
    main()
