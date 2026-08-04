#!/usr/bin/env python3
"""
NVG Verification: Closed-form identities of the single QCD anchor M_Omega.

Derives exact closed forms for the Genesis/bounce scale chain directly from
rho_c = M_Omega^4/(hbar c)^3 and checks them numerically:

  (1) rho_c * r_c^2 = 3 c^2 / (8 pi G)                [exact geometric identity]
  (2) M_1 = (4pi/3) rho_c r_c^3 = c^2 r_c / (2G) = c^3 t_b / (2G)
      -> the mass of the first cycle is LINEAR in the instanton radius.
  (3) T_b = M_Omega * (30/(pi^2 g_*))^{1/4}           [pure dimensionless number]
  (4) M_crit = (9/(8 sqrt(2pi))) M_Pl^3 / M_Omega^2   [repo formula, reproduced]
  (5) M_Pl^3 / M_Omega^2 (bare Chandrasekhar-type scale) vs observed M_max
  (6) holographic entropy S_GH = 4 pi r_c^2 / (4 l_Pl^2)

All inputs: M_Omega = 859 +/- 8 MeV, g_* = 47.5. No free parameters.
"""

import math

# --- constants (SI) ---
C = 2.99792458e8           # m/s
G = 6.67430e-11            # m^3 kg^-1 s^-2
HBAR = 1.054571817e-34     # J s
KB = 1.380649e-23          # J/K
MSUN = 1.98892e30          # kg
MEV_J = 1.602176634e-13    # J per MeV
HC_MEVFM = 197.3269804     # MeV fm
GSTAR = 47.5               # QGP dof at the bounce (repo convention)


def bounce_chain(m_omega_mev: float) -> dict:
    """Full scale chain from a single input M_Omega (MeV)."""
    m_kg = m_omega_mev * MEV_J / C**2

    # critical density rho_c = M_Omega^4/(hbar c)^3
    rho_nat = m_omega_mev**4                    # MeV^4 in natural units
    rho_mevfm3 = rho_nat / HC_MEVFM**3          # MeV/fm^3
    rho_si = rho_mevfm3 * MEV_J / 1e-45 / C**2  # kg/m^3

    # instanton radius / bounce time
    omega_b = math.sqrt(8.0 * math.pi * G * rho_si / 3.0)  # 1/s
    r_c = C / omega_b                                    # m
    t_b = 1.0 / omega_b                                  # s

    # first-cycle mass: M_1 = (4pi/3) rho_c r_c^3
    m1 = 4.0 * math.pi / 3.0 * rho_si * r_c**3

    # bounce temperature (Stefan-Boltzmann of the QGP, natural units)
    t_b_mev = m_omega_mev * (30.0 / (math.pi**2 * GSTAR)) ** 0.25

    # critical horizonless mass (repo closed form)
    m_pl = math.sqrt(HBAR * C / G)
    m_crit = 9.0 / (8.0 * math.sqrt(2.0 * math.pi)) * m_pl**3 / m_kg**2
    m_bare = m_pl**3 / m_kg**2  # bare Chandrasekhar-type scale

    # holographic entropy of the instanton surface
    l_pl = math.sqrt(HBAR * G / C**3)
    s_gh = math.pi * r_c**2 / l_pl**2

    return {
        "rho_mevfm3": rho_mevfm3, "rho_si": rho_si,
        "r_c": r_c, "t_b": t_b, "m1": m1,
        "t_b_mev": t_b_mev, "m_crit": m_crit, "m_bare": m_bare,
        "s_gh": s_gh,
    }


def main():
    print("=" * 72)
    print(" NVG ANCHOR IDENTITIES: CLOSED FORMS FROM M_Omega = 859 +/- 8 MeV")
    print("=" * 72)

    cen = bounce_chain(859.0)
    lo = bounce_chain(851.0)   # M_Omega - 8 MeV
    hi = bounce_chain(867.0)   # M_Omega + 8 MeV

    # (1) geometric identity
    lhs = cen["rho_si"] * cen["r_c"]**2
    rhs = 3.0 * C**2 / (8.0 * math.pi * G)
    print(f"[1] rho_c r_c^2  = {lhs:.6e}  vs  3c^2/8piG = {rhs:.6e}  "
          f"(rel. diff {abs(lhs - rhs) / rhs:.1e})")

    # (2) first-cycle mass, three equivalent forms
    m1_alt1 = C**2 * cen["r_c"] / (2.0 * G)
    m1_alt2 = C**3 * cen["t_b"] / (2.0 * G)
    print(f"[2] M_1 = (4pi/3)rho_c r_c^3 = {cen['m1']/MSUN:.4f} M_sun")
    print(f"       = c^2 r_c/(2G)        = {m1_alt1/MSUN:.4f} M_sun")
    print(f"       = c^3 t_b/(2G)        = {m1_alt2/MSUN:.4f} M_sun   "
          f"(repo claim: 0.38 M_sun)")

    # (3) dimensionless bounce temperature
    ratio = cen["t_b_mev"] / 859.0
    print(f"[3] T_b = {cen['t_b_mev']:.2f} MeV ;  T_b/M_Omega = "
          f"(30/pi^2 g_*)^(1/4) = {ratio:.4f} (repo claim: 432 MeV)")

    # (4)-(5) critical masses
    print(f"[4] M_crit = (9/8sqrt(2pi)) M_Pl^3/M_Omega^2 = "
          f"{cen['m_crit']/MSUN:.4f} M_sun  (repo interval [0.97, 1.01])")
    print(f"[5] M_Pl^3/M_Omega^2 (bare) = {cen['m_bare']/MSUN:.3f} M_sun ; "
          f"M_crit/(M_Pl^3/M_Omega^2) = 9/8sqrt(2pi) = "
          f"{9/(8*math.sqrt(2*math.pi)):.4f}")

    # (6) holographic entropy — honest audit of two possible radii
    print(f"[6] S_GH(instanton) = pi r_c^2/l_Pl^2 = {cen['s_gh']:.3e}")
    r_macro = 1.128e10  # m; radius that reproduces the repo's quoted value
    print(f"    S_GH(R = 1.128e10 m)          = "
          f"{math.pi * r_macro**2 / (HBAR * G / C**3):.3e}  (repo: 2.15e91)")
    print("    NOTE: the repo figure 2.15e91 corresponds to a macroscopic")
    print("    radius ~1.128e10 m (10^7 x r_c); the instanton-area entropy")
    print("    at Genesis itself is 1.53e76. Kept here as a documented audit.")

    # core chain numbers
    print("-" * 72)
    print(f"rho_c   = {cen['rho_mevfm3']:.3e} MeV/fm^3  (repo: 7.09e4)")
    print(f"r_c     = {cen['r_c']/1e3:.4f} km   (repo: 1.13 km)")
    print(f"t_b     = {cen['t_b']:.4e} s    (repo: 3.76e-6 s)")

    # uncertainty propagation from M_Omega +/- 8 MeV
    print("-" * 72)
    print("Uncertainty propagation (M_Omega = 859 +/- 8 MeV):")
    for key, name, scale in [("r_c", "r_c [km]", 1e3),
                             ("t_b", "t_b [us]", 1e-6),
                             ("m1", "M_1 [M_sun]", MSUN),
                             ("m_crit", "M_crit [M_sun]", MSUN),
                             ("t_b_mev", "T_b [MeV]", 1.0)]:
        print(f"  {name:<14s} = {cen[key]/scale:.4f}  "
              f"[{lo[key]/scale:.4f} .. {hi[key]/scale:.4f}]")
    print(f"  fractional anchor error: dM_crit/M_crit = "
          f"{abs(cen['m_crit'] - hi['m_crit'])/cen['m_crit']*100:.1f}% "
          f"(M_crit ~ M_Omega^-2)")

    # verdicts
    print("=" * 72)
    ok1 = abs(lhs - rhs) / rhs < 1e-10
    ok2 = abs(cen["m1"] - m1_alt1) < 1e-9 * cen["m1"]
    print(f"Identity rho_c r_c^2 = 3c^2/8piG     : {'PASS (exact)' if ok1 else 'FAIL'}")
    print(f"Closed form M_1 = c^2 r_c/(2G)       : {'PASS' if ok2 else 'FAIL'}")
    print("All six quantities derive from the single input M_Omega: no free "
          "parameters are introduced by this extension.")
    print("=" * 72)


if __name__ == "__main__":
    main()
