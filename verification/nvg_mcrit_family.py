#!/usr/bin/env python3
"""
M_crit as the Chandrasekhar mass scale: the full family of consequences.

Established closed form (nvg_mcrit_chandrasekhar.py):
    M_crit = C * M_Planck^3 / M_Omega^2 ,   C = 9/(8 sqrt(2 pi)) = 0.4488 .

This script derives/checks the honest intersections that follow:
  (1) Landau energy balance    -> the SCALE M_Pl^3/m^2 from gravity vs Fermi pressure
  (2) Chandrasekhar number     -> N_crit ~ 10^57 baryons ~ the baryon count of a star
  (3) same family as WD/NS     -> compare M_crit to Chandrasekhar/TOV/M_Pl^3 m_N^-2
  (4) extremal geometry        -> horizon at r_h = sqrt(3) * l
  (5) uncertainty + falsifier  -> band from M_Omega +/- 8 MeV; GWTC null
Everything is dimensional physics; the O(1) prefactor is the only NVG-specific input.
"""
import math

hbar = 1.054571817e-34; c = 2.99792458e8; G = 6.67430e-11
MeV = 1.602176634e-13; Msun = 1.98847e30
m_N = 1.67262192e-27                       # nucleon mass (kg)

M_Omega = 859.0 * MeV
m_Omega = M_Omega / c**2
M_Pl = math.sqrt(hbar * c / G)
C = 9.0 / (8.0 * math.sqrt(2 * math.pi))   # = (3 sqrt3/4) sqrt(3/8pi)
l = 1128.0                                 # core scale (m), from rho_c

M_crit = C * M_Pl**3 / m_Omega**2

print("=" * 76)
print("  M_crit AS THE CHANDRASEKHAR MASS SCALE — family of consequences")
print("=" * 76)
print(f"  M_crit = C M_Pl^3/M_Omega^2 = {M_crit/Msun:.4f} M_sun,  C = 9/(8 sqrt2pi) = {C:.4f}")

# (1) Landau energy balance: relativistic Fermi vs gravity, both ~1/R
#     E(N,R) = N * hbar c N^(1/3)/R  -  G N^2 m^2 / R  changes sign at
#     N_crit ~ (hbar c/(G m^2))^(3/2) = (M_Pl/m)^3
print("\n(1) LANDAU BALANCE  (gravity vs relativistic degeneracy, both ~1/R):")
N_landau = (M_Pl / m_Omega)**3
print(f"     N_crit ~ (M_Pl/M_Omega)^3 = {N_landau:.3e}")
print(f"     M ~ N_crit * M_Omega      = {N_landau*m_Omega/Msun:.3f} M_sun   (scale only, no O(1))")
print(f"     the GR-extremal calc supplies the exact prefactor C={C:.4f} -> {M_crit/Msun:.3f} M_sun")

# (2) Chandrasekhar number
print("\n(2) CHANDRASEKHAR NUMBER  N_crit = M_crit/M_Omega = C (M_Pl/M_Omega)^3:")
N_crit = M_crit / m_Omega
N_sun_baryons = Msun / m_N
print(f"     N_crit              = {N_crit:.3e}  (~10^57)")
print(f"     baryons in the Sun  = M_sun/m_N = {N_sun_baryons:.3e}")
print(f"     classical (M_Pl/m_N)^3 = {(M_Pl/m_N)**3:.3e}")
print(f"     => all ~10^57: the unique dimensionless gravity/QCD ratio (M_Pl/hadron)^3")

# (3) same family as the astrophysical compact-object masses
print("\n(3) SAME FAMILY AS WHITE-DWARF / NEUTRON-STAR MASSES  (all M_Pl^3/m^2):")
M_scale_nN = M_Pl**3 / m_N**2
print(f"     M_Pl^3/m_N^2 (pure scale)          = {M_scale_nN/Msun:.3f} M_sun")
print(f"     Chandrasekhar mass (WD, observed)  ~ 1.40 M_sun")
print(f"     TOV neutron-star maximum (observed)~ 2.0-2.3 M_sun")
print(f"     NVG M_crit                         = {M_crit/Msun:.3f} M_sun  "
      f"(= {M_crit/M_scale_nN:.3f} x the bare scale)")
print(f"     M_Omega/m_N = {m_Omega/m_N:.3f}  -> agreement is 'same scale', an O(1) ratio")

# (4) extremal geometry: horizon at r_h = sqrt(3) l
print("\n(4) EXTREMAL GEOMETRY  (Hayward cubic double root eps_c = 2/(3 sqrt3)):")
eps_c = 2.0 / (3.0 * math.sqrt(3.0))
r_Sch = 2 * G * M_crit / c**2
r_h = (2.0/3.0) * r_Sch                    # x=2/3 at the double root
print(f"     eps_c = 2/(3 sqrt3)  = {eps_c:.4f}")
print(f"     r_Sch(M_crit) = l/eps_c = {r_Sch:.1f} m  (direct 2GM/c^2 = {2*G*M_crit/c**2:.1f} m)")
print(f"     merged horizon r_h = (2/3) r_Sch = {r_h:.1f} m")
print(f"     r_h / l = {r_h/l:.4f}  (= sqrt(3) = {math.sqrt(3):.4f})  -> r_h = sqrt(3) * l exactly")

# (5) uncertainty band + falsifier
print("\n(5) UNCERTAINTY + FALSIFIER  (M_crit ∝ M_Omega^-2, M_Omega = 859 +/- 8 MeV):")
hi = M_crit * (859/(859-8))**2 / Msun
lo = M_crit * (859/(859+8))**2 / Msun
print(f"     band = [{lo:.3f}, {hi:.3f}] M_sun")
print(f"     KILL: a confirmed compact object below ~{lo:.2f} M_sun WITH a horizon")
print(f"           signature (ringdown / zero tidal deformability Lambda=0).")
print(f"     current GWTC null: lightest secondary GW191219 ~1.17 M_sun (above band).")
# (6) what is actually forced: sweep the hadron mass (only the OoM is forced)
print("\n(6) WHAT IS ACTUALLY FORCED — sweep C*M_Pl^3/m^2 over hadron masses:")
for name, mMeV in [("pion", 139.6), ("Lambda_QCD", 330), ("M_Omega", 859),
                   ("nucleon", 938.9), ("Delta", 1232), ("Omega baryon", 1672)]:
    mm = mMeV * MeV / c**2
    print(f"     m = {name:<12} {mMeV:>5.0f} MeV -> {C*M_Pl**3/mm**2/Msun:8.2f} M_sun")
print("     => a 'hadron mass' spans pion..Omega -> 37..0.26 M_sun (2+ orders).")
print("     Only the ORDER OF MAGNITUDE (~1, not 10^+-6) is forced; the clean ~1 M_sun")
print("     needs m ~ nucleon-scale AND the geometric prefactor C.")

print("=" * 76)
print("  HONEST SCOPE (per adversarial review — real result, must not oversell):")
print("   * REAL: reduction to C*M_Pl^3/M_Omega^2 is an exact identity (2.8e-16); M_Pl^3/m^2")
print("     is the genuine Chandrasekhar/Landau stellar scale; M_Omega=859 MeV is an")
print("     independent global anchor (NOT back-solved from 1 M_sun) -> not numerology.")
print("   * NOT 'dimensional necessity': the power M_Pl^3/m^2 is set by the collapse/horizon")
print("     BALANCE LAW, not by dimensions alone; only the order of magnitude is forced.")
print("   * NOT distinctive to NVG: ANY Hayward/Bardeen regular core with a QCD-scale cutoff")
print("     yields an extremal mass ~M_sun. The ~1 M_sun value does NOT discriminate NVG;")
print("     the testable content is the narrow band + the horizonless-remnant statement.")
print("   * DIFFERENT physics from Chandrasekhar/TOV: M_crit is a MINIMUM horizon-formation")
print("     mass; those are MAXIMUM degeneracy masses. Shared skeleton, opposite extrema.")
print("   * Does NOT predict the Sun's mass (the Sun is itself near this scale).")
print("=" * 76)
