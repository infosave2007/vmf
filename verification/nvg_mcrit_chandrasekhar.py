#!/usr/bin/env python3
"""
Why the NVG Critical Horizon Mass is ~1 solar mass: it is the Chandrasekhar scale.

The extremal (horizon-forming) mass of the QCD-anchored Hayward core is

    M_crit = (3*sqrt(3)/4) * l * c^2 / G ,   l = sqrt(3 c^2 / (8 pi G rho_c)) ,
    rho_c(mass) = M_Omega^4 / ((hbar c)^3 c^2) = m_Omega^4 c^3 / hbar^3 .

Substituting l and simplifying, ALL the machine-specific factors collapse and M_crit
reduces to the pure Chandrasekhar-type combination (Planck-mass cubed over a hadron
mass squared):

    M_crit = C * M_Planck^3 / m_Omega^2 ,
    C = (3*sqrt(3)/4) * sqrt(3/(8*pi)) = (9 / (8*sqrt(2*pi)))  ~ 0.4488  (dimensionless).

This is the SAME  M_Planck^3 / m_nucleon^2  scale (~1.85 Msun) that sets the
Chandrasekhar mass, the white-dwarf and neutron-star maxima. Because the QCD condensate
scale M_Omega = 859 MeV is a hadron mass (~ the nucleon), M_crit lands at ~1 Msun by
DIMENSIONAL NECESSITY, not by tuning. The number is de-mystified: its proximity to a
solar mass is the generic reason every self-gravitating compact object sits near ~1 Msun.

This script (1) computes M_crit directly, (2) computes the closed form C*M_Planck^3/m_Omega^2
and confirms they agree to machine precision, and (3) compares to the standard stellar scale.
"""
import math

# ---- constants (SI) ----
hbar = 1.054571817e-34
c    = 2.99792458e8
G    = 6.67430e-11
MeV  = 1.602176634e-13          # J
Msun = 1.98847e30
m_N  = 1.67262192e-27           # nucleon mass (kg)

M_Omega_energy = 859.0 * MeV
m_Omega = M_Omega_energy / c**2                  # QCD condensate scale AS A MASS
M_Planck = math.sqrt(hbar * c / G)

# ---- (1) direct computation (the repo's definition) ----
rho_c_energy = M_Omega_energy**4 / (hbar * c)**3          # J/m^3
rho_c_mass   = rho_c_energy / c**2                        # kg/m^3
l            = math.sqrt(3 * c**2 / (8 * math.pi * G * rho_c_mass))
M_crit_direct = (3 * math.sqrt(3) / 4) * l * c**2 / G

# ---- (2) closed form: C * M_Planck^3 / m_Omega^2 ----
C = (3 * math.sqrt(3) / 4) * math.sqrt(3 / (8 * math.pi))   # = 9/(8 sqrt(2 pi))
M_crit_closed = C * M_Planck**3 / m_Omega**2

# ---- (3) the standard stellar / Chandrasekhar scale ----
M_stellar_scale = M_Planck**3 / m_N**2                     # ~1.85 Msun

print("=" * 74)
print("  NVG CRITICAL HORIZON MASS = THE CHANDRASEKHAR SCALE (de-mystified)")
print("=" * 74)
print(f"  M_Omega              = 859 MeV  -> m_Omega = {m_Omega:.4e} kg")
print(f"  m_Omega / m_nucleon  = {m_Omega/m_N:.4f}   (a hadron mass, ~ the nucleon)")
print(f"  M_Planck             = {M_Planck:.4e} kg")
print("-" * 74)
print(f"  inner core scale l   = {l:.1f} m  ({l/1e3:.3f} km)")
print(f"  M_crit  (direct)     = {M_crit_direct/Msun:.4f} M_sun")
print(f"  M_crit  (closed form C*M_Planck^3/m_Omega^2) = {M_crit_closed/Msun:.4f} M_sun")
print(f"  prefactor C = 9/(8*sqrt(2*pi)) = {C:.6f}")
print(f"  agreement |direct-closed|/direct = {abs(M_crit_direct-M_crit_closed)/M_crit_direct:.2e}")
assert abs(M_crit_direct - M_crit_closed) / M_crit_direct < 1e-12, "closed form must match"
print("-" * 74)
print(f"  stellar scale  M_Planck^3/m_nucleon^2 = {M_stellar_scale/Msun:.3f} M_sun")
print(f"  (Chandrasekhar mass 1.4 M_sun, NS maxima ~2 M_sun live on this same scale)")
print(f"  ratio M_crit / (M_Planck^3/m_Omega^2) = {C:.4f}  (pure O(1) number)")
print("=" * 74)
print("""  CONCLUSION: M_crit ~ 1 M_sun is NOT a coincidence or a tuned prediction.
  It is the Chandrasekhar-type combination  M_Planck^3 / m^2  evaluated at the QCD
  condensate mass m = M_Omega. Any theory whose collapse scale is set by a hadron
  mass and gravity lands at ~1 M_sun; that is the same dimensional reason white
  dwarfs and neutron stars top out near a solar mass. The striking-looking
  "critical horizon mass = mass of the Sun" is dimensional physics, not mysticism.""")
print("=" * 74)
