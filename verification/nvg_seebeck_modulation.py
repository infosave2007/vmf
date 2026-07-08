#!/usr/bin/env python3
"""
NVG prediction for a lab-modulated Seebeck coefficient via the vacuum mass
fraction:  m* = W * m0,  W = sqrt(1 - rho/rho_c).

Derivation (degenerate carriers, Mott formula):
  For a 3D parabolic band the diffusive Seebeck coefficient is
     S = (8 pi^2 k_B^2 / 3 e h^2) * m* * T * (pi/3n)^(2/3)      (Snyder-Toberer 2008)
  i.e. at fixed carrier density n and temperature T,  S is PROPORTIONAL to m*.
  If NVG scales the (effective) mass universally, m* = W*m0, then
     dS/S = dm*/m* = dW/W.
  The melting law W = sqrt(1 - rho/rho_c) gives, for W~1,
     W^2 = 1 - rho/rho_c  ->  2W dW = -drho/rho_c  ->  dW/W = -drho/(2 W^2 rho_c)
                                                              ~ -drho/(2 rho_c).
  Hence the ENTIRE predicted effect is the ratio of the injected lab energy
  density to the vacuum-condensate scale:
     dS = S * drho / (2 rho_c).

This script puts numbers on drho (what a bench can inject) vs rho_c (a QCD scale)
and compares dS to the real Seebeck measurement floor.
"""
import math

# ---- unit conversions ----
MeV = 1.602176634e-13          # J
fm3 = 1e-45                    # m^3
MeV_fm3_to_J_m3 = MeV / fm3    # 1 MeV/fm^3 -> J/m^3  = 1.602e32

# ---- candidate condensate scales rho_c (J/m^3) ----
RHO_C = {
    "QCD cosmological rho_c^cosmo (7.09e4 MeV/fm^3)": 7.09e4 * MeV_fm3_to_J_m3,
    "nuclear saturation eps0 (150 MeV/fm^3)  [most optimistic]": 150.0 * MeV_fm3_to_J_m3,
    "chiral scale (~300 MeV/fm^3)": 300.0 * MeV_fm3_to_J_m3,
}

# ---- lab energy densities drho we can actually inject (J/m^3) ----
mu0 = 1.25663706212e-6
c = 2.99792458e8
DRHO = {
    "Peltier module thermal modulation (C_v*dT, dT=200K)": 1.2e6 * 200,   # ~2.4e8
    "strong static magnet B=100 T (B^2/2mu0)": (100.0**2)/(2*mu0),         # ~4.0e9
    "intense laser 1e15 W/cm^2 (I/c)": (1e19)/c,                           # ~3.3e10
    "petawatt focus 1e23 W/cm^2 (I/c)": (1e27)/c,                          # ~3.3e18
    "extreme HED / NIF (~10 Gbar = 1e15 Pa)": 1e15,                        # 1e15
}

# ---- observable baseline & measurement floor ----
S0 = 200e-6          # V/K, typical Bi2Te3 Seebeck coefficient
S_FLOOR = 1e-9       # V/K, heroic (nV/K, lock-in) Seebeck resolution
S_ROUTINE = 1e-7     # V/K, routine (~0.1 uV/K)

print("="*80)
print("NVG lab-modulated Seebeck coefficient:  dS = S0 * drho / (2 rho_c)")
print("="*80)
print(f"\nBaseline S0 = {S0*1e6:.0f} uV/K;  measurement floor ~ {S_FLOOR*1e9:.0f} nV/K "
      f"(routine ~{S_ROUTINE*1e6:.1f} uV/K)")

# use the MOST OPTIMISTIC (smallest) rho_c to give the effect its best chance
rho_c_opt_name = "nuclear saturation eps0 (150 MeV/fm^3)  [most optimistic]"
rho_c_opt = RHO_C[rho_c_opt_name]
print(f"\nUsing the most optimistic scale: {rho_c_opt_name} = {rho_c_opt:.2e} J/m^3")
print(f"\n{'lab energy source':52} {'drho[J/m^3]':>11} {'dW/W':>10} {'dS[V/K]':>10} {'vs floor':>10}")
print("-"*100)
for name, drho in sorted(DRHO.items(), key=lambda kv: kv[1]):
    dW_W = drho/(2*rho_c_opt)
    dS = S0*dW_W
    orders = math.log10(dS/S_FLOOR)
    print(f"{name:52} {drho:11.2e} {dW_W:10.1e} {dS:10.1e} {orders:+9.1f} dex")

# ---- break-even: what drho is needed to reach the floor? ----
print("\n" + "-"*80)
dW_W_needed = S_FLOOR/S0
drho_needed = 2*rho_c_opt*dW_W_needed
print(f"To reach the {S_FLOOR*1e9:.0f} nV/K floor you need dW/W = {dW_W_needed:.1e},")
print(f"i.e. drho = {drho_needed:.2e} J/m^3.")
# compare to references
peltier = DRHO["Peltier module thermal modulation (C_v*dT, dT=200K)"]
petawatt = DRHO["petawatt focus 1e23 W/cm^2 (I/c)"]
ns_core = 5*150.0*MeV_fm3_to_J_m3   # ~5 n0 neutron-star core energy density
print(f"  = {drho_needed/peltier:.1e}x a Peltier module's thermal modulation,")
print(f"  = {drho_needed/petawatt:.1e}x a petawatt laser focus,")
print(f"  = {drho_needed/ns_core:.1e}x a neutron-star core energy density ({ns_core:.1e} J/m^3).")

print("\n" + "="*80)
print("HONEST VERDICT")
print("="*80)
print("""\
 1) SCALE. The whole effect is drho_lab / rho_c. The condensate is pinned at a
    QCD energy scale (rho_c ~ 1e34-1e37 J/m^3); a bench injects drho ~ 1e8-1e18
    J/m^3. So dS/S ~ 1e-27 (Peltier) to ~1e-17 (petawatt) -- 11 to 21 orders of
    magnitude BELOW the Seebeck measurement floor. Break-even would need energy
    densities approaching a neutron-star interior. Dead in practice.

 2) COUPLING (worse). In standard NVG, W is the QCD vacuum condensate: it sets
    HADRON masses. The electron/carrier mass is ELECTROWEAK (Higgs), not QCD, so
    modulating W does not change m* at all unless one POSTULATES W as a universal
    dilaton coupling to every mass -- which is beyond NVG's stated content and not
    established physics. Even granting it, killer (1) applies.

 CONCLUSION: the effect is real only as a metaphor; as a number it is ~1e-20 x
 the detection threshold. This is the SAME physics as the neutron-star result --
 even a 7 n0 NS core melts the deep vacuum only ~0.8%; a lab is ~20 orders of
 magnitude softer still, so it modulates W by ~1e-27. No bench experiment (Peltier
 or otherwise) can measure an NVG vacuum-mass modulation of the Seebeck coefficient.

 The only way to get a MEASURABLE Seebeck shift is to let 'W' be a REAL electronic
 order parameter (a superconducting/CDW/excitonic gap), where dW/W ~ O(1) near a
 transition -- but that is mainstream correlated-electron thermoelectrics, not a
 vacuum/NVG effect, and carries no free energy and no anomaly.
""")
