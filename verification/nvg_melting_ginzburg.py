#!/usr/bin/env python3
"""
Task A — Ginzburg criterion for the rho_c melting transition: is the mean-field
sqrt-law (beta=1/2) self-consistent, or do fluctuations force a 3D exponent?

Uses the framework's OWN potential parameters:
  V(|Phi|) = (lambda_v/4)(|Phi|^2 - v0^2)^2,  lambda_v = 1.02  (nvg_g2_mechanism.py),
  v0 = M_Omega_0 = 859 MeV,  radial (sigma) mass m_sigma = sqrt(2 lambda_v) v0,
  rho_c = 7.09e4 MeV/fm^3.

Mean-field (Landau) holds only when the order-parameter fluctuation over one
correlation volume is small compared to the order parameter itself -- i.e. when
the bare correlation length xi_0 is much larger than the microscopic scale, so
there is a temperature window where fluctuations average out. If xi_0 ~ the
microscopic scale and the quartic coupling is O(1), there is NO mean-field window
and the transition takes its 3D universality-class exponent (beta ~ 0.33).
"""
import math

hbar_c = 197.3269804        # MeV fm
lambda_v = 1.02             # vacuum quartic (framework value)
v0 = 859.0                  # MeV (M_Omega_0)
rho_c = 7.09e4              # MeV/fm^3 (critical energy density)

m_sigma = math.sqrt(2.0*lambda_v)*v0         # radial mode mass, MeV
xi0 = hbar_c/m_sigma                          # bare correlation length, fm

# microscopic length from the critical energy density (natural units: eps ~ L^-4)
rho_c_fm4 = rho_c/hbar_c                       # MeV/fm^3 -> fm^-4
l_micro = rho_c_fm4**(-0.25)                   # fm

print("="*72)
print("GINZBURG CRITERION FOR THE rho_c MELTING TRANSITION")
print("="*72)
print(f"\nFramework inputs:")
print(f"  lambda_v = {lambda_v}   (vacuum quartic)")
print(f"  v0 = M_Omega_0 = {v0:.0f} MeV")
print(f"  m_sigma = sqrt(2 lambda_v) v0 = {m_sigma:.0f} MeV  (radial 'sigma' mode)")
print(f"  rho_c = {rho_c:.2e} MeV/fm^3")

print(f"\nDerived scales:")
print(f"  bare correlation length  xi_0 = hbar c / m_sigma = {xi0:.3f} fm")
print(f"  microscopic length       l    = rho_c^(-1/4)     = {l_micro:.3f} fm")
print(f"  ratio  xi_0 / l = {xi0/l_micro:.2f}")

# Ginzburg-Levanyuk reduced-temperature width of the fluctuation region.
# Dimensionless renormalized quartic at the correlation scale (d=3 reduced theory):
#   u ~ lambda_v / (4 pi)   (loop expansion parameter);  mean-field window ends at
#   t_G ~ u^2  (Ginzburg number).  Mean field is valid only for |t| >> t_G.
u = lambda_v/(4.0*math.pi)
t_G = u**2                                          # Ginzburg number ~ u^2 (d=3)
print(f"\nGinzburg number (loop parameter u = lambda_v/4pi = {u:.3f}):")
print(f"  t_G ~ u^2 = {u**2:.3f}   (mean-field valid only for reduced |t| >> t_G)")

print("\n" + "="*72)
print("VERDICT  (the two criteria genuinely disagree -- reported honestly)")
print("="*72)
print(f"  * FOR mean field (beta=1/2): the 4pi-suppressed loop parameter")
print(f"    u = lambda_v/4pi = {u:.3f} is small, so the naive Ginzburg number")
print(f"    t_G ~ u^2 = {u**2:.3f} would keep mean field valid except within")
print(f"    |t| < {u**2:.3f} of rho_c. On this crude estimate alone, mean field")
print(f"    is NOT obviously broken.")
print(f"  * AGAINST mean field: (a) xi_0 = {xi0:.2f} fm is only ~{xi0/l_micro:.1f}x the")
print(f"    microscopic length {l_micro:.2f} fm, so a 'correlation volume' barely")
print(f"    contains one microscopic cell -- the mean-field averaging assumption is")
print(f"    marginal; (b) lambda_v = {lambda_v} is O(1), so the 4pi-suppressed u")
print(f"    understates the true coupling strength at a QCD-scale transition.")
print(f"  * DECISIVE input is external: lattice QCD establishes this transition is")
print(f"    NOT mean field -- the QCD critical point is 3D-Ising (Stephanov/Rajagopal)")
print(f"    -> beta=0.326; the N_f=2 chiral transition is O(4) -> beta=0.38.")
print(f"\n  CONCLUSION: whether NVG's rho_c melting is mean-field (beta=1/2) or 3D")
print(f"  (beta~0.33) is genuinely BORDERLINE on the crude Ginzburg number with the")
print(f"  framework's own lambda_v -- it is settled in favour of NON-mean-field by")
print(f"  lattice-QCD universality. That is precisely why the BES-II cumulant")
print(f"  measurement is the clean, decisive test of the vacuum melting exponent.")
