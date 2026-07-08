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

HONEST-SCIENCE NOTES (adversarial-review corrections):
  * beta=1/2 is NOT an independent derivation. It follows directly and ONLY from
    the assumed LINEAR-in-density melting  v^2 = v0^2 (1 - rho/rho_c):  any melting
    power p gives beta = p/2, so identifying the sqrt-law with mean field merely
    RESTATES the Landau ansatz -- it is not a prediction.
  * The naive Ginzburg number t_G ~ 0.007 computed below is UNRELIABLE and must
    NOT be counted as genuine evidence for mean field: (a) the numbers here give
    xi_0 = 0.161 fm < l_micro = 0.230 fm (xi_0 ~ 0.70 l_micro), so the correlation
    length NEVER exceeds the microscopic scale -- there is no scale separation and
    the Ginzburg-Levanyuk long-wavelength expansion PREMISE FAILS; (b) u =
    lambda_v/(4 pi) is an arbitrary weak-coupling loop factor that UNDERSTATES the
    true QCD-scale coupling (a proper 3D dimensionally-reduced Ginzburg number
    could differ by orders of magnitude).
  * Legitimate ingredients kept intact: W = |Phi| is a valid order parameter;
    t = 1 - rho/rho_c is a valid Landau control variable; the quoted 3D-Ising /
    O(4) / XY exponents are correct. The TRUE exponent is fixed externally by
    universality, not derived here, and is falsifiable via BES-II net-proton
    cumulants.

Net result: the in-framework diagnostic (no mean-field window, xi_0 < l_micro)
and external lattice-QCD universality AGREE -- both point to NON-mean-field,
beta ~ 0.33. This is not "borderline".
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
print("VERDICT  (both diagnostics AGREE -- reported honestly)")
print("="*72)
print(f"  * beta=1/2 is NOT an independent derivation: it follows directly and")
print(f"    ONLY from the assumed LINEAR-in-density melting v^2 = v0^2(1 - rho/rho_c)")
print(f"    -- any melting power p gives beta = p/2. Identifying the sqrt-law with")
print(f"    mean field merely RESTATES the Landau ansatz; it is not a prediction.")
print(f"  * The naive Ginzburg number t_G ~ u^2 = {u**2:.3f} is UNRELIABLE and does")
print(f"    NOT count as evidence for mean field: (a) xi_0 = {xi0:.2f} fm is only")
print(f"    ~{xi0/l_micro:.1f}x the microscopic length {l_micro:.2f} fm -- the correlation")
print(f"    length NEVER exceeds the microscopic scale, so there is NO scale")
print(f"    separation and the Ginzburg-Levanyuk long-wavelength expansion PREMISE")
print(f"    FAILS; (b) the 4pi-suppressed loop factor u = lambda_v/4pi = {u:.3f} is")
print(f"    an arbitrary weak-coupling estimate that UNDERSTATES the true QCD-scale")
print(f"    coupling (a proper 3D dimensionally-reduced Ginzburg number could differ")
print(f"    by orders of magnitude).")
print(f"  * In-framework diagnostic therefore gives NO mean-field window (xi_0 <")
print(f"    l_micro, lambda_v = {lambda_v} is O(1)) -> NON-mean-field, beta ~ 0.33.")
print(f"  * External input agrees: lattice QCD establishes this transition is NOT")
print(f"    mean field -- the QCD critical point is 3D-Ising (Stephanov/Rajagopal)")
print(f"    -> beta=0.326; the N_f=2 chiral transition is O(4) -> beta=0.38.")
print(f"\n  CONCLUSION: this is NOT 'borderline'. The one leg that had favoured mean")
print(f"  field -- the naive t_G ~ {u**2:.3f} -- fails its own long-wavelength premise")
print(f"  (xi_0 < l_micro, no scale separation) and is discarded. The in-framework")
print(f"  diagnostic and external lattice-QCD universality then point the SAME way:")
print(f"  NON-mean-field, beta ~ 0.33 -- they AGREE rather than disagree. The true")
print(f"  exponent is fixed externally by universality (not derived here); W=|Phi|")
print(f"  remains a legitimate order parameter and t = 1 - rho/rho_c a legitimate")
print(f"  Landau control variable. This is why the BES-II net-proton cumulant")
print(f"  measurement is the clean, decisive, falsifiable test of the exponent.")
