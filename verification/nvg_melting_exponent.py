#!/usr/bin/env python3
"""
Derivation of the vacuum melting exponent beta in W(rho) ~ (1 - rho/rho_c)^beta
from the NVG potential V(|Phi|) = (lam/4)(|Phi|^2 - v0^2)^2, following the
Madelung / Fokker-Planck stationary route (workflow finding #1) and the
QCD-critical universality route (workflow finding #6).

Result summary:
  * The phenomenological sqrt-law (beta=1/2) is EXACTLY the mean-field extremum
    of the NVG quartic (for a homogeneous condensate the Madelung quantum
    pressure Q=0, so the stationary balance = classical minimum).
  * Beyond mean field, universality corrects beta. A QCD-anchored finite-density
    critical point is in the 3D Ising class -> beta = 0.326 (or 3D XY = 0.349 if
    the U(1) phase theta is a true Goldstone). Both differ sharply from 1/2.
  * This is falsifiable now against RHIC BES-II net-proton cumulants, and it
    propagates into the bounce EOS w(rho -> rho_c).
"""
import numpy as np
from scipy.optimize import minimize_scalar

v0 = 859.0          # MeV, v0 = M_Omega_0 (QCD anchor)
lam = 1.0           # overall scale; does not move the minimum

# ---- 1. mean-field: minimise V_eff(W; t), t = 1 - rho/rho_c ----------------
# Density enters as a linearly-melting symmetry-breaking scale v^2(rho)=v0^2 * t
# (the standard Landau coupling that makes the condensate vanish at rho_c).
def V_eff(W, t):
    return 0.25 * lam * (W*W - v0*v0*max(t, 0.0))**2

ts = np.logspace(-6.0, -0.5, 60)
Wmf = np.array([minimize_scalar(lambda W: V_eff(W, t), bounds=(0, 1.5*v0),
                                method='bounded').x for t in ts])
# fit exponent from the small-t log-log slope
m = ts < 1e-2
beta_mf = np.polyfit(np.log(ts[m]), np.log(Wmf[m]), 1)[0]
print("="*74)
print("VACUUM MELTING EXPONENT beta  (W ~ (1 - rho/rho_c)^beta)")
print("="*74)
print(f"\n1. MEAN-FIELD (minimise the NVG quartic V):")
print(f"   fitted beta = {beta_mf:.4f}   ->  W = v0*(1-rho/rho_c)^{beta_mf:.3f}")
print(f"   i.e. the phenomenological sqrt-law IS the mean-field extremum (Landau).")
print(f"   [homogeneous condensate: Madelung quantum pressure Q = -hbar^2/2m *")
print(f"    nabla^2 sqrt(W)/sqrt(W) = 0, so the stationary Fokker-Planck balance")
print(f"    reduces to V'(W)=0 -> exactly this minimum. No fluctuations => beta=1/2.]")

# ---- 2. universality-corrected exponents (cited, not fitted) ----------------
BETA = {"mean-field / sqrt-law": 0.5,
        "3D XY  (theta a Goldstone)": 0.3486,
        "3D Ising (theta pinned; QCD critical point class)": 0.3264}
print("\n2. BEYOND MEAN-FIELD (universality of a genuine 3D critical point):")
print("   The QCD critical point is established to be 3D-Ising (Stephanov/Rajagopal).")
print("   A finite-T transition (the bounce T_b=432 MeV, heavy-ion T_c=157 MeV)")
print("   dimensionally reduces 3+1D -> 3D, so beta takes a 3D value, NOT 1/2:")
for k, b in BETA.items():
    print(f"     beta = {b:.4f}   {k}")

# ---- 3. how different are the melting profiles? -----------------------------
print("\n3. MELTING PROFILE W/v0 = (1-rho/rho_c)^beta  (how far the laws diverge):")
xs = [0.5, 0.9, 0.99, 0.999]  # rho/rho_c
print(f"   {'rho/rho_c':>10}", *[f"{k.split()[0]:>10}" for k in BETA])
for x in xs:
    t = 1 - x
    print(f"   {x:>10.3f}", *[f"{t**b:>10.4f}" for b in BETA.values()])
# quantify near the bounce
t_edge = 1e-3
w_mf = t_edge**0.5; w_is = t_edge**0.3264
print(f"\n   Near the bounce (rho/rho_c=0.999): the 3D-Ising condensate is "
      f"{w_is/w_mf:.1f}x LARGER than the sqrt-law")
print(f"   -> it melts MORE SLOWLY, so more condensate energy survives close to rho_c.")

# ---- 4. consequence for the bounce equation of state ------------------------
# condensate energy density ~ V curvature * W^2 ~ W^2 ; its share of the total
# vanishes as (1-rho/rho_c)^{2 beta}. The SEC-violating term that halts the
# crunch scales with this. beta smaller -> term dies slower -> the bounce is
# approached differently.
print("\n4. CONSEQUENCE (bounce): condensate energy ~ W^2 ~ (1-rho/rho_c)^{2 beta}")
for k, b in BETA.items():
    print(f"     {k.split('(')[0].strip():28} exponent 2*beta = {2*b:.3f}")
print("   The modified Friedmann term (1-rho/rho_c) that produces the SEC violation")
print("   assumes 2*beta = 1 (mean-field). If beta=0.326, that exponent is 0.653,")
print("   so H^2 = (8piG/3) rho (1-rho/rho_c)^{0.65} -- a DIFFERENT bounce, testable")
print("   through the e-folds / CMB low-l cutoff and the pre-bounce w(z).")

print("\n" + "="*74)
print("VERDICT")
print("="*74)
print(" * The sqrt-law is not arbitrary -- it IS the mean-field extremum of NVG's own")
print("   quartic potential. But note this is really the LANDAU ANSATZ RESTATED: beta=1/2")
print("   follows only from the assumed LINEAR melting v^2 ~ (1-rho/rho_c); any melting")
print("   power p gives beta = p/2. So it is a consequence of the linear ansatz, not an")
print("   independent prediction of the value 1/2.")
print(" * beta=1/2 is a mean-field ASSUMPTION. A QCD-anchored finite-density transition,")
print("   IF it is a genuine critical point, sits in a 3D universality class -> beta =")
print("   0.326 (Ising) or 0.349 (XY). The clean future test is RHIC BES-II (existing")
print("   BES-I central values cannot yet distinguish the classes; see nvg_melting_beta_besii.py).")
print(" * beta APPEARS in NS deep core, heavy-ion, and the cosmological bounce, but is")
print("   MEASURABLE in at most one: NS are blind to it (identifiability + tail-sensitivity)")
print("   and the CMB does not constrain it (horizon-chain). Heavy-ion is the sole handle.")
