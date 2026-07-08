#!/usr/bin/env python3
"""
Tail-sensitivity test (HONEST-SCIENCE CORRECTION after adversarial review):
is the neutron-star TAIL sensitive to the chiral melting exponent beta, or blind?

WHAT THIS SCRIPT ACTUALLY MEASURES (corrected):
In this hybrid EOS the matter above the 2 n0 CSS transition is PURE
constant-sound-speed (CSS) matter with NO chiral melting law. The +/-9% Gaussian
bump placed at 5.5 n0 therefore lies ENTIRELY inside the CSS-masked region. Its
only physical effect is a small (~2.4%) shift of the CSS-branch intercept via the
numerically-differentiated transition pressure at 2 n0. So the measured
|Delta R_2.0| ~ 0.52 km is a response to a GENERIC high-density stiffness change
-- one that co-moves M_max, R_1.4 and R_2.0 together and is fully degenerate with
cs2_q, the transition density and the latent heat -- NOT a measurement of the
melting exponent beta. A heavy-NS radius is therefore NOT a "second, independent
handle on beta".

Moreover W_{0.326}-W_{0.5} for (1-x/x_c)^beta is a one-sided, monotone,
slope-diverging function toward x_c, not a symmetric Gaussian; the 0.09 bump
amplitude was asserted, not derived; and x_c was never assigned. The Gaussian is
NOT a faithful proxy for the true W_{0.326}-W_{0.5} difference.

CONCLUSION: within this hybrid model the neutron-star tail is effectively BLIND to
beta (consistent with the identifiability result). A heavy-NS radius is NOT shown
to be a handle on the chiral exponent. A proper test would put the melting law
where it physically acts -- push the CSS transition above the max central density,
or apply beta to the hadronic branch below the transition -- and propagate the
true Delta W(x); left as future work.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import nvg_eos_beta_saturated_vector as base
import nvg_tidal_deformability as td

n_0 = base.n_0; W0 = base.M_Omega_0
def set_g(g): base.M_Omega = lambda n_b, k1, k2, _g=g: W0*_g(max(n_b/n_0, 0.0))

def obs(g, npc=80):
    set_g(g)
    try: eos = td.EOS(p_match=1.5, Gamma=1.35)
    except Exception: return None
    M=[];R=[]
    for pc in np.logspace(-1.0, 3.4, npc):
        try: m,r,k2,lam = td.solve_tov_tidal(eos, pc)
        except Exception: continue
        if m>0.5 and r>5: M.append(m);R.append(r)
    if len(M)<5: return None
    M=np.array(M);R=np.array(R); i=int(np.argmax(M)); M,R=M[:i+1],R[:i+1]; o=np.argsort(M)
    Mmax=float(M.max())
    R14=float(np.interp(1.4,M[o],R[o])) if M.max()>=1.4 else 0.
    R20=float(np.interp(2.0,M[o],R[o])) if M.max()>=2.0 else 0.
    return Mmax,R14,R20

g_power = lambda x: (1.0+0.80*x)**(-0.25/0.80)      # baseline reference (calibrates cleanly)

print("="*74)
print("TAIL-SENSITIVITY: is the neutron-star tail of the chiral condensate")
print("sensitive to the melting exponent beta = 0.326 vs 0.5?")
print("="*74)

b = obs(g_power)
print(f"\n[baseline RMF power-law]  M_max={b[0]:.3f}  R_1.4={b[1]:.2f}  R_2.0={b[2]:.2f}")

print("""
(1) Direct replacement of the melting law by a chiral critical form
    W=(1-x/x_c)^beta FAILS -- either the n0 self-calibration does not converge or
    the star runs away (M_max>6, R->cap). Robust conclusion: within this
    framework the CSS quark crossover at 2 n0 (c_s^2 -> 1/3) governs the
    high-density branch and MASKS any sharp chiral melting, so no critical
    exponent can be imprinted on NS structure directly.""")

# (2) NOT a beta-signal. The region above the 2 n0 CSS transition is pure CSS
# matter, so a bump placed at 5.5 n0 sits INSIDE the CSS-masked region and cannot
# carry the melting exponent. It only shifts the CSS-branch intercept (~2.4%) via
# the numerically-differentiated transition pressure at 2 n0 -- i.e. a GENERIC
# high-density stiffness change, degenerate with cs2_q / transition density / latent
# heat. Note also W_{0.326}-W_{0.5} is a one-sided, monotone, slope-diverging
# function toward x_c -- NOT a symmetric Gaussian; the 0.09 amplitude was asserted,
# not derived, and x_c was never assigned. We keep the +/-9% Gaussian only as a
# crude high-density stiffness perturbation to read the observable response; it is
# NOT a faithful proxy for the true W_{0.326}-W_{0.5} difference.
def bump(delta, x0=5.5, w=1.2):
    return lambda x: g_power(x)*(1.0 + delta*math.exp(-0.5*((x-x0)/w)**2))
print("(2) Generic-stiffness probe (NOT a beta measurement): apply a +/-9% Gaussian")
print(f"    bump at 5.5 n0 -- inside the CSS-masked region -- to the calibrated baseline:")
print(f"    {'perturbation':>18} {'M_max':>7} {'R_1.4':>7} {'R_2.0':>7}")
res = {}
for name, d in (("baseline", 0.0), ("+9% at 5.5 n0", +0.09), ("-9% at 5.5 n0", -0.09)):
    o = obs(bump(d)); res[name] = o
    if o: print(f"    {name:>18} {o[0]:7.3f} {o[1]:7.2f} {o[2]:7.2f}")
if res.get("+9% at 5.5 n0") and res.get("-9% at 5.5 n0"):
    op, om = res["+9% at 5.5 n0"], res["-9% at 5.5 n0"]
    print(f"\n    generic-stiffness response (full +/-9% swing, NOT a beta-signal): "
          f"|Delta M_max|={abs(op[0]-om[0]):.3f}  |Delta R_1.4|={abs(op[1]-om[1]):.2f} km  "
          f"|Delta R_2.0|={abs(op[2]-om[2]):.2f} km")

print("\n" + "="*74)
print("VERDICT")
print("="*74)
print(" * Heavy-NS radius precision: NICER ~0.5-1 km; STROBE-X/eXTP/ET ~0.2-0.3 km.")
print(" * The |Delta R_2.0| ~ 0.52 km above is REAL and above TOV discretization noise,")
print("   but it is NOT a measurement of beta. The bump lives inside the CSS-masked")
print("   region, so R_2.0 responds to a ~2.4% shift of the 2 n0 CSS transition anchor /")
print("   overall high-density stiffness -- a GENERIC DEGENERACY, confounded with cs2_q")
print("   and the transition parameters, that co-moves M_max, R_1.4 and R_2.0 together.")
print(" * CONCLUSION: within this hybrid model the NS tail is effectively BLIND to beta,")
print("   consistent with the identifiability result. A heavy-NS radius is NOT shown to")
print("   be a handle on the chiral melting exponent. A proper test would put the melting")
print("   law where it physically acts -- push the CSS transition above the max central")
print("   density, or apply beta to the hadronic branch below the transition -- and")
print("   propagate the true Delta W(x); left as future work.")
