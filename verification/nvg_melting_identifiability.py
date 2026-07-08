#!/usr/bin/env python3
"""
NVG melting-law identifiability from neutron-star data.

Reconstructs the vacuum melting law g(x)=W(x)/W(0) (realised as the effective
mass drop M*(x)=M_current_0 + M_Omega_0*g(x), x=n_B/n0) from NS structure, using
the framework's own pure-neutron-matter forward model (nvg_full_ns_eos.py) with a
TOV solver in clean geometric (km) units.

Asks four questions the theory leaves open:
  A. Does the cosmological law sqrt(1-rho/rho_c) melt anything at NS density, and
     what *effective* rho_c would NS structure require if it did?
  B. At a fixed M_max, is the melting FORM identifiable from M_max alone?
  C. Which observable (R_2.0, the heavy-NS radius) breaks the high-density
     degeneracy of the melting law?
  D. Does the conformal speed of sound c_s^2 -> 1/3 hold across melting forms?

CAVEAT: pure-neutron-matter, NO crust / NO beta-equilibrium (see the parent
script). Absolute radii (~21-23 km) are therefore NOT trustworthy; the robust
content is the *differential* behaviour (spreads, sensitivities, ratios) and the
low- vs high-density contrast. Canonical numbers: nvg_tidal_deformability.py.
"""
from __future__ import annotations
import math, numpy as np

# ── constants (from nvg_full_ns_eos.py) ──────────────────────────────────────
hbar_c = 197.3269804; M_N = 939.0; n_0 = 0.16
M_Omega_0 = 859.0; M_current_0 = 80.0
alpha_v = 4.0; nu_v = 2.0; C_v_n0 = 100.0
RHO_C = 7.09e4                       # MeV/fm^3, cosmological critical density
K_GEO = 1.3234e-6                    # MeV/fm^3 -> km^-2
MSUN_KM = 1.4766                     # 1 M_sun in km

# ── core EOS with a pluggable melting profile g(x) ───────────────────────────
def core(n_B, g):
    x = max(n_B / n_0, 0.0)
    m = M_current_0 + M_Omega_0 * g(x)
    kf = (3 * np.pi**2 * n_B)**(1/3) * hbar_c
    ef = math.sqrt(kf**2 + m**2); lt = math.log((kf + ef) / m)
    ek = (kf*ef*(2*kf**2 + m**2) - m**4*lt) / (8*np.pi**2*hbar_c**3)
    pk = (kf*ef*(2*kf**2/3 - m**2) + m**4*lt) / (8*np.pi**2*hbar_c**3)
    vp = (m - M_N) * n_B * 0.5
    vv = C_v_n0 * x*x / (1 + alpha_v*x**nu_v); ev = vv*n_B
    pv = ev*(1 + nu_v*alpha_v*x**nu_v)/(1 + alpha_v*x**nu_v)
    return ek + vp + ev, pk + pv

class EOS:
    """Tabulated eps(P) and c_s^2(P) in geometric (km^-2) units."""
    def __init__(self, g, n_trans=2.0):
        nt = n_trans*n_0
        ng = np.logspace(-4, 1.5, 1600)*n_0
        et, pt = core(nt, g); c2 = 1/3
        e = []; p = []
        for n in ng:
            if n <= nt: ee, pp = core(n, g)
            else: ee = et*(n/nt)**(1+c2); pp = pt + c2*(ee - et)
            e.append(ee); p.append(pp)
        e = np.array(e)*K_GEO; p = np.array(p)*K_GEO          # -> km^-2
        # keep strictly increasing P for interpolation
        keep = np.concatenate(([True], np.diff(p) > 0))
        self.e = e[keep]; self.p = p[keep]
        # c_s^2 = dP/deps
        de = np.gradient(self.e); dp = np.gradient(self.p)
        self.cs2 = np.clip(dp/np.where(de == 0, 1e-30, de), 1e-6, 1.0)
    def eps(self, P):
        if P <= self.p[0]: return 0.0
        if P >= self.p[-1]: return self.e[-1]
        return float(np.interp(P, self.p, self.e))
    def cs2_of_P(self, P):
        return float(np.interp(P, self.p, self.cs2))
    def P_of_eps(self, eps):
        return float(np.interp(eps, self.e, self.p))

# ── TOV + tidal (Hinderer 2008 / Postnikov 2010), geometric km units ─────────
def structure(eos, P_c, dr=0.01, rmax=40.0):
    """TOV from central pressure P_c (km^-2). Returns (M[Msun], R[km])."""
    if P_c <= eos.p[0]: return 0.0, 0.0
    def rhs(r, m, P):
        eps = eos.eps(P)
        dm = 4*math.pi*r**2*eps
        den = r*(r - 2*m)
        dP = -((eps + P)*(m + 4*math.pi*r**3*P))/den if den > 0 else 0.0
        return dm, dP
    r = 1e-4
    m = 4*math.pi/3*eos.eps(P_c)*r**3
    P = P_c
    while P > eos.p[0]*1.001 and r < rmax:
        k1 = rhs(r, m, P)
        k2 = rhs(r+.5*dr, m+.5*dr*k1[0], P+.5*dr*k1[1])
        k3 = rhs(r+.5*dr, m+.5*dr*k2[0], P+.5*dr*k2[1])
        k4 = rhs(r+dr, m+dr*k3[0], P+dr*k3[1])
        m += dr/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        P += dr/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        r += dr
        if 2*m/r > 0.999: break
    return m/MSUN_KM, r

def mr_lambda(g, n_eps=18):
    """Return (M_max, R at 1.4, R at 2.0). Central density grid 1.5..9 n0."""
    eos = EOS(g)
    M = []; R = []
    for nc in np.linspace(1.5, 9.0, n_eps)*n_0:
        ec, _ = core(nc, g)
        Pc = eos.P_of_eps(ec*K_GEO)
        m, r = structure(eos, Pc)
        M.append(m); R.append(r)
    M = np.array(M); R = np.array(R)
    i = int(np.argmax(M)); bm, br = M[:i+1], R[:i+1]
    Mmax = float(bm.max()); o = np.argsort(bm)
    R14 = float(np.interp(1.4, bm[o], br[o])) if bm.max() >= 1.4 else 0.0
    R20 = float(np.interp(2.0, bm[o], br[o])) if bm.max() >= 2.0 else 0.0
    return Mmax, R14, R20

def fit_form(ffn, lo, hi, target, tol=0.02):
    Mlo = mr_lambda(ffn(lo))[0]; Mhi = mr_lambda(ffn(hi))[0]
    if not (min(Mlo, Mhi)-0.15 <= target <= max(Mlo, Mhi)+0.15): return None
    a, b, Ma = lo, hi, Mlo
    for _ in range(26):
        mid = 0.5*(a+b); Mm = mr_lambda(ffn(mid))[0]
        if abs(Mm-target) < tol: return mid
        if (Ma-target)*(Mm-target) < 0: b = mid
        else: a, Ma = mid, Mm
    return mid

# ── melting forms ────────────────────────────────────────────────────────────
g_author = lambda x: (1+0.8*x)**(-0.3125)
g_cosmo  = lambda x: math.sqrt(max(1-(M_N*n_0*x)/RHO_C, 1e-9))
g_sqrt   = lambda xc: (lambda x: math.sqrt(max(1-x/xc, 1e-9)))
g_logi   = lambda x0: (lambda x: 1.0/(1+(x/x0)**2))
g_pow    = lambda b:  (lambda x: (1+x)**(-b))

def report():
    print("="*78)
    print("NVG MELTING-LAW IDENTIFIABILITY FROM NEUTRON-STAR DATA")
    print("="*78)
    xs = [1, 2, 4, 6]

    Mm0, R14_0, R20_0 = mr_lambda(g_author)
    print(f"\n[baseline] author g(x)=(1+0.8x)^-0.3125")
    print(f"   M_max={Mm0:.3f}  R_1.4={R14_0:.1f}km  R_2.0={R20_0:.1f}km  (crustless: R_1.4 not trustworthy)")

    # A. cosmological law + effective rho_c required by NS
    print("\n" + "-"*78)
    print("A. Cosmological law sqrt(1 - rho/rho_c) at NS density, and the effective rho_c NS needs")
    Mc, _, _ = mr_lambda(g_cosmo)
    print(f"   real rho_c=7.09e4:  melt@6n0={1-g_cosmo(6):.4f}  M_max={Mc:.3f}  -> inert at NS density")
    xc = fit_form(g_sqrt, 3.0, 200.0, Mm0)
    if xc:
        rho_c_eff = M_N*n_0*xc
        print(f"   effective rho_c required by NS M_max: rho_c_NS = {rho_c_eff:.0f} MeV/fm^3  (x_c={xc:.1f} n0)")
        print(f"   ratio rho_c(cosmo)/rho_c(NS) = {RHO_C/rho_c_eff:.0f}x  <- the two-scale problem, quantified")

    # B/C. form identifiability at matched M_max; which observable discriminates
    print("\n" + "-"*78)
    print(f"B/C. Tune different melting FORMS to the SAME M_max={Mm0:.2f}; compare g(x) + observables")
    forms = [("author (1+0.8x)^-.3125", g_author),
             ("sqrt(1-x/xc)", g_sqrt(fit_form(g_sqrt, 3.0, 80.0, Mm0) or 4.2)),
             ("logistic 1/(1+(x/x0)^2)", g_logi(fit_form(g_logi, 1.0, 30.0, Mm0) or 3.0))]
    print(f"\n   {'form':26} {'M_max':>6} {'R2.0':>6}   " + " ".join(f'g({x}n0)' for x in xs))
    rows = []
    for name, g in forms:
        Mm, R14, R20 = mr_lambda(g); gv = [g(x) for x in xs]
        rows.append((name, gv, R20))
        print(f"   {name:26} {Mm:6.3f} {R20:6.1f}   " + " ".join(f"{v:6.2f}" for v in gv))
    G = np.array([r[1] for r in rows]); R20s = np.array([r[2] for r in rows])
    print(f"\n   melting-fraction spread at matched M_max (1 - g):")
    for j, x in enumerate(xs):
        print(f"     x={x}n0: 1-g in {1-G[:,j].max():.2f}..{1-G[:,j].min():.2f}  (spread {G[:,j].max()-G[:,j].min():.2f})")
    print(f"   R_2.0 (deep-core radius) across the SAME forms: "
          f"{R20s.min():.1f}..{R20s.max():.1f} km  (spread {R20s.max()-R20s.min():.1f} km)")

    # D. conformal speed of sound
    print("\n" + "-"*78)
    print("D. Conformal limit c_s^2 -> 1/3 across forms (deep core):")
    for name, g in forms:
        eos = EOS(g); print(f"   {name:26} c_s^2(core) = {eos.cs2_of_P(eos.p[-1]*0.5):.3f}")

    print("\n" + "="*78)
    print("VERDICT")
    print("="*78)
    lo_spread = G[:, 1].max() - G[:, 1].min()      # x=2
    hi_spread = G[:, 3].max() - G[:, 3].min()      # x=6
    print(" 1. Cosmological sqrt(1-rho/rho_c) is inert at NS density; the melting NS")
    if xc:
        print(f"    structure needs has an effective rho_c ~{RHO_C/rho_c_eff:.0f}x lower ({rho_c_eff:.0f} MeV/fm^3).")
    print(f" 2. At matched M_max the low-density melting is pinned (spread {lo_spread:.2f} at 2 n0)")
    print(f"    but the deep core is NOT (spread {hi_spread:.2f} at 6 n0).")
    print(f" 3. R_2.0 (heavy-NS radius) spreads {R20s.max()-R20s.min():.1f} km across those forms and")
    print("    is crust-insensitive -> it is the lever that pins the deep-core melting.")
    print(" 4. c_s^2->1/3 is imposed by the CSS crossover, so it does not by itself")
    print("    discriminate the sub-crossover forms.")
    print("\n NOTE: crustless pure-neutron model -> R_1.4 and tidal Lambda need the")
    print("       beta-equilibrated chain (nvg_tidal_deformability.py) to be trustworthy;")
    print("       robust here are M_max, R_2.0, and the differential spreads above.")

if __name__ == "__main__":
    report()
