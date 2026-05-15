#!/usr/bin/env python3
"""NVG Dense Matter v3: fast analytical Fermi integrals + parameter scan."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq

hbar_c=197.3269804; M_N=939.0; n_0=0.16
MeV_fm3_to_geo=1.3234e-6; M_sun_km=1.4766
M_Omega_0=859.0; M_current_0=80.0  # from lattice σ-terms

def M_Omega(n,k1,k2):
    x=n/n_0; return M_Omega_0*(1+k2*x)**(-k1/k2) if k2*x<500 else 0.

def Mstar(n,k1,k2):
    mc=M_current_0*max(1-44.0*n/(93.**2*140.**2),0.)
    return mc+M_Omega(n,k1,k2)

def kF(n): return (1.5*np.pi**2*n)**(1./3.)*hbar_c

def eps_fg(n,m):
    """Analytical relativistic Fermi gas energy density."""
    if n<1e-12: return 0.
    k=kF(n); E=np.sqrt(k**2+m**2)
    # ε = γ/(16π²ℏc³) [k·E(2k²+m²) - m⁴·ln((k+E)/m)]
    g=4; f=g/(16*np.pi**2*hbar_c**3)
    return f*(k*E*(2*k**2+m**2)-m**4*np.log((k+E)/max(m,1e-3)))

def ns_fg(n,m):
    """Analytical scalar density."""
    if n<1e-12 or m<1e-3: return 0.
    k=kF(n); E=np.sqrt(k**2+m**2)
    g=4; f=g/(4*np.pi**2*hbar_c**3)
    return f*m*(k*E-m**2*np.log((k+E)/m))

def build_eos(k1,k2,Cw,Cs,nn=400):
    narr=np.linspace(0.02*n_0,8*n_0,nn)
    eps=np.zeros(nn)
    for i,n in enumerate(narr):
        m=Mstar(n,k1,k2)
        ns=ns_fg(n,m)
        eps[i]=eps_fg(n,m)+0.5*Cw*n**2-0.5*Cs*ns**2
    P=narr**2*np.gradient(eps/narr,narr)
    cs2=np.gradient(P,narr)/np.gradient(eps,narr)
    ms=np.array([Mstar(n,k1,k2) for n in narr])
    return narr,eps,P,ms,cs2

def calibrate(k1,k2,Cs):
    def obj(Cw):
        n=n_0; m=Mstar(n,k1,k2)
        e=eps_fg(n,m)+0.5*Cw*n**2-0.5*Cs*ns_fg(n,m)**2
        return e/n-M_N-(-16.)
    try: return brentq(obj,0,20000)
    except: return None

def tov(eofP,Pc):
    c=MeV_fm3_to_geo; r0=.001; ec=eofP(Pc); m0=4/3*np.pi*r0**3*ec*c
    def rhs(r,y):
        m,p=y
        if p<=0: return[0,0]
        e=eofP(p); d=r*(r-2*m)
        if d<=0: return[0,0]
        return[4*np.pi*r**2*e*c,-(e*c+p*c)*(m+4*np.pi*r**3*p*c)/d]
    def stop(r,y): return y[1]
    stop.terminal=True; stop.direction=-1
    s=solve_ivp(rhs,[r0,50],[m0,Pc],events=stop,max_step=.05,rtol=1e-8,atol=1e-12)
    if s.t_events[0].size>0:
        return s.y_events[0][0][0]/M_sun_km, s.t_events[0][0]
    return s.y[0,-1]/M_sun_km, s.t[-1]

def run(k1,k2,Cs):
    Cw=calibrate(k1,k2,Cs)
    if Cw is None: return None
    narr,eps,P,ms,cs2=build_eos(k1,k2,Cw,Cs)
    ok=(P>0)&np.isfinite(cs2)&(cs2>0)
    if ok.sum()<20: return None
    Ps,es=P[ok],eps[ok]; si=np.argsort(Ps); Ps,es=Ps[si],es[si]
    mo=np.concatenate([[True],np.diff(Ps)>0]); Ps,es=Ps[mo],es[mo]
    if len(Ps)<10: return None
    eofP=interp1d(Ps,es,bounds_error=False,fill_value=(es[0],es[-1]))
    Pc_arr=np.logspace(np.log10(max(Ps[1],.1)),np.log10(Ps[-1]*.8),60)
    MM,RR=[],[]
    for Pc in Pc_arr:
        try:
            m,r=tov(eofP,Pc)
            if 0<m<5 and 4<r<25: MM.append(m); RR.append(r)
        except: pass
    if len(MM)<5: return None
    MM,RR=np.array(MM),np.array(RR)
    Mmax=MM.max(); imax=MM.argmax()
    R14=None
    st=np.arange(imax+1)
    if len(st)>3 and MM[st].min()<1.4<MM[st].max():
        try: R14=float(interp1d(MM[st],RR[st])(1.4))
        except: pass
    return dict(k1=k1,k2=k2,Cw=Cw,Cs=Cs,ms_n0=Mstar(n_0,k1,k2),
                Mmax=Mmax,R14=R14,cs2_max=np.nanmax(cs2[ok]),
                causal=np.nanmax(cs2[ok])<=1.0)

# === MAIN ===
print("="*60)
print("NVG DENSE MATTER v3 — БЫСТРЫЙ СКАН")
print("M_Ω,0 = 859 МэВ (lattice σ-terms, фиксировано)")
print("="*60)

results=[]
for k1 in [0.15, 0.20, 0.25]:
    for k2 in [0.5, 0.8]:
        for Cs in [200, 400, 600]:
            r=run(k1,k2,Cs)
            if r: results.append(r)
    print(f"  κ₁={k1:.2f} done ({len(results)} total)")

viable=[r for r in results if r['Mmax']>=2.01 and r['Mmax']<=2.8
        and r['causal'] and r['R14'] and 10.0<r['R14']<15.0]

print(f"\nВсего моделей: {len(results)}")
print(f"Жизнеспособных: {len(viable)}")

if viable:
    viable.sort(key=lambda r: abs(r['Mmax']-2.15))
    print(f"\n{'κ₁':>5}{'κ₂':>5}{'Cs':>5}{'Cω':>8}{'M*/MN':>7}{'Mmax':>7}{'R1.4':>6}{'cs2':>6}")
    print("-"*60)
    for r in viable[:20]:
        print(f"{r['k1']:5.2f}{r['k2']:5.2f}{r['Cs']:5.0f}{r['Cw']:8.1f}"
              f"{r['ms_n0']/M_N:7.3f}{r['Mmax']:7.3f}{r['R14']:6.1f}{r['cs2_max']:6.3f}")
    b=viable[0]
    print(f"\n✓ ЛУЧШАЯ: κ₁={b['k1']}, κ₂={b['k2']}, Cs={b['Cs']}")
    print(f"  M_max={b['Mmax']:.3f} M☉, R₁.₄={b['R14']:.1f} км")
    print(f"  M_Ω(n₀)={M_Omega(n_0,b['k1'],b['k2']):.0f}, "
          f"M_Ω(3n₀)={M_Omega(3*n_0,b['k1'],b['k2']):.0f}, "
          f"M_Ω(6n₀)={M_Omega(6*n_0,b['k1'],b['k2']):.0f} МэВ")
    print(f"\n  ВЫВОД: двухкомпонентная масса (M_current + M_Ω)")
    print(f"  с вакуумным входом из lattice QCD")
    print(f"  порождает реалистичные нейтронные звёзды.")
else:
    results.sort(key=lambda r: abs(r['Mmax']-2.15))
    print("\nБлижайшие:")
    for r in results[:10]:
        R14s=f"{r['R14']:.1f}" if r['R14'] else "—"
        print(f"  κ₁={r['k1']} κ₂={r['k2']} Cs={r['Cs']} Mmax={r['Mmax']:.2f} R14={R14s}")
