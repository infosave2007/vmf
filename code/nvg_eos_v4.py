#!/usr/bin/env python3
"""NVG Dense Matter v4: Testing the PRC Draft EOS explicitly."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq

hbar_c = 197.3269804; M_N = 939.0; n_0 = 0.16
MeV_fm3_to_geo = 1.3234e-6; M_sun_km = 1.4766
M_Omega_0 = 859.0; M_current_0 = 80.0

def M_Omega(n, k1, k2):
    x = n/n_0
    return M_Omega_0 * (1+k2*x)**(-k1/k2) if k2*x < 500 else 0.

def Mstar(n, k1, k2):
    mc = M_current_0 * max(1 - 44.0*n/(93.**2*140.**2), 0.)
    return mc + M_Omega(n, k1, k2)

def kF(n): 
    return (1.5*np.pi**2*n)**(1./3.)*hbar_c

def eps_fg(n, m):
    """Analytical relativistic Fermi gas energy density."""
    if n < 1e-12: return 0.
    k = kF(n); E = np.sqrt(k**2+m**2)
    g = 4; f = g/(16*np.pi**2*hbar_c**3)
    return f*(k*E*(2*k**2+m**2)-m**4*np.log((k+E)/max(m,1e-3)))

def build_eos(k1, k2, Cw, nn=400):
    narr = np.linspace(0.02*n_0, 8*n_0, nn)
    eps = np.zeros(nn)
    for i, n in enumerate(narr):
        m = Mstar(n, k1, k2)
        # Direct implementation of equation from the PRC draft:
        # ε(n_B) = ε_kin(n_B, M*) + (C_w/2)*n_B^2 + n_B*(M_Ω(n_B) - M_Ω,0)
        delta_V = n * (M_Omega(n, k1, k2) - M_Omega_0)
        eps[i] = eps_fg(n, m) + 0.5 * Cw * n**2 + delta_V
        
    P = narr**2 * np.gradient(eps/narr, narr)
    cs2 = np.gradient(P, narr) / np.gradient(eps, narr)
    return narr, eps, P, cs2

def calibrate(k1, k2):
    def obj(Cw):
        n = n_0; m = Mstar(n, k1, k2)
        delta_V = n * (M_Omega(n, k1, k2) - M_Omega_0)
        e = eps_fg(n, m) + 0.5 * Cw * n**2 + delta_V
        return e/n - M_N - (-16.)
    try: return brentq(obj, 0, 5000)
    except: return None

def tov(eofP, Pc):
    c = MeV_fm3_to_geo; r0 = .001; ec = eofP(Pc); m0 = 4/3*np.pi*r0**3*ec*c
    def rhs(r, y):
        m, p = y
        if p <= 0: return [0, 0]
        e = eofP(p); d = r*(r-2*m)
        if d <= 0: return [0, 0]
        return [4*np.pi*r**2*e*c, -(e*c+p*c)*(m+4*np.pi*r**3*p*c)/d]
    def stop(r, y): return y[1]
    stop.terminal = True; stop.direction = -1
    s = solve_ivp(rhs, [r0, 50], [m0, Pc], events=stop, max_step=.05, rtol=1e-8, atol=1e-12)
    if s.t_events[0].size > 0:
        return s.y_events[0][0][0]/M_sun_km, s.t_events[0][0]
    return s.y[0,-1]/M_sun_km, s.t[-1]

def run(k1, k2):
    Cw = calibrate(k1, k2)
    if Cw is None: return None
    narr, eps, P, cs2 = build_eos(k1, k2, Cw)
    
    ok = (P > 0) & np.isfinite(cs2) & (cs2 > 0)
    if ok.sum() < 20: return None
    Ps, es = P[ok], eps[ok]
    si = np.argsort(Ps); Ps, es = Ps[si], es[si]
    mo = np.concatenate([[True], np.diff(Ps) > 0]); Ps, es = Ps[mo], es[mo]
    if len(Ps) < 10: return None
    
    eofP = interp1d(Ps, es, bounds_error=False, fill_value=(es[0], es[-1]))
    Pc_arr = np.logspace(np.log10(max(Ps[1], .1)), np.log10(Ps[-1]*.8), 60)
    MM, RR = [], []
    for Pc in Pc_arr:
        try:
            m, r = tov(eofP, Pc)
            if 0 < m < 5 and 4 < r < 25: MM.append(m); RR.append(r)
        except: pass
    if len(MM) < 5: return None
    MM, RR = np.array(MM), np.array(RR)
    
    imax = MM.argmax()
    R14 = None
    st = np.arange(imax+1)
    if len(st) > 3 and MM[st].min() < 1.4 < MM[st].max():
        try: R14 = float(interp1d(MM[st], RR[st])(1.4))
        except: pass
        
    return dict(k1=k1, k2=k2, Cw=Cw, ms_n0=Mstar(n_0, k1, k2),
                Mmax=MM.max(), R14=R14, cs2_max=np.nanmax(cs2[ok]),
                causal=np.nanmax(cs2[ok])<=1.0)

if __name__ == '__main__':
    print("="*60)
    print("ТЕСТ СТАТЬИ PRC: ε = ε_kin + 0.5*Cw*n^2 + n*(M_Ω(n) - M_Ω,0)")
    print("="*60)
    
    results = []
    # Grid of parameters
    for k1 in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        for k2 in [0.1, 0.3, 0.5, 0.8, 1.2]:
            r = run(k1, k2)
            if r: results.append(r)
            
    print(f"\nВсего проверено моделей: {len(results)}")
    
    viable = [r for r in results if 2.01 <= r['Mmax'] <= 2.8 and r['causal']]
    
    print(f"Жизнеспособных (M_max ∈ [2.01, 2.8], каузальны): {len(viable)}\n")
    
    print(f"{'κ₁':>5} {'κ₂':>5} {'C_ω':>7} {'M*/M_N':>7} {'M_max':>7} {'R₁.₄':>6} {'c²s':>6}")
    print("-" * 60)
    
    if viable:
        viable.sort(key=lambda x: abs(x['Mmax'] - 2.15))
        for r in viable[:15]:
            r14_str = f"{r['R14']:6.1f}" if r['R14'] else "  —"
            print(f"{r['k1']:5.2f} {r['k2']:5.2f} {r['Cw']:7.1f} {r['ms_n0']/M_N:7.3f} {r['Mmax']:7.3f} {r14_str} {r['cs2_max']:6.3f}")
        
        print("\nУСПЕХ: Найдено точное подмножество параметров, дающих реалистичные массы звёзд.")
    else:
        results.sort(key=lambda x: x['Mmax'])
        for r in results[:15]:
            r14_str = f"{r['R14']:6.1f}" if r['R14'] else "  —"
            print(f"{r['k1']:5.2f} {r['k2']:5.2f} {r['Cw']:7.1f} {r['ms_n0']/M_N:7.3f} {r['Mmax']:7.3f} {r14_str} {r['cs2_max']:6.3f}")
        print("\nНЕТ ТОЧНЫХ СОВПАДЕНИЙ. Показаны ближайшие результаты (минимальная масса).")
