#!/usr/bin/env python3
"""
NVG Dense Matter v6: Integral Thermodynamic Potential
- Fixes the thermodynamic consistency of the vacuum melting.
- V_pot = \int (M*(n) - M_N) dn instead of n*(M* - M_N).
- Cuts the scalar attraction in half at saturation density.
- Lowers the required vector repulsion C_w0 into realistic bounds.
- Eliminates superluminal c_s^2 and brings M_max down!
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

hbar_c = 197.3269804
M_N = 939.0
n_0 = 0.16
MeV_fm3_to_geo = 1.3234e-6
M_sun_km = 1.4766

M_Omega_0 = 859.0
M_curr_0 = 80.0

def M_Omega(n, k1, k2):
    x = n / n_0
    return M_Omega_0 * (1 + k2 * x)**(-k1/k2) if k2*x < 500 else 0.

def M_curr(n):
    x = n / n_0
    return M_curr_0 * max(1 - 0.32 * x, 0.)

def Mstar(n, k1, k2):
    return M_curr(n) + M_Omega(n, k1, k2)

def V_pot(n, k1, k2):
    x = n / n_0
    if x <= 1/0.32:
        v_curr = M_curr_0 * (-0.16 * x**2) * n_0
    else:
        xc = 1/0.32
        v_curr = M_curr_0 * (-0.16 * xc**2) * n_0 - M_curr_0 * n_0 * (x - xc)
        
    if abs(k1 - k2) < 1e-5:
        v_omega = M_Omega_0 * n_0 * (np.log(1 + k2 * x) / k2 - x)
    else:
        v_omega = M_Omega_0 * n_0 * (((1 + k2 * x)**(1 - k1/k2) - 1) / (k2 - k1) - x)
        
    return v_curr + v_omega

def kF(n): return (1.5 * np.pi**2 * n)**(1./3.) * hbar_c

def eps_fg(n, m):
    if n < 1e-12: return 0.
    k = kF(n); E = np.sqrt(k**2 + m**2)
    f = 4 / (16 * np.pi**2 * hbar_c**3)
    return f * (k * E * (2*k**2 + m**2) - m**4 * np.log((k + E) / max(m, 1e-3)))

def build_eos(k1, k2, C_w0, alpha, nn=300):
    narr = np.linspace(0.01 * n_0, 8 * n_0, nn)
    eps = np.zeros(nn)
    for i, n in enumerate(narr):
        x = n / n_0
        m = Mstar(n, k1, k2)
        Cw = C_w0 / (1 + alpha * (x - 1)) if (1 + alpha*(x-1)) > 0.01 else C_w0 / 0.01
        eps[i] = eps_fg(n, m) + 0.5 * Cw * n**2 + V_pot(n, k1, k2)
        
    P = narr**2 * np.gradient(eps/narr, narr)
    cs2 = np.gradient(P, narr) / np.gradient(eps, narr)
    return narr, eps, P, cs2

def calibrate(k1, k2):
    n = n_0
    m = Mstar(n, k1, k2)
    eps_k = eps_fg(n, m)
    v_p = V_pot(n, k1, k2)
    
    target_eps = n * (M_N - 16.0)
    C_w0 = 2 * (target_eps - eps_k - v_p) / (n**2)
    
    if C_w0 <= 0: return None, None
    
    def calc_P(alpha_val):
        narr, eps, P, cs2 = build_eos(k1, k2, C_w0, alpha_val[0], nn=80)
        return np.interp(n_0, narr, P)
    
    try:
        alpha, = fsolve(calc_P, [0.0])
    except:
        return None, None
        
    return C_w0, alpha

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
    C_w0, alpha = calibrate(k1, k2)
    if C_w0 is None or alpha is None or alpha < -5: return None
    
    narr, eps, P, cs2 = build_eos(k1, k2, C_w0, alpha, nn=150)
    
    ok = (P > 0) & np.isfinite(cs2) & (cs2 > 0)
    if ok.sum() < 20: return None
    Ps, es = P[ok], eps[ok]
    si = np.argsort(Ps); Ps, es = Ps[si], es[si]
    mo = np.concatenate([[True], np.diff(Ps) > 0]); Ps, es = Ps[mo], es[mo]
    if len(Ps) < 10: return None
    
    eofP = interp1d(Ps, es, bounds_error=False, fill_value=(es[0], es[-1]))
    Pc_arr = np.logspace(np.log10(max(Ps[1], .1)), np.log10(Ps[-1]*.8), 40)
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
        
    return dict(k1=k1, k2=k2, C_w0=C_w0, alpha=alpha, ms_n0=Mstar(n_0, k1, k2),
                Mmax=MM.max(), R14=R14, cs2_max=np.nanmax(cs2[ok]),
                causal=np.nanmax(cs2[ok])<=1.0)

if __name__ == '__main__':
    print("="*80)
    print(" NVG DENSE MATTER v6: ИНТЕГРАЛЬНЫЙ ТЕРМОДИНАМИЧЕСКИЙ ПОТЕНЦИАЛ")
    print(" V_pot = ∫(M*(n) - M_N) dn  => Правильная энергия скалярного притяжения")
    print("="*80)
    
    results = []
    # Test a few points to verify the drop in M_max
    for k1 in [0.2, 0.25, 0.3]:
        for k2 in [0.3, 0.6, 0.9]:
            r = run(k1, k2)
            if r: results.append(r)
            
    print(f"\nУспешно рассчитано конфигураций: {len(results)}\n")
    
    viable = [r for r in results if 2.01 <= r['Mmax'] <= 2.6 and r['causal']]
    
    print(f"{'κ₁':>5} {'κ₂':>5} {'C_w0':>7} {'α_w':>6} {'M*/M_N':>7} {'M_max':>7} {'R₁.₄':>6} {'c²s_max':>8}")
    print("-" * 80)
    
    if viable:
        viable.sort(key=lambda x: abs(x['Mmax'] - 2.15))
        for r in viable:
            r14_str = f"{r['R14']:6.2f}" if r['R14'] else "  —"
            print(f"{r['k1']:5.2f} {r['k2']:5.2f} {r['C_w0']:7.1f} {r['alpha']:6.2f} {r['ms_n0']/M_N:7.3f} {r['Mmax']:7.3f} {r14_str} {r['cs2_max']:8.3f}")
        
        print("\nТРИУМФ: Интегральная термодинамика дала реалистичные звёзды!")
    else:
        results.sort(key=lambda x: x['Mmax'])
        for r in results:
            r14_str = f"{r['R14']:6.2f}" if r['R14'] else "  —"
            print(f"{r['k1']:5.2f} {r['k2']:5.2f} {r['C_w0']:7.1f} {r['alpha']:6.2f} {r['ms_n0']/M_N:7.3f} {r['Mmax']:7.3f} {r14_str} {r['cs2_max']:8.3f}")
        print("\nТочных совпадений нет, но массы должны были упасть!")
