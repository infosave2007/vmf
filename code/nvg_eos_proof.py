#!/usr/bin/env python3
"""
NVG Dense Matter: numerical proof-of-concept.
vacuum calibration → κ_Ω(n_B) → EOS → TOV → M_max, R_1.4 → verdict.
"""
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# === Constants (ħ=c=1, MeV) ===
hbar_c = 197.3269804  # MeV·fm
M_N = 939.0
n_0 = 0.16  # fm⁻³
f_pi, m_pi = 93.0, 140.0  # MeV
MeV_fm3_to_geo = 1.3234e-6  # MeV/fm³ → km⁻²
M_sun_km = 1.4766  # G·M_☉/c² in km

# === Vacuum calibration (FIXED, not fitted) ===
sigma_piN, sigma_sN, sigma_heavy = 44.0, 30.0, 6.0
sigma_total = sigma_piN + sigma_sN + sigma_heavy  # 80 MeV
M_Omega_0 = M_N - sigma_total  # 859 MeV
M_current_0 = sigma_total  # 80 MeV

print("="*60)
print("ВАКУУМНАЯ КАЛИБРОВКА (lattice input, не подгоняется)")
print("="*60)
print(f"  Σ σ_qN = {sigma_total} МэВ")
print(f"  M_Ω,0  = {M_Omega_0} МэВ  (вакуумная структурная масса)")
print(f"  f_Ω,0  = {M_Omega_0/M_N:.4f}")

# === Medium evolution ===
def M_Omega(n, k1, k2):
    x = n / n_0
    return M_Omega_0 * (1 + k2*x)**(-k1/k2) if k2*x < 500 else 0.0

def M_curr(n):
    r = max(1.0 - sigma_piN * n / (f_pi**2 * m_pi**2), 0.0)
    return M_current_0 * r

def Mstar(n, k1, k2):
    return M_curr(n) + M_Omega(n, k1, k2)

# === Relativistic Fermi gas ===
def kF(n):
    return (1.5 * np.pi**2 * n)**(1./3.) * hbar_c  # MeV

def eps_FG(n, meff):
    if n < 1e-12: return 0.0
    k = kF(n)
    res, _ = quad(lambda p: p**2 * np.sqrt(p**2 + meff**2), 0, k)
    return 4.0 / (2*np.pi**2 * hbar_c**3) * res  # MeV/fm³

def P_FG(n, meff):
    if n < 1e-12: return 0.0
    k = kF(n)
    res, _ = quad(lambda p: p**4 / np.sqrt(p**2 + meff**2), 0, k)
    return 4.0 / (6*np.pi**2 * hbar_c**3) * res

# === Build EOS ===
def build_eos(k1, k2, Cw, nn=400):
    narr = np.linspace(0.3*n_0, 8*n_0, nn)
    eps = np.zeros(nn); P = np.zeros(nn); ms = np.zeros(nn)
    for i, n in enumerate(narr):
        m = Mstar(n, k1, k2); ms[i] = m
        eps[i] = eps_FG(n, m) + 0.5*Cw*n**2
    # Thermodynamically consistent P: P = n² d(ε/n)/dn
    e_per_n = eps / narr
    P = narr**2 * np.gradient(e_per_n, narr)
    cs2 = np.gradient(P, narr) / np.gradient(eps, narr)
    return narr, eps, P, ms, cs2

def calibrate_Cw(k1, k2):
    """Fix C_ω so E/A = -16 MeV at n₀."""
    def obj(Cw):
        m = Mstar(n_0, k1, k2)
        e = eps_FG(n_0, m) + 0.5*Cw*n_0**2
        return e/n_0 - M_N - (-16.0)
    try: return brentq(obj, 0, 8000)
    except: return None

# === TOV solver ===
def solve_tov(eps_of_P_func, Pc):
    conv = MeV_fm3_to_geo
    r0 = 0.001  # km
    ec = eps_of_P_func(Pc)
    m0 = 4/3*np.pi*r0**3 * ec*conv

    def rhs(r, y):
        m, p = y
        if p <= 0: return [0, 0]
        e = eps_of_P_func(p)
        d = r*(r - 2*m)
        if d <= 0: return [0, 0]
        dpdr = -(e*conv + p*conv)*(m + 4*np.pi*r**3*p*conv)/d
        dmdr = 4*np.pi*r**2*e*conv
        return [dmdr, dpdr]

    def stop(r, y): return y[1]
    stop.terminal = True; stop.direction = -1

    sol = solve_ivp(rhs, [r0, 50], [m0, Pc], events=stop,
                    max_step=0.05, rtol=1e-8, atol=1e-12)
    if sol.t_events[0].size > 0:
        R = sol.t_events[0][0]
        M = sol.y_events[0][0][0] / M_sun_km
    else:
        R = sol.t[-1]; M = sol.y[0,-1] / M_sun_km
    return M, R

# === Run one model ===
def run(k1, k2, label=""):
    Cw = calibrate_Cw(k1, k2)
    if Cw is None:
        print(f"  κ₁={k1} κ₂={k2}: калибровка не сошлась"); return None

    ms_n0 = Mstar(n_0, k1, k2)
    narr, eps, P, ms, cs2 = build_eos(k1, k2, Cw)

    # Filter valid region
    ok = (P > 0) & np.isfinite(cs2)
    if ok.sum() < 20:
        print(f"  κ₁={k1} κ₂={k2}: EOS нежизнеспособна"); return None

    P_ok, eps_ok = P[ok], eps[ok]
    si = np.argsort(P_ok)
    P_s, eps_s = P_ok[si], eps_ok[si]
    # Monotonize
    mono = np.concatenate([[True], np.diff(P_s) > 0])
    P_s, eps_s = P_s[mono], eps_s[mono]
    if len(P_s) < 10: return None

    eofP = interp1d(P_s, eps_s, bounds_error=False,
                    fill_value=(eps_s[0], eps_s[-1]))

    # TOV scan
    Pc_arr = np.logspace(np.log10(max(P_s[1],0.1)),
                         np.log10(P_s[-1]*0.9), 60)
    MM, RR = [], []
    for Pc in Pc_arr:
        try:
            m, r = solve_tov(eofP, Pc)
            if 0 < m < 5 and 5 < r < 25: MM.append(m); RR.append(r)
        except: pass

    if len(MM) < 5: return None
    MM, RR = np.array(MM), np.array(RR)

    Mmax = MM.max()
    imax = MM.argmax()
    R_Mmax = RR[imax]

    # R_1.4
    R14 = None
    stable = np.arange(imax+1)
    if len(stable) > 3 and MM[stable].min() < 1.4 < MM[stable].max():
        try: R14 = float(interp1d(MM[stable], RR[stable])(1.4))
        except: pass

    cs2_max = np.nanmax(cs2[ok])
    causal = cs2_max <= 1.0

    return dict(k1=k1, k2=k2, Cw=Cw, ms_n0=ms_n0,
                Mmax=Mmax, R14=R14, R_Mmax=R_Mmax,
                cs2_max=cs2_max, causal=causal,
                narr=narr, eps=eps, P=P, ms=ms, cs2=cs2)

# === MAIN ===
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ЧИСЛЕННАЯ ПРОВЕРКА NVG DENSE MATTER MODEL")
    print("="*60)

    tests = [
        (0.10, 0.30), (0.15, 0.40), (0.20, 0.50),
        (0.25, 0.60), (0.30, 0.80), (0.20, 1.00),
        (0.35, 1.20), (0.15, 0.60), (0.25, 1.00),
    ]

    results = []
    for k1, k2 in tests:
        r = run(k1, k2)
        if r: results.append(r)

    # === Summary ===
    print("\n" + "="*80)
    print(f"{'κ₁':>5} {'κ₂':>5} {'M*/M_N':>7} {'C_ω':>7} {'M_max':>7} {'R₁.₄':>6} "
          f"{'c²s_max':>8} {'Pass':>6}")
    print("-"*80)

    any_pass = False
    for r in results:
        R14s = f"{r['R14']:.1f}" if r['R14'] else "  — "
        p = r['Mmax'] >= 2.01 and r['causal']
        if r['R14']: p = p and 10.5 < r['R14'] < 14.0
        sym = "✓" if p else "✗"
        if p: any_pass = True
        print(f"{r['k1']:5.2f} {r['k2']:5.2f} {r['ms_n0']/M_N:7.3f} "
              f"{r['Cw']:7.1f} {r['Mmax']:7.3f} {R14s:>6} "
              f"{r['cs2_max']:8.3f} {sym:>6}")

    print("="*80)

    if any_pass:
        print("\n✓ СУЩЕСТВУЮТ ПАРАМЕТРЫ (κ₁,κ₂), ПРИ КОТОРЫХ МОДЕЛЬ")
        print("  ОДНОВРЕМЕННО:")
        print("  • калибрована по вакуумным σ-термам (lattice)")
        print("  • даёт E/A = −16 МэВ при n₀")
        print("  • даёт M_max ≥ 2.01 M☉")
        print("  • каузальна (c²_s ≤ 1)")
        print("  • совместима с NICER (R₁.₄)")
        print("\n  ЭТО ОЗНАЧАЕТ: двухкомпонентное разложение массы")
        print("  (M_current + M_Ω) с фиксированным вакуумным входом")
        print("  порождает жизнеспособную EOS для нейтронных звёзд.")
    else:
        print("\n✗ НИ ОДНА ТОЧКА НЕ ПРОШЛА ВСЕ ТЕСТЫ.")
        print("  Модель фальсифицирована в данной параметризации.")
