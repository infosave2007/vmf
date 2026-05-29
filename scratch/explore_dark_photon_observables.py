import math

# Physical constants (CGS)
c = 2.99792458e10           # cm/s, speed of light
hbar = 1.0545718e-27         # erg s, Planck constant
alpha_EM = 1.0 / 137.036     # EM coupling
G_gravity = 6.6743e-8        # dyn cm^2/g^2, Gravitational constant
M_sun = 1.98847e33           # g, solar mass

# Benchmark Dark Photon Parameters
eps_0 = 1.0
m_A_vac = 1.16               # GeV

# Conversion factors: GeV to CGS
GeV_to_erg = 1.60217663e-3
GeV_inv_to_cm = 1.9732698e-14
GeV_inv_to_s = 6.58211956e-25

# S factor helper
def S_factor(r: float) -> float:
    return max(0.0, 1.0 - r**2)

# 1. Dilepton excess in central/peripheral collisions
def dilepton_excess_ratio(r_central: float) -> float:
    S_cent = S_factor(r_central)
    eps_cent = eps_0 * S_cent
    C = 1.0
    return 1.0 + C * (eps_cent**2) / alpha_EM

# 2. Realistic SN1987A energy loss calculation via NN -> N N A'
def energy_loss_rate_bremsstrahlung(m_A_med_GeV: float, T_MeV: float, eps_eff: float) -> float:
    """Calculate energy loss rate (erg / cm^3 / s) via nucleon-nucleon bremsstrahlung.
    Includes proper threshold phase-space suppression and numerical factors.
    """
    T_GeV = T_MeV * 1e-3
    m_N = 0.939 # GeV, nucleon mass
    
    # Bremsstrahlung scaling:
    # Q ~ (eps_eff^2 * alpha_EM) * (g_pi_N^2 / 4pi)^2 * (T^6 / m_N^2) * (T / m_A)^1.5 * exp(-m_A/T)
    # in GeV^5. Here g_pi_N^2 / 4pi ~ 14 is the strong coupling.
    g_pi_N_coupling = 14.0
    
    exp_factor = math.exp(-m_A_med_GeV / T_GeV) if m_A_med_GeV / T_GeV < 100 else 0.0
    # Phase space / threshold factor (T/m_A)^1.5
    ps_factor = (T_GeV / m_A_med_GeV)**1.5 if m_A_med_GeV > 0 else 0.0
    
    Q_nat = (eps_eff**2) * alpha_EM * (g_pi_N_coupling**2) * (T_GeV**6 / (m_N**2)) * ps_factor * exp_factor
    
    # Convert GeV^5 to erg / (cm^3 * s)
    val_erg = Q_nat * GeV_to_erg
    val_vol_time = val_erg / ((GeV_inv_to_cm**3) * GeV_inv_to_s)
    return val_vol_time

def calculate_sn_luminosity(m_A_med_GeV: float, T_MeV: float, r_core: float) -> float:
    S_sn = S_factor(r_core)
    eps_eff = eps_0 * S_sn
    
    Q = energy_loss_rate_bremsstrahlung(m_A_med_GeV, T_MeV, eps_eff)
    V_core = (4.0 / 3.0) * math.pi * (1.2e6)**3 # 12 km core radius (cm^3)
    return Q * V_core

# 3. Magnetic deformation (ellipticity)
def calculate_ellipticity(B_surf: float, r_center: float) -> tuple[float, float, float]:
    S_cent = S_factor(r_center)
    mu_ratio = 1.0 - S_cent
    B_core = B_surf / max(mu_ratio, 0.01)
    
    # Standard magnetic ellipticity: epsilon_ellip ~ 1e-6 * (B_core / 1e15 G)^2
    epsilon_ellip = 1e-6 * (B_core / 1e15)**2
    
    # GW strain at 10 kpc (d = 3.086e22 cm), f = 100 Hz
    d_cm = 3.086e22
    f_rot = 100.0
    I_NS = 0.35 * (1.4 * M_sun) * (1.2e6)**2
    h_0 = (16.0 * math.pi**2 * G_gravity * I_NS * f_rot**2 * epsilon_ellip) / (c**4 * d_cm)
    
    return B_core, epsilon_ellip, h_0

def main():
    print("=" * 80)
    print("    NVG DENSITY-ACTIVATED DARK PHOTON OBSERVABLES & CONFIRMATION")
    print("=" * 80)
    
    print("\n1. DILEPTON EXCESS IN HEAVY-ION COLLISIONS (HADES/FAIR)")
    print("-" * 80)
    print(f"Target Dilepton invariant mass bin: M_ee ~ {m_A_vac:.2f} GeV")
    print(f"{'Collision Type':<25} | {'rho/n0':<12} | {'r = sigma/s0':<15} | {'Dilepton Excess Ratio':<22}")
    print("-" * 80)
    
    scenarios = [
        ("Peripheral (vacuum-like)", 0.0, 1.0),
        ("Semicentral", 1.0, 0.801),
        ("Central (HADES max)", 2.0, 0.312),
        ("Supercentral (CBM/FAIR)", 3.0, 0.145)
    ]
    for name, rho, r in scenarios:
        exc = dilepton_excess_ratio(r)
        print(f"{name:<25} | {rho:<12.1f} | {r:<15.3f} | {exc:.3f}x")
        
    print("\n2. SN1987A COOLING CONSTRAINTS ON IN-MEDIUM MASS SHIFT (CORRECTED BREMSSTRAHLUNG)")
    print("-" * 80)
    print("SN1987A Core Temperature T = 30 MeV, Core density rho ~ n0 (r = 0.801)")
    print("Allowed energy loss luminosity limit: L_loss < 3.00e+52 erg/s")
    print("-" * 80)
    print(f"{'In-Medium Mass m_A*(rho) (GeV)':<32} | {'Mass Drop (%)':<15} | {'Emission Luminosity L_A (erg/s)':<30}")
    print("-" * 80)
    
    mass_values = [1.16, 0.80, 0.50, 0.30, 0.20, 0.15]
    for m in mass_values:
        drop = (1.0 - m / m_A_vac) * 100.0
        L_A = calculate_sn_luminosity(m, 30.0, 0.801)
        status = "ALLOWED" if L_A < 3.0e52 else "EXCLUDED (Too much cooling!)"
        print(f"{m:<32.2f} | {drop:<15.1f} | {L_A:.3e} erg/s ({status})")
        
    print("\n3. NS MAGNETIC DEFORMATION & CONTINUOUS GRAVITATIONAL WAVES")
    print("-" * 80)
    print("For a canonical 1.4 M_sun NS at distance d = 10 kpc, spin frequency f = 100 Hz")
    print("-" * 80)
    print(f"{'Surface Field B_surf (G)':<25} | {'Core Field B_core (G)':<25} | {'Ellipticity epsilon':<20} | {'GW Strain h_0':<15}")
    print("-" * 80)
    
    b_surfs = [1.0e12, 1.0e13, 1.0e14, 3.0e14]
    for b in b_surfs:
        b_core, eps_ellip, h_0 = calculate_ellipticity(b, 0.312)
        print(f"{b:25.1e} | {b_core:25.1e} | {eps_ellip:20.2e} | {h_0:15.2e}")
    print("=" * 80)

if __name__ == "__main__":
    main()
