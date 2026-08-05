#!/usr/bin/env python3
"""
NVG Peltier Generator v2: RESONANT theta-HALOSCOPE (parameter-free redesign)
=============================================================================
The user's challenge: the old generator failed because of WRONG parameters and
a tuned gain G_param = 4.5e5. Can the NVG PRINCIPLE itself be made to work
with physically correct, UNTUNED parameters?

This script rebuilds the device from the repository's own anchor chain with
ZERO free parameters:

  (1) FREQUENCY is fixed by the theta-mode mass (from the theta-seesaw /
      neutrino batch):  m_theta = 53 ueV  ->  f_theta = m c^2 / h = 12.8 GHz.
      The 150 kHz PWM of the old design was 85000x off resonance; a real
      parametric/resonant device must sit AT f_theta (or 2*f_theta).

  (2) COUPLING is fixed by the fifth-force batch: the theta sector survives
      spin-dipole bounds at f_a = 1.07e11 GeV. The photon coupling is then
      the standard axion-electrodynamics one, g = (alpha/2pi)*(E/N-1.92)/f_a,
      with KSVZ/DFSZ model coefficients (no extra knob).

  (3) CONSISTENCY CHECK (no tuning): the two INDEPENDENT repo numbers,
      m_theta = 53 ueV and f_a = 1.07e11 GeV, must satisfy the QCD axion
      band  m_a * f_a ~ m_pi*f_pi*sqrt(z)/(1+z). We compute the mismatch.

  (4) ENERGY RESERVOIR: the vacuum condensate (rho_c = 7.09e4 MeV/fm^3) is
      the GROUND STATE - by definition no net work can be extracted from it
      (first/second law; this is the no-go theorem, not a parameter issue).
      The only REAL reservoir associated with the theta sector is the local
      dark-matter halo density rho_DM = 0.4 GeV/cm^3, whose theta quanta
      convert to photons in a strong B field (Sikivie haloscope principle).

  (5) HONEST POWER: signal power P = g^2 (rho/m) B^2 V C Q (natural units),
      compared with the quantum noise floor and the absolute DM-flux ceiling.

Verdict: the PRINCIPLE works - but as a resonant theta-dark-matter RECEIVER
(a genuine new NVG lab experiment), not as a power source. Numbers below.
"""
import math

# ----------------------------------------------------------------------------
# 0. Constants
# ----------------------------------------------------------------------------
HBAR = 1.054571817e-34        # J s
H_PLANCK = 6.62607015e-34     # J s
KB = 1.380649e-23             # J/K
EV = 1.602176634e-19          # J
GEV_TO_J = 1.602176634e-10
C = 2.99792458e8
ALPHA_EM = 1.0 / 137.035999

# ----------------------------------------------------------------------------
# 1. Repo anchor numbers (NO free parameters)
# ----------------------------------------------------------------------------
m_theta_ueV = 53.0            # from nvg_neutrinoless_dbeta / theta-seesaw
f_a_GeV = 1.07e11             # from nvg_theta_fifth_force (spin-dipole edge)
gamma_bound = 1.0e-4          # fifth-force survival bound

f_theta = m_theta_ueV * 1e-6 * EV / H_PLANCK       # Hz
omega_theta = 2.0 * math.pi * f_theta

# QCD axion band: m_pi f_pi sqrt(z)/(1+z), z = m_u/m_d = 0.477
m_pi_f_pi = 134.977e6 * 92.2e6        # eV^2
z_ud = 0.477
axion_band = m_pi_f_pi * math.sqrt(z_ud) / (1.0 + z_ud)   # eV^2
implied_fa = axion_band / (m_theta_ueV * 1e-6) / 1e9  # GeV if on band
mismatch = implied_fa / f_a_GeV

# Photon coupling g_theta-gamma (KSVZ..DFSZ range)
def g_theta_gamma(E_over_N):
    return ALPHA_EM / (2.0 * math.pi) * abs(E_over_N - 1.92) / f_a_GeV  # GeV^-1

g_ksvz = g_theta_gamma(0.0)     # |E/N - 1.92| = 1.92
g_dfsz = g_theta_gamma(8.0/3.0) # |E/N - 1.92| = 0.75

# ----------------------------------------------------------------------------
# 2. Energy reservoirs
# ----------------------------------------------------------------------------
# (a) vacuum condensate: 7.09e4 MeV/fm^3
rho_c_J_m3 = 7.09e4 * 1.602176634e-13 / 1e-45      # J/m^3
# (b) local dark matter halo
rho_dm_GeV_cm3 = 0.40
rho_dm_J_m3 = rho_dm_GeV_cm3 * GEV_TO_J / 1e-6     # J/m^3
n_theta = rho_dm_J_m3 / (m_theta_ueV * 1e-6 * EV)  # m^-3, occupation number
v_halo = 2.2e5                                     # m/s (virial velocity)
flux_dm = rho_dm_J_m3 * v_halo                     # W/m^2, kinetic mass flux

# ----------------------------------------------------------------------------
# 3. Haloscope signal power (natural units -> SI)
#    P = g^2 (rho/m) B^2 V C Q   [hbar=c=1], converted via 1 GeV^2 = 2.42e14 W
# ----------------------------------------------------------------------------
GEV2_TO_W = GEV_TO_J / (6.582119569e-25)           # 2.43e14 W per GeV^2
GEV3_TO_M3 = 1.3012e47                             # 1 m^3 in GeV^-3

def signal_power_W(g_GeV_inv, B_T, V_m3, Q, C_form=0.5):
    rho_GeV4 = rho_dm_GeV_cm3 * 1e6 / GEV3_TO_M3
    m_GeV = m_theta_ueV * 1e-6 * 1e-9
    B_GeV2 = B_T * 195.0e-18                       # 1 T = 195 eV^2
    P_nat = (g_GeV_inv**2) * (rho_GeV4 / m_GeV) * (B_GeV2**2) * (V_m3 * GEV3_TO_M3) * C_form * Q
    return P_nat * GEV2_TO_W

# Bench hardware (ADMX-class, nothing exotic):
B0 = 10.0          # T (NbTi/Nb3Sn solenoid)
V_lit = 1.0e-3     # m^3 (1 liter cavity, TM010 at 12.8 GHz)
Q_cav = 1.0e5      # loaded quality factor
C_form = 0.5       # TM010 form factor

P_sig_small = signal_power_W(g_ksvz, B0, V_lit, Q_cav, C_form)
P_sig_big = signal_power_W(g_ksvz, B0, 1.0, Q_cav, C_form)   # 1 m^3

# ----------------------------------------------------------------------------
# 4. Noise floor & radiometer integration
# ----------------------------------------------------------------------------
df_band = f_theta / Q_cav                  # matched bandwidth
T_quantum = H_PLANCK * f_theta / KB        # quantum noise temperature
T_sys = T_quantum                          # quantum-limited amplifier (JPA/SQUID)
P_noise = KB * T_sys * df_band

SNR_1s = (P_sig_small / P_noise) * math.sqrt(df_band * 1.0)
t_snr5 = (5.0 / SNR_1s) ** 2               # seconds to reach SNR = 5 (radiometer)

# ----------------------------------------------------------------------------
# 5. Absolute extraction ceiling
# ----------------------------------------------------------------------------
# Even converting ALL DM rest energy passing through a 1 m^2 aperture:
P_ceiling_flux = flux_dm * 1.0             # W per m^2
# The old claim for comparison:
P_old_claim = 300.0

print("=" * 78)
print("  NVG GENERATOR v2: RESONANT theta-HALOSCOPE (parameter-free redesign)")
print("=" * 78)
print("[1] RESONANCE (frequency fixed by the anchor, not chosen):")
print(f"  m_theta = {m_theta_ueV:.0f} ueV  ->  f_theta = {f_theta:.2e} Hz = {f_theta/1e9:.1f} GHz")
print(f"  old PWM frequency 150 kHz -> detuning {omega_theta/(2*math.pi*150e3):.1e}x (dead).")
print(f"  correct pump/receive frequency: {f_theta/1e9:.1f} GHz (X/Ku-band microwave).")
print("-" * 78)
print("[2] COUPLING (fixed by fifth-force f_a, model range KSVZ-DFSZ):")
print(f"  f_a = {f_a_GeV:.2e} GeV  ->  g_KSVZ = {g_ksvz:.2e} GeV^-1, g_DFSZ = {g_dfsz:.2e} GeV^-1")
print("-" * 78)
print("[3] CONSISTENCY CHECK (no tuning allowed):")
print(f"  QCD axion band m_a*f_a       : {axion_band:.3e} eV^2")
print(f"  repo m_theta * f_a           : {m_theta_ueV*1e-6 * f_a_GeV*1e9:.3e} eV^2")
print(f"  f_a implied by the band      : {implied_fa:.2e} GeV")
print(f"  ratio to fifth-force f_a     : {mismatch:.3f}  ({abs(1-mismatch)*100:.1f}% off)")
if abs(mismatch - 1.0) < 0.15:
    print("  -> the two INDEPENDENT repo constraints land ON the QCD axion band.")
print("-" * 78)
print("[4] ENERGY RESERVOIRS:")
print(f"  vacuum condensate rho_c      : {rho_c_J_m3:.2e} J/m^3  -- GROUND STATE,")
print("     net extraction forbidden (1st/2nd law). Not a reservoir.")
print(f"  local theta-DM halo rho      : {rho_dm_J_m3:.2e} J/m^3")
print(f"  occupation n_theta           : {n_theta:.2e} m^-3 (classical field)")
print(f"  DM mass-energy flux          : {flux_dm:.1f} W/m^2 (absolute ceiling,")
print("     requires unit conversion efficiency -- unattainable; see [6]).")
print("-" * 78)
print("[5] SIGNAL POWER, ADMX-class hardware (B = %.0f T, Q = %.0e):" % (B0, Q_cav))
print(f"  V = 1 liter, g = g_KSVZ      : P_sig = {P_sig_small:.2e} W")
print(f"  V = 1 m^3 (scaled)           : P_sig = {P_sig_big:.2e} W")
print(f"  matched bandwidth f/Q        : {df_band:.1f} Hz")
print(f"  quantum noise temperature    : {T_quantum:.2f} K")
print(f"  noise power kT*df            : {P_noise:.2e} W")
print(f"  SNR in 1 s                   : {SNR_1s:.2e}")
print(f"  integration to SNR = 5       : {t_snr5/3600:.1f} h")
print("-" * 78)
print("[6] VERDICT")
print("=" * 78)
print(f"""\
 THE PRINCIPLE WORKS -- as a RESONANT RECEIVER, not as a power source.

 (a) With zero tuned parameters the redesign lands on a real, existing
     experimental technique: the axion HALOSCOPE. The frequency (12.8 GHz)
     is fixed by m_theta, the coupling by f_a, and the two independent
     repo numbers sit on the QCD axion band within {abs(1-mismatch)*100:.0f}%. The expected
     signal {P_sig_small:.1e} W (1 L, 10 T) is within ADMX-class reach after
     ~{t_snr5/3600:.0f} h integration. This would be the FIRST laboratory experiment
     targeting the NVG theta sector directly -- a far stronger result for
     the framework than a heater.

 (b) As an ENERGY SOURCE it cannot work at ANY parameters, and this is a
     theorem, not an engineering gap:
       - the condensate is the ground state: extracting net work from it
         violates the first/second laws (the 'no-go' behind all audits);
       - the only real reservoir is the DM halo: even the ABSOLUTE ceiling
         (100% conversion of all DM mass-energy through 1 m^2) is
         {flux_dm:.0f} W/m^2, and the coupling-limited extractable power is the
         [5] number, {P_sig_small:.1e} W -- {P_old_claim/P_sig_small:.0e}x below the old 300 W claim.
     No frequency, material or geometry changes this: the ceiling is set
     by rho_DM and gamma_topo <= 1e-4, both measured/observed quantities.

 (c) What to build: replace the TEC stack + flyback with a 12.8 GHz TM010
     cavity inside a 10 T solenoid, quantum-limited amplifier, and run the
     radiometer protocol. The old 150 kHz Peltier hardware has no role:
     its frequency is 5 orders off resonance and its coupling channel
     (monopole) is identically zero.
""")
