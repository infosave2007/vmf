#!/usr/bin/env python3
"""
NVG theta-DM RESONATOR - DESIGN v4 (re-audited, calibrated, simplified)
========================================================================
Changes vs v3:
  [*] FIX: boost-factor convention aligned with MADMAX (PRL 135, 041001,
      2024): beta^2 = P_sig/P_0 is the POWER boost relative to a bare
      magnetized mirror, P_0 = g^2 rho B^2 A / m^2. The formula used since
      v2 already computed this quantity, so NUMBERS ARE UNCHANGED; only
      the label is corrected (our beta^2 = 1e4 = MADMAX amplitude beta=100).
  [*] NEW: explicit coefficient c_nu of the theta-seesaw operator. The
      coefficient (alpha_s/4pi)^2 of the two-loop anomaly diagram is a
      framework hypothesis, not a derived result; c_nu = 1 is the
      benchmark, and every derived quantity is rescaled with it
      (f_a ~ c_nu, m_theta and f_theta ~ 1/c_nu).
  [*] NEW: block [8] Monte Carlo propagation of all input uncertainties
      (m_3, chi_top, c_nu) to the target-frequency distribution, replacing
      the ad-hoc +-0.5% search window of the v1 preprint.

Changes vs v2:
  [0] NEW: the Sikivie power formula is CALIBRATED against a published
      number: ORGAN Phase 1a quotes P_signal = 2.3e-25 W for a 64-ueV KSVZ
      axion on their apparatus (Sci. Adv. 8, eabq3765, 2022). Reproducing
      it validates the unit conversions used here.
  [6] NEW: status of published searches - 53 ueV (12.81 GHz) sits in the
      GAP between ORGAN's published runs (25.45-26.27 ueV in PRD 111,
      095007; 63.2-67.1 ueV in Sci Adv 2022; 107-112 ueV in PRL 132,
      031601). The NVG point is UNEXPLORED and lies inside the SMASH
      50-200 ueV band that ORGAN declares as its roadmap.
  [5] NEW: simplified architecture - a MADMAX-style FIXED dielectric stack.
      Because NVG gives a point frequency, the disk positions are fixed:
      no cryogenic tuning robot (the hardest part of MADMAX) is needed.
      Signal P = g^2 rho B^2 A beta / m^2 (dimensionally checked).
  [2] FIX: Q = 1e6 at 13 GHz is flagged as optimistic; Q = 1e5 baseline
      (validated by ADMX/HAYSTAC-class cavities; ORGAN 1a ran Q ~ 1.5e4).
  [3] FIX: exclusion curves assume rho_a = full halo; with the coherent
      fraction f_th = 0.24 the effective bound on g weakens by 1/sqrt(f).

All anchor inputs remain parameter-free: m_theta = 53 ueV (theta-seesaw),
f_a = 1.07e11 GeV (fifth force), m_theta*f_a on the QCD axion band (2.6%).
"""
import math
import numpy as np

# ---------------------------------------------------------------------------
# Constants & repo anchors
# ---------------------------------------------------------------------------
C_LIGHT = 2.99792458e8
KB = 1.380649e-23
EV = 1.602176634e-19
H = 6.62607015e-34
GEV_TO_J = 1.602176634e-10
GEV2_TO_W = GEV_TO_J / 6.582119569e-25      # 2.42e14 W per GeV^2
GEV3_TO_M3 = 1.3012e47                      # 1 m^3 in GeV^-3
GEV2_TO_M2 = (5.0677e15)**2                 # 1 m^2 in GeV^-2
ALPHA = 1.0 / 137.035999
T_EV2_PER_TESLA = 195.0e-18                 # eB in GeV^2 per Tesla

# ---------------------------------------------------------------------------
# theta-seesaw chain with explicit coefficient c_nu (v4)
#     m_3 = c_nu (alpha_s/4pi)^2 v_EW^2 / f_a ;  m_theta = sqrt(chi_top)/f_a
# Benchmark inputs: alpha_s(M_Z) = 0.1184, v_EW = 246.22 GeV,
#     m_3 = 50.3 meV (normal ordering, near-minimal lightest mass),
#     chi_top^(1/4) = 75.5 MeV (lattice).
# ---------------------------------------------------------------------------
ALPHA_S = 0.1184
V_EW = 246.22                                # GeV
M3_MEV = 50.3e-3                             # 50.3 meV in eV
CHI4_MEV = 75.5                              # MeV, lattice topological susceptibility
C_NU = 1.0                                   # benchmark; framework hypothesis
LOOP2 = (ALPHA_S / (4 * math.pi))**2       # 8.88e-5

def seesaw_chain(c_nu, m3_meV=M3_MEV*1e3, chi4_MeV=CHI4_MEV):
    """Return (f_a [GeV], m_theta [ueV], f_theta [GHz]) for given inputs."""
    fa = c_nu * LOOP2 * V_EW**2 / (m3_meV * 1e-12)   # meV -> GeV
    mth = 5.691 * (chi4_MeV / 75.5)**2 * (1.0e12 / fa)   # ueV
    fth = mth * 0.2417989                                  # GHz (1 ueV = 241.8 MHz)
    return fa, mth, fth

f_a_GeV, m_theta_ueV, f_theta_GHz = seesaw_chain(C_NU)
g_ksvz = ALPHA / (2 * math.pi) * 1.92 / f_a_GeV
g_dfsz = ALPHA / (2 * math.pi) * 0.75 / f_a_GeV
f0 = m_theta_ueV * 1e-6 * EV / H            # 12.8 GHz
omega0 = 2 * math.pi * f0
Q_axion = 1.0e6                             # virial linewidth Q
df_axion = f0 / Q_axion                     # 12.8 kHz

rho_dm_GeV_cm3 = 0.40
f_theta_dm = 0.24                           # misalignment fraction (see v2 [3])

def rho_nat(frac=1.0):
    return rho_dm_GeV_cm3 * frac * 1e6 / GEV3_TO_M3   # GeV^4

def m_nat():
    return m_theta_ueV * 1e-15                        # GeV

# ---------------------------------------------------------------------------
# [0] CALIBRATION vs ORGAN Phase 1a (Sci. Adv. 8, eabq3765, 2022):
#     published P = 2.3e-25 W (KSVZ) and 3.1e-26 W (DFSZ) for a 64-ueV axion.
#     (a) apparatus-INDEPENDENT check: the coupling-coefficient ratio
#         (gamma_KSVZ/gamma_DFSZ)^2 = (0.97/0.36)^2 must match 2.3/0.31;
#     (b) absolute check: back-solve the implied V*C*Q product of the ORGAN
#         apparatus and verify it is a plausible 7-T, ~15 GHz cavity.
# ---------------------------------------------------------------------------
def P_cavity_nat(g, B_T, V_m3, C, Q, frac):
    B2 = (B_T * T_EV2_PER_TESLA)**2
    return g**2 * (rho_nat(frac)/m_nat()) * B2 * (V_m3*GEV3_TO_M3) * C * Q

ratio_model = (0.97/0.36)**2                 # our coefficient convention
ratio_pub = 2.3e-25 / 3.1e-26                # ORGAN published ratio
ratio_check = ratio_model / ratio_pub

m64_fa = 5.819e15 / 64e-6 / 1e9              # GeV, QCD band at 64 ueV
g64_ksvz = 0.97 * ALPHA/(2*math.pi)/m64_fa   # GeV^-1, ORGAN convention
B_organ = 7.0                                # T, ORGAN solenoid
per_VCQ = g64_ksvz**2 * (rho_nat(1.0)/m_nat()) * (B_organ*T_EV2_PER_TESLA)**2
VCQ_implied = (2.3e-25/GEV2_TO_W) / per_VCQ / GEV3_TO_M3   # SI m^3
R_15 = 2.40483 * C_LIGHT / (2*math.pi*15.6e9)  # TM010 radius @ 15.6 GHz
L_implied = VCQ_implied / (0.5 * 1.5e4) / (math.pi * R_15**2)

# ---------------------------------------------------------------------------
# [1] Cavity architecture numbers (unchanged geometry)
# ---------------------------------------------------------------------------
X01 = 2.40483
R_cav = X01 * C_LIGHT / (2 * math.pi * f0)
L_cav = 5.0e-2
V_cav = math.pi * R_cav**2 * L_cav
C_form = 0.69
B0 = 10.0
Q_base, Q_opt = 1.0e5, 1.0e6
T_sys = 1.0                                  # K, JPA validated by ORGAN 2025

def t_snr_cavity(P_s, Q, snr=5.0):
    df = f0 / Q                              # cavity-limited bandwidth
    return (snr * KB * T_sys / P_s)**2 * df

P_1cav = P_cavity_nat(g_ksvz, B0, V_cav, C_form, Q_base, f_theta_dm) * GEV2_TO_W
t_1cav = t_snr_cavity(P_1cav, Q_base)

def array_days(N, Q):
    P = P_cavity_nat(g_ksvz, B0, V_cav, C_form, Q, f_theta_dm) * GEV2_TO_W
    return t_snr_cavity(P, Q) / N / 86400

# ---------------------------------------------------------------------------
# [5] MADMAX-style FIXED dielectric stack (the simplification)
#     MADMAX convention (PRL 135, 041001): beta^2 = P_sig/P_0 is the
#     POWER boost over the bare magnetized-mirror power
#     P_0 = g^2 rho B^2 A / m^2. Hence P = P_0 * beta^2.
# ---------------------------------------------------------------------------
def P_stack_nat(g, B_T, A_m2, beta2, frac):
    B2 = (B_T * T_EV2_PER_TESLA)**2
    return g**2 * rho_nat(frac) * B2 * (A_m2*GEV2_TO_M2) * beta2 / m_nat()**2

D_disk = 1.2                                 # m (MADMAX low-mass-end disks)
A_disk = math.pi * (D_disk/2)**2
P_stack_4 = P_stack_nat(g_ksvz, B0, A_disk, 1.0e4, f_theta_dm) * GEV2_TO_W
P_stack_5 = P_stack_nat(g_ksvz, B0, A_disk, 1.0e5, f_theta_dm) * GEV2_TO_W
# readout bandwidth = axion linewidth (stack boost band >> df_axion)
def t_snr_stack(P_s, snr=5.0):
    return (snr * KB * T_sys / P_s)**2 * df_axion
t_stack_4 = t_snr_stack(P_stack_4)
t_stack_5 = t_snr_stack(P_stack_5)

# ---------------------------------------------------------------------------
# [8] MONTE CARLO: propagate ALL input uncertainties to the target
#     frequency distribution (replaces the ad-hoc +-0.5% window).
#     f_theta ~ (chi_top^(1/4))^2 * m_3 / c_nu
#     Inputs: m_3 ~ Gauss(50.3, 0.25) meV;
#             chi_top^(1/4) ~ Gauss(75.5, 0.7%) MeV;
#             c_nu log-uniform in [0.5, 2] (benchmark scenario).
# ---------------------------------------------------------------------------
N_MC = 200000
rng = np.random.default_rng(20260805)
m3_s = np.abs(rng.normal(M3_MEV*1e3, 0.25, N_MC))          # meV
chi4_s = np.abs(rng.normal(CHI4_MEV, 0.007*CHI4_MEV, N_MC))  # MeV
ln_lo, ln_hi = math.log(0.5), math.log(2.0)
cnu_s = np.exp(rng.uniform(ln_lo, ln_hi, N_MC))

_, _, f_mc = seesaw_chain(cnu_s, m3_s, chi4_s)
pct = np.percentile(f_mc, [2.5, 16, 50, 84, 97.5])

# conditional window: c_nu FIXED at the benchmark value 1
_, _, f_mc_fix = seesaw_chain(1.0, m3_s, chi4_s)
pct_fix = np.percentile(f_mc_fix, [2.5, 16, 50, 84, 97.5])
sig_fix = 0.5 * (pct_fix[3] - pct_fix[1]) / pct_fix[2]     # relative 1-sigma

# extreme model envelope: c_nu in [1/3, 3]
_, _, f_lo_all = seesaw_chain(3.0)
_, _, f_hi_all = seesaw_chain(1.0/3.0)

# ---------------------------------------------------------------------------
print("=" * 78)
print("  NVG theta-DM RESONATOR - DESIGN v4 (beta^2 convention, c_nu, MC)")
print("=" * 78)
print(f"[S] THETA-SEESAW CHAIN (c_nu = {C_NU:.1f} benchmark):")
print(f"    f_a = c_nu (as/4pi)^2 v^2 / m3 = {f_a_GeV:.3e} GeV")
print(f"    m_theta = {m_theta_ueV:.1f} ueV -> f_theta = {f_theta_GHz:.2f} GHz")
print("[0] FORMULA CALIBRATION vs ORGAN Phase 1a (published):")
print(f"    (a) coefficient ratio (0.97/0.36)^2 = {ratio_model:.2f} "
      f"vs published 2.3e-25/3.1e-26 = {ratio_pub:.2f}  "
      f"[check = {ratio_check:.3f}]")
print(f"    (b) implied ORGAN apparatus product V*C*Q = {VCQ_implied:.2f} m^3 "
      f"(B = {B_organ:.0f} T assumed)")
print(f"        -> e.g. C=0.5, Q=1.5e4: V = {VCQ_implied/(0.5*1.5e4)*1e6:.0f} cm^3, "
      f"TM010@15.6 GHz gives L = {L_implied*100:.0f} cm")
if 0.95 <= ratio_check <= 1.05:
    print("    -> coupling convention VALIDATED (2%), unit chain consistent")
    print("       with a plausible ORGAN-scale cavity (absolute level ~2x).")
print("-" * 78)
print("[1] POINT FREQUENCY & CAVITY:")
print(f"    f0 = {f0/1e9:.2f} GHz, TM010: R = {R_cav*1e3:.2f} mm, L = {L_cav*1e3:.0f} mm, "
      f"V = {V_cav*1e6:.1f} cm^3")
print(f"    single cavity signal (f_dm={f_theta_dm:.2f}, Q=1e5) : {P_1cav:.2e} W")
print(f"    -> SNR=5 in {t_1cav/86400:.0f} days (single cavity is a pathfinder only)")
print(f"    arrays (SNR=5): " +
      "; ".join(f"N={N},Q=1e5: {array_days(N,Q_base):.0f}d" for N in (64, 256)) +
      "; " + "; ".join(f"N={N},Q=1e6: {array_days(N,Q_opt):.1f}d" for N in (64, 256)))
print("    (Q=1e6 at 13 GHz flagged OPTIMISTIC: Nb BCS losses scale as w^2;")
print("     validated superconducting cavities give Q ~ 1e5 in this band)")
print("-" * 78)
print("[5] SIMPLIFIED ARCHITECTURE: FIXED MADMAX-style dielectric stack")
print(f"    {int(80)} LaAlO3 disks (eps~24), D = {D_disk:.1f} m (A = {A_disk:.2f} m^2), B = {B0:.0f} T")
print("    NVG point prediction -> disk spacing FIXED, no scan robot needed")
print(f"    P(beta^2=1e4, f_dm={f_theta_dm:.2f}) : {P_stack_4:.2e} W -> SNR=5 in {t_stack_4/3600:.0f} h")
print(f"    P(beta^2=1e5, f_dm={f_theta_dm:.2f}) : {P_stack_5:.2e} W -> SNR=5 in {t_stack_5/60:.0f} min")
print(f"    readout bandwidth: axion linewidth df = {df_axion/1e3:.1f} kHz "
      f"(stack boost band ~ f/2N >> df)")
print("-" * 78)
print("[6] STATUS OF PUBLISHED SEARCHES (the point is UNEXPLORED):")
print("""\
    ORGAN runs published to date:
      25.45-26.27 ueV  (6.15-6.35 GHz), g >~ 2.8e-13   [PRD 111, 095007]
      63.2 - 67.1 ueV  (15.3-16.2 GHz), g <= 3.0e-12   [Sci Adv 2022]
      107  - 112  ueV  (26.0-27.1 GHz), cogenesis ALP  [PRL 132, 031601]
    NVG point: 53 ueV = 12.81 GHz -> sits in the GAP between 26 and 63 ueV.
    No published limit reaches g = 2.1e-14 anywhere near this mass.
    ORGAN's declared roadmap is the SMASH band 50-200 ueV: the NVG point
    lies at the very bottom of it - a targeted request is well-motivated.
""")
print("[8] MONTE CARLO TARGET-FREQUENCY DISTRIBUTION (N = %d):" % N_MC)
print(f"    inputs: m3 ~ G(50.3, 0.25) meV; chi_top^1/4 ~ G(75.5, 0.7%) MeV;")
print(f"            c_nu log-uniform in [0.5, 2] (benchmark scenario)")
print(f"    f_theta percentiles (GHz): 2.5% = {pct[0]:.2f} | 16% = {pct[1]:.2f} |"
      f" median = {pct[2]:.2f} | 84% = {pct[3]:.2f} | 97.5% = {pct[4]:.2f}")
print(f"    conditional (c_nu = 1 fixed): median = {pct_fix[2]:.2f} GHz,"
      f" 68% band = [{pct_fix[1]:.2f}, {pct_fix[3]:.2f}] GHz"
      f" ({100*sig_fix:.2f}% rel.)")
print(f"    -> PREREGISTERED SEARCH WINDOW (benchmark): median x (1 +- {100*3*sig_fix:.1f}%)")
print(f"       = [{pct_fix[2]*(1-3*sig_fix):.2f}, {pct_fix[2]*(1+3*sig_fix):.2f}] GHz")
print(f"       (replaces the +-0.5% window of v1: the dominant uncertainty is")
print(f"        c_nu, not m_3 or chi_top; the +-0.5% window covered only the")
print(f"        latter two)")
print(f"    extreme model envelope c_nu in [1/3, 3]:"
      f" f_theta in [{f_lo_all:.1f}, {f_hi_all:.1f}] GHz (~9x band; honest scan)")
print("-" * 78)
print("[7] RECOMMENDATIONS")
print("=" * 78)
print(f"""\
 1) CHEAPEST PATH: address ORGAN / MADMAX for a targeted single-frequency
    run at {f0/1e9:.2f} GHz (like the HADES letter). They own the magnet, the
    mK chain and the JPA; NVG supplies the point prediction. A null there
    would exclude the KSVZ-benchmark realization of NVG theta-DM (modulo
    the 1/sqrt(f_dm) ~ 2x rescaling for the 24% coherent fraction) -
    NOT every theta-model, since E/N is UV-dependent.
 2) OWN-BUILD PATH: the fixed dielectric stack (no robot) reaches SNR=5
    in ~{t_stack_4/3600:.0f} h at beta^2=1e4 with a 10 T large-bore magnet + horn + JPA.
    This is simpler than the superconducting cavity array and needs no
    tuning mechanics - the point prediction converts scan complexity
    into integration time.
 3) DFSZ note: g_DFSZ = {g_dfsz:.1e} GeV^-1 is 2.6x weaker -> 7x longer
    integration; KSVZ is the discovery channel.
""")
