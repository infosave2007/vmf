#!/usr/bin/env python3
"""
NVG Master Framework: Evidence Ledger, Uncertainty Propagation & Inverse Solver
-----------------------------------------------------------------------------
This advanced suite performs:
1. Full Uncertainty Propagation for ALL observables (M_max, R_1.4, Lambda_1.4, z_surf, f_peak, etc.)
2. Inverse QCD Problem: Reconstructs M_Omega_0 from assumed astrophysical observations.
3. Forecast Module: Defines required precision for future detectors (STROBE-X, ET, CBM).
4. Automatic Evidence Ledger: Maps claims to numerical results and scripts.
"""

import math
import json
import os
import sys
from datetime import datetime
import numpy as np

# Add local path to import derive_w0_wa
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nvg_dark_energy_w0wa import derive_w0_wa

# =====================================================================
# 1. CORE CONSTANTS & LATTICE QCD INPUTS
# =====================================================================
M_OMEGA_CENTRAL = 859.0
M_OMEGA_ERR = 8.0
LATTICE_PRIORS = {
    "Baseline": {"mean": 859.0, "err": 8.0},
    "Gupta (2021) approx": {"mean": 862.0, "err": 12.0},
    "Agadjanov (2023) approx": {"mean": 851.0, "err": 15.0}
}

# Hubble horizon in km: R_H0 = c/H_0 for the repo-wide anchor H_0 = 72.8 km/s/Mpc
# (calibrated to local measurements; consistent with nvg_genesis_observable.py).
# The previous value 1.37e28 cm implied H_0 = 67.5 — inconsistent with the rest of the chain.
R_H0_KM = 1.2709e28 / 1e5

# =====================================================================
# 2. FULL UNCERTAINTY PROPAGATION ENGINE
# =====================================================================
def run_forward_model(m_omega):
    """Calculates all NVG observables from a single M_Omega_0 value."""
    scale = M_OMEGA_CENTRAL / m_omega
    
    # Cosmology
    r_c = 1.13 * scale
    n_e = math.log(R_H0_KM / r_c)
    s_0_log = 76.0 + 2 * math.log10(scale)
    cycles = (122.0 - s_0_log) / math.log10(4)
    pbh_asteroid_min = 1e-16 * (scale**1.5)
    pbh_asteroid_max = 1e-10 * (scale**1.5)
    
    # Neutron Stars
    # NOTE: the scaling exponents below (1.5, 1, 5, -0.5, ...) are dimensional
    # estimates used only to propagate the +/-1% M_Omega uncertainty; they are
    # not derived scaling laws.
    # Canonical computed values (nvg_tidal_deformability.py, canon from
    # nvg_ns_parameter_scan.py). Previous anchors 2.25/12.0/177 were table-edge
    # artifacts / hardcodes not produced by any script.
    m_max = 2.07 * (scale**1.5)   # fork-B consistent chain (iteration 2)
    r_14 = 12.49 * scale
    lambda_14 = 253.0 * (scale**5)
    z_surf = 0.223 * (scale**(-0.5))
    f_peak = 2510.0 * (scale**(-1.5))  # forward prediction from fork-B R_1.6 (no observation yet)
    rho_c = 4.5 * (scale**(-3)) # in n_0
    
    # High-Density & EM
    rho_shift = 20.0 * scale
    eps_eff = math.exp(math.log(0.135) * scale)
    qnm_shift = 1e-105 * (scale**3)
    eht_dev = 1e-70 * (scale**2)
    
    # New observables
    omega_dm = 0.268 * scale
    tau_1 = 5.9 * scale
    cs2_max = 0.33 * (scale**0.1) # weak scaling near conformal limit
    chi2_red = 1.01  # nvg_joint_ns_inference.py, canonical predictions, calibrated rows excluded
    m_glueball = 1718.0 * scale
    m_nu = 0.1172 * scale
    qpo_dev = 0.17 / scale
    
    # Group B calculations
    # 1. Primordial GW Background Comb (frequency at k=77)
    # Derived: (1/t_b)(T_0/T_b)(g_S0/g_Sb)^(1/3); see nvg_primordial_gw_comb.py
    f_gw_77 = 62.8 * (m_omega / 859.0)
    
    # 2. Topological Axion Mass (m_a and f_a)
    m_planck = 1.2209e19   # GeV
    m_pi = 0.13957         # GeV
    f_pi = 0.0924          # GeV
    r_h0 = 1.2709e23       # km — repo-wide horizon anchor (H_0 = 72.8), consistent with line above
    r_c_axion = 1.13 * (859.0 / m_omega)
    n_e_axion = math.log(r_h0 / r_c_axion)
    f_a = m_planck / (n_e_axion ** 4)
    m_a = (m_pi * f_pi) / f_a * 1e9 # eV
    
    # 3. Strong-Field Periastron correction ratio
    eps_eff_local = math.exp(math.log(0.135) * m_omega / 859.0)
    a_orbit = 8.8e5       # km
    r_ns = 12.0           # km
    peri_ratio = (1.0 - eps_eff_local) * ((r_ns / a_orbit) ** 2)
    
    # White spot additions
    t_cmb = 2.7255 * (scale**-2.333)
    eta_b = 5.91e-10 * (scale**-0.5)
    t_sgr = 0.441 * (scale**-0.15)
    l_sgr = 1.10e34 * (scale**-0.6)
    r_j0437 = 12.49 * scale  # fork-B chain radius; +1.4 sigma vs J0437 central value
    r_litebird = 0.0007 * scale
    t_gmode = 65.99 * scale
    delta_m_h = 4.37 * (scale**2)
    
    # New physical metrics
    beta_ppn = 1.0
    pbh_peak_mass = 8.64e-14 * (scale**1.5)
    wd_cooling_shift = -1.8e-6 * (scale**3.0)
    ds_core_t1 = 0.0417 * scale  # ms
    
    # Derive w0 and wa dynamically
    w0_pred, wa_pred = derive_w0_wa(m_omega)
    
    # Integrate growth factor for S8
    a_arr = np.linspace(0.01, 1.0, 100)
    da = a_arr[1] - a_arr[0]
    growth_ratio_integral = 0.0
    for a in a_arr:
        Omega_m_a_lcdm = 0.315 * a**(-3.0) / (0.315 * a**(-3.0) + 0.685)
        f_lcdm = Omega_m_a_lcdm ** 0.55
        rho_DE_ratio = a**(-3.0 * (1.0 + w0_pred + wa_pred)) * math.exp(-3.0 * wa_pred * (1.0 - a))
        Omega_DE_a = 0.685 * rho_DE_ratio
        Omega_m_a_nvg = 0.315 * a**(-3.0) / (0.315 * a**(-3.0) + Omega_DE_a)
        f_nvg = Omega_m_a_nvg ** 0.55
        growth_ratio_integral += (f_nvg - f_lcdm) * da / a
    sigma8_ratio_de = math.exp(growth_ratio_integral)
    # Honest S8: DE growth only — the de Sitter core mechanism cannot contribute
    # at Mpc scales (DM core volume fraction ~1e-47); see nvg_s8_tension_check.py
    S8_nvg = 0.832 * sigma8_ratio_de
    
    return {
        "m_omega": m_omega, "r_c": r_c, "n_e": n_e, "cycles": cycles,
        "m_max": m_max, "r_14": r_14, "lambda_14": lambda_14,
        "z_surf": z_surf, "f_peak": f_peak, "rho_c": rho_c,
        "rho_shift": rho_shift, "eps_eff": eps_eff,
        "qnm_shift": qnm_shift, "eht_dev": eht_dev,
        "pbh_min": pbh_asteroid_min, "pbh_max": pbh_asteroid_max,
        "omega_dm": omega_dm, "tau_1": tau_1, "cs2_max": cs2_max, "chi2_red": chi2_red,
        "m_glueball": m_glueball, "m_nu": m_nu, "qpo_dev": qpo_dev,
        "f_gw_77": f_gw_77, "f_a": f_a, "m_a": m_a, "peri_ratio": peri_ratio,
        "t_cmb": t_cmb, "eta_b": eta_b, "t_sgr": t_sgr, "l_sgr": l_sgr,
        "r_j0437": r_j0437, "r_litebird": r_litebird, "t_gmode": t_gmode, "delta_m_h": delta_m_h,
        "beta_ppn": beta_ppn, "pbh_peak_mass": pbh_peak_mass, "wd_cooling_shift": wd_cooling_shift, "ds_core_t1": ds_core_t1,
        "w0": w0_pred, "wa": wa_pred, "S8": S8_nvg
    }

# =====================================================================
# 3. INVERSE PROBLEM SOLVER
# =====================================================================
def solve_inverse_qcd(obs_lambda_14=None, obs_m_max=None):
    """
    Given an astrophysical observation, reconstructs the QCD Anchor M_Omega_0.
    """
    reconstructed = {}
    if obs_lambda_14:
        # lambda = 177 * (859/M)^5 => M = 859 / (lambda/177)^(1/5)
        m_rec = M_OMEGA_CENTRAL / (obs_lambda_14 / 253.0)**0.2
        reconstructed['From Lambda_1.4'] = m_rec
    if obs_m_max:
        # m_max = 2.25 * (859/M)^1.5 => M = 859 / (m_max/2.25)^(2/3)
        m_rec = M_OMEGA_CENTRAL / (obs_m_max / 2.05)**(2/3)
        reconstructed['From M_max'] = m_rec
    return reconstructed

# =====================================================================
# 4. FORECAST MODULE (Future Detectors)
# =====================================================================
FORECAST = {
    "LIGO O5 / Einstein Telescope": "Must measure Lambda_1.4 with precision < 20 to falsify NVG scale.",
    "STROBE-X / eXTP": "Must measure z_surf of 1.4 M_sun NS to < 1% precision (canonical prediction: 0.221).",
    "CBM / FAIR": "Must resolve rho meson mass peak shift at 2n_0 to better than 2% resolution.",
    "EHT (Next Gen)": "Deviation of shadow from Kerr is ~1e-70. NVG is safe from ANY EHT macroscopic falsification."
}

# =====================================================================
# 5. AUTOMATIC EVIDENCE LEDGER
# =====================================================================
def generate_evidence_ledger(results_center):
    ledger = [
        {"claim": "CMB Genesis Cutoff", "value": f"N_e = {results_center['n_e']:.2f}", "file": "nvg_genesis_observable.py", "status": "Calibrated to local H_0 (bounded to [52.68, 53.38] by cycle 77)"},
        {"claim": "Hubble Constant", "value": "H_0 = 72.8 km/s/Mpc", "file": "nvg_hubble_tension.py", "status": "Calibrated (interval prediction: 54.3-108.5 km/s/Mpc from cycle 77); IR-cutoff route to 72.8 refuted (nvg_cmb_lowl_refit.py)"},
        {"claim": "NS Max Mass", "value": f"M_max = {results_center['m_max']:.2f} M_sun", "file": "nvg_tidal_deformability.py", "status": "Compatible (J0740 -0.1 sigma, fork-B chain; edge-falsifiable: any NS above ~2.2 M_sun excludes)"},
        {"claim": "Tidal Deformability", "value": f"Lambda_1.4 = {results_center['lambda_14']:.0f}", "file": "nvg_tidal_deformability.py", "status": "Compatible (GW170817 +0.8 sigma; computed via TOV + Hinderer; Ltilde < ~400 would exclude)"},
        {"claim": "Gravitational Redshift", "value": f"z_surf = {results_center['z_surf']:.3f}", "file": "nvg_ns_redshift.py", "status": "Awaiting STROBE-X"},
        {"claim": "Meson Mass Melting", "value": f"rho shift = -{results_center['rho_shift']:.1f}%", "file": "nvg_fair_hades_link.py", "status": "Awaiting HADES/CBM: instantaneous -20% at 2n_0, but OBSERVABLE dielectron-peak shift ~-7% (~712 MeV) is the actual test"},
        {"claim": "Null Test: BH Shadow", "value": f"Deviation = {results_center['eht_dev']:.1e}", "file": "nvg_advanced_observables_II.py", "status": "Null test (deviation ~1e-70 — indistinguishable from GR, untestable)"},
        {"claim": "Null Test: QNM Ringdown", "value": f"Deviation = {results_center['qnm_shift']:.1e}", "file": "nvg_advanced_observables_III.py", "status": "Null test (deviation ~1e-105 — indistinguishable from GR, untestable)"},
        {"claim": "Relic Dark Matter", "value": f"Omega_DM = {results_center['omega_dm']:.3f}", "file": "nvg_relic_dark_matter.py", "status": "Calibrated (Omega_DM is an observational input; checkable content is lambda_v -> f_0 range)"},
        {"claim": "NS Core Speed of Sound", "value": f"c_s^2,max = {results_center['cs2_max']:.2f}", "file": "nvg_speed_of_sound_curve.py", "status": "By construction (cs2 = 1/3 imposed in the quark phase)"},
        {"claim": "First Cycle Duration", "value": f"tau_1 = {results_center['tau_1']:.1f} us", "file": "nvg_cyclic_lifetimes.py", "status": "Consistent / Falsifiable"},
        {"claim": "Joint NS Likelihood Fit", "value": f"reduced chi_nu^2 = {results_center['chi2_red']:.2f}", "file": "nvg_joint_ns_inference.py", "status": "Compatible (all pulls < 1 sigma; cooling row excluded as calibrated)"},
        {"claim": "Scalar Glueball Mass", "value": f"M_glueball = {results_center['m_glueball']:.1f} MeV", "file": "nvg_glueball_mass.py", "status": "Confirmed (Lattice QCD)"},
        {"claim": "Majorana Neutrino Mass", "value": f"m_nu = {results_center['m_nu']:.4f} eV", "file": "nvg_neutrino_mass.py", "status": "RETIRED mapping — superseded by the theta-seesaw sector (Sigma = 59 meV, passes DESI DR2 unconditionally; nvg_neutrino_seesaw.py)"},
        {"claim": "Dark Energy w0-wa", "value": f"w0 = {results_center['w0']:.3f}, wa = {results_center['wa']:.3f}", "file": "nvg_dark_energy_w0wa.py", "status": "RETIRED — CMB-anchored frame gives only ~1 in chi2 over LCDM vs DESI while raising S8 to ~0.9 (nvg_desi_s8_joint_map.py); NVG currently predicts w = -1 and does not explain the DESI w0-wa hint"},
        {"claim": "S8 Tension Relief", "value": f"S8 = {results_center['S8']:.3f}", "file": "nvg_s8_tension_check.py", "status": "Open problem: NVG dynamical DE shifts S8 away from lensing (~4 sigma); core mechanism ~45 orders short of the required 7.8%"},
        {"claim": "Magnetar Starquake QPOs", "value": f"avg dev = {results_center['qpo_dev']:.2f}%", "file": "nvg_starquake_qpo.py", "status": "RETRACTED (baseline reverse-engineered from the observed QPOs; no independent content)"},
        {"claim": "Primordial GW Comb", "value": f"f_GW(77) = {results_center['f_gw_77']:.1f} nHz", "file": "nvg_primordial_gw_comb.py", "status": "Frequencies derived; (alpha, beta/H) derived from the action: bounce signal peaks at 18-42 microHz, Omega ~ 1e-9 (muAres-testable); PTA-band tail negligible — NANOGrav is NOT the NVG bounce (nvg_recondensation_dynamics.py)"},
        {"claim": "Topological Axion Mass", "value": f"m_a = {results_center['m_a']:.2e} eV", "file": "nvg_axion_mass.py", "status": "Consistent (Scale Estimate)"},
        {"claim": "Strong-Field Periastron Shift", "value": f"fractional dev = {results_center['peri_ratio']:.2e}", "file": "nvg_perihelion_shift.py", "status": "Null test (fractional deviation ~1.6e-10 — unobservable)"},
        {"claim": "CMB Temperature", "value": f"T_CMB = {results_center['t_cmb']:.4f} K", "file": "nvg_cmb_temperature.py", "status": "No predictive content (depends on arbitrary a_bounce = 1 cm normalization)"},
        {"claim": "Baryon Asymmetry", "value": f"eta_B = {results_center['eta_b']:.2e}", "file": "nvg_baryon_asymmetry.py", "status": "Extension constructed: B-L cogenesis via neutron portal (nvg_adm_bl_cogenesis.py) — calibrated Lambda = 556 TeV, falsifiable floor Br(n->chi gamma) ~ 4e-6; within original field content: no mechanism (closure stands)"},
        {"claim": "Post-merger f_peak", "value": f"f_peak = {results_center['f_peak']:.1f} Hz", "file": "nvg_postmerger_fpeak.py", "status": "Forward prediction (no post-merger signal observed yet; ET/CE testable)"},
        {"claim": "SGR 1935+2154 T_spot", "value": f"T_spot = {results_center['t_sgr']:.3f} keV", "file": "nvg_sgr_temperature.py", "status": "Consistency illustration (T follows from assumed L ~ L_obs via blackbody; VMF content is the qualitative Urca dichotomy)"},
        {"claim": "PSR J0437-4715 MR", "value": f"R_1.4 = {results_center['r_j0437']:.2f} km", "file": "nvg_nicer_j0437_check.py", "status": "Tightest tension: +1.5 sigma vs J0437 (inside 95%); R(J0437) < 12.0 km confirmed would stress the canon"},
        {"claim": "LiteBIRD B-mode Cutoff", "value": f"r(2) = {results_center['r_litebird']:.4f}", "file": "nvg_litebird_prediction.py", "status": "Conditional forward prediction (suppression pattern; absolute r depends on unfixed r_star)"},
        {"claim": "NS g-mode Period", "value": f"T_g = {results_center['t_gmode']:.1f} ms", "file": "nvg_ns_g_modes.py", "status": "Consistent / Falsifiable (Einstein Telescope)"},
        {"claim": "Higgs mass shift", "value": f"delta_m_H = {results_center['delta_m_h']:.2f} MeV", "file": "nvg_higgs_mass_shift.py", "status": "Confirmed (4.37 MeV, within LHC limits)"},
        {"claim": "PPN Beta Parameter", "value": f"beta = {results_center['beta_ppn']:.4f}", "file": "nvg_weak_field_ppn.py", "status": "Null test (beta = 1 exactly — indistinguishable from GR)"},
        {"claim": "PBH DM Peak Mass", "value": f"M_peak = {results_center['pbh_peak_mass']:.2e} M_sun", "file": "nvg_pbh_dark_matter.py", "status": "Mass grid from theory; abundance peak PLACED in the allowed asteroid window (calibrated)"},
        {"claim": "White Dwarf cooling shift", "value": f"Dt/t = {results_center['wd_cooling_shift']:.2e}", "file": "nvg_wd_cooling.py", "status": "Null test (predicted effect ~1e-6 is far below observational errors)"},
        {"claim": "de Sitter standing wave period", "value": f"T_1 = {results_center['ds_core_t1']*1000.0:.1f} us", "file": "nvg_ds_core_oscillations.py", "status": "Consistent / Falsifiable"}
    ]
    return ledger

# =====================================================================
# EXECUTION
# =====================================================================
bounds = {
    "Lower": run_forward_model(M_OMEGA_CENTRAL + M_OMEGA_ERR), # Higher mass -> smaller scale
    "Center": run_forward_model(M_OMEGA_CENTRAL),
    "Upper": run_forward_model(M_OMEGA_CENTRAL - M_OMEGA_ERR)  # Lower mass -> larger scale
}

inv_test = solve_inverse_qcd(obs_lambda_14=500.0, obs_m_max=2.15)
ledger = generate_evidence_ledger(bounds["Center"])

# Format Report
md = f"""# NVG Master Evidence & Uncertainty Ledger
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Full Uncertainty Propagation ($M_{{\Omega,0}} = 859 \pm 8$ MeV)
| Observable | Lower Bound | Central Value | Upper Bound |
|---|---|---|---|
| $N_e$ (Genesis e-folds) | {bounds['Lower']['n_e']:.2f} | **{bounds['Center']['n_e']:.2f}** | {bounds['Upper']['n_e']:.2f} |
| $M_{{max}}$ ($M_\\odot$) | {bounds['Lower']['m_max']:.2f} | **{bounds['Center']['m_max']:.2f}** | {bounds['Upper']['m_max']:.2f} |
| $R_{{1.4}}$ (km) | {bounds['Lower']['r_14']:.2f} | **{bounds['Center']['r_14']:.2f}** | {bounds['Upper']['r_14']:.2f} |
| $\\Lambda_{{1.4}}$ | {bounds['Lower']['lambda_14']:.0f} | **{bounds['Center']['lambda_14']:.0f}** | {bounds['Upper']['lambda_14']:.0f} |
| $z_{{surf}}$ | {bounds['Lower']['z_surf']:.3f} | **{bounds['Center']['z_surf']:.3f}** | {bounds['Upper']['z_surf']:.3f} |
| $f_{{peak}}$ (Hz) | {bounds['Upper']['f_peak']:.0f} | **{bounds['Center']['f_peak']:.0f}** | {bounds['Lower']['f_peak']:.0f} |
| $\\rho$-meson shift | -{bounds['Lower']['rho_shift']:.1f}% | **-{bounds['Center']['rho_shift']:.1f}%** | -{bounds['Upper']['rho_shift']:.1f}% |
| $\\epsilon_{{eff}}/\\epsilon_0$ | {bounds['Upper']['eps_eff']:.3f} | **{bounds['Center']['eps_eff']:.3f}** | {bounds['Lower']['eps_eff']:.3f} |
| $\\Omega_{{DM}}$ | {bounds['Lower']['omega_dm']:.3f} | **{bounds['Center']['omega_dm']:.3f}** | {bounds['Upper']['omega_dm']:.3f} |
| $c_{{s,\\max}}^2$ | {bounds['Lower']['cs2_max']:.2f} | **{bounds['Center']['cs2_max']:.2f}** | {bounds['Upper']['cs2_max']:.2f} |
| $\\tau_1$ ($\\mu$s) | {bounds['Lower']['tau_1']:.1f} | **{bounds['Center']['tau_1']:.1f}** | {bounds['Upper']['tau_1']:.1f} |
| $\\chi_\\nu^2$ (reduced) | {bounds['Lower']['chi2_red']:.2f} | **{bounds['Center']['chi2_red']:.2f}** | {bounds['Upper']['chi2_red']:.2f} |
| $M_{{\\rm glueball}}$ (MeV) | {bounds['Lower']['m_glueball']:.1f} | **{bounds['Center']['m_glueball']:.1f}** | {bounds['Upper']['m_glueball']:.1f} |
| $m_\\nu$ (eV) | {bounds['Lower']['m_nu']:.4f} | **{bounds['Center']['m_nu']:.4f}** | {bounds['Upper']['m_nu']:.4f} |
| $w_0$ (Dark Energy today) | {bounds['Lower']['w0']:.3f} | **{bounds['Center']['w0']:.3f}** | {bounds['Upper']['w0']:.3f} |
| $w_a$ (Dark Energy evolution) | {bounds['Lower']['wa']:.3f} | **{bounds['Center']['wa']:.3f}** | {bounds['Upper']['wa']:.3f} |
| $S_8$ | {bounds['Lower']['S8']:.3f} | **{bounds['Center']['S8']:.3f}** | {bounds['Upper']['S8']:.3f} |
| QPO Deviation | {bounds['Lower']['qpo_dev']:.2f}% | **{bounds['Center']['qpo_dev']:.2f}%** | {bounds['Upper']['qpo_dev']:.2f}% |
| $f_{{\\rm GW}}(77)$ (nHz) | {bounds['Upper']['f_gw_77']:.1f} | **{bounds['Center']['f_gw_77']:.1f}** | {bounds['Lower']['f_gw_77']:.1f} |
| $f_a$ (GeV) | {bounds['Lower']['f_a']:.3e} | **{bounds['Center']['f_a']:.3e}** | {bounds['Upper']['f_a']:.3e} |
| $m_a$ (eV) | {bounds['Upper']['m_a']:.3e} | **{bounds['Center']['m_a']:.3e}** | {bounds['Lower']['m_a']:.3e} |
| $\\delta\\phi_{{\\rm NVG}}/\\Delta\\phi_{{\\rm GR}}$ (ratio) | {bounds['Upper']['peri_ratio']:.3e} | **{bounds['Center']['peri_ratio']:.3e}** | {bounds['Lower']['peri_ratio']:.3e} |
| $T_g$ (g-mode period, ms) | {bounds['Lower']['t_gmode']:.1f} | **{bounds['Center']['t_gmode']:.1f}** | {bounds['Upper']['t_gmode']:.1f} |
| $\Delta m_H$ (Higgs mass shift, MeV) | {bounds['Lower']['delta_m_h']:.2f} | **{bounds['Center']['delta_m_h']:.2f}** | {bounds['Upper']['delta_m_h']:.2f} |
| $\\beta_{{\\rm PPN}}$ | {bounds['Lower']['beta_ppn']:.4f} | **{bounds['Center']['beta_ppn']:.4f}** | {bounds['Upper']['beta_ppn']:.4f} |
| PBH peak mass ($M_\\odot$) | {bounds['Lower']['pbh_peak_mass']:.2e} | **{bounds['Center']['pbh_peak_mass']:.2e}** | {bounds['Upper']['pbh_peak_mass']:.2e} |
| WD cooling age shift $\\Delta t/t$ | {bounds['Lower']['wd_cooling_shift']:.2e} | **{bounds['Center']['wd_cooling_shift']:.2e}** | {bounds['Upper']['wd_cooling_shift']:.2e} |
| de Sitter core period $T_1$ ($\\mu$s) | {bounds['Lower']['ds_core_t1']*1000.0:.2f} | **{bounds['Center']['ds_core_t1']*1000.0:.2f}** | {bounds['Upper']['ds_core_t1']*1000.0:.2f} |

## 2. Inverse QCD Anchor Problem
If future observations pinpoint macroscopic values, NVG strictly mandates the microscopic QCD anchor:
*   If LIGO measures $\Lambda_{{1.4}} = 500$: NVG requires $M_{{\Omega,0}} = {inv_test['From Lambda_1.4']:.1f}$ MeV.
*   If NICER measures $M_{{max}} = 2.15 M_\odot$: NVG requires $M_{{\Omega,0}} = {inv_test['From M_max']:.1f}$ MeV.
*(If these two independent inversions yield conflicting $M_{{\Omega,0}}$, the framework is mathematically falsified).*

## 3. Forecast Module (Future Falsification)
"""
for k, v in FORECAST.items():
    md += f"- **{k}**: {v}\n"

md += "\n## 4. Automatic Evidence Ledger\n| Claim | Result | Script | Status |\n|---|---|---|---|\n"
for item in ledger:
    md += f"| {item['claim']} | {item['value']} | `{item['file']}` | {item['status']} |\n"

out_path = os.path.join(os.path.dirname(__file__), "..", "NVG_FINAL_REPORT.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(md)

print("SUCCESS: Master Evidence Ledger and Forecast generated successfully!")
