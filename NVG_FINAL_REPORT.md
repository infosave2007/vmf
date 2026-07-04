# NVG Master Evidence & Uncertainty Ledger
**Generated:** 2026-07-04 09:15:14

## 1. Full Uncertainty Propagation ($M_{\Omega,0} = 859 \pm 8$ MeV)
| Observable | Lower Bound | Central Value | Upper Bound |
|---|---|---|---|
| $N_e$ (Genesis e-folds) | 53.09 | **53.08** | 53.07 |
| $M_{max}$ ($M_\odot$) | 2.02 | **2.05** | 2.08 |
| $R_{1.4}$ (km) | 12.43 | **12.55** | 12.67 |
| $\Lambda_{1.4}$ | 495 | **519** | 544 |
| $z_{surf}$ | 0.222 | **0.221** | 0.220 |
| $f_{peak}$ (Hz) | 2387 | **2421** | 2455 |
| $\rho$-meson shift | -19.8% | **-20.0%** | -20.2% |
| $\epsilon_{eff}/\epsilon_0$ | 0.132 | **0.135** | 0.138 |
| $\Omega_{DM}$ | 0.266 | **0.268** | 0.271 |
| $c_{s,\max}^2$ | 0.33 | **0.33** | 0.33 |
| $\tau_1$ ($\mu$s) | 5.8 | **5.9** | 6.0 |
| $\chi_\nu^2$ (reduced) | 1.01 | **1.01** | 1.01 |
| $M_{\rm glueball}$ (MeV) | 1702.1 | **1718.0** | 1734.2 |
| $m_\nu$ (eV) | 0.1161 | **0.1172** | 0.1183 |
| $w_0$ (Dark Energy today) | -0.878 | **-0.876** | -0.875 |
| $w_a$ (Dark Energy evolution) | -0.660 | **-0.667** | -0.674 |
| $S_8$ | 0.779 | **0.778** | 0.777 |
| QPO Deviation | 0.17% | **0.17%** | 0.17% |
| $f_{\rm GW}(77)$ (nHz) | 62.2 | **62.8** | 63.4 |
| $f_a$ (GeV) | 1.537e+12 | **1.538e+12** | 1.539e+12 |
| $m_a$ (eV) | 8.377e-06 | **8.383e-06** | 8.389e-06 |
| $\delta\phi_{\rm NVG}/\Delta\phi_{\rm GR}$ (ratio) | 1.604e-10 | **1.608e-10** | 1.613e-10 |
| $T_g$ (g-mode period, ms) | 65.4 | **66.0** | 66.6 |
| $\Delta m_H$ (Higgs mass shift, MeV) | 4.29 | **4.37** | 4.45 |
| $\beta_{\rm PPN}$ | 1.0000 | **1.0000** | 1.0000 |
| PBH peak mass ($M_\odot$) | 8.52e-14 | **8.64e-14** | 8.76e-14 |
| WD cooling age shift $\Delta t/t$ | -1.75e-06 | **-1.80e-06** | -1.85e-06 |
| de Sitter core period $T_1$ ($\mu$s) | 41.32 | **41.70** | 42.09 |

## 2. Inverse QCD Anchor Problem
If future observations pinpoint macroscopic values, NVG strictly mandates the microscopic QCD anchor:
*   If LIGO measures $\Lambda_{1.4} = 500$: NVG requires $M_{\Omega,0} = 865.4$ MeV.
*   If NICER measures $M_{max} = 2.15 M_\odot$: NVG requires $M_{\Omega,0} = 832.2$ MeV.
*(If these two independent inversions yield conflicting $M_{\Omega,0}$, the framework is mathematically falsified).*

## 3. Forecast Module (Future Falsification)
- **LIGO O5 / Einstein Telescope**: Must measure Lambda_1.4 with precision < 20 to falsify NVG scale.
- **STROBE-X / eXTP**: Must measure z_surf of 1.4 M_sun NS to < 1% precision (canonical prediction: 0.221).
- **CBM / FAIR**: Must resolve rho meson mass peak shift at 2n_0 to better than 2% resolution.
- **EHT (Next Gen)**: Deviation of shadow from Kerr is ~1e-70. NVG is safe from ANY EHT macroscopic falsification.

## 4. Automatic Evidence Ledger
| Claim | Result | Script | Status |
|---|---|---|---|
| CMB Genesis Cutoff | N_e = 53.08 | `nvg_genesis_observable.py` | Calibrated to local H_0 (bounded to [52.68, 53.38] by cycle 77) |
| Hubble Constant | H_0 = 72.8 km/s/Mpc | `nvg_hubble_tension.py` | Calibrated (interval prediction: 54.3-108.5 km/s/Mpc from cycle 77); IR-cutoff route to 72.8 refuted (nvg_cmb_lowl_refit.py) |
| NS Max Mass | M_max = 2.05 M_sun | `nvg_tidal_deformability.py` | Compatible (J0740 -0.4 sigma; edge-falsifiable: any NS above ~2.2 M_sun excludes) |
| Tidal Deformability | Lambda_1.4 = 519 | `nvg_tidal_deformability.py` | Compatible (GW170817 +0.8 sigma; computed via TOV + Hinderer; Ltilde < ~400 would exclude) |
| Gravitational Redshift | z_surf = 0.221 | `nvg_ns_redshift.py` | Awaiting STROBE-X |
| Meson Mass Melting | rho shift = -20.0% | `nvg_fair_hades_link.py` | Awaiting CBM/FAIR (Derived from W-field Coupling) |
| Null Test: BH Shadow | Deviation = 1.0e-70 | `nvg_advanced_observables_II.py` | Null test (deviation ~1e-70 — indistinguishable from GR, untestable) |
| Null Test: QNM Ringdown | Deviation = 1.0e-105 | `nvg_advanced_observables_III.py` | Null test (deviation ~1e-105 — indistinguishable from GR, untestable) |
| Relic Dark Matter | Omega_DM = 0.268 | `nvg_relic_dark_matter.py` | Calibrated (Omega_DM is an observational input; checkable content is lambda_v -> f_0 range) |
| NS Core Speed of Sound | c_s^2,max = 0.33 | `nvg_speed_of_sound_curve.py` | By construction (cs2 = 1/3 imposed in the quark phase) |
| First Cycle Duration | tau_1 = 5.9 us | `nvg_cyclic_lifetimes.py` | Consistent / Falsifiable |
| Joint NS Likelihood Fit | reduced chi_nu^2 = 1.01 | `nvg_joint_ns_inference.py` | Compatible (all pulls < 1 sigma; cooling row excluded as calibrated) |
| Scalar Glueball Mass | M_glueball = 1718.0 MeV | `nvg_glueball_mass.py` | Confirmed (Lattice QCD) |
| Majorana Neutrino Mass | m_nu = 0.1172 eV | `nvg_neutrino_mass.py` | Excluded in LCDM by DESI DR2 (<0.064 eV); viable only with dynamical DE (<0.16) — co-testable with the NVG w0-wa claim |
| Dark Energy w0-wa | w0 = -0.876, wa = -0.667 | `nvg_dark_energy_w0wa.py` | Consistent (Scale Estimate) |
| S8 Tension Relief | S8 = 0.778 | `nvg_s8_tension_check.py` | Calibrated (7.8% suppression fitted to lensing S8, not derived) |
| Magnetar Starquake QPOs | avg dev = 0.17% | `nvg_starquake_qpo.py` | RETRACTED (baseline reverse-engineered from the observed QPOs; no independent content) |
| Primordial GW Comb | f_GW(77) = 62.8 nHz | `nvg_primordial_gw_comb.py` | Frequencies derived (anchor 62.8 nHz from t_b + adiabatic redshift, spacing 4^(-1/3) from the Tolman law); amplitude derivation pending |
| Topological Axion Mass | m_a = 8.38e-06 eV | `nvg_axion_mass.py` | Consistent (Scale Estimate) |
| Strong-Field Periastron Shift | fractional dev = 1.61e-10 | `nvg_perihelion_shift.py` | Null test (fractional deviation ~1.6e-10 — unobservable) |
| CMB Temperature | T_CMB = 2.7255 K | `nvg_cmb_temperature.py` | No predictive content (depends on arbitrary a_bounce = 1 cm normalization) |
| Baryon Asymmetry | eta_B = 5.91e-10 | `nvg_baryon_asymmetry.py` | Consistent (Scale Estimate) |
| Post-merger f_peak | f_peak = 2421.0 Hz | `nvg_postmerger_fpeak.py` | Forward prediction (no post-merger signal observed yet; ET/CE testable) |
| SGR 1935+2154 T_spot | T_spot = 0.441 keV | `nvg_sgr_temperature.py` | Consistency illustration (T follows from assumed L ~ L_obs via blackbody; VMF content is the qualitative Urca dichotomy) |
| PSR J0437-4715 MR | R_1.4 = 12.55 km | `nvg_nicer_j0437_check.py` | Tightest tension: +1.5 sigma vs J0437 (inside 95%); R(J0437) < 12.0 km confirmed would stress the canon |
| LiteBIRD B-mode Cutoff | r(2) = 0.0007 | `nvg_litebird_prediction.py` | Conditional forward prediction (suppression pattern; absolute r depends on unfixed r_star) |
| NS g-mode Period | T_g = 66.0 ms | `nvg_ns_g_modes.py` | Consistent / Falsifiable (Einstein Telescope) |
| Higgs mass shift | delta_m_H = 4.37 MeV | `nvg_higgs_mass_shift.py` | Confirmed (4.37 MeV, within LHC limits) |
| PPN Beta Parameter | beta = 1.0000 | `nvg_weak_field_ppn.py` | Null test (beta = 1 exactly — indistinguishable from GR) |
| PBH DM Peak Mass | M_peak = 8.64e-14 M_sun | `nvg_pbh_dark_matter.py` | Mass grid from theory; abundance peak PLACED in the allowed asteroid window (calibrated) |
| White Dwarf cooling shift | Dt/t = -1.80e-06 | `nvg_wd_cooling.py` | Null test (predicted effect ~1e-6 is far below observational errors) |
| de Sitter standing wave period | T_1 = 41.7 us | `nvg_ds_core_oscillations.py` | Consistent / Falsifiable |
