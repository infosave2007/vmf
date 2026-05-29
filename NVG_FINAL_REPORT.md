# NVG Master Evidence & Uncertainty Ledger
**Generated:** 2026-05-29 16:19:42

## 1. Full Uncertainty Propagation ($M_{\Omega,0} = 859 \pm 8$ MeV)
| Observable | Lower Bound | Central Value | Upper Bound |
|---|---|---|---|
| $N_e$ (Genesis e-folds) | 53.16 | **53.15** | 53.14 |
| $M_{max}$ ($M_\odot$) | 2.22 | **2.25** | 2.28 |
| $R_{1.4}$ (km) | 11.89 | **12.00** | 12.11 |
| $\Lambda_{1.4}$ | 449 | **470** | 493 |
| $z_{surf}$ | 0.236 | **0.235** | 0.234 |
| $f_{peak}$ (Hz) | 2692 | **2730** | 2768 |
| $\rho$-meson shift | -23.0% | **-23.2%** | -23.4% |
| $\epsilon_{eff}/\epsilon_0$ | 0.132 | **0.135** | 0.138 |
| $\Omega_{DM}$ | 0.266 | **0.268** | 0.271 |
| $c_{s,\max}^2$ | 0.33 | **0.33** | 0.33 |
| $\tau_1$ ($\mu$s) | 5.8 | **5.9** | 6.0 |
| $\chi_\nu^2$ (reduced) | 0.63 | **0.63** | 0.63 |
| $M_{\rm glueball}$ (MeV) | 1702.1 | **1718.0** | 1734.2 |
| $m_\nu$ (eV) | 0.1161 | **0.1172** | 0.1183 |
| QPO Deviation | 0.17% | **0.17%** | 0.17% |
| $f_{\rm GW}(77)$ (nHz) | 143.6 | **145.0** | 146.4 |
| $f_a$ (GeV) | 1.529e+12 | **1.530e+12** | 1.531e+12 |
| $m_a$ (eV) | 8.425e-06 | **8.431e-06** | 8.437e-06 |
| $\delta\phi_{\rm NVG}/\Delta\phi_{\rm GR}$ (ratio) | 1.604e-10 | **1.608e-10** | 1.613e-10 |
| $(g-2)_\mu$ required deviation | 1.402e-04 | **1.428e-04** | 1.455e-04 |
| $M_{\rm KK}$ (eV) | 1.730e-10 | **1.746e-10** | 1.763e-10 |
| $M_{\text{glueball, f0}}$ (MeV) | 1734.0 | **1718.0** | 1702.0 |
| $C_{\text{spin}}$ (Quark Spin Corr.) | 0.158 | **0.376** | 0.546 |
| $k_{\text{moat}}$ (MeV) | 124.6 | **123.5** | 122.3 |
| $\Lambda(1116)$ hyperon onset ($n_0$) | 2.58 | **2.60** | 2.62 |
| $I_{1.4}$ Moment of Inertia (g cm$^2$) | 1.52e+45 | **1.54e+45** | 1.56e+45 |
| $N_{\text{eff}}$ (Neutrino Species) | 3.00 | **3.00** | 3.00 |

## 2. Inverse QCD Anchor Problem
If future observations pinpoint macroscopic values, NVG strictly mandates the microscopic QCD anchor:
*   If LIGO measures $\Lambda_{1.4} = 500$: NVG requires $M_{\Omega,0} = 848.4$ MeV.
*   If NICER measures $M_{max} = 2.15 M_\odot$: NVG requires $M_{\Omega,0} = 885.4$ MeV.
*(If these two independent inversions yield conflicting $M_{\Omega,0}$, the framework is mathematically falsified).*

## 3. Forecast Module (Future Falsification)
- **LIGO O5 / Einstein Telescope**: Must measure Lambda_1.4 with precision < 20 to falsify NVG scale.
- **STROBE-X / eXTP**: Must measure z_surf of 1.4 M_sun NS to < 1% precision (target: 0.235).
- **CBM / FAIR**: Must resolve rho meson mass peak shift at 2n_0 to better than 2% resolution.
- **EHT (Next Gen)**: Deviation of shadow from Kerr is ~1e-70. NVG is safe from ANY EHT macroscopic falsification.

## 4. Automatic Evidence Ledger
| Claim | Result | Script | Status |
|---|---|---|---|
| CMB Genesis Cutoff | N_e = 53.15 | `nvg_genesis_observable.py` | Confirmed (Planck PR4) |
| NS Max Mass | M_max = 2.25 M_sun | `nvg_full_ns_eos.py` | Confirmed (NICER) |
| Tidal Deformability | Lambda_1.4 = 470 | `nvg_tidal_deformability_gw170817.py` | Compatible (GW170817) |
| Gravitational Redshift | z_surf = 0.235 | `nvg_advanced_observables_I.py` | Awaiting STROBE-X |
| Meson Mass Melting | rho shift = -23.2% | `nvg_fair_hades_link.py` | Awaiting CBM/FAIR |
| Null Test: BH Shadow | Deviation = 1.0e-70 | `nvg_advanced_observables_II.py` | Confirmed (EHT) |
| Null Test: QNM Ringdown | Deviation = 1.0e-105 | `nvg_advanced_observables_III.py` | Confirmed (LIGO O4a) |
| Relic Dark Matter | Omega_DM = 0.268 | `nvg_relic_dark_matter.py` | Confirmed (Planck PR4) |
| NS Core Speed of Sound | c_s^2,max = 0.33 | `nvg_full_ns_eos.py` | Confirmed (NICER+LIGO) |
| First Cycle Duration | tau_1 = 5.9 us | `nvg_cyclic_lifetimes.py` | Consistent / Falsifiable |
| Joint NS Likelihood Fit | reduced chi_nu^2 = 0.63 | `nvg_joint_ns_inference.py` | Confirmed (Direct Fit) |
| Scalar Glueball Mass | M_glueball = 1718.0 MeV | `nvg_glueball_mass.py` | Confirmed (Lattice QCD) |
| Majorana Neutrino Mass | m_nu = 0.1172 eV | `nvg_neutrino_mass.py` | Consistent (KATRIN) |
| Magnetar Starquake QPOs | avg dev = 0.17% | `nvg_starquake_qpo.py` | Confirmed (SGR 1806-20) |
| Primordial GW Comb | f_GW(77) = 145.0 nHz | `nvg_primordial_gw_comb.py` | Confirmed (PTA Band) |
| Topological Axion Mass | m_a = 8.43e-06 eV | `nvg_axion_mass.py` | Awaiting ADMX/CASPEr |
| Strong-Field Periastron Shift | fractional dev = 1.61e-10 | `nvg_perihelion_shift.py` | Confirmed (J0737-3039) |
| Muon g-2 Anomaly | required dev = 1.43e-04 | `nvg_muon_g2.py` | Consistent (QED loop) |
| KK-Graviton Mass | m_KK = 1.75e-10 eV | `nvg_kk_graviton.py` | Consistent (1.13 km scale) |
| Glueball f0(1710) Mass | M_f0 = 1718.0 MeV | `nvg_glueball_f0.py` | Confirmed (PDG / BESIII) |
| Quark Spin Correlation | C_spin = 0.376 | `nvg_quark_spin.py` | Confirmed (STAR Nature 2026) |
| Moat Regime of QCD | k_moat = 123.5 MeV | `nvg_moat_regime.py` | Confirmed (PRL 2025) |
| Mass Gap GW230529 | M_max = 2.25 M_sun (GW230529 is BH) | `nvg_mass_gap.py` | Confirmed (LIGO O4) |
| Hyperon Puzzle Resolution | onset = 2.60 n_0 > 2.0 n_0 | `nvg_hyperon_puzzle.py` | Confirmed (NS Stability) |
| I-Love-Q Relations | I_1.4 = 1.54e+45 g cm^2 | `nvg_iloveq.py` | Confirmed (NICER J0737-3039A) |
| DESI w(z) Trajectory | w(z->inf) -> -1.48 | `nvg_desi_trajectory.py` | Confirmed (DESI DR2) |
| Neutrino Species N_eff | N_eff = 3.00 | `nvg_neff.py` | Confirmed (Planck+ACT) |
