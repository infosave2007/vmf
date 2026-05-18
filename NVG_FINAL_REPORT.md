# NVG Master Evidence & Uncertainty Ledger
**Generated:** 2026-05-18 11:31:18

## 1. Full Uncertainty Propagation ($M_{\Omega,0} = 859 \pm 8$ MeV)
| Observable | Lower Bound | Central Value | Upper Bound |
|---|---|---|---|
| $N_e$ (Genesis e-folds) | 53.16 | **53.15** | 53.14 |
| $M_{max}$ ($M_\odot$) | 2.22 | **2.25** | 2.28 |
| $R_{1.4}$ (km) | 11.89 | **12.00** | 12.11 |
| $\Lambda_{1.4}$ | 449 | **470** | 493 |
| $z_{surf}$ | 0.236 | **0.235** | 0.234 |
| $f_{peak}$ (Hz) | 2692 | **2730** | 2768 |
| $ho$-meson shift | -23.0% | **-23.2%** | -23.4% |
| $\epsilon_{eff}/\epsilon_0$ | 0.132 | **0.135** | 0.138 |

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
| CMB Genesis Cutoff | N_e = 53.15 | `nvg_advanced_observables_II.py` | Confirmed (Planck PR4) |
| NS Max Mass | M_max = 2.25 M_sun | `nvg_observables_O_S.py` | Confirmed (NICER) |
| Tidal Deformability | Lambda_1.4 = 470 | `nvg_grmhd_surrogate.py` | Compatible (GW170817) |
| Gravitational Redshift | z_surf = 0.235 | `run_nvg_suite.py` | Awaiting STROBE-X |
| Meson Mass Melting | rho shift = -23.2% | `nvg_advanced_observables_III.py` | Awaiting CBM/FAIR |
| Null Test: BH Shadow | Deviation = 1.0e-70 | `nvg_advanced_observables_II.py` | Confirmed (EHT) |
| Null Test: QNM Ringdown | Deviation = 1.0e-105 | `nvg_advanced_observables_III.py` | Confirmed (LIGO O4a) |
