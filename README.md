# Null-Vector Gravity (NVG) & Vacuum Mass Fraction (VMF) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://python.org)
[![CI Build & Verification](https://github.com/infosave2007/vmf/actions/workflows/verify.yml/badge.svg)](https://github.com/infosave2007/vmf/actions/workflows/verify.yml)

**Русская версия:** [README_RU.md](README_RU.md) · **Publication Status & Provenance Policy:** [PUBLICATION_STATUS.md](docs/reports/PUBLICATION_STATUS.md)

**NVG / VMF** is a covariant vacuum-condensate effective field theory (featuring a $U(1)_B$ vector field and a scalar/cuscuton condensate) bridging dense nuclear matter, finite nuclei, finite-temperature Ward identities, neutron-star structure (TOV + 1st/2nd-order Hartle), and nonsingular QCD-scale bounce cosmology with **zero parameter refitting between sectors**.

> [!IMPORTANT]
> **Scientific Boundary (`evidence_weight = 0.0`):** Every computational audit and cross-sector bridge carries zero independent empirical evidence weight. This README strictly separates: (1) mathematically closed zero-refit bridges, (2) conditional/in-sample comparisons with data, (3) forward falsifiable forecasts, and (4) explicit tensions, blocked mechanisms, and retired hypotheses. See [PUBLICATION_STATUS.md](docs/reports/PUBLICATION_STATUS.md) and [verification/registry.json](verification/registry.json).

### Contents
1. [Key Results: Matches, Tensions & Status Ledger](#1-key-results-matches-tensions--status-ledger)
2. [Key Falsifiable Predictions (Forward Forecasts)](#2-key-falsifiable-predictions-forward-forecasts)
3. [Model Architecture & Technical Audits](#3-model-architecture--technical-audits)
4. [Quick Start & Reproducible Verification](#4-quick-start--reproducible-verification)
5. [Repository Structure](#5-repository-structure)
6. [Historical Preprints (Zenodo)](#6-historical-preprints-zenodo)

---

## 1. Key Results: Matches, Tensions & Status Ledger

### 1.1 Unified 8-Bridge Covariant System (`zero-refit`, Cycles 1–3)

An auditable cross-sector closure across **8 validated bridges** (`evidence_weight = 0.0`, zero parameter refitting between sectors). Detailed reports: [Cycle 1 (Bridges 1–4)](docs/unified/NVG_UNIFIED_SYSTEM_BRIDGE_2026_11_26_RU.md), [Cycle 2 (Bridges 5–7)](docs/unified/NVG_UNIFIED_SYSTEM_CYCLE2_2026_11_26_RU.md), [Cycle 3 (Bridge 8)](docs/unified/NVG_UNIFIED_SYSTEM_CYCLE3_2026_11_26_RU.md).

| Bridge | Sector | Key Numerical Result | Producer / Validator |
|---|---|---|---|
| **Bridge 1** | Quartic vector melting in dense matter (Law 1) | Stabilizes branch without $W \to 0$ collapse at nonzero $n_B > 0$ | [nvg_quartic_vector_melting_audit.py](verification/nvg_quartic_vector_melting_audit.py) |
| **Bridge 2** | Globally bounded two-condensate EOS (Law 2) | Bounded-below scalar sector; $M_{\max} = 2.093\,M_\odot$, $R_{1.4} = 12.95$ km | [nvg_two_condensate_unified_audit.py](verification/nvg_two_condensate_unified_audit.py) |
| **Bridge 3** | Widened $\zeta_{\rm RMF}$ saturation gate (Cycle 3: Bridge 8) | $\zeta_{\rm RMF} \in \{0.0165, 0.0175, 0.0185\}$ yields **8 primary-gate survivors**, winner $\Lambda_{1.4} = 575.3$ | [nvg_unified_primary_gate_pairing_bridge.py](verification/nvg_unified_primary_gate_pairing_bridge.py) |
| **Bridge 4** | Spherical Dirac–Hartree finite nuclei (`zero-refit`, Law 3A) | `O16..Pb208` with zero nuclear refitting: $\text{RMS}(B/A) = 0.236$ MeV, $\text{RMS}(R_{\rm ch}) = 1.44\%$, positive `Ca48/Pb208` skins | [nvg_unified_dirac_hartree_probe.py](verification/nvg_unified_dirac_hartree_probe.py) |
| **Bridge 5** | Tin isotopic chain RH-BCS pairing (Bridge 8) | $C_n = 21.0$ MeV gives exact $\Delta_n = 0$ collapse on doubly-magic `Sn132` and physical gaps `0.63–0.96` MeV on `Sn116/120/124` | [nvg_unified_primary_gate_pairing_bridge.py](verification/nvg_unified_primary_gate_pairing_bridge.py) |
| **Bridge 6** | Finite-$T$ Ward & 5×5 Schur-Hessian thermal response | Exact Ward closure to `4.4e-15` (vs `1.44` failure on inconsistent `ph-only` control) | [nvg_unified_thermal_response_bridge.py](verification/nvg_unified_thermal_response_bridge.py) |
| **Bridge 7** | Independent 1st + 2nd-order Hartle `I-Love-Q` | Direct ODE on `PSR J0737-3039A`: $\bar{I} = 14.69$, $\bar{Q} = 6.74$, `0.07%–0.21%` Yagi–Yunes agreement | [nvg_unified_astrophysics_stability_bridge.py](verification/nvg_unified_astrophysics_stability_bridge.py) |
| **Bridge 8** | Covariant gravity, Hayward BH core & bounce (Laws 3B, 4) | Proves $U(1)_B$ decoupling, anisotropic $P_r = -\varepsilon \neq P_t$ BH interior, and 6-variable cuscuton bounce | [nvg_unified_gravity_bridge_audit.py](verification/nvg_unified_gravity_bridge_audit.py) |

### 1.2 Runtime Comparisons, Null Checks, Tensions & Retired Claims

**Status legend:** ✅ consistent with benchmark / lattice · ⚙️ conditional or calibration check · 📏 interval forecast · 📐 mathematical / kinematic result · ⚪ null test · ⚠️ tension flagged · ❌ open problem / retired.

| # | Observable / Sector | NVG Runtime Output | Data / Benchmark | Status & Scope |
|---|---|---|---|---|
| 1 | QCD anchor scale ($M_{\Omega,0}$) | $M_{\Omega,0} = 859 \pm 8$ MeV (~91% of nucleon mass budget) | Subtracted $\sigma_{\pi N} \approx 44$ MeV, $\sigma_{sN} \approx 30$ MeV from $M_N$ | ⚙️ Declared input convention (not a prediction) |
| 2 | Neutron star $M_{\max}$ (canonical) | $M_{\max} = 2.048\,M_\odot$ (runtime canonical TOV chain) | PSR J0740+6620: $2.14 \pm 0.10\,M_\odot$ ($-0.92\sigma$) | ⚙️ Conditional / in-sample ([run_nvg_suite.py](verification/run_nvg_suite.py)) |
| 3 | Neutron star radius $R_{1.4}$ (canonical) | $R_{1.4} = 12.550$ km (runtime canonical TOV chain) | NICER J0030: $12.2 \pm 0.5$ km ($+0.70\sigma$); J0437: $11.36 \pm 0.8$ km | ⚙️ Conditional / in-sample ([run_nvg_suite.py](verification/run_nvg_suite.py)) |
| 4 | Tidal deformability $\Lambda_{1.4}$ | $\Lambda_{1.4} = 519.4$ (TOV + Hinderer $y$-equation) | GW170817: $190^{+390}_{-120}$ ($+0.84\sigma$ pull) | ⚙️ Conditional / in-sample ([nvg_tidal_deformability.py](verification/nvg_tidal_deformability.py)) |
| 5 | Joint NS inference ($\chi_\nu^2$) | Reduced $\chi_\nu^2 = 0.684$ over 3 runtime rows | Transition selected on J0740 / GW170817 / NICER | ⚙️ Conditional / in-sample ([nvg_joint_ns_inference.py](verification/nvg_joint_ns_inference.py)) |
| 6 | `I-Love-Q` (2nd-order vs legacy) | Direct 2nd-order Hartle gives $\tilde{q} = 6.508$ (`0.61%` Yagi–Yunes dev.); legacy [nvg_iloveq_plot.py](verification/nvg_iloveq_plot.py) is **transform-only** | Yagi–Yunes universal relations | ⚙️ Conditional single-EOS probe; legacy is transform-only |
| 7 | Blind isovector symmetry energy ($j$) | Extracted $j = 11.136$ MeV, $J_{\rm design} = 23.780$ MeV, $g_\rho = 6.60$ | Residual collapse `202.7 -> -3.4 MeV` (~60×); published RMF `4.1–6.8` | 📐 Pre-registered blind identification ([report](docs/reports/NVG_J_IDENTIFICATION_RU.md)) |
| 8 | Scalar glueball mass ($0^{++}$) | $M_{\rm glueball} = 2 M_\Omega \approx 1.72$ GeV | Lattice QCD: $1.7 \pm 0.1$ GeV (experimental ID unsettled) | ✅ Lattice-consistent ([nvg_glueball_mass.py](verification/nvg_glueball_mass.py)) |
| 9 | Weak-field PPN & GW speed | $\gamma_{\rm PPN} = 1.0$, $\beta_{\rm PPN} = 1.0$, $c_T = c$, periastron shift $\sim 1.6 \times 10^{-10}$ | Cassini, LLR, GW170817 ($|c_T/c - 1| < 10^{-15}$) | ✅ Within measurement precision ([nvg_weak_field_ppn.py](verification/nvg_weak_field_ppn.py)) |
| 10 | Genesis e-folds & $H_0$ ($n=77$) | $N_e \in [52.68, 53.38]$, turnaround horizon $H_0 \in [54.3, 108.5]$ km/s/Mpc (midpoint $\approx 76.8$) | SH0ES $73.04 \pm 1.04$, Planck $67.4$; low-$\ell$ CMB lite check is **not the Planck likelihood** | 📏 Interval forecast ([nvg_cmb_lowl_refit.py](verification/nvg_cmb_lowl_refit.py)) |
| 11 | Structure growth $S_8$ & DESI DE | $S_8 \approx 0.851$ ($4.4\sigma$ away from lensing); melting $(w_0,w_a)$ mechanism retired | Melting shift worsens $S_8$; no parameter region satisfies DESI DR2 and lensing simultaneously | ⚠️ Worsens tension; open problem ([nvg_s8_tension_check.py](verification/nvg_s8_tension_check.py)) |
| 12 | White dwarf cooling & LZ 248 keV | WD age shift $\Delta t/t \approx -1.8\times 10^{-6}$; LZ kinematics excludes $0.783 / 1.245$ GeV elastic modes | Gaia/SDSS age error $\sim 5\%$; 248 keV requires DM $\ge 70.66$ GeV | ⚪ / 📐 Unobservably small for WDs; LZ unexplained ([audit](docs/reports/NVG_LZ_248KEV_AUDIT_RU.md)) |
| 13 | SN1987A dark-photon scan | Conditional sensitivity scan only; establishes no evidence or Raffelt limit without a transport solver and likelihood | Full proto-NS transport and spectral likelihood absent | ⚙️ Conditional (no evidence; physical claim blocked) |
| 14 | Physical NS cooling & $g$-modes | Exact barotropic null $N^2 = 0$; physical cooling/hyperon/g-mode claims blocked without $\Gamma_1(Y_i)$ and matrix elements | Legacy Cas A, hyperon, and static frequency tables withheld | ❌ Blocked ([nvg_cooling_gmode_dependency_audit.py](verification/nvg_cooling_gmode_dependency_audit.py)) |
| 15 | NANOGrav stochastic GW background | **Retired.** Live PBH audit shows $\sim 3\times 10^5$ amplitude deficit; bounce signal sits in the $\mu$Hz band | No mechanism claimed for the PTA signal | ❌ Retired ([P1-S3-pbh-nanograv-audit.json](verification/P1-S3-pbh-nanograv-audit.json)) |
| 16 | JWST early SMBH seeding ($z=6\text{–}15$) | **Object-level JWST seeding is retired:** no machine-readable catalogue, occupation model, or selection likelihood | $0.38 \times 4^N\,M_\odot$ ladder is an unnormalized sensitivity template only | ❌ Retired ([nvg_prospective_falsification_reach.py](verification/nvg_prospective_falsification_reach.py)) |
| 17 | FRB DM, DNA chirality & baryogenesis | Linked repeater DM/mass catalogues, molecular PVED coupling, and CP/washout likelihoods are absent | Previous empirical claims withdrawn | ❌ Retired ([nvg_chime_frb_check.py](verification/nvg_chime_frb_check.py), [nvg_dna_chirality.py](verification/nvg_dna_chirality.py)) |
| 18 | Quantum foundations (Bell / Madelung) | Madelung reproduces $|\psi|^2$; contextual quantized $\theta$ yields $S=2\sqrt{2}$; classical spacetime $\theta$ obeys $S \le 2$ | Quantum mechanics is an input representation, not derived from the classical action | 📐 Representation, not derivation ([nvg_bell_from_action.py](verification/nvg_bell_from_action.py)) |

---

## 2. Key Falsifiable Predictions (Forward Forecasts)

All items below are **forward forecasts** or **extension-conditional tests** with zero current observational confirmation.

| Channel / Observable | Model Forecast | Experiment | Falsification Criterion | Script |
|---|---|---|---|---|
| `PSR J0737-3039A` inertia & quadrupole | $\tilde{q} = 6.508$ (2nd-order Hartle, canonical EOS); conditional $I_A$ band | Radio timing (SKA / MeerKAT) | Measured $(I_A, \Lambda_A, R_A)$ outside the conditional forecast band | [nvg_ns_frozen_forecasts.py](verification/nvg_ns_frozen_forecasts.py) |
| Surface redshift & post-merger $f_{\rm peak}$ | $z_{\rm surf} \approx 0.222$ ($1.4\,M_\odot$); $f_{\rm peak} \approx 2.42$ kHz ($R_{1.6} \approx 12.65$ km) | STROBE-X / eXTP; LIGO O5 / ET | Direct $z_{\rm surf}$ or $f_{\rm peak}$ measurement outside forecast | [nvg_ns_redshift.py](verification/nvg_ns_redshift.py) |
| In-medium $\rho$-meson lineshape | Architecture selector: pole $\ge 755$ MeV (arch. A) or $715\text{–}755$ MeV (arch. B) | HADES / CBM (FAIR) / NICA | Fitted pole $< 715$ MeV or no spectral shift/broadening | [HADES_PREREGISTRATION.md](verification/HADES_PREREGISTRATION.md) |
| Neutrino mass triangle ($\theta$-seesaw) | $\Sigma m_\nu = 59.0$ meV; $m_\beta = 8.9$ meV (corrected from 29 meV); $m_{\beta\beta} \le 6.4$ meV | CMB-S4 · Project 8 · $0\nu\beta\beta$ | Robust $\Sigma m_\nu < 45$ meV, $m_\beta > 20$ meV, or $m_{\beta\beta} > 10$ meV | [nvg_sigma_mnu_corner.py](verification/nvg_sigma_mnu_corner.py), [nvg_katrin_mbeta.py](verification/nvg_katrin_mbeta.py) |
| O4 GW echo observer card | For $M_{\rm tot} = 61.8\text{–}70.2\,M_\odot$: $\Delta t_{\rm echo} = 20.0\text{–}23.9$ ms, $f_{\rm QNM} = 245\text{–}278$ Hz | Deep coherent stack of O4+ | Null deep stack on tabulated delays drives $R_{\rm eff} \to 0$ | [nvg_echo_o4_observers.py](verification/nvg_echo_o4_observers.py) |
| Tensor-to-scalar ratio & cosmic birefringence | $r(\ell < 10) < 0.001$; static branch $\beta_{\rm biref} = 0$ | LiteBIRD | Detection of $r > 0.001$ or confirmed $\beta \neq 0$ (pins $\gamma_{\rm topo}\Delta\theta$) | [nvg_litebird_prediction.py](verification/nvg_litebird_prediction.py), [nvg_cmb_birefringence.py](verification/nvg_cmb_birefringence.py) |
| RHIC Bell test (entanglement death above $T_c$) | $S_{\rm CHSH}(T > T_c = 157\text{ MeV}) \to 0$ (future protocol) | RHIC BES-II / heavy ions | Persistent $S > 2$ above deconfinement | [nvg_rhic_bell_test.py](verification/nvg_rhic_bell_test.py) |
| Horizonless tidal heating & superradiance | Zero EMRI absorption phase lag (vs ~32 cycles Kerr); Kerr PBH band $M \sim 10^{-6}\,M_\odot$ | LISA / astrophysics | Kerr-like ~32-cycle absorption lag or confirmed horizon in band | [nvg_tidal_heating_null.py](verification/nvg_tidal_heating_null.py), [nvg_theta_superradiance.py](verification/nvg_theta_superradiance.py) |

---

## 3. Model Architecture & Technical Audits

Built around the QCD vacuum mass budget $M_{\Omega,0} = 859 \pm 8$ MeV, the maintained repository partitions into three sectors with explicit mathematical boundaries:

1. **Dense Nuclear Matter (VMF) & Finite Nuclei:** The unmodified one-component `source_complete` action does not saturate or bind finite Thomas–Fermi droplets ($C = 2.72164 > 1/4$, $\lambda_c^{\rm formal} \approx 0.0011$ vs $1.05$; see [Q4 audit](docs/reports/NVG_SOURCE_COMPLETE_Q4_DROPLET_AUDIT_RU.md), [NN audit](docs/reports/NVG_SOURCE_COMPLETE_NN_STATIC_AUDIT_RU.md), [S-wave threshold](docs/reports/NVG_SOURCE_COMPLETE_FORMAL_SWAVE_THRESHOLD_AUDIT_RU.md)). Extending the scalar potential to sixth order with quartic vector melting (**Bridges 1–5**) resolves this gap: calibrating strictly on infinite nuclear matter predicts 6 doubly-magic nuclei in spherical Dirac–Hartree without nuclear refitting.
2. **Density-Tail Universality & Finite-Temperature Response:** For regular potentials $U \sim a W^p$, the asymptotic limit satisfies $\lim(P/\varepsilon) = \lim(dP/d\varepsilon) = \max(1/3, (p-2)/(p+2))$ ([report](docs/reports/NVG_DENSITY_UNIVERSALITY_RU.md)). At $T > 0$, **160 + 32** complex thermal response rows and **32** composition-Hessian rows satisfy full 5×5 Schur-Hessian Ward identities to $10^{-15}$ ([thermal master response](docs/reports/NVG_THERMAL_MASTER_RESPONSE_RU.md), [composition Hessian](docs/reports/NVG_THERMAL_COMPOSITION_HESSIAN_RU.md)).
3. **Cosmology, Regular Core & Practical Engineering Audit:** Includes the conditional type-II vortex sector ($\beta = 2.53 > 1$, [report](docs/reports/NVG_SOURCE_COMPLETE_CONDITIONAL_VORTEX_SECTOR_AUDIT_RU.md)), minimal vacuum kinematic closure ([report](docs/reports/NVG_SOURCE_COMPLETE_MINIMAL_VACUUM_CLOSURE_AUDIT_RU.md)), and the zero-weight [practical energy audit](verification/nvg_generator_practical_audit.py) ([Russian report](verification/NVG_GENERATOR_PRACTICAL_AUDIT_RU.md)), which rejects unsupported vacuum/over-unity claims and retains only a configurable waste-heat Seebeck thermoelectric model.

---

## 4. Quick Start & Reproducible Verification

The canonical scientific contour runs on **Python 3.12** using [requirements.txt](requirements.txt) and [constraints-python312.txt](constraints-python312.txt) (`NumPy 1.26.4`, `SciPy 1.13.1`, `mpmath 1.3.0`, `PyCBC`).

```bash
# 1. Create a Python 3.12 environment and install pinned dependencies
python3.12 -m venv .venv-research
.venv-research/bin/python -m pip install -r requirements.txt -c constraints-python312.txt
.venv-research/bin/python -m pip check

# 2. Run the canonical scientific verification front door (identical to GitHub Actions)
.venv-research/bin/python verification/run_verification.py
```

### What [`verification/run_verification.py`](verification/run_verification.py) Verifies
1. **Registry & Manifest Provenance:** Validates [`verification/registry.json`](verification/registry.json), [`verification/data/provenance.json`](verification/data/provenance.json), and [`verification/artifact_manifest.json`](verification/artifact_manifest.json).
2. **Predictive Integration Ledger (`P4-S1`):** Runs [`nvg_predictive_research_ledger.py`](verification/nvg_predictive_research_ledger.py) to authenticate live producers and artifacts fail-closed.
3. **Canonical Runtime Ledger:** Executes [`verification/run_nvg_suite.py`](verification/run_nvg_suite.py) and writes [NVG_FINAL_REPORT.md](docs/reports/NVG_FINAL_REPORT.md).
4. **Full Semantic Pytest Suite:** Runs `python -m pytest -q verification` across all 120+ test suites.

*(The separate [`verification/run_all_checks.py`](verification/run_all_checks.py) script is a process-only smoke harness and does not constitute scientific evidence.)*

<details>
<summary>Focused commands for the Unified 8-Bridge Validators & Thermal Bridges</summary>

```bash
.venv-research/bin/python verification/nvg_quartic_vector_melting_audit.py --validate verification/nvg_quartic_vector_melting_results.json
.venv-research/bin/python verification/nvg_two_condensate_unified_audit.py --validate verification/nvg_two_condensate_unified_results.json
.venv-research/bin/python verification/nvg_unified_dirac_hartree_probe.py --validate verification/nvg_unified_dirac_hartree_results.json
.venv-research/bin/python verification/nvg_unified_primary_gate_pairing_bridge.py --validate verification/nvg_unified_primary_gate_pairing_results.json
.venv-research/bin/python verification/nvg_unified_thermal_response_bridge.py --validate verification/nvg_unified_thermal_response_results.json
.venv-research/bin/python verification/nvg_unified_astrophysics_stability_bridge.py --validate verification/nvg_unified_astrophysics_stability_results.json
.venv-research/bin/python verification/nvg_unified_gravity_bridge_audit.py --validate verification/nvg_unified_gravity_bridge_results.json

PYTHONDONTWRITEBYTECODE=1 .venv-research/bin/python -B verification/nvg_thermal_master_response.py --output /tmp/thermal_master.json
PYTHONDONTWRITEBYTECODE=1 .venv-research/bin/python -B verification/nvg_thermal_composition_hessian.py --output /tmp/thermal_hessian.json
```

</details>

---

## 5. Repository Structure

```text
NVG-Research/
├── README.md                              # English overview (this file)
├── README_RU.md                           # Russian overview
├── docs/
│   ├── unified/                           # Unified 8-bridge system cycle reports (Cycles 1–3)
│   └── reports/                           # Standalone audits, PUBLICATION_STATUS.md, NVG_FINAL_REPORT.md
├── verification/                          # Executable physics, bridges 1–8, tests, manifests
│   ├── run_verification.py                # Canonical CI scientific front door
│   ├── run_nvg_suite.py                   # Canonical runtime ledger (TOV + Hinderer)
│   ├── nvg_predictive_research_ledger.py  # P4-S1 fail-closed integration ledger
│   ├── registry.json                      # Script, test & artifact role index
│   ├── artifact_manifest.json             # SHA-256 digests of generated artifacts
│   ├── data/                              # Input catalogues and provenance.json
│   └── nvg_*.py / test_*.py               # Physics modules, unified bridges & pytest suites
├── article/                               # Published papers and supporting documents
├── preprints/                             # Preprint bundles (TeX / PDF)
├── visualization/                         # Interactive 3D HTML dashboards
├── requirements.txt                       # Python dependencies
└── constraints-python312.txt              # Pinned package versions (NumPy 1.26.4, SciPy 1.13.1, etc.)
```

---

## 6. Historical Preprints (Zenodo)

<details>
<summary>Archived preprint bibliography (not current runtime evidence)</summary>

Published titles and links are retained as a historical bibliography (not current runtime evidence). Their conclusions are not automatically results of the current executable model.

- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20214457-blue.svg)](https://doi.org/10.5281/zenodo.20214457) *Lattice Sigma Terms as an Anchor for the Dense Nuclear Matter Equation of State*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269567-blue.svg)](https://doi.org/10.5281/zenodo.20269567) *Analytic Derivation of the Dense Matter Equation of State and Maximum Neutron Star Mass via QCD Vacuum Condensate Phase Transitions*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269725-blue.svg)](https://doi.org/10.5281/zenodo.20269725) *Geometric Truncation of Low-Multipole CMB Power and Null B-Mode Prediction from a QCD-Scale Euclidean Instanton Bounce*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269816-blue.svg)](https://doi.org/10.5281/zenodo.20269816) *A Discrete Cyclic Mass Hierarchy 4^N for Primordial Black Holes: Bridging Asteroid-Mass Dark Matter to Early JWST Heavy Seeds*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20270025-blue.svg)](https://doi.org/10.5281/zenodo.20270025) *Resolution of the Slow-Rotating Magnetar Paradox via QCD Vacuum Permeability Phase Transitions*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20270202-blue.svg)](https://doi.org/10.5281/zenodo.20270202) *Eliminating the Observer Effect: Wave Function Collapse as Deterministic Topological Reconnection in a Condensate Vacuum*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20463836-blue.svg)](https://doi.org/10.5281/zenodo.20463836) *Neutron Star Structure from a Single QCD Parameter: Equation of State, Tidal Deformability, and Cooling Threshold in the Null-Vector Gravity Framework*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20473318-blue.svg)](https://doi.org/10.5281/zenodo.20473318) *Dynamics of the QCD Vacuum Condensate Amplitude in Dense Matter and Cosmology* (historical proposal; vector energy diverges as $W^{-2}$ in the maintained `source_complete` action, so $2.05\,n_0$ density melting/bounce is unsupported)
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20476890-blue.svg)](https://doi.org/10.5281/zenodo.20476890) *Seven Results of the Vacuum Condensate: From Nuclear Matter to Quantum Mechanics* (Madelung & Einstein–Cartan illustrations; no torsion sector exists in the maintained action)
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20485836-blue.svg)](https://doi.org/10.5281/zenodo.20485836) *Resolution of the Hyperon Puzzle via QCD Vacuum Condensate Melting in the NVG/VMF Framework* (no complete beta-equilibrated EOS or likelihood is implemented in the maintained repo)

</details>

---

## Author

**Oleg Kirichenko** — Independent Researcher — urevich55@gmail.com

## License

MIT License — see [LICENSE](LICENSE) for details.
