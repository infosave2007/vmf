# Null-Vector Gravity (NVG) & Vacuum Mass Fraction (VMF) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![CI Build & Verification](https://github.com/infosave2007/vmf/actions/workflows/verify.yml/badge.svg)](https://github.com/infosave2007/vmf/actions/workflows/verify.yml)
> **Evidence status (2026-08-27):** No aggregate “confirmed” or prediction-count badge is maintained. Current runtime provenance and publication boundaries are documented in [PUBLICATION_STATUS.md](PUBLICATION_STATUS.md).

### Reproducible verification interface

The maintained computational surface is indexed in the machine-readable
[script and artifact registry](verification/registry.json). Run the ordinary
scientific gate from any working directory with
`python verification/run_verification.py`; it validates registry/provenance,
regenerates the canonical runtime ledger, and runs
`python -m pytest -q verification`. Registry roles distinguish `canonical`,
`validation`, `forecast`, `synthetic`, `retired`, `utility`, and `historical`
surfaces. The separate [`run_all_checks.py`](verification/run_all_checks.py)
command is a process-only smoke harness and is not scientific evidence.
Input provenance and generated-artifact ownership are recorded in the
[`provenance.json`](verification/data/provenance.json) and
[`artifact_manifest.json`](verification/artifact_manifest.json) manifests.

**Preprints:**
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20214457-blue.svg)](https://doi.org/10.5281/zenodo.20214457) *Lattice Sigma Terms as an Anchor for the Dense Nuclear Matter Equation of State*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269567-blue.svg)](https://doi.org/10.5281/zenodo.20269567) *Analytic Derivation of the Dense Matter Equation of State and Maximum Neutron Star Mass via QCD Vacuum Condensate Phase Transitions*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269725-blue.svg)](https://doi.org/10.5281/zenodo.20269725) *Geometric Truncation of Low-Multipole CMB Power and Null B-Mode Prediction from a QCD-Scale Euclidean Instanton Bounce*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269816-blue.svg)](https://doi.org/10.5281/zenodo.20269816) *A Discrete Cyclic Mass Hierarchy 4^N for Primordial Black Holes: Bridging Asteroid-Mass Dark Matter to Early JWST Heavy Seeds*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20270025-blue.svg)](https://doi.org/10.5281/zenodo.20270025) *Resolution of the Slow-Rotating Magnetar Paradox via QCD Vacuum Permeability Phase Transitions*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20270202-blue.svg)](https://doi.org/10.5281/zenodo.20270202) *Eliminating the Observer Effect: Wave Function Collapse as Deterministic Topological Reconnection in a Condensate Vacuum*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20463836-blue.svg)](https://doi.org/10.5281/zenodo.20463836) *Neutron Star Structure from a Single QCD Parameter: Equation of State, Tidal Deformability, and Cooling Threshold in the Null-Vector Gravity Framework*
  Historical publication bundle (not current runtime evidence). The maintained code exposes a hyperon-threshold proxy, raw RMF comparisons, and a calibrated cooling threshold; a complete beta-equilibrated EOS and independent likelihood are not implemented, so no hyperon-puzzle resolution is claimed.
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20473318-blue.svg)](https://doi.org/10.5281/zenodo.20473318) *Dynamics of the QCD Vacuum Condensate Amplitude in Dense Matter and Cosmology*
  Provides a rigorous mathematical derivation of the classical equations of motion for the radial mode $W(x)$ governing in-medium hadron masses, its gauge-invariant coupling to baryon currents, and its cosmological FLRW reduction. Demonstrates how vacuum melting $W \to 0$ violates the Strong Energy Condition (SEC) to trigger a smooth cosmological bounce at $n_B \approx 2.05\,n_0$, avoiding the Big Bang singularity.
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20476890-blue.svg)](https://doi.org/10.5281/zenodo.20476890) *Seven Results of the Vacuum Condensate: From Nuclear Matter to Quantum Mechanics*
  Obtains physical consequences from a single vacuum condensate order parameter $\Phi = \Wc\,\ee^{\ii\theta}$. Note: The theoretical framework has been mathematically refactored for strict rigor. The Heisenberg principle is derived as the classical Fourier limit of the Madelung fluid (with $p \equiv \hbar \nabla \theta$); Hawking radiation is derived classically via Stochastic Electrodynamics (SED) where the horizon acts as an Unruh bath for vacuum fluctuations; and the cosmological bounce is rigorously driven by Einstein-Cartan spin-torsion of the quark condensate, violating the SEC naturally without ad-hoc signs.
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20485836-blue.svg)](https://doi.org/10.5281/zenodo.20485836) *Resolution of the Hyperon Puzzle via QCD Vacuum Condensate Melting in the NVG/VMF Framework*
  Historical publication bundle (not current runtime evidence). It describes a proposed hyperonic phase-transition model; the maintained repository does not provide the independent EOS construction/likelihood needed to call this an empirical resolution.

## Overview

This repository contains the complete theoretical, numerical, and experimental framework for **Null-Vector Gravity (NVG)** and its underlying dense matter model, the **Vacuum Mass Fraction (VMF)**.

The core premise is that ~91% of the nucleon mass is generated by nonperturbative QCD dynamics (gluon field energy, confinement, trace anomaly). By treating this vacuum energy as a dynamic macroscopic field $\mathcal{W}$ anchored rigidly to lattice QCD data ($M_{\Omega,0} = 859$ MeV), we derive a parameter-free phenomenological bridge extending from nuclear physics to cosmology.

### The Unified Field Action
All three pillars of the framework are derived from a single, unified action density:

$$ S = \int d^4x \sqrt{-g} \left[ \frac{R}{16\pi G} - g^{\mu
u} \partial_\mu \Phi^* \partial_
u \Phi - V(|\Phi|) - \frac{1}{4} Z_{\rm EM}(\mathcal{W}) F_{\mu
u} F^{\mu
u} + \gamma_{\rm topo} \frac{\alpha_{\rm EM}}{8\pi} \theta F_{\mu
u} \tilde{F}^{\mu
u} \right] $$

where $\Phi(x) = \mathcal{W}(x) e^{i\theta(x)}$ is the complex vacuum condensate order parameter.
* **Vacuum Melting ($\mathcal{W}$):** Controls the amplitude of the vacuum energy density. In-medium melting in dense nuclear cores dictates the Equation of State (VMF) and resolves black hole/cosmological singularities when $\mathcal{W} \to 0$ and $V(0) = \frac{\lambda_v}{4} M_{\Omega,0}^4$ violates the Strong Energy Condition.
* **Emergent Time & Topological Flow ($\theta$):** The gradient of the Goldstone phase defines a preferred unit timelike vector field $u_\mu \equiv \partial_\mu \theta / \sqrt{-g^{\alpha\beta}\partial_\alpha \theta \partial_\beta \theta}$, anchoring the coordinate time direction. Its time evolution during collapse ($\dot{\theta} \neq 0$) couples to the EM field via the axion-like topological theta-term ($\theta F \tilde{F}$), driving exponential chiral magnetic field amplification in magnetars.

###### Mathematical Derivation of $\mathcal{W}$-field Dynamics & Melting
The dynamics of the vacuum condensate amplitude $\mathcal{W}(x)$ and its phase transition (melting) in a dense hadronic medium are derived from QFT first principles. The potential parameters are calibrated using pion-nucleon and strange QCD sigma-terms, and the coupling to the baryon current is established via the $\omega$-meson vector field. The violation of the Strong Energy Condition (SEC) during complete melting generates a quadratic cosmological correction of the form $-\rho^2/\rho_c$, preventing the Big Bang singularity.

For a complete step-by-step mathematical proof, see the local preprint: [NVG_VACUUM_W_FIELD_DERIVATION_EN.md](article/NVG_VACUUM_W_FIELD_DERIVATION_EN.md)

The framework consists of three main pillars:

### Pillar I: Dense Nuclear Matter (VMF)
The melting of the vacuum mass in dense environments supplies an Equation of State (EOS) calculation for neutron stars (see preprint [Zenodo 20463836](https://doi.org/10.5281/zenodo.20463836)). The maintained runtime surfaces:
- **Compute a hyperon threshold proxy and raw NL3+Lambda curves.** A complete beta-equilibrated hyperonic EOS and independent likelihood are not implemented, so no puzzle resolution is claimed.
- **Compute a calibrated Direct-Urca cooling threshold.** The threshold is an in-sample calibration, not an independent cooling-population confirmation.
- **Computes a canonical conditional neutron-star chain** (runtime $M_{\max}=2.048\,M_\odot$, with a CSS branch at $c_s^2=1/3$; transition parameters are selected in-sample and are not independent evidence).
- **Computes the canonical tidal chain and a transform-only I-Love-Q visualization.** The plotted $I(\Lambda)$ and $Q(\Lambda)$ points apply the Yagi–Yunes fits to runtime $\Lambda$ values; no independent moment-of-inertia or quadrupole solve is present, so this is not an EOS-universality test. The canonical transition is selected in-sample from J0740, GW170817, and NICER constraints; downstream NS comparisons are conditional/in-sample, not independent confirmations (`verification/nvg_iloveq_plot.py`; [runtime report](verification/fig_iloveq_universal_report.json)).
- **Eliminates empirical parameter degeneracies** by utilizing a single QCD vacuum anchor ($M_{\Omega,0} = 859$ MeV) with zero free parameters.
- Provides a forward $\rho$-meson mass/line-shape template for FAIR/HADES; no event likelihood is present.

### Pillar II: Cyclic Cosmology & Genesis (NVG)

As the universe collapses, the macroscopic melting of the $\mathcal{W}$-field naturally halts the Big Crunch at a critical density $\rho_c = M_{\Omega,0}^4/(\hbar c)^3 \approx 7.09 \times 10^4$ MeV/fm³. This density is $\sim 10^{77}$ orders below the Planck scale, placing the bounce entirely within semi-classical physics.

**Bounce dynamics** are derived (not postulated) from the FLRW minisuperspace reduction of the VMF action, yielding the modified Friedmann equation:

$$H^2 = \frac{8\pi G}{3}\rho_{\rm tot}\left(1 - \frac{\rho_{\rm tot}}{\rho_c}\right) - \frac{kc^2}{a^2} + \frac{\Lambda_{\rm eff}c^2}{3}$$

Key derived quantities (zero free parameters):

| Quantity | Value | Derivation |
|----------|-------|------------|
| Bounce density $\rho_c$ | $7.09 \times 10^4$ MeV/fm³ | $M_{\Omega,0}^4/(\hbar c)^3$ |
| $\rho_c / \rho_{\rm Planck}$ | $2.5 \times 10^{-77}$ | Semi-classical regime |
| Bounce timescale $t_b$ | $3.76 \times 10^{-6}$ s | $(8\pi G\rho_c/3)^{-1/2}$ |
| Bounce temperature $T_b$ | 432 MeV | Stefan-Boltzmann QGP ($g_*=47.5$) |
| Holographic entropy (Universe) | $2.15 \times 10^{91}$ | $4\pi r_0^2 / 4\ell_{\rm Pl}^2$ |
| CMB/BAO $\delta H/H$ | $\sim 10^{-38}$ | Exact ΛCDM compatibility |

**Genesis Phase — The First Cycle:**
The origin of the first universe is modeled as a Euclidean instanton tunneling event. Under standard Hartle-Hawking boundary conditions, the universe is born exactly at $\rho = \rho_c$ with $\dot{a}=0$. The finite instanton radius is:

$$r_c = \frac{c}{\sqrt{8\pi G \rho_c / 3}} \approx 1.13 \text{ km}$$

This gives an initial mass of only $M_1 \approx 0.38\,M_\odot$ and a first-cycle lifetime of **~5.9 microseconds**. Time itself is proposed to emerge as the Goldstone mode ($dt \propto d\theta$) of spontaneous $U(1)$ symmetry breaking in the vacuum phase sector.

**Tolman's Entropy Snowball:**
Each cycle generates irreversible entropy (radiation, black hole formation), which is preserved through the bounce, causing subsequent cycles to expand larger and live longer. Through this "snowball effect":
- **Cycle 1:** $r_c \approx 1.13$ km, lifetime $\sim 5.9\,\mu$s
- **Cycle 77 (now; ≈76 completed doublings):** $M \approx 10^{56}$ g, turnaround lifetime $\approx 24.7$ Byr

**CMB Low-$\ell$ Prediction:**
The finite instanton size $r_c$ stretched by $N_e \approx 53$ e-folds (calibrated to the local $H_0$) spans the present Hubble horizon by construction; the falsifiable content is the resulting deterministic infrared cutoff that suppresses CMB multipoles $\ell=2,3$. The maintained script runs a documented lite likelihood over vendored Planck-shaped arrays whose provenance is unknown — it is **not the Planck likelihood** and its $\Delta\chi^2$ (about $+0.9$ for the default cutoff) is descriptive only, not a Planck-fit p-value or confirmation. The cutoff does not shift the CMB-inferred $H_0$ ([nvg_cmb_lowl_refit.py](verification/nvg_cmb_lowl_refit.py)).

### Pillar III: Black Hole Singularity Resolution & Information Preservation
The framework replaces the black hole singularity with a **regular de Sitter core** (Hayward metric), where the de Sitter core length $\ell = \sqrt{3c^2/(8\pi G\rho_c)} \approx 1.128$ km — mass-independent, fixed by the QCD anchor, not a free parameter — sets the metric (the mass-enclosing radius $r_0 = (3M/4\pi\rho_c)^{1/3}$, $3.35$ km at $10\,M_\odot$, is a distinct quantity). For a $10\,M_\odot$ black hole this gives two horizons (inner Cauchy: 1.15 km, outer: 29.34 km) and a finite Kretschmann scalar $K(0) = 24/\ell^4 = 14.86\,\text{km}^{-4}$.

**Singularity resolution mechanism:** At density $\rho \to \rho_c$, the strong energy condition is violated ($\varepsilon + 3P < 0$), halting collapse and replacing the singularity with a de Sitter vacuum. This is the same mechanism that produces the cosmological bounce, applied at stellar scales. The de Sitter core is the fixed point of NVG's own modified Friedmann equation ($H \to 0$, $w \to -1$ as $\rho \to \rho_c$): once the vacuum melts and $\varepsilon + 3P < 0$, the effective repulsion halts the collapse and the interior asymptotes to pure de Sitter. (No Ricci flow is invoked — a positive-curvature Einstein metric such as de Sitter is a *shrinking* Ricci soliton that collapses to a point in finite time, **not** a Ricci-flow fixed point, so Perelman's Riemannian, fictitious-time machinery does not apply to this Lorentzian, physical-time statement.)

**Information preservation:** The absence of a singularity eliminates the "point of information destruction." NVG provides a concrete physical mechanism for unitarity:
- Holographic compression: entropy is compressed by $\sim 10^{32}\times$ at the core but never destroyed.
- Unitary transfer: $\mathcal{I}_{n+1} = \mathcal{U}_b\,\mathcal{I}_n$ — information is carried through the core via the $\mathcal{W}$-sector.
- The regular causal structure (two horizons, no singularity) ensures no loss-of-information region exists.

**Key advantage over competing models:** Unlike generic Bardeen/Hayward (free parameter $g$), fuzzballs ($10^{500}$ string vacua), or loop quantum gravity (Planck-scale, untestable), NVG derives $r_0$ from a **measured** QCD quantity ($M_{\Omega,0} = 859$ MeV), yielding a falsifiable prediction: gravitational wave post-merger echoes at $\Delta t_{\rm echo} \approx 22$ ms spin-corrected for a 65 $M_\odot$ merger (an order-of-magnitude value — the tortoise-coordinate delay is logarithmically sensitive to the near-horizon UV cutoff, spanning $\sim 16$–$34$ ms over the physically reasonable range, so the $M_{\Omega,0}$ $\pm 8$ MeV margin is subdominant; the robust, parameter-free content is the *existence* of a macroscopic finite-time echo, LIGO-testable). Notably, this delay strongly hints at a resonance effect: the inner de Sitter core supports standing $\mathcal{W}$-field oscillations with a period $T_1 \approx 42\,\mu$s, suggesting the 22.0 ms macroscopic echo window contains a fine sub-structure of precisely $\approx 524$ internal resonant cycles, producing a distinct frequency comb rather than a single simple reflection. Note that this internal $42\,\mu$s resonance is a distinct physical channel from the standard external macroscopic Kerr ringdown ($\sim 3.6$ ms), and the $22$ ms macroscopic delay corresponds to a tortoise-coordinate integral bounded by a physical wave-packet cutoff ($\sim 2$ mm from the horizon). Furthermore, because the inner horizon's effective Hawking temperature is incredibly low ($T_H \sim 10^{-15}$ K), the Boltzmann absorption factor $\exp(-hf/k_B T_H)$ is astronomically small. This renders the inner boundary a nearly perfect mirror, predicting a highly distinct **long-lived, weakly attenuating train of macroscopic echoes** (dozens to hundreds of repeats), sharply differentiating NVG from other exotic compact object models that predict rapid attenuation (2-3 pulses).

**Event Horizon Telescope (EHT) Indistinguishability (Known non-falsifiable channel):** The outer photon sphere (determining the black hole shadow) for NVG deviates from standard Schwarzschild GR by only $\sim -0.002\%$. Thus, the regular core is perfectly hidden from EHT observations; the *only* way to test the model is through GW post-merger echoes.

## Observational Status & Runtime Evidence Boundaries (May 2026)

The NVG/VMF framework has **zero free cosmological parameters** — every number is derived from a single QCD input: $M_{\Omega,0} = 859$ MeV. 
For the sake of scientific rigor, the results are divided into: direct unique predictions, consistency checks (null tests), and falsifiable forecasts.

### 1. Runtime Comparisons and Model-Status Ledger
These rows summarize declared runtime calculations, formal identities, conditional comparisons, and forecasts; they are not an aggregate confirmation count.

**Status legend:** ✅ compatible with data · 📏 interval prediction · 🔭 forward prediction (no measurement yet) · ⚙️ calibration / consistency check · ⚪ null test · 📐 mathematical result (not an experimental confirmation) · ❓ conjectural · ❌ open problem / retracted · ⏳ awaiting experiment. Falsifiable rows state the measurement that would exclude the model. Statuses are per-row. The global-significance entry point computes descriptive conditional pulls but **withholds any global p-value** because no traceable held-out producer exists; the README table is presentation metadata, not a source of statistics. NS values below are runtime outputs of the canonical TOV chain and all comparisons are conditional/in-sample with zero independent evidence weight. Provenance note: $T_c = 157$ MeV, used across the scripts, is *adopted* from lattice QCD (HotQCD: $156.5 \pm 1.5$ MeV) — an input identification, not an NVG prediction, and it is deliberately excluded from the test count.

| # | NVG Prediction | Observational Data | Status |
|---|---|---|---|
| 1 | Nucleon mass: 91% from non-perturbative QCD ($M_{\Omega,0} = 859 \pm 8$ MeV) | Lattice QCD $\sigma_{\pi N} \approx 44$ MeV, $\sigma_{sN} \approx 30$ MeV | ✅ Confirmed |
| 2 | $M_{\max} = 2.048\,M_\odot$ (runtime canonical TOV chain; transition selected in-sample) | PSR J0740+6620: $2.14 \pm 0.10\,M_\odot$ ($-0.92\sigma$). A confirmed NS above the declared mass bound would test the chain; this comparison is conditional/in-sample, not an independent confirmation | ⚙️ Conditional/in-sample |
| 3 | $R_{1.4} = 12.550$ km (runtime canonical TOV chain) | NICER J0030: $12.2 \pm 0.5$ km ($+0.70\sigma$). The transition was selected using J0740/GW170817/NICER constraints, so this is conditional/in-sample with zero independent evidence weight | ⚙️ Conditional/in-sample |
| 4 | Genesis Instanton $r_c \to$ e-folds bounded to $N_e \in [52.68, 53.38]$ for cycle $n=77$; the position inside the interval, $N_e = 53.08$, is set by the local $H_0$ | $R_{H0} = c/H_0 \approx 1.27 \times 10^{28}$ cm; $N_e = \ln(R_{H0}/r_c)$ at $H_0 = 72.8$ km/s/Mpc ([nvg_hubble_tension.py](verification/nvg_hubble_tension.py) / [nvg_genesis_observable.py](verification/nvg_genesis_observable.py)) | 📏 Interval prediction |
| 5 | NS cooling dichotomy via a direct-Urca threshold at $1.45 M_\odot$ (threshold position set by the coupling $\alpha_v$) | Cas A (slow) vs Vela (fast): the two-regime cooling is reproduced ([nvg_direct_urca.py](verification/nvg_direct_urca.py)) | ⚙️ Calibrated threshold |
| 6 | Tidal deformability: runtime $\Lambda_{1.4} = 519.4$ (TOV + Hinderer) | GW170817 declared input: $\Lambda_{1.4}=190^{+390}_{-120}$; runtime pull $+0.84\sigma$. This row is conditional/in-sample; no independent evidence weight is assigned | ⚙️ Conditional/in-sample |
| 7 | In-medium $\rho$: HADES data show the meson **completely melted** into a thermal continuum (no peak exists — Nature Phys. 15, 1040), so peak-position tests are inapplicable. The consistent theory leaves two architectures differing in the in-medium pole entering the excess *line shape*: A ($\approx 770$ MeV, broadening-only) vs B ($\approx 735$ MeV, scalar-field pull) | Preregistered selector windows and feasibility (3$\sigma$ with $\sim 3$–$4\times 10^3$ excess counts): [HADES_PREREGISTRATION.md](verification/HADES_PREREGISTRATION.md), [nvg_hades_lineshape_feasibility.py](verification/nvg_hades_lineshape_feasibility.py); decided by the collaboration's 2026–27 in-medium analysis | ⏳ Preregistered, pending HADES |
| 8 | Cosmic bounce density: $\rho_c = 7.09 \times 10^4$ MeV/fm³ (strictly from $M_{\Omega,0}^4$) | The bounce mechanism (ä > 0) is rigorously derived using Einstein-Cartan gravity: the intrinsic spin-spin torsion of the quark condensate provides an exact negative quadratic density correction, naturally violating the SEC to produce H=0. | ✅ Mathematically Rigorous (Einstein-Cartan) |
| 9 | Hubble constant: cycle-77 turnaround horizons bound $H_0 \in [54.3, 108.5]$ km/s/Mpc ($R_{77} = r_c \cdot 2^{76}$); the mid-cycle value is $H_0 = H_{77}/\sqrt{2} \approx 76.8$ km/s/Mpc (measure-dependent center: $62$–$81$) | SH0ES ($73.04 \pm 1.04$) lies within 5% of the mid-cycle value; Planck ($67.4$) within the measure spread. The IR cutoff leaves the CMB-inferred $H_0$ unchanged (quantitative re-fit: [nvg_cmb_lowl_refit.py](verification/nvg_cmb_lowl_refit.py)) | 📏 Interval prediction |
| 10 | Surface gravitational redshift: $z_{\rm surf} \approx 0.222$ for $1.4\,M_\odot$ (computed from runtime $R_{1.4}=12.550$ km) | Direct measurements are currently absent; this is a forward calculation testable by STROBE-X/eXTP ([nvg_ns_redshift.py](verification/nvg_ns_redshift.py)) | 🔭 Forward prediction |
| 11 | Multi-meson instantaneous mass-shift hierarchy at $2n_0$: $\rho, \omega$ ($-20\%$), $K^*$ ($-7.8\%$), $\phi$ ($-2.9\%$), $J/\psi$ ($-0.4\%$) — light non-Goldstone mesons shift most (observable spectral shifts are smaller, per row 7) | HADES, CBM (FAIR), NICA, LHC in-medium invariant-mass spectra ([fair_hades_link.py](verification/nvg_fair_hades_link.py)) | ⏳ Pending verification |
| 12 | Cosmic bounce temperature: $T_b = 432$ MeV (derived from Stefan-Boltzmann with $g_* = 47.5$) | Consistent with QGP deconfinement scale ($T_c \approx 155\text{-}175$ MeV) at bounce | ✅ Consistent / Falsifiable |
| 13 | Effective vacuum dielectric constant: $\varepsilon_{\rm eff} \approx 0.135\,\varepsilon_0$ in NS cores (from $e^{-2\alpha_v f_{\rm melt}}$ with phenomenological melting parameters $\kappa$) | Amplifies magnetar seed fields by $1/\sqrt{\varepsilon_{\rm eff}} \approx 2.7\times$ | ⚙️ Scale estimate |
| 14 | Relic dark matter: the observed $\Omega_{\rm DM} = 0.268$ determines the condensate self-coupling $\lambda_v$ | The inferred $\lambda_v$ lands in the $f_0(1370)$–$f_0(1500)$ scalar-meson range ([nvg_relic_dark_matter.py](verification/nvg_relic_dark_matter.py)) | ⚙️ Consistency check |
| 15 | NS core speed of sound: the quark phase follows a CSS ansatz with $c_{s}^2 = 1/3$ (conformal limit) | Compatible with joint NICER+LIGO posterior limits ([speed_of_sound_curve.py](verification/nvg_speed_of_sound_curve.py)) | ⚪ Ansatz parameter |
| 16 | First cycle duration: $\tau_1 = 5.9\,\mu\text{s}$ | Derived from QCD bounce scale $\rho_c \to t_b$; solves CCC/LQC boundary problem | ✅ Consistent / Falsifiable |
| 17 | Joint NS comparison: conditional/in-sample reduced $\chi_\nu^2 = 0.684$ over 3 runtime rows (calibration target excluded) | Runtime outputs $M_{\max}=2.048$, $R_{1.4}=12.550$, $\Lambda_{1.4}=519.4$ from the canonical chain; transition selected on J0740/GW170817/NICER ([nvg_joint_ns_inference.py](verification/nvg_joint_ns_inference.py)) | ⚙️ Conditional/in-sample |
| 18 | Scalar Glueball mass: $M_{\rm glueball} = 2 M_\Omega \approx 1.72$ GeV | Lightest scalar glueball ($0^{++}$) from the trace-anomaly excitation vs **Lattice QCD** $1.7 \pm 0.1$ GeV. Caveat: this is a theory-vs-lattice comparison — the *experimental* glueball identification remains unsettled ($f_0(1710)$ candidacy debated; BESIII's $X(2370)$ unconfirmed in other channels) | ✅ Compatible with lattice (experiment unsettled) |
| 19 | Primordial GW background from the bounce: anchor $f_{77} = 62.8$ nHz and tooth spacing derived from $t_b$, adiabatic redshift and the Tolman law ([primordial_gw_comb.py](verification/nvg_primordial_gw_comb.py)) | $(\alpha, \beta/H)$ derived from the action place the bounce signal at $18$–$42\,\mu$Hz with $\Omega_{\rm GW} h^2 \sim 10^{-9}$ ($\mu$Ares band); the PTA-band tail is negligible — the NANOGrav signal is **not** the NVG bounce ([nvg_recondensation_dynamics.py](verification/nvg_recondensation_dynamics.py)) | 🔭 $\mu$Hz forward prediction |
| 20 | Strong-Field Periastron Advance & PPN Parameters | NVG vacuum polarization correction is $\approx 1.6 \times 10^{-10}$ for double pulsar; Solar System PPN parameters $\gamma_{\rm PPN} = 1.0$ and $\beta_{\rm PPN} = 1.0$ exactly satisfy Cassini and LLR bounds ([weak_field_ppn.py](verification/nvg_weak_field_ppn.py)) | ✅ Within Observational Limits |
| 21 | JWST SMBH Mass Spectrum (z = 6–15) | A conditional growth template can be evaluated for a selected mass rung, but no machine-readable JWST object catalogue, occupation model, or selection likelihood is present | ❌ Retired (missing catalogue/likelihood) |
| 22 | FRB DM Population Statistics | No repeater-linked DM/magnetar-mass catalogue or population likelihood is present; the former synthetic KS comparison is retired | ❌ Retired (missing catalogue/likelihood) |
| 23 | Higgsless Proton-to-Pion Mass Ratio | Baryon/pion mass ratio ($M_p \approx 941.4$ MeV, $M_\pi \to 0$ in the chiral limit) anchored to $M_\Omega = 859$ MeV — standard chiral-symmetry-breaking / trace-anomaly mass generation restated in NVG variables | ✅ Consistent (derivation check) |
| 24 | QCD Phase Diagram Vacuum Melting | Vacuum melting boundary at $T_{\rm melt}(\mu_B) = T_b (1 - (\mu_B/1200)^4)^{0.25}$ MeV with $T_b \approx 432$ MeV at $\mu_B=0$ | ✅ Consistent / Falsifiable |
| 25 | PTA-LIGO O4 SGWB Cross-Correlation | Primordial SGWB turn-down at $f > 145$ nHz predicts high-frequency amplitude $\Omega_{\rm GW}(100\text{ Hz}) < 10^{-15}$ | ✅ Consistent with Null |
| 26 | NICER radii near $1.4\,M_\odot$ (J0437, J0614) | Runtime canonical chain gives $R \approx 12.55$ km at $1.4\,M_\odot$; J0437 $11.36 \pm 0.8$ km is a $+1.49\sigma$ pull, while the edge-on J0614-3329 comparison is a separate observational check. These comparisons are conditional/in-sample; no independent evidence weight is assigned ([nvg_ns_nicer_joint_audit.py](verification/nvg_ns_nicer_joint_audit.py)) | ⚙️ Conditional/in-sample |
| 27 | CMB Temperature $T_{\rm CMB} = 2.725$ K | The result scales linearly with the arbitrary normalization $a_{\rm bounce} = 1$ cm, so the numerical match carries no predictive content; only the adiabatic scaling form is model content ([cmb_temperature.py](verification/nvg_cmb_temperature.py)) | ⚪ No predictive content |
| 28 | Baryon asymmetry | A bounded-temperature no-go argument is retained as a formal result, but no baryon-number/CP-violating source, washout network, or independent abundance likelihood is implemented; the former empirical cogenesis closure is retired ([nvg_baryon_asymmetry.py](verification/nvg_baryon_asymmetry.py)) | 📐 Formal no-go; empirical closure retired |
| 29 | Post-merger peak frequency $f_{\rm peak} \approx 2.42$ kHz | Derived from runtime $R_{1.6}\approx12.65$ km with the stated numerical-relativity fit; no post-merger signal observed yet — testable by LIGO O5 / Einstein Telescope | 🔭 Forward prediction |
| 30 | Quiescent temperature of SGR 1935+2154 | With the heating luminosity set near the observed value and a typical spot size, $T_{\rm spot}$ follows from Stefan-Boltzmann by construction; the VMF content is the qualitative Urca dichotomy (light magnetar keeps a hot spot) ([sgr_temperature.py](verification/nvg_sgr_temperature.py)) | ⚙️ Consistency illustration |
| 31 | LiteBIRD B-mode Polarization Cutoff | predicted tensor-to-scalar ratio $r(l)$ drops below 0.001 at large scales $l < 10$ ([litebird_prediction.py](verification/nvg_litebird_prediction.py)) | ✅ Consistent / Falsifiable |
| 32 | $S_8$ structure growth | **Open problem.** The maintained calculation gives $S_8 \approx 0.851$ ($4.4\sigma$ from the weak-lensing input): the input-dependent $w_0/w_a$ growth shift moves away from the lensing value. The retired empirical IDE drag-factor resolution claim is not evidence ([nvg_s8_tension_check.py](verification/nvg_s8_tension_check.py)) | ⚠️ WORSENS (not an independent prediction) |
| 33 | NANOGrav SGWB | Retired attribution. The maintained PBH abundance cross-check leaves a large amplitude deficit and the bounce calculation radiates in the $\mu$Hz band; NVG supplies no mechanism for the PTA signal ([nvg_nanograv_background.py](verification/nvg_nanograv_background.py)) | ❌ Retired (no mechanism) |
| 34 | Higgs boson mass shift $\delta m_H \approx 4.4$ MeV | Propagator mass shift $\delta m_H = g_s^2 W_0^2 / 2m_H$ induced by scalar QCD vacuum condensate, within LHC experimental limits ([higgs_mass_shift.py](verification/nvg_higgs_mass_shift.py)) | ✅ Within LHC Limits |
| 35 | PBH DM Fraction Peak | The discrete mass ladder follows from the theory (spacing $\times 2$ per rung); the abundance peak ($N=-21$, $\sim 10^{20}$ g) is in the asteroid-mass window. Critically, since this mass is $\sim 13$ orders of magnitude below $M_{\rm crit}$ (Row 54), these PBHs are strictly naked de Sitter remnants lacking event horizons, fundamentally altering their Hawking evaporation signatures ([pbh_dark_matter.py](verification/nvg_pbh_dark_matter.py)) | ⚙️ Ladder predicted; abundance calibrated |
| 36 | White Dwarf cooling age shift | Predicted effect $\Delta t/t \approx -1.8 \times 10^{-6}$ is far below Gaia/SDSS age uncertainties ($\sim 5\%$) — indistinguishable from zero ([wd_cooling.py](verification/nvg_wd_cooling.py)) | ⚪ Null test (unobservably small) |
| 37 | Neutron star core g-modes | Runtime WKB forecast over an assumed composition grid gives periods in a 50–150 ms sensitivity band; no detector likelihood or confirmation is available ([nvg_ns_g_modes.py](verification/nvg_ns_g_modes.py)) | 🔭 Forecast (assumed composition) |
| 38 | SN1987A dark-photon sensitivity scan | The current reference scan includes a 19.8% mass-drop point with $L_A=8.265\times10^{55}$ erg/s, above the stated Raffelt reference. Without a sourced SN1987A likelihood and validated transport model, it establishes no mass-drop bound or limit status ([nvg_dark_photon_observables.py](verification/nvg_dark_photon_observables.py)) | ⚙️ Conditional scan (no evidence) |
| 39 | de Sitter core standing waves | W-field oscillations inside regular cores (period $T_1 \approx 42\,\mu$s for $65 M_\odot$) predict GW echo sub-structure ([ds_core_oscillations.py](verification/nvg_ds_core_oscillations.py)) | ⏳ Awaiting future data |
| 40 | Strong CP Problem Solution | $\bar{\theta}_{\rm QCD} = 0$ automatically: global minimum of $V(W_0, \theta)$ is at $\theta = 0$ due to vacuum condensate structure. No Peccei-Quinn mechanism needed, given the identification $\bar\theta_{\rm QCD} \equiv \theta$ (a physical postulate) ([strong_cp_solution.py](verification/nvg_strong_cp_solution.py)) | 📐 Semi-rigorous derivation |
| 41 | Arrow of Time from Topology | Entropy current $s^\mu = s \cdot u^\mu$, $u^\mu \propto \partial^\mu \theta$ — monotonic entropy growth follows from $Q = (1/2\pi)\oint d\theta = 1 > 0$. H-theorem for the direction of time; the quantitative entropy budget remains open ([arrow_of_time.py](verification/nvg_arrow_of_time.py)) | 📐 Theorem (qualitative) |
| 42 | Double-slit interference from vacuum hydrodynamics | $|\psi|^2$ pattern reproduced by Huygens-Fresnel integral over vacuum phase $\theta$ (Madelung representation), $r_{\rm Pearson} = 1.000$ ([double_slit_madelung.py](verification/nvg_double_slit_madelung.py)) | 📐 Reproduces standard QM (Madelung) |
| 43 | Null WIMP signal in direct detectors | EFT cross-section sensitivities are compared with named experiment curves as benchmark display inputs; no detector event likelihood is implemented and no exclusion is claimed ([dm_direct_detection.py](verification/nvg_dm_direct_detection.py)) | ⚙️ Model sensitivity (no evidence) |
| 44 | Bell-CHSH correlations in the condensate | The correlation form $E = -\cos(a-b)$ is postulated (preprint Limitations: conjectural); by Bell's theorem a local derivation is impossible — a shared phase read out locally is a hidden variable giving $S \le 2$, so any derivation from the action must contain an explicitly nonlocal or contextual element. **Resolved as a dichotomy** ([nvg_bell_from_action.py](verification/nvg_bell_from_action.py)): a classical spacetime $\theta$ of any dynamics gives $S \le 2$ (verified by exhaustion) — excluded by loophole-free experiments; the quantized W-field yields the configuration-space $\theta$ automatically ([nvg_bell_contextual.py](verification/nvg_bell_contextual.py) reproduces $S = 2\sqrt{2}$) but then QM is the input. The quantum block is a consistent hydrodynamic *representation*, not a derivation; its falsifiable content is $S(T > T_c) \to 0$ (row 51) | 📐 Resolved: representation, not derivation |
| 45 | Heisenberg uncertainty | $\Delta x \cdot \Delta p \geq \hbar/2$ is rigorously derived as the classical Fourier limit (Cauchy-Schwarz inequality) for the Madelung hydrodynamics of the vacuum condensate, where $\hbar$ serves purely as the dimensionful scale coupling phase gradient to momentum ($p = \hbar\nabla\theta$). | 📐 Rigorous (Classical Hydrodynamics) |
| 46 | Wave function collapse = θ-phase thermalization | "Measurement" = coupling $\theta$ to thermal reservoir (apparatus). $\tau_{\rm collapse} = \hbar/(k_B T) = 25$ fs at 300 K. Born rule = Boltzmann weight $P \propto e^{-V(\theta)/T}$. Supplies decoherence dynamics and the timescale; pointer-state selection remains open ([wavefunction_collapse.py](verification/nvg_wavefunction_collapse.py)) | 📐 Semi-rigorous |
| 47 | Neutrino mass from θ-seesaw | $m_\nu = (\alpha_s/4\pi)^2 v_{\rm EW}^2/f_a$: ABJ chiral anomaly couples θ-mode to lepton current. Single parameter $f_a = 1.07 \times 10^{11}$ GeV gives $m_3 = 50.3$ meV (atm.) AND $m_\theta = 53$ μeV (ADMX). No right-handed neutrinos; the axion is a separate extension field with its own $f_a$ (per the $\theta$-sector audit — the condensate phase itself is the $\eta'$). $\Sigma m_\nu = 59$ meV < 72 meV (DESI) ([neutrino_seesaw.py](verification/nvg_neutrino_seesaw.py)). **Adopted as the primary neutrino sector** (passes DESI DR2 in $\Lambda$CDM unconditionally; carries the ADMX co-prediction $m_\theta = 53\,\mu$eV) | ⚙️ Primary sector (scale estimate) |
| 48 | Quantum gravity without quantization | Hawking radiation is rigorously derived using Stochastic Electrodynamics (SED): the classical $\theta$-field zero-point fluctuations appear as a thermal Planck spectrum to an observer at the Schwarzschild horizon (Unruh effect), exactly producing $T_H = \hbar c^3/(8\pi G M k_B)$. | ✅ Rigorous (Stochastic Electrodynamics) |
| 49 | Fine structure constant: physical-cutoff interpretation | Recomputed with $W_0 = 859$ MeV: $1/\alpha(M_Z) = 126.6$ is standard 1-loop QED running (independent of $W_0$; 2-loop hadronic terms close the gap to $127.95$). The NVG content is interpretational — the UV cutoff is the physical condensate scale and $\alpha_{\rm bare} = 1/132.8$ is inferred from the measured $1/137$, not derived ([fine_structure.py](verification/nvg_fine_structure.py)) | ⚙️ Reinterpretation (no independent prediction) |
| 50 | Antimatter as $\theta \to -\theta$ | C-conjugation and phase reversal are formal identities. The maintained baryogenesis calculation is retired because no baryon-number/CP-violating source dynamics or washout likelihood is implemented ([nvg_baryon_asymmetry.py](verification/nvg_baryon_asymmetry.py)) | 📐 Formal identity; baryogenesis retired |
| 51 | 🔥 RHIC Bell Test — entanglement death | $S_{\rm CHSH}(T > T_c = 157\text{ MeV}) \to 0$ is a future protocol with no current detector data; it is a forecast, not a confirmation ([rhic_bell_test.py](verification/nvg_rhic_bell_test.py)) | ⏳ Awaiting RHIC BES-II |
| 52 | Homochirality from QCD topology | A QCD-to-molecular PVED coupling, reaction kinetics, and independent chirality likelihood are absent; the former $>99\%$ closure is retired ([nvg_dna_chirality.py](verification/nvg_dna_chirality.py)) | ❌ Retired (missing coupling/likelihood) |
| 53 | Primordial Gravitational Waves (BICEP/Keck) | NVG Genesis predicts that the universe starts from a de Sitter core (bounce at maximum QCD density, $\sim 1$ GeV), not from Planck-scale inflation. Because of this, the tensor-to-scalar ratio $r \sim (E_{bounce}/E_{Planck})^4 \sim 10^{-76}$, meaning primordial gravitational waves on observable scales are completely absent. | ✅ Consistent with BICEP/Keck ($r < 0.033$) |
| 54 | Critical Horizon Mass | $M_{\rm crit} \in [0.97, 1.01]\,M_\odot$. Derived from the precise lattice QCD measurement $M_{\Omega,0} = 859 \pm 8$ MeV ($M_{\rm crit} \propto M_{\Omega,0}^{-2}$). In closed form $M_{\rm crit} = \tfrac{9}{8\sqrt{2\pi}}\,M_{\rm Pl}^3/M_{\Omega,0}^2$ — the Chandrasekhar-type combination $M_{\rm Pl}^3/m^2$ at the QCD condensate mass, so its $\sim\!1\,M_\odot$ value is the generic gravity+QCD scale (the same one behind white-dwarf/neutron-star masses), not tuned. Honest scope: this scale is *generic* to any Hayward/Bardeen regular core with a QCD-scale cutoff, so the $\sim\!1\,M_\odot$ value does not by itself discriminate NVG — the falsifiable content is the horizonless remnant below the band, not the mass value ([nvg_mcrit_chandrasekhar.py](verification/nvg_mcrit_chandrasekhar.py), [nvg_mcrit_family.py](verification/nvg_mcrit_family.py)). Objects lighter than this (e.g., PBHs in the asteroid-mass window) cannot form event horizons, rendering them naked regular de Sitter remnants rather than true black holes. Note: this is a fundamental gravitational limit, completely distinct from the astrophysical stellar "mass gap" ($2-5\,M_\odot$). | 🔭 Forward prediction |
| 55 | Hawking Temperature Ceiling | Exact Hayward temperature on the QCD anchor ($l = 1.128$ km): $T_H \to 0$ at $M_{\rm crit}$ and the global maximum over all masses is $T_{\max} = 3.6\times10^{-8}$ K at $M = 1.29\,M_\odot$ — every black hole the theory admits is colder than the CMB throughout the cycle, so net Hawking mass loss never occurs. Below $M_{\rm crit}$ objects are horizonless with zero Hawking flux, so the evaporation bounds that close the ordinary-PBH dark-matter window below $\sim 10^{17}$ g do not apply; the $10^{10}$–$10^{17}$ g range is available to NVG remnants ([nvg_hayward_evaporation.py](verification/nvg_hayward_evaporation.py), [nvg_pbh_dark_matter.py](verification/nvg_pbh_dark_matter.py)). Exclusion criteria: a confirmed PBH evaporation burst (HAWC/CTA/Fermi), a confirmed Hawking component of the MeV $\gamma$ background (AMEGO-X/e-ASTROGAM targets), or any confirmed sub-solar black hole with a horizon | 🔭 Forward prediction |

**5. Inner-Horizon Saturation & Area Deficit:**
Scanning the horizon roots of the Hayward metric as $M \to \infty$ shows that the inner horizon $r_{\rm in}$ asymptotes to the mass-independent vacuum length scale $l = \sqrt{3c^2 / (8\pi G \rho_c)} \approx 1.128$ km, fixed by the QCD core density $\rho_c$. The inner horizon therefore carries a fixed Bekenstein–Hawking area:
$$ S_{\rm in}(M \to \infty) \to \frac{k_B c^3 (4\pi l^2)}{4G\hbar} \approx 2.2 \times 10^{76} \text{ bits} $$
This is a property of the regular core, not a new thermodynamic reservoir: it is a mass-independent constant, negligible next to the outer-horizon entropy $S_{\rm out} \approx S_{\rm Sch} \propto M^2$. By Vieta's formulas on the dimensionless horizon cubic $z^3 - z^2 + \epsilon^2 = 0$ ($e_1 = 1$, $e_2 = 0 \Rightarrow \sum z_i^2 = 1$), the outer+inner horizon area falls short of Schwarzschild by exactly $\Delta A = 4\pi r_{\rm Sch}^2\, z_{\rm phantom}^2$, where $z_{\rm phantom} < 0$ is the third (unphysical, negative) root. This is a compact algebraic expression for the area deficit — not a physical conservation law, and unrelated to the unitarity of Hawking evaporation or the information paradox.

### 2. Theoretical & Methodological Solutions
These points are not direct independent observations, but conceptually solve long-standing astrophysical enigmas.

| Area | NVG Interpretation | Impact on Physics |
|---|---|---|
| Origin of Magnetars | Reconstructed mass-field correlation ($R \approx 0.51$) via core field amplification up to $\sim 7.4\times$ (topological vortex-coupling / Josephson phase-locking). | Solves the paradox of strong fields in slowly rotating magnetars ($E_{\rm rot} \sim 10^{52}$ erg SNR energy tension). |
| PBH Mass Spectrum | A single ladder maps bounce-mass growth per cycle from $10^{-14} M_\odot$ to $10^6 M_\odot$; under the corrected Tolman law the rung spacing is $\times 2$ (denser than the earlier $4^N$; [nvg_tolman_law_derivation.py](verification/nvg_tolman_law_derivation.py)). | Bridges the asteroid-window dark matter with early JWST supermassive BHs, given an abundance model for the heavy rungs. |
| JWST SMBH Seeding | A conditional PBH mass-rung growth template can be evaluated for a selected cycle. | Object-level JWST seeding is retired: no machine-readable catalogue, occupation model, or selection likelihood is present. |
| Joint Multi-Messenger Inference | Conditional/in-sample reduced $\chi^2_\nu = 0.684$ over three runtime structural rows; cooling is a calibration target and excluded. | The transition was selected using J0740, NICER, and GW170817 inputs, so this is descriptive and carries zero independent evidence weight. |
| Emergent Quantization & Duality | Wave-particle duality mapped via Madelung quantum potential $Q(x)$ from vacuum density $\mathcal{W}$ and Goldstone phase $\theta$. | Derives the Schrödinger equation from classical vacuum fluid dynamics, bypassing Derrick's theorem via dynamic wave resonances (PR Research 2026). |
| Observer Effect | Wave function as physical field; collapse as deterministic topological vortex reconnection. | Eliminates Copenhagen idealism, restoring local determinism via classical Madelung vacuum. |

### 3. Consistency Checks (Null Tests)
NVG must not break General Relativity where it works reliably. These items summarize formal/process checks and declared comparisons; they do not prove empirical agreement or replace a detector likelihood.

| Physical Aspect | NVG Prediction | Observational Data |
|---|---|---|
| EOS Causality | $c_s^2 \leq 0.33$ | LIGO/NICER limits: $c_s^2 < 1$ |
| Gravitational Waves | $\gamma_{\rm PPN} \equiv 1$, $c_T = c$ | Cassini, GW170817: $|c_T/c - 1| < 10^{-15}$ |
| External BH Metric | Strict Kerr/Schwarzschild outside horizon | LIGO O4a: 42 mergers, no macro-deviations |
| Tidal Deformability | Runtime canonical $\Lambda_{1.4} = 519.4$ (conditional/in-sample) | GW170817 declared input: $\Lambda_{1.4} = 190^{+390}_{-120}$; no independent confirmation is claimed |
| Dark Energy (DESI) | **Retired mechanism.** The mass-melting derivation of $(w_0, w_a)$ was today-anchored; in the CMB-anchored frame it improves on $\Lambda$CDM by only $\Delta\chi^2 \approx 1$ against DESI DR2 while raising $S_8$ to $\approx 0.86$–$0.90$ and $\Omega_m$ to $0.35$ — no parameter region satisfies DESI and weak lensing together ([nvg_desi_s8_joint_map.py](verification/nvg_desi_s8_joint_map.py)). NVG currently predicts $w = -1$; the DESI $w_0 w_a$ preference, if confirmed, is unexplained by the model — an open problem |
| BH Shadows (EHT) | Deviation from Kerr $\sim 10^{-70}$ | EHT (M87*, Sgr A*) sees no deviation from GR |
| Lorentz Invariance | $0.0$ vacuum dispersion and birefringence | GRB 041219A / 090510 (Fermi/Swift) |
| QNM Ringdown | Ringdown frequency shift $\sim 10^{-105}$ (Hayward core) | LIGO O4a: ringdown is mathematically indistinguishable from Kerr |
| CMB $P(k)$ Spectrum | Perfect match with $\Lambda$CDM for $\ell > 10$ | Planck PR4: exact match at high multipoles |
| BBN and Recombination | $\delta H/H \sim 10^{-13}$, $\delta r_s/r_s \approx 0$ | Preserves nucleosynthesis and $r_s = 147.09$ Mpc |

### 4. Falsifiable Forecasts (Awaiting Verification)
The highest-risk testable forecasts of the theory. Future measurements can constrain or falsify these channels; none is a current confirmation.

| Direction | Forecasted Value / Interpretation | Experiment / Current Status |
|---|---|---|
| **CMB Anomaly $\ell < 10$** | Genesis physical cutoff, NOT cosmic variance | Planck PR4 sees lack of power. Awaiting LiteBIRD. |
| **In-medium $\rho$ line shape** | Preregistered architecture selector: fitted in-medium pole $\geq 755$ MeV $\Rightarrow$ arch. A; $715$–$755$ $\Rightarrow$ arch. B; $< 715$ $\Rightarrow$ both excluded ([HADES_PREREGISTRATION.md](verification/HADES_PREREGISTRATION.md)) | HADES in-medium analysis announced for 2026–27 |
| **Gravitational Echo** | Echo spacing $\Delta t \approx 0.022$ s ($65\,M_\odot$) with decay amplitude $A_n \propto (1 - \mathcal{T})^n$ | Searched O1–O4b open data (coherent time-slide, 89 events): **no evidence**, upper limit set (echoes $\gtrsim 0.3\times$ the GW150914 signal excluded). A naive O4 stack first showed a spurious $2.4\sigma$ excess — resolved as primary-signal (ringdown) leakage via a gap-tooth discriminant. |
| **NS Gravitational Redshift** | $z_{\rm surf}(1.4 M_\odot) \approx 0.222$ (runtime $R_{1.4}=12.550$ km; conditional chain) | STROBE-X / eXTP (future X-ray observatories) |
| **Post-merger $f_{\rm peak}$** | $f_{\rm peak} \approx 2420$ Hz from runtime $R_{1.6}\approx12.65$ km | LIGO O5 / Einstein Telescope (future detectors) |
| **Vacuum melting exponent $\beta$** | $W\sim(1-\rho/\rho_c)^{\beta}$: the $\sqrt{\;}$-law is mean-field $\beta=1/2$, but a QCD-anchored 3-D critical point gives $\beta=0.326$ (Ising) or $0.349$ (XY) — reshaping the bounce term to $(1-\rho/\rho_c)^{2\beta}$ | RHIC **BES-II** net-proton cumulant scaling near $T_c\approx157$ MeV — *existing data*. Derivation & consequences: [`NVG_MELTING_LAW_ANALYSIS.md`](NVG_MELTING_LAW_ANALYSIS.md) |

### External Verification Outreach

A formal letter has been sent to the **HADES Collaboration** (GSI/FAIR, Prof. Dr. J. Stroth) requesting comparison of the VMF ρ-meson prediction against their existing Au+Au and Ag+Ag dielectron data. Clarification established by the audit: $M_\rho^* \approx 621$ MeV at $2n_0$ is the *instantaneous* in-medium mass; the *observable* fireball-integrated dielectron peak — the quantity HADES actually measures — is predicted at $\approx 712$ MeV. The prediction is directly falsifiable by their published invariant-mass spectra.

### 5. Quantitative Comparisons and Evidence Boundaries
Maintained runtime entry points expose conditional comparisons, calibrations, forward models, and retired claims against declared data inputs; a process run is not observational verification:
- **Hubble Constant ($H_0$):** Cycle-77 turnaround horizons bound $H_0 \in [54.3, 108.5]$ km/s/Mpc with a mid-cycle value $\approx 76.8$ (within 5% of SH0ES); the point value $72.8$ is set by the local measurement, and a quantitative CMB re-fit shows the Genesis IR cutoff does not shift the CMB-inferred $H_0$ — the tension is not resolved by this mechanism (`verification/nvg_cmb_lowl_refit.py`).
- **Weak Lensing $S_8$:** Open problem. The maintained calculation gives $S_8 \approx 0.851$ ($4.4\sigma$ from lensing): the input-dependent $w_0/w_a$ shift slightly increases growth, and the former empirical drag-factor resolution route is retired (`verification/nvg_s8_tension_check.py`).
- **CMB Low-$\ell$ Suppression:** The maintained calculation reports a descriptive lite-likelihood comparison over vendored, unknown-provenance Planck-shaped arrays (not the Planck likelihood): $\ell_c = 3.42$ and default $\chi^2$/gain values are model bookkeeping, not a Planck p-value or independent confirmation. The cutoff *shape* comparison remains descriptive (`verification/nvg_cmb_lowl_refit.py`, `verification/nvg_cutoff_shape_derivation.py`).
- **DESI 2024 Dark Energy $w(z)$:** Retired claim. The mass-melting $(w_0, w_a)$ derivation was frame-dependent: CMB-anchored, it yields effective $(-0.813, -0.909)$ — a marginal $\Delta\chi^2 \approx 1$ over $\Lambda$CDM — while pushing $S_8$ to $\approx 0.86$–$0.90$; the joint map excludes the whole $(\beta, a_{\rm on})$ grid (`verification/nvg_desi_s8_joint_map.py`). NVG's dark sector currently predicts $w = -1$, and the DESI dynamical-DE hint stands as an open challenge to the model.
- **GW170817 Tidal Deformability:** The runtime canonical EOS yields $\Lambda_{1.4}=519.4$ (and a conditional binary calculation), compared with the declared GW170817 interval $[70,720]$. The transition was selected on this constraint, so no independent confirmation is claimed.
- **Young Neutron Star Cooling:** The maintained route exposes a calibrated Direct-Urca threshold, but no independent cooling-population likelihood is implemented; the old object-level match is not evidence.
- **JWST Early SMBH Seeding:** A mass-rung growth template remains available, but object-level JWST seeding is retired pending a machine-readable catalogue, occupation model, and selection likelihood.
- **Pulsar Population Dichotomy:** The former ATNF/mock-population claim is retired; no mass/age/luminosity likelihood is present (`verification/nvg_pulsar_population_test.py`).
- **GW Echo Matched Filtering:** The maintained route computes a real-GWTC mass-only forward delay; no strain likelihood or matched-filter detection claim is made (`verification/nvg_new_directions_verification.py`).
- **LiteBIRD B-mode Polarization:** Predicts tensor-to-scalar ratio $r < 0.001$ at CMB scales ($\ell < 10$) due to the Genesis cutoff, serving as a template check for the 2032 LiteBIRD mission (`verification/nvg_litebird_prediction.py`).
- **NICER PSR J0437-4715 Radius:** The runtime canonical radius is $R\approx12.55$ km, a $+1.5\sigma$ conditional pull against the 2024 NICER measurement ($11.36 \pm 0.8$ km at $1.418 M_\odot$). The transition was selected in-sample; this is not an independent confirmation (`verification/nvg_nicer_j0437_check.py`).
- **NANOGrav 15yr SGWB:** Retired claim. The two-population analysis (`verification/nvg_pbh_two_population.py`) shows the JWST-calibrated heavy-PBH population falls $\sim 1{,}700\times$ short of the NANOGrav strain, and the bounce radiates at $18$–$42\,\mu$Hz — NVG offers no mechanism for the PTA signal.
- **Hubble Constant (interval form):** The cycle-quantized horizon chain ($n=77 \to N_e \in [52.68, 53.38] \to H_0$) gives the interval and mid-cycle values above; note that $N_e = 53.08$, the CMB low-$\ell$ cutoff scale and the point $H_0$ all trace to the SAME single calibration against the local $H_0$ — one fitted parameter, not three independent confirmations (`verification/nvg_hubble_tension.py`).
- **SGR 1935+2154 FRBs:** Computes a mass-to-rate sensitivity scaling; no burst-rate catalogue or population likelihood is implemented, so the $>3\times$ contrast is not an observed confirmation (`verification/nvg_sgr_frb_rate.py`).
- **S8 (duplicate of the bullet above):** see the honest accounting — computed $S_8 \approx 0.851$, $4.4\sigma$ from lensing, open problem (`verification/nvg_s8_tension_check.py`).
- **CHIME Repeating FRBs:** Retired pending a repeater-linked DM/magnetar-mass catalogue and defensible population likelihood; no synthetic p-value or confirmation is reported (`verification/nvg_chime_frb_check.py`).
- **LIGO O4 Echo Candidates:** Predicts echo delays in the $0.021 - 0.024$ s range for massive O4 events (GW230518, GW230615, GW230922, GW231215) using regular core geometries (`verification/nvg_ligo_o4_echo_candidates.py`).
- **Advanced Calculations:** Emits a runtime evidence-status ledger for formal/forward rows and explicitly retired JWST, FRB, PTA, and neutrino claims; a process run is not observational verification (`verification/nvg_advanced_calculations.py`).
- **Vacuum Melting Law $W(\rho)$ — Identifiability & Critical Exponent:** A focused, reproducible study of what the melting law is actually measured by, using the framework's own forward models ([**NVG_MELTING_LAW_ANALYSIS.md**](NVG_MELTING_LAW_ANALYSIS.md)). The deep-core identifiability study uses an alternate dated calibration and is descriptive only (zero independent evidence weight); it must not be confused with the runtime canonical chain. The cosmological $\sqrt{1-\rho/\rho_c}$ law is inert at NS density, while a QCD-anchored 3-D critical point gives $\beta\approx0.326$ (`verification/nvg_melting_freeform_beta.py`, `verification/nvg_melting_identifiability.py`, `verification/nvg_melting_exponent.py`).

---

### 6. Additional Calculations (August 2026)

A maintained set of scripts in `verification/` computes formal identities, conditional model outputs, and forward forecasts. Each entry point exposes its evidence boundary; a successful process run is not an observational confirmation.

#### Evidential-strength hierarchy (audit without stretching)

Per the contract [NVG_RESEARCH_CONTRACT.md](NVG_RESEARCH_CONTRACT.md) (`SPEC-v2`), compatibility with data is never labelled “proof”; the tiers below are the maximum an honest audit may claim:

| Tier | Content | Basis |
|------|---------|-------|
| **I. Exact identities (mathematically ironclad)** | $\rho_c r_c^2 = 3c^2/8\pi G$ (precision $2\times10^{-16}$), $M_1 = c^3 t_b/(2G)$, $T_b/M_\Omega = (30/\pi^2 g_*)^{1/4}$, $M_{\rm crit} = (9/8\sqrt{2\pi})M_{\rm Pl}^3/M_\Omega^2$ — all from the SINGLE input $M_\Omega = 859\pm8$ MeV, zero free parameters (`nvg_anchor_identities.py`, PASS) | algebra + dimensions |
| **II. Overdetermined predictions (one parameter → two or more independent outputs)** | $\theta$-seesaw: a single $f_a = 1.07\times10^{11}$ GeV yields simultaneously $m_\theta = 53.2$ μeV, $m_{\nu_3} \approx 50$ meV and $\sum m_\nu = 63.3$ meV with no right-handed neutrinos ($m_{\beta\beta} > 10$ meV excludes the sector) | action-derived + DESI/KATRIN data |
| **III. Conditional compatibilities** | $2M_\Omega$ and the scalar glueball (lattice QCD), $M_{\rm crit} = 0.992\,M_\odot$ vs the absence of sub-solar horizons in GWTC, Cas A/Vela bimodality, and the runtime canonical $\Lambda_{1.4}=519.4$ comparison | declared data inputs; NS chain selected in-sample, with zero independent evidence weight (`SPEC-v2` C3) |
| **IV. Calibrations (not evidence)** | $H_0 = 72.8$, Direct Urca threshold $1.45\,M_\odot$, EOS shape parameters | one fitted input per block |
| **V. Conjectures (CONJECTURED)** | the $m_a \leftrightarrow \rho_{\rm DE}$ link via $N_e$ (see the new table row) | ansatz + shared $H_0$ calibration (reverse-solution structure, `SPEC-v2` rejection boundary item 2) |

The ironclad core of the framework is tiers I–II; tiers III–V are honestly labelled and never pooled into a single evidence count. Note: the triple-hierarchy identities (`nvg_dual_hierarchy_identity.py`) are tier-I algebra with a calibrated $H_0$ input — a consistency closure, not evidence.


| Script | Result | Status |
|--------|--------|--------|
| `nvg_anchor_identities.py` | 6 closed identities from the single input $M_\Omega$: $\rho_c r_c^2 = 3c^2/8\pi G$ (precision $2\times10^{-16}$), $M_1 = c^3 t_b/(2G) = 0.382\,M_\odot$, $T_b = 432.0$ MeV, $M_{\rm crit} = 0.992\,M_\odot$, bare Planck scale $M_{\rm Pl}^3/M_\Omega^2 = 2.21\,M_\odot$ | PASS + $S_{\rm GH}$ audit |
| `nvg_cmb_birefringence.py` | Cosmic birefringence from $\theta F\tilde F$: $\beta = 0.033°$ per unit $\gamma_{\rm topo}\Delta\theta$; static branch $\beta=0$ compatible with Planck/ACT; rolling branch bounded at $\|\gamma_{\rm topo}\Delta\theta\| < 7.2$ | forward null test (LiteBIRD $\sigma\sim0.03°$) |
| `nvg_moment_of_inertia_j0737.py` | Hartle slow-rotation exploratory calculation; its alternate EOS outputs are dated calibration diagnostics, not the maintained canonical chain | zero independent evidence weight; future J0737A measurement remains a forecast |
| `nvg_theta_fifth_force.py` | $\lambda_\theta = 3.72$ mm; the pseudoscalar monopole channel is identically zero (null prediction for Eöt-Wash); the spin-dipole channel places $f_a = 1.07\times10^{11}$ GeV at the edge of current bounds | tension diagnostic |
| `nvg_neutrinoless_dbeta.py` | $\theta$-seesaw: $\sum m_\nu = 63.3$ meV < 72 meV (DESI DR2); $m_\beta = 9.5$ meV; **$m_{\beta\beta} \in [0, 6.4]$ meV** — below LEGEND-1000/nEXO reach | forward null test |
| `nvg_spin_limits_rmodes.py` | Exploratory fork-B Kepler/r-mode limits (dated alternate calibration); these outputs are not the maintained canonical NS chain | descriptive diagnostic; zero independent evidence weight |
| `nvg_theta_superradiance.py` | $\theta$-mode superradiance band: $M_{\rm BH} \in [5.0\times10^{-7}, 1.3\times10^{-6}]\,M_\odot$ (~0.3 Earth masses); closest $4^N$ ladder rung: $N=9$ (1.41× band center); horizonless remnants: channel closed | forward null test |
| `nvg_tidal_heating_null.py` | Tidal heating: GW phase deficit ~32 cycles (EMRI, LISA) for a Kerr BH vs 0 for a horizonless remnant | forward null test |
| `nvg_de_axion_crosscheck.py` | Cross-check against the dark-dimension window (0.2–100 μm): $\Lambda_{\rm DE}^{1/4} = 2.28$ meV, window KK masses 2 meV–1 eV; NVG pseudoscalar Compton radii ($R_a = 23.5$ mm, $R_\theta = 3.71$ mm) miss the window by ×235–×3710 → geometric identification REFUTED; internal ansatz sensitivity $d\ln m_a/d\ln\rho_{\rm DE} = -2/N_e = -3.77\%$ with an inverse anchor ($m_a$ to 1% → $H_0$ to ±0.18 km/s/Mpc); NVG haloscope band (μeV) and the KK tower band (meV–eV) are disjoint | CONJECTURED diagnostic (`SPEC-v2`) |
| `nvg_dual_hierarchy_identity.py` | Quad-hierarchy identity: $\rho_c r_c^2 = \rho_{\rm crit}R_{H0}^2 = 3c^2/8\pi G$ closes density ($\rho_c/\rho_{\rm DE} = e^{2N_e}/\Omega_{\rm DE} = 1.85\times10^{46}$), entropy ($S_{\rm now}/S_{\rm gen} = 4^n$), mass ($M_U = M_1 2^n = 4.30\times10^{22}\,M_\odot$) and time ($t_{H0}/t_b = e^{N_e}$, precision $2\times10^{-15}$) with the single number $e^{N_e} = R_{H0}/r_c$ at $n = N_e/\ln2 = 76.58$; audit of item 15: "77.2" = $\log_4(2.6\times10^{122}/10^{76}) = 77.09$ from rounded anchors, "±0.3" = $\ln2/2$ (cycle-interval half-width; true QCD sensitivity ±0.013) | closed identity (calibrated $H_0$) |
| `nvg_ns_mass_bound.py` | Parameter-free dimensional mass scale $M_{\rm Pl}^3/M_\Omega^2$ and its live-data comparison; this bound is separate from the runtime canonical TOV chain and carries its own declared assumptions | registered bound diagnostic; do not combine with fork-B or count as independent NS evidence |
| `nvg_sigma_mnu_corner.py` | The cornered $\theta$-seesaw prediction $\Sigma m_\nu = 59.0$ meV (minimum of normal ordering, $m_1\approx0$) vs live bounds: Planck UL 120 (margin +61), DESI DR1 UL 72 (**margin only +13 meV**), disputed DESI DR2 $-75\pm51$ (pull 2.6$\sigma$); CMB-S4 forecast $\sigma\sim15$ meV — a $\sim4\sigma$ exclusion or a $\sim2\sigma$ confirmation; falsifier: a robust $\Sigma<45$ meV | live monitor (cornered) |
| `nvg_katrin_mbeta.py` | Kinematic mass $m_\beta$: the PMNS-weighted observable $= 8.9$ meV (minimal NO) — correcting the earlier 29 meV (equi-weighting $m_3/\sqrt3$ is NOT the KATRIN observable); KATRIN's 200 meV design reach is a factor $\sim$20 away — honestly INSENSITIVE; falsifier: a robust $m_\beta > 20$ meV excludes the sector (IO floor 49.2 meV); two-sided test only with Project 8 ultimate ($\sigma\sim10$ meV) | registered one-sided falsifier |
| `nvg_echo_o4_observers.py` | Observer card for the four O4 echo candidates (GW230518/230615/230922/231215): echo spacing $20.0$–$23.9$ ms (remnant-mass systematic $\times0.95$ included), carrier $f_{\rm QNM} = 245$–$278$ Hz, damping $\tau = 3.3$–$3.7$ ms, 14–16 pulses per comb; search protocol consistent with the repo pipeline | observer card |

Key new falsifiable content: (i) **moment of inertia of J0737A** — orbital precession of the double pulsar will measure $I$ at ~10%, discriminating stiff from soft EOS; (ii) **$m_{\beta\beta} > 10$ meV excludes the minimal $\theta$-seesaw**; (iii) **any monopole fifth-force signal at 3–4 mm excludes the $\theta$ sector**; (iv) a confirmed cosmic birefringence $\beta \neq 0$ at LiteBIRD precision pins $\gamma_{\rm topo}\Delta\theta$; (v) superradiance and tidal heating provide population tests of horizonless remnants; (vi) **a detection in the μeV band vs the meV–eV band** discriminates NVG from dark-dimension KK dark matter, and DESI $w(z) \neq -1$ at growing significance is the sharpest live falsifier of the dark sector; (vii) KATRIN/Project 8: $m_\beta = 8.9$ meV (PMNS-weighted; the earlier 29 meV corrected — equi-weighting $m_3/\sqrt3$ is not the KATRIN observable) — KATRIN is honestly insensitive (factor $\sim$20), but any robust $m_\beta > 20$ meV excludes the $\theta$-seesaw (IO floor 49.2 meV), and the two-sided test belongs to Project 8 ultimate ($\sigma\sim10$ meV); (viii) audit of the PBH ladder $M_N = 0.38\times4^N\,M_\odot$: the shared ×4 spacing is structural (horizon masses), but the peak rung $N=-21$ is calibrated to the asteroid window (formation epoch $t = 8.5\times10^{-19}$ s, $T\approx642$ TeV matches no natural scale); the closure hypothesis $\sqrt{m_{\rm Pl}M_1}$ is REFUTED (×1.3×10⁶); the peak is Hawking-stable with a $\sim10^{18}$ lifetime margin, evaporation boundary at $N=-31$.

---

## Analog Optical Verification

Predicted NVG/VMF functional dependencies were encoded as optical signals and measured through an analog integrating channel (γ=1.56, DR=86:1, SNR=38).

| Test | NVG Prediction | Optical Result | Correlation |
|------|---------------|----------------|-------------|
| Meson hierarchy $\rho > K^* > \phi > J/\psi$ | $-20.0\%,\;-7.8\%,\;-2.9\%,\;-0.4\%$ | $-20.0\%,\;-8.8\%,\;-2.3\%,\;0.0\%$ | $r = 0.997$ |
| Melting curve $W(\rho)=\sqrt{1-\rho/\rho_c}$ | $\sqrt{1-x}$ vs linear | $\sqrt{1-x}$: $r=0.983$; linear: $r=0.896$ | $r = 0.983$ |
| Modified Friedmann $H^2 \propto \rho(1-\rho/\rho_c)$ | Parabola, zeros at $0$ and $\rho_c$ | Max at center, both zeros confirmed | $r = 0.983$ |
| Tolman growth law | $S_{\rm GH} \times 4$ per cycle **derived** from turnaround dynamics ($a_t \times 2 \Leftrightarrow S_{\rm GH} \times 4$); the mass law is corrected to $M \times 2$ per cycle — the old $M \times 4$ would give $S_{\rm GH} \times 16$ and is excluded ([nvg_tolman_law_derivation.py](verification/nvg_tolman_law_derivation.py)) | derived |

All physical scales derive from the QCD anchor $M_\Omega = 859$ MeV combined with nuclear-calibrated EOS shape parameters. The claims table distinguishes computed forward predictions, compatibility checks, calibrations and null tests; every number traces to a script in `verification/`, and each falsifiable row states what future measurement would exclude the model. The optical channel discriminates $\sqrt{1-x}$ from a linear model ($\Delta r = 0.087$), confirming internal consistency of the mathematical structure.

**Ruled-out hypotheses (checked and closed):** (i) a "golden angle" / $\theta \approx 52.8°$ intra-cycle phase — the angle is convention-dependent (49.5°–54.4°) and was never derived; a golden-ratio (irrational-winding) cycle structure is incompatible with the integer topological charge $Q = 1$ underlying the arrow-of-time theorem; (ii) the NANOGrav signal as the NVG bounce — excluded by the $(\alpha, \beta/H)$ derived from the action (the bounce radiates at $\mu$Hz); (iii) the de Sitter core mechanism for $S_8$ — capacity short by $\sim 45$ orders of magnitude.

**Scope:** analog verification confirms the mathematical structure, not the physics. Experimental confirmation requires HADES/NICER/LIGO/RHIC data.

---

## Repository Structure

```
NVG-Research/
├── article/
│   ├── NVG_SCIENTIFIC_ARTICLE_EN.md        # Pillar I: Dense Nuclear Matter (VMF)
│   ├── NVG_SCIENTIFIC_ARTICLE_RU.md        # Russian version of Pillar I
│   ├── NVG_CYCLIC_COSMOLOGY_PREPRINT_EN.md # Pillar II: NVG Cyclic Cosmology
│   ├── NVG_CYCLIC_COSMOLOGY_PREPRINT_RU.md # Russian version
│   ├── NVG_GENESIS_MODEL_EN.md             # Pillar II: The First Cycle
│   ├── NVG_GENESIS_MODEL_RU.md             # Russian version
│   ├── NVG_MAGNETAR_PREPRINT_V3.md         # Revised magnetar preprint with new quantitative closure tests
│   ├── NVG_MAGNETAR_PREPRINT_V3.tex        # Publication LaTeX for the revised magnetar preprint
│   ├── NVG_MAGNETAR_PREPRINT_V3.pdf        # Publication PDF for the revised magnetar preprint
│   ├── NVG_MAGNETAR_PREPRINT_V4.md         # Version 4 preprint with mass correlation audit & predictions
│   ├── NVG_MAGNETAR_PREPRINT_V4.tex        # LaTeX file for Version 4 preprint
│   ├── NVG_MAGNETAR_PREPRINT_V4.pdf        # PDF for Version 4 preprint
│   ├── NVG_MAGNETAR_POPULATION_APPENDIX.md # Source-by-source magnetar population appendix
│   ├── NVG_UNIFIED_FIELD_EQUATIONS.md      # Mathematical derivation of the unified field action and equations
│   ├── NVG_UNIFIED_FIELD_EQUATIONS.tex      # LaTeX file for the unified field equations
│   ├── NVG_UNIFIED_FIELD_EQUATIONS.pdf      # PDF for the unified field equations
│   ├── NVG_VACUUM_W_FIELD_DERIVATION_EN.md  # QFT derivation of the vacuum condensate amplitude W (EN)
│   ├── NVG_VACUUM_W_FIELD_DERIVATION_RU.md  # QFT derivation of the vacuum condensate amplitude W (RU)
│   └── *.pdf                               # PDF renders of all articles
├── verification/
│   ├── nvg_verification_suite.py           # Master automated verification test suite
│   ├── nvg_advanced_calculations.py        # Runtime evidence-status ledger; retired/forward rows remain non-evidence
│   ├── nvg_full_ns_eos.py                  # NS EOS + TOV solver → M_max, R_1.4
│   ├── nvg_hyperon_puzzle_solution.py      # Hyperon threshold proxy (no complete-likelihood resolution)
│   ├── nvg_hyperon_puzzle_tov.py           # TOV solver for Hyperon Puzzle (NL3/SLy baselines & figures)
│   ├── nvg_hadrons_magnetic_fields.py      # Meson mass shifts, magnetic fields
│   ├── nvg_weak_field_ppn.py               # PPN parameter verification (γ=1)
│   ├── nvg_cosmology_tensions.py           # Hubble/S8 tensions, BBN constraints
│   ├── nvg_cooling_dark_matter.py          # PBH Dark Matter, NS Cooling (Direct Urca)
│   ├── nvg_black_hole_entropy.py           # BH core regularity, Tolman entropy balance
│   ├── nvg_cmb_smbh_cyclic.py              # CMB anomalies, cyclic parameters, early SMBHs
│   ├── nvg_iloveq_gw_echoes.py             # I-Love-Q transform, GW Echo templates
│   ├── nvg_bbn_reionization.py             # BBN and reionization checks
│   ├── nvg_gravitational_waves_tests.py    # Additional GW constraint checks
│   ├── nvg_advanced_observables_I.py       # Runtime ledger: HADES forward, canonical NS chain, cycles
│   ├── nvg_advanced_observables_II.py      # Runtime forward ledger: CMB/EHT/PBH (no likelihood)
│   ├── nvg_advanced_observables_III.py     # Runtime ledger: mesons, Lorentz, cooling, QNM
│   ├── nvg_em_maxwell_decoherence.py       # Maxwell equations (eps_eff) & Decoherence
│   ├── nvg_grmhd_surrogate.py              # EOB surrogate BNS merger (GW Strain)
│   ├── nvg_detector_forward_model.py       # HADES/CBM/NICA Forward Model
│   ├── nvg_pulsar_population_test.py       # Retired: no ATNF mass/age/luminosity likelihood
│   ├── nvg_magnetar_closure.py             # Magnetar closure checks and structural-amplification benchmarks
│   ├── nvg_1e161348_fallback_torque.py     # Fallback-disk torque model for 1E 161348-5055
│   ├── nvg_magnetar_population_scan.py     # Magnetar catalog scan, gamma-fit, and appendix export
│   ├── nvg_magnetar_mass_correlation.py    # Reconstructed magnetar mass-field correlation statistical audit
│   ├── nvg_new_predictions.py              # Quantitative multi-messenger predictions (FAIR, GW, LMXB)
│   ├── nvg_unified_field_equations.py      # Verification of the unified field equations and core limits
│   ├── nvg_hades_dielectron_sim.py         # HADES/CBM in-medium rho-meson dielectron spectral simulation
│   ├── nvg_gw_echo_waveforms.py            # Post-merger black hole GW echoes waveform template simulator
│   ├── nvg_dark_energy_w0wa.py             # CPL w0-wa parameter derivation from VMF cyclic cosmology
│   ├── nvg_dark_energy_desi.py             # Cosmological dark energy w0-wa parameter alignment with DESI DR2
│   ├── nvg_pbh_jwst_seeds.py               # JWST early supermassive black hole seeding puzzle simulation
│   ├── nvg_pbh_continuity_test.py          # Continuous PBH mass spectrum
│   ├── nvg_joint_ns_inference.py           # Joint NS Inference (Multi-Messenger Likelihood)
│   ├── nvg_observational_data_fit.py       # Quantitative fits to Planck, DESI, GW170817, and cooling data
│   ├── nvg_new_directions_verification.py  # Status ledger; JWST/ATNF retired, GW echo is mass-only forward
│   ├── nvg_litebird_prediction.py          # B-mode polarization tensor cutoff predictions (LiteBIRD 2032)
│   ├── nvg_nicer_j0437_check.py            # Mass-radius check against NICER 2024 PSR J0437-4715 bounds
│   ├── nvg_nanograv_background.py          # Retired PBH–NANOGrav channel abundance cross-check
│   ├── nvg_hubble_tension.py               # Hubble tension analysis and Genesis bounds
│   ├── nvg_sgr_frb_rate.py                 # Magnetar mass-stability relation and FRB rate for SGR 1935+2154
│   ├── nvg_s8_tension_check.py             # Maintained S8 tension accounting (input-dependent route; currently worsens)
│   ├── nvg_chime_frb_check.py              # Retired: no repeater-linked CHIME DM/mass likelihood
│   ├── nvg_ligo_o4_echo_candidates.py      # Predicted echo time delays for massive LIGO O4 remnants (M ~ 65 M_sun)
│   ├── nvg_relic_dark_matter.py            # Relic instanton dark matter density and coupling inference
│   ├── nvg_glueball_mass.py                # Scalar glueball mass calculation
│   ├── nvg_neutrino_mass.py                # Majorana neutrino mass see-saw limit
│   ├── nvg_starquake_qpo.py                # Magnetar starquake QPO shear frequencies
│   ├── nvg_primordial_gw_comb.py           # Primordial gravitational wave frequency comb generator
│   ├── nvg_axion_mass.py                   # Topological axion decay constant and mass calculation
│   ├── nvg_de_axion_crosscheck.py          # Dark-energy/axion cross-check vs the dark-dimension window (CONJECTURED)
│   ├── nvg_dual_hierarchy_identity.py      # Triple-hierarchy identity (density/entropy/mass from 3c²/8πG)
│   ├── nvg_ns_mass_bound.py                # Parameter-free NS mass bound M_Pl³/M_Ω² = 2.21 M_⊙ + live cut
│   ├── nvg_sigma_mnu_corner.py             # Cornered sum Σm_ν = 59 meV vs DESI/CMB-S4
│   ├── nvg_katrin_mbeta.py                 # m_β = 8.9 meV vs KATRIN/Project 8 (29 meV corrected)
│   ├── nvg_echo_o4_observers.py            # Observer card: echo parameters for 4 O4 candidates
│   ├── nvg_perihelion_shift.py             # Binary pulsar strong-field periastron shift calculation
│   ├── nvg_cmb_temperature.py              # CMB temperature today from QCD bounce scale
│   ├── nvg_baryon_asymmetry.py            # Baryon asymmetry (eta_B) from Genesis bounce
│   ├── nvg_postmerger_fpeak.py            # Post-merger peak GW frequency from VMF TOV R_1.6
│   ├── nvg_ns_redshift.py                 # Surface gravitational redshift z_surf from VMF R_1.4
│   ├── nvg_sgr_temperature.py             # SGR 1935+2154 quiescent spot temperature
│   ├── nvg_speed_of_sound_curve.py        # Speed of sound c_s^2(n_B) profile and conformal bound
│   ├── nvg_ns_g_modes.py                  # WKB g-mode forecast with assumed composition (no detector likelihood)
│   ├── nvg_higgs_mass_shift.py            # Higgs boson mass shift from QCD vacuum condensate
│   ├── nvg_dna_chirality.py               # DNA homochirality and biological θ-coherence scales
│   ├── nvg_ds_core_oscillations.py        # de Sitter core standing wave oscillations
│   ├── nvg_pbh_dark_matter.py             # PBH DM fraction Subaru/LIGO check
│   ├── nvg_wd_cooling.py                  # White Dwarf cooling rate under VMF
│   ├── run_nvg_suite.py                    # MASTER SCRIPT: generates final uncertainty report
│   ├── run_all_checks.py                   # Automated suite runner for all physical verifications
│   ├── nvg_genesis_observable.py           # Genesis instanton → Hubble horizon match
│   ├── nvg_vacuum_w_field_derivation.py    # Numerical verification of the W-field phase transition
│   ├── nvg_strong_cp_solution.py           # Strong CP problem solution from V(W,θ)
│   ├── nvg_double_slit_madelung.py         # Double-slit interference from W-condensate Madelung hydrodynamics
│   ├── nvg_arrow_of_time.py                # Arrow of time from vacuum phase θ topology
│   ├── nvg_dm_direct_detection.py          # Proof that W ≠ WIMP: null WIMP prediction from 3 coupling channels
│   ├── nvg_bell_inequality.py              # Bell violation from vacuum phase θ coherence
│   ├── nvg_heisenberg_proof.py             # Heisenberg uncertainty = Cauchy-Schwarz theorem
│   ├── nvg_wavefunction_collapse.py        # "Collapse" = thermalization of vacuum phase θ
│   ├── nvg_neutrino_seesaw.py              # Neutrino mass from θ-seesaw without right-handed neutrinos
│   ├── nvg_quantum_gravity.py             # Quantum gravity without quantization: Hawking from θ-thermalization
│   ├── nvg_fine_structure.py              # α_EM = 1/137 from vacuum polarization Z_EM(W₀)
│   ├── nvg_antimatter_topology.py         # Antimatter as θ → −θ, annihilation = vortex reconnection
│   └── nvg_rhic_bell_test.py              # 🔥 RHIC Bell Test: S_CHSH(T > T_c) → 0, experimental protocol
├── visualization/
│   ├── nvg_3d_viz_v2.html                  # Interactive 3D Tolman Cycles Simulator
│   ├── nvg_ns_merger_3d.html               # Interactive 3D BNS Merger & Mass Melting
│   └── nvg_3d_viz_v2_ru.html              # Interactive 3D Universe Simulator (RU)
├── .docs/
│   ├── NVG_VERIFICATION_MATRIX_RU.md       # Matrix of falsifiable predictions
│   ├── NVG_EM_OBSERVABLES.md               # Strict Checklist of EM Observables
│   ├── NVG_ELECTROMAGNETIC_EXTENSIONS.md   # EM waves, wave-particle duality, research directions (RU)
│   └── NVG_ELECTROMAGNETIC_EXTENSIONS_EN.md # English version
├── README.md
└── README_RU.md
```

## Quick Start (Automated In-Silico Suite)

The repository includes a process-check suite and a canonical runtime ledger. These execute declared calculations and expose their evidence boundaries; a green process exit is not an observational confirmation.

```bash
# Install dependencies
pip install numpy scipy

# Run the master verification suites
python verification/nvg_verification_suite.py     # Process checks only (not scientific verification)
python verification/nvg_advanced_calculations.py  # Runtime evidence-status ledger; retired/forward rows remain non-evidence
python verification/run_all_checks.py             # Executes the process harness (not scientific verification)

# Run specific predictive scripts
python verification/nvg_gw_echoes.py               # Predicts LIGO/Virgo GW Echoes
python verification/nvg_cyclic_lifetimes.py        # Calculates Tolman cycle durations
python verification/nvg_hadron_mass_fractions.py   # Shows the 91% nonperturbative QCD mass
python verification/nvg_full_ns_eos.py             # Solves the NS EOS and TOV equations
python verification/nvg_fair_hades_link.py         # Predicts the 20% rho-meson mass drop
python verification/nvg_magnetar_closure.py        # Closure checks for the revised magnetar scenario
python verification/nvg_1e161348_fallback_torque.py # Fallback-disk braking for 1E 161348-5055
python verification/nvg_magnetar_population_scan.py # Catalog scan and appendix export for the magnetar population
python verification/nvg_magnetar_mass_correlation.py # Statistical correlation audit of reconstructed masses
python verification/nvg_new_predictions.py          # Quantitative predictions (FAIR, post-merger GW shift, LMXB)
python verification/nvg_unified_field_equations.py  # Verification of the unified field equations (bounce and magnetars)
python verification/nvg_hades_dielectron_sim.py     # HADES/CBM in-medium rho dielectron spectral simulation
python verification/nvg_gw_echo_waveforms.py        # Post-merger black hole GW echoes waveform template simulator
python verification/nvg_dark_energy_desi.py         # Dark energy w0-wa parameter alignment with DESI DR2
python verification/nvg_pbh_jwst_seeds.py           # JWST early black hole seeding puzzle simulation

# Electromagnetic extensions and vacuum properties
python verification/nvg_em_extensions_proofs.py     # Lorentz invariance, vacuum polarization
python verification/nvg_em_priority2_formal.py     # Maxwell from S[g,W,A], ε_eff, decoherence

# Astrophysical and cosmological observables
python verification/nvg_cosmology_tensions.py      # Hubble/S8 tensions, BBN constraints
python verification/nvg_cooling_dark_matter.py     # PBH Dark Matter, NS Cooling dichotomy
python verification/nvg_iloveq_gw_echoes.py        # I-Love-Q transform, GW echo templates
python verification/nvg_cmb_smbh_cyclic.py         # CMB anomalies, Early SMBHs
python verification/nvg_black_hole_entropy.py      # BH core, entropy reset
python verification/nvg_hyperon_puzzle_solution.py # Hyperon threshold proxy (no complete-likelihood resolution)
python verification/nvg_hyperon_puzzle_tov.py      # TOV solver for Hyperon Puzzle (NL3/SLy baselines & figures)
python verification/nvg_advanced_observables_I.py  # HADES spectrum, z_surf, cycles
python verification/nvg_advanced_observables_II.py # CMB P(k), EHT shadows, PBH mass
python verification/nvg_advanced_observables_III.py# Mesons, Lorentz, NS Cooling
python verification/nvg_em_maxwell_decoherence.py  # Maxwell (eps_eff), Transfer Function
python verification/nvg_grmhd_surrogate.py         # EOB surrogate BNS merger (GW Strain)
python verification/nvg_detector_forward_model.py  # HADES/CBM Forward Model
python verification/nvg_pulsar_population_test.py  # Retired: no ATNF mass/age/luminosity likelihood
python verification/nvg_pbh_continuity_test.py     # PBH continuous mass spectrum
python verification/nvg_joint_ns_inference.py      # Joint NS Inference (Likelihood)
python verification/nvg_observational_data_fit.py   # Fits Planck PR4, DESI DR2, GW170817, and cooling
python verification/nvg_new_directions_verification.py # Status ledger; JWST/ATNF claims retired, GW echo is mass-only forward
python verification/nvg_litebird_prediction.py      # Predicts B-mode polarization tensor cutoff (LiteBIRD 2032)
python verification/nvg_nicer_j0437_check.py        # Validates NVG radius against 2024 NICER PSR J0437-4715 bounds
python verification/nvg_nanograv_background.py      # Retired PBH–NANOGrav channel abundance cross-check
python verification/nvg_hubble_tension.py           # Calculates H_0 bounds from the Tolman cycle
python verification/nvg_sgr_frb_rate.py             # Models magnetar mass vs stability and FRB burst rate
python verification/nvg_dark_energy_w0wa.py         # Derives CPL dark energy parameters w0-wa
python verification/nvg_dark_energy_desi.py         # Verifies dark energy w0-wa alignment vs DESI DR2
python verification/nvg_s8_tension_check.py         # Maintained S8 tension accounting (currently worsens)
python verification/nvg_chime_frb_check.py          # Retired: no repeater-linked CHIME DM/mass likelihood
python verification/nvg_ligo_o4_echo_candidates.py  # Echo delay times for O4 candidates (M ~ 65 M_sun)
python verification/nvg_relic_dark_matter.py        # Relic instanton dark matter abundance and coupling check
python verification/nvg_glueball_mass.py           # Calculates the scalar glueball mass
python verification/nvg_neutrino_mass.py           # Calculates the Majorana neutrino mass limit
python verification/nvg_starquake_qpo.py           # Validates magnetar QPO starquake frequencies
python verification/nvg_primordial_gw_comb.py      # Calculates bounce frequencies for Tolman cycles
python verification/nvg_axion_mass.py              # Calculates topological axion mass limits
python verification/nvg_de_axion_crosscheck.py     # Dark-energy <-> axion-sector cross-check (dark-dimension window)
python verification/nvg_dual_hierarchy_identity.py # Triple-hierarchy identity + cycle-count audit
python verification/nvg_ns_mass_bound.py     # Parameter-free NS mass bound + live data cut
python verification/nvg_sigma_mnu_corner.py  # Σm_ν = 59 meV: cornered θ-seesaw prediction
python verification/nvg_katrin_mbeta.py      # m_β = 8.9 meV vs KATRIN/Project 8 (falsifier m_β > 20 meV)
python verification/nvg_echo_o4_observers.py # Echo observer card for the 4 O4 candidates (Δt = 20-24 ms)
python verification/nvg_perihelion_shift.py        # Verifies binary pulsar strong-field periastron shift
python verification/nvg_vacuum_w_field_derivation.py # Models QFT W-field phase transition & VEV
python verification/nvg_cmb_temperature.py      # Derives CMB temperature $T_{\rm CMB} = 2.725$ K from QCD bounce scale
python verification/nvg_baryon_asymmetry.py     # Computes primordial baryon asymmetry $\eta_B \approx 6 \times 10^{-10}$
python verification/nvg_postmerger_fpeak.py     # Reconstructs post-merger peak gravitational wave frequency
python verification/nvg_ns_redshift.py          # Solves TOV to compute surface gravitational redshift $z_{\rm surf} = 0.235$
python verification/nvg_sgr_temperature.py      # Simulates quiescent thermal cap emission for light magnetars (SGR 1935+2154)
python verification/nvg_ns_g_modes.py                  # WKB g-mode forecast with assumed composition (no detector likelihood)
python verification/nvg_ds_core_oscillations.py        # Computes standing wave oscillations in de Sitter cores
python verification/nvg_pbh_dark_matter.py             # Computes PBH dark matter fraction and limits
python verification/nvg_wd_cooling.py                  # Computes VMF white dwarf cooling rate deviation
python verification/run_nvg_suite.py               # Canonical runtime ledger (NVG_FINAL_REPORT.md)
```

## Key Testable Predictions (Falsifiability)

Unlike abstract quantum gravity models, the NVG/VMF framework is rigidly anchored to the QCD energy scale, making it strictly falsifiable across multiple disciplines:

1. **Gravitational Wave Echoes:** Prediction of a macroscopic finite-time post-merger echo at $\Delta t_{\rm echo} \approx 22$ ms (order of magnitude) spin-corrected for a 65 $M_\odot$ black hole merger — logarithmically sensitive to the near-horizon UV cutoff ($\sim 16$–$34$ ms); the parameter-free, robust content is the echo's existence and the QCD-fixed core scale (LIGO/Virgo testable).
2. **Heavy-Ion Collisions (FAIR/HADES/NICA):** A ~20% drop in the invariant mass of the $\rho$-meson at $2n_0$. If no in-medium hadron mass shifts are observed at $n_B \sim 3$–$5\,n_0$, the VMF mass melting chain is falsified.
3. **CMB Genesis Cutoff:** The low-$\ell$ suppression ($\ell=2,3$) is a deterministic physical cutoff from the $1.13$ km Genesis instanton stretched by $\sim 53$ e-folds, not merely "cosmic variance".
4. **Neutron Stars:** The runtime canonical chain reaches $M_{\max}=2.048 M_\odot$ with a CSS branch at $c_s^2=1/3$; the transition is selected in-sample and is not independent evidence.
5. **Lattice QCD Anchor:** Future lattice calculations shifting $M_{\Omega,0}$ outside $851$–$867$ MeV will explicitly shift all bounce parameters.
6. **EHT Null Test (Black Hole Shadows):** VMF predicts an absolute match with the Schwarzschild/Kerr exterior. The event horizon deviation is $\sim 10^{-35}$, and the photon ring ($r_{ph}$) deviation is $\sim 10^{-70}$. Any observed macroscopic deviation in EHT shadows would falsify the theory.
7. **Tolman Cycle Count:** The current universe is predicted to be cycle $\sim 77$, with a turnaround lifetime of $\approx 24.7$ Byr.
8. **Tidal Deformability (GW170817):** The runtime canonical chain computes $\Lambda_{1.4}=519.4$; the comparison with the declared GW170817 interval $[70,720]$ is conditional/in-sample, not an independent confirmation.
9. **Multi-Meson Spectroscopy:** In-medium at $2n_0$, masses shift in a strict hierarchy: $\rho, \omega$ (-20.0%), $K^*$ (-7.8%), $\phi$ (-2.9%), $J/\psi$ (-0.4%). (Template for HADES/CBM/NICA).
10. **Quantitative CMB Suppression:** For $\ell > 10$ ($k > 10^{-3}$ Mpc$^{-1}$) the spectrum coincides with $\Lambda$CDM (ratio 1.000). However, at $k < 3 \times 10^{-4}$ it drops exponentially due to the finite size of the Genesis instanton.
11. **Multi-Mass PBH Spectrum (Dark Matter):** The cycle ladder supplies a conditional mass-rung template from the asteroid window through heavy candidates; object-level JWST seeding is retired pending a machine-readable catalogue, occupation model, and selection likelihood.
12. **GW Echo Template:** Parameterized echo train with decaying amplitude ($R_{\rm core}^n$) and alternating phase — ready-to-use template for LIGO matched-filtering.
13. **NS Cooling Population Dichotomy:** Strict threshold at $1.45 M_\odot$. Regardless of envelope composition, light NSs are bright ($10^{33}$ erg/s), while heavy ones (Direct Urca) drop to $10^{31}$ erg/s. An old, hot heavy star falsifies the EOS.
14. **Gravitational Redshift and f_peak:** Runtime canonical calculations give $z_{surf} \approx 0.222$ at $1.4 M_\odot$ and $f_{peak} \approx 2.42$ kHz from $R_{1.6}\approx12.65$ km; both are forward calculations, with no observed target yet.
15. **Cycles and Genesis Robustness:** The closed form $n = N_e/\ln 2 = 76.58 \pm 0.013$ (anchored $H_0 = 72.8$) lands exactly inside integer cycle 77 of the Tolman ladder: the interval $N_e \in [76\ln2, 77\ln2] = [52.68, 53.38]$ (`verification/nvg_hubble_tension.py`). Audit: the previously quoted "77.2" is $\log_4(2.6\times10^{122}/10^{76}) = 77.09$ built on rounded E&L entropy anchors, and "±0.3" is the cycle-interval half-width $\ln2/2$, not the QCD sensitivity (true value ±0.013) (`verification/nvg_dual_hierarchy_identity.py`).
16. **EM Sector ($\epsilon_{eff}$):** The effective vacuum dielectric constant in a NS core drops to $\epsilon_{eff} = 0.135 \epsilon_0$, preserving QED on Earth ($\epsilon_{eff} = \epsilon_0$).
17. **W-Sector Lorentz Invariance:** Outside dense media, vacuum dispersion and birefringence are strictly $0.0$, satisfying the most stringent GRB astrophysical limits.
18. **Kerr QNM (Ringdown):** The Hayward core modification at the Planck scale shifts Quasi-Normal Mode frequencies by $\sim 10^{-105}$, making the geometry mathematically indistinguishable for LIGO/LISA.
19. **Moment of inertia of J0737A:** The linked Hartle calculation is a dated alternate calibration, not the maintained canonical chain and carries zero independent evidence weight; future orbital precession measurements remain the discriminating test (`verification/nvg_moment_of_inertia_j0737.py`).
20. **0νββ null test:** The minimal $\theta$-seesaw predicts $m_{\beta\beta} \in [0, 6.4]$ meV; a confirmed signal with $m_{\beta\beta} > 10$ meV excludes the sector (`verification/nvg_neutrinoless_dbeta.py`).
21. **Fifth force at 3.7 mm:** The monopole channel of the $\theta$ mode is identically zero (pseudoscalar); any unpolarized Yukawa signal at 3–4 mm falsifies the sector (`verification/nvg_theta_fifth_force.py`).
22. **Cosmic birefringence:** Static branch $\beta = 0$; a detection of $\beta \neq 0$ at LiteBIRD precision pins the combination $\gamma_{\rm topo}\Delta\theta$ ($0.033°$ per unit) (`verification/nvg_cmb_birefringence.py`).
23. **$\theta$-mode superradiance:** Kerr band $M_{\rm BH} \in [5\times10^{-7}, 1.3\times10^{-6}]\,M_\odot$; a spinning horizon PBH inside the band excludes the horizonless interpretation of the corresponding remnants (`verification/nvg_theta_superradiance.py`).
24. **Tidal heating:** Horizonless remnants predict a zero absorption-phase contribution in EMRIs vs ~32 cycles for a Kerr BH (LISA) (`verification/nvg_tidal_heating_null.py`).
25. **Dark-sector band discrimination:** The NVG bands (8–53 μeV) and the dark-dimension KK tower band (2 meV–1 eV at $R\in[0.2,100]$ μm) are disjoint — a detection in either band discriminates the programs; live falsifiers: DESI $w(z)\neq-1$ and the kinematic neutrino mass — 2026-08 correction: the KATRIN observable is $m_\beta = \sqrt{\sum|U_{ei}|^2m_i^2} = 8.9$ meV, not 29 meV (equi-weighting $m_3/\sqrt3$); registered falsifier $m_\beta > 20$ meV (`verification/nvg_de_axion_crosscheck.py`).
26. **Quad-hierarchy identity:** $\rho_c r_c^2 = \rho_{\rm crit}R_{H0}^2 = 3c^2/8\pi G$ → four hierarchies are powers of the single number $e^{N_e} = R_{H0}/r_c$: density $\rho_c/\rho_{\rm DE} = e^{2N_e}/\Omega_{\rm DE} = 1.85\times10^{46}$, entropy $S_{\rm now}/S_{\rm gen} = e^{2N_e} = 4^n$, mass $M_U = M_1\times 2^n = 4.30\times10^{22}\,M_\odot$ (the Hubble-sphere mass) and time $t_{H0}/t_b = e^{N_e} = 1.13\times10^{23}$ (precision $2\times10^{-15}$); the entropy-per-baryon ratio honestly does not close (area law vs volume count). Status: closed identity with a calibrated $H_0$ — consistency, not evidence (`verification/nvg_dual_hierarchy_identity.py`).
27. **Parameter-free neutron-star mass bound:** $M_{\rm Pl}^3/M_\Omega^2 = 2.211^{+0.042}_{-0.041}\,M_\odot$ is a separate dimensional bound diagnostic from the single anchor $M_\Omega$, not the runtime canonical TOV result. Its live-data comparison and falsifier are documented in `verification/nvg_ns_mass_bound.py`; this bound and all alternate fork-B outputs carry their own assumptions and zero independent evidence weight relative to the canonical chain.
28. **The cornered neutrino-mass sum:** $\Sigma m_\nu = 59.0$ meV — the minimum of normal ordering at $m_1\approx0$, rigidly fixed by the oscillation splittings; DESI DR1 + CMB + SN (UL 72 meV) leaves a margin of only +13 meV; the disputed negative DESI DR2 posterior ($-75\pm51$ meV) yields a $2.6\sigma$ pull; CMB-S4 ($\sigma\sim15$ meV) will deliver a two-sided verdict: a $\sim4\sigma$ exclusion if the central stays near zero, or a confirmation if it lands at 59 meV. Falsifier: a robust $\Sigma m_\nu < 45$ meV or a confirmed negative best fit excludes the minimal $\theta$-seesaw (`verification/nvg_sigma_mnu_corner.py`).
29. **Kinematic neutrino-mass correction and one-sided falsifier:** the KATRIN observable $m_\beta^2 = \sum_i|U_{ei}|^2m_i^2 = 8.9$ meV (minimal NO; 9.0 meV for the anchor spectrum) — earlier "29 meV" citations were the equi-weighting $m_3/\sqrt3$, not the observable; KATRIN (200 meV design) is honestly insensitive, the two-sided test is Project 8 ultimate ($\sigma\sim10$ meV); registered falsifier: any robust $m_\beta > 20$ meV excludes the sector, since the inverted-ordering floor is 49.2 meV; the consistency triangle $\Sigma m_\nu = 59$ meV / $m_\beta = 8.9$ meV / $m_{\beta\beta}\le 6.4$ meV from one anchor is rigidly closed (`verification/nvg_katrin_mbeta.py`).
30. **O4 echo observer card:** for the four ROADMAP candidates (GW230518/230615/230922/231215, $M_{\rm tot} = 61.8$–$70.2\,M_\odot$) the predicted echo spacing is $\Delta t_{\rm echo} = 20.0$–$23.9$ ms (remnant-mass systematic $\times0.95$ included), carrier $f_{\rm QNM} = 245$–$278$ Hz, damping $\tau = 3.3$–$3.7$ ms, 14–16 pulses per comb at $R_{\rm eff} = 0.95$; the protocol (search window, bandpass, delay scan $\times0.73$–$\times1.55$, coherent statistic vs time-slide background) matches the repo pipeline; falsifier: a null deep stack of the full O4 catalog at the tabulated delays drives $R_{\rm eff}\to0$ and falsifies the reflective-core picture of the remnant (`verification/nvg_echo_o4_observers.py`).

---

## Speculative Directions & Future Tech

### 1. Macroscopic Quantum Entanglement via Vacuum Condensate (QCD to Quantum Optics)
If the VMF vacuum condensate is globally coherent, two spatially separated NVG auto-oscillators should exhibit a non-local correlation mediated by the Goldstone phase $\theta$. This predicts a tiny, anomalous time-dependent contribution to Bell inequality violations. High-precision atomic clock arrays (e.g. at NIST, PTB) could test this macroscopic phase coherence, opening a novel bridge from QCD to quantum optics.

### 2. Dark Matter as a Relic VMF Instanton Condensate
During the post-bounce expansion at $T < T_b$, a toy relic-instanton budget can be evaluated. The freeze-out scale and self-coupling are calibration inputs; without a formation/occupation likelihood the resulting $\Omega_{\rm def}$ is a conditional model sensitivity, not a measured dark-matter fraction or verification (`verification/nvg_relic_dark_matter.py`).

---

## Automatic Verification & Reporting (Runtime Ledger)

The repository includes a unified pipeline for reviewers: `verification/run_nvg_suite.py`. 
Running this script automatically generates `NVG_FINAL_REPORT.md`, which features:
1. **Canonical runtime ledger:** Executes one TOV + Hinderer chain and records $M_{\max}$, $R_{1.4}$, $\Lambda_{1.4}$, and the conditional reduced $\chi^2$ with provenance.
2. **Honest boundary:** Off-anchor uncertainty propagation and inverse QCD reconstruction are currently unsupported and are not reported as results.
3. **Forecast module:** Records future measurements that could test the canonical chain; a process exit code is not a scientific verdict.
4. **Automatic Evidence Ledger:** Maps each maintained result to its source and explicit input/calibration/conditional status.

---

## Author

**Oleg Kirichenko** — Independent Researcher — urevich55@gmail.com

## License

MIT License — see [LICENSE](LICENSE) for details.
