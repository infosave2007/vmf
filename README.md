# Null-Vector Gravity (NVG) & Vacuum Mass Fraction (VMF) Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://python.org)
[![CI Build & Verification](https://github.com/infosave2007/vmf/actions/workflows/verify.yml/badge.svg)](https://github.com/infosave2007/vmf/actions/workflows/verify.yml)


### Reproducible verification interface

The maintained computational surface is indexed in the machine-readable
[script and artifact registry](verification/registry.json). Run the ordinary
scientific gate from the repository root with
`python verification/run_verification.py`; it validates registry/provenance,
regenerates the canonical runtime ledger, and runs
`python -m pytest -q verification`. Registry roles distinguish `canonical`,
`validation`, `forecast`, `synthetic`, `retired`, `utility`, and `historical`
surfaces. The separate [`run_all_checks.py`](verification/run_all_checks.py)
command is a process-only smoke harness and is not scientific evidence.
Input provenance and generated-artifact ownership are recorded in the
[`provenance.json`](verification/data/provenance.json) and
[`artifact_manifest.json`](verification/artifact_manifest.json) manifests. Current runtime provenance and the distinction from historical publications are described in [PUBLICATION_STATUS.md](PUBLICATION_STATUS.md).

### Latest accepted finite-temperature bridges

The accepted finite-temperature closure adds two bounded, live calculations.
The master response rebuilds eight W8/U16 states and publishes **160** complex
off-pole/static rows plus **32** analytic spacelike absorptive-cut rows.  Each
response keeps all six components of the three density/scalar/longitudinal-
current operators, the Ward/static/dynamic controls, the full 4x4 saddle
elimination, and both declared scales without applying a scale twice.  The
composition bridge publishes **32** canonical rows with `mu_B`, `mu_D`, free
energy, pressure, the full Hessian and inverse, and the `(B,Q)` coordinate map.
The fixed `j=11.13601607117819 MeV` contact is a conditional known input only:
these are local mathematical bridges, not a new force, an all-frequency
positive spectral-measure proof, or experimental confirmation.

Reports: [thermal master response](NVG_THERMAL_MASTER_RESPONSE_RU.md),
[thermal composition Hessian](NVG_THERMAL_COMPOSITION_HESSIAN_RU.md), and the
[accepted closure note](NVG_CLOSURE_ADHD_2026_09_20_RU.md).

Focused reproduction (write JSON outside the checkout):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-research/bin/python3.12 -B verification/nvg_thermal_master_response.py \
  --output /tmp/nvg_thermal_master_response_results.json
PYTHONDONTWRITEBYTECODE=1 .venv-research/bin/python3.12 -B verification/nvg_thermal_composition_hessian.py \
  --output /tmp/nvg_thermal_composition_hessian_results.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=verification .venv-research/bin/python3.12 -B -m unittest \
  verification.test_thermal_master_response \
  verification.test_thermal_composition_hessian -v
```

Use Python 3.12 and a fresh environment; CI uses the same requirements and
[resolved constraints](constraints-python312.txt). From the repository root:

```bash
python3.12 -m venv .venv-research
.venv-research/bin/python -m pip install -r requirements.txt -c constraints-python312.txt
.venv-research/bin/python -m pip check
.venv-research/bin/python verification/run_verification.py
```

The constraints pin NumPy 1.26.4, SciPy 1.13.1, mpmath 1.3.0 and the remaining
verification dependencies. Python 3.9 environments from older runs, including
an existing `.venv`, are not the supported setup. Preserve them if needed and
use the fresh environment above. With another working directory, pass the
absolute path to both the environment's Python and `run_verification.py`.
PyCBC is included because the semantic suite imports the echo-search modules.
Optional live GW fetching additionally requires `gwosc` and external data access.

The [source-complete model document](SOURCE_COMPLETE_MODEL_RU.md) (Russian)
states the action, parameter provenance, EOS and independently calculated
consequences of the effective one-component model. Its fixed parameters do
not reproduce symmetric nuclear-matter saturation: at the reference density,
the pressure is positive and the energy per baryon exceeds the rest mass.
Numerical consistency therefore does not establish a physical neutron-star
model or empirical confirmation. The maintained calculation has an
[equilibrium audit](verification/source_complete_solution_audit.py), a
[mathematical stellar sequence](verification/source_complete_stellar_prediction.py),
and a [vacuum-spectrum/conditional-response audit](verification/source_complete_resonator_prediction.py).
Current artifact acceptance is recorded in the registry and manifest above;
the model document explains the physical limits independently of that status.

For the generator branch, see the [practical energy audit producer](verification/nvg_generator_practical_audit.py),
[semantic tests](verification/test_generator_practical_audit.py),
[deterministic JSON](verification/nvg_generator_practical_audit_results.json), and
[Russian report](verification/NVG_GENERATOR_PRACTICAL_AUDIT_RU.md). It is a
zero-weight utility/non-evidence audit: unsupported vacuum/CISS/over-unity paths
are rejected, while only a configurable conventional waste-heat Seebeck model is
retained.

**Preprints:** Published titles and links are retained as a historical bibliography. Their conclusions are not automatically results of the current executable model.
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20214457-blue.svg)](https://doi.org/10.5281/zenodo.20214457) *Lattice Sigma Terms as an Anchor for the Dense Nuclear Matter Equation of State*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269567-blue.svg)](https://doi.org/10.5281/zenodo.20269567) *Analytic Derivation of the Dense Matter Equation of State and Maximum Neutron Star Mass via QCD Vacuum Condensate Phase Transitions*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269725-blue.svg)](https://doi.org/10.5281/zenodo.20269725) *Geometric Truncation of Low-Multipole CMB Power and Null B-Mode Prediction from a QCD-Scale Euclidean Instanton Bounce*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269816-blue.svg)](https://doi.org/10.5281/zenodo.20269816) *A Discrete Cyclic Mass Hierarchy 4^N for Primordial Black Holes: Bridging Asteroid-Mass Dark Matter to Early JWST Heavy Seeds*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20270025-blue.svg)](https://doi.org/10.5281/zenodo.20270025) *Resolution of the Slow-Rotating Magnetar Paradox via QCD Vacuum Permeability Phase Transitions*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20270202-blue.svg)](https://doi.org/10.5281/zenodo.20270202) *Eliminating the Observer Effect: Wave Function Collapse as Deterministic Topological Reconnection in a Condensate Vacuum*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20463836-blue.svg)](https://doi.org/10.5281/zenodo.20463836) *Neutron Star Structure from a Single QCD Parameter: Equation of State, Tidal Deformability, and Cooling Threshold in the Null-Vector Gravity Framework*
  Historical publication bundle (not current runtime evidence). The maintained code exposes a hyperon-threshold proxy, raw RMF comparisons, and a calibrated cooling threshold; a complete beta-equilibrated EOS and independent likelihood are not implemented, so no hyperon-puzzle resolution is claimed.
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20473318-blue.svg)](https://doi.org/10.5281/zenodo.20473318) *Dynamics of the QCD Vacuum Condensate Amplitude in Dense Matter and Cosmology*
  Historical proposal for radial-field dynamics and an FLRW melting/bounce interpretation near $n_B\approx2.05\,n_0$. The maintained source-complete action instead has positive vector energy diverging as $W^{-2}$ at fixed nonzero baryon density, so that density-melting/bounce claim is not supported by the current model.
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20476890-blue.svg)](https://doi.org/10.5281/zenodo.20476890) *Seven Results of the Vacuum Condensate: From Nuclear Matter to Quantum Mechanics*
  Historical collection of Madelung, stochastic-electrodynamics and Einstein–Cartan proposals. Fourier uncertainty identities and illustrative standard-formula calculations do not derive quantum measurement, Hawking radiation or a cosmological bounce from the maintained source-complete action; that action includes no torsion sector.
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20485836-blue.svg)](https://doi.org/10.5281/zenodo.20485836) *Resolution of the Hyperon Puzzle via QCD Vacuum Condensate Melting in the NVG/VMF Framework*
  Historical publication bundle (not current runtime evidence). It describes a proposed hyperonic phase-transition model; the maintained repository does not provide the independent EOS construction/likelihood needed to call this an empirical resolution.

## Density-tail universality and calibration limits

The latest [calculation and derivation (Russian)](NVG_DENSITY_UNIVERSALITY_RU.md)
move beyond finite droplets to the homogeneous cold-matter model. For a
regular positive power-law potential `U ~ a W^p`, constant positive couplings
and the stated minimum/regularity assumptions,

`lim(P/epsilon) = lim(dP/depsilon) = max(1/3, (p-2)/(p+2))`.

An explicit positive deformation preserves the vacuum and calibration
potential derivatives through order three but changes the formal tail limit
from `3/5` to `7/9`. The first local distinguishing coefficients are
`Delta Z = 2.1207879107 MeV` and `0.04849382613 MeV` for the two declared
calibrated examples. These are within-model mathematical results, not a
universal law, measurements, a new physical interaction, or a bounce proof.

Reproduce the five-case/35-state calculation and its 13 focused tests
with `mpmath` installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B verification/nvg_density_universality_audit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=verification python -B -m unittest verification.test_nvg_density_universality_audit -v
```

The program computes its output afresh and rejects corrupted numerical
evidence. See the [frozen contract](verification/contracts/density_universality.md)
for assumptions and the [research directions](NVG_BEYOND_DROPLETS_ADHD_RU.md)
for the full idea selection. The shared `BulkModel`/`inverse_potential_jet`
API is included as a dependency; the separate upstream audit CLI is outside
this focused workflow.

## Finite-nucleus static branch: blind isovector probe and its identification

The static W8.93 branch (no rho, s* = 0.2268, Ca40 anchor) was extended
through a pre-registered chain of finite-nucleus probes: isovector contact
with form factor, the held-out Ca48 test, and the four-nucleus
[discriminating set](NVG_SN132_DISCRIMINATING_SET_RU.md)
{Zr90, Pb208, Ca48, Sn132}. Verdict: **UNIVERSAL_CONSTANT_SUPPORTED** — a
single bulk isovector constant j = 11.14 MeV (J_design = 23.78 MeV) collapses
the binding-energy residual 202.7 -> -3.4 MeV (~60x) with anchor drift
-0.04%; the four tangents span 10.44-12.38 MeV (mean 11.136 MeV).

The follow-up [identification audit](NVG_J_IDENTIFICATION_RU.md) shows what
this constant is: the **interaction part of the classical nuclear symmetry
energy** — the 1935 Bethe-Weizsacker asymmetry term, equal in the
zero-range limit to the rho-meson channel of QHD:

| Model quantity | Value | Literature match | Deviation |
|---|---|---|---|
| T_CONTACT = k_F^2/(6 E_F) | 12.644 MeV | Fermi-gas kinetic symmetry energy | band 11.0-13.0 |
| J_design = T + j | 23.780 MeV | LDM a_sym 22.90-23.7 MeV | +0.3...+3.9% |
| j (interaction piece) | 11.136 MeV | a_sym - T_kin = 10.9-11.4 MeV | within 2.5% |
| implied g_rho (tau/2) | 6.60 | published RMF range 4.1-6.8 | inside |

No new particle or force is claimed. The new element is the method: an
isoscalar core measuring the symmetry-energy interaction part blind, with
pre-registration, at 1-3% accuracy. Radii/skins remain outside the static
branch (shell physics); adopting the isovector term as a theory element is a
pending author decision; all these audits carry evidence_weight = 0.

The closing [transferability probe](NVG_ISOVECTOR_TRANSFERABILITY_RU.md)
tests the constant from both sides. Part A: a secant re-anchoring of s\* on
Ca40 under the rho contact gives ds/s = -0.041% (s\*_rho = 0.2267063) —
verdict **REANCHORED_UNIVERSAL_J_SURVIVES**, the residual collapse
202.7 -> -2.6 MeV survives a clean recalibration of the anchor. Part B: the
canonical NS chain is one-component (isoscalar EOS — the rho term cannot
enter it), and transferring j = 11.136 MeV into the two-component
saturated-vector family (C_rho -> 2j/n0 = 139.2 MeV fm^3) yields realistic
J = 29.0 / L = 85 MeV and RAISES M_max 2.048 -> 2.095 Msun (verdict
**TRANSFERABLE_TO_NS_SECTOR**), but breaks the secondary screening bands
(R_1.4 = 13.45 km > 13.2; Lambda_tilde ~ 940/870 > 720): the mechanism is
the tied saturation calibration compensating the softer isospin by raising
c_omega0 (1794 -> 2074), stiffening the EOS. Conclusion: j is not a freely
transferable drop-in constant — the NS family would require re-selection,
not a single substitution.

That re-selection is now done: the pre-registered [soft-isospin re-selection
scan](NVG_SOFT_ISOSPIN_RESELECTION_RU.md) (3150-point grid over the isoscalar
family shape and transition parameters at C_rho = 2j/n0) answers in two
stages. Stage 1 (transition parameters alone at the baseline isoscalar
shape): 0 of 42 survivors — drop-in re-selection cannot absorb the shift.
Stage 2 (full family): 75 survivors, verdict **SOFT_ISOSPIN_SURVIVOR_FOUND**.
Best point k1 = 0.20, k2 = 0.60, Cs = 900, n_tr = 1.8, de = 0.10: M_max =
2.0405 Msun, R_1.4 = 12.588 km, Lambda_tilde = 634/624 (margin +0.431,
binding: GW170817), UNIQUE_STABLE_BRANCH, confirmed at 120-point resolution;
the surviving region is a wide plateau (k1 in {0.20, 0.25}) rather than a
fine-tuned needle. A mildly faster scalar-mass decay (k1 0.25 -> 0.20)
compensates the isospin-driven stiffening and restores all three bands while
keeping the identified j — one constant, both sectors, within the family
(conditional, in-sample, evidence_weight = 0).

The first genuinely external test — the [PREX-II/CREX consistency
probe](NVG_PREX_CREX_CONSISTENCY_RU.md) — then compares the point neutron
skins of Pb208/Ca48 on both branches against the published two-sigma bands
(pre-registered; neither j nor s* was ever tuned to a skin). Verdict:
**SKINS_EXCLUDED_BOTH_BRANCHES** (Pb208: -5.1 sigma no_rho / -4.4 sigma
identified_j; Ca48: -3.4 / -2.8 sigma). The plane-wave F_W and A_PV run high
by the same compactness deficit, and the naive contact strength the skins
would require (58-72 MeV) is ~5x the identified j. This cleanly separates the
bulk energy channel (where j works at 1-3%) from the surface/shell channel
(direction IV): the exclusion confirms the documented boundary of the static
TF branch rather than failing the constant (evidence_weight = 0).

Reproduce the probes, the identification, the transferability, the
re-selection and the external-consistency probe (141 focused tests total):

```bash
PYTHONDONTWRITEBYTECODE=1 python -B verification/nvg_sn132_discriminating_set_probe.py
PYTHONDONTWRITEBYTECODE=1 python -B verification/nvg_j_identification_audit.py
PYTHONDONTWRITEBYTECODE=1 python -B verification/nvg_isovector_transferability_probe.py
PYTHONDONTWRITEBYTECODE=1 python -B verification/nvg_soft_isospin_ns_reselection_probe.py
PYTHONDONTWRITEBYTECODE=1 python -B verification/nvg_prex_crex_external_consistency_probe.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=verification python -B -m unittest \
  verification.test_nvg_sn132_discriminating_set_probe \
  verification.test_nvg_j_identification_audit \
  verification.test_nvg_isovector_transferability_probe \
  verification.test_nvg_soft_isospin_ns_reselection_probe \
  verification.test_nvg_prex_crex_external_consistency_probe -v
```

Contracts: [nuclear isospin jet](verification/contracts/nuclear_isospin_jet.md),
[nuclear closure](verification/contracts/nuclear_closure.md). Reports:
[isovector form-factor probe](NVG_ISOVECTOR_FORMFACTOR_PROBE_RU.md),
[Ca48 held-out probe](NVG_CA48_QUANTUM_SURFACE_PROBE_RU.md),
[discriminating set](NVG_SN132_DISCRIMINATING_SET_RU.md),
[j identification](NVG_J_IDENTIFICATION_RU.md),
[isovector transferability](NVG_ISOVECTOR_TRANSFERABILITY_RU.md),
[soft-isospin re-selection](NVG_SOFT_ISOSPIN_RESELECTION_RU.md),
[PREX/CREX consistency](NVG_PREX_CREX_CONSISTENCY_RU.md). The full scientific
front door (registry validation with 512 entries, predictive ledger, canonical
suite, 1120 semantic tests) is green at this state.

## Overview

This repository contains theoretical proposals, numerical models, diagnostic audits and illustrative experiments for **Null-Vector Gravity (NVG)** and the **Vacuum Mass Fraction (VMF)** program. Their implementation and evidence status differ; the source-complete model and its limits are documented above.

The historical mass-budget convention assigns about 91% of the nucleon mass to the remainder after subtracting adopted sigma-term contributions. The quoted $M_{\Omega,0}=859$ MeV is a derived budget/model input, not a direct lattice measurement of a macroscopic field. Identifying it with $W_0$, selecting the potential and choosing couplings require additional assumptions. The maintained model is therefore not a parameter-free bridge from QCD to cosmology.

### Historical field-action proposal
The following schematic action summarizes historical extensions. Its electromagnetic/topological terms do not establish the three research directions below as consequences of the maintained source-complete action, whose normalization, matter coupling and Gauss constraint are specified in the model document.

$$ S = \int d^4x \sqrt{-g} \left[ \frac{R}{16\pi G} - g^{\mu
u} \partial_\mu \Phi^* \partial_
u \Phi - V(|\Phi|) - \frac{1}{4} Z_{\rm EM}(\mathcal{W}) F_{\mu
u} F^{\mu
u} + \gamma_{\rm topo} \frac{\alpha_{\rm EM}}{8\pi} \theta F_{\mu
u} \tilde{F}^{\mu
u} \right] $$

where $\Phi(x) = \mathcal{W}(x) e^{i\theta(x)}$ is the complex vacuum condensate order parameter.
* **Scalar amplitude ($\mathcal W$):** The historical proposal links suppression of the field to dense-matter and cosmological effects. In the current fixed-density homogeneous action, complete melting at nonzero density is excluded by the positive $W^{-2}$ vector term; singularity resolution does not follow.
* **Phase and topological-flow proposals ($\theta$):** A timelike phase gradient and a $\theta F\tilde F$ interaction can be posited in an extension. Emergent time and magnetar amplification require specified dynamics and matching; the gauge phase of the maintained Higgs sector is not an additional physical Goldstone oscillator.

###### Historical field-dynamics and melting analysis
The linked preprint develops a field ansatz and a proposed connection to QCD sigma terms. It does not supply the missing operator matching, parameter uncertainties or a derivation of the cosmological correction $-\rho^2/\rho_c$ from the current action. The source-complete equilibrium audit preserves the accepted parameters and finds a density-melting no-go.

For the historical derivation and assumptions, see [NVG_VACUUM_W_FIELD_DERIVATION_EN.md](article/NVG_VACUUM_W_FIELD_DERIVATION_EN.md).

The repository retains three research directions with separate assumptions and validation boundaries:

### Pillar I: Dense Nuclear Matter (VMF)
The historical melting-based EOS program (see [Zenodo 20463836](https://doi.org/10.5281/zenodo.20463836)) and the canonical calibrated TOV chain are distinct from the current source-complete one-component EOS. The maintained runtime surfaces:
- **Compute a hyperon threshold proxy and raw NL3+Lambda curves.** A complete beta-equilibrated hyperonic EOS and independent likelihood are not implemented, so no puzzle resolution is claimed.
- **Compute a calibrated Direct-Urca cooling threshold.** The threshold is an in-sample calibration, not an independent cooling-population confirmation.
- **Computes a canonical conditional neutron-star chain** (runtime $M_{\max}=2.048\,M_\odot$, with a CSS branch at $c_s^2=1/3$; transition parameters are selected in-sample and are not independent evidence).
- **Computes the canonical tidal chain and a transform-only I-Love-Q visualization.** The plotted $I(\Lambda)$ and $Q(\Lambda)$ points apply the Yagi–Yunes fits to runtime $\Lambda$ values; no independent moment-of-inertia or quadrupole solve is present, so this is not an EOS-universality test. The canonical transition is selected in-sample from J0740, GW170817, and NICER constraints; downstream NS comparisons are conditional/in-sample, not independent confirmations (`verification/nvg_iloveq_plot.py`; [runtime report](verification/fig_iloveq_universal_report.json)).
- **Use the adopted $859$ MeV mass-budget scale** alongside declared couplings and EOS calibration parameters; no elimination of parameter degeneracy or independent lattice calibration of $W_0$ is established.
- Provides a forward $\rho$-meson mass/line-shape template for FAIR/HADES; no event likelihood is present.

### Pillar II: Cyclic Cosmology & Genesis (NVG)

The historical cosmology assigns $\rho_c=M_{\Omega,0}^4/(\hbar c)^3\approx7.09\times10^4$ MeV/fm³ from the adopted mass scale. The ratio $2.5\times10^{-77}$ to the Planck density is a dimensional comparison, not proof that a bounce occurs or that all quantum corrections are controlled.

**Conditional bounce ansatz:** Historical calculations use the modified Friedmann equation below. Its negative quadratic term is an additional assumption; the current source-complete action provides no derived FLRW or Einstein–Cartan torsion completion that generates it. Algebraic evaluation of this equation is not a derivation of the bounce mechanism.

$$H^2 = \frac{8\pi G}{3}\rho\left(1 - \frac{\rho}{\rho_c}\right),\qquad k=0$$

Here rho is mass-equivalent energy density, with no separately added Lambda. This conditional flat case now has an exact solution and independent integration through the bounce. Its pressure is constrained by conservation: epsilon_eff=F(epsilon) requires P_eff=(epsilon+P)Fprime-F. The modification is still not derived from the NVG action. [Equations, checks and limitations (Russian)](BLACK_HOLES_AND_CYCLES_RU.md).

Conditional scales computed from this ansatz and its declared inputs:

| Quantity | Value | Derivation |
|----------|-------|------------|
| Assigned energy-density scale $\varepsilon_c$ | $7.09 \times 10^4$ MeV/fm³ | $M_{\Omega,0}^4/(\hbar c)^3$ |
| $\rho_c / \rho_{\rm Planck}$ | $2.5 \times 10^{-77}$ | Scale comparison only |
| Bounce timescale $t_b$ | $3.76 \times 10^{-6}$ s | $(8\pi G\rho_c/3)^{-1/2}$ |
| Bounce temperature $T_b$ | 432 MeV | Stefan-Boltzmann QGP ($g_*=47.5$) |
| Holographic entropy (Universe) | $2.15 \times 10^{91}$ | $4\pi r_0^2 / 4\ell_{\rm Pl}^2$ |
| CMB/BAO $\delta H/H$ | $\sim 10^{-38}$ | Historical ansatz estimate; no CMB/BAO likelihood validation |

**Genesis Phase — The First Cycle:**
The historical Genesis construction assumes a Euclidean instanton, Hartle–Hawking-type boundary conditions and a starting point at the assigned density with zero expansion rate. Under those assumptions its radius is:

$$r_c = \frac{c}{\sqrt{8\pi G \rho_c / 3}} \approx 1.13 \text{ km}$$

This gives an initial mass of only $M_1 \approx 0.38\,M_\odot$ and a first-cycle lifetime of **~5.9 microseconds**. Time itself is proposed to emerge as the Goldstone mode ($dt \propto d\theta$) of spontaneous $U(1)$ symmetry breaking in the vacuum phase sector.

**Tolman's Entropy Snowball:**
The cycle-growth scenario assumes that entropy production survives a bounce and enlarges subsequent cycles. Neither the transfer mechanism nor a measured cycle count is supplied by the equilibrium action. Its illustrative sequence is:
- **Cycle 1:** $r_c \approx 1.13$ km, lifetime $\sim 5.9\,\mu$s
- **Assigned element 77 (76 mass doublings):** the shared producer gives $M\approx5.7393\times10^{55}$ g and a conditional scale $\pi GM/c^3\approx14.1592$ Byr. This is neither an observed cycle nor time remaining to turnaround; the former 24.7 Byr does not follow from these inputs.

**CMB Low-$\ell$ Prediction:**
The finite instanton size $r_c$ stretched by $N_e \approx 53$ e-folds (calibrated to the local $H_0$) spans the present Hubble horizon by construction; the falsifiable content is the resulting deterministic infrared cutoff that suppresses CMB multipoles $\ell=2,3$. The maintained script runs a documented lite likelihood over vendored Planck-shaped arrays whose provenance is unknown — it is **not the Planck likelihood** and its $\Delta\chi^2$ (about $+0.9$ for the default cutoff) is descriptive only, not a Planck-fit p-value or confirmation. The cutoff does not shift the CMB-inferred $H_0$ ([nvg_cmb_lowl_refit.py](verification/nvg_cmb_lowl_refit.py)).

### Pillar III: Regular-core geometries and information-transfer proposals
A separate Hayward-metric ansatz uses a regular de Sitter core with $\ell=\sqrt{3c^2/(8\pi G\rho_c)}\approx1.128$ km from the adopted density scale. The mass-enclosing radius $r_0=(3M/4\pi\rho_c)^{1/3}$ is distinct: $3.35$ km at $10\,M_\odot$. The quoted $10\,M_\odot$ example has horizons at 1.15 and 29.34 km and finite $K(0)=24/\ell^4=14.86\,\mathrm{km}^{-4}$. These are properties of the chosen metric, not a solved collapse or a measured core.

**Dynamical boundary:** A de Sitter-like stress or a regular metric does not by itself show that collapsing matter reaches that state. The maintained action does not derive the proposed melting transition, a stable inner horizon or a unitary evolution through a bounce. Those require separate dynamical and stability calculations.

**Information-transfer hypotheses:** The following ideas remain proposals; metric regularity alone does not establish quantum unitarity:
- Holographic compression: the quoted $\sim10^{32}$ factor is model bookkeeping, not a derived information-conservation law.
- Unitary transfer: $\mathcal I_{n+1}=\mathcal U_b\mathcal I_n$ names an assumed map; its operator and dynamics are not constructed.
- Two horizons and finite curvature do not prove the absence of information loss or inner-horizon instabilities.

**Conditional echo scales:** A reflective-core template gives $\Delta t_{\rm echo}\approx22$ ms for a $65\,M_\odot$ benchmark, with a stated $\sim16$–$34$ ms cutoff sensitivity. The separate constant-density standing-wave toy model gives $T_1\approx42\,\mu$s; the ratio of these scales is about 524. Neither the ratio nor the quoted $\sim3.6$ ms Kerr ringdown, $\sim2$ mm cutoff or $\sim10^{-15}$ K inner-horizon temperature establishes a physical $W$-mode or reflectivity. A Boltzmann factor alone does not calculate an absorbing boundary. Echo existence, amplitude and lifetime require a sourced perturbation/boundary model; they are not parameter-free consequences of the current action.

**EHT scope:** The historical benchmark quotes a photon-sphere deviation of $\sim-0.002\%$. This metric-level sensitivity estimate is not an EHT likelihood or a proof of exact Kerr/Schwarzschild equivalence. Its assumptions must be fixed before comparison; echoes are a separate conditional channel.

## Observational Status & Runtime Evidence Boundaries (May 2026)

The tables mix historical proposals, assumed scales, calibrated outputs and mathematical checks. They do not describe a single parameter-free model: the mass-budget scale, couplings, EOS choices, cosmological normalizations and extension-specific inputs have distinct provenance.
For the sake of scientific rigor, the results are divided into: direct unique predictions, consistency checks (null tests), and falsifiable forecasts.

### 1. Runtime Comparisons and Model-Status Ledger
These rows summarize declared runtime calculations, formal identities, conditional comparisons, and forecasts; they are not an aggregate confirmation count.

**Status legend:** ✅ compatible with data · 📏 interval prediction · 🔭 forward prediction (no measurement yet) · ⚙️ calibration / consistency check · ⚪ null test · 📐 mathematical result (not an experimental confirmation) · ❓ conjectural · ❌ open problem / retracted · ⏳ awaiting experiment. Falsifiable rows state the measurement that would exclude the model. Statuses are per-row. The global-significance entry point computes descriptive conditional pulls but **withholds any global p-value** because no traceable held-out producer exists; the README table is presentation metadata, not a source of statistics. NS values below are runtime outputs of the canonical TOV chain and all comparisons are conditional/in-sample with zero independent evidence weight. Provenance note: $T_c = 157$ MeV, used across the scripts, is *adopted* from lattice QCD (HotQCD: $156.5 \pm 1.5$ MeV) — an input identification, not an NVG prediction, and it is deliberately excluded from the test count.

| # | NVG Prediction | Observational Data | Status |
|---|---|---|---|
| 1 | Historical nucleon mass-budget remainder: about 91% ($M_{\Omega,0}=859\pm8$ MeV) | Computed after subtracting adopted sigma-term inputs, including $\sigma_{\pi N}\approx44$ MeV and $\sigma_{sN}\approx30$ MeV; identifying the remainder with a scalar-field scale is a model assumption | ⚙️ Input convention; not an independent confirmation |
| 2 | $M_{\max} = 2.048\,M_\odot$ (runtime canonical TOV chain; transition selected in-sample) | PSR J0740+6620: $2.14 \pm 0.10\,M_\odot$ ($-0.92\sigma$). A confirmed NS above the declared mass bound would test the chain; this comparison is conditional/in-sample, not an independent confirmation | ⚙️ Conditional/in-sample |
| 3 | $R_{1.4} = 12.550$ km (runtime canonical TOV chain) | NICER J0030: $12.2 \pm 0.5$ km ($+0.70\sigma$). The transition was selected using J0740/GW170817/NICER constraints, so this is conditional/in-sample with zero independent evidence weight | ⚙️ Conditional/in-sample |
| 4 | Genesis Instanton $r_c \to$ e-folds bounded to $N_e \in [52.68, 53.38]$ for cycle $n=77$; the position inside the interval, $N_e = 53.08$, is set by the local $H_0$ | $R_{H0} = c/H_0 \approx 1.27 \times 10^{28}$ cm; $N_e = \ln(R_{H0}/r_c)$ at $H_0 = 72.8$ km/s/Mpc ([nvg_hubble_tension.py](verification/nvg_hubble_tension.py) / [nvg_genesis_observable.py](verification/nvg_genesis_observable.py)) | 📏 Interval prediction |
| 5 | NS cooling dichotomy via a direct-Urca threshold at $1.45 M_\odot$ (threshold position set by the coupling $\alpha_v$) | Cas A (slow) vs Vela (fast): the two-regime cooling is reproduced ([nvg_direct_urca.py](verification/nvg_direct_urca.py)) | ⚙️ Calibrated threshold |
| 6 | Tidal deformability: runtime $\Lambda_{1.4} = 519.4$ (TOV + Hinderer) | GW170817 declared input: $\Lambda_{1.4}=190^{+390}_{-120}$; runtime pull $+0.84\sigma$. This row is conditional/in-sample; no independent evidence weight is assigned | ⚙️ Conditional/in-sample |
| 7 | Historical in-medium $\rho$ line-shape templates compare architecture A ($\approx 770$ MeV, broadening-only) and B ($\approx 735$ MeV, scalar-field pull); these are not a mass shift derived from the current source-complete branch | Repository selector windows and conditional feasibility ($3\sigma$ with $\sim 3$–$4\times 10^3$ excess counts): [HADES_PREREGISTRATION.md](verification/HADES_PREREGISTRATION.md), [nvg_hades_lineshape_feasibility.py](verification/nvg_hades_lineshape_feasibility.py); a physical test requires collaboration data and response modelling | 🔭 Conditional line-shape forecast |
| 8 | Assigned cosmological density scale $\rho_c=7.09\times10^4$ MeV/fm³ | Dimensional scale from the adopted mass input. The negative-density correction and torsion completion are not derived from the maintained action; an algebraic H=0 point does not validate collapse/bounce dynamics. | ❓ Historical bounce ansatz |
| 9 | Hubble constant: cycle-77 turnaround horizons bound $H_0 \in [54.3, 108.5]$ km/s/Mpc ($R_{77} = r_c \cdot 2^{76}$); the mid-cycle value is $H_0 = H_{77}/\sqrt{2} \approx 76.8$ km/s/Mpc (measure-dependent center: $62$–$81$) | SH0ES ($73.04 \pm 1.04$) lies within 5% of the mid-cycle value; Planck ($67.4$) within the measure spread. The IR cutoff leaves the CMB-inferred $H_0$ unchanged (quantitative re-fit: [nvg_cmb_lowl_refit.py](verification/nvg_cmb_lowl_refit.py)) | 📏 Interval prediction |
| 10 | Surface gravitational redshift: $z_{\rm surf} \approx 0.222$ for $1.4\,M_\odot$ (computed from runtime $R_{1.4}=12.550$ km) | Direct measurements are currently absent; this is a forward calculation testable by STROBE-X/eXTP ([nvg_ns_redshift.py](verification/nvg_ns_redshift.py)) | 🔭 Forward prediction |
| 11 | Historical-template instantaneous mass-shift hierarchy at $2n_0$: $\rho, \omega$ ($-20\%$), $K^*$ ($-7.8\%$), $\phi$ ($-2.9\%$), $J/\psi$ ($-0.4\%$) — light non-Goldstone mesons shift most (observable spectral shifts are smaller, per row 7) | HADES, CBM (FAIR), NICA, LHC in-medium invariant-mass spectra ([fair_hades_link.py](verification/nvg_fair_hades_link.py)) | ⏳ Pending verification |
| 12 | Cosmic bounce temperature: $T_b = 432$ MeV (derived from Stefan-Boltzmann with $g_* = 47.5$) | Conditional thermodynamic scale for the assigned density and $g_*=47.5$; comparing it with $T_c\approx155$–$175$ MeV supplies no observation of a bounce | ⚙️ Conditional scale |
| 13 | Effective vacuum dielectric constant: $\varepsilon_{\rm eff} \approx 0.135\,\varepsilon_0$ in NS cores (from $e^{-2\alpha_v f_{\rm melt}}$ with phenomenological melting parameters $\kappa$) | Amplifies magnetar seed fields by $1/\sqrt{\varepsilon_{\rm eff}} \approx 2.7\times$ | ⚙️ Scale estimate |
| 14 | Relic dark matter: the observed $\Omega_{\rm DM} = 0.268$ determines the condensate self-coupling $\lambda_v$ | The inferred $\lambda_v$ lands in the $f_0(1370)$–$f_0(1500)$ scalar-meson range ([nvg_relic_dark_matter.py](verification/nvg_relic_dark_matter.py)) | ⚙️ Consistency check |
| 15 | NS core speed of sound: the quark phase follows a CSS ansatz with $c_{s}^2 = 1/3$ (conformal limit) | Compatible with joint NICER+LIGO posterior limits ([speed_of_sound_curve.py](verification/nvg_speed_of_sound_curve.py)) | ⚪ Ansatz parameter |
| 16 | First cycle duration: $\tau_1 = 5.9\,\mu\text{s}$ | A timescale assigned through $\rho_c\to t_b$ in the historical cosmology; no CCC/LQC boundary problem is solved by this dimensional calculation | ⚙️ Conditional timescale |
| 17 | Joint NS comparison: conditional/in-sample reduced $\chi_\nu^2 = 0.684$ over 3 runtime rows (calibration target excluded) | Runtime outputs $M_{\max}=2.048$, $R_{1.4}=12.550$, $\Lambda_{1.4}=519.4$ from the canonical chain; transition selected on J0740/GW170817/NICER ([nvg_joint_ns_inference.py](verification/nvg_joint_ns_inference.py)) | ⚙️ Conditional/in-sample |
| 18 | Scalar Glueball mass: $M_{\rm glueball} = 2 M_\Omega \approx 1.72$ GeV | Lightest scalar glueball ($0^{++}$) from the trace-anomaly excitation vs **Lattice QCD** $1.7 \pm 0.1$ GeV. Caveat: this is a theory-vs-lattice comparison — the *experimental* glueball identification remains unsettled ($f_0(1710)$ candidacy debated; BESIII's $X(2370)$ unconfirmed in other channels) | ✅ Compatible with lattice (experiment unsettled) |
| 19 | Historical bounce GW template: anchor $f_{77} = 62.8$ nHz and tooth spacing computed from adopted $t_b$, adiabatic redshift and the Tolman law ([primordial_gw_comb.py](verification/nvg_primordial_gw_comb.py)) | $(\alpha, \beta/H)$ in the adopted recondensation ansatz place its signal at $18$–$42\,\mu$Hz with $\Omega_{\rm GW} h^2 \sim 10^{-9}$ ($\mu$Ares band); the PTA-band tail is negligible — the template does not identify the NANOGrav signal with a bounce ([nvg_recondensation_dynamics.py](verification/nvg_recondensation_dynamics.py)) | 🔭 $\mu$Hz forward prediction |
| 20 | Strong-Field Periastron Advance & PPN Parameters | NVG vacuum polarization correction is $\approx 1.6 \times 10^{-10}$ for double pulsar; Solar System PPN parameters $\gamma_{\rm PPN} = 1.0$ and $\beta_{\rm PPN} = 1.0$ exactly satisfy Cassini and LLR bounds ([weak_field_ppn.py](verification/nvg_weak_field_ppn.py)) | ✅ Within Observational Limits |
| 21 | JWST SMBH Mass Spectrum (z = 6–15) | A conditional growth template can be evaluated for a selected mass rung, but no machine-readable JWST object catalogue, occupation model, or selection likelihood is present | ❌ Retired (missing catalogue/likelihood) |
| 22 | FRB DM Population Statistics | No repeater-linked DM/magnetar-mass catalogue or population likelihood is present; the former synthetic KS comparison is retired | ❌ Retired (missing catalogue/likelihood) |
| 23 | Higgsless Proton-to-Pion Mass Ratio | Baryon/pion mass ratio ($M_p \approx 941.4$ MeV, $M_\pi \to 0$ in the chiral limit) anchored to $M_\Omega = 859$ MeV — standard chiral-symmetry-breaking / trace-anomaly mass generation restated in NVG variables | 📐 Mass-budget/chiral-limit illustration |
| 24 | QCD Phase Diagram Vacuum Melting | Vacuum melting boundary at $T_{\rm melt}(\mu_B) = T_b (1 - (\mu_B/1200)^4)^{0.25}$ MeV with $T_b \approx 432$ MeV at $\mu_B=0$ | ❓ Assumed phase-boundary template |
| 25 | PTA-LIGO O4 SGWB Cross-Correlation | Primordial SGWB turn-down at $f > 145$ nHz predicts high-frequency amplitude $\Omega_{\rm GW}(100\text{ Hz}) < 10^{-15}$ | 🔭 Conditional spectral template; no joint likelihood |
| 26 | NICER radii near $1.4\,M_\odot$ (J0437, J0614) | Runtime canonical chain gives $R \approx 12.55$ km at $1.4\,M_\odot$; J0437 $11.36 \pm 0.8$ km is a $+1.49\sigma$ pull, while the edge-on J0614-3329 comparison is a separate observational check. These comparisons are conditional/in-sample; no independent evidence weight is assigned ([nvg_ns_nicer_joint_audit.py](verification/nvg_ns_nicer_joint_audit.py)) | ⚙️ Conditional/in-sample |
| 27 | CMB Temperature $T_{\rm CMB} = 2.725$ K | The result scales linearly with the arbitrary normalization $a_{\rm bounce} = 1$ cm, so the numerical match carries no predictive content; only the adiabatic scaling form is model content ([cmb_temperature.py](verification/nvg_cmb_temperature.py)) | ⚪ No predictive content |
| 28 | Baryon asymmetry | A bounded-temperature no-go argument is retained as a formal result, but no baryon-number/CP-violating source, washout network, or independent abundance likelihood is implemented; the former empirical cogenesis closure is retired ([nvg_baryon_asymmetry.py](verification/nvg_baryon_asymmetry.py)) | 📐 Formal no-go; empirical closure retired |
| 29 | Post-merger peak frequency $f_{\rm peak} \approx 2.42$ kHz | Derived from runtime $R_{1.6}\approx12.65$ km with the stated numerical-relativity fit; no post-merger signal observed yet — testable by LIGO O5 / Einstein Telescope | 🔭 Forward prediction |
| 30 | Quiescent temperature of SGR 1935+2154 | With the heating luminosity set near the observed value and a typical spot size, $T_{\rm spot}$ follows from Stefan-Boltzmann by construction; the VMF content is the qualitative Urca dichotomy (light magnetar keeps a hot spot) ([sgr_temperature.py](verification/nvg_sgr_temperature.py)) | ⚙️ Consistency illustration |
| 31 | LiteBIRD B-mode Polarization Cutoff | predicted tensor-to-scalar ratio $r(l)$ drops below 0.001 at large scales $l < 10$ ([litebird_prediction.py](verification/nvg_litebird_prediction.py)) | 🔭 Assumed cutoff forecast |
| 32 | $S_8$ structure growth | **Open problem.** The maintained calculation gives $S_8 \approx 0.851$ ($4.4\sigma$ from the weak-lensing input): the input-dependent $w_0/w_a$ growth shift moves away from the lensing value. The retired empirical IDE drag-factor resolution claim is not evidence ([nvg_s8_tension_check.py](verification/nvg_s8_tension_check.py)) | ⚠️ WORSENS (not an independent prediction) |
| 33 | NANOGrav SGWB | Retired attribution. The maintained PBH abundance cross-check leaves a large amplitude deficit and the bounce calculation radiates in the $\mu$Hz band; NVG supplies no mechanism for the PTA signal ([nvg_nanograv_background.py](verification/nvg_nanograv_background.py)) | ❌ Retired (no mechanism) |
| 34 | Higgs boson mass shift $\delta m_H \approx 4.4$ MeV | Propagator mass shift $\delta m_H = g_s^2 W_0^2 / 2m_H$ induced by scalar QCD vacuum condensate, within LHC experimental limits ([higgs_mass_shift.py](verification/nvg_higgs_mass_shift.py)) | ⚙️ Assumed-coupling scale; no collider likelihood |
| 35 | PBH DM Fraction Peak | The discrete mass ladder follows from the theory (spacing $\times 2$ per rung); the abundance peak ($N=-21$, $\sim 10^{20}$ g) is in the asteroid-mass window. Critically, since this mass is $\sim 13$ orders of magnitude below $M_{\rm crit}$ (Row 54), these PBHs are strictly naked de Sitter remnants lacking event horizons, fundamentally altering their Hawking evaporation signatures ([pbh_dark_matter.py](verification/nvg_pbh_dark_matter.py)) | ⚙️ Ladder predicted; abundance calibrated |
| 36 | White Dwarf cooling age shift | Predicted effect $\Delta t/t \approx -1.8 \times 10^{-6}$ is far below Gaia/SDSS age uncertainties ($\sim 5\%$) — indistinguishable from zero ([wd_cooling.py](verification/nvg_wd_cooling.py)) | ⚪ Null test (unobservably small) |
| 37 | Neutron star core g-modes | Runtime WKB forecast over an assumed composition grid gives periods in a 50–150 ms sensitivity band; no detector likelihood or confirmation is available ([nvg_ns_g_modes.py](verification/nvg_ns_g_modes.py)) | 🔭 Forecast (assumed composition) |
| 38 | SN1987A dark-photon sensitivity scan | The current reference scan includes a 19.8% mass-drop point with $L_A=8.265\times10^{55}$ erg/s, above the stated Raffelt reference. Without a sourced SN1987A likelihood and validated transport model, it establishes no mass-drop bound or limit status ([nvg_dark_photon_observables.py](verification/nvg_dark_photon_observables.py)) | ⚙️ Conditional scan (no evidence) |
| 39 | de Sitter core standing waves | Constant-density standing-wave toy model with $T_1\approx42\,\mu$s for a $65\,M_\odot$ benchmark; no matched physical $W$-mode, detector waveform or echo likelihood ([ds_core_oscillations.py](verification/nvg_ds_core_oscillations.py)) | 🔭 Conditional toy spectrum |
| 40 | Conditional strong-CP potential | The minimum of an added $\chi(1-\cos\theta)$ term is at $\theta=0$ modulo $2\pi$. Identifying this phase with $\bar\theta_{\rm QCD}$ is a separate physical postulate, not a result of the maintained Higgs action. ([strong_cp_solution.py](verification/nvg_strong_cp_solution.py)) | 📐 Conditional potential identity |
| 41 | Arrow of Time from Topology | Illustration of a selected phase winding $Q=1$ and entropy-growth prescription. Winding sign alone does not derive a statistical H-theorem or its entropy budget. ([arrow_of_time.py](verification/nvg_arrow_of_time.py)) | 📐 Synthetic illustration |
| 42 | Double-slit interference from vacuum hydrodynamics | $|\psi|^2$ pattern reproduced by Huygens-Fresnel integral over vacuum phase $\theta$ (Madelung representation), $r_{\rm Pearson} = 1.000$ ([double_slit_madelung.py](verification/nvg_double_slit_madelung.py)) | 📐 Reproduces standard QM (Madelung) |
| 43 | Null WIMP signal in direct detectors | EFT cross-section sensitivities are compared with named experiment curves as benchmark display inputs; no detector event likelihood is implemented and no exclusion is claimed ([dm_direct_detection.py](verification/nvg_dm_direct_detection.py)) | ⚙️ Model sensitivity (no evidence) |
| 44 | Bell-CHSH correlations in the condensate | The correlation form $E = -\cos(a-b)$ is postulated (preprint Limitations: conjectural); by Bell's theorem a local derivation is impossible — a shared phase read out locally is a hidden variable giving $S \le 2$, so any derivation from the action must contain an explicitly nonlocal or contextual element. **Resolved as a dichotomy** ([nvg_bell_from_action.py](verification/nvg_bell_from_action.py)): a classical spacetime $\theta$ of any dynamics gives $S \le 2$ (verified by exhaustion) — excluded by loophole-free experiments; the quantized W-field yields the configuration-space $\theta$ automatically ([nvg_bell_contextual.py](verification/nvg_bell_contextual.py) reproduces $S = 2\sqrt{2}$) but then QM is the input. The quantum block is a consistent hydrodynamic *representation*, not a derivation; its falsifiable content is $S(T > T_c) \to 0$ (row 51) | 📐 Resolved: representation, not derivation |
| 45 | Heisenberg uncertainty | $\Delta x\,\Delta p\ge\hbar/2$ is the standard Fourier/Cauchy–Schwarz identity for the stated wavefunction and momentum definitions. Demonstrating it does not derive quantum dynamics from the classical action. | 📐 Standard identity; synthetic examples |
| 46 | Phase-thermalization illustration | The toy estimate $\tau=\hbar/(k_BT)\approx25$ fs at 300 K assumes a reservoir. It does not derive the Born rule from a Boltzmann weight or select a measurement outcome. ([wavefunction_collapse.py](verification/nvg_wavefunction_collapse.py)) | ❓ Measurement mechanism unclosed |
| 47 | Neutrino mass from θ-seesaw | Separate scale ansatz $m_\nu=(\alpha_s/4\pi)^2v_{\rm EW}^2/f_a$ with adopted $f_a=1.07\times10^{11}$ GeV, oscillation inputs and a standard axion mass relation gives quoted scales $m_3\approx50.3$ meV and $m_\theta\approx53$ μeV. The lepton-number-violating operator/matching is not derived from the current action; comparing a mass sum with a cosmology-dependent bound is conditional. ([neutrino_seesaw.py](verification/nvg_neutrino_seesaw.py)) | ❓ Extension-specific scale ansatz |
| 48 | Hawking-formula illustration | Computes the standard $T_H=\hbar c^3/(8\pi GMk_B)$ and area entropy with a proposed stochastic-phase interpretation. No radiation spectrum or quantum-gravity completion is derived from the maintained classical action. | 📐 Synthetic illustration |
| 49 | Fine structure constant: physical-cutoff interpretation | Recomputed with $W_0 = 859$ MeV: $1/\alpha(M_Z) = 126.6$ is standard 1-loop QED running (independent of $W_0$; 2-loop hadronic terms close the gap to $127.95$). The NVG content is interpretational — the UV cutoff is the physical condensate scale and $\alpha_{\rm bare} = 1/132.8$ is inferred from the measured $1/137$, not derived ([fine_structure.py](verification/nvg_fine_structure.py)) | ⚙️ Reinterpretation (no independent prediction) |
| 50 | Antimatter as $\theta \to -\theta$ | C-conjugation and phase reversal are formal identities. The maintained baryogenesis calculation is retired because no baryon-number/CP-violating source dynamics or washout likelihood is implemented ([nvg_baryon_asymmetry.py](verification/nvg_baryon_asymmetry.py)) | 📐 Formal identity; baryogenesis retired |
| 51 | 🔥 RHIC Bell Test — entanglement death | $S_{\rm CHSH}(T > T_c = 157\text{ MeV}) \to 0$ is a future protocol with no current detector data; it is a forecast, not a confirmation ([rhic_bell_test.py](verification/nvg_rhic_bell_test.py)) | ⏳ Awaiting RHIC BES-II |
| 52 | Homochirality from QCD topology | A QCD-to-molecular PVED coupling, reaction kinetics, and independent chirality likelihood are absent; the former $>99\%$ closure is retired ([nvg_dna_chirality.py](verification/nvg_dna_chirality.py)) | ❌ Retired (missing coupling/likelihood) |
| 53 | Primordial Gravitational Waves (BICEP/Keck) | Historical Genesis dimensional estimate $r\sim(E_{\rm bounce}/E_{\rm Planck})^4\sim10^{-76}$. The reported limit $r<0.033$ is external context; without derived primordial perturbations and a likelihood the estimate neither confirms a bounce nor predicts measured B-modes. | ❓ Conditional dimensional estimate |
| 54 | Critical Horizon Mass | The quoted $M_{\rm crit}\in[0.97,1.01]\,M_\odot$ follows from the adopted $859\pm8$ MeV mass-budget input and the Hayward-family ansatz, with $M_{\rm crit}=\frac9{8\sqrt{2\pi}}M_{\rm Pl}^3/M_{\Omega,0}^2$. This is a conditional horizon threshold, distinct from the astrophysical $2$–$5\,M_\odot$ mass gap; it does not establish the formation or existence of horizonless remnants. ([nvg_mcrit_chandrasekhar.py](verification/nvg_mcrit_chandrasekhar.py)) ([nvg_mcrit_family.py](verification/nvg_mcrit_family.py)) | 🔭 Conditional metric prediction |
| 55 | Hawking Temperature Ceiling | For the stationary Hayward ansatz with $l=1.128$ km, the quoted temperature reaches $T_{\max}=3.6\times10^{-8}$ K at $M=1.29\,M_\odot$ and tends to zero at $M_{\rm crit}$. Extending this to cosmological lifetimes or an allowed $10^{10}$–$10^{17}$ g dark-matter window requires formation, transport and abundance calculations. This is not a demonstrated exemption from observational limits. ([nvg_hayward_evaporation.py](verification/nvg_hayward_evaporation.py)) ([nvg_pbh_dark_matter.py](verification/nvg_pbh_dark_matter.py)) | 🔭 Conditional stationary-metric calculation |

**5. Inner-Horizon Saturation & Area Deficit:**
Scanning the horizon roots of the Hayward metric as $M \to \infty$ shows that the inner horizon $r_{\rm in}$ asymptotes to the mass-independent vacuum length scale $l = \sqrt{3c^2 / (8\pi G \rho_c)} \approx 1.128$ km, fixed by the QCD core density $\rho_c$. The inner horizon therefore carries a fixed Bekenstein–Hawking area:
$$ S_{\rm in}(M \to \infty) \to \frac{k_B c^3 (4\pi l^2)}{4G\hbar} \approx 2.2 \times 10^{76} \text{ bits} $$
This is a property of the regular core, not a new thermodynamic reservoir: it is a mass-independent constant, negligible next to the outer-horizon entropy $S_{\rm out} \approx S_{\rm Sch} \propto M^2$. By Vieta's formulas on the dimensionless horizon cubic $z^3 - z^2 + \epsilon^2 = 0$ ($e_1 = 1$, $e_2 = 0 \Rightarrow \sum z_i^2 = 1$), the outer+inner horizon area falls short of Schwarzschild by exactly $\Delta A = 4\pi r_{\rm Sch}^2\, z_{\rm phantom}^2$, where $z_{\rm phantom} < 0$ is the third (unphysical, negative) root. This is a compact algebraic expression for the area deficit — not a physical conservation law, and unrelated to the unitarity of Hawking evaporation or the information paradox.

### 2. Theoretical interpretations and unresolved mechanisms
These entries record model interpretations and possible research directions. They do not establish solutions of the underlying physical problems.

| Area | NVG Interpretation | Impact on Physics |
|---|---|---|
| Origin of Magnetars | The quoted reconstructed correlation $R\approx0.51$ and $\sim7.4$ field factor belong to a proposed amplification model. | Reconstruction, coupling and saturation assumptions require independent tests; no resolution of the $E_{\rm rot}\sim10^{52}$ erg SNR tension is established. |
| PBH Mass Spectrum | A conditional ladder maps assumed bounce-mass growth per cycle from $10^{-14} M_\odot$ to $10^6 M_\odot$; under the corrected Tolman law the rung spacing is $\times 2$ (denser than the earlier $4^N$; [nvg_tolman_law_derivation.py](verification/nvg_tolman_law_derivation.py)). | This scale map does not supply formation, abundance or object-level JWST likelihoods; it does not establish a dark-matter or early-BH explanation. |
| JWST SMBH Seeding | A conditional PBH mass-rung growth template can be evaluated for a selected cycle. | Object-level JWST seeding is retired: no machine-readable catalogue, occupation model, or selection likelihood is present. |
| Joint Multi-Messenger Inference | Conditional/in-sample reduced $\chi^2_\nu = 0.684$ over three runtime structural rows; cooling is a calibration target and excluded. | The transition was selected using J0740, NICER, and GW170817 inputs, so this is descriptive and carries zero independent evidence weight. |
| Emergent Quantization & Duality | Madelung variables rewrite a wavefunction using amplitude and phase; the quantum potential $Q(x)$ is part of that formulation. | A representation of assumed quantum dynamics is not a derivation of the Schrödinger equation from the maintained classical action. |
| Observer Effect | Phase thermalization and topological reconnection are proposed illustrations. | No Born-rule derivation, outcome-selection law or resolution of the measurement problem is supplied. |

### 3. Consistency Checks (Null Tests)
NVG must not break General Relativity where it works reliably. These items summarize formal/process checks and declared comparisons; they do not prove empirical agreement or replace a detector likelihood.

| Physical Aspect | NVG Prediction | Observational Data |
|---|---|---|
| EOS Causality | $c_s^2=1/3$ is assigned in a CSS branch, not a universal bound on every EOS here | The causal condition $c_s^2\le1$ is checked for each specified EOS; this is not itself a NICER/LIGO likelihood |
| Gravitational Waves | $\gamma_{\rm PPN} \equiv 1$, $c_T = c$ | Cassini, GW170817: $|c_T/c - 1| < 10^{-15}$ |
| External BH Metric | A chosen exterior/regular-core metric provides a formal comparison | No independent merger-waveform likelihood establishes exact Kerr/Schwarzschild equivalence |
| Tidal Deformability | Runtime canonical $\Lambda_{1.4} = 519.4$ (conditional/in-sample) | GW170817 declared input: $\Lambda_{1.4} = 190^{+390}_{-120}$; no independent confirmation is claimed |
| Dark Energy (DESI) | **Retired mechanism.** The mass-melting derivation of $(w_0, w_a)$ was today-anchored; in the CMB-anchored frame it improves on $\Lambda$CDM by only $\Delta\chi^2 \approx 1$ against DESI DR2 while raising $S_8$ to $\approx 0.86$–$0.90$ and $\Omega_m$ to $0.35$ — no parameter region satisfies DESI and weak lensing together ([nvg_desi_s8_joint_map.py](verification/nvg_desi_s8_joint_map.py)). NVG currently predicts $w = -1$; the DESI $w_0 w_a$ preference, if confirmed, is unexplained by the model — an open problem |
| BH Shadows (EHT) | Historical $\sim10^{-70}$ estimate uses assumptions distinct from the macroscopic-core example above | Not a sourced EHT fit or a portable bound for the source-complete action |
| Lorentz Invariance | $0.0$ vacuum dispersion and birefringence | GRB 041219A / 090510 (Fermi/Swift) |
| QNM Ringdown | Historical $\sim10^{-105}$ shift estimate from a separate cutoff choice | No derived perturbation spectrum or detector likelihood establishes this number for the maintained macroscopic core |
| CMB $P(k)$ Spectrum | Assumed cutoff approaches the baseline at high multipoles | The maintained lite comparison is not the Planck likelihood and cannot establish an exact data match |
| BBN and Recombination | Historical estimates $\delta H/H\sim10^{-13}$, $\delta r_s/r_s\approx0$ | A small correction estimate does not replace nucleosynthesis/recombination calculations or validate the quoted $r_s=147.09$ Mpc |

### 4. Falsifiable Forecasts (Awaiting Verification)
The highest-risk testable forecasts of the theory. Future measurements can constrain or falsify these channels; none is a current confirmation.

| Direction | Forecasted Value / Interpretation | Experiment / Current Status |
|---|---|---|
| **CMB Anomaly $\ell<10$** | Assumed Genesis cutoff is a candidate template; cosmic variance is not eliminated | No traceable Planck likelihood establishes its origin; future data could compare specified alternatives |
| **In-medium $\rho$ line shape** | Preregistered architecture selector: fitted in-medium pole $\geq 755$ MeV $\Rightarrow$ arch. A; $715$–$755$ $\Rightarrow$ arch. B; $< 715$ $\Rightarrow$ both excluded ([HADES_PREREGISTRATION.md](verification/HADES_PREREGISTRATION.md)) | HADES in-medium analysis announced for 2026–27 |
| **Gravitational Echo** | Echo spacing $\Delta t \approx 0.022$ s ($65\,M_\odot$) with decay amplitude $A_n \propto (1 - \mathcal{T})^n$ | Searched O1–O4b open data (coherent time-slide, 89 events): **no evidence**, upper limit set (echoes $\gtrsim 0.3\times$ the GW150914 signal excluded). A naive O4 stack first showed a spurious $2.4\sigma$ excess — resolved as primary-signal (ringdown) leakage via a gap-tooth discriminant. |
| **NS Gravitational Redshift** | $z_{\rm surf}(1.4 M_\odot) \approx 0.222$ (runtime $R_{1.4}=12.550$ km; conditional chain) | STROBE-X / eXTP (future X-ray observatories) |
| **Post-merger $f_{\rm peak}$** | $f_{\rm peak} \approx 2420$ Hz from runtime $R_{1.6}\approx12.65$ km | LIGO O5 / Einstein Telescope (future detectors) |
| **Vacuum melting exponent $\beta$** | $W\sim(1-\rho/\rho_c)^{\beta}$: the $\sqrt{\;}$-law is mean-field $\beta=1/2$, but a QCD-anchored 3-D critical point gives $\beta=0.326$ (Ising) or $0.349$ (XY) — reshaping the bounce term to $(1-\rho/\rho_c)^{2\beta}$ | RHIC **BES-II** net-proton cumulant scaling near $T_c\approx157$ MeV — *existing data*. Derivation & consequences: [`NVG_MELTING_LAW_ANALYSIS.md`](NVG_MELTING_LAW_ANALYSIS.md) |

### External Verification Outreach

The repository records an outreach proposal to the **HADES Collaboration** (GSI/FAIR, Prof. Dr. J. Stroth) concerning Au+Au and Ag+Ag dielectron data; this is not a collaboration result or endorsement. Historical model values $M_\rho^*\approx621$ MeV at $2n_0$ and an integrated-template feature near $712$ MeV are not measured peak positions. A meaningful test requires the full excess line shape, fireball evolution and detector response.

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
| **I. Exact identities within definitions** | $\rho_c r_c^2=3c^2/8\pi G$, $M_1=c^3t_b/(2G)$, $T_b/M_\Omega=(30/\pi^2g_*)^{1/4}$ and $M_{\rm crit}=(9/8\sqrt{2\pi})M_{\rm Pl}^3/M_\Omega^2$ close algebraically under their definitions and adopted inputs, including $M_\Omega=859\pm8$ MeV | Algebra and units; not measurements of a bounce or field |
| **II. Conditional extension maps** | The separate $\theta$-seesaw ansatz quotes $m_\theta=53.2$ μeV, $m_{\nu_3}\approx50$ meV and $\sum m_\nu=63.3$ meV using $f_a=1.07\times10^{11}$ GeV and additional standard inputs | Shared parameter dependence is not independent evidence; no matching to the source-complete action is supplied |
| **III. Conditional compatibilities** | $2M_\Omega$ and the scalar glueball (lattice QCD), $M_{\rm crit} = 0.992\,M_\odot$ vs the absence of sub-solar horizons in GWTC, Cas A/Vela bimodality, and the runtime canonical $\Lambda_{1.4}=519.4$ comparison | declared data inputs; NS chain selected in-sample, with zero independent evidence weight (`SPEC-v2` C3) |
| **IV. Calibrations (not evidence)** | $H_0 = 72.8$, Direct Urca threshold $1.45\,M_\odot$, EOS shape parameters | one fitted input per block |
| **V. Conjectures (CONJECTURED)** | the $m_a \leftrightarrow \rho_{\rm DE}$ link via $N_e$ (see the new table row) | ansatz + shared $H_0$ calibration (reverse-solution structure, `SPEC-v2` rejection boundary item 2) |

Algebraic identities check the stated definitions; conditional maps check their chosen assumptions. Neither is empirical confirmation. The triple-hierarchy identities (`nvg_dual_hierarchy_identity.py`) also use calibrated Hubble input, so their closure is a consistency check.


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
| `nvg_sigma_mnu_corner.py` | The cornered $\theta$-seesaw prediction $\Sigma m_\nu = 59.0$ meV (minimum of normal ordering, $m_1\approx0$) vs bounds encoded in the script: Planck UL 120 (margin +61), DESI DR1 UL 72 (**margin only +13 meV**), disputed DESI DR2 $-75\pm51$ (pull 2.6$\sigma$); CMB-S4 forecast $\sigma\sim15$ meV — a $\sim4\sigma$ exclusion or conditional agreement, not identification of the $\theta$ sector; falsifier: a robust $\Sigma<45$ meV | dated-input sensitivity diagnostic |
| `nvg_katrin_mbeta.py` | Kinematic mass $m_\beta$: the PMNS-weighted observable $= 8.9$ meV (minimal NO) — correcting the earlier 29 meV (equi-weighting $m_3/\sqrt3$ is NOT the KATRIN observable); KATRIN's 200 meV design reach is a factor $\sim$20 away — honestly INSENSITIVE; falsifier: a robust $m_\beta > 20$ meV excludes the sector (IO floor 49.2 meV); two-sided test only with Project 8 ultimate ($\sigma\sim10$ meV) | registered one-sided falsifier |
| `nvg_echo_o4_observers.py` | Observer card for the four O4 echo candidates (GW230518/230615/230922/231215): echo spacing $20.0$–$23.9$ ms (remnant-mass systematic $\times0.95$ included), carrier $f_{\rm QNM} = 245$–$278$ Hz, damping $\tau = 3.3$–$3.7$ ms, 14–16 pulses per comb; search protocol consistent with the repo pipeline | observer card |

Historical conditional test proposals for separate extension sectors (not predictions established by the current source-complete action): (i) **moment of inertia of J0737A** — orbital precession of the double pulsar will measure $I$ at ~10%, discriminating stiff from soft EOS; (ii) **$m_{\beta\beta} > 10$ meV excludes the minimal $\theta$-seesaw**; (iii) **any monopole fifth-force signal at 3–4 mm excludes the $\theta$ sector**; (iv) a confirmed cosmic birefringence $\beta \neq 0$ at LiteBIRD precision pins $\gamma_{\rm topo}\Delta\theta$; (v) superradiance and tidal heating provide population tests of horizonless remnants; (vi) **a detection in the μeV band vs the meV–eV band** discriminates NVG from dark-dimension KK dark matter, and DESI $w(z) \neq -1$ at growing significance is the sharpest live falsifier of the dark sector; (vii) KATRIN/Project 8: $m_\beta = 8.9$ meV (PMNS-weighted; the earlier 29 meV corrected — equi-weighting $m_3/\sqrt3$ is not the KATRIN observable) — KATRIN is honestly insensitive (factor $\sim$20), but any robust $m_\beta > 20$ meV excludes the $\theta$-seesaw (IO floor 49.2 meV), and the two-sided test belongs to Project 8 ultimate ($\sigma\sim10$ meV); (viii) audit of the PBH ladder $M_N = 0.38\times4^N\,M_\odot$: the shared ×4 spacing is structural (horizon masses), but the peak rung $N=-21$ is calibrated to the asteroid window (formation epoch $t = 8.5\times10^{-19}$ s, $T\approx642$ TeV matches no natural scale); the closure hypothesis $\sqrt{m_{\rm Pl}M_1}$ is REFUTED (×1.3×10⁶); the peak is Hawking-stable with a $\sim10^{18}$ lifetime margin, evaporation boundary at $N=-31$.

---

## Analog optical demonstration

The historical optical exercise encoded selected NVG/VMF formula shapes as input signals and reported analog-channel measurements (γ=1.56, DR=86:1, SNR=38). These are input/output transfer comparisons, not measurements of nuclear matter or cosmological evolution.

| Test | NVG Prediction | Optical Result | Correlation |
|------|---------------|----------------|-------------|
| Meson hierarchy $\rho > K^* > \phi > J/\psi$ | $-20.0\%,\;-7.8\%,\;-2.9\%,\;-0.4\%$ | $-20.0\%,\;-8.8\%,\;-2.3\%,\;0.0\%$ | $r = 0.997$ |
| Melting curve $W(\rho)=\sqrt{1-\rho/\rho_c}$ | $\sqrt{1-x}$ vs linear | $\sqrt{1-x}$: $r=0.983$; linear: $r=0.896$ | $r = 0.983$ |
| Modified Friedmann $H^2 \propto \rho(1-\rho/\rho_c)$ | Parabola, zeros at $0$ and $\rho_c$ | Encoded central maximum and endpoint zeros reproduced | $r = 0.983$ |
| Tolman growth law | Conditional scaling check: adopting $a_t \times 2$ and $S_{\rm GH}\propto a_t^2$ gives $S_{\rm GH}\times4$; with $M\propto a_t$, $M\times2$, whereas $M\times4$ would give $S_{\rm GH}\times16$ ([nvg_tolman_law_derivation.py](verification/nvg_tolman_law_derivation.py)) | Algebraic consistency, not an optical measurement of cycles | — |

The quoted optical difference $\Delta r=0.087$ concerns two encoded shapes, $\sqrt{1-x}$ and a line. It does not select the physical EOS or derive the encoded melting law. Physical scales and parameters retain their separate adopted/calibrated provenance.

**Historical hypothesis checks:** (i) the proposed “golden angle” / $\theta\approx52.8°$ is convention-dependent (49.5°–54.4°), and an irrational winding is incompatible with the imposed integer charge $Q=1$; this does not prove a physical arrow of time; (ii) the adopted recondensation/GW template places its signal at $\mu$Hz rather than identifying NANOGrav with a bounce; the present action does not establish that bounce dynamics; (iii) the proposed de Sitter-core explanation of $S_8$ fails the repository's capacity estimate by $\sim45$ orders of magnitude.

**Scope:** this analog exercise checks selected signal shapes in its apparatus. It neither validates the physical model nor proves its complete mathematical structure; independent physical tests require appropriate data and a fully specified comparison model.

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

The following list retains historical test proposals and conditional forecasts. Each applies only to its specified extension, inputs and observational model; it is not a list of established consequences of the source-complete action.

1. **Gravitational Wave Echoes:** A reflective-core template quotes $\Delta t_{\rm echo}\approx22$ ms for $65\,M_\odot$ with $\sim16$–$34$ ms cutoff sensitivity. Echo existence and amplitude require an independently specified boundary response; they are not fixed by the adopted QCD-scale input alone.
2. **Heavy-Ion Collisions (FAIR/HADES/NICA):** The historical template quotes an instantaneous $\rho$-mass drop of about 20% at $2n_0$. A detector test requires the full in-medium line shape, fireball evolution and experimental response; this drop is not generated by the current source-complete branch.
3. **CMB Genesis Cutoff:** The historical construction stretches a $1.13$ km scale by about 53 calibrated e-folds and imposes a low-$\ell$ cutoff. It is a candidate template, not an established explanation of $\ell=2,3$ or an exclusion of cosmic variance.
4. **Neutron Stars:** The runtime canonical chain reaches $M_{\max}=2.048 M_\odot$ with a CSS branch at $c_s^2=1/3$; the transition is selected in-sample and is not independent evidence.
5. **Mass-budget input:** Revising the adopted $M_{\Omega,0}$ interval $851$–$867$ MeV changes dependent dimensional scales. Testing the identification of this budget with a macroscopic $W_0$ requires explicit operator matching and input covariance.
6. **EHT comparison:** Historical estimates $\sim10^{-35}$ at the horizon and $\sim10^{-70}$ at the photon ring belong to separate cutoff assumptions; they do not establish exact agreement for the macroscopic Hayward core or a detector-level falsifier for the source-complete action.
7. **Tolman cycle assignment:** Index 77 is an input. The shared 76-doubling calculation yields the conditional scale $\pi GM/c^3\approx14.1592$ Byr, not a measured age or remaining lifetime; recurrent-cycle dynamics is not derived.
8. **Tidal Deformability (GW170817):** The runtime canonical chain computes $\Lambda_{1.4}=519.4$; the comparison with the declared GW170817 interval $[70,720]$ is conditional/in-sample, not an independent confirmation.
9. **Multi-Meson Spectroscopy:** The historical $2n_0$ template uses $\rho,\omega$ (-20.0%), $K^*$ (-7.8%), $\phi$ (-2.9%) and $J/\psi$ (-0.4%). These are branch-dependent model shifts for a line-shape study, not a measured hierarchy or outputs of the current one-component EOS.
10. **Quantitative CMB template:** The imposed cutoff tends to a baseline ratio of 1.000 above $k\sim10^{-3}$ Mpc$^{-1}$ and suppresses power below about $3\times10^{-4}$ Mpc$^{-1}$. This by-construction shape does not establish an exact Planck-data match or its physical origin.
11. **Multi-Mass PBH Spectrum (Dark Matter):** The cycle ladder supplies a conditional mass-rung template from the asteroid window through heavy candidates; object-level JWST seeding is retired pending a machine-readable catalogue, occupation model, and selection likelihood.
12. **GW Echo Template:** Parameterized echo train with decaying amplitude ($R_{\rm core}^n$) and alternating phase — ready-to-use template for LIGO matched-filtering.
13. **NS cooling sensitivity:** The $1.45M_\odot$ threshold is calibrated; the quoted $10^{33}$ versus $10^{31}$ erg/s regimes are illustrative. Composition, envelope and heating inputs matter, and no independent thermal-population likelihood makes one hot object a complete EOS falsifier.
14. **Gravitational Redshift and f_peak:** Runtime canonical calculations give $z_{surf} \approx 0.222$ at $1.4 M_\odot$ and $f_{peak} \approx 2.42$ kHz from $R_{1.6}\approx12.65$ km; both are forward calculations, with no observed target yet.
15. **Cycles and Genesis Robustness:** The closed form $n = N_e/\ln 2 = 76.58 \pm 0.013$ (anchored $H_0 = 72.8$) lands exactly inside integer cycle 77 of the Tolman ladder: the interval $N_e \in [76\ln2, 77\ln2] = [52.68, 53.38]$ (`verification/nvg_hubble_tension.py`). Audit: the previously quoted "77.2" is $\log_4(2.6\times10^{122}/10^{76}) = 77.09$ built on rounded E&L entropy anchors, and "±0.3" is the cycle-interval half-width $\ln2/2$, not the QCD sensitivity (true value ±0.013) (`verification/nvg_dual_hierarchy_identity.py`).
16. **EM-sector sensitivity:** $\epsilon_{\rm eff}=0.135\epsilon_0$ is a conditional value in a phenomenological melting/EM ansatz. It is not a calculated material coefficient of the current source-complete branch.
17. **Lorentz-symmetry check:** The stated vacuum ansatz introduces no explicit Lorentz-violating photon term. A zero coefficient in that ansatz is a formal check, not a new fit to GRB timing or polarization data.
18. **QNM scope:** The historical $\sim10^{-105}$ frequency-shift estimate assumes a separate Planck-scale cutoff. A perturbation calculation for the macroscopic regular-core geometry is needed before claiming detector indistinguishability.
19. **Moment of inertia of J0737A:** The linked Hartle calculation is a dated alternate calibration, not the maintained canonical chain and carries zero independent evidence weight; future orbital precession measurements remain the discriminating test (`verification/nvg_moment_of_inertia_j0737.py`).
20. **0νββ null test:** The minimal $\theta$-seesaw predicts $m_{\beta\beta} \in [0, 6.4]$ meV; a confirmed signal with $m_{\beta\beta} > 10$ meV excludes the sector (`verification/nvg_neutrinoless_dbeta.py`).
21. **Fifth force at 3.7 mm:** The monopole channel of the $\theta$ mode is identically zero (pseudoscalar); any unpolarized Yukawa signal at 3–4 mm falsifies the sector (`verification/nvg_theta_fifth_force.py`).
22. **Cosmic birefringence:** Static branch $\beta = 0$; a detection of $\beta \neq 0$ at LiteBIRD precision pins the combination $\gamma_{\rm topo}\Delta\theta$ ($0.033°$ per unit) (`verification/nvg_cmb_birefringence.py`).
23. **$\theta$-mode superradiance:** Kerr band $M_{\rm BH} \in [5\times10^{-7}, 1.3\times10^{-6}]\,M_\odot$; a spinning horizon PBH inside the band excludes the horizonless interpretation of the corresponding remnants (`verification/nvg_theta_superradiance.py`).
24. **Tidal heating:** Horizonless remnants predict a zero absorption-phase contribution in EMRIs vs ~32 cycles for a Kerr BH (LISA) (`verification/nvg_tidal_heating_null.py`).
25. **Dark-sector band discrimination:** The NVG bands (8–53 μeV) and the dark-dimension KK tower band (2 meV–1 eV at $R\in[0.2,100]$ μm) are disjoint — a detection in either band discriminates the programs; live falsifiers: DESI $w(z)\neq-1$ and the kinematic neutrino mass — 2026-08 correction: the KATRIN observable is $m_\beta = \sqrt{\sum|U_{ei}|^2m_i^2} = 8.9$ meV, not 29 meV (equi-weighting $m_3/\sqrt3$); registered falsifier $m_\beta > 20$ meV (`verification/nvg_de_axion_crosscheck.py`).
26. **Quad-hierarchy identity:** $\rho_c r_c^2 = \rho_{\rm crit}R_{H0}^2 = 3c^2/8\pi G$ → four hierarchies are powers of the single number $e^{N_e} = R_{H0}/r_c$: density $\rho_c/\rho_{\rm DE} = e^{2N_e}/\Omega_{\rm DE} = 1.85\times10^{46}$, entropy $S_{\rm now}/S_{\rm gen} = e^{2N_e} = 4^n$, mass $M_U = M_1\times 2^n = 4.30\times10^{22}\,M_\odot$ (the Hubble-sphere mass) and time $t_{H0}/t_b = e^{N_e} = 1.13\times10^{23}$ (precision $2\times10^{-15}$); the entropy-per-baryon ratio honestly does not close (area law vs volume count). Status: closed identity with a calibrated $H_0$ — consistency, not evidence (`verification/nvg_dual_hierarchy_identity.py`).
27. **Dimensional mass-scale diagnostic:** $M_{\rm Pl}^3/M_\Omega^2 = 2.211^{+0.042}_{-0.041}\,M_\odot$ is a separate dimensional bound diagnostic from the single anchor $M_\Omega$, not the runtime canonical TOV result. Its live-data comparison and falsifier are documented in `verification/nvg_ns_mass_bound.py`; this bound and all alternate fork-B outputs carry their own assumptions and zero independent evidence weight relative to the canonical chain.
28. **The cornered neutrino-mass sum:** $\Sigma m_\nu = 59.0$ meV — the minimum of normal ordering at $m_1\approx0$, rigidly fixed by the oscillation splittings; DESI DR1 + CMB + SN (UL 72 meV) leaves a margin of only +13 meV; the disputed negative DESI DR2 posterior ($-75\pm51$ meV) yields a $2.6\sigma$ pull; CMB-S4 ($\sigma\sim15$ meV) will deliver a two-sided verdict: a $\sim4\sigma$ exclusion if the central stays near zero, or conditional agreement if it lands at 59 meV; agreement with this standard normal-ordering limit would not identify an NVG mechanism. Falsifier: a robust $\Sigma m_\nu < 45$ meV under a specified physical cosmological likelihood excludes the minimal $\theta$-seesaw (`verification/nvg_sigma_mnu_corner.py`).
29. **Kinematic neutrino-mass correction and one-sided falsifier:** the KATRIN observable $m_\beta^2 = \sum_i|U_{ei}|^2m_i^2 = 8.9$ meV (minimal NO; 9.0 meV for the anchor spectrum) — earlier "29 meV" citations were the equi-weighting $m_3/\sqrt3$, not the observable; KATRIN (200 meV design) is honestly insensitive, the two-sided test is Project 8 ultimate ($\sigma\sim10$ meV); registered falsifier: any robust $m_\beta > 20$ meV excludes the sector, since the inverted-ordering floor is 49.2 meV; the consistency triangle $\Sigma m_\nu = 59$ meV / $m_\beta = 8.9$ meV / $m_{\beta\beta}\le 6.4$ meV from one anchor is rigidly closed (`verification/nvg_katrin_mbeta.py`).
30. **O4 echo observer card:** for the four ROADMAP candidates (GW230518/230615/230922/231215, $M_{\rm tot} = 61.8$–$70.2\,M_\odot$) the predicted echo spacing is $\Delta t_{\rm echo} = 20.0$–$23.9$ ms (remnant-mass systematic $\times0.95$ included), carrier $f_{\rm QNM} = 245$–$278$ Hz, damping $\tau = 3.3$–$3.7$ ms, 14–16 pulses per comb at $R_{\rm eff} = 0.95$; the protocol (search window, bandpass, delay scan $\times0.73$–$\times1.55$, coherent statistic vs time-slide background) matches the repo pipeline; falsifier: a null deep stack of the full O4 catalog at the tabulated delays drives $R_{\rm eff}\to0$ and falsifies the reflective-core picture of the remnant (`verification/nvg_echo_o4_observers.py`).

---

## Speculative Directions & Future Tech

### 1. Proposed macroscopic phase correlations (QCD to quantum optics)
Historical suggestions connect macroscopic oscillators or atomic clocks through a coherent condensate phase. The maintained action supplies neither such oscillators nor a measurable coupling, nonlocal correlation law or detector response. A testable proposal requires those missing ingredients; no predicted Bell anomaly is established.

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
