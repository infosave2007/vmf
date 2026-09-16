# NVG/VMF Research Roadmap

This document outlines the short-term, medium-term, and long-term milestones for the Null-Vector Gravity (NVG) and Vacuum Mass Fraction (VMF) cosmological and astrophysical framework.

---

## 📅 Timeline & Milestones

### 1. Short-Term (2026) — Active Observational Cross-Checks
* **LIGO O4 Gravitational Wave Echoes:**
  * Implement matched-filtering search on public O4 strain data using regular Kerr-Hayward core waveform templates.
  * Analyze target candidates: GW230518, GW230615, GW230922, and GW231215.
* **FRB Mass-Stability Statistics:**
  * Perform complete population checks of repeating FRBs vs estimated magnetar masses using CHIME Catalog 1 and Catalog 2.
* **Structure Growth & S8 Tension:**
  * Refine the Jeans-like small-scale growth suppression models with upcoming weak lensing surveys (DES Y6, Euclid).
* **Vacuum Melting Exponent $\beta$ (RHIC BES-II, *existing data*):**
  * Test $W\sim(1-\rho/\rho_c)^\beta$: a two-hypothesis data-collapse of BES-II net-proton cumulants ($C_4/C_2$, $C_3/C_2$) near $T_c\approx157$ MeV selects mean-field $\beta=1/2$ (the $\sqrt{\;}$-law) vs 3-D Ising $\beta=0.326$; $\beta$ also reshapes the bounce term $(1-\rho/\rho_c)^{2\beta}$. Derivation & identifiability study: [NVG_MELTING_LAW_ANALYSIS.md](NVG_MELTING_LAW_ANALYSIS.md).
* **Dark-Sector Band Discrimination (dark-dimension cross-check):**
  * NVG pseudoscalar bands ($m_\theta = 53.2$ μeV, topological axion 8.4 μeV) are disjoint from the dark-dimension KK tower window (2 meV–1 eV at $R \in [0.2, 100]$ μm) — any haloscope detection discriminates the two programs. Registered live falsifiers: DESI $w(z) \neq -1$ at growing significance (NVG predicts $w=-1$ exactly) and KATRIN around $m_\beta \approx 29$ meV (two-sided $\theta$-seesaw test). Diagnostic: `verification/nvg_de_axion_crosscheck.py` (CONJECTURED status under `SPEC-v2`).

### 2. Medium-Term (2027–2030) — Lab Tests & Next-Gen Surveys
* **HADES Collaboration Cross-Check:**
  * Compare the parameter-free ρ-meson mass shift prediction ($M_\rho^* \approx 596$ MeV at $2n_0$) against GSI/FAIR dielectron spectra.
* **JWST Early SMBH Seeds:**
  * Compare cycle accretion histories ($N=10$ seeds) with spectroscopic data of newly discovered high-z massive black holes.

### 3. Long-Term (2030+) — Space Missions & Numerical GR
* **LiteBIRD CMB B-mode Polarization (2032):**
  * Test the predicted B-mode tensor power spectrum cutoff ($r < 0.001$ at $\ell < 10$) using LiteBIRD polarization data.
* **Numerical General Relativity:**
  * Build exact numerical GR solvers for de Sitter core regularized black holes and the Genesis cosmological bounce.

## Dependency-ordered development plan

The [current development plan](THEORY_DEVELOPMENT_PLAN_RU.md) defines the
execution priorities, prerequisite calculations, independent validation and
stop conditions. The topic list above is a research backlog, not a set of
validated predictions or committed delivery dates; its proposed signals must
pass the action-to-observable checks in the development plan before use.

The first implemented step is the [live foundation/identifiability audit](NVG_FOUNDATION_IDENTIFIABILITY_RU.md),
with a [provisional physical-validation protocol](NVG_FIRST_VALIDATION_PROTOCOL_RU.md).
It preserves the original parameters and reports mathematical controls, not
an empirical confirmation or completion of the research backlog.


The next implemented step is the [static in-medium spatial response](NVG_SPATIAL_IDENTIFIABILITY_RU.md):
72 live local-TF rows distinguish a bulk-degenerate scale family at
nondegenerate finite wave numbers. This is conditional mathematical evidence,
not a finite-nucleus or empirical test. The remaining action-to-observable
bridge must include nonlocal/dynamic response and experimental uncertainties.


The [finite-momentum Dirac/Hartree response](NVG_NONLOCAL_RESPONSE_RU.md)
now replaces the local fermion kernel with an explicitly subtracted static
medium kernel, using unchanged parameters. Its 54 live rows quantify the TF
approximation shift and preserve conditional scale discrimination. The
frequency-dependent/finite-nucleus observable and independent vacuum matching
remain open; this is not full vacuum RPA or an empirical validation.

The [retarded finite-momentum response](NVG_RETARDED_RESPONSE_RU.md) now
closes the finite-frequency step for the same subtracted-medium Hartree
prescription, with no parameter refit. The full longitudinal scalar/vector
block yields 48 complex samples and four resolved Q4 poles across the two
wave numbers and two scales. At scale 1 their energies are 46.07243 and
92.37040 MeV; the eight W8 cases have no resolved pole in the bounded
scanned intervals, not a global absence result. The equivalent long-wave
condition `K > (3 kF²/EF)(3 mu/EF−2)` connects compressibility and current
response without establishing a universal physical law.

The remaining observable bridge is a specific experimental operator with
finite-size effects, controlled approximation errors, and independent data.
This calculation is neither full renormalized vacuum RPA nor empirical
validation; the original Q4 saturation failure is unchanged.
