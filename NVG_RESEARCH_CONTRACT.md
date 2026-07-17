# NVG Data-Only Validation — Research Contract (`SPEC-v1`)

Compiled from the Research-Contract Prompt Kit (scope freeze → completion contract →
rejection boundary → search portfolio → adversarial audit → honest terminal state),
instantiated for the Null-Vector Gravity / VMF framework. The execution report
(`NVG_CONTRACT_EXECUTION.md`) is maintained locally and not tracked in this
repository.

---

## 0. Investigator role and truthfulness

Act as lead investigator and hostile reviewer simultaneously. Correctness is never
inferred from plausibility, effort, or authority. Exploratory claims stay separate
from confirmatory ones. No silent changes of model, target, metric, or dataset: any
change creates `SPEC-v2` and is logged. This contract was executed by a single agent
building on the repository's audited verification chain; no parallel independent
agents were used, and none are claimed.

## 1. Scope freeze — `SPEC-v1`

### 1.1 Primary task

> **Determine the strongest honest epistemic status of the NVG framework using only
> already-collected, publicly available data — no new experiments — and either
> (a) validate it by the bar of §1.2, (b) falsify it by a preregistered kill
> condition, or (c) return the audited frontier: exactly what existing data can and
> cannot decide.**

Task type: `MIXED` (empirical hypothesis testing + provenance audit).

### 1.2 Operationalization of "prove the theory"

A physical theory is never proved deductively; the strongest achievable closure from
data is **VALIDATED**, defined as: there exists at least one claim `C` such that

- **C1 (derivation):** `C` follows from the frozen anchors (§1.4) with no parameter
  fitted to the confirming dataset — verifiable from code and git history;
- **C2 (significance):** `C` is confirmed in public data at ≥3σ after
  look-elsewhere accounting;
- **C3 (discrimination):** the baseline (GR + ΛCDM + SM, §3) does not predict `C`
  or predicts the opposite;
- **C4 (reproducibility):** the confirmation is reproducible by an independent
  party from public data and the repository scripts.

**FALSIFIED** is defined as: any preregistered kill condition of
`verification/nvg_falsifier_dashboard.py` is met by published data.

Quantifier order, exactly: *there exists* a claim `C` derived from the frozen
anchors *such that for* the already-public datasets, the post-trials significance
against the baseline exceeds 3σ. (Not: "for every observable NVG is compatible" —
compatibility is necessary but never sufficient.)

### 1.3 Information and resource model

Allowed: GWOSC open strain and GWTC catalogs (O1–O4b); published NICER/XMM radii;
Planck 2018 likelihood data; DESI DR2 BAO; NANOGrav 15yr; published lattice-QCD
results; published HADES/NA60/STAR spectra; JWST photometric catalogs; pulsar
timing; the repository's verification scripts. All analyses must run on commodity
hardware and trace claim→script→source.

Forbidden: new experiments or observations; private data; posterior tuning of
anchors; any confirming dataset that entered the derivation of the tested claim
(provenance audit is mandatory, §8).

### 1.4 Frozen anchors

`M_Ω = 859 ± 8` MeV (lattice σ-terms); `ρ_c = M_Ω⁴/(ℏc)³ = 7.09×10⁴` MeV/fm³;
Hayward length `l = 1.128` km; `M_crit = (9/8√(2π)) M_Pl³/M_Ω² = 0.99 M_⊙`;
topological winding `Q = 1`; Tolman mass law ×2 per cycle; canonical fork-B EOS
(K = 240 MeV, L = 55 MeV, `n_tr = 1.6 n₀`, `cs²_q = 0.50` — the three CSS
transition parameters are FITTED and labeled as such; claims built on them cannot
count the fitted observables toward C1). Two canon variants exist (causal
`cs²=1/3` vs stiff `cs²=0.50`); claims must state which is used.

### 1.5 Scope-change rule

`SPEC-v1` is frozen. Contradictions or missing definitions are recorded with the
smallest repair as `SPEC-v2`; claims under different specs are never merged.

## 2. Completion contract

### 2.1 Acceptance matrix

| ID | Obligation | Pass/fail test | Required evidence |
|---|---|---|---|
| A1 | Provenance (C1) | Confirming dataset absent from derivation chain; no reverse-solved constants | git history + script inputs |
| A2 | Significance (C2) | ≥3σ post-LEE, computed, not asserted | script output + trials accounting |
| A3 | Discrimination (C3) | Baseline prediction stated and differs | side-by-side computation |
| A4 | Reproducibility (C4) | Public data + repo scripts suffice | run instructions, versions |
| A5 | Full scope | Claim not restricted to a convenient subclass post hoc | selection rules frozen before evaluation |
| A6 | Audit survival | §8 hostile checks pass on the frozen candidate | audit log |

### 2.2 Claim-status vocabulary

`PROVED` (mathematical result surviving audit) · `REPRODUCED` (independent rerun of
external result) · `EMPIRICALLY SUPPORTED` (prespecified test passed with stated
uncertainty) · `CONSISTENT-NULL` (data compatible; baseline predicts the same —
carries no validation weight) · `CONJECTURED` · `OPEN` · `REFUTED/RETRACTED`.

## 3. Verified baseline and gap

Baseline hypothesis: GR + ΛCDM + SM. It predicts: clean Kerr ringdown, no echoes,
no sub-solar horizon requirement, w = −1, standard PBH evaporation, no in-medium
ρ-pole shift of NVG's specific form, S_CHSH = 2√2 at any temperature.

Prior attempts (audited 2026-07-03 … 2026-07-17; each row names the first
false/unsupported step — these are the exact traps for this problem):

| Attempt | What was actually established | First false/unsupported step |
|---|---|---|
| H₀ = 72.8 "prediction" | Circular: N_e = 53.08 computed from target H₀ | anchor fitted to confirming dataset (fails A1) |
| T_c = 157.3 "0.33σ hit" | T_c is adopted FROM HotQCD | lattice compared with itself (fails A1) |
| S8 melting relief | Frame artifact; CMB-anchored frame worsens to 7σ | today-anchored growth frame |
| NANOGrav comb amplitude | Derived (α, β/H) close the PTA band; peak moves to µHz | literature-generic β/H assumed, not derived |
| SGR QPO ladder | "Standard frequencies" manufactured as observed/√0.135 | reverse-solved comparison set |
| M_max = 2.25 / R = 12.0 headline | Hardcoded; true computed M_max was 1.79–2.07 | EOS table truncation artifact |
| Echo candidate GW200210 (3.99σ local) | Single-detector; fails coherent time-slide (p = 0.89) | LEE + coherence ignored |
| O4 echo excess 2.4σ | Primary-signal leakage; residual-subtracted p = 0.77 | template response to ringdown scored as echo |
| Entropy f_S = 2.545 | Reverse-solved from observed 10⁸⁸ | dressed reverse-engineering |
| Phantom-horizon entropy | Vieta identity restated; "resolves information paradox" | numerology + overclaim |

The exact gap: **every channel capable of ≥3σ discrimination either has no released
data yet (HADES in-medium line shape, RHIC BES-II Bell protocol, DESI DR3, µHz GW),
or has returned a null/adverse result (echoes, DESI DR2 vs w = −1, NICER compact
radii). The surviving positive content is sub-2σ or non-discriminating.**

## 4. Results that do not count

Beyond the kit's generic list, the following NVG-specific patterns are rejected
(each has already occurred in this project at least once):

1. Postdiction of an adopted input (T_c-class): comparing a number with its source.
2. Circular anchors (H₀/N_e-class): the confirming value inside the derivation.
3. Calibrations relabeled as predictions (Ω_DM "Locked", S8 0.078, QPO √0.135).
4. **Consistency-with-null presented as support**: echo non-detection, WIMP nulls,
   EHT/QNM nulls — the baseline predicts the same null (fails C3). Status ceiling:
   `CONSISTENT-NULL`.
5. Sub-3σ pulls marketed as hits (Λ̃ = 313 at +0.05σ is a fine consistency point,
   but its block shares three fitted CSS parameters → one effective test, not a
   discriminating confirmation).
6. Single-detector GW candidates; scans without trials factors.
7. Reverse-solved constants presented as derivations (f_S-class).
8. Scale coincidences (0.3819 vs 1/φ²; M_crit ≈ 1 M_⊙ is generic gravity+QCD —
   README row 54 already concedes it does not discriminate NVG).
9. Internal theorems (baryogenesis no-go, g = 2 chain, temperature ceiling) as
   *data* validation: they are `PROVED`-class structure, not observations.
10. "Explains X qualitatively" without a computed likelihood against the baseline.

## 5. Search portfolio (data-only routes)

| Route | Mechanism | Load-bearing test | Earliest falsifier | Status |
|---|---|---|---|---|
| R1 GW echoes | Macroscopic Δt ≈ 22 ms (M/65) comb from QCD-anchored core | Coherent comb in O1–O4 open strain | Time-slide null | **EXECUTED — NULL** (241 events, Fisher p = 0.77, ρ₉₀ ≈ 3/det) |
| R2 NS structure | Fork-B EOS vs NICER/J0740/GW170817 | Joint χ² of 7 constraints | R_1.4 < 12.0 confirmed | **EXECUTED** — stiff 1.64/dof, causal 3.95/dof; adverse trend (J0614 +2.2σ) |
| R3 Sub-solar horizon gap | M < M_crit horizonless | Any confirmed sub-solar BH with horizon | SSM detection | **EXECUTED — CONSISTENT-NULL** (282 GWTC events; none in band) |
| R4 QCD spectrum | Glueball 2M_Ω = 1718 MeV; η′ = θ | Lattice 1700 ± 100 | Lattice ID excluding scalar glueball there | **EXECUTED** — postdiction, lattice-only, ID unsettled (fails C2/C3) |
| R5 CMB IR cutoff | Causal k³ cutoff at k_c = 1/R_H | Planck low-ℓ TT/TE Δχ² | Steep-cutoff preference reversing | **EXECUTED** — Δχ² ≈ +1.8 (TT) +0.75 (TE): sub-σ, cannot reach 3σ |
| R6 Dark energy | w = −1 identically (nothing rolls) | DESI BAO w₀wₐ posterior | DR3 confirming evolving DE | **EXECUTED — ADVERSE** (DR2 disfavors w = −1 at ~4.5σ; binary at DR3) |
| R7 PTA/µHz comb | Derived recondensation bump 2×10⁻⁹ @ 25 µHz, log-2 comb | NANOGrav 15yr band; µAres-class floor | µHz mission null at 10⁻¹³ | **EXECUTED** — NANOGrav window closed by derived (α, β/H); no µHz data exist |
| R8 Evaporation shutdown | T_max = 3.6×10⁻⁸ K; horizonless sub-M_crit | PBH burst searches (HAWC/Fermi) | One confirmed burst | **EXECUTED — CONSISTENT-NULL** (all burst searches null; baseline also allows null) |

Kill/reopen rules: a route killed by a null stays `CONSISTENT-NULL` for validation
but remains a live falsification channel; a route blocked by missing data (R6 DR3,
HADES, BES-II) reopens automatically on data release — no new experiments needed,
only analyses of data already taken or scheduled surveys.

## 6. Concrete work-product rule

Every route must point to an executable script in `verification/` and its inputs;
"the search found nothing" requires the search artifact (background distributions,
χ² tables, scan grids), not prose.

## 7. Task-specific obligations

Empirical routes follow the frozen pipelines (time-slide backgrounds with
coherence decomposition for GW; one-sided mass constraints and asymmetric errors
for NS; trials-corrected scans everywhere). Provenance obligations: every claimed
derivation must survive a git-history check for retrofitting (the H₀ lesson).

## 8. Adversarial audit

Universal checks per the kit, plus mandatory NVG-specific hostile tests:

1. **Provenance attack:** attempt to show the confirming number entered the
   derivation (git archaeology; reverse-solve test).
2. **Baseline attack:** compute what ΛCDM/GR/SM predicts for the same observable;
   if identical, demote to `CONSISTENT-NULL`.
3. **Trials attack:** recount look-elsewhere over the full scanned space, not the
   reported slice.
4. **Block-correlation attack:** pulls sharing fitted parameters collapse to one
   effective test (`nvg_global_significance.py` accounting).
5. **Best-case-removal:** the conclusion must survive deletion of the single most
   favorable row.

## 9. Honest terminal states

The execution report must begin with exactly one of: **COMPLETE RESOLUTION** /
**EMPIRICALLY SUPPORTED RESULT** / **AUDITED FRONTIER — INCOMPLETE**, and may
declare VALIDATED or FALSIFIED only by the §1.2 bars.
