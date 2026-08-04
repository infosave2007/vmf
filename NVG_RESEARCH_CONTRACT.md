# NVG Data-Only Validation — Research Contract (`SPEC-v2`)

`SPEC-v2` is the audit-repair version of the data-only validation contract for
Null-Vector Gravity / Vacuum Mass Fraction (NVG/VMF). It supersedes `SPEC-v1`
at base revision `ab4ac516d840bedf272770fb5ca7a901c2d8dab1` and is executed in
the local report `NVG_CONTRACT_EXECUTION.md`.

This document is **not a retrospective preregistration**. It freezes what can be
audited now, records conflicts found in the repository, and prevents those
conflicts from being counted as evidence. Any new claim, model variant, dataset,
selection, statistic, or threshold requires `SPEC-v3` before looking at the new
result.

## Change log from `SPEC-v1`

| Change | Reason |
|---|---|
| Freeze the repository revision, data cutoff, commands, hashes, and execution grades | `SPEC-v1` did not define a reproducible snapshot |
| Replace the unbounded “there exists a claim” search with claims `C01`–`C08` | Otherwise the trials factor is undefined |
| Mark the EOS route ineligible until one complete canon is selected | Repository scripts use incompatible parameter/output vectors |
| Separate public, unpublished, and future data | Pending HADES/BES-II analyses and DESI DR3 cannot validate a current data-only claim |
| Demote `nvg_falsifier_dashboard.py` to a diagnostic | Six entries return an unconditional `True`; its `8/8 ALIVE` is not a data-driven falsification audit |
| Add route→script→input→artifact mapping | Required by §6 but absent from `SPEC-v1` |
| Correct the DESI evidence statement | The DESI paper reports `3.1σ` for DESI BAO+CMB and `2.8–4.2σ` after adding different SN samples, not one universal `4.5σ` result |
| Record actual execution topology | Parallel repository/contract/test audits are not independent scientific replications |

## 0. Investigator role and truthfulness

Act as lead investigator and hostile reviewer simultaneously. Correctness is
never inferred from plausibility, effort, a zero exit code, or authority.
Exploratory, fitted, postdictive, and confirmatory results remain separate.

The `SPEC-v2` execution used one coordinating Codex agent plus parallel subagents
for repository mapping, contract review, and test auditing. They shared the same
repository and are **not** statistically independent reproductions. No external
independent replication is claimed.

## 1. Scope freeze

### 1.1 Primary task

> Determine the strongest honest epistemic status of NVG using already-collected,
> publicly available data only, and return exactly one of: validation by §1.2,
> falsification by a registered kill predicate, or the audited frontier of what
> the frozen evidence can and cannot decide.

Task type: `MIXED` (empirical testing + provenance/reproducibility audit).

### 1.2 Decision rules

`VALIDATED` requires at least one claim from the frozen registry `C01`–`C08` to
pass all of the following:

- **C1 — derivation:** the prediction and its uncertainty follow from the frozen
  model without fitting to the confirming dataset;
- **C2 — significance:** the registered statistic reaches at least `3σ` after
  the registered look-elsewhere correction;
- **C3 — discrimination:** the registered baseline predicts a materially
  different distribution for the same observable and nuisance model;
- **C4 — reproducibility:** a clean checkout plus public inputs and declared
  dependencies reproduces the result.

`FALSIFIED` requires a frozen, machine-evaluable kill predicate to be triggered
by its named public dataset at the registered confidence threshold. A prose
forecast, hardcoded boolean, central-value comparison, or unpublished result
cannot trigger falsification.

If neither rule is met, the only permitted terminal state is
`AUDITED FRONTIER — INCOMPLETE`. A physical theory is never labelled “proved” by
compatibility alone.

### 1.3 Snapshot and resource model

| Field | Frozen value |
|---|---|
| Base revision | `ab4ac516d840bedf272770fb5ca7a901c2d8dab1` |
| Dataset/publication cutoff | `2026-07-17` |
| Execution platform | macOS arm64; system Python `3.9.6` |
| Executed core packages | NumPy `1.26.4`; SciPy `1.13.1`; CAMB `1.6.0`; Matplotlib `3.9.4`; PyCBC `2.9.0` |
| Repository-declared environment | Python `3.10` in CI; unpinned `numpy`, `scipy`, `camb` in `requirements.txt` |
| Hardware/network | Commodity hardware; public network data allowed; no private data or new observations |

The environment mismatch and incomplete dependency declaration are
reproducibility failures, not harmless metadata differences.

### 1.4 Frozen theoretical anchors and unresolved model choice

The following theory-side values are frozen for this execution:

- `M_Ω = 859 ± 8 MeV`;
- `ρ_c = M_Ω⁴/(ℏc)³ = 7.09×10⁴ MeV/fm³`;
- Hayward length `l = 1.128 km`;
- `M_crit = 0.9928 M_⊙`, with propagated band `[0.9746, 1.0116] M_⊙`;
- topological winding `Q = 1`;
- corrected Tolman mass growth `×2` per cycle.

The dense-matter EOS is **not frozen successfully**. At least three incompatible
vectors are used as if canonical:

| Variant/source | Model information actually available | Hardcoded/reported outputs |
|---|---|---|
| causal fork-B (`nvg_fork_b_full_chain.py`) | `n_tr=1.6 n₀`, `Δε=0.25 ε_tr`, `cs²_q=1/3`; fitted to observations | `M_max=1.89`, `R_1.4=13.11`, `Λ_1.4=393` |
| stiff audit (`nvg_ns_nicer_joint_audit.py`) | only `cs²_q=0.50` is named; the complete generating parameter vector is not encoded in the audit | `M_max=2.05`, `R_1.4=12.55`, `Λ_1.4=519` |
| README/global-significance chain | three CSS parameters are described as fitted, but numbers differ across files | `M_max=2.07`, `R_1.4=12.49`, `Λ_1.4=253`, `Λ̃=313` |

No variant is silently promoted. `C02/R2` is therefore ineligible for validation
or falsification under `SPEC-v2`; its reruns are sensitivity diagnostics only.

### 1.5 Data classes

- **Public and eligible now:** GWOSC O1–O4b/GWTC-5.0, published NICER/XMM
  results, Planck 2018 products, DESI DR2 cosmology results, NANOGrav 15-year
  products, published lattice-QCD and PBH-search results.
- **Taken but target result not public:** the proposed HADES in-medium
  line-shape selector and the proposed RHIC BES-II Bell-like analysis. These are
  watch items, not current evidence.
- **Future release:** DESI DR3 and future NICER/GW catalogs.
- **Future instrument:** a µHz GW mission at the stated sensitivity.

Private data, posterior anchor tuning, and a confirming dataset that entered the
prediction derivation are forbidden.

## 2. Completion contract

### 2.1 Acceptance matrix

| ID | Obligation | Pass test | Required evidence |
|---|---|---|---|
| A1 | Provenance / C1 | confirming data absent from derivation; no reverse-solved constant | timestamped derivation, git/DOI history, explicit fitted parameters |
| A2 | Significance / C2 | `≥3σ` global for the registered statistic | local statistic, full scan domain, trials calculation |
| A3 | Discrimination / C3 | baseline likelihood for the same observable is materially worse | side-by-side likelihood with matched nuisance treatment |
| A4 | Reproducibility / C4 | clean environment reproduces the result from public inputs | pinned dependencies, commands, hashes, immutable outputs |
| A5 | Full scope | no post-result subclass or dataset change | frozen selection and claim registry |
| A6 | Hostile audit | provenance, baseline, trials, correlation, and best-case-removal attacks survive | audit log |

Execution grades are `PASS`, `FAIL`, `NOT RUN`, and `NOT TESTABLE`. `NOT RUN` and
`NOT TESTABLE` never count as passes.

### 2.2 Epistemic vocabulary

One primary status is assigned per claim:

`PROVED` · `REPRODUCED` · `EMPIRICALLY SUPPORTED` · `CONSISTENT-NULL` ·
`POSTDICTION` · `CONJECTURED` · `OPEN` · `REFUTED/RETRACTED`.

Separate flags may record `adverse`, `fitted`, `subthreshold`, `spec_conflict`,
`data_not_public`, or `baseline_shared`. Composite invented statuses are forbidden.

### 2.3 Required work products

1. This frozen contract and its change log.
2. A route registry with exact scripts, inputs, outputs, and commands (§5).
3. Input/artifact hashes (§6).
4. An execution report beginning literally with one allowed terminal state.
5. A per-claim A1–A6 matrix and hostile-audit findings.
6. A record of what was rerun, recomputed from an artifact, inspected only, or
   not run.

## 3. Baselines and frozen claim registry

There is no single one-line baseline for every domain. Each claim uses the
domain baseline named below. In particular, `2√2` is the Tsirelson upper bound,
not the Standard-Model prediction at every temperature and measurement setting.

| Claim | Route | Frozen test and baseline | Trials rule | Kill predicate / eligibility |
|---|---|---|---|---|
| C01 | R1 echoes | coherent residual echo-comb statistic vs time-slide GR/no-echo background over the script's full event/delay scan | empirical slide background includes the registered scan | detection may support NVG at `≥3σ` global; no amplitude floor is derived, so a null cannot falsify NVG |
| C02 | R2 NS structure | one chosen EOS vs the joint NICER/mass/tidal likelihood and competing EOS families | all named constraints; fitted EOS parameters reduce effective dof | **ineligible:** complete EOS canon conflicts across files |
| C03 | R3 sub-solar horizon | a confirmed horizon-bearing object wholly below the propagated `M_crit` band vs GR compact-object interpretation | full frozen GWTC catalog | falsification requires both a mass posterior below the band and independent horizon evidence; current script alone cannot establish the latter |
| C04 | R4 QCD spectrum | `2M_Ω` vs a prespecified scalar-glueball observable; baseline is QCD/lattice itself | one frozen state, no candidate shopping | validation ineligible because the comparison is not baseline-discriminating; an experimental state assignment was not frozen |
| C05 | R5 CMB cutoff | fixed `k³`/registered cutoff shape and scale vs Planck ΛCDM using the same likelihood | include shape/scale scan and added-parameter penalty | subthreshold preference is not validation; current lite likelihood cannot trigger falsification |
| C06 | R6 dark energy | exact `w=-1` vs DESI DR2 BAO+CMB as primary combination; SN combinations are sensitivity checks | one primary combination; no best-SN selection | kill at `≥5σ` with the collaboration likelihood or a preregistered equivalent; current published primary result is `3.1σ` |
| C07 | R7 µHz comb | derived spectrum vs a µHz stochastic-background likelihood; astrophysical foreground is baseline | full registered frequency/contrast template scan | `OPEN`: no eligible µHz dataset; source also contains unresolved parameter/documentation contradictions |
| C08 | R8 evaporation | confirmed terminal PBH evaporation signature vs ordinary transient models | search collaboration's published trials | one confirmed terminal evaporation event falsifies the shutdown claim; published nulls are baseline-shared |

HADES and RHIC BES-II remain watch items `W01` and `W02`, not additions to the
current claim universe. DESI DR3 is `W03`. Promoting a watch item requires
`SPEC-v3` before its result is inspected.

## 4. Rejection boundary

The following never count as validation:

1. comparison with the dataset from which an anchor was adopted;
2. circular `H₀/N_e`, `T_c`, entropy, or similar reverse solutions;
3. fitted/calibrated outputs relabelled as parameter-free predictions;
4. a null also predicted by the baseline;
5. sub-`3σ` pulls, local-only scans, or a selected best dataset;
6. a single-detector GW candidate without coherent background trials;
7. scale coincidences and candidate shopping;
8. an internal theorem presented as empirical evidence;
9. qualitative explanation without a frozen likelihood;
10. a successful process exit or assertion that only checks a hardcoded range;
11. a hardcoded `True` in the falsifier dashboard;
12. combining mutually inconsistent model variants into one evidence count.

Historical examples already rejected by the repository audit include the
`H₀=72.8` circular chain, the adopted `T_c` “hit”, the frame-dependent `S8`
claim, manufactured SGR QPO comparison frequencies, hardcoded EOS headlines,
single-detector/local echo excesses, and reverse-solved entropy factors.

## 5. Route-to-work-product registry

| Route | Script(s) | Frozen input(s) | Output/artifact | Safe execution mode in this audit |
|---|---|---|---|---|
| R1 | `nvg_echo_residual_stack.py` | GWOSC strain + `data/gwtc_events.csv` | `data/nvg_residual_stack_results.csv` | do **not** rerun in place; recompute statistics read-only from committed CSV |
| R2 | `nvg_ns_nicer_joint_audit.py` | published values embedded in script | stdout | `PYTHONDONTWRITEBYTECODE=1 python3 verification/nvg_ns_nicer_joint_audit.py` |
| R3 | `nvg_mcrit_gwtc_check.py` | `data/gwtc_events.csv` | stdout | `PYTHONDONTWRITEBYTECODE=1 python3 verification/nvg_mcrit_gwtc_check.py` |
| R4 | `nvg_glueball_mass.py` | anchor + benchmark embedded in script | stdout | `PYTHONDONTWRITEBYTECODE=1 python3 verification/nvg_glueball_mass.py` |
| R5 | `nvg_cmb_lowl_refit.py`; `nvg_cmb_te_check.py` | Planck TT/TE text files | TT figure + stdout; TE stdout | TE rerun; TT marked `NOT RUN` because it is multi-minute and overwrites a tracked figure |
| R6 | `nvg_desi_dr3_forecast.py` | hardcoded Gaussian surrogate + DESI paper | stdout | rerun; publication values override the surrogate's universal `4.5σ` wording |
| R7 | `nvg_recondensation_dynamics.py`; `nvg_primordial_gw_comb.py` | action constants embedded in scripts | figure/stdout | frequency script rerun; recondensation calculation `NOT RUN`; no empirical µHz input exists |
| R8 | `nvg_hayward_evaporation.py`; `nvg_pbh_dark_matter.py` | anchors and literature limits embedded in scripts | stdout | theory scripts rerun; external PBH search likelihood not reproduced |

`nvg_dark_energy_desi.py` is explicitly excluded: it executes a retired
mass-melting model and returns a `4.77σ` tension with its own hardcoded contour.
The `nvg_global_significance.py` output is a curated compatibility diagnostic,
not a likelihood-ratio validation test.

The R1 writer opens its committed CSV with mode `w` before network processing.
It is therefore non-atomic and can destroy the previous artifact on a failed
fetch. A future implementation must write a temporary file, require a minimum
successful-event count, then atomically replace the artifact.

## 6. Frozen repository data and artifact hashes

Hashes are SHA-256 at the base revision.

| File | Data rows (excluding header where applicable) | SHA-256 |
|---|---:|---|
| `verification/data/gwtc_events.csv` | 391 catalog rows; 282 have usable component masses in R3 | `5c9876917eff320145d02f60328e6e1b6eef8d040e9711f4e94d85b3601c2ec3` |
| `verification/data/nvg_residual_stack_results.csv` | 241 analysed events | `3e41405339939f7cc55d1d419a0dc33871b05fb7891a003706591cae24e9d346` |
| `verification/data/planck2018_tt_full.txt` | 2508 lines | `ccf3113604020536f6f13ccf51680a7316ad0f32da558eee7f625e613bdd5522` |
| `verification/data/planck2018_te_full.txt` | 1996 lines | `8b2c97d8865ebfdfb2b23c3e6883a39820734b804ba6e353533657b3a2f71425` |
| `verification/data/hades_templates_20MeV.csv` | 31 rows; watch item only | `93e89bbb3e0f92774d1added70a30fbb60a8f452542634d86269d8462275fd82` |
| `verification/data/hades_templates_40MeV.csv` | 16 rows; watch item only | `0cbad36b31b419d8a365535728674dbb5bbe10bda230b102afb0c52d14ca64ac` |
| `verification/data/nvg_gw_template.txt` | 404 lines | `e5296f65641aa01528208a911576147640aae7988ed2a0ac0aaf4c736c80129e` |

Primary public-source checks used by the execution report:

- [GWOSC O4b/GWTC-5.0 release](https://gwosc.org/news/o4b-open-data-release/)
- [DESI DR2 cosmology paper](https://arxiv.org/abs/2503.14738)
- [NICER J0437-4715](https://arxiv.org/abs/2407.06789)
- [NICER/XMM J0614-3329](https://arxiv.org/abs/2506.14883)
- [NANOGrav public 15-year data](https://nanograv.org/science/data)
- [HAWC PBH burst search](https://arxiv.org/abs/1911.04356)

The vendored Planck files lack an in-repository retrieval recipe and upstream
checksum. Their repository hashes establish local identity, not full provenance.

## 7. Adversarial audit

Every candidate receives these attacks:

1. **Provenance:** attempt to find the confirming number or a reverse solution
   in the derivation history.
2. **Baseline:** evaluate the same statistic under the domain baseline.
3. **Trials:** count the full claim, dataset, event, shape, scale, and template
   search—not only the reported slice.
4. **Correlation:** collapse outputs sharing fitted parameters or data.
5. **Model identity:** reject evidence assembled from different EOS/cosmology
   variants.
6. **Artifact safety:** a failed or partial run must not replace a valid result.
7. **Best-case removal:** deleting the most favourable eligible row must not
   change the terminal verdict.

Git archaeology is limited to the history present in this clone. It cannot prove
that a prediction predates all external exposure without an independently
timestamped DOI, release, or preregistration.

## 8. Honest terminal state

The execution report must start with exactly one of:

`COMPLETE RESOLUTION`

`EMPIRICALLY SUPPORTED RESULT`

`AUDITED FRONTIER — INCOMPLETE`

`VALIDATED` or `FALSIFIED` may appear only after the relevant decision rule and
all required A1–A6 cells pass. Under the frozen `SPEC-v2` execution, the expected
honest terminal state is `AUDITED FRONTIER — INCOMPLETE`.

## 9. Scope-change rule

`SPEC-v2` is frozen after this execution. Selecting one EOS canon, repairing the
dashboard, adding a public HADES/BES-II result, changing a threshold, or adding a
claim creates `SPEC-v3`. Results from different specifications are never pooled
without an explicit hierarchical model and trials audit.
