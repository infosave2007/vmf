# Stellar repair, 2026-09-08

Status: fresh complete generation, producer self-validation and all 70 focused tests succeeded; canonical artifacts are written. Independent fresh-process public validation is running.

## Implemented numerical and data repairs

- The centre recurrence retains its source-pressure coordinate, now explicitly declared as `rho=sqrt(K_CONV)*r_km`, `m_formal=sqrt(K_CONV)*m_km`. Its consumer converts the coordinate before evaluating every power and returns geometric mass and pressure. Independent dimensional TOV coefficients and isolated monomials test the boundary. Every consumed binary64 coefficient is bound to its formal coefficient projection; recomputing a digest cannot hide an altered launch coefficient.
- Pressure RK4 now uses actual uniform steps 0.05, 0.025, 0.0125 km. Each trajectory records its actual spacing, start radius and number of full steps. Serialized observables are bound to their convergence summaries. No copied enthalpy trajectory supplies pressure outputs.
- Exact-vacuum derivatives use `[dm,dw,dy]`. The coefficient for `w=(P_source/pstar)^(2/5)` uses source pressure units, with the free-gas coefficient `g/(30*pi^2*M_N*hbar_c^3)`; the finite-density limit converges to it.
- Primary enthalpy rows own their actual leading-regular launch and direct stationary-source certificate. Pressure recurrence certificates belong to pressure rows. Terminal charts contain their own 16/32/64 ladder and adaptive oracle. Fixed schema is v2.
- Wall-clock telemetry is outside scientific/proof identities and saved scientific output. The producer no longer emits elapsed time in interval records. Canonical identity is mandatory; nonfinite payloads and invalid supplied writer inputs fail before writing. CLI reports the computed status.
- Source and contract paths use maintained portable snapshots. Upstream parameters are unchanged. The primary integrator's unverified leading-series substitute after failed numerical warm-up was removed; failed integration now raises.

## Supported scope of interval and tangent diagnostics

The earlier coefficientwise stationarity labels were unsupported: all orders reused one central scalar residual. Their fields now state their actual role as central branch checks accompanying numerical W density derivatives. The recurrence arithmetic precision is 50 decimal digits, its source evaluations at least 120, and original-equation substitution 180; metadata now distinguishes them.

The fabricated tangent “a posteriori tube” was removed. The tangent is a numerically integrated bracket selector, cross-checked against independent ordinary five-point stencils with the unchanged fixed comparison threshold. Central source diagnostics do not claim trajectory error certification.

The terminal machinery is explicitly a numerical interval diagnostic, **not a rigorous global error theorem or correct-rounding certificate**. It preserves the original mathematical error thresholds and independent arithmetic replay. Repairs include:

- exact 120-digit remainder serialization, instead of comparing a float projection with a decimal value for exact equality;
- separate parent-Hermite and subdivision endpoints, continuity and coverage checks;
- binding independently reconstructed matrices, residuals, forcing and propagated error maps to their outputs, and propagating the reconstructed incoming error;
- genuine interval products in the main producer/reconstruction matrix kernels, all 128 positive-series terms, and immutable direct-error limits;
- explicit numerical self-map/error/rounding-cell check names and `rigorous_error_certified=False`.

The strict acceptance evidence for the stellar observables is independently computed numerical convergence and equation consistency. Unresolved binary64 S6−S4 differences have null observed orders; no analytic power is substituted for an observation. A broader rigorously enclosed whole-trajectory theorem has not been established.

## Completed early checks

All values below were freshly integrated; historical values were diagnostic references only.

At x=0.1, primary M=0.5992993651640034 M_sun and Lambda=10732388.574144613. Three actual pressure steps completed in 8.20 seconds in the original diagnostic runtime. At the finest step, relative differences were M=4.45e-9, R=2.48e-7, k2=5.43e-6, Lambda=6.65e-6. All three step sizes produced distinct observables.

Pinned Python 3.12 all-density smoke took 66.92 seconds. Finest Lambda differences were 8.29e-7 at x=0.001, 2.95e-5 near the independently evaluated maximum-density witness, and 2.92e-5 at x=10. The remaining two predeclared low-density interval replays both passed their numerical checks. All retained cross-method limits passed.

The fresh x=0.001, dr=0.05 diagnostic has `independent_replay_pass=True`, `interval_error_check_pass=True` and `direct_error_pass=True`, but `rounding_cell_check_pass=False`: the k2, Lambda and y_R rounding-cell checks are false. That negative result is retained. Passing the direct-error diagnostic is not described as correct rounding or a rigorous error certificate.

Focused pre-generation batch: 12 passed, one obsolete source-string assertion failed because it described wrapper comments instead of the implementation. It was replaced by a real cache-disabled run and counter check; that test passed independently (1 passed in 13.60 seconds). The initial finite-density exact-zero limit test at x=1e-12 resolved a real O(n^(1/3)) interaction correction, not an implementation mismatch; lowering the diagnostic density to 1e-18 retained the 1e-5 comparison threshold and gave relative error 4.18e-6.

Frozen producer SHA256: `b22c9e4860f3a9a3d7cd2204f0e7529f67ede10cd178cd9dcfceabbcc8f81a17`.

Frozen focused-test SHA256: `30ed8a95679ab5b5c35d1bbf5e4da9a6ec6a4ee697ba2f9fdce43d85fcf6a196`.

The first complete-generation attempt computed both matched grids and all pressure-refinement trajectories, finishing the convergence-summary stage at 632.706 seconds. It then failed in the external scratch logging wrapper, which indexed a nonexistent positional argument for the zero-argument gamma=2 control. The final producer self-validation and artifact writes were not reached. This is not counted as a successful build. The original failure log is preserved as `scratch/stellar-build-2026-09-08.log`.

The scratch-only wrapper was corrected and all five logging branches were checked with cheap argument/result-preserving stubs. The completed-summary fields are now computed before artifact writing, so invalid logging-field access cannot interrupt a successful write afterward. Source and focused tests were not changed. The genuinely fresh second build completed with separate stage telemetry in `scratch/stellar-build-2026-09-08-attempt2.log`.

## Successful full generation

Command: `PYTHONDONTWRITEBYTECODE=1 .venv-research/bin/python scratch/stellar-build-2026-09-08.py build`.

Attempt 2 exited 0 at **627.154 seconds**, including live producer self-validation and writing both artifacts. The grid32 stage ended at 155.515 seconds, grid64 at 427.045 seconds, convergence summary at 624.207 seconds, and gamma=2 control at 624.747 seconds. No physics/math acceptance threshold was changed to achieve this result. A subsequent fresh-process schema/semantic check returned `(True, [])`.

- Status: `STABLE_BRANCH_TERMINATED_AT_FIRST_MASS_MAXIMUM`; numerical convergence true; physical evidence weight remains zero.
- Grid32: 129 raw rows, 110 retained stable rows. Matched grid64: 257 raw rows. Eighteen actual pressure-refinement trajectories, each with its own terminal ladder/oracle.
- Real owner ledger: 1032 requested, 873 unique and executed strict enthalpy integrations, 159 exact-key hits/reuses. Serialized cap remains 1200 actual integrations, not elapsed seconds.
- All three independent terminal diagnostic replays passed. All three aggregate rounding-cell checks were false. No rigorous whole-trajectory or correct-rounding claim is made.
- Maximum independently located at `x=2.485075342166858`: `M=2.7425498519222313 M_sun`, `R=19.673224944429776 km`, `k2=0.02921164433168922`, `Lambda=52.69331973210884`.
- Grid32-to64 relative maximum shifts: mass `1.0347050e-13`, central density `6.4695881e-8`; both pass the unchanged uncertainty comparisons.
- All six independent 180-decimal original-equation centre-substitution replays passed. Observed substitution residual orders approach 14 for mass and y and 12 for pressure; retained thresholds are 9.5, 6.5 and 6.5. In contrast, every binary64 start-halving order is null because those S6−S4 differences are unresolved. The higher-precision substitution result is not substituted for the missing binary64 observation.

Worst finest-pressure versus primary relative differences across all six independent density witnesses: M `1.5304e-8`, R `2.0956e-6`, k2 `1.9851e-5`, Lambda `2.9533e-5`. The retained acceptance limits are M/R `1e-3`, k2/Lambda `5e-3`. Worst final-two-pressure-mesh differences are M `5.1622e-8`, R `4.9791e-6`, k2 `4.1931e-5`, Lambda `6.7083e-5`; limits remain M/R `5e-4`, k2/Lambda `2e-3`. These are observed numerical comparisons, not rigorous bounds on the continuum solution.

Artifact identities:

- Canonical scientific result SHA256: `8bd852937d6ff5e8a512bd6bd932bf6970ed9dd435ea9f827e21bffb5eed9dec`.
- `verification/source_complete_stellar_prediction_results.json`: 43,686,931 bytes, SHA256 `4b46bcac26e5a4b884d36b0d864de4307f13c2efc3b7267c0a1923a19d9611b8`.
- `verification/source_complete_stellar_prediction_report.md`: 2635 bytes, SHA256 `91d365515a92764482d3b4253866aeee1fe87aba81f095a5363666cb1ebbfa2a`.

## Full focused tests

Command: `PYTHONDONTWRITEBYTECODE=1 .venv-research/bin/python -m pytest -q verification/test_source_complete_stellar_prediction.py`.

Result: **70 passed in 374.62 seconds (6:14), exit 0**, log `scratch/stellar-focused-2026-09-08.log`. No test was skipped or removed during this acceptance run. The existing synthetic terminal-jet control was slow but completed; it was not replaced by a passing flag or a copied physical witness.

The run includes positive stored-artifact semantic acceptance before mutation tests; specific rejection reasons; live independent M/R/k2/Lambda comparisons on three actual meshes; dimensional centre and exact-zero derivative checks; coefficient aliases mutated with refreshed digest; invalid writer input and missing identity/nonfinite rejection; terminal parent-Hermite/immutable-limit/directed-arithmetic replay mutations; physical terminal diagnostic positive control; and repeated strict-star payload equality.

The parent owns a separate fresh-process public validator without evaluator injection or artifact writes. Its exact outcome will be appended after completion. Source, tests and canonical artifacts remain frozen throughout. All elapsed-time telemetry stays outside the deterministic scientific artifacts.

## Scope of this worker's file changes

Maintained implementation/artifacts: `verification/source_complete_stellar_prediction.py`, `verification/test_source_complete_stellar_prediction.py`, `verification/source_complete_stellar_prediction_results.json`, `verification/source_complete_stellar_prediction_report.md`.

Worker-owned external evidence: this report, `scratch/stellar-smoke-2026-09-08.py`, `scratch/stellar-live-witness-2026-09-08.json`, `scratch/stellar-build-2026-09-08.py`, both build-attempt logs, and `scratch/stellar-focused-2026-09-08.log`. The separate plain public-validation driver/log belong to the parent. No parameter, upstream implementation, dependency, registry, manifest or physical model document was edited by this worker.

Remaining scientific limitations are substantive: this is a one-component mathematical sequence with fixed parameters that fail nuclear saturation, not a physical neutron-star prediction; missing composition/crust inputs are not supplied; global rigorous error enclosure and universal correct rounding remain unproved; null binary64 start orders remain null. The repaired pipeline offers source consistency, independent numerical convergence and runtime-scoped reproducibility, not empirical confirmation.
