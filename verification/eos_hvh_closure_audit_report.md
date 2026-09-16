# HVH closure audit — maintained beta-equilibrated EOS

## Control block

- Status: **FAIL_CLOSED_MAINTAINED_STRESS_CHEMICAL_PATH_ABSENT** (the scientific result is intentionally fail-closed).
- Scope: independent audit files named `eos_hvh_closure_audit` plus this report; no maintained producer, registry, README, or article edits.
- Producer: `verification/nvg_eos_beta_saturated_vector.py` (runtime SHA-256 `72a62749ba87d6db1104ed108066acd68360853bf24e8f2e935b9a56d46df490`).
- Source scan found 19 first-party consumers/importers of this producer; the complete list is in the generated JSON inventory.
- Terminal JSON: `verification/eos_hvh_closure_audit_results.json`.
- Generated repair verification report: `verification/eos_hvh_closure_audit_report.md`.

## Contract inventory and derivation

The maintained producer returns `eps_no_vec`, composition, `m_dirac`, and `mu_e`; its pressure is `n**2*gradient(epsilon/n,n)`. A stress tensor and nucleon chemical-potential path are **absent**, as are explicit rearrangement and stationarity residuals. The audit therefore reconstructs
`p_HVH = sum_i n_i mu_i - epsilon` and compares it with fourth-order five-point estimates of `n**2 d(epsilon/n)/dn` at relative stencil steps 0.02, 0.01, and 0.005. Density-dependent mass and vector terms include `Sigma_R,s = n_s dM_base/dn` and `Sigma_R,v = Cw0*n**2*(dg/dn)/2`.

## Results

| surface | result |
|---|---:|
| analytic power-law derivative benchmark | 2.320e-10 relative error (PASS) |
| ideal cold npe-mu HVH closure | 4.223e-10 max relative |
| ideal derivative convergence | 1.470e-08 max relative |
| interacting smooth-mass-branch HVH closure | 8.382e-06 max relative |
| interacting no-rearrangement closure | 1.925e+00 max relative |
| interacting derivative convergence | 5.721e-06 max relative |
| max scalar rearrangement | 1.485e+02 MeV |
| max vector rearrangement | 3.291e+02 MeV |
| max Dirac stationarity residual | 1.807e-06 MeV |

Interacting inputs are the maintained screening point `(k1,k2,Cs,Crho,alpha_v,nu_v)=(0.25,0.80,900,600,4.0,2.0)`, with calibrated `Cw0=1794.22`. The no-rearrangement comparison is a diagnostic, not a replacement EOS.

## Judgment and boundary

The ideal reference and the independently reconstructed interacting energy functional satisfy the HVH identity on the smooth mass branch within the reported numerical controls. This does not certify the maintained API: because its stress/chemical-potential path is absent, the audit remains **FAIL_CLOSED** and no maintained-producer closure PASS is claimed. The current-mass clipping kink is retained and reported in the JSON rows; derivatives across that non-smooth point are not silently promoted to a smooth theorem.

Exact repair commands and counts are frozen in `Lunacy/runs/deep-physics-audit/phases/phase-1/evidence/hvh-closure-repair-1-terminal-verification.log`.
