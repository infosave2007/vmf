# Finite-droplet residual-enrichment contract (P1/I1)

This contract defines the live calculation in
`verification/nvg_droplet_residual_enrichment_audit.py`.  It is a bounded,
zero-temperature, symmetric, Coulomb-free Thomas--Fermi second-variation
calculation around the accepted `N=208` inverse W8 backgrounds at `y*=.90`
and `.93`.  Its independent-evidence weight is zero.  It tests declared
larger finite trial spaces; it is not a frequency, a nonlinear theorem, full
stability, a real-nucleus calculation, or empirical evidence.

## Physical source and base representation

The old stability producer remains the equation owner.  The new producer
recomputes its backgrounds, displacement/edge integration, source-complete
matter/scalar/vector Hessian, positive Gauss solve, physical norm, fixed-N,
translation, finite-difference and inherited control predicates.  Old source
files and their default outputs are read-only.

The base space is every actual column of the existing size-18-per-block basis:
36 raw columns for each `ell=0,...,6`.  On the fresh `24 fm/1600` selection
grid, and again on every replay grid, columns are represented by the weighted
thin SVD of

```text
F = [sqrt(w)*x*u/n0_dim ; sqrt(w)*x*v],
```

where `w` are the positive Simpson weights of the sampled grid.  No singular
direction is discarded by the inherited `1e-10` Gram cutoff and no
pseudoinverse is used.  The same coordinate map transforms `u`, `v` and
`vprime`; the Hessian and norm are then reassembled from those physical
fields.  Reconstruction, Gram orthogonality, direct Rayleigh consistency,
raw singular values, norms, condition number and unchanged-cutoff rank are
reported.  An unrepresentable base direction invalidates the finite-space
classification.  In `ell=1`, only the known joint translation is removed by
the physical `G`-orthogonal quotient using the inverse coordinate map.

## Candidate dictionaries and selection

The training dictionary is fixed before scoring.  For each fresh training
edge `E`, centers are `E*(.25,.40,.55,.70,.85,1.00,1.15)` and widths are
`E*(.12,.20)`.  For each center/width there is one density displacement and
one scalar candidate.  With `z=(x-center)/width`,

```text
b(z)=(1-z^2)^4                       (|z|<1; zero otherwise)
b'=-8*z*(1-z^2)^3/width
b''=(1-z^2)^2*(56*z^2-8)/width^2.
```

Density candidates use the existing continuity derivative, including the
fresh `n'`, and are exactly zero outside occupied matter.  Scalar candidates
use `v=b, vprime=b'`.  Every `ell=0` density candidate receives the existing
independent edge-aware fixed-number projection; its correction and measured
residual are recorded.  Zero-support candidates are ineligible, not deleted
base directions.

Before each score, the candidate is projected from the whole current physical
space and, for `ell=1`, from the translation direction.  A candidate whose
remaining Gram norm is at or below the inherited `1e-10` relative resolution
is recorded as near-dependent and is not counted.  The score is

```text
sum_i |H(z,phi_i)-lambda_i G(z,phi_i)|^2 / G(z,z),
```

over the first three current internal modes, with a whole boundary-degenerate
cluster included when its measured separation is below numerical sign
resolution.  Scores use absolute residuals and include negative eigenvalues;
deterministic dictionary order breaks exact ties.  Four directions are added,
scores are recomputed, and four more are added.  The selected IDs and their
absolute training centers/widths are frozen thereafter.

## Replay and held-out probes

Each frozen selection is replayed on `800/1600/3200` intervals, `24/32 fm`,
and the tighter same-domain `24 fm` background.  Fresh background fields are
used on every replay; candidates are never reselected or recentered.  The
three nested stages are `0,4,8` added directions.  The output compares the
lowest five internal curvatures and first-three principal angles, and checks
Rayleigh--Ritz nonincrease within the measured residual allowance.  Violated
nesting, physical, rank, fixed-number, operator or inherited controls fail
closed; no value is clipped.

The independent held-out dictionary is never passed to selection: centers are
`E*(.325,.475,.625,.775,.925,1.075)` and width is `E*.16`, with density and
scalar candidates at each center.  Every probe reports its physical norm,
remaining eligibility/zero-support status, residuals normalized by
`max(abs(lambda_i),1e-8)`, and density/scalar/interior/edge attribution at all
three stages.  The terminal held-out diagnostic gate is the fixed inherited
`0.05` discretization limit.  It is an additional finite-space diagnostic,
not an infinite-dimensional error bound.  A failed gate leaves the result
unresolved and does not change the dictionary or limit.

## Interface and interpretation

The command is strict JSON and writes nothing:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=verification python3 -B verification/nvg_droplet_residual_enrichment_audit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=verification python3 -B -m unittest verification.test_nvg_droplet_residual_enrichment_audit -v
```

`COMPUTED_FINITE_W8_RESIDUAL_ENRICHMENT_ZERO_EVIDENCE` means the declared
calculation was freshly produced.  It does not promote a positive finite
space to full stability.  A robust negative direction, if observed with all
physical/refinement/error guards, is a finite-space variational witness.
Residuals cannot prove completeness: a decoupled negative direction may have
zero residual against every current low mode.  The variational identities are
mathematical tools applied to this calculation, not new physical laws.

No Coulomb/asymmetry, pairing, shell structure, quantum RPA, time dynamics,
gravity, real-nucleus phenomenology, independent empirical input or universal
NVG conclusion follows from this audit.
