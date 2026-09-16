# Finite-droplet coordinate-response contract (P1/I2)

This contract defines the live calculation in
`verification/nvg_droplet_coordinate_response_audit.py`.  It is a bounded,
zero-temperature, symmetric, Coulomb-free Thomas--Fermi calculation around the
accepted `N=208` W8 backgrounds at `y*=.90` and `.93`.  Its independent-evidence
weight is zero.  It does not turn a finite trial space into a full stability or
empirical result.

## Coordinate replay

The existing stability producer assembles the same physical Hessian `H` and
Gram form `G`.  The new representation is the congruence

`c = D z`, `Hbar = D^T H D`, `Gbar = D^T G D`, `tbar = D^-1 t`,

with `D_jj = 1/sqrt(G_jj)`.  The existing positive Gram cutoff is applied to
the transformed pair; raw and transformed ranks are both retained.  No
physical column is removed, no pseudoinverse is used, and recovered rank is
not a positivity certificate.  The full inherited basis/grid/domain protocol
remains `ell=0..6`, block sizes `6,10,14,18`, intervals `800,1600,3200`,
domains `24,32 fm`, and the tighter same-domain background.

Terminal rank replays retain the predeclared raw-only
`S_jj=10^(k*j/(m-1))` diagnostic and, separately, apply `S` followed by
`D_S=diag(S^T G S)^(-1/2)` for `k=-6,0,6` on the `18/1600/32 fm` pair in all
fourteen target/sector combinations.  The latter 42 rows are compared with
the independently computed unit-equilibrated reference and include inverse
transformation of `t`, full raw/internal rank fields, inherited record guards,
and physical norm/Rayleigh/translation checks.  Raw rank sensitivity is
conditioning evidence, not a veto on a valid normalized sector.  Missing or
failed normalized rows leave the conservative sector result
`UNRESOLVED_FINITE_BASIS_SIGN_OR_CONVERGENCE`; they cannot create a new
positive class.  Both positive and negative labels are downgraded on any
mandatory case or physical-control failure.

The `ell=1` vector is transformed with the inverse scale and is still removed
only through the existing one-dimensional joint translation projector.  The
audit separately checks physical lifting, direct quadratic Rayleigh values,
physical Gram norm and translation orthogonality.  Weighted Simpson SVD/QR
column independence uses the same positive radial Gram weight and is marked
diagnostic only.

Known-answer mathematical controls are part of the result: the two-by-two
`G=diag(1,1e-12)`, `H=diag(1,-1e-12)` oracle must return `[-1,+1]` after
equilibration; a well-conditioned nonorthogonal congruence must preserve the
generalized spectrum; and a manufactured negative translation mode must remain
negative in the raw pair while being removed only by the declared projector.

## Schur diagnostic

For `ell=0,2,...,6`, the existing density/scalar block is written

`H = [[S,R],[R^T,T]]`, `Hred = T - R^T S^-1 R`.

The inverse is a strict solve and is used only when all eigenvalues of `S` are
strictly positive.  The audit checks full/reduced inertia and the square
completion

`Q = (cu + S^-1 R cv)^T S (cu + S^-1 R cv) + cv^T Hred cv`.

The density interpretation is explicitly skipped for `ell=1`, where the joint
translation quotient must be retained.  Schur algebra does not override a
physical rank, refinement, fixed-number or sign failure.

## Angular response

For fixed radial profiles and `tau=ell(ell+1)`, let

`w_tau = K_tau^-1 s`, `s = g u - 2 q^2 y a v`.

On the common full radial domain,

`Q(tau2)-Q(tau1) = (tau2-tau1)[integral(v^2 dx)-integral(w_tau2*w_tau1 dx)]`,

and `dQ/dtau = integral(v^2-w_tau^2) dx`.  The density profile is restricted
to occupied matter only; scalar profiles use the full domain.  The fixed smooth
bump is centered at half the edge, has half-edge-width, and is peak-normalized.
The three profiles are density-only, scalar-only and coupled density--scalar.
Pairs `ell -> ell+1` are `2->3`, `4->5`, `6->7`, and `10->11`, on both
background domains and all three grids.  Continuum Simpson quantities are
checked against the inherited `0.05` discretization limit, and each fixed
domain/profile/ell group records finest changes for the calculated continuum
quantities and identity error.  Production P1 operators and mass matrices are
stored as exact tridiagonal bands and solved in O(n) with the existing
tridiagonal routine; a small dense oracle is test-only.  The P1 identity uses
the exact angular mass bands and exactly the tridiagonal solver load weights;
its relative identity limit is `1e-10`.  Positive operator/source controls and
the decreasing vector contribution for the pure-density profile are explicit.

The decreasing vector term for `v=0` is not a negative total curvature, and no
automatic high-`ell` positivity or convexity of an optimized lowest eigenvalue
is claimed.

## Interface and interpretation

The producer emits strict JSON and writes nothing:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=verification python3 -B verification/nvg_droplet_coordinate_response_audit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=verification python3 -B -m unittest verification.test_nvg_droplet_coordinate_response_audit -v
```

`COMPUTED_FINITE_W8_COORDINATE_RESPONSE_ZERO_EVIDENCE` means that the declared
calculation and its diagnostics were freshly produced.  It does not mean all
diagnostic controls pass: unresolved rank/protocol rows are retained in the
JSON.  No Coulomb/asymmetry, pairing, shells, quantum RPA, dynamics, gravity,
real-nucleus phenomenology or independent empirical input is included.
