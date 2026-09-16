# Spatial static-response contract

This contract freezes the bounded calculation implemented by
`verification/nvg_spatial_identifiability_audit.py`. It is a conditional
zero-temperature local Thomas--Fermi result, not an empirical validation,
quantum RPA, finite-nucleus, transport, or full-dynamics result.

## Fixed functional and coordinates

Use natural units with `n` in MeV^3, `y=W/W0` dimensionless, `k` in MeV,
and `Z=W0^2` in MeV^2. At a homogeneous state, before eliminating the
static vector constraint `A0`, the quadratic Hessian for a Fourier mode is

```
H = [[a, b, g],
     [b, d0 + Z_s*k^2, h],
     [g, h, -(t0 + k^2)]],
```

where `a=F_nn`, `b=F_ny`, `d0=F_yy+U_yy-momega^2*A0^2`,
`g=gomega`, `h=-2*momega^2*y*A0`, `t0=momega^2*y^2`, and
`Z_s=(s W0)^2`. The negative vector entry is a Gauss saddle/constraint;
it is not a negative-energy propagating ghost. The fixed density functional
uses `A0=g*n/(momega^2*y^2)`.

Set `L=t0+k^2` and eliminate the vector constraint. The resulting matrix is

```
D = a + g^2/L
B = b + g*h/L
C = d0 + h^2/L + Z_s*k^2
S = D - B^2/C
```

A positive static susceptibility is reported only when `C>0` and `S>0`.
An independent direct solve of `H u=(1,0,0)` must give `u_n=1/S`; a
Schur formula is not accepted as its own independent check. The `chi` used
here is defined by a source `-mu_ext*n`, so `delta n=chi*delta mu_ext`.
For a source `+V_ext*n`, the conventional response is
`delta n/delta V_ext=-chi`.

## Scale family and identifiability

The correlated family is `W0 -> s W0`, `g_s -> g_s/s`,
`q_phi -> q_phi/s`, with `M_N`, `m_omega`, `g_omega` and `U(y)` held fixed.
For Q4, `lambda -> lambda/s^4`; for each retained W8 polynomial the
coefficients in `y` are copied unchanged. The actual candidate combinations
are recomputed rather than pinned to a baseline. Homogeneous rows remain
invariant, while the canonical gradient coefficient is `Z_s=s^2 Z`.

The physical canonical vacuum scalar mass is computed from the actual
potential, `m_sigma^2 = U_yy(1)/W0^2 = d^2 U(W/W0)/dW^2|_{W=W0}`. For Q4 this
reduces to `sqrt(2*lambda)*W0` because `U=A*(y^2-1)^2/4` and
`A=lambda*W0^4`. For W8 the retained `z^2,z^3,z^4` polynomial coefficients
define the potential and its curvature; the `BulkModel` quartic `lambda` and
derived `A` are unused auxiliary fields, not physical W8 potential
parameters. Serialized scale metadata marks those auxiliaries explicitly and
preserves the polynomial coefficients.

On a common stable domain, for two scales `s1,s2`,

```
S(s2)-S(s1) = B^2*Z*k^2*(s2^2-s1^2)/(C1*C2)
```

and `dS/d(s^2)=B^2*Z*k^2/C^2 >= 0`; consequently
`dchi/d(s^2) <= 0` where `chi` exists. The exact `k=0` mode and any `B=0`
mode are scale-blind. A positive blind wave number (when it exists) is

```
k_blind^2 = -g*h/b - t0
```

for nonzero `b`. Such a high-k local-TF point is scalar decoupling, not a
transparency resonance or an all-k physical prediction.

For a measured/otherwise-known stable `chi`, the conditional inversion is

```
Z_s = [B^2/(D - 1/chi) - (d0+h^2/L)]/k^2.
```

It refuses `k=0`, `B=0`, nonpositive denominators, unstable rows, and
nonpositive/nonfinite reconstructed values. Both cancellation-sensitive
subtractions must also meet the declared numerical precision criterion:

```
r_den = |D - 1/chi| / max(|D|, |1/chi|, 1e-100) > 1e-45
r_sub = |B^2/(D - 1/chi) - (d0+h^2/L)| /
        max(|B^2/(D - 1/chi)|, |d0+h^2/L|, 1e-100) > 1e-45.
```

Failure of either separation is reported as a numerical-conditioning refusal,
not silently converted into a large or sign-changing `Z_s`. These thresholds
guard retained arithmetic precision only; they are not an experimental
uncertainty estimate, so a physical inversion still requires supplied
measurement errors and their covariance. Repeated nondegenerate wave numbers
should return one `Z_s` only when the other coefficients and the response
itself are known and the inversion is well conditioned. This does not promise
that an experiment can isolate `chi`, and it does not remove existing tail--jet
ambiguity.

## Declared numerical atlas

The producer recomputes the baseline Q4 branch at
`n/n0 = 0.37, 1, 2.75, 7.25`, and the two existing three-target inverse W8
alternatives with `y*=0.90, 0.93` at `n0`. These W8 constructions train on
`n0=0.16 fm^-3`, binding `-16 MeV`, pressure zero, and `K=240 MeV`; they are
not adopted physical models or held-out evidence. Every background is shown
at scales `0.5, 1, 1.3, 2` and `k=0,100,200 MeV`, for 72 rows total. Stable,
unstable, and singular statuses are retained rather than filtered.

The `k=0` comparison is the grand-canonical compressibility/long-wavelength
limit and matches the homogeneous relaxed derivative. The exact uniform mode
is excluded at fixed total particle number. Finite `k=100,200 MeV` rows are
local-density demonstrations only; omitted finite-size, quantum, dynamic,
isovector, composition, Coulomb, and uncertainty effects remain outside this
contract.

All source values are recalculated at 80 and 110 decimal digits and compared
by live observables. The command prints strict JSON and does not write a
result table. Evidence weight is zero and no empirical PASS is permitted.
