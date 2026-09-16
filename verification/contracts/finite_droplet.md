# Finite-N W8 droplet audit contract (P1/I1)

This contract freezes the numerical object checked by
`verification/nvg_finite_droplet_audit.py`.  It is a conditional,
zero-temperature Thomas--Fermi calculation with zero independent-evidence
weight.  The two inverse W8 designs `y*=0.90` and `y*=0.93` are inherited
from `source_complete_scaling_saturation_audit.inverse_potential_jet`; their
`n0=.16 fm^-3`, `E/A-M=-16 MeV`, and `K=240 MeV` values are calibration
inputs, not held-out predictions.  `N=40` and `N=208` are numerical controls,
not identifications with calcium or lead.

## Frozen radial equations and units

Use

\[
x= W_0r/(\hbar c),\quad a=A/W_0,
\]

and dimensionless densities `n_d=n_nat/W0^3` and `n_s,d=n_s/W0^3`.  With
`q=momega/W0` and `g=gomega`,

\[
\begin{aligned}
y_{xx}+2y_x/x&=U_y/W_0^4+(M/W_0)n_{s,d}-q^2ya^2,\\
a_{xx}+2a_x/x&=q^2y^2a-g n_d,\\
Q_x&=x^2n_d,\\quad
N&=4\pi Q(X).
\end{aligned}
\]

The local KKT closure is

\[
\nu=\mu-gW_0a,
\quad
n_d=\begin{cases}
{d\over6\pi^2W_0^3}(\nu^2-M^2y^2)^{3/2},&\nu>M y,\\
0,&\nu\le M y.
\end{cases}
\]

At the regular origin `y_x=a_x=Q=0`; at the finite box `y=1`, `a=0`.
The finite box is an approximation.  A converged branch must be checked on
the larger 24-fm box; `mu<M` and a negligible exterior density are required
for a localized vacuum exterior.  A solution with `mu>=M`, appreciable
exterior density, or `E/N>=M` is reported as unbound/nonlocalized or
unresolved, never as a bound droplet.

## Energy and Gauss obligations

All components are independently integrated in MeV:

\[
\begin{aligned}
T_W&={W_0\over2}4\pi\int x^2y_x^2dx,&
V_U&=W_0 4\pi\int x^2U/W_0^4dx,\\
E_F&=W_0 4\pi\int x^2F_4/W_0^4dx,&
T_A&={W_0\over2}4\pi\int x^2a_x^2dx,\\
V_A&={W_0\over2}4\pi\int x^2q^2y^2a^2dx.
\end{aligned}
\]

The total is `E=T_W+V_U+E_F+T_A+V_A`.  The independently measured source
`S=∫g n A` and finite-boundary flux
`B=4*pi*W0*X^2*a(X)*a_x(X)` must satisfy

\[
2(T_A+V_A)=S+B.
\]

The boundary term is retained even though it is small on localized branches;
the audit does not claim an exact finite-box identity with the flux omitted.
Terminal stationarity additionally requires finite diagnostics on the positive
physical branch (`y>0`, `n>=0`) and
`|2(T_A+V_A)-S-B|/[2(T_A+V_A)] <= 1e-6`.  The denominator is the positive
Gauss energy, not the total rest-mass energy; this guard is part of the
terminal acceptance predicate.

## Fixed-N scale check

For `n_lambda(r)=lambda^3 n(lambda*r)` and `y_lambda(r)=y(lambda*r)`, the
infinite-space envelope derivative is

\[
D={dE\over d\ln\lambda}=-T_W-3V_U+3\int P_Fd^3r+T_A+3V_A.
\]

The numerical finite difference re-solves the Gauss equation separately at
each lambda.  Its raw quadrature `N` drift is reported; before each Gauss
re-solve, the scaled density is multiplied by the tiny target-`N` normalization
correction, and the controlled drift is reported and bounded.  It uses several
centered steps and a Richardson comparison; rescaling an old `A` field is not
accepted as the derivative oracle.  The controlled-number bound is `1e-10` in
relative units; the raw drift and the normalization-correction magnitude remain
visible diagnostics.  The identity

\[
E-\mu N={2\over3}(T_W-T_A)-{D\over3}
\]

is reported as an additional bookkeeping check.  Positive curvature in this
single scale direction is not a full stability or breathing-frequency claim.

## Protocol, labels and interface

Each case runs the deterministic radius factors `.8, 1, 1.2`, then a
coarse/medium/fine 16-fm refinement and a 24-fm domain check.  Every seed
attempt and every distinct converged branch is included in the JSON.  A
`NUMERICAL_STATIONARY_BOUND_CANDIDATE` requires all three declared refinement
levels, both adjacent refinement changes and the fine-to-domain energy/rms
changes at most `.02 MeV/.02 fm`, plus a converged larger-domain row.  It also
requires relative number error at most `1e-6`, independently measured field
residual at most `1e-4` in the stated dimensionless scales, local
chemical-equilibrium error at most `.01 MeV`, `|D|/E <= 1e-3`, the Gauss
identity guard, and the fixed-`N` finite-difference check.  A failed or missed
limit remains visible as `FAILED_NUMERICAL_PROTOCOL` or a residual-limit
label; a successful smaller-box row never promotes a failed domain check and
is not silently converted into a physical no-go.  The N=40 controls miss some
protocol limits and are not a no-binding theorem.

The CLI is strict JSON and writes nothing:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B verification/nvg_finite_droplet_audit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=verification python3 -B -m unittest verification.test_nvg_finite_droplet_audit -v
```

This surface does not introduce Coulomb, isospin/rho, pairing, shell,
spin-orbit, quantum RPA, crust, beta equilibrium, gravity, observational data,
or a universal finite-nucleus claim.
