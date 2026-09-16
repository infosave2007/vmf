# Nuclear closure audit contract (P1)

This contract freezes the object checked by
`verification/nvg_nuclear_closure_audit.py`.  It is a mathematical
conditional calculation with zero independent-evidence weight; it is not a
new physical parameter fit.

## Sealed inputs

The only accepted branches are the already declared inverse W8 designs
`y*=0.90` and `y*=0.93`, plus the `y*=0.75` negative control.  All use

\[
n_0=0.16\ {\rm fm}^{-3},\qquad K=240\ {\rm MeV},\qquad
E/A-M=-16\ {\rm MeV},\qquad \mu=M-16=923\ {\rm MeV},
\]

with `M=939 MeV`, `W0=859 MeV`, `mω=782.6 MeV`, `gω=10.12`,
`hbar c=197.3269804 MeV fm`, Fermi degeneracy `d=4`, and `Cρ=0`.
The inverse W8 coefficients are reconstructed in the audit from these exact
(decimal) declarations; a saved answer table is not an input.

## Global-support obligation

For `z=y²−1`,

\[
U_8(y)=a_2z^2+a_3z^3+a_4z^4,
\quad
\Pi_\mu(y)=\sup_{n\ge0}\left[\mu n-F_4(n,My)-\frac{C_vn^2}{2y^2}\right],
\quad H=U_8-\Pi_\mu.
\]

The two-species reduction is allowed only because each Fermi term is strictly
convex and `Cρ≥0`: the fixed-total-density minimum is symmetric.  The live
proof must retain the positive-integrand identities `F_nn>0` and `F_yy≥0`,
`C_v>0`, `D_n=F_nn+C_v/y²>0`, a directed coefficient enclosure, and an
explicit union of accepted intervals.  Extracted interval endpoints may not
be rounded at a lower ambient precision.  It must separately cover

* `0<y≤1/2` by `U≥(9/16) min[-1,-3/4] q` and
  `Π≤μ²/(8 C_v)`;
* each target's exact `H=H'=0` neighborhood with a positive lower bound on
  `H''=U''+G_yy−G_ny²/G_nn` (the analytic `F_yy≥0` lower bound is allowed);
* the remaining compact interval by outward tangent/Lipschitz boxes with no
  gaps, empty leaves, or unresolved leaves;
* the one-sided onset `y_on=μ/M`, where `Π=0` for `y≥y_on`, and the complete
  `y→∞` tail through `q(z)>0`, `a4>0`, and the vacuum equality at `y=1`.

Each atlas endpoint is canonicalized once: that same decimal is consumed by
the interval constructor, published as `lo/hi`, and checked as the leaf's
`proof_box_endpoints`. Each tangent bound evaluates its point gap at the
canonical midpoint and multiplies the derivative bound by an outward maximum
of the two distances from that midpoint to the canonical endpoints. Serialized joins are compared as exact decimal
rationals, with no adjacency tolerance. The final mass-bound box ends at the fixed decimal
`0.9829605963791267305644302449425`, strictly above the exact rational
`923/939`; its positive-part mass bound is therefore a valid overlap with the
analytic `Π=0` tail rather than a representational gap.

Positive point samples are diagnostics only and can never set a proof status.
Exact tangency is structural, not a residual tolerance: the inverse-jet
matrix has exact determinant `16 y³(y²−1)^6≠0`, while `F+P=nE_F` and the
declared `C_v`, `U*`, `U_y*` construction give `E=μn`, `E_y=0`, `E_n=μ`.
Interval residuals containing zero are consistency checks on the same
enclosed coefficient solution.  They do not certify a perturbed polynomial.
If any obligation is absent or unresolved, the validator must fail closed and
the global status is `INCONCLUSIVE_NO_FULL_DOMAIN_CERTIFICATE`.

## Static finite-q obligation

With `x=q²`, `Z=W0²`, `t0=mω²y²`, and the uneliminated canonical saddle
Hessian

\[
\begin{pmatrix}a&b&g\\b&d+Zx&h\\g&h&-(t_0+x)\end{pmatrix},
\]

the constrained vector Schur reduction uses
`Dq=a+g²/(t0+x)>0`, `Bq=b+gh/(t0+x)`, and
`Cq=d+Zx+h²/(t0+x)`.  The determinant sign is the quadratic

\[
P(x)=aZx^2+[a(d+Zt_0)+g^2Z-b^2]x
 +[t_0(ad-b^2)+ah^2+g^2d-2bgh].
\]

`Qc=(t0+x)Cq` is also checked as a quadratic.  Half-line classification
must distinguish strict, marginal-at-zero, marginal-at-finite-x,
unstable-band, and unstable-tail cases, while rejecting non-finite inputs.
At `x=0`, the `P0` identity is checked against `K/(9n)`; the actual
canonical `Z=W0²` is never modified to satisfy a threshold.  The analytic
mixing derivative `h` is independently compared to nested differentiation of
the full three-variable saddle, and omitted/wrong-sign `h` controls must be
rejected.  The actual branches additionally require directed positive lower
bounds for every coefficient of `P` and `Qc`, a directed interval containing
zero for `P(0)-Qc(0)K/(9n)`, and a strict directed margin `Z-G_min>0` (equality
at `G_min` is marginal).

## Status and reproducibility

`audit_integrity` describes calculation/coverage integrity.  The separate
`physical_status` remains conditional and has zero empirical weight.  The
CLI prints strict JSON to stdout and never writes files:

```text
python3 -B verification/nvg_nuclear_closure_audit.py
python3 -B -m unittest verification.test_nvg_nuclear_closure_audit -v
```

The result is not a finite-nucleus, finite-N, quantum-RPA, dynamical,
gravitational, observational, or universe-closure calculation.
