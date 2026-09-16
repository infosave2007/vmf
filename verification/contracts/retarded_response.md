# Retarded finite-momentum response contract

This contract freezes `verification/nvg_retarded_response.py`.  It is a
conditional low-energy conserving Hartree response of the same Dirac/Hartree
medium and canonical scalar/Higgs-vector action used by the finite-q static
producer.  It has zero independent empirical weight.  It is **not** full
vacuum RPA, a measured finite-nucleus spectrum, a transport calculation, a
globally positive spectral measure, or an all-plane stability certificate.

## Retarded kernel and subtraction

Natural units are used.  For `z` in the upper half-plane and
`E=sqrt(p^2+M^2)`, `R=sqrt(r^2+M^2)`, the matched zero-density determinant
subtraction is retained as the occupied-positive plus negative-intermediate
kernel

```text
Pi_ij = d/(4 pi^2) integral_0^kF (p dp/q)
        { [F_ij,+]_(Rlo-E)^(Rhi-E) + [F_ij,-]_(Rlo+E)^(Rhi+E) }.
```

For the positive intermediate state use `Rlo=max(EF,sqrt((p-q)^2+M^2))`
when the interval is nonempty; the negative state uses the full kinematic
interval.  With

```text
J1=(log(x-z)+log(x+z))/2
J2=x+z*(log(x-z)-log(x+z))/2
J3=x^2/2+z^2*J1
```

the three primitives are

```text
Fvv,+=[J3+4E J2+(4E^2-q^2)J1]/(2E)
Fss,+=[-J3+(4M^2+q^2)J1]/(2E)
Fvs,+=(M/E)J2+2M J1
Fvv,-=[J3-4E J2+(4E^2-q^2)J1]/(2E)
Fss,-=Fss,+
Fvs,-=-(M/E)J2+2M J1.
```

On the real retarded rim,
`log(x-omega-i0)=ln|x-omega|-i*pi*theta(omega-x)` and the companion
`log(x+omega+i0)` has `+i*pi` for negative omega when `x<|omega|`.  The
radial quadrature is split at kinematic and endpoint-transfer roots; the
declared default tolerances are `epsrel=1e-10, epsabs=1e-6` and the control
run uses the stricter `1e-11,1e-7`; no
finite eta is substituted for the boundary and no absorptive part is clipped.
`eta=.4,.2,.1 MeV` values are convergence controls only.

For `0<omega<q` below pair threshold the independent particle-hole check is

```text
t=q^2-omega^2
Elo=max(M, EF-omega, (q*sqrt(1+4M^2/t)-omega)/2)
Im Pi_vv=d/(16*pi*q) * integral_Elo^EF ((2E+omega)^2-q^2) dE
Im Pi_ss=d*(4M^2+t)*(EF-Elo)/(16*pi*q)
Im Pi_vs=d*M*(EF^2-Elo^2+omega*(EF-Elo))/(8*pi*q).
```

The declared particle-hole edge is
`omega_ph+=sqrt((kF+q)^2+M^2)-EF`; an outside-cut zero or pole is not
silently interpreted as a damped second-sheet mode.

## Conserving closure

For `A_L=hat(q).A`, `exp(-izt+iqx)`, and source `-delta_mu*n`, define

```text
a=1/Pi_vv, b=M_N Pi_vs/Pi_vv,
dF=M_N^2*(Pi_vs^2/Pi_vv-Pi_ss),
c0=dF+Uyy-momega^2*A0^2, h=-2*momega^2*y*A0,
t0=momega^2*y^2, Z=(s W0)^2,
```

and use exactly

```text
H=[[a,b,g,-g*z/q],
   [b,c0+Z*(q^2-z^2),h,0],
   [g,h,-(q^2+t0),z*q],
   [-g*z/q,0,z*q,t0-z^2]].
```

The direct solve is `H u=(1,0,0,0)` and `chi=u[0]`.  Eliminating the vector
block gives `L=t0+q^2-z^2`,

```text
D=a+g^2*(1-z^2/q^2)/L
B=b+g*h/L
C=c0+Z*(q^2-z^2)+h^2*(1-z^2/t0)/L
chi=1/(D-B^2/C).
```

Direct and Schur values are compared off poles.  A zero `Pi_vv`, `L`, or
elimination denominator is explicit singular input, never a clipped
division.  The longitudinal variable and scalar-vector mixing are mandatory;
negative controls intentionally omit them and must change the finite-frequency
prediction.

The raw Hessian is written in mixed physical coordinates and its determinant is
therefore reported with its coordinate-dependent units (`MeV^6` for the
finite-q 4x4 closure, `MeV^4` for the static q=0 3x3 closure), never as a
dimensionless number.  For pole diagnostics use the congruence
`H_hat=S H S/(n0 W0)` with `S=diag(n0,1,W0,W0)` (the q=0 sub-block is
`diag(n0,1,W0)`).  `H_hat` and `det(H_hat)` are dimensionless; the full
determinant identity `det(H)=-t0*L*(D*C-B^2)` is checked independently.  The
serialized `regular`/`direct_status` fields describe numerical solvability;
`physical_stability_assessed=false` is explicit, and no `stable=true` label is
inferred from an invertible complex sample.

At `q=0,z!=0`, number conservation returns zero density and density-scalar
channels; the remote scalar pair channel is retained but is not inverted for
the conserved density source.  At
`q=0,z=0`, the homogeneous grand-canonical static susceptibility is returned
with the fixed-total-`N` ensemble caveat.

## Scope, poles, and numerical evidence

Rows cover Q4 at `n0` and inverse-designed W8 `y*=.90,.93`, `q=100,200 MeV`,
scales `1,1.3`, and frequency fractions `0,.25,.75,1.25` of the live edge
(48 rows).  All rows remain in the output, including complex absorption and
any numerical red status.  The bounded pole scan covers each of the 12
model/q/scale cases on both 33- and 65-node grids over
`[edge*(1+1e-5), .95*q]`.  A root is accepted only with finite `Pi_vv,L,C`,
the full normalized determinant/nullvector, nonzero density overlap, bracket
and per-candidate dual-grid confirmation, a converged positive finite residue
`-1/(d chi^{-1}/d omega)` (central differences with `h=.001` and `.0005` MeV
and a genuine relative denominator are retained), and finite elimination
factors.  Rejected root candidates remain in the payload with explicit
reasons.  A sign change of a Schur singularity, a scalar-decoupled determinant
zero, or an unresolved point is not a pole;
`NONE_RESOLVED_IN_SCANNED_INTERVAL` is not a theorem of absence.

The mathematically subtracted kernel has no arbitrary high-energy cutoff and
can be evaluated at real energies beyond this report's low-energy scope; such
values are not thereby physically supported or a global spectral prediction.
The result includes a numerical-payload fingerprint in addition to independent
source/contract provenance hashes; validation therefore rejects altered rows,
controls, or coverage rather than treating a source hash as numerical proof.
The CLI prints strict JSON and writes no result table, cache, or bytecode.
The numerical tolerances describe quadrature/model consistency, not physical
uncertainty.  W8 alternatives retain their calibration-only status, and no
parameter, damping width, or counterterm is fitted.

## Primary references

- Furnstahl, Piekarewicz & Serot, *Covariant RPA in Effective Hadronic Field
  Theory*, [arXiv:nucl-th/0205048](https://arxiv.org/abs/nucl-th/0205048).
- Kurasawa & Suzuki, *Roles of Antinucleon Degrees of Freedom in the
  Relativistic Random Phase Approximation*,
  [PTEP 2015 113D02](https://doi.org/10.1093/ptep/ptv151).
