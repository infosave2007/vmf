# Finite-q nonlocal response contract

This contract freezes the bounded calculation in
`verification/nvg_nonlocal_response.py`.  It is a conditional, real static
medium response and carries zero independent empirical weight.  It is not a
finite-frequency response, full vacuum RPA, finite-nucleus calculation,
transport model, or experimental validation.

## Adopted prescription

Use natural units and the same zero-temperature symmetric Hartree action as
the live `BulkModel`, with the zero-density determinant at the same fluctuating
mass and vector shift subtracted.  This is named
`SUBTRACTED_MEDIUM_DIRAC_STATIC`.  The subtraction leaves the occupied-to-
negative-energy intermediate-state contribution needed by the fixed-density
mass curvature.  It does not add a fitted counterterm or a physical
positive-spectral-measure claim; unresolved vacuum derivative coefficients
remain outside the truncated action.

For `E_p=sqrt(p^2+M^2)`, `E_r=sqrt((p+q)^2+M^2)`,
`Lambda_s=(1+s H_D/E_p)/2`, and vertices `Gamma_v=1`, `Gamma_s=beta`, the
real static prescription is

```text
Pi_ij(q) = (d/2) PV integral d^3p/(2*pi)^3 sum_{s,t=+,-}
           [f_s(p)-f_t(p+q)]/[t E_r-s E_p]
           Tr[Lambda_s(p) Gamma_i Lambda_t(p+q) Gamma_j].
```

The implementation uses the exact one-dimensional radial form for `q>0`,
with `L=log(abs((q+2p)/(q-2p)))`:

```text
Pi_vv = d/(2*pi^2) integral_0^kF dp p^2/E
        [1 + (E^2-q^2/4)/(p*q) L],
Pi_ss = d/(2*pi^2) integral_0^kF dp p^2/E
        [-1 + (M^2+q^2/4)/(p*q) L],
Pi_vs = d*M/(2*pi^2*q) integral_0^kF dp p L.
```

At `q=0`, with `N0=d*kF*EF/(2*pi^2)` and
`I=d/(2*pi^2)*integral_0^kF p^4/E^3 dp`, the exact limit is

```text
Pi_vv=N0,
Pi_vs=(M/EF)N0,
Pi_ss=(M/EF)^2*N0-I.
```

The `I` term is mandatory.  A particle-hole-only scalar kernel gives the
wrong `Pi_ss(0)` and is retained as a negative control, never as a tunable
alternative.  The logarithmic row at `p=q/2` is integrable: the code
subtracts the smooth logarithmic coefficient at that point and restores its
analytic logarithm integral.  No width, clipping, interpolation, or singular
row deletion is permitted.  `q=2*kF` is treated as an endpoint logarithm.

## Legendre bridge and comparison

For each live kernel set,

```text
a=1/Pi_vv,
b=M_N*Pi_vs/Pi_vv,
d_F=M_N^2*(Pi_vs^2/Pi_vv-Pi_ss),
d0=d_F+U_yy-m_omega^2*A0^2,
g=g_omega,
h=-2*m_omega^2*y*A0,
t0=m_omega^2*y^2,
Z_s=(s*W0)^2.
```

In the implementation a scaled candidate already carries `W0=s*W0`; the
candidate's dimensional `W0` is therefore used with a unit relative factor.
This single-owner rule prevents applying the scale twice.  The public helper
`finite_q_coefficients(model, state, q, scale)` retains its meaning of applying
the explicitly supplied relative scale to the provided model.  A live
regression checks `Z(1.3)/Z(1)=1.3^2` and compares the corresponding finite-q
direct Hessian solve.

The same constrained density/y/A0 Hessian is used for both predictions:

```text
H = [[a,b,g], [b,d0+Z_s*q^2,h], [g,h,-(t0+q^2)]].
```

`chi_nonlocal` uses the finite-q irreducible coefficients.  `chi_TF` freezes
only that irreducible fermion kernel to the exact q=0 values while retaining
the same finite q in the boson propagator and canonical scalar gradient.  A
positive susceptibility is reported only when scalar curvature and density
Schur complement are positive; unstable and singular rows remain explicit.

## Declared live coverage and controls

The backgrounds are original Q4 at `n0`, plus the retained inverse-designed
W8 alternatives `y*=0.90` and `0.93` at `n0`.  W8 trains on
`n0=0.16 fm^-3`, binding `-16 MeV`, pressure zero and `K=240 MeV`; those are
calibration alternatives, not observations.  Scales are `1` and `1.3`, and
each background/scale has `q/kF=0,0.1,0.5,1,1.5,2,2.5` plus physical
`q=100,200 MeV`, for 54 retained rows.

The producer computes with 40 decimal digits and repeats live 27 distinct
kernel points and resulting responses at 55 digits; the declared numerical
relative tolerance is `1e-9`, not a physical uncertainty.  At q=0 it compares
the polarization bridge to independent fixed-(n,y) derivatives of the live
`BulkModel` Fermi energy and to `1/state.mu_prime`; it also checks continuity
at `q/kF=1e-4`, an independent mixed-kernel quadrature and the nonrelativistic
3-D Lindhard shape

```text
F(x)=1/2+(1-x^2)/(4*x)*log(abs((1+x)/(1-x))), x=q/(2*kF),
F(0)=1, F(1)=1/2.
```

The nonrelativistic control sets `M/kF=1e4` and checks
`q/kF=0.5,1,2,2.5` against `N0*F(x)` to `1e-6`, including the exact `x=1`
branch.  The analytic-shape controls explicitly retain `x=.25,1,1.25` and
the small-input regression `x=10^-20`; extreme small/large x use convergent
forms rather than accepting a cancellation artefact.

The command prints strict JSON only and does not consume saved result tables.
The canonical parameter guard is reused from the maintained foundation
passport.  Agreement among these controls is numerical/model consistency,
not empirical confirmation or proof of a complete NVG theory.

## Primary references

- R. J. Furnstahl, J. Piekarewicz, B. D. Serot, *Covariant RPA in Effective Hadronic Field Theory*, [arXiv:nucl-th/0205048](https://arxiv.org/abs/nucl-th/0205048), especially the same-background subtraction and particle-hole/negative-energy separation.
- H. Kurasawa and T. Suzuki, *Roles of Antinucleon Degrees of Freedom in the Relativistic Random Phase Approximation*, [PTEP 2015 113D02](https://doi.org/10.1093/ptep/ptv151), especially the subtraction and spectral caveats.
- J. Lindhard, *On the Properties of a Gas of Charged Particles*, [MFM 28 (1954)](https://gymarkiv.sdu.dk/MFM/kdvs/mfm%2020-29/mfm-28-8.pdf), for the nonrelativistic static control only.
