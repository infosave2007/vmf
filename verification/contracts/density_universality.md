# Density-tail and calibration-null audit contract (P1/I1)

This contract freezes the calculation implemented by
`verification/nvg_density_universality_audit.py`.  It is a mathematical and
conditional zero-temperature bulk audit with `evidence_weight=0`; it is not a
new interaction, an empirical fit, or an EOS-validity certificate.

## Functional, units, and assumptions

The only production functional is the maintained homogeneous source-complete
one, written in the dimensionless field `y=W/W0`:

\[
 E(n,y)=F_F(n,g_sW_0y)+U(y)+C_v n^2/(2y^2),
\qquad
 E_y=M_N n_s+U_y-C_vn^2/y^3.
\]

Here `n` is natural `MeV^3`, `U`, `epsilon`, and the vector/potential terms
are `MeV^4`, `W0=859 MeV`, `M_N=939 MeV`, `hbar*c=197.3269804 MeV fm`,
degeneracy `d=4`, and `n0=.16 fm^-3`.  `C_v=g_omega^2/m_omega^2` and all
other couplings are inherited unchanged from the maintained
`BulkModel`.  The inverse W8 designs `y*=.90,.93`, binding `-16 MeV`, and
`K=240 MeV` are declared calibration inputs, not validation data.

For the same functional in the dimensional field,
\[
 \epsilon(n,W)=F_F(n,g_sW)+U_W(W)+{G n^2\over2W^2},
 \qquad W=W_0y,\qquad G=C_vW_0^2.
\]
Thus if `U_W(W)~a_WW^p`, then `U(y)~a_yy^p` with the unambiguous relation
`a_y=a_WW0^p`; `a_W` and `a_y` are not independent physical parameters.

The tail theorem assumes positive constant couplings, `n>0,y>0`, a locally
stable stationary branch, `U_W(W)~a_WW^p` with `a_W>0`, and matching
derivative asymptotics sufficient to differentiate the expansion. A global
endpoint claim additionally requires a lower-bounded, coercive, lower
semicontinuous functional at every accessible endpoint. In particular, the
positive vector term controls `W->0+` only when no negative singularity wins:
`U_W=a_WW^p-b/W^3`, with `b>0` in `MeV^7`, makes `E->-infinity` at that endpoint, despite its positive
large-W tail. The computation reports a sign-changing positive root and a
finite sign-change coverage diagnostic, not all positive roots.

## Frozen asymptotic claim

For `p>4`, the vector/potential balance gives

\[
 y_{as}=\left({C_v\over pa_y}\right)^{1/(p+2)}n^{2/(p+2)},
\quad
 \epsilon_{as}=\left(1+{p\over2}\right)a_y
 \left({C_v\over pa_y}\right)^{p/(p+2)}n^{2p/(p+2)}.
\]

Equivalently, with `W=n^(2/(p+2))z` and division by
`n^(2p/(p+2))`, the leading objective is
`Phi_p(z)=a_W z^p+G/(2z^2)`. It is coercive at both ends and, for `p>4`,
`Phi_p''=p(p-1)a_Wz^(p-2)+3Gz^-4>0`; hence its positive minimizer
`z*=(G/(pa_W))^(1/(p+2))` is unique and nondegenerate. Uniform decay of
the potential remainder and its needed derivatives then gives a smooth
branch by the implicit-function theorem, which is the derivative regularity
needed below rather than an inference from a fitted logarithmic slope.

The calculated subleading powers are
`W/k_F ~ n^((4-p)/(3(p+2)))`,
`F/epsilon_as ~ n^(2(4-p)/(3(p+2)))`, and the matter contribution to the
field equation divided by the vector term is
`n^(4(4-p)/(3(p+2)))`.  Thus, formally,

\[
 w_\infty={c_{\rm th}^2{}_{\infty}\over c^2}={p-2\over p+2},\quad
 {U\over\epsilon}\to {2\over p+2},\quad
 {V\over\epsilon}\to {p\over p+2},\quad
 {\epsilon-3P\over\epsilon}\to {2(4-p)\over p+2}.
\]

For `0<p<=4`, the rescaling `W=n^(1/3)z` gives
`Phi_<=4(z)=phi(g_s z)+G/(2z^2)+1_{p=4}a_Wz^4`, where
`F_F(n,m)=n^(4/3)phi(m/n^(1/3))`. The vector term diverges at `z->0+`,
while the Fermi term (for `g_s>0`) and, at `p=4`, the quartic term grow at
`z->infinity`; the sum is coercive. Its positive second derivative (including
the vector/quartic contributions) makes this limiting objective strictly
convex with a unique nondegenerate positive minimum. Uniform `C^r` remainder
bounds again provide branch derivative regularity. Thus the formal limit is
`w=c_th^2/c^2=1/3`; the kinetic-free value is a negative control, not a
replacement calculation.

The compact within-class result for `p>0` is
`w_infty=lim(c_th^2/c^2)=max(1/3,(p-2)/(p+2))`. Here
`c_th^2=c^2(dP/depsilon)_eq` is an equilibrium thermodynamic derivative,
not a certified microscopic mode speed or an RPA result. This is a
mathematical statement for the fixed homogeneous mean-field class, not a new
law and not removal of the mean-field approximation. Running couplings,
`G=0`, non-power tails, negative leading coefficients, new species, or a
different branch are outside this contract. The exact formula controls
include `p=2,4,6,8,16`; in particular cold `p=2` retains the Fermi/vector
`1/3` limit and rejects the naive power-tail value zero.

## Calibration-null deformation

For each W8 target, the frozen deformation is

\[
 \Delta U(y)=\eta (y^2-1)^4(y^2-y_*^2)^4,
 \qquad \eta=a_4>0,
\]

where `a4` is freshly reconstructed as the positive leading coefficient of
the corresponding W8 polynomial.  It is nonnegative for every real `y`, has
degree 16 in `y`, and its first four jets—orders `0,1,2,3`, not four nonzero
derivatives—vanish exactly at both `y=1` and `y=y*`.  Therefore the local
saturation energy, pressure, chemical potential, scalar curvature, and `K`
agree at the declared point, while the formal tail class changes from `p=8`
to `p=16`. The positive U16 is a calibration-null counterexample, not an
adopted physical repair; the original quartic and calibrated W8 variants are
different theories even where selected local data coincide.
The first local reduced-energy discriminator is also recorded:

\[
 \Delta\epsilon''''(n_0)=\Delta U''''(y_*)
 \left({dy/d\ln n\over n_0}\right)^4,
 \quad
 \Delta U''''(y_*)=384\eta y_*^4(1-y_*^2)^4.
\]

This is a model diagnostic, not a measured coefficient.  With
`x=(n-n0)/n0` and `e=epsilon/n`, its local form is
`Delta e=Delta Z*x^4/1944+O(x^5)`; it concerns a small neighborhood only and
does not imply a 10% density-shift estimate, experimental measurability,
agreement over a density interval, finite droplets, spectra, or observations.

## Numerical protocol and labels

The primary matrix contains the original quartic, W8 `.90/.93`, and U16
`.90/.93` cases at `n/n0=10^2,10^4,10^8,10^12,10^18,10^24,10^36`.
Roots use logarithmic positive bracketing and bisection on the normalized
stationarity residual.  Every row reports `y`, energy, pressure, `mu`, the
relaxed `(C,B,D)` Hessian and `c_s^2`, trace, field response, component
fractions, asymptotic ratios, finite sign-change coverage, and fixed
`h=10^-5,5*10^-6` log-density derivative checks.  The primary calculation
uses 70 decimal digits.  Independent reconstructions at 110 digits cover
both high-density controls for all five cases and all four W8/U16 calibration
entries; every reported field agrees relatively within `10^-25`.

The final `10^36` markers use absolute `10^-5` limits for `P/epsilon` and
`c_s^2`, and relative `10^-4` limits for the field and energy ratios.  A
failed marker remains a convergence diagnostic; no grid or coefficient is
changed after seeing it.  Finite pressures/curvatures and positive `C,mu` are
not claims of arbitrary-density EFT validity.

For canonical dynamics in unchanged spatially flat GR with `Lambda0=0`, the
full NEC combination is
`rho+P=Wdot^2+n*E_F+C_v*n^2/y^2>0`; hence `dot H<0` and no
contraction-to-expansion bounce follows even without adiabatic tracking. The
static identity `epsilon+P=n*mu`, with `mu=E_F+C_v*n/y^2`, is only its
`Wdot=0` equilibrium special case. The nuclear calibration modulus `K=240
MeV` is distinct from the spacetime Kretschmann scalar
`mathcal K=R_{mu nu rho sigma}R^{mu nu rho sigma}`. In flat FLRW,
`R=8*pi*G_N*(rho-3P)` and
`mathcal K=12*((Hdot+H^2)^2+H^4)`, so
`mathcal K/((8*pi*G_N)^2*rho^2)=((1+3w)^2+4)/3`. A normalized trace
tending to zero is not absolute `R->0`; exact radiation `R=0` still leaves
`mathcal K` divergent when `rho` diverges. No exact `R=0` claim is made for
the original quartic.

The independently checked conditional tracking proxies for `p>4`, assuming
equilibrium-energy dominance and minimum following, are
`Wdot=-(6/(p+2))*H*W`,
`T_radial/epsilon_eq~6*W^2/((p+2)^2*M_P^2)`,
`C_W/H^2~6*p*M_P^2/W^2`, and
`C_W~p*(p+2)*a_W*W^(p-2)`. They are warnings, not physical mode
frequencies or new simulations; contraction also grows `W` and gives
antifriction for `H<0`. The shared static `2V=pU` and oscillatory
`2<T>=p<U>` virial algebra is not dynamical equivalence; see [Turner
(1983)](https://doi.org/10.1103/PhysRevD.28.1243) and
[Boyle--Caldwell--Kamionkowski (spintessence)](https://arxiv.org/abs/astro-ph/0105318).
Prior art for ordinary-density calibration with divergent high-density
predictions is [Mueller--Serot (1996)](https://arxiv.org/abs/nucl-th/9603037);
it does not establish this particular polynomial or exponent.

The CLI is strict JSON and writes nothing:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -B verification/nvg_density_universality_audit.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=verification python3 -B -m unittest verification.test_nvg_density_universality_audit -v
```

No result table, cache, public manifest, article, gravity extension, finite-N
producer, or empirical input is consumed or modified by this surface.

The focused package depends on `mpmath` and the shared `BulkModel` and
`inverse_potential_jet` APIs in `source_complete_scaling_saturation_audit.py`.
It does not invoke that module's separate upstream audit CLI, which requires
`source_complete_solution_audit.py` and is outside this package's standalone
verification surface. No saved upstream result is needed by the two commands.
