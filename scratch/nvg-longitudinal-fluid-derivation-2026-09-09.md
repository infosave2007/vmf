# Local longitudinal NVG matter: conditional fluid-closure audit

Date: 2026-09-09. Status: bounded analytic research result and future work item,
not a stability proof of the full NVG action, its quantum fermions, or NVG plus
cuscuton gravity. No couplings or baseline parameters are changed.

## Scope and assumptions

Use the accepted source-complete action in
`verification/contracts/source_complete_action.md`, signature (+---), and
natural units. Write the fermion coupling as `g = gomega`, the phase charge as
`qphi`, the fermion mass as `m = gs W`, and the vector mass squared as
`M2 = qphi^2 W^2`. Work in unitary gauge at a homogeneous stationary state with
`n > 0`, `W > 0`, no spatial vector background, and

\[
 A_0=\frac{gn}{M^2},\qquad
 U'(W)+g_s n_s-q_\phi^2 W A_0^2=0.
\]

The local metric is frozen Minkowski. The fermions are replaced by an
isentropic perfect fluid whose local rest-frame energy is the zero-temperature
Fermi functional `epsilon_F(n,gs W)`, with degeneracy `d=4` for the original
baseline. Both the scalar and longitudinal vector remain dynamical; the
temporal vector is a Gauss constraint. A local cosmological application also
needs wavelength/time scales short relative to background variation. This
calculation does not integrate out lapse, shift, or cuscuton perturbations.

Define derivatives of the *kinetic Fermi functional*, not the total vector-
eliminated energy:

\[
 \mu_F=E_F=\sqrt{k_F^2+m^2},\qquad
 a=\epsilon_{F,nn}=\frac{k_F^2}{3nE_F}>0,\qquad
 b=\epsilon_{F,nW}=\frac{g_s m}{E_F},
\]
\[
 c=2q_\phi^2WA_0=\frac{2gn}{W},\qquad
 s=U''(W)+g_s^2\left(\frac{\partial n_s}{\partial m}\right)_n
       -q_\phi^2 A_0^2.
\]

Here `a` has mass dimension -2, `b` is dimensionless, and `c,s,M2` have mass
dimension 2. Importantly, `s` is the scalar curvature at fixed `A0`; it is
**not** the scalar curvature after Gauss elimination.

## Quadratic action and exact dispersion matrix

For a longitudinal displacement `xi`, put `delta n = -n partial_x xi` and
`v = dot xi`. Let `w=delta W`, `phi=delta A0`, and `A=delta A_L`, where `A` is
the ordinary physical spatial component. Up to boundary terms the quadratic
Lagrangian density is

\[
\begin{aligned}
 \mathcal L_2={}&\frac{n\mu_F}{2}\dot\xi^2
 -\frac a2(\delta n)^2-b\,\delta n\,w
 +\frac12\dot w^2-\frac12(\partial_x w)^2-\frac s2w^2\\
 &+\frac12(\dot A+\partial_x\phi)^2
 +\frac{M^2}{2}(\phi^2-A^2)
 +(cw-g\delta n)\phi+gn\dot\xi A.
\end{aligned}
\]

The fluid inertia here is `n mu_F`. The extra vector contribution to the
long-wavelength inertia is generated dynamically; inserting the full chemical
potential in this Lagrangian would double count it.

With `exp(-i omega t+i k x)`, `Omega=omega^2`, `z=k^2`, and
`L=omega k delta A_L`, the four equations at nonzero `k` are

\[
 \begin{pmatrix}
 \mu_F\Omega/n-az&-bz&-gz&g\\
 b&z+s-\Omega&-c&0\\
 -g&c&z+M^2&-1\\
 -g\Omega&0&\Omega z&M^2-\Omega
 \end{pmatrix}
 \begin{pmatrix}\delta n\\w\\\phi\\L\end{pmatrix}=0.
\]

The determinant is cubic in `Omega`. The matrix avoids division by `Omega`
or `M2-Omega`; it can therefore be used to test threshold modes without
deleting a root. The `k=0` statements below are continuous `k -> 0` limits;
the strictly homogeneous conserved-number perturbation needs separate
treatment. At an exactly zero-frequency mode, the original variables and
equations should also be checked because `L=omega k A` is not invertible.

## Positive-energy certificate for every mathematical wave number

The canonical Hamiltonian before minimizing electric momentum is

\[
\begin{aligned}
 \mathcal H_2={}&
 \frac{(P_\xi-gn A)^2}{2n\mu_F}
 +\frac{\Pi_w^2}{2}+\frac{\Pi_A^2}{2}
 +\frac{(\partial_x\Pi_A+J)^2}{2M^2}
 +\frac{M^2A^2}{2}\\
 &+\frac12\left[a(\delta n)^2+2b\delta n w
       +(\partial_x w)^2+s w^2\right],
 \qquad J=g\delta n-cw .
\end{aligned}
\]

Completing the electric-momentum square for each Fourier mode leaves the
static density/scalar Hessian

\[
 K(z)=\begin{pmatrix}
 a+g^2/(z+M^2)&b-gc/(z+M^2)\\
 b-gc/(z+M^2)&z+s+c^2/(z+M^2)
 \end{pmatrix}.
\]

Because the remaining momentum squares are positive when `n mu_F>0` and
`M2>0`, positive definiteness of `K(z)` is a sufficient positive-Hamiltonian
certificate for the full longitudinal **fluid** system, including its dynamic
scalar and vector. At each fixed nonzero `k`, it excludes exponentially
growing normal modes in this constant-coefficient conservative system.

Since `K_nn>0`, the test reduces to

\[
 (z+M^2)\det K(z)=a z^2+P_1z+P_0,
\]
\[
 P_1=a(M^2+s)+g^2-b^2,
\qquad
 P_0=M^2(as-b^2)+ac^2+g^2s+2bgc.
\]

For strict positivity at all `z>=0`, the exact criterion is

\[
 P_0>0\quad\text{and}\quad
 \left[P_1\geq0\ \text{or}\
       P_0-\frac{P_1^2}{4a}>0\right].
\]

Equality gives a marginal mode, not a strict stability certificate. This is
an all-wave-number statement about the stated mathematical fluid closure;
it is not a claim that hydrodynamics is physically valid at arbitrarily high
wave number.

## Independent thermodynamic and mode checks

Let

\[
 E(n,W)=\epsilon_F(n,g_sW)+U(W)+\frac{g^2n^2}{2q_\phi^2W^2}.
\]

Exactly,

\[
 E_{nn}=a+g^2/M^2,\quad
 E_{nW}=b-gc/M^2,\quad
 E_{WW}=s+c^2/M^2,\quad
 \mu_{\rm total}=\mu_F+g^2n/M^2.
\]

Direct symbolic differentiation of the determinant at `(Omega,z)=(0,0)`
gives the acoustic branch

\[
 \left.\frac{d\Omega}{dz}\right|_0
 =-\left.\frac{\partial_z\det\mathcal M}
                  {\partial_\Omega\det\mathcal M}\right|_0
 =\frac{n}{\mu_{\rm total}}
       \left(E_{nn}-\frac{E_{nW}^2}{E_{WW}}\right).
\]

This is exactly the independently derived relaxed thermodynamic sound speed
in `source_complete_scaling_saturation_audit.py`, not `3P/epsilon` and not a
formula with `mu_F` alone in its denominator.

The three `z -> 0` roots are

\[
 \Omega=0,\qquad \Omega=E_{WW},\qquad
 \Omega=M^2+\frac{g^2n}{\mu_F}.
\]

Setting `g=b=c=0` independently factorizes the modes into
`Omega=n a z/mu_F`, `Omega=z+s`, and `Omega=z+M2`.

These determinant, Hessian-polynomial, zero-wave-number, and decoupling
identities were checked by SymPy on system Python, not by fitting numerical
eigenvalues to desired outcomes.

## Original-parameter numerical spot check

Using `BulkModel` from `verification/source_complete_scaling_saturation_audit.py`
with 80-digit mpmath arithmetic and **unchanged** inputs
`W0=859, lambda=1.05, MN=939, momega=782.6, gomega=10.12, d=4`, solve its
original stationary scalar equation at the following eight densities. `P1`
is dimensionless; the displayed `P0/MN^2` is dimensionless.

| n/n0 | W/W0 | P1 | P0/MN^2 | acoustic cs^2 |
|---:|---:|---:|---:|---:|
| 0.001 | 0.999998990848 | 533.005725893 | 394.117965882 | 0.000479096693300 |
| 0.01 | 0.999989936877 | 301.352153365 | 278.790479693 | 0.00337774023742 |
| 0.1 | 0.999901743171 | 193.508686487 | 225.113500213 | 0.0265914811432 |
| 1 | 0.999233898913 | 142.838562994 | 200.128932537 | 0.194260298192 |
| 2 | 0.998936965412 | 133.601555938 | 195.980955773 | 0.319019189620 |
| 4 | 0.999718045424 | 126.211587261 | 193.731126907 | 0.473834478116 |
| 10 | 1.01192977294 | 118.865962136 | 198.036860709 | 0.644425960146 |
| 100 | 1.41215240004 | 111.295317121 | 408.732131080 | 0.462754641537 |

At each of these eight stationary states, `P1>0` and `P0>0` certify all
mathematical `z>=0` in the fluid model. These eight states do **not** certify
the entire continuous density axis. Comparing the three independently
constructed entries of `K(0)` with the existing energy derivatives
`D, B_y/W0, C_y/W0^2` gave maximum relative difference below `1.2e-80`.

## Required negative controls for a future executable audit

1. Keep both the `c` mixing and `-qphi^2 A0^2` contribution in `s`. Dropping
   either must fail the zero-wave-number Hessian match to the accepted energy.
2. Verify the factorized `g=b=c=0` system and the acoustic derivative identity;
   do not accept root clipping or discarding complex roots.
3. Test a formal dimensionless counterexample with
   `a=b=g=M2=n=mu_F=1`, `c=2`, `s=-69/20`. Its zero-wave-number roots are
   `0, 11/20, 2`, and `K(0)` is positive, but
   `(z+M2)detK=(20z^2-49z+2)/20` is negative over a finite interval.
   At `z=1` the exact determinant has roots approximately
   `-0.241206070638, 1.22597946170, 4.56522660894` in `Omega`.
   Thus testing only relaxed compressibility and zero-wave-number scalar
   curvature misses a genuine finite-wave instability in this formal control.
   These artificial coefficients are not a new NVG parameter set.
4. Treat `n=0` separately because the displacement inertia vanishes. At fixed
   `n>0`, `W=0` is already outside the finite-energy accepted branch.

## Closure boundary and independent literature

Zero temperature does not establish local hydrodynamics. A collisional fluid
description requires the corresponding small `omega tau_coll` and
`k vF tau_coll` regime; a collisionless Fermi system instead has Fermi-surface
distortions, zero sound, and a particle-hole continuum. These distinctions are
derived using kinetic theory in
[Watabe, Osawa and Nikuni, *Zero and First Sound in Normal Fermi Systems*](https://arxiv.org/abs/0910.3283)
and [Belitz and Kirkpatrick, *Soft modes in Fermi liquids at arbitrary temperatures*](https://arxiv.org/abs/2112.14246).

The relevant next calculation is a scalar/vector Vlasov or RPA response with
the NVG-specific `W`-dependent vector mass retained. A primary relativistic
example is [Rabhi et al., *Collective modes in relativistic cold asymmetric
nuclear matter within the covariant Vlasov approach*, equations 6–24](https://arxiv.org/html/2411.06960v1).
Its detailed Walecka-model formulas cannot be copied unchanged into NVG.

There is a useful distinction for homogeneous cosmology: inside semiclassical
collisionless quasiparticle theory, an initially isotropic Fermi sphere keeps
its comoving momentum distribution in homogeneous `W(t), A0(t)` and FLRW.
Thus homogeneous `n proportional to a^-3` and the instantaneous Fermi
integrals need not rely on collisions. This observation does not remove the
quantum closure assumption: varying mass/geometry may produce particle and
antiparticle excitations or vacuum polarization, and conserved net baryon
number alone does not preserve the assumed quasiparticle population.

Finally, a frozen positive matter Hamiltonian does not settle coupled
gravitational/cuscuton constraints, Jeans-scale modes, time-dependent
amplification, strong coupling, quantum corrections, or microcausality. The
previous one-free-canonical-scalar `w=1` cuscuton perturbation formulas are
not used anywhere in this derivation.

## Optional homogeneous tracking diagnostic

For a scalar stationary branch `W_*(n)` with `E_WW>0`,
`W_*'= -E_nW/E_WW`. In the homogeneous scalar equation, the forcing of
`delta W=W-W_*(n)` is, to linear order about the prescribed branch,

\[
 \ddot{\delta W}+3H\dot{\delta W}+E_{WW}\delta W
 =3n\dot H\,W_*'-9H^2n^2 W_*''.
\]

Consequently `H=0` at a bounce does not imply perfect adiabatic following:
the first forcing term survives. This diagnostic needs small forced
displacement, slowly varying oscillator parameters, and control of the
initial free oscillation. It does not replace solving the dynamic scalar
equation or testing fermionic quantum adiabaticity.
