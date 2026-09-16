# Local nuclear isospin-jet contract

This contract records the P1 sealed calculation.  It is a local homogeneous
diagnostic, not a new fundamental NVG action and not a finite-nucleus or
transport likelihood.

## Functional and parameter roles

Let `n=nn+np`, `delta=(nn-np)/n`, and give each species spin degeneracy two.
The read-only source-complete `BulkModel` supplies

\[
 E_0(n,y)=E_{\rm kin}(n,y)+U(y)+C_v n^2/(2y^2),
 \qquad M^*=M_N y.
\]

The separately declared conventional effective extension is

\[
 E_\rho=C_\rho n^2\delta^2/8,\qquad C_\rho=g_\rho^2/m_\rho^2\geq0.
\]

`n0`, binding, `Ktarget`, the residual-y grid and `J=32 MeV` are declared
design/calibration inputs.  `J=31` and `J=34` are conditional reference
controls.  The inverse polynomial coefficients are generated live by the
existing `inverse_potential_jet`; no coefficient or result table is an input.

The executable protocol is sealed to `n0=.16 fm^-3`, binding `-16 MeV`,
`Ktarget=240 MeV`, residual `y={.60,.75,.85,.90,.93}`, reference
`J={31,34} MeV` plus nominal `J=32 MeV`, design sensitivity
`K={220,260} MeV`, direct-check `y={.75,.90}`, and working precisions
`{80,110}`.  Lists are order-independent but must be nonempty, unique, and
contain exactly these semantic values; nonfinite/bool values and altered
nominal scalars fail closed.

## Accepted local identities

At the symmetric stationary point define

\[
 k=(3\pi^2n/2)^{1/3},\quad E_F=\sqrt{k^2+M^{*2}},\quad
 T={k^2\over6E_F},\quad u={k^2\over E_F^2},
\]

and let `C=E0_yy`, `B=E0_ny`, `D=E0_nn` at fixed `y`.  Then

\[
 J=T+C_\rho n/8,
 \quad L=(2-u)T+3nT_y(-B/C)+3C_\rho n/8,
\]

\[
 T_y=-T(1-u)/y,\quad K_{\rm fr}=9nD,\quad
 K=9n(D-B^2/C),\quad y_n=-B/C,
\]

\[
 S_4^{\rm kin}={k^2(4+3u+3u^2)\over648E_F},\qquad
 S_4=S_4^{\rm kin}-{nT_y^2\over2C}.
\]

The mixing-eliminated form is used only when `B` and `Kfr-K` are resolved:

\[
 Q=L-3J+(1+u)T,\qquad
 S_4=S_4^{\rm kin}-{Q^2\over2(K_{\rm fr}-K)}.
\]

At exact or near degeneracy, the direct Hessian expression remains the
reported value and the eliminated expression is explicitly `null`; a
`0/0` is never converted to zero.  A finite `C<=0`, nonfinite quantity,
negative `C_rho`, invalid density/field, or inconsistent `Kfr-K<0` fails
closed.

The rho contribution before scalar re-equilibration is

\[
 \epsilon_\rho=p_\rho=C_\rho n^2\delta^2/8,
 \quad \mu_n^\rho=C_\rho n\delta/4,\quad
 \mu_p^\rho=-C_\rho n\delta/4,\quad
 \mu_{n\,\mathrm{fixed}\,\delta}^\rho=C_\rho n\delta^2/4.
\]

Direct two-species quadrature and finite-difference re-stationarization are
cross-checks only.  `L` and `S4` are held out of the parameter construction.
Negative `S4` is retained as a Taylor-coefficient sign result and is not by
itself an isospin-instability claim when local `C>0` and `J>0` hold.

The result boundary requires exactly `2` controls, `5` inverse rows, `15`
conditional-`J` rows, and `2` sensitivity rows, with both direct checks on
each precision state.  Missing/duplicate rows, null or nonfinite diagnostics,
or a residual outside the declared finite-difference/quadrature numerical gate
are errors; a globally rejected inverse potential remains a visible negative
physical control and is not relabelled as a numerical failure.  The 80- and
110-digit structures are compared only after both have independently passed
these gates.  These are computational checks, not physical uncertainties or
empirical pass/fail tests.
