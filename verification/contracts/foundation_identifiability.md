# Foundation identifiability and finite-momentum contract

## Control block

This contract freezes a numerical/model-class audit only.  It does not add an
interaction, refit the maintained model, or assign empirical evidence.  The
independent-evidence weight of every item in this scope is **0.0**.

The accepted homogeneous functional is the one in
[`source_complete_action.md`](source_complete_action.md), reduced to the
dimensionless field `y=W/W0`:

```text
epsilon(n,y) = F_Fermi(n, M_N y) + A (y^2-1)^2/4
               + Cv n^2/(2 y^2),
A = lambda W0^4,  Cv = g_omega^2/m_omega^2.
```

Here `n` is natural-unit density in `MeV^3`, `d=4`, and all dimensional
quantities are explicitly carried in the executable passport.  The scalar
stationarity residual is `d epsilon/dy` in `MeV^4`; it is not a residual with
units of `MeV^3`.  Likewise `C_y=d^2 epsilon/dy^2` has units `MeV^4`, while
the dimensional-field Hessian is `C_W=d^2 epsilon/dW^2=C_y/W0^2` in `MeV^2`.
The reported
original point is the live state at `n=n0=0.16 fm^-3`; a positive binding
energy there is a failure of the stated `-16 MeV` saturation target, not a
successful fit.

## Identifiability map

At fixed physical `M_N`, `m_omega`, and `g_omega`, the declared map is

```text
W0 -> s W0,       lambda -> lambda/s^4,
g_s -> g_s/s,     q_phi -> q_phi/s,       s>0.
```

Consequently `A`, `Cv`, `M_N`, and `d` are unchanged.  The homogeneous
energy, pressure, chemical potential, and scalar stationarity/Hessian values
are therefore invariant at fixed physical `n` and fixed `y=W/W0`.  The
dimensional field `W=s W0 y` is not itself invariant.

This is not a symmetry of the full canonically normalised dynamics.  The
radial scalar mass and the coefficient of the `y` gradient term obey

```text
m_sigma^2 = 2 lambda W0^2,  m_sigma -> m_sigma/s,
L_gradient = (W0^2/2)(partial y)^2,  W0^2 -> s^2 W0^2.
```

For a vacuum tree-level scalar exchange with wave number `k` in `MeV`,

```text
K_0(k) = -g_s^2/(k^2+m_sigma^2),
K_s(k) = -(g_s/s)^2/(k^2+(m_sigma/s)^2)
         = -g_s^2/(s^2 k^2+m_sigma^2).
```

The kernels agree at `k=0` and differ at nonzero `k` for `s != 1`.  The
wave-number symbol `k` is deliberately distinct from the dimensionless Higgs
charge `q_phi`; this vacuum exchange is not an in-medium mode calculation.

## Numerical and provenance obligations

The producer must recompute, in a fresh process when invoked as a CLI:

1. the original live `n0` state and necessary vector ceiling;
2. the two-target quartic calibration using only `n0` and `E/A-M_N=-16 MeV`,
   then an independently differentiated `K` negative control;
3. fixed-`y` and independently solved positive stationary rows for
   `s=0.5,1,1.3,2` and off-grid densities;
4. nonzero-`k` exchange values, not only shared parameter combinations;
5. independent Fermi momentum quadratures and a two-precision replay
   (`80` and `110` decimal digits here).

Calibration targets are training inputs and cannot be called held-out
evidence.  A missing, malformed, nonfinite, mutated, or stale value must fail
closed.  The maintained source passport and its seven base plus two derived
records are reused and validated; the blocked renormalisation metadata is not
silently promoted to a fit or an uncertainty.

The eight declared upstream inputs (`W0`, `lam`, `MN`, `momega`, `gomega`,
`hbarc`, `n0_fm3`, and `d`) are a sealed baseline contract.  Before a nominal
result is produced, their decimal values must agree with the live upstream
`INPUTS`, the values actually consumed by `BulkModel` (including the natural-
unit density conversion), and the seven numeric base passport records with
their contracted units.  Any input, consumed-value, passport-value, or unit
drift fails closed rather than being relabelled as the baseline.

The CLI prints JSON only and performs no writes.  The audit does not establish
finite-nucleus, transport, isovector, gravitational, in-medium, or empirical
validation, and it makes no universal-theory or free-energy-transition claim.
