# Source-complete stellar prediction artifact

Status: **STABLE_BRANCH_TERMINATED_AT_FIRST_MASS_MAXIMUM**.  Every row is `MATHEMATICAL_ONE_COMPONENT_SEQUENCE`, evidence weight 0.  The physical neutron-star claim is blocked by `BLOCKED_MISSING_BETA_EQUILIBRIUM_CHARGE_NEUTRALITY_LEPTONS_CRUST_AND_HIGH_DENSITY_COMPLETION`.

## Scope and gates

    The direct source action is evaluated on the closed domain `0 <= n/n0 <= 10` with degeneracy 4, exact vacuum enthalpy surface `h=0`, and no crust, beta equilibrium, leptons, CSS branch, clipping, extrapolation, or invented composition.  The upstream source digest is `5d39d3d07a3f00d2fcbe853ad27359b34a86d853c842ba01861196c1aaa517a8`.  Causality/stability and Hilbert/Legendre pressure gates are strict; invalid states fail closed.

Primary rows use DOP853 in enthalpy; independent pressure-coordinate RK4 uses actual uniform steps `0.05, 0.025, 0.0125` km and an exact `P=0` terminal chart. The source recurrence is evaluated at `rho=sqrt(K_CONV)*r_km`, with mass and pressure converted at the same boundary. Central grid: `10^(-3+j/32)`, j=0..128. First mass maximum: 2.742549852 M_sun at n_c/n0=2.485075342, R=19.673225 km. All 18 pressure trajectories retain an independent `N={16,32,64}` terminal chart and adaptive oracle. Three predeclared `(x,dr)` keys additionally carry independently replayed numerical interval diagnostics. These diagnostics do not establish a rigorous global error bound or correct rounding. Unresolved binary64 convergence orders are reported as null.

The fixed parameters also fail the nuclear saturation comparison discussed in [the physical model assessment](../SOURCE_COMPLETE_MODEL_RU.md). Repairing the integrators does not resolve that discrepancy or establish a physical neutron-star model.

## Observational context

The serialized records are context-only (`comparison_status=BLOCKED_MODEL_NOT_PHYSICAL_NS`, evidence weight 0).  No fit, pass, or fail claim is made.  Sources: Fonseca et al. [arXiv:2104.00880](https://arxiv.org/abs/2104.00880); Salmi et al. [arXiv:2406.14466](https://arxiv.org/abs/2406.14466); Dittmann et al. [arXiv:2406.14467](https://arxiv.org/abs/2406.14467); Miller et al. [arXiv:1912.05705](https://arxiv.org/abs/1912.05705); Abbott et al. [arXiv:1805.11579](https://arxiv.org/abs/1805.11579) and [arXiv:1805.11581](https://arxiv.org/abs/1805.11581).

The artifact is deterministic JSON plus this report within the pinned runtime. Wall-clock telemetry is excluded from scientific identities. Numerical convergence and direct-equation consistency checks do not constitute empirical confirmation or a rigorous error theorem.
