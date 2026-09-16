# Prediction-contract adjudication 1

## Control block

- Status: **FINAL — BIND_MATHEMATICAL_STELLAR_SEQUENCE; BIND_FIXED_VACUUM_SPECTRUM; CONDITIONAL_OPEN_RESPONSE; PHYSICAL_NS_AND_NUMERICAL_POWER_BLOCKED**.
- Binding: `workerRoute: sol-high; phaseId: phase-2; stepId: prediction-contract-adjudication; attemptEpoch: 1`.
- Authority: `PROJECT_NOTES.md`, this run's plan/decisions and Phase-1 reports, the accepted candidate-action adjudication, and theory-reconstruction Gate Pack 4. The accepted action and density-melting no-go are unchanged.
- Decision: Phase 3 may implement the two contracts below. It may not add composition, a crust, a high-density continuation, a fitted CSS branch, a source/reservoir, or favorable device defaults.
- Scope effect: this immutable adjudication only; no implementation, generated artifact, registry, public synthesis, or canonical article was edited.

## Shared authority and reuse boundary

`verification/source_complete_solution_audit.py` is the only EOS/spectrum parameter producer: import its `Params`, branch functions, `passport()` and `validate_passport()`; do not copy or silently override its constants. Its source digest at adjudication is `aba1986ff4c9145685e112f9dc434ee4fdc5ab598888f1c889eb2880c3ce75a1`; a changed digest requires regeneration and re-adjudication of semantic drift, not stale artifact reuse.

For stellar structure, reuse the geometric conventions and the displayed TOV, Hinderer `y`, surface `k2`, `Lambda`, and binary `Lambda_tilde` equations from `verification/nvg_tidal_deformability.py`. Do **not** instantiate its canonical `EOS`, import its beta+CSS table, use its `p_match=1.5`/`Gamma=1.35` continuation, accept its `d epsilon/dP -> 1` clipping, or treat its `P=1e-4` cutoff as a surface. The sibling GW170817 file is not a second implementation. `nvg_full_ns_eos.py`, `nvg_eos_beta_checked.py`, and `nvg_eos_beta_saturated_vector.py` are comparison-only and supply no physics to this branch.

For the resonator branch, reuse only the no-gain/passivity semantics of `verification/nvg_generator_practical_audit.py`. The theta-haloscope scripts and `nvg_ds_core_oscillations.py` supply neither a `W/A` mode nor a coupling, pump, loss, volume, Q, or power. Their 53-micro-eV field and device numbers are forbidden inputs.

## Stellar contract: one-component mathematical sequence

### EOS adapter, domain, and surface

At baryon density `n=x n0`, evaluate the accepted positive-`W` stationary branch directly. Bind

```text
epsilon(n) = F_B(n,g_s W) + lambda(W^2-W0^2)^2/4
             + g_omega^2 n^2/(2 q_phi^2 W^2),
mu(n)      = sqrt(k_F^2+(g_s W)^2) + g_omega^2 n/(q_phi^2 W^2),
P(n)       = mu(n)n-epsilon(n),
h(n)       = ln(mu(n)/M_N),                 k_F=(6 pi^2 n/4)^(1/3).
```

The closed domain is exactly `0 <= x <= 10`, with the vacuum endpoint `(n,P,epsilon,h)=(0,0,0,0)` and upper endpoint taken from a fresh producer evaluation (the accepted snapshot gives `P(10 n0)=1692.0183516252062 MeV/fm^3`). No endpoint clamping or extrapolation is permitted. The authoritative solver performs no fitted EOS interpolation: for `0<h<=h(10n0)` it inverts monotone `h(n)` by a bracketed root and evaluates the source-complete functions; cancellation-sensitive low-density values are evaluated at 50 decimal digits. A 4097-positive-node grid `x_j=10^(-12+13j/4096)` plus the exact vacuum point may be used only as a log-coordinate PCHIP cross-check. It may not replace or enlarge the direct domain.

The stellar surface is the first exact `h=0` event, hence the vacuum `P=epsilon=n=0`; there is no finite-density surface and no crust. Start the enthalpy integration with the regular centre expansion at `h_c-delta_h`, `delta_h=min(1e-8,1e-6 h_c)`,

```text
r0^2 = delta_h/[2 pi(epsilon_c/3+P_c)],
m0   = 4 pi epsilon_c r0^3/3,                y0=2,
```

with geometric `epsilon,P` in `km^-2`. A missing surface event, a state outside the EOS domain, `r<=2m`, or nonfinite state is a failure, never a truncated star.

### Equations, units, and gates

Use `m,r` in km, `epsilon_g=k_conv epsilon`, `P_g=k_conv P`, and

```text
dm/dr = 4 pi r^2 epsilon_g,
dP/dr = -(epsilon_g+P_g)(m+4 pi r^3 P_g)/[r(r-2m)],
dy/dr = -(y^2+y F+r^2 Q)/r,
F = [1-4 pi r^2(epsilon_g-P_g)]/(1-2m/r),
Q = 4 pi[5epsilon_g+9P_g+(epsilon_g+P_g)/c_s^2]/(1-2m/r)
    -6/[r^2(1-2m/r)]
    -4[m+4 pi r^3P_g]^2/[r^4(1-2m/r)^2],
c_s^2=dP/d epsilon.
```

The primary independent variable is `h`, using `dr/dh=(dr/dP)(epsilon_g+P_g)` and the corresponding products for `dm/dh,dy/dh`. At the surface use the maintained Hinderer expression for `k2` and `Lambda=(2/3)k2 C^-5`, `C=m/R`; reject rather than zero-fill if `C<=0`, `C>=1/2`, the denominator is singular, or `k2/Lambda` is nonpositive or nonfinite. There is no density-jump match.

Bind the maintained primary conversions `k_conv=1.3234e-6 km^-2/(MeV/fm^3)` and `M_sun_km=1.4766 km`; independently reconstruct both from declared SI constants and require relative agreement `<=5e-4`. A factor of `8 pi`, `10^3`, or solar-mass conversion mutation must fail.

Before any star is accepted, validate on the direct branch and one-sided endpoints: finite `W,A0,epsilon,mu,P`; `W>0`, `epsilon>0` and `P>0` for `n>0`; `d epsilon/dn=mu>0`; `dP/dn>0`; and strictly `0<c_s^2<=1`. Derivatives use a five-point log-density stencil with step ladder `1e-3,5e-4,2.5e-4`, 50-digit endpoint checks, and maximum relative ladder spread `1e-6`. Values above one are not accepted by tolerance: a result within `1e-8` of either boundary is recomputed at 80 digits, then classified by the strict sign. Also require `P=mu n-epsilon`, `d epsilon/dn=mu`, and Hilbert/Legendre pressure agreement to relative `1e-8` (`1e-10` away from the vacuum underflow region).

### Central grid, stable branch, and allowed outputs

The deterministic central-density grid is `x_c,j=10^(-3+j/32)`, `j=0,...,128`. Stop before the first invalid EOS gate or at `10 n0`; never continue the EOS. The stable branch is the connected low-density sequence with `dM/dn_c>0` through the first mass maximum. Detect the first sign change on the fixed grid, refine the bracket by bounded maximization in `ln n_c` to `Delta n_c/n_c<=1e-6`, and include the refined maximum once. Rows after it are excluded. If no bracketed maximum occurs below `10 n0`, emit `DOMAIN_CENSORED_STABLE_TO_10N0`, set `M_max=null`, and do not call the endpoint a maximum.

Allowed row fields are `n_c/n0, epsilon_c, P_c, h_c, M/M_sun, R_km, C, y_R, k2, Lambda`, gate results, and numerical provenance, all labelled `MATHEMATICAL_ONE_COMPONENT_SEQUENCE` with `independent_evidence_weight=0`. `R(M)`, `k2(M)`, and `Lambda(M)` may be monotone-PCHIP interpolated only within the attained stable mass interval; an unavailable target is `BLOCKED_TARGET_OUTSIDE_SEQUENCE`, never extrapolated. `Lambda_tilde` may be evaluated only when both component masses lie inside that interval and remains mathematical/zero-weight.

`M_max`, `R_1.4`, `Lambda_1.4`, and binary values are mathematical properties, not neutron-star predictions. Every result must also carry `BLOCKED_MISSING_BETA_EQUILIBRIUM_CHARGE_NEUTRALITY_LEPTONS_CRUST_AND_HIGH_DENSITY_COMPLETION`. No proton fraction, beta equilibrium, electrons/muons, crust, phase transition, CSS completion, or uncertainty band may be inferred.

### Numerical acceptance

Primary integration is DOP853 in `h` with final `rtol=1e-10`, component `atol=(1e-11 km,1e-12 km,1e-10)` for `(r,m,y)`, and no step crossing `h=0`. The tolerance ladder `(rtol,scale*atol)=(1e-8,100),(1e-9,10),(1e-10,1)` must give final-two relative changes `<=1e-5` for `M,R` and `<=2e-4` for `k2,Lambda` on every fifth grid row, both endpoints, every target-mass bracket, and the maximum bracket.

An independent pressure-coordinate RK4 implementation must use the same direct EOS without derivative clipping, exact surface interpolation, and `dr=(0.05,0.025,0.0125) km`. Its final two resolutions must agree within `5e-4` for `M,R` and `2e-3` for `k2,Lambda`; its finest result must agree with DOP853 within `1e-3` and `5e-3`, respectively. Failure of any checked row fails the artifact. Doubling the central grid to 64 points/decade must move a resolved maximum by `<1e-3` in mass and central density and any reported target `R,Lambda` by `<2e-3`.

### Observational boundary

Observational records may be serialized only as external context with `comparison_status=BLOCKED_MODEL_NOT_PHYSICAL_NS`, `independent_evidence_weight=0`, and no likelihood, score, pass/fail, overlap, inside/outside, confirmation, exclusion, or fit language. The exact frozen comparisons are: J0740 timing `2.08+-0.07 M_sun` and its conservative `M_max>=2.01 M_sun` edge (Fonseca 2021); J0740 NICER/XMM radii `12.49 +1.28/-0.88 km` and `12.92 +2.09/-1.13 km` (Salmi/Dittmann 2024); J0030 `M=1.44 +0.15/-0.14 M_sun`, `R=13.02 +1.24/-1.06 km` (Miller 2019); GW170817 low-spin `Lambda_tilde=300 +420/-230` (70--720), high-spin 0--630, and the separately conditioned `Lambda_1.4=190 +390/-120`, `R=11.9+-1.4 km` (Abbott 2018). The repository's 11.2--13.2-km selection box, old `12.45+-0.65 km`, canonical `M_max~2.05`, `R_1.4~12.55`, and `Lambda_tilde~610` are forbidden comparisons and inputs.

## Resonator contract

### Fixed vacuum spectrum

Use `Phi=(W0+sigma)exp(i pi/W0)/sqrt(2)` and the source-free vacuum. The quadratic contract is

```text
L2 = (partial sigma)^2/2 - m_sigma^2 sigma^2/2 - F^2/4
     +(partial pi-m_A A)^2/2 + bar N(i slash-partial-M_N)N,
m_sigma^2=2 lambda W0^2,       m_A^2=q_phi^2 W0^2=m_omega^2,
omega_i^2=k^2+m_i^2;           omega_SI^2=(c k_SI)^2+(m_i/hbar)^2.
```

Count one physical radial polarization and three physical massive-vector polarizations. `A0` is a constraint, `pi` is the eaten Goldstone, and neither is an additional oscillator. Emit only the `k=0` gaps plus the symbolic dispersion unless a separately declared wave number is supplied; no medium shift or width exists in this vacuum contract.

Bind `hbar=6.582119569e-22 MeV s`, `h=4.135667696e-21 MeV s`, `c=299792458 m/s`, and the upstream `hbar c=197.3269804 MeV fm`. Compute `omega=m/hbar`, `f=m/h`, `bar-lambda=hbar c/m`, and `lambda_C=2 pi bar-lambda`; independently require `omega=2 pi f`, `lambda_C=c/f`, and the four conversions to relative `5e-10`. Expected central values are `m_sigma=1244.809262 MeV`, `f_sigma=3.0099354e23 Hz`, `m_A=782.600000 MeV`, and `f_A=1.8923184e23 Hz`, with uncertainty status `NOT_SUPPLIED`, not `+-0`.

### Conditional open-system schema

A physical response requires an added declared interaction `L_ext=sigma s_ext+A_mu j_ext^mu` (with `partial_mu j_ext^mu=0` for a vector drive), a physical region `V`, mode shape and normalization, positive mode inertia `M_i`, overlap/coupling `kappa_i=int_V u_i source_shape`, real drive amplitude `u0`, every damping channel `gamma_j>=0`, and one named output channel. With `gamma_tot=sum_j gamma_j>0`, bind

```text
M_i a_ddot+2M_i gamma_tot a_dot+M_i omega_i^2 a=kappa_i u0 cos(omega t),
X=kappa_i u0/[M_i(omega_i^2-omega^2-2i gamma_tot omega)],
Q_tot=omega_i/(2 gamma_tot),
Ebar=M_i(omega^2+omega_i^2)|X|^2/4,
P_j=M_i gamma_j omega^2|X|^2,       P_pump=<F a_dot>=sum_j P_j,
dE/dt=F a_dot-2M_i gamma_tot a_dot^2.
```

`V`, normalization, `M_i`, `kappa_i` with units/sign, `u0`, drive frequency/phase, each damping channel (including intrinsic and output), and output-port definition are mandatory before any joule, watt, bandwidth, ring-up time, or numerical Q is emitted. Missing values are explicit `null` plus `CONDITIONAL_RESPONSE/BLOCKED_NUMERICAL_POWER`; zero and missing are distinct. No defaults may be taken from theta, LC, mechanical, 150-kHz, or favorable hardware examples. A submitted Q is redundant and must agree with the channel sum to relative `1e-10`; otherwise reject. Negative loss, nonpositive inertia/volume, nonfinite input, undeclared port, hidden drive, `u0=0` with nonzero response, or average `P_out>P_pump` fails closed. Passive steady state requires the power sum and time-integrated energy balance to relative `1e-10`; transient integration must close to `1e-7` of the larger of input, dissipated energy, and stored-energy change.

A dimensionless verifier benchmark **is required**, but is not routed through the physical-power schema: `z''+2 zeta_tot z'+z=cos(Omega tau)` with `Mhat=omegahat=kappahat=u0hat=1`, `zeta_int=zeta_out=1/4`, `zeta_tot=1/2`, and `Omega in {0,1/2,1,2}`. At `Omega=1`, require `|Xhat|=1`, `Ebar_hat=1/2`, `Ppump_hat=1/2`, and `Pint_hat=Pout_hat=1/4`; zero drive gives all zero. Its artifact fields are `NORMALIZED_IDENTITY_BENCHMARK`, `device_mapping=false`, `physical_units=false`, and `independent_evidence_weight=0`. It licenses no volume, Q, power, gain, feasibility, or device claim.

## Passports, mutation gates, and artifacts

Both branches must embed the exact validated seven-base/two-derived upstream passport, action/adjudication reference, producer/test source digests, schema version, units, domain, solver/interpolation policy, tolerances, artifact-generation command, uncertainty status, claim status, and evidence weight. Derived `q_phi,g_s,m_sigma,m_A` must recompute from the submitted payload; physical-response records additionally require source interaction, provider/reference, value, uncertainty, unit, normalization, volume, overlap, drive, damping-channel, and port records. Placeholders, unresolved references, silent covariance independence, stale derived values, or a blocked renormalization record used as input invalidate the artifact.

Stellar mutations must reject: degeneracy 4->2; swapped `q_phi/g_s`; pressure not from the common generator; crust/CSS/beta/lepton injection; EOS endpoint clamp/extrapolation; `d epsilon/dP` clipping; wrong unit factors; central density above `10 n0`; finite-pressure surface; omitted surface event; inclusion after the first maximum; target-mass extrapolation; stale source digest; any observational weight above zero; or removal of the physical-NS blocker. Resonator mutations must reject: `sqrt(lambda)W0` as the radial mass; wrong Higgs mass/charge; `2 pi` conversion errors; two/four vector polarizations; a physical Goldstone; nonzero closed-vacuum response; omitted `V/M/kappa/u0/loss/port`; injected default Q/coupling; factor-two Q error; negative/hidden damping or drive; nonzero zero-drive output; or output/power sum exceeding the pump.

Phase 3 should own two unique producers/tests and sorted UTF-8 artifacts: `verification/source_complete_stellar_prediction.py`, `test_source_complete_stellar_prediction.py`, `source_complete_stellar_prediction_results.json`, `source_complete_stellar_prediction_report.md`; and the analogous four `source_complete_resonator_prediction*` paths. Each builder must serialize/render twice with identical bytes, equal the committed artifacts, contain only finite JSON numbers (blocked values are `null`), and pass compile plus focused semantic suites. Registration/public synthesis remains Phase 4 work; no existing canonical producer or artifact may be overwritten.

## Claim ceiling

The executable result may establish a deterministic mathematical one-fluid relativistic sequence and the accepted EFT's vacuum mass gaps. It may expose a conditional passive transfer function and a unitless identity fixture. It may not claim a physical neutron-star prediction, observational agreement/exclusion, an ordinary resonator, numerical power/Q/bandwidth, amplification, energy gain, free energy, or restoration of the `2.5 n0` bounce without separately adjudicated missing physics and measured open-system inputs.
