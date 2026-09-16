# Source-complete resonator prediction audit

## Control block

- Status: **PASS_FIXED_SPECTRUM_CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER**.
- Fixed vacuum spectrum is derived from the validated upstream passport; no theta-haloscope, LC, mechanical, or 150-kHz defaults are imported.
- Numerical response/power is conditional on a complete open-system passport and remains explicitly blocked for the default incomplete input.

## Spectrum

| mode | mass (MeV) | frequency (Hz) | reduced Compton (fm) | physical polarizations |
|---|---:|---:|---:|---:|
| scalar_radial | 1244.80926 | 3.009935406e+23 | 0.158519852 | 1 |
| massive_vector | 782.6 | 1.892318381e+23 | 0.252142832 | 3 |

The quadratic action gives `m_sigma^2=2 lambda W0^2` and `m_A=q_phi W0=m_omega`; the phase is an eaten Goldstone and `A0` is a Gauss constraint. The massless phase frequency is not an observable.

## Open-system response and passivity

The response equation is `M a_ddot + 2 M gamma_tot a_dot + M omega_0^2 a = Re(F0 exp(i omega t))`, with `Q=omega_0/(2 gamma_tot)` and `F0=kappa*u0*exp(i phase)`. Stored energy is `Ebar=M(omega^2+omega_0^2)|X|^2/4`; each damping channel has `P_j=M gamma_j omega^2 |X|^2`. Pump is evaluated independently as `P_pump=Re(F0*conj(-i omega X))/2`; the loss sum is only a comparison. A separate physical DOP853 transient integrates every submitted parameter and channel work in scaled time; the normalized fixture is never substituted.
Default response status: **CONDITIONAL_RESPONSE_BLOCKED_NUMERICAL_POWER**; missing fields are explicit and numerical watts/Q/bandwidth are null. Output cannot exceed measured pump under the passivity gate.

## Normalized identity benchmark

`NORMALIZED_IDENTITY_BENCHMARK` uses `z''+2 zeta_tot z'+z=cos(Omega tau)`, `zeta_int=zeta_out=1/4`, `zeta_tot=1/2`, and `Omega in {0,1/2,1,2}`. At resonance it checks `|X|=1`, `Ebar=1/2`, direct `Ppump=1/2`, and `Pint=Pout=1/4`; zero drive is identically zero. Device mapping and physical units are false; evidence weight is zero.
Transient normalized energy balance residual: `1.032e-08` (PASS=True).

## Boundaries and audit contract

Typed source records use exact scalar/vector discriminants and bind the actual amplitude, phase, overlap, and selected mode. Ports use exact `damping_output` records and exactly one role=`output` channel. Missing/nonfinite/negative loss, stale provenance, lexical or hidden drive, omitted volume/inertia/overlap/phase/port, Goldstone double counting, wrong Higgs mass, factor-of-two Q errors, conversion drift, transient closure failure, or output greater than independently computed pump fail closed. The fixed spectrum is a mathematical EFT output with zero independent-evidence weight; no numerical device power, gain, free-energy, or ordinary resonator claim follows.
