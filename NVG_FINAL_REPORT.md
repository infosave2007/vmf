# NVG Runtime Evidence Ledger
**Generated (UTC):** 2026-09-16T18:20:15+00:00

## Canonical executable result

The table below is populated from one executed TOV + Hinderer chain. It is not an uncertainty propagation: the current EOS implementation does not expose an off-anchor M_Omega,0 parameter.

| Quantity | Runtime value | Semantics |
|---|---:|---|
| M_Omega,0 input (MeV) | 859.0 +/- 8.0 | lattice/QCD input |
| M_max (M_sun) | 2.048 | nvg_tidal_deformability.EOS + solve_tov_tidal; conditional/in-sample selection |
| R_1.4 (km) | 12.550 | nvg_tidal_deformability.EOS + solve_tov_tidal; conditional/in-sample selection |
| Lambda_1.4 | 519.4 | nvg_tidal_deformability.EOS + solve_tov_tidal; conditional/in-sample selection |
| Conditional/in-sample reduced chi-squared | 0.684 | 3 rows; calibration excluded |

## Canonical selection provenance

The transition point is selected in-sample from J0740, GW170817, and NICER constraints. It is conditional model input, not an independent confirmation.

- status: `CONDITIONAL_IN_SAMPLE`
- selected by: `verification/nvg_ns_parameter_scan.py`
- method: `best_margin_survivor_on_explicit_grid`
- selected transition parameters: `{'n_trans_ratio': 2.0, 'delta_eps_ratio': 0.0, 'cs2_q': 0.3333333333333333}`
- held-out observations: none

## Evidence ledger

| Claim | Result | Script/input | Status |
|---|---|---|---|
| QCD anchor | M_Omega,0 = 859.0 +/- 8.0 MeV | `declared lattice input` | INPUT (not a model prediction) |
| NS maximum mass | M_max = 2.048 M_sun | `nvg_tidal_deformability.py` | DERIVED at runtime from canonical TOV chain; conditional/in-sample selection |
| NS radius | R_1.4 = 12.550 km | `nvg_tidal_deformability.py` | DERIVED at runtime from canonical TOV chain; conditional/in-sample selection |
| Tidal deformability | Lambda_1.4 = 519.4 | `nvg_tidal_deformability.py` | DERIVED at runtime from canonical TOV + Hinderer chain; conditional/in-sample selection |
| Joint NS likelihood | reduced chi-squared = 0.684 | `nvg_joint_ns_inference.py` | CONDITIONAL_IN_SAMPLE; selected on J0740/GW170817/NICER; Cooling_Dichotomy excluded as calibration |

## Forecasts (not evidence)

- **LIGO O5 / Einstein Telescope:** An independent tidal-deformability measurement tests the canonical TOV output.
- **STROBE-X / eXTP:** An independent surface-redshift measurement tests the EOS mass-radius branch.
- **CBM / FAIR:** An independently calibrated rho-meson peak measurement tests the dense-matter input.
- **EHT (Next Gen):** A resolved shadow deviation would test the exterior null estimate; no detection claim is made here.

## Limitations

- Cooling is a calibration target in the joint comparison and is not counted in chi-squared.
- Off-anchor uncertainty propagation and inverse QCD reconstruction are unsupported until an independent parameterized EOS chain is implemented.
- A successful process run is not a scientific verification of unrelated claims.
