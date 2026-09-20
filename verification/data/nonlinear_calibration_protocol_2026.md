# Public numerical reproduction protocol: W8/U16 nonlinear inverse response

This file is the public, versioned numerical specification for the live
`verification/nvg_nonlinear_calibration_response.py` producer.  It contains no
saved result values and is sufficient, together with the public verification
sources, to reproduce the calculation in a checkout that does not contain the
run workspace.  The producer binds every result to the bytes of this file.

## Machine-readable controls

The following lines are part of the protocol contract and must agree with the
producer constants:

```text
protocol_version: nonlinear-calibration-public-v1
anchors: 0.90,0.93
deformation_amplitudes: 0,1
response_u: -0.20,-0.10,0.10,0.20
fd_u_steps: 0.005,0.002
primary_dps: 90
independent_control_dps: 120
stationarity_relative_limit: 1e-70
inverse_relative_limit: 1e-60
branch_window_ratio: 0.5,2.0
```

## Fixed physical scope

Use the live `BulkModel` in
`verification/source_complete_scaling_saturation_audit.py`, with natural-unit
conversion `n0=0.16 fm^-3`, binding `E/A-M_N=-16 MeV`, and incompressibility
`K=240 MeV`.  Reconstruct the two declared anchors `y*=0.90` and `y*=0.93` in
the cold homogeneous fixed-composition branch.  Compare only `alpha=0` W8 and
`alpha=1` W8 plus the fixed positive even-power U16 deformation
`DeltaU=eta (y^2-1)^4 (y^2-y*^2)^4`; do not fit a new physical parameter or read
stored output as an input.

Report only the local smooth continuation from the anchor.  Require positive
`y`, `C_y`, and `dmu/dn`, finite values, and `0.5 n0 <= n <= 2 n0`.  Compute the
live stationary energy and its derivatives through fourth order, then invert
the live chemical potential for both signs of each declared `response_u`.
Check the inverse-calculus identities, the dimensionless `Z` conversion, and
the analytic toy control described in the Russian derivation document.

## Numerical scope and provenance boundary

Primary rows use 90 decimal digits; an independent rebuild uses 120 digits.
Stationarity and inverse residual limits, branch continuation, and the
decreasing symmetric five-point controls are exactly those in the machine
readable lines and producer.  The output is a reproducible numerical
specification, **not an independently timestamped experimental preregistration**.

The physical anchors, amplitudes, equations, branch restrictions, and
acceptance limits are unchanged physical inputs.  An earlier internal freeze
recorded finite-difference steps `|u|=0.10,0.05`; the final producer uses
`|u|=0.005,0.002` as a numerical convergence refinement.  This observed
difference is retained as provenance rather than silently rewritten; it does
not introduce a new physical fit or claim that the entire final protocol was
frozen before every result row.

## Reproduction and fail-closed boundary

From the repository root:

```bash
python3 verification/nvg_nonlinear_calibration_response.py \
  --output verification/nvg_nonlinear_calibration_response_results.json
python3 -m unittest -v verification/test_nonlinear_calibration_response.py
```

The producer rebuilds all scientific rows from live code and emits strict
finite JSON.  Missing, unreadable, or marker-incompatible protocol bytes are a
failure; no fallback to a private run file or a missing/`None` hash is allowed.
The result metadata therefore records `protocol_frozen_before_rows: false`,
`physical_inputs_fixed: true`, and the explicit public provenance label.

