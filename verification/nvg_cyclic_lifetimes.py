#!/usr/bin/env python3
"""Conditional horizon-chain lifetime scales, not measured cosmic cycles.

Uses the same producer and input conventions as nvg_cyclic_cosmology.
pi*G*M/c^3 is an assumed scenario timescale, not a prediction of time remaining.
No instanton, entropy-production law or dynamical recollapse is solved.
"""
from __future__ import annotations
import json
from nvg_cyclic_cosmology import compute_cyclic_state, G, c, yr_to_s


def compute_lifetimes(*, cycle_index=77, entropy_factor=4.0):
    first = compute_cyclic_state(cycle_index=1, entropy_factor=entropy_factor)
    assigned = compute_cyclic_state(cycle_index=cycle_index, entropy_factor=entropy_factor)
    return {"first_cycle_scenario_yr": first["lifetime_yr"], "assigned_cycle": assigned,
            "evidence_status": "CALIBRATED_HORIZON_CHAIN", "observed_likelihood": None,
            "limitation": "No observed cycle, instanton solution, entropy transfer or time-to-turnaround prediction."}


def main():
    state = compute_lifetimes()
    print(json.dumps(state, indent=2, allow_nan=False))
    return state


if __name__ == "__main__":
    main()
