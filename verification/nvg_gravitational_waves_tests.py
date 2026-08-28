#!/usr/bin/env python3
"""GW/cosmology checks with computed EOS state and explicit assumptions.

The old version selected CPL coefficients from a desired turnaround and then
called the result a DESI confirmation.  A turnaround time alone does not
derive a CPL likelihood, so this entry point reports that route as an
assumption and keeps only executable scale/consistency calculations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _eos_chain() -> dict:
    verification_dir = Path(__file__).resolve().parent
    if str(verification_dir) not in sys.path:
        sys.path.insert(0, str(verification_dir))
    from nvg_bbn_reionization import compute_eos_chain, radius_at

    chain = compute_eos_chain()
    chain["radius_at"] = radius_at
    return chain


def compute_results() -> dict:
    chain = _eos_chain()

    # Inputs describing the cyclic scenario are not enough to fit w0/wa.
    t_now = 13.8  # Gyr, observational age input
    t_turn = 37.0  # Gyr, scenario input (not derived here)
    if t_turn <= t_now:
        raise ValueError("turnaround must be later than the current age")
    a_turn = (t_turn / t_now) ** (2.0 / 3.0)
    cyclic = {
        "t_now_gyr": t_now,
        "t_turn_gyr": t_turn,
        "a_turn_assumption": a_turn,
        "cpl_parameters": None,
        "status": "ASSUMPTION_NO_INDEPENDENT_CPL_DERIVATION",
    }

    # A dimensional bounce-scale estimate.  P_S and the bound are observational
    # inputs; this does not establish a full inflation/transfer-function model.
    M_omega = 859.0  # MeV, model scale
    E_planck = 1.22e22  # MeV, physical constant
    P_s = 2.1e-9  # scalar power, observational input
    tensor_power = (M_omega / E_planck) ** 4
    r_model = tensor_power / P_s
    r_bound = 0.036  # BICEP/Keck upper-bound input
    tensor = {
        "tensor_power": tensor_power,
        "scalar_power_input": P_s,
        "r_scale_estimate": r_model,
        "bound_input": r_bound,
        "below_bound": bool(r_model < r_bound),
        "status": "SCALE_ESTIMATE_NOT_FULL_PRIMORDIAL_PREDICTION",
    }

    # Kepler estimates consume R(M) and M_max from the computed EOS branch.
    radius_at = chain["radius_at"]
    m_ref = 1.4
    r_ref = radius_at(chain, m_ref)
    m_max = chain["m_max"]
    r_max = radius_at(chain, m_max)
    f_k_ref = 1.08 * (m_ref / 1.4) ** 0.5 * (10.0 / r_ref) ** 1.5
    f_k_max = 1.08 * (m_max / 1.4) ** 0.5 * (10.0 / r_max) ** 1.5
    spin = {
        "m_max_msun": m_max,
        "r_14_km": r_ref,
        "r_max_km": r_max,
        "f_k_14_khz": f_k_ref,
        "f_k_max_khz": f_k_max,
        "status": "COMPUTED_MODEL_ESTIMATE",
    }

    return {"cyclic_dark_energy": cyclic, "tensor_ratio": tensor, "spin": spin}


RESULTS = compute_results()


def main() -> None:
    print("=" * 72)
    print("  NVG VERIFICATION: GW/COSMOLOGY CHECKS (COMPUTED CHAIN)")
    print("=" * 72)
    cyclic = RESULTS["cyclic_dark_energy"]
    print(
        "L. Cyclic turnaround inputs: "
        f"t_now={cyclic['t_now_gyr']:.1f} Gyr, "
        f"t_turn={cyclic['t_turn_gyr']:.1f} Gyr, "
        f"a_turn≈{cyclic['a_turn_assumption']:.3f}"
    )
    print(f"   CPL w0/wa: not computed ({cyclic['status']})")

    tensor = RESULTS["tensor_ratio"]
    print(
        f"M. r scale estimate={tensor['r_scale_estimate']:.3e}; "
        f"bound input={tensor['bound_input']:.3f}; status={tensor['status']}"
    )

    spin = RESULTS["spin"]
    print(
        f"N. EOS chain M_max={spin['m_max_msun']:.3f} M_sun, "
        f"R_1.4={spin['r_14_km']:.3f} km, "
        f"f_K(1.4)={spin['f_k_14_khz']:.3f} kHz, "
        f"f_K(M_max)={spin['f_k_max_khz']:.3f} kHz"
    )
    print(f"   status={spin['status']}")
    print("No DESI compatibility claim is emitted: no independent CPL likelihood is executed.")


if __name__ == "__main__":
    main()
