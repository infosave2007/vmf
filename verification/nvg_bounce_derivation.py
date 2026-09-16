#!/usr/bin/env python3
"""Explicit background conservation derivation, not an Einstein-Cartan action.

The former print-only torsion proof equated an undetermined coupling with an
adopted QCD scale. That inference was invalid. This module now exposes the
conditional calculation and missing inputs; the numerical solution lives in
nvg_cyclic_bounce.py.
"""
from __future__ import annotations
import json
from nvg_cyclic_bounce import compute_bounce_state, M_Omega_0, hbar_c


def derive_modified_friedmann():
    state = compute_bounce_state()
    print("Assume epsilon_eff=F(epsilon), with conserved microscopic fluid.")
    print("Then P_eff=(epsilon+P)*Fprime-F, by continuity.")
    print("Quadratic F=epsilon*(1-epsilon/epsilon_c) is an EXTRA assumption.")
    print("For radiation: a/a_min=(1+4*(t/t_char)^2)^(1/4).")
    print("Neither this coefficient nor a torsion action is derived from NVG.")
    print(json.dumps(state, indent=2, allow_nan=False))
    return state


def prove_rho_c_uniqueness(coefficient=1.0):
    import math
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("coefficient must be finite and positive")
    return {"epsilon_scale_MeV_fm3": coefficient*M_Omega_0**4/hbar_c**3,
            "coefficient_input": coefficient, "unique_coefficient_derived": False}


def check_cmb_bao_compatibility():
    return {"evidence_status": "NO_CMB_BAO_LIKELIHOOD", "observed_likelihood": None,
            "reason": "No perturbation/transfer or likelihood calculation in this module."}


def main():
    state = derive_modified_friedmann()
    print(json.dumps(prove_rho_c_uniqueness(), indent=2))
    print(json.dumps(check_cmb_bao_compatibility(), indent=2))
    return state


if __name__ == "__main__":
    main()
