#!/usr/bin/env python3
"""Regenerate scoped BH/cosmology results; no parameters fitted to outcomes.

This joins independent mathematical checks, NOT independent observations.
--write is required to replace the dedicated artifact. Failed algebra or
conservation checks refuse the write. A physical no-go is a valid result.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import mpmath as mp

import nvg_black_hole_interior as interior
import nvg_cyclic_bounce as bounce
import nvg_cyclic_cosmology as cycles
import regular_core_geometry_audit as geometry
from source_complete_scaling_saturation_audit import BulkModel, _check_state

RESULT_PATH = Path(__file__).with_name("bh_cycle_consistency_results.json")


def source_branch_witness():
    """Independent arbitrary precision EOS, unchanged accepted input parameters.

    Point samples supplement, not replace, the analytic Cv>=Cs obstruction
    proved in SOURCE_COMPLETE_UNIFICATION_RU.md. Only stationary, homogeneous
    zero-T one-component matter is used as the GR cosmological fluid.
    """
    with mp.workdps(80):
        model = BulkModel()
        if (not mp.isfinite(model.Cv) or not mp.isfinite(model.Cs)
                or model.Cs <= 0 or model.Cv < model.Cs):
            raise ArithmeticError("global baseline obstruction requires finite Cv>=Cs>0")
        rows = []
        for x in ("0.001", "0.1", "1", "2.5", "10", "100"):
            state = model.equilibrium(mp.mpf(x)*model.n0, guess=("0.8", "1.7"))
            if any(not mp.isfinite(value) for value in state.values() if isinstance(value, (mp.mpf, float, int))):
                raise ArithmeticError("nonfinite source state cannot certify an obstruction")
            _check_state(state)
            e, p = state["energy_total"], state["pressure_total"]
            if e <= 0 or p <= 0:
                raise ArithmeticError("positive-pressure baseline witness failed")
            # Common conversion from natural MeV^4 to MeV/fm^3.
            e, p = e/model.hbarc**3, p/model.hbarc**3
            rows.append({"n_over_n0": x, "W_over_W0": mp.nstr(state["y"],28),
                         "epsilon_MeV_fm3": mp.nstr(e,28), "P_MeV_fm3": mp.nstr(p,28),
                         "epsilon_plus_3P_MeV_fm3": mp.nstr(e+3*p,28)})
        return {"Cv_over_Cs": mp.nstr(model.Cv/model.Cs,28),
                "analytic_no_binding_condition": bool(model.Cv >= model.Cs),
                "samples": rows,
                "scope": "Original homogeneous stationary branch used as a GR fluid, no added sector.",
                "consequence": "epsilon>0 prevents flat H=0; epsilon+3P>0 makes any k=+1 H=0 a maximum, not a bounce.",
                "excluded_claim": "Not a no-go for dynamical scalars, other potentials, modified gravity, or all cyclic cosmologies."}


def build_result():
    geo = geometry.build_result()
    if geo["status"] != "GEOMETRY_IDENTITIES_VERIFIED":
        raise ArithmeticError("geometry identities failed")
    eos = interior.compute_state()
    if not eos["sampled_checks"]["closure_within_1e_minus_11"]:
        raise ArithmeticError("quasiparticle thermodynamic closure failed")
    background = bounce.compute_bounce_state()
    return {
        "schema_version": 1,
        "status": "CONSISTENCY_CHECKS_PASSED_PHYSICAL_UNIFICATION_UNRESOLVED",
        "evidence_weight": 0,
        "observed_likelihood": None,
        "general_identity": {
            "energy_map": "epsilon_eff=F(epsilon)",
            "pressure": "P_eff=(epsilon+P)*Fprime-F",
            "trace": "Theta_eff=Fprime*Theta+4*(F-epsilon*Fprime)",
            "source_trace_on_shell": "Theta=4U-W*Uprime",
            "assumptions": ["F differentiable and a function of epsilon alone",
                            "same vacuum subtraction for all energies",
                            "separately conserved microscopic fluid or zero-T Legendre composition",
                            "source trace only on homogeneous stationary branch, not arbitrary field dynamics"],
            "interpretation": "Conservation identity and diagnostic, not a new physical law or chosen NVG interaction.",
        },
        "source_complete_GR_obstruction": source_branch_witness(),
        "repaired_legacy_interior": eos,
        "regular_metric_geometry": geo,
        "conditional_bounce_background": background,
        "conditional_cycle_assignment": cycles.compute_cyclic_state(),
        "physical_completion_missing": [
            "Stress-tensor realization of the regular metric from a specified matter action",
            "A stable action producing any bounce modification and fixing its coefficient",
            "Dynamical collapse/junction and perturbation analysis",
            "Recollapse, entropy transfer and recurrence dynamics",
            "Independent observational likelihoods"],
    }


def validate_result(result):
    """Live regeneration in this process, not an independent implementation."""
    if not isinstance(result, dict):
        return False
    try:
        serialized=json.dumps(result,sort_keys=True,allow_nan=False)
    except (ValueError, TypeError):
        return False
    return serialized == json.dumps(build_result(),sort_keys=True,allow_nan=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args(argv)
    result=build_result()  # all checks finish before opening any output file
    data=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
    if args.write:
        RESULT_PATH.write_text(data,encoding="utf-8")
    print(data,end="")
    return result


if __name__ == "__main__":
    main()
