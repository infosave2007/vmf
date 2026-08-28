#!/usr/bin/env python3
"""Dimensional neutron-star mass-bound diagnostic.

The scale ``M_Pl^3/M_Omega^2`` is a declared-anchor calculation, separate from
the canonical TOV chain.  Literature masses are inputs for a falsifier cut;
no EOS retuning or static "resolution" is embedded here.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

M_OMEGA = 859.0
DM_OMEGA = 8.0
M_PL_KG = math.sqrt(1.054571817e-34 * 2.99792458e8 / 6.67430e-11)
M_SUN_KG = 1.98892e30
M_OMEGA_KG = M_OMEGA * 1.78266192e-30

OBSERVED_SAMPLE = (
    ("PSR J0740+6620", 2.08, 0.07, "NICER+XMM, Fargo+24"),
    ("PSR J1614-2230", 1.97, 0.04, "Shapiro delay, Fonseca+16"),
    ("PSR J0348+0432", 2.01, 0.04, "Shapiro delay, Antoniadis+13"),
    ("PSR J0952-0607", 2.35, 0.17, "spectroscopic, Romani+22 (disputed)"),
)

EVIDENCE_STATUS = "DIMENSIONAL_BOUND_INPUT_COMPARISON_NO_EOS_RETUNE"


def compute_bound() -> dict[str, float]:
    m_bare = M_PL_KG ** 3 / M_OMEGA_KG ** 2 / M_SUN_KG
    upper = m_bare * (M_OMEGA / (M_OMEGA - DM_OMEGA)) ** 2
    lower = m_bare * (M_OMEGA / (M_OMEGA + DM_OMEGA)) ** 2
    return {"m_bare": float(m_bare), "m_lower": float(lower), "m_upper": float(upper),
            "sigma_high": float(upper - m_bare), "sigma_low": float(m_bare - lower)}


def live_comparison(bound: dict[str, float] | None = None) -> list[dict[str, Any]]:
    bound = bound or compute_bound()
    rows = []
    for name, mass, sigma, reference in OBSERVED_SAMPLE:
        margin = (bound["m_upper"] - mass) / sigma
        disputed = "disputed" in reference.lower()
        rows.append({"name": name, "mass_msun": mass, "sigma": sigma,
                     "reference": reference, "margin_sigma": margin,
                     "status": "TENSE_DISPUTED" if disputed else ("ALIVE" if margin > 0 else "REFUTED")})
    return rows


def canonical_context() -> dict[str, Any]:
    """Read the canonical runtime mass without modifying this bound."""

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from nvg_joint_ns_inference import compute_nvg_predictions
        predictions, metadata = compute_nvg_predictions()
        return {"m_max": predictions["M_max"], "status": "CONDITIONAL_IN_SAMPLE_CANONICAL_TOV",
                "solver": metadata["M_max"]["source"]}
    except Exception as exc:  # pragma: no cover - diagnostics remain usable
        return {"m_max": None, "status": "CANONICAL_CONTEXT_UNAVAILABLE", "error": str(exc)}


def compute_report() -> dict[str, Any]:
    bound = compute_bound()
    return {"bound": bound, "sample": live_comparison(bound),
            "canonical": canonical_context(), "status": EVIDENCE_STATUS,
            "falsifier": f"robust cold-NS mass above {bound['m_upper']:.3f} M_sun"}


def main() -> dict[str, Any]:
    report = compute_report()
    bound = report["bound"]
    print("=" * 74)
    print("  NVG PARAMETER-FREE NS MASS-BOUND DIAGNOSTIC")
    print("=" * 74)
    print(f"M_bare={bound['m_bare']:.3f} M_sun (+{bound['sigma_high']:.3f}/-{bound['sigma_low']:.3f})")
    print(f"Declared bound interval=[{bound['m_lower']:.3f}, {bound['m_upper']:.3f}] M_sun")
    for row in report["sample"]:
        print(f"{row['name']:<18} {row['mass_msun']:.2f}+/-{row['sigma']:.2f} -> {row['status']} ({row['margin_sigma']:+.1f} sigma)")
    print(f"Canonical TOV context: M_max={report['canonical']['m_max']}; status={report['canonical']['status']}")
    print(f"Status: {report['status']}; falsifier: {report['falsifier']}")
    print("No gamma retune or alternate fork mass is used to support this diagnostic.")
    print("=" * 74)
    return report


if __name__ == "__main__":
    main()
