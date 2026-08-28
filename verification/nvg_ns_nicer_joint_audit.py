#!/usr/bin/env python3
"""Current NICER/maximum-mass/tidal comparison for the canonical NS chain.

Observations in this file are declared literature inputs.  Canonical values are
loaded from the maintained TOV/Hinderer solver at runtime; no dated ``M_max``,
radius, or Lambda snapshot is treated as a solver result.  Because the
transition was selected using the J0740/GW170817/NICER constraints, the audit
is explicitly conditional/in-sample and carries no independent evidence weight.
"""

from __future__ import annotations

import os
import sys
from typing import Any

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Current measurements (literature inputs, not model outputs).
MASS = [
    ("J0740+6620 mass", 2.08, 0.07, "Fonseca+2021"),
    ("J0952-0607 mass", 2.35, 0.17, "Romani+2022 (systematics-limited)"),
]
RADII = [
    ("J0030+0451 R(1.44)", 12.71, 1.15, "Riley+2019 NICER"),
    ("J0437-4715 R(1.42)", 11.36, 0.80, "Choudhury+2024 NICER"),
    ("J0740+6620 R(2.08)", 12.49, 1.00, "Salmi+2024 NICER+XMM"),
    ("J0614-3329 R(1.44)", 10.29, 1.01, "Mauviard+2025 NICER+XMM"),
]
LAMBDA = ("GW170817 Lambda(1.4)", 190.0, 390.0, 120.0, "Abbott+2018")


def one_sided_mass_pull(m_max: float, m_obs: float, sigma: float) -> float:
    """Penalise only if the EOS cannot support the observed pulsar."""

    return (m_max - m_obs) / sigma if m_max < m_obs else 0.0


def canonical_predictions() -> dict[str, Any]:
    """Run and return the one canonical prediction chain plus provenance."""

    import nvg_joint_ns_inference as joint

    predictions, metadata = joint.compute_nvg_predictions()
    return {
        "M_max": float(predictions["M_max"]),
        "R_1.4": float(predictions["R_1.4"]),
        "R_1.42": predictions.get("R_1.42"),
        "R_2.08": predictions.get("R_2.08"),
        "Lambda_1.4": float(predictions["Lambda_1.4"]),
        "solver": "nvg_tidal_deformability.EOS + solve_tov_tidal",
        "selection_parameters": metadata["canonical_selection"]["parameters"],
        "selection_provenance": metadata["selection_provenance"],
    }


def run_audit() -> dict[str, Any]:
    """Compute descriptive pulls against declared observations."""

    canonical = canonical_predictions()
    m_max = canonical["M_max"]
    r14 = canonical["R_1.4"]
    # Do not substitute a nearby mass branch when the requested branch is
    # unavailable.  A missing canonical mass point is unsupported and must be
    # excluded rather than silently scored as R_1.4.
    r142 = canonical["R_1.42"]
    r208 = canonical["R_2.08"]
    chi2 = 0.0
    rows: list[dict[str, Any]] = []

    for name, value, sigma, source in MASS:
        pull = one_sided_mass_pull(m_max, value, sigma)
        chi2 += pull**2
        rows.append({"name": name, "observation": value, "prediction": m_max,
                     "pull": pull, "source": source, "kind": "conditional/in-sample",
                     "included": True})

    for name, value, sigma, source in RADII:
        if "2.08" in name:
            prediction = r208
        elif "1.42" in name:
            prediction = r142
        else:
            prediction = r14
        if prediction is None:
            rows.append({"name": name, "observation": value, "prediction": None,
                         "pull": None, "source": source, "kind": "unsupported: branch does not reach mass",
                         "included": False})
            continue
        pull = (prediction - value) / sigma
        chi2 += pull**2
        rows.append({"name": name, "observation": value, "prediction": prediction,
                     "pull": pull, "source": source, "kind": "conditional/in-sample",
                     "included": True})

    name, value, sigma_upper, sigma_lower, source = LAMBDA
    sigma = sigma_upper if canonical["Lambda_1.4"] > value else sigma_lower
    pull = (canonical["Lambda_1.4"] - value) / sigma
    chi2 += pull**2
    rows.append({"name": name, "observation": value, "prediction": canonical["Lambda_1.4"],
                 "pull": pull, "source": source, "kind": "conditional/in-sample",
                 "included": True})

    return {
        "canonical": canonical,
        "rows": rows,
        "chi_squared": chi2,
        "dof": sum(1 for row in rows if row.get("included", True)),
        "reduced_chi_squared": chi2 / max(sum(1 for row in rows if row.get("included", True)), 1),
        "status": "CONDITIONAL_IN_SAMPLE",
        "selection_provenance": canonical["selection_provenance"],
    }


def main() -> dict[str, Any]:
    result = run_audit()
    c = result["canonical"]
    print("=" * 84)
    print("  NVG CANONICAL EOS vs CURRENT NICER / MASS / GW170817")
    print("  CONDITIONAL_IN_SAMPLE audit (transition selected on J0740/GW170817/NICER)")
    print("=" * 84)
    print(f"  Solver: {c['solver']}")
    print(f"  Canonical transition parameters: {c['selection_parameters']}")
    print(f"  Canonical runtime outputs: M_max={c['M_max']:.3f}, R_1.4={c['R_1.4']:.3f}, "
          f"Lambda_1.4={c['Lambda_1.4']:.1f}")
    print(f"  Selection method: {result['selection_provenance']['selection_method']}")
    print(f"    {'constraint':<31}{'obs':>14}{'pred':>10}{'pull':>9}")
    print("    " + "-" * 68)
    for row in result["rows"]:
        prediction = "--" if row["prediction"] is None else f"{row['prediction']:.3f}"
        pull = "unsupported" if row["pull"] is None else f"{row['pull']:+.2f}"
        print(f"    {row['name']:<31}{row['observation']:>14.2f} {prediction:>9}"
              f" {pull:>9}")
    print("    " + "-" * 68)
    print(f"    conditional chi-squared = {result['chi_squared']:.3f} over {result['dof']} rows")
    print(f"    reduced conditional chi-squared = {result['reduced_chi_squared']:.3f}")
    print("    This in-sample statistic is descriptive only; no independent claim is made.")
    print("=" * 84)
    return result


if __name__ == "__main__":
    main()
