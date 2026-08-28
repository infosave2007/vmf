#!/usr/bin/env python3
"""Generate a compact, runtime-backed NVG evidence ledger.

This entry point used to manufacture a large uncertainty table from hand-set
scaling constants and then describe those values as predictions.  The ledger
now reports only the canonical neutron-star chain actually executed in this
repository.  Future measurements remain forecasts, and calibration inputs are
identified rather than counted as independent evidence.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timezone
from typing import Any


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

M_OMEGA_CENTRAL = 859.0  # MeV; declared QCD/lattice anchor input
M_OMEGA_ERR = 8.0  # MeV; declared uncertainty on the input anchor

LATTICE_PRIORS = {
    "Baseline": {"mean": M_OMEGA_CENTRAL, "err": M_OMEGA_ERR},
    "Gupta (2021) approx": {"mean": 862.0, "err": 12.0},
    "Agadjanov (2023) approx": {"mean": 851.0, "err": 15.0},
}

FORECAST = {
    "LIGO O5 / Einstein Telescope": "An independent tidal-deformability measurement tests the canonical TOV output.",
    "STROBE-X / eXTP": "An independent surface-redshift measurement tests the EOS mass-radius branch.",
    "CBM / FAIR": "An independently calibrated rho-meson peak measurement tests the dense-matter input.",
    "EHT (Next Gen)": "A resolved shadow deviation would test the exterior null estimate; no detection claim is made here.",
}

_CANONICAL_CACHE: dict[str, Any] | None = None


def _canonical_state() -> dict[str, Any]:
    """Execute the canonical joint EOS once and cache its structured result."""

    global _CANONICAL_CACHE
    if _CANONICAL_CACHE is None:
        from nvg_joint_ns_inference import compute_nvg_predictions, run_joint_inference

        predictions, metadata = compute_nvg_predictions()
        inference = run_joint_inference(predictions, metadata)
        _CANONICAL_CACHE = {
            "predictions": predictions,
            "metadata": metadata,
            "inference": inference,
            "selection": metadata["canonical_selection"],
            "solver": "nvg_tidal_deformability.EOS + solve_tov_tidal",
        }
    return _CANONICAL_CACHE


def run_forward_model(m_omega: float = M_OMEGA_CENTRAL) -> dict[str, Any]:
    """Return canonical outputs and explicit anchor semantics.

    The maintained EOS chain currently has no parameterized dependence on a
    variable ``m_omega``.  Consequently, values requested away from the
    central anchor are not fabricated as an uncertainty propagation; they are
    returned with a warning that only the central chain was evaluated.
    """

    if not math.isfinite(m_omega) or m_omega <= 0.0:
        raise ValueError("m_omega must be a positive finite MeV value")
    canonical = _canonical_state()
    predictions = canonical["predictions"]
    inference = canonical["inference"]
    selection = canonical["selection"]
    is_central = math.isclose(m_omega, M_OMEGA_CENTRAL, rel_tol=0.0, abs_tol=1e-12)
    return {
        "m_omega": float(m_omega),
        "m_omega_err": M_OMEGA_ERR,
        "evaluated_at_central_anchor": is_central,
        "m_max": predictions["M_max"],
        "r_14": predictions["R_1.4"],
        "lambda_14": predictions["Lambda_1.4"],
        "chi2_red": inference["reduced_chi"],
        "chi2_conditional": inference["chi_squared_total"],
        "conditional_rows": inference["in_sample_count"],
        "calibrated_rows": [row["key"] for row in inference["rows"] if not row["included"]],
        "comparison_status": inference["comparison_status"],
        "selection_parameters": selection["parameters"],
        "selection_provenance": selection["provenance"],
        "solver": canonical["solver"],
        "status": (
            "canonical runtime output; conditional/in-sample selection"
            if is_central
            else "central chain reused; no off-anchor EOS evaluation performed; conditional/in-sample"
        ),
    }


def solve_inverse_qcd(obs_lambda_14: float | None = None, obs_m_max: float | None = None) -> dict[str, Any]:
    """Decline an inverse claim when no independent inverse model is present."""

    inputs = {}
    if obs_lambda_14 is not None:
        inputs["Lambda_1.4"] = float(obs_lambda_14)
    if obs_m_max is not None:
        inputs["M_max"] = float(obs_m_max)
    return {
        "status": "UNSUPPORTED: no independent inverse QCD mapping is implemented",
        "inputs": inputs,
    }


def generate_evidence_ledger(results_center: dict[str, Any]) -> list[dict[str, str]]:
    """Build rows directly from the canonical result and its provenance."""

    return [
        {
            "claim": "QCD anchor",
            "value": f"M_Omega,0 = {results_center['m_omega']:.1f} +/- {results_center['m_omega_err']:.1f} MeV",
            "file": "declared lattice input",
            "status": "INPUT (not a model prediction)",
        },
        {
            "claim": "NS maximum mass",
            "value": f"M_max = {results_center['m_max']:.3f} M_sun",
            "file": "nvg_tidal_deformability.py",
            "status": "DERIVED at runtime from canonical TOV chain; conditional/in-sample selection",
        },
        {
            "claim": "NS radius",
            "value": f"R_1.4 = {results_center['r_14']:.3f} km",
            "file": "nvg_tidal_deformability.py",
            "status": "DERIVED at runtime from canonical TOV chain; conditional/in-sample selection",
        },
        {
            "claim": "Tidal deformability",
            "value": f"Lambda_1.4 = {results_center['lambda_14']:.1f}",
            "file": "nvg_tidal_deformability.py",
            "status": "DERIVED at runtime from canonical TOV + Hinderer chain; conditional/in-sample selection",
        },
        {
            "claim": "Joint NS likelihood",
            "value": f"reduced chi-squared = {results_center['chi2_red']:.3f}",
            "file": "nvg_joint_ns_inference.py",
            "status": (
                "CONDITIONAL_IN_SAMPLE; selected on J0740/GW170817/NICER; "
                + ", ".join(results_center["calibrated_rows"])
                + " excluded as calibration"
            ),
        },
    ]


def _format_markdown(results: dict[str, Any], ledger: list[dict[str, str]]) -> str:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# NVG Runtime Evidence Ledger",
        f"**Generated (UTC):** {timestamp}",
        "",
        "## Canonical executable result",
        "",
        "The table below is populated from one executed TOV + Hinderer chain. "
        "It is not an uncertainty propagation: the current EOS implementation "
        "does not expose an off-anchor M_Omega,0 parameter.",
        "",
        "| Quantity | Runtime value | Semantics |",
        "|---|---:|---|",
        f"| M_Omega,0 input (MeV) | {results['m_omega']:.1f} +/- {results['m_omega_err']:.1f} | lattice/QCD input |",
        f"| M_max (M_sun) | {results['m_max']:.3f} | {results['solver']}; conditional/in-sample selection |",
        f"| R_1.4 (km) | {results['r_14']:.3f} | {results['solver']}; conditional/in-sample selection |",
        f"| Lambda_1.4 | {results['lambda_14']:.1f} | {results['solver']}; conditional/in-sample selection |",
        f"| Conditional/in-sample reduced chi-squared | {results['chi2_red']:.3f} | {results['conditional_rows']} rows; calibration excluded |",
        "",
        "## Canonical selection provenance",
        "",
        "The transition point is selected in-sample from J0740, GW170817, and NICER constraints. "
        "It is conditional model input, not an independent confirmation.",
        "",
        f"- status: `{results['comparison_status']}`",
        f"- selected by: `{results['selection_provenance']['selected_by']}`",
        f"- method: `{results['selection_provenance']['selection_method']}`",
        f"- selected transition parameters: `{results['selection_parameters']}`",
        "- held-out observations: none",
        "",
        "## Evidence ledger",
        "",
        "| Claim | Result | Script/input | Status |",
        "|---|---|---|---|",
    ]
    for item in ledger:
        lines.append(f"| {item['claim']} | {item['value']} | `{item['file']}` | {item['status']} |")
    lines.extend(["", "## Forecasts (not evidence)", ""])
    for key, value in FORECAST.items():
        lines.append(f"- **{key}:** {value}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Cooling is a calibration target in the joint comparison and is not counted in chi-squared.",
            "- Off-anchor uncertainty propagation and inverse QCD reconstruction are unsupported until an independent parameterized EOS chain is implemented.",
            "- A successful process run is not a scientific verification of unrelated claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> dict[str, Any]:
    results = run_forward_model(M_OMEGA_CENTRAL)
    ledger = generate_evidence_ledger(results)
    report = _format_markdown(results, ledger)
    out_path = os.path.join(os.path.dirname(__file__), "..", "NVG_FINAL_REPORT.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print("Runtime evidence ledger generated:", os.path.abspath(out_path))
    print(
        f"Canonical outputs: M_max={results['m_max']:.3f} M_sun, "
        f"R_1.4={results['r_14']:.3f} km, Lambda_1.4={results['lambda_14']:.1f}"
    )
    print(
        f"Conditional/in-sample reduced chi-squared={results['chi2_red']:.3f}; "
        f"calibrated rows excluded={results['calibrated_rows']}"
    )
    return results


if __name__ == "__main__":
    main()
