#!/usr/bin/env python3
"""Joint neutron-star comparison using one canonical executable EOS chain.

The observations below are declared literature inputs.  The three structural
rows are generated at runtime by ``nvg_tidal_deformability`` (TOV + Hinderer).
The cooling dichotomy is retained only as a calibration record: this script
does not contain an independent cooling calculation, so that row is excluded
from the likelihood and cannot improve the reported fit.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

try:
    from nvg_ns_canonical import canonical_selection
except ImportError:  # pragma: no cover - package/direct-script compatibility
    from verification.nvg_ns_canonical import canonical_selection


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


OBSERVATIONS: dict[str, dict[str, Any]] = {
    "M_max": {"value": 2.14, "sigma": 0.10, "source": "NICER PSR J0740+6620"},
    "R_1.4": {"value": 12.2, "sigma": 0.50, "source": "NICER PSR J0030+0451"},
    "Lambda_1.4": {
        "value": 190.0,
        "sigma_upper": 390.0,
        "sigma_lower": 120.0,
        "source": "LIGO GW170817",
    },
    "Cooling_Dichotomy": {
        "value": 1.45,
        "sigma": 0.05,
        "source": "Cas A (Slow) vs Vela (Fast)",
    },
}
# Backward-compatible read-only alias for callers of the former script-level
# name; the values remain declared observations, never model predictions.
observations = OBSERVATIONS

# Explicit evidence semantics for every row.  The cooling value is an observed
# calibration target, not an independent EOS output.
ROW_METADATA: dict[str, dict[str, Any]] = {
    "M_max": {
        "kind": "derived",
        "source": "nvg_tidal_deformability.EOS + solve_tov_tidal",
        "comparison_status": "CONDITIONAL_IN_SAMPLE",
    },
    "R_1.4": {
        "kind": "derived",
        "source": "nvg_tidal_deformability.EOS + solve_tov_tidal",
        "comparison_status": "CONDITIONAL_IN_SAMPLE",
    },
    "Lambda_1.4": {
        "kind": "derived",
        "source": "nvg_tidal_deformability.EOS + solve_tov_tidal",
        "comparison_status": "CONDITIONAL_IN_SAMPLE",
    },
    "Cooling_Dichotomy": {
        "kind": "calibration",
        "source": "observational input; no independent cooling solver in this script",
        "calibration_parameter": "alpha_v / direct-urca threshold",
    },
}
CALIBRATED = {key for key, meta in ROW_METADATA.items() if meta["kind"] == "calibration"}


def compute_nvg_predictions() -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    """Run the canonical EOS once and return predictions plus provenance."""

    try:
        import nvg_tidal_deformability as td
    except ImportError:  # pragma: no cover - package/direct-script compatibility
        from verification import nvg_tidal_deformability as td

    eos = td.EOS(p_match=1.5, Gamma=1.35)
    # Match the canonical tidal entry point's full branch scan so every
    # downstream report interpolates the same runtime sequence.
    pressure_centers = np.logspace(-1.0, 3.4, 120)
    rows = []
    for pressure_center in pressure_centers:
        mass, radius, k2, lam = td.solve_tov_tidal(eos, float(pressure_center))
        if mass > 0.0 and radius > 0.0 and k2 > 0.0 and lam > 0.0:
            rows.append((float(mass), float(radius), float(lam)))
    if not rows:
        raise RuntimeError("canonical EOS produced no valid TOV solutions")

    masses = np.asarray([row[0] for row in rows])
    max_index = int(np.argmax(masses))
    stable = rows[: max_index + 1]
    stable_masses = np.asarray([row[0] for row in stable])
    stable_radii = np.asarray([row[1] for row in stable])
    stable_lambdas = np.asarray([row[2] for row in stable])
    if stable_masses.max() < 1.4:
        raise RuntimeError("canonical EOS branch does not reach 1.4 solar masses")

    def at_mass(target: float, values: np.ndarray) -> float:
        if target < stable_masses.min() or target > stable_masses.max():
            raise RuntimeError(f"canonical EOS branch does not reach {target} solar masses")
        return float(np.interp(target, stable_masses, values))

    predictions: dict[str, float | None] = {
        "M_max": float(masses[max_index]),
        "R_1.4": at_mass(1.4, stable_radii),
        "Lambda_1.4": at_mass(1.4, stable_lambdas),
        "R_1.42": at_mass(1.42, stable_radii) if stable_masses.max() >= 1.42 else None,
        "R_2.08": at_mass(2.08, stable_radii) if stable_masses.max() >= 2.08 else None,
        # There is deliberately no cooling prediction.  The observed value is
        # printed as a calibration target and excluded in run_joint_inference.
        "Cooling_Dichotomy": None,
    }
    metadata = {key: dict(value) for key, value in ROW_METADATA.items()}
    selection = canonical_selection()
    metadata["canonical_selection"] = selection
    metadata["selection_provenance"] = selection["provenance"]
    for key in ROW_METADATA:
        metadata[key].setdefault("selection_provenance", selection["provenance"])
    metadata["M_max"]["stable_rows"] = len(stable)
    metadata["R_1.4"]["stable_rows"] = len(stable)
    metadata["Lambda_1.4"]["stable_rows"] = len(stable)
    return predictions, metadata


def run_joint_inference(
    predictions: dict[str, float | None] | None = None,
    metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute descriptive in-sample pulls from predictions and inputs.

    The transition was selected with the J0740/GW170817/NICER constraints, so
    this statistic is explicitly conditional/in-sample and cannot be reported
    as independent evidence.
    """

    if predictions is None or metadata is None:
        predictions, metadata = compute_nvg_predictions()

    rows: list[dict[str, Any]] = []
    chi_squared_total = 0.0
    in_sample_count = 0
    for key, observation in OBSERVATIONS.items():
        pred = predictions.get(key)
        meta = metadata[key]
        if key in CALIBRATED:
            rows.append(
                {
                    "key": key,
                    "prediction": pred,
                    "observation": observation["value"],
                    "sigma": observation["sigma"],
                    "pull": None,
                    "included": False,
                    "kind": meta["kind"],
                    "in_sample": False,
                    "status": "calibration target excluded from conditional chi-squared",
                }
            )
            continue
        if pred is None:
            raise RuntimeError(f"missing derived prediction for {key}")
        if key == "Lambda_1.4":
            sigma = observation["sigma_upper"] if pred > observation["value"] else observation["sigma_lower"]
        else:
            sigma = observation["sigma"]
        pull = (pred - observation["value"]) / sigma
        chi_squared_total += pull**2
        in_sample_count += 1
        rows.append(
            {
                "key": key,
                "prediction": pred,
                "observation": observation["value"],
                "sigma": sigma,
                "pull": pull,
                "included": True,
                "kind": meta["kind"],
                "in_sample": True,
                "status": "conditional/in-sample derived row",
            }
        )

    dof = in_sample_count
    reduced_chi = chi_squared_total / dof if dof else float("nan")
    return {
        "predictions": predictions,
        "metadata": metadata,
        "rows": rows,
        "chi_squared_total": chi_squared_total,
        "in_sample_count": in_sample_count,
        "dof": dof,
        "reduced_chi": reduced_chi,
        "comparison_status": "CONDITIONAL_IN_SAMPLE",
        "selection_provenance": canonical_selection()["provenance"],
        "status": "COMPATIBLE (conditional/in-sample; transition selected on these constraints)"
        if np.isfinite(reduced_chi) and reduced_chi < 2.0
        else "NOT COMPATIBLE under declared conditional/in-sample rows",
    }


def _print_report(result: dict[str, Any]) -> None:
    print("=" * 70)
    print(" NVG JOINT NS INFERENCE (MULTI-MESSENGER LIKELIHOOD)")
    print("=" * 70)
    print("Runtime model: nvg_tidal_deformability EOS + TOV/Hinderer solver")
    print("Comparison status: CONDITIONAL_IN_SAMPLE (transition selected on J0740/GW170817/NICER)")
    print(f"{'Observable':<20} | {'NVG output':<12} | {'Observation':<15} | Pull / status")
    print("-" * 78)
    for row in result["rows"]:
        key = row["key"]
        obs = OBSERVATIONS[key]
        pred = row["prediction"]
        if row["included"]:
            print(
                f"{key:<20} | {pred:<12.3f} | {obs['value']:<5.2f} +/- {row['sigma']:<5.2f} | "
                f"{row['pull']:>6.2f} sigma"
            )
        else:
            print(
                f"{key:<20} | {'calibration target':<12} | {obs['value']:<5.2f} +/- {row['sigma']:<5.2f} | "
                "excluded (no independent solver); calibration only"
            )
    print("-" * 78)
    print(f"Conditional/in-sample chi-squared: {result['chi_squared_total']:.3f}")
    print(f"Degrees of freedom             : {result['dof']}")
    print(f"Reduced chi-squared            : {result['reduced_chi']:.3f}")
    print(f"Conditional status             : {result['status']}")
    print("Cooling_Dichotomy is a calibration target and cannot improve this value.")
    print("No independent/global-significance claim is made by this script.")
    print("=" * 70)


def main() -> dict[str, Any]:
    result = run_joint_inference()
    _print_report(result)
    return result


if __name__ == "__main__":
    main()
