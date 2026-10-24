"""No-fit identifiability audit from NVG's live *static* spatial response.

This module intentionally does not invent a real-frequency pole, damping rate,
or energy port.  It rebuilds the maintained finite-temperature static response
and records exactly what a minimal one-inertia completion would predict only
*relatively*, as well as the coefficients that static information cannot fix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SCHEMA_VERSION = "nvg-dynamic-response-identifiability.v1"
STATUS_PASS = "PASS_STATIC_IDENTIFIABILITY_AUDIT"
STATUS_FAIL = "FAIL_STATIC_IDENTIFIABILITY_AUDIT"
EVIDENCE_WEIGHT = "derived_conditional_not_real_frequency_measurement"
Q_GRID = ("0", "50", "100", "200")
EXPECTED_STATES = 8
EXPECTED_ROWS = EXPECTED_STATES * len(Q_GRID)
SPATIAL_SOURCE = HERE / "nvg_thermal_spatial_response.py"


def _number(value: Any) -> str:
    return format(float(value), ".17g")


def _payload_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _finite_positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _state_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (str(row["anchor"]), str(row["T_MeV"]), str(row["mu_MeV"]), str(row["model"]))


def _derive(live: Mapping[str, Any], *, spatial_source: Path) -> dict[str, Any]:
    """Reduce a freshly rebuilt static result without reading any result file."""
    if live.get("status") != "PASS_LIVE_THERMAL_SPATIAL_RESPONSE":
        raise ValueError("upstream static response did not pass")
    if live.get("no_saved_results_as_inputs") is not True or live.get("physical_inputs_fixed") is not True:
        raise ValueError("upstream provenance contract failed")

    groups: dict[tuple[str, str, str, str], dict[str, Mapping[str, Any]]] = {}
    for raw in live.get("rows", []):
        if not isinstance(raw, Mapping) or str(raw.get("scale")) != "1":
            continue
        q = str(raw.get("q_MeV"))
        if q not in Q_GRID:
            continue
        groups.setdefault(_state_key(raw), {})[q] = raw
    if len(groups) != EXPECTED_STATES or any(set(rows) != set(Q_GRID) for rows in groups.values()):
        raise ValueError("incomplete static grid")

    states: list[dict[str, Any]] = []
    max_inverse_identity = 0.0
    for key in sorted(groups):
        rows = groups[key]
        q0 = rows["0"]
        chi0 = float(q0["chi_nonlocal_MeV2"])
        s0 = float(q0["S_MeVminus2"])
        if not _finite_positive(chi0) or not _finite_positive(s0):
            raise ValueError("nonpositive q=0 static response")
        points: list[dict[str, Any]] = []
        for q in Q_GRID:
            row = rows[q]
            chi = float(row["chi_nonlocal_MeV2"])
            stiffness = float(row["S_MeVminus2"])
            if not _finite_positive(chi) or not _finite_positive(stiffness):
                raise ValueError("nonpositive static response")
            inverse_error = abs(chi * stiffness - 1.0)
            max_inverse_identity = max(max_inverse_identity, inverse_error)
            # Conditional ansatz only:
            # K(omega,q)=S(q)-I*omega^2, I>0 independent of q and no damping.
            # I cancels in omega(q)/omega(0), but its value is not supplied by
            # the static EOS.  This is a normalized shape, not a frequency.
            relative_frequency = math.sqrt(stiffness / s0)
            same_from_chi = math.sqrt(chi0 / chi)
            points.append(
                {
                    "q_MeV": q,
                    "chi_static_MeV2": _number(chi),
                    "S_static_MeVminus2": _number(stiffness),
                    "static_stiffening_over_q0": _number(stiffness / s0 - 1.0),
                    "relative_frequency_if_constant_inertia": _number(relative_frequency),
                    "relative_frequency_from_chi": _number(same_from_chi),
                    "inverse_identity_absolute": _number(inverse_error),
                }
            )
        states.append(
            {
                "anchor": key[0],
                "T_MeV": key[1],
                "mu_MeV": key[2],
                "model": key[3],
                "q0_chi_static_MeV2": _number(chi0),
                "q0_S_static_MeVminus2": _number(s0),
                "points": points,
            }
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_PASS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "physical_inputs_fixed": True,
        "no_saved_results_as_inputs": True,
        "fit_used": False,
        "upstream_provenance": {
            "producer": "verification/nvg_thermal_spatial_response.py",
            "upstream_status": live["status"],
            "upstream_integrity_sha256": live.get("integrity_sha256"),
            "upstream_source_sha256": live.get("source_sha256"),
            "selected_scale": "1",
            "selected_q_grid_MeV": list(Q_GRID),
        },
        "coverage": {"state_count": len(states), "point_count": sum(len(s["points"]) for s in states)},
        "minimal_completion": {
            "conditional_kernel": "K(omega,q)=S_static(q)-I*omega^2",
            "assumptions": [
                "single positive inertial coefficient I independent of q over the displayed grid",
                "no damping and no additional poles",
                "the density-source normalization is the same at all q",
            ],
            "what_static_data_fix": "S_static(q)=1/chi_static(q) and, under the listed ansatz only, omega(q)/omega(0)=sqrt(S(q)/S(0))",
            "what_static_data_do_not_fix": [
                "I and therefore every absolute frequency",
                "damping or a spectral linewidth",
                "extra dynamical poles, memory kernels, or finite-temperature noise",
                "external-source coupling, finite-volume normalization, load port, power, or efficiency",
            ],
            "real_frequency_prediction": False,
            "device_or_power_prediction": False,
        },
        "states": states,
        "controls": {
            "complete_grid": {"pass": len(states) == EXPECTED_STATES and sum(len(s["points"]) for s in states) == EXPECTED_ROWS},
            "positive_static_stiffness": {"pass": all(_finite_positive(p["S_static_MeVminus2"]) and _finite_positive(p["chi_static_MeV2"]) for s in states for p in s["points"])},
            "inverse_identity": {"max_absolute": _number(max_inverse_identity), "pass": max_inverse_identity < 2e-8},
            "relative_shape_identity": {
                "max_absolute": _number(max(abs(float(p["relative_frequency_if_constant_inertia"]) - float(p["relative_frequency_from_chi"])) for s in states for p in s["points"])),
                "pass": all(abs(float(p["relative_frequency_if_constant_inertia"]) - float(p["relative_frequency_from_chi"])) < 2e-12 for s in states for p in s["points"]),
            },
            "underdetermination_explicit": {"pass": True, "reason": "no value for I, damping, source coupling, or output port is inserted"},
        },
        "interpretation_limits": [
            "The displayed relative shape follows only after the explicitly conditional constant-inertia, zero-damping ansatz.",
            "A static susceptibility does not determine a real-frequency response or equal-time fluctuation spectrum.",
            "This audit is not an experimental measurement, a proof of a covariant action, a black-hole/cosmology result, or a power-generator calculation.",
            "The upstream states are local static branches; they are not a global phase-equilibrium certificate.",
        ],
        "source_sha256": {
            "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "thermal_spatial_response": hashlib.sha256(spatial_source.resolve().read_bytes()).hexdigest(),
        },
    }
    result["all_controls_pass"] = bool(all(item.get("pass") is True for item in result["controls"].values()))
    result["status"] = STATUS_PASS if result["all_controls_pass"] else STATUS_FAIL
    result["integrity_sha256"] = hashlib.sha256(_payload_bytes(result)).hexdigest()
    return result


def build_result() -> dict[str, Any]:
    # Deliberately lazy: validating a stored output should not need the costly
    # SymPy dependency tree or execute an upstream scientific calculation.
    import nvg_thermal_spatial_response as spatial

    return _derive(spatial.build_result(), spatial_source=Path(spatial.__file__))


def validate_result(result: Any) -> bool:
    try:
        if not isinstance(result, Mapping):
            return False
        if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != STATUS_PASS:
            return False
        if result.get("physical_inputs_fixed") is not True or result.get("no_saved_results_as_inputs") is not True or result.get("fit_used") is not False:
            return False
        if result.get("coverage") != {"state_count": EXPECTED_STATES, "point_count": EXPECTED_ROWS}:
            return False
        if result.get("minimal_completion", {}).get("real_frequency_prediction") is not False or result.get("minimal_completion", {}).get("device_or_power_prediction") is not False:
            return False
        if any(item.get("pass") is not True for item in result.get("controls", {}).values()):
            return False
        states = result.get("states")
        if not isinstance(states, list) or len(states) != EXPECTED_STATES:
            return False
        for state in states:
            points = state.get("points", [])
            if [p.get("q_MeV") for p in points] != list(Q_GRID):
                return False
            chi0 = float(state["q0_chi_static_MeV2"])
            s0 = float(state["q0_S_static_MeVminus2"])
            if not _finite_positive(chi0) or not _finite_positive(s0):
                return False
            for point in points:
                chi, stiffness = float(point["chi_static_MeV2"]), float(point["S_static_MeVminus2"])
                relative = float(point["relative_frequency_if_constant_inertia"])
                if not _finite_positive(chi) or not _finite_positive(stiffness):
                    return False
                if abs(chi * stiffness - 1.0) > 2e-8:
                    return False
                if abs(relative - math.sqrt(stiffness / s0)) > 2e-12 or abs(relative - math.sqrt(chi0 / chi)) > 2e-12:
                    return False
        expected_sources = {
            "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "thermal_spatial_response": hashlib.sha256(SPATIAL_SOURCE.read_bytes()).hexdigest(),
        }
        if result.get("source_sha256") != expected_sources:
            return False
        payload = dict(result)
        digest = payload.pop("integrity_sha256", None)
        return isinstance(digest, str) and hashlib.sha256(_payload_bytes(payload)).hexdigest() == digest
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "nvg_dynamic_response_identifiability_results.json")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate:
        ok = validate_result(json.loads(args.validate.read_text(encoding="utf-8")))
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    result = build_result()
    _write_json(result, args.output)
    print(result["status"])
    return 0 if result["status"] == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
