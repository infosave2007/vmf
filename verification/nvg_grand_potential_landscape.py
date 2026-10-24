"""Bounded grand-potential landscape audit for the homogeneous thermal NVG EOS.

For one fixed action and fixed (T, mu), eliminate the vector/number equation
at each scalar-field value y, scan the resulting on-shell pressure, and locate
stationary points by gap-equation sign changes.  This is a finite-domain
mean-field landscape scan, not a proof of global phase equilibrium.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_nonlinear_calibration_response as nonlinear  # noqa: E402
import nvg_thermal_observable_bridge as thermal  # noqa: E402


SCHEMA_VERSION = "nvg-grand-potential-landscape.v1"
STATUS_PASS = "PASS_LIVE_GRAND_POTENTIAL_LANDSCAPE"
STATUS_FAIL = "FAIL_LIVE_GRAND_POTENTIAL_LANDSCAPE"
EVIDENCE_WEIGHT = "bounded_homogeneous_mean_field_landscape"
ANCHORS = ("0.90", "0.93")
MODELS = ("w8", "u16")
TEMPERATURE_MEV = mp.mpf("70")
CHEMICAL_POTENTIALS = (mp.mpf("775"), mp.mpf("875"))
Y_MIN = mp.mpf("0.15")
Y_MAX = mp.mpf("3.00")
PRIMARY_STEP = mp.mpf("0.05")
REFINED_STEP = mp.mpf("0.025")
EXPECTED_ROWS = len(ANCHORS) * len(MODELS) * len(CHEMICAL_POTENTIALS)
MAX_ROOTS_EXPECTED_IN_THIS_SCAN = 8


def _mp(value: Any) -> mp.mpf:
    return value if isinstance(value, mp.mpf) else mp.mpf(str(value))


def _number(value: Any, digits: int = 28) -> str:
    value = _mp(value)
    if not mp.isfinite(value):
        raise ValueError("nonfinite scientific output")
    return mp.nstr(value, digits)


def _relative(left: Any, right: Any, floor: Any = "1e-40") -> mp.mpf:
    a, b = _mp(left), _mp(right)
    return abs(a - b) / max(abs(a), abs(b), _mp(floor))


def _payload_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _on_shell_point(
    integrator: thermal.ThermalIntegrator,
    model: Any,
    mu: mp.mpf,
    y: mp.mpf,
    *,
    nu_seed: mp.mpf | None = None,
) -> dict[str, mp.mpf]:
    """Solve the number equation at fixed y and evaluate pressure and gap."""
    if not (Y_MIN <= y <= Y_MAX):
        raise ValueError("scalar field lies outside the declared scan interval")

    def number_residual(nu: Any) -> mp.mpf:
        return thermal._residuals(integrator, model, mu, _mp(nu), y)[0]

    center = mu if nu_seed is None else nu_seed
    try:
        nu = mp.findroot(
            number_residual,
            (center - mp.mpf("30"), center + mp.mpf("30")),
            solver="secant",
            tol=mp.eps ** mp.mpf("0.70"),
            maxsteps=60,
            verify=True,
        )
    except Exception as exc:
        raise ValueError(f"number equation failed at y={_number(y, 8)}: {type(exc).__name__}: {exc}") from exc
    residual_number, gap = thermal._residuals(integrator, model, mu, nu, y)
    data = integrator.integrals(nu, y)
    if not (mp.isfinite(data["n"]) and data["n"] > 0):
        raise ValueError("nonpositive density on constrained scalar scan")
    potential, potential_y, _ = thermal._model_potential(model, y)
    vector = model.Cv * data["n"] ** 2 / (2 * y**2)
    pressure = data["pressure"] - potential + vector
    gap_scale = max(abs(model.MN * data["ns"]), abs(potential_y), abs(model.Cv * data["n"] ** 2 / y**3), mp.mpf(1))
    number_scale = max(abs(nu), abs(mu), mp.mpf(1))
    return {
        "nu": _mp(nu),
        "n": data["n"],
        "pressure": pressure,
        "gap": gap,
        "gap_relative": abs(gap) / gap_scale,
        "number_residual_relative": abs(residual_number) / number_scale,
        "gap_scale": gap_scale,
    }


def _scan_one(
    anchor: str,
    model_name: str,
    mu: mp.mpf,
    *,
    dps: int,
    nquad: int,
    cutoff: mp.mpf,
    step: mp.mpf,
) -> dict[str, Any]:
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        model = nonlinear._build_models(anchor)[model_name]["model"]
        integrator = thermal._integrator_for(TEMPERATURE_MEV, cutoff=cutoff, nquad=nquad)
        intervals = (Y_MAX - Y_MIN) / step
        count_steps = int(mp.floor(intervals + mp.mpf("1e-18")))
        grid = [Y_MIN + i * step for i in range(count_steps + 1)]
        if abs(grid[-1] - Y_MAX) > step * mp.mpf("1e-12"):
            grid.append(Y_MAX)
        else:
            grid[-1] = Y_MAX
        points: list[dict[str, mp.mpf]] = []
        failures: list[dict[str, str]] = []
        nu_seed: mp.mpf | None = None
        for y in grid:
            try:
                point = _on_shell_point(integrator, model, mu, y, nu_seed=nu_seed)
                point["y"] = y
                points.append(point)
                nu_seed = point["nu"]
            except Exception as exc:
                failures.append({"y": _number(y, 8), "error": f"{type(exc).__name__}: {exc}"})
                nu_seed = None
        if len(points) != len(grid):
            raise ValueError(f"landscape scan incomplete: {len(points)}/{len(grid)} points")

        brackets: list[tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]] = []
        for left, right in zip(points, points[1:]):
            gl, gr = left["gap"], right["gap"]
            if gl == 0 or gr == 0 or gl * gr < 0:
                brackets.append((left["y"], right["y"], gl, gr))

        roots: list[dict[str, Any]] = []
        for y_left, y_right, gap_left, gap_right in brackets:
            def gap_at_y(y_value: Any) -> mp.mpf:
                return _on_shell_point(integrator, model, mu, _mp(y_value))["gap"]

            try:
                root_y = mp.findroot(
                    gap_at_y,
                    (y_left, y_right),
                    solver="secant",
                    tol=mp.eps ** mp.mpf("0.70"),
                    maxsteps=70,
                    verify=True,
                )
                root = _on_shell_point(integrator, model, mu, root_y)
                delta = min(step / 5, (root_y - Y_MIN) / 3, (Y_MAX - root_y) / 3)
                if delta <= 0:
                    raise ValueError("stationary point too close to scan boundary")
                p_minus = _on_shell_point(integrator, model, mu, root_y - delta)["pressure"]
                p_plus = _on_shell_point(integrator, model, mu, root_y + delta)["pressure"]
                p_second = (p_plus - 2 * root["pressure"] + p_minus) / delta**2
                is_maximum = bool(p_second < 0 and gap_left < 0 < gap_right)
                check_y = root_y + mp.mpf("0.02")
                check_h = mp.mpf("1e-5")
                check_state = _on_shell_point(integrator, model, mu, check_y)
                check_pressure_plus = _on_shell_point(integrator, model, mu, check_y + check_h)["pressure"]
                check_pressure_minus = _on_shell_point(integrator, model, mu, check_y - check_h)["pressure"]
                pressure_gradient = (check_pressure_plus - check_pressure_minus) / (2 * check_h)
                pressure_gradient_identity_error = _relative(pressure_gradient, -check_state["gap"])
                roots.append({
                    "y": root_y,
                    "nu": root["nu"],
                    "n_MeV3": root["n"],
                    "pressure_MeV4": root["pressure"],
                    "pressure_MeV_fm3": root["pressure"] / _mp(thermal.HBARC_MEV_FM) ** 3,
                    "gap_residual_relative": root["gap_relative"],
                    "number_residual_relative": root["number_residual_relative"],
                    "gap_sign_left": "negative" if gap_left < 0 else "positive" if gap_left > 0 else "zero",
                    "gap_sign_right": "negative" if gap_right < 0 else "positive" if gap_right > 0 else "zero",
                    "pressure_second_derivative_MeV4": p_second,
                    "pressure_gradient_identity_relative": pressure_gradient_identity_error,
                    "pressure_local_maximum": is_maximum,
                    "bracket_y": [_number(y_left, 10), _number(y_right, 10)],
                })
            except Exception as exc:
                raise ValueError(f"failed to refine a gap zero in [{_number(y_left)},{_number(y_right)}]: {exc}") from exc

        roots.sort(key=lambda root: root["y"])
        maxima = [root for root in roots if root["pressure_local_maximum"]]
        boundary_pressures = [points[0]["pressure"], points[-1]["pressure"]]
        all_candidates = maxima + [
            {"pressure_MeV4": boundary_pressures[0], "candidate_type": "lower_scan_boundary"},
            {"pressure_MeV4": boundary_pressures[1], "candidate_type": "upper_scan_boundary"},
        ]
        preferred = max(all_candidates, key=lambda item: item["pressure_MeV4"])
        profile_sign_changes = sum(1 for left, right in zip(points, points[1:]) if left["gap"] * right["gap"] < 0)
        row = {
            "anchor": anchor,
            "model": model_name,
            "T_MeV": _number(TEMPERATURE_MEV),
            "mu_MeV": _number(mu),
            "settings": {"dps": dps, "nquad": nquad, "cutoff_MeV": _number(cutoff), "y_min": _number(Y_MIN), "y_max": _number(Y_MAX), "y_step": _number(step)},
            "scan": {
                "grid_points": len(grid),
                "successful_points": len(points),
                "failed_points": failures,
                "gap_sign_change_count": profile_sign_changes,
                "gap_zero_bracket_count": len(brackets),
                "stationary_points": [
                    {
                        "y": _number(root["y"]),
                        "nu_MeV": _number(root["nu"]),
                        "n_MeV3": _number(root["n_MeV3"]),
                        "pressure_MeV4": _number(root["pressure_MeV4"]),
                        "pressure_MeV_fm3": _number(root["pressure_MeV_fm3"]),
                        "gap_residual_relative": _number(root["gap_residual_relative"]),
                        "number_residual_relative": _number(root["number_residual_relative"]),
                        "gap_sign_left": root["gap_sign_left"],
                        "gap_sign_right": root["gap_sign_right"],
                        "pressure_second_derivative_MeV4": _number(root["pressure_second_derivative_MeV4"]),
                        "pressure_gradient_identity_relative": _number(root["pressure_gradient_identity_relative"]),
                        "pressure_local_maximum": root["pressure_local_maximum"],
                        "bracket_y": root["bracket_y"],
                    }
                    for root in roots
                ],
                "boundary_pressure_MeV4": [_number(value) for value in boundary_pressures],
                "preferred_candidate": {
                    "type": preferred.get("candidate_type", "interior_stationary_maximum"),
                    "y": _number(preferred["y"]) if "y" in preferred else None,
                    "pressure_MeV4": _number(preferred["pressure_MeV4"]),
                },
                "multiple_homogeneous_pressure_maxima_detected": len(maxima) > 1,
                "single_interior_maximum_beats_boundaries": len(maxima) == 1 and preferred.get("candidate_type") is None,
                "scan_profile": [
                    {"y": _number(point["y"], 8), "gap_MeV4": _number(point["gap"], 12), "pressure_MeV4": _number(point["pressure"], 16)}
                    for point in points
                ],
            },
        }
        return row
    finally:
        mp.mp.dps = old_dps


def _build_result() -> dict[str, Any]:
    # Keep the cross-resolution comparison above either individual solver's
    # precision so sub-1e-15 changes are not rounded to an artificial zero.
    mp.mp.dps = 60
    primary: list[dict[str, Any]] = []
    refined: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        for model in MODELS:
            for mu in CHEMICAL_POTENTIALS:
                primary.append(_scan_one(anchor, model, mu, dps=32, nquad=64, cutoff=mp.mpf("4500"), step=PRIMARY_STEP))
                refined.append(_scan_one(anchor, model, mu, dps=42, nquad=96, cutoff=mp.mpf("5200"), step=REFINED_STEP))
    refined_by_key = {(row["anchor"], row["model"], row["mu_MeV"]): row for row in refined}
    comparisons: list[dict[str, Any]] = []
    for row in primary:
        ref = refined_by_key[(row["anchor"], row["model"], row["mu_MeV"])]
        roots = row["scan"]["stationary_points"]
        refined_roots = ref["scan"]["stationary_points"]
        root_count_matches = len(roots) == len(refined_roots)
        root_differences: list[mp.mpf] = []
        if root_count_matches:
            for root, other in zip(roots, refined_roots):
                root_differences.extend([
                    _relative(root["y"], other["y"]),
                    _relative(root["pressure_MeV4"], other["pressure_MeV4"]),
                ])
        max_difference = max(root_differences, default=mp.inf)
        comparisons.append({
            "anchor": row["anchor"],
            "model": row["model"],
            "mu_MeV": row["mu_MeV"],
            "primary_stationary_count": len(roots),
            "refined_stationary_count": len(refined_roots),
            "max_relative_y_or_pressure_difference": _number(max_difference) if mp.isfinite(max_difference) else "Infinity",
            "pass": bool(root_count_matches and max_difference < mp.mpf("3e-3")),
        })

    row_passes = all(
        row["scan"]["successful_points"] == row["scan"]["grid_points"]
        and row["scan"]["single_interior_maximum_beats_boundaries"]
        and len(row["scan"]["stationary_points"]) == 1
        and row["scan"]["stationary_points"][0]["pressure_local_maximum"]
        and _mp(row["scan"]["stationary_points"][0]["gap_residual_relative"]) < mp.mpf("2e-20")
        and _mp(row["scan"]["stationary_points"][0]["pressure_gradient_identity_relative"]) < mp.mpf("1e-5")
        for row in primary
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_PASS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "physical_inputs_fixed": True,
        "no_saved_results_as_inputs": True,
        "fit_used": False,
        "coverage": {"expected_rows": EXPECTED_ROWS, "primary_rows": len(primary), "refined_rows": len(refined)},
        "comparison_precision_dps": 60,
        "ensemble": "homogeneous grand-canonical mean-field branch at fixed T and mu; pressure is minus grand-potential density",
        "selection_rule": "among stationary solutions of one fixed model at identical T,mu, the equilibrium candidate is the solution with maximal pressure; W8 and U16 are distinct model parameterizations, not phases to compare as coexisting states",
        "scan_domain": {"y_min": _number(Y_MIN), "y_max": _number(Y_MAX), "meaning": "bounded effective-mass/scalar-field interval; does not cover other fields, inhomogeneous states, or y outside this interval"},
        "caveats": [
            "A finite grid and sign-change scan can miss tangential/double gap roots between grid points; this is not a proof that no other stationary roots exist.",
            "The scan compares homogeneous roots within each fixed action and thermodynamic point only; it does not establish global equilibrium of real nuclear matter or a coexistence line.",
            "No Maxwell construction, nucleation dynamics, latent heat, hysteresis rate, transport coefficient, gravity, or energy-production process is inferred.",
        ],
        "primary_rows": primary,
        "refined_rows": refined,
        "refinement": comparisons,
        "source_sha256": {
            "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "thermal_observable_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(),
            "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest(),
        },
    }
    controls = {
        "coverage": {"pass": len(primary) == EXPECTED_ROWS and len(refined) == EXPECTED_ROWS},
        "complete_scans_one_maximum_per_fixed_model_state": {"pass": row_passes},
        "refinement": {
            "max_relative_y_or_pressure_difference": _number(max(_mp(item["max_relative_y_or_pressure_difference"]) for item in comparisons)) if comparisons else "Infinity",
            "pass": all(item["pass"] for item in comparisons),
        },
    }
    result["controls"] = controls
    result["all_controls_pass"] = bool(all(item["pass"] for item in controls.values()))
    result["status"] = STATUS_PASS if result["all_controls_pass"] else STATUS_FAIL
    result["integrity_sha256"] = hashlib.sha256(_payload_bytes(result)).hexdigest()
    return result


def build_result() -> dict[str, Any]:
    return _build_result()


def validate_result(result: Any) -> bool:
    with mp.workdps(60):
        return _validate_result(result)


def _validate_result(result: Any) -> bool:
    try:
        if not isinstance(result, Mapping) or result.get("schema_version") != SCHEMA_VERSION or result.get("status") != STATUS_PASS:
            return False
        if result.get("physical_inputs_fixed") is not True or result.get("no_saved_results_as_inputs") is not True or result.get("fit_used") is not False:
            return False
        if result.get("coverage") != {"expected_rows": EXPECTED_ROWS, "primary_rows": EXPECTED_ROWS, "refined_rows": EXPECTED_ROWS}:
            return False
        rows = result.get("primary_rows")
        refined = result.get("refined_rows")
        if not isinstance(rows, list) or not isinstance(refined, list) or len(rows) != EXPECTED_ROWS or len(refined) != EXPECTED_ROWS:
            return False
        expected = {(a, m, _number(mu)) for a in ANCHORS for m in MODELS for mu in CHEMICAL_POTENTIALS}
        for collection in (rows, refined):
            if {(r.get("anchor"), r.get("model"), r.get("mu_MeV")) for r in collection} != expected:
                return False
            for row in collection:
                scan = row["scan"]
                roots = scan["stationary_points"]
                if scan["successful_points"] != scan["grid_points"] or scan["failed_points"]:
                    return False
                profile = scan["scan_profile"]
                if not isinstance(profile, list) or len(profile) != scan["grid_points"]:
                    return False
                sign_changes = sum(
                    1
                    for left, right in zip(profile, profile[1:])
                    if _mp(left["gap_MeV4"]) * _mp(right["gap_MeV4"]) < 0
                )
                if sign_changes != scan["gap_sign_change_count"] or scan["gap_zero_bracket_count"] != sign_changes:
                    return False
                if len(roots) != 1 or roots[0].get("pressure_local_maximum") is not True:
                    return False
                if _mp(roots[0]["gap_residual_relative"]) >= mp.mpf("2e-20"):
                    return False
                if roots[0]["gap_sign_left"] != "negative" or roots[0]["gap_sign_right"] != "positive":
                    return False
                if _mp(roots[0]["pressure_second_derivative_MeV4"]) >= 0:
                    return False
                if _mp(roots[0]["pressure_gradient_identity_relative"]) >= mp.mpf("1e-5"):
                    return False
                if scan["single_interior_maximum_beats_boundaries"] is not True or scan["multiple_homogeneous_pressure_maxima_detected"] is not False:
                    return False
                preferred = scan["preferred_candidate"]
                if preferred.get("type") != "interior_stationary_maximum" or _relative(preferred["y"], roots[0]["y"]) > mp.mpf("1e-20"):
                    return False
                if any(_mp(boundary) >= _mp(roots[0]["pressure_MeV4"]) for boundary in scan["boundary_pressure_MeV4"]):
                    return False
        comparisons = result.get("refinement")
        if not isinstance(comparisons, list) or len(comparisons) != EXPECTED_ROWS:
            return False
        primary_by_key = {(row["anchor"], row["model"], row["mu_MeV"]): row for row in rows}
        refined_by_key = {(row["anchor"], row["model"], row["mu_MeV"]): row for row in refined}
        if {(item.get("anchor"), item.get("model"), item.get("mu_MeV")) for item in comparisons} != expected:
            return False
        for item in comparisons:
            key = (item["anchor"], item["model"], item["mu_MeV"])
            first = primary_by_key[key]["scan"]["stationary_points"][0]
            second = refined_by_key[key]["scan"]["stationary_points"][0]
            actual = max(_relative(first["y"], second["y"]), _relative(first["pressure_MeV4"], second["pressure_MeV4"]))
            if item.get("pass") is not True or actual >= mp.mpf("3e-3"):
                return False
            if _relative(actual, item["max_relative_y_or_pressure_difference"], floor="1e-40") > mp.mpf("1e-8"):
                return False
        if any(item.get("pass") is not True for item in result.get("controls", {}).values()):
            return False
        hashes = {
            "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "thermal_observable_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(),
            "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest(),
        }
        if result.get("source_sha256") != hashes:
            return False
        payload = dict(result)
        digest = payload.pop("integrity_sha256", None)
        return isinstance(digest, str) and hashlib.sha256(_payload_bytes(payload)).hexdigest() == digest
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "nvg_grand_potential_landscape_results.json")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate:
        ok = validate_result(json.loads(args.validate.read_text(encoding="utf-8")))
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    result = build_result()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["status"])
    return 0 if result["status"] == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
