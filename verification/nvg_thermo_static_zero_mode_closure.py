"""Independent q=0 closure between the static saddle response and the EOS.

At fixed temperature, the homogeneous thermodynamic derivative
chi_T=(partial n/partial mu)_T must agree with the zero-wave-number static
density response after relaxing the same scalar/vector mean-field saddle.
The two sides are evaluated with different integral/derivative methods.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_nonlinear_calibration_response as nonlinear  # noqa: E402
import nvg_thermal_observable_bridge as thermal  # noqa: E402


SCHEMA_VERSION = "nvg-thermo-static-zero-mode.v1"
STATUS_PASS = "PASS_LIVE_THERMO_STATIC_ZERO_MODE"
STATUS_FAIL = "FAIL_LIVE_THERMO_STATIC_ZERO_MODE"
EVIDENCE_WEIGHT = "conditional_cross_sector_consistency"
ANCHORS = ("0.90", "0.93")
MODELS = ("w8", "u16")
TEMPERATURE_MEV = mp.mpf("70")
CHEMICAL_POTENTIALS = (mp.mpf("775"), mp.mpf("875"))
DEGENERACY = 4
HBARC = mp.mpf(thermal.HBARC_MEV_FM)
EXPECTED_ROWS = len(ANCHORS) * len(MODELS) * len(CHEMICAL_POTENTIALS)
MATCH_LIMIT = mp.mpf("2e-7")
SCHUR_LIMIT = mp.mpf("1e-24")
REFINEMENT_LIMIT = mp.mpf("3e-5")


def _mp(value: Any) -> mp.mpf:
    return value if isinstance(value, mp.mpf) else mp.mpf(str(value))


def _number(value: Any, digits: int = 30) -> str:
    value = _mp(value)
    if not mp.isfinite(value):
        raise ValueError("nonfinite scientific output")
    return mp.nstr(value, digits)


def _public(value: Any) -> Any:
    """Convert nested high-precision values into deterministic JSON scalars."""
    if isinstance(value, mp.mpf):
        return _number(value)
    if isinstance(value, Mapping):
        return {str(key): _public(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_public(item) for item in value]
    return value


def _relative(left: Any, right: Any, floor: Any = "1e-40") -> mp.mpf:
    a, b = _mp(left), _mp(right)
    return abs(a - b) / max(abs(a), abs(b), _mp(floor))


def _payload_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _fermi(argument: np.ndarray) -> np.ndarray:
    out = np.empty_like(argument, dtype=float)
    high = argument > 40.0
    low = argument < -40.0
    middle = ~(high | low)
    out[high] = np.exp(-argument[high])
    out[low] = 1.0 - np.exp(argument[low])
    out[middle] = 1.0 / (1.0 + np.exp(argument[middle]))
    return out


def _intervals(mass: float, nu: float, temperature: float, cutoff: float) -> tuple[tuple[float, float], ...]:
    points = [0.0, 600.0, 1200.0, 2400.0, cutoff]
    if nu > mass:
        pf = math.sqrt(max(nu * nu - mass * mass, 0.0))
        width = max(30.0, 24.0 * temperature)
        points.extend((pf, max(0.0, pf - width), min(cutoff, pf + width)))
    values = sorted(set(max(0.0, min(cutoff, value)) for value in points))
    return tuple((left, right) for left, right in zip(values[:-1], values[1:]) if right > left)


def _q0_polarization(mass: Any, nu: Any, temperature: Any, *, order: int, cutoff: float) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Compute Pi_vv, Pi_vs, Pi_ss from the q=0 thermodynamic integrals."""
    m, n, t = float(_mp(mass)), float(_mp(nu)), float(_mp(temperature))
    pref = DEGENERACY / (2.0 * math.pi**2)
    totals = np.zeros(3, dtype=float)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    for low, high in _intervals(m, n, t, float(cutoff)):
        p = (low + high) / 2.0 + (high - low) / 2.0 * nodes
        w = (high - low) / 2.0 * weights
        energy = np.sqrt(p * p + m * m)
        fp = _fermi((energy - n) / t)
        fa = _fermi((energy + n) / t)
        dp = fp * (1.0 - fp) / t
        da = fa * (1.0 - fa) / t
        terms = np.array([
            p * p * (dp + da),
            p * p * m / energy * (dp - da),
            p * p * ((m / energy) ** 2 * (dp + da) - p * p / energy**3 * (fp + fa)),
        ])
        totals += pref * np.sum(w[None, :] * terms, axis=1)
    return tuple(mp.mpf(str(value)) for value in totals)  # type: ignore[return-value]


def _static_q0_response(model: Any, state: Mapping[str, Any], pi: Sequence[Any]) -> dict[str, Any]:
    vv, vs, ss = tuple(_mp(value) for value in pi)
    if vv <= 0:
        raise ValueError("nonpositive Pi_vv")
    mn, mw = _mp(model.MN), _mp(model.momega)
    y, n, uyy = _mp(state["y"]), _mp(state["n_MeV3"]), _mp(state["Uyy_MeV4"])
    g = _mp(model.gomega)
    a0 = g * n / (mw**2 * y**2)
    a = 1 / vv
    b = mn * vs / vv
    dferm = mn**2 * (vs**2 / vv - ss)
    h = -2 * mw**2 * y * a0
    t0 = mw**2 * y**2
    d0 = dferm + uyy - mw**2 * a0**2
    z = _mp(model.W0) ** 2
    # Eliminate the constrained vector field, then the scalar field, by the
    # same Schur-complement algebra as the maintained static-response sector.
    d = a + g**2 / t0
    mix = b + g * h / t0
    c = d0 + h**2 / t0  # Z*q^2 term vanishes exactly at q=0.
    if c <= 0:
        raise ValueError("nonpositive relaxed scalar curvature at q=0")
    schur = d - mix**2 / c
    if schur <= 0:
        raise ValueError("nonpositive relaxed density curvature at q=0")
    hessian = mp.matrix([[a, b, g], [b, d0, h], [g, h, -t0]])
    full_inverse = mp.lu_solve(hessian, mp.matrix([1, 0, 0]))
    chi_static = 1 / schur
    chi_direct = _mp(full_inverse[0])
    return {
        "polarization": tuple(_mp(value) for value in pi),
        "polarization_units": "MeV^2 for each of Pi_vv, Pi_vs, Pi_ss",
        "coefficients": {"a": a, "b": b, "d0": d0, "g": g, "h": h, "t0": t0, "Z": z, "A0_MeV": a0, "Uyy_MeV4": uyy},
        "coefficient_units": {"a": "MeV^-2", "b": "MeV", "d0": "MeV^4", "g": "1", "h": "MeV^3", "t0": "MeV^2", "Z": "MeV^2", "A0_MeV": "MeV", "Uyy_MeV4": "MeV^4"},
        "D_MeVminus2": d,
        "B_MeV": mix,
        "C_MeV4": c,
        "S_MeVminus2": schur,
        "chi_static_MeV2": chi_static,
        "chi_full_hessian_MeV2": chi_direct,
        "full_vs_schur_relative": _relative(chi_direct, chi_static),
    }


def _row(anchor: str, model_name: str, mu: mp.mpf, *, dps: int, nquad: int, cutoff: mp.mpf, q0_order: int, q0_cutoff: float) -> dict[str, Any]:
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        model = nonlinear._build_models(anchor)[model_name]["model"]
        integrator = thermal._integrator_for(TEMPERATURE_MEV, cutoff=cutoff, nquad=nquad)
        nu, y, root_residual, density = thermal._solve_start(integrator, model, mu, mu, _mp(anchor))
        state = thermal._state(integrator, model, mu, nu, y, with_derivatives=True)
        if not (state["local_stable_C_positive"] and state["local_stable_fnn_positive"] and root_residual < mp.mpf("2e-24")):
            raise ValueError("EOS root is unresolved or locally unstable")
        pi = _q0_polarization(model.MN * y, nu, TEMPERATURE_MEV, order=q0_order, cutoff=q0_cutoff)
        response = _static_q0_response(model, state, pi)
        chi_eos = _mp(state["chi1_MeV2"])
        chi_schur = 1 / _mp(state["f_nn"])
        match_error = _relative(response["chi_static_MeV2"], chi_eos)
        schur_error = _relative(response["chi_static_MeV2"], chi_schur)
        chi_eos_schur_error = _relative(chi_eos, chi_schur)
        return {
            "anchor": anchor,
            "model": model_name,
            "T_MeV": _number(TEMPERATURE_MEV),
            "mu_MeV": _number(mu),
            "settings": {"root_dps": dps, "eos_nquad": nquad, "eos_cutoff_MeV": _number(cutoff), "q0_order": q0_order, "q0_cutoff_MeV": _number(q0_cutoff)},
            "state": {"nu_MeV": _number(nu), "y": _number(y), "mstar_MeV": _number(model.MN * y), "n_MeV3": _number(density), "root_residual_relative": _number(root_residual), "locally_stable": True},
            "thermodynamic_side": {"definition": "chi_T=(partial n/partial mu)_T after implicit scalar/vector saddle relaxation", "chi_eos_MeV2": _number(chi_eos), "chi_from_fnn_inverse_MeV2": _number(chi_schur), "chi_eos_vs_fnn_relative": _number(chi_eos_schur_error)},
            "static_q0_side": _public(response),
            "checks": {
                "static_vs_thermodynamic_relative": _number(match_error),
                "static_vs_fnn_schur_relative": _number(schur_error),
                "full_hessian_vs_static_schur_relative": _number(response["full_vs_schur_relative"]),
                "pass": bool(match_error < MATCH_LIMIT and schur_error < MATCH_LIMIT and response["full_vs_schur_relative"] < SCHUR_LIMIT),
            },
        }
    finally:
        mp.mp.dps = old_dps


def _build_result() -> dict[str, Any]:
    mp.mp.dps = 60
    primary: list[dict[str, Any]] = []
    refined: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        for model_name in MODELS:
            for mu in CHEMICAL_POTENTIALS:
                primary.append(_row(anchor, model_name, mu, dps=40, nquad=80, cutoff=mp.mpf("4500"), q0_order=120, q0_cutoff=5000.0))
                refined.append(_row(anchor, model_name, mu, dps=48, nquad=112, cutoff=mp.mpf("5200"), q0_order=180, q0_cutoff=6000.0))
    refined_map = {(r["anchor"], r["model"], r["mu_MeV"]): r for r in refined}
    refinement: list[dict[str, Any]] = []
    for row in primary:
        other = refined_map[(row["anchor"], row["model"], row["mu_MeV"])]
        chi0_error = _relative(row["static_q0_side"]["chi_static_MeV2"], other["static_q0_side"]["chi_static_MeV2"])
        eos_error = _relative(row["thermodynamic_side"]["chi_eos_MeV2"], other["thermodynamic_side"]["chi_eos_MeV2"])
        worst = max(chi0_error, eos_error)
        refinement.append({"anchor": row["anchor"], "model": row["model"], "mu_MeV": row["mu_MeV"], "static_q0_relative_change": _number(chi0_error), "thermodynamic_chi_relative_change": _number(eos_error), "maximum_relative_change": _number(worst), "pass": bool(worst < REFINEMENT_LIMIT)})
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_PASS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "physical_inputs_fixed": True,
        "no_saved_results_as_inputs": True,
        "fit_used": False,
        "coverage": {"expected_rows": EXPECTED_ROWS, "primary_rows": len(primary), "refined_rows": len(refined)},
        "ensemble_and_units": "grand canonical at fixed T,mu; n in MeV^3; chi=(partial n/partial mu)_T and static q=0 chi in MeV^2; hbar=c=1",
        "independence_contract": ["EOS chi uses implicit differentiation of the converged coupled number/gap root in mpmath.", "Static q=0 polarization uses a separately coded piecewise NumPy Gauss-Legendre quadrature, then a full scalar/vector Schur reduction.", "Both calculations use the same fixed action and thermodynamic background; this is a cross-sector consistency check, not independent physical evidence."],
        "interpretation_limits": ["The equality is an expected thermodynamic/response consistency identity; passing it validates a bridge, not the truth of NVG.", "It applies only to the homogeneous grand-canonical isoscalar density channel at q=0 and fixed T on the scanned locally stable branch.", "No finite-q dynamics, equal-time covariance, experimental fit, phase transition, gravity, or energy source is inferred."],
        "primary_rows": primary,
        "refined_rows": refined,
        "refinement": refinement,
        "source_sha256": {"producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "thermal_observable_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(), "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest()},
    }
    controls = {
        "coverage": {"pass": len(primary) == EXPECTED_ROWS and len(refined) == EXPECTED_ROWS},
        "primary_cross_sector_identity": {"max_relative_error": _number(max(_mp(r["checks"]["static_vs_thermodynamic_relative"]) for r in primary)), "pass": all(r["checks"]["pass"] for r in primary)},
        "refined_cross_sector_identity": {"max_relative_error": _number(max(_mp(r["checks"]["static_vs_thermodynamic_relative"]) for r in refined)), "pass": all(r["checks"]["pass"] for r in refined)},
        "quadrature_and_root_refinement": {"max_relative_change": _number(max(_mp(r["maximum_relative_change"]) for r in refinement)), "pass": all(r["pass"] for r in refinement)},
    }
    result["controls"] = controls
    result["all_controls_pass"] = bool(all(c["pass"] for c in controls.values()))
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
        expected = {(a, m, _number(mu)) for a in ANCHORS for m in MODELS for mu in CHEMICAL_POTENTIALS}
        primary, refined = result.get("primary_rows"), result.get("refined_rows")
        if not isinstance(primary, list) or not isinstance(refined, list) or len(primary) != EXPECTED_ROWS or len(refined) != EXPECTED_ROWS:
            return False
        for rows in (primary, refined):
            if {(r.get("anchor"), r.get("model"), r.get("mu_MeV")) for r in rows} != expected:
                return False
            for row in rows:
                eos = _mp(row["thermodynamic_side"]["chi_eos_MeV2"])
                schur_eos = _mp(row["thermodynamic_side"]["chi_from_fnn_inverse_MeV2"])
                static = row["static_q0_side"]
                pi_vv, pi_vs, pi_ss = map(_mp, static["polarization"])
                c = static["coefficients"]
                a, b, d0, g, h, t0 = (_mp(c[key]) for key in ("a", "b", "d0", "g", "h", "t0"))
                D = 1 / pi_vv + g**2 / t0
                B = b + g * h / t0
                C = d0 + h**2 / t0
                S = D - B**2 / C
                chi = 1 / S
                if _relative(chi, static["chi_static_MeV2"]) > mp.mpf("1e-20"):
                    return False
                if _relative(chi, static["chi_full_hessian_MeV2"]) > SCHUR_LIMIT:
                    return False
                if not (eos > 0 and _mp(static["C_MeV4"]) > 0 and S > 0):
                    return False
                if _relative(chi, eos) >= MATCH_LIMIT or _relative(eos, schur_eos) >= MATCH_LIMIT:
                    return False
                if row.get("checks", {}).get("pass") is not True:
                    return False
        refined_map = {(r["anchor"], r["model"], r["mu_MeV"]): r for r in refined}
        comparisons = result.get("refinement")
        if not isinstance(comparisons, list) or len(comparisons) != EXPECTED_ROWS:
            return False
        if {(r.get("anchor"), r.get("model"), r.get("mu_MeV")) for r in comparisons} != expected:
            return False
        primary_map = {(r["anchor"], r["model"], r["mu_MeV"]): r for r in primary}
        for item in comparisons:
            key = (item["anchor"], item["model"], item["mu_MeV"])
            p, q = primary_map[key], refined_map[key]
            actual = max(_relative(p["static_q0_side"]["chi_static_MeV2"], q["static_q0_side"]["chi_static_MeV2"]), _relative(p["thermodynamic_side"]["chi_eos_MeV2"], q["thermodynamic_side"]["chi_eos_MeV2"]))
            if item.get("pass") is not True or actual >= REFINEMENT_LIMIT or _relative(actual, item["maximum_relative_change"], floor="1e-40") > mp.mpf("1e-8"):
                return False
        if any(control.get("pass") is not True for control in result.get("controls", {}).values()):
            return False
        hashes = {"producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "thermal_observable_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(), "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest()}
        if result.get("source_sha256") != hashes:
            return False
        payload = dict(result)
        digest = payload.pop("integrity_sha256", None)
        return isinstance(digest, str) and hashlib.sha256(_payload_bytes(payload)).hexdigest() == digest
    except (ArithmeticError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "nvg_thermo_static_zero_mode_results.json")
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
