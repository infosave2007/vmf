"""No-fit adiabatic sound-speed bridge on NVG's finite-temperature EOS.

Only the hydrodynamic, local-equilibrium limit is calculated.  The script does
not infer a microscopic pole, damping rate, finite-size mode, or device power.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_nonlinear_calibration_response as nonlinear  # noqa: E402
import nvg_thermal_observable_bridge as thermal  # noqa: E402


SCHEMA_VERSION = "nvg-adiabatic-sound-bridge.v1"
STATUS_PASS = "PASS_LIVE_ADIABATIC_SOUND_BRIDGE"
STATUS_FAIL = "FAIL_LIVE_ADIABATIC_SOUND_BRIDGE"
EVIDENCE_WEIGHT = "derived_local_equilibrium_hydrodynamic_limit"
ANCHORS = ("0.90", "0.93")
MODELS = ("w8", "u16")
TEMPERATURE_MEV = mp.mpf("70")
CHEMICAL_POTENTIALS = (mp.mpf("775"), mp.mpf("875"))
EXPECTED_ROWS = len(ANCHORS) * len(MODELS) * len(CHEMICAL_POTENTIALS)


def _mp(value: Any) -> mp.mpf:
    if isinstance(value, mp.mpf):
        return value
    return mp.mpf(str(value))


def _number(value: Any, digits: int = 30) -> str:
    value = _mp(value)
    if not mp.isfinite(value):
        raise ValueError("nonfinite scientific output")
    return mp.nstr(value, digits)


def _relative(left: Any, right: Any, floor: Any = "1e-40") -> mp.mpf:
    a, b = _mp(left), _mp(right)
    return abs(a - b) / max(abs(a), abs(b), _mp(floor))


def _payload_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _state(model: Any, temperature: mp.mpf, chemical_potential: mp.mpf, seed: tuple[mp.mpf, mp.mpf], *, nquad: int, cutoff: mp.mpf) -> dict[str, Any]:
    """Solve the local thermal branch with explicitly scaled equations.

    The upstream root helper is intentionally not used here: its rounded cache
    is appropriate for its own bounded production grid, but can mask a small
    finite-difference displacement in this derivative audit.
    """
    integrator = thermal._integrator_for(temperature, cutoff=cutoff, nquad=nquad)
    scale_one, scale_two = model.MN, model.MN**4

    def equations(nu: Any, y: Any) -> tuple[mp.mpf, mp.mpf]:
        r1, r2 = thermal._residuals(integrator, model, chemical_potential, _mp(nu), _mp(y))
        return r1 / scale_one, r2 / scale_two

    try:
        nu, y = mp.findroot(equations, seed, solver="mdnewton", tol=mp.eps**mp.mpf("0.70"), maxsteps=90, verify=True)
    except Exception as exc:
        raise ValueError(f"canonical root failed: {type(exc).__name__}: {exc}") from exc
    residuals = thermal._residuals(integrator, model, chemical_potential, nu, y)
    residual = max(abs(residuals[0]), abs(residuals[1])) / thermal._root_residual_scale(integrator, model, nu, y)
    state = thermal._state(integrator, model, chemical_potential, nu, y, with_derivatives=False)
    if not (mp.isfinite(residual) and residual < mp.mpf("2e-24") and state["local_stable_C_positive"] and state["local_stable_fnn_positive"]):
        raise ValueError("unstable or unresolved local state")
    return {"nu": nu, "y": y, "state": state, "root_residual_relative": residual}


def _constrained_state(model: Any, temperature: mp.mpf, sigma_target: mp.mpf, seed: tuple[mp.mpf, mp.mpf, mp.mpf], *, nquad: int, cutoff: mp.mpf) -> dict[str, Any]:
    """Solve the same branch at fixed entropy per baryon s/n."""
    integrator = thermal._integrator_for(temperature, cutoff=cutoff, nquad=nquad)
    scale_one, scale_two = model.MN, model.MN**4

    def equations(nu: Any, y: Any, mu: Any) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
        nu, y, mu = _mp(nu), _mp(y), _mp(mu)
        r1, r2 = thermal._residuals(integrator, model, mu, nu, y)
        state = thermal._state(integrator, model, mu, nu, y, with_derivatives=False)
        return r1 / scale_one, r2 / scale_two, state["entropy_MeV3"] / state["n_MeV3"] - sigma_target

    try:
        nu, y, mu = mp.findroot(equations, seed, solver="mdnewton", tol=mp.eps**mp.mpf("0.70"), maxsteps=100, verify=True)
    except Exception as exc:
        raise ValueError(f"isentropic root failed: {type(exc).__name__}: {exc}") from exc
    residuals = thermal._residuals(integrator, model, mu, nu, y)
    residual = max(abs(residuals[0]), abs(residuals[1])) / thermal._root_residual_scale(integrator, model, nu, y)
    state = thermal._state(integrator, model, mu, nu, y, with_derivatives=False)
    entropy_residual = abs(state["entropy_MeV3"] / state["n_MeV3"] - sigma_target)
    if not (mp.isfinite(residual) and residual < mp.mpf("2e-24") and entropy_residual < mp.mpf("2e-24") and state["local_stable_C_positive"] and state["local_stable_fnn_positive"]):
        raise ValueError("unresolved isentropic local state")
    return {"nu": nu, "y": y, "mu": mu, "state": state, "root_residual_relative": residual, "entropy_residual": entropy_residual}


def _one_row(anchor: str, model_name: str, mu0: mp.mpf, *, dps: int, nquad: int, cutoff: mp.mpf, h: mp.mpf) -> dict[str, Any]:
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        model = nonlinear._build_models(anchor)[model_name]["model"]
        base = _state(model, TEMPERATURE_MEV, mu0, (mu0, _mp(anchor)), nquad=nquad, cutoff=cutoff)
        b = base["state"]
        sigma0 = b["entropy_MeV3"] / b["n_MeV3"]
        seed = (base["nu"], base["y"])

        t_plus = _state(model, TEMPERATURE_MEV + h, mu0, seed, nquad=nquad, cutoff=cutoff)["state"]
        t_minus = _state(model, TEMPERATURE_MEV - h, mu0, seed, nquad=nquad, cutoff=cutoff)["state"]
        m_plus = _state(model, TEMPERATURE_MEV, mu0 + h, seed, nquad=nquad, cutoff=cutoff)["state"]
        m_minus = _state(model, TEMPERATURE_MEV, mu0 - h, seed, nquad=nquad, cutoff=cutoff)["state"]

        def derivative(field: str, plus: Mapping[str, Any], minus: Mapping[str, Any]) -> mp.mpf:
            return (_mp(plus[field]) - _mp(minus[field])) / (2 * h)

        p_t, p_mu = derivative("pressure_MeV4", t_plus, t_minus), derivative("pressure_MeV4", m_plus, m_minus)
        e_t, e_mu = derivative("energy_MeV4", t_plus, t_minus), derivative("energy_MeV4", m_plus, m_minus)
        sig_t = (t_plus["entropy_MeV3"] / t_plus["n_MeV3"] - t_minus["entropy_MeV3"] / t_minus["n_MeV3"]) / (2 * h)
        sig_mu = (m_plus["entropy_MeV3"] / m_plus["n_MeV3"] - m_minus["entropy_MeV3"] / m_minus["n_MeV3"]) / (2 * h)
        if abs(sig_mu) < mp.mpf("1e-30"):
            raise ValueError("isentropic derivative singular")
        mu_prime = -sig_t / sig_mu
        p_path, e_path = p_t + p_mu * mu_prime, e_t + e_mu * mu_prime
        cs2 = p_path / e_path

        constrained_seed = (base["nu"], base["y"], mu0)
        iso_plus = _constrained_state(model, TEMPERATURE_MEV + h, sigma0, constrained_seed, nquad=nquad, cutoff=cutoff)
        iso_minus = _constrained_state(model, TEMPERATURE_MEV - h, sigma0, constrained_seed, nquad=nquad, cutoff=cutoff)
        direct_cs2 = (_mp(iso_plus["state"]["pressure_MeV4"]) - _mp(iso_minus["state"]["pressure_MeV4"])) / (_mp(iso_plus["state"]["energy_MeV4"]) - _mp(iso_minus["state"]["energy_MeV4"]))

        p_mu_error = _relative(p_mu, b["n_MeV3"])
        p_t_error = _relative(p_t, b["entropy_MeV3"])
        path_error = _relative(cs2, direct_cs2)
        if not (mp.mpf(0) < cs2 < mp.mpf(1) and mp.mpf(0) < direct_cs2 < mp.mpf(1)):
            raise ValueError("acausal or negative adiabatic sound speed")
        return {
            "anchor": anchor,
            "model": model_name,
            "T_MeV": _number(TEMPERATURE_MEV),
            "mu_MeV": _number(mu0),
            "settings": {"dps": dps, "nquad": nquad, "cutoff_MeV": _number(cutoff), "derivative_step_MeV": _number(h)},
            "base": {"n_fm3": _number(b["n_fm3"]), "entropy_per_baryon": _number(sigma0), "pressure_MeV_fm3": _number(b["pressure_MeV_fm3"]), "energy_MeV_fm3": _number(b["energy_MeV_fm3"]), "y": _number(base["y"]), "root_residual_relative": _number(base["root_residual_relative"])},
            "derivatives": {"P_T_MeV3": _number(p_t), "P_mu_MeV3": _number(p_mu), "epsilon_T_MeV3": _number(e_t), "epsilon_mu_MeV3": _number(e_mu), "sigma_T_per_MeV": _number(sig_t), "sigma_mu_per_MeV": _number(sig_mu), "dmu_dT_at_sigma": _number(mu_prime)},
            "adiabatic": {"cs2_implicit": _number(cs2), "cs_implicit": _number(mp.sqrt(cs2)), "cs2_direct_isentrope": _number(direct_cs2), "cs_direct_isentrope": _number(mp.sqrt(direct_cs2),), "relative_implicit_vs_direct": _number(path_error), "iso_plus_mu_MeV": _number(iso_plus["mu"]), "iso_minus_mu_MeV": _number(iso_minus["mu"])},
            "gates": {"positive_causal": True, "P_mu_equals_n_relative": _number(p_mu_error), "P_T_equals_s_relative": _number(p_t_error), "implicit_direct_relative": _number(path_error), "pass": bool(p_mu_error < mp.mpf("2e-4") and p_t_error < mp.mpf("2e-4") and path_error < mp.mpf("4e-4"))},
        }
    finally:
        mp.mp.dps = old_dps


def _build_result() -> dict[str, Any]:
    primary: list[dict[str, Any]] = []
    refined: list[dict[str, Any]] = []
    for anchor in ANCHORS:
        for model in MODELS:
            for mu in CHEMICAL_POTENTIALS:
                primary.append(_one_row(anchor, model, mu, dps=44, nquad=80, cutoff=mp.mpf("4500"), h=mp.mpf("0.5")))
                refined.append(_one_row(anchor, model, mu, dps=56, nquad=112, cutoff=mp.mpf("5200"), h=mp.mpf("0.25")))
    pairs = {(r["anchor"], r["model"], r["mu_MeV"]): r for r in refined}
    refinement: list[dict[str, Any]] = []
    for row in primary:
        ref = pairs[(row["anchor"], row["model"], row["mu_MeV"])]
        err = _relative(row["adiabatic"]["cs2_implicit"], ref["adiabatic"]["cs2_implicit"])
        refinement.append({"anchor": row["anchor"], "model": row["model"], "mu_MeV": row["mu_MeV"], "relative_cs2_difference": _number(err), "pass": bool(err < mp.mpf("3e-3"))})
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_PASS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "physical_inputs_fixed": True,
        "no_saved_results_as_inputs": True,
        "fit_used": False,
        "coverage": {"expected_rows": EXPECTED_ROWS, "primary_rows": len(primary), "refined_rows": len(refined)},
        "hydrodynamic_assumptions": ["local thermal equilibrium", "wavelength and period long compared with microscopic relaxation scales", "homogeneous symmetric branch", "no finite-volume boundary, viscosity, diffusion, or multi-fluid mode is calculated"],
        "formula": "cs2=(dP/dT|mu + dP/dmu|T*dmu/dT|sigma)/(depsilon/dT|mu + depsilon/dmu|T*dmu/dT|sigma), with sigma=s/n and dmu/dT|sigma=-(dsigma/dT|mu)/(dsigma/dmu|T)",
        "primary_rows": primary,
        "refinement": refinement,
        "interpretation_limits": ["This is the local-equilibrium adiabatic hydrodynamic speed, not an independently solved microscopic spectral pole.", "No damping, attenuation, finite nucleus/collision geometry, transport coefficient, real-frequency response, black-hole/cosmology result, or device power is calculated.", "All states are bounded locally stable branches; this does not establish global phase equilibrium."],
        "source_sha256": {"producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "thermal_observable_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(), "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest()},
    }
    controls = {
        "coverage": {"pass": len(primary) == EXPECTED_ROWS and len(refined) == EXPECTED_ROWS},
        "row_gates": {"pass": all(row["gates"]["pass"] for row in primary)},
        "refinement": {"max_relative_cs2_difference": _number(max(_mp(item["relative_cs2_difference"]) for item in refinement)), "pass": all(item["pass"] for item in refinement)},
        "causality": {"pass": all(mp.mpf("0") < _mp(row["adiabatic"]["cs2_implicit"]) < mp.mpf("1") for row in primary)},
    }
    result["controls"] = controls
    result["all_controls_pass"] = bool(all(item["pass"] for item in controls.values()))
    result["status"] = STATUS_PASS if result["all_controls_pass"] else STATUS_FAIL
    result["integrity_sha256"] = hashlib.sha256(_payload_bytes(result)).hexdigest()
    return result


def build_result() -> dict[str, Any]:
    return _build_result()


def validate_result(result: Any) -> bool:
    try:
        if not isinstance(result, Mapping) or result.get("schema_version") != SCHEMA_VERSION or result.get("status") != STATUS_PASS:
            return False
        if result.get("physical_inputs_fixed") is not True or result.get("no_saved_results_as_inputs") is not True or result.get("fit_used") is not False:
            return False
        if result.get("coverage") != {"expected_rows": EXPECTED_ROWS, "primary_rows": EXPECTED_ROWS, "refined_rows": EXPECTED_ROWS}:
            return False
        rows = result.get("primary_rows", [])
        if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
            return False
        actual = {(r.get("anchor"), r.get("model"), r.get("mu_MeV")) for r in rows}
        expected = {(a, m, _number(mu)) for a in ANCHORS for m in MODELS for mu in CHEMICAL_POTENTIALS}
        if actual != expected or any(r.get("gates", {}).get("pass") is not True for r in rows):
            return False
        for row in rows:
            cs2, direct = _mp(row["adiabatic"]["cs2_implicit"]), _mp(row["adiabatic"]["cs2_direct_isentrope"])
            if not (mp.mpf("0") < cs2 < mp.mpf("1") and mp.mpf("0") < direct < mp.mpf("1")):
                return False
            if _relative(cs2, direct) > mp.mpf("4e-4"):
                return False
            hbarc = _mp(thermal.HBARC_MEV_FM)
            n = _mp(row["base"]["n_fm3"]) * hbarc**3
            entropy = _mp(row["base"]["entropy_per_baryon"]) * n
            p_mu_error = _relative(row["derivatives"]["P_mu_MeV3"], n)
            p_t_error = _relative(row["derivatives"]["P_T_MeV3"], entropy)
            if p_mu_error > mp.mpf("2e-4") or p_t_error > mp.mpf("2e-4"):
                return False
            # Result fields are serialized to 30 significant digits, while the
            # gate is formed before serialization at higher precision.
            if _relative(row["gates"]["P_mu_equals_n_relative"], p_mu_error) > mp.mpf("1e-8") or _relative(row["gates"]["P_T_equals_s_relative"], p_t_error) > mp.mpf("1e-8"):
                return False
        refinement = result.get("refinement", [])
        if not isinstance(refinement, list) or len(refinement) != EXPECTED_ROWS:
            return False
        refined_points = {(item.get("anchor"), item.get("model"), item.get("mu_MeV")) for item in refinement}
        if refined_points != expected or any(item.get("pass") is not True or _mp(item.get("relative_cs2_difference")) >= mp.mpf("3e-3") for item in refinement):
            return False
        if any(item.get("pass") is not True for item in result.get("controls", {}).values()):
            return False
        expected_hashes = {"producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "thermal_observable_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(), "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest()}
        if result.get("source_sha256") != expected_hashes:
            return False
        payload = dict(result)
        digest = payload.pop("integrity_sha256", None)
        return isinstance(digest, str) and hashlib.sha256(_payload_bytes(payload)).hexdigest() == digest
    except (KeyError, TypeError, ValueError, ArithmeticError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "nvg_adiabatic_sound_bridge_results.json")
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
