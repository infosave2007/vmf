"""Audit the local covariant-perfect-fluid completion of NVG's thermal EOS.

This reconstructs only the thermodynamic fluid sector.  It neither identifies
a microscopic NVG field action nor extrapolates the EOS to gravity, a bounce,
or an energy-producing device.
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

import nvg_adiabatic_sound_bridge as sound  # noqa: E402


SCHEMA_VERSION = "nvg-covariant-fluid-bridge.v1"
STATUS_PASS = "PASS_LIVE_COVARIANT_FLUID_BRIDGE"
STATUS_FAIL = "FAIL_LIVE_COVARIANT_FLUID_BRIDGE"
EVIDENCE_WEIGHT = "local_covariant_perfect_fluid_reconstruction"
ANCHORS = sound.ANCHORS
MODELS = sound.MODELS
TEMPERATURE_MEV = sound.TEMPERATURE_MEV
CHEMICAL_POTENTIALS = sound.CHEMICAL_POTENTIALS
EXPECTED_ROWS = len(ANCHORS) * len(MODELS) * len(CHEMICAL_POTENTIALS)
THERMO_RELATIVE_TOL = mp.mpf("4e-4")
REFINEMENT_RELATIVE_TOL = mp.mpf("3e-3")


def _mp(value: Any) -> mp.mpf:
    return value if isinstance(value, mp.mpf) else mp.mpf(str(value))


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


def _one_row(anchor: str, model_name: str, mu0: mp.mpf, *, dps: int, nquad: int, cutoff: mp.mpf, h: mp.mpf) -> dict[str, Any]:
    """Check the first-law identities required by a fluid action at fixed s/n."""
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        model = sound.nonlinear._build_models(anchor)[model_name]["model"]
        base = sound._state(model, TEMPERATURE_MEV, mu0, (mu0, _mp(anchor)), nquad=nquad, cutoff=cutoff)
        state = base["state"]
        sigma = state["entropy_MeV3"] / state["n_MeV3"]
        seed = (base["nu"], base["y"], mu0)
        plus = sound._constrained_state(model, TEMPERATURE_MEV + h, sigma, seed, nquad=nquad, cutoff=cutoff)
        minus = sound._constrained_state(model, TEMPERATURE_MEV - h, sigma, seed, nquad=nquad, cutoff=cutoff)

        n, eps, pressure = state["n_MeV3"], state["energy_MeV4"], state["pressure_MeV4"]
        n_plus, n_minus = plus["state"]["n_MeV3"], minus["state"]["n_MeV3"]
        if abs(n_plus - n_minus) < mp.mpf("1e-30"):
            raise ValueError("isentropic density derivative is singular")
        epsilon_n_sigma = (plus["state"]["energy_MeV4"] - minus["state"]["energy_MeV4"]) / (n_plus - n_minus)
        pressure_n_sigma = (plus["state"]["pressure_MeV4"] - minus["state"]["pressure_MeV4"]) / (n_plus - n_minus)
        pressure_from_action = n * epsilon_n_sigma - eps
        enthalpy_per_baryon = (eps + pressure) / n
        grand_canonical_enthalpy = mu0 + TEMPERATURE_MEV * sigma
        cs2_from_action = pressure_n_sigma / epsilon_n_sigma
        cs2_from_isentrope = (plus["state"]["pressure_MeV4"] - minus["state"]["pressure_MeV4"]) / (plus["state"]["energy_MeV4"] - minus["state"]["energy_MeV4"])
        pressure_error = _relative(pressure_from_action, pressure)
        enthalpy_error = _relative(epsilon_n_sigma, enthalpy_per_baryon)
        euler_error = _relative(enthalpy_per_baryon, grand_canonical_enthalpy)
        speed_error = _relative(cs2_from_action, cs2_from_isentrope)
        stable = bool(
            state["local_stable_C_positive"]
            and state["local_stable_fnn_positive"]
            and plus["state"]["local_stable_C_positive"]
            and plus["state"]["local_stable_fnn_positive"]
            and minus["state"]["local_stable_C_positive"]
            and minus["state"]["local_stable_fnn_positive"]
        )
        gate = bool(
            stable
            and mp.mpf("0") < cs2_from_action < mp.mpf("1")
            and pressure_error < THERMO_RELATIVE_TOL
            and enthalpy_error < THERMO_RELATIVE_TOL
            and euler_error < THERMO_RELATIVE_TOL
            and speed_error < mp.mpf("1e-20")
        )
        return {
            "anchor": anchor,
            "model": model_name,
            "T_MeV": _number(TEMPERATURE_MEV),
            "mu_MeV": _number(mu0),
            "settings": {"dps": dps, "nquad": nquad, "cutoff_MeV": _number(cutoff), "isentropic_temperature_step_MeV": _number(h)},
            "base": {
                "n_MeV3": _number(n),
                "epsilon_MeV4": _number(eps),
                "pressure_MeV4": _number(pressure),
                "sigma_entropy_per_baryon": _number(sigma),
                "root_residual_relative": _number(base["root_residual_relative"]),
            },
            "fluid_action_reconstruction": {
                "epsilon_n_at_fixed_sigma_MeV": _number(epsilon_n_sigma),
                "pressure_n_at_fixed_sigma_MeV": _number(pressure_n_sigma),
                "pressure_from_n_epsilon_n_minus_epsilon_MeV4": _number(pressure_from_action),
                "enthalpy_per_baryon_MeV": _number(enthalpy_per_baryon),
                "mu_plus_T_sigma_MeV": _number(grand_canonical_enthalpy),
                "cs2_from_action": _number(cs2_from_action),
                "cs2_from_direct_isentrope": _number(cs2_from_isentrope),
            },
            "isentropic_neighbors": {
                "plus_mu_MeV": _number(plus["mu"]),
                "minus_mu_MeV": _number(minus["mu"]),
                "plus_root_residual_relative": _number(plus["root_residual_relative"]),
                "minus_root_residual_relative": _number(minus["root_residual_relative"]),
                "plus_entropy_residual": _number(plus["entropy_residual"]),
                "minus_entropy_residual": _number(minus["entropy_residual"]),
            },
            "gates": {
                "local_stable": stable,
                "positive_causal_sound": bool(mp.mpf("0") < cs2_from_action < mp.mpf("1")),
                "pressure_identity_relative": _number(pressure_error),
                "enthalpy_gradient_relative": _number(enthalpy_error),
                "grand_canonical_euler_relative": _number(euler_error),
                "sound_identity_relative": _number(speed_error),
                "pass": gate,
            },
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
    refined_by_key = {(r["anchor"], r["model"], r["mu_MeV"]): r for r in refined}
    refinement = []
    for row in primary:
        other = refined_by_key[(row["anchor"], row["model"], row["mu_MeV"])]
        action = row["fluid_action_reconstruction"]
        other_action = other["fluid_action_reconstruction"]
        pressure_error = _relative(action["pressure_from_n_epsilon_n_minus_epsilon_MeV4"], other_action["pressure_from_n_epsilon_n_minus_epsilon_MeV4"])
        sound_error = _relative(action["cs2_from_action"], other_action["cs2_from_action"])
        maximum = max(pressure_error, sound_error)
        refinement.append({
            "anchor": row["anchor"],
            "model": row["model"],
            "mu_MeV": row["mu_MeV"],
            "relative_pressure_reconstruction_difference": _number(pressure_error),
            "relative_cs2_difference": _number(sound_error),
            "pass": bool(maximum < REFINEMENT_RELATIVE_TOL),
        })
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_PASS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "physical_inputs_fixed": True,
        "no_saved_results_as_inputs": True,
        "fit_used": False,
        "coverage": {"expected_rows": EXPECTED_ROWS, "primary_rows": len(primary), "refined_rows": len(refined)},
        "effective_action": "S_m=integral sqrt(-g)[-epsilon(n,sigma)+J^mu(partial_mu phi+alpha partial_mu beta+theta partial_mu sigma)] d4x, with n=sqrt(-J_mu J^mu)/sqrt(-g); standard Clebsch/entropy constraints are shown schematically.",
        "derived_identities": ["P=n*(partial epsilon/partial n)|sigma-epsilon", "(partial epsilon/partial n)|sigma=(epsilon+P)/n=mu+T*sigma", "cs2=(dP/dn)|sigma/(d epsilon/dn)|sigma=(dP/d epsilon)|sigma"],
        "interpretation_limits": ["This proves only a local perfect-fluid effective representation wherever the checked thermodynamic map is regular; it does not reconstruct a unique microscopic NVG action.", "The calculation contains no scalar/vector kinetic sector beyond the effective fluid variables, no viscosity/diffusion, no finite-frequency pole, and no global phase construction.", "Coupling this effective fluid to gravity is a subsequent model choice, not a derived black-hole, bounce, dark-energy, or cyclic-universe result."],
        "primary_rows": primary,
        "refinement": refinement,
        "source_sha256": {
            "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "adiabatic_sound_bridge": hashlib.sha256(Path(sound.__file__).resolve().read_bytes()).hexdigest(),
            "thermal_observable_bridge": hashlib.sha256(Path(sound.thermal.__file__).resolve().read_bytes()).hexdigest(),
            "nonlinear_calibration": hashlib.sha256(Path(sound.nonlinear.__file__).resolve().read_bytes()).hexdigest(),
        },
    }
    controls = {
        "coverage": {"pass": len(primary) == EXPECTED_ROWS and len(refined) == EXPECTED_ROWS},
        "row_gates": {"pass": all(row["gates"]["pass"] for row in primary)},
        "refinement": {
            "max_relative_pressure_reconstruction_difference": _number(max(_mp(r["relative_pressure_reconstruction_difference"]) for r in refinement)),
            "max_relative_cs2_difference": _number(max(_mp(r["relative_cs2_difference"]) for r in refinement)),
            "pass": all(r["pass"] for r in refinement),
        },
    }
    result["controls"] = controls
    result["all_controls_pass"] = bool(all(control["pass"] for control in controls.values()))
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
        rows = result.get("primary_rows")
        if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
            return False
        expected = {(anchor, model, _number(mu)) for anchor in ANCHORS for model in MODELS for mu in CHEMICAL_POTENTIALS}
        if {(row.get("anchor"), row.get("model"), row.get("mu_MeV")) for row in rows} != expected:
            return False
        for row in rows:
            if row.get("gates", {}).get("pass") is not True:
                return False
            action = row["fluid_action_reconstruction"]
            eps_n = _mp(action["epsilon_n_at_fixed_sigma_MeV"])
            pressure_from_action = _mp(action["pressure_from_n_epsilon_n_minus_epsilon_MeV4"])
            pressure = _mp(row["base"]["pressure_MeV4"])
            cs2_action = _mp(action["cs2_from_action"])
            cs2_direct = _mp(action["cs2_from_direct_isentrope"])
            enthalpy = _mp(action["enthalpy_per_baryon_MeV"])
            euler = _mp(action["mu_plus_T_sigma_MeV"])
            if not (eps_n > 0 and mp.mpf("0") < cs2_action < mp.mpf("1")):
                return False
            if _relative(pressure_from_action, pressure) > THERMO_RELATIVE_TOL:
                return False
            if _relative(eps_n, enthalpy) > THERMO_RELATIVE_TOL or _relative(enthalpy, euler) > THERMO_RELATIVE_TOL:
                return False
            if _relative(cs2_action, cs2_direct) > mp.mpf("1e-20"):
                return False
        refinement = result.get("refinement")
        if not isinstance(refinement, list) or len(refinement) != EXPECTED_ROWS:
            return False
        if {(row.get("anchor"), row.get("model"), row.get("mu_MeV")) for row in refinement} != expected:
            return False
        if any(row.get("pass") is not True or max(_mp(row["relative_pressure_reconstruction_difference"]), _mp(row["relative_cs2_difference"])) >= REFINEMENT_RELATIVE_TOL for row in refinement):
            return False
        if any(control.get("pass") is not True for control in result.get("controls", {}).values()):
            return False
        hashes = {
            "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "adiabatic_sound_bridge": hashlib.sha256(Path(sound.__file__).resolve().read_bytes()).hexdigest(),
            "thermal_observable_bridge": hashlib.sha256(Path(sound.thermal.__file__).resolve().read_bytes()).hexdigest(),
            "nonlinear_calibration": hashlib.sha256(Path(sound.nonlinear.__file__).resolve().read_bytes()).hexdigest(),
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
    parser.add_argument("--output", type=Path, default=HERE / "nvg_covariant_fluid_bridge_results.json")
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
