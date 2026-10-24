#!/usr/bin/env python3
"""Surgical registry patch: register 24 missing maintained scripts/tests that block front-door.

Follows anchored-insertion pattern of previous patches.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "verification" / "registry.json"
ANCHOR = '    }\n  },\n  "front_door": {'

def render(key: str, value: dict) -> str:
    block = {key: value}
    text = json.dumps(block, indent=2, ensure_ascii=True)
    lines = text.splitlines()
    body = lines[1:-1]
    return "\n".join("  " + line if line.strip() else line for line in body)

NEW_ENTRIES: dict[str, dict] = {
    "verification/nvg_adiabatic_sound_bridge.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "No-fit adiabatic sound-speed bridge on the finite-temperature NVG EOS; hydrodynamic local-equilibrium limit only, no pole, damping or device power.",
        "producer": None,
        "consumers": ["verification/test_adiabatic_sound_bridge.py"],
        "inputs": ["verification/nvg_nonlinear_calibration_response.py", "verification/nvg_thermal_observable_bridge.py"],
        "outputs": [],
        "tests": ["verification/test_adiabatic_sound_bridge.py"],
        "deprecation": "Bounded homogeneous local-equilibrium hydrodynamic limit; not a finite-q dynamics, mode or empirical claim.",
        "quarantine": None,
    },
    "verification/nvg_covariant_fluid_bridge.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Local covariant perfect-fluid reconstruction of the thermal EOS; thermodynamic fluid sector only, no microscopic field action or gravity extrapolation.",
        "producer": None,
        "consumers": ["verification/test_covariant_fluid_bridge.py"],
        "inputs": ["verification/nvg_adiabatic_sound_bridge.py"],
        "outputs": [],
        "tests": ["verification/test_covariant_fluid_bridge.py"],
        "deprecation": "Local fluid reconstruction only; not a bounce, gravity or device model.",
        "quarantine": None,
    },
    "verification/nvg_grand_potential_landscape.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Bounded grand-potential landscape scan for the homogeneous thermal NVG EOS; on-shell pressure at fixed (T,mu) with gap-equation sign changes, not a global phase proof.",
        "producer": None,
        "consumers": ["verification/test_grand_potential_landscape.py"],
        "inputs": ["verification/nvg_nonlinear_calibration_response.py", "verification/nvg_thermal_observable_bridge.py"],
        "outputs": [],
        "tests": ["verification/test_grand_potential_landscape.py"],
        "deprecation": "Finite-domain mean-field landscape only; not a phase-equilibrium certificate.",
        "quarantine": None,
    },
    "verification/nvg_dynamic_response_identifiability.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Identifiability audit from the live static spatial response; records only relative predictions of a minimal one-inertia completion and unfixed coefficients.",
        "producer": None,
        "consumers": ["verification/test_dynamic_response_identifiability.py"],
        "inputs": ["verification/nvg_thermal_spatial_response.py"],
        "outputs": [],
        "tests": ["verification/test_dynamic_response_identifiability.py"],
        "deprecation": "Conditional derived probe; no real-frequency pole, damping or energy port.",
        "quarantine": None,
    },
    "verification/nvg_field_energy_transfer.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Live static density-scalar field-energy transfer coefficients from the maintained finite-q Dirac kernel and vector-constraint Schur reduction; redistribution of a supplied perturbation only.",
        "producer": None,
        "consumers": ["verification/test_field_energy_transfer.py"],
        "inputs": ["verification/nvg_nonlocal_response.py", "verification/nvg_spatial_identifiability_audit.py", "verification/source_complete_scaling_saturation_audit.py"],
        "outputs": [],
        "tests": ["verification/test_field_energy_transfer.py"],
        "deprecation": "Quadratic static redistribution only; no source, load, volume, damping, power or reservoir.",
        "quarantine": None,
    },
    "verification/nvg_pressure_observable_bridge.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Reduced-pressure four-point contrast for the W8/U16 pair removing constant, slope and curvature nuisance terms; prospective model-discrimination budget only.",
        "producer": None,
        "consumers": ["verification/test_pressure_observable_bridge.py"],
        "inputs": ["verification/nvg_nonlinear_calibration_response.py"],
        "outputs": [],
        "tests": ["verification/test_pressure_observable_bridge.py"],
        "deprecation": "Homogeneous cold-matter construct only; not an experimental or finite-temperature/nucleus observable.",
        "quarantine": None,
    },
    "verification/nvg_proton_acceptance_bridge.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Conditional local-CGF bridge from the live baryon response to HADES protons via Cooper-Frye acceptance and leading extensive one-charge saddle; bounded conditional approximation.",
        "producer": None,
        "consumers": ["verification/test_proton_acceptance_bridge.py"],
        "inputs": ["verification/nvg_thermal_observable_bridge.py"],
        "outputs": [],
        "tests": ["verification/test_proton_acceptance_bridge.py"],
        "deprecation": "Forward model only, not a calibration or fit; not an exact finite-B projection.",
        "quarantine": None,
    },
    "verification/nvg_dirac_hartree_calcium.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Self-consistent spherical no-sea Dirac-Hartree calcium with kinetic-balanced Galerkin radial Hamiltonian and live W8.93 fields; conservative norm, radial and field checks.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_finite_monopole.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Bounded finite-nucleus model probe with zero independent empirical weight.",
        "quarantine": None,
    },
    "verification/nvg_driven_density_load.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Source to finite-q NVG density mode to explicit reciprocal RLC load; open-system algebra scenario with maintained retarded response and dimensionless RLC.",
        "producer": None,
        "consumers": ["verification/test_driven_density_load.py"],
        "inputs": ["verification/nvg_retarded_response.py"],
        "outputs": [],
        "tests": ["verification/test_driven_density_load.py"],
        "deprecation": "Dimensionless scenario only; no SI watts or hardware recommendation.",
        "quarantine": None,
    },
    "verification/nvg_passive_response_probe.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Passive series-RLC consistency probe with frozen synthetic benchmark; input/validator/fit tool with endpoint-constrained and sweep-only fits.",
        "producer": None,
        "consumers": ["verification/test_passive_response_probe.py"],
        "inputs": [],
        "outputs": [],
        "tests": ["verification/test_passive_response_probe.py"],
        "deprecation": "Input validation tool only, not a physical NVG model or universal passive certificate.",
        "quarantine": None,
    },
    "verification/test_adiabatic_sound_bridge.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Live adiabatic sound-speed bridge validation, refinement and no-saved-input controls.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_adiabatic_sound_bridge.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for a zero-independent-weight hydrodynamic limit.",
        "quarantine": None,
    },
    "verification/test_covariant_fluid_bridge.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Covariant fluid reconstruction validation and no-saved-input controls.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_covariant_fluid_bridge.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for a zero-independent-weight local fluid completion.",
        "quarantine": None,
    },
    "verification/test_grand_potential_landscape.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Grand-potential landscape scan validation including pressure-maximum, refinement and mutation controls.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_grand_potential_landscape.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for a bounded mean-field landscape scan.",
        "quarantine": None,
    },
    "verification/test_dynamic_response_identifiability.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Static identifiability audit validation and mutation controls.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_dynamic_response_identifiability.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for a conditional static-to-dynamic probe.",
        "quarantine": None,
    },
    "verification/test_field_energy_transfer.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Field-energy transfer coefficient validation with regeneration and no-saved-input checks.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_field_energy_transfer.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for static redistribution coefficients only.",
        "quarantine": None,
    },
    "verification/test_pressure_observable_bridge.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Reduced-pressure bridge validation with recomputation and provenance checks.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_pressure_observable_bridge.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for a homogeneous cold-matter discriminator.",
        "quarantine": None,
    },
    "verification/test_proton_acceptance_bridge.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Proton-acceptance bridge validation with full-matrix and identity checks.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_proton_acceptance_bridge.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for a conditional CGF forward model.",
        "quarantine": None,
    },
    "verification/test_driven_density_load.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Finite-q to RLC load scenario validation with independent response checks.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_driven_density_load.py", "verification/nvg_retarded_response.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for a dimensionless open-system scenario.",
        "quarantine": None,
    },
    "verification/test_passive_response_probe.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Passive RLC probe validation with synthetic and mutation controls.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_passive_response_probe.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for an input-validation tool.",
        "quarantine": None,
    },
    "verification/test_thermal_observable_bridge.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Thermal observable bridge validation with fixed-grid and contract checks.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_thermal_observable_bridge.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for finite-temperature EOS construction.",
        "quarantine": None,
    },
    "verification/test_thermal_spatial_response.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Thermal spatial response validation with coverage and mutation checks.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_thermal_spatial_response.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for thermal spatial susceptibility.",
        "quarantine": None,
    },
    "verification/test_nonlinear_calibration_response.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Nonlinear calibration response validation with strict JSON and mutation controls.",
        "producer": None,
        "consumers": [],
        "inputs": ["verification/nvg_nonlinear_calibration_response.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for calibration-response construction.",
        "quarantine": None,
    },
    "generator_project/calc_nvg_closed_mixing_bridge.py": {
        "kind": "script",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Live closed-mixing, caloric and fixed-pressure bridge for the accepted EOS; sealed homogeneous-equilibrium endpoint study.",
        "producer": None,
        "consumers": ["generator_project/test_nvg_closed_mixing_bridge.py"],
        "inputs": ["verification/nvg_nonlinear_calibration_response.py", "verification/nvg_thermal_observable_bridge.py", "verification/nvg_thermal_composition_hessian.py"],
        "outputs": [],
        "tests": ["generator_project/test_nvg_closed_mixing_bridge.py"],
        "deprecation": "Bounded equilibrium endpoint study; no rate, power, device or global phase claim.",
        "quarantine": None,
    },
    "generator_project/test_nvg_closed_mixing_bridge.py": {
        "kind": "test",
        "role": "validation",
        "status": "maintained",
        "evidence_weight": 0.0,
        "description": "Closed-mixing bridge validation with numeric, mutation and provenance checks.",
        "producer": None,
        "consumers": [],
        "inputs": ["generator_project/calc_nvg_closed_mixing_bridge.py"],
        "outputs": [],
        "tests": [],
        "deprecation": "Regression controls for a homogeneous endpoint study.",
        "quarantine": None,
    },
}

def main() -> int:
    text = REGISTRY.read_text(encoding="utf-8")
    if text.count(ANCHOR) != 1:
        print(f"FAIL: anchor count = {text.count(ANCHOR)}")
        return 1
    registry = json.loads(text)
    entries = registry["entries"]
    for key in NEW_ENTRIES:
        if key in entries:
            print(f"FAIL: {key} already registered")
            return 1
    known = set(entries) | set(NEW_ENTRIES)
    for key, entry in NEW_ENTRIES.items():
        for field in ("inputs", "consumers", "tests"):
            for ref in entry.get(field) or []:
                if ref not in known:
                    print(f"FAIL: {key} references unregistered {ref}")
                    return 1
    blocks = ",\n".join(render(key, value) for key, value in NEW_ENTRIES.items())
    patched = text.replace(ANCHOR, "    },\n" + blocks + "\n  },\n  \"front_door\": {", 1)
    reparsed = json.loads(patched)
    if sum(1 for k in NEW_ENTRIES if k in reparsed["entries"]) != len(NEW_ENTRIES):
        print("FAIL: reparsed missing entries")
        return 1
    REGISTRY.write_text(patched, encoding="utf-8")
    print(f"OK: registered {len(NEW_ENTRIES)} entries (total {len(reparsed['entries'])})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
