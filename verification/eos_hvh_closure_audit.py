#!/usr/bin/env python3
"""Independent thermodynamic/HVH closure audit for the maintained NVG EOS.

The maintained producer :mod:`nvg_eos_beta_saturated_vector` returns a
beta-equilibrated composition and an energy density, then obtains pressure
with ``n**2 * gradient(epsilon/n, n)``.  It does not expose a stress tensor,
chemical potentials, or rearrangement terms.  This audit reconstructs the
zero-temperature chemical-potential pressure from the same energy functional
and compares it with convergence-controlled density derivatives.  A separate
constant-mass, zero-interaction npe-mu gas is used as an ideal reference.

No parameters are fitted here.  The interacting point is the maintained
saturated-vector screening point used by downstream EOS consumers.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import brentq

import nvg_eos_beta_saturated_vector as maintained


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "Lunacy" / "runs" / "deep-physics-audit"
RESULT_PATH = Path(__file__).resolve().with_name("eos_hvh_closure_audit_results.json")
# The original Phase-1 report is immutable.  Repair regeneration writes a
# verification-local report; the repair Control Block is frozen separately.
REPORT_PATH = Path(__file__).resolve().with_name("eos_hvh_closure_audit_report.md")
ORIGINAL_REPORT_PATH = RUN_ROOT / "REPORT-hvh-closure.md"

N0 = float(maintained.n_0)
M_N = float(maintained.M_N)
HBC = float(maintained.hbar_c)
ME = float(maintained.m_e)
MMU = float(maintained.m_mu)

# This is the first-principles-independent baseline selected by the maintained
# producer's own default screening table, not a refit by this audit.
INTERACTING_INPUTS = {
    "k1": 0.25,
    "k2": 0.80,
    "Cs": 900.0,
    "Crho": 600.0,
    "alpha_v": 4.0,
    "nu_v": 2.0,
}
DERIVATIVE_STEPS = (0.02, 0.01, 0.005)


def fermi_energy_density(density: float, mass: float) -> float:
    """Spin-two zero-temperature Fermi-gas energy density in MeV/fm^3."""

    density = float(density)
    mass = float(mass)
    if density <= 0.0:
        return 0.0
    if not (np.isfinite(density) and np.isfinite(mass) and mass > 0.0):
        raise ValueError("density must be finite and mass must be positive")
    kf = (3.0 * np.pi**2 * density) ** (1.0 / 3.0) * HBC
    ef = math.sqrt(kf * kf + mass * mass)
    log_term = math.log((kf + ef) / mass)
    numerator = kf * ef * (2.0 * kf * kf + mass * mass) - mass**4 * log_term
    return float(numerator / (8.0 * np.pi**2 * HBC**3))


def fermi_chemical_potential(density: float, mass: float) -> float:
    """Single-particle Fermi energy for a spin-two species."""

    density = float(density)
    if density <= 0.0:
        return float(mass)
    kf = (3.0 * np.pi**2 * density) ** (1.0 / 3.0) * HBC
    return float(math.sqrt(kf * kf + float(mass) ** 2))


def lepton_density(mu: float, mass: float) -> float:
    """Spin-two lepton density, with the zero-density threshold explicit."""

    mu = float(mu)
    mass = float(mass)
    if mu <= mass:
        return 0.0
    return float((mu * mu - mass * mass) ** 1.5 / (3.0 * np.pi**2 * HBC**3))


def five_point_derivative(function: Callable[[float], float], x: float, h: float) -> float:
    """Fourth-order symmetric derivative, requiring a smooth local branch."""

    x = float(x)
    h = float(h)
    if not (np.isfinite(x) and np.isfinite(h) and h > 0.0 and x - 2.0 * h > 0.0):
        raise ValueError("five-point derivative requires a positive interior stencil")
    values = [float(function(x + offset * h)) for offset in (-2.0, -1.0, 1.0, 2.0)]
    if not np.isfinite(values).all():
        raise ValueError("non-finite value in derivative stencil")
    return float((values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (12.0 * h))


def relative_difference(lhs: float, rhs: float) -> float:
    scale = max(abs(float(lhs)), abs(float(rhs)), 1.0e-8)
    return float(abs(float(lhs) - float(rhs)) / scale)


def ideal_beta_state(density: float) -> dict[str, float]:
    """Solve charge-neutral beta-equilibrated npe-mu matter with no interactions."""

    density = float(density)
    if density <= 0.0 or not np.isfinite(density):
        raise ValueError("ideal density must be positive and finite")

    def residual(y_p: float) -> float:
        n_p = float(y_p) * density
        n_n = density - n_p
        mu_n = fermi_chemical_potential(n_n, M_N)
        mu_p = fermi_chemical_potential(n_p, M_N)
        mu_e = max(mu_n - mu_p, ME)
        return n_p - lepton_density(mu_e, ME) - lepton_density(mu_e, MMU)

    y_lo, y_hi = 1.0e-10, 0.5 - 1.0e-10
    f_lo, f_hi = residual(y_lo), residual(y_hi)
    if f_lo * f_hi > 0.0:
        raise RuntimeError("ideal beta-equilibrium charge bracket is empty")
    y_p = float(brentq(residual, y_lo, y_hi, xtol=1.0e-12, rtol=1.0e-12))
    n_p = y_p * density
    n_n = density - n_p
    mu_n = fermi_chemical_potential(n_n, M_N)
    mu_p = fermi_chemical_potential(n_p, M_N)
    mu_e = max(mu_n - mu_p, ME)
    n_e = lepton_density(mu_e, ME)
    n_mu = lepton_density(mu_e, MMU)
    epsilon = (
        fermi_energy_density(n_n, M_N)
        + fermi_energy_density(n_p, M_N)
        + fermi_energy_density(n_e, ME)
        + fermi_energy_density(n_mu, MMU)
    )
    pressure_chem = mu_n * n_n + mu_p * n_p + mu_e * n_e + mu_e * n_mu - epsilon
    return {
        "n": density,
        "n_n": n_n,
        "n_p": n_p,
        "n_e": n_e,
        "n_mu": n_mu,
        "y_p": y_p,
        "mu_n": mu_n,
        "mu_p": mu_p,
        "mu_e": mu_e,
        "epsilon": epsilon,
        "pressure_chem": pressure_chem,
        "charge_residual": n_p - n_e - n_mu,
        "beta_residual": mu_n - mu_p - mu_e,
    }


def ideal_energy(density: float) -> float:
    return float(ideal_beta_state(float(density))["epsilon"])


def ideal_reference() -> dict[str, object]:
    densities = np.geomspace(0.05 * N0, 4.0 * N0, 14)
    rows: list[dict[str, float]] = []
    derivative_by_step: dict[str, list[float]] = {str(step): [] for step in DERIVATIVE_STEPS}
    for density in densities:
        state = ideal_beta_state(float(density))
        row = dict(state)
        for step in DERIVATIVE_STEPS:
            h = float(step) * density
            derivative_by_step[str(step)].append(
                float(density * density * five_point_derivative(lambda value: ideal_energy(value) / value, density, h))
            )
        rows.append(row)
    finest = np.asarray(derivative_by_step[str(DERIVATIVE_STEPS[-1])], dtype=float)
    chem = np.asarray([row["pressure_chem"] for row in rows], dtype=float)
    closure = max(relative_difference(a, b) for a, b in zip(finest, chem))
    convergence = max(
        relative_difference(derivative_by_step[str(DERIVATIVE_STEPS[-1])][i], derivative_by_step[str(DERIVATIVE_STEPS[-2])][i])
        for i in range(len(rows))
    )
    analytic = analytic_powerlaw_benchmark()
    return {
        "name": "zero-interaction cold beta-equilibrated n-p-e-mu reference",
        "rows": rows,
        "derivative_pressure_mev_fm3": derivative_by_step,
        "controls": {
            "closure_max_relative": float(closure),
            "derivative_convergence_max_relative": float(convergence),
            "max_charge_residual_fm3": float(max(abs(row["charge_residual"]) for row in rows)),
            "max_beta_residual_mev": float(max(abs(row["beta_residual"]) for row in rows)),
            "analytic_powerlaw_max_relative": float(analytic["max_relative_error"]),
        },
        "analytic_powerlaw": analytic,
    }


def analytic_powerlaw_benchmark() -> dict[str, float | str]:
    """Reference epsilon=m*n+K*n^gamma, whose pressure is (gamma-1)K*n^gamma."""

    mass, coefficient, gamma = 700.0, 19.0, 4.0 / 3.0
    densities = np.geomspace(0.03, 1.7, 12)

    def energy_per_particle(density: float) -> float:
        return mass + coefficient * density ** (gamma - 1.0)

    errors = []
    for density in densities:
        derivative = density * density * five_point_derivative(
            energy_per_particle, density, 0.005 * density
        )
        exact = (gamma - 1.0) * coefficient * density**gamma
        errors.append(relative_difference(derivative, exact))
    return {
        "equation": "epsilon=m*n+K*n^gamma; p=(gamma-1)K*n^gamma",
        "max_relative_error": float(max(errors)),
        "status": "PASS" if max(errors) < 1.0e-8 else "FAIL",
    }


def vector_factor_derivative(density: float, alpha_v: float, nu_v: float) -> float:
    x = float(density) / N0
    if x <= 1.0:
        return 0.0
    denominator = 1.0 + alpha_v * (x - 1.0) ** nu_v
    return float(-alpha_v * nu_v * (x - 1.0) ** (nu_v - 1.0) / (N0 * denominator**2))


def mass_reference_derivative(density: float, k1: float, k2: float) -> tuple[float, str]:
    """Piecewise analytic derivative of maintained M_base, exposing clipping."""

    density = float(density)
    x = max(density / N0, 0.0)
    current_correction = maintained.sigma_piN * density * HBC**3 / (maintained.f_pi**2 * maintained.m_pi**2)
    if current_correction < 1.0:
        current_derivative = -maintained.M_current_0 * maintained.sigma_piN * HBC**3 / (
            maintained.f_pi**2 * maintained.m_pi**2
        )
        branch = "unclipped"
    else:
        current_derivative = 0.0
        branch = "current-mass-clipped"
    omega = maintained.M_Omega(density, k1, k2)
    omega_derivative = -omega * k1 / (N0 * (1.0 + k2 * x))
    return float(current_derivative + omega_derivative), branch


def _interacting_parameters() -> tuple[dict[str, float], float]:
    values = {key: float(value) for key, value in INTERACTING_INPUTS.items()}
    c_omega, state_n0 = maintained.calibrate_c_omega0(
        values["k1"], values["k2"], values["Cs"], values["Crho"]
    )
    if c_omega is None or state_n0 is None:
        raise RuntimeError("maintained EOS calibration returned no vector coupling")
    return values, float(c_omega)


def interacting_state(density: float, parameters: dict[str, float], c_omega: float) -> dict[str, float | str]:
    density = float(density)
    state = maintained.beta_equilibrium_state(
        density,
        parameters["k1"],
        parameters["k2"],
        parameters["Cs"],
        parameters["Crho"],
    )
    if state is None:
        raise RuntimeError(f"maintained beta-equilibrium state missing at n={density:g}")

    n_n, n_p = float(state["n_n"]), float(state["n_p"])
    n_e = maintained.lepton_density_from_mu(float(state["mu_e"]), ME)
    n_mu = maintained.lepton_density_from_mu(float(state["mu_e"]), MMU)
    m_dirac = float(state["m_dirac"])
    m_ref = float(state["m_ref"])
    n_scalar = float(
        maintained.fermion_scalar_density(n_n, m_dirac)
        + maintained.fermion_scalar_density(n_p, m_dirac)
    )
    g = float(maintained.vector_factor(density, parameters["alpha_v"], parameters["nu_v"]))
    dg_dn = vector_factor_derivative(density, parameters["alpha_v"], parameters["nu_v"])
    dmref_dn, mass_branch = mass_reference_derivative(density, parameters["k1"], parameters["k2"])
    delta = n_n - n_p
    efn = fermi_chemical_potential(n_n, m_dirac)
    efp = fermi_chemical_potential(n_p, m_dirac)
    vector_mean = c_omega * g * density
    vector_rearrangement = 0.5 * c_omega * density**2 * dg_dn
    scalar_rearrangement = n_scalar * dmref_dn
    common_full = vector_mean + vector_rearrangement + scalar_rearrangement
    common_naive = vector_mean
    mu_n = efn + parameters["Crho"] * delta + common_full
    mu_p = efp - parameters["Crho"] * delta + common_full
    mu_n_naive = efn + parameters["Crho"] * delta + common_naive
    mu_p_naive = efp - parameters["Crho"] * delta + common_naive
    epsilon = float(state["eps_no_vec"] + maintained.vector_energy_density(
        density, c_omega, parameters["alpha_v"], parameters["nu_v"]
    ))
    pressure_chem = mu_n * n_n + mu_p * n_p + float(state["mu_e"]) * (n_e + n_mu) - epsilon
    pressure_naive = mu_n_naive * n_n + mu_p_naive * n_p + float(state["mu_e"]) * (n_e + n_mu) - epsilon
    gap_residual = m_dirac - m_ref + parameters["Cs"] * n_scalar
    charge_residual = n_p - n_e - n_mu
    beta_residual = mu_n - mu_p - float(state["mu_e"])
    return {
        "n": density,
        "n_n": n_n,
        "n_p": n_p,
        "n_e": n_e,
        "n_mu": n_mu,
        "y_p": float(state["y_p"]),
        "m_ref": m_ref,
        "m_dirac": m_dirac,
        "epsilon": epsilon,
        "mu_n": float(mu_n),
        "mu_p": float(mu_p),
        "mu_e": float(state["mu_e"]),
        "pressure_chem": float(pressure_chem),
        "pressure_chem_without_rearrangement": float(pressure_naive),
        "vector_factor": g,
        "vector_mean_mev": float(vector_mean),
        "vector_rearrangement_mev": float(vector_rearrangement),
        "scalar_rearrangement_mev": float(scalar_rearrangement),
        "mass_reference_derivative_mev_fm3": float(dmref_dn),
        "mass_reference_branch": mass_branch,
        "stationarity_residual_mev": float(gap_residual),
        "charge_residual_fm3": float(charge_residual),
        "beta_residual_mev": float(beta_residual),
    }


def interacting_energy(density: float, parameters: dict[str, float], c_omega: float) -> float:
    return float(interacting_state(float(density), parameters, c_omega)["epsilon"])


def interacting_reference() -> dict[str, object]:
    parameters, c_omega = _interacting_parameters()
    densities = np.geomspace(0.05 * N0, 6.0 * N0, 18)
    rows: list[dict[str, float | str]] = []
    derivative_by_step: dict[str, list[float]] = {str(step): [] for step in DERIVATIVE_STEPS}
    for density in densities:
        density = float(density)
        state = interacting_state(density, parameters, c_omega)
        row = dict(state)
        for step in DERIVATIVE_STEPS:
            h = float(step) * density
            derivative_by_step[str(step)].append(
                float(density * density * five_point_derivative(
                    lambda value: interacting_energy(value, parameters, c_omega) / value,
                    density,
                    h,
                ))
            )
        rows.append(row)

    finest = np.asarray(derivative_by_step[str(DERIVATIVE_STEPS[-1])], dtype=float)
    medium = np.asarray(derivative_by_step[str(DERIVATIVE_STEPS[-2])], dtype=float)
    chem = np.asarray([row["pressure_chem"] for row in rows], dtype=float)
    naive = np.asarray([row["pressure_chem_without_rearrangement"] for row in rows], dtype=float)
    full_closure = [relative_difference(a, b) for a, b in zip(finest, chem)]
    naive_closure = [relative_difference(a, b) for a, b in zip(finest, naive)]
    convergence = [relative_difference(a, b) for a, b in zip(finest, medium)]
    smooth_rows = [row for row in rows if row["mass_reference_branch"] == "unclipped"]
    smooth_indices = [i for i, row in enumerate(rows) if row["mass_reference_branch"] == "unclipped"]
    smooth_full = [full_closure[i] for i in smooth_indices]
    smooth_naive = [naive_closure[i] for i in smooth_indices]
    controls = {
        "closure_max_relative_all": float(max(full_closure)),
        "closure_max_relative_smooth_mass_branch": float(max(smooth_full)),
        "naive_closure_max_relative_all": float(max(naive_closure)),
        "naive_closure_max_relative_smooth_mass_branch": float(max(smooth_naive)),
        "derivative_convergence_max_relative": float(max(convergence)),
        "max_stationarity_residual_mev": float(max(abs(float(row["stationarity_residual_mev"])) for row in rows)),
        "max_charge_residual_fm3": float(max(abs(float(row["charge_residual_fm3"])) for row in rows)),
        "max_beta_residual_mev": float(max(abs(float(row["beta_residual_mev"])) for row in rows)),
        "max_vector_rearrangement_mev": float(max(abs(float(row["vector_rearrangement_mev"])) for row in rows)),
        "max_scalar_rearrangement_mev": float(max(abs(float(row["scalar_rearrangement_mev"])) for row in rows)),
        "smooth_sample_count": int(len(smooth_rows)),
        "sample_count": int(len(rows)),
    }
    controls["closure_gate"] = bool(
        controls["closure_max_relative_smooth_mass_branch"] < 5.0e-3
        and controls["derivative_convergence_max_relative"] < 5.0e-3
        and controls["max_stationarity_residual_mev"] < 1.0e-5
        # The maintained composition root uses y_p xtol=1e-8; retain a
        # density-scale tolerance that exposes failures without pretending
        # to have tighter composition data than the producer supplies.
        and controls["max_charge_residual_fm3"] < 2.0e-8
        and controls["max_beta_residual_mev"] < 1.0e-5
    )
    return {
        "name": "maintained saturated-vector beta-equilibrated NVG EOS",
        "parameters": {**parameters, "Cw0": float(c_omega)},
        "rows": rows,
        "derivative_pressure_mev_fm3": derivative_by_step,
        "controls": controls,
    }


def _returned_state_keys(source_text: str, function_name: str) -> set[str]:
    """Read returned dictionary keys from one maintained function, if present."""

    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != function_name:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Return) or not isinstance(child.value, ast.Dict):
                continue
            keys = {key.value for key in child.value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
            if keys:
                return keys
    return set()


def maintained_inventory() -> dict[str, object]:
    source = Path(maintained.__file__).resolve()
    source_text = source.read_text(encoding="utf-8")
    state_keys = _returned_state_keys(source_text, "beta_equilibrium_state")
    function_names = {
        node.name.lower()
        for node in ast.walk(ast.parse(source_text))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    has_chemical_api = bool({"mu_n", "mu_p", "mu_b"} & {key.lower() for key in state_keys})
    has_stress_api = any("stress" in name or "pressure_from_mu" in name for name in function_names)
    chemical_path = "AVAILABLE" if has_chemical_api else "ABSENT from returned state/API (mu_e and starred values are local variables)"
    stress_path = "AVAILABLE" if has_stress_api else "ABSENT"
    required_path_available = bool(has_chemical_api and has_stress_api)
    consumers = []
    for candidate in sorted((ROOT / "verification").glob("*.py")):
        if candidate.resolve() in {source, Path(__file__).resolve()}:
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        if "nvg_eos_beta_saturated_vector" in text:
            consumers.append(str(candidate.relative_to(ROOT)))
    return {
        "producer": str(source.relative_to(ROOT)),
        "producer_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "consumers_found_by_source_scan": consumers,
        "consumer_count": len(consumers),
        "functions": {
            "beta_equilibrium_state_signature": str(inspect.signature(maintained.beta_equilibrium_state)),
            "build_eos_signature": str(inspect.signature(maintained.build_eos)),
            "vector_energy_density_signature": str(inspect.signature(maintained.vector_energy_density)),
        },
        "energy_path": "beta_equilibrium_state.eps_no_vec + vector_energy_density(n,Cw0,alpha_v,nu_v)",
        "pressure_path": "build_eos: n**2 * np.gradient((epsilon/n), n)",
        "chemical_potential_path": chemical_path,
        "stress_tensor_path": stress_path,
        "rearrangement_terms": "AVAILABLE" if any("rearrang" in name for name in function_names) else "ABSENT from maintained producer; reconstructed independently here",
        "stationarity_residual": "AVAILABLE" if "stationarity_residual" in {key.lower() for key in state_keys} else "ABSENT from returned state; gap residual reconstructed independently here",
        "returned_state_keys": sorted(state_keys),
        "required_path_available": required_path_available,
    }


def _is_declared_available(value: object) -> bool:
    return isinstance(value, str) and "AVAILABLE" in value.upper() and "ABSENT" not in value.upper()


def _inventory_contract(inventory: dict[str, object]) -> tuple[bool, bool]:
    """Return (inventory_is_well_formed, required_path_is_available)."""

    required = ("chemical_potential_path", "stress_tensor_path", "required_path_available")
    if any(key not in inventory for key in required):
        return False, False
    declared = inventory["required_path_available"]
    if not isinstance(declared, bool):
        return False, False
    chemical = _is_declared_available(inventory["chemical_potential_path"])
    stress = _is_declared_available(inventory["stress_tensor_path"])
    derived = bool(chemical and stress)
    # Contradictions between source-derived path descriptions and the summary
    # flag are a fail-closed inventory failure, not a reason to trust either.
    if declared != derived:
        return False, False
    return True, derived


def _finite_below(value: object, limit: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number < limit)


def derive_terminal_controls(
    inventory: dict[str, object], ideal: dict[str, object], interacting: dict[str, object]
) -> tuple[str, dict[str, bool | float | str]]:
    """Derive status and gates from runtime inventory and nested closure controls.

    This function is intentionally pure so semantic tests can mutate each
    input and prove that future API/result drift cannot leave a stale PASS.
    """

    inventory_valid, maintained_path = _inventory_contract(inventory)
    ideal_controls = ideal.get("controls") if isinstance(ideal, dict) else None
    analytic = ideal.get("analytic_powerlaw") if isinstance(ideal, dict) else None
    interacting_controls = interacting.get("controls") if isinstance(interacting, dict) else None
    analytic_pass = bool(isinstance(analytic, dict) and analytic.get("status") == "PASS")
    ideal_pass = bool(
        isinstance(ideal_controls, dict)
        and _finite_below(ideal_controls.get("closure_max_relative"), 5.0e-3)
        and _finite_below(ideal_controls.get("derivative_convergence_max_relative"), 5.0e-3)
    )
    interacting_pass = bool(
        isinstance(interacting_controls, dict)
        and interacting_controls.get("closure_gate") is True
        and _finite_below(interacting_controls.get("closure_max_relative_smooth_mass_branch"), 5.0e-3)
        and _finite_below(interacting_controls.get("derivative_convergence_max_relative"), 5.0e-3)
    )
    if not inventory_valid:
        status = "FAIL_CLOSED_INVALID_MAINTAINED_INVENTORY"
    elif not maintained_path:
        status = "FAIL_CLOSED_MAINTAINED_STRESS_CHEMICAL_PATH_ABSENT"
    elif not analytic_pass:
        status = "FAIL_CLOSED_ANALYTIC_REFERENCE"
    elif not ideal_pass:
        status = "FAIL_CLOSED_IDEAL_CLOSURE_GATE"
    elif not interacting_pass:
        status = "FAIL_CLOSED_INTERACTING_CLOSURE_GATE"
    else:
        status = "PASS_HVH_CLOSURE"
    overall = bool(
        inventory_valid and maintained_path and analytic_pass and ideal_pass and interacting_pass
    )
    return status, {
        "analytic_reference_pass": analytic_pass,
        "ideal_closure_pass": ideal_pass,
        "interacting_closure_pass_on_smooth_branch": interacting_pass,
        "maintained_path_gate": maintained_path,
        "inventory_contract_valid": inventory_valid,
        "overall_scientific_pass": overall,
    }


def run_audit() -> dict[str, object]:
    ideal = ideal_reference()
    interacting = interacting_reference()
    inventory = maintained_inventory()
    status, terminal_controls = derive_terminal_controls(inventory, ideal, interacting)
    return {
        "audit": "eos_hvh_closure_audit",
        "status": status,
        "producer_inventory": inventory,
        "derivation": {
            "HVH": "p = sum_i(mu_i*n_i) - epsilon; beta equilibrium and neutrality reduce this to n_B*mu_n - epsilon",
            "density_derivative": "p = n_B**2 d(epsilon/n_B)/dn_B, evaluated with fourth-order five-point stencils",
            "scalar_rearrangement": "Sigma_R,s = n_s dM_base/dn_B",
            "vector_rearrangement": "Sigma_R,v = (Cw0/2) n_B**2 dg/dn_B",
            "chemical_potentials": "mu_n,p = E_F,n,p* +/- C_rho(n_n-n_p) + Cw0*g*n_B + Sigma_R,s + Sigma_R,v",
            "ideal_reference": "constant M_N, C_s=C_rho=Cw=0, cold npe-mu beta equilibrium",
        },
        "ideal_reference": ideal,
        "interacting_nvg": interacting,
        "terminal_controls": terminal_controls,
    }


def render_report(result: dict[str, object]) -> str:
    ideal = result["ideal_reference"]
    interacting = result["interacting_nvg"]
    ic = ideal["controls"]
    nc = interacting["controls"]
    p = interacting["parameters"]
    inv = result["producer_inventory"]
    return f"""# HVH closure audit — maintained beta-equilibrated EOS

## Control block

- Status: **{result['status']}** (the scientific result is intentionally fail-closed).
- Scope: independent audit files named `eos_hvh_closure_audit` plus this report; no maintained producer, registry, README, or article edits.
- Producer: `{inv['producer']}` (runtime SHA-256 `{inv['producer_sha256']}`).
- Source scan found {inv['consumer_count']} first-party consumers/importers of this producer; the complete list is in the generated JSON inventory.
- Terminal JSON: `verification/eos_hvh_closure_audit_results.json`.
- Generated repair verification report: `verification/eos_hvh_closure_audit_report.md`.

## Contract inventory and derivation

The maintained producer returns `eps_no_vec`, composition, `m_dirac`, and `mu_e`; its pressure is `n**2*gradient(epsilon/n,n)`. A stress tensor and nucleon chemical-potential path are **absent**, as are explicit rearrangement and stationarity residuals. The audit therefore reconstructs
`p_HVH = sum_i n_i mu_i - epsilon` and compares it with fourth-order five-point estimates of `n**2 d(epsilon/n)/dn` at relative stencil steps 0.02, 0.01, and 0.005. Density-dependent mass and vector terms include `Sigma_R,s = n_s dM_base/dn` and `Sigma_R,v = Cw0*n**2*(dg/dn)/2`.

## Results

| surface | result |
|---|---:|
| analytic power-law derivative benchmark | {ideal['analytic_powerlaw']['max_relative_error']:.3e} relative error ({ideal['analytic_powerlaw']['status']}) |
| ideal cold npe-mu HVH closure | {ic['closure_max_relative']:.3e} max relative |
| ideal derivative convergence | {ic['derivative_convergence_max_relative']:.3e} max relative |
| interacting smooth-mass-branch HVH closure | {nc['closure_max_relative_smooth_mass_branch']:.3e} max relative |
| interacting no-rearrangement closure | {nc['naive_closure_max_relative_smooth_mass_branch']:.3e} max relative |
| interacting derivative convergence | {nc['derivative_convergence_max_relative']:.3e} max relative |
| max scalar rearrangement | {nc['max_scalar_rearrangement_mev']:.3e} MeV |
| max vector rearrangement | {nc['max_vector_rearrangement_mev']:.3e} MeV |
| max Dirac stationarity residual | {nc['max_stationarity_residual_mev']:.3e} MeV |

Interacting inputs are the maintained screening point `(k1,k2,Cs,Crho,alpha_v,nu_v)=({p['k1']:.2f},{p['k2']:.2f},{p['Cs']:.0f},{p['Crho']:.0f},{p['alpha_v']:.1f},{p['nu_v']:.1f})`, with calibrated `Cw0={p['Cw0']:.6g}`. The no-rearrangement comparison is a diagnostic, not a replacement EOS.

## Judgment and boundary

The ideal reference and the independently reconstructed interacting energy functional satisfy the HVH identity on the smooth mass branch within the reported numerical controls. This does not certify the maintained API: because its stress/chemical-potential path is absent, the audit remains **FAIL_CLOSED** and no maintained-producer closure PASS is claimed. The current-mass clipping kink is retained and reported in the JSON rows; derivatives across that non-smooth point are not silently promoted to a smooth theorem.

Exact repair commands and counts are frozen in `Lunacy/runs/deep-physics-audit/phases/phase-1/evidence/hvh-closure-repair-1-terminal-verification.log`.
"""


def write_outputs(result: dict[str, object]) -> None:
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")


def main() -> int:
    result = run_audit()
    write_outputs(result)
    print(json.dumps({
        "status": result["status"],
        "analytic_reference_pass": result["terminal_controls"]["analytic_reference_pass"],
        "ideal_closure_pass": result["terminal_controls"]["ideal_closure_pass"],
        "interacting_closure_pass_on_smooth_branch": result["terminal_controls"]["interacting_closure_pass_on_smooth_branch"],
        "maintained_path_gate": result["terminal_controls"]["maintained_path_gate"],
        "result_path": str(RESULT_PATH),
        "report_path": str(REPORT_PATH),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
