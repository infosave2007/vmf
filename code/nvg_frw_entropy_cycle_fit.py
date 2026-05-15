#!/usr/bin/env python3

"""Pilot FRW + entropy-clock + bounce prototype for NVG Path A.

What this script does:
1. Links the entropy-slowed emergent time variable to FRW expansion.
2. Fits a small parameter grid to compact observational anchors.
3. Implements bounce/reset dynamically through an entropy-drain term near a_b.

Important scope note:
- Cosmic-chronometer and BAO anchors are included directly as a compact pilot set.
- Full Pantheon/Pantheon+ data are not present in this repository, so the script uses
  compressed SN-Ia-like distance-modulus anchors generated from a flat LCDM reference.
  This preserves the fitting pipeline shape without pretending to be a publication-grade
  likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable

import numpy as np


C_KM_S = 299792.458
MPC_IN_KM = 3.0856775814913673e19
SECONDS_PER_GYR = 365.25 * 24.0 * 3600.0 * 1.0e9
HUBBLE_CONVERSION = SECONDS_PER_GYR / MPC_IN_KM
R_D_MPC = 147.1
AGE_TARGET_GYR = 13.8
AGE_SIGMA_GYR = 0.4
CYCLE_TARGET_GYR = 25.0
CYCLE_SIGMA_GYR = 2.5


def hubble_km_s_mpc_to_gyr_inv(hubble: float) -> float:
    return hubble * HUBBLE_CONVERSION


def hubble_gyr_inv_to_km_s_mpc(hubble: float) -> float:
    return hubble / HUBBLE_CONVERSION


def dark_energy_factor(a: np.ndarray | float, w0: float, wa: float) -> np.ndarray | float:
    return np.power(a, -3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))


def trapz_step(y0: float, y1: float, dx: float) -> float:
    return 0.5 * (y0 + y1) * dx


def distance_modulus_from_luminosity_distance(distance_mpc: float) -> float:
    return 5.0 * math.log10(distance_mpc) + 25.0


@dataclass(frozen=True)
class ModelParams:
    h0_meta: float
    omega_m: float
    w0: float
    wa: float
    gamma_0: float
    gamma_h: float
    a_turn: float
    a_bounce: float = 0.05
    bounce_power: int = 8
    turn_power: int = 4
    omega_r: float = 9.0e-5
    s_star: float = 1.0
    q_power: float = 1.0
    q_floor: float = 1.0e-4
    gamma_drain: float = 5.0
    sigma_b: float = 0.05
    gamma_a: float = 0.003


@dataclass
class BranchSolution:
    a_values: np.ndarray
    meta_time_gyr: np.ndarray
    tau_gyr: np.ndarray
    entropy: np.ndarray
    q_values: np.ndarray
    h_meta_gyr_inv: np.ndarray
    h_obs_gyr_inv: np.ndarray
    dc_mpc: np.ndarray


@dataclass
class ModelEvaluation:
    params: ModelParams
    chi2_total: float
    chi2_hz: float
    chi2_bao: float
    chi2_sn: float
    chi2_age: float
    chi2_cycle: float
    age_tau_gyr: float
    cycle_tau_gyr: float
    q0: float
    s0_fraction: float
    bounce_entropy_in: float
    bounce_entropy_out: float
    h0_observed: float
    expansion_solution: BranchSolution


COSMIC_CHRONOMETER_DATA = np.array(
    [
        (0.179, 75.0, 4.0),
        (0.351, 83.0, 14.0),
        (0.593, 104.0, 13.0),
        (0.781, 105.0, 12.0),
        (1.037, 154.0, 20.0),
        (1.363, 160.0, 33.0),
        (1.965, 186.5, 50.4),
    ],
    dtype=float,
)


BAO_DATA = (
    {"z": 0.38, "dm_over_rd": 10.23, "sigma_dm": 0.17, "dh_over_rd": 24.98, "sigma_dh": 0.76},
    {"z": 0.51, "dm_over_rd": 13.36, "sigma_dm": 0.21, "dh_over_rd": 22.33, "sigma_dh": 0.58},
    {"z": 0.61, "dm_over_rd": 15.45, "sigma_dm": 0.31, "dh_over_rd": 20.49, "sigma_dh": 0.60},
)


SN_PROXY_Z = np.array([0.07, 0.15, 0.30, 0.50, 0.80, 1.00], dtype=float)
SN_PROXY_SIGMA = 0.12


def build_sn_proxy_reference() -> np.ndarray:
    reference = ModelParams(
        h0_meta=68.0,
        omega_m=0.315,
        w0=-1.0,
        wa=0.0,
        gamma_0=0.0,
        gamma_h=0.0,
        a_turn=50.0,
        a_bounce=1.0e-4,
        gamma_drain=0.0,
        gamma_a=0.0,
    )
    solution = integrate_expansion_branch(reference, s_bounce=0.0, a_stop=1.0)
    sn_rows = []
    for z in SN_PROXY_Z:
        mu_model = predict_mu(solution, z)
        sn_rows.append((z, mu_model, SN_PROXY_SIGMA))
    return np.array(sn_rows, dtype=float)


SN_PROXY_DATA: np.ndarray | None = None


def envelope_factor(a: float, params: ModelParams) -> float:
    bounce_term = 1.0 - (params.a_bounce / a) ** params.bounce_power
    turn_term = 1.0 - (a / params.a_turn) ** params.turn_power
    return max(0.0, bounce_term * turn_term)


def e2_base(a: float, params: ModelParams) -> float:
    omega_de = 1.0 - params.omega_m - params.omega_r
    return (
        params.omega_r * a ** (-4.0)
        + params.omega_m * a ** (-3.0)
        + omega_de * dark_energy_factor(a, params.w0, params.wa)
    )


def h_meta_gyr_inv(a: float, params: ModelParams) -> float:
    h0_gyr_inv = hubble_km_s_mpc_to_gyr_inv(params.h0_meta)
    return h0_gyr_inv * math.sqrt(max(0.0, e2_base(a, params) * envelope_factor(a, params)))


def entropy_source_gyr_inv(a: float, h_meta: float, params: ModelParams) -> float:
    h0_gyr_inv = hubble_km_s_mpc_to_gyr_inv(params.h0_meta)
    return params.gamma_0 + params.gamma_h * (h_meta / h0_gyr_inv) + params.gamma_a * a ** 3


def entropy_drain_gyr_inv(a: float, params: ModelParams) -> float:
    x = (a - params.a_bounce) / params.sigma_b
    return params.gamma_drain * math.exp(-(x * x))


def q_of_s(entropy: float, params: ModelParams) -> float:
    x = max(entropy / params.s_star, 0.0)
    q_value = max(0.0, 1.0 - x ** params.q_power)
    return max(params.q_floor, q_value)


def integrate_branch(
    params: ModelParams,
    s_initial: float,
    a_start: float,
    a_stop: float,
    steps: int,
) -> BranchSolution:
    a_values = np.linspace(a_start, a_stop, steps)
    meta_time = np.zeros(steps)
    tau_values = np.zeros(steps)
    entropy_values = np.zeros(steps)
    q_values = np.zeros(steps)
    h_meta_values = np.zeros(steps)
    h_obs_values = np.zeros(steps)

    entropy_values[0] = s_initial
    h_meta_values[0] = h_meta_gyr_inv(a_values[0], params)
    q_values[0] = q_of_s(entropy_values[0], params)
    h_obs_values[0] = h_meta_values[0] / q_values[0]

    for index in range(steps - 1):
        a_left = a_values[index]
        a_right = a_values[index + 1]
        a_mid = 0.5 * (a_left + a_right)
        da = a_right - a_left
        h_mid = max(h_meta_gyr_inv(a_mid, params), 1.0e-10)
        dt = abs(da) / (a_mid * h_mid)

        source = entropy_source_gyr_inv(a_mid, h_mid, params)
        drain = entropy_drain_gyr_inv(a_mid, params)
        entropy_mid = max(entropy_values[index], 0.0)
        entropy_next = max(entropy_mid + (source - drain * entropy_mid) * dt, 0.0)
        q_mid = q_of_s(entropy_mid, params)

        meta_time[index + 1] = meta_time[index] + dt
        tau_values[index + 1] = tau_values[index] + q_mid * dt
        entropy_values[index + 1] = entropy_next
        h_meta_values[index + 1] = h_meta_gyr_inv(a_right, params)
        q_values[index + 1] = q_of_s(entropy_next, params)
        h_obs_values[index + 1] = h_meta_values[index + 1] / q_values[index + 1]

    return BranchSolution(
        a_values=a_values,
        meta_time_gyr=meta_time,
        tau_gyr=tau_values,
        entropy=entropy_values,
        q_values=q_values,
        h_meta_gyr_inv=h_meta_values,
        h_obs_gyr_inv=h_obs_values,
        dc_mpc=np.array([]),
    )


def integrate_expansion_branch(params: ModelParams, s_bounce: float, a_stop: float) -> BranchSolution:
    a_start = params.a_bounce * 1.001
    steps = 1400
    solution = integrate_branch(params, s_bounce, a_start, a_stop, steps)
    solution.dc_mpc = build_distance_table(solution)
    return solution


def integrate_full_cycle(params: ModelParams, s_bounce: float) -> tuple[BranchSolution, BranchSolution]:
    a_min = params.a_bounce * 1.001
    a_max = params.a_turn * 0.999
    expansion = integrate_branch(params, s_bounce, a_min, a_max, 1000)
    contraction = integrate_branch(params, expansion.entropy[-1], a_max, a_min, 1000)
    contraction.meta_time_gyr += expansion.meta_time_gyr[-1]
    contraction.tau_gyr += expansion.tau_gyr[-1]
    return expansion, contraction


def periodic_bounce_entropy(params: ModelParams, iterations: int = 4) -> tuple[float, float, float]:
    entropy_in = 0.2
    entropy_out = entropy_in
    for _ in range(iterations):
        _, contraction = integrate_full_cycle(params, entropy_in)
        entropy_out = contraction.entropy[-1]
        if abs(entropy_out - entropy_in) < 1.0e-5:
            break
        entropy_in = entropy_out
    return entropy_in, entropy_out, abs(entropy_out - entropy_in)


def build_distance_table(solution: BranchSolution) -> np.ndarray:
    a_values = solution.a_values
    h_obs_km_s_mpc = hubble_gyr_inv_to_km_s_mpc(solution.h_obs_gyr_inv)
    integrand = C_KM_S / (a_values ** 2 * h_obs_km_s_mpc)
    d_c = np.zeros_like(a_values)
    for index in range(len(a_values) - 2, -1, -1):
        da = a_values[index + 1] - a_values[index]
        d_c[index] = d_c[index + 1] + trapz_step(integrand[index], integrand[index + 1], da)
    return d_c


def interp_monotonic(x: np.ndarray, y: np.ndarray, x_target: float) -> float:
    return float(np.interp(x_target, x, y))


def predict_hz(solution: BranchSolution, z: float) -> float:
    a_target = 1.0 / (1.0 + z)
    h_obs_km_s_mpc = hubble_gyr_inv_to_km_s_mpc(solution.h_obs_gyr_inv)
    return interp_monotonic(solution.a_values, h_obs_km_s_mpc, a_target)


def predict_dc(solution: BranchSolution, z: float) -> float:
    a_target = 1.0 / (1.0 + z)
    if solution.dc_mpc.size == 0:
        solution.dc_mpc = build_distance_table(solution)
    return interp_monotonic(solution.a_values, solution.dc_mpc, a_target)


def predict_mu(solution: BranchSolution, z: float) -> float:
    d_l = (1.0 + z) * predict_dc(solution, z)
    return distance_modulus_from_luminosity_distance(d_l)


def get_sn_proxy_data() -> np.ndarray:
    global SN_PROXY_DATA
    if SN_PROXY_DATA is None:
        SN_PROXY_DATA = build_sn_proxy_reference()
    return SN_PROXY_DATA


def chi2_hz(solution: BranchSolution) -> float:
    total = 0.0
    for z_value, h_obs, sigma_h in COSMIC_CHRONOMETER_DATA:
        model = predict_hz(solution, z_value)
        total += ((model - h_obs) / sigma_h) ** 2
    return total


def chi2_bao(solution: BranchSolution) -> float:
    total = 0.0
    for row in BAO_DATA:
        model_dm = predict_dc(solution, row["z"]) / R_D_MPC
        model_dh = C_KM_S / (predict_hz(solution, row["z"]) * R_D_MPC)
        total += ((model_dm - row["dm_over_rd"]) / row["sigma_dm"]) ** 2
        total += ((model_dh - row["dh_over_rd"]) / row["sigma_dh"]) ** 2
    return total


def chi2_sn(solution: BranchSolution) -> float:
    total = 0.0
    for z_value, mu_obs, sigma_mu in get_sn_proxy_data():
        model = predict_mu(solution, z_value)
        total += ((model - mu_obs) / sigma_mu) ** 2
    return total


def evaluate_model(params: ModelParams) -> ModelEvaluation | None:
    bounce_entropy_in, bounce_entropy_out, _ = periodic_bounce_entropy(params)
    expansion_solution = integrate_expansion_branch(params, bounce_entropy_out, a_stop=1.0)
    full_expansion, full_contraction = integrate_full_cycle(params, bounce_entropy_out)
    age_tau = expansion_solution.tau_gyr[-1]
    cycle_tau = full_contraction.tau_gyr[-1]
    q0 = expansion_solution.q_values[-1]
    h0_observed = hubble_gyr_inv_to_km_s_mpc(expansion_solution.h_obs_gyr_inv[-1])

    if not np.isfinite(age_tau) or not np.isfinite(cycle_tau):
        return None
    if q0 <= params.q_floor + 1.0e-7:
        return None

    chi_hz = chi2_hz(expansion_solution)
    chi_bao_total = chi2_bao(expansion_solution)
    chi_sn_total = chi2_sn(expansion_solution)
    chi_age = ((age_tau - AGE_TARGET_GYR) / AGE_SIGMA_GYR) ** 2
    chi_cycle = ((cycle_tau - CYCLE_TARGET_GYR) / CYCLE_SIGMA_GYR) ** 2
    chi_total = chi_hz + chi_bao_total + chi_sn_total + chi_age + chi_cycle

    return ModelEvaluation(
        params=params,
        chi2_total=chi_total,
        chi2_hz=chi_hz,
        chi2_bao=chi_bao_total,
        chi2_sn=chi_sn_total,
        chi2_age=chi_age,
        chi2_cycle=chi_cycle,
        age_tau_gyr=age_tau,
        cycle_tau_gyr=cycle_tau,
        q0=q0,
        s0_fraction=expansion_solution.entropy[-1] / params.s_star,
        bounce_entropy_in=bounce_entropy_in,
        bounce_entropy_out=bounce_entropy_out,
        h0_observed=h0_observed,
        expansion_solution=expansion_solution,
    )


def parameter_grid() -> Iterable[ModelParams]:
    for h0_meta, omega_m, w0, wa, gamma_0, gamma_h, a_turn in product(
        [46.0, 48.0],
        [0.29, 0.31],
        [-1.00, -0.95],
        [0.00],
        [0.010, 0.014],
        [0.002, 0.004],
        [1.8, 2.0],
    ):
        yield ModelParams(
            h0_meta=h0_meta,
            omega_m=omega_m,
            w0=w0,
            wa=wa,
            gamma_0=gamma_0,
            gamma_h=gamma_h,
            a_turn=a_turn,
        )


def print_anchor_checks(evaluation: ModelEvaluation) -> None:
    solution = evaluation.expansion_solution
    print()
    print("Chronometer anchors")
    print(f"{'z':>6} {'H_data':>10} {'H_model':>10} {'resid/sig':>10}")
    print("-" * 42)
    for z_value, h_obs, sigma_h in COSMIC_CHRONOMETER_DATA:
        model = predict_hz(solution, z_value)
        resid = (model - h_obs) / sigma_h
        print(f"{z_value:6.3f} {h_obs:10.2f} {model:10.2f} {resid:10.3f}")

    print()
    print("BAO anchors")
    print(f"{'z':>6} {'DM/rd data':>12} {'DM/rd mod':>12} {'DH/rd data':>12} {'DH/rd mod':>12}")
    print("-" * 64)
    for row in BAO_DATA:
        model_dm = predict_dc(solution, row['z']) / R_D_MPC
        model_dh = C_KM_S / (predict_hz(solution, row['z']) * R_D_MPC)
        print(
            f"{row['z']:6.2f} {row['dm_over_rd']:12.3f} {model_dm:12.3f} "
            f"{row['dh_over_rd']:12.3f} {model_dh:12.3f}"
        )

    print()
    print("SN-like anchors")
    print(f"{'z':>6} {'mu_ref':>10} {'mu_model':>10} {'resid/sig':>10}")
    print("-" * 42)
    for z_value, mu_obs, sigma_mu in get_sn_proxy_data():
        model = predict_mu(solution, z_value)
        resid = (model - mu_obs) / sigma_mu
        print(f"{z_value:6.2f} {mu_obs:10.3f} {model:10.3f} {resid:10.3f}")


def main() -> None:
    print("=" * 88)
    print("NVG FRW + ENTROPY-CLOCK + BOUNCE PILOT FIT")
    print("FRW in meta-time, observable clock d tau / dt = q(S), dynamic entropy drain near bounce")
    print("=" * 88)
    print()
    print("Datasets used")
    print("  1. Compact cosmic-chronometer H(z) anchors")
    print("  2. Compact BAO DM/rd and DH/rd anchors")
    print("  3. SN-Ia-like compressed distance anchors from a flat LCDM reference")
    print()

    evaluations: list[ModelEvaluation] = []
    total_models = 0
    for params in parameter_grid():
        total_models += 1
        evaluation = evaluate_model(params)
        if evaluation is not None:
            evaluations.append(evaluation)

    evaluations.sort(key=lambda item: item.chi2_total)

    print(f"Scanned models: {total_models}")
    print(f"Evaluated models: {len(evaluations)}")
    print()

    if not evaluations:
        print("No numerically stable model found.")
        return

    best = evaluations[0]
    print("Best model")
    print(f"  chi2_total      = {best.chi2_total:.3f}")
    print(f"  chi2_H(z)       = {best.chi2_hz:.3f}")
    print(f"  chi2_BAO        = {best.chi2_bao:.3f}")
    print(f"  chi2_SN-like    = {best.chi2_sn:.3f}")
    print(f"  chi2_age        = {best.chi2_age:.3f}")
    print(f"  chi2_cycle      = {best.chi2_cycle:.3f}")
    print()
    print("Best-fit parameters")
    print(f"  H0_meta         = {best.params.h0_meta:.2f} km/s/Mpc")
    print(f"  omega_m         = {best.params.omega_m:.3f}")
    print(f"  w0              = {best.params.w0:.3f}")
    print(f"  wa              = {best.params.wa:.3f}")
    print(f"  gamma_0         = {best.params.gamma_0:.4f} Gyr^-1")
    print(f"  gamma_h         = {best.params.gamma_h:.4f} Gyr^-1")
    print(f"  a_turn          = {best.params.a_turn:.3f}")
    print()
    print("Derived present/cycle quantities")
    print(f"  H0_observed     = {best.h0_observed:.2f} km/s/Mpc")
    print(f"  tau_0           = {best.age_tau_gyr:.3f} Gyr")
    print(f"  tau_cycle       = {best.cycle_tau_gyr:.3f} Gyr")
    print(f"  q0              = {best.q0:.3f}")
    print(f"  S0 / S_*        = {best.s0_fraction:.3f}")
    print(f"  S_bounce,in     = {best.bounce_entropy_in:.5f}")
    print(f"  S_bounce,out    = {best.bounce_entropy_out:.5f}")
    print()
    print("Interpretation")
    print("  1. Step 1 is implemented: H_obs(z) is computed from FRW using the internal clock tau, not meta-time t.")
    print("  2. Step 2 is implemented as a pilot fit to chronometer, BAO, and SN-like anchors plus age and cycle priors.")
    print("  3. Step 3 is implemented through a dynamic entropy-drain term near the bounce instead of a manual reset map.")

    print_anchor_checks(best)


if __name__ == "__main__":
    main()