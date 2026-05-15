#!/usr/bin/env python3

"""Toy experiments for an entropy-slowed emergent time variable.

This script does not prove fundamental physics. It tests whether a specific
assumption set is mathematically self-consistent:

1. There is a meta-time t in which a cycle evolves.
2. The observable/emergent time tau flows with rate q(S) = d tau / dt.
3. Entropy S(t) grows during a cycle and suppresses the clock rate q(S).
4. When S reaches a critical value S_*, the emergent clock stops.
5. A bounce/reset maps S -> S_reset and a new cycle begins.

Main theorem used here:

If dS/dt >= Gamma > 0 and q(S) <= max(0, 1 - S / S_*), then the total
emergent duration of one cycle is finite and obeys

    tau_stop <= S_* / (2 * Gamma).

The proof is one line: S(t) >= Gamma * t, hence q(t) <= max(0, 1 - Gamma t / S_*),
and integrating the right-hand side up to t_* = S_* / Gamma yields the bound.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


TARGET_TAU_GYR = 25.0
PRESENT_AGE_GYR = 13.8


def clamp_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def q_linear(s: float, s_star: float) -> float:
    x = clamp_unit(s / s_star)
    return max(0.0, 1.0 - x)


def q_quadratic(s: float, s_star: float) -> float:
    x = clamp_unit(s / s_star)
    return max(0.0, 1.0 - x) ** 2


def q_cosine(s: float, s_star: float) -> float:
    x = clamp_unit(s / s_star)
    if x >= 1.0:
        return 0.0
    return math.cos(0.5 * math.pi * x) ** 2


def s_constant_growth(meta_time_gyr: float, gamma: float) -> float:
    return gamma * meta_time_gyr


@dataclass(frozen=True)
class ToyModel:
    name: str
    s_star: float
    gamma: float
    clock_rate: callable
    analytic_tau_stop_gyr: float


def integrate_cycle(model: ToyModel, dt_gyr: float = 0.001) -> dict[str, float]:
    meta_time = 0.0
    tau = 0.0
    max_steps = 5_000_000
    steps = 0
    while True:
        entropy = s_constant_growth(meta_time, model.gamma)
        rate = model.clock_rate(entropy, model.s_star)
        if rate <= 0.0:
            break
        tau += rate * dt_gyr
        meta_time += dt_gyr
        steps += 1
        if steps >= max_steps:
            raise RuntimeError(f"integration did not stop for model: {model.name}")

    entropy = s_constant_growth(meta_time, model.gamma)
    rate = model.clock_rate(entropy, model.s_star)
    return {
        "meta_stop_gyr": meta_time,
        "tau_stop_gyr": tau,
        "entropy_stop": entropy,
        "clock_stop": rate,
    }


def theorem_bound_linear_envelope(s_star: float, gamma: float) -> float:
    return s_star / (2.0 * gamma)


def find_meta_time_for_tau(model: ToyModel, target_tau_gyr: float, dt_gyr: float = 0.0005) -> dict[str, float] | None:
    meta_time = 0.0
    tau = 0.0
    prev_tau = 0.0
    max_steps = 10_000_000
    steps = 0
    while True:
        entropy = s_constant_growth(meta_time, model.gamma)
        rate = model.clock_rate(entropy, model.s_star)
        if rate <= 0.0:
            return None
        prev_tau = tau
        tau += rate * dt_gyr
        meta_time += dt_gyr
        steps += 1
        if steps >= max_steps:
            raise RuntimeError(f"target tau was not reached for model: {model.name}")
        if tau >= target_tau_gyr:
            entropy = s_constant_growth(meta_time, model.gamma)
            rate = model.clock_rate(entropy, model.s_star)
            return {
                "meta_time_gyr": meta_time,
                "tau_gyr": tau,
                "tau_prev_gyr": prev_tau,
                "entropy": entropy,
                "entropy_fraction": entropy / model.s_star,
                "clock_rate": rate,
            }


def calibrate_models() -> list[ToyModel]:
    # For q = (1 - x)^p with constant entropy growth, tau_stop = S_* / ((p + 1) * gamma).
    linear_gamma = 1.0 / (2.0 * TARGET_TAU_GYR)
    quadratic_gamma = 1.0 / (3.0 * TARGET_TAU_GYR)
    # Integral_0^1 cos^2(pi x / 2) dx = 1/2.
    cosine_gamma = 1.0 / (2.0 * TARGET_TAU_GYR)
    return [
        ToyModel(
            name="Linear clock",
            s_star=1.0,
            gamma=linear_gamma,
            clock_rate=q_linear,
            analytic_tau_stop_gyr=TARGET_TAU_GYR,
        ),
        ToyModel(
            name="Quadratic clock",
            s_star=1.0,
            gamma=quadratic_gamma,
            clock_rate=q_quadratic,
            analytic_tau_stop_gyr=TARGET_TAU_GYR,
        ),
        ToyModel(
            name="Cosine clock",
            s_star=1.0,
            gamma=cosine_gamma,
            clock_rate=q_cosine,
            analytic_tau_stop_gyr=TARGET_TAU_GYR,
        ),
    ]


def run_cycle_reset_demo(model: ToyModel, cycles: int = 3) -> list[dict[str, float]]:
    results: list[dict[str, float]] = []
    meta_offset = 0.0
    tau_offset = 0.0
    for cycle_index in range(1, cycles + 1):
        cycle = integrate_cycle(model)
        meta_offset += cycle["meta_stop_gyr"]
        tau_offset += cycle["tau_stop_gyr"]
        results.append(
            {
                "cycle": float(cycle_index),
                "meta_cumulative_gyr": meta_offset,
                "tau_cumulative_gyr": tau_offset,
            }
        )
    return results


def print_header() -> None:
    print("=" * 78)
    print("NVG TOY EXPERIMENT: ENTROPY-SLOWED EMERGENT TIME")
    print("meta-time t, entropy S(t), emergent clock d tau / dt = q(S)")
    print("goal: test whether a finite 25 Gyr emergent cycle is mathematically possible")
    print("=" * 78)
    print()


def print_theorem() -> None:
    print("Theorem used in the toy model")
    print("  If dS/dt >= Gamma > 0 and q(S) <= max(0, 1 - S/S_*),")
    print("  then tau_stop <= S_* / (2 Gamma).")
    print()


def main() -> None:
    print_header()
    print_theorem()

    models = calibrate_models()
    print(f"Target emergent duration per cycle: {TARGET_TAU_GYR:.1f} Gyr")
    print()
    print(f"{'Model':<18} {'Gamma':>10} {'meta stop':>12} {'tau stop':>12} {'bound':>12}")
    print("-" * 70)
    for model in models:
        numerical = integrate_cycle(model)
        theorem_bound = theorem_bound_linear_envelope(model.s_star, model.gamma)
        print(
            f"{model.name:<18} "
            f"{model.gamma:10.6f} "
            f"{numerical['meta_stop_gyr']:12.3f} "
            f"{numerical['tau_stop_gyr']:12.3f} "
            f"{theorem_bound:12.3f}"
        )
    print()

    linear_model = models[0]
    snapshot = find_meta_time_for_tau(linear_model, PRESENT_AGE_GYR)
    if snapshot is not None:
        print("Linear-model snapshot at a present-like emergent age")
        print(f"  tau_now        = {snapshot['tau_gyr']:.3f} Gyr")
        print(f"  meta time      = {snapshot['meta_time_gyr']:.3f} Gyr")
        print(f"  entropy/S_*    = {snapshot['entropy_fraction']:.3f}")
        print(f"  clock fraction = {snapshot['clock_rate']:.3f}")
        print()

    print("Cycle reset demo using the linear model")
    for result in run_cycle_reset_demo(linear_model, cycles=3):
        cycle_index = int(result["cycle"])
        print(
            f"  cycle {cycle_index}: meta cumulative = {result['meta_cumulative_gyr']:.3f} Gyr, "
            f"tau cumulative = {result['tau_cumulative_gyr']:.3f} Gyr"
        )
    print()
    print("Interpretation")
    print("  1. A finite emergent duration is easy to obtain once the clock rate goes to zero at a critical entropy.")
    print("  2. The 25 Gyr number is not derived here; it is a calibration target used to choose Gamma.")
    print("  3. The result is therefore a consistency proof for a toy model, not evidence that Nature uses it.")


if __name__ == "__main__":
    main()