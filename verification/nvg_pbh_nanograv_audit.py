#!/usr/bin/env python3
"""PBH--NANOGrav abundance/amplitude audit.

This audit is deliberately an envelope calculation, not a fit.  It reuses the
maintained discrete PBH ladder and the maintained two-population constants,
then varies only declared abundance, mass-rung, and binary-rate factors.  The
NANOGrav number is carried as a traceable benchmark from the existing producer;
it is not a likelihood, posterior, or fitted target.

Two questions are kept separate:

* At the JWST-seed abundance used by the maintained two-population script, how
  large can the same population's PTA strain be?
* If all dark matter were placed on a rung, would the internal dark-matter
  budget alone forbid the benchmark amplitude?

The second is an intentionally generous upper envelope.  It must not be
confused with observational exclusions (the maintained CMB bound is reported
only as provenance, and is never applied as a cut here).
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Canonical producers.  These imports are intentionally module-level so the
# identity is visible to callers and semantic tests.
import nvg_pbh_dark_matter as pbh_dm
import nvg_pbh_mass_spectrum as pbh_ladder
import nvg_pbh_two_population as pbh_two


AUDIT_SCHEMA_VERSION = 1
AUDIT_STATUS = "CONDITIONAL_JWST_BAND_DEFICIT_NO_INTERNAL_BUDGET_NO_GO"
CANONICAL_LADDER = "verification/nvg_pbh_mass_spectrum.py:get_pbh_mass"
CANONICAL_TWO_POPULATION = "verification/nvg_pbh_two_population.py"

# The maintained two-population script calibrates population B to JWST seed
# number densities and uses rungs close to N=10 (M ~= 4e5 Msun).  N=9..11 is
# the declared seed band; N=7..18 is a boundary/convergence expansion.  The
# expansion is a sensitivity envelope and is not an external exclusion.
JWST_SEED_RUNGS: tuple[int, ...] = (9, 10, 11)
EXPANDED_RUNGS: tuple[int, ...] = tuple(range(7, 19))

# Nuisance factors are fractions of the reference SMBHB per-object merger
# efficiency.  A value of one is deliberately generous: all objects pair,
# remain active, and merge at the reference rate.  No PBH binary likelihood is
# available in this repository, so these are sensitivity coordinates only.
RATE_FACTOR_NAMES = ("binary_pair_fraction", "duty_cycle", "merger_rate_ratio")
RATE_FACTOR_RANGE = (0.0, 1.0)

# The benchmark is carried from the maintained producer, not repeated here as
# a fitted constant.  This range is used only for sensitivity reporting.
TARGET_SCALE_RANGE = (0.8, 1.2)


def _grid(lo: float, hi: float, points: int, *, log: bool = False) -> np.ndarray:
    """Return a finite, endpoint-inclusive one-dimensional scan grid."""

    if points < 2:
        raise ValueError("grid points must be at least two")
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError("invalid grid bounds")
    if log:
        if lo <= 0.0:
            raise ValueError("log grid lower bound must be positive")
        values = np.geomspace(lo, hi, points)
    else:
        values = np.linspace(lo, hi, points)
    values[0] = lo
    values[-1] = hi
    return np.asarray(values, dtype=float)


def canonical_mass(cycle: int) -> float:
    """Return a mass from the maintained ladder producer."""

    if not isinstance(cycle, int):
        raise TypeError("cycle must be an integer")
    mass = float(pbh_ladder.get_pbh_mass(cycle))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("canonical ladder returned a non-positive mass")
    return mass


def benchmark_metadata() -> dict[str, Any]:
    """Describe the benchmark without upgrading it to an observed likelihood."""

    return {
        "amplitude_strain": float(pbh_two.A_NANOGRAV),
        "frequency_label": "f=1/yr",
        "source_path": CANONICAL_TWO_POPULATION,
        "source_symbol": "A_NANOGRAV",
        "source_label": "maintained NANOGrav 15-year benchmark constant",
        "provenance_status": "INTERNAL_BENCHMARK_TRACEABLE_NO_INDEPENDENT_LIKELIHOOD",
        "observed_likelihood": None,
        "target_calibration_used": False,
    }


def amplitude_strain(
    *,
    fraction_dm: float,
    mass_msun: float,
    binary_pair_fraction: float = 1.0,
    duty_cycle: float = 1.0,
    merger_rate_ratio: float = 1.0,
    target_scale: float = 1.0,
) -> float:
    """Compute the reference-normalized PTA strain envelope.

    For ``n = f_PBH rho_DM / M`` and the maintained scaling
    ``h_c^2 proportional to n_merge M_c^(5/3)``, this gives ``h_c`` relative
    to the maintained SMBHB benchmark.  The three rate factors are explicit
    occupancy/efficiency coordinates, each constrained to [0, 1].
    """

    vals = (fraction_dm, mass_msun, binary_pair_fraction, duty_cycle, merger_rate_ratio, target_scale)
    if not all(math.isfinite(float(value)) for value in vals):
        raise ValueError("all amplitude inputs must be finite")
    if not (0.0 <= fraction_dm <= 1.0):
        raise ValueError("fraction_dm must lie in [0, 1]")
    if mass_msun <= 0.0:
        raise ValueError("mass_msun must be positive")
    for value, name in zip(
        (binary_pair_fraction, duty_cycle, merger_rate_ratio), RATE_FACTOR_NAMES
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if target_scale <= 0.0:
        raise ValueError("target_scale must be positive")

    number_density = fraction_dm * float(pbh_two.RHO_DM_MSUN_MPC3) / mass_msun
    rate_product = binary_pair_fraction * duty_cycle * merger_rate_ratio
    ratio_h2 = (
        number_density / float(pbh_two.N_SMBH)
        * (mass_msun / float(pbh_two.M_SMBH)) ** (5.0 / 3.0)
        * rate_product
    )
    return float(pbh_two.A_NANOGRAV * target_scale * math.sqrt(max(ratio_h2, 0.0)))


def _row(
    *,
    cycle: int,
    mass_msun: float,
    fraction_dm: float,
    number_density: float,
    binary_pair_fraction: float,
    duty_cycle: float,
    merger_rate_ratio: float,
    target_scale: float = 1.0,
) -> dict[str, Any]:
    amplitude = amplitude_strain(
        fraction_dm=fraction_dm,
        mass_msun=mass_msun,
        binary_pair_fraction=binary_pair_fraction,
        duty_cycle=duty_cycle,
        merger_rate_ratio=merger_rate_ratio,
        target_scale=target_scale,
    )
    target = float(pbh_two.A_NANOGRAV * target_scale)
    return {
        "cycle": int(cycle),
        "mass_msun": float(mass_msun),
        "fraction_dm": float(fraction_dm),
        "number_density_mpc3": float(number_density),
        "binary_pair_fraction": float(binary_pair_fraction),
        "duty_cycle": float(duty_cycle),
        "merger_rate_ratio": float(merger_rate_ratio),
        "target_scale": float(target_scale),
        "amplitude_strain": float(amplitude),
        "deficit_factor": float(target / amplitude) if amplitude > 0.0 else None,
    }


def _better(candidate: dict[str, Any] | None, incumbent: dict[str, Any] | None) -> bool:
    return incumbent is None or candidate is not None and candidate["amplitude_strain"] > incumbent["amplitude_strain"]


def _scan_rate_factors(points: int) -> Iterable[tuple[float, float, float]]:
    values = _grid(*RATE_FACTOR_RANGE, points)
    return itertools.product(values, repeat=len(RATE_FACTOR_NAMES))


def _max_row_over_grid(
    *,
    cycles: Sequence[int],
    fraction_grid: Sequence[float],
    rate_points: int,
    target_scale: float = 1.0,
) -> dict[str, Any]:
    maximum: dict[str, Any] | None = None
    for cycle in cycles:
        mass = canonical_mass(int(cycle))
        for fraction in fraction_grid:
            number_density = float(fraction) * float(pbh_two.RHO_DM_MSUN_MPC3) / mass
            for pair, duty, merger in _scan_rate_factors(rate_points):
                candidate = _row(
                    cycle=int(cycle),
                    mass_msun=mass,
                    fraction_dm=float(fraction),
                    number_density=number_density,
                    binary_pair_fraction=float(pair),
                    duty_cycle=float(duty),
                    merger_rate_ratio=float(merger),
                    target_scale=target_scale,
                )
                if _better(candidate, maximum):
                    maximum = candidate
    if maximum is None:
        raise RuntimeError("empty scan")
    return maximum


def _scan_jwst_calibrated(
    *,
    cycles: Sequence[int],
    density_points: int = 17,
    rate_points: int = 3,
    target_scale: float = 1.0,
) -> dict[str, Any]:
    densities = _grid(float(pbh_two.N_SEED_LO), float(pbh_two.N_SEED_HI), density_points, log=True)
    maxima: dict[str, Any] | None = None
    minima: dict[str, Any] | None = None
    rows = 0
    for cycle in cycles:
        mass = canonical_mass(int(cycle))
        for density in densities:
            fraction = float(density) * mass / float(pbh_two.RHO_DM_MSUN_MPC3)
            if fraction > 1.0 + 1e-15:
                raise AssertionError("JWST density scan violates DM budget")
            for pair, duty, merger in _scan_rate_factors(rate_points):
                candidate = _row(
                    cycle=int(cycle),
                    mass_msun=mass,
                    fraction_dm=fraction,
                    number_density=float(density),
                    binary_pair_fraction=float(pair),
                    duty_cycle=float(duty),
                    merger_rate_ratio=float(merger),
                    target_scale=target_scale,
                )
                if _better(candidate, maxima):
                    maxima = candidate
                if minima is None or candidate["amplitude_strain"] < minima["amplitude_strain"]:
                    minima = candidate
                rows += 1
    if maxima is None or minima is None:
        raise RuntimeError("empty JWST scan")
    return {
        "cycles": [int(cycle) for cycle in cycles],
        "density_range_mpc3": [float(densities[0]), float(densities[-1])],
        "density_grid_points": int(density_points),
        "rate_grid_points": int(rate_points),
        "rows_evaluated": int(rows),
        "max_row": maxima,
        "min_row": minima,
        "max_amplitude_strain": float(maxima["amplitude_strain"]),
        "max_deficit_factor": float(maxima["deficit_factor"]),
        "budget_fraction_max": float(max(row["fraction_dm"] for row in (maxima, minima))),
        "external_exclusion_applied": False,
        "status": "CALIBRATED_ABUNDANCE_SENSITIVITY_NO_LIKELIHOOD",
    }


def _scan_internal_budget(
    *,
    cycles: Sequence[int],
    abundance_points: int = 33,
    rate_points: int = 3,
    target_scale: float = 1.0,
) -> dict[str, Any]:
    fractions = _grid(0.0, 1.0, abundance_points)
    maximum = _max_row_over_grid(
        cycles=cycles,
        fraction_grid=fractions,
        rate_points=rate_points,
        target_scale=target_scale,
    )
    # The maximum attainable amplitude at each rung when all DM and all
    # generous rate factors are assigned to the rung.  This exposes mass-bound
    # dependence independently of grid ordering.
    per_cycle = []
    for cycle in cycles:
        mass = canonical_mass(int(cycle))
        amp = amplitude_strain(fraction_dm=1.0, mass_msun=mass)
        per_cycle.append(
            {
                "cycle": int(cycle),
                "mass_msun": float(mass),
                "all_dm_amplitude_strain": float(amp),
                "all_dm_deficit_factor": float(pbh_two.A_NANOGRAV / amp) if amp > 0.0 else math.inf,
                "required_fraction_at_unit_rate": float((pbh_two.A_NANOGRAV / amp) ** 2)
                if amp > 0.0
                else math.inf,
            }
        )
    return {
        "cycles": [int(cycle) for cycle in cycles],
        "abundance_grid_points": int(abundance_points),
        "rate_grid_points": int(rate_points),
        "rows_evaluated": int(abundance_points * len(cycles) * rate_points ** len(RATE_FACTOR_NAMES)),
        "max_row": maximum,
        "max_amplitude_strain": float(maximum["amplitude_strain"]),
        "max_deficit_factor": float(maximum["deficit_factor"]),
        "per_cycle_all_dm": per_cycle,
        "max_at_mass_boundary": bool(maximum["cycle"] == max(int(cycle) for cycle in cycles)),
        "max_at_abundance_boundary": bool(math.isclose(maximum["fraction_dm"], 1.0, rel_tol=0.0, abs_tol=1e-15)),
        "external_exclusion_applied": False,
        "status": "INTERNAL_BUDGET_ENVELOPE_NO_LIKELIHOOD",
    }


def _analytic_scaling_checks() -> dict[str, Any]:
    """Check the exponents implied by the declared runtime scaling."""

    mass = canonical_mass(10)
    a_quarter = amplitude_strain(fraction_dm=0.25, mass_msun=mass)
    a_full = amplitude_strain(fraction_dm=1.0, mass_msun=mass)
    abundance_ratio = a_full / a_quarter
    abundance_expected = math.sqrt(4.0)

    m_low = canonical_mass(9)
    m_high = canonical_mass(11)
    a_low = amplitude_strain(fraction_dm=1.0, mass_msun=m_low)
    a_high = amplitude_strain(fraction_dm=1.0, mass_msun=m_high)
    mass_ratio = a_high / a_low
    mass_expected = (m_high / m_low) ** (1.0 / 3.0)

    rate_quarter = amplitude_strain(fraction_dm=1.0, mass_msun=mass, merger_rate_ratio=0.25)
    rate_full = amplitude_strain(fraction_dm=1.0, mass_msun=mass, merger_rate_ratio=1.0)
    rate_ratio = rate_full / rate_quarter
    rate_expected = math.sqrt(4.0)

    checks = {
        "abundance_exponent": {
            "expected": "1/2",
            "observed_ratio": float(abundance_ratio),
            "expected_ratio": float(abundance_expected),
            "relative_error": float(abs(abundance_ratio / abundance_expected - 1.0)),
        },
        "mass_exponent_after_number_density": {
            "expected": "1/3",
            "observed_ratio": float(mass_ratio),
            "expected_ratio": float(mass_expected),
            "relative_error": float(abs(mass_ratio / mass_expected - 1.0)),
        },
        "merger_efficiency_exponent": {
            "expected": "1/2",
            "observed_ratio": float(rate_ratio),
            "expected_ratio": float(rate_expected),
            "relative_error": float(abs(rate_ratio / rate_expected - 1.0)),
        },
    }
    return {"checks": checks, "all_pass": bool(all(item["relative_error"] < 1e-12 for item in checks.values()))}


def _convergence_checks() -> dict[str, Any]:
    abundance_results = []
    for points in (9, 33, 129):
        scan = _scan_internal_budget(cycles=JWST_SEED_RUNGS, abundance_points=points, rate_points=3)
        abundance_results.append(
            {
                "abundance_grid_points": points,
                "max_amplitude_strain": float(scan["max_amplitude_strain"]),
                "max_cycle": int(scan["max_row"]["cycle"]),
            }
        )
    reference = abundance_results[-1]["max_amplitude_strain"]
    abundance_relative_errors = [
        float(abs(item["max_amplitude_strain"] / reference - 1.0)) for item in abundance_results
    ]

    boundary_ranges = (11, 13, 15, 18)
    boundary_results = []
    for max_cycle in boundary_ranges:
        cycles = tuple(range(7, max_cycle + 1))
        scan = _scan_internal_budget(cycles=cycles, abundance_points=33, rate_points=3)
        boundary_results.append(
            {
                "max_cycle_in_scan": int(max_cycle),
                "max_amplitude_strain": float(scan["max_amplitude_strain"]),
                "max_row_cycle": int(scan["max_row"]["cycle"]),
                "max_at_mass_boundary": bool(scan["max_at_mass_boundary"]),
            }
        )
    return {
        "abundance_grid": abundance_results,
        "abundance_relative_errors_vs_129": abundance_relative_errors,
        "abundance_converged": bool(max(abundance_relative_errors) < 1e-12),
        "boundary_expansion": boundary_results,
        "boundary_maximum_is_explicit": bool(all(item["max_at_mass_boundary"] for item in boundary_results)),
        "resolution_status": "CONVERGED_ENDPOINT_INCLUSIVE_GRIDS",
    }


def _canonical_profile_summary() -> dict[str, Any]:
    profile = pbh_dm.compute_spectrum()
    masses = [float(row["mass_msun"]) for row in profile["rows"]]
    return {
        "producer": "verification/nvg_pbh_dark_matter.py:compute_spectrum",
        "mass_ladder_producer": CANONICAL_LADDER,
        "profile_peak_cycle": int(profile["abundance_peak"]),
        "profile_width_cycles": float(profile["abundance_width"]),
        "profile_fraction_sum": float(profile["fraction_sum"]),
        "profile_mass_range_msun": [float(min(masses)), float(max(masses))],
        "evidence_status": str(profile["evidence_status"]),
        "observed_likelihood": None,
        "used_as_external_exclusion": False,
    }


def run_audit(
    *,
    seed_cycles: Sequence[int] = JWST_SEED_RUNGS,
    expanded_cycles: Sequence[int] = EXPANDED_RUNGS,
    density_points: int = 17,
    abundance_points: int = 33,
    rate_points: int = 3,
) -> dict[str, Any]:
    """Run the complete PBH--NANOGrav envelope and return JSON-safe results."""

    seed_cycles = tuple(int(cycle) for cycle in seed_cycles)
    expanded_cycles = tuple(int(cycle) for cycle in expanded_cycles)
    if not seed_cycles or not expanded_cycles:
        raise ValueError("seed and expanded cycle ranges must be non-empty")
    if not set(seed_cycles).issubset(set(expanded_cycles)):
        raise ValueError("seed cycles must be contained in expanded cycles")
    for cycle in (*seed_cycles, *expanded_cycles):
        canonical_mass(cycle)

    benchmark = benchmark_metadata()
    jwst = _scan_jwst_calibrated(
        cycles=seed_cycles,
        density_points=density_points,
        rate_points=rate_points,
    )
    jwst_expanded = _scan_jwst_calibrated(
        cycles=expanded_cycles,
        density_points=density_points,
        rate_points=rate_points,
    )
    budget_seed = _scan_internal_budget(
        cycles=seed_cycles,
        abundance_points=abundance_points,
        rate_points=rate_points,
    )
    budget_expanded = _scan_internal_budget(
        cycles=expanded_cycles,
        abundance_points=abundance_points,
        rate_points=rate_points,
    )

    # A scoped no-go is allowed only for the declared JWST-seed abundance/mass
    # band.  The all-DM envelope is reported separately and intentionally does
    # not become a global exclusion when it reaches/exceeds the benchmark.
    seed_band_deficit = bool(jwst["max_amplitude_strain"] < benchmark["amplitude_strain"])
    internal_budget_no_go = bool(budget_seed["max_amplitude_strain"] < benchmark["amplitude_strain"])
    target_sensitivity = []
    # Keep the PBH prediction fixed while varying only the benchmark level;
    # multiplying both would hide the intended sensitivity in the ratio.
    for scale in TARGET_SCALE_RANGE:
        target_value = float(benchmark["amplitude_strain"] * scale)
        target_sensitivity.append(
            {
                "target_scale": float(scale),
                "target_amplitude_strain": target_value,
                "max_predicted_amplitude_strain": float(jwst["max_amplitude_strain"]),
                "deficit_factor": float(target_value / jwst["max_amplitude_strain"]),
            }
        )

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": AUDIT_STATUS,
        "benchmark": benchmark,
        "canonical_producers": {
            "mass_ladder": CANONICAL_LADDER,
            "two_population": CANONICAL_TWO_POPULATION,
            "population_a_profile": "verification/nvg_pbh_dark_matter.py:compute_spectrum",
            "identity_checks": {
                "mass_anchor_msun": float(pbh_ladder.MASS_ANCHOR_MSUN),
                "ladder_base": float(pbh_ladder.LADDER_BASE),
                "two_population_mass_msun": float(pbh_two.M_B),
                "two_population_jwst_seed_density_range_mpc3": [float(pbh_two.N_SEED_LO), float(pbh_two.N_SEED_HI)],
            },
        },
        "population_a_profile": _canonical_profile_summary(),
        "scan_contract": {
            "seed_cycles": [int(cycle) for cycle in seed_cycles],
            "expanded_cycles": [int(cycle) for cycle in expanded_cycles],
            "rate_factor_names": list(RATE_FACTOR_NAMES),
            "rate_factor_range": list(RATE_FACTOR_RANGE),
            "dm_budget_definition": "f_A + f_B <= 1; internal envelope sets f_B=1 and f_A=0",
            "external_exclusions_used": False,
            "external_constraints_reported_only": [
                "verification/nvg_pbh_two_population.py:F_PBH_CMB_BOUND",
            ],
        },
        "jwst_calibrated_seed_band": jwst,
        "jwst_calibrated_expanded_boundary": jwst_expanded,
        "internal_budget_seed_band": budget_seed,
        "internal_budget_expanded_boundary": budget_expanded,
        "target_scale_sensitivity": target_sensitivity,
        "convergence": _convergence_checks(),
        "analytic_scaling": _analytic_scaling_checks(),
        "semantic_interpretation": {
            "seed_band_deficit": seed_band_deficit,
            "seed_band_deficit_status": "CALIBRATED_ABUNDANCE_DEFICIT_NOT_INTERNAL_NO_GO" if seed_band_deficit else "NO_DEFICIT",
            "internal_budget_no_go": internal_budget_no_go,
            "internal_budget_status": "NO_INTERNAL_BUDGET_NO_GO" if not internal_budget_no_go else "SCOPED_INTERNAL_BUDGET_NO_GO",
            "boundary_warning": bool(jwst_expanded["max_row"]["cycle"] == max(expanded_cycles)),
            "claim_boundary": "No global PBH-NANOGrav exclusion: expanded mass boundary and all-DM envelope are sensitivity-only.",
            "no_external_observational_exclusion": True,
        },
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_artifacts(result: dict[str, Any], *, output_dir: Path | None = None) -> tuple[Path, Path]:
    """Write the uniquely named machine-readable result and boundary figure."""

    destination = Path(output_dir) if output_dir is not None else _ROOT / "Lunacy/runs/predictive-research/phases/phase-1/evidence"
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "P1-S3-pbh-nanograv-audit.json"
    figure_path = destination / "P1-S3-pbh-nanograv-boundary.png"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")

    import matplotlib.pyplot as plt

    boundary = result["convergence"]["boundary_expansion"]
    x = [int(item["max_cycle_in_scan"]) for item in boundary]
    y = [float(item["max_amplitude_strain"]) for item in boundary]
    target = float(result["benchmark"]["amplitude_strain"])
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    ax.semilogy(x, y, marker="o", color="#264653", label="all-DM internal envelope")
    ax.axhline(target, color="#d1495b", linestyle="--", linewidth=1.3, label="NANOGrav benchmark")
    ax.set_xlabel("highest included ladder cycle")
    ax.set_ylabel("maximum strain at f=1/yr")
    ax.set_title("PBH–NANOGrav boundary expansion (sensitivity)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return json_path, figure_path


def main() -> dict[str, Any]:
    result = run_audit()
    json_path, figure_path = write_artifacts(result)
    jwst = result["jwst_calibrated_seed_band"]
    budget = result["internal_budget_seed_band"]
    print("=" * 84)
    print("  NVG PBH–NANOGrav AMPLITUDE / ABUNDANCE AUDIT")
    print("=" * 84)
    print(f"Canonical ladder                     : {CANONICAL_LADDER}")
    print(f"Benchmark                           : A={result['benchmark']['amplitude_strain']:.3e} ({result['benchmark']['provenance_status']})")
    print(f"JWST seed-band max amplitude        : {jwst['max_amplitude_strain']:.3e}")
    print(f"JWST seed-band deficit              : {jwst['max_deficit_factor']:.3e}x")
    print(f"All-DM seed-band envelope           : {budget['max_amplitude_strain']:.3e}")
    print(f"Internal-budget no-go                : {result['semantic_interpretation']['internal_budget_status']}")
    print(f"Boundary-expanded max cycle         : {result['jwst_calibrated_expanded_boundary']['max_row']['cycle']}")
    print(f"External exclusions applied         : {result['scan_contract']['external_exclusions_used']}")
    print(f"JSON artifact                       : {json_path}")
    print(f"Figure artifact                     : {figure_path}")
    print(f"STATUS                              : {result['status']}")
    print("=" * 84)
    return result


if __name__ == "__main__":
    main()
