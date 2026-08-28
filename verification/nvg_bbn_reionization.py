#!/usr/bin/env python3
"""Computed observables G--K with explicit model/input provenance.

The radius-dependent quantities consume the maintained TOV/tidal solver at
runtime.  No presentation-table values are used as computed results.  Where
this entry point has no independent likelihood, it reports an assumption or
model output rather than an observational confirmation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# This grid is part of the canonical NS calculation contract.  The same
# pressure centres are used by ``nvg_tidal_deformability`` and the tracked
# I--Love report generator; keeping it named here makes accidental sibling
# re-gridding visible to callers and semantic tests.
CANONICAL_PRESSURE_GRID = np.logspace(-1.0, 3.4, 120)
CANONICAL_CHAIN_SOURCE = "nvg_tidal_deformability.EOS + solve_tov_tidal"


def compute_eos_chain() -> dict:
    """Run the canonical TOV/tidal chain and return its stable branch.

    The transition point and its in-sample provenance come from the shared
    canonical EOS constructor.  This entry point is a consumer only: it does
    not define a second EOS table or a second pressure grid.
    """
    verification_dir = Path(__file__).resolve().parent
    if str(verification_dir) not in sys.path:
        sys.path.insert(0, str(verification_dir))
    from nvg_tidal_deformability import EOS, solve_tov_tidal

    eos = EOS()
    rows = []
    for pressure in CANONICAL_PRESSURE_GRID:
        mass, radius, k2, tidal_lambda = solve_tov_tidal(eos, float(pressure))
        if mass > 0.5 and radius > 5.0 and k2 > 0.0 and tidal_lambda > 0.0:
            rows.append((float(mass), float(radius), float(k2), float(tidal_lambda)))
    if not rows:
        raise RuntimeError("EOS chain produced no valid TOV states")
    max_index = int(np.argmax([row[0] for row in rows]))
    rows = sorted(rows[: max_index + 1], key=lambda row: row[0])
    masses = np.array([row[0] for row in rows])
    if masses[-1] < 1.4:
        raise RuntimeError("EOS chain does not reach the 1.4 M_sun reference")
    return {
        "rows": rows,
        "masses": masses,
        "radii": np.array([row[1] for row in rows]),
        "lambdas": np.array([row[3] for row in rows]),
        "m_max": float(masses[-1]),
        "pressure_grid": CANONICAL_PRESSURE_GRID.copy(),
        "source": CANONICAL_CHAIN_SOURCE,
        "canonical_selection": eos.canonical_selection,
        "selection_provenance": eos.canonical_provenance,
    }


def radius_at(chain: dict, mass: float) -> float:
    if mass < float(chain["masses"][0]) or mass > float(chain["masses"][-1]):
        raise ValueError(f"mass {mass} outside computed EOS branch")
    return float(np.interp(mass, chain["masses"], chain["radii"]))


def compute_results() -> dict:
    chain = compute_eos_chain()
    G_over_c2 = 1.4766  # km/M_sun

    # G. Universal moment-of-inertia estimate using the computed radius.
    M_A = 1.338  # measured pulsar mass (observational input)
    R_A = radius_at(chain, M_A)
    compactness = G_over_c2 * M_A / R_A
    I_approx = 0.237 * M_A * R_A**2 * (
        1.0 + 4.2 * compactness + 90.0 * compactness**4
    )
    I_cgs = I_approx * 1.989e43

    # H. Surface redshift from the same computed radius branch.
    redshift_rows = []
    for mass in [1.4, 1.8, 2.0, chain["m_max"]]:
        radius = radius_at(chain, float(mass))
        compactness = G_over_c2 * mass / radius
        if 2.0 * compactness >= 1.0:
            raise RuntimeError("computed EOS branch is inside its Schwarzschild radius")
        redshift_rows.append(
            {
                "mass_msun": float(mass),
                "radius_km": radius,
                "z": (1.0 - 2.0 * compactness) ** -0.5 - 1.0,
            }
        )

    # I. Empirical post-merger relation, using R_1.6 from that branch.
    R_14 = radius_at(chain, 1.4)
    R_16 = radius_at(chain, 1.6)
    f_peak = 6.67 - 0.334 * R_16
    f_peak_alt = 1.0 + 0.22 * (14.0 - R_14)

    # J. This simplified symmetry-energy calculation is a declared model
    # assumption, not a calibrated Cas A/Vela fit or independent evidence.
    n_0 = 0.16
    hbar_c = 197.327
    S_0, L_sym = 32.0, 70.0  # illustrative VMF assumptions
    gamma_sym = L_sym / (3.0 * S_0)

    def proton_fraction(n_b: float) -> float:
        e_sym = S_0 * (n_b / n_0) ** gamma_sym
        y_p = 0.04
        for _ in range(100):
            mu_e = 4.0 * e_sym * (1.0 - 2.0 * y_p)
            n_e = max(mu_e, 0.0) ** 3 / (3.0 * np.pi**2 * hbar_c**3)
            y_new = min(0.5, max(0.001, n_e / n_b))
            if abs(y_new - y_p) < 1.0e-12:
                break
            y_p = 0.5 * y_p + 0.5 * y_new
        return float(y_p)

    density_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    yp_rows = [{"n_over_n0": float(x), "y_p": proton_fraction(x * n_0)} for x in density_grid]
    du_threshold = 0.111
    du_onset = next((row["n_over_n0"] for row in yp_rows if row["y_p"] > du_threshold), None)

    # K. Planck values are observational baseline inputs.  The tiny correction
    # is a consistency estimate, not a new cosmological evidence claim.
    rho_bounce = 7.0e16
    rho_rec = 1.0e-21
    delta_h_over_h = rho_rec / rho_bounce
    r_s_standard = 147.09  # Planck 2018 baseline input, Mpc
    r_s_nvg = r_s_standard * (1.0 + delta_h_over_h)

    return {
        "canonical_chain": {
            "source": chain["source"],
            "pressure_grid_points": int(len(chain["pressure_grid"])),
            "selection_provenance": chain["selection_provenance"],
        },
        "eos": {"m_max": chain["m_max"], "r_14": R_14, "r_16": R_16},
        "moment_of_inertia": {
            "mass_msun": M_A,
            "radius_km": R_A,
            "I_cgs": I_cgs,
            "status": "COMPUTED_MODEL_ESTIMATE",
        },
        "redshift": {"rows": redshift_rows, "status": "COMPUTED_MODEL_OUTPUT"},
        "postmerger": {
            "r_16_km": R_16,
            "f_peak_khz": f_peak,
            "f_peak_alt_khz": f_peak_alt,
            "status": "COMPUTED_MODEL_RELATION",
        },
        "direct_urca": {
            "rows": yp_rows,
            "onset_n0": du_onset,
            "status": "ASSUMPTION_SENSITIVITY_NOT_INDEPENDENT_EVIDENCE",
        },
        "sound_horizon": {
            "standard_mpc": r_s_standard,
            "nvg_mpc": r_s_nvg,
            "relative_difference": abs(r_s_nvg - r_s_standard) / r_s_standard,
            "status": "BASELINE_CONSISTENCY_CHECK",
        },
    }


RESULTS = compute_results()


def main() -> None:
    print("=" * 72)
    print("  NVG VERIFICATION: OBSERVABLES G–K (COMPUTED CHAIN)")
    print("=" * 72)
    eos = RESULTS["eos"]
    print(f"EOS chain: M_max={eos['m_max']:.3f} M_sun, R_1.4={eos['r_14']:.3f} km")
    chain_meta = RESULTS["canonical_chain"]
    print(
        f"   source={chain_meta['source']}; pressure_grid_points="
        f"{chain_meta['pressure_grid_points']}; status=CONDITIONAL_IN_SAMPLE"
    )

    moi = RESULTS["moment_of_inertia"]
    print(f"G. I_1.338={moi['I_cgs']:.3e} g cm² ({moi['status']})")
    print("   No precision measurement is included here; this is a model estimate.")

    print("H. Surface redshift from computed EOS branch:")
    for row in RESULTS["redshift"]["rows"]:
        print(f"   M={row['mass_msun']:.3f} M_sun, R={row['radius_km']:.3f} km, z={row['z']:.4f}")

    pm = RESULTS["postmerger"]
    print(f"I. f_peak={pm['f_peak_khz']:.3f} kHz (R_1.6={pm['r_16_km']:.3f} km)")
    print("   Empirical relation output; no detected post-merger datum is fitted.")

    du = RESULTS["direct_urca"]
    print(f"J. Direct-Urca sensitivity onset={du['onset_n0']!r} n_0; status={du['status']}")

    sound = RESULTS["sound_horizon"]
    print(f"K. r_s baseline={sound['standard_mpc']:.2f} Mpc, model estimate={sound['nvg_mpc']:.2f} Mpc")
    print(f"   status={sound['status']}, Δr_s/r_s={sound['relative_difference']:.2e}")
    print("Summary statuses are derived from RESULTS; unsupported observational claims are omitted.")


if __name__ == "__main__":
    main()
