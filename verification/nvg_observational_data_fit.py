#!/usr/bin/env python3
"""Computed observational comparisons with explicit provenance and calibration status.

Historical arrays remain available for reproducibility, but this module does
not turn an embedded fit or hand-set value into an independent prediction.
"""

import json
import math
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import nvg_tidal_deformability_gw170817 as td

PROVENANCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "provenance.json")


def _provenance_entry(name: str) -> dict[str, object]:
    try:
        with open(PROVENANCE_PATH, encoding="utf-8") as fh:
            entries = json.load(fh).get("entries", {})
    except (OSError, ValueError, TypeError):
        return {"trust_status": "unknown_provenance", "source": None, "source_url": None}
    return dict(entries.get(name, {"trust_status": "unknown_provenance", "source": None, "source_url": None}))


def run_cmb_verification() -> dict[str, object]:
    """Fit the illustrative low-l profile; the fitted parameter is calibration."""
    print("Running CMB low-l comparison (calibrated embedded points)...")
    D_l_obs = {2: 221.0, 3: 562.0, 4: 840.0}
    D_l_lcdm = {2: 1150.0, 3: 1010.0, 4: 905.0}
    ratios_obs = {l: D_l_obs[l] / D_l_lcdm[l] for l in [2, 3, 4]}
    errors = {l: math.sqrt(2.0 / (2.0 * l + 1.0)) * ratios_obs[l] for l in [2, 3, 4]}
    l_c_grid = np.linspace(1.5, 6.0, 451)
    best_l_c, min_chi2 = 0.0, float("inf")
    for l_c in l_c_grid:
        predictions = {l: 1.0 - math.exp(-(l / l_c) ** 2) for l in ratios_obs}
        chi2 = sum(((predictions[l] - ratios_obs[l]) / errors[l]) ** 2 for l in ratios_obs)
        if chi2 < min_chi2:
            best_l_c, min_chi2 = float(l_c), float(chi2)
    best_S = {l: 1.0 - math.exp(-(l / best_l_c) ** 2) for l in ratios_obs}
    return {
        "best_l_c": best_l_c, "min_chi2": min_chi2,
        "p_value": math.exp(-min_chi2 / 2.0), "ratios_obs": ratios_obs,
        "errors": errors, "best_S": best_S,
        "fit_status": "calibrated_to_embedded_points",
        "evidence_status": "not_independent_prediction",
        "input_provenance": _provenance_entry("planck2018_tt_full.txt"),
    }


def run_desi_verification() -> dict[str, object]:
    """Compare hand-set CPL values with an unversioned illustrative array."""
    print("Running DESI w(z) comparison (hand-set calibration values)...")
    w0_pred, wa_pred = -0.888, -0.597
    z_bins = np.array([0.15, 0.35, 0.55, 0.75, 1.05, 1.45, 1.95])
    w_desi = np.array([-0.83, -0.95, -1.05, -1.13, -1.18, -1.24, -1.27])
    w_err = np.array([0.08, 0.06, 0.05, 0.05, 0.06, 0.08, 0.11])
    w_nvg = w0_pred + wa_pred * z_bins / (1.0 + z_bins)
    z_scores = np.abs(w_nvg - w_desi) / w_err
    chi2 = float(np.sum(((w_nvg - w_desi) / w_err) ** 2))
    return {
        "z_bins": z_bins, "w_desi": w_desi, "w_err": w_err,
        "w_nvg": w_nvg, "z_scores": z_scores, "chi2": chi2,
        "fit_status": "hand_set_CPL_parameters",
        "evidence_status": "unverified_input_comparison",
        "input_provenance": {"trust_status": "unknown_provenance", "source": None, "source_url": None},
    }


def run_gw_verification() -> dict[str, object]:
    """Generate a tidal-deformability curve from the existing EOS execution."""
    print("Running GW170817 tidal-deformability EOS calculation...")
    eos = td.EOS(p_match=1.5, Gamma=1.35)
    Mc = 1.186
    m1_arr = np.linspace(1.36, 1.60, 25)
    m2_arr = []
    for m1 in m1_arr:
        m2_grid = np.linspace(1.10, 1.36, 1000)
        ch_grid = (m1 * m2_grid) ** 0.6 / (m1 + m2_grid) ** 0.2
        m2_arr.append(float(m2_grid[np.argmin(np.abs(ch_grid - Mc))]))
    P_centers = np.logspace(-1.0, 2.8, 100)
    masses, lambdas = [], []
    for Pc in P_centers:
        M, R, k2, Lam = td.solve_tov_tidal(eos, Pc)
        if M > 0.5 and R > 5.0 and k2 > 0 and Lam > 0:
            masses.append(M); lambdas.append(Lam)
    masses, lambdas = np.asarray(masses), np.asarray(lambdas)
    if not masses.size:
        raise RuntimeError("EOS execution returned no physical tidal points")
    idx_max = int(np.argmax(masses))
    sort_idx = np.argsort(masses[:idx_max + 1])
    masses, lambdas = masses[:idx_max + 1][sort_idx], lambdas[:idx_max + 1][sort_idx]
    if min(float(np.min(m1_arr)), float(np.min(m2_arr))) < float(masses[0]) or max(float(np.max(m1_arr)), float(np.max(m2_arr))) > float(masses[-1]):
        raise ValueError("GW component mass lies outside computed EOS curve; interpolation rejected")
    L1_arr, L2_arr = np.interp(m1_arr, masses, lambdas), np.interp(m2_arr, masses, lambdas)
    Lt_arr = [td.binary_lambda_tilde(m1_arr[i], m2_arr[i], L1_arr[i], L2_arr[i]) for i in range(len(m1_arr))]
    return {
        "m1": m1_arr, "m2": np.asarray(m2_arr), "L1": L1_arr, "L2": L2_arr, "Lt": Lt_arr,
        "fit_status": "computed_EOS_curve", "evidence_status": "comparison_not_evaluated",
    }


def run_cooling_verification() -> dict[str, object]:
    """Retain historical cooling rows as calibration values, not predictions."""
    print("Running cooling comparison (literal calibration values)...")
    return {
        "casa_age": 340.0, "casa_Ts_obs": 2.12e6,
        "casa_dTs_dt_obs": -3500.0, "casa_dTs_dt_nvg": -3650.0,
        "vela_age": 11000.0, "vela_Ts_obs": 6.8e5,
        "vela_Ts_nvg": 6.95e5, "vela_Ts_standard": 1.35e6,
        "fit_status": "literal_calibration_values",
        "evidence_status": "unverified_input_comparison",
        "input_provenance": {"trust_status": "unknown_provenance", "source": None, "source_url": None},
    }


def main() -> None:
    print("=" * 80)
    print("  NVG OBSERVATIONAL DATA: COMPUTED COMPARISONS AND INPUT LEDGER")
    print("=" * 80)
    cmb, desi, gw, cooling = (run_cmb_verification(), run_desi_verification(),
                              run_gw_verification(), run_cooling_verification())
    max_desi_z = float(np.max(desi["z_scores"]))
    max_gw_lambda = float(np.max(gw["Lt"]))
    print(f"CMB fit: chi2={cmb['min_chi2']:.4g}; status={cmb['evidence_status']}")
    print(f"DESI comparison: max |z|={max_desi_z:.3g}; status={desi['evidence_status']}")
    print(f"GW curve: max Lambda_tilde={max_gw_lambda:.4g}; status={gw['evidence_status']}")
    print(f"Cooling rows: status={cooling['evidence_status']}")

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NVG_OBSERVATIONAL_VERIFICATION_RU.md")
    lines = [
        "# NVG/VMF observational-data comparison (audit-safe)", "",
        "These routines execute numerical comparisons using embedded or model-generated arrays. They do not establish independent predictions: several inputs have unknown provenance and the CPL/cooling values are calibration assumptions.", "",
        "| Surface | Computed summary | Evidence status | Input provenance |", "|---|---:|---|---|",
        f"| CMB low-l fit | chi2={cmb['min_chi2']:.6g}, l_c={cmb['best_l_c']:.6g} | {cmb['evidence_status']} | {cmb['input_provenance'].get('trust_status', 'unknown_provenance')} |",
        f"| DESI w(z) | max abs(z)={max_desi_z:.6g}, chi2={desi['chi2']:.6g} | {desi['evidence_status']} | unknown_provenance |",
        f"| GW170817 EOS curve | max Lambda_tilde={max_gw_lambda:.6g} | {gw['evidence_status']} | catalog comparison not loaded |",
        f"| Cas A/Vela cooling | observed/model rows retained | {cooling['evidence_status']} | unknown_provenance |", "",
        "No row is marked within a confidence interval, detectable, viable, excluded, or confirmed by this script. A sourced likelihood and independently versioned data are required before such a claim.",
    ]
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Audit-safe report written to {report_path}")


if __name__ == "__main__":
    main()
