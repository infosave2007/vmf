#!/usr/bin/env python3
"""Compute the advanced NVG tests and report only the computed state.

The original version of this script printed a hand-written O--S table whose
labels contradicted the numerical calculation (in particular for ``S8``), and
called the SMBH and rotating-black-hole sections predictions without running
the required calculations.  The functions below keep the useful toy
calculations, but return structured records first; all prose and the final
summary are rendered from those records.  A section that has no independent
calculation is explicitly marked as an estimate or unsupported.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# Constants used by the executable calculations.  The Hubble, lensing, and
# JWST numbers are observational inputs; they are not model outputs.
hbar_c = 197.327
c_cgs = 2.998e10
G_cgs = 6.674e-8
M_Omega_0 = 859.0
MeV_fm3_to_gcm3 = 1.7827e12
eps_max = M_Omega_0**4 / hbar_c**3
rho_c_cgs = eps_max * MeV_fm3_to_gcm3
H_0_planck = 67.4  # km/s/Mpc, Planck observational input
H_0_local = 73.04  # km/s/Mpc, SH0ES observational input
Omega_m = 0.315
Omega_DE = 0.685


def compute_hubble_test() -> dict[str, Any]:
    """Return the Hubble-tension boundary calculation."""

    m_w_mev = M_Omega_0
    # c_cgs is cm/s; convert to fm/s before using hbar*c in MeV fm.
    tau_w = hbar_c / (m_w_mev * c_cgs * 1e13)
    t_bbn = 1.0
    suppression_bbn = float(np.exp(-min(t_bbn / tau_w, 700.0)))
    suppression_rec = 0.0
    delta_h_early = 0.0
    r_s_standard = 147.09
    r_s_nvg = r_s_standard * (1.0 + delta_h_early)
    h0_inferred = H_0_planck * r_s_standard / r_s_nvg
    gap_pct = (H_0_local - H_0_planck) / H_0_planck * 100.0
    return {
        "status": "BOUNDARY: NVG does not address the Hubble tension",
        "m_w_mev": m_w_mev,
        "tau_w_s": tau_w,
        "suppression_bbn": suppression_bbn,
        "suppression_recombination": suppression_rec,
        "delta_h_early": delta_h_early,
        "r_s_mpc": r_s_nvg,
        "h0_inferred": h0_inferred,
        "h0_planck": H_0_planck,
        "h0_local": H_0_local,
        "gap_pct": gap_pct,
    }


def compute_s8_test() -> dict[str, Any]:
    """Integrate the displayed growth approximation and classify its sign.

    ``w_0`` and ``w_a`` are explicit inputs to this toy calculation.  They are
    not independently inferred in this script, so the result is not upgraded
    to a model prediction.
    """

    s8_planck = 0.832
    s8_lensing = 0.776
    w_0 = -0.83
    w_a = -1.05
    n_steps = 1000
    a_arr = np.linspace(0.01, 1.0, n_steps)
    da = float(a_arr[1] - a_arr[0])

    growth_ratio_integral = 0.0
    for a in a_arr:
        omega_m_a_lcdm = Omega_m * a**-3 / (Omega_m * a**-3 + Omega_DE)
        f_lcdm = omega_m_a_lcdm**0.55
        rho_de_ratio = a ** (-3.0 * (1.0 + w_0 + w_a)) * np.exp(
            -3.0 * w_a * (1.0 - a)
        )
        omega_de_a = Omega_DE * rho_de_ratio
        omega_m_a_nvg = Omega_m * a**-3 / (Omega_m * a**-3 + omega_de_a)
        f_nvg = omega_m_a_nvg**0.55
        growth_ratio_integral += float((f_nvg - f_lcdm) * da / a)

    sigma8_ratio = float(np.exp(growth_ratio_integral))
    s8_nvg = float(s8_planck * sigma8_ratio)
    initial_distance = abs(s8_planck - s8_lensing)
    nvg_distance = abs(s8_nvg - s8_lensing)
    if nvg_distance < initial_distance:
        direction = "toward weak-lensing value"
        status = "PARTIAL: calculated shift reduces the lensing distance"
    elif nvg_distance > initial_distance:
        direction = "away from weak-lensing value"
        status = "WORSENS: calculated shift increases the lensing distance"
    else:
        direction = "no change relative to Planck"
        status = "NO SHIFT: calculated value leaves the lensing distance unchanged"

    return {
        "status": status + " (input-dependent w0/wa; not an independent prediction)",
        "s8_planck": s8_planck,
        "s8_lensing": s8_lensing,
        "w_0": w_0,
        "w_a": w_a,
        "sigma8_ratio": sigma8_ratio,
        "s8_nvg": s8_nvg,
        "initial_tension_sigma": initial_distance / 0.017,
        "remaining_tension_sigma": nvg_distance / 0.017,
        "shift_pct": (sigma8_ratio - 1.0) * 100.0,
        "direction": direction,
    }


def compute_smbh_test() -> dict[str, Any]:
    """Compute the toy cyclic-seed table without asserting a SMBH solution."""

    f_growth = 1.35
    f_pbh = 1e-4
    m_genesis = 0.38
    t_edd = 45e6
    t_z6 = 0.9e9
    rows = []
    for n_past in (10, 30, 50, 70, 76):
        m_cycle = m_genesis * f_growth**n_past
        m_pbh = m_cycle * f_pbh
        m_z6 = m_pbh * math.exp(t_z6 / t_edd)
        rows.append(
            {
                "past_cycle": n_past,
                "pbh_seed_msun": m_pbh,
                "accreted_msun": m_z6,
            }
        )
    # There is no independent formation/accretion simulation or likelihood in
    # this file.  The table is therefore an assumption-dependent illustration,
    # not evidence that NVG explains the JWST sources.
    return {
        "status": "ILLUSTRATIVE ONLY: no independent SMBH formation calculation",
        "inputs": {
            "growth_per_cycle": f_growth,
            "pbh_fraction": f_pbh,
            "genesis_mass_msun": m_genesis,
            "eddington_time_years": t_edd,
            "time_to_z6_years": t_z6,
        },
        "rows": rows,
        "observations": {
            "UHZ1": {"mass_msun": 1e7, "source": "JWST literature input"},
            "GN-z11": {"mass_msun": 1e6, "source": "JWST literature input"},
            "CEERS-1019": {"mass_msun": 1e7, "source": "JWST literature input"},
        },
    }


def compute_stochastic_gw_test() -> dict[str, Any]:
    """Compute the simple QCD-bounce frequency and amplitude estimate."""

    r_c_cm = 1.13e5
    h_bounce = c_cgs / r_c_cm
    f_bounce = h_bounce / (2.0 * np.pi)
    t_bounce_ev = 500e6
    t_cmb_ev = 2.35e-4
    z_bounce = t_bounce_ev / t_cmb_ev
    f_today = f_bounce / (1.0 + z_bounce)
    m_planck_hz = 1.85e43
    omega_gw = (h_bounce / m_planck_hz) ** 2
    return {
        "status": "ESTIMATE: amplitude enhancement from a phase transition is uncomputed",
        "r_c_cm": r_c_cm,
        "h_bounce_s_inv": h_bounce,
        "f_bounce_hz": f_bounce,
        "z_bounce": z_bounce,
        "f_today_hz": f_today,
        "omega_gw": omega_gw,
    }


def compute_rotating_bh_test() -> dict[str, Any]:
    """Compute exterior scales; do not claim a rotating solution was solved."""

    m_bh = 65.0
    a_star = 0.7
    rho_c = rho_c_cgs
    l_sq = 2.0 * G_cgs**2 * (m_bh * 1.989e33) ** 2 / (c_cgs**4 * rho_c)
    l_cm = math.sqrt(l_sq)
    r_s = 2.0 * G_cgs * m_bh * 1.989e33 / c_cgs**2
    r_isco_kerr = r_s / 2.0 * (
        3.0 + math.sqrt(3.0) - math.sqrt(3.0 * (3.0 - 2.0 * a_star))
    )
    ratio = l_cm / r_s
    return {
        "status": "UNSUPPORTED: rotating field equations and stability were not solved",
        "mass_msun": m_bh,
        "spin": a_star,
        "rho_c": rho_c,
        "l_cm": l_cm,
        "r_s_cm": r_s,
        "l_over_r_s": ratio,
        "r_isco_over_rg": r_isco_kerr / r_s * 2.0,
        "exterior_correction": ratio**4,
    }


def compute_tests() -> dict[str, dict[str, Any]]:
    """Run O--S and return the complete structured state."""

    return {
        "O": compute_hubble_test(),
        "P": compute_s8_test(),
        "Q": compute_smbh_test(),
        "R": compute_stochastic_gw_test(),
        "S": compute_rotating_bh_test(),
    }


def _print_report(state: dict[str, dict[str, Any]]) -> None:
    """Render CLI output exclusively from ``state``."""

    o = state["O"]
    print("=" * 72)
    print("  NVG: ADVANCED COSMOLOGICAL TESTS (O–S)")
    print("=" * 72)

    print("\n" + "=" * 72)
    print("  O. HUBBLE TENSION: CAN NVG SHIFT H₀?")
    print("=" * 72)
    print(
        f"  W-field mass = {o['m_w_mev']:.1f} MeV; Compton time = "
        f"{o['tau_w_s']:.2e} s"
    )
    print(f"  BBN suppression = {o['suppression_bbn']:.2e}; recombination = {o['suppression_recombination']:.1e}")
    print(f"  δH/H at recombination = {o['delta_h_early']:.1f}; r_s = {o['r_s_mpc']:.4f} Mpc")
    print(f"  Inferred H₀ = {o['h0_inferred']:.1f} km/s/Mpc; Planck input = {o['h0_planck']:.1f}")
    print(f"  SH0ES input = {o['h0_local']:.2f}; gap = {o['gap_pct']:.1f}%")
    print(f"  STATUS: {o['status']}")

    p = state["P"]
    print("\n" + "=" * 72)
    print("  P. S8 TENSION: STRUCTURE GROWTH WITH NVG w(z)")
    print("=" * 72)
    print(f"  Planck S8 = {p['s8_planck']:.3f}; weak lensing S8 = {p['s8_lensing']:.3f}")
    print(f"  Input w₀ = {p['w_0']:.3f}, wₐ = {p['w_a']:.3f}")
    print(f"  Computed σ8(NVG)/σ8(ΛCDM) = {p['sigma8_ratio']:.4f}")
    print(f"  Computed S8 = {p['s8_nvg']:.3f}; signed shift = {p['shift_pct']:+.1f}%")
    print(f"  Direction from computed distances: {p['direction']}")
    print(f"  Tension: {p['initial_tension_sigma']:.1f}σ → {p['remaining_tension_sigma']:.1f}σ")
    print(f"  STATUS: {p['status']}")

    q = state["Q"]
    print("\n" + "=" * 72)
    print("  Q. EARLY SMBHs FROM CYCLIC PBH SEEDS (JWST)")
    print("=" * 72)
    print("  Assumption-dependent seed/accretion illustration:")
    print(f"  {'Past Cycle':>12} | {'PBH seed (M_sun)':>18} | {'Accretion to z=6':>20}")
    print("  " + "-" * 55)
    for row in q["rows"]:
        print(
            f"  {row['past_cycle']:12d} | {row['pbh_seed_msun']:18.2e} | "
            f"{row['accreted_msun']:20.2e}"
        )
    print("  JWST masses are literature inputs, not fitted by this calculation.")
    print(f"  STATUS: {q['status']}")

    r = state["R"]
    print("\n" + "=" * 72)
    print("  R. STOCHASTIC GW BACKGROUND FROM BOUNCE")
    print("=" * 72)
    print(f"  H_bounce = {r['h_bounce_s_inv']:.2e} s⁻¹; f_bounce = {r['f_bounce_hz']:.2e} Hz")
    print(f"  z_bounce = {r['z_bounce']:.1e}; f_today = {r['f_today_hz']:.2e} Hz")
    print(f"  Ω_GW scaling estimate = {r['omega_gw']:.2e}")
    print(f"  STATUS: {r['status']}")

    s = state["S"]
    print("\n" + "=" * 72)
    print("  S. ROTATING REGULAR BLACK HOLES (KERR GENERALIZATION)")
    print("=" * 72)
    print(f"  Exterior scale check: M = {s['mass_msun']:.1f} M_sun, a* = {s['spin']:.2f}")
    print(f"  r_s = {s['r_s_cm']:.2e} cm; l = {s['l_cm']:.2e} cm; l/r_s = {s['l_over_r_s']:.2e}")
    print(f"  Kerr reference ISCO = {s['r_isco_over_rg']:.2f} r_g")
    print(f"  Exterior correction estimate (l/r_s)^4 = {s['exterior_correction']:.2e}")
    print(f"  STATUS: {s['status']}")

    print("\n" + "=" * 72)
    print("  SUMMARY: ADVANCED TESTS O–S")
    print("=" * 72)
    print(f"  O  Hubble tension       — {o['status']}")
    print(f"  P  S8 tension           — {p['status']}")
    print(f"  Q  Early SMBH seeds     — {q['status']}")
    print(f"  R  Stochastic GW        — {r['status']}")
    print(f"  S  Rotating regular BH  — {s['status']}")
    print("=" * 72)


def main() -> dict[str, dict[str, Any]]:
    """Run the calculations and print the generated report."""

    state = compute_tests()
    _print_report(state)
    return state


if __name__ == "__main__":
    main()
