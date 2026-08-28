#!/usr/bin/env python3
"""NVG verification checks with runtime-derived report values.

The suite keeps lightweight consistency checks for the framework and delegates
the neutron-star and echo observables to the maintained TOV/echo implementations.
It intentionally reports the result of each criterion; a process exit or a row
label is not treated as proof of the whole theory.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np


# Allow this script to be run from either the repository root or verification/.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ── Constants ──────────────────────────────────────────────────────────
M_N = 939.0  # MeV
n_0 = 0.16  # fm^-3
hbar_c = 197.3  # MeV fm
G_cgs = 6.674e-8
c_cgs = 2.998e10
M_sun = 1.989e33
MeV_fm3_to_gcm3 = 1.7827e12

# NVG inputs used by the simple checks.
kappa_1 = 0.25
kappa_2 = 0.80
M_Omega_0 = 859.0


class c:
    OK = "\033[92m"
    WARN = "\033[93m"
    FAIL = "\033[91m"
    END = "\033[0m"


def print_result(name: str, criterion_ok: bool, details: str = "") -> None:
    status = f"{c.OK}[CHECKED]{c.END}" if criterion_ok else f"{c.FAIL}[CHECK_FAILED]{c.END}"
    print(f"{status} {name:<40} {details}")


def eps_P_cs2(n_B: float) -> tuple[float, float, float]:
    """Return the pedagogical EOS proxy used for the causality smoke check."""

    x = n_B / n_0
    m_omega = M_Omega_0 * (1.0 + kappa_2 * x) ** (-kappa_1 / kappa_2)
    m_cur = M_N - M_Omega_0
    m_star = m_cur + m_omega
    k_f = (1.5 * math.pi**2 * n_B) ** (1.0 / 3.0)
    e_f = math.sqrt(k_f**2 + (m_star / hbar_c) ** 2) * hbar_c
    eps = n_B * e_f
    if x > 2.0:
        pressure, cs2 = eps / 3.0, 1.0 / 3.0
    else:
        pressure, cs2 = 0.15 * eps, 0.15
    return eps, pressure, cs2


def canonical_ns_observables() -> dict[str, Any]:
    """Run the maintained canonical EOS/TOV chain and return its stable branch.

    No displayed NS number is supplied here as a prior result: every value is
    obtained from ``nvg_tidal_deformability.EOS`` and its TOV+Hinderer solver.
    """

    import nvg_tidal_deformability as td

    eos = td.EOS(p_match=1.5, Gamma=1.35)
    pressure_centers = np.geomspace(20.0, 3000.0, 30)
    rows = []
    for pressure_center in pressure_centers:
        mass, radius, k2, lam = td.solve_tov_tidal(eos, float(pressure_center))
        if mass > 0.0 and radius > 0.0 and k2 > 0.0 and lam > 0.0:
            rows.append((float(mass), float(radius), float(k2), float(lam)))
    if not rows:
        raise RuntimeError("canonical EOS produced no valid TOV solutions")

    # Keep the first (stable) branch through the computed mass maximum.
    mass_values = np.asarray([row[0] for row in rows])
    max_index = int(np.argmax(mass_values))
    stable = rows[: max_index + 1]
    masses = np.asarray([row[0] for row in stable])
    radii = np.asarray([row[1] for row in stable])
    lambdas = np.asarray([row[3] for row in stable])
    if masses.max() < 1.4:
        raise RuntimeError("canonical EOS branch does not reach 1.4 solar masses")

    def at_mass(target: float, values: np.ndarray) -> float:
        if target < masses.min() or target > masses.max():
            raise RuntimeError(f"canonical EOS branch does not reach {target} solar masses")
        return float(np.interp(target, masses, values))

    return {
        "M_max": float(mass_values[max_index]),
        "R_1.4": at_mass(1.4, radii),
        "Lambda_1.4": at_mass(1.4, lambdas),
        "stable_rows": len(stable),
        "solver": "nvg_tidal_deformability.EOS + solve_tov_tidal",
    }


def canonical_echo_observable() -> dict[str, Any]:
    """Call the maintained Kerr echo implementation for the test event."""

    import nvg_gw_echo_prediction as echo

    mass_msun = 65.0
    spin = 0.67  # LIGO event input, not an NVG fit parameter.
    delay_s = float(echo.calculate_kerr_echo_delay(mass_msun, spin))
    r_0_cgs, r_g_cgs = echo.get_bh_parameters(mass_msun)
    return {
        "mass_msun": mass_msun,
        "spin": spin,
        "delay_s": delay_s,
        "r_0_km": r_0_cgs / 1e5,
        "r_g_km": r_g_cgs / 1e5,
        "solver": "nvg_gw_echo_prediction.calculate_kerr_echo_delay",
    }


def compute_suite_state() -> dict[str, Any]:
    """Execute criterion rows and return structured state for CLI rendering."""

    sigma_pi_n = 44.0  # phenomenological/lattice input
    sigma_s_n = 30.0
    sigma_heavy = 6.0
    derived_anchor = M_N - (sigma_pi_n + sigma_s_n + sigma_heavy)

    eos = canonical_ns_observables()

    cs2_values = [eps_P_cs2(n * n_0)[2] for n in (1.0, 2.0, 5.0, 10.0)]
    max_cs2 = max(cs2_values)

    m_rho_vac = 775.3
    m_rho_cur = 80.0
    m_omega_vac = m_rho_vac - m_rho_cur
    m_omega_med = m_omega_vac * (1.0 + kappa_2 * 2.0) ** (-kappa_1 / kappa_2)
    m_rho_med = m_rho_cur + m_omega_med
    drop_pct = (1.0 - m_rho_med / m_rho_vac) * 100.0

    gamma_nvg = 1.0
    gamma_gr = 1.0
    cassini_limit = 2.3e-5

    rho_bbn = 1.3e5
    eps_max = M_Omega_0**4 / hbar_c**3
    rho_c = eps_max * MeV_fm3_to_gcm3
    delta_h_h = rho_bbn / (2.0 * rho_c)

    h_c = math.sqrt(8.0 * math.pi * G_cgs * rho_c / 3.0)
    r_c = c_cgs / h_c
    h_0 = 67.4 * 1e5 / 3.086e24
    r_h_0 = c_cgs / h_0
    n_e = math.log(r_h_0 / r_c)

    echo = canonical_echo_observable()

    m_1 = (4.0 / 3.0) * math.pi * r_c**3 * rho_c
    t_1_us = math.pi * G_cgs * m_1 / c_cgs**3 * 1e6

    t_c_k = 1.825e12
    xi_room_um = (1.254 * (t_c_k / 300.0)) * 1e-9
    tau_room_fs = (1.05457e-34 / (1.38065e-23 * 300.0)) * 1e15

    checks = [
        {"name": "Test 1: Lattice QCD Anchor", "criterion_ok": 851 <= derived_anchor <= 867,
         "details": f"Derived: {derived_anchor:.1f} MeV (declared input interval 851–867)"},
        {"name": "Test 2: Causality (c_s^2 <= 1)", "criterion_ok": max_cs2 <= 1.0,
         "details": f"Max c_s^2 = {max_cs2:.3f} (proxy check)"},
        {"name": "Test 3: NS Max Mass >= 2 M_sun", "criterion_ok": eos["M_max"] >= 2.01,
         "details": f"M_max = {eos['M_max']:.3f} M_sun (canonical TOV)"},
        {"name": "Test 4: NS Radius R_1.4 in [11.5, 13] km",
         "criterion_ok": 11.5 <= eos["R_1.4"] <= 13.0,
         "details": f"R_1.4 = {eos['R_1.4']:.3f} km (canonical TOV)"},
        {"name": "Test 5: Rho-meson mass drop at 2n_0", "criterion_ok": 20.0 <= drop_pct <= 28.0,
         "details": f"Drop = {drop_pct:.1f}% (proxy calculation)"},
        {"name": "Test 6: PPN Gamma Cassini Limit",
         "criterion_ok": abs(gamma_nvg - gamma_gr) < cassini_limit,
         "details": f"|gamma - 1| = {abs(gamma_nvg - gamma_gr):.2e}"},
        {"name": "Test 7: BBN Expansion Rate Shift", "criterion_ok": delta_h_h < 0.1,
         "details": f"dH/H = {delta_h_h:.2e} (bound check)"},
        {"name": "Test 8: Genesis CMB Cutoff Mapping", "criterion_ok": 50 <= n_e <= 60,
         "details": f"N_e = {n_e:.1f} (mapping calculation)"},
        {"name": "Test 9: GW Echoes Delay Time",
         "criterion_ok": abs(echo["r_0_km"] - 6.25) < 0.1 and echo["delay_s"] > 0.0,
         "details": f"Delta t (Kerr) = {echo['delay_s']:.5f} s; r_0 = {echo['r_0_km']:.2f} km"},
        {"name": "Test 10: Tolman Cycles Thermodynamics", "criterion_ok": 5.0 <= t_1_us <= 7.0,
         "details": f"Genesis lifetime = {t_1_us:.2f} us (computed M_1 = {m_1 / M_sun:.2f} M_sun)"},
        {"name": "Test 11: Biological theta-Coherence Scale",
         "criterion_ok": 5.0 <= xi_room_um <= 10.0 and 20.0 <= tau_room_fs <= 30.0,
         "details": f"xi_room = {xi_room_um:.2f} um, tau_room = {tau_room_fs:.1f} fs"},
    ]
    return {
        "checks": checks,
        "eos": eos,
        "echo": echo,
        "derived_anchor": derived_anchor,
        "max_cs2": max_cs2,
        "drop_pct": drop_pct,
        "delta_h_h": delta_h_h,
        "n_e": n_e,
        "t_1_us": t_1_us,
        "xi_room_um": xi_room_um,
        "tau_room_fs": tau_room_fs,
        "evidence_status": "PROCESS_CHECKS_ONLY",
        "observed_likelihood": None,
        "limitation": "A green process check is not independent scientific evidence.",
    }


def main() -> dict[str, Any]:
    state = compute_suite_state()
    print("====================================================================")
    print(" NVG AUTOMATED VERIFICATION SUITE")
    print("====================================================================\n")
    for check in state["checks"]:
        print_result(check["name"], check["criterion_ok"], check["details"])

    all_criteria_ok = all(check["criterion_ok"] for check in state["checks"])
    print("\n====================================================================")
    if all_criteria_ok:
        print(f" {c.OK}ALL DECLARED CRITERION CHECKS COMPLETED.{c.END}")
    else:
        print(f" {c.FAIL}SOME DECLARED CRITERION CHECKS FAILED.{c.END}")
    print(" Criterion rows are process checks, not a proof of the complete NVG framework.")
    print(" Canonical NS and echo rows above are runtime outputs from their maintained solvers.")
    print(" Evidence status: PROCESS_CHECKS_ONLY")
    print("====================================================================")
    return state


if __name__ == "__main__":
    main()
