#!/usr/bin/env python3
"""Independent variational audit of the NVG complex-scalar W action.

The maintained article uses one Lagrangian but moves between three inequivalent
homogeneous variational problems.  This module keeps those problems separate:

* action stationarity at fixed phase frequency (the Euler--Lagrange equation),
* minimisation of the physical energy at fixed frequency, and
* the fixed-charge/Routhian problem.

The gauged-current map is the article's declared mean-field map
``A0 = g_omega n_B / m_omega**2`` with the fm^-3 to MeV^3 conversion made
explicit.  No maintained calculator is imported: all quantities below are
re-derived from the declared inputs so that the result can expose a sign or
ensemble mismatch rather than silently inheriting it.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class Inputs:
    n0_fm3: float = 0.16
    hbar_c_MeV_fm: float = 197.327
    M_N_MeV: float = 939.0
    m_pi_MeV: float = 139.57
    f_pi_MeV: float = 92.4
    g_omega: float = 10.12
    m_omega_MeV: float = 782.6
    W_vac_MeV: float = 859.0
    sigma_piN_MeV: float = 44.0
    sigma_sN_MeV: float = 30.0
    G_coupling_MeV_minus2: float = 5.01e-6
    M_q_MeV: float = 335.0
    Lambda_njl_MeV: float = 650.0
    N_f: float = 2.0
    lambda_model: float = 1.05


def calibration(inp: Inputs) -> Dict[str, float]:
    """Recompute all dimensional parameters used by the maintained route."""
    h = inp.hbar_c_MeV_fm
    I1 = math.log(
        (inp.Lambda_njl_MeV
         + math.sqrt(inp.Lambda_njl_MeV**2 + inp.M_q_MeV**2))
        / inp.M_q_MeV
    ) - inp.Lambda_njl_MeV / math.sqrt(
        inp.Lambda_njl_MeV**2 + inp.M_q_MeV**2
    )
    C_scaling = 1.0 / (
        1.0
        - 2.0
        * inp.G_coupling_MeV_minus2
        * inp.N_f
        * inp.M_q_MeV**2
        * I1
        / math.pi**2
    )
    q = inp.g_omega * inp.W_vac_MeV / (2.0 * inp.f_pi_MeV)
    lambda_sigma = (
        inp.M_N_MeV**3
        * inp.m_pi_MeV**2
        * C_scaling
        / (
            4.0
            * inp.f_pi_MeV
            * inp.W_vac_MeV**3
            * (inp.sigma_piN_MeV + inp.sigma_sN_MeV)
        )
    )
    mu_theta = inp.m_omega_MeV
    mu2 = mu_theta**2 + inp.lambda_model * inp.W_vac_MeV**2
    mu = math.sqrt(mu2)
    # n_B is converted to MeV^3 before the Proca mean-field relation.
    A0_per_n0 = (
        inp.g_omega
        * inp.n0_fm3
        * h**3
        / inp.m_omega_MeV**2
    )
    gamma = q * A0_per_n0
    V0 = inp.lambda_model * inp.W_vac_MeV**4 / 4.0
    return {
        "I1": I1,
        "C_scaling": C_scaling,
        "q": q,
        "lambda_sigma": lambda_sigma,
        "lambda_model": inp.lambda_model,
        "mu_theta": mu_theta,
        "mu_squared": mu2,
        "mu": mu,
        "A0_per_n0_MeV": A0_per_n0,
        "gamma_per_n0_MeV": gamma,
        "V0_MeV4": V0,
        "V0_MeV_fm3": V0 / h**3,
        "scalar_mass_MeV": math.sqrt(2.0 * inp.lambda_model) * inp.W_vac_MeV,
    }


def omega_gauged(x_n0: float, inp: Inputs, cal: Dict[str, float]) -> float:
    """Gauge-invariant timelike phase gradient Omega = mu_theta - q A0."""
    return cal["mu_theta"] - cal["gamma_per_n0_MeV"] * x_n0


def potential(W: float, inp: Inputs, cal: Dict[str, float]) -> float:
    """V(W) with the constant chosen so the fixed-frequency vacuum has E=0."""
    return (
        -0.5 * cal["mu_squared"] * W**2
        + 0.25 * inp.lambda_model * W**4
        + cal["V0_MeV4"]
    )


def energy_density(W: float, Omega: float, inp: Inputs, cal: Dict[str, float]) -> float:
    return 0.5 * Omega**2 * W**2 + potential(W, inp, cal)


def pressure(W: float, Omega: float, inp: Inputs, cal: Dict[str, float]) -> float:
    # For (+---), T_ii/a^2 = L = +1/2 W^2 Omega^2 - V for a homogeneous field.
    return 0.5 * Omega**2 * W**2 - potential(W, inp, cal)


def article_section7_energy(
    W: float, Omega: float, inp: Inputs, cal: Dict[str, float]
) -> float:
    """The later article expression labelled ``rho_W`` (MeV^4).

    It is retained only as a comparison surface.  Its quartic shift and
    zero-point subtraction are not the same potential used in the action, and
    differentiating it does not reproduce the article's advertised branch.
    """
    quartic_shift = 0.25 * inp.lambda_model * (W**2 - inp.W_vac_MeV**2) ** 2
    return (
        0.5 * W**2 * Omega**2
        + quartic_shift
        - 0.5 * inp.W_vac_MeV**2 * cal["mu_theta"] ** 2
    )


def action_branch(Omega: float, inp: Inputs, cal: Dict[str, float]) -> Dict[str, Any]:
    """Stationary solution of the Euler--Lagrange action at fixed Omega.

    The action's static effective potential is U_A = V - Omega^2 W^2/2.
    Its non-zero minimum therefore contains ``mu^2 + Omega^2``.  This is a
    dynamical (radial-oscillation) stability statement, not an energy-at-fixed-
    frequency statement.
    """
    lam = inp.lambda_model
    mu2 = cal["mu_squared"]
    W2 = (mu2 + Omega**2) / lam
    curvature = 2.0 * (mu2 + Omega**2)
    return {
        "W_MeV": math.sqrt(W2),
        "W_squared_MeV2": W2,
        "effective_potential_hessian_MeV2": curvature,
        "stable": curvature > 0.0,
        "eom_residual_MeV3": (lam * W2 - mu2 - Omega**2) * math.sqrt(W2),
        "branch": "nonzero_action_stationary",
    }


def fixed_frequency_branch(
    Omega: float, inp: Inputs, cal: Dict[str, float]
) -> Dict[str, Any]:
    """Minimise physical energy at externally fixed phase frequency."""
    lam = inp.lambda_model
    mu2 = cal["mu_squared"]
    delta = mu2 - Omega**2
    # The exact critical root is often represented with a few ulps of roundoff
    # after solving Omega(n); classify that narrow numerical band as critical.
    critical_tol = 1.0e-9 * max(mu2, 1.0)
    if delta > critical_tol:
        W2 = delta / lam
        W = math.sqrt(W2)
        hessian = 2.0 * delta
        branch = "nonzero_energy_minimum"
        stable = hessian > 0.0
        eom_residual = (lam * W2 - mu2 + Omega**2) * W
    elif delta < -critical_tol:
        W2 = 0.0
        W = 0.0
        hessian = -delta
        branch = "melted_zero_energy_minimum"
        stable = hessian > 0.0
        eom_residual = 0.0
    else:
        W2 = 0.0
        W = 0.0
        hessian = 0.0
        branch = "critical_flat_quadratic"
        stable = False
        eom_residual = 0.0
    rho = energy_density(W, Omega, inp, cal)
    p = pressure(W, Omega, inp, cal)
    return {
        "W_MeV": W,
        "W_squared_MeV2": W2,
        "energy_hessian_MeV2": hessian,
        "stable": stable,
        "branch": branch,
        "rho_MeV4": rho,
        "pressure_MeV4": p,
        "SEC_MeV4": rho + 3.0 * p,
        "eom_residual_MeV3": eom_residual,
    }


def fixed_charge_branch(
    charge_ratio: float, inp: Inputs, cal: Dict[str, float]
) -> Dict[str, Any]:
    """Stationary Routhian branch at fixed phase charge density.

    The canonical phase charge is N = W^2 Omega (MeV^3).  ``charge_ratio``
    scales N relative to the maintained vacuum value W_vac^2 mu_theta.  The
    Routhian potential is U_Q = N^2/(2 W^2) + V(W), so N != 0 excludes W=0.
    """
    lam = inp.lambda_model
    mu2 = cal["mu_squared"]
    N = charge_ratio * inp.W_vac_MeV**2 * cal["mu_theta"]

    # U_N'(W)=0 gives lambda*W^6-mu^2*W^4-N^2=0.  With y=W^2 this is
    # the monotone cubic lambda*y^3-mu^2*y^2-N^2=0 above y=mu^2/lambda.
    # A bracketed solve keeps MeV^6 dimensions explicit and avoids a
    # cancellation-prone closed cubic formula.
    def cubic(y: float) -> float:
        return lam * y**3 - mu2 * y**2 - N**2

    lo = mu2 / lam
    if N == 0.0:
        W2 = lo
    else:
        hi = max(2.0 * lo, (N**2 / lam) ** (1.0 / 3.0) + lo)
        while cubic(hi) <= 0.0:
            hi *= 2.0
        for _ in range(256):
            mid = 0.5 * (lo + hi)
            if cubic(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        W2 = 0.5 * (lo + hi)
    W = math.sqrt(W2)
    curvature = 6.0 * lam * W2 - 4.0 * mu2
    Omega = N / W2 if W2 else math.nan
    rho = energy_density(W, Omega, inp, cal)
    p = pressure(W, Omega, inp, cal)
    return {
        "charge_ratio": charge_ratio,
        "canonical_phase_charge_MeV3": N,
        "W_MeV": W,
        "W_squared_MeV2": W2,
        "Omega_MeV": Omega,
        "routhian_hessian_MeV2": curvature,
        "stable": curvature > 0.0,
        "zero_branch_allowed": abs(N) == 0.0,
        "rho_MeV4": rho,
        "pressure_MeV4": p,
        "SEC_MeV4": rho + 3.0 * p,
        "stationarity_residual_MeV3": (
            lam * W**6 - mu2 * W**4 - N**2
        ) / max(W**3, 1.0),
        "stationarity_residual_relative": abs(
            lam * W**6 - mu2 * W**4 - N**2
        ) / max(lam * W**6, mu2 * W**4, N**2, 1.0),
        "stationarity_equation": "lambda*y^3-mu^2*y^2-N^2=0, y=W^2",
    }


def legacy_python_comparison(
    inp: Inputs, cal: Dict[str, float], sample_grid: Iterable[float]
) -> Dict[str, Any]:
    """Reproduce, rather than import, the maintained script's key semantics."""
    h3 = inp.hbar_c_MeV_fm**3
    legacy_rows: List[Dict[str, Any]] = []
    article_rows: List[Dict[str, Any]] = []
    for x in sample_grid:
        A0 = inp.g_omega / inp.m_omega_MeV**2 * (x * inp.n0_fm3) * h3
        Omega = cal["mu_theta"] - cal["q"] * A0
        W2 = max(0.0, cal["mu_squared"] - Omega**2) / inp.lambda_model
        W = math.sqrt(W2)
        # Exact expressions in verification/nvg_vacuum_w_field_derivation.py.
        v_shifted = 0.25 * inp.lambda_model * (W2 - inp.W_vac_MeV**2) ** 2
        rho_legacy = (0.5 * W2 * Omega**2 + v_shifted) / h3
        p_legacy = (-0.5 * W2 * Omega**2 - v_shifted) / h3
        legacy_rows.append(
            {
                "x_n0": x,
                "Omega_MeV": Omega,
                "W_MeV": W,
                "rho_MeV_fm3": rho_legacy,
                "pressure_MeV_fm3": p_legacy,
                "melted": W == 0.0,
            }
        )
        article_rho = article_section7_energy(W, Omega, inp, cal) / h3
        article_rows.append(
            {
                "x_n0": x,
                "rho_section7_MeV_fm3": article_rho,
                "quadratic_approx_MeV_fm3": -(
                    (3.0 * cal["mu_theta"] ** 2
                     - inp.lambda_model * inp.W_vac_MeV**2)
                    / (2.0 * inp.lambda_model)
                    * cal["gamma_per_n0_MeV"] ** 2
                    * (x * inp.n0_fm3 / inp.n0_fm3) ** 2
                    / h3
                ),
            }
        )
    vacuum = legacy_rows[0]
    return {
        "script": "verification/nvg_vacuum_w_field_derivation.py",
        "legacy_vacuum_rho_MeV_fm3": vacuum["rho_MeV_fm3"],
        "legacy_vacuum_pressure_MeV_fm3": vacuum["pressure_MeV_fm3"],
        "legacy_first_melted_grid_x_n0": next(
            (r["x_n0"] for r in legacy_rows if r["melted"]), None
        ),
        "legacy_rows": legacy_rows,
        "article_section7_rows": article_rows,
        "article_section7_stationarity_residual_at_vacuum_MeV3": (
            cal["mu_theta"] ** 2 * inp.W_vac_MeV
        ),
        "vacuum_energy_expected_MeV_fm3": 0.0,
        "canonical_phase_pressure_sign": "+0.5*W^2*Omega^2 - V",
        "maintained_script_phase_pressure_sign": "-0.5*W^2*Omega^2 - V",
    }


def build_audit(inp: Inputs | None = None) -> Dict[str, Any]:
    inp = inp or Inputs()
    cal = calibration(inp)
    h3 = inp.hbar_c_MeV_fm**3
    positive_melting = (cal["mu_theta"] + cal["mu"]) / cal["gamma_per_n0_MeV"]
    negative_density_root = (cal["mu_theta"] - cal["mu"]) / cal["gamma_per_n0_MeV"]
    omega_zero = cal["mu_theta"] / cal["gamma_per_n0_MeV"]
    article_threshold = 2.5
    rho_c_density_article = (
        2.0
        * inp.lambda_model
        * inp.M_N_MeV**2
        * inp.n0_fm3**2
        * h3
        / (
            (3.0 * cal["mu_theta"] ** 2 - inp.lambda_model * inp.W_vac_MeV**2)
            * cal["gamma_per_n0_MeV"] ** 2
        )
    )
    rho_c_cosmology_article = inp.W_vac_MeV**4 / h3
    sample_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0]
    rows: List[Dict[str, Any]] = []
    for x in sample_grid:
        Omega = omega_gauged(x, inp, cal)
        ff = fixed_frequency_branch(Omega, inp, cal)
        ac = action_branch(Omega, inp, cal)
        rows.append(
            {
                "x_n0": x,
                "Omega_MeV": Omega,
                "action": ac,
                "fixed_frequency": ff,
                "fixed_frequency_rho_MeV_fm3": ff["rho_MeV4"] / h3,
                "fixed_frequency_pressure_MeV_fm3": ff["pressure_MeV4"] / h3,
                "fixed_frequency_SEC_MeV_fm3": ff["SEC_MeV4"] / h3,
            }
        )
    charge_rows = [fixed_charge_branch(r, inp, cal) for r in (0.0, 1.0)]
    article_eom_sign = "mu^2 - Omega^2"  # article's algebraic minimum line
    derived_eom_sign = "mu^2 + Omega^2"  # Euler--Lagrange action
    article_pressure_sign = "-phase-term"
    derived_pressure_sign = "+phase-term"
    legacy = legacy_python_comparison(inp, cal, sample_grid)
    # These booleans are calculated from the equations/data above; the status
    # is intentionally fail-closed when a maintained contract disagrees.
    checks = {
        "vacuum_fixed_frequency_anchor": abs(
            rows[0]["fixed_frequency"]["W_MeV"] - inp.W_vac_MeV
        ) < 1e-9,
        "vacuum_fixed_frequency_energy_zero": abs(
            rows[0]["fixed_frequency_rho_MeV_fm3"]
        ) < 1e-9,
        "action_eom_sign_matches_article": derived_eom_sign == article_eom_sign,
        "article_threshold_matches_derived": abs(
            article_threshold - positive_melting
        ) / positive_melting < 1e-6,
        "canonical_pressure_sign_matches_maintained": (
            derived_pressure_sign == article_pressure_sign
        ),
        "fixed_charge_has_no_melting_at_vacuum_charge": all(
            r["stable"] and r["W_MeV"] > 0.0 for r in charge_rows
        ),
        "fixed_charge_cubic_dimensions_consistent": (
            charge_rows[1]["stationarity_equation"]
            == "lambda*y^3-mu^2*y^2-N^2=0, y=W^2"
            and charge_rows[1]["stationarity_residual_relative"] < 1.0e-12
        ),
        "gauged_threshold_is_on_positive_density_branch": positive_melting > 0.0,
        "article_section7_energy_stationary_on_advertised_branch": (
            abs(legacy["article_section7_stationarity_residual_at_vacuum_MeV3"])
            < 1.0e-6
        ),
    }
    incompatible = [name for name, ok in checks.items() if not ok]
    status = "PASS" if not incompatible else "FAIL_CLOSED_INCOMPATIBLE_CONTRACT"
    return {
        "schema": "static_w_branch_audit.v1",
        "status": status,
        "inputs": asdict(inp),
        "conventions": {
            "metric": "+---",
            "field": "Phi = W exp(i theta)/sqrt(2)",
            "Omega": "partial_0 theta - q A_0",
            "potential": "V=-mu^2 W^2/2+lambda W^4/4+V0",
            "V0_choice": "lambda*W_vac^4/4, fixed-frequency vacuum E=0",
            "density_conversion": "n[fm^-3]*(hbar*c)^3 = n[MeV^3]",
            "phase_charge": "N = W^2 Omega (MeV^3), physical gauged charge q*N",
        },
        "calibration": cal,
        "repair": {
            "attempt": "w-fixed-charge-repair-1",
            "fixed_charge_equation": "lambda*y^3-mu^2*y^2-N^2=0, y=W^2",
            "fixed_charge_hessian": "6*lambda*y-4*mu^2",
            "original_final_report_preserved": True,
        },
        "thresholds": {
            "Omega_equals_plus_mu_x_n0": negative_density_root,
            "Omega_equals_zero_x_n0": omega_zero,
            "Omega_equals_minus_mu_positive_x_n0": positive_melting,
            "article_claim_x_n0": article_threshold,
            "article_claim_relative_error": (
                article_threshold - positive_melting
            ) / positive_melting,
            "action_branch_melting": False,
            "fixed_frequency_melting_condition": "abs(Omega) >= mu",
            "article_rho_c_density_MeV_fm3": rho_c_density_article,
            "article_rho_c_cosmology_MeV_fm3": rho_c_cosmology_article,
        },
        "branches_at_declared_grid": rows,
        "fixed_charge_branches": charge_rows,
        "contract_comparison": {
            "article_eom_minimum_sign": article_eom_sign,
            "action_euler_lagrange_sign": derived_eom_sign,
            "article_pressure_phase_sign": article_pressure_sign,
            "canonical_pressure_phase_sign": derived_pressure_sign,
            "article_threshold_x_n0": article_threshold,
            "derived_threshold_x_n0": positive_melting,
            "maintained_script": legacy,
        },
        "checks": checks,
        "incompatible_contracts": incompatible,
    }


def _fmt(x: float, digits: int = 6) -> str:
    return f"{x:.{digits}g}"


def render_report(result: Dict[str, Any]) -> str:
    c = result["calibration"]
    t = result["thresholds"]
    cmp = result["contract_comparison"]
    legacy = cmp["maintained_script"]
    charge = result["fixed_charge_branches"][1]
    failed = ", ".join(result["incompatible_contracts"]) or "none"
    rows = result["branches_at_declared_grid"]
    row_lines = []
    for r in rows:
        ff = r["fixed_frequency"]
        row_lines.append(
            "| {x:.1f} | {om:.3f} | {w:.3f} | {b} | {h:.3g} | {rho:.6g} | {p:.6g} |".format(
                x=r["x_n0"],
                om=r["Omega_MeV"],
                w=ff["W_MeV"],
                b=ff["branch"],
                h=ff["energy_hessian_MeV2"],
                rho=r["fixed_frequency_rho_MeV_fm3"],
                p=r["fixed_frequency_pressure_MeV_fm3"],
            )
        )
    return "\n".join(
        [
            "# W-branch repair-1 audit report",
            "",
            f"Status: **{result['status']}**.",
            "",
            "This report is generated by `verification/static_w_branch_audit.py` "
            "from the declared inputs; no result table is used as an input.",
            "",
            "## Scope and conventions",
            "",
            "The audit uses metric signature (+---), "
            "`Phi=W exp(i theta)/sqrt(2)`, and `Omega=partial_0 theta-q A_0`. "
            "It evaluates action stationarity, fixed-frequency energy minimisation, "
            "the fixed-charge Routhian, and the declared Proca mean-field current map. "
            "For the canonical stress tensor the homogeneous pressure is "
            "`P=+W^2 Omega^2/2-V`; the sign is fixed by (+---).",
            "",
            "## Recomputed calibration",
            "",
            f"`I1={_fmt(c['I1'])}`, `C={_fmt(c['C_scaling'])}`, "
            f"`q={_fmt(c['q'])}`, `lambda_sigma={_fmt(c['lambda_sigma'])}`, "
            f"`lambda_model={_fmt(c['lambda_model'])}`; `mu={_fmt(c['mu'])} MeV`, "
            f"`gamma={_fmt(c['gamma_per_n0_MeV'])} MeV` per `n_B/n0`, "
            f"`V0={_fmt(c['V0_MeV_fm3'])} MeV/fm^3`.",
            "",
            "## Branch and threshold results",
            "",
            "The Euler--Lagrange action branch obeys "
            "`W^2=(mu^2+Omega^2)/lambda`; its radial effective-potential "
            "curvature is positive and it never melts.  Fixed-frequency energy "
            "minimisation instead obeys `W^2=(mu^2-Omega^2)/lambda` while "
            "`abs(Omega)<mu`, followed by a stable W=0 branch.  Fixed charge has "
            "`lambda*y^3-mu^2*y^2-N^2=0` for `y=W^2`, with positive curvature "
            "`6 lambda*y-4 mu^2`; therefore it has no zero-amplitude solution "
            f"for nonzero N.  At charge ratio 1 the repaired solution is "
            f"`W={_fmt(charge['W_MeV'])} MeV`, `Omega={_fmt(charge['Omega_MeV'])} MeV`, "
            f"with Hessian `{_fmt(charge['routhian_hessian_MeV2'])} MeV^2` and "
            f"relative stationarity residual `{_fmt(charge['stationarity_residual_relative'])}`.",
            "",
            f"The gauged map gives `Omega=+mu` at `x={_fmt(t['Omega_equals_plus_mu_x_n0'])}` "
            f"(negative density), `Omega=0` at `x={_fmt(t['Omega_equals_zero_x_n0'])}`, "
            f"and the physical positive-density energy-melting threshold "
            f"`Omega=-mu` at `x={_fmt(t['Omega_equals_minus_mu_positive_x_n0'])}`. "
            f"The maintained article claims `x=2.5`; relative discrepancy is "
            f"`{_fmt(t['article_claim_relative_error'])}`.",
            f"Its separately quoted density scales recompute to "
            f"`rho_c,density={_fmt(t['article_rho_c_density_MeV_fm3'])} MeV/fm^3` "
            f"and `rho_c,cosmology={_fmt(t['article_rho_c_cosmology_MeV_fm3'])} MeV/fm^3`; "
            "these are not the same observable as the fixed-frequency branch root.",
            "",
            "| x=n_B/n0 | Omega (MeV) | W fixed-freq (MeV) | branch | Hessian (MeV²) | rho (MeV/fm³) | P (MeV/fm³) |",
            "|---:|---:|---:|---|---:|---:|---:|",
            *row_lines,
            "",
            "## Maintained-surface reconciliation",
            "",
            f"Article minimum line uses `{cmp['article_eom_minimum_sign']}`, while direct "
            f"Euler--Lagrange variation gives `{cmp['action_euler_lagrange_sign']}`. "
            f"Article pressure uses `{cmp['article_pressure_phase_sign']}`, while the "
            f"canonical stress tensor gives `{cmp['canonical_pressure_phase_sign']}`. "
            f"The maintained Python script reproduces a vacuum `rho={_fmt(legacy['legacy_vacuum_rho_MeV_fm3'])} "
            f"MeV/fm^3` and `P={_fmt(legacy['legacy_vacuum_pressure_MeV_fm3'])} MeV/fm^3` "
            "despite the declared zero-vacuum offset, because it adds the phase term "
            "to a shifted quartic and flips the pressure sign.  Its first melted grid "
            f"row is `x={legacy['legacy_first_melted_grid_x_n0']}`, whereas the derived "
            f"root is `x={_fmt(t['Omega_equals_minus_mu_positive_x_n0'])}`.  The article's "
            f"Section-7 energy expression has a non-zero vacuum stationarity residual "
            f"`{_fmt(legacy['article_section7_stationarity_residual_at_vacuum_MeV3'])} MeV^3` "
            "on the advertised branch, so its zero-point subtraction cannot be used "
            "as an independent variational potential.",
            "",
            "The original run report is preserved at `Lunacy/runs/deep-physics-audit/REPORT-w-branch.md`. "
            "This repair only changes the fixed-charge solver/result/report; the "
            "independent action/pressure/threshold conflicts remain fail-closed.",
            "",
            "## Fail-closed boundary",
            "",
            f"Failed checks: `{failed}`.  The result is not a validation of the "
            "maintained bounce claim: the action, fixed-frequency, and fixed-charge "
            "ensembles are different contracts, and the article/script mix them. "
            "A positive-frequency-energy branch can be used only with its stated "
            "external-frequency ensemble; it cannot be presented as an Euler--Lagrange "
            "or fixed-charge result.",
            "",
            "Source surfaces inspected: `article/NVG_VACUUM_W_FIELD_DERIVATION_RU.md` "
            "(Sections 1--7) and `verification/nvg_vacuum_w_field_derivation.py`.",
            "",
        ]
    )


def write_outputs(
    output_path: Path | None = None,
    report_path: Path | None = None,
    control_report_path: Path | None = None,
) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output_path = output_path or root / "verification/static_w_branch_audit_results.json"
    report_path = report_path or root / "verification/static_w_branch_audit_repair_1_report.md"
    control_report_path = control_report_path or root / "Lunacy/runs/deep-physics-audit/REPORT-w-branch-repair-1.md"
    result = build_audit()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    control_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(render_report(result), encoding="utf-8")
    if control_report_path != report_path:
        control_report_path.write_text(render_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--control-report", type=Path, default=None)
    args = parser.parse_args()
    result = write_outputs(args.output, args.report, args.control_report)
    print(json.dumps({"status": result["status"], "failed_checks": result["incompatible_contracts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
