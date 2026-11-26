#!/usr/bin/env python3
"""Bridge 8: Widened zeta_RMF Primary-Gate Closure, Relativistic Hartree-BCS & Cross-Bridge Check.

Cycle-3 integration module completing the unified NVG nuclear-to-gravity chain
with ZERO finite-nucleus refitting across sectors:

1. Stage A (Widened zeta_RMF Saturation Envelope):
   Scans the pre-registered grid:
     zeta_RMF in (0.0165, 0.0175, 0.0185),
     n0       in (0.156, 0.160) fm^-3,
     M*/M_N   in (0.60, 0.61, 0.62),
     L        in (50.0, 56.0, 60.0) MeV,
   under the EXACT pre-registered primary gate from Bridge 6:
     global vacuum stability PASS, causality up to 10 n0 PASS,
     M_max >= 2.08 M_sun, Lambda_1.4 <= 580, R_1.4 <= 13.0 km,
     M*/M_N in [0.59, 0.62],
   selecting the passing point minimizing |j_eff(0.11) - REPO_BLIND_J_MEV|.

2. Stage B1 (Blind 6-Nucleus Spherical Dirac-Hartree on Primary-Gate Winner):
   Evaluates O16, Ca40, Ca48, Zr90, Sn132, and Pb208 with zero refitting.

3. Stage B2 (Self-Consistent Relativistic Hartree-BCS on Semi-Magic Sn Chain):
   Evaluates Sn116, Sn120, Sn124, and closed-shell control Sn132 with fixed
   literature constant-G neutron pairing strength G_n = C_n / A (C_n = 21.0 MeV,
   zero fit to Sn isotopes) above the N=28 core, plus unpaired controls.
   Verifies exact normal-state collapse Delta_n = 0 on doubly-magic Sn132,
   physical pairing gaps and binding-energy gain on open-shell Sn116/120/124,
   and exact matching of E_eig vs E_direct with E_pair = -Delta_n^2 / G_n.

4. Stage C (Cross-Bridge Thermal Schur-Ward & Independent Hartle I-Love-Q):
   Verifies the primary-gate winner's frozen parameters end-to-end against
   Bridge 5's Schur-Hessian / Ward identity at (T=70 MeV, mu_B=875 MeV) and
   Bridge 7's 1st+2nd-order Hartle ODE at PSR J0737-3039A (1.3381 M_sun) and
   canonical 1.4 M_sun.

SPEC-v2 fail-closed audit: evidence_weight = 0.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

for _blas_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "1")

import numpy as np
from scipy.optimize import brentq, root

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_two_condensate_unified_audit as law2  # noqa: E402
import nvg_unified_astrophysics_stability_bridge as b7  # noqa: E402
import nvg_unified_calibration_envelope_probe as b6  # noqa: E402
import nvg_unified_dirac_hartree_probe as dh  # noqa: E402
import nvg_unified_thermal_response_bridge as b5  # noqa: E402

SOURCE_PATH = Path(__file__).resolve()
LAW2_RESULTS_PATH = HERE / "nvg_two_condensate_unified_results.json"
DEFAULT_OUTPUT_PATH = HERE / "nvg_unified_primary_gate_pairing_results.json"

SCHEMA_VERSION = "nvg_unified_primary_gate_pairing_bridge.v1"
BRIDGE_ID = "BRIDGE8_PRIMARY_GATE_AND_RHBCS_PAIRING"
STATUS_PASS = "PASS_PRIMARY_GATE_RHBCS_AND_CROSS_BRIDGE_ZERO_EVIDENCE"
EVIDENCE_WEIGHT = 0.0

SCAN_ZETA_GRID = (0.0165, 0.0175, 0.0185)
SCAN_N0_GRID = (0.156, 0.160)
SCAN_YSTAR_GRID = (0.60, 0.61, 0.62)
SCAN_L_GRID = (50.0, 56.0, 60.0)

PAIRING_C_N_MEV = 21.0
PAIRING_CORE_N = 28

TIN_CHAIN: tuple[tuple[str, int, int], ...] = (
    ("Sn116", 50, 66),
    ("Sn120", 50, 70),
    ("Sn124", 50, 74),
    ("Sn132", 50, 82),
)

# Cited experimental benchmarks: AME2020; Angeli & Marinova ADNDT 99 (2013) 69
EXP_TIN: dict[str, dict[str, float]] = {
    "Sn116": {"B_over_A_MeV": -8.52312, "R_ch_fm": 4.6250},
    "Sn120": {"B_over_A_MeV": -8.50455, "R_ch_fm": 4.6519},
    "Sn124": {"B_over_A_MeV": -8.46742, "R_ch_fm": 4.6735},
    "Sn132": {"B_over_A_MeV": -8.35361, "R_ch_fm": 4.7093},
}


class PrimaryGatePairingError(ValueError):
    """Fail-closed validation or computation error."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False) + "\n"


def _finite_tree(val: Any) -> bool:
    if isinstance(val, bool) or val is None or isinstance(val, (str, int)):
        return True
    if isinstance(val, float):
        return math.isfinite(val)
    if isinstance(val, Mapping):
        return all(_finite_tree(v) for v in val.values())
    if isinstance(val, (list, tuple)):
        return all(_finite_tree(v) for v in val)
    return False


def _to_b5_params(p: law2.ModelParams) -> dict[str, float]:
    """Map ModelParams directly to Bridge 5 parameter dictionary without refitting."""
    return {
        "W0": float(p.W0),
        "lam_W": float(p.lam_W),
        "M_N": float(p.M_N),
        "m_omega": float(p.m_omega),
        "m_rho": float(p.m_rho),
        "m_s": float(p.m_s),
        "n0_fm3": float(p.n0_fm3),
        "hbar_c": float(p.hbar_c),
        "g_W": float(p.g_W),
        "q_phi": float(p.q_phi),
        "m_W": float(p.m_W),
        "g_s_ch": float(p.g_s_ch),
        "g_omega": float(p.g_omega),
        "g_rho": float(p.g_rho),
        "c2": float(p.c2),
        "b3": float(p.b3),
        "b4": float(p.b4),
        "b6": float(p.b6),
        "zeta_rmf": float(p.zeta_rmf),
        "zeta_g": float(p.zeta_g),
        "Lambda_v": float(p.Lambda_v),
    }


def evaluate_widened_stage_a_point(
    n0_fm3: float,
    y_star: float,
    L_target: float,
    zeta_rmf: float,
) -> dict[str, Any]:
    """Evaluate one pre-registered saturation point at arbitrary zeta_RMF with peak-refined TOV."""
    p = b6.calibrate_envelope_point(n0_fm3, y_star, L_target, zeta_rmf=zeta_rmf)
    stab = law2.check_global_vacuum_stability(p)
    so = law2.predict_saturation_and_spin_orbit(p)
    jeff = law2.predict_blind_isovector_jeff(p)

    ns_grid = np.unique(np.concatenate([np.linspace(0.075, 0.25, 10), np.linspace(0.28, 10.0 * n0_fm3, 18)]))
    rows = []
    guess = (p.W_star_n0, p.Phi_s_n0, p.A0_n0)
    for nf in ns_grid:
        row = law2.solve_beta_point(p, float(nf), guess)
        guess = (row["W_MeV"], row["Phi_s_MeV"], row["A0_MeV"])
        rows.append(row)
    core_p = np.array([r["P_MeV_fm3"] for r in rows])
    core_e = np.array([r["eps_MeV_fm3"] for r in rows])
    cs2 = np.gradient(core_p, core_e)
    causal = bool(np.all((cs2 > 0.0) & (cs2 < 1.0)))

    eos = law2.TabulatedCrustCoreEOS(core_p, core_e, crust_mode="dh_sly4")
    pcs = np.geomspace(22.0, min(core_p[-1] * 0.94, 700.0), 16)
    stars = [law2.solve_tov_tidal_star(eos, float(pc), rtol=1.0e-7, atol=1.0e-9, max_step=0.06) for pc in pcs]
    ms = np.array([s[0] for s in stars])
    imax = int(np.argmax(ms))
    if 0 < imax < len(stars) - 1:
        for pcr in np.linspace(pcs[imax - 1], pcs[imax + 1], 5)[1:-1]:
            stars.append(law2.solve_tov_tidal_star(eos, float(pcr), rtol=1.0e-7, atol=1.0e-9, max_step=0.06))
            pcs = np.append(pcs, pcr)
        order = np.argsort(pcs)
        pcs = pcs[order]
        stars = [stars[i] for i in order]
        ms = np.array([s[0] for s in stars])
        imax = int(np.argmax(ms))
    m_max = float(ms[imax])

    ms_s = ms[: imax + 1]
    pcs_s = pcs[: imax + 1]
    idx = int(np.searchsorted(ms_s, 1.4))
    p_lo = float(pcs_s[max(0, idx - 1)])
    p_hi = float(pcs_s[min(len(pcs_s) - 1, idx)])
    pc14 = brentq(
        lambda pc: law2.solve_tov_tidal_star(eos, pc, rtol=1.0e-7, atol=1.0e-9, max_step=0.06)[0] - 1.4,
        p_lo,
        p_hi,
        xtol=1.0e-4,
    )
    _, r14, _, l14 = law2.solve_tov_tidal_star(eos, pc14, rtol=1.0e-8, atol=1.0e-10, max_step=0.05)

    passes_primary = bool(
        stab["no_false_vacuum_or_runaway"]
        and causal
        and m_max >= b6.PRIMARY_GATE["M_max_min_Msun"]
        and l14 <= b6.PRIMARY_GATE["Lambda_1_4_max"]
        and r14 <= b6.PRIMARY_GATE["R_1_4_max_km"]
        and b6.PRIMARY_GATE["M_eff_over_M_N_range"][0] <= y_star <= b6.PRIMARY_GATE["M_eff_over_M_N_range"][1]
    )

    return {
        "zeta_RMF": float(zeta_rmf),
        "n0_fm3": float(n0_fm3),
        "M_eff_over_M_N": float(y_star),
        "L_MeV": float(L_target),
        "b6_MeV_minus2": float(p.b6),
        "global_stability_pass": bool(stab["no_false_vacuum_or_runaway"]),
        "causality_pass": causal,
        "max_cs2_up_to_10n0": float(np.max(cs2)),
        "M_max_Msun": float(m_max),
        "R_1_4_km": float(r14),
        "Lambda_1_4": float(l14),
        "V0_minus_S_MeV": float(so["V0_minus_S_spin_orbit_depth_MeV"]),
        "j_eff_011_MeV": float(jeff["j_eff_011_MeV"]),
        "abs_j_eff_diff_MeV": float(abs(float(jeff["j_eff_011_MeV"]) - b6.REPO_BLIND_J_MEV)),
        "passes_primary_gate": passes_primary,
        "frozen_params": p.to_frozen_dict(),
    }


def _solve_neutron_bcs(
    cands: list[dict[str, Any]],
    n_int: int,
    m_n: float,
    g_n: float,
    *,
    core_n: int = PAIRING_CORE_N,
    lam_guess: float | None = None,
    delta_guess: float = 1.1,
) -> dict[str, Any]:
    """Solve constant-G BCS equations for bound valence neutron states above core_n."""
    bound = [c for c in cands if c["E"] < m_n - 1.0e-5]
    core: list[tuple[dict[str, Any], float]] = []
    valence: list[dict[str, Any]] = []
    cum = 0
    for c in bound:
        cn = cum + c["degeneracy"]
        if cn <= core_n:
            core.append((c, float(c["degeneracy"])))
        elif cum < core_n < cn:
            raise PrimaryGatePairingError(f"Core boundary {core_n} splits orbital kappa={c['kappa']}")
        else:
            valence.append(c)
        cum = cn

    n_val = float(n_int) - sum(occ for _, occ in core)
    max_val_cap = sum(c["degeneracy"] for c in valence)
    if n_val <= 0.0 or n_val > max_val_cap:
        raise PrimaryGatePairingError(f"Valence neutrons {n_val} outside bound window capacity {max_val_cap}")

    # Normal-state Fermi level (Delta = 0)
    rem = n_val
    lam0 = valence[0]["E"]
    for idx, c in enumerate(valence):
        deg = float(c["degeneracy"])
        if rem <= deg + 1.0e-12:
            if abs(rem - deg) < 1.0e-9 and idx + 1 < len(valence):
                lam0 = 0.5 * (c["E"] + valence[idx + 1]["E"])
            else:
                lam0 = c["E"]
            break
        rem -= deg

    omegas = np.array([0.5 * c["degeneracy"] for c in valence], dtype=float)
    es = np.array([c["E"] for c in valence], dtype=float)

    # Normal-state threshold condition: (G/2) * sum Omega_k / |eps_k - lam_0| <= 1 => Delta = 0
    min_dist = float(np.min(np.abs(es - lam0)))
    threshold_ratio: float | None = (
        0.5 * g_n * float(np.sum(omegas / np.abs(es - lam0))) if min_dist > 1.0e-6 else None
    )

    if (threshold_ratio is not None and threshold_ratio <= 1.0) or g_n <= 0.0:
        occ_pairs = list(core)
        rem = n_val
        for c in valence:
            occ = min(float(c["degeneracy"]), max(0.0, rem))
            rem -= occ
            occ_pairs.append((c, occ))
        return {
            "occ_pairs": occ_pairs,
            "Delta_n_MeV": 0.0,
            "lambda_n_MeV": float(lam0 - m_n),
            "E_pair_MeV": 0.0,
            "threshold_ratio": float(threshold_ratio) if threshold_ratio is not None else None,
            "bcs_eq_residual": 0.0,
            "collapsed_to_normal": True,
        }

    lam_init = lam0 if lam_guess is None else float(lam_guess)
    u_init = math.log(max(float(delta_guess), 0.25))

    def _bcs_res(x: np.ndarray) -> list[float]:
        lam = float(x[0])
        d = math.exp(float(x[1]))
        ek = np.sqrt((es - lam) ** 2 + d * d)
        f_n = float(np.sum(omegas * (1.0 - (es - lam) / ek))) - n_val
        f_d = 0.5 * g_n * float(np.sum(omegas / ek)) - 1.0
        return [f_n, f_d]

    sol = root(_bcs_res, np.array([lam_init, u_init], dtype=float), tol=1.0e-13)
    if not sol.success:
        sol = root(_bcs_res, np.array([lam0, math.log(1.0)], dtype=float), tol=1.0e-13)
    if not sol.success:
        raise PrimaryGatePairingError(f"BCS 2D root solver failed: {sol.message}")

    lam_star = float(sol.x[0])
    delta_star = math.exp(float(sol.x[1]))
    res_norm = float(np.max(np.abs(_bcs_res(sol.x))))
    if res_norm > 1.0e-10:
        raise PrimaryGatePairingError(f"BCS 2D residual too large: {res_norm}")

    ek = np.sqrt((es - lam_star) ** 2 + delta_star * delta_star)
    vk2 = 0.5 * (1.0 - (es - lam_star) / ek)
    occ_pairs = list(core) + [(c, float(c["degeneracy"] * vk2[i])) for i, c in enumerate(valence)]

    return {
        "occ_pairs": occ_pairs,
        "Delta_n_MeV": float(delta_star),
        "lambda_n_MeV": float(lam_star - m_n),
        "E_pair_MeV": float(-(delta_star * delta_star) / g_n),
        "threshold_ratio": float(threshold_ratio) if threshold_ratio is not None else None,
        "bcs_eq_residual": res_norm,
        "collapsed_to_normal": False,
    }


def solve_nucleus_rh_bcs(
    Z: int,
    N: int,
    params: Mapping[str, Any],
    *,
    pairing_C_n_MeV: float = PAIRING_C_N_MEV,
    core_n: int = PAIRING_CORE_N,
    R_max: float = 20.0,
    M_field: int = 80,
    n_basis: int = 40,
    n_quad: int = 110,
    tol: float = 1.0e-9,
    max_iter: int = 100,
) -> dict[str, Any]:
    """Self-consistent spherical Relativistic Hartree-BCS solver using Stage-B Dirac-Hartree building blocks."""
    z_int, n_int = int(Z), int(N)
    a_int = z_int + n_int
    p = dh.normalize_params(params)
    g_n = float(pairing_C_n_MeV) / float(a_int) if pairing_C_n_MeV > 0.0 else 0.0

    grid = dh.EvenChebyshevGrid(R_max, m_intervals=M_field)
    dirac = dh.SpectralDiracSolver(R_max, n_basis=n_basis, n_quad=n_quad, mass_ref=p["M_N"])
    interp_q = grid.interp_matrix(dirac.t_q)
    r = grid.r

    r0 = 1.2 * (a_int ** (1.0 / 3.0))
    ws = 1.0 / (1.0 + np.exp((r - r0) / 0.55))
    w = p["W0"] * (1.0 - 0.001 * ws)
    s = (350.0 / p["g_s"]) * ws
    a0 = (280.0 / p["g_omega"]) * ws
    rho0 = -5.0 * ((n_int - z_int) / a_int) * ws
    v_c = np.where(
        r <= r0,
        z_int * dh.ALPHA_EM * dh.HBARC_MEV_FM / (2.0 * r0) * (3.0 - (r / r0) ** 2),
        z_int * dh.ALPHA_EM * dh.HBARC_MEV_FM / np.maximum(r, 1.0e-12),
    )

    kappas = dh._candidate_kappas_for_A(a_int)
    hist_x: list[np.ndarray] = []
    hist_f: list[np.ndarray] = []
    lam_g: float | None = None
    delta_g = 1.1
    converged = False
    scf_err = math.inf
    iterations = 0

    def _eval_step(
        w_f: np.ndarray,
        s_f: np.ndarray,
        a0_f: np.ndarray,
        rho0_f: np.ndarray,
        vc_f: np.ndarray,
        lam_hint: float | None,
        delta_hint: float,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, list[tuple[dict[str, Any], float]]], dict[str, Any], np.ndarray]:
        m_eff = p["M_N"] * (w_f / p["W0"]) - p["g_s"] * s_f
        v_p = p["g_omega"] * a0_f + 0.5 * p["g_rho"] * rho0_f + vc_f
        v_n = p["g_omega"] * a0_f - 0.5 * p["g_rho"] * rho0_f
        m_eff_q = interp_q @ m_eff
        vp_q = interp_q @ v_p
        vn_q = interp_q @ v_n

        dens_out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        occs_out: dict[str, list[tuple[dict[str, Any], float]]] = {}
        bcs_info: dict[str, Any] = {}

        for sp, cnt, vq in (("p", z_int, vp_q), ("n", n_int, vn_q)):
            cands: list[dict[str, Any]] = []
            for kap in kappas:
                vals, vecs, h_mat, phi_e, chi_e, phi0, chi0 = dirac.solve_sector(kap, m_eff_q, vq, grid.t)
                pos_idx = np.where(vals > 0.0)[0]
                for nr, ii in enumerate(pos_idx[:6]):
                    e_val = float(vals[ii])
                    vec = vecs[:, ii]
                    c_l = vec[:n_basis]
                    c_s = vec[n_basis:]
                    eig_res = float(np.linalg.norm(h_mat @ vec - e_val * vec) / max(abs(e_val), 1.0))
                    cands.append(
                        {
                            "E": e_val,
                            "kappa": kap,
                            "n_radial": nr,
                            "degeneracy": 2 * abs(kap),
                            "G": c_l @ phi_e,
                            "F": c_s @ chi_e,
                            "g0": float(c_l @ phi0),
                            "f0": float(c_s @ chi0),
                            "vec": vec,
                            "eig_res": eig_res,
                        }
                    )
            cands.sort(key=lambda item: item["E"])
            if sp == "p" or g_n <= 0.0:
                rem = float(cnt)
                occ_pairs: list[tuple[dict[str, Any], float]] = []
                for c in cands:
                    if rem <= 1.0e-12:
                        break
                    occ = min(float(c["degeneracy"]), rem)
                    rem -= occ
                    occ_pairs.append((c, occ))
                if rem > 1.0e-8:
                    raise PrimaryGatePairingError(f"Insufficient bound {sp} orbitals for (Z={z_int}, N={n_int})")
                if sp == "n":
                    bcs_info = {
                        "Delta_n_MeV": 0.0,
                        "lambda_n_MeV": float(occ_pairs[-1][0]["E"] - p["M_N"]),
                        "E_pair_MeV": 0.0,
                        "threshold_ratio": 0.0,
                        "bcs_eq_residual": 0.0,
                        "collapsed_to_normal": True,
                    }
            else:
                bcs_res = _solve_neutron_bcs(
                    cands,
                    n_int,
                    p["M_N"],
                    g_n,
                    core_n=core_n,
                    lam_guess=lam_hint,
                    delta_guess=delta_hint,
                )
                occ_pairs = bcs_res.pop("occ_pairs")
                bcs_info = bcs_res

            rho_v = np.zeros_like(r)
            rho_sc = np.zeros_like(r)
            for c, occ in occ_pairs:
                if occ <= 0.0:
                    continue
                g_r, f_r = c["G"], c["F"]
                rho_v[1:] += occ * (g_r[1:] ** 2 + f_r[1:] ** 2) / (4.0 * math.pi * r[1:] ** 2)
                rho_sc[1:] += occ * (g_r[1:] ** 2 - f_r[1:] ** 2) / (4.0 * math.pi * r[1:] ** 2)
                rho_v[0] += occ * (c["g0"] ** 2 + c["f0"] ** 2) / (4.0 * math.pi)
                rho_sc[0] += occ * (c["g0"] ** 2 - c["f0"] ** 2) / (4.0 * math.pi)
            dens_out[sp] = (rho_v, rho_sc)
            occs_out[sp] = occ_pairs
        return dens_out, occs_out, bcs_info, m_eff_q

    for it in range(1, max_iter + 1):
        iterations = it
        dens, _, bcs_info, _ = _eval_step(w, s, a0, rho0, v_c, lam_g, delta_g)
        if bcs_info["Delta_n_MeV"] > 0.0:
            lam_g = bcs_info["lambda_n_MeV"] + p["M_N"]
            delta_g = bcs_info["Delta_n_MeV"]
        rho_p, rhos_p = dens["p"]
        rho_n, rhos_n = dens["n"]
        w_new, s_new, a0_new, rho0_new, vc_new = dh._solve_fields_newton(
            grid, p, rhos_p + rhos_n, rho_p, rho_n, w, s, a0, rho0
        )
        x_cur = np.concatenate([w, s, a0, rho0, v_c])
        x_new = np.concatenate([w_new, s_new, a0_new, rho0_new, vc_new])
        scf_err = float(np.max(np.abs(x_new - x_cur)))
        if scf_err < tol:
            w, s, a0, rho0, v_c = w_new, s_new, a0_new, rho0_new, vc_new
            converged = True
            break
        x_next = dh._anderson_mix(x_cur, x_new, hist_x, hist_f, beta=0.40, m_max=6)
        w_t, s_t, a0_t, rho0_t, vc_t = np.split(x_next, 5)
        if np.min(w_t) <= 0.2 * p["W0"] or np.min(p["M_N"] * (w_t / p["W0"]) - p["g_s"] * s_t) <= 50.0:
            x_next = 0.75 * x_cur + 0.25 * x_new
            hist_x.clear()
            hist_f.clear()
            w_t, s_t, a0_t, rho0_t, vc_t = np.split(x_next, 5)
        w, s, a0, rho0, v_c = w_t, s_t, a0_t, rho0_t, vc_t

    if not converged:
        raise PrimaryGatePairingError(f"RH-BCS SCF failed for (Z={z_int}, N={n_int}) err={scf_err:.3e}")

    # Final exact synchronization pass
    dens, occs, bcs_info, m_eff_q = _eval_step(w, s, a0, rho0, v_c, lam_g, delta_g)
    rho_p, rhos_p = dens["p"]
    rho_n, rhos_n = dens["n"]
    rho_s = rhos_p + rhos_n
    rho_b = rho_p + rho_n
    rho_3 = rho_p - rho_n
    fres = dh._field_residuals(grid, p, w, s, a0, rho0, v_c, rho_s, rho_p, rho_n)

    w3 = 4.0 * math.pi * (r * r) * grid.w
    hc3 = dh.HBARC_MEV_FM**3
    q_phi2 = p["q_phi"] ** 2
    d_s = p["d_s"]
    e_pair = float(bcs_info["E_pair_MeV"])

    # Formula 1: weighted sp eigenvalue sum + field rearrangement + E_pair
    eig_sum = sum(occ * c["E"] for sp in ("p", "n") for c, occ in occs[sp] if occ > 0.0)
    e_field_eig = float(
        np.dot(
            w3,
            (
                0.5 * p["g_s"] * s * rho_s
                - (p["b_s"] / 6.0) * (s**3) / hc3
                - (p["c_s"] / 4.0) * (s**4) / hc3
                - (d_s / 3.0) * (s**6) / hc3
                - 0.5 * (p["M_N"] / p["W0"]) * (w - p["W0"]) * rho_s
                - 0.25 * p["lam_W"] * ((w - p["W0"]) ** 3) * (w + p["W0"]) / hc3
                + 0.5 * q_phi2 * w * (w - p["W0"]) * (a0**2) / hc3
                - 0.5 * p["g_omega"] * a0 * rho_b
                + (p["zeta"] / 24.0) * (p["g_omega"] ** 4) * (a0**4) / hc3
                - 0.25 * p["g_rho"] * rho0 * rho_3
                + p["Lambda_v"] * (p["g_omega"] ** 2) * (p["g_rho"] ** 2) * (a0**2) * (rho0**2) / hc3
                - 0.5 * v_c * rho_p
            ),
        )
    )
    e_tot_eig = eig_sum + e_field_eig + e_pair

    # Formula 2: direct Dirac kinetic+mass expectation + explicit T^{00} field functional + E_pair
    e_kin_mass = 0.0
    max_orb_eig_res = 0.0
    for sp in ("p", "n"):
        for c, occ in occs[sp]:
            if occ <= 0.0:
                continue
            q_l, q_s, h_sl, _, _, _, _ = dirac.get_sector(c["kappa"], grid.t)
            c_l = c["vec"][:n_basis]
            c_s = c["vec"][n_basis:]
            max_orb_eig_res = max(max_orb_eig_res, c["eig_res"])
            kin_mass_i = float(
                c_l @ ((q_l.T * m_eff_q[None, :]) @ q_l) @ c_l
                - c_s @ ((q_s.T * m_eff_q[None, :]) @ q_s) @ c_s
                + 2.0 * (c_s @ (h_sl @ c_l))
            )
            e_kin_mass += occ * kin_mass_i

    dw, ds, da0, drho0, dvc = (grid.D1 @ arr for arr in (w, s, a0, rho0, v_c))
    e_field_direct = float(
        np.dot(
            w3,
            (
                0.5 * (dw**2) / dh.HBARC_MEV_FM
                + 0.25 * p["lam_W"] * ((w * w - p["W0"] ** 2) ** 2) / hc3
                + 0.5 * (ds**2) / dh.HBARC_MEV_FM
                + (
                    0.5 * (p["m_s"] ** 2) * (s**2)
                    + (p["b_s"] / 3.0) * (s**3)
                    + 0.25 * p["c_s"] * (s**4)
                    + (d_s / 6.0) * (s**6)
                )
                / hc3
                + 0.5 * (da0**2) / dh.HBARC_MEV_FM
                + (0.5 * q_phi2 * (w**2) * (a0**2) + (p["zeta"] / 8.0) * (p["g_omega"] ** 4) * (a0**4))
                / hc3
                + 0.5 * (drho0**2) / dh.HBARC_MEV_FM
                + (
                    0.5 * (p["m_rho"] ** 2) * (rho0**2)
                    + 3.0 * p["Lambda_v"] * (p["g_omega"] ** 2) * (p["g_rho"] ** 2) * (a0**2) * (rho0**2)
                )
                / hc3
                + (dvc**2) / (8.0 * math.pi * dh.ALPHA_EM * dh.HBARC_MEV_FM)
            ),
        )
    ) + R_max * (v_c[-1] ** 2) / (2.0 * dh.ALPHA_EM * dh.HBARC_MEV_FM)
    e_tot_direct = e_kin_mass + e_field_direct + e_pair

    e_cm = -0.75 * 41.0 * (a_int ** (-1.0 / 3.0))
    ba_eig = (e_tot_eig + e_cm - a_int * p["M_N"]) / a_int
    ba_direct = (e_tot_direct + e_cm - a_int * p["M_N"]) / a_int

    z_int_dens = float(np.dot(w3, rho_p))
    n_int_dens = float(np.dot(w3, rho_n))
    r_p = math.sqrt(float(np.dot(w3, (r * r) * rho_p)) / z_int_dens)
    r_n = math.sqrt(float(np.dot(w3, (r * r) * rho_n)) / n_int_dens)
    r_ch = math.sqrt(r_p * r_p + dh.PROTON_CHARGE_RADIUS_FM**2)

    return {
        "Z": z_int,
        "N": n_int,
        "A": a_int,
        "G_n_MeV": float(g_n),
        "converged": converged,
        "iterations": int(iterations),
        "Delta_n_MeV": float(bcs_info["Delta_n_MeV"]),
        "lambda_n_MeV": float(bcs_info["lambda_n_MeV"]),
        "E_pair_MeV": e_pair,
        "threshold_ratio": (
            float(bcs_info["threshold_ratio"]) if bcs_info["threshold_ratio"] is not None else None
        ),
        "collapsed_to_normal": bool(bcs_info["collapsed_to_normal"]),
        "bcs_eq_residual": float(bcs_info["bcs_eq_residual"]),
        "max_field_residual_rel": float(fres["max_rel"]),
        "max_orbital_eig_residual": float(max_orb_eig_res),
        "Z_rel_err": abs(z_int_dens - z_int) / z_int,
        "N_rel_err": abs(n_int_dens - n_int) / n_int,
        "particle_rel_err": max(abs(z_int_dens - z_int) / z_int, abs(n_int_dens - n_int) / n_int),
        "E_tot_eig_MeV": float(e_tot_eig),
        "E_tot_direct_MeV": float(e_tot_direct),
        "two_energy_formula_diff_MeV": float(abs(e_tot_eig - e_tot_direct)),
        "B_over_A_MeV": float(ba_eig),
        "B_over_A_direct_MeV": float(ba_direct),
        "R_p_fm": float(r_p),
        "R_n_fm": float(r_n),
        "R_ch_fm": float(r_ch),
        "Delta_R_np_fm": float(r_n - r_p),
    }


def _solve_stage_c_cross_bridge(p_law2: law2.ModelParams) -> dict[str, Any]:
    """Verify the Stage-A primary-gate winner directly across Bridge 5 (thermal) and Bridge 7 (Hartle I-Love-Q)."""
    # Thermal Schur-Hessian + Ward check at (T=70 MeV, mu_B=875 MeV, delta=0)
    p_b5 = _to_b5_params(p_law2)
    k_grid = b5._gl_grid()
    st = b5.solve_gc_for_asymmetry(p_b5, 70.0, 875.0, 0.0, k_grid=k_grid)
    sh = b5.schur_hessian_and_entropy(st, p_b5)
    dmu = 0.05
    g0 = (st["W"], st["s"], st["A0"], st["rho0"])
    sp1 = b5.solve_gc_two_species(p_b5, 70.0, st["mu_B"] + dmu, st["mu_D"], guess=g0, k_grid=k_grid)
    sm1 = b5.solve_gc_two_species(p_b5, 70.0, st["mu_B"] - dmu, st["mu_D"], guess=g0, k_grid=k_grid)
    sp2 = b5.solve_gc_two_species(p_b5, 70.0, st["mu_B"] + 2.0 * dmu, st["mu_D"], guess=g0, k_grid=k_grid)
    sm2 = b5.solve_gc_two_species(p_b5, 70.0, st["mu_B"] - 2.0 * dmu, st["mu_D"], guess=g0, k_grid=k_grid)
    d1 = b5.solve_gc_two_species(p_b5, 70.0, st["mu_B"], st["mu_D"] + dmu, guess=g0, k_grid=k_grid)
    dm1 = b5.solve_gc_two_species(p_b5, 70.0, st["mu_B"], st["mu_D"] - dmu, guess=g0, k_grid=k_grid)
    d2 = b5.solve_gc_two_species(p_b5, 70.0, st["mu_B"], st["mu_D"] + 2.0 * dmu, guess=g0, k_grid=k_grid)
    dm2 = b5.solve_gc_two_species(p_b5, 70.0, st["mu_B"], st["mu_D"] - 2.0 * dmu, guess=g0, k_grid=k_grid)
    chi_fd = np.array(
        [
            [
                (-sp2["B_nat"] + 8.0 * sp1["B_nat"] - 8.0 * sm1["B_nat"] + sm2["B_nat"]) / (12.0 * dmu),
                (-d2["B_nat"] + 8.0 * d1["B_nat"] - 8.0 * dm1["B_nat"] + dm2["B_nat"]) / (12.0 * dmu),
            ],
            [
                (-sp2["D_nat"] + 8.0 * sp1["D_nat"] - 8.0 * sm1["D_nat"] + sm2["D_nat"]) / (12.0 * dmu),
                (-d2["D_nat"] + 8.0 * d1["D_nat"] - 8.0 * dm1["D_nat"] + dm2["D_nat"]) / (12.0 * dmu),
            ],
        ],
        dtype=float,
    )
    chi_schur = sh["chi_T_nat"]
    schur_rel_err = float(np.max(np.abs(chi_schur - chi_fd)) / max(1.0, float(np.max(np.abs(chi_schur)))))
    res_qz = b5._finite_q_master_5x5(p_b5, st, 100.0, 1j * 70.0)

    # Independent Hartle 1st+2nd-order I-Love-Q at PSR J0737-3039A (1.3381 M_sun) and 1.4 M_sun
    eos_adapt, eos_dh, _, core_p, _ = b7.build_law2_beta_eos(p_law2)
    pcs = np.geomspace(22.0, min(float(core_p[-1]) * 0.94, 700.0), 16)
    ms = np.array([law2.solve_tov_tidal_star(eos_dh, float(pc), rtol=1.0e-7, atol=1.0e-9, max_step=0.06)[0] for pc in pcs])
    imax = int(np.argmax(ms))
    ms_s = ms[: imax + 1]
    pcs_s = pcs[: imax + 1]
    hartle_rows = []
    for label, m_target in (("PSR_J0737_3039A", 1.3381), ("canonical_1_4", 1.4000)):
        idx = int(np.searchsorted(ms_s, m_target))
        p_lo = float(pcs_s[max(0, idx - 1)])
        p_hi = float(pcs_s[min(len(pcs_s) - 1, idx)])
        pc_star = float(
            brentq(
                lambda pc: law2.solve_tov_tidal_star(eos_dh, pc, rtol=1.0e-8, atol=1.0e-10, max_step=0.05)[0] - m_target,
                p_lo,
                p_hi,
                xtol=1.0e-5,
            )
        )
        star = b7.solve_full_hartle_star(eos_adapt, eos_dh, pc_star)
        star["label"] = label
        hartle_rows.append(star)

    return {
        "thermal_response_T70_mu875": {
            "T_MeV": 70.0,
            "mu_B_MeV": 875.0,
            "n_B_fm3": float(st["B_fm3"]),
            "K_T_MeV_fm3": float(sh["K_T_MeV_fm3"]),
            "K_sigma_MeV_fm3": float(sh["K_sigma_MeV_fm3"]),
            "cs2_adiabatic": float(sh["cs2_adiabatic"]),
            "schur_vs_fd_rel_err": schur_rel_err,
            "ward_max_rel_err": float(res_qz["ward_max_rel_err"]),
            "ph_only_negative_control_ward_rel_err": float(res_qz["ph_only_negative_control_ward_rel_err"]),
            "direct_5x5_vs_schur_rel_err": float(res_qz["direct_vs_schur_rel_err"]),
        },
        "hartle_iloveq_stars": hartle_rows,
    }


def build_result() -> dict[str, Any]:
    law2_sha = _sha256(LAW2_RESULTS_PATH)

    # Stage A: widened zeta_RMF scan
    grid_rows: list[dict[str, Any]] = []
    for zeta in SCAN_ZETA_GRID:
        for n0 in SCAN_N0_GRID:
            for y_star in SCAN_YSTAR_GRID:
                for l_target in SCAN_L_GRID:
                    grid_rows.append(evaluate_widened_stage_a_point(n0, y_star, l_target, zeta))

    primary_survivors = [r for r in grid_rows if r["passes_primary_gate"]]
    if not primary_survivors:
        raise PrimaryGatePairingError("Widened zeta_RMF scan yielded no primary-gate survivors")

    #Deterministic tie-breaking: minimize |j_eff(0.11) - REPO_BLIND_J_MEV|, then maximize M_max
    winner = min(primary_survivors, key=lambda r: (r["abs_j_eff_diff_MeV"], -r["M_max_Msun"]))
    winner_frozen = dict(winner["frozen_params"])

    compact_grid = [{k: v for k, v in r.items() if k != "frozen_params"} for r in grid_rows]
    compact_survivors = [{k: v for k, v in r.items() if k != "frozen_params"} for r in primary_survivors]
    compact_winner = {k: v for k, v in winner.items() if k != "frozen_params"}

    # Stage B1: zero-refit 6 closed-shell nuclei on primary-gate winner
    stage_b1_nuclei = b6.run_six_nuclei_benchmark(winner_frozen, "primary_gate_winner")

    # Stage B2: self-consistent RH-BCS Tin chain (Sn116, Sn120, Sn124, Sn132) + unpaired controls
    tin_rows = []
    max_bcs_eq_res = 0.0
    max_tin_field_res = 0.0
    max_tin_two_e_diff = 0.0
    max_tin_part_err = 0.0
    ba_errs_bcs = []
    rch_rels_bcs = []
    for name, Z, N in TIN_CHAIN:
        bcs_row = solve_nucleus_rh_bcs(Z, N, winner_frozen, pairing_C_n_MeV=PAIRING_C_N_MEV, core_n=PAIRING_CORE_N)
        unp_row = solve_nucleus_rh_bcs(Z, N, winner_frozen, pairing_C_n_MeV=0.0, core_n=PAIRING_CORE_N)
        exp = EXP_TIN[name]
        e_gain = float(bcs_row["E_tot_eig_MeV"] - unp_row["E_tot_eig_MeV"])
        ba_err = float(bcs_row["B_over_A_MeV"] - exp["B_over_A_MeV"])
        rch_rel = float((bcs_row["R_ch_fm"] - exp["R_ch_fm"]) / exp["R_ch_fm"])
        ba_errs_bcs.append(ba_err)
        rch_rels_bcs.append(rch_rel)
        max_bcs_eq_res = max(max_bcs_eq_res, bcs_row["bcs_eq_residual"])
        max_tin_field_res = max(max_tin_field_res, bcs_row["max_field_residual_rel"], unp_row["max_field_residual_rel"])
        max_tin_two_e_diff = max(
            max_tin_two_e_diff, bcs_row["two_energy_formula_diff_MeV"], unp_row["two_energy_formula_diff_MeV"]
        )
        max_tin_part_err = max(max_tin_part_err, bcs_row["particle_rel_err"], unp_row["particle_rel_err"])
        tin_rows.append(
            {
                "nucleus": name,
                "Z": Z,
                "N": N,
                "A": Z + N,
                "bcs": bcs_row,
                "unpaired": unp_row,
                "pairing_energy_gain_MeV": e_gain,
                "exp": exp,
                "B_over_A_err_MeV": ba_err,
                "R_ch_rel_err_percent": 100.0 * rch_rel,
            }
        )

    sn_by_name = {r["nucleus"]: r for r in tin_rows}
    open_tin = [sn_by_name[n] for n in ("Sn116", "Sn120", "Sn124")]

    # Stage C: cross-bridge thermal + Hartle verification on primary-gate winner
    p_winner_law2 = b6.calibrate_envelope_point(
        winner["n0_fm3"], winner["M_eff_over_M_N"], winner["L_MeV"], zeta_rmf=winner["zeta_RMF"]
    )
    stage_c = _solve_stage_c_cross_bridge(p_winner_law2)
    th = stage_c["thermal_response_T70_mu875"]
    max_yy_dev = max(
        max(s["YY_IQ_rel_dev"], s["YY_ILove_rel_dev"], s["YY_QLove_rel_dev"])
        for s in stage_c["hartle_iloveq_stars"]
    )

    s1_sum = stage_b1_nuclei["summary_metrics"]
    gates = {
        "stage_a_primary_gate_satisfied": bool(len(primary_survivors) >= 1 and winner["passes_primary_gate"]),
        "stage_a_all_points_bounded_and_causal": bool(
            all(r["global_stability_pass"] and r["causality_pass"] for r in grid_rows)
        ),
        "stage_b1_six_nuclei_converged_and_accurate": bool(
            all(bool(r["converged"]) for r in stage_b1_nuclei["nuclei"])
            and s1_sum["max_particle_rel_err"] < 1.0e-6
            and s1_sum["max_two_energy_diff_MeV"] < 1.0e-3
            and s1_sum["max_field_residual_rel"] < 1.0e-7
            and s1_sum["all_skins_heavy_positive"]
        ),
        "stage_b2_sn132_natural_normal_collapse": bool(
            sn_by_name["Sn132"]["bcs"]["collapsed_to_normal"] is True
            and sn_by_name["Sn132"]["bcs"]["Delta_n_MeV"] == 0.0
            and abs(sn_by_name["Sn132"]["pairing_energy_gain_MeV"]) < 1.0e-6
        ),
        "stage_b2_open_tin_physical_bcs_gaps_and_gain": bool(
            all(0.4 <= r["bcs"]["Delta_n_MeV"] <= 2.2 for r in open_tin)
            and all(r["pairing_energy_gain_MeV"] < -0.05 for r in open_tin)
            and max_bcs_eq_res < 1.0e-10
            and max_tin_field_res < 1.0e-7
            and max_tin_two_e_diff < 1.0e-3
            and max_tin_part_err < 1.0e-6
        ),
        "stage_c_thermal_schur_ward_passed": bool(
            th["schur_vs_fd_rel_err"] < 1.0e-9
            and th["ward_max_rel_err"] < 1.0e-8
            and th["ph_only_negative_control_ward_rel_err"] > 1.0e-2
            and th["direct_5x5_vs_schur_rel_err"] < 1.0e-12
        ),
        "stage_c_hartle_iloveq_within_1pct": bool(max_yy_dev < 1.0e-2),
    }
    if not all(gates.values()):
        raise PrimaryGatePairingError(f"One or more Bridge 8 gates failed: {gates}")

    return {
        "schema_version": SCHEMA_VERSION,
        "bridge_id": BRIDGE_ID,
        "status": STATUS_PASS,
        "evidence_weight": EVIDENCE_WEIGHT,
        "provenance": {
            "source_script": f"verification/{SOURCE_PATH.name}",
            "source_sha256": _sha256(SOURCE_PATH),
            "law2_results_path": f"verification/{LAW2_RESULTS_PATH.name}",
            "law2_results_sha256": law2_sha,
            "zero_finite_nucleus_fitting": True,
            "fixed_pairing_constant_C_n_MeV": PAIRING_C_N_MEV,
            "pairing_core_N": PAIRING_CORE_N,
        },
        "stage_a_widened_primary_gate": {
            "pre_registered_grid": {
                "zeta_RMF": list(SCAN_ZETA_GRID),
                "n0_fm3": list(SCAN_N0_GRID),
                "M_eff_over_M_N": list(SCAN_YSTAR_GRID),
                "L_MeV": list(SCAN_L_GRID),
                "fixed_B_over_A_MeV": b6.FIXED_BA_MEV,
                "fixed_K_MeV": b6.FIXED_K_MEV,
                "fixed_J_MeV": b6.FIXED_J_MEV,
            },
            "primary_gate": b6.PRIMARY_GATE,
            "primary_gate_satisfied": True,
            "primary_survivor_count": len(primary_survivors),
            "winner": compact_winner,
            "winner_frozen_params": winner_frozen,
            "primary_survivors": compact_survivors,
            "all_grid_rows": compact_grid,
        },
        "stage_b1_blind_six_nuclei_winner": stage_b1_nuclei,
        "stage_b2_tin_chain_rh_bcs": {
            "isotopes": [r[0] for r in TIN_CHAIN],
            "pairing_law": "constant G_n = C_n / A above closed N=28 core",
            "C_n_MeV": PAIRING_C_N_MEV,
            "core_N": PAIRING_CORE_N,
            "rows": tin_rows,
            "summary_metrics": {
                "rms_B_over_A_err_MeV": float(math.sqrt(sum(e * e for e in ba_errs_bcs) / len(ba_errs_bcs))),
                "rms_R_ch_rel_err_percent": float(100.0 * math.sqrt(sum(e * e for e in rch_rels_bcs) / len(rch_rels_bcs))),
                "max_bcs_eq_residual": float(max_bcs_eq_res),
                "max_field_residual_rel": float(max_tin_field_res),
                "max_two_energy_diff_MeV": float(max_tin_two_e_diff),
                "max_particle_rel_err": float(max_tin_part_err),
            },
        },
        "stage_c_cross_bridge_verification": stage_c,
        "gates": gates,
        "honest_caveats": [
            "evidence_weight=0.0: Cycle-3 forward multi-scale unification check (Stage A primary gate + zero-refit nuclei + RH-BCS + thermal/Hartle), not observational likelihood fitting.",
            "Constant-G pairing (C_n = 21.0 MeV above N=28 core) uses a fixed literature scaling law without fitting to Sn116/120/124/132 masses; proton pairing is identically zero due to closed Z=50 shell.",
            "Nuclear calculations remain spherically symmetric closed/semi-magic systems; deformed open-shell nuclei are outside spherical Hartree-BCS scope.",
        ],
    }


def validate_artifact(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"JSON parse error: {exc}"]
    if not _finite_tree(payload):
        errors.append("Non-finite float detected")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("Invalid schema_version")
    if payload.get("bridge_id") != BRIDGE_ID:
        errors.append("Invalid bridge_id")
    if payload.get("status") != STATUS_PASS:
        errors.append("Invalid status")
    if payload.get("evidence_weight") != 0.0:
        errors.append("evidence_weight must be 0.0")
    prov = payload.get("provenance", {})
    if prov.get("source_sha256") != _sha256(SOURCE_PATH):
        errors.append("Source SHA-256 mismatch")
    if prov.get("law2_results_sha256") != _sha256(LAW2_RESULTS_PATH):
        errors.append("Law-2 SHA-256 mismatch")
    gates = payload.get("gates", {})
    if not gates or not all(bool(v) for v in gates.values()):
        errors.append(f"Gate failure: {gates}")
    try:
        sa = payload["stage_a_widened_primary_gate"]
        expected_len = len(SCAN_ZETA_GRID) * len(SCAN_N0_GRID) * len(SCAN_YSTAR_GRID) * len(SCAN_L_GRID)
        if len(sa["all_grid_rows"]) != expected_len:
            errors.append(f"Expected {expected_len} grid rows, got {len(sa['all_grid_rows'])}")
        win = sa["winner"]
        p_check = b6.calibrate_envelope_point(win["n0_fm3"], win["M_eff_over_M_N"], win["L_MeV"], win["zeta_RMF"])
        if abs(p_check.b6 - win["b6_MeV_minus2"]) > 1.0e-18:
            errors.append("Winner b6 mismatch on live recomputation")
    except Exception as exc:
        errors.append(f"Stage-A structure/live mismatch: {exc}")
    return len(errors) == 0, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bridge 8: Widened zeta_RMF Primary-Gate Closure, RH-BCS Pairing & Cross-Bridge Check"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--write", action="store_true", help="Write canonical JSON output to default path")
    parser.add_argument("--validate", nargs="?", const=str(DEFAULT_OUTPUT_PATH), default=None)
    args = parser.parse_args(argv)
    if args.validate is not None:
        ok, errs = validate_artifact(Path(args.validate))
        print(json.dumps({"valid": ok, "errors": errs}, indent=2))
        return 0 if ok else 1
    result = build_result()
    out_text = _canonical_json(result)
    out_path = args.output if args.output is not None else (DEFAULT_OUTPUT_PATH if args.write else None)
    if out_path is not None:
        out_path.write_text(out_text, encoding="utf-8")
    else:
        print(out_text, end="")
    return 0 if result["status"] == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
