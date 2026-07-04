#!/usr/bin/env python3
"""NVG dense matter: add a CSS softening branch to the best saturated-vector baseline.

This script does not claim a microscopic quark EOS. It tests one explicit
high-density softening mechanism on top of the current best hadronic baseline:

1. keep the n0-calibrated beta-equilibrated hadronic core EOS from
   nvg_eos_beta_saturated_vector.py;
2. choose a transition density n_trans;
3. insert a first-order transition with latent heat Delta epsilon;
4. continue with a constant-sound-speed (CSS) high-density branch.

The question is narrow and falsifiable: can one explicit softening channel move
the best causal core solution toward realistic neutron-star masses/radii?
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import interp1d

import nvg_eos_beta_saturated_vector as base


BEST_BASELINE = {
    "k1": 0.25,
    "k2": 0.80,
    "Cs": 900.0,
    "Crho": 600.0,
    "alpha_v": 4.0,
    "nu_v": 2.0,
}


def baseline_reference():
    return base.run_model(
        BEST_BASELINE["k1"],
        BEST_BASELINE["k2"],
        BEST_BASELINE["Cs"],
        BEST_BASELINE["Crho"],
        BEST_BASELINE["alpha_v"],
        BEST_BASELINE["nu_v"],
        with_tov=True,
    )


def build_baseline_arrays(nn=220):
    c_omega0, state_n0 = base.calibrate_c_omega0(
        BEST_BASELINE["k1"],
        BEST_BASELINE["k2"],
        BEST_BASELINE["Cs"],
        BEST_BASELINE["Crho"],
    )
    if c_omega0 is None:
        return None

    eos = base.build_eos(
        BEST_BASELINE["k1"],
        BEST_BASELINE["k2"],
        BEST_BASELINE["Cs"],
        BEST_BASELINE["Crho"],
        c_omega0,
        BEST_BASELINE["alpha_v"],
        BEST_BASELINE["nu_v"],
        nn=nn,
    )
    if eos is None:
        return None

    narr, eps, pressure, mdirac, yprot, cs2 = eos
    good = (pressure > 0.0) & np.isfinite(pressure) & np.isfinite(eps) & np.isfinite(cs2)
    if good.sum() < 20:
        return None

    narr = narr[good]
    eps = eps[good]
    pressure = pressure[good]
    mdirac = mdirac[good]
    yprot = yprot[good]
    order = np.argsort(narr)
    narr = narr[order]
    eps = eps[order]
    pressure = pressure[order]
    mdirac = mdirac[order]
    yprot = yprot[order]

    # Keep the monotonic pressure envelope in density order; this is the same
    # physical branch we want to hybridize, but without demanding every raw
    # finite-difference point be strictly increasing.
    running_max = -np.inf
    keep = np.zeros_like(pressure, dtype=bool)
    for index, value in enumerate(pressure):
        if value > running_max + 1.0e-8:
            keep[index] = True
            running_max = value

    if keep.sum() < 16:
        return None

    return {
        "n_b": narr[keep],
        "eps": eps[keep],
        "pressure": pressure[keep],
        "mdirac": mdirac[keep],
        "yprot": yprot[keep],
        "state_n0": state_n0,
        "c_omega0": c_omega0,
    }


def build_css_hybrid_eos(baseline, n_trans_ratio, delta_eps_ratio, cs2_q):
    narr = baseline["n_b"]
    eps = baseline["eps"]
    pressure = baseline["pressure"]

    n_trans = n_trans_ratio * base.n_0
    if not (narr[0] < n_trans < narr[-1]):
        return None
    if not (0.0 < cs2_q <= 1.0):
        return None

    p_trans = float(np.interp(n_trans, narr, pressure))
    e_trans = float(np.interp(n_trans, narr, eps))
    if not (np.isfinite(p_trans) and np.isfinite(e_trans) and p_trans > 0.0):
        return None

    delta_eps = delta_eps_ratio * e_trans
    tiny_dp = max(1.0e-6 * p_trans, 1.0e-6)

    had_mask = pressure < p_trans
    if had_mask.sum() < 8:
        return None

    p_had = pressure[had_mask]
    e_had = eps[had_mask]

    p_plateau = np.array([p_trans, p_trans + tiny_dp])
    e_plateau = np.array([e_trans, e_trans + delta_eps])

    # The CSS branch must cover all central pressures reachable in TOV integration.
    # The old cap max(2*p_last, 25*p_trans) ~ 214 MeV/fm^3 truncated the table below
    # massive-star cores; interpolation clamping beyond the edge then made matter
    # artificially incompressible and inflated M_max (2.13/2.25/2.30 were artifacts).
    p_css_max = max(pressure[-1] * 2.0, p_trans * 25.0, 4000.0)
    p_css = np.geomspace(p_trans + 2.0 * tiny_dp, p_css_max, 200)
    e_css = e_trans + delta_eps + (p_css - (p_trans + tiny_dp)) / cs2_q

    p_full = np.concatenate([p_had, p_plateau, p_css])
    e_full = np.concatenate([e_had, e_plateau, e_css])

    monotonic = np.concatenate([[True], np.diff(p_full) > 0.0])
    p_full = p_full[monotonic]
    e_full = e_full[monotonic]
    if len(p_full) < 20:
        return None

    return {
        "p_sorted": p_full,
        "e_sorted": e_full,
        "n_trans": n_trans,
        "p_trans": p_trans,
        "e_trans": e_trans,
        "delta_eps": delta_eps,
        "cs2_q": cs2_q,
    }


def evaluate_hybrid_model(baseline, n_trans_ratio, delta_eps_ratio, cs2_q):
    hybrid = build_css_hybrid_eos(baseline, n_trans_ratio, delta_eps_ratio, cs2_q)
    if hybrid is None:
        return None

    masses, radii = fast_tov_scan(hybrid["p_sorted"], hybrid["e_sorted"])
    if masses is None:
        return None

    imax = int(np.argmax(masses))
    mmax = float(masses[imax])
    stable = np.arange(imax + 1)
    r14 = None
    if len(stable) > 3 and masses[stable].min() < 1.4 < masses[stable].max():
        try:
            r14 = float(interp1d(masses[stable], radii[stable])(1.4))
        except Exception:
            r14 = None

    return {
        "n_trans_ratio": n_trans_ratio,
        "delta_eps_ratio": delta_eps_ratio,
        "cs2_q": cs2_q,
        "mmax": mmax,
        "r14": r14,
        "p_trans": hybrid["p_trans"],
        "e_trans": hybrid["e_trans"],
        "delta_eps": hybrid["delta_eps"],
        "target_mmax": 2.01 <= mmax <= 2.60,
        "target_r14": r14 is not None and 11.0 <= r14 <= 14.0,
    }


def fast_tov_scan(p_sorted, e_sorted):
    eps_of_p = interp1d(
        p_sorted,
        e_sorted,
        bounds_error=False,
        fill_value=(float(e_sorted[0]), float(e_sorted[-1])),
    )
    p_grid = np.geomspace(max(p_sorted[1], 1.0e-4), p_sorted[-1] * 0.85, 10)
    masses = []
    radii = []
    for p_c in p_grid:
        try:
            mass, radius = base.solve_tov(eps_of_p, p_c)
        except Exception:
            continue
        if 0.0 < mass < 5.5 and 5.0 < radius < 30.0:
            masses.append(mass)
            radii.append(radius)
    if len(masses) < 4:
        return None, None
    return np.array(masses), np.array(radii)


def main():
    print("=" * 88)
    print("NVG BETA-SATURATED + CSS SOFTENING TEST")
    print("best hadronic baseline -> explicit high-density softening channel -> TOV scan")
    print("=" * 88)

    baseline_tov = baseline_reference()
    if baseline_tov is None:
        print("Could not build the best saturated-vector baseline.")
        return 1

    baseline = build_baseline_arrays()
    if baseline is None:
        print("Could not build monotonic hadronic EOS arrays from the baseline model.")
        return 1

    print("Baseline reference")
    print(f"  k1 = {BEST_BASELINE['k1']:.2f}, k2 = {BEST_BASELINE['k2']:.2f}")
    print(f"  alpha_v = {BEST_BASELINE['alpha_v']:.1f}, nu_v = {BEST_BASELINE['nu_v']:.1f}")
    print(f"  M_D(n0)/M_N = {baseline_tov['mdirac_n0']/base.M_N:.3f}")
    print(f"  Yp(n0) = {baseline_tov['yp_n0']:.3f}")
    print(f"  P(n0) = {baseline_tov['p_n0']:.3f} MeV/fm^3")
    print(f"  c_s^2,max = {baseline_tov['cs2_max']:.3f}")
    print(f"  Baseline Mmax = {baseline_tov['mmax']:.3f} Msun")
    print(f"  Baseline R1.4 = {baseline_tov['r14'] if baseline_tov['r14'] is not None else '--'}")
    print()

    tests = []
    for n_trans_ratio in [1.8, 2.2]:
        for delta_eps_ratio in [0.60, 1.00]:
            for cs2_q in [1.0 / 3.0]:
                tests.append((n_trans_ratio, delta_eps_ratio, cs2_q))

    results = []
    for index, (n_trans_ratio, delta_eps_ratio, cs2_q) in enumerate(tests, start=1):
        print(
            f"Running {index}/{len(tests)}: n_t/n0 = {n_trans_ratio:.2f}, "
            f"Delta eps / eps_t = {delta_eps_ratio:.2f}, c_q^2 = {cs2_q:.3f}",
            flush=True,
        )
        result = evaluate_hybrid_model(baseline, n_trans_ratio, delta_eps_ratio, cs2_q)
        if result is not None:
            results.append(result)
            r14_str = f"{result['r14']:.2f}" if result['r14'] is not None else "--"
            print(f"  -> Mmax = {result['mmax']:.3f} Msun, R1.4 = {r14_str}", flush=True)
        else:
            print("  -> no valid hybrid branch", flush=True)

    if not results:
        print("No hybrid CSS branches produced valid stellar sequences.")
        return 1

    print(f"Scanned hybrid models: {len(tests)}")
    print(f"Valid hybrid stellar branches: {len(results)}")
    print()
    print(
        f"{'n_t/n0':>6} {'dE/e_t':>7} {'c_q^2':>6} {'Mmax':>7} {'R1.4':>6} {'M_ok':>5} {'R_ok':>5}"
    )
    print("-" * 56)
    for result in results:
        r14_str = f"{result['r14']:.2f}" if result['r14'] is not None else "  -- "
        print(
            f"{result['n_trans_ratio']:6.2f} {result['delta_eps_ratio']:7.2f} {result['cs2_q']:6.3f} "
            f"{result['mmax']:7.3f} {r14_str:>6} {'yes' if result['target_mmax'] else 'no':>5} "
            f"{'yes' if result['target_r14'] else 'no':>5}"
        )

    viable = [item for item in results if item['target_mmax']]
    if viable:
        best = sorted(viable, key=lambda item: (abs(item['mmax'] - 2.15), 0 if item['target_r14'] else 1))[0]
        print()
        print("Best softening candidate (mass-first criterion):")
        print(f"  n_trans / n0 = {best['n_trans_ratio']:.2f}")
        print(f"  Delta eps / eps_t = {best['delta_eps_ratio']:.2f}")
        print(f"  c_q^2 = {best['cs2_q']:.3f}")
        print(f"  Mmax = {best['mmax']:.3f} Msun")
        print(f"  R1.4 = {best['r14']:.2f} km" if best['r14'] is not None else "  R1.4 = --")
        print("Interpretation: one explicit softening channel can move the baseline away from the 4.5 Msun regime.")
    else:
        best = sorted(results, key=lambda item: item['mmax'])[0]
        print()
        print("No hybrid branch reached the target Mmax window yet.")
        print(f"Lowest Mmax found = {best['mmax']:.3f} Msun at n_t/n0 = {best['n_trans_ratio']:.2f}, ")
        print(f"Delta eps / eps_t = {best['delta_eps_ratio']:.2f}, c_q^2 = {best['cs2_q']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())