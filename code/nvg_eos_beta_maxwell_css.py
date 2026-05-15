#!/usr/bin/env python3
"""NVG dense matter: Maxwell-matched CSS quark branch on top of the best hadronic baseline.

This is a stricter follow-up to nvg_eos_beta_css_softening.py.
Instead of appending a soft CSS branch directly in P(epsilon), this script:

1. builds the best hadronic baseline from nvg_eos_beta_saturated_vector.py;
2. computes the baryon chemical potential mu_B = (epsilon + P) / n_B;
3. chooses a transition point on the hadronic branch;
4. constructs a constant-sound-speed quark branch in P(mu_B) that matches
   the hadronic branch at the same transition pressure and chemical potential;
5. inserts the resulting first-order Maxwell jump in epsilon and runs a pilot TOV scan.

This is still phenomenological, but it is thermodynamically sharper than the
previous direct CSS appendage because the transition is matched in mu_B and P.
"""

from __future__ import annotations

import math

import numpy as np

import nvg_eos_beta_css_softening as css


def hadronic_mu_b(baseline):
    return (baseline["eps"] + baseline["pressure"]) / baseline["n_b"]


def css_quark_constants(mu_t, p_t, eps_q_t, cs2_q):
    if not (mu_t > 0.0 and p_t > 0.0 and eps_q_t > p_t / cs2_q):
        return None, None
    alpha = (1.0 + cs2_q) / cs2_q
    eps0 = eps_q_t - p_t / cs2_q
    coeff = (p_t + eps0 * cs2_q / (1.0 + cs2_q)) / (mu_t**alpha)
    if not (np.isfinite(coeff) and coeff > 0.0 and np.isfinite(eps0)):
        return None, None
    return coeff, eps0


def build_maxwell_css_eos(baseline, n_trans_ratio, delta_eps_ratio, cs2_q):
    narr = baseline["n_b"]
    eps_h = baseline["eps"]
    p_h = baseline["pressure"]
    mu_h = hadronic_mu_b(baseline)

    n_trans = n_trans_ratio * css.base.n_0
    if not (narr[0] < n_trans < narr[-1]):
        return None
    if not (0.0 < cs2_q <= 1.0):
        return None

    p_t = float(np.interp(n_trans, narr, p_h))
    eps_h_t = float(np.interp(n_trans, narr, eps_h))
    mu_t = float(np.interp(n_trans, narr, mu_h))
    if not (np.isfinite(p_t) and np.isfinite(eps_h_t) and np.isfinite(mu_t) and p_t > 0.0):
        return None

    delta_eps = delta_eps_ratio * eps_h_t
    eps_q_t = eps_h_t + delta_eps

    coeff, eps0 = css_quark_constants(mu_t, p_t, eps_q_t, cs2_q)
    if coeff is None:
        return None

    alpha = (1.0 + cs2_q) / cs2_q
    n_h_t = n_trans
    n_q_t = alpha * coeff * mu_t ** (alpha - 1.0)
    if not (np.isfinite(n_q_t) and n_q_t > n_h_t):
        return None

    p_target = max(p_h[-1] * 2.0, p_t * 25.0)
    mu_max = ((p_target + eps0 * cs2_q / (1.0 + cs2_q)) / coeff) ** (1.0 / alpha)
    if not (np.isfinite(mu_max) and mu_max > mu_t):
        return None

    mu_q = np.geomspace(mu_t * (1.0 + 1.0e-6), mu_max, 160)
    p_q = coeff * mu_q**alpha - eps0 * cs2_q / (1.0 + cs2_q)
    n_q = alpha * coeff * mu_q ** (alpha - 1.0)
    eps_q = mu_q * n_q - p_q

    valid_q = np.isfinite(p_q) & np.isfinite(eps_q) & (p_q > p_t)
    if valid_q.sum() < 30:
        return None
    p_q = p_q[valid_q]
    eps_q = eps_q[valid_q]

    had_mask = p_h < p_t
    if had_mask.sum() < 8:
        return None

    tiny_dp = max(1.0e-6 * p_t, 1.0e-6)
    p_plateau = np.array([p_t, p_t + tiny_dp])
    e_plateau = np.array([eps_h_t, eps_q_t])

    p_full = np.concatenate([p_h[had_mask], p_plateau, p_q])
    e_full = np.concatenate([eps_h[had_mask], e_plateau, eps_q])

    monotonic = np.concatenate([[True], np.diff(p_full) > 0.0])
    p_full = p_full[monotonic]
    e_full = e_full[monotonic]
    if len(p_full) < 20:
        return None

    return {
        "p_sorted": p_full,
        "e_sorted": e_full,
        "n_trans": n_trans,
        "p_trans": p_t,
        "mu_trans": mu_t,
        "eps_h_trans": eps_h_t,
        "eps_q_trans": eps_q_t,
        "delta_eps": delta_eps,
        "n_h_trans": n_h_t,
        "n_q_trans": n_q_t,
        "density_jump": n_q_t / n_h_t,
        "cs2_q": cs2_q,
    }


def evaluate_maxwell_model(baseline, n_trans_ratio, delta_eps_ratio, cs2_q):
    hybrid = build_maxwell_css_eos(baseline, n_trans_ratio, delta_eps_ratio, cs2_q)
    if hybrid is None:
        return None

    masses, radii = css.fast_tov_scan(hybrid["p_sorted"], hybrid["e_sorted"])
    if masses is None:
        return None

    imax = int(np.argmax(masses))
    mmax = float(masses[imax])
    stable = np.arange(imax + 1)
    r14 = None
    if len(stable) > 3 and masses[stable].min() < 1.4 < masses[stable].max():
        try:
            r14 = float(np.interp(1.4, masses[stable], radii[stable]))
        except Exception:
            r14 = None

    return {
        "n_trans_ratio": n_trans_ratio,
        "delta_eps_ratio": delta_eps_ratio,
        "cs2_q": cs2_q,
        "mmax": mmax,
        "r14": r14,
        "mu_trans": hybrid["mu_trans"],
        "p_trans": hybrid["p_trans"],
        "delta_eps": hybrid["delta_eps"],
        "density_jump": hybrid["density_jump"],
        "target_mmax": 2.01 <= mmax <= 2.60,
        "target_r14": r14 is not None and 11.0 <= r14 <= 14.0,
    }


def main():
    print("=" * 88)
    print("NVG BETA-SATURATED + MAXWELL/CSS MATCHING TEST")
    print("best hadronic baseline -> match quark CSS branch in mu_B and P -> pilot TOV scan")
    print("=" * 88)

    baseline_tov = css.baseline_reference()
    if baseline_tov is None:
        print("Could not build the best saturated-vector baseline.")
        return 1

    baseline = css.build_baseline_arrays()
    if baseline is None:
        print("Could not build monotonic hadronic baseline arrays.")
        return 1

    print("Baseline reference")
    print(f"  Baseline Mmax = {baseline_tov['mmax']:.3f} Msun")
    print(f"  Baseline R1.4 = {baseline_tov['r14'] if baseline_tov['r14'] is not None else '--'}")
    print(f"  M_D(n0)/M_N = {baseline_tov['mdirac_n0']/css.base.M_N:.3f}")
    print(f"  Yp(n0) = {baseline_tov['yp_n0']:.3f}")
    print()

    tests = [
        (1.8, 0.60, 1.0 / 3.0),
        (1.8, 1.00, 1.0 / 3.0),
        (2.2, 0.60, 1.0 / 3.0),
        (2.2, 1.00, 1.0 / 3.0),
    ]

    results = []
    for index, (n_trans_ratio, delta_eps_ratio, cs2_q) in enumerate(tests, start=1):
        print(
            f"Running {index}/{len(tests)}: n_t/n0 = {n_trans_ratio:.2f}, "
            f"Delta eps / eps_h,t = {delta_eps_ratio:.2f}, c_q^2 = {cs2_q:.3f}",
            flush=True,
        )
        result = evaluate_maxwell_model(baseline, n_trans_ratio, delta_eps_ratio, cs2_q)
        if result is None:
            print("  -> no valid Maxwell-matched branch", flush=True)
            continue
        results.append(result)
        r14_str = f"{result['r14']:.2f}" if result['r14'] is not None else "--"
        print(
            f"  -> Mmax = {result['mmax']:.3f} Msun, R1.4 = {r14_str}, "
            f"n_q/n_h = {result['density_jump']:.3f}",
            flush=True,
        )

    if not results:
        print("No valid Maxwell/CSS hybrid branches were produced.")
        return 1

    print(f"Scanned Maxwell models: {len(tests)}")
    print(f"Valid Maxwell stellar branches: {len(results)}")
    print()
    print(f"{'n_t/n0':>6} {'dE/e_h':>7} {'c_q^2':>6} {'nq/nh':>7} {'Mmax':>7} {'R1.4':>6} {'M_ok':>5}")
    print("-" * 60)
    for result in results:
        r14_str = f"{result['r14']:.2f}" if result['r14'] is not None else "  -- "
        print(
            f"{result['n_trans_ratio']:6.2f} {result['delta_eps_ratio']:7.2f} {result['cs2_q']:6.3f} "
            f"{result['density_jump']:7.3f} {result['mmax']:7.3f} {r14_str:>6} "
            f"{'yes' if result['target_mmax'] else 'no':>5}"
        )

    viable = [item for item in results if item['target_mmax']]
    if viable:
        best = sorted(viable, key=lambda item: abs(item['mmax'] - 2.15))[0]
        print()
        print("Best Maxwell/CSS candidate:")
        print(f"  n_t/n0 = {best['n_trans_ratio']:.2f}")
        print(f"  Delta eps / eps_h,t = {best['delta_eps_ratio']:.2f}")
        print(f"  c_q^2 = {best['cs2_q']:.3f}")
        print(f"  n_q / n_h = {best['density_jump']:.3f}")
        print(f"  Mmax = {best['mmax']:.3f} Msun")
        print(f"  R1.4 = {best['r14']:.2f} km" if best['r14'] is not None else "  R1.4 = --")
    else:
        best = sorted(results, key=lambda item: item['mmax'])[0]
        print()
        print("No Maxwell/CSS branch reached the target Mmax window yet.")
        print(f"Lowest Mmax found = {best['mmax']:.3f} Msun")
        print(f"  at n_t/n0 = {best['n_trans_ratio']:.2f}, Delta eps / eps_h,t = {best['delta_eps_ratio']:.2f}")
        print(f"  c_q^2 = {best['cs2_q']:.3f}, n_q / n_h = {best['density_jump']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())