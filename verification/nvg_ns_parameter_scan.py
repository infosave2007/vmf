#!/usr/bin/env python3
"""
NVG Verification: NS-sector transition-parameter scan (provenance of the canon)
================================================================================
Reproducible scan that selects the canonical CSS transition parameters used by
nvg_tidal_deformability.py, and documents WHY the previous parameterization was
abandoned.

Historical audit reference (2026-08-27; dated non-evidence retained only to
explain the repair, not as a current prediction):
  - The repo previously used (n_trans = 1.8 n_0, delta_eps = 0.4*eps_t) and
    advertised M_max = 2.25, R_1.4 = 12.0, Lambda_1.4 = 177. Those numbers were
    artifacts: the EOS table ended at P ~ 214 MeV/fm^3 and interpolation
    clamping beyond the edge made matter artificially incompressible; the TOV
    scan also stopped before the mass-curve turnover. With the table extended
    analytically (the CSS phase P = P_t + (eps - eps_t)/3 is exact), the TRUE
    maximum mass of that parameterization is 1.79 M_sun — falsified by
    PSR J0740+6620 (2.08 +/- 0.07 M_sun).
  - This scan finds the surviving region using the honest criteria below and
    selects the point with the largest margin to the nearest constraint edge.

Criteria (all simultaneous):
  - M_max >= 2.01 M_sun            (PSR J0740+6620: 2.08 +/- 0.07, 1-sigma low)
  - binary Lambda-tilde <= 720     (GW170817 90% CI upper bound, low-spin;
                                    computed for 1.36+1.36 and 1.46+1.27,
                                    NOT the weaker Lambda_1.4 proxy)
  - 11.2 <= R_1.4 <= 13.2 km       (NICER J0030 ~2-sigma band)

The canonical point is represented by the machine-readable selection record
below and is re-evaluated by this scan.  Its transition parameters are
conditional/in-sample inputs; all observables printed by the scan are runtime
outputs, not static claims or independent confirmations.

Runtime: well under a minute (grid of TOV + tidal integrations).
"""

from __future__ import annotations
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import nvg_eos_beta_css_softening as soft
from nvg_ns_canonical import (
    CANONICAL_PARAMETERS,
    canonical_selection,
)

# Compatibility tuple for older callers; the canonical record below is the
# authoritative machine-readable source used by both tidal entry points.
CANON = (
    CANONICAL_PARAMETERS["n_trans_ratio"],
    CANONICAL_PARAMETERS["delta_eps_ratio"],
)
OLD = (1.8, 0.4)
GRID_NTR = (1.8, 2.0, 2.2)
GRID_DE = (0.0, 0.05, 0.10, 0.15, 0.20)

CANONICAL_SELECTION = canonical_selection()


def star_family(hyb):
    """M_max, R_1.4, Lambda_1.4, Ltilde_sym, Ltilde_asym for a hybrid EOS."""
    import nvg_tidal_deformability as td

    p, e = hyb["p_sorted"], hyb["e_sorted"]
    if p[-1] < 3500.0:  # safety: analytic CSS continuation
        p_ext = np.geomspace(p[-1] * 1.001, 4000.0, 160)
        e_ext = e[-1] + (p_ext - p[-1]) * 3.0
        p, e = np.concatenate([p, p_ext]), np.concatenate([e, e_ext])
    # Candidate EOS arrays must satisfy the same strict contract as the
    # canonical constructor.  In particular, solve_tov_tidal calls
    # ``get_eps`` before integration and the pressure-table lower/upper bounds
    # are required attributes; do not bypass them with a fallback default.
    eos = td.EOS.__new__(td.EOS)
    eos.p_arr = np.asarray(p, dtype=float)
    eos.eps_arr = np.asarray(e, dtype=float)
    if eos.p_arr.ndim != 1 or eos.eps_arr.ndim != 1 or len(eos.p_arr) != len(eos.eps_arr):
        raise ValueError("hybrid EOS pressure/energy arrays must be one-dimensional and paired")
    if len(eos.p_arr) < 2 or not np.all(np.isfinite(eos.p_arr)) or not np.all(np.isfinite(eos.eps_arr)):
        raise ValueError("hybrid EOS arrays must contain at least two finite points")
    if np.any(np.diff(eos.p_arr) <= 0.0):
        raise ValueError("hybrid EOS pressure grid must be strictly increasing")
    eos.table_pressure_min = float(eos.p_arr[0])
    eos.pressure_min = 0.0
    eos.pressure_max = float(eos.p_arr[-1])
    eos.p_match, eos.Gamma = 1.5, 1.35
    if not eos.table_pressure_min <= eos.p_match <= eos.pressure_max:
        raise ValueError("hybrid EOS matching pressure lies outside its table domain")
    eos.eps_match = float(np.interp(eos.p_match, eos.p_arr, eos.eps_arr))
    if not np.isfinite(eos.eps_match) or eos.eps_match <= 0.0:
        raise ValueError("hybrid EOS matching energy must be finite and positive")
    rows = []
    # Match the canonical tidal entry point's pressure grid so the selected
    # point and downstream runtime chain use the same integration resolution.
    for pc in np.logspace(-1.0, 3.4, 120):
        try:
            row = td.solve_tov_tidal(eos, float(pc))
        except (FloatingPointError, OverflowError, ValueError):
            continue
        if all(np.isfinite(value) for value in row):
            rows.append(row)
    if not rows:
        return (float("nan"),) * 5
    ms = np.array([r[0] for r in rows])
    i = int(np.argmax(ms))
    mm = ms[: i + 1]
    rr = np.array([r[1] for r in rows])[: i + 1]
    ll = np.array([r[3] for r in rows])[: i + 1]

    def at(m, arr):
        if not np.all(np.isfinite(arr)) or mm.max() < m or mm.min() > m:
            return float("nan")
        return float(np.interp(m, mm, arr))

    lt_sym = at(1.36, ll)
    lt_asym = td.binary_lambda_tilde(1.46, 1.27, at(1.46, ll), at(1.27, ll))
    return float(ms[i]), at(1.4, rr), at(1.4, ll), lt_sym, lt_asym


def margin(mmax, r14, lts, lta):
    """Distance to the nearest constraint edge (rough sigma units)."""
    if not all(np.isfinite(value) for value in (mmax, r14, lts, lta)):
        return float("-inf")
    return min((mmax - 2.01) / 0.07, (720.0 - max(lts, lta)) / 200.0,
               (13.2 - r14) / 0.5, (r14 - 11.2) / 0.5)


def select_canonical_point(results):
    """Select the best-margin survivor and attach explicit provenance.

    ``results`` is the runtime output of this scan.  No downstream consumer is
    allowed to treat the selected point as an independent hold-out test.
    """

    survivors = {key: value for key, value in results.items() if value[5]}
    best = max(survivors, key=lambda key: survivors[key][6]) if survivors else None
    selected = canonical_selection()
    selected["selected_point"] = (
        {
            "n_trans_ratio": best[0],
            "delta_eps_ratio": best[1],
            "cs2_q": CANONICAL_PARAMETERS["cs2_q"],
        }
        if best is not None
        else None
    )
    selected["selected_margin"] = float(survivors[best][6]) if best is not None else None
    selected["survivor_count"] = len(survivors)
    selected["evaluated_count"] = len(results)
    if selected["selected_point"] is not None:
        selected["parameters"] = dict(selected["selected_point"])
    selected["provenance"]["status"] = "CONDITIONAL_IN_SAMPLE"
    selected["provenance"]["independent"] = False
    return best, selected


def main():
    print("=" * 84)
    print("  NVG NS-SECTOR TRANSITION SCAN (provenance of the canonical parameters)")
    print("=" * 84)
    baseline = soft.build_baseline_arrays()
    assert baseline is not None, "baseline hadronic EOS failed to build"

    print(f"\n{'n_tr':>4} {'dE':>5} | {'M_max':>6} {'R_1.4':>6} {'L_1.4':>6} "
          f"{'Lt_sym':>7} {'Lt_asym':>7} | ok  margin")
    results = {}
    for n_tr in GRID_NTR:
        for de in GRID_DE:
            hyb = soft.build_css_hybrid_eos(baseline, n_trans_ratio=n_tr,
                                            delta_eps_ratio=de, cs2_q=1.0 / 3.0)
            if hyb is None:
                continue
            mmax, r14, l14, lts, lta = star_family(hyb)
            ok = (mmax >= 2.01 and max(lts, lta) <= 720.0
                  and min(lts, lta) >= 70.0 and 11.2 <= r14 <= 13.2)
            marg = margin(mmax, r14, lts, lta)
            results[(n_tr, de)] = (mmax, r14, l14, lts, lta, ok, marg)
            print(f"{n_tr:4.1f} {de:5.2f} | {mmax:6.3f} {r14:6.2f} {l14:6.0f} "
                  f"{lts:7.0f} {lta:7.0f} | {'Y' if ok else 'n'}  {marg:+.2f}")

    best, selection = select_canonical_point(results)

    print("\n" + "─" * 84)
    print(f"Surviving points: {selection['survivor_count']} of {len(results)}")
    # Falsification reference: the pre-canon parameterization lies OUTSIDE the
    # fine grid above, so compute it explicitly.
    hyb_old = soft.build_css_hybrid_eos(baseline, n_trans_ratio=OLD[0],
                                        delta_eps_ratio=OLD[1], cs2_q=1.0 / 3.0)
    mmax_o, r14_o = star_family(hyb_old)[:2]
    print(f"Reference parameterization {OLD}: M_max = {mmax_o:.2f} M_sun "
          f"(vs J0740 2.08 ± 0.07 → {(mmax_o - 2.08) / 0.07:+.1f} sigma) — FALSIFIED")
    if best:
        mb = results[best]
        print(f"Best-margin survivor {best}: M_max = {mb[0]:.2f}, R_1.4 = {mb[1]:.2f}, "
              f"L_1.4 = {mb[2]:.0f}, Ltilde = {mb[3]:.0f}/{mb[4]:.0f}")

    print("\nCANONICAL SELECTION PROVENANCE (machine-readable semantics):")
    print(f"  status = {selection['provenance']['status']}")
    print(f"  selected_by = {selection['provenance']['selected_by']}")
    print(f"  method = {selection['provenance']['selection_method']}")
    print("  constraints = J0740, GW170817, NICER (selection inputs; no hold-out)")

    if best is None or CANON not in results:
        raise RuntimeError("canonical selection has no runtime survivor")

    print("\nFALSIFIABILITY (conditional canonical point):")
    mc = results[CANON]
    print(f"  - A confirmed NS above ~2.2 M_sun excludes the model "
          f"(canon M_max = {mc[0]:.2f}).")
    print(f"  - A GW170817-like Ltilde measurement below ~400 excludes it "
          f"(canon ~{mc[3]:.0f}).")
    print(f"  - R_1.4 measured below ~12.0 km excludes it (canon {mc[1]:.2f} km).")

    # ── Assertions ──────────────────────────────────────────────────────
    assert mmax_o < 2.01, "old point should be falsified by J0740"
    assert CANON in results and results[CANON][5], "canonical point must satisfy all constraints"
    assert best == CANON, (f"best-margin survivor {best} != canonical {CANON}; "
                           f"update the canon in nvg_tidal_deformability.py")
    print("\nRuntime selection complete: canonical point is the best-margin survivor; status is conditional/in-sample.")
    print("=" * 84)


if __name__ == "__main__":
    main()
