#!/usr/bin/env python3
"""Numerical sign map for the maintained CPL growth calculation.

This is a Phase 1 predictive-research audit, not a replacement for the
maintained S8 producer.  ``nvg_black_hole_entropy.compute_s8_test`` remains
the canonical route and is executed as the regression anchor.  The map
generalises the *same* growth-index approximation to a declared sensitivity
grid in ``(w0, wa, Omega_m)`` and compares it with an explicitly labelled
DESI overlay.  The DESI numbers are inherited from the retired/unknown-
provenance joint-map script and are therefore never called an observed
likelihood here.

The result intentionally distinguishes three statements:

* the maintained point's sign (it increases S8 and worsens the lensing
  distance);
* a scoped no-go on the maintained flat-background ``Omega_m=0.315`` slice
  at the one-sigma declared lensing target; and
* the fact that expanding Omega_m admits sensitivity-only overlaps, so no
  global observational no-go is claimed.

All values in the generated JSON are computed by this module at run time.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the maintained producer and its inputs.  Importing the retired map is
# intentional: it is the single existing source for the old DESI ellipse, and
# the semantics below explicitly downgrade that ellipse to sensitivity-only.
from nvg_black_hole_entropy import compute_s8_test
from nvg_desi_s8_joint_map import (
    RHO as DESI_RHO,
    S_W0 as DESI_SIGMA_W0,
    S_WA as DESI_SIGMA_WA,
    W0_D as DESI_W0,
    WA_D as DESI_WA,
)


CANONICAL_SOURCE = "verification/nvg_black_hole_entropy.py::compute_s8_test"
DESI_SOURCE = "verification/nvg_desi_s8_joint_map.py (retired/unknown-provenance overlay)"
GROWTH_INDEX = 0.55
ALT_GROWTH_INDEX = 0.545
A_MIN = 0.01
S8_LENSING_SIGMA = 0.017  # declared scalar in the maintained producer's comparison
DESI_CHI2_LIMIT = 4.0  # the existing map's 2-sigma sensitivity contour
EARLY_MATTER_FRACTION_MIN = 0.95


@dataclass(frozen=True)
class Domain:
    """A declared scan boundary, with no statistical interpretation."""

    name: str
    w0: tuple[float, float]
    wa: tuple[float, float]
    omega_m: tuple[float, float]
    shape: tuple[int, int, int]


# The canonical producer fixes Omega_m=0.315.  This slice varies CPL inputs
# over a deliberately broad audit window while preserving that maintained
# background input.  The expanded domain tests whether a conclusion survives
# releasing that assumption; it is a sensitivity exercise, not a prior.
MAINTAINED_DOMAIN = Domain(
    "maintained_omega_m_slice",
    (-1.20, -0.60),
    (-1.50, 0.50),
    (0.315, 0.315),
    (25, 25, 1),
)
EXPANDED_DOMAIN = Domain(
    "expanded_omega_m_sensitivity",
    (-1.50, -0.30),
    (-2.50, 1.50),
    (0.28, 0.36),
    (25, 25, 9),
)


def cpl_density_ratio(a: np.ndarray | float, w0: float, wa: float) -> np.ndarray:
    """Return rho_DE(a)/rho_DE(1) for the standard CPL parameterisation."""

    arr = np.asarray(a, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("scale factors must be finite and positive")
    log_ratio = -3.0 * (1.0 + float(w0) + float(wa)) * np.log(arr)
    log_ratio -= 3.0 * float(wa) * (1.0 - arr)
    # The declared scan boundaries do not overflow, but clipping keeps a
    # malformed caller from manufacturing NaNs in the structured output.
    return np.exp(np.clip(log_ratio, -700.0, 700.0))


def background(a: np.ndarray, w0: float, wa: float, omega_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matter fraction, DE density, and E^2 on an increasing-a grid."""

    if not (0.0 < omega_m < 1.0):
        raise ValueError("Omega_m must lie strictly between zero and one")
    rho_m = omega_m * a ** -3.0
    rho_de = (1.0 - omega_m) * cpl_density_ratio(a, w0, wa)
    e2 = rho_m + rho_de
    if np.any(~np.isfinite(e2)) or np.any(e2 <= 0.0):
        raise ValueError("CPL background has non-positive or non-finite E^2")
    omega = rho_m / e2
    return omega, rho_de, e2


def growth_index_ratio(
    w0: float,
    wa: float,
    omega_m: float,
    *,
    n_steps: int = 1000,
    gamma: float = GROWTH_INDEX,
) -> float:
    """Compute D/D_LCDM using the maintained f=Omega_m**gamma integral.

    This is the same displayed approximation used by
    :func:`nvg_black_hole_entropy.compute_s8_test`, with its parameters made
    explicit for a map.  ``n_steps=1000`` and ``A_MIN=0.01`` reproduce the
    canonical point to floating-point precision.
    """

    if n_steps < 8:
        raise ValueError("growth grid needs at least eight points")
    a = np.linspace(A_MIN, 1.0, int(n_steps))
    omega, _, _ = background(a, w0, wa, omega_m)
    omega_lcdm = omega_m * a ** -3.0 / (omega_m * a ** -3.0 + (1.0 - omega_m))
    delta = omega**float(gamma) - omega_lcdm**float(gamma)
    log_ratio = float(np.trapz(delta, np.log(a)))
    return float(np.exp(log_ratio))


def growth_ode_ratio(w0: float, wa: float, omega_m: float, *, n_steps: int = 1000) -> float:
    """Independent linear-growth ODE check for the sign map.

    The ODE is the standard GR growth equation in a smooth CPL background.
    It is used only as an approximation cross-check; the maintained producer
    remains the growth-index integral above.
    """

    if n_steps < 16:
        raise ValueError("ODE grid needs at least sixteen points")
    a = np.linspace(A_MIN, 1.0, int(n_steps))

    def rhs(x: float, y: np.ndarray, w0_: float, wa_: float, om_: float) -> np.ndarray:
        om_frac, rho_de, e2 = background(np.asarray([x]), w0_, wa_, om_)
        # d ln(E^2)/da, evaluated analytically to avoid finite-difference noise.
        dlog_de = -3.0 * (1.0 + w0_ + wa_) / x + 3.0 * wa_
        de = float(rho_de[0])
        e2v = float(e2[0])
        d_e2 = -3.0 * om_ * x**-4.0 + de * dlog_de
        dln_e2 = d_e2 / e2v
        d2 = -(3.0 / x + 0.5 * dln_e2) * y[1] + 1.5 * float(om_frac[0]) * y[0] / x**2
        return np.array([y[1], d2], dtype=float)

    def integrate(w0_: float, wa_: float, om_: float) -> float:
        y = np.array([A_MIN, 1.0], dtype=float)
        for x0, x1 in zip(a[:-1], a[1:]):
            h = float(x1 - x0)
            k1 = rhs(x0, y, w0_, wa_, om_)
            k2 = rhs(x0 + h / 2.0, y + h * k1 / 2.0, w0_, wa_, om_)
            k3 = rhs(x0 + h / 2.0, y + h * k2 / 2.0, w0_, wa_, om_)
            k4 = rhs(x1, y + h * k3, w0_, wa_, om_)
            y += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return float(y[0])

    return integrate(w0, wa, omega_m) / integrate(-1.0, 0.0, omega_m)


def analytic_sign_control(w0: float, wa: float, *, n_steps: int = 1000) -> dict[str, Any]:
    """Check the sign of the CPL DE contrast relative to Lambda.

    With the same present-day Omega_m, ``g(a)=ln(rho_DE/rho_Lambda)`` is
    ``-3[(1+w0+wa)ln(a)+wa(1-a)]``.  Positive g lowers Omega_m(a), hence lowers
    f=Omega_m**gamma for positive gamma and tends to suppress growth.  A mixed
    sign requires the integrated numerical control rather than a verbal
    quadrant argument.
    """

    a = np.linspace(A_MIN, 1.0, int(n_steps))
    g = -3.0 * ((1.0 + float(w0) + float(wa)) * np.log(a) + float(wa) * (1.0 - a))
    tol = 2e-10
    positive = bool(np.all(g >= -tol))
    negative = bool(np.all(g <= tol))
    if positive and not negative:
        sign_class = "suppression_sign_everywhere"
    elif negative and not positive:
        sign_class = "enhancement_sign_everywhere"
    else:
        sign_class = "mixed_sign_integral_required"
    return {
        "g_min": float(np.min(g)),
        "g_max": float(np.max(g)),
        "early_sign_control": "positive" if float(g[0]) > 0 else ("negative" if float(g[0]) < 0 else "zero"),
        "sign_class": sign_class,
        "integral_sign_rule": "sign(log(D/D_LCDM)) = sign(∫[f_model-f_LCDM] dln a)",
        "formula": "g(a)=-3*((1+w0+wa)*ln(a)+wa*(1-a))",
    }


def desi_overlay_chi2(w0: float, wa: float) -> float:
    """Return the retired-map ellipse distance, explicitly sensitivity-only."""

    cov = np.array(
        [
            [DESI_SIGMA_W0**2, DESI_RHO * DESI_SIGMA_W0 * DESI_SIGMA_WA],
            [DESI_RHO * DESI_SIGMA_W0 * DESI_SIGMA_WA, DESI_SIGMA_WA**2],
        ],
        dtype=float,
    )
    d = np.array([float(w0) - DESI_W0, float(wa) - DESI_WA], dtype=float)
    return float(d @ np.linalg.inv(cov) @ d)


def point_record(w0: float, wa: float, omega_m: float, *, n_steps: int = 1000) -> dict[str, Any]:
    """Compute one fully structured map point."""

    omega, _, _ = background(np.array([A_MIN, 1.0]), w0, wa, omega_m)
    growth_ratio = growth_index_ratio(w0, wa, omega_m, n_steps=n_steps)
    # The maintained producer labels ``s8_planck`` as S8 (rather than a bare
    # sigma8).  Preserve that contract by normalising the Omega_m sensitivity
    # to its canonical 0.315 value; this makes the canonical point exactly the
    # producer's 0.851 result instead of applying the sqrt factor twice.
    s8_planck = float(_CANONICAL["s8_planck"])
    s8_lensing = float(_CANONICAL["s8_lensing"])
    s8_value = s8_planck * growth_ratio * math.sqrt(omega_m / 0.315)
    baseline_s8 = s8_planck
    distance_baseline = abs(baseline_s8 - s8_lensing)
    distance = abs(s8_value - s8_lensing)
    chi2 = desi_overlay_chi2(w0, wa)
    early_matter_ok = bool(omega[0] >= EARLY_MATTER_FRACTION_MIN)
    return {
        "w0": float(w0),
        "wa": float(wa),
        "omega_m": float(omega_m),
        "growth_ratio_gamma055": float(growth_ratio),
        "s8": float(s8_value),
        "s8_shift_from_canonical_pct": float((s8_value / baseline_s8 - 1.0) * 100.0),
        "distance_to_declared_lensing": float(distance),
        "moves_toward_lensing": bool(distance < distance_baseline),
        "moves_away_from_lensing": bool(distance > distance_baseline),
        "lensing_1sigma": bool(abs(s8_value - s8_lensing) <= S8_LENSING_SIGMA),
        "lensing_2sigma": bool(abs(s8_value - s8_lensing) <= 2.0 * S8_LENSING_SIGMA),
        "desi_overlay_chi2": float(chi2),
        "desi_overlay_2sigma": bool(chi2 <= DESI_CHI2_LIMIT),
        "early_matter_fraction": float(omega[0]),
        "background_ok": early_matter_ok,
        "sign": analytic_sign_control(w0, wa, n_steps=n_steps),
    }


def _grid_values(domain: Domain) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.linspace(*domain.w0, domain.shape[0]),
        np.linspace(*domain.wa, domain.shape[1]),
        np.linspace(*domain.omega_m, domain.shape[2]),
    )


def scan_domain(domain: Domain, *, n_steps: int = 1000, retain_points: bool = False) -> dict[str, Any]:
    """Scan a domain and return counts plus optionally compact point rows."""

    w0_values, wa_values, om_values = _grid_values(domain)
    rows: list[dict[str, Any]] = []
    counts = {
        "total": 0,
        "background_ok": 0,
        "toward_lensing": 0,
        "away_from_lensing": 0,
        "lensing_1sigma": 0,
        "lensing_2sigma": 0,
        "desi_overlay_2sigma": 0,
        "both_lensing_1sigma_and_desi_overlay": 0,
        "both_lensing_2sigma_and_desi_overlay": 0,
    }
    best_toward: dict[str, Any] | None = None
    best_overlap: dict[str, Any] | None = None
    for w0 in w0_values:
        for wa in wa_values:
            for om in om_values:
                row = point_record(float(w0), float(wa), float(om), n_steps=n_steps)
                counts["total"] += 1
                if row["background_ok"]:
                    counts["background_ok"] += 1
                for row_key, count_key in (
                    ("moves_toward_lensing", "toward_lensing"),
                    ("moves_away_from_lensing", "away_from_lensing"),
                    ("lensing_1sigma", "lensing_1sigma"),
                    ("lensing_2sigma", "lensing_2sigma"),
                ):
                    if row[row_key]:
                        counts[count_key] += 1
                if row["desi_overlay_2sigma"]:
                    counts["desi_overlay_2sigma"] += 1
                if row["background_ok"] and row["desi_overlay_2sigma"] and row["lensing_1sigma"]:
                    counts["both_lensing_1sigma_and_desi_overlay"] += 1
                if row["background_ok"] and row["desi_overlay_2sigma"] and row["lensing_2sigma"]:
                    counts["both_lensing_2sigma_and_desi_overlay"] += 1
                if row["moves_toward_lensing"] and row["background_ok"]:
                    if best_toward is None or row["distance_to_declared_lensing"] < best_toward["distance_to_declared_lensing"]:
                        best_toward = row
                if row["background_ok"] and row["desi_overlay_2sigma"] and row["lensing_1sigma"]:
                    if best_overlap is None or row["desi_overlay_chi2"] < best_overlap["desi_overlay_chi2"]:
                        best_overlap = row
                if retain_points:
                    # Keep the machine-readable map explicit without
                    # repeating the analytic sign metadata in every cell.
                    rows.append(
                        {
                            key: row[key]
                            for key in (
                                "w0",
                                "wa",
                                "omega_m",
                                "growth_ratio_gamma055",
                                "s8",
                                "s8_shift_from_canonical_pct",
                                "distance_to_declared_lensing",
                                "moves_toward_lensing",
                                "moves_away_from_lensing",
                                "lensing_1sigma",
                                "lensing_2sigma",
                                "desi_overlay_chi2",
                                "desi_overlay_2sigma",
                                "early_matter_fraction",
                                "background_ok",
                            )
                        }
                    )
    result: dict[str, Any] = {"domain": domain.name, "shape": list(domain.shape), "counts": counts}
    result["best_toward_lensing"] = best_toward
    result["best_1sigma_overlay_overlap"] = best_overlap
    if retain_points:
        result["points"] = rows
    return result


def convergence_summary(w0: float, wa: float, omega_m: float) -> dict[str, Any]:
    """Check resolution convergence and canonical-producer identity."""

    resolutions = (250, 500, 1000, 2000)
    values = [growth_index_ratio(w0, wa, omega_m, n_steps=n) for n in resolutions]
    spread = max(values) - min(values)
    canonical_ratio = float(_CANONICAL["sigma8_ratio"])
    identity_ratio = growth_index_ratio(
        float(_CANONICAL["w_0"]), float(_CANONICAL["w_a"]), 0.315, n_steps=1000
    )
    return {
        "resolutions": list(resolutions),
        "growth_ratio": values,
        "max_minus_min": float(spread),
        "relative_spread": float(spread / values[-1]),
        "converged": bool(spread / values[-1] < 2e-3),
        "canonical_source": CANONICAL_SOURCE,
        "canonical_sigma8_ratio": canonical_ratio,
        "map_sigma8_ratio_at_canonical_point": float(identity_ratio),
        "canonical_identity_abs_error": float(abs(identity_ratio - canonical_ratio)),
        "canonical_identity_pass": bool(abs(identity_ratio - canonical_ratio) < 3e-6),
    }


def map_resolution_summary(domain: Domain = MAINTAINED_DOMAIN) -> dict[str, Any]:
    """Check that the scoped overlap count is not a grid-resolution artefact."""

    resolutions = (9, 17, 25)
    rows = []
    for resolution in resolutions:
        test_domain = replace(domain, shape=(resolution, resolution, domain.shape[2]))
        scan = scan_domain(test_domain, n_steps=1000, retain_points=False)
        rows.append(
            {
                "resolution": resolution,
                "total": scan["counts"]["total"],
                "background_ok": scan["counts"]["background_ok"],
                "one_sigma_overlay_overlap": scan["counts"]["both_lensing_1sigma_and_desi_overlay"],
                "two_sigma_overlay_overlap": scan["counts"]["both_lensing_2sigma_and_desi_overlay"],
            }
        )
    return {
        "domain": domain.name,
        "rows": rows,
        "one_sigma_overlap_stable_zero": bool(all(row["one_sigma_overlay_overlap"] == 0 for row in rows)),
    }


def alternative_approximations(w0: float, wa: float, omega_m: float) -> dict[str, Any]:
    """Compare gamma=0.545 and direct ODE signs at the maintained point."""

    gamma055 = growth_index_ratio(w0, wa, omega_m, gamma=0.55)
    gamma0545 = growth_index_ratio(w0, wa, omega_m, gamma=ALT_GROWTH_INDEX)
    ode = growth_ode_ratio(w0, wa, omega_m)
    return {
        "canonical_growth_index_gamma": 0.55,
        "alternate_growth_index_gamma": ALT_GROWTH_INDEX,
        "ratio_gamma055": gamma055,
        "ratio_gamma0545": gamma0545,
        "ratio_direct_growth_ode": ode,
        "all_enhance_at_maintained_point": bool(gamma055 > 1.0 and gamma0545 > 1.0 and ode > 1.0),
        "sign_agreement": bool((gamma055 - 1.0) * (gamma0545 - 1.0) > 0 and (gamma055 - 1.0) * (ode - 1.0) > 0),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def build_result() -> dict[str, Any]:
    """Run the complete audit and return a structured, provenance-labelled result."""

    maintained_point = point_record(
        float(_CANONICAL["w_0"]), float(_CANONICAL["w_a"]), 0.315, n_steps=1000
    )
    maintained_scan = scan_domain(MAINTAINED_DOMAIN, n_steps=1000, retain_points=True)
    expanded_scan = scan_domain(EXPANDED_DOMAIN, n_steps=1000, retain_points=True)
    map_convergence = map_resolution_summary(MAINTAINED_DOMAIN)
    convergence = convergence_summary(
        float(_CANONICAL["w_0"]), float(_CANONICAL["w_a"]), 0.315
    )
    alternatives = alternative_approximations(
        float(_CANONICAL["w_0"]), float(_CANONICAL["w_a"]), 0.315
    )

    scoped_no_go = (
        maintained_scan["counts"]["both_lensing_1sigma_and_desi_overlay"] == 0
        and convergence["converged"]
        and convergence["canonical_identity_pass"]
        and alternatives["sign_agreement"]
        and map_convergence["one_sigma_overlap_stable_zero"]
    )
    expanded_overlap = expanded_scan["counts"]["both_lensing_1sigma_and_desi_overlay"]
    status = (
        "SCOPED_NO_GO_MAINTAINED_OMEGA_M"
        if scoped_no_go
        else "NO_ROBUST_NO_GO"
    )
    if expanded_overlap:
        status += "; EXPANDED_OMEGA_M_SENSITIVITY_OVERLAP"

    return {
        "schema_version": 1,
        "status": status,
        "claim_scope": {
            "observational_claim": False,
            "summary": "The scoped result concerns the maintained Omega_m=0.315 background slice and a declared one-sigma S8 target; it is not an empirical DESI likelihood or a global no-go theorem.",
            "maintained_background": "flat CPL comparison with Omega_m fixed to 0.315, positive E^2, and Omega_m(a=0.01)>=0.95",
            "s8_semantics": "declared scalar input from the maintained producer; no external lensing likelihood is reconstructed",
            "desi_semantics": "sensitivity_overlay_unknown_provenance",
        },
        "provenance": {
            "canonical_growth_source": CANONICAL_SOURCE,
            "canonical_route_executed": True,
            "desi_overlay_source": DESI_SOURCE,
            "desi_overlay_is_observed_likelihood": False,
            "generated_by": "verification/nvg_s8_no_go.py",
        },
        "inputs": {
            "canonical": dict(_CANONICAL),
            "s8_lensing_sigma_declared": S8_LENSING_SIGMA,
            "desi_overlay": {
                "w0": DESI_W0,
                "wa": DESI_WA,
                "sigma_w0": DESI_SIGMA_W0,
                "sigma_wa": DESI_SIGMA_WA,
                "rho": DESI_RHO,
                "chi2_limit": DESI_CHI2_LIMIT,
            },
            "early_matter_fraction_min": EARLY_MATTER_FRACTION_MIN,
            "a_min": A_MIN,
        },
        "maintained_point": maintained_point,
        "analytic_sign_control": maintained_point["sign"],
        "convergence": convergence,
        "alternative_approximations": alternatives,
        "map_resolution_convergence": map_convergence,
        "scans": {
            "maintained": maintained_scan,
            "expanded": expanded_scan,
        },
        "boundary_conclusion": {
            "scoped_no_go": bool(scoped_no_go),
            "expanded_omega_m_overlay_overlap_count": int(expanded_overlap),
            "interpretation": "No one-sigma S8/lensing plus DESI-overlay overlap survives on the maintained Omega_m slice; releasing Omega_m creates sensitivity-only overlaps, so the no-go is deliberately scoped.",
        },
    }


def write_artifacts(result: dict[str, Any], *, json_path: Path | None = None, figure_path: Path | None = None) -> tuple[Path, Path]:
    """Write the uniquely named structured map and a compact diagnostic figure."""

    json_path = json_path or HERE / "nvg_s8_no_go_map.json"
    figure_path = figure_path or HERE / "fig_s8_no_go_map.png"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - the runtime has matplotlib
        raise RuntimeError("matplotlib is required to render the S8 no-go figure") from exc

    # A compact, fully recomputed map on the maintained Omega_m slice.
    w0s = np.linspace(*MAINTAINED_DOMAIN.w0, 61)
    was = np.linspace(*MAINTAINED_DOMAIN.wa, 61)
    grid_s8 = np.empty((len(was), len(w0s)))
    grid_chi = np.empty_like(grid_s8)
    for iy, wa in enumerate(was):
        for ix, w0 in enumerate(w0s):
            row = point_record(float(w0), float(wa), 0.315, n_steps=1000)
            grid_s8[iy, ix] = row["s8"]
            grid_chi[iy, ix] = row["desi_overlay_chi2"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    xx, yy = np.meshgrid(w0s, was)
    image = axes[0].pcolormesh(xx, yy, grid_s8, shading="auto", cmap="coolwarm")
    axes[0].contour(xx, yy, grid_s8, levels=[float(_CANONICAL["s8_planck"])], colors="k", linewidths=0.8)
    axes[0].contour(
        xx,
        yy,
        grid_s8,
        levels=[
            float(_CANONICAL["s8_lensing"]) - S8_LENSING_SIGMA,
            float(_CANONICAL["s8_lensing"]) + S8_LENSING_SIGMA,
        ],
        colors="lime",
        linewidths=1.0,
    )
    axes[0].scatter([_CANONICAL["w_0"]], [_CANONICAL["w_a"]], color="white", edgecolor="black", s=40, label="maintained point")
    axes[0].set_title("S8 on maintained Omega_m slice")
    axes[0].set_xlabel("w0")
    axes[0].set_ylabel("wa")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.colorbar(image, ax=axes[0], label="computed S8")
    image2 = axes[1].pcolormesh(xx, yy, np.log10(np.maximum(grid_chi, 1e-12)), shading="auto", cmap="viridis")
    axes[1].contour(xx, yy, grid_chi, levels=[DESI_CHI2_LIMIT], colors="white", linewidths=1.1)
    axes[1].scatter([DESI_W0], [DESI_WA], color="red", edgecolor="white", s=36, label="DESI overlay center")
    axes[1].set_title("retired DESI overlay (sensitivity only)")
    axes[1].set_xlabel("w0")
    axes[1].set_ylabel("wa")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.colorbar(image2, ax=axes[1], label="log10 chi2")
    fig.suptitle("NVG S8 sign/no-go audit; no observational likelihood claim")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return json_path, figure_path


def main() -> dict[str, Any]:
    global _CANONICAL
    _CANONICAL = dict(compute_s8_test())
    result = build_result()
    paths = write_artifacts(result)
    print("NVG S8 SIGN / NO-GO AUDIT")
    print(f"STATUS: {result['status']}")
    print(f"Maintained point S8 = {result['maintained_point']['s8']:.6f}; direction = {('toward' if result['maintained_point']['moves_toward_lensing'] else 'away')}")
    print(f"Maintained-slice 1σ + DESI-overlay overlaps = {result['scans']['maintained']['counts']['both_lensing_1sigma_and_desi_overlay']}")
    print(f"Expanded-Omega_m sensitivity overlaps = {result['boundary_conclusion']['expanded_omega_m_overlay_overlap_count']}")
    print(f"JSON: {paths[0]}")
    print(f"FIGURE: {paths[1]}")
    return result


_CANONICAL: dict[str, Any] = dict(compute_s8_test())


if __name__ == "__main__":
    main()
