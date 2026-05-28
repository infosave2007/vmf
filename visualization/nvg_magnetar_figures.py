"""Generate publication figures for the magnetar V3 preprint.

Reads the population scan module, runs the catalog pipeline, and emits three
PDF/PNG figures into article/figures/:

1. fig_magnetar_corridor_coverage.{pdf,png}
   Bar chart of population coverage under the narrow fossil corridor
   (1e3..1e4 G) versus the magnetic-progenitor corridor (1e3..1e5 G).
2. fig_magnetar_gamma_likelihood.{pdf,png}
   Pseudo-log-likelihood profile for the global Gamma_struct scale,
   with the benchmark band and the data-preferred band marked.
3. fig_magnetar_bdip_vs_psupport.{pdf,png}
   Scatter of inferred dipole field versus per-source support probability,
   with the corridor edges marked.

Usage::

    python3 visualization/nvg_magnetar_figures.py

The figures are referenced from article/NVG_MAGNETAR_PREPRINT_V3.md.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "verification"))

import nvg_magnetar_population_scan as scan  # noqa: E402

FIG_DIR = REPO_ROOT / "article" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_objects() -> tuple[list, str]:
    try:
        objects, note = scan.build_sample()
    except Exception as exc:  # pragma: no cover
        print(f"Sample build failed ({exc}); using fallback list.")
        objects = list(scan.FALLBACK_MAGNETARS) + list(scan.CURATED_NON_MAGNETARS)
        note = f"fallback ({len(scan.FALLBACK_MAGNETARS)} magnetars)"
    return objects, note


def figure_corridor_coverage(objects: list) -> None:
    ordinary = [o for o in objects if o.family != "long-period outlier"]
    n_total = len(ordinary)

    inside_narrow = below_narrow = above_narrow = 0
    for obj in ordinary:
        status, _, _ = scan.classify_object(obj)
        if status == "inside fossil corridor":
            inside_narrow += 1
        elif status == "below fossil corridor":
            below_narrow += 1
        elif status == "above fossil corridor":
            above_narrow += 1

    inside_mag = below_mag = above_mag = 0
    for obj in ordinary:
        mstatus = scan.magnetic_channel_status(obj)
        if mstatus == "inside magnetic channel":
            inside_mag += 1
        elif mstatus == "below magnetic channel":
            below_mag += 1
        elif mstatus == "above magnetic channel":
            above_mag += 1

    categories = ["Inside", "Sub-critical\n(consistent)", "Above corridor\n(tension)"]
    narrow = [inside_narrow, below_narrow, above_narrow]
    wide = [inside_mag, below_mag, above_mag]

    x = list(range(len(categories)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bars1 = ax.bar([xi - width / 2 for xi in x], narrow, width,
                   label=r"Narrow fossil $10^{3}-10^{4}$ G", color="#3a6ea5")
    bars2 = ax.bar([xi + width / 2 for xi in x], wide, width,
                   label=r"Magnetic progenitor $10^{3}-10^{5}$ G", color="#c0392b")

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{int(h)}/{n_total}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(f"Number of sources (N={n_total})")
    ax.set_ylim(0, n_total + 3)
    ax.set_title("Population coverage of the NVG magnetar channel")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_magnetar_corridor_coverage.{ext}", dpi=200)
    plt.close(fig)


def figure_gamma_likelihood(objects: list) -> None:
    fit = scan.gamma_likelihood_scan(objects)
    scales = [r[0] for r in fit["rows"]]
    loglike = [r[2] for r in fit["rows"]]
    best = fit["best_loglike"]
    delta = [ll - best for ll in loglike]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(scales, delta, color="#2c3e50", lw=1.6)
    ax.axhline(-0.5, ls="--", color="#27ae60", lw=0.8,
               label=r"$\Delta \log L = -0.5$ (narrow)")
    ax.axhline(-2.0, ls="--", color="#c0392b", lw=0.8,
               label=r"$\Delta \log L = -2.0$ (wide)")

    bench_lo = 1.0
    bench_hi = scan.GAMMA_STRUCT_BENCHMARK_MAX / scan.GAMMA_STRUCT_BENCHMARK_MIN
    ax.axvspan(1.0, bench_hi, color="#3a6ea5", alpha=0.15,
               label="V1 benchmark band")
    ax.axvspan(fit["narrow_min"], fit["narrow_max"], color="#f1c40f", alpha=0.20,
               label="Narrow support band")
    ax.axvline(fit["best_scale"], color="#e67e22", lw=1.0,
               label=f"Best fit: scale={fit['best_scale']:.2f}")

    ax.set_xlabel(r"Global scale on $\Gamma_{\rm struct}$ benchmark")
    ax.set_ylabel(r"$\Delta \log L$ vs best fit")
    ax.set_xlim(min(scales), max(scales))
    ax.set_ylim(min(min(delta), -8.0), 0.5)
    ax.set_title("Global pseudo-likelihood for the structural amplification")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_magnetar_gamma_likelihood.{ext}", dpi=200)
    plt.close(fig)


def figure_bdip_vs_psupport(objects: list) -> None:
    rows = scan.source_probability_summary(objects)
    prob_map = {name: prob for name, _, prob in rows}

    fams = {"magnetar": ("#c0392b", "o"),
            "transition": ("#8e44ad", "s"),
            "high-B pulsar": ("#2c7fb8", "^")}

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for fam, (color, marker) in fams.items():
        xs = []
        ys = []
        for obj in objects:
            if obj.family != fam:
                continue
            xs.append(obj.b_dip_g)
            ys.append(prob_map.get(obj.name, 0.0))
        ax.scatter(xs, ys, c=color, marker=marker, edgecolor="black",
                   linewidth=0.4, s=46, label=fam, alpha=0.85)

    highlights = ["SGR 1806-20", "1E 1841-045", "SGR 1900+14", "SGR 0418+5729"]
    for obj in objects:
        if obj.name in highlights and obj.name in prob_map:
            ax.annotate(obj.name, (obj.b_dip_g, prob_map[obj.name]),
                        textcoords="offset points", xytext=(6, -4),
                        fontsize=7, color="#2c3e50")

    ax.set_xscale("log")
    ax.set_xlabel(r"$B_{\rm dip}$ (G)")
    ax.set_ylabel(r"$P_{\rm inside}$ (magnetic-progenitor corridor)")
    ax.set_ylim(-0.04, 1.08)
    ax.axhline(0.80, ls="--", color="#27ae60", lw=0.8, label="high-support")
    ax.axhline(0.20, ls="--", color="#c0392b", lw=0.8, label="low-support")
    ax.set_title("Per-source support vs inferred dipole field")
    ax.legend(loc="lower left", fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_magnetar_bdip_vs_psupport.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    objects, note = load_objects()
    print(f"Catalog: {note}; {len(objects)} sources total.")
    figure_corridor_coverage(objects)
    figure_gamma_likelihood(objects)
    figure_bdip_vs_psupport(objects)
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
