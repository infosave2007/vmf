from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "article" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _upper_bound_gamma0(mass_amu: float, size_m: float, rho_env: float, grad_env: float, time_s: float, delta: int) -> float:
    mass_factor = mass_amu / 1.0
    size_factor = size_m / 1e-9
    rho_factor = rho_env / 1e3
    grad_factor = (grad_env / 1e13) ** delta
    prefactor = mass_factor * size_factor * rho_factor * grad_factor
    return 1.0 / (time_s * prefactor)


def allowed_region_figure() -> Path:
    # Benchmarks from NVG_WAVE_COLLAPSE_PREPRINT.md
    l_int = 1e-7
    l_det = 1e-6
    gamma_lower = 1e-6
    gamma_upper_soft = 4e37
    gamma_upper_grad = 4e63

    x = np.logspace(-9, -5, 600)
    y = np.logspace(-12, 66, 800)
    xx, yy = np.meshgrid(x, y)

    allowed_soft = np.ones_like(xx, dtype=bool)
    # Detector must localize if threshold is below the detector scale.
    allowed_soft &= np.where(xx < l_det, yy > gamma_lower, False)
    # Interferometry constrains only if threshold is below the interferometer scale.
    allowed_soft &= np.where(xx < l_int, yy < gamma_upper_soft, True)

    allowed_grad = np.ones_like(xx, dtype=bool)
    allowed_grad &= np.where(xx < l_det, yy > gamma_lower, False)
    allowed_grad &= np.where(xx < l_int, yy < gamma_upper_grad, True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    configs = [
        (axes[0], allowed_soft, gamma_upper_soft, "Soft benchmark: (1,1,1,1)"),
        (axes[1], allowed_grad, gamma_upper_grad, "Gradient benchmark: (1,1,1,2)"),
    ]

    for ax, allowed, gamma_upper, title in configs:
        ax.contourf(
            xx,
            yy,
            allowed.astype(int),
            levels=[-0.5, 0.5, 1.5],
            colors=["#f4b7b7", "#b9e3b2"],
            alpha=0.9,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(l_int, color="#1f4e79", linestyle="--", linewidth=1.4, label=r"$L_{\rm int}$")
        ax.axvline(l_det, color="#7a3e00", linestyle="-.", linewidth=1.4, label=r"$L_{\rm det}$")
        ax.axhline(gamma_lower, color="#8b0000", linestyle="--", linewidth=1.4, label=r"$\Gamma_0^{\rm min}$")
        if math.isfinite(gamma_upper):
            ax.axhline(gamma_upper, color="#006400", linestyle=":", linewidth=1.8, label=r"$\Gamma_0^{\rm max}$")

        ax.text(2.0e-9, 3.0e-10, "detector too slow", fontsize=9, color="#6b0000")
        ax.text(2.5e-8, 2.0e10, "allowed", fontsize=10, color="#174d17", weight="bold")
        ax.text(2.2e-7, 2.0e10, "interference-safe\nby threshold", fontsize=9, color="#174d17")
        ax.text(1.8e-6, 2.0e10, "no detector\nlocalisation", fontsize=9, color="#6b0000")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"threshold scale $L_{\rm crit}$ [m]")
        ax.set_ylabel(r"reference rate $\Gamma_0$ [s$^{-1}$]")
        ax.set_xlim(1e-9, 1e-5)
        ax.set_ylim(1e-12, 1e66)
        ax.grid(True, which="both", alpha=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Illustrative allowed/excluded corridor for the NVG collapse ansatz", fontsize=13)

    out_path = FIG_DIR / "fig_wave_collapse_allowed_region.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def benchmark_lines_figure() -> Path:
    l_det = 1e-6
    gamma_lower = 1e-6
    x_min, x_max = 1e-9, 1e-5
    y_min, y_max = 1e-12, 1e66

    benchmarks = [
        {
            "label": "Atomic interferometer (representative)",
            "mass_amu": 87.0,
            "size_m": 1e-6,
            "rho_env": 1e-11,
            "grad_env": 1e-11,
            "time_s": 1.0,
            "color": "#2b6cb0",
        },
        {
            "label": r"C$_{60}$-class molecule (representative)",
            "mass_amu": 720.0,
            "size_m": 1e-7,
            "rho_env": 1e-11,
            "grad_env": 1e-11,
            "time_s": 1e-3,
            "color": "#7b3f98",
        },
        {
            "label": "25 kDa molecule (Fein et al.-class)",
            "mass_amu": 2.5e4,
            "size_m": 1e-7,
            "rho_env": 1e-13,
            "grad_env": 1e-13,
            "time_s": 1e-2,
            "color": "#008b5e",
        },
        {
            "label": "Levitated mesoscopic object (target scale)",
            "mass_amu": 1e9,
            "size_m": 1e-6,
            "rho_env": 1e-10,
            "grad_env": 1e-10,
            "time_s": 1e-3,
            "color": "#b35c00",
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), constrained_layout=True)
    for ax, delta, title in zip(
        axes,
        [1, 2],
        ["Representative bounds: soft benchmark $\\delta=1$", "Representative bounds: gradient benchmark $\\delta=2$"],
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, which="both", alpha=0.18)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"threshold scale $L_{\rm crit}$ [m]")
        ax.set_ylabel(r"reference rate $\Gamma_0$ [s$^{-1}$]")

        ax.axvline(l_det, color="#8b0000", linestyle="-.", linewidth=1.4, label=r"detector threshold $L_{\rm det}$")
        ax.axhline(gamma_lower, color="#8b0000", linestyle="--", linewidth=1.4, label=r"detector lower bound")

        for item in benchmarks:
            gamma_upper = _upper_bound_gamma0(
                item["mass_amu"],
                item["size_m"],
                item["rho_env"],
                item["grad_env"],
                item["time_s"],
                delta,
            )
            ax.hlines(
                gamma_upper,
                x_min,
                item["size_m"],
                colors=item["color"],
                linewidth=2.0,
                label=item["label"],
            )
            ax.vlines(
                item["size_m"],
                max(y_min, gamma_upper / 4.0),
                min(y_max, gamma_upper * 4.0),
                colors=item["color"],
                linewidth=1.0,
                alpha=0.6,
            )

        ax.text(1.6e-9, 2e-8, "below red line:\ndetector too slow", fontsize=8.8, color="#6b0000")
        ax.text(2.3e-7, 1e20, "vacuum benchmarks only\nconstrain for $L_{\\rm crit}<L_{\\rm sys}$", fontsize=8.8, color="#1f1f1f")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.035), fontsize=8.8)
    fig.suptitle("Representative experimental benchmark lines for the NVG collapse ansatz", fontsize=13)

    out_path = FIG_DIR / "fig_wave_collapse_benchmark_lines.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path1 = allowed_region_figure()
    path2 = benchmark_lines_figure()
    print(path1)
    print(path2)