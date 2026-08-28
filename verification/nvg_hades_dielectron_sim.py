#!/usr/bin/env python3
"""Forward simulation of a heavy-ion rho dielectron spectrum.

The time integral is a numerical demonstration of the VMF mass law, not a fit
to HADES data. The rho mass is sourced from the maintained
``nvg_fair_hades_link.in_medium_mass`` producer. Since this repository has no
event-level HADES likelihood or detector acceptance, the result remains a
``FORWARD_MODEL_ONLY`` template.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nvg_fair_hades_link import HADRONS, in_medium_mass, n_0

# Declared physical/response inputs for the illustrative forward model.
alpha_EM = 1.0 / 137.036
hbar_c = 197.327  # MeV fm
M_rho_vac = float(HADRONS["Rho (rho)"][0])
Gamma_rho_vac = 149.1  # MeV, PDG rho width
M_rho_cur = float(HADRONS["Rho (rho)"][1])
M_rho_omega_vac = M_rho_vac - M_rho_cur
kappa_1 = 0.25  # compatibility alias; the producer owns this value
kappa_2 = 0.80

n_max = 2.5 * n_0
t_peak = 8.0
tau = 3.0
t_max = 20.0
dt = 0.1
delta_Gamma_coll = 50.0


def get_density(t: float) -> float:
    """Gaussian fireball density profile used by the forward integration."""

    return n_max * math.exp(-((t - t_peak) ** 2) / (2.0 * tau**2))


def get_in_medium_mass(n_B: float) -> float:
    """Use the canonical FAIR/HADES VMF rho mass producer."""

    return float(in_medium_mass(M_rho_vac, M_rho_cur, n_B))


def get_in_medium_width(n_B: float) -> float:
    """Illustrative collision-broadening scenario (not an inferred width)."""

    return Gamma_rho_vac + delta_Gamma_coll * (n_B / n_0)


def spectral_function(M: float | np.ndarray, M_r: float, Gamma_r: float) -> np.ndarray:
    """Breit-Wigner rho spectral function."""

    M_arr = np.asarray(M, dtype=float)
    numerator = M_r**3 * Gamma_r
    denominator = (M_arr**2 - M_r**2) ** 2 + M_r**2 * Gamma_r**2
    return numerator / denominator


def simulate_spectrum(M_array: np.ndarray, mode: str) -> np.ndarray:
    """Integrate the normalized forward dielectron rate over the fireball."""

    M_array = np.asarray(M_array, dtype=float)
    if M_array.ndim != 1 or len(M_array) < 2 or not np.all(np.diff(M_array) > 0.0):
        raise ValueError("M_array must be a strictly increasing one-dimensional grid")
    if np.any(M_array <= 0.0):
        raise ValueError("invariant masses must be positive")
    if mode not in {"vacuum", "broadening", "vmf"}:
        raise ValueError(f"Unknown simulation mode: {mode}")

    spectrum = np.zeros_like(M_array)
    t_points = np.arange(0.0, t_max, dt)
    for t in t_points:
        n_B = get_density(float(t))
        if mode == "vacuum":
            M_r, Gamma_r = M_rho_vac, Gamma_rho_vac
        elif mode == "broadening":
            M_r, Gamma_r = M_rho_vac, get_in_medium_width(n_B)
        else:
            M_r, Gamma_r = get_in_medium_mass(n_B), get_in_medium_width(n_B)
        spectrum += (spectral_function(M_array, M_r, Gamma_r) / M_array**2) * dt

    area = float(np.trapz(spectrum, M_array))
    if area <= 0.0 or not np.isfinite(area):
        raise RuntimeError(f"{mode} forward spectrum has no finite positive area")
    return spectrum / area


def compute_simulation(M_array: np.ndarray | None = None) -> dict[str, Any]:
    """Run all three spectra and return only runtime-derived summary values."""

    masses = np.linspace(300.0, 1000.0, 700) if M_array is None else np.asarray(M_array, dtype=float)
    spec_vac = simulate_spectrum(masses, "vacuum")
    spec_broad = simulate_spectrum(masses, "broadening")
    spec_vmf = simulate_spectrum(masses, "vmf")
    peaks = {
        mode: float(masses[int(np.argmax(spec))])
        for mode, spec in (("vacuum", spec_vac), ("broadening", spec_broad), ("vmf", spec_vmf))
    }
    return {
        "mass_grid_mev": masses,
        "spectra": {"vacuum": spec_vac, "broadening": spec_broad, "vmf": spec_vmf},
        "peaks_mev": peaks,
        "peak_shifts_pct": {
            mode: (peak - peaks["vacuum"]) / peaks["vacuum"] * 100.0
            for mode, peak in peaks.items()
        },
        "instantaneous_mass_at_peak_density_mev": get_in_medium_mass(n_max),
        "evidence_status": "FORWARD_MODEL_ONLY",
        "observed_likelihood": None,
        "canonical_producer": "verification/nvg_fair_hades_link.py:in_medium_mass",
        "limitation": "No traceable HADES/CBM event-level likelihood or acceptance model is present.",
    }


def main() -> dict[str, Any]:
    state = compute_simulation()
    peaks = state["peaks_mev"]
    shifts = state["peak_shifts_pct"]
    print("=" * 80)
    print("      NVG HADES/CBM DIELECTRON SPECTRUM FORWARD SIMULATION")
    print("=" * 80)
    print(f"Fireball peak density : {n_max / n_0:.2f} n_0 (declared profile)")
    print(f"Vacuum peak           : {peaks['vacuum']:.1f} MeV")
    print(f"Broadening-only peak  : {peaks['broadening']:.1f} MeV ({shifts['broadening']:+.2f}%)")
    print(f"VMF integrated peak   : {peaks['vmf']:.1f} MeV ({shifts['vmf']:+.2f}%)")
    print(f"Instantaneous VMF mass: {state['instantaneous_mass_at_peak_density_mev']:.1f} MeV")
    print("Evidence status       : FORWARD_MODEL_ONLY")
    print("Observation likelihood: unavailable; values are not fitted to HADES data")
    print("The spectra are numerical templates for a future detector comparison.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
