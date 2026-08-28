#!/usr/bin/env python3
"""Forward simulation of the NVG rho-meson dielectron spectrum.

The calculation is deliberately kept as a forward model.  It reuses the
maintained in-medium mass law from :mod:`nvg_fair_hades_link` and computes the
detector-smearing integral at runtime.  No HADES/CBM event sample or detector
likelihood is bundled in this repository, so the output cannot be promoted to
an observation or an observational result.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from nvg_fair_hades_link import HADRONS, in_medium_mass, n_0


# Declared literature/experimental inputs.  They are not fitted to a HADES
# spectrum and are used only to define the forward response.
MASS_VAC = float(HADRONS["Rho (rho)"][0])
MASS_CURRENT = float(HADRONS["Rho (rho)"][1])
WIDTH_VAC = 149.0  # MeV, PDG rho width
WIDTH_MEDIUM = 250.0  # MeV, illustrative collisional broadening scenario
RESOLUTION_SIGMA = 15.0  # MeV, detector-response benchmark
PEAK_DENSITY_RATIO = 2.0


def breit_wigner(m: np.ndarray, m0: float, gamma: float) -> np.ndarray:
    """Relativistic Breit-Wigner distribution."""

    m = np.asarray(m, dtype=float)
    num = m * m0 * gamma
    den = (m**2 - m0**2) ** 2 + (m0 * gamma) ** 2
    return num / den


def gaussian(m: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian detector response."""

    m = np.asarray(m, dtype=float)
    return np.exp(-0.5 * ((m - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def convolve_spectrum(mass_array: np.ndarray, spectrum: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve a spectrum with a Gaussian response using the grid integral."""

    mass_array = np.asarray(mass_array, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    if mass_array.ndim != 1 or spectrum.shape != mass_array.shape or len(mass_array) < 2:
        raise ValueError("mass_array and spectrum must be one-dimensional and have >=2 points")
    if not np.all(np.isfinite(mass_array)) or not np.all(np.isfinite(spectrum)) or sigma <= 0.0:
        raise ValueError("spectrum inputs must be finite and sigma positive")
    dm = float(np.mean(np.diff(mass_array)))
    if not np.allclose(np.diff(mass_array), dm, rtol=1e-8, atol=1e-12) or dm <= 0.0:
        raise ValueError("mass_array must be strictly increasing and uniformly spaced")
    weights = np.exp(-0.5 * ((mass_array[:, None] - mass_array[None, :]) / sigma) ** 2)
    weights /= sigma * np.sqrt(2.0 * np.pi)
    return weights @ spectrum * dm


def compute_forward_model(
    *,
    peak_density_ratio: float = PEAK_DENSITY_RATIO,
    vacuum_fraction: float = 0.30,
    mass_grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute detector-level forward spectra and runtime-derived peak values.

    ``in_medium_mass`` is the canonical producer for the VMF rho mass.  The
    mixture fraction, broadening and detector resolution remain explicit
    forward-model assumptions; they are not inferred from data here.
    """

    if peak_density_ratio < 0.0 or not 0.0 <= vacuum_fraction <= 1.0:
        raise ValueError("density ratio must be non-negative and vacuum_fraction in [0, 1]")
    m_range = np.linspace(200.0, 1200.0, 500) if mass_grid is None else np.asarray(mass_grid, dtype=float)
    if m_range.ndim != 1 or len(m_range) < 2:
        raise ValueError("mass_grid must be a one-dimensional array with >=2 points")

    mass_medium = float(in_medium_mass(MASS_VAC, MASS_CURRENT, peak_density_ratio * n_0))
    spec_vac = breit_wigner(m_range, MASS_VAC, WIDTH_VAC)
    spec_med = breit_wigner(m_range, mass_medium, WIDTH_MEDIUM)
    spec_mixed = vacuum_fraction * spec_vac + (1.0 - vacuum_fraction) * spec_med
    spec_observed = convolve_spectrum(m_range, spec_mixed, RESOLUTION_SIGMA)
    peak_observed = float(m_range[int(np.argmax(spec_observed))])
    return {
        "mass_grid_mev": m_range,
        "vacuum_spectrum": spec_vac,
        "medium_spectrum": spec_med,
        "mixed_spectrum": spec_mixed,
        "observed_spectrum": spec_observed,
        "mass_vacuum_mev": MASS_VAC,
        "mass_medium_mev": mass_medium,
        "peak_observed_mev": peak_observed,
        "peak_density_ratio": float(peak_density_ratio),
        "vacuum_fraction": float(vacuum_fraction),
        "evidence_status": "FORWARD_MODEL_ONLY",
        "observed_likelihood": None,
        "canonical_producer": "verification/nvg_fair_hades_link.py:in_medium_mass",
        "limitation": "No traceable HADES/CBM event likelihood is present in the repository.",
    }


def main() -> dict[str, Any]:
    state = compute_forward_model()
    print("=" * 70)
    print(" NVG DETECTOR-LEVEL FORWARD MODEL (HADES / CBM / NICA)")
    print("=" * 70)
    print(f"  Vacuum rho mass       : {state['mass_vacuum_mev']:.1f} MeV (declared input)")
    print(f"  VMF rho mass at {state['peak_density_ratio']:.1f} n_0: {state['mass_medium_mev']:.1f} MeV (canonical law)")
    print(f"  Collisional width     : {WIDTH_MEDIUM:.1f} MeV (forward-model scenario)")
    print(f"  Detector sigma        : {RESOLUTION_SIGMA:.1f} MeV (response benchmark)")
    print(f"  Integrated peak       : {state['peak_observed_mev']:.1f} MeV (runtime convolution)")
    print("  Evidence status       : FORWARD_MODEL_ONLY")
    print("  Observation likelihood: unavailable; no HADES/CBM event data are bundled")
    print("  The result is a falsifiable template, not an observational result.")
    print("=" * 70)
    return state


if __name__ == "__main__":
    main()
