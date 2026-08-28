#!/usr/bin/env python3
"""Runtime forward calculations for CMB, EHT and the PBH ladder.

The former version printed fixed values as if they were Planck/EHT/JWST
measurements.  These functions deliberately separate a theory/forward curve
from an observational likelihood.  The PBH ladder is computed from the
declared cyclic relation; its abundance is not inferred here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

EVIDENCE_STATUS = {
    "cmb_spectrum": "FORWARD_ONLY_NO_CMB_LIKELIHOOD",
    "eht_shadow": "FORMAL_EXTERIOR_NULL_NO_EHT_LIKELIHOOD",
    "pbh_ladder": "DERIVED_THEORY_LADDER_NO_ABUNDANCE_CALCULATION",
}


def primordial_power(k: np.ndarray | float, *, amplitude: float = 2.1e-9,
                     spectral_index: float = 0.965, cutoff: float | None = None) -> np.ndarray:
    """Return a primordial spectrum with an optional IR cutoff."""

    kval = np.asarray(k, dtype=float)
    if np.any(~np.isfinite(kval)) or np.any(kval <= 0.0):
        raise ValueError("wavenumbers must be finite and positive")
    out = float(amplitude) * (kval / 0.05) ** (float(spectral_index) - 1.0)
    if cutoff is not None:
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("cutoff must be positive and finite")
        out = out * (1.0 - np.exp(-(kval / float(cutoff)) ** 2))
    return out


def hayward_shadow_shift(l_over_rs: float = 1.0e-35) -> dict[str, float]:
    """Compute the leading exterior null scaling (no image likelihood)."""

    if not np.isfinite(l_over_rs) or l_over_rs < 0.0:
        raise ValueError("l_over_rs must be finite and non-negative")
    standard_photon_radius = 3.0
    fractional_shift = -(4.0 / 27.0) * float(l_over_rs) ** 2
    return {
        "l_over_rs": float(l_over_rs),
        "photon_radius_gr": standard_photon_radius,
        "photon_radius_fractional_shift": fractional_shift,
        "shadow_radius_fractional_shift": fractional_shift,
    }


def pbh_mass(cycle: int | float, *, base_mass: float | None = None,
             growth: float | None = None) -> float:
    """Evaluate the canonical discrete PBH ladder (or an explicit sensitivity).

    With the default arguments this delegates to the maintained
    ``nvg_pbh_mass_spectrum.get_pbh_mass`` producer.  Optional base/growth
    values are retained only for a transparent theory-level sensitivity and
    never acquire abundance or observational meaning.
    """

    if not np.isfinite(cycle):
        raise ValueError("ladder cycle must be finite")
    if base_mass is None and growth is None:
        if float(cycle).is_integer():
            from nvg_pbh_mass_spectrum import get_pbh_mass

            return float(get_pbh_mass(int(cycle)))
        base_mass, growth = 0.38, 4.0
    else:
        base_mass = 0.38 if base_mass is None else base_mass
        growth = 4.0 if growth is None else growth
    if not np.isfinite(base_mass) or not np.isfinite(growth):
        raise ValueError("ladder inputs must be finite")
    if base_mass <= 0.0 or growth <= 0.0:
        raise ValueError("ladder base and growth must be positive")
    return float(base_mass * growth ** float(cycle))


def compute_observables() -> dict[str, Any]:
    k_values = np.array([1e-5, 5e-5, 1e-4, 3.2e-4, 1e-3, 1e-2, 1e-1], dtype=float)
    cutoff = 3.2e-4
    lcdm = primordial_power(k_values)
    vmf = primordial_power(k_values, cutoff=cutoff)
    ladder_cycles = (-28, -25, -21, -15, 0, 10)
    ladder = [{"cycle": int(c), "mass_msun": pbh_mass(c)} for c in ladder_cycles]
    return {
        "cmb_spectrum": {
            "k_mpc": k_values,
            "lcdm": lcdm,
            "cutoff": cutoff,
            "cutoff_spectrum": vmf,
            "status": EVIDENCE_STATUS["cmb_spectrum"],
            "missing": "Boltzmann/Planck likelihood with nuisance and cosmic-variance treatment",
        },
        "eht_shadow": {
            **hayward_shadow_shift(),
            "status": EVIDENCE_STATUS["eht_shadow"],
            "missing": "resolved EHT image likelihood and a parameterized radiative-transfer model",
        },
        "pbh_ladder": {
            "rows": ladder,
            "status": EVIDENCE_STATUS["pbh_ladder"],
            "missing": "formation abundance and lensing/accretion likelihood",
        },
    }


def main() -> dict[str, Any]:
    state = compute_observables()
    print("=" * 72)
    print("  NVG: ADVANCED OBSERVABLES II (FORWARD / EVIDENCE-STATUS LEDGER)")
    print("=" * 72)
    cmb = state["cmb_spectrum"]
    print(f"CMB cutoff curve: k_c={cmb['cutoff']:.3g} Mpc^-1; status={cmb['status']}")
    eht = state["eht_shadow"]
    print(f"EHT exterior null scaling: delta_shadow={eht['shadow_radius_fractional_shift']:.3e}; status={eht['status']}")
    for row in state["pbh_ladder"]["rows"]:
        print(f"PBH ladder N={row['cycle']:>3}: {row['mass_msun']:.4e} M_sun")
    print(f"PBH status={state['pbh_ladder']['status']}; no abundance/DM fraction is inferred.")
    print("No independent Planck, EHT or JWST confirmation is claimed by this entry point.")
    print("=" * 72)
    return state


if __name__ == "__main__":
    main()
