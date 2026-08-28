#!/usr/bin/env python3
"""New-direction checks with real-input boundaries.

The old entry point injected JWST/ATNF/LIGO populations and wrote a public
report describing those simulations as observations.  It now reuses the
owned runtime ledgers and, where a real GWTC row is available, computes only a
forward echo-delay estimate; no strain likelihood or p-value is manufactured.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_STATUS = {
    "jwst": "RETIRED_NO_OBJECT_LEVEL_JWST_LIKELIHOOD",
    "pulsar": "RETIRED_NO_ATNF_COOLING_LIKELIHOOD",
    "echo": "FORWARD_ONLY_REAL_GWTC_MASS_NO_STRAIN_LIKELIHOOD",
}


def cosmic_age_Gyr(z: float) -> float:
    if not np.isfinite(z) or z < -1.0:
        raise ValueError("redshift must be finite and >= -1")
    h0_gyr = 67.4 * 1.022689e-3
    factor = 2.0 / (3.0 * h0_gyr * math.sqrt(0.685))
    x = 0.685 / (0.315 * (1.0 + float(z)) ** 3)
    return float(factor * math.log(math.sqrt(x) + math.sqrt(1.0 + x)))


def required_f_edd(M_seed: float, M_obs: float, t_start: float, t_end: float) -> float:
    if min(M_seed, M_obs) <= 0.0:
        raise ValueError("masses must be positive")
    dt = float(t_end) - float(t_start)
    return 0.0 if dt <= 0.0 else float(math.log(M_obs / M_seed) * 0.1 * 0.045 / (0.9 * dt))


def run_jwst_verification() -> dict[str, Any]:
    # Reuse the abundance calculation from the sibling advanced suite; it is
    # deliberately marked as a calibration/retirement and contains no mock
    # object catalogue.
    from nvg_advanced_calculations import run_jwst_smbh_check

    result = run_jwst_smbh_check()
    return {"status": EVIDENCE_STATUS["jwst"], "abundance": result,
            "missing": result["missing"]}


def run_pulsar_verification() -> dict[str, Any]:
    from nvg_pulsar_population_test import run_pulsar_population_test

    result = run_pulsar_population_test()
    return {"status": EVIDENCE_STATUS["pulsar"], "catalog": result["catalog"],
            "missing": result["missing"]}


def _gwtc150914_mass() -> tuple[float | None, str]:
    path = os.path.join(HERE, "data", "gwtc_events.csv")
    if not os.path.exists(path):
        return None, path
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("commonName") != "GW150914":
                continue
            for key in ("final_mass_source", "total_mass_source"):
                try:
                    value = float(row[key])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0.0:
                    return value, path
    return None, path


def run_echo_verification() -> dict[str, Any]:
    mass, catalog_path = _gwtc150914_mass()
    if mass is None:
        return {"mass_msun": None, "delay_s": None, "status": "RETIRED_NO_GWTC_MASS_OR_STRAIN",
                "catalog": catalog_path, "missing": "GWTC event mass and open strain"}
    from nvg_gw_echo_prediction import calculate_kerr_echo_delay

    # The spin is a declared literature input for GW150914; no fit is done.
    spin = 0.67
    delay = float(calculate_kerr_echo_delay(mass, spin))
    return {
        "mass_msun": mass,
        "spin": spin,
        "delay_s": delay,
        "status": EVIDENCE_STATUS["echo"],
        "catalog": catalog_path,
        "missing": "open detector strain, residual subtraction and a calibrated search likelihood",
    }


def main() -> dict[str, Any]:
    state = {"jwst": run_jwst_verification(), "pulsar": run_pulsar_verification(),
             "echo": run_echo_verification()}
    print("=" * 80)
    print("  NVG NEW DIRECTIONS (RUNTIME EVIDENCE-STATUS LEDGER)")
    print("=" * 80)
    print(f"JWST: status={state['jwst']['status']}")
    print(f"Pulsar: status={state['pulsar']['status']}")
    echo = state["echo"]
    print(f"GWTC echo delay: mass={echo['mass_msun']}, delay={echo['delay_s']}; status={echo['status']}")
    print("No synthetic catalog, injected noise, public report, p-value or confirmation is emitted.")
    print("=" * 80)
    return state


if __name__ == "__main__":
    main()
