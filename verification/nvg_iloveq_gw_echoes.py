#!/usr/bin/env python3
"""I--Love transform and echo-template calculations.

The I--Love relation is evaluated from the Lambda produced by the maintained
TOV/tidal chain.  There is no second, independent moment-of-inertia solve in
this entry point, so the relation is reported as a transform rather than a
"proof" of EOS consistency.  Echo delays come from the canonical Hayward
calculator instead of a copied target value.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


def _chain() -> dict:
    verification_dir = Path(__file__).resolve().parent
    if str(verification_dir) not in sys.path:
        sys.path.insert(0, str(verification_dir))
    from nvg_bbn_reionization import compute_eos_chain

    chain = compute_eos_chain()
    if chain.get("source") != "nvg_tidal_deformability.EOS + solve_tov_tidal":
        raise RuntimeError("I--Love sibling is not connected to the canonical EOS chain")
    if len(chain.get("pressure_grid", ())) != 120:
        raise RuntimeError("I--Love sibling must use the canonical 120-point pressure grid")
    return chain


def universal_i_from_lambda(lambda_14: float) -> dict:
    """Return the Yagi--Yunes I-bar transform for a computed Lambda."""
    if not math.isfinite(lambda_14) or lambda_14 <= 0.0:
        raise ValueError("Lambda must be a finite positive value")
    a, b, c, d, e = 1.496, 0.05951, 0.02238, -6.953e-4, 8.345e-6
    ln_l = math.log(lambda_14)
    ln_i = a + b * ln_l + c * ln_l**2 + d * ln_l**3 + e * ln_l**4
    i_bar = math.exp(ln_i)
    G_cgs, c_cgs, M_sun_g = 6.674e-8, 2.998e10, 1.989e33
    m_geom = 1.4 * M_sun_g * G_cgs / c_cgs**2
    i_cgs = i_bar * m_geom**3 / (G_cgs / c_cgs**2)
    return {"lambda": float(lambda_14), "i_bar": i_bar, "i_cgs": i_cgs}


def generate_echo_params(mass_msun: float, a_spin: float = 0.7) -> dict:
    """Combine canonical Hayward delay with a standard QNM frequency estimate."""
    verification_dir = Path(__file__).resolve().parent
    if str(verification_dir) not in sys.path:
        sys.path.insert(0, str(verification_dir))
    from nvg_gw_echoes import calculate_echo_delay

    G_cgs, c_cgs, M_sun_g = 6.674e-8, 2.998e10, 1.989e33
    if not (0.0 <= a_spin < 1.0):
        raise ValueError("spin must be in [0, 1)")
    f_qnm = c_cgs**3 / (2.0 * np.pi * G_cgs * mass_msun * M_sun_g)
    f_qnm *= 1.0 - 0.63 * (1.0 - a_spin) ** 0.3
    tau = 2.0 / (np.pi * f_qnm)
    delay = calculate_echo_delay(float(mass_msun))
    return {
        "mass_msun": float(mass_msun),
        "spin": float(a_spin),
        "f_qnm_hz": float(f_qnm),
        "tau_s": float(tau),
        "delta_t_echo_s": float(delay["delta_t_echo_s"]),
        "source": "nvg_gw_echoes.calculate_echo_delay",
    }


def compute_results() -> dict:
    chain = _chain()
    lambda_14 = float(np.interp(1.4, chain["masses"], chain["lambdas"]))
    i_love = universal_i_from_lambda(lambda_14)
    i_love.update(
        {
            "source": chain["source"],
            "pressure_grid_points": int(len(chain["pressure_grid"])),
            "selection_provenance": chain["selection_provenance"],
        }
    )
    echo_rows = [generate_echo_params(mass, spin) for mass in (30.0, 65.0, 150.0) for spin in (0.0, 0.7)]
    return {
        "i_love": {
            **i_love,
            "status": "TRANSFORM_ONLY_NO_INDEPENDENT_I_COMPARISON",
        },
        "echoes": {
            "rows": echo_rows,
            "status": "COMPUTED_MODEL_TEMPLATE_NO_DATA_COMPARISON",
        },
    }


RESULTS = compute_results()


def main() -> None:
    print("=" * 72)
    print("  NVG: I–LOVE TRANSFORM AND GW ECHO TEMPLATES")
    print("=" * 72)
    ilove = RESULTS["i_love"]
    print(f"AA. Computed EOS Lambda_1.4={ilove['lambda']:.3f}")
    print(f"    Yagi–Yunes I-bar transform={ilove['i_bar']:.4f}")
    print(f"    I from Lambda transform (no independent I solve)={ilove['i_cgs']:.3e} g cm²")
    print(f"    status={ilove['status']}")
    print(
        f"    source={ilove['source']}; pressure_grid_points={ilove['pressure_grid_points']}"
    )
    print("    canonical selection status=CONDITIONAL_IN_SAMPLE (zero independent evidence weight)")

    print("BB. Echo template parameters from canonical delay calculator:")
    for row in RESULTS["echoes"]["rows"]:
        print(
            f"    M={row['mass_msun']:.1f} M_sun, a={row['spin']:.1f}, "
            f"f_QNM={row['f_qnm_hz']:.1f} Hz, "
            f"tau={row['tau_s']*1e3:.3f} ms, "
            f"Δt={row['delta_t_echo_s']*1e3:.3f} ms"
        )
    print(f"    status={RESULTS['echoes']['status']}")
    print("No fixed Lambda/I pair or echo target is used as an acceptance claim.")


if __name__ == "__main__":
    main()
