#!/usr/bin/env python3
"""Render the maintained S8 calculation with its actual scientific status.

The former version of this sibling entry point carried a retired ``w0/wa``
route and an empirical drag coefficient, then asserted a favourable S8 value.
The maintained calculation lives in :mod:`nvg_black_hole_entropy`; this module
is deliberately only a compatibility/reporting wrapper around that computed
route.  Its inputs are explicit toy-model values and the result is not an
independent prediction.
"""

from __future__ import annotations

import os
import sys
from typing import Any


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def compute_s8_result() -> dict[str, Any]:
    """Return the structured S8 result from the maintained computation."""

    from nvg_black_hole_entropy import compute_s8_test

    result = dict(compute_s8_test())
    # Keep the route/provenance discoverable to callers and generated reports.
    result["source"] = "nvg_black_hole_entropy.compute_s8_test"
    result["wrapper_status"] = "CONNECTED_MAINTAINED_ROUTE_NO_INDEPENDENT_PREDICTION"
    return result


def run_s8_tension_check() -> dict[str, Any]:
    """Compatibility name retained for scripts that imported this entry point."""

    return compute_s8_result()


def main() -> dict[str, Any]:
    result = run_s8_tension_check()
    print("=" * 72)
    print("  NVG COSMOLOGY: S8 TENSION ACCOUNTING (MAINTAINED ROUTE)")
    print("=" * 72)
    print(
        f"  Planck input S8 = {result['s8_planck']:.3f}; "
        f"weak-lensing input S8 = {result['s8_lensing']:.3f}"
    )
    print(f"  Input w0 = {result['w_0']:.3f}, wa = {result['w_a']:.3f}")
    print(f"  Computed sigma8(NVG)/sigma8(LCDM) = {result['sigma8_ratio']:.4f}")
    print(
        f"  Computed S8 = {result['s8_nvg']:.3f}; "
        f"tension = {result['initial_tension_sigma']:.1f} sigma -> "
        f"{result['remaining_tension_sigma']:.1f} sigma"
    )
    print(f"  Direction from computed distances: {result['direction']}")
    print(f"  STATUS: {result['status']}")
    print(f"  SOURCE: {result['source']}")
    print("  No independent S8 likelihood or resolution claim is made.")
    print("=" * 72)
    return result


if __name__ == "__main__":
    main()
