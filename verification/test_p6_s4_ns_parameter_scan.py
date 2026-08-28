"""Focused regression tests for the parameter-scan EOS construction contract."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_eos_beta_css_softening as soft
import nvg_ns_parameter_scan as scan


class ParameterScanEOSContractTests(unittest.TestCase):
    def test_hybrid_scan_path_initializes_strict_tidal_bounds(self):
        baseline = soft.build_baseline_arrays()
        self.assertIsNotNone(baseline)
        hybrid = soft.build_css_hybrid_eos(
            baseline,
            n_trans_ratio=scan.CANON[0],
            delta_eps_ratio=scan.CANON[1],
            cs2_q=1.0 / 3.0,
        )
        self.assertIsNotNone(hybrid)

        family = scan.star_family(hybrid)
        self.assertTrue(all(np.isfinite(value) for value in family))
        self.assertGreater(family[0], 2.0)
        self.assertGreater(family[1], 11.2)
        self.assertLess(family[1], 13.2)

    def test_invalid_hybrid_arrays_fail_before_tov(self):
        with self.assertRaises(ValueError):
            scan.star_family(
                {
                    "p_sorted": np.array([0.1, 0.1]),
                    "e_sorted": np.array([20.0, 21.0]),
                }
            )


if __name__ == "__main__":
    unittest.main()
