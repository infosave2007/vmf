"""Focused compatibility proof for the NS g-mode trapezoidal integration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_ns_g_modes as g_modes


class P6S3NSGModesCompatibilityTests(unittest.TestCase):
    def test_integration_helper_matches_available_numpy_trapezoid(self):
        cases = (
            (
                np.array([0.0, 0.25, 1.5, 2.0, 4.0]),
                np.array([1.0, -2.0, 3.0, 0.5, 5.0]),
            ),
            (
                np.array([4.0, 3.0, 1.0, -1.0]),
                np.array([2.0, 0.0, 1.0, 3.0]),
            ),
        )
        for coordinates, values in cases:
            with self.subTest(coordinates=coordinates.tolist()):
                expected = np.trapz(values, coordinates)
                actual = g_modes.trapezoidal_integral(values, coordinates)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
                if hasattr(np, "trapezoid"):
                    np.testing.assert_allclose(
                        np.trapezoid(values, coordinates), expected, rtol=0.0, atol=0.0
                    )

    def test_owned_source_guards_newer_numpy_api(self):
        source = (HERE / "nvg_ns_g_modes.py").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"np\.trapezoid\s*\(")
        self.assertIn('getattr(np, "trapezoid", None)', source)
        self.assertIn("integration = np.trapz", source)


if __name__ == "__main__":
    unittest.main()
