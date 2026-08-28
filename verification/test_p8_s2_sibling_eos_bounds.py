"""Semantic checks for the Phase 8 sibling EOS-bound repair."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_eos_fork_a as fork_a
import nvg_eos_fork_b as fork_b
import nvg_eos_fork_b_nl as fork_b_nl
import nvg_melting_identifiability as melting
import nvg_sensitivity_analysis as sensitivity


class SiblingEOSBoundsSemanticTests(unittest.TestCase):
    def test_sensitivity_lookup_and_central_pressure_bounds(self):
        eos = sensitivity.UnifiedEOS(859.0)
        self.assertTrue(np.isfinite(eos.get_eps(eos.p_arr[0])))
        self.assertTrue(np.isfinite(eos.get_eps(eos.p_arr[-1])))
        self.assertEqual(eos.get_eps(0.0), 0.0)  # explicit vacuum continuation
        for pressure in (-1.0, float("nan"), eos.p_arr[-1] + 1.0):
            with self.assertRaises(ValueError):
                eos.get_eps(pressure)
        with self.assertRaises(ValueError):
            sensitivity.solve_tov(eos, eos.p_arr[-1] + 1.0)

    def test_melting_lookup_and_central_pressure_bounds(self):
        eos = melting.EOS(melting.g_author)
        self.assertTrue(np.isfinite(eos.eps(eos.p[0])))
        self.assertTrue(np.isfinite(eos.eps(eos.p[-1])))
        self.assertEqual(eos.eps(0.0), 0.0)  # explicit vacuum continuation
        self.assertTrue(np.isfinite(eos.cs2_of_P(eos.p[-1])))
        self.assertTrue(np.isfinite(eos.P_of_eps(eos.e[-1])))
        self.assertEqual(eos.P_of_eps(0.0), 0.0)  # explicit vacuum continuation
        for pressure in (-1.0, float("nan"), eos.p[-1] + 1.0):
            with self.assertRaises(ValueError):
                eos.eps(pressure)
        with self.assertRaises(ValueError):
            eos.cs2_of_P(eos.p[-1] + 1.0)
        with self.assertRaises(ValueError):
            eos.P_of_eps(eos.e[-1] + 1.0)
        with self.assertRaises(ValueError):
            melting.structure(eos, eos.p[-1] + 1.0)

    def test_fork_tov_lookup_paths_reject_unsupported_pressures(self):
        table_p = np.array([1.0, 2.0, 3.0])
        table_e = np.array([10.0, 20.0, 30.0])
        for module in (fork_a, fork_b, fork_b_nl):
            self.assertEqual(module.pressure_to_energy_checked(0.0, table_p, table_e), 0.0)
            self.assertEqual(module.pressure_to_energy_checked(2.0, table_p, table_e), 20.0)
            self.assertTrue(np.isfinite(
                module.pressure_to_energy_checked(table_p[-1], table_p, table_e)
            ))
            for pressure in (-1.0, float("nan"), table_p[-1] + 1.0):
                with self.assertRaises(ValueError):
                    module.pressure_to_energy_checked(pressure, table_p, table_e)


if __name__ == "__main__":
    unittest.main()
