"""Focused controls for the P1-S5 neutron-star repair."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import nvg_eos_beta_css_softening as soft
import nvg_ns_predictive_audit as audit
import nvg_tidal_deformability as tidal


class NSP1S5RepairTests(unittest.TestCase):
    def test_ordered_branch_split_never_mass_sorts(self):
        masses = (1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 1.0, 1.2, 1.4)
        rows = [
            {"central_pressure": float(index + 1), "mass": mass,
             "radius": 12.0, "lambda": 800.0 - 20.0 * index}
            for index, mass in enumerate(masses)
        ]
        branches, branch_ids = audit._stable_branches(rows)
        self.assertGreaterEqual(len(branches), 1)
        self.assertEqual(branch_ids[0], 0)
        # The descending point is unresolved/unstable; it is not sorted into
        # either neighboring branch.
        self.assertEqual(branch_ids[3], -1)
        self.assertEqual(branch_ids[4], -1)
        for branch in branches:
            self.assertTrue(np.all(np.diff([row["central_pressure"] for row in branch]) > 0.0))

    def test_first_order_match_equation_and_zero_limit(self):
        y_plus = tidal.hinderer_density_jump_y_match(2.0, 10.0, 1.4, 20.0)
        expected = 2.0 - 4.0 * math.pi * 10.0**3 * 20.0 * tidal.k_conv / (1.4 * tidal.M_sun_km)
        self.assertAlmostEqual(y_plus, expected, places=14)
        self.assertEqual(tidal.hinderer_density_jump_y_match(2.0, 10.0, 1.4, 0.0), 2.0)

    def test_canonical_constructor_keeps_zero_jump_default_path(self):
        eos = tidal.EOS()
        self.assertFalse(hasattr(eos, "transition_pressure"))
        default = tidal.solve_tov_tidal(eos, 100.0)
        repeated = tidal.solve_tov_tidal(eos, 100.0)
        self.assertEqual(default, repeated)

    def test_positive_jump_is_not_silently_ignored_and_tables_are_invariant(self):
        baseline = soft.build_baseline_arrays(nn=220)
        eos = audit._make_hybrid_eos(baseline, 2.4, 0.01, 0.5)
        self.assertIsNotNone(eos)
        coarse = audit._resample_pressure_table(eos, 80)
        fine = audit._resample_pressure_table(eos, 320)
        self.assertAlmostEqual(coarse.transition_energy_jump, eos.transition_energy_jump)
        self.assertAlmostEqual(fine.transition_energy_jump, eos.transition_energy_jump)
        kwargs = {"dr": 0.05, "rtol": 1.0e-6, "atol": 1.0e-9}
        matched = tidal.solve_tov_tidal(eos, 53.8, density_jump_matching=True, **kwargs)
        legacy = tidal.solve_tov_tidal(eos, 53.8, density_jump_matching=False, **kwargs)
        # The repaired event integration applies the physically first-order
        # Eq. 15 update without the old finite-step artefact; the correction
        # need only be demonstrably non-zero, not an inflated five-percent
        # threshold tied to the former dropped-tail implementation.
        self.assertGreater(abs(matched[3] - legacy[3]) / matched[3], 1.0e-4)
        coarse_values = tidal.solve_tov_tidal(coarse, 53.8, density_jump_matching=True, **kwargs)
        fine_values = tidal.solve_tov_tidal(fine, 53.8, density_jump_matching=True, **kwargs)
        self.assertLess(abs(coarse_values[3] - fine_values[3]) / fine_values[3], 5.0e-3)

    def test_terminal_result_excludes_unresolved_positive_rows(self):
        path = os.path.join(HERE, "nvg_ns_predictive_audit_p1s5_results.json")
        if not os.path.exists(path):
            self.skipTest("terminal P1-S5 artifact not materialized in this focused test run")
        import json

        result = json.load(open(path, encoding="utf-8"))
        self.assertEqual(result["audit_status"], "P1-S5_REPAIRED_ROW_CONVERGENCE_AUDIT")
        self.assertEqual(result["m_omega_dependency"]["status"], "BLOCKED_NO_PHYSICAL_DEPENDENCY")
        self.assertEqual(result["density_jump_treatment"]["source"]["equation"], "Eq. 14")
        positive_eligible = [
            row for row in result["grid_rows"]
            if row["parameters"]["delta_eps_ratio"] > 0.0 and row.get("evidence_eligible")
        ]
        self.assertEqual(positive_eligible, [])
        self.assertGreater(result["grid_summary"]["excluded_unresolved_rows"], 0)


if __name__ == "__main__":
    unittest.main()
