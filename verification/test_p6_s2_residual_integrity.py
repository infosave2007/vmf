"""Semantic checks for the Phase 6 endpoint and transform repair."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_eos_existence_rescreen as existence
import nvg_iloveq_gw_echoes as iloveq_echoes
import nvg_selfbound_gate as selfbound
import nvg_tidal_deformability as tidal
import nvg_tidal_deformability_gw170817 as tidal_sibling


class ResidualEndpointTransformTests(unittest.TestCase):
    def test_single_tov_lookup_is_checked_on_valid_and_invalid_domain(self):
        path = HERE / "test_single_tov.py"
        spec = importlib.util.spec_from_file_location("single_tov_fixture", path)
        module = importlib.util.module_from_spec(spec)
        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            assert spec.loader is not None
            spec.loader.exec_module(module)
        p0, p1 = float(module.p_arr[0]), float(module.p_arr[-1])
        self.assertTrue(np.isfinite(module.eps_of_p(p0)))
        self.assertTrue(np.isfinite(module.eps_of_p(p1)))
        for pressure in (p0 - 1.0, p1 + 1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                module.eps_of_p(pressure)

    def test_existence_rescreen_pressure_lookup_is_checked(self):
        p_grid = np.array([0.1, 1.0, 10.0])
        e_grid = np.array([20.0, 100.0, 500.0])
        self.assertAlmostEqual(
            existence.pressure_to_energy_checked(1.0, p_grid, e_grid), 100.0
        )
        for pressure in (-1.0, 11.0, float("nan")):
            with self.assertRaises(ValueError):
                existence.pressure_to_energy_checked(pressure, p_grid, e_grid)
        with self.assertRaises(ValueError):
            existence.pressure_to_energy_checked(1.0, p_grid[::-1], e_grid[::-1])

    def test_existence_rescreen_crude_tov_does_not_clamp_unsupported_centers(self):
        # The scan's central-pressure grid starts at 30; this candidate EOS
        # ends below it, so no unsupported endpoint may be reused as a state.
        candidate = {
            "p": np.array([0.1, 0.5, 1.0]),
            "eps": np.array([20.0, 40.0, 80.0]),
        }
        self.assertEqual(existence.crude_tov(candidate), (None, None))

    def test_selfbound_gate_requires_a_resolved_zero_pressure_crossing(self):
        crossing = {
            "n": np.array([0.6, 0.8, 1.0]),
            "p": np.array([-2.0, 1.0, 3.0]),
            "mu": np.array([900.0, 940.0, 960.0]),
            "eps": np.array([540.0, 752.0, 960.0]),
        }
        ep_a, n_p0 = selfbound.dense_branch_p0(crossing)
        self.assertIsNotNone(ep_a)
        self.assertIsNotNone(n_p0)

        no_crossing = {key: value.copy() for key, value in crossing.items()}
        no_crossing["p"] = np.array([1.0, 2.0, 3.0])
        self.assertEqual(selfbound.dense_branch_p0(no_crossing), (None, None))

    def test_tidal_eos_paths_keep_valid_values_and_reject_invalid_pressures(self):
        for module in (tidal, tidal_sibling):
            eos = module.EOS()
            self.assertGreater(eos.pressure_max, eos.table_pressure_min)
            self.assertGreater(eos.get_eps(eos.table_pressure_min), 0.0)
            self.assertGreater(eos.get_eps(eos.pressure_max), 0.0)
            self.assertTrue(np.isfinite(eos.get_dedp(1.0e-9)))
            for pressure in (-1.0, eos.pressure_max + 1.0, float("nan")):
                with self.assertRaises(ValueError):
                    eos.get_eps(pressure)
            with self.assertRaises(ValueError):
                module.solve_tov_tidal(eos, eos.pressure_max + 1.0)

    def test_transform_routes_are_explicitly_non_independent(self):
        with self.assertRaises(ValueError):
            iloveq_echoes.universal_i_from_lambda(float("nan"))
        self.assertIn("NO_INDEPENDENT_I", iloveq_echoes.RESULTS["i_love"]["status"])
        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            iloveq_echoes.main()
        text = rendered.getvalue()
        self.assertIn("I-bar transform", text)
        self.assertIn("no independent I solve", text)
        self.assertNotIn("Universal I-bar=", text)

        for module in (tidal, tidal_sibling):
            source = (HERE / (module.__name__.split(".")[-1] + ".py")).read_text(encoding="utf-8")
            self.assertIn("I from Lambda transform (no independent I solve)", source)
            self.assertIn("Observed interval (context only; no validation)", source)
            self.assertNotIn('print(f"  I (NVG)', source)


if __name__ == "__main__":
    unittest.main()
