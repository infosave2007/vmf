"""Focused tests for the live finite-temperature composition Hessian bridge."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_thermal_composition_hessian as bridge  # noqa: E402


class ThermalCompositionHessianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The live producer rebuilds the accepted calibration and thermal
        # integrals.  The serialized result artifact is never an input.
        cls.result = bridge.build_result()

    def test_live_pass_and_exact_bounded_coverage(self):
        result = self.result
        self.assertEqual(result["status"], bridge.STATUS_PASS)
        self.assertEqual(result["evidence_weight"], bridge.EVIDENCE_WEIGHT)
        self.assertTrue(result["physical_inputs_fixed"])
        self.assertTrue(result["no_saved_results_as_inputs"])
        self.assertEqual(result["coverage"]["reference_count"], 8)
        self.assertEqual(result["coverage"]["composition_count"], 16)
        self.assertEqual(result["coverage"]["row_count"], 32)
        self.assertTrue(result["coverage"]["all_rows_present"])
        self.assertTrue(bridge.validate_result(result))

    def test_rows_are_live_stable_and_keep_both_contact_conventions(self):
        rows = self.result["rows"]
        self.assertEqual({row["contact"] for row in rows}, set(bridge.CONTACT_ORDER))
        self.assertEqual({row["delta"] for row in rows}, set(bridge.DELTAS))
        for row in rows:
            self.assertEqual(row["status"], "CANONICAL_ROOT_OK")
            self.assertTrue(row["pass"])
            root = row["root"]
            thermo = row["thermodynamics"]
            self.assertGreater(float(root["n_n_fm3"]), 0.0)
            self.assertGreater(float(root["n_p_fm3"]), 0.0)
            self.assertLess(float(root["gap_residual_relative"]), 2e-25)
            self.assertGreater(float(thermo["H_min_eigenvalue_MeV_fm3"]), 0.0)
            self.assertGreater(float(thermo["H_det_(MeV_fm3)^2"]), 0.0)
            self.assertEqual(thermo["H_BD_MeV_fm3"][0][1], thermo["H_BD_MeV_fm3"][1][0])
            self.assertIn("pressure_MeV_fm3", thermo)
            self.assertLess(float(thermo["pressure_identity_relative"]), 2e-16)
            self.assertEqual(row["units"]["Hessian"], "MeV fm^3")

    def test_finite_difference_maxwell_static_tie_and_contact_identity(self):
        controls = self.result["controls"]
        self.assertTrue(controls["finite_difference_refinement"]["all_pass"])
        self.assertTrue(controls["static_q0_bridge"]["all_pass"])
        self.assertTrue(controls["contact_identity"]["all_pass"])
        self.assertTrue(controls["pressure_identity"]["all_pass"])
        for row in self.result["rows"]:
            fd = row["finite_difference_controls"]
            self.assertTrue(fd["all_pass"])
            self.assertTrue(fd["refinement_trend_pass"])
            self.assertEqual(len(fd["steps"]), 3)
            self.assertLessEqual(max(float(step["max_relative_error"]) for step in fd["steps"]), 3e-6)
            self.assertLessEqual(max(float(step["Maxwell_fd_relative"]) for step in fd["steps"]), 3e-6)
            self.assertLessEqual(max(float(step["envelope_max_relative_error"]) for step in fd["steps"]), 3e-6)
            self.assertLessEqual(float(row["derivative_controls"]["Maxwell_relative"]), 2e-16)

    def test_species_exchange_and_charge_coordinate_map(self):
        self.assertTrue(self.result["controls"]["species_exchange"]["all_pass"])
        saved_charge = self.result["controls"]["charge_coordinates"]
        self.assertTrue(saved_charge["all_pass"])
        public = json.loads(json.dumps(self.result))
        with mp.workdps(65):
            charge = bridge._charge_coordinate_controls(public["rows"])
        self.assertTrue(charge["all_pass"])
        self.assertEqual(charge["jacobian_BD_from_BQ"], [[mp.mpf("1"), mp.mpf("0")], [mp.mpf("1"), mp.mpf("-2")]])
        for row in charge["rows"]:
            self.assertTrue(row["finite"])
            self.assertLessEqual(float(row["max_relative"]), 2e-16)
        for row in self.result["controls"]["species_exchange"]["rows"]:
            self.assertTrue(row["pass"])

    def test_charge_coordinate_negative_control_rejects_corrupt_export(self):
        public = json.loads(json.dumps(self.result))
        public["rows"][0]["thermodynamics"]["susceptibility_BD_fm3_per_MeV"][0][0] = "999"
        with mp.workdps(65):
            corrupted = bridge._charge_coordinate_controls(public["rows"])
        self.assertFalse(corrupted["all_pass"])
        self.assertFalse(corrupted["rows"][0]["pass"])
        corrupted_result = copy.deepcopy(public)
        corrupted_result["controls"]["charge_coordinates"]["all_pass"] = True
        self.assertFalse(bridge.validate_result(corrupted_result))

    def test_frozen_contact_provenance_and_units(self):
        inputs = self.result["inputs"]
        self.assertEqual(inputs["j_MeV"], "11.13601607117819")
        self.assertEqual(inputs["n0_fm3"], "0.16")
        self.assertEqual(inputs["hbarc_MeV_fm"], "197.3269804")
        expected = 2 * mp.mpf("11.13601607117819") / mp.mpf("0.16")
        self.assertLess(abs(float(mp.mpf(inputs["C_rho_MeV_fm3"]) - expected)), 1e-12)
        self.assertEqual(inputs["species_degeneracy"], 2)
        self.assertEqual(inputs["maintained_total_degeneracy"], 4)
        self.assertTrue(all("C_rho=0" in text or "historical" in text for text in inputs["contacts"].values()))

    def test_mutation_fails_closed_without_result_json_input(self):
        mutated = copy.deepcopy(self.result)
        mutated["rows"][0]["thermodynamics"]["mu_B_MeV"] = "999"
        self.assertFalse(bridge.validate_result(mutated))
        mutated = copy.deepcopy(self.result)
        mutated["controls"]["contact_identity"]["all_pass"] = False
        self.assertFalse(bridge.validate_result(mutated))

    def test_invalid_inputs_and_failed_cli_are_explicit(self):
        with self.assertRaises(bridge.CompositionBridgeError):
            bridge._canonical_root(None, None, 0, 0, nu_n0=1, nu_p0=1, y0=1)
        with self.assertRaises(bridge.CompositionBridgeError):
            bridge._build_tree(20, nquad=32, cutoff=bridge.mp.mpf("100"), include_fd=False, include_exchange=False)
        original = bridge.build_result
        try:
            bridge.build_result = lambda: (_ for _ in ()).throw(bridge.CompositionBridgeError("synthetic failed CLI"))
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "failed.json"
                self.assertEqual(bridge.main(["--output", str(output)]), 1)
                failure = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(failure["status"], bridge.STATUS_FAIL)
                self.assertIn("synthetic failed CLI", failure["error"])
        finally:
            bridge.build_result = original

    def test_public_source_has_no_run_workspace_or_p1_dependency(self):
        source = (HERE / "nvg_thermal_composition_hessian.py").read_text(encoding="utf-8")
        self.assertNotIn("Lunacy/runs", source)
        self.assertNotIn("evidence_parent", source)
        self.assertNotIn("nvg_thermal_master_response", source)
        self.assertNotIn("nvg_thermal_master_response_results", source)


if __name__ == "__main__":
    unittest.main()
