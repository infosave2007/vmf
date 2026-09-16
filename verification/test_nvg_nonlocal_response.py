"""Focused live tests for the finite-q static response calculation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import mpmath as mp

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_nonlocal_response as response  # noqa: E402
import source_complete_scaling_saturation_audit as upstream  # noqa: E402


class NonlocalResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = response.build_result()

    def test_schema_grid_hashes_and_public_scope(self):
        result = self.result
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["scheme"], response.SCHEME)
        self.assertEqual(result["status"], response.STATUS)
        self.assertEqual(result["evidence_weight"], 0.0)
        self.assertEqual(result["coverage"]["row_count"], response.EXPECTED_ROW_COUNT)
        self.assertTrue(result["coverage"]["all_rows_present"])
        self.assertEqual(result["inputs_and_scope"]["q_over_kF_grid"], list(response.Q_RATIO_GRID))
        self.assertEqual(result["inputs_and_scope"]["physical_q_MeV"], list(response.PHYSICAL_Q_GRID_MEV))
        self.assertEqual(result["inputs_and_scope"]["scales"], list(response.SCALES))
        self.assertEqual(result["source_sha256"], hashlib.sha256(response.SOURCE_PATH.read_bytes()).hexdigest())
        self.assertEqual(result["contract_sha256"], hashlib.sha256(response.CONTRACT_PATH.read_bytes()).hexdigest())
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertEqual(json.loads(payload)["coverage"], result["coverage"])
        for forbidden in (".codex", "Lunacy/", ".work", "private", "hidden"):
            self.assertNotIn(forbidden, payload)

    def test_all_rows_are_retained_and_have_actual_q_and_two_responses(self):
        rows = self.result["rows"]
        expected = {
            (family, target, scale, kind, label)
            for family, target in (("Q4", None), ("W8", "0.90"), ("W8", "0.93"))
            for scale in response.SCALES
            for kind, labels in (("q_over_kF", response.Q_RATIO_GRID),
                                 ("physical_q", response.PHYSICAL_Q_GRID_MEV))
            for label in labels
        }
        actual = {(row["family"], row["target_y"], row["scale"], row["q_kind"], row["q_label"])
                  for row in rows}
        self.assertEqual(actual, expected)
        for row in rows:
            self.assertGreaterEqual(float(row["q_MeV"]), 0)
            self.assertGreater(float(row["q_over_kF"]), -1e-15)
            self.assertIn(row["status"], {
                "STABLE_STATIC_NONLOCAL_RESPONSE", "STABLE_STATIC_TF_RESPONSE", "UNSTABLE_SCALAR_CURVATURE",
                "UNSTABLE_DENSITY_RESPONSE", "SINGULAR_DENSITY_RESPONSE",
                "SINGULAR_FULL_HESSIAN",
            })
            self.assertIn("chi_nonlocal_MeV2", row)
            self.assertIn("chi_TF_MeV2", row)
            self.assertIn("C_nonlocal_MeV4", row)
            self.assertIn("S_nonlocal_MeVminus2", row)
            self.assertIn("C_TF_MeV4", row)
            self.assertNotIn(row["Pi_vv_MeV2"], ("nan", "inf", "-inf"))

    def test_q0_continuity_lindhard_and_independent_mixed_controls(self):
        self.assertTrue(all(item["pass"] for item in self.result["q0_controls"]))
        self.assertTrue(all(item["pass"] for item in self.result["continuity_controls"]))
        self.assertTrue(all(item["pass"] for item in self.result["nonrelativistic_controls"]))
        self.assertTrue(all(item["pass"] for item in self.result["independent_mixed_quadrature_controls"]))
        # q=0 is a thermodynamic equality; the nonzero rows are not all zero.
        for background in ("Q4:n_over_n0=1", "W8:y_star=0.90", "W8:y_star=0.93"):
            zero = next(row for row in self.result["rows"]
                        if row["background_id"] == background and row["scale"] == "1"
                        and row["q_kind"] == "q_over_kF" and row["q_label"] == "0")
            finite = next(row for row in self.result["rows"]
                          if row["background_id"] == background and row["scale"] == "1"
                          and row["q_kind"] == "physical_q" and row["q_label"] == "200")
            self.assertLess(float(zero["relative_difference"]), 1e-12)
            self.assertLess(float(finite["relative_difference_percent"]), -1e-4)
        for control in self.result["q0_controls"]:
            self.assertIn("independent_bulk_fermion_hessian", control)
            self.assertIn("S", control["relative_errors"])
            self.assertIn("chi", control["relative_errors"])
        self.assertLess(abs(float(response.lindhard_shape("1e-20")) - 1), 1e-30)
        self.assertEqual(response.lindhard_shape("1"), mp.mpf("0.5"))
        self.assertEqual([item["x"] for item in self.result["lindhard_controls"][:3]], ["0.25", "1.0", "1.25"])
        self.assertTrue(all(item["pass"] for item in self.result["lindhard_controls"]))

    def test_scalar_completeness_negative_control_and_scale_effect(self):
        for control in self.result["scalar_completeness_negative_controls"]:
            self.assertTrue(control["detected_and_rejected"])
            self.assertGreater(float(control["omitted_I_MeV2"]), 0)
            self.assertGreater(float(control["wrong_response_relative_error"]), 1e-9)
        # The finite-q effect and scale comparison are both computed, not a
        # hardcoded successful-point assertion.
        q200 = [row for row in self.result["rows"]
                if row["q_kind"] == "physical_q" and row["q_label"] == "200" and row["scale"] == "1"]
        self.assertEqual(len(q200), 3)
        self.assertTrue(all(row["relative_difference"] is not None for row in q200))
        for records in self.result["scale_maps"].values():
            self.assertEqual([item["scale"] for item in records], list(response.SCALES))
            self.assertTrue(all(item["homogeneous_state_invariant"] for item in records))
            self.assertTrue(all(item["homogeneous_q0_kernel_invariant"] for item in records))

    def test_gradient_scale_is_applied_once_and_direct_response_uses_it(self):
        with mp.workdps(60):
            model = upstream.BulkModel()
            state = model.equilibrium(model.n0)
            base = response.finite_q_coefficients(model, state, "200", "1")
            scaled = response.finite_q_coefficients(model, state, "200", "1.3")
            self.assertLess(float(response._rel(scaled["Z"] / base["Z"], mp.mpf("1.3") ** 2)), 1e-45)
            candidate = response.ScaledNonlocalModel("1.3", model)
            candidate_state = candidate.state(state["n"], state["y"])
            pi = response.polarization_kernel(state["m"], state["k"], "200", d=model.d, dps=60)
            once = response._coeff(candidate, candidate_state, mp.mpf(1), pi)
            self.assertLess(float(response._rel(once["Z"], scaled["Z"])), 1e-45)
            expected = response.response_from_hessian(once, once["Z"], "200")
        row = next(item for item in self.result["rows"]
                   if item["background_id"] == "Q4:n_over_n0=1" and item["scale"] == "1.3"
                   and item["q_kind"] == "physical_q" and item["q_label"] == "200")
        self.assertLess(float(response._rel(row["chi_nonlocal_MeV2"], expected["chi"])), 1e-12)
        self.assertLess(float(response._rel(row["nonlocal_response"]["chi_MeV2"], expected["direct_chi"])), 1e-12)

    def test_guards_and_explicit_unstable_singular_statuses(self):
        for args in (("0", "1", "1"), ("1", "0", "1"), ("1", "1", "-1"),
                     ("1", "1", "NaN"), ("1", "1", "1")):
            if args == ("1", "1", "1"):
                with self.assertRaises(response.NonlocalResponseError):
                    response.polarization_kernel(*args, d="3")
            else:
                with self.assertRaises(response.NonlocalResponseError):
                    response.polarization_kernel(*args)
        unstable = response.response_from_hessian(
            {"a": "1", "b": "0", "d0": "-1", "g": "1", "h": "0", "t0": "1"}, "4", "0"
        )
        singular = response.response_from_hessian(
            {"a": "1", "b": "0", "d0": "0", "g": "1", "h": "0", "t0": "1"}, "4", "0"
        )
        self.assertEqual(unstable["status"], "UNSTABLE_SCALAR_CURVATURE")
        self.assertFalse(unstable["stable"])
        self.assertEqual(singular["status"], "SINGULAR_SCALAR_CURVATURE")
        self.assertFalse(singular["stable"])

    def test_cli_is_strict_json_and_does_not_write_result_cache(self):
        cache = HERE / "nvg_nonlocal_response_results.json"
        existed = cache.exists()
        before = cache.read_bytes() if existed else None
        completed = subprocess.run(
            [sys.executable, str(response.SOURCE_PATH), "--dps", "40"],
            check=True, capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(HERE)},
        )
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["coverage"]["row_count"], response.EXPECTED_ROW_COUNT)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(cache.exists(), existed)
        if existed:
            self.assertEqual(cache.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
