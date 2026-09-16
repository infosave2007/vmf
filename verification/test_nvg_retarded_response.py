"""Focused live controls for the bounded retarded-response calculation."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_retarded_response as response  # noqa: E402


class RetardedResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = response.build_result()

    def test_schema_scope_grid_and_complex_serialization(self):
        result = self.result
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["scheme"], response.SCHEME)
        self.assertEqual(result["status"], response.STATUS)
        self.assertEqual(result["evidence_weight"], 0.0)
        self.assertEqual(result["coverage"]["row_count"], response.EXPECTED_ROW_COUNT)
        self.assertEqual(result["coverage"]["pole_case_count"], response.EXPECTED_POLE_CASES)
        self.assertEqual(result["inputs_and_scope"]["scales"], list(response.SCALES))
        self.assertEqual(result["inputs_and_scope"]["q_MeV"], list(response.Q_GRID_MEV))
        self.assertEqual(result["inputs_and_scope"]["frequency_fractions_of_live_ph_edge"], list(response.FREQUENCY_FRACTIONS))
        self.assertEqual(result["source_sha256"], hashlib.sha256(response.SOURCE_PATH.read_bytes()).hexdigest())
        self.assertEqual(result["contract_sha256"], hashlib.sha256(response.CONTRACT_PATH.read_bytes()).hexdigest())
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertEqual(json.loads(payload)["coverage"], result["coverage"])
        for forbidden in (".codex", ".work", "private", "hidden"):
            self.assertNotIn(forbidden, payload)
        for row in result["rows"]:
            self.assertEqual(set(row["Pi"]), {"vv_MeV2", "vs_MeV2", "ss_MeV2"})
            for value in row["Pi"].values():
                self.assertEqual(set(value), {"real", "imag"})
                float(value["real"])
                float(value["imag"])
            self.assertNotIn("stable", row["response"])
            self.assertFalse(row["response"]["physical_stability_assessed"])
            self.assertEqual(len(row["response"]["normalization_coordinate_scales"]), 4)
            for scale in row["response"]["normalization_coordinate_scales"]:
                self.assertGreater(float(scale), 0.0)
            self.assertIn("determinant_raw_MeV6", row["response"])
            self.assertIn("determinant_normalized_dimensionless", row["response"])

    def test_all_48_rows_are_retained_with_cut_and_response(self):
        expected = {
            (family, target, scale, q, fraction)
            for family, target in (("Q4", None), ("W8", "0.90"), ("W8", "0.93"))
            for scale in response.SCALES for q in response.Q_GRID_MEV
            for fraction in response.FREQUENCY_FRACTIONS
        }
        actual = {(r["family"], r["target_y"], r["scale"], r["q_MeV"], r["omega_fraction"]) for r in self.result["rows"]}
        self.assertEqual(actual, expected)
        for row in self.result["rows"]:
            self.assertIn(row["status"], {"RETARDED_RESPONSE_OFF_POLE", "SINGULAR_FULL_HESSIAN", "SINGULAR_SCALAR_ELIMINATION"})
            self.assertIn("chi_MeV2", row)
            self.assertIn("absorptive_Im_chi_over_pi_MeV2", row)
            self.assertNotEqual(row["Pi"]["vv_MeV2"]["real"], "nan")
        self.assertTrue(any(r["inside_particle_hole_continuum"] for r in self.result["rows"]))
        self.assertTrue(any(not r["inside_particle_hole_continuum"] and r["omega_fraction"] == "1.25" for r in self.result["rows"]))

    def test_static_imaginary_and_direct_schur_controls(self):
        self.assertEqual(len(self.result["static_limit_controls"]), 12)
        self.assertTrue(all(c["pass"] for c in self.result["static_limit_controls"]))
        self.assertTrue(all(c["pass"] for c in self.result["analytic_imaginary_controls"]))
        self.assertEqual({c["region"] for c in self.result["analytic_imaginary_controls"]}, {"inside_particle_hole_cut", "above_particle_hole_cut"})
        self.assertTrue(all(c["pass"] for c in self.result["off_pole_direct_schur_controls"]))
        self.assertTrue(all(float(c["direct_vs_schur_relative_error"]) < 2e-8 for c in self.result["off_pole_direct_schur_controls"]))

    def test_regulator_conjugation_conservation_and_ward_controls(self):
        self.assertEqual([c["eta_MeV"] for c in self.result["controls"]["regulator_controls"]], ["0.40000000000000002", "0.20000000000000001", "0.10000000000000001"])
        errors = [float(c["relative_to_boundary"][0]) for c in self.result["controls"]["regulator_controls"]]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertTrue(self.result["controls"]["negative_frequency_control"]["pass"])
        self.assertTrue(self.result["controls"]["q0_conservation_controls"]["pass"])
        self.assertTrue(all(c["pass"] for c in self.result["controls"]["ward_controls"]))
        self.assertTrue(all(float(c["ph_only_Ward_violation"]) > 1e-4 for c in self.result["controls"]["ward_controls"]))

    def test_longitudinal_mixing_and_scale_negative_controls(self):
        neg = self.result["controls"]["missing_longitudinal_or_mixing_control"]
        self.assertTrue(neg["detected_and_rejected"])
        self.assertGreater(float(neg["missing_A_L_relative_error"]), 1e-6)
        self.assertGreater(float(neg["missing_h_relative_error"]), 1e-6)
        rows = [r for r in self.result["rows"] if r["background_id"].startswith("Q4:") and r["q_MeV"] == "100" and r["omega_fraction"] == "0.75"]
        self.assertEqual(len(rows), 2)
        z1 = rows[0]["response"]["H"][1][1]
        z13 = rows[1]["response"]["H"][1][1]
        self.assertNotEqual(z1["real"], z13["real"])
        self.assertTrue(all(c["accepted"] is False for c in self.result["controls"]["pole_validation_negative_controls"].values()))

    def test_pole_coverage_and_q4_resolved_w8_bounded_none(self):
        cases = self.result["pole_cases"]
        self.assertEqual(len(cases), 12)
        q4 = [c for c in cases if c["background_id"].startswith("Q4:")]
        w8 = [c for c in cases if c["background_id"].startswith("W8:")]
        self.assertTrue(all(c["status"] == "RESOLVED_DENSITY_POLE" for c in q4))
        self.assertTrue(all(c["grid_root_converged"] for c in q4))
        self.assertTrue(all(r["accepted_resolved_pole"] and r["nonzero_density_overlap"] for c in q4 for r in c["roots"]))
        self.assertTrue(all(r["positive_finite_weight"] and r["residue_step_converged"] and not r["rejection_reasons"] for c in q4 for r in c["roots"]))
        self.assertTrue(all(float(r["normalized_determinant_residual"]) < response.POLE_DETERMINANT_RESIDUAL_TOLERANCE for c in q4 for r in c["roots"]))
        self.assertTrue(all(float(r["determinant_identity_relative_error"]) < response.POLE_DETERMINANT_IDENTITY_TOLERANCE for c in q4 for r in c["roots"]))
        self.assertTrue(all(c["status"] == "NONE_RESOLVED_IN_SCANNED_INTERVAL" for c in w8))
        self.assertTrue(all(c["no_root_is_not_absence_theorem"] for c in cases))
        self.assertTrue(all(c["grids"]["33"]["node_count"] == 33 and c["grids"]["65"]["node_count"] == 65 for c in cases))

    def test_guards_and_q0_api(self):
        base = response.upstream.BulkModel()
        state = base.equilibrium(base.n0)
        for mass, kf, q, degeneracy in ((0, 260, 100, 4), (300, 0, 100, 4), (300, 260, -1, 4), (300, 260, 100, 3)):
            with self.assertRaises(response.RetardedResponseError):
                response.polarization_kernel(mass, kf, q, d=degeneracy)
        dynamic_pi = response.polarization_kernel(state["m"], state["k"], 0, 1 + 0.2j)
        self.assertEqual(dynamic_pi[:2], (0j, 0j))
        self.assertNotEqual(dynamic_pi[2], 0j)
        with self.assertRaises(response.RetardedResponseError):
            response._coefficients(base, state, (0j, 1j, 1j), 1.0)
        zero = response.retarded_response(base, state, 0, 1 + 0.2j)
        self.assertEqual(zero["status"], "Q0_CONSERVATION_ZERO")
        self.assertEqual(zero["chi"], 0j)
        self.assertEqual(zero["Pi"], dynamic_pi)
        self.assertNotEqual(zero["Pi"][2], 0j)
        for invalid in (1 - 0.2j, (float("nan"), 0.0), (1.0, float("inf"))):
            with self.assertRaises(response.RetardedResponseError):
                response.polarization_kernel(300, 260, 100, invalid)
        with self.assertRaises(response.RetardedResponseError):
            response.polarization_kernel(300, 260, 100, 1 + 0.2j, eta=-0.1)
        high_energy = response.polarization_kernel(300, 260, 100, 2500.0)
        self.assertTrue(all(response._finite_complex(x) for x in high_energy))

    def test_pole_predicate_and_payload_validation_reject_mutations(self):
        accepted, reasons = response._pole_acceptance(
            grid_match=True, outside_cut=True, bracket_confirmed=True, finite_elimination_factors=True,
            residue_converged=True, positive_finite_weight=False, null_residual=0.0,
            determinant_residual=0.0, determinant_identity_error=0.0, density_overlap=1.0)
        self.assertFalse(accepted)
        self.assertIn("residue_weight_not_positive_finite", reasons)
        accepted, reasons = response._pole_acceptance(
            grid_match=False, outside_cut=True, bracket_confirmed=True, finite_elimination_factors=True,
            residue_converged=True, positive_finite_weight=True, null_residual=1.0,
            determinant_residual=1.0, determinant_identity_error=0.0, density_overlap=1.0)
        self.assertFalse(accepted)
        self.assertIn("dual_grid_not_converged", reasons)
        self.assertIn("full_normalized_null_residual_too_large", reasons)
        self.assertTrue(response.validate_result(self.result))
        for mutate in (
            lambda x: x["rows"].clear(),
            lambda x: x["pole_cases"].clear(),
            lambda x: x["rows"][0]["Pi"]["vv_MeV2"].update(real="0"),
            lambda x: x["controls"]["negative_frequency_control"].__setitem__("pass", False),
            lambda x: x.update(source_sha256="0" * 64),
        ):
            candidate = copy.deepcopy(self.result)
            mutate(candidate)
            self.assertFalse(response.validate_result(candidate))

    def test_cli_is_strict_json_and_does_not_write_cache(self):
        cache = HERE / "nvg_retarded_response_results.json"
        existed = cache.exists()
        before = cache.read_bytes() if existed else None
        completed = subprocess.run([sys.executable, str(response.SOURCE_PATH)], check=True, capture_output=True, text=True,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(HERE)})
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["coverage"]["row_count"], response.EXPECTED_ROW_COUNT)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(cache.exists(), existed)
        if existed:
            self.assertEqual(cache.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
