"""Semantic tests for the phase-1 renormalisation/covariance audit."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

from verification import nvg_renormalization_covariance_audit as audit


ROOT = Path(__file__).resolve().parents[1]


class RenormalizationCovarianceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit.run_audit()

    def test_maintained_budget_is_recomputed_from_producer(self) -> None:
        budget = self.result["maintained_mass_budget"]
        self.assertEqual(budget["producer"], "verification/nvg_hadron_mass_fractions.py::compute_mass_budget")
        self.assertAlmostEqual(budget["sum_sigma_MeV"], 80.0)
        self.assertAlmostEqual(budget["M_Omega_MeV"], 859.0)
        self.assertAlmostEqual(budget["f_Omega"], 1.0 - 80.0 / 939.0)
        self.assertAlmostEqual(budget["sum_sigma_error_quadrature_MeV"], (3.0**2 + 7.0**2 + 3.0**2) ** 0.5)

    def test_rg_invariant_product_control_changes_parts_not_product(self) -> None:
        control = self.result["rg_invariant_controls"]
        self.assertEqual(control["status"], "PASS_RG_INVARIANT_PRODUCT_ALGEBRA")
        self.assertTrue(all(control["checks"].values()))
        example = control["example"]
        self.assertNotEqual(example["m_original"], example["m_transformed"])
        self.assertAlmostEqual(example["product_original"], example["product_transformed"], places=12)

    def test_missing_maintained_metadata_blocks_scalar_conversion(self) -> None:
        conversion = self.result["scalar_operator_conversion"]
        self.assertEqual(conversion["status"], "BLOCKED_MISSING_OPERATOR_TRANSFORMATION")
        self.assertIsNone(conversion["matrix_element"])
        self.assertIn("Z_m Z_S relation", " ".join(conversion["required"]))
        row = next(item for item in self.result["inventory"] if item["input_id"].endswith("sigma_piN"))
        with self.assertRaises(ValueError):
            audit.require_metadata(row)

    def test_source_backed_reference_records_are_explicitly_renormalised(self) -> None:
        records = audit.source_backed_flag_mass_records()
        self.assertEqual(len(records), 3)
        for record in records:
            data = asdict(record)
            self.assertTrue(audit.metadata_ready(data))
            self.assertEqual(record.scheme, "MSbar")
            self.assertEqual(record.scale_GeV, 2.0)
            self.assertIn("Z_m", record.transformation)

    def test_heavy_matching_is_derived_and_does_not_replace_literal(self) -> None:
        heavy = self.result["heavy_quark_matching"]
        self.assertEqual(heavy["status"], "REFERENCE_LO_SEQUENTIAL_MATCHING_DERIVED_NOT_COMPARABLE")
        remainder = 939.0 - 44.0 - 30.0
        self.assertAlmostEqual(heavy["references"]["charm"]["sigma_reference_MeV"], (2.0 / 27.0) * remainder)
        self.assertAlmostEqual(heavy["references"]["bottom"]["sigma_reference_MeV"], (2.0 / 25.0) * remainder)
        self.assertAlmostEqual(heavy["references"]["top"]["sigma_reference_MeV"], (2.0 / 23.0) * remainder)
        self.assertAlmostEqual(
            heavy["explicit_sum_reference"]["sigma_c_plus_sigma_b_plus_sigma_t_MeV"],
            ((2.0 / 27.0) + (2.0 / 25.0) + (2.0 / 23.0)) * remainder,
        )
        self.assertEqual(heavy["comparison_status"], "NOT_COMPARABLE_MISSING_HEAVY_CONVENTION")
        self.assertNotIn("difference_pull", heavy)
        self.assertNotIn("difference_MeV", heavy)
        self.assertEqual(heavy["remainder"]["assumption_status"], "UNSOURCED_APPROXIMATION_NOT_IDENTITY")
        self.assertIn("correlated", heavy["remainder"]["correlation_note"])
        self.assertEqual(self.result["claims"]["parameter_refit"], "NOT_PERFORMED")

    def test_inventory_and_surface_scan_are_machine_readable(self) -> None:
        inventory = self.result["inventory"]
        self.assertGreaterEqual(len(inventory), 10)
        required = {"sigma_product", "EOS_anchor", "quark_mass", "in_medium_prediction"}
        self.assertTrue(required.issubset({row["role"] for row in inventory}))
        scan = self.result["source_surface_scan"]
        self.assertEqual(scan["status"], "PASS_MAINTAINED_SURFACES_SCANNED")
        self.assertGreaterEqual(scan["surface_count"], 8)
        self.assertTrue(all(row["exists"] for row in scan["rows"]))

    def test_cli_regenerates_json_and_exposes_fail_closed_status(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "verification" / "nvg_renormalization_covariance_audit.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("COMPLETE_WITH_RENORM_BLOCKS", completed.stdout)
        self.assertIn("PASS_RG_INVARIANT_PRODUCT_ALGEBRA", completed.stdout)
        self.assertIn("NOT_COMPARABLE_MISSING_HEAVY_CONVENTION", completed.stdout)
        self.assertNotIn("sigma discrepancy", completed.stdout.lower())
        payload = json.loads((ROOT / "verification" / "renormalization_covariance_audit_results.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "COMPLETE_WITH_RENORM_BLOCKS")
        self.assertEqual(payload["renormalization_contract"]["status"], "BLOCKED_MISSING_RENORM_METADATA")
        generated_report = ROOT / "verification" / "renormalization_covariance_audit_report.md"
        self.assertTrue(generated_report.exists())
        self.assertIn("generated", generated_report.read_text(encoding="utf-8").splitlines()[0])

    def test_top_level_status_is_derived_from_nested_gates(self) -> None:
        metadata = {"status": "BLOCKED_MISSING_RENORM_METADATA"}
        scalar = {"status": "BLOCKED_MISSING_OPERATOR_TRANSFORMATION"}
        invariant = {"status": "PASS_RG_INVARIANT_PRODUCT_ALGEBRA"}
        surfaces = {"status": "PASS_MAINTAINED_SURFACES_SCANNED"}
        self.assertEqual(
            audit.derive_audit_status(metadata, scalar, invariant, surfaces),
            "COMPLETE_WITH_RENORM_BLOCKS",
        )
        self.assertEqual(
            audit.derive_audit_status(metadata, scalar, {"status": "FAIL_RG_ALGEBRA"}, surfaces),
            "FAIL_RG_INVARIANT_CONTROL",
        )
        self.assertEqual(
            audit.derive_audit_status(
                {"status": "PASS_RENORM_METADATA_COMPLETE"},
                {"status": "PASS_SOURCE_BACKED_SCALAR_EXTRACTION"},
                invariant,
                surfaces,
            ),
            "COMPLETE_RENORM_AUDIT",
        )

    def test_generated_artifacts_are_byte_regenerable(self) -> None:
        result = audit.run_audit()
        result_bytes, inventory_bytes, report_bytes = audit.serialized_artifacts(result)
        self.assertEqual(
            result_bytes,
            (ROOT / "verification" / "renormalization_covariance_audit_results.json").read_bytes(),
        )
        self.assertEqual(
            inventory_bytes,
            (ROOT / "verification" / "renormalization_covariance_audit_inventory.json").read_bytes(),
        )
        self.assertEqual(
            report_bytes,
            (ROOT / "verification" / "renormalization_covariance_audit_report.md").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
