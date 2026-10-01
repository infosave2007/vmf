"""Regression coverage for the fail-closed dark-matter budget audit."""
from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_dm_budget_audit as audit  # noqa: E402
import nvg_adm_bl_cogenesis as cogenesis  # noqa: E402
import nvg_relic_dark_matter as relic  # noqa: E402


class DarkMatterBudgetAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_path = HERE / "nvg_dm_budget_audit_results.json"
        cls.result = json.loads(cls.result_path.read_text(encoding="utf-8"))
        cls.input = audit._load_input()

    def test_serialized_result_rebuilds_and_fails_closed_scientifically(self) -> None:
        self.assertTrue(audit.validate_result(self.result))
        self.assertEqual(self.result["status"], audit.STATUS_NOT_CLOSED)
        self.assertEqual(self.result["unified_abundance_status"], audit.CLOSURE_STATUS)
        self.assertTrue(self.result["audit_controls_pass"])
        self.assertFalse(self.result["fit_used"])
        self.assertIsNone(self.result["composition"])

    def test_planck_reference_is_arithmetic_only_and_legacy_calibration_is_explicit(self) -> None:
        reference = self.result["planck_reference"]
        h = Decimal(reference["h"])
        omega_c = Decimal(reference["omega_c"])
        self.assertAlmostEqual(float(Decimal(reference["Omega_c_derived_for_comparison_only"]) * h * h), float(omega_c), places=12)
        self.assertIn("arithmetic", reference["derived_Omega_c_caveat"])
        relic_component = self.result["component_audit"]["relic_defect_inversion"]
        relic_state = relic.run_dm_verification()
        expected_omega = Decimal(str(relic_state["calibration_target_Omega_DM"])) * (
            Decimal(str(relic_state["calibration_H0_km_s_Mpc"])) / Decimal("100")
        ) ** 2
        self.assertEqual(relic_component["implied_target_omega"], audit._number(expected_omega))
        self.assertGreater(Decimal(relic_component["relative_difference_from_Planck_omega_c"]), Decimal("0.18"))
        self.assertEqual(relic_component["calibration_status"], relic.CALIBRATION_STATUS)
        self.assertEqual(relic_component["calibration_input_artifact"], "verification/data/legacy_relic_dark_matter_calibration.json")
        self.assertEqual(relic_component["status"], "LEGACY_CALIBRATION_NOT_AN_INDEPENDENT_ABUNDANCE_PREDICTION")

    def test_pbh_shape_never_becomes_a_component_abundance(self) -> None:
        pbh_component = self.result["component_audit"]["pbh_population_A"]
        self.assertAlmostEqual(float(pbh_component["normalized_profile_sum"]), 1.0, places=12)
        self.assertIsNone(pbh_component["physical_omega_h2"])
        self.assertEqual(pbh_component["status"], "UNIDENTIFIABLE_NO_ABSOLUTE_ABUNDANCE_NORMALIZATION")
        pbh_b = self.result["component_audit"]["pbh_population_B"]
        self.assertIsNone(pbh_b["physical_omega_h2"])
        self.assertEqual(pbh_b["status"], "CALIBRATED_SEED_TRACE_NOT_ABSOLUTE_COSMOLOGY")
        self.assertEqual(pbh_b["calibration_status"], "LEGACY_UNDOCUMENTED_SEED_TRACE_NOT_ABSOLUTE_COSMOLOGY")
        self.assertEqual(pbh_b["calibration_input_artifact"], "verification/data/pbh_population_b_seed_calibration.json")
        self.assertEqual(pbh_b["canonical_ladder_nearest_cycle"], 10)
        self.assertEqual(self.result["manual_remainder_allocation"]["status"], "REJECTED_NOT_DERIVED")

    def test_conditional_chi_arithmetic_is_not_promoted_to_closure(self) -> None:
        chi = self.result["component_audit"]["dark_neutron_chi"]
        omega_chi = Decimal(chi["conditional_omega_chi"])
        omega_c = Decimal(self.result["planck_reference"]["omega_c"])
        self.assertAlmostEqual(float(omega_chi / omega_c), float(Decimal(chi["conditional_fraction_of_Planck_omega_c"])), places=11)
        self.assertEqual(chi["status"], "CONDITIONAL_ON_RETIRED_COGENESIS_EXTENSION")
        self.assertEqual(chi["baryogenesis_evidence_status"], "RETIRED_MISSING_BARYOGENESIS_SOURCE")
        self.assertIn("unassigned_reference_gap_for_diagnostic_only", chi)
        self.assertNotIn("conditional_remaining_omega_for_other_components", chi)

    def test_conditional_chi_historical_inputs_are_explicit_not_source_literals(self) -> None:
        chi = self.result["component_audit"]["dark_neutron_chi"]
        expected_hash = hashlib.sha256(cogenesis.INPUT_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            chi["historical_input_artifact"],
            "verification/data/legacy_dark_neutron_cogenesis_inputs.json",
        )
        self.assertEqual(chi["historical_input_sha256"], expected_hash)
        self.assertEqual(chi["historical_input_status"], cogenesis.INPUT_STATUS)
        self.assertEqual(self.result["source_sha256"]["cogenesis_historical_input"], expected_hash)
        self.assertTrue(
            self.result["controls"][
                "cogenesis_historical_input_passport_is_explicit_and_conditional"
            ]["pass"]
        )

    def test_cogenesis_passport_mismatch_fails_closed(self) -> None:
        expected_hash = hashlib.sha256(cogenesis.INPUT_PATH.read_bytes()).hexdigest()
        with patch.object(
            audit.cogenesis,
            "historical_input_passport",
            return_value={
                "input_artifact": "verification/data/legacy_dark_neutron_cogenesis_inputs.json",
                "input_sha256": expected_hash,
                "input_status": "PROMOTED_WITHOUT_MECHANISM",
            },
        ):
            with self.assertRaises(ValueError):
                audit._build_result()

    def test_bad_external_input_or_result_mutation_fails_closed(self) -> None:
        changed = copy.deepcopy(self.result)
        changed["composition"] = {"invented": "allocation"}
        self.assertFalse(audit.validate_result(changed))
        invalid = copy.deepcopy(self.input)
        invalid["parameters"]["omega_c"]["units"] = "banana"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(ValueError):
                audit._load_input(path)

    def test_planck_passport_rejects_wrong_source_scope_and_unphysical_values(self) -> None:
        mutations = (
            ("source_url", "https://example.invalid/planck"),
            ("reference_model", None),
            ("omega_c", "99"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            for index, (field, value) in enumerate(mutations):
                invalid = copy.deepcopy(self.input)
                if field == "source_url":
                    invalid["source"][field] = value
                elif field == "reference_model":
                    invalid[field] = value
                else:
                    invalid["parameters"][field]["value"] = value
                path = Path(temporary) / f"invalid-{index}.json"
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit._load_input(path)

    def test_a_copied_valid_planck_passport_has_a_graceful_external_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "planck-copy.json"
            path.write_text(json.dumps(self.input), encoding="utf-8")
            rebuilt = audit._build_result(path)
            self.assertEqual(rebuilt["input_artifact"], str(path.resolve()))
            self.assertTrue(audit.validate_result(rebuilt, path))

    def test_key_name_or_null_cannot_create_a_pbh_abundance_contract(self) -> None:
        self.assertFalse(audit._has_absolute_pbh_omega_contract({"omega_pbh": "0.12"}))
        self.assertFalse(
            audit._has_absolute_pbh_omega_contract(
                {
                    "physical_omega_h2": None,
                    "physical_omega_h2_units": "dimensionless",
                    "formation_solver": "invented",
                    "abundance_source_provenance": {"source_url": "https://example.invalid"},
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
