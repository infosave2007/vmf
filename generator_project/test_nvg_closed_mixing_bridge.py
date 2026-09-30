"""Focused live tests for the sealed I1 closed-mixing bridge."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import calc_nvg_closed_mixing_bridge as bridge  # noqa: E402


class ClosedMixingBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exactly one live build.  No serialized result is used as a scientific
        # input; the optional write is only a validator/debug payload.
        cls.result = bridge.build_result()
        if cls.result.get("status") == bridge.STATUS_PASS and bridge.validate_result(cls.result):
            bridge._write_json(cls.result, HERE / "nvg_closed_mixing_bridge_results.json")

    def test_live_coverage_and_status(self):
        result = self.result
        self.assertEqual(result["status"], bridge.STATUS_PASS)
        self.assertTrue(result["all_controls_pass"])
        self.assertTrue(result["physical_inputs_fixed"])
        self.assertTrue(result["no_saved_results_as_inputs"])
        self.assertFalse(result["fit_used"])
        self.assertEqual(result["coverage"]["reference_count"], 4)
        self.assertEqual(result["coverage"]["composition_count"], 36)
        self.assertEqual(result["coverage"]["row_count"], 72)
        self.assertTrue(result["coverage"]["all_rows_present"])
        self.assertTrue(bridge.validate_result(result))

    def test_distinct_protocols_and_contacts(self):
        rows = self.result["rows"]
        self.assertEqual({row["delta"] for row in rows}, set(bridge.DELTAS))
        self.assertEqual({row["contact"] for row in rows}, set(bridge.CONTACT_ORDER))
        for row in rows:
            self.assertEqual(row["status"], "CANONICAL_ROOT_OK")
            self.assertTrue(row["pass"])
            self.assertLess(float(row["prepared"]["root_residual_relative"]), float(bridge.ROOT_REL_LIMIT))
            self.assertLess(float(row["prepared"]["energy_ledger_residual"]), float(bridge.ENERGY_REL_LIMIT))
            self.assertLess(float(row["prepared"]["pressure_identity_residual"]), float(bridge.GIBBS_REL_LIMIT))
            self.assertIn("closed_mixing", row["closed_protocol"])
            self.assertIn("reversible_separation", row["closed_protocol"])
            self.assertIn("composed_endpoint", row["closed_protocol"])
            self.assertIn("B_ratio", self.result["isobaric"][0])

    def test_independent_work_and_unified_identity(self):
        for row in self.result["rows"]:
            iso = row["isothermal_separation"]
            self.assertLess(float(iso["work_identity_residual"]), 2e-18)
            if row["delta"] == "0.2":
                self.assertFalse(iso["quadrature_omitted"])
                self.assertEqual({int(x["points"]) for x in iso["muD_integral"]}, {8, 12})
                self.assertTrue(iso["quadrature_pass"])
            else:
                self.assertTrue(iso["quadrature_omitted"] or row["delta"] == "0")
        self.assertTrue(self.result["controls"]["contact_identity"]["all_pass"])
        for payload in self.result["coefficients"].values():
            self.assertLess(float(payload["identity_relative"]), 1e-10)
            self.assertGreater(float(payload["cv_dimensionless"]), 0)

    def test_classical_null_is_not_heating_claim(self):
        null = self.result["controls"]["classical_null"]
        self.assertTrue(null["all_pass"])
        for row in null["rows"]:
            self.assertAlmostEqual(float(row["S_U"]), 0.0, places=30)
            self.assertAlmostEqual(float(row["Tmix_minus_T0"]), 0.0, places=30)
            self.assertAlmostEqual(float(row["A_P"]), 0.0, places=30)

    def test_validator_rejects_rehashed_numeric_mutations(self):
        # Rehashing a corrupted payload must still fail because numeric gates
        # are recomputed from states rather than accepting trusted flags.
        for path, value in (
            (("rows", 0, "prepared", "energy_MeV4"), "999"),
            (("rows", 1, "closed_protocol", "closed_mixing", "Tmix_MeV"), "70.0"),
            (("rows", 1, "closed_protocol", "composed_endpoint", "delta_sigma_cycle"), "999"),
            (("rows", 0, "closed_protocol", "reversible_separation", "entropy_target_per_B"), "999"),
            (("rows", 0, "isothermal_separation", "wiso_MeV"), "999"),
            (("coefficients", next(iter(self.result["coefficients"])), "S_F_MeV"), "999"),
            (("controls", "classical_null", "rows", 1, "Phi"), "999"),
        ):
            mutated = copy.deepcopy(self.result)
            node = mutated
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            mutated.pop("integrity_sha256", None)
            mutated["integrity_sha256"] = hashlib.sha256(bridge._canonical_payload(mutated).encode()).hexdigest()
            self.assertFalse(bridge.validate_result(mutated), path)
        duplicate = copy.deepcopy(self.result)
        duplicate["rows"].append(copy.deepcopy(duplicate["rows"][0]))
        self.assertFalse(bridge.validate_result(duplicate))

    def test_source_scope_and_fixed_inputs(self):
        source = (HERE / "calc_nvg_closed_mixing_bridge.py").read_text(encoding="utf-8")
        self.assertNotIn("Lunacy/runs", source)
        self.assertNotIn("evidence_parent", source)
        self.assertEqual(self.result["inputs"]["j_MeV"], "11.13601607117819")
        self.assertEqual(self.result["inputs"]["n0_fm3"], "0.16")
        self.assertEqual(self.result["inputs"]["T0_MeV"], "70")


if __name__ == "__main__":
    unittest.main()
