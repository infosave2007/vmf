"""Focused semantic controls for the P3-S2 cooling/g-mode no-go audit."""

from __future__ import annotations

import subprocess
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from verification import nvg_cooling_gmode_dependency_audit as audit


ROOT = Path(__file__).resolve().parents[1]


class P3S2CoolingGModeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit.run_audit()

    def test_barotropic_relativistic_null_is_neutral(self) -> None:
        null = self.result["buoyancy"]["null_limit"]
        self.assertEqual(null["status"], "DERIVED_NULL_CONTROL_BAROTROPIC_NEUTRAL_BUOYANCY")
        self.assertLess(null["max_abs_discriminant"], 1.0e-14)
        self.assertLess(null["max_abs_n2_km2"], 1.0e-14)
        self.assertEqual(null["physical_frequency_status"], "BLOCKED_MISSING_GAMMA1_COMPOSITION_DERIVATIVES")

    def test_newtonian_and_relativistic_sign_and_units_controls(self) -> None:
        controls = self.result["controls"]["analytic_null_and_sign"]
        self.assertEqual(controls["status"], "PASS_ANALYTIC_NULL_UNITS_SIGN_DOMAIN")
        self.assertLess(controls["newtonian_null_max_abs_s2"], 1.0e-10)
        self.assertGreater(controls["newtonian_stable_min_s2"], 0.0)
        self.assertLess(controls["relativistic_null_max_abs_km2"], 1.0e-14)
        self.assertEqual(controls["units"]["newtonian_n2"], "s^-2")
        self.assertEqual(controls["units"]["relativistic_n2"], "km^-2")

    def test_frozen_composition_branch_never_invents_gamma1(self) -> None:
        branch = self.result["buoyancy"]["frozen_composition_branch"]
        self.assertTrue(branch["status"].startswith("BLOCKED_"))
        self.assertTrue(branch["no_frequency_invented"])
        self.assertTrue(any("Gamma_1" in item for item in branch["required"]))
        sensitivity = self.result["buoyancy"]["frozen_gamma_reference"]
        self.assertEqual(sensitivity["status"], "SENSITIVITY_ONLY_FROZEN_GAMMA1_NOT_MICROPHYSICS")
        self.assertTrue(sensitivity["not_a_physical_gmode"])

    def test_ledger_covers_legacy_surfaces_and_blocks_physical_claims(self) -> None:
        rows = self.result["dependency_ledger"]
        self.assertGreaterEqual(len(rows), 10)
        ids = {row["claim_id"] for row in rows}
        for required in {
            "ns_gmode_wkb",
            "cas_a_cooling_curve",
            "direct_urca_threshold",
            "p2_zero_interaction_reference",
            "pulsar_population_cooling",
            "sn1987a_cooling_scan",
        }:
            self.assertIn(required, ids)
        for row in rows:
            self.assertFalse(row["legacy_array_imported_as_evidence"])
            self.assertEqual(row["evidence_weight"], 0.0)
            self.assertTrue(row["missing_producers"])
            self.assertTrue(row["static_or_illustrative_inputs"])
        self.assertFalse(self.result["claims_by_id"]["ns_gmode_wkb"]["frequency_from_sourced_microphysics"])

    def test_required_cooling_closure_and_no_go_statuses(self) -> None:
        dependencies = self.result["required_cooling_dependencies"]
        expected = {
            "composition",
            "heat_capacity",
            "neutrino_channels_and_matrix_elements",
            "temperature_dependence",
            "superfluid_gaps_and_suppression",
            "conductivity",
            "stellar_profile",
            "envelope_relation",
        }
        self.assertEqual(set(dependencies), expected)
        self.assertTrue(all(item["status"].startswith("BLOCKED_") for item in dependencies.values()))
        self.assertTrue(all(item["producer"] == "none in maintained repository" for item in dependencies.values()))
        for value in self.result["physical_no_go"].values():
            self.assertTrue(value.startswith("BLOCKED_"))

    def test_canonical_and_p2_regressions_are_read_only_passes(self) -> None:
        self.assertEqual(self.result["controls"]["canonical_regression"]["status"], "PASS")
        self.assertEqual(self.result["controls"]["p2_regression"]["status"], "PASS")
        self.assertEqual(
            self.result["controls"]["p2_regression"]["stable_mapping"],
            "BLOCKED_NO_STABLE_TOV_BRANCH",
        )
        self.assertEqual(
            self.result["controls"]["p2_regression"]["reference_model"],
            "DERIVED_ZERO_INTERACTION_FREE_GAS_REFERENCE_ONLY",
        )

    def test_grid_convergence_and_fail_closed_mutations(self) -> None:
        convergence = self.result["controls"]["grid_convergence"]
        self.assertEqual(convergence["status"], "PASS_NULL_GRID_CONVERGENCE")
        self.assertEqual(convergence["grid_points"], [61, 121, 241, 481])
        self.assertTrue(convergence["all_below_1e-14"])
        mutations = self.result["controls"]["adversarial_mutations"]
        self.assertEqual(mutations["status"], "PASS_FAIL_CLOSED_MUTATIONS")
        self.assertTrue(all(mutations["checks"].values()))
        derivative = self.result["controls"]["derivative_convergence"]
        self.assertEqual(derivative["status"], "PASS_DERIVATIVE_IDENTITY_GRID_CONVERGENCE")
        self.assertTrue(derivative["all_below_1e-14_relative"])

    def test_public_helpers_validate_domain_and_reproduce_null(self) -> None:
        pressure = np.array([1.0, 2.0, 3.0])
        energy = 10.0 + 4.0 * pressure
        gradient = np.array([-1.0, -1.0, -1.0])
        gamma = (energy + pressure) / (4.0 * pressure)
        n2 = audit.relativistic_buoyancy_n2(pressure, energy, gradient, 4.0 * gradient, gamma)
        np.testing.assert_allclose(n2, 0.0, rtol=0.0, atol=1.0e-15)
        with self.assertRaises(ValueError):
            audit.newtonian_buoyancy_n2(1.0, [1.0], [1.0], [-1.0], [-1.0], [0.0])
        with self.assertRaises(ValueError):
            audit.relativistic_buoyancy_n2(pressure, energy, gradient, gradient, gamma, redshift_factor=0.0)

    def test_cli_emits_terminal_status(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "verification" / "nvg_cooling_gmode_dependency_audit.py")],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("COMPLETE_WITH_PHYSICAL_COOLING_AND_GMODE_BLOCKS", completed.stdout)
        self.assertIn("BLOCKED_MISSING_GAMMA1_COMPOSITION_DERIVATIVES", completed.stdout)
        self.assertIn("canonical_regression=PASS", completed.stdout)
        self.assertIn("payload_sha256=", completed.stdout)

    def test_complete_payload_digest_and_regeneration(self) -> None:
        integrity = self.result["payload_integrity"]
        self.assertEqual(integrity["status"], "PASS_COMPLETE_PAYLOAD_DIGEST")
        self.assertTrue(integrity["complete"])
        self.assertTrue(integrity["deterministic_regeneration"])
        self.assertEqual(integrity["sha256"], integrity["payload_sha256"])
        self.assertEqual(integrity["sha256"], audit.canonical_payload_digest(self.result))
        audit.assert_artifact_provenance(self.result)

    def test_complete_payload_mutations_fail_closed(self) -> None:
        # These fields were intentionally left mutable by the P3-S2 assertion;
        # the P3-S6 digest/regeneration boundary must reject each one.
        mutations = []

        equation = deepcopy(self.result)
        equation["buoyancy"]["criterion"]["relativistic"] = "WRONG"
        mutations.append(equation)

        ledger = deepcopy(self.result)
        ledger["dependency_ledger"][0]["audit_status"] = "PASS_FAKE"
        mutations.append(ledger)

        blocker = deepcopy(self.result)
        blocker["physical_no_go"]["cooling_curves"] = "BLOCKED_FAKE"
        mutations.append(blocker)

        line_scope = deepcopy(self.result)
        line_scope["dependency_ledger"][0]["source_lines"] = "verification/other.py:1-2"
        mutations.append(line_scope)

        numeric = deepcopy(self.result)
        numeric["buoyancy"]["null_limit"]["n2_km2"][0] = 1.0
        mutations.append(numeric)

        source_hash = deepcopy(self.result)
        source_hash["dependency_ledger"][0]["source_sha256"] = "0" * 64
        mutations.append(source_hash)

        for mutated in mutations:
            with self.assertRaises(AssertionError):
                audit.assert_artifact_provenance(mutated)


if __name__ == "__main__":
    unittest.main()
