"""Focused semantic controls for the P2-S2 composition/Urca audit."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from verification import nvg_beta_hyperon_urca_audit as audit


class P2S2BetaHyperonUrcaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = audit.run_audit(quick=True)

    def test_multispecies_conservation_and_equilibrium_residuals(self):
        composition = self.result["composition"]
        fractions = composition["fractions"]
        total = np.sum(np.asarray([fractions[name] for name in ("n", "p", "Lambda")]), axis=0)
        self.assertLess(float(np.max(np.abs(total - 1.0))), 1.0e-8)
        controls = self.result["eos"]["controls"]
        self.assertLess(controls["charge_neutrality_max_abs_fm3"], 1.0e-10)
        self.assertLess(controls["beta_equilibrium_max_abs_mev"], 1.0e-8)
        self.assertLess(controls["lambda_complementarity_max_abs_mev"], 1.0e-8)

    def test_thermodynamics_causality_and_onset_limits(self):
        controls = self.result["eos"]["controls"]
        self.assertTrue(controls["finite_all_arrays"])
        self.assertTrue(controls["pressure_monotone"])
        self.assertTrue(controls["energy_monotone"])
        self.assertTrue(controls["causal_sound_speed"])
        self.assertLess(controls["pressure_identity_max_relative"], 1.0e-8)
        self.assertLess(controls["thermodynamic_derivative_max_relative"], 5.0e-3)
        self.assertGreaterEqual(controls["zero_density_pressure_limit"], 0.0)
        self.assertEqual(self.result["model"]["couplings"]["all_interaction_couplings"], 0.0)

    def test_independent_triangle_condition_and_urca_statuses(self):
        channels = self.result["urca"]["channels"]
        identity = channels["nucleonic_fraction_identity"]
        self.assertTrue(identity["boolean_identity"])
        self.assertEqual(identity["boolean_disagreement_count"], 0)
        self.assertIsNone(channels["e"]["onset_density_ratio"])
        self.assertIsNone(channels["mu"]["onset_density_ratio"])
        self.assertEqual(channels["e"]["status"], "DERIVED_FREE_GAS_KINEMATIC_REFERENCE")
        self.assertEqual(channels["mu"]["status"], "DERIVED_FREE_GAS_KINEMATIC_REFERENCE")
        self.assertTrue(channels["Lambda_e"]["status"].startswith("BLOCKED_"))
        self.assertTrue(channels["Lambda_mu"]["status"].startswith("BLOCKED_"))
        self.assertEqual(
            self.result["urca"]["emissivity_status"],
            "BLOCKED_MISSING_WEAK_MATRIX_ELEMENTS_TEMPERATURE_AND_SUPERFLUID_GAPS",
        )

    def test_missing_physical_dependencies_are_fail_closed(self):
        deps = self.result["dependency_audit"]
        self.assertEqual(
            deps["physical_hyperon_status"],
            "BLOCKED_MISSING_TRACEABLE_COUPLINGS_PHASE_AND_NVG_DEPENDENCY",
        )
        statuses = {row["node"]: row["status"] for row in deps["dependency_graph"]}
        self.assertTrue(statuses["interacting_hyperon_mean_fields"].startswith("BLOCKED_"))
        self.assertTrue(statuses["phase_construction"].startswith("BLOCKED_"))
        self.assertTrue(statuses["hyperonic_emissivity"].startswith("BLOCKED_"))
        self.assertTrue(statuses["nvg_hyperon_dependency"].startswith("BLOCKED_"))
        source = Path(audit.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import nvg_eos_beta_lambda_hyperon", source)
        self.assertNotIn("import nvg_direct_urca", source)

    def test_tov_mapping_never_promotes_unstable_reference_branch(self):
        mapping = self.result["stellar_mapping"]
        self.assertEqual(mapping["status"], "BLOCKED_NO_STABLE_TOV_BRANCH")
        self.assertEqual(mapping["stable_count"], 0)
        self.assertIsNone(mapping["onset_mass_msun"])
        self.assertEqual(mapping["onset_mapping_status"], "BLOCKED_NO_STABLE_TOV_BRANCH")

    def test_provenance_sensitivity_and_canonical_regression(self):
        self.assertEqual(self.result["artifacts"]["result_json"], "verification/nvg_beta_hyperon_urca_audit_p2s2_results.json")
        self.assertEqual(self.result["artifacts"]["figure_png"], "verification/fig_p2_s2_beta_hyperon_urca_audit.png")
        self.assertEqual(self.result["source_provenance"]["external_inputs"]["count"], 0)
        self.assertEqual(self.result["source_provenance"]["external_inputs"]["status"], "NONE_USED")
        self.assertEqual(self.result["canonical_regression"]["status"], "PASS")
        audit.assert_artifact_provenance(self.result)
        self.assertEqual(self.result["sensitivity"]["status"], "NUMERICAL_GRID_SENSITIVITY_ONLY_NO_FIT")
        self.assertLess(self.result["sensitivity"]["max_relative_y_p"], 1.0e-3)
        self.assertLess(self.result["sensitivity"]["max_relative_y_Lambda"], 1.0e-3)

    def test_cli_regenerates_unique_json_artifact(self):
        completed = subprocess.run(
            [sys.executable, str(Path(audit.__file__)), "--quick", "--no-figure"],
            cwd=audit.ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("physical hyperon status=BLOCKED_", completed.stdout)
        self.assertTrue(audit.RESULT_PATH.exists())
        payload = json.loads(audit.RESULT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["audit"], "P2-S2")
        self.assertEqual(payload["status"], self.result["status"])


if __name__ == "__main__":
    unittest.main()
