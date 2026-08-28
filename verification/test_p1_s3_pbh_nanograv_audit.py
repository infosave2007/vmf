"""Focused semantic checks for the P1-S3 PBH--NANOGrav audit."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_pbh_nanograv_audit as audit
import nvg_pbh_mass_spectrum as ladder
import nvg_pbh_two_population as two_population


class P1S3PBHNanoGravAuditTests(unittest.TestCase):
    def test_canonical_producers_and_runtime_ladder_identity(self):
        self.assertEqual(audit.canonical_mass(10), ladder.get_pbh_mass(10))
        self.assertAlmostEqual(audit.canonical_mass(11) / audit.canonical_mass(10), 4.0, places=12)
        # The maintained two-population seed mass is the N=10 rung to rounding.
        self.assertLess(abs(two_population.M_B / audit.canonical_mass(10) - 1.0), 0.01)
        self.assertEqual(audit.benchmark_metadata()["source_path"], "verification/nvg_pbh_two_population.py")

    def test_budget_conservation_and_scoped_semantics(self):
        result = audit.run_audit(density_points=5, abundance_points=9, rate_points=3)
        self.assertEqual(result["scan_contract"]["external_exclusions_used"], False)
        self.assertTrue(result["scan_contract"]["dm_budget_definition"].startswith("f_A + f_B"))
        self.assertAlmostEqual(result["population_a_profile"]["profile_fraction_sum"], 1.0, places=12)
        for key in ("jwst_calibrated_seed_band", "jwst_calibrated_expanded_boundary"):
            self.assertLessEqual(result[key]["max_row"]["fraction_dm"], 1.0)
        self.assertLessEqual(result["internal_budget_expanded_boundary"]["max_row"]["fraction_dm"], 1.0)
        self.assertEqual(result["semantic_interpretation"]["internal_budget_status"], "NO_INTERNAL_BUDGET_NO_GO")
        self.assertTrue(result["semantic_interpretation"]["seed_band_deficit"])
        self.assertGreater(result["jwst_calibrated_seed_band"]["max_deficit_factor"], 100.0)
        self.assertTrue(result["semantic_interpretation"]["boundary_warning"])

    def test_convergence_and_analytic_scaling_controls(self):
        result = audit.run_audit(density_points=5, abundance_points=9, rate_points=3)
        self.assertTrue(result["convergence"]["abundance_converged"])
        self.assertTrue(result["convergence"]["boundary_maximum_is_explicit"])
        self.assertTrue(result["analytic_scaling"]["all_pass"])
        self.assertTrue(all(item["max_at_mass_boundary"] for item in result["convergence"]["boundary_expansion"]))

    def test_output_artifacts_and_cli_are_runtime_derived(self):
        result = audit.run_audit(density_points=3, abundance_points=5, rate_points=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, figure_path = audit.write_artifacts(result, output_dir=Path(temp_dir))
            self.assertTrue(json_path.is_file())
            self.assertTrue(figure_path.is_file())
            self.assertGreater(json_path.stat().st_size, 1000)
            self.assertGreater(figure_path.stat().st_size, 1000)

        completed = subprocess.run(
            [sys.executable, str(HERE / "nvg_pbh_nanograv_audit.py")],
            cwd=str(HERE),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("CONDITIONAL_JWST_BAND_DEFICIT_NO_INTERNAL_BUDGET_NO_GO", completed.stdout)
        self.assertIn("NO_INTERNAL_BUDGET_NO_GO", completed.stdout)

    def test_no_target_fit_or_external_exclusion_shortcut(self):
        source = (HERE / "nvg_pbh_nanograv_audit.py").read_text(encoding="utf-8")
        self.assertNotIn("2.4e-15", source)
        benchmark = audit.benchmark_metadata()
        self.assertIsNone(benchmark["observed_likelihood"])
        self.assertFalse(benchmark["target_calibration_used"])
        self.assertIn("F_PBH_CMB_BOUND", audit.run_audit(density_points=3, abundance_points=5, rate_points=3)["scan_contract"]["external_constraints_reported_only"][0])


if __name__ == "__main__":
    unittest.main()
