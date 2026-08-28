"""Focused semantic checks for the Phase 1 S8 sign/no-go audit."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_s8_no_go as audit


class P1S2S8NoGoTests(unittest.TestCase):
    def test_maintained_route_identity_and_sign(self) -> None:
        canonical = audit.compute_s8_test()
        point = audit.point_record(canonical["w_0"], canonical["w_a"], 0.315)
        self.assertEqual(audit.CANONICAL_SOURCE, "verification/nvg_black_hole_entropy.py::compute_s8_test")
        self.assertAlmostEqual(point["growth_ratio_gamma055"], canonical["sigma8_ratio"], places=5)
        self.assertAlmostEqual(point["s8"], canonical["s8_nvg"], places=5)
        self.assertTrue(point["moves_away_from_lensing"])
        self.assertGreater(point["growth_ratio_gamma055"], 1.0)
        self.assertEqual(point["sign"]["sign_class"], "mixed_sign_integral_required")

    def test_convergence_and_alternative_approximations(self) -> None:
        result = audit.build_result()
        convergence = result["convergence"]
        self.assertTrue(convergence["converged"])
        self.assertTrue(convergence["canonical_identity_pass"])
        self.assertLess(convergence["relative_spread"], 2e-3)
        self.assertTrue(result["map_resolution_convergence"]["one_sigma_overlap_stable_zero"])
        alternatives = result["alternative_approximations"]
        self.assertTrue(alternatives["sign_agreement"])
        self.assertTrue(alternatives["all_enhance_at_maintained_point"])

    def test_boundary_expansion_and_scope_are_explicit(self) -> None:
        result = audit.build_result()
        self.assertEqual(result["status"].split(";")[0], "SCOPED_NO_GO_MAINTAINED_OMEGA_M")
        maintained = result["scans"]["maintained"]["counts"]
        expanded = result["scans"]["expanded"]["counts"]
        self.assertEqual(maintained["both_lensing_1sigma_and_desi_overlay"], 0)
        self.assertGreater(expanded["both_lensing_1sigma_and_desi_overlay"], 0)
        self.assertFalse(result["provenance"]["desi_overlay_is_observed_likelihood"])
        self.assertFalse(result["claim_scope"]["observational_claim"])

    def test_cli_and_generated_artifacts(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(HERE / "nvg_s8_no_go.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("SCOPED_NO_GO_MAINTAINED_OMEGA_M", completed.stdout)
        payload = json.loads((HERE / "nvg_s8_no_go_map.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["provenance"]["canonical_growth_source"], audit.CANONICAL_SOURCE)
        self.assertFalse(payload["provenance"]["desi_overlay_is_observed_likelihood"])
        self.assertGreater((HERE / "fig_s8_no_go_map.png").stat().st_size, 1000)

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            audit.main()
        self.assertIn("JSON:", rendered.getvalue())
        self.assertIn("FIGURE:", rendered.getvalue())


if __name__ == "__main__":
    unittest.main()
