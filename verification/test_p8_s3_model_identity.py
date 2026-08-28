"""Semantic checks for the Phase 8 sibling model-identity repair."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


class SiblingModelIdentityTests(unittest.TestCase):
    def test_ilove_sibling_uses_canonical_grid_and_artifact_lambda(self):
        import nvg_bbn_reionization as bbn
        import nvg_iloveq_gw_echoes as ilove

        artifact = json.loads(
            (HERE / "fig_iloveq_universal_report.json").read_text(encoding="utf-8")
        )
        chain = bbn.compute_eos_chain()
        expected_grid = np.logspace(-1.0, 3.4, 120)
        np.testing.assert_allclose(chain["pressure_grid"], expected_grid, rtol=0.0, atol=0.0)
        self.assertEqual(chain["source"], "nvg_tidal_deformability.EOS + solve_tov_tidal")
        self.assertFalse(chain["selection_provenance"]["independent"])
        self.assertEqual(bbn.RESULTS["canonical_chain"]["pressure_grid_points"], 120)
        self.assertEqual(bbn.RESULTS["canonical_chain"]["source"], chain["source"])
        lambda_14 = float(np.interp(1.4, chain["masses"], chain["lambdas"]))
        self.assertAlmostEqual(lambda_14, artifact["summary"]["lambda_14"], places=10)
        self.assertAlmostEqual(ilove.RESULTS["i_love"]["lambda"], lambda_14, places=10)
        self.assertEqual(ilove.RESULTS["i_love"]["pressure_grid_points"], 120)
        self.assertEqual(ilove.RESULTS["i_love"]["status"], "TRANSFORM_ONLY_NO_INDEPENDENT_I_COMPARISON")

    def test_nicer_cli_is_conditional_and_alternate_routes_are_quarantined(self):
        import nvg_moment_of_inertia_j0737 as moment
        import nvg_nicer_j0437_check as nicer
        import nvg_spin_limits_rmodes as spin

        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            result = nicer.run_nicer_check()
        text = rendered.getvalue()
        self.assertEqual(result["metadata"]["status"], "CONDITIONAL_IN_SAMPLE")
        self.assertFalse(result["metadata"]["independent"])
        self.assertEqual(result["metadata"]["evidence_weight"], 0.0)
        self.assertIn("CONDITIONAL_IN_SAMPLE", text)
        self.assertIn("no independent evidence", text.lower())
        self.assertNotIn("Model Predictions", text)
        self.assertNotIn("Predicted Radius", text)
        self.assertNotIn("COMPATIBLE (within 1.5", text)

        for module in (spin, moment):
            identity = module.MODEL_IDENTITY
            self.assertEqual(identity["status"], "QUARANTINED_ALTERNATE_CALIBRATION")
            self.assertEqual(identity["as_of"], "2026-08-27")
            self.assertFalse(identity["canonical"])
            self.assertFalse(identity["independent"])
            self.assertEqual(identity["evidence_weight"], 0.0)
            self.assertAlmostEqual(identity["anchor_radius_km"], 12.49)

        for module in (spin, moment):
            rendered = io.StringIO()
            with contextlib.redirect_stdout(rendered):
                module.main()
            text = rendered.getvalue()
            self.assertIn("QUARANTINED_ALTERNATE_CALIBRATION", text)
            self.assertIn("evidence_weight=0.0", text)
            self.assertNotIn("Prediction", text)
            self.assertNotIn("PASS", text)


if __name__ == "__main__":
    unittest.main()
