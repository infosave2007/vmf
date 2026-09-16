"""Focused live tests for the residual-guided finite-droplet enrichment."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_droplet_residual_enrichment_audit as audit  # noqa: E402


class _ToyDesign:
    n0_dim = 1.0


class _ToyBackground:
    def __init__(self):
        self.x = np.linspace(0.0, 1.0, 11)
        self.design = _ToyDesign()
        self.edge_x = 1.0


class DropletResidualEnrichmentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit.build_result()

    def test_live_schema_coverage_and_frozen_selection(self):
        result = self.result
        self.assertEqual(result["schema_version"], audit.SCHEMA)
        self.assertEqual(result["status"], audit.STATUS)
        self.assertEqual(result["evidence_weight"], 0.0)
        self.assertEqual(set(result["cases"]), set(audit.TARGETS))
        self.assertTrue(result["model_scope"]["no_physics_or_parameter_change"])
        self.assertEqual(result["model_scope"]["enrichment_stages_added"], [0, 4, 8])
        for target in audit.TARGETS:
            case = result["cases"][target]
            self.assertEqual(set(case["sectors"]), {str(ell) for ell in audit.ELL_VALUES})
            self.assertEqual(set(case["selections"]), {str(ell) for ell in audit.ELL_VALUES})
            for ell in audit.ELL_VALUES:
                selection = case["selections"][str(ell)]
                self.assertTrue(selection["heldout_never_used_for_selection"])
                self.assertEqual(selection["actual_selected_count"], 8)
                self.assertEqual(selection["selected_candidate_ids"], selection["stage_additions"]["stage4"] + selection["stage_additions"]["stage8"])
                training_ids = {item["candidate_id"] for item in selection["candidate_specs"]}
                holdout_ids = {item["candidate_id"] for item in selection["holdout_specs"]}
                self.assertEqual(len(training_ids), 28)
                self.assertEqual(len(holdout_ids), 12)
                self.assertTrue(training_ids.isdisjoint(holdout_ids))
                sector = case["sectors"][str(ell)]
                self.assertEqual(len(sector["validation_rows"]), 9)
                self.assertTrue(sector["all_validation_rows_reported"])
                for row in sector["validation_rows"]:
                    self.assertEqual([stage["stage"] for stage in row["stage_rows"]], [0, 4, 8])
                    self.assertEqual(row["stage_rows"][2]["heldout"]["probe_count"], 12)

    def test_bump_derivatives_and_zero_outside_support(self):
        x = np.linspace(0.0, 2.0, 2001)
        center, width = 1.0, 0.25
        bump, first, second = audit._bump_with_derivatives(x, center, width)
        inside = np.abs((x - center) / width) < 0.8
        h = x[1] - x[0]
        numerical_first = np.gradient(bump, h, edge_order=2)
        numerical_second = np.gradient(numerical_first, h, edge_order=2)
        np.testing.assert_allclose(first[inside], numerical_first[inside], rtol=2.0e-3, atol=2.0e-3)
        np.testing.assert_allclose(second[inside], numerical_second[inside], rtol=5.0e-2, atol=5.0e-2)
        self.assertTrue(np.all(bump[np.abs(x - center) >= width] == 0.0))
        self.assertTrue(np.all(first[np.abs(x - center) >= width] == 0.0))
        self.assertTrue(np.all(second[np.abs(x - center) >= width] == 0.0))

    def test_generic_full_physical_whitening_is_coherent(self):
        background = _ToyBackground()
        u = np.column_stack((1.0 + background.x, 2.0 - background.x))
        v = np.column_stack((0.4 * background.x, 0.2 + background.x**2))
        vp = np.column_stack((np.ones_like(background.x), 2.0 * background.x))
        columns = {
            "u": u,
            "v": v,
            "vprime": vp,
            "labels": ["density_a", "scalar_a"],
            "translation_coefficient_vector": np.array((1.0, 0.2)),
        }
        whitened = audit._physical_whiten(background, 2, columns)
        self.assertTrue(whitened["representation_reliable"])
        self.assertTrue(whitened["all_nonzero_columns_used"])
        self.assertFalse(whitened["uses_pseudoinverse"])
        self.assertEqual(whitened["white_gram_rank_after_unchanged_cutoff"], 2)
        np.testing.assert_allclose(whitened["white_gram"], np.eye(2), rtol=1.0e-10, atol=1.0e-10)
        np.testing.assert_allclose(whitened["vprime"], vp @ whitened["transform"], rtol=1.0e-12, atol=1.0e-12)

    def test_zero_support_candidate_is_explicitly_ineligible(self):
        background = SimpleNamespace(
            x=np.linspace(0.0, 1.0, 11),
            n=np.zeros(11),
            nprime=np.zeros(11),
            edge_x=1.0,
            design=_ToyDesign(),
        )
        spec = audit._candidate_specs(edge=1.0)[0]
        candidate = audit._candidate_field(background, 1, spec)
        self.assertTrue(candidate.zero_support)
        self.assertEqual(candidate.raw_norm, 0.0)

    def test_duplicate_candidate_is_near_dependent_after_projection(self):
        spec = {"candidate_id": "duplicate", "kind": "scalar"}
        field = audit.CandidateField(
            spec=spec,
            u=np.zeros(3),
            v=np.ones(3),
            vprime=np.zeros(3),
            correction={},
            raw_norm=1.0,
            support_left=0.0,
            support_right=1.0,
            support_region="interior",
            zero_support=False,
        )
        dimension = audit.BASE_DIMENSION + 2
        gram = np.eye(dimension)
        gram[-2:, -2:] = 1.0
        space = audit.CombinedSpace(
            background=None,
            ell=0,
            base={"translation_new": np.zeros(audit.BASE_DIMENSION)},
            candidates=[field, field],
            u=np.zeros((3, dimension)),
            v=np.zeros((3, dimension)),
            vprime=np.zeros((3, dimension)),
            hessian=np.eye(dimension),
            gram=gram,
            labels=[],
            translation=np.zeros(dimension),
            diagnostics={},
        )
        current = audit._identity_coefficients(space)
        first = audit._candidate_complement(space, 0, current)
        self.assertTrue(first["eligible"])
        current = audit._append_candidate(current, first["normalized_coefficient"])
        duplicate = audit._candidate_complement(space, 1, current)
        self.assertTrue(duplicate["near_dependent"])
        self.assertFalse(duplicate["eligible"])

    def test_independent_mathematical_controls_include_hidden_negative(self):
        controls = audit._synthetic_controls()
        self.assertTrue(controls["all_checks_pass"])
        self.assertEqual(controls["hidden_outside_space_negative"]["old_low_mode_residual_score"], 0.0)
        self.assertTrue(controls["hidden_outside_space_negative"]["negative_exposed_after_enrichment"])
        self.assertTrue(controls["nonorthogonal_translation_quotient"]["translation_removed_exactly_one"])
        self.assertTrue(controls["nonorthogonal_translation_quotient"]["negative_preserved"])

    def test_degenerate_score_is_invariant_and_keeps_lambda_sign(self):
        hessian = np.diag((2.0, 2.0, 4.0))
        gram = np.eye(3)
        candidate = np.array((0.3, -0.4, 0.5))
        modes = np.eye(3)[:, :2]
        score_a = audit._residual_score(hessian, gram, candidate, (2.0, 2.0), modes)
        rotation = np.array(((0.0, 1.0), (1.0, 0.0)))
        score_b = audit._residual_score(hessian, gram, candidate, (2.0, 2.0), modes @ rotation)
        self.assertAlmostEqual(score_a["score"], score_b["score"], places=12)
        negative = audit._residual_score(
            np.diag((-1.0, 3.0)), np.eye(2), np.array((0.2, 0.5)), (-1.0,), np.array((1.0, 0.0))[:, None]
        )
        self.assertTrue(np.isfinite(negative["score"]))

    def test_live_holdout_failure_is_unresolved_not_hidden(self):
        for target in audit.TARGETS:
            case = self.result["cases"][target]
            self.assertFalse(case["terminal_controls_pass"])
            self.assertEqual(case["terminal_control_failures"], ["residual_enrichment_controls"])
            for ell in audit.ELL_VALUES:
                sector = case["sectors"][str(ell)]
                self.assertFalse(sector["heldout_terminal_gate_pass"])
                self.assertIn("heldout_residual_gate", sector["terminal_control_failures"])
                self.assertEqual(
                    sector["conclusion"],
                    "UNRESOLVED_FINITE_ENRICHED_SPACE_SIGN_OR_CONVERGENCE",
                )
                self.assertTrue(all(row["rayleigh_ritz_monotonicity"]["monotone_with_allowance"] for row in sector["validation_rows"]))

    def test_validator_and_cache_fail_closed_on_mutations(self):
        self.assertTrue(audit.validate_result(self.result))
        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["sectors"]["0"]["validation_rows"][0]["stage_rows"][0]["positive_operator"] = False
        self.assertFalse(audit.validate_result(changed))
        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["sectors"]["0"]["terminal_validation"]["stage_rows"][2]["heldout"]["max_normalized_residual"] = 0.0
        self.assertFalse(audit.validate_result(changed))
        changed = copy.deepcopy(self.result)
        changed["cases"]["0.93"]["selections"]["1"]["selected_candidate_ids"].append("density_c0p25_w0p12")
        self.assertFalse(audit.validate_result(changed))
        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["backgrounds"]["24fm"]["field_residual_relative"] = float("nan")
        self.assertFalse(audit.validate_result(changed))
        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["sectors"]["0"]["validation_rows"][0]["stage_rows"][0]["internal_eigen_residual_absolute_Q_over_W0"][0] = 0.0
        self.assertFalse(audit.validate_result(changed))
        self.assertTrue(audit.validate_result(self.result))

    def test_cli_emits_strict_json_without_writing(self):
        before = set(HERE.iterdir())
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = audit.main([])
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(audit._validate_shape(payload))
        self.assertEqual(before, set(HERE.iterdir()))


if __name__ == "__main__":
    unittest.main()
