"""Focused tests for the coordinate-response and angular-resolvent audit."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_droplet_coordinate_response_audit as audit  # noqa: E402
import nvg_droplet_stability_audit as stability  # noqa: E402


class _ToyBackground:
    """Minimal live-background boundary for the private stability seam test."""

    def __init__(self):
        self.x = np.linspace(0.0, 1.0, 3)
        self.box_fm = 32.0
        self.design = object()
        self.solution = None
        self.solver_row = None


class DropletCoordinateResponseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The producer computes both fresh backgrounds once; every mutation
        # test below receives a deep copy and cannot poison the private cache.
        cls.result = audit.build_result()

    def test_schema_scope_and_full_protocol_are_live(self):
        self.assertEqual(self.result["schema_version"], audit.SCHEMA)
        self.assertEqual(self.result["status"], audit.STATUS)
        self.assertEqual(self.result["evidence_weight"], 0.0)
        self.assertEqual(set(self.result["cases"]), set(audit.TARGETS))
        self.assertEqual(self.result["model_scope"]["target_N"], audit.TARGET_N)
        self.assertTrue(self.result["model_scope"]["no_physics_or_parameter_change"])
        self.assertEqual(len(self.result["weighted_column_independence"]), 14)
        self.assertEqual(len(self.result["physical_low_eigenvector_checks"]), 14)
        self.assertTrue(self.result["synthetic_controls"]["all_checks_pass"])
        for target in audit.TARGETS:
            case = self.result["cases"][target]
            self.assertEqual(set(str(row["ell"]) for row in case["coordinate_comparison"]["sectors"]),
                             {str(ell) for ell in audit.ELL_VALUES})
            self.assertEqual(len(case["coordinate_comparison"]["sectors"]), 7)
            self.assertFalse(case["coordinate_comparison"]["all_protocol_rows_pass"])

    def test_original_branch_is_the_live_upstream_result(self):
        upstream = stability.build_result()
        for target in audit.TARGETS:
            self.assertEqual(self.result["cases"][target]["original"], upstream["cases"][target])
            self.assertEqual(
                self.result["cases"][target]["conservative_sector_conclusions"]["0"],
                "FINITE_BASIS_POSITIVE_ONLY",
            )

    def test_coordinate_oracles_recover_signs_and_preserve_congruence(self):
        hessian = np.diag((1.0, -1.0e-12))
        gram = np.diag((1.0, 1.0e-12))
        hbar, gbar, _, metadata = audit._unit_gram_transform(hessian, gram, None)
        spectrum = stability._generalized_spectrum(hbar, gbar, ["positive", "negative"])
        np.testing.assert_allclose(
            spectrum["eigenvalues_Q_over_W0"],
            np.array((-1.0, 1.0)),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        self.assertEqual(metadata["normalized_gram_rank"], 2)

        a = np.array(((1.0, 0.2), (0.1, 1.3)))
        h0 = np.diag((-0.7, 1.2))
        g0 = np.diag((1.0, 0.5))
        original = stability._generalized_spectrum(a.T @ h0 @ a, a.T @ g0 @ a, ["n", "p"])
        transformed_h, transformed_g, _, _ = audit._unit_gram_transform(a.T @ h0 @ a, a.T @ g0 @ a, None)
        transformed = stability._generalized_spectrum(transformed_h, transformed_g, ["n", "p"])
        np.testing.assert_allclose(
            transformed["eigenvalues_Q_over_W0"],
            original["eigenvalues_Q_over_W0"],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

        translated = stability._generalized_spectrum(
            np.diag((-1.0, 2.0)), np.eye(2), ["translation", "physical"], excluded_vector=np.array((1.0, 0.0))
        )
        self.assertLess(float(translated["eigenvalues_Q_over_W0"][0]), 0.0)
        self.assertGreater(float(translated["internal"]["eigenvalues_Q_over_W0"][0]), 0.0)
        self.assertTrue(audit._validate_synthetic_controls(audit._jsonable(audit._synthetic_controls())))

    def test_rank_replays_separate_raw_sensitivity_from_normalized_invariance(self):
        for replay in self.result["rank_replays"]:
            self.assertEqual(replay["row_count"], 42)
            self.assertEqual(replay["expected_row_count"], 42)
            self.assertEqual(len(replay["rows"]), 42)
            self.assertEqual(len(replay["raw_rows"]), 21)
            self.assertEqual(len(replay["normalized_rows"]), 21)
            self.assertFalse(replay["all_raw_rank_replays_pass"])
            self.assertTrue(replay["all_normalized_replays_pass"])
            self.assertFalse(replay["all_rank_replays_pass"])
            self.assertEqual(
                replay["raw_rank_invariance_failures"],
                [row for row in replay["raw_rows"] if not row["passes"]],
            )
            self.assertFalse(replay["normalized_rank_invariance_failures"])
            self.assertTrue(all(row["passes"] for row in replay["normalized_rows"]))
        self.assertEqual(
            self.result["cases"]["0.90"]["conservative_sector_conclusions"]["6"],
            "UNRESOLVED_FINITE_BASIS_SIGN_OR_CONVERGENCE",
        )
        self.assertEqual(
            self.result["cases"]["0.93"]["conservative_sector_conclusions"]["4"],
            "FINITE_BASIS_POSITIVE_ONLY",
        )
        self.assertTrue(self.result["cases"]["0.93"]["normalized_replay_controls_pass"])

    def test_failed_case_control_downgrades_actual_classification(self):
        case = self.result["cases"]["0.93"]
        damaged = copy.deepcopy(case["equilibrated"])
        damaged["terminal_controls_pass"] = False
        comparison = copy.deepcopy(case["coordinate_comparison"])
        comparison["equilibrated_controls_pass"] = False
        self.assertEqual(
            audit._conservative_conclusion(
                case["original"], damaged, comparison, 0,
            ),
            "UNRESOLVED_FINITE_BASIS_SIGN_OR_CONVERGENCE",
        )

    def _aggregate_live_fixture(self, replay=None, physical=None):
        """Run the real aggregation against fixture-shaped live dependencies."""

        target = "0.93"
        case = self.result["cases"][target]
        replay = copy.deepcopy(
            self.result["rank_replays"][1] if replay is None else replay
        )
        physical = copy.deepcopy(
            [
                row
                for row in self.result["physical_low_eigenvector_checks"]
                if row["target_y"] == target
            ]
            if physical is None
            else physical
        )

        def source_case(requested_target, spectrum_transform=None):
            self.assertEqual(requested_target, target)
            return copy.deepcopy(
                case["original"] if spectrum_transform is None else case["equilibrated"]
            )

        with patch.object(stability, "_case_result", side_effect=source_case):
            return audit._case_result(
                target,
                normalized_replay=replay,
                physical_checks=physical,
            )

    def _assert_live_aggregation_downgraded(self, actual):
        self.assertFalse(actual["all_conservative_controls_pass"])
        self.assertFalse(actual["normalized_replay_controls_pass"] and actual["physical_replay_diagnostics_pass"])
        self.assertEqual(
            set(actual["conservative_sector_conclusions"].values()),
            {"UNRESOLVED_FINITE_BASIS_SIGN_OR_CONVERGENCE"},
        )

    def test_live_aggregation_accepts_positive_replay_and_labels(self):
        actual = self._aggregate_live_fixture()
        self.assertTrue(actual["normalized_replay_controls_pass"])
        self.assertTrue(actual["physical_replay_diagnostics_pass"])
        self.assertTrue(actual["all_conservative_controls_pass"])
        self.assertEqual(
            actual["conservative_sector_conclusions"],
            self.result["cases"]["0.93"]["conservative_sector_conclusions"],
        )

    def test_live_aggregation_rejects_stale_replay_rank(self):
        replay = copy.deepcopy(self.result["rank_replays"][1])
        replay["normalized_rows"][0]["replayed_raw_rank"] -= 1
        self.assertTrue(replay["normalized_rows"][0]["passes"])
        self._assert_live_aggregation_downgraded(self._aggregate_live_fixture(replay=replay))

    def test_live_aggregation_rejects_stale_replay_eigenvalue(self):
        replay = copy.deepcopy(self.result["rank_replays"][1])
        replay["normalized_rows"][0]["physical_checks"][
            "replayed_coordinate_eigenvalue_Q_over_W0"
        ] *= -1.0
        self.assertTrue(replay["normalized_rows"][0]["physical_checks"]["passes"])
        self._assert_live_aggregation_downgraded(self._aggregate_live_fixture(replay=replay))

    def test_live_aggregation_rejects_stale_physical_norm(self):
        physical = [
            copy.deepcopy(row)
            for row in self.result["physical_low_eigenvector_checks"]
            if row["target_y"] == "0.93"
        ]
        physical[0]["equilibrated_physical_vector"]["physical_gram_norm"] = 2.0
        self.assertTrue(physical[0]["passes"])
        self._assert_live_aggregation_downgraded(self._aggregate_live_fixture(physical=physical))

    def test_banded_p1_matches_small_dense_oracle(self):
        x = np.linspace(0.0, 2.0, 9)
        y = 0.7 + 0.1 * np.sin(x) + 0.02 * x**2
        h = float(x[1] - x[0])
        nodes = np.arange(1, x.size - 1)
        mass_diagonal = np.full(nodes.size, 2.0 * h / 3.0)
        mass_off = np.full(nodes.size - 1, h / 6.0)

        def fake_operator(design, grid_x, grid_y, ell):
            tau = float(ell * (ell + 1))
            base_diagonal = 2.0 + 0.03 * np.asarray(grid_y)[nodes]
            base_off = np.full(nodes.size - 1, -0.12)
            return (
                base_off + tau * mass_off,
                base_diagonal + tau * mass_diagonal,
                base_off + tau * mass_off,
                nodes,
                float(np.min(base_diagonal + tau * mass_diagonal)),
            )

        with patch.object(audit.stability, "_operator_tridiagonal", side_effect=fake_operator):
            bands = {
                ell: audit._p1_operator_and_angular_mass(object(), x, y, ell)
                for ell in (2, 3, 6)
            }
            source = np.linspace(0.3, 0.9, nodes.size)
            solutions = {}
            for ell, item in bands.items():
                diagonal = item["operator_diagonal"]
                lower = item["operator_lower"]
                upper = item["operator_upper"]
                dense = np.diag(diagonal)
                dense += np.diag(upper, 1) + np.diag(lower, -1)
                rhs = item["load_weights"] * source
                solved, pivot = stability._solve_tridiagonal(lower, diagonal, upper, rhs)
                np.testing.assert_allclose(solved, np.linalg.solve(dense, rhs), rtol=1e-12, atol=1e-12)
                self.assertGreater(pivot, 0.0)
                mass_dense = np.diag(item["mass_diagonal"])
                mass_dense += np.diag(item["mass_upper"], 1) + np.diag(item["mass_lower"], -1)
                left = np.linspace(-0.2, 0.8, nodes.size)
                right = np.linspace(0.6, -0.4, nodes.size)
                self.assertAlmostEqual(
                    audit._band_bilinear(
                        left,
                        item["mass_diagonal"],
                        item["mass_lower"],
                        item["mass_upper"],
                        right,
                    ),
                    float(left @ mass_dense @ right),
                    places=13,
                )
                solutions[ell] = (item, solved)
            item2, w2 = solutions[2]
            item3, w3 = solutions[3]
            v = np.linspace(0.5, -0.1, nodes.size)
            scalar = audit._band_bilinear(v, item2["mass_diagonal"], item2["mass_lower"], item2["mass_upper"], v)
            cross = audit._band_bilinear(w3, item2["mass_diagonal"], item2["mass_lower"], item2["mass_upper"], w2)
            tau1, tau2 = 6.0, 12.0
            q1 = tau1 * scalar + float((item2["load_weights"] * source) @ w2)
            q2 = tau2 * scalar + float((item3["load_weights"] * source) @ w3)
            self.assertAlmostEqual(q2 - q1, (tau2 - tau1) * (scalar - cross), places=12)

    def test_weighted_columns_and_physical_vectors_remain_diagnostics(self):
        for row in self.result["weighted_column_independence"]:
            self.assertTrue(row["physical_column_independence_is_diagnostic_only"])
            self.assertEqual(row["rank_matches_actual_gram"], row["weighted_svd_rank"] == row["gram_rank"])
        self.assertTrue(all(row["passes"] for row in self.result["physical_low_eigenvector_checks"]))
        for row in self.result["physical_low_eigenvector_checks"]:
            self.assertLessEqual(row["rayleigh_relative_error_max"], audit.SPECTRUM_COMPARISON_LIMIT)

    def test_schur_and_angular_controls_are_explicit(self):
        for row in self.result["density_schur_diagnostics"]:
            if row["ell"] == 1:
                self.assertEqual(row["status"], "SKIPPED_JOINT_TRANSLATION_CONSTRAINT")
            else:
                self.assertTrue(row["density_block_positive"])
                self.assertTrue(row["inertia_identity_pass"])
                self.assertTrue(row["reconstruction_pass"])
            self.assertTrue(row["all_checks_pass"])
        for item in self.result["angular_resolvent_diagnostics"]:
            self.assertEqual(item["row_count"], 72)
            self.assertTrue(item["all_continuum_identities_pass"])
            self.assertTrue(item["all_p1_identities_pass"])
            self.assertTrue(item["all_operator_controls_pass"])
            self.assertTrue(item["all_source_field_controls_pass"])
            self.assertTrue(item["fixed_profile_pure_density_vector_decrease_all_pass"])
            self.assertTrue(item["all_continuum_refinement_pass"])
            self.assertTrue(item["all_angular_controls_pass"])
            self.assertEqual(len(item["continuum_refinement_diagnostics"]), 24)
            self.assertFalse(item["continuum_refinement_failures"])
            self.assertFalse(item["continuum_identity_failures"])
            self.assertFalse(item["p1_identity_failures"])
            for row in item["rows"]:
                self.assertLessEqual(row["continuum_identity_relative_error"], audit.ANGULAR_CONTINUUM_LIMIT)
                self.assertLessEqual(row["p1_identity_relative_error"], audit.ANGULAR_DISCRETE_LIMIT)
                self.assertEqual(row["p1_operator_representation"], "exact_tridiagonal_bands")

    def test_private_transform_seam_keeps_default_record_shape(self):
        columns = {
            "u": np.zeros((3, 1)),
            "v": np.zeros((3, 1)),
            "vprime": np.zeros((3, 1)),
            "labels": ["u", "v"],
            "translation_basis_present": False,
            "translation_cutoff_min": 0.0,
        }
        hessian = np.diag((1.0, 2.0))
        gram = np.eye(2)
        constraints = {
            "number_constraint_residuals_relative": [0.0, 0.0],
            "number_constraint_denominators": [1.0, 1.0],
            "number_constraint_residual_max": 0.0,
            "number_constraint_zero_vector_count": 0,
        }
        with patch.object(stability, "_basis_columns", return_value=columns), \
             patch.object(stability, "_hessian_quadratic", return_value=(hessian, {
                 "operator_min_pivot": 1.0, "positive_operator": True,
             })), \
             patch.object(stability, "_gram_matrix", return_value=gram), \
             patch.object(stability, "_number_constraint_diagnostics", return_value=constraints):
            default = stability._spectrum_record(_ToyBackground(), 0, 1, 2)
            transformed = stability._spectrum_record(
                _ToyBackground(), 0, 1, 2, spectrum_transform=audit._unit_gram_transform
            )
        self.assertNotIn("spectrum_transform_metadata", default)
        self.assertIn("spectrum_transform_metadata", transformed)
        self.assertEqual(default["eigenvalues_Q_over_W0"].tolist(), transformed["eigenvalues_Q_over_W0"].tolist())
        self.assertEqual(default["gram_rank"], transformed["gram_rank"])

    def test_shape_guard_rejects_actual_classification_and_data_corruption(self):
        changed = copy.deepcopy(self.result)
        changed["rank_replays"][0]["rows"].pop()
        self.assertFalse(audit._validate_shape(changed))

        changed = copy.deepcopy(self.result)
        changed["cases"]["0.90"]["conservative_sector_conclusions"]["6"] = "FINITE_BASIS_POSITIVE_ONLY"
        self.assertFalse(audit._validate_shape(changed))

        changed = copy.deepcopy(self.result)
        changed["angular_resolvent_diagnostics"][0]["rows"][0]["continuum_prediction_Q_difference"] *= -1.0
        self.assertFalse(audit._validate_shape(changed))

        changed = copy.deepcopy(self.result)
        changed["angular_resolvent_diagnostics"][0]["rows"][0]["q1_Q_over_W0"] = float("nan")
        self.assertFalse(audit._validate_shape(changed))

        changed = copy.deepcopy(self.result)
        changed["synthetic_controls"]["oracle_2x2"]["computed_eigenvalues_after_unit_gram"][0] = 0.0
        self.assertFalse(audit._validate_shape(changed))

        changed = copy.deepcopy(self.result)
        changed["rank_replays"][0]["normalized_rows"][0]["replayed_raw_rank"] += 1
        self.assertFalse(audit._validate_shape(changed))

    def test_cli_shape_guard_rejects_missing_coordinate_replay(self):
        damaged = copy.deepcopy(self.result)
        damaged["rank_replays"][0]["rows"].pop()
        output = io.StringIO()
        with patch.object(audit, "build_result", return_value=damaged), \
             patch.object(audit, "validate_result", side_effect=AssertionError("recursive validation")), \
             redirect_stdout(output):
            status = audit.main([])
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 2)
        self.assertEqual(payload["status"], "INVALID_OR_FAILED_DROPLET_COORDINATE_RESPONSE_AUDIT")

    def test_cache_isolation_and_strict_json(self):
        first = audit.build_result()
        first["cases"]["0.90"]["coordinate_comparison"]["sectors"].clear()
        first["angular_resolvent_diagnostics"][0]["rows"].clear()
        second = audit.build_result()
        self.assertEqual(second, self.result)
        roundtripped = json.loads(json.dumps(second, ensure_ascii=False, allow_nan=False))
        self.assertTrue(audit._validate_shape(roundtripped))


if __name__ == "__main__":
    unittest.main()
