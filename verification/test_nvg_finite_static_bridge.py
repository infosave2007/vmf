"""Focused guards for the conditional finite-static bridge.

These tests intentionally exercise protocol/data seams without turning a
serialized result table into solver input.  The complete live bridge run is a
separate terminal computation owned by the implementer.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_finite_monopole as static
import nvg_finite_static_bridge as bridge


class FiniteStaticBridgeTests(unittest.TestCase):
    @staticmethod
    def _valid_terminal_row() -> dict[str, object]:
        """Small gate-complete row for protocol/validator unit seams."""

        row = {
            "converged": True,
            "solver_status": 0,
            "stationarity_acceptance": True,
            "localized_vacuum_exterior": True,
            "binding_condition_E_per_A_lt_M": True,
            "energy_per_A_minus_M_MeV": -8.0,
            "collocation_max_rms": 1.0e-10,
            "conserved_NZ_relative_error": 1.0e-10,
            "field_residual_relative": 1.0e-10,
            "first_order_state_residual_max_abs_dimensionless": 1.0e-10,
            "first_order_state_y_residual_max_abs_dimensionless": 1.0e-10,
            "first_order_state_vector_residual_max_abs_dimensionless": 1.0e-10,
            "first_order_state_coulomb_residual_max_abs_dimensionless": 1.0e-10,
            "first_order_field_residual_max_abs_dimensionless": 1.0e-10,
            "first_order_scalar_ode_residual_max_abs_dimensionless": 1.0e-10,
            "first_order_vector_ode_residual_max_abs_dimensionless": 1.0e-10,
            "first_order_field_residual_relative": 1.0e-10,
            "scalar_residual_max_abs_dimensionless": 1.0e-10,
            "vector_residual_max_abs_dimensionless": 1.0e-10,
            "coulomb_residual_relative": 1.0e-10,
            "first_order_coulomb_residual_max_abs_dimensionless": 1.0e-10,
            "first_order_coulomb_residual_relative": 1.0e-10,
            "kkt_max_error_MeV": 1.0e-10,
            "gauss_identity_relative": 1.0e-10,
            "coulomb_identity_relative": 1.0e-10,
            "central_y": 0.5,
            "min_y": 0.5,
            "central_density_fm_minus3": 0.1,
            "boundary_densities_dimensionless": {"neutron": 0.0, "proton": 0.0},
            "outer_number_fraction_max": 0.0,
            "rms_neutron_radius_fm": 3.0,
            "rms_point_proton_radius_fm": 3.0,
            "chemical_potentials_MeV": {"neutron": 930.0, "proton": 930.0},
        }
        nested_keys = (
            "stationarity_acceptance", "localized_vacuum_exterior", "binding_condition_E_per_A_lt_M",
            "energy_per_A_minus_M_MeV", "central_y", "min_y", "central_density_fm_minus3",
            "chemical_potentials_MeV", "boundary_densities_dimensionless", "outer_number_fraction_max",
            "conserved_NZ_relative_error", "field_residual_relative",
            "first_order_state_residual_max_abs_dimensionless", "first_order_state_y_residual_max_abs_dimensionless",
            "first_order_state_vector_residual_max_abs_dimensionless", "first_order_state_coulomb_residual_max_abs_dimensionless",
            "first_order_field_residual_max_abs_dimensionless", "first_order_scalar_ode_residual_max_abs_dimensionless",
            "first_order_vector_ode_residual_max_abs_dimensionless", "first_order_field_residual_relative",
            "scalar_residual_max_abs_dimensionless", "vector_residual_max_abs_dimensionless",
            "coulomb_residual_relative", "first_order_coulomb_residual_max_abs_dimensionless",
            "first_order_coulomb_residual_relative", "kkt_max_error_MeV", "gauss_identity_relative",
            "coulomb_identity_relative",
        )
        row["nested_diagnostic"] = {key: row[key] for key in nested_keys}
        return row

    @staticmethod
    def _valid_profile_comparison() -> dict[str, object]:
        return {
            "passed": True,
            "common_physical_box_fm": 24.0,
            "max_abs_y_diff": 0.0,
            "max_abs_vector_potential_diff_MeV": 0.0,
            "max_abs_coulomb_potential_diff_MeV": 0.0,
            "max_abs_species_density_diff_fm_minus3": 0.0,
            "max_abs_mu_diff_MeV": 0.0,
            "abs_binding_per_A_diff_MeV": 0.0,
            "species_radius_diffs_fm": {"neutron": 0.0, "point_proton": 0.0},
        }

    def test_input_bundle_is_cited_data_only_and_nz_consistent(self) -> None:
        payload = bridge._load_inputs()
        self.assertEqual(payload["status"], "CITED_EVALUATED_INPUTS_ONLY_NOT_MODEL_OUTPUTS")
        for nucleus in bridge.BRIDGE_NUCLEI:
            row = payload["binding"]["values"][nucleus]
            self.assertEqual(row["A"], row["N"] + row["Z"])
            self.assertIn("role", row)
        # Data loading returns a deep copy, not a mutable module-level object.
        changed = copy.deepcopy(payload)
        changed["binding"]["values"]["Ca40"]["B_atom_per_A_MeV"] = -1.0
        self.assertGreater(payload["binding"]["values"]["Ca40"]["B_atom_per_A_MeV"], 0.0)

    def test_electron_formula_sign_units_and_atomic_to_bare_conversion(self) -> None:
        be1 = bridge.electron_binding_eV(1)
        be20 = bridge.electron_binding_eV(20)
        correction = bridge.electron_correction_MeV(20)
        self.assertGreater(be20, 20.0 * be1)
        self.assertAlmostEqual(correction, (be20 - 20.0 * be1) * 1.0e-6, places=14)
        row = bridge._load_inputs()["binding"]["values"]["Ca40"]
        bare = bridge.bare_binding_per_A_MeV(row)
        self.assertLess(bare, row["B_atom_per_A_MeV"])
        self.assertAlmostEqual(bare, 8.550847043275436, places=12)
        with self.assertRaises(bridge.FiniteStaticBridgeError):
            bridge.electron_correction_MeV(20.5)

    def test_continuous_correlated_scale_keeps_bulk_inputs_and_d2(self) -> None:
        base = static.FiniteDesign.from_continuous("W8.90", 1.0)
        half = static.FiniteDesign.from_continuous("W8.90", 0.5)
        self.assertEqual(static.SCALES, ("0.5", "1", "2"))
        self.assertEqual(base.d, 2.0)
        self.assertEqual(half.d, 2.0)
        self.assertAlmostEqual(half.W0, 0.5 * base.W0)
        self.assertAlmostEqual(half.M, base.M)
        self.assertAlmostEqual(half.momega, base.momega)
        self.assertAlmostEqual(half.gomega, base.gomega)
        self.assertTrue(np.allclose(np.asarray(half.coefficients_dim), np.asarray(base.coefficients_dim) / 0.5**4))
        self.assertAlmostEqual(half.n0_dim, base.n0_dim / 0.5**3)
        # Continuous values require the explicit opt-in seam.
        with self.assertRaises(static.FiniteMonopoleError):
            static.FiniteDesign.from_family("W8.90", "0.25")

    def test_static_solver_reports_raw_species_moments_and_nz(self) -> None:
        design = static.FiniteDesign.from_continuous("W8.90", 0.25)
        solved = static._solve_static_once(
            design,
            "Ca40",
            "no_rho",
            box_fm=16.0,
            nodes=401,
            tol=2.0e-6,
            factor=1.0,
            nucleus_data=static.BRIDGE_NUCLEI,
        )
        self.assertTrue(solved.row["converged"])
        self.assertLess(abs(solved.row["conserved_N_integrated"] - 20.0), 1.0e-5)
        self.assertLess(abs(solved.row["conserved_Z_integrated"] - 20.0), 1.0e-5)
        self.assertGreater(solved.row["rms_point_proton_radius_fm"], 0.0)
        self.assertGreater(solved.row["rms_neutron_radius_fm"], 0.0)
        ints = solved.row["species_radius_integrals_dimensionless"]
        self.assertIn("first_order_field_residual_relative", solved.row)
        self.assertIn("nested_diagnostic", solved.row)
        self.assertEqual(solved.row["diagnostic_rule"]["phase_search"], False)
        self.assertAlmostEqual(
            solved.row["rms_point_proton_radius_fm"] ** 2,
            (design.hbarc / design.W0) ** 2 * ints["proton_x4"] / ints["proton_x2"],
            places=10,
        )

    def test_each_first_order_state_corruption_rejects_primary_and_nested_gate(self) -> None:
        design = static.FiniteDesign.from_continuous("W8.93", 0.22682948649281853)
        solved = static._solve_static_once(
            design,
            "Ca40",
            "no_rho",
            box_fm=24.0,
            nodes=801,
            tol=2.0e-8,
            factor=0.8,
            nucleus_data=static.BRIDGE_NUCLEI,
        )
        self.assertTrue(solved.row["stationarity_acceptance"])
        original = solved.solution.sol
        for state_index, metric_name in (
            (0, "first_order_state_y_residual_max_abs_dimensionless"),
            (2, "first_order_state_vector_residual_max_abs_dimensionless"),
            (4, "first_order_state_coulomb_residual_max_abs_dimensionless"),
        ):
            corrupted = copy.copy(solved)
            corrupted.solution = copy.copy(solved.solution)

            def corrupt(x, nu=0, state_index=state_index):
                values = np.asarray(original(x, nu), dtype=float).copy()
                if nu == 1:
                    values[state_index] += 0.01
                return values

            corrupted.solution.sol = corrupt
            diagnostics = static._static_diagnostics(corrupted, solved.solution.x)
            self.assertFalse(diagnostics["stationarity_acceptance"])
            self.assertGreater(diagnostics[metric_name], static.FIELD_RESIDUAL_LIMIT)

    def test_shared_row_gate_recomputes_numeric_stationarity_localization_and_binding(self) -> None:
        row = self._valid_terminal_row()
        self.assertTrue(bridge._row_numerical_gate(row))
        for key, bad in (
            ("first_order_state_y_residual_max_abs_dimensionless", 2.0e-4),
            ("field_residual_relative", 2.0e-4),
            ("outer_number_fraction_max", 2.0e-8),
            ("energy_per_A_minus_M_MeV", 1.0),
        ):
            altered = dict(row)
            altered[key] = bad
            self.assertFalse(bridge._row_numerical_gate(altered), key)
        nested_altered = copy.deepcopy(row)
        nested_altered["nested_diagnostic"]["first_order_state_vector_residual_max_abs_dimensionless"] = 2.0e-4
        # The combined row remains green; the nested numeric control itself
        # must still be read and rejected.
        self.assertFalse(bridge._row_numerical_gate(nested_altered))

    def test_serialized_validator_recomputes_numeric_profile_metrics_not_green_flag(self) -> None:
        # Build one real terminal witness so all nested row metrics are present.
        case = bridge._terminal_public(
            bridge._terminal_case("W8.93", 0.22682948649281853, "Ca40", multi_seed=True)
        )
        self.assertTrue(bridge._validate_terminal_public(case))
        mutated = copy.deepcopy(case)
        selected = mutated["branch_candidates"][0]
        selected["domain_profile_comparison"]["max_abs_y_diff"] = bridge.LINEAGE_MAX_Y_DIFF * 2.0
        # Leave every passed flag green: the validator must recompute.
        self.assertFalse(bridge._validate_terminal_public(mutated))

    def test_coulomb_tail_has_positive_potential_and_negative_slope(self) -> None:
        design = static.FiniteDesign.from_continuous("W8.90", 0.25)
        x = np.linspace(0.0, 3.0, 1001)
        density = np.exp(-x * x)
        potential, derivative = static._solve_coulomb_density(design, x, density)
        self.assertTrue(np.all(potential > 0.0))
        self.assertTrue(np.all(derivative[1:] <= 1.0e-12))
        self.assertAlmostEqual(derivative[-1] + potential[-1] / x[-1], 0.0, places=12)

    def test_radius_mapping_keeps_unknown_corrections_explicit(self) -> None:
        inputs = bridge._load_inputs()
        row = {
            "rms_point_proton_radius_fm": 3.2,
            "rms_neutron_radius_fm": 3.25,
            "rms_baryon_radius_fm": 3.22,
            "species_radius_integrals_dimensionless": {"proton_x2": 1.0, "proton_x4": 2.0},
        }
        mapped = bridge._radius_mapping(row, "Ca40", inputs)
        self.assertIn("R1b_square_fm2", mapped)
        self.assertIsNone(mapped["uncomputed_corrections"]["delta_SO_fm2"])
        self.assertEqual(mapped["sensitivity_of_delta_required"]["delta_required_per_rn2_fm2"], -1.0)

    def test_binding_roles_keep_ca_anchor_separate_from_heldout_rows(self) -> None:
        values = bridge._load_inputs()["binding"]["values"]
        self.assertEqual(values["Ca40"]["role"], "sole_conditional_calibration_anchor")
        for nucleus in ("Zr90", "Pb208"):
            self.assertEqual(values[nucleus]["role"], "heldout_descriptive_comparison_only")
        self.assertFalse(bridge._load_inputs()["charge_radii"]["values"]["Zr90"]["role"].startswith("sole_"))

    def test_binding_targets_and_comparisons_are_linked_to_cited_inputs(self) -> None:
        retained = HERE.parent / "Lunacy/runs/observable-bridge-2026-09-15/evidence_R2/bridge_result_r2.json"
        if not retained.exists():
            self.skipTest("retained R2 result is unavailable")
        result = json.loads(retained.read_text(encoding="utf-8"))
        self.assertTrue(bridge._validate_structure(result))

        altered = copy.deepcopy(result)
        altered["input_targets"]["Zr90"]["B_nuc_target_per_A_MeV"] += 1.0e-2
        self.assertFalse(bridge._validate_structure(altered))

        altered = copy.deepcopy(result)
        altered["calibrations"]["W8.93"]["target"]["B_nuc_target_per_A_MeV"] += 1.0e-2
        self.assertFalse(bridge._validate_structure(altered))

        altered = copy.deepcopy(result)
        altered["calibrations"]["W8.93"]["root"]["target_B_nuc_per_A_MeV"] += 1.0e-2
        self.assertFalse(bridge._validate_structure(altered))

        for field in (
            "B_atom_input_per_A_MeV",
            "experimental_uncertainty_MeV_per_A",
            "electronic_convention_correction_total_MeV",
            "B_nuc_target_per_A_MeV",
            "model_minus_B_nuc_target_per_A_MeV",
        ):
            altered = copy.deepcopy(result)
            prediction = altered["calibrations"]["W8.93"]["frozen_predictions"][1]
            prediction["binding_comparison"][field] += 1.0e-2
            self.assertFalse(bridge._validate_prediction(prediction), field)
            self.assertFalse(bridge._validate_structure(altered), field)

        coordinated = copy.deepcopy(result)
        coordinated["input_targets"]["Zr90"]["B_nuc_target_per_A_MeV"] += 0.02
        comparison = coordinated["calibrations"]["W8.93"]["frozen_predictions"][1]["binding_comparison"]
        comparison["B_nuc_target_per_A_MeV"] += 0.02
        comparison["model_minus_B_nuc_target_per_A_MeV"] -= 0.02
        self.assertFalse(bridge._validate_structure(coordinated))

    def test_heldout_inputs_do_not_change_calibration_bracket_or_choice(self) -> None:
        """Exercise calibration with doubles while changing every held-out input."""

        base_inputs = bridge._load_inputs()
        calls: list[tuple[float, dict[str, object]]] = []

        def fake_probe(family: str, scale: float, nucleus: str = "Ca40") -> dict[str, object]:
            del family, nucleus
            binding = 8.0 + 2.0 * float(scale)
            row = {
                "localized_vacuum_exterior": True,
                "energy_per_A_minus_M_MeV": -binding,
                "chemical_potentials_MeV": {"neutron": 930.0, "proton": 930.0},
                "central_y": 0.5,
            }
            return {"scale": float(scale), "_representatives": [(row, object())]}

        def fake_root(family: str, bracket: dict[str, object], target: float, *, cache: dict[object, object]) -> dict[str, object]:
            del family, cache
            calls.append((target, dict(bracket)))
            return {
                "status": "CALIBRATED_CONDITIONAL_ROOT",
                "scale": 0.25,
                "scale_label": "0.25",
                "target_B_nuc_per_A_MeV": target,
                "binding_per_A_MeV": target,
                "binding_residual_MeV": 0.0,
                "bracket_final": [0.25, 0.5],
                "bracket_width": 0.25,
                "accepted": True,
                "_final_case": {},
            }

        def run(inputs: dict[str, object]) -> dict[str, object]:
            with patch.object(bridge, "_coarse_probe", side_effect=fake_probe), \
                patch.object(bridge, "_adaptive_probe_series", return_value=[]), \
                patch.object(bridge, "_find_root", side_effect=fake_root), \
                patch.object(bridge, "_envelope_derivative", return_value={"status": "stub"}), \
                patch.object(bridge, "_prediction_case", return_value={}), \
                patch.object(bridge, "_frozen_prediction", side_effect=lambda family, scale, nucleus, inputs, case: {"nucleus": nucleus}):
                return bridge._family_calibration("W8.93", inputs, cache={})

        baseline = run(base_inputs)
        heldout = copy.deepcopy(base_inputs)
        for nucleus in ("Zr90", "Pb208"):
            heldout["binding"]["values"][nucleus]["B_atom_per_A_MeV"] += 0.37
            heldout["binding"]["values"][nucleus]["B_atom_per_A_sigma_MeV"] += 0.11
        for nucleus in bridge.BRIDGE_NUCLEI:
            heldout["charge_radii"]["values"][nucleus]["R_ch_fm"] += 0.37
            heldout["charge_radii"]["values"][nucleus]["R_ch_sigma_fm"] += 0.11
        changed = run(heldout)

        self.assertEqual(baseline["target"], changed["target"])
        self.assertEqual(baseline["root"]["target_B_nuc_per_A_MeV"], changed["root"]["target_B_nuc_per_A_MeV"])
        self.assertEqual(baseline["root"]["bracket_final"], changed["root"]["bracket_final"])
        self.assertEqual(baseline["root"]["scale"], changed["root"]["scale"])
        self.assertEqual(baseline["localized_brackets"], changed["localized_brackets"])
        self.assertEqual(calls[0][0], calls[1][0])
        self.assertEqual(bridge._public_bracket(calls[0][1]), bridge._public_bracket(calls[1][1]))

    def test_lowest_controlled_branch_uses_domain_energy_not_fit_residual(self) -> None:
        row_a = self._valid_terminal_row()
        row_b = self._valid_terminal_row()
        row_a["energy_per_A_minus_M_MeV"] = -8.0
        row_b["energy_per_A_minus_M_MeV"] = -8.0
        domain_a = dict(row_a, energy_per_A_minus_M_MeV=-8.01)
        domain_b = dict(row_b, energy_per_A_minus_M_MeV=-7.99)
        fresh = dict(row_a)
        solution_a, solution_b = object(), object()
        fresh_solution = object()
        profile = self._valid_profile_comparison()

        def fake_solve(design, nucleus, *, level, factor=None, previous=None, recover=True):
            del design, nucleus, recover
            if level["label"] == "fine24":
                if factor == bridge.SEED_FACTORS[0]:
                    return dict(row_a), solution_a
                return dict(row_b), solution_b
            if level["label"] == "fresh24_independent":
                return dict(fresh), fresh_solution
            if previous is solution_a:
                return dict(domain_a), object()
            if previous is solution_b:
                return dict(domain_b), object()
            raise AssertionError(f"unexpected solve level/previous: {level!r} {previous!r}")

        groups = [[(dict(row_a, seed_factor=bridge.SEED_FACTORS[0]), solution_a)],
                  [(dict(row_b, seed_factor=bridge.SEED_FACTORS[-1]), solution_b)]]
        with patch.object(bridge, "_solve", side_effect=fake_solve), \
            patch.object(bridge, "_group_same_scale_candidates", return_value=(groups, [])), \
            patch.object(bridge, "_profile_consistency", return_value=profile):
            case = bridge._terminal_case("W8.93", 0.2268, "Ca40", multi_seed=True)

        self.assertTrue(case["terminal_protocol_acceptance"])
        self.assertEqual(case["selected_branch_id"], "fine_branch_1")
        self.assertEqual(case["lowest_found_branch_id"], "fine_branch_1")
        self.assertEqual(case["branch_selection_basis"], "lowest_energy_among_fully_controlled_localized_branches")
        self.assertAlmostEqual(case["binding_per_A_MeV"], 8.01)

    def test_failure_and_nonlocalized_rows_are_not_promoted(self) -> None:
        self.assertFalse(bridge._terminal_valid({"terminal_protocol_acceptance": False, "terminal_localized": True}))
        self.assertFalse(bridge._terminal_valid({"terminal_protocol_acceptance": True, "terminal_localized": False, "binding_per_A_MeV": 8.0}))
        self.assertFalse(bridge._is_localized({"converged": True, "localized_vacuum_exterior": False, "stationarity_acceptance": True}))

    def test_validator_recomputes_nested_terminal_rows_not_forged_flags(self) -> None:
        forged = {
            "terminal_protocol_acceptance": True,
            "terminal_localized": True,
            "binding_per_A_MeV": 8.0,
            "fine_selected": {"converged": False},
            "domain40_check": {"converged": True},
            "fresh24_check": {"converged": True},
            "fresh24_initial_attempt": {"converged": True},
            "refinement_summary": {
                "domain_energy_pass": True,
                "fresh24_energy_pass": True,
                "domain_species_radius_pass": True,
                "fresh24_species_radius_pass": True,
                "all_terminal_rows_converged_stationary_localized_bound": True,
            },
            "lineage": {"status": "CONSISTENT"},
        }
        self.assertFalse(bridge._terminal_valid(forged))

    def test_repair_contract_removes_grid_postselection_and_fresh_substitution(self) -> None:
        source = (HERE / "nvg_finite_static_bridge.py").read_text(encoding="utf-8")
        self.assertIn("RECOVERY_LEVELS", source)
        self.assertIn('"nodes": 2401', source)
        self.assertIn('"nodes": 4001', source)
        self.assertNotIn("finer_dense_edge_phase", source)
        self.assertNotIn("fresh24_warm_start_recovery_after_retained_failure", source)
        self.assertNotIn('"same_branch_only": True', source)

    def test_default_cli_contract_is_strict_and_output_is_opt_in(self) -> None:
        source = (HERE / "nvg_finite_static_bridge.py").read_text(encoding="utf-8")
        self.assertIn("--output", source)
        self.assertIn("no network access", source)
        self.assertNotIn("build_result()", source.split("if __name__", 1)[0].split("def calculate", 1)[0])
        payload = json.loads(json.dumps(bridge._input_targets(bridge._load_inputs()), allow_nan=False))
        self.assertIn("Ca40", payload)

    def test_recovery_runs_declared_bvp_ladder_and_preserves_original_failure(self) -> None:
        initial = self._valid_terminal_row()
        initial.update({
            "first_order_state_y_residual_max_abs_dimensionless": 2.0e-4,
            "stationarity_acceptance": False,
            "solver_status": 0,
        })
        recovery_2401 = self._valid_terminal_row()
        recovery_2401.update({"solver_status": 0, "stationarity_acceptance": False})
        recovery_4001 = self._valid_terminal_row()
        recovery_4001.update({"solver_status": 0})
        calls: list[tuple[str, int, float, object | None]] = []
        live = object()

        def fake_raw(design, nucleus, *, level, factor=None, previous=None):
            calls.append((str(level["label"]), int(level["nodes"]), float(level["box_fm"]), previous))
            if level["label"] == "fine24":
                initial["protocol_label"] = "fine24"
                return initial, live
            if level["label"] == "recovery2401":
                item = dict(recovery_2401)
                item["protocol_label"] = "recovery2401"
                return item, live
            item = dict(recovery_4001)
            item["protocol_label"] = "recovery4001"
            return item, live

        design = static.FiniteDesign.from_continuous("W8.90", 0.25)
        with patch.object(bridge, "_raw_solve", side_effect=fake_raw):
            row, solved = bridge._solve(design, "Ca40", level=bridge.FINE_LEVEL, factor=1.0, previous=live)
        self.assertIs(solved, live)
        self.assertEqual([item[:2] for item in calls], [("fine24", 801), ("recovery2401", 2401), ("recovery4001", 4001)])
        self.assertTrue(all(item[2] == 24.0 for item in calls))
        self.assertIs(row["original_attempt"], initial)
        self.assertEqual([item["protocol_label"] for item in row["recovery_attempts"]], ["recovery2401", "recovery4001"])
        self.assertEqual(row["recovery_selected"], "recovery4001")

    def test_failed_fresh_state_triggers_fresh_recovery_without_warm_substitution(self) -> None:
        initial = self._valid_terminal_row()
        initial.update({
            "first_order_state_y_residual_max_abs_dimensionless": 2.0e-4,
            "stationarity_acceptance": False,
            "solver_status": 0,
        })
        calls: list[object | None] = []
        live = object()

        recovery_2401 = self._valid_terminal_row()
        recovery_2401.update({"solver_status": 0, "stationarity_acceptance": False})
        recovery_4001 = self._valid_terminal_row()
        recovery_4001.update({"solver_status": 0})

        def fake_raw(design, nucleus, *, level, factor=None, previous=None):
            del design, nucleus, factor
            calls.append(previous)
            if level["label"] == "fine24":
                return initial, live
            item = dict(recovery_2401 if level["label"] == "recovery2401" else recovery_4001)
            item["protocol_label"] = level["label"]
            return item, live

        design = static.FiniteDesign.from_continuous("W8.90", 0.25)
        with patch.object(bridge, "_raw_solve", side_effect=fake_raw):
            row, solved = bridge._solve(
                design, "Ca40", level=bridge.FINE_LEVEL, factor=1.0, previous=None
            )
        self.assertIs(solved, live)
        self.assertEqual(calls, [None, None, None])
        self.assertEqual([item["protocol_label"] for item in row["recovery_attempts"]], ["recovery2401", "recovery4001"])
        self.assertEqual(row.get("recovery_selected"), "recovery4001")
        self.assertEqual(row.get("recovery_independence"), "fresh_previous_none")
        self.assertTrue(bridge._fresh_attempt_valid(dict(row, independent_previous=None)))

    def test_fresh_controls_need_one_valid_independent_match_and_retain_failures(self) -> None:
        row = self._valid_terminal_row()
        row["seed_factor"] = 1.0
        row["independent_previous"] = None
        case = {
            "terminal_protocol_acceptance": True,
            "terminal_localized": True,
            "binding_per_A_MeV": 8.0,
            "selected_branch_id": "fine_branch_1",
            "fine_selected": row,
            "domain40_check": row,
            "fresh24_check": row,
            "fresh24_initial_attempt": row,
            "fresh24_attempts": [row, row, row],
            "fresh24_seed_factors": list(bridge.FRESH_CONTROL_FACTORS),
            "fresh24_profile_comparisons": [{"fresh_seed_factor": 1.0, "profile_consistency": self._valid_profile_comparison(), "passed": True}],
            "domain_profile_comparison": self._valid_profile_comparison(),
            "refinement_summary": {
                "fine_converged": True,
                "fine_stationarity": True,
                "fine_localized": True,
                "domain_converged": True,
                "domain_stationarity": True,
                "domain_localized": True,
                "fresh24_converged": True,
                "fresh24_stationarity": True,
                "fresh24_localized": True,
                "fresh24_independent_initial_pass": True,
                "domain_energy_pass": True,
                "fresh24_energy_pass": True,
                "domain_species_radius_pass": True,
                "fresh24_species_radius_pass": True,
                "fresh24_independent_rows_valid": True,
                "fresh24_valid_attempt_count": 3,
                "fresh24_failed_attempt_count": 0,
                "fresh24_branch_profiles_pass": True,
                "all_terminal_rows_converged_stationary_localized_bound": True,
            },
            "lineage": {"status": "CONSISTENT"},
            "branch_candidates": [{
                "branch_id": "fine_branch_1",
                "terminal_protocol_acceptance": True,
                "fine_selected": row,
                "domain40_check": row,
                "fresh24_check": row,
                "fresh24_initial_attempt": row,
                "fresh24_attempts": [row, row, row],
                "fresh24_profile_comparisons": [{"fresh_seed_factor": 1.0, "profile_consistency": self._valid_profile_comparison(), "passed": True}],
                "domain_profile_comparison": self._valid_profile_comparison(),
                "fine_converged": True,
                "domain_converged": True,
                "fresh24_independent_initial_pass": True,
                "fresh24_effective_pass": True,
                "fresh24_independent_rows_valid": True,
                "fresh24_valid_attempt_count": 3,
                "fresh24_failed_attempt_count": 0,
                "fresh24_branch_profiles_pass": True,
                "domain_energy_change_MeV": 0.0,
                "fresh24_energy_change_MeV": 0.0,
                "domain_species_radius_changes_fm": {"neutron": 0.0, "point_proton": 0.0},
                "fresh24_species_radius_changes_fm": {"neutron": 0.0, "point_proton": 0.0},
                "domain_energy_pass": True,
                "fresh24_energy_pass": True,
                "domain_species_radius_pass": True,
                "fresh24_species_radius_pass": True,
                "terminal_localized": True,
            }],
        }
        self.assertTrue(bridge._terminal_valid(case))
        failed = dict(row)
        failed["converged"] = False
        substituted = copy.deepcopy(case)
        substituted["fresh24_attempts"] = [failed, row, row]
        substituted["branch_candidates"] = [dict(case["branch_candidates"][0], fresh24_attempts=[failed, row, row], fresh24_valid_attempt_count=2, fresh24_failed_attempt_count=1)]
        # One valid independent row remains sufficient under R2 semantics.
        self.assertTrue(bridge._terminal_valid(substituted))
        no_independent = copy.deepcopy(case)
        no_independent["fresh24_attempts"] = [failed, failed, failed]
        no_independent["branch_candidates"] = [dict(case["branch_candidates"][0], fresh24_attempts=[failed, failed, failed], fresh24_valid_attempt_count=0, fresh24_failed_attempt_count=3, terminal_protocol_acceptance=False)]
        no_independent["terminal_protocol_acceptance"] = False
        self.assertFalse(bridge._terminal_valid(no_independent))

    def test_same_scale_grouping_keeps_profile_mismatch_as_distinct_branch(self) -> None:
        candidates = []
        for factor in bridge.SEED_FACTORS:
            candidates.append(({
                "seed_factor": factor,
                "chemical_potentials_MeV": {"neutron": 930.0, "proton": 930.0},
                "central_y": 0.5,
            }, object()))

        def fake_profile(left, right):
            return {"passed": left[0]["seed_factor"] == right[0]["seed_factor"]}

        with patch.object(bridge, "_same_branch", return_value=True), patch.object(bridge, "_profile_consistency", side_effect=fake_profile):
            groups, comparisons = bridge._group_same_scale_candidates(candidates)
        self.assertEqual([[item[0]["seed_factor"] for item in group] for group in groups], [[0.8], [1.0], [1.2]])
        self.assertEqual(len(comparisons), 3)

    def test_failed_prediction_does_not_publish_failed_energy_as_model_binding(self) -> None:
        case = {
            "terminal_protocol_acceptance": False,
            "terminal_localized": True,
            "binding_per_A_MeV": 7.5,
            "domain40_check": {"rms_point_proton_radius_fm": 3.0},
        }
        prediction = bridge._frozen_prediction("W8.90", 0.25, "Zr90", bridge._load_inputs(), case)
        self.assertIsNone(prediction["model_binding_per_A_MeV"])
        self.assertEqual(prediction["unaccepted_domain_binding_per_A_MeV"], 7.5)


if __name__ == "__main__":
    unittest.main()
