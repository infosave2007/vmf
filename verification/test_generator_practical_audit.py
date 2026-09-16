"""Semantic and invariant checks for the conventional generator audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# The verification directory is a script collection rather than a package;
# make direct discovery from the repository root import the sibling module.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nvg_generator_practical_audit import (
    RESULT_PATH,
    REPORT_PATH,
    assert_no_speculative_gain,
    calculate,
    conventional_teg,
    module_requirement,
    russian_report,
    write_outputs,
)


class GeneratorPracticalAuditTests(unittest.TestCase):
    def test_legacy_gain_is_explicit_and_not_accepted_as_source(self):
        legacy = calculate()["legacy_peltier"]
        self.assertAlmostEqual(float(legacy["power_without_gain_w"]), 8.973310651640998e-4, places=12)
        self.assertAlmostEqual(float(legacy["power_with_legacy_gain_w"]), 403.79897932384495, places=9)
        self.assertLess(float(legacy["net_without_gain_w"]), 0.0)
        self.assertEqual(legacy["status"], "ARITHMETIC_ONLY_UNSUPPORTED_SOURCE")

    def test_legacy_diode_is_separate_from_no_gain_loss_floor(self):
        legacy = calculate()["legacy_peltier"]
        floor = float(legacy["no_gain_loss_floor_w"])
        no_gain_diode = float(legacy["diode_loss_without_gain_w"])
        gain_diode = float(legacy["legacy_output_dependent_diode_loss_w"])
        self.assertGreater(floor, 0.0)
        self.assertGreater(gain_diode, no_gain_diode)
        self.assertAlmostEqual(
            float(legacy["modeled_losses_without_gain_w"]), floor + no_gain_diode, places=12
        )
        self.assertAlmostEqual(
            float(legacy["modeled_losses_with_legacy_gain_w"]), floor + gain_diode, places=12
        )
        self.assertAlmostEqual(
            float(legacy["net_without_gain_w"]),
            float(legacy["power_without_gain_w"]) - float(legacy["modeled_losses_without_gain_w"]),
            places=12,
        )
        self.assertAlmostEqual(
            float(legacy["net_without_gain_floor_w"]),
            float(legacy["power_without_gain_w"]) - floor,
            places=12,
        )

    def test_teg_efficiency_and_power_balance(self):
        case = conventional_teg()
        self.assertLess(float(case["device_efficiency"]), float(case["carnot_efficiency"]))
        self.assertAlmostEqual(float(case["gross_electrical_w"]), 4.934202283693, places=9)
        self.assertLessEqual(
            float(case["gross_electrical_w"]),
            float(case["heat_per_module_w"]) * float(case["carnot_efficiency"]),
        )
        self.assertLessEqual(float(case["net_electrical_w"]), float(case["converter_output_w"]))
        self.assertLessEqual(float(case["converter_output_w"]), float(case["gross_electrical_w"]))
        self.assertIn("waste heat", str(case["energy_source"]))
        self.assertEqual(case["auxiliary_scope"], "per_module")
        self.assertFalse(case["manufacturer_rating_used"])
        self.assertEqual(case["independent_evidence_weight"], 0.0)

    def test_module_requirement_reports_heat_budget_infeasible(self):
        case = conventional_teg()
        feasible = module_requirement(10.0, case)
        self.assertTrue(feasible["feasible"])
        self.assertEqual(feasible["modules"], 3)
        self.assertEqual(feasible["thermal_input_w"], 600.0)
        self.assertEqual(feasible["module_topology"], "parallel; thermal input sums across modules")
        self.assertEqual(feasible["auxiliary_total_w"], 0.0)
        limited = module_requirement(10.0, case, available_heat_w=500.0)
        self.assertFalse(limited["feasible"])
        self.assertEqual(limited["status"], "INFEASIBLE_HEAT_BUDGET")
        dead = dict(case, net_electrical_w=0.0)
        self.assertFalse(module_requirement(1.0, dead)["feasible"])

    def test_system_auxiliary_is_shared_and_thermal_scales_in_parallel(self):
        case = conventional_teg(auxiliary_w=0.25, heat_per_module_w=123.0)
        sized = module_requirement(10.0, case, system_auxiliary_w=1.0)
        self.assertEqual(sized["modules"], 5)
        self.assertEqual(sized["thermal_input_w"], 615.0)
        self.assertAlmostEqual(float(sized["auxiliary_total_w"]), 2.25, places=12)
        self.assertAlmostEqual(
            float(sized["predicted_net_w"]), 5.0 * float(case["net_electrical_w"]) - 1.0, places=12
        )
        with self.assertRaises(ValueError):
            module_requirement(1.0, case, system_auxiliary_w=-1.0)

    def test_sensitivity_is_cartesian_and_not_target_fit(self):
        sensitivity = calculate()["sensitivity"]
        self.assertEqual(len(sensitivity["rows"]), 27)
        envelope = sensitivity["gross_power_envelope_w"]
        self.assertAlmostEqual(float(envelope["min"]), 0.7710888958791228, places=12)
        self.assertAlmostEqual(float(envelope["max"]), 8.990897566362053, places=12)
        self.assertIn("no row was fitted", sensitivity["selection_rule"])
        self.assertTrue(sensitivity["target_independent"])
        self.assertTrue(sensitivity["requested_target_not_used"])
        self.assertEqual(
            calculate({"example_requested_net_w": 1000.0})["sensitivity"], sensitivity
        )

    def test_speculative_family_evidence_weights_are_zero(self):
        families = calculate()["families"]
        self.assertTrue(families)
        self.assertTrue(all(row["independent_evidence_weight"] == 0.0 for row in families))
        names = {row["family"] for row in families}
        self.assertTrue({"antigravity_theta", "high_speed_magnetic", "theta_haloscope"} <= names)
        for row in families:
            self.assertIn("dimension_check", row)
            self.assertIn("empirical_status", row)
            if row["family"] in {"theta_haloscope", "antigravity_theta", "high_speed_magnetic"}:
                self.assertIn("predicted_signal_vs_noise", row)

    def test_report_does_not_start_with_status_banner(self):
        report = russian_report(calculate())
        lines = report.splitlines()
        self.assertEqual(lines[0], "# Практический аудит генератора NVG")
        self.assertNotIn("Статус:", "\n".join(lines[:4]))
        self.assertIn("PASS_CONVENTIONAL_ONLY", report)
        self.assertIn("не являются паспортной характеристикой", report)

    def test_forbidden_gain_keys_fail_closed(self):
        with self.assertRaises(ValueError):
            assert_no_speculative_gain({"nvg_gain": 1.0})
        with self.assertRaises(ValueError):
            assert_no_speculative_gain({"nested": [{"casimir_gain": 2.0}]})

    def test_deterministic_write_and_cli_no_write(self):
        data = calculate()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "result.json"
            report = root / "report.md"
            write_outputs(data, result, report)
            first_result = result.read_bytes()
            first_report = report.read_bytes()
            write_outputs(data, result, report)
            self.assertEqual(first_result, result.read_bytes())
            self.assertEqual(first_report, report.read_bytes())
            payload = json.loads(first_result)
            self.assertEqual(payload["schema"], "nvg-generator-practical-audit/v1")

        tracked = [path for path in (RESULT_PATH, REPORT_PATH) if path.exists()]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("nvg_generator_practical_audit.py"))],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn('"schema": "nvg-generator-practical-audit/v1"', proc.stdout)
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
