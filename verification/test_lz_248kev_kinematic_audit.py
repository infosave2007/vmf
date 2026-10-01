"""Focused regressions for the LZ 248-keV kinematic necessary-condition screen."""
from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal, getcontext
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nvg_lz_248kev_kinematic_audit as audit  # noqa: E402
import nvg_adm_bl_cogenesis as cogenesis  # noqa: E402


class LZ248KeVKinematicAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = HERE / "nvg_lz_248kev_kinematic_results.json"
        cls.result = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.data = audit._load_data()

    def test_serialized_result_rebuilds_and_declares_no_fit(self) -> None:
        self.assertTrue(audit.validate_result(self.result))
        self.assertEqual(self.result["status"], audit.STATUS_PASS)
        self.assertTrue(self.result["all_controls_pass"])
        self.assertFalse(self.result["fit_used"])
        self.assertEqual(self.result["independent_evidence_weight"], 0.0)

    def test_sourced_shm_envelope_is_derived_not_a_magic_speed(self) -> None:
        cap = audit.bound_shm_speed_cap_km_s(self.data["bound_standard_halo_model_screen"])
        self.assertAlmostEqual(float(cap), 824.354664503010, places=9)
        self.assertAlmostEqual(
            float(self.result["declared_screen"]["bound_shm_speed_cap_km_s"]), float(cap), places=12
        )
        source = self.result["source_provenance"]["recommended_standard_halo_model"]
        self.assertTrue(source["source_url"].startswith("https://arxiv.org/"))

    def test_exact_elastic_forward_and_inverse_kinematics_close(self) -> None:
        screen = self.result["declared_screen"]
        mass = Decimal(screen["minimum_exact_elastic_particle_mass_at_speed_cap_GeV"])
        target = Decimal(screen["target_mass_GeV"])
        speed = Decimal(screen["bound_shm_speed_cap_km_s"])
        recoil = Decimal(screen["event_recoil_keV"])
        self.assertAlmostEqual(float(audit.exact_elastic_recoil_maximum_keV(mass, target, speed)), float(recoil), places=18)
        self.assertAlmostEqual(float(audit.exact_elastic_minimum_speed_km_s(mass, target, recoil)), float(speed), places=18)

    def test_full_declared_dark_neutron_corridor_fails_exact_screen(self) -> None:
        rows = self.result["current_nvg_candidate_screens"]
        neutron_rows = [row for row in rows if row["candidate_id"].startswith("nvg_dark_neutron_")]
        self.assertEqual(len(neutron_rows), 2)
        for row in neutron_rows:
            self.assertFalse(row["elastic_kinematically_accessible_under_screen"])
            self.assertGreater(float(row["elastic_speed_cap_ratio"]), 47.0)
            delta = row["formal_delta_interval_if_declared_mass_is_treated_as_incoming_state_keV"]
            self.assertLess(float(delta["maximum"]), 0.0)
            release = row["hypothetical_heavier_excitation_to_declared_ground_state"]
            self.assertGreater(float(release["minimum_required_release_MeV"]), 31.0)

    def test_dark_neutron_corridor_has_an_explicit_legacy_input_passport(self) -> None:
        passport = self.result["conditional_input_passports"]["dark_neutron_cogenesis"]
        expected_hash = hashlib.sha256(cogenesis.INPUT_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            passport["input_artifact"],
            "verification/data/legacy_dark_neutron_cogenesis_inputs.json",
        )
        self.assertEqual(passport["input_sha256"], expected_hash)
        self.assertEqual(passport["input_status"], cogenesis.INPUT_STATUS)
        self.assertEqual(self.result["source_sha256"]["dark_neutron_historical_input"], expected_hash)
        self.assertTrue(
            self.result["controls"][
                "dark_neutron_corridor_has_explicit_legacy_conditional_input_passport"
            ]["pass"]
        )

    def test_dark_neutron_passport_mismatch_fails_closed(self) -> None:
        expected_hash = hashlib.sha256(cogenesis.INPUT_PATH.read_bytes()).hexdigest()
        with patch.object(
            audit.dark_neutron,
            "historical_input_passport",
            return_value={
                "input_artifact": "verification/data/legacy_dark_neutron_cogenesis_inputs.json",
                "input_sha256": expected_hash,
                "input_status": "PROMOTED_WITHOUT_MECHANISM",
            },
        ):
            with self.assertRaises(ValueError):
                audit._build_result()

    def test_exact_heavier_excitation_window_reconstructs_endpoint_angles(self) -> None:
        rows = self.result["current_nvg_candidate_screens"]
        w_row = next(row for row in rows if row["candidate_id"] == "nvg_W_like_relic_excitation_proxy")
        excitation = w_row["hypothetical_heavier_excitation_to_declared_ground_state"]
        lower = Decimal(excitation["minimum_required_release_MeV"]) * Decimal("1000")
        upper = Decimal(excitation["maximum_required_release_MeV"]) * Decimal("1000")
        self.assertLess(lower, upper)
        self.assertGreater(lower, Decimal(0))
        self.assertLess(Decimal(w_row["initial_kinetic_energy_at_speed_cap_keV"]), Decimal("5"))
        self.assertGreater(lower / Decimal("1000"), Decimal("24"))

        ground_mass = Decimal(w_row["mass_GeV"])
        screen = self.result["declared_screen"]
        target = Decimal(screen["target_mass_GeV"])
        recoil = Decimal(screen["event_recoil_keV"]) / Decimal("1e6")
        beta = Decimal(screen["bound_shm_speed_cap_km_s"]) / audit.C_KM_S
        gamma = Decimal(1) / (Decimal(1) - beta * beta).sqrt()
        q = (recoil * (recoil + Decimal(2) * target)).sqrt()

        def reconstructed_cosine(release_keV: Decimal) -> Decimal:
            incoming_mass = ground_mass + release_keV / Decimal("1e6")
            momentum = gamma * incoming_mass * beta
            energy = gamma * incoming_mass
            return (ground_mass * ground_mass - (energy - recoil) ** 2 + momentum * momentum + q * q) / (Decimal(2) * momentum * q)

        self.assertAlmostEqual(float(reconstructed_cosine(lower)), 1.0, places=15)
        self.assertAlmostEqual(float(reconstructed_cosine(upper)), -1.0, places=15)
        midpoint = (lower + upper) / Decimal(2)
        delta_min, delta_max = audit.exact_delta_interval_at_recoil_keV(
            ground_mass + midpoint / Decimal("1e6"),
            target,
            Decimal(screen["event_recoil_keV"]),
            Decimal(screen["bound_shm_speed_cap_km_s"]),
        )
        self.assertLessEqual(delta_min, -midpoint)
        self.assertLessEqual(-midpoint, delta_max)

    def test_source_complete_modes_are_screened_without_claiming_a_halo_population(self) -> None:
        rows = {row["candidate_id"]: row for row in self.result["current_nvg_candidate_screens"]}
        vector = rows["source_complete_massive_vector_mode"]
        scalar = rows["source_complete_scalar_radial_mode"]
        self.assertAlmostEqual(float(vector["mass_GeV"]), 0.7826, places=12)
        self.assertGreater(float(scalar["mass_GeV"]), 1.24)
        for row in (vector, scalar):
            self.assertFalse(row["elastic_kinematically_accessible_under_screen"])
            self.assertIn("does not establish", row["caveat"])
            self.assertGreater(
                float(row["hypothetical_heavier_excitation_to_declared_ground_state"]["minimum_required_release_MeV"]),
                20.0,
            )

    def test_exact_elastic_accessibility_matches_zero_in_fixed_delta_interval(self) -> None:
        screen = self.result["declared_screen"]
        target = Decimal(screen["target_mass_GeV"])
        recoil = Decimal(screen["event_recoil_keV"])
        speed = Decimal(screen["bound_shm_speed_cap_km_s"])
        for mass in (Decimal("0.7826"), Decimal("1.2448092624976728"), Decimal("1000")):
            delta_min, delta_max = audit.exact_delta_interval_at_recoil_keV(mass, target, recoil, speed)
            endpoint = audit.exact_elastic_recoil_maximum_keV(mass, target, speed)
            self.assertEqual(delta_min <= 0 <= delta_max, endpoint >= recoil)

    def test_heavier_excitation_helper_is_independent_of_callers_decimal_precision(self) -> None:
        screen = self.result["declared_screen"]
        previous_precision = getcontext().prec
        try:
            getcontext().prec = 28
            window = audit.exact_heavier_excitation_release_interval_for_declared_ground_state_keV(
                Decimal("0.9378996"),
                Decimal(screen["target_mass_GeV"]),
                Decimal(screen["event_recoil_keV"]),
                Decimal(screen["bound_shm_speed_cap_km_s"]),
            )
        finally:
            getcontext().prec = previous_precision
        self.assertGreater(window[0], Decimal("31000"))
        self.assertLess(window[1], Decimal("33000"))

    def test_w_proxy_and_external_comparator_are_not_conflated(self) -> None:
        rows = self.result["current_nvg_candidate_screens"]
        w_proxy = next(row for row in rows if row["candidate_id"] == "nvg_W_like_relic_excitation_proxy")
        self.assertFalse(w_proxy["elastic_kinematically_accessible_under_screen"])
        external = self.result["external_comparator_not_adopted_by_nvg"]
        self.assertTrue(external["kinematically_accessible_under_same_screen"])
        self.assertGreater(float(external["speed_cap_ratio"]), 0.9)
        self.assertLess(float(external["speed_cap_ratio"]), 1.0)
        interval = external["exact_delta_interval_for_fixed_recoil_at_speed_cap_keV"]
        self.assertGreater(float(interval["maximum"]), 350.0)

    def test_invalid_shm_input_fails_closed(self) -> None:
        invalid = copy.deepcopy(self.data)
        invalid["bound_standard_halo_model_screen"]["earth_orbital_velocity_basis"]["cosine_coefficients"] = ["1", "0"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(ValueError):
                audit._load_data(path)

    def test_mutation_or_invalid_input_fails_closed(self) -> None:
        changed = copy.deepcopy(self.result)
        changed["declared_screen"]["minimum_exact_elastic_particle_mass_at_speed_cap_GeV"] = "1"
        self.assertFalse(audit.validate_result(changed))
        with self.assertRaises(ValueError):
            audit.exact_elastic_mass_threshold_GeV(Decimal("1"), Decimal("10000000"), Decimal("1"))
        with self.assertRaises(ValueError):
            audit.exact_delta_interval_at_recoil_keV(Decimal("1"), Decimal("1"), Decimal("1"), Decimal("299792.458"))


if __name__ == "__main__":
    unittest.main()
