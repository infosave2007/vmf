"""Semantic contracts for the Phase 11 repository organization interface."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from registry import (  # noqa: E402
    RegistryValidationError,
    discover_maintained_scripts,
    validate_registry,
)


class RepositoryOrganizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry_path = HERE / "registry.json"
        self.payload = json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _validate_mutation(self, mutate) -> None:
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "registry.json"
            candidate.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RegistryValidationError):
                validate_registry(ROOT, candidate)

    def test_registry_covers_every_maintained_python_surface(self):
        discovered = discover_maintained_scripts(ROOT)
        registered = {
            path
            for path, entry in self.payload["entries"].items()
            if entry["kind"] in {"script", "test"}
        }
        self.assertTrue(discovered, "the maintained executable inventory is empty")
        self.assertEqual(discovered, registered)
        counts = validate_registry(ROOT)
        self.assertEqual(counts["scripts"], len(discovered))
        self.assertGreaterEqual(counts["artifacts"], 10)

    def test_anchor_chain_and_front_door_are_explicit(self):
        entries = self.payload["entries"]
        chain = [
            "verification/nvg_eos_beta_saturated_vector.py",
            "verification/nvg_eos_beta_css_softening.py",
            "verification/nvg_ns_parameter_scan.py",
            "verification/nvg_ns_canonical.py",
            "verification/nvg_tidal_deformability.py",
            "verification/nvg_joint_ns_inference.py",
            "verification/run_nvg_suite.py",
        ]
        self.assertEqual([entries[path]["role"] for path in chain], ["canonical"] * len(chain))
        for child, parent in zip(chain[1:], chain[:-1]):
            self.assertEqual(entries[child]["producer"], parent)
        front_door = self.payload["front_door"]
        self.assertEqual(front_door["path"], "verification/run_verification.py")
        self.assertEqual(front_door["canonical_producer"], "verification/run_nvg_suite.py")
        self.assertNotEqual(front_door["command"], front_door["process_smoke_command"])

    def test_registry_rejects_unregistered_or_invalid_surfaces(self):
        self._validate_mutation(
            lambda payload: payload["entries"]["verification/run_nvg_suite.py"].update(
                role="not-a-role"
            )
        )
        self._validate_mutation(
            lambda payload: payload["entries"].pop("verification/run_verification.py")
        )
        self._validate_mutation(
            lambda payload: payload["entries"]["verification/run_nvg_suite.py"].update(
                evidence_weight=0.0
            )
        )
        self._validate_mutation(
            lambda payload: payload["entries"]["verification/run_nvg_suite.py"]["outputs"].append(
                "verification/missing-artifact.json"
            )
        )
        self._validate_mutation(
            lambda payload: payload["entries"]["verification/run_nvg_suite.py"].update(
                producer="verification/nvg_fair_hades_link.py"
            )
        )

    def test_retired_and_forward_boundaries_carry_zero_weight(self):
        entries = self.payload["entries"]
        for path in (
            "verification/nvg_chime_frb_check.py",
            "verification/nvg_dna_chirality.py",
            "verification/nvg_moment_of_inertia_j0737.py",
        ):
            self.assertEqual(entries[path]["role"], "retired")
            self.assertIn(entries[path]["status"], {"retired", "quarantined"})
            self.assertEqual(entries[path]["evidence_weight"], 0.0)
            self.assertTrue(entries[path]["deprecation"] or entries[path]["quarantine"])
        self.assertEqual(entries["verification/nvg_gw_echo_prediction.py"]["role"], "forecast")
        self.assertEqual(entries["verification/nvg_gw_echo_prediction.py"]["evidence_weight"], 0.0)

    def test_artifact_and_data_manifests_are_registered(self):
        entries = self.payload["entries"]
        artifacts = json.loads((HERE / "artifact_manifest.json").read_text(encoding="utf-8"))["entries"]
        provenance = json.loads(
            (HERE / "data" / "provenance.json").read_text(encoding="utf-8")
        )["entries"]
        for path, artifact in artifacts.items():
            if (ROOT / path).is_file() and not artifact.get("status", "").startswith("ignored_not_tracked"):
                self.assertEqual(entries[path]["kind"], "artifact")
        for name in provenance:
            expected_kind = "data" if provenance[name].get("kind") == "input" else "artifact"
            self.assertEqual(entries[f"verification/data/{name}"]["kind"], expected_kind)

    def test_documentation_and_ignore_boundary_expose_stable_commands(self):
        for path in (ROOT / "README.md", ROOT / "README_RU.md", ROOT / "PUBLICATION_STATUS.md"):
            text = path.read_text(encoding="utf-8")
            self.assertIn("verification/registry.json", text)
            self.assertIn("verification/run_verification.py", text)
            self.assertIn("verification/run_all_checks.py", text)
            self.assertIn("verification/data/provenance.json", text)
            self.assertIn("verification/artifact_manifest.json", text)
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertRegex(ignore, r"(?m)^Lunacy/$")
        self.assertNotRegex(ignore, r"(?m)^verification/$")

    def test_validator_cli_is_cwd_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, str(HERE / "registry.py")],
                cwd=temporary,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Registry valid", result.stdout)


if __name__ == "__main__":
    unittest.main()
