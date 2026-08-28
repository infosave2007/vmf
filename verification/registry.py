"""Strict validation helpers for the maintained NVG computational registry.

The registry is deliberately data-first: it records the stable repository
surface without changing the scientific modules themselves.  This module is
safe to import from any caller working directory and is also a small CLI for
CI and local diagnostics.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).with_name("registry.json")
ALLOWED_ROLES = {
    "canonical",
    "validation",
    "forecast",
    "synthetic",
    "retired",
    "utility",
    "historical",
}
ALLOWED_STATUSES = {
    "maintained",
    "forward_only",
    "synthetic",
    "retired",
    "quarantined",
    "utility",
    "historical",
}
ALLOWED_KINDS = {"script", "test", "data", "artifact", "manifest"}
REFERENCE_FIELDS = ("consumers", "inputs", "outputs", "tests")
ENTRY_FIELDS = {
    "kind",
    "role",
    "status",
    "evidence_weight",
    "description",
    "producer",
    "consumers",
    "inputs",
    "outputs",
    "tests",
    "deprecation",
    "quarantine",
}


class RegistryValidationError(ValueError):
    """Raised when the registry or its provenance contracts are invalid."""

    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(str(error) for error in errors)
        super().__init__("registry validation failed: " + "; ".join(self.errors))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryValidationError((f"missing JSON manifest: {path}",)) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryValidationError((f"invalid JSON manifest {path}: {exc}",)) from exc
    if not isinstance(payload, dict):
        raise RegistryValidationError((f"JSON manifest must be an object: {path}",))
    return payload


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Load the machine-readable registry without changing the working cwd."""

    return _read_json(Path(path).resolve())


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or value.startswith("/"):
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in value


def _tracked_executables(root: Path) -> set[str]:
    """Return tracked Python/shell executables outside verification where available."""

    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "*.py", "*.sh"],
            cwd=root,
            check=True,
            capture_output=True,
            text=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    return {
        item
        for item in result.stdout.decode("utf-8", errors="replace").split("\0")
        if item
    }


def discover_maintained_scripts(root: Path = ROOT) -> set[str]:
    """Discover the physical script/test surface covered by the registry.

    Every physical Python/shell file under ``verification/`` is maintained by
    default, including untracked phase tests present in a working checkout.
    Outside that directory only tracked project scripts are included; private
    ``.code/``, ``Lunacy/`` and ``vacuum_neural_networks/`` trees are excluded.
    """

    discovered = {
        path.relative_to(root).as_posix()
        for suffix in ("*.py", "*.sh")
        for path in (root / "verification").rglob(suffix)
        if "__pycache__" not in path.parts
    }
    for relative in _tracked_executables(root):
        if relative.startswith(("verification/", ".code/", "Lunacy/", "vacuum_neural_networks/")):
            continue
        path = root / relative
        if path.is_file():
            discovered.add(relative)
    return discovered


def _manifest_status_allows_missing(status: Any) -> bool:
    return isinstance(status, str) and status.startswith("ignored_not_tracked")


def _validate_provenance(
    root: Path,
    entries: dict[str, Any],
    errors: list[str],
) -> tuple[int, int]:
    """Validate P11-S2 data/artifact manifests and their script references."""

    manifest_path = root / "verification" / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        errors.append("verification/artifact_manifest.json must use schema_version=1")
    manifest_entries = manifest.get("entries")
    if not isinstance(manifest_entries, dict) or not manifest_entries:
        errors.append("artifact manifest entries must be a non-empty object")
        manifest_entries = {}
    artifact_count = 0
    for relative, item in manifest_entries.items():
        artifact_count += 1
        if not _safe_relative(relative) or not isinstance(item, dict):
            errors.append(f"invalid artifact manifest entry: {relative!r}")
            continue
        if item.get("kind") != "generated_artifact":
            errors.append(f"artifact {relative} must have kind generated_artifact")
        for required in ("producer", "command", "status", "canonical_path"):
            if required not in item:
                errors.append(f"artifact {relative} missing field {required}")
        canonical_path = item.get("canonical_path")
        if canonical_path != relative:
            errors.append(f"artifact {relative} canonical_path mismatch: {canonical_path!r}")
        status = item.get("status")
        path = root / relative
        if not path.is_file() and not _manifest_status_allows_missing(status):
            errors.append(f"artifact manifest path missing: {relative}")
        # Ignored generated reports are intentionally outside the public
        # registry; artifact_manifest remains their provenance authority.
        if path.is_file() and relative not in entries and not _manifest_status_allows_missing(status):
            errors.append(f"artifact is not registered: {relative}")
        producer = item.get("producer")
        if producer is not None:
            if not _safe_relative(producer) or producer not in entries:
                errors.append(f"artifact {relative} references missing producer {producer!r}")
            elif entries[producer].get("kind") not in {"script", "test"}:
                errors.append(f"artifact {relative} producer is not executable: {producer}")
            if relative in entries and entries[relative].get("producer") != producer:
                errors.append(f"artifact {relative} producer disagrees with registry")

    provenance_path = root / "verification" / "data" / "provenance.json"
    provenance = _read_json(provenance_path)
    if provenance.get("schema_version") != 2:
        errors.append("verification/data/provenance.json must use schema_version=2")
    provenance_entries = provenance.get("entries")
    if not isinstance(provenance_entries, dict) or not provenance_entries:
        errors.append("data provenance entries must be a non-empty object")
        provenance_entries = {}
    input_count = 0
    for name, item in provenance_entries.items():
        input_count += 1
        relative = f"verification/data/{name}"
        if not _safe_relative(name) or not isinstance(item, dict):
            errors.append(f"invalid data provenance entry: {name!r}")
            continue
        for required in ("kind", "status", "producer", "config"):
            if required not in item:
                errors.append(f"provenance {relative} missing field {required}")
        if not (root / relative).is_file():
            errors.append(f"provenance path missing: {relative}")
        if relative not in entries:
            errors.append(f"provenance path is not registered: {relative}")
        producer = item.get("producer")
        if producer is not None:
            if not _safe_relative(producer) or producer not in entries:
                errors.append(f"provenance {relative} references missing producer {producer!r}")
            elif entries[producer].get("kind") not in {"script", "test"}:
                errors.append(f"provenance {relative} producer is not executable: {producer}")
            if relative in entries and entries[relative].get("producer") != producer:
                errors.append(f"provenance {relative} producer disagrees with registry")
        if item.get("kind") not in {"input", "generated_artifact"}:
            errors.append(f"provenance {relative} has invalid kind {item.get('kind')!r}")
    return artifact_count, input_count


def validate_registry(
    root: Path = ROOT,
    registry_path: Path | None = None,
) -> dict[str, int]:
    """Validate registry coverage, schema, references and provenance.

    A successful call returns compact counts.  Any contract violation raises
    :class:`RegistryValidationError`, making failures deterministic in CI and
    easy to assert in semantic tests.
    """

    root = Path(root).resolve()
    path = Path(registry_path or root / "verification" / "registry.json").resolve()
    payload = load_registry(path)
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("registry must use schema_version=1")
    roles = payload.get("roles")
    statuses = payload.get("statuses")
    kinds = payload.get("kinds")
    if set(roles or ()) != ALLOWED_ROLES:
        errors.append("registry roles do not match the maintained role vocabulary")
    if set(statuses or ()) != ALLOWED_STATUSES:
        errors.append("registry statuses do not match the maintained status vocabulary")
    if set(kinds or ()) != ALLOWED_KINDS:
        errors.append("registry kinds do not match the maintained kind vocabulary")
    entries = payload.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise RegistryValidationError(errors + ["registry entries must be a non-empty object"])

    for relative, entry in entries.items():
        if not _safe_relative(relative):
            errors.append(f"registry path is not safe/relative: {relative!r}")
            continue
        if not isinstance(entry, dict):
            errors.append(f"registry entry is not an object: {relative}")
            continue
        missing_fields = ENTRY_FIELDS.difference(entry)
        if missing_fields:
            errors.append(f"{relative} missing fields: {sorted(missing_fields)}")
        unknown_fields = set(entry).difference(ENTRY_FIELDS)
        if unknown_fields:
            errors.append(f"{relative} has unknown fields: {sorted(unknown_fields)}")
        kind = entry.get("kind")
        role = entry.get("role")
        status = entry.get("status")
        if kind not in ALLOWED_KINDS:
            errors.append(f"{relative} has invalid kind {kind!r}")
        if role not in ALLOWED_ROLES:
            errors.append(f"{relative} has invalid role {role!r}")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{relative} has invalid status {status!r}")
        if kind in {"script", "test"} and not relative.endswith((".py", ".sh")):
            errors.append(f"executable registry entry is not Python/shell: {relative}")
        actual = root / relative
        if not actual.is_file():
            errors.append(f"registry path missing: {relative}")
        weight = entry.get("evidence_weight")
        valid_weight = (
            isinstance(weight, (int, float))
            and not isinstance(weight, bool)
            and math.isfinite(float(weight))
        )
        numeric_weight = float(weight) if valid_weight else None
        if not valid_weight:
            errors.append(f"{relative} evidence_weight must be finite numeric")
        elif not 0.0 <= numeric_weight <= 1.0:
            errors.append(f"{relative} evidence_weight must be between 0 and 1")
        if not isinstance(entry.get("description"), str) or not entry["description"].strip():
            errors.append(f"{relative} description must be non-empty")
        for field in REFERENCE_FIELDS:
            refs = entry.get(field)
            if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
                errors.append(f"{relative} {field} must be a list of strings")
                continue
            if len(refs) != len(set(refs)):
                errors.append(f"{relative} {field} contains duplicate references")
            for ref in refs:
                if field == "inputs" and ref.startswith(("external:", "declared:")):
                    continue
                if ref.startswith(("external:", "declared:")):
                    errors.append(f"{relative} {field} must reference a registered repository path: {ref}")
                    continue
                if not _safe_relative(ref):
                    errors.append(f"{relative} {field} has unsafe reference {ref!r}")
                elif ref not in entries:
                    errors.append(f"{relative} {field} references unregistered path {ref}")
                elif field == "tests" and entries[ref].get("kind") != "test":
                    errors.append(f"{relative} tests reference non-test {ref}")
                elif field == "outputs" and entries[ref].get("kind") not in {"artifact", "manifest"}:
                    errors.append(f"{relative} outputs reference non-artifact {ref}")
                elif field == "consumers" and entries[ref].get("kind") not in {"script", "test"}:
                    errors.append(f"{relative} consumers reference non-executable {ref}")
        producer = entry.get("producer")
        if producer is not None:
            if not _safe_relative(producer) or producer not in entries:
                errors.append(f"{relative} references missing producer {producer!r}")
            elif entries[producer].get("kind") not in {"script", "test"}:
                errors.append(f"{relative} producer is not executable: {producer}")
            elif producer == relative:
                errors.append(f"{relative} cannot produce itself")
            elif entry.get("kind") in {"script", "test"} and relative not in entries[producer].get("consumers", []):
                errors.append(f"{relative} producer/consumer references disagree with {producer}")
        deprecation = entry.get("deprecation")
        quarantine = entry.get("quarantine")
        if deprecation is not None and not isinstance(deprecation, str):
            errors.append(f"{relative} deprecation must be a string or null")
        if quarantine is not None and not isinstance(quarantine, str):
            errors.append(f"{relative} quarantine must be a string or null")

        # Role/status/evidence combinations are part of the contract.  In
        # particular, a canonical producer cannot silently become zero-weight,
        # retired or quarantined.
        if role == "canonical":
            if status != "maintained":
                errors.append(f"canonical {relative} must have status maintained")
            if numeric_weight is None or numeric_weight <= 0.0:
                errors.append(f"canonical {relative} has zero evidence weight")
            if deprecation is not None or quarantine is not None:
                errors.append(f"canonical {relative} has deprecation/quarantine metadata")
            if kind != "script" and producer is None:
                errors.append(f"canonical non-script {relative} lacks a producing script")
            if producer in entries and entries[producer].get("role") != "canonical":
                errors.append(f"canonical {relative} references non-canonical producer {producer}")
        elif role == "forecast" and (status != "forward_only" or numeric_weight != 0.0):
            errors.append(f"forecast {relative} must be forward_only with zero weight")
        elif role == "synthetic" and (status != "synthetic" or numeric_weight != 0.0):
            errors.append(f"synthetic {relative} must be synthetic with zero weight")
        elif role == "retired":
            if status not in {"retired", "quarantined"} or numeric_weight != 0.0:
                errors.append(f"retired {relative} has contradictory status/weight")
            if not deprecation and not quarantine:
                errors.append(f"retired {relative} lacks deprecation/quarantine reason")
        elif role == "historical" and (status != "historical" or numeric_weight != 0.0):
            errors.append(f"historical {relative} must be historical with zero weight")
        elif role == "utility" and (status not in {"utility", "maintained"} or numeric_weight != 0.0):
            errors.append(f"utility {relative} must be zero-weight utility/manifest")

    discovered = discover_maintained_scripts(root)
    registered = {
        relative
        for relative, entry in entries.items()
        if isinstance(entry, dict) and entry.get("kind") in {"script", "test"}
    }
    for missing in sorted(discovered - registered):
        errors.append(f"maintained script/test is unregistered: {missing}")
    for stale in sorted(registered - discovered):
        errors.append(f"registered script/test path is missing: {stale}")

    front_door = payload.get("front_door")
    if not isinstance(front_door, dict):
        errors.append("front_door metadata is required")
    else:
        front_path = front_door.get("path")
        if front_path not in entries or entries.get(front_path, {}).get("kind") != "script":
            errors.append("front_door.path must reference a registered script")
        canonical = front_door.get("canonical_producer")
        if canonical not in entries or entries.get(canonical, {}).get("role") != "canonical":
            errors.append("front_door.canonical_producer must reference a canonical script")
        if front_door.get("process_smoke_command") == front_door.get("command"):
            errors.append("process smoke command must remain distinct from the front door")

    for metadata_field in ("artifact_manifest", "data_provenance"):
        metadata_path = payload.get(metadata_field)
        if not _safe_relative(metadata_path):
            errors.append(f"{metadata_field} must be a safe repository-relative path")
        elif not (root / metadata_path).is_file():
            errors.append(f"{metadata_field} path missing: {metadata_path}")

    # P4-S1's machine ledger is an explicit front-door integration artifact.
    # Keep its metadata fail-closed: the path and producer must be registered,
    # the producer must actually emit the artifact, and the ledger itself is
    # always zero independent-evidence weight.  This check intentionally does
    # not authenticate scientific values; nvg_predictive_research_ledger.py
    # performs that live producer/artifact authentication.
    predictive_ledger = payload.get("predictive_ledger")
    if not isinstance(predictive_ledger, dict):
        errors.append("predictive_ledger metadata is required")
    else:
        ledger_path = predictive_ledger.get("path")
        ledger_producer = predictive_ledger.get("producer")
        if not _safe_relative(ledger_path):
            errors.append("predictive_ledger.path must be a safe repository-relative path")
        elif not (root / ledger_path).is_file():
            errors.append(f"predictive_ledger.path missing: {ledger_path}")
        if not _safe_relative(ledger_producer) or ledger_producer not in entries:
            errors.append("predictive_ledger.producer must reference a registered path")
        elif entries[ledger_producer].get("kind") != "script":
            errors.append("predictive_ledger.producer must reference a script")
        if _safe_relative(ledger_path) and ledger_path in entries:
            ledger_entry = entries[ledger_path]
            if ledger_entry.get("kind") != "artifact":
                errors.append("predictive_ledger.path must reference an artifact entry")
            if ledger_entry.get("producer") != ledger_producer:
                errors.append("predictive_ledger producer disagrees with artifact entry")
            if ledger_entry.get("evidence_weight") != 0.0:
                errors.append("predictive_ledger artifact must have zero evidence weight")
        if predictive_ledger.get("independent_evidence_weight") != 0.0:
            errors.append("predictive_ledger independent evidence weight must be zero")

    # An output reference is a producer claim, not merely a link.  Keep it
    # consistent with the artifact's own producer and do not let canonical
    # entries point at retired/historical artifacts.
    for relative, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        for output in entry.get("outputs", ()) if isinstance(entry.get("outputs"), list) else ():
            target = entries.get(output)
            if not isinstance(target, dict):
                continue
            target_producer = target.get("producer")
            if target_producer is not None and target_producer != relative:
                errors.append(
                    f"{relative} claims output {output}, whose producer is {target_producer}"
                )
            if entry.get("role") == "canonical" and target.get("role") in {"retired", "historical"}:
                errors.append(f"canonical {relative} points at non-current artifact {output}")

    artifact_count, input_count = _validate_provenance(root, entries, errors)
    if errors:
        raise RegistryValidationError(errors)
    return {
        "entries": len(entries),
        "scripts": len(registered),
        "artifacts": artifact_count,
        "inputs": input_count,
    }


def main() -> int:
    try:
        counts = validate_registry()
    except RegistryValidationError as exc:
        for error in exc.errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Registry valid: "
        + ", ".join(f"{key}={value}" for key, value in counts.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
