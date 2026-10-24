#!/usr/bin/env python3
"""One-off surgical registry patch: register the isovector transferability probe.

Follows the anchored-insertion pattern of scratch/patch_registry_nuclear.py:
renders the two new entries as 2-space-shifted JSON and inserts them before
the unique front_door anchor, then re-parses and validates fail-closed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "verification" / "registry.json"
ANCHOR = '    }\n  },\n  "front_door": {'


def render(key: str, value: dict) -> str:
    block = {key: value}
    text = json.dumps(block, indent=2, ensure_ascii=True)
    # Shift the whole block two spaces right and drop the outer braces.
    lines = text.splitlines()
    body = lines[1:-1]
    return "\n".join("  " + line if line.strip() else line for line in body)


NEW_ENTRIES: dict[str, dict] = {
    "verification/nvg_isovector_transferability_probe.py": {
        "consumers": ["verification/test_nvg_isovector_transferability_probe.py"],
        "deprecation": "Conditional zero-evidence diagnostic; the re-anchored rho branch and the NS-sector transfer are declared probes, not a recalibration of production files or a theory change.",
        "description": "Isovector transferability: secant re-anchoring of s* on Ca40 under the universal-j rho contact with all five nuclei re-solved, plus NS-sector consistency of the saturated-vector family (symmetry-energy audit, cross-branch contact ratio, transferred-contact TOV chain against pre-registered bands).",
        "evidence_weight": 0.0,
        "inputs": [
            "verification/nvg_finite_static_bridge.py",
            "verification/nvg_isovector_formfactor_probe.py",
            "verification/nvg_sn132_discriminating_set_probe.py",
            "verification/nvg_eos_beta_saturated_vector.py",
            "verification/nvg_eos_beta_css_softening.py",
            "verification/nvg_ns_predictive_audit.py",
            "verification/nvg_tidal_deformability.py",
            "verification/nvg_ns_predictive_audit_results.json",
        ],
        "kind": "script",
        "outputs": [],
        "producer": None,
        "quarantine": None,
        "role": "validation",
        "status": "maintained",
        "tests": ["verification/test_nvg_isovector_transferability_probe.py"],
    },
    "verification/test_nvg_isovector_transferability_probe.py": {
        "consumers": [],
        "deprecation": "Focused tests for the zero-evidence isovector transferability diagnostic.",
        "description": "Cited-anchor live verification, transfer algebra, pre-registered verdict-band logic, symmetry-energy sanity bands and retained-artifact payload checks for the transferability probe.",
        "evidence_weight": 0.0,
        "inputs": [
            "verification/nvg_isovector_transferability_probe.py",
            "verification/nvg_sn132_discriminating_set_probe.py",
        ],
        "kind": "test",
        "outputs": [],
        "producer": None,
        "quarantine": None,
        "role": "validation",
        "status": "maintained",
        "tests": [],
    },
}


def main() -> int:
    text = REGISTRY.read_text(encoding="utf-8")
    if text.count(ANCHOR) != 1:
        print(f"FAIL: front_door anchor count = {text.count(ANCHOR)}")
        return 1
    registry = json.loads(text)
    entries = registry["entries"]
    for key in NEW_ENTRIES:
        if key in entries:
            print(f"FAIL: {key} already registered")
            return 1
    # References may point at entries registered together in this patch.
    known = set(entries) | set(NEW_ENTRIES)
    for key, entry in NEW_ENTRIES.items():
        for field in ("inputs", "consumers", "tests"):
            for ref in entry.get(field) or []:
                if ref not in known:
                    print(f"FAIL: {key} references unregistered entry {ref}")
                    return 1
    blocks = ",\n".join(render(key, value) for key, value in NEW_ENTRIES.items())
    # Anchor = `    }` (last entry close) + `  },` (entries close) + front_door.
    # Rebuild as: last entry close WITH comma, the new entry blocks, then the
    # entries close and the front_door key.
    old = ANCHOR
    new = "    },\n" + blocks + "\n  },\n  \"front_door\": {"
    patched = text.replace(old, new, 1)
    reparsed = json.loads(patched)
    added = sum(1 for key in NEW_ENTRIES if key in reparsed["entries"])
    if added != len(NEW_ENTRIES):
        print("FAIL: re-parse did not contain the new entries")
        return 1
    REGISTRY.write_text(patched, encoding="utf-8")
    print(f"OK: registered {len(NEW_ENTRIES)} entries (total {len(reparsed['entries'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
