#!/usr/bin/env python3
"""One-off surgical registry patch: register the soft-isospin re-selection probe.

Follows the anchored-insertion pattern of scratch/patch_registry_transferability.py.
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
    lines = text.splitlines()
    body = lines[1:-1]
    return "\n".join("  " + line if line.strip() else line for line in body)


NEW_ENTRIES: dict[str, dict] = {
    "verification/nvg_soft_isospin_ns_reselection_probe.py": {
        "consumers": ["verification/test_nvg_soft_isospin_ns_reselection_probe.py"],
        "deprecation": "Conditional zero-evidence family-viability diagnostic; the isovector coupling is pinned to the identified j and the three screening constraints are selection inputs exactly as for the canonical chain, not independent evidence.",
        "description": "Soft-isospin NS-family re-selection: pre-registered 3150-point grid (k1, k2, Cs, n_trans, delta_eps) at C_rho = 2j/n0 with the saturation calibration intact, against the three canonical screening constraints; includes fail-closed reproduction of both retained transferability chains.",
        "evidence_weight": 0.0,
        "inputs": [
            "verification/nvg_eos_beta_css_softening.py",
            "verification/nvg_isovector_transferability_probe.py",
            "verification/nvg_ns_predictive_audit.py",
        ],
        "kind": "script",
        "outputs": [],
        "producer": None,
        "quarantine": None,
        "role": "validation",
        "status": "maintained",
        "tests": ["verification/test_nvg_soft_isospin_ns_reselection_probe.py"],
    },
    "verification/test_nvg_soft_isospin_ns_reselection_probe.py": {
        "consumers": [],
        "deprecation": "Focused tests for the zero-evidence soft-isospin re-selection diagnostic.",
        "description": "Pre-registered grid and verdict-band checks, canonical constraint mirror, baseline-patch restoration, retained-control reproductions and retained-artifact payload checks for the soft-isospin re-selection probe.",
        "evidence_weight": 0.0,
        "inputs": [
            "verification/nvg_soft_isospin_ns_reselection_probe.py",
            "verification/nvg_eos_beta_css_softening.py",
            "verification/nvg_isovector_transferability_probe.py",
            "verification/nvg_ns_predictive_audit.py",
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
