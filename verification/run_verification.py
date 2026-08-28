#!/usr/bin/env python3
"""Canonical scientific verification front door.

This command validates repository organization/provenance, regenerates the
canonical runtime ledger, and runs the semantic pytest suite.  The broad
``run_all_checks.py`` harness remains a separate process-only smoke command;
its exit status is intentionally not part of this scientific gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from registry import RegistryValidationError, validate_registry


def _run(label: str, command: list[str]) -> int:
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True)
    if result.returncode:
        print(f"[{label}] FAILED (exit {result.returncode})", flush=True)
    else:
        print(f"[{label}] PASS", flush=True)
    return result.returncode


def main() -> int:
    print("NVG SCIENTIFIC VERIFICATION FRONT DOOR", flush=True)
    print(f"Repository root: {ROOT}", flush=True)
    print("Scientific semantics: registry/provenance + canonical runtime ledger + semantic pytest", flush=True)
    print("Process smoke is separate: python verification/run_all_checks.py", flush=True)

    try:
        counts = validate_registry(ROOT)
    except RegistryValidationError as exc:
        print("[registry/provenance] FAILED", flush=True)
        for error in exc.errors:
            print(f"  ERROR: {error}", flush=True)
        return 1
    print(
        "[registry/provenance] PASS "
        + ", ".join(f"{key}={value}" for key, value in counts.items()),
        flush=True,
    )

    predictive_rc = _run(
        "scientific/predictive-ledger",
        [sys.executable, str(HERE / "nvg_predictive_research_ledger.py")],
    )
    if predictive_rc:
        return predictive_rc

    canonical_rc = _run(
        "scientific/canonical-ledger",
        [sys.executable, str(HERE / "run_nvg_suite.py")],
    )
    if canonical_rc:
        return canonical_rc

    semantic_rc = _run(
        "scientific/semantic-pytest",
        [sys.executable, "-m", "pytest", "-q", "verification"],
    )
    if semantic_rc:
        return semantic_rc

    print(
        "\n[process-smoke] NOT RUN by this front door; use "
        "python verification/run_all_checks.py for process-only diagnostics.",
        flush=True,
    )
    print("Scientific verification front door: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
