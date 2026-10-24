"""Independent fresh-process acceptance through the public stellar API.

No evaluator injection, monkeypatch, saved numerical fallback or artifact
write is performed. Elapsed time belongs only to this external run log.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verification"))
import source_complete_stellar_prediction as stellar

started = time.perf_counter()
json_before = stellar.RESULT_PATH.read_bytes()
report_before = stellar.REPORT_PATH.read_bytes()
result = json.loads(json_before)
source_before = stellar.file_digest(Path(stellar.__file__))
test_before = stellar.file_digest(stellar.TEST_PATH)
print(json.dumps({"stage": "start_public_validation", "source_sha256": source_before,
                  "test_sha256": test_before, "python": sys.version}), flush=True)

# This public entry point must regenerate from the live source and compare
# canonical scientific bytes; a schema-only check is not sufficient.
accepted, errors = stellar.validate_contract(result)
assert accepted, errors
assert result["convergence"]["pass"] is True
assert result["status"] == "STABLE_BRANCH_TERMINATED_AT_FIRST_MASS_MAXIMUM"
assert stellar.RESULT_PATH.read_bytes() == json_before
assert stellar.REPORT_PATH.read_bytes() == report_before
assert stellar.render_report(result).encode("utf-8") == report_before
assert stellar.file_digest(Path(stellar.__file__)) == source_before
assert stellar.file_digest(stellar.TEST_PATH) == test_before
print(json.dumps({
    "stage": "public_validation_completed", "accepted": accepted, "errors": errors,
    "wall_seconds": time.perf_counter() - started,
    "json_sha256": hashlib.sha256(json_before).hexdigest(),
    "report_sha256": hashlib.sha256(report_before).hexdigest(),
    "canonical_result_sha256": result["canonical_result_sha256"],
    "maximum": {k: result["mass_maximum"][k]
                for k in ("nc_over_n0", "M_Msun", "R_km", "Lambda")},
    "numerical_convergence": result["convergence"]["pass"],
    "evidence_weight": result["evidence_weight"],
}, sort_keys=True), flush=True)
