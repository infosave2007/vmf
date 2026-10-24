"""Fresh stellar production/validation with external elapsed-time stage logs."""
import functools
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))
import source_complete_stellar_prediction as s

started = time.perf_counter()

def stage(message, **fields):
    print(json.dumps({"stage": message, "wall_seconds": round(time.perf_counter()-started, 3), **fields}, sort_keys=True), flush=True)

def instrument(name):
    original = getattr(s, name)
    @functools.wraps(original)
    def run(*args, **kwargs):
        argument = "grids32,64" if name == "_convergence_summary" else (str(args[0])[:80] if args else "")
        stage(name + ":begin", argument=argument)
        value = original(*args, **kwargs)
        stage(name + ":end")
        return value
    setattr(s, name, run)

for name in ("_matched_grid", "_convergence_summary", "_frobenius_coefficients", "integrate_pressure_rk4", "gamma2_newtonian_control"):
    instrument(name)

original_enthalpy = s.integrate_enthalpy
enthalpy_count = 0
@functools.wraps(original_enthalpy)
def enthalpy(*args, **kwargs):
    global enthalpy_count
    value = original_enthalpy(*args, **kwargs)
    enthalpy_count += 1
    if enthalpy_count % 50 == 0:
        stage("enthalpy_progress", completed=enthalpy_count)
    return value
s.integrate_enthalpy = enthalpy

mode = sys.argv[1] if len(sys.argv) > 1 else "build"
stage("start", mode=mode, python=sys.version, upstream=s.upstream_digest())
if mode == "build":
    result = s.build_result()
    if result["convergence"]["pass"] is not True:
        raise RuntimeError("Numerical convergence failed; canonical artifacts were not changed")
    completed = dict(status=result["status"], canonical_result_sha256=result["canonical_result_sha256"],
                     owner_registry=result["owner_evaluation_registry"],
                     interval_diagnostics=[{"x": w["x"], "replay": w["interval_diagnostic"]["independent_replay_pass"],
                                            "rounding_cell_check": w["interval_diagnostic"]["rounding_cell_check_pass"]}
                                           for w in result["terminal_interval_witnesses"]])
    s.write_artifacts(result)
    stage("completed", **completed)
elif mode == "validate":
    result = json.loads(s.RESULT_PATH.read_text())
    ok, errors = s.validate_contract(result)
    stage("public_validation", accepted=ok, errors=errors)
    if not ok:
        raise SystemExit(1)
else:
    raise ValueError(mode)
