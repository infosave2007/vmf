"""Live numerical/proof smoke; no stored result supplies integration inputs."""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "verification"))
import source_complete_stellar_prediction as s

start = time.perf_counter()
q = s.integrate_pressure_rk4(.001, dr=.05)
audit = q["surface_event"]["jet"]["roundoff_audit"]
path = pathlib.Path(__file__).with_name("stellar-live-witness-2026-09-08.json")
path.write_text(json.dumps(q, indent=2, sort_keys=True, allow_nan=False) + "\n")
print("LIVE_WITNESS", time.perf_counter()-start, audit["status"], flush=True)

def trace(frame, event, arg):
    if event == "return" and arg is False and frame.f_code.co_name in (
        "_terminal_independent_replay_impl", "_roundoff_audit_accepts_impl"
    ):
        print("REPLAY_REJECTION", frame.f_code.co_name, frame.f_lineno, flush=True)
    if event == "exception" and frame.f_code.co_name in (
        "_terminal_independent_replay_impl", "_roundoff_audit_accepts_impl"
    ):
        print("REPLAY_EXCEPTION", frame.f_code.co_name, frame.f_lineno, repr(arg[1]), flush=True)
    return trace if frame.f_code.co_name in ("_terminal_independent_replay_impl", "_roundoff_audit_accepts_impl") else None

sys.settrace(trace)
print("ACCEPTOR", s._roundoff_audit_accepts(audit), flush=True)
sys.settrace(None)
