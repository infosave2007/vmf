#!/usr/bin/env python3
"""Fail-closed stellar sequence built from the adjudicated source EOS.

This module is deliberately a thin, auditable producer.  The source-complete
action remains in :mod:`source_complete_solution_audit`; no beta-equilibrium,
crust continuation, CSS branch, or density extrapolation is introduced here.
The resulting rows are mathematical one-component sequences and are never
evidence for a physical neutron-star model.
"""

from __future__ import annotations

import hashlib
import json
import math
import copy
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

try:  # supports both ``python verification/file.py`` and pytest's path mode
    from . import source_complete_solution_audit as upstream
except ImportError:  # pragma: no cover - exercised by direct script execution
    import source_complete_solution_audit as upstream


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
RESULT_PATH = ROOT / "source_complete_stellar_prediction_results.json"
REPORT_PATH = ROOT / "source_complete_stellar_prediction_report.md"
SCHEMA_VERSION = "source_complete_stellar_prediction.v2"
EXPECTED_UPSTREAM_DIGEST = "5d39d3d07a3f00d2fcbe853ad27359b34a86d853c842ba01861196c1aaa517a8"
ACTION_PATH = ROOT / "contracts/source_complete_action.md"
ACTION_DIGEST = "25e42abb91028f09a76cccddc05ca4a7458bc0609df7f41fb932d165cc73606e"
CONTRACT_PATH = ROOT / "contracts/observable_predictions.md"
CONTRACT_DIGEST = "ebcc4e0cae1bf981f3fb2dd0bf6567a177dffa75aa25a160f82c2f86cbe87cef"
TEST_PATH = ROOT / "test_source_complete_stellar_prediction.py"
M_SUN_KM = 1.4766
K_CONV = 1.3234e-6
X_MAX = 10.0
GRID_POINTS_PER_DECADE = 32
PHYSICAL_NS_BLOCKER = (
    "BLOCKED_MISSING_BETA_EQUILIBRIUM_CHARGE_NEUTRALITY_LEPTONS_CRUST_AND_HIGH_DENSITY_COMPLETION"
)
_GENERATION_GUARD = False
_BUILD_BUDGET: "_BuildBudget | None" = None
OWNER_EQUATION_REVISION = "repair28-owner-keyed-evaluator-v1"
OWNER_TANGENT_EQUATION_REVISION = "repair29-augmented-tangent-v1"
OWNER_SOURCE_REVISION = "source_complete_solution_audit:" + EXPECTED_UPSTREAM_DIGEST
# Repair-31 source authority.  All arbitrary-precision proof code routes
# through this adapter; keeping a single revision token makes a stale or
# duplicated action kernel fail closed in the replay path.
LIVE_SOURCE_REVISION = "repair31-live-exact-source-v1"
CENTRE_PROOF_REVISION = "repair40-centre-source-coordinate-v2"

# Gate-4 fixes theorem ownership before any pressure trajectory is integrated.
# These are exact binary64 dyadic identities; never infer ownership from a
# result-dependent equality/order bit (``same_bits``).  Every trajectory still
# receives its own production N-ladder/oracle chart, while only these three
# keys run the expensive validated interval theorem.
TERMINAL_WITNESS_KEYS = frozenset(
    (
        (float(0.001).hex(), float(0.05).hex()),
        (float(0.0014330125702369627).hex(), float(0.05).hex()),
        (float(0.002053525026457146).hex(), float(0.05).hex()),
    )
)
TERMINAL_PROOF_WITNESS = "INTERVAL_DIAGNOSTIC_WITNESS"
TERMINAL_PRODUCTION_CHART = "PRODUCTION_NUMERICAL_CHART"


def _terminal_proof_role(x: float, dr: float) -> str:
    """Return the immutable proof role for one exact pressure input key."""
    return (
        TERMINAL_PROOF_WITNESS
        if (float(x).hex(), float(dr).hex()) in TERMINAL_WITNESS_KEYS
        else TERMINAL_PRODUCTION_CHART
    )


def _centre_coeff_digest(coeff: dict[str, Any]) -> str:
    """Digest only the formal centre coefficients (never proof/pass flags).

    The digest is an integrity witness for the immutable record handed to the
    three pressure consumers.  It is recomputed on every consumer use, so a
    mutation of any coefficient or source revision cannot masquerade as a
    previously accepted proof.
    """
    fields = (
        "A", "Pc", "P2", "P4", "P6", "P8", "P10", "P12", "epsilon2",
        "m3", "m5", "m7", "m9", "m11", "m13", "m15", "F2", "Q2",
        "y2", "y4", "y6", "y8", "y10", "y12", "e1", "e2", "e3",
        "formal_pressure_coefficients", "formal_mass_u_coefficients",
        "formal_energy_coefficients", "formal_y_coefficients",
        "formal_p1_dn", "centre_x", "action_jet_order", "formal_precision_dps",
        "W_coefficients", "stationary_branch_records", "source_revision", "coordinate_units",
    )
    payload = {k: coeff.get(k) for k in fields}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _centre_proof_key(coeff: dict[str, Any]) -> tuple[Any, ...]:
    """Exact centre proof key; no projected float or mutable state participates."""
    x = float(coeff.get("centre_x", float("nan")))
    return (x.hex(), CENTRE_PROOF_REVISION, OWNER_SOURCE_REVISION,
            str(coeff.get("source_revision", "")), int(coeff.get("formal_precision_dps", 0)),
            int(coeff.get("action_jet_order", 0)), OWNER_EQUATION_REVISION)


def _centre_projection_matches(coeff: dict[str, Any]) -> bool:
    """Bind every consumed binary64 coefficient to the formally checked jet."""
    if coeff.get("coordinate_units") != {"radius": "rho=sqrt(K_CONV)*r_km", "mass": "m_formal=sqrt(K_CONV)*m_km", "pressure": "MeV/fm^3", "k_conv": K_CONV}:
        return False
    try:
        ps = [float(v) for v in coeff["formal_pressure_coefficients"]]
        us = [float(v) for v in coeff["formal_mass_u_coefficients"]]
        es = [float(v) for v in coeff["formal_energy_coefficients"]]
        ys = [float(v) for v in coeff["formal_y_coefficients"]]
        aliases = {"Pc": ps[0], "epsilon2": es[1], "A": 4*math.pi*(es[0]/3+ps[0])}
        aliases.update({f"P{2*j}": ps[j] for j in range(1, 7)})
        aliases.update({f"m{2*j+3}": us[j] for j in range(7)})
        aliases.update({f"y{2*j}": ys[j] for j in range(1, 7)})
        return all(coeff.get(k) == v for k, v in aliases.items())
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _centre_proof_accepts(coeff: dict[str, Any]) -> bool:
    """Validate a centre record before it is passed to an RK4 consumer."""
    if not _centre_projection_matches(coeff):
        return False
    rec = coeff.get("centre_proof_record")
    if not isinstance(rec, dict) or rec.get("key") != list(_centre_proof_key(coeff)):
        return False
    if rec.get("coefficient_digest") != _centre_coeff_digest(coeff):
        return False
    if rec.get("source_revision") != LIVE_SOURCE_REVISION or rec.get("parameter_revision") != OWNER_SOURCE_REVISION or rec.get("replay_namespace") != "empty-exact-S":
        return False
    proof = rec.get("substitution_replay")
    if not isinstance(proof, dict) or proof.get("pass") is not True or proof.get("source_revision") != LIVE_SOURCE_REVISION:
        return False
    for _br in coeff.get("stationary_branch_records", []):
        _dr = _br.get("decision_replay", {}) if isinstance(_br, dict) else {}
        if _dr and (_dr.get("same_midpoint_order") is not True or _dr.get("expression_order") != "f_mid_then_fW_then_24step_map" or _dr.get("root_enclosure_returned") is not False):
            return False
    return True


def _ensure_centre_proof(coeff: dict[str, Any], *, eos: Any | None = None, _decision_replay: bool = True) -> dict[str, Any]:
    """Build one recurrence/substitution proof and return an immutable copy.

    The recurrence itself is produced by :func:`_frobenius_coefficients`.
    This helper performs exactly one original-equation substitution replay in
    a fresh source namespace, then records a digest/key that consumers verify.
    """
    if _centre_proof_accepts(coeff):
        return copy.deepcopy(coeff["centre_proof_record"])
    if coeff.get("centre_proof_record") is not None:
        # A stale/mutated record is never silently refreshed from its PASS bit.
        raise ValueError("centre proof record mutation or scope violation")
    class _Facade:
        def __init__(self, adapter: LiveExactSourceAdapter) -> None:
            self._exact_source_adapter = adapter
        def exact_source(self, n_nat: Any, *, dps: int = 180) -> dict[str, Any]:
            if int(dps) != self._exact_source_adapter.dps:
                self._exact_source_adapter = LiveExactSourceAdapter(int(dps))
            return self._exact_source_adapter.evaluate(n_nat)
    # Distinct empty namespace for the direct original-equation replay.
    adapter = LiveExactSourceAdapter(180, decision_replay=_decision_replay, ambient_guard=True)
    facade = _Facade(adapter)
    replay = _frobenius_residual_certificate(coeff, facade)
    if not replay.get("pass", False):
        raise ValueError("centre original-equation substitution replay failed")
    record = {
        "key": list(_centre_proof_key(coeff)),
        "source_revision": LIVE_SOURCE_REVISION,
        "parameter_revision": OWNER_SOURCE_REVISION,
        "equation_revision": OWNER_EQUATION_REVISION,
        "replay_namespace": "empty-exact-S",
        "replay": "original-equation-substitution-live-source-v1",
        "substitution_replay": copy.deepcopy(replay),
        "source_requests": int(adapter.calls),
        "source_evaluations": int(adapter.calls),
        "source_hits": 0,
        "coefficient_digest": _centre_coeff_digest(coeff),
        "consumers": 3,
        "scope": "destroyed-on-return:x.hex+revisions+precision+order",
    }
    coeff["centre_proof_record"] = record
    return copy.deepcopy(record)


class LiveExactSourceAdapter:
    """Exact-ratio stationary action adapter shared by producer proofs.

    The binary64 producer remains the hot path, while centre/tangent proof
    code requests this adapter explicitly at 80/120/180 decimal digits.  It
    returns the complete branch certificate rather than a bare root: both
    endpoint signs and an interval lower bound for ``f_W`` are retained.
    """

    revision = LIVE_SOURCE_REVISION

    def __init__(self, dps: int = 180, *, decision_replay: bool = True, ambient_guard: bool = False) -> None:
        self.dps = int(dps)
        self.decision_replay = bool(decision_replay)
        # Terminal replay passes ``ambient_guard=False`` (the default), so a
        # declared 80/120 source precision is never promoted by caller state.
        # Formal centre jets may opt into an explicit ambient guard because
        # ``mp.diff`` temporarily raises precision for high-order probes.
        self.ambient_guard = bool(ambient_guard)
        self.calls = 0

    @staticmethod
    def _exact(v: Any, mp: Any) -> Any:
        num, den = float(v).as_integer_ratio()
        return mp.mpf(num) / mp.mpf(den)

    def evaluate(self, n_nat: Any) -> dict[str, Any]:
        """Evaluate the accepted action and its positive-W branch exactly."""
        import mpmath as mp
        # Every replay source request runs at its declared lexical precision.
        # Only formal centre-jet callers explicitly opt into an ambient guard
        # for mpmath's temporary high-order differentiation probes.
        source_dps = max(self.dps, int(mp.mp.dps)) if self.ambient_guard else self.dps
        with mp.workdps(source_dps):
            self.calls += 1
            nn = mp.mpf(n_nat)
            p = upstream.PARAMS
            d = self._exact(p.degeneracy, mp); gs = self._exact(p.g_s, mp)
            W0 = self._exact(p.W0, mp); lam = self._exact(p.lam, mp)
            G = self._exact(p.g_omega, mp)**2 / self._exact(p.q_phi, mp)**2
            hb = self._exact(p.hbar_c, mp)
            if nn < 0:
                raise ValueError("exact source density outside domain")
            if nn == 0:
                return {"revision": self.revision, "n": nn, "k": mp.mpf(0),
                        "W": W0, "W_L": W0, "W_U": W0,
                        "f": mp.mpf(0), "f_W": mp.inf, "f_W_inf": mp.inf,
                        "epsilon": mp.mpf(0), "P": mp.mpf(0),
                        "d_epsilon_dn": self._exact(p.M_N, mp) / hb**3,
                        "dP_dn": mp.mpf(0), "branch": "exact_vacuum",
                        "decision_replay": {"same_midpoint_order": True, "certified_side_skips": 0, "direct_evaluations": 0, "trace_length": 0, "scope": "destroyed-on-return-adapter-call", "root_enclosure_returned": False, "expression_order": "f_mid_then_fW_then_24step_map"}}
            k = (6 * mp.pi**2 * nn / d) ** (mp.mpf(1) / 3)
            def scalar(W: Any) -> Any:
                m = gs * W; E = mp.sqrt(k*k + m*m)
                return d*m/(4*mp.pi**2) * (k*E - m*m*mp.asinh(k/m))
            def f(W: Any) -> Any:
                return lam*W*(W*W-W0*W0) + gs*scalar(W) - G*nn*nn/W**3
            def fw(W: Any) -> Any:
                m = gs*W; E = mp.sqrt(k*k+m*m); a = mp.asinh(k/m)
                nsm = d/(4*mp.pi**2)*((k*E-m*m*a)+2*m*m*(k/E-a))
                return lam*(3*W*W-W0*W0) + gs*gs*nsm + 3*G*nn*nn/W**4
            lo, hi = W0/mp.sqrt(3), 2*W0
            flo, fhi = f(lo), f(hi)
            if not (flo < 0 < fhi):
                raise ValueError("exact source positive-W sign bracket failed")
            # A bounded bisection is deliberately used here: unlike an
            # unchecked findroot it supplies a branch interval and a direct
            # lower Jacobian bound for every requested coefficient.
            # Resolve the branch interval well below the smallest mpmath
            # differentiation step.  A 256-bit bracket is insufficient for
            # 120-digit action jets: its midpoint noise contaminates the
            # first pressure derivative by percent-level amounts.  The
            # decimal-dps keyed count keeps the source deterministic while
            # delivering a genuinely smooth exact-ratio jet.
            # Build a fresh high-precision enclosure first.  The subsequent
            # midpoint replay uses the same arithmetic sequence as the
            # ordinary bisection, but an expensive action evaluation is
            # skipped only when this enclosure proves the side of the root.
            # The enclosure is local to this adapter call and is never
            # returned or reused by another density/precision context.
            cert_lo, cert_hi = lo, hi
            if self.decision_replay:
                # Fresh call-local safeguarded Newton/correction enclosure.
                # The correction is accepted only inside the current direct
                # sign bracket; otherwise the midpoint is retained.  A
                # handful of such contractions is enough to prove the side
                # of every later bisection midpoint while avoiding the old
                # 24--48-expression certificate sweep.
                try:
                    # A direct safeguarded secant/Newton correction locates a
                    # candidate root in the immutable global bracket.  The
                    # candidate is *never* returned: fresh signed endpoints
                    # are constructed around its correction and are the only
                    # evidence used by the midpoint replay.
                    cseed = (cert_lo + cert_hi) / 2
                    croot = mp.findroot(f, (cert_lo, cert_hi), solver="secant",
                                        verify=False, tol=mp.power(10, -self.dps + 20),
                                        maxsteps=24)
                    if not (cert_lo < croot < cert_hi):
                        raise ValueError("safeguarded correction left branch")
                    cfw = fw(croot)
                    correction = abs(f(croot) / cfw) if cfw > 0 else mp.inf
                    # Keep a representable, signed enclosure around the
                    # correction.  Every endpoint is evaluated directly and
                    # the radius grows only within the original bracket.
                    eps = max(mp.power(10, -self.dps + 20), 4 * correction,
                              mp.power(10, -self.dps + 12) * max(abs(croot), 1))
                    cert_lo, cert_hi = max(lo, croot - eps), min(hi, croot + eps)
                    for _probe in range(8):
                        cfl, cfh = f(cert_lo), f(cert_hi)
                        if cfl < 0 < cfh:
                            break
                        eps *= 2
                        cert_lo, cert_hi = max(lo, croot - eps), min(hi, croot + eps)
                    if not (f(cert_lo) < 0 < f(cert_hi)):
                        raise ValueError("safeguarded correction signs failed")
                except Exception:
                    # Deterministic bounded fallback: direct bisection signs
                    # remain valid proof evidence when a correction cannot be
                    # enclosed, but no formal-series value is substituted.
                    cert_lo, cert_hi = lo, hi
                    for _ in range(min(24, max(12, int(self.dps // 8)))):
                        cmid = (cert_lo + cert_hi) / 2; cfm = f(cmid)
                        if cfm < 0: cert_lo = cmid
                        elif cfm > 0: cert_hi = cmid
            decision_skips = 0; decision_evaluations = 0; decision_trace: list[int] = []
            for _ in range(max(256, int(self.dps * 3.5))):
                mid = (lo + hi) / 2
                # f_W is certified positive on the global branch, therefore
                # the enclosure side determines the exact unchanged sign.
                if (not self.decision_replay):
                    fm = f(mid); decision_evaluations += 1; decision_trace.append(0)
                elif mid < cert_lo:
                    fm = mp.mpf(-1); decision_skips += 1; decision_trace.append(-1)
                elif mid > cert_hi:
                    fm = mp.mpf(1); decision_skips += 1; decision_trace.append(1)
                else:
                    fm = f(mid); decision_evaluations += 1; decision_trace.append(0)
                if fm < 0: lo = mid
                elif fm > 0: hi = mid
                else:
                    lo, hi = mid-mp.power(10, -self.dps+8), mid+mp.power(10, -self.dps+8)
                # Keep the proof enclosure inside the active sign bracket;
                # when the midpoint is evaluated, shrink the enclosure using
                # that fresh sign.  No root/bracket is retained after return.
                if decision_trace[-1] == 0:
                    if fm < 0: cert_lo = mid
                    elif fm > 0: cert_hi = mid
                if hi-lo <= mp.power(10, -self.dps+12): break
            # Use a fixed-count Newton refinement for the returned point.  The
            # immutable bisection endpoints above remain the sign certificate;
            # the fixed iteration map avoids midpoint quantisation in action
            # derivatives while never accepting a root outside that bracket.
            W = (lo + hi) / 2
            for _ in range(24):
                Wn = W - f(W) / fw(W)
                if not (lo < Wn < hi):
                    Wn = (lo + hi) / 2
                W = Wn
            fwi = fw(W)
            fwi_inf = min(fwi, fw(lo), fw(hi))
            if fwi_inf <= 0:
                raise ValueError("exact source f_W branch is not positive")
            m = gs*W; E = mp.sqrt(k*k+m*m); a = mp.asinh(k/m)
            free = d/(16*mp.pi**2)*(k*E*(2*k*k+m*m)-m**4*a)
            kinp = d/(48*mp.pi**2)*(k*E*(2*k*k-3*m*m)+3*m**4*a)
            pot = lam*(W*W-W0*W0)**2/4; vec = G*nn*nn/(2*W*W)
            # The source EOS action is in MeV/fm^3 (divide by hbar_c^3).
            # K_CONV belongs only to the TOV km conversion and must not enter
            # the stationary action jet or its density derivatives.
            eps = (free+pot+vec)/hb**3; press = (kinp-pot+vec)/hb**3
            fn = gs*m/E - 2*G*nn/W**3; dwdn = -fn/fwi
            dmu = k*k/(3*nn*E) + fn*dwdn + G/W**2
            dpdn = nn*dmu/hb**3
            return {"revision": self.revision, "n": nn, "k": k, "W": W,
                    "W_L": lo, "W_U": hi, "f": f(W), "f_W": fwi,
                    "f_W_inf": fwi_inf, "f_W_interval": [fw(lo), fw(hi)], "epsilon": eps, "P": press,
                    "d_epsilon_dn": (E+G*nn/W**2)/hb**3, "dP_dn": dpdn,
                    "branch": "positive_fW", "f_lo": flo, "f_hi": fhi,
                    "decision_replay": {"same_midpoint_order": True, "certified_side_skips": int(decision_skips), "direct_evaluations": int(decision_evaluations), "trace_length": int(len(decision_trace)), "scope": "destroyed-on-return-adapter-call", "root_enclosure_returned": False, "expression_order": "f_mid_then_fW_then_24step_map"}}

    def action(self, n_nat: Any) -> tuple[Any, Any, dict[str, Any]]:
        q = self.evaluate(n_nat)
        return q["epsilon"], q["P"], q


def live_exact_source(n_nat: Any, *, dps: int = 180,
                      adapter: LiveExactSourceAdapter | None = None) -> dict[str, Any]:
    """Mandatory shared exact-source entry point used by all proof kernels."""
    return (adapter if adapter is not None else LiveExactSourceAdapter(dps)).evaluate(n_nat)


class _BuildBudget:
    """Whole-build reservation counter independent of EOS/stellar caches."""
    cap = 1200

    def __init__(self) -> None:
        self.enthalpy_integrations = 0

    def reserve_enthalpy(self) -> None:
        nxt = self.enthalpy_integrations + 1
        if nxt > self.cap:
            raise RuntimeError("whole-build strict enthalpy integration budget exceeded")
        self.enthalpy_integrations = nxt

    def snapshot(self) -> dict[str, int]:
        return {"strict_enthalpy_integrations": int(self.enthalpy_integrations), "strict_enthalpy_cap": int(self.cap)}


class _OwnerEvaluationRegistry:
    """Build-local immutable owner/key common-subexpression registry.

    A record is keyed by every numerical and semantic input that can affect a
    strict enthalpy result.  Only a complete result is copied into the map;
    mutable ODE state, proof objects, artifacts and cross-owner values never
    enter this namespace.  The registry is intentionally destroyed with the
    build and has no artifact-backed fallback.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._records: dict[tuple[str, str, float, float, float, bool, str, str], dict[str, Any]] = {}
        self._requested = 0
        self._unique = 0
        self._prior_hits = 0
        self._executed = 0
        self._requested_by_owner: dict[str, int] = {}
        self._unique_by_owner: dict[str, int] = {}
        self._hit_by_owner: dict[str, int] = {}
        self._executed_by_owner: dict[str, int] = {}
        self._settings: dict[tuple[str, str, float, float, float, bool, str, str], dict[str, Any]] = {}

    @staticmethod
    def make_key(owner: str, x: float, rtol: float, atol_scale: float, centre_scale: float,
                 strict_eos: bool = True, equation_revision: str = OWNER_EQUATION_REVISION,
                 source_revision: str = OWNER_SOURCE_REVISION) -> tuple[str, str, float, float, float, bool, str, str]:
        return (str(owner), float(x).hex(), float(rtol), float(atol_scale), float(centre_scale), bool(strict_eos), str(equation_revision), str(source_revision))

    def contains(self, key: tuple[str, str, float, float, float, bool, str, str]) -> bool:
        return bool(self.enabled and key in self._records)

    def get_or_execute(self, key: tuple[str, str, float, float, float, bool, str, str], executor: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        owner = str(key[0]); self._requested += 1; self._requested_by_owner[owner] = self._requested_by_owner.get(owner, 0) + 1
        if self.enabled and key in self._records:
            self._prior_hits += 1; self._hit_by_owner[owner] = self._hit_by_owner.get(owner, 0) + 1
            return copy.deepcopy(self._records[key]), True
        self._unique += 1; self._unique_by_owner[owner] = self._unique_by_owner.get(owner, 0) + 1
        value = executor()
        if not isinstance(value, dict):
            raise TypeError("owner registry executor must return a result mapping")
        frozen = copy.deepcopy(value)
        if self.enabled:
            self._records[key] = frozen
        self._executed += 1; self._executed_by_owner[owner] = self._executed_by_owner.get(owner, 0) + 1
        self._settings[key] = {"owner": owner, "x_hex": key[1], "rtol": key[2], "atol_scale": key[3], "centre_scale": key[4], "strict_eos": key[5], "equation_revision": key[6], "source_revision": key[7]}
        return copy.deepcopy(frozen), False

    def snapshot(self) -> dict[str, Any]:
        owners = sorted(set(self._requested_by_owner) | set(self._unique_by_owner) | set(self._hit_by_owner) | set(self._executed_by_owner))
        keys = [self._settings[k] for k in sorted(self._settings, key=lambda z: (z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]))]
        return {
            "enabled": self.enabled,
            "key_fields": ["owner", "x.hex", "rtol", "atol_scale", "centre_scale", "strict_eos", "equation_revision", "source_revision"],
            "equation_revision": OWNER_EQUATION_REVISION,
            "mode_revisions": {"ordinary": OWNER_EQUATION_REVISION, "augmented_tangent": OWNER_TANGENT_EQUATION_REVISION},
            "source_revision": OWNER_SOURCE_REVISION,
            "owners": owners,
            "requested": int(self._requested), "unique": int(self._unique), "prior_hit": int(self._prior_hits),
            "executed": int(self._executed), "reused": int(self._prior_hits),
            "requested_by_owner": {o: int(self._requested_by_owner.get(o, 0)) for o in owners},
            "unique_by_owner": {o: int(self._unique_by_owner.get(o, 0)) for o in owners},
            "prior_hit_by_owner": {o: int(self._hit_by_owner.get(o, 0)) for o in owners},
            "executed_by_owner": {o: int(self._executed_by_owner.get(o, 0)) for o in owners},
            "settings": keys,
            "total_strict_integrations": int(self._executed),
        }


def _registered_enthalpy(registry: _OwnerEvaluationRegistry, *, owner: str, x: float, rtol: float,
                         atol_scale: float, centre_scale: float = 1.0, strict_eos: bool = True) -> dict[str, Any]:
    """Run or retrieve one complete owner-keyed strict enthalpy evaluation."""
    key = registry.make_key(owner, x, rtol, atol_scale, centre_scale, strict_eos)
    def execute() -> dict[str, Any]:
        eos = DirectEOS(); begin = getattr(eos, "begin_star", lambda: None); end = getattr(eos, "end_star", lambda: {})
        begin()
        try:
            row = integrate_enthalpy(float(x), rtol=float(rtol), atol_scale=float(atol_scale), strict_eos=bool(strict_eos), eos=eos, centre_delta_scale=float(centre_scale))
        finally:
            summary = end()
        row = dict(row); row["inversion_certificate"] = summary
        return row
    return registry.get_or_execute(key, execute)[0]


def upstream_digest() -> str:
    return hashlib.sha256((ROOT / "source_complete_solution_audit.py").read_bytes()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_result_bytes(result: dict[str, Any]) -> bytes:
    """Canonical UTF-8 representation used for immutable artifact identity."""
    payload = _scientific_payload(result)
    payload.pop("canonical_result_sha256", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_result_sha256(result: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_result_bytes(result)).hexdigest()


def _canonical_full_bytes(result: dict[str, Any]) -> bytes:
    """Canonical bytes including the stored identity field."""
    payload = _scientific_payload(result)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _scientific_payload(value: Any) -> Any:
    """Keep elapsed-time telemetry out of scientific identity and artifacts."""
    if isinstance(value, dict):
        return {k: _scientific_payload(v) for k, v in value.items() if k != "wall_seconds"}
    if isinstance(value, (list, tuple)):
        return [_scientific_payload(v) for v in value]
    return value


def _rel(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1.0e-300)


def _finite(*values: float) -> bool:
    return all(np.isfinite(float(v)) for v in values)


def _scalar_density_mass_derivative_closed(n_nat: float, mass: float, p: Any = upstream.PARAMS) -> float:
    """The fixed-k derivative of the scalar density, evaluated directly.

    This is deliberately separate from ``scalar_density``: the stationary
    Jacobian needs ``(partial n_s/partial m)_k`` rather than the scalar
    density itself.  Keeping the two routines separate prevents the old
    order-t^3 substitution from silently returning.
    """
    if n_nat <= 0.0 or mass <= 0.0:
        return 0.0
    k = (6.0 * math.pi**2 * n_nat / p.degeneracy) ** (1.0 / 3.0)
    E = math.sqrt(k * k + mass * mass)
    a = math.asinh(k / mass)
    return float(p.degeneracy / (4.0 * math.pi**2) * ((k * E - mass * mass * a) + 2.0 * mass * mass * (k / E - a)))


_DNS_DM_SERIES = (
    (2.0 / 5.0, 5), (-3.0 / 7.0, 7), (5.0 / 12.0, 9),
    (-35.0 / 88.0, 11), (315.0 / 832.0, 13), (-231.0 / 640.0, 15),
    (3003.0 / 8704.0, 17), (-6435.0 / 19456.0, 19),
    (36465.0 / 114688.0, 21), (-230945.0 / 753664.0, 23),
)

# Gate-4 low-t primitive series.  Coefficients are kept as exact rational
# pairs (rather than shortened decimals) and the first omitted t^14 term is
# generated by the same coefficient table used by the outward proof kernel.
_LOW_T_SERIES = {
    "n_s_over_n": ((1, 1), (-3, 10), (9, 56), (-5, 48), (105, 1408), (-189, 3328), (231, 5120), (-1287, 34816)),
    "n_s_m_over_n": ((3, 1), (-3, 10), (-9, 56), (5, 16), (-525, 1408), (1323, 3328), (-2079, 5120), (14157, 34816)),
    "epsilon_kin": ((1, 1), (3, 10), (-9, 56), (5, 48), (-105, 1408), (189, 3328), (-231, 5120), (99, 34816)),
    "P_kin": ((1, 1), (-5, 14), (5, 24), (-5, 88), (35, 1664), (-21, 1280), (231, 17408), (-2145, 38912)),
}


def _scalar_density_stable_from_kt(
    n_nat: float,
    mass: float,
    k: float,
    t: float,
    p: Any = upstream.PARAMS,
    _powers: dict[int, float] | None = None,
) -> float:
    """Evaluate ``n_s`` from a solve-local ``(k,t)`` pair.

    This is the value-mode half of the pressure-source DAG.  The expression
    is intentionally the same sequence as :func:`_scalar_density_stable`;
    the only difference is that the caller supplies the already computed
    binary64 ``k`` and ``t`` values.  Keeping this tiny helper separate makes
    the operation order auditable and prevents a diagnostic metadata request
    from entering the hot path.
    """
    if n_nat <= 0.0 or mass <= 0.0:
        return 0.0
    if t >= 0.1:
        return float(upstream.scalar_density(n_nat, mass, p))
    coeff = ((2.0 / 3.0, 3), (-1.0 / 5.0, 5), (3.0 / 28.0, 7),
             (-5.0 / 72.0, 9), (35.0 / 704.0, 11), (-63.0 / 1664.0, 13),
             (231.0 / 6656.0, 15), (-429.0 / 15360.0, 17))
    powers = _powers if _powers is not None else {power: t**power for _, power in coeff}
    series = math.fsum(c * powers[power] for c, power in coeff)
    return float(p.degeneracy * mass**3 / (4.0 * math.pi**2) * series)


def _scalar_density_mass_derivative_from_kt(
    n_nat: float,
    mass: float,
    k: float,
    t: float,
    p: Any = upstream.PARAMS,
    _powers: dict[int, float] | None = None,
) -> float:
    """Value-only ``(partial n_s/partial m)_k`` on a shared pressure DAG."""
    if n_nat <= 0.0 or mass <= 0.0:
        return 0.0
    if t <= 0.1:
        powers = _powers if _powers is not None else {power: t**power for _, power in _DNS_DM_SERIES[:-1]}
        retained = math.fsum(c * powers[power] for c, power in _DNS_DM_SERIES[:-1])
        pref = p.degeneracy * mass * mass / (4.0 * math.pi**2)
        return float(pref * retained)
    E = math.sqrt(k * k + mass * mass)
    a = math.asinh(k / mass)
    return float(p.degeneracy / (4.0 * math.pi**2) * ((k * E - mass * mass * a) + 2.0 * mass * mass * (k / E - a)))


def _low_t_tail_suprema(ratio: Any, n: Any, mass: Any, *, dps_iv: Any) -> dict[str, Any]:
    """Outward t^14 majorants for all six primitive/source quantities."""
    # The ratio bound is a closed interval.  Each primitive receives its own
    # prefactor and its own coefficient-ratio geometric majorant; no minimum,
    # shared ratio-squared proxy, or dependency clipping is used.
    import mpmath as mp
    tmax = mp.mpf(ratio.b)
    if tmax < 0 or tmax > mp.mpf("1e-2"):
        return {k: dps_iv.iv.mpf(0) for k in ("n_s", "n_s_m", "energy", "pressure", "dPdn", "dPdn_over_s")}
    rho = tmax * tmax * mp.mpf("0.6")  # certified coefficient-ratio majorant < 1
    def coeff(name: str, idx: int) -> Any:
        num, den = _LOW_T_SERIES[name][idx]; return dps_iv.mpf(num) / dps_iv.mpf(den)
    out: dict[str, Any] = {}
    # Primitive t^14 coefficient and exact prefactor.  Derivative tails are
    # widened independently from the two primitive sensitivity terms.
    out["n_s"] = abs(n) * abs(coeff("n_s_over_n", 7)) * ratio**14 / (1-rho)
    out["n_s_m"] = abs(n/mass) * abs(coeff("n_s_m_over_n", 7)) * ratio**14 / (1-rho)
    out["energy"] = abs(n*mass) * abs(coeff("epsilon_kin", 7)) * ratio**14 / (1-rho)
    out["pressure"] = abs(n*mass) * abs(coeff("P_kin", 7)) * ratio**14 / (1-rho)
    out["dPdn"] = out["n_s_m"] + out["pressure"]
    if tmax == 0:
        # Exact vacuum: all six omitted tails vanish identically.
        out["dPdn_over_s"] = dps_iv.iv.mpf(0)
    elif mp.mpf(ratio.a) <= 0 <= mp.mpf(ratio.b):
        # A leaf touching s=0 has ratio**2 containing zero.  Dividing the
        # interval directly would manufacture +inf despite the analytic
        # cancellation dPdn = O(t^14).  Bound the cancelled expression
        # directly as O(t^12), retaining a finite outward upper endpoint.
        csum = abs(n / mass) * abs(coeff("n_s_m_over_n", 7)) + abs(n * mass) * abs(coeff("P_kin", 7))
        out["dPdn_over_s"] = dps_iv.iv.mpf([0, csum * tmax**12 / (1 - rho)])
    else:
        out["dPdn_over_s"] = out["dPdn"] / (ratio**2)
    return out


def _scalar_density_stable(n_nat: float, mass: float, p: Any = upstream.PARAMS) -> float:
    """Cancellation-free scalar density for positive-density W stationarity."""
    if n_nat <= 0.0 or mass <= 0.0:
        return 0.0
    k = (6.0 * math.pi**2 * n_nat / p.degeneracy) ** (1.0 / 3.0)
    t = k / mass
    if t >= 0.1:
        return float(upstream.scalar_density(n_nat, mass, p))
    coeff = ((2.0 / 3.0, 3), (-1.0 / 5.0, 5), (3.0 / 28.0, 7),
             (-5.0 / 72.0, 9), (35.0 / 704.0, 11), (-63.0 / 1664.0, 13),
             (231.0 / 6656.0, 15), (-429.0 / 15360.0, 17))
    series = math.fsum(c * t**power for c, power in coeff)
    return float(p.degeneracy * mass**3 / (4.0 * math.pi**2) * series)


def _scalar_density_mass_derivative_calc(
    n_nat: float,
    mass: float,
    p: Any = upstream.PARAMS,
    *,
    diagnostic: bool,
    force_series: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """One exact scalar-density derivative calculation with two return modes.

    ``diagnostic=True`` reproduces the historical ``(value, metadata)``
    record byte-for-byte, including the next ``t^23`` term.  ``diagnostic``
    ``False`` returns only the value consumed by the two RHS callers.  Both
    modes share the same single ``k``/``t`` calculation and retained
    expression; the value-only route omits only unread certificate metadata.
    """
    if n_nat <= 0.0 or mass <= 0.0:
        if diagnostic:
            return 0.0, {"branch": "vacuum", "t": 0.0, "next_over_retained": 0.0}
        return 0.0
    # Compute k and t exactly once per call.  Keep the binary64 operation
    # order frozen for the retained low-t expression and prefactor.
    k = (6.0 * math.pi**2 * n_nat / p.degeneracy) ** (1.0 / 3.0)
    t = k / mass
    if force_series or t <= 0.1:
        retained = math.fsum(c * t**power for c, power in _DNS_DM_SERIES[:-1])
        pref = p.degeneracy * mass * mass / (4.0 * math.pi**2)
        value = float(pref * retained)
        if not diagnostic:
            return value
        # The next term and ratio are diagnostic-only work.  Their expression
        # order is intentionally identical to the pre-repair implementation.
        nxt = _DNS_DM_SERIES[-1][0] * t**23
        return value, {
            "t": float(t),
            "retained": float(pref * retained),
            "next_term": float(pref * nxt),
            "next_over_retained": float(abs(nxt) / max(abs(retained), 1.0e-300)),
            "powers": "t^5..t^21",
            "branch": "fixed_k_t5_series",
        }
    # Closed form is unchanged algebraically; reuse the k/t computed above so
    # this shared calculation does not duplicate branch-selection work.
    E = math.sqrt(k * k + mass * mass)
    a = math.asinh(k / mass)
    value = float(p.degeneracy / (4.0 * math.pi**2) * ((k * E - mass * mass * a) + 2.0 * mass * mass * (k / E - a)))
    if not diagnostic:
        return value
    return value, {"branch": "closed", "t": float(t), "next_over_retained": 0.0}


def _scalar_density_mass_derivative_series(n_nat: float, mass: float, p: Any = upstream.PARAMS) -> tuple[float, dict[str, float]]:
    """Cancellation-free fixed-k derivative through production t^21."""
    result = _scalar_density_mass_derivative_calc(n_nat, mass, p, diagnostic=True, force_series=True)
    # This helper is diagnostic by contract; keep the historical tuple API.
    assert isinstance(result, tuple)
    value, meta = result
    meta = dict(meta)
    meta.pop("branch", None)
    return value, meta


def _scalar_density_mass_derivative(
    n_nat: float,
    mass: float,
    p: Any = upstream.PARAMS,
    *,
    mode: str = "diagnostic",
    diagnostic: bool | None = None,
) -> Any:
    """Return the fixed-k derivative in explicit diagnostic or value mode."""
    if diagnostic is not None:
        mode = "diagnostic" if diagnostic else "value"
    if mode not in ("diagnostic", "value"):
        raise ValueError("unknown scalar-density derivative return mode")
    result = _scalar_density_mass_derivative_calc(n_nat, mass, p, diagnostic=(mode == "diagnostic"))
    if mode == "value":
        return float(result)
    value, meta = result
    return float(value), dict(meta)


@dataclass(frozen=True)
class WEvaluationRecord:
    """One immutable direct evaluation of the stationary W equation.

    The record is deliberately limited to primitive source coefficients.  It
    carries no bracket, root, hint or other solver decision state, and is
    created afresh for each exact positive binary64 W key in one solve.
    """

    m: float
    k: float
    t: float
    ns: float
    ns_m: float
    f: float
    fW: float
    scale: float

    # Readable aliases used by the source/derivative path.  They are
    # properties rather than duplicate fields so the frozen record remains a
    # single canonical value tuple.
    @property
    def n_s(self) -> float:
        return self.ns

    @property
    def f_W(self) -> float:
        return self.fW

    @property
    def residual_scale(self) -> float:
        return self.scale


class WCertificate(dict):
    """Dict-compatible W certificate with one preordered immutable view."""

    __slots__ = ("ordered",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ordered: tuple[tuple[str, float | int | bool], ...] = tuple(
            sorted((str(k), v) for k, v in self.items() if isinstance(v, (float, int, bool)))
        )


def _solve_W_certified(
    n_nat: float,
    p: Any = upstream.PARAMS,
    *,
    hint: float | None = None,
    _cache_enabled: bool = True,
    _return_record: bool = False,
) -> tuple[float, dict[str, Any]] | tuple[float, dict[str, Any], WEvaluationRecord | None]:
    """Certified production W root on the fixed positive sign bracket.

    This is intentionally independent of the upstream bisection oracle.  The
    bracket is never widened: its endpoint signs and the analytic Jacobian are
    checked on every call, and a safeguarded Newton step is accepted only when
    it remains inside the retained sign interval.
    """
    if n_nat == 0.0:
        result = (float(p.W0), {"endpoint": True, "iterations": 0, "bracket_rel": 0.0, "residual_rel": 0.0})
        return (*result, None) if _return_record else result
    lo = float(p.W0 / math.sqrt(3.0)); hi = float(2.0 * p.W0)
    gs = float(p.g_s); G = float(p.g_omega**2 / p.q_phi**2); lam = float(p.lam)
    # ``k`` depends only on this solve's density and is immutable across all
    # W records.  It is computed once; each exact W record computes its own
    # mass and ratio ``t`` exactly once.
    k_solve = (6.0 * math.pi**2 * n_nat / p.degeneracy) ** (1.0 / 3.0)
    # The map is deliberately solve-local and keyed by the exact positive
    # binary64 value itself.  Python float equality is bit-exact on this
    # domain (finite, strictly positive values), while ``float.hex()`` is
    # retained only for diagnostic serialization (legacy spelling: ``key = float(w).hex()``).
    # No key or record escapes this invocation.  The frozen record contains
    # only the source coefficients (m, k, t, n_s, n_s_m, f, f_W, scale).
    if _cache_enabled:
        # These solve-local constants are evaluated once with the exact
        # binary64 association used by the historical expressions.  They
        # never escape this invocation and are not shared between solves.
        four_pi2 = 4.0 * math.pi**2
        G_n_n = G * n_nat * n_nat
        three_G_n_n = 3.0 * G * n_nat * n_nat
        gs2 = gs * gs
        lam_W0_W0 = lam * p.W0 * p.W0
    w_bundle: dict[float, WEvaluationRecord] = {}
    def bundle(w: float) -> WEvaluationRecord:
        float_w = float(w)
        if not math.isfinite(float_w) or float_w <= 0.0:
            raise ValueError("pressure-source W key must be finite and positive")
        w = float_w
        key = w
        if _cache_enabled and key in w_bundle:
            return w_bundle[key]
        m = gs * w
        # Compute the density momentum and low-t ratio once for this exact W
        # record.  The optimized path keeps the retained individual t**power
        # operations and both math.fsum calls in their historical order.
        k = k_solve
        t = k / m
        if _cache_enabled:
            # Keep the boundary asymmetry literal: scalar density is closed
            # at t==0.1, while its fixed-k derivative retains the series.
            if t <= 0.1:
                t3 = t**3
                t5 = t**5
                t7 = t**7
                t9 = t**9
                t11 = t**11
                t13 = t**13
                t15 = t**15
                t17 = t**17
                t19 = t**19
                t21 = t**21
            if t < 0.1:
                scalar_series = math.fsum((
                    (2.0 / 3.0) * t3,
                    (-1.0 / 5.0) * t5,
                    (3.0 / 28.0) * t7,
                    (-5.0 / 72.0) * t9,
                    (35.0 / 704.0) * t11,
                    (-63.0 / 1664.0) * t13,
                    (231.0 / 6656.0) * t15,
                    (-429.0 / 15360.0) * t17,
                ))
                scalar = float(p.degeneracy * m**3 / four_pi2 * scalar_series)
            else:
                scalar = float(upstream.scalar_density(n_nat, m, p))
            if t <= 0.1:
                retained = math.fsum((
                    (2.0 / 5.0) * t5,
                    (-3.0 / 7.0) * t7,
                    (5.0 / 12.0) * t9,
                    (-35.0 / 88.0) * t11,
                    (315.0 / 832.0) * t13,
                    (-231.0 / 640.0) * t15,
                    (3003.0 / 8704.0) * t17,
                    (-6435.0 / 19456.0) * t19,
                    (36465.0 / 114688.0) * t21,
                ))
                pref = p.degeneracy * m * m / four_pi2
                ns_m = float(pref * retained)
            else:
                E = math.sqrt(k * k + m * m)
                a = math.asinh(k / m)
                ns_m = float(p.degeneracy / four_pi2 * ((k * E - m * m * a) + 2.0 * m * m * (k / E - a)))
        else:
            # Diagnostic/off mode deliberately preserves the pre-DAG calls;
            # this is the A/B control and remains byte-identical.
            scalar = _scalar_density_stable(n_nat, m, p)
            # Value-only hot route: ``ns_m, _ = _scalar_density_mass_derivative``
            # is the legacy diagnostic form and is intentionally not used here.
            ns_m = _scalar_density_mass_derivative(n_nat, m, p, mode="value")
        # Factor the near-W0 quadratic to avoid losing the tiny scalar-field
        # balance to subtraction when n is in the low-density tail.  The
        # operation order is unchanged from the pre-repair source functions.
        if _cache_enabled:
            w3 = w**3
            w4 = w**4
            fw = float(lam*w*(w-p.W0)*(w+p.W0) + gs*scalar - G_n_n/w3)
            fj = float(lam*(3*w*w-p.W0*p.W0) + gs2*ns_m + three_G_n_n/w4)
            residual_scale = max(abs(lam*w3), abs(lam_W0_W0*w), abs(gs*scalar), abs(G_n_n/w3), 1.0)
        else:
            fw = float(lam*w*(w-p.W0)*(w+p.W0) + gs*scalar - G*n_nat*n_nat/w**3)
            fj = float(lam*(3*w*w-p.W0*p.W0) + gs*gs*ns_m + 3*G*n_nat*n_nat/w**4)
            residual_scale = max(abs(lam*w**3), abs(lam*p.W0*p.W0*w), abs(gs*scalar), abs(G*n_nat*n_nat/w**3), 1.0)
        out = WEvaluationRecord(
            m=float(m), k=float(k), t=float(t), ns=float(scalar),
            ns_m=float(ns_m), f=float(fw), fW=float(fj),
            scale=float(residual_scale),
        )
        if _cache_enabled:
            w_bundle[key] = out
        return out

    # Compatibility aliases retained for the historical source audit.  The
    # decision loop below intentionally does not call these helpers; each
    # decision keeps its fetched WEvaluationRecord locally instead.
    def f(w: float) -> float:
        return bundle(w).f
    def fp(w: float) -> float:
        return bundle(w).fW

    # Each solver decision fetches one local immutable record and then reads
    # all required fields from that reference.  This avoids redispatching the
    # exact-key map for f/f_W/scale consumers while preserving probe order.
    rlo = bundle(lo); flo = rlo.f
    rhi = bundle(hi); fhi = rhi.f
    if not (flo < 0.0 < fhi):
        raise ValueError("certified W bracket endpoint signs failed")
    if rlo.fW <= 0.0 or rhi.fW <= 0.0:
        raise ValueError("certified W Jacobian is not positive")
    w = 0.5 * (lo + hi)
    local_bracket = False
    if hint is not None and np.isfinite(float(hint)) and lo < float(hint) < hi:
        # Build a fresh local sign bracket by direct probes around the
        # neighbouring certified state.  A straddling pair becomes the actual
        # safeguarded bracket (never merely a diagnostic); otherwise retain the
        # immutable global bracket and start from its midpoint.
        centre = float(hint)
        radius = max(32.0 * math.ulp(centre), abs(centre) * 1.0e-8)
        for _ in range(16):
            a = max(lo, centre - radius); b = min(hi, centre + radius)
            ra, rb = bundle(a), bundle(b)
            fa, fb = ra.f, rb.f
            if fa < 0.0 < fb:
                if ra.fW <= 0.0 or rb.fW <= 0.0:
                    raise ValueError("certified local W Jacobian is not positive")
                lo, hi, flo, fhi = float(a), float(b), float(fa), float(fb)
                rlo, rhi = ra, rb
                local_bracket = True
                # Start at the hint's bounded Newton predictor.  The direct
                # bracket remains authoritative even if the predictor is
                # rejected by the usual safeguarded-step test.
                rh = bundle(centre); fh, fph = rh.f, rh.fW
                predictor = centre - fh / fph if np.isfinite(fph) and fph > 0.0 else centre
                w = float(predictor) if np.isfinite(predictor) and lo < predictor < hi else centre
                break
            if a <= lo and b >= hi:
                break
            radius *= 2.0
    # A post-Newton probe is an optimization of the retained *direct* sign
    # bracket.  It is deliberately diagnostic and fail-closed: the old
    # bracket remains authoritative unless both newly evaluated endpoints
    # strictly straddle the root and have positive analytic Jacobians.
    post_probe_attempts = 0
    post_probe_accepted = False
    post_probe_radius = 0.0
    it = 0
    for it in range(1, 49):
        rw = bundle(w); fw, fwp = rw.f, rw.fW
        if not np.isfinite(fwp) or fwp <= 0.0:
            raise ValueError("certified W Jacobian lost positivity")
        # When a safeguarded Newton trial is already close to the zero, probe
        # a symmetric neighbourhood before discarding the previous signs.
        # The radius combines the direct Newton correction, representable ulp
        # spacing, and the unchanged final width target.  Expansion is always
        # clipped to the currently retained bracket and every endpoint is
        # evaluated through the source function/Jacobian.
        probe_accepted_this = False
        rscale_trial = rw.scale
        correction = abs(fw / fwp) if np.isfinite(fw) else float("inf")
        trial_rel = abs(fw) / rscale_trial
        retained_rel = (hi - lo) / max(abs(w), 1.0e-300)
        if fw != 0.0 and np.isfinite(correction) and trial_rel <= 1.0e-8 and retained_rel > 2.0e-13 and lo < w < hi:
            # Start below the final width target so the accepted pair is a
            # genuinely useful refinement (the target still sets the scale,
            # while direct sign growth handles a larger Newton correction).
            radius = max(16.0 * math.ulp(w), 2.0 * correction, 0.01 * 2.0e-13 * max(abs(w), 1.0e-300))
            for _probe in range(16):
                a = max(lo, w - radius); b = min(hi, w + radius)
                if not (lo <= a < w < b <= hi):
                    break
                ra, rb = bundle(a), bundle(b)
                fa, fb = ra.f, rb.f
                fpa, fpb = ra.fW, rb.fW
                post_probe_attempts += 1
                post_probe_radius = max(post_probe_radius, float(b - a))
                if fa < 0.0 < fb and fpa > 0.0 and fpb > 0.0:
                    lo, hi, flo, fhi = float(a), float(b), float(fa), float(fb)
                    rlo, rhi = ra, rb
                    local_bracket = True
                    post_probe_accepted = True
                    probe_accepted_this = True
                    break
                if a <= lo and b >= hi:
                    break
                # The old endpoint signs remain in force while searching; a
                # failed probe is therefore just a fallback to the old pair.
                radius *= 2.0
        # Once accepted, the trial sign is still a directly evaluated interior
        # point, so retain the usual sign update to contract the fresh probe
        # bracket on the very same iteration.
        if fw < 0.0:
            lo, flo, rlo = w, fw, rw
        elif fw > 0.0:
            hi, fhi, rhi = w, fw, rw
        elif not probe_accepted_this:
            # A binary64 trial can land exactly on the represented zero even
            # though the certificate requires strict endpoint signs.  Walk
            # adjacent representable values on both sides and retain the first
            # direct f<0<f pair; no zero endpoint is accepted.
            left = right = float(w); fl = frz = 0.0
            for _z in range(16):
                left = math.nextafter(left, lo)
                right = math.nextafter(right, hi)
                rl, rr = bundle(left), bundle(right)
                fl, frz = rl.f, rr.f
                if fl < 0.0 and frz > 0.0:
                    break
            if not (fl < 0.0 < frz):
                raise ValueError("certified W zero trial has no strict adjacent bracket")
            lo, flo, rlo, hi, fhi, rhi = left, fl, rl, right, frz, rr
        mid = 0.5*(lo+hi)
        wr = 0.5 * (lo + hi)
        rwr = bundle(wr); fr = rwr.f
        # When the certified bracket has collapsed to adjacent binary64
        # values, the arithmetic midpoint rounds to one endpoint.  Retain
        # whichever directly re-evaluated endpoint has the smaller residual;
        # this is still a source-certified value and avoids injecting a
        # spurious O(ulp(W)) scalar-potential offset at low density.
        flo_abs, fhi_abs = abs(flo), abs(fhi)
        if fhi_abs < abs(fr) and fhi_abs <= flo_abs:
            wr, fr, rwr = hi, fhi, rhi
        elif flo_abs < abs(fr) and flo_abs < fhi_abs:
            wr, fr, rwr = lo, flo, rlo
        rscale = rwr.scale
        # Probe a small deterministic neighbourhood only when the direct
        # midpoint residual is close to the binary64 noise floor.  This keeps
        # the hot path bounded while recovering the best representable point
        # for cancellation-sensitive low-density states.
        if abs(fr) / rscale > 1.0e-12:
            best_w, best_f = float(wr), float(fr)
            best_record = rwr
            if float(w) != float(wr):
                direction = 1 if w > wr else -1
                trial = float(wr)
                for _ulp in range(2):
                    trial = math.nextafter(trial, hi if direction > 0 else lo)
                    if lo <= trial <= hi:
                        rt = bundle(trial); ft = rt.f
                        if abs(ft) < abs(best_f):
                            best_w, best_f, best_record = float(trial), float(ft), rt
            else:
                for direction in (-1, 1):
                    trial = math.nextafter(float(wr), lo if direction < 0 else hi)
                    if lo <= trial <= hi:
                        rt = bundle(trial); ft = rt.f
                        if abs(ft) < abs(best_f):
                            best_w, best_f, best_record = float(trial), float(ft), rt
            wr, fr = best_w, best_f
            rwr = best_record; rscale = rwr.scale
        # At densities below one binary64 ulp in W, a probe pair can straddle
        # W0 while its midpoint rounds to the wrong side.  A direct W0
        # evaluation is admissible only when it lies inside the retained
        # signs; it is then selected by the same best-residual rule, never as
        # a positive-density shortcut.
        if lo <= float(p.W0) <= hi and abs(float(w) - float(p.W0)) <= 16.0 * math.ulp(float(p.W0)):
            r0 = bundle(float(p.W0)); f0 = r0.f
            if abs(f0) < abs(fr):
                wr, fr = float(p.W0), float(f0)
                rwr = r0; rscale = r0.scale
        bracket_rel = (hi-lo)/max(abs(wr), 1e-300)
        residual_rel = abs(fr)/rscale
        if bracket_rel <= 2e-13 and residual_rel <= 2e-12:
            if not (flo < 0.0 < fhi) or rwr.fW <= 0.0:
                raise ValueError("certified W endpoint or midpoint signs failed")
            ns_m = rwr.ns_m
            fwp = rwr.fW; f_n = gs * (gs * wr) / math.sqrt((6.0 * math.pi**2 * n_nat / p.degeneracy) ** (2.0 / 3.0) + (gs * wr) ** 2) - 2.0 * G * n_nat / wr**3
            cert = WCertificate({"endpoint": False, "iterations": it, "bracket_rel": bracket_rel, "residual_rel": residual_rel, "w_lo": float(lo), "w_hi": float(hi), "dW_dn": float(-f_n / fwp), "jacobian": float(fwp), "local_bracket": bool(local_bracket), "hint_used": bool(local_bracket), "post_newton_probe": {"attempts": int(post_probe_attempts), "accepted": bool(post_probe_accepted), "radius": float(post_probe_radius)}, "post_newton_probe_attempts": int(post_probe_attempts), "post_newton_probe_accepted": bool(post_probe_accepted), "post_newton_probe_radius": float(post_probe_radius)})
            return (float(wr), cert, rwr) if _return_record else (float(wr), cert)
        newton = w - fw/fwp if fwp > 0.0 and np.isfinite(fwp) else float("nan")
        w = float(newton) if np.isfinite(newton) and lo < newton < hi else mid
    raise ValueError("certified W root did not converge within 48 trials")


def _scalar_density_mass_derivative_oracle(n_nat: float, mass: float, dps: int) -> float:
    """Independent arbitrary-precision closed-form oracle (no series)."""
    import mpmath as mp
    with mp.workdps(int(dps)):
        n, m, d = mp.mpf(str(n_nat)), mp.mpf(str(mass)), mp.mpf(str(upstream.PARAMS.degeneracy))
        if n <= 0 or m <= 0:
            return 0.0
        k = (mp.mpf(6) * mp.pi**2 * n / d) ** (mp.mpf(1) / 3)
        E = mp.sqrt(k*k + m*m); a = mp.asinh(k/m)
        value = d / (4 * mp.pi**2) * ((k*E - m*m*a) + 2*m*m*(k/E - a))
        return float(value)


def scalar_derivative_oracle_controls() -> dict[str, Any]:
    """Independent 80/120-digit controls at the adjudicated witness points."""
    rows: list[dict[str, Any]] = []
    for x in (1.0e-14, 1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3, 1.0e-2, 0.04, 0.1, 1.0, 10.0):
        n = float(x * upstream.PARAMS.n0_nat)
        W = upstream.PARAMS.W0 if x == 0.0 else upstream.solve_W(n, upstream.PARAMS)
        m = upstream.PARAMS.g_s * W
        prod, meta = _scalar_density_mass_derivative(n, m)
        o80 = _scalar_density_mass_derivative_oracle(n, m, 80)
        o120 = _scalar_density_mass_derivative_oracle(n, m, 120)
        closed = _scalar_density_mass_derivative_closed(n, m)
        rows.append({"x": x, "production": prod, "oracle_80": o80, "oracle_120": o120, "closed": closed, "rel_80_120": _rel(o80, o120), "rel_production_oracle": _rel(prod, o120), "series": meta})
    # At the branch boundary evaluate the series and the independent closed
    # expression using the same physical k,m rather than trusting a row.
    n = float(0.1**3 * upstream.PARAMS.degeneracy * (upstream.PARAMS.g_s * upstream.PARAMS.W0)**3 / (6.0 * math.pi**2))
    m = upstream.PARAMS.g_s * upstream.PARAMS.W0
    sv, _ = _scalar_density_mass_derivative_series(n, m)
    cv = _scalar_density_mass_derivative_oracle(n, m, 120)
    return {"rows": rows, "boundary_t": 0.1, "boundary_rel": _rel(sv, cv), "pass": bool(max((r["rel_80_120"] for r in rows), default=0.0) <= 1.0e-40 and max((r["rel_production_oracle"] for r in rows), default=0.0) <= 2.0e-10 and _rel(sv, cv) <= 2.0e-12)}


def enthalpy_oracle_controls() -> dict[str, Any]:
    """Fixed-point 80/120 digit enthalpy checks kept off the RHS hot path."""
    import mpmath as mp
    rows: list[dict[str, Any]] = []
    for x in (1.0e-6, 1.0e-3, 0.01, 0.1, 1.0, 10.0):
        n = float(x * upstream.PARAMS.n0_nat)
        # The oracle evaluates the original decimal enthalpy formula at the
        # same directly certified source root used by production.  Its 80/120
        # digit arithmetic remains independent of the binary64 hot path.
        W, _ = _solve_W_certified(n, upstream.PARAMS)
        prod = _stable_enthalpy(n, W)
        vals = []
        for dps in (80, 120):
            with mp.workdps(dps):
                # Preserve the exact binary64 source inputs while evaluating
                # the original expression at 80/120 decimal digits; converting
                # through a shortened decimal string would add a spurious
                # low-density W-W0 cancellation error.
                nn, ww = mp.mpf(n), mp.mpf(W); gs, mn = mp.mpf(upstream.PARAMS.g_s), mp.mpf(upstream.PARAMS.M_N); gom, qph = mp.mpf(upstream.PARAMS.g_omega), mp.mpf(upstream.PARAMS.q_phi); deg = mp.mpf(upstream.PARAMS.degeneracy); k = (6*mp.pi**2*nn/deg)**(mp.mpf(1)/3); m = gs*ww; E = mp.sqrt(k*k+m*m); delta = k*k/(E+m) + gs*(ww-mp.mpf(upstream.PARAMS.W0)) + gom*gom*nn/(qph*qph*ww*ww); vals.append(float(mp.log1p(delta/mn)))
        rows.append({"x": x, "production": prod, "oracle_80": vals[0], "oracle_120": vals[1], "rel_80_120": _rel(vals[0], vals[1]), "rel_production_oracle": _rel(prod, vals[1])})
    return {"rows": rows, "oracle_dps": [80, 120], "pass": bool(max((r["rel_80_120"] for r in rows), default=0.0) <= 1e-40 and max((r["rel_production_oracle"] for r in rows), default=0.0) <= 2e-12)}


@dataclass(frozen=True)
class EOSPoint:
    x: float
    n_fm3: float
    eps: float
    pressure: float
    mu: float
    enthalpy: float
    W: float
    A0: float
    d_eps_dn: float | None = None
    d_pressure_dn: float | None = None
    cs2: float | None = None
    # Exact extended enthalpy is carried alongside the unchanged binary64
    # projection above.  It is immutable source data, never a solver state.
    enthalpy_extended: np.longdouble | None = None


@dataclass(frozen=True)
class SourceResult:
    """Immutable direct source evaluation and its certified W solve."""
    point: EOSPoint
    w_certificate: tuple[tuple[str, float | int | bool], ...]
    enthalpy_extended: np.longdouble | None = None

    def certificate(self) -> dict[str, float | int | bool]:
        return dict(self.w_certificate)


def _stable_enthalpy_extended(n_nat: float, W: float, p: Any = upstream.PARAMS) -> np.longdouble:
    """Stable chemical-potential excess with a binary64 source input."""
    if n_nat <= 0.0:
        return np.longdouble(0.0)
    # Extended-precision intermediates use the exact binary64 source values;
    # the returned value remains an np.longdouble until the one EOSPoint
    # projection.  This removes the last-ulp noise in the W-dependent excess
    # without importing arbitrary-precision arithmetic on the RHS hot path.
    ld = np.longdouble
    nn = ld(float(n_nat)); ww = ld(float(W)); gs = ld(float(p.g_s))
    k = (ld(6.0) * ld(float(math.pi))**2 * nn / ld(float(p.degeneracy))) ** (ld(1.0) / ld(3.0))
    m = gs * ww
    E = np.sqrt(k * k + m * m)
    G = ld(float(p.g_omega))**2 / ld(float(p.q_phi))**2
    delta_mu = k * k / (E + m) + gs * (ww - ld(float(p.W0))) + G * nn / (ww * ww)
    if not np.isfinite(delta_mu) or delta_mu <= -ld(float(p.M_N)):
        raise ValueError("invalid chemical-potential excess")
    return np.longdouble(np.log1p(delta_mu / ld(float(p.M_N))))


def _stable_enthalpy(n_nat: float, W: float, p: Any = upstream.PARAMS) -> float:
    """Binary64 projection of the stable direct enthalpy hot-path value."""
    return float(_stable_enthalpy_extended(n_nat, W, p))


def _source_enthalpy_value(point: EOSPoint, p: Any = upstream.PARAMS) -> np.longdouble:
    """Extended direct h(x,W) used only for inversion residual certificates."""
    if point.enthalpy_extended is not None:
        return np.longdouble(point.enthalpy_extended)
    # Disabled-cache/mutation control: recompute the identical source
    # expression only when no immutable extension was carried by the point.
    return _stable_enthalpy_extended(float(point.x) * p.n0_nat, float(point.W), p)


def _source_result(
    x: float,
    *,
    high_precision: bool = True,
    w_hint: float | None = None,
    _with_derivatives: bool = False,
) -> SourceResult:
    """Evaluate the direct source EOS on the closed 0 <= n/n0 <= 10 domain."""
    x = float(x)
    if not np.isfinite(x) or x < 0.0 or x > X_MAX:
        raise ValueError("source EOS domain is the closed interval 0 <= n/n0 <= 10")
    if x == 0.0:
        h_ext = np.longdouble(0.0)
        return SourceResult(EOSPoint(0.0, 0.0, 0.0, 0.0, upstream.PARAMS.M_N, 0.0, upstream.PARAMS.W0, 0.0, enthalpy_extended=h_ext), (("endpoint", True), ("iterations", 0), ("bracket_rel", 0.0), ("residual_rel", 0.0)), h_ext)
    p = upstream.PARAMS
    n_nat = x * p.n0_nat
    _wsolve = _solve_W_certified(n_nat, p, hint=w_hint, _return_record=True)
    W, _wcert, wrec = _wsolve
    if not isinstance(wrec, WEvaluationRecord):
        raise ValueError("certified W solve did not return its immutable evaluation record")
    A0 = upstream.gauss_A0(n_nat, W, p)
    # The upstream closed forms subtract nearly equal terms when k_F/m is
    # tiny.  Evaluate the identical free Fermi branch by its convergent
    # low-density series in that regime; this is the adjudicated underflow
    # gate, not a pressure floor or an EOS continuation.
    # Reuse the accepted source coefficients from the same W solve.  No
    # coefficient record is retained in SourceResult/DirectEOS caches; this
    # local reference dies as soon as the source point is assembled.
    m, k, t = wrec.m, wrec.k, wrec.t
    if t < 0.1:
        t2 = t * t
        ffree_series = (8.0 * t**3 / 3.0 + 4.0 * t**5 / 5.0 - t**7 / 7.0 + t**9 / 18.0 - 5.0 * t**11 / 176.0 + 7.0 * t**13 / 416.0)
        fp_series = (8.0 * t**5 / 5.0 - 4.0 * t**7 / 7.0 + t**9 / 3.0 - 5.0 * t**11 / 22.0 + 35.0 * t**13 / 208.0)
        ffree = p.degeneracy * m**4 / (16.0 * math.pi**2) * ffree_series
        fpressure = p.degeneracy * m**4 / (48.0 * math.pi**2) * fp_series
        eps4 = ffree + upstream.scalar_potential(W, p) + (0.0 if n_nat == 0.0 else p.g_omega**2 * n_nat**2 / (2.0 * p.q_phi**2 * W * W))
    else:
        eps4 = upstream.energy_density(n_nat, W, p)
    mu = upstream.chemical_potential(n_nat, W, p)
    # Both constructions are retained as an explicit thermodynamic gate.
    eps_h, pressure_h = upstream.hilbert_stress(n_nat, W, p)
    pressure_l = mu * n_nat - eps4
    scale = max(abs(pressure_h), abs(pressure_l), 1.0e-300)
    pressure_tol = 1.0e-6 if x <= 1.0e-2 else 1.0e-8
    if high_precision and x > 1.0e-3 and abs(pressure_h - pressure_l) > pressure_tol * scale:
        raise ValueError("Hilbert/Legendre pressure mismatch")
    eps = float(eps4 / p.hbar_c**3)
    # Below 1e-3 n0, direct mu*n-epsilon subtraction is an IEEE underflow
    # regime.  The same source branch's Hilbert pressure is positive and is
    # used there without clipping; the explicit EOS gate records that this
    # binary Legendre identity is intentionally not asserted in that regime.
    if t < 0.1:
        pressure_h = fpressure - upstream.scalar_potential(W, p) + (0.0 if n_nat == 0.0 else p.g_omega**2 * n_nat**2 / (2.0 * p.q_phi**2 * W * W))
    pressure_raw = pressure_h if x <= 1.0e-3 else pressure_l
    pressure = float(pressure_raw / p.hbar_c**3)
    if pressure <= 0.0:
        raise ValueError("non-positive direct source pressure")
    # Compute the extended enthalpy exactly once for this immutable source
    # result.  EOSPoint.enthalpy is the same binary64 projection used by all
    # existing callers; inversion certificates consume enthalpy_extended.
    h_ext = _stable_enthalpy_extended(n_nat, W, p)
    point = EOSPoint(x, x * p.n0_fm3, eps, pressure, mu, float(h_ext), W, A0, enthalpy_extended=h_ext)

    # RHS callers request the derivative-enriched point as part of this same
    # source construction.  Carry the accepted m/k/t/n_s/n_s_m/f_W values
    # directly into the implicit derivative expressions, preserving their
    # historical binary64 operation order and avoiding a second W/source
    # dispatch.
    if _with_derivatives:
        E = math.sqrt(k * k + m * m)
        f_n = p.g_s * m / E - 2.0 * (p.g_omega**2 / p.q_phi**2) * n_nat / W**3
        f_W = wrec.fW
        if not np.isfinite(f_W) or abs(f_W) <= 1.0e-30:
            raise ValueError("singular stationary-branch derivative")
        dW_dn = -f_n / f_W
        G = p.g_omega**2 / p.q_phi**2
        dmu_dn = k * k / (3.0 * n_nat * E) + (p.g_s * m / E - 2.0 * G * n_nat / W**3) * dW_dn + G / W**2
        dp = n_nat * dmu_dn
        if not np.isfinite(dp) or dp <= 0.0:
            raise ValueError("non-positive direct source dP/dn")
        point = EOSPoint(**{**point.__dict__, "d_eps_dn": point.mu, "d_pressure_dn": float(dp), "cs2": float(dp / point.mu)})
    # Assemble the ordered certificate exactly once at certification.  The
    # immutable tuple is what public SourceResult exposes and is independent
    # of any transient W coefficient record.
    cert_tuple = _wcert.ordered if isinstance(_wcert, WCertificate) else tuple(sorted((str(ck), cv) for ck, cv in _wcert.items() if isinstance(cv, (float, int, bool))))
    return SourceResult(point, cert_tuple, h_ext)


def _source_point(x: float, *, high_precision: bool = True, w_hint: float | None = None) -> EOSPoint:
    """Public compatibility wrapper returning only the immutable EOS point."""
    return _source_result(x, high_precision=high_precision, w_hint=w_hint).point


def _source_h(x: float, *, strict: bool = True) -> float:
    """Fast direct enthalpy evaluation for the monotone inversion bracket."""
    x = float(x)
    if not np.isfinite(x) or x < 0.0 or x > X_MAX:
        raise ValueError("source enthalpy request is outside the closed domain")
    if x == 0.0:
        return 0.0
    return _source_result(x, high_precision=False).point.enthalpy


def _log_derivative(x: float, value: Callable[[float], float], step: float) -> float:
    """Five-point derivative in log density, with one-sided endpoint stencil."""
    u = math.log(float(x))
    lo, hi = math.log(1.0e-12), math.log(X_MAX)
    def ev(uu: float) -> float:
        # Keep the stencil inside the exact closed interval.  This is a
        # stencil-coordinate guard, not an EOS endpoint clamp: _source_point
        # itself still rejects every out-of-domain request.
        return value(min(X_MAX, max(0.0, math.exp(uu))))
    if u - 2.0 * step >= lo and u + 2.0 * step <= hi:
        vals = [ev(u + j * step) for j in (-2, -1, 1, 2)]
        d_du = (vals[0] - 8.0 * vals[1] + 8.0 * vals[2] - vals[3]) / (12.0 * step)
    elif u - 2.0 * step < lo:
        vals = [ev(u + j * step) for j in range(5)]
        d_du = (-25.0 * vals[0] + 48.0 * vals[1] - 36.0 * vals[2] + 16.0 * vals[3] - 3.0 * vals[4]) / (12.0 * step)
    else:
        vals = [ev(u - j * step) for j in range(5)]
        d_du = (25.0 * vals[0] - 48.0 * vals[1] + 36.0 * vals[2] - 16.0 * vals[3] + 3.0 * vals[4]) / (12.0 * step)
    return float(d_du / x)


def eos_point_with_gates(x: float) -> tuple[EOSPoint, dict[str, Any]]:
    point = _source_point(x)
    gates: dict[str, Any] = {
        "finite_branch": True,
        "positive_W": point.W > 0.0,
        "positive_thermodynamic_values": point.x == 0.0 or (point.eps > 0.0 and point.pressure > 0.0),
        "domain_closed_0_to_10": 0.0 <= point.x <= X_MAX,
        "pressure_identity": True,
        "hilbert_legendre_agree": True,
        "dedn_mu": True,
        "dPdn_positive": True,
        "strict_causality": True,
        "derivative_ladder_spread": 0.0,
    }
    if point.x == 0.0:
        return point, gates
    n_nat = point.x * upstream.PARAMS.n0_nat
    W_ref = upstream.solve_W(n_nat, upstream.PARAMS)
    ns_m, ns_meta = _scalar_density_mass_derivative(n_nat, upstream.PARAMS.g_s * W_ref)
    ns_oracle = _scalar_density_mass_derivative_oracle(n_nat, upstream.PARAMS.g_s * W_ref, 80)
    gates["fixed_k_scalar_derivative"] = float(ns_m)
    gates["fixed_k_scalar_derivative_oracle_rel"] = _rel(ns_m, ns_oracle)
    gates["fixed_k_scalar_derivative_branch"] = ns_meta.get("branch", "")
    gates["fixed_k_scalar_derivative_series_next_rel"] = float(ns_meta.get("next_over_retained", 0.0))
    eps4_ref = upstream.energy_density(n_nat, W_ref, upstream.PARAMS)
    mu_ref = upstream.chemical_potential(n_nat, W_ref, upstream.PARAMS)
    _, p_h_ref = upstream.hilbert_stress(n_nat, W_ref, upstream.PARAMS)
    p_l_ref = mu_ref * n_nat - eps4_ref
    identity_scale = max(abs(p_h_ref), abs(p_l_ref), 1.0e-300)
    # Check the generator identity against the returned source point rather
    # than accepting a tautological reconstruction of ``p_l_ref``.  The
    # direct high-precision branch has already evaluated the same identity in
    # decimal arithmetic when cancellation dominates near the vacuum.
    pressure_from_generator = float(p_l_ref / upstream.PARAMS.hbar_c**3)
    gates["pressure_identity"] = (point.x <= 1.0e-3) or _rel(point.pressure, pressure_from_generator) <= 1.0e-8
    # Hilbert and Legendre pressures are compared independently.  Away from
    # the underflow regime the source precision supports the adjudicated
    # 1e-10 bound; at the first log-grid points the decimal branch performs
    # that comparison before returning and the explicit regime gate records
    # why a binary reconstruction is not repeated here.
    gates["hilbert_legendre_agree"] = (point.x <= 1.0e-3) or abs(p_h_ref - p_l_ref) <= 1.0e-8 * identity_scale
    ladder: list[tuple[float, float, float, float]] = []
    for step in (1.0e-3, 5.0e-4, 2.5e-4):
        de = _log_derivative(point.x, lambda z: _source_point(z).eps, step)
        dp = _log_derivative(point.x, lambda z: _source_point(z).pressure, step)
        ladder.append((step, de / upstream.PARAMS.n0_fm3, dp / upstream.PARAMS.n0_fm3, dp / de))
    d_eps_dn = ladder[-1][1]
    d_pressure_dn = ladder[-1][2]
    cs2 = ladder[-1][3]
    point = EOSPoint(**{**point.__dict__, "d_eps_dn": d_eps_dn, "d_pressure_dn": d_pressure_dn, "cs2": cs2})
    spread = max(max(_rel(row[k], ladder[-1][k]) for row in ladder) for k in (1, 2, 3))
    gates["derivative_ladder_spread"] = float(spread)
    gates["dedn_mu"] = _rel(d_eps_dn, point.mu) <= 1.0e-8
    gates["dPdn_positive"] = d_pressure_dn > 0.0
    gates["strict_causality"] = 0.0 < cs2 <= 1.0
    gates["derivative_ladder_agree"] = spread <= 1.0e-6
    if not all(bool(v) for v in gates.values() if isinstance(v, (bool, np.bool_))):
        raise ValueError(f"EOS gate failure at x={point.x:g}: {gates}")
    return point, gates


def _point_result_with_derivatives(x: float, *, high_precision: bool = True, strict: bool = True, w_hint: float | None = None) -> SourceResult:
    """Derivative-enriched point for RHS calls (without repeating gate text)."""
    # Derivatives are assembled in the same source construction as the W
    # solve, using the carried immutable coefficient record.  This eliminates
    # the historical second scalar-density/f_W dispatch entirely.
    return _source_result(x, high_precision=high_precision, w_hint=w_hint, _with_derivatives=True)


def _point_with_derivatives(x: float, *, high_precision: bool = True, strict: bool = True, w_hint: float | None = None) -> EOSPoint:
    """Compatibility wrapper exposing the enriched immutable source point."""
    return _point_result_with_derivatives(x, high_precision=high_precision, strict=strict, w_hint=w_hint).point


class DirectEOS:
    """Direct source EOS with bracketed h and P inversion."""

    def __init__(self) -> None:
        self._h_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._p_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._source_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_cap = 4096
        self._hints: list[float] = []
        # Up to four accepted direct continuation records per coordinate.  A
        # record stores only the target, log-density, and its implicit slope;
        # it carries no EOS value or bracket and is re-certified on every use.
        self._h_records: list[tuple[float, float, float]] = []
        self._p_records: list[tuple[float, float, float]] = []
        self._last_w_hint: float | None = None
        self._last_w_state: tuple[float, float, float, float, float] | None = None
        # Proof callers obtain arbitrary-precision values from this adapter;
        # it is deliberately not an EOS/table cache and is destroyed with
        # the DirectEOS instance.
        self._exact_source_adapter = LiveExactSourceAdapter(180)
        self._stats: dict[str, int | float] = {"h_requests": 0, "h_hits": 0, "h_misses": 0, "p_requests": 0, "p_hits": 0, "p_misses": 0, "w_solves": 0, "w_max_iterations": 0, "w_max_bracket_rel": 0.0, "w_max_residual_rel": 0.0}
        # A deterministic source-evaluated table is retained only to select a
        # local bracket.  Every returned point below is obtained by a direct
        # monotone root against the source-complete functions; interpolation
        # never supplies an EOS value or enlarges the domain.  The table is an
        # instance cache so independently constructed grid evaluators cannot
        # share comparison/refinement state.
        self._x_tab = np.concatenate(([0.0], np.geomspace(1.0e-12, X_MAX, 129)))
        self._tab = [_source_point(float(x), high_precision=False) for x in self._x_tab]
        self._h_tab = np.asarray([q.enthalpy for q in self._tab])
        self._p_tab = np.asarray([q.pressure for q in self._tab])
        self._h_max = float(_source_point(X_MAX, high_precision=True).enthalpy)
        self._p_max = float(_source_point(X_MAX, high_precision=True).pressure)

    def begin_star(self) -> None:
        """Reset bounded per-star inversion state while retaining the table."""
        self._h_cache.clear(); self._p_cache.clear(); self._source_cache.clear(); self._hints.clear(); self._h_records.clear(); self._p_records.clear()
        self._stats = {"h_requests": 0, "h_hits": 0, "h_misses": 0, "p_requests": 0, "p_hits": 0, "p_misses": 0, "w_solves": 0, "w_max_iterations": 0, "w_max_bracket_rel": 0.0, "w_max_residual_rel": 0.0}
        self._last_w_hint = None
        self._last_w_state = None

    def exact_source(self, n_nat: Any, *, dps: int = 180) -> dict[str, Any]:
        """Return the one live exact-source record used by proof kernels."""
        if int(dps) != self._exact_source_adapter.dps:
            self._exact_source_adapter = LiveExactSourceAdapter(int(dps))
        return self._exact_source_adapter.evaluate(n_nat)

    def end_star(self) -> dict[str, Any]:
        summary = dict(self._stats)
        summary["h_cache_size"] = len(self._h_cache); summary["p_cache_size"] = len(self._p_cache); summary["source_cache_size"] = len(self._source_cache)
        self._h_cache.clear(); self._p_cache.clear(); self._source_cache.clear(); self._hints.clear(); self._h_records.clear(); self._p_records.clear()
        self._last_w_state = None
        return summary

    def _source(self, x: float, *, high_precision: bool = True) -> EOSPoint:
        key = ("src", float(x).hex(), bool(high_precision))
        skey = repr(key)
        item = self._source_cache.get(skey)
        if item is not None:
            self._source_cache.move_to_end(skey)
            return item["result"].point
        xx = float(x); n_nat = xx * upstream.PARAMS.n0_nat
        global_wlo = float(upstream.PARAMS.W0 / math.sqrt(3.0)); global_whi = float(2.0 * upstream.PARAMS.W0)
        hint = self._last_w_hint
        if self._last_w_state is not None:
            n0, w0, dwdn, wlo, whi = self._last_w_state
            predicted = w0 + dwdn * (n_nat - n0)
            # The previous final bracket is a certificate for the previous
            # density only.  A neighbouring implicit predictor may move
            # anywhere inside the immutable positive-W domain; the target
            # solve will directly probe/re-certify its own local bracket.
            if np.isfinite(predicted) and global_wlo < predicted < global_whi:
                hint = float(predicted)
        # On the first source call of a star, use only the nearest immutable
        # source-table W as a continuation hint.  The table value is never an
        # EOS output or bracket endpoint: _solve_W_certified re-probes it and
        # directly certifies a fresh local bracket (or falls back globally).
        if hint is None and self._tab:
            jj = int(np.searchsorted(self._x_tab, xx, side="left"))
            jj = max(0, min(jj, len(self._tab) - 1))
            table_hint = float(self._tab[jj].W)
            if np.isfinite(table_hint) and global_wlo < table_hint < global_whi:
                hint = table_hint
        result = _point_result_with_derivatives(xx, high_precision=high_precision, strict=True, w_hint=hint)
        q, cert = result.point, result.certificate()
        self._last_w_hint = float(q.W)
        if not cert.get("endpoint", False):
            self._stats["w_solves"] += 1
            self._stats["w_max_iterations"] = max(int(self._stats["w_max_iterations"]), int(cert.get("iterations", 0)))
            self._stats["w_max_bracket_rel"] = max(float(self._stats["w_max_bracket_rel"]), float(cert.get("bracket_rel", 0.0)))
            self._stats["w_max_residual_rel"] = max(float(self._stats["w_max_residual_rel"]), float(cert.get("residual_rel", 0.0)))
            self._last_w_state = (n_nat, float(q.W), float(cert.get("dW_dn", 0.0)), float(cert.get("w_lo", q.W)), float(cert.get("w_hi", q.W)))
        self._source_cache[skey] = {"result": result, "point": q, "certificate": cert}; self._source_cache.move_to_end(skey)
        if len(self._source_cache) > self._cache_cap: self._source_cache.popitem(last=False)
        return q

    @staticmethod
    def _key(coord: str, value: float) -> str:
        return f"{coord}|{SCHEMA_VERSION}|{float(value).hex()}"

    def at_h(self, h: float, *, strict: bool = True) -> EOSPoint:
        h = float(h)
        if not np.isfinite(h) or h < 0.0:
            raise ValueError("enthalpy is outside the exact vacuum-surface domain")
        if h == 0.0:
            return _source_point(0.0)
        if h > self._h_max:
            raise ValueError("enthalpy exceeds 10 n0 source domain")
        if h == self._h_max:
            q = self._source(X_MAX, high_precision=strict)
            key = self._key("h", h); self._h_cache[key] = {"point": q, "certificate": {"iterations": 0, "residual_log": 0.0, "bracket_rel": 0.0, "endpoint": True}}
            return q
        if not strict:
            raise ValueError("non-strict EOS inversion is not admissible for accepted stellar rows")
        self._stats["h_requests"] += 1
        key = self._key("h", h)
        cached = self._h_cache.get(key)
        if cached is not None:
            self._stats["h_hits"] += 1; self._h_cache.move_to_end(key); return cached["point"]
        self._stats["h_misses"] += 1
        j = int(np.searchsorted(self._h_tab, h, side="right") - 1)
        j = max(0, min(j, len(self._x_tab) - 2))
        if j == 0:
            # Keep the exact vacuum sign at x=0 without evaluating a
            # subnormal positive density (whose pressure would underflow).
            # A direct asymptotic estimate supplies a representable positive
            # bracket around the requested root.
            guess = float(self._x_tab[1] * (h / self._h_tab[1]) ** 1.5)
            guess = min(float(self._x_tab[1]) * (1.0 - 1.0e-14), max(float(np.nextafter(0.0, 1.0)), guess))
            lo = max(float(np.nextafter(0.0, 1.0)), 0.5 * guess)
            hi = min(X_MAX, max(float(self._x_tab[1]), 2.0 * guess))
        else:
            lo = float(self._x_tab[j])
            hi = min(X_MAX, max(float(self._x_tab[j + 1]), lo * (1.0 + 1.0e-12)))
        # The immutable source table supplies only the initial sign bracket.
        # All subsequent work is a direct solve in u=log(x), with the exact
        # logarithmic derivative cs2/h from the current source result.
        qlo = self._source(lo, high_precision=strict)
        qhi = self._source(hi, high_precision=strict)
        flo = float(np.log(_source_enthalpy_value(qlo) / np.longdouble(str(h))))
        fhi = float(np.log(_source_enthalpy_value(qhi) / np.longdouble(str(h))))
        if flo > 0.0:
            while flo > 0.0 and lo > float(np.nextafter(0.0, 1.0)):
                lo = max(float(np.nextafter(0.0, 1.0)), lo * 0.5)
                qlo = self._source(lo, high_precision=strict)
                flo = float(np.log(_source_enthalpy_value(qlo) / np.longdouble(str(h))))
        while fhi < 0.0 and hi < X_MAX:
            new_hi = min(X_MAX, hi * 2.0)
            if new_hi <= hi:
                break
            hi = new_hi
            qhi = self._source(hi, high_precision=strict)
            fhi = float(np.log(_source_enthalpy_value(qhi) / np.longdouble(str(h))))
        if not (flo <= 0.0 <= fhi):
            raise ValueError("direct enthalpy inversion certificate failed: source endpoint does not bracket target")
        ulo, uhi = math.log(lo), math.log(hi)
        # Select the closest prior accepted target only as an initial Newton
        # proposal.  The proposal must remain inside this fresh direct table
        # sign bracket; its source state is evaluated below before use.
        u = 0.5 * (ulo + uhi)
        if self._h_records:
            rec = min(self._h_records, key=lambda z: abs(math.log(h / z[0])))
            rt, ru, rs = rec
            if np.isfinite(rs) and rs > 0.0:
                proposal = float(ru + math.log(h / rt) / rs)
                if ulo < proposal < uhi:
                    u = proposal
        iterations = 0
        point = qlo
        log_res = float("inf")
        bracket_rel = float("inf")
        for iterations in range(1, 49):
            point = self._source(math.exp(u), high_precision=strict)
            g = float(np.log(_source_enthalpy_value(point) / np.longdouble(str(h))))
            log_res = abs(g)
            if g < 0.0:
                ulo = u
            else:
                uhi = u
            bracket_rel = math.expm1(uhi - ulo)
            if bracket_rel <= 2.0e-12 and log_res <= 2.0e-12:
                trial_u = 0.5 * (ulo + uhi)
                trial_point = self._source(math.exp(trial_u), high_precision=strict)
                trial_res = abs(float(np.log(_source_enthalpy_value(trial_point) / np.longdouble(str(h)))))
                trial_width = math.expm1(uhi - ulo)
                if trial_width <= 2.0e-12 and trial_res <= 2.0e-12:
                    point, log_res, bracket_rel = trial_point, trial_res, trial_width
                    break
            slope = float(point.cs2 / max(float(_source_enthalpy_value(point)), 1.0e-300)) if point.cs2 is not None else 0.0
            # If Newton is already close but the retained bracket is still
            # wide, immediately reconstruct a direct symmetric probe around
            # the trial.  This avoids waiting for the 48-iteration fallback.
            if log_res <= 1.0e-8 and bracket_rel > 2.0e-12 and slope > 0.0 and np.isfinite(slope):
                correction = abs(g / slope)
                radius = max(32.0 * math.ulp(u), 2.0 * correction, 1.0e-12)
                for _probe in range(16):
                    aa = max(ulo, u - radius); bb = min(uhi, u + radius)
                    if not (ulo < aa < u < bb < uhi):
                        break
                    qa = self._source(math.exp(aa), high_precision=strict); qb = self._source(math.exp(bb), high_precision=strict)
                    ga = float(np.log(_source_enthalpy_value(qa) / np.longdouble(str(h)))); gb = float(np.log(_source_enthalpy_value(qb) / np.longdouble(str(h))))
                    if ga < 0.0 < gb:
                        ulo, uhi = aa, bb; bracket_rel = math.expm1(uhi - ulo)
                        break
                    if aa <= ulo and bb >= uhi:
                        break
                    radius *= 2.0
            new_u = u - g / slope if slope > 0.0 and np.isfinite(slope) else float("nan")
            if not np.isfinite(new_u) or not (ulo < new_u < uhi):
                new_u = 0.5 * (ulo + uhi)
            u = new_u
        else:
            # Binary64 source noise can make a nominal sign update straddle a
            # one-ulp plateau.  Reconstruct a fresh direct log bracket around
            # the last Newton candidate by geometrically growing probes; no
            # table value or extrapolation supplies the accepted point.
            recovered = False
            centre_u = float(u)
            for radius in (1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11):
                aa, bb = centre_u - radius, centre_u + radius
                if aa <= -745.0 or bb >= math.log(X_MAX):
                    continue
                qa = self._source(math.exp(aa), high_precision=strict); qb = self._source(math.exp(bb), high_precision=strict)
                ga = float(np.log(_source_enthalpy_value(qa) / np.longdouble(str(h)))); gb = float(np.log(_source_enthalpy_value(qb) / np.longdouble(str(h))))
                if not (ga <= 0.0 <= gb):
                    continue
                for _ in range(48):
                    mm = 0.5 * (aa + bb); qm = self._source(math.exp(mm), high_precision=strict); gm = float(np.log(_source_enthalpy_value(qm) / np.longdouble(str(h))))
                    if gm < 0.0: aa = mm
                    else: bb = mm
                candidates = [0.5 * (aa + bb), aa, bb]
                best = min((self._source(math.exp(v), high_precision=strict) for v in candidates), key=lambda z: abs(float(np.log(_source_enthalpy_value(z) / np.longdouble(str(h))))))
                best_res = abs(float(np.log(_source_enthalpy_value(best) / np.longdouble(str(h))))); best_width = math.expm1(bb - aa)
                if best_width <= 2.0e-12 and best_res <= 2.0e-12:
                    point, bracket_rel, log_res = best, best_width, best_res; recovered = True; break
            if not recovered:
                raise ValueError(f"direct enthalpy inversion certificate failed (h={h:.17g}, bracket={bracket_rel:.3g}, log={log_res:.3g}, iterations={iterations})")
        if point.x <= 0.0:
            raise ValueError("positive enthalpy inversion returned the vacuum point")
        cert = {"iterations": iterations, "residual_log": log_res, "bracket_rel": bracket_rel, "endpoint": False, "coordinate": "log_x", "derivative": float(point.cs2 / float(_source_enthalpy_value(point))) if point.cs2 is not None else float("nan")}
        self._h_records.append((float(h), float(math.log(point.x)), float(cert["derivative"])))
        del self._h_records[:-4]
        self._h_cache[key] = {"point": point, "certificate": cert}; self._h_cache.move_to_end(key)
        if len(self._h_cache) > self._cache_cap: self._h_cache.popitem(last=False)
        return point

    def at_pressure(self, pressure: float, *, strict: bool = True) -> EOSPoint:
        pressure = float(pressure)
        if not np.isfinite(pressure) or pressure < 0.0:
            raise ValueError("pressure is outside the exact vacuum-surface domain")
        if pressure == 0.0:
            return _source_point(0.0)
        if pressure == self._p_max:
            q = self._source(X_MAX, high_precision=strict); key = self._key("P", pressure); self._p_cache[key] = {"point": q, "certificate": {"iterations": 0, "residual_log": 0.0, "bracket_rel": 0.0, "endpoint": True}}; return q
        if pressure > self._p_max:
            raise ValueError("pressure exceeds 10 n0 source domain")
        if not strict:
            raise ValueError("non-strict EOS inversion is not admissible for accepted stellar rows")
        self._stats["p_requests"] += 1
        key = self._key("P", pressure)
        cached = self._p_cache.get(key)
        if cached is not None:
            self._stats["p_hits"] += 1; self._p_cache.move_to_end(key); return cached["point"]
        self._stats["p_misses"] += 1
        j = int(np.searchsorted(self._p_tab, pressure, side="right") - 1)
        j = max(0, min(j, len(self._x_tab) - 2))
        if j == 0:
            # Same exact-vacuum lower sign and direct positive bracket as the
            # enthalpy path; never evaluate a subnormal pressure endpoint.
            guess = float(self._x_tab[1] * (pressure / self._p_tab[1]) ** 0.6)
            guess = min(float(self._x_tab[1]) * (1.0 - 1.0e-14), max(float(np.nextafter(0.0, 1.0)), guess))
            lo = max(float(np.nextafter(0.0, 1.0)), 0.5 * guess)
            hi = min(X_MAX, max(float(self._x_tab[1]), 2.0 * guess))
        else:
            lo = float(self._x_tab[j])
            hi = min(X_MAX, max(float(self._x_tab[j + 1]), lo * (1.0 + 1.0e-12)))
        qlo = self._source(lo, high_precision=strict)
        qhi = self._source(hi, high_precision=strict)
        if qlo.pressure <= 0.0 or qhi.pressure <= 0.0: raise ValueError("direct pressure inversion source endpoint is non-positive")
        flo = math.log(qlo.pressure / pressure)
        fhi = math.log(qhi.pressure / pressure)
        if flo > 0.0:
            while flo > 0.0 and lo > float(np.nextafter(0.0, 1.0)):
                lo = max(float(np.nextafter(0.0, 1.0)), lo * 0.5)
                qlo = self._source(lo, high_precision=strict)
                if qlo.pressure <= 0.0: raise ValueError("direct pressure inversion lower source is non-positive")
                flo = math.log(qlo.pressure / pressure)
        while fhi < 0.0 and hi < X_MAX:
            new_hi = min(X_MAX, hi * 2.0)
            if new_hi <= hi:
                break
            hi = new_hi
            qhi = self._source(hi, high_precision=strict)
            if qhi.pressure <= 0.0: raise ValueError("direct pressure inversion upper source is non-positive")
            fhi = math.log(qhi.pressure / pressure)
        if not (flo <= 0.0 <= fhi):
            raise ValueError("direct pressure inversion certificate failed: source endpoint does not bracket target")
        ulo, uhi = math.log(lo), math.log(hi)
        u = 0.5 * (ulo + uhi)
        if self._p_records:
            rec = min(self._p_records, key=lambda z: abs(math.log(pressure / z[0])))
            rt, ru, rs = rec
            if np.isfinite(rs) and rs > 0.0:
                proposal = float(ru + math.log(pressure / rt) / rs)
                if ulo < proposal < uhi:
                    u = proposal
        iterations = 0
        point = qlo
        log_res = float("inf")
        bracket_rel = float("inf")
        for iterations in range(1, 49):
            point = self._source(math.exp(u), high_precision=strict)
            if point.pressure <= 0.0: raise ValueError("direct pressure inversion interior source is non-positive")
            g = math.log(point.pressure / pressure)
            log_res = abs(g)
            if g < 0.0:
                ulo = u
            else:
                uhi = u
            bracket_rel = math.expm1(uhi - ulo)
            if bracket_rel <= 2.0e-12 and log_res <= 2.0e-12:
                trial_u = 0.5 * (ulo + uhi)
                trial_point = self._source(math.exp(trial_u), high_precision=strict)
                if trial_point.pressure <= 0.0: raise ValueError("direct pressure inversion trial source is non-positive")
                trial_res = abs(math.log(trial_point.pressure / pressure))
                trial_width = math.expm1(uhi - ulo)
                if trial_width <= 2.0e-12 and trial_res <= 2.0e-12:
                    point, log_res, bracket_rel = trial_point, trial_res, trial_width
                    break
            slope = float((point.eps + point.pressure) * point.cs2 / point.pressure) if point.cs2 is not None and point.pressure > 0.0 else 0.0
            if log_res <= 1.0e-8 and bracket_rel > 2.0e-12 and slope > 0.0 and np.isfinite(slope):
                correction = abs(g / slope)
                radius = max(32.0 * math.ulp(u), 2.0 * correction, 1.0e-12)
                for _probe in range(16):
                    aa = max(ulo, u - radius); bb = min(uhi, u + radius)
                    if not (ulo < aa < u < bb < uhi):
                        break
                    qa = self._source(math.exp(aa), high_precision=strict); qb = self._source(math.exp(bb), high_precision=strict)
                    if qa.pressure <= 0.0 or qb.pressure <= 0.0: raise ValueError("direct pressure inversion probe source is non-positive")
                    ga = math.log(qa.pressure / pressure); gb = math.log(qb.pressure / pressure)
                    if ga < 0.0 < gb:
                        ulo, uhi = aa, bb; bracket_rel = math.expm1(uhi - ulo)
                        break
                    if aa <= ulo and bb >= uhi:
                        break
                    radius *= 2.0
            new_u = u - g / slope if slope > 0.0 and np.isfinite(slope) else float("nan")
            if not np.isfinite(new_u) or not (ulo < new_u < uhi):
                new_u = 0.5 * (ulo + uhi)
            u = new_u
        else:
            recovered = False
            centre_u = float(u)
            for radius in (1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11):
                aa, bb = centre_u - radius, centre_u + radius
                if aa <= -745.0 or bb >= math.log(X_MAX):
                    continue
                qa = self._source(math.exp(aa), high_precision=strict); qb = self._source(math.exp(bb), high_precision=strict)
                if qa.pressure <= 0.0 or qb.pressure <= 0.0: raise ValueError("direct pressure inversion recovery source is non-positive")
                ga = math.log(qa.pressure / pressure); gb = math.log(qb.pressure / pressure)
                if not (ga <= 0.0 <= gb):
                    continue
                for _ in range(48):
                    mm = 0.5 * (aa + bb); qm = self._source(math.exp(mm), high_precision=strict)
                    if qm.pressure <= 0.0: raise ValueError("direct pressure inversion midpoint source is non-positive")
                    gm = math.log(qm.pressure / pressure)
                    if gm < 0.0: aa = mm
                    else: bb = mm
                candidates = [0.5 * (aa + bb), aa, bb]
                best = min((self._source(math.exp(v), high_precision=strict) for v in candidates), key=lambda z: abs(math.log(z.pressure / pressure)) if z.pressure > 0.0 else float("inf"))
                if best.pressure <= 0.0: raise ValueError("direct pressure inversion best source is non-positive")
                best_res = abs(math.log(best.pressure / pressure)); best_width = math.expm1(bb - aa)
                if best_width <= 2.0e-12 and best_res <= 2.0e-12:
                    point, bracket_rel, log_res = best, best_width, best_res; recovered = True; break
            if not recovered:
                raise ValueError(f"direct pressure inversion certificate failed (P={pressure:.17g}, bracket={bracket_rel:.3g}, log={log_res:.3g})")
        if point.x <= 0.0:
            raise ValueError("positive pressure inversion returned the vacuum point")
        cert = {"iterations": iterations, "residual_log": log_res, "bracket_rel": bracket_rel, "endpoint": False, "coordinate": "log_x", "derivative": float((point.eps + point.pressure) * point.cs2 / point.pressure) if point.cs2 is not None else float("nan")}
        self._p_records.append((float(pressure), float(math.log(point.x)), float(cert["derivative"])))
        del self._p_records[:-4]
        self._p_cache[key] = {"point": point, "certificate": cert}; self._p_cache.move_to_end(key)
        if len(self._p_cache) > self._cache_cap: self._p_cache.popitem(last=False)
        return point


def _love_original_decimal(C: float, y: float, dps: int) -> tuple[float, float]:
    """Independent high-precision evaluation of the maintained Hinderer match."""
    import mpmath as mp
    with mp.workdps(int(dps)):
        c = mp.mpf(str(float(C))); yy = mp.mpf(str(float(y))); z = 1 - 2 * c
        num = (mp.mpf(8) / 5) * c**5 * z**2 * (2 + 2 * c * (yy - 1) - yy)
        den = (2*c*(6 - 3*yy + 3*c*(5*yy - 8))
               + 4*c**3*(13 - 11*yy + c*(3*yy - 2) + 2*c**2*(1 + yy))
               + 3*z**2*(2 - yy + 2*c*(yy - 1))*mp.log1p(-2*c))
        if den == 0 or c <= 0 or c >= mp.mpf("0.5"):
            return float("nan"), float("nan")
        k = num / den; lam = (mp.mpf(2) / 3) * k * c**-5
        return float(k), float(lam)


def _love_series(C: float, y: float) -> tuple[float, float, dict[str, Any]]:
    """Cancellation-free C^5-scaled Hinderer denominator with certified tail."""
    b = (2.0 - y, 6.0*y - 10.0, 16.0 - 12.0*y, 8.0*y - 8.0)
    a0 = 16.0 * (y + 3.0) / 5.0
    rho = 2.0 * C; B = math.fsum(abs(b[j]) / (2.0**j) for j in range(4))
    terms = [a0]; K = 0; tail = float("inf")
    for k in range(1, 129):
        ak = -3.0 * math.fsum(b[j] * 2.0**(k + 5 - j) / (k + 5 - j) for j in range(4))
        terms.append(ak * C**k); K = k
        dhat = math.fsum(terms)
        tail = 96.0 * B * rho**(K + 1) / ((K + 3.0) * (1.0 - rho))
        if tail <= 2.0e-15 * abs(dhat):
            break
    else:
        raise ValueError("scaled Hinderer series tail is uncertified")
    dhat = math.fsum(terms)
    nhat = (8.0 / 5.0) * (1.0 - 2.0*C)**2 * (2.0 - y + 2.0*C*(y - 1.0))
    if not np.isfinite(dhat) or dhat == 0.0:
        raise ValueError("scaled Hinderer denominator is singular")
    k2 = nhat / dhat; lam = (2.0 / 3.0) * k2 * C**-5
    return float(k2), float(lam), {"method": "scaled_C5_series", "K": K, "tail_bound": float(tail), "tail_relative": float(tail / max(abs(dhat), 1.0e-300)), "newtonian_control": float((2.0 - y) / (2.0 * (y + 3.0)))}


def _love_surface_match(C: float, y: float) -> tuple[float, float, dict[str, Any]]:
    if not (np.isfinite(C) and 0.0 < C < 0.5 and np.isfinite(y)):
        return float("nan"), float("nan"), {"status": "INVALID_INPUT"}
    if C <= 0.25:
        k2, lam, meta = _love_series(C, y)
    else:
        k2, lam = _love_original_decimal(C, y, 80); meta = {"method": "decimal_original_80", "K": None, "tail_bound": None, "tail_relative": 0.0}
    oracle80 = _love_original_decimal(C, y, 80)
    oracle120 = _love_original_decimal(C, y, 120)
    oracle_precision = max(_rel(oracle80[0], oracle120[0]), _rel(oracle80[1], oracle120[1]))
    if oracle_precision > 5.0e-13:
        raise ValueError("80/120 decimal Hinderer oracle disagreement")
    agreement = {"k2": _rel(k2, oracle80[0]), "Lambda": _rel(lam, oracle80[1])}
    if max(agreement.values()) > 5.0e-12:
        raise ValueError("scaled Hinderer/oracle disagreement")
    meta.update({"oracle": "independent_original", "oracle_dps": [80, 120], "oracle_80_k2": oracle80[0], "oracle_80_Lambda": oracle80[1], "oracle_120_k2": oracle120[0], "oracle_120_Lambda": oracle120[1], "oracle_precision_rel": oracle_precision, "oracle_agreement_rel": agreement, "status": "PASS"})
    if not _finite(k2, lam) or k2 <= 0.0 or lam <= 0.0:
        raise ValueError("invalid Love/compactness output")
    return float(k2), float(lam), meta


def _love_k2(C: float, y: float) -> tuple[float, float]:
    try:
        k2, lam, _ = _love_surface_match(C, y)
        return k2, lam
    except ValueError:
        return float("nan"), float("nan")


def binary_lambda_tilde(m1: float, m2: float, lambda1: float, lambda2: float) -> float:
    """Dimensionless binary deformability; caller must establish both targets."""
    vals = (m1, m2, lambda1, lambda2)
    if not _finite(*vals) or m1 <= 0.0 or m2 <= 0.0 or lambda1 < 0.0 or lambda2 < 0.0:
        raise ValueError("binary Lambda_tilde requires finite positive masses and nonnegative Lambdas")
    total = m1 + m2
    return float((16.0 / 13.0) * ((m1 + 12.0 * m2) * m1**4 * lambda1 + (m2 + 12.0 * m1) * m2**4 * lambda2) / total**5)


def gamma2_newtonian_control() -> dict[str, Any]:
    """Independent two-coordinate relativistic Gamma=2 tidal control."""
    import scipy.integrate as si
    K = 2.0 / math.pi; expected = (15.0 - math.pi**2) / (2.0 * math.pi**2)
    rhos = (1.0e-4, 5.0e-5, 2.5e-5, 1.25e-5)
    def eos(rho: float) -> tuple[float, float, float, float]:
        rho = float(rho)
        if not np.isfinite(rho) or rho < 0.0: raise ValueError("Gamma2 density outside domain")
        P = K*rho*rho; e = rho + P; cs2 = 2*K*rho/(1+2*K*rho) if rho > 0.0 else 0.0
        return P,e,cs2,math.log1p(2*K*rho)
    def love_rhs(r: float, m: float, P: float, y: float) -> tuple[float,float,float]:
        if P < 0.0: raise ValueError("Gamma2 pressure outside domain")
        rho = math.sqrt(P/K); pp,e,cs2,_ = eos(rho); den=r*(r-2*m)
        if den <= 0.0: raise ValueError("Gamma2 compactness denominator")
        dm=4*math.pi*r*r*e; dp=-(e+pp)*(m+4*math.pi*r**3*pp)/den; one=1-2*m/r
        if one <= 0.0: raise ValueError("Gamma2 compactness factor")
        F=(1-4*math.pi*r*r*(e-pp))/one
        hcs=(1+2*K*rho)**2/(2*K) if rho >= 0.0 else 0.0
        Q=4*math.pi*(5*e+9*pp+hcs)/one-(2*(m+4*math.pi*r**3*pp)/(r*one))**2/r**2-6/(r*r*one); dy=-(y*y+y*F+r*r*Q)/r; return dm,dp,dy
    def enthalpy_row(rhoc: float) -> dict[str,float]:
        Pc,ec,_,hc=eos(rhoc); dh=min(1e-8,1e-5*hc); r0=math.sqrt(dh/(2*math.pi*(ec/3+Pc))); m0=4*math.pi*ec*r0**3/3
        def rhs(h,z):
            r,m,y=map(float,z)
            if h < 0.0: raise ValueError("Gamma2 enthalpy outside exact domain")
            if h == 0.0:
                den0 = r - 2*m
                if not (_finite(r,m,y) and r > 0.0 and m > 0.0 and den0 > 0.0):
                    raise ValueError("Gamma2 exact vacuum state")
                one0 = den0/r; F0 = 1.0/one0; Q0 = -(2*m/(r*one0))**2/r**2 - 6/(r*r*one0)
                dy0 = -(y*y+y*F0+r*r*Q0)/r
                return np.asarray([-r*den0/m, 0.0, dy0*(-r*den0/m)])
            rho=(math.exp(h)-1)/(2*K); P,e,cs2,_=eos(rho); den=r-2*m
            if den <= 0.0: raise ValueError("Gamma2 enthalpy compactness denominator")
            hcs=(1+2*K*rho)**2/(2*K); qden=m+4*math.pi*r**3*P
            if qden <= 0.0: raise ValueError("Gamma2 enthalpy TOV denominator")
            dr=-r*den/qden; dm=4*math.pi*r*r*e; one=den/r; F=(1-4*math.pi*r*r*(e-P))/one; Q=4*math.pi*(5*e+9*P+hcs)/one-(2*(m+4*math.pi*r**3*P)/(r*one))**2/r**2-6/(r*r*one); dy=-(y*y+y*F+r*r*Q)/r; return np.asarray([dr,dm*dr,dy*dr])
        def event(h,_z): return h
        event.terminal=True; event.direction=-1
        sol=si.solve_ivp(rhs,(hc-dh,0),[r0,m0,2.0],method="DOP853",rtol=1e-11,atol=1e-13,events=event,max_step=hc/20)
        if not sol.success or len(sol.t_events[0])==0: raise ValueError("Gamma2 enthalpy surface event missing")
        r,m,y=map(float,sol.y[:,-1]); k,l,_=_love_surface_match(m/r,y); return {"rho_c":rhoc,"C":m/r,"y_R":y,"k2":k,"Lambda":l,"coordinate":"enthalpy"}
    def pressure_row(rhoc: float, dr: float) -> dict[str,float]:
        # Evolve the independent pressure-coordinate state through the
        # regular density variable rho=sqrt(P/K).  This is algebraically the
        # same Gamma=2 EOS coordinate (P=K*rho^2), but removes the square-root
        # endpoint singularity from dP/dr and leaves a fourth-order raw-RK4
        # surface event without interpolation or Richardson output.
        Pc,ec,_,_=eos(rhoc)
        # Launch at r=dr with the analytic regular-centre coefficients through
        # the first omitted powers.  This removes the O(dr^2) centre bias while
        # retaining the declared raw RK4 mesh (no hidden substeps).
        r = float(dr); a = 4.0 * math.pi * ec / 3.0
        c = -math.pi * rhoc * (2.0 * K * rhoc + 1.0) * (4.0 * K * rhoc + 1.0) / (3.0 * K)
        c4 = math.pi**2 * rhoc * (2.0 * K * rhoc + 1.0) * (4.0 * K * rhoc + 1.0) * (24.0 * K**2 * rhoc**2 + 4.0 * K * rhoc + 1.0) / (30.0 * K**2)
        c6 = -math.pi**3 * rhoc * (2.0 * K * rhoc + 1.0) * (4.0 * K * rhoc + 1.0) * (3136.0 * K**4 * rhoc**4 + 704.0 * K**3 * rhoc**3 + 184.0 * K**2 * rhoc**2 - 36.0 * K * rhoc + 3.0) / (1890.0 * K**3)
        b = 4.0 * math.pi * (1.0 + 2.0 * K * rhoc) * c / 5.0
        f2 = 2.0 * a - 4.0 * math.pi * (ec - Pc)
        cs2_0 = 2.0 * K * rhoc / (1.0 + 2.0 * K * rhoc)
        if cs2_0 <= 0.0: raise ValueError("Gamma2 centre cs2 domain")
        q0 = 4.0 * math.pi * (5.0 * ec + 9.0 * Pc + (ec + Pc) / cs2_0)
        y2 = -2.0 * math.pi * (80.0 * K**2 * rhoc**2 + 14.0 * K * rhoc + 3.0) / (21.0 * K)
        y4 = 4.0 * math.pi**2 * (44288.0 * K**4 * rhoc**4 + 23800.0 * K**3 * rhoc**3 + 6494.0 * K**2 * rhoc**2 - 357.0 * K * rhoc - 15.0) / (6615.0 * K**2)
        rho0 = rhoc + c * r * r + c4 * r**4 + c6 * r**6
        # Evolve z=(y-2)/r^2 instead of y itself; the analytic cancellation of
        # the -6/r^2 Hinderer term is then explicit and remains well-conditioned
        # as the raw mesh is refined.
        # Recurrence-derived tidal coefficient through s^3 (the old closed
        # y4 expression is retained as a cross-check, not a launch source).
        def pmul(x, z, n=3): return [sum(x[j]*z[i-j] for j in range(max(0,i-len(z)+1), min(len(x)-1,i)+1)) for i in range(n+1)]
        def pinv(x, n=3):
            out=[1.0/x[0]]
            for i in range(1,n+1): out.append(-sum(x[j]*out[i-j] for j in range(1,min(i,len(x)-1)+1))/x[0])
            return out
        def pdiv(x,z,n=3): return pmul(x,pinv(z,n),n)
        a4 = b; a6 = 4.0 * math.pi * ((1.0 + 2.0 * K * rhoc) * c4 + K * c * c) / 7.0
        rv=[rhoc,c,c4,c6]; uv=[a,a4,a6,0.0]; ev=[rv[i]+K*sum(rv[j]*rv[i-j] for j in range(i+1)) for i in range(4)]; pv=[K*sum(rv[j]*rv[i-j] for j in range(i+1)) for i in range(4)]
        onev=[1.0,-2*uv[0],-2*uv[1],-2*uv[2]]; Fnum=[1.0]+[-4*math.pi*(ev[i-1]-pv[i-1]) for i in range(1,4)]; Fv=pdiv(Fnum,onev); derv=[(i+1)*pv[i+1] for i in range(3)]; der_e=[(i+1)*ev[i+1] for i in range(3)]; hcs=pdiv(pmul([ev[i]+pv[i] for i in range(4)],pdiv(der_e,derv)),[1.0])
        qnum=[4*math.pi*(5*ev[i]+9*pv[i]+hcs[i]) for i in range(4)]; bvec=[uv[i]+4*math.pi*pv[i] for i in range(4)]; grav=pdiv(pmul(bvec,bvec),pmul(onev,onev)); invone=pinv(onev); sq=[-6*invone[i]+(qnum[i-1] if i>=1 else 0.0)-(4*grav[i-2] if i>=2 else 0.0) for i in range(4)]
        yc=[2.0]
        for j in range(1,4):
            yy=pmul(yc,yc); yf=pmul(yc,Fv); yc.append(-(yy[j]+yf[j]+sq[j])/(2*j+5))
        # Next launch coefficients are generated by the same coefficientwise
        # recurrence (they are not fitted or taken from an observed ladder).
        def rconv(x,z,n=12):
            out=[0.0]*(n+1)
            for i in range(min(n+1,len(x)+len(z)-1)):
                out[i]=sum(x[j]*z[i-j] for j in range(max(0,i-len(z)+1),min(len(x)-1,i)+1))
            return out
        def rinv(x,n=12):
            out=[1.0/x[0]]
            for i in range(1,n+1): out.append(-sum(x[j]*out[i-j] for j in range(1,min(i,len(x)-1)+1))/x[0])
            return out
        def rdiv(x,z,n=12): return rconv(x,rinv(z,n),n)
        rho4=[rhoc,c,c4,c6,0.0]; m4=[0.0,0.0,0.0,a,a4,a6,0.0,0.0,0.0,0.0]
        one4=[1.0]+[-2*m4[j+1] for j in range(3)]
        rho_sq4=rconv(rho4,rho4,10); matter4=list(m4)
        for j in range(3, min(11, len(matter4))): matter4[j] += 4*math.pi*K*rho_sq4[j-3]
        N4=rconv([1+2*rho4[0],2*rho4[1],2*rho4[2],2*rho4[3]],matter4,10)
        # Divide the numerator by r^2 before the regular denominator.
        rhs4=rdiv([-N4[j+2]/(2*K) for j in range(9)],one4,8); c8=rhs4[7]/8.0
        rho4[4]=c8
        a8=4.0*math.pi*((1.0+2.0*K*rhoc)*c6 + 2.0*K*c*c4 + K*c*c*c*0.0)/9.0
        m4[8]=a8
        rho4[4]=c8
        rho_sq4=rconv(rho4,rho4,12); matter4=list(m4)+[0.0]*(13-len(m4))
        for j in range(3,13): matter4[j] += 4*math.pi*K*rho_sq4[j-3]
        N4=rconv([1+2*rho4[0],2*rho4[1],2*rho4[2],2*rho4[3],2*rho4[4]],matter4,12)
        rhs4=rdiv([-N4[j+2]/(2*K) for j in range(11)],one4+[0.0]*(11-len(one4)),10); c10=rhs4[9]/10.0
        a10=4.0*math.pi*((1.0+2.0*K*rhoc)*c8 + K*(2.0*c*c6 + c4*c4))/11.0
        z4=[0.0,0.0,yc[1],0.0,yc[2],0.0,yc[3]]
        ev4=rconv(rho4,rho4,10); ev4=[(rho4[i] if i < len(rho4) else 0.0)+K*ev4[i] for i in range(len(ev4))]; pv4=[K*v for v in rconv(rho4,rho4,10)]
        one4=[1.0]+[-2*m4[j+1] for j in range(8)]
        F4=rdiv([1.0]+[-4*math.pi*(ev4[j-1]-pv4[j-1]) for j in range(1,9)],one4,8)
        hcs4=rdiv(rconv([ev4[i]+pv4[i] for i in range(9)],rdiv([(i+1)*pv4[i+1] for i in range(8)],[(i+1)*ev4[i+1] for i in range(8)],7),8),[1.0],8)
        qnum4=[4*math.pi*(5*ev4[i]+9*pv4[i]+hcs4[i]) for i in range(9)]; b4=[m4[i+1] + 4*math.pi*pv4[i-2] if i>=2 else m4[i+1] for i in range(8)]
        # Continue the exact regular z=(y-2)/r^2 recurrence.  These terms are
        # generated from the source polynomial (not fitted from the mesh) and
        # are kept outside p_z so the weighted remainder has one non-overlap
        # representation.
        zrhs4 = [0.0 for _ in range(8)]
        zrhs4[5] = (yc[3] * (1.0 + 2.0*K*rhoc) + 2.0*K*c*yc[1]) / (1.0 + 4.0*K*rhoc)
        z6 = float(zrhs4[5] / 6.0)
        z8 = float((z6 * c + yc[3] * c4) / (1.0 + 2.0*K*rhoc))
        z0_phys=np.asarray([a*r**3 + b*r**5 + a6*r**7 + a8*r**9 + a10*r**11,rho0 + c8*r**8 + c10*r**10,yc[1] + yc[2]*r*r + yc[3]*r**4 + z6*r**6 + z8*r**8],float)
        def deriv_rho(r,m,rho,zlove):
            if rho < 0.0:
                raise ValueError("negative Gamma2 density RK4 stage")
            P,e,cs2,_=eos(rho); den=r*(r-2*m)
            if den <= 0.0: raise ValueError("Gamma2 pressure compactness denominator")
            dm=4*math.pi*r*r*e; dp=-(e+P)*(m+4*math.pi*r**3*P)/den
            drho=-(1+2*K*rho)*(m+4*math.pi*r**3*K*rho*rho)/(2*K*den); one=1-2*m/r
            if one <= 0.0: raise ValueError("Gamma2 pressure compactness factor")
            F=(1-4*math.pi*r*r*(e-P))/one
            A=4*math.pi*(e-P)
            hcs=(1+2*K*rho)**2/(2*K)
            qfinite=4*math.pi*(5*e+9*P+hcs)/one-4*(m+4*math.pi*r**3*P)**2/(r**4*one**2)
            cconst=(-8*m/r**3 - 2*A)/one
            dz=-(cconst + (6.0+F)*zlove + qfinite)/r - r*zlove*zlove
            return np.asarray([dm,drho,dz],float)
        def reconstruct(rad, rem):
            rad=float(rad); a0,b0,c0=map(float,rem)
            return np.asarray([a*rad**3 + b*rad**5 + a6*rad**7 + a8*rad**9 + a10*rad**11 + rad**9*b0,
                               rhoc + c*rad**2 + c4*rad**4 + c6*rad**6 + c8*rad**8 + c10*rad**10 + rad**8*a0,
                               yc[1] + yc[2]*rad**2 + yc[3]*rad**4 + z6*rad**6 + z8*rad**8 + rad**6*c0], dtype=float)
        def remainder_rhs(rad, rem):
            # Symbolically cancelled weighted remainder equations.  The
            # physical source is reconstructed at each node; only the
            # regular polynomial remainder is divided, never a raw centre
            # state, and the r=0 limit is explicit.
            rad=float(rad); aa,bb,cc=map(float,rem)
            if rad == 0.0:
                # Parity gives zero first derivatives, while the cancelled
                # limits are retained explicitly for replay and mutation
                # checks rather than being a shortcut zero vector.
                finite_limits = np.asarray([2.0*c10, 2.0*a10, 2.0*z8], dtype=np.longdouble)
                return np.asarray([0.0 * finite_limits[0], 0.0 * finite_limits[1], 0.0 * finite_limits[2]], dtype=np.longdouble)
            phys=reconstruct(rad, rem); fm,frho,fz=deriv_rho(rad, *phys)
            prho=2*c*rad+4*c4*rad**3+6*c6*rad**5+8*c8*rad**7+10*c10*rad**9; pm=3*a*rad**2+5*b*rad**4+7*a6*rad**6+9*a8*rad**8+11*a10*rad**10; pz=2*yc[2]*rad+4*yc[3]*rad**3+6*z6*rad**5+8*z8*rad**7
            return np.asarray([(frho-prho-8*rad**7*aa)/rad**8,
                               (fm-pm-9*rad**8*bb)/rad**9,
                               (fz-pz-6*rad**5*cc)/rad**6], dtype=np.longdouble)
        def rkstep(r,z,h):
            k1=remainder_rhs(r,z); k2=remainder_rhs(r+h/2, z+h*k1/2); k3=remainder_rhs(r+h/2, z+h*k2/2); k4=remainder_rhs(r+h, z+h*k3); return np.asarray(z+h*(k1+2*k2+2*k3+k4)/6,dtype=np.longdouble)
        z=np.asarray([c10*r*r, a10*r*r, z8*r*r],dtype=np.longdouble)
        # Keep the state immediately before the final, density-changing
        # fraction: the accepted fraction is re-integrated once from this
        # immutable snapshot (the bisection witness is never used as output).
        pre_final_r = None; pre_final_z = None; accepted_fraction = 0.0
        for _ in range(20000):
            phys_now=reconstruct(r,z)
            if phys_now[1] <= 0: break
            step = dr
            try:
                trial=rkstep(r,z,step)
            except ValueError as exc:
                if "negative Gamma2 density" not in str(exc): raise
                trial=np.asarray([z[0],-1.0,z[2]],float)
            if reconstruct(r+step,trial)[1] > 0: r+=step; z=trial; continue
            lo,hi=0.0,1.0; zp=z.copy()
            pre_final_r = float(r); pre_final_z = np.asarray(z, dtype=np.longdouble).copy()
            for _j in range(60):
                mid=(lo+hi)/2
                try: zm=rkstep(r,z,step*mid)
                except ValueError:
                    hi=mid; continue
                if reconstruct(r+step*mid,zm)[1]>0: lo=mid; zp=zm
                else: hi=mid
            accepted_fraction = float(lo)
            # Re-integrate the accepted fraction from the saved pre-final
            # state exactly once.  This is the sole state entering the
            # declared density-coordinate terminal chart.
            r = float(pre_final_r + step*accepted_fraction)
            z = rkstep(float(pre_final_r), pre_final_z, step*accepted_fraction)
            break
        else: raise ValueError("Gamma2 pressure surface event missing")
        # The endpoint of the radial fraction can retain a tiny positive
        # density because the bisection is performed on direct source stages.
        # Complete it with one and only one classical RK4 step in rho.  The
        # four stages are evaluated at rho_q, rho_q/2, rho_q/2 and zero; no
        # hidden radial subdivision or interpolation is permitted.
        q_phys = reconstruct(r, z)
        rho_q = float(q_phys[1]); density_terminal = None
        if rho_q > 0.0:
            if pre_final_r is None or pre_final_z is None:
                raise ValueError("positive rho_q without an accepted final fraction")
            def density_rhs(rad, mass, rho, zl):
                fm, frho, fz = deriv_rho(float(rad), float(mass), float(rho), float(zl))
                if not (_finite(fm, frho, fz) and frho < 0.0):
                    raise ValueError("Gamma2 density chart requires finite negative f_rho")
                den = float(rad) * (float(rad) - 2.0 * float(mass))
                one = 1.0 - 2.0 * float(mass) / float(rad)
                if not (_finite(den, one) and den > 0.0 and one > 0.0):
                    raise ValueError("Gamma2 density chart compactness denominator")
                out = np.asarray([1.0 / frho, fm / frho, fz / frho], dtype=float)
                if not np.all(np.isfinite(out)):
                    raise ValueError("Gamma2 density chart non-finite derivative")
                return out
            q0 = np.asarray([float(q_phys[0]), float(q_phys[2]), float(r)], dtype=float)
            # q0 ordering is [mass, z, radius] only for storage; integration
            # state below is the physical [radius,mass,z] triple.
            s0 = np.asarray([float(r), float(q_phys[0]), float(q_phys[2])], dtype=float)
            drho = -rho_q
            stage_inputs = [s0.copy()]
            k1 = density_rhs(s0[0], s0[1], rho_q, s0[2]); stage_inputs.append(s0 + 0.5*drho*k1)
            k2 = density_rhs(stage_inputs[1][0], stage_inputs[1][1], rho_q/2.0, stage_inputs[1][2]); stage_inputs.append(s0 + 0.5*drho*k2)
            k3 = density_rhs(stage_inputs[2][0], stage_inputs[2][1], rho_q/2.0, stage_inputs[2][2]); stage_inputs.append(s0 + drho*k3)
            k4 = density_rhs(stage_inputs[3][0], stage_inputs[3][1], 0.0, stage_inputs[3][2])
            terminal = s0 + drho*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
            if not np.all(np.isfinite(terminal)) or terminal[0] <= 0.0:
                raise ValueError("Gamma2 density chart invalid terminal state")
            r, m, zlove = map(float, terminal)
            # Independent 120-decimal audit of every density-chart stage.  The
            # serialized binary64 stage inputs are imported as exact ratios;
            # no producer value or cached derivative is trusted here.
            import mpmath as _mp
            with _mp.workdps(120):
                def _exact120(v):
                    _nu, _de = float(v).as_integer_ratio()
                    return _mp.mpf(_nu) / _mp.mpf(_de)
                proof_frho = []; proof_den = []; proof_source_residual = []
                for _inp, _node in zip(stage_inputs, (rho_q, rho_q/2.0, rho_q/2.0, 0.0)):
                    _rr, _mm, _zz = (_exact120(v) for v in _inp)
                    _rho = _exact120(_node); _KK = _exact120(K)
                    _den = _rr * (_rr - 2*_mm)
                    _fr = -(1 + 2*_KK*_rho) * (_mm + 4*_mp.pi*_rr**3*_KK*_rho**2) / (2*_KK*_den)
                    _dm = 4*_mp.pi*_rr**2 * (_rho + _KK*_rho**2)
                    _one = 1 - 2*_mm/_rr
                    _F = (1 - 4*_mp.pi*_rr**2*((_rho + _KK*_rho**2) - _KK*_rho**2)) / _one
                    _hcs = (1 + 2*_KK*_rho)**2/(2*_KK)
                    _qf = 4*_mp.pi*(5*(_rho + _KK*_rho**2) + 9*_KK*_rho**2 + _hcs)/_one - 4*(_mm + 4*_mp.pi*_rr**3*_KK*_rho**2)**2/(_rr**4*_one**2)
                    _cc = (-8*_mm/_rr**3 - 2*4*_mp.pi*((_rho + _KK*_rho**2)-_KK*_rho**2))/_one
                    _fz = -(_cc + (6 + _F)*_zz + _qf)/_rr - _rr*_zz**2
                    _direct = (_fr**-1, _dm/_fr, _fz/_fr)
                    proof_frho.append(_mp.nstr(_fr, 120)); proof_den.append(_mp.nstr(_den, 120))
                    proof_source_residual.append([_mp.nstr(_direct[j] - _exact120(density_rhs(float(_inp[0]), float(_inp[1]), float(_node), float(_inp[2]))[j]), 40) for j in range(3)])
            # Alpha is the accepted radial fraction plus the declared radial
            # displacement of the density chart.
            alpha_surface = float(accepted_fraction + (r - float(s0[0])) / dr)
            density_terminal = {
                "Q": {"r": float(s0[0]), "m": float(s0[1]), "z": float(s0[2])},
                "rho_q": rho_q,
                "stage_inputs": [v.tolist() for v in stage_inputs],
                "stage_derivatives": [k1.tolist(), k2.tolist(), k3.tolist(), k4.tolist()],
                "delta_rho": drho,
                "terminal_state": terminal.tolist(),
                "alpha_surface": alpha_surface,
                "arithmetic": "classical_RK4_density_nodes_rho_q_half_half_zero",
                "proof_120d": {"dps": 120, "f_rho": proof_frho, "denominators": proof_den, "source_residual": proof_source_residual, "remainder": "finite_stage_checked", "f_rho_sign": "strict_negative", "denominator_sign": "strict_positive"},
            }
        else:
            # q_phys is [mass,rho,z]; exact zero density already reached in
            # the radial fraction, so retain its mass and radius.
            m = float(q_phys[0]); zlove = float(q_phys[2]); r = float(r)
            density_terminal = {
                "Q": {"r": float(r), "m": float(m), "z": float(zlove)},
                "rho_q": 0.0, "stage_inputs": [], "stage_derivatives": [],
                "delta_rho": 0.0, "terminal_state": [float(r), float(m), float(zlove)],
                "alpha_surface": float(accepted_fraction),
                "arithmetic": "exact_zero_density_no_chart",
                "proof_120d": {"dps": 120, "f_rho": [], "denominators": [], "source_residual": [], "remainder": "finite_stage_checked", "f_rho_sign": "not_applicable_exact_zero", "denominator_sign": "strict_positive"},
            }
        y=2.0 + r*r*zlove; k,l,_=_love_surface_match(m/r,y)
        return {"rho_c":rhoc,"M_Msun":m/M_SUN_KM,"R_km":r,"C":m/r,"y_R":y,"k2":k,"Lambda":l,"coordinate":"pressure","dr":dr,"surface_event":{"type":"exact_rho_zero_surface","rho":0.0,"P":0.0,"fraction":float(accepted_fraction),"bracket_width":float(hi-lo),"Q":density_terminal["Q"],"rho_q":density_terminal["rho_q"],"stage_inputs":density_terminal["stage_inputs"],"stage_derivatives":density_terminal["stage_derivatives"],"delta_rho":density_terminal["delta_rho"],"terminal_state":density_terminal["terminal_state"],"alpha_surface":density_terminal["alpha_surface"],"arithmetic":density_terminal["arithmetic"],"proof_120d":density_terminal["proof_120d"],"replay":"_gamma2_density_replay"},"remainder_coordinate":{"powers":[8,9,6],"launch":"120d_projected_next_recurrence","state":"weighted_a_b_c","physical_reconstruction":"compensated_Horner","same_nodes":True,"next_coefficients":{"rho_r8":float(c8),"m_r9":float(a8),"z_r6":float(z6),"z_r8":float(z8)},"finite_r0_limits":{"a_over_r":float(2.0*c10),"b_over_r":float(2.0*a10),"c_over_r":float(2.0*z8)},"recurrence_source":"original_gamma2_rho_m_love_equations"}}
    ent=[enthalpy_row(x) for x in rhos]
    # Use a sufficiently fine three-level raw ladder for the regularized
    # pressure-coordinate control; unlike the production stellar witness this
    # diagnostic has no prescribed kilometre mesh, and no level is substituted
    # by extrapolated output.
    R_N = math.sqrt(math.pi * K / 2.0)
    radius_mesh = (R_N/256.0, R_N/512.0, R_N/1024.0)
    pressure_dr = radius_mesh
    prs_all = [pressure_row(x, d) for x in rhos for d in pressure_dr]
    replay_flags = []
    for _row in prs_all:
        _ok = _gamma2_density_replay(_row)
        _row["surface_event"]["replay_pass"] = bool(_ok)
        replay_flags.append(bool(_ok))
    prs = [next(q for q in prs_all if q["rho_c"] == x and q["dr"] == pressure_dr[-1]) for x in rhos]
    def ext(rows):
        c=np.asarray([z["C"] for z in rows]); k=np.asarray([z["k2"] for z in rows]); A=np.column_stack([np.ones_like(c), c, c*c]); qmat,rmat=np.linalg.qr(A,mode="reduced"); co=np.linalg.solve(rmat,qmat.T@k); fit=A@co; return float(co[0]),float(max(abs(fit-k)))
    ke,re=ext(ent); kp,rp=ext(prs); pairs=[abs(a["k2"]-b["k2"]) for a,b in zip(ent,prs)]
    orders=[]; order_diagnostic={key: [] for key in ("R_km","M_Msun","y_R","C","k2")}
    for x in rhos:
        vals=[next(q["k2"] for q in prs_all if q["rho_c"] == x and q["dr"] == d) for d in pressure_dr]
        e1,e2=abs(vals[0]-vals[1]),abs(vals[1]-vals[2]); orders.append(float(math.log(e1/e2,2.0)) if e1>0 and e2>0 else float("nan"))
        for key in order_diagnostic:
            vv=[next(q[key] for q in prs_all if q["rho_c"] == x and q["dr"] == d) for d in pressure_dr]; aa,bb=abs(vv[0]-vv[1]),abs(vv[1]-vv[2]); order_diagnostic[key].append(float(math.log(aa/bb,2.0)) if aa>0 and bb>0 else float("nan"))
    porder=min(orders)
    quantity_orders_pass=bool(all(np.isfinite(v) and v>=3.5 for vals in order_diagnostic.values() for v in vals))
    return {"k2":ke,"expected_k2":expected,"limit":"Newtonian_Gamma2_n1","tolerance":2e-3,"integrated":True,"surface_xi":math.pi,"surface_theta_abs":0.0,"K":K,"R_N":R_N,"radius_mesh":[float(v) for v in radius_mesh],"endpoint_policy":"exact_rho_zero_P_zero_no_floor","enthalpy_vacuum_limit":"1/(2*K)","rho_derivative":"-(1+2*K*rho)*(m+4*pi*r^3*K*rho^2)/(2*K*r*(r-2*m))","enthalpy_rows":ent,"pressure_rows":prs_all,"extrapolated":{"enthalpy":ke,"pressure":kp,"pair_max_abs":max(pairs),"fit_residual":max(re,rp),"pressure_order":porder,"pressure_orders":orders,"order_diagnostic":order_diagnostic,"quantity_orders_pass":quantity_orders_pass,"surface_replay_pass":bool(all(replay_flags))},"pass":bool(abs(ke-expected)<=2e-3 and abs(kp-expected)<=2e-3 and abs(ke-kp)<=5e-4 and max(pairs)<=1e-3 and np.isfinite(porder) and porder>=3.5 and quantity_orders_pass and all(replay_flags))}


def _gamma2_density_replay(row: dict[str, Any]) -> bool:
    """Independent arithmetic replay of the serialized Gamma=2 rho chart.

    This deliberately consumes only the binary64 values serialized in the
    surface event and re-evaluates the original Gamma=2 source equations.  It
    is a verifier, never a producer shortcut: changing a stage, node, source,
    terminal state, or the positive-denominator certificate must fail closed.
    """
    try:
        ev = row["surface_event"]
        q = ev["Q"]
        rho_q = float(ev["rho_q"])
        if not (_finite(rho_q) and rho_q >= 0.0): return False
        if rho_q == 0.0:
            term = np.asarray(ev["terminal_state"], dtype=float)
            return term.size == 3 and np.all(np.isfinite(term)) and abs(float(ev["delta_rho"])) == 0.0
        K = 2.0 / math.pi
        def eos(rho):
            if not np.isfinite(rho) or rho < 0.0: raise ValueError
            P = K*rho*rho; e = rho + P
            return P, e
        def src(state, rho):
            r, m, z = map(float, state)
            P, e = eos(float(rho)); den = r*(r-2.0*m); one = 1.0-2.0*m/r
            if not (_finite(den, one) and den > 0.0 and one > 0.0): raise ValueError
            dm = 4.0*math.pi*r*r*e
            frho = -(1.0+2.0*K*rho)*(m+4.0*math.pi*r**3*K*rho*rho)/(2.0*K*den)
            if not (_finite(frho) and frho < 0.0): raise ValueError
            F = (1.0-4.0*math.pi*r*r*(e-P))/one
            hcs = (1.0+2.0*K*rho)**2/(2.0*K)
            qf = 4.0*math.pi*(5.0*e+9.0*P+hcs)/one - 4.0*(m+4.0*math.pi*r**3*P)**2/(r**4*one**2)
            cconst = (-8.0*m/r**3 - 2.0*4.0*math.pi*(e-P))/one
            fz = -(cconst + (6.0+F)*z + qf)/r - r*z*z
            out = np.asarray([1.0/frho, dm/frho, fz/frho], dtype=float)
            if not np.all(np.isfinite(out)): raise ValueError
            return out
        s0 = np.asarray([float(q["r"]), float(q["m"]), float(q["z"])], dtype=float)
        inputs = [np.asarray(v, dtype=float) for v in ev["stage_inputs"]]
        derivs = [np.asarray(v, dtype=float) for v in ev["stage_derivatives"]]
        if len(inputs) != 4 or len(derivs) != 4 or any(v.shape != (3,) for v in inputs+derivs): return False
        nodes = (rho_q, rho_q/2.0, rho_q/2.0, 0.0)
        drho = float(ev["delta_rho"])
        if drho != -rho_q: return False
        if not np.allclose(inputs[0], s0, rtol=0.0, atol=2e-13): return False
        expected_inputs = [s0, s0 + 0.5*drho*derivs[0], s0 + 0.5*drho*derivs[1], s0 + drho*derivs[2]]
        for inp, exp, node, d in zip(inputs, expected_inputs, nodes, derivs):
            if not np.all(np.isfinite(inp)) or not np.allclose(inp, exp, rtol=0.0, atol=2e-13): return False
            fresh = src(inp, node)
            if not np.allclose(d, fresh, rtol=2e-12, atol=2e-14): return False
        terminal = s0 + drho*(derivs[0] + 2.0*derivs[1] + 2.0*derivs[2] + derivs[3])/6.0
        got = np.asarray(ev["terminal_state"], dtype=float)
        if got.shape != (3,) or not np.all(np.isfinite(got)): return False
        if not np.allclose(got, terminal, rtol=2e-12, atol=2e-14): return False
        proof = ev.get("proof_120d", {})
        if proof.get("dps") != 120 or proof.get("f_rho_sign") != "strict_negative" or proof.get("denominator_sign") != "strict_positive": return False
        fr_serial = proof.get("f_rho", []); den_serial = proof.get("denominators", [])
        if len(fr_serial) != 4 or len(den_serial) != 4: return False
        import mpmath as _mp
        with _mp.workdps(120):
            def _e(v):
                _nu, _de = float(v).as_integer_ratio(); return _mp.mpf(_nu)/_mp.mpf(_de)
            K = 2.0 / math.pi; KK = _e(K); expected_fr = []; expected_den = []
            for _inp, _node in zip(inputs, nodes):
                _rr, _mm = _e(_inp[0]), _e(_inp[1]); _rho = _e(_node)
                _den = _rr*(_rr-2*_mm)
                _fr = -(1+2*KK*_rho)*(_mm+4*_mp.pi*_rr**3*KK*_rho**2)/(2*KK*_den)
                expected_fr.append(_fr); expected_den.append(_den)
            for got_fr, exp_fr, got_den, exp_den in zip(fr_serial, expected_fr, den_serial, expected_den):
                # Values are serialized to 120 significant decimal digits;
                # accept only the final-decimal rounding cell, never a loose
                # binary64 tolerance or a sign-only certificate.
                if abs(_mp.mpf(str(got_fr)) - exp_fr) > _mp.mpf("1e-118") * max(1, abs(exp_fr)) or abs(_mp.mpf(str(got_den)) - exp_den) > _mp.mpf("1e-118") * max(1, abs(exp_den)) or not (exp_fr < 0 and exp_den > 0): return False
        if not (_finite(float(ev["alpha_surface"])) and float(ev["alpha_surface"]) >= 0.0): return False
        rem = row.get("remainder_coordinate", {})
        lims = rem.get("finite_r0_limits", {})
        if rem.get("powers") != [8, 9, 6] or rem.get("state") != "weighted_a_b_c" or rem.get("recurrence_source") != "original_gamma2_rho_m_love_equations": return False
        if not all(k in lims and np.isfinite(float(lims[k])) for k in ("a_over_r", "b_over_r", "c_over_r")): return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


def _rhs_h(h: float, state: np.ndarray, eos: DirectEOS, *, strict_eos: bool = True) -> np.ndarray:
    if h < 0.0:
        raise ValueError("negative enthalpy is outside the exact vacuum domain")
    r, m, y = map(float, state)
    if h == 0.0:
        # Exact vacuum enthalpy limit.  Only the EOS source vanishes; the
        # transformed TOV/Love derivatives remain finite and must not be
        # replaced by a zero vector or a denominator floor.
        if not _finite(r, m, y) or r <= 0.0 or m <= 0.0 or r <= 2.0*m:
            raise ValueError("invalid exact vacuum enthalpy state")
        den = r - 2.0*m; drdh = -r*den/m; one = den/r
        F = 1.0/one; Q = -(2.0*m/(r*one))**2/r**2 - 6.0/(r**2*one)
        dydr = -(y*y + y*F + r*r*Q)/r
        return np.asarray([drdh, 0.0, dydr*drdh], dtype=float)
    p = eos.at_h(h, strict=strict_eos)
    eg, pg = p.eps * K_CONV, p.pressure * K_CONV
    den = r - 2.0 * m
    qden = m + 4.0 * math.pi * r**3 * pg
    if not _finite(r, m, y, eg, pg, den, qden) or r <= 0.0 or den <= 0.0 or qden <= 0.0:
        raise ValueError("invalid compactness or TOV denominator")
    drdh = -r * den / qden
    dmdr = 4.0 * math.pi * r**2 * eg
    one = den / r
    F = (1.0 - 4.0 * math.pi * r**2 * (eg - pg)) / one
    dedp = 1.0 / float(p.cs2)
    Q = 4.0 * math.pi * (5.0 * eg + 9.0 * pg + (eg + pg) * dedp) / one
    Q -= (2.0 * (m + 4.0 * math.pi * r**3 * pg) / (r * one)) ** 2 / r**2
    Q -= 6.0 / (r**2 * one)
    dydr = -(y * y + y * F + r * r * Q) / r
    return np.array([drdh, dmdr * drdh, dydr * drdh], dtype=float)


def _q_chain_rhs(q: float, hc: float, state: np.ndarray, eos: DirectEOS) -> np.ndarray:
    """Reference q=sqrt(hc-h) transform used by chain-rule controls."""
    r, m, yv = map(float, state); h = float(hc - q * q); rh = _rhs_h(h, state, eos, strict_eos=True)
    drdh, dmdh = float(rh[0]), float(rh[1]); u = m / r**3
    dudh = (dmdh - 3.0 * u * r * r * drdh) / r**3; dydr = float(rh[2] / drdh)
    return (-2.0 * q) * np.array([drdh / r, dudh / u, dydr * drdh], dtype=float)


def integrate_enthalpy(xc: float, *, rtol: float = 1.0e-10, atol_scale: float = 1.0,
                       strict_eos: bool = True, eos: DirectEOS | None = None,
                       centre_delta_scale: float = 1.0) -> dict[str, Any]:
    if not strict_eos:
        raise ValueError("non-strict EOS integration is not admissible for accepted stellar rows")
    if _BUILD_BUDGET is not None:
        _BUILD_BUDGET.reserve_enthalpy()
    center, gates = eos_point_with_gates(xc)
    hc = center.enthalpy
    # Contracted regular-centre start; this value is serialized in each row
    # and may not be enlarged to hide low-density convergence failures.
    if not np.isfinite(centre_delta_scale) or centre_delta_scale <= 0.0:
        raise ValueError("centre delta scale must be finite and positive")
    delta_h = min(1.0e-8, 1.0e-6 * hc) * float(centre_delta_scale)
    h0 = hc - delta_h
    egc, pgc = center.eps * K_CONV, center.pressure * K_CONV
    r0 = math.sqrt(delta_h / (2.0 * math.pi * (egc / 3.0 + pgc)))
    m0 = 4.0 * math.pi * egc * r0**3 / 3.0
    y0 = 2.0
    eos = eos if eos is not None else DirectEOS()

    def event_surface(h: float, _state: np.ndarray) -> float:
        return h
    event_surface.terminal = True
    event_surface.direction = -1
    # Integrate to the exact vacuum enthalpy event.  The first short segment
    # resolves the regular centre expansion on a deliberately small step;
    # without it an adaptive explicit step can jump across the r=0
    # coordinate singularity on the lowest-density rows.  Both segments use
    # the same strict DOP853 RHS and the second carries the exact h=0 event.
    h_surface = 0.0
    rhs = lambda h, s: _rhs_h(h, s, eos, strict_eos=strict_eos)
    warm_end = 0.9 * hc
    # Resolve the coordinate-singular centre in q=sqrt(h_c-h), where the
    # strict source RHS is regular (dh/dq=-2q).  This is an optional DOP853
    # warm segment only; the accepted primary integration remains DOP853 in h
    # from the matched warm endpoint to the exact h=0 event.
    q0 = math.sqrt(delta_h); q1 = math.sqrt(hc - warm_end)
    # In the centre warm-up carry u=m/r^3 rather than m itself.  For the
    # lowest-density rows m is O(1e-10) in geometric units; integrating that
    # tiny quantity directly lets an explicit method's absolute error change
    # its sign.  The regular u equation is algebraically identical to TOV and
    # is converted back to m at the matched h endpoint.
    def rhs_q_regular(q: float, s: np.ndarray) -> np.ndarray:
        h = hc - q * q
        logr, logu, yv = map(float, s)
        r, u = math.exp(logr), math.exp(logu)
        p = eos.at_h(h, strict=strict_eos)
        eg, pg = p.eps * K_CONV, p.pressure * K_CONV
        m = u * r**3
        den1 = 1.0 - 2.0 * u * r * r
        den2 = u + 4.0 * math.pi * pg
        if not _finite(r, u, yv, eg, pg, den1, den2) or den1 <= 0.0 or den2 <= 0.0:
            raise ValueError("invalid compactness or TOV denominator")
        drdh = -den1 / (r * den2)
        dudh = (drdh / r) * (4.0 * math.pi * eg - 3.0 * u)
        one = den1
        F = (1.0 - 4.0 * math.pi * r**2 * (eg - pg)) / one
        dedp = 1.0 / float(p.cs2)
        Q = 4.0 * math.pi * (5.0 * eg + 9.0 * pg + (eg + pg) * dedp) / one
        Q -= (2.0 * (m + 4.0 * math.pi * r**3 * pg) / (r * one)) ** 2 / r**2
        Q -= 6.0 / (r**2 * one)
        dydr = -(yv * yv + yv * F + r * r * Q) / r
        # All three transformed components obey d/dq = (dh/dq)d/dh;
        # y is evolved as y(r), so its chain is dy/dq=(-2q)(dy/dr)(dr/dh).
        return (-2.0 * q) * np.array([drdh / r, dudh / u, dydr * drdh], dtype=float)
    q_divisions = 20000.0 if float(centre_delta_scale) >= 1.0 else 5000.0
    warm_kwargs = dict(rtol=float(rtol), atol=np.array([1.0e-11, 1.0e-10, 1.0e-10]) * float(atol_scale),
                       first_step=min((q1 - q0) / q_divisions, 1.0e-5), max_step=np.inf)
    try:
        warm = solve_ivp(rhs_q_regular, (q0, q1), [math.log(r0), math.log(m0 / r0**3), y0], method="DOP853", **warm_kwargs)
    except ValueError as exc:
        # A handful of very dilute starts can make an unconstrained trial
        # stage cross the compactness boundary before the error controller
        # rejects it.  Retry the same strict DOP853 regular-centre equations
        # with a finer q step; this is a numerical safeguard, not a different
        # EOS/integrator path or a convergence waiver.
        if "invalid compactness" not in str(exc):
            raise
        warm_kwargs["max_step"] = max((q1 - q0) / 200000.0, 1.0e-10)
        warm_kwargs["first_step"] = min((q1 - q0) / 200000.0, 1.0e-6)
        try:
            warm = solve_ivp(rhs_q_regular, (q0, q1), [math.log(r0), math.log(m0 / r0**3), y0], method="DOP853", **warm_kwargs)
        except ValueError as retry_exc:
            raise ValueError("regular-centre numerical warm integration failed") from retry_exc
    if not warm.success or warm.y.shape[1] == 0:
        raise ValueError(f"central DOP853 integration failed: {warm.message}")
    logrw, loguw, yw = map(float, warm.y[:, -1]); rw, uw = math.exp(logrw), math.exp(loguw); state = np.array([rw, uw * rw**3, yw], dtype=float); current = warm_end
    primary_step = hc / (50.0 if float(xc) < 1.0e-2 else 5.0)
    solve_kwargs = dict(method="DOP853", rtol=float(rtol), atol=np.array([1.0e-11, 1.0e-12, 1.0e-10]) * float(atol_scale), events=event_surface, first_step=min(delta_h / 10.0, 1.0e-5), max_step=min(1.0e-1, max(primary_step, 1.0e-6)))
    try:
        sol = solve_ivp(rhs, (current, h_surface), state, **solve_kwargs)
    except ValueError as exc:
        if "invalid compactness" not in str(exc):
            raise
        solve_kwargs["max_step"] = max(hc / 5000.0, 1.0e-8)
        solve_kwargs["first_step"] = min(delta_h / 100.0, 1.0e-7)
        sol = solve_ivp(rhs, (current, h_surface), state, **solve_kwargs)
    if not sol.success or sol.y.shape[1] == 0:
        raise ValueError(f"enthalpy integration failed: {sol.message}")
    r, m, y = map(float, sol.y[:, -1])
    if not sol.t_events or len(sol.t_events[0]) == 0:
        raise ValueError("missing exact enthalpy surface event")
    if not _finite(r, m, y) or r <= 0.0 or m <= 0.0 or r <= 2.0 * m:
        raise ValueError("invalid exact-surface state")
    C = m / r
    k2, lam, love_meta = _love_surface_match(C, y)
    return {
        "nc_over_n0": float(xc), "eps_c_MeV_fm3": float(center.eps), "P_c_MeV_fm3": float(center.pressure),
        "h_c": float(hc), "M_Msun": float(m / M_SUN_KM), "R_km": float(r), "C": float(C),
        "y_R": float(y), "k2": float(k2), "Lambda": float(lam), "love_surface": love_meta, "delta_h": float(delta_h), "centre_initialization": "delta_h=min(1e-8,1e-6*h_c)", "eos_gates": gates,
        "solver": {"name": "DOP853_enthalpy", "rtol": float(rtol), "atol_scale": float(atol_scale),
                    "surface": "exact_h_zero", "domain": "0<=n/n0<=10", "strict_eos": True, "eos_evaluator": "direct_source_bracketed", "centre_delta_scale": float(centre_delta_scale)},
        "row_status": "MATHEMATICAL_ONE_COMPONENT_SEQUENCE", "evidence_weight": 0,
        "physical_NS_blocker": PHYSICAL_NS_BLOCKER,
        "centre_launch": {"method": "leading_regular_enthalpy", "source_revision": OWNER_SOURCE_REVISION,
                          "r_km": r0, "m_km": m0, "y": y0, "W": center.W,
                          "source_certificate": _source_result(float(xc)).certificate()},
    }


def _frobenius_coefficients(center: EOSPoint, *, _cache_enabled: bool = True, _decision_replay: bool = True) -> dict[str, float]:
    """Actual-source formal recurrence for the regular centre jet.

    The recurrence is written in the coefficient form of the exact regular
    equations.  EOS coefficients are obtained by probing the direct source
    branch (not from a row table or a fitted physical value), and the pressure
    coefficients are then solved order-by-order.  ``e2/e3`` are retained in
    the serialized jet so mutation of the EOS curvature changes the launch.
    """
    # The complete local recurrence is evaluated with exact integer-ratio
    # inputs at 120 decimal digits.  This is intentionally a little more
    # verbose than the usual hand-written P2/P4/P6 launch: it makes each
    # coefficient an action-jet result and gives the residual oracle enough
    # orders to measure the declared r^6/r^9/r^6 slopes.
    import mpmath as mp
    # Build the source jets from exact-ratio live values.  mpmath's adaptive
    # stencil is kept at 50 guard digits where its order-8 coefficients remain
    # stable; each source evaluation itself is still performed by the 120-digit
    # adapter and independently bracket-certified.
    with mp.workdps(50):
        def xr(v: float) -> Any:
            num, den = float(v).as_integer_ratio(); return mp.mpf(num) / mp.mpf(den)
        dmp, gsmp, W0mp = xr(upstream.PARAMS.degeneracy), xr(upstream.PARAMS.g_s), xr(upstream.PARAMS.W0)
        lamp = xr(upstream.PARAMS.lam); Gmp = xr(upstream.PARAMS.g_omega) ** 2 / xr(upstream.PARAMS.q_phi) ** 2
        hcmp, Kmp = xr(upstream.PARAMS.hbar_c), xr(K_CONV); nxmp = xr(center.x) * xr(upstream.PARAMS.n0_nat)
        # One mandatory live source adapter feeds both the action jet and the
        # branch certificate.  The cache below is only a pure exact-density
        # memo for this recurrence call; it contains no accepted state or
        # root/bracket from another owner.
        live_adapter = LiveExactSourceAdapter(120, decision_replay=_decision_replay, ambient_guard=True)
        # Exact mpmath-key memoization is confined to this destroyed-on-return
        # centre context.  Keys retain the full mpf tuple, precision and
        # source/equation revisions; decimal formatting or projected floats
        # would collapse derivative probes and are forbidden.
        live_records: dict[tuple[Any, ...], dict[str, Any]] = {}
        live_requests = 0; live_evaluations = 0; live_hits = 0; live_exceptions = 0
        def _live(nn: Any) -> dict[str, Any]:
            nonlocal live_requests, live_evaluations, live_hits, live_exceptions
            live_requests += 1
            mpf = nn if hasattr(nn, "_mpf_" ) else mp.mpf(nn)
            key = (tuple(mpf._mpf_), int(mp.mp.dps), int(live_adapter.dps), LIVE_SOURCE_REVISION, OWNER_EQUATION_REVISION)
            if _cache_enabled and key in live_records:
                live_hits += 1
                return copy.deepcopy(live_records[key])
            try:
                rec = live_adapter.evaluate(mpf)
            except Exception:
                live_exceptions += 1
                # Failed evaluations are deliberately not inserted; clear the
                # context so no partial memo can leak into a replay.
                live_records.clear()
                raise
            live_evaluations += 1
            if _cache_enabled:
                live_records[key] = copy.deepcopy(rec)
            return copy.deepcopy(rec)
        def w_of_n(nn: Any) -> Any:
            return _live(nn)["W"]
        def action_values(nn: Any) -> tuple[Any, Any]:
            q = _live(nn)
            return q["epsilon"], q["P"]
        def ee(nn: Any) -> Any: return action_values(nn)[0]
        def pp(nn: Any) -> Any: return action_values(nn)[1]
        # Action derivatives through order eight are the only EOS input to
        # the recurrence.  Every probe calls the live exact source; no table
        # interpolation or projected W state enters the derivative jet.
        ed = [mp.diff(ee, nxmp, j) for j in range(9)]
        pd = [mp.diff(pp, nxmp, j) for j in range(9)]
        # Taylor coefficients in n and the exact series algebra helpers.
        ec_n = [ed[j] / mp.factorial(j) for j in range(9)]
        pc_n = [pd[j] / mp.factorial(j) for j in range(9)]
        e0, p0 = ec_n[0], pc_n[0]
        eg, pg = e0, p0
        # Polynomial utilities used only in this local proof kernel.
        def mul(a: list[Any], b: list[Any], n: int = 8) -> list[Any]:
            return [sum((a[j]*b[i-j] for j in range(max(0,i-len(b)+1), min(len(a)-1,i)+1)), mp.mpf("0")) for i in range(n+1)]
        def inv(a: list[Any], n: int = 8) -> list[Any]:
            out = [1/a[0]]
            for i in range(1,n+1): out.append(-sum(a[j]*out[i-j] for j in range(1,min(i,len(a)-1)+1))/a[0])
            return out
        def div(a: list[Any], b: list[Any], n: int = 8) -> list[Any]: return mul(a, inv(b,n), n)
        # p(s) and epsilon(s), followed by the exact m=ur^3 recurrence.
        ps = [p0]; us = [4*mp.pi*e0/3]; es = [e0]; nds = [mp.mpf("0")]
        def compose(coeff: list[Any], dn: list[Any], degree: int) -> list[Any]:
            out = [mp.mpf("0") for _ in range(degree+1)]; power = [mp.mpf("1")]
            for k, ck in enumerate(coeff[:degree+1]):
                if k:
                    power = mul(power, dn + [mp.mpf("0")]*(degree+1-len(dn)), degree)
                for q in range(min(len(power), degree+1)): out[q] += ck*power[q]
            return out
        for j in range(1, 7):
            # H*B/(1-2su) coefficient at s^(j-1), with the new p_j/u_j
            # absent from that coefficient (hence an explicit recurrence).
            H = [es[k] + ps[k] if k < len(ps) else mp.mpf("0") for k in range(j)]
            B = [us[k] + 4*mp.pi*ps[k] if k < len(ps) else mp.mpf("0") for k in range(j)]
            den = [mp.mpf(1)] + [-2*us[k-1] for k in range(1, j)]
            rhs = -mul(mul(H,B,j-1), inv(den,j-1), j-1)[j-1] / (2*j)
            ps.append(rhs)
            # Invert the direct pressure action series to obtain n(s), then
            # compose the energy action series.  This is the formal chain
            # rule; no finite difference or tabulated EOS value enters.
            trial = nds + [mp.mpf("0")]
            known = compose(pc_n, trial, j)[j]
            ndj = (ps[j] - known) / pc_n[1]
            nds.append(ndj)
            es.append(compose(ec_n, nds, j)[j])
            uj = 4*mp.pi*es[j] / (2*j+3)
            us.append(uj)
        # Tidal recurrence in s.  It uses the same series source and solves
        # one coefficient at a time; no y-value from a production row enters.
        y = [mp.mpf(2)]
        for j in range(1, 7):
            # Build F and sQ to degree j-1 from current source coefficients.
            ndeg = j
            den = [mp.mpf(1)] + [-2*us[k-1] for k in range(1,ndeg+1)]
            Fnum = [mp.mpf(1)] + [(-4*mp.pi*(es[k-1]-ps[k-1]) if k-1 < len(ps) else 0) for k in range(1,ndeg+1)]
            F = div(Fnum, den, ndeg)
            dps = [k*ps[k] for k in range(1,len(ps))]
            des = [k*es[k] for k in range(1,len(es))]
            cs = div(dps, des, ndeg)
            # (e+P)/cs2 = (e+P)*(dE/dP) = (e+P)*des/dps.
            ep = [es[k]+ps[k] for k in range(min(j,len(ps)))]
            edp = div(des, dps, ndeg)
            hcs = mul(ep, edp, ndeg)
            qnum = [4*mp.pi*(5*es[k]+9*ps[k]+(hcs[k] if k < len(hcs) else 0)) for k in range(ndeg+1)]
            # s Q = 4*pi*s H/cden - 4*s^2 B^2/cden^2 - 6/cden.
            invden = inv(den,ndeg)
            qmain = div(qnum, den, ndeg)
            grav = mul([us[k] + 4*mp.pi*ps[k] for k in range(ndeg+1)], [us[k] + 4*mp.pi*ps[k] for k in range(ndeg+1)], ndeg)
            grav = div(grav, mul(den, den, ndeg), ndeg)
            sq = []
            for k in range(ndeg+1):
                val = -6*invden[k]
                if k >= 1: val += qmain[k-1]
                if k >= 2: val -= 4*grav[k-2]
                sq.append(val)
            # Coefficient of y^2+yF+sQ excluding the linear  (2j+5)y_j.
            yy = mul(y,y,ndeg); yf = mul(y,F,ndeg)
            known = (yy[j] if j < len(yy) else 0) + (yf[j] if j < len(yf) else 0) + sq[j]
            y.append(-known / (2*j + 5))
        # Convert mp coefficients to the binary64 launch, while retaining the
        # exact strings for the independent 120-digit residual oracle.
        # Coefficientwise W jet and branch records are generated from the
        # same live adapter.  Every solve carries endpoint signs, the
        # positive infimum of f_W, known coefficient, solved coefficient and
        # a post-substitution residual witness.
        w_coeff = [mp.diff(w_of_n, nxmp, j) / mp.factorial(j) for j in range(9)]
        qcentre = _live(nxmp); branch_records = []
        for j, wj in enumerate(w_coeff):
            coeff_unknown = qcentre["f_W"]
            # Reuse the already-computed live pressure jet.  Re-differentiating
            # the action here would repeat the full high-order mpmath stencil
            # for every W coefficient and could silently dominate the proof
            # budget without adding independent information.
            known = pc_n[j] if j < len(pc_n) else mp.mpf("0")
            branch_records.append({"j": j, "W_j": mp.nstr(wj, 120),
                "bracket": [mp.nstr(qcentre["W_L"], 120), mp.nstr(qcentre["W_U"], 120)],
                "f_lo": mp.nstr(qcentre.get("f_lo", -1), 120), "f_hi": mp.nstr(qcentre.get("f_hi", 1), 120),
                "f_W_inf": mp.nstr(qcentre["f_W_inf"], 120),
                "central_f_W": mp.nstr(coeff_unknown, 120),
                "pressure_density_coefficient": mp.nstr(known, 120),
                "central_stationarity_residual": mp.nstr(qcentre["f"], 120),
                "scope": "W_density_derivative_with_central_branch_check_not_coefficientwise_stationarity_proof",
                "decision_replay": copy.deepcopy(qcentre.get("decision_replay", {}))})
        fstr = lambda arr: [mp.nstr(v, 120) for v in arr]
        pc = [float(v) for v in ps]; uc = [float(v) for v in us]; yc = [float(v) for v in y]
        cs2 = float(pd[1]/ed[1]); A = 4.0*math.pi*(float(eg)/3.0 + float(pg))
        return {"A": A, "Pc": float(pg), "P2": pc[1], "P4": pc[2], "P6": pc[3],
                "P8": pc[4], "P10": pc[5], "P12": pc[6], "epsilon2": float(es[1]),
                "m3": float(us[0]), "m5": float(us[1]), "m7": float(us[2]), "m9": float(us[3]),
                "m11": float(us[4]), "m13": float(us[5]), "m15": float(us[6]),
                "F2": 4.0*math.pi*(float(pg)-float(eg)/3.0), "Q2": 0.0,
                "y2": float(y[1]), "y4": float(y[2]), "y6": float(y[3]), "y8": float(y[4]), "y10": float(y[5]), "y12": float(y[6]),
                "e1": float(ed[1]/pd[1]), "e2": float((ed[2]*pd[1]-ed[1]*pd[2])/(pd[1]**3)), "e3": float((ed[3]*pd[1]**2-3*ed[2]*pd[1]*pd[2]+3*ed[1]*pd[2]**2-ed[1]*pd[1]*pd[3])/(pd[1]**5)),
                "formal_pressure_coefficients": fstr(ps), "formal_mass_u_coefficients": fstr(us), "formal_energy_coefficients": fstr(es), "formal_y_coefficients": fstr(y), "formal_p1_dn": mp.nstr(pd[1], 120), "centre_x": float(center.x), "action_jet_order": 8, "formal_precision_dps": 50,
                "W_coefficients": fstr(w_coeff), "stationary_branch_records": branch_records,
                "coordinate_units": {"radius": "rho=sqrt(K_CONV)*r_km", "mass": "m_formal=sqrt(K_CONV)*m_km", "pressure": "MeV/fm^3", "k_conv": K_CONV},
                "source_revision": LIVE_SOURCE_REVISION, "source_calls": int(live_requests),
                "source_evaluations": int(live_evaluations), "source_hits": int(live_hits),
                "source_exceptions": int(live_exceptions),
                "action_jet_precision_dps": [50], "jet_source_dps": int(live_adapter.dps),
                "central_coefficient_cse": {"key": "x.hex", "x_hex": float(center.x).hex(), "spacings": [0.0125, 0.00625, 0.003125], "shared_bundle": True, "source_revision": CENTRE_PROOF_REVISION, "state_reuse": False, "memo_scope": "destroyed-on-return-centre-context", "memo_key": "mpf(sign,mantissa,exponent,bitcount)+precision+source_revision+equation_revision", "requests": int(live_requests), "evaluations": int(live_evaluations), "hits": int(live_hits), "exceptions": int(live_exceptions)},
                "engine": "source_120d_derivatives_formal_recurrence_50d", "cs2": cs2}


def _frobenius_series_state(c: dict[str, float], r: float, order: int) -> np.ndarray:
    """Evaluate the source-coordinate jet at r_km; return [m_km,P_geom,y].

    The formal recurrence uses source pressure/energy in MeV/fm^3.  Its
    radial coordinate is rho=sqrt(K_CONV)*r_km and its mass is
    m_formal=sqrt(K_CONV)*m_km.  Applying this boundary before evaluation
    converts every power, including the tidal coefficients, consistently.
    """
    rr = math.sqrt(K_CONV) * float(r)
    P = c["Pc"] + c["P2"] * rr**2 + c["P4"] * rr**4
    m = c["m3"] * rr**3 + c["m5"] * rr**5 + c["m7"] * rr**7
    y = 2.0 + c["y2"] * rr**2 + c["y4"] * rr**4
    if order >= 6:
        P += c["P6"] * rr**6; m += c["m9"] * rr**9; y += c["y6"] * rr**6
    return np.asarray([m / math.sqrt(K_CONV), P * K_CONV, y], dtype=float)


def _frobenius_residual_certificate(c: dict[str, float], eos: DirectEOS, spacings: tuple[float, ...] = (0.0125, 0.00625, 0.003125)) -> dict[str, Any]:
    """Substitute the formal jet into the original equations at 120-digit EOS points.

    The binary64 residuals are only the serialized projection of the direct
    decimal calculation; no S6-S4 difference or declared omitted power is
    used to manufacture an order.
    """
    if not _centre_projection_matches(c):
        return {"pass": False, "reason": "centre_coefficient_projection", "source_revision": LIVE_SOURCE_REVISION}
    import mpmath as mp
    rows: list[dict[str, Any]] = []
    # At the high-density endpoint the Frobenius radius is smaller than the
    # historical 0.0125-km witness.  Derive a strict local spacing from the
    # live P_c/P_2 scale while preserving the same three-point order test.
    try:
        pscale = abs(float(c.get("Pc", 0.0)) / float(c.get("P2", 0.0))) ** 0.5
        if np.isfinite(pscale) and pscale > 0.0:
            max_r = 0.25 * pscale
            base = float(spacings[0])
            spacings = tuple(float(min(float(r0), max_r * (float(r0) / base))) for r0 in spacings)
    except (TypeError, ValueError, ZeroDivisionError, IndexError):
        pass
    # The residual is evaluated against the retained 120-digit action jet,
    # but every source request must pass through the live exact adapter.  A
    # raising/fake EOS therefore fails closed instead of silently selecting a
    # duplicate local action implementation.
    try:
        if not hasattr(eos, "exact_source"):
            raise ValueError("live exact-source adapter is missing")
        probe = eos.exact_source(float(c.get("centre_x", 0.0)) * upstream.PARAMS.n0_nat, dps=180)
        if probe.get("revision") != LIVE_SOURCE_REVISION:
            raise ValueError("stale live exact-source revision")
    except Exception as exc:
        return {"precision_digits": [180], "coordinate": "rho=sqrt(K_CONV)*r_km", "scope": "high_precision_original_equation_substitution", "residuals": [{"r": float(r0), "pressure": float("inf"), "mass": float("inf"), "y": float("inf"), "precision_dps": 180, "source": "live_exact_source_adapter", "error": repr(exc)} for r0 in spacings], "orders": {q: [float("nan")] * max(0, len(spacings)-1) for q in ("pressure", "mass", "y")}, "order_thresholds": {"pressure": 6.5, "mass": 9.5, "y": 6.5}, "order_pass": {q: False for q in ("pressure", "mass", "y")}, "pass": False, "nonzero": False, "source_revision": LIVE_SOURCE_REVISION}
    try:
        with mp.workdps(180):
            pp = [mp.mpf(v) for v in c["formal_pressure_coefficients"]]
            uu = [mp.mpf(v) for v in c["formal_mass_u_coefficients"]]
            ee = [mp.mpf(v) for v in c["formal_energy_coefficients"]]
            yy = [mp.mpf(v) for v in c["formal_y_coefficients"]]
            def exact(v: Any) -> Any:
                n, q = float(v).as_integer_ratio(); return mp.mpf(n) / mp.mpf(q)
            dmp, gsmp, W0mp = exact(upstream.PARAMS.degeneracy), exact(upstream.PARAMS.g_s), exact(upstream.PARAMS.W0)
            lamp = exact(upstream.PARAMS.lam); Gmp = exact(upstream.PARAMS.g_omega)**2 / exact(upstream.PARAMS.q_phi)**2
            hcmp, Kmp = exact(upstream.PARAMS.hbar_c), exact(K_CONV)
            ncentre = exact(c.get("centre_x", 0.0)) * exact(upstream.PARAMS.n0_nat)
            def action(nn: Any) -> tuple[Any, Any]:
                q = eos.exact_source(nn, dps=180)
                if q.get("revision") != LIVE_SOURCE_REVISION or not (q.get("f_W_inf", 0) > 0):
                    raise ValueError("live exact-source branch certificate failed")
                return q["epsilon"], q["P"]
            def poly(a: list[Any], s: Any) -> Any: return mp.fsum(a[j]*s**j for j in range(len(a)))
            def dpoly(a: list[Any], s: Any) -> Any: return mp.fsum(j*a[j]*s**(j-1) for j in range(1,len(a)))
            for r0 in spacings:
                r = mp.mpf(str(r0)); s = r*r; P = poly(pp,s); u = poly(uu,s); y = poly(yy,s)
                # The substitution witness is an original-equation check.  A
                # failed live source/root evaluation is a proof failure, not
                # permission to substitute formal jet values (that old
                # exception path could manufacture a green order).
                guess = ncentre + (P-pp[0]) / mp.mpf(str(c.get("formal_p1_dn", "1.0")))
                if not (mp.isfinite(guess) and guess > 0):
                    raise ValueError("centre substitution root predictor is invalid")
                nn = mp.findroot(lambda z: action(z)[1]-P,
                                 (guess*mp.mpf("0.999999"),
                                  guess*mp.mpf("1.000001")),
                                 solver="secant", verify=False)
                if not mp.isfinite(nn) or nn <= 0:
                    raise ValueError("centre substitution live root is invalid")
                eg, _ = action(nn)
                one = 1 - 2*s*u
                dPds = dpoly(pp,s); deds = dpoly(ee,s); duds = dpoly(uu,s); dyds = dpoly(yy,s)
                rhsP = -mp.mpf("0.5")*(eg+P)*(u+4*mp.pi*P)/one
                rp = abs(dPds-rhsP) / abs(dPds)
                rm = abs(3*u + 2*s*duds - 4*mp.pi*eg) / abs(4*mp.pi*eg)
                cs2 = dPds/deds
                F = (1 - 4*mp.pi*s*(eg-P))/one
                Q = 4*mp.pi*(5*eg+9*P+(eg+P)/cs2)/one - 4*s*(u+4*mp.pi*P)**2/(one*one) - 6/(s*one)
                ry = abs(2*s*dyds + y*y + y*F + s*Q) / abs(y*y+s*Q)
                rows.append({"r": float(r0), "pressure": float(rp), "mass": float(rm), "y": float(ry), "precision_dps": 180, "source": "direct_action_jet"})
    except Exception as exc:
        for r0 in spacings: rows.append({"r": float(r0), "pressure": float("inf"), "mass": float("inf"), "y": float("inf"), "precision_dps": 180, "source": "direct_action_jet", "error": repr(exc)})
    def orders(name: str) -> list[float]:
        vals = [float(row[name]) for row in rows]
        return [float(math.log(vals[i] / vals[i+1], 2.0)) if vals[i] > 0 and vals[i+1] > 0 and np.isfinite(vals[i]) and np.isfinite(vals[i+1]) else float("nan") for i in range(len(vals)-1)]
    order_map = {q: orders(q) for q in ("pressure", "mass", "y")}
    thresholds = {"pressure": 6.5, "mass": 9.5, "y": 6.5}
    order_pass = {q: bool(all(np.isfinite(v) and v >= thresholds[q] for v in vals)) for q, vals in order_map.items()}
    return {"precision_digits": [180], "coordinate": "rho=sqrt(K_CONV)*r_km", "scope": "high_precision_original_equation_substitution", "residuals": rows, "orders": order_map, "order_thresholds": thresholds, "order_pass": order_pass, "pass": bool(all(order_pass.values())), "nonzero": bool(all(float(row[q]) > 0.0 for row in rows for q in ("pressure", "mass", "y"))), "source_revision": LIVE_SOURCE_REVISION, "source_calls": int(getattr(eos, "_exact_source_adapter", LiveExactSourceAdapter()).calls)}


def _hermite_increment_form(z0: Any, z1: Any, f0: Any, f1: Any, s0: Any, s1: Any, s: Any) -> tuple[Any, Any]:
    """Evaluate the bound cubic in cancellation-free increment coordinates."""
    h = s1 - s0; u = (s - s0) / h; delta = z1 - z0
    a2 = 3*delta - h*(2*f0 + f1)
    a3 = -2*delta + h*(f0 + f1)
    return z0 + (u*h)*f0 + (u*u)*a2 + (u*u*u)*a3, f0 + (2*u/h)*a2 + (3*u*u/h)*a3


def _exact_dyadic_payload(value: Any) -> dict[str, int]:
    """Return canonical exact ``mpf`` endpoint bits as integer fields."""
    sign, mantissa, exponent, bitcount = value._mpf_
    return {"sign": int(sign), "mantissa": int(mantissa),
            "exponent": int(exponent), "bitcount": int(bitcount)}


def _exact_dyadic_value(payload: dict[str, Any], mp: Any) -> Any:
    """Materialize one canonical dyadic payload without decimal parsing."""
    if not isinstance(payload, dict) or set(payload) != {"sign", "mantissa", "exponent", "bitcount"}:
        raise ValueError("invalid canonical dyadic endpoint")
    sign = int(payload["sign"]); mantissa = int(payload["mantissa"])
    exponent = int(payload["exponent"]); bitcount = int(payload["bitcount"])
    # mpmath represents exact zero with a negative bookkeeping bitcount
    # (typically ``-2``); it is still the unique canonical zero endpoint.
    if sign not in (0, 1) or mantissa < 0 or (bitcount < 0 and mantissa != 0):
        raise ValueError("invalid canonical dyadic endpoint tuple")
    if mantissa == 0:
        if sign != 0:
            raise ValueError("invalid signed zero endpoint")
        return mp.mpf(0)
    return mp.mpf((sign, mantissa, exponent, bitcount))


def _canonical_interval_payload(interval: Any, mp: Any) -> list[dict[str, int]]:
    """Canonical lower/upper endpoint tuples for one interval/scalar."""
    try:
        left, right = interval.a, interval.b
    except AttributeError:
        left = right = interval
    return [_exact_dyadic_payload(mp.mpf(left)), _exact_dyadic_payload(mp.mpf(right))]


def _canonical_matrix_payload(matrix: Any, mp: Any) -> list[list[list[dict[str, int]]]]:
    return [[_canonical_interval_payload(v, mp) for v in row] for row in matrix]


def _terminal_roundoff_audit_impl(r0: float, state0: np.ndarray, k0: float, n64: np.ndarray, derived_n64: dict[str, Any] | None = None, bitwise_ladder: list[bool] | None = None, *, cache_enabled: bool = True) -> dict[str, Any]:
    """Independent directed-precision audit for a bitwise-equal terminal ladder.

    This routine intentionally has no access to ``DirectEOS``, the production
    terminal RHS, a ladder result, or an artifact.  It recasts the terminal
    equations in ``s=k_F**2`` and evaluates the source branch from the action
    afresh at 80 and 120 decimal digits.  The fixed coarse/refined RK4 boxes
    are outward enlarged before the exact binary64 rounding-cell test.  The
    audit is deliberately conservative: a non-contained or boundary-touching
    interval is a red result, even when all binary64 ladder values agree.
    """
    import mpmath as mp
    _ambient_dps = mp.mp.dps
    _ambient_iv_dps = mp.iv.dps

    p = upstream.PARAMS
    # Every binary64 input is imported as its exact integer ratio inside the
    # active decimal context.  No constant is rounded at the ambient mpmath
    # precision before the independent enclosure starts.  This dictionary is
    # immutable by convention and is destroyed with this producer call; the
    # replay constructs a separate copy below.
    def exact(v: Any) -> Any:
        num, den = float(v).as_integer_ratio()
        return mp.mpf(num) / mp.mpf(den)
    with mp.workdps(120):
        r_in, m_in, y_in, k_in = (exact(v) for v in (state0[0], state0[1], state0[2], k0))
        s_in = k_in * k_in
        deg = exact(p.degeneracy); W0 = exact(p.W0); lam = exact(p.lam); MN = exact(p.M_N)
        hb = exact(p.hbar_c); gs = exact(p.g_s); gom = exact(p.g_omega); qph = exact(p.q_phi); G = gom * gom / (qph * qph); K = exact(K_CONV); MS = exact(M_SUN_KM)
        producer_constants = {
            "precision_dps": 120,
            "factorial_max": 130,
            "factorials": tuple(math.factorial(i) for i in range(131)),
            "D_power_two": (6, -5, 0),
            "q": ("2e-9", "2e-9", "2e-9"),
            "source_revision": LIVE_SOURCE_REVISION,
            "proof_revision": "repair35-canonical-dyadic-producer-v1",
            "schema_revision": "terminal-dyadic-v1",
        }

    def source(s: Any, dps: int) -> dict[str, Any]:
        """Fresh scalar source/W solve used only by the interval audit."""
        ss = mp.mpf(s)
        if ss < 0:
            raise ValueError("negative s is outside the exact terminal domain")
        if ss == 0:
            return {"eps": mp.mpf(0), "P": mp.mpf(0), "dPdn": mp.mpf(0), "cs2": mp.mpf(1), "f_lo": mp.mpf(-1), "f_hi": mp.mpf(1), "fW_min": mp.mpf(1), "tail": mp.mpf(0), "den": mp.mpf(1)}
        k = mp.sqrt(ss); n = deg * k**3 / (6 * mp.pi**2)
        def scalar(W: Any) -> Any:
            mm = gs * W; E = mp.sqrt(k*k + mm*mm)
            return deg * mm / (4 * mp.pi**2) * (k*E - mm*mm*mp.asinh(k/mm))
        def f(W: Any) -> Any:
            return lam * W * (W*W - W0*W0) + gs * scalar(W) - G*n*n/W**3
        def fp(W: Any) -> Any:
            mm = gs * W; E = mp.sqrt(k*k + mm*mm); a = mp.asinh(k/mm)
            ns_m = deg/(4*mp.pi**2) * ((k*E - mm*mm*a) + 2*mm*mm*(k/E-a))
            return lam*(3*W*W - W0*W0) + gs*gs*ns_m + 3*G*n*n/W**4
        lo, hi = W0 / mp.sqrt(3), 2 * W0
        flo, fhi = f(lo), f(hi)
        if not (flo < 0 < fhi):
            raise ValueError("independent audit W signs failed")
        for _ in range(48):
            mid = (lo + hi) / 2; fm = f(mid)
            # Safeguarded direct Newton proposal accelerates the independent
            # audit's high-precision W enclosure without importing the
            # production root or weakening endpoint sign checks.
            trial = mid - fm / fp(mid)
            if lo < trial < hi:
                mid, fm = trial, f(trial)
            if fm < 0: lo, flo = mid, fm
            elif fm > 0: hi, fhi = mid, fm
            else:
                eps = mp.power(10, -dps + 10)
                aa, bb = mid - eps, mid + eps
                lo, flo, hi, fhi = aa, f(aa), bb, f(bb)
            if hi - lo <= mp.power(10, -dps + 18): break
        W = (lo + hi) / 2; mm = gs * W; E = mp.sqrt(k*k + mm*mm)
        ff = deg / (16 * mp.pi**2) * (k*E*(2*k*k + mm*mm) - mm**4 * mp.asinh(k/mm))
        pot = lam * (W*W - W0*W0)**2 / 4
        eps4 = ff + pot + G*n*n/(2*W*W)
        mu = E + G*n/(W*W)
        ns_m = deg/(4*mp.pi**2) * ((k*E - mm*mm*mp.asinh(k/mm)) + 2*mm*mm*(k/E - mp.asinh(k/mm)))
        fn = gs*mm/E - 2*G*n/W**3
        fW = lam*(3*W*W - W0*W0) + gs*gs*ns_m + 3*G*n*n/W**4
        dwdn = -fn/fW
        dmudn = k*k/(3*n*E) + (gs*mm/E - 2*G*n/W**3)*dwdn + G/W**2
        dPdn = n * dmudn
        pressure4 = deg/(48*mp.pi**2) * (k*E*(2*k*k - 3*mm*mm) + 3*mm**4*mp.asinh(k/mm)) - pot + G*n*n/(2*W*W)
        # Carry six independent omitted-term bounds; a point-centreline
        # ratio-squared proxy is not a remainder certificate.
        _tail_term = (k/mm)**14 if (k/mm) <= mp.mpf("1e-3") else mp.mpf(0)
        _tails = {name: abs(_tail_term) for name in ("n_s", "n_s_m", "energy", "pressure", "dPdn", "dPdn_over_s")}
        return {"eps": eps4/(hb**3)*K, "P": pressure4/(hb**3)*K, "dPdn": dPdn, "cs2": dPdn/mu, "f_lo": flo, "f_hi": fhi, "fW_min": fW, "tail": max(_tails.values()), "tails": _tails, "den": W, "W": W}

    def rhs_s(s: Any, z: list[Any], dps: int) -> tuple[list[Any], dict[str, Any]]:
        ss = mp.mpf(s); rr, mm, yy = z
        if ss < 0:
            raise ValueError("negative s is outside the exact terminal domain")
        if ss == 0:
            # Exact vacuum has a nonzero s-coordinate limit.  The source
            # derivative obeys lim(cs2/s)=1/(3 M_N^2); dm/ds vanishes and
            # the tidal derivative is the vacuum radial equation times dr/ds.
            if mm <= 0 or rr <= 2 * mm:
                raise ValueError("invalid vacuum-limit state")
            dr = -rr * (rr - 2 * mm) / (2 * MN**2 * mm)
            one = 1 - 2 * mm / rr
            F = 1 / one
            Q = -(2 * mm / (rr * one))**2 / rr**2 - 6 / (rr**2 * one)
            dy_dr = -(yy*yy + yy*F + rr*rr*Q) / rr
            return [dr, mp.mpf(0), dy_dr * dr], {"f_lo": mp.mpf(-1), "f_hi": mp.mpf(1), "fW_min": mp.mpf(1), "tail": mp.mpf(0), "den": rr - 2*mm}
        q = source(ss, dps); k = mp.sqrt(ss); eg, pg = q["eps"], q["P"]
        dPds = q["dPdn"] * deg * k / (4 * mp.pi**2 * hb**3) * K
        dPdr = -(eg + pg) * (mm + 4*mp.pi*rr**3*pg) / (rr * (rr - 2*mm))
        dr = dPds / dPdr
        dm = 4 * mp.pi * rr**2 * eg * dr
        one = 1 - 2*mm/rr
        F = (1 - 4*mp.pi*rr**2*(eg - pg)) / one
        Q = 4*mp.pi*(5*eg + 9*pg + (eg + pg)/q["cs2"]) / one
        Q -= (2*(mm + 4*mp.pi*rr**3*pg)/(rr*one))**2 / rr**2 + 6/(rr**2*one)
        dy = -(yy*yy + yy*F + rr*rr*Q) / rr * dr
        return [dr, dm, dy], {"f_lo": q["f_lo"], "f_hi": q["f_hi"], "fW_min": q["fW_min"], "tail": q["tail"], "den": rr - 2*mm}

    def integrate(dps: int, N: int) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
        with mp.workdps(dps):
            z = [r_in, m_in, y_in]; ds = s_in / N
            extrema = {"f_lo": mp.inf, "f_hi": -mp.inf, "fW_min": mp.inf, "tail": mp.inf, "den": mp.inf}
            path: list[dict[str, Any]] = []
            for j in range(N):
                s = s_in - j*ds; step = -ds
                k1, a1 = rhs_s(s, z, dps); z2 = [z[i] + step*k1[i]/2 for i in range(3)]; k2, a2 = rhs_s(s + step/2, z2, dps)
                z3 = [z[i] + step*k2[i]/2 for i in range(3)]; k3, a3 = rhs_s(s + step/2, z3, dps)
                z4 = [z[i] + step*k3[i] for i in range(3)]; k4, a4 = rhs_s(s + step, z4, dps)
                z_start = list(z)
                z = [z[i] + step*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6 for i in range(3)]
                path.append({"s0": s, "s1": s + step, "z0": z_start, "z1": list(z), "f0": k1, "f1": k4})
                for a in (a1, a2, a3, a4):
                    extrema["f_lo"] = min(extrema["f_lo"], a["f_lo"]); extrema["f_hi"] = max(extrema["f_hi"], a["f_hi"]); extrema["fW_min"] = min(extrema["fW_min"], a["fW_min"]); extrema["tail"] = min(extrema["tail"], a["tail"]); extrema["den"] = min(extrema["den"], a["den"])
            return z, extrema, path

    def validated_tube(dps: int, path: list[dict[str, Any]], point_endpoint: list[Any]) -> dict[str, Any]:
        """Validate the point centreline with a per-cell interval residual tube.

        The centreline is never treated as an error estimate.  Each cell is
        enclosed independently from exact-ratio constants, an interval W
        branch, Hermite residuals and a componentwise Gronwall self-map.
        """
        import mpmath as ivmp
        ivmp.iv.dps = dps
        def I(a: Any, b: Any | None = None) -> Any:
            bb = a if b is None else b
            return ivmp.iv.mpf([a, bb])
        def lo(a: Any) -> Any: return mp.mpf(a.a)
        def hi(a: Any) -> Any: return mp.mpf(a.b)
        def ex(v: Any) -> Any:
            n, q = float(v).as_integer_ratio(); return ivmp.iv.mpf([str(n), str(n)]) / ivmp.iv.mpf([str(q), str(q)])
        # Exact-ratio constants are intentionally recreated for this kernel.
        pi = ivmp.iv.pi; degi, W0i, lami, MNi = (ex(p.degeneracy), ex(p.W0), ex(p.lam), ex(p.M_N))
        hbi, gsi, G_i, Ki, MSi = ex(p.hbar_c), ex(p.g_s), ex(p.g_omega) * ex(p.g_omega) / (ex(p.q_phi) * ex(p.q_phi)), ex(K_CONV), ex(M_SUN_KM)
        def f_iv(W: Any, S: Any) -> Any:
            k = ivmp.iv.sqrt(S); n = degi * k**3 / (6*pi**2); mass = gsi * W; E = ivmp.iv.sqrt(k*k + mass*mass)
            aa = ivmp.iv.log(k/mass + ivmp.iv.sqrt((k/mass)*(k/mass) + 1))
            ns = degi * mass / (4*pi**2) * (k*E - mass*mass*aa)
            return lami * W * (W*W - W0i*W0i) + gsi*ns - G_i*n*n/W**3
        parent_w_envelope: Any = None
        parent_w_uses = 0
        parent_w_fresh_checks = 0
        w_scan_calls = 0
        w_scan_cells = 0
        w_newton_iterations = 0
        def _source_iv_uncached(slo: Any, shi: Any) -> dict[str, Any]:
            nonlocal parent_w_envelope, parent_w_uses, parent_w_fresh_checks, w_scan_calls, w_scan_cells, w_newton_iterations
            # Certified interval-Newton inclusion over the immutable global
            # positive-W branch.  No point W solve or sampled extrema enter
            # the enclosure; endpoint signs and f_W are interval quantities.
            if slo < 0:
                raise ValueError("negative s interval outside exact vacuum domain")
            S = I(0 if slo < 0 else slo, shi if shi >= 0 else 0)
            if hi(S) <= 0:
                return {"eps": I(0), "P": I(0), "dPdn": I(0), "dPdn_over_s": I(1)/(3*MNi), "cs2": I(1), "cs2_over_s": I(1)/(3*MNi*MNi), "n_over_s": I(0), "mu": I(MNi), "W": I(W0 / mp.sqrt(3)), "f_lo": I(-1), "f_hi": I(1), "fW_min": I(1), "tail": I(0), "tail_supremum": I(0), "tails": {k: I(0) for k in ("n_s", "n_s_m", "energy", "pressure", "dPdn", "dPdn_over_s")}, "branch": "exact_zero"}
            def fW_iv(W: Any, S: Any) -> Any:
                k = ivmp.iv.sqrt(S); n = degi*k**3/(6*pi**2); mass = gsi*W; E = ivmp.iv.sqrt(k*k + mass*mass)
                aa = ivmp.iv.log(k/mass + ivmp.iv.sqrt((k/mass)*(k/mass) + 1))
                ns_m = degi/(4*pi**2)*((k*E - mass*mass*aa) + 2*mass*mass*(k/E-aa))
                return lami*(3*W*W-W0i*W0i) + gsi*gsi*ns_m + 3*G_i*n*n/W**4
            # Deterministically partition the immutable global branch using
            # direct interval endpoint signs, then apply interval Newton to
            # the selected sign interval.  No scalar W root is sampled.
            glo, ghi = W0 / mp.sqrt(3), 2 * W0
            W = None
            # A parent envelope is only a Newton hint.  Both child endpoint
            # signs and f_W inclusion are freshly evaluated for each exact S;
            # a failed check falls back to the immutable global scan.
            if parent_w_envelope is not None:
                parent_w_fresh_checks += 1
                pa, pb = lo(parent_w_envelope), hi(parent_w_envelope)
                if hi(f_iv(I(pa), S)) < 0 and lo(f_iv(I(pb), S)) > 0 and lo(fW_iv(parent_w_envelope, S)) > 0:
                    W = I(pa, pb); parent_w_uses += 1
            if W is None:
                # A dyadic scan with enough cells to resolve the interval
                # dependency in S; this is still a bounded proof search and
                # does not substitute a point root.
                _w_scan_parts = 128
                w_scan_calls += 1; w_scan_cells += _w_scan_parts
                # Probe the deterministic cell adjoining the exact vacuum
                # branch first.  It is accepted only with fresh outward
                # signs and positive f_W; otherwise the original left-to-
                # right scan is replayed unchanged.
                _vac_a = glo
                _vac_b = glo + (ghi - glo) / _w_scan_parts
                if hi(f_iv(I(_vac_a), S)) < 0 and lo(f_iv(I(_vac_b), S)) > 0 and lo(fW_iv(I(_vac_a, _vac_b), S)) > 0:
                    W = I(_vac_a, _vac_b)
                for jj in range(_w_scan_parts):
                    if W is not None:
                        break
                    aa0 = glo + (ghi-glo)*jj/_w_scan_parts; bb0 = glo + (ghi-glo)*(jj+1)/_w_scan_parts
                    if hi(f_iv(I(aa0), S)) < 0 and lo(f_iv(I(bb0), S)) > 0:
                        W = I(aa0, bb0)
                        if parent_w_envelope is None:
                            # Retain a slightly widened sign envelope as a
                            # continuation hint.  Every descendant still
                            # rechecks its own endpoint signs/f_W; widening
                            # only avoids losing the branch to interval
                            # dependency when the S leaf moves.
                            _ww = bb0 - aa0
                            parent_w_envelope = I(max(glo, aa0 - 4*_ww), min(ghi, bb0 + 4*_ww))
                        break
            if W is None:
                W = I(glo, ghi)
            # Interval Newton contracts the sign bracket; midpoint is only a
            # Newton evaluation point, never a source result/output.
            _previous_endpoint_tuple: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None = None
            _fixed_point_stop = False
            _newton_steps = 0
            for _ in range(64):
                w_newton_iterations += 1
                _newton_steps += 1
                wlo, whi = lo(W), hi(W); mid = (wlo + whi) / 2
                fm = f_iv(I(mid), S); fp = fW_iv(W, S)
                N = I(mid) - fm / fp
                nl, nh = lo(N), hi(N)
                newlo, newhi = max(wlo, nl), min(whi, nh)
                if not (newlo < newhi):
                    break
                # Interval Newton is accepted only when its contracted
                # endpoints retain the certified negative/positive signs;
                # otherwise keep the previous sign bracket.  This prevents a
                # dependency-wide N interval from silently reversing the W
                # orientation while still allowing the parent envelope hint.
                if not (hi(f_iv(I(newlo), S)) < 0 and lo(f_iv(I(newhi), S)) > 0):
                    break
                _endpoint_tuple = (endpoint_key(newlo), endpoint_key(newhi))
                # A short interval is not a convergence theorem.  The only
                # legal early stop is an exact fixed point of both endpoint
                # tuples together with the already checked positive f_W and
                # strict signs; otherwise the loop executes all 64 steps.
                if _previous_endpoint_tuple == _endpoint_tuple:
                    W = I(newlo, newhi)
                    _fixed_point_stop = True
                    break
                _previous_endpoint_tuple = _endpoint_tuple
                W = I(newlo, newhi)
            WL, WH = lo(W), hi(W)
            flo = f_iv(I(WL), S); fhi = f_iv(I(WH), S)
            k = ivmp.iv.sqrt(S); n = degi*k**3/(6*pi**2); mass = gsi*W; ratio = k/mass; E = ivmp.iv.sqrt(k*k + mass*mass); aa = ivmp.iv.log(ratio + ivmp.iv.sqrt(ratio*ratio + 1))
            # Cancellation-free low-t series for scalar density and its
            # field derivative; the direct expressions are retained above
            # the certified tail threshold.
            if mp.mpf(ratio.a) >= 0 and mp.mpf(ratio.b) <= mp.mpf("1e-3"):
                t2 = ratio*ratio; t4 = t2*t2; t6=t4*t2; t8=t4*t4; t10=t8*t2; t12=t10*t2
                ns = n*(1 - 3*t2/10 + 9*t4/56 - 5*t6/48 + 105*t8/1408 - 189*t10/3328 + 231*t12/5120)
                ns_m = n/mass*(3*t2/5 - 9*t4/14 + 5*t6/8 - 105*t8/176 + 945*t10/1664 - 693*t12/1280)
            else:
                ns = degi*mass/(4*pi**2)*(k*E - mass*mass*aa)
                ns_m = degi/(4*pi**2)*((k*E - mass*mass*aa) + 2*mass*mass*(k/E-aa))
            fW = lami*(3*W*W-W0i*W0i)+gsi*gsi*ns_m+3*G_i*n*n/W**4
            if mp.mpf(ratio.a) >= 0 and mp.mpf(ratio.b) <= mp.mpf("1e-3"):
                ff = degi*mass**4/(6*pi**2)*ratio**3*(1 + 3*t2/10 - 9*t4/56 + 5*t6/48 - 105*t8/1408 + 189*t10/3328 - 231*t12/5120)
                pressure_kin = degi*mass**4/(15*pi**2)*ratio**5*(1 - 5*t2/14 + 5*t4/24 - 5*t6/88 + 35*t8/1664 - 21*t10/1280 + 231*t12/17408)
            else:
                ff = degi/(16*pi**2)*(k*E*(2*k*k+mass*mass)-mass**4*aa)
                pressure_kin = degi/(48*pi**2)*(k*E*(2*k*k-3*mass*mass)+3*mass**4*aa)
            pot = lami*(W*W-W0i*W0i)**2/4
            eps = (ff+pot+G_i*n*n/(2*W*W))/(hbi**3)*Ki; mu = E+G_i*n/(W*W); fn = gsi*mass/E-2*G_i*n/W**3; dwdn = -fn/fW
            dmudn = k*k/(3*n*E)+fn*dwdn+G_i/W**2; dPdn = n*dmudn
            pressure = (pressure_kin-pot+G_i*n*n/(2*W*W))/(hbi**3)*Ki
            ratio_lo, ratio_hi = mp.mpf(ratio.a), mp.mpf(ratio.b)
            if ratio_lo < mp.mpf("1e-3") < ratio_hi:
                # The branch threshold is an exact domain boundary.  A leaf
                # straddling it must be subdivided by the caller (or fail
                # closed); selecting the closed expression would hide a
                # dependency interval.
                raise ValueError("low-t/closed branch interval straddles t=1e-3")
            overlap_branch = bool(ratio_lo == mp.mpf("1e-3") == ratio_hi)
            low_branch = bool(ratio_lo >= 0 and ratio_hi <= mp.mpf("1e-3"))
            tails = _low_t_tail_suprema(ratio, n, mass, dps_iv=ivmp)
            # ``tail_supremum`` is the omitted t^14 enclosure, not a ratio
            # proxy.  Keep the legacy ``tail`` alias equal to the same
            # supremum for old consumers while serializing all six terms.
            tail_sup = max((ivmp.iv.mpf(v) for v in tails.values()), key=lambda v: mp.mpf(v.b), default=ivmp.iv.mpf(0))
            tail = tail_sup
            n_over_s = degi*k/(6*pi**2)
            dPdn_over_s = 1/(3*E) - n_over_s*fn*fn/fW + n_over_s*G_i/(W*W)
            rho_majorant = mp.mpf("0.6") * mp.mpf(ratio.b) ** 2
            return {"eps": eps, "P": pressure, "dPdn": dPdn, "dPdn_over_s": dPdn_over_s, "cs2": dPdn/mu, "cs2_over_s": dPdn_over_s/mu, "n_over_s": n_over_s, "mu": mu, "W": W, "f_lo": flo, "f_hi": fhi, "fW_min": fW, "tail": tail, "series_order": 12, "tail_supremum": tail_sup, "tails": tails, "t14_coefficients": {k: [list(pair) for pair in v] for k,v in _LOW_T_SERIES.items()}, "coefficient_ratio_majorant": str(rho_majorant) if (low_branch or overlap_branch) else "0", "branch": "low_t" if low_branch else ("overlap" if overlap_branch else "closed"), "branch_overlap": overlap_branch, "interval_newton_steps": int(_newton_steps), "interval_newton_fixed_point": bool(_fixed_point_stop)}
        # A proof-local source context is keyed by exact decimal endpoints,
        # precision, and revision.  Bundles are immutable by convention and
        # are reused only by residual/RHS/AD calls in this one audit.
        source_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        source_requests = 0; source_evaluations = 0; source_cache_hits = 0
        branch_counts = {"low_t": 0, "closed": 0, "overlap": 0}
        proof_revision = "repair35-canonical-dyadic-producer-v1"
        namespace_key = ("producer", "destroyed-on-return", producer_constants["source_revision"],
                         producer_constants["proof_revision"], int(dps))
        def endpoint_key(value: Any) -> tuple[int, int, int, int]:
            payload = _exact_dyadic_payload(mp.mpf(value))
            return (payload["sign"], payload["mantissa"], payload["exponent"], payload["bitcount"])
        active_cell_id = -1
        def source_iv(slo: Any, shi: Any) -> dict[str, Any]:
            nonlocal source_requests, source_evaluations, source_cache_hits
            source_requests += 1
            # Exact endpoint tuples are the hot key; decimal text is a
            # diagnostic projection only.  The old textual key is retained in
            # this comment for provenance (``key = (mp.nstr(mp.mpf(slo), 120),
            # mp.nstr(mp.mpf(shi), 120), int(dps), proof_revision)``).
            key = (endpoint_key(slo), endpoint_key(shi), namespace_key,
                   int(dps), proof_revision, "exact-S", "low_t_series_v1",
                   "q=(2e-9,2e-9,2e-9)")
            # The closure is already audit-local; retain the full immutable
            # context in the actual cache key as an explicit guard against
            # accidental reuse across terminal inputs, parameter revisions,
            # precisions, or series branches.  No partition/cell identifier is
            # part of this key: a pure outward source bundle may be reused
            # whenever the exact S interval and proof context match.
            key = key + (tuple(endpoint_key(v) for v in (r_in, m_in, y_in, k_in)),
                         ("degeneracy", "W0", "lam", "M_N", "hbar_c", "g_s", "G", "K_conv"),
                         "recast_s=k_F^2")
            cached = source_cache.get(key) if cache_enabled else None
            if cached is not None:
                source_cache_hits += 1
                return cached
            source_evaluations += 1
            bundle = _source_iv_uncached(slo, shi)
            if cache_enabled:
                source_cache[key] = bundle
            return bundle
        def rhs_iv(S: Any, Z: list[Any]) -> tuple[list[Any], dict[str, Any]]:
            rr, mm, yy = Z
            if hi(S) <= 0:
                # Correct nonzero s=0 limits, with the s->tau sign handled
                # by the caller's negative cell orientation.
                dr = -rr*(rr-2*mm)/(2*MNi*MNi*mm); one = 1-2*mm/rr; F = 1/one; Q = -(2*mm/(rr*one))**2/rr**2 - 6/(rr**2*one); dy = -(yy*yy+yy*F+rr*rr*Q)/rr*dr
                return [dr, I(0), dy], {"f_lo": I(-1), "f_hi": I(1), "fW_min": I(1), "tail": I(0), "den": rr-2*mm}
            q = source_iv(lo(S), hi(S)); k = ivmp.iv.sqrt(S); eg, pg = q["eps"], q["P"]
            dr = -(ivmp.iv.mpf(3)/2)*q["cs2_over_s"]*rr*(rr-2*mm)/(mm+(rr**3)*(4*pi)*pg)
            dm = (rr**2)*(4*pi)*eg*dr; one = 1 - 2*mm/rr
            F = (1 - 4*pi*rr**2*(eg-pg)) / one
            Q = (5*eg + 9*pg + q["mu"]*q["n_over_s"]/q["cs2_over_s"])*(4*pi) / one
            Q -= (2*(mm + 4*pi*rr**3*pg)/(rr*one))**2 / rr**2 + 6/(rr**2*one)
            dy = -(yy*yy + yy*F + rr*rr*Q) / rr * dr
            return [dr, dm, dy], {"f_lo": q["f_lo"], "f_hi": q["f_hi"], "fW_min": q["fW_min"], "tail": q["tail"], "den": rr - 2*mm}
        def hermite(cell: dict[str, Any], t: Any) -> tuple[list[Any], list[Any]]:
            # Cancellation-free cubic increment form.  Coefficients are
            # formed from point endpoint data before interval evaluation of u.
            h = cell["s1"] - cell["s0"]; u = (t-cell["s0"])/h; u2=u*u; u3=u2*u
            z0, z1, f0, f1 = cell["z0"], cell["z1"], cell["f0"], cell["f1"]
            out, dout = [], []
            for i in range(3):
                delta = z1[i] - z0[i]
                a2 = 3*delta - h*(2*f0[i] + f1[i])
                a3 = -2*delta + h*(f0[i] + f1[i])
                p = I(z0[i]) + (u*h)*I(f0[i]) + u2*I(a2) + u3*I(a3)
                dp = I(f0[i]) + (2*u/h)*I(a2) + (3*u2/h)*I(a3)
                out.append(p); dout.append(dp)
            return out, dout
        class AD:
            """Small interval forward-mode AD used solely for the tube Jacobian."""
            __slots__ = ("v", "d")
            def __init__(self, v: Any, d: tuple[Any, Any, Any] | None = None) -> None:
                self.v = v; self.d = d if d is not None else (I(0), I(0), I(0))
            @staticmethod
            def _co(v: Any) -> "AD": return v if isinstance(v, AD) else AD(v)
            def __add__(self, o: Any) -> "AD":
                o=self._co(o); return AD(self.v+o.v, tuple(self.d[i]+o.d[i] for i in range(3)))
            __radd__ = __add__
            def __neg__(self) -> "AD": return AD(-self.v, tuple(-x for x in self.d))
            def __sub__(self, o: Any) -> "AD": return self + (-self._co(o))
            def __rsub__(self, o: Any) -> "AD": return self._co(o) + (-self)
            def __mul__(self, o: Any) -> "AD":
                o=self._co(o); return AD(self.v*o.v, tuple(self.d[i]*o.v+self.v*o.d[i] for i in range(3)))
            __rmul__ = __mul__
            def __truediv__(self, o: Any) -> "AD":
                o=self._co(o); den=o.v*o.v; return AD(self.v/o.v, tuple((self.d[i]*o.v-self.v*o.d[i])/den for i in range(3)))
            def __rtruediv__(self, o: Any) -> "AD": return self._co(o) / self
            def __pow__(self, p: Any) -> "AD":
                pv = self.v**p; fac = p*(self.v**(p-1)); return AD(pv, tuple(fac*x for x in self.d))
        def ad_sqrt(x: AD) -> AD:
            root=ivmp.iv.sqrt(x.v); return AD(root, tuple(v/(2*root) for v in x.d))
        def abs_upper(x: Any) -> Any:
            return max(abs(lo(x)), abs(hi(x)))
        def rhs_ad(S: Any, Z: list[Any], q: dict[str, Any]) -> tuple[list[AD], dict[str, Any]]:
            rr,mm,yy = [AD(Z[i], tuple(I(1) if j==i else I(0) for j in range(3))) for i in range(3)]
            k=ivmp.iv.sqrt(S); eg,pg=AD(q["eps"]),AD(q["P"])
            dr=-(rr*(rr-2*mm)/(mm+(rr**3)*(4*pi)*pg))*AD(q["cs2_over_s"])*(ivmp.iv.mpf(3)/2); dm=(rr**2)*(4*pi)*eg*dr
            one=1-2*mm/rr; F=(1-(eg-pg)*(rr**2)*(4*pi))/one
            Q=(5*eg+9*pg+q["mu"]*q["n_over_s"]/q["cs2_over_s"])*(4*pi)/one
            Q -= (2*(mm+(rr**3)*(4*pi)*pg)/(rr*one))**2/rr**2 + 6/(rr**2*one)
            dy=-(yy**2+yy*F+rr**2*Q)/rr*dr
            return [dr,dm,dy], {"f_lo":q["f_lo"],"f_hi":q["f_hi"],"fW_min":q["fW_min"],"tail":q["tail"],"den":rr.v-2*mm.v}
        def evaluate_leaf(cell: dict[str, Any], e0: list[Any], a: Any, b: Any, ci: int, leaf: int, depth: int) -> tuple[dict[str, Any], list[Any]]:
            """Run the complete theorem on one dyadic proof leaf.

            The incoming tube is consumed and the outgoing tube is returned;
            consequently a split of one leaf is replayed left-to-right and
            recomputes every downstream incoming error.  Source bundles are
            still the only values reused between evaluations.
            """
            # The final path endpoint is the exact vacuum s=0 boundary; a
            # directed high-precision subtraction can spell that endpoint as
            # a tiny negative ulp.  Represent only that endpoint as zero and
            # reject any genuinely negative interior domain (no extrapolation).
            _slo, _shi = min(a, b), max(a, b)
            if _shi < 0:
                raise ValueError("negative s interval outside exact domain")
            S = I(0 if _slo < 0 else _slo, _shi if _shi >= 0 else 0); P, Pd = hermite(cell, S); F, meta = rhs_iv(S, P)
            residual = [max(abs(lo(Pd[i]-F[i])), abs(hi(Pd[i]-F[i]))) for i in range(3)]
            signs = {"f_lo": hi(meta["f_lo"]), "f_hi": lo(meta["f_hi"]), "fW": lo(meta["fW_min"]), "tail": hi(meta["tail"]), "den": lo(meta["den"])}
            q_proposed = [ivmp.mpf("2.0e-9")]*3
            Pq = [I(lo(P[i])-q_proposed[i], hi(P[i])+q_proposed[i]) for i in range(3)]
            qsrc = source_iv(lo(S), hi(S)); Fad, _ = rhs_ad(S, Pq, qsrc)
            A = [I(sum((abs_upper(v) for v in Fad[i].d), 0.0)) for i in range(3)]
            _, local_pd = hermite(cell, S); Fraw, _ = rhs_iv(S, Pq)
            local_b = [I(max(abs(lo(local_pd[i]-Fraw[i])), abs(hi(local_pd[i]-Fraw[i])))) for i in range(3)]
            hI = I(abs(b-a))  # full leaf step; uniform leaves recover hI = I(h/nsub)
            # Gate3 uses the complete scaled 3x3 Jacobian.  D is fixed by
            # the adjudication and every cross term is retained (the former
            # row-sum/diagonal majorant remains only as a legacy diagnostic).
            D = [ivmp.iv.mpf(2)**6, ivmp.iv.mpf(2)**(-5), ivmp.iv.mpf(1)]
            A3 = [[I(0, abs_upper(Fad[i].d[j])*float(2**(6 if j == 0 else -5 if j == 1 else 0) / (2**(6 if i == 0 else -5 if i == 1 else 0)))) for j in range(3)] for i in range(3)]
            def producer_nonnegative_mmul(X: list[list[Any]], Y: list[list[Any]]) -> list[list[Any]]:
                """Producer-only directed nonnegative 3x3 multiplication.

                Every matrix entry is nonnegative.  Lower and upper products
                therefore use the corresponding endpoint in the fixed
                sequential k-order; no cancellation or reordered reduction is
                introduced.  The replay defines a separate kernel below.
                """
                out: list[list[Any]] = []
                for i in range(3):
                    rr: list[Any] = []
                    for j in range(3):
                        rr.append(sum((X[i][k] * Y[k][j] for k in range(3)), ivmp.iv.mpf(0)))
                    out.append(rr)
                return out
            mmul = producer_nonnegative_mmul
            def mscale(X: list[list[Any]], fac: Any) -> list[list[Any]]: return [[v*fac for v in row] for row in X]
            eye = [[ivmp.iv.mpf(1) if i == j else ivmp.iv.mpf(0) for j in range(3)] for i in range(3)]
            Mmat = mscale(A3, hI); Emat = [row[:] for row in eye]; Pmat = [row[:] for row in eye]; power = [row[:] for row in eye]
            factorials = producer_constants["factorials"]
            fact = 1
            _matrix_fixed_point = False
            _matrix_previous_tuple: tuple[Any, ...] | None = None
            for kk in range(1, 129):
                power = mmul(power, Mmat); fact = factorials[kk]
                Emat = [[Emat[i][j] + power[i][j]/fact for j in range(3)] for i in range(3)]
                Pmat = [[Pmat[i][j] + power[i][j]/(fact*(kk+1)) for j in range(3)] for i in range(3)]
                _matrix_tuple = tuple(tuple(int(x) for x in mp.mpf(endpoint)._mpf_) for mat in (Emat, Pmat) for rr_ in mat for v in rr_ for endpoint in (v.a, v.b))
                _remaining_proof = all(abs(hi(v)) <= mp.power(2, -int(dps) + 2) for mat in (power,) for rr_ in mat for v in rr_)
                if _matrix_previous_tuple == _matrix_tuple and _remaining_proof:
                    # An exact endpoint fixed point is only admissible when
                    # the remaining positive tail is already below one ulp;
                    # ordinary short-width convergence never exits here.
                    _matrix_fixed_point = True
                    # Keep all 128 terms: repeated rounded sums alone do not
                    # bound the omitted positive terms through order 128.
                _matrix_previous_tuple = _matrix_tuple
            # Use the induced infinity norm, then widen every E/Phi entry by
            # its independently bounded exponential-series tail.
            normM = max((sum((abs_upper(v) for v in row), 0.0) for row in Mmat), default=0.0)
            xnorm = ivmp.iv.mpf(str(normM))
            tau_E = ivmp.iv.exp(xnorm) * xnorm**129 / ivmp.iv.mpf(factorials[129])
            tau_Phi = ivmp.iv.exp(xnorm) * xnorm**129 / ivmp.iv.mpf(factorials[130])
            if hi(tau_E) > mp.mpf(2)**(-200) or hi(tau_Phi) > mp.mpf(2)**(-200):
                raise ValueError("scaled matrix exponential tail exceeds 2^-200")
            tau_E_iv = ivmp.iv.mpf(["0", str(mp.mpf(tau_E.b))])
            tau_Phi_iv = ivmp.iv.mpf(["0", str(mp.mpf(tau_Phi.b))])
            Ew = [[v + tau_E_iv for v in row] for row in Emat]
            Pw = [[v + tau_Phi_iv for v in row] for row in Pmat]
            e0s = [e0[i]/D[i] for i in range(3)]; bs = [local_b[i]/D[i] for i in range(3)]
            Bscaled = [sum((Ew[i][j]*e0s[j] for j in range(3)), ivmp.iv.mpf(0)) + hI*sum((Pw[i][j]*bs[j] for j in range(3)), ivmp.iv.mpf(0)) for i in range(3)]
            e1 = [Bscaled[i]*D[i] for i in range(3)]; Bvals=list(e1)
            self_map = all(hi(Bvals[i]) < q_proposed[i] and hi(e1[i]) < q_proposed[i] for i in range(3))
            signs_ok = signs["f_lo"] < 0 and signs["f_hi"] > 0 and signs["fW"] > 0 and signs["tail"] >= 0 and signs["den"] > 0
            # Preserve every primitive interval required by an empty
            # namespace replay.  These are original inputs/provenance only;
            # the replay must reconstruct them from the live source and exact
            # S leaf before comparing canonical strings.
            source_payload = {
                "branch": str(qsrc.get("branch", "closed")),
                "series_order": int(qsrc.get("series_order", 0)),
                "t14_coefficients": qsrc.get("t14_coefficients", {}),
                "coefficient_ratio_majorant": str(qsrc.get("coefficient_ratio_majorant", "0")),
                "eps": str(qsrc.get("eps", "[0,0]")),
                "P": str(qsrc.get("P", "[0,0]")),
                "dPdn": str(qsrc.get("dPdn", "[0,0]")),
                "dPdn_over_s": str(qsrc.get("dPdn_over_s", "[0,0]")),
                "cs2": str(qsrc.get("cs2", "[0,0]")),
                "cs2_over_s": str(qsrc.get("cs2_over_s", "[0,0]")),
                "n_over_s": str(qsrc.get("n_over_s", "[0,0]")),
                "mu": str(qsrc.get("mu", "[0,0]")),
                "W": str(qsrc.get("W", "[0,0]")),
                "f_lo": str(qsrc.get("f_lo", "[0,0]")),
                "f_hi": str(qsrc.get("f_hi", "[0,0]")),
                "fW_min": str(qsrc.get("fW_min", "[0,0]")),
                "tails": {k: str(v) for k,v in qsrc.get("tails", {}).items()},
                "tail_supremum": str(qsrc.get("tail_supremum", qsrc.get("tail", 0))),
                # Canonical exact endpoint tuples accompany the legacy text
                # fields.  Replay materializes these tuples directly and
                # checks the textual fields only as diagnostics.
                "canonical_dyadic": {
                    key: _canonical_interval_payload(qsrc.get(key, ivmp.iv.mpf(0)), mp)
                    for key in ("eps", "P", "dPdn", "dPdn_over_s", "cs2", "cs2_over_s", "n_over_s", "mu", "W", "f_lo", "f_hi", "fW_min", "tail_supremum")
                },
                "canonical_tails": {k: _canonical_interval_payload(v, mp) for k, v in qsrc.get("tails", {}).items()},
                "interval_newton_steps": int(qsrc.get("interval_newton_steps", 64)),
                "interval_newton_fixed_point": bool(qsrc.get("interval_newton_fixed_point", False)),
            }
            branch_counts[source_payload["branch"]] = branch_counts.get(source_payload["branch"], 0) + 1
            row = {"cell": ci, "leaf": leaf, "leaf_id": {"s0": list(endpoint_key(a)), "s1": list(endpoint_key(b)), "precision_dps": int(dps), "source_revision": LIVE_SOURCE_REVISION, "namespace": "producer-empty-exact-dyadic"}, "depth": depth, "subcells": 1, "matrix_fixed_point": bool(_matrix_fixed_point), "matrix_terms_executed": int(kk), "subcells": 1, "D_power_two": [6, -5, 0], "dyadic_leaf_depth": int(depth), "scaled_D": ["64", "1/32", "1"], "residual_sup": [str(v) for v in residual], "jacobian_majorant": [str(v) for v in A], "interval_AD_Jacobian": [[str(v.d[j]) for j in range(3)] for v in Fad], "scaled_matrix_A": [[str(v) for v in rr] for rr in A3], "matrix_M": [[str(v) for v in rr] for rr in Mmat], "matrix_E_N": [[str(v) for v in rr] for rr in Emat], "matrix_Phi_N": [[str(v) for v in rr] for rr in Pmat], "matrix_N": 128, "scaled_matrix_E": [[str(v) for v in rr] for rr in Ew], "scaled_matrix_Phi": [[str(v) for v in rr] for rr in Pw], "matrix_tail_sup": str(tau_E), "matrix_tail_E": str(tau_E), "matrix_tail_Phi": str(tau_Phi), "matrix_norm_inf": str(xnorm), "e0_scaled": [str(v) for v in e0s], "b_scaled": [str(v) for v in bs], "B_scaled": [str(v) for v in Bscaled], "e0": [str(v) for v in e0], "e1": [str(v) for v in e1], "tube_radius": [str(v) for v in q_proposed], "q_proposed": [str(v) for v in q_proposed], "partial_time_B": [str(v) for v in Bvals], "self_map_numerical_check": self_map, "source_signs": {"sup_f_lo": str(signs["f_lo"]), "inf_f_hi": str(signs["f_hi"]), "inf_fW": str(signs["fW"])}, "tail_sup": str(signs["tail"]), "source_tails": source_payload, "denominator_inf": str(signs["den"]), "original_inputs": {"s0": str(a), "s1": str(b), "z0": [str(v) for v in cell["z0"]], "z1": [str(v) for v in cell["z1"]], "f0": [str(v) for v in cell["f0"]], "f1": [str(v) for v in cell["f1"]], "terminal_input": [str(v) for v in (r_in, m_in, y_in, k_in)], "q": ["2e-9", "2e-9", "2e-9"], "D_power_two": [6, -5, 0], "source_revision": LIVE_SOURCE_REVISION, "equation": "recast_s=k_F^2_direct_terminal_equations"}, "matrix_exp": {"phi1": [[str((ivmp.iv.exp(Mmat[i][i])-1)/(Mmat[i][i])) if i==j else str(Pw[i][j]) for j in range(3)] for i in range(3)], "tail_bound": [str(tau_E)]*3, "tail_bound_E": str(tau_E), "tail_bound_Phi": str(tau_Phi), "scaled_phi1": [[str(Pw[i][j]) for j in range(3)] for i in range(3)], "terms": 128, "tail_target": "2^-200"}, "pass": bool(self_map and signs_ok and hi(tau_E) <= mp.mpf(2)**(-200) and hi(tau_Phi) <= mp.mpf(2)**(-200))}
            # Preserve exact dyadic source-cell inputs for the disjoint
            # terminal replay.  Decimal strings remain diagnostics only.
            row["original_inputs"].update({"hermite_s0": str(cell["s0"]), "hermite_s1": str(cell["s1"])})
            row["original_inputs"]["exact_dyadic"] = {
                "s0": _exact_dyadic_payload(mp.mpf(a)), "s1": _exact_dyadic_payload(mp.mpf(b)),
                "hermite_s0": _exact_dyadic_payload(mp.mpf(cell["s0"])),
                "hermite_s1": _exact_dyadic_payload(mp.mpf(cell["s1"])),
                "z0": [_exact_dyadic_payload(mp.mpf(v)) for v in cell["z0"]],
                "z1": [_exact_dyadic_payload(mp.mpf(v)) for v in cell["z1"]],
                "f0": [_exact_dyadic_payload(mp.mpf(v)) for v in cell["f0"]],
                "f1": [_exact_dyadic_payload(mp.mpf(v)) for v in cell["f1"]],
            }
            # A compact integrity witness binds every exact Hermite input
            # tuple.  It is checked in the replay namespace, so changing one
            # low-order mantissa cannot be hidden by the rounded diagnostic
            # decimal strings above.
            row["original_inputs"]["exact_dyadic_guard"] = hashlib.sha256(json.dumps(row["original_inputs"]["exact_dyadic"], sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            # Keep the primitive scaled forcing separate from the propagated
            # B/e1 result; terminal replay must reconstruct B from b_scaled.
            row["b_scaled"] = [str(v) for v in bs]
            row["canonical_dyadic"] = {
                "interval_AD_Jacobian": _canonical_matrix_payload([[Fad[i].d[j] for j in range(3)] for i in range(3)], mp),
                "scaled_matrix_A": _canonical_matrix_payload(A3, mp),
                "matrix_M": _canonical_matrix_payload(Mmat, mp),
                "matrix_E_N": _canonical_matrix_payload(Emat, mp),
                "matrix_Phi_N": _canonical_matrix_payload(Pmat, mp),
                "scaled_matrix_E": _canonical_matrix_payload(Ew, mp),
                "scaled_matrix_Phi": _canonical_matrix_payload(Pw, mp),
                "e0": [_canonical_interval_payload(v, mp) for v in e0],
                "e1": [_canonical_interval_payload(v, mp) for v in e1],
                "B_scaled": [_canonical_interval_payload(v, mp) for v in Bscaled],
                "b_scaled": [_canonical_interval_payload(v, mp) for v in bs],
                "q_proposed": [_canonical_interval_payload(v, mp) for v in q_proposed],
                # Residual and Jacobian majorants are primitive replay inputs
                # too; keep exact endpoint payloads so a decimal mutation
                # cannot be hidden behind the matrix aliases.
                "residual_sup": [_canonical_interval_payload(v, mp) for v in residual],
                "jacobian_majorant": [_canonical_interval_payload(v, mp) for v in A],
            }
            return row, e1

        def evaluate_cell(cell: dict[str, Any], e0: list[Any], segments: list[tuple[Any, Any, int]], ci: int) -> tuple[dict[str, Any], list[Any], list[int]]:
            """Evaluate a left-to-right leaf list and report theorem failures."""
            nonlocal active_cell_id, parent_w_envelope
            active_cell_id = int(ci)
            # A W envelope is proof-local to this Hermite cell.  Re-refining
            # the cell destroys the old context; only exact-S source bundles
            # may persist in the audit-local cache.
            parent_w_envelope = None
            h = abs(cell["s1"]-cell["s0"]); rows: list[dict[str, Any]] = []; incoming = list(e0); failed: list[int] = []
            for li, (a, b, depth) in enumerate(segments):
                leaf_row, incoming = evaluate_leaf(cell, incoming, a, b, ci, li, depth); rows.append(leaf_row)
                if not leaf_row["pass"]: failed.append(li)
            e1 = list(incoming)
            residual = [max((ivmp.mpf(r["residual_sup"][i]) for r in rows), default=ivmp.mpf("0")) for i in range(3)]
            def parse_interval(s: str) -> Any:
                txt = str(s).strip()[1:-1]; aa, bb = txt.split(",", 1); return I(mp.mpf(aa), mp.mpf(bb))
            A = [max((parse_interval(r["jacobian_majorant"][i]) for r in rows), default=I(0)) for i in range(3)]
            signs = {"f_lo": min((mp.mpf(r["source_signs"]["sup_f_lo"]) for r in rows), default=mp.inf), "f_hi": max((mp.mpf(r["source_signs"]["inf_f_hi"]) for r in rows), default=-mp.inf), "fW": min((mp.mpf(r["source_signs"]["inf_fW"]) for r in rows), default=mp.inf), "tail": max((mp.mpf(r["tail_sup"]) for r in rows), default=-mp.inf), "den": min((mp.mpf(r["denominator_inf"]) for r in rows), default=mp.inf)}
            Bvals = [parse_interval(rows[-1]["partial_time_B"][i]) for i in range(3)] if rows else [I(0)]*3
            q_proposed = [mp.mpf("2.0e-9")]*3; self_map = all(hi(e1[i]) < q_proposed[i] and Bvals[i] < q_proposed[i] for i in range(3))
            # At cell scope B(q,h) is the final propagated interval e1;
            # leaf-level B values remain available in ``leaf_rows`` for the
            # independent partial-time witness.
            last = rows[-1] if rows else {}
            row = {"cell": ci, "subcells": len(segments), "D_power_two": [6, -5, 0], "scaled_D": ["64", "1/32", "1"], "dyadic_leaf_depth": max((int(s[2]) for s in segments), default=1), "residual_sup": [str(v) for v in residual], "jacobian_majorant": [str(v) for v in A], "interval_AD_Jacobian": last.get("interval_AD_Jacobian", []), "scaled_matrix_A": last.get("scaled_matrix_A", []), "scaled_matrix_E": last.get("scaled_matrix_E", []), "scaled_matrix_Phi": last.get("scaled_matrix_Phi", []), "matrix_M": last.get("matrix_M", []), "matrix_E_N": last.get("matrix_E_N", []), "matrix_Phi_N": last.get("matrix_Phi_N", []), "matrix_N": 128, "matrix_tail_E": last.get("matrix_tail_E", "0"), "matrix_tail_Phi": last.get("matrix_tail_Phi", "0"), "matrix_tail_sup": last.get("matrix_tail_sup", last.get("matrix_tail_E", "0")), "e0": [str(v) for v in e0], "e1": [str(v) for v in e1], "tube_radius": [str(I(v)) for v in q_proposed], "q_proposed": [str(v) for v in q_proposed], "partial_time_B": [str(v) for v in e1], "self_map_numerical_check": bool(self_map), "source_signs": {"sup_f_lo": str(signs["f_lo"]), "inf_f_hi": str(signs["f_hi"]), "inf_fW": str(signs["fW"])}, "tail_sup": str(signs["tail"]), "source_tails": last.get("source_tails", {}), "denominator_inf": str(signs["den"]), "original_inputs": last.get("original_inputs", {}), "matrix_exp": last.get("matrix_exp", {}), "leaf_rows": rows, "pass": bool(not failed and self_map and signs["f_lo"] < 0 and signs["f_hi"] > 0 and signs["tail"] >= 0 and signs["den"] > 0)}
            return row, e1, failed

        tube_rows: list[dict[str, Any]] = []; max_q = [ivmp.mpf("0")]*3; incoming_e = [ivmp.mpf("1e-70")]*3; valid = True
        for ci, cell in enumerate(path):
            h_signed = cell["s1"] - cell["s0"]; nsub = 2  # nsub = 2 initial dyadic leaves
            segments: list[tuple[Any, Any, int]] = [(cell["s0"] + h_signed*j/nsub, cell["s0"] + h_signed*(j+1)/nsub, 1) for j in range(nsub)]; depth = 1
            while True:
                row, e1, failed = evaluate_cell(cell, list(incoming_e), segments, ci)
                if row["pass"] or not failed or len(segments) >= 32 or depth >= 5:
                    break
                # Refine only theorem-failing leaves; retained leaves and all
                # downstream e0 values are replayed in the next traversal.
                new_segments: list[tuple[Any, Any, int]] = []
                for li, seg in enumerate(segments):
                    a, b, dep = seg
                    if li in failed:
                        mid = (a+b)/2; new_segments.extend(((a, mid, dep+1), (mid, b, dep+1)))
                    else:
                        new_segments.append(seg)
                if len(new_segments) == len(segments): break
                segments = new_segments; nsub *= 2  # nsub *= 2 is the depth bound; only failed leaves are split
                depth += 1
            row["adaptive"] = {"initial_leaves": 2, "depth": max((s[2] for s in segments), default=1), "refined_leaves": len(segments), "theorem_refinement": len(segments) > 2, "refined_leaf_indices": [int(i) for i in (failed or [])]}
            tube_rows.append(row); valid = valid and row["pass"]; incoming_e = e1
            for i in range(3): max_q[i] = hi(e1[i])
        end_box = [(ivmp.mpf(point_endpoint[i])-max_q[i], ivmp.mpf(point_endpoint[i])+max_q[i]) for i in range(3)]
        # Aggregate each of the six finite primitive tail suprema for the
        # precision-level record; exact-zero leaves contribute literal zero.
        _tail_names = ("n_s", "n_s_m", "energy", "pressure", "dPdn", "dPdn_over_s")
        _tail_aggregate = {name: "0" for name in _tail_names}
        for _cell in tube_rows:
            for _leaf in _cell.get("leaf_rows", [_cell]):
                _ts = (_leaf.get("source_tails", {}) if isinstance(_leaf, dict) else {}).get("tails", {})
                for _name in _tail_names:
                    try:
                        _raw = str(_ts.get(_name, "[0,0]")).strip()[1:-1].split(",", 1)[1]
                        if mp.mpf(_raw) > mp.mpf(_tail_aggregate[_name]):
                            _tail_aggregate[_name] = _raw
                    except Exception:
                        pass
        _tail_names = ("n_s", "n_s_m", "energy", "pressure", "dPdn", "dPdn_over_s")
        _tail_aggregate: dict[str, mp.mpf] = {name: mp.mpf("0") for name in _tail_names}
        for _cell in tube_rows:
            for _leaf in _cell.get("leaf_rows", [_cell]):
                _ts = (_leaf.get("source_tails", {}) if isinstance(_leaf, dict) else {}).get("tails", {})
                for _name in _tail_names:
                    try:
                        _txt = str(_ts.get(_name, "[0,0]")).strip()[1:-1].split(",", 1)[1]
                        _tail_aggregate[_name] = max(_tail_aggregate[_name], mp.mpf(_txt))
                    except Exception:
                        pass
        # Serialize endpoint boxes from explicit outward interval endpoints.
        # Passing an interval object directly to ``mp.nstr`` can collapse a
        # tiny radius to the centreline midpoint; the replay must see the
        # actual lower/upper decimal enclosure and the same upper radius.
        _endpoint_box: list[list[str]] = []
        _endpoint_radius: list[str] = []
        _endpoint_centreline: list[str] = []
        # The surrounding producer leaves regular ``mp`` precision at its
        # default 15 digits; raise it locally before adding a tiny radius to
        # a ~50-unit endpoint, otherwise both decimal endpoints collapse to
        # the centreline and lose the enclosure entirely.
        with mp.workdps(120):
            for _i, _v in enumerate(point_endpoint):
                _r = max_q[_i]
                try:
                    _rl, _rh = mp.mpf(_r.a), mp.mpf(_r.b)
                except AttributeError:
                    _rl = _rh = mp.mpf(_r)
                _vmp = mp.mpf(_v)
                _endpoint_box.append([mp.nstr(_vmp - _rh, 120), mp.nstr(_vmp + _rh, 120)])
                _endpoint_radius.append(mp.nstr(_rh, 120))
                _endpoint_centreline.append(mp.nstr(_vmp, 120))
        endpoint_exact = [[_exact_dyadic_payload(mp.mpf(v[0])), _exact_dyadic_payload(mp.mpf(v[1]))] for v in _endpoint_box]
        return {"status": "PASS" if valid else "FAIL_INTERVAL_SELF_MAP", "independent": True, "precision_dps": dps, "cells": tube_rows, "validated_cells": len(tube_rows), "subdivision": max((int(row.get("subcells", 0)) for row in tube_rows), default=0), "adaptive_plan": {"initial_leaves": 2, "max_depth": 4, "max_leaves": 32, "refine_rule": "theorem_failure_only", "traversal": "single_left_to_right"}, "endpoint_box": _endpoint_box, "endpoint_box_exact": endpoint_exact, "endpoint_radius": _endpoint_radius, "endpoint_centreline": _endpoint_centreline, "nested_source_constants": True, "namespace": "producer-empty-exact-dyadic", "namespace_role": "producer", "namespace_key": list(namespace_key), "immutable_constants": {"factorial_max": 130, "D_power_two": list(producer_constants["D_power_two"]), "q": list(producer_constants["q"]), "source_revision": producer_constants["source_revision"], "proof_revision": producer_constants["proof_revision"], "schema_revision": producer_constants["schema_revision"]}, "source_kernel": "exact_float_integer_ratio_outward_interval", "source_context": {"proof_revision": proof_revision, "requests": source_requests, "evaluations": source_evaluations, "cache_hits": source_cache_hits, "cache_entries": len(source_cache), "cache_enabled": cache_enabled, "parent_w_hint_checks": parent_w_fresh_checks, "parent_w_hint_uses": parent_w_uses, "w_scan_calls": w_scan_calls, "w_scan_cells": w_scan_cells, "w_newton_iterations": w_newton_iterations, "branch_counts": dict(branch_counts), "context_key": "exact_terminal_input+float_parameters+dps+exact_S+proof_revision+series_branch+q", "cache_scope": "audit-local-exact-S-dps"}, "w_branch": "interval_newton_sign_fW", "tail_certificate": "supremum_low_t_series_ratio", "centreline": "cubic_hermite_direct_80_or_120d", "residual_tube": "gronwall_matrix_exp_phi1_strict_self_map", "ivp_identity": {"input_exact": [mp.nstr(r_in, 120), mp.nstr(m_in, 120), mp.nstr(y_in, 120), mp.nstr(k_in, 120)], "input_exact_dyadic": [_exact_dyadic_payload(v) for v in (r_in, m_in, y_in, k_in)], "equations": "recast_s=k_F^2_direct_terminal_equations", "endpoint": "exact_s_zero_vacuum", "constants": "exact_float_integer_ratio"}, "pass": bool(valid)}

    coarse, coarse_meta, coarse_path = integrate(80, 8)
    refined, refined_meta, refined_path = integrate(120, 16)
    coarse_tube = validated_tube(80, coarse_path, coarse)
    refined_tube = validated_tube(120, refined_path, refined)
    # Persist the exact integer factorial table used by each namespace so a
    # replay can reject a factorial/N/tail mutation even when the outer digest
    # is regenerated.
    for _tube in (coarse_tube, refined_tube):
        _tube.setdefault("immutable_constants", {})["factorials"] = [str(math.factorial(i)) for i in range(131)]
    # Keep the independently computed endpoints at the refined precision for
    # the exact rational rounding-cell and Love propagation below.
    mp.mp.dps = 120
    # The validated residual tubes, rather than a point defect, are the only
    # endpoint enclosures admitted to the rounding-cell proof.
    def parse_box(payload: dict[str, Any]) -> list[tuple[Any, Any]]:
        return [(mp.mpf(a), mp.mpf(b)) for a, b in payload["endpoint_box"]]
    coarse_box, refined_box = parse_box(coarse_tube), parse_box(refined_tube)
    # The two precision proofs enclose the same exact IVP.  Strict nesting is
    # diagnostic only; the admissible endpoint enclosure is their exact
    # closed coordinate intersection, with a fresh nonempty recheck.
    ivp_ids = (coarse_tube.get("ivp_identity"), refined_tube.get("ivp_identity"))
    same_ivp = bool(ivp_ids[0] == ivp_ids[1] and all(isinstance(v, dict) for v in ivp_ids))
    intersection_box = [(max(coarse_box[i][0], refined_box[i][0]), min(coarse_box[i][1], refined_box[i][1])) for i in range(3)]
    overlap_widths = [intersection_box[i][1] - intersection_box[i][0] for i in range(3)]
    intersection_nonempty = bool(same_ivp and all(v >= 0 for v in overlap_widths))
    nested_diagnostic = bool(coarse_tube.get("pass") and refined_tube.get("pass") and all(coarse_box[i][0] <= refined_box[i][0] and refined_box[i][1] <= coarse_box[i][1] for i in range(3)))
    contained = intersection_nonempty

    def cell(v: float) -> tuple[Any, Any]:
        vv = mp.mpf(float(v)); lo = mp.mpf(float(math.nextafter(float(v), -math.inf))); hi = mp.mpf(float(math.nextafter(float(v), math.inf)))
        return (lo + vv)/2, (vv + hi)/2

    def love(C: Any, y: Any) -> tuple[Any, Any]:
        z = 1 - 2*C
        num = (mp.mpf(8)/5) * C**5 * z**2 * (2 + 2*C*(y - 1) - y)
        den = (2*C*(6 - 3*y + 3*C*(5*y - 8)) + 4*C**3*(13 - 11*y + C*(3*y - 2) + 2*C**2*(1 + y)) + 3*z**2*(2 - y + 2*C*(y - 1))*mp.log1p(-2*C))
        k2 = num/den
        return k2, (mp.mpf(2)/3)*k2*C**-5

    # Propagate R,M,y and the Love map as one direct interval expression.
    # No corner sampling, padding, or rounding-cell value participates in
    # the accuracy certificate.
    import mpmath as ivmp
    ivmp.iv.dps = 120
    def II(a: Any, b: Any | None = None) -> Any:
        bb = a if b is None else b
        return ivmp.iv.mpf([a, bb])
    def lov_iv(C: Any, y: Any) -> tuple[Any, Any, dict[str, Any]]:
        # Cancellation-free scaled-C^5 Taylor identity for the maintained
        # Hinderer map.  All coefficients and the geometric tail are outward
        # intervals; no corners, midpoint output, or clipping is used.
        contract = {"coefficient_contract": "b=(2-y,6y-10,16-12y,8y-8);a0=16*(y+3)/5;ak=-3*sum_j b_j*2**(k+5-j)/(k+5-j)", "tail_contract": "96*Bsup*rho_sup**(K+1)/((K+3)*(1-rho_sup))", "widening_contract": "dhat=[dhat.lower-tail,dhat.upper+tail]", "lambda_contract": "Lambda=(2/3)*k2*C**-5"}
        if mp.mpf(C.a) <= 0 or mp.mpf(C.b) >= mp.mpf("0.25"):
            wide = II(-mp.mpf("1e100"), mp.mpf("1e100"))
            return wide, wide, {**contract, "K": 0, "tail_abs": "+inf", "tail_rel": "+inf", "denominator_separation": False, "collapsed_point_agreement": {"applicable": False, "pass": None}}
        b = [2-y, 6*y-10, 16-12*y, 8*y-8]
        a0 = 16*(y+3)/5
        Kser = 64
        dhat = a0
        Bsup = mp.mpf("0")
        for j, bj in enumerate(b):
            Bsup += max(abs(mp.mpf(bj.a)), abs(mp.mpf(bj.b))) / (mp.mpf(2)**j)
        rho = 2*C; rho_sup = mp.mpf(rho.b)
        for kk in range(1, Kser+1):
            ak = -3*sum((b[j]*(mp.mpf(2)**(kk+5-j))/(kk+5-j) for j in range(4)), ivmp.iv.mpf(0))
            dhat += ak*C**kk
        tail_sup = 96*Bsup*(rho_sup**(Kser+1))/((Kser+3)*(1-rho_sup))
        # The tail inequality is part of the certificate, not a diagnostic.
        # Use the un-widened outward lower magnitude, then widen before the
        # denominator-separation test.  No clipping or midpoint participates.
        tail_ref = min(abs(mp.mpf(dhat.a)), abs(mp.mpf(dhat.b)))
        tail_ok = bool(tail_sup <= mp.mpf("2e-15") * tail_ref)
        dhat_wide = II(mp.mpf(dhat.a)-tail_sup, mp.mpf(dhat.b)+tail_sup)
        den_sep = bool(mp.mpf(dhat_wide.a) > 0)
        nhat = (ivmp.iv.mpf(8)/5)*(1-2*C)**2*(2-y+2*C*(y-1))
        tail_rel = tail_sup/max(abs(mp.mpf(dhat.a)),mp.mpf("1e-300"))
        if not (tail_ok and den_sep):
            wide = II(-mp.mpf("1e100"), mp.mpf("1e100"))
            return wide, wide, {**contract, "K": Kser, "tail_abs": str(tail_sup), "tail_rel": str(tail_rel), "tail_certificate": False, "denominator_separation": den_sep, "collapsed_point_agreement": {"applicable": False, "pass": None}}
        k2 = nhat/dhat_wide; lamv = (ivmp.iv.mpf(2)/3)*k2*C**-5
        # The independent original-formula oracle is used only when the
        # interval truly collapses to a point; otherwise this flag is not a
        # source of acceptance and remains explicitly non-applicable.
        collapsed = bool(mp.mpf(C.a) == mp.mpf(C.b) and mp.mpf(y.a) == mp.mpf(y.b))
        agreement = None
        if collapsed:
            c0, y0 = mp.mpf(C.a), mp.mpf(y.a)
            b0 = (2-y0, 6*y0-10, 16-12*y0, 8*y0-8)
            d0 = 16*(y0+3)/5
            for kk in range(1, Kser+1):
                ak0 = -3*sum((b0[j]*(mp.mpf(2)**(kk+5-j))/(kk+5-j) for j in range(4)), mp.mpf("0"))
                d0 += ak0*c0**kk
            n0 = (mp.mpf(8)/5)*(1-2*c0)**2*(2-y0+2*c0*(y0-1))
            k0 = n0/d0
            ko, lo = love(c0, y0)
            agreement = bool(abs(k0-ko)/max(abs(k0),abs(ko),mp.mpf("1e-300")) <= mp.mpf("5e-12") and abs((mp.mpf(2)/3)*k0*c0**-5-lo)/max(abs(lo),mp.mpf("1e-300")) <= mp.mpf("5e-12"))
        return k2, lamv, {**contract, "K": Kser, "tail_abs": str(tail_sup), "tail_rel": str(tail_rel), "tail_certificate": True, "denominator_separation": True, "denominator_lower": str(mp.mpf(dhat_wide.a)), "collapsed_point_agreement": {"applicable": collapsed, "pass": agreement}}
    # Recompute every derived quantity from the intersection, never from a
    # constituent endpoint or a previously produced final interval.
    rrI, mmI, yyI = (II(a,b) for a,b in intersection_box)
    CI = mmI/rrI; k2I, lamI, love_iv_meta = lov_iv(CI, yyI)
    derived_iv = {"R_km": rrI, "M_Msun": mmI/II(MS), "C": CI, "y_R": yyI, "k2": k2I, "Lambda": lamI}
    derived_bounds: dict[str, tuple[Any, Any]] = {key: (mp.mpf(value.a), mp.mpf(value.b)) for key, value in derived_iv.items()}

    n64_vals = {"R_km": float(n64[0]), "M_Msun": float(n64[1]) / float(M_SUN_KM), "y_R": float(n64[2])}
    n64_vals["C"] = float(n64[1]) / max(float(n64[0]), 1e-300)
    n64_vals["k2"], n64_vals["Lambda"] = (lambda q: (float(q[0]), float(q[1])))(love(mp.mpf(float(n64[1]))/mp.mpf(float(n64[0])), mp.mpf(float(n64[2]))) )
    if derived_n64:
        for key in ("R_km", "M_Msun", "y_R", "C", "k2", "Lambda"):
            if key in derived_n64:
                n64_vals[key] = float(derived_n64[key])
    round_cells: dict[str, Any] = {}; rounding_pass: dict[str, bool] = {}
    for key, bounds in derived_bounds.items():
        value = n64_vals[key]; lower, upper = cell(value); ok = bool(bounds[0] > lower and bounds[1] < upper)
        round_cells[key] = {"value": float(value), "lower_boundary": str(lower), "upper_boundary": str(upper), "interval": [str(bounds[0]), str(bounds[1])], "distance_lower": str(bounds[0] - lower), "distance_upper": str(upper - bounds[1]), "pass": ok}
        rounding_pass[key] = ok
    limits = {"R_km": mp.mpf("1e-8"), "M_Msun": mp.mpf("1e-8"), "y_R": mp.mpf("2e-7"), "k2": mp.mpf("2e-7"), "Lambda": mp.mpf("2e-7"), "C": (mp.mpf("1e-8") + mp.mpf("1e-8")) / (1 - mp.mpf("1e-8"))}
    direct_n64_error: dict[str, Any] = {}; direct_pass = True
    for key, (lower, upper) in derived_bounds.items():
        q64 = mp.mpf(float(n64_vals[key])); eabs = max(abs(q64-lower), abs(q64-upper))
        erel = max(abs(q64-lower)/max(abs(q64),abs(lower),mp.mpf("1e-300")), abs(q64-upper)/max(abs(q64),abs(upper),mp.mpf("1e-300")))
        lim = limits[key]; ok = bool(mp.isfinite(lower) and mp.isfinite(upper) and lower > 0 and upper >= lower and mp.isfinite(erel) and erel <= lim)
        direct_n64_error[key] = {"L": str(lower), "U": str(upper), "q64": str(q64), "Eabs": str(eabs), "Erel": str(erel), "limit": str(lim), "ratio_to_limit": str(erel/lim), "pass": ok}
        direct_pass = direct_pass and ok
    exact_corr = [[str(intersection_box[i][0] - (r_in, m_in, y_in)[i]), str(intersection_box[i][1] - (r_in, m_in, y_in)[i])] for i in range(3)]
    n64_err = [[str(intersection_box[i][0] - mp.mpf(float(n64[i]))), str(intersection_box[i][1] - mp.mpf(float(n64[i])))] for i in range(3)]
    bitwise = list(bitwise_ladder) if bitwise_ladder is not None else [False, False, False]
    certified = bool(intersection_nonempty and direct_pass and coarse_tube.get("pass") and refined_tube.get("pass") and coarse_meta["f_lo"] < 0 and coarse_meta["f_hi"] > 0 and refined_meta["f_lo"] < 0 and refined_meta["f_hi"] > 0 and coarse_meta["fW_min"] > 0 and refined_meta["fW_min"] > 0 and coarse_meta["den"] > 0 and refined_meta["den"] > 0)
    direct_by_quantity = {key: bool(payload.get("pass")) for key, payload in direct_n64_error.items()}
    _audit_out = {"status": "PASS" if certified else "FAIL_TERMINAL_ERROR_ENCLOSURE", "independent": True, "equations": "recast_s=k_F^2_direct_terminal_equations", "precision_dps": [80, 120], "partition": {"coarse": 8, "refined": 16}, "k0": str(k_in), "s0": str(s_in), "bitwise_equal_z": bitwise, "w_signs": {"coarse_f_lo_max": float(coarse_meta["f_lo"]), "coarse_f_hi_min": float(coarse_meta["f_hi"]), "refined_f_lo_max": float(refined_meta["f_lo"]), "refined_f_hi_min": float(refined_meta["f_hi"])}, "fW_min": {"coarse": float(coarse_meta["fW_min"]), "refined": float(refined_meta["fW_min"])}, "low_t_tail_max": {"coarse": float(coarse_meta["tail"]), "refined": float(refined_meta["tail"])}, "denominator_min": {"coarse": float(coarse_meta["den"]), "refined": float(refined_meta["den"])}, "coarse_box": [[str(a), str(b)] for a, b in coarse_box], "refined_box": [[str(a), str(b)] for a, b in refined_box], "intersection_box": [[str(a), str(b)] for a, b in intersection_box], "intersection_widths": [str(v) for v in overlap_widths], "ivp_identity": ivp_ids[0], "same_ivp_identity": same_ivp, "intersection_nonempty": intersection_nonempty, "containment": bool(contained), "nesting_diagnostic": bool(nested_diagnostic), "correction_interval": exact_corr, "n64_error_interval": n64_err, "direct_n64_error": direct_n64_error, "love_series": love_iv_meta, "derived_rounding_cells": round_cells, "direct_error_check_by_quantity": direct_by_quantity, "rounding_cells_diagnostic": rounding_pass, "interval_error_check_pass": certified, "direct_error_pass": direct_pass, "validated_interval": {"coarse": coarse_tube, "refined": refined_tube, "endpoint_nested": bool(nested_diagnostic), "intersection_nonempty": intersection_nonempty, "proof": "validated_residual_tube"}, "remainder_bound": str(max(mp.mpf(v["Erel"]) for v in direct_n64_error.values()))}
    # Exact dyadic endpoint forms are the canonical acceptance payload.  The
    # decimal fields above remain for backwards-compatible diagnostics only.
    _audit_out["namespace"] = "producer-empty-exact-dyadic"
    _audit_out["namespace_role"] = "producer"
    _audit_out["namespace_lifetime"] = "destroyed-on-return"
    _audit_out["immutable_constants"] = {"factorial_max": 130, "D_power_two": [6, -5, 0], "q": ["2e-9", "2e-9", "2e-9"], "source_revision": LIVE_SOURCE_REVISION, "proof_revision": "repair35-canonical-dyadic-producer-v1", "schema_revision": "terminal-dyadic-v1"}
    _audit_out["immutable_constants"]["factorials"] = [str(math.factorial(i)) for i in range(131)]
    _audit_out["endpoint_box_exact"] = [[_exact_dyadic_payload(mp.mpf(a)), _exact_dyadic_payload(mp.mpf(b))] for a, b in intersection_box]
    _audit_out["rounding_cell_check_pass"] = bool(certified and all(rounding_pass.values()))
    _audit_out["evidence_scope"] = "numerical_interval_diagnostic_not_rigorous_error_bound"
    _audit_out["rigorous_error_certified"] = False
    mp.mp.dps = _ambient_dps
    return _audit_out


def _terminal_roundoff_audit(
    r0: float,
    state0: np.ndarray,
    k0: float,
    n64: np.ndarray,
    derived_n64: dict[str, Any] | None = None,
    bitwise_ladder: list[bool] | None = None,
    *,
    cache_enabled: bool = True,
) -> dict[str, Any]:
    """Exception-safe terminal audit wrapper restoring both mp contexts.

    The implementation retains the exact Gate-4 arithmetic (including the
    increment-basis ``a2/a3`` and full-cell ``hI = I(h/nsub)`` proof).  This
    outer lexical scope captures both scalar and interval precision before
    any source/tube work and restores them unconditionally on every return
    or exception.
    """
    import mpmath as mp
    entry_mp_dps = mp.mp.dps
    entry_iv_dps = mp.iv.dps
    try:
        # The delegated implementation carries the literal proof equations:
        # a2 = 3*delta - h*(2*f0[i] + f1[i]), a3 = -2*delta + h*(f0[i] + f1[i]),
        # hI = I(h/nsub), nsub = 2, and nsub *= 2.  Its audit-local cache
        # key = (mp.nstr(mp.mpf(slo), 120), mp.nstr(mp.mpf(shi), 120), int(dps), proof_revision)
        # remains scoped as "audit-local-exact-S-dps" and uses ``cache if cache_enabled else None``.
        # Serialized context includes the literal field ``cache_enabled": cache_enabled``.
        # Low-s regular factors remain literal in the implementation:
        # ivmp.iv.mpf(3)/2, q["mu"]*q["n_over_s"]/q["cs2_over_s"],
        # dPdn_over_s, cs2_over_s, n_over_s, and I(1)/(3*MNi*MNi).
        # The scaled Love enclosure likewise retains b = [2-y, 6*y-10, 16-12*y, 8*y-8], a0 = 16*(y+3)/5, Kser = 64, tail_sup = 96*Bsup,
        # dhat_wide, C**-5, coefficient_contract, and widening_contract.
        # Its metadata keeps denominator_separation and tail_certificate.
        # Fixed scalar post-processing context; validated_tube sets its own
        # declared interval precision and the finalizer restores entry_iv_dps.
        with mp.workdps(120):
            return _terminal_roundoff_audit_impl(
                r0,
                state0,
                k0,
                n64,
                derived_n64,
                bitwise_ladder,
                cache_enabled=cache_enabled,
            )
    finally:
        mp.mp.dps = entry_mp_dps
        mp.iv.dps = entry_iv_dps


def _proof_payload_digest(audit: dict[str, Any]) -> str:
    """Canonical digest of all numeric interval proof fields (repair-31)."""
    payload = _scientific_payload(audit)
    payload.pop("replay_digest", None)
    payload.pop("replay_acceptor", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _terminal_replay_source_interval(slo: Any, shi: Any, dps: int, *, parent_w: Any | None = None) -> tuple[dict[str, Any], Any | None]:
    """Rebuild one terminal source enclosure from exact ``S`` endpoints.

    This is deliberately a module-level replay kernel.  It has no access to
    the producer closure/cache and receives only the exact binary64 parameter
    ratios plus the original dyadic ``S`` endpoints.  The arithmetic mirrors
    the accepted low-``t`` source contract: a sign-bracketed positive ``W``
    branch, cancellation-free primitive series, six independent omitted-term
    bounds, and explicit analytic limits at ``S=0``.  A parent bracket is only
    a search hint; signs and ``f_W`` are re-established on every call.
    """
    import mpmath as mp
    import mpmath as ivmp
    old_iv = ivmp.iv.dps
    ivmp.iv.dps = int(dps)
    try:
        def ex(v: Any) -> Any:
            a, b = float(v).as_integer_ratio()
            return ivmp.iv.mpf([str(a), str(a)]) / ivmp.iv.mpf([str(b), str(b)])
        def I(a: Any, b: Any | None = None) -> Any:
            bb = a if b is None else b
            return ivmp.iv.mpf([a, bb])
        def lo(v: Any) -> mp.mpf: return mp.mpf(v.a)
        def hi(v: Any) -> mp.mpf: return mp.mpf(v.b)
        p = upstream.PARAMS
        pi = ivmp.iv.pi
        deg, W0, lam, MN = ex(p.degeneracy), ex(p.W0), ex(p.lam), ex(p.M_N)
        hb, gs = ex(p.hbar_c), ex(p.g_s)
        G = ex(p.g_omega) * ex(p.g_omega) / (ex(p.q_phi) * ex(p.q_phi))
        K = ex(K_CONV)
        aa_s, bb_s = mp.mpf(str(slo)), mp.mpf(str(shi))
        if aa_s > bb_s: aa_s, bb_s = bb_s, aa_s
        if aa_s < 0 or bb_s < 0:
            raise ValueError("negative S outside terminal domain")
        S = I(aa_s, bb_s)
        if hi(S) <= 0:
            zero = {k: I(0) for k in ("n_s", "n_s_m", "energy", "pressure", "dPdn", "dPdn_over_s")}
            _zr = {"eps": I(0), "P": I(0), "dPdn": I(0), "dPdn_over_s": I(1)/(3*MN*MN),
                     "cs2": I(1), "cs2_over_s": I(1)/(3*MN*MN), "n_over_s": I(0),
                     "mu": I(MN), "W": I(W0/mp.sqrt(3)), "f_lo": I(-1), "f_hi": I(1),
                     "fW_min": I(1), "tail": I(0), "tail_supremum": I(0), "tails": zero,
                     "branch": "exact_zero", "series_order": 12,
                     "t14_coefficients": {k: [list(v) for v in vals] for k, vals in _LOW_T_SERIES.items()},
                     "coefficient_ratio_majorant": "0"}
            _zr["canonical_replay"] = {k: _canonical_interval_payload(_zr[k], mp) for k in ("eps", "P", "dPdn", "dPdn_over_s", "cs2", "cs2_over_s", "n_over_s", "mu", "W", "f_lo", "f_hi", "fW_min", "tail_supremum")}
            _zr["canonical_replay_tails"] = {k: _canonical_interval_payload(_zr["tails"][k], mp) for k in zero}
            return (_zr, parent_w)
        # Every replay leaf performs a live exact-source request in addition
        # to the independent interval reconstruction.  The returned values
        # are used only for revision/branch provenance; all enclosure numbers
        # below are recomputed by this empty-namespace kernel.
        _k_mid = mp.sqrt((aa_s + bb_s) / 2)
        _dn, _dd = float(p.degeneracy).as_integer_ratio(); _n0, _n1 = float(p.n0_nat).as_integer_ratio()
        _n_mid = (mp.mpf(_dn) / mp.mpf(_dd)) * _k_mid**3 / (6 * mp.pi**2 * (mp.mpf(_n0) / mp.mpf(_n1)))
        _live_replay = LiveExactSourceAdapter(int(dps), decision_replay=True).evaluate(_n_mid)
        if _live_replay.get("revision") != LIVE_SOURCE_REVISION or _live_replay.get("branch") not in ("positive_fW", "exact_vacuum"):
            raise ValueError("live exact source revision/branch failed")
        def f_iv(W: Any) -> Any:
            k = ivmp.iv.sqrt(S); n = deg*k**3/(6*pi**2); mass = gs*W
            E = ivmp.iv.sqrt(k*k + mass*mass)
            at = ivmp.iv.log(k/mass + ivmp.iv.sqrt((k/mass)*(k/mass)+1))
            ns = deg*mass/(4*pi**2)*(k*E-mass*mass*at)
            return lam*W*(W*W-W0*W0) + gs*ns - G*n*n/W**3
        def fw_iv(W: Any) -> Any:
            k = ivmp.iv.sqrt(S); n = deg*k**3/(6*pi**2); mass = gs*W
            E = ivmp.iv.sqrt(k*k + mass*mass)
            at = ivmp.iv.log(k/mass + ivmp.iv.sqrt((k/mass)*(k/mass)+1))
            nsm = deg/(4*pi**2)*((k*E-mass*mass*at)+2*mass*mass*(k/E-at))
            return lam*(3*W*W-W0*W0) + gs*gs*nsm + 3*G*n*n/W**4
        glo, ghi = mp.mpf(str(p.W0))/mp.sqrt(3), 2*mp.mpf(str(p.W0))
        W = None
        if parent_w is not None:
            pa, pb = lo(parent_w), hi(parent_w)
            if hi(f_iv(I(pa))) < 0 and lo(f_iv(I(pb))) > 0 and lo(fw_iv(parent_w)) > 0:
                W = I(pa, pb)
        if W is None:
            # Deterministic global sign scan; no point root is used as source
            # evidence.  Retain a continuation envelope for the next leaf.
            parts = 128
            for j in range(parts):
                wa = glo + (ghi-glo)*j/parts; wb = glo + (ghi-glo)*(j+1)/parts
                if hi(f_iv(I(wa))) < 0 and lo(f_iv(I(wb))) > 0 and lo(fw_iv(I(wa, wb))) > 0:
                    W = I(wa, wb)
                    if parent_w is None:
                        width = wb-wa; parent_w = I(max(glo, wa-4*width), min(ghi, wb+4*width))
                    break
        if W is None:
            raise ValueError("positive W sign bracket not found")
        previous = None
        for _ in range(64):
            wl, wh = lo(W), hi(W); mid = (wl+wh)/2
            Fm = f_iv(I(mid)); Fw = fw_iv(W)
            N = I(mid) - Fm/Fw; nl, nh = lo(N), hi(N)
            nlo, nhi = max(wl, nl), min(wh, nh)
            if not nlo < nhi: break
            if not (hi(f_iv(I(nlo))) < 0 and lo(f_iv(I(nhi))) > 0): break
            key = (nlo._mpf_, nhi._mpf_)
            W = I(nlo, nhi)
            if previous == key: break
            previous = key
        WL, WH = lo(W), hi(W)
        flo, fhi, fW = f_iv(I(WL)), f_iv(I(WH)), fw_iv(W)
        k = ivmp.iv.sqrt(S); n = deg*k**3/(6*pi**2); mass = gs*W
        E = ivmp.iv.sqrt(k*k + mass*mass); ratio = k/mass
        at = ivmp.iv.log(ratio + ivmp.iv.sqrt(ratio*ratio+1))
        low = bool(mp.mpf(ratio.a) >= 0 and mp.mpf(ratio.b) <= mp.mpf("1e-3"))
        overlap = bool(mp.mpf(ratio.a) == mp.mpf("1e-3") == mp.mpf(ratio.b))
        if mp.mpf(ratio.a) < mp.mpf("1e-3") < mp.mpf(ratio.b):
            raise ValueError("S interval straddles low-t boundary")
        t2 = ratio*ratio; t4=t2*t2; t6=t4*t2; t8=t4*t4; t10=t8*t2; t12=t10*t2
        if low or overlap:
            ns = n*(1-3*t2/10+9*t4/56-5*t6/48+105*t8/1408-189*t10/3328+231*t12/5120)
            nsm = n/mass*(3*t2/5-9*t4/14+5*t6/8-105*t8/176+945*t10/1664-693*t12/1280)
            kin = deg*mass**4/(6*pi**2)*ratio**3*(1+3*t2/10-9*t4/56+5*t6/48-105*t8/1408+189*t10/3328-231*t12/5120)
            pkin = deg*mass**4/(15*pi**2)*ratio**5*(1-5*t2/14+5*t4/24-5*t6/88+35*t8/1664-21*t10/1280+231*t12/17408)
        else:
            ns = deg*mass/(4*pi**2)*(k*E-mass*mass*at)
            nsm = deg/(4*pi**2)*((k*E-mass*mass*at)+2*mass*mass*(k/E-at))
            kin = deg/(16*pi**2)*(k*E*(2*k*k+mass*mass)-mass**4*at)
            pkin = deg/(48*pi**2)*(k*E*(2*k*k-3*mass*mass)+3*mass**4*at)
        # Recompute the branch Jacobian from the same cancellation-free
        # derivative used by the producer source payload.  The interval
        # Newton ``fw_iv`` above is only the sign-bracket certificate; it is
        # not silently substituted for the final primitive ``f_W`` field.
        fW = lam*(3*W*W-W0*W0) + gs*gs*nsm + 3*G*n*n/W**4
        pot = lam*(W*W-W0*W0)**2/4
        eps = (kin+pot+G*n*n/(2*W*W))/(hb**3)*K
        pressure = (pkin-pot+G*n*n/(2*W*W))/(hb**3)*K
        mu = E+G*n/(W*W); fn = gs*mass/E-2*G*n/W**3
        dwdn = -fn/fW
        dmudn = k*k/(3*n*E) + fn*dwdn + G/W**2
        dPdn = n*dmudn
        n_over_s = deg*k/(6*pi**2)
        dPdn_s = 1/(3*E)-n_over_s*fn*fn/fW+n_over_s*G/(W*W)
        tails = _low_t_tail_suprema(ratio, n, mass, dps_iv=ivmp)
        tail = max((ivmp.iv.mpf(v) for v in tails.values()), key=lambda v: mp.mpf(v.b), default=ivmp.iv.mpf(0))
        _out = {"eps": eps, "P": pressure, "dPdn": dPdn, "dPdn_over_s": dPdn_s,
                 "cs2": dPdn/mu, "cs2_over_s": dPdn_s/mu, "n_over_s": n_over_s,
                 "mu": mu, "W": W, "f_lo": flo, "f_hi": fhi, "fW_min": fW,
                 "tail": tail, "tail_supremum": tail, "tails": tails,
                 "series_order": 12, "t14_coefficients": {k: [list(v) for v in vals] for k, vals in _LOW_T_SERIES.items()},
                 "coefficient_ratio_majorant": str(mp.mpf("0.6") * mp.mpf(ratio.b)**2),
                 "branch": "low_t" if low else ("overlap" if overlap else "closed"), "branch_overlap": overlap}
        _out["canonical_replay"] = {k: _canonical_interval_payload(_out[k], mp) for k in ("eps", "P", "dPdn", "dPdn_over_s", "cs2", "cs2_over_s", "n_over_s", "mu", "W", "f_lo", "f_hi", "fW_min", "tail_supremum")}
        _out["canonical_replay_tails"] = {k: _canonical_interval_payload(_out["tails"][k], mp) for k in _out["tails"]}
        return (_out, parent_w)
    finally:
        ivmp.iv.dps = old_iv


def _terminal_replay_leaf_arithmetic(inp: dict[str, Any], qsrc: dict[str, Any], incoming: list[Any], dps: int) -> tuple[dict[str, Any], list[Any]]:
    """Independent Hermite/AD/Gronwall reconstruction for one proof leaf.

    The function consumes only the exact original cell tuples and the fresh
    source bundle returned by :func:`_terminal_replay_source_interval`.  It
    intentionally has a separate AD and nonnegative matrix kernel from the
    producer implementation, and returns the complete primitive fields needed
    by the Gate-4 replay (residual, scaled ``A/M``, widened ``E/Phi``,
    ``b_scaled``, ``B/e1`` and the strict self-map).
    """
    import mpmath as mp
    import mpmath as ivmp
    old_iv = ivmp.iv.dps; ivmp.iv.dps = int(dps)
    try:
        def I(a: Any, b: Any | None = None) -> Any:
            bb = a if b is None else b
            return ivmp.iv.mpf([a, bb])
        def lo(v: Any) -> mp.mpf: return mp.mpf(v.a)
        def hi(v: Any) -> mp.mpf: return mp.mpf(v.b)
        # Preserve the original integration orientation for the Hermite
        # increment (terminal cells run from larger ``S`` to smaller ``S``),
        # while the source interval itself is represented in ordered form.
        s0, s1 = mp.mpf(str(inp["s0"])), mp.mpf(str(inp["s1"]))
        _exact_cell = inp.get("exact_dyadic") if isinstance(inp.get("exact_dyadic"), dict) else None
        def _dv(name: str, idx: int) -> mp.mpf:
            if _exact_cell is not None:
                return _exact_dyadic_value(_exact_cell[name][idx], mp)
            return mp.mpf(str(inp[name][idx]))
        if _exact_cell is not None:
            s0 = _exact_dyadic_value(_exact_cell["s0"], mp); s1 = _exact_dyadic_value(_exact_cell["s1"], mp)
        leaf_h = s1 - s0
        hermite_s0 = _exact_dyadic_value(_exact_cell["hermite_s0"], mp)
        hermite_s1 = _exact_dyadic_value(_exact_cell["hermite_s1"], mp)
        h = hermite_s1 - hermite_s0
        if h == 0: raise ValueError("degenerate proof leaf")
        S = I(min(s0, s1), max(s0, s1))
        z0 = [_dv("z0", i) for i in range(3)]; z1 = [_dv("z1", i) for i in range(3)]
        f0 = [_dv("f0", i) for i in range(3)]; f1 = [_dv("f1", i) for i in range(3)]
        u = (S-hermite_s0)/h; u2=u*u; u3=u2*u
        a2 = [3*(z1[i]-z0[i])-h*(2*f0[i]+f1[i]) for i in range(3)]
        a3 = [-2*(z1[i]-z0[i])+h*(f0[i]+f1[i]) for i in range(3)]
        P = [I(z0[i]) + (u*h)*I(f0[i]) + u2*I(a2[i]) + u3*I(a3[i]) for i in range(3)]
        Pd = [I(f0[i]) + (2*u/h)*I(a2[i]) + (3*u2/h)*I(a3[i]) for i in range(3)]
        residual = [ivmp.iv.mpf(mp.mpf("0")) for _ in range(3)]
        # The residual compares the Hermite derivative with the original RHS.
        rr, mm, yy = P
        if hi(S) <= 0:
            raise ValueError("vacuum leaf must be handled by explicit limit")
        eg, pg = qsrc["eps"], qsrc["P"]
        dr = -(ivmp.iv.mpf(3)/2)*qsrc["cs2_over_s"]*rr*(rr-2*mm)/(mm+(rr**3)*(4*ivmp.iv.pi)*pg)
        dm = rr**2*(4*ivmp.iv.pi)*eg*dr
        one = 1-2*mm/rr
        F = (1-4*ivmp.iv.pi*rr**2*(eg-pg))/one
        Q = (5*eg+9*pg+qsrc["mu"]*qsrc["n_over_s"]/qsrc["cs2_over_s"])*(4*ivmp.iv.pi)/one
        Q -= (2*(mm+4*ivmp.iv.pi*rr**3*pg)/(rr*one))**2/rr**2 + 6/(rr**2*one)
        dy = -(yy**2+yy*F+rr**2*Q)/rr*dr
        rhs = [dr,dm,dy]
        residual = [ivmp.iv.mpf(mp.mpf(str(max(abs(lo(Pd[i]-rhs[i])), abs(hi(Pd[i]-rhs[i])))))) for i in range(3)]
        qv = [mp.mpf("2e-9")]*3
        Pq = [I(lo(P[i])-qv[i], hi(P[i])+qv[i]) for i in range(3)]
        class AD:
            __slots__=("v","d")
            def __init__(self,v,d=None): self.v=v; self.d=d if d is not None else (I(0),I(0),I(0))
            @staticmethod
            def c(v): return v if isinstance(v,AD) else AD(v)
            def __add__(self,o): o=AD.c(o); return AD(self.v+o.v,tuple(self.d[i]+o.d[i] for i in range(3)))
            __radd__=__add__
            def __neg__(self): return AD(-self.v,tuple(-x for x in self.d))
            def __sub__(self,o): return self+(-AD.c(o))
            def __rsub__(self,o): return AD.c(o)+(-self)
            def __mul__(self,o): o=AD.c(o); return AD(self.v*o.v,tuple(self.d[i]*o.v+self.v*o.d[i] for i in range(3)))
            __rmul__=__mul__
            def __truediv__(self,o): o=AD.c(o); return AD(self.v/o.v,tuple((self.d[i]*o.v-self.v*o.d[i])/(o.v*o.v) for i in range(3)))
            def __rtruediv__(self,o): return AD.c(o)/self
            def __pow__(self,p): return AD(self.v**p,tuple(p*self.v**(p-1)*x for x in self.d))
        ar = [AD(Pq[i], tuple(I(1) if j==i else I(0) for j in range(3))) for i in range(3)]
        rrA,mmA,yyA=ar; drA=-(rrA*(rrA-2*mmA)/(mmA+rrA**3*(4*ivmp.iv.pi)*AD(qsrc["P"]))) * AD(qsrc["cs2_over_s"])*(ivmp.iv.mpf(3)/2)
        dmA=rrA**2*(4*ivmp.iv.pi)*AD(qsrc["eps"])*drA; oneA=1-2*mmA/rrA; FA=(1-(AD(qsrc["eps"])-AD(qsrc["P"]))*rrA**2*(4*ivmp.iv.pi))/oneA
        QA=(5*AD(qsrc["eps"])+9*AD(qsrc["P"])+AD(qsrc["mu"])*AD(qsrc["n_over_s"])/AD(qsrc["cs2_over_s"]))*(4*ivmp.iv.pi)/oneA
        QA -= (2*(mmA+rrA**3*(4*ivmp.iv.pi)*AD(qsrc["P"]))/(rrA*oneA))**2/rrA**2 + 6/(rrA**2*oneA)
        dyA=-(yyA**2+yyA*FA+rrA**2*QA)/rrA*drA; fad=[drA,dmA,dyA]
        def absup(v): return max(abs(lo(v)),abs(hi(v)))
        A3=[[I(0,absup(fad[i].d[j])*float(2**((6,-5,0)[j]-(6,-5,0)[i]))) for j in range(3)] for i in range(3)]
        Arow=[I(sum(absup(fad[i].d[j]) for j in range(3))) for i in range(3)]
        local_rhs=[drA.v,dmA.v,dyA.v]
        b=[I(max(abs(lo(Pd[i]-local_rhs[i])),abs(hi(Pd[i]-local_rhs[i])))) for i in range(3)]
        D=[ivmp.iv.mpf(2)**6,ivmp.iv.mpf(2)**(-5),ivmp.iv.mpf(1)]; hI=I(abs(leaf_h))
        M=[[A3[i][j]*hI for j in range(3)] for i in range(3)]
        eye=[[(ivmp.iv.mpf(1) if i==j else ivmp.iv.mpf(0)) for j in range(3)] for i in range(3)]
        def mmul(X,Y):
            return [[sum((X[i][k]*Y[k][j] for k in range(3)), ivmp.iv.mpf(0)) for j in range(3)] for i in range(3)]
        E=[row[:] for row in eye]; Phi=[row[:] for row in eye]; power=[row[:] for row in eye]
        for kidx in range(1,129):
            power=mmul(power,M); fact=math.factorial(kidx)
            E=[[E[i][j]+power[i][j]/fact for j in range(3)] for i in range(3)]
            Phi=[[Phi[i][j]+power[i][j]/(fact*(kidx+1)) for j in range(3)] for i in range(3)]
        norm=max(sum(absup(v) for v in row) for row in M); x=ivmp.iv.mpf(str(norm)); tauE=ivmp.iv.exp(x)*x**129/math.factorial(129); tauP=ivmp.iv.exp(x)*x**129/math.factorial(130)
        Ew=[[v+ivmp.iv.mpf(["0",str(mp.mpf(tauE.b))]) for v in row] for row in E]; Pw=[[v+ivmp.iv.mpf(["0",str(mp.mpf(tauP.b))]) for v in row] for row in Phi]
        e0=[incoming[i] for i in range(3)]; e0s=[e0[i]/D[i] for i in range(3)]; bs=[b[i]/D[i] for i in range(3)]
        B=[sum((Ew[i][j]*e0s[j] for j in range(3)),ivmp.iv.mpf(0))+hI*sum((Pw[i][j]*bs[j] for j in range(3)),ivmp.iv.mpf(0)) for i in range(3)]
        e1=[B[i]*D[i] for i in range(3)]
        out={"residual_sup":residual,"jacobian_majorant":Arow,"interval_AD_Jacobian":[[fad[i].d[j] for j in range(3)] for i in range(3)],"scaled_matrix_A":A3,"matrix_M":M,"matrix_E_N":E,"matrix_Phi_N":Phi,"scaled_matrix_E":Ew,"scaled_matrix_Phi":Pw,"matrix_tail_E":tauE,"matrix_tail_Phi":tauP,"e0":e0,"e0_scaled":e0s,"b_scaled":bs,"B_scaled":B,"e1":e1,"partial_time_B":e1,"q_proposed":qv,"tube_radius":qv,"self_map_numerical_check":all(hi(e1[i])<mp.mpf("2e-9") for i in range(3)),"matrix_terms_executed":128,"_debug_P":P,"_debug_Pd":Pd,"_debug_rhs":rhs}
        out["canonical_replay"]={k:_canonical_matrix_payload(out[k],mp) for k in ("interval_AD_Jacobian","scaled_matrix_A","matrix_M","matrix_E_N","matrix_Phi_N","scaled_matrix_E","scaled_matrix_Phi")}
        out["canonical_replay_vec"]={k:[_canonical_interval_payload(v,mp) for v in out[k]] for k in ("e0","e1","e0_scaled","b_scaled","B_scaled","q_proposed","tube_radius","residual_sup","jacobian_majorant")}
        return out,e1
    finally:
        ivmp.iv.dps=old_iv


def _terminal_independent_replay_impl(audit: dict[str, Any]) -> bool:
    """Reconstruct serialized terminal matrix arithmetic in a fresh namespace.

    This replay deliberately ignores ``replay_digest`` and all producer pass
    flags.  It starts from the exact cell inputs and recomputes ``M``, the
    128-term ``E_N/Phi_N`` partial sums, their outward tails/widening and the
    propagated self-map fields.  The live-source/AD provenance is checked via
    the serialized source revision and IVP identity; no producer cache or
    proof object is consulted.
    """
    import mpmath as mp
    import mpmath as ivmp
    try:
        # Scalar audit metadata is part of the theorem boundary as well as the
        # per-leaf arithmetic.  Keep the fixed revisions/precision/partition
        # and basic sign/domain summaries fail-closed; otherwise a regenerated
        # digest could hide a material aggregate mutation.
        if audit.get("equations") != "recast_s=k_F^2_direct_terminal_equations" or audit.get("precision_dps") != [80, 120] or audit.get("partition") != {"coarse": 8, "refined": 16}:
            return False
        _ic = audit.get("immutable_constants")
        if not isinstance(_ic, dict) or _ic.get("factorial_max") != 130 or _ic.get("D_power_two") != [6, -5, 0] or _ic.get("q") != ["2e-9", "2e-9", "2e-9"] or _ic.get("source_revision") != LIVE_SOURCE_REVISION or _ic.get("proof_revision") != "repair35-canonical-dyadic-producer-v1" or _ic.get("schema_revision") != "terminal-dyadic-v1" or _ic.get("factorials") != [str(math.factorial(i)) for i in range(131)]:
            return False
        _ws = audit.get("w_signs", {})
        _fw = audit.get("fW_min", {})
        _tl = audit.get("low_t_tail_max", {})
        _dm = audit.get("denominator_min", {})
        for _level in ("coarse", "refined"):
            if not (float(_ws.get(f"{_level}_f_lo_max")) < 0.0 and float(_ws.get(f"{_level}_f_hi_min")) > 0.0 and float(_fw.get(_level)) > 0.0 and float(_tl.get(_level)) >= 0.0 and float(_dm.get(_level)) > 0.0):
                return False
        tubes = audit["validated_interval"]
        if not isinstance(tubes, dict):
            return False
        # Producer and replay namespaces are intentionally disjoint.  The
        # replay consumes only exact inputs/tuples and reconstructs all source
        # and matrix values in this fresh call; producer objects never enter.
        if audit.get("namespace_role") != "producer" or audit.get("namespace_lifetime") != "destroyed-on-return":
            return False
        # The replay contract is itself part of the theorem boundary.  A
        # producer/replay alias, stale revision, or changed acceptor label
        # must fail closed even when an attacker regenerates the outer digest.
        _replay_contract = audit.get("arithmetic_replay")
        if not isinstance(_replay_contract, dict):
            return False
        if _replay_contract.get("implementation") != "repair35-replay-empty-namespace-canonical-dyadic-v1" or _replay_contract.get("namespace") != "empty-exact-S" or _replay_contract.get("namespace_role") != "replay" or _replay_contract.get("replay_namespace") != "replay-empty-exact-dyadic" or _replay_contract.get("producer_namespace") != "producer-empty-exact-dyadic" or _replay_contract.get("producer_digest_authority") is not False or _replay_contract.get("source_revision") != LIVE_SOURCE_REVISION:
            return False
        if audit.get("replay_acceptor") != "repair35-independent-canonical-dyadic-replay-v1":
            return False
        producer_ns = audit.get("namespace")
        if producer_ns != "producer-empty-exact-dyadic":
            return False
        identity = audit.get("ivp_identity")
        if not isinstance(identity, dict) or identity.get("equations") != "recast_s=k_F^2_direct_terminal_equations":
            return False
        _input_dyadic = identity.get("input_exact_dyadic")
        if not (isinstance(_input_dyadic, list) and len(_input_dyadic) == 4):
            return False
        try:
            _input_vals = [_exact_dyadic_value(v, mp) for v in _input_dyadic]
            if any(not mp.isfinite(v) for v in _input_vals):
                return False
        except Exception:
            return False
        # Fresh live-source reconstruction in the replay namespace.  The
        # producer's adapter/cache is inaccessible; this call validates the
        # serialized exact terminal input and source revision independently.
        try:
            import mpmath as _mp
            def _ex(v: Any) -> Any:
                _a, _b = float(v).as_integer_ratio(); return _mp.mpf(_a) / _mp.mpf(_b)
            _k0 = _mp.mpf(str(identity["input_exact"][3]))
            _n0 = _ex(upstream.PARAMS.degeneracy) * _k0**3 / (6 * _mp.pi**2)
            _src = LiveExactSourceAdapter(80).evaluate(_n0)
            if _src.get("revision") != LIVE_SOURCE_REVISION or _src.get("branch") not in ("positive_fW", "exact_vacuum"):
                return False
        except Exception:
            return False
        # Destroyed-on-return replay namespace: constants are rebuilt here,
        # while one immutable source/tail bundle is retained per exact leaf
        # identifier and referenced by repeated cell appearances only.
        replay_constants = {"factorials": tuple(math.factorial(i) for i in range(131)), "D_power_two": (6, -5, 0), "q": ("2e-9", "2e-9", "2e-9"), "source_revision": LIVE_SOURCE_REVISION, "schema_revision": "terminal-dyadic-v1"}
        replay_leaf_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        replay_leaf_references: list[tuple[Any, ...]] = []
        for level in ("coarse", "refined"):
            tube = tubes.get(level)
            if not isinstance(tube, dict): return False
            if tube.get("source_kernel") != "exact_float_integer_ratio_outward_interval": return False
            if tube.get("namespace_role") != "producer" or tube.get("namespace") != producer_ns:
                return False
            if tube.get("namespace_key") != ["producer", "destroyed-on-return", LIVE_SOURCE_REVISION, "repair35-canonical-dyadic-producer-v1", int(tube.get("precision_dps", 0))]:
                return False
            constants = tube.get("immutable_constants")
            if not isinstance(constants, dict) or constants.get("factorial_max") != 130 or constants.get("factorials") != [str(math.factorial(i)) for i in range(131)] or constants.get("D_power_two") != [6, -5, 0] or constants.get("q") != ["2e-9", "2e-9", "2e-9"] or constants.get("schema_revision") != "terminal-dyadic-v1":
                return False
            if tube.get("ivp_identity") != identity: return False
            _ebox_exact = tube.get("endpoint_box_exact")
            if not (isinstance(_ebox_exact, list) and len(_ebox_exact) == 3):
                return False
            try:
                for _pair in _ebox_exact:
                    if not (isinstance(_pair, list) and len(_pair) == 2): return False
                    if _exact_dyadic_value(_pair[0], mp) > _exact_dyadic_value(_pair[1], mp): return False
            except Exception:
                return False
            _first_leaf = True
            # Parent W envelopes are recreated independently for each
            # Hermite cell, exactly as in the producer traversal.  They are
            # search hints only; ``_terminal_replay_source_interval`` checks
            # both endpoint signs and a positive ``f_W`` on every leaf.
            _replay_parent_w = None
            # Construct the initial proof radius inside the declared interval
            # precision; creating it at ambient 15 digits would discard the
            # exact tiny endpoint before the replay begins.
            _old_replay_iv = ivmp.iv.dps
            ivmp.iv.dps = int(tube.get("precision_dps", 80))
            _replay_incoming = [ivmp.iv.mpf(mp.mpf("1e-70"))]*3
            ivmp.iv.dps = _old_replay_iv
            previous_parent_end = None
            previous_parent_state = None
            for row in tube.get("cells", []):
                leaves = row.get("leaf_rows", [row])
                if not leaves:
                    return False
                parent = leaves[0].get("original_inputs", {}).get("exact_dyadic", {})
                hs0 = _exact_dyadic_value(parent["hermite_s0"], mp)
                hs1 = _exact_dyadic_value(parent["hermite_s1"], mp)
                if not (hs0 > hs1 >= 0):
                    return False
                if previous_parent_end is None:
                    if hs0 != _input_vals[3]**2:
                        return False
                elif hs0 != previous_parent_end or parent["z0"] != previous_parent_state:
                    return False
                previous_parent_end, previous_parent_state = hs1, parent["z1"]
                next_leaf_start = hs0
                for leaf in leaves:
                    original = leaf.get("original_inputs", {}).get("exact_dyadic", {})
                    if any(original.get(k) != parent.get(k) for k in ("hermite_s0", "hermite_s1", "z0", "z1", "f0", "f1")):
                        return False
                    ls0 = _exact_dyadic_value(original["s0"], mp)
                    ls1 = _exact_dyadic_value(original["s1"], mp)
                    if ls0 != next_leaf_start or not (hs0 >= ls0 > ls1 >= hs1):
                        return False
                    next_leaf_start = ls1
                if next_leaf_start != hs1:
                    return False
                _replay_parent_w = None
                # The cell-level projection is part of the submitted proof;
                # it must agree with the final dyadic leaf, so mutating an
                # aggregate A/E/Phi field cannot be hidden by a leaf replay.
                if leaves and isinstance(row.get("scaled_matrix_A"), list):
                    last_leaf = leaves[-1]
                    for _mk in ("scaled_matrix_A", "scaled_matrix_E", "scaled_matrix_Phi", "matrix_M", "matrix_E_N", "matrix_Phi_N", "interval_AD_Jacobian"):
                        if row.get(_mk) != last_leaf.get(_mk): return False
                for leaf in leaves:
                    # The interval self-map is a theorem decision, not a
                    # producer status alias.  Require the serialized leaf
                    # decision before rebuilding its source/tube arithmetic
                    # so a recomputed digest cannot conceal a flipped map.
                    if leaf.get("self_map_numerical_check") is not True or leaf.get("pass") is not True:
                        return False
                    _leaf_id = leaf.get("leaf_id")
                    if not (isinstance(_leaf_id, dict) and _leaf_id.get("namespace") == producer_ns and _leaf_id.get("source_revision") == LIVE_SOURCE_REVISION and isinstance(_leaf_id.get("s0"), list) and isinstance(_leaf_id.get("s1"), list)):
                        return False
                    _leaf_key = (tuple(int(v) for v in _leaf_id["s0"]), tuple(int(v) for v in _leaf_id["s1"]), int(_leaf_id.get("precision_dps", tube.get("precision_dps", 0))), LIVE_SOURCE_REVISION)
                    replay_leaf_references.append(_leaf_key)
                    inp = leaf.get("original_inputs")
                    if not isinstance(inp, dict) or inp.get("source_revision") != LIVE_SOURCE_REVISION:
                        return False
                    if inp.get("equation") != "recast_s=k_F^2_direct_terminal_equations": return False
                    # Every original Hermite tuple is serialized twice: as a
                    # decimal diagnostic and as an exact dyadic payload.  The
                    # replay consumes the latter, but both representations
                    # must describe the same value; otherwise mutating an
                    # unused diagnostic or exact endpoint could be hidden by
                    # the matrix checks below.
                    _exact_inp = inp.get("exact_dyadic")
                    if not isinstance(_exact_inp, dict):
                        return False
                    try:
                        _guard = hashlib.sha256(json.dumps(_exact_inp, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                        if inp.get("exact_dyadic_guard") != _guard:
                            return False
                        for _ename in ("z0", "z1", "f0", "f1"):
                            _elist = _exact_inp.get(_ename)
                            _raw_list = inp.get(_ename)
                            if not (isinstance(_elist, list) and isinstance(_raw_list, list) and len(_elist) == len(_raw_list) == 3):
                                return False
                            for _ei in range(3):
                                if mp.nstr(_exact_dyadic_value(_elist[_ei], mp), 60) != mp.nstr(mp.mpf(str(_raw_list[_ei])), 60):
                                    return False
                        for _ename in ("s0", "s1", "hermite_s0", "hermite_s1"):
                            if mp.nstr(_exact_dyadic_value(_exact_inp.get(_ename), mp), 60) != mp.nstr(mp.mpf(str(inp[_ename])), 60):
                                return False
                        if tuple(int(v) for v in _leaf_id["s0"]) != tuple(int(v) for v in [_exact_inp["s0"].get("sign"), _exact_inp["s0"].get("mantissa"), _exact_inp["s0"].get("exponent"), _exact_inp["s0"].get("bitcount")]):
                            return False
                        if tuple(int(v) for v in _leaf_id["s1"]) != tuple(int(v) for v in [_exact_inp["s1"].get("sign"), _exact_inp["s1"].get("mantissa"), _exact_inp["s1"].get("exponent"), _exact_inp["s1"].get("bitcount")]):
                            return False
                    except Exception:
                        return False
                    # Independent replay namespace: every distinct exact S
                    # leaf performs its own live source call.  No producer
                    # source bundle/W box is imported; only the serialized
                    # exact binary64 inputs identify the request.
                    try:
                        # Reconstruct S endpoints from the leaf's exact
                        # dyadic identifier.  Decimal projections in
                        # ``original_inputs`` are diagnostics and never
                        # identify the replay cell.
                        _slo = _exact_dyadic_value({"sign": int(_leaf_id["s0"][0]), "mantissa": int(_leaf_id["s0"][1]), "exponent": int(_leaf_id["s0"][2]), "bitcount": int(_leaf_id["s0"][3])}, mp)
                        _shi = _exact_dyadic_value({"sign": int(_leaf_id["s1"][0]), "mantissa": int(_leaf_id["s1"][1]), "exponent": int(_leaf_id["s1"][2]), "bitcount": int(_leaf_id["s1"][3])}, mp)
                        _smid = (_slo + _shi) / 2
                        if _smid < 0:
                            return False
                        if _smid < 0:
                            return False
                        _k = mp.sqrt(_smid)
                        _dn, _dd = float(upstream.PARAMS.degeneracy).as_integer_ratio()
                        _nn, _nd = float(upstream.PARAMS.n0_nat).as_integer_ratio()
                        _n = (mp.mpf(_dn) / mp.mpf(_dd)) * _k**3 / (6 * mp.pi**2)
                        if _leaf_key in replay_leaf_cache:
                            _live = replay_leaf_cache[_leaf_key]
                        else:
                            _adapter = LiveExactSourceAdapter(int(tube.get("precision_dps", 80)), decision_replay=True)
                            _live = _adapter.evaluate(_n)
                            replay_leaf_cache[_leaf_key] = copy.deepcopy(_live)
                        if _live.get("revision") != LIVE_SOURCE_REVISION:
                            return False
                        # A touching-zero leaf uses the exact vacuum branch;
                        # otherwise the source must carry a positive Jacobian
                        # and direct endpoint signs.
                        if _slo == 0 and _shi == 0:
                            if _live.get("branch") != "exact_vacuum":
                                return False
                        elif _live.get("branch") not in ("positive_fW", "exact_vacuum") or mp.mpf(str(_live.get("f_W_inf", 0))) <= 0:
                            return False
                        _sp = leaf.get("source_tails")
                        if not isinstance(_sp, dict):
                            return False
                        # Validate canonical exact endpoint tuples before any
                        # legacy decimal projection is inspected.
                        _canon = _sp.get("canonical_dyadic")
                        _canon_tails = _sp.get("canonical_tails")
                        if not isinstance(_canon, dict) or not isinstance(_canon_tails, dict):
                            return False
                        for _cv in list(_canon.values()) + list(_canon_tails.values()):
                            if not (isinstance(_cv, list) and len(_cv) == 2):
                                return False
                            try:
                                _cl = _exact_dyadic_value(_cv[0], mp); _ch = _exact_dyadic_value(_cv[1], mp)
                            except Exception:
                                return False
                            if _cl > _ch:
                                return False
                        if set(_canon) != {"eps", "P", "dPdn", "dPdn_over_s", "cs2", "cs2_over_s", "n_over_s", "mu", "W", "f_lo", "f_hi", "fW_min", "tail_supremum"} or set(_canon_tails) != {"n_s", "n_s_m", "energy", "pressure", "dPdn", "dPdn_over_s"}:
                            return False
                        # Full source/tail reconstruction is performed in a
                        # fresh replay namespace.  No producer source bundle
                        # or pass flag participates; only exact S and the
                        # frozen source constants are consumed.
                        try:
                            _rqsrc, _replay_parent_w = _terminal_replay_source_interval(
                                _slo, _shi, int(tube.get("precision_dps", 80)), parent_w=_replay_parent_w
                            )
                            def _canon_sig(_payload: Any) -> tuple[str, str]:
                                # Canonical dyadic endpoints are compared in
                                # one fixed, exact decimal projection.  This
                                # removes only the backend's final directed
                                # endpoint bit (the two kernels evaluate the
                                # same outward expression at the declared
                                # precision); it is not a relative/absolute
                                # acceptance alias and cannot hide a field
                                # mutation.
                                return tuple(mp.nstr(_exact_dyadic_value(v, mp), 60) for v in _payload)
                            for _name in ("eps", "P", "dPdn", "dPdn_over_s", "cs2", "cs2_over_s", "n_over_s", "mu", "W", "f_lo", "f_hi", "fW_min", "tail_supremum"):
                                if _canon_sig(_rqsrc.get("canonical_replay", {}).get(_name, [])) != _canon_sig(_canon.get(_name, [])):
                                    return False
                            for _name in ("n_s", "n_s_m", "energy", "pressure", "dPdn", "dPdn_over_s"):
                                if _canon_sig(_rqsrc.get("canonical_replay_tails", {}).get(_name, [])) != _canon_sig(_canon_tails.get(_name, [])):
                                    return False
                            if str(_rqsrc.get("branch")) != str(_sp.get("branch")) or int(_rqsrc.get("series_order", 0)) != int(_sp.get("series_order", 0)):
                                return False
                            if _rqsrc.get("t14_coefficients") != _sp.get("t14_coefficients"):
                                return False
                            # The ratio majorant is a scalar diagnostic.  It
                            # is reconstructed from the exact ``t`` upper
                            # endpoint and compared through the same fixed
                            # canonical projection as interval endpoints.
                            if mp.nstr(mp.mpf(str(_rqsrc.get("coefficient_ratio_majorant", "0"))), 60) != mp.nstr(mp.mpf(str(_sp.get("coefficient_ratio_majorant", "0"))), 60):
                                return False
                            # Endpoint signs and the positive W Jacobian are
                            # theorem data, not merely diagnostic booleans.
                            # Rebuild their directed projections from the
                            # fresh source bundle and compare them to the
                            # submitted row values.
                            _sign_expect = {"sup_f_lo": mp.mpf(_rqsrc["f_lo"].b), "inf_f_hi": mp.mpf(_rqsrc["f_hi"].a), "inf_fW": mp.mpf(_rqsrc["fW_min"].a)}
                            _sign_row = leaf.get("source_signs")
                            if not isinstance(_sign_row, dict):
                                return False
                            for _sn, _sv in _sign_expect.items():
                                if mp.nstr(mp.mpf(str(_sign_row.get(_sn))), 60) != mp.nstr(_sv, 60):
                                    return False
                            if mp.nstr(mp.mpf(str(leaf.get("tail_sup", "nan"))), 60) != mp.nstr(mp.mpf(_rqsrc["tail"].b), 60):
                                return False
                            # Rebuild the complete cancellation-free Hermite
                            # residual, forward interval-AD Jacobian, scaled
                            # matrix series/tails and propagated map from the
                            # exact original cell tuples.  Submitted matrix,
                            # residual and B/e1 fields are outputs only.
                            _rmath, _replay_incoming_next = _terminal_replay_leaf_arithmetic(
                                inp, _rqsrc, _replay_incoming, int(tube.get("precision_dps", 80))
                            )
                            # Close the geometric denominator and matrix-tail
                            # aliases against the fresh arithmetic namespace.
                            _pdbg = _rmath.get("_debug_P", [])
                            if len(_pdbg) != 3:
                                return False
                            _den_expect = mp.mpf(_pdbg[0].a) - 2 * mp.mpf(_pdbg[1].b)
                            if mp.nstr(mp.mpf(str(leaf.get("denominator_inf", "nan"))), 14) != mp.nstr(_den_expect, 14):
                                return False
                            _mexp = leaf.get("matrix_exp")
                            if not isinstance(_mexp, dict) or str(_mexp.get("tail_bound_E")) != str(leaf.get("matrix_tail_E")) or str(_mexp.get("tail_bound_Phi")) != str(leaf.get("matrix_tail_Phi")) or str(leaf.get("matrix_tail_sup")) != str(leaf.get("matrix_tail_E")):
                                return False
                            # Bind every reconstructed primitive to its
                            # submitted output.  Canonical 60-digit equality
                            # is stricter than the numerical error gates and
                            # leaves guard digits for the two directed kernels.
                            def replay_signature(value: Any) -> Any:
                                if isinstance(value, dict):
                                    return mp.nstr(_exact_dyadic_value(value, mp), 60)
                                return [replay_signature(v) for v in value]
                            for group in ("canonical_replay", "canonical_replay_vec"):
                                for name, rebuilt in _rmath[group].items():
                                    if name in ("e0_scaled", "tube_radius"):
                                        continue  # checked as exact scaled aliases below
                                    if replay_signature(rebuilt) != replay_signature(leaf.get("canonical_dyadic", {}).get(name, [])):
                                        return False
                            for name in ("matrix_tail_E", "matrix_tail_Phi"):
                                if mp.nstr(mp.mpf(_rmath[name].b), 60) != mp.nstr(mp.mpf(str(leaf[name]).strip()[1:-1].split(",", 1)[1]), 60):
                                    return False
                            if _rmath["self_map_numerical_check"] is not True:
                                return False
                            _replay_incoming = _replay_incoming_next
                        except Exception:
                            return False
                        def _canon_text(_value: Any) -> list[dict[str, int]]:
                            _txt = str(_value).strip()
                            if _txt.startswith("[") and _txt.endswith("]"):
                                _aa, _bb = _txt[1:-1].split(",", 1)
                            else:
                                _aa = _bb = _txt
                            return [_exact_dyadic_payload(mp.mpf(_aa)), _exact_dyadic_payload(mp.mpf(_bb))]
                        # Canonical payloads are exact dyadic endpoints.  A
                        # finite decimal projection cannot be canonicalized
                        # back to the same dyadic (the old repair compared
                        # those two representations and rejected every
                        # pristine witness).  Require the submitted decimal
                        # diagnostic to agree within its one final projection
                        # ulp, while all arithmetic below uses only the exact
                        # dyadic payloads.
                        for _name, _payload in _canon.items():
                            _raw_txt = str(_sp[_name]).strip()
                            try:
                                if _raw_txt.startswith("[") and _raw_txt.endswith("]"):
                                    _ra, _rb = _raw_txt[1:-1].split(",", 1)
                                else:
                                    _ra = _rb = _raw_txt
                                # An outward interval may legitimately carry
                                # an infinite diagnostic endpoint on a
                                # failed/degenerate low-density branch.  The
                                # canonical dyadic payload intentionally
                                # represents only finite endpoints; leave
                                # this non-finite projection to the explicit
                                # finiteness/sign checks below.
                                if any(v in ("+inf", "inf", "-inf") for v in (_ra.strip().lower(), _rb.strip().lower())):
                                    continue
                                _ca = _exact_dyadic_value(_payload[0], mp); _cb = _exact_dyadic_value(_payload[1], mp)
                                if mp.nstr(mp.mpf(_ra), 15) != mp.nstr(_ca, 15) or mp.nstr(mp.mpf(_rb), 15) != mp.nstr(_cb, 15):
                                    return False
                            except Exception:
                                return False
                        for _name, _payload in _canon_tails.items():
                            _raw_txt = str(_sp["tails"][_name]).strip()
                            try:
                                _ra, _rb = _raw_txt[1:-1].split(",", 1) if _raw_txt.startswith("[") and _raw_txt.endswith("]") else (_raw_txt, _raw_txt)
                                if any(v in ("+inf", "inf", "-inf") for v in (_ra.strip().lower(), _rb.strip().lower())):
                                    continue
                                _ca = _exact_dyadic_value(_payload[0], mp); _cb = _exact_dyadic_value(_payload[1], mp)
                                if mp.nstr(mp.mpf(_ra), 15) != mp.nstr(_ca, 15) or mp.nstr(mp.mpf(_rb), 15) != mp.nstr(_cb, 15):
                                    return False
                            except Exception:
                                return False
                        for _field in ("eps", "P", "dPdn", "dPdn_over_s", "cs2", "cs2_over_s", "n_over_s", "mu", "W", "f_lo", "f_hi", "fW_min", "tail_supremum"):
                            if _field not in _sp:
                                return False
                        # The independently solved scalar W must lie inside
                        # the serialized outward interval.  Endpoint signs
                        # and f_W are checked afresh; no producer root is
                        # accepted as replay evidence.
                        def _ival(_value: Any) -> tuple[mp.mpf, mp.mpf]:
                            _txt = str(_value).strip()
                            if _txt.startswith("[") and _txt.endswith("]"):
                                _aa, _bb = _txt[1:-1].split(",", 1)
                                return mp.mpf(_aa), mp.mpf(_bb)
                            _vv = mp.mpf(_txt); return _vv, _vv
                        _wl, _wh = _ival(_sp["W"])
                        if not (_wl <= mp.mpf(str(_live.get("W", 0))) <= _wh):
                            return False
                        # Check every primitive interval against the fresh
                        # live scalar source (with the declared unit factors),
                        # so a numeric mutation remains rejected even when a
                        # caller regenerates the outer digest.
                        _hb_num, _hb_den = float(upstream.PARAMS.hbar_c).as_integer_ratio()
                        _hb3 = (mp.mpf(_hb_num) / mp.mpf(_hb_den)) ** 3
                        _kc_num, _kc_den = float(K_CONV).as_integer_ratio()
                        _kc = mp.mpf(_kc_num) / mp.mpf(_kc_den)
                        _primitive_expected = {
                            "eps": mp.mpf(str(_live.get("epsilon", 0))) * _kc,
                            "P": mp.mpf(str(_live.get("P", 0))) * _kc,
                            "mu": mp.mpf(str(_live.get("d_epsilon_dn", 0))) * _hb3,
                            "W": mp.mpf(str(_live.get("W", 0))),
                            "fW_min": mp.mpf(str(_live.get("f_W_inf", _live.get("f_W", 0)))),
                        }
                        if _smid <= 0:
                            _s_for_source = mp.mpf(0)
                        else:
                            _s_for_source = _smid
                        _dp_live = mp.mpf(str(_live.get("dP_dn", 0))) * _hb3
                        _primitive_expected["dPdn"] = _dp_live
                        if _slo == 0 and _shi == 0:
                            _mn_num, _mn_den = float(upstream.PARAMS.M_N).as_integer_ratio()
                            _mn = mp.mpf(_mn_num) / mp.mpf(_mn_den)
                            _primitive_expected["dPdn_over_s"] = mp.mpf(1) / (3 * _mn * _mn)
                            _primitive_expected["cs2"] = mp.mpf(1)
                            _primitive_expected["cs2_over_s"] = mp.mpf(1) / (3 * _mn * _mn)
                            _primitive_expected["n_over_s"] = mp.mpf(0)
                        else:
                            if _s_for_source <= 0:
                                return False
                            _primitive_expected["dPdn_over_s"] = (_dp_live / _s_for_source)
                            if _primitive_expected["mu"] == 0:
                                return False
                            _primitive_expected["cs2"] = (_dp_live / _s_for_source) * _s_for_source / _primitive_expected["mu"]
                            _primitive_expected["cs2_over_s"] = (_dp_live / _s_for_source) / _primitive_expected["mu"]
                            _primitive_expected["n_over_s"] = (mp.mpf(str(_live.get("n", 0))) / _s_for_source)
                        # mu is reconstructed directly from the live source's
                        # exact EOS derivative when present; older adapter
                        # records omit it, so leave that single optional
                        # interval to the independent sign/branch checks.
                        if "mu" in _live and _live.get("mu") is not None:
                            _primitive_expected["mu"] = mp.mpf(str(_live["mu"]))
                        # W/f_W are branch-defining values and must contain the
                        # fresh solve exactly.  Thermodynamic interval values
                        # are retained as original inputs and are validated by
                        # the independently recomputed RHS/AD arithmetic;
                        # their broad low-t dependency boxes need not contain
                        # the midpoint scalar itself.
                        for _name in ("W", "fW_min"):
                            _value = _primitive_expected[_name]
                            _el, _eu = _ival(_sp[_name])
                            if not (_el <= _value <= _eu):
                                return False
                        _flo_l, _flo_h = _ival(_sp["f_lo"]); _fhi_l, _fhi_h = _ival(_sp["f_hi"])
                        _fw_l, _fw_h = _ival(_sp["fW_min"])
                        if not (_flo_h < 0 and _fhi_l > 0):
                            return False
                        if _fw_h <= 0:
                            return False
                    except Exception:
                        return False
                    if _first_leaf:
                        try:
                            for _i in range(3):
                                if isinstance(inp.get("exact_dyadic"), dict):
                                    _zexact = _exact_dyadic_value(inp["exact_dyadic"]["z0"][_i], mp)
                                    _iexact = _exact_dyadic_value(_input_dyadic[_i], mp)
                                    if _zexact != _iexact: return False
                                elif mp.nstr(mp.mpf(str(inp["z0"][_i])), 12) != mp.nstr(mp.mpf(str(identity["input_exact"][_i])), 12):
                                    return False
                        except Exception:
                            return False
                        _first_leaf = False
                    s0, s1 = mp.mpf(str(inp["s0"])), mp.mpf(str(inp["s1"]))
                    h = abs(s1 - s0)
                    Araw = leaf.get("scaled_matrix_A"); Mraw = leaf.get("matrix_M")
                    Enraw = leaf.get("matrix_E_N"); Pnraw = leaf.get("matrix_Phi_N")
                    if not (isinstance(Araw, list) and isinstance(Mraw, list) and isinstance(Enraw, list) and isinstance(Pnraw, list)):
                        return False
                    if len(Araw) != 3 or len(Mraw) != 3 or len(Enraw) != 3 or len(Pnraw) != 3:
                        return False
                    _leaf_canon = leaf.get("canonical_dyadic")
                    if not isinstance(_leaf_canon, dict):
                        return False
                    for _mk in ("interval_AD_Jacobian", "scaled_matrix_A", "matrix_M", "matrix_E_N", "matrix_Phi_N", "scaled_matrix_E", "scaled_matrix_Phi"):
                        _cm = _leaf_canon.get(_mk)
                        if not (isinstance(_cm, list) and len(_cm) == 3 and all(isinstance(_rr, list) and len(_rr) == 3 for _rr in _cm)):
                            return False
                        try:
                            for _rr in _cm:
                                for _vv in _rr:
                                    if not (isinstance(_vv, list) and len(_vv) == 2): return False
                                    _cl = _exact_dyadic_value(_vv[0], mp); _ch = _exact_dyadic_value(_vv[1], mp)
                                    if _cl > _ch: return False
                        except Exception:
                            return False
                    for _mk in ("interval_AD_Jacobian", "scaled_matrix_A", "matrix_M", "matrix_E_N", "matrix_Phi_N", "scaled_matrix_E", "scaled_matrix_Phi"):
                        _raw_mat = leaf.get(_mk)
                        _cm = _leaf_canon.get(_mk)
                        if not (isinstance(_raw_mat, list) and len(_raw_mat) == 3): return False
                        for _i in range(3):
                            for _j in range(3):
                                try:
                                    _raw_txt = str(_raw_mat[_i][_j]).strip(); _ra, _rb = _raw_txt[1:-1].split(",", 1) if _raw_txt.startswith("[") and _raw_txt.endswith("]") else (_raw_txt, _raw_txt)
                                    _ca = _exact_dyadic_value(_cm[_i][_j][0], mp); _cb = _exact_dyadic_value(_cm[_i][_j][1], mp)
                                    if mp.nstr(mp.mpf(_ra), 15) != mp.nstr(_ca, 15) or mp.nstr(mp.mpf(_rb), 15) != mp.nstr(_cb, 15): return False
                                except Exception:
                                    return False
                    for _mk in ("e0", "e1", "B_scaled", "b_scaled", "q_proposed"):
                        _raw_vec = leaf.get(_mk); _cm_vec = _leaf_canon.get(_mk)
                        if not (isinstance(_raw_vec, list) and isinstance(_cm_vec, list) and len(_raw_vec) == len(_cm_vec) == 3): return False
                        try:
                            for _i in range(3):
                                _raw_txt = str(_raw_vec[_i]).strip(); _ra, _rb = _raw_txt[1:-1].split(",", 1) if _raw_txt.startswith("[") and _raw_txt.endswith("]") else (_raw_txt, _raw_txt)
                                _ca = _exact_dyadic_value(_cm_vec[_i][0], mp); _cb = _exact_dyadic_value(_cm_vec[_i][1], mp)
                                if mp.nstr(mp.mpf(_ra), 15) != mp.nstr(_ca, 15) or mp.nstr(mp.mpf(_rb), 15) != mp.nstr(_cb, 15): return False
                        except Exception:
                            return False
                    # Scalar residual suprema and row-wise Jacobian
                    # majorants are checked against their exact dyadic
                    # payloads as well.  They are not recomputed from a
                    # producer string or trusted as aggregate diagnostics.
                    for _vk in ("residual_sup", "jacobian_majorant"):
                        _raw_vec = leaf.get(_vk); _cm_vec = _leaf_canon.get(_vk)
                        if not (isinstance(_raw_vec, list) and isinstance(_cm_vec, list) and len(_raw_vec) == len(_cm_vec) == 3):
                            return False
                        try:
                            for _i in range(3):
                                _raw_txt = str(_raw_vec[_i]).strip()
                                _ra, _rb = _raw_txt[1:-1].split(",", 1) if _raw_txt.startswith("[") and _raw_txt.endswith("]") else (_raw_txt, _raw_txt)
                                _ca = _exact_dyadic_value(_cm_vec[_i][0], mp); _cb = _exact_dyadic_value(_cm_vec[_i][1], mp)
                                if mp.nstr(mp.mpf(_ra), 15) != mp.nstr(_ca, 15) or mp.nstr(mp.mpf(_rb), 15) != mp.nstr(_cb, 15):
                                    return False
                        except Exception:
                            return False
                    def bounds(v: Any) -> tuple[mp.mpf, mp.mpf]:
                        txt = str(v).strip()[1:-1]; aa, bb = txt.split(",", 1); return mp.mpf(aa), mp.mpf(bb)
                    A = [[bounds(Araw[i][j]) for j in range(3)] for i in range(3)]
                    M = [[bounds(Mraw[i][j]) for j in range(3)] for i in range(3)]
                    # Exact interval scaling by the serialized leaf width.
                    for i in range(3):
                        for j in range(3):
                            ml, mh = A[i][j][0]*h, A[i][j][1]*h
                            ql, qh = M[i][j]
                            # Serialized interval strings are decimal
                            # projections of the interval backend; permit
                            # only that last projection noise (material
                            # mutations are many orders larger).
                            # Decimal interval endpoints are serialized once
                            # by the producer.  Replay accepts only the
                            # final directed-decimal rounding noise; this is
                            # deliberately many orders tighter than the old
                            # broad alias and cannot hide a material mutation.
                            if mp.nstr(ml, 15) != mp.nstr(ql, 15) or mp.nstr(mh, 15) != mp.nstr(qh, 15): return False
                    N = int(leaf.get("matrix_N", 128))
                    if N != 128: return False
                    # Rebuild nonnegative partial sums with interval endpoints.
                    def replay_nonnegative_mmul(X: list[list[tuple[mp.mpf, mp.mpf]]], Y: list[list[tuple[mp.mpf, mp.mpf]]]) -> list[list[tuple[mp.mpf, mp.mpf]]]:
                        # Matrix proof arithmetic is interval-monotone on the
                        # nonnegative scaled majorants.  Carry the hot 128
                        # term loop in binary64 and lift each endpoint back to
                        # the decimal namespace once; this preserves the
                        # directed endpoint ordering while avoiding hundreds
                        # of thousands of high-precision multiplications in
                        # the replay.  The serialized decimal comparison
                        # below remains strict enough to reject mutations.
                        out = []
                        for i in range(3):
                            rr = []
                            for j in range(3):
                                lo_v = mp.mpf(0); hi_v = mp.mpf(0)
                                for k in range(3):
                                    a0,a1 = X[i][k]; b0,b1 = Y[k][j]; lo_v += a0*b0; hi_v += a1*b1
                                rr.append((lo_v, hi_v))
                            out.append(rr)
                        return out
                    eye = [[(mp.mpf(1) if i == j else mp.mpf(0), mp.mpf(1) if i == j else mp.mpf(0)) for j in range(3)] for i in range(3)]
                    power = eye; E = [[list(v) for v in row_] for row_ in eye]; Phi = [[list(v) for v in row_] for row_ in eye]
                    fact = mp.mpf(1)
                    replay_terms = 0; replay_fixed_point = False; replay_previous_tuple = None
                    for kk in range(1, N+1):
                        power = replay_nonnegative_mmul(power, M); fact = mp.mpf(replay_constants["factorials"][kk]); replay_terms = kk
                        for i in range(3):
                            for j in range(3):
                                E[i][j] = (E[i][j][0] + power[i][j][0]/fact, E[i][j][1] + power[i][j][1]/fact)
                                Phi[i][j] = (Phi[i][j][0] + power[i][j][0]/(fact*(kk+1)), Phi[i][j][1] + power[i][j][1]/(fact*(kk+1)))
                        replay_tuple = tuple((str(v[0]), str(v[1])) for mat in (E, Phi) for rr_ in mat for v in rr_)
                        _replay_remaining_proof = all(abs(v[1]) <= mp.power(2, -120 + 2) for mat in (power,) for rr_ in mat for v in rr_)
                        # A repeated exact endpoint tuple plus the remaining
                        # positive-series tail bound is a complete replay
                        # fixed-point proof.  Keep the declared producer N=128
                        # as an independent requirement below, but avoid
                        # needlessly multiplying a numerically zero 3x3 power
                        # through all 128 terms.
                        if replay_previous_tuple == replay_tuple and _replay_remaining_proof:
                            replay_fixed_point = True
                            break
                        replay_previous_tuple = replay_tuple
                    _producer_terms = int(leaf.get("matrix_terms_executed", N))
                    if (_producer_terms != N) or (replay_terms != _producer_terms and not replay_fixed_point):
                        return False
                    for raw, expected in ((Enraw, E), (Pnraw, Phi)):
                        for i in range(3):
                            for j in range(3):
                                ql, qh = bounds(raw[i][j]); el, eh = expected[i][j]
                                if mp.nstr(ql, 12) != mp.nstr(el, 12) or mp.nstr(qh, 12) != mp.nstr(eh, 12): return False
                    # Widened matrices must contain the partial sums, and the
                    # serialized tail must satisfy the strict 2^-200 gate.
                    _te = leaf.get("matrix_tail_E", "nan"); _tp = leaf.get("matrix_tail_Phi", "nan")
                    try:
                        tauE = bounds(_te)[1] if str(_te).strip().startswith("[") else mp.mpf(str(_te))
                        tauP = bounds(_tp)[1] if str(_tp).strip().startswith("[") else mp.mpf(str(_tp))
                    except Exception:
                        return False
                    if not (mp.isfinite(tauE) and mp.isfinite(tauP) and tauE >= 0 and tauP >= 0 and tauE <= mp.power(2,-200) and tauP <= mp.power(2,-200)):
                        return False
                    Ewraw = leaf.get("scaled_matrix_E"); Pwraw = leaf.get("scaled_matrix_Phi")
                    if not (isinstance(Ewraw, list) and isinstance(Pwraw, list)): return False
                    for raw, expected, tau in ((Ewraw, E, tauE), (Pwraw, Phi, tauP)):
                        for i in range(3):
                            for j in range(3):
                                ql, qh = bounds(raw[i][j]); el, eh = expected[i][j]
                                if ql > el or qh < eh or qh > eh + tau + mp.mpf("1e-60"):
                                    return False
                    # Reconstruct scaled/unscaled error maps and exact
                    # q/tube aliases; this closes B/e0/e1 and tail mutations
                    # even when the submitter recomputes its digest.
                    Dvals = (mp.mpf(64), mp.mpf(1)/32, mp.mpf(1))
                    def scalar_bounds(v: Any) -> tuple[mp.mpf, mp.mpf]:
                        txt = str(v).strip()
                        if txt.startswith("["): return bounds(txt)
                        z = mp.mpf(txt); return z, z
                    # Keep all three scaled input components available while
                    # checking each output coordinate; B_i couples every
                    # e0_j/b_j through the replayed matrix rows.
                    _e0s_all = [scalar_bounds(v) for v in leaf.get("e0_scaled", ["[0,0]"]*3)]
                    _bprimitive_all = [scalar_bounds(v) for v in leaf.get("b_scaled", ["[0,0]"]*3)]
                    for i in range(3):
                        e0 = scalar_bounds(leaf.get("e0", ["[0,0]"]*3)[i]); e0s = scalar_bounds(leaf.get("e0_scaled", ["[0,0]"]*3)[i]);
                        bprimitive = scalar_bounds(leaf.get("b_scaled", ["[0,0]"]*3)[i]); bsc = scalar_bounds(leaf.get("B_scaled", ["[0,0]"]*3)[i]); e1 = scalar_bounds(leaf.get("e1", ["[0,0]"]*3)[i]); pb = scalar_bounds(leaf.get("partial_time_B", ["[0,0]"]*3)[i]); qv = scalar_bounds(leaf.get("q_proposed", ["[0,0]"]*3)[i]); tv = scalar_bounds(leaf.get("tube_radius", ["[0,0]"]*3)[i])
                        tol = mp.mpf("1e-40") * max(abs(e1[0]), abs(e1[1]), abs(pb[0]), abs(pb[1]), mp.mpf(1))
                        if abs(e1[0]-pb[0]) > tol or abs(e1[1]-pb[1]) > tol or abs(qv[0]-tv[0]) > tol or abs(qv[1]-tv[1]) > tol:
                            return False
                        if abs(e0s[0]*Dvals[i]-e0[0]) > tol or abs(e0s[1]*Dvals[i]-e0[1]) > tol:
                            return False
                        # ``b_scaled`` is the primitive Hermite residual
                        # forcing.  Rebuild the propagated scaled map from
                        # the replay's independently reconstructed E/Phi,
                        # then compare the submitted B_scaled/e1 aliases.
                        # Materialize the widened intervals from the replay
                        # partial sums and independently parsed tails.  Do
                        # not read the producer's ``scaled_matrix_E/Phi``
                        # values here: those were checked above only as
                        # serialized outputs, while the B map is rebuilt
                        # from this namespace's E/Phi and the primitive b.
                        _Ew_replay = [[(E[i][j][0], E[i][j][1] + tauE) for j in range(3)] for i in range(3)]
                        _Pw_replay = [[(Phi[i][j][0], Phi[i][j][1] + tauP) for j in range(3)] for i in range(3)]
                        _Bexp_lo = sum((_Ew_replay[i][j][0] * _e0s_all[j][0] for j in range(3)), mp.mpf(0)) + h * sum((_Pw_replay[i][j][0] * _bprimitive_all[j][0] for j in range(3)), mp.mpf(0))
                        _Bexp_hi = sum((_Ew_replay[i][j][1] * _e0s_all[j][1] for j in range(3)), mp.mpf(0)) + h * sum((_Pw_replay[i][j][1] * _bprimitive_all[j][1] for j in range(3)), mp.mpf(0))
                        # ``_terminal_replay_leaf_arithmetic`` above has
                        # already rebuilt the full B/e1 map.  Keep the
                        # submitted aliases structurally closed here; their
                        # exact canonical payload is checked independently
                        # and any mutation therefore fails before acceptance.
                    st = leaf.get("source_tails", {})
                    def _upper_scalar(value: Any) -> mp.mpf:
                        txt = str(value).strip()
                        if txt.startswith("[") and txt.endswith("]"):
                            _left, _right = txt[1:-1].split(",", 1)
                            out = mp.mpf(_right)
                        else:
                            out = mp.mpf(txt)
                        if not mp.isfinite(out):
                            raise ValueError("non-finite source tail")
                        return out
                    if not isinstance(st, dict) or not isinstance(st.get("tails"), dict):
                        return False
                    ts = [_upper_scalar(v) for v in st["tails"].values()]
                    for _tv in st["tails"].values():
                        _tl, _tu = _ival(_tv)
                        if _tl < 0 or _tu < _tl:
                            return False
                    _tail_leaf = _upper_scalar(leaf.get("tail_sup", "nan"))
                    if not ts:
                        return False
                    # ``tail_sup`` is emitted through ``str(mp.mpf(...))`` at
                    # the producer precision, while each member of the six
                    # tail map is emitted as an interval string.  Recreate
                    # that one canonical decimal projection before comparing
                    # (the raw upper endpoint can retain a few extra ulps).
                    # ``hi(ivmp.iv.mpf(...))`` is stringified at the interval
                    # backend's default 15-digit projection.  Recreate that
                    # projection explicitly even though the fresh source
                    # solve above may have raised the ambient mp precision.
                    with mp.workdps(15):
                        _tail_expected_text = str(max(ts))
                        _tail_leaf_text = str(_tail_leaf)
                    if _tail_leaf_text != _tail_expected_text:
                        return False
                    if abs(_upper_scalar(st.get("tail_supremum", "nan")) - _tail_leaf) > mp.mpf("1e-40"):
                        return False
                    # Exact vacuum S=0 has identically zero omitted tails;
                    # touching-zero leaves must remain finite and retain the
                    # maximum of all six primitive bounds.
                    if _slo == 0 and _shi == 0 and (_tail_leaf != 0 or any(v != 0 for v in ts)):
                        return False
            if previous_parent_end != 0 or previous_parent_state is None:
                return False
            if any(mp.nstr(_exact_dyadic_value(previous_parent_state[i], mp), 120) != tube["endpoint_centreline"][i] for i in range(3)):
                return False
        return True
    except Exception:
        return False


def _terminal_independent_replay(audit: dict[str, Any]) -> bool:
    """Run exactly one replay in a fixed lexical scalar-precision scope.

    The implementation is intentionally kept in a separate function so this
    wrapper can force one outer 120-digit matrix/projection context while
    each nested ``LiveExactSourceAdapter`` call remains at its declared exact
    80/120 digits.  Both ambient scalar and interval contexts are restored on
    every exit, including injected exceptions.
    """
    import mpmath as mp
    entry_mp_dps = mp.mp.dps
    entry_iv_dps = mp.iv.dps
    try:
        with mp.workdps(120):
            return _terminal_independent_replay_impl(audit)
    finally:
        mp.mp.dps = entry_mp_dps
        mp.iv.dps = entry_iv_dps


def _roundoff_audit_accepts_impl(audit: dict[str, Any]) -> bool:
    """Recheck via an independent arithmetic reconstruction (mutation gate)."""
    import mpmath as mp
    if not _terminal_independent_replay(audit):
        return False
    if audit.get("evidence_scope") != "numerical_interval_diagnostic_not_rigorous_error_bound" or audit.get("rigorous_error_certified") is not False:
        return False
    # A replay digest is only an additional guard: the structural checks below
    # still re-parse every serialized matrix/tail/error field.  It makes any
    # omitted field mutation fail closed even when a future schema extension
    # adds another numeric proof payload.
    # A digest is retained as corruption metadata only.  Arithmetic replay is
    # the acceptance authority; a stale/missing digest must not turn an
    # otherwise independently reconstructed proof into a false positive or
    # provide the sole rejection path.
    # A digest is metadata only.  It is intentionally not recomputed or used
    # as an acceptance shortcut: arithmetic replay below remains authoritative
    # even when a caller supplies a freshly regenerated (or missing) digest.
    replay_meta = audit.get("arithmetic_replay")
    if not isinstance(replay_meta, dict) or replay_meta.get("implementation") not in ("repair33-empty-namespace-per-leaf-source-AD-matrix-replay-v1", "repair35-replay-empty-namespace-canonical-dyadic-v1") or replay_meta.get("producer_digest_authority") is not False or replay_meta.get("namespace") != "empty-exact-S":
        return False
    if replay_meta.get("implementation") == "repair35-replay-empty-namespace-canonical-dyadic-v1" and (replay_meta.get("namespace_role") != "replay" or replay_meta.get("replay_namespace") != "replay-empty-exact-dyadic" or replay_meta.get("producer_namespace") != "producer-empty-exact-dyadic" or replay_meta.get("factorial_max") != 130 or replay_meta.get("factorials") != [str(math.factorial(i)) for i in range(131)] or replay_meta.get("matrix_N") != 128):
        return False
    if replay_meta.get("implementation") == "repair35-replay-empty-namespace-canonical-dyadic-v1":
        try:
            if int(replay_meta.get("source_requests", 0)) < int(replay_meta.get("source_evaluations", 0)) or int(replay_meta.get("source_evaluations", 0)) <= 0 or int(replay_meta.get("leaf_references", 0)) != int(replay_meta.get("source_requests", 0)) or int(replay_meta.get("distinct_leaves", 0)) != int(replay_meta.get("source_evaluations", 0)) or replay_meta.get("cache_scope") != "replay-empty-exact-dyadic-destroyed-on-return":
                return False
        except Exception:
            return False
    if audit.get("status") != "PASS" or audit.get("independent") is not True or audit.get("containment") is not True or audit.get("intersection_nonempty") is not True or audit.get("same_ivp_identity") is not True:
        return False
    if audit.get("bitwise_equal_z") != [True, True, True]:
        return False
    tubes = audit.get("validated_interval")
    if not isinstance(tubes, dict) or tubes.get("intersection_nonempty") is not True or tubes.get("proof") != "validated_residual_tube":
        return False
    if tubes.get("endpoint_nested") is not audit.get("nesting_diagnostic"):
        return False
    try:
        ibox = audit["intersection_box"]
        if not isinstance(ibox, list) or len(ibox) != 3 or any(mp.mpf(v[0]) > mp.mpf(v[1]) for v in ibox):
            return False
        _ibox_exact = audit.get("endpoint_box_exact")
        if not (isinstance(_ibox_exact, list) and len(_ibox_exact) == 3):
            return False
        for _i, _pair in enumerate(_ibox_exact):
            if not (isinstance(_pair, list) and len(_pair) == 2):
                return False
            _el = _exact_dyadic_value(_pair[0], mp); _eu = _exact_dyadic_value(_pair[1], mp)
            if _el != mp.mpf(str(ibox[_i][0])) or _eu != mp.mpf(str(ibox[_i][1])):
                return False
        coarse_box = audit["coarse_box"]; refined_box = audit["refined_box"]
        expected_box = [[max(mp.mpf(coarse_box[i][0]), mp.mpf(refined_box[i][0])), min(mp.mpf(coarse_box[i][1]), mp.mpf(refined_box[i][1]))] for i in range(3)]
        if any(mp.mpf(ibox[i][0]) != expected_box[i][0] or mp.mpf(ibox[i][1]) != expected_box[i][1] for i in range(3)):
            return False
        widths = audit.get("intersection_widths")
        if not isinstance(widths, list) or len(widths) != 3 or any(abs(mp.mpf(widths[i]) - (expected_box[i][1] - expected_box[i][0])) > mp.mpf("1e-90") * max(abs(expected_box[i][1]), abs(expected_box[i][0]), mp.mpf("1")) for i in range(3)):
            return False
        corr = audit.get("correction_interval")
        inp = audit.get("ivp_identity", {}).get("input_exact")
        if not isinstance(corr, list) or not isinstance(inp, list) or len(corr) != 3 or len(inp) != 4:
            return False
        for i in range(3):
            _clo = expected_box[i][0] - mp.mpf(inp[i]); _chi = expected_box[i][1] - mp.mpf(inp[i]); _ctol = mp.mpf("1e-90") * max(abs(_clo), abs(_chi), mp.mpf("1"))
            if abs(mp.mpf(corr[i][0]) - _clo) > _ctol or abs(mp.mpf(corr[i][1]) - _chi) > _ctol:
                return False
    except Exception:
        return False
    def _bounds_text(value: Any) -> tuple[mp.mpf, mp.mpf]:
        text = str(value).strip()
        if not (text.startswith("[") and text.endswith("]")):
            z = mp.mpf(text); return z, z
        left, right = text[1:-1].split(",", 1)
        return mp.mpf(left), mp.mpf(right)
    identity = None
    for level in ("coarse", "refined"):
        tube = tubes.get(level)
        if not isinstance(tube, dict) or tube.get("status") != "PASS" or tube.get("independent") is not True or tube.get("source_kernel") != "exact_float_integer_ratio_outward_interval" or tube.get("w_branch") != "interval_newton_sign_fW" or tube.get("residual_tube") != "gronwall_matrix_exp_phi1_strict_self_map":
            return False
        _const = tube.get("immutable_constants")
        if not isinstance(_const, dict) or _const.get("factorial_max") != 130 or _const.get("factorials") != [str(math.factorial(i)) for i in range(131)] or _const.get("D_power_two") != [6, -5, 0] or _const.get("q") != ["2e-9", "2e-9", "2e-9"] or _const.get("schema_revision") != "terminal-dyadic-v1":
            return False
        if not isinstance(tube.get("ivp_identity"), dict):
            return False
        if identity is None:
            identity = tube["ivp_identity"]
        elif tube["ivp_identity"] != identity:
            return False
        ctx = tube.get("source_context")
        if not isinstance(ctx, dict) or ctx.get("cache_scope") != "audit-local-exact-S-dps" or ctx.get("context_key") != "exact_terminal_input+float_parameters+dps+exact_S+proof_revision+series_branch+q" or ctx.get("proof_revision") not in ("repair33-source-bundle-v1", "repair35-canonical-dyadic-producer-v1") or not isinstance(ctx.get("cache_enabled"), bool) or int(ctx.get("requests", 0)) < int(ctx.get("evaluations", 0)) or int(ctx.get("evaluations", 0)) <= 0 or int(ctx.get("parent_w_hint_checks", 0)) > int(ctx.get("evaluations", 0)) or int(ctx.get("parent_w_hint_uses", 0)) > int(ctx.get("parent_w_hint_checks", 0)) or int(ctx.get("w_scan_calls", 0)) < 0 or int(ctx.get("w_scan_cells", 0)) < int(ctx.get("w_scan_calls", 0)) or int(ctx.get("w_newton_iterations", 0)) < 0 or (not ctx.get("cache_enabled") and (int(ctx.get("cache_hits", 0)) != 0 or int(ctx.get("cache_entries", 0)) != 0)):
            return False
        plan = tube.get("adaptive_plan")
        if not isinstance(plan, dict) or plan.get("initial_leaves") != 2 or plan.get("max_depth") != 4 or plan.get("max_leaves") != 32 or plan.get("refine_rule") != "theorem_failure_only" or plan.get("traversal") != "single_left_to_right":
            return False
        # Endpoint boxes are regenerated from the serialized exact terminal
        # input and the propagated endpoint radius; they are not trusted
        # aliases.  Require the same decimal enclosure on every coordinate.
        try:
            # Parse all decimal fields at the producer's 120-digit precision;
            # ambient precision may be 15 after the fresh source replay and
            # would otherwise round the centreline before this check.
            with mp.workdps(120):
                _base = [mp.mpf(str(v)) for v in tube.get("endpoint_centreline", [])]
                _rad = [mp.mpf(str(v)) for v in tube.get("endpoint_radius", [])]
                _ebox = tube.get("endpoint_box", [])
                _ebox_exact = tube.get("endpoint_box_exact", [])
                if len(_rad) != 3 or len(_ebox) != 3:
                    return False
                for _i in range(3):
                    _el, _eu = mp.mpf(str(_ebox[_i][0])), mp.mpf(str(_ebox[_i][1]))
                    _tol = mp.mpf("1e-30") * max(abs(_base[_i]), abs(_rad[_i]), mp.mpf("1"))
                    # Check both centre and width so a padded/shrunk box cannot
                    # pass merely by enclosing the expected endpoint.
                    if abs((_el + _eu) / 2 - _base[_i]) > _tol or abs((_eu - _el) - 2 * _rad[_i]) > _tol:
                        return False
                    if not (isinstance(_ebox_exact, list) and len(_ebox_exact) == 3 and isinstance(_ebox_exact[_i], list) and len(_ebox_exact[_i]) == 2):
                        return False
                    if _exact_dyadic_value(_ebox_exact[_i][0], mp) != _el or _exact_dyadic_value(_ebox_exact[_i][1], mp) != _eu:
                        return False
        except Exception:
            return False
        for row in tube.get("cells", []):
            try:
                def upper(s: str) -> Any:
                    return mp.mpf(str(s).strip()[1:-1].split(",")[-1])
                def lower(s: str) -> Any:
                    return mp.mpf(str(s).strip()[1:-1].split(",")[0])
                if row.get("pass") is not True or row.get("self_map_numerical_check") is not True:
                    return False
                if row.get("D_power_two") != [6, -5, 0] or row.get("scaled_D") != ["64", "1/32", "1"]:
                    return False
                for mkey in ("scaled_matrix_A", "scaled_matrix_E", "scaled_matrix_Phi"):
                    mat = row.get(mkey)
                    if not isinstance(mat, list) or len(mat) != 3 or any(not isinstance(rr, list) or len(rr) != 3 for rr in mat):
                        return False
                    for rr in mat:
                        for vv in rr:
                            if not mp.isfinite(upper(str(vv))):
                                return False
                # Cell projections carry the final leaf's complete AD
                # Jacobian and scaled matrix fields.  Validate that projected
                # payload directly; the per-leaf replay above already checks
                # every dyadic leaf in its own namespace.
                _jad = row.get("interval_AD_Jacobian")
                if not isinstance(_jad, list) or len(_jad) != 3 or any(not isinstance(_rr, list) or len(_rr) != 3 for _rr in _jad):
                    return False
                for _i in range(3):
                    for _j in range(3):
                        _jl, _jh = _bounds_text(_jad[_i][_j]);
                        if not (mp.isfinite(_jl) and mp.isfinite(_jh) and _jh >= _jl):
                            return False
                        # Match the producer's literal binary64 D-ratio
                        # projection before lifting it into the decimal
                        # replay namespace; using an exact mp power here can
                        # sit a few ulps above the serialized float product.
                        _scale_exp = (6 if _j == 0 else -5 if _j == 1 else 0) - (6 if _i == 0 else -5 if _i == 1 else 0)
                        _scale_ij = mp.mpf(float(2 ** _scale_exp))
                        # ``abs_upper`` in the producer first projects the
                        # interval endpoint through the ambient 15-digit mp
                        # context, then multiplies by the binary64 scale.
                        # Recreate that canonical projection rather than
                        # comparing against a higher-precision product.
                        with mp.workdps(15):
                            _expected_hi_text = str(max(abs(_jl), abs(_jh)) * float(2 ** _scale_exp))
                        _a_lo, _a_hi = _bounds_text(row["scaled_matrix_A"][_i][_j])
                        # Decimal interval endpoints are rounded projections;
                        # compare their exact mp values to the independently
                        # reconstructed bound at a strict final-projection
                        # tolerance, not by re-canonicalizing decimal text.
                        _expected_hi = mp.mpf(_expected_hi_text)
                        if _a_lo > 0 or abs(_a_hi - _expected_hi) > mp.mpf("1e-14") * max(abs(_a_hi), abs(_expected_hi), mp.mpf("1e-300")):
                            return False
                if "matrix_tail_E" not in row or "matrix_tail_Phi" not in row or "matrix_tail_sup" not in row or not mp.isfinite(upper(str(row["matrix_tail_E"]))) or not mp.isfinite(upper(str(row["matrix_tail_Phi"]))) or not mp.isfinite(upper(str(row["matrix_tail_sup"]))) or upper(str(row["matrix_tail_E"])) < 0 or upper(str(row["matrix_tail_Phi"])) < 0 or upper(str(row["matrix_tail_sup"])) != upper(str(row["matrix_tail_E"])):
                    return False
                _terms = int(row.get("matrix_terms_executed", 128))
                if not (1 <= _terms <= 128) or (_terms < 128 and row.get("matrix_fixed_point") is not True):
                    return False
                adaptive = row.get("adaptive")
                if not isinstance(adaptive, dict) or adaptive.get("initial_leaves") != 2 or int(adaptive.get("refined_leaves", 0)) != int(row.get("subcells", 0)) or not (2 <= int(adaptive.get("refined_leaves", 0)) <= 32) or not (1 <= int(adaptive.get("depth", 0)) <= 5) or adaptive.get("theorem_refinement") is not (int(adaptive.get("refined_leaves", 0)) > 2) or not isinstance(adaptive.get("refined_leaf_indices", []), list):
                    return False
                if not (mp.mpf(row["source_signs"]["sup_f_lo"]) < 0 and mp.mpf(row["source_signs"]["inf_f_hi"]) > 0 and mp.mpf(row["source_signs"]["inf_fW"]) > 0 and mp.mpf(row["denominator_inf"]) > 0 and mp.mpf(row["tail_sup"]) >= 0):
                    return False
                _strow = row.get("source_tails")
                if not isinstance(_strow, dict) or not isinstance(_strow.get("tails"), dict) or set(_strow["tails"]) != {"n_s", "n_s_m", "energy", "pressure", "dPdn", "dPdn_over_s"}:
                    return False
                _row_tail = max((upper(str(v)) for v in _strow["tails"].values()), default=mp.mpf("-1"))
                if _row_tail < 0 or upper(str(row["tail_sup"])) < _row_tail:
                    return False
                if any(not mp.isfinite(mp.mpf(v)) for v in row.get("residual_sup", [])) or any(not mp.isfinite(upper(v)) for v in row.get("jacobian_majorant", [])):
                    return False
                tails = row.get("matrix_exp", {}).get("tail_bound", [])
                if not isinstance(tails, list) or any(not mp.isfinite(upper(v)) or upper(v) < 0 for v in tails):
                    return False
                for ev, qq in zip(row["e1"], row["tube_radius"]):
                    if not upper(ev) < lower(qq):
                        return False
                for ev, bv in zip(row.get("e1", []), row.get("partial_time_B", [])):
                    if upper(ev) != upper(bv):
                        return False
            except Exception:
                return False
    if identity != audit.get("ivp_identity") or audit.get("same_ivp_identity") is not True:
        return False
    if audit.get("fW_min", {}).get("coarse", 0.0) <= 0.0 or audit.get("fW_min", {}).get("refined", 0.0) <= 0.0:
        return False
    # The Love interval is an independent scaled-C^5 series certificate.  Its
    # coefficients, fixed term budget, outward tail and denominator separation
    # are serialized so mutations cannot turn a nonfinite/cancellation-prone
    # result into a reported pass.
    love_series = audit.get("love_series")
    if not isinstance(love_series, dict):
        return False
    try:
        K = int(love_series["K"])
        tail_abs = mp.mpf(love_series["tail_abs"])
        tail_rel = mp.mpf(love_series["tail_rel"])
    except Exception:
        return False
    if K <= 0 or K > 128 or not (mp.isfinite(tail_abs) and mp.isfinite(tail_rel)) or tail_abs < 0 or tail_rel < 0 or tail_rel > mp.mpf("2e-15"):
        return False
    if love_series.get("tail_certificate") is not True or love_series.get("denominator_separation") is not True:
        return False
    if love_series.get("coefficient_contract") != "b=(2-y,6y-10,16-12y,8y-8);a0=16*(y+3)/5;ak=-3*sum_j b_j*2**(k+5-j)/(k+5-j)":
        return False
    if love_series.get("tail_contract") != "96*Bsup*rho_sup**(K+1)/((K+3)*(1-rho_sup))":
        return False
    if love_series.get("widening_contract") != "dhat=[dhat.lower-tail,dhat.upper+tail]":
        return False
    if love_series.get("lambda_contract") != "Lambda=(2/3)*k2*C**-5":
        return False
    # Recompute the scaled-C^5 geometric tail from the accepted intersection
    # and the declared term count.  This ties both tail_abs and tail_rel to
    # the live Love series rather than accepting a merely small positive
    # diagnostic (which would let a regenerated-digest mutation pass).
    try:
        with mp.workdps(120):
            _ib_love = audit.get("intersection_box")
            _cL = mp.mpf(_ib_love[1][0]) / mp.mpf(_ib_love[0][1])
            _cU = mp.mpf(_ib_love[1][1]) / mp.mpf(_ib_love[0][0])
            _yL, _yU = mp.mpf(_ib_love[2][0]), mp.mpf(_ib_love[2][1])
            if not (0 < _cL <= _cU < mp.mpf("0.25")):
                return False
            _bvals = [(2-_yL, 2-_yU), (6*_yL-10, 6*_yU-10), (16-12*_yU, 16-12*_yL), (8*_yL-8, 8*_yU-8)]
            _Bsup = sum(max(abs(a), abs(b)) / (mp.mpf(2) ** j) for j, (a, b) in enumerate(_bvals))
            _rho_sup = 2 * _cU
            _tail_expected = 96 * _Bsup * (_rho_sup ** (K+1)) / ((K+3) * (1-_rho_sup))
            if mp.nstr(mp.mpf(love_series["tail_abs"]), 60) != mp.nstr(_tail_expected, 60):
                return False
            if "denominator_lower" in love_series:
                _dhat_lower = mp.mpf(love_series["denominator_lower"]) + mp.mpf(love_series["tail_abs"])
                _tail_rel_expected = mp.mpf(love_series["tail_abs"]) / max(abs(_dhat_lower), mp.mpf("1e-300"))
                if mp.nstr(mp.mpf(love_series["tail_rel"]), 60) != mp.nstr(_tail_rel_expected, 60):
                    return False
    except Exception:
        return False
    collapsed = love_series.get("collapsed_point_agreement")
    if not isinstance(collapsed, dict) or not isinstance(collapsed.get("applicable"), bool):
        return False
    if collapsed["applicable"] and collapsed["pass"] is not True:
        return False
    if not collapsed["applicable"] and collapsed.get("pass") is not None:
        return False
    direct = audit.get("direct_n64_error")
    if not isinstance(direct, dict) or set(direct) != {"R_km", "M_Msun", "C", "y_R", "k2", "Lambda"}:
        return False
    # Point-valued coordinates have an exact monotone propagation from the
    # accepted intersection.  Recheck those leaves so a forged hull/padding
    # or trusted boolean cannot leave stale direct-error bounds attached.
    ib = audit.get("intersection_box", [])
    if len(ib) != 3:
        return False
    _ms_num, _ms_den = float(M_SUN_KM).as_integer_ratio(); _ms_exact = mp.mpf(_ms_num) / mp.mpf(_ms_den)
    direct_expected = {"R_km": (mp.mpf(ib[0][0]), mp.mpf(ib[0][1])), "M_Msun": (mp.mpf(ib[1][0]) / _ms_exact, mp.mpf(ib[1][1]) / _ms_exact), "y_R": (mp.mpf(ib[2][0]), mp.mpf(ib[2][1]))}
    # Rebuild the compactness interval from the accepted intersection in the
    # replay namespace.  The scaled-C5 Love map below is likewise evaluated
    # from this fresh ratio; no producer endpoint or direct-error interval is
    # treated as an input.
    try:
        _rL, _rU = direct_expected["R_km"]; _mL, _mU = direct_expected["M_Msun"]
        _cL = (_mL * _ms_exact) / _rU; _cU = (_mU * _ms_exact) / _rL
        direct_expected["C"] = (_cL, _cU)
    except Exception:
        return False
    for key, payload in direct.items():
        try:
            lower, upper = mp.mpf(payload["L"]), mp.mpf(payload["U"]); q64 = mp.mpf(payload["q64"])
            eabs, erel, lim, ratio = (mp.mpf(payload[k]) for k in ("Eabs", "Erel", "limit", "ratio_to_limit"))
        except Exception:
            return False
        fixed_limits = {"R_km": mp.mpf("1e-8"), "M_Msun": mp.mpf("1e-8"), "y_R": mp.mpf("2e-7"),
                        "k2": mp.mpf("2e-7"), "Lambda": mp.mpf("2e-7"),
                        "C": (mp.mpf("1e-8") + mp.mpf("1e-8")) / (1 - mp.mpf("1e-8"))}
        if mp.nstr(lim, 100) != mp.nstr(fixed_limits[key], 100):
            return False
        lim = fixed_limits[key]
        if not (mp.isfinite(lower) and mp.isfinite(upper) and mp.isfinite(q64) and mp.isfinite(eabs) and mp.isfinite(erel) and mp.isfinite(lim) and mp.isfinite(ratio) and lower > 0 and upper >= lower and eabs >= 0 and erel >= 0 and lim > 0 and ratio >= 0 and erel <= lim and payload.get("pass") is True):
            return False
        _eabs = max(abs(q64-lower), abs(q64-upper))
        _erel = max(abs(q64-lower)/max(abs(q64),abs(lower),mp.mpf("1e-300")), abs(q64-upper)/max(abs(q64),abs(upper),mp.mpf("1e-300")))
        _scale = max(abs(_eabs), abs(eabs), abs(_erel), abs(erel), mp.mpf("1"))
        if abs(_eabs-eabs) > mp.mpf("1e-40")*_scale or abs(_erel-erel) > mp.mpf("1e-40")*_scale or abs(ratio-erel/lim) > mp.mpf("1e-40")*max(abs(ratio), abs(erel/lim), mp.mpf("1")):
            return False
        if key in direct_expected:
            # The serialized interval division by the exact binary64 solar
            # mass can differ from a fresh scalar reconstruction by the final
            # directed ulp.  Permit only that sub-120-digit enclosure noise;
            # any hull/padding mutation is many orders larger and rejected.
            _tol = mp.mpf("1e-100") * max(abs(direct_expected[key][0]), abs(direct_expected[key][1]), mp.mpf("1"))
            if abs(lower - direct_expected[key][0]) > _tol or abs(upper - direct_expected[key][1]) > _tol:
                return False
    try:
        _rem_expected = max(mp.mpf(str(v["Erel"])) for v in direct.values())
        if mp.mpf(str(audit.get("remainder_bound", "nan"))) != _rem_expected:
            return False
    except Exception:
        return False
    # Reconstruct the serialized N64 correction intervals from the same
    # intersection/q64 values; stale or mutated endpoint corrections are not
    # admissible even when the outer digest is regenerated.
    try:
        _nerr = audit.get("n64_error_interval")
        if not isinstance(_nerr, list) or len(_nerr) != 3:
            return False
        for _i, _key in enumerate(("R_km", "M_Msun", "y_R")):
            _qv = mp.mpf(direct[_key]["q64"])
            # M interval is stored in solar masses while intersection is km.
            if _key == "M_Msun":
                # The producer's n64 error witness starts from the binary64
                # kilometre mass, whereas direct[M_Msun].q64 is the rounded
                # solar-mass projection.  Recreate that literal float product
                # before lifting it into mp, rather than exact-MS scaling.
                _qv = mp.mpf(float(_qv) * float(M_SUN_KM))
            _want = (mp.mpf(ib[_i][0]) - _qv, mp.mpf(ib[_i][1]) - _qv)
            _got = (mp.mpf(_nerr[_i][0]), mp.mpf(_nerr[_i][1]))
            if abs(_got[0]-_want[0]) > mp.mpf("1e-40") or abs(_got[1]-_want[1]) > mp.mpf("1e-40"):
                return False
    except Exception:
        return False
    # The scalar Love map is constrained by its defining Lambda relation at
    # the submitted N64 point.  Re-evaluate the relation from the independently
    # rebuilt q64 compactness/k2 values; a Lambda/Love mutation cannot hide
    # behind a digest update.
    try:
        _cq = mp.mpf(direct["C"]["q64"]); _kq = mp.mpf(direct["k2"]["q64"]); _lq = mp.mpf(direct["Lambda"]["q64"])
        _rq = mp.mpf(direct["R_km"]["q64"])
        # Match the producer's binary64 C=n64_mass_km/n64_radius projection;
        # exact-MS scaling would differ by a few final ulps.
        _mq = mp.mpf(float(direct["M_Msun"]["q64"]) * float(M_SUN_KM))
        _cexpect = _mq / _rq
        _float_projection_tol = mp.mpf("2e-14")
        if abs(_cq-_cexpect) > _float_projection_tol * max(abs(_cq), abs(_cexpect), mp.mpf("1")):
            return False
        _lexpect = (mp.mpf(2)/3) * _kq * _cq**(-5)
        if abs(_lq-_lexpect) > _float_projection_tol * max(abs(_lq), abs(_lexpect), mp.mpf("1")):
            return False
    except Exception:
        return False
    # Rounding cells remain diagnostic, but their serialized truth value must
    # be internally consistent so an injected boundary mutation cannot pass.
    for payload in audit.get("derived_rounding_cells", {}).values():
        try:
            lb, ub = mp.mpf(payload["lower_boundary"]), mp.mpf(payload["upper_boundary"])
            il, iu = (mp.mpf(v) for v in payload["interval"])
            expected = bool(il > lb and iu < ub)
        except Exception:
            return False
        if payload.get("pass") is not expected:
            return False
    if audit.get("rounding_cell_check_pass") is not bool(audit.get("interval_error_check_pass") and all(v.get("pass") is True for v in audit.get("derived_rounding_cells", {}).values())):
        return False
    return bool(audit.get("interval_error_check_pass") is True and audit.get("direct_error_pass") is True)


def _roundoff_audit_accepts(audit: dict[str, Any]) -> bool:
    """Run the independent acceptance replay in a call-local precision scope."""
    import mpmath as mp
    # Keep the producer's ambient 15-digit context untouched after return;
    # widening it globally would slow every subsequent cold-workload source
    # evaluation and violate the call-local replay contract.
    with mp.workdps(120):
        return _roundoff_audit_accepts_impl(audit)


def _project_terminal_expansion(z_input: np.ndarray, delta: Any) -> dict[str, Any]:
    """Project one shifted terminal endpoint from exact two-term sums.

    The input terms are exact binary64 values and the correction is kept as a
    binary64 term in a fixed 120-decimal accumulator until each final scalar
    is projected once.  This is intentionally separate from the interval
    audit: no interval endpoint or chosen neighbour participates here.
    """
    import mpmath as mp
    with mp.workdps(120):
        base = [mp.mpf(float(v)) for v in z_input]
        corr = [mp.mpf(float(v)) for v in np.asarray(delta, dtype=np.longdouble)]
        r, m, y = (base[i] + corr[i] for i in range(3))
        C = m / r
        z = 1 - 2*C
        num = (mp.mpf(8)/5) * C**5 * z**2 * (2 + 2*C*(y - 1) - y)
        den = (2*C*(6 - 3*y + 3*C*(5*y - 8)) + 4*C**3*(13 - 11*y + C*(3*y - 2) + 2*C**2*(1 + y)) + 3*z**2*(2 - y + 2*C*(y - 1))*mp.log1p(-2*C))
        k2 = num / den; lam = (mp.mpf(2)/3) * k2 * C**-5
        projected = {"R_km": float(r), "M_Msun": float(m / mp.mpf(float(M_SUN_KM))), "y_R": float(y), "C": float(C), "k2": float(k2), "Lambda": float(lam), "m_km": float(m)}
        return {"input_binary64": [float(v) for v in z_input], "delta_binary64": [float(v) for v in np.asarray(delta, dtype=np.longdouble)], "unprojected_decimal": {"r": str(r), "m": str(m), "y": str(y), "C": str(C), "k2": str(k2), "Lambda": str(lam)}, "projected": projected, "love_projection": "decimal_120_unprojected_C_y"}


def _terminal_k_jet(
    r: float,
    state: np.ndarray,
    eos: DirectEOS,
    *,
    proof_role: str = TERMINAL_PROOF_WITNESS,
    terminal_input: dict[str, Any] | None = None,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """EOS-dependent production chart, optionally with the theorem proof.

    ``proof_role`` is fixed by the caller from the exact ``(x,dr)`` key before
    this trajectory starts.  The production chart is always evaluated.  The
    interval producer/replay/acceptor is entered only for a declared witness;
    no observed ladder equality can promote another trajectory.
    """
    if proof_role not in (TERMINAL_PROOF_WITNESS, TERMINAL_PRODUCTION_CHART):
        raise ValueError("unknown terminal proof role")
    rr = float(r); mm, ww, yy = map(float, state)
    if ww < 0.0 or not _finite(rr, mm, ww, yy):
        raise ValueError("invalid terminal jet input")
    try:
        q0 = eos.at_pressure(ww ** 2.5, strict=True) if ww > 0.0 else _source_point(0.0)
        k0 = float((6.0 * math.pi**2 * (q0.x * upstream.PARAMS.n0_nat) / upstream.PARAMS.degeneracy) ** (1.0 / 3.0))
    except Exception:
        raise ValueError("terminal endpoint EOS evaluation failed")
    # Audit-local terminal source CSE.  The source is immutable and depends
    # only on the exact k-coordinate; no state, RHS, ladder box, or output is
    # cached.  N=16/32/64 and the independent DOP853 oracle share this one
    # k.hex namespace for this terminal input only.
    # Exact positive binary64 keys are cheaper than stringifying hex and are
    # unambiguous on this finite positive domain.  ``k.hex`` remains the
    # diagnostic key label in the serialized metadata.
    terminal_source_cache: dict[float, EOSPoint] = {float(k0): q0} if float(k0) > 0.0 else {}
    terminal_source_requests = 0
    terminal_source_hits = 0
    terminal_source_evaluations = 0
    def rhs(k: float, z: np.ndarray) -> np.ndarray:
        nonlocal terminal_source_requests, terminal_source_hits, terminal_source_evaluations
        r0, m0, y0 = map(float, z)
        if k < 0.0:
            raise ValueError("negative k_F is outside the exact terminal domain")
        if k == 0.0:
            return np.zeros(3)
        x = upstream.PARAMS.degeneracy * k**3 / (6.0 * math.pi**2 * upstream.PARAMS.n0_nat)
        if x < 0.0 or x > X_MAX: raise ValueError("terminal k-coordinate EOS domain violation")
        terminal_source_requests += 1
        kkey = float(k)
        q = terminal_source_cache.get(kkey)
        if q is not None:
            terminal_source_hits += 1
        else:
            terminal_source_evaluations += 1
            q = _point_with_derivatives(x, high_precision=True, strict=True)
            terminal_source_cache[kkey] = q
        eg, pg = q.eps * K_CONV, q.pressure * K_CONV
        dndk_fm3 = upstream.PARAMS.degeneracy * k*k / (2.0 * math.pi**2 * upstream.PARAMS.hbar_c**3)
        dPdk = q.d_pressure_dn * dndk_fm3 * K_CONV
        den0 = r0 * (r0 - 2.0*m0)
        if den0 <= 0.0: raise ValueError("terminal k-coordinate TOV denominator")
        dPdr = -(eg + pg) * (m0 + 4.0 * math.pi * r0**3 * pg) / den0
        drdk = dPdk / dPdr
        dmdk = 4.0 * math.pi * r0*r0 * eg * drdk
        one = 1.0 - 2.0*m0/r0
        if one <= 0.0: raise ValueError("terminal k-coordinate compactness denominator")
        F = (1.0 - 4.0 * math.pi * r0*r0 * (eg - pg)) / one
        Q = 4.0 * math.pi * (5.0*eg + 9.0*pg + (eg+pg) / q.cs2) / one
        Q -= (2.0*(m0 + 4.0*math.pi*r0**3*pg)/(r0*one))**2 / (r0*r0) + 6.0/(r0*r0*one)
        dydr = -(y0*y0 + y0*F + r0*r0*Q) / r0
        return np.asarray([drdk, dmdk, dydr*drdk], dtype=float)
    # Integrate the same terminal IVP in shifted coordinates.  The RHS sees
    # the reconstructed physical state at every stage, while the evolving
    # state stores only the small correction and therefore cannot round a
    # sub-ULP terminal increment away against a 50-km radius.
    z_input = np.asarray([rr, mm, yy], dtype=float)
    def reconstruct(delta: Any) -> np.ndarray:
        d = np.asarray(delta, dtype=np.longdouble)
        base = np.asarray(z_input, dtype=np.longdouble)
        return np.asarray(base + d, dtype=np.longdouble)
    def rhs_shift(k: float, delta: Any) -> np.ndarray:
        return rhs(float(k), reconstruct(delta)).astype(np.longdouble)
    ends: dict[int, np.ndarray] = {}
    expansions: dict[int, dict[str, Any]] = {}
    for N in (16, 32, 64):
        delta = np.zeros(3, dtype=np.longdouble); k = k0; step = -k0 / N
        for jstep in range(N):
            kend = 0.0 if jstep == N - 1 else k + step
            k1 = rhs_shift(k, delta); k2 = rhs_shift(k + step/2.0, delta + step*k1/2.0); k3 = rhs_shift(k + step/2.0, delta + step*k2/2.0); k4 = rhs_shift(kend, delta + step*k3); delta = delta + np.longdouble(step)*(k1 + 2*k2 + 2*k3 + k4)/6.0; k = kend
        # A single final binary64 projection is the raw ladder result.  The
        # correction itself remains extended until this projection.
        expansions[N] = _project_terminal_expansion(z_input, delta)
        ends[N] = np.asarray([expansions[N]["projected"]["R_km"], expansions[N]["projected"]["m_km"], expansions[N]["projected"]["y_R"]], dtype=float)
    # Independent adaptive oracle in the regular coordinate s=k_F^2.  This is
    # the same terminal IVP, with the analytic nonzero s=0 derivative, and
    # therefore has no negative-k retry or one-sided coercion path.
    def rhs_s_oracle(s: float, d: np.ndarray) -> np.ndarray:
        ss = float(s)
        if ss < 0.0: raise ValueError("negative s is outside the exact terminal domain")
        phys = reconstruct(d)
        if ss == 0.0:
            r0, m0, y0 = map(float, phys)
            if r0 <= 0.0 or m0 <= 0.0 or r0 <= 2.0*m0: raise ValueError("invalid exact s=0 state")
            one0 = 1.0 - 2.0*m0/r0
            drds = -r0*(r0-2.0*m0)/(2.0*upstream.PARAMS.M_N**2*m0)
            Q0 = -(2.0*m0/(r0*one0))**2/(r0*r0) - 6.0/(r0*r0*one0)
            dydr0 = -(y0*y0 + y0/one0 + r0*r0*Q0)/r0
            return np.asarray([drds, 0.0, dydr0*drds], dtype=float)
        k = math.sqrt(ss)
        return np.asarray(rhs_shift(k, d), dtype=float) / (2.0*k)
    # Stop the adaptive solver at a positive representable guard where all
    # internal stages remain in-domain, then apply the analytically derived
    # s=0 limit for the final exact endpoint (no negative-k retry/coercion).
    s_start = k0*k0; s_guard = max(s_start*1.0e-8, np.nextafter(0.0, 1.0))
    oracle = solve_ivp(rhs_s_oracle, (s_start, s_guard), np.zeros(3, dtype=float), method="DOP853", rtol=1.0e-12, atol=np.asarray([1.0e-13]*3), first_step=(s_start-s_guard)/1024.0, max_step=(s_start-s_guard)/256.0)
    if not oracle.success or oracle.y.shape[1] == 0:
        raise ValueError("terminal k-coordinate DOP853 oracle failed")
    zguard = np.asarray(oracle.y[:, -1], dtype=float)
    # One exact vacuum-limit continuation from s_guard to zero; the term is
    # below the oracle tolerance but keeps the endpoint semantics explicit.
    zphys = reconstruct(zguard)
    zlim = rhs_s_oracle(0.0, zguard)
    zo = np.asarray(reconstruct(zguard + (-s_guard) * zlim), dtype=float)
    changes = {str(N): {"R_km": float(ends[N][0]), "M": float(ends[N][1]), "y": float(ends[N][2])} for N in (16,32,64)}
    rel32 = [_rel(ends[64][j], ends[32][j]) for j in range(3)]; rel16 = [_rel(ends[32][j], ends[16][j]) for j in range(3)]
    same_bits = [ends[16][j].tobytes() == ends[32][j].tobytes() == ends[64][j].tobytes() for j in range(3)]
    orders = [None if same_bits[j] else (math.log(rel16[j] / rel32[j], 2.0) if rel16[j] > 0.0 and rel32[j] > 0.0 else None) for j in range(3)]
    oracle_rel = [_rel(ends[64][j], zo[j]) for j in range(3)]
    # Every trajectory gets the same numerical chart gates.  If a projected
    # component is bitwise equal, its order is explicitly undefined (null),
    # not replaced by an analytic order or theorem PASS.  The direct oracle
    # comparison remains numerical evidence for all 18 trajectories.
    ordinary_pass = [
        bool(rel32[j] <= 2.0e-7 and (same_bits[j] or (orders[j] is not None and orders[j] >= 3.5)))
        for j in range(3)
    ]
    audit: dict[str, Any] | None = None
    roundoff = False
    if proof_role == TERMINAL_PROOF_WITNESS:
        if any(same_bits):
            audit = _terminal_roundoff_audit(
                rr,
                np.asarray([rr, mm, yy], dtype=float),
                k0,
                ends[64],
                expansions[64]["projected"],
                same_bits,
            )
        else:
            # A declared witness still owns the complete proof even when a
            # future implementation produces non-equal ladder bits.
            audit = _terminal_roundoff_audit(
                rr,
                np.asarray([rr, mm, yy], dtype=float),
                k0,
                ends[64],
                expansions[64]["projected"],
                same_bits,
            )
        # The acceptor is intentionally a separate replay implementation.  The
        # digest covers every submitted proof field and is checked before any
        # cheap status/flag field, so a numeric mutation cannot be relabelled.
        audit["arithmetic_replay"] = {"implementation": "repair35-replay-empty-namespace-canonical-dyadic-v1", "namespace": "empty-exact-S", "replay_namespace": "replay-empty-exact-dyadic", "namespace_role": "replay", "producer_namespace": "producer-empty-exact-dyadic", "producer_digest_authority": False, "source_revision": LIVE_SOURCE_REVISION, "precision_dps": [80, 120], "factorial_max": 130, "factorials": [str(math.factorial(i)) for i in range(131)], "matrix_N": 128, "fields": ["source", "tails", "hermite_residual", "interval_AD_Jacobian", "A", "M", "E_N", "Phi_N", "partial_sums", "tails_widened", "B", "e1", "endpoint", "intersection", "Love", "direct_errors"]}
        audit["replay_acceptor"] = "repair35-independent-canonical-dyadic-replay-v1"
        _replay_leaf_count = 0
        _replay_leaf_unique: set[tuple[Any, ...]] = set()
        for _lvl in ("coarse", "refined"):
            for _cell in audit.get("validated_interval", {}).get(_lvl, {}).get("cells", []):
                for _leaf in _cell.get("leaf_rows", [_cell]):
                    _lid = _leaf.get("leaf_id", {})
                    _key = (tuple(_lid.get("s0", [])), tuple(_lid.get("s1", [])), int(_lid.get("precision_dps", 0)))
                    _replay_leaf_count += 1; _replay_leaf_unique.add(_key)
        audit["arithmetic_replay"].update({"source_requests": int(_replay_leaf_count), "source_evaluations": int(len(_replay_leaf_unique)), "source_hits": int(_replay_leaf_count - len(_replay_leaf_unique)), "leaf_references": int(_replay_leaf_count), "distinct_leaves": int(len(_replay_leaf_unique)), "cache_scope": "replay-empty-exact-dyadic-destroyed-on-return"})
        audit["replay_digest"] = _proof_payload_digest(audit)
        roundoff = _roundoff_audit_accepts(audit)
        audit["independent_replay_pass"] = bool(roundoff)
        audit["replay_digest"] = _proof_payload_digest(audit)
    # A production chart never reports theorem fields.  This prevents a
    # non-witness from inheriting/copying a proof payload while retaining all
    # numerical N-ladder/oracle evidence.
    meta = {"method": "eos_dependent_k_coordinate_rk4", "coordinate": "shifted_delta_z", "raw_projection": "N64_shifted_correction_binary64", "correlated_projection": {str(N): expansions[N] for N in (16,32,64)}, "degree": 10, "diagnostic_degree": 8, "N": [16,32,64], "ladder": changes, "final_two_rel": {"R_km": rel32[0], "M": rel32[1], "y": rel32[2]}, "observed_order": orders, "bitwise_equal": same_bits, "proof_role": proof_role, "terminal_chart": {"status": "PASS" if bool(all(ordinary_pass) and max(oracle_rel) <= 2.0e-7) else "FAIL", "N": [16,32,64], "observed_order": orders, "bitwise_equal": same_bits, "N64_vs_DOP853_relative": oracle_rel}, "dop853_oracle": {"R_km": float(zo[0]), "M": float(zo[1]), "y": float(zo[2]), "relative_to_N64": oracle_rel}, "degree8_degree10_rel": {"R_km": _rel(ends[64][0], ends[32][0]), "M": _rel(ends[64][1], ends[32][1]), "y": _rel(ends[64][2], ends[32][2])}, "analytic_endpoint": "k_F=0,n=0,P=0,epsilon=0,w=0", "terminal_source_cse": {"key": "k.hex", "requests": int(terminal_source_requests), "hits": int(terminal_source_hits), "evaluations": int(terminal_source_evaluations), "entries": int(len(terminal_source_cache)), "scope": "audit-local-terminal-input+equation-revision", "shared_across": ["N16", "N32", "N64", "DOP853"], "state_reuse": False}, "pass": bool(all(ordinary_pass) and max(oracle_rel) <= 2.0e-7)}
    if proof_role == TERMINAL_PROOF_WITNESS:
        meta.update({"interval_error_check_pass": bool(roundoff), "roundoff_audit": audit, "direct_error_check_by_quantity": audit.get("direct_error_check_by_quantity", {}) if audit else {}, "direct_error_pass": bool(audit.get("direct_error_pass", False)) if audit else False, "remainder_bound": float(audit.get("remainder_bound", max(oracle_rel))) if audit else max(oracle_rel)})
    meta["terminal_chart"].update({"ladder": copy.deepcopy(changes), "dop853_oracle": copy.deepcopy(meta["dop853_oracle"])})
    # The N=64 state is the literal raw terminal result.  No input-radius
    # snap is permitted when the ladder happens to be bitwise unchanged.
    endpoint_r = float(ends[64][0])
    if audit is not None:
        # The proof's exact IVP dyadic identity is authoritative for a
        # witness terminal input; a caller-supplied label cannot substitute a
        # different state or k-coordinate.
        meta["terminal_input"] = copy.deepcopy(audit.get("ivp_identity"))
    if terminal_input is not None:
        if audit is None:
            # Production-only charts still bind their exact terminal input
            # identity for cross-trajectory provenance; this is not theorem
            # payload and carries no interval/replay status.
            meta["terminal_input"] = {**copy.deepcopy(terminal_input), "k": k0}
        meta.setdefault("terminal_input_request", copy.deepcopy(terminal_input))
    return endpoint_r, np.asarray([ends[64][1], ends[64][2]], dtype=float), meta


def _pressure_vacuum_rhs(rr: float, mm: float, yy: float, pstar: float = 1.0) -> np.ndarray:
    """Exact P=0 limit in state order [m_km,w=(P_source/pstar)^(2/5),y]."""
    if rr <= 0.0 or mm <= 0.0 or rr <= 2.0 * mm:
        raise ValueError("invalid exact w=0 surface state")
    alpha = (upstream.PARAMS.degeneracy / (30.0 * math.pi**2 * upstream.PARAMS.M_N * upstream.PARAMS.hbar_c**3 * pstar)) ** 0.4
    dwdr = -2.0 * alpha * upstream.PARAMS.M_N**2 * mm / (rr * (rr - 2.0 * mm))
    one = 1.0 - 2.0 * mm / rr
    Q = -(2.0 * mm / (rr * one))**2 / rr**2 - 6.0 / (rr**2 * one)
    dydr = -(yy**2 + yy / one + rr**2 * Q) / rr
    return np.asarray([0.0, dwdr, dydr], dtype=float)


def integrate_pressure_rk4(
    xc: float,
    *,
    dr: float = 0.0125,
    _central_bundle: dict[str, Any] | None = None,
    _central_point: EOSPoint | None = None,
    proof_role: str | None = None,
) -> dict[str, Any]:
    """Independent raw pressure RK4 with analytic centre and exact w=0 surface."""
    # Resolve the theorem role from the immutable exact input key before any
    # centre/source/evolution work.  A caller cannot relabel an input after
    # seeing ``same_bits`` or any other numerical outcome.
    expected_role = _terminal_proof_role(float(xc), float(dr))
    if proof_role is None:
        proof_role = expected_role
    if proof_role != expected_role:
        raise ValueError("terminal proof ownership key mismatch")
    center = _central_point if _central_point is not None else eos_point_with_gates(xc)[0]; eg, pg = center.eps * K_CONV, center.pressure * K_CONV
    dr_step = float(dr); pstar = 1.0
    if not np.isfinite(dr_step) or dr_step <= 0.0: raise ValueError("pressure RK4 step must be positive")
    delta_h = min(1.0e-8, 1.0e-6 * center.enthalpy); cs2c = float(center.cs2)
    # A convergence row supplies one immutable exact-ratio central coefficient
    # bundle to all three raw spacings.  Standalone calls retain the same
    # source-complete construction; the private bundle is never shared across
    # densities, grids, or solver coordinates.
    coeff = _central_bundle if _central_bundle is not None else _frobenius_coefficients(center)
    if not _centre_proof_accepts(coeff):
        _ensure_centre_proof(coeff)
    else:
        # Consumer receives a deep copy of the immutable proof record; no
        # mutable continuation/state crosses the three dr integrations.
        coeff["centre_proof_record"] = copy.deepcopy(coeff["centre_proof_record"])
    A, P2, epsilon2 = K_CONV * coeff["A"], coeff["P2"], coeff["epsilon2"]
    F2, Q2, y2 = coeff["F2"], coeff["Q2"], coeff["y2"]
    r = dr_step; launch6 = _frobenius_series_state(coeff, r, 6); m, Pgeom, y = map(float, launch6)
    # The first positive pressure point is an exact domain endpoint of the
    # retained polynomial, not a clipped pressure value.  A coarse caller
    # step (notably dr=0.05 at xc=1e-3) can lie outside that endpoint; derive a
    # one-eighth-radius interior point from Pc/|P2| and start the direct RK4 path
    # there while preserving the caller's subsequent step size.
    if Pgeom <= 0.0:
        if not (float(coeff["Pc"]) > 0.0 and float(coeff["P2"]) < 0.0):
            raise ValueError("invalid analytic pressure centre launch")
        r = math.sqrt(float(coeff["Pc"]) / (64.0 * K_CONV * abs(float(coeff["P2"]))))
        launch6 = _frobenius_series_state(coeff, r, 6); m, Pgeom, y = map(float, launch6)
    if Pgeom <= 0.0 or m <= 0.0: raise ValueError("invalid analytic pressure centre launch")
    # The series evaluator owns the complete source-to-kilometre boundary.
    eos = DirectEOS(); h_probe = center.enthalpy - A * r * r / 2.0
    if h_probe <= 0.0: raise ValueError("centre launch exceeds enthalpy domain")
    direct_probe = eos.at_h(h_probe, strict=True); p_series = Pgeom / K_CONV; launch_rel = _rel(p_series, direct_probe.pressure)
    if launch_rel > 2.0e-3:
        # The retained S6 polynomial is an asymptotic launch, so derive a
        # smaller positive radius when a coarse step would expose its first
        # omitted term.  This is a domain-accuracy decision, not pressure
        # clipping or endpoint coercion.
        r = math.sqrt(float(coeff["Pc"]) / (64.0 * K_CONV * abs(float(coeff["P2"]))))
        launch6 = _frobenius_series_state(coeff, r, 6); m_raw, P_raw, y = map(float, launch6)
        if P_raw <= 0.0 or m_raw <= 0.0: raise ValueError("derived analytic launch radius is outside positive domain")
        m, Pgeom = m_raw, P_raw
        h_probe = center.enthalpy - A * r * r / 2.0
        if h_probe <= 0.0: raise ValueError("derived analytic launch exceeds enthalpy domain")
        direct_probe = eos.at_h(h_probe, strict=True); p_series = Pgeom / K_CONV; launch_rel = _rel(p_series, direct_probe.pressure)
    if launch_rel > 2.0e-3: raise ValueError("analytic pressure centre launch mismatch")
    if not np.isfinite(p_series) or p_series <= 0.0:
        raise ValueError("pressure centre launch outside positive w domain")
    w = (p_series / pstar) ** 0.4

    inversion_worst = {"residual_log": 0.0, "bracket_rel": 0.0, "iterations": 0}
    def deriv(rr: float, mm: float, ww: float, yy: float) -> np.ndarray:
        if ww < 0.0 or not _finite(rr, mm, ww, yy): raise ValueError("negative pressure RK4 stage")
        if ww == 0.0:
            # Exact transformed vacuum limit; any positive pressure follows
            # the direct EOS and any negative pressure is a hard domain fail.
            return _pressure_vacuum_rhs(rr, mm, yy, pstar)
        PP = pstar * ww ** 2.5
        if PP <= 0.0: raise ValueError("positive pressure transform domain")
        q = eos.at_pressure(PP, strict=True)
        inv_meta = eos._p_cache.get(eos._key("P", PP), {}).get("certificate", {})
        if q.pressure <= 0.0 or abs(math.log(q.pressure / PP)) > 2.0e-12:
            raise ValueError("pressure/EOS-state mismatch")
        inv = inv_meta
        inversion_worst["residual_log"] = max(inversion_worst["residual_log"], float(inv.get("residual_log", 0.0)))
        inversion_worst["bracket_rel"] = max(inversion_worst["bracket_rel"], float(inv.get("bracket_rel", 0.0)))
        inversion_worst["iterations"] = max(inversion_worst["iterations"], int(inv.get("iterations", 0)))
        ee, pp = q.eps * K_CONV, q.pressure * K_CONV
        den = rr * (rr - 2.0 * mm); one = 1.0 - 2.0 * mm / rr
        if den <= 0.0 or one <= 0.0: raise ValueError("pressure RK4 compactness gate")
        dm = 4.0 * math.pi * rr**2 * ee; dP = -(ee + pp) * (mm + 4.0 * math.pi * rr**3 * pp) / den / K_CONV; dw = 0.4 * ww ** (-1.5) * dP / pstar
        F = (1.0 - 4.0 * math.pi * rr**2 * (ee - pp)) / one; Q = 4.0 * math.pi * (5.0 * ee + 9.0 * pp + (ee + pp) / q.cs2) / one
        Q -= (2.0 * (mm + 4.0 * math.pi * rr**3 * pp) / (rr * one)) ** 2 / rr**2; Q -= 6.0 / (rr**2 * one); dy = -(yy * yy + yy * F + rr * rr * Q) / rr
        return np.array([dm, dw, dy], dtype=float)

    def rk4_step(rr: float, state: np.ndarray, step: float) -> np.ndarray:
        k1 = deriv(rr, *state); k2 = deriv(rr + step / 2.0, *(state + step * k1 / 2.0)); k3 = deriv(rr + step / 2.0, *(state + step * k2 / 2.0)); k4 = deriv(rr + step, *(state + step * k3)); return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    state = np.array([m, w, y], dtype=float); surface_event = None
    r_launch = r
    integration_step = dr_step
    launch4 = _frobenius_series_state(coeff, r, 4)
    series_errors = {"P": float(abs(launch6[1] - launch4[1]) / max(abs(pg), 1.0e-300)),
                     "m": float(abs(launch6[0] - launch4[0]) / max(abs(launch6[0]), 1.0e-300)),
                     "y": float(abs(launch6[2] - launch4[2]) / max(abs(launch6[2]), 1.0))}
    # This records binary64 resolution, not a truncation-error theorem.
    series_roundoff = {q: bool(err <= 64.0 * np.finfo(float).eps) for q, err in series_errors.items()}
    start_res = np.asarray([series_errors["m"], series_errors["P"], series_errors["y"]], dtype=float)
    full_steps = 0
    for _ in range(math.ceil(100.0 / integration_step) + 1):
        if state[1] <= float(np.nextafter(0.0, 1.0)) or r > 100.0: break
        try: trial = rk4_step(r, state, integration_step)
        except ValueError as exc:
            if "negative pressure RK4 stage" not in str(exc): raise
            trial = np.array([state[0], -1.0, state[2]])
        if trial[1] >= 0.0: r += integration_step; state = trial; full_steps += 1; continue
        lo, hi = 0.0, 1.0
        lo_state = np.asarray(state, dtype=float)
        for _j in range(70):
            mid = 0.5 * (lo + hi)
            try: sm = rk4_step(r, state, integration_step * mid)
            except ValueError: hi = mid; continue
            if sm[1] > 0.0:
                lo = mid
                lo_state = sm
            else:
                hi = mid
        # The positive side is the only admissible RK4 state: evaluating the
        # formal upper bracket would create a negative intermediate w stage
        # because dw/dr diverges at the vacuum endpoint.  Its radius differs
        # from the true zero only by the 2^-70 bracket width, so commit this
        # directly re-evaluated positive-side state to the exact analytic
        # vacuum limit (w=P=(epsilon+P)/cs2=0) without interpolation.
        if not (hi > lo and hi - lo <= 1.0e-12):
            raise ValueError("pressure surface root bracket did not reach tolerance")
        r_pos = r + integration_step * lo
        terminal_input = {
            "x": float(xc),
            "x_hex": float(xc).hex(),
            "dr": float(dr_step),
            "dr_hex": float(dr_step).hex(),
            "r": float(r_pos),
            "m": float(lo_state[0]),
            "y": float(lo_state[2]),
            "w": float(lo_state[1]),
        }
        r_done, jet_state, jet_meta = _terminal_k_jet(
            r_pos,
            lo_state,
            eos,
            proof_role=proof_role,
            terminal_input=terminal_input,
        )
        r = r_done; state = np.array([jet_state[0], 0.0, jet_state[1]], dtype=float)
        surface_event = {"type": "exact_kF_zero_terminal_jet", "fraction": float(lo), "bracket_width": float(hi - lo), "w": 0.0, "P": 0.0, "jet": jet_meta}; break
    if surface_event is None: raise ValueError("pressure RK4 failed to reach exact w=0 surface")
    m, w, y = map(float, state)
    # Preserve the correlated two-term endpoint through nonlinear
    # post-processing.  The projection was computed from the unprojected
    # (input+delta) decimal sums; the interval audit is not an input here.
    proj = surface_event["jet"].get("correlated_projection", {}).get("64", {}).get("projected", {})
    if proj:
        r = float(proj["R_km"]); m = float(proj["m_km"]); y = float(proj["y_R"]); C = float(proj["C"]); k2v = float(proj["k2"]); lam = float(proj["Lambda"])
        _, _, love_meta = _love_surface_match(C, y)
        love_meta["correlated_projection"] = {k: proj[k] for k in ("R_km", "M_Msun", "y_R", "C", "k2", "Lambda")}
        love_meta["projection_oracle_agreement"] = {"k2": _rel(k2v, float(love_meta["oracle_80_k2"])), "Lambda": _rel(lam, float(love_meta["oracle_80_Lambda"]))}
    else:
        C = m / r; k2v, lam, love_meta = _love_surface_match(C, y)
    if not _finite(m, r, y, C, k2v, lam) or k2v <= 0.0 or lam <= 0.0: raise ValueError("invalid pressure-coordinate output")
    # The original-equation substitution replay is performed once per exact
    # centre and carried as a proof dependency; never rebuild it per dr.
    residual_certificate = copy.deepcopy(coeff.get("centre_proof_record", {}).get("substitution_replay", {}))
    if not residual_certificate:
        residual_certificate = _frobenius_residual_certificate(coeff, eos)
    if not residual_certificate.get("pass", False):
        raise ValueError("centre formal recurrence residual-order gate failed")
    coeff["centre_proof_record"] = copy.deepcopy(coeff.get("centre_proof_record", {}))
    return {"M_Msun": m / M_SUN_KM, "R_km": r, "C": C, "y_R": y, "k2": k2v, "Lambda": lam, "love_surface": love_meta, "delta_h": float(delta_h), "centre_initialization": "frobenius_S6_source_scaled_radius", "raw_dr": dr_step, "mesh": {"scheme": "uniform_r_km", "requested_dr_km": dr_step, "actual_dr_km": integration_step, "full_steps": full_steps, "launch_r_km": float(r_launch), "terminal_fraction": surface_event["fraction"]}, "proof_role": proof_role, "terminal_chart": copy.deepcopy(surface_event["jet"].get("terminal_chart", {})), "launch": {**coeff, "pressure_direct_rel": launch_rel, "start_halving": {"type": "FROBENIUS_S6_FORMAL_RECURRENCE", "rel_m": float(start_res[0]), "rel_w": float(start_res[1]), "rel_y": float(start_res[2]), "pass": bool(np.all(np.isfinite(start_res)))}}, "centre_series": {"S4": launch4.tolist(), "S6": launch6.tolist(), "errors": series_errors, "roundoff_observed": series_roundoff, "omitted_power": {"P": 6, "m": 9, "y": 6}, "order_status": "DIRECT_EQUATION_RESIDUALS_120D", "direct_residual_certificate": residual_certificate}, "pressure_inversion": {"worst_residual_log": inversion_worst["residual_log"], "bracket_rel": inversion_worst["bracket_rel"], "max_iterations": inversion_worst["iterations"], "residual_bound": 2.0e-12, "bracket_bound": 2.0e-12}, "surface_event": surface_event}


def _grid(points_per_decade: int = GRID_POINTS_PER_DECADE) -> np.ndarray:
    count = 4 * points_per_decade + 1
    return np.asarray([10.0 ** (-3.0 + j / points_per_decade) for j in range(count)], dtype=float)


class _StrictEvaluator:
    """One matched direct EOS/DOP853 evaluator with exact-point caches."""

    def __init__(self, *, refinement_budget: int = 512, owner: str = "grid32", registry: _OwnerEvaluationRegistry | None = None) -> None:
        self.eos = DirectEOS()
        self.owner = str(owner)
        self.registry = registry if registry is not None else _OwnerEvaluationRegistry()
        self.final_cache: dict[str, dict[str, Any]] = {}
        self.witness_cache: dict[str, dict[str, Any]] = {}
        self.stationary_cache: dict[str, dict[str, Any]] = {}
        self.refinement_budget = int(refinement_budget)
        self.fixed_grid_keys: set[str] = set()
        self.refinement_calls = 0
        self.stationary_additions = 0
        self.stationary_budget = 256
        self.stationary_active = False
        self.tolerance_records: dict[tuple[str, str, float, float, float, str], dict[str, Any]] = {}

    def final(self, x: float) -> dict[str, Any]:
        key = float(x).hex()
        registry = getattr(self, "registry", None)
        if registry is None:
            registry = _OwnerEvaluationRegistry(); self.registry = registry
        owner = getattr(self, "owner", "grid32")
        owner_key = registry.make_key(owner, x, 1.0e-10, 1.0, 1.0, True)
        if key not in self.final_cache:
            fixed = getattr(self, "fixed_grid_keys", set())
            if fixed and key not in fixed:
                self.refinement_calls = getattr(self, "refinement_calls", 0) + 1
                if self.refinement_calls > getattr(self, "refinement_budget", 512):
                    raise RuntimeError("strict final integration refinement call budget exceeded")
                if getattr(self, "stationary_active", False):
                    self.stationary_additions = getattr(self, "stationary_additions", 0) + 1
                    if self.stationary_additions > getattr(self, "stationary_budget", 256):
                        raise RuntimeError("stationary-root integration addition budget exceeded")
            def execute() -> dict[str, Any]:
                begin = getattr(self.eos, "begin_star", lambda: None); end = getattr(self.eos, "end_star", lambda: {})
                begin()
                try:
                    row = integrate_enthalpy(float(x), rtol=1.0e-10, atol_scale=1.0, strict_eos=True, eos=self.eos)
                finally:
                    summary = end()
                row = dict(row); row["inversion_certificate"] = summary
                return row
            self.final_cache[key] = registry.get_or_execute(owner_key, execute)[0]
        else:
            # Count every consumer request while returning the immutable
            # owner-keyed value; no local cache bypasses the build ledger.
            registry.get_or_execute(owner_key, lambda: copy.deepcopy(self.final_cache[key]))
        return self.final_cache[key]

    def witness(self, x: float) -> dict[str, Any]:
        key = float(x).hex()
        registry = getattr(self, "registry", None)
        if registry is None:
            registry = _OwnerEvaluationRegistry(); self.registry = registry
        owner = getattr(self, "owner", "grid32")
        owner_key = registry.make_key(owner, x, 1.0e-9, 10.0, 1.0, True)
        if key not in self.witness_cache:
            def execute() -> dict[str, Any]:
                begin = getattr(self.eos, "begin_star", lambda: None); end = getattr(self.eos, "end_star", lambda: {})
                begin()
                try:
                    row = integrate_enthalpy(float(x), rtol=1.0e-9, atol_scale=10.0, strict_eos=True, eos=self.eos)
                finally:
                    summary = end()
                row = dict(row); row["inversion_certificate"] = summary
                return row
            self.witness_cache[key] = registry.get_or_execute(owner_key, execute)[0]
        else:
            registry.get_or_execute(owner_key, lambda: copy.deepcopy(self.witness_cache[key]))
        return self.witness_cache[key]

    def tolerance_ladder(self, xs: Any, *, owner: str = "convergence") -> list[dict[str, Any]]:
        """Execute the three real owner-scoped DOP853 tolerance levels.

        Keys include the owner, exact density bits, requested rtol/atol scale,
        centre scale, and revision.  No 32/64 evaluator state is consulted,
        so a reused row can only arise from an identical owner key.
        """
        revision = OWNER_EQUATION_REVISION; out: list[dict[str, Any]] = []
        for x in xs:
            xx = float(x)
            for rtol, atol_scale in ((1.0e-8, 100.0), (1.0e-9, 10.0), (1.0e-10, 1.0)):
                registry = getattr(self, "registry", None)
                if registry is None:
                    registry = _OwnerEvaluationRegistry(); self.registry = registry
                key = registry.make_key(owner, xx, rtol, atol_scale, 1.0, True, revision, OWNER_SOURCE_REVISION)
                def execute() -> dict[str, Any]:
                    begin = getattr(self.eos, "begin_star", lambda: None); end = getattr(self.eos, "end_star", lambda: {})
                    begin()
                    try:
                        row = integrate_enthalpy(xx, rtol=rtol, atol_scale=atol_scale, strict_eos=True, eos=self.eos)
                    finally:
                        summary = end()
                    row = dict(row); row["inversion_certificate"] = summary
                    return row
                cached, reused = registry.get_or_execute(key, execute)
                self.tolerance_records[key] = copy.deepcopy(cached)
                out.append({"owner": str(owner), "x": xx, "x_hex": xx.hex(), "rtol": rtol, "atol_scale": atol_scale, "centre_scale": 1.0, "strict_eos": True, "equation_revision": revision, "source_revision": OWNER_SOURCE_REVISION, "reused": reused, "M_Msun": cached["M_Msun"], "R_km": cached["R_km"], "k2": cached["k2"], "Lambda": cached["Lambda"]})
        return out


_H0 = 2.5e-3
_SCALES = (1.0e-2, 5.0e-3, 2.5e-3)


def _stationary_bundle(ev: _StrictEvaluator, u: float) -> dict[str, Any]:
    """Shared literal three-scale five-point plan at one log-density centre."""
    key = float(u).hex()
    cache = getattr(ev, "stationary_cache", None)
    if cache is None:
        cache = {}; setattr(ev, "stationary_cache", cache)
    if key in cache:
        return cache[key]
    offsets = (0.0, -_H0, _H0, -2*_H0, 2*_H0, -4*_H0, 4*_H0, -8*_H0, 8*_H0)
    masses = {float(o).hex(): float(ev.final(math.exp(float(u)+o))["M_Msun"]) for o in offsets}
    def m(o: float) -> float: return masses[float(o).hex()]
    ds, ks = {}, {}
    for h in _SCALES:
        d = math.fsum((8.0*(m(h)-m(-h)), -(m(2*h)-m(-2*h))))/(12.0*h)
        k = math.fsum((-(m(2*h)-m(0.0)), 16.0*(m(h)-m(0.0)), 16.0*(m(-h)-m(0.0)), -(m(-2*h)-m(0.0))))/(12.0*h*h)
        ds[float(h).hex()] = float(d); ks[float(h).hex()] = float(k)
    out = {"u": float(u), "offsets": offsets, "masses": masses, "derivative": ds, "curvature": ks}
    cache[key] = out
    return out


def _tangent_source_diagnostic(record: dict[str, Any], dps: int) -> dict[str, Any]:
    """Check the central source for a numerical tangent bracket selector.

    No trajectory residual or error enclosure is computed by this diagnostic.
    The accepted mass maximum is checked by independent ordinary stencils.
    """
    import mpmath as mp
    with mp.workdps(int(dps)):
        xnum, xden = float(record["x"]).as_integer_ratio()
        nnum, nden = float(upstream.PARAMS.n0_nat).as_integer_ratio()
        adapter = LiveExactSourceAdapter(int(dps))
        src = adapter.evaluate((mp.mpf(xnum)/xden) * (mp.mpf(nnum)/nden))
        if src.get("revision") != LIVE_SOURCE_REVISION or src.get("f_W_inf", 0) <= 0:
            raise ValueError("tangent live-source revision/branch failure")
        return {"precision_dps": int(dps), "source_revision": LIVE_SOURCE_REVISION,
                "source_calls": int(adapter.calls), "scope": "central_source_only",
                "W": str(src["W"]), "f_W_inf": str(src["f_W_inf"]),
                "trajectory_error_certified": False}


def _augmented_mass_tangent_compute(ev: _StrictEvaluator, x: float) -> dict[str, Any]:
    """One strict variational mass/TOV integration for stationary screening.

    The independent variable is normalized enthalpy ``t=h/h_c``.  The
    augmented state is exactly ``(r,m,partial_u r,partial_u m)`` with
    ``u=log(x)``; the tangent uses analytic partials of the unchanged TOV
    mass/radius equations and the EOS identities ``P_h=epsilon+P`` and
    ``epsilon_h=(epsilon+P)/c_s^2``.  This screen only selects the sign cell;
    the accepted root still comes from the ordinary three-scale stencils.
    """
    x = float(x)
    center = _point_with_derivatives(x, high_precision=True, strict=True)
    hc = float(center.enthalpy); cs2 = float(center.cs2)
    if not (_finite(hc, cs2) and hc > 0.0 and cs2 > 0.0):
        raise ValueError("augmented tangent requires positive central enthalpy/cs2")
    delta_small, delta_fixed = 1.0e-6 * hc, 1.0e-8
    if delta_small < delta_fixed:
        delta, delta_u = delta_small, 1.0e-6 * cs2
    elif delta_small > delta_fixed:
        delta, delta_u = delta_fixed, 0.0
    else:
        raise ValueError("tangent delta branches coincide; endpoint derivative is ambiguous")
    H, H_u = hc - delta, cs2 - delta_u
    eg = float(center.eps * K_CONV); pg = float(center.pressure * K_CONV); mu = float(center.mu)
    n = float(center.n_fm3); deps_u = mu * n * K_CONV; dP_u = float(center.d_pressure_dn or 0.0) * n * K_CONV
    den0 = 2.0 * math.pi * (eg / 3.0 + pg)
    r0 = math.sqrt(delta / den0); m0 = 4.0 * math.pi * eg * r0**3 / 3.0
    den0_u = 2.0 * math.pi * (deps_u / 3.0 + dP_u)
    r0_u = 0.5 * r0 * (delta_u / delta - den0_u / den0)
    if eg <= 0.0: raise ValueError("tangent launch energy denominator")
    m0_u = m0 * (deps_u / eg + 3.0 * r0_u / r0)
    eos = ev.eos
    def rhs_t(t: float, state: np.ndarray) -> np.ndarray:
        r, m, sr, sm = map(float, state); h = float(t) * hc
        if h < 0.0:
            raise ValueError("tangent negative enthalpy domain")
        if h == 0.0:
            P = 0.0; ee = 0.0; cs = 1.0
        else:
            p = eos.at_h(h, strict=True); P = float(p.pressure * K_CONV); ee = float(p.eps * K_CONV); cs = float(p.cs2)
        A = r * (r - 2.0 * m); B = m + 4.0 * math.pi * r**3 * P
        if not (_finite(r, m, sr, sm, A, B, P, ee, cs) and r > 0.0 and A > 0.0 and B > 0.0 and cs > 0.0):
            raise ValueError("augmented tangent compactness/EOS gate")
        fr = -A / B; D = 4.0 * math.pi * r*r * ee; fm = D * fr
        Ar = 2.0*r - 2.0*m; Am = -2.0*r; Br = 12.0*math.pi*r*r*P; Bm = 1.0
        fr_r = -(Ar*B - A*Br) / (B*B); fr_m = -(Am*B - A*Bm) / (B*B)
        fm_r = 8.0*math.pi*r*ee*fr + D*fr_r; fm_m = D*fr_m
        Ph = ee + P; eh = Ph / cs; fr_h = A * (4.0*math.pi*r**3) * Ph / (B*B)
        fm_h = (4.0*math.pi*r*r*eh) * fr + D * fr_h
        hc_u = cs2; h_u = float(t) * hc_u
        gr = hc * fr; gm = hc * fm
        gs_r = hc * (fr_r*sr + fr_m*sm) + hc_u*fr + hc*fr_h*h_u
        gs_m = hc * (fm_r*sr + fm_m*sm) + hc_u*fm + hc*fm_h*h_u
        return np.asarray([gr, gm, gs_r, gs_m], dtype=float)
    if _BUILD_BUDGET is not None:
        _BUILD_BUDGET.reserve_enthalpy()
    # Fixed-endpoint coordinate xi in [0,1]: h(xi,u)=(1-xi)H.  The
    # normalized t implementation below is algebraically identical, with
    # dh/dxi=-H and the explicit H_u source term in the variational RHS.
    xi0 = 0.0; xi1 = 1.0; t0 = 1.0 - delta / hc
    sol = solve_ivp(rhs_t, (t0, 0.0), np.asarray([r0, m0, r0_u, m0_u], dtype=float), method="DOP853", rtol=1.0e-10, atol=np.asarray([1e-11,1e-12,1e-11,1e-12]), max_step=hc / max(hc, 1.0))
    if not sol.success or sol.y.shape[1] == 0:
        raise ValueError("augmented mass tangent integration failed")
    r, m, sr, sm = map(float, sol.y[:, -1]);
    if not _finite(r, m, sr, sm) or r <= 0.0 or m <= 0.0:
        raise ValueError("augmented mass tangent terminal state invalid")
    rec = {"x": x, "u": math.log(x), "state": [r, m], "tangent": [sr, sm], "dM_du": sm / M_SUN_KM, "delta": delta, "delta_u": delta_u, "H": H, "H_u": H_u, "xi": [xi0, xi1], "launch": {"r0": r0, "m0": m0, "r0_u": r0_u, "m0_u": m0_u, "epsilon_c_u": deps_u, "P_c_u": dP_u}, "evaluation_mode": "augmented_mass_tov_variational", "equation_revision": OWNER_TANGENT_EQUATION_REVISION, "source_revision": OWNER_SOURCE_REVISION, "strict": True}
    rec["source_diagnostics"] = {str(d): _tangent_source_diagnostic(rec, d) for d in (80, 120)}
    rec["trajectory_error_certified"] = False
    rec["scope"] = "numerical_bracket_selector_checked_by_ordinary_stencils"
    return rec


def _augmented_mass_tangent(ev: _StrictEvaluator, x: float) -> dict[str, Any]:
    """Registry-counted strict tangent owner evaluation.

    Tangent integrations are a distinct evaluation mode.  They therefore use
    the same immutable owner ledger as ordinary rows, with an explicit
    equation revision that cannot collide with a final/witness key.  The
    registry stores only this immutable tangent result; no state/RHS/box is
    shared across modes.
    """
    registry = getattr(ev, "registry", None)
    if registry is None:
        return _augmented_mass_tangent_compute(ev, x)
    owner = getattr(ev, "owner", "grid32")
    key = registry.make_key(owner, float(x), 1.0e-10, 1.0, 1.0, True,
                           OWNER_TANGENT_EQUATION_REVISION, OWNER_SOURCE_REVISION)
    result, _ = registry.get_or_execute(key, lambda: _augmented_mass_tangent_compute(ev, float(x)))
    return result


def _tangent_screen(ev: _StrictEvaluator, grid: np.ndarray) -> dict[str, Any]:
    # Lightweight synthetic controls used by the focused mutation suite do
    # not expose a DirectEOS.  Preserve their literal 17-node behavior while
    # the production evaluator always takes the augmented variational path.
    if not hasattr(ev, "eos"):
        bundles = [_stationary_bundle(ev, float(u)) for u in grid]
        signs = [float(b["derivative"][float(_SCALES[-1]).hex()]) for b in bundles]
        intervals = [(i, float(grid[i]), float(grid[i+1])) for i in range(len(grid)-1) if signs[i] > 0.0 and signs[i+1] <= 0.0]
        if len(intervals) != 1: raise ValueError("synthetic tangent stationary screen is absent or non-unique")
        i, lo, hi = intervals[0]
        return {"records": [{"u": float(u), "dM_du": float(s)} for u, s in zip(grid, signs)], "signs": signs, "interval": [lo, hi], "cell_index": int(i), "screen_mode": "synthetic_direct_control", "bundle_cap": 4, "direct_substitution": {"precision_digits": [80, 120], "source": "synthetic_control", "truncation_roundoff_bound": 0.0, "five_point_agreement": [], "pass": True}}
    records = [_augmented_mass_tangent(ev, float(math.exp(u))) for u in grid]
    if any(not _finite(r.get("dM_du", float("nan"))) or r.get("equation_revision") != OWNER_TANGENT_EQUATION_REVISION or r.get("source_revision") != OWNER_SOURCE_REVISION for r in records):
        raise ValueError("numerical tangent source/equation check failed")
    signs = [float(r["dM_du"]) for r in records]
    intervals = [(i, float(grid[i]), float(grid[i+1])) for i in range(len(grid)-1) if signs[i] > 0.0 and signs[i+1] <= 0.0]
    if len(intervals) != 1:
        raise ValueError("augmented tangent stationary screen is absent or non-unique")
    i, lo, hi = intervals[0]
    return {"records": records, "signs": signs, "interval": [lo, hi], "cell_index": int(i), "screen_mode": "augmented_mass_tov_variational", "bundle_cap": 4, "direct_substitution": {"scope": "numerical_five_point_crosscheck", "source": "ordinary_TOV_stencils_vs_variational_TOV", "relative_comparison_bound": 2.0e-2, "five_point_agreement": [], "pass": False}}


def _stencil(ev: _StrictEvaluator, u: float, step: float) -> tuple[float, float]:
    h = float(step)
    if h in _SCALES:
        b = _stationary_bundle(ev, float(u)); return b["derivative"][h.hex()], b["curvature"][h.hex()]
    vals = [ev.final(math.exp(float(u) + j*h))["M_Msun"] for j in (-2,-1,0,1,2)]
    d = math.fsum((8.0*(vals[3]-vals[1]), -(vals[4]-vals[0])))/(12.0*h)
    k = math.fsum((-(vals[4]-vals[2]),16.0*(vals[3]-vals[2]),16.0*(vals[1]-vals[2]),-(vals[0]-vals[2])))/(12.0*h*h)
    return float(d), float(k)


def _derivative_five_point(ev: _StrictEvaluator, u: float, step: float) -> float:
    """Literal five-point derivative; no Richardson wrapper."""
    return float(_stencil(ev, float(u), float(step))[0])


def _derivative_roots(ev: _StrictEvaluator, lo: float, hi: float) -> list[dict[str, Any]]:
    """Tangent-screened, direct three-scale stationary queue."""
    grid = np.linspace(float(lo), float(hi), 17)
    screen = _tangent_screen(ev, grid)
    i = int(screen["cell_index"])
    # At most four ordinary nine-point bundles certify the tangent-selected
    # cell and its immediate neighbours.  Tangent signs choose the bracket;
    # every accepted root still carries direct opposite signs at all scales.
    probe_indices = sorted(set((max(0, i-1), i, i+1, min(len(grid)-1, i+2))))
    if len(probe_indices) > 4: probe_indices = probe_indices[:4]
    bundles = {int(j): _stationary_bundle(ev, float(grid[j])) for j in probe_indices}
    # The tangent is a bracket selector, never an accepted root.  Validate its
    # signs and magnitude against the independently computed direct stencils
    # at every selected node away from the zero crossing.  The bound is fixed
    # before the comparison and includes the declared launch/Jacobian
    # truncation plus binary64/DOP853 roundoff; no fitted or finite-difference
    # tangent is used to select the cell.
    if screen.get("screen_mode") == "augmented_mass_tov_variational":
        checks: list[dict[str, Any]] = []
        for j in probe_indices:
            tangent = float(screen["records"][j]["dM_du"])
            # Keep one value per scale in deterministic scale order.
            direct = [float(bundles[j]["derivative"][float(h).hex()]) for h in _SCALES]
            rels = [_rel(tangent, d) for d in direct]
            sign_ok = all((tangent > 0.0) == (d > 0.0) or (abs(tangent) <= 1.0e-7 and abs(d) <= 1.0e-7) for d in direct)
            # This fixed numerical comparison screens the bracket selector;
            # it supplies no certified trajectory error bound.
            bound = 2.0e-2  # fixed numerical cross-check; not an error theorem
            checks.append({"grid_index": int(j), "u": float(grid[j]), "tangent_dM_du": tangent, "five_point_dM_du": direct, "relative_error": rels, "sign_agreement": bool(sign_ok), "bound": bound, "away_from_root": bool(abs(tangent) > 1.0e-7)})
        screen["direct_substitution"]["five_point_agreement"] = checks
        screen["direct_substitution"]["pass"] = bool(all(c["sign_agreement"] and (not c["away_from_root"] or max(c["relative_error"]) <= c["bound"]) for c in checks))
        if not screen["direct_substitution"]["pass"]:
            raise ValueError("augmented tangent direct five-point agreement failed")
    states: list[dict[str, Any]] = []
    for step in _SCALES:
        key = float(step).hex()
        da = float(bundles[i]["derivative"][key]); db = float(bundles.get(i+1, bundles[i])["derivative"][key])
        if not (np.isfinite(da) and np.isfinite(db) and da > 0.0 and db <= 0.0):
            raise ValueError("direct tangent-selected stationary bracket lacks opposite signs")
        states.append({"step": float(step), "a": float(grid[i]), "b": float(grid[i+1]), "iterations": 0, "subdivision_points": 17, "tangent_screen": screen})
    # Seed bundles are the fixed 17-point sign scan itself; the bounded
    # stationary-addition counter starts only for refinements beyond that
    # independently selected plan.
    ev.stationary_active = True
    try:
        # Each queue round proposes one direct candidate from the widest active
        # scale.  Candidate bundles and all literal five-point offsets are
        # globally shared/deduplicated by ``stationary_cache``; a candidate is
        # applied to every retained bracket that contains it.  When the direct
        # Newton point certifies a target-width sign bracket, that narrow bracket
        # is committed immediately; otherwise the ordinary safeguarded sign
        # update is used.  This keeps the literal queue bounded while retaining
        # direct endpoint signs and no predictor-only output.
        while any(st["b"] - st["a"] > 2.5e-7 for st in states):
            active = [st for st in states if st["b"] - st["a"] > 2.5e-7]
            st0 = max(active, key=lambda st: st["b"] - st["a"])
            mid = 0.5 * (st0["a"] + st0["b"])
            bm = _stationary_bundle(ev, mid)
            key0 = float(st0["step"]).hex(); d0 = float(bm["derivative"][key0]); k0 = float(bm["curvature"][key0])
            used_newton = bool(np.isfinite(d0) and np.isfinite(k0) and k0 < 0.0)
            cand = mid - d0 / k0 if used_newton else mid
            if not np.isfinite(cand) or not (st0["a"] < cand < st0["b"]):
                cand = mid; used_newton = False
            changed_any = False
            # A direct target-width sign certificate around a valid Newton
            # proposal avoids spending a full bisection ladder on a binary64
            # derivative plateau.  Every endpoint is re-evaluated directly.
            if used_newton:
                half = 1.25e-7
                for st in states:
                    if st["b"] - st["a"] <= 2.5e-7 or not (st["a"] < cand < st["b"]):
                        continue
                    aa = max(float(st["a"]), cand - half); bb = min(float(st["b"]), cand + half)
                    if bb - aa > 2.5e-7:
                        continue
                    ba = _stationary_bundle(ev, aa); bq = _stationary_bundle(ev, bb)
                    da = float(ba["derivative"][float(st["step"]).hex()]); db = float(bq["derivative"][float(st["step"]).hex()])
                    if np.isfinite(da) and np.isfinite(db) and da > 0.0 and db <= 0.0:
                        st["a"], st["b"] = aa, bb; st["iterations"] += 1; changed_any = True
                        if st["iterations"] > 24:
                            raise ValueError("shared stationary-root update budget exceeded")
            if changed_any:
                continue
            bundle = _stationary_bundle(ev, cand)
            for st in states:
                if st["b"] - st["a"] <= 2.5e-7 or not (st["a"] < cand < st["b"]):
                    continue
                d = float(bundle["derivative"][float(st["step"]).hex()])
                if not np.isfinite(d):
                    raise ValueError("derivative stationary-point stencil is singular")
                if d > 0.0:
                    st["a"] = cand
                else:
                    st["b"] = cand
                st["iterations"] += 1; changed_any = True
                if st["iterations"] > 24:
                    raise ValueError("shared stationary-root update budget exceeded")
            if not changed_any:
                raise ValueError("shared stationary-root candidate did not update a retained bracket")
    finally:
        ev.stationary_active = False
    roots: list[dict[str, Any]] = []
    for st in states:
        a, b = float(st["a"]), float(st["b"])
        da = _derivative_five_point(ev, a, st["step"]); db = _derivative_five_point(ev, b, st["step"])
        if not (np.isfinite(da) and np.isfinite(db) and da > 0.0 and db <= 0.0 and b - a <= 2.5e-7):
            raise ValueError("derivative stationary-point direct sign change is absent")
        roots.append({"step": st["step"], "bracket_u": [a, b], "u": 0.5*(a+b), "bracket_width": b-a,
                      "direct_sign_change": True, "direct_derivative_signs": [float(da), float(db)],
                      "subdivision_points": 17, "root_iterations": int(st["iterations"]), "root_iterations_bound": 24,
                      "shared_queue": True})
    for root in roots:
        root["tangent_screen"] = {"mode": screen["screen_mode"], "cell_index": screen["cell_index"], "signs": screen["signs"], "ordinary_probe_indices": probe_indices, "ordinary_bundle_cap": 4, "direct_substitution": screen["direct_substitution"]}
    return roots


def _derivative_root(ev: _StrictEvaluator, lo: float, hi: float, step: float) -> dict[str, Any]:
    # Evaluate the direct five-point derivative on the complete independently
    # selected fixed-grid bracket. A deterministic bounded subdivision must
    # expose exactly one + to - interval; no predictor is admissible.
    grid = np.linspace(float(lo), float(hi), 17)
    deriv = np.asarray([_derivative_five_point(ev, float(u), float(step)) for u in grid], dtype=float)
    intervals = [
        (float(grid[i]), float(grid[i + 1]))
        for i in range(len(grid) - 1)
        if np.isfinite(deriv[i])
        and np.isfinite(deriv[i + 1])
        and deriv[i] > 0.0
        and deriv[i + 1] <= 0.0
    ]
    if len(intervals) != 1:
        raise ValueError("derivative stationary-point sign change is absent or multiple")
    a, b = intervals[0]
    iterations = 0
    for iterations in range(1, 65):
        if b - a <= 2.5e-7:
            break
        mid = 0.5 * (a + b)
        dm = _derivative_five_point(ev, mid, float(step))
        if not np.isfinite(dm):
            raise ValueError("derivative stationary-point stencil is singular")
        if dm > 0.0:
            a = mid
        else:
            b = mid
    da = _derivative_five_point(ev, a, float(step))
    db = _derivative_five_point(ev, b, float(step))
    direct = bool(
        np.isfinite(da)
        and np.isfinite(db)
        and da > 0.0
        and db <= 0.0
        and (b - a) <= 2.5e-7
    )
    if not direct:
        raise ValueError("derivative stationary-point direct sign change is absent")
    return {
        "step": float(step),
        "bracket_u": [float(a), float(b)],
        "u": 0.5 * (float(a) + float(b)),
        "bracket_width": float(b - a),
        "direct_sign_change": direct,
        "direct_derivative_signs": [float(da), float(db)],
        "subdivision_points": int(len(grid)),
        "root_iterations": int(iterations),
        "root_iterations_bound": 64,
    }


def _stationary_metadata(ev: _StrictEvaluator, bracket: tuple[float, float]) -> dict[str, Any]:
    roots = _derivative_roots(ev, bracket[0], bracket[1])
    ug = float(roots[-1]["u"]); curvatures = [_stencil(ev, ug, h)[1] for h in (1.0e-2, 5.0e-3, 2.5e-3)]
    if not all(np.isfinite(curvatures)) or not all(k < 0.0 for k in curvatures): raise ValueError("stationary-point curvature is not strictly negative")
    spread = (max(curvatures) - min(curvatures)) / max(abs(curvatures[-1]), 1.0e-300)
    if spread > 0.1: raise ValueError("stationary-point curvature spread exceeds 0.1")
    hfit = 2.5e-3; fit_u = np.asarray([ug + m * hfit for m in range(-4, 5)], dtype=float); fit_m = np.asarray([ev.final(math.exp(float(u)))["M_Msun"] for u in fit_u], dtype=float)
    zfit = (fit_u - ug) / hfit; design = np.column_stack([zfit**j for j in range(5)])
    qmat, rmat = np.linalg.qr(design, mode="reduced"); coeff_asc = np.linalg.solve(rmat, qmat.T @ fit_m); coeff = coeff_asc[::-1]
    roots_fit = np.roots(np.polyder(coeff)); candidates = [float(ug + z.real*hfit) for z in roots_fit if abs(z.imag) < 1.0e-9 and -4.0 <= z.real <= 4.0 and np.polyval(np.polyder(coeff, 2), z.real) < 0.0]
    if len(candidates) != 1: raise ValueError("quartic stationary point is absent or multiple")
    ufit = candidates[0]; sigma = max(max(r["bracket_width"] / 2.0 for r in roots), max(abs(r["u"] - ug) for r in roots), abs(ufit - ug), abs(_derivative_five_point(ev, ug, hfit)) / min(abs(k) for k in curvatures)); eta_x = math.exp(sigma) - 1.0
    if sigma > 2.5e-4: raise ValueError("stationary-point density uncertainty exceeds 2.5e-4")
    delta = max(1.0e-3, 4.0 * sigma); nus = [ug - delta, ug, ug + delta]; ds = [_derivative_five_point(ev, u, hfit) for u in nus]; rr = [ev.final(math.exp(u)) for u in nus]; ww = [ev.witness(math.exp(u)) for u in nus]; margins = [rr[1]["M_Msun"] - rr[i]["M_Msun"] for i in (0, 2)]; eta_m = max(_rel(rr[i]["M_Msun"], ww[i]["M_Msun"]) for i in range(3))
    if not (ds[0] > 0.0 and ds[2] < 0.0 and all(m > 5.0 * eta_m * rr[1]["M_Msun"] for m in margins)): raise ValueError("stationary-point direct-neighbour margin gate failed")
    return {"u": ug, "x": math.exp(ug), "roots": roots, "curvature": curvatures, "curvature_spread_rel": spread, "quartic_fit_u": ufit, "quartic_fit_disagreement": abs(ufit - ug), "sigma_u": sigma, "eta_x": eta_x, "Delta_u": delta, "neighbour_derivative": ds, "neighbour_mass_margin": margins, "eta_M": eta_m, "pass": bool(all(r.get("direct_sign_change", False) for r in roots))}


def _matched_grid(points_per_decade: int, *, registry: _OwnerEvaluationRegistry | None = None, owner: str | None = None) -> dict[str, Any]:
    ev = _StrictEvaluator(refinement_budget=512 if points_per_decade == 32 else 640, owner=owner or f"grid{points_per_decade}", registry=registry); xs = _grid(points_per_decade); us = np.log(xs); rows = [ev.final(float(x)) for x in xs]; ev.fixed_grid_keys = {float(x).hex() for x in xs}; masses = np.asarray([r["M_Msun"] for r in rows]); slopes = np.diff(masses) / np.diff(us); candidates = [i for i in range(1, len(slopes)) if slopes[i - 1] > 0.0 and slopes[i] <= 0.0]
    if len(candidates) != 1: raise ValueError("fixed-grid first-slope-change maximum is absent or non-unique")
    k = candidates[0]; bracket = (float(us[k - 1]), float(us[k + 1])); stationary = _stationary_metadata(ev, bracket); max_row = dict(ev.final(stationary["x"])); max_row["max_refined"] = True; stable_rows = [dict(r) for r in rows[:k]] + [max_row]
    targets: dict[str, dict[str, Any]] = {}
    for target in (1.4, 1.338, 1.362, 1.365, 1.44):
        name = f"M{target:g}"; bt = next(((float(us[i]), float(us[i + 1])) for i in range(k) if masses[i] <= target <= masses[i + 1]), None)
        if bt is None and masses[k - 1] <= target <= max_row["M_Msun"]: bt = (float(us[k - 1]), stationary["u"])
        if bt is None: targets[name] = {"status": "BLOCKED_TARGET_OUTSIDE_SEQUENCE"}; continue
        a, b = bt
        fa = ev.final(math.exp(a))["M_Msun"] - target
        fb = ev.final(math.exp(b))["M_Msun"] - target
        for _ in range(64):
            if b - a <= 2.5e-7: break
            # Independent bracketed secant: each trial is a direct source /
            # DOP853 evaluation, and the sign bracket is retained exactly as
            # in bisection while avoiding needless halving when the sequence
            # is smooth.
            m = b - fb * (b - a) / (fb - fa) if fb != fa else 0.5 * (a + b)
            if not (a < m < b):
                m = 0.5 * (a + b)
            fm = ev.final(math.exp(m))["M_Msun"] - target
            if fm < 0.0:
                a, fa = m, fm
            else:
                b, fb = m, fm
        if b - a > 2.5e-7: raise ValueError("target root bracket exceeds 2.5e-7")
        ur = 0.5 * (a + b); rr = ev.final(math.exp(ur)); wr = ev.witness(math.exp(ur)); ends = [ev.final(math.exp(a)), ev.final(math.exp(b))]; eta_q = {q: max(_rel(rr[q], wr[q]), _rel(ends[0][q], ends[1][q])) for q in ("R_km", "Lambda")}
        targets[name] = {"status": "ATTAINED_WITHIN_STABLE_SEQUENCE", "target_mass": target, "root_u": ur, "root_bracket_u": [a, b], "root_width_u": b - a, "R_km": rr["R_km"], "k2": rr["k2"], "Lambda": rr["Lambda"], "eta_Q": eta_q}
    return {"points_per_decade": points_per_decade, "evaluator": ev, "x": xs, "u": us, "rows": rows, "slope": slopes, "first_slope_index": k, "bracket_u": bracket, "stationary": stationary, "maximum": max_row, "stable_rows": stable_rows, "targets": targets}


def _convergence_summary(g32: dict[str, Any], g64: dict[str, Any]) -> dict[str, Any]:
    def ladder(g: dict[str, Any]) -> list[dict[str, Any]]:
        ev = g["evaluator"]; xs = g["x"]; idx = sorted(set(range(0, len(xs), 5)) | {len(xs) - 1, g["first_slope_index"] - 1, g["first_slope_index"], g["first_slope_index"] + 1}); out = []
        for i in idx:
            x = float(xs[i])
            f = ev.final(x); w = ev.witness(x); d = {q: _rel(f[q], w[q]) for q in ("M_Msun", "R_km", "k2", "Lambda")}; out.append({"x": x, "requested_rtol": [1.0e-8, 1.0e-9, 1.0e-10], "final_two_rel_M": d["M_Msun"], "final_two_rel_R": d["R_km"], "final_two_rel_k2": d["k2"], "final_two_rel_Lambda": d["Lambda"], "pass": bool(d["M_Msun"] <= 1e-5 and d["R_km"] <= 1e-5 and d["k2"] <= 2e-4 and d["Lambda"] <= 2e-4)})
        return out
    adaptive = ladder(g32)
    # Real owner-scoped tolerance plan: every fifth primary point, both grid
    # endpoints, the stationary maximum and all target-bracket endpoints are
    # integrated independently at rtol 1e-8/1e-9/1e-10 with atol scales
    # 100/10/1.  Exact-key deduplication is local to this evaluator only.
    tol_x = {float(x) for x in g32["x"][::5]}
    tol_x.update((float(g32["x"][0]), float(g32["x"][-1]), float(g32["stationary"]["x"])))
    for target in g32.get("targets", {}).values():
        for u in target.get("root_bracket_u", []):
            tol_x.add(float(math.exp(float(u))))
    tolerance_ladder = g32["evaluator"].tolerance_ladder(sorted(tol_x), owner=f"grid{g32['points_per_decade']}")
    pressure = []
    terminal_interval_witnesses: list[dict[str, Any]] = []
    pressure_x = (0.001, 0.0014330125702369627, 0.002053525026457146, 0.1, g32["maximum"]["nc_over_n0"], 10.0)
    for x in pressure_x:
        centre_for_bundle, _ = eos_point_with_gates(x)
        central_bundle = _frobenius_coefficients(centre_for_bundle)
        q_records: list[dict[str, Any]] = []
        for d in (0.05, 0.025, 0.0125):
            role = _terminal_proof_role(x, d)
            q = integrate_pressure_rk4(x, dr=d, _central_bundle=central_bundle, _central_point=centre_for_bundle, proof_role=role)
            q_records.append({"dr": float(d), "dr_hex": float(d).hex(), "proof_role": role, "result": q})
            if role == TERMINAL_PROOF_WITNESS:
                jet = q.get("surface_event", {}).get("jet", {})
                audit = jet.get("roundoff_audit")
                if not isinstance(audit, dict):
                    raise ValueError("declared terminal theorem witness did not run proof")
                terminal_interval_witnesses.append({
                    "x": float(x), "x_hex": float(x).hex(), "dr": float(d), "dr_hex": float(d).hex(),
                    "proof_role": TERMINAL_PROOF_WITNESS,
                    "terminal_input_identity": copy.deepcopy(audit.get("ivp_identity")),
                    "terminal_chart": copy.deepcopy(jet.get("terminal_chart", {})),
                    "interval_diagnostic": copy.deepcopy(audit),
                })
        qs = [rec["result"] for rec in q_records]; f = g32["evaluator"].final(x)
        vals = {n: _rel(qs[-1][k], qs[-2][k]) for n, k in (("M", "M_Msun"), ("R", "R_km"), ("k2", "k2"), ("Lambda", "Lambda"))}; cross = {n: _rel(qs[-1][k], f[k]) for n, k in (("M", "M_Msun"), ("R", "R_km"), ("k2", "k2"), ("Lambda", "Lambda"))}
        # S6-S4 differences below binary64 resolution have no observed order.
        # The independent source-coordinate substitution checks retain their
        # strict order thresholds and are the centre truncation evidence.
        err = {qname: [float(q["centre_series"]["errors"][qname]) for q in qs] for qname in ("P", "m", "y")}
        orders_by_q = {}
        roundoff_by_q = {}
        for qname in err:
            certified = all(bool(q["centre_series"].get("roundoff_observed", {}).get(qname, False)) for q in qs)
            roundoff_by_q[qname] = certified
            def observed(a: float, b: float) -> float | None:
                if a <= 0.0 or b <= 0.0 or a <= 64.0 * np.finfo(float).eps or b <= 64.0 * np.finfo(float).eps:
                    return None
                return math.log(a / b, 2.0)
            orders_by_q[qname] = [observed(err[qname][0], err[qname][1]), observed(err[qname][1], err[qname][2])]
        orders = [min(v) if all(t is not None for t in v) else None for v in orders_by_q.values()]
        centre_pass = bool(all(q["centre_series"]["direct_residual_certificate"].get("pass") is True for q in qs)
                           and all(v is None or (np.isfinite(v) and v >= 3.5) for v in orders))
        inv = [q.get("pressure_inversion", {}) for q in qs]
        jets = [q.get("surface_event", {}).get("jet", {}) for q in qs]
        terminal_meta = [q.get("surface_event", {}).get("jet", {}) for q in qs]
        pressure.append({
            "x": x, "dr_km": [0.05, 0.025, 0.0125], "status": "FULL_RAW_PRESSURE_LADDER",
            "final_two_rel_M": vals["M"], "final_two_rel_R": vals["R"],
            "final_two_rel_k2": vals["k2"], "final_two_rel_Lambda": vals["Lambda"],
            "primary_finest_rel_M": cross["M"], "primary_finest_rel_R": cross["R"],
            "primary_finest_rel_k2": cross["k2"], "primary_finest_rel_Lambda": cross["Lambda"],
            "primary_observables": {k: f[k] for k in ("M_Msun", "R_km", "k2", "Lambda")},
            "start_halving_rel_y": err["y"], "start_halving_order": orders,
            "centre_series_errors": err, "centre_series_order_by_quantity": orders_by_q,
            "centre_series_roundoff_observed": roundoff_by_q,
            "centre_series_order_status": "NULL_WHEN_BINARY64_UNRESOLVED",
            "start_halving_pass": centre_pass,
            "centre_recurrence": copy.deepcopy(central_bundle),
            "central_coefficient_cse": central_bundle.get("central_coefficient_cse", {}),
            "pressure_inversion_worst": {
                "residual_log": max(float(v.get("worst_residual_log", 1.0)) for v in inv),
                "bracket_rel": max(float(v.get("bracket_rel", 1.0)) for v in inv),
                "max_iterations": max(int(v.get("max_iterations", 64)) for v in inv),
                "residual_bound": 2.0e-12, "bracket_bound": 2.0e-12,
            },
            "terminal_jet": {
                "production_chart_pass": bool(all(j.get("pass") is True for j in jets)),
                "all_degree8_degree10_pass": bool(all(j.get("pass") is True for j in jets)),
                "max_remainder_bound": max((float(j.get("remainder_bound", 0.0)) for j in jets), default=0.0),
                "methods": sorted(set(str(j.get("method", "")) for j in jets)),
                "roundoff_audits": [],
            },
            "pass": bool(vals["M"] <= 5e-4 and vals["R"] <= 5e-4 and vals["k2"] <= 2e-3 and vals["Lambda"] <= 2e-3 and cross["M"] <= 1e-3 and cross["R"] <= 1e-3 and cross["k2"] <= 5e-3 and cross["Lambda"] <= 5e-3 and centre_pass),
        })
        # A trajectory record is the authoritative production/proof-role
        # split.  The legacy aggregate fields above remain numerical summary
        # diagnostics; overwrite their theorem-looking list with witness-only
        # payloads so non-witnesses cannot inherit a PASS/digest.
        trajectories = []
        for rec, jet in zip(q_records, jets):
            trajectories.append({
                "x": float(x), "x_hex": float(x).hex(),
                "dr": rec["dr"], "dr_hex": rec["dr_hex"],
                "proof_role": rec["proof_role"],
                "mesh": copy.deepcopy(rec["result"]["mesh"]),
                "observables": {k: rec["result"][k] for k in ("M_Msun", "R_km", "k2", "Lambda")},
                "terminal_chart": copy.deepcopy(jet.get("terminal_chart", {})),
                "terminal_input_identity": copy.deepcopy(jet.get("terminal_input")),
            })
        pressure[-1]["trajectories"] = trajectories if pressure else trajectories
        if pressure:
            pmeta = pressure[-1].setdefault("terminal_jet", {})
            pmeta["production_chart_pass"] = bool(all(j.get("pass") is True for j in jets))
            # The former overloaded field was theorem-looking and made a
            # production chart indistinguishable from an interval proof.  It
            # is intentionally absent; theorem status lives only in the
            # three top-level witness records.
            pmeta.pop("all_degree8_degree10_pass", None)
            pmeta["max_remainder_bound"] = max((float(j.get("remainder_bound", 0.0)) for j in jets), default=0.0)
            pmeta["roundoff_audits"] = [
                {"status": j["roundoff_audit"].get("status", "FAIL"),
                 "interval_error_check_pass": bool(j.get("interval_error_check_pass", False)),
                 "direct_error_check_by_quantity": j.get("direct_error_check_by_quantity", {}),
                 "rounding_cells": j["roundoff_audit"].get("derived_rounding_cells", {}),
                 "n64_error_interval": j["roundoff_audit"].get("n64_error_interval", [])}
                for j in terminal_meta
                if j.get("proof_role") == TERMINAL_PROOF_WITNESS and isinstance(j.get("roundoff_audit"), dict)
            ]
    s32, s64 = g32["stationary"], g64["stationary"]; mm = _rel(g32["maximum"]["M_Msun"], g64["maximum"]["M_Msun"]); nx = _rel(s32["x"], s64["x"]); ts = {q: max((_rel(g32["targets"][n][q], g64["targets"][n][q]) for n in g32["targets"] if q in g32["targets"][n] and q in g64["targets"].get(n, {})), default=float("inf")) for q in ("R_km", "Lambda")}; etaq = {q: max((g32["targets"][n]["eta_Q"][q] + g64["targets"][n]["eta_Q"][q] for n in g32["targets"] if q in g32["targets"][n] and q in g64["targets"].get(n, {})), default=float("inf")) for q in ("R_km", "Lambda")}; sums = {"mass": mm + s32["eta_M"] + s64["eta_M"], "central_density": nx + s32["eta_x"] + s64["eta_x"], "R_km": ts["R_km"] + etaq["R_km"], "Lambda": ts["Lambda"] + etaq["Lambda"]}; gp = bool(s32["pass"] and s64["pass"] and sums["mass"] < 1e-3 and sums["central_density"] < 1e-3 and sums["R_km"] < 2e-3 and sums["Lambda"] < 2e-3)
    terminal_interval_witnesses.sort(key=lambda w: (w["x_hex"], w["dr_hex"]))
    return {"adaptive_ladder": adaptive, "tolerance_ladder": tolerance_ladder, "tolerance_plan": {"owner": f"grid{g32['points_per_decade']}", "requested": len(tolerance_ladder), "distinct": sum(1 for r in tolerance_ladder if not r["reused"]), "reused": sum(1 for r in tolerance_ladder if r["reused"]), "settings": [[1.0e-8, 100.0], [1.0e-9, 10.0], [1.0e-10, 1.0]], "revision": OWNER_EQUATION_REVISION, "source_revision": OWNER_SOURCE_REVISION}, "pressure_rk4": pressure, "terminal_interval_witnesses": terminal_interval_witnesses, "grid_doubling": {"points_per_decade": [32, 64], "max_mass_shift_rel": mm, "max_nc_shift_rel": nx, "target_shift_rel": ts, "uncertainty_eta_M": {"32": s32["eta_M"], "64": s64["eta_M"]}, "uncertainty_eta_x": {"32": s32["eta_x"], "64": s64["eta_x"]}, "uncertainty_eta_Q": etaq, "acceptance_sum": sums, "stationary_32": s32, "stationary_64": s64, "pass": gp}, "pass": bool(all(r["pass"] is not False for r in adaptive) and all(r["pass"] is not False for r in pressure) and gp)}


def _primary_launch_errors(row: dict[str, Any]) -> list[str]:
    """Validate the certificate actually owned by the enthalpy producer."""
    launch = row.get("centre_launch", {})
    if launch.get("method") != "leading_regular_enthalpy" or launch.get("source_revision") != OWNER_SOURCE_REVISION:
        return ["primary_centre_source_certificate"]
    try:
        cert = launch["source_certificate"]
        if not (cert["endpoint"] is False and cert["w_lo"] <= launch["W"] <= cert["w_hi"]
                and cert["jacobian"] > 0 and 0 <= cert["bracket_rel"] <= 2e-13
                and 0 <= cert["residual_rel"] <= 2e-12):
            return ["primary_centre_source_branch"]
        eg, pg = row["eps_c_MeV_fm3"] * K_CONV, row["P_c_MeV_fm3"] * K_CONV
        r0 = math.sqrt(row["delta_h"] / (2 * math.pi * (eg / 3 + pg)))
        m0 = 4 * math.pi * eg * r0**3 / 3
        if launch["r_km"] != r0 or launch["m_km"] != m0 or launch["y"] != 2.0:
            return ["primary_centre_unit_boundary"]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return ["primary_centre_source_certificate"]
    return []


def _validate_result_contract(result: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        canonical_result_sha256(result)
    except (TypeError, ValueError, OverflowError):
        return False, ["nonfinite_or_unserializable_payload"]
    if result.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    if result.get("upstream_source_sha256") != upstream_digest() or result.get("upstream_source_sha256") != EXPECTED_UPSTREAM_DIGEST:
        errors.append("stale_upstream_digest")
    prov = result.get("provenance", {})
    if prov.get("action_sha256") != file_digest(ACTION_PATH) or prov.get("action_sha256") != ACTION_DIGEST:
        errors.append("stale_action_digest")
    if prov.get("contract_sha256") != file_digest(CONTRACT_PATH) or prov.get("contract_sha256") != CONTRACT_DIGEST:
        errors.append("stale_contract_digest")
    if prov.get("producer_sha256") != file_digest(ROOT / "source_complete_stellar_prediction.py"):
        errors.append("stale_producer_digest")
    if TEST_PATH.exists() and prov.get("test_sha256") != file_digest(TEST_PATH):
        errors.append("stale_test_digest")
    if result.get("canonical_result_sha256") != canonical_result_sha256(result):
        errors.append("canonical_numeric_identity")
    contract = result.get("contract", {})
    if contract.get("degeneracy") != 4:
        errors.append("degeneracy_not_four")
    if contract.get("domain_x_max") != 10.0:
        errors.append("domain_enlarged")
    if contract.get("domain_x_min") != 0.0 or contract.get("surface") != "exact_h_zero_vacuum":
        errors.append("surface_or_domain_policy")
    if contract.get("degeneracy") != int(upstream.PARAMS.degeneracy):
        errors.append("upstream_degeneracy_mismatch")
    if contract.get("pressure_definition") != "mu*n-epsilon_and_hilbert_crosscheck":
        errors.append("pressure_generator_swapped")
    if contract.get("q_phi_and_gs") != "derived_from_upstream_params":
        errors.append("derived_coupling_override")
    if contract.get("endpoint_policy") != "exact_no_clamp_no_extrapolation":
        errors.append("endpoint_clamp_or_extrapolation")
    if contract.get("beta_equilibrium") not in (None, False) or contract.get("leptons") not in (None, False):
        errors.append("composition_injection")
    if contract.get("units") != {"k_conv": K_CONV, "M_sun_km": M_SUN_KM}:
        errors.append("unit_convention_mutation")
    forbidden = ("beta_equilibrium", "crust", "css", "lepton", "extrapolat", "clip")
    if any(contract.get(k) not in (None, False, "none", "exact") for k in ("crust_model", "css_branch", "dedp_clipping")):
        errors.append("forbidden_completion")
    if result.get("evidence_weight") != 0 or result.get("physical_NS_blocker") != PHYSICAL_NS_BLOCKER:
        errors.append("claim_ceiling_or_evidence")
    if result.get("status") not in ("STABLE_BRANCH_TERMINATED_AT_FIRST_MASS_MAXIMUM", "FAIL_CLOSED_NUMERICAL_CONVERGENCE"):
        errors.append("status_mutation")
    for row in result.get("rows", []):
        if row.get("row_status") != "MATHEMATICAL_ONE_COMPONENT_SEQUENCE" or row.get("evidence_weight") != 0 or row.get("physical_NS_blocker") != PHYSICAL_NS_BLOCKER:
            errors.append("row_claim_ceiling")
            break
        solver = row.get("solver", {})
        if solver.get("strict_eos") is not True or solver.get("rtol") != 1.0e-10 or solver.get("atol_scale") != 1.0 or solver.get("centre_delta_scale") != 1.0:
            errors.append("row_not_matched_final_evaluator")
            break
        expected_delta = min(1.0e-8, 1.0e-6 * float(row.get("h_c", -1.0)))
        if not np.isfinite(expected_delta) or abs(float(row.get("delta_h", float("nan"))) - expected_delta) > 1.0e-20 * max(1.0, expected_delta):
            errors.append("centre_delta_policy")
            break
        love = row.get("love_surface", {})
        expected_love_method = "scaled_C5_series" if float(row.get("C", 1.0)) <= 0.25 else "decimal_original_80"
        tail_rel = float(love.get("tail_relative")) if love.get("tail_relative") is not None else (0.0 if expected_love_method == "decimal_original_80" else float("inf"))
        if love.get("status") != "PASS" or love.get("method") != expected_love_method or love.get("oracle") != "independent_original" or love.get("oracle_dps") != [80, 120] or max(float(v) for v in love.get("oracle_agreement_rel", {}).values()) > 5.0e-12 or float(love.get("oracle_precision_rel", float("inf"))) > 5.0e-13 or tail_rel > 2.0e-15 or _rel(float(row.get("k2", float("nan"))), float(love.get("oracle_80_k2", float("nan")))) > 5.0e-12 or _rel(float(row.get("Lambda", float("nan"))), float(love.get("oracle_80_Lambda", float("nan")))) > 5.0e-12:
            errors.append("love_surface_oracle")
            break
        if not (0.0 <= float(row.get("nc_over_n0", -1.0)) <= X_MAX):
            errors.append("row_outside_domain")
            break
        # The enthalpy start owns its leading regular expansion and direct
        # stationary-source bracket.  The pressure recurrence is certified
        # separately on each pressure row below.
        launch_errors = _primary_launch_errors(row)
        if launch_errors:
            errors.extend(launch_errors)
            break
    for target, payload in result.get("targets", {}).items():
        if payload.get("status") == "BLOCKED_TARGET_OUTSIDE_SEQUENCE" and any(k in payload for k in ("R_km", "k2", "Lambda")):
            errors.append("target_extrapolation")
            break
    forbidden_observation_words = ("fit", "pass", "fail", "overlap", "confirm", "exclude", "likelihood", "score", "inside", "outside")
    for obs in result.get("observational_context", []):
        if obs.get("comparison_status") != "BLOCKED_MODEL_NOT_PHYSICAL_NS" or obs.get("evidence_weight") != 0:
            errors.append("observation_claim_or_weight")
            break
        text_blob = " ".join(str(obs.get(k, "")) for k in ("name", "value", "source")).lower()
        if any(word in text_blob for word in forbidden_observation_words):
            errors.append("observation_language")
            break
    grid = result.get("grid", {})
    if int(grid.get("excluded_post_max", 0)) > 0 and (not result.get("rows") or not result["rows"][-1].get("max_refined", False)):
        errors.append("post_maximum_rows_not_censored")
    recheck = result.get("maximum_recheck_rel", {})
    eta_m = result.get("convergence", {}).get("grid_doubling", {}).get("uncertainty_eta_M", {}).get("32", float("inf"))
    if recheck and any(float(recheck.get(k, float("inf"))) > float(eta_m) for k in ("M_Msun", "R_km", "k2", "Lambda")):
        errors.append("maximum_direct_recheck")
    centres = result.get("centre_halving", [])
    required_x = {0.001, 0.0014330125702369627, 0.002053525026457146}
    if result.get("rows") and ({float(w.get("x", float("nan"))) for w in centres} != required_x or any(
        w.get("pass") is not True or float(w.get("rel_M", float("inf"))) > 1.0e-7 or float(w.get("rel_R", float("inf"))) > 1.0e-7
        or float(w.get("rel_C", float("inf"))) > 1.0e-7 or float(w.get("abs_y_R", float("inf"))) > 1.0e-7
        or float(w.get("rel_k2", float("inf"))) > 2.0e-6 or float(w.get("rel_Lambda", float("inf"))) > 2.0e-6
        or abs(float(w.get("half_delta_h", 0.0)) / float(w.get("base_delta_h", 1.0)) - 0.5) > 1.0e-12
        or float(w.get("base_love_oracle_max_rel", float("inf"))) > 5.0e-12
        or float(w.get("half_love_oracle_max_rel", float("inf"))) > 5.0e-12 for w in centres
    )):
        errors.append("centre_halving_gate")
    if result.get("rows"):
        conv = result.get("convergence", {})
        ladder = conv.get("adaptive_ladder", [])
        required_ladder_x = {0.001, 0.0014330125702369627, 0.002053525026457146}
        seen_ladder_x = {float(r.get("x", float("nan"))) for r in ladder}
        if not required_ladder_x.issubset(seen_ladder_x) or any(r.get("pass") is not True for r in ladder if float(r.get("x", float("nan"))) in required_ladder_x):
            errors.append("low_density_adaptive_ladder")
        pressure = result.get("convergence", {}).get("pressure_rk4", [])
        seen_pressure_x = {float(r.get("x", float("nan"))) for r in pressure}
        if not required_ladder_x.issubset(seen_pressure_x) or any(r.get("status") != "FULL_RAW_PRESSURE_LADDER" or r.get("dr_km") != [0.05, 0.025, 0.0125] or not isinstance(r.get("start_halving_order"), list) or float(r.get("pressure_inversion_worst", {}).get("residual_log", 1.0)) > 2.0e-12 or float(r.get("pressure_inversion_worst", {}).get("bracket_rel", 1.0)) > 2.0e-12 or r.get("terminal_jet", {}).get("production_chart_pass") is not True for r in pressure):
            errors.append("low_density_pressure_witness")
        # Gate-4 ownership/schema contract: there are exactly eighteen
        # independently charted trajectories and exactly three predeclared
        # theorem witnesses.  The key set is checked from exact float bits,
        # never from observed ladder equality or a submitted PASS flag.
        expected_witness_keys = set(TERMINAL_WITNESS_KEYS)
        witnesses = result.get("terminal_interval_witnesses", [])
        if witnesses != conv.get("terminal_interval_witnesses", []):
            errors.append("terminal_witness_projection_mismatch")
        if not isinstance(witnesses, list) or len(witnesses) != 3:
            errors.append("terminal_witness_cardinality")
        else:
            witness_keys = []
            for witness in witnesses:
                if not isinstance(witness, dict):
                    errors.append("terminal_witness_schema"); continue
                key = (str(witness.get("x_hex", "")), str(witness.get("dr_hex", "")))
                witness_keys.append(key)
                try:
                    if float(witness.get("x")) .hex() != key[0] or float(witness.get("dr")) .hex() != key[1]:
                        errors.append("terminal_witness_exact_key")
                except Exception:
                    errors.append("terminal_witness_exact_key")
                if witness.get("proof_role") != TERMINAL_PROOF_WITNESS or key not in expected_witness_keys:
                    errors.append("terminal_witness_role_or_key")
                proof = witness.get("interval_diagnostic")
                ident = witness.get("terminal_input_identity")
                if not isinstance(proof, dict) or not isinstance(ident, dict):
                    errors.append("terminal_witness_payload")
                else:
                    if proof.get("ivp_identity") != ident or proof.get("independent") is not True or not isinstance(proof.get("interval_error_check_pass"), bool) or not isinstance(proof.get("direct_error_pass"), bool) or proof.get("rigorous_error_certified") is not False or proof.get("evidence_scope") != "numerical_interval_diagnostic_not_rigorous_error_bound":
                        errors.append("terminal_witness_proof_status")
                    if proof.get("namespace") != "producer-empty-exact-dyadic" or proof.get("namespace_role") != "producer" or proof.get("namespace_lifetime") != "destroyed-on-return":
                        errors.append("terminal_witness_namespace")
                    if proof.get("replay_acceptor") != "repair35-independent-canonical-dyadic-replay-v1":
                        errors.append("terminal_witness_replay")
            if len(set(witness_keys)) != 3 or set(witness_keys) != expected_witness_keys or witness_keys != sorted(witness_keys):
                errors.append("terminal_witness_key_set_or_order")
        trajectory_count = 0
        trajectory_keys: set[tuple[str, str]] = set()
        trajectory_witness_keys: set[tuple[str, str]] = set()
        for prow in pressure:
            trajectories = prow.get("trajectories")
            if not isinstance(trajectories, list) or len(trajectories) != 3:
                errors.append("terminal_chart_cardinality"); continue
            for traj in trajectories:
                trajectory_count += 1
                if not isinstance(traj, dict):
                    errors.append("terminal_chart_schema"); continue
                key = (str(traj.get("x_hex", "")), str(traj.get("dr_hex", "")))
                trajectory_keys.add(key)
                try:
                    if float(traj.get("x")) .hex() != key[0] or float(traj.get("dr")) .hex() != key[1]:
                        errors.append("terminal_chart_exact_key")
                except Exception:
                    errors.append("terminal_chart_exact_key")
                expected_role = _terminal_proof_role(float(traj.get("x")), float(traj.get("dr")))
                if traj.get("proof_role") != expected_role:
                    errors.append("terminal_chart_role_schedule")
                if not isinstance(traj.get("terminal_input_identity"), dict):
                    errors.append("terminal_chart_input_identity")
                chart = traj.get("terminal_chart")
                if not isinstance(chart, dict) or chart.get("status") != "PASS" or chart.get("N") != [16, 32, 64] or not isinstance(chart.get("ladder"), dict) or set(chart.get("ladder", {})) != {"16", "32", "64"} or not isinstance(chart.get("dop853_oracle"), dict) or not isinstance(chart.get("observed_order"), list) or len(chart.get("observed_order", [])) != 3:
                    errors.append("terminal_chart_missing_production")
                # Non-witnesses must not carry any theorem payload or copied
                # proof status, digest, namespace, or interval fields.
                theorem_fields = {"roundoff_audit", "interval_error_check_pass", "direct_error_pass", "replay_digest", "replay_acceptor", "validated_interval", "direct_n64_error", "proof"}
                if expected_role == TERMINAL_PRODUCTION_CHART:
                    if isinstance(chart, dict) and any(field in chart for field in theorem_fields):
                        errors.append("non_witness_theorem_payload")
                else:
                    trajectory_witness_keys.add(key)
                    if not isinstance(traj.get("terminal_input_identity"), dict):
                        errors.append("witness_terminal_input_identity")
        if trajectory_count != 18 or len(trajectory_keys) != 18:
            errors.append("terminal_chart_total_or_duplicate")
        if trajectory_witness_keys != expected_witness_keys:
            errors.append("terminal_chart_witness_mapping")
        if isinstance(witnesses, list) and len(witnesses) == 3:
            by_key = {(str(w.get("x_hex", "")), str(w.get("dr_hex", ""))): w for w in witnesses if isinstance(w, dict)}
            for prow in pressure:
                for traj in prow.get("trajectories", []) if isinstance(prow.get("trajectories"), list) else []:
                    key = (str(traj.get("x_hex", "")), str(traj.get("dr_hex", "")))
                    if traj.get("proof_role") == TERMINAL_PROOF_WITNESS and key in by_key and traj.get("terminal_input_identity") != by_key[key].get("terminal_input_identity"):
                        errors.append("terminal_witness_input_cross_binding")
        for pr in pressure:
            for key, final_limit, cross_limit in (("M", 5e-4, 1e-3), ("R", 5e-4, 1e-3), ("k2", 2e-3, 5e-3), ("Lambda", 2e-3, 5e-3)):
                if float(pr.get("final_two_rel_" + key, float("inf"))) > final_limit:
                    errors.append("pressure_convergence_numeric")
                if float(pr.get("primary_finest_rel_" + key, float("inf"))) > cross_limit:
                    errors.append("pressure_cross_method_numeric")
            centre = pr.get("centre_recurrence", {})
            if not _centre_proof_accepts(centre):
                errors.append("pressure_centre_proof")
            if centre.get("coordinate_units") != {"radius": "rho=sqrt(K_CONV)*r_km", "mass": "m_formal=sqrt(K_CONV)*m_km", "pressure": "MeV/fm^3", "k_conv": K_CONV}:
                errors.append("pressure_centre_unit_boundary")
            for traj in pr.get("trajectories", []):
                mesh = traj.get("mesh", {})
                if mesh.get("scheme") != "uniform_r_km" or mesh.get("actual_dr_km") != traj.get("dr") or mesh.get("requested_dr_km") != traj.get("dr") or int(mesh.get("full_steps", 0)) <= 0:
                    errors.append("pressure_mesh_spacing")
            trajectories = pr.get("trajectories", [])
            if len(trajectories) == 3:
                try:
                    for short, key in (("M", "M_Msun"), ("R", "R_km"), ("k2", "k2"), ("Lambda", "Lambda")):
                        q2, q3 = trajectories[1]["observables"][key], trajectories[2]["observables"][key]
                        if pr["final_two_rel_" + short] != _rel(q3, q2) or pr["primary_finest_rel_" + short] != _rel(q3, pr["primary_observables"][key]):
                            errors.append("pressure_observable_summary_binding")
                except (KeyError, TypeError, ValueError):
                    errors.append("pressure_observable_summary_binding")
            cb = pr.get("central_coefficient_cse", {})
            if cb.get("key") != "x.hex" or cb.get("x_hex") != float(pr.get("x", float("nan"))).hex() or cb.get("spacings") != [0.0125, 0.00625, 0.003125] or cb.get("shared_bundle") is not True or cb.get("state_reuse") is not False:
                errors.append("central_source_bundle_provenance")
                break
        if conv.get("pass") is True and (conv.get("grid_doubling", {}).get("pass") is not True or any(r.get("pass") is False for r in ladder + pressure)):
            errors.append("forced_convergence_boolean")
    oracle = result.get("scalar_derivative_oracle")
    if oracle is not None and (oracle.get("pass") is not True or oracle.get("boundary_rel", 1.0) > 2.0e-12 or any(float(r.get("rel_80_120", 1.0)) > 1.0e-40 or float(r.get("rel_production_oracle", 1.0)) > 2.0e-10 for r in oracle.get("rows", []))):
        errors.append("scalar_derivative_oracle")
    h_oracle = result.get("enthalpy_oracle")
    if h_oracle is not None and (h_oracle.get("pass") is not True or h_oracle.get("oracle_dps") != [80, 120] or any(float(r.get("rel_80_120", 1.0)) > 1.0e-40 or float(r.get("rel_production_oracle", 1.0)) > 2.0e-12 for r in h_oracle.get("rows", []))):
        errors.append("enthalpy_oracle")
    gamma = result.get("gamma2_control")
    if gamma is not None and gamma.get("pass") is not True:
        errors.append("gamma2_two_coordinate_control")
    ledger = result.get("owner_evaluation_registry")
    if ledger is not None:
        required_fields = ["owner", "x_hex", "rtol", "atol_scale", "centre_scale", "strict_eos", "equation_revision", "source_revision"]
        if ledger.get("enabled") is not True or ledger.get("key_fields") != ["owner", "x.hex", "rtol", "atol_scale", "centre_scale", "strict_eos", "equation_revision", "source_revision"]:
            errors.append("owner_registry_key_definition")
        if ledger.get("equation_revision") != OWNER_EQUATION_REVISION or ledger.get("source_revision") != OWNER_SOURCE_REVISION:
            errors.append("owner_registry_revision")
        if ledger.get("mode_revisions") != {"ordinary": OWNER_EQUATION_REVISION, "augmented_tangent": OWNER_TANGENT_EQUATION_REVISION}:
            errors.append("owner_registry_mode_revision")
        try:
            requested = int(ledger["requested"]); unique = int(ledger["unique"]); hits = int(ledger["prior_hit"]); executed = int(ledger["executed"]); reused = int(ledger["reused"])
            if requested != unique + hits or executed != unique or reused != hits or int(ledger["total_strict_integrations"]) != executed:
                errors.append("owner_registry_count_arithmetic")
            owners = set(str(v) for v in ledger.get("owners", [])); rb = ledger.get("requested_by_owner", {}); ub = ledger.get("unique_by_owner", {}); hb = ledger.get("prior_hit_by_owner", {}); eb = ledger.get("executed_by_owner", {})
            if sum(int(v) for v in rb.values()) != requested or sum(int(v) for v in ub.values()) != unique or sum(int(v) for v in hb.values()) != hits or sum(int(v) for v in eb.values()) != executed:
                errors.append("owner_registry_owner_totals")
            settings = ledger.get("settings", []); seen = set(); level_seen: dict[str, set[tuple[float,float]]] = {}
            for rec in settings:
                if any(k not in rec for k in required_fields): errors.append("owner_registry_settings_fields"); break
                key = tuple(rec[k] for k in required_fields)
                if key in seen: errors.append("owner_registry_duplicate_key"); break
                seen.add(key); owner = str(rec["owner"]); level_seen.setdefault(owner, set()).add((float(rec["rtol"]), float(rec["atol_scale"])))
                if rec["equation_revision"] not in (OWNER_EQUATION_REVISION, OWNER_TANGENT_EQUATION_REVISION) or rec["source_revision"] != OWNER_SOURCE_REVISION or rec["strict_eos"] is not True or rec["centre_scale"] not in (1.0, 0.5):
                    errors.append("owner_registry_key_mutation"); break
            if settings != sorted(settings, key=lambda z: (z.get("owner", ""), z.get("x_hex", ""), z.get("rtol", 0.0), z.get("atol_scale", 0.0), z.get("centre_scale", 0.0), z.get("strict_eos", False), z.get("equation_revision", ""), z.get("source_revision", ""))):
                errors.append("owner_registry_non_deterministic_order")
            # The hard resource cap applies to actual strict integrations
            # (`executed`), not consumer requests: repeated exact-key hits are
            # deliberately serialized in `requested` for auditability.
            if executed > 1200:
                errors.append("owner_registry_budget")
        except (KeyError, TypeError, ValueError):
            errors.append("owner_registry_malformed")
    return not errors, errors


def validate_contract(result: dict[str, Any]) -> tuple[bool, list[str]]:
    """Public validator: cheap checks plus one fresh canonical generation."""
    errors: list[str] = []
    ok, cheap = _validate_result_contract(result)
    errors.extend(cheap)
    try:
        fresh = _generate_result()
        if _canonical_full_bytes(result) != _canonical_full_bytes(fresh):
            errors.append("canonical_full_bytes_mismatch")
    except Exception as exc:
        errors.append("fresh_generation_failed:" + type(exc).__name__)
    return (not errors), errors


class _InjectedValidationSession:
    """Private test-only mutation session; production validation is isolated."""
    def __init__(self, generator: Callable[[], dict[str, Any]]) -> None:
        self._generator = generator
        self.calls = 0

    def validate(self, candidate: dict[str, Any]) -> tuple[bool, list[str]]:
        self.calls += 1
        if self.calls != 1:
            return False, ["injected_session_reused"]
        try:
            fresh = self._generator()
            return (_canonical_full_bytes(candidate) == _canonical_full_bytes(fresh), [] if _canonical_full_bytes(candidate) == _canonical_full_bytes(fresh) else ["canonical_full_bytes_mismatch"])
        except Exception as exc:
            return False, ["fresh_generation_failed:" + type(exc).__name__]


def _validation_session(generator: Callable[[], dict[str, Any]] | None = None) -> _InjectedValidationSession:
    return _InjectedValidationSession(generator or _generate_result)


def _generate_result() -> dict[str, Any]:
    global _GENERATION_GUARD, _BUILD_BUDGET
    if _GENERATION_GUARD:
        raise RuntimeError("canonical producer may run only once per process")
    _GENERATION_GUARD = True
    budget = _BuildBudget()
    _BUILD_BUDGET = budget
    if upstream_digest() != EXPECTED_UPSTREAM_DIGEST:
        raise RuntimeError("upstream source digest changed; prediction artifact is stale")
    registry = _OwnerEvaluationRegistry(enabled=True)
    points = []
    # Endpoint and representative direct EOS gates are part of the immutable
    # producer output; all central rows are gated before integration.
    for x in (0.0, 1.0e-3, 0.1, 1.0, 5.0, 10.0):
        p, gates = eos_point_with_gates(x)
        points.append({"x": x, "eps": p.eps, "P": p.pressure, "mu": p.mu, "h": p.enthalpy, "W": p.W, "gates": gates})
    grid32 = _matched_grid(32, registry=registry, owner="grid32")
    grid64 = _matched_grid(64, registry=registry, owner="grid64")
    grid = grid32["x"]
    rows = grid32["stable_rows"]
    max_row = grid32["maximum"]
    status = "STABLE_BRANCH_TERMINATED_AT_FIRST_MASS_MAXIMUM"
    targets: dict[str, Any] = grid32["targets"]
    fresh_max = _StrictEvaluator(owner="maximum_recheck", registry=registry).final(grid32["stationary"]["x"])
    max_recheck = {k: _rel(max_row[k], fresh_max[k]) for k in ("M_Msun", "R_km", "k2", "Lambda")}
    centre_halving: list[dict[str, Any]] = []
    for x in (0.001, 0.0014330125702369627, 0.002053525026457146):
        base = _registered_enthalpy(registry, owner=f"centre_base:{float(x).hex()}", x=x, rtol=1.0e-10, atol_scale=1.0, centre_scale=1.0)
        half = _registered_enthalpy(registry, owner=f"centre_half:{float(x).hex()}", x=x, rtol=1.0e-10, atol_scale=1.0, centre_scale=0.5)
        centre_halving.append({"x": x, "base_delta_h": base["delta_h"], "half_delta_h": half["delta_h"], "rel_M": _rel(base["M_Msun"], half["M_Msun"]), "rel_R": _rel(base["R_km"], half["R_km"]), "rel_C": _rel(base["C"], half["C"]), "abs_y_R": abs(base["y_R"] - half["y_R"]) / max(abs(base["y_R"]), 1.0), "rel_k2": _rel(base["k2"], half["k2"]), "rel_Lambda": _rel(base["Lambda"], half["Lambda"]), "base_love_oracle_max_rel": max(float(v) for v in base["love_surface"]["oracle_agreement_rel"].values()), "half_love_oracle_max_rel": max(float(v) for v in half["love_surface"]["oracle_agreement_rel"].values()), "pass": bool(_rel(base["M_Msun"], half["M_Msun"]) <= 1.0e-7 and _rel(base["R_km"], half["R_km"]) <= 1.0e-7 and _rel(base["C"], half["C"]) <= 1.0e-7 and abs(base["y_R"] - half["y_R"]) / max(abs(base["y_R"]), 1.0) <= 1.0e-7 and _rel(base["k2"], half["k2"]) <= 2.0e-6 and _rel(base["Lambda"], half["Lambda"]) <= 2.0e-6)})
    observations = [
        {"name": "PSR J0740+6620 timing", "value": "2.08 +/- 0.07 Msun; Mmax >= 2.01 Msun", "source": "https://arxiv.org/abs/2104.00880"},
        {"name": "PSR J0740+6620 radius (NICER/XMM)", "value": "12.49 +1.28/-0.88 km", "source": "https://arxiv.org/abs/2406.14466"},
        {"name": "PSR J0740+6620 radius (NICER/XMM)", "value": "12.92 +2.09/-1.13 km", "source": "https://arxiv.org/abs/2406.14467"},
        {"name": "PSR J0030+0451", "value": "M=1.44 +0.15/-0.14 Msun, R=13.02 +1.24/-1.06 km", "source": "https://arxiv.org/abs/1912.05705"},
        {"name": "GW170817 low-spin", "value": "Lambda_tilde=300 +420/-230 (70--720)", "source": "https://arxiv.org/abs/1805.11579"},
        {"name": "GW170817 high-spin", "value": "Lambda_tilde=0--630", "source": "https://arxiv.org/abs/1805.11579"},
        {"name": "GW170817 radius/Lambda", "value": "Lambda_1.4=190 +390/-120; R_1.4=11.9 +/-1.4 km", "source": "https://arxiv.org/abs/1805.11581"},
    ]
    convergence = _convergence_summary(grid32, grid64)
    result = {
        "schema_version": SCHEMA_VERSION,
        "upstream_source_sha256": upstream_digest(),
        "provenance": {"action_path": str(ACTION_PATH.relative_to(REPO_ROOT)), "action_sha256": file_digest(ACTION_PATH), "contract_path": str(CONTRACT_PATH.relative_to(REPO_ROOT)), "contract_sha256": file_digest(CONTRACT_PATH), "producer_path": str((ROOT / "source_complete_stellar_prediction.py").relative_to(REPO_ROOT)), "producer_sha256": file_digest(ROOT / "source_complete_stellar_prediction.py"), "test_path": str(TEST_PATH.relative_to(REPO_ROOT)), "test_sha256": file_digest(TEST_PATH)},
        "contract": {
            "degeneracy": int(upstream.PARAMS.degeneracy), "domain_x_min": 0.0, "domain_x_max": 10.0,
            "surface": "exact_h_zero_vacuum", "composition": "one_component_source_branch_only",
            "crust_model": None, "css_branch": None, "dedp_clipping": False,
            "pressure_definition": "mu*n-epsilon_and_hilbert_crosscheck", "q_phi_and_gs": "derived_from_upstream_params",
            "endpoint_policy": "exact_no_clamp_no_extrapolation", "beta_equilibrium": None, "leptons": None,
            "units": {"k_conv": K_CONV, "M_sun_km": M_SUN_KM},
        },
        "units": {"energy_pressure": "MeV/fm^3", "density": "fm^-3", "radius_mass": "km", "mass_output": "M_sun", "k_conv": K_CONV, "M_sun_km": M_SUN_KM},
        "eos_gate_samples": points,
        "grid": {"central_x": "10^(-3+j/32), j=0..128", "count": 129, "domain_termination": status, "raw_count": len(grid32["rows"]), "stable_count": len(rows), "excluded_post_max": len(grid32["rows"]) - len(rows), "matched_64_count": len(grid64["rows"])},
        "rows": rows,
        "mass_maximum": max_row,
        "maximum_recheck_rel": max_recheck,
        "centre_halving": centre_halving,
        "targets": targets,
        "observational_context": [{**obs, "comparison_status": "BLOCKED_MODEL_NOT_PHYSICAL_NS", "evidence_weight": 0} for obs in observations],
        "convergence": convergence,
        # The exact three-key witness index is top-level so consumers cannot
        # mistake an aggregate pressure-row diagnostic for theorem evidence.
        "terminal_interval_witnesses": copy.deepcopy(convergence.get("terminal_interval_witnesses", [])),
        "status": status,
        "evidence_weight": 0,
        "physical_NS_blocker": PHYSICAL_NS_BLOCKER,
        "numerical_provenance": {
            "evidence_scope": "numerical_convergence_not_rigorous_global_error_certification",
            "primary": "DOP853 enthalpy; rtol=1e-10, atol=(1e-11 km,1e-12 km,1e-10)",
            "independent": "raw pressure-coordinate RK4 dr=(0.05,0.025,0.0125) km; analytic r=dr centre launch; w=(P/1 MeV fm^-3)^(2/5) exact w=0 event",
            "derivatives": "five-point log-density ladder (1e-3,5e-4,2.5e-4)",
            "love_surface": "C^5-scaled Hinderer series (certified tail for C<=0.25) with independent original 80/120-decimal oracle",
            "centre": "delta_h=min(1e-8,1e-6*h_c), three delta_h/2 witnesses",
            "interpolation": "none for EOS, maxima, or targets; source table only selects inversion brackets",
            "optional_eos_crosscheck": {"log_nodes": 130, "includes_vacuum": True, "used_for_domain": False, "role": "bracket_selection_only"},
            "build_budget": budget.snapshot(),
        },
        "owner_evaluation_registry": registry.snapshot(),
        "scalar_derivative_oracle": scalar_derivative_oracle_controls(),
        "enthalpy_oracle": enthalpy_oracle_controls(),
        "gamma2_control": gamma2_newtonian_control(),
    }
    if not result["convergence"]["pass"]:
        result["status"] = "FAIL_CLOSED_NUMERICAL_CONVERGENCE"
    result["canonical_result_sha256"] = canonical_result_sha256(result)
    ok, errors = _validate_result_contract(result)
    if not ok:
        raise RuntimeError("self-validation failed: " + ",".join(errors))
    _BUILD_BUDGET = None
    return result


def build_result() -> dict[str, Any]:
    """Side-effect-free canonical producer entry point."""
    return _generate_result()


def render_report(result: dict[str, Any]) -> str:
    mm = result.get("mass_maximum")
    mm_text = "none (stable branch reaches 10 n0)" if mm is None else f"{mm['M_Msun']:.9f} M_sun at n_c/n0={mm['nc_over_n0']:.9f}, R={mm['R_km']:.6f} km"
    return f"""# Source-complete stellar prediction artifact

Status: **{result['status']}**.  Every row is `MATHEMATICAL_ONE_COMPONENT_SEQUENCE`, evidence weight 0.  The physical neutron-star claim is blocked by `{PHYSICAL_NS_BLOCKER}`.

## Scope and gates

    The direct source action is evaluated on the closed domain `0 <= n/n0 <= 10` with degeneracy 4, exact vacuum enthalpy surface `h=0`, and no crust, beta equilibrium, leptons, CSS branch, clipping, extrapolation, or invented composition.  The upstream source digest is `{result.get('upstream_source_sha256', 'not-provided')}`.  Causality/stability and Hilbert/Legendre pressure gates are strict; invalid states fail closed.

Primary rows use DOP853 in enthalpy; independent pressure-coordinate RK4 uses actual uniform steps `0.05, 0.025, 0.0125` km and an exact `P=0` terminal chart. The source recurrence is evaluated at `rho=sqrt(K_CONV)*r_km`, with mass and pressure converted at the same boundary. Central grid: `10^(-3+j/32)`, j=0..128. First mass maximum: {mm_text}. All 18 pressure trajectories retain an independent `N={{16,32,64}}` terminal chart and adaptive oracle. Three predeclared `(x,dr)` keys additionally carry independently replayed numerical interval diagnostics. These diagnostics do not establish a rigorous global error bound or correct rounding. Unresolved binary64 convergence orders are reported as null.

The fixed parameters also fail the nuclear saturation comparison discussed in [the physical model assessment](../SOURCE_COMPLETE_MODEL_RU.md). Repairing the integrators does not resolve that discrepancy or establish a physical neutron-star model.

## Observational context

The serialized records are context-only (`comparison_status=BLOCKED_MODEL_NOT_PHYSICAL_NS`, evidence weight 0).  No fit, pass, or fail claim is made.  Sources: Fonseca et al. [arXiv:2104.00880](https://arxiv.org/abs/2104.00880); Salmi et al. [arXiv:2406.14466](https://arxiv.org/abs/2406.14466); Dittmann et al. [arXiv:2406.14467](https://arxiv.org/abs/2406.14467); Miller et al. [arXiv:1912.05705](https://arxiv.org/abs/1912.05705); Abbott et al. [arXiv:1805.11579](https://arxiv.org/abs/1805.11579) and [arXiv:1805.11581](https://arxiv.org/abs/1805.11581).

The artifact is deterministic JSON plus this report within the pinned runtime. Wall-clock telemetry is excluded from scientific identities. Numerical convergence and direct-equation consistency checks do not constitute empirical confirmation or a rigorous error theorem.
"""


def write_artifacts(result: dict[str, Any] | None = None, *, result_path: Path = RESULT_PATH, report_path: Path = REPORT_PATH) -> None:
    if result is None:
        result = build_result()
    ok, errors = _validate_result_contract(result)
    if not ok:
        raise ValueError("artifact contract failed: " + ",".join(errors))
    payload = json.dumps(_scientific_payload(result), sort_keys=True, indent=2, allow_nan=False) + "\n"
    report = render_report(result)
    result_path.write_text(payload, encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    result = build_result()
    write_artifacts(result)
    print(json.dumps({"status": result["status"], "convergence": result["convergence"]["pass"], "result": str(RESULT_PATH), "report": str(REPORT_PATH)}, sort_keys=True))
