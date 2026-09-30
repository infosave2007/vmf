#!/usr/bin/env python3
"""Live closed-mixing, caloric and fixed-pressure bridge for the accepted EOS.

This producer intentionally owns a *new* thermodynamic calculation.  It uses
only the maintained finite-temperature integrator and the accepted W8/U16
constructor; saved JSON is an output snapshot and is never a scientific input.
The result is a bounded homogeneous-equilibrium endpoint study, not a rate,
power, device, global phase, or experimental claim.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import mpmath as mp

HERE = Path(__file__).resolve().parent
VERIFICATION = HERE.parent / "verification"
if str(VERIFICATION) not in sys.path:
    sys.path.insert(0, str(VERIFICATION))

# Maintained EOS/integral producers.  We use constructors and live integrals,
# never their saved result JSON or the previous private nonlinear build.
import nvg_nonlinear_calibration_response as nonlinear  # noqa: E402
import nvg_thermal_observable_bridge as thermal  # noqa: E402
import nvg_thermal_composition_hessian as composition  # noqa: E402

SCHEMA_VERSION = "nvg_closed_mixing_bridge.v1"
STATUS_PASS = "PASS_LIVE_CLOSED_MIXING_BRIDGE"
STATUS_FAIL = "FAIL_LIVE_CLOSED_MIXING_BRIDGE"
EVIDENCE_WEIGHT = "0.0"

ANCHORS = ("0.90", "0.93")
MODEL_ORDER = ("w8", "u16")
MODEL_AMPLITUDES = {"w8": 0, "u16": 1}
CONTACT_ORDER = ("zero", "fixed_j")
T0_TEXT = "70"
MU_REF_TEXT = "775"
# Baseline plus both signs are retained: no hidden sign symmetry is assumed by
# the validator.  Positive rows are used for the declared 8/12-point work test.
DELTAS = ("0", "-0.025", "0.025", "-0.05", "0.05", "-0.1", "0.1", "-0.2", "0.2")
POSITIVE_DELTAS = ("0.025", "0.05", "0.1", "0.2")

J_MEV_TEXT = composition.J_MEV_TEXT
N0_FM3_TEXT = composition.N0_FM3_TEXT
HBARC_MEV_FM_TEXT = composition.HBARC_MEV_FM_TEXT
with mp.workdps(80):
    J_MEV = mp.mpf(J_MEV_TEXT)
    N0_FM3 = mp.mpf(N0_FM3_TEXT)
    HBARC = mp.mpf(HBARC_MEV_FM_TEXT)
    C_RHO_MEV_FM3 = 2 * J_MEV / N0_FM3
    C_RHO_NATURAL = C_RHO_MEV_FM3 / HBARC**3

PRIMARY_DPS = 44
CONTROL_DPS = 56
PRIMARY_NQUAD = 80
CONTROL_NQUAD = 112
PRIMARY_CUTOFF = mp.mpf("4500")
CONTROL_CUTOFF = mp.mpf("5200")
ROOT_REL_LIMIT = mp.mpf("2e-25")
ENERGY_REL_LIMIT = mp.mpf("2e-18")
GIBBS_REL_LIMIT = mp.mpf("3e-18")
COEFFICIENT_REL_LIMIT = mp.mpf("5e-5")
REFINEMENT_REL_LIMIT = mp.mpf("3e-4")
ENDPOINT_REL_LIMIT = mp.mpf("2e-3")
ISOBARIC_REL_LIMIT = mp.mpf("2e-18")
T_STEPS = (mp.mpf("0.1"), mp.mpf("0.05"))
B_REL_STEPS = (mp.mpf("1e-3"), mp.mpf("5e-4"))
ROOT_T_MIN = mp.mpf("0.5")
ROOT_T_MAX = mp.mpf("400")
ROOT_B_MIN_RATIO = mp.mpf("0.15")
ROOT_B_MAX_RATIO = mp.mpf("3")
EXPECTED_REFERENCE_COUNT = len(ANCHORS) * len(MODEL_ORDER)
EXPECTED_COMPOSITION_COUNT = EXPECTED_REFERENCE_COUNT * len(DELTAS)
EXPECTED_ROW_COUNT = EXPECTED_COMPOSITION_COUNT * len(CONTACT_ORDER)


class ClosedMixingError(ValueError):
    """Invalid input, failed local endpoint or failed numerical control."""


def _mp(value: Any) -> mp.mpf:
    if isinstance(value, mp.mpf):
        out = value
    elif isinstance(value, bool):
        raise ClosedMixingError("boolean is not a finite scalar")
    else:
        try:
            out = mp.mpf(str(value))
        except (TypeError, ValueError) as exc:
            raise ClosedMixingError("value must be a finite real scalar") from exc
    if not mp.isfinite(out):
        raise ClosedMixingError("value must be finite")
    return out


def _num(value: Any, digits: int = 42) -> str:
    value = _mp(value)
    return mp.nstr(value, digits)


def _rel(a: Any, b: Any, *, floor: Any = "1e-35") -> mp.mpf:
    aa, bb = _mp(a), _mp(b)
    return abs(aa - bb) / max(abs(aa), abs(bb), _mp(floor))


def _abs_scaled(a: Any, scale: Any, *, floor: Any = "1e-35") -> mp.mpf:
    return abs(_mp(a)) / max(abs(_mp(scale)), _mp(floor))


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(k) and _finite_tree(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(v) for v in value)
    if isinstance(value, (str, bool)) or value is None:
        return True
    try:
        return bool(mp.isfinite(_mp(value)))
    except (TypeError, ValueError, ClosedMixingError):
        return False


def _jsonable(value: Any) -> Any:
    if isinstance(value, mp.mpf):
        return _num(value)
    if isinstance(value, mp.mpc):
        raise ClosedMixingError("complex scientific value cannot be serialized")
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _canonical_payload(value: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _matrix_relative(a: Sequence[Sequence[Any]], b: Sequence[Sequence[Any]], *, floor: Any = "1e-35") -> mp.mpf:
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return mp.inf
    return max((_rel(x, y, floor=floor) for ra, rb in zip(a, b) for x, y in zip(ra, rb)), default=mp.mpf(0))


def _gauss_legendre_average(a: mp.mpf, b: mp.mpf, count: int) -> tuple[tuple[mp.mpf, mp.mpf], ...]:
    nodes, weights = mp.gauss_quadrature(int(count), "legendre")
    mid, half = (a + b) / 2, (b - a) / 2
    return tuple((mid + half * nodes[i], half * weights[i]) for i in range(int(count)))


def _model_info(anchor: str) -> dict[str, dict[str, Any]]:
    infos = nonlinear._build_models(anchor)
    return infos


class _CanonicalIntegrator:
    """Cheap n/ns-only view of the maintained thermal quadrature.

    Canonical Newton iterations need only net density and scalar density.  The
    final state still uses the maintained full integrator for pressure, energy,
    entropy and direct quadrature.  Reusing its grid/functions avoids repeated
    log-pressure and entropy work at every root trial without changing an EOS
    equation.
    """

    def __init__(self, full: Any):
        self.temperature = full.temperature
        self.cutoff = full.cutoff
        self.nquad = full.nquad
        self._grid = full._grid
        self._cache: dict[tuple[str, str], dict[str, mp.mpf]] = {}

    def integrals(self, nu: Any, y: Any, *, mass_scale: Any | None = None) -> dict[str, mp.mpf]:
        if mass_scale is not None:
            # Canonical roots never request this branch; retain a clear failure
            # rather than silently returning an inconsistent derivative.
            raise ClosedMixingError("n/ns-only root integrator does not expose mass derivatives")
        nu, y = _mp(nu), _mp(y)
        key = (_num(nu, mp.mp.dps + 8), _num(y, mp.mp.dps + 8))
        if key in self._cache:
            return self._cache[key]
        m = _mp(thermal._THERMAL_MASS_SCALE) * y
        pref = mp.mpf(thermal.DEGENERACY) / (2 * mp.pi**2)
        nsum: list[mp.mpf] = []
        nssum: list[mp.mpf] = []
        for p, weight in self._grid:
            e = mp.sqrt(p * p + m * m)
            fp = thermal._fermi_argument((e - nu) / self.temperature)
            fa = thermal._fermi_argument((e + nu) / self.temperature)
            p2 = p * p
            nsum.append(weight * p2 * (fp - fa))
            nssum.append(weight * p2 * (m / e) * (fp + fa))
        out = {"n": pref * mp.fsum(nsum), "ns": pref * mp.fsum(nssum)}
        self._cache[key] = out
        return out


@dataclass
class Reference:
    anchor: str
    model_name: str
    model_id: str
    amplitude: int
    target_y: mp.mpf
    model: Any
    T0: mp.mpf
    mu_reference: mp.mpf
    nu_reference: mp.mpf
    y_reference: mp.mpf
    B0: mp.mpf
    B0_fm3: mp.mpf
    reference_pressure: mp.mpf
    reference_root_residual: mp.mpf


class LiveEOS:
    """One anchor/model live EOS with canonical roots cached by (T,B,D)."""

    def __init__(self, reference: Reference, *, dps: int, nquad: int, cutoff: mp.mpf):
        self.reference = reference
        self.dps = int(dps)
        self.nquad = int(nquad)
        self.cutoff = _mp(cutoff)
        self.integrators: dict[str, Any] = {}
        self.root_integrators: dict[str, Any] = {}
        self._base_states: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._roots: dict[tuple[str, str, str], Any] = {}
        self._endpoint_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._isobaric_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def integrator(self, temperature: Any) -> Any:
        key = _num(temperature, self.dps + 8)
        if key not in self.integrators:
            self.integrators[key] = thermal._integrator_for(temperature, cutoff=self.cutoff, nquad=self.nquad)
        return self.integrators[key]

    def root_integrator(self, temperature: Any) -> Any:
        key = _num(temperature, self.dps + 8)
        if key not in self.root_integrators:
            self.root_integrators[key] = _CanonicalIntegrator(self.integrator(temperature))
        return self.root_integrators[key]

    def _guess(self, temperature: mp.mpf, B: mp.mpf, D: mp.mpf) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
        # Prefer a nearby solved state; otherwise use the accepted symmetric
        # grand-canonical root and a small declared composition split.
        if self._roots:
            target = (temperature / self.reference.T0 - 1, B / self.reference.B0 - 1, D / self.reference.B0)
            key = min(self._roots, key=lambda k: abs(_mp(k[0]) - target[0]) + abs(_mp(k[1]) - target[1]) + abs(_mp(k[2]) - target[2]))
            old = self._roots[key]
            return old.nu_n, old.nu_p, old.y
        split = mp.mpf("35") * D / max(B, mp.mpf("1"))
        return self.reference.nu_reference + split, self.reference.nu_reference - split, self.reference.y_reference

    def _root(self, temperature: mp.mpf, B: mp.mpf, D: mp.mpf) -> Any:
        if B <= 0 or abs(D) >= B:
            raise ClosedMixingError("canonical state requires B>0 and |D|<B")
        key = (_num(temperature, self.dps + 8), _num(B, self.dps + 8), _num(D, self.dps + 8))
        if key in self._roots:
            return self._roots[key]
        # Root iterations use the n/ns-only view of the same maintained
        # quadrature grid; all published thermodynamics uses the full view.
        integrator = self.root_integrator(temperature)
        guesses = [self._guess(temperature, B, D)]
        base = guesses[0]
        for ds in (mp.mpf("0"), mp.mpf("25"), mp.mpf("-25")):
            for ys in (mp.mpf("1"), mp.mpf("0.92"), mp.mpf("1.08")):
                guesses.append((base[0] + ds, base[1] - ds, base[2] * ys))
        last: Exception | None = None
        root = None
        for guess in guesses:
            try:
                root = composition._canonical_root(
                    integrator, self.reference.model, B, D,
                    nu_n0=guess[0], nu_p0=guess[1], y0=guess[2],
                )
                break
            except Exception as exc:  # bounded local retries only
                last = exc
        if root is None:
            raise ClosedMixingError(f"canonical root failed: {type(last).__name__}: {last}")
        self._roots[key] = root
        return root

    def base_state(self, temperature: Any, B: Any | None = None, D: Any | None = None, *, with_hessian: bool = True) -> dict[str, Any]:
        temperature = _mp(temperature)
        B = self.reference.B0 if B is None else _mp(B)
        D = mp.mpf(0) if D is None else _mp(D)
        key = (_num(temperature, self.dps + 8), _num(B, self.dps + 8), _num(D, self.dps + 8))
        if key in self._base_states and (not with_hessian or "hessian_nat" in self._base_states[key]):
            return self._base_states[key]
        root = self._root(temperature, B, D)
        integrator = self.integrator(temperature)
        ndata = composition._species_integrals(integrator, root.nu_n, root.y)
        pdata = composition._species_integrals(integrator, root.nu_p, root.y)
        kinetic_p = ndata["pressure"] + pdata["pressure"]
        kinetic_u = ndata["energy"] + pdata["energy"]
        entropy = ndata["entropy"] + pdata["entropy"]
        potential, _, _ = thermal._model_potential(self.reference.model, root.y)
        vector = self.reference.model.Cv * B**2 / (2 * root.y**2)
        free = root.nu_n * root.n_n + root.nu_p * root.n_p - kinetic_p + potential + vector
        pressure = kinetic_p - potential + vector
        energy = kinetic_u + potential + vector
        out: dict[str, Any] = {
            "temperature": temperature, "B": B, "D": D, "delta": D / B,
            "root": root, "nu_n": root.nu_n, "nu_p": root.nu_p, "y": root.y,
            "n_n": root.n_n, "n_p": root.n_p, "kinetic_pressure": kinetic_p,
            "kinetic_energy": kinetic_u, "entropy": entropy, "potential": potential,
            "vector_energy": vector, "free_energy": free, "pressure": pressure,
            "energy": energy, "energy_direct": energy, "root_residual": root.residual_relative,
            "B_fm3": B / HBARC**3, "D_fm3": D / HBARC**3,
            "entropy_fm3": entropy / HBARC**3, "free_energy_fm3": free / HBARC**3,
            "pressure_fm3": pressure / HBARC**3, "energy_fm3": energy / HBARC**3,
            "branch": "bounded_positive_species_positive_y_local_canonical_branch",
        }
        # Thermodynamic identities computed from independent integral pieces.
        out["energy_ledger_residual"] = _rel(energy, free + temperature * entropy, floor="1e-25")
        out["mu_B"] = (root.nu_n + root.nu_p) / 2 + self.reference.model.Cv * B / root.y**2
        out["mu_D"] = (root.nu_n - root.nu_p) / 2
        out["gibbs_identity_residual"] = _rel(free + pressure, out["mu_B"] * B + out["mu_D"] * D, floor="1e-25")
        if with_hessian:
            derivatives = composition._derivatives(integrator, self.reference.model, root)
            h0 = composition._hessian_and_state(integrator, self.reference.model, root, derivatives, mp.mpf(0))
            H_fm = h0["H_BD_MeV_fm3"]
            H_nat = [[v / HBARC**3 for v in row] for row in H_fm]
            out["hessian_fm"] = H_fm
            out["hessian_nat"] = H_nat
            out["field_curvature"] = h0["field_curvature_MeV4"]
            out["local_stable"] = bool(h0["local_stable"])
            out["pressure_identity_residual"] = h0["pressure_identity_relative"]
            # Set above for both Hessian and no-Hessian states.
        self._base_states[key] = out
        return out

    def state(self, temperature: Any, B: Any | None = None, D: Any | None = None, contact: str = "zero", *, with_hessian: bool = True) -> dict[str, Any]:
        base = self.base_state(temperature, B, D, with_hessian=with_hessian)
        if contact not in CONTACT_ORDER:
            raise ClosedMixingError("unknown contact")
        c = C_RHO_NATURAL if contact == "fixed_j" else mp.mpf(0)
        if c == 0:
            contact_energy = mp.mpf(0)
        else:
            contact_energy = c * base["D"]**2 / 2
        out = dict(base)
        out["contact"] = contact
        out["contact_nat"] = c
        out["contact_energy"] = contact_energy
        out["free_energy"] = base["free_energy"] + contact_energy
        out["pressure"] = base["pressure"] + contact_energy
        out["energy"] = base["energy"] + contact_energy
        out["energy_direct"] = out["energy"]
        out["free_energy_fm3"] = out["free_energy"] / HBARC**3
        out["pressure_fm3"] = out["pressure"] / HBARC**3
        out["energy_fm3"] = out["energy"] / HBARC**3
        out["energy_ledger_residual"] = _rel(out["energy"], out["free_energy"] + out["temperature"] * out["entropy"], floor="1e-25")
        out["mu_B"] = base.get("mu_B", (base["nu_n"] + base["nu_p"]) / 2 + self.reference.model.Cv * base["B"] / base["y"]**2)
        out["mu_D"] = base.get("mu_D", (base["nu_n"] - base["nu_p"]) / 2) + c * base["D"]
        out["pressure_identity_residual"] = _rel(out["pressure"], out["mu_B"] * base["B"] + out["mu_D"] * base["D"] - out["free_energy"], floor="1e-25")
        if with_hessian and "hessian_nat" in base:
            H = [list(row) for row in base["hessian_nat"]]
            H[1][1] += c
            Hfm = [[v * HBARC**3 for v in row] for row in H]
            out["hessian_nat"] = H
            out["hessian_fm"] = Hfm
            det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
            out["local_stable"] = bool(base.get("field_curvature", 0) > 0 and H[0][0] > 0 and det > 0)
            out["pressure_identity_residual"] = _rel(out["pressure"], out["mu_B"] * base["B"] + out["mu_D"] * base["D"] - out["free_energy"], floor="1e-25")
        out["contact_shift_expected"] = c * base["D"]**2 / 2
        return out

    def _solve_temperature(self, function: Any, guess: Any, *, scale: Any, label: str) -> tuple[mp.mpf, dict[str, Any], mp.mpf]:
        guess = _mp(guess)
        cache: dict[str, tuple[mp.mpf, dict[str, Any]]] = {}
        def eval_at(t: Any) -> mp.mpf:
            t = _mp(t)
            if t <= ROOT_T_MIN or t >= ROOT_T_MAX:
                raise ClosedMixingError(f"{label} temperature left bounded interval")
            key = _num(t, self.dps + 8)
            if key not in cache:
                state = function(t)
                cache[key] = (_mp(state[0]), state[1])
            return cache[key][0]
        try:
            root_t = mp.findroot(eval_at, (max(ROOT_T_MIN * 2, guess - mp.mpf("1")), guess + mp.mpf("1")), solver="secant", tol=mp.eps ** mp.mpf("0.68"), maxsteps=30, verify=False)
        except Exception:
            # Explicit monotone bracket fallback; no silent endpoint estimate.
            lo, hi = max(ROOT_T_MIN * 2, guess / 2), min(ROOT_T_MAX / 2, guess * 2)
            flo, fhi = eval_at(lo), eval_at(hi)
            for _ in range(12):
                if flo * fhi <= 0:
                    break
                if abs(flo) < abs(fhi):
                    lo = max(ROOT_T_MIN * 2, lo / 2)
                    flo = eval_at(lo)
                else:
                    hi = min(ROOT_T_MAX * mp.mpf("0.99"), hi * 1.5)
                    fhi = eval_at(hi)
            if flo * fhi > 0:
                raise ClosedMixingError(f"{label} endpoint has no bounded bracket")
            for _ in range(100):
                mid = (lo + hi) / 2
                fm = eval_at(mid)
                if abs(fm) <= mp.eps ** mp.mpf("0.65") * max(abs(scale), mp.mpf(1)):
                    break
                if flo * fm <= 0:
                    hi, fhi = mid, fm
                else:
                    lo, flo = mid, fm
            root_t = mid
        root_t = _mp(root_t)
        if not (ROOT_T_MIN < root_t < ROOT_T_MAX):
            raise ClosedMixingError(f"{label} endpoint outside bound")
        residual, state = cache.get(_num(root_t, self.dps + 8), (eval_at(root_t), function(root_t)[1]))
        rel = _abs_scaled(residual, scale, floor="1e-30")
        if rel > mp.mpf("2e-18"):
            raise ClosedMixingError(f"{label} endpoint residual {rel}")
        return root_t, state, rel

    def solve_energy_temperature(self, target_energy: mp.mpf, B: mp.mpf, D: mp.mpf, contact: str, guess: mp.mpf) -> tuple[mp.mpf, dict[str, Any], mp.mpf]:
        def fun(t: mp.mpf) -> tuple[mp.mpf, dict[str, Any]]:
            s = self.state(t, B, D, contact, with_hessian=False)
            return s["energy"] - target_energy, s
        return self._solve_temperature(fun, guess, scale=max(abs(target_energy), mp.mpf(1)), label="energy")

    def solve_entropy_temperature(self, target_entropy: mp.mpf, B: mp.mpf, D: mp.mpf, contact: str, guess: mp.mpf) -> tuple[mp.mpf, dict[str, Any], mp.mpf]:
        def fun(t: mp.mpf) -> tuple[mp.mpf, dict[str, Any]]:
            s = self.state(t, B, D, contact, with_hessian=False)
            return s["entropy"] / B - target_entropy, s
        return self._solve_temperature(fun, guess, scale=max(abs(target_entropy), mp.mpf(1)), label="entropy")

    def solve_pressure_density(self, target_pressure: mp.mpf, delta: mp.mpf, T: mp.mpf, contact: str, guess: mp.mpf) -> tuple[mp.mpf, dict[str, Any], mp.mpf]:
        cache: dict[str, tuple[mp.mpf, dict[str, Any]]] = {}
        def eval_at(B: Any) -> mp.mpf:
            B = _mp(B)
            if B <= self.reference.B0 * ROOT_B_MIN_RATIO or B >= self.reference.B0 * ROOT_B_MAX_RATIO:
                raise ClosedMixingError("isobaric B left bounded interval")
            key = _num(B, self.dps + 8)
            if key not in cache:
                state = self.state(T, B, delta * B, contact, with_hessian=False)
                cache[key] = (state["pressure"] - target_pressure, state)
            return cache[key][0]
        try:
            root_b = mp.findroot(eval_at, (guess * mp.mpf("0.995"), guess * mp.mpf("1.005")), solver="secant", tol=mp.eps ** mp.mpf("0.68"), maxsteps=30, verify=False)
        except Exception as exc:
            raise ClosedMixingError(f"isobaric root failed: {type(exc).__name__}: {exc}") from exc
        root_b = _mp(root_b)
        if root_b <= 0:
            raise ClosedMixingError("isobaric root nonpositive")
        residual = eval_at(root_b)
        state = cache[_num(root_b, self.dps + 8)][1]
        rel = _abs_scaled(residual, target_pressure, floor="1e-20")
        if rel > ISOBARIC_REL_LIMIT:
            raise ClosedMixingError(f"isobaric pressure residual {rel}")
        return root_b, state, rel


def _reference(dps: int, *, nquad: int, cutoff: mp.mpf) -> list[Reference]:
    refs: list[Reference] = []
    with mp.workdps(dps):
        for anchor in ANCHORS:
            for model_name in MODEL_ORDER:
                info = _model_info(anchor)[model_name]
                model = info["model"]
                T0, mu = _mp(T0_TEXT), _mp(MU_REF_TEXT)
                integ = thermal._integrator_for(T0, cutoff=cutoff, nquad=nquad)
                try:
                    grand = thermal._root_record(integ, model, mu, mu, _mp(anchor), with_derivatives=False)
                except Exception as exc:
                    raise ClosedMixingError(f"reference root failed {anchor}/{model_name}: {exc}") from exc
                if not grand.get("stable"):
                    raise ClosedMixingError(f"reference root unstable {anchor}/{model_name}")
                state = grand["state"]
                B0 = _mp(state["n_MeV3"])
                refs.append(Reference(anchor, model_name, info["id"], int(info["amplitude"]), _mp(info["target_y"]), model, T0, mu, _mp(grand["nu"]), _mp(grand["y"]), B0, B0 / HBARC**3, _mp(state["pressure_MeV4"]), _mp(grand["residual_relative"])))
    return refs


def _public_root(root: Any) -> dict[str, Any]:
    return {
        "B_MeV3": root.B, "D_MeV3": root.D,
        "B_fm3": root.B / HBARC**3, "D_fm3": root.D / HBARC**3,
        "delta": root.D / root.B, "nu_n_MeV": root.nu_n, "nu_p_MeV": root.nu_p,
        "y": root.y, "n_n_fm3": root.n_n / HBARC**3, "n_p_fm3": root.n_p / HBARC**3,
        "gap_residual_MeV4": root.gap_residual, "root_residual_relative": root.residual_relative,
        "density_identity_relative": max(_rel(root.n_n + root.n_p, root.B, floor="1e-30"), _rel(root.n_n - root.n_p, root.D, floor="1e-30")),
        "branch": "bounded_positive_species_positive_y_local_canonical_branch",
    }


def _public_state(state: Mapping[str, Any]) -> dict[str, Any]:
    hnat = state.get("hessian_nat")
    hfm = state.get("hessian_fm")
    keep = {
        "T_MeV": state["temperature"], "B_MeV3": state["B"], "D_MeV3": state["D"], "B_fm3": state["B_fm3"], "D_fm3": state["D_fm3"], "delta": state["delta"],
        "y": state["y"], "free_energy_MeV4": state["free_energy"], "free_energy_MeV_fm3": state["free_energy_fm3"], "energy_MeV4": state["energy"], "energy_MeV_fm3": state["energy_fm3"], "energy_direct_MeV4": state["energy_direct"],
        "entropy_MeV3": state["entropy"], "entropy_fm3": state["entropy_fm3"], "pressure_MeV4": state["pressure"], "pressure_MeV_fm3": state["pressure_fm3"], "mu_B_MeV": state.get("mu_B", mp.nan), "mu_D_MeV": state.get("mu_D", mp.nan),
        "root": _public_root(state["root"]), "energy_ledger_residual": state["energy_ledger_residual"], "gibbs_identity_residual": state["gibbs_identity_residual"], "pressure_identity_residual": state.get("pressure_identity_residual", mp.nan), "root_residual_relative": state["root_residual"], "local_stable": state.get("local_stable", True), "contact": state["contact"], "contact_energy_MeV4": state["contact_energy"],
    }
    if hnat is not None:
        keep["hessian_nat_MeVminus2"] = hnat
        keep["hessian_MeV_fm3"] = hfm
    return keep


def _solve_endpoints(eos: LiveEOS, delta: mp.mpf, contact: str, *, base0: dict[str, Any], prep: dict[str, Any]) -> dict[str, Any]:
    B, D, T0 = eos.reference.B0, delta * eos.reference.B0, eos.reference.T0
    # The endpoint thermodynamics is even in D.  Cache by |delta| so the
    # signed rows remain present without duplicating three nonlinear solves.
    cache_key = (_num(abs(delta), eos.dps + 8), contact)
    if cache_key in eos._endpoint_cache:
        return eos._endpoint_cache[cache_key]
    if delta == 0:
        zero = eos.state(T0, B, mp.mpf(0), contact, with_hessian=False)
        endpoint = {
            "closed_mixing": {"Tmix_MeV": T0, "Tmix_minus_T0_MeV": mp.mpf(0), "mix_endpoint": _public_state(zero), "delta_sigma_mix": mp.mpf(0), "energy_residual_relative": mp.mpf(0)},
            "reversible_separation": {"Tsep_MeV": T0, "Tsep_minus_T0_MeV": mp.mpf(0), "sep_endpoint": _public_state(zero), "entropy_target_per_B": zero["entropy"] / B, "entropy_residual_relative": mp.mpf(0), "wrev_MeV": mp.mpf(0)},
            "composed_endpoint": {"Tcycle_MeV": T0, "Tcycle_minus_T0_MeV": mp.mpf(0), "cycle_endpoint": _public_state(zero), "delta_sigma_cycle": mp.mpf(0), "energy_residual_relative": mp.mpf(0), "x_hot_MeV": mp.mpf(0)},
        }
        eos._endpoint_cache[cache_key] = endpoint
        return endpoint
    # Closed mixing: mean of +/-D prepared volumes equals the +D state by
    # exact charge-exchange symmetry, but both signs remain live rows.
    tmix, mix, mix_res = eos.solve_energy_temperature(prep["energy"], B, mp.mpf(0), contact, T0)
    sigma_mix = (mix["entropy"] - prep["entropy"]) / B
    # Reversible isentropic demixing from symmetric reference.
    sep_target = base0["entropy"] / B
    tsep, sep, sep_res = eos.solve_entropy_temperature(sep_target, B, D, contact, T0)
    wrev = (sep["energy"] - base0["energy"]) / B
    # Return to symmetric composition adiabatically, then measure irreversible
    # entropy production at the same energy.
    tcycle, cycle, cycle_res = eos.solve_energy_temperature(sep["energy"], B, mp.mpf(0), contact, T0)
    sigma_cycle = (cycle["entropy"] - sep["entropy"]) / B
    xhot = wrev - T0 * sigma_cycle
    endpoint = {
        "closed_mixing": {"Tmix_MeV": tmix, "Tmix_minus_T0_MeV": tmix - T0, "mix_endpoint": _public_state(mix), "delta_sigma_mix": sigma_mix, "energy_residual_relative": mix_res},
        "reversible_separation": {"Tsep_MeV": tsep, "Tsep_minus_T0_MeV": tsep - T0, "sep_endpoint": _public_state(sep), "entropy_target_per_B": sep_target, "entropy_residual_relative": sep_res, "wrev_MeV": wrev},
        "composed_endpoint": {"Tcycle_MeV": tcycle, "Tcycle_minus_T0_MeV": tcycle - T0, "cycle_endpoint": _public_state(cycle), "delta_sigma_cycle": sigma_cycle, "energy_residual_relative": cycle_res, "x_hot_MeV": xhot},
    }
    eos._endpoint_cache[cache_key] = endpoint
    return endpoint


def _isothermal_work(eos: LiveEOS, delta: mp.mpf, contact: str, base0: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    B, T0 = eos.reference.B0, eos.reference.T0
    D = delta * B
    delta_u = (prepared["energy"] - base0["energy"]) / B
    delta_s = (prepared["entropy"] - base0["entropy"]) / B
    wiso = (prepared["free_energy"] - base0["free_energy"]) / B
    # Rebuild μ_D along the live canonical path; this is independent of the
    # free-energy difference and therefore catches a false heat/work shortcut.
    estimates: list[dict[str, Any]] = []
    for count in (8, 12):
        integral = mp.mpf(0)
        for d, weight in _gauss_legendre_average(mp.mpf(0), D, count):
            st = eos.state(T0, B, d, contact, with_hessian=False)
            integral += weight * st["mu_D"]
        value = integral / B
        estimates.append({"points": count, "integral_work_MeV": value, "relative_to_wiso": _rel(value, wiso, floor="1e-30")})
    return {"delta_u_per_B_MeV": delta_u, "delta_sigma_sep": delta_s, "heat_Qprep_MeV": T0 * delta_s, "wiso_MeV": wiso, "work_identity_residual": _rel(wiso, delta_u - T0 * delta_s, floor="1e-30"), "muD_integral": estimates, "quadrature_pass": bool(max(r["relative_to_wiso"] for r in estimates) <= mp.mpf("2e-8"))}


def _derivative_state(eos: LiveEOS, T: mp.mpf, B: mp.mpf, contact: str) -> dict[str, Any]:
    return eos.state(T, B, mp.mpf(0), contact, with_hessian=True)


def _coefficient_payload(eos: LiveEOS, contact: str, base0: dict[str, Any]) -> dict[str, Any]:
    B, T = eos.reference.B0, eos.reference.T0
    H = base0["hessian_nat"]
    sf0 = B * H[1][1] / 2
    kt = 9 * B * H[0][0]
    b_checks: list[dict[str, Any]] = []
    # Four-point central derivative using each declared h and 2h; retain both
    # levels so the refinement is visible rather than fitted away.
    for relh in B_REL_STEPS:
        h = relh * B
        sm2 = B - 2*h; sp2 = B + 2*h; sm1 = B - h; sp1 = B + h
        vals = [_derivative_state(eos, T, x, contact) for x in (sm2, sm1, sp1, sp2)]
        sfm2, sfm1, sfp1, sfp2 = [x["B"] * x["hessian_nat"][1][1] / 2 for x in vals]
        d4 = (8 * (sfp1 - sfm1) - (sfp2 - sfm2)) / (12 * h)
        b_checks.append({"relative_step": relh, "dB_SF": d4, "L_F": 3 * B * d4, "states": [_public_state(x) for x in vals]})
    t_checks: list[dict[str, Any]] = []
    for h in T_STEPS:
        vals = [_derivative_state(eos, x, B, contact) for x in (T - 2*h, T - h, T + h, T + 2*h)]
        sfm2, sfm1, sfp1, sfp2 = [B * x["hessian_nat"][1][1] / 2 for x in vals]
        d4 = (8 * (sfp1 - sfm1) - (sfp2 - sfm2)) / (12 * h)
        um2, um1, up1, up2 = [x["energy"] / B for x in vals]
        cv = (8 * (up1 - um1) - (up2 - um2)) / (12 * h)
        t_checks.append({"step_MeV": h, "dT_SF": d4, "cv": cv, "states": [_public_state(x) for x in vals]})
    dB = b_checks[-1]["dB_SF"]
    dT = t_checks[-1]["dT_SF"]
    cv = t_checks[-1]["cv"]
    sf = sf0
    su = sf - T * dT
    theta_mix = su / cv
    theta_sep = T * dT / cv
    lf = 3 * B * dB
    a_mu = 9 * (sf - lf / 3) / kt
    a_p = -3 * lf / kt
    return {
        "S_F_MeV": sf, "S_U_MeV": su, "cv_dimensionless": cv, "dT_S_F": dT, "dB_S_F": dB,
        "Theta_mix_MeV": theta_mix, "Theta_sep_MeV": theta_sep, "L_F_MeV": lf, "K_T_MeV": kt,
        "A_mu": a_mu, "A_P": a_p, "identity_left": kt * (a_mu - a_p), "identity_right": 9 * cv * (theta_mix + theta_sep), "identity_target": 9 * sf,
        "identity_relative": max(_rel(kt * (a_mu - a_p), 9 * sf, floor="1e-30"), _rel(9 * cv * (theta_mix + theta_sep), 9 * sf, floor="1e-30")),
        "B_refinement": b_checks, "T_refinement": t_checks,
    }


def _classical_control() -> dict[str, Any]:
    # Use a high-precision decimal context even when called after the live
    # tree's workdps block has closed; these analytic controls are serialized
    # and later re-derived by the fail-closed validator.
    old_dps = mp.mp.dps
    mp.mp.dps = 80
    try:
        rows = []
        for delta_text in DELTAS:
            d = mp.mpf(delta_text)
            phi = ((1 + d) * mp.log(1 + d) + (1 - d) * mp.log(1 - d)) / 2 if d else mp.mpf(0)
            rows.append({"delta": d, "Phi": phi, "S_F": mp.mpf("35"), "S_U": mp.mpf(0), "cv": mp.mpf("1.5"), "Tmix_minus_T0": mp.mpf(0), "wiso_over_T": phi, "delta_sigma_mix": phi, "Tsep_over_T0": mp.exp(phi / mp.mpf("1.5")), "A_mu": mp.mpf("0.5"), "A_P": mp.mpf(0), "pass": bool(abs((mp.mpf("35") / mp.mpf("70")) - mp.mpf("0.5")) < mp.mpf("1e-40"))})
        return {"definition": "Phi(delta)=((1+delta)ln(1+delta)+(1-delta)ln(1-delta))/2", "rows": rows, "all_pass": all(r["pass"] for r in rows), "interpretation": "analytic classical ideal-mixture null: mixing entropy is not an energy source"}
    finally:
        mp.mp.dps = old_dps


def _public_reference(ref: Reference) -> dict[str, Any]:
    return {"anchor": ref.anchor, "model": ref.model_name, "model_id": ref.model_id, "amplitude": ref.amplitude, "target_y": ref.target_y, "T0_MeV": ref.T0, "mu_reference_MeV": ref.mu_reference, "nu_reference_MeV": ref.nu_reference, "y_reference": ref.y_reference, "B0_MeV3": ref.B0, "B0_fm3": ref.B0_fm3, "reference_pressure_MeV4": ref.reference_pressure, "reference_root_residual_relative": ref.reference_root_residual, "contact_Crho_MeV_fm3": C_RHO_MEV_FM3}


def _row(eos: LiveEOS, ref: Reference, delta_text: str, contact: str, *, coefficients: dict[str, Any], base0: dict[str, Any], include_quad: bool = True) -> dict[str, Any]:
    d = _mp(delta_text)
    B, D, T = ref.B0, d * ref.B0, ref.T0
    row_id = f"{ref.anchor}/{ref.model_name}/{delta_text}/{contact}"
    try:
        prepared = eos.state(T, B, D, contact, with_hessian=True)
        endpoint = _solve_endpoints(eos, d, contact, base0=base0, prep=prepared)
        if d == 0:
            iso = {"delta_u_per_B_MeV": mp.mpf(0), "delta_sigma_sep": mp.mpf(0), "heat_Qprep_MeV": mp.mpf(0), "wiso_MeV": mp.mpf(0), "work_identity_residual": mp.mpf(0), "muD_integral": [], "quadrature_pass": True, "quadrature_omitted": False}
        elif d == mp.mpf("0.2") and include_quad:
            # The sealed protocol reserves the expensive independent 8/12
            # point quadrature for delta=+0.2 only (both contacts); other grid
            # rows retain the ledger but explicitly omit that check.
            iso = _isothermal_work(eos, d, contact, base0, prepared)
            iso["quadrature_omitted"] = False
        else:
            delta_u = (prepared["energy"] - base0["energy"]) / B
            delta_s = (prepared["entropy"] - base0["entropy"]) / B
            wiso = (prepared["free_energy"] - base0["free_energy"]) / B
            iso = {"delta_u_per_B_MeV": delta_u, "delta_sigma_sep": delta_s, "heat_Qprep_MeV": T * delta_s, "wiso_MeV": wiso, "work_identity_residual": _rel(wiso, delta_u - T * delta_s, floor="1e-30"), "muD_integral": [], "quadrature_pass": True, "quadrature_omitted": True}
        # Exact small-delta tests use the frozen coefficient calculated at D=0;
        # no coefficient fitting is performed.
        sf, su, thm, ths = coefficients["S_F_MeV"], coefficients["S_U_MeV"], coefficients["Theta_mix_MeV"], coefficients["Theta_sep_MeV"]
        exact = {
            "free_coefficient_residual": _abs_scaled((prepared["free_energy"] - base0["free_energy"]) / B - sf * d**2, max(abs(sf * d**2), mp.mpf("1e-25"))),
            "internal_coefficient_residual": _abs_scaled((prepared["energy"] - base0["energy"]) / B - su * d**2, max(abs(su * d**2), mp.mpf("1e-25"))),
            "Tmix_coefficient_residual": _abs_scaled(endpoint["closed_mixing"]["Tmix_minus_T0_MeV"] - thm * d**2, max(abs(thm * d**2), mp.mpf("1e-25"))),
            "Tsep_coefficient_residual": _abs_scaled(endpoint["reversible_separation"]["Tsep_minus_T0_MeV"] - ths * d**2, max(abs(ths * d**2), mp.mpf("1e-25"))),
            "sigma_mix_coefficient_residual": _abs_scaled(endpoint["closed_mixing"]["delta_sigma_mix"] - sf / T * d**2, max(abs(sf / T * d**2), mp.mpf("1e-25"))),
        }
        gates = {
            "root": prepared["root_residual"] <= ROOT_REL_LIMIT,
            "stable": bool(prepared.get("local_stable", False)),
            "energy_ledger": prepared["energy_ledger_residual"] <= ENERGY_REL_LIMIT,
            "gibbs_identity": prepared["gibbs_identity_residual"] <= GIBBS_REL_LIMIT,
            "pressure_identity": prepared.get("pressure_identity_residual", mp.inf) <= GIBBS_REL_LIMIT,
            "endpoints": all(endpoint[key]["energy_residual_relative"] <= mp.mpf("2e-18") if key != "reversible_separation" else endpoint[key]["entropy_residual_relative"] <= mp.mpf("2e-18") for key in endpoint),
            "isothermal": iso["quadrature_pass"] and iso["work_identity_residual"] <= mp.mpf("2e-18"),
            "exact_endpoint_finite": all(_finite_tree(v) for v in exact.values()),
        }
        return {"row_id": row_id, "status": "CANONICAL_ROOT_OK" if all(gates.values()) else "CONTROL_FAIL", "pass": bool(all(gates.values())), "anchor": ref.anchor, "model": ref.model_name, "amplitude": ref.amplitude, "delta": delta_text, "contact": contact, "reference": _public_reference(ref), "prepared": _public_state(prepared), "closed_protocol": endpoint, "isothermal_separation": iso, "coefficient_endpoint_checks": exact, "gates": gates}
    except Exception as exc:
        return {"row_id": row_id, "status": "ROOT_FAILURE", "pass": False, "anchor": ref.anchor, "model": ref.model_name, "amplitude": ref.amplitude, "delta": delta_text, "contact": contact, "error": f"{type(exc).__name__}: {exc}", "gates": {"exception": False}}


def _isobaric_rows(eos: LiveEOS, ref: Reference, contact: str, delta_text: str, base0: dict[str, Any]) -> dict[str, Any]:
    d = _mp(delta_text)
    if d == 0:
        return {"anchor": ref.anchor, "model": ref.model_name, "delta": delta_text, "contact": contact, "status": "BASELINE", "pass": True, "B_ratio": mp.mpf(1), "V_ratio": mp.mpf(1), "pressure_residual_relative": mp.mpf(0)}
    try:
        b, st, residual = eos.solve_pressure_density(base0["pressure"], d, ref.T0, contact, ref.B0)
        return {"anchor": ref.anchor, "model": ref.model_name, "delta": delta_text, "contact": contact, "status": "ISOBARIC_ROOT_OK", "pass": bool(residual <= ISOBARIC_REL_LIMIT), "B_MeV3": b, "B_ratio": b / ref.B0, "V_ratio": ref.B0 / b, "pressure_residual_relative": residual, "state": _public_state(st)}
    except Exception as exc:
        return {"anchor": ref.anchor, "model": ref.model_name, "delta": delta_text, "contact": contact, "status": "ISOBARIC_ROOT_FAILURE", "pass": False, "error": f"{type(exc).__name__}: {exc}"}


def _precision(primary: Mapping[str, Any], control: Mapping[str, Any]) -> dict[str, Any]:
    pmap = {r["row_id"]: r for r in primary["rows"]}
    cmap = {r["row_id"]: r for r in control["rows"]}
    checks = []
    for rid, p in pmap.items():
        c = cmap.get(rid)
        if not c or not p.get("pass") or not c.get("pass"):
            checks.append({"row_id": rid, "pass": False, "status": "MISSING_OR_FAILED_CONTROL"})
            continue
        fields = ("free_energy_MeV4", "energy_MeV4", "entropy_MeV3", "pressure_MeV4", "y")
        vals = {f: _rel(p["prepared"][f], c["prepared"][f]) if f != "y" else _rel(p["prepared"][f], c["prepared"][f]) for f in fields}
        checks.append({"row_id": rid, "fields": vals, "max_relative_difference": max(vals.values()), "pass": bool(max(vals.values()) <= REFINEMENT_REL_LIMIT)})
    return {"rows": checks, "all_pass": bool(checks and all(r.get("pass") for r in checks)), "primary": {"dps": PRIMARY_DPS, "nquad": PRIMARY_NQUAD, "cutoff_MeV": PRIMARY_CUTOFF}, "control": {"dps": CONTROL_DPS, "nquad": CONTROL_NQUAD, "cutoff_MeV": CONTROL_CUTOFF}}


def _build_tree(dps: int, *, nquad: int, cutoff: mp.mpf, include_expensive: bool) -> dict[str, Any]:
    refs = _reference(dps, nquad=nquad, cutoff=cutoff)
    all_rows: list[dict[str, Any]] = []
    all_isobaric: list[dict[str, Any]] = []
    coeffs: dict[str, dict[str, Any]] = {}
    with mp.workdps(dps):
        for ref in refs:
            eos = LiveEOS(ref, dps=dps, nquad=nquad, cutoff=cutoff)
            base_zero = eos.state(ref.T0, ref.B0, mp.mpf(0), "zero", with_hessian=True)
            for contact in CONTACT_ORDER:
                base = eos.state(ref.T0, ref.B0, mp.mpf(0), contact, with_hessian=True)
                cp = _coefficient_payload(eos, contact, base)
                coeffs[f"{ref.anchor}/{ref.model_name}/{contact}"] = cp
                # Warm the exact positive/negative grid once.  The endpoint
                # solver reuses roots; no result snapshot is an input.
                for dt in DELTAS:
                    all_rows.append(_row(eos, ref, dt, contact, coefficients=cp, base0=base, include_quad=include_expensive))
                    if include_expensive:
                        all_isobaric.append(_isobaric_rows(eos, ref, contact, dt, base))
        if len(refs) != EXPECTED_REFERENCE_COUNT or len(all_rows) != EXPECTED_ROW_COUNT:
            raise ClosedMixingError("live grid coverage incomplete")
    # Derived contact identity, independently from row booleans.
    contact_checks = []
    for key in sorted({k.rsplit("/", 1)[0] for k in coeffs}):
        z, j = coeffs[key + "/zero"], coeffs[key + "/fixed_j"]
        expected = C_RHO_NATURAL * next(r.B0 for r in refs if f"{r.anchor}/{r.model_name}" == key) / 2
        contact_checks.append({"reference": key, "S_F_shift": j["S_F_MeV"] - z["S_F_MeV"], "S_U_shift": j["S_U_MeV"] - z["S_U_MeV"], "expected_shift": expected, "dT_SF_shift": j["dT_S_F"] - z["dT_S_F"], "identity_relative": max(_rel(j["S_F_MeV"] - z["S_F_MeV"], expected, floor="1e-30"), _rel(j["S_U_MeV"] - z["S_U_MeV"], expected, floor="1e-30")), "isentropic_derivative_unchanged": _rel(j["dT_S_F"], z["dT_S_F"], floor="1e-30") <= mp.mpf("1e-10")})
    controls = {
        "reference_coverage": {"count": len(refs), "expected": EXPECTED_REFERENCE_COUNT, "all_pass": all(r.reference_root_residual <= ROOT_REL_LIMIT for r in refs)},
        "composition_coverage": {"count": len({(r["anchor"], r["model"], r["delta"]) for r in all_rows}), "expected": EXPECTED_COMPOSITION_COUNT, "all_pass": len({(r["anchor"], r["model"], r["delta"]) for r in all_rows}) == EXPECTED_COMPOSITION_COUNT},
        "row_coverage": {"count": len(all_rows), "expected": EXPECTED_ROW_COUNT, "all_pass": len(all_rows) == EXPECTED_ROW_COUNT},
        "root_and_energy_gates": {"all_pass": all(r.get("pass", False) for r in all_rows), "failed_rows": [r["row_id"] for r in all_rows if not r.get("pass", False)]},
        "coefficient_identity": {"rows": coeffs, "all_pass": all(_mp(v["identity_relative"]) <= mp.mpf("1e-10") for v in []) if False else all(_mp(v["identity_relative"]) <= mp.mpf("1e-10") for v in [{"identity_relative": cp["identity_relative"]} for cp in coeffs.values()])},
        "contact_identity": {"rows": contact_checks, "all_pass": bool(contact_checks and all(c["identity_relative"] <= mp.mpf("1e-10") and c["isentropic_derivative_unchanged"] for c in contact_checks))},
        "classical_null": _classical_control(),
        "isobaric_coverage": {"rows": all_isobaric, "expected": EXPECTED_REFERENCE_COUNT * len(CONTACT_ORDER) * len(DELTAS), "all_pass": (not include_expensive) or (len(all_isobaric) == EXPECTED_REFERENCE_COUNT * len(CONTACT_ORDER) * len(DELTAS) and all(r.get("pass", False) for r in all_isobaric))},
    }
    return {"references": refs, "rows": all_rows, "coefficients": coeffs, "isobaric": all_isobaric, "controls": controls}


def _public_tree(tree: Mapping[str, Any]) -> dict[str, Any]:
    return {"references": [_public_reference(r) for r in tree["references"]], "rows": tree["rows"], "coefficients": tree["coefficients"], "isobaric": tree["isobaric"], "controls": tree["controls"]}


def _make_result() -> dict[str, Any]:
    primary = _build_tree(PRIMARY_DPS, nquad=PRIMARY_NQUAD, cutoff=PRIMARY_CUTOFF, include_expensive=True)
    control = _build_tree(CONTROL_DPS, nquad=CONTROL_NQUAD, cutoff=CONTROL_CUTOFF, include_expensive=False)
    precision = _precision(primary, control)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "status": STATUS_PASS, "evidence_weight": EVIDENCE_WEIGHT,
        "protocol_provenance": "SEALED_I1_CLOSED_MIXING_REVERSIBLE_CALORIC_FIXED_PRESSURE_FROM_ACCEPTED_EOS",
        "physical_inputs_fixed": True, "no_saved_results_as_inputs": True, "fit_used": False, "new_interaction": False,
        "inputs": {"anchors": list(ANCHORS), "model_order": list(MODEL_ORDER), "deformation_amplitudes": MODEL_AMPLITUDES, "T0_MeV": T0_TEXT, "mu_reference_MeV": MU_REF_TEXT, "deltas": list(DELTAS), "contacts": {"zero": "C_rho=0", "fixed_j": "C_rho=2j/n0 historical"}, "j_MeV": J_MEV_TEXT, "n0_fm3": N0_FM3_TEXT, "hbarc_MeV_fm": HBARC_MEV_FM_TEXT, "C_rho_MeV_fm3": C_RHO_MEV_FM3, "C_rho_natural_MeVminus2": C_RHO_NATURAL, "primary_dps": PRIMARY_DPS, "control_dps": CONTROL_DPS, "primary_nquad": PRIMARY_NQUAD, "control_nquad": CONTROL_NQUAD, "primary_cutoff_MeV": PRIMARY_CUTOFF, "control_cutoff_MeV": CONTROL_CUTOFF, "T_fd_steps_MeV": T_STEPS, "B_fd_relative_steps": B_REL_STEPS, "isothermal_gauss_points": [8, 12]},
        "units_and_conventions": {"natural": "hbar=c=1: B,D in MeV^3, f,u,P in MeV^4, s in MeV^3", "public_density": "fm^-3", "delta": "D/B", "protocols": {"closed_mixing": "fixed B, D=+/-delta B, no heat/work during mixing; solve u(B,0,Tmix)=u(B,D,T0)", "isothermal_separation": "fixed B,T0; wiso=delta f/B and independent integral of mu_D dD/B", "reversible_separation": "fixed B, s/B conserved; wrev=delta u/B", "isobaric": "fixed T0 and delta, solve P(B,delta B,T0)=P0"}},
        "references": [_public_reference(r) for r in primary["references"]], "rows": primary["rows"], "coefficients": primary["coefficients"], "isobaric": primary["isobaric"], "coverage": {"reference_count": len(primary["references"]), "expected_reference_count": EXPECTED_REFERENCE_COUNT, "composition_count": len({(r["anchor"], r["model"], r["delta"]) for r in primary["rows"]}), "expected_composition_count": EXPECTED_COMPOSITION_COUNT, "row_count": len(primary["rows"]), "expected_row_count": EXPECTED_ROW_COUNT, "all_rows_present": len(primary["rows"]) == EXPECTED_ROW_COUNT}, "controls": primary["controls"], "precision_controls": precision, "control_tree_summary": {"reference_count": len(control["references"]), "row_count": len(control["rows"]), "isobaric_count": len(control["isobaric"])},
        "derived_identity": "K_T(A_mu-A_P)=9 cv (Theta_mix+Theta_sep)=9 S_F; A_mu is fixed-mu_B local coefficient, A_P is fixed-P coefficient; ensembles remain distinct.",
        "interpretation_limits": ["Equilibrium endpoints conditional on a bounded stable homogeneous local branch; no global phase proof.", "Interface energy is omitted only in the macroscopic limit.", "No transition time, diffusivity, rate, power, household generator or device efficiency is calculated.", "The fixed-j contact is a historical conditional term, not a new interaction.", "The ideal classical control demonstrates that mixing entropy is not free energy or heat-source fuel."],
        "source_sha256": {"producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "thermal_observable_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(), "thermal_composition_hessian": hashlib.sha256(Path(composition.__file__).resolve().read_bytes()).hexdigest(), "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest()},
    }
    gates = ("reference_coverage", "composition_coverage", "row_coverage", "root_and_energy_gates", "coefficient_identity", "contact_identity", "classical_null", "isobaric_coverage")
    result["all_controls_pass"] = bool(all(result["controls"].get(name, {}).get("all_pass", False) for name in gates) and precision["all_pass"])
    result["status"] = STATUS_PASS if result["all_controls_pass"] else STATUS_FAIL
    result["integrity_sha256"] = hashlib.sha256(_canonical_payload(result).encode()).hexdigest()
    return _jsonable(result)


def _expected_row_keys() -> set[tuple[str, str, str, str]]:
    return {(a, m, d, c) for a in ANCHORS for m in MODEL_ORDER for d in DELTAS for c in CONTACT_ORDER}


def _true_number(value: Any) -> mp.mpf | None:
    try:
        x = _mp(value)
        return x if mp.isfinite(x) else None
    except Exception:
        return None


def _validate_numeric_states(result: Mapping[str, Any]) -> bool:
    """Recompute physical gates from serialized numbers, never booleans."""
    # JSON strings retain ~42 digits; avoid manufacturing residuals in products
    # such as mu*B under a caller's default 15-digit context.
    if mp.mp.dps < 65:
        old_dps = mp.mp.dps
        mp.mp.dps = 80
        try:
            return _validate_numeric_states(result)
        finally:
            mp.mp.dps = old_dps
    rows = result.get("rows", [])
    if not isinstance(rows, list):
        return False
    references = {(str(r["anchor"]), r["model"]): r for r in result.get("references", []) if isinstance(r, Mapping)}
    coeffs = result.get("coefficients", {})
    try:
        # Every nonzero-composition ledger is relative to its own exported
        # symmetric row.  Build that map first so a rehashed convenience
        # value (temperature, entropy change, or work) cannot be detached
        # from the endpoint states while the lower-level state identities
        # still happen to hold.
        baselines = {
            (str(row["anchor"]), row["model"], row["contact"]): row["prepared"]
            for row in rows
            if row.get("delta") == "0" and isinstance(row.get("prepared"), Mapping)
        }
        if len(baselines) != len(ANCHORS) * len(MODEL_ORDER) * len(CONTACT_ORDER):
            return False
        for row in rows:
            p = row["prepared"]
            # Canonical density identities and positive Hessian are derived
            # from the exported root/Hessian, not from ``local_stable``.
            root = p["root"]
            B, D = _mp(root["B_MeV3"]), _mp(root["D_MeV3"])
            nn, np_ = _mp(root["n_n_fm3"]), _mp(root["n_p_fm3"])
            Bf, Df = _mp(root["B_fm3"]), _mp(root["D_fm3"])
            if nn <= 0 or np_ <= 0 or _rel(nn + np_, Bf, floor="1e-30") > ROOT_REL_LIMIT or _rel(nn - np_, Df, floor="1e-30") > ROOT_REL_LIMIT:
                return False
            H = [[_mp(v) for v in part] for part in p["hessian_nat_MeVminus2"]]
            if len(H) != 2 or any(len(x) != 2 for x in H):
                return False
            det = H[0][0] * H[1][1] - H[0][1] * H[1][0]
            if H[0][0] <= 0 or H[1][1] <= 0 or det <= 0 or _rel(H[0][1], H[1][0], floor="1e-35") > mp.mpf("1e-20"):
                return False
            # Independent energy/Legendre ledgers.
            T = _mp(p["T_MeV"]); f = _mp(p["free_energy_MeV4"]); u = _mp(p["energy_MeV4"]); ud = _mp(p["energy_direct_MeV4"]); ss = _mp(p["entropy_MeV3"])
            if _rel(u, f + T * ss, floor="1e-25") > ENERGY_REL_LIMIT or _rel(ud, u, floor="1e-25") > ENERGY_REL_LIMIT:
                return False
            mu_b, mu_d, P = _mp(p["mu_B_MeV"]), _mp(p["mu_D_MeV"]), _mp(p["pressure_MeV4"])
            if _rel(P, mu_b * B + mu_d * D - f, floor="1e-25") > GIBBS_REL_LIMIT:
                return False
            if _mp(p["root_residual_relative"]) > ROOT_REL_LIMIT or _mp(p["energy_ledger_residual"]) > ENERGY_REL_LIMIT or _mp(p["pressure_identity_residual"]) > GIBBS_REL_LIMIT:
                return False
            if row.get("status") != "CANONICAL_ROOT_OK":
                return False
            # Closed-mixing energy, reversible entropy, and composed energy are
            # checked from endpoint states themselves.
            end = row["closed_protocol"]
            mixed, reversible, composed = (end[k] for k in ("closed_mixing", "reversible_separation", "composed_endpoint"))
            mix = mixed["mix_endpoint"]
            sep = reversible["sep_endpoint"]
            cyc = composed["cycle_endpoint"]
            baseline = baselines[(str(row["anchor"]), row["model"], row["contact"])]
            Bp, Dp = _mp(p["B_MeV3"]), _mp(p["D_MeV3"])
            Tbase = _mp(baseline["T_MeV"])
            # These are redundant serialized summary fields by design; bind
            # them back to the actual endpoint state and the declared T0.
            # This is deliberately independent of the producer's pass flags.
            if (_rel(_mp(mixed["Tmix_MeV"]), _mp(mix["T_MeV"]), floor="1e-25") > mp.mpf("2e-18")
                or _rel(_mp(reversible["Tsep_MeV"]), _mp(sep["T_MeV"]), floor="1e-25") > mp.mpf("2e-18")
                or _rel(_mp(composed["Tcycle_MeV"]), _mp(cyc["T_MeV"]), floor="1e-25") > mp.mpf("2e-18")
                or _rel(_mp(mixed["Tmix_minus_T0_MeV"]), _mp(mixed["Tmix_MeV"]) - Tbase, floor="1e-25") > mp.mpf("2e-18")
                or _rel(_mp(reversible["Tsep_minus_T0_MeV"]), _mp(reversible["Tsep_MeV"]) - Tbase, floor="1e-25") > mp.mpf("2e-18")
                or _rel(_mp(composed["Tcycle_minus_T0_MeV"]), _mp(composed["Tcycle_MeV"]) - Tbase, floor="1e-25") > mp.mpf("2e-18")):
                return False
            if (_rel(_mp(mix["B_MeV3"]), Bp, floor="1e-25") > mp.mpf("2e-18")
                or abs(_mp(mix["D_MeV3"])) > mp.mpf("2e-18") * max(abs(Bp), mp.mpf(1))
                or _rel(_mp(sep["B_MeV3"]), Bp, floor="1e-25") > mp.mpf("2e-18")
                or _rel(abs(_mp(sep["D_MeV3"])), abs(Dp), floor="1e-25") > mp.mpf("2e-18")
                or _rel(_mp(cyc["B_MeV3"]), Bp, floor="1e-25") > mp.mpf("2e-18")
                or abs(_mp(cyc["D_MeV3"])) > mp.mpf("2e-18") * max(abs(Bp), mp.mpf(1))):
                return False
            if _rel(_mp(mix["energy_MeV4"]), u, floor="1e-25") > mp.mpf("2e-18"):
                return False
            if _rel(_mp(sep["entropy_MeV3"]) / _mp(sep["B_MeV3"]), _mp(end["reversible_separation"]["entropy_target_per_B"]), floor="1e-25") > mp.mpf("2e-18"):
                return False
            if _rel(_mp(cyc["energy_MeV4"]), _mp(sep["energy_MeV4"]), floor="1e-25") > mp.mpf("2e-18"):
                return False
            if _mp(end["closed_mixing"]["energy_residual_relative"]) > mp.mpf("2e-18") or _mp(end["reversible_separation"]["entropy_residual_relative"]) > mp.mpf("2e-18") or _mp(end["composed_endpoint"]["energy_residual_relative"]) > mp.mpf("2e-18"):
                return False
            if (_rel(_mp(mixed["delta_sigma_mix"]), (_mp(mix["entropy_MeV3"]) - ss) / Bp, floor="1e-25") > mp.mpf("2e-18")
                or _rel(_mp(composed["delta_sigma_cycle"]), (_mp(cyc["entropy_MeV3"]) - _mp(sep["entropy_MeV3"])) / Bp, floor="1e-25") > mp.mpf("2e-18")
                or _rel(_mp(reversible["wrev_MeV"]), (_mp(sep["energy_MeV4"]) - _mp(baseline["energy_MeV4"])) / Bp, floor="1e-25") > mp.mpf("2e-18")
                or _rel(_mp(composed["x_hot_MeV"]), _mp(reversible["wrev_MeV"]) - Tbase * _mp(composed["delta_sigma_cycle"]), floor="1e-25") > mp.mpf("2e-18")):
                return False
            # Isothermal separation ledger is independently recomputed.
            iso = row["isothermal_separation"]
            du, ds, q, wi = (_mp(iso[k]) for k in ("delta_u_per_B_MeV", "delta_sigma_sep", "heat_Qprep_MeV", "wiso_MeV"))
            if _rel(q, T * ds, floor="1e-25") > mp.mpf("2e-18") or _rel(wi, du - q, floor="1e-25") > mp.mpf("2e-18") or _mp(iso["work_identity_residual"]) > mp.mpf("2e-18"):
                return False
            if row["delta"] == "0.2" and not bool(iso.get("quadrature_omitted")):
                quadrature = iso.get("muD_integral", [])
                if {int(x["points"]) for x in quadrature} != {8, 12} or any(_mp(x["relative_to_wiso"]) > mp.mpf("2e-8") for x in quadrature):
                    return False
            if row["delta"] == "0" and (_mp(end["closed_mixing"]["delta_sigma_mix"]) != 0 or _mp(end["reversible_separation"]["wrev_MeV"]) != 0):
                return False
        # Recompute each coefficient identity from its numeric fields.
        for key, cp in coeffs.items():
            sf, su, cv = (_mp(cp[k]) for k in ("S_F_MeV", "S_U_MeV", "cv_dimensionless"))
            tm, ts, kt, lf, am, ap = (_mp(cp[k]) for k in ("Theta_mix_MeV", "Theta_sep_MeV", "K_T_MeV", "L_F_MeV", "A_mu", "A_P"))
            if cv <= 0 or kt <= 0 or _rel(kt * (am - ap), 9 * sf, floor="1e-30") > mp.mpf("1e-10") or _rel(9 * cv * (tm + ts), 9 * sf, floor="1e-30") > mp.mpf("1e-10"):
                return False
        # Contact shift is an equation, not an exported gate.
        for item in result["controls"]["contact_identity"]["rows"]:
            if _rel(_mp(item["S_F_shift"]), _mp(item["expected_shift"]), floor="1e-30") > mp.mpf("1e-10") or _rel(_mp(item["S_U_shift"]), _mp(item["expected_shift"]), floor="1e-30") > mp.mpf("1e-10") or _abs_scaled(item["dT_SF_shift"], item["expected_shift"], floor="1e-30") > mp.mpf("1e-8") or not bool(item["isentropic_derivative_unchanged"]):
                return False
        # Recompute the analytic ideal-mixture null row by row.
        for item in result["controls"]["classical_null"]["rows"]:
            d = _mp(item["delta"])
            phi = ((1 + d) * mp.log(1 + d) + (1 - d) * mp.log(1 - d)) / 2 if d else mp.mpf(0)
            expected = {"Phi": phi, "S_F": mp.mpf("35"), "S_U": mp.mpf(0), "cv": mp.mpf("1.5"), "Tmix_minus_T0": mp.mpf(0), "wiso_over_T": phi, "delta_sigma_mix": phi, "Tsep_over_T0": mp.exp(phi / mp.mpf("1.5")), "A_mu": mp.mpf("0.5"), "A_P": mp.mpf(0)}
            for k, value in expected.items():
                if _rel(item[k], value, floor="1e-30") > mp.mpf("1e-25"):
                    return False
        # Primary isobaric rows carry a direct pressure equality and B/V map.
        for item in result.get("isobaric", []):
            if item.get("status") == "BASELINE":
                continue
            ref = references.get((str(item["anchor"]), item["model"]))
            if ref is None:
                return False
            st = item["state"]
            p0 = _mp(ref["reference_pressure_MeV4"])
            if _rel(_mp(st["pressure_MeV4"]), p0, floor="1e-25") > mp.mpf("2e-18"):
                return False
            br = _mp(item["B_ratio"]); vr = _mp(item["V_ratio"])
            if br <= 0 or _rel(br * _mp(ref["B0_MeV3"]), _mp(st["B_MeV3"]), floor="1e-25") > mp.mpf("2e-18") or _rel(br * vr, 1, floor="1e-30") > mp.mpf("2e-18"):
                return False
        return True
    except (KeyError, TypeError, ValueError, ArithmeticError, ZeroDivisionError):
        return False


def validate_result(result: Any) -> bool:
    """Fail-closed live validator; booleans/checksums are never sole authority."""
    if mp.mp.dps < 65:
        old_dps = mp.mp.dps
        mp.mp.dps = 80
        try:
            return validate_result(result)
        finally:
            mp.mp.dps = old_dps
    try:
        if not isinstance(result, Mapping) or not _finite_tree(result):
            return False
        if result.get("schema_version") != SCHEMA_VERSION or result.get("status") != STATUS_PASS or result.get("all_controls_pass") is not True:
            return False
        if result.get("evidence_weight") != EVIDENCE_WEIGHT or result.get("physical_inputs_fixed") is not True or result.get("no_saved_results_as_inputs") is not True or result.get("fit_used") is not False:
            return False
        rows = result.get("rows")
        if not isinstance(rows, list) or len(rows) != EXPECTED_ROW_COUNT:
            return False
        actual = {(str(r.get("anchor")), r.get("model"), r.get("delta"), r.get("contact")) for r in rows if isinstance(r, Mapping)}
        if actual != _expected_row_keys() or len({r.get("row_id") for r in rows}) != EXPECTED_ROW_COUNT:
            return False
        if not _validate_numeric_states(result):
            return False
        controls = result.get("controls", {})
        required = ("reference_coverage", "composition_coverage", "row_coverage", "root_and_energy_gates", "coefficient_identity", "contact_identity", "classical_null", "isobaric_coverage")
        if any(controls.get(name, {}).get("all_pass") is not True for name in required):
            return False
        # Recheck ideal null numerically; a mutated true flag must not pass.
        for c in controls["classical_null"]["rows"]:
            d = _mp(c["delta"])
            phi = ((1 + d) * mp.log(1 + d) + (1 - d) * mp.log(1 - d)) / 2 if d else mp.mpf(0)
            if _rel(c["Phi"], phi, floor="1e-30") > mp.mpf("1e-25") or _rel(c["wiso_over_T"], phi, floor="1e-30") > mp.mpf("1e-25") or _rel(c["delta_sigma_mix"], phi, floor="1e-30") > mp.mpf("1e-25"):
                return False
        if result.get("precision_controls", {}).get("all_pass") is not True:
            return False
        expected_hashes = {"producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "thermal_observable_bridge": hashlib.sha256(Path(thermal.__file__).resolve().read_bytes()).hexdigest(), "thermal_composition_hessian": hashlib.sha256(Path(composition.__file__).resolve().read_bytes()).hexdigest(), "nonlinear_calibration": hashlib.sha256(Path(nonlinear.__file__).resolve().read_bytes()).hexdigest()}
        if result.get("source_sha256") != expected_hashes:
            return False
        digest = result.get("integrity_sha256")
        copy_result = dict(result); copy_result.pop("integrity_sha256", None)
        if not isinstance(digest, str) or hashlib.sha256(_canonical_payload(copy_result).encode()).hexdigest() != digest:
            return False
        return True
    except Exception:
        return False


def build_result() -> dict[str, Any]:
    return _make_result()


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "nvg_closed_mixing_bridge_results.json")
    parser.add_argument("--validate", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.validate is not None:
        loaded = json.loads(args.validate.read_text(encoding="utf-8"))
        ok = validate_result(loaded)
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    try:
        result = build_result()
        _write_json(result, args.output)
        print(result["status"])
        print(f"rows={len(result.get('rows', []))} isobaric={len(result.get('isobaric', []))} output={args.output}")
        return 0 if result["status"] == STATUS_PASS else 1
    except Exception as exc:
        print(f"{STATUS_FAIL}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
