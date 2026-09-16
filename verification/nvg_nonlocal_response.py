#!/usr/bin/env python3
"""Live finite-q Dirac/Hartree static response.

The zero-density determinant is subtracted from the finite-density determinant
at the same fields.  This explicit ``SUBTRACTED_MEDIUM_DIRAC_STATIC``
scheme retains the occupied-to-negative-energy term required by the matched
no-sea thermodynamic Hessian; it is not dynamic or full vacuum RPA.  The
producer prints strict JSON and never writes a result cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import mpmath as mp

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import nvg_foundation_identifiability_audit as foundation  # noqa: E402
import nvg_spatial_identifiability_audit as spatial  # noqa: E402
import source_complete_scaling_saturation_audit as upstream  # noqa: E402

SOURCE_PATH = Path(__file__).resolve()
UPSTREAM_PATH = HERE / "source_complete_scaling_saturation_audit.py"
FOUNDATION_PATH = HERE / "nvg_foundation_identifiability_audit.py"
CONTRACT_PATH = HERE / "contracts" / "nonlocal_response.md"
SCHEMA_VERSION, SCHEME, EVIDENCE_WEIGHT = 1, "SUBTRACTED_MEDIUM_DIRAC_STATIC", 0.0
STATUS = "LIVE_SUBTRACTED_MEDIUM_DIRAC_STATIC_NONLOCAL_RESPONSE_ZERO_EVIDENCE_NOT_EMPIRICAL"
PRIMARY_DPS, CONTROL_DPS, PRECISION_LIMIT = 40, 55, mp.mpf("1e-9")
Q_RATIO_GRID = ("0", "0.1", "0.5", "1", "1.5", "2", "2.5")
PHYSICAL_Q_GRID_MEV, SCALES, W8_TARGETS = ("100", "200"), ("1", "1.3"), ("0.90", "0.93")
EXPECTED_ROW_COUNT = 3 * len(SCALES) * (len(Q_RATIO_GRID) + len(PHYSICAL_Q_GRID_MEV))

class NonlocalResponseError(ValueError):
    """Malformed input or unresolved numerical result."""

def _mp(value: Any, name: str) -> mp.mpf:
    if isinstance(value, bool):
        raise NonlocalResponseError(f"{name} must be a finite real, not bool")
    try:
        x = value if isinstance(value, mp.mpf) else mp.mpf(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise NonlocalResponseError(f"{name} must be a finite real") from exc
    if not mp.isfinite(x):
        raise NonlocalResponseError(f"{name} must be finite")
    return x

def _gt(value: Any, name: str) -> mp.mpf:
    x = _mp(value, name)
    if x <= 0:
        raise NonlocalResponseError(f"{name} must be positive")
    return x

def _ge(value: Any, name: str) -> mp.mpf:
    x = _mp(value, name)
    if x < 0:
        raise NonlocalResponseError(f"{name} must be nonnegative")
    return x

def _deg(value: Any) -> int:
    x = _gt(value, "d")
    if x != mp.floor(x) or int(x) % 2:
        raise NonlocalResponseError("d must be a positive even integer")
    return int(x)

def _finite(x: Any) -> bool:
    try:
        return bool(mp.isfinite(x if isinstance(x, mp.mpf) else mp.mpf(str(x))))
    except (TypeError, ValueError, OverflowError):
        return False

def _num(x: Any, digits: int = 16) -> str:
    x = _mp(x, "derived number")
    return mp.nstr(x, digits)

def _rel(a: Any, b: Any) -> mp.mpf:
    a, b = _mp(a, "left"), _mp(b, "right")
    return abs(a - b) / max(abs(a), abs(b), mp.mpf(1))

def _show(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, mp.matrix):
        return [[_show(x[i, j]) for j in range(x.cols)] for i in range(x.rows)]
    if isinstance(x, mp.mpf):
        return _num(x)
    if isinstance(x, Mapping):
        return {str(k): _show(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_show(v) for v in x]
    if isinstance(x, float):
        if not math.isfinite(x):
            raise NonlocalResponseError("nonfinite floating output")
        return x
    return x

def _log_ratio(p: mp.mpf, q: mp.mpf) -> mp.mpf:
    if p == 0:
        return mp.mpf(0)
    r = 2 * p / q
    if r == 1:
        return mp.inf
    z = r if r < 1 else 1 / r
    return mp.log1p(z) - mp.log1p(-z) if z else mp.mpf(0)

def _log_integral(q: mp.mpf, kf: mp.mpf) -> mp.mpf:
    r = 2 * kf / q
    if abs(r) < mp.mpf("1e-5"):
        total = mp.mpf(0)
        for n in range(201):
            term = 2 * r ** (2 * n + 2) / ((2 * n + 1) * (2 * n + 2))
            total += term
            if n and abs(term) <= max(abs(total), mp.mpf(1)) * mp.eps * 4:
                break
        return q * total / 2
    second = 0 if r == 1 else (1 - r) * mp.log(abs(1 - r))
    return q * ((1 + r) * mp.log1p(r) + second) / 2

def _log_weight(weight, q: mp.mpf, kf: mp.mpf) -> mp.mpf:
    """PV integral with the singular coefficient subtracted and restored."""
    p0 = q / 2
    if 0 < p0 <= kf:
        w0 = weight(p0)
        def regular(p):
            return mp.mpf(0) if p == p0 else (weight(p) - w0) * _log_ratio(p, q)
        residual = mp.quad(regular, [0, kf] if p0 == kf else [0, p0, kf])
        return residual + w0 * _log_integral(q, kf)
    return mp.quad(lambda p: weight(p) * _log_ratio(p, q), [0, kf])

def _q0(m: mp.mpf, k: mp.mpf, d: int) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    e = mp.sqrt(m * m + k * k)
    f = mp.mpf(d) / (2 * mp.pi ** 2)
    n0, mix = f * k * e, m / e * f * k * e
    I = f * mp.quad(lambda p: p ** 4 / (p * p + m * m) ** mp.mpf("1.5"), [0, k])
    out = (n0, mix, (m / e) ** 2 * n0 - I)
    if not all(_finite(x) for x in out):
        raise NonlocalResponseError("nonfinite q=0 polarization")
    return out

def _kernel(m: mp.mpf, k: mp.mpf, q: mp.mpf, d: int) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    if q == 0:
        return _q0(m, k, d)
    f = mp.mpf(d) / (2 * mp.pi ** 2)
    E = lambda p: mp.sqrt(p * p + m * m)
    vv0 = lambda p: p * p / E(p)
    vvL = lambda p: p * (E(p) ** 2 - q * q / 4) / (E(p) * q)
    ss0 = lambda p: -p * p / E(p)
    ssL = lambda p: p * (m * m + q * q / 4) / (E(p) * q)
    vsL = lambda p: m * p / q
    out = (f * (mp.quad(vv0, [0, k]) + _log_weight(vvL, q, k)),
           f * _log_weight(vsL, q, k),
           f * (mp.quad(ss0, [0, k]) + _log_weight(ssL, q, k)))
    if not all(_finite(x) for x in out):
        raise NonlocalResponseError("nonfinite finite-q polarization")
    return out

def polarization_kernel(mass: Any, kf: Any, q: Any, d: Any = 4, *, dps: int | None = None):
    """Return ``(Pi_vv, Pi_vs, Pi_ss)`` in MeV²."""
    m, k, q, d = _gt(mass, "M"), _gt(kf, "kF"), _ge(q, "q"), _deg(d)
    work = max(mp.mp.dps, PRIMARY_DPS) if dps is None else dps
    if not isinstance(work, int) or not 40 <= work <= 200:
        raise NonlocalResponseError("dps must be an integer in [40,200]")
    with mp.workdps(work):
        return _kernel(m, k, q, d)

finite_q_polarization = dirac_polarization = radial_polarization = polarization = polarization_kernel
compute_polarization = dirac_static_kernel = polarization_kernel

def lindhard_shape(x: Any) -> mp.mpf:
    x = _ge(x, "x")
    with mp.workdps(max(mp.mp.dps, PRIMARY_DPS)):
        if x == 0:
            return mp.mpf(1)
        if x == 1:
            return mp.mpf("0.5")
        # The same closed form has a convergent expansion on either side of
        # x=1; using it at extreme x avoids cancellation to 0.5.
        if x < mp.mpf("1e-4") or x > mp.mpf("1e4"):
            r = x if x < 1 else 1 / x
            sign = -1 if x < 1 else 1
            return mp.mpf(1) + sign * sum(
                r ** (2 * n) / ((2 * n - 1) * (2 * n + 1)) for n in range(1, 80)
            ) if x < 1 else sum(
                r ** (2 * n) / ((2 * n - 1) * (2 * n + 1)) for n in range(1, 80)
            )
        z = x if x < 1 else 1 / x
        log_ratio = mp.log1p(z) - mp.log1p(-z)
        return mp.mpf("0.5") + (1 - x * x) / (4 * x) * log_ratio

nonrelativistic_lindhard_shape = lindhard_shape
thermodynamic_kernel = lambda mass, kf, d=4: polarization_kernel(mass, kf, 0, d)

def _lindhard_controls():
    controls = []
    for label in ("0.25", "1", "1.25"):
        x = mp.mpf(label)
        if x == 1:
            expected = mp.mpf("0.5")
        else:
            z = x if x < 1 else 1 / x
            expected = mp.mpf("0.5") + (1 - x * x) / (4 * x) * (mp.log1p(z) - mp.log1p(-z))
        value = lindhard_shape(x)
        controls.append({"x": x, "value": value, "expected": expected,
                         "relative_error": _rel(value, expected), "pass": _rel(value, expected) <= mp.mpf("1e-12")})
    small_x = mp.mpf("1e-20")
    small_value = lindhard_shape(small_x)
    controls.append({"x": small_x, "value": small_value, "expected_limit": mp.mpf(1),
                     "relative_error": _rel(small_value, 1), "pass": _rel(small_value, 1) <= mp.mpf("1e-30")})
    return controls

def response_from_hessian(coefficients: Mapping[str, Any], Z: Any, q: Any) -> dict[str, Any]:
    """Reuse the maintained constrained 3x3 Hessian/direct solve."""
    try:
        return spatial.response_from_hessian(coefficients, Z, q)
    except (ValueError, ArithmeticError, spatial.SpatialIdentifiabilityError) as exc:
        raise NonlocalResponseError(str(exc)) from exc

static_response = response_from_hessian
ScaledNonlocalModel = spatial.ScaledResponseModel

def _guard(base):
    return spatial._assert_baseline_contract(base)

def _backgrounds(base):
    out = [{"background_id": "Q4:n_over_n0=1", "family": "Q4", "density_ratio": "1",
            "target_y": None, "model": base, "state": base.equilibrium(base.n0), "inverse_design": None}]
    for target in W8_TARGETS:
        model, _, jet = upstream.inverse_potential_jet(target, "240", "0.16", "-16")
        out.append({"background_id": f"W8:y_star={target}", "family": "W8", "density_ratio": "1",
                    "target_y": target, "model": model, "state": model.state(model.n0, mp.mpf(target)),
                    "inverse_design": jet})
    for b in out:
        s, f = b["state"], b["model"].fermi(b["state"]["n"], b["state"]["y"])
        if s["y"] <= 0 or s["C_y"] <= 0 or not _finite(s["y"]):
            raise ArithmeticError(f"invalid background {b['background_id']}")
        b.update(mass=f["m"], kF=f["k"])
    return out

def _coeff(model, state, scale, pi):
    vv, vs, ss = map(lambda x: _mp(x, "polarization"), pi)
    if vv == 0:
        raise NonlocalResponseError("Pi_vv=0 is a singular density kernel")
    n, y = state["n"], state["y"]
    A0 = model.gomega * n / (model.momega ** 2 * y ** 2)
    c = {"a": 1 / vv, "b": model.MN * vs / vv,
         "dferm": model.MN ** 2 * (vs ** 2 / vv - ss),
         "g": model.gomega, "h": -2 * model.momega ** 2 * y * A0,
         "t0": model.momega ** 2 * y ** 2, "Z": (scale * model.W0) ** 2,
         "Pi_vv": vv, "Pi_vs": vs, "Pi_ss": ss, "A0": A0}
    c["d0"] = c["dferm"] + state["Uyy"] - model.momega ** 2 * A0 ** 2
    if not all(_finite(x) for x in c.values()):
        raise NonlocalResponseError("nonfinite Legendre coefficients")
    return c

def finite_q_coefficients(model, state, q, scale="1"):
    q, scale = _ge(q, "q"), _gt(scale, "scale")
    f = model.fermi(state["n"], state["y"])
    with mp.workdps(max(mp.mp.dps, PRIMARY_DPS)):
        return _coeff(model, state, scale, _kernel(f["m"], f["k"], q, _deg(model.d)))

fermion_coefficients = finite_q_coefficients

def _homogeneous_fermion_hessian(model, state):
    """Independent fixed-(n,y) derivatives of the live BulkModel Fermi term."""
    n, y = state["n"], state["y"]
    energy = lambda nn, yy: model.fermi(nn, yy)["energy"]
    return {
        "a": mp.diff(lambda nn: energy(nn, y), n, 2),
        "b": mp.diff(lambda nn: mp.diff(lambda yy: energy(nn, yy), y), n),
        "dferm": mp.diff(lambda yy: energy(n, yy), y, 2),
    }

def _mixed_quad(m, k, q, d):
    if q == 0:
        return _q0(m, k, d)[1]
    x = q / (2 * k)
    def lu(u):
        if u == 0:
            return mp.mpf(0)
        if u == x:
            return mp.inf
        z = u / x if u < x else x / u
        return mp.log1p(z) - mp.log1p(-z)
    if x == 1:
        I = mp.quad(lambda t: 0 if t == 0 else (1 - t*t) * lu(1 - t*t) * 2*t, [0, 1])
    elif 0 < x < 1:
        left = lambda t: 0 if t == 0 or x*(1-t*t) == x else (x*(1-t*t)) * lu(x*(1-t*t)) * 2*x*t
        right = lambda t: 0 if t == 0 or x+(1-x)*t*t == x else (x+(1-x)*t*t) * lu(x+(1-x)*t*t) * 2*(1-x)*t
        I = mp.quad(left, [0, 1]) + mp.quad(right, [0, 1])
    else:
        I = mp.quad(lambda u: u * lu(u), [0, 1])
    result = mp.mpf(d) * m * k * I / (4 * mp.pi ** 2 * x)
    if not _finite(result):
        raise NonlocalResponseError("independent mixed quadrature failed")
    return result

def _nr_controls():
    m, k, d = mp.mpf("10000"), mp.mpf(1), 4
    N = mp.mpf(d) * k * mp.sqrt(m*m+k*k) / (2 * mp.pi**2)
    out = []
    for text in ("0.5", "1", "2", "2.5"):
        ratio, expected = mp.mpf(text), N * lindhard_shape(mp.mpf(text) / 2)
        pi = _kernel(m, k, ratio * k, d)
        errors = {n: _rel(pi[i], expected) for i, n in enumerate(("Pi_vv", "Pi_vs", "Pi_ss"))}
        out.append({"M_over_kF": mp.mpf("1e4"), "q_over_kF": ratio, "Pi": pi,
                    "expected_N0_F": expected, "relative_errors": errors,
                    "pass": max(errors.values()) <= mp.mpf("1e-6")})
    return out

def _specs(k):
    return ([{"kind": "q_over_kF", "label": x, "q": mp.mpf(x) * k, "q_over_kF": mp.mpf(x)} for x in Q_RATIO_GRID]
            + [{"kind": "physical_q", "label": x, "q": mp.mpf(x), "q_over_kF": mp.mpf(x) / k} for x in PHYSICAL_Q_GRID_MEV])

def _snapshot(dps):
    with mp.workdps(dps):
        base, backgrounds = upstream.BulkModel(), None
        guard = _guard(base)
        backgrounds = _backgrounds(base)
        rows, maps, q0c, cont, mixed, negative, probes = [], {}, [], [], [], [], {}
        def put(name, value):
            probes[name] = value
        for b in backgrounds:
            model, state, k, m, d = b["model"], b["state"], b["kF"], b["mass"], _deg(b["model"].d)
            specs, pis = _specs(k), {}
            for sp in specs:
                pis[(sp["kind"], sp["label"])] = _kernel(m, k, sp["q"], d)
            pi0 = pis[("q_over_kF", "0")]
            maps[b["background_id"]] = []
            ref0 = _coeff(model, state, mp.mpf(1), pi0)
            for scale_text in SCALES:
                scale, candidate = mp.mpf(scale_text), ScaledNonlocalModel(scale_text, model)
                cs = candidate.state(state["n"], state["y"])
                fields = ("energy_total", "pressure_total", "mu", "residual", "C_y", "B_y", "D", "K", "cs2")
                sd = {x: _rel(cs[x], state[x]) for x in fields}
                # The provided candidate model is already rescaled; its W0
                # is the sole owner of the canonical gradient normalization.
                c0 = _coeff(candidate, cs, mp.mpf(1), pi0)
                cd = {x: _rel(c0[x], ref0[x]) for x in ("a", "b", "dferm", "d0", "g", "h", "t0")}
                maps[b["background_id"]].append({"scale": scale_text, "state_diff": sd, "coeff_diff": cd, "Z": c0["Z"]})
                for sp in specs:
                    pi = pis[(sp["kind"], sp["label"])]
                    # candidate already owns W0=s*W0_original; use a unit
                    # relative factor here so the canonical gradient scales
                    # once, as candidate.W0**2 rather than s^4.
                    c, tfc = _coeff(candidate, cs, mp.mpf(1), pi), _coeff(candidate, cs, mp.mpf(1), pi0)
                    actual, tf = response_from_hessian(c, c["Z"], sp["q"]), response_from_hessian(tfc, tfc["Z"], sp["q"])
                    rel = None if actual["chi"] is None or tf["chi"] in (None, 0) else actual["chi"] / tf["chi"] - 1
                    row = {"row_id": f"{b['background_id']}:scale={scale_text}:q={sp['kind']}={sp['label']}",
                           "background_id": b["background_id"], "family": b["family"], "density_ratio": b["density_ratio"],
                           "target_y": b["target_y"], "scale": scale_text, **sp, "Pi": pi, "coefficients": c,
                           "response": actual, "tf_response": tf, "relative_difference": rel}
                    rows.append(row)
                    for name, value in zip(("vv", "vs", "ss"), pi): put(f"{row['row_id']}:Pi:{name}", value)
                    for tag, rr in (("response", actual), ("tf", tf)):
                        for name in ("D", "B", "C", "determinant", "S", "chi"):
                            if rr[name] is not None: put(f"{row['row_id']}:{tag}:{name}", rr[name])
            exact = _q0(m, k, d)
            independent = _homogeneous_fermion_hessian(model, state)
            expected = {"a": ref0["a"], "b": ref0["b"], "dferm": ref0["dferm"]}
            q0resp = response_from_hessian(ref0, ref0["Z"], 0)
            qe = {key: _rel(independent[key], expected[key]) for key in expected}
            qe["S"] = _rel(q0resp["S"], state["mu_prime"])
            qe["chi"] = _rel(q0resp["chi"], 1 / state["mu_prime"])
            q0c.append({"background_id": b["background_id"], "Pi_q0": pi0, "Pi_thermodynamic": exact,
                        "independent_bulk_fermion_hessian": independent, "expected_legendre_coefficients": expected,
                        "S_from_hessian": q0resp["S"], "homogeneous_dmu_dn": state["mu_prime"],
                        "relative_errors": qe, "pass": max(qe.values()) <= mp.mpf("1e-10")})
            for key, value in independent.items():
                put(f"{b['background_id']}:bulk:{key}", value)
            put(f"{b['background_id']}:bulk:S", q0resp["S"])
            small = _kernel(m, k, mp.mpf("1e-4") * k, d)
            ce = {n: _rel(small[i], exact[i]) for i, n in enumerate(("Pi_vv", "Pi_vs", "Pi_ss"))}
            cont.append({"background_id": b["background_id"], "q_over_kF": mp.mpf("1e-4"), "relative_errors": ce,
                         "pass": max(ce.values()) <= mp.mpf("1e-6")})
            qmix, mixpi = mp.mpf("0.5") * k, _kernel(m, k, mp.mpf("0.5") * k, d)
            iq, closed = _mixed_quad(m, k, qmix, d), mp.mpf(d) * m * k / (2 * mp.pi ** 2) * lindhard_shape(mp.mpf("0.25"))
            mixed.append({"background_id": b["background_id"], "q_over_kF": mp.mpf("0.5"), "radial_Pi_vs": mixpi[1],
                          "independent_quadrature_Pi_vs": iq, "closed_Lindhard_Pi_vs": closed,
                          "relative_error": _rel(mixpi[1], iq), "pass": _rel(mixpi[1], iq) <= mp.mpf("1e-9")})
            vv, vs, ss = exact
            ssph = (m / mp.sqrt(m*m+k*k)) ** 2 * vv
            good, wrong = _coeff(model, state, mp.mpf(1), exact), _coeff(model, state, mp.mpf(1), (vv, vs, ssph))
            gr, wr = response_from_hessian(good, good["Z"], 0), response_from_hessian(wrong, wrong["Z"], 0)
            err = None if gr["chi"] is None or wr["chi"] is None else _rel(gr["chi"], wr["chi"])
            negative.append({"background_id": b["background_id"], "matched_Pi_ss_MeV2": ss, "particle_hole_only_Pi_ss_MeV2": ssph,
                             "omitted_I_MeV2": ssph - ss, "matched_dferm_MeV4": good["dferm"],
                             "particle_hole_only_dferm_MeV4": wrong["dferm"], "wrong_response_relative_error": err,
                             "detected_and_rejected": ssph != ss and (err is None or err > mp.mpf("1e-9"))})
        if len(rows) != EXPECTED_ROW_COUNT:
            raise ArithmeticError(f"row count {len(rows)} != {EXPECTED_ROW_COUNT}")
        nr, lindhard = _nr_controls(), _lindhard_controls()
        for item in nr:
            for n, value in zip(("vv", "vs", "ss"), item["Pi"]): put(f"nr:{item['q_over_kF']}:{n}", value)
        for item in lindhard:
            put(f"lindhard:{item['x']}", item["value"])
        return locals()

def _response_json(r):
    return {"D_MeVminus2": _num(r["D"]), "B_MeV": _num(r["B"]), "C_MeV4": _num(r["C"]),
            "S_MeVminus2": None if r["S"] is None else _num(r["S"]),
            "chi_MeV2": None if r["chi"] is None else _num(r["chi"]), "L_MeV2": _num(r["L"]),
            "direct_solution": _show(r["direct_solution"]), "direct_status": r["direct_status"],
            "direct_vs_schur_relative_error": None if r["direct_relative_error"] is None else _num(r["direct_relative_error"]),
            "stable": r["stable"], "status": r["status"]}

def _snapshot_json(s):
    backs = []
    for b in s["backgrounds"]:
        st, model = b["state"], b["model"]
        backs.append({"background_id": b["background_id"], "family": b["family"], "density_ratio": b["density_ratio"],
                      "target_y": b["target_y"], "state": _show({"y": st["y"], "W_MeV": model.W0 * st["y"], "mass_MeV": st["m"], "kF_MeV": st["k"],
                      "binding_MeV": st["energy_total"] / st["n"] - model.MN, "mu_MeV": st["mu"], "pressure_MeV_fm3": st["pressure_total"] / model.hbarc**3,
                      "C_y_MeV4": st["C_y"], "K_MeV": st["K"], "cs2": st["cs2"]}),
                      "inverse_design": None if b["inverse_design"] is None else {"target_y": b["target_y"], "training_targets": {"n0_fm3": "0.16", "binding_MeV": "-16", "K_MeV": "240", "pressure_MeV_fm3": "0"}, "not_adopted_physical_model": True}})
    maps = {bid: [{"scale": x["scale"], "state_relative_differences": _show(x["state_diff"]), "homogeneous_state_invariant": max(x["state_diff"].values()) <= mp.mpf("1e-9"),
                    "q0_coefficient_relative_differences": _show(x["coeff_diff"]), "homogeneous_q0_kernel_invariant": max(x["coeff_diff"].values()) <= mp.mpf("1e-9"), "gradient_coefficient_Z_MeV2": _num(x["Z"])} for x in xs] for bid, xs in s["maps"].items()}
    rows = []
    for x in s["rows"]:
        r, tf, pi = x["response"], x["tf_response"], x["Pi"]
        rows.append({"row_id": x["row_id"], "background_id": x["background_id"], "family": x["family"], "density_ratio": x["density_ratio"], "target_y": x["target_y"], "scale": x["scale"],
                     "q_kind": x["kind"], "q_label": x["label"], "q_MeV": _num(x["q"]), "q_over_kF": _num(x["q_over_kF"]),
                     "Pi_vv_MeV2": _num(pi[0]), "Pi_vs_MeV2": _num(pi[1]), "Pi_ss_MeV2": _num(pi[2]), "Pi_vv_vs_ss_MeV2": [_num(v) for v in pi],
                     "chi_nonlocal_MeV2": None if r["chi"] is None else _num(r["chi"]), "chi_TF_MeV2": None if tf["chi"] is None else _num(tf["chi"]),
                     "relative_difference": None if x["relative_difference"] is None else _num(x["relative_difference"]), "relative_difference_percent": None if x["relative_difference"] is None else _num(100*x["relative_difference"]),
                     "C_MeV4": _num(r["C"]), "S_MeVminus2": None if r["S"] is None else _num(r["S"]), "C_nonlocal_MeV4": _num(r["C"]), "S_nonlocal_MeVminus2": None if r["S"] is None else _num(r["S"]),
                     "C_TF_MeV4": _num(tf["C"]), "S_TF_MeVminus2": None if tf["S"] is None else _num(tf["S"]), "nonlocal_response": _response_json(r), "tf_response": _response_json(tf), "status": r["status"], "response_status": r["status"]})
    return {"dps": s["dps"], "inputs_and_scope": {"scheme": SCHEME, "natural_units": True, "q_over_kF_grid": list(Q_RATIO_GRID), "physical_q_MeV": list(PHYSICAL_Q_GRID_MEV), "scales": list(SCALES), "expected_row_count": EXPECTED_ROW_COUNT,
            "source_definition": "chi=delta_n/delta_mu_ext for source -mu_ext*n", "tf_reference": "q=0 irreducible fermion kernel with the same finite q in the boson propagator", "new_fitted_coefficient": False, "original_parameters_not_refit": True,
            "negative_energy_mixing_retained": True, "not_full_vacuum_RPA": True, "not_dynamic_or_spectral": True, "not_finite_nucleus_or_transport": True, "evidence_weight": EVIDENCE_WEIGHT},
            "baseline_guard": s["guard"], "backgrounds": backs, "scale_maps": maps, "rows": rows, "q0_controls": _show(s["q0c"]), "q0_thermodynamic_matches": _show(s["q0c"]), "continuity_controls": _show(s["cont"]), "independent_mixed_quadrature_controls": _show(s["mixed"]), "scalar_completeness_negative_controls": _show(s["negative"]), "nonrelativistic_controls": _show(s["nr"]), "lindhard_controls": _show(s["lindhard"]),
            "coverage": {"row_count": len(rows), "expected_row_count": EXPECTED_ROW_COUNT, "all_rows_present": len(rows) == EXPECTED_ROW_COUNT, "all_statuses_retained": True}}

def calculate(dps=PRIMARY_DPS):
    if not isinstance(dps, int) or not 40 <= dps <= 200:
        raise NonlocalResponseError("dps must be an integer in [40,200]")
    return _snapshot_json(_snapshot(dps))
def build_result():
    with mp.workdps(CONTROL_DPS + 10):
        a, b = _snapshot(PRIMARY_DPS), _snapshot(CONTROL_DPS)
        if set(a["probes"]) != set(b["probes"]):
            raise ArithmeticError("precision probe key set changed")
        errors = {k: _rel(a["probes"][k], b["probes"][k]) for k in a["probes"]}
        maximum = max(errors.values(), default=mp.mpf(0))
        if maximum > PRECISION_LIMIT:
            raise ArithmeticError(f"40/55 precision mismatch: {maximum}")
        precision = {"dps": [PRIMARY_DPS, CONTROL_DPS], "probes_compared": len(errors), "max_relative_error": maximum, "declared_relative_tolerance": PRECISION_LIMIT, "pass": True}
        return {"schema_version": SCHEMA_VERSION, "status": STATUS, "scheme": SCHEME, "evidence_weight": EVIDENCE_WEIGHT,
                "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(), "upstream_source_sha256": hashlib.sha256(UPSTREAM_PATH.read_bytes()).hexdigest(), "foundation_source_sha256": hashlib.sha256(FOUNDATION_PATH.read_bytes()).hexdigest(), "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(), "precision_controls": _show(precision), **_snapshot_json(a),
                "scope": {"same_dirac_hartree_action": True, "subtracted_zero_density_determinant": True, "negative_energy_intermediates_retained": True, "canonical_scalar_gradient_retained": True, "gauss_vector_constraint_retained": True, "original_parameters_not_refit": True, "not_full_renormalized_vacuum_RPA": True, "not_dynamic_response": True, "not_empirical_validation": True, "tail_jet_and_vacuum_counterterms_unresolved": True}}

def validate_result(result: Any) -> bool:
    try:
        return isinstance(result, dict) and result == build_result()
    except (ArithmeticError, OSError, TypeError, ValueError, NonlocalResponseError, KeyError):
        return False

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dps", type=int, choices=range(40, 201), default=None)
    args = p.parse_args(argv)
    print(json.dumps(calculate(args.dps) if args.dps is not None else build_result(), indent=2, ensure_ascii=False, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
