#!/usr/bin/env python3
"""Exact Hayward geometry-to-stress audit, in c=G=1 units.

This is an inverse Einstein-equation fingerprint of a chosen metric. It does
not derive that metric from the accepted NVG action, solve collapse, supply
information transfer, or establish a flat FLRW bounce. build_result is pure;
the CLI prints JSON and writes its dedicated artifact only with --write.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.integrate import quad
import sympy as sp

RESULT_PATH = Path(__file__).with_name("regular_core_geometry_audit_results.json")
EVIDENCE_STATUS = "GEOMETRY_IDENTITY_AUDIT_NO_MATTER_REALIZATION"


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _parameters(radius, mass, length):
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    return radius, _positive(mass, "mass"), _positive(length, "length")


def mass_function(radius, mass, length):
    r, M, ell = _parameters(radius, mass, length)
    q = r**3 / (2 * M * ell**2)
    return M * q / (1 + q)


def metric_function(radius, mass, length):
    r, M, ell = _parameters(radius, mass, length)
    q = r**3 / (2 * M * ell**2)
    return 1 - r**2 / (ell**2 * (1 + q))


def stress_tensor(radius, mass, length):
    """Principal stress eigenvalues; rho is measured in an orthonormal frame.

    In a trapped region t and r exchange causal roles. Their equal mixed
    eigenvalues (-rho,-rho) preserve this density/pressure interpretation.
    """
    r, M, ell = _parameters(radius, mass, length)
    q = r**3 / (2 * M * ell**2)
    rho = 3 / (8 * math.pi * ell**2 * (1 + q)**2)
    pr = -rho
    pt = rho * (2 * q - 1) / (1 + q)
    return {"q": q, "rho": rho, "p_r": pr, "p_t": pt,
            "anisotropy": 3 * rho * q / (1 + q)}


def conservation_residual(radius, mass, length):
    """p_r' + (rho+p_r) f'/(2f) + 2(p_r-p_t)/r.

    The gravitational term vanishes identically before evaluation, including
    at a horizon. The r=0 expression denotes its continuous zero limit.
    """
    r, M, ell = _parameters(radius, mass, length)
    if r == 0:
        return 0.0
    state = stress_tensor(r, M, ell)
    rho0 = 3 / (8 * math.pi * ell**2)
    pressure_derivative = 6 * rho0 * r**2 / (2 * M * ell**2 * (1 + state["q"])**3)
    return pressure_derivative + 2 * (state["p_r"] - state["p_t"]) / r


def curvature_invariants(radius, mass, length):
    """Ricci scalar, Ricci square, Kretschmann scalar, and Weyl square."""
    r, M, ell = _parameters(radius, mass, length)
    q = r**3 / (2 * M * ell**2)
    denominator = ell**4 * (1 + q)**6
    return {
        "ricci_scalar": 6 * (2 - q) / (ell**2 * (1 + q)**3),
        "ricci_squared": 18 * (5 * q**2 - 2 * q + 2) / denominator,
        "kretschmann": 12 * (q**4 - 4 * q**3 + 18 * q**2 - 2 * q + 2) / denominator,
        "weyl_squared": 12 * q**2 * (q - 2)**2 / denominator,
    }


def energy_conditions(radius, mass, length):
    state = stress_tensor(radius, mass, length)
    rho, pr, pt = (state[key] for key in ("rho", "p_r", "p_t"))
    return {
        "radial_nec": rho + pr,
        "tangential_nec": 3 * rho * state["q"] / (1 + state["q"]),
        "sec_sum": rho + pr + 2 * pt,
        "radial_dec_margin": rho - abs(pr),
        "tangential_dec_margin": rho - abs(pt),
    }


def schwarzschild_deviation(radius, mass, length):
    """Exact f_Hayward-f_Schwarzschild for r>0, without subtractive cancellation."""
    r, M, ell = _parameters(radius, mass, length)
    if r == 0:
        raise ValueError("Schwarzschild comparison requires radius > 0")
    return 4 * M**2 * ell**2 / (r * (r**3 + 2 * M * ell**2))


def horizon_mass(radius, length):
    """M(r_h)=r_h^3/[2(r_h^2-l^2)] on either positive-radius horizon branch."""
    r, ell = _positive(radius, "radius"), _positive(length, "length")
    if r <= ell:
        raise ValueError("positive-mass horizon requires radius > length")
    return r**3 / (2 * (r**2 - ell**2))


def horizon_temperature(radius, length):
    """Signed surface-gravity temperature; positive on the outer branch.

    Values use hbar=k_B=1 in addition to the geometry units c=G=1.
    """
    r, ell = _positive(radius, "radius"), _positive(length, "length")
    if r <= ell:
        raise ValueError("positive-mass horizon requires radius > length")
    rcrit = math.sqrt(3) * ell
    return ((r - rcrit) / r) * ((r + rcrit) / r) / (4 * math.pi * r)


def critical_scales(length):
    ell = _positive(length, "length")
    return {"r_crit": math.sqrt(3) * ell, "M_crit": 3 * math.sqrt(3) * ell / 4,
            "r_temperature_peak": 3 * ell, "M_temperature_peak": 27 * ell / 16,
            "M_peak_over_M_crit": 3 * math.sqrt(3) / 4,
            "T_max": 1 / (18 * math.pi * ell)}


def symbolic_identities():
    """Derive checks from m(r) and metric derivatives, not asserted pass flags."""
    r, M, ell = sp.symbols("r M ell", positive=True)
    q = r**3 / (2 * M * ell**2)
    m = M * r**3 / (r**3 + 2 * M * ell**2)
    f = 1 - 2 * m / r
    rho = 3 / (8 * sp.pi * ell**2 * (1 + q)**2)
    pr, pt = -rho, rho * (2 * q - 1) / (1 + q)
    R = -sp.diff(f, r, 2) - 4 * sp.diff(f, r) / r + 2 * (1 - f) / r**2
    Rt = -sp.diff(f, r, 2) / 2 - sp.diff(f, r) / r
    Ra = (1 - f - r * sp.diff(f, r)) / r**2
    R2 = 2 * Rt**2 + 2 * Ra**2
    K = sp.diff(f, r, 2)**2 + 4 * (sp.diff(f, r) / r)**2 + 4 * ((1 - f) / r**2)**2
    denominator = ell**4 * (1 + q)**6
    Mh = r**3 / (2 * (r**2 - ell**2))
    Th = (r**2 - 3 * ell**2) / (4 * sp.pi * r**3)
    expressions = {
        "density_from_mass_derivative": rho - sp.diff(m, r) / (4 * sp.pi * r**2),
        "radial_pressure_from_einstein_tensor": pr - (r * sp.diff(f, r) + f - 1) / (8 * sp.pi * r**2),
        "tangential_pressure_from_mass_derivative": pt + sp.diff(m, r, 2) / (8 * sp.pi * r),
        "stress_conservation": sp.diff(pr, r) + (rho + pr) * sp.diff(f, r) / (2 * f) + 2 * (pr - pt) / r,
        "ricci_scalar": R - 6 * (2 - q) / (ell**2 * (1 + q)**3),
        "ricci_squared": R2 - 18 * (5 * q**2 - 2 * q + 2) / denominator,
        "kretschmann": K - 12 * (q**4 - 4 * q**3 + 18 * q**2 - 2 * q + 2) / denominator,
        "weyl_squared": K - 2 * R2 + R**2 / 3 - 12 * q**2 * (q - 2)**2 / denominator,
        "schwarzschild_deviation": f - (1 - 2 * M / r) - 4 * M**2 * ell**2 / (r * (r**3 + 2 * M * ell**2)),
        "asymptotic_deviation_coefficient": sp.limit(r**4 * (f - 1 + 2 * M / r), r, sp.oo) - 4 * M**2 * ell**2,
        "horizon_mass_relation": f.subs(M, Mh),
        "horizon_temperature": (sp.diff(f, r) / (4 * sp.pi)).subs(M, Mh) - Th,
        "horizon_mass_derivative": sp.diff(Mh, r) - r**2 * (r**2 - 3 * ell**2) / (2 * (r**2 - ell**2)**2),
        "horizon_temperature_derivative": sp.diff(Th, r) - (9 * ell**2 - r**2) / (4 * sp.pi * r**4),
        "extremal_mass_stationary": sp.diff(Mh, r).subs(r, sp.sqrt(3) * ell),
        "extremal_mass": Mh.subs(r, sp.sqrt(3) * ell) - 3 * sp.sqrt(3) * ell / 4,
        "extremal_temperature": Th.subs(r, sp.sqrt(3) * ell),
        "extremal_double_root_curvature": sp.diff(f, r, 2).subs(
            {M: 3 * sp.sqrt(3) * ell / 4, r: sp.sqrt(3) * ell}) - 2 / (3 * ell**2),
        "temperature_peak_stationary": sp.diff(Th, r).subs(r, 3 * ell),
        "temperature_peak_mass": Mh.subs(r, 3 * ell) - sp.Rational(27, 16) * ell,
        "temperature_peak_ratio": Mh.subs(r, 3 * ell) / Mh.subs(r, sp.sqrt(3) * ell) - 3 * sp.sqrt(3) / 4,
        "temperature_maximum": Th.subs(r, 3 * ell) - 1 / (18 * sp.pi * ell),
        "horizon_q": q.subs(M, Mh) - (r**2 / ell**2 - 1),
        "central_ricci": sp.limit(R, r, 0) - 12 / ell**2,
        "central_ricci_squared": sp.limit(R2, r, 0) - 36 / ell**4,
        "central_kretschmann": sp.limit(K, r, 0) - 24 / ell**4,
    }
    residuals = {key: sp.simplify(value) for key, value in expressions.items()}
    return {key: {"residual": str(value), "passed": bool(value == 0)}
            for key, value in residuals.items()}


def build_result():
    identities = symbolic_identities()
    mass, length = 2.0, 0.7
    scale = (2 * mass * length**2)**(1 / 3)
    recovered_mass, error = quad(
        lambda x: 4 * math.pi * (scale * x)**2 * stress_tensor(scale * x, mass, length)["rho"] * scale,
        0, math.inf, epsabs=1e-11, epsrel=1e-11,
    )
    quadrature_passed = abs(recovered_mass - mass) < 1e-10 * mass
    all_passed = all(item["passed"] for item in identities.values()) and quadrature_passed
    return {
        "schema_version": 1,
        "status": "GEOMETRY_IDENTITIES_VERIFIED" if all_passed else "GEOMETRY_IDENTITY_FAILURE",
        "evidence_status": EVIDENCE_STATUS,
        "evidence_weight": 0,
        "observed_likelihood": None,
        "assumptions": ["Four-dimensional Einstein gravity with c=G=1.",
                        "The Hayward metric is selected as an ansatz; M>0 and l>0.",
                        "Geometric temperatures also set hbar=k_B=1.",
                        "Stress eigenvalues are inferred from the Einstein tensor."],
        "stress_fingerprint": {
            "q": "r^3/(2 M l^2)", "rho0": "3/(8 pi l^2)",
            "rho": "rho0/(1+q)^2", "p_r": "-rho", "p_t": "rho (2q-1)/(1+q)",
            "anisotropy": "p_t-p_r=3 rho q/(1+q)",
            "radial_nec": "saturated for every r", "tangential_nec": "nonnegative for every r",
            "sec": "satisfied iff q>=1/2", "dec": "satisfied iff q<=2",
            "horizon_relation": "q_h=r_h^2/l^2-1; every nonextremal outer horizon has q_h>2",
            "interpretation": "Necessary source fingerprint, not an NVG matter realization.",
        },
        "central_invariants_at_unit_length": curvature_invariants(0, 1, 1),
        "unit_length_horizon_scales": critical_scales(1),
        "schwarzschild_asymptotics": {
            "exact_delta_f": "4 M^2 l^2/[r(r^3+2 M l^2)]",
            "leading_delta_f": "4 M^2 l^2/r^4", "remainder_order": "O(r^-7)",
        },
        "checks": {"symbolic": identities,
                   "mass_quadrature": {"input_mass": mass, "length": length,
                                       "recovered_mass": recovered_mass, "estimated_error": error,
                                       "passed": quadrature_passed}},
        "limitations": [
            "No field configuration realizing this stress from the accepted NVG action has been derived.",
            "No collapse solution, perturbative stability analysis, or evaporation history is solved.",
            "Finite central curvature and SEC violation do not imply a flat cosmological bounce.",
            "The metric and stress tensor establish no information-transfer mechanism.",
            "No observational likelihood or experimental confirmation is supplied.",
        ],
    }


def validate_result(result):
    """Recompute and compare JSON values without Python's bool/number equality."""
    if not isinstance(result, dict) or result.get("status") != "GEOMETRY_IDENTITIES_VERIFIED":
        return False
    try:
        submitted = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = json.dumps(build_result(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return False
    return submitted == expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the dedicated geometry result JSON")
    args = parser.parse_args(argv)
    result = build_result()
    data = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.write:
        if result["status"] != "GEOMETRY_IDENTITIES_VERIFIED":
            raise ArithmeticError("geometry audit failed; artifact was not written")
        RESULT_PATH.write_text(data, encoding="utf-8")
    print(data, end="")
    return result


if __name__ == "__main__":
    main()
