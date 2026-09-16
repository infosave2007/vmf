#!/usr/bin/env python3
"""Conditional long-wavelength, normal-state Hartree/Vlasov NVG response.

This is not a full quantum RPA, finite-wave-number stability calculation or
NVG+cuscuton perturbation theorem. Original couplings and source artifacts
are unchanged. The CLI prints JSON and has no artifact-writing operation.
"""
from __future__ import annotations

import argparse
import json

import mpmath as mp
import sympy as sp

from source_complete_scaling_saturation_audit import BulkModel, INPUTS


STATUS = "conditional_long_wavelength_collisionless_response_not_full_NVG_stability"
REFERENCES = [
    "https://arxiv.org/html/2411.06960v1",
    "https://arxiv.org/abs/0910.3283",
    "https://arxiv.org/abs/2112.14246",
]
REQUIRED_IDENTITIES = frozenset({
    "scalar_gauss_C", "scalar_gauss_B", "static_susceptibility",
    "Lorentz_effective_mass", "current_feedback_F1", "first_sound_limit",
    "two_moment_determinant", "two_moment_continuity",
    "branch_existence_coefficient", "subluminal_upper_endpoint",
    "positive_l0_energy", "positive_l1_energy",
})
REQUIRED_RESIDUALS = frozenset({
    "C_matches_source_energy", "B_matches_source_energy",
    "a_plus_V_matches_source_energy", "first_sound_matches_source",
    "static_susceptibility_matches_source",
})


def finite(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool is not a physical number")
    try:
        value = mp.mpf(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name}: finite real number required") from exc
    if not mp.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name}: finite {'positive ' if positive else ''}number required")
    return value


def precision(dps):
    if isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 300:
        raise ValueError("dps must be an integer from 80 to 300")


def number(value):
    return mp.nstr(finite(value, "derived number"), 35)


def symbolic_checks():
    n, ef, V, NF, B, C, vf = sp.symbols("n ef V NF B C vf", positive=True)
    b, c, s0 = sp.symbols("b c s0", nonzero=True)
    ma, g, q, W, A0, gs, ns_m, Upp = sp.symbols("ma g q W A0 gs ns_m Upp", positive=True)
    r = n*V/ef
    F0 = NF*(V-B*B/C)
    F1 = -3*r/(1+r)
    mu = ef+n*V
    first = vf*vf*(1+F0)*(1+F1/3)/3
    ell0, ell1, L, sig = sp.symbols("F0 F1 L sig")
    moments = sp.Matrix([[1-ell0*L, -ell1*sig*L],
                         [-ell0*sig*L, 1-ell1*(sig*sig*L-sp.Rational(1, 3))]])
    nu0, nu1 = sp.symbols("nu0 nu1")
    field_solution = sp.Matrix([[s0, -c], [c, ma]]).inv()*sp.Matrix([-b, g])
    identities = {
        "scalar_gauss_C": (s0+c*c/ma).subs({s0: Upp+gs*gs*ns_m-q*q*A0*A0,
                                            c: 2*q*q*W*A0, ma: q*q*W*W})
                          -(Upp+gs*gs*ns_m+3*q*q*A0*A0),
        "scalar_gauss_B": field_solution[0]+(b-g*c/ma)/(s0+c*c/ma),
        "static_susceptibility": -NF/(1+F0)+1/(1/NF+V-B*B/C),
        "Lorentz_effective_mass": 1+F1/3-ef/mu,
        "current_feedback_F1": (F1+NF*V*vf*vf/(1+r)).subs(NF, 3*n/(ef*vf*vf)),
        "first_sound_limit": (first-n*(1/NF+V-B*B/C)/mu).subs(NF, 3*n/(ef*vf*vf)),
        "two_moment_determinant": moments.det()-(1+ell1/3)*(1-(ell0+ell1*sig*sig/(1+ell1/3))*L),
        "two_moment_continuity": (sig*nu0-nu1-ell1*nu1/3)-(sig*nu0-(1+ell1/3)*nu1),
        "branch_existence_coefficient": F0+F1/(1+F1/3)-(F0-3*r),
        "subluminal_upper_endpoint": (vf*vf*F0/(3*r)-(1-B*B/(C*V))).subs(NF, 3*n/(ef*vf*vf)),
        "positive_l0_energy": 1+F0-NF*(1/NF+V-B*B/C),
        "positive_l1_energy": 1+F1/3-1/(1+r),
    }
    if set(identities) != REQUIRED_IDENTITIES:
        raise ArithmeticError("incomplete symbolic certificate")
    result = {}
    for name, expression in identities.items():
        residual = sp.factor(sp.simplify(expression))
        result[name] = {"residual": str(residual), "passed": bool(residual == 0)}
    return result


def coefficients(n_ratio="1", *, dps=80):
    """Original equilibrium and independent scalar-density integral.

    The explicit numerical domain is an audit coverage choice, not a physical
    density cutoff. Values returned here are mpmath numbers, not serialized.
    """
    precision(dps)
    with mp.workdps(dps):
        nr = finite(n_ratio, "n_ratio", positive=True)
        if not mp.mpf("1e-18") <= nr <= mp.mpf("1e6"):
            raise ValueError("numerical audit covers 1e-18 <= n/n0 <= 1e6")
        model = BulkModel()
        scale = nr**(mp.mpf(1)/3)
        st = model.equilibrium(nr*model.n0,
                               guess=(max(mp.mpf(".8"), scale/4), max(mp.mpf("1.7"), scale)))
        n, W = st["n"], model.W0*st["y"]
        k, mass, ef = st["k"], st["m"], st["ef"]
        gs, q, g = model.MN/model.W0, model.momega/model.W0, model.gomega
        M2, vf = q*q*W*W, k/ef
        A0 = g*n/M2
        ns_m = model.d/(2*mp.pi**2)*mp.quad(lambda p: p**4/(p*p+mass*mass)**mp.mpf("1.5"), [0, k])
        NF = model.d*k*ef/(2*mp.pi**2)
        a, b, c = 1/NF, gs*mass/ef, 2*q*q*W*A0
        s0 = model.lam*(3*W*W-model.W0**2)+gs*gs*ns_m-q*q*A0*A0
        C, B, V = s0+c*c/M2, b-g*c/M2, g*g/M2
        if C <= 0:
            raise ArithmeticError("long-wavelength scalar elimination requires C>0")
        r = n*V/ef
        F0, F1 = NF*(V-B*B/C), -3*r/(1+r)
        mu = ef+n*V
        first = vf*vf*(1+F0)*(1+F1/3)/3
        residuals = {
            "C_matches_source_energy": (C-st["C_y"]/model.W0**2)/max(abs(C), 1),
            "B_matches_source_energy": (B-st["B_y"]/model.W0)/max(abs(B), 1),
            "a_plus_V_matches_source_energy": (a+V-st["D"])/max(abs(a+V), mp.mpf("1e-100")),
            "first_sound_matches_source": first-st["cs2"],
            "static_susceptibility_matches_source": (-NF/(1+F0))/(-1/st["mu_prime"])-1,
        }
        result = {"n_ratio": nr, "n": n, "W": W, "y": st["y"], "kF": k,
                "EF": ef, "vF2": vf*vf, "NF": NF, "a": a, "b": b,
                "c": c, "s0": s0, "M2": M2, "B": B, "C": C, "V": V,
                "r": r, "F0": F0, "F1": F1, "mu": mu,
                "first_sound_squared": first, "residuals": residuals}
        for name, value in result.items():
            if name != "residuals":
                finite(value, name)
        for name, value in residuals.items():
            finite(value, name)
        return result


def lindhard_from_log_gap(log_gap):
    """L(sigma), sigma=1+exp(log_gap), retaining exponentially tiny gaps."""
    gap = mp.exp(log_gap)
    return (1+gap)*(mp.log(2+gap)-log_gap)/2-1


def dispersion_from_log_gap(log_gap, F0, current_coefficient):
    gap = mp.exp(log_gap)
    # Do not form F0-current_coefficient*(1+gap)^2: that loses the gap
    # close to the continuum, even at arbitrarily high fixed precision.
    coefficient = (F0-current_coefficient)-current_coefficient*gap*(2+gap)
    return coefficient*lindhard_from_log_gap(log_gap)-1


def zero_sound_root(F0, r, *, dps=80):
    """Unique sigma>1 pole when F0>3r; r=0 is a deliberate useful control.

    For fixed-state r>=0, L>0 and L'<0. On the positive-coefficient interval,
    (F0-3r sigma^2)L decreases from infinity to zero. If F0<=3r its
    coefficient is nonpositive everywhere at sigma>1, so there is no pole.
    No claim of a unique threshold as a function of density is made.
    """
    precision(dps)
    with mp.workdps(dps):
        F0, r = finite(F0, "F0"), finite(r, "r")
        if r < 0:
            raise ValueError("r must be nonnegative")
        R, D = 3*r, F0-3*r
        if D <= 0:
            return None
        if R:
            gap_upper = D/(R*(mp.sqrt(F0/R)+1))
        else:
            gap_upper = max(mp.mpf(1), F0+1)
        hi = mp.log(gap_upper)
        lo = min(hi-1, mp.log(2)-4-2/D)
        f = lambda ell: dispersion_from_log_gap(ell, F0, R)
        for _ in range(100):
            if f(lo) > 0:
                break
            lo -= max(abs(lo), 1)
        else:
            raise ArithmeticError("could not bracket log-gap root")
        if not f(hi) < 0:
            raise ArithmeticError("upper log-gap endpoint is not outside the pole")
        for _ in range(4*dps+40):
            middle = (lo+hi)/2
            if middle in (lo, hi):
                break
            if f(middle) > 0:
                lo = middle
            else:
                hi = middle
        ell = (lo+hi)/2
        residual = f(ell)
        if abs(residual) >= mp.mpf("1e-60"):
            raise ArithmeticError("collisionless log-gap root did not converge")
        return {"log_gap": ell, "gap": mp.exp(ell), "residual": residual}


def threshold_witness(*, dps=80):
    """A numerical crossing witnessed between 100 n0 and 200 n0, not uniqueness."""
    precision(dps)
    with mp.workdps(dps):
        def difference(nr):
            c = coefficients(nr, dps=dps)
            return c["F0"]-3*c["r"]
        low, high = difference(mp.mpf(100)), difference(mp.mpf(200))
        if not low > 0 > high:
            raise ArithmeticError("declared threshold witness is not bracketed")
        root = mp.findroot(difference, (100, 200))
        if not 100 < root < 200 or abs(difference(root)) >= mp.mpf("1e-55"):
            raise ArithmeticError("numerical threshold witness failed")
        return {"n_over_n0": number(root), "difference_at_100_n0": number(low),
                "difference_at_200_n0": number(high),
                "residual": number(difference(root)),
                "globally_unique_density_threshold_proven": False,
                "meaning": "loss of an isolated sigma>1 pole, not collapse or instability"}


def audit(n_ratio="1", *, dps=80):
    precision(dps)
    with mp.workdps(dps):
        c = coefficients(n_ratio, dps=dps)
        root = zero_sound_root(c["F0"], c["r"], dps=dps)
        symbolic = symbolic_checks()
        symbolic_pass = (isinstance(symbolic, dict) and set(symbolic) == REQUIRED_IDENTITIES
                         and all(isinstance(row, dict) and row.get("passed") is True
                                 and row.get("residual") == "0" for row in symbolic.values()))
        residuals = c.get("residuals")
        if not isinstance(residuals, dict) or set(residuals) != REQUIRED_RESIDUALS:
            raise ArithmeticError("incomplete independent residual certificate")
        try:
            checked_residuals = {k: finite(v, k) for k, v in residuals.items()}
        except ValueError as exc:
            raise ArithmeticError("nonfinite or nonnumeric independent residual certificate") from exc
        residual_pass = all(abs(x) < mp.mpf("1e-55") for x in checked_residuals.values())
        branch = None
        if root is not None:
            increment = c["vF2"]*root["gap"]*(2+root["gap"])
            speed2 = c["vF2"]+increment
            endpoint = 1-c["B"]**2/(c["C"]*c["V"])
            if not root["gap"] > 0 or not 0 < speed2 < endpoint <= 1:
                raise ArithmeticError("zero-sound pole violates its subluminal bracket")
            branch = {"log_sigma_minus_one": number(root["log_gap"]),
                      "sigma_minus_one": number(root["gap"]),
                      "zero_sound_speed_squared": number(speed2),
                      "speed_squared_above_continuum": number(increment),
                      "strict_upper_speed_squared": number(endpoint),
                      "dispersion_residual": number(root["residual"]),
                      "positive_gap_retained_separately_from_rounded_speed": True}
        return {
            "status": STATUS, "evidence_weight": 0,
            "mathematical_checks_passed": bool(symbolic_pass and residual_pass),
            "inputs": {"n_over_n0": number(c["n_ratio"]), "dps": dps,
                       "original_parameters": dict(INPUTS)},
            "coefficients": {k: number(c[k]) for k in
                             ("y", "EF", "vF2", "NF", "M2", "B", "C", "V", "r", "F0", "F1", "mu")},
            "first_sound_squared": number(c["first_sound_squared"]),
            "static_density_susceptibility": number(-c["NF"]/(1+c["F0"])),
            "isolated_zero_sound_exists": bool(c["F0"] > 3*c["r"]),
            "zero_sound": branch,
            "Landau_energy_conditions": {"one_plus_F0_positive": bool(1+c["F0"] > 0),
                                         "one_plus_F1_over_3_positive": bool(1+c["F1"]/3 > 0),
                                         "higher_l_interactions": "zero within this Hartree Landau closure"},
            "residuals": {k: number(v) for k, v in c["residuals"].items()},
            "symbolic_checks": symbolic,
            "threshold_witness": threshold_witness(dps=dps),
            "scope": [
                "Stationary isotropic normal T=0 Hartree/Vlasov quasiparticles, one isoscalar channel with d=4.",
                "k/kF -> 0 and k,omega small compared with the coupled scalar/vector gaps; omega/k held fixed.",
                "No collision term, pairing, exchange/Fock, Dirac-sea polarization or particle-production calculation.",
                "The static limit takes omega -> 0 before k -> 0; its thermodynamic closure is not first sound substituted for zero sound.",
                "No finite-k, time-dependent bounce, full-gravity/cuscuton or quantum-NVG stability claim.",
                "The fixed couplings are not calibrated here; the existing nuclear-data mismatch is not repaired.",
            ],
            "references": list(REFERENCES),
        }


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--n-ratio", default="1")
    parser.add_argument("--dps", type=int, default=80)
    try:
        args = parser.parse_args(argv)
        result = audit(args.n_ratio, dps=args.dps)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        print(json.dumps({"status": "invalid_or_failed_audit", "evidence_weight": 0,
                          "mathematical_checks_passed": False, "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["mathematical_checks_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
