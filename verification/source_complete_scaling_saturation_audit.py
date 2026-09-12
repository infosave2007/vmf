#!/usr/bin/env python3
"""Independent mean-field identities and explicitly labelled inverse experiments.

Nothing in this module changes the accepted source-complete action, parameters,
stellar tables or resonator. Decimal inputs are assumptions, not output tables.
The two-target quartic calibration and three-target polynomial example are
research counterfactuals. Agreement with their calibration targets is not evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mpmath as mp


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "source_complete_scaling_saturation_results.json"
SOURCE_PATH = Path(__file__).resolve()
BASELINE_PATH = HERE / "source_complete_solution_audit.py"
INPUTS = {"W0": "859", "lam": "1.05", "MN": "939", "momega": "782.6",
          "gomega": "10.12", "hbarc": "197.3269804", "n0_fm3": "0.16", "d": 4}


class BulkModel:
    """Natural-unit bulk functional in y=W/W0; instantiate inside workdps."""

    def __init__(self, *, lam=None, gomega=None, polynomial=None):
        for key, value in INPUTS.items():
            setattr(self, key, mp.mpf(value))
        if lam is not None:
            self.lam = mp.mpf(lam)
        if gomega is not None:
            self.gomega = mp.mpf(gomega)
        if self.lam <= 0 or self.gomega < 0:
            raise ValueError("positive quartic scale and nonnegative vector coupling required")
        self.n0 = self.n0_fm3 * self.hbarc**3
        self.A = self.lam * self.W0**4
        self.Cv = self.gomega**2 / self.momega**2
        self.Cs = self.MN**2 / (2 * self.A)
        self.polynomial = None if polynomial is None else tuple(map(mp.mpf, polynomial))
        if self.polynomial is not None and len(self.polynomial) != 3:
            raise ValueError("polynomial needs coefficients of z^2,z^3,z^4")

    def fermi(self, n, y):
        if n <= 0 or y <= 0:
            raise ValueError("this finite-density branch requires n>0,y>0")
        k = (6 * mp.pi**2 * n / self.d)**(mp.mpf(1) / 3)
        m = self.MN * y
        ef = mp.sqrt(k*k + m*m)
        a = mp.asinh(k / m)
        e = self.d / (16 * mp.pi**2) * (k*ef*(2*k*k+m*m)-m**4*a)
        p = self.d / (48 * mp.pi**2) * (k*ef*(2*k*k-3*m*m)+3*m**4*a)
        ns = self.d * m / (4 * mp.pi**2) * (k*ef-m*m*a)
        ns_m = 3*ns/m - 3*n/ef
        return {"k": k, "m": m, "ef": ef, "energy": e, "pressure": p,
                "ns": ns, "ns_m": ns_m}

    def potential(self, y):
        z = y*y - 1
        if self.polynomial is None:
            return self.A*z*z/4, self.A*y*z, self.A*(3*y*y-1)
        u = uy = uyy = mp.mpf(0)
        for j, a in enumerate(self.polynomial, start=2):
            u += a*z**j
            uy += 2*j*a*y*z**(j-1)
            uyy += 2*j*a*z**(j-1) + 4*j*(j-1)*a*y*y*z**(j-2)
        return u, uy, uyy

    def state(self, n, y):
        f = self.fermi(n, y)
        u, uy, uyy = self.potential(y)
        v = self.Cv*n*n/(2*y*y)
        energy, pressure = f["energy"]+u+v, f["pressure"]-u+v
        mu = f["ef"] + self.Cv*n/(y*y)
        residual = self.MN*f["ns"] + uy - self.Cv*n*n/y**3
        C = uyy + self.MN**2*f["ns_m"] + 3*self.Cv*n*n/y**4
        B = self.MN**2*y/f["ef"] - 2*self.Cv*n/y**3
        D = f["k"]**2/(3*n*f["ef"]) + self.Cv/y**2
        relaxed = D-B*B/C if C else mp.nan
        return dict(f, n=n, y=y, U=u, Uy=uy, Uyy=uyy, vector=v,
                    energy_total=energy, pressure_total=pressure, mu=mu,
                    residual=residual, C_y=C, B_y=B, D=D,
                    mu_prime=relaxed, K=9*n*relaxed, cs2=n*relaxed/mu)

    def equilibrium(self, n, guess=("0.98", "1.02")):
        y = mp.findroot(lambda a: self.state(n, a)["residual"], tuple(map(mp.mpf, guess)))
        result = self.state(n, y)
        if y <= 0 or result["C_y"] <= 0:
            raise ValueError("not a positive locally stable scalar stationary state")
        return result


def quartic_calibration(n_fm3="0.16", binding="-16"):
    """Fit TWO declared targets; K is a held-out consequence, never a target."""
    base = BulkModel()
    n = mp.mpf(n_fm3)*base.hbarc**3
    mu0 = base.MN + mp.mpf(binding)
    k = base.fermi(n, mp.mpf(1))["k"]
    if not k < mu0 < base.MN:
        raise ValueError("inverse proof requires kF < target mu < vacuum mass")
    ymax = mp.sqrt(mu0**2-k*k)/base.MN

    def equation(y):
        f = base.fermi(n, y)
        return mu0*(1+y*y) - 2*f["ef"] + 4*f["pressure"]/n

    # h is strictly decreasing in the physical domain; establish a bracket.
    lo, hi = ymax*mp.mpf("1e-8"), ymax
    if not equation(lo) > 0 > equation(hi):
        raise ValueError("target is not bracketed in the positive-coupling domain")
    y = mp.findroot(equation, (lo, hi), solver="anderson")
    f = base.fermi(n, y)
    V = mu0-f["ef"]
    A = (4*f["pressure"]+2*n*V)/(1-y*y)**2
    Cv = V*y*y/n
    model = BulkModel(lam=A/base.W0**4, gomega=base.momega*mp.sqrt(Cv))
    return model, model.state(n, y)


def vector_ceiling(n_fm3="0.16", binding="-16"):
    """Necessary bound on gomega for ANY scalar potential, same mass relations."""
    base = BulkModel()
    n = mp.mpf(n_fm3)*base.hbarc**3
    mu0 = base.MN+mp.mpf(binding)
    k = base.fermi(n, mp.mpf(1))["k"]
    if not k < mu0:
        raise ValueError("target chemical potential must exceed massless Fermi energy")
    e_star = (mu0+mp.sqrt(mu0**2+3*k*k))/3
    y_star = mp.sqrt(e_star**2-k*k)/base.MN
    Cv_max = y_star**2*(mu0-e_star)/n
    y_min = mp.findroot(lambda y: base.MN**2*y/base.fermi(n, y)["ef"]
                       - 2*base.Cv*n/y**3, (mp.mpf("0.6"), mp.mpf("0.9")))
    mu_min = base.fermi(n, y_min)["ef"]+base.Cv*n/y_min**2
    return {"gomega_max": base.momega*mp.sqrt(Cv_max), "y_at_ceiling": y_star,
            "Cv_max": Cv_max, "original_mu_min_MeV": mu_min,
            "original_y_at_mu_min": y_min, "mu_target_MeV": mu0,
            "finite_curvature_K_at_ceiling_MeV": 3*mu0}


def polynomial_vacuum_bound(coefficients):
    """Exact quadratic minimum of q(z) on z>=-1 for U=z²q(z)."""
    a2, a3, a4 = map(mp.mpf, coefficients)
    if a4 > 0:
        zmin = max(mp.mpf(-1), -a3/(2*a4))
        qmin = a2+a3*zmin+a4*zmin*zmin
    elif a4 == 0 and a3 >= 0:
        zmin, qmin = mp.mpf(-1), a2-a3
    else:
        zmin = qmin = None
    return {"q_min": qmin, "z_at_q_min": zmin,
            "globally_nonnegative_potential": bool(qmin is not None and qmin >= 0),
            "quartic_in_z_leading_positive": bool(a4 > 0)}


def inverse_potential_jet(y, K_target="240", n_fm3="0.16", binding="-16"):
    """Local inverse design. Chosen y and all three targets are declared inputs."""
    base = BulkModel()
    y, K = mp.mpf(y), mp.mpf(K_target)
    n, mu0 = mp.mpf(n_fm3)*base.hbarc**3, base.MN+mp.mpf(binding)
    f = base.fermi(n, y)
    V = mu0-f["ef"]
    if V <= 0 or K <= 0 or y == 1:
        raise ValueError("jet needs positive vector coupling, K, and y!=1")
    Cv = V*y*y/n
    D = f["k"]**2/(3*n*f["ef"])+Cv/y**2
    B = base.MN**2*y/f["ef"]-2*Cv*n/y**3
    denominator = D-K/(9*n)
    cancellation_scale = base.MN**2*y/f["ef"]+abs(2*Cv*n/y**3)
    if denominator <= 0 or abs(B) <= 100*mp.eps*max(1,cancellation_scale):
        raise ValueError("inverse curvature is impossible or non-unique; B=0,K=3*mu needs a separate family")
    C = B*B/denominator
    U = f["pressure"]+n*V/2
    Uy = (n*V-f["m"]*f["ns"])/y
    Uyy = C-base.MN**2*f["ns_m"]-3*Cv*n*n/y**4
    z = y*y-1
    matrix = mp.matrix([[z**j for j in (2,3,4)],
                        [2*j*y*z**(j-1) for j in (2,3,4)],
                        [2*j*z**(j-1)+4*j*(j-1)*y*y*z**(j-2) for j in (2,3,4)]])
    coeff = tuple(mp.lu_solve(matrix, mp.matrix([U, Uy, Uyy])))
    model = BulkModel(gomega=base.momega*mp.sqrt(Cv), polynomial=coeff)
    return model, model.state(n, y), {"U": U, "Uy": Uy, "Uyy": Uyy,
                "coefficients": coeff, **polynomial_vacuum_bound(coeff)}


def _number(value):
    if not mp.isfinite(value):
        raise ValueError("nonfinite scientific value")
    return mp.nstr(value, 28)


def _shown_state(s, model):
    return {"y": _number(s["y"]), "P_MeV_fm3": _number(s["pressure_total"]/model.hbarc**3),
            "binding_MeV": _number(s["energy_total"]/s["n"]-model.MN),
            "mu_MeV": _number(s["mu"]), "K_MeV": _number(s["K"]),
            "cs2": _number(s["cs2"]), "C_y_MeV4": _number(s["C_y"])}


def _check_state(s):
    scale = max(abs(s["energy_total"]), mp.mpf(1))
    normalized = [abs(s["residual"])/scale,
                  abs(s["energy_total"]+s["pressure_total"]-s["n"]*s["mu"])/scale,
                  abs(s["energy_total"]-3*s["pressure_total"]-4*s["U"]+s["y"]*s["Uy"])/scale]
    if max(normalized) >= mp.mpf("1e-60"):
        raise ArithmeticError("stationarity, Legendre or scale identity failed")


def calculate(dps=80):
    if dps < 80:
        raise ValueError("verification requires at least 80 decimal digits")
    with mp.workdps(dps):
        base = BulkModel()
        original = base.equilibrium(base.n0)
        _check_state(original)
        xmin, ymin = mp.findroot(lambda x,y: (base.state(x*base.n0,y)["residual"],
                                              base.state(x*base.n0,y)["B_y"]), (2, 1))
        minimum = base.state(xmin*base.n0,ymin)
        _check_state(minimum)
        if abs(minimum["cs2"]-mp.mpf(1)/3) > mp.mpf("1e-60"):
            raise ArithmeticError("minimum-field sound-speed identity failed")
        xreturn = mp.findroot(lambda x: base.state(x*base.n0,mp.mpf(1))["residual"], (4,5))
        returned = base.state(xreturn*base.n0, mp.mpf(1))
        _check_state(returned)
        calibrated, sat = quartic_calibration()
        _check_state(sat)
        # Independently differentiate equilibrium mu; K was not in the root equation.
        derivative = mp.diff(lambda n: calibrated.equilibrium(n,(sat["y"]*mp.mpf(".99"),sat["y"]*mp.mpf("1.01")))["mu"], sat["n"])
        if abs(9*sat["n"]*derivative-sat["K"]) > mp.mpf("1e-55"):
            raise ArithmeticError("independent incompressibility derivative failed")
        examples = []
        for chosen_y in ("0.75", "0.90"):
            model, state, jet = inverse_potential_jet(chosen_y)
            _check_state(state)
            if abs(state["K"]-240) > mp.mpf("1e-55"):
                raise ArithmeticError("local inverse curvature residual")
            if (abs(state["pressure_total"])/model.hbarc**3 > mp.mpf("1e-55")
                    or abs(state["energy_total"]/state["n"]-model.MN+16) > mp.mpf("1e-55")):
                raise ArithmeticError("local inverse saturation targets failed")
            examples.append({"chosen_y_not_an_observation": chosen_y,
                "calibration_targets": {"n_fm3":"0.16", "binding_MeV":"-16", "K_MeV":"240"},
                "gomega": _number(model.gomega), "local_state": _shown_state(state,model),
                "a2_a3_a4_MeV4": [_number(v) for v in jet["coefficients"]],
                "globally_nonnegative_potential": jet["globally_nonnegative_potential"],
                "q_min_MeV4": None if jet["q_min"] is None else _number(jet["q_min"]),
                "held_out_isoscalar_only_symmetry_energy_MeV": _number(state["k"]**2/(6*state["ef"])),
                "physical_validation": "NOT_ESTABLISHED; isovector interaction, composition and independent calibration missing"})
        # Tiny root residuals are summarized by checked predicates, not rounded
        # into fictitious nonzero physical pressures in calibration rows.
        for s in (sat,):
            if abs(s["pressure_total"])/base.hbarc**3 > mp.mpf("1e-55"):
                raise ArithmeticError("calibration pressure target failed")
        sat_shown = _shown_state(sat, calibrated)
        sat_shown["P_MeV_fm3"] = "0 (target; residual < 1e-55)"
        for row in examples:
            row["local_state"]["P_MeV_fm3"] = "0 (target; residual < 1e-55)"
        return {"original_inputs": INPUTS.copy(),
                "original": {"state_at_n0": _shown_state(original,base),
                    "Cv_fm2": _number(base.Cv*base.hbarc**2), "Cs_fm2": _number(base.Cs*base.hbarc**2),
                    "Cv_over_Cs": _number(base.Cv/base.Cs),
                    "global_no_binding_sufficient_condition": bool(base.Cv >= base.Cs),
                    "proof_scope": "quartic positive scalar potential, linear fermion mass, inverse-square vector energy; all n>0,y>0"},
                "landmarks": {"field_minimum": {"x":_number(xmin), **_shown_state(minimum,base)},
                              "zero_trace_nonvacuum": {"x":_number(xreturn), **_shown_state(returned,base)}},
                "any_scalar_potential_vector_bound": {k:_number(v) for k,v in vector_ceiling().items()},
                "two_target_quartic_calibration": {
                    "training_targets": {"n_fm3":"0.16", "binding_MeV":"-16"},
                    "fitted_lambda":_number(calibrated.lam), "fitted_gomega":_number(calibrated.gomega),
                    "state":sat_shown, "K_was_fitted":False,
                    "held_out_K_comparison": "outside both 220-260 and broader 200-350 MeV reference ranges"
                        if sat["K"] > 350 else "requires comparison with stated reference assumptions",
                    "deployed_to_baseline":False},
                "inverse_potential_examples": examples,
                "identity_checks": "stationarity, Legendre, scale, field-minimum sound speed, independently differentiated K"}


def build_result():
    a, b = calculate(80), calculate(120)
    if a != b:
        raise ArithmeticError("80/120-digit science differs in serialized 28-digit values")
    original_status = ("ORIGINAL_NO_BINDING" if a["original"]["global_no_binding_sufficient_condition"]
                       else "ORIGINAL_NOT_EXCLUDED_BY_SUFFICIENT_BOUND")
    return {"schema_version":1,
            "status":f"IDENTITIES_VERIFIED_{original_status}_CALIBRATED_EXAMPLES_NOT_VALIDATED",
            "evidence_weight":0,
            "precision_check":{"dps":[80,120],"significant_digits_compared":28,"pass":True},
            "source_sha256":hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            "baseline_source_sha256":hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest(),
            "science":a}


def validate_result(result):
    """Regenerate all numbers in this process, not a static success/schema check.

    A caller can additionally invoke the CLI in a fresh process; this function
    itself does not create a new process or use a second numerical implementation.
    """
    try:
        fresh = build_result()
        return result == fresh
    except (ValueError, ArithmeticError, TypeError):
        return False


def write_result(result, path=RESULT_PATH):
    if not validate_result(result):
        raise ValueError("invalid/stale result; output was not changed")
    data = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n"
    Path(path).write_text(data, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write only the dedicated research result")
    args = parser.parse_args()
    result = build_result()
    if args.write:
        write_result(result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
