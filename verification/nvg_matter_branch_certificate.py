#!/usr/bin/env python3
"""Global sufficient certificate for the ORIGINAL quartic homogeneous NVG EOS.

This is an analytic certificate, not a density scan promoted to a proof.  Write
y=W/W0, A=lambda W0**4, Cv=gomega**2/momega**2, Cs=MN**2/(2*A), R=Cv/Cs.
For n>0 the stationary residual is
 F=MN*ns+A*y*(y*y-1)-Cv*n*n/y**3, with 0<ns<n and ns_m>0.
Completing the square in n gives
 F <= A*y*(y*y-1)+MN**2*y**3/(4*Cv).
Thus every root has y>=yL=(1+1/(2*R))**(-1/2), and F<0 below yL.
For R>=1/4 and y>=yL,
 C_y=F_y=A*(3*y*y-1)+MN**2*ns_m+3*Cv*n*n/y**4>0.
Together with endpoint coercivity this proves one positive global energy
minimum for every n>0, smooth in n.  For R>1/4 the uniform squared gap is
 C_W >= lambda*W0**2*(4*R-1)/(2*R+1)>0.

Let B=E_ny, D=E_nn, mu=E_n>0.  On shell C_y+3*n*B/y=2*A and
 cs2=n*(D-B*B/C_y)/mu=(1-2*A*y*B/(mu*C_y))/3.
If B<0, cs2>1/3.  If B>=0, 0<=B<=MN, D>=Cv/y**2, and
 D-B*B/C_y >= Cv/y**2-MN**2/[A*(3*y*y-1)]>0
for R>5/4: the dimensionless numerator at yL is R-5/4.
Also cs2<=n*D/mu<1, since mu-n*D=(m*m+2*kF*kF/3)/EF>0.
These are global sufficient conditions, NOT necessary stability thresholds.

At any B=0, y_n=0 and dB/dn=B_n=-MN**2*y*kF**2/(3*n*EF**3)
-2*Cv/y**3<0.  B tends to MN>0 at n->0.  At large n, rescale
W=a*n**(1/3): the limiting coefficient h(a)=FF(1,gs*a)+lambda*a**4/4
+Gv/(2*a*a) is strictly convex and coercive, with one minimum a>0.
The exact rescaled equation is h'(a)=lambda*W0**2*a*n**(-2/3);
the implicit-function theorem gives a smooth expansion at n**(-2/3)=0.
Hence W~a*n**(1/3), B<0 eventually, and there is exactly one field minimum;
it is exactly the unique cs2=1/3 crossing, with C_W=2*lambda*W0**2.
Formally epsilon=h*n**(4/3)-(lambda*W0**2*a*a/2)*n**(2/3)+O(1),
cs2=1/3+lambda*W0**2*a*a/(6*h)*n**(-2/3)+O(n**(-4/3)).
High-density limits are formal extrapolations, not demonstrated physical EOS.

All conclusions concern zero-T homogeneous, instantaneous ideal-Fermi closure
on the stationary scalar branch, not scalar tracking, collective-mode poles,
perturbative causality, stellar stability, or a derivation of added gravity.
The CLI emits JSON only and writes no artifact or fitted parameter.
"""

import argparse
import json

import mpmath as mp

try:
    from .source_complete_scaling_saturation_audit import BulkModel
except ImportError:
    from source_complete_scaling_saturation_audit import BulkModel


REQUIRED_SYMBOLIC_CHECKS = frozenset({
    "completed_square", "on_shell_curvature", "fixed_field_response",
    "curvature_bound", "positive_response_margin", "trace_on_shell", "sound_identity",
})


def require_symbolic_certificates(checks):
    """Reject missing, extra, false, empty or merely truthy certificates."""
    if (not isinstance(checks, dict) or set(checks) != REQUIRED_SYMBOLIC_CHECKS
            or not all(value is True for value in checks.values())):
        raise ArithmeticError("complete named symbolic certificates must all be bool True")
    return True


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(name + " must be a finite positive number, not bool")
    try:
        result = mp.mpf(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + " must be a finite positive number") from exc
    if not mp.isfinite(result) or result <= 0:
        raise ValueError(name + " must be finite and positive")
    return result


def sufficient_bounds(R, W0="859", lam="1.05"):
    """Failure of a sufficient threshold is explicitly inconclusive."""
    R, W0, lam = _positive(R, "R"), _positive(W0, "W0"), _positive(lam, "lambda")
    yL = mp.sqrt(2*R/(2*R+1))
    uniqueness = R >= mp.mpf(1)/4
    gap = lam*W0**2*(4*R-1)/(2*R+1)
    return {"R": R, "y_lower": yL,
            "unique_global_branch_certified": bool(uniqueness),
            "positive_uniform_gap_certified": bool(R > mp.mpf(1)/4),
            "C_W_lower_MeV2": gap if uniqueness else None,
            "gap_lower_MeV": mp.sqrt(gap) if gap >= 0 else None,
            "zero_to_one_cs2_certified": bool(R > mp.mpf(5)/4),
            "threshold_failure_meaning": "inconclusive, not evidence of instability"}


def symbolic_certificates():
    """Exact algebraic checks of the identities used by the analytic proof."""
    import sympy as sp
    A, cv, n, y, M, ns, ef, k, R = sp.symbols("A cv n y M ns ef k R", positive=True)
    residual = M*ns+A*y*(y*y-1)-cv*n*n/y**3
    B = M*M*y/ef-2*cv*n/y**3
    C = A*(3*y*y-1)+M*M*(3*ns/(M*y)-3*n/ef)+3*cv*n*n/y**4
    D = k*k/(3*n*ef)+cv/y**2
    mu = ef+cv*n/y**2
    yL2 = 2*R/(2*R+1)
    completed = M*M*y**3/(4*cv)-cv*(n-M*y**3/(2*cv))**2/y**3
    bb, cc, mm = sp.symbols("bb cc mm", nonzero=True)
    cs = (mm-y*bb)/(3*mm)-n*bb*bb/(mm*cc)
    checks = {
        "completed_square": sp.simplify(M*n-cv*n*n/y**3-completed),
        "on_shell_curvature": sp.simplify(C+3*n*B/y-2*A-3*residual/y),
        "fixed_field_response": sp.simplify((mu-3*n*D-y*B)*ef).subs(ef**2,k*k+M*M*y*y).simplify(),
        "curvature_bound": sp.simplify(3*yL2-1-(4*R-1)/(2*R+1)),
        "positive_response_margin": sp.simplify(R*(3-1/yL2)/2-1-(R-sp.Rational(5,4))),
        "trace_on_shell": sp.simplify(M*y*ns+A*(y*y-1)**2-cv*n*n/y**2-A*(1-y*y)-y*residual),
        "sound_identity": sp.factor(cs-(1-y*bb*(cc+3*n*bb/y)/(mm*cc))/3),
    }
    result = {name: bool(expression == 0) for name, expression in checks.items()}
    require_symbolic_certificates(result)
    return result


def _equilibrium(model, n, yL):
    lo, hi = yL, max(mp.mpf(2), yL*2)
    while model.state(n, hi)["residual"] <= 0:
        hi *= 2
    for _ in range(4*mp.mp.dps):
        mid = (lo+hi)/2
        if model.state(n, mid)["residual"] > 0:
            hi = mid
        else:
            lo = mid
    return model.state(n, (lo+hi)/2)


def calculate(dps=60, density_ratios=("0.01", "0.1", "1", "2", "5", "10")):
    """Unchanged original parameters; independent quadrature at sanity points."""
    if isinstance(dps, bool) or not isinstance(dps, int) or not 50 <= dps <= 300:
        raise ValueError("dps must be an integer in [50,300]")
    with mp.workdps(dps):
        symbols = symbolic_certificates()
        require_symbolic_certificates(symbols)
        ratios = tuple(_positive(x, "density ratio") for x in density_ratios)
        if not ratios:
            raise ValueError("at least one density ratio is required")
        model = BulkModel()
        bounds = sufficient_bounds(model.Cv/model.Cs, model.W0, model.lam)
        points = []
        tol = mp.mpf(10)**(-(dps-15))
        for x in ratios:
            n = x*model.n0
            s = _equilibrium(model, n, bounds["y_lower"])
            k, m = s["k"], s["m"]
            # Independent integral formulas, not the closed-form Fermi primitives.
            factor = model.d/(2*mp.pi**2)
            ff = factor*mp.quad(lambda p:p*p*mp.sqrt(p*p+m*m), [0,k])
            pf = factor/3*mp.quad(lambda p:p**4/mp.sqrt(p*p+m*m), [0,k])
            ns = factor*mp.quad(lambda p:p*p*m/mp.sqrt(p*p+m*m), [0,k])
            errors = [abs(ff-s["energy"])/ff, abs(pf-s["pressure"])/pf,
                      abs(ns-s["ns"])/ns,
                      abs(s["residual"])/s["energy_total"],
                      abs(s["cs2"]-(1-2*model.A*s["y"]*s["B_y"]/(s["mu"]*s["C_y"]))/3)]
            if max(errors) > tol or not (0 < s["cs2"] < 1):
                raise ArithmeticError("independent sanity check failed")
            points.append({"n_over_n0": x, "y": s["y"], "cs2": s["cs2"],
                           "C_W_MeV2": s["C_y"]/model.W0**2,
                           "max_normalized_residual": max(errors),
                           "scope": "research domain" if x <= 10 else "formal extrapolation"})
        xmin, ymin = mp.findroot(lambda x,y:(model.state(x*model.n0,y)["residual"],
                                               model.state(x*model.n0,y)["B_y"]), (2,1))
        minimum = model.state(xmin*model.n0,ymin)
        if abs(minimum["cs2"]-mp.mpf(1)/3) > tol:
            raise ArithmeticError("minimum crossing failed")
        gs, Gv = model.MN/model.W0, model.Cv*model.W0**2
        a = mp.findroot(lambda a:gs*model.fermi(mp.mpf(1),a/model.W0)["ns"]
                        +model.lam*a**3-Gv/a**3, (2,3))
        h = model.fermi(mp.mpf(1),a/model.W0)["energy"]+model.lam*a**4/4+Gv/(2*a*a)
        return {"status": "conditional_mathematical_certificate", "evidence_weight": 0,
                "mathematical_checks_passed": True,
                "scope": "analytic sufficient stationary homogeneous EOS certificate; no fits",
                "dps": dps, "bounds": bounds, "symbolic": symbols,
                "sanity_points_not_global_proof": points,
                "unique_field_minimum": {"n_over_n0": xmin, "y": ymin,
                    "cs2": minimum["cs2"], "C_W_MeV2": minimum["C_y"]/model.W0**2},
                "formal_high_density": {"W_over_n_one_third": a,
                    "epsilon_over_n_four_thirds": h,
                    "cs2_correction_coefficient": model.lam*model.W0**2*a*a/(6*h),
                    "physical_validity_asserted": False}}


def _json(value):
    if isinstance(value, mp.mpf):
        if not mp.isfinite(value):
            raise ValueError("nonfinite output")
        return mp.nstr(value, 35)
    if isinstance(value, dict):
        return {key:_json(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)):
        return [_json(item) for item in value]
    return value


class _JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = _JSONArgumentParser(description=__doc__)
    parser.add_argument("--dps", type=int, default=60)
    try:
        args = parser.parse_args(argv)
        result = calculate(args.dps)
        require_symbolic_certificates(result.get("symbolic"))
        if result.get("mathematical_checks_passed") is not True:
            raise ArithmeticError("mathematical checks did not pass")
        rendered = json.dumps(_json(result), ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as exc:
        print(json.dumps({"status": "invalid_or_failed", "evidence_weight": 0,
                          "mathematical_checks_passed": False, "error": str(exc)}, allow_nan=False))
        return 2
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
