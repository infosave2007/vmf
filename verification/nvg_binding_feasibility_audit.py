#!/usr/bin/env python3
"""Transparent inverse nuclear-binding feasibility, not a baseline repair.

The original gomega=10.12 cannot reach the declared saturation chemical
potential, for any density-independent U(W) with the same mass structure.
The envelope construction changes gomega AND the scalar potential. All
saturation inputs are design targets, never held-out predictions. It supplies
a homogeneous bulk common-tangent certificate and a local fluid finite-k
test, not finite nuclei, quantum matter or full gravity stability.

No saved artifacts are read as answers or written. The CLI prints JSON.

Natural units, degeneracy d=4, y=W/W0>0, and the no-sea normal T=0
homogeneous fermion closure are assumed. For any fixed scalar potential,
mu=EF+Cv*n/y^2. At declared n and mu this gives
Cv=(mu-EF)*(EF^2-kF^2)/(M^2*n); maximizing over EF is independent of U.
For the inverse jet, D=E_nn, B=E_ny, C=E_yy and K=9n(D-B^2/C).

The density-dependent optimizer below defines a function U_floor(y) ONCE;
it is not an evolving-density coupling. E_nn>0 gives its unique maximum,
so E-mu*n>=U-U_floor. At the liquid point U_floor''=-F_yy-V_yy+B^2/D.
The added barrier has curvature C-B^2/D>0 there and zero value/slope.
Its only positive-y roots are y=1 and y=y_star. Strict density convexity
then proves the stated two equality states, not merely sampled minima.
At x=mu-M*y down to zero, n_bar ~ d*(2mu)^(3/2)*x^(3/2)/(6pi^2)
and U_floor ~ d*(2mu)^(3/2)*x^(5/2)/(15pi^2). This establishes C2/not C3.
For y^2>=2 and 0<y_star<1, U>=beta*y^8/16; at fixed n>0 the vector
energy diverges as y approaches zero. Neither statement supplies gravity.
"""
from __future__ import annotations

import argparse
import json

import mpmath as mp
import sympy as sp

from source_complete_scaling_saturation_audit import (
    BulkModel, INPUTS, inverse_potential_jet, vector_ceiling,
)


STATUS = "counterfactual_binding_feasibility_not_accepted_NVG_or_prediction"
REQUIRED_SYMBOLIC = frozenset({
    "vector_strength_derivative", "vector_maximum_stationary",
    "K_ceiling_at_zero_mixing", "inverse_K_identity", "grand_curvature_gap",
    "barrier_target_value", "barrier_target_slope", "barrier_target_curvature",
    "barrier_vacuum_value", "barrier_vacuum_slope", "barrier_vacuum_curvature",
    "finite_wave_determinant", "finite_wave_constant_from_K",
    "finite_wave_linear_lower_bound", "octavic_dominant_balance",
    "octavic_energy_coefficient", "octavic_asymptotic_sound",
})


def finite(value, name, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool is not a physical number")
    try:
        value = mp.mpf(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: finite real number required") from exc
    if not mp.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name}: finite {'positive ' if positive else ''}number required")
    return value


def precision(dps):
    if isinstance(dps, bool) or not isinstance(dps, int) or not 80 <= dps <= 300:
        raise ValueError("dps must be an integer from 80 to 300")


def number(value):
    return mp.nstr(finite(value, "derived number"), 35)


def _stable_fermi_integral(k, mass, degeneracy, *, pressure=False):
    """Positive T=0 Fermi energy/pressure without subtractive closed forms.

    For r=k/m<=1/2, integrate the convergent binomial expansion term by
    term: sum_j binom(eta,j)*r^(2j)/(b+2j), with (eta,b)=(1/2,3)
    for energy or (-1/2,5) for pressure. The next-term ratio in magnitude
    is below r^2<=1/4, so the omitted tail is bounded by 4/3 of that term.
    At larger r, use a positive integral scaled by EF to avoid cancellation
    and extreme dimensional integrands. No absolute-value or clipping repair.
    This helper is local to the new counterfactual; BulkModel is unchanged.
    """
    r = k/mass
    if r <= mp.mpf(".5"):
        eta = -mp.mpf(".5") if pressure else mp.mpf(".5")
        base = 5 if pressure else 3
        term = total = mp.mpf(1)/base
        j = 0
        while True:
            term *= ((eta-j)/(j+1))*r*r*(base+2*j)/(base+2*j+2)
            if abs(term)*mp.mpf(4)/3 <= mp.eps*abs(total)/8:
                break
            total += term
            j += 1
        if pressure:
            return degeneracy*k**5/(6*mp.pi**2*mass)*total
        return degeneracy*k**3*mass/(2*mp.pi**2)*total
    ef = mp.sqrt(k*k+mass*mass)
    km, mm = k/ef, mass/ef
    if pressure:
        integral = mp.quad(lambda t: t**4/mp.sqrt(mm*mm+km*km*t*t), [0, 1])
        return degeneracy*k**5/(6*mp.pi**2*ef)*integral
    integral = mp.quad(lambda t: t*t*mp.sqrt(mm*mm+km*km*t*t), [0, 1])
    return degeneracy*k**3*ef/(2*mp.pi**2)*integral


def symbolic_checks():
    E, mu, k, M, n, D, B, K, C = sp.symbols("E mu k M n D B K C", positive=True)
    Cv = (E*E-k*k)*(mu-E)/(M*M*n)
    peak = (mu+sp.sqrt(mu*mu+3*k*k))/3
    cap = 9*(mu-E)+3*k*k/E
    Cy = B*B/(D-K/(9*n))
    y, ys, beta, gap = sp.symbols("y ys beta gap", positive=True)
    zs = ys*ys-1
    barrier = beta*(y*y-1)**2*(y*y-ys*ys)**2
    z, a, b, c, s, ma, g = sp.symbols("z a b c s ma g", nonzero=True)
    knn, knw, kww = a+g*g/(z+ma), b-g*c/(z+ma), z+s+c*c/(z+ma)
    P1 = a*(ma+s)+g*g-b*b
    P0 = ma*(a*s-b*b)+a*c*c+g*g*s+2*b*g*c
    cc, bb, nn = sp.symbols("cc bb nn", positive=True)
    aa = (cc/(8*bb))**sp.Rational(1, 10)
    e_leading = 5*bb*aa**8*nn**sp.Rational(8, 5)
    identities = {
        "vector_strength_derivative": sp.diff(Cv, E)-(-3*E*E+2*mu*E+k*k)/(M*M*n),
        "vector_maximum_stationary": sp.diff(Cv, E).subs(E, peak),
        "K_ceiling_at_zero_mixing": (cap-3*mu).subs(k*k, 3*E*E-2*mu*E),
        "inverse_K_identity": 9*n*(D-B*B/Cy)-K,
        "grand_curvature_gap": (Cy-B*B/D)-Cy*K/(9*n*D),
        "barrier_target_value": barrier.subs(y, ys),
        "barrier_target_slope": sp.diff(barrier, y).subs(y, ys),
        "barrier_target_curvature": sp.diff(barrier, y, 2).subs(y, ys)-8*beta*ys*ys*zs*zs,
        "barrier_vacuum_value": barrier.subs(y, 1),
        "barrier_vacuum_slope": sp.diff(barrier, y).subs(y, 1),
        "barrier_vacuum_curvature": sp.diff(barrier, y, 2).subs(y, 1)-8*beta*zs*zs,
        "finite_wave_determinant": (z+ma)*(knn*kww-knw*knw)-(a*z*z+P1*z+P0),
        "finite_wave_constant_from_K": (P0-ma*C*K/(9*n)).subs(
            {s: C-c*c/ma, b: B+g*c/ma, a: D-g*g/ma}).subs(K, 9*n*(D-B*B/C)),
        "finite_wave_linear_lower_bound": P1.subs(s, C-c*c/ma)
                                             -(g*g-b*b-a*c*c/ma)-a*(ma+C),
        "octavic_dominant_balance": 8*bb*aa**10-cc,
        "octavic_energy_coefficient": bb*aa**8+cc/(2*aa**2)-5*bb*aa**8,
        "octavic_asymptotic_sound": nn*sp.diff(e_leading, nn, 2)/sp.diff(e_leading, nn)-sp.Rational(3, 5),
    }
    if set(identities) != REQUIRED_SYMBOLIC:
        raise ArithmeticError("incomplete symbolic certificate")
    return {key: {"residual": str(sp.factor(sp.simplify(value))),
                  "passed": bool(sp.factor(sp.simplify(value)) == 0)}
            for key, value in identities.items()}


class BindingDesign:
    """A fixed newly designed potential; never recalibrate at the queried n.

    U_floor(y)=sup_{n>=0}[mu_target*n-F(n,M*y)-Cv*n^2/(2*y^2)].
    Its optimizer is an auxiliary definition of a function of y only, NOT
    the actual evolving baryon density. Strict convexity in n proves that
    the optimizer is unique. The barrier chooses the two equality points.
    """
    def __init__(self, y=".75", K_target="240", n_fm3=".16", binding="-16",
                 *, dps=80, barrier_sign=1):
        precision(dps)
        if isinstance(barrier_sign, bool) or barrier_sign not in (-1, 1):
            raise ValueError("barrier_sign must be -1 or +1")
        self.dps = dps
        with mp.workdps(dps):
            self.base = BulkModel()
            self.y = finite(y, "target y", positive=True)
            self.K = finite(K_target, "target K", positive=True)
            self.n_fm3 = finite(n_fm3, "target n_fm3", positive=True)
            self.binding = finite(binding, "target binding")
            self.mu = self.base.MN+self.binding
            self.n = self.n_fm3*self.base.hbarc**3
            f = self.base.fermi(self.n, self.y)
            self.k, self.EF = f["k"], f["ef"]
            if not self.k < self.mu < self.base.MN or not 0 < self.y < 1:
                raise ValueError("design requires kF < mu_target < M and 0<y<1")
            self.V = self.mu-self.EF
            if self.V <= 0:
                raise ValueError("chosen effective mass leaves no positive vector coupling")
            self.Cv = self.V*self.y*self.y/self.n
            self.gomega = self.base.momega*mp.sqrt(self.Cv)
            self.D = self.k**2/(3*self.n*self.EF)+self.Cv/self.y**2
            self.By = self.base.MN**2*self.y/self.EF-2*self.V/self.y
            denominator = self.D-self.K/(9*self.n)
            if abs(self.By) < 100*mp.eps*max(self.base.MN, abs(self.V/self.y)):
                raise ValueError("zero-mixing ceiling family is unsupported here; K=3*mu permits a separate finite-C family")
            if denominator <= 0:
                raise ValueError("finite positive scalar curvature is incompatible with this K/y design")
            if denominator == self.D:
                raise ValueError("target K is unresolved at the requested precision; increase dps")
            self.Cy = self.By*self.By/denominator
            self.delta_y = self.Cy-self.By*self.By/self.D
            if self.delta_y <= 0:
                raise ValueError("positive target curvature gap is unresolved at the requested precision; increase dps")
            self.zs = self.y*self.y-1
            self.beta = barrier_sign*self.delta_y/(8*self.y*self.y*self.zs*self.zs)
            self.U = f["pressure"]+self.n*self.V/2
            self.Uy = (self.n*self.V-f["m"]*f["ns"])/self.y
            self.Uyy = self.Cy-self.base.MN**2*f["ns_m"]-3*self.Cv*self.n**2/self.y**4
            self.barrier_sign = barrier_sign

    def _precision(self):
        # mp.diff raises working precision internally. Never suppress it.
        return mp.workdps(max(self.dps, mp.mp.dps))

    def envelope_state(self, y):
        with self._precision():
            y = finite(y, "y", positive=True)
            mass = self.base.MN*y
            if mass >= self.mu:
                return {"n": mp.mpf(0), "k": mp.mpf(0), "value": mp.mpf(0)}
            excess = self.mu-mass
            density = lambda kk: self.base.d*kk**3/(6*mp.pi**2)
            # Both nonnegative contributions must separately be <=excess.
            # Their independent bounds keep the bracket useful as y -> 0.
            upper = min(mp.sqrt(excess*(self.mu+mass)),
                        (6*mp.pi**2*y*y*excess/(self.base.d*self.Cv))**(mp.mpf(1)/3))
            equation = lambda kk: (kk*kk/(mp.sqrt(kk*kk+mass*mass)+mass)
                                    +self.Cv*density(kk)/(y*y)-excess)
            kk = mp.findroot(equation, (mp.mpf(0), upper), solver="anderson",
                             maxsteps=2*mp.mp.dps+50)
            nn = density(kk)
            if not 0 < kk < upper or nn <= 0:
                raise ArithmeticError("auxiliary Legendre optimizer left its bracket")
            pressure_value = _stable_fermi_integral(kk, mass, self.base.d, pressure=True)
            return {"n": nn, "k": kk,
                    "value": pressure_value+self.Cv*nn*nn/(2*y*y)}

    def floor(self, y):
        return self.envelope_state(y)["value"]

    def barrier(self, y):
        with self._precision():
            y = finite(y, "y", positive=True)
            z = y*y-1
            return self.beta*z*z*(z-self.zs)**2

    def potential(self, y):
        with self._precision():
            return self.floor(y)+self.barrier(y)

    def energy(self, n, y):
        with self._precision():
            n, y = finite(n, "n"), finite(y, "y", positive=True)
            if n < 0:
                raise ValueError("n must be nonnegative")
            if n == 0:
                return self.potential(y)
            k = (6*mp.pi**2*n/self.base.d)**(mp.mpf(1)/3)
            ef_energy = _stable_fermi_integral(k, self.base.MN*y, self.base.d)
            return ef_energy+self.Cv*n*n/(2*y*y)+self.potential(y)

    def grand_gap(self, n, y):
        with self._precision():
            return self.energy(n, y)-self.mu*finite(n, "n")

    def finite_wave(self):
        """At the fitted point, the local ideal-fluid energy matrix is positive.

        z=k^2; after eliminating vector A0 its entries are
        [a+g^2/(z+M2), b-g*c/(z+M2); same, z+s+c^2/(z+M2)].
        c=2*g*n/W retains the W-dependent vector-mass mixing; C=s+c^2/M2.
        Positive scalar, fluid and Proca kinetic energies are assumptions of
        this closure. Arbitrarily large mathematical k is NOT a controlled
        hydrodynamic approximation to the collisionless microscopic theory.
        """
        with self._precision():
            W = self.y*self.base.W0
            q, gs = self.base.momega/self.base.W0, self.base.MN/self.base.W0
            M2, g = q*q*W*W, self.gomega
            a = self.k*self.k/(3*self.n*self.EF)
            b, c = gs*self.base.MN*self.y/self.EF, 2*g*self.n/W
            # The actual curvature changes under the negative-barrier control.
            Cy_actual = self.By*self.By/self.D+self.barrier_sign*self.delta_y
            C = Cy_actual/self.base.W0**2
            s = C-c*c/M2
            P1 = a*(M2+s)+g*g-b*b
            P0 = M2*(a*s-b*b)+a*c*c+g*g*s+2*b*g*c
            certificate = (C > 0 and P0 > 0 and (P1 >= 0 or P0 > P1*P1/(4*a)))
            return {"a": a, "b": b, "c": c, "s": s, "M2": M2,
                    "C_WW": C, "P1": P1, "P0": P0,
                    "P1_lower_bound": g*g-b*b-a*c*c/M2,
                    "all_mathematical_wave_numbers_positive": bool(certificate)}


def polynomial_examples(*, dps=80):
    precision(dps)
    with mp.workdps(dps):
        rows = []
        for y in (".75", ".90"):
            model, st, jet = inverse_potential_jet(y, "240", ".16", "-16")
            a2, a3, a4 = jet["coefficients"]
            # Independent exact quadratic minimum of q(z), z>=-1.
            if a4 > 0:
                at = max(mp.mpf(-1), -a3/(2*a4))
                minimum = a2+a3*at+a4*at*at
            else:
                at, minimum = None, None
            rows.append({"target_y": y, "target_K_MeV": "240",
                         "gomega": number(model.gomega),
                         "coefficients_MeV4": [number(x) for x in jet["coefficients"]],
                         "leading_coefficient_positive": bool(a4 > 0),
                         "globally_nonnegative_U": bool(minimum is not None and minimum >= 0),
                         "q_min_MeV4": None if minimum is None else number(minimum),
                         "z_at_q_min": None if at is None else number(at),
                         "ground_state_coexistence_proven_by_this_polynomial_check": False})
        return rows


def audit(*, dps=80, barrier_sign=1):
    precision(dps)
    with mp.workdps(dps):
        design = BindingDesign(dps=dps, barrier_sign=barrier_sign)
        cap = vector_ceiling()
        symbols = symbolic_checks()
        symbolic_pass = (isinstance(symbols, dict) and set(symbols) == REQUIRED_SYMBOLIC
                         and all(isinstance(row, dict) and row.get("passed") is True
                                 and row.get("residual") == "0" for row in symbols.values()))
        aux = design.envelope_state(design.y)
        # Finite differences of the fixed constructed U, not repeated fitting.
        Uy = mp.diff(design.potential, design.y)
        Uyy = mp.diff(design.potential, design.y, 2)
        f = design.base.fermi(design.n, design.y)
        C_actual = Uyy+design.base.MN**2*f["ns_m"]+3*design.Cv*design.n**2/design.y**4
        K_actual = 9*design.n*(design.D-design.By**2/C_actual)
        escale = design.mu*design.n
        residuals = {
            "auxiliary_density_at_liquid": aux["n"]/design.n-1,
            "liquid_common_tangent": design.grand_gap(design.n, design.y)/escale,
            "liquid_stationarity": (Uy-design.Uy)/escale,
            "liquid_target_curvature": (Uyy-design.Uyy)/max(abs(design.Uyy), 1),
            "K_target": (K_actual-design.K)/design.K,
            "vacuum_common_tangent": design.grand_gap(0, 1)/escale,
        }
        numeric_pass = (set(residuals) == {"auxiliary_density_at_liquid", "liquid_common_tangent",
                                          "liquid_stationarity", "liquid_target_curvature",
                                          "K_target", "vacuum_common_tangent"}
                        and all(mp.isfinite(x) and abs(x) < mp.mpf("1e-55") for x in residuals.values()))
        global_certificate = bool(design.beta > 0 and design.delta_y > 0 and design.Cv > 0
                                  and 0 < design.y < 1 and design.mu < design.base.MN)
        local = design.finite_wave()
        vacuum_mass2 = 8*design.beta*design.zs**2/design.base.W0**2
        return {
            "status": STATUS, "evidence_weight": 0,
            "mathematical_checks_passed": bool(symbolic_pass and numeric_pass and global_certificate
                                               and local["all_mathematical_wave_numbers_positive"]),
            "design_inputs": {"n_fm_minus3": number(design.n_fm3), "binding_MeV": number(design.binding),
                              "mu_MeV": number(design.mu), "K_MeV": number(design.K),
                              "y": number(design.y), "dps": dps,
                              "all_are_calibration_inputs_not_predictions": True,
                              "potential_uses_target_mu_explicitly": True},
            "original_parameters_unchanged": dict(INPUTS),
            "original_vector_no_go": {
                "gomega_original": INPUTS["gomega"], "gomega_ceiling_any_U": number(cap["gomega_max"]),
                "original_minimum_mu_MeV": number(cap["original_mu_min_MeV"]),
                "K_at_ceiling_with_finite_C_MeV": number(cap["finite_curvature_K_at_ceiling_MeV"]),
                "target_impossible_for_original_gomega_any_U": bool(design.base.gomega > cap["gomega_max"])},
            "constructed_potential": {"gomega_new": number(design.gomega),
                                      "Cv_MeV_minus2": number(design.Cv), "beta_MeV4": number(design.beta),
                                      "Delta_y_MeV4": number(design.delta_y),
                                      "K_actual_MeV": number(K_actual),
                                      "vacuum_scalar_mass_squared_MeV2": number(vacuum_mass2),
                                      "vacuum_scalar_mass_MeV": number(mp.sqrt(vacuum_mass2)) if vacuum_mass2 >= 0 else None,
                                      "original_vacuum_scalar_mass_MeV": number(mp.sqrt(2*design.base.lam)*design.base.W0),
                                      "new_vacuum_mass_is_unfitted_unvalidated_consequence": True,
                                      "original_potential_replaced": True,
                                      "coercive_if_beta_positive": bool(design.beta > 0)},
            "homogeneous_global_certificate": {
                "passed": global_certificate,
                "following_claims_conditional_on_passed": True,
                "inequality": "E(n,y)-mu_target*n >= beta*(y^2-1)^2*(y^2-y_star^2)^2 >= 0",
                "equality_states": "(n=0,y=1) and (n=n_target,y=y_target), on y>0",
                "bulk_vacuum_liquid_mixtures": "the common tangent is attained for average densities from 0 to n_target",
                "not_a_claim_for_finite_size_or_general_inhomogeneous_fields": True},
            "local_fluid_finite_wave": {k: (v if isinstance(v, bool) else number(v)) for k, v in local.items()},
            "envelope_regularity": {"onset_y": number(design.mu/design.base.MN),
                                    "density_exponent": "3/2", "potential_exponent": "5/2",
                                    "at_onset": "C2 but not C3; not an analytic microscopic scalar potential"},
            "formal_high_density_consequence": {
                "conditional_on_positive_beta": bool(design.beta > 0),
                "balance": "y ~ a*n^(1/5), a=(Cv/(8*beta))^(1/10); E ~ 5*beta*a^8*n^(8/5)",
                "w_and_relaxed_cs_squared_limit": "3/5",
                "reason_fermi_subleading": "m/kF ~ n^(-2/15), so Fermi energy ~ n^(4/3) is subleading",
                "not_the_original_quartic_limit_or_a_universal_law": True,
                "formal_mean_field_asymptotic_not_a_claim_of_EFT_validity_at_arbitrary_density": True},
            "existing_polynomial_examples": polynomial_examples(dps=dps),
            "residuals": {k: number(v) for k, v in residuals.items()},
            "symbolic_checks": symbols,
            "scope": [
                "A new explicitly engineered density-independent function U(y) and changed gomega, not the accepted quartic action.",
                "The original scalar/vector mass relations, M, W0, momega and degeneracy are retained.",
                "Only homogeneous T=0 normal mean-field matter and bulk volume mixtures are covered by the global energy inequality.",
                "The finite-k certificate is local at the fitted liquid point and only within the ideal-fluid closure.",
                "The new vacuum scalar mass is an unfitted consequence, not an empirical validation; it differs greatly from the original scalar mass.",
                "No surface energy, finite nuclei, pairing, quantum response, experimental confirmation or full bounce compatibility is established.",
            ],
        }


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--dps", type=int, default=80)
    try:
        args = parser.parse_args(argv)
        result = audit(dps=args.dps)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        print(json.dumps({"status": "invalid_or_failed_audit", "evidence_weight": 0,
                          "mathematical_checks_passed": False, "error": str(exc)}, allow_nan=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result["mathematical_checks_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
