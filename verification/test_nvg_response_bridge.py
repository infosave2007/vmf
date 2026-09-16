"""Cross-check distinct closures; do not promote matching formulas to data.

The binding polynomial below is a labelled thermodynamic counterexample,
not an NVG interaction, fitted parameter set or proposal for nuclear matter.
"""

import mpmath as mp
import pytest
import sympy as sp

import nvg_collisionless_response_audit as kinetic
import nvg_longitudinal_fluid_audit as fluid
from source_complete_scaling_saturation_audit import BulkModel


def _relative_error(left, right):
    assert mp.isfinite(left) and mp.isfinite(right)
    return abs(left-right)/max(abs(left), abs(right), mp.mpf("1e-70"))


@pytest.mark.parametrize("ratio", ["1", "10", "200"])
def test_two_closures_recover_same_static_hessian_and_thermodynamic_sound(ratio):
    with mp.workdps(80):
        f = fluid.coefficients(ratio, dps=80)
        k = kinetic.coefficients(ratio, dps=80)
        h = fluid.static_hessian(f["coefficients"], 0)
        pairs = ((f["W_over_W0"], k["y"]), (h[0][0], k["a"]+k["V"]),
                 (h[0][1], k["B"]), (h[1][1], k["C"]),
                 (f["mu_total"], k["mu"]),
                 (f["acoustic_cs2"], k["first_sound_squared"]))
        assert all(_relative_error(left, right) < mp.mpf("1e-60")
                   for left, right in pairs)
        # Static response and hydrodynamic sound follow the same Schur
        # complement, but this does not identify the collisionless pole.
        stiffness = h[0][0]-h[0][1]**2/h[1][1]
        assert _relative_error(stiffness, (1+k["F0"])/k["NF"]) < mp.mpf("1e-60")
        landau_sound = k["vF2"]*(1+k["F0"])*(1+k["F1"]/3)/3
        assert _relative_error(landau_sound, f["acoustic_cs2"]) < mp.mpf("1e-60")


@pytest.mark.parametrize("ratio", ["1", "10"])
def test_independent_re_solved_branch_derivatives(ratio):
    with mp.workdps(80):
        model = BulkModel()
        n = mp.mpf(ratio)*model.n0
        state = model.equilibrium(n)
        # Re-solve W at every derivative argument. Do not differentiate a
        # fixed-W pressure, or substitute the Hessian formula being tested.
        energy_prime = mp.diff(lambda nn: model.equilibrium(nn)["energy_total"], n)
        mu_prime = mp.diff(lambda nn: model.equilibrium(nn)["mu"], n)
        pressure_prime = mp.diff(lambda nn: model.equilibrium(nn)["pressure_total"], n)
        per_particle_prime = mp.diff(lambda nn: model.equilibrium(nn)["energy_total"]/nn, n)
        k = kinetic.coefficients(ratio, dps=80)
        pairs = ((energy_prime, state["mu"]),
                 (mu_prime, (1+k["F0"])/k["NF"]),
                 (pressure_prime, state["mu"]*k["first_sound_squared"]),
                 (per_particle_prime, state["pressure_total"]/n**2))
        assert all(_relative_error(left, right) < mp.mpf("1e-55")
                   for left, right in pairs)
        assert state["pressure_total"] > 0
        assert state["energy_total"]/n > model.MN


def test_exact_identities_used_by_smooth_branch_binding_obstruction():
    n = sp.symbols("n", positive=True)
    energy = sp.Function("energy")(n)
    mu = sp.diff(energy, n)
    pressure = n*mu-energy
    sound2 = n*sp.diff(mu, n)/mu
    assert sp.simplify(sp.diff(pressure, n)-mu*sound2) == 0
    assert sp.simplify(sp.diff(energy/n, n)-pressure/n**2) == 0
    # Together with P(0)=0, mu>0 and sound2>0 these identities give P>0
    # and increasing E/n. The integration argument is in the report;
    # symbolic equality alone is not an automatic proof of its premises.


def test_labelled_bound_polynomial_requires_nonmonotone_homogeneous_pressure():
    n, mass, attraction, repulsion = sp.symbols("n mass attraction repulsion", positive=True)
    energy = mass*n-attraction*n**2+repulsion*n**3
    pressure = sp.expand(n*sp.diff(energy, n)-energy)
    saturation = attraction/(2*repulsion)
    assert sp.simplify(pressure.subs(n, saturation)) == 0
    assert sp.simplify((energy/n-mass).subs(n, saturation)+attraction**2/(4*repulsion)) == 0
    assert sp.simplify(sp.diff(pressure, n).subs(n, attraction/(6*repulsion))
                       +attraction**2/(6*repulsion)) == 0
    # At saturation P'>0, but the intervening smooth branch has P'<0.
    # At saturation, positive chemical potential needs mass>a^2/(4b).
    # Positivity on the entire branch instead needs mass>a^2/(3b).
    assert sp.simplify(sp.diff(pressure, n).subs(n, saturation)
                       -attraction**2/(2*repulsion)) == 0


def test_missing_undamped_pole_is_not_negative_compressibility():
    with mp.workdps(80):
        k = kinetic.coefficients("200", dps=80)
        assert k["F0"] < 3*k["r"]
        assert kinetic.zero_sound_root(k["F0"], k["r"], dps=80) is None
        assert 1+k["F0"] > 0
        assert 1+k["F1"]/3 > 0
        assert 0 < k["first_sound_squared"] < 1
        assert fluid.coefficients("200", dps=80)["positive_energy"]["strict_positive_all_z"] is True
