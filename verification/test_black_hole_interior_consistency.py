"""Independent thermodynamic and semantic checks for the interior toy EOS."""

import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
from scipy.integrate import quad

import nvg_black_hole_interior as interior


def independent_energy(n, mass=None, degeneracy=2):
    """Adaptive integral in physical momentum, independent of production nodes."""
    mass = interior.M_star(n) if mass is None else mass
    k = (6 * math.pi**2 * n / degeneracy) ** (1 / 3) * interior.hbar_c
    integral = quad(lambda p: p*p * math.hypot(p, mass), 0, k,
                    epsabs=1e-8, epsrel=2e-12)[0]
    return degeneracy * integral / (2 * math.pi**2 * interior.hbar_c**3)


def derivative_stencil(function, point, step):
    fm2, fm1, f0, fp1, fp2 = [function(point + offset * step)
                              for offset in (-2, -1, 0, 1, 2)]
    first = (fm2 - 8*fm1 + 8*fp1 - fp2) / (12*step)
    second = (-fm2 + 16*fm1 - 30*f0 + 16*fp1 - fp2) / (12*step**2)
    return first, second


@pytest.mark.parametrize("ratio", [0.0, 0.005, 0.1, 0.5, 1.0, 2.0, 3.3, 5.0, 10.0, 20.0, 500.0, 1e6])
def test_fermi_surface_energy_against_high_precision_rounding(ratio):
    """Catch the legacy Python hypot last-bit drift, without a saved table."""
    import mpmath as mp

    density = ratio * interior.n_0
    mass = interior.M_star(density)
    result = interior.fermi_integrals(density, mass)
    with mp.workdps(120):
        # Convert the actual binary inputs exactly, not their decimal labels.
        k = mp.mpf(result["kf_mev"])
        m = mp.mpf(mass)
        expected = float(mp.sqrt(k*k + m*m))
    assert result["fermi_energy_mev"] == expected


@pytest.mark.parametrize("ratio", [0.005, 0.1, 0.5, 1.0, 2.0, 3.0, 3.3, 5.0, 20.0, 500.0, 1e6])
def test_energy_derivatives_determine_mu_pressure_and_sound_speed(ratio):
    n = ratio * interior.n_0
    state = interior.eos_state(n)
    independent = independent_energy(n)
    assert state["energy_mev_fm3"] == pytest.approx(independent, rel=2e-11)
    mu, second = derivative_stencil(independent_energy, n, n*0.002)
    assert state["chemical_potential_mev"] == pytest.approx(mu, rel=2e-8)
    assert state["energy_second_mev_fm3"] == pytest.approx(second, rel=2e-5)
    assert state["pressure_mev_fm3"] == pytest.approx(n*mu-independent, rel=2e-5, abs=2e-9)
    assert state["cs2"] == pytest.approx(n*second/mu, rel=2e-5, abs=2e-9)


@pytest.mark.parametrize("ratio", [0.01, 1.0, 3.0, 3.3, 10.0, 500.0])
def test_mass_derivatives_independently(ratio):
    n = ratio * interior.n_0
    mass, first, second = interior.mass_derivatives(n)
    numerical_first, numerical_second = derivative_stencil(interior.M_star, n, n*0.003)
    assert mass == interior.M_star(n)
    assert first == pytest.approx(numerical_first, rel=1e-8)
    assert second == pytest.approx(numerical_second, rel=1e-5)


@pytest.mark.parametrize("degeneracy", [2, 4])
@pytest.mark.parametrize("mass", [0.0, 1e-6, 0.1, 80.0, 939.0, 1e8])
def test_integrals_span_massless_and_nonrelativistic_limits(mass, degeneracy):
    n = interior.n_0
    state = interior.quasiparticle_eos(n, mass, 0.0, 0.0, degeneracy)
    assert state["energy_mev_fm3"] == pytest.approx(independent_energy(n, mass, degeneracy), rel=2e-11)
    k = state["kf_mev"]
    expected_cs2 = k*k / (3*(k*k + mass*mass))
    assert state["cs2"] == pytest.approx(expected_cs2, rel=2e-13)
    assert state["chemical_potential_mev"] == pytest.approx(math.hypot(k, mass))
    if mass == 0:
        assert state["energy_mev_fm3"] == pytest.approx(degeneracy*k**4 / (8*math.pi**2*interior.hbar_c**3))
        assert state["pressure_over_energy"] == pytest.approx(1/3)
    elif mass == 1e8:
        # The stable pressure remains positive even when n*mu-e nearly cancels.
        assert state["pressure_mev_fm3"] == pytest.approx(n*k*k/(5*mass), rel=1e-10)


@pytest.mark.parametrize("mass", [0.01, 80.0, 939.0, 1e6])
def test_scalar_integrals_are_independent_mass_derivatives(mass):
    n = interior.n_0
    result = interior.fermi_integrals(n, mass)
    k = result["kf_mev"]
    prefactor = 2 / (2*math.pi**2*interior.hbar_c**3)
    scalar = prefactor * quad(lambda p: p*p*mass/math.hypot(p, mass), 0, k,
                             epsabs=1e-10, epsrel=2e-11)[0]
    curvature = prefactor * quad(lambda p: p**4/math.hypot(p, mass)**3, 0, k,
                                epsabs=1e-10, epsrel=2e-11)[0]
    assert result["scalar_density_fm3"] == pytest.approx(scalar, rel=2e-8)
    assert result["mass_curvature_fm3_per_mev"] == pytest.approx(curvature, rel=2e-8)


def test_rearrangement_is_nonzero_and_instability_is_not_clipped():
    state = interior.eos_state(interior.n_0)
    assert state["chemical_rearrangement_mev"] < 0
    assert state["pressure_rearrangement_mev_fm3"] < 0
    assert state["pressure_mev_fm3"] < 0
    assert state["cs2"] < 0
    assert state["status"] == "mechanically_unstable"
    assert state["pressure_mev_fm3"] == pytest.approx(
        interior.n_0 * state["chemical_potential_mev"] - state["energy_mev_fm3"], abs=2e-11)
    # A synthetic smooth local mass law creates a deliberate acausal control.
    acausal = interior.quasiparticle_eos(interior.n_0, 939.0, 0.0, 1e7)
    assert acausal["cs2"] > 1
    assert acausal["status"] == "superluminal"


def test_extreme_representable_sound_speed_avoids_intermediate_overflow():
    state = interior.quasiparticle_eos(2.0, 939.0, 0.0, 1e308)
    assert math.isfinite(state["energy_second_mev_fm3"])
    assert math.isfinite(state["cs2"])
    assert state["cs2"] == pytest.approx(2.80001941021782e305, rel=2e-14)
    assert state["status"] == "superluminal"
    json.dumps(state, allow_nan=False)


def test_extreme_mass_derivative_square_is_applied_with_its_coefficient():
    state = interior.quasiparticle_eos(2.0, 939.0, 1e155, 0.0)
    # Decimal arithmetic independently checks the product whose unweighted
    # square cannot be represented as a float.
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 60
        coefficient = Decimal.from_float(state["mass_curvature_fm3_per_mev"])
        prime = Decimal.from_float(1e155)
        expected = float(coefficient * prime * prime)
    assert state["energy_second_mev_fm3"] == pytest.approx(expected, rel=2e-14)
    assert math.isfinite(state["cs2"])
    json.dumps(state, allow_nan=False)


@pytest.mark.parametrize("prime, second", [(1e308, 0.0), (0.0, 1e308)])
def test_nonrepresentable_derived_quantities_fail_closed(prime, second):
    with pytest.raises(ValueError, match="derived EOS field"):
        interior.quasiparticle_eos(3.0, 939.0, prime, second)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_mass_inputs_and_derivatives_fail_closed(invalid):
    for args in ((invalid, 0.0, 0.0), (939.0, invalid, 0.0), (939.0, 0.0, invalid)):
        with pytest.raises(ValueError, match="finite"):
            interior.quasiparticle_eos(interior.n_0, *args)


def test_vacuum_is_a_finite_one_sided_limit():
    state = interior.eos_state(0)
    assert state["energy_mev_fm3"] == state["pressure_mev_fm3"] == state["cs2"] == 0
    assert state["chemical_potential_mev"] == interior.M_N
    assert state["energy_second_mev_fm3"] is None
    assert state["status"] == "vacuum_one_sided_limit"
    assert abs(interior.eos_state(1e-12)["cs2"]) < 1e-7
    assert interior.quasiparticle_eos(0, 0, 0, 0)["cs2"] == 1/3
    assert interior.conformal_EOS(0) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_densities_are_rejected(invalid):
    for function in (interior.M_Omega, interior.M_current, interior.M_star,
                     interior.mass_derivatives, interior.energy_density,
                     interior.eos_state, interior.conformal_EOS):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            function(invalid)
    with pytest.raises(ValueError):
        interior.compute_state([invalid])


@pytest.mark.parametrize("degeneracy", [0, -2, 2.5, True, float("nan")])
def test_degeneracy_is_an_explicit_positive_integer(degeneracy):
    with pytest.raises(ValueError, match="degeneracy"):
        interior.fermi_integrals(interior.n_0, 939, degeneracy)


def test_kink_has_continuous_energy_but_distinct_one_sided_mu():
    kink = interior.CURRENT_MASS_KINK_DENSITY
    energy = interior.energy_density(kink)
    assert math.isfinite(energy)
    for function in (interior.mass_derivatives, interior.eos_state, interior.conformal_EOS):
        with pytest.raises(ValueError, match="kink"):
            function(kink)
    step = kink*1e-7
    left = interior.eos_state(kink-step)
    right = interior.eos_state(kink+step)
    left_slope = (energy - interior.energy_density(kink-step))/step
    right_slope = (interior.energy_density(kink+step)-energy)/step
    assert left_slope == pytest.approx(left["chemical_potential_mev"], rel=2e-7)
    assert right_slope == pytest.approx(right["chemical_potential_mev"], rel=2e-7)
    assert right_slope-left_slope > 50
    row = interior.compute_state([kink/interior.n_0])["eos_rows"][0]
    assert row["status"] == "undefined_thermodynamics_at_mass_kink"
    assert row["pressure_mev_fm3"] is None and row["cs2"] is None


def test_density_scale_is_not_a_bound_and_conformal_matter_is_not_de_sitter():
    assert interior.epsilon_max_estimate() == interior.epsilon_density_scale()
    state = interior.eos_state(1e6*interior.n_0)
    assert state["energy_mev_fm3"] > 1000*interior.epsilon_density_scale()
    assert state["pressure_over_energy"] == pytest.approx(1/3, abs=1e-6)
    assert state["cs2"] == pytest.approx(1/3, abs=1e-6)
    assert state["pressure_over_energy"] != pytest.approx(-1)


def test_hayward_crossover_requires_an_explicit_model_and_has_correct_scaling():
    with pytest.raises(ValueError, match="explicit"):
        interior.de_sitter_core_radius(10)
    epsilon = interior.epsilon_density_scale()
    radius = interior.de_sitter_core_radius(10, model="assumed_hayward")
    # Independent equivalent relation: r0^3=3*M*c^2/(4*pi*epsilon_core).
    expected = (3*10*interior.M_SUN_KG*interior.C_SI**2 /
                (4*math.pi*epsilon*interior.MEV_FM3_TO_PA)) ** (1/3) / 1000
    assert radius == pytest.approx(expected, rel=2e-15)
    assert interior.de_sitter_core_radius(80, model="assumed_hayward") == pytest.approx(2*radius)
    assert interior.de_sitter_core_radius(10, model="assumed_hayward", assumed_core_density=8*epsilon) == pytest.approx(radius/2)
    with pytest.raises(ValueError):
        interior.de_sitter_core_radius(0, model="assumed_hayward")
    with pytest.raises(ValueError):
        interior.de_sitter_core_radius(10, model="assumed_hayward", assumed_core_density=float("nan"))


def test_compute_state_statuses_are_calculated_and_json_is_finite():
    default = interior.compute_state()
    checks = default["sampled_checks"]
    assert checks["closure_within_1e_minus_11"] is True
    assert checks["mechanically_unstable_density_ratios"]
    assert checks["all_positive_samples_locally_stable_and_subluminal"] is False
    assert checks["differentiable_at_all_samples"] is False
    assert checks["energy_exceeds_density_scale_at_ratios"]
    stable = interior.compute_state([20, 100])
    assert stable["sampled_checks"]["all_positive_samples_locally_stable_and_subluminal"] is True
    assert stable["sampled_checks"]["differentiable_at_all_samples"] is True
    assert stable["sampled_checks"]["mechanically_unstable_density_ratios"] == []
    assert default["geometry"]["status"] == "not_computed_no_geometry_model"
    assert interior.compute_state(core_model="assumed_hayward")["geometry"]["radii"]
    assert json.loads(json.dumps(default, allow_nan=False)) == default
    assert interior.compute_state([])["sampled_checks"]["closure_within_1e_minus_11"] is False


def test_cli_uses_the_same_computed_state():
    path = Path(interior.__file__)
    process = subprocess.run([sys.executable, str(path), "--json"], check=True,
                             capture_output=True, text=True)
    assert json.loads(process.stdout) == interior.compute_state()
    text = subprocess.run([sys.executable, str(path)], check=True,
                          capture_output=True, text=True).stdout
    assert "mechanically_unstable" in text
    assert "undefined_thermodynamics_at_mass_kink" in text
    assert "Uncomputed: collapse dynamics; singularity resolution; exterior matching" in text
    assert "EXACTLY Schwarzschild" not in text
    assert "PASS" not in text
