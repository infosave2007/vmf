#!/usr/bin/env python3
"""Kinematic screen of declared NVG dark-sector proxies against LZ's 248-keV event.

This is deliberately a narrow, reproducible necessary-condition calculation.
It does not fit an LZ event or turn one event into evidence for NVG.  It asks
whether a pointlike nonrelativistic halo particle with a mass already declared
by an NVG extension could transfer the reported energy to xenon.

The primary screen derives a generous bound-halo laboratory-speed envelope
from the Standard Halo Model (SHM) parameters recommended for direct-detection
reporting.  Exact special-relativistic two-body identities are used for the
recoil endpoints and an optional exothermic diagnostic.  Nuclear form factors,
isotopes, detector response, rates, backgrounds, and likelihoods are outside
this screen.

The current NVG candidates screened here are:
  * the radial scalar and massive-vector vacuum modes derived from the bound
    source-complete action.  The action does not establish either as a stable
    cold-halo population; they are screened conditionally rather than promoted
    to dark-matter candidates;
  * the dark-neutron mass corridor from nvg_adm_bl_cogenesis.py;
  * the W-like excitation mass produced by nvg_relic_dark_matter.py, treated
    only as a conditional particle proxy because the bound action does not
    define it as a stable WIMP with a xenon-scattering interaction.

An external 1-TeV, 350-keV-splitting Higgsino point is a source-reading
comparator only.  It is not adopted by NVG.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_PATH = HERE / "data" / "lz_248kev_event_2026.json"
SCHEMA_VERSION = "nvg-lz-248kev-kinematic-audit.v5"
STATUS_PASS = "PASS_KINEMATIC_SCREEN_NO_CURRENT_NVG_PARTICLE_EXPLANATION"
STATUS_FAIL = "FAIL_KINEMATIC_SCREEN"
C_KM_S = Decimal("299792.458")
TOLERANCE = Decimal("1e-24")
# Consumer-side provenance invariants, not physical parameters. A producer
# cannot silently relabel this retired input as a prediction or move it.
DARK_NEUTRON_INPUT_ARTIFACT = "verification/data/legacy_dark_neutron_cogenesis_inputs.json"
DARK_NEUTRON_INPUT_STATUS = "LEGACY_UNDOCUMENTED_CONDITIONAL_INPUTS_NOT_MECHANISM"

import nvg_adm_bl_cogenesis as dark_neutron  # noqa: E402
import nvg_relic_dark_matter as relic  # noqa: E402
import source_complete_solution_audit as source_complete  # noqa: E402


def _historical_dark_neutron_input_passport() -> dict[str, str]:
    """Verify the legacy cogenesis input identity before using its corridor.

    The corridor remains only a conditional historical diagnostic. This check
    prevents a downstream result from silently inheriting source-level
    constants or a substituted passport while retaining the same claim
    boundary.
    """

    passport = dark_neutron.historical_input_passport()
    if not isinstance(passport, Mapping) or set(passport) != {
        "input_artifact",
        "input_sha256",
        "input_status",
    }:
        raise ValueError("dark-neutron producer returned an invalid historical input passport")
    producer_artifact = dark_neutron.INPUT_PATH.relative_to(dark_neutron.ROOT).as_posix()
    if producer_artifact != DARK_NEUTRON_INPUT_ARTIFACT:
        raise ValueError("dark-neutron producer moved its historical input passport; review required")
    expected_sha256 = hashlib.sha256((ROOT / DARK_NEUTRON_INPUT_ARTIFACT).read_bytes()).hexdigest()
    if (
        passport["input_artifact"] != DARK_NEUTRON_INPUT_ARTIFACT
        or passport["input_sha256"] != expected_sha256
        or passport["input_status"] != DARK_NEUTRON_INPUT_STATUS
    ):
        raise ValueError("dark-neutron producer historical input passport does not match live bytes")
    return {
        "input_artifact": DARK_NEUTRON_INPUT_ARTIFACT,
        "input_sha256": expected_sha256,
        "input_status": DARK_NEUTRON_INPUT_STATUS,
    }


def _decimal(value: Any, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"not a finite decimal: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"not a finite decimal: {value!r}")
    if positive and result <= 0:
        raise ValueError(f"must be positive: {value!r}")
    if nonnegative and result < 0:
        raise ValueError(f"must be nonnegative: {value!r}")
    return result


def _number(value: Decimal) -> str:
    value = _decimal(value)
    with localcontext() as context:
        context.prec = 60
        return format(+value, ".30g")


def _payload_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _decimal_vector(value: Any, *, name: str) -> tuple[Decimal, Decimal, Decimal]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-component vector")
    return tuple(_decimal(component) for component in value)  # type: ignore[return-value]


def _load_data(path: Path = DATA_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "lz-248kev-event-input.v3":
        raise ValueError("unrecognized LZ event input schema")
    for key in ("sources", "event", "target", "bound_standard_halo_model_screen", "external_higgsino_comparator"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"LZ event input misses object {key}")
    sources = payload["sources"]
    for key in ("lz_extended_nuclear_recoil_search", "higgsino_comparator", "recommended_standard_halo_model"):
        if not isinstance(sources.get(key), dict) or not str(sources[key].get("source_url", "")).startswith("https://"):
            raise ValueError(f"LZ event input misses source URL for {key}")
    event, target, halo, comparator = (
        payload[key]
        for key in ("event", "target", "bound_standard_halo_model_screen", "external_higgsino_comparator")
    )
    for value in (
        event.get("recoil_energy_keV"),
        target.get("natural_xenon_atomic_mass_u"),
        target.get("atomic_mass_unit_GeV"),
        halo.get("galactic_escape_speed_km_s"),
        halo.get("earth_orbital_velocity_basis", {}).get("mean_speed_km_s") if isinstance(halo.get("earth_orbital_velocity_basis"), dict) else None,
        comparator.get("mass_GeV"),
        comparator.get("endothermic_mass_splitting_keV"),
    ):
        _decimal(value, positive=True)
    _decimal_vector(halo.get("local_standard_of_rest_velocity_km_s"), name="local_standard_of_rest_velocity_km_s")
    _decimal_vector(halo.get("solar_peculiar_velocity_km_s"), name="solar_peculiar_velocity_km_s")
    orbital = halo.get("earth_orbital_velocity_basis")
    if not isinstance(orbital, dict):
        raise ValueError("bound SHM input misses earth_orbital_velocity_basis")
    _decimal(orbital.get("mean_speed_km_s"), positive=True)
    _decimal_vector(orbital.get("cosine_coefficients"), name="earth_orbital_velocity_basis.cosine_coefficients")
    _decimal_vector(orbital.get("sine_coefficients"), name="earth_orbital_velocity_basis.sine_coefficients")
    if _decimal(event.get("global_significance_sigma"), positive=True) >= Decimal("5"):
        raise ValueError("input incorrectly represents the LZ result as discovery-level")
    if _decimal(event.get("maximum_local_significance_sigma"), positive=True) < _decimal(event.get("global_significance_sigma"), positive=True):
        raise ValueError("local significance must not be below declared global significance")
    return payload


def _beta(speed_km_s: Decimal) -> Decimal:
    speed_km_s = _decimal(speed_km_s, nonnegative=True)
    beta = speed_km_s / C_KM_S
    if beta >= 1:
        raise ValueError("speed must be below c")
    return beta


def _gamma(beta: Decimal) -> Decimal:
    beta = _decimal(beta, nonnegative=True)
    if beta >= 1:
        raise ValueError("beta must be below one")
    return Decimal(1) / (Decimal(1) - beta * beta).sqrt()


def reduced_mass_GeV(mass_GeV: Decimal, target_mass_GeV: Decimal) -> Decimal:
    mass_GeV = _decimal(mass_GeV, positive=True)
    target_mass_GeV = _decimal(target_mass_GeV, positive=True)
    return mass_GeV * target_mass_GeV / (mass_GeV + target_mass_GeV)


def maximum_recommended_earth_orbital_speed_km_s(orbital: Mapping[str, Any]) -> Decimal:
    """Maximise |v_bar (a cos t + b sin t)| from the cited SHM expression."""
    scale = _decimal(orbital["mean_speed_km_s"], positive=True)
    cosine = _decimal_vector(orbital["cosine_coefficients"], name="earth_orbital_velocity_basis.cosine_coefficients")
    sine = _decimal_vector(orbital["sine_coefficients"], name="earth_orbital_velocity_basis.sine_coefficients")
    a2 = sum(component * component for component in cosine)
    b2 = sum(component * component for component in sine)
    ab = sum(a * b for a, b in zip(cosine, sine))
    largest_gram_eigenvalue = (a2 + b2 + ((a2 - b2) * (a2 - b2) + Decimal(4) * ab * ab).sqrt()) / Decimal(2)
    return scale * largest_gram_eigenvalue.sqrt()


def bound_shm_speed_cap_km_s(halo: Mapping[str, Any]) -> Decimal:
    """Return v_esc + |v_0 + v_pec| + max_t|v_earth(t)| from SHM inputs.

    This is an upper envelope by the triangle inequality, not a likelihood
    assumption or universal velocity cutoff.
    """
    v_esc = _decimal(halo["galactic_escape_speed_km_s"], positive=True)
    orbital = halo["earth_orbital_velocity_basis"]
    if not isinstance(orbital, Mapping):
        raise ValueError("bound SHM input misses earth_orbital_velocity_basis")
    earth = maximum_recommended_earth_orbital_speed_km_s(orbital)
    v0 = _decimal_vector(halo["local_standard_of_rest_velocity_km_s"], name="local_standard_of_rest_velocity_km_s")
    vpec = _decimal_vector(halo["solar_peculiar_velocity_km_s"], name="solar_peculiar_velocity_km_s")
    solar_motion = tuple(a + b for a, b in zip(v0, vpec))
    solar_speed = sum(component * component for component in solar_motion).sqrt()
    return v_esc + solar_speed + earth


def exact_elastic_recoil_maximum_keV(mass_GeV: Decimal, target_mass_GeV: Decimal, speed_km_s: Decimal) -> Decimal:
    """Exact 2->2 maximum elastic recoil for a target initially at rest."""
    mass_GeV = _decimal(mass_GeV, positive=True)
    target_mass_GeV = _decimal(target_mass_GeV, positive=True)
    beta = _beta(speed_km_s)
    gamma = _gamma(beta)
    momentum = gamma * mass_GeV * beta
    invariant_s = mass_GeV * mass_GeV + target_mass_GeV * target_mass_GeV + Decimal(2) * target_mass_GeV * gamma * mass_GeV
    return Decimal(2) * target_mass_GeV * momentum * momentum / invariant_s * Decimal("1e6")


def elastic_recoil_maximum_nr_keV(mass_GeV: Decimal, target_mass_GeV: Decimal, speed_km_s: Decimal) -> Decimal:
    """Leading nonrelativistic elastic recoil endpoint, retained as a cross-check."""
    mu = reduced_mass_GeV(mass_GeV, target_mass_GeV)
    beta = _beta(speed_km_s)
    return Decimal(2) * mu * mu * beta * beta / _decimal(target_mass_GeV, positive=True) * Decimal("1e6")


def exact_elastic_minimum_speed_km_s(mass_GeV: Decimal, target_mass_GeV: Decimal, recoil_keV: Decimal) -> Decimal:
    """Exact threshold speed for elastic scattering at a specified recoil.

    Solves the forward-scattering endpoint analytically.  It is used even for
    the deliberately impossible light-proxy extrapolations, so the resulting
    speed is not interpreted as a halo prediction.
    """
    mass_GeV = _decimal(mass_GeV, positive=True)
    target_mass_GeV = _decimal(target_mass_GeV, positive=True)
    recoil = _decimal(recoil_keV, positive=True) / Decimal("1e6")
    q = (recoil * (recoil + Decimal(2) * target_mass_GeV)).sqrt()
    r = recoil * target_mass_GeV / mass_GeV
    beta = (q * recoil + r * (q * q + r * r - recoil * recoil).sqrt()) / (q * q + r * r)
    if beta <= 0 or beta >= 1:
        raise ValueError("specified recoil has no subluminal elastic threshold for this mass")
    return beta * C_KM_S


def exact_elastic_mass_threshold_GeV(target_mass_GeV: Decimal, recoil_keV: Decimal, speed_km_s: Decimal) -> Decimal:
    """Smallest elastic incident mass compatible with the recoil/speed pair."""
    target_mass_GeV = _decimal(target_mass_GeV, positive=True)
    recoil = _decimal(recoil_keV, positive=True) / Decimal("1e6")
    beta = _beta(speed_km_s)
    gamma = _gamma(beta)
    q = (recoil * (recoil + Decimal(2) * target_mass_GeV)).sqrt()
    denominator = gamma * (beta * q - recoil)
    if denominator <= 0:
        raise ValueError("declared speed cannot produce recoil for any finite incident mass")
    return recoil * target_mass_GeV / denominator


def exact_delta_interval_at_recoil_keV(
    initial_mass_GeV: Decimal,
    target_mass_GeV: Decimal,
    recoil_keV: Decimal,
    speed_km_s: Decimal,
) -> tuple[Decimal, Decimal]:
    """Exact allowed interval for delta=m_final-m_initial at fixed recoil/speed.

    It follows from energy--momentum conservation for a free target nucleus
    initially at rest.  A negative delta is an exothermic transition.  This
    diagnostic deliberately does not add an NVG excited state, an interaction,
    a lifetime, a halo population, a rate, or detector response.
    """
    initial_mass_GeV = _decimal(initial_mass_GeV, positive=True)
    target_mass_GeV = _decimal(target_mass_GeV, positive=True)
    recoil = _decimal(recoil_keV, positive=True) / Decimal("1e6")
    beta = _beta(speed_km_s)
    gamma = _gamma(beta)
    q = (recoil * (recoil + Decimal(2) * target_mass_GeV)).sqrt()
    initial_energy = gamma * initial_mass_GeV
    initial_momentum = gamma * initial_mass_GeV * beta
    final_energy = initial_energy - recoil
    if final_energy <= 0:
        raise ValueError("target recoil exceeds incident total energy")

    def endpoint(cosine: Decimal) -> Decimal:
        radicand = final_energy * final_energy - (
            initial_momentum * initial_momentum
            + q * q
            - Decimal(2) * initial_momentum * q * cosine
        )
        if radicand < 0:
            raise ValueError("no physical final state at the requested endpoint")
        return (radicand.sqrt() - initial_mass_GeV) * Decimal("1e6")

    delta_minus = endpoint(Decimal(-1))
    delta_plus = endpoint(Decimal(1))
    if delta_minus > delta_plus:
        raise AssertionError("exact delta interval is reversed")
    return delta_minus, delta_plus


def exact_heavier_excitation_release_interval_for_declared_ground_state_keV(
    ground_mass_GeV: Decimal,
    target_mass_GeV: Decimal,
    recoil_keV: Decimal,
    speed_km_s: Decimal,
) -> tuple[Decimal, Decimal]:
    """Return the exact exothermic-release window for ``X* -> X``.

    This is deliberately different from evaluating ``delta`` at a fixed
    initial mass.  Here the already declared mass is treated as a prospective
    *ground-state* mass ``m_g`` and a hypothetical incoming state is

        m_i = m_g + Delta,  m_f = m_g,  Delta > 0.

    At each two-body endpoint ``cos(theta)=+/-1``, energy--momentum
    conservation gives a quadratic for ``m_i``:

        m_i^2 + b_eta m_i - (m_g^2 + 2*M*E_R) = 0,
        b_eta = 2*gamma*(eta*beta*q - E_R),
        q = sqrt(E_R*(E_R + 2*M)), eta in {-1,+1}.

    The two positive roots define the inclusive release interval.  It is a
    kinematic target only: it creates neither the excited state nor a halo
    abundance, lifetime, scattering operator, detector response, or rate.
    """
    # This helper is public and its exact endpoint reconstruction must not
    # depend on whichever Decimal precision the caller happened to select.
    with localcontext() as context:
        context.prec = 70
        return _exact_heavier_excitation_release_interval_for_declared_ground_state_keV(
            ground_mass_GeV, target_mass_GeV, recoil_keV, speed_km_s
        )


def _exact_heavier_excitation_release_interval_for_declared_ground_state_keV(
    ground_mass_GeV: Decimal,
    target_mass_GeV: Decimal,
    recoil_keV: Decimal,
    speed_km_s: Decimal,
) -> tuple[Decimal, Decimal]:
    """Precision-controlled implementation for the public transition helper."""
    ground_mass_GeV = _decimal(ground_mass_GeV, positive=True)
    target_mass_GeV = _decimal(target_mass_GeV, positive=True)
    recoil = _decimal(recoil_keV, positive=True) / Decimal("1e6")
    beta = _beta(speed_km_s)
    gamma = _gamma(beta)
    q = (recoil * (recoil + Decimal(2) * target_mass_GeV)).sqrt()
    releases: list[Decimal] = []
    for eta in (Decimal(-1), Decimal(1)):
        coefficient = Decimal(2) * gamma * (eta * beta * q - recoil)
        discriminant = coefficient * coefficient + Decimal(4) * (
            ground_mass_GeV * ground_mass_GeV + Decimal(2) * target_mass_GeV * recoil
        )
        initial_mass = (-coefficient + discriminant.sqrt()) / Decimal(2)
        release_keV = (initial_mass - ground_mass_GeV) * Decimal("1e6")
        if release_keV <= 0:
            raise ValueError("declared ground state has no positive exothermic excitation window")

        # Reconstruct the corresponding exact endpoint.  eta=-1 represents
        # cos(theta)=-1 and eta=+1 represents cos(theta)=+1.
        delta_minus, delta_plus = exact_delta_interval_at_recoil_keV(
            initial_mass, target_mass_GeV, recoil_keV, speed_km_s
        )
        endpoint_delta = delta_minus if eta == Decimal(-1) else delta_plus
        if abs(endpoint_delta + release_keV) > TOLERANCE:
            raise AssertionError("exothermic excitation endpoint reconstruction failed")
        releases.append(release_keV)
    lower, upper = min(releases), max(releases)
    midpoint_release = (lower + upper) / Decimal(2)
    midpoint_initial_mass = ground_mass_GeV + midpoint_release / Decimal("1e6")
    delta_minus, delta_plus = exact_delta_interval_at_recoil_keV(
        midpoint_initial_mass, target_mass_GeV, recoil_keV, speed_km_s
    )
    if not delta_minus <= -midpoint_release <= delta_plus:
        raise AssertionError("exothermic excitation interval midpoint is not physical")
    return lower, upper


def exact_minimum_speed_for_endothermic_delta_km_s(
    mass_GeV: Decimal,
    target_mass_GeV: Decimal,
    recoil_keV: Decimal,
    splitting_keV: Decimal,
    *,
    upper_speed_km_s: Decimal = Decimal("100000"),
) -> Decimal:
    """Bisection threshold for a positive exact inelastic splitting.

    The function is used only for the external comparator.  It fails closed
    when the requested event cannot be reached below the explicitly supplied
    upper speed.
    """
    splitting_keV = _decimal(splitting_keV, nonnegative=True)
    upper_speed_km_s = _decimal(upper_speed_km_s, positive=True)
    if upper_speed_km_s >= C_KM_S:
        upper_speed_km_s = C_KM_S * Decimal("0.999999999999")
    low = Decimal(0)
    try:
        _, upper_delta = exact_delta_interval_at_recoil_keV(mass_GeV, target_mass_GeV, recoil_keV, upper_speed_km_s)
    except ValueError as exc:
        raise ValueError("requested inelastic threshold has no physical upper endpoint") from exc
    if upper_delta < splitting_keV:
        raise ValueError("specified splitting cannot reach the recoil below the declared upper speed")
    for _ in range(180):
        midpoint = (low + upper_speed_km_s) / Decimal(2)
        _, midpoint_delta = exact_delta_interval_at_recoil_keV(mass_GeV, target_mass_GeV, recoil_keV, midpoint)
        if midpoint_delta >= splitting_keV:
            upper_speed_km_s = midpoint
        else:
            low = midpoint
    return upper_speed_km_s


def _candidate_row(
    *,
    candidate_id: str,
    mass_GeV: Decimal,
    mass_origin: str,
    target_mass_GeV: Decimal,
    recoil_keV: Decimal,
    speed_cap_km_s: Decimal,
    caveat: str,
) -> dict[str, Any]:
    exact_emax = exact_elastic_recoil_maximum_keV(mass_GeV, target_mass_GeV, speed_cap_km_s)
    nr_emax = elastic_recoil_maximum_nr_keV(mass_GeV, target_mass_GeV, speed_cap_km_s)
    vmin = exact_elastic_minimum_speed_km_s(mass_GeV, target_mass_GeV, recoil_keV)
    beta = _beta(speed_cap_km_s)
    kinetic_cap_keV = (_gamma(beta) - Decimal(1)) * mass_GeV * Decimal("1e6")
    delta_minus, delta_plus = exact_delta_interval_at_recoil_keV(mass_GeV, target_mass_GeV, recoil_keV, speed_cap_km_s)
    compatible = delta_minus <= 0 <= delta_plus
    release_low, release_high = exact_heavier_excitation_release_interval_for_declared_ground_state_keV(
        mass_GeV, target_mass_GeV, recoil_keV, speed_cap_km_s
    )
    if compatible or release_low <= 0 or release_high <= release_low:
        raise AssertionError("light-state kinematic classification is inconsistent")
    excitation = {
        "declared_ground_state_mass_GeV": _number(mass_GeV),
        "minimum_required_release_MeV": _number(release_low / Decimal("1000")),
        "maximum_required_release_MeV": _number(release_high / Decimal("1000")),
        "incoming_excited_mass_minimum_GeV": _number(mass_GeV + release_low / Decimal("1e6")),
        "incoming_excited_mass_maximum_GeV": _number(mass_GeV + release_high / Decimal("1e6")),
        "transition_definition": "m_in=m_declared_ground+Delta, m_out=m_declared_ground, Delta>0",
        "interpretation": "This is the exact kinematic window only if a new long-lived incoming excitation decays to the already declared mass. Current NVG supplies no such partner, transition operator, lifetime, halo fraction, rate, or detector response.",
    }
    return {
        "candidate_id": candidate_id,
        "mass_GeV": _number(mass_GeV),
        "mass_origin": mass_origin,
        "exact_elastic_recoil_maximum_at_speed_cap_keV": _number(exact_emax),
        "nonrelativistic_elastic_recoil_maximum_at_speed_cap_keV": _number(nr_emax),
        "elastic_endpoint_nr_relative_difference": _number(abs(exact_emax - nr_emax) / exact_emax),
        "exact_elastic_minimum_speed_for_event_km_s": _number(vmin),
        "elastic_speed_cap_ratio": _number(vmin / speed_cap_km_s),
        "initial_kinetic_energy_at_speed_cap_keV": _number(kinetic_cap_keV),
        "formal_delta_interval_if_declared_mass_is_treated_as_incoming_state_keV": {
            "minimum": _number(delta_minus),
            "maximum": _number(delta_plus),
            "zero_elastic_splitting_is_allowed": compatible,
            "interpretation": "This fixed-initial-mass diagnostic is retained to expose the exact two-body endpoints. It is not the usual proposal of a new heavier excitation decaying to the declared state; use hypothetical_heavier_excitation_to_declared_ground_state instead for that scenario.",
        },
        "hypothetical_heavier_excitation_to_declared_ground_state": excitation,
        "elastic_kinematically_accessible_under_screen": compatible,
        "classification": (
            "KINEMATICALLY_INCOMPATIBLE_WITH_DECLARED_BOUND_SHM_SCREEN"
            if not compatible
            else "KINEMATICALLY_ACCESSIBLE_NOT_A_RATE_OR_LIKELIHOOD_RESULT"
        ),
        "caveat": caveat,
    }


def _build_result(data_path: Path = DATA_PATH) -> dict[str, Any]:
    with localcontext() as context:
        context.prec = 70
        data = _load_data(data_path)
        event, target, halo, comparator = (
            data[key]
            for key in ("event", "target", "bound_standard_halo_model_screen", "external_higgsino_comparator")
        )
        recoil = _decimal(event["recoil_energy_keV"], positive=True)
        target_mass = _decimal(target["natural_xenon_atomic_mass_u"], positive=True) * _decimal(target["atomic_mass_unit_GeV"], positive=True)
        speed_cap = bound_shm_speed_cap_km_s(halo)

        dark_neutron_input = _historical_dark_neutron_input_passport()
        dark_neutron_low_MeV = _decimal(dark_neutron.M_N, positive=True) - _decimal(dark_neutron.S_N_BE9, positive=True)
        dark_neutron_high_MeV = _decimal(dark_neutron.M_P, positive=True) + _decimal(dark_neutron.M_E, positive=True)
        if dark_neutron_high_MeV <= dark_neutron_low_MeV:
            raise ValueError("dark-neutron corridor is invalid")
        relic_state = relic.run_dm_verification()
        w_proxy_mass_GeV = _decimal(relic_state["m_W"], positive=True) / Decimal("1000")
        source_scalar_mass_GeV = _decimal(source_complete.PARAMS.scalar_mass, positive=True) / Decimal("1000")
        source_vector_mass_GeV = _decimal(source_complete.PARAMS.q_phi * source_complete.PARAMS.W0, positive=True) / Decimal("1000")
        expected_vector_mass_GeV = _decimal(source_complete.PARAMS.m_omega, positive=True) / Decimal("1000")
        if abs(source_vector_mass_GeV - expected_vector_mass_GeV) > TOLERANCE:
            raise ValueError("source-complete vector mass no longer matches its bound Higgs calibration")

        candidates = [
            _candidate_row(
                candidate_id="source_complete_massive_vector_mode",
                mass_GeV=source_vector_mass_GeV,
                mass_origin="source_complete_solution_audit.py: m_A=q_phi*W0=m_omega; quadratic massive-vector mode of the bound action",
                target_mass_GeV=target_mass,
                recoil_keV=recoil,
                speed_cap_km_s=speed_cap,
                caveat="The source-complete action derives this vacuum mode, but does not establish it as a stable cosmological cold-halo particle or calculate a xenon rate; this is only its conditional free-target kinematic screen.",
            ),
            _candidate_row(
                candidate_id="source_complete_scalar_radial_mode",
                mass_GeV=source_scalar_mass_GeV,
                mass_origin="source_complete_solution_audit.py: m_sigma=sqrt(2*lambda)*W0; radial scalar mode of the bound action",
                target_mass_GeV=target_mass,
                recoil_keV=recoil,
                speed_cap_km_s=speed_cap,
                caveat="The source-complete action derives this vacuum mode, but does not establish it as a stable cosmological cold-halo particle or calculate a xenon rate; this is only its conditional free-target kinematic screen.",
            ),
            _candidate_row(
                candidate_id="nvg_dark_neutron_lower_corridor_edge",
                mass_GeV=dark_neutron_low_MeV / Decimal("1000"),
                mass_origin=(
                    "nvg_adm_bl_cogenesis.py plus its explicit "
                    f"{dark_neutron_input['input_artifact']} passport: M_N-S_N_BE9; "
                    "speculative B-L extension, not a source-complete NVG particle"
                ),
                target_mass_GeV=target_mass,
                recoil_keV=recoil,
                speed_cap_km_s=speed_cap,
                caveat="No xenon-scattering amplitude or event-rate likelihood is supplied by the current NVG extension; this screen is kinematic only.",
            ),
            _candidate_row(
                candidate_id="nvg_dark_neutron_upper_corridor_edge",
                mass_GeV=dark_neutron_high_MeV / Decimal("1000"),
                mass_origin=(
                    "nvg_adm_bl_cogenesis.py plus its explicit "
                    f"{dark_neutron_input['input_artifact']} passport: M_P+M_E; "
                    "speculative B-L extension, not a source-complete NVG particle"
                ),
                target_mass_GeV=target_mass,
                recoil_keV=recoil,
                speed_cap_km_s=speed_cap,
                caveat="Screening both corridor endpoints bounds all masses in the narrow declared corridor for elastic recoil.",
            ),
            _candidate_row(
                candidate_id="nvg_W_like_relic_excitation_proxy",
                mass_GeV=w_proxy_mass_GeV,
                mass_origin="nvg_relic_dark_matter.py: m_W inferred after inputting Omega_DM; conditional excitation proxy, not a defined stable WIMP",
                target_mass_GeV=target_mass,
                recoil_keV=recoil,
                speed_cap_km_s=speed_cap,
                caveat="The current action does not provide a stable-particle spectrum or xenon-scattering Lagrangian for this proxy; kinematic failure is conditional on treating it as a cold-halo particle.",
            ),
        ]
        external_mass = _decimal(comparator["mass_GeV"], positive=True)
        external_delta = _decimal(comparator["endothermic_mass_splitting_keV"], positive=True)
        external_delta_interval = exact_delta_interval_at_recoil_keV(external_mass, target_mass, recoil, speed_cap)
        external_vmin = exact_minimum_speed_for_endothermic_delta_km_s(
            external_mass,
            target_mass,
            recoil,
            external_delta,
            upper_speed_km_s=speed_cap,
        )
        threshold_mass = exact_elastic_mass_threshold_GeV(target_mass, recoil, speed_cap)
        threshold_emax = exact_elastic_recoil_maximum_keV(threshold_mass, target_mass, speed_cap)

        candidate_failures = [not row["elastic_kinematically_accessible_under_screen"] for row in candidates]
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS_PASS,
            "independent_evidence_weight": 0.0,
            "fit_used": False,
            "external_event_role": "observational comparison input only; neither event energy nor significance is fitted back into NVG",
            "input_artifact": data_path.relative_to(ROOT).as_posix(),
            "input_artifact_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "conditional_input_passports": {
                "dark_neutron_cogenesis": dark_neutron_input,
            },
            "source_provenance": data["sources"],
            "method": {
                "exact_elastic_endpoint_formula": "E_R,max=2*m_A*(gamma*m_chi*beta)^2/s, s=m_chi^2+m_A^2+2*m_A*gamma*m_chi",
                "exact_fixed_recoil_delta_formula": "q=sqrt(E_R*(E_R+2*m_A)); delta_-=sqrt((E_i-E_R)^2-(p_i+q)^2)-m_i at cos(theta)=-1; delta_+=sqrt((E_i-E_R)^2-(p_i-q)^2)-m_i at cos(theta)=+1",
                "exact_heavier_excitation_formula": "m_in=m_g+Delta, m_out=m_g; endpoint roots satisfy m_in^2+2*gamma*(eta*beta*q-E_R)*m_in-(m_g^2+2*m_A*E_R)=0 for eta=-1,+1; Delta=m_in-m_g",
                "nonrelativistic_cross_check": "E_R,max,NR=2*mu_chiA^2*v^2/m_A",
                "scope": "free-nucleus two-body kinematic necessary condition only",
            },
            "declared_screen": {
                "event_recoil_keV": _number(recoil),
                "target_mass_GeV": _number(target_mass),
                "bound_shm_speed_cap_km_s": _number(speed_cap),
                "speed_cap_formula": halo["formula"],
                "speed_cap_interpretation": halo["interpretation"],
                "minimum_exact_elastic_particle_mass_at_speed_cap_GeV": _number(threshold_mass),
                "threshold_is_not_a_fit": True,
            },
            "current_nvg_candidate_screens": candidates,
            "external_comparator_not_adopted_by_nvg": {
                "mass_GeV": _number(external_mass),
                "endothermic_mass_splitting_keV": _number(external_delta),
                "exact_delta_interval_for_fixed_recoil_at_speed_cap_keV": {
                    "minimum": _number(external_delta_interval[0]),
                    "maximum": _number(external_delta_interval[1]),
                },
                "minimum_exact_speed_for_LZ_event_km_s": _number(external_vmin),
                "speed_cap_ratio": _number(external_vmin / speed_cap),
                "kinematically_accessible_under_same_screen": external_delta_interval[0] <= external_delta <= external_delta_interval[1],
                "interpretation": "Shows why the published heavy inelastic benchmark is a qualitatively different kinematic regime. It neither validates the benchmark nor supplies an NVG particle.",
            },
            "what_current_nvg_cannot_claim": [
                "The current source-complete action does not contain a derived stable ~TeV dark particle or an excited partner with a ~350-keV splitting.",
                "The source-complete radial and vector vacuum modes are not established as a long-lived, cosmologically abundant cold-halo population.",
                "No current NVG module matches the action couplings to xenon isotopes or computes an LZ detector response, halo integral, background model, recoil spectrum, event count, or profile likelihood.",
                "Primordial-black-hole and topological-defect proposals are not particle-recoil models here; without their transport and detector response they are not evaluated by this two-body screen.",
            ],
            "minimal_non_ad_hoc_extension_requirements": [
                "Derive a stable field excitation and all of its mass eigenstates from an explicit action, rather than inserting a new particle mass.",
                "Derive the ordinary-matter coupling and calculate the xenon nuclear response, including isotope dependence and form factors.",
                "Derive or independently constrain the cosmological abundance and local halo phase-space distribution before calculating rates.",
                "Use the public LZ response/data release and nuisance likelihood to make a forward prediction; do not tune the mass splitting to the single event.",
            ],
            "interpretation_limits": [
                "LZ reports one event of interest with 2.6-sigma global and 3.4-sigma maximum local significance, not a dark-matter discovery.",
                "The derived SHM/bound-halo cap is a deliberately generous triangle-inequality envelope, not a universal speed theorem.",
                "Kinematic incompatibility of a declared proxy under this screen does not falsify all NVG variants, relativistic populations, or an as-yet-unwritten interaction sector.",
                "The exact exothermic intervals identify only a necessary new transition scale; they do not furnish such a state, a rate, or an LZ explanation.",
                "Kinematic accessibility of the external comparator is necessary but far from sufficient: it does not calculate a rate or establish an explanation.",
            ],
            "controls": {
                "candidate_corridor_order": {"pass": dark_neutron_low_MeV < dark_neutron_high_MeV},
                "dark_neutron_corridor_has_explicit_legacy_conditional_input_passport": {
                    "pass": dark_neutron_input["input_status"] == DARK_NEUTRON_INPUT_STATUS,
                },
                "exact_elastic_inverse_threshold": {"absolute_error_keV": _number(abs(threshold_emax - recoil)), "pass": abs(threshold_emax - recoil) < TOLERANCE},
                "nonrelativistic_endpoint_is_negligibly_close_at_speed_cap": {
                    "maximum_relative_difference": _number(max(Decimal(row["elastic_endpoint_nr_relative_difference"]) for row in candidates)),
                    "pass": max(Decimal(row["elastic_endpoint_nr_relative_difference"]) for row in candidates) < Decimal("1e-4"),
                },
                "all_current_proxy_candidates_fail_screen": {"pass": all(candidate_failures)},
                "all_screened_light_states_need_a_new_exothermic_excitation": {
                    "pass": all(
                        Decimal(row["formal_delta_interval_if_declared_mass_is_treated_as_incoming_state_keV"]["maximum"]) < 0
                        and Decimal(row["hypothetical_heavier_excitation_to_declared_ground_state"]["minimum_required_release_MeV"]) > 0
                        for row in candidates
                    ),
                },
                "source_complete_mode_masses_are_live_action_derivations": {
                    "pass": source_vector_mass_GeV == expected_vector_mass_GeV and source_scalar_mass_GeV > 0,
                },
                "external_comparator_is_near_the_speed_boundary": {"pass": external_vmin <= speed_cap and external_vmin > Decimal("0.9") * speed_cap},
                "no_fit_or_detector_likelihood": {"pass": True},
            },
            "source_sha256": {
                "producer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "dark_neutron_module": hashlib.sha256(Path(dark_neutron.__file__).read_bytes()).hexdigest(),
                "dark_neutron_historical_input": dark_neutron_input["input_sha256"],
                "relic_module": hashlib.sha256(Path(relic.__file__).read_bytes()).hexdigest(),
                "source_complete_module": hashlib.sha256(Path(source_complete.__file__).read_bytes()).hexdigest(),
            },
        }
        result["all_controls_pass"] = all(bool(control["pass"]) for control in result["controls"].values())
        result["status"] = STATUS_PASS if result["all_controls_pass"] else STATUS_FAIL
        result["integrity_sha256"] = hashlib.sha256(_payload_bytes(result)).hexdigest()
        return result


def validate_result(result: Any, data_path: Path = DATA_PATH) -> bool:
    try:
        return isinstance(result, Mapping) and dict(result) == _build_result(data_path)
    except (ArithmeticError, InvalidOperation, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=HERE / "nvg_lz_248kev_kinematic_results.json")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate:
        valid = validate_result(json.loads(args.validate.read_text(encoding="utf-8")), args.input)
        print("PASS" if valid else "FAIL")
        return 0 if valid else 1
    result = _build_result(args.input)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["status"])
    return 0 if result["all_controls_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
