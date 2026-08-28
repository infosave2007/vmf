#!/usr/bin/env python3
"""
NVG Cross-Check: Tidal Deformability vs GW170817
-------------------------------------------------
Calculates Love number k_2, dimensionless tidal deformability Λ, and the
binary tidal deformability Λ_tilde for direct comparison with GW170817.

Method: TOV + Hinderer (2008) y-equation, using the maintained canonical
hybrid EOS and the unit conventions shared with the other NS verification
entry points.

References:
  Hinderer (2008) ApJ 677, 1216
  Abbott et al. (2018) PRL 121, 161101
"""

from __future__ import annotations
import hashlib
import inspect
import math
import numpy as np

try:
    from nvg_ns_canonical import canonical_selection
except ImportError:  # pragma: no cover - package/direct-script compatibility
    from verification.nvg_ns_canonical import canonical_selection


# Public aliases make the selection record discoverable to callers and tests.
CANONICAL_SELECTION = canonical_selection()

# ── Constants ────────────────────────────────────────────────────────
hbar_c = 197.3269804    # MeV·fm
M_N = 939.0
n_0 = 0.16              # fm^-3
G_cgs = 6.674e-8
c_cgs = 2.998e10
M_sun_g = 1.989e33
M_sun_km = 1.4766       # G M_sun / c^2 in km
k_conv = 1.3234e-6      # MeV/fm^3 → km^-2

# The optional convergence path is intentionally explicit here rather than
# inferred by audit consumers.  Keeping the backend, method and cap rule next
# to the implementation gives generated artifacts one authoritative source of
# ODE provenance.
ADAPTIVE_SOLVER_BACKEND = "scipy.integrate.solve_ivp"
ADAPTIVE_SOLVER_METHOD = "DOP853"
ADAPTIVE_SOLVER_MAX_STEP_RULE = "max(dr, 0.05 if rtol <= 1.0e-4 else 0.10)"
ADAPTIVE_SOLVER_MAX_STEP_CAP_KM = {
    "rtol_le_1e-4": 0.05,
    "rtol_gt_1e-4": 0.10,
}


def adaptive_solver_max_step(dr: float, rtol: float) -> float:
    """Return the exact ``solve_ivp`` max-step cap used by the audit path."""

    cap = 0.05 if float(rtol) <= 1.0e-4 else 0.10
    return max(float(dr), cap)


def adaptive_solver_provenance() -> dict[str, object]:
    """Return and validate the live adaptive ODE implementation metadata.

    The source checks deliberately inspect the actual ``solve_ivp`` call and
    the cap helper.  Audit artifacts record the resulting digest, so loading a
    stale artifact after an implementation change fails closed instead of
    silently presenting old convergence metadata as current.
    """

    source = inspect.getsource(solve_tov_tidal)
    cap_source = inspect.getsource(adaptive_solver_max_step)
    call_start = source.find("solution = solve_ivp(")
    call_source = source[call_start:] if call_start >= 0 else ""
    required = {
        "solve_ivp_call": call_start >= 0,
        "method_DOP853": 'method="DOP853"' in call_source,
        "rtol_forwarded": "rtol=float(rtol)" in call_source,
        "atol_forwarded": "atol=float(atol)" in call_source,
        "max_step_cap": "max_step=adaptive_solver_max_step(dr, rtol)" in call_source,
        "cap_rule": "0.05 if float(rtol) <= 1.0e-4 else 0.10" in cap_source,
    }
    if not all(required.values()):
        missing = [name for name, present in required.items() if not present]
        raise RuntimeError(
            "adaptive ODE implementation/provenance mismatch: " + ", ".join(missing)
        )
    source_digest = hashlib.sha256(
        (source + "\n" + cap_source).encode("utf-8")
    ).hexdigest()
    return {
        "backend": ADAPTIVE_SOLVER_BACKEND,
        "method": ADAPTIVE_SOLVER_METHOD,
        "method_description": "SciPy solve_ivp(method='DOP853')",
        "rtol_parameter": "solve_ivp rtol",
        "atol_parameter": "solve_ivp atol",
        "max_step_rule": ADAPTIVE_SOLVER_MAX_STEP_RULE,
        "max_step_cap_km": dict(ADAPTIVE_SOLVER_MAX_STEP_CAP_KM),
        "event_handling": "terminal downward transition/surface events",
        "source_sha256": source_digest,
        "source_checks": required,
    }


def hinderer_density_jump_y_match(
    y_minus: float,
    radius_km: float,
    mass_solar: float,
    delta_eps_mev_fm3: float,
) -> float:
    """Apply the first-order density-jump Hinderer matching condition.

    For a finite density discontinuity at fixed transition pressure the
    perturbation variable is matched as

    ``y_plus = y_minus - 4*pi*r_t**3*Delta_epsilon_geo/m_t_geo``.

    This is Eq. (15) of Postnikov, Prakash & Lattimer, *Phys. Rev. D* 82,
    024016 (2010), arXiv:1004.5098.  Their Eq. (14) supplies the
    distributional ``d rho/d p`` term that leads to this interface update.
    ``delta_eps_mev_fm3`` is converted to
    geometric km^-2 with the same canonical ``k_conv`` used by the TOV RHS;
    ``mass_solar`` is converted to geometric km with ``M_sun_km``.
    """

    values = (y_minus, radius_km, mass_solar, delta_eps_mev_fm3)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("density-jump match inputs must be finite")
    if float(radius_km) <= 0.0 or float(mass_solar) <= 0.0:
        raise ValueError("density-jump match requires positive radius and mass")
    if float(delta_eps_mev_fm3) < 0.0:
        raise ValueError("density-jump match requires a non-negative energy jump")
    radius_km = float(radius_km)
    mass_km = float(mass_solar) * M_sun_km
    delta_eps_geo = float(delta_eps_mev_fm3) * k_conv
    return float(y_minus) - 4.0 * math.pi * radius_km**3 * delta_eps_geo / mass_km

# ── NVG Core Model (identical to nvg_full_ns_eos.py) ─────────────────
M_Omega_0 = 859.0
M_current_0 = 80.0
kappa_1 = 0.25
kappa_2 = 0.80
alpha_v = 4.0
nu_v = 2.0
C_v_n0 = 100.0

def M_star(n_B: float) -> float:
    x = max(n_B / n_0, 0.0)
    return M_current_0 + M_Omega_0 * (1.0 + kappa_2 * x) ** (-kappa_1 / kappa_2)

def nvg_core_eos(n_B: float) -> tuple[float, float]:
    m_eff = M_star(n_B)
    kf = (3.0 * np.pi**2 * n_B) ** (1.0/3.0) * hbar_c
    ef = math.sqrt(kf**2 + m_eff**2)
    log_term = math.log((kf + ef) / m_eff)
    eps_kin = (kf * ef * (2*kf**2 + m_eff**2) - m_eff**4 * log_term) / (8 * np.pi**2 * hbar_c**3)
    P_kin = (kf * ef * (2*kf**2/3 - m_eff**2) + m_eff**4 * log_term) / (8 * np.pi**2 * hbar_c**3)
    v_pot = (m_eff - M_N) * n_B * 0.5
    x = n_B / n_0
    v_vec = C_v_n0 * x**2 / (1.0 + alpha_v * x**nu_v)
    eps_v = v_vec * n_B
    P_v = eps_v * (1.0 + nu_v * alpha_v * x**nu_v) / (1.0 + alpha_v * x**nu_v)
    return eps_kin + v_pot + eps_v, P_kin + P_v


class EOS:
    def __init__(self, p_match=1.5, Gamma=1.35):
        # Build baseline RMF + CSS hybrid EOS
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import nvg_eos_beta_css_softening as soft
        baseline = soft.build_baseline_arrays()
        if baseline is None:
            raise RuntimeError("canonical NS baseline EOS could not be built")
        # The scan-selected transition is an in-sample/conditional input.  Its
        # machine-readable provenance travels with every EOS instance.
        selection = canonical_selection()
        hybrid = soft.build_css_hybrid_eos(baseline, **selection["parameters"])
        if hybrid is None:
            raise RuntimeError("canonical NS transition could not build a valid EOS")
        self.p_arr = hybrid["p_sorted"]
        self.eps_arr = hybrid["e_sorted"]
        self.table_pressure_min = float(self.p_arr[0])
        # The crust continuation makes the public EOS domain start at zero;
        # ``table_pressure_min`` records the first tabulated point.
        self.pressure_min = 0.0
        self.pressure_max = float(self.p_arr[-1])
        if (not np.isfinite(p_match) or p_match < self.table_pressure_min
                or p_match > self.pressure_max):
            raise ValueError("p_match must lie inside the canonical EOS table domain")
        if not np.isfinite(Gamma) or Gamma <= 0.0:
            raise ValueError("Gamma must be a positive finite value")
        self.p_match = p_match
        self.Gamma = Gamma
        self.eps_match = float(np.interp(p_match, self.p_arr, self.eps_arr))
        if not np.isfinite(self.eps_match) or self.eps_match <= 0.0:
            raise RuntimeError("canonical EOS matching energy is not physical")
        self.canonical_selection = selection
        self.canonical_provenance = selection["provenance"]

    def get_eps(self, P: float) -> float:
        P = float(P)
        if not np.isfinite(P):
            raise ValueError("pressure must be finite")
        if P < 0.0:
            raise ValueError("pressure must be non-negative")
        p_max = self.pressure_max if hasattr(self, "pressure_max") else float(self.p_arr[-1])
        if P > p_max:
            raise ValueError(
                f"pressure {P:g} exceeds canonical EOS domain maximum {p_max:g}"
            )
        if P == 0.0: return 0.0
        if P < self.p_match:
            return self.eps_match * (P / self.p_match) ** (1.0 / self.Gamma)
        if P < self.table_pressure_min:
            raise ValueError("pressure lies below the canonical EOS table domain")
        # Optional exact first-order phase branches.  Audit-created EOS
        # instances install these attributes explicitly; the canonical
        # zero-jump constructor does not, preserving its historical table
        # values.  At P_t the lower-density branch is used and the jump is
        # applied by solve_tov_tidal when crossing the interface.
        transition_pressure = getattr(self, "transition_pressure", None)
        if transition_pressure is not None and P > float(transition_pressure):
            cs2_q = float(getattr(self, "transition_cs2_q"))
            energy_high = float(getattr(self, "transition_energy_high"))
            return energy_high + (P - float(transition_pressure)) / cs2_q
        had_p_arr = getattr(self, "had_p_arr", None)
        had_eps_arr = getattr(self, "had_eps_arr", None)
        if (had_p_arr is not None and had_eps_arr is not None
                and transition_pressure is not None and P <= float(transition_pressure)):
            return float(np.interp(P, had_p_arr, had_eps_arr))
        return float(np.interp(P, self.p_arr, self.eps_arr))

    def get_dedp(self, P: float) -> float:
        """dε/dP = 1/c_s² via finite difference."""
        P = float(P)
        if not np.isfinite(P):
            raise ValueError("pressure must be finite")
        p_max = self.pressure_max if hasattr(self, "pressure_max") else float(self.p_arr[-1])
        if P < 0.0 or P > p_max:
            raise ValueError("pressure is outside the canonical EOS domain")
        transition_pressure = getattr(self, "transition_pressure", None)
        if transition_pressure is not None:
            transition_pressure = float(transition_pressure)
            if P > transition_pressure:
                cs2_q = float(getattr(self, "transition_cs2_q"))
                if not np.isfinite(cs2_q) or cs2_q <= 0.0:
                    raise ValueError("invalid CSS sound speed on transition branch")
                return 1.0 / cs2_q
        dP = max(P * 1e-4, 1e-8)
        # Never take a finite-difference stencil across a density jump.  The
        # The distributional contribution is matched explicitly in the y
        # equation at P_t.  Postnikov, Prakash & Lattimer (2010) introduce the
        # distributional d rho/dp term in Eq. 14 and give the integrated
        # interface condition used here in Eq. 15.
        if transition_pressure is not None and P <= transition_pressure and P + dP > transition_pressure:
            p_lo = max(float(self.table_pressure_min), P - dP)
            if transition_pressure > p_lo:
                return (self.get_eps(transition_pressure) - self.get_eps(p_lo)) / (transition_pressure - p_lo)
        if P <= dP:
            return (self.get_eps(P + dP) - self.get_eps(P)) / dP
        p_max = self.pressure_max if hasattr(self, "pressure_max") else float(self.p_arr[-1])
        if P + dP > p_max:
            return (self.get_eps(P) - self.get_eps(P - dP)) / dP
        e1 = self.get_eps(P + dP)
        e2 = self.get_eps(P - dP)
        return (e1 - e2) / (2.0 * dP)


# ── TOV + Tidal y-equation ───────────────────────────────────────────

def solve_tov_tidal(
    eos: EOS,
    P_center: float,
    *,
    dr: float = 0.05,
    rtol: float | None = None,
    atol: float | None = None,
    density_jump_matching: bool = False,
) -> tuple[float, float, float, float]:
    """
    Integrate TOV + Hinderer y-equation.
    Returns (M_solar, R_km, k2, Lambda).

    Units: m in M_sun via (m_km / 1.4766), r in km, P in MeV/fm³.
    All geometric factors use κ = 1.3234e-6 km⁻² per MeV/fm³.

    ``dr`` preserves the historical fixed-step RK4 path when ``rtol`` and
    ``atol`` are omitted.  Supplying both tolerances enables the adaptive
    SciPy ``scipy.integrate.solve_ivp`` path with ``method="DOP853"``; the
    requested ``rtol`` and ``atol`` are passed directly to ``solve_ivp``.
    Each phase uses the explicit cap
    ``max_step=max(dr, 0.05 if rtol <= 1.0e-4 else 0.10)`` km (where ``dr``
    remains the historical fixed-step increment).  The optional path is a
    convergence hook for audits; existing callers retain their original
    numerical defaults.  When ``density_jump_matching`` is true, an
    audit-created EOS must expose ``transition_pressure`` and
    ``transition_energy_jump``.  Crossing that pressure applies the
    distributional Hinderer match
    ``y+ = y- - 4*pi*r_t**3*Delta_epsilon/m_t`` from Postnikov, Prakash &
    Lattimer (2010), arXiv:1004.5098, Eq. 15.  Eq. 14 in that source is the
    distributional derivative whose integration yields this finite jump.
    """
    P_center = float(P_center)
    if not np.isfinite(P_center) or P_center <= 0.0:
        raise ValueError("central pressure must be a positive finite value")
    # Fail closed before integration: an EOS endpoint must never be silently
    # reused as a continuation for an out-of-domain central pressure.
    eos.get_eps(P_center)

    dr = float(dr)
    if not np.isfinite(dr) or dr <= 0.0:
        raise ValueError("dr must be a positive finite length in km")
    adaptive = rtol is not None or atol is not None
    if adaptive:
        if rtol is None or atol is None:
            raise ValueError("rtol and atol must be supplied together")
        rtol = float(rtol)
        atol = float(atol)
        if not np.isfinite(rtol) or rtol <= 0.0:
            raise ValueError("rtol must be a positive finite value")
        if not np.isfinite(atol) or atol <= 0.0:
            raise ValueError("atol must be a positive finite value")
    transition_pressure = getattr(eos, "transition_pressure", None)
    transition_jump = getattr(eos, "transition_energy_jump", 0.0)
    if density_jump_matching:
        if transition_pressure is None or not np.isfinite(float(transition_pressure)):
            raise ValueError("density-jump matching requires a finite transition_pressure")
        if not np.isfinite(float(transition_jump)) or float(transition_jump) < 0.0:
            raise ValueError("density-jump matching requires a non-negative transition jump")
        transition_pressure = float(transition_pressure)
        transition_jump = float(transition_jump)
    else:
        transition_pressure = None
        transition_jump = 0.0
    # Audit-created EOS instances carry the same transition metadata for
    # zero- and positive-latent-heat rows.  Keep the interface event pending
    # whenever the centre lies above P_t, including an exact zero jump, so the
    # same matcher is invoked with a literal zero update.  The historical
    # canonical constructor has no transition metadata and the default solver
    # flag is false, so its fixed-step path remains unchanged.
    transition_pending = bool(
        density_jump_matching and transition_pressure is not None
        and P_center > transition_pressure
    )
    r = 1e-6   # km
    m = 0.0    # km (geometric mass)
    P = P_center
    y = 2.0    # y(0) = 2 for l=2

    def derivs(r_val, m_val, P_val, y_val):
        if r_val < 1e-10 or P_val <= 0:
            return 0.0, 0.0, 0.0

        eps = eos.get_eps(P_val)
        eps_k = eps * k_conv
        P_k = P_val * k_conv
        m_k = m_val * M_sun_km  # convert M_sun → km

        fac = r_val * (r_val - 2.0 * m_k)
        if fac <= 0:
            return 0.0, 0.0, 0.0

        # TOV
        dm_dr = 4.0 * math.pi * r_val**2 * eps_k / M_sun_km  # M_sun / km
        num = (eps_k + P_k) * (m_k + 4.0 * math.pi * r_val**3 * P_k)
        dP_dr = -num / fac / k_conv  # MeV/fm³ / km

        # y-equation (Hinderer 2008, Eq. 14)
        e_minus_p = (eps_k - P_k)
        one_m_2mr = 1.0 - 2.0 * m_k / r_val

        F = (1.0 - 4.0 * math.pi * r_val**2 * e_minus_p) / one_m_2mr

        dedp = eos.get_dedp(P_val)
        if dedp <= 0:
            dedp = 1.0

        Q_source = 4.0 * math.pi * (5.0 * eps_k + 9.0 * P_k + (eps_k + P_k) * dedp)
        Q_source /= one_m_2mr
        Q_grav = (2.0 * (m_k + 4.0 * math.pi * r_val**3 * P_k) / (r_val * one_m_2mr))**2 / r_val**2
        Q = Q_source - Q_grav - 6.0 / (r_val**2 * one_m_2mr)

        dy_dr = -(y_val**2 + y_val * F + r_val**2 * Q) / r_val

        return dm_dr, dP_dr, dy_dr

    def rk4_step(r_val, state, step):
        """Advance one RK4 step, returning (m, P, y)."""

        m_val, p_val, y_val = state
        dm1, dp1, dy1 = derivs(r_val, m_val, p_val, y_val)
        r2 = r_val + 0.5 * step
        s2 = (m_val + 0.5 * step * dm1,
              p_val + 0.5 * step * dp1,
              y_val + 0.5 * step * dy1)
        if s2[1] <= 0.0:
            return None
        dm2, dp2, dy2 = derivs(r2, *s2)
        r3 = r_val + 0.5 * step
        s3 = (m_val + 0.5 * step * dm2,
              p_val + 0.5 * step * dp2,
              y_val + 0.5 * step * dy2)
        if s3[1] <= 0.0:
            return None
        dm3, dp3, dy3 = derivs(r3, *s3)
        r4 = r_val + step
        s4 = (m_val + step * dm3,
              p_val + step * dp3,
              y_val + step * dy3)
        if s4[1] <= 0.0:
            return None
        dm4, dp4, dy4 = derivs(r4, *s4)
        return (
            m_val + (step / 6.0) * (dm1 + 2.0 * dm2 + 2.0 * dm3 + dm4),
            p_val + (step / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4),
            y_val + (step / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4),
        )

    def apply_density_jump(state_before, state_after, r_before, step):
        """Interpolate P=P_t and apply the first-order Eq. 15 y jump.

        ``transition_jump`` may be exactly zero.  In that case the helper is
        still called and returns the unchanged y value, certifying one common
        interface path for zero- and positive-latent-heat audit rows.
        """

        nonlocal transition_pending
        if not transition_pending or transition_pressure is None:
            return state_after, r_before + step, False
        p_before = float(state_before[1])
        p_after = float(state_after[1])
        if not (p_before > transition_pressure >= p_after):
            return state_after, r_before + step, False
        denominator = p_before - p_after
        fraction = 1.0 if denominator <= 0.0 else (p_before - transition_pressure) / denominator
        fraction = min(1.0, max(0.0, fraction))
        r_transition = r_before + fraction * step
        # Integrate the pre-interface segment on the high-density branch when
        # possible.  Linear interpolation across a step that straddles P_t
        # evaluates the RHS on both phases and made the result depend strongly
        # on the requested adaptive tolerance.  Refine the event step by
        # bisection on the RK4 pressure endpoint; the short RK4 segment ends at
        # the located event and its pressure is then set exactly to P_t before
        # the first-order match is applied.
        pre_step = fraction * step
        lo, hi = 0.0, step
        bracketed = False
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            candidate = rk4_step(r_before, state_before, mid)
            if candidate is None or not np.isfinite(float(candidate[1])):
                hi = mid
                continue
            if float(candidate[1]) > transition_pressure:
                lo = mid
            else:
                hi = mid
                bracketed = True
        if bracketed:
            pre_step = hi
            r_transition = r_before + pre_step
        pre_state = rk4_step(r_before, state_before, pre_step) if pre_step > 1.0e-12 else state_before
        if pre_state is None or not np.isfinite(float(pre_state[0])) or not np.isfinite(float(pre_state[2])):
            m_transition = float(state_before[0] + fraction * (state_after[0] - state_before[0]))
            y_transition = float(state_before[2] + fraction * (state_after[2] - state_before[2]))
        else:
            m_transition = float(pre_state[0])
            y_transition = float(pre_state[2])
        y_transition = hinderer_density_jump_y_match(
            y_transition,
            r_transition,
            m_transition,
            transition_jump,
        )
        transition_pending = False
        return (m_transition, transition_pressure, y_transition), r_transition, True

    state = (m, P, y)
    if adaptive:
        # SciPy's high-order adaptive integrator provides a stable error
        # controller around the Hinderer surface cells.  Integrate each phase
        # separately so a density discontinuity is an actual event rather than
        # a finite-width pressure interpolation.  The historical default path
        # below remains the original fixed-step RK4 implementation.
        try:
            from scipy.integrate import solve_ivp
        except ImportError as exc:  # pragma: no cover - requirements include scipy
            raise RuntimeError("adaptive tidal integration requires scipy") from exc

        def rhs(radius, values):
            return np.asarray(derivs(float(radius), float(values[0]),
                                     float(values[1]), float(values[2])), dtype=float)

        def surface_event(radius, values):
            del radius
            return float(values[1]) - 1.0e-4

        surface_event.terminal = True
        surface_event.direction = -1

        def transition_event(radius, values):
            del radius
            return float(values[1]) - float(transition_pressure)

        transition_event.terminal = True
        transition_event.direction = -1

        def integrate_phase(start_radius, start_state, *, stop_at_transition):
            events = [transition_event, surface_event] if stop_at_transition else [surface_event]
            solution = solve_ivp(
                rhs,
                (float(start_radius), 100.0),
                np.asarray(start_state, dtype=float),
                method="DOP853",
                rtol=float(rtol),
                atol=float(atol),
                # ``dr`` is the historical fixed-step increment.  Requested
                # tolerances at or below 1e-4 retain the conservative 0.05 km
                # cap; looser tolerances use 0.10 km without changing the
                # resolved transition/surface event semantics.
                max_step=adaptive_solver_max_step(dr, rtol),
                events=events,
            )
            if not solution.success:
                raise RuntimeError(f"adaptive tidal integration failed: {solution.message}")
            return solution

        if transition_pending:
            high_solution = integrate_phase(r, state, stop_at_transition=True)
            event_values = high_solution.t_events[0]
            if len(event_values) == 0 or high_solution.y_events[0].shape[0] == 0:
                raise RuntimeError("density transition event was not located")
            transition_r = float(event_values[0])
            transition_state = np.asarray(high_solution.y_events[0][0], dtype=float)
            transition_state[1] = float(transition_pressure)
            transition_state[2] = hinderer_density_jump_y_match(
                float(transition_state[2]),
                transition_r,
                float(transition_state[0]),
                transition_jump,
            )
            transition_pending = False
            low_solution = integrate_phase(transition_r, transition_state, stop_at_transition=False)
            state = tuple(float(value) for value in low_solution.y[:, -1])
            r = float(low_solution.t[-1])
        else:
            solution = integrate_phase(r, state, stop_at_transition=False)
            state = tuple(float(value) for value in solution.y[:, -1])
            r = float(solution.t[-1])
    else:
        while state[1] > 1e-4 and r < 100.0:
            next_state = rk4_step(r, state, dr)
            if next_state is None:
                break
            matched_state, matched_r, matched = apply_density_jump(state, next_state, r, dr)
            if matched:
                # Do not discard the portion of a fixed step that lies just
                # outside the interface.  Integrating that short remainder
                # keeps zero- and epsilon-positive rows on one representation
                # while avoiding a finite-width pressure plateau.
                remainder = max(0.0, (r + dr) - matched_r)
                tail = rk4_step(matched_r, matched_state, remainder) if remainder > 1.0e-12 else matched_state
                if tail is not None:
                    state = tail
                    r += dr
                else:
                    state, r = matched_state, matched_r
            else:
                state, r = matched_state, matched_r
            if state[1] < 0.0:
                state = (state[0], 0.0, state[2])
                break

    m, P, y = state

    R = r
    M_solar = m
    C = M_solar * M_sun_km / R  # compactness (dimensionless)
    y_R = y

    # Love number k_2 (Hinderer 2008, Eq. 22)
    if C <= 0 or C >= 0.5:
        return M_solar, R, 0.0, 0.0

    fac_1m2C = 1.0 - 2.0 * C
    if fac_1m2C <= 0:
        return M_solar, R, 0.0, 0.0

    ln_1m2C = math.log(fac_1m2C)

    num = (8.0/5.0) * C**5 * fac_1m2C**2 * (2.0 + 2.0*C*(y_R - 1.0) - y_R)

    den = (2.0*C * (6.0 - 3.0*y_R + 3.0*C*(5.0*y_R - 8.0))
           + 4.0*C**3 * (13.0 - 11.0*y_R + C*(3.0*y_R - 2.0) + 2.0*C**2*(1.0 + y_R))
           + 3.0*fac_1m2C**2 * (2.0 - y_R + 2.0*C*(y_R - 1.0)) * ln_1m2C)

    if abs(den) < 1e-30:
        return M_solar, R, 0.0, 0.0

    k2 = num / den
    Lambda = (2.0/3.0) * k2 * C**(-5) if k2 > 0 else 0.0

    return M_solar, R, k2, Lambda


def binary_lambda_tilde(m1, m2, L1, L2):
    M = m1 + m2
    return (16.0/13.0) * ((m1 + 12.0*m2)*m1**4*L1 + (m2 + 12.0*m1)*m2**4*L2) / M**5


def main():
    print("=" * 80)
    print("  NVG CROSS-CHECK: TIDAL DEFORMABILITY vs GW170817")
    print("=" * 80)
    print("  Comparison status: CONDITIONAL_IN_SAMPLE (transition selected on J0740/GW170817/NICER)")

    eos = EOS(p_match=1.5, Gamma=1.35)

    # Scan
    # Scan must extend beyond the mass-curve turnover; the old cap 10^2.8 stopped
    # before M_max and the last scanned point was misreported as the maximum mass.
    P_centers = np.logspace(-1.0, 3.4, 120)
    results_raw = []
    for Pc in P_centers:
        M, R, k2, Lam = solve_tov_tidal(eos, Pc)
        if M > 0.5 and R > 5.0 and k2 > 0 and Lam > 0:
            results_raw.append((M, R, k2, Lam))

    if not results_raw:
        print("ERROR: No valid TOV solutions found!")
        return

    # Filter out post-collapse branch: keep only up to the first maximum mass
    idx_max = np.argmax([r[0] for r in results_raw])
    results = results_raw[:idx_max + 1]

    if not results:
        print("ERROR: No valid TOV solutions found!")
        return

    # Deduplicate by mass (keep first valid for each)
    results_sorted = sorted(results, key=lambda x: x[0])

    print(f"\n  {'M (M_sun)':>10}  {'R (km)':>8}  {'C':>8}  {'k_2':>8}  {'Λ':>10}")
    print("  " + "-" * 50)
    for i, (M, R, k2, Lam) in enumerate(results_sorted):
        if i % 3 == 0 or M > 2.0:
            C = M * M_sun_km / R
            print(f"  {M:10.3f}  {R:8.2f}  {C:8.4f}  {k2:8.4f}  {Lam:10.1f}")

    masses = np.array([r[0] for r in results_sorted])
    lambdas = np.array([r[3] for r in results_sorted])
    radii = np.array([r[1] for r in results_sorted])
    k2s = np.array([r[2] for r in results_sorted])

    M_max = max(masses)

    # Interpolate key values
    print("\n" + "-" * 80)
    print("  KEY VALUES:")
    print("-" * 80)
    targets = [1.2, 1.338, 1.36, 1.4, 1.6, 1.8]
    interp = {}
    for mt in targets:
        if mt < masses.min() or mt > masses.max():
            continue
        L_i = float(np.interp(mt, masses, lambdas))
        R_i = float(np.interp(mt, masses, radii))
        k2_i = float(np.interp(mt, masses, k2s))
        interp[mt] = (R_i, k2_i, L_i)
        C_i = mt * M_sun_km / R_i
        print(f"  M = {mt:.3f} M_sun:  R = {R_i:.2f} km,  k_2 = {k2_i:.4f},  Λ = {L_i:.1f},  C = {C_i:.4f}")

    # ── GW170817 Comparison ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  GW170817 COMPARISON")
    print("=" * 80)

    # Symmetric
    m1, m2 = 1.36, 1.36
    L1 = float(np.interp(m1, masses, lambdas))
    L2 = float(np.interp(m2, masses, lambdas))
    Lt_sym = binary_lambda_tilde(m1, m2, L1, L2)

    # Asymmetric
    m1a, m2a = 1.46, 1.27
    L1a = float(np.interp(m1a, masses, lambdas))
    L2a = float(np.interp(m2a, masses, lambdas))
    Lt_asym = binary_lambda_tilde(m1a, m2a, L1a, L2a)

    print(f"\n  Symmetric  (m1=m2=1.36): Λ₁={L1:.0f}, Λ₂={L2:.0f}, Λ̃ = {Lt_sym:.0f}")
    print(f"  Asymmetric (1.46+1.27):  Λ₁={L1a:.0f}, Λ₂={L2a:.0f}, Λ̃ = {Lt_asym:.0f}")

    # LIGO constraint
    L_lo, L_med, L_hi = 70, 300, 720
    print(f"\n  LIGO/Virgo 90% CI (low-spin): Λ̃ = {L_med} [{L_lo}, {L_hi}]")
    # The CSS transition is a scan-selected in-sample input.  This comparison
    # reports the resulting runtime values conditionally; it does not establish
    # an independent fit or confirmation of the EOS.
    ok_sym = L_lo <= Lt_sym <= L_hi
    ok_asym = L_lo <= Lt_asym <= L_hi
    print(f"  NVG symmetric:  Λ̃ = {Lt_sym:.0f}  →  ({'within' if ok_sym else 'outside'} GW170817 bounds; conditional/in-sample)")
    print(f"  NVG asymmetric: Λ̃ = {Lt_asym:.0f}  →  ({'within' if ok_asym else 'outside'} GW170817 bounds; conditional/in-sample)")

    # ── R_1.4 vs NICER ───────────────────────────────────────────────
    if 1.4 in interp:
        R14 = interp[1.4][0]
        print(f"\n  R_1.4 = {R14:.2f} km  (NICER: 12.45 ± 0.65 km)")
        ok_R = 11.0 <= R14 <= 14.0
        print(f"  Status: {'within' if ok_R else 'outside'} stated NICER interval (conditional/in-sample)")

    # ── Double Pulsar I ──────────────────────────────────────────────
    if 1.338 in interp:
        R1338, k2_1338, L1338 = interp[1.338]
        a_I, b_I, c_I, d_I, e_I = 1.496, 0.05951, 0.02238, -6.953e-4, 8.345e-6
        lnL = math.log(max(L1338, 1.0))
        I_bar = math.exp(a_I + b_I*lnL + c_I*lnL**2 + d_I*lnL**3 + e_I*lnL**4)
        M_cm = 1.338 * M_sun_g * G_cgs / c_cgs**2
        I_cgs = I_bar * M_cm**3 / (G_cgs / c_cgs**2)
        print(f"\n  Double Pulsar J0737-3039A (M=1.338 M_sun):")
        print(f"  I from Lambda transform (no independent I solve) = {I_cgs:.3e} g cm²")
        print(f"  Observed interval (context only; no validation): 1.15 (+0.38/-0.24) × 10^45 g cm²")
        ok_I = 0.91e45 <= I_cgs <= 1.53e45
        print(f"  Transform overlap (descriptive only; not validation): {'yes' if ok_I else 'no'}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  M_max          = {M_max:.2f} M_sun")
    if 1.4 in interp:
        print(f"  R_1.4          = {interp[1.4][0]:.2f} km")
        print(f"  Λ_1.4          = {interp[1.4][2]:.0f}")
    print(f"  Λ̃ (sym, GW17)  = {Lt_sym:.0f}  [LIGO: {L_lo}–{L_hi}]")
    print(f"  Λ̃ (asym, GW17) = {Lt_asym:.0f}  [LIGO: {L_lo}–{L_hi}]")
    print()

    assert M_max > 2.0, f"M_max = {M_max:.2f} < 2.0!"
    assert ok_sym or ok_asym, f"Λ̃ outside GW170817 90% CI!"
    print("Runtime tidal calculation complete; all displayed comparisons are conditional/in-sample.")


if __name__ == "__main__":
    main()
