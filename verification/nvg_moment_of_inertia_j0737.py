#!/usr/bin/env python3
"""
NVG Verification: Moment of inertia of the double pulsar PSR J0737-3039A.

The moment of inertia I(M) is the cleanest radius-independent discriminator
of the VMF EOS: relativistic timing of the double pulsar is projected to
measure I_A at the ~10% level (Lattimer & Schutz 2005).

Method (no free parameters beyond the fork-B EOS anchor):
  1. Piecewise-polytrope EOS calibrated to the fork-B canonical star:
     R(1.4 M_sun) = 12.49 km, transition to quark matter near 4 rho_0
     with the conformal CSS limit c_s^2 = 1/3.
  2. Full TOV integration (geometric units, G = c = 1) for M_max, R(M).
  3. Slow-rotation frame-dragging ODE (Hartle 1967; geometric units):
        (1/r^4)(r^4 om_bar')' + (4/r) j' om_bar = 0,
        j(r) = e^{-nu} sqrt(1 - 2m/r),
     expanded to
        om_bar'' = 4(2m - r + pi r^3 (eps+p))/(r(r-2m)) om_bar'
                   + 16 pi r (eps+p)/(r-2m) om_bar
     with regular central seed om_bar ~ om0(1 + 8 pi r^2 (eps+p)/5) and
     exterior matching om_bar_>(r) = Omega (1 - 2I/r^3), giving
        I = (r^4/6) om_bar'(R)/om_bar(R).
     Verified in-code against the constant-density star series
        I/(MR^2) = (2/5)(1 + 6Z/7 + 106Z^2/105 + ...), Z = GM/Rc^2.

Prediction targets:
  - I(1.4 M_sun) and I_A = I(1.2489 M_sun) for PSR J0737-3039A
  - falsification threshold for fork B once I_A is measured at 10%.
"""

import math

# --- physical constants (CGS) ---
CC = 2.99792458e10       # cm/s
GG = 6.67430e-8          # cm^3 g^-1 s^-2
MSUN = 1.98892e33        # g
MEV = 1.602176634e-6     # erg per MeV
FM3 = 1.0e-39            # cm^3 per fm^3
K_CM = GG / CC**2        # mass [g] -> length [cm]
K_P = GG / CC**4         # energy density [erg/cm^3] -> 1/cm^2

RHO0_MEVFM3 = 150.0      # nuclear saturation energy density ~ 0.16 fm^-3


def mevfm3_to_cgs(rho_mevfm3: float) -> float:
    """Energy density MeV/fm^3 -> erg/cm^3."""
    return rho_mevfm3 * MEV / FM3


# ---------------------------------------------------------------- EOS
class ForkBEOS:
    """Piecewise polytrope calibrated to fork B: R_1.4 ~ 12.5 km, M_max ~ 2."""

    def __init__(self):
        # all internal EOS values in CGS energy density (erg/cm^3)
        self.rho0 = mevfm3_to_cgs(RHO0_MEVFM3)
        self.rho_tr = 2.0 * self.rho0
        self.gamma1 = 2.80
        p_nuc = mevfm3_to_cgs(3.4)                 # P(rho0)
        self.k1 = p_nuc / self.rho0**self.gamma1
        self.gamma2 = 2.30
        p2 = mevfm3_to_cgs(32.0)                   # P(2 rho0)
        self.k2 = p2 / self.rho_tr**self.gamma2
        # quark CSS segment anchored at 4 rho0: pressure jump 1.3, c_s^2 = 1/3
        self.rho_q = 4.0 * self.rho0
        p_at_q = self.k2 * self.rho_q**self.gamma2
        self.p_q = 1.3 * p_at_q
        self.eps_q = self.rho_q + self.p_q - p_at_q  # bag jump at constant P

    def pressure(self, eps: float) -> float:
        """Pressure p(epsilon), CGS erg/cm^3."""
        if eps < self.rho_tr:
            return self.k1 * eps**self.gamma1
        if eps < self.rho_q:
            return self.k2 * eps**self.gamma2
        if eps <= self.eps_q:
            return self.p_q / 1.3                  # mixed phase
        return self.p_q + (eps - self.eps_q) / 3.0

    def energy_density(self, p: float) -> float:
        """Inverse eps(p), CGS erg/cm^3."""
        if p < self.k1 * self.rho_tr**self.gamma1:
            return (p / self.k1)**(1.0 / self.gamma1)
        if p < self.p_q / 1.3:
            return (p / self.k2)**(1.0 / self.gamma2)
        if p < self.p_q:
            return self.eps_q                      # mixed phase
        return self.eps_q + 3.0 * (p - self.p_q)


# ---------------------------------------------------------------- TOV
class TOVResult:
    pass


def tov_star(eos: ForkBEOS, eps_c_cgs: float, dr: float = 5.0e3,
             n_steps: int = 8000, p_c_cgs: float = None) -> TOVResult:
    """Integrate TOV in geometric units (G = c = 1, r in cm)."""
    # convert to geometric units
    eps_c = eps_c_cgs * K_P
    p_c = (eos.pressure(eps_c_cgs) if p_c_cgs is None else p_c_cgs) * K_P
    p_cut = 1.0e-8 * eos.pressure(eos.rho0) * K_P  # deep-crust cutoff

    def p_geom_of(eps_geom: float) -> float:
        return eos.pressure(eps_geom / K_P) * K_P

    def eps_geom_of(p_geom: float) -> float:
        return eos.energy_density(p_geom / K_P) * K_P

    # analytic seed at r = dr (avoids the r=0 coordinate singularity)
    r = dr
    p = p_c - (2.0 * math.pi / 3.0) * (eps_c + p_c) * (eps_c + 3.0 * p_c) * r**2
    m = 4.0 * math.pi / 3.0 * eps_c * r**3
    nu = 0.0

    rs = [r]; eps_a = [eps_c]; p_a = [p]; m_a = [m]; nu_a = [nu]

    def rhs(r, p, m):
        if p <= p_cut:
            return 0.0, 0.0, 0.0
        eps = eps_geom_of(p)
        f = 1.0 - 2.0 * m / r
        dp = -(eps + p) * (m + 4.0 * math.pi * r**3 * p) / (r**2 * f)
        dm = 4.0 * math.pi * r**2 * eps
        dnu = 2.0 * (m + 4.0 * math.pi * r**3 * p) / (r**2 * f)
        return dp, dm, dnu

    for _ in range(n_steps):
        if p <= p_cut:
            break
        p0, m0, nu0 = p, m, nu
        k1 = rhs(r, p0, m0)
        k2 = rhs(r + dr / 2, max(p0 + dr / 2 * k1[0], p_cut), m0 + dr / 2 * k1[1])
        k3 = rhs(r + dr / 2, max(p0 + dr / 2 * k2[0], p_cut), m0 + dr / 2 * k2[1])
        k4 = rhs(r + dr, max(p0 + dr * k3[0], p_cut), m0 + dr * k3[1])
        p = p0 + dr / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        m = m0 + dr / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        nu = nu0 + dr / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        r += dr
        rs.append(r); eps_a.append(max(eps_geom_of(max(p, p_cut)), p_cut))
        p_a.append(p); m_a.append(m); nu_a.append(nu)

    res = TOVResult()
    res.rs, res.eps_a, res.p_a, res.m_a = rs, eps_a, p_a, m_a
    # match nu to exterior Schwarzschild at the surface
    res.nu_a = [x + (math.log(1.0 - 2.0 * m / r) - nu) for x in nu_a]
    res.r_cm = r
    res.m_g = m / K_CM
    return res


def interp1(x: float, xs, ys) -> float:
    """Linear interpolation with clamping at both ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = max(1, min(len(xs) - 1, int(x / xs[-1] * (len(xs) - 1)) + 1))
    while xs[i] < x:
        i += 1
    while xs[i - 1] > x:
        i -= 1
    w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + w * (ys[i] - ys[i - 1])


def moment_of_inertia(star: TOVResult) -> float:
    """Frame-dragging ODE integration; returns I in g cm^2."""
    rs, eps, p, m_a = star.rs, star.eps_a, star.p_a, star.m_a
    r0, r_s = rs[0], rs[-1]
    dr = (r_s - r0) / max(len(rs) - 1, 1)

    s0 = eps[0] + p[0]
    w = 1.0 + (8.0 / 5.0) * math.pi * s0 * r0**2
    wp = (16.0 / 5.0) * math.pi * s0 * r0

    def ode(r, w, wp):
        s = interp1(r, rs, eps) + interp1(r, rs, p)
        m = interp1(r, rs, m_a)
        den = r - 2.0 * m
        return (4.0 * (2.0 * m - r + math.pi * r**3 * s) / (r * den) * wp
                + 16.0 * math.pi * r * s / den * w)

    # RK4 from r0 to the surface
    n = max(int((r_s - r0) / dr), 2)
    h = (r_s - r0) / n
    r = r0
    for _ in range(n):
        k1w = wp;                    k1p = ode(r, w, wp)
        k2w = wp + h / 2 * k1p;      k2p = ode(r + h / 2, w + h / 2 * k1w, k2w)
        k3w = wp + h / 2 * k2p;      k3p = ode(r + h / 2, w + h / 2 * k2w, k3w)
        k4w = wp + h * k3p;          k4p = ode(r + h, w + h * k3w, k4w)
        w += h / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
        wp += h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        r += h

    # exterior match: om_bar_> = Omega(1 - 2I/r^3)
    # -> q = R^4 w'/w = 6I/(1 - 2I/R^3)  ->  I = q/(6 + 2q/R^3)
    q = r_s**4 * wp / w
    i_geom = q / (6.0 + 2.0 * q / r_s**3)   # cm^3
    return i_geom * CC**2 / GG              # g cm^2


class ConstantDensityEOS:
    """Unphysical constant-density EOS for the Hartle sanity test.

    eps0 is the CGS energy density; the TOV integrator converts internally.
    """

    def __init__(self, eps0_cgs: float):
        self.eps0 = eps0_cgs

    @property
    def rho0(self):
        return self.eps0

    def pressure(self, eps_cgs):
        # arbitrary monotone function; the profile follows the TOV equation
        return eps_cgs * 1e-10

    def energy_density(self, p_cgs):
        return self.eps0


def constant_density_test():
    """Check against I/(MR^2) = (2/5)(1 + 6Z/7 + 106Z^2/105), Z = M/R."""
    z_target = 0.05
    r_cm = 2.0e6                       # 20 km
    m_geom = z_target * r_cm
    eps_geom = 3.0 * m_geom / (4.0 * math.pi * r_cm**3)
    # exact interior-Schwarzschild central pressure for constant density
    sq = math.sqrt(1.0 - 2.0 * z_target)
    p0_geom = eps_geom * (1.0 - sq) / (3.0 * sq - 1.0)
    eos = ConstantDensityEOS(eps_geom / K_P)
    star = tov_star(eos, eps_geom / K_P, p_c_cgs=p0_geom / K_P)
    i = moment_of_inertia(star)
    m_g = star.m_g
    z = K_CM * m_g / star.r_cm
    ratio = i / (m_g * star.r_cm**2)
    series = 0.4 * (1.0 + 6.0 * z / 7.0 + 106.0 * z**2 / 105.0)
    return ratio, series, z


def main():
    print("=" * 72)
    print(" NVG/VMF MOMENT OF INERTIA: PSR J0737-3039A (DOUBLE PULSAR)")
    print("=" * 72)

    ratio, series, z = constant_density_test()
    print(f"[sanity] constant-density star, Z = {z:.4f}:")
    print(f"         I/(MR^2) numerical = {ratio:.4f} ; "
          f"series (2/5)(1+6Z/7+106Z^2/105) = {series:.4f} ; "
          f"rel. diff = {abs(ratio-series)/series*100:.2f}%")
    assert abs(ratio - series) / series < 0.02, "Hartle ODE check failed"
    print("-" * 72)

    eos = ForkBEOS()

    results = []
    for eps_c_mult in [1.2, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.5, 8.0, 10.0]:
        star = tov_star(eos, eos.rho0 * eps_c_mult)
        m_msun = star.m_g / MSUN
        r_km = star.r_cm / 1e5
        compact = K_CM * star.m_g / star.r_cm
        if compact > 0.6:     # unphysical / beyond turning point
            continue
        i_grcm2 = moment_of_inertia(star)
        results.append((m_msun, r_km, i_grcm2, compact))

    m_max = max(r[0] for r in results)
    print(f"{'M [M_sun]':>10} {'R [km]':>8} {'I [1e45 g cm2]':>15} "
          f"{'C=GM/Rc2':>9}")
    for m, r, i, c in results:
        print(f"{m:10.3f} {r:8.2f} {i/1e45:15.3f} {c:9.3f}")
    print("-" * 72)
    print(f"M_max = {m_max:.2f} M_sun (fork-B target >= 2.0; NICER J0740: 2.08)")

    def interp(target_m):
        for a, b in zip(results[:-1], results[1:]):
            if a[0] <= target_m <= b[0]:
                w = (target_m - a[0]) / (b[0] - a[0])
                return (a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2]))
        return None

    for label, mt in (("Canonical 1.40 M_sun", 1.40),
                      ("J0737-3039A (1.2489 M_sun)", 1.2489)):
        out = interp(mt)
        if out:
            r_km, i = out
            print(f"{label}: R = {r_km:.2f} km, I = {i/1e45:.3f}e45 g cm^2")
    print("-" * 72)

    out14 = interp(1.40)
    if out14:
        i_forkb = out14[1]
        i_stiff = i_forkb * (13.1 / 12.49) ** 2   # R -> 13.1 km (NL3-like)
        print("DISCRIMINATION vs a stiffer canonical EOS (R_1.4 = 13.1 km):")
        print(f"  fork B:  I_1.4 = {i_forkb/1e45:.2f}e45 g cm^2")
        print(f"  stiff:   I_1.4 ~ {i_stiff/1e45:.2f}e45 g cm^2  "
              f"(~{(i_stiff/i_forkb-1)*100:.0f}% higher)")
        print("  A 10% measurement of I_A (double pulsar) separates the two at")
        print(f"  ~{(i_stiff/i_forkb-1)/0.10:.1f} sigma. Falsification of fork B: "
              f"I_1.4 > {1.10*i_forkb/1e45:.2f}e45 g cm^2")
    print("=" * 72)


if __name__ == "__main__":
    main()
