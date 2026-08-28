#!/usr/bin/env python3
"""
NVG Population-Level Confirmation: Magnetar Mass-Field Correlation
------------------------------------------------------------------
This script reconstructs the masses of the McGill magnetars using the unified
NVG EOS TOV solver and VMF core field amplification. It fixes the progenitor seed
field to a physical 30 kG value and performs a rigorous statistical audit
(subgroup analysis, p-values, bootstrap 95% CIs, age-controlled partial correlations,
Fisher z-transform Bayesian posteriors, and leverage/Cook's distance diagnostics).
"""

import math
import os
import sys
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path to load scan module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvg_magnetar_population_scan import build_sample, CompactObject

# Physical constants
hbar_c = 197.3269804   # MeV·fm
M_N = 939.0            # MeV
n_0 = 0.16             # fm^-3
M_sun = 1.989e30       # kg

# ── 1. NVG Core Model & Unified EOS ──────────────────────────────────
M_Omega_0 = 859.0
M_current_0 = 80.0
kappa_1 = 0.25
kappa_2 = 0.80

alpha_v = 4.0
nu_v = 2.0
C_v_n0 = 100.0

def M_star(n_B: float) -> float:
    x = max(n_B / n_0, 0.0)
    m_omega = M_Omega_0 * (1.0 + kappa_2 * x) ** (-kappa_1 / kappa_2)
    return M_current_0 + m_omega

def nvg_core_eos(n_B: float) -> tuple[float, float]:
    m_eff = M_star(n_B)
    kf = (3.0 * np.pi**2 * n_B) ** (1.0/3.0) * hbar_c
    
    # Kinetic contribution
    ef = math.sqrt(kf**2 + m_eff**2)
    log_term = math.log((kf + ef) / m_eff)
    eps_kin = (kf * ef * (2*kf**2 + m_eff**2) - m_eff**4 * log_term) / (8 * np.pi**2 * hbar_c**3)
    P_kin = (kf * ef * (2*kf**2/3 - m_eff**2) + m_eff**4 * log_term) / (8 * np.pi**2 * hbar_c**3)
    
    # Vacuum potential
    v_pot = (m_eff - M_N) * n_B * 0.5 
    
    # Vector repulsion
    x = n_B / n_0
    v_vector = C_v_n0 * x**2 / (1.0 + alpha_v * x**nu_v)
    eps_v = v_vector * n_B
    P_v = eps_v * (1.0 + nu_v * alpha_v * x**nu_v) / (1.0 + alpha_v * x**nu_v)
    
    eps = eps_kin + v_pot + eps_v
    P = P_kin + P_v
    return eps, P

class UnifiedEOS:
    def __init__(self, n_trans: float = 2.0, delta_eps: float = 350.0):
        self.n_trans = n_trans * n_0
        self.delta_eps = delta_eps
        
        self.n_grid = np.logspace(-4, 1.5, 2000) * n_0
        self.eps_arr = []
        self.p_arr = []
        
        eps_t, P_t = nvg_core_eos(self.n_trans)
        
        for n in self.n_grid:
            if n <= self.n_trans:
                eps, P = nvg_core_eos(n)
                self.eps_arr.append(eps)
                self.p_arr.append(P)
            else:
                eps_quark_start = eps_t + self.delta_eps
                c2 = 1.0/3.0
                eps = eps_quark_start * (n / self.n_trans)**(1.0 + c2)
                P = P_t + c2 * (eps - eps_quark_start)
                self.eps_arr.append(eps)
                self.p_arr.append(P)
                
        self.eps_arr = np.array(self.eps_arr)
        self.p_arr = np.array(self.p_arr)
        
    def get_eps(self, P: float) -> float:
        if not np.isfinite(P) or P < self.p_arr[0] or P > self.p_arr[-1]:
            raise ValueError(
                f"pressure {P!r} outside EOS domain [{self.p_arr[0]}, {self.p_arr[-1]}]"
            )
        return np.interp(P, self.p_arr, self.eps_arr)

# ── 2. TOV Solver ────────────────────────────────────────────────────
def solve_tov(eos: UnifiedEOS, P_center: float) -> tuple[float, float]:
    if not np.isfinite(P_center) or not (eos.p_arr[0] < P_center <= eos.p_arr[-1]):
        raise ValueError("central pressure outside EOS domain")

    def table_energy(pressure: float) -> float:
        # The toy EOS has no crust.  Once integration reaches the validated
        # lower table boundary, continue through vacuum with epsilon=0 rather
        # than extrapolating or clamping the EOS.
        if pressure <= eos.p_arr[0]:
            return 0.0
        return eos.get_eps(pressure)

    def rk4_step(r, m, p, dr):
        if p <= 0:
            return m, 0, 0
        eps = table_energy(p)
        
        def dp_dr(r_val, m_val, p_val, eps_val):
            if r_val < 1e-10:
                return 0.0, 0.0
            k = 1.3234e-6
            eps_k = eps_val * k
            p_k = p_val * k
            m_k = m_val * 1.4766
            
            dm_dr_k = 4.0 * math.pi * r_val**2 * eps_k
            num = (eps_k + p_k) * (m_k + 4.0 * math.pi * r_val**3 * p_k)
            den = r_val * (r_val - 2.0 * m_k)
            
            dp_dr_k = -num / den if den > 0 else 0.0
            dm_dr = dm_dr_k / 1.4766
            dp_dr = dp_dr_k / k
            return dm_dr, dp_dr

        k1_m, k1_p = dp_dr(r, m, p, eps)
        eps_mid = table_energy(p + 0.5 * dr * k1_p)
        k2_m, k2_p = dp_dr(r + 0.5*dr, m + 0.5*dr*k1_m, p + 0.5*dr*k1_p, eps_mid)
        eps_mid = table_energy(p + 0.5 * dr * k2_p)
        k3_m, k3_p = dp_dr(r + 0.5*dr, m + 0.5*dr*k2_m, p + 0.5*dr*k2_p, eps_mid)
        eps_end = table_energy(p + dr * k3_p)
        k4_m, k4_p = dp_dr(r + dr, m + dr*k3_m, p + dr*k3_p, eps_end)
        
        m_new = m + (dr/6.0) * (k1_m + 2*k2_m + 2*k3_m + k4_m)
        p_new = p + (dr/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        return m_new, p_new

    r = 1e-6
    m = 0.0
    p = P_center
    dr = 0.05
    while p > eos.p_arr[0] and r < 100.0:
        m, p = rk4_step(r, m, p, dr)
        r += dr
    return m, r


def mass_from_density(nc_over_n0: float, n_c_arr: np.ndarray, m_arr: np.ndarray) -> float:
    """Map a central density only inside the executed TOV lookup domain."""

    n_values = np.asarray(n_c_arr, dtype=float)
    m_values = np.asarray(m_arr, dtype=float)
    if len(n_values) < 2 or len(n_values) != len(m_values):
        raise ValueError("central-density lookup is empty or mismatched")
    if not np.all(np.isfinite(n_values)) or not np.all(np.isfinite(m_values)):
        raise ValueError("central-density lookup contains non-finite values")
    if np.any(np.diff(n_values) <= 0.0):
        raise ValueError("central-density lookup must be strictly increasing")
    if not np.isfinite(nc_over_n0) or not (n_values[0] <= nc_over_n0 <= n_values[-1]):
        raise ValueError(
            f"central density {nc_over_n0!r} outside lookup domain "
            f"[{n_values[0]}, {n_values[-1]}]"
        )
    return float(np.interp(nc_over_n0, n_values, m_values))

def build_tov_lookup():
    eos = UnifiedEOS(2.0, 350.0)
    pressures = np.logspace(0.5, 3.2, 100)
    n_c_vals = []
    masses = []
    
    for Pc in pressures:
        M, R = solve_tov(eos, Pc)
        n_c = np.interp(Pc, eos.p_arr, eos.n_grid)
        n_c_vals.append(n_c / n_0)
        masses.append(M)
        
    return np.array(n_c_vals), np.array(masses)

# ── 3. Statistical Audit Engine ──────────────────────────────────────
def run_statistical_audit(masses: np.ndarray, b_births: np.ndarray, b_dips: np.ndarray, ages_yr: np.ndarray, names: list = None) -> dict:
    """Run p-value, bootstrap CI, partial correlation, Bayesian posteriors, and leverage audits on the population."""
    N = len(masses)
    
    # 1. Pearson R & p-value (Mass vs. Birth Field)
    r_birth, p_birth = stats.pearsonr(masses, b_births)
    
    # 2. Pearson R & p-value (Mass vs. Observed Dipole Field)
    r_dip, p_dip = stats.pearsonr(masses, b_dips)
    
    # 3. Bootstrap CI for R_birth
    boot_corrs = []
    np.random.seed(42)
    for _ in range(10000):
        idx = np.random.choice(N, size=N, replace=True)
        m_b = masses[idx]
        b_b = b_births[idx]
        if np.std(m_b) > 0 and np.std(b_b) > 0:
            r, _ = stats.pearsonr(m_b, b_b)
            boot_corrs.append(r)
    ci_low, ci_high = np.percentile(boot_corrs, [2.5, 97.5])
    
    # 4. Age-controlled Partial Correlation between Mass and log10(B_birth)
    log_B = np.log10(b_births)
    log_t = np.log10(ages_yr)
    
    r_xy, _ = stats.pearsonr(masses, log_B)
    r_xz, _ = stats.pearsonr(masses, log_t)
    r_yz, _ = stats.pearsonr(log_B, log_t)
    
    # Partial correlation formula
    num = r_xy - r_xz * r_yz
    den = math.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
    r_partial = num / den if den > 0 else 0.0
    
    # p-value of partial correlation (df = N - 3)
    if abs(r_partial) < 1.0:
        t_stat = r_partial * math.sqrt((N - 3) / (1.0 - r_partial**2))
        p_partial = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), N - 3))
    else:
        p_partial = 0.0
        
    # 5. Fisher z-transform Bayesian posteriors
    if N > 3:
        z_obs = 0.5 * np.log((1.0 + r_birth) / (1.0 - r_birth)) if abs(r_birth) < 1.0 else 0.0
        sigma_z = 1.0 / math.sqrt(N - 3)
        p_bayes_flat = stats.norm.cdf(z_obs / sigma_z)
        
        # Weak prior zeta ~ N(0, 1)
        prec_prior = 1.0
        prec_data = 1.0 / (sigma_z**2)
        prec_post = prec_prior + prec_data
        mu_post = (prec_data * z_obs) / prec_post
        sigma_post = 1.0 / math.sqrt(prec_post)
        p_bayes_weak = stats.norm.cdf(mu_post / sigma_post)
    else:
        p_bayes_flat = 0.5
        p_bayes_weak = 0.5
        
    # 6. Leverage & Cook's distance (log10(B_birth) vs Mass)
    h_arr = np.zeros(N)
    cooks_arr = np.zeros(N)
    if N > 2:
        x_mean = np.mean(masses)
        x_dev = masses - x_mean
        sum_x_dev2 = np.sum(x_dev**2)
        if sum_x_dev2 > 0:
            h_arr = 1.0 / N + (x_dev**2) / sum_x_dev2
            
            # Simple regression log10(B_birth) on Mass
            slope, intercept, r_val, p_val, std_err = stats.linregress(masses, log_B)
            y_pred = intercept + slope * masses
            residuals = log_B - y_pred
            mse = np.sum(residuals**2) / (N - 2)
            if mse > 0:
                cooks_arr = (residuals**2 / (2 * mse)) * (h_arr / (1.0 - h_arr)**2)
                
    leverages = {}
    cooks_ds = {}
    if names is not None:
        for i, name in enumerate(names):
            leverages[name] = h_arr[i]
            cooks_ds[name] = cooks_arr[i]
            
    return {
        "N": N,
        "r_birth": r_birth,
        "p_birth": p_birth,
        "r_dip": r_dip,
        "p_dip": p_dip,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "r_partial": r_partial,
        "p_partial": p_partial,
        "p_bayes_flat": p_bayes_flat,
        "p_bayes_weak": p_bayes_weak,
        "leverages": leverages,
        "cooks_ds": cooks_ds
    }

# ── 4. Main Pipeline ──────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("      NVG POPULATION-LEVEL AUDIT: REFRACTED MASS-FIELD CORRELATION")
    print("=" * 80)
    
    print("1. Integrating TOV equations for NVG EOS...")
    n_c_arr, m_arr = build_tov_lookup()
    M_max = np.max(m_arr)
    print(f"   TOV Grid complete. Max physical mass: {M_max:.2f} M_sun")

    sample, note = build_sample()
    # Filter out long-period outlier
    magnetars = [o for o in sample if o.family == "magnetar" and o.name != "1E 161348-5055"]
    print(f"2. Magnetar population input: {len(magnetars)} sources (outlier removed); source={note}")
    if "NON-EVIDENCE" in note:
        print("   WARNING: fallback rows are illustrative only; population inference is withheld.")
    
    # This benchmark is an explicit model assumption; no source metadata for the
    # 30 kG value is committed, so it is not counted as an independent input.
    B_seed_surf = 30000.0  # G
    tau_d = 1.0e4          # yr, decay timescale
    gamma = 0.5            # decay exponent
    flux_gain = (1.0e11 / 1.2e6)**2 # flux conservation gain ~6.94e9
    B_seed_NS = B_seed_surf * flux_gain # ~2.08e14 G
    
    print(f"3. Using assumed progenitor surface field: {B_seed_surf/1e3:.1f} kG (unverified input)")
    print(f"   NS fossil seed field                : {B_seed_NS:.2e} G")
    print(f"   Field decay model                   : B_birth = B_dip * (1 + t/10k yr)^0.5")
    print()
    
    print(f"  {'Source':<20} | {'B_dip (G)':<10} | {'Age (kyr)':<10} | {'B_birth (G)':<11} | {'Gamma_req':<10} | {'Reconstructed M_NS':<18}")
    print("-" * 90)
    
    reconstructed_data = []
    for obj in magnetars:
        tau_yr = obj.tau_kyr * 1000.0 if obj.tau_kyr is not None else 10000.0
        B_birth = obj.b_dip_g * ((1.0 + tau_yr / tau_d)**gamma)
        gamma_req = B_birth / B_seed_NS
        
        if gamma_req > 1.0:
            rho_c_ratio = (gamma_req**1.6 - 1.0) / 0.80
        else:
            rho_c_ratio = 0.0
            
        try:
            m_ns = mass_from_density(rho_c_ratio, n_c_arr, m_arr)
            mass_status = "supported"
        except ValueError as exc:
            m_ns = None
            mass_status = f"unsupported ({exc})"
        
        reconstructed_data.append({
            "name": obj.name,
            "b_dip": obj.b_dip_g,
            "tau_yr": tau_yr,
            "B_birth": B_birth,
            "gamma_req": gamma_req,
            "mass": m_ns,
            "mass_status": mass_status,
        })
        
        mass_text = f"{m_ns:15.3f} M_sun" if m_ns is not None else "unsupported lookup"
        print(f"  {obj.name:<20} | {obj.b_dip_g:10.2e} | {tau_yr/1000.0:10.1f} | {B_birth:11.2e} | {gamma_req:10.2f} | {mass_text} [{mass_status}]")
        
    # Extract data arrays
    supported_data = [d for d in reconstructed_data if d["mass"] is not None]
    if len(supported_data) < 4:
        raise ValueError("fewer than four magnetar rows have supported EOS masses")
    names = np.array([d["name"] for d in supported_data])
    masses = np.array([d["mass"] for d in supported_data])
    b_births = np.array([d["B_birth"] for d in supported_data])
    b_dips = np.array([d["b_dip"] for d in supported_data])
    ages = np.array([d["tau_yr"] for d in supported_data])
    
    # ── Subgroup Partition ────────────────────────────────────────────
    # Young magnetars: age < 10,000 years
    young_mask = ages < 10000.0
    
    print("\n" + "=" * 80)
    print("      STATISTICAL AUDIT REPORT")
    print("=" * 80)
    
    groups = [
        ("FULL POPULATION (N=" + str(len(masses)) + ")", masses, b_births, b_dips, ages),
        ("YOUNG POPULATION (<10 kyr, N=" + str(len(masses[young_mask])) + ")", masses[young_mask], b_births[young_mask], b_dips[young_mask], ages[young_mask])
    ]
    
    audit_results = {}
    for title, m, bb, bd, ag in groups:
        names_list = list(names) if "FULL" in title else list(names[young_mask])
        audit = run_statistical_audit(m, bb, bd, ag, names=names_list)
        audit_results[title] = audit
        
        print(f"--- {title} ---")
        print(f"  Mass Range                        : {np.min(m):.3f} -- {np.max(m):.3f} M_sun")
        print(f"  Pearson R (Mass vs. Dipole)       : {audit['r_dip']:.4f} (p = {audit['p_dip']:.2e})")
        print(f"  Pearson R (Mass vs. Birth B)      : {audit['r_birth']:.4f} (p = {audit['p_birth']:.2e})")
        print(f"  Bootstrap 95% CI for R_birth      : [{audit['ci_low']:.4f}, {audit['ci_high']:.4f}]")
        print(f"  Partial R (Mass vs. B | Age)      : {audit['r_partial']:.4f} (p = {audit['p_partial']:.2e})")
        
        if "YOUNG" in title:
            print(f"  Bayesian P(R_pop > 0 | data) (flat)   : {audit['p_bayes_flat']:.4f} ({audit['p_bayes_flat']*100:.2f}%)")
            print(f"  Bayesian P(R_pop > 0 | data) (N(0,1)) : {audit['p_bayes_weak']:.4f} ({audit['p_bayes_weak']*100:.2f}%)")
            # Print leverage top 3
            sorted_cook = sorted(audit['cooks_ds'].items(), key=lambda x: x[1], reverse=True)
            print("  Top Influence Points (Cook's distance):")
            for name, cd in sorted_cook[:3]:
                h_val = audit['leverages'][name]
                print(f"    - {name:<22}: Cook's D = {cd:.4f}, Leverage h = {h_val:.4f}")
        print()
        
    # ── Outlier robustness analysis ───────────────────────────────────
    print("--- Outlier robustness analysis (Young Subgroup) ---")
    young_names = names[young_mask]
    young_masses = masses[young_mask]
    young_b_births = b_births[young_mask]
    young_log_B = np.log10(young_b_births)
    
    # 1. Clean analysis (excluding SGR 1806-20)
    clean_mask = young_names != "SGR 1806-20"
    r_clean, p_clean = stats.pearsonr(young_masses[clean_mask], young_b_births[clean_mask])
    r_clean_log, p_clean_log = stats.pearsonr(young_masses[clean_mask], young_log_B[clean_mask])
    print(f"  Excluding SGR 1806-20 (N={len(young_masses[clean_mask])}):")
    print(f"    Pearson R (Mass vs. Birth B)        : {r_clean:.4f} (p = {p_clean:.4f})")
    print(f"    Pearson R (Mass vs. log10(Birth B)) : {r_clean_log:.4f} (p = {p_clean_log:.4f})")
    
    print("  No post-hoc mass anchors are applied; all inferential statistics use")
    print("  masses reconstructed from the executed EOS lookup. Anchor-based")
    print("  sensitivity claims are excluded because they are not independent.")
    print()
    
    # ── Figure Generation ─────────────────────────────────────────────
    print("4. Generating diagnostic figures...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Scatter log10(B_birth) vs Mass
    young_audit = next(val for key, val in audit_results.items() if "YOUNG" in key)
    cooks = np.array([young_audit["cooks_ds"].get(n, 0.0) for n in young_names])
    sizes = 40 + cooks * 400
    
    ax1.scatter(young_masses, young_log_B, s=sizes, color="#3498db", alpha=0.7, edgecolor="black", label="Reconstructed")
    
    # Regression line for standard
    slope, intercept, r_val, p_val, std_err = stats.linregress(young_masses, young_log_B)
    x_grid = np.linspace(1.0, 2.4, 100)
    y_grid = intercept + slope * x_grid
    ax1.plot(x_grid, y_grid, color="#2c3e50", linestyle="-", lw=1.5, label=f"Standard Fit (R={r_val:.3f})")
    
    # Labels
    for i, name in enumerate(young_names):
        if name in ["SGR 1806-20", "Swift J1818.0-1607", "Swift J1834.9-0846", "1E 1048.1-5937", "SGR 1900+14"]:
            ax1.annotate(name.split()[-1], (young_masses[i], young_log_B[i]), textcoords="offset points", xytext=(5,5), fontsize=8, weight="bold")
            
    ax1.set_xlabel("Neutron Star Mass ($M_{\\odot}$)", fontsize=10)
    ax1.set_ylabel("$\\log_{10}(B_{\\text{birth}} / \\text{G})$", fontsize=10)
    ax1.set_title("A. Mass vs. Birth Magnetic Field (Young Subgroup)", fontsize=12, weight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    
    # Panel B: Bootstrap distribution comparison
    boot_corrs_std = []
    np.random.seed(42)
    for _ in range(10000):
        idx = np.random.choice(len(young_masses), size=len(young_masses), replace=True)
        if np.std(young_masses[idx]) > 0 and np.std(young_log_B[idx]) > 0:
            r_s, _ = stats.pearsonr(young_masses[idx], young_log_B[idx])
            boot_corrs_std.append(r_s)
            
    ax2.hist(boot_corrs_std, bins=50, density=True, color="#3498db", alpha=0.3, label="Standard Bootstrap")
    
    # Add density curves
    kde_std = stats.gaussian_kde(boot_corrs_std)
    r_grid = np.linspace(-0.4, 1.0, 500)
    ax2.plot(r_grid, kde_std(r_grid), color="#2980b9", lw=1.5)
    
    # CIs
    ci_std = [young_audit["ci_low"], young_audit["ci_high"]]
    ax2.axvline(ci_std[0], color="#2980b9", linestyle="--", lw=1.2)
    ax2.axvline(ci_std[1], color="#2980b9", linestyle="--", lw=1.2)
    
    # Zero line
    ax2.axvline(0, color="grey", linestyle="-", lw=1, alpha=0.7)
    
    ax2.text(ci_std[0]-0.01, 1, f"Std CI: [{ci_std[0]:.2f}, {ci_std[1]:.2f}]", color="#2980b9", rotation=90, ha="right", fontsize=8)
    
    ax2.set_xlabel("Pearson Correlation Coefficient ($R$)", fontsize=10)
    ax2.set_ylabel("Probability Density", fontsize=10)
    ax2.set_title("B. Bootstrap $R$ Distribution Comparison", fontsize=12, weight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=8)
    
    plt.tight_layout()
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "article", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, "fig_magnetar_young_diagnostics.png"), dpi=200)
    fig.savefig(os.path.join(fig_dir, "fig_magnetar_young_diagnostics.pdf"))
    plt.close(fig)
    print("   Diagnostics figure saved to article/figures/fig_magnetar_young_diagnostics.{png,pdf}")
    print()

    print("  INTERPRETATION: values above are descriptive outputs of a toy EOS/decay model.")
    print("  The catalog source note is retained; no mass anchors or independent significance")
    print("  claims are applied, and a fallback sample is non-evidence for population inference.")
    print("=" * 80)
    
    # Save audit report to text file
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "nvg_magnetar_mass_correlation_output.txt")
    
    L = [
        "NVG Magnetar Mass-Field Correlation Analysis Output (Audited)",
        "===========================================================",
        "Generated by: verification/nvg_magnetar_mass_correlation.py",
        f"Assumed progenitor surface field (unverified): {B_seed_surf:.1f} G",
        f"NS fossil seed field: {B_seed_NS:.2e} G",
        f"Catalog source: {note}",
        "Evidence status: descriptive model output; not an independent population prediction",
        f"EOS lookup status: {len(supported_data)}/{len(reconstructed_data)} rows in domain; out-of-domain rows excluded (no endpoint substitution)",
        ""
    ]
    for title, audit in audit_results.items():
        L.append(f"--- {title} ---")
        L.append(f"Pearson R (Mass vs. Dipole): {audit['r_dip']:.4f} (p={audit['p_dip']:.2e})")
        L.append(f"Pearson R (Mass vs. Birth B): {audit['r_birth']:.4f} (p={audit['p_birth']:.2e})")
        L.append(f"Bootstrap 95% CI: [{audit['ci_low']:.4f}, {audit['ci_high']:.4f}]")
        L.append(f"Partial R (Mass vs. B | Age): {audit['r_partial']:.4f} (p={audit['p_partial']:.2e})")
        L.append(f"Bayesian P(R > 0) (flat prior): {audit['p_bayes_flat']:.4f}")
        L.append(f"Bayesian P(R > 0) (N(0,1) prior): {audit['p_bayes_weak']:.4f}")
        L.append("")
        
    L.append("--- Outlier robustness (no post-hoc anchors) ---")
    L.append(f"Young subgroup excluding SGR 1806-20: R = {r_clean:.4f} (p = {p_clean:.4f})")
    L.append("Post-hoc mass-anchor analyses: excluded (not independent evidence)")
    L.append("")
    
    for d in reconstructed_data:
        mass_text = f"{d['mass']:.3f} M_sun" if d["mass"] is not None else "UNSUPPORTED_EOS_DOMAIN"
        L.append(f"{d['name']}: B_dip={d['b_dip']:.2e} G, Age={d['tau_yr']/1e3:.1f} kyr, B_birth={d['B_birth']:.2e} G, Mass={mass_text}")
        
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"Saved results to {out_path}")
 
if __name__ == "__main__":
    main()
