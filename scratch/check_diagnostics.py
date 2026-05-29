import numpy as np
import scipy.stats as stats

# Data for the 15 young magnetars (< 10 kyr)
data = [
    {"name": "CXOU J010043.1-721134", "mass": 1.609, "B_birth": 5.09e14},
    {"name": "SGR 0526-66", "mass": 1.528, "B_birth": 6.47e14},
    {"name": "1E 1048.1-5937", "mass": 1.642, "B_birth": 4.65e14},
    {"name": "1E 1547.0-5408", "mass": 2.025, "B_birth": 3.29e14},
    {"name": "PSR J1622-4950", "mass": 1.993, "B_birth": 3.25e14},
    {"name": "SGR 1627-41", "mass": 1.100, "B_birth": 2.48e14},
    {"name": "1RXS J170849.0-400910", "mass": 1.530, "B_birth": 6.44e14},
    {"name": "CXOU J171405.7-381031", "mass": 1.601, "B_birth": 5.24e14},
    {"name": "SGR J1745-2900", "mass": 1.418, "B_birth": 2.76e14},
    {"name": "SGR 1806-20", "mass": 2.298, "B_birth": 1.98e15},
    {"name": "Swift J1818.0-1607", "mass": 2.208, "B_birth": 3.58e14},
    {"name": "Swift J1834.9-0846", "mass": 1.100, "B_birth": 1.74e14},
    {"name": "1E 1841-045", "mass": 2.298, "B_birth": 8.49e14},
    {"name": "SGR 1900+14", "mass": 1.473, "B_birth": 7.31e14},
    {"name": "SGR 1935+2154", "mass": 1.100, "B_birth": 2.54e14}
]

names = [d["name"] for d in data]
masses = np.array([d["mass"] for d in data])
b_births = np.array([d["B_birth"] for d in data])
log_B = np.log10(b_births)

N = len(data)

# Let's test with both B_birth and log10(B_birth)
for label, B_val in [("B_birth (G)", b_births), ("log10(B_birth)", log_B)]:
    print(f"\n--- Regression of {label} (Y) on Mass (X) ---")
    x = masses
    y = B_val
    x_mean = np.mean(x)
    x_dev = x - x_mean
    sum_x_dev2 = np.sum(x_dev**2)
    
    # Leverage
    h = 1.0 / N + (x_dev**2) / sum_x_dev2
    
    # Fit
    slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
    y_pred = intercept + slope * x
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (N - 2)
    
    # Cook's Distance
    cooks_d = (residuals**2 / (2 * mse)) * (h / (1 - h)**2)
    
    # Print table sorted by Cook's distance
    print(f"{'Source':<24} | {'Mass':<6} | {'B_birth':<10} | {'Leverage (h)':<12} | {'Cooks D':<10}")
    print("-" * 75)
    sorted_idx = np.argsort(cooks_d)[::-1]
    for idx in sorted_idx:
        print(f"{names[idx]:<24} | {masses[idx]:<6.3f} | {b_births[idx]:10.2e} | {h[idx]:12.4f} | {cooks_d[idx]:10.4f}")

# Let's also compute Fisher z posterior
r_obs = 0.5436
z_obs = 0.5 * np.log((1 + r_obs) / (1 - r_obs))
sigma_z = 1.0 / np.sqrt(N - 3)
p_flat = stats.norm.cdf(z_obs / sigma_z)
print(f"\nFisher z transform for R = {r_obs}, N = {N}:")
print(f"z_obs = {z_obs:.4f}, sigma_z = {sigma_z:.4f}")
print(f"Posterior P(R_pop > 0 | data) with flat prior: {p_flat:.4f} ({p_flat*100:.2f}%)")

# Weak conjugate normal prior zeta ~ N(0, 1)
prec_prior = 1.0
prec_data = 1.0 / (sigma_z**2)
prec_post = prec_prior + prec_data
mu_post = (prec_data * z_obs) / prec_post
sigma_post = 1.0 / np.sqrt(prec_post)
p_weak = stats.norm.cdf(mu_post / sigma_post)
print(f"Posterior P(R_pop > 0 | data) with N(0,1) prior: {p_weak:.4f} ({p_weak*100:.2f}%)")

# Calculate Pearson correlation coefficient without SGR 1806-20
clean_masses = np.array([d["mass"] for d in data if d["name"] != "SGR 1806-20"])
clean_b_births = np.array([d["B_birth"] for d in data if d["name"] != "SGR 1806-20"])
r_clean, p_clean = stats.pearsonr(clean_masses, clean_b_births)
print(f"\nYoung subgroup excluding SGR 1806-20 (N=14):")
print(f"Pearson R (Mass vs. Birth B): {r_clean:.4f} (p = {p_clean:.4f})")

r_clean_log, p_clean_log = stats.pearsonr(clean_masses, np.log10(clean_b_births))
print(f"Pearson R (Mass vs. log10(Birth B)): {r_clean_log:.4f} (p = {p_clean_log:.4f})")

# Test independent mass anchor points in bootstrap CI
# Let's define independent mass estimates for a few key objects:
# SGR 1806-20: M = 2.1 M_sun (independent constraint from cluster Westerlund 1 / LBV wind cluster, mass ~2.1)
# 1E 1048.1-5937: M = 1.6 M_sun (independent constraint from stellar wind bubble, mass ~1.6)
# SGR 1900+14: M = 1.45 M_sun (independent constraint from cluster Cl 1900+14)
anchor_masses = masses.copy()
# Find indices
for i, name in enumerate(names):
    if name == "SGR 1806-20":
        anchor_masses[i] = 2.10
    elif name == "1E 1048.1-5937":
        anchor_masses[i] = 1.60
    elif name == "SGR 1900+14":
        anchor_masses[i] = 1.45

# Let's run bootstrap CI with anchor masses vs log10(B_birth)
boot_corrs_anchor = []
np.random.seed(42)
for _ in range(10000):
    idx = np.random.choice(N, size=N, replace=True)
    m_b = anchor_masses[idx]
    b_b = log_B[idx]
    if np.std(m_b) > 0 and np.std(b_b) > 0:
        r, _ = stats.pearsonr(m_b, b_b)
        boot_corrs_anchor.append(r)
ci_low_anc, ci_high_anc = np.percentile(boot_corrs_anchor, [2.5, 97.5])
r_anc, p_anc = stats.pearsonr(anchor_masses, log_B)
print(f"\nYoung subgroup with 3 independent mass anchors (N=15):")
print(f"Pearson R (Mass vs. log10(Birth B)): {r_anc:.4f} (p = {p_anc:.4f})")
print(f"Bootstrap 95% CI: [{ci_low_anc:.4f}, {ci_high_anc:.4f}]")

# Sensitivity test: only 2 anchors (1E 1048.1-5937 and SGR 1900+14), SGR 1806-20 reconstructed (not anchored)
anchor_masses_2 = masses.copy()
for i, name in enumerate(names):
    if name == "1E 1048.1-5937":
        anchor_masses_2[i] = 1.60
    elif name == "SGR 1900+14":
        anchor_masses_2[i] = 1.45

boot_corrs_anchor_2 = []
np.random.seed(42)
for _ in range(10000):
    idx = np.random.choice(N, size=N, replace=True)
    m_b = anchor_masses_2[idx]
    b_b = log_B[idx]
    if np.std(m_b) > 0 and np.std(b_b) > 0:
        r, _ = stats.pearsonr(m_b, b_b)
        boot_corrs_anchor_2.append(r)
ci_low_anc_2, ci_high_anc_2 = np.percentile(boot_corrs_anchor_2, [2.5, 97.5])
r_anc_2, p_anc_2 = stats.pearsonr(anchor_masses_2, log_B)
print(f"\nYoung subgroup with 2 independent mass anchors (N=15, SGR 1806-20 reconstructed):")
print(f"Pearson R (Mass vs. log10(Birth B)): {r_anc_2:.4f} (p = {p_anc_2:.4f})")
print(f"Bootstrap 95% CI: [{ci_low_anc_2:.4f}, {ci_high_anc_2:.4f}]")



