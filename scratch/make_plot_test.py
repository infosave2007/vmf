import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Data
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

# Leverage & Cook's distance for log_B vs Mass
x = masses
y = log_B
x_mean = np.mean(x)
x_dev = x - x_mean
sum_x_dev2 = np.sum(x_dev**2)
h = 1.0 / N + (x_dev**2) / sum_x_dev2
slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
y_pred = intercept + slope * x
residuals = y - y_pred
mse = np.sum(residuals**2) / (N - 2)
cooks_d = (residuals**2 / (2 * mse)) * (h / (1 - h)**2)

# Anchor masses
anchor_masses = masses.copy()
for i, name in enumerate(names):
    if name == "SGR 1806-20":
        anchor_masses[i] = 2.10
    elif name == "1E 1048.1-5937":
        anchor_masses[i] = 1.60
    elif name == "SGR 1900+14":
        anchor_masses[i] = 1.45

# Bootstrap calculations
boot_corrs_std = []
boot_corrs_anc = []
np.random.seed(42)
for _ in range(10000):
    idx = np.random.choice(N, size=N, replace=True)
    if np.std(masses[idx]) > 0 and np.std(log_B[idx]) > 0:
        r_s, _ = stats.pearsonr(masses[idx], log_B[idx])
        boot_corrs_std.append(r_s)
    if np.std(anchor_masses[idx]) > 0 and np.std(log_B[idx]) > 0:
        r_a, _ = stats.pearsonr(anchor_masses[idx], log_B[idx])
        boot_corrs_anc.append(r_a)

ci_std = np.percentile(boot_corrs_std, [2.5, 97.5])
ci_anc = np.percentile(boot_corrs_anc, [2.5, 97.5])

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Scatter log10(B_birth) vs Mass
sizes = 40 + cooks_d * 400
ax1.scatter(x, y, s=sizes, color="#3498db", alpha=0.7, edgecolor="black", label="Young Magnetars")
ax1.scatter(anchor_masses[[9, 2, 13]], log_B[[9, 2, 13]], s=100, color="#e74c3c", marker="*", edgecolor="black", label="Independent Anchors")

# Fit line for standard
x_grid = np.linspace(1.0, 2.4, 100)
y_grid = intercept + slope * x_grid
ax1.plot(x_grid, y_grid, color="#2c3e50", linestyle="-", lw=1.5, label=f"Standard Fit (R={r_val:.3f})")

# Labels
for i, name in enumerate(names):
    if name in ["SGR 1806-20", "Swift J1818.0-1607", "Swift J1834.9-0846", "1E 1048.1-5937", "SGR 1900+14"]:
        ax1.annotate(name.split()[-1], (x[i], y[i]), textcoords="offset points", xytext=(5,5), fontsize=8, weight="bold")

ax1.set_xlabel("Neutron Star Mass ($M_{\\odot}$)", fontsize=10)
ax1.set_ylabel("$\\log_{10}(B_{\\text{birth}} / \\text{G})$", fontsize=10)
ax1.set_title("A. Mass vs. Birth Magnetic Field (Young Subgroup)", fontsize=12, weight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower right", frameon=True, fontsize=9)

# Panel B: Bootstrap distribution
ax2.hist(boot_corrs_std, bins=50, density=True, color="#3498db", alpha=0.4, label="Standard Bootstrap")
ax2.hist(boot_corrs_anc, bins=50, density=True, color="#e74c3c", alpha=0.4, label="Anchored Bootstrap")

# Add density curves
kde_std = stats.gaussian_kde(boot_corrs_std)
kde_anc = stats.gaussian_kde(boot_corrs_anc)
r_grid = np.linspace(-0.4, 1.0, 500)
ax2.plot(r_grid, kde_std(r_grid), color="#2980b9", lw=2)
ax2.plot(r_grid, kde_anc(r_grid), color="#c0392b", lw=2)

# CIs
ax2.axvline(ci_std[0], color="#2980b9", linestyle="--", lw=1.5)
ax2.axvline(ci_std[1], color="#2980b9", linestyle="--", lw=1.5)
ax2.axvline(ci_anc[0], color="#c0392b", linestyle="--", lw=1.5)
ax2.axvline(ci_anc[1], color="#c0392b", linestyle="--", lw=1.5)

# Zero line
ax2.axvline(0, color="grey", linestyle="-", lw=1, alpha=0.7)

ax2.text(ci_std[0]-0.02, 1, f"CI: [{ci_std[0]:.2f}, {ci_std[1]:.2f}]", color="#2980b9", rotation=90, ha="right", fontsize=9)
ax2.text(ci_anc[0]+0.02, 1, f"CI: [{ci_anc[0]:.2f}, {ci_anc[1]:.2f}]", color="#c0392b", rotation=90, ha="left", fontsize=9)

ax2.set_xlabel("Pearson Correlation Coefficient ($R$)", fontsize=10)
ax2.set_ylabel("Probability Density", fontsize=10)
ax2.set_title("B. Bootstrap $R$ Distribution comparison", fontsize=12, weight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left", frameon=True, fontsize=9)

plt.tight_layout()
fig.savefig("/Users/oleg/Documents/NVG-Research/article/figures/fig_magnetar_young_diagnostics.png", dpi=200)
fig.savefig("/Users/oleg/Documents/NVG-Research/article/figures/fig_magnetar_young_diagnostics.pdf")
print("Done generating test figure!")
