#!/usr/bin/env python3
"""Exclusion figure for the bounded-T baryogenesis no-go paper.

(T_max, Lambda) plane for the EFT operator (qqql)/Lambda^2:
  - below Lambda_viable(T_max): cosmologically capable of generating eta_B
    (spontaneous baryogenesis with maximal thermal bias, epsilon_min);
  - below Lambda_p: proton decays faster than the Super-K bound;
  - the two regions never overlap below T ~ 1e13 GeV: the EFT desert.
  - vertical line: SM sphaleron threshold (the SM's own escape).
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_PL, G_STAR = 1.22e19, 20.0
ETA = 8.6e-11
TAU_SK_GEV = 2.4e34 * 3.156e7 / 6.58e-25
M_P = 0.938

eta_eq_max = 45.0 / (12.0 * math.pi * G_STAR)          # mu = pi T bias
eps_min = ETA / eta_eq_max
LAM_P = (TAU_SK_GEV * M_P ** 5) ** 0.25

T = np.logspace(math.log10(0.005), 4, 400)             # 5 MeV .. 10 TeV
lam_eq = (T ** 3 * M_PL / (1.66 * math.sqrt(G_STAR))) ** 0.25
lam_viable = lam_eq * eps_min ** -0.25

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.fill_between(T, 1e-1, lam_viable, alpha=0.25, color="tab:blue",
                label=r"cosmologically viable: $\Gamma/H \geq \epsilon_{\min}$")
ax.axhspan(1e-1, LAM_P, alpha=0.18, color="tab:red",
           label=r"proton unstable: $\tau_p < 2.4\times10^{34}$ yr")
ax.plot(T, lam_viable, color="tab:blue", lw=2)
ax.axhline(LAM_P, color="tab:red", lw=2)
ax.axvline(130.0, color="k", ls="--", lw=1.5)
ax.text(150, 3e3, "SM sphalerons\navailable", fontsize=9)
ax.text(0.02, 3e17, r"proton safe, cosmologically irrelevant", fontsize=9)
ax.text(0.02, 8e4, "EFT desert:\nno viable scale", fontsize=10, weight="bold")
ax.annotate("", xy=(0.432, LAM_P), xytext=(0.432, lam_viable[np.argmin(abs(T-0.432))]),
            arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
ax.text(0.55, 3e10, r"$10^{10}$ in $\Lambda$" + "\n" + r"($\sim$40 orders in $\tau_p$)",
        fontsize=9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(5e-3, 1e4); ax.set_ylim(1e2, 1e18)
ax.set_xlabel(r"$T_{\max}$  [GeV]")
ax.set_ylabel(r"$\Lambda$  [GeV]   for  $(qqq\ell)/\Lambda^2$")
ax.set_title("Perturbative baryogenesis below the sphaleron threshold")
ax.legend(loc="center right", fontsize=8)
fig.tight_layout()
fig.savefig("papers/bounded_T_nogo/fig_exclusion.png", dpi=160)
print(f"eps_min = {eps_min:.2e}; Lambda_p = {LAM_P:.2e} GeV")
i = np.argmin(abs(T - 0.432))
print(f"at T_max = 432 MeV: Lambda_viable = {lam_viable[i]:.2e} GeV; "
      f"gap = {LAM_P/lam_viable[i]:.1e} in Lambda = {4*math.log10(LAM_P/lam_viable[i]):.0f} orders in tau_p")
T_close = (LAM_P ** 4 * 1.66 * math.sqrt(G_STAR) * eps_min / M_PL) ** (1.0/3.0)
print(f"EFT desert closes only at T = {T_close:.1e} GeV (academic; sphalerons rescue at 130 GeV)")
