#!/usr/bin/env python3
"""
The phase-winding topological bit: quantify how an integer winding number
survives continuous local noise, and locate the protection threshold.

A phase field theta_i on a ring of N sites carries a winding
  W = (1/2pi) sum_i wrap(theta_{i+1}-theta_i),  wrap -> (-pi,pi].
W is an INTEGER that cannot change under continuous deformation: it only
changes through a discrete 'phase slip' when a link difference crosses pi.
This is the honest, computable core of "information stored in a topological
invariant is protected against continuous deformation."
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

def wrap(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def winding(theta):
    d = wrap(np.diff(np.concatenate([theta, theta[:1]])))
    return int(round(d.sum()/(2*np.pi)))

def survival(N, sigma, trials=20000):
    base = 2*np.pi*np.arange(N)/N          # smooth W = 1 state
    ok = 0
    for _ in range(trials):
        noisy = base + rng.normal(0, sigma, N)
        ok += (winding(noisy) == 1)
    return ok/trials

if __name__ == "__main__":
    N = 64
    sigmas = np.linspace(0.1, 3.0, 30)
    P = np.array([survival(N, s) for s in sigmas])
    err = 1 - P

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(sigmas, P, 'o-', color='#1f77b4', lw=2, ms=4)
    ax[0].set_xlabel(r"local phase noise  $\sigma$  [rad]")
    ax[0].set_ylabel(r"P(winding preserved)")
    ax[0].set_title(f"(a) topological-bit survival, N={N}")
    ax[0].grid(alpha=0.3)
    # analytic per-link slip probability (link diff ~ N(0, 2 sigma^2), slip>|pi|)
    from math import erfc, sqrt
    p_link = np.array([erfc(np.pi/(sqrt(2)*sqrt(2)*s))/1 for s in sigmas])
    floor = 1.0/20000
    meas = err > floor
    ax[1].semilogy(sigmas[meas], err[meas], 'o', color='#d62728',
                   label="Monte-Carlo (resolved)")
    ax[1].semilogy(sigmas, np.minimum(N*p_link,1.0), '-', color='k',
                   label=r"$N\cdot\mathrm{erfc}(\pi/2\sigma)$ phase-slip")
    ax[1].axhline(floor, color='gray', ls=':', lw=1)
    ax[1].text(2.2, floor*1.5, "MC floor = 1/trials", fontsize=7, color='gray')
    ax[1].axvline(0.6, color='#1f77b4', ls='--', lw=1)
    ax[1].text(0.63, 1e-40, r"threshold $\sigma\!\approx\!0.6$", fontsize=8,
               color='#1f77b4', rotation=90, va='center')
    ax[1].set_ylim(1e-60, 2)
    ax[1].set_xlabel(r"local phase noise  $\sigma$  [rad]")
    ax[1].set_ylabel("bit-error rate  (1 - P)")
    ax[1].set_title("(b) exponential protection below threshold")
    ax[1].legend(fontsize=8, loc='lower right'); ax[1].grid(alpha=0.3, which='major')
    fig.tight_layout()
    fig.savefig("topological_info/fig_winding_bit.pdf")
    fig.savefig("topological_info/fig_winding_bit.png", dpi=150)

    # report threshold: sigma where error crosses 1%
    i = np.argmin(abs(err-0.01))
    print(f"protection threshold (1% error): sigma ~ {sigmas[i]:.2f} rad")
    for s in (0.5, 1.0, 1.5, 2.0):
        print(f"  sigma={s}: P(preserved)={survival(N,s):.4f}")
    # scaling with N at fixed sigma
    print("N-scaling at sigma=1.0 (error grows linearly with N):")
    for n in (16, 64, 256):
        print(f"  N={n:>3}: error = {1-survival(n,1.0):.4f}")
    print("figure written")
