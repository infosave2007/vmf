# NVG — Vacuum-Deviation Framework for Hadron Mass and Dense Matter

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)

## Overview

This repository contains the research materials for the **Null-Vector Gravity (NVG)** program — a phenomenological framework that interprets physical mass as a measure of local deviation from the vacuum ground state of QCD.

### Key Idea

Approximately **91% of the nucleon mass** does not come from the Higgs-generated current quark masses, but from nonperturbative QCD dynamics (gluonic energy, trace anomaly, confinement). This repository formalizes this observation through a single dimensionless parameter:

$$f_\Omega^{(N)} = 1 - \frac{1}{M_N}\sum_{q} \sigma_{qN} \approx 0.91 \pm 0.02$$

and explores its consequences for dense nuclear matter equations of state.

### What This Repository Contains

| Directory | Contents |
|-----------|----------|
| `article/` | Scientific article (Russian) with full derivations |
| `code/` | All computational prototypes (Python, reproducible) |
| `docs/` | Working documents, theoretical notes, letters |
| `letters/` | Correspondence with experimentalists (Podkletnov, Modanese) |

### What Is Proven

1. **Nucleon mass decomposition** via lattice QCD sigma terms: $M_{\Omega,0} = 859 \pm 8$ MeV
2. **Weak-field GR recovery**: Poisson equation, Shapiro delay, light deflection
3. **Local Lorentz invariance**: speed of light = $c$ in all local inertial frames
4. **Bookkeeping identity**: $E_\Omega = M_\Omega c^2$ for stationary systems
5. **Core-level causality** achieved with density-dependent vector saturation ($c_s^2 \leq 1$)

### What Is NOT Proven

- This is **not** a replacement for General Relativity
- This is **not** a solution to the cosmological constant problem
- Dense matter EOS does **not** yet reproduce realistic $R_{1.4}$ and $M_{\max}$
- Cyclic cosmology is **not** observationally confirmed

## Quick Start

```bash
# Install dependencies
pip install numpy scipy

# Run the main EOS verification
python code/nvg_eos_beta_saturated_vector.py

# Run with TOV scan
python code/nvg_eos_beta_saturated_vector.py --with-tov

# Run the hadron mass fraction calculation
python code/nvg_hadron_mass_fractions.py

# Run all verification tests
python code/run_all_checks.py
```

## Citation

If you use this work, please cite:

```
Kirchenko, O. (2026). Vacuum-deviation variables in metric gravity:
residual nonperturbative mass fractions and dense matter EOS prototypes.
NVG Research Program. https://github.com/[username]/NVG-Research
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
