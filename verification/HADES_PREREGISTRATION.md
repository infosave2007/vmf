# Preregistered architecture selector for the HADES in-medium analysis

**Fixed:** 2026-07-05, before any comparison with HADES Gen2 data and
before the collaboration's announced 2026-27 in-medium vector-meson
analysis. Any later change to these windows will be visible in the
repository history.

## Context

The thermodynamic-consistency audit of the VMF/NVG equation-of-state
sector (see `nvg_selfbound_gate.py`, `nvg_eos_fork_a.py`,
`nvg_eos_fork_b_nl.py`, `nvg_fork_b_full_chain.py`) leaves exactly two
consistent architectures for condensate melting in dense matter. Both
reproduce the melted thermal continuum observed by HADES (Nature Phys.
15, 1040); they differ in the in-medium pole mass entering the excess
line shape:

| | Architecture A (residual melting) | Architecture B (scalar-field melting) |
|---|---|---|
| melting at saturation | pinned by E/A = −16: ~2% | derived: m*/M = 0.650 |
| in-medium ρ pole at ⟨n⟩ ≈ 1.5 n₀ | ≈ 770 MeV (vacuum-like) | ≈ 735 MeV (mapping λ ≤ 0.09, NA60-bounded) |
| line shape | broadening-only | broadening + pole pull |

Templates and the discrimination power (3σ with 3,000–3,800 excess
counts) are computed in `nvg_hades_lineshape_feasibility.py`.

## The preregistered criterion

Observable: the in-medium pole-mass parameter `m_pole` from a full
line-shape fit of the acceptance-corrected excess spectrum
(0.2–0.8 GeV), with the width free.

| Fitted `m_pole` | Verdict |
|---|---|
| ≥ 755 MeV | **Architecture A selected** |
| 715 – 755 MeV | **Architecture B selected**; the value measures the mapping exponent λ |
| < 715 MeV | **both architectures excluded** (exceeds the NA60-compatible pull) |
| broadening-only fit preferred at ≥ 3σ over any free-pole fit | treated as **A-compatible** |

Inputs' independence: the melt fraction f(n) is calibrated on nuclear
saturation; the mapping bound on λ comes from NA60 dileptons; neither is
tuned to HADES data.

## Commitments

1. These windows will not be adjusted after HADES results appear.
2. If the selected architecture contradicts other sectors of the model,
   that contradiction will be reported, not reinterpreted.
3. The losing architecture's scripts remain in the repository as record.
