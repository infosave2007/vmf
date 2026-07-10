# M_crit as the Chandrasekhar mass scale — derivations, intersections, honest scope

Verification: `nvg_mcrit_chandrasekhar.py` (exact reduction), `nvg_mcrit_family.py` (family + sweep).
Adversarially reviewed; verdict: **real physics, not numerology — but do not oversell** (see §6).

## 1. Closed form + Landau energy-balance

The QCD-anchored Hayward core has inner scale `ℓ = √(3c²/(8πGρ_c)) = 1128 m`, with
`ρ_c = M_Ω⁴/(ℏc)³`. Its extremal (horizon-forming) mass reduces **exactly** to

```
M_crit = (3√3/4)·ℓc²/G  =  C · M_Pl³ / M_Ω²,     C = 9/(8√(2π)) = 0.44881
```

(direct vs closed form agree to 2.8×10⁻¹⁶ — an identity, not a fit). The combination
`M_Pl³/m²` is the **Chandrasekhar/Landau stellar scale**: balancing the relativistic Fermi
energy per particle `~ℏcN^{1/3}/R` against gravity `~GNm²/R` (both ∝1/R) gives
`N_crit ~ (M_Pl/m)³`, `M ~ M_Pl³/m²`. With `m = M_Ω`: bare scale `= 2.21 M⊙`; the GR-extremal
prefactor `C` brings it to **0.99 M⊙**.

## 2. The Chandrasekhar number, N ~ 10⁵⁷

`N_crit = M_crit/M_Ω = C(M_Pl/M_Ω)³ = 1.29×10⁵⁷`. Same decade as: baryons in the Sun
`M⊙/m_N = 1.19×10⁵⁷`, classical `(M_Pl/m_N)³ = 2.20×10⁵⁷`. All are `(M_Pl/hadron)³` — the one
dimensionless gravity/QCD ratio that sets the baryon content of any self-gravitating compact object.

## 3. QCD origin — why M_crit lands in the Chandrasekhar family

Both `M_Ω = 859 MeV` and the nucleon (939 MeV) are hadron masses from the same `Λ_QCD`
(confinement / trace anomaly; `M_Ω/m_N = 0.916`). So `M_crit ∝ M_Ω⁻²` and the astrophysical
`M_Chandra, M_TOV ∝ m_N⁻²` sit in the **same family** — not an independent numerical accident.
Importantly `M_Ω` is an **independent global anchor** (σ-terms, reused across ~25 scripts), **not**
back-solved from 1 M⊙ — the circularity worry is absent.

## 4. Extremal geometry

Horizon cubic `x³−x²+ε²=0` has its double root at `ε_c = 2/(3√3) = 0.3849`, `x = 2/3`. At
`M_crit`: `r_Sch = 2GM_crit/c² = 2930.7 m`, merged horizon `r_h = (2/3)r_Sch = 1953.8 m`, i.e.
**`r_h = √3·ℓ` exactly**. Below M_crit no horizon forms — a horizonless de-Sitter-core remnant.

## 5. Uncertainty + falsifier

`M_crit ∝ M_Ω⁻²`, so `M_Ω = 859±8 MeV → M_crit ∈ [0.974, 1.011] M⊙`. **Kill test:** a confirmed
compact object below `~0.97 M⊙` with a horizon signature (ringdown / `Λ=0`). Current GWTC null:
lightest secondary GW191219 `~1.17 M⊙`, above the band.

## 6. Honest scope (per adversarial review — both claims "partly-overstated")

- **REAL, not numerology:** the reduction is an exact identity; `M_Pl³/m²` is the genuine
  Chandrasekhar scale; the QCD anchor is independent (no reverse-fit to 1 M⊙).
- **Not "dimensional necessity":** pure dimensions permit `M_Pl^a m^{1-a}` for any `a`; the power
  `M_Pl³/m²` is fixed by the **collapse/horizon balance law**, and only the **order of magnitude**
  is forced. A "hadron mass" spans pion→Ω, giving `37.6 → 0.26 M⊙` (2+ orders); the clean ~1 M⊙
  needs `m ~ nucleon-scale` **and** the prefactor `C`.
- **Not distinctive to NVG:** any Hayward/Bardeen regular core with a QCD-scale cutoff gives an
  extremal mass of order M⊙. The ~1 M⊙ value does **not** discriminate NVG; the testable content is
  the narrow band + the horizonless-remnant statement.
- **Different physics from Chandrasekhar/TOV:** `M_crit` is a **minimum** horizon-formation mass;
  those are **maximum** degeneracy masses. Shared skeleton `M_Pl³/m²`, opposite extrema, unrelated
  O(1) coefficients — so the ~1 M⊙ agreement is "same family, factor ~2," not point-for-point.
- **Does not predict the Sun's mass** (the Sun is itself near this scale).

**Bottom line:** a legitimate, exact, de-mystifying result — "critical horizon mass = mass of the
Sun" is gravity+QCD, not mysticism — provided it is stated as *scale*, not as a distinctive
narrow forward prediction.
