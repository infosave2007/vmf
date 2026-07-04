# NVG Seven Theorems — version 2 changelog

Corrections established by a systematic audit of the framework (2026-07);
full provenance in the repository history (github.com/infosave2007/vmf).

1. **Vacuum value corrected: W₀ = M_Ω,0 = 859 MeV.** Version 1 stated
   W₀ = 432.2 MeV in three places (the minimum location, the
   defining-assumption paragraph, and the Peccei–Quinn comparison). That
   number is the *bounce temperature* T_b = (30 M_Ω⁴/π²g_*)^{1/4} = 432 MeV,
   mislabeled as the vacuum value; the bounce density ρ_c = W₀⁴/(ħc)³ =
   7.09×10⁴ MeV/fm³ quoted in the same document requires W₀ = 859 MeV.

2. **Bulk-entropy subsection of the appendix rewritten.** Version 1
   reproduced the observed bulk entropy 1.7×10⁸⁸ k_B through a compounded
   per-cycle factor f_S ≈ 2.545 described as "realistic dissipation"; that
   factor was in fact fitted to the observed value and is withdrawn. The
   present-day bulk entropy follows adiabatically from the most recent
   bounce (s·a³ = const) and equals s₀·V_H ≈ 2×10⁸⁸ k_B with no free
   parameter — a consistency statement, not a prediction. The
   radiation-era scaling R ∝ √M used in v1 was also dynamically
   inconsistent with the R ∝ M turnaround relation of the holographic
   subsection; the correct exponents are R ∝ M^{2/3} (radiation) and
   R ∝ M (matter-dominated late cycles, the regime of the doubling chain).

3. **θ-mass formula corrected.** Version 1 quoted m_θ = √χ_top/f_π ≈ 958
   MeV, which does not follow numerically from any standard χ_top. The
   correct statement is the Witten–Veneziano identification of the
   condensate phase with the η′: m_θ = √(6χ_top^quenched)/f_π ≈ 967 MeV
   vs m_η′ = 958 MeV. The physical conclusion of that section (instant
   CP relaxation) is unchanged and strengthened.

The rigor classification of the seven results (Limitations table) is
unchanged. The holographic horizon chain (R×2 ⟺ S_GH×4 per cycle, the
77-cycle count) is unchanged and is now additionally supported by an
explicit turnaround-dynamics derivation in the repository
(verification/nvg_tolman_law_derivation.py).
