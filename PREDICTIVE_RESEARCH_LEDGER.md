# Публичный реестр прогнозного исследования NVG

_Public predictive-research ledger · Phase 4 S2 · снимок 2026-08-28 (Europe/Istanbul)_

## Обязательная граница свидетельств / mandatory evidence boundary

**Ни один результат ниже не является независимым подтверждением полной теории NVG.**
`derived_conditional`, `sensitivity_only`, `null`, `prospective` и `BLOCKED_*` —
это точные семантические статусы, а не общий счёт подтверждений. Условные и
чувствительные строки имеют нулевой независимый вес; отсутствие ошибки процесса,
зелёный тест или наличие файла не являются наблюдательным свидетельством. Глобальный
p-value и независимая held-out-проверка не заявляются.

Основание реестра — финальный P3-S8 gate (final control), с безопасными
границами P1-S10 и P2-S9 (gate labels retained in the machine ledger). Публичные
числа ниже сверяются с живыми JSON, а не с устаревшими статьями или
`NVG_FINAL_REPORT.md`.

`derived_conditional` = вычисление внутри явно заданной модели/ветви;
`sensitivity_only` = лестница, диапазон или скан чувствительности;
`null/control` = проверенное тождество или нулевой контроль;
`prospective` = будущий тест, ещё без наблюдаемого результата;
`BLOCKED_*` = необходимый производитель, likelihood или физика отсутствуют.

## 1. Выведенные условные прогнозы / derived conditional forecasts

| Результат | Поддерживаемое отображение (raw values остаются в JSON) | Статус и граница | Как опровергнуть или разблокировать | Трассировка |
|---|---|---|---|---|
| **J0737A, точная Hartle–Hinderer цель** (`M_A=1.3381±0.0007 M☉`) | Ветвь `1`, `gap_crossed=false`, freeze-before-observation. Центральные raw: `R=12.466899931727045 km`, `Λ=682.3319386687238`, `I=1.4856539538408488e45 g cm²`, `Ī=14.303017715186515`, `C=0.15848675058116762`. Поддержанное outward-отображение: `R_A=[12.466,12.468] km` (3 digits), `Λ_A=[680,686]` (0 digits), `I_A=[1.484e45,1.487e45] g cm²` (3 digits), центрально `R=12.467`, `Λ=682`, `I=1.486e45`, `Ī=14.3`. | Центральные наблюдаемые `derived_conditional`; диапазоны `sensitivity_only`, model-conditional, не confidence intervals. Q и inverse-mass — `blocked`; широкая Phase-1 envelope — `BLOCKED_NO_HARTLE_REVALIDATION`. | Будущая согласованная измерительная likelihood с компонентой вне полного envelope falsifies именно эту конструкцию. Для более широкой envelope нужны same-background Hartle revalidation, rotation/systematics и held-out observation; альтернативная ветвь не разрешена этим результатом. | [producer](verification/nvg_ns_frozen_forecasts.py) · [result](verification/nvg_ns_frozen_forecasts_p3s1_results.json) · [test](verification/test_p3_ns_frozen_forecasts.py) · P3-S5/P3-S8 final gate labels |

## 2. Проверенные численные и null-контроли / validated numerical and null controls

| Контроль | Поддерживаемое значение | Интерпретация | Какой новый результат нужен | Трассировка |
|---|---|---|---|---|
| **Barotropic neutral-buoyancy null** | `max_abs_N2=3.283728511497663e-18 km^-2`; `max_abs_discriminant=2.220446049250313e-16`; grids `61/121/241/481`, all `<1e-14`. | `DERIVED_NULL_CONTROL_BAROTROPIC_NEUTRAL_BUOYANCY`; математический/reference control, не физическая частота. | Сourced `Γ₁=(∂P/∂ε)_{s,Y_i}`, composition gradients and mode likelihood are required before a physical g-mode claim. | [producer](verification/nvg_cooling_gmode_dependency_audit.py) · [result](verification/nvg_cooling_gmode_dependency_audit_p3s2_results.json) · [test](verification/test_p3_cooling_gmode_dependency_audit.py) · P3-S6/P3-S8 final gate labels |
| **Canonical NS regression** | `M_max=2.047950740578197 M☉`, `R_1.4=12.550001000000044 km`, `Λ_1.4=519.4223807918132`; exact equality to frozen regression. | Process/numerical `PASS`, but chain selection is `CONDITIONAL_IN_SAMPLE` (J0740/GW170817/NICER) with zero independent weight. | A preregistered held-out NS likelihood and an independent EOS/branch construction are needed for evidence. | [producer](verification/nvg_ns_predictive_audit.py) · [result](verification/nvg_ns_predictive_audit_results.json) · [test](verification/test_p4_s1_ns_integrity.py) · P1-S10/P2-S9 final gate labels |
| **I–Love–Q overlay** | 12 sequence rows; `max_relative_deviation=0.05287175963086869`; `tolerance_gate=null`; `independent_i_solve=false`, `independent_q_solve=false`. | `TRANSFORM_ONLY_NO_INDEPENDENT_IQ_SOLVE`: Yagi–Yunes transform/descriptive overlay only; no universality test or calibration. | Independent quadrupole and moment-of-inertia equations plus an external likelihood would be required. | [producer](verification/nvg_iloveq_plot.py) · [report JSON](verification/fig_iloveq_universal_report.json) · [test](verification/test_p4_s2_s8_iloveq_integrity.py) · P2-S9 final gate label |

## 3. Ограниченные no-go и sensitivity findings / scoped findings

| Канал | Live result и exact status | Что не следует заключать | Falsifier / unblock requirement | Трассировка |
|---|---|---|---|---|
| **S8, maintained scalar slice** | При `Ω_m=0.315`: one-sigma overlap `0`; `S8_NVG=0.8512718316553592`, `S8_Planck=0.832`, lensing input `0.776`; tension `3.29411764705882σ → 4.427754803256424σ`, shift `+2.316325920115303%`, status `WORSENS`. | Это условный scalar result, не observational/global no-go. DESI overlay — `unknown_provenance_sensitivity_overlay`, `observed_likelihood=false`; global status `BLOCKED_NO_SOURCED_JOINT_LIKELIHOOD`. Critical boundary withdrawn: `NOT_IDENTIFIABLE_GRID_AND_NORMALIZATION_DEPENDENT`; crossings `.300,.300,.305` (25/51/101), current normalization counts `[45,2,0,0,0,0]`, alternate/no-Ω all zero. | Независимые S8/lensing и joint `(w0,wa,Ωm)` likelihoods, preregistered domain and no legacy overlay. Such a result could falsify the scoped no-go or its global interpretation. | [producer](verification/nvg_prospective_falsification_reach.py) · [result](verification/nvg_prospective_falsification_reach_p3s3_results.json) + [map](verification/nvg_s8_no_go_map.json) · [test](verification/test_p3_prospective_falsification_reach.py) · P3-S7/P3-S8 final gate labels |
| **PBH/JWST seed-band deficit** | Minimum required rate product `299558.79267302185` (all declared seed-band rows outside `[0,1]`); first expanded JWST-upper-density reachable rung `17`; rung `18` is the all-upper-boundary envelope. PTA amplitude `2.4e-15` is an internal benchmark only. | Algebraic/calibrated sensitivity; no external CMB exclusion and no global PBH–PTA claim. Rungs `17/18` are boundary outcomes, not a solution. | Independently sourced JWST occupation/abundance and PBH binary-rate likelihood reaching `[0,1]` for rungs 9–11 without target-tuned abundance, with DM-budget accounting. | [producer](verification/nvg_pbh_nanograv_audit.py) · [tracked result](verification/P1-S3-pbh-nanograv-audit.json) · [reach ledger](verification/nvg_prospective_falsification_reach_p3s3_results.json) · [test](verification/test_p1_s3_pbh_nanograv_audit.py) · P3-S7/P3-S8 final gate labels |

## 4. Перспективные projections / prospective sensitivity

| Канал | Поддерживаемое отображение | Статус | Минимальный будущий тест | Трассировка |
|---|---|---|---|---|
| **Echo synthetic-null reach** | 391-row public catalog → 259 selected events; live q matrix shape `(259,15)`. Raw A90 union `[0.03319408380348728,0.03356685238591077]`; supported display `0.03` (2 decimals), outward `[0.03,0.04]`. | `sensitivity` / `SENSITIVITY_ONLY_CONDITIONAL_SYNTHETIC_NULL`; never an observed echo upper limit. q payload is `PASS_RECOMPUTED_LIVE_FULL_MATRIX` with digest `80e20bd2713a9e777d7c2111509a7a82a54ba7367642533789e42a17edab05a0`. | Keyed event-level strain, posterior samples, detector PSDs and empirical time-slide bank, then the same likelihood with an independently preregistered statistic. Current blocks: `BLOCKED_MISSING_STRAIN_POSTERIOR_NOISE_PRODUCTS` and `BLOCKED_NO_EMPIRICAL_TIME_SLIDE_BANK`. | [producer](verification/nvg_prospective_falsification_reach.py) · [result](verification/nvg_prospective_falsification_reach_p3s3_results.json) · [source result](verification/nvg_echo_hierarchical_upper_limit_p2s3_results.json) · [test](verification/test_p3_prospective_falsification_reach.py) · P3-S8 final gate label |

## 5. Точные blockers / precise blockers

| Claim surface | Exact blocker | Required producer/data before promotion |
|---|---|---|
| Physical g-mode frequency | `BLOCKED_MISSING_GAMMA1_COMPOSITION_DERIVATIVES_AND_MODE_LIKELIHOOD` | Sourced `Y_i(n_b,T)`, `dY_i/dr`, `(∂P/∂Y_i)_{n,s}`, thermodynamic `Γ₁`, stellar profile and mode likelihood. The frozen `Γ₁` offset is `SENSITIVITY_ONLY_FROZEN_GAMMA1_NOT_MICROPHYSICS` (max reference `2585.7787899840355 Hz`), not a frequency prediction. |
| Cooling curves / Cas A | `BLOCKED_MISSING_COMPOSITION_HEAT_CAPACITY_EMISSIVITY_PAIRING_CONDUCTIVITY_ENVELOPE`; Cas A `BLOCKED_NO_THERMAL_SOLVER_OR_SPECTRAL_LIKELIHOOD` | Complete composition, heat capacity, neutrino matrix elements, temperature dependence, pairing/suppression, conductivity, stellar profile, envelope relation and a sourced thermal/spectral likelihood. |
| Hyperon/Urca cooling and stellar mapping | `BLOCKED_MISSING_INTERACTIONS_WEAK_MATRIX_ELEMENTS_AND_STABLE_TOV_BRANCH`; P2 reference is `DERIVED_ZERO_INTERACTION_FREE_GAS_REFERENCE_ONLY`, stable mapping `BLOCKED_NO_STABLE_TOV_BRANCH`. | Traceable physical couplings, coexistence/phase construction, weak rates, temperature/superfluid inputs and a stable TOV branch. |
| Echo observational claim | `BLOCKED_MISSING_STRAIN_POSTERIOR_NOISE_PRODUCTS`; empirical slides `BLOCKED_NO_EMPIRICAL_TIME_SLIDE_BANK`. | Real keyed strain/posterior/PSD products and retained detector time-slide bank. |
| S8 global statement | `BLOCKED_NO_SOURCED_JOINT_LIKELIHOOD`. | Independent lensing/S8 and joint CPL `(w0,wa,Ωm)` likelihood; evaluate without the unknown-provenance DESI overlay. |
| Q, inverse-mass, wider EOS envelope | `blocked` / `BLOCKED_NO_HARTLE_REVALIDATION`. | Second-order Hartle quadrupole solve, direct inertia likelihood, and same-background revalidation of candidate rows; no branch or rotation upgrade is implied. |

## Машиночитаемая карта / machine-readable mapping

The following JSON is intentionally small and mirrors the rows above. Paths are
repository-relative; values are strings where display precision or status text
must not be silently rounded. P4-S3 can compare these rows with the live
artifacts and final gates.

```json
[
  {
    "id": "ns_j0737a",
    "category": "derived_conditional_forecast",
    "status": "derived_conditional",
    "display": "R_A=[12.466,12.468] km; Lambda_A=[680,686]; I_A=[1.484e45,1.487e45] g cm^2; Ibar=14.3",
    "producer": "verification/nvg_ns_frozen_forecasts.py",
    "result": "verification/nvg_ns_frozen_forecasts_p3s1_results.json",
    "test": "verification/test_p3_ns_frozen_forecasts.py",
    "gate": "P3-S8-FINAL",
    "falsifier_or_unblock": "future I_A or joint I-Lambda-R outside the complete conditional envelope; wider envelope requires Hartle revalidation"
  },
  {
    "id": "cooling_barotropic_null",
    "category": "validated_null_control",
    "status": "DERIVED_NULL_CONTROL_BAROTROPIC_NEUTRAL_BUOYANCY",
    "display": "max_abs_N2=3.283728511497663e-18 km^-2; max_abs_discriminant=2.220446049250313e-16",
    "producer": "verification/nvg_cooling_gmode_dependency_audit.py",
    "result": "verification/nvg_cooling_gmode_dependency_audit_p3s2_results.json",
    "test": "verification/test_p3_cooling_gmode_dependency_audit.py",
    "gate": "P3-S8-FINAL",
    "falsifier_or_unblock": "sourced Gamma1/composition derivatives and mode likelihood"
  },
  {
    "id": "iloveq_transform_only",
    "category": "validated_transform_control",
    "status": "TRANSFORM_ONLY_NO_INDEPENDENT_IQ_SOLVE",
    "display": "sequence_count=12; max_relative_deviation=0.05287175963086869; tolerance_gate=null",
    "producer": "verification/nvg_iloveq_plot.py",
    "result": "verification/fig_iloveq_universal_report.json",
    "test": "verification/test_p4_s2_s8_iloveq_integrity.py",
    "gate": "P2-S9-FINAL",
    "falsifier_or_unblock": "independent I and Q solves plus external likelihood"
  },
  {
    "id": "s8_omega_m_0315",
    "category": "scoped_sensitivity",
    "status": "WORSENS_SCOPED_CONDITIONAL; BLOCKED_NO_SOURCED_JOINT_LIKELIHOOD",
    "display": "Omega_m=0.315; S8_NVG=0.8512718316553592; overlap_1sigma=0; tension=4.427754803256424 sigma",
    "producer": "verification/nvg_prospective_falsification_reach.py; verification/nvg_s8_no_go.py",
    "result": "verification/nvg_prospective_falsification_reach_p3s3_results.json; verification/nvg_s8_no_go_map.json",
    "test": "verification/test_p3_prospective_falsification_reach.py; verification/test_p1_s2_s8_no_go.py",
    "gate": "P3-S8-FINAL",
    "falsifier_or_unblock": "sourced joint (w0,wa,Omega_m) likelihood without legacy overlay"
  },
  {
    "id": "echo_synthetic_a90",
    "category": "prospective_sensitivity",
    "status": "SENSITIVITY_ONLY_CONDITIONAL_SYNTHETIC_NULL",
    "display": "selected=259/391; q_shape=(259,15); raw_A90=[0.03319408380348728,0.03356685238591077]; display=[0.03,0.04]",
    "producer": "verification/nvg_prospective_falsification_reach.py",
    "result": "verification/nvg_prospective_falsification_reach_p3s3_results.json",
    "test": "verification/test_p3_prospective_falsification_reach.py",
    "gate": "P3-S8-FINAL",
    "falsifier_or_unblock": "keyed strain/posterior/PSD and empirical time-slide likelihood"
  },
  {
    "id": "pbh_seed_band",
    "category": "scoped_sensitivity",
    "status": "SENSITIVITY_ONLY_RUNG_BOUNDARY_SCAN",
    "display": "minimum_required_rate_product=299558.79267302185; first_reachable_rung=17; expanded_boundary_cycle=18",
    "producer": "verification/nvg_pbh_nanograv_audit.py",
    "result": "verification/P1-S3-pbh-nanograv-audit.json; verification/nvg_prospective_falsification_reach_p3s3_results.json",
    "test": "verification/test_p1_s3_pbh_nanograv_audit.py",
    "gate": "P3-S8-FINAL",
    "falsifier_or_unblock": "sourced JWST occupation and PBH binary-rate likelihood in the declared rate box"
  },
  {
    "id": "physical_cooling_gmode",
    "category": "precise_blocker",
    "status": "BLOCKED_MISSING_GAMMA1_COMPOSITION_DERIVATIVES_AND_MODE_LIKELIHOOD",
    "display": "no physical frequency or cooling confirmation emitted",
    "producer": "verification/nvg_cooling_gmode_dependency_audit.py",
    "result": "verification/nvg_cooling_gmode_dependency_audit_p3s2_results.json",
    "test": "verification/test_p3_cooling_gmode_dependency_audit.py",
    "gate": "P3-S8-FINAL",
    "falsifier_or_unblock": "complete named microphysical producers and independent likelihood"
  }
]
```

Полный машинный ledger, provenance и digests публикуются отдельно
`verification/predictive_research_ledger.json` (generated by P4-S1); эта страница
не подменяет его и не редактирует `NVG_FINAL_REPORT.md`.
