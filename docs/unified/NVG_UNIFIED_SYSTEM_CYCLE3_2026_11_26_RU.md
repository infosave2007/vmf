# Единая система NVG — Цикл 3: Первичные ворота при $\zeta_{\text{RMF}} = 0.0165\text{–}0.0185$, релятивистский Хартри–БКС и закрытие реестра (2026-11-26)

> [!IMPORTANT]
> **Статус доказательности (`SPEC-v2`):**
> - Все 8 мостов единой системы и 3 отчёта зарегистрированы в [registry.json](file:///Users/oleg/Documents/NVG-Research/verification/registry.json) и [artifact_manifest.json](file:///Users/oleg/Documents/NVG-Research/verification/artifact_manifest.json) с **`evidence_weight = 0.0`**, `role = "validation"`, `status = "maintained"`.
> - Между секторами материи, конечных ядер, теплового отклика и нейтронных звёзд **не проводилось никакой подгонки параметров (`zero-refit`)**; константа спаривания $C_n = 21.0\text{ МэВ}$ ($G_n = C_n / A$) фиксирована из литературы без подгонки по массам Sn.

---

## 1. Исправление `QLOVE_COEFFS` и целостность репозитория (Шаг 1)

По результату независимого расчёта Хартла 2-го порядка из Моста 7 исправлена опечатка в историческом плоттинг-скрипте [nvg_iloveq_plot.py](file:///Users/oleg/Documents/NVG-Research/verification/nvg_iloveq_plot.py#L50-L56):
- В словаре `QLOVE_COEFFS` (Yagi & Yunes 2017, Phys. Rep. 681, Table 1) коэффициенты приведены к точным значениям `d = -4.283e-3` (было `-4.283e-4`) и `e = 1.245e-4` (было `1.245e-6`).
- Перегенерированы артефакты [fig_iloveq_universal.png](file:///Users/oleg/Documents/NVG-Research/verification/fig_iloveq_universal.png) и [fig_iloveq_universal_report.json](file:///Users/oleg/Documents/NVG-Research/verification/fig_iloveq_universal_report.json) (`Lambda_1.4 = 519.422`, `max |I_radius - I_transform|/I_transform = 5.287%`).
- Все 4 тестовых контракта целостности репозитория ([test_p11_s2_artifact_provenance.py](file:///Users/oleg/Documents/NVG-Research/verification/test_p11_s2_artifact_provenance.py), [test_p4_s2_s8_iloveq_integrity.py](file:///Users/oleg/Documents/NVG-Research/verification/test_p4_s2_s8_iloveq_integrity.py), [test_p8_s3_model_identity.py](file:///Users/oleg/Documents/NVG-Research/verification/test_p8_s3_model_identity.py), [test_p13_s1_final_consistency.py](file:///Users/oleg/Documents/NVG-Research/verification/test_p13_s1_final_consistency.py)) прошли успешно (`17 passed`).

---

## 2. Мост 8 (`nvg_unified_primary_gate_pairing_bridge.py`): Закрытие первичных ворот, RH-BCS и сквозная проверка

Модуль: [nvg_unified_primary_gate_pairing_bridge.py](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_primary_gate_pairing_bridge.py) · [тесты](file:///Users/oleg/Documents/NVG-Research/verification/test_nvg_unified_primary_gate_pairing_bridge.py) (`11 passed`) · [результаты](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_primary_gate_pairing_results.json) (`--validate` OK).

### Stage A: Расширенный скан по $\zeta_{\text{RMF}} \in \{0.0165, 0.0175, 0.0185\}$ — **первичные ворота пройдены**

На предзаданной сетке из `54` точек (`zeta_RMF in (0.0165, 0.0175, 0.0185)`, `n0 in (0.156, 0.160)`, `M*/M_N in (0.60, 0.61, 0.62)`, `L in (50, 56, 60)`) все `54` точки глобально вакуумно устойчивы и каузальны до $10\,n_0$, а **`8` точек** одновременно удовлетворяют **всем первичным воротам** ($M_{\max} \ge 2.08\,M_\odot$, $R_{1.4} \le 13.0\text{ км}$, $\Lambda_{1.4} \le 580$, $M^*/M_N \in [0.59, 0.62]$):

| $\zeta_{\text{RMF}}$ | $n_0$, $\text{фм}^{-3}$ | $M^*/M_N$ | $L$, МэВ | $b_6$, $\text{МэВ}^{-2}$ | $M_{\max}$, $M_\odot$ | $R_{1.4}$, км | $\Lambda_{1.4}$ | $j_{\text{eff}}(0.11)$, МэВ | $V_{\text{SO}}$, МэВ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.0165` (победитель)** | **`0.160`** | **`0.62`** | **`60`** | **`1.7132e-09`** | **`2.0933`** | **`12.9495`** | **`575.29`** | **`11.8086`** | **`640.97`** |
| `0.0175` | `0.160` | `0.61` | `56` | `1.2011e-09` | `2.0915` | `12.8994` | `574.47` | `12.2392` | `658.91` |
| `0.0165` | `0.160` | `0.62` | `56` | `1.7132e-09` | `2.0934` | `12.8819` | `566.31` | `12.2724` | `640.97` |
| `0.0175` | `0.160` | `0.61` | `50` | `1.2011e-09` | `2.0917` | `12.7840` | `562.41` | `13.0407` | `658.91` |
| `0.0165` | `0.160` | `0.62` | `50` | `1.7132e-09` | `2.0937` | `12.7745` | `554.17` | `13.0408` | `640.97` |
| `0.0165` | `0.160` | `0.61` | `50` | `1.4345e-09` | `2.1096` | `12.8148` | `572.42` | `13.0426` | `658.91` |
| `0.0185` | `0.160` | `0.60` | `50` | `1.0057e-09` | `2.0908` | `12.7978` | `573.13` | `13.0500` | `676.82` |
| `0.0175` | `0.156` | `0.62` | `50` | `1.0057e-09` | `2.0887` | `12.8645` | `574.26` | `13.2449` | `641.88` |

По предварительно зафиксированному критерию минимума $|j_{\text{eff}}(0.11) - 11.136016|$ выбрана точка-победитель `zeta_RMF = 0.0165, n0 = 0.160 фм^-3, M*/M_N = 0.62, L = 60 МэВ` (`abs_j_eff_diff = 0.6726 МэВ`, `max c_s^2 = 0.4369`).

---

### Stage B1: Слепой тест 6 закрыто-оболочечных ядер (`zero-refit`) на точке-победителе

Сводные метрики: `rms_B_over_A_err_MeV = 0.2360 МэВ`, `rms_R_ch_rel_err_percent = 1.4412%`, $\Delta R_{\text{ch}}(\text{Ca48}-\text{Ca40}) = 0.0152\text{ фм}$, $\Delta R_{np}(\text{Ca48}) = +0.1903\text{ фм}$, $\Delta R_{np}(\text{Pb208}) = +0.1964\text{ фм}$, `max_field_residual_rel = 1.04e-09`, `max_particle_rel_err = 7.38e-08`, `max_two_energy_diff_MeV = 2.10e-05 МэВ`.

| Ядро | $B/A_{\text{exp}}$ | $B/A$ (расч.) | $R_{\text{ch,exp}}$, фм | $R_{\text{ch}}$ ($\delta\%$), фм | $\Delta R_{np}$, фм | Старшее `SO` ($n / p$), МэВ |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **O16** | `-7.9762` | `-7.8461` | `2.6991` | `2.6912 (-0.29%)` | `-0.0304` | `1p`: `5.916 / 5.843` |
| **Ca40** | `-8.5513` | `-8.3256` | `3.4776` | `3.4104 (-1.93%)` | `-0.0541` | `1d`: `6.137 / 6.064` |
| **Ca48** | `-8.6667` | `-8.3816` | `3.4771` | `3.4256 (-1.48%)` | `+0.1903` | `1f`: `6.945 / 7.285` |
| **Zr90** | `-8.7100` | `-8.4147` | `4.2694` | `4.2006 (-1.61%)` | `+0.0819` | `1g`: `6.801 / 6.916` |
| **Sn132** | `-8.3536` | `-8.1406` | `4.7093` | `4.6411 (-1.45%)` | `+0.2613` | `1g`: `5.465 / 5.723` |
| **Pb208** | `-7.8675` | `-7.6383` | `5.5012` | `5.4283 (-1.33%)` | `+0.1964` | `1g`: `3.638 / 3.931` |

---

### Stage B2: Самосогласованный релятивистский Хартри–БКС (RH-BCS) для цепочки Sn (`C_n = 21.0 МэВ`, $N > 28$)

| Изотоп | $\Delta_n$, МэВ | $\lambda_n$, МэВ | Порог $(\frac{G_n}{2}\sum\frac{\Omega_k}{\|\epsilon_k-\lambda_0\|})$ | $E_{\text{pair}}$, МэВ | Выигрыш $\Delta E_{\text{BCS}-\text{unp}}$, МэВ | $B/A_{\text{BCS}}$ (unp / exp) | $R_{\text{ch}}$ (exp), фм | Разность $E_{\text{eig}}-E_{\text{direct}}$ | Невязка полей |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sn116** | `0.7454` | `-8.6572` | `open-shell` | `-3.0693` | `-0.8926` | `-8.2365` (`-8.2288` / `-8.5231`) | `4.5427` (`4.6250`) | `5.15e-04` | `5.88e-11` |
| **Sn120** | `0.6326` | `-7.8301` | `1.1922` | `-2.2869` | `-0.1366` | `-8.2370` (`-8.2359` / `-8.5046`) | `4.5707` (`4.6519`) | `5.13e-04` | `1.25e-10` |
| **Sn124** | `0.9644` | `-7.2828` | `open-shell` | `-5.4916` | `-2.7482` | `-8.2130` (`-8.1909` / `-8.4674`) | `4.5945` (`4.6735`) | `5.30e-04` | `2.95e-10` |
| **Sn132** | **`0.0000`** | `-4.6834` | **`0.6587 <= 1`** | `0.0000` | `0.0000` | `-8.1406` (`-8.1406` / `-8.3536`) | `4.6411` (`4.7093`) | `5.03e-04` | `3.05e-11` |

- **Дважды магический `Sn132`:** из-за большой оболочечной щели $N=82$ критерий нормального состояния равен `0.6587 <= 1`, поэтому БКС-решение **естественно коллапсирует к $\Delta_n = 0$** без каких-либо искусственных отключений.
- **Открыто-оболочечные `Sn116, Sn120, Sn124`:** дают физические щели $\Delta_n \in [0.63, 0.96]\text{ МэВ}$ (`Sn120` попадает на нейтронную подоболочку $N=70$ между `3s1/2` и `1h11/2`, где перестройка самосогласованного поля оставляет чистый выигрыш `-0.1366 МэВ`, а в `Sn124` при заполнении `1h11/2` выигрыш возрастает до `-2.7482 МэВ`), точность БКС-уравнений `max_bcs_eq_residual = 1.85e-13`, сохранение числа частиц `<= 1.81e-08`.

---

### Stage C: Сквозная проверка точки-победителя на Мостах 5 и 7

1. **Мост 5 (тепловой Schur-гессиан и Уорд при $T = 70\text{ МэВ}, \mu_B = 875\text{ МэВ}, \delta = 0$):**
   - $n_B = 0.1897\text{ фм}^{-3}$, $K_T = 703.38\text{ МэВ}$, $K_\sigma = 1122.87\text{ МэВ}$, $c_{s,\text{ad}}^2 = 0.1923$;
   - `schur_vs_fd_rel_err = 5.65e-13`, `ward_max_rel_err = 4.44e-15` (отрицательный контроль `ph-only`: `1.4449`), `direct_5x5_vs_schur_rel_err = 2.45e-16`.
2. **Мост 7 (прямой ОДУ Хартла 1-го + 2-го порядка на EOS победителя):**
   - **`PSR J0737-3039A` ($1.3381\,M_\odot$):** $R = 12.9094\text{ км}$, $I_{45} = 1.5257$, $\bar{I} = 14.6889$, $\bar{Q} = 6.7394$, $\Lambda = 757.54$, отклонения от Yagi–Yunes (2017) `(0.07%, 0.11%, 0.17%)`.
   - **`canonical_1_4` ($1.4000\,M_\odot$):** $R = 12.9066\text{ км}$, $I_{45} = 1.6248$, $\bar{I} = 13.6580$, $\bar{Q} = 6.2017$, $\Lambda = 579.29$, отклонения от Yagi–Yunes (2017) `(0.07%, 0.08%, 0.21%)`.

---

## 3. Команды полного воспроизведения всей 8-мостовой системы

Объединённый прогон 12 тестовых наборов (`8` мостов + `4` контракта целостности): **`130 passed, 186 warnings in 76.27s`**:

```zsh
PYTHONPATH=/Users/oleg/Documents/NVG-Research/verification python3 -B -m pytest -q -p no:cacheprovider \
  verification/test_nvg_quartic_vector_melting_audit.py \
  verification/test_nvg_two_condensate_unified_audit.py \
  verification/test_nvg_unified_dirac_hartree_probe.py \
  verification/test_nvg_unified_gravity_bridge_audit.py \
  verification/test_nvg_unified_thermal_response_bridge.py \
  verification/test_nvg_unified_calibration_envelope_probe.py \
  verification/test_nvg_unified_astrophysics_stability_bridge.py \
  verification/test_nvg_unified_primary_gate_pairing_bridge.py \
  verification/test_p11_s2_artifact_provenance.py \
  verification/test_p4_s2_s8_iloveq_integrity.py \
  verification/test_p8_s3_model_identity.py \
  verification/test_p13_s1_final_consistency.py
```

Валидация `--validate` для всех 8 мостов и проверка реестра `verification/registry.py`:

```zsh
for pair in \
  "verification/nvg_quartic_vector_melting_audit.py verification/nvg_quartic_vector_melting_results.json" \
  "verification/nvg_two_condensate_unified_audit.py verification/nvg_two_condensate_unified_results.json" \
  "verification/nvg_unified_dirac_hartree_probe.py verification/nvg_unified_dirac_hartree_results.json" \
  "verification/nvg_unified_gravity_bridge_audit.py verification/nvg_unified_gravity_bridge_results.json" \
  "verification/nvg_unified_thermal_response_bridge.py verification/nvg_unified_thermal_response_results.json" \
  "verification/nvg_unified_calibration_envelope_probe.py verification/nvg_unified_calibration_envelope_results.json" \
  "verification/nvg_unified_astrophysics_stability_bridge.py verification/nvg_unified_astrophysics_stability_results.json" \
  "verification/nvg_unified_primary_gate_pairing_bridge.py verification/nvg_unified_primary_gate_pairing_results.json"; do
  set -- ${=pair}
  python3 -B "$1" --validate "$2" || exit 1
done
python3 -B verification/registry.py
```
