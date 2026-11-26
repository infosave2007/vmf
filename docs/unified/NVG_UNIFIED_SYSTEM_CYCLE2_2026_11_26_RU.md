# Единая система NVG — Цикл 2: Тепловой отклик, калибровочный конверт, I-Love-Q и двухконденсатный отскок (2026-11-26)

> [!IMPORTANT]
> **Статус доказательности (`SPEC-v2`):**
> - Все 7 мостов единой системы (Цикл 1: мосты 1–4; Цикл 2: мосты 5–7) прошли объединённый прогон: **`102 passed, 14 warnings in 51.24s`**, все 7 проверок `--validate` вернули `{"valid": true, "errors": []}` (`ALL_7_VALIDATIONS_OK`).
> - В отчёте соблюдена строгая честность `SPEC-v2`: **`evidence_weight = 0.0`** до включения новых модулей в `verification/registry.json` и `artifact_manifest.json`; при построении мостов 5–7 **не выполнялось никакой подгонки параметров между секторами (zero-refit)**; ни один ранее отслеживаемый (tracked) файл репозитория и ни один файл Цикла 1 не был изменён.

---

## 1. Общая архитектура объединённой 7-мостовой системы

Во втором цикле оркестрации на замороженной базовой ветви материи Цикла 1 (Закон 2: двухконденсатный изоскалярный + изовекторный сектор с квартовым векторным плавлением из Закона 1) последовательно надстроены три новых независимых моста — от флуктуаций в плотной/горячей ядерной материи и слепого сканирования конечных ядер до независимой релятивистской астрофизики вращающихся нейтронных звёзд и космологического отскока:

| Цикл | Мост / Модуль | Ключевая физическая связь | Скрипт / Тест / JSON |
| :--- | :--- | :--- | :--- |
| Цикл 1 | Закон 1 (квартовое плавление $V_0$) | Неразрывный $\omega_0$-конденсат и каузальный EoS | [nvg_quartic_vector_melting_audit.py](file:///Users/oleg/Documents/NVG-Research/verification/nvg_quartic_vector_melting_audit.py) · [test](file:///Users/oleg/Documents/NVG-Research/verification/test_nvg_quartic_vector_melting_audit.py) · [JSON](file:///Users/oleg/Documents/NVG-Research/verification/nvg_quartic_vector_melting_results.json) |
| Цикл 1 | Закон 2 (единый двухконденсатный EoS) | Замороженная материя ($\sigma$-квантовый предел + $b_6$ + $\Lambda_v \omega^2 \rho^2$) | [nvg_two_condensate_unified_audit.py](file:///Users/oleg/Documents/NVG-Research/verification/nvg_two_condensate_unified_audit.py) · [test](file:///Users/oleg/Documents/NVG-Research/verification/test_nvg_two_condensate_unified_audit.py) · [JSON](file:///Users/oleg/Documents/NVG-Research/verification/nvg_two_condensate_unified_results.json) |
| Цикл 1 | Закон 3A (ядра Дирака–Хартри zero-refit) | Конечные сферические ядра напрямую из замороженных констант Закона 2 | [nvg_unified_dirac_hartree_probe.py](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_dirac_hartree_probe.py) · [test](file:///Users/oleg/Documents/NVG-Research/verification/test_nvg_unified_dirac_hartree_probe.py) · [JSON](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_dirac_hartree_results.json) |
| Цикл 1 | Закон 3B / 4 (TOV, $I$-Love-$Q$ через сдвиг и отскок) | Ковариантная гравитация, ИК-отворот и космологический отскок | [nvg_unified_gravity_bridge_audit.py](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_gravity_bridge_audit.py) · [test](file:///Users/oleg/Documents/NVG-Research/verification/test_nvg_unified_gravity_bridge_audit.py) · [JSON](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_gravity_bridge_results.json) |
| **Цикл 2** | **Мост 5 (тепловой мастер-отклик и Schur-гессиан)** | Ковариантный 5×5 RPA-отклик, тождества Уорда, изовекторный нуль-звук, флуктуации $R_2$ | [nvg_unified_thermal_response_bridge.py](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_thermal_response_bridge.py) · [test](file:///Users/oleg/Documents/NVG-Research/verification/test_nvg_unified_thermal_response_bridge.py) · [JSON](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_thermal_response_results.json) |
| **Цикл 2** | **Мост 6 (калибровочный конверт и слепой тест 6 ядер)** | Предзаданный скан насыщения (Stage A) и слепой zero-refit Дирак–Хартри (Stage B) | [nvg_unified_calibration_envelope_probe.py](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_calibration_envelope_probe.py) · [test](file:///Users/oleg/Documents/NVG-Research/verification/test_nvg_unified_calibration_envelope_probe.py) · [JSON](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_calibration_envelope_results.json) |
| **Цикл 2** | **Мост 7 (независимый Хартл $I$-Love-$Q$ и 6-переменный отскок)** | Прямой ОДУ 1-го и 2-го порядка Хартла + синхронизованный двухконденсатный отскок | [nvg_unified_astrophysics_stability_bridge.py](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_astrophysics_stability_bridge.py) · [test](file:///Users/oleg/Documents/NVG-Research/verification/test_nvg_unified_astrophysics_stability_bridge.py) · [JSON](file:///Users/oleg/Documents/NVG-Research/verification/nvg_unified_astrophysics_stability_results.json) |

Подробный отчёт по Циклу 1 сохранён в [NVG_UNIFIED_SYSTEM_BRIDGE_2026_11_26_RU.md](file:///Users/oleg/Documents/NVG-Research/NVG_UNIFIED_SYSTEM_BRIDGE_2026_11_26_RU.md).

---

## 2. Мост 5: Единый тепловой мастер-отклик и Schur-гессиан

Мост 5 (`14/14` тестов `pytest`, `--validate` OK) строит тепловой и плотностный матричный отклик непосредственно на замороженных константах Закона 2 (`nvg_two_condensate_unified_results.json`), связывая 5 коллективных/термодинамических мод через редукцию Шура и проверяя калибровочную инвариантность (тождества Уорда).

### Ключевые точки при $T = 70\text{ МэВ}$

| $\mu_B$, МэВ | $\delta$ | $n_B$, $\text{фм}^{-3}$ | $K_T$, МэВ | $K_\sigma$, МэВ | $c_{s,\text{ad}}^2$ | $R_2$ | Ошибка Шура |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `775.0` | `0.0` | `0.07048` | `1050.78` | `1822.49` | `0.11681` | `0.9451` | `2.99e-12` |
| `775.0` | `0.2` | `0.07196` | `1041.84` | `1800.96` | `0.11778` | `0.9352` | `2.98e-12` |
| `875.0` | `0.0` | `0.19849` | `678.90` | `1063.12` | `0.19073` | `0.5195` | `2.73e-12` |
| `875.0` | `0.2` | `0.20125` | `681.44` | `1060.33` | `0.19256` | `0.5108` | `5.80e-12` |

### Верифицированные тождества и физические пороги

- **Тождество прямого 5×5 и Schur-редуцированного гессиана:** `max_direct_5x5_vs_schur_rel_err = 4.8622e-16` (машинная точность); ошибка Schur-редукции в точках $T=70\text{ МэВ}$ не превышает `5.80e-12`.
- **Адиабатическая траектория:** `max_adiabatic_path_rel_err = 1.4295e-08`.
- **Тождество Уорда vs отрицательный контроль (только particle-hole):**
  - полный согласованный отклик: `max_ward_rel_err = 1.2217e-09`;
  - обрезанный (только ph) несогласованный отклик: `min_ph_only_negative_control_ward_rel_err = 0.7831`.
- **Низко-q термодинамический предел:** `max_low_q_thermo_recovery_rel_err = 8.3346e-07`.
- **Изовекторный нулевой звук:** $u_{\text{IV}} / c = 0.401364$.
- **Сверхкритический жидкостно-газовый пик:** `max_supercritical_r2 = 10.0715`, что заметно превышает симметричные значения на изотерме $T = 70\text{ МэВ}$ (`[0.9451, 0.5195]`) и экспериментальный порог HADES (`3.8264`).

---

## 3. Мост 6: Предзаданный калибровочный конверт (Stage A) и слепой тест 6 ядер Дирака–Хартри (Stage B)

Мост 6 (`12/12` тестов `pytest`, `--validate` OK) проверяет жёсткий перекрёстный зазор между свойствами насыщения материи, нейтронными звёздами и конечными магическими ядрами.

### Stage A: Диагностика зазора на сечении $\zeta_{\text{RMF}} = 0.020$

При сканировании по предварительно зарегистрированной сетке при фиксированном `zeta_RMF = 0.020` установлено честное отсутствие одновременного решения первичных ворот (`M*/M_N in [0.59, 0.62]`, `M_max >= 2.08 M_sun`, `Lambda_1.4 <= 580`):
- `primary_gate_satisfied = False`, режим: `PARETO_OPTIMAL_DIAGNOSTIC_CANDIDATE_PRIMARY_GATE_EMPTY`.
- Отобран Парето-оптимальный диагностический кандидат:
  - `n0 = 0.16 фм^-3`, `M*/M_N = 0.61`, `L = 62.0 МэВ`, `b6 = 7.05095e-10`;
  - $M_{\max} = 2.0503\,M_\odot$, $R_{1.4} = 12.9325\text{ км}$, $\Lambda_{1.4} = 564.20$, $j_{\text{eff}}(0.11) = 11.5175\text{ МэВ}$, $V_{\text{SO}} = 658.91\text{ МэВ}$.

### Stage B: Слепое сравнение 6 закрыто-оболочечных ядер (zero-refit Дирак–Хартри)

Глобальные характеристики по всем 6 ядрам (`O16`, `Ca40`, `Ca48`, `Zr90`, `Sn132`, `Pb208`):

| Метрика | `cycle1_baseline` | `stage_a_candidate` | Физический эффект перехода |
| :--- | :---: | :---: | :--- |
| `rms_B_over_A_err_MeV` | `0.2068` | `0.3013` | Небольшой рост RMS ошибки по $B/A$ (`+0.094 МэВ`) |
| `rms_R_ch_rel_err_percent` | `1.5976%` | `1.4786%` | Снижение RMS относительной ошибки по $R_{\text{ch}}$ |
| $\Delta R_{\text{ch}}(\text{Ca48}-\text{Ca40})$, фм | `0.0268` | `0.0116` | Сокращение разности зарядовых радиусов в 2.3 раза |
| $\Delta R_{np}(\text{Ca48})$, фм | `0.1899` | `0.1909` | Устойчивая положительная нейтронная шуба |
| $\Delta R_{np}(\text{Pb208})$, фм | `0.1903` | `0.1993` | Устойчивая положительная нейтронная шуба |
| `all_skins_heavy_positive` | `True` | `True` | Выполнено для всех тяжёлых ядер ($A \ge 48$) |
| `max_field_residual_rel` | `7.43e-09` | `2.82e-09` | Невязка уравнений полей $\lesssim 7.4 \times 10^{-9}$ |
| `max_particle_rel_err` | `1.04e-07` | `6.21e-08` | Сохранение числа частиц $N, Z$ лучше $1.1 \times 10^{-7}$ |
| `max_two_energy_diff_MeV` | `2.68e-05` | `1.84e-05` | Согласованность двух выражений энергии $\lesssim 2.7 \times 10^{-5}\text{ МэВ}$ |

Подробная поядерная таблица (спин-орбитальные расщепления `SO` указаны для старшей замкнутой оболочки $(n / p)$ в МэВ):

| Ядро | $B/A_{\text{exp}}$ | `baseline` $B/A$ | `cand` $B/A$ | $R_{\text{ch,exp}}$ | `baseline` $R_{\text{ch}}$ ($\delta\%$) | `cand` $R_{\text{ch}}$ ($\delta\%$) | Шуба (`base` / `cand`), фм | `SO` (`base` $n/p$ $\to$ `cand` $n/p$), МэВ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **O16** | `-7.9762` | `-7.9711` | `-7.7216` | `2.6991` | `2.6768 (-0.83%)` | `2.6926 (-0.24%)` | `-0.0295` / `-0.0308` | `1p`: `5.233 / 5.176` $\to$ `6.211 / 6.132` |
| **Ca40** | `-8.5513` | `-8.4124` | `-8.2348` | `3.4776` | `3.3965 (-2.33%)` | `3.4112 (-1.91%)` | `-0.0528` / `-0.0546` | `1d`: `5.413 / 5.360` $\to$ `6.420 / 6.342` |
| **Ca48** | `-8.6667` | `-8.4118` | `-8.3159` | `3.4771` | `3.4233 (-1.55%)` | `3.4229 (-1.56%)` | `+0.1899` / `+0.1909` | `1f`: `6.145 / 6.458` $\to$ `7.255 / 7.603` |
| **Zr90** | `-8.7100` | `-8.4435` | `-8.3582` | `4.2694` | `4.1952 (-1.74%)` | `4.1989 (-1.65%)` | `+0.0820` / `+0.0821` | `1g`: `5.986 / 6.109` $\to$ `7.111 / 7.223` |
| **Sn132** | `-8.3536` | `-8.1342` | `-8.1030` | `4.7093` | `4.6436 (-1.40%)` | `4.6370 (-1.53%)` | `+0.2544` / `+0.2644` | `1g`: `4.712 / 4.948` $\to$ `5.734 / 6.000` |
| **Pb208** | `-7.8675` | `-7.6369` | `-7.6030` | `5.5012` | `5.4269 (-1.35%)` | `5.4252 (-1.38%)` | `+0.1903` / `+0.1993` | `1g`: `3.138 / 3.396` $\to$ `3.807 / 4.114` |

> [!NOTE]
> **Физический вывод Моста 6:** снижение $M^*/M_N$ в Парето-кандидате усиливает спин-орбитальное расщепление на $\sim 18\text{–}21\%$ (например, `O16 1p`: $5.23 \to 6.21\text{ МэВ}$, `Ca40 1d`: $5.41 \to 6.42\text{ МэВ}$, `Ca48 1f`: $6.15 \to 7.26\text{ МэВ}$), улучшает $R_{\text{ch}}$ и сжимает $\Delta R_{\text{ch}}(\text{Ca48}-\text{Ca40})$, но на фиксированном $\zeta_{\text{RMF}} = 0.020$ упирается в компромисс по $M_{\max}$ и слегка увеличивает ошибку энергии связи $B/A$.

---

## 4. Мост 7: Независимый Хартл $I$-Love-$Q$ и двухконденсатный отскок кускутона

Мост 7 (`14/14` тестов `pytest`, `--validate` OK) устраняет ключевой методологический пробел Цикла 1: вместо оценки $I$ и $Q$ через аппроксимационную формулу Yagi–Yunes на основе приливной деформируемости $\Lambda$ здесь решаются **прямые релятивистские уравнения медленного вращения Хартла 1-го и 2-го порядка** поверх TOV-профиля единого двухконденсатного EoS Закона 2, а космологический отскок переносится на синхронизованную 6-переменную систему с тем же самым двухконденсатным сектором материи.

### Часть A: Прямой ОДУ Хартла 1-го + 2-го порядка ($I$-Love-$Q$ и прогноз для `PSR J0737-3039A`)

| Звезда | $M$, $M_\odot$ | $R$, км | $I_{45}$, $10^{45}\,\text{г}\cdot\text{см}^2$ | $\bar{I}$ | $\bar{Q}$ | $\Lambda$ | Откл. от Yagi–Yunes (2017) $(I\text{-}\Lambda, Q\text{-}\Lambda, I\text{-}Q)$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`PSR_J0737_3039A`** | `1.3381` | `12.6492` | `1.4694` | `14.1470` | `6.4612` | `660.07` | `(0.0015, 0.0007, 0.0005)` |
| `canonical_1_4` | `1.4000` | `12.6363` | `1.5628` | `13.1366` | `5.9287` | `500.75` | `(0.0019, 0.0007, 0.0003)` |
| `heavy_1_8` | `1.8000` | `12.3377` | `2.1431` | `8.4761` | `3.3673` | `82.43` | `(0.0009, 0.0004, 0.0002)` |
| `massive_2_0` | `2.0000` | `11.7221` | `2.2913` | `6.6065` | `2.3300` | `23.90` | `(0.0030, 0.0009, 0.0006)` |

#### Калибровочные и сходимостные контроли Части A:
- **Ньютоновский предел сфероида Маклорена ($C \to 0$):**
  - $I/(M R^2) = 0.39999817$ (точное значение $2/5 = 0.4$; абс. ошибка `1.83e-06`);
  - $Q/(\Omega^2 R^5) = -0.49999358$ (точное значение $-1/2 = -0.5$; абс. ошибка `6.42e-06`);
  - `maclaurin_qtilde_ratio_C0 = 0.99996308` (абс. ошибка `3.69e-05`).
- **Масштабная инвариантность по $\Omega$:** `spin_rescaling_invariance_rel_err = 4.1280e-10`.
- **Сеточная радиальная сходимость на `PSR J0737-3039A`:** `j0737_Ibar_radial_conv_rel_err = 1.9138e-09` (`coarse = 14.14056`, `fine = 14.14698`), `j0737_Qbar_radial_conv_rel_err = 1.0364e-05`.

> [!WARNING]
> **Обнаруженная опечатка в историческом скрипте (документировано без правки файла):**
> Независимый расчёт квадруполя 2-го порядка выявил расхождение в старом плоттинг-скрипте [nvg_iloveq_plot.py](file:///Users/oleg/Documents/NVG-Research/verification/nvg_iloveq_plot.py#L53-L55): в словаре `QLOVE_COEFFS` записаны `d: -4.283e-4, e: 1.245e-6`, тогда как точные коэффициенты таблицы I из Yagi & Yunes (2017, Phys. Rep. 681) равны `d = -4.283e-3, e = 1.245e-4`. В Мосте 7 использована корректная параметризация Yagi & Yunes (2017), что дало согласие `0.02%–0.09%` по $Q$-$\Lambda$ и `0.02%–0.06%` по $I$-$Q$ во всём диапазоне масс; старый скрипт оставлен нетронутым согласно правилу неизменности отслеживаемых файлов.

### Часть B: Синхронизованный 6-переменный двухконденсатный отскок кускутона

| Сценарий | Ошибка связи (`cerr`) | Ошибка отскока (`berr`) | $K_{\max}$ | Диапазон $y_W$ | Диапазон $y_s$ | $Q_{\min}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `vacuum_bag_kappa1_tracking` | `3.32e-14` | `2.22e-15` | `5.7661` | `[0.9935, 1.0366]` | `[0.0557, 0.1189]` | `1.525` |
| `w0_scale_mu20_tracking` | `2.11e-15` | `5.11e-15` | `5.5970` | `[0.9968, 1.0608]` | `[0.0865, 0.1226]` | `1.525` |
| `w0_scale_W_zero_crossing` | `9.52e-12` | `9.55e-15` | `2.9853` | `[-1.3597, 1.3446]` | `[-0.1902, 0.1865]` | `1.525` |

Во всех трёх режимах гамильтонова связь сохраняется на уровне $\le 9.52 \times 10^{-12}$, условие отскока выполняется на уровне машинной точности ($\le 9.55 \times 10^{-15}$), а кинетический фактор стабильности остаётся строго ограниченным снизу ($Q_{\min} = 1.525 > 0$) без духовых неустойчивостей, в том числе при плавном пересечении $W = 0$.

---

## 5. Честные ограничения (`SPEC-v2`) и открытые задачи

1. **Первичные ворота насыщения при $\zeta_{\text{RMF}} = 0.020$ пусты (`primary_gate_satisfied = False`):** внутри заданного окна $M^*/M_N \in [0.59, 0.62]$ Парето-кандидат даёт $M_{\max} = 2.0503\,M_\odot < 2.08\,M_\odot$ и ухудшает `rms_B_over_A` по сравнению с базовой ветвью Цикла 1. Требуется либо ослабление/расширение ветви $\zeta_{\text{RMF}}$ и изовекторного сектора, либо учёт явных поправок к плотности/спариванию.
2. **Ограничения ядерной модели:** расчёт Дирака–Хартри пока охватывает 6 сферически-симметричных дважды-магических ядер без учёта квадрупольной деформации и спаривания (BCS/Hartree–Bogoliubov).
3. **Регистрация доказательности:** для всех 7 новых мостов `evidence_weight = 0.0`, пока они не зарегистрированы в `verification/registry.json` и `artifact_manifest.json` после ревью.
4. **Отложенный мелкий багфикс:** исправление `QLOVE_COEFFS` в [nvg_iloveq_plot.py](file:///Users/oleg/Documents/NVG-Research/verification/nvg_iloveq_plot.py#L53-L55) зафиксировано как отдельная чистая задача.

---

## 6. Команды полного воспроизведения

Полный объединённый прогон всех 7 мостов (`102 passed, 14 warnings in 51.24s` в системном `python3`):

```zsh
PYTHONPATH=/Users/oleg/Documents/NVG-Research/verification python3 -B -m pytest -q -p no:cacheprovider \
  verification/test_nvg_quartic_vector_melting_audit.py \
  verification/test_nvg_two_condensate_unified_audit.py \
  verification/test_nvg_unified_dirac_hartree_probe.py \
  verification/test_nvg_unified_gravity_bridge_audit.py \
  verification/test_nvg_unified_thermal_response_bridge.py \
  verification/test_nvg_unified_calibration_envelope_probe.py \
  verification/test_nvg_unified_astrophysics_stability_bridge.py
```

Валидация JSON-артефактов всех 7 мостов (`ALL_7_VALIDATIONS_OK`):

```zsh
for f in \
  verification/nvg_quartic_vector_melting_audit.py \
  verification/nvg_two_condensate_unified_audit.py \
  verification/nvg_unified_dirac_hartree_probe.py \
  verification/nvg_unified_gravity_bridge_audit.py \
  verification/nvg_unified_thermal_response_bridge.py \
  verification/nvg_unified_calibration_envelope_probe.py \
  verification/nvg_unified_astrophysics_stability_bridge.py; do
  python3 -B "$f" --validate || exit 1
done
```
