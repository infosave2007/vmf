# Гравитация Нулевого Вектора (NVG) и Вакуумная Массовая Фракция (VMF)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://python.org)
[![CI Build & Verification](https://github.com/infosave2007/vmf/actions/workflows/verify.yml/badge.svg)](https://github.com/infosave2007/vmf/actions/workflows/verify.yml)

**English version:** [README.md](README.md) · **Политика публикаций и provenance:** [PUBLICATION_STATUS.md](docs/reports/PUBLICATION_STATUS.md)

**NVG / VMF** — ковариантная эффективная теория поля вакуумного конденсата (с $U(1)_B$-вектором и скаляром/кускутоном), связывающая уравнение состояния плотной ядерной материи, структуру магических ядер, тепловые тождества Уорда, нейтронные звёзды (TOV + 1-й/2-й порядок Хартла) и неосингулярный космологический отскок на масштабе КХД без переподгонки параметров между секторами (`zero-refit`).

> [!IMPORTANT]
> **Граница научной строгости (`evidence_weight = 0.0`):** все расчёты в репозитории имеют нулевой независимый эмпирический вес. Документ чётко разделяет: (1) математически и численно замкнутые мосты, (2) условные/внутривыборочные сравнения с наблюдениями, (3) проверяемые форвардные предсказания и (4) открытые проблемы и отозванные гипотезы (`Отозвано`). Подробнее см. [PUBLICATION_STATUS.md](docs/reports/PUBLICATION_STATUS.md) и [реестр доказательств](docs/reports/NVG_EVIDENCE_STATUS_RU.md).

### Навигация
1. [Главное: таблица совпадений, расхождений и статусов](#1-главное-таблица-совпадений-расхождений-и-статусов)
2. [Ключевые проверяемые предсказания (фальсифицируемость)](#2-ключевые-проверяемые-предсказания-фальсифицируемость)
3. [Архитектура модели и профильные аудиты](#3-архитектура-модели-и-профильные-аудиты)
4. [Быстрый старт и воспроизводимость](#4-быстрый-старт-и-воспроизводимость)
5. [Структура репозитория](#5-структура-репозитория)
6. [Исторические препринты (Zenodo)](#6-исторические-препринты-zenodo)

---

## 1. Главное: таблица совпадений, расхождений и статусов

### 1.1 Единая 10-мостовая ковариантная система (`zero-refit`, Циклы 1–4)

Сквозная вычислительная цепочка из **10 валидированных мостов** с нулевой переподгонкой параметров между масштабами ($k_{\rm fin,nuc} = 0$) и нулевым независимым весом доказательств (`evidence_weight = 0.0`). Подробные отчёты: [Цикл 1 (мосты 1–4)](docs/unified/NVG_UNIFIED_SYSTEM_BRIDGE_2026_11_26_RU.md), [Цикл 2 (мосты 5–7)](docs/unified/NVG_UNIFIED_SYSTEM_CYCLE2_2026_11_26_RU.md), [Цикл 3 (Мост 8)](docs/unified/NVG_UNIFIED_SYSTEM_CYCLE3_2026_11_26_RU.md), [Цикл 4 (Мосты 9A–9B)](docs/unified/NVG_UNIFIED_SYSTEM_CYCLE4_2026_11_26_RU.md) · **Статья APS (`revtex4-2`, PDF):** [NVG_Zero_Refit_Nuclear_NS_Paper.pdf](preprints/NVG_Zero_Refit_Nuclear_NS_Paper.pdf).

| Мост | Физический сектор | Ключевой численный результат | Скрипт / Артефакт |
|---|---|---|---|
| **Мост 1** | Плавление вектора в плотной материи (Закон 1) | Квартичный член $\lambda_v(\sigma, W)$ стабилизирует ветку; $W_0$ не коллапсирует при $n_B > 0$ | [nvg_quartic_vector_melting_audit.py](verification/nvg_quartic_vector_melting_audit.py) |
| **Мост 2** | Двухконденсатная EOS материи и НЗ (Закон 2) | Глобально ограниченная снизу EOS; $M_{\max} = 2.093\,M_\odot$, $R_{1.4} = 12.95$ км | [nvg_two_condensate_unified_audit.py](verification/nvg_two_condensate_unified_audit.py) |
| **Мост 3** | Первичные ворота насыщения (Цикл 3: Мост 8) | Скан $\zeta_{\rm RMF} \in \{0.0165, 0.0175, 0.0185\}$: **8 выживших точек**, победитель $\Lambda_{1.4} = 575.3$ | [nvg_unified_primary_gate_pairing_bridge.py](verification/nvg_unified_primary_gate_pairing_bridge.py) |
| **Мост 4** | Конечные ядра Дирака–Хартри (`zero-refit`, Закон 3A) | `O16..Pb208` без подгонки по ядрам: $\text{RMS}(B/A) = 0.236$ МэВ, $\text{RMS}(R_{\rm ch}) = 1.44\%$, положительные кожи `Ca48/Pb208` | [nvg_unified_dirac_hartree_probe.py](verification/nvg_unified_dirac_hartree_probe.py) |
| **Мост 5** | Спаривание RH-BCS в изотопах олова (Мост 8) | Самосогласованная связь $C_n = 21.0$ МэВ: точный коллапс $\Delta_n = 0$ на `Sn132` и щели $\Delta_n \in [0.6, 1.0]$ МэВ на `Sn116/120/124` | [nvg_unified_primary_gate_pairing_bridge.py](verification/nvg_unified_primary_gate_pairing_bridge.py) |
| **Мост 6** | Тепловой отклик и тождества Уорда (5×5 Schur) | Согласование с точностью `4.4e-15` (против провала `1.44` на несогласованном `ph-only` контроле) | [nvg_unified_thermal_response_bridge.py](verification/nvg_unified_thermal_response_bridge.py) |
| **Мост 7** | Независимый Хартл 1-го и 2-го порядка `I-Love-Q` | Для `PSR J0737-3039A` напрямую из ОДУ: $\bar{I} = 14.69$, $\bar{Q} = 6.74$, согласие с Yagi–Yunes `0.07%–0.21%` | [nvg_unified_astrophysics_stability_bridge.py](verification/nvg_unified_astrophysics_stability_bridge.py) |
| **Мост 8** | Ковариантная гравитация, регулярное ядро ЧД и отскок | Доказаны $U(1)_B$-отщепление, анизотропия Хейворда ($P_r = -\varepsilon \neq P_t$) и 6-переменный отскок кускутона (Законы 3B и 4) | [nvg_unified_gravity_bridge_audit.py](verification/nvg_unified_gravity_bridge_audit.py) |
| **Мост 9A** | Спинодаль кора–ядро, DUrca и моды Коулинга $f/g_1/g_2$ | Все 8 выживших: $n_t^{\rm dyn} = 0.0785\text{–}0.0901 < n_t^{\rm ther}$, $I_{\rm crust}/I = 3.84\%\text{–}5.16\%$ ($>1.4\%$ Вела), $M_{\rm DUrca} = 1.926\,M_\odot$, $\nu_f = 2.06\text{–}2.36$ кГц, $\nu_{g_1} \approx 200$ Гц, $\nu_{g_2} \approx 126\text{–}128$ Гц | [nvg_unified_ns_microphysics_bridge.py](verification/nvg_unified_ns_microphysics_bridge.py) |
| **Мост 9B** | Полная цепочка `Sn100..Sn134`, `PREX-2/CREX` и бенчмарк RMF | 11 изотопов Sn: двойной коллапс при $N=50, 82$, перегиб $\delta R_{\rm ch}(^{134}\text{Sn}-^{132}\text{Sn}) = +0.0173$ фм (77% от Gorges 2019); единственная из 6 моделей, проходящая вакуум и НЗ при $k_{\rm fin,nuc}=0$ | [nvg_unified_nuclear_benchmarks_bridge.py](verification/nvg_unified_nuclear_benchmarks_bridge.py) |

### 1.2 Сводная таблица runtime-сравнений с данными, расхождений и статусов

**Легенда:** ✅ согласовано с эталоном / решёткой · ⚙️ условный или калибровочный расчёт · 📏 интервальное предсказание · 📐 математическое тождество / кинематика · ⚪ нуль-тест · ⚠️ расхождение с данными · ❌ открытая проблема или отозванная гипотеза.

| # | Наблюдаемая / сектор | Результат NVG (runtime) | Эксперимент / эталон | Статус и научная граница |
|---|---|---|---|---|
| 1 | Якорь КХД ($M_{\Omega,0}$) | $M_{\Omega,0} = 859 \pm 8$ МэВ ($\approx 91\%$ массы нуклона) | Вычитание $\sigma_{\pi N} \approx 44$ МэВ, $\sigma_{sN} \approx 30$ МэВ из $M_N$ | ⚙️ Заданный входной масштаб (не предсказание) |
| 2 | $M_{\max}$ нейтронной звезды (канон) | $M_{\max} = 2.048\,M_\odot$ (каноническая TOV-цепочка) | PSR J0740+6620: $2.14 \pm 0.10\,M_\odot$ ($-0.92\sigma$) | ⚙️ Условно/внутривыборочно ([run_nvg_suite.py](verification/run_nvg_suite.py)) |
| 3 | Радиус $R_{1.4}$ НЗ (канон) | $R_{1.4} = 12.550$ км (каноническая TOV-цепочка) | NICER J0030: $12.2 \pm 0.5$ км ($+0.70\sigma$); J0437: $11.36 \pm 0.8$ км | ⚙️ Условно/внутривыборочно ([run_nvg_suite.py](verification/run_nvg_suite.py)) |
| 4 | Приливная деформируемость $\Lambda_{1.4}$ | $\Lambda_{1.4} = 519.4$ (TOV + уравнение Хиндерера) | GW170817: $190^{+390}_{-120}$ (пул $+0.84\sigma$) | ⚙️ Условно/внутривыборочно ([nvg_tidal_deformability.py](verification/nvg_tidal_deformability.py)) |
| 5 | Совместный пул НЗ ($\chi_\nu^2$) | Сокращённое $\chi_\nu^2 = 0.684$ по 3 runtime-строкам | Выбор CSS-перехода по J0740 / GW170817 / NICER | ⚙️ Описательная проверка внутри выборки ([nvg_joint_ns_inference.py](verification/nvg_joint_ns_inference.py)) |
| 6 | Соотношение `I-Love-Q` (legacy vs 2-й пор.) | В новом ОДУ 2-го порядка $\tilde{q} = 6.508$ (откл. Yagi–Yunes `0.61%`); старый [nvg_iloveq_plot.py](verification/nvg_iloveq_plot.py) работает **только как преобразование** | Универсальные соотношения Yagi–Yunes | ⚙️ Условно для одного EOS; legacy-график только как преобразование |
| 7 | Слепая энергия симметрии ($j$) | Константа $j = 11.136$ МэВ, $J_{\rm design} = 23.780$ МэВ, $g_\rho = 6.60$ | Остаток Бете–Вайцзеккера падает `202.7 → -3.4 МэВ` (~60×); RMF `4.1–6.8` | 📐 Пререгистрированная идентификация ([отчёт](docs/reports/NVG_J_IDENTIFICATION_RU.md)) |
| 8 | Масса скалярного глюбола ($0^{++}$) | $M_{\rm glueball} = 2 M_\Omega \approx 1.72$ ГэВ | Решёточная КХД: $1.7 \pm 0.1$ ГэВ (эксперимент $f_0(1710)$ не устоялся) | ✅ Совместимо с решёткой ([nvg_glueball_mass.py](verification/nvg_glueball_mass.py)) |
| 9 | PPN в слабом поле и скорость ГВ | $\gamma_{\rm PPN} = 1.0$, $\beta_{\rm PPN} = 1.0$, $c_T = c$, сдвиг периастра $\sim 1.6 \times 10^{-10}$ | Cassini, LLR, GW170817 ($|c_T/c - 1| < 10^{-15}$) | ✅ В пределах точности измерений ([nvg_weak_field_ppn.py](verification/nvg_weak_field_ppn.py)) |
| 10 | Инстантон Генезиса и $H_0$ (цикл $n=77$) | $N_e \in [52.68, 53.38]$, горизонт поворота даёт $H_0 \in [54.3, 108.5]$ км/с/Мпк (середина $\approx 76.8$) | SH0ES $73.04 \pm 1.04$, Planck $67.4$; lite-сравнение CMB низких $\ell$ — **не likelihood Planck** | 📏 Интервальное предсказание ([nvg_cmb_lowl_refit.py](verification/nvg_cmb_lowl_refit.py)) |
| 11 | Рост структур $S_8$ и ТЭ DESI | $S_8 \approx 0.851$ ($4.4\sigma$ от линзирования); прежний механизм $w_0\text{–}w_a$ в отставке | Плавление сдвигается в сторону *ухудшения* $S_8$; нет совместной области с DESI DR2 | ⚠️ Ухудшает напряжение; открытая проблема ([nvg_s8_tension_check.py](verification/nvg_s8_tension_check.py)) |
| 12 | Охлаждение белых карликов и LZ 248 кэВ | Сдвиг возраста БК $\Delta t/t \approx -1.8\times 10^{-6}$; кинематический экран LZ исключает упругое рассеяние мод $0.783 / 1.245$ ГэВ | Погрешность Gaia/SDSS $\sim 5\%$; для 248 кэВ нужен кандидат $\ge 70.66$ ГэВ | ⚪ / 📐 Ненаблюдаемо мал для БК; LZ не объяснён моделью ([разбор](docs/reports/NVG_LZ_248KEV_AUDIT_RU.md)) |
| 13 | Скан тёмного фотона SN1987A | Условный (conditional) sensitivity-скан без источникового likelihood и транспорта не устанавливает предел Раффельта | Тест требует полноценного расчёта переноса и спектрального likelihood SN1987A | ⚙️ Условный (без доказательств; физический вывод заблокирован) |
| 14 | Физические кривые охлаждения (Cas A / гипероны / $g$-моды) | В холодной баротропной EOS тождественно $N^2 = 0$; без $\Gamma_1(Y_i)$ и матричных элементов физический расчёт не замкнут | Наблюдательные утверждения по Cas A, гиперонной загадке и частотам $g$-мод сняты до появления микрофизики | ❌ Заблокировано ([nvg_cooling_gmode_dependency_audit.py](verification/nvg_cooling_gmode_dependency_audit.py)) |
| 15 | Стохастический фон ГВ (PTA / NANOGrav) | **Отозвано.** Аудит ПЧД фиксирует дефицит амплитуды $\sim 3\times 10^5$ в seed-окне, а космологический отскок лежит в мкГц-полосе | Сигнал PTA текущей моделью не объясняется и механизма не имеет | ❌ Отозвано ([P1-S3-pbh-nanograv-audit.json](verification/P1-S3-pbh-nanograv-audit.json)) |
| 16 | Сверхмассивные ЧД JWST ($z=6\text{–}15$) | **Объектное заявление JWST отозвано:** нет машиночитаемого каталога, модели заселённости и likelihood отбора | Шаблон ступеней ПЧД $0.38 \times 4^N\,M_\odot$ условен и не нормирован как доля ТМ | ❌ Отозвано ([nvg_prospective_falsification_reach.py](verification/nvg_prospective_falsification_reach.py)) |
| 17 | FRB-статистика, гомохиральность ДНК и бариогенезис | Отсутствуют связанные каталоги DM/масс магнетаров, молекулярный PVED и washout-сеть CP-нарушения | Прежние эмпирические сопоставления и проценты доверия сняты как неподтверждённые | ❌ Отозвано ([nvg_chime_frb_check.py](verification/nvg_chime_frb_check.py), [nvg_dna_chirality.py](verification/nvg_dna_chirality.py)) |
| 18 | Квантовые основания (Белл / Маделунг / $1/\alpha$) | Форма Маделунга воспроизводит $|\psi|^2$, а контекстуальная квантовая $\theta$ даёт $S=2\sqrt{2}$; классическая локальная $\theta$ даёт $S \le 2$ | Квантовая механика является *входом* представления, а не выводом из классического действия | 📐 Представление, не вывод ([nvg_bell_from_action.py](verification/nvg_bell_from_action.py)) |

---

## 2. Ключевые проверяемые предсказания (фальсифицируемость)

Все строки ниже являются **форвардными расчётами** или **условными тестами расширений**, ожидающими независимых данных (никакое текущее подтверждение не заявляется).

| Канал / Наблюдаемая | Предсказание модели | Эксперимент | Критерий фальсификации | Скрипт |
|---|---|---|---|---|
| Момент инерции и квадруполь `PSR J0737-3039A` | $\tilde{q} = 6.508$ (2-й пор. Хартл, канон EOS); $I_A$ из условной полосы | Радиотайминг (SKA / MeerKAT) | Выход измеренного $(I_A, \Lambda_A, R_A)$ за условную полосу прогноза | [nvg_ns_frozen_forecasts.py](verification/nvg_ns_frozen_forecasts.py) |
| Поверхностное красное смещение и пост-мержер НЗ | $z_{\rm surf} \approx 0.222$ ($1.4\,M_\odot$); $f_{\rm peak} \approx 2.42$ кГц ($R_{1.6} \approx 12.65$ км) | STROBE-X / eXTP; LIGO O5 / ET | Отклонение прямого измерения $z_{\rm surf}$ или пика $f_{\rm peak}$ | [nvg_ns_redshift.py](verification/nvg_ns_redshift.py) |
| Спектр $\rho$-мезона в ядерной материи | Селектор архитектур: полюс $\ge 755$ МэВ (арх. A) или $715\text{–}755$ МэВ (арх. B) | HADES / CBM (FAIR) / NICA | Надёжный полюс $< 715$ МэВ или отсутствие сдвига/уширения | [HADES_PREREGISTRATION.md](verification/HADES_PREREGISTRATION.md) |
| Треугольник масс нейтрино ($\theta$-seesaw) | $\Sigma m_\nu = 59.0$ мэВ; $m_\beta = 8.9$ мэВ (исправл. с прежних 29 мэВ); $m_{\beta\beta} \le 6.4$ мэВ | CMB-S4 · Project 8 · $0\nu\beta\beta$ | $\Sigma m_\nu < 45$ мэВ, $m_\beta > 20$ мэВ или $m_{\beta\beta} > 10$ мэВ исключают сектор | [nvg_sigma_mnu_corner.py](verification/nvg_sigma_mnu_corner.py), [nvg_katrin_mbeta.py](verification/nvg_katrin_mbeta.py) |
| Карточка эхо ГВ на кандидатах O4 | Для $M_{\rm tot} = 61.8\text{–}70.2\,M_\odot$: $\Delta t_{\rm echo} = 20.0\text{–}23.9$ мс, $f_{\rm QNM} = 245\text{–}278$ Гц | Полный когерентный стек LIGO O4+ | Нулевой глубокий стек O4 на табличных задержках гонит $R_{\rm eff} \to 0$ | [nvg_echo_o4_observers.py](verification/nvg_echo_o4_observers.py) |
| Тензорная мода и двупреломление CMB | $r(\ell < 10) < 0.001$; статическая ветвь $\beta_{\rm biref} = 0$ | LiteBIRD | Детекция $r > 0.001$ или подтверждение $\beta \neq 0$ (фиксирует $\gamma_{\rm topo}\Delta\theta$) | [nvg_litebird_prediction.py](verification/nvg_litebird_prediction.py), [nvg_cmb_birefringence.py](verification/nvg_cmb_birefringence.py) |
| RHIC Bell Test (смерть запутанности при $T_c$) | $S_{\rm CHSH}(T > T_c = 157\text{ МэВ}) \to 0$ (будущий протокол) | RHIC BES-II / тяжелые ионы | Сохранение квантовой Белл-корреляции $S > 2$ выше деконфайнмента | [nvg_rhic_bell_test.py](verification/nvg_rhic_bell_test.py) |
| Безгоризонтный приливный нагрев и сверхизлучение | Нулевой фазовый сдвиг поглощения в EMRI (против ~32 циклов Керра); полоса ПЧД $M \sim 10^{-6} M_\odot$ | LISA / наблюдательная астрофизика | Обнаружение стандартного керровского поглощения ~32 циклов или горизонта | [nvg_tidal_heating_null.py](verification/nvg_tidal_heating_null.py), [nvg_theta_superradiance.py](verification/nvg_theta_superradiance.py) |
| Галоскоп линии $\theta$-ТМ (`c_nu = 1`) | $m_\theta = 53.2$ мкэВ, $f_\theta = 12.86$ ГГц, $g_{\theta\gamma\gamma}^{\rm KSVZ} = 2.08\times 10^{-14}\text{ ГэВ}^{-1}$; $\sim 18$ ч при $\beta^2 = 10^4$ | Окно ORGAN / фиксированный диэл. стек | Нулевой целевой поиск исключает KSVZ-ориентир (с учётом доли $f_{\rm DM} = 0.24$) | [nvg_theta_haloscope_design.py](verification/nvg_theta_haloscope_design.py) |

---

## 3. Архитектура модели и профильные аудиты

Теоретический каркас строится вокруг единого калибровочного масштаба $M_{\Omega,0} = 859 \pm 8$ МэВ и делится на три сектора, разделённых по степени математической замкнутости:

1. **Плотная ядерная материя (VMF) и конечные ядра:** однокомпонентное `source_complete` действие строго не даёт насыщения и самосвязанных капель Томаса–Ферми ($C = 2.72164 > 1/4$, $\lambda_{c}^{\rm formal} \approx 0.0011$ против $1.05$; см. [аудит Q4](docs/reports/NVG_SOURCE_COMPLETE_Q4_DROPLET_AUDIT_RU.md), [NN-аудит](docs/reports/NVG_SOURCE_COMPLETE_NN_STATIC_AUDIT_RU.md), [S-wave порог](docs/reports/NVG_SOURCE_COMPLETE_FORMAL_SWAVE_THRESHOLD_AUDIT_RU.md)). Добавление скалярного потенциала с шестой степенью и квартичного векторного плавления (**Мосты 1–5**) устраняет этот разрыв: единая фиксация параметров из бесконечной материи воспроизводит 6 дважды магических ядер в схеме Дирака–Хартри без единой подгонки по ядрам.
2. **Универсальность плотностного предела и конечные температуры:** для потенциалов $U \sim a W^p$ доказан предел $\lim (P/\varepsilon) = \lim (dP/d\varepsilon) = \max(1/3, (p-2)/(p+2))$ ([отчёт](docs/reports/NVG_DENSITY_UNIVERSALITY_RU.md)); при $T > 0$ построены 160+32 строк комплексного отклика и 32 строки composition-Гессиана, удовлетворяющих тождеству Уорда на уровне $10^{-15}$ ([тепловой отклик](docs/reports/NVG_THERMAL_MASTER_RESPONSE_RU.md), [Hessian](docs/reports/NVG_THERMAL_COMPOSITION_HESSIAN_RU.md), [статический предел](docs/reports/NVG_THERMO_STATIC_ZERO_MODE_RU.md)).
3. **Космология, регулярное ядро и прикладная ветка:** проанализированы условный type-II вихревой сектор ($\beta = 2.53 > 1$, [отчёт](docs/reports/NVG_SOURCE_COMPLETE_CONDITIONAL_VORTEX_SECTOR_AUDIT_RU.md)), кинематическое замыкание минимального вакуума ([отчёт](docs/reports/NVG_SOURCE_COMPLETE_MINIMAL_VACUUM_CLOSURE_AUDIT_RU.md)) и **прикладной аудит генерации энергии** ([nvg_generator_practical_audit.py](verification/nvg_generator_practical_audit.py), [отчёт](verification/NVG_GENERATOR_PRACTICAL_AUDIT_RU.md)), который отбрасывает неподтверждённые вакуумные/over-unity схемы и оставляет расчётную инженерную модель термоэлектрического генератора (Seebeck) на отходящем тепле.

---

## 4. Быстрый старт и воспроизводимость

Обычный научный контур поддерживается на **Python 3.12** с зафиксированными версиями в [requirements.txt](requirements.txt) и [constraints-python312.txt](constraints-python312.txt) (`NumPy 1.26.4`, `SciPy 1.13.1`, `mpmath 1.3.0`, `PyCBC`).

```bash
# 1. Создать окружение Python 3.12 и установить зафиксированные зависимости
python3.12 -m venv .venv-research
.venv-research/bin/python -m pip install -r requirements.txt -c constraints-python312.txt
.venv-research/bin/python -m pip check

# 2. Запустить каноническую научную верификацию (единая точка входа, как в GitHub Actions)
.venv-research/bin/python verification/run_verification.py
```

### Что проверяет [`verification/run_verification.py`](verification/run_verification.py)
1. **Целостность реестра и артефактов:** валидирует [реестр скриптов и артефактов](verification/registry.json) и манифесты [`verification/data/provenance.json`](verification/data/provenance.json) и [`verification/artifact_manifest.json`](verification/artifact_manifest.json).
2. **Интеграционный реестр (`P4-S1`):** запускает [nvg_predictive_research_ledger.py](verification/nvg_predictive_research_ledger.py) с аутентификацией всех живых производителей и артефактов.
3. **Канонический runtime-отчёт:** пересчитывает `M_max`, `R_1.4`, `Lambda_1.4` и `chi2` через [`verification/run_nvg_suite.py`](verification/run_nvg_suite.py) в [NVG_FINAL_REPORT.md](docs/reports/NVG_FINAL_REPORT.md).
4. **Полный семантический `pytest`:** выполняет `python -m pytest -q verification` по всем 120+ тестовым наборам.

*(Отдельная команда [`verification/run_all_checks.py`](verification/run_all_checks.py) является лишь process-smoke проверкой запускаемости скриптов и не считается научным свидетельством.)*

<details>
<summary>Точечные команды для проверки 8-мостовой системы и тепловых мостов</summary>

```bash
# Единая 8-мостовая система (валидация сохранённых артефактов и прогон pytest)
.venv-research/bin/python verification/nvg_quartic_vector_melting_audit.py --validate verification/nvg_quartic_vector_melting_results.json
.venv-research/bin/python verification/nvg_two_condensate_unified_audit.py --validate verification/nvg_two_condensate_unified_results.json
.venv-research/bin/python verification/nvg_unified_dirac_hartree_probe.py --validate verification/nvg_unified_dirac_hartree_results.json
.venv-research/bin/python verification/nvg_unified_primary_gate_pairing_bridge.py --validate verification/nvg_unified_primary_gate_pairing_results.json
.venv-research/bin/python verification/nvg_unified_thermal_response_bridge.py --validate verification/nvg_unified_thermal_response_results.json
.venv-research/bin/python verification/nvg_unified_astrophysics_stability_bridge.py --validate verification/nvg_unified_astrophysics_stability_results.json
.venv-research/bin/python verification/nvg_unified_gravity_bridge_audit.py --validate verification/nvg_unified_gravity_bridge_results.json

# Тепловой отклик и composition Hessian
PYTHONDONTWRITEBYTECODE=1 .venv-research/bin/python -B verification/nvg_thermal_master_response.py --output /tmp/thermal_master.json
PYTHONDONTWRITEBYTECODE=1 .venv-research/bin/python -B verification/nvg_thermal_composition_hessian.py --output /tmp/thermal_hessian.json
```

</details>

---

## 5. Структура репозитория

```text
NVG-Research/
├── README.md                              # Английская документация (зеркало)
├── README_RU.md                           # Русская документация (этот файл)
├── docs/
│   ├── unified/                           # Отчёты единой 8-мостовой системы (Циклы 1–3)
│   └── reports/                           # Профильные аудиты, PUBLIC_STATUS.md, NVG_FINAL_REPORT.md
├── verification/                          # Основной вычислительный код, тесты, реестры и артефакты
│   ├── run_verification.py                # Главная научная точка входа CI
│   ├── run_nvg_suite.py                   # Канонический runtime-отчёт (TOV + Hinderer)
│   ├── nvg_predictive_research_ledger.py  # Сводный fail-closed реестр P4-S1
│   ├── registry.json                      # Индекс ролей скриптов, тестов и артефактов
│   ├── artifact_manifest.json             # SHA-256 контрольные суммы сгенерированных артефактов
│   ├── data/                              # Входные каталоги и provenance.json
│   └── nvg_*.py / test_*.py               # Физические модули, мосты 1–8 и pytest-наборы
├── article/                               # Опубликованные статьи и материалы
├── preprints/                             # Сборники препринтов (TeX / PDF)
├── visualization/                         # Интерактивные 3D HTML-визуализации
├── requirements.txt                       # Зависимости Python
└── constraints-python312.txt              # Зафиксированные версии (NumPy 1.26.4, SciPy 1.13.1 и др.)
```

---

## 6. Исторические препринты (Zenodo)

<details>
<summary>Библиография опубликованных препринтов (не текущее runtime-свидетельство)</summary>

Опубликованные названия и ссылки сохранены как историческая библиография (не текущее runtime-свидетельство). Их выводы не становятся автоматически результатами текущей исполняемой модели.

- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20214457-blue.svg)](https://doi.org/10.5281/zenodo.20214457) *Lattice Sigma Terms as an Anchor for the Dense Nuclear Matter Equation of State*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269567-blue.svg)](https://doi.org/10.5281/zenodo.20269567) *Analytic Derivation of the Dense Matter Equation of State and Maximum Neutron Star Mass via QCD Vacuum Condensate Phase Transitions*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269725-blue.svg)](https://doi.org/10.5281/zenodo.20269725) *Geometric Truncation of Low-Multipole CMB Power and Null B-Mode Prediction from a QCD-Scale Euclidean Instanton Bounce*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20269816-blue.svg)](https://doi.org/10.5281/zenodo.20269816) *A Discrete Cyclic Mass Hierarchy 4^N for Primordial Black Holes: Bridging Asteroid-Mass Dark Matter to Early JWST Heavy Seeds*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20270025-blue.svg)](https://doi.org/10.5281/zenodo.20270025) *Resolution of the Slow-Rotating Magnetar Paradox via QCD Vacuum Permeability Phase Transitions*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20270202-blue.svg)](https://doi.org/10.5281/zenodo.20270202) *Eliminating the Observer Effect: Wave Function Collapse as Deterministic Topological Reconnection in a Condensate Vacuum*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20463836-blue.svg)](https://doi.org/10.5281/zenodo.20463836) *Neutron Star Structure from a Single QCD Parameter: Equation of State, Tidal Deformability, and Cooling Threshold in the Null-Vector Gravity Framework*
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20473318-blue.svg)](https://doi.org/10.5281/zenodo.20473318) *Dynamics of the QCD Vacuum Condensate Amplitude in Dense Matter and Cosmology* (историческое предложение; в `source_complete` действии векторная энергия расходится как $W^{-2}$, плавление при $2.05\,n_0$ не поддерживается)
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20476890-blue.svg)](https://doi.org/10.5281/zenodo.20476890) *Seven Results of the Vacuum Condensate: From Nuclear Matter to Quantum Mechanics* (иллюстрации Маделунга и Эйнштейна–Картана; сектор кручения в поддерживаемом действии отсутствует)
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20485836-blue.svg)](https://doi.org/10.5281/zenodo.20485836) *Resolution of the Hyperon Puzzle via QCD Vacuum Condensate Melting in the NVG/VMF Framework* (полной $\beta$-равновесной EOS и likelihood в репозитории нет)
- [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21812161-blue.svg)](https://doi.org/10.5281/zenodo.21812161) *A Point-Like Dark-Matter Target at 12.8 GHz from a QCD-Anchored Vacuum Condensate: Theory and Haloscope Test Protocol* (условный целевой протокол с параметром $c_\nu$; воспроизведён в [nvg_theta_haloscope_design.py](verification/nvg_theta_haloscope_design.py))

</details>

---

**Автор:** Олег Кириченко (`Oleg Kirichenko`) — Независимый исследователь — urevich55@gmail.com  
**Лицензия:** MIT License — см. [LICENSE](LICENSE).
