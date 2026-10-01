# Бюджет тёмной материи NVG: единая нормировка ещё не получена

## Что исправлено

Старый [`verification/nvg_dm_budget_audit.py`](verification/nvg_dm_budget_audit.py)
выдавал ручное «решение» вида `дефекты + χ + trace PBH = Omega_DM`.
Это выглядело как сведение теории, но не следовало из её вычислителей:
недостающий остаток просто назначался одному сектору. Скрипт заменён на
воспроизводимый **fail-closed** аудит. Он не подгоняет остаток, не складывает
несовместимые нормировки и не печатает состав Вселенной, пока для всех
компонентов нет независимо рассчитанной плотности.

## Один язык для сравнения: `omega = Omega h²`

Для reference-сравнения взяты значения base ΛCDM Planck
`TT,TE,EE+lowE+lensing`:

```text
omega_b = 0.02237 ± 0.00015
omega_c = 0.1200 ± 0.0012
H0      = 67.36 ± 0.54 км/с/Мпк
```

При `h=0.6736` это лишь арифметически переводится в `Omega_c≈0.26447` из
тех же коррелированных входов. Это **не**
автоматический fit NVG: если NVG меняет фон или возмущения CMB, нужна
совместная CMB-подгонка. Но пока такого fit нет, единая физическая плотность
`omega` — минимально честная общая шкала
([Planck 2018](https://arxiv.org/abs/1807.06209)).

## Что реально дают нынешние скрипты

| Сектор | Что вычислитель реально выдаёт | Почему это ещё не компонент единого бюджета |
|---|---|---|
| Реликтовые дефекты/W-прокси | Скрипт читает legacy-входы `Omega_DM=0.268` и `H0=72.8`, затем обращает их в `lambda_v` и `m_W` | Паспорт имеет статус `LEGACY_UNDOCUMENTED_CALIBRATION_NOT_ABUNDANCE_PREDICTION`: источник, ошибка и likelihood этих чисел в старом скрипте не сохранены. Получается `omega=0.142035712`, на **18.36%** выше reference `omega_c`; это калибровка, не производство частиц |
| PBH, популяция A | Нормированная форма профиля с суммой `fraction_sum=1` | Сумма формы не есть абсолютная доля PBH. Нет solver-а образования или `omega_PBH` |
| PBH, популяция B | Калиброванный диапазон seed-чисел и reference-плотность дают лишь условное `1.21×10⁻¹⁰…1.21×10⁻⁹` | Это не общий `f_PBH`: нет solver-а образования, общей космологии, `omega_PBH` или JWST likelihood. Масса `4×10⁵ M_sun` теперь сопоставлена с ближайшей канонической ступенью `N=10`, а не с прежней ошибочной `N≈20` |
| Тёмный нейтрон χ | При дополнительном предположении `N_chi=N_B`: `omega_chi=(m_chi/m_p)omega_b≈0.02240`, то есть около 18.64% reference CDM | Это условная арифметика. Механизм baryogenesis/cogenesis в родительском аудите помечен `RETIRED_MISSING_BARYOGENESIS_SOURCE` |

Вычитание условного χ из reference `omega_c` даёт `0.09763`. В результатах
оно намеренно называется `unassigned_reference_gap_for_diagnostic_only`:
это диагностическая разность, **не** цель для другого сектора и не готовое
распределение. Нельзя получать число вычитанием и объявлять его
предсказанием.

## Новый честный статус

```text
NOT_CLOSED_MISSING_PBH_NORMALIZATION_AND_VIABLE_COGENESIS
```

Это не означает, что «вся теория опровергнута». Это точный ответ на более
узкий вопрос: из текущих скриптов нельзя получить единый состав тёмной
материи без подгонки. До закрытия трёх конкретных дыр — абсолютной
нормировки PBH-A, перевода PBH-B из seed-калибровки в единый физический
расчёт и жизнеспособного source-complete cogenesis —
`composition=null` намеренно.

## Что даст настоящее продвижение

1. Вывести абсолютную скорость производства PBH/дефектов, включая обе
   PBH-популяции, из заранее заданного механизма и решить
   Boltzmann/formation equation.
2. Получить `omega_i` без ввода наблюдаемого `Omega_DM` в качестве цели.
3. Завершить динамику CP-источника, washout и энтропии для cogenesis либо
   снять χ с претензии на космологическую долю.
4. Затем проверить **один раз** `sum_i omega_i = omega_c` в одной фоновой
   космологии и только после этого сравнивать с CMB/BBN/структурой.

## Воспроизведение

```bash
python3 -B verification/nvg_dm_budget_audit.py \
  --output verification/nvg_dm_budget_audit_results.json
python3 -B verification/nvg_dm_budget_audit.py \
  --validate verification/nvg_dm_budget_audit_results.json
python3 -B -m pytest -q -p no:cacheprovider \
  verification/test_nvg_dm_budget_audit.py
```

Внешняя reference-нормировка хранится отдельно в
[`verification/data/planck2018_dm_budget.json`](verification/data/planck2018_dm_budget.json);
неподтверждённые legacy-входы реликтовой инверсии изолированы в
[`verification/data/legacy_relic_dark_matter_calibration.json`](verification/data/legacy_relic_dark_matter_calibration.json),
а legacy-входы seed-trace PBH-B — в
[`verification/data/pbh_population_b_seed_calibration.json`](verification/data/pbh_population_b_seed_calibration.json);
результат и условия fail-closed находятся в
[`verification/nvg_dm_budget_audit_results.json`](verification/nvg_dm_budget_audit_results.json).
