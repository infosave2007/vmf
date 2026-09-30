# Практический прототип согласованности пассивного RLC

Статус: **синтетический инженерный диагностический прототип**, `evidence_weight=0`. `REPORT_A.md` и `evidence_A/` — сохранённый исходный проход; после adversarial review его численные результаты **заменены для приёмки** исправленным `REPORT_A1.md` и `evidence_A1/`. Это не измерение NVG, не сертификация пассивности, не универсальный rational-fit-оракул и не заявление о физическом коэффициенте.

## Внешний JSON и знаки

`verification/nvg_passive_response_probe.py` принимает `nvg_passive_response_input.v1` с:

1. строго возрастающими частотами в Hz и комплексной проводимостью в S;
2. явной конвенцией `e^-iomega_t` и знаком `negative_for_capacitive_e_minus_iomega_t`;
3. объявленной независимой декартовой Gaussian-моделью (`sigma_real_S`, `sigma_imag_S`);
4. **отдельно измеренными** ограничениями низкочастотной ёмкости (`low_frequency_capacitance_F`) и высокочастотной индуктивности (`high_frequency_inductance_H`) с неопределённостями и `constraint_metadata.independent=true`.

```bash
python3 verification/nvg_passive_response_probe.py \
  --input verification/data/passive_response_example_2026.json
```

JSON в stdout строгий; запись выполняется только при явном `--output path.json`. Ошибки JSON, `NaN/Inf`, дубликаты частот, неверные единицы/полярность, нулевой шум и попытка выдать параметры, извлечённые из той же развертки, за независимые дают `status="FAIL"`, а не ложный PASS.

Зафиксирована серия

\[
Y(s)=\frac{1}{R+sL+1/(sC)},\qquad s=-i\omega.
\]

При `exp(-iωt)` у идеального конденсатора мнимая часть `Y` отрицательна на низких частотах, у индуктивности положительна на высоких. Для **пикового** фазора мощность проверяется независимо от представления:

\[
P=\tfrac12\operatorname{Re}(VI^*)
 =\tfrac12|V|^2\operatorname{Re}Y.
\]

`absorbed_power_watts(voltage_peak_v, ...)` принимает именно peak voltage; RMS-вход сначала умножается на `sqrt(2)`. Положительная серия `R,L,C` ограничена численными bounds (`R:10^{-6}..10^6 Ω`, `L:10^{-12}..10^6 H`, `C:10^{-15}..10^2 F`). Для параллельных ветвей endpoint-истина: `C_eff=ΣC_i`, `1/L_eff=Σ(1/L_i)`.

## Исправленные численные гейты

Кандидат фита пригоден только при явной сходимости оптимизатора и конечных параметрах, остатках, Jacobian и модельных значениях. Если нет конечного сошедшегося кандидата, CLI выдаёт строгий `status="FAIL"`, `decision="INDETERMINATE_NUMERICAL_FIT"`; это не физическое отвержение. Для условных интервалов требуется ранг Jacobian 3 и condition number `≤10^12`; иначе интервал помечается invalid и псевдообратная матрица не используется.

Whitened-Jacobian covariance равна `(J.T J)^-1` в координатах `log10(R,L,C)` при известных объявленных Gaussian σ, без умножения на остаточную дисперсию. Физические границы считаются как `10**(log10(value) ± 1.95996398454 σ_log10)` с overflow/underflow-статусом, без тихого clipping. Это локальная/asymptotic условная неопределённость, не универсальное coverage-утверждение и не глобальная граница неизвестного хвоста.

`sweep_only_baseline` фитирует только спектр, затем оценивает endpoint-остаток; `constrained_fit` совместно использует спектр и независимые endpoint. Для baseline `dof=2N-3`, для joint `dof=2N+2-3`; каждый метод получает собственные поля `decision_alpha` и `decision_threshold_chi2`. Оба метода получают один и тот же вход, superiority не заявляется.

## Исправленный benchmark A1

Полный замороженный до benchmark протокол: `Lunacy/runs/practical-prototypes-2026-09-19/evidence_A1/PROTOCOL.md`. Старый блок `2026092001..2026092128` воспроизведён только как `development_replay_old_holdout` без нового operating claim. Новый чистый split заранее объявлен и не пересекается со старым train/holdout/fixed:

- A1 train: `2026093901..2026093964`, 64 nominal rows;
- A1 heldout: `2026094001..2026094128`, **128 genuinely untouched nominal rows**;
- fresh fixed seeds `2026095001..2026095007`: exact, nominal noise, polarity, phase, extra positive branch, finite-band hidden branch, active/nonpassive control.

Результаты A1 heldout (все 128 строк сохранены в `evidence_A1/benchmark_results.json`):

| Метод | n | ложные тревоги | доля | Wilson 95% |
|---|---:|---:|---:|---:|
| sweep-only baseline | 128 | 1 | 0.0078125 | [0.0013804, 0.0429263] |
| constrained fit | 128 | 1 | 0.0078125 | [0.0013804, 0.0429263] |

Для синтетической nominal-проверки условные интервалы обеих процедур содержали известную истину в долях `R=122/128=0.953125`, `L=121/128=0.9453125`, `C=120/128=0.9375`; это только containment-диагностика данного генератора, не универсальное coverage-утверждение.

Преимущества constrained fit на этом фиксированном тесте не обнаружено. Все строки и ошибки сохранены: reversed-polarity control честно даёт `INDETERMINATE_NUMERICAL_FIT` (нет сошедшегося положительного RLC-кандидата), а не ошибочно объявляется физическим FAIL; phase fault/negative-R дополнительно имеют конечнополосный negative-real signal. Extra positive branch отвергается одной серией, а finite-band hidden branch проходит, демонстрируя неидентифицируемость полосы, а не отсутствие моды.

Синтетический эталон `R=10 Ω`, `L=0.01 H`, `C=1 µF` не является NVG-физикой. Нет global positive-real сертификата и доказательства отсутствия положительных мод вне окна; спектральная/Hankel-положительность на одной полосе их не находит.

## Проверки и ограничения

```bash
python3 -m pytest -q verification/test_passive_response_probe.py
python3 verification/nvg_passive_response_probe.py --benchmark \
  --output Lunacy/runs/practical-prototypes-2026-09-19/evidence_A1/benchmark_results.json \
  > Lunacy/runs/practical-prototypes-2026-09-19/evidence_A1/benchmark_stdout.json
```

Тесты покрывают знак и мощность, endpoint-алгебру, malformed/nonfinite/duplicate/unit/polarity/scaling controls, endpoint independence, truth-label non-use, convergence/rank/covariance gates, exactdata nonzero intervals, alpha/dof semantics, overflow-safe CLI, phase/active controls и finite-band limit. Нет hardware, material calibration, GUI, SDP, network access, строгой сертификации positive-realness или физической интерпретации NVG.
