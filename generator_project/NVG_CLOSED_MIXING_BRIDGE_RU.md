# I1 — закрытое смешение, обратимая калорика и сравнение при `P=const`

**Дата/контракт:** 28.09.2026, sealed run `nvg-closed-response-adhd-2026-09-28`.
Это приватный воспроизводимый результат; публичных записей/Git-публикации нет.
Использованы только живые конструкторы принятого W8/U16 EOS и конечные-T интегралы.
Результат JSON — выходной снимок, не вход в расчёт; коэффициенты не подгонялись и новое
взаимодействие не вводилось.

## Короткий FINAL

**PASS_LIVE_CLOSED_MIXING_BRIDGE.** Построены 4 ссылки (`y*=0.90,0.93`, W8/U16),
36 композиций (`δ=D/B=0,±0.025,±0.05,±0.1,±0.2`) и два условных контакта
(`Cρ=0`, frozen historical `j`), то есть 72 строки. На primary tree прошли 72/72
локальных стабильных корня и 72/72 fixed-pressure endpoint; независимая работа
`μ_D` с 8/12 точками выполнена только для `δ=+0.2` и всех 4 ссылок × 2 контактов.
Проверка `--validate` и 6 тестов одного живого `build_result()` прошли.

Это условие на равновесные однородные endpoints в локально выпуклой ветви, не
доказательство глобальной фазы, не скорость/мощность/устройство и не экспериментальная
верификация. `x_hot` — идеальная термодинамическая верхняя граница относительно `T0`,
не КПД прибора.

## 1. Что именно посчитано

Единицы внутри: `ℏ=c=1`, `B,D` в MeV³, `f,u,P` в MeV⁴, `s` в MeV³; публичные
плотности — fm⁻³. Зафиксированы `T0=70 MeV`, `μref=775 MeV`, `j=11.13601607117819 MeV`,
`n0=0.16 fm⁻³`, `ℏc=197.3269804 MeV fm`, без fit.

* **Closed mixing:** две равные макроскопические камеры с `B=B0`, `D=±δB`,
  жёсткий объём и фиксированные `Nn,Np`; нет внешнего тепла/работы. Решается
  `u(B,0,Tmix)=u(B,D,T0)`, затем считается `Δσmix/B`.
* **Изотермическое разделение:** `B,T0` фиксированы. Хранится
  `Δu/B`, `Qprep=T0 Δσsep`, `wiso=Δf/B=Δu/B−Qprep`; для `δ=+0.2`
  независимо проверено `B⁻¹∫_0^D μD(B,d,T0) dd` 8- и 12-точечной квадратурой.
* **Обратимое разделение:** при фиксированном `B` решается
  `s(B,D,Tsep)/B=s(B,0,T0)/B`; `wrev=[u(B,D,Tsep)-u(B,0,T0)]/B` (не `Δf`).
  Составной endpoint решает `u(B,0,Tcycle)=u(B,D,Tsep)` и даёт
  `Δσcycle/B` и `xhot=wrev−T0Δσcycle`.
* **Отдельное `P=const`:** для каждого подготовленного `δ` решается
  `P(B,δB,T0)=P0`; выдаются `B/B0` и `V/V0=B0/B`. Это не смешивается с closed
  protocol и не объявляет свободное превращение `n↔p`.

Primary: `44 dps / 80 nodes / 4500 MeV`; refinement: `56 dps / 112 nodes /
5200 MeV`. Для коэффициентных FD заморожены `ΔT=0.1,0.05 MeV`,
`ΔB/B=10⁻³,5×10⁻⁴`; используются центральные 4-точечные stencils, без fit.
Локальная ветвь требует положительный `y`, оба species `n_i>0`, положительный
Hessian/`cv` и малый root residual. Интерфейсная энергия отброшена только в
макроскопическом пределе.

## 2. Численные результаты

| ссылка | модель | `B0` fm⁻³ | `S_F` MeV | `S_U` MeV | `cv` | `Θmix` MeV | `Θsep` MeV | `Aμ` | `AP` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0.90|W8|0.1021793165|37.02779621|3.446075979|1.745532407|1.974226297|19.23866902|0.563222741|-0.034309017|
|0.90|U16|0.1021797530|37.02780624|3.446092775|1.745532250|1.974236096|19.23866686|0.563214209|-0.034308618|
|0.93|W8|0.1128630447|37.20809977|3.733162592|1.735252506|2.151365625|19.29110435|0.471619214|-0.029723439|
|0.93|U16|0.1128630550|37.20809999|3.733162957|1.735252498|2.151365845|19.29110436|0.471618941|-0.029723423|

Здесь `S_F=B H_DD/2`, `S_U=S_F−T∂T S_F`, `cv=(∂T u)_(B,D=0)/B`,
`Θmix=S_U/cv`, `Θsep=T∂T S_F/cv`, `L_F=3B∂B S_F`, `K_T=9B H_BB`.
Живое тождество одного EOS:

```text
K_T (Aμ − AP) = 9 cv (Θmix + Θsep) = 9 S_F.
```

Максимальная относительная невязка этого тождества в JSON меньше `2.2×10⁻¹⁶`;
контакт действительно сдвигает `S_F,S_U` на `CρB/2`, но оставляет `∂T S_F`
и `Tsep` неизменными. Для 0.90/W8 с `Cρ=0` при `δ=+0.2`:
`Tmix=70.07896164`, `Tsep=70.77909460`, `Tcycle=70.85741258 MeV`,
`Δσmix/B=0.02129314`, `wiso=1.49059757 MeV`, `wrev=1.49811396 MeV`,
`xhot=0.00910374 MeV`, `B/B0(P)=0.99863098`, `V/V0=1.00137089`.
Для остальных endpoint чисел и обеих знаковых ветвей см. полный JSON; валидация
проверяет каждую строку, а не только этот пример.

Максимумы primary residuals: root `2.72×10⁻⁴¹`, energy ledger
`1.01×10⁻²¹`, Gibbs/pressure `1.41×10⁻⁴²`/`2.13×10⁻⁴¹`, work ledger
`1.80×10⁻²¹`, 8/12-point quadrature `5.48×10⁻²¹`, isobaric pressure
`2.88×10⁻⁴¹`; refinement precision max `2.11×10⁻²⁰`. Все величины получены из
численных states; validator повторно выводит identities из чисел и отвергает
мутацию, даже если флаг и integrity hash пересчитаны.

## 3. Контроль классической идеальной смеси

Обязательный null-control:
`Φ(δ)=[(1+δ)ln(1+δ)+(1−δ)ln(1−δ)]/2`.
Он даёт `S_F=T/2`, `S_U=0`, `cv=3/2`, `Tmix=T0`,
`wiso=TΦ`, `Δσmix=Φ`, `Tsep/T0=exp(Φ/cv)`, `Aμ=1/2`, `AP=0`.
При `δ=0.2`, например, `Φ=0.02013551355`. Это специально предотвращает
ошибку «энтропия смешения является доступным топливом»: у ideal gas изменение
`f` возникает из `−T s`, а не из положительного внутреннего энергоисточника.

## 4. Проверки контракта и артефакты

* `generator_project/calc_nvg_closed_mixing_bridge.py` — единственный live producer;
  использует maintained `nvg_thermal_observable_bridge.py`,
  `nvg_thermal_composition_hessian.py`, `nvg_nonlinear_calibration_response.py`
  только как read-only constructors/integrals.
* `generator_project/test_nvg_closed_mixing_bridge.py` — 6 focused tests; один
  `build_result()` на class setup, затем mutation tests и numeric validator.
* `generator_project/nvg_closed_mixing_bridge_results.json` — 72 primary rows,
  72 primary isobaric rows, 4 references, refinement tree, source hashes;
  `status=PASS_LIVE_CLOSED_MIXING_BRIDGE`, integrity
  `0ab8203828b91da733d0fea47a8b59e6a893295f53cf01be63f38029e2440052`.
* Команды: `PYTHONDONTWRITEBYTECODE=1 .venv-research/bin/python3.12 -m pytest -p no:cacheprovider -q generator_project/test_nvg_closed_mixing_bridge.py`
  → `6 passed in 725.63s`; `... calc_nvg_closed_mixing_bridge.py --validate ...` → `PASS`.
* Scope audit: не читались и не импортировались `evidence_parent` numerics/oracle;
  нет `τ`, diffusivity, rate, power, nucleus/star/BH/bounce или «universal theory»
  claims. `SOURCES.md` run остаётся sealed read-only.

## 5. Источники (контекст, не NVG-валидация)

Список первичных источников зафиксирован в
`Lunacy/runs/nvg-closed-response-adhd-2026-09-28/SOURCES.md`:

1. [nucl-th/0609035](https://arxiv.org/abs/nucl-th/0609035) — finite-T symmetry
   energy/free-energy distinction.
2. [arXiv:1308.5527](https://arxiv.org/abs/1308.5527) — warm nuclear systems,
   finite-T composition/free/internal-energy coefficients.
3. [arXiv:1504.00177](https://arxiv.org/abs/1504.00177) — finite-T asymmetric
   nuclear matter from chiral EFT.
4. [arXiv:1510.06417](https://arxiv.org/abs/1510.06417) — static response of
   neutron matter; future finite-q context, not used as current data.

## 6. Полный ADHD pool (30 идей; N/V/F и score)

| id | frame/cluster | идея | N/V/F | score |
|---|---|---|---:|---:|
|C1|competitor/spatial|Совместный и встречный импульс потоков|7/5/8|6.45|
|C2|competitor/ensemble|Условные флуктуации конечного объёма|6/6/8|6.50|
|C3|competitor/ensemble|Кривая состава при постоянном давлении|7/9/10|8.55|
|C4|competitor/cycle|Замкнутый цикл химических потенциалов|4/9/6|6.50|
|C5|competitor/dynamic|Кодированное возбуждение и отделение дрейфа|7/3/7|5.40|
|C6|competitor/dynamic|Частотная вторая гармоника и её фаза|8/3/9|6.25|
|L1|logistics/cycle|Компенсационный буфер в цикле состава|6/5/7|5.85|
|L2|logistics/spatial|Чередующиеся карманы состава и краевой поток|7/5/8|6.45|
|L3|logistics/ensemble|Двойной клапан теплового ансамбля|5/5/8|5.75|
|L4|logistics/cycle|Раздельный учёт переноса частиц и химической работы|6/9/8|7.70|
|L5|logistics/spatial|Два пространственных маршрута одинакового дисбаланса|7/7/8|7.25|
|L6|logistics/spatial|Три ячейки с замкнутым балансом частиц|7/7/8|7.25|
|X1|child/dynamic|Квазистатическое выпрямление изовекторной тряски|6/6/8|6.50|
|X2|child/caloric|Смешение двух противоположных составов в закрытом ящике|8/9/9|8.65|
|X3|child/ensemble|Две камеры, обмен составом и поршень|7/8/8|7.65|
|X4|child/cycle|Изоспиновая батарейка с учётом источника|4/9/7|6.75|
|X5|child/cycle|Разность поршневой работы по двум маршрутам|7/7/8|7.25|
|X6|child/phase|Исчезновение макроскопических границ состава|7/6/8|6.85|
|I1|inversion/phase|Комплексная каустика и радиус сходимости|8/3/6|5.50|
|I2|inversion/phase|Выпуклая оболочка и фракционировка двух фаз|7/5/9|6.70|
|I3|inversion/spatial|Поворот мягкой моды при конечном q|7/7/8|7.25|
|I4|inversion/dynamic|Спектральные суммы без выбора столкновений|8/3/8|6.00|
|I5|inversion/phase|Контур смены знака композиционного сжатия|7/6/8|6.85|
|I6|inversion/phase|Конечный порог изовекторного источника|7/5/8|6.45|
|Z1|zero/spatial|Радиальный хвост локального источника|7/6/8|6.85|
|Z2|zero/spatial|Различить нулевую и конечноволновую неустойчивости|6/7/8|6.90|
|Z3|zero/ensemble|Знак кросс-ёмкости около симметрии|3/9/6|6.15|
|Z4|zero/phase|Продолжить ветвь до разворота источника|7/5/8|6.45|
|Z5|zero/caloric|Обратимый нагрев при изменении состава без теплообмена|7/9/9|8.30|
|Z6|zero/spatial|Поляризация первой мягкой моды|6/7/8|6.90|

Оркестрация по sealed contract: 5 изолированных frame, максимум 3 параллельных
ветки; при ограничении ёмкости применена схема 3+2. Это именно генерация
гипотез, не список принятых физических утверждений.

## 7. Три focus branches

1. **★ X2 (8.65), closed isochoric mixing.** Почему: наиболее прямой endpoint,
   не требует нового coupling и отличает `u` от `f`. Эскиз: `±D` при общем `B`,
   solve `Tmix`, entropy/energy ledger, ideal-gas null. Риск: перепутать
   entropy/free energy с доступной энергией. Первый шаг: живой canonical root и
   `u(B,0,Tmix)=u(B,D,T0)`. Child ideas: (i) unequal-volume weighting;
   (ii) finite interface correction in macroscopic limit; (iii) two-stage
   mixing followed by entropy audit. Провокация: «закрытое смешение само не
   создаёт работу — покажите ledger».
2. **★ C3 (8.55), fixed-pressure composition curve.** Почему: независимая
   ensemble-проверка и чувствительность к contact shift; не продолжает старый
   fixed-`μB` coefficient. Эскиз: solve `P(B,δB,T0)=P0`, report `B/B0,V/V0`.
   Риск: незаметно заменить fixed-P на fixed-`μB` или объявить `n↔p` free
   conversion. Первый шаг: pressure root с тем же EOS и frozen `Cρ`. Child ideas:
   (i) pressure hysteresis diagnostic (не dynamics); (ii) enthalpy-per-B
   comparison; (iii) constrained `P,T,δ` response. Провокация: «тот же A при
   другом ансамбле — почти наверно ошибка constraint».
3. **★ Z5 (8.30), reversible isentropic composition change.** Почему: даёт
   caloric endpoint и separates reversible `Δu` from isothermal `Δf`. Эскиз:
   solve `s/B` at fixed B, then compose endpoint at fixed energy. Риск: назвать
   `wrev` устройственной мощностью или считать цикл reversible after irreversible
   mix. Первый шаг: entropy root plus `Δσcycle≥0`, `xhot` bookkeeping. Child
   ideas: (i) signed `D` parity; (ii) bath-temperature exergy scan; (iii)
   constrained `s/B` continuation near convexity boundary. Провокация:
   «обратимость — это path constraint, не красивое слово в отчёте».

## 8. Ограничения и следующий честный шаг

Нет утверждения о глобальном minimum/phase coexistence: только bounded local branch.
Нет transition rate, diffusivity, relaxation, power, household generator, nucleus,
star, black hole, bounce или universal confirmation. Следующий шаг может быть только
новым независимым физическим вопросом (например, finite-q observable) после отдельного
решения; текущие коэффициенты и старые accepted/private files frozen.
