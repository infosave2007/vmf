# Условный вихревой сектор source-complete действия

## Главное

В исходном действии действительно есть локальное абелево-хиггсовское ядро.
Если рассматривать **только** source-free классический сектор

\[
N=J_B=A_0=0,
\]

и убрать неописанные spectator-поля, то его безразмерное отношение масс
однозначно вычисляется. Оно оказывается type II, а не BPS:

\[
\beta\equiv{m_\sigma^2\over m_A^2}
={2\lambda\over q_\phi^2}=2.5300315402\ldots>1,
\]

\[
\kappa_{\rm GL}={m_\sigma\over\sqrt2m_A}
=1.1247291986\ldots>{1\over\sqrt2}.
\]

Это интересное **математическое направление**, но не открытие нового
физического объекта: текущий контракт задаёт локальную \(U(1)\)-форму, но не
декларирует компактность группы, решётку зарядов или large gauge
transformations. Поэтому нельзя честно объявлять топологически защищённую
струну, сверхпроводник, космическую струну, вихрь нейтронной звезды или
источник энергии.

Исполняемые артефакты:

- `/Users/oleg/Documents/NVG-Research/verification/nvg_source_complete_conditional_vortex_sector_audit.py`;
- `/Users/oleg/Documents/NVG-Research/verification/nvg_source_complete_conditional_vortex_sector_results.json`;
- `/Users/oleg/Documents/NVG-Research/verification/test_nvg_source_complete_conditional_vortex_sector_audit.py`.

## Вывод без подгонки

Из уже фиксированных величин

\[
q_\phi={m_\omega\over W_0}=0.91105937136\ldots,
\qquad m_A=q_\phi W_0=782.6\ \mathrm{MeV},
\]

\[
m_\sigma^2=2\lambda W_0^2,
\qquad m_\sigma=1244.80926250\ldots\ \mathrm{MeV}.
\]

Важно: здесь применяется \(q_\phi\), а не \(g_\omega\). Последняя
величина — отдельная связь материального тока и не является хиггсовским
зарядом в этом действии.

BPS-граница в нормировке этого действия:

\[
\lambda_{\rm BPS}={q_\phi^2\over2}
=0.41501458907\ldots .
\]

Здесь есть точная связь с уже найденной кросс-масштабной картой: та же
величина равна

\[
\lambda_{\rm BPS}={m_\omega^2\over2W_0^2},
\]

то есть точке \(m_\sigma=m_A\), где в старом tree/static ядре меняется
порядок скалярной и векторной дальностей. Это общая алгебра одного действия,
а не новое экспериментальное предсказание и не причина менять \(\lambda\).

При живом \(\lambda=1.05\)

\[
\lambda-\lambda_{\rm BPS}>0,
\quad \beta-1>0,
\quad \kappa_{\rm GL}^2-{1\over2}>0.
\]

Все три неравенства хранятся в JSON как точные рациональные сертификаты.
Ни данные материала, ни критическое поле, ни ширина, ни экспериментальная
цель не читаются.

## Что было бы нужно для классического вихря

Только при дополнительном, **пока не записанном в контракте**, глобальном
допущении о компактной \(U(1)\) с разрешённым целым winding \(n\) можно
рассматривать стандартный конеч-энергетический ansatz на поперечной
\(\mathbb R^2\):

\[
W=W_0f(r),\qquad \theta=n\varphi,
\qquad A_\varphi={n\over q_\phi}a(r),
\]

\[
f(0)=a(0)=0,\qquad f(\infty)=a(\infty)=1.
\]

В координате \(x=m_A r\) локальное действие даёт BVP

\[
f''+{f'\over x}-{n^2(1-a)^2\over x^2}f
-{\beta\over2}f(f^2-1)=0,
\]

\[
a''-{a'\over x}+f^2(1-a)=0.
\]

При таком **добавочном** global completion поток был бы
\(\Phi_B=2\pi n/q_\phi\). Формула BPS-напряжения
\(T_n=\pi W_0^2|n|\) справедлива только на
\(\lambda=\lambda_{\rm BPS}\), а потому не применяется к живому baseline.
Скрипт намеренно не публикует численный профиль или tension для baseline.

## Почему это не противоречит прежнему no-go

Ранее полученная расходимость \(W^{-2}\) относится к однородной плотной
материи после исключения \(A_0\) при ненулевом \(g_\omega n_B\). В условном
source-free вихревом усечении \(n_B=A_0=0\), поэтому тот однородный аргумент
нельзя подставлять в сердцевину вихря по точкам. И наоборот: source-free BVP
ничего не доказывает для вихря в барионной среде — для неё нужны полные
неоднородные уравнения материи, скаляра и Максвелла.

## Чего результат не доказывает

Он не доказывает:

- существование глобально топологического или квантово стабильного объекта;
- реальный сверхпроводник, решётку Абрикосова, космические струны или
  нейтронно-звёздные вихри;
- гравитационное действие, космологическую численность или наблюдаемый поток;
- генератор, свободную энергию, тёмную материю либо экспериментальное
  предсказание.

Следующий корректный шаг — сначала явно задать global completion и все
поля действия, затем отдельно решить BVP с заранее объявленными сеточными,
граничными, residual и energy-balance проверками. Подгонять \(\lambda\) к
BPS или выдавать условную формулу потока за физический результат нельзя.

## Воспроизведение

```bash
cd /Users/oleg/Documents/NVG-Research
PYTHONPATH=verification python3 -B \
  verification/nvg_source_complete_conditional_vortex_sector_audit.py
PYTHONPATH=verification python3 -B \
  verification/nvg_source_complete_conditional_vortex_sector_audit.py \
  --validate verification/nvg_source_complete_conditional_vortex_sector_results.json
PYTHONPATH=verification python3 -B -m pytest -q -p no:cacheprovider \
  verification/test_nvg_source_complete_conditional_vortex_sector_audit.py
```
