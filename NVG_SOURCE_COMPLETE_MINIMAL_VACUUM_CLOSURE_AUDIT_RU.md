# Кинематическое замыкание минимального вакуумного сектора

## Результат простыми словами

Из уже зафиксированного source-complete действия следует один новый, но
**узкий** факт: в его минимальном вакуумном наборе полей нет открытого
двухчастичного распада для радиального скаляра \(\sigma\) и массивного
вектора \(A\) на tree level.

Это не означает, что найдена стабильная реальная частица, тёмная материя,
высокий \(Q\), источник энергии или устройство. Это означает только: из
показанной части действия нельзя честно приписать этим модам внутреннюю
вакуумную ширину через перечисленные tree-level каналы.

Исполняемые артефакты:

- `/Users/oleg/Documents/NVG-Research/verification/nvg_source_complete_minimal_vacuum_closure_audit.py`;
- `/Users/oleg/Documents/NVG-Research/verification/nvg_source_complete_minimal_vacuum_closure_results.json`;
- `/Users/oleg/Documents/NVG-Research/verification/test_nvg_source_complete_minimal_vacuum_closure_audit.py`.

## Что выводится из действия

В source-free, gauge-neutral вакууме выбирается unitary gauge

\[
W=W_0+\sigma,\qquad \theta=0,\qquad A_\mu=0.
\]

Из того же действия, без новой константы, следуют

\[
m_A=q_\phi W_0=m_\omega=782.6\ \mathrm{MeV},
\qquad
m_\sigma^2=2\lambda W_0^2,
\]

\[
m_\sigma=1244.80926249767\ldots\ \mathrm{MeV}.
\]

В разложении присутствуют нужные для проверки вершины

\[
-g_s\sigma\bar NN,qquad
-g_\omega A_\mu\bar N\gamma^\mu N,
\qquad {m_A^2\over W_0}\sigma A_\mu A^\mu,
\qquad -\lambda W_0\sigma^3.
\]

Здесь \(\theta\) — поглощённый Goldstone, а \(A_0\) — Gauss-ограничение;
они не считаются отдельными конечными частицами. Неопределённое
`L_spectator` также не подставляется вместо недостающей физики.

## Три точных пороговых сертификата

Код сохраняет рациональные числители и знаменатели, поэтому знаки не
получены округлением. Для живых параметров:

\[
4M_N^2-m_A^2=2\,914\,421.24\ \mathrm{MeV}^2>0,
\]

\[
4M_N^2-m_\sigma^2=1\,977\,333.9\ \mathrm{MeV}^2>0,
\]

\[
4m_A^2-m_\sigma^2=900\,300.94\ \mathrm{MeV}^2>0.
\]

То есть строго

\[
m_A<M_N<m_\sigma<2m_A<2M_N.
\]

Поэтому фазовое пространство закрыто для ровно тех физических
двухчастичных срезов, которые создают показанные вершины:

\[
\sigma\to N\bar N,\qquad A\to N\bar N,\qquad \sigma\to AA.
\]

Во всех трёх случаях соответствующий \(\Theta\)-множитель равен нулю. Кубическая
скалярная вершина не меняет этот вывод: \(\sigma\to\sigma\sigma\) невозможен
при положительной \(m_\sigma\), поскольку \(m_\sigma<2m_\sigma\).

Для полноты скрипт также закрывает crossed-процессы тех же вершин:
\(N\to N+\sigma\), \(N\to N+A\), \(A\to A+\sigma\) и
\(\sigma\to\sigma+\sigma\). Они невозможны уже потому, что к массе
родителя добавляется строго положительная масса излучаемой моды; строка для
\(N\) одновременно представляет процесс для \(\bar N\). Поэтому в
результате нет скрыто пропущенного 1→2 crossing внутри **перечисленного**
минимального набора вершин.

## Полезная карта параметра — не рекомендация его менять

Если только как алгебраическую диагностику заменить \(\lambda\) на
гипотетическое \(\lambda_d\), оставив остальные параметры фиксированными,
то пороги открытия были бы

\[
\sigma\to AA:\quad \lambda_d>2q_\phi^2
=1.66005835629\ldots,
\]

\[
\sigma\to N\bar N:\quad \lambda_d>2g_s^2
=2.38987316383\ldots.
\]

Живое \(\lambda=1.05\) ниже обоих. Это **не** фит и не предложение менять
действие: изменение \(\lambda\) меняет сам скалярный потенциал и массу.

## Что остаётся неизвестным

Расчёт не включает и не исключает:

- loop-эффекты и сдвиги физических полюсов;
- неизвестные spectator-поля и любые новые операторы;
- гравитационные, плотностные, температурные или внешне-накачиваемые потери;
- экспериментальную идентификацию \(A\) или \(\sigma\);
- время жизни, linewidth, \(Q\), энергию, усиление либо генератор.

Следующий честный шаг, если появится полное дополнение к действию, — записать
его поля и вершины, затем независимо пересчитать все разрешённые срезы и
спектральную функцию. Нельзя заменять отсутствующий сектор произвольной
таблицей ширин или желаемым \(Q\).

## Воспроизведение

```bash
cd /Users/oleg/Documents/NVG-Research
PYTHONPATH=verification python3 -B \
  verification/nvg_source_complete_minimal_vacuum_closure_audit.py
PYTHONPATH=verification python3 -B \
  verification/nvg_source_complete_minimal_vacuum_closure_audit.py \
  --validate verification/nvg_source_complete_minimal_vacuum_closure_results.json
PYTHONPATH=verification python3 -B -m pytest -q -p no:cacheprovider \
  verification/test_nvg_source_complete_minimal_vacuum_closure_audit.py
```
