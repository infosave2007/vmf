# Масштабное тождество и три различные плотности — 2026-09-08

Исследовательская проверка исходного функционала из `SOURCE_COMPLETE_MODEL_RU.md` и `.docs/SOURCE_COMPLETE_RESPONSE_AUDIT.md`, без изменения параметров и производственных файлов. Все выводы относятся к однородной холодной среде с фиксированным составом, постоянными безразмерными связями, массой `m=gs W`, векторным членом `G n²/(2W²)` и без вакуумных фермионных петель. Показанные разряды — вычислительная точность при фиксированных десятичных входах, не физическая точность модели.

## Общий вывод для произвольного U(W)

Пусть `H(n,W)=F_F(n,gs W)+G n²/(2W²)`. В трёх пространственных измерениях `H(a³n,aW)=a⁴H(n,W)` при положительном `a`; это непосредственно следует из замены импульса в ферми-интеграле. Теорема Эйлера даёт `3n H_n+W H_W=4H`. Для `E=H+U`, на гладкой стационарной ветви `E_W=0`, с `ε=E`, `μ=E_n`, `P=nμ−ε` получаем

\[
T\equiv\varepsilon-3P=4\varepsilon-3n\mu=4U-WU'.
\]

Это общий результат именно указанного класса функционалов, а не произвольной EOS. Постоянный сдвиг потенциала на `b` сдвигает `T` на `4b`; ниже используется заданный нуль `U(W0)=0`. Скалярная стационарность нужна для тождества следа; локальная устойчивость не нужна для самого алгебраического равенства, но нужна для физического использования ветви. Для производных далее необходимы `n>0`, `W>0`, `C=E_WW≠0`, `μ≠0` и гладкость ветви. В проверяемой устойчивой ветви `C>0`, `μ>0`.

Обозначим `A=U''−3U'/W`, `B=E_nW`, `D=E_nn`. Дифференцирование масштабного равенства по `n` и по `W` соответственно даёт

\[
3nD+WB=\mu,\qquad C=A-\frac{3nB}{W}.
\]

Поскольку `W'=−B/C`, `T'_W=3U'−WU''=−WA`, имеем

\[
\frac{dT}{dn}=\frac{WAB}{C}=\mu(1-3c_s^2),\qquad
c_s^2=\frac13\left(1-\frac{WAB}{\mu C}\right).
\]

Таким образом, в любой гладкой точке `B=0` такого функционала `c_s²=1/3`, независимо от формы `U`. Обратное заключение в общем случае требует `A≠0`; при `A=0` тоже возможно `c_s²=1/3`. На фазовой границе с разрывом или смешанной фазой эти производные нельзя переносить без отдельного анализа.

## Что специально для принятого квартичного потенциала

Для `U=λ(W²−W0²)²/4`, `λ>0`, `W0>0`:

\[
A=2\lambda W_0^2,\quad
T=\lambda W_0^2(W_0^2-W^2),\quad
c_s^2=\frac13\left(1-\frac{2\lambda W_0^2WB}{\mu C}\right)
=\frac13\left(1+\frac{2\lambda W_0^2W}{\mu}\frac{dW}{dn}\right).
\]

На ветви с `C>0`, `μ>0` знак `c_s²−1/3` равен знаку `dW/dn`. В минимуме `W` след `T` имеет максимум и `c_s²=1/3`; нулевой след требует `W=W0` и не требует `c_s²=1/3`. Ни равенство скорости в одной точке, ни нулевой след в одной точке не доказывают конформную симметрию теории.

Независимые численные ориентиры при исходных параметрах (`x=n/n0`):

| Событие | x | W, MeV | c_s² |
|---|---:|---:|---:|
| Минимум W, максимум T | 2.14102631011896052803864901227 | 858.082916153435188666429222638 | 1/3 |
| Возврат W=W0 при n>0, T=0 | 4.31093715562408078768016377634 | 859 | 0.490849927813870691007443928959 |
| Локальный максимум c_s², формальное продолжение | 15.0243471375003253943302815194 | 885.323752540972958009535088519 | 0.667157319386440629764789063279 |

В последней точке `d²c_s²/dx²=−0.0012186260838331332032262144342`. Эта точка находится **вне заявленной области исследования 0≤x≤10**. Код ищет локальный максимум и проверяет сетку расширенного диапазона; это не интервальное доказательство глобального максимума или единственности всех стационарных ветвей. Плотности не получены подгонкой. Несоответствие исходной модели ядерному насыщению (`P(n0)=18.7236 MeV/fm³`, связь `+124.3175 MeV`) сохраняется.

## Формальная асимптотика, не область установленной физической применимости

При `n→∞` положим `W=a n^(1/3)`. Ведущая энергия есть `n^(4/3) h(a)`, где

\[
h(a)=F_F(1,g_sa)+\frac\lambda4a^4+\frac{G}{2a^2},\qquad
h'(a)=\lambda a^3+g_s n_s(1,g_sa)-G/a^3=0.
\]

Здесь `F_F(1,g_s a)` и `n_s(1,g_s a)` — безразмерные интегралы после масштабирования импульса на `n^(1/3)`. Положительная кривизна `h''` следует из положительных фермионного, квартичного и векторного вкладов; `h'` меняет знак от отрицательного при `a→0+` к положительному при `a→∞`, поэтому ведущий минимум единствен. Численно `a=2.18530377508240050327601700389`. Разложение около этого невырожденного минимума даёт

\[
\begin{aligned}
W&=a n^{1/3}[1+O(n^{-2/3})],\\
\varepsilon&=h(a)n^{4/3}-\tfrac12\lambda W_0^2a^2n^{2/3}+O(1),\\
T&=-\lambda W_0^2a^2n^{2/3}+O(1),\\
\frac{T}{\varepsilon}&\to0^- ,\qquad
c_s^2=\frac13+\frac{\lambda W_0^2a^2}{6h(a)}n^{-2/3}+O(n^{-4/3}).
\end{aligned}
\]

Абсолютный след не стремится к нулю: он растёт по модулю отрицательным как `n^(2/3)`, тогда как нормированный след стремится к нулю. Это формальная экстраполяция данного функционала, не вывод о реальном веществе, КХД, новых степенях свободы или пригодности EOS для нейтронных звёзд.

## Известные результаты и исследовательская ценность

Форма `4V−χV'` уже приведена для однородной барионной среды в уравнении (10) первичного авторского [препринта H. K. Lee, 2024](https://www.preprints.org/manuscript/202403.1812/v1/download); версия прямо помечена как не прошедшая рецензирование. Более общий термодинамический разбор связи следа и скорости звука, включая различие точек `c_s²=1/3` и `ε−3P=0`, имеется у [Fujimoto et al., 2022, уравнения (4)–(6) и (9)](https://arxiv.org/pdf/2207.06753). Поэтому здесь не заявляется новая универсальная формула или приоритет; независимо выведены её специализация и численные следствия для фиксированного NVG-функционала. Его классическое нарушение масштабной симметрии нельзя без согласования отождествить с квантовой аномалией следа КХД.

Первый практический шаг уже выполнен: сравнить вычисленные независимо `ε−3P` и `λW0²(W0²−W²)`, а затем две формулы скорости звука и производную следа. Основной риск интерпретации — принять совпадение внутренних тождеств за подтверждение ядерной физики. Направления продолжения: (1) проверка остатка следа при каждом изменении EOS; (2) раздельное сравнение трёх плотностей; (3) явный учёт новых масштабных членов через `4U−WU'`; (4) реконструкция поля `W²=W0²−T/(λW0²)` из независимо заданной EOS как необходимая проверка совместимости, без скрытой смены параметров.

## Воспроизводимый независимый расчёт

Из корня проекта: `awk '/^```python$/{run=1;next}/^```$/{if(run)exit}run' scratch/unification-scale-check-2026-09-08.md | .venv-research/bin/python`. Скрипт ничего не записывает; все параметры и решения создаются заново внутри контекста точности. Он не импортирует производственные EOS-функции.

Выполнение 8 сентября 2026 завершилось с кодом 0: совпадение 80/120 разрядов до 28 значащих цифр для возвращаемых ориентиров, все тождества с нормированным допуском `10^-60`, 201 точка логарифмической сетки `0.001≤x≤100`, отрицательная вторая производная в найденном максимуме и сходимость двух асимптотических отношений. При `x=10^12` отношение `W/(a n^(1/3))=1.00000002149058449`, отношение вычисленной поправки к `c_s²` к ведущей асимптотической поправке — `1.00000003672611783`. Эти огромные плотности используются исключительно для проверки формальной асимптотики.

```python
import mpmath as mp


def calculate(dps, scan=False):
    with mp.workdps(dps):
        W0, lam, MN, hb, go, mo = map(mp.mpf,
            ('859', '1.05', '939', '197.3269804', '10.12', '782.6'))
        n0 = mp.mpf('0.16') * hb**3
        gs, G, third = MN / W0, (go * W0 / mo)**2, mp.mpf(1) / 3
        tol = mp.mpf('1e-60')

        def k(n):
            return (3 * mp.pi**2 * n / 2)**third

        def ns(n, W):
            p, mass = k(n), gs * W
            EF = mp.sqrt(p*p + mass*mass)
            return mass / mp.pi**2 * (p*EF - mass**2 * mp.asinh(p/mass))

        def free(n, W):
            p, mass = k(n), gs * W
            EF = mp.sqrt(p*p + mass*mass)
            z = mp.asinh(p/mass)
            e = (p*EF*(2*p*p + mass*mass) - mass**4*z) / (4*mp.pi**2)
            pfree = (p*EF*(2*p*p - 3*mass*mass) + 3*mass**4*z) / (12*mp.pi**2)
            return e, pfree

        def f(n, W):
            return lam*W*(W*W-W0*W0) + gs*ns(n, W) - G*n*n/W**3

        def mixed(n, W):
            return gs**2*W/mp.sqrt(k(n)**2+(gs*W)**2) - 2*G*n/W**3

        def branch(x):
            n = x*n0
            guess = max(W0, (G/lam)**(third/2)*n**third)
            return mp.findroot(lambda W: f(n, W), (guess, guess*mp.mpf('1.01')))

        def response(x, check=False):
            n, W = x*n0, branch(x)
            EF = mp.sqrt(k(n)**2+(gs*W)**2)
            C = mp.diff(lambda w: f(n, w), W)
            B = mixed(n, W)
            D, mu = k(n)**2/(3*n*EF)+G/W**2, EF+G*n/W**2
            ef, pf = free(n, W)
            U, V = lam*(W*W-W0*W0)**2/4, G*n*n/(2*W*W)
            energy, pressure = ef+U+V, pf-U+V
            cs2 = n*(D-B*B/C)/mu
            trace = lam*W0**2*(W0**2-W**2)
            if check:
                assert abs(f(n, W))/max(W0**3, n) < tol
                assert abs((energy-3*pressure)-trace)/energy < tol
                assert abs(energy+pressure-n*mu)/energy < tol
                assert abs(3*n*D+W*B-mu)/mu < tol
                assert abs(C-(2*lam*W0**2-3*n*B/W))/C < tol
                assert abs(cs2-(1-2*lam*W0**2*W*B/(mu*C))/3) < tol
                assert C > 0 and mu > 0
            return W, cs2, trace, energy, pressure, C, mu

        xmin, wmin = mp.findroot(lambda x, W: (f(x*n0, W), mixed(x*n0, W)),
                                (mp.mpf('2'), W0))
        xreturn = mp.findroot(lambda x: f(x*n0, W0), (mp.mpf('4'), mp.mpf('5')))
        xmax = mp.findroot(lambda x: mp.diff(lambda t: response(t)[1], x),
                          (mp.mpf('14'), mp.mpf('16')))
        rows = []
        print('PRECISION', dps)
        for name, x in [('minimum', xmin), ('return_W0', xreturn), ('local_cs2_max', xmax)]:
            row = (x,) + response(x, check=True)
            rows.append(row)
            print(name, [mp.nstr(v, 32) for v in row])
            # Differentiate the independently evaluated energy/pressure trace.
            dtrace_dx = mp.diff(lambda t: response(t)[3]-3*response(t)[4], x)
            expected = n0*row[-1]*(1-3*row[2])
            assert abs(dtrace_dx-expected)/(n0*row[-1]) < tol
        assert abs(rows[0][2]-third) < tol
        assert abs(rows[1][1]-W0) < tol
        assert abs(mp.diff(lambda x: response(x)[1], xmax)) < tol
        second = mp.diff(lambda x: response(x)[1], xmax, 2)
        assert second < 0 and mp.diff(branch, xmin, 2) > 0
        print('MAX_SECOND_DERIVATIVE', mp.nstr(second, 32))
        if scan:
            grid = [mp.power(10, -3+mp.mpf(i)/40) for i in range(201)]
            values = [response(x, check=True) for x in grid]
            assert all(0 < row[1] < 1 for row in values)
            assert max(row[1] for row in values) < rows[2][2]
            print('LOG_GRID_0.001_TO_100', len(grid), 'passed; not an interval proof')

        a = mp.findroot(lambda w: lam*w**3+gs*ns(mp.mpf(1), w)-G/w**3,
                        (mp.mpf('2'), mp.mpf('3')))
        H = free(mp.mpf(1), a)[0]+lam*a**4/4+G/(2*a*a)
        coefficient = lam*W0**2*a*a/(6*H)
        print('FORMAL_ASYMPTOTIC_a_h_cs2_coefficient', [mp.nstr(v, 32) for v in (a,H,coefficient)])
        errors = []
        for x in map(mp.mpf, ('1e6', '1e12')):
            row = response(x, check=True)
            ratio_W = row[0]/(a*(x*n0)**third)
            ratio_sound = (row[1]-third)*(x*n0)**(2*third)/coefficient
            assert row[2] < 0 and row[1] > third
            errors.append(abs(ratio_W-1)+abs(ratio_sound-1))
            print('FORMAL_EXTRAPOLATION', mp.nstr(x, 8), mp.nstr(ratio_W, 24), mp.nstr(ratio_sound,24))
        assert errors[1] < errors[0]
        # Format within the precision context; never compare values rounded later.
        return tuple(mp.nstr(v, 28) for row in rows for v in row)+(mp.nstr(a,28),mp.nstr(second,28))


before = mp.mp.dps
r80 = calculate(80, scan=True)
r120 = calculate(120)
assert mp.mp.dps == before
assert r80 == r120
print('80_120_DIGIT_AGREEMENT_TO_28_SIGNIFICANT_DIGITS', True)
```
