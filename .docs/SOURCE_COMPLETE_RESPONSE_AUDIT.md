# Независимый расчёт отклика source_complete — 2026-09-08

Расчёт относится к уравнениям из [описания модели](../SOURCE_COMPLETE_MODEL_RU.md). Он использует только `mpmath` и десятичные входы модели. Производственные функции, сохранённые EOS/звёздные таблицы, результат TOV, подбор параметров и внешние значения радиусов не используются. Разница между точными десятичными входами здесь и их binary64-представлением в рабочем коде несущественна для показанных 9–12 значащих цифр; это не требование побитового совпадения с производственным JSON.

Для воспроизведения из корня репозитория (вывод только в терминал):

```sh
awk '/^```python$/{run=1;next}/^```$/{if(run)exit}run' .docs/SOURCE_COMPLETE_RESPONSE_AUDIT.md | .venv-research/bin/python
```

Скрипт повторно создаёт все константы и решения при 80 и 120 разрядах; контекст точности восстанавливается после каждого вызова. Обычное время работы — несколько секунд. Это не полный звёздный прогон и не интервальная сертификация непрерывной ветви.

```python
import mpmath as mp


def calculate(dps, scan=False):
    with mp.workdps(dps):
        W0, lam, MN = map(mp.mpf, ("859", "1.05", "939"))
        hb, go, mo = map(mp.mpf, ("197.3269804", "10.12", "782.6"))
        n0 = mp.mpf("0.16") * hb**3
        d = 4
        gs, q = MN / W0, mo / W0
        G = go**2 / q**2

        def momentum(n):
            return (6 * mp.pi**2 * n / d)**(mp.mpf(1) / 3)

        def scalar_density(n, W):
            mass, k = gs * W, momentum(n)
            EF = mp.sqrt(k*k + mass*mass)
            return d * mass / (4 * mp.pi**2) * (
                k * EF - mass*mass * mp.asinh(k / mass)
            )

        def stationarity(n, W):
            return (lam * W * (W*W - W0*W0)
                    + gs * scalar_density(n, W) - G*n*n / W**3)

        def mixed(n, W):
            EF = mp.sqrt(momentum(n)**2 + (gs * W)**2)
            return gs**2 * W / EF - 2*G*n / W**3

        def branch(x):
            return mp.findroot(
                lambda W: stationarity(x * n0, W),
                (W0 * mp.mpf("0.99"), W0 * mp.mpf("1.01")),
            )

        def response(x):
            n, W = x * n0, branch(x)
            mass, k = gs * W, momentum(n)
            EF = mp.sqrt(k*k + mass*mass)
            a = mp.asinh(k / mass)
            e_free = d / (16*mp.pi**2) * (
                k*EF*(2*k*k + mass*mass) - mass**4*a
            )
            p_free = d / (48*mp.pi**2) * (
                k*EF*(2*k*k - 3*mass*mass) + 3*mass**4*a
            )
            potential = lam*(W*W - W0*W0)**2 / 4
            vector = G*n*n / (2*W*W)
            energy = e_free + potential + vector
            pressure = p_free - potential + vector
            mu = EF + G*n / W**2
            # Independent differentiation of the exact stationary equation.
            C = mp.diff(lambda w: stationarity(n, w), W)
            B = mixed(n, W)
            D = k*k / (3*n*EF) + G / W**2
            relaxed = D - B*B / C
            assert abs(energy + pressure - n*mu) / energy < mp.mpf("1e-50")
            assert abs(C - (2*lam*W0**2 - 3*n*B/W)) / C < mp.mpf("1e-50")
            return {
                "W_MeV": W, "W_over_W0": W/W0,
                "epsilon_MeV_fm3": energy/hb**3,
                "P_MeV_fm3": pressure/hb**3, "mu_MeV": mu,
                "cs2": n*relaxed/mu, "C_MeV2": C,
                "dW_dx_MeV": -n0*B/C,
                "binding_MeV": energy/n - MN,
                "frozen_W_cs2": n*D/mu,
                "scalar_mass_shift_MeV": mass - MN,
                "fermi_kinetic_MeV": e_free/n - mass,
                "scalar_potential_per_baryon_MeV": potential/n,
                "vector_energy_per_baryon_MeV": vector/n,
            }

        def shown(row):
            return {key: mp.nstr(value, 24) for key, value in row.items()}

        print("PRECISION", dps)
        rows = {}
        for text in ("0.001", "0.1", "1", "2.5", "5", "10"):
            rows[text] = response(mp.mpf(text))
            print("ROW", text, shown(rows[text]))
        xmin, wmin = mp.findroot(
            lambda x, W: (stationarity(x*n0, W), mixed(x*n0, W)),
            (mp.mpf("2"), W0),
        )
        minimum = {
            "x": xmin, "W_MeV": wmin, "W_over_W0": wmin/W0,
            "C_MeV2": mp.diff(lambda W: stationarity(xmin*n0, W), wmin),
            "d2W_dx2_MeV": mp.diff(branch, xmin, 2),
        }
        assert minimum["d2W_dx2_MeV"] > 0
        assert abs(stationarity(xmin*n0, wmin)) / W0**3 < mp.mpf("1e-50")
        assert abs(mixed(xmin*n0, wmin)) < mp.mpf("1e-50")
        print("MINIMUM", shown(minimum))
        xcheck = mp.mpf("2.5")
        mu_derivative = mp.diff(lambda x: response(x)["mu_MeV"], xcheck)
        identity_derivative = rows["2.5"]["cs2"] * rows["2.5"]["mu_MeV"] / xcheck
        assert abs(mu_derivative - identity_derivative) < mp.mpf("1e-50")
        print("THERMODYNAMIC_DERIVATIVE_RESIDUAL", mp.nstr(abs(mu_derivative-identity_derivative), 8))
        if scan:
            grid = [response(mp.mpf(i)/100) for i in range(1, 1001)]
            assert all(r["P_MeV_fm3"] > 0 and r["C_MeV2"] > 0
                       and 0 < r["cs2"] < 1 for r in grid)
            for key in ("W_MeV", "C_MeV2", "cs2"):
                print("SAMPLED_RANGE", key, mp.nstr(min(r[key] for r in grid), 20),
                      mp.nstr(max(r[key] for r in grid), 20))
        # Compare strings here so raising ambient precision cannot round the
        # verification values after this precision context has closed.
        return (mp.nstr(xmin, 28), mp.nstr(wmin, 28),
                tuple(mp.nstr(v, 28) for r in rows.values() for v in r.values()))


a = calculate(80, scan=True)
b = calculate(120)
assert a == b, "80/120-digit calculations disagree in the first 28 digits"
print("80/120_DIGIT_AGREEMENT_TO_28_DIGITS", True)
```

## Результаты и смысл проверок

Основные значения:

- $P(n_0)=18.723616823830875498\,\mathrm{MeV/fm^3}$.
- $\varepsilon(n_0)/n_0-M_N=124.31746962356431146\,\mathrm{MeV}$.
- $n_{\min}/n_0=2.14102631011896052803864901227$.
- $W_{\min}=858.082916153435188666429222638\,\mathrm{MeV}$.
- $C(n_{\min})=1549550.1\,\mathrm{MeV^2}=2\lambda W_0^2$.
- $d^2W/d(n/n_0)^2=0.39561729432424183067\,\mathrm{MeV}$ в минимуме.
- $c_s^2(10n_0)/c^2=0.644425960146363857784037846356$.
- При замороженном $W$ последняя величина вместо этого равна $0.67577369516592848505$.

Проверки сравнивают отдельные термодинамические выражения, дифференцирование вдоль повторно решаемой ветви, аналитическую связь элементов Гессиана и увеличение точности. Они не доказывают уникальность корня во всём пространстве состояний, устойчивость к любым неоднородным возмущениям или применимость модели к реальным ядрам. Сетка не заменяет интервальную оценку между узлами.

Эмпирическая цель для насыщения, если объявляется ядерная интерпретация, — минимум энергии на частицу около $-16\,\mathrm{MeV}$ при плотности около $0.16$–$0.17\,\mathrm{fm}^{-3}$, следовательно $P\simeq0$. Источник: [PTEP 2016, раздел 3.2](https://academic.oup.com/ptep/article/2016/6/06A106/2594768). Вычисленные здесь $+124.3175\,\mathrm{MeV}$ и положительное давление не согласуются с этой целью. Она не использована для изменения исходных параметров.

## Связь с исправлением звёздного запуска

Выявленный 8 сентября сбой передачи центрального разложения в TOV проверен отдельно при $n_c/n_0=0.1$: неправильные единицы приводили к стартовому давлению $0.21094523277081537$ вместо центрального $0.21427170712136365\,\mathrm{MeV/fm^3}$. Это соответствовало эффективной плотности $0.09918871874209298n_0$. Неподправленный основной решатель при этой плотности воспроизводил ошибочную массу независимого давления с относительной разницей около $8.2\times10^{-9}$. Только согласованное преобразование единиц центрального полинома в диагностическом процессе уменьшило расхождение при настоящем $0.1n_0$ до $4.4\times10^{-12}$ по массе и $1.1\times10^{-7}$ по приливной деформируемости.

Это причинная диагностика конкретного численного сбоя, без изменения параметров взаимодействия. Она не является полной приёмкой исправленного звёздного производителя и не меняет приведённого выше результата насыщения.
