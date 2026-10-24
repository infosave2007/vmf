# Диагностика точности старого Euler/Ward-контроля

Это проверка численного контрольного примера, не изменение действия
или физических параметров. Эталонные JSON/Markdown и их производитель
`verification/source_complete_solution_audit.py` на момент этой диагностики
не изменялись. Последующая явная миграция описана в
[продолжении](../NVG_NUMERICAL_CLOSURE_AND_BINDING_RU.md): канонический
контроль теперь высокоточный, а прежняя функция сохранена как
`_legacy_manufactured_euler_ward_controls`. Приведённый ниже пример
обращается именно к исторической реализации.

## Что наблюдалось

При совместной регрессии новых и прежних расчётов один запуск дал
`1 failed, 285 passed in 26.89s`. Отказал только
`test_committed_generated_artifacts_match_reconstruction`: первый
различающийся байт 4739 находится в `euler_phi_abs` старого искусственного
контроля. Его сохранённое значение — `79562.440738`.
Первоначальный вывод pytest не сохранил целиком второе число; приписывать
ему конкретное значение по одному различающемуся байту нельзя.

Отдельная пересборка совпала побайтно. Последующие полные прогоны дали
`286 passed in 26.39s` с диагностическим pytest-hook и
`286 passed in 24.82s` обычным pytest. Новые 48 тестов отдельно прошли
на Python 3.9.6. Прямая утечка состояния новых тестов не обнаружена:
проверялись контекст точности mpmath, режим округления, NumPy error state,
неизменность используемых функций и матриц Γ/η. Это не доказывает
отсутствие всех эффектов порядка исполнения.

Повторный успешный запуск **не устраняет** наблюдавшийся непостоянный
отказ. Поэтому в старый тест добавлен только читаемый unified diff при
ошибке. Строгое побайтное сравнение, эталоны и физические gates сохранены.

## Независимая аналитическая проверка

Сигнатура (+---), ρ=φ̄φ, W=√(2ρ). Из того же локального действия
вариация по φ при независимом φ̄ даёт

\[
E_\phi=-\Box\bar\phi-iq(\partial\!\cdot\! A)\bar\phi
-2iqA^\mu\partial_\mu\bar\phi+q^2A^2\bar\phi
-2\lambda(\rho-W_0^2/2)\bar\phi
-g_s\frac{\bar\phi}{W}\bar NN.
\]

Формула вычислена при 80 знаках на **тех же уже округлённых бинарных
входах искусственного jet**, которые фактически получил старый код.
Отдельное значение производной не обязано быть нулём; тождество Ward
связывает все вариации и дивергенцию тока.

Результат независимого пересчёта родительским процессом:

```text
analytic |E_phi|  = 79562.440740079140757537028655381962418805794
FD |E_phi|        = 79562.4407382643437519617125597
magnitude error  = 0.000001814797005575316095656364
relative complex error = 4.138742930241037278865647e-11
reported |E_phi|  = 79562.440738
original Ward gate = True
```

Низшие цифры FD-значения зависят от конкретного вычислительного пути.
Независимые проверки аналитического выражения дали то же mp80-значение.
Показано, что печатный шаг `1e-6` меньше уже имеющейся ошибки вложенных
конечных разностей. Поэтому побайтная стабильность этой нормы и точность
самого тождества — разные требования. Это новый дефект воспроизводимости,
не прежнее различие алгоритмов `sum` в Python 3.9 и 3.12.

Наблюдались также изменения низших битов комплексных сокращений при
неизменных численных входах и изменении распределения памяти. Однако
точный низкоуровневый триггер отказавшего полного запуска не установлен.
Не следует объявлять найденную чувствительность доказанной единственной
причиной отказа или сообщать, что проблема устранена.

## Воспроизводимый независимый расчёт

Запускать из корня репозитория в исправном окружении Python 3.12 с
NumPy и mpmath. `setprofile` здесь лишь читает локальные переменные
контроля; прежний profiler восстанавливается в `finally`. Исходная функция
и результаты не подменяются, файлы не записываются.

```sh
PYTHONPATH=verification python3 -B - <<'PY'
import sys
import mpmath as mp
import source_complete_solution_audit as audit

captured = {}
def capture(frame, event, arg):
    if event == 'return' and frame.f_code is audit._legacy_manufactured_euler_ward_controls.__code__:
        captured.update(frame.f_locals)

previous = sys.getprofile()
try:
    sys.setprofile(capture)
    result = audit._legacy_manufactured_euler_ward_controls()
finally:
    sys.setprofile(previous)

with mp.workdps(80):
    def exact_binary_complex(value):
        return mp.mpc(mp.mpf(float(value.real)), mp.mpf(float(value.imag)))
    z = exact_binary_complex
    p = captured['p']
    q, gs, lam, W0 = map(mp.mpf, (p.q_phi, p.g_s, p.lam, p.W0))
    eta = (1, -1, -1, -1)
    phi, barphi = z(captured['phi']), z(captured['phibar'])
    rho, W = phi*barphi, mp.sqrt(2*phi*barphi)
    box = mp.fsum(eta[i]*z(captured['second_phibar'][i,i]) for i in range(4))
    div_A = mp.fsum(eta[i]*mp.mpf(float(captured['first_A'][i,i])) for i in range(4))
    A_dp = mp.fsum(eta[i]*mp.mpf(float(captured['A'][i]))*z(captured['dphibar'][i]) for i in range(4))
    A2 = mp.fsum(eta[i]*mp.mpf(float(captured['A'][i]))**2 for i in range(4))
    barNN = mp.fsum(z(captured['bar_N'][i])*z(captured['N'][i]) for i in range(4))
    exact = (-box-1j*q*div_A*barphi-2j*q*A_dp+q*q*A2*barphi
             -2*lam*(rho-W0*W0/2)*barphi-gs*barphi/W*barNN)
    finite_difference = z(captured['E_phi'])
    print('analytic magnitude:', mp.nstr(abs(exact), 45))
    print('FD magnitude:', mp.nstr(abs(finite_difference), 30))
    print('magnitude error:', mp.nstr(abs(abs(exact)-abs(finite_difference)), 25))
    print('relative complex error:', mp.nstr(abs(exact-finite_difference)/abs(exact), 25))
    print('original report and gate:', result['euler_phi_abs'], result['all_pass'])
PY
```

Исправление самого старого производителя требует явной миграции:
заранее определить точность и независимый аналитический/высокоточный
контроль, затем пересобрать эталоны и проверить зависимые отпечатки.
Нельзя просто округлить результат сильнее до удобного совпадения,
заменить ожидаемое число после отказа или ослабить тест без такого обоснования.
