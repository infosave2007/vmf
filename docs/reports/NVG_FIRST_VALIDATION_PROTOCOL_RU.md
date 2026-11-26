# Первый физический протокол NVG: кандидаты, источники и независимый тест

**Срез протокола:** 14 сентября 2026 г.  **Текущий статус:**
`PENDING_EMPIRICAL_INPUTS`. Это **предлагаемый дизайн** физической проверки, а не замороженная empirical preregistration и не результат сравнения с данными.
В этом цикле не выполнялись подгонка, импорт новых экспериментальных файлов или
эмпирический тест; поэтому эмпирический вес равен нулю и научного `PASS` нет.

Точные raw/statistical products, cuts, covariance и полный forward model пока не
доступны. Поэтому ниже различаются предложенный unused check и доказанная
независимость: последняя ещё не установлена. До превращения этого текста в
preregistration нужно отдельно просмотреть доступную историю проекта/репозитория
и provenance источников на предмет прежнего использования этих продуктов. Этот
текст не утверждает новизну holdout и не импортирует новый эксперимент.

## 1. Что именно сравнивается

### 1.1. Две существующие гипотезы; три заранее заданных comparison units

На уровне model classes scope ограничен двумя уже существующими гипотезами:

| ID гипотезы | Зафиксированная гипотеза | Роль сейчас | Что запрещено |
|---|---|---|---|
| **Q4-control** | Исходное source-complete действие с \(U_4(W)=\lambda(W^2-W_0^2)^2/4\), каноническими кинетическими членами, \(d=4\), однородной холодной симметричной средой и исходными входами | Отрицательный контроль. При \(n_0=0.16\,\mathrm{fm}^{-3}\) уже известны \(P=18.7236\,\mathrm{MeV/fm^3}\) и \(E/A-M_N=+124.3175\,\mathrm{MeV}\); это не связанное насыщение | Подгонять \((\lambda,g_\omega,W_0)\) под выбранный эксперимент или скрывать исходный провал |
| **W8-family** | Уже объявленное \(U_8(y)=a_2z^2+a_3z^3+a_4z^4\), где \(z=y^2-1\), \(y=W/W_0\), с двумя обратными конструкциями \(y_*=0.90\) и \(0.93\) | Альтернативный существующий класс без нового оператора; \(n_0\), \(E/A-M_N=-16\,\mathrm{MeV}\) и \(K=240\,\mathrm{MeV}\) заданы его построением | Добавлять оператор, менять \(y_*\) после просмотра остатка или выбирать design по residual |

Группировка под одним именем **W8-family** не устраняет multiplicity. Внутри
неё заранее тестируются две отдельные design-единицы **W8.90** и **W8.93**;
Q4-control остаётся negative control. Это три comparison units при двух
уровнях гипотез. Обе W8-конструкции не принимаются как физически подтверждённые; это только заранее заданные расчётные альтернативы и не
становятся «победителем» по просмотренному residual. Если когда-либо делается
утверждение о W8-family, процедура учёта этой множественности и правило выбора
должны быть объявлены до residual; в R1 новая candidate не добавляется.

Ранее полученная двухцелевой калибровка квартичной формы с
\(K\simeq692\,\mathrm{MeV}\) остаётся видимым counterfactual/negative control;
она не является третьим кандидатом и её \(K\) не является held-out проверкой.
Калибровочные цели не получают доказательного веса.

Причина узкого выбора — не обещание, что W8 физически верен, а защита от model
shopping: исходная форма остаётся контролем, а две уже существующие W8 design
тестируются одним frozen-by-design интерфейсом. Но сам интерфейс пока не
является preregistration: до его заморозки нужен history/provenance audit,
включая проверку, не использовались ли эти продукты при выборе \(y_*=0.90\) или
\(0.93\).

### 1.2. Что уже известно и чего это не доказывает

Исполняемая модель описывает нулевую температуру, однородный mean-field, один
фермионный состав и \(d=4\). В ней нет полного изоспинового сектора, конечного
ядра, кулоновского поля, оболочек, спаривания, поверхности, динамического
RPA/линейного отклика или экспериментальной модели установки. Поэтому
совпадение насыщения, статической кривизны, конечной капли либо одной звёздной
величины не является проверкой реального ядра. Статический finite-\(q\) TF-аудит
из существующего закрытия — вычислительный контроль алгебры, но не измерение
динамического отклика.

## 2. Идентифицируемость: два разных вида вырождения

### 2.1. Невидимая нормировка поля в однородном bulk

При фиксированном \(y=W/W_0\) однородная квартичная энергия зависит от
комбинаций

\[
 A=\lambda W_0^4,
 \qquad M_N,
 \qquad C_v={g_\omega^2\over m_\omega^2},
 \qquad d,
\]

(и от объявленных единиц плотности), а не от отдельной нормировки \(W_0\).
Следовательно, при фиксированных физических \(M_N,m_\omega,g_\omega\)
преобразование

\[
 W_0\mapsto sW_0,\qquad \lambda\mapsto\lambda/s^4,\qquad
 g_s\mapsto g_s/s,\qquad q_\phi\mapsto q_\phi/s
\]

оставляет \(M^*=g_sW\), \(m_\omega=q_\phi W_0\), \(A\) и bulk-EOS при том
же \(y\) неизменными. Это **параметрическая неидентифицируемость** отдельных
\(W_0\) и \(\lambda\), условная на разрешённом совместном изменении обоих и
на отсутствии независимой информации о \(\lambda\) или массе; это не новая
симметрия полной динамики.

Канонический статический пространственный член имеет коэффициент

\[
 {1\over2}(\nabla W)^2={W_0^2\over2}(\nabla y)^2,
\]

поэтому при таком преобразовании коэффициент \(y\)-градиента умножается на
\(s^2\), а \(m_\sigma=\sqrt{2\lambda}\,W_0\) переходит в \(m_\sigma/s\). Для
вакуумного древесного скалярного обмена при волновом числе \(q\) (in MeV)
получается

\[
-\frac{(g_s^{\prime})^2}{q^2+(m_\sigma^{\prime})^2}
=-\frac{g_s^2}{s^2q^2+m_\sigma^2}.
\]

При \(q=0\) это совпадает с исходным ядром; при \(q>0\) — уже нет. Здесь \(q\)
— волновое число, не заряд \(q_\phi\). Более того, вакуумное exchange kernel
не следует автоматически отождествлять с in-medium collective mode: измеримый
ядерный отклик требует вывода полной смешанной задачи
scalar--vector--fermions. Полный observed nuclear response не является
изолированным \(K_\sigma(q)\).

**Следствие:** ещё один однородный EOS в другой нормировке поля не различит
отдельные параметры. Один конечный-\(q\) tree-level observable уже может
идентифицировать range/scale, если \(A\), общая нормировка и forward kernel
независимо известны. Второй конечный-\(q\) point заранее выбран как более сильный
shape-check и защита от дополнительной nuisance-degeneracy, а не как
математическая необходимость в этом идеальном случае. Для реального полного
response нужны динамический или статический изоскалярный отклик и явный
forward model; \(q=0\) служит контрольной точкой, а не достаточным доказательством.

### 2.2. Невыбранный хвост потенциала

Это не то же самое, что нормировка поля. Конечный локальный jet при насыщении
не фиксирует продолжение \(U\) вне этой точки. Уже имеющийся математический
контрпример — не принимаемая новая теория:

\[
 \Delta U(y)=a_4(y^2-1)^4(y^2-y_*^2)^4\geq0.
\]

Он сохраняет значение и производные порядков (0,1,2,3) в вакууме и в выбранном
\(y_*\), но меняет старший хвост с \(p=8\) на \(p=16\). При одном и том же
background конечный-\(q\) linear observable также может совпасть, если локальные
данные, которые он видит, совпадают. Измерение градиентного коэффициента
различает \(W_0\)-scale family, потому что меняется \(W_0^2/2\), но автоматически не
различает \(W_8\)-tail и \(U_{16}\)-tail в той же calibration point.

Для tail discriminator нужна density excursion: нелинейный response конечной
амплитуды, несколько плотностей или heterogeneous finite-nucleus profile,
который реально посещает \(y\ne y_*\) и чувствует более высокие производные.
Поэтому \(\Delta U\) не добавляется в список кандидатов, а finite-\(q\) одного
background не объявляется универсальным tail-тестом.

## 3. Первичные источники и фактическая доступность

Дата проверки ссылок: 14 сентября 2026 г. Проверялись статьи коллабораций /
экспериментальных групп или официальный репозиторий данных, а не пересказы.
Приведённые ниже опубликованные числа — только контроль происхождения; они не
загружались в расчёт и не являются результатом NVG.

Для ISGMR важно не смешивать уровни продукта. Экспериментально измеряемая
energy--angle величина — дифференциальное сечение
\(d^2\sigma/(d\Omega\,dE)\) либо counts \(N_{ij}\) детектора после заявленных
response/background corrections. MDA представляет угловые bins basis-функциями
и извлекает коэффициенты \(a_L(E)\); \(a_0(E)\) — extracted monopole coefficient,
а не измеряемое \(a_0(E,\theta)\). Если используется MDA output, его covariance
должна включать angular-fit/extraction uncertainty и общие nuisance; нельзя
подменять им исходные measured cross sections.

Явный product path таков: detector counts \(N_{ij}\) → corrected cross section \(d^2\sigma/(d\Omega\,dE)\) → angular MDA fit \(a_L(E)\) → strength \(S_0(E)\), moments и centroid. Каждый шаг может вносить shared nuisance и терять информацию; covariance следующего узла должна это отражать. `a_0(E,\theta)` не является допустимым смешанным обозначением measured и extracted levels.

| dataset ID | Первичный источник и версия | Что реально доступно | Ковариация и provenance | Роль и текущий статус |
|---|---|---|---|---|
| `ISGMR_6Li_RCNP_2025_v1` | Arroyo *et al.*, arXiv v1 (24.12.2024), **Phys. Rev. C 111, 014308 (2025)**, [первичный текст](https://arxiv.org/abs/2412.18658), [DOI](https://doi.org/10.1103/PhysRevC.111.014308) | RCNP: \(^{6}\mathrm{Li}\), 343 MeV, мишени \(^{58}\mathrm{Ni},^{90}\mathrm{Zr},^{116}\mathrm{Sn},^{208}\mathrm{Pb}\); spectra/cross sections binned by angle and energy, \(0.5^\circ<\theta_{lab}<9.2^\circ\), \(E_x=5\)--30 MeV; MDA reports \(a_L(E_x)\) and moments. Примеры: для \(^{90}\mathrm{Zr}\) \(E_{center}=16.6\pm0.1\) MeV, для \(^{208}\mathrm{Pb}\) \(13.6\pm0.1\) MeV. | В проверенном тексте/HTML нет машиночитаемых сырых counts и полной матрицы covariance для angular extraction. Табличные ошибки — 95% CI fit/stat; для `%EWSR` отдельно указана систематика до порядка ±20%. Raw counts не обязательны, если источник даст joint likelihood или достаточный processed cross-section/MDA product с covariance и nuisance; сейчас это отсутствует. | Primary proposed unused energy--angle holdout: сначала \(d^2\sigma/(d\Omega\,dE)\)/counts, иначе только \(a_L(E)\) с extraction covariance. Сейчас `PENDING_EMPIRICAL_INPUTS`; centroid не объявляется ни reused, ни independent без provenance audit. |
| `PREX2_PV_2021_v1` | Adhikari *et al.* (PREX), arXiv v1, **Phys. Rev. Lett. 126, 172502 (2021)**, [первичный текст](https://arxiv.org/abs/2102.10767), [DOI](https://doi.org/10.1103/PhysRevLett.126.172502) | \(A_{PV}=550\pm16_{stat}\pm8_{syst}\) ppb при \(Q^2=0.00616\,\mathrm{GeV^2}\); опубликованы \(F_W\), объединённый \(R_n-R_p=0.283\pm0.071\) fm и описание 96 slug-ов. | Статья разделяет stat/syst и описывает acceptance convolution, но в проверенном публичном тексте нет полного covariance сырых slug/acceptance nuisance. Одна source-marginal проверка может быть корректна при предоставленном joint likelihood или достаточном processed product с nuisance; совместный PREX/CREX test требует cross-experiment covariance либо source-declared factorization. | Отдельный finite-size/isovector gate, не прямое измерение scalar exchange; сейчас `PENDING_EMPIRICAL_INPUTS`. Не считать raw omission запретом любой inference. |
| `CREX_Ca48_2022_v1` | Adhikari *et al.* (CREX), arXiv v1, **Phys. Rev. Lett. 129, 042501 (2022)**, [первичный текст](https://arxiv.org/abs/2205.11593), [DOI](https://doi.org/10.1103/PhysRevLett.129.042501) | При \(q=0.8733\,\mathrm{fm^{-1}}\): \(A_{PV}=2668\pm106_{stat}\pm40_{syst}\) ppb, \(F_W=0.1304\pm0.0052_{stat}\pm0.0020_{syst}\), \(R_n-R_p=0.121\pm0.026_{exp}\pm0.024_{model}\) fm. | Опубликованы отдельные stat/syst/model компоненты, но не полная общая covariance raw acceptance и корреляций PREX/CREX. Отдельный CREX marginal возможен только при достаточном processed likelihood/product; joint PREX+CREX остаётся `UNRESOLVED` без off-diagonal/cross-source provenance. Нельзя заполнять missing off-diagonal нулями или предполагать independence. | Ортогональный weak-form-factor/isovector gate; это опубликованная ненулевая-\(q\) точка, но не изолированный \(K_\sigma(q)\). Сейчас `PENDING_EMPIRICAL_INPUTS`. |
| `AME2020_mass_release` | Huang *et al.*, Part I, DOI [10.1088/1674-1137/abddb0](https://doi.org/10.1088/1674-1137/abddb0); Wang *et al.*, Part II, DOI [10.1088/1674-1137/abddaf](https://doi.org/10.1088/1674-1137/abddaf); официальный [AMDC release](https://amdc.impcas.ac.cn/web/masseval.html) | AME2020 `mass_1.mas20`, `massround.mas20`, reaction tables и `covariance.mas20` (zip) перечислены официальным центром; это evaluated masses, а не новая сырая экспериментальная выборка. | Официальная страница сообщает наличие variances/covariances для primary nuclides и требует ссылаться на оригинальные статьи. До загрузки нужны release URL, дата, файл SHA256, единицы, правила выбора nuclides и provenance каждой входной массы. | Будущий mass/binding gate конечного функционала; не заменяет finite-size calculation и не используется сейчас. |

Связь \(K=240\,\mathrm{MeV}\) с опубликованным ISGMR centroid нельзя заключать
только из того, что оба числа относятся к compressional physics. \(K=240\) здесь
является calibration input. Centroid может иметь общую model/systematic provenance
с калибровкой, а может быть независимым продуктом; это вопрос audit, не доказанный
reuse и не доказанная independence. Пока provenance не установлена, centroid —
source check, но не held-out evidence.

PREX/CREX — ненулевой переданный импульс, но \(F_W\) — weak form factor ядра, а
не автоматически \(K_\sigma(q)\) NVG. Поэтому источник нельзя использовать как
готовое scalar observable без изовекторного, кулоновского и электрослабого
forward model.

Методологический primary context также проверен: [Furnstahl--Serot,
Phys. Rev. C 47, 2338 (1993)](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.47.2338)
показывает, что модель может попасть в saturation, но провалить charge densities,
surface energetics, shell и spin-orbit конечных ядер; [Roca-Maza *et al.*, Phys.
Rev. C 84, 054309 (2011)](https://arxiv.org/abs/1110.2311v3) описывает DD-ME\(\delta\),
где microscopic nuclear-matter inputs и отдельное согласование finite-nucleus
energies/radii разделены. Это prior art, не подтверждение NVG. Для неединственности
high-density continuation см. [Mueller--Serot (1996)](https://arxiv.org/abs/nucl-th/9603037);
источник не доказывает конкретную W8-конструкцию.

## 4. Предлагаемый physical-validation protocol (пока не preregistration)

### 4.1. До получения и до объявления независимости

1. Зафиксировать ID comparison unit: Q4-control, W8.90 или W8.93; сохранить
   исходное действие, \(W_0,M_N,m_\omega,g_\omega,d\), units и все calibration
   roles. Нельзя читать сохранённые response tables как входы и нельзя менять
   параметр после просмотра residual.
2. Провести history/provenance audit до preregistration: проверить доступную
   историю проекта/репозитория, даты источников, reuse response tables и факт,
   использовались ли \(a_L(E)\), centroid, moments или \(y_*=0.90/0.93\) при
   выборе параметров. До этого только предлагается unused quantity; её
   independence не продемонстрирована.
3. Объявить proposed primary unused check: полный measured energy--angle
   продукт \(d^2\sigma/(d\Omega\,dE)\) или counts \(N_{ij}\) для \(^{90}\mathrm{Zr}\)
   из `ISGMR_6Li_RCNP_2025_v1`; \(^{208}\mathrm{Pb}\) — заранее объявленный
   cross-check. Если доступны только MDA coefficients, использовать \(a_L(E)\),
   прежде всего extracted \(a_0(E)\), только с extraction covariance; никогда не
   писать \(a_0(E,\theta)\) как measured input. Не выбирать отдельные bins по
   близости.
4. Centroid и \(K\) остаются development/calibration quantities до завершения
   provenance audit. Данные уже были просмотрены при составлении текста, поэтому
   слово «слепой» не используется; proposed unused означает только «не использован
   для построения параметров/выбора W8.90 против W8.93», а не доказанную
   независимость.
5. CREX \(F_W(q=0.8733\,\mathrm{fm^{-1}})\) и PREX form-factor point
   зарегистрировать как orthogonal isovector/finite-size gate. Он может стать
   статистическим holdout только после заморозки isospin, Coulomb и electroweak
   forward model; сейчас он не измеряет \(K_\sigma(q)\) напрямую.
6. Для каждого фактически используемого продукта получить provenance manifest:
   URL/DOI, дата, размер и SHA256, units, binning, cuts, calibration/background,
   detector/reaction response, nuisance and covariance. Если raw отсутствует,
   это не ошибка само по себе: записать `MISSING` и проверить, есть ли joint
   likelihood либо достаточный processed product с полностью определёнными
   nuisance. Никогда не записывать missing covariance как нули или independence.

### 4.2. Какой продукт и какая микрофизика нужны

Для ISGMR исходный measured product и extracted MDA product должны быть разведены:
\(d^2\sigma/(d\Omega\,dE)\) или \(N_{ij}\) сравниваются с forward-folded response;
\(a_L(E)\) сравниваются только с моделью того же MDA extraction или с заранее
определённым inverse map. Covariance extracted output включает uncertainty
angular fit, basis/MDA choice, optical-model/reaction nuisance и shared
systematics. Один опубликованный centroid или moment не заменяет этот product.

Для ISGMR нужен конечный сферический ground state и **самосогласованный**
линеаризованный response из того же функционала: канонические gradients
\(W,A_\mu\), scalar--vector--Fermi mixing, particle-hole kernel, continuum и
boundary conditions. Нужны proton/neutron degrees of freedom, Coulomb,
isovector operator (если принят отдельной гипотезой), surface, shell, pairing и
center-of-mass treatment. Reaction model (DWBA/optical potential), detector
acceptance и MDA basis объявляются nuisance, а не подгоняются так, чтобы исчезло
расхождение.

Для PREX/CREX дополнительно нужны density fields \(\rho_p,\rho_n\), charge form
factors, Coulomb-distorted electron waves, weak charges/radiative corrections и
experimental acceptance. Нельзя подставлять neutron skin из статьи в NVG без
этого forward model. Ни один недостающий член не принимается настоящим текстом
как новый закон.

### 4.3. Расчёт и критерий различения

1. Сначала воспроизвести для Q4-control, W8.90 и W8.93 только объявленные bulk
   controls и показать, какие из них calibration inputs. Q4 провал насыщения не
   исправлять; W8.90/W8.93 не перенастраивать.
2. Вывести из полной динамической задачи именно тот observed product, который
   предоставил источник, при исходных beam/angle/energy kinematics. Не назначать
   \(q\) по желанию: вычислять каждый \(q_i\) из source-provided kinematics и
   хранить conversion в fm/MeV. Один finite-\(q\) point вместе с \(q=0\) может
   идентифицировать tree-level range/scale, если \(A\), absolute normalization и
   forward kernel независимо известны. Второй finite-\(q\) point — protocol choice
   для проверки shape и дополнительной nuisance-degeneracy; требование «не менее
   двух» не является математической необходимостью в идеальном случае с известным
   \(q=0\). Если normalization неизвестна либо доступна только одна scalar point,
   результат для scale помечается `UNIDENTIFIABLE`, а не нулём.
3. Полный observed nuclear response не равен isolated scalar exchange kernel.
   Сначала сравнивать весь energy--angle response (или weak form factor в его
   исходной kinematic point), а не одну цифру, полученную из тех же данных.
   Хранить отдельно theory, experimental-statistical, experimental-systematic,
   extraction и probe-model covariance. Для одной source-marginal проверки
   разрешено вычислять \(r^T C_i^{-1}r\), если source-provided likelihood/product
   задаёт полный \(C_i\) и nuisance priors. Для joint PREX+CREX нужен полный block
   covariance \(C_{ij}\) с off-diagonal terms либо source-declared factorization;
   missing off-diagonal precludes именно joint test, но не любую marginal inference.
   Нельзя складывать error components в квадрат или ставить off-diagonal zero без
   документированной независимости.
4. До получения достаточного product, covariance и полного forward model никаких
   sigma, p-value, likelihood, Bayesian evidence или empirical PASS. Центральные
   числа выше — опубликованные source checks, а не присвоенная NVG-ошибка.
5. Оба W8 design проходят один и тот же заранее описанный interface, но
   multiplicity остаётся явной: результаты W8.90 и W8.93 публикуются раздельно,
   без выбора winner после residual. Q4-control маркируется negative control. При
   несовпадении не разрешается менять \(y_*\), выбирать другую ветвь или добавлять
   параметр.

### 4.4. Fail-closed проверки

* отсутствие raw counts **не** является автоматическим fail, если имеется
  source-provided joint likelihood или достаточный processed product с units,
  cuts, response, nuisance и covariance; отсутствие необходимого covariance,
  extraction uncertainty или forward model → `PENDING`/`UNRESOLVED` по заранее
  выбранному test;
* missing off-diagonal covariance запрещает только соответствующий joint test,
  если не доказана factorization; отдельная source-marginal проверка остаётся
  возможной при достаточном \(C_i\), и нули/independence не изобретаются;
* NaN, отрицательная дисперсия, дубликат версии, неизвестная единица,
  неописанный cut, неразрешённая response correction или mutation одного
  counts/bin/ковариационной строки → отказ provenance/validation и отсутствие
  зелёного результата;
* подмена \(q\) и \(q_\phi\), удаление bins, подмена extracted \(a_L(E)\) raw
  cross section или попытка назвать \(a_0(E,\theta)\) measured input → fail closed;
* calibration target, сохранённая таблица, centroid без provenance, результат уже
  просмотренной капли или model-dependent neutron-skin extraction не считаются
  доказанной независимой экспериментальной точкой;
* совпадение при \(q=0\), положительный static Hessian или совпадение одной
  производной не закрывают finite-\(q\) scale discriminator, а finite-\(q\) одного
  background не закрывает tail discriminator.

## 5. Ближайший физический вывод

Самый маленький релевантный следующий шаг — не новая подгонка EOS, а один
самосогласованный ненулевой-\(q\) изоскалярный response с полными scalar/vector
gradients и finite-nucleus boundary. Практическая primary реализация —
зафиксированный measured energy--angle ISGMR product (или source-provided
sufficient processed/MDA product) для `ISGMR_6Li_RCNP_2025_v1`; CREX/PREX служат
отдельным isovector/weak-form-factor gate и не подменяют его. Один finite-\(q\)
может быть математически достаточен при известной normalization/A, второй —
заранее выбранная защита от shape/nuisance degeneracy. Для W8-tail дополнительно
нужен density excursion или heterogeneous profile. Только после получения
достаточного product, covariance и недостающей микрофизики можно решить, различает
ли эксперимент scale choices \(s=0.5,1,2\) или W8-tail choices. До этого честный
итог один: `PENDING_EMPIRICAL_INPUTS`, empirical weight \(=0\), без научного PASS.
