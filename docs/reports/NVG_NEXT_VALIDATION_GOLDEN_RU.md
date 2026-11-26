# Следующая независимая проверка W8: Golden-дизайн

**Статус:** проект следующей проверки; не код, не запуск и не принятие результата.
**Цикл:** Golden / Adoption 001 / Decision 003, 2026-09-12.

## 1. Назначение и границы

После того как отдельно выполняемый расчёт residual-enrichment завершится и
его артефакты будут заморожены владельцем, предлагается одна независимая
проверка вывода о конечной устойчивости W8. Цель — выяснить, согласуется ли
вторая вариация с движущейся границей вещества с одной и той же
вариационной структурой, которая порождает текущие остатки. Это проверка
алгебры, интерфейса и конечного численного пространства, а не новое
физическое взаимодействие и не обещание положительного ответа.

Сохраняются исходные параметры, обратные конструкции W8 `y*=0.90` и `0.93`,
`N=208`, полная реакция Гаусса, фиксированное число частиц, внешняя область,
сетка, граничные условия и единственный известный совместный translation-
квоциент. Не добавляются кулон, асимметрия, pairing, shell-структура,
динамика/RPA, гравитация, эмпирические данные или новая физическая степень
свободы. Положительность Ritz-минимума и малый остаток остаются признаками
только заявленного конечного пространства; они не исключают невидимую
отрицательную моду, а decoupled negative mode может иметь нулевой остаток.

Текущие исходные интерфейсы: [контракт статической второй вариации](verification/contracts/droplet_stability.md), [контракт координатного отклика](verification/contracts/droplet_coordinate_response.md), [аудит стабильности](verification/nvg_droplet_stability_audit.py), [аудит координатного отклика](verification/nvg_droplet_coordinate_response_audit.py) и [аудит конечного `N`](verification/nvg_finite_droplet_audit.py). Сводка конечного кандидата находится в [NVG_FINITE_DROPLET_RU.md](NVG_FINITE_DROPLET_RU.md), а текущая семантика координат — в [NVG_DROPLET_COORDINATE_RESPONSE_RU.md](NVG_DROPLET_COORDINATE_RESPONSE_RU.md).

### Принятые и ожидающие факты

Приняты исходные симметричные бескулоновские Thomas–Fermi-фоны с `N=208`,
фиксированными калибровочными входами, полной Gauss-реакцией, moving
Thomas–Fermi edge, положительной физической нормой и одним явно заданным
совместным translation-направлением. Предыдущие расчёты дают семь
положительных классификаций в конечных пространствах и семь
неразрешённых; в частности, `.90` сохраняет только `ell=0`, а `.93` —
`ell=0..5`, при консервативной трактовке всех остальных секторов.

Ожидается только завершение отдельной residual-enrichment попытки. Её
изменяемые продукты до заморозки не являются доказательством и не являются
входом этого дизайна. После заморозки они могут быть provenance для
сопоставления, но не заменяют независимую проверку и не дают права менять
выбранные направления.

## 2. Полный пул 30 идей

Ниже сохранены формулировки пяти изолированных генераторов. Оценки — исходные
`N/V/F` из `SCORING-001.md`; итоговая оценка равна `0.35N+0.40V+0.25F`.
Оценки — суждения о новизне, реализуемости и соответствии задаче, а не
научное свидетельство.

### R — сохранение допустимого физического пространства

1. **R3 — 7.95 (`N6/V9/F9`).** Independently reconstruct every implemented constraint and the joint-translation quotient, publishing rank, nullity, and principal-angle diagnostics for the retained physical tangent space.
2. **R5 — 3.40 (`N6/V2/F2`).** Assemble the complete fourteen-sector block Hessian with every numerically allowed off-sector coupling and permit Schur reduction only after certifying a positive invertible eliminated block on the identical constrained domain. **Ловушка:** combines 14 sectors across two different designs; rotationally invariant angular blocks cannot acquire invented physical cross-couplings.
3. **R2 — 6.90 (`N6/V7/F8`).** Certify the projected generalized Hessian’s finite-space inertia with interval-bounded symmetric-indefinite factorization, preserving an explicit unresolved bucket for near-zero pivots.
4. **R6 — 3.90 (`N6/V2/F4`).** Run a constraint-projected linearized evolution with independent time integrators and a quadratic-energy ledger, treating only reproducible exponential growth as a negative-direction witness. **Ловушка:** exponential-growth interpretation requires a justified kinetic/time-dependent model outside this static design scope.
5. **R1 — 8.15 (`N7/V8/F10`).** Construct an independent matrix-free second-variation oracle for the constrained energy and compare its bilinear symmetry, quadratic scaling, and term-level Gauss and moving-edge contributions with the assembled Hessian.
6. **R4 — 7.85 (`N8/V7/F9`).** Apply a partition-of-unity audit that decomposes each constrained quadratic form into bulk, moving-edge, and exterior pieces with all cross terms recorded and recombined.

### C — независимая реконструкция связи и второй вариации

1. **C3 — 8.55 (`N7/V9/F10`).** Compute constrained inertia from both a nullspace reduction and a bordered KKT pencil, retaining every physical mode and only the known joint-translation quotient.
2. **C4 — 4.95 (`N6/V4/F5`).** Assemble the Hessian in a rotated Cartesian basis with all angular cross-sector and moving-edge coupling blocks retained, then compare the resulting constrained quadratic form by congruence. **Ловушка:** treating off-ell angular couplings as physical violates spherical symmetry; mere rotation/congruence is already a diagnostic.
3. **C6 — 8.10 (`N8/V7/F10`).** Compare independently derived primal-energy and mixed weak-form/stress-tensor bilinear forms with identical boundary, norm, N, Gauss, and quotient treatment.
4. **C1 — 6.85 (`N7/V6/F8`).** Use a nonlinear constraint-manifold retraction to follow fixed-N, Gauss-compatible dilation and shear paths of the entire droplet, evaluating the full-energy second difference.
5. **C5 — 7.55 (`N6/V8/F9`).** Apply outward-rounded interval arithmetic to the finite-dimensional constrained Hessian, enclosing pivot signs and labeling every zero-crossing interval unresolved.
6. **C2 — 8.45 (`N9/V7/F10`).** Re-derive the moving-edge second variation with Hadamard shape calculus, including interface jumps and flux terms, on the identical admissible domain.

### H — аппаратная и арифметическая независимость

1. **H1 — 6.95 (`N5/V8/F8`).** Install an inertia watchdog: factor the full finite-space Hessian on the exact fixed-N/Gauss tangent space, removing only the known joint-translation null, and record its signed pivot count.
2. **H2 — 8.20 (`N6/V9/F10`).** Route the constraint bus through two independent algebraic layouts—bordered KKT coordinates and an explicit nullspace basis—and compare their finite-space inertia signatures.
3. **H3 — 6.90 (`N9/V5/F7`).** Apply cache-tiled domain-decomposition bracketing with retained interface traces, computing constrained local lower and upper spectral bounds for the same assembled quadratic form.
4. **H4 — 7.55 (`N6/V8/F9`).** Generate a shadow Hessian through an independent residual-differentiation path, then compare full constrained operator actions and symmetry blocks against the production assembly.
5. **H5 — 8.50 (`N8/V8/F10`).** Attach directed-rounding intervals to the generalized Hessian and positive physical norm, propagating pivot and conditioning uncertainty into finite-space sign enclosures.
6. **H6 — 7.10 (`N5/V9/F7`).** Run deterministic bus-timing replays with reordered sparse reductions, alternate precision, and fused versus unfused accumulation, comparing inertia and eigenvalue sign classifications.

### I — независимая инверсия и область пространства

1. **I1 — 8.15 (`N7/V8/F10`).** Reconstruct constrained Hessian–vector products through analytic variation, automatic differentiation, and symmetric energy differences along exact fixed-N/full-Gauss paths including the moving edge.
2. **I2 — 8.20 (`N6/V9/F10`).** Form an independently assembled bordered KKT operator for the full constrained second variation, quotienting exactly the single known joint translation, and compare inertia sector by sector.
3. **I3 — 6.85 (`N7/V6/F8`).** Re-express the same finite-domain quadratic form in an independent mixed radial–angular basis with independent quadrature and compute the constrained Morse index across all 14 sectors.
4. **I4 — 7.55 (`N6/V8/F9`).** Transport each frozen stationary candidate across nested outer domains and boundary closures, tracking constrained inertia and mode localization as the matter edge separates from the boundary.
5. **I5 — 8.25 (`N8/V8/F9`).** Apply verified finite-dimensional linear algebra to enclose near-zero constrained eigenvalues and inertia under rounding and assembly perturbations.
6. **I6 — 7.95 (`N6/V9/F9`).** Audit every Schur reduction by proving positivity and invertibility of the eliminated constrained block on the identical domain, then reconcile reduced and full inertia.

### B — биологические метафоры, сведённые к техническим механизмам

1. **B1 — 8.20 (`N6/V9/F10`).** Thymic negative selection: construct the admissible tangent space independently through null-space/SVD projection and bordered-KKT inertia, retaining exactly the known joint-translation quotient.
2. **B2 — 7.25 (`N7/V7/F8`).** Cell-signaling cascade: evaluate scalar–vector–matter coupling through multiple Schur-complement orderings, using an elimination only when its block is positive and invertible on the same fully constrained domain.
3. **B3 — 7.60 (`N8/V7/F8`).** Neural plasticity: re-express the frozen second variation in independently constructed constraint-preserving coordinate charts that include the moving matter edge.
4. **B4 — 6.90 (`N9/V5/F7`).** Evolutionary search: mutate and recombine trial subspaces from a predeclared grammar of global, interface-localized, angular, and mixed scalar–vector–matter shapes without reading residuals.
5. **B5 — 8.15 (`N7/V8/F10`).** Gut-flora cross-feeding: assemble a bilinear ledger of scalar, vector, matter, and pairwise cross-sector second variations, then compare reconstructed mixed quadratic forms with direct full-operator evaluations.
6. **B6 — 7.90 (`N7/V8/F9`).** Pulse–chase signaling: apply paired positive and negative amplitudes along independently constraint-corrected perturbations and fit even and odd energy responses while retaining the moving edge.

### Выбор и удалённые ловушки

Выбраны три различающихся seed-а: **C3 (8.55)** — две формы constrained
inertia; **H5 (8.50)** — интервальное разделение арифметического знака; и
**★ C2 (8.45)** — происхождение moving-edge второй вариации. C3 проверяет
представление ограничений, H5 — неопределённость округления/сборки, C2 —
самый специфичный непросмотренный интерфейсный источник кривизны. Они не
являются тремя подтверждениями и не предписывают положительный результат.

R5, R6 и C4 вынесены отдельно как ловушки с указанными выше причинами; они
не входят в shortlist. Близкие варианты KKT и interval сгруппированы и не
считаются независимыми подтверждениями.

## 3. Три focus-эскиза и source-дефекты

### Focus C3 — constrained inertia

**Эскиз (дословно из focus-файла):** “After the pending residual-enrichment calculation settles, freeze its outputs as provenance and run a separate finite-dimensional audit on the original .90/.93 candidates, parameters, domains, grids, and boundary treatment. For each candidate, assemble one canonical Hessian H, positive physical norm M, full linearized constraint Jacobian C, and the single known joint-translation vector t, without deleting any other physical coordinate. Route A uses a rank-revealing nullspace basis Z for the admissible tangent space and applies only the predeclared t quotient, then reports the inertia of Z^T H Z relative to Z^T M Z together with rank, residual, and near-zero tolerance diagnostics. Route B uses the same objects and quotient in a bordered KKT pencil, K(lambda) = [[H, Cq^T], [Cq, 0]] - lambda [[M, 0], [0, 0]], separating finite physical eigenvalues from multiplier-associated infinite or zero structure. Compare sign counts, nullities, translation multiplicity, constraint residuals, and eigenvalue paths across both routes and predeclared grid/domain variants, treating agreement only as a finite-dimensional consistency check. Any disagreement or tolerance-sensitive sign remains negative or unresolved, and frozen held-out probes are only a post hoc crosswalk, not experimental validation or coverage of unseen directions. A positive Ritz minimum or small residual cannot rule out an unseen negative direction, and a decoupled negative mode may have zero residual.”

**Риск:** обе реализации могут совместно наследовать ошибочные `C`, edge-Jacobian, metric или translation vector; совпадение тогда проверяет bookkeeping, а не устойчивость.

**Первый шаг:** после заморозки отдельного расчёта составить manifest одного кандидата с `H,M,C,t`, картами границы/edge и индексами, затем принудительно подать одинаковые массивы в обе реализации и ровно один translation gauge.

**Дочерние идеи (дословно):** “Run a rank, tolerance, and precision ladder for both formulations with independent factorizations where possible, reporting classifications as stable, near-zero, or unresolved rather than using stale PASS flags.” “Perform a discretization and domain continuation that tracks finite constrained eigenvalues and moving-edge contributions, preserving sector-level negative and unresolved findings when signs cross zero.” “Apply admissible invertible coordinate transformations and independent matrix-assembly orderings as metamorphic tests; inertia and the single translation quotient should remain invariant even if eigenvectors and conditioning change.” “After the active worker's artifact is archived, make a frozen crosswalk between its selected directions or held-out probes and the inertia results, using mismatches for diagnosis without rerunning or altering that work.”

### Focus H5 — verified finite-space signs

**Эскиз (дословно из focus-файла):** “After the pending residual-enrichment calculation settles, the verifier would consume its frozen basis, grid/domain, constraint, and quotient manifests without changing the selected directions. Using directed rounding or verified arbitrary-precision intervals, it would independently enclose the constrained generalized Hessian H and positive physical norm G on the same admissible domain, including full Gauss response, fixed particle number, the moving matter edge, and the single joint-translation quotient. A verified factorization or generalized-eigenvalue procedure would propagate pivot intervals, conditioning, and reconstruction uncertainty into finite-space sign enclosures. Any Schur reduction would be permitted only after establishing a positive invertible block under the same domain and full constraints. Sign intervals crossing zero would remain unresolved rather than becoming stale PASS flags. The output would certify arithmetic signs only in tested finite spaces, preserving negative and unresolved sectors while making no claim about unseen directions, continuum limits, or omitted nuclear physics.”

**Риск:** почти сингулярные нормы, constraint blocks или pivots могут расширить интервалы до unresolved; неверный quotient/edge bookkeeping может дать точную, но несостоятельную оболочку.

**Первый шаг:** read-only manifest одного замороженного finite-space с параметрами, сетками, constraints, Gauss, `N`, edge и quotient, затем список всех interval quantities, которые нужно восстановить.

**Дочерние идеи (дословно):** “Use verified inertia counts from interval LDLᵀ factorizations or Sturm-style spectral checks as an independent alternative to a lowest-Ritz calculation.” “Add an interval constraint-and-quotient audit that reports nullspace dimension and checks fixed particle number, Gauss response, moving-edge terms, and exactly one translation quotient without silent projection.” “Reconstruct the same frozen pencil with two independent verified arithmetic backends or precision schedules, treating disagreement as a conditioning diagnostic.” “Maintain a refinement ledger across predeclared grids and domains that records enclosure widths and sign stability without claiming a continuum error bound or hidden-spectrum gap.” “Package replayable interval certificates and provenance so a later potential-observable comparison can be labeled separately from numerical certification, holdout probes, and model validation.”

**Source-дефект:** “verified” не означает автоматически правильный edge или functional; интервалы могут сертифицировать только арифметические знаки уже правильно заданной конечной формы.

### ★ Focus C2 — Hadamard moving-edge derivation

**Эскиз (дословно из focus-файла):** “After the pending residual-enrichment calculation settles, identify the exact constrained functional—if one exists—whose Euler–Lagrange equations match the current stationary residuals, while freezing the original parameters, fixed particle number, full Gauss response, nominal outer domain, and known joint-translation quotient. Represent the matter edge as a shape inside that same reference domain and pull every perturbed field back by a flow x -> x + εV(x), making edge motion explicit without changing the admissible constraint set. Apply first- and second-order Hadamard differentiation to bulk terms and interface conditions, retaining jumps of normal scalar/vector fluxes, curvature and normal-derivative terms, edge displacement of matter, and fixed-N/Gauss multiplier contributions. The result is a coupled quadratic form in bulk perturbations and normal edge displacement, with tangential reparameterizations treated only as coordinate redundancy and the known translation quotient explicitly audited. On identical pulled-back meshes and domains, compare this form with finite differences of fully constrained re-solves under prescribed edge deformations and with the existing bulk Hessian, using held-out edge shapes. Localize discrepancies into interface jumps, fluxes, constraint response, quotient handling, and discretization or quadrature rather than converting agreement into a PASS flag. Agreement could reveal whether omitted edge curvature terms affect the verification, but a positive Ritz minimum or small residual would not exclude unseen negative directions or a decoupled zero-residual negative mode. Any later mapping to nuclear or other observables must be a separate model-comparison exercise, not experimental validation of this internal numerical check.”

**Риск:** derivation имеет смысл только при differentiable edge и единой вариационной структуре residuals, jump conditions, constraints и discretization; иначе distributional/sign errors могут создать или убрать curvature.

**Первый шаг:** заморозить manifest одного `.90/.93` кандидата: точные поля и edge, normal/flux conventions, interface equations, fixed-`N` и Gauss linearizations, outer pullback, mesh/quadrature и quotient basis; затем term-by-term проверить, что один functional дифференцируется в текущие residuals.

**Дочерние идеи (дословно):** “Promote normal edge displacement to an explicit generalized-eigenproblem coordinate and compare the coupled spectrum with the bulk-only spectrum; use a Schur reduction only after verifying a genuinely positive, invertible bulk block on the same admissible domain and constraints.” “Build a manufactured piecewise-smooth interface test with prescribed fields and known flux jumps to test first- and second-shape derivative identities independently of the candidate droplet.” “Cross-check continuous Hadamard terms against a discrete-adjoint or automatic-differentiation derivative of the assembled constrained discretization, keeping continuum-identity error separate from discretization error.” “Partition edge deformations by angular sector and localization width, including a translation-like family and held-out deformations, to diagnose whether unresolved sectors are edge-dominated without treating sector-wise agreement as a stability proof.” “Repeat the comparison with interface-aligned and unfitted meshes plus varying edge regularity, reporting convergence trends only as numerical diagnostics because no continuum error bound or hidden-spectrum gap is available.”

**Провокация C2 (дословно, автор — выбранный C2 ADHD worker):** “Could the apparent moving edge be a level set of a diffuse matter profile rather than a sharp interface, and would a distributional or level-set second variation distinguish physical edge curvature from representation artifacts without changing the model's parameters?”

**Обязательная поправка Decision 003:** disagreement сам по себе не является negative-mode witness; интервальный padding вокруг binary64 не есть outward enclosure assembly/BVP error; текущая TF matter edge рождается occupied support и continuity displacement. Нельзя предполагать независимую membrane degree of freedom, ненулевой flux jump или дополнительный физический interface, пока это не выведено из того же declared functional. Shared-H KKT agreement не тестирует shared erroneous Hessian.

## 4. Выбранный дизайн: функциональный gate + условная Hadamard-проверка

### 4.1. Замораживаемая постановка

Для каждого из двух кандидатов фиксируется один manifest: `y*=.90/.93`,
`N=208`, исходные W8-параметры, BVP solution, `24/32 fm` outer box,
`800/1600/3200` intervals и объявленные block sizes `6/10/14/18`, boundary
conditions, quadrature/load weights, `mu`, `y,a,n,n_s`, `n'`, `fermi_margin`,
edge location, full Gauss solve, fixed-`N` linearization и basis quotient.
Manifest не выбирает лучший sector и не читает residual-selected directions.

В текущем коде `Background.edge_x` находит последний узел с
`fermi_margin = nu - mass > 0` и линейно интерполирует crossing. Вторая
вариация строится в `_hessian_quadratic`; `A=F_nn` и `B=F_ny` интегрируются
через `_occupied_integral_matrix` только до этой moving edge, а scalar terms
идут по полной radial области. `_displacement_density` задаёт continuity-
generated density direction и обнуляет её вне occupied support. Для `ell=1`
`_basis_columns` явно добавляет joint translation, а
`_generalized_spectrum` применяет только один `G`-orthogonal quotient.
Эти факты задают интерфейс, который нужно объяснить, а не переписать.

### 4.2. Functional-consistency gate (обязательный нулевой этап)

До любой записи Hadamard-формул нужно предъявить один constrained functional
с теми же полями, Lagrange multipliers, boundary conditions и admissible
domain, чей первый variation даёт текущие stationary residuals из
`W8Design.bvp_fun`, включая локальную TF-ветвь
`nu=mu-gomega W0 a`, `n=0` при `nu<=M y`, положительный Gauss operator,
fixed-`N` и внешние условия `y=1`, `A=0`.

Если такой functional не найден, если edge является только sampled active-mask
crossing без differentiable shape, или если residual и discrete assembly не
являются его consistent discretization, C2 останавливается с
`NEEDS-DECISION/UNRESOLVED`: interface jump, curvature и normal-flux terms не
добавляются и никакой continuum conclusion не делается. В частности, нельзя
вводить независимую surface/membrane coordinate только потому, что она
удобна для Hadamard calculus.

Если gate проходит, flow `Phi_epsilon(x)=x+epsilon V(x)` задаёт форму edge,
а все bulk fields pull back на одну reference domain. Tangential edge motion
отмечается как reparameterization; нормальное смещение — не новая физика, а
геометрическое представление уже имеющегося occupied support.

### 4.3. Производные и независимое сопоставление

Из прошедшего functional gate выводятся first/second shape derivatives для
bulk, occupied-support boundary, scalar/vector normal flux и fixed-`N`/Gauss
multiplier terms. Jump terms включаются только если обе стороны, их
regularity и jump/continuity laws следуют из этого functional. Ненулевой jump
не постулируется.

Полученная форма должна быть разложена на bulk, edge, flux, constraint и
quotient contributions и затем собрана обратно. Но перед сравнением
фиксируется **один и тот же вариационный объект**. Пусть `q=(n,y)`, а
`a[q]` — единственное поле, которое решается заново из Gauss-уравнения для
заданных `q`. Сравнивается reduced energy
`Ered[q]=E[n,y,a[q]]`, где `n` и `y` являются предписанными аргументами
пути. `n` и `y` не решаются заново, не минимизируются и не заменяются
локальной TF-замыканием после задания perturbation.

Есть два явно разделённых finite-difference режима:

* **Существующий same-Hessian режим.** Для направления `(u,v,v')` задаётся
  симметричная affine path `n_e=n0+e*u/sqrt(4*pi)`,
  `y_e=y0+e*v/sqrt(4*pi)`, `y'_e=y'_0+e*v'/sqrt(4*pi)`, с `e` и `-e` из
  объявленного списка. Проверяются `n_e>=0`, `y_e>0` и fixed-`N` из того же
  `u`; затем для каждого знака **только** `a_e=a[n_e,y_e]` повторно
  решается положительный Gauss BVP. Энергия оценивается именно
  `Ered[n_e,y_e]`, а центр — `Ered[n0,y0]`; `n,y` не re-solve. Это буквально
  соответствует `_fd_energy_check` (строки 1311–1405), где
  `_gauss_for_profile` вызывается после additive `n,y` perturbation.
  Центральная разность `(E_+ + E_- - 2E_0)/e^2` сравнивается с
  `_hessian_quadratic` для того же `(u,v,v')`, той же нормировки и той же
  reference grid. Clipping, `max(n,0)` на пути, скрытая renormalization или
  подмена на stationary re-solve запрещены; при нарушении admissibility
  amplitude помечается недоступной.
* **C2 shape режим.** Сначала объявляются reference support и flow
  `Phi_e=x+e V`, включая знак нормального displacement. Density переносится
  как заданная mass-preserving форма на support:
  `n_e(Phi_e(x)) det(D Phi_e(x))=n0(x)`; scalar field задаётся тем же
  pullback-путём с отдельно указанным первым derivative, а tangential часть
  считается reparameterization. Для каждого `e` и `-e` вычисляется
  `Ered[q_e]` при Gauss re-solve только для `a_e`. Это не стационарный BVP
  solve и не minimization по `n,y`. Если используется не affine path, его
  second path derivative/acceleration должен быть либо (a) аналитически
  включён одновременно в Hadamard second derivative и FD comparison, либо
  (b) обнулён выбором пути в reference coordinates. Нельзя сравнивать
  Hessian по first derivative с FD пути, имеющим неучтённую acceleration:
  при ненулевом численном first variation она даёт дополнительный член.
  `N` и occupied support должны быть выполнены самим transport, без
  постфактум изменения `n`.

Hadamard expression и FD oracle therefore compare the same declared object:
в режиме affine — вторую производную `Ered` по заданному `(u,v)`; в shape
режиме — вторую производную `Ered[q_e]` по точно объявленному flow/transport
пути, с matching acceleration convention. Любое слово “fully constrained
re-solve” в старом draft заменяется на “constrained energy evaluation with
prescribed fields; Gauss-only re-solve”; полный Euler–Lagrange re-solve
является отдельным relaxed-energy experiment и не может служить FD oracle
для unrelaxed Hessian.

На одинаковых pulled-back mesh/domain сравниваются Hadamard form, existing
assembled finite-space Hessian (только с тем же first-order direction),
вышеописанный matching FD oracle и predeclared held-out edge deformations,
включая translation-like family без residual-based selection. Любое
расхождение локализуется по термам, а не превращается в PASS; agreement
означает лишь consistency заявленного finite variational object.

### 4.4. Независимые guards и adverse cases

Обязательны следующие контролируемые случаи:

1. **Functional/edge boundary:** отсутствующий functional, edge на внешней
   границе, немонотонный или неразрешённый `nu-M y`, несогласованные normal и
   flux conventions — stop/unresolved, без ручного продолжения.
2. **Constraint boundary:** fixed-`N` residual выше объявленного предела,
   неположительный Gauss operator/solve, missing full response или более
   одного удалённого quotient — failure, не положительный результат.
3. **Source disagreement:** disagreement между nullspace/KKT, Hadamard и
   finite-difference route остаётся unresolved; одно disagreement не
   объявляется negative mode.
4. **Decoupled-negative synthetic control:** отдельная известная отрицательная
   координата, не входящая в residual probe, должна сохранять отрицательный
   inertia/energy sign при нулевом probe residual. Это демонстрирует предел
   residual coverage и не является физическим результатом W8.
5. **Known algebra controls:** сохранить finite coordinate-response controls:
   `G=diag(1,1e-12), H=diag(1,-1e-12)` даёт `[-1,+1]` после equilibration;
   non-orthogonal congruence сохраняет generalized spectrum; manufactured
   negative translation остаётся negative в raw pair и удаляется только
   explicit projector.
6. **Physical guard boundary:** rank loss, tolerance-sensitive sign, failed
   translation orthogonality, failed number integral, failed residual,
   nonpositive norm или missing edge integration переводят sector в
   `UNRESOLVED_FINITE_BASIS_SIGN_OR_CONVERGENCE`; ослабление cutoff не
   допускается.

### 4.5. Mesh, quadrature и outer-domain ladder

После functional gate запускаются только заранее объявленные finite checks:
interface-aligned и unfitted mesh, `800→1600→3200` intervals, `24→32 fm`
outer domain, declared edge regularity and quadrature variants. Отчёт хранит
bulk/edge/flux/constraint error separately, а также `Q/W0` units, physical
norm, number drift, Gauss positivity, operator pivots, rank and translation
orthogonality. Тренд сетки — numerical diagnostic; это не continuum error
bound и не hidden-spectrum gap.

### 4.6. Наблюдаемые критерии завершения и фальсификации

**Статусы завершения должны быть раздельными.**

* `C2_GATE_STOPPED_FUNCTIONAL_NOT_ESTABLISHED`: gate закончен, но единый
  functional, differentiable support или consistent discretization не
  установлен. Это полноценный диагностический исход дизайна; Hadamard,
  shape-FD, sign и downstream acceptance не выполняются и не считаются
  пропущенными PASS-условиями. Он не означает, что проверка успешна или что
  Hessian подтверждён.
* `C2_GATE_PASSED_DOWNSTREAM_COMPLETE`: functional gate прошёл; affine и/или
  shape object, transport/acceleration convention, Gauss-only re-solves,
  term ledger, held-out deformations, guards и raw receipts завершены. Только
  тогда downstream result может отдельно классифицировать finite-space
  signs как positive, robust negative или unresolved.
* `C2_GATE_PASSED_DOWNSTREAM_UNRESOLVED`: gate прошёл, но хотя бы один
  downstream guard, matching-path check, rank, near-zero sign, mesh/domain
  comparison или receipt не дал требуемого результата. Это не успешная
  проверка и не negative witness; соответствующий сектор остаётся unresolved.

Для успешного downstream completion нужны одновременно: (i) gate и единый
manifest; (ii) точное различение prescribed-field energy evaluation и
Gauss-only re-solve; (iii) matching first/second path derivatives; (iv)
term-level ledger and guards; (v) сохранение всех исходных
negative/unresolved findings и ровно одного translation quotient. Один лишь
завершённый gate никогда не запускает ретроспективно downstream outputs.

**Фальсификация/отказ:** functional mismatch; необоснованный jump или
independent membrane coordinate; fixed-`N`/Gauss/edge/quotient failure;
несогласованный sign/normal convention; near-zero interval; rank loss;
доменная/сеточная чувствительность без безопасного запаса; или decoupled
negative control, ошибочно объявленный несущественным из-за нулевого
residual. Каждый такой случай останавливает соответствующий вывод и
оставляет его `UNRESOLVED`, а не переписывает его в положительный.

## 5. Что не выбирается сейчас

C3 и H5 остаются полезными будущими ветвями: C3 потребует независимой
reconstruction `H,M,C,t` и KKT/nullspace implementations; H5 потребует
verified arithmetic backend и enclosure proof. Их нельзя выдавать за
доказательство и нельзя добавлять к C2 без отдельного assignment. Будущая
связь с ядерными или иными observables — самостоятельное model-comparison
исследование, не экспериментальная проверка этой численной процедуры.

## 6. Ownership, модель и stop conditions

Этот документ — только design deliverable. Golden substantive author был
назначен на exact `gpt-5.6-luna/high`, fresh native context, без nested agents;
родитель принимает результат, но worker не принимает его за весь проект.
Будущая реализация требует отдельной авторизации, не меняет этот report и
не получает права редактировать contracts, registry, old reports или active
residual attempt. При любом unknown effect, overlap, change of functional,
new physical degree, unsafe shared writer или невозможности завершить raw
receipt соответствующий barrier остаётся OPEN и отправляется
`DECISION_REQUIRED`/`BLOCKED` с указанием точного владельца и bounded
reconciliation.
