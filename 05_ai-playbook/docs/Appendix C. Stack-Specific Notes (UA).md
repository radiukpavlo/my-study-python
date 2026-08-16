# Додаток C. Stack-Specific Notes

**Статус:** Draft
**Дата:** 22 квітня 2026

AI працює у різних стеках з різною ефективністю. Той самий `/plan → /apply → /review` дає x3–x5 приріст на Node.js + React і рівно нульовий — на мобільному проєкті, де 70% часу йде на візуальне QA. Різницю не покриває загальна методологія плейбуку: вона покривається **стек-специфічними налаштуваннями** — моделлю, skills, rules, MCP-серверами, структурою entry-point-ів і очікуваннями щодо того, які задачі віддаються AI, а які лишаються руками.

Базова теза цього додатку: **AI-ефективність — функція трьох змінних.** Мова (наскільки вона присутня у training data), фреймворк (наскільки його конвенції стабільні і чітко документовані) і тип проєкту (greenfield легко, legacy важко, UI-intensive ще важче). Не існує "універсально сильного" AI-setup-у; існує setup, правильно налаштований під конкретну комбінацію цих трьох змінних.

Цей додаток — довідкова мапа. Його не треба читати послідовно; знайди свій стек, прочитай відповідний розділ, і повертайся до основного плейбуку (розділи 2–6) уже з контекстом, чого саме очікувати від AI на твоєму коді.

## C.1. Що читач має винести з цього розділу

Після прочитання інженер повинен:

- розуміти, чому той самий AI-інструмент дає різну продуктивність на різних стеках — і які саме дві-три змінні визначають цю різницю;
- знати типові сильні і слабкі сторони AI на своєму основному стеку, не очікувати x5 там, де об'єктивно буде x1.5;
- знати рекомендовану модель, набір skills, rules і MCP-серверів для свого стеку — як стартову конфігурацію, не догму;
- упізнавати типові пастки свого стеку до того, як витратити години на їх повторне відкриття;
- розуміти, як тип проєкту (greenfield / maintenance / legacy / presale) змінює AI-ефективність незалежно від стеку.

## C.2. Від чого залежить AI-ефективність

Перш ніж занурюватися у стек-специфіку, корисно зафіксувати три змінні, що пояснюють більшість розбіжностей у продуктивності.

**Покриття у training data.** Python, JavaScript/TypeScript, Java, Go, C# — мови з величезним обсягом публічного коду; моделі тримають їх на рівні senior-інженера. Dart/Flutter, Blazor, специфічні framework-и (Symfony вище 6, NestJS 10+, Angular 17+ signals, React Server Components) — значно рідше; відповіді нижчої якості, потребують більше context-engineering-у. Clojure, Elixir, Haskell, Rust-у rare corners — модель або галюцинує, або знижує якість на 30–50%.

**Стабільність конвенцій фреймворку.** Spring Boot, Laravel, Angular — опіковані "the right way" у документації; модель цю документацію читала і дотримується. React без prescriptive framework-у поверх (Next.js / Remix / TanStack Start) — варіативність висока, модель змішує патерни з п'яти різних кодбейзів.

**Тип проєкту.** Greenfield / presale / PoC — AI дає x5–x10 (нічого старого, нічого ламати). Зрілі монолітні проєкти з 3–5 років історії — x2–x3 за умови якісного context-engineering-у. Legacy без документації, з custom framework-ом поверх стандартного — x1.2–x1.5, переважно на локальних задачах, а не на фічах. UI-intensive проєкти (pixel-perfect mobile, Figma-to-code) — AI дає швидкий прототип, але 60–80% часу йде на візуальне QA, яке AI не робить.

Таблиця нижче — стартова мапа: високий/середній/низький рівень AI-leverage на типових задачах стеку. **Не використовуй її як контракт.** Реальна ефективність залежить від якості entry-point-ів, skills, рівня дисципліни у Plan → Apply → Review.

| Стек                         | Бекенд-логіка | UI / компоненти | Тести | Рефакторинг | Інфра / скрипти | Документація |
| ---------------------------- | ------------- | --------------- | ----- | ----------- | --------------- | ------------ |
| Node.js / NestJS / TypeScript | Високий       | —               | Високий | Високий   | Середній        | Високий      |
| Java / Spring Boot           | Високий       | —               | Високий | Високий   | Середній        | Високий      |
| Go                           | Високий       | —               | Високий | Середній  | Високий         | Високий      |
| PHP / Laravel / Symfony      | Високий       | —               | Середній | Середній | Низький         | Середній     |
| .NET / C# / Blazor           | Високий       | Середній        | Високий | Середній  | Середній        | Високий      |
| Python (скрипти / DevOps)    | Високий       | —               | Високий | Середній  | Високий         | Високий      |
| React / Next.js              | Високий       | Середній        | Високий | Високий   | Середній        | Високий      |
| Angular                      | Середній      | Середній        | Середній | Середній | —               | Середній     |
| Vue                          | Високий       | Середній        | Середній | Середній  | —               | Середній     |
| Flutter / Dart               | Середній      | Низький         | Середній | Низький   | —               | Середній     |
| Terraform / IaC              | —             | —               | Низький | Середній   | Високий         | Високий      |

## C.3. Backend

Бекенд — найсильніша зона AI-leverage. Типові задачі (CRUD, repository patterns, middleware, валідація, міграції, queue consumers) добре представлені у training data; конвенції фреймворків прескриптивні; тести легко автоматизувати. Приріст на щоденних задачах — x2–x3 стабільно, x5 на greenfield.

### C.3.1. Node.js / NestJS / TypeScript

**Де AI сильний.** Controller → service → repository layering; DTO + class-validator схеми; guards і interceptors; TypeORM/Prisma міграції; Jest/Vitest unit-тести з моками; OpenAPI/Swagger-анотації. Рефакторинг між NestJS-версіями (9 → 10 → 11) модель виконує точно, якщо у промпті зафіксована цільова версія.

**Де AI слабкий.** Exotic RxJS operator chains; performance-tuning Node.js event loop; діагностика memory leak-ів у production; custom decorator metadata (reflect-metadata edge cases); складні generic-и з conditional types.

**Модель.** Sonnet-tier для 80% задач; Opus — для планування архітектури модуля, вибору між repository vs. active-record, складних generic-refactor-ів.

**Skills і rules.**

- Skill `/write-module` — scaffold NestJS-модуля за проєктним template-ом (module + controller + service + DTO + spec).
- Skill `/write-query` — генерація TypeORM/Prisma query за описом домену.
- Rule always-on: `Prefer existing DTO/entity patterns. Before introducing a new validation decorator, grep existing @IsX() usages.`

**MCP і tools.** GitHub MCP (PR + gh CLI), Jira MCP (тікети), Prisma MCP для DB-інспекції на проєктах з Prisma.

**Типові пастки.**

- Модель генерує `any` у місцях, де можна conditional type; додай до rules `Do not use any — use unknown with type guards or request stronger typing`.
- Plain Jest-тести, що мокають все через `jest.fn()` без assertions на поведінку → `test behavior, not call sequences` у `/write-test` skill-і.
- NestJS DI: модель тягне імпорти через `forwardRef` при першому ж circular hint замість реального редизайну модулів.

### C.3.2. Java / Spring Boot

**Де AI сильний.** Controller → service → repository (JPA); Spring Security налаштування; `@ConfigurationProperties`; Spring Data специфікації і `@Query` методи; `@RestController` з `@Valid` DTO; Testcontainers-based integration tests; Liquibase/Flyway міграції.

**Де AI слабкий.** Kafka Streams або Reactor-специфічні сценарії; Spring Cloud Gateway конфіг з custom filters; performance-tuning JVM GC; legacy Spring XML-configs; глибокий AspectJ.

**Модель.** Sonnet-tier для щоденної роботи; Opus — для архітектури module/service boundaries, вибору transactional-стратегій, складного DI з `@Primary` / `@Qualifier`.

**Skills і rules.**

- Skill `/write-endpoint` — controller + service + DTO + validation за описом API.
- Skill `/write-integration-test` — Testcontainers-based test з описаними fixtures.
- Rule: `Prefer constructor injection. Never use field injection (@Autowired on fields).`
- Rule: `Use DTO for all request/response; never expose JPA entities in controllers.`

**MCP і tools.** GitHub MCP, Jira MCP. Для багато-модульних Maven/Gradle проєктів корисно мапити module-dependency у `AGENTS.md`, щоб модель не тягла імпорти через module boundaries.

**Типові пастки.**

- Модель змішує JPA lazy-fetch з DTO-трансформацією → LazyInitializationException в runtime. Фікс: у skill-і явно `fetch data inside the transactional boundary, then map to DTO`.
- `@Transactional` на приватних методах — не працює, модель іноді це пише. Додай rule: `@Transactional must be on public methods; transaction propagation via self-injection is a last resort.`
- Generic `ResponseEntity<?>` замість типізованого response — небажано у публічному API. Rule: `Return ResponseEntity<DTO>, not ResponseEntity<?>.`

### C.3.3. Go

**Де AI сильний.** HTTP handlers (chi, gin, stdlib `net/http`); middleware-chaining; `context.Context` propagation; table-driven tests; `errgroup` і базова concurrency; SQL-репозиторії через sqlc/squirrel; gRPC services з `.proto`-файлу.

**Де AI слабкий.** Складні `sync`-примітиви (WaitGroup+channel комбінації з backpressure); performance-sensitive goroutine pooling; low-level unsafe/reflect-трюки; custom generics з type constraints поверх Go 1.21+.

**Модель.** Sonnet-tier на 90% задач; Opus — для конкурентних задач, де помилка дорога (race conditions, deadlocks).

**Skills і rules.**

- Skill `/write-handler` — HTTP handler + validation + error handling за проєктним template-ом.
- Skill `/write-table-test` — table-driven test з error-cases.
- Rule: `Wrap errors with fmt.Errorf("%w", err) and context. Never use errors.New in packages where wrapping is used elsewhere.`
- Rule: `No naked returns in functions longer than 5 lines.`

**MCP і tools.** GitHub MCP, Jira MCP. `golangci-lint` у pre-commit обов'язково — AI-код часто проходить `gofmt`, але валиться на `errcheck` / `staticcheck`.

**Типові пастки.**

- `if err != nil { return err }` без контексту → втрата діагностики. Rule фіксує обов'язковий wrap.
- Модель пропонує interface на кожному repository "про всяк випадок" — Go-конвенція "accept interfaces, return structs" забувається. Rule: `Interfaces are defined where they are consumed, not where implementations live.`
- Goroutine без `context` + cancel → leak. У skill-і `/write-handler` явно: every goroutine must have a cancellation path.

### C.3.4. PHP / Laravel / Symfony

**Де AI сильний.** Eloquent/Doctrine моделі і відносини; Laravel controllers + form requests; Symfony services + DI configuration; міграції; Artisan/Console commands; PHPUnit unit-тести.

**Де AI слабкий.** Symfony bundle-архітектура з кастомною extension-логікою; Doctrine DQL-performance і складні JOIN-стратегії; legacy PHP 7.x код без type-declarations; Blade/Twig темплейти з великою кількістю component-ів.

**Модель.** Sonnet-tier; Opus — для міграції між мажорними версіями Symfony (5 → 6, 6 → 7) або Laravel (9 → 10, 10 → 11), де breaking changes потребують аналізу.

**Skills і rules.**

- Skill `/write-doctrine-entity` — entity з відносинами, `#[ORM]`-атрибутами, конвенціями ID-шок (ULID як binary → `toBase32()` на серіалізації тощо).
- Skill `/write-service` — Symfony service з interface + `#[AsAlias]`.
- Rule: `Always use Doctrine query parameters; never sprintf SQL.`
- Rule: `Use DateTimeImmutable over DateTime. Use DatePoint for current/relative time.`

**MCP і tools.** GitHub MCP. Корисно додати у `AGENTS.md` посилання на проєктні `make-migration.sh` / artisan-команди, щоб модель не генерувала міграції руками.

**Типові пастки.**

- Модель додає `save()` / `remove()` у repository, коли проєкт використовує EntityManager напряму. Rule: `Repositories must not have save/remove methods — use EntityManager directly.`
- N+1 у Eloquent/Doctrine: модель тягне відносини у циклі. У skill-і тестів перевіряй query-кількість; у code review роби окремий прохід на N+1.
- Blade/Twig-компоненти копіюють одну і ту саму partial замість `x-`-компонента. Rule: `Before creating a new partial, grep existing components/ for similar name or structure.`

### C.3.5. .NET / C# / Blazor

**Де AI сильний.** ASP.NET Core controllers і minimal APIs; Entity Framework Core моделі і міграції; DI конфігурація; xUnit тести з моками; MediatR + CQRS-handlers; SignalR hub-и.

**Де AI слабкий.** Blazor Server стану між circuits; складні JS-interop сценарії; Source Generators і Roslyn Analyzers; performance-tuning EF Core (`AsNoTracking`, `Include` optimization, compiled queries); legacy .NET Framework-код; WPF/WinForms.

**Модель.** Sonnet-tier; Opus — для Blazor архітектурних рішень (Server vs WebAssembly, state management), міграцій між версіями .NET, EF Core performance-задач.

**Skills і rules.**

- Skill `/write-endpoint` — minimal API endpoint + validator + handler.
- Skill `/write-ef-migration` — генерація EF-міграції з перевіркою на backward-compatible schema changes.
- Rule: `Use records for DTOs. Use primary constructors for services with immutable dependencies.`
- Rule: `Prefer async/await end-to-end; never .Result or .Wait() in production code.`

**MCP і tools.** GitHub MCP, Jira MCP. Якщо проєкт використовує Azure — Azure MCP для infra-задач. Telerik/Blazor MCP — якщо використовується UI-kit, щоб модель точно підтягувала component API.

**Типові пастки.**

- AutoMapper + version-migration болісно: модель припускає старий API. Фіксуй версію у `AGENTS.md` + skill з прикладами profile-ів.
- EF tracking leak-и: модель пропонує `AsNoTracking()` лише коли запитано, а за замовчуванням все tracked. Rule: `Read queries must use AsNoTracking unless the result is immediately updated.`
- Blazor Server compoents з важкою state-логікою в UI-layer → circuit bloat. У skill-і `/write-blazor-component` — чітке розділення: UI-stateless, state у сервісі, підписка через event.

### C.3.6. Python (скрипти, DevOps, data tooling)

У межах цього плейбуку Python покриває переважно automation-скрипти, ETL, DevOps-утиліти, data-processing. Повноцінні бекенд-фреймворки (Django, FastAPI) розглядаються окремо в разі появи профільних проєктів.

**Де AI сильний.** argparse/click CLI; pandas/polars data-transform; requests/httpx API-інтеграції; pytest тести; typed-скрипти з dataclass/pydantic; boto3-автоматизація AWS.

**Де AI слабкий.** asyncio-performance у high-concurrency сценаріях; C-extensions; ML-специфічні задачі без окремого RAG-контексту поверх; dependency resolution конфлікти у великих monorepo.

**Модель.** Sonnet-tier; Opus — для складних data-pipelines з транзакційністю і idempotency.

**Skills і rules.**

- Skill `/write-cli` — click-based CLI з чіткою структурою command → option → handler.
- Skill `/write-pytest` — тести з fixtures і parametrize.
- Rule: `Scripts must be idempotent — re-running must not double-apply side effects.`
- Rule: `All external I/O (HTTP, S3, DB) through wrapper functions that can be mocked in tests.`

**Типові пастки.**

- Модель ігнорує type hints у швидких скриптах → складний рефакторинг через 3 місяці. Rule: `All new Python code must have type hints on function signatures.`
- `requests` без timeout → hang у проді. Rule: `All HTTP calls must set a timeout explicitly.`
- `os.system` / `subprocess.call` зі string-конкатенацією → shell injection. Rule: `Use subprocess.run with list args; never shell=True unless explicitly required and reviewed.`

## C.4. Frontend

Frontend — друга за розміром зона; водночас найбільше джерело болю. Логіка (state, hooks, API-інтеграції) покривається AI добре. Візуальна частина (pixel-perfect верстка, респонсивність, кастомні компоненти) — стабільно слабка. Очікуй x2–x3 на логіці і x1–x1.5 на UI, з істотним часом на ручне QA.

### C.4.1. React / Next.js

**Де AI сильний.** Presentational компоненти з TypeScript props; custom hooks; TanStack Query + fetcher-и; Zustand slices і selectors; Next.js API routes; Server Actions (за умови чіткого skill-а); React Hook Form + Zod-валідація; Jest/Vitest + React Testing Library тести.

**Де AI слабкий.** Складні `useEffect` cleanup-сценарії з race conditions; Suspense boundaries поверх third-party libs, які не підтримують Suspense; React Server Components + client component boundary у Next 14/15 (модель плутає серверні і клієнтські імпорти); custom animation з Framer Motion + gesture-handling; складні `forwardRef` + generic-patterns.

**Модель.** Sonnet-tier; Opus — для state-architecture рішень, складних RSC-boundary рефакторингів.

**Skills і rules.**

- Skill `/write-component` — presentational component за проєктним file-naming (`*.component.tsx`) і структурою (директорія + sub-components).
- Skill `/write-query` — TanStack Query hook + adapter + queryKey factory.
- Skill `/write-store` — Zustand slice з setters і persistence patterns.
- Skill `/write-router` — routing патерн (React Router v6/v7, Next app router).
- Rule: `Use existing MUI/AntD/Radix primitives before writing custom HTML + CSS.`
- Rule: `Never use any; use unknown with narrowing. Prefer generics over any.`

**MCP і tools.** GitHub MCP, Jira MCP, Figma MCP (для витягання дизайн-структури, не для авто-верстки — див. §C.4.4).

**Типові пастки.**

- Модель додає `useEffect` для того, що мало б бути `useMemo` або derived state. Rule: `Effects are for side effects, not for derived state.`
- Import-порядок не відповідає ESLint `simple-import-sort` / конфіг проєкту → кожен PR має diff тільки з reorder-ом. Rule посилається на `.eslintrc` як ground truth; запускай `/lint:fix` перед commit.
- Next.js: `"use client"` directive додається випадково, коли не треба, або не додається, коли треба. Skill `/write-component` має явний decision-tree: if uses hooks/state/browser API → client; otherwise server.

### C.4.2. Angular

**Де AI сильний.** Standalone components + signals (`input()`, `output()`, `model()`); `@if` / `@for` нові control flow syntax; reactive forms; RxJS basic operators (map, filter, switchMap); lazy-loaded routes; NgRx slices.

**Де AI слабкий.** Міграції між мажорними версіями (14 → 17 → 19) — великий обсяг file-level breaking changes; складні RxJS pipeline-и з backpressure; `ChangeDetectionStrategy.OnPush` + non-signal inputs edge cases; темплейти з глибоко-вкладеними structural directives у legacy; `ng-template` + `TemplateRef` dynamic rendering.

**Модель.** Sonnet-tier; Opus — для міграцій між версіями і для складних RxJS-рефакторингів.

**Skills і rules.**

- Skill `/write-component` — standalone component з `OnPush`, signal inputs, template з новим syntax-ом.
- Skill `/write-signal-store` — state management через signals або NgRx signal-store.
- Rule: `OnPush everywhere. Signal-based APIs (input/output/model) for new components.`
- Rule: `Use @if/@for, not *ngIf/*ngFor, in new templates.`
- Rule: `No hardcoded colours in SCSS — use CSS variables from theme tokens.`

**MCP і tools.** Angular MCP — нативна CLI-утиліта, що дає AI доступ до schematics і актуальних conventions; підключення одноразове і помітно покращує якість. GitHub MCP, Jira MCP.

**Типові пастки.**

- Великі version-migrations (14 → 20): один промпт на все → модель губиться після 10 файлів. Розбивай міграцію на фази: 1) control flow (`*ngIf` → `@if`), 2) signals inputs, 3) OnPush, 4) standalone. Кожна — окрема гілка + PR.
- Зайві `async` pipes там, де signal прямий і простіший. Після міграції на signals — skill з декількома before/after прикладами.
- RxJS `subscribe()` у компоненті без `takeUntilDestroyed()` → memory leak. Rule: `Use takeUntilDestroyed() in component subscriptions; prefer async pipe or toSignal where possible.`

### C.4.3. Vue

**Де AI сильний.** Composition API `<script setup>`; Pinia stores; Vue Router routes; `defineProps` / `defineEmits` з TypeScript-типами; VueUse composables; Vitest тести.

**Де AI слабкий.** Options API legacy-код (модель мимоволі модернізує частину, лишаючи inconsistency); складні `provide` / `inject` з generics; custom directives з життєвим циклом, що залежить від DOM-mount порядку; Vue 2 → Vue 3 міграція у нетривіальних проєктах.

**Модель.** Sonnet-tier; Opus — для міграції Options API → Composition API на проєктах більше 50 компонентів.

**Skills і rules.**

- Skill `/write-component` — Composition API, `<script setup>`, typed props, Pinia-store-інтеграція.
- Skill `/write-store` — Pinia-store з getters/actions за проєктною конвенцією.
- Rule: `New components use Composition API with <script setup>. Do not mix Options and Composition in one component.`
- Rule: `Props and emits are typed via TypeScript interfaces, not runtime arrays.`

**МКП і tools.** GitHub MCP, Jira MCP.

**Типові пастки.**

- Reactive unwrapping: модель забуває `.value` у `ref`-ах у non-template коді. У skill-і — приклади.
- `watch` vs `watchEffect` вибирається довільно; explicit rule — `Use watch when you need to observe a specific source; use watchEffect only when dependencies are clearly tracked by closure.`
- Pinia-store розростається у god-object. Rule: `One store = one domain. Split when a store exceeds ~10 actions.`

### C.4.4. Крос-фронтендні проблеми: UI, pixel-perfect, Figma-to-code

Це одна з найчастіше названих "слабких зон AI" на проєктах, незалежно від фреймворку. Слабка не модель сама по собі — слабка інтеграція "візуальний output → людське QA → feedback у модель".

**Симптоми.**

- 15–30 ітерацій на один компонент: "посунь 4px вгору", "червоний неправильний", "hover-стан не працює".
- PR проходить, але QA на staging-і знаходить 8–10 візуальних багів.
- Figma MCP дає структуру, але компоненти не вписуються у проєктну design-system.
- Респонсивність ламається на breakpoint-ах, що не були показані у промпті.

**Стартовий setup.**

- **Design-system skill.** Перше, що налаштовується на фронтенд-проєкті: skill з tokens (spacing, color palette, typography scale), списком доступних компонентів UI-kit (MUI/AntD/shadcn/Radix/Vuetify/Angular Material), правилами variant-ів і темінгу.
- **Figma MCP — як джерело даних.** Витягати структуру, text-content, id-шники компонентів. Не очікувати "pixel-perfect auto-generation". Після MCP запускається `/write-component` skill з design-system-контекстом.
- **Screenshot-based verification.** Skill, що робить скріншот згенерованого компонента (Puppeteer/Playwright) і кладе поруч із референсом із Figma; AI сам порівнює і запитує, що виправити. Зменшує ручне QA у 2–3 рази.
- **Visual regression у CI.** Percy / Chromatic / Playwright screenshots — обов'язково на будь-якому фронтенд-проєкті, де AI генерує UI. Без цього регресії не ловляться у review.

**Модель.** Sonnet-tier на implementation; Opus — коли треба перепланувати компонентну бібліотеку або проаналізувати узгодженість Figma-системи з кодом.

> 💡 **Hint:** Правило трьох спроб для UI — якщо після трьох ітерацій AI не попадає у дизайн, зупини. Або дизайн недоописаний (треба уточнити у дизайнера), або tokens у skill-і не відповідають реальності, або задача не підходить під AI-loop (складна анімація, gesture, 3D). Не тіскай "ще один промпт" — відкрий Inspector у Figma, пиши CSS руками.

## C.5. Mobile

Мобільна розробка залишається найскладнішою зоною для AI. Причини: менше публічного коду у training data, кожна платформа має власні конвенції, візуальний output вимагає девайса (симулятор / реальний телефон) для верифікації.

### C.5.1. Flutter / Dart

**Де AI сильний.** StatelessWidget / StatefulWidget структура; basic state management (Provider, Riverpod v2+); routing (go_router); http-клієнти; repository-pattern; unit-тести з mocktail.

**Де AI слабкий.** Custom `CustomPaint` і складна анімація з `AnimationController`; platform channels (iOS Swift ↔ Dart, Android Kotlin ↔ Dart); складні Gesture-сценарії; performance profiling (rebuild-оптимізації); адаптація під нестандартні device-розміри / tablet-лейаути; integration з native-libs, які не мають Flutter-wrapper-а.

**Модель.** Sonnet-tier для щоденних задач; Opus — для state-architecture, platform-channel-дизайну, складних анімацій.

**Skills і rules.**

- Skill `/write-widget` — widget з Riverpod-providers, themed styling за проєктними tokens, тестами.
- Skill `/write-repository` — repository + DTO + мок для unit-test-а.
- Rule: `Use const constructors wherever possible for performance.`
- Rule: `No setState in widgets larger than 50 lines — use state management (Riverpod/Bloc).`

**MCP і tools.** Figma MCP (структура екрана, text, icons). GitHub MCP, Jira MCP. Dart/Flutter MCP-утиліти — у проєктах з ними якість помітно краща, особливо для налаштування `flutter pub` і generated code.

**Типові пастки.**

- `setState` у контейнерах з важкою логікою — rebuild усього дерева. Rule + skill з прикладами винесення state у provider.
- Модель змішує Provider v5 і Riverpod patterns у одному проєкті. У `AGENTS.md` — явна фіксація обраного state-management-а і версії.
- Pixel-perfect на mobile ще гірше, ніж на вебі, через DPI-розбіжності між девайсами. Візуальне QA на двох-трьох реальних девайсах — обов'язкова фаза, яку AI не замінює.
- Українська / російська локалізація у ICU-форматі: модель іноді генерує `plural` без правильних категорій (one/few/many). Skill з прикладами з проєкту.

### C.5.2. Native iOS / Android

У межах цього плейбуку native mobile покривається лише загальними принципами — проєктів мало, а training data на Swift/Kotlin дає прийнятну якість тільки на типових задачах.

**Де AI сильний.** Базові UIKit / SwiftUI views; Jetpack Compose базові компоненти; Retrofit / Alamofire API-клієнти; Room / CoreData моделі; юніт-тести.

**Де AI слабкий.** Combine / RxSwift складні pipeline-и; Kotlin Coroutines + Flow з backpressure; iOS-specific UX (navigation patterns, modal presentation); Android fragment-lifecycle edge cases; performance на складних RecyclerView / LazyColumn.

**Рекомендація.** На native-проєктах підсилюй AGENTS.md посиланнями на Apple / Google офіційні HIG / Material design guidelines. Моделі, що читали ці документи, дають помітно кращий результат на UX-рішеннях.

## C.6. DevOps / Infrastructure

DevOps — зона, де AI дає найшвидший приріст на рутині (boilerplate Terraform, Kubernetes-маніфести, bash-скрипти), але вимагає підвищеної обережності через вартість помилки (прод-інциденти, compliance).

### C.6.1. Terraform / IaC

**Де AI сильний.** Resource blocks для типових AWS/GCP/Azure сервісів; module structure; variables / outputs / locals; data sources; state migration-скрипти; CI-pipeline для `terraform plan/apply`.

**Де AI слабкий.** Складні conditional/dynamic blocks з вкладеними looks; cross-account / cross-region networking; drift-detection у великих state-файлах; Terragrunt-обгортки; custom providers.

**Модель.** Sonnet-tier; Opus — для architectural-рівня (module decomposition, state split, migrations).

**Skills і rules.**

- Skill `/write-module` — Terraform-модуль з README, variables.tf, outputs.tf, examples/.
- Skill `/plan-explain` — пояснює `terraform plan` output на plain language.
- Rule: `Never commit .tfstate or .tfvars with secrets. Use remote state backend (S3 + DynamoDB lock).`
- Rule: `Destructive changes (replace/delete on stateful resources) must be flagged in PR description.`

**MCP і tools.** AWS MCP / Azure MCP для інспекції актуального стану cloud-ресурсів. GitHub MCP для PR-автоматизації.

**Типові пастки.**

- Модель генерує `count` там, де треба `for_each` — лінивий рефакторинг ламає state-index-и. Rule: `Prefer for_each over count for stable addressing.`
- `provider "aws" {}` з hardcoded region у модулях → модулі стають непортованими. Rule: `Modules do not declare providers — caller passes via provider block.`
- AI пропонує `terraform apply --auto-approve` у CI → аварія. Fix: `apply` виключно після `plan` review; `--auto-approve` — лише для dev-середовищ з revert-планом.

### C.6.2. Kubernetes / Helm

**Де AI сильний.** Deployment / Service / Ingress / ConfigMap / Secret-маніфести; resource requests/limits; liveness/readiness probes; HPA / PDB; Helm chart structure з values.yaml.

**Де AI слабкий.** Network policies для складних multi-tenant setup-ів; operator pattern (CRD + controller); RBAC для fine-grained scenarios; StatefulSet-міграції; custom admission webhooks.

**Модель.** Sonnet-tier; Opus — для архітектури cluster (namespace design, multi-tenancy, network topology).

**Skills і rules.**

- Skill `/write-manifest` — маніфест за проєктним шаблоном (labels, resource limits, probes).
- Rule: `Every container must declare resource requests AND limits. No naked deployments.`
- Rule: `Secrets via external secret manager (ESO / Vault), never in plain YAML.`

**Типові пастки.**

- `imagePullPolicy: Always` з fixed tag — модель так пише, CI-cache бʼється. Fix: `Always` тільки з mutable tag (`:latest`, `:dev`); для релізних tag-ів — `IfNotPresent`.
- `replicas: 1` на stateless-сервісах, які мають бути HA. Skill `/write-manifest` має в decision-tree перевірку "чи є цей сервіс HA-critical".

### C.6.3. AWS / Cloud CLI

**Де AI сильний.** `aws` CLI команди; boto3-скрипти; CloudFormation прості стеки; IAM-policies за найменш-privileged-патерном; Lambda-функції з базовими тригерами.

**Де AI слабкий.** VPC-peering і Transit Gateway networking; cross-account IAM-trust-policies; CloudWatch Logs Insights складні query; Athena/Glue-пайплайни.

**Модель.** Sonnet-tier; Opus — для security-sensitive policy-дизайну, де помилка = security incident.

**Rules.**

- `IAM policies: explicit Allow with specific Resource ARN. Never use "Resource": "*" unless documented as required.`
- `All production actions through break-glass role with MFA; never from agent-connected machine.`

**Типові пастки.**

- AI у allow-list-і `aws` CLI з прод-credential-ами → потенційно виконана прод-команда через галюцинацію. Розділяй: dev-машина — dev/staging credentials; прод — окремий обмежений доступ через bastion (див. §7.6.3 у Anti-pattern Catalog).
- Secrets у `--parameters` до CloudFormation напряму. Rule: `Use SecureString parameter / Secrets Manager / SSM; never pass secrets as CLI arg.`

### C.6.4. Bash / shell scripting

**Де AI сильний.** Linear скрипти автоматизації (build, deploy, backup); `find` + `xargs` pipelines; basic signal handling; `trap` cleanup.

**Де AI слабкий.** POSIX-portable code (моделі генерують bash-isms, що ламаються у sh/dash); складне error handling у довгих pipelines.

**Rules.**

- `All scripts start with: set -euo pipefail; IFS=$'\n\t'.`
- `Use "${var}" quoting for all variable expansions involving paths or user input.`
- `shellcheck must pass in CI; no exceptions.`

## C.7. За типом проєкту

Стек — не єдиний фактор AI-ефективності. Тип проєкту і його життєвий цикл змінюють очікування, іноді сильніше, ніж сам стек.

### C.7.1. Greenfield / presale / PoC

**Характеристика.** Код з нуля, немає legacy, немає конвенцій, нічого не ламати.

**AI-ефективність.** x5–x10 на etalon-задачах. Presale-демо, що раніше займало 2 тижні, генерується за 2 дні разом з базовим UI.

**Setup.** Мінімальний `AGENTS.md` (50–100 рядків): стек, версії, style guide, команди. Default skills — 2–3 (write-module, write-test, write-component). Не зупиняйся на context-engineering-у довше 2–3 годин перед стартом.

**Пастки.**

- "Копіюю архітектуру зі старого проєкту": AI переносить техборг разом із конвенціями. Не копіюй `AGENTS.md` з зрілого проєкту — запиши наново.
- Over-engineering з перших промптів (5 рівнів абстракції на валідацію email-а). Constraint у промптах: `Prefer the simplest solution. Add abstraction only when duplication proves it.`

### C.7.2. Maintenance зрілих проєктів

**Характеристика.** 1–5 років історії, 50k–500k рядків, стабільний стек, є документація і конвенції.

**AI-ефективність.** x2–x3 стабільно; залежить від якості entry-point-ів і skills.

**Setup.** Розгорнутий `AGENTS.md` (100–150 рядків) + per-module nested `AGENTS.md`. 5–10 skills під основні повторювані задачі. MCP-сервери під project management (Jira/Linear) і VCS (GitHub/GitLab).

**Пастки.**

- "Заодно" рефакторинг у PR для фічі (див. §7.5.5). На maintenance — суворий scope control: окремий PR на кожну структурну зміну.
- AI не "бачить" існуючий utility-шар → duplication. Rule у кожному implementation skill-і: `Before creating a new util, grep lib/ and shared/ for existing implementations.`
- Stale docs у Confluence → AI генерує код за "документованим" API, якого вже немає (див. §7.6.7). Правило: code > docs як ground truth.

### C.7.3. Legacy

**Характеристика.** 5+ років; custom framework поверх стандартного; мало документації; тестів мало або вони flaky; коміти 2018 року.

**AI-ефективність.** x1.2–x1.5, переважно на локальних задачах (зрозуміти функцію, пояснити SQL-query, написати юніт-тест під існуючий метод).

**Setup.** Перший крок — **archaeology пасом**: `/explain` skill, який читає модуль і пише коротку пояснювальну записку у `AGENTS.md`. Поступово нарощуй entry-point знизу вгору. Skills — орієнтовані на малі rolling tasks: `/write-characterization-test`, `/explain-query`, `/rename-safely`.

**Пастки.**

- Спроба "переписати разом із AI" на маstере — катастрофа. Legacy рефакториться малими кроками з characterization-тестами; AI допомагає писати ці тести, а не переписує код одним промптом.
- AI впевнено пропонує "modern pattern", що ламає custom-framework-конвенції. Rule: `Follow the existing style of the file. Do not introduce modern idioms unless explicitly requested.`
- Власник коду пішов 3 роки тому → немає кому верифікувати AI-вивід. Будь-яка нетривіальна зміна потребує double-review і прогонки на staging.

### C.7.4. UI-intensive (дизайн-driven)

**Характеристика.** Mobile-app або складний web з pixel-perfect вимогами; Figma — основне джерело правди для UI; дизайнер активно донорить нові екрани.

**AI-ефективність.** x2–x3 на логіці, x1–x1.5 на UI (з design-system skill-ом і screenshot-verification — до x2).

**Setup.** Див. §C.4.4. Ключові артефакти: design-system skill, Figma MCP, screenshot-based verification, visual regression у CI.

**Пастки.**

- Pixel-perfect loop без design-system (див. §7.6.11 у Anti-pattern Catalog). Не починай implement, поки не є skill-а з tokens.
- Plan → Apply → Review без окремої Visual QA фази → регресії в проді. Додавай гейт `/visual-qa` перед PR merge.

### C.7.5. Data-intensive / ML-adjacent

**Характеристика.** ETL-pipelines, data warehouses, data-processing з великими обсягами; іноді — ML-inference-ендпоінти.

**AI-ефективність.** x2 на pipeline-коді (boilerplate); x1 на analysis/feature-engineering (де треба domain-знання).

**Setup.** Schema-catalog skill — мапа таблиць, колонок, зв'язків; AI читає її перед будь-якою SQL-задачею. Test-data-fixtures skill (з обов'язковим використанням synthetic data, не prod copies — див. §7.6.2).

**Пастки.**

- AI впевнено генерує SQL проти таблиці, якої не існує, або колонки, що перейменована. Rule: `Before writing SQL, verify schema via /db-schema skill or explicit grep in migrations/.`
- ML-завдання без RAG-контексту на domain (клінічні дані, фінансові інструменти) — галюцинації. Рекомендація: mounted RAG з domain-glossary.

## C.8. Чек-лист підготовки стеку під AI

Мінімальний набір, який покриває 80% типових проблем, незалежно від стеку. Якщо на 8+ пунктів відповідь "так" — setup у здоровому діапазоні.

- `AGENTS.md` / `CLAUDE.md` у корені проєкту; ≤ 150 рядків; містить стек, версії, основні команди, головні конвенції.
- Per-module nested entry-point для репозиторіїв > 50k рядків.
- 3–5 skills під основні повторювані задачі стеку (write-component / write-endpoint / write-test / write-query / ...).
- Принаймні один MCP-сервер, специфічний для стеку або framework-у (Angular MCP, Azure MCP, Figma MCP, Prisma MCP тощо).
- Явно зафіксовані версії ключових бібліотек у `AGENTS.md` (щоб модель не посилалася на застарілий API).
- Model routing: default → Sonnet-tier, складне → Opus-tier; зафіксовано у проєктному tool-config.
- Linter + formatter у pre-commit hook; не обходиться через `--no-verify`.
- Skill / rule, що вимагає `Before creating new utilities, grep existing lib/ and shared/` — проти duplication.
- Rule "admit uncertainty" у always-on (`If you are not certain, say so. Do not claim a fix without evidence.`).
- Для фронтенд-проєктів: design-system skill + visual regression у CI.
- Для бекенд-проєктів: тестовий harness (unit + integration), який AI використовує для верифікації після apply.
- Для DevOps / infra: заборона destructive commands у allow-list агента; прод-credentials не на машині з агентом.

> 💡 **Hint:** Якщо твій стек не потрапив у цей додаток або покритий коротко — не чекай, поки "хтось напише розділ під мене". Збери власний `stack-notes.md` у проєкті за цим самим шаблоном (де AI сильний / слабкий / рекомендована модель / skills / MCP / пастки) і запропонуй його як контрибуцію в наступну версію плейбуку. Локальне відкриття "у моєму стеку є специфіка" без фіксації означає, що наступний інженер на тому ж стеку повторить твій шлях з нуля.

На цьому L1-скоуп плейбуку вичерпано. Сам додаток C — довідковий, не обов'язковий до послідовного прочитання. Він переглядається, коли: починається проєкт на новому для інженера стеку; очікування продуктивності розходиться з реальністю; новий member команди потребує швидкого onboarding-у на проєктний стек.
