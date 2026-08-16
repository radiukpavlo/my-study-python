# Додаток B. Resources

**Статус:** Draft
**Дата:** 22 квітня 2026

Цей додаток — curated довідник усіх інструментів, сервісів, стандартів і матеріалів, на які плейбук посилається у розділах 1–7. Використовується у двох режимах: (а) швидка орієнтація, куди ще глянути при налаштуванні чи траблшутингу; (б) source-of-truth при заповненні `AGENTS.md` / `CLAUDE.md` на новому проєкті — щоб не видумувати назви і посилання.

Базова теза: **довідник — нормативний**. Якщо щось не з цього списку — воно або не валідоване на проєктах, або належить до advanced-режиму (Level 2/3) і вимагає окремого узгодження. Нові пункти сюди додаються тільки після того, як їх реально використано на 2+ проєктах без regressions.

## B.1. Як читати цей довідник

Кожен запис має фіксовані поля:

- **Назва та посилання** — офіційне джерело або package registry.
- **Що це** — 1 речення: призначення.
- **Коли використовувати** — конкретний use case.
- **Статус** — `canonical` (baseline, рекомендовано за замовчуванням) / `supported` (перевірено, можна вибрати) / `optional` (під вузький кейс) / `advanced` (Level 2/3, не для baseline).
- **Де у плейбуку** — посилання на розділ, де інструмент згаданий у контексті.

> 💡 **Hint:** Не треба ставити всі інструменти зі списку. Обираєш один primary tool + 1–2 допоміжних + MCP-сервери під конкретний проєкт. Повний стек — це оверкіл, який з'їдає токени і час на підтримку.

## B.2. AI-інструменти та IDE

### B.2.1. Primary tools

Інструменти, на яких реально ведеться production-розробка. Вибирається **один** primary tool на проєкт; інші — допоміжні.

| Інструмент | Посилання | Що це | Статус | Розділ |
| --- | --- | --- | --- | --- |
| Cursor | `cursor.com` | AI-native IDE на базі VS Code; вбудовані chat, Composer, agent mode, Plan mode, `.cursor/rules` | canonical | §2.1, §3.1 |
| Claude Code | `docs.anthropic.com/claude-code` | Terminal-native AI-agent від Anthropic; IDE-agnostic; skills, commands, sub-agents, hooks | canonical | §2.1, §3.1 |
| GitHub Copilot | `github.com/features/copilot` | AI-plugin у VS Code, JetBrains, Xcode, Android Studio; сильний inline autocomplete, Agent mode | supported | §2.1, §3.1 |
| JetBrains AI Assistant | `jetbrains.com/ai` | Нативний AI в IntelliJ / WebStorm / PyCharm / RubyMine; multi-model через JetBrains subscription | supported | §2.1, §3.1 |

**Правило вибору:** decision tree у §2.1. Коротко — VS Code → Cursor, JetBrains → JetBrains AI + Claude Code CLI, Xcode / Android Studio / Visual Studio → Copilot, terminal-centric workflow → Claude Code.

### B.2.2. Допоміжні інструменти

Закривають вузькі ніші поверх primary tool.

| Інструмент | Посилання | Use case | Статус | Розділ |
| --- | --- | --- | --- | --- |
| ChatGPT (web / desktop) | `chatgpt.com` | Cross-check іншою моделлю, brainstorm, швидкі питання поза клієнтським кодом | supported | §2.1, §3.7 |
| Gemini (web / Pro) | `gemini.google.com` | Аналіз великих кодбаз (1M context), альтернативна думка | supported | §2.1 |
| OpenAI Codex (через Codex CLI / ChatGPT) | `openai.com/codex` | Незалежний verifier плану, PR-review поверх Claude-генерації | optional | §2.1 |
| Ollama | `ollama.com` | Локальні моделі (Qwen Coder, DeepSeek, Llama) для sensitive / NDA-коду | optional | §2.1, §3.7 |
| LM Studio | `lmstudio.ai` | GUI-альтернатива Ollama; локальна робота з PDF / закритими матеріалами | optional | §2.1 |
| Continue.dev | `continue.dev` | OSS-плагін для IDE з підтримкою локальних моделей; працює з Ollama-endpoint | optional | §3.7 |
| Cline | `cline.bot` | OSS-agent для VS Code; альтернатива для local-only workflow | optional | §3.7 |
| Spokenly | `spokenly.com` | Голосовий ввід промптів (macOS) через Whisper API | optional | §2.1, §3.7 |

> 💡 **Hint:** Не роби Ollama чи LM Studio primary tool-ом "щоб економити на токенах". Якість локальних моделей нижча, швидкість залежить від GPU; економія не компенсує ітерації. Local — лише коли NDA / data residency забороняє cloud.

### B.2.3. Інструменти поза scope

Не використовуються як primary (зауважені, щоб не шукати вручну):

- **Amazon Q / Kiro** — прив'язка до AWS-акаунту, обмежений за межами AWS-стеку; допускається, якщо клієнт вимагає.
- **Codeium / Windsurf** — сильний competitor; у опитаних не набрав critical mass; не відхиляється, але потребує пілоту перед вибором на проєкті.
- **Replit Agent** — для web-sandbox прототипування; не для client production.

## B.3. Моделі

Model routing — у §2.2 і §3.4. Список моделей, на які плейбук реально спирається.

### B.3.1. Хмарні reasoning

| Модель | Постачальник | Клас | Типове використання | Розділ |
| --- | --- | --- | --- | --- |
| Claude Opus 4.x | Anthropic | Reasoning | Architecture design, multi-file refactor, складний debug, cross-verification | §2.2, §5.2 |
| Claude Sonnet 4.6 | Anthropic | Balanced | Default на 50–60% часу: implementation, tests, docs | §2.2, §3.4 |
| GPT-5.x / GPT Codex | OpenAI | Balanced / Execution | Verification, boilerplate, automation, PR-review поверх Claude | §2.2, §5.4 |
| GPT-4o | OpenAI | Balanced | Fallback, коли Sonnet недоступний у конкретному tool | §2.2 |
| Gemini Pro (2.x) | Google | Large context | Аналіз монорепо >500k LOC, context window до 1M токенів | §2.2 |
| Gemini Flash | Google | Execution | Швидкий boilerplate, великі обсяги рутинних операцій | §2.2 |

### B.3.2. Локальні моделі

| Модель | Через що запускати | Use case |
| --- | --- | --- |
| Qwen 2.5 Coder (14B+) | Ollama, LM Studio | Sensitive code generation |
| DeepSeek Coder V2 | Ollama | Складніший reasoning локально |
| Llama 3.x | Ollama | General-purpose fallback |

**Правило:** default — Sonnet. Opus — вручну і тимчасово. Local — коли NDA. Перемикання моделей всередині однієї сесії без причини — зливання токенів.

> 💡 **Hint:** Фіксуй модель на початку сесії під тип задачі і не перемикай до кінця. Якщо треба незалежний verifier — це **окрема** сесія, а не зміна моделі у поточному chat.

## B.4. MCP-сервери

Model Context Protocol — відкритий стандарт для підключення зовнішніх систем до AI-агента. Специфікація: `modelcontextprotocol.io`.

Базовий стартовий набір і принципи підключення — §3.5. Нижче — реєстр перевірених серверів. Канонічна вимога: на клієнтському проєкті підключати тільки `official` або `approved` сервери (підтверджені клієнтом або security officer).

### B.4.1. Ticketing та issue tracking

| Сервер | Package / source | Що робить | Статус |
| --- | --- | --- | --- |
| Atlassian / Jira MCP | `@atlassian/mcp-server` (official) | Читати тікети, створювати, оновлювати статус | canonical |
| Linear MCP | `@linear/mcp-server` | Те саме для Linear | supported |
| GitHub MCP | `@github/mcp-server` або native integration у Copilot/Cursor | PR / issues / code search | canonical |

### B.4.2. Design

| Сервер | Package / source | Що робить | Статус |
| --- | --- | --- | --- |
| Figma MCP (local / dev-mode) | `figma.com/developers/mcp` | Витягує структуру дизайну та text-контент | supported (with caveats) |

> 💡 **Hint:** Figma MCP — **не** pixel-perfect generator. Він дає структуру, token-и і текст. Візуальне QA — ручне. Див. §7.3 (anti-pattern "Figma MCP як чорна коробка").

### B.4.3. Framework-specific

Сучасні фреймворки випускають офіційні MCP-сервери з актуальним LLM-контекстом (best practices, API signatures). Якщо проєкт на відповідній версії — підключай offіcial MCP до того, як писати rules вручну.

| Сервер | Фреймворк | Package | Статус |
| --- | --- | --- | --- |
| Angular MCP | Angular 18+ | `@angular/mcp` | canonical for Angular |
| Blazor / Telerik MCP | .NET / Blazor з Telerik | від Telerik | supported |
| MUI MCP | Material UI | `@mui/mcp` | supported |
| Context7 | Generic documentation | `context7.com` | canonical for legacy-stack support |

### B.4.4. Інфраструктура

| Сервер | Package | Use case | Статус |
| --- | --- | --- | --- |
| Azure MCP | `@azure/mcp` | Хмарна інфраструктура, AKS, storage | supported |
| AWS MCP / `awslabs/mcp` | AWS CLI wrappers | AWS operations | supported |
| Filesystem MCP | `@modelcontextprotocol/server-filesystem` | Локальна файлова система поза cwd агента | optional |

### B.4.5. Обмеження MCP

- Кожен додатковий MCP з'їдає токени **кожну сесію**. Вмикай тільки ті, що потрібні для типового workflow проєкту.
- Community MCP на client-проєктах — тільки після аудиту (джерело, maintenance, permissions).
- MCP з write-доступом до production-систем (prod DB, prod k8s) — **заборонені** за замовчуванням; див. §7.6.

## B.5. Стандарти та протоколи

### B.5.1. Entry-point файли

| Стандарт | Опис | Де використовується | Розділ |
| --- | --- | --- | --- |
| `AGENTS.md` | Універсальний entry-point, читається Cursor, Copilot (через fallback), Codex та іншими | Root репо; моно-рівень на monorepo | §3.3, §4.2 |
| `CLAUDE.md` | Нативний entry-point для Claude Code | Root репо; symlink на `AGENTS.md` або дзеркальна копія | §3.3, §4.2 |
| `.github/copilot-instructions.md` | Single project-instructions файл для GitHub Copilot | Root репо; fallback, якщо tool не читає `AGENTS.md` | §3.3 |
| `.cursor/rules/*.mdc` | Rules для Cursor (always-on або glob-scoped) | Repo-level | §4.3 |

### B.5.2. Model Context Protocol

- **Spec:** `modelcontextprotocol.io/specification`.
- **SDK:** `github.com/modelcontextprotocol` (TypeScript, Python).
- **Registry перевірених серверів:** див. §B.4; джерело — офіційні каталоги Anthropic та постачальників.

### B.5.3. Git / delivery

| Стандарт | Посилання | Коли потрібен | Розділ |
| --- | --- | --- | --- |
| Conventional Commits | `conventionalcommits.org` | PR-message generation, автоматичні changelog-и | §5.5, §6.6 |
| Keep a Changelog | `keepachangelog.com` | Формат `CHANGELOG.md`, на який агент дописує зміни | §7.4 |
| Architecture Decision Records (ADR) | `adr.github.io` | Фіксація архітектурних рішень, на які посилаються rules і skills | §1, §7.6 |
| Semantic Versioning | `semver.org` | Release notes, package versioning | §5.5 |

### B.5.4. Security / compliance стандарти

| Стандарт | Для чого | Розділ |
| --- | --- | --- |
| OWASP Top 10 | Check-list під AI-generated security-sensitive код | §7.6 |
| OWASP LLM Top 10 | Специфічні ризики LLM-застосувань (prompt injection, insecure output handling) | §7.6 |
| ISO 27001 | Information security на клієнтських проєктах | §2.3 |
| SOC 2 | Audit trail AI-використання | §2.3 |
| HIPAA | Data residency для health-проєктів | §2.3 |
| NIST AI RMF | Reference framework для AI-risk management | §1 (довідково) |

## B.6. Secrets та credentials

| Інструмент | Призначення | Статус |
| --- | --- | --- |
| 1Password (з CLI `op`) | Зберігання API-ключів, injection у env через `op run` | canonical |
| Bitwarden (з CLI `bw`) | OSS-альтернатива 1Password | supported |
| macOS Keychain / Windows Credential Manager | OS-рівень, коли централізованого менеджера немає | baseline |
| GitHub Actions Secrets / GitLab CI Variables | CI/CD-ключі | canonical |
| HashiCorp Vault | Enterprise-рівень secrets | optional (якщо клієнт використовує) |

**Заборонено:** API-ключі у plain config, `.env`, комітаних `settings.json`; ключі у чаті з AI; ключі у prompt-ах, які логуються провайдером.

## B.7. Verification та QA

### B.7.1. Static checks

| Інструмент | Мова/стек | Коли підключати до AI-flow |
| --- | --- | --- |
| ESLint + type-check (`tsc --noEmit`) | TS/JS | `PostToolUse` hook після кожного edit (§3.6) |
| Biome | TS/JS | Швидка альтернатива ESLint + Prettier |
| Prettier | TS/JS/CSS | Форматування, `pre-commit` |
| Pylint / Ruff | Python | `PostToolUse` hook |
| golangci-lint | Go | `pre-commit` |
| Checkstyle / SpotBugs | Java | CI |

### B.7.2. Tests

| Інструмент | Use case | Розділ |
| --- | --- | --- |
| Vitest | TS/JS unit-тести; reference у §4, skill `write-test` | §3.4, §4.3 |
| Jest | TS/JS legacy / React Native | §3.4 |
| Playwright | E2E / UI snapshots | §7.5 |
| Cypress | E2E | supported |
| Storybook | Component gallery + inteгрована візуальна регресія | optional |

### B.7.3. Visual regression

| Інструмент | Призначення | Коли потрібен |
| --- | --- | --- |
| Percy | Snapshot-based visual regression | UI-heavy фронтенд-проєкти |
| Chromatic | Storybook-based visual regression | Проєкти зі Storybook |

**Правило:** коли в пайплайні є AI-generated UI — візуальний regression-tool **обов'язковий**. AI не ловить pixel-drift; `tsc --noEmit` проходить на хибному UI.

### B.7.4. Independent reviewer models

- **GitHub Copilot PR review** (native) — base layer, не замінює людського review.
- **Codex CLI / Codex review agent** — незалежний bug-hunt поверх Claude-генерації.
- **Gemini Pro на large diff-и** — коли PR не вміщається у контекст Claude.

## B.8. Методології та патерни

### B.8.1. Level 1 canonical

| Концепт | Опис | Де у плейбуку |
| --- | --- | --- |
| Plan → Apply → Review | Базовий цикл L1 | §5 |
| Human-in-the-Loop (HITL) | Ревʼю кожного diff | §1, §2.3 |
| Context engineering hierarchy | Entry-point → rules → skills → commands → subagents | §4 |
| Model routing (Smart / Balanced / Execute) | Per-task вибір моделі | §2.2 |

### B.8.2. Level 2/3 advanced (згадано, не canonical для baseline)

| Концепт | Короткий опис | Коли розглядати |
| --- | --- | --- |
| Spec-Driven Development (SDD) | Spec/ADR генерується та ревізується **до** коду; код — деривація | Після стабільного L1 |
| Ralph Loop | Loop автоматичного self-correction агента через diff-аналіз | L3, з обережністю до безкінечних циклів |
| Master orchestrator + sub-agents | Оркестратор делегує задачі спеціалізованим агентам | L3, не для baseline |
| FORDEC | Decision-model з авіації (Facts, Options, Risks, Decision, Execution, Check), адаптований під структуру AI-задач | L2/L3 |
| Dual-model verification | План у reasoning-моделі → verification у незалежній моделі | Critical decisions, не на рутині |
| Cross-project vector context (напр. Milvus + Ollama) | Векторна база як shared memory для 10+ репозиторіїв | Enterprise multi-repo setups |

> 💡 **Hint:** Level 2/3 patterns — це не "наступний крок наступного тижня". Перехід відбувається тільки після того, як команда проходить L1 стабільно (Plan → Apply → Review на кожній задачі 2–3 місяці без деградацій якості).

### B.8.3. SDD-фреймворки (довідково)

Якщо проєкт переходить на L2 — розглянути один із:

- **GitHub Spec-Kit** (`github.com/github/spec-kit`) — OSS, agnostic до tool.
- **BMAD-METHOD** (`github.com/bmad-code-org/BMAD-METHOD`) — multi-agent spec workflow.
- **Kiro** (Amazon) — closed, прив'язка до AWS.

Вибір SDD-фреймворку — окреме рішення, не частина baseline. Документується на рівні проєкту в `AGENTS.md`.

## B.9. Довідкова документація

### B.9.1. Офіційні docs tool-ів

| Tool | URL |
| --- | --- |
| Anthropic (Claude, Claude Code) | `docs.anthropic.com` |
| OpenAI API | `platform.openai.com/docs` |
| Cursor | `docs.cursor.com` |
| GitHub Copilot | `docs.github.com/copilot` |
| JetBrains AI | `jetbrains.com/help/ai-assistant` |
| Google Gemini | `ai.google.dev/gemini-api/docs` |
| Ollama | `ollama.com/docs` |

### B.9.2. Стандарти та специфікації

| Документ | URL |
| --- | --- |
| Model Context Protocol | `modelcontextprotocol.io` |
| Conventional Commits | `conventionalcommits.org` |
| Keep a Changelog | `keepachangelog.com` |
| Semantic Versioning | `semver.org` |
| ADR templates | `github.com/joelparkerhenderson/architecture-decision-record` |
| OWASP LLM Top 10 | `owasp.org/www-project-top-10-for-large-language-model-applications` |
| OWASP Top 10 | `owasp.org/Top10` |

### B.9.3. Рекомендоване читання

Мінімальний reading list для інженера на Level 1. Додаткові матеріали читаються за потреби.

| Матеріал | Що дає | Для кого |
| --- | --- | --- |
| Anthropic — **Claude Code Best Practices** (`anthropic.com/engineering/claude-code-best-practices`) | Канонічний набір практик від постачальника tool | Усі, хто на Claude Code |
| Anthropic — **Prompt engineering guide** (`docs.anthropic.com/claude/docs/prompt-engineering`) | База для §6 (Prompt Library) | Baseline |
| Anthropic — **Building effective agents** | Поняття agent vs workflow, Sub-agent patterns | Level 2/3 |
| GitHub — **Spec-Kit docs** | Вступ у spec-driven development | Level 2 |
| Cursor — **Rules, Skills, Hooks docs** (`docs.cursor.com`) | Налаштування context engineering на Cursor | Усі, хто на Cursor |
| GitHub Copilot — **Customizing Copilot** | `copilot-instructions.md`, Agent mode | Усі, хто на Copilot |
| OWASP — **LLM Top 10** | Security baseline для AI-integrated коду | Senior, security reviewers |
| Simon Willison — **weblog та AI-нотатки** (`simonwillison.net`) | Тренди, реальні кейси, новини моделей | Level 2/3, технічний tracking |

> 💡 **Hint:** Читати все зі списку за раз — зайва витрата часу. Мінімум: офіційні docs твого primary tool + §6 плейбуку. Решта — за потреби, коли конкретна задача упреться в обмеження baseline.

## B.10. Внутрішні артефакти

Документи, які живуть поза цим плейбуком, але на які він явно спирається.

| Артефакт | Де шукати | Призначення |
| --- | --- | --- |
| AI Engineering Maturity Levels | Внутрішня knowledge base | Референсна модель L1/L2/L3, що цитується у §1 |
| AGENTS.md / CLAUDE.md workspace-level для playbook repo | `/AGENTS.md` у repo плейбуку | Editorial rules при генерації нових розділів |
| `engineering-playbook-author` skill | `.cursor/skills/engineering-playbook-author/SKILL.md` | Структурні гейти для нових розділів |
| Додаток A. Glossary | `docs/A. Glossary (UA).md` | Єдина термінологія плейбуку |
| Додаток C. Stack-Specific Notes | `docs/C. Stack-Specific Notes (UA).md` | Варіації AI-ефективності за стеками |
| Додаток D. Metric Baseline | `docs/D. Metric Baseline (UA).md` | Самозвітні дані по продуктивності |

## B.11. Як підтримується цей довідник

Нормативні правила для майбутніх edit-ів:

- Додавати новий tool / стандарт — **тільки** після того, як він використаний на 2+ клієнтських проєктах без regressions, або явно прийнятий як baseline на рівні компанії.
- Статус `canonical` присвоюється після внутрішнього ревʼю (playbook maintainer + senior engineer профільного стеку).
- Deprecated-інструменти не видаляються одразу — переводяться у статус `deprecated: <причина>` і зберігаються 2 quarter-и, щоб проєкти встигли мігрувати.
- Посилання валідуються раз у quarter; broken-links позначаються `(⚠ broken — замінити)` і правляться у наступному перегляді.

> 💡 **Hint:** Цей список не повинен рости нескінченно. Якщо на новому tool-і працює 1–2 людини разові — це ще не `supported`. Ознака готовності до включення: setup задокументований, reproducible за 1 годину на чистій машині, був ревʼюнутий не-автором.
