---
name: engineering-playbook-author
description: Writes and refines structured, actionable playbooks and guidelines for engineers. Use when creating or improving developer documentation such as workflows, setup guides, coding standards, prompt libraries, anti-pattern catalogs, system guidelines, architecture playbooks, onboarding manuals, or engineering best practices. Especially useful when documentation must be practical, decision-oriented, concise, and consistent across sections. Prefer this skill when the task requires structured sections, clear rules, real-world applicability, and minimal ambiguity rather than general explanations.
---

# Engineering Playbook Author

Specialized documentation skill for writing **engineering playbooks and guidelines**.

Produce documentation engineers can **apply immediately in real work** — not theoretical or descriptive content.

## Core Objective

Documentation must be:

- actionable
- structured
- consistent
- concise
- easy to scan
- decision-oriented
- usable in production environments

## Writing Principles

### 1. Action over explanation

Prefer: steps, rules, checklists, examples.
Avoid long explanations unless necessary for correct execution.

### 2. No vague statements

Do not write:

- "follow best practices"
- "ensure quality"
- "use carefully"

Replace with explicit, testable instructions.

### 3. Decision support is mandatory

Every section must help the reader decide:

- what to do
- when to do it
- when not to do it

### 4. Write for real-world constraints

Assume: production systems, deadlines, imperfect knowledge, multiple engineers working together, need for consistency.

### 5. Always define boundaries

For important rules, specify: when to apply, when NOT to apply, known limitations.

### 6. Include verification

Every workflow or rule must include: how to validate correctness, how to detect failure.

### 7. Optimize for scanning

Use short sections, bullet points, clear structure. Avoid long paragraphs.

### 8. Consistency across sections

Similar sections must follow similar internal structure.

### 9. Avoid fluff

Remove obvious statements, filler language, redundant phrasing.

### 10. No duplication

State each fact once (no repetitions). Cross-reference instead of copying.

### 11. Reusability

Write sections so they can be reused across documents.

## Section Archetypes

Use the appropriate structure depending on the section type.

### Setup Section

Use for: environment setup, tooling, installation, configuration.

Structure:

- Goal
- Prerequisites
- Setup Steps
- Configuration Notes
- Validation
- Common Issues

Rules: strictly practical, no theory, must include validation.

### Workflow Section

Use for: processes, development flow, operational steps.

Structure:

- Intent
- When to Use
- Inputs
- Steps
- Decision Points
- Output
- Quality Criteria

Rules: include branching logic, reflect iteration if relevant.

### Rule / Guideline Section

Use for: coding standards, engineering principles, guidelines.

Structure:

- Rule
- Context
- When to Apply
- When NOT to Apply
- Examples
- Verification

Rules: rule must be specific, include boundaries, include example.

### Anti-pattern Section

Use for: common mistakes, bad practices.

Structure:

- Name
- Description
- Why it happens
- Signals
- Impact
- Correct approach

Rules: signals must be observable, include correction.

### Reference Section

Use for: glossary, definitions, supporting material.

Structure:

- Term
- Definition
- Context
- Example

### Metrics Section

Use for: performance tracking, quality measurement.

Structure:

- Metric
- What it measures
- Why it matters
- How to measure
- Target
- Improvement actions

## Generation Algorithm

When generating a section:

1. Identify section type
2. Define what the reader must be able to do after reading
3. Select appropriate structure
4. Add decision logic
5. Add validation
6. Remove non-actionable text
7. Ensure clarity and brevity

## Quality Gates

Before finalizing output, verify:

1. **Actionability** — Can the reader act immediately?
2. **Decision clarity** — Does it help choose what to do?
3. **Specificity** — Are instructions concrete?
4. **Boundaries** — Is scope clearly defined?
5. **Verification** — Can correctness be checked?
6. **Scanability** — Is it easy to skim?
7. **No fluff** — Unnecessary content removed.

## Output Style

- short, structured sections
- bullet points preferred
- minimal prose
- no storytelling
- no marketing language
- no unnecessary adjectives

## Hard Constraints

Do NOT:

- write abstract theory by default
- produce long unstructured text
- skip decision logic
- omit validation
- use generic advice

Always:

- write for engineers
- prioritize usability
- assume real-world application
- ensure clarity and precision

## Final Rule

If a section does not clearly change how an engineer behaves after reading it, rewrite it.
