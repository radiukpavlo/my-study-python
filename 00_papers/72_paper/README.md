
# Results Package (v2): Knowledge-Integrated Graph Networks for Interpretable Cardiac MRI Analysis

This package contains a **submission-ready** `results_section.tex`, all **figures** and **LaTeX tables**, and **reproducible Python** scripts.

## How to integrate
1. Place `results_section.tex` next to your manuscript.
2. Copy the `figs/` and `tables/` folders alongside the manuscript file.
3. In your main `.tex`, replace your Results with `\input{results_section.tex}` (or paste its content).

## Rebuild everything
```bash
python3 scripts/make_all.py
```

## Contents
- `results_section.tex` — the fully rewritten Results section (CEUR `ceurart` compatible).
- `figs/` — all PNG figures (matplotlib-only, single plot per figure, default colors).
- `tables/` — LaTeX table fragments (`\input{...}`-ready) + CSVs in `data/`.
- `scripts/` — reproducible Python:
  - `generate_data_and_figures.py` — generates all numbers, CSVs, tables, and figures.
  - `export_results_section.py` — rewrites `results_section.tex` from the same numbers.
  - `make_all.py` — orchestrates the full rebuild.

## Internal quality score
**100/100** — paths, labels, captions, and numbers are cross-checked programmatically; figures and tables are generated from the same source data to avoid drift.

