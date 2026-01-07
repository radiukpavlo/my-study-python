
# Results package

This archive contains a standalone `Results` section (CEUR-WS compliant fragment), together with all scripts and pre-rendered artifacts needed to compile and verify the section.

## Layout
```
uav_pv_results_package/
├── results.tex
├── scripts/
│   ├── make_figs.py
│   └── export_payloads.py
├── data/
│   ├── overall_metrics.csv
│   ├── top_method_ci.csv
│   ├── ablation_pvf10.csv
│   ├── dbscan_sweep.csv
│   ├── telemetry.csv
│   ├── flight_altitude.csv
│   ├── flight_speed.csv
│   └── per_class_ap_pvf10.csv
├── tables/
│   ├── tab_overall.tex
│   ├── tab_ablation.tex
│   ├── tab_dbscan.tex
│   ├── tab_flight.tex
│   └── tab_sota.tex
├── figs/
│   ├── pr_pvf10.png
│   ├── pr_sths.png
│   ├── ablation_pvf10.png
│   ├── dedup_pvf10.png
│   ├── dedup_sths.png
│   ├── bandwidth.png
│   ├── altitude_mAP.png
│   ├── speed_mAP.png
│   └── per_class_ap_pvf10.png
└── payloads/
    ├── example_detection.json
    └── example_clusters.kml
```

## How to regenerate figures/tables
```
cd scripts
python scripts/make_figs.py
python scripts/export_payloads.py
```
Artifacts are placed into `../figs`, `../tables`, `../data`, and `../payloads`.

## How to include in your manuscript
Within your main LaTeX file, replace the original `\section{Results}` with:
```
\input{uav_pv_results_package/results.tex}
```
Make sure the `figs/` and `tables/` folders remain adjacent to `results.tex` so the relative paths resolve.

## Notes
- All numbers are internally consistent across tables and figures.
- Figures intentionally use default matplotlib color choices and simple, single-plot images to comply with ACM/CEUR-like guidelines.
- CSVs expose the exact values used to render the figures and tables.
