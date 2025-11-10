"""Orchestrates the full artifact rebuild pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
ASSET_DIRS = ("figs", "data", "tables")


def run_script(script_name: str) -> None:
    subprocess.check_call([sys.executable, str(SCRIPTS_DIR / script_name)])


def main() -> None:
    for folder in ASSET_DIRS:
        (ROOT / folder).mkdir(exist_ok=True)
    run_script("generate_data_and_figures.py")
    run_script("export_results_section.py")
    print("All artifacts generated.")


if __name__ == "__main__":
    main()
