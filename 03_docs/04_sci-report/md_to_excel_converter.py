"""
Markdown to Excel Converter

This utility reads a Markdown document, stores the full text in the first
worksheet, and extracts every Markdown table into its own worksheet inside an
.xlsx workbook. It defaults to converting `my_doc.md` to `my_doc.xlsx`.

Usage:
    python md_to_excel_converter.py [input_file] [output_file]

Examples:
    python md_to_excel_converter.py               # my_doc.md -> my_doc.xlsx
    python md_to_excel_converter.py notes.md      # notes.md -> notes.xlsx
    python md_to_excel_converter.py src.md out.xlsx
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


# --- Dependency management ----------------------------------------------------
def ensure_dependency(package_name: str, import_name: str | None = None) -> None:
    """Install a package via pip if it's missing."""
    import_name = import_name or package_name
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing dependency: {package_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", package_name]
        )


ensure_dependency("openpyxl")
from openpyxl import Workbook  # type: ignore  # installed at runtime if missing


# --- Markdown table parsing ---------------------------------------------------
TABLE_DIVIDER_RE = re.compile(
    r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$"
)


def looks_like_table_header(line: str) -> bool:
    """Basic guard to detect a potential header row."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    return stripped.count("|") >= 2


def looks_like_table_divider(line: str) -> bool:
    return bool(TABLE_DIVIDER_RE.match(line))


def split_markdown_row(row: str) -> List[str]:
    """Split a Markdown table row into cells."""
    working = row.strip()
    if working.startswith("|"):
        working = working[1:]
    if working.endswith("|"):
        working = working[:-1]
    return [cell.strip() for cell in working.split("|")]


def normalize_rows(header: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """Pad rows so every row has the same number of columns."""
    max_width = max(len(header), max((len(r) for r in rows), default=0))
    padded_header = header + [""] * (max_width - len(header))
    padded_rows = [r + [""] * (max_width - len(r)) for r in rows]
    return padded_header, padded_rows


def find_heading_above(lines: List[str], start_index: int) -> str:
    """Locate the nearest Markdown heading above a table."""
    for i in range(start_index, -1, -1):
        text = lines[i].strip()
        if text.startswith("#"):
            return text.lstrip("#").strip()
    return "Document"


def extract_tables(lines: List[str]) -> List[dict]:
    """Scan Markdown lines and return parsed tables with context."""
    tables = []
    i = 0
    while i < len(lines) - 1:
        header_line = lines[i].rstrip("\n")
        divider_line = lines[i + 1].rstrip("\n")
        if looks_like_table_header(header_line) and looks_like_table_divider(divider_line):
            block = [header_line, divider_line]
            j = i + 2
            while j < len(lines):
                candidate = lines[j].rstrip("\n")
                if candidate.strip() and "|" in candidate:
                    block.append(candidate)
                    j += 1
                else:
                    break

            header = split_markdown_row(block[0])
            data_rows = [split_markdown_row(line) for line in block[2:]]
            header, data_rows = normalize_rows(header, data_rows)

            tables.append(
                {
                    "header": header,
                    "rows": data_rows,
                    "start_line": i + 1,  # 1-based for readability
                    "heading": find_heading_above(lines, i),
                }
            )
            i = j
        else:
            i += 1
    return tables


# --- Workbook writing ---------------------------------------------------------
INVALID_SHEET_CHARS = re.compile(r"[\\/*?:\[\]]")


def sanitize_title(text: str) -> str:
    cleaned = INVALID_SHEET_CHARS.sub(" ", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "Table"


def make_sheet_title(base: str, index: int, used: set[str]) -> str:
    base_clean = sanitize_title(base)
    candidate = f"T{index}_{base_clean}"[:31]
    suffix = 1
    while candidate in used:
        suffix += 1
        candidate = f"T{index}_{base_clean}_{suffix}"[:31]
    return candidate


def write_document_text_sheet(wb: Workbook, lines: Iterable[str]) -> None:
    ws = wb.active
    ws.title = "DocumentText"
    ws.append(["Line", "Text"])
    for idx, line in enumerate(lines, start=1):
        ws.append([idx, line.rstrip("\n")])


def write_table_sheets(wb: Workbook, tables: List[dict]) -> None:
    used_titles = {wb.active.title}
    for idx, table in enumerate(tables, start=1):
        title = make_sheet_title(table["heading"], idx, used_titles)
        used_titles.add(title)
        ws = wb.create_sheet(title)
        ws.append(table["header"])
        for row in table["rows"]:
            ws.append(row)


# --- Conversion pipeline ------------------------------------------------------
def validate_input(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() != ".md":
        raise ValueError(f"Input must be a Markdown (.md) file, got {path.suffix}")
    return path


def validate_output(path_str: str, default_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.suffix:
        path = path.with_suffix(".xlsx")
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Output must be an Excel (.xlsx) file, got {path.suffix}")
    if not path.is_absolute():
        path = (default_dir / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def convert_md_to_excel(input_file: str, output_file: str | None) -> Path:
    in_path = validate_input(input_file)
    out_path = validate_output(
        output_file or in_path.with_suffix(".xlsx").name, default_dir=in_path.parent
    )

    lines = in_path.read_text(encoding="utf-8").splitlines()
    tables = extract_tables(lines)

    wb = Workbook()
    write_document_text_sheet(wb, lines)
    write_table_sheets(wb, tables)
    wb.save(out_path)

    print(f"Converted {in_path.name} -> {out_path.name}")
    print(f"  DocumentText sheet lines: {len(lines)}")
    print(f"  Tables extracted: {len(tables)}")
    return out_path


# --- CLI ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Markdown document to an Excel workbook with one sheet per table.",
        epilog="Defaults to my_doc.md -> my_doc.xlsx if no arguments are provided.",
    )
    parser.add_argument("input_file", nargs="?", default="my_doc.md", help="Source Markdown file")
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Destination Excel file (.xlsx). Defaults to the input filename with .xlsx",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        convert_md_to_excel(args.input_file, args.output_file)
    except Exception as exc:  # keep console-friendly errors
        print(f"Conversion failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
