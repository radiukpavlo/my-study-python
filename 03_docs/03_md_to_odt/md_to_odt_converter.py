# md_to_odt_converter.py
"""
Markdown to ODT Converter

This script converts Markdown files (.md) into OpenDocument Text files (.odt)
compatible with LibreOffice and OpenOffice.

Features:
- Auto-installs dependencies (pypandoc).
- Auto-downloads the necessary Pandoc binary.
- Supports Table of Contents, footnotes, tables, and metadata.
- **NATIVE MATH SUPPORT**: Converts LaTeX equations ($...$ and $$...$$) 
  into native LibreOffice Math objects.

Usage:
  python md_to_odt_converter.py input.md output.odt
"""

import sys
import subprocess
import importlib.util
from pathlib import Path
import argparse

# --- Dependency Management ---
def install_and_import_pypandoc():
    """
    Checks if pypandoc is installed. If not, installs it via pip.
    Then checks if the Pandoc binary is present.
    """
    if importlib.util.find_spec("pypandoc") is None:
        print("📦 Installing required library: pypandoc...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pypandoc"])
        except subprocess.CalledProcessError:
            print("❌ Failed to install pypandoc. Please install it manually: pip install pypandoc")
            sys.exit(1)

    import pypandoc
    
    # Ensure the actual Pandoc binary is downloaded/installed
    try:
        # Check if pandoc is available on system path or via pypandoc wrapper
        pypandoc.get_pandoc_version()
    except OSError:
        print("⚙️  Pandoc binary not found. Downloading (this happens only once)...")
        try:
            pypandoc.download_pandoc()
            print("✅ Pandoc binary installed.")
        except Exception as e:
            print(f"❌ Failed to download Pandoc binary: {e}")
            sys.exit(1)
            
    return pypandoc

# Load dependency
pypandoc = install_and_import_pypandoc()

# --- Core Logic ---

def validate_input(input_path):
    """Validates the input Markdown file."""
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() not in ['.md', '.markdown']:
        raise ValueError(f"Input must be a Markdown file (.md), got: {path.suffix}")
    return path

def validate_output(output_path):
    """Validates the output ODT path."""
    path = Path(output_path).resolve()
    if path.suffix.lower() != '.odt':
        raise ValueError(f"Output must be an ODT file (.odt), got: {path.suffix}")
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def convert_md_to_odt(input_file, output_file, use_toc=True, reference_odt=None):
    """
    Converts Markdown to ODT.
    
    Args:
        input_file (str): Path to source .md file
        output_file (str): Path to destination .odt file
        use_toc (bool): Generate a Table of Contents
        reference_odt (str): Path to a reference .odt file for styling (optional)
    """
    in_path = validate_input(input_file)
    out_path = validate_output(output_file)

    print(f"📄 Reading: {in_path.name}")
    print(f"🚀 Converting to OpenDocument Text...")

    # --- Pandoc Arguments Configuration ---
    # --standalone: Produces a full document with header/footer
    extra_args = ['--standalone']
    
    if use_toc:
        extra_args.append('--toc')
    
    # Style Reference Logic
    if reference_odt:
        ref_path = Path(reference_odt).resolve()
        if ref_path.exists():
            extra_args.append(f'--reference-doc={str(ref_path)}')
            print(f"🎨 Using reference style: {ref_path.name}")
        else:
            print(f"⚠️ Warning: Reference file {ref_path.name} not found. Using default styles.")

    # --- Math & Equation Logic ---
    # We use 'markdown' format. In Pandoc, standard 'markdown' enables 'tex_math_dollars' 
    # by default. This ensures $E=mc^2$ and $$...$$ are parsed as LaTeX.
    # When converting to 'odt', Pandoc automatically translates these parsed LaTeX nodes 
    # into native OpenDocument formula objects (ODF) without requiring external filters.
    input_format = 'markdown'

    try:
        output = pypandoc.convert_file(
            str(in_path),
            'odt',
            format=input_format,
            outputfile=str(out_path),
            extra_args=extra_args,
            encoding='utf-8'
        )
        print(f"✅ Success! File saved to:\n   {out_path}")
        print("   (Equations have been converted to editable Native ODF Math objects)")
        
    except RuntimeError as e:
        print(f"❌ Conversion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        sys.exit(1)

# --- CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser(
        description="Convert Markdown to LibreOffice ODT format with Native Math support.",
        epilog="Supports LaTeX equations: $inline$ and $$block$$."
    )
    
    parser.add_argument('input_file', help='Source Markdown file (.md)')
    parser.add_argument('output_file', help='Destination ODT file (.odt)')
    parser.add_argument('--no-toc', action='store_true', help='Disable Table of Contents generation')
    parser.add_argument('--style', help='Path to an existing .odt file to use as a style template', default=None)

    args = parser.parse_args()

    convert_md_to_odt(
        args.input_file, 
        args.output_file, 
        use_toc=not args.no_toc,
        reference_odt=args.style
    )

if __name__ == "__main__":
    main()
